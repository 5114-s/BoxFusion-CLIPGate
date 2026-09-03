#!/usr/bin/env python3
"""Read-only dev3 geometry ceiling for sealed OWLv2+Boxer per-view OBBs.

For each sampled frame this audit selects exactly the highest frozen source
scores under budgets K={2,4,6,8}.  Equal scores are resolved by ascending
sealed source-row index and then sealed NPZ row index.  All four selections
are completed and hashed before the first GT or axis-alignment path is opened.

The report is a post-hoc dev diagnostic before any MobileSAM stage.  It cannot
authorize H10, full100, active birth, or threshold tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(_REPOSITORY_ROOT))

from tools.audit_scannet_boxer_unexplained_oracle import (  # noqa: E402
    aligned_iou_matrix,
    load_axis_alignment,
    load_baseline_boxes,
    load_gt_minmax,
    strict_maximum_matching,
)


SCHEMA = "boxfusion.scannet_boxer_per_view_topk_ceiling.v1"
INPUT_SCHEMA = "boxfusion.owl_boxer_shadow_candidates.v1"
TOP_K_BUDGETS = (2, 4, 6, 8)
THRESHOLDS = (0.15, 0.25, 0.50)
TARGET_RECALL_POINTS = 10.0
SCENE_PATTERN = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")

_EXPECTED_ARRAYS = {
    "per_view_center_world",
    "per_view_extent_xyz",
    "per_view_frame_id",
    "per_view_quaternion_wxyz",
    "per_view_scene_index",
    "per_view_source_instance_id",
    "per_view_source_row",
    "per_view_source_score",
    "scene_ids",
    "tracked_center_world",
    "tracked_extent_xyz",
    "tracked_instance_id",
    "tracked_quaternion_wxyz",
    "tracked_scene_index",
    "tracked_source_row",
    "tracked_source_score",
}
_SIGNS = np.asarray(
    [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
    dtype=np.float64,
)


class BoxerTopKCeilingError(ValueError):
    """Raised when a sealed input or the read-only audit contract is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise BoxerTopKCeilingError(
            f"{label} must be a regular non-symlink file: {path}"
        )
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    path = _regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BoxerTopKCeilingError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise BoxerTopKCeilingError(f"{label} must contain an object: {path}")
    return value


def _array_content_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _selection_sha256(indices_by_scene: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for scene_index, values in enumerate(indices_by_scene):
        array = np.ascontiguousarray(values, dtype=np.int64)
        digest.update(np.asarray([scene_index, len(array)], dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _threshold_key(threshold: float) -> str:
    return f"{threshold:.2f}"


def _load_sealed_sidecar(
    json_path: Path, npz_path: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray], tuple[str, ...], dict[str, str]]:
    json_path = _regular_file(json_path, "sealed Boxer JSON")
    npz_path = _regular_file(npz_path, "sealed Boxer NPZ")
    before = {"json": _sha256(json_path), "npz": _sha256(npz_path)}
    manifest = _read_json(json_path, "sealed Boxer JSON")
    required = {
        "schema": INPUT_SCHEMA,
        "profile": "clean_in2",
        "mode": "shadow",
        "output_inert": True,
        "birth": False,
        "gt_access": False,
        "gt_access_guard_verified": True,
        "semantic_source_exported": False,
        "native_clip_unchanged": True,
        "native_before_after_identity": True,
        "coordinate_frame": "scannet_world",
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise BoxerTopKCeilingError(
                f"sealed Boxer contract mismatch for {key}: {manifest.get(key)!r}"
            )
    if manifest.get("npz_file") != npz_path.name:
        raise BoxerTopKCeilingError("sealed Boxer NPZ filename mismatch")
    if manifest.get("npz_sha256") != before["npz"]:
        raise BoxerTopKCeilingError("sealed Boxer NPZ SHA-256 mismatch")
    assets = manifest.get("assets_and_protocol")
    if not isinstance(assets, Mapping):
        raise BoxerTopKCeilingError("sealed asset/protocol ledger is missing")
    expected_assets = {
        "profile": "clean_in2",
        "detector": "owl",
        "threshold_2d": 0.25,
        "threshold_3d": 0.5,
        "nms_iou_2d": 0.5,
        "start_n": 1,
        "skip_n": 25,
    }
    for key, expected in expected_assets.items():
        if assets.get(key) != expected:
            raise BoxerTopKCeilingError(
                f"sealed asset/protocol mismatch for {key}: {assets.get(key)!r}"
            )
    try:
        with np.load(npz_path, allow_pickle=False) as source:
            if set(source.files) != _EXPECTED_ARRAYS:
                raise BoxerTopKCeilingError("unexpected sealed Boxer NPZ schema")
            arrays = {name: np.array(source[name], copy=True) for name in source.files}
    except (OSError, ValueError) as error:
        if isinstance(error, BoxerTopKCeilingError):
            raise
        raise BoxerTopKCeilingError(f"invalid sealed Boxer NPZ: {npz_path}") from error
    if _sha256(npz_path) != before["npz"]:
        raise BoxerTopKCeilingError("sealed Boxer NPZ changed while loading")
    if manifest.get("candidate_content_sha256") != _array_content_sha256(arrays):
        raise BoxerTopKCeilingError("sealed Boxer candidate content hash mismatch")

    scene_ids = arrays["scene_ids"]
    if scene_ids.ndim != 1 or scene_ids.dtype.kind != "U":
        raise BoxerTopKCeilingError("sealed scene_ids schema is invalid")
    scenes = tuple(str(value) for value in scene_ids.tolist())
    if (
        not scenes
        or len(set(scenes)) != len(scenes)
        or any(SCENE_PATTERN.fullmatch(scene) is None for scene in scenes)
        or manifest.get("scene_count") != len(scenes)
    ):
        raise BoxerTopKCeilingError("sealed scene order is invalid")
    count = len(arrays["per_view_scene_index"])
    if manifest.get("per_view_candidate_count") != count:
        raise BoxerTopKCeilingError("sealed per-view candidate count mismatch")
    shapes = {
        "per_view_center_world": (count, 3),
        "per_view_extent_xyz": (count, 3),
        "per_view_frame_id": (count,),
        "per_view_quaternion_wxyz": (count, 4),
        "per_view_scene_index": (count,),
        "per_view_source_instance_id": (count,),
        "per_view_source_row": (count,),
        "per_view_source_score": (count,),
    }
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            raise BoxerTopKCeilingError(f"sealed {name} shape mismatch")
    for name in (
        "per_view_frame_id",
        "per_view_scene_index",
        "per_view_source_instance_id",
        "per_view_source_row",
    ):
        if arrays[name].dtype.kind not in "iu":
            raise BoxerTopKCeilingError(f"sealed {name} is not integer")
    finite = np.concatenate(
        [
            arrays["per_view_center_world"].reshape(-1),
            arrays["per_view_extent_xyz"].reshape(-1),
            arrays["per_view_quaternion_wxyz"].reshape(-1),
            arrays["per_view_source_score"].reshape(-1),
        ]
    )
    if not np.isfinite(finite).all():
        raise BoxerTopKCeilingError("sealed per-view geometry/score is non-finite")
    if np.any(arrays["per_view_extent_xyz"] <= 0.0):
        raise BoxerTopKCeilingError("sealed per-view extent is non-positive")
    if np.any(
        (arrays["per_view_source_score"] < 0.0)
        | (arrays["per_view_source_score"] >= 1.0)
    ):
        raise BoxerTopKCeilingError("sealed source score is outside [0,1)")
    if np.any(arrays["per_view_frame_id"] < 0):
        raise BoxerTopKCeilingError("sealed per-view frame ID is negative")
    scene_indices = arrays["per_view_scene_index"].astype(np.int64, copy=False)
    if np.any((scene_indices < 0) | (scene_indices >= len(scenes))):
        raise BoxerTopKCeilingError("sealed per-view scene index is out of range")
    norms = np.linalg.norm(arrays["per_view_quaternion_wxyz"], axis=1)
    if np.any(norms <= 1e-6):
        raise BoxerTopKCeilingError("sealed per-view quaternion is degenerate")

    scene_ledger = manifest.get("scenes")
    if not isinstance(scene_ledger, list) or len(scene_ledger) != len(scenes):
        raise BoxerTopKCeilingError("sealed scene ledger is invalid")
    for scene_index, (scene, ledger) in enumerate(zip(scenes, scene_ledger)):
        if (
            not isinstance(ledger, Mapping)
            or ledger.get("scene_id") != scene
            or ledger.get("scene_index") != scene_index
            or ledger.get("gt_access_guard_verified") is not True
            or ledger.get("per_view_extra_schedule_rows_excluded") != 0
        ):
            raise BoxerTopKCeilingError(f"sealed scene contract mismatch for {scene}")
        positions = np.flatnonzero(scene_indices == scene_index)
        if ledger.get("per_view_kept_rows") != len(positions):
            raise BoxerTopKCeilingError(f"sealed per-view scene count mismatch for {scene}")
        source_rows = arrays["per_view_source_row"][positions]
        if len(np.unique(source_rows)) != len(source_rows):
            raise BoxerTopKCeilingError(f"duplicate sealed source row for {scene}")
    return manifest, arrays, scenes, before


def _select_per_frame_topk(
    arrays: Mapping[str, np.ndarray], scenes: Sequence[str]
) -> dict[int, list[np.ndarray]]:
    """Freeze selections by (-score, source_row, sealed_npz_row)."""

    scene_index_array = arrays["per_view_scene_index"]
    frame_ids = arrays["per_view_frame_id"]
    scores = arrays["per_view_source_score"]
    source_rows = arrays["per_view_source_row"]
    selections: dict[int, list[np.ndarray]] = {
        budget: [] for budget in TOP_K_BUDGETS
    }
    for scene_index, _scene in enumerate(scenes):
        scene_positions = np.flatnonzero(scene_index_array == scene_index)
        selected: dict[int, list[int]] = {budget: [] for budget in TOP_K_BUDGETS}
        for frame_id in sorted(np.unique(frame_ids[scene_positions]).tolist()):
            positions = scene_positions[frame_ids[scene_positions] == frame_id]
            order = sorted(
                positions.tolist(),
                key=lambda row: (
                    -float(scores[row]),
                    int(source_rows[row]),
                    int(row),
                ),
            )
            for budget in TOP_K_BUDGETS:
                selected[budget].extend(order[:budget])
        for budget in TOP_K_BUDGETS:
            selections[budget].append(np.asarray(selected[budget], dtype=np.int64))
    return selections


def _quaternion_rotation(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    norm_squared = float(q @ q)
    if q.shape != (4,) or not math.isfinite(norm_squared) or norm_squared <= 1e-12:
        raise BoxerTopKCeilingError("invalid per-view quaternion")
    w, x, y, z = q
    scale = 2.0 / norm_squared
    return np.asarray(
        [
            [
                1.0 - scale * (y * y + z * z),
                scale * (x * y - z * w),
                scale * (x * z + y * w),
            ],
            [
                scale * (x * y + z * w),
                1.0 - scale * (x * x + z * z),
                scale * (y * z - x * w),
            ],
            [
                scale * (x * z - y * w),
                scale * (y * z + x * w),
                1.0 - scale * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _aligned_candidate_minmax(
    arrays: Mapping[str, np.ndarray], positions: np.ndarray, alignment: np.ndarray
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for position in positions:
        rotation = _quaternion_rotation(arrays["per_view_quaternion_wxyz"][position])
        local = _SIGNS * (arrays["per_view_extent_xyz"][position] / 2.0)
        corners = (
            local @ rotation.T + arrays["per_view_center_world"][position]
        )
        aligned = corners @ alignment[:3, :3].T + alignment[:3, 3]
        rows.append(np.concatenate((aligned.min(axis=0), aligned.max(axis=0))))
    return np.asarray(rows, dtype=np.float64).reshape((-1, 6))


def audit_scannet_boxer_per_view_topk_ceiling(
    *,
    shadow_json: Path,
    shadow_npz: Path,
    baseline_root: Path,
    gt_root: Path,
    scan_root: Path,
) -> dict[str, Any]:
    shadow_json = shadow_json.resolve()
    shadow_npz = shadow_npz.resolve()
    baseline_root = baseline_root.resolve()
    gt_root = gt_root.resolve()
    scan_root = scan_root.resolve()
    if not baseline_root.is_dir():
        raise BoxerTopKCeilingError("baseline root must be a directory")

    manifest, arrays, scenes, sidecar_before = _load_sealed_sidecar(
        shadow_json, shadow_npz
    )

    # This is the only deployable selection stage.  It completes before any
    # GT/axis file is resolved, opened, or hashed.
    selections = _select_per_frame_topk(arrays, scenes)
    selection_hashes = {
        str(budget): _selection_sha256(selections[budget])
        for budget in TOP_K_BUDGETS
    }
    for left, right in zip(TOP_K_BUDGETS, TOP_K_BUDGETS[1:]):
        for scene_index in range(len(scenes)):
            if not set(selections[left][scene_index]).issubset(
                set(selections[right][scene_index])
            ):
                raise BoxerTopKCeilingError("Top-K selections are not nested")

    baseline_paths: dict[str, Path] = {}
    baseline_before: dict[str, str] = {}
    for scene in scenes:
        path = _regular_file(
            baseline_root / f"{scene}_boxes.pkl", f"frozen T05 prediction for {scene}"
        )
        baseline_paths[scene] = path
        baseline_before[scene] = _sha256(path)

    gt_counts: list[int] = []
    baseline_iou: list[np.ndarray] = []
    candidate_iou: dict[int, list[np.ndarray]] = {
        budget: [] for budget in TOP_K_BUDGETS
    }
    input_before: dict[str, Any] = {
        "shadow_json": sidecar_before["json"],
        "shadow_npz": sidecar_before["npz"],
        "scenes": {},
    }
    for scene_index, scene in enumerate(scenes):
        gt_path = _regular_file(gt_root / f"{scene}_bbox.npy", f"ScanNet GT for {scene}")
        axis_path = _regular_file(
            scan_root / scene / f"{scene}.txt", f"axis alignment for {scene}"
        )
        alignment = load_axis_alignment(axis_path)
        gt = load_gt_minmax(gt_path)
        _, native_aligned = load_baseline_boxes(baseline_paths[scene], alignment)
        gt_counts.append(len(gt))
        baseline_iou.append(aligned_iou_matrix(native_aligned, gt))
        for budget in TOP_K_BUDGETS:
            candidate_aligned = _aligned_candidate_minmax(
                arrays, selections[budget][scene_index], alignment
            )
            candidate_iou[budget].append(aligned_iou_matrix(candidate_aligned, gt))
        input_before["scenes"][scene] = {
            "baseline": baseline_before[scene],
            "gt": _sha256(gt_path),
            "axis_alignment": _sha256(axis_path),
        }

    total_gt = int(sum(gt_counts))
    required_matches = int(math.ceil(TARGET_RECALL_POINTS * total_gt / 100.0))
    budget_reports: dict[str, Any] = {}
    for budget in TOP_K_BUDGETS:
        scene_counts = {
            scene: int(len(selections[budget][scene_index]))
            for scene_index, scene in enumerate(scenes)
        }
        threshold_reports: dict[str, Any] = {}
        supports: list[bool] = []
        for threshold in THRESHOLDS:
            candidate_total = native_total = union_total = 0
            per_scene: dict[str, Any] = {}
            for scene, candidate, native in zip(
                scenes, candidate_iou[budget], baseline_iou
            ):
                candidate_pairs = strict_maximum_matching(candidate, threshold)
                native_pairs = strict_maximum_matching(native, threshold)
                union_pairs = strict_maximum_matching(
                    np.concatenate((native, candidate), axis=0), threshold
                )
                candidate_total += len(candidate_pairs)
                native_total += len(native_pairs)
                union_total += len(union_pairs)
                per_scene[scene] = {
                    "candidate_count": int(len(candidate)),
                    "candidate_maximum_matching_count": len(candidate_pairs),
                    "native_maximum_matching_count": len(native_pairs),
                    "native_union_maximum_matching_count": len(union_pairs),
                    "additional_union_matching_over_native": len(union_pairs)
                    - len(native_pairs),
                }
            additional = union_total - native_total
            headroom_points = 100.0 * additional / total_gt if total_gt else 0.0
            supported = additional >= required_matches
            supports.append(supported)
            threshold_reports[_threshold_key(threshold)] = {
                "iou_threshold": threshold,
                "strict_iou_comparison": ">",
                "candidate_maximum_matching_count": candidate_total,
                "native_maximum_matching_count": native_total,
                "native_union_maximum_matching_count": union_total,
                "additional_union_matching_over_native": additional,
                "incremental_recall_headroom_points": headroom_points,
                "plus_10_required_additional_matches": required_matches,
                "supports_plus_10_recall_headroom": supported,
                "per_scene": per_scene,
            }
        budget_reports[str(budget)] = {
            "top_k_per_frame": budget,
            "selection_rule": "descending_source_score_then_ascending_source_row",
            "selection_sha256": selection_hashes[str(budget)],
            "candidate_count": int(sum(scene_counts.values())),
            "candidate_count_by_scene": scene_counts,
            "supports_plus_10_recall_headroom_all_thresholds": all(supports),
            "per_threshold": threshold_reports,
        }

    input_after: dict[str, Any] = {
        "shadow_json": _sha256(shadow_json),
        "shadow_npz": _sha256(shadow_npz),
        "scenes": {},
    }
    for scene in scenes:
        input_after["scenes"][scene] = {
            "baseline": _sha256(baseline_paths[scene]),
            "gt": _sha256(gt_root / f"{scene}_bbox.npy"),
            "axis_alignment": _sha256(scan_root / scene / f"{scene}.txt"),
        }
    if input_after != input_before:
        raise BoxerTopKCeilingError("one or more audit inputs changed during execution")

    scene_frame_counts = {
        scene: int(
            len(
                np.unique(
                    arrays["per_view_frame_id"][
                        arrays["per_view_scene_index"] == scene_index
                    ]
                )
            )
        )
        for scene_index, scene in enumerate(scenes)
    }
    return {
        "schema": SCHEMA,
        "posthoc_dev_diagnostic": True,
        "not_deployable": True,
        "before_mobilesam": True,
        "MobileSAM_used": False,
        "H10_not_authorized": True,
        "full100_not_authorized": True,
        "active_birth_authorized": False,
        "threshold_tuning_performed": False,
        "selection_used_gt": False,
        "selection_used_semantics": False,
        "selection_used_only_frozen_source_score": True,
        "selection_tie_break": "ascending_source_row_then_sealed_npz_row",
        "selection_completed_before_gt_access": True,
        "class_mode": "class_agnostic",
        "scene_order": list(scenes),
        "thresholds": list(THRESHOLDS),
        "top_k_budgets": list(TOP_K_BUDGETS),
        "gt_count": total_gt,
        "target_recall_headroom_points": TARGET_RECALL_POINTS,
        "plus_10_required_additional_matches": required_matches,
        "sealed_per_view_candidate_count": int(len(arrays["per_view_scene_index"])),
        "frames_with_candidates_by_scene": scene_frame_counts,
        "budgets": budget_reports,
        "input_sha256_before": input_before,
        "input_sha256_after": input_after,
        "input_hash_identity": input_before == input_after,
        "sealed_sidecar": {
            "json_path": os.fspath(shadow_json),
            "json_sha256": sidecar_before["json"],
            "npz_path": os.fspath(shadow_npz),
            "npz_sha256": sidecar_before["npz"],
            "candidate_content_sha256": manifest["candidate_content_sha256"],
            "schema": manifest["schema"],
        },
        "conclusion_guardrail": (
            "Raw dev3 geometry headroom under fixed score-only per-frame budgets; "
            "not a deployable gate and cannot authorize H10 or full100."
        ),
    }


def _write_json_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise BoxerTopKCeilingError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise BoxerTopKCeilingError(f"refusing to overwrite output: {path}") from error


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-json", required=True, type=Path)
    parser.add_argument("--shadow-npz", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--scan-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output = args.out.resolve()
    protected = (
        args.baseline_root.resolve(),
        args.gt_root.resolve(),
        args.scan_root.resolve(),
    )
    if any(_is_relative_to(output, root) for root in protected) or output in {
        args.shadow_json.resolve(),
        args.shadow_npz.resolve(),
    }:
        raise BoxerTopKCeilingError("output must be outside all protected inputs")
    report = audit_scannet_boxer_per_view_topk_ceiling(
        shadow_json=args.shadow_json,
        shadow_npz=args.shadow_npz,
        baseline_root=args.baseline_root,
        gt_root=args.gt_root,
        scan_root=args.scan_root,
    )
    _write_json_create_only(output, report)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "gt_count": report["gt_count"],
                "top_k_budgets": list(TOP_K_BUDGETS),
                "H10_not_authorized": True,
                "out": os.fspath(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
