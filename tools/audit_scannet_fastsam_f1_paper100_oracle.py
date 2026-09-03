#!/usr/bin/env python3
"""Frozen paper100 F1 oracle for the inert FastSAM F0 candidates.

This is a read-only, GT-assisted capacity audit.  It neither materializes
predictions nor enables birth.  The implementation deliberately reuses the
ScanNet loading, AABB IoU, deterministic maximum matching, and official
constant-score evaluator from ``audit_scannet_boxer_unexplained_oracle``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
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
    load_scene_list,
    official_constant_evaluate,
    strict_maximum_matching,
)


SCHEMA = "boxfusion.scannet_fastsam_f1_paper100_oracle.v1"
F0_SCENE_SCHEMA = "boxfusion.scannet_fastsam_f0_full200.scene.v1"
F0_RECEIPT_SCHEMA = "boxfusion.scannet_fastsam_f0_full200.merge.v1"
THRESHOLDS = (0.15, 0.25, 0.50)
REQUIRED_ADDITIONAL_MATCHES = 144
TARGET_DELTA_AP_POINTS = 10.0
GEOMETRY_ATOL = 1e-12
GEOMETRY_RTOL = 1e-12

EXPECTED = {
    "scene_list_sha256": "4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5",
    "full200_scene_list_sha256": "0e7e722d3e93ec4b721f12293a3f1e98ca62d475b42cc8b9d491878a897e9bd1",
    "f0_receipt_sha256": "07249ead31ad150cb43d7a35f4c922ac70a8a2f95bcf0fcd24f61f944c1e58a1",
    "official_evaluator_sha256": "aea2a72940b7cc53ee273f9f235e2efc848e1994e22da5f439af9751e1e27c27",
    "sidecar_ledger_sha256": "c2666903a2f8098771d4359d21171fccd8b1df35e38166ef4920251abb94dac7",
    "native_ledger_sha256": "a5566c8b314917d2fa33b69f3e1f7f5372c4e0fe87caf3ab14216e63e6030066",
    "gt_ledger_sha256": "160dec394d87545ee6407f4a734266f65455b61d0b8d4c2701ae70f45f64b287",
    "axis_ledger_sha256": "dead3486b0c6647ae19083673a3451821b88d974a6a9401d06ea252edcbc3e5c",
    "run_signature_sha256": "bfc1e14bbbb5507226831efd8b864f69d5ada5dec4b68293d940a45366383286",
    "scene_count": 100,
    "full_scene_count": 200,
    "keyframe_count": 6817,
    "successful_frame_count": 6726,
    "candidate_count": 52299,
    "native_count": 1788,
    "gt_count": 1433,
}
EXPECTED_BASELINE_AP_POINTS = {
    "0.15": 31.0130259031,
    "0.25": 26.7911284298,
    "0.50": 12.0668518301,
}

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_KEYS = {
    "confidence",
    "mask_sha256",
    "pixel_count",
    "points_and_voxel_keys_sha256",
    "rank",
    "raw_index",
    "residual_pixel_count",
    "residual_ratio",
    "stored_point_count",
    "support_pixel_count",
    "tight_box_xyxy",
    "valid_pixel_count",
    "valid_ratio",
    "voxel_count",
    "world_center",
    "world_extent",
    "world_q02",
    "world_q98",
}
_SIGNS = np.asarray(
    [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)],
    dtype=bool,
)


class F1OracleError(ValueError):
    """Raised when an input violates the frozen F1 contract."""


@dataclass(frozen=True)
class FastSAMCandidate:
    scene_id: str
    frame_id: int
    frame_ordinal: int
    candidate_index: int
    rank: int
    raw_index: int
    aligned_minmax: np.ndarray

    @property
    def candidate_id(self) -> str:
        return (
            f"{self.scene_id}:frame{self.frame_id}:"
            f"candidate{self.candidate_index}:raw{self.raw_index}"
        )


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise F1OracleError(f"{label} must be a regular non-symlink file: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise F1OracleError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise F1OracleError(f"{label} must contain a JSON object: {path}")
    return value


def _threshold_key(threshold: float) -> str:
    return f"{threshold:.2f}"


def canonical_ordered_hash_ledger(
    scenes: Sequence[str], paths: Sequence[Path], label: str
) -> dict[str, Any]:
    if len(scenes) != len(paths):
        raise F1OracleError(f"{label} path count does not match scene count")
    entries = [
        [scene, _sha256(_regular_file(path, f"{label} for {scene}"))]
        for scene, path in zip(scenes, paths)
    ]
    packed = json.dumps(entries, ensure_ascii=True, separators=(",", ":")).encode(
        "ascii"
    )
    return {"entries": entries, "sha256": hashlib.sha256(packed).hexdigest()}


def _input_snapshot(
    *,
    scenes: Sequence[str],
    scene_list: Path,
    full_scene_list: Path,
    f0_receipt: Path,
    official_evaluator: Path,
    sidecar_root: Path,
    baseline_root: Path,
    gt_root: Path,
    scan_root: Path,
) -> dict[str, Any]:
    fixed_paths = {
        "scene_list": scene_list,
        "full200_scene_list": full_scene_list,
        "f0_receipt": f0_receipt,
        "official_evaluator": official_evaluator,
    }
    fixed = {
        name: {
            "path": os.fspath(path),
            "sha256": _sha256(_regular_file(path, name)),
        }
        for name, path in fixed_paths.items()
    }
    ledgers = {
        "sidecars": canonical_ordered_hash_ledger(
            scenes, [sidecar_root / f"{s}.json" for s in scenes], "F0 sidecar"
        ),
        "native_predictions": canonical_ordered_hash_ledger(
            scenes,
            [baseline_root / f"{s}_boxes.pkl" for s in scenes],
            "native prediction",
        ),
        "ground_truth": canonical_ordered_hash_ledger(
            scenes, [gt_root / f"{s}_bbox.npy" for s in scenes], "ScanNet GT"
        ),
        "axis_alignment": canonical_ordered_hash_ledger(
            scenes,
            [scan_root / s / f"{s}.txt" for s in scenes],
            "axis alignment",
        ),
    }
    return {"fixed_files": fixed, "ordered_scene_ledgers": ledgers}


def _validate_frozen_hashes(snapshot: Mapping[str, Any]) -> None:
    fixed = snapshot["fixed_files"]
    checks = {
        "scene_list": EXPECTED["scene_list_sha256"],
        "full200_scene_list": EXPECTED["full200_scene_list_sha256"],
        "f0_receipt": EXPECTED["f0_receipt_sha256"],
        "official_evaluator": EXPECTED["official_evaluator_sha256"],
    }
    for name, expected in checks.items():
        actual = fixed[name]["sha256"]
        if actual != expected:
            raise F1OracleError(
                f"frozen {name} SHA-256 mismatch: expected={expected}, actual={actual}"
            )
    ledgers = snapshot["ordered_scene_ledgers"]
    ledger_checks = {
        "sidecars": EXPECTED["sidecar_ledger_sha256"],
        "native_predictions": EXPECTED["native_ledger_sha256"],
        "ground_truth": EXPECTED["gt_ledger_sha256"],
        "axis_alignment": EXPECTED["axis_ledger_sha256"],
    }
    for name, expected in ledger_checks.items():
        actual = ledgers[name]["sha256"]
        if actual != expected:
            raise F1OracleError(
                f"frozen {name} ledger mismatch: expected={expected}, actual={actual}"
            )


def _vector3(value: Any, label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise F1OracleError(f"{label} must be a numeric length-3 vector") from error
    if array.shape != (3,) or not np.isfinite(array).all():
        raise F1OracleError(f"{label} must be a finite length-3 vector")
    return array


def candidate_q02_q98_aligned_minmax(
    row: Mapping[str, Any], alignment: np.ndarray, label: str = "candidate"
) -> np.ndarray:
    """Validate one sealed candidate and transform all eight AABB corners."""

    if set(row) != _CANDIDATE_KEYS:
        raise F1OracleError(
            f"{label} schema mismatch: missing={sorted(_CANDIDATE_KEYS - set(row))}, "
            f"extra={sorted(set(row) - _CANDIDATE_KEYS)}"
        )
    for key in ("mask_sha256", "points_and_voxel_keys_sha256"):
        if not isinstance(row[key], str) or _HASH_RE.fullmatch(row[key]) is None:
            raise F1OracleError(f"{label}.{key} is not a SHA-256 digest")
    for key in (
        "pixel_count",
        "rank",
        "raw_index",
        "residual_pixel_count",
        "stored_point_count",
        "support_pixel_count",
        "valid_pixel_count",
        "voxel_count",
    ):
        if type(row[key]) is not int or row[key] < 0:
            raise F1OracleError(f"{label}.{key} must be a non-negative integer")
    for key in ("confidence", "residual_ratio", "valid_ratio"):
        if isinstance(row[key], bool) or not isinstance(row[key], (int, float)):
            raise F1OracleError(f"{label}.{key} must be numeric")
        if not math.isfinite(float(row[key])):
            raise F1OracleError(f"{label}.{key} must be finite")
    tight = row["tight_box_xyxy"]
    if (
        not isinstance(tight, list)
        or len(tight) != 4
        or any(type(value) is not int for value in tight)
    ):
        raise F1OracleError(f"{label}.tight_box_xyxy must contain four integers")
    q02 = _vector3(row["world_q02"], f"{label}.world_q02")
    q98 = _vector3(row["world_q98"], f"{label}.world_q98")
    center = _vector3(row["world_center"], f"{label}.world_center")
    extent = _vector3(row["world_extent"], f"{label}.world_extent")
    if np.any(q98 <= q02):
        raise F1OracleError(f"{label} requires world_q98 > world_q02")
    if not np.allclose(
        center, (q02 + q98) / 2.0, rtol=GEOMETRY_RTOL, atol=GEOMETRY_ATOL
    ):
        raise F1OracleError(f"{label} center is inconsistent with q02/q98")
    if not np.allclose(
        extent, q98 - q02, rtol=GEOMETRY_RTOL, atol=GEOMETRY_ATOL
    ):
        raise F1OracleError(f"{label} extent is inconsistent with q02/q98")
    matrix = np.asarray(alignment, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise F1OracleError("axis alignment must be a finite 4x4 matrix")
    corners = np.where(_SIGNS, q98[None, :], q02[None, :])
    aligned = corners @ matrix[:3, :3].T + matrix[:3, 3]
    return np.concatenate((aligned.min(axis=0), aligned.max(axis=0)))


def _load_f0_candidates(
    *,
    path: Path,
    scene: str,
    scene_index: int,
    alignment: np.ndarray,
    receipt_sidecar_sha256: str,
) -> tuple[list[FastSAMCandidate], int, int]:
    if _sha256(_regular_file(path, f"F0 sidecar for {scene}")) != receipt_sidecar_sha256:
        raise F1OracleError(f"receipt sidecar hash mismatch: {scene}")
    payload = _read_json(path, f"F0 sidecar for {scene}")
    required = {
        "schema": F0_SCENE_SCHEMA,
        "complete": True,
        "scene_id": scene,
        "scene_index": scene_index,
        "run_signature_sha256": EXPECTED["run_signature_sha256"],
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise F1OracleError(
                f"F0 sidecar contract mismatch for {scene}.{key}: {payload.get(key)!r}"
            )
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise F1OracleError(f"F0 sidecar frames must be a list: {scene}")
    candidates: list[FastSAMCandidate] = []
    successful = 0
    seen_frame_ids: set[int] = set()
    for ordinal, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise F1OracleError(f"invalid F0 frame row: {scene}:{ordinal}")
        frame_id = frame.get("frame_id")
        if type(frame_id) is not int or frame_id < 0 or frame_id in seen_frame_ids:
            raise F1OracleError(f"invalid/duplicate F0 frame ID: {scene}:{ordinal}")
        seen_frame_ids.add(frame_id)
        if frame.get("frame_ordinal") != ordinal or type(frame.get("successful")) is not bool:
            raise F1OracleError(f"invalid F0 frame ordinal/status: {scene}:{frame_id}")
        if not frame["successful"]:
            if frame.get("funnel") is not None:
                raise F1OracleError(f"failed F0 frame has a funnel: {scene}:{frame_id}")
            continue
        successful += 1
        funnel = frame.get("funnel")
        if not isinstance(funnel, dict) or not isinstance(funnel.get("candidates"), list):
            raise F1OracleError(f"successful F0 frame lacks candidates: {scene}:{frame_id}")
        rows = funnel["candidates"]
        if funnel.get("selected_count") != len(rows):
            raise F1OracleError(f"F0 selected_count mismatch: {scene}:{frame_id}")
        for candidate_index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise F1OracleError(f"invalid candidate row: {scene}:{frame_id}")
            aligned_minmax = candidate_q02_q98_aligned_minmax(
                row, alignment, f"{scene}:{frame_id}:{candidate_index}"
            )
            if row["rank"] != candidate_index:
                raise F1OracleError(f"candidate rank/order mismatch: {scene}:{frame_id}")
            candidates.append(
                FastSAMCandidate(
                    scene_id=scene,
                    frame_id=frame_id,
                    frame_ordinal=ordinal,
                    candidate_index=candidate_index,
                    rank=row["rank"],
                    raw_index=row["raw_index"],
                    aligned_minmax=aligned_minmax,
                )
            )
    summary = payload.get("summary")
    counts = summary.get("counts") if isinstance(summary, dict) else None
    if not isinstance(counts, dict):
        raise F1OracleError(f"F0 sidecar summary counts missing: {scene}")
    if (
        counts.get("keyframes") != len(frames)
        or counts.get("successful_frames") != successful
        or counts.get("accepted_lifts") != len(candidates)
    ):
        raise F1OracleError(f"F0 sidecar summary count mismatch: {scene}")
    return candidates, len(frames), successful


def _json_evaluation(
    evaluation: Mapping[str, Any], scenes: Sequence[str]
) -> dict[str, Any]:
    masks = evaluation["matched_gt_masks"]
    assert isinstance(masks, list)
    return {
        key: value
        for key, value in evaluation.items()
        if key not in {"matched_gt_masks", "evaluation_order"}
    } | {
        "per_scene": {
            scene: {
                "greedy_tp": int(np.count_nonzero(mask)),
                "unmatched_gt_count": int(len(mask) - np.count_nonzero(mask)),
                "matched_gt_indices": np.flatnonzero(mask).astype(int).tolist(),
            }
            for scene, mask in zip(scenes, masks)
        }
    }


def evaluate_f1_threshold(
    *,
    scenes: Sequence[str],
    native_iou: Sequence[np.ndarray],
    candidate_iou: Sequence[np.ndarray],
    candidates: Sequence[Sequence[FastSAMCandidate]],
    gt_counts: Sequence[int],
    baseline_evaluation: Mapping[str, Any],
    threshold: float,
) -> dict[str, Any]:
    lengths = {len(scenes), len(native_iou), len(candidate_iou), len(candidates), len(gt_counts)}
    if len(lengths) != 1:
        raise F1OracleError("per-scene F1 inputs have inconsistent lengths")
    baseline_masks = baseline_evaluation["matched_gt_masks"]
    if not isinstance(baseline_masks, list) or len(baseline_masks) != len(scenes):
        raise F1OracleError("official baseline masks have inconsistent scene count")
    native_total = candidate_total = union_total = 0
    selected_total = 0
    suffix_iou: list[np.ndarray] = []
    selected_rows: dict[str, list[dict[str, Any]]] = {}
    per_scene: dict[str, Any] = {}
    for scene, native, candidate, rows, gt_count, official_mask in zip(
        scenes, native_iou, candidate_iou, candidates, gt_counts, baseline_masks
    ):
        if native.ndim != 2 or candidate.ndim != 2 or native.shape[1] != gt_count or candidate.shape[1] != gt_count:
            raise F1OracleError(f"IoU/GT shape mismatch: {scene}")
        native_pairs = strict_maximum_matching(native, threshold)
        candidate_pairs = strict_maximum_matching(candidate, threshold)
        union_pairs = strict_maximum_matching(
            np.concatenate((native, candidate), axis=0), threshold
        )
        suffix_pairs = strict_maximum_matching(
            candidate, threshold, ~np.asarray(official_mask, dtype=bool)
        )
        selected_indices = sorted(index for index, _ in suffix_pairs)
        target_by_candidate = {index: target for index, target in suffix_pairs}
        suffix_iou.append(candidate[selected_indices])
        selected_rows[scene] = [
            {
                "candidate_id": rows[index].candidate_id,
                "candidate_index": index,
                "target_gt_index": target_by_candidate[index],
                "target_iou": float(candidate[index, target_by_candidate[index]]),
            }
            for index in selected_indices
        ]
        native_total += len(native_pairs)
        candidate_total += len(candidate_pairs)
        union_total += len(union_pairs)
        selected_total += len(suffix_pairs)
        per_scene[scene] = {
            "native_maximum_matching_count": len(native_pairs),
            "candidate_maximum_matching_count": len(candidate_pairs),
            "union_maximum_matching_count": len(union_pairs),
            "additional_union_matching_over_native": len(union_pairs) - len(native_pairs),
            "gt_selected_suffix_count": len(suffix_pairs),
        }
    combined = [
        np.concatenate((native, suffix), axis=0)
        for native, suffix in zip(native_iou, suffix_iou)
    ]
    suffix_evaluation = official_constant_evaluate(combined, gt_counts, threshold)
    baseline_ap_points = float(baseline_evaluation["ap_points"])
    delta_ap_points = float(suffix_evaluation["ap_points"]) - baseline_ap_points
    additional = union_total - native_total
    return {
        "iou_threshold": threshold,
        "strict_iou_comparison": ">",
        "native_maximum_matching_count": native_total,
        "candidate_maximum_matching_count": candidate_total,
        "union_maximum_matching_count": union_total,
        "additional_union_matching_over_native": additional,
        "required_additional_union_matches": REQUIRED_ADDITIONAL_MATCHES,
        "passes_geometry_capacity": additional >= REQUIRED_ADDITIONAL_MATCHES,
        "baseline_official_constant_score": _json_evaluation(baseline_evaluation, scenes),
        "gt_selected_candidate_suffix": {
            "oracle_only": True,
            "deployable": False,
            "threshold_specific": True,
            "mathematical_upper_bound": False,
            "selection": "maximum_matching_to_official_native_greedy_unmatched_gt",
            "candidate_order": "F0_scene_frame_candidate_order",
            "native_rows_are_unchanged_scene_prefix": True,
            "formal_score": 1.0,
            "official_tie_order": "numpy.argsort_default_all_scores_1.0",
            "selected_candidate_count": selected_total,
            "official_evaluation": _json_evaluation(suffix_evaluation, scenes),
            "delta_ap_points": delta_ap_points,
            "passes_plus10_ap": delta_ap_points >= TARGET_DELTA_AP_POINTS,
            "per_scene_selection": selected_rows,
        },
        "per_scene": per_scene,
    }


def _validate_receipt(
    receipt: Mapping[str, Any], full_scenes: Sequence[str]
) -> dict[str, Mapping[str, Any]]:
    required = {
        "schema": F0_RECEIPT_SCHEMA,
        "complete": True,
        "overall_pass": True,
        "run_signature_sha256": EXPECTED["run_signature_sha256"],
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise F1OracleError(f"F0 receipt contract mismatch for {key}")
    coverage = receipt.get("coverage")
    if (
        not isinstance(coverage, dict)
        or coverage.get("scene_count") != EXPECTED["full_scene_count"]
        or coverage.get("scene_order") != list(full_scenes)
    ):
        raise F1OracleError("F0 receipt full200 coverage/order mismatch")
    rows = receipt.get("scenes")
    if not isinstance(rows, list) or len(rows) != len(full_scenes):
        raise F1OracleError("F0 receipt scene ledger mismatch")
    result: dict[str, Mapping[str, Any]] = {}
    for index, (scene, row) in enumerate(zip(full_scenes, rows)):
        if not isinstance(row, dict) or row.get("scene_id") != scene or row.get("scene_index") != index:
            raise F1OracleError("F0 receipt scene order mismatch")
        sidecar = row.get("sidecar")
        if not isinstance(sidecar, dict) or _HASH_RE.fullmatch(str(sidecar.get("sha256"))) is None:
            raise F1OracleError(f"F0 receipt sidecar seal missing: {scene}")
        result[scene] = row
    return result


def audit_scannet_fastsam_f1_paper100_oracle(
    *,
    scene_list: Path,
    full_scene_list: Path,
    f0_receipt: Path,
    sidecar_root: Path,
    baseline_root: Path,
    gt_root: Path,
    scan_root: Path,
    official_evaluator: Path,
) -> dict[str, Any]:
    scenes = load_scene_list(_regular_file(scene_list, "paper100 scene list"))
    full_scenes = load_scene_list(_regular_file(full_scene_list, "F0 full200 scene list"))
    if len(scenes) != EXPECTED["scene_count"] or len(full_scenes) != EXPECTED["full_scene_count"]:
        raise F1OracleError("frozen scene count mismatch")
    if scenes != full_scenes[: EXPECTED["scene_count"]]:
        raise F1OracleError("paper100 scene order is not the full200 prefix")

    before = _input_snapshot(
        scenes=scenes,
        scene_list=scene_list,
        full_scene_list=full_scene_list,
        f0_receipt=f0_receipt,
        official_evaluator=official_evaluator,
        sidecar_root=sidecar_root,
        baseline_root=baseline_root,
        gt_root=gt_root,
        scan_root=scan_root,
    )
    _validate_frozen_hashes(before)
    receipt_rows = _validate_receipt(_read_json(f0_receipt, "F0 final receipt"), full_scenes)

    gt_counts: list[int] = []
    native_iou: list[np.ndarray] = []
    candidate_iou: list[np.ndarray] = []
    candidates_by_scene: list[list[FastSAMCandidate]] = []
    scene_reports: dict[str, Any] = {}
    keyframes = successful_frames = 0
    native_count = gt_count = candidate_count = 0
    for scene_index, scene in enumerate(scenes):
        alignment = load_axis_alignment(scan_root / scene / f"{scene}.txt")
        gt = load_gt_minmax(gt_root / f"{scene}_bbox.npy")
        _, native = load_baseline_boxes(
            baseline_root / f"{scene}_boxes.pkl", alignment
        )
        receipt_sidecar = receipt_rows[scene]["sidecar"]
        rows, scene_keyframes, scene_successful = _load_f0_candidates(
            path=sidecar_root / f"{scene}.json",
            scene=scene,
            scene_index=scene_index,
            alignment=alignment,
            receipt_sidecar_sha256=str(receipt_sidecar["sha256"]),
        )
        boxes = (
            np.stack([row.aligned_minmax for row in rows])
            if rows
            else np.empty((0, 6), dtype=np.float64)
        )
        native_matrix = aligned_iou_matrix(native, gt)
        candidate_matrix = aligned_iou_matrix(boxes, gt)
        gt_counts.append(len(gt))
        native_iou.append(native_matrix)
        candidate_iou.append(candidate_matrix)
        candidates_by_scene.append(rows)
        keyframes += scene_keyframes
        successful_frames += scene_successful
        native_count += len(native)
        gt_count += len(gt)
        candidate_count += len(rows)
        scene_reports[scene] = {
            "scene_index": scene_index,
            "keyframe_count": scene_keyframes,
            "successful_frame_count": scene_successful,
            "candidate_count": len(rows),
            "native_prediction_count": len(native),
            "gt_count": len(gt),
        }
    actual_counts = {
        "scene_count": len(scenes),
        "keyframe_count": keyframes,
        "successful_frame_count": successful_frames,
        "candidate_count": candidate_count,
        "native_prediction_count": native_count,
        "gt_count": gt_count,
    }
    expected_counts = {
        "scene_count": EXPECTED["scene_count"],
        "keyframe_count": EXPECTED["keyframe_count"],
        "successful_frame_count": EXPECTED["successful_frame_count"],
        "candidate_count": EXPECTED["candidate_count"],
        "native_prediction_count": EXPECTED["native_count"],
        "gt_count": EXPECTED["gt_count"],
    }
    if actual_counts != expected_counts:
        raise F1OracleError(
            f"frozen paper100 census mismatch: expected={expected_counts}, actual={actual_counts}"
        )

    baseline = {
        threshold: official_constant_evaluate(native_iou, gt_counts, threshold)
        for threshold in THRESHOLDS
    }
    baseline_checks: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        key = _threshold_key(threshold)
        actual = float(baseline[threshold]["ap_points"])
        expected = EXPECTED_BASELINE_AP_POINTS[key]
        passed = math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9)
        baseline_checks[key] = {
            "expected_ap_points": expected,
            "actual_ap_points": actual,
            "absolute_error": abs(actual - expected),
            "passed": passed,
        }
        if not passed:
            raise F1OracleError(
                f"official native AP reproduction failed at IoU {key}: {actual} != {expected}"
            )

    per_threshold = {
        _threshold_key(threshold): evaluate_f1_threshold(
            scenes=scenes,
            native_iou=native_iou,
            candidate_iou=candidate_iou,
            candidates=candidates_by_scene,
            gt_counts=gt_counts,
            baseline_evaluation=baseline[threshold],
            threshold=threshold,
        )
        for threshold in THRESHOLDS
    }
    geometry_pass = all(row["passes_geometry_capacity"] for row in per_threshold.values())
    suffix_pass = all(
        row["gt_selected_candidate_suffix"]["passes_plus10_ap"]
        for row in per_threshold.values()
    )
    after = _input_snapshot(
        scenes=scenes,
        scene_list=scene_list,
        full_scene_list=full_scene_list,
        f0_receipt=f0_receipt,
        official_evaluator=official_evaluator,
        sidecar_root=sidecar_root,
        baseline_root=baseline_root,
        gt_root=gt_root,
        scan_root=scan_root,
    )
    if after != before:
        raise F1OracleError("one or more frozen inputs changed during F1 audit")
    overall_pass = geometry_pass and suffix_pass
    return {
        "schema": SCHEMA,
        "protocol": "F1-frozen-FastSAM-paper100-q02-q98-geometry-oracle",
        "oracle_only": True,
        "deployable": False,
        "gt_used": True,
        "birth_enabled": False,
        "native_predictions_modified": False,
        "candidate_filtering": "none",
        "cross_view_clustering": False,
        "candidate_geometry": "sealed_world_q02_q98_AABB_all_8_corners_axis_aligned",
        "candidate_order": "scene_then_frame_then_funnel_candidate",
        "score_mode": "constant_1.0",
        "official_tie_order": "numpy.argsort_default_all_scores_1.0",
        "class_mode": "class_agnostic",
        "strict_iou_comparison": ">",
        "thresholds": list(THRESHOLDS),
        "scene_order": scenes,
        "totals": actual_counts,
        "integrity": {
            "all_frozen_hashes_passed": True,
            "f0_receipt_overall_pass": True,
            "f0_run_signature_sha256": EXPECTED["run_signature_sha256"],
            "candidate_schema_validated": True,
            "census_passed": True,
            "official_baseline_reproduction": baseline_checks,
            "all_inputs_before_after_identity": True,
        },
        "per_threshold": per_threshold,
        "decision": {
            "target_delta_ap_points_each_threshold": TARGET_DELTA_AP_POINTS,
            "required_additional_union_matches_each_threshold": REQUIRED_ADDITIONAL_MATCHES,
            "geometry_capacity_passes_all_thresholds": geometry_pass,
            "constructive_suffix_plus10_ap_passes_all_thresholds": suffix_pass,
            "overall_pass": overall_pass,
            "authorize_past_only_selector_work": overall_pass,
            "authorize_active_fastsam_birth": False,
            "result": (
                "f1_pass_authorize_gt_free_past_only_selector_research"
                if overall_pass
                else "f1_fail_do_not_implement_active_fastsam_birth_from_f0_geometry"
            ),
        },
        "scenes": scene_reports,
        "input_sha256_before": before,
        "input_sha256_after": after,
    }


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _validate_output_path(output: Path, protected_roots: Sequence[Path]) -> None:
    if output.suffix.lower() != ".json":
        raise F1OracleError("F1 output must have a .json suffix")
    if output.exists() or output.is_symlink():
        raise F1OracleError(f"refusing to overwrite F1 output: {output}")
    if any(_is_within(output, root) for root in protected_roots):
        raise F1OracleError("F1 output must not be inside a protected input root")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen FastSAM paper100 F1 oracle")
    parser.add_argument(
        "--scene-list", type=Path,
        default=Path("evaluation/data_util/meta_data/scannetv2_val.txt")
    )
    parser.add_argument(
        "--full-scene-list", type=Path,
        default=Path("evaluation/data_util/meta_data/scannetv2_val_f0_full200.txt")
    )
    parser.add_argument(
        "--f0-receipt", type=Path,
        default=Path("logs/scannet_fastsam_f0_full200_score05/final/F0_FASTSAM_FULL200.json")
    )
    parser.add_argument(
        "--sidecar-root", type=Path,
        default=Path("logs/scannet_fastsam_f0_full200_score05/scenes")
    )
    parser.add_argument(
        "--baseline-root", type=Path,
        default=Path("results/scannet_t05_boxer_replay_active_score05")
    )
    parser.add_argument(
        "--gt-root", type=Path,
        default=Path("evaluation/data_util/scannet_train_detection_data")
    )
    parser.add_argument("--scan-root", type=Path, default=Path("/extra/ZhaoX/scannet_data/scans"))
    parser.add_argument(
        "--official-evaluator", type=Path,
        default=Path("upstream_clean/BoxFusion_shallow/evaluation/eval_scannet.py")
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("reports/fastsam_f1_paper100_oracle/F1_FASTSAM_PAPER100_ORACLE.json")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_output_path(
        args.out,
        (
            args.scene_list.parent,
            args.full_scene_list.parent,
            args.f0_receipt.parent,
            args.sidecar_root,
            args.baseline_root,
            args.gt_root,
            args.scan_root,
            args.official_evaluator.parent,
        ),
    )
    report = audit_scannet_fastsam_f1_paper100_oracle(
        scene_list=args.scene_list,
        full_scene_list=args.full_scene_list,
        f0_receipt=args.f0_receipt,
        sidecar_root=args.sidecar_root,
        baseline_root=args.baseline_root,
        gt_root=args.gt_root,
        scan_root=args.scan_root,
        official_evaluator=args.official_evaluator,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "out": os.fspath(args.out),
                "totals": report["totals"],
                "decision": report["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
