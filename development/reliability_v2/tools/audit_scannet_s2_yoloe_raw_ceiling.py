#!/usr/bin/env python3
"""Read-only post-hoc geometry ceiling for frozen S2 YOLOE diagnostics.

This audit deliberately consumes *every* direct diagnostic row.  The sealed
S2 materializer manifest is used only to annotate the terminal disposition of
each row; it cannot select, suppress, rank, or alter a candidate here.  Five
point-cloud AABB hypotheses are compared with the producer-reported AABB:
q00/q100, q01/q99, q02/q98, q05/q95, and q10/q90.

This is a dev-set diagnosis, not a deployable policy.  It never authorizes H10
access or active birth, irrespective of the numbers it reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
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
    official_constant_evaluate,
    strict_maximum_matching,
)


SCHEMA = "boxfusion.scannet_s2_yoloe_raw_ceiling.v1"
SHADOW_SCHEMA = "boxfusion.s2_yoloe_direct_shadow.v1"
SCENE_PATTERN = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")
DIAGNOSTIC_SUFFIX = "_tracks.npz"
PREDICTION_SUFFIX = "_boxes.pkl"
THRESHOLDS = (0.15, 0.25, 0.50)
GEOMETRY_QUANTILES: Mapping[str, float | None] = {
    "reported": None,
    "points_q00": 0.00,
    "points_q01": 0.01,
    "points_q02": 0.02,
    "points_q05": 0.05,
    "points_q10": 0.10,
}
_REQUIRED_DIAGNOSTIC_ARRAYS = {
    "scene_id",
    "boxes",
    "scores",
    "points",
    "point_mask",
    "source_indices",
    "track_ids",
    "result_indices",
}
_TERMINAL_REJECTION_KEYS = {
    "native_overlap_rejected_diagnostic_rows": "native_overlap",
    "self_nms_rejected_diagnostic_rows": "self_nms",
    "output_cap_rejected_diagnostic_rows": "output_cap",
}


class S2RawCeilingError(ValueError):
    """Raised when a frozen input or the read-only audit contract is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise S2RawCeilingError(f"{label} must be a regular non-symlink file: {path}")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    path = _regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise S2RawCeilingError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise S2RawCeilingError(f"{label} must contain a JSON object: {path}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise S2RawCeilingError(f"{label} must be an object")
    return value


def _threshold_key(threshold: float) -> str:
    return f"{threshold:.2f}"


def _aabb_corners_from_center_extent(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 6:
        raise S2RawCeilingError("candidate boxes must have shape Nx6")
    lower = boxes[:, :3] - boxes[:, 3:] / 2.0
    upper = boxes[:, :3] + boxes[:, 3:] / 2.0
    signs = np.asarray(
        [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)],
        dtype=np.int8,
    )
    return np.where(signs[None, :, :] == 0, lower[:, None, :], upper[:, None, :])


def _aabb_corners_from_minmax(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if lower.shape != upper.shape or lower.ndim != 2 or lower.shape[1] != 3:
        raise S2RawCeilingError("point-derived lower/upper arrays must have shape Nx3")
    signs = np.asarray(
        [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)],
        dtype=np.int8,
    )
    return np.where(signs[None, :, :] == 0, lower[:, None, :], upper[:, None, :])


def _align_corners_to_minmax(corners: np.ndarray, alignment: np.ndarray) -> np.ndarray:
    corners = np.asarray(corners, dtype=np.float64)
    if corners.ndim != 3 or corners.shape[1:] != (8, 3):
        raise S2RawCeilingError("candidate corners must have shape Nx8x3")
    aligned = corners @ alignment[:3, :3].T + alignment[:3, 3]
    return np.concatenate((aligned.min(axis=1), aligned.max(axis=1)), axis=1)


def _point_quantile_corners(
    points: np.ndarray, point_mask: np.ndarray, quantile: float
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for row, mask in zip(points, point_mask):
        valid = np.asarray(row[mask], dtype=np.float64)
        if len(valid) == 0:
            raise S2RawCeilingError("candidate point row has no valid samples")
        lower = np.quantile(valid, quantile, axis=0)
        upper = np.quantile(valid, 1.0 - quantile, axis=0)
        rows.append(np.concatenate((lower, upper)))
    minmax = np.asarray(rows, dtype=np.float64).reshape((-1, 6))
    return _aabb_corners_from_minmax(minmax[:, :3], minmax[:, 3:])


def _load_diagnostic(path: Path, scene: str) -> dict[str, np.ndarray]:
    path = _regular_file(path, f"frozen direct diagnostic for {scene}")
    try:
        with np.load(path, allow_pickle=False) as source:
            if not _REQUIRED_DIAGNOSTIC_ARRAYS.issubset(source.files):
                raise S2RawCeilingError(f"diagnostic schema is incomplete for {scene}")
            # Labels are intentionally neither read nor returned.
            arrays = {
                name: np.array(source[name], copy=True)
                for name in sorted(_REQUIRED_DIAGNOSTIC_ARRAYS)
            }
    except (OSError, ValueError) as error:
        if isinstance(error, S2RawCeilingError):
            raise
        raise S2RawCeilingError(f"invalid diagnostic NPZ for {scene}: {path}") from error
    if arrays["scene_id"].shape != () or str(arrays["scene_id"].item()) != scene:
        raise S2RawCeilingError(f"diagnostic scene ID mismatch for {scene}")
    boxes = arrays["boxes"]
    count = len(boxes)
    shapes = {
        "boxes": (count, 6),
        "scores": (count,),
        "points": (count, 512, 3),
        "point_mask": (count, 512),
        "source_indices": (count,),
        "track_ids": (count,),
        "result_indices": (count,),
    }
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            raise S2RawCeilingError(
                f"diagnostic {name} shape mismatch for {scene}: {arrays[name].shape}"
            )
    if arrays["point_mask"].dtype != np.dtype(bool):
        raise S2RawCeilingError(f"diagnostic point_mask is not Boolean for {scene}")
    for name in ("source_indices", "track_ids", "result_indices"):
        if arrays[name].dtype.kind not in "iu":
            raise S2RawCeilingError(f"diagnostic {name} is not integer for {scene}")
    if not np.isfinite(boxes).all() or not np.isfinite(arrays["scores"]).all():
        raise S2RawCeilingError(f"diagnostic boxes/scores are non-finite for {scene}")
    if np.any(boxes[:, 3:] <= 0.0):
        raise S2RawCeilingError(f"diagnostic box has non-positive extent for {scene}")
    if np.any((arrays["scores"] <= 0.0) | (arrays["scores"] > 1.0)):
        raise S2RawCeilingError(f"diagnostic score is outside (0,1] for {scene}")
    if np.any(arrays["source_indices"] != -1):
        raise S2RawCeilingError(f"direct diagnostic contains a non-supplemental row: {scene}")
    result_indices = arrays["result_indices"].astype(np.int64, copy=False)
    if np.any(result_indices < 0) or len(np.unique(result_indices)) != count or (
        count > 1 and np.any(np.diff(result_indices) <= 0)
    ):
        raise S2RawCeilingError(f"diagnostic result order is not strictly increasing for {scene}")
    if np.any(arrays["point_mask"].sum(axis=1) <= 0):
        raise S2RawCeilingError(f"diagnostic candidate has no valid points for {scene}")
    valid_xyz = arrays["points"][arrays["point_mask"]]
    if not np.isfinite(valid_xyz).all():
        raise S2RawCeilingError(f"diagnostic valid points are non-finite for {scene}")
    return arrays


def _load_manifest_preflight(
    *, manifest_path: Path, candidate_root: Path, baseline_root: Path
) -> tuple[dict[str, Any], tuple[str, ...], dict[str, dict[int, str]], dict[str, str]]:
    manifest_path = _regular_file(manifest_path, "sealed S2 manifest")
    before = {"manifest": _sha256(manifest_path)}
    manifest = _read_json(manifest_path, "sealed S2 manifest")
    required = {
        "schema": SHADOW_SCHEMA,
        "mode": "shadow",
        "output_inert": True,
        "birth": False,
        "active_authorized": False,
        "gt_access": False,
        "oracle_access": False,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise S2RawCeilingError(
                f"sealed S2 contract mismatch for {key}: {manifest.get(key)!r}"
            )
    scenes_raw = manifest.get("scene_order")
    if (
        not isinstance(scenes_raw, list)
        or not scenes_raw
        or any(not isinstance(scene, str) for scene in scenes_raw)
    ):
        raise S2RawCeilingError("sealed S2 scene order is invalid")
    scenes = tuple(scenes_raw)
    if len(set(scenes)) != len(scenes) or any(
        SCENE_PATTERN.fullmatch(scene) is None for scene in scenes
    ):
        raise S2RawCeilingError("sealed S2 scene order contains an invalid scene")
    if manifest.get("scene_count") != len(scenes):
        raise S2RawCeilingError("sealed S2 scene count differs from scene order")
    inputs = _mapping(manifest.get("input"), "sealed S2 input")
    if Path(str(inputs.get("candidate_root"))).resolve() != candidate_root:
        raise S2RawCeilingError("CLI candidate root differs from sealed candidate root")
    if Path(str(inputs.get("baseline_root"))).resolve() != baseline_root:
        raise S2RawCeilingError("CLI baseline root differs from sealed baseline root")
    scene_ledger = _mapping(manifest.get("scenes"), "sealed S2 scenes")
    if set(scene_ledger) != set(scenes):
        raise S2RawCeilingError("sealed S2 scene ledger differs from scene order")

    disposition: dict[str, dict[int, str]] = {}
    diagnostic_hashes: dict[str, str] = {}
    for scene in scenes:
        ledger = _mapping(scene_ledger[scene], f"sealed S2 scene {scene}")
        count = ledger.get("diagnostic_row_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise S2RawCeilingError(f"invalid diagnostic row count for {scene}")
        current: dict[int, str] = {}
        accepted = ledger.get("accepted_candidates")
        if not isinstance(accepted, list):
            raise S2RawCeilingError(f"invalid accepted candidate ledger for {scene}")
        for item in accepted:
            row = _mapping(item, f"accepted candidate for {scene}").get("diagnostic_row")
            if isinstance(row, bool) or not isinstance(row, int):
                raise S2RawCeilingError(f"invalid accepted diagnostic row for {scene}")
            if row in current:
                raise S2RawCeilingError(f"duplicate terminal mapping for {scene} row {row}")
            current[row] = "accepted"
        rejections = _mapping(ledger.get("terminal_rejections"), f"rejections for {scene}")
        if set(rejections) != set(_TERMINAL_REJECTION_KEYS):
            raise S2RawCeilingError(f"terminal rejection schema changed for {scene}")
        for key, reason in _TERMINAL_REJECTION_KEYS.items():
            rows = rejections[key]
            if not isinstance(rows, list) or any(
                isinstance(row, bool) or not isinstance(row, int) for row in rows
            ):
                raise S2RawCeilingError(f"invalid {key} ledger for {scene}")
            for row in rows:
                if row in current:
                    raise S2RawCeilingError(
                        f"duplicate terminal mapping for {scene} row {row}"
                    )
                current[row] = reason
        if set(current) != set(range(count)):
            raise S2RawCeilingError(
                f"terminal mapping does not cover every diagnostic row for {scene}"
            )
        if ledger.get("supplemental_rows_read_source_index_minus_one") != count:
            raise S2RawCeilingError(f"sealed S2 did not read every direct row for {scene}")
        hashes = {
            ledger.get("diagnostic_sha256_before"),
            ledger.get("diagnostic_sha256_after"),
        }
        if len(hashes) != 1 or not all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
            for value in hashes
        ):
            raise S2RawCeilingError(f"invalid diagnostic hash seal for {scene}")
        disposition[scene] = current
        diagnostic_hashes[scene] = next(iter(hashes))
    return manifest, scenes, disposition, {**before, **diagnostic_hashes}


def _best_iou_distribution(matrices: Sequence[np.ndarray]) -> dict[str, Any]:
    maxima = np.concatenate(
        [matrix.max(axis=1) if matrix.shape[1] else np.zeros(len(matrix)) for matrix in matrices]
    ) if matrices else np.empty(0, dtype=np.float64)
    if len(maxima) == 0:
        return {
            "count": 0,
            "minimum": 0.0,
            "mean": 0.0,
            "q25": 0.0,
            "median": 0.0,
            "q75": 0.0,
            "q90": 0.0,
            "q95": 0.0,
            "maximum": 0.0,
            "strictly_above": {_threshold_key(t): 0 for t in THRESHOLDS},
        }
    return {
        "count": int(len(maxima)),
        "minimum": float(maxima.min()),
        "mean": float(maxima.mean()),
        "q25": float(np.quantile(maxima, 0.25)),
        "median": float(np.quantile(maxima, 0.50)),
        "q75": float(np.quantile(maxima, 0.75)),
        "q90": float(np.quantile(maxima, 0.90)),
        "q95": float(np.quantile(maxima, 0.95)),
        "maximum": float(maxima.max()),
        "strictly_above": {
            _threshold_key(threshold): int(np.count_nonzero(maxima > threshold))
            for threshold in THRESHOLDS
        },
    }


def _row_annotation(
    *, scene: str, row: int, arrays: Mapping[str, np.ndarray], reason: str
) -> dict[str, Any]:
    return {
        "scene_id": scene,
        "diagnostic_row": int(row),
        "result_index": int(arrays["result_indices"][row]),
        "track_id": int(arrays["track_ids"][row]),
        "score": float(arrays["scores"][row]),
        "terminal_disposition": "accepted" if reason == "accepted" else "rejected",
        "terminal_reason": reason,
    }


def _audit_geometry(
    *,
    scenes: Sequence[str],
    candidate_iou: Sequence[np.ndarray],
    baseline_iou: Sequence[np.ndarray],
) -> dict[str, Any]:
    per_threshold: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        candidate_total = native_total = union_total = 0
        per_scene: dict[str, Any] = {}
        for scene, candidates, native in zip(scenes, candidate_iou, baseline_iou):
            candidate_pairs = strict_maximum_matching(candidates, threshold)
            native_pairs = strict_maximum_matching(native, threshold)
            union_pairs = strict_maximum_matching(
                np.concatenate((native, candidates), axis=0), threshold
            )
            candidate_total += len(candidate_pairs)
            native_total += len(native_pairs)
            union_total += len(union_pairs)
            per_scene[scene] = {
                "candidate_maximum_matching_count": len(candidate_pairs),
                "candidate_maximum_matching_pairs": [list(pair) for pair in candidate_pairs],
                "native_maximum_matching_count": len(native_pairs),
                "native_union_maximum_matching_count": len(union_pairs),
                "additional_union_matching_over_native": len(union_pairs)
                - len(native_pairs),
            }
        per_threshold[_threshold_key(threshold)] = {
            "iou_threshold": threshold,
            "strict_iou_comparison": ">",
            "candidate_maximum_matching_count": candidate_total,
            "native_maximum_matching_count": native_total,
            "native_union_maximum_matching_count": union_total,
            "additional_union_matching_over_native": union_total - native_total,
            "per_scene": per_scene,
        }
    return {
        "candidate_count": int(sum(len(matrix) for matrix in candidate_iou)),
        "best_iou_distribution": _best_iou_distribution(candidate_iou),
        "per_threshold": per_threshold,
    }


def _reported_unmatched_recoveries(
    *,
    scenes: Sequence[str],
    arrays_by_scene: Mapping[str, Mapping[str, np.ndarray]],
    disposition: Mapping[str, Mapping[int, str]],
    reported_iou: Sequence[np.ndarray],
    baseline_evaluation: Mapping[float, Mapping[str, Any]],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        baseline_masks = baseline_evaluation[threshold]["matched_gt_masks"]
        if not isinstance(baseline_masks, list):
            raise S2RawCeilingError("official evaluator did not return matched GT masks")
        all_rows: list[dict[str, Any]] = []
        matched_rows: list[dict[str, Any]] = []
        for scene, matrix, matched_mask in zip(scenes, reported_iou, baseline_masks):
            unmatched = ~np.asarray(matched_mask, dtype=bool)
            allowed = np.flatnonzero(unmatched)
            arrays = arrays_by_scene[scene]
            if len(allowed):
                restricted = matrix[:, allowed]
                best_position = restricted.argmax(axis=1) if restricted.shape[1] else np.zeros(len(matrix), int)
                best_value = restricted[np.arange(len(matrix)), best_position]
                for row in np.flatnonzero(best_value > threshold):
                    item = _row_annotation(
                        scene=scene,
                        row=int(row),
                        arrays=arrays,
                        reason=disposition[scene][int(row)],
                    )
                    item.update(
                        {
                            "baseline_official_unmatched_gt_index": int(
                                allowed[best_position[row]]
                            ),
                            "iou": float(best_value[row]),
                        }
                    )
                    all_rows.append(item)
            pairs = strict_maximum_matching(matrix, threshold, unmatched)
            for row, gt_index in pairs:
                item = _row_annotation(
                    scene=scene,
                    row=int(row),
                    arrays=arrays,
                    reason=disposition[scene][int(row)],
                )
                item.update(
                    {
                        "baseline_official_unmatched_gt_index": int(gt_index),
                        "iou": float(matrix[row, gt_index]),
                    }
                )
                matched_rows.append(item)
        report[_threshold_key(threshold)] = {
            "iou_threshold": threshold,
            "rows_with_any_strict_overlap_count": len(all_rows),
            "rows_with_any_strict_overlap": all_rows,
            "maximum_matching_recovery_count": len(matched_rows),
            "maximum_matching_recovery_pairs": matched_rows,
        }
    return report


def audit_scannet_s2_yoloe_raw_ceiling(
    *,
    candidate_root: Path,
    sealed_manifest: Path,
    baseline_root: Path,
    gt_root: Path,
    scan_root: Path,
) -> dict[str, Any]:
    """Audit every frozen direct candidate under six geometry hypotheses."""

    candidate_root = candidate_root.resolve()
    sealed_manifest = sealed_manifest.resolve()
    baseline_root = baseline_root.resolve()
    gt_root = gt_root.resolve()
    scan_root = scan_root.resolve()
    if not candidate_root.is_dir() or not baseline_root.is_dir():
        raise S2RawCeilingError("candidate and baseline roots must be directories")

    manifest, scenes, disposition, sealed_hashes = _load_manifest_preflight(
        manifest_path=sealed_manifest,
        candidate_root=candidate_root,
        baseline_root=baseline_root,
    )
    scene_ledger = _mapping(manifest["scenes"], "sealed S2 scenes")
    arrays_by_scene: dict[str, dict[str, np.ndarray]] = {}
    input_before: dict[str, Any] = {
        "sealed_manifest": _sha256(sealed_manifest),
        "scenes": {},
    }
    native_iou: list[np.ndarray] = []
    gt_counts: list[int] = []
    geometry_iou: dict[str, list[np.ndarray]] = {
        name: [] for name in GEOMETRY_QUANTILES
    }
    baseline_paths: dict[str, Path] = {}
    no_gt_hashes: dict[str, dict[str, str]] = {}

    # Complete the sealed no-GT preflight for every scene before opening the
    # first GT or axis-alignment file.
    for scene in scenes:
        diagnostic_path = _regular_file(
            candidate_root / f"{scene}{DIAGNOSTIC_SUFFIX}",
            f"frozen direct diagnostic for {scene}",
        )
        baseline_path = _regular_file(
            baseline_root / f"{scene}{PREDICTION_SUFFIX}",
            f"frozen T05 prediction for {scene}",
        )
        current_diagnostic_hash = _sha256(diagnostic_path)
        if current_diagnostic_hash != sealed_hashes[scene]:
            raise S2RawCeilingError(f"diagnostic SHA-256 differs from seal for {scene}")
        ledger = _mapping(scene_ledger[scene], f"sealed S2 scene {scene}")
        native_hash = _sha256(baseline_path)
        if native_hash not in {
            ledger.get("native_prediction_sha256_before"),
            ledger.get("native_prediction_sha256_after"),
        } or ledger.get("native_prediction_sha256_before") != native_hash or ledger.get(
            "native_prediction_sha256_after"
        ) != native_hash:
            raise S2RawCeilingError(f"baseline SHA-256 differs from seal for {scene}")

        arrays = _load_diagnostic(diagnostic_path, scene)
        if len(arrays["boxes"]) != ledger.get("diagnostic_row_count"):
            raise S2RawCeilingError(f"diagnostic row count differs from seal for {scene}")
        native_prefix_count = ledger.get("native_prefix_row_count")
        if (
            isinstance(native_prefix_count, bool)
            or not isinstance(native_prefix_count, int)
            or native_prefix_count < 0
        ):
            raise S2RawCeilingError(f"invalid sealed native prefix count for {scene}")
        expected_result_indices = np.arange(
            native_prefix_count,
            native_prefix_count + len(arrays["boxes"]),
            dtype=np.int64,
        )
        if not np.array_equal(
            arrays["result_indices"].astype(np.int64, copy=False),
            expected_result_indices,
        ):
            raise S2RawCeilingError(
                f"diagnostic result indices do not follow the sealed native prefix for {scene}"
            )
        arrays_by_scene[scene] = arrays
        baseline_paths[scene] = baseline_path
        no_gt_hashes[scene] = {
            "diagnostic": current_diagnostic_hash,
            "baseline": native_hash,
        }

    for scene in scenes:
        arrays = arrays_by_scene[scene]
        baseline_path = baseline_paths[scene]
        gt_path = _regular_file(gt_root / f"{scene}_bbox.npy", f"ScanNet GT for {scene}")
        alignment_path = _regular_file(
            scan_root / scene / f"{scene}.txt", f"axis alignment for {scene}"
        )
        alignment = load_axis_alignment(alignment_path)
        gt = load_gt_minmax(gt_path)
        _, native_aligned = load_baseline_boxes(baseline_path, alignment)
        native_iou.append(aligned_iou_matrix(native_aligned, gt))
        gt_counts.append(len(gt))

        reported_corners = _aabb_corners_from_center_extent(arrays["boxes"])
        for name, quantile in GEOMETRY_QUANTILES.items():
            corners = (
                reported_corners
                if quantile is None
                else _point_quantile_corners(
                    arrays["points"], arrays["point_mask"], quantile
                )
            )
            aligned = _align_corners_to_minmax(corners, alignment)
            geometry_iou[name].append(aligned_iou_matrix(aligned, gt))

        input_before["scenes"][scene] = {
            **no_gt_hashes[scene],
            "gt": _sha256(gt_path),
            "axis_alignment": _sha256(alignment_path),
        }

    baseline_evaluation = {
        threshold: official_constant_evaluate(native_iou, gt_counts, threshold)
        for threshold in THRESHOLDS
    }
    geometries = {
        name: {
            "source": (
                "producer_reported_center_extent"
                if quantile is None
                else f"valid_diagnostic_points_world_q{quantile:.2f}_q{1.0-quantile:.2f}"
            ),
            "candidate_membership": "all_direct_diagnostic_rows_unchanged",
            **_audit_geometry(
                scenes=scenes,
                candidate_iou=geometry_iou[name],
                baseline_iou=native_iou,
            ),
        }
        for name, quantile in GEOMETRY_QUANTILES.items()
    }
    recoveries = _reported_unmatched_recoveries(
        scenes=scenes,
        arrays_by_scene=arrays_by_scene,
        disposition=disposition,
        reported_iou=geometry_iou["reported"],
        baseline_evaluation=baseline_evaluation,
    )

    input_after: dict[str, Any] = {
        "sealed_manifest": _sha256(sealed_manifest),
        "scenes": {},
    }
    for scene in scenes:
        input_after["scenes"][scene] = {
            "diagnostic": _sha256(candidate_root / f"{scene}{DIAGNOSTIC_SUFFIX}"),
            "baseline": _sha256(baseline_root / f"{scene}{PREDICTION_SUFFIX}"),
            "gt": _sha256(gt_root / f"{scene}_bbox.npy"),
            "axis_alignment": _sha256(scan_root / scene / f"{scene}.txt"),
        }
    if input_after != input_before:
        raise S2RawCeilingError("one or more audit inputs changed during execution")

    candidate_counts = {
        scene: int(len(arrays_by_scene[scene]["boxes"])) for scene in scenes
    }
    return {
        "schema": SCHEMA,
        "posthoc_dev_diagnostic": True,
        "not_deployable": True,
        "H10_not_authorized": True,
        "h10_gt_accessed": False,
        "active_birth_authorized": False,
        "candidate_selection_applied": False,
        "candidate_suppression_applied": False,
        "candidate_ranking_applied": False,
        "candidate_geometry_mutated": False,
        "labels_read": False,
        "labels_used": False,
        "labels_output": False,
        "class_mode": "class_agnostic",
        "strict_iou_comparison": ">",
        "scene_order": list(scenes),
        "thresholds": list(THRESHOLDS),
        "geometry_order": list(GEOMETRY_QUANTILES),
        "candidate_membership": "all_direct_diagnostic_rows_unchanged",
        "candidate_count_by_scene": candidate_counts,
        "candidate_count": int(sum(candidate_counts.values())),
        "gt_count": int(sum(gt_counts)),
        "terminal_disposition_counts": {
            reason: int(
                sum(
                    value == reason
                    for scene in scenes
                    for value in disposition[scene].values()
                )
            )
            for reason in ("accepted", "native_overlap", "self_nms", "output_cap")
        },
        "geometries": geometries,
        "reported_geometry_official_unmatched_recoveries": recoveries,
        "input_sha256_before": input_before,
        "input_sha256_after": input_after,
        "input_hash_identity": input_before == input_after,
        "sealed_manifest_contract": {
            "path": os.fspath(sealed_manifest),
            "sha256": sealed_hashes["manifest"],
            "schema": manifest["schema"],
            "terminal_mapping_only": True,
        },
        "conclusion_guardrail": (
            "Post-hoc dev geometry ceiling only; no measured geometry or terminal "
            "reason is a deployable gate, and this report cannot authorize H10."
        ),
    }


def _write_json_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise S2RawCeilingError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise S2RawCeilingError(f"refusing to overwrite output: {path}") from error


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only raw-candidate geometry ceiling for frozen S2 dev scenes"
    )
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--sealed-manifest", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--scan-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    protected_roots = (
        args.candidate_root.resolve(),
        args.baseline_root.resolve(),
        args.gt_root.resolve(),
        args.scan_root.resolve(),
    )
    output = args.out.resolve()
    if any(_is_relative_to(output, root) for root in protected_roots):
        raise S2RawCeilingError("output must be outside every protected input root")
    if output == args.sealed_manifest.resolve():
        raise S2RawCeilingError("output cannot replace the sealed manifest")
    report = audit_scannet_s2_yoloe_raw_ceiling(
        candidate_root=args.candidate_root,
        sealed_manifest=args.sealed_manifest,
        baseline_root=args.baseline_root,
        gt_root=args.gt_root,
        scan_root=args.scan_root,
    )
    _write_json_create_only(output, report)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "scene_count": len(report["scene_order"]),
                "candidate_count": report["candidate_count"],
                "posthoc_dev_diagnostic": True,
                "H10_not_authorized": True,
                "out": os.fspath(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
