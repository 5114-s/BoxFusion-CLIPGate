#!/usr/bin/env python3
"""Audit frozen-B6/SGCDet controls without cross-run byte false positives.

The strict contract is checked inside each run, where the diagnostics contain
the exact pre/post sparse-refiner rows and their exported result indices.
Independent S0/S1/S2 GPU runs are compared for diagnostics only.  Their row
count, label/order, floating-point, and score-order drift are all reported but
are never treated as evidence that an observer mutated its own run.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


SCHEMA = "boxfusion.sgcdet_sparse_identity_audit.v2"
CROSS_RUN_MATCH_MIN_AABB_IOU = 0.50


@dataclass(frozen=True)
class Prediction:
    labels: np.ndarray
    corners: np.ndarray
    scores: np.ndarray

    @property
    def count(self) -> int:
        return int(self.scores.shape[0])


def _load_prediction(path: Path) -> Prediction:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(f"{path}: expected one-scene outer list")
    rows = payload[0]
    if not isinstance(rows, list):
        raise ValueError(f"{path}: prediction rows must be a list")
    labels: List[int] = []
    corners: List[np.ndarray] = []
    scores: List[float] = []
    for index, row in enumerate(rows):
        if not isinstance(row, tuple) or len(row) != 3:
            raise ValueError(
                f"{path}: row {index} must be (label, corners, score)"
            )
        label, raw_corners, score = row
        if isinstance(label, bool) or not isinstance(label, (int, np.integer)):
            raise ValueError(f"{path}: row {index} label must be an integer")
        array = np.asarray(raw_corners)
        if array.shape != (8, 3):
            raise ValueError(
                f"{path}: row {index} corners must have shape (8, 3)"
            )
        if array.dtype != np.float32:
            raise ValueError(
                f"{path}: row {index} corners must be float32, got {array.dtype}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{path}: row {index} corners are non-finite")
        value = float(score)
        if not np.isfinite(value):
            raise ValueError(f"{path}: row {index} score is non-finite")
        labels.append(int(label))
        corners.append(array.copy())
        scores.append(value)
    corner_array = (
        np.stack(corners).astype(np.float32, copy=False)
        if corners
        else np.empty((0, 8, 3), dtype=np.float32)
    )
    return Prediction(
        labels=np.asarray(labels, dtype=np.int64),
        corners=corner_array,
        scores=np.asarray(scores, dtype=np.float32),
    )


def _read_scene_names(scene_list: Path) -> Tuple[str, ...]:
    names = tuple(
        line.split()[0]
        for line in scene_list.read_text(encoding="utf-8").splitlines()
        if line.split()
    )
    if not names:
        raise ValueError(f"scene list is empty: {scene_list}")
    if len(set(names)) != len(names):
        raise ValueError(f"scene list contains duplicate IDs: {scene_list}")
    return names


def _prediction_paths(root: Path, scenes: Sequence[str]) -> Dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"missing prediction directory: {root}")
    expected = {f"{scene}_boxes.pkl" for scene in scenes}
    found = {path.name for path in root.glob("*_boxes.pkl") if path.is_file()}
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    if missing or extra:
        raise ValueError(
            f"{root}: prediction set mismatch; missing={missing}, extra={extra}"
        )
    return {scene: root / f"{scene}_boxes.pkl" for scene in scenes}


def _scalar(data: Mapping[str, np.ndarray], key: str) -> Any:
    if key not in data:
        raise KeyError(f"missing diagnostic key {key!r}")
    value = np.asarray(data[key])
    if value.shape != ():
        raise ValueError(f"diagnostic key {key!r} must be scalar")
    return value.item()


def _same_array(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_name: str,
    right_name: str,
    issues: List[str],
) -> None:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.dtype != right_array.dtype:
        issues.append(
            f"{left_name}/{right_name}: dtype differs "
            f"({left_array.dtype} != {right_array.dtype})"
        )
        return
    if left_array.shape != right_array.shape:
        issues.append(
            f"{left_name}/{right_name}: shape differs "
            f"({left_array.shape} != {right_array.shape})"
        )
        return
    if not np.array_equal(left_array, right_array, equal_nan=True):
        issues.append(f"{left_name}/{right_name}: values are not exact identity")


def _require_false(data: Mapping[str, np.ndarray], key: str, issues: List[str]) -> None:
    try:
        value = _scalar(data, key)
    except (KeyError, ValueError) as error:
        issues.append(str(error))
        return
    if bool(value):
        issues.append(f"{key}: expected false, received {value!r}")


def _require_true(data: Mapping[str, np.ndarray], key: str, issues: List[str]) -> None:
    try:
        value = _scalar(data, key)
    except (KeyError, ValueError) as error:
        issues.append(str(error))
        return
    if not bool(value):
        issues.append(f"{key}: expected true, received {value!r}")


def _strict_control_scene(
    *,
    label: str,
    expected_profile: str,
    prediction: Prediction,
    diagnostic_path: Path,
    sparse_model_enabled: bool,
) -> Dict[str, Any]:
    issues: List[str] = []
    full_prepost_available = False
    strict_geometry_rows = 0
    if not diagnostic_path.is_file():
        return {
            "ok": False,
            "issues": [f"missing diagnostics: {diagnostic_path}"],
        }
    try:
        archive_context = np.load(diagnostic_path, allow_pickle=True)
    except Exception as error:  # pragma: no cover - corrupt external artifact
        return {"ok": False, "issues": [f"cannot load {diagnostic_path}: {error}"]}
    with archive_context as data:
        try:
            profile = str(_scalar(data, "online_ablation_profile"))
        except (KeyError, ValueError) as error:
            profile = ""
            issues.append(str(error))
        if profile != expected_profile:
            issues.append(
                f"online_ablation_profile: expected {expected_profile!r}, "
                f"received {profile!r}"
            )

        for key in (
            "mutation_refit_enabled",
            "mutation_box_refiner_enabled",
            "mutation_joint_local_head_enabled",
            "mutation_joint_geometry_enabled",
            "mutation_joint_scores_enabled",
            "mutation_sparse_geometry_enabled",
            "mutation_supplemental_output_enabled",
            "mutation_soft_nms_enabled",
        ):
            _require_false(data, key, issues)
        _require_true(data, "mutation_quality_enabled", issues)
        _require_true(data, "sparse_collect_diagnostics", issues)
        if sparse_model_enabled:
            _require_true(data, "mutation_sparse_refiner_enabled", issues)
        else:
            _require_false(data, "mutation_sparse_refiner_enabled", issues)

        required_arrays = (
            "source_indices",
            "result_indices",
            "sparse_pair_source_indices",
            "sparse_pair_stable_ids",
            "track_ids",
            "boxes",
            "scores",
            "sparse_original_boxes",
            "sparse_active_boxes",
            "sparse_original_corners",
            "sparse_active_corners",
            "sparse_final_b6_scores",
            "refit_original_boxes",
            "refit_candidate_boxes",
            "refit_original_corners",
            "refit_candidate_corners",
            "refit_applied",
            "refit_boundary_delta",
            "sparse_input_valid",
            "sparse_output_valid",
            "sparse_accepted",
            "sparse_center_residual",
            "sparse_center_residual_fraction",
            "sparse_log_dimension_residual",
        )
        missing = [key for key in required_arrays if key not in data]
        if missing:
            issues.append(f"missing diagnostic arrays: {missing}")
            return {
                "ok": False,
                "profile": profile,
                "prediction_rows": prediction.count,
                "issues": issues,
            }

        result_indices = np.asarray(data["result_indices"])
        source_indices = np.asarray(data["source_indices"])
        pair_indices = np.asarray(data["sparse_pair_source_indices"])
        if result_indices.ndim != 1 or result_indices.dtype.kind not in "iu":
            issues.append("result_indices must be a one-dimensional integer array")
            mapped = np.empty((0,), dtype=np.int64)
        else:
            mapped = result_indices.astype(np.int64, copy=False)
            if mapped.size and (
                int(mapped.min()) < 0 or int(mapped.max()) >= prediction.count
            ):
                issues.append("result_indices contains an out-of-range output row")
            if mapped.size != np.unique(mapped).size:
                issues.append("result_indices contains duplicate output rows")
            if mapped.size > 1 and not np.all(np.diff(mapped) > 0):
                issues.append("result_indices must preserve strict output order")
        mapped_valid = bool(
            mapped.size == 0
            or (
                int(mapped.min()) >= 0
                and int(mapped.max()) < prediction.count
            )
        )
        if source_indices.ndim != 1 or source_indices.dtype.kind not in "iu":
            issues.append("source_indices must be a one-dimensional integer array")
        else:
            if source_indices.size and int(source_indices.min()) < 0:
                issues.append("source_indices contains a negative source row")
            if source_indices.size != np.unique(source_indices).size:
                issues.append("source_indices contains duplicate source rows")
            if source_indices.size > 1 and not np.all(np.diff(source_indices) > 0):
                issues.append("source_indices must preserve strict source order")
        _same_array(
            pair_indices,
            source_indices,
            left_name="sparse_pair_source_indices",
            right_name="source_indices",
            issues=issues,
        )
        _same_array(
            np.asarray(data["sparse_pair_stable_ids"]),
            np.asarray(data["track_ids"]),
            left_name="sparse_pair_stable_ids",
            right_name="track_ids",
            issues=issues,
        )

        row_count = int(mapped.size)
        strict_geometry_rows = row_count
        for key in required_arrays:
            array = np.asarray(data[key])
            if array.ndim >= 1 and int(array.shape[0]) != row_count:
                issues.append(
                    f"{key}: first dimension {array.shape[0]} != mapped rows {row_count}"
                )

        if mapped.size and mapped_valid:
            exported_corners = prediction.corners[mapped]
            exported_scores = prediction.scores[mapped]
            _same_array(
                exported_corners,
                np.asarray(data["sparse_original_corners"]),
                left_name="exported_corners[result_indices]",
                right_name="sparse_original_corners",
                issues=issues,
            )
            _same_array(
                exported_scores,
                np.asarray(data["sparse_final_b6_scores"]),
                left_name="exported_scores[result_indices]",
                right_name="sparse_final_b6_scores",
                issues=issues,
            )

        full_output_keys = (
            "output_geometry_schema",
            "output_pre_geometry_boxes",
            "output_pre_geometry_corners",
            "output_post_geometry_boxes",
            "output_post_geometry_corners",
            "output_source_indices",
            "output_stable_ids",
            "output_refit_applied",
        )
        present_full_keys = tuple(
            key for key in full_output_keys if key in data
        )
        if present_full_keys and len(present_full_keys) != len(full_output_keys):
            issues.append(
                "partial full-output pre/post diagnostics; present="
                f"{list(present_full_keys)}"
            )
        elif len(present_full_keys) == len(full_output_keys):
            full_prepost_available = True
            strict_geometry_rows = prediction.count
            try:
                output_schema = str(_scalar(data, "output_geometry_schema"))
            except (KeyError, ValueError) as error:
                output_schema = ""
                issues.append(str(error))
            expected_output_schema = (
                "boxfusion.full_output_geometry_prepost.v1"
            )
            if output_schema != expected_output_schema:
                issues.append(
                    "output_geometry_schema: expected "
                    f"{expected_output_schema!r}, received {output_schema!r}"
                )
            full_arrays = {
                key: np.asarray(data[key])
                for key in full_output_keys
                if key != "output_geometry_schema"
            }
            expected_full_shapes = {
                "output_pre_geometry_boxes": (prediction.count, 6),
                "output_pre_geometry_corners": (prediction.count, 8, 3),
                "output_post_geometry_boxes": (prediction.count, 6),
                "output_post_geometry_corners": (prediction.count, 8, 3),
                "output_source_indices": (prediction.count,),
                "output_stable_ids": (prediction.count,),
                "output_refit_applied": (prediction.count,),
            }
            expected_full_dtypes = {
                "output_pre_geometry_boxes": np.dtype(np.float32),
                "output_pre_geometry_corners": np.dtype(np.float32),
                "output_post_geometry_boxes": np.dtype(np.float32),
                "output_post_geometry_corners": np.dtype(np.float32),
                "output_source_indices": np.dtype(np.int64),
                "output_stable_ids": np.dtype(np.int64),
                "output_refit_applied": np.dtype(np.bool_),
            }
            for key, array in full_arrays.items():
                if array.shape != expected_full_shapes[key]:
                    issues.append(
                        f"{key}: shape {array.shape} != "
                        f"{expected_full_shapes[key]}"
                    )
                if array.dtype != expected_full_dtypes[key]:
                    issues.append(
                        f"{key}: dtype {array.dtype} != "
                        f"{expected_full_dtypes[key]}"
                    )
            _same_array(
                full_arrays["output_pre_geometry_boxes"],
                full_arrays["output_post_geometry_boxes"],
                left_name="output_pre_geometry_boxes",
                right_name="output_post_geometry_boxes",
                issues=issues,
            )
            _same_array(
                full_arrays["output_pre_geometry_corners"],
                full_arrays["output_post_geometry_corners"],
                left_name="output_pre_geometry_corners",
                right_name="output_post_geometry_corners",
                issues=issues,
            )
            _same_array(
                full_arrays["output_post_geometry_corners"],
                prediction.corners,
                left_name="output_post_geometry_corners",
                right_name="exported_prediction_corners",
                issues=issues,
            )
            full_sources = full_arrays["output_source_indices"]
            full_stable_ids = full_arrays["output_stable_ids"]
            if full_sources.size and (
                int(full_sources.min()) < 0
                or np.unique(full_sources).size != full_sources.size
                or (
                    full_sources.size > 1
                    and not np.all(np.diff(full_sources) > 0)
                )
            ):
                issues.append(
                    "output_source_indices must be unique, non-negative, "
                    "and strictly ordered"
                )
            if np.unique(full_stable_ids).size != full_stable_ids.size:
                issues.append("output_stable_ids must be unique")
            if full_arrays["output_refit_applied"].astype(bool).any():
                issues.append(
                    "output_refit_applied: control run contains a full-output "
                    "geometry mutation"
                )
            if mapped.size and mapped_valid:
                for full_key, observed_key in (
                    ("output_pre_geometry_boxes", "sparse_original_boxes"),
                    ("output_pre_geometry_corners", "sparse_original_corners"),
                    ("output_post_geometry_boxes", "sparse_active_boxes"),
                    ("output_post_geometry_corners", "sparse_active_corners"),
                    ("output_source_indices", "source_indices"),
                    ("output_stable_ids", "track_ids"),
                    ("output_refit_applied", "refit_applied"),
                ):
                    _same_array(
                        full_arrays[full_key][mapped],
                        np.asarray(data[observed_key]),
                        left_name=f"{full_key}[result_indices]",
                        right_name=observed_key,
                        issues=issues,
                    )
        for left, right in (
            ("boxes", "sparse_original_boxes"),
            ("sparse_original_boxes", "sparse_active_boxes"),
            ("sparse_original_corners", "sparse_active_corners"),
            ("refit_original_boxes", "sparse_original_boxes"),
            ("refit_candidate_boxes", "sparse_active_boxes"),
            ("refit_original_corners", "sparse_original_corners"),
            ("refit_candidate_corners", "sparse_active_corners"),
            ("scores", "sparse_final_b6_scores"),
        ):
            _same_array(
                np.asarray(data[left]),
                np.asarray(data[right]),
                left_name=left,
                right_name=right,
                issues=issues,
            )

        for key in (
            "refit_applied",
            "sparse_accepted",
        ):
            if np.asarray(data[key], dtype=bool).any():
                issues.append(f"{key}: control run contains an accepted mutation")
        boundary_delta = np.asarray(data["refit_boundary_delta"])
        if not np.isfinite(boundary_delta).all() or np.any(boundary_delta != 0):
            issues.append("refit_boundary_delta: control values must be exact zeros")

        try:
            summary = json.loads(str(_scalar(data, "summary_json")))
        except (KeyError, ValueError, json.JSONDecodeError) as error:
            summary = {}
            issues.append(f"summary_json: {error}")
        if not isinstance(summary, dict):
            issues.append("summary_json must decode to an object")
            summary = {}
        summary_flags = {
            "online_ablation_profile": expected_profile,
            "mutation_refit_enabled": False,
            "mutation_box_refiner_enabled": False,
            "mutation_joint_local_head_enabled": False,
            "mutation_joint_geometry_enabled": False,
            "mutation_joint_scores_enabled": False,
            "mutation_sparse_refiner_enabled": sparse_model_enabled,
            "mutation_sparse_geometry_enabled": False,
            "mutation_supplemental_output_enabled": False,
            "mutation_soft_nms_enabled": False,
            "sparse_collect_diagnostics": True,
        }
        for key, expected in summary_flags.items():
            if summary.get(key) != expected:
                issues.append(
                    f"summary_json.{key}: expected {expected!r}, "
                    f"received {summary.get(key)!r}"
                )
        if summary.get("sparse_accepted") != 0:
            issues.append(
                "summary_json.sparse_accepted: control run must have zero "
                f"cumulative accepted mutations, received "
                f"{summary.get('sparse_accepted')!r}"
            )
        # These summary values are cumulative over every intermediate
        # finalize call in a scene, not counts for the final exported rows.
        # Keep them as provenance only; do not use them as identity assertions.
        cumulative_summary = {
            key: summary.get(key)
            for key in (
                "sparse_instances",
                "sparse_inputs_valid",
                "sparse_invalid_identity",
                "sparse_unobserved_identity",
                "sparse_accepted",
            )
        }

    return {
        "ok": not issues,
        "label": label,
        "profile": profile,
        "prediction_rows": prediction.count,
        "mapped_rows": row_count,
        "exact_row_coverage": (
            float(strict_geometry_rows / prediction.count)
            if prediction.count
            else 1.0
        ),
        "strict_geometry_rows": strict_geometry_rows,
        "full_output_prepost_available": full_prepost_available,
        "unmapped_rows_without_prepost": prediction.count - row_count,
        "coverage_note": (
            "Full-output pre/post geometry is present and was checked with "
            "exact equality, including rows without object memory."
            if full_prepost_available
            else (
                "Legacy diagnostics: exact pre/post/export equality is "
                "available only for observed result_indices. Unmapped rows "
                "have no stored pre-refinement array, so byte identity is "
                "not claimed for those rows."
            )
        ),
        "cumulative_summary_report_only": cumulative_summary,
        "issues": issues,
    }


def _score_order(scores: np.ndarray) -> np.ndarray:
    indices = np.arange(scores.shape[0], dtype=np.int64)
    return np.lexsort((indices, -scores.astype(np.float64)))


def _aabb_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    left_min = left.min(axis=1).astype(np.float64)
    left_max = left.max(axis=1).astype(np.float64)
    right_min = right.min(axis=1).astype(np.float64)
    right_max = right.max(axis=1).astype(np.float64)
    intersection = np.maximum(
        np.minimum(left_max, right_max) - np.maximum(left_min, right_min),
        0.0,
    ).prod(axis=1)
    left_volume = np.maximum(left_max - left_min, 0.0).prod(axis=1)
    right_volume = np.maximum(right_max - right_min, 0.0).prod(axis=1)
    union = left_volume + right_volume - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


def _pairwise_aabb_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return every left/right AABB IoU without an external dependency."""
    if left.shape[0] == 0 or right.shape[0] == 0:
        return np.empty((left.shape[0], right.shape[0]), dtype=np.float64)
    left_min = left.min(axis=1).astype(np.float64)
    left_max = left.max(axis=1).astype(np.float64)
    right_min = right.min(axis=1).astype(np.float64)
    right_max = right.max(axis=1).astype(np.float64)
    intersection = np.maximum(
        np.minimum(left_max[:, None, :], right_max[None, :, :])
        - np.maximum(left_min[:, None, :], right_min[None, :, :]),
        0.0,
    ).prod(axis=2)
    left_volume = np.maximum(left_max - left_min, 0.0).prod(axis=1)
    right_volume = np.maximum(right_max - right_min, 0.0).prod(axis=1)
    union = left_volume[:, None] + right_volume[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


def _label_aware_iou_pairs(
    baseline: Prediction,
    control: Prediction,
    *,
    min_aabb_iou: float = CROSS_RUN_MATCH_MIN_AABB_IOU,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Greedily pair same-label rows by IoU with deterministic tie breaks.

    This is deliberately a report-only matcher rather than part of the strict
    identity proof.  Candidate pairs are ordered by descending IoU, then by
    baseline and control index.  The returned matches are sorted by baseline
    index so score-rank comparisons have a stable common-row coordinate.
    """
    pairs: List[Tuple[int, int, float]] = []
    matched_baseline: set[int] = set()
    matched_control: set[int] = set()
    labels = sorted(
        set(baseline.labels.astype(int).tolist())
        | set(control.labels.astype(int).tolist())
    )
    for label in labels:
        baseline_indices = np.flatnonzero(baseline.labels == label)
        control_indices = np.flatnonzero(control.labels == label)
        if baseline_indices.size == 0 or control_indices.size == 0:
            continue
        ious = _pairwise_aabb_iou(
            baseline.corners[baseline_indices],
            control.corners[control_indices],
        )
        candidates = sorted(
            (
                -float(ious[left_position, right_position]),
                int(left_index),
                int(right_index),
            )
            for left_position, left_index in enumerate(baseline_indices)
            for right_position, right_index in enumerate(control_indices)
            if float(ious[left_position, right_position]) >= min_aabb_iou
        )
        for negative_iou, left_index, right_index in candidates:
            if (
                left_index in matched_baseline
                or right_index in matched_control
            ):
                continue
            matched_baseline.add(left_index)
            matched_control.add(right_index)
            pairs.append((left_index, right_index, -negative_iou))
            if len(matched_baseline) == min(baseline.count, control.count):
                break
    pairs.sort(key=lambda item: (item[0], item[1]))
    unmatched_baseline = sorted(set(range(baseline.count)) - matched_baseline)
    unmatched_control = sorted(set(range(control.count)) - matched_control)
    return pairs, unmatched_baseline, unmatched_control


def _quantiles(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0:
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    result = np.quantile(array, (0.0, 0.50, 0.95, 0.99, 1.0))
    return {
        "min": float(result[0]),
        "p50": float(result[1]),
        "p95": float(result[2]),
        "p99": float(result[3]),
        "max": float(result[4]),
    }


def _cross_run_report(
    *,
    baseline_paths: Mapping[str, Path],
    control_paths: Mapping[str, Path],
    scenes: Sequence[str],
) -> Dict[str, Any]:
    structural_issues: List[str] = []
    corner_differences: List[float] = []
    score_differences: List[float] = []
    aligned_ious: List[float] = []
    rank_displacements: List[int] = []
    rank_mismatch_positions = 0
    byte_identical_files = 0
    prediction_rows = 0
    unmatched_baseline_total = 0
    unmatched_control_total = 0
    per_scene: Dict[str, Any] = {}
    for scene in scenes:
        baseline_path = baseline_paths[scene]
        control_path = control_paths[scene]
        baseline = _load_prediction(baseline_path)
        control = _load_prediction(control_path)
        byte_equal = baseline_path.read_bytes() == control_path.read_bytes()
        byte_identical_files += int(byte_equal)
        scene_report: Dict[str, Any] = {
            "baseline_rows": baseline.count,
            "control_rows": control.count,
            "pickle_bytes_equal": byte_equal,
        }
        if baseline.count != control.count:
            structural_issues.append(
                f"{scene}: prediction count differs "
                f"({baseline.count} != {control.count})"
            )
        labels_equal = np.array_equal(baseline.labels, control.labels)
        scene_report["label_sequence_equal"] = bool(labels_equal)
        scene_report["label_sequence_comparable"] = bool(
            baseline.count == control.count
        )
        if baseline.count == control.count and not labels_equal:
            structural_issues.append(f"{scene}: label sequence differs")
        pairs, unmatched_baseline, unmatched_control = _label_aware_iou_pairs(
            baseline,
            control,
            min_aabb_iou=CROSS_RUN_MATCH_MIN_AABB_IOU,
        )
        baseline_indices = np.asarray(
            [left_index for left_index, _, _ in pairs], dtype=np.int64
        )
        control_indices = np.asarray(
            [right_index for _, right_index, _ in pairs], dtype=np.int64
        )
        ious = np.asarray([iou for _, _, iou in pairs], dtype=np.float64)
        matched_count = len(pairs)
        prediction_rows += matched_count
        unmatched_baseline_total += len(unmatched_baseline)
        unmatched_control_total += len(unmatched_control)
        order_preserved = bool(
            control_indices.size <= 1 or np.all(np.diff(control_indices) > 0)
        )
        if not order_preserved:
            structural_issues.append(
                f"{scene}: matched prediction order changed"
            )
        if unmatched_baseline or unmatched_control:
            structural_issues.append(
                f"{scene}: same-label IoU matching at >= "
                f"{CROSS_RUN_MATCH_MIN_AABB_IOU:.2f} left "
                f"{len(unmatched_baseline)}/{len(unmatched_control)} "
                "baseline/control rows unmatched"
            )
        scene_report.update(
            {
                "matching_method": "label_aware_greedy_aabb_iou_v1",
                "matching_min_aabb_iou": CROSS_RUN_MATCH_MIN_AABB_IOU,
                "matched_prediction_order_preserved": order_preserved,
                "matched_pairs": matched_count,
                "matched_baseline_indices": baseline_indices.tolist(),
                "matched_control_indices": control_indices.tolist(),
                "unmatched_baseline_count": len(unmatched_baseline),
                "unmatched_control_count": len(unmatched_control),
                "unmatched_baseline_indices": unmatched_baseline,
                "unmatched_control_indices": unmatched_control,
            }
        )
        if matched_count == 0:
            scene_report.update(
                {
                    "corner_abs_max": 0.0,
                    "score_abs_max": 0.0,
                    "matched_aabb_iou_min": 0.0,
                    "score_rank_identical": True,
                    "score_rank_mismatch_positions": 0,
                    "score_rank_max_displacement": 0,
                }
            )
            per_scene[scene] = scene_report
            continue
        corner_abs = np.abs(
            baseline.corners[baseline_indices].astype(np.float64)
            - control.corners[control_indices].astype(np.float64)
        )
        score_abs = np.abs(
            baseline.scores[baseline_indices].astype(np.float64)
            - control.scores[control_indices].astype(np.float64)
        )
        baseline_order = _score_order(baseline.scores[baseline_indices])
        control_order = _score_order(control.scores[control_indices])
        mismatch = int(np.count_nonzero(baseline_order != control_order))
        baseline_rank = np.empty(matched_count, dtype=np.int64)
        control_rank = np.empty(matched_count, dtype=np.int64)
        baseline_rank[baseline_order] = np.arange(matched_count)
        control_rank[control_order] = np.arange(matched_count)
        displacement = np.abs(baseline_rank - control_rank)
        corner_differences.extend(corner_abs.ravel().tolist())
        score_differences.extend(score_abs.tolist())
        aligned_ious.extend(ious.tolist())
        rank_displacements.extend(displacement.tolist())
        rank_mismatch_positions += mismatch
        scene_report.update(
            {
                "corner_abs_max": float(corner_abs.max(initial=0.0)),
                "score_abs_max": float(score_abs.max(initial=0.0)),
                "matched_aabb_iou_min": (
                    float(ious.min()) if ious.size else 0.0
                ),
                "score_rank_identical": bool(mismatch == 0),
                "score_rank_mismatch_positions": mismatch,
                "score_rank_max_displacement": int(
                    displacement.max(initial=0)
                ),
            }
        )
        per_scene[scene] = scene_report
    return {
        "structural_ok": not structural_issues,
        "structural_issues": structural_issues,
        "scene_count": len(scenes),
        "prediction_rows_compared": prediction_rows,
        "matching_method": "label_aware_greedy_aabb_iou_v1",
        "matching_min_aabb_iou": CROSS_RUN_MATCH_MIN_AABB_IOU,
        "unmatched_baseline_rows": unmatched_baseline_total,
        "unmatched_control_rows": unmatched_control_total,
        "byte_identical_files_report_only": byte_identical_files,
        "corner_abs_drift": _quantiles(corner_differences),
        "score_abs_drift": _quantiles(score_differences),
        "matched_aabb_iou": _quantiles(aligned_ious),
        "score_rank": {
            "mismatch_positions": rank_mismatch_positions,
            "row_displacement": _quantiles(rank_displacements),
        },
        "per_scene": per_scene,
    }


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    scenes = _read_scene_names(args.scene_list)
    baseline_paths = _prediction_paths(args.baseline_root, scenes)
    observer_paths = _prediction_paths(args.observer_root, scenes)
    identity_paths = _prediction_paths(args.identity_root, scenes)

    strict: Dict[str, Any] = {}
    strict_issues: List[str] = []
    for label, profile, paths, diagnostics_root, model_enabled in (
        (
            "S1 observer",
            "sgcdet_sparse_observer",
            observer_paths,
            args.observer_diagnostics_root,
            False,
        ),
        (
            "S2 identity",
            "sgcdet_sparse_identity",
            identity_paths,
            args.identity_diagnostics_root,
            True,
        ),
    ):
        scene_reports: Dict[str, Any] = {}
        for scene in scenes:
            report = _strict_control_scene(
                label=label,
                expected_profile=profile,
                prediction=_load_prediction(paths[scene]),
                diagnostic_path=diagnostics_root / f"{scene}_tracks.npz",
                sparse_model_enabled=model_enabled,
            )
            scene_reports[scene] = report
            strict_issues.extend(
                f"{label}/{scene}: {issue}" for issue in report["issues"]
            )
        strict[label] = {
            "ok": all(report["ok"] for report in scene_reports.values()),
            "prediction_rows": sum(
                int(report.get("prediction_rows", 0))
                for report in scene_reports.values()
            ),
            "mapped_rows": sum(
                int(report.get("mapped_rows", 0))
                for report in scene_reports.values()
            ),
            "strict_geometry_rows": sum(
                int(report.get("strict_geometry_rows", 0))
                for report in scene_reports.values()
            ),
            "full_output_prepost_scenes": sum(
                bool(report.get("full_output_prepost_available", False))
                for report in scene_reports.values()
            ),
            "scenes": scene_reports,
        }
        total_rows = int(strict[label]["prediction_rows"])
        mapped_rows = int(strict[label]["mapped_rows"])
        strict_rows = int(strict[label]["strict_geometry_rows"])
        strict[label]["exact_row_coverage"] = (
            float(strict_rows / total_rows) if total_rows else 1.0
        )
        strict[label]["observed_diagnostic_coverage"] = (
            float(mapped_rows / total_rows) if total_rows else 1.0
        )

    cross_run = {
        "S0_vs_S1": _cross_run_report(
            baseline_paths=baseline_paths,
            control_paths=observer_paths,
            scenes=scenes,
        ),
        "S0_vs_S2": _cross_run_report(
            baseline_paths=baseline_paths,
            control_paths=identity_paths,
            scenes=scenes,
        ),
        "S1_vs_S2": _cross_run_report(
            baseline_paths=observer_paths,
            control_paths=identity_paths,
            scenes=scenes,
        ),
    }
    cross_run_warnings = [
        f"{comparison}: {issue}"
        for comparison, report in cross_run.items()
        for issue in report["structural_issues"]
    ]
    # Cross-run outputs are produced by independent upstream CUDA executions.
    # Even count/label/order can drift before the observer boundary, so these
    # comparisons are useful diagnostics but cannot invalidate the same-run
    # identity contract.  Only strict within-run evidence is a hard failure.
    issues = strict_issues
    full_prepost_complete = all(
        int(report.get("full_output_prepost_scenes", 0)) == len(scenes)
        for report in strict.values()
    )
    return {
        "schema": SCHEMA,
        "ok": not issues,
        "scene_list": str(args.scene_list),
        "scene_count": len(scenes),
        "contract": {
            "within_run": (
                "exact full-output pre/post geometry when the v1 full-output "
                "schema is present; legacy diagnostics fall back to exact "
                "observed result_indices"
            ),
            "cross_run": (
                "all independent-run drift, including count and label/order, "
                "is report-only; it never affects identity ok"
            ),
        },
        "known_limitation": (
            None
            if full_prepost_complete
            else (
                "One or more legacy diagnostics do not store full-scene "
                "pre/post arrays for unobserved output rows; strict byte "
                "identity is claimed only for their observed mapped rows. "
                "Rerun S1/S2 with the new diagnostics schema for 100% "
                "coverage."
            )
        ),
        "strict_within_run": strict,
        "cross_run_report": cross_run,
        "warnings": cross_run_warnings,
        "issues": issues,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--observer-root", type=Path, required=True)
    parser.add_argument("--identity-root", type=Path, required=True)
    parser.add_argument("--observer-diagnostics-root", type=Path, required=True)
    parser.add_argument("--identity-diagnostics-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete per-scene JSON report instead of a compact summary",
    )
    return parser


def _print_compact(report: Mapping[str, Any]) -> None:
    print(f"schema: {report.get('schema')}")
    print(f"ok: {str(bool(report.get('ok'))).lower()}")
    strict = report.get("strict_within_run", {})
    if isinstance(strict, Mapping):
        for label, raw in strict.items():
            if not isinstance(raw, Mapping):
                continue
            print(
                f"{label}: ok={str(bool(raw.get('ok'))).lower()}, "
                f"exact_rows={raw.get('strict_geometry_rows', 0)}/"
                f"{raw.get('prediction_rows', 0)}, "
                f"coverage={float(raw.get('exact_row_coverage', 0.0)):.4f}, "
                f"observed_rows={raw.get('mapped_rows', 0)}, "
                f"full_prepost_scenes="
                f"{raw.get('full_output_prepost_scenes', 0)}/"
                f"{len(raw.get('scenes', {}))}"
            )
    cross = report.get("cross_run_report", {})
    if isinstance(cross, Mapping):
        for label, raw in cross.items():
            if not isinstance(raw, Mapping):
                continue
            corner = raw.get("corner_abs_drift", {})
            score = raw.get("score_abs_drift", {})
            iou = raw.get("matched_aabb_iou", {})
            rank = raw.get("score_rank", {})
            print(
                f"{label}: structure_ok="
                f"{str(bool(raw.get('structural_ok'))).lower()}, "
                f"matched={int(raw.get('prediction_rows_compared', 0))}, "
                f"unmatched={int(raw.get('unmatched_baseline_rows', 0))}/"
                f"{int(raw.get('unmatched_control_rows', 0))}, "
                f"corner_max={float(corner.get('max', 0.0)):.8f}, "
                f"score_max={float(score.get('max', 0.0)):.8f}, "
                f"matched_iou_min={float(iou.get('min', 0.0)):.8f}, "
                f"rank_mismatch_positions={int(rank.get('mismatch_positions', 0))}"
            )
    limitation = report.get("known_limitation")
    if limitation:
        print(f"coverage limitation: {limitation}")
    issues = report.get("issues", [])
    if issues:
        print("issues:")
        for issue in issues:
            print(f"  - {issue}")
    warnings = report.get("warnings", [])
    if warnings:
        print("cross-run warnings (report-only):")
        for warning in warnings:
            print(f"  - {warning}")


def main() -> int:
    args = _parser().parse_args()
    try:
        report = audit(args)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        report = {
            "schema": SCHEMA,
            "ok": False,
            "issues": [str(error)],
        }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ok"] else 1
    else:
        _print_compact(report)
    if report["ok"]:
        print(
            "Sparse-refiner controls passed: strict same-run identity is "
            "valid; all cross-run drift was reported without affecting ok."
        )
        return 0
    print("Sparse-refiner control audit failed; inspect issues above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
