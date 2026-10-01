#!/usr/bin/env python3
"""Audit the frozen P1S candidate geometry without changing any input artifact.

The report deliberately reuses the P1-v2 recall coordinate and matching
contract.  It measures:

* each candidate's best aligned-GT IoU in fixed, machine-readable bands;
* the best P1S candidate for every GT missed by score-ordered B6 matching;
* center-only and size-only GT oracles for diagnostically useful subsets;
* the theoretical world-AABB representation ceiling and the exact
  aligned-GT -> enclosing-world-AABB -> aligned-AABB target round trip.

Prediction pickle files are trusted local artifacts and must not come from an
untrusted source.  All inputs are opened read-only.  ``--output`` is the only
operation that writes a file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.report_p1_residual_recall import (  # noqa: E402
    center_size_to_corners,
    corners_to_minmax,
    load_axis_alignment,
    load_gt_boxes,
    load_p1_candidates,
    load_predictions,
    pairwise_aabb_iou,
    score_ordered_match,
    transform_corners,
)


SCHEMA = "boxfusion.p1g.geometry_audit.v1"
RECALL_SCHEMA = "boxfusion.p1v2.recall_report.v1"
P1_DIAGNOSTIC_SCHEMA = "boxfusion.p1.residual_proposal_observer.v1"
P1S_PROFILE = "p1s_native_sparse_context_observer"
P1S_HEAD = "native_sparse_context_v1"
P1S_TARGET_SCOPE = "snapshot_inside_only"
MATCHING_CONTRACT = (
    "class-agnostic, stable score-descending, strict IoU > threshold, "
    "one-to-one per scene"
)
IOU_THRESHOLD = 0.50
_SCENE = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")
_AXES = ("x", "y", "z")

# The two bands adjacent to 0.50 intentionally distinguish equality from the
# strict ScanNet success condition.
_BANDS = (
    ("0p00_to_0p05", 0.00, 0.05, True, False),
    ("0p05_to_0p15", 0.05, 0.15, True, False),
    ("0p15_to_0p25", 0.15, 0.25, True, False),
    ("0p25_to_0p40", 0.25, 0.40, True, False),
    ("0p40_to_0p45", 0.40, 0.45, True, False),
    ("0p45_to_0p475", 0.45, 0.475, True, False),
    ("0p475_to_0p49", 0.475, 0.49, True, False),
    ("0p49_to_0p50_inclusive", 0.49, 0.50, True, True),
    ("strict_gt_0p50", 0.50, 1.00, False, True),
)
_QUANTILES = (
    ("q00", 0.00),
    ("q10", 0.10),
    ("q25", 0.25),
    ("q50", 0.50),
    ("q75", 0.75),
    ("q90", 0.90),
    ("q95", 0.95),
    ("q99", 0.99),
    ("q100", 1.00),
)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"recall JSON contains non-finite constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"recall JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _read_recall_report(path: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: malformed recall JSON") from error
    report = dict(_mapping(payload, "recall report"))
    expected_scalars = {
        "schema": RECALL_SCHEMA,
        "stage": "P1S",
        "matching_contract": MATCHING_CONTRACT,
        "observer_only": True,
    }
    for key, expected in expected_scalars.items():
        if report.get(key) != expected:
            raise ValueError(
                f"{path}: recall {key}={report.get(key)!r}, "
                f"expected {expected!r}"
            )
    if report.get("unsafe_scenes") != []:
        raise ValueError(f"{path}: recall report is not observer-safe")

    per_scene = _mapping(report.get("per_scene"), "recall per_scene")
    scenes = tuple(sorted(str(scene) for scene in per_scene))
    if not scenes:
        raise ValueError(f"{path}: recall report contains no scenes")
    invalid = [scene for scene in scenes if _SCENE.fullmatch(scene) is None]
    if invalid:
        raise ValueError(f"{path}: invalid ScanNet scene id {invalid[0]!r}")
    if _nonnegative_integer(report.get("scene_count"), "scene_count") != len(
        scenes
    ):
        raise ValueError(f"{path}: scene_count disagrees with per_scene")

    totals = {
        "ground_truth_count": 0,
        "baseline_prediction_count": 0,
        "p1_candidate_count": 0,
    }
    for scene in scenes:
        row = _mapping(per_scene[scene], f"per_scene[{scene}]")
        if row.get("p1_mutation_enabled") is not False:
            raise ValueError(f"{path}: {scene} is not observer-only")
        if _nonnegative_integer(
            row.get("p1_applied_count"), f"{scene}.p1_applied_count"
        ) != 0:
            raise ValueError(f"{path}: {scene} applied formal P1 output")
        totals["ground_truth_count"] += _nonnegative_integer(
            row.get("ground_truth_count"), f"{scene}.ground_truth_count"
        )
        totals["baseline_prediction_count"] += _nonnegative_integer(
            row.get("baseline_predictions"), f"{scene}.baseline_predictions"
        )
        totals["p1_candidate_count"] += _nonnegative_integer(
            row.get("p1_candidates"), f"{scene}.p1_candidates"
        )
        threshold = _mapping(
            _mapping(row.get("thresholds"), f"{scene}.thresholds").get("0.50"),
            f"{scene}.thresholds[0.50]",
        )
        for field in (
            "ground_truth_count",
            "b6_true_positives",
            "p1_true_positives",
            "union_true_positives",
        ):
            _nonnegative_integer(threshold.get(field), f"{scene}.{field}")

    for field, observed in totals.items():
        if _nonnegative_integer(report.get(field), field) != observed:
            raise ValueError(f"{path}: {field} disagrees with per_scene")
    p1 = _mapping(report.get("p1"), "recall p1")
    if _nonnegative_integer(p1.get("candidate_count"), "p1.candidate_count") != (
        totals["p1_candidate_count"]
    ):
        raise ValueError(f"{path}: p1 candidate count aliases disagree")
    return report, scenes


def _np_scalar(
    archive: Mapping[str, np.ndarray], key: str, path: Path
) -> Any:
    if key not in archive:
        raise ValueError(f"{path}: missing {key}")
    value = np.asarray(archive[key])
    if value.shape != () or value.dtype.hasobject:
        raise ValueError(f"{path}: {key} must be a non-object scalar")
    result = value.item()
    return result.decode("utf-8") if isinstance(result, bytes) else result


def _bool_scalar(
    archive: Mapping[str, np.ndarray],
    key: str,
    expected: bool,
    path: Path,
) -> None:
    if key not in archive:
        raise ValueError(f"{path}: missing {key}")
    value = np.asarray(archive[key])
    if value.shape != () or value.dtype != np.dtype(bool):
        raise ValueError(f"{path}: {key} must be a Boolean scalar")
    observed = bool(value.item())
    if observed is not expected:
        raise ValueError(
            f"{path}: unsafe {key}={observed}, expected {expected}"
        )


def _validate_p1s_diagnostic(path: Path, scene: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        expected_text = {
            "scene_id": scene,
            "p1_schema": P1_DIAGNOSTIC_SCHEMA,
            "p1_stage": "P1S",
            "p1_profile": P1S_PROFILE,
            "p1_head_architecture": P1S_HEAD,
            "p1_target_assignment_scope": P1S_TARGET_SCOPE,
        }
        for key, expected in expected_text.items():
            observed = _np_scalar(archive, key, path)
            if observed != expected:
                raise ValueError(
                    f"{path}: {key}={observed!r}, expected {expected!r}"
                )
        for key, expected in {
            "p1_enabled": True,
            "p1_observer_only": True,
            "p1_uses_ground_truth": False,
            "p1_reads_semantic_labels": False,
            "p1_mutation_enabled": False,
            "p1_complete": True,
            "p1_class_agnostic": True,
        }.items():
            _bool_scalar(archive, key, expected, path)
        applied = _np_scalar(archive, "p1_applied_count", path)
        if isinstance(applied, bool) or not isinstance(applied, int):
            raise ValueError(f"{path}: p1_applied_count must be an integer")
        if applied != 0:
            raise ValueError(f"{path}: observer applied formal output rows")
        regression_dim = _np_scalar(archive, "p1_regression_dim", path)
        if (
            isinstance(regression_dim, bool)
            or not isinstance(regression_dim, int)
            or regression_dim != 6
        ):
            raise ValueError(f"{path}: P1S regression must remain 6-D")
        if "p1_step_failed" in archive:
            failed = np.asarray(archive["p1_step_failed"])
            if failed.ndim != 1 or failed.dtype != np.dtype(bool):
                raise ValueError(f"{path}: invalid p1_step_failed")
            if bool(np.any(failed)):
                raise ValueError(f"{path}: P1S diagnostic contains failed steps")


def _minmax_to_center_size(boxes: np.ndarray) -> np.ndarray:
    values = np.asarray(boxes, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("minmax boxes must have shape [N,6]")
    sizes = values[:, 3:] - values[:, :3]
    if not np.isfinite(values).all() or (
        len(values) and np.any(sizes <= 0.0)
    ):
        raise ValueError("minmax boxes must have finite positive extents")
    return np.concatenate(
        ((values[:, :3] + values[:, 3:]) * 0.5, sizes), axis=1
    )


def _sorted_corners(corners: np.ndarray) -> np.ndarray:
    rows = [tuple(float(item) for item in row) for row in corners]
    return np.asarray(sorted(rows), dtype=np.float64)


def _validate_world_aabb_corners(corners: np.ndarray, path: Path) -> None:
    values = np.asarray(corners, dtype=np.float64)
    boxes = corners_to_minmax(values)
    canonical = center_size_to_corners(_minmax_to_center_size(boxes))
    for index, (observed, expected) in enumerate(zip(values, canonical)):
        if not np.allclose(
            _sorted_corners(observed),
            _sorted_corners(expected),
            rtol=1e-5,
            atol=1e-5,
        ):
            raise ValueError(
                f"{path}: P1S candidate {index} is not a world AABB"
            )


def _validate_alignment(alignment: np.ndarray, scene: str) -> np.ndarray:
    matrix = np.asarray(alignment, dtype=np.float64)
    linear = matrix[:3, :3]
    if not np.allclose(
        linear.T @ linear, np.eye(3), rtol=5e-4, atol=5e-4
    ) or not math.isclose(
        abs(float(np.linalg.det(linear))), 1.0, rel_tol=5e-4, abs_tol=5e-4
    ):
        raise ValueError(f"{scene}: axisAlignment must be a rigid transform")
    if (
        np.max(np.abs(linear[:2, 2])) > 5e-5
        or np.max(np.abs(linear[2, :2])) > 5e-5
        or not math.isclose(
            abs(float(linear[2, 2])), 1.0, rel_tol=5e-5, abs_tol=5e-5
        )
    ):
        raise ValueError(
            f"{scene}: world-AABB ceiling requires planar ScanNet alignment"
        )
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{scene}: singular axisAlignment") from error
    if not np.isfinite(inverse).all():
        raise ValueError(f"{scene}: non-finite inverse axisAlignment")
    return inverse


def _corresponding_aabb_iou(
    first_center: np.ndarray,
    first_size: np.ndarray,
    second_center: np.ndarray,
    second_size: np.ndarray,
) -> np.ndarray:
    left_center = np.asarray(first_center, dtype=np.float64)
    left_size = np.asarray(first_size, dtype=np.float64)
    right_center = np.asarray(second_center, dtype=np.float64)
    right_size = np.asarray(second_size, dtype=np.float64)
    if (
        left_center.shape != left_size.shape
        or left_center.shape != right_center.shape
        or left_center.shape != right_size.shape
        or left_center.ndim != 2
        or left_center.shape[1] != 3
        or not all(
            np.isfinite(value).all()
            for value in (left_center, left_size, right_center, right_size)
        )
        or np.any(left_size <= 0.0)
        or np.any(right_size <= 0.0)
    ):
        raise ValueError("corresponding center/size inputs are invalid")
    left_min = left_center - 0.5 * left_size
    left_max = left_center + 0.5 * left_size
    right_min = right_center - 0.5 * right_size
    right_max = right_center + 0.5 * right_size
    intersection = np.prod(
        np.maximum(
            np.minimum(left_max, right_max)
            - np.maximum(left_min, right_min),
            0.0,
        ),
        axis=1,
    )
    left_volume = np.prod(left_size, axis=1)
    right_volume = np.prod(right_size, axis=1)
    union = left_volume + right_volume - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


def world_aabb_representation_ceiling(
    gt_boxes: np.ndarray, alignment: np.ndarray
) -> np.ndarray:
    """Return the exact centered world-AABB IoU ceiling for planar alignment.

    A world AABB transformed by a planar alignment has aligned XY aspect ratio
    constrained to the interval obtained from the two positive columns of
    ``abs(R_xy)``.  Center and Z extent are freely representable.  Clamping the
    GT XY aspect ratio to that interval gives the exact maximum centered IoU.
    """

    boxes = np.asarray(gt_boxes, dtype=np.float64)
    center_size = _minmax_to_center_size(boxes)
    rotation = np.abs(np.asarray(alignment, dtype=np.float64)[:2, :2])
    tiny = np.finfo(np.float64).tiny
    endpoints = (
        float(rotation[0, 1] / max(rotation[1, 1], tiny)),
        float(rotation[0, 0] / max(rotation[1, 0], tiny)),
    )
    lower = min(endpoints)
    upper = max(endpoints)
    if upper > 1.0 / tiny:
        upper = math.inf
    target_ratio = center_size[:, 3] / center_size[:, 4]
    represented_ratio = np.maximum(target_ratio, lower)
    if math.isfinite(upper):
        represented_ratio = np.minimum(represented_ratio, upper)
    ceiling = np.minimum(
        target_ratio / represented_ratio,
        represented_ratio / target_ratio,
    )
    if not np.isfinite(ceiling).all():
        raise ValueError("world-AABB ceiling computation became non-finite")
    return np.clip(ceiling, 0.0, 1.0)


def world_aabb_target_roundtrip_iou(
    gt_boxes: np.ndarray,
    alignment: np.ndarray,
    inverse_alignment: np.ndarray | None = None,
) -> np.ndarray:
    """Measure the exact enclosing-world-AABB target encoding round trip."""

    boxes = np.asarray(gt_boxes, dtype=np.float64)
    inverse = (
        np.linalg.inv(np.asarray(alignment, dtype=np.float64))
        if inverse_alignment is None
        else np.asarray(inverse_alignment, dtype=np.float64)
    )
    aligned_corners = center_size_to_corners(
        _minmax_to_center_size(boxes)
    )
    world_oriented = transform_corners(aligned_corners, inverse)
    world_aabb = corners_to_minmax(world_oriented)
    world_aabb_corners = center_size_to_corners(
        _minmax_to_center_size(world_aabb)
    )
    roundtrip = corners_to_minmax(
        transform_corners(world_aabb_corners, alignment)
    )
    first = _minmax_to_center_size(roundtrip)
    second = _minmax_to_center_size(boxes)
    return _corresponding_aabb_iou(
        first[:, :3], first[:, 3:], second[:, :3], second[:, 3:]
    )


def _quantiles(values: Iterable[float]) -> dict[str, float | None]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("distribution values must be a finite vector")
    if not len(array):
        return {name: None for name, _ in _QUANTILES}
    return {
        name: float(np.quantile(array, quantile))
        for name, quantile in _QUANTILES
    }


def _band_rows(values: Iterable[float]) -> list[dict[str, Any]]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("IoU band values must be a finite vector")
    if len(array) and (np.any(array < -1e-9) or np.any(array > 1.0 + 1e-9)):
        raise ValueError("IoU band values must lie in [0,1]")
    array = np.clip(array, 0.0, 1.0)
    rows: list[dict[str, Any]] = []
    assigned = np.zeros(len(array), dtype=bool)
    for name, lower, upper, include_lower, include_upper in _BANDS:
        lower_mask = array >= lower if include_lower else array > lower
        upper_mask = array <= upper if include_upper else array < upper
        mask = lower_mask & upper_mask
        if bool(np.any(assigned & mask)):
            raise AssertionError("internal IoU bands overlap")
        assigned |= mask
        count = int(np.sum(mask))
        rows.append(
            {
                "name": name,
                "lower": float(lower),
                "upper": float(upper),
                "include_lower": bool(include_lower),
                "include_upper": bool(include_upper),
                "count": count,
                "fraction": float(count / len(array)) if len(array) else 0.0,
            }
        )
    if len(array) and not bool(np.all(assigned)):
        raise AssertionError("internal IoU bands do not cover [0,1]")
    return rows


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(tuple(values), dtype=np.float64)
    return {
        "count": int(len(array)),
        "quantiles": _quantiles(array),
        "bands": _band_rows(array),
        "strict_gt_0p50_count": int(np.sum(array > IOU_THRESHOLD)),
        "at_or_below_0p50_count": int(np.sum(array <= IOU_THRESHOLD)),
    }


def _axis_distributions(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("axis statistics require shape [N,3]")
    return {
        axis: _quantiles(array[:, index])
        for index, axis in enumerate(_AXES)
    }


def _oracle_row(
    *,
    scene: str,
    gt_index: int,
    candidate_index: int,
    candidate_id: Any,
    candidate_score: float,
    original_iou: float,
    candidate_box: np.ndarray,
    gt_box: np.ndarray,
) -> dict[str, Any]:
    candidate = _minmax_to_center_size(
        np.asarray(candidate_box, dtype=np.float64)[None]
    )[0]
    target = _minmax_to_center_size(
        np.asarray(gt_box, dtype=np.float64)[None]
    )[0]
    candidate_center = candidate[:3]
    candidate_size = candidate[3:]
    gt_center = target[:3]
    gt_size = target[3:]
    center_oracle = _corresponding_aabb_iou(
        gt_center[None],
        candidate_size[None],
        gt_center[None],
        gt_size[None],
    )[0]
    size_oracle = _corresponding_aabb_iou(
        candidate_center[None],
        gt_size[None],
        gt_center[None],
        gt_size[None],
    )[0]
    center_delta = np.abs(candidate_center - gt_center)
    center_distance = float(np.linalg.norm(center_delta))
    gt_diagonal = float(np.linalg.norm(gt_size))
    return {
        "scene_id": scene,
        "gt_index": int(gt_index),
        "candidate_index": int(candidate_index),
        "candidate_id": _json_identifier(candidate_id),
        "candidate_score": float(candidate_score),
        "original_iou": float(original_iou),
        "center_offset_m": center_distance,
        "center_offset_over_gt_diagonal": float(center_distance / gt_diagonal),
        "center_abs_offset_over_gt_extent": (
            center_delta / gt_size
        ).tolist(),
        "extent_ratio": (candidate_size / gt_size).tolist(),
        "volume_ratio": float(
            np.prod(candidate_size) / np.prod(gt_size)
        ),
        "center_oracle_iou": float(center_oracle),
        "size_oracle_iou": float(size_oracle),
        "joint_center_size_oracle_iou": 1.0,
    }


def _json_identifier(value: Any) -> str | int:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("candidate IDs must be strings or integers")
    return value


def _oracle_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        empty_axis = np.empty((0, 3), dtype=np.float64)
        return {
            "count": 0,
            "original_iou": _quantiles(()),
            "center_offset_m": _quantiles(()),
            "center_offset_over_gt_diagonal": _quantiles(()),
            "center_abs_offset_over_gt_extent": _axis_distributions(empty_axis),
            "extent_ratio": _axis_distributions(empty_axis),
            "volume_ratio": _quantiles(()),
            "center_oracle_iou": _quantiles(()),
            "size_oracle_iou": _quantiles(()),
            "center_oracle_gain": _quantiles(()),
            "size_oracle_gain": _quantiles(()),
            "original_strict_gt_0p50_count": 0,
            "center_oracle_strict_gt_0p50_count": 0,
            "size_oracle_strict_gt_0p50_count": 0,
            "either_oracle_strict_gt_0p50_count": 0,
        }
    original = np.asarray([row["original_iou"] for row in rows], dtype=float)
    center = np.asarray([row["center_oracle_iou"] for row in rows], dtype=float)
    size = np.asarray([row["size_oracle_iou"] for row in rows], dtype=float)
    center_axis = np.asarray(
        [row["center_abs_offset_over_gt_extent"] for row in rows], dtype=float
    )
    extent = np.asarray([row["extent_ratio"] for row in rows], dtype=float)
    return {
        "count": int(len(rows)),
        "original_iou": _quantiles(original),
        "center_offset_m": _quantiles(
            float(row["center_offset_m"]) for row in rows
        ),
        "center_offset_over_gt_diagonal": _quantiles(
            float(row["center_offset_over_gt_diagonal"]) for row in rows
        ),
        "center_abs_offset_over_gt_extent": _axis_distributions(center_axis),
        "extent_ratio": _axis_distributions(extent),
        "volume_ratio": _quantiles(
            float(row["volume_ratio"]) for row in rows
        ),
        "center_oracle_iou": _quantiles(center),
        "size_oracle_iou": _quantiles(size),
        "center_oracle_gain": _quantiles(center - original),
        "size_oracle_gain": _quantiles(size - original),
        "original_strict_gt_0p50_count": int(
            np.sum(original > IOU_THRESHOLD)
        ),
        "center_oracle_strict_gt_0p50_count": int(
            np.sum(center > IOU_THRESHOLD)
        ),
        "size_oracle_strict_gt_0p50_count": int(
            np.sum(size > IOU_THRESHOLD)
        ),
        "either_oracle_strict_gt_0p50_count": int(
            np.sum((center > IOU_THRESHOLD) | (size > IOU_THRESHOLD))
        ),
    }


def _select_iou(
    rows: Sequence[Mapping[str, Any]], lower: float, upper: float
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if float(row["original_iou"]) >= lower
        and float(row["original_iou"]) <= upper
    ]


def _exact_artifact_set(
    root: Path, expected: set[str], pattern: str, role: str
) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"{role} root not found: {root}")
    actual = {path.name for path in root.glob(pattern) if path.is_file()}
    if actual != expected:
        raise ValueError(
            f"{role} set mismatch: "
            f"missing={sorted(expected - actual)[:8]}, "
            f"extra={sorted(actual - expected)[:8]}"
        )


def _validate_recall_scene_counts(
    report: Mapping[str, Any],
    scene: str,
    *,
    gt_count: int,
    baseline_count: int,
    candidate_count: int,
    b6_tp50: int,
    p1_tp50: int,
    union_tp50: int,
) -> None:
    row = _mapping(
        _mapping(report["per_scene"], "per_scene")[scene],
        f"per_scene[{scene}]",
    )
    expected = {
        "ground_truth_count": gt_count,
        "baseline_predictions": baseline_count,
        "p1_candidates": candidate_count,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(
                f"{scene}: recall {key}={row.get(key)!r}, observed {value}"
            )
    threshold = _mapping(
        _mapping(row["thresholds"], f"{scene}.thresholds")["0.50"],
        f"{scene}.thresholds[0.50]",
    )
    for key, value in {
        "ground_truth_count": gt_count,
        "b6_true_positives": b6_tp50,
        "p1_true_positives": p1_tp50,
        "union_true_positives": union_tp50,
    }.items():
        if threshold.get(key) != value:
            raise ValueError(
                f"{scene}: recall 0.50 {key}={threshold.get(key)!r}, "
                f"recomputed {value}"
            )


def build_report(
    *,
    recall_report: Path,
    diagnostics_root: Path,
    prediction_root: Path,
    gt_root: Path,
    scans_root: Path,
) -> dict[str, Any]:
    """Build a deterministic, read-only P1S geometry audit report."""

    recall_path = Path(recall_report)
    diagnostics = Path(diagnostics_root)
    predictions = Path(prediction_root)
    ground_truth = Path(gt_root)
    scans = Path(scans_root)
    report, scenes = _read_recall_report(recall_path)
    _exact_artifact_set(
        diagnostics,
        {f"{scene}_tracks.npz" for scene in scenes},
        "scene*_tracks.npz",
        "diagnostic",
    )
    _exact_artifact_set(
        predictions,
        {f"{scene}_boxes.pkl" for scene in scenes},
        "scene*_boxes.pkl",
        "prediction",
    )
    if not ground_truth.is_dir():
        raise FileNotFoundError(f"ground-truth root not found: {ground_truth}")
    if not scans.is_dir():
        raise FileNotFoundError(f"scan root not found: {scans}")

    all_candidate_best: list[np.ndarray] = []
    all_missed_best: list[np.ndarray] = []
    all_ceiling: list[np.ndarray] = []
    all_roundtrip: list[np.ndarray] = []
    missed_ceiling: list[np.ndarray] = []
    missed_roundtrip: list[np.ndarray] = []
    candidate_oracle_rows: list[dict[str, Any]] = []
    gt_oracle_rows: list[dict[str, Any]] = []
    missed_records: list[dict[str, Any]] = []
    per_scene: dict[str, Any] = {}
    totals = {
        "ground_truth_count": 0,
        "baseline_prediction_count": 0,
        "candidate_count": 0,
        "b6_tp50": 0,
        "p1_tp50": 0,
        "union_tp50": 0,
    }

    for scene in scenes:
        diagnostic_path = diagnostics / f"{scene}_tracks.npz"
        prediction_path = predictions / f"{scene}_boxes.pkl"
        gt_path = ground_truth / f"{scene}_bbox.npy"
        _validate_p1s_diagnostic(diagnostic_path, scene)
        candidates = load_p1_candidates(
            diagnostic_path, expected_scene_id=scene
        )
        _validate_world_aabb_corners(
            candidates.corners_world, diagnostic_path
        )
        baseline = load_predictions(prediction_path)
        alignment = load_axis_alignment(scans, scene)
        inverse = _validate_alignment(alignment, scene)
        candidate_boxes = corners_to_minmax(
            transform_corners(candidates.corners_world, alignment)
        )
        baseline_boxes = corners_to_minmax(
            transform_corners(baseline.corners_world, alignment)
        )
        gt_boxes = load_gt_boxes(gt_path)
        candidate_iou = pairwise_aabb_iou(candidate_boxes, gt_boxes)
        baseline_iou = pairwise_aabb_iou(baseline_boxes, gt_boxes)

        if len(gt_boxes):
            candidate_best = (
                candidate_iou.max(axis=1)
                if len(candidate_boxes)
                else np.empty(0, dtype=np.float64)
            )
            candidate_gt = (
                candidate_iou.argmax(axis=1)
                if len(candidate_boxes)
                else np.empty(0, dtype=np.int64)
            )
        else:
            candidate_best = np.zeros(len(candidate_boxes), dtype=np.float64)
            candidate_gt = np.full(len(candidate_boxes), -1, dtype=np.int64)
        if len(candidate_boxes):
            gt_best = (
                candidate_iou.max(axis=0)
                if len(gt_boxes)
                else np.empty(0, dtype=np.float64)
            )
            gt_candidate = (
                candidate_iou.argmax(axis=0)
                if len(gt_boxes)
                else np.empty(0, dtype=np.int64)
            )
        else:
            gt_best = np.zeros(len(gt_boxes), dtype=np.float64)
            gt_candidate = np.full(len(gt_boxes), -1, dtype=np.int64)

        b6_match = score_ordered_match(
            baseline_iou, baseline.scores, IOU_THRESHOLD
        )
        p1_match = score_ordered_match(
            candidate_iou,
            candidates.scores,
            IOU_THRESHOLD,
            tie_break_ids=candidates.candidate_ids,
        )
        union_iou = np.concatenate((baseline_iou, candidate_iou), axis=0)
        union_scores = np.concatenate(
            (baseline.scores, candidates.scores), axis=0
        )
        union_match = score_ordered_match(
            union_iou, union_scores, IOU_THRESHOLD
        )
        b6_covered = np.zeros(len(gt_boxes), dtype=bool)
        b6_covered[b6_match.matched_gt] = True
        missed = ~b6_covered

        ceiling = world_aabb_representation_ceiling(gt_boxes, alignment)
        roundtrip = world_aabb_target_roundtrip_iou(
            gt_boxes, alignment, inverse
        )
        if np.any(roundtrip > ceiling + 1e-7):
            raise AssertionError(
                f"{scene}: target round trip exceeds representation ceiling"
            )

        for candidate_index, gt_index in enumerate(candidate_gt.tolist()):
            if gt_index < 0:
                continue
            candidate_oracle_rows.append(
                _oracle_row(
                    scene=scene,
                    gt_index=gt_index,
                    candidate_index=candidate_index,
                    candidate_id=candidates.candidate_ids[candidate_index],
                    candidate_score=candidates.scores[candidate_index],
                    original_iou=candidate_best[candidate_index],
                    candidate_box=candidate_boxes[candidate_index],
                    gt_box=gt_boxes[gt_index],
                )
            )
        for gt_index, candidate_index in enumerate(gt_candidate.tolist()):
            if candidate_index < 0:
                continue
            row = _oracle_row(
                scene=scene,
                gt_index=gt_index,
                candidate_index=candidate_index,
                candidate_id=candidates.candidate_ids[candidate_index],
                candidate_score=candidates.scores[candidate_index],
                original_iou=gt_best[gt_index],
                candidate_box=candidate_boxes[candidate_index],
                gt_box=gt_boxes[gt_index],
            )
            row["b6_missed_at_0p50"] = bool(missed[gt_index])
            gt_oracle_rows.append(row)

        scene_missed_records: list[dict[str, Any]] = []
        for gt_index in np.flatnonzero(missed).tolist():
            record: dict[str, Any] = {
                "scene_id": scene,
                "gt_index": int(gt_index),
                "best_candidate_iou": float(gt_best[gt_index]),
                "world_aabb_representation_ceiling_iou": float(
                    ceiling[gt_index]
                ),
                "world_aabb_target_roundtrip_iou": float(
                    roundtrip[gt_index]
                ),
            }
            candidate_index = int(gt_candidate[gt_index])
            if candidate_index >= 0:
                geometry = _oracle_row(
                    scene=scene,
                    gt_index=gt_index,
                    candidate_index=candidate_index,
                    candidate_id=candidates.candidate_ids[candidate_index],
                    candidate_score=candidates.scores[candidate_index],
                    original_iou=gt_best[gt_index],
                    candidate_box=candidate_boxes[candidate_index],
                    gt_box=gt_boxes[gt_index],
                )
                record.update(geometry)
                record["best_candidate_iou"] = record.pop("original_iou")
            else:
                record.update(
                    {
                        "candidate_index": None,
                        "candidate_id": None,
                        "candidate_score": None,
                    }
                )
            scene_missed_records.append(record)
            missed_records.append(record)

        _validate_recall_scene_counts(
            report,
            scene,
            gt_count=len(gt_boxes),
            baseline_count=len(baseline_boxes),
            candidate_count=len(candidate_boxes),
            b6_tp50=b6_match.true_positive_count,
            p1_tp50=p1_match.true_positive_count,
            union_tp50=union_match.true_positive_count,
        )
        all_candidate_best.append(candidate_best)
        all_missed_best.append(gt_best[missed])
        all_ceiling.append(ceiling)
        all_roundtrip.append(roundtrip)
        missed_ceiling.append(ceiling[missed])
        missed_roundtrip.append(roundtrip[missed])
        totals["ground_truth_count"] += len(gt_boxes)
        totals["baseline_prediction_count"] += len(baseline_boxes)
        totals["candidate_count"] += len(candidate_boxes)
        totals["b6_tp50"] += b6_match.true_positive_count
        totals["p1_tp50"] += p1_match.true_positive_count
        totals["union_tp50"] += union_match.true_positive_count
        per_scene[scene] = {
            "ground_truth_count": int(len(gt_boxes)),
            "baseline_prediction_count": int(len(baseline_boxes)),
            "candidate_count": int(len(candidate_boxes)),
            "candidate_best_iou": _distribution(candidate_best),
            "b6_true_positives_at_0p50": int(b6_match.true_positive_count),
            "b6_missed_count_at_0p50": int(np.sum(missed)),
            "b6_missed_best_candidate_iou": _distribution(gt_best[missed]),
            "world_aabb_representation_ceiling_iou": _distribution(ceiling),
            "world_aabb_target_roundtrip_iou": _distribution(roundtrip),
            "b6_missed_records": scene_missed_records,
        }

    candidate_best_values = np.concatenate(all_candidate_best)
    missed_best_values = np.concatenate(all_missed_best)
    ceiling_values = np.concatenate(all_ceiling)
    roundtrip_values = np.concatenate(all_roundtrip)
    missed_ceiling_values = np.concatenate(missed_ceiling)
    missed_roundtrip_values = np.concatenate(missed_roundtrip)
    missed_oracle_rows = [
        row for row in gt_oracle_rows if row["b6_missed_at_0p50"]
    ]
    expected_totals = {
        "ground_truth_count": report["ground_truth_count"],
        "baseline_prediction_count": report["baseline_prediction_count"],
        "candidate_count": report["p1_candidate_count"],
    }
    for key, expected in expected_totals.items():
        if totals[key] != expected:
            raise ValueError(
                f"recall {key}={expected}, recomputed {totals[key]}"
            )
    threshold50 = _mapping(
        _mapping(report["thresholds"], "recall thresholds")["0.50"],
        "recall thresholds[0.50]",
    )
    for report_key, total_key in (
        ("b6_true_positives", "b6_tp50"),
        ("p1_true_positives", "p1_tp50"),
        ("union_true_positives", "union_tp50"),
    ):
        if threshold50.get(report_key) != totals[total_key]:
            raise ValueError(
                f"recall {report_key}={threshold50.get(report_key)!r}, "
                f"recomputed {totals[total_key]}"
            )

    ceiling_ratio = np.divide(
        roundtrip_values,
        ceiling_values,
        out=np.zeros_like(roundtrip_values),
        where=ceiling_values > 0.0,
    )
    return {
        "schema": SCHEMA,
        "stage": "P1G_GEOMETRY_AUDIT_OF_P1S",
        "observer_only": True,
        "matching_contract": MATCHING_CONTRACT,
        "iou_threshold": {
            "value": IOU_THRESHOLD,
            "comparison": "strict_greater_than",
        },
        "inputs": {
            "recall_report": str(recall_path.resolve()),
            "diagnostics_root": str(diagnostics.resolve()),
            "prediction_root": str(predictions.resolve()),
            "ground_truth_root": str(ground_truth.resolve()),
            "scans_root": str(scans.resolve()),
        },
        "scene_count": int(len(scenes)),
        "ground_truth_count": int(totals["ground_truth_count"]),
        "baseline_prediction_count": int(
            totals["baseline_prediction_count"]
        ),
        "candidate_count": int(totals["candidate_count"]),
        "candidate_best_iou": _distribution(candidate_best_values),
        "b6_missed_best_iou": {
            "definition": (
                "per-GT best P1S IoU after B6 score-ordered matching misses "
                "that GT at strict IoU > 0.50"
            ),
            **_distribution(missed_best_values),
            "records": missed_records,
        },
        "center_size_oracle": {
            "definition": (
                "center oracle preserves candidate extent and uses GT center; "
                "size oracle preserves candidate center and uses GT extent"
            ),
            "candidate_best_iou_0p25_to_0p50_inclusive": _oracle_summary(
                _select_iou(candidate_oracle_rows, 0.25, 0.50)
            ),
            "candidate_best_iou_0p45_to_0p50_inclusive": _oracle_summary(
                _select_iou(candidate_oracle_rows, 0.45, 0.50)
            ),
            "b6_missed_gt_best_iou_0p25_to_0p50_inclusive": _oracle_summary(
                _select_iou(missed_oracle_rows, 0.25, 0.50)
            ),
            "b6_missed_gt_best_iou_0p40_to_0p50_inclusive": _oracle_summary(
                _select_iou(missed_oracle_rows, 0.40, 0.50)
            ),
        },
        "world_aabb_representation_ceiling": {
            "definition": (
                "exact maximum aligned-AABB IoU attainable by a centered "
                "world AABB under the scene's planar rigid axisAlignment"
            ),
            "all_ground_truth": _distribution(ceiling_values),
            "b6_missed_at_0p50": _distribution(missed_ceiling_values),
            "enclosing_target_roundtrip": {
                "definition": (
                    "aligned GT AABB -> inverse alignment -> enclosing world "
                    "AABB -> alignment -> aligned AABB"
                ),
                "all_ground_truth": _distribution(roundtrip_values),
                "b6_missed_at_0p50": _distribution(
                    missed_roundtrip_values
                ),
                "iou_divided_by_theoretical_ceiling": _quantiles(
                    ceiling_ratio
                ),
            },
        },
        "per_scene": per_scene,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recall-report", required=True, type=Path)
    parser.add_argument("--diagnostics-root", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--scans-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_report(
        recall_report=args.recall_report,
        diagnostics_root=args.diagnostics_root,
        prediction_root=args.prediction_root,
        gt_root=args.gt_root,
        scans_root=args.scans_root,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        os.replace(temporary, args.output)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
