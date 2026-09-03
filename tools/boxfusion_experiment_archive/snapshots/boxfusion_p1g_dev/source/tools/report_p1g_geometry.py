#!/usr/bin/env python3
"""Audit observer-only P1G geometry against its exact P1S parent stream.

The online P1G observer is not allowed to modify BoxFusion detections, add a
candidate, delete a candidate, change P1S objectness, or read ground truth.
This offline tool first validates that safety/lineage contract and only then
uses ScanNet ground truth to compare four class-agnostic candidate streams:

* frozen B6 predictions;
* the one-to-one P1S parent candidates;
* the one-to-one P1G refined candidates; and
* a diagnostic-only identity-vs-refined oracle which chooses one geometry for
  each parent using ground truth.

The oracle is deliberately labelled non-deployable.  It helps distinguish a
bad learned/internal geometry choice from a lack of useful association or
geometric evidence.  Prediction pickle files are trusted local experiment
artifacts and must never be loaded from an untrusted source.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.report_p1_residual_recall import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    center_size_to_corners,
    corners_to_minmax,
    load_axis_alignment,
    load_gt_boxes,
    load_predictions,
    pairwise_aabb_iou,
    read_scene_ids,
    score_ordered_match,
    transform_corners,
    validate_thresholds,
)


REPORT_SCHEMA = "boxfusion.p1g.geometry_report.v1"
P1G_DIAGNOSTIC_SCHEMA = "boxfusion.p1g.multiview_geometry_observer.v1"
P1G_PROFILE = "p1g_multiview_occupancy_msr_observer"
P1_DIAGNOSTIC_SCHEMA = "boxfusion.p1.residual_proposal_observer.v1"
P1S_PROFILE = "p1s_native_sparse_context_observer"
P1S_HEAD = "native_sparse_context_v1"
P1S_TARGET_SCOPE = "snapshot_inside_only"
MATCHING_CONTRACT = (
    "class-agnostic, stable score-descending, strict IoU > threshold, "
    "one-to-one per scene"
)
DIAGNOSIS_CLASSES = (
    "production_effective",
    "parameter_or_internal_gate_problem",
    "association_or_evidence_method_problem",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class P1GScene:
    """Validated one-to-one parent/refined rows for one scene."""

    parent_ids: np.ndarray
    refined_ids: np.ndarray
    parent_boxes: np.ndarray
    parent_corners: np.ndarray
    refined_boxes: np.ndarray
    refined_corners: np.ndarray
    scores: np.ndarray
    reasons: tuple[str, ...]
    runtime_seconds: float
    parent_runtime_seconds: float
    step_count: int


def _scalar(
    archive: Mapping[str, np.ndarray],
    key: str,
    path: Path,
) -> Any:
    if key not in archive:
        raise ValueError(f"{path}: missing {key}")
    value = np.asarray(archive[key])
    if value.shape != () or value.dtype.hasobject:
        raise ValueError(f"{path}: {key} must be a non-object scalar")
    result = value.item()
    return result.decode("utf-8") if isinstance(result, bytes) else result


def _text_scalar(
    archive: Mapping[str, np.ndarray],
    key: str,
    expected: str,
    path: Path,
) -> None:
    observed = _scalar(archive, key, path)
    if observed != expected:
        raise ValueError(
            f"{path}: {key}={observed!r}, expected {expected!r}"
        )


def _bool_scalar(
    archive: Mapping[str, np.ndarray],
    key: str,
    expected: bool,
    path: Path,
) -> None:
    value = np.asarray(archive.get(key))
    if value.shape != () or value.dtype != np.dtype(bool):
        raise ValueError(f"{path}: {key} must be a Boolean scalar")
    observed = bool(value.item())
    if observed is not expected:
        raise ValueError(
            f"{path}: unsafe {key}={observed}, expected {expected}"
        )


def _integer_scalar(
    archive: Mapping[str, np.ndarray],
    key: str,
    path: Path,
) -> int:
    value = np.asarray(archive.get(key))
    if value.shape != () or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{path}: {key} must be an integer scalar")
    return int(value.item())


def _float_scalar(
    archive: Mapping[str, np.ndarray],
    key: str,
    path: Path,
) -> float:
    value = np.asarray(archive.get(key))
    if value.shape != () or not np.issubdtype(value.dtype, np.floating):
        raise ValueError(f"{path}: {key} must be a floating scalar")
    result = float(value.item())
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{path}: {key} must be finite and non-negative")
    return result


def _string_vector(
    archive: Mapping[str, np.ndarray],
    key: str,
    rows: int,
    path: Path,
) -> np.ndarray:
    if key not in archive:
        raise ValueError(f"{path}: missing {key}")
    value = np.asarray(archive[key])
    if (
        value.shape != (rows,)
        or value.dtype.hasobject
        or value.dtype.kind not in {"U", "S"}
    ):
        raise ValueError(
            f"{path}: {key} must be a non-object string vector [{rows}]"
        )
    decoded = [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in value.tolist()
    ]
    if any(not item for item in decoded):
        raise ValueError(f"{path}: {key} contains an empty identifier")
    return np.asarray(decoded, dtype=np.str_)


def _float_vector(
    archive: Mapping[str, np.ndarray],
    key: str,
    rows: int,
    path: Path,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> np.ndarray:
    if key not in archive:
        raise ValueError(f"{path}: missing {key}")
    value = np.asarray(archive[key])
    if (
        value.shape != (rows,)
        or not np.issubdtype(value.dtype, np.floating)
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{path}: {key} must be finite floating [{rows}]")
    result = np.asarray(value, dtype=np.float64)
    if lower is not None and np.any(result < lower):
        raise ValueError(f"{path}: {key} contains values below {lower}")
    if upper is not None and np.any(result > upper):
        raise ValueError(f"{path}: {key} contains values above {upper}")
    return result


def _box_rows(
    archive: Mapping[str, np.ndarray],
    key: str,
    rows: int,
    path: Path,
) -> np.ndarray:
    if key not in archive:
        raise ValueError(f"{path}: missing {key}")
    value = np.asarray(archive[key])
    if (
        value.shape != (rows, 6)
        or not np.issubdtype(value.dtype, np.floating)
        or not np.isfinite(value).all()
        or (rows and np.any(value[:, 3:] <= 0.0))
    ):
        raise ValueError(f"{path}: {key} must be finite boxes [{rows},6]")
    return np.array(value, copy=True)


def _corner_rows(
    archive: Mapping[str, np.ndarray],
    key: str,
    rows: int,
    path: Path,
) -> np.ndarray:
    if key not in archive:
        raise ValueError(f"{path}: missing {key}")
    value = np.asarray(archive[key])
    if (
        value.shape != (rows, 8, 3)
        or not np.issubdtype(value.dtype, np.floating)
        or not np.isfinite(value).all()
    ):
        raise ValueError(
            f"{path}: {key} must be finite corners [{rows},8,3]"
        )
    return np.array(value, copy=True)


def _box_minmax(boxes: np.ndarray) -> np.ndarray:
    values = np.asarray(boxes, dtype=np.float64)
    return np.concatenate(
        (values[:, :3] - 0.5 * values[:, 3:],
         values[:, :3] + 0.5 * values[:, 3:]),
        axis=1,
    )


def _corners_agree_with_boxes(
    boxes: np.ndarray,
    corners: np.ndarray,
    *,
    key: str,
    path: Path,
) -> None:
    expected = _box_minmax(boxes)
    observed = corners_to_minmax(np.asarray(corners, dtype=np.float64))
    if not np.allclose(expected, observed, rtol=1e-5, atol=1e-5):
        raise ValueError(f"{path}: {key} box/corner aliases disagree")


def _validate_config(archive: Mapping[str, np.ndarray], path: Path) -> None:
    raw = _scalar(archive, "p1g_config_json", path)
    if not isinstance(raw, str):
        raise ValueError(f"{path}: p1g_config_json must be text")
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: malformed p1g_config_json") from error
    if not isinstance(config, Mapping):
        raise ValueError(f"{path}: p1g_config_json must encode a mapping")
    for key, expected in (
        ("enabled", True),
        ("observer_only", True),
        ("mutate", False),
    ):
        if key not in config:
            raise ValueError(f"{path}: p1g_config_json is missing {key}")
        if config[key] is not expected:
            raise ValueError(f"{path}: unsafe p1g_config_json.{key}")


def _bool_vector(
    archive: Mapping[str, np.ndarray],
    key: str,
    rows: int,
    path: Path,
) -> np.ndarray:
    value = np.asarray(archive.get(key))
    if value.shape != (rows,) or value.dtype != np.dtype(bool):
        raise ValueError(f"{path}: {key} must be Boolean [{rows}]")
    return np.array(value, copy=True)


def _integer_vector(
    archive: Mapping[str, np.ndarray],
    key: str,
    rows: int,
    path: Path,
    *,
    lower: int | None = None,
) -> np.ndarray:
    value = np.asarray(archive.get(key))
    if value.shape != (rows,) or not np.issubdtype(
        value.dtype, np.integer
    ):
        raise ValueError(f"{path}: {key} must be integer [{rows}]")
    result = np.asarray(value, dtype=np.int64)
    if lower is not None and np.any(result < lower):
        raise ValueError(f"{path}: {key} contains values below {lower}")
    return result


def _validate_evidence_rows(
    archive: Mapping[str, np.ndarray],
    *,
    rows: int,
    path: Path,
) -> None:
    """Validate row-aligned evidence used to diagnose P1G failures."""

    _bool_vector(archive, "p1g_is_candidate", rows, path)
    _string_vector(archive, "p1g_sources", rows, path)
    matched = _integer_vector(
        archive, "p1g_matched_view_counts", rows, path, lower=0
    )
    selected = _integer_vector(
        archive, "p1g_selected_view_counts", rows, path, lower=0
    )
    if np.any(selected > matched):
        raise ValueError(
            f"{path}: selected P1G views exceed matched views"
        )
    frame_ids = np.asarray(archive.get("p1g_selected_frame_ids"))
    if (
        frame_ids.ndim != 2
        or frame_ids.shape[0] != rows
        or not np.issubdtype(frame_ids.dtype, np.integer)
    ):
        raise ValueError(
            f"{path}: p1g_selected_frame_ids must be integer [N,K]"
        )
    for row_index, expected_count in enumerate(selected.tolist()):
        row = np.asarray(frame_ids[row_index], dtype=np.int64)
        observed = row[row >= 0]
        padding = row[row < 0]
        if (
            len(observed) != expected_count
            or len(np.unique(observed)) != len(observed)
            or np.any(padding != -1)
        ):
            raise ValueError(
                f"{path}: selected frame IDs disagree at row {row_index}"
            )
        if len(observed) and not np.all(row[: len(observed)] >= 0):
            raise ValueError(
                f"{path}: selected frame IDs are not prefix-packed"
            )
    _integer_vector(
        archive, "p1g_cropped_point_counts", rows, path, lower=0
    )
    for key in (
        "p1g_face_residuals",
        "p1g_face_support",
        "p1g_face_uncertainty",
    ):
        value = np.asarray(archive.get(key))
        if (
            value.shape != (rows, 3, 2)
            or not np.issubdtype(value.dtype, np.floating)
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"{path}: {key} must be finite [N,3,2]")
    supported = np.asarray(archive.get("p1g_face_supported"))
    if supported.shape != (rows, 3, 2) or supported.dtype != np.dtype(bool):
        raise ValueError(
            f"{path}: p1g_face_supported must be Boolean [N,3,2]"
        )
    features = np.asarray(archive.get("p1g_feature_vectors"))
    if (
        features.shape != (rows, 48)
        or not np.issubdtype(features.dtype, np.floating)
        or not np.isfinite(features).all()
    ):
        raise ValueError(
            f"{path}: p1g_feature_vectors must be finite [N,48]"
        )


def _parent_runtime(
    archive: Mapping[str, np.ndarray],
    path: Path,
) -> float:
    total = 0.0
    present = False
    lengths: set[int] = set()
    for key in (
        "p1_step_voxelize_seconds",
        "p1_step_head_seconds",
        "p1_step_nms_seconds",
    ):
        if key not in archive:
            continue
        values = np.asarray(archive[key])
        if (
            values.ndim != 1
            or not np.issubdtype(values.dtype, np.floating)
            or not np.isfinite(values).all()
            or np.any(values < 0.0)
        ):
            raise ValueError(f"{path}: invalid {key}")
        present = True
        lengths.add(len(values))
        total += float(np.sum(values, dtype=np.float64))
    if not present:
        raise ValueError(f"{path}: missing P1S parent runtime arrays")
    if len(lengths) != 1:
        raise ValueError(f"{path}: P1S parent runtime arrays disagree")
    return total


def load_p1g_scene(path: Path, *, scene_id: str) -> P1GScene:
    """Load and fail-closed validate one P1S->P1G diagnostic archive."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as source:
        archive = {
            key: np.array(source[key], copy=True) for key in source.files
        }

    for key, expected in {
        "scene_id": scene_id,
        "p1_schema": P1_DIAGNOSTIC_SCHEMA,
        "p1_stage": "P1S",
        "p1_profile": P1S_PROFILE,
        "p1_head_architecture": P1S_HEAD,
        "p1_target_assignment_scope": P1S_TARGET_SCOPE,
        "p1g_schema": P1G_DIAGNOSTIC_SCHEMA,
        "p1g_stage": "P1G",
        "p1g_profile": P1G_PROFILE,
        "p1g_parent_stage": "P1S",
    }.items():
        _text_scalar(archive, key, expected, path)
    for key, expected in {
        "p1_enabled": True,
        "p1_observer_only": True,
        "p1_uses_ground_truth": False,
        "p1_reads_semantic_labels": False,
        "p1_mutation_enabled": False,
        "p1_complete": True,
        "p1_class_agnostic": True,
        "p1g_enabled": True,
        "p1g_observer_only": True,
        "p1g_uses_ground_truth": False,
        "p1g_reads_semantic_labels": False,
        "p1g_mutation_enabled": False,
        "p1g_complete": True,
        "p1g_class_agnostic": True,
    }.items():
        _bool_scalar(archive, key, expected, path)
    if _integer_scalar(archive, "p1_applied_count", path) != 0:
        raise ValueError(f"{path}: P1S parent applied formal output")
    if _integer_scalar(archive, "p1g_applied_count", path) != 0:
        raise ValueError(f"{path}: P1G observer applied formal output")
    if _integer_scalar(archive, "p1g_regression_dim", path) != 6:
        raise ValueError(f"{path}: P1G regression must remain six-dimensional")
    if _integer_scalar(archive, "p1g_failure_count", path) != 0:
        raise ValueError(f"{path}: P1G observer contains failed operations")
    checkpoint = _scalar(archive, "p1g_parent_checkpoint_sha256", path)
    if not isinstance(checkpoint, str) or _SHA256.fullmatch(
        checkpoint.lower()
    ) is None:
        raise ValueError(f"{path}: invalid P1G parent checkpoint SHA")
    _validate_config(archive, path)

    if "p1_candidate_ids" not in archive:
        raise ValueError(f"{path}: missing p1_candidate_ids")
    p1_ids_raw = np.asarray(archive["p1_candidate_ids"])
    if (
        p1_ids_raw.ndim != 1
        or p1_ids_raw.dtype.hasobject
        or p1_ids_raw.dtype.kind not in {"U", "S"}
    ):
        raise ValueError(f"{path}: p1_candidate_ids must be strings")
    p1_ids = np.asarray(
        [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in p1_ids_raw.tolist()
        ],
        dtype=np.str_,
    )
    rows = len(p1_ids)
    if len(np.unique(p1_ids)) != rows:
        raise ValueError(f"{path}: p1_candidate_ids are not unique")
    p1_boxes = _box_rows(archive, "p1_candidate_boxes", rows, path)
    p1_corners = _corner_rows(archive, "p1_candidate_corners", rows, path)
    p1_scores = _float_vector(
        archive,
        "p1_candidate_scores",
        rows,
        path,
        lower=0.0,
        upper=1.0,
    )
    _corners_agree_with_boxes(
        p1_boxes, p1_corners, key="P1S parent", path=path
    )

    parent_ids = _string_vector(
        archive, "p1g_parent_candidate_ids", rows, path
    )
    refined_ids = _string_vector(
        archive, "p1g_refined_candidate_ids", rows, path
    )
    if len(np.unique(refined_ids)) != rows:
        raise ValueError(f"{path}: refined candidate IDs are not unique")
    if not np.array_equal(parent_ids, p1_ids):
        raise ValueError(
            f"{path}: P1G parent IDs/order disagree with exact P1S stream"
        )
    parent_boxes = _box_rows(
        archive, "p1g_parent_boxes", rows, path
    )
    parent_corners = _corner_rows(
        archive, "p1g_parent_corners", rows, path
    )
    refined_boxes = _box_rows(
        archive, "p1g_refined_boxes", rows, path
    )
    refined_corners = _corner_rows(
        archive, "p1g_refined_corners", rows, path
    )
    scores = _float_vector(
        archive,
        "p1g_candidate_scores",
        rows,
        path,
        lower=0.0,
        upper=1.0,
    )
    if not np.array_equal(parent_boxes, p1_boxes):
        raise ValueError(f"{path}: P1G parent boxes changed P1S geometry")
    if not np.array_equal(parent_corners, p1_corners):
        raise ValueError(f"{path}: P1G parent corners changed P1S geometry")
    if not np.array_equal(scores, p1_scores):
        raise ValueError(f"{path}: P1G changed P1S candidate scores/order")
    _corners_agree_with_boxes(
        parent_boxes, parent_corners, key="P1G parent", path=path
    )
    _corners_agree_with_boxes(
        refined_boxes, refined_corners, key="P1G refined", path=path
    )

    applied = np.asarray(archive.get("p1g_candidate_applied"))
    if applied.shape != (rows,) or applied.dtype != np.dtype(bool):
        raise ValueError(
            f"{path}: p1g_candidate_applied must be Boolean [{rows}]"
        )
    if bool(np.any(applied)):
        raise ValueError(f"{path}: P1G candidate rows mutated formal output")
    reasons = _string_vector(archive, "p1g_reasons", rows, path)
    _validate_evidence_rows(archive, rows=rows, path=path)

    runtime = _float_scalar(archive, "p1g_runtime_seconds", path)
    step_seconds = np.asarray(archive.get("p1g_step_total_seconds"))
    if (
        step_seconds.shape != (rows,)
        or not np.issubdtype(step_seconds.dtype, np.floating)
        or not np.isfinite(step_seconds).all()
        or np.any(step_seconds < 0.0)
    ):
        raise ValueError(
            f"{path}: p1g_step_total_seconds must be finite [{rows}]"
        )
    step_total = float(np.sum(step_seconds, dtype=np.float64))
    if step_total > runtime + max(1e-9, 1e-6 * max(runtime, 1.0)):
        raise ValueError(
            f"{path}: P1G step runtime exceeds scene runtime scalar"
        )
    if "p1g_candidate_seconds" in archive:
        candidate_seconds = _float_vector(
            archive, "p1g_candidate_seconds", rows, path, lower=0.0
        )
        if float(candidate_seconds.sum()) > runtime + max(
            1e-9, 1e-6 * max(runtime, 1.0)
        ):
            raise ValueError(
                f"{path}: P1G candidate runtime exceeds scene runtime"
            )

    return P1GScene(
        parent_ids=parent_ids,
        refined_ids=refined_ids,
        parent_boxes=parent_boxes,
        parent_corners=parent_corners,
        refined_boxes=refined_boxes,
        refined_corners=refined_corners,
        scores=scores,
        reasons=tuple(str(value) for value in reasons.tolist()),
        runtime_seconds=runtime,
        parent_runtime_seconds=_parent_runtime(archive, path),
        step_count=len(step_seconds),
    )


def _stream_metrics(
    *,
    baseline_iou: np.ndarray,
    baseline_scores: np.ndarray,
    candidate_iou: np.ndarray,
    candidate_scores: np.ndarray,
    candidate_ids: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    baseline = score_ordered_match(
        baseline_iou, baseline_scores, threshold
    )
    candidate = score_ordered_match(
        candidate_iou,
        candidate_scores,
        threshold,
        tie_break_ids=candidate_ids,
    )
    covered = np.zeros(baseline_iou.shape[1], dtype=bool)
    covered[baseline.matched_gt] = True
    novel = score_ordered_match(
        candidate_iou,
        candidate_scores,
        threshold,
        allowed_gt=~covered,
        tie_break_ids=candidate_ids,
    )
    union_iou = np.concatenate((baseline_iou, candidate_iou), axis=0)
    union_scores = np.concatenate((baseline_scores, candidate_scores))
    union = score_ordered_match(union_iou, union_scores, threshold)
    return {
        "candidate_true_positives": candidate.true_positive_count,
        "novel_true_positives": novel.true_positive_count,
        "union_true_positives": union.true_positive_count,
    }


def _crossing(before: float, after: float, threshold: float) -> str:
    was_above = before > threshold
    now_above = after > threshold
    if not was_above and now_above:
        return "up"
    if was_above and not now_above:
        return "down"
    return "above" if was_above else "below"


def _finite_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {
            "count": 0,
            "minimum": None,
            "q25": None,
            "median": None,
            "mean": None,
            "q75": None,
            "maximum": None,
        }
    if not np.isfinite(array).all():
        raise ValueError("internal geometry summary received non-finite values")
    quantiles = np.quantile(array, (0.0, 0.25, 0.5, 0.75, 1.0))
    return {
        "count": int(len(array)),
        "minimum": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "mean": float(np.mean(array)),
        "q75": float(quantiles[3]),
        "maximum": float(quantiles[4]),
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.quantile(array, quantile)) if len(array) else 0.0


def _exact_scene_set(
    root: Path,
    expected: set[str],
    *,
    suffix: str,
    role: str,
) -> None:
    actual = {
        path.name[: -len(suffix)]
        for path in root.glob(f"scene*{suffix}")
        if path.is_file()
    }
    if actual != expected:
        raise ValueError(
            f"{role} scene set mismatch: "
            f"missing={sorted(expected-actual)[:8]}, "
            f"extra={sorted(actual-expected)[:8]}"
        )


def build_report(
    *,
    scene_list: Path,
    prediction_root: Path,
    diagnostics_root: Path,
    gt_root: Path,
    scans_root: Path,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    minimum_novel_tp50: int = 5,
    minimum_parent_tp25_delta: int = 0,
    maximum_p1g_runtime_seconds_per_scene: float = 0.18,
    maximum_total_runtime_seconds_per_scene: float = 0.80,
    maximum_candidates_per_scene: float = 256.0,
) -> dict[str, Any]:
    """Build a strict P1G report and a frozen production/diagnosis decision."""

    thresholds = validate_thresholds(thresholds)
    threshold_keys = {f"{value:.2f}" for value in thresholds}
    if not {"0.15", "0.25", "0.50"}.issubset(threshold_keys):
        raise ValueError("P1G report requires thresholds 0.15, 0.25, 0.50")
    if (
        isinstance(minimum_novel_tp50, bool)
        or int(minimum_novel_tp50) != minimum_novel_tp50
        or int(minimum_novel_tp50) < 1
    ):
        raise ValueError("minimum_novel_tp50 must be a positive integer")
    if (
        isinstance(minimum_parent_tp25_delta, bool)
        or int(minimum_parent_tp25_delta) != minimum_parent_tp25_delta
    ):
        raise ValueError("minimum_parent_tp25_delta must be an integer")
    for value, name in (
        (
            maximum_p1g_runtime_seconds_per_scene,
            "maximum_p1g_runtime_seconds_per_scene",
        ),
        (
            maximum_total_runtime_seconds_per_scene,
            "maximum_total_runtime_seconds_per_scene",
        ),
        (maximum_candidates_per_scene, "maximum_candidates_per_scene"),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    scenes = read_scene_ids(scene_list)
    prediction_root = Path(prediction_root)
    diagnostics_root = Path(diagnostics_root)
    gt_root = Path(gt_root)
    scans_root = Path(scans_root)
    for root, role in (
        (prediction_root, "prediction"),
        (diagnostics_root, "diagnostic"),
        (gt_root, "ground-truth"),
        (scans_root, "scan"),
    ):
        if not root.is_dir():
            raise FileNotFoundError(f"{role} root not found: {root}")
    expected = set(scenes)
    _exact_scene_set(
        prediction_root,
        expected,
        suffix="_boxes.pkl",
        role="prediction",
    )
    _exact_scene_set(
        diagnostics_root,
        expected,
        suffix="_tracks.npz",
        role="diagnostic",
    )

    source_names = ("b6", "parent", "refined", "oracle")
    totals = {
        f"{threshold:.2f}": {
            source: {
                "candidate_true_positives": 0,
                "novel_true_positives": 0,
                "union_true_positives": 0,
            }
            for source in source_names
        }
        for threshold in thresholds
    }
    # B6 is not a candidate stream; its three aliases are set explicitly.
    total_gt = 0
    total_baseline = 0
    total_candidates = 0
    p1g_scene_seconds: list[float] = []
    total_scene_seconds: list[float] = []
    same_gt_deltas: list[float] = []
    best_gt_deltas: list[float] = []
    relevant_015: list[float] = []
    relevant_025: list[float] = []
    crossings = {
        f"{threshold:.2f}": {
            state: 0 for state in ("up", "down", "above", "below")
        }
        for threshold in thresholds
    }
    improved = harmed = unchanged = severe_harm = switched_gt = 0
    oracle_refined = 0
    reason_histogram: dict[str, int] = {}
    per_scene: dict[str, Any] = {}

    for scene in scenes:
        prediction = load_predictions(
            prediction_root / f"{scene}_boxes.pkl"
        )
        p1g = load_p1g_scene(
            diagnostics_root / f"{scene}_tracks.npz",
            scene_id=scene,
        )
        alignment = load_axis_alignment(scans_root, scene)
        baseline_boxes = corners_to_minmax(
            transform_corners(prediction.corners_world, alignment)
        )
        parent_boxes = corners_to_minmax(
            transform_corners(p1g.parent_corners, alignment)
        )
        refined_boxes = corners_to_minmax(
            transform_corners(p1g.refined_corners, alignment)
        )
        gt_boxes = load_gt_boxes(gt_root / f"{scene}_bbox.npy")
        baseline_iou = pairwise_aabb_iou(baseline_boxes, gt_boxes)
        parent_iou = pairwise_aabb_iou(parent_boxes, gt_boxes)
        refined_iou = pairwise_aabb_iou(refined_boxes, gt_boxes)

        if len(gt_boxes):
            parent_best_gt = np.argmax(parent_iou, axis=1)
            parent_best = parent_iou[
                np.arange(len(parent_iou)), parent_best_gt
            ]
            refined_best_gt = np.argmax(refined_iou, axis=1)
            refined_best = refined_iou[
                np.arange(len(refined_iou)), refined_best_gt
            ]
            refined_same = refined_iou[
                np.arange(len(refined_iou)), parent_best_gt
            ]
        else:
            parent_best_gt = np.full(len(parent_iou), -1, dtype=np.int64)
            refined_best_gt = np.full(len(refined_iou), -1, dtype=np.int64)
            parent_best = np.zeros(len(parent_iou), dtype=np.float64)
            refined_best = np.zeros(len(refined_iou), dtype=np.float64)
            refined_same = np.zeros(len(refined_iou), dtype=np.float64)
        same_delta = refined_same - parent_best
        best_delta = refined_best - parent_best
        same_gt_deltas.extend(same_delta.tolist())
        best_gt_deltas.extend(best_delta.tolist())
        relevant_015.extend(same_delta[parent_best > 0.15].tolist())
        relevant_025.extend(same_delta[parent_best > 0.25].tolist())
        improved += int(np.sum(same_delta > 1e-12))
        harmed += int(np.sum(same_delta < -1e-12))
        unchanged += int(np.sum(np.abs(same_delta) <= 1e-12))
        severe_harm += int(np.sum(same_delta <= -0.05))
        switched_gt += int(
            np.sum(
                (parent_best_gt != refined_best_gt)
                & (parent_best_gt >= 0)
            )
        )
        for threshold in thresholds:
            key = f"{threshold:.2f}"
            for before, after in zip(parent_best, refined_same):
                crossings[key][_crossing(
                    float(before), float(after), threshold
                )] += 1

        choose_refined = refined_best > parent_best
        oracle_refined += int(np.sum(choose_refined))
        oracle_boxes = np.where(
            choose_refined[:, None], refined_boxes, parent_boxes
        )
        oracle_iou = pairwise_aabb_iou(oracle_boxes, gt_boxes)

        scene_thresholds: dict[str, Any] = {}
        for threshold in thresholds:
            key = f"{threshold:.2f}"
            baseline_match = score_ordered_match(
                baseline_iou, prediction.scores, threshold
            )
            totals[key]["b6"] = {
                "candidate_true_positives": (
                    totals[key]["b6"]["candidate_true_positives"]
                    + baseline_match.true_positive_count
                ),
                "novel_true_positives": (
                    totals[key]["b6"]["novel_true_positives"]
                    + baseline_match.true_positive_count
                ),
                "union_true_positives": (
                    totals[key]["b6"]["union_true_positives"]
                    + baseline_match.true_positive_count
                ),
            }
            scene_thresholds[key] = {
                "b6_true_positives": baseline_match.true_positive_count
            }
            for source, candidate_iou, ids in (
                ("parent", parent_iou, p1g.parent_ids),
                ("refined", refined_iou, p1g.refined_ids),
                ("oracle", oracle_iou, p1g.parent_ids),
            ):
                row = _stream_metrics(
                    baseline_iou=baseline_iou,
                    baseline_scores=prediction.scores,
                    candidate_iou=candidate_iou,
                    candidate_scores=p1g.scores,
                    candidate_ids=ids,
                    threshold=threshold,
                )
                for field, value in row.items():
                    totals[key][source][field] += int(value)
                scene_thresholds[key][source] = row

        for reason in p1g.reasons:
            reason_histogram[reason] = reason_histogram.get(reason, 0) + 1
        scene_geometry = {
            "same_gt_iou_delta": _finite_summary(same_delta.tolist()),
            "best_gt_iou_delta": _finite_summary(best_delta.tolist()),
            "improved": int(np.sum(same_delta > 1e-12)),
            "harmed": int(np.sum(same_delta < -1e-12)),
            "severe_harm_le_minus_0p05": int(
                np.sum(same_delta <= -0.05)
            ),
        }
        per_scene[scene] = {
            "ground_truth_count": int(len(gt_boxes)),
            "baseline_predictions": int(len(baseline_boxes)),
            "candidate_count": int(len(parent_boxes)),
            "p1g_runtime_seconds": p1g.runtime_seconds,
            "p1s_parent_runtime_seconds": p1g.parent_runtime_seconds,
            "thresholds": scene_thresholds,
            "geometry": scene_geometry,
        }
        total_gt += len(gt_boxes)
        total_baseline += len(baseline_boxes)
        total_candidates += len(parent_boxes)
        p1g_scene_seconds.append(p1g.runtime_seconds)
        total_scene_seconds.append(
            p1g.runtime_seconds + p1g.parent_runtime_seconds
        )

    threshold_report: dict[str, Any] = {}
    for key, sources in totals.items():
        threshold_report[key] = {}
        for source, row in sources.items():
            count = total_baseline if source == "b6" else total_candidates
            threshold_report[key][source] = {
                **{field: int(value) for field, value in row.items()},
                "candidate_count": int(count),
                "candidate_recall": float(
                    row["candidate_true_positives"] / max(total_gt, 1)
                ),
                "novel_recall_gain": float(
                    row["novel_true_positives"] / max(total_gt, 1)
                ),
                "union_recall": float(
                    row["union_true_positives"] / max(total_gt, 1)
                ),
            }
        parent = threshold_report[key]["parent"]
        refined = threshold_report[key]["refined"]
        oracle = threshold_report[key]["oracle"]
        threshold_report[key]["refined_minus_parent"] = {
            field: int(refined[field] - parent[field])
            for field in (
                "candidate_true_positives",
                "novel_true_positives",
                "union_true_positives",
            )
        }
        threshold_report[key]["oracle_minus_refined"] = {
            field: int(oracle[field] - refined[field])
            for field in (
                "candidate_true_positives",
                "novel_true_positives",
                "union_true_positives",
            )
        }

    parent25 = threshold_report["0.25"]["parent"][
        "novel_true_positives"
    ]
    refined25 = threshold_report["0.25"]["refined"][
        "novel_true_positives"
    ]
    refined50 = threshold_report["0.50"]["refined"][
        "novel_true_positives"
    ]
    oracle25 = threshold_report["0.25"]["oracle"][
        "novel_true_positives"
    ]
    oracle50 = threshold_report["0.50"]["oracle"][
        "novel_true_positives"
    ]
    mean_p1g = float(
        np.mean(p1g_scene_seconds) if p1g_scene_seconds else 0.0
    )
    mean_total = float(
        np.mean(total_scene_seconds) if total_scene_seconds else 0.0
    )
    candidates_per_scene = float(
        total_candidates / max(len(scenes), 1)
    )
    geometry_checks = {
        "refined_novel_tp50_ge_minimum": bool(
            refined50 >= int(minimum_novel_tp50)
        ),
        "refined_novel_tp25_noninferior_to_parent": bool(
            refined25
            >= parent25 + int(minimum_parent_tp25_delta)
        ),
    }
    operational_checks = {
        "p1g_runtime_passes": bool(
            mean_p1g <= maximum_p1g_runtime_seconds_per_scene
        ),
        "p1s_plus_p1g_runtime_passes": bool(
            mean_total <= maximum_total_runtime_seconds_per_scene
        ),
        "candidate_bound_passes": bool(
            candidates_per_scene <= maximum_candidates_per_scene
        ),
    }
    production_passes = bool(
        all(geometry_checks.values()) and all(operational_checks.values())
    )
    oracle_geometry_passes = bool(
        oracle50 >= int(minimum_novel_tp50)
        and oracle25
        >= parent25 + int(minimum_parent_tp25_delta)
    )
    if production_passes:
        diagnosis = "production_effective"
    elif oracle_geometry_passes:
        diagnosis = "parameter_or_internal_gate_problem"
    else:
        diagnosis = "association_or_evidence_method_problem"
    assert diagnosis in DIAGNOSIS_CLASSES

    geometry_rows = len(same_gt_deltas)
    report = {
        "schema": REPORT_SCHEMA,
        "diagnostic_schema": P1G_DIAGNOSTIC_SCHEMA,
        "stage": "P1G",
        "profile": P1G_PROFILE,
        "matching_contract": MATCHING_CONTRACT,
        "observer_only": True,
        "scene_count": int(len(scenes)),
        "ground_truth_count": int(total_gt),
        "baseline_prediction_count": int(total_baseline),
        "candidate_count": int(total_candidates),
        "candidates_per_scene": candidates_per_scene,
        "thresholds": threshold_report,
        "geometry": {
            "same_parent_best_gt_iou_delta": _finite_summary(
                same_gt_deltas
            ),
            "best_gt_iou_delta": _finite_summary(best_gt_deltas),
            "same_gt_delta_parent_iou_gt_0p15": _finite_summary(
                relevant_015
            ),
            "same_gt_delta_parent_iou_gt_0p25": _finite_summary(
                relevant_025
            ),
            "improved": int(improved),
            "harmed": int(harmed),
            "unchanged": int(unchanged),
            "severe_harm_le_minus_0p05": int(severe_harm),
            "harm_rate": float(harmed / max(geometry_rows, 1)),
            "severe_harm_rate": float(
                severe_harm / max(geometry_rows, 1)
            ),
            "best_gt_switched": int(switched_gt),
            "threshold_crossings": crossings,
        },
        "identity_vs_refined_oracle": {
            "uses_ground_truth": True,
            "deployable": False,
            "selection_rule": (
                "per parent choose refined only when its best-GT IoU is "
                "strictly greater; ties keep identity"
            ),
            "selected_identity": int(total_candidates - oracle_refined),
            "selected_refined": int(oracle_refined),
            "geometry_gate_passes": oracle_geometry_passes,
        },
        "runtime": {
            "p1g_seconds": float(sum(p1g_scene_seconds)),
            "p1g_mean_seconds_per_scene": mean_p1g,
            "p1g_p95_seconds_per_scene": _percentile(
                p1g_scene_seconds, 0.95
            ),
            "p1s_plus_p1g_mean_seconds_per_scene": mean_total,
            "p1s_plus_p1g_p95_seconds_per_scene": _percentile(
                total_scene_seconds, 0.95
            ),
        },
        "reason_histogram": dict(sorted(reason_histogram.items())),
        "go_no_go": {
            "minimum_novel_tp50": int(minimum_novel_tp50),
            "minimum_parent_tp25_delta": int(
                minimum_parent_tp25_delta
            ),
            "maximum_p1g_runtime_seconds_per_scene": float(
                maximum_p1g_runtime_seconds_per_scene
            ),
            "maximum_total_runtime_seconds_per_scene": float(
                maximum_total_runtime_seconds_per_scene
            ),
            "maximum_candidates_per_scene": float(
                maximum_candidates_per_scene
            ),
            **geometry_checks,
            **operational_checks,
            "passes": production_passes,
            "decision": (
                "GO_DESIGN_P1Q" if production_passes else "STOP_P1G"
            ),
        },
        "diagnosis": {
            "classification": diagnosis,
            "production_geometry_passes": bool(
                all(geometry_checks.values())
            ),
            "production_operational_passes": bool(
                all(operational_checks.values())
            ),
            "oracle_geometry_passes": oracle_geometry_passes,
            "interpretation": {
                "production_effective": (
                    "The frozen refined-only stream passes geometry and "
                    "online-budget gates."
                ),
                "parameter_or_internal_gate_problem": (
                    "Identity/refined evidence contains enough GT-only "
                    "potential, but the frozen production choice or runtime "
                    "does not realize it."
                ),
                "association_or_evidence_method_problem": (
                    "Even the per-parent GT-only identity/refined selector "
                    "cannot meet the frozen geometry gate; threshold tuning "
                    "alone is not supported."
                ),
            }[diagnosis],
        },
        "per_scene": per_scene,
    }
    # Enforce strict RFC-compliant JSON before returning to callers.
    json.dumps(report, allow_nan=False)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--diagnostics-root", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--scans-root", required=True, type=Path)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
    )
    parser.add_argument("--minimum-novel-tp50", type=int, default=5)
    parser.add_argument(
        "--minimum-parent-tp25-delta", type=int, default=0
    )
    parser.add_argument(
        "--maximum-p1g-runtime-seconds-per-scene",
        type=float,
        default=0.18,
    )
    parser.add_argument(
        "--maximum-total-runtime-seconds-per-scene",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--maximum-candidates-per-scene", type=float, default=256.0
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_report(
        scene_list=args.scene_list,
        prediction_root=args.prediction_root,
        diagnostics_root=args.diagnostics_root,
        gt_root=args.gt_root,
        scans_root=args.scans_root,
        thresholds=args.thresholds,
        minimum_novel_tp50=args.minimum_novel_tp50,
        minimum_parent_tp25_delta=args.minimum_parent_tp25_delta,
        maximum_p1g_runtime_seconds_per_scene=(
            args.maximum_p1g_runtime_seconds_per_scene
        ),
        maximum_total_runtime_seconds_per_scene=(
            args.maximum_total_runtime_seconds_per_scene
        ),
        maximum_candidates_per_scene=args.maximum_candidates_per_scene,
    )
    rendered = json.dumps(
        report, indent=2, sort_keys=True, allow_nan=False
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(rendered + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(rendered)
    return 0 if report["go_no_go"]["passes"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
