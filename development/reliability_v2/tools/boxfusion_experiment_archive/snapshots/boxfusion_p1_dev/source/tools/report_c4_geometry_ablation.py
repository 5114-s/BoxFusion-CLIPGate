#!/usr/bin/env python3
"""Offline report for the C4 generic SAM3 Mask-RGBD observer.

The C4 runtime is deliberately observer-only: its verified candidate boxes are
stored in ``*_tracks.npz`` but never exported.  This tool measures proposal
geometry against ScanNet ground truth and simulates replacing *only* verified
boxes while retaining the exact B6 scores and prediction order.  Schema v2
uses the exported oriented corners directly, preserving yaw through ScanNet
axis alignment; six-dimensional center/size rows are diagnostics-only.

Prediction pickle files are trusted local BoxFusion experiment artifacts.  Do
not run this program on untrusted pickle files.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.analyze_fused_oracle import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    center_size_to_minmax,
    corners_to_minmax,
    load_axis_alignment,
    load_scene_predictions,
    pairwise_aabb_iou,
    ranked_metrics,
    read_scene_ids,
    score_scene,
    transform_corners,
)


PREDICTION_SUFFIX = "_boxes.pkl"
DIAGNOSTIC_SUFFIX = "_tracks.npz"
C4_DIAGNOSTIC_SCHEMA = "generic_mask_rgbd_local_geometry_v2"
REPORT_SCHEMA = "boxfusion.c4_geometry_ablation_report"
REPORT_FORMAT_VERSION = 2
THRESHOLDS = tuple(float(value) for value in DEFAULT_THRESHOLDS)


@dataclass(frozen=True)
class C4SceneDiagnostics:
    """Strict, pickle-free C4 observer rows for one scene."""

    result_indices: np.ndarray
    stable_ids: np.ndarray
    scores: np.ndarray
    labels: tuple[str, ...]
    normalized_labels: tuple[str, ...]
    reasons: tuple[str, ...]
    sources: tuple[str, ...]
    attempted: np.ndarray
    proposed: np.ndarray
    verified: np.ndarray
    applied: np.ndarray
    original_boxes: np.ndarray
    candidate_boxes: np.ndarray
    original_corners: np.ndarray
    candidate_corners: np.ndarray
    failed_open: bool
    error: str


def _threshold_key(threshold: float) -> str:
    return f"{float(threshold):.2f}"


def _scalar_text(value: np.ndarray, *, name: str, path: Path) -> str:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.hasobject:
        raise ValueError(f"{path}: {name} must be a non-object scalar string")
    scalar = array.item()
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    if not isinstance(scalar, str):
        raise ValueError(f"{path}: {name} must be a string")
    return scalar


def _scalar_boolean(value: np.ndarray, *, name: str, path: Path) -> bool:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype != np.bool_:
        raise ValueError(f"{path}: {name} must be a Boolean scalar")
    return bool(array.item())


def _string_rows(
    value: np.ndarray,
    *,
    name: str,
    rows: int,
    path: Path,
) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.shape != (rows,) or array.dtype.hasobject:
        raise ValueError(f"{path}: {name} must have non-object shape [{rows}]")
    output: list[str] = []
    for scalar in array.tolist():
        if isinstance(scalar, bytes):
            scalar = scalar.decode("utf-8")
        if not isinstance(scalar, str):
            raise ValueError(f"{path}: {name} entries must be strings")
        output.append(scalar)
    return tuple(output)


def _boolean_rows(
    value: np.ndarray,
    *,
    name: str,
    rows: int,
    path: Path,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (rows,) or array.dtype != np.bool_:
        raise ValueError(f"{path}: {name} must have Boolean shape [{rows}]")
    return np.array(array, dtype=bool, copy=True)


def load_c4_diagnostics(
    path: str | Path,
    *,
    expected_scene_id: str,
) -> C4SceneDiagnostics:
    """Load and validate the immutable C4 observer diagnostic contract."""

    diagnostic_path = Path(path)
    if not diagnostic_path.is_file():
        raise FileNotFoundError(diagnostic_path)
    required = {
        "scene_id",
        "c4_diagnostics_schema",
        "c4_enabled",
        "c4_mutation_enabled",
        "c4_fail_open",
        "c4_error",
        "c4_result_indices",
        "c4_stable_ids",
        "c4_scores",
        "c4_attempted",
        "c4_proposed",
        "c4_verified",
        "c4_applied",
        "c4_reason",
        "c4_source",
        "c4_label",
        "c4_normalized_label",
        "c4_original_boxes",
        "c4_candidate_boxes",
        "c4_original_corners",
        "c4_candidate_corners",
    }
    with np.load(diagnostic_path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(
                f"{diagnostic_path}: missing C4 fields {sorted(missing)}"
            )
        scene_id = _scalar_text(
            archive["scene_id"], name="scene_id", path=diagnostic_path
        )
        schema = _scalar_text(
            archive["c4_diagnostics_schema"],
            name="c4_diagnostics_schema",
            path=diagnostic_path,
        )
        enabled = _scalar_boolean(
            archive["c4_enabled"], name="c4_enabled", path=diagnostic_path
        )
        mutation_enabled = _scalar_boolean(
            archive["c4_mutation_enabled"],
            name="c4_mutation_enabled",
            path=diagnostic_path,
        )
        failed_open = _scalar_boolean(
            archive["c4_fail_open"],
            name="c4_fail_open",
            path=diagnostic_path,
        )
        error = _scalar_text(
            archive["c4_error"], name="c4_error", path=diagnostic_path
        )

        result_indices = np.asarray(archive["c4_result_indices"])
        if (
            result_indices.ndim != 1
            or not np.issubdtype(result_indices.dtype, np.integer)
            or np.any(result_indices < 0)
        ):
            raise ValueError(
                f"{diagnostic_path}: c4_result_indices must be "
                "non-negative integers"
            )
        result_indices = np.asarray(result_indices, dtype=np.int64)
        if len(np.unique(result_indices)) != len(result_indices):
            raise ValueError(
                f"{diagnostic_path}: c4_result_indices must be unique"
            )
        rows = len(result_indices)

        stable_ids = np.asarray(archive["c4_stable_ids"])
        if (
            stable_ids.shape != (rows,)
            or not np.issubdtype(stable_ids.dtype, np.integer)
        ):
            raise ValueError(
                f"{diagnostic_path}: c4_stable_ids must have integer "
                f"shape [{rows}]"
            )
        scores = np.asarray(archive["c4_scores"], dtype=np.float64)
        if scores.shape != (rows,) or not np.isfinite(scores).all():
            raise ValueError(
                f"{diagnostic_path}: c4_scores must be finite shape [{rows}]"
            )
        labels = _string_rows(
            archive["c4_label"],
            name="c4_label",
            rows=rows,
            path=diagnostic_path,
        )
        normalized_labels = _string_rows(
            archive["c4_normalized_label"],
            name="c4_normalized_label",
            rows=rows,
            path=diagnostic_path,
        )
        reasons = _string_rows(
            archive["c4_reason"],
            name="c4_reason",
            rows=rows,
            path=diagnostic_path,
        )
        sources = _string_rows(
            archive["c4_source"],
            name="c4_source",
            rows=rows,
            path=diagnostic_path,
        )
        attempted = _boolean_rows(
            archive["c4_attempted"],
            name="c4_attempted",
            rows=rows,
            path=diagnostic_path,
        )
        proposed = _boolean_rows(
            archive["c4_proposed"],
            name="c4_proposed",
            rows=rows,
            path=diagnostic_path,
        )
        verified = _boolean_rows(
            archive["c4_verified"],
            name="c4_verified",
            rows=rows,
            path=diagnostic_path,
        )
        applied = _boolean_rows(
            archive["c4_applied"],
            name="c4_applied",
            rows=rows,
            path=diagnostic_path,
        )
        original_boxes = np.asarray(
            archive["c4_original_boxes"], dtype=np.float64
        )
        candidate_boxes = np.asarray(
            archive["c4_candidate_boxes"], dtype=np.float64
        )
        original_corners = np.asarray(
            archive["c4_original_corners"], dtype=np.float64
        )
        candidate_corners = np.asarray(
            archive["c4_candidate_corners"], dtype=np.float64
        )

    if scene_id != expected_scene_id:
        raise ValueError(
            f"{diagnostic_path}: scene {scene_id!r} does not match "
            f"{expected_scene_id!r}"
        )
    if schema != C4_DIAGNOSTIC_SCHEMA:
        raise ValueError(
            f"{diagnostic_path}: unsupported C4 schema {schema!r}; "
            f"expected {C4_DIAGNOSTIC_SCHEMA!r}"
        )
    if not enabled:
        raise ValueError(f"{diagnostic_path}: C4 observer is not enabled")
    if mutation_enabled:
        raise ValueError(
            f"{diagnostic_path}: C4 mutation must be disabled for this report"
        )
    if np.any(proposed & ~attempted):
        raise ValueError(f"{diagnostic_path}: proposed rows must be attempted")
    if np.any(verified & ~proposed):
        raise ValueError(f"{diagnostic_path}: verified rows must be proposed")
    if np.any(applied & ~verified):
        raise ValueError(f"{diagnostic_path}: applied rows must be verified")
    if np.any(applied):
        raise ValueError(
            f"{diagnostic_path}: observer diagnostics contain applied rows"
        )
    if original_boxes.shape != (rows, 6):
        raise ValueError(
            f"{diagnostic_path}: c4_original_boxes must have shape [{rows},6]"
        )
    if candidate_boxes.shape != (rows, 6):
        raise ValueError(
            f"{diagnostic_path}: c4_candidate_boxes must have shape [{rows},6]"
        )
    if original_corners.shape != (rows, 8, 3):
        raise ValueError(
            f"{diagnostic_path}: c4_original_corners must have shape "
            f"[{rows},8,3]"
        )
    if candidate_corners.shape != (rows, 8, 3):
        raise ValueError(
            f"{diagnostic_path}: c4_candidate_corners must have shape "
            f"[{rows},8,3]"
        )
    for name, boxes, flags in (
        ("c4_original_boxes", original_boxes, attempted | proposed | verified),
        ("c4_candidate_boxes", candidate_boxes, proposed | verified),
    ):
        selected = boxes[flags]
        if len(selected) and (
            not np.isfinite(selected).all()
            or np.any(selected[:, 3:6] <= 0.0)
        ):
            raise ValueError(f"{diagnostic_path}: active {name} rows are invalid")
    for name, corners, flags in (
        ("c4_original_corners", original_corners, np.ones(rows, dtype=bool)),
        ("c4_candidate_corners", candidate_corners, proposed | verified),
    ):
        selected = corners[flags]
        if len(selected) and not np.isfinite(selected).all():
            raise ValueError(f"{diagnostic_path}: active {name} rows are invalid")
    if failed_open and not error:
        raise ValueError(
            f"{diagnostic_path}: c4_fail_open requires a non-empty c4_error"
        )
    if not failed_open and error:
        raise ValueError(
            f"{diagnostic_path}: c4_error is set without c4_fail_open"
        )

    return C4SceneDiagnostics(
        result_indices=np.array(result_indices, copy=True),
        stable_ids=np.asarray(stable_ids, dtype=np.int64).copy(),
        scores=np.array(scores, copy=True),
        labels=labels,
        normalized_labels=normalized_labels,
        reasons=reasons,
        sources=sources,
        attempted=attempted,
        proposed=proposed,
        verified=verified,
        applied=applied,
        original_boxes=np.array(original_boxes, copy=True),
        candidate_boxes=np.array(candidate_boxes, copy=True),
        original_corners=np.array(original_corners, copy=True),
        candidate_corners=np.array(candidate_corners, copy=True),
        failed_open=failed_open,
        error=error,
    )


def _aligned_corners_minmax(
    corners: np.ndarray, transform: np.ndarray
) -> np.ndarray:
    """Axis-align exported oriented corners without reconstructing an AABB."""

    return corners_to_minmax(transform_corners(corners, transform))


def _paired_aabb_iou(
    left_corners: np.ndarray, right_corners: np.ndarray
) -> np.ndarray:
    """Return row-wise AABB IoU for two equally-sized corner tensors."""

    left = corners_to_minmax(left_corners)
    right = corners_to_minmax(right_corners)
    if left.shape != right.shape:
        raise ValueError(
            f"paired corner tensors disagree: {left.shape} versus {right.shape}"
        )
    if not len(left):
        return np.empty(0, dtype=np.float64)
    return np.asarray(
        [
            pairwise_aabb_iou(
                left[index : index + 1], right[index : index + 1]
            )[0, 0]
            for index in range(len(left))
        ],
        dtype=np.float64,
    )


def _best_gt(iou: np.ndarray) -> tuple[int | None, float]:
    values = np.asarray(iou, dtype=np.float64).reshape(-1)
    if not len(values):
        return None, 0.0
    index = int(np.argmax(values))
    return index, float(values[index])


def _crossing(before: float, after: float, threshold: float) -> str:
    before_above = before >= threshold
    after_above = after >= threshold
    if not before_above and after_above:
        return "up"
    if before_above and not after_above:
        return "down"
    return "above" if before_above else "below"


def _finite_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "count": len(finite),
        "mean": float(statistics.fmean(finite)),
        "median": float(statistics.median(finite)),
        "minimum": float(min(finite)),
        "maximum": float(max(finite)),
    }


def _metric_report(
    records: Sequence[tuple[float, bool]],
    *,
    ground_truth: int,
    predictions: int,
    maximum_matches: int,
) -> dict[str, Any]:
    ap, recall, precision = ranked_metrics(records, ground_truth)
    return {
        "predictions": int(predictions),
        "ground_truth": int(ground_truth),
        "maximum_matches": int(maximum_matches),
        "ap": float(ap),
        "recall": float(recall),
        "final_precision": float(precision),
    }


def _metric_delta(
    observer: Mapping[str, Any], simulated: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        key: (
            int(simulated[key]) - int(observer[key])
            if key in {"predictions", "ground_truth", "maximum_matches"}
            else float(simulated[key]) - float(observer[key])
        )
        for key in observer
    }


def _empty_class_flow() -> dict[str, Any]:
    return {
        "rows": 0,
        "attempted": 0,
        "proposed": 0,
        "verified": 0,
        "applied": 0,
        "improved": 0,
        "harmed": 0,
        "unchanged": 0,
        "_best_deltas": [],
        "_same_gt_deltas": [],
    }


def build_report(
    *,
    pred_root: str | Path,
    diagnostics_root: str | Path,
    scene_list: str | Path,
    gt_root: str | Path,
    scan_root: str | Path,
    exclude_scene_list: str | Path | None = None,
) -> dict[str, Any]:
    """Build the C4 observer geometry and score-preserving simulation report."""

    prediction_root = Path(pred_root)
    diagnostic_root = Path(diagnostics_root)
    gt_path = Path(gt_root)
    scans_path = Path(scan_root)
    requested_scenes = read_scene_ids(Path(scene_list))
    excluded: set[str] = set()
    if exclude_scene_list is not None:
        excluded = set(read_scene_ids(Path(exclude_scene_list)))
    scenes = [scene for scene in requested_scenes if scene not in excluded]
    if not scenes:
        raise ValueError("No held-out scenes remain after exclusions")

    methods = ("observer", "verified_replacement")
    records = {
        method: {threshold: [] for threshold in THRESHOLDS}
        for method in methods
    }
    matches = {
        method: {threshold: 0 for threshold in THRESHOLDS}
        for method in methods
    }
    total_predictions = 0
    total_ground_truth = 0
    flow = Counter()
    reasons: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    classes: defaultdict[str, dict[str, Any]] = defaultdict(_empty_class_flow)
    candidate_rows: list[dict[str, Any]] = []
    scene_reports: list[dict[str, Any]] = []
    original_best_values: list[float] = []
    candidate_best_values: list[float] = []
    same_gt_candidate_values: list[float] = []
    best_deltas: list[float] = []
    same_gt_deltas: list[float] = []
    score_rows_checked = 0
    maximum_score_delta = 0.0
    corner_rows_checked = 0
    maximum_corner_delta = 0.0
    export_box_ious: list[float] = []
    crossing_totals = {
        _threshold_key(threshold): Counter()
        for threshold in THRESHOLDS
    }

    for scene in scenes:
        corners, scores = load_scene_predictions(
            prediction_root / f"{scene}{PREDICTION_SUFFIX}"
        )
        diagnostics = load_c4_diagnostics(
            diagnostic_root / f"{scene}{DIAGNOSTIC_SUFFIX}",
            expected_scene_id=scene,
        )
        if (
            len(diagnostics.result_indices)
            and int(np.max(diagnostics.result_indices)) >= len(scores)
        ):
            raise ValueError(
                f"{scene}: c4_result_indices exceed exported predictions"
            )
        diagnostic_export_scores = scores[diagnostics.result_indices]
        diagnostic_export_corners = corners[diagnostics.result_indices]
        if len(diagnostic_export_scores):
            deltas = np.abs(diagnostic_export_scores - diagnostics.scores)
            maximum_score_delta = max(maximum_score_delta, float(deltas.max()))
            score_rows_checked += len(deltas)
            if not np.allclose(
                diagnostic_export_scores,
                diagnostics.scores,
                rtol=0.0,
                atol=1e-7,
            ):
                raise ValueError(
                    f"{scene}: C4 diagnostic scores disagree with exported B6 "
                    "scores"
                )
            corner_deltas = np.abs(
                diagnostic_export_corners - diagnostics.original_corners
            )
            maximum_corner_delta = max(
                maximum_corner_delta, float(corner_deltas.max())
            )
            corner_rows_checked += len(diagnostic_export_corners)
            if not np.allclose(
                diagnostic_export_corners,
                diagnostics.original_corners,
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError(
                    f"{scene}: C4 diagnostic original corners disagree "
                    "point-wise with exported B6 corners"
                )
            scene_export_corner_ious = _paired_aabb_iou(
                diagnostics.original_corners, diagnostic_export_corners
            )
            if not np.allclose(
                scene_export_corner_ious, 1.0, rtol=0.0, atol=1e-6
            ):
                raise ValueError(
                    f"{scene}: C4 diagnostic original corners disagree in "
                    "IoU with exported B6 corners"
                )
            export_box_ious.extend(scene_export_corner_ious.tolist())
        else:
            scene_export_corner_ious = np.empty(0, dtype=np.float64)

        transform = load_axis_alignment(scans_path, scene)
        observer_minmax = corners_to_minmax(
            transform_corners(corners, transform)
        )
        simulated_minmax = np.array(observer_minmax, copy=True)
        gt_payload = np.load(
            gt_path / f"{scene}_bbox.npy", allow_pickle=False
        )
        gt_minmax = center_size_to_minmax(gt_payload)

        scene_flow = {
            "rows": len(diagnostics.result_indices),
            "attempted": int(np.count_nonzero(diagnostics.attempted)),
            "proposed": int(np.count_nonzero(diagnostics.proposed)),
            "verified": int(np.count_nonzero(diagnostics.verified)),
            "applied": int(np.count_nonzero(diagnostics.applied)),
            "fail_open": diagnostics.failed_open,
            "error": diagnostics.error,
        }
        flow.update({key: value for key, value in scene_flow.items() if key not in {"fail_open", "error"}})
        flow["fail_open_scenes"] += int(diagnostics.failed_open)

        for row_index in range(len(diagnostics.result_indices)):
            reason = diagnostics.reasons[row_index] or "<empty>"
            label = (
                diagnostics.normalized_labels[row_index]
                or diagnostics.labels[row_index]
                or "<unknown>"
            )
            reasons[reason] += 1
            if not diagnostics.verified[row_index]:
                rejection_reasons[reason] += 1
            class_flow = classes[label]
            class_flow["rows"] += 1
            for name, flags in (
                ("attempted", diagnostics.attempted),
                ("proposed", diagnostics.proposed),
                ("verified", diagnostics.verified),
                ("applied", diagnostics.applied),
            ):
                class_flow[name] += int(flags[row_index])

        for row_index in np.flatnonzero(diagnostics.proposed):
            prediction_index = int(diagnostics.result_indices[row_index])
            original_minmax = _aligned_corners_minmax(
                diagnostics.original_corners[row_index : row_index + 1],
                transform,
            )
            candidate_minmax = _aligned_corners_minmax(
                diagnostics.candidate_corners[row_index : row_index + 1],
                transform,
            )
            original_iou = pairwise_aabb_iou(original_minmax, gt_minmax)[0]
            candidate_iou = pairwise_aabb_iou(candidate_minmax, gt_minmax)[0]
            original_gt, original_best = _best_gt(original_iou)
            candidate_gt, candidate_best = _best_gt(candidate_iou)
            candidate_same_gt = (
                0.0 if original_gt is None else float(candidate_iou[original_gt])
            )
            best_delta = candidate_best - original_best
            same_delta = candidate_same_gt - original_best
            outcome = (
                "improved"
                if best_delta > 1e-12
                else "harmed"
                if best_delta < -1e-12
                else "unchanged"
            )
            label = (
                diagnostics.normalized_labels[row_index]
                or diagnostics.labels[row_index]
                or "<unknown>"
            )
            classes[label][outcome] += 1
            classes[label]["_best_deltas"].append(best_delta)
            classes[label]["_same_gt_deltas"].append(same_delta)
            original_best_values.append(original_best)
            candidate_best_values.append(candidate_best)
            same_gt_candidate_values.append(candidate_same_gt)
            best_deltas.append(best_delta)
            same_gt_deltas.append(same_delta)
            threshold_crossing = {
                _threshold_key(threshold): _crossing(
                    original_best, candidate_best, threshold
                )
                for threshold in THRESHOLDS
            }
            same_gt_crossing = {
                _threshold_key(threshold): _crossing(
                    original_best, candidate_same_gt, threshold
                )
                for threshold in THRESHOLDS
            }
            for key, value in threshold_crossing.items():
                crossing_totals[key][value] += 1
            export_iou = float(scene_export_corner_ious[row_index])
            candidate_rows.append(
                {
                    "scene_id": scene,
                    "diagnostic_row": int(row_index),
                    "prediction_index": prediction_index,
                    "stable_id": int(diagnostics.stable_ids[row_index]),
                    "score": float(scores[prediction_index]),
                    "label": diagnostics.labels[row_index],
                    "normalized_label": diagnostics.normalized_labels[row_index],
                    "source": diagnostics.sources[row_index],
                    "reason": diagnostics.reasons[row_index],
                    "verified": bool(diagnostics.verified[row_index]),
                    "original_box_world": diagnostics.original_boxes[
                        row_index
                    ].tolist(),
                    "candidate_box_world": diagnostics.candidate_boxes[
                        row_index
                    ].tolist(),
                    "original_corners_world": diagnostics.original_corners[
                        row_index
                    ].tolist(),
                    "candidate_corners_world": diagnostics.candidate_corners[
                        row_index
                    ].tolist(),
                    "original_best_gt": {
                        "index": original_gt,
                        "iou": original_best,
                    },
                    "candidate_best_gt": {
                        "index": candidate_gt,
                        "iou": candidate_best,
                    },
                    "best_gt_iou_delta": best_delta,
                    "same_original_best_gt": {
                        "index": original_gt,
                        "candidate_iou": candidate_same_gt,
                        "delta": same_delta,
                    },
                    "outcome": outcome,
                    "threshold_crossing": threshold_crossing,
                    "same_gt_threshold_crossing": same_gt_crossing,
                    "original_export_box_iou": export_iou,
                }
            )
            if diagnostics.verified[row_index]:
                simulated_minmax[prediction_index] = candidate_minmax[0]

        observer_iou = pairwise_aabb_iou(observer_minmax, gt_minmax)
        simulated_iou = pairwise_aabb_iou(simulated_minmax, gt_minmax)
        for method, iou in (
            ("observer", observer_iou),
            ("verified_replacement", simulated_iou),
        ):
            for threshold in THRESHOLDS:
                real, _, maximum_matched = score_scene(iou, scores, threshold)
                records[method][threshold].extend(real)
                matches[method][threshold] += int(maximum_matched)
        total_predictions += len(scores)
        total_ground_truth += len(gt_minmax)
        scene_reports.append({"scene_id": scene, **scene_flow})

    threshold_reports: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        observer = _metric_report(
            records["observer"][threshold],
            ground_truth=total_ground_truth,
            predictions=total_predictions,
            maximum_matches=matches["observer"][threshold],
        )
        simulated = _metric_report(
            records["verified_replacement"][threshold],
            ground_truth=total_ground_truth,
            predictions=total_predictions,
            maximum_matches=matches["verified_replacement"][threshold],
        )
        threshold_reports[_threshold_key(threshold)] = {
            "observer": observer,
            "verified_replacement": simulated,
            "delta": _metric_delta(observer, simulated),
        }

    class_report: dict[str, Any] = {}
    for label, values in sorted(classes.items()):
        class_report[label] = {
            key: int(values[key])
            for key in (
                "rows",
                "attempted",
                "proposed",
                "verified",
                "applied",
                "improved",
                "harmed",
                "unchanged",
            )
        }
        class_report[label]["best_gt_iou_delta"] = _finite_summary(
            values["_best_deltas"]
        )
        class_report[label]["same_gt_iou_delta"] = _finite_summary(
            values["_same_gt_deltas"]
        )

    return {
        "schema": REPORT_SCHEMA,
        "format_version": REPORT_FORMAT_VERSION,
        "diagnostic_schema": C4_DIAGNOSTIC_SCHEMA,
        "requested_scene_count": len(requested_scenes),
        "excluded_scene_count": len(requested_scenes) - len(scenes),
        "scene_count": len(scenes),
        "thresholds": threshold_reports,
        "flow": {
            **{key: int(value) for key, value in sorted(flow.items())},
            "reason_histogram": dict(sorted(reasons.items())),
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
        },
        "geometry": {
            "axis_alignment_source": "diagnostic_oriented_corners",
            "six_dimensional_boxes_role": "diagnostics_only",
            "proposed_rows": len(candidate_rows),
            "improved": sum(row["outcome"] == "improved" for row in candidate_rows),
            "harmed": sum(row["outcome"] == "harmed" for row in candidate_rows),
            "unchanged": sum(row["outcome"] == "unchanged" for row in candidate_rows),
            "original_best_gt_iou": _finite_summary(original_best_values),
            "candidate_best_gt_iou": _finite_summary(candidate_best_values),
            "candidate_same_original_gt_iou": _finite_summary(
                same_gt_candidate_values
            ),
            "best_gt_iou_delta": _finite_summary(best_deltas),
            "same_gt_iou_delta": _finite_summary(same_gt_deltas),
            "threshold_crossings": {
                key: {
                    state: int(counter.get(state, 0))
                    for state in ("up", "down", "above", "below")
                }
                for key, counter in crossing_totals.items()
            },
            "original_export_box_iou": _finite_summary(export_box_ious),
        },
        "score_preservation": {
            "rows_checked": score_rows_checked,
            "scores_equal_with_atol_1e-7": True,
            "maximum_absolute_score_delta": maximum_score_delta,
            "corner_rows_checked": corner_rows_checked,
            "corners_equal_pointwise_with_atol_1e-6": True,
            "maximum_absolute_corner_delta": maximum_corner_delta,
            "corners_equal_by_aabb_iou_atol_1e-6": True,
            "prediction_count_unchanged": True,
            "prediction_order_unchanged": True,
        },
        "classes": class_report,
        "scenes": scene_reports,
        "candidate_rows": candidate_rows,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline C4 observer geometry report and score-preserving verified "
            "candidate simulation."
        )
    )
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--exclude-scene-list", type=Path)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        pred_root=args.pred_root,
        diagnostics_root=args.diagnostics_root,
        scene_list=args.scene_list,
        exclude_scene_list=args.exclude_scene_list,
        gt_root=args.gt_root,
        scan_root=args.scan_root,
    )
    rendered = json.dumps(
        report, indent=2, sort_keys=True, allow_nan=False
    )
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
