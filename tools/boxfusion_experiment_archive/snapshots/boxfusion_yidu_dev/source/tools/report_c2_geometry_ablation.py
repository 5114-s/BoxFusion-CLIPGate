#!/usr/bin/env python3
"""Offline paired report for the C2 depth-occupancy geometry ablation.

This program reads completed prediction pickles, C2 diagnostic ``.npz`` files,
ScanNet metadata, and ScanNet ground truth.  It never imports the BoxFusion
runtime and must not be imported by the online inference path.

Prediction pickle files are trusted local experiment artifacts with the
BoxFusion layout ``[[ (label, corners[8,3], score), ... ]]``.  Do not use this
tool on untrusted pickle files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# ``python tools/report_c2_geometry_ablation.py`` places ``tools/`` rather
# than the repository root on sys.path.  Add only the repository root in that
# direct-script case so the shared offline evaluator remains the sole source
# of prediction loading, IoU, matching, and AP implementations.
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
REPORT_SCHEMA = "boxfusion.c2_geometry_ablation_report"
REPORT_FORMAT_VERSION = 1
THRESHOLDS = tuple(float(value) for value in DEFAULT_THRESHOLDS)

_CORNER_SIGNS = np.asarray(
    [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class C2SceneDiagnostics:
    """Validated final-output rows and cumulative C2 scene counters."""

    result_indices: np.ndarray
    track_ids: np.ndarray
    labels: tuple[str, ...]
    branches: tuple[str, ...]
    reasons: tuple[str, ...]
    attempted: np.ndarray
    proposed: np.ndarray
    verified: np.ndarray
    applied: np.ndarray
    original_boxes: np.ndarray
    candidate_boxes: np.ndarray
    summary: Mapping[str, Any]


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


def _string_rows(
    value: np.ndarray,
    *,
    name: str,
    rows: int,
    path: Path,
) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.shape != (rows,) or array.dtype.hasobject:
        raise ValueError(
            f"{path}: {name} must have non-object shape [{rows}]"
        )
    output = []
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


def _summary_count(
    summary: Mapping[str, Any],
    name: str,
    *,
    path: Path,
) -> int:
    value = summary.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{path}: summary {name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{path}: summary {name} must be non-negative")
    return result


def _summary_rejections(
    summary: Mapping[str, Any],
    *,
    path: Path,
) -> dict[str, int]:
    value = summary.get("c2_rejections")
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: summary c2_rejections must be a mapping")
    output: dict[str, int] = {}
    for raw_reason, raw_count in value.items():
        if not isinstance(raw_reason, str) or not raw_reason:
            raise ValueError(f"{path}: C2 rejection reasons must be strings")
        if (
            isinstance(raw_count, bool)
            or not isinstance(raw_count, (int, np.integer))
            or int(raw_count) < 0
        ):
            raise ValueError(
                f"{path}: C2 rejection count for {raw_reason!r} is invalid"
            )
        output[raw_reason] = int(raw_count)
    return dict(sorted(output.items()))


def load_c2_diagnostics(
    path: str | Path,
    *,
    expected_scene_id: str,
) -> C2SceneDiagnostics:
    """Load the strict, pickle-free C2 output-row diagnostic contract."""

    diagnostic_path = Path(path)
    if not diagnostic_path.is_file():
        raise FileNotFoundError(diagnostic_path)
    required = {
        "scene_id",
        "result_indices",
        "track_ids",
        "labels",
        "c2_attempted",
        "c2_proposed",
        "c2_verified",
        "c2_applied",
        "c2_reason",
        "c2_branch",
        "c2_original_boxes",
        "c2_candidate_boxes",
        "summary_json",
    }
    with np.load(diagnostic_path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(
                f"{diagnostic_path}: missing C2 fields {sorted(missing)}"
            )
        scene_id = _scalar_text(
            archive["scene_id"],
            name="scene_id",
            path=diagnostic_path,
        )
        result_indices = np.asarray(archive["result_indices"])
        if (
            result_indices.ndim != 1
            or not np.issubdtype(result_indices.dtype, np.integer)
            or np.any(result_indices < 0)
        ):
            raise ValueError(
                f"{diagnostic_path}: result_indices must be non-negative "
                "integers"
            )
        result_indices = np.asarray(result_indices, dtype=np.int64)
        if len(np.unique(result_indices)) != len(result_indices):
            raise ValueError(
                f"{diagnostic_path}: result_indices must be unique"
            )
        rows = len(result_indices)

        track_ids = np.asarray(archive["track_ids"])
        if (
            track_ids.shape != (rows,)
            or not np.issubdtype(track_ids.dtype, np.integer)
        ):
            raise ValueError(
                f"{diagnostic_path}: track_ids must have integer shape "
                f"[{rows}]"
            )
        track_ids = np.asarray(track_ids, dtype=np.int64)
        labels = _string_rows(
            archive["labels"],
            name="labels",
            rows=rows,
            path=diagnostic_path,
        )
        branches = _string_rows(
            archive["c2_branch"],
            name="c2_branch",
            rows=rows,
            path=diagnostic_path,
        )
        reasons = _string_rows(
            archive["c2_reason"],
            name="c2_reason",
            rows=rows,
            path=diagnostic_path,
        )
        attempted = _boolean_rows(
            archive["c2_attempted"],
            name="c2_attempted",
            rows=rows,
            path=diagnostic_path,
        )
        proposed = _boolean_rows(
            archive["c2_proposed"],
            name="c2_proposed",
            rows=rows,
            path=diagnostic_path,
        )
        verified = _boolean_rows(
            archive["c2_verified"],
            name="c2_verified",
            rows=rows,
            path=diagnostic_path,
        )
        applied = _boolean_rows(
            archive["c2_applied"],
            name="c2_applied",
            rows=rows,
            path=diagnostic_path,
        )
        original_boxes = np.asarray(
            archive["c2_original_boxes"], dtype=np.float64
        )
        candidate_boxes = np.asarray(
            archive["c2_candidate_boxes"], dtype=np.float64
        )
        if original_boxes.shape != (rows, 6):
            raise ValueError(
                f"{diagnostic_path}: c2_original_boxes must have shape "
                f"[{rows},6]"
            )
        if candidate_boxes.shape != (rows, 6):
            raise ValueError(
                f"{diagnostic_path}: c2_candidate_boxes must have shape "
                f"[{rows},6]"
            )
        summary_text = _scalar_text(
            archive["summary_json"],
            name="summary_json",
            path=diagnostic_path,
        )

    if scene_id != expected_scene_id:
        raise ValueError(
            f"{diagnostic_path}: scene {scene_id!r} does not match "
            f"{expected_scene_id!r}"
        )
    try:
        summary = json.loads(summary_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{diagnostic_path}: summary_json is invalid"
        ) from error
    if not isinstance(summary, Mapping):
        raise ValueError(f"{diagnostic_path}: summary_json must be an object")

    if np.any(proposed & ~attempted):
        raise ValueError(f"{diagnostic_path}: proposed rows must be attempted")
    if np.any(verified & ~proposed):
        raise ValueError(f"{diagnostic_path}: verified rows must be proposed")
    if np.any(applied & ~verified):
        raise ValueError(f"{diagnostic_path}: applied rows must be verified")

    active_geometry = attempted | proposed | verified | applied
    for name, boxes in (
        ("c2_original_boxes", original_boxes),
        ("c2_candidate_boxes", candidate_boxes),
    ):
        selected = boxes[active_geometry]
        if (
            len(selected)
            and (
                not np.isfinite(selected).all()
                or np.any(selected[:, 3:6] <= 0.0)
            )
        ):
            raise ValueError(
                f"{diagnostic_path}: active {name} rows are invalid"
            )

    for name, row_flags in (
        ("c2_attempted", attempted),
        ("c2_proposed", proposed),
        ("c2_verified", verified),
        ("c2_applied", applied),
    ):
        if int(np.count_nonzero(row_flags)) > _summary_count(
            summary, name, path=diagnostic_path
        ):
            raise ValueError(
                f"{diagnostic_path}: output-row {name} exceeds scene summary"
            )
    _summary_rejections(summary, path=diagnostic_path)

    return C2SceneDiagnostics(
        result_indices=np.array(result_indices, copy=True),
        track_ids=np.array(track_ids, copy=True),
        labels=labels,
        branches=branches,
        reasons=reasons,
        attempted=attempted,
        proposed=proposed,
        verified=verified,
        applied=applied,
        original_boxes=np.array(original_boxes, copy=True),
        candidate_boxes=np.array(candidate_boxes, copy=True),
        summary=dict(summary),
    )


def _center_size_to_corners(boxes: np.ndarray) -> np.ndarray:
    values = np.asarray(boxes, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 6
        or not np.isfinite(values).all()
        or (len(values) and np.any(values[:, 3:6] <= 0.0))
    ):
        raise ValueError("center/size boxes must be finite positive [N,6]")
    return (
        values[:, None, :3]
        + _CORNER_SIGNS[None] * (0.5 * values[:, None, 3:6])
    )


def _aligned_box_minmax(
    boxes: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    return corners_to_minmax(
        transform_corners(_center_size_to_corners(boxes), transform)
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
    baseline: Mapping[str, Any],
    c2: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "predictions": int(c2["predictions"]) - int(baseline["predictions"]),
        "ground_truth": int(c2["ground_truth"])
        - int(baseline["ground_truth"]),
        "maximum_matches": int(c2["maximum_matches"])
        - int(baseline["maximum_matches"]),
        "ap": float(c2["ap"]) - float(baseline["ap"]),
        "recall": float(c2["recall"]) - float(baseline["recall"]),
        "final_precision": float(c2["final_precision"])
        - float(baseline["final_precision"]),
    }


def build_report(
    *,
    baseline_pred_root: str | Path,
    c2_pred_root: str | Path,
    c2_diagnostics_root: str | Path,
    scene_list: str | Path,
    gt_root: str | Path,
    scan_root: str | Path,
) -> dict[str, Any]:
    """Build a class-agnostic paired C2 report from completed artifacts."""

    baseline_root = Path(baseline_pred_root)
    candidate_root = Path(c2_pred_root)
    diagnostic_root = Path(c2_diagnostics_root)
    gt_path = Path(gt_root)
    scans_path = Path(scan_root)
    scenes = read_scene_ids(Path(scene_list))

    records: dict[str, dict[float, list[tuple[float, bool]]]] = {
        "baseline": {threshold: [] for threshold in THRESHOLDS},
        "c2": {threshold: [] for threshold in THRESHOLDS},
    }
    matches: dict[str, dict[float, int]] = {
        "baseline": {threshold: 0 for threshold in THRESHOLDS},
        "c2": {threshold: 0 for threshold in THRESHOLDS},
    }
    prediction_counts = {"baseline": 0, "c2": 0}
    total_ground_truth = 0
    flow_totals = {
        "attempted": 0,
        "proposed": 0,
        "verified": 0,
        "applied": 0,
    }
    rejection_totals: Counter[str] = Counter()
    scene_flows: list[dict[str, Any]] = []
    applied_rows: list[dict[str, Any]] = []
    counts_equal = True
    scores_equal = True
    maximum_score_delta = 0.0

    for scene in scenes:
        baseline_corners, baseline_scores = load_scene_predictions(
            baseline_root / f"{scene}{PREDICTION_SUFFIX}"
        )
        c2_corners, c2_scores = load_scene_predictions(
            candidate_root / f"{scene}{PREDICTION_SUFFIX}"
        )
        diagnostics = load_c2_diagnostics(
            diagnostic_root / f"{scene}{DIAGNOSTIC_SUFFIX}",
            expected_scene_id=scene,
        )
        transform = load_axis_alignment(scans_path, scene)
        baseline_minmax = corners_to_minmax(
            transform_corners(baseline_corners, transform)
        )
        c2_minmax = corners_to_minmax(
            transform_corners(c2_corners, transform)
        )
        gt_payload = np.load(
            gt_path / f"{scene}_bbox.npy", allow_pickle=False
        )
        gt_minmax = center_size_to_minmax(gt_payload)
        baseline_iou = pairwise_aabb_iou(baseline_minmax, gt_minmax)
        c2_iou = pairwise_aabb_iou(c2_minmax, gt_minmax)

        prediction_counts["baseline"] += len(baseline_scores)
        prediction_counts["c2"] += len(c2_scores)
        total_ground_truth += len(gt_minmax)
        for method, iou, scores in (
            ("baseline", baseline_iou, baseline_scores),
            ("c2", c2_iou, c2_scores),
        ):
            for threshold in THRESHOLDS:
                real, _, maximum_matched = score_scene(
                    iou, scores, threshold
                )
                records[method][threshold].extend(real)
                matches[method][threshold] += int(maximum_matched)

        scene_count_equal = len(baseline_scores) == len(c2_scores)
        counts_equal = counts_equal and scene_count_equal
        scene_scores_equal = bool(
            scene_count_equal
            and np.array_equal(baseline_scores, c2_scores)
        )
        scores_equal = scores_equal and scene_scores_equal
        if scene_count_equal and len(c2_scores):
            maximum_score_delta = max(
                maximum_score_delta,
                float(np.max(np.abs(c2_scores - baseline_scores))),
            )
        elif not scene_count_equal:
            maximum_score_delta = float("nan")

        if (
            len(diagnostics.result_indices)
            and int(np.max(diagnostics.result_indices)) >= len(c2_scores)
        ):
            raise ValueError(
                f"{scene}: diagnostic result index exceeds C2 predictions"
            )

        scene_flow = {
            name: _summary_count(
                diagnostics.summary,
                f"c2_{name}",
                path=diagnostic_root / f"{scene}{DIAGNOSTIC_SUFFIX}",
            )
            for name in flow_totals
        }
        scene_rejections = _summary_rejections(
            diagnostics.summary,
            path=diagnostic_root / f"{scene}{DIAGNOSTIC_SUFFIX}",
        )
        for name, value in scene_flow.items():
            flow_totals[name] += int(value)
        rejection_totals.update(scene_rejections)
        scene_flows.append(
            {
                "scene_id": scene,
                **scene_flow,
                "rejections": scene_rejections,
            }
        )

        applied_indices = np.flatnonzero(diagnostics.applied)
        for diagnostic_row in applied_indices:
            prediction_index = int(
                diagnostics.result_indices[diagnostic_row]
            )
            original_box = diagnostics.original_boxes[
                diagnostic_row : diagnostic_row + 1
            ]
            candidate_box = diagnostics.candidate_boxes[
                diagnostic_row : diagnostic_row + 1
            ]
            original_minmax = _aligned_box_minmax(
                original_box, transform
            )
            candidate_minmax = _aligned_box_minmax(
                candidate_box, transform
            )
            original_ious = pairwise_aabb_iou(
                original_minmax, gt_minmax
            )[0]
            candidate_ious = pairwise_aabb_iou(
                candidate_minmax, gt_minmax
            )[0]
            original_best_index, original_best_iou = _best_gt(
                original_ious
            )
            candidate_best_index, candidate_best_iou = _best_gt(
                candidate_ious
            )
            paired_candidate_iou = (
                0.0
                if original_best_index is None
                else float(candidate_ious[original_best_index])
            )
            exported_candidate_iou = float(
                pairwise_aabb_iou(
                    candidate_minmax,
                    c2_minmax[prediction_index : prediction_index + 1],
                )[0, 0]
            )
            baseline_export_iou = (
                None
                if prediction_index >= len(baseline_minmax)
                else float(
                    pairwise_aabb_iou(
                        original_minmax,
                        baseline_minmax[
                            prediction_index : prediction_index + 1
                        ],
                    )[0, 0]
                )
            )
            crossings = {
                _threshold_key(threshold): _crossing(
                    original_best_iou,
                    paired_candidate_iou,
                    threshold,
                )
                for threshold in THRESHOLDS
            }
            applied_rows.append(
                {
                    "scene_id": scene,
                    "diagnostic_row": int(diagnostic_row),
                    "prediction_index": prediction_index,
                    "track_id": int(
                        diagnostics.track_ids[diagnostic_row]
                    ),
                    "label": diagnostics.labels[diagnostic_row],
                    "branch": diagnostics.branches[diagnostic_row],
                    "reason": diagnostics.reasons[diagnostic_row],
                    "original_box_world": original_box[0].tolist(),
                    "candidate_box_world": candidate_box[0].tolist(),
                    "original_best_gt": {
                        "index": original_best_index,
                        "iou": original_best_iou,
                    },
                    "candidate_best_gt": {
                        "index": candidate_best_index,
                        "iou": candidate_best_iou,
                    },
                    "same_original_best_gt": {
                        "index": original_best_index,
                        "original_iou": original_best_iou,
                        "candidate_iou": paired_candidate_iou,
                        "delta": paired_candidate_iou
                        - original_best_iou,
                    },
                    "threshold_crossing": crossings,
                    "baseline_export_box_iou": baseline_export_iou,
                    "c2_export_box_iou": exported_candidate_iou,
                }
            )

    threshold_reports: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        baseline_report = _metric_report(
            records["baseline"][threshold],
            ground_truth=total_ground_truth,
            predictions=prediction_counts["baseline"],
            maximum_matches=matches["baseline"][threshold],
        )
        c2_report = _metric_report(
            records["c2"][threshold],
            ground_truth=total_ground_truth,
            predictions=prediction_counts["c2"],
            maximum_matches=matches["c2"][threshold],
        )
        threshold_reports[_threshold_key(threshold)] = {
            "baseline": baseline_report,
            "c2": c2_report,
            "delta": _metric_delta(baseline_report, c2_report),
        }

    return {
        "schema": REPORT_SCHEMA,
        "format_version": REPORT_FORMAT_VERSION,
        "scene_count": len(scenes),
        "thresholds": threshold_reports,
        "c2_flow": {
            **flow_totals,
            "rejections": dict(sorted(rejection_totals.items())),
            "scenes": scene_flows,
        },
        "pairing": {
            "detection_counts_equal": bool(counts_equal),
            "scores_equal_by_position": bool(scores_equal),
            "maximum_absolute_score_delta": (
                maximum_score_delta
                if math.isfinite(maximum_score_delta)
                else None
            ),
        },
        "applied_rows": applied_rows,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline AP and geometry report for paired baseline/C2 ScanNet "
            "predictions."
        )
    )
    parser.add_argument("--baseline-pred-root", type=Path, required=True)
    parser.add_argument("--c2-pred-root", type=Path, required=True)
    parser.add_argument("--c2-diagnostics-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        baseline_pred_root=args.baseline_pred_root,
        c2_pred_root=args.c2_pred_root,
        c2_diagnostics_root=args.c2_diagnostics_root,
        scene_list=args.scene_list,
        gt_root=args.gt_root,
        scan_root=args.scan_root,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
