#!/usr/bin/env python3
"""Offline recall audit for observer-only P2-v3 reliability fusion.

The formal baseline is always the frozen class-agnostic ``B6 ∪ P1 ∪ P2``
stream.  P2-v3 fused candidates are matched only against ground-truth boxes
left uncovered by that baseline.  The formal go/no-go gate is therefore
measured relative to exactly the same baseline used by the P2-v2 audit.

For diagnosis, every fused candidate is compared with its paired, immutable
P2-v2 component candidate.  Both controls use the same candidate subset,
scores and stable ordering; only box geometry changes.  Ground truth is read
solely by this offline tool and is never available to the online observer.

Prediction pickle files are trusted local experiment artifacts and must not
come from an untrusted source.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.report_p1_residual_recall import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    load_axis_alignment,
    load_gt_boxes,
    read_scene_ids,
    score_ordered_match,
    validate_thresholds,
)
from tools.report_p2_occupancy_recall import (  # noqa: E402
    CandidateStream,
    _concatenate,
    _merge_p1_p2_unique,
    _stream,
    _validate_exact_scene_set,
    corners_to_minmax,
    load_p2_diagnostic,
    load_predictions,
    pairwise_aabb_iou,
    transform_corners,
)
from tools.report_p2v2_local_geometry_recall import (  # noqa: E402
    load_p2v2_candidates,
)
from tools.validate_p2v3_run_artifacts import (  # noqa: E402
    P2V3_DIAGNOSTIC_SCHEMA,
    P2V3_PROFILE,
    P2V3_SOURCE,
    P2V3Diagnostic,
    load_p2v3_diagnostic,
)


REPORT_SCHEMA = "boxfusion.p2v3.reliability_fusion_recall_report.v1"


def _distribution(values: Sequence[np.ndarray]) -> dict[str, float | None]:
    non_empty = [
        np.asarray(value, dtype=np.float64).reshape(-1)
        for value in values
        if np.asarray(value).size
    ]
    if not non_empty:
        return {
            "minimum": None,
            "q10": None,
            "q50": None,
            "q90": None,
            "maximum": None,
            "mean": None,
        }
    merged = np.concatenate(non_empty)
    return {
        "minimum": float(np.min(merged)),
        "q10": float(np.quantile(merged, 0.10)),
        "q50": float(np.quantile(merged, 0.50)),
        "q90": float(np.quantile(merged, 0.90)),
        "maximum": float(np.max(merged)),
        "mean": float(np.mean(merged)),
    }


def _baseline_stream(
    *,
    prediction_path: Path,
    diagnostic_path: Path,
    scene_id: str,
    alignment: np.ndarray,
) -> tuple[CandidateStream, Any, Any, P2V3Diagnostic]:
    predictions = load_predictions(prediction_path)
    diagnostic = load_p2_diagnostic(
        diagnostic_path, expected_scene_id=scene_id
    )
    p2v2 = load_p2v2_candidates(
        diagnostic_path, expected_scene_id=scene_id
    )
    p2v3 = load_p2v3_diagnostic(
        diagnostic_path, expected_scene_id=scene_id
    )

    b6 = _stream(
        corners_to_minmax(
            transform_corners(predictions.corners_world, alignment)
        ),
        predictions.scores,
        np.asarray(
            [f"b6:{index:06d}" for index in range(len(predictions.scores))],
            dtype=np.str_,
        ),
    )
    p1 = _stream(
        corners_to_minmax(
            transform_corners(diagnostic.p1.corners_world, alignment)
        ),
        diagnostic.p1.scores,
        diagnostic.p1.candidate_ids,
    )
    p2 = _stream(
        corners_to_minmax(
            transform_corners(diagnostic.p2.corners_world, alignment)
        ),
        diagnostic.p2.objectness_scores,
        diagnostic.p2.candidate_ids,
    )
    p1_p2, _ = _merge_p1_p2_unique(p1, p2)
    baseline = _concatenate((("b6", b6), ("p1p2", p1_p2)))
    return baseline, diagnostic, p2v2, p2v3


def _paired_streams(
    *,
    p2v2: Any,
    p2v3: P2V3Diagnostic,
    alignment: np.ndarray,
) -> tuple[CandidateStream, CandidateStream, dict[str, int]]:
    """Return geometry-only component/fused controls with identical ordering."""

    parent_ids = [str(value) for value in p2v3.parent_candidate_ids.tolist()]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError(
            "P2-v3 parent P2-v2 candidate IDs must be one-to-one"
        )
    component_lookup = {
        str(candidate_id): index
        for index, candidate_id in enumerate(p2v2.candidate_ids.tolist())
    }
    found_rows = [
        (index, component_lookup[candidate_id])
        for index, candidate_id in enumerate(parent_ids)
        if candidate_id in component_lookup
    ]
    # P2-v3 consumes the typed per-step P2-v2 candidates before scene NMS.
    # Consequently, a legitimate parent may be absent from P2-v2's final
    # scene table.  Whenever the parent survives that NMS, verify the geometry
    # exactly; otherwise the core's detached component alias is the control.
    for p2v3_index, p2v2_index in found_rows:
        if not np.allclose(
            p2v2.corners_world[p2v2_index],
            p2v3.component_corners[p2v3_index],
            rtol=1e-5,
            atol=1e-5,
        ):
            raise ValueError(
                "P2-v3 component geometry disagrees with paired P2-v2 "
                "candidate"
            )

    shared_ids = np.asarray(
        [f"pair:{index:06d}:{value}" for index, value in enumerate(parent_ids)],
        dtype=np.str_,
    )
    component = _stream(
        corners_to_minmax(
            transform_corners(p2v3.component_corners, alignment)
        ),
        p2v3.scores,
        shared_ids,
    )
    fused = _stream(
        corners_to_minmax(
            transform_corners(p2v3.fused_corners, alignment)
        ),
        p2v3.scores,
        shared_ids,
    )
    return component, fused, {
        "parent_rows_verified_against_p2v2_scene_table": len(found_rows),
        "parent_rows_consumed_before_p2v2_scene_nms": (
            len(parent_ids) - len(found_rows)
        ),
    }


def _freeze_baseline(
    baseline: CandidateStream,
    gt_boxes: np.ndarray,
    threshold: float,
) -> tuple[int, np.ndarray]:
    matched = score_ordered_match(
        pairwise_aabb_iou(baseline.boxes, gt_boxes),
        baseline.scores,
        threshold,
        tie_break_ids=baseline.ids,
    )
    uncovered = np.ones(len(gt_boxes), dtype=bool)
    uncovered[matched.matched_gt] = False
    return int(matched.true_positive_count), uncovered


def _incremental_true_positives(
    candidates: CandidateStream,
    gt_boxes: np.ndarray,
    threshold: float,
    uncovered: np.ndarray,
) -> int:
    matched = score_ordered_match(
        pairwise_aabb_iou(candidates.boxes, gt_boxes),
        candidates.scores,
        threshold,
        allowed_gt=uncovered,
        tie_break_ids=candidates.ids,
    )
    return int(matched.true_positive_count)


def _paired_best_iou(
    component: CandidateStream,
    fused: CandidateStream,
    gt_boxes: np.ndarray,
    uncovered: np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    if len(component) != len(fused):
        raise ValueError("paired P2-v2/P2-v3 candidate counts disagree")
    if len(component) == 0:
        component_best = np.zeros((0,), dtype=np.float64)
        fused_best = np.zeros((0,), dtype=np.float64)
    elif not np.any(uncovered):
        component_best = np.zeros((len(component),), dtype=np.float64)
        fused_best = np.zeros((len(fused),), dtype=np.float64)
    else:
        component_best = np.max(
            pairwise_aabb_iou(component.boxes, gt_boxes[:, :6])[
                :, uncovered
            ],
            axis=1,
        )
        fused_best = np.max(
            pairwise_aabb_iou(fused.boxes, gt_boxes[:, :6])[:, uncovered],
            axis=1,
        )
    delta = fused_best - component_best
    tolerance = 1e-9
    return {
        "candidate_count": int(len(delta)),
        "fused_best_iou_improved_count": int(np.sum(delta > tolerance)),
        "fused_best_iou_tied_count": int(
            np.sum(np.abs(delta) <= tolerance)
        ),
        "fused_best_iou_worsened_count": int(np.sum(delta < -tolerance)),
        "component_to_fused_cross_up_count": int(
            np.sum(
                (component_best <= threshold)
                & (fused_best > threshold)
            )
        ),
        "component_to_fused_cross_down_count": int(
            np.sum(
                (component_best > threshold)
                & (fused_best <= threshold)
            )
        ),
        "best_iou_delta_sum": float(np.sum(delta, dtype=np.float64)),
    }


def evaluate(
    *,
    scenes: Sequence[str],
    prediction_root: str | os.PathLike[str],
    diagnostics_root: str | os.PathLike[str],
    gt_root: str | os.PathLike[str],
    scans_root: str | os.PathLike[str],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    minimum_delta_r25_pp: float = 3.0,
    minimum_delta_r50_pp: float = 1.0,
) -> dict[str, Any]:
    thresholds = validate_thresholds(thresholds)
    threshold_keys = {f"{value:.2f}" for value in thresholds}
    if not {"0.25", "0.50"}.issubset(threshold_keys):
        raise ValueError("go/no-go requires IoU thresholds 0.25 and 0.50")
    for name, value in (
        ("minimum_delta_r25_pp", minimum_delta_r25_pp),
        ("minimum_delta_r50_pp", minimum_delta_r50_pp),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    scene_ids = tuple(str(scene) for scene in scenes)
    if not scene_ids or len(scene_ids) != len(set(scene_ids)):
        raise ValueError("scenes must be non-empty and unique")
    prediction_directory = Path(prediction_root)
    diagnostic_directory = Path(diagnostics_root)
    gt_directory = Path(gt_root)
    scans_directory = Path(scans_root)
    for role, root in (
        ("prediction", prediction_directory),
        ("diagnostics", diagnostic_directory),
        ("ground-truth", gt_directory),
        ("scans", scans_directory),
    ):
        if not root.is_dir():
            raise FileNotFoundError(f"{role} root not found: {root}")
    _validate_exact_scene_set(
        prediction_directory,
        scene_ids,
        suffix="_boxes.pkl",
        role="prediction",
    )
    _validate_exact_scene_set(
        diagnostic_directory,
        scene_ids,
        suffix="_tracks.npz",
        role="diagnostic",
    )

    totals = {
        f"{threshold:.2f}": {
            "baseline": 0,
            "component": 0,
            "fused": 0,
            "improved": 0,
            "tied": 0,
            "worsened": 0,
            "cross_up": 0,
            "cross_down": 0,
            "best_iou_delta_sum": 0.0,
        }
        for threshold in thresholds
    }
    total_gt = 0
    candidate_count = 0
    p1_runtime = 0.0
    p2_runtime = 0.0
    p2v2_runtime = 0.0
    p2v3_runtime = 0.0
    step_count = 0
    input_candidate_count = 0
    eligible_candidate_count = 0
    pre_scene_nms_candidate_count = 0
    verified_parent_rows = 0
    pre_scene_parent_rows = 0
    checkpoint_shas: set[str] = set()
    reliability_values: dict[str, list[np.ndarray]] = {
        "component_weight": [],
        "center_component_weight": [],
        "extent_component_weight": [],
        "component_reliability": [],
        "parent_reliability": [],
        "mask_reliability": [],
        "depth_reliability": [],
        "support_reliability": [],
        "agreement_reliability": [],
    }
    per_scene: dict[str, Any] = {}

    for scene_id in scene_ids:
        diagnostic_path = diagnostic_directory / f"{scene_id}_tracks.npz"
        alignment = load_axis_alignment(scans_directory, scene_id)
        baseline, p2, p2v2, p2v3 = _baseline_stream(
            prediction_path=prediction_directory / f"{scene_id}_boxes.pkl",
            diagnostic_path=diagnostic_path,
            scene_id=scene_id,
            alignment=alignment,
        )
        component, fused, pair_contract = _paired_streams(
            p2v2=p2v2, p2v3=p2v3, alignment=alignment
        )
        gt_boxes = load_gt_boxes(gt_directory / f"{scene_id}_bbox.npy")
        scene_thresholds: dict[str, Any] = {}
        for threshold in thresholds:
            key = f"{threshold:.2f}"
            baseline_tp, uncovered = _freeze_baseline(
                baseline, gt_boxes, threshold
            )
            component_tp = _incremental_true_positives(
                component, gt_boxes, threshold, uncovered
            )
            fused_tp = _incremental_true_positives(
                fused, gt_boxes, threshold, uncovered
            )
            paired = _paired_best_iou(
                component,
                fused,
                gt_boxes,
                uncovered,
                threshold=threshold,
            )
            totals[key]["baseline"] += baseline_tp
            totals[key]["component"] += component_tp
            totals[key]["fused"] += fused_tp
            totals[key]["improved"] += paired[
                "fused_best_iou_improved_count"
            ]
            totals[key]["tied"] += paired["fused_best_iou_tied_count"]
            totals[key]["worsened"] += paired[
                "fused_best_iou_worsened_count"
            ]
            totals[key]["cross_up"] += paired[
                "component_to_fused_cross_up_count"
            ]
            totals[key]["cross_down"] += paired[
                "component_to_fused_cross_down_count"
            ]
            totals[key]["best_iou_delta_sum"] += paired[
                "best_iou_delta_sum"
            ]
            denominator = max(len(gt_boxes), 1)
            scene_thresholds[key] = {
                "baseline_true_positives": baseline_tp,
                "component_incremental_true_positives": component_tp,
                "fused_incremental_true_positives": fused_tp,
                "fused_minus_component_true_positives": (
                    fused_tp - component_tp
                ),
                "component_recall_gain": component_tp / denominator,
                "fused_recall_gain": fused_tp / denominator,
                "paired_geometry": paired,
            }

        per_scene[scene_id] = {
            "ground_truth_count": int(len(gt_boxes)),
            "baseline_candidate_count": int(len(baseline)),
            "paired_candidate_count": int(len(fused)),
            "pairing_contract": pair_contract,
            "p1_runtime_seconds": float(p2.p1.runtime_seconds),
            "p2_incremental_runtime_seconds": float(
                p2.p2.incremental_runtime_seconds
            ),
            "p2v2_incremental_runtime_seconds": float(
                p2v2.runtime_seconds
            ),
            "p2v3_incremental_runtime_seconds": float(
                p2v3.runtime_seconds
            ),
            "thresholds": scene_thresholds,
        }
        total_gt += len(gt_boxes)
        candidate_count += len(fused)
        p1_runtime += p2.p1.runtime_seconds
        p2_runtime += p2.p2.incremental_runtime_seconds
        p2v2_runtime += p2v2.runtime_seconds
        p2v3_runtime += p2v3.runtime_seconds
        step_count += len(p2v3.frame_ids)
        input_candidate_count += int(
            np.sum(p2v3.input_candidate_counts)
        )
        eligible_candidate_count += int(
            np.sum(p2v3.eligible_candidate_counts)
        )
        pre_scene_nms_candidate_count += int(
            np.sum(p2v3.step_candidate_counts)
        )
        verified_parent_rows += pair_contract[
            "parent_rows_verified_against_p2v2_scene_table"
        ]
        pre_scene_parent_rows += pair_contract[
            "parent_rows_consumed_before_p2v2_scene_nms"
        ]
        checkpoint_shas.add(p2v3.parent_p2_checkpoint_sha256)
        for name, values in (
            ("component_weight", p2v3.component_weights),
            (
                "center_component_weight",
                p2v3.center_component_weights,
            ),
            (
                "extent_component_weight",
                p2v3.extent_component_weights,
            ),
            ("component_reliability", p2v3.component_reliabilities),
            ("parent_reliability", p2v3.parent_reliabilities),
            ("mask_reliability", p2v3.mask_reliabilities),
            ("depth_reliability", p2v3.depth_reliabilities),
            ("support_reliability", p2v3.support_reliabilities),
            ("agreement_reliability", p2v3.agreement_reliabilities),
        ):
            reliability_values[name].append(
                np.asarray(values, dtype=np.float64)
            )
    if len(checkpoint_shas) != 1:
        raise ValueError("P2-v3 parent checkpoint changed across scenes")

    threshold_report: dict[str, Any] = {}
    for key, row in totals.items():
        baseline_tp = int(row["baseline"])
        component_tp = int(row["component"])
        fused_tp = int(row["fused"])
        component_gain = component_tp / max(total_gt, 1)
        fused_gain = fused_tp / max(total_gt, 1)
        threshold_report[key] = {
            "ground_truth_count": int(total_gt),
            "baseline": {
                "source": "b6_p1_p2_union",
                "true_positives": baseline_tp,
                "recall": baseline_tp / max(total_gt, 1),
            },
            "paired_p2v2_component_control": {
                "candidate_count": int(candidate_count),
                "true_positives": component_tp,
                "precision": component_tp / max(candidate_count, 1),
                "recall_gain": component_gain,
                "recall_gain_percentage_points": 100.0 * component_gain,
            },
            "p2v3_fused_incremental": {
                "candidate_count": int(candidate_count),
                "true_positives": fused_tp,
                "precision": fused_tp / max(candidate_count, 1),
                "recall_gain": fused_gain,
                "recall_gain_percentage_points": 100.0 * fused_gain,
            },
            "paired_delta": {
                "fused_minus_component_true_positives": (
                    fused_tp - component_tp
                ),
                "fused_minus_component_recall_percentage_points": (
                    100.0 * (fused_gain - component_gain)
                ),
                "fused_best_iou_improved_count": int(row["improved"]),
                "fused_best_iou_tied_count": int(row["tied"]),
                "fused_best_iou_worsened_count": int(row["worsened"]),
                "component_to_fused_cross_up_count": int(row["cross_up"]),
                "component_to_fused_cross_down_count": int(
                    row["cross_down"]
                ),
                "mean_best_uncovered_gt_iou_delta": float(
                    row["best_iou_delta_sum"]
                    / max(candidate_count, 1)
                ),
            },
            "combined": {
                "source": "b6_p1_p2_union_plus_p2v3_fused",
                "true_positives": baseline_tp + fused_tp,
                "recall": (baseline_tp + fused_tp) / max(total_gt, 1),
            },
        }

    delta_r25 = threshold_report["0.25"]["p2v3_fused_incremental"][
        "recall_gain_percentage_points"
    ]
    delta_r50 = threshold_report["0.50"]["p2v3_fused_incremental"][
        "recall_gain_percentage_points"
    ]
    passed = bool(
        delta_r25 >= minimum_delta_r25_pp
        and delta_r50 >= minimum_delta_r50_pp
    )
    scene_count = len(scene_ids)
    return {
        "schema": REPORT_SCHEMA,
        "diagnostic_contract": {
            "schema": P2V3_DIAGNOSTIC_SCHEMA,
            "profile": P2V3_PROFILE,
            "source": P2V3_SOURCE,
        },
        "matching_contract": (
            "class-agnostic; freeze stable score-ordered B6/P1/P2 "
            "matches, then independently match paired P2-v2 component and "
            "P2-v3 fused geometry with identical rows/scores against only "
            "uncovered GT; strict IoU > threshold; one-to-one per scene"
        ),
        "observer_only": True,
        "ground_truth_usage": "offline_evaluation_only",
        "safety": {
            "validated": True,
            "uses_ground_truth_online": False,
            "reads_semantic_labels": False,
            "mutation_enabled": False,
            "applied_count": 0,
            "parent_p2_checkpoint_sha256": next(iter(checkpoint_shas)),
        },
        "scene_count": int(scene_count),
        "ground_truth_count": int(total_gt),
        "candidate_count": int(candidate_count),
        "candidate_count_per_scene": candidate_count / max(scene_count, 1),
        "observer_work": {
            "step_count": int(step_count),
            "input_candidate_count": int(input_candidate_count),
            "eligible_candidate_count": int(eligible_candidate_count),
            "pre_scene_nms_candidate_count": int(
                pre_scene_nms_candidate_count
            ),
            "parent_rows_verified_against_p2v2_scene_table": int(
                verified_parent_rows
            ),
            "parent_rows_consumed_before_p2v2_scene_nms": int(
                pre_scene_parent_rows
            ),
        },
        "reliability_summary": {
            name: _distribution(values)
            for name, values in reliability_values.items()
        },
        "runtime_seconds": {
            "p1": float(p1_runtime),
            "p2_incremental": float(p2_runtime),
            "p2v2_incremental": float(p2v2_runtime),
            "baseline_p1_p2_total": float(p1_runtime + p2_runtime),
            "p2v3_incremental": float(p2v3_runtime),
            "p2v3_incremental_per_scene": (
                p2v3_runtime / max(scene_count, 1)
            ),
            "p2v3_incremental_per_step": (
                p2v3_runtime / max(step_count, 1)
            ),
            "full_observer_chain_total": float(
                p1_runtime + p2_runtime + p2v2_runtime + p2v3_runtime
            ),
            "p2v3_overhead_fraction_of_p1_p2_p2v2": (
                p2v3_runtime
                / (p1_runtime + p2_runtime + p2v2_runtime)
                if p1_runtime + p2_runtime + p2v2_runtime > 0.0
                else None
            ),
        },
        "thresholds": threshold_report,
        "go_no_go": {
            "baseline": "b6_p1_p2_union",
            "paired_component_control_is_gate_baseline": False,
            "scope": (
                "fixed10" if scene_count == 10 else f"{scene_count}_scenes"
            ),
            "minimum_delta_recall_at_025_percentage_points": float(
                minimum_delta_r25_pp
            ),
            "minimum_delta_recall_at_050_percentage_points": float(
                minimum_delta_r50_pp
            ),
            "observed_delta_recall_at_025_percentage_points": float(
                delta_r25
            ),
            "observed_delta_recall_at_050_percentage_points": float(
                delta_r50
            ),
            "passed": passed,
            "decision": "GO_TO_P3" if passed else "STOP_P2V3",
        },
        "per_scene": per_scene,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--diagnostics-root", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--scans-root", required=True, type=Path)
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS
    )
    parser.add_argument(
        "--minimum-delta-r25-pp", type=float, default=3.0
    )
    parser.add_argument(
        "--minimum-delta-r50-pp", type=float, default=1.0
    )
    parser.add_argument("--output", "--output-json", dest="output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate(
        scenes=read_scene_ids(args.scene_list),
        prediction_root=args.prediction_root,
        diagnostics_root=args.diagnostics_root,
        gt_root=args.gt_root,
        scans_root=args.scans_root,
        thresholds=args.thresholds,
        minimum_delta_r25_pp=args.minimum_delta_r25_pp,
        minimum_delta_r50_pp=args.minimum_delta_r50_pp,
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
