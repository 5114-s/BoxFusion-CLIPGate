#!/usr/bin/env python3
"""GT-only counterfactual audit for R3 anchor-near TR3D corrections.

R3 is an observer: it may associate one TR3D proposal with a frozen-G0
anchor, but it must never mutate the frozen prediction tree.  This audit is
the only R3 component allowed to read ScanNet ground truth.  It evaluates a
small, pre-registered family of deterministic selection rules while keeping
the original G0 score and stable order for replacement counterfactuals.

The fixed validation subset is veto-only.  A passing result permits a later
train-only calibration experiment; it never permits direct activation or
validation-set threshold fitting.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.frozen_anchor_manifest import verify_frozen_anchor_manifest  # noqa: E402
from boxfusion.tr3d_r2_cache import tr3d_r2_cache_path  # noqa: E402
from boxfusion.tr3d_r2_provenance import (  # noqa: E402
    canonical_json_sha256,
    frame_artifact_tree,
    load_prefix_manifest,
    sha256_file,
)
from boxfusion.tr3d_r2b_cache import tr3d_r2b_cache_path  # noqa: E402
from boxfusion.tr3d_residual_cache import tr3d_residual_cache_path  # noqa: E402
from tools.audit_tr3d_residual_observer import (  # noqa: E402
    _alignment,
    _gt_boxes,
    _load_b6,
    _minmax,
    _transform,
    maximum_cardinality,
    pairwise_iou,
)
from tools.run_tr3d_r3_near_observer import (  # noqa: E402
    REPORT_SCHEMA as R3_EXPORT_SCHEMA,
    _code_hash as current_r3_code_hash,
)
from tools.tr3d_data import discover_frame_bundle, read_scene_list  # noqa: E402


REPORT_SCHEMA = "boxfusion.tr3d_r3_near_correction_audit.v1"
IOU_THRESHOLDS = (0.15, 0.25, 0.50)
FIXED_RULES = (
    "tr3d_score_gt_anchor_score",
    "tr3d_score",
    "score_anchor_iou",
    "score_depth_quality",
    "fixed_joint",
)
PRIMARY_RULE = "tr3d_score_gt_anchor_score"
_EPSILON = 1e-12


@dataclass(frozen=True)
class SceneCounterfactual:
    """Validated scene arrays used by the pure counterfactual evaluator."""

    scene_id: str
    anchor_boxes: np.ndarray
    anchor_scores: np.ndarray
    gt_boxes: np.ndarray
    candidate_boxes: np.ndarray
    proposal_ids: np.ndarray
    anchor_indices: np.ndarray
    anchor_iou: np.ndarray
    tr3d_score: np.ndarray
    depth_available: np.ndarray
    depth_quality: np.ndarray
    feature_available: np.ndarray
    feature_cosine: np.ndarray


def _as_array(
    value: object,
    *,
    dtype: np.dtype,
    shape: tuple[int | None, ...],
    name: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(dtype) or array.ndim != len(shape):
        raise ValueError(f"{name} must have dtype {np.dtype(dtype)} and shape {shape}")
    if any(expected is not None and actual != expected for actual, expected in zip(array.shape, shape)):
        raise ValueError(f"{name} must have shape {shape}")
    return array


def validate_scene(scene: SceneCounterfactual) -> SceneCounterfactual:
    """Fail closed on malformed or ambiguous candidate/anchor relations."""

    if not isinstance(scene.scene_id, str) or not scene.scene_id:
        raise ValueError("scene_id must be non-empty")
    anchors = _as_array(
        scene.anchor_boxes, dtype=np.float64, shape=(None, 6), name="anchor_boxes"
    )
    scores = _as_array(
        scene.anchor_scores, dtype=np.float64, shape=(len(anchors),), name="anchor_scores"
    )
    gt = _as_array(scene.gt_boxes, dtype=np.float64, shape=(None, 6), name="gt_boxes")
    candidates = _as_array(
        scene.candidate_boxes, dtype=np.float64, shape=(None, 6), name="candidate_boxes"
    )
    count = len(candidates)
    proposal_ids = _as_array(
        scene.proposal_ids, dtype=np.int64, shape=(count,), name="proposal_ids"
    )
    anchor_indices = _as_array(
        scene.anchor_indices, dtype=np.int64, shape=(count,), name="anchor_indices"
    )
    anchor_iou = _as_array(
        scene.anchor_iou, dtype=np.float64, shape=(count,), name="anchor_iou"
    )
    tr3d_score = _as_array(
        scene.tr3d_score, dtype=np.float64, shape=(count,), name="tr3d_score"
    )
    depth_quality = _as_array(
        scene.depth_quality, dtype=np.float64, shape=(count,), name="depth_quality"
    )
    depth_available = _as_array(
        scene.depth_available,
        dtype=np.bool_,
        shape=(count,),
        name="depth_available",
    )
    feature_available = _as_array(
        scene.feature_available,
        dtype=np.bool_,
        shape=(count,),
        name="feature_available",
    )
    feature_cosine = _as_array(
        scene.feature_cosine,
        dtype=np.float64,
        shape=(count,),
        name="feature_cosine",
    )
    for name, values in (
        ("anchor_boxes", anchors),
        ("anchor_scores", scores),
        ("gt_boxes", gt),
        ("candidate_boxes", candidates),
        ("anchor_iou", anchor_iou),
        ("tr3d_score", tr3d_score),
        ("depth_quality", depth_quality),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
    if feature_available.any() and not np.isfinite(feature_cosine[feature_available]).all():
        raise ValueError("available feature cosine contains non-finite values")
    if np.any(feature_cosine[~feature_available] != 0):
        raise ValueError("unavailable feature cosine must use the zero sentinel")
    if len(np.unique(proposal_ids)) != count:
        raise ValueError("proposal_ids must be unique")
    if count and (np.any(anchor_indices < 0) or np.any(anchor_indices >= len(anchors))):
        raise ValueError("anchor_indices are out of range")
    if np.any((anchor_iou < 0) | (anchor_iou > 1)):
        raise ValueError("anchor_iou must be in [0,1]")
    if np.any((tr3d_score < 0) | (tr3d_score > 1)):
        raise ValueError("tr3d_score must be in [0,1]")
    if np.any((depth_quality < 0) | (depth_quality > 1)):
        raise ValueError("depth_quality must be in [0,1]")
    if np.any(depth_quality[~depth_available] != 0):
        raise ValueError("unavailable depth quality must use the zero sentinel")
    if feature_available.any() and np.any(
        (feature_cosine[feature_available] < -1 - _EPSILON)
        | (feature_cosine[feature_available] > 1 + _EPSILON)
    ):
        raise ValueError("feature cosine must be in [-1,1]")
    if len(anchors) and np.any(anchors[:, 3:] <= anchors[:, :3]):
        raise ValueError("anchor boxes must have positive extent")
    if len(candidates) and np.any(candidates[:, 3:] <= candidates[:, :3]):
        raise ValueError("candidate boxes must have positive extent")
    if len(gt) and np.any(gt[:, 3:] <= gt[:, :3]):
        raise ValueError("GT boxes must have positive extent")
    return scene


def fixed_rule_scores(scene: SceneCounterfactual) -> dict[str, np.ndarray]:
    """Return the four pre-registered, threshold-free R3 ranking signals."""

    validate_scene(scene)
    feature_quality = np.where(
        scene.feature_available,
        np.clip((scene.feature_cosine + 1.0) * 0.5, 0.0, 1.0),
        0.5,
    )
    return {
        "tr3d_score_gt_anchor_score": scene.tr3d_score.copy(),
        "tr3d_score": scene.tr3d_score.copy(),
        "score_anchor_iou": scene.tr3d_score * scene.anchor_iou,
        "score_depth_quality": scene.tr3d_score * scene.depth_quality,
        "fixed_joint": (
            scene.tr3d_score
            * (0.5 + 0.5 * scene.anchor_iou)
            * (0.75 + 0.25 * scene.depth_quality)
            * (0.75 + 0.25 * feature_quality)
        ),
    }


def select_one_per_anchor(
    scene: SceneCounterfactual, signal: object
) -> np.ndarray:
    """Select one row per represented anchor with deterministic tie breaks."""

    validate_scene(scene)
    values = np.asarray(signal, dtype=np.float64)
    if values.shape != (len(scene.candidate_boxes),) or not np.isfinite(values).all():
        raise ValueError("selection signal must be finite [candidate_count]")
    selected: list[int] = []
    for anchor in np.unique(scene.anchor_indices):
        rows = np.flatnonzero(scene.anchor_indices == anchor)
        # Primary: signal descending.  Secondary: proposal id ascending.
        # np.lexsort uses the last key as primary.
        order = np.lexsort((scene.proposal_ids[rows], -values[rows]))
        selected.append(int(rows[int(order[0])]))
    return np.asarray(selected, dtype=np.int64)


def replacement_boxes(scene: SceneCounterfactual, selected: object) -> np.ndarray:
    rows = np.asarray(selected, dtype=np.int64)
    if rows.ndim != 1 or (len(rows) and (np.any(rows < 0) or np.any(rows >= len(scene.candidate_boxes)))):
        raise ValueError("selected candidate rows are invalid")
    anchors = scene.anchor_indices[rows]
    if len(np.unique(anchors)) != len(anchors):
        raise ValueError("replacement must contain at most one row per anchor")
    result = scene.anchor_boxes.copy()
    result[anchors] = scene.candidate_boxes[rows]
    return result


def replacement_rows_for_rule(
    scene: SceneCounterfactual, selected: object, rule: str
) -> np.ndarray:
    """Apply the frozen primary safety condition without tuning a threshold."""

    rows = np.asarray(selected, dtype=np.int64)
    if rule == PRIMARY_RULE:
        keep = (
            scene.tr3d_score[rows]
            > scene.anchor_scores[scene.anchor_indices[rows]]
        )
        return rows[keep]
    return rows


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    changing = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[changing + 1] - mrec[changing]) * mpre[changing + 1]))


def scored_detection_metrics(
    scenes: Sequence[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    threshold: float,
) -> dict[str, float | int]:
    """Class-agnostic VOC AP with stable frozen score/order matching.

    Each input row is ``(scene_id, boxes, scores, gt_boxes)``.  Matching is
    intentionally identical to ScanNet's evaluator: a prediction chooses its
    maximum-IoU GT first and is a false positive when that GT was already
    claimed, even if another unmatched GT also overlaps.
    """

    if not math.isfinite(threshold) or threshold <= 0 or threshold > 1:
        raise ValueError("threshold must be finite in (0,1]")
    records: list[tuple[float, int, str, int, np.ndarray]] = []
    total_gt = 0
    gt_by_scene: dict[str, np.ndarray] = {}
    for scene_order, (scene_id, boxes, scores, gt) in enumerate(scenes):
        box_array = np.asarray(boxes, dtype=np.float64)
        score_array = np.asarray(scores, dtype=np.float64)
        gt_array = np.asarray(gt, dtype=np.float64)
        if box_array.ndim != 2 or box_array.shape[1:] != (6,):
            raise ValueError("prediction boxes must be [N,6]")
        if score_array.shape != (len(box_array),) or not np.isfinite(score_array).all():
            raise ValueError("prediction scores must be finite [N]")
        if gt_array.ndim != 2 or gt_array.shape[1:] != (6,):
            raise ValueError("GT boxes must be [M,6]")
        if scene_id in gt_by_scene:
            raise ValueError("scene ids must be unique")
        gt_by_scene[scene_id] = gt_array
        total_gt += len(gt_array)
        for row, (score, box) in enumerate(zip(score_array, box_array)):
            records.append((float(score), scene_order, scene_id, row, box))
    records.sort(key=lambda item: (-item[0], item[1], item[3]))
    used = {scene_id: np.zeros(len(gt), dtype=np.bool_) for scene_id, gt in gt_by_scene.items()}
    tp = np.zeros(len(records), dtype=np.float64)
    fp = np.ones(len(records), dtype=np.float64)
    for index, (_, _, scene_id, _, box) in enumerate(records):
        gt = gt_by_scene[scene_id]
        if not len(gt):
            continue
        overlaps = pairwise_iou(box[None], gt)[0]
        target = int(np.argmax(overlaps))
        if float(overlaps[target]) > threshold and not used[scene_id][target]:
            used[scene_id][target] = True
            tp[index] = 1.0
            fp[index] = 0.0
    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    # The +1e-6 mirrors evaluation/utils/eval_det.py exactly.
    recall = cumulative_tp / float(total_gt + 1e-6)
    precision = cumulative_tp / np.maximum(
        cumulative_tp + cumulative_fp, np.finfo(np.float64).eps
    )
    return {
        "predictions": len(records),
        "ground_truth": total_gt,
        "matched_tp": int(tp.sum()),
        "average_precision": _voc_ap(recall, precision) if len(records) else 0.0,
        "final_precision": float(precision[-1]) if len(precision) else 0.0,
        "final_recall": float(recall[-1]) if len(recall) else 0.0,
    }


def _maximum_matches_by_scene(
    scenes: Sequence[tuple[np.ndarray, np.ndarray]], threshold: float
) -> int:
    return sum(maximum_cardinality(pairwise_iou(boxes, gt), threshold) for boxes, gt in scenes)


def _scene_match_counts(
    scenes: Sequence[tuple[str, np.ndarray, np.ndarray]], threshold: float
) -> dict[str, int]:
    return {
        scene_id: maximum_cardinality(pairwise_iou(boxes, gt), threshold)
        for scene_id, boxes, gt in scenes
    }


def per_anchor_oracle_upper_bound(
    scene: SceneCounterfactual, threshold: float
) -> int:
    """Maximum matches when every anchor slot may keep or replace geometry.

    For a fixed threshold, a slot-to-GT edge exists when either the original
    anchor or any candidate assigned to that anchor overlaps the GT.  A
    bipartite matching on these compressed edges is an exact cardinality
    upper bound under the one-geometry-per-anchor constraint.
    """

    validate_scene(scene)
    slot_iou = pairwise_iou(scene.anchor_boxes, scene.gt_boxes)
    candidate_iou = pairwise_iou(scene.candidate_boxes, scene.gt_boxes)
    for row, anchor in enumerate(scene.anchor_indices):
        slot_iou[int(anchor)] = np.maximum(slot_iou[int(anchor)], candidate_iou[row])
    return maximum_cardinality(slot_iou, threshold)


def evaluate_counterfactuals(
    scenes: Sequence[SceneCounterfactual],
    thresholds: Sequence[float] = IOU_THRESHOLDS,
    available_rules: Sequence[str] = FIXED_RULES,
) -> dict[str, Any]:
    """Evaluate all fixed rules without selecting any validation threshold."""

    validated = [validate_scene(scene) for scene in scenes]
    if len({scene.scene_id for scene in validated}) != len(validated):
        raise ValueError("scene ids must be unique")
    threshold_values = tuple(float(value) for value in thresholds)
    if threshold_values != IOU_THRESHOLDS:
        raise ValueError(f"R3 audit thresholds are frozen to {IOU_THRESHOLDS}")
    rule_values = tuple(str(value) for value in available_rules)
    if not rule_values or len(set(rule_values)) != len(rule_values):
        raise ValueError("available_rules must be non-empty and unique")
    if any(value not in FIXED_RULES for value in rule_values):
        raise ValueError("available_rules contains an unregistered rule")
    if PRIMARY_RULE not in rule_values:
        raise ValueError("the pre-registered primary rule cannot be disabled")
    baseline_inputs = [
        (scene.scene_id, scene.anchor_boxes, scene.anchor_scores, scene.gt_boxes)
        for scene in validated
    ]
    baseline_oracle_inputs = [(scene.anchor_boxes, scene.gt_boxes) for scene in validated]
    baseline_scene_inputs = [
        (scene.scene_id, scene.anchor_boxes, scene.gt_boxes) for scene in validated
    ]
    selected_by_rule = {
        rule: {
            scene.scene_id: select_one_per_anchor(scene, fixed_rule_scores(scene)[rule])
            for scene in validated
        }
        for rule in rule_values
    }

    baseline: dict[str, Any] = {}
    all_near_union: dict[str, Any] = {}
    oracle: dict[str, Any] = {}
    rules: dict[str, Any] = {rule: {"selected_candidates": 0, "thresholds": {}} for rule in rule_values}
    for rule in rule_values:
        rules[rule]["selected_candidates"] = sum(
            len(selected_by_rule[rule][scene.scene_id]) for scene in validated
        )

    for threshold in threshold_values:
        key = f"{threshold:.2f}"
        baseline_scored = scored_detection_metrics(baseline_inputs, threshold)
        baseline_oracle_count = _maximum_matches_by_scene(baseline_oracle_inputs, threshold)
        baseline_scene_counts = _scene_match_counts(baseline_scene_inputs, threshold)
        baseline[key] = {
            "scored": baseline_scored,
            "maximum_matching": baseline_oracle_count,
        }
        union_inputs = [
            (np.concatenate((scene.anchor_boxes, scene.candidate_boxes), axis=0), scene.gt_boxes)
            for scene in validated
        ]
        union_count = _maximum_matches_by_scene(union_inputs, threshold)
        all_near_union[key] = {
            "maximum_matching": union_count,
            "delta_matches": union_count - baseline_oracle_count,
        }
        oracle_count = sum(per_anchor_oracle_upper_bound(scene, threshold) for scene in validated)
        oracle[key] = {
            "safe_keep_or_replace_maximum_matching": oracle_count,
            "delta_matches": oracle_count - baseline_oracle_count,
        }

        for rule in rule_values:
            replacement_inputs = []
            replacement_oracle_inputs = []
            replacement_scene_inputs = []
            add_one_inputs = []
            add_one_scored_inputs = []
            hit_count = 0
            hit_scenes = 0
            improvement_count = 0
            crossing_gain = 0
            crossing_loss = 0
            crossing_positive_scenes = 0
            crossing_negative_scenes = 0
            for scene in validated:
                selected = selected_by_rule[rule][scene.scene_id]
                applied = replacement_rows_for_rule(scene, selected, rule)
                replacement = replacement_boxes(scene, applied)
                replacement_inputs.append(
                    (scene.scene_id, replacement, scene.anchor_scores, scene.gt_boxes)
                )
                replacement_oracle_inputs.append((replacement, scene.gt_boxes))
                replacement_scene_inputs.append((scene.scene_id, replacement, scene.gt_boxes))
                add_one_inputs.append(
                    (
                        np.concatenate((scene.anchor_boxes, scene.candidate_boxes[applied]), axis=0),
                        scene.gt_boxes,
                    )
                )
                add_one_scored_inputs.append(
                    (
                        scene.scene_id,
                        np.concatenate(
                            (scene.anchor_boxes, scene.candidate_boxes[applied]),
                            axis=0,
                        ),
                        np.concatenate(
                            (
                                scene.anchor_scores,
                                np.zeros(len(applied), dtype=np.float64),
                            )
                        ),
                        scene.gt_boxes,
                    )
                )
                candidate_iou = pairwise_iou(scene.candidate_boxes[applied], scene.gt_boxes)
                anchor_iou = pairwise_iou(
                    scene.anchor_boxes[scene.anchor_indices[applied]], scene.gt_boxes
                )
                candidate_max = candidate_iou.max(axis=1) if candidate_iou.shape[1] else np.zeros(len(applied))
                anchor_max = anchor_iou.max(axis=1) if anchor_iou.shape[1] else np.zeros(len(applied))
                hits = candidate_max > threshold
                anchor_hits = anchor_max > threshold
                scene_gain = int(np.count_nonzero(hits & ~anchor_hits))
                scene_loss = int(np.count_nonzero(~hits & anchor_hits))
                crossing_gain += scene_gain
                crossing_loss += scene_loss
                crossing_positive_scenes += int(scene_gain > scene_loss)
                crossing_negative_scenes += int(scene_loss > scene_gain)
                hit_count += int(np.count_nonzero(hits))
                hit_scenes += int(np.any(hits))
                improvement_count += int(np.count_nonzero(candidate_max > anchor_max + _EPSILON))
            replacement_scored = scored_detection_metrics(replacement_inputs, threshold)
            replacement_oracle_count = _maximum_matches_by_scene(replacement_oracle_inputs, threshold)
            replacement_scene_counts = _scene_match_counts(replacement_scene_inputs, threshold)
            positive_scenes = sum(
                replacement_scene_counts[name] > baseline_scene_counts[name]
                for name in baseline_scene_counts
            )
            negative_scenes = sum(
                replacement_scene_counts[name] < baseline_scene_counts[name]
                for name in baseline_scene_counts
            )
            selected_count = sum(
                len(
                    replacement_rows_for_rule(
                        scene,
                        selected_by_rule[rule][scene.scene_id],
                        rule,
                    )
                )
                for scene in validated
            )
            add_count = _maximum_matches_by_scene(add_one_inputs, threshold)
            add_scored = scored_detection_metrics(add_one_scored_inputs, threshold)
            rules[rule]["thresholds"][key] = {
                "replacement": {
                    "scored": replacement_scored,
                    "delta_scored_ap": (
                        replacement_scored["average_precision"]
                        - baseline_scored["average_precision"]
                    ),
                    "delta_scored_tp": (
                        replacement_scored["matched_tp"] - baseline_scored["matched_tp"]
                    ),
                    "maximum_matching": replacement_oracle_count,
                    "delta_maximum_matches": replacement_oracle_count - baseline_oracle_count,
                    "positive_scene_coverage": positive_scenes,
                    "negative_scene_coverage": negative_scenes,
                },
                "add_one": {
                    "maximum_matching": add_count,
                    "delta_maximum_matches": add_count - baseline_oracle_count,
                    "conservative_source_score": 0.0,
                    "scored_supplementary_only": add_scored,
                    "delta_scored_ap_supplementary_only": (
                        add_scored["average_precision"]
                        - baseline_scored["average_precision"]
                    ),
                },
                "candidate_hits": {
                    "hits": hit_count,
                    "selected": selected_count,
                    "precision": hit_count / selected_count if selected_count else 0.0,
                    "scene_coverage": hit_scenes,
                    "geometry_improves_anchor": improvement_count,
                    "geometry_improvement_rate": (
                        improvement_count / selected_count if selected_count else 0.0
                    ),
                },
                "paired_threshold_crossing": {
                    "gain": crossing_gain,
                    "loss": crossing_loss,
                    "net_gain_minus_loss": crossing_gain - crossing_loss,
                    "gain_precision_among_crossings": (
                        crossing_gain / (crossing_gain + crossing_loss)
                        if crossing_gain + crossing_loss
                        else 0.0
                    ),
                    "positive_scene_coverage": crossing_positive_scenes,
                    "negative_scene_coverage": crossing_negative_scenes,
                },
                "selection": {
                    "evaluated_one_per_anchor": int(rules[rule]["selected_candidates"]),
                    "eligible_replacements": selected_count,
                    "condition": (
                        "tr3d_score > frozen_anchor_score"
                        if rule == PRIMARY_RULE
                        else "unconditional diagnostic"
                    ),
                },
            }

    passing_rules: list[str] = []
    gate_rows: dict[str, Any] = {}
    for rule in rule_values:
        row15 = rules[rule]["thresholds"]["0.15"]["replacement"]
        row25 = rules[rule]["thresholds"]["0.25"]["replacement"]
        row50 = rules[rule]["thresholds"]["0.50"]["replacement"]
        crossing50 = rules[rule]["thresholds"]["0.50"][
            "paired_threshold_crossing"
        ]
        pass_rule = bool(
            crossing50["net_gain_minus_loss"] >= 5
            and crossing50["positive_scene_coverage"] >= 3
            and crossing50["gain_precision_among_crossings"] >= 0.70
            and row50["delta_scored_ap"] >= 0.03
            and row15["delta_scored_ap"] >= -0.005
            and row25["delta_scored_ap"] >= -0.005
        )
        if pass_rule:
            passing_rules.append(rule)
        gate_rows[rule] = {
            "cross50_gain": crossing50["gain"],
            "cross50_loss": crossing50["loss"],
            "cross50_gain_minus_loss": crossing50["net_gain_minus_loss"],
            "cross50_replacement_precision": crossing50[
                "gain_precision_among_crossings"
            ],
            "cross50_positive_scene_coverage": crossing50[
                "positive_scene_coverage"
            ],
            "delta_scored_ap50": row50["delta_scored_ap"],
            "delta_scored_ap15": row15["delta_scored_ap"],
            "delta_scored_ap25": row25["delta_scored_ap"],
            "pass": pass_rule,
        }
    primary_pass = bool(gate_rows[PRIMARY_RULE]["pass"])
    return {
        "thresholds": list(threshold_values),
        "available_fixed_rules": list(rule_values),
        "unavailable_fixed_rules": [value for value in FIXED_RULES if value not in rule_values],
        "baseline": baseline,
        "all_near_union_oracle": all_near_union,
        "per_anchor_gt_oracle_upper_bound": oracle,
        "fixed_rules": rules,
        "pre_registered_gate": {
            "required": (
                "heldout90 primary conditional rule: cross50 gain-loss>=5, "
                "positive scene coverage>=3, replacement precision>=0.70, "
                "paired AP50 gain>=0.03, and AP15/AP25 loss<=0.005 each"
            ),
            "rules": gate_rows,
            "primary_rule": PRIMARY_RULE,
            "primary_rule_pass": primary_pass,
            "exploratory_passing_rules": passing_rules,
            "pass": primary_pass,
            "authorization_if_passed": "train-only calibration only",
            "direct_activation_authorized": False,
            "validation_threshold_or_rule_selection_permitted": False,
        },
    }


def _snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "anchor_name": payload["anchor_name"],
        "prediction_tree_sha256": payload["prediction_tree_sha256"],
        "artifact_tree_sha256": payload["artifact_tree_sha256"],
        "scene_list_sha256": payload["scene_list_sha256"],
    }


def _create_only(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"immutable R3 audit report exists: {path}") from error
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _load_export_report(path: Path, scenes: Sequence[str]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != R3_EXPORT_SCHEMA:
        raise ValueError("unsupported R3 export schema")
    if not payload.get("observer_only") or payload.get("mutation_enabled"):
        raise ValueError("R3 export violates observer-only contract")
    if int(payload.get("applied_count", -1)) != 0:
        raise ValueError("R3 export applied_count must be zero")
    if payload.get("ground_truth_access"):
        raise ValueError("R3 export improperly accessed ground truth")
    if payload.get("clip_access") or not payload.get("clip_semantics_unchanged"):
        raise ValueError("R3 export changed or accessed CLIP semantics")
    config = payload.get("r3_config")
    if not isinstance(config, dict) or canonical_json_sha256(config) != payload.get(
        "r3_config_sha256"
    ):
        raise ValueError("R3 export config hash mismatch")
    if (
        not config.get("observer_only")
        or config.get("mutation_enabled")
        or config.get("ground_truth_access")
        or config.get("clip_access")
        or not config.get("clip_semantics_unchanged")
    ):
        raise ValueError("R3 config violates observer/CLIP contract")
    if current_r3_code_hash() != payload.get("r3_code_sha256"):
        raise ValueError("current R3 code differs from immutable export")
    ordered = [str(row.get("scene_id")) for row in payload.get("scenes", [])]
    if (
        ordered != list(scenes)
        or int(payload.get("scene_count", -1)) != len(scenes)
        or len(scenes) not in (10, 100)
    ):
        raise ValueError("R3 audit requires the exact ordered fixed10 or full100 scene set")
    return payload


def _validate_optional_export_report_paths(
    export: Mapping[str, Any],
    *,
    r2a_enabled: bool,
    r2b_enabled: bool,
    r2a_export_report: Path | None,
    r2b_export_report: Path | None,
) -> Mapping[str, Any]:
    """Validate R2 report lineage only in the canonical input_reports node."""

    if (r2a_export_report is not None) != r2a_enabled:
        raise ValueError("R3 optional lineage argument r2a_export_report presence mismatch")
    if (r2b_export_report is not None) != r2b_enabled:
        raise ValueError("R3 optional lineage argument r2b_export_report presence mismatch")
    input_reports = export.get("input_reports", {})
    if not isinstance(input_reports, Mapping):
        raise ValueError("R3 input_reports must be a mapping")
    if bool(input_reports.get("r2a_enabled")) != r2a_enabled or bool(
        input_reports.get("r2b_enabled")
    ) != r2b_enabled:
        raise ValueError("R3 input report availability mismatch")
    for label, enabled, path in (
        ("r2a", r2a_enabled, r2a_export_report),
        ("r2b", r2b_enabled, r2b_export_report),
    ):
        if enabled:
            assert path is not None
            if Path(str(input_reports.get(f"{label}_export_report", ""))).resolve() != path.resolve():
                raise ValueError(f"R3 {label} report path mismatch")
            if input_reports.get(f"{label}_export_report_sha256") != sha256_file(path.resolve()):
                raise ValueError(f"R3 {label} report bytes changed")
    return input_reports


def _partition_counterfactuals(
    scenes: Sequence[SceneCounterfactual],
    development_scene_ids: Sequence[str] | None,
    *,
    available_rules: Sequence[str] = FIXED_RULES,
) -> dict[str, Any]:
    """Build development/all/held-out reports without an ad-hoc 90-list."""

    ordered = [scene.scene_id for scene in scenes]
    by_id = {scene.scene_id: scene for scene in scenes}
    if len(by_id) != len(ordered):
        raise ValueError("R3 scene list contains duplicates")
    if development_scene_ids is None:
        if len(scenes) != 10:
            raise ValueError(
                "full100 audit requires --development-scene-list for a held-out90 gate"
            )
        development = evaluate_counterfactuals(scenes, available_rules=available_rules)
        return {
            "mode": "development_fixed10_veto_only",
            "development10": development,
            "decision_partition": None,
            "heldout_gate_authoritative": False,
            "reason": (
                "fixed10 may veto the route but cannot authorize calibration; "
                "run full100 with the frozen development list"
            ),
        }
    development_ids = [str(value) for value in development_scene_ids]
    if len(development_ids) != 10 or len(set(development_ids)) != 10:
        raise ValueError("development scene list must contain exactly 10 unique scenes")
    missing = sorted(set(development_ids) - set(ordered))
    if missing:
        raise ValueError(f"development scenes are absent from main scene list: {missing}")
    if len(scenes) != 100:
        raise ValueError("held-out audit requires exactly 100 main scenes")
    heldout_ids = [scene_id for scene_id in ordered if scene_id not in set(development_ids)]
    if len(heldout_ids) != 90:
        raise ValueError("main/development difference must contain exactly 90 scenes")
    all_report = evaluate_counterfactuals(scenes, available_rules=available_rules)
    development_report = evaluate_counterfactuals(
        [by_id[value] for value in development_ids], available_rules=available_rules
    )
    heldout_report = evaluate_counterfactuals(
        [by_id[value] for value in heldout_ids], available_rules=available_rules
    )
    return {
        "mode": "full100_with_frozen_heldout90",
        "all100": all_report,
        "development10": development_report,
        "heldout90": heldout_report,
        "development_scene_ids": development_ids,
        "heldout_scene_ids": heldout_ids,
        "decision_partition": "heldout90",
        "heldout_gate_authoritative": True,
        "decision": heldout_report["pre_registered_gate"],
    }


def _scene_from_r3_cache(
    *,
    scene_id: str,
    frozen_root: Path,
    gt_root: Path,
    scans_root: Path,
    cache: Any,
) -> SceneCounterfactual:
    """Thin adapter kept separate so the immutable R3 cache can evolve once."""

    transform = _alignment(scans_root, scene_id)
    anchor_corners, anchor_scores = _load_b6(frozen_root / f"{scene_id}_boxes.pkl")
    anchor_boxes = _minmax(_transform(anchor_corners, transform))
    candidate_corners = np.asarray(cache.proposal_corners_world, dtype=np.float64)
    candidate_boxes = _minmax(_transform(candidate_corners, transform))
    return SceneCounterfactual(
        scene_id=scene_id,
        anchor_boxes=np.asarray(anchor_boxes, dtype=np.float64),
        anchor_scores=np.asarray(anchor_scores, dtype=np.float64),
        gt_boxes=np.asarray(_gt_boxes(gt_root / f"{scene_id}_bbox.npy"), dtype=np.float64),
        candidate_boxes=np.asarray(candidate_boxes, dtype=np.float64),
        proposal_ids=np.asarray(cache.proposal_ids, dtype=np.int64),
        anchor_indices=np.asarray(cache.anchor_index, dtype=np.int64),
        anchor_iou=np.asarray(cache.anchor_iou, dtype=np.float64),
        tr3d_score=np.asarray(cache.tr3d_score, dtype=np.float64),
        depth_available=np.asarray(cache.r2a_evidence_available, dtype=np.bool_),
        depth_quality=np.asarray(cache.r2a_depth_quality, dtype=np.float64),
        feature_available=np.asarray(cache.r2b_multiview_available, dtype=np.bool_),
        feature_cosine=np.asarray(cache.r2b_pairwise_cosine_mean, dtype=np.float64),
    )


def audit(args: argparse.Namespace) -> dict[str, Any]:
    """Load the immutable chain, run the pure audit, and reverify G0."""

    # Imported lazily so pure evaluator tests remain usable while the separate
    # R3 cache implementation is being landed in the isolated worktree.
    from boxfusion.tr3d_r3_cache import (  # type: ignore[import-not-found]
        load_tr3d_r3_cache,
        tr3d_r3_cache_path,
    )

    before = verify_frozen_anchor_manifest(args.frozen_manifest.resolve())
    before_snapshot = _snapshot(before)
    frozen_root = Path(before["reference_result_root"]).resolve()
    scenes = read_scene_list(args.scene_list.resolve())
    export = _load_export_report(args.r3_export_report.resolve(), scenes)
    if export.get("prefix_id") != args.prefix_id:
        raise ValueError("R3 export prefix id mismatch")
    if export.get("expected_parent_checkpoint_sha256") != args.expected_parent_checkpoint_sha256:
        raise ValueError("R3 export checkpoint SHA mismatch")
    if export.get("expected_parent_config_sha256") != args.expected_parent_config_sha256:
        raise ValueError("R3 export parent config SHA mismatch")
    if export.get("frozen_manifest_sha256") != sha256_file(args.frozen_manifest.resolve()):
        raise ValueError("R3 export frozen manifest bytes changed")
    if export.get("frozen_prediction_tree_sha256") != before["prediction_tree_sha256"]:
        raise ValueError("R3 export frozen prediction tree mismatch")
    expected_paths = {
        "frozen_manifest": args.frozen_manifest,
        "parent_cache_root": args.parent_cache_root,
        "prefix_manifest": args.prefix_manifest,
        "r3_cache_root": args.r3_cache_root,
        "scene_list": args.scene_list,
        "scans_root": args.scans_root,
    }
    for name, expected in expected_paths.items():
        if Path(str(export.get(name, ""))).resolve() != expected.resolve():
            raise ValueError(f"R3 export {name} mismatch")
    config = export["r3_config"]
    r2a_enabled = bool(config.get("r2a_enabled"))
    r2b_enabled = bool(config.get("r2b_enabled"))
    if r2b_enabled and not r2a_enabled:
        raise ValueError("R3 export enables R2b without R2a")
    optional_cli = {
        "r2a_cache_root": args.r2a_cache_root,
        "frames_root": args.frames_root,
        "r2b_cache_root": args.r2b_cache_root,
    }
    required_presence = {
        "r2a_cache_root": r2a_enabled,
        "frames_root": r2a_enabled,
        "r2b_cache_root": r2b_enabled,
    }
    for name, value in optional_cli.items():
        if (value is not None) != required_presence[name]:
            raise ValueError(f"R3 optional lineage argument {name} presence mismatch")
        exported = export.get(name)
        if value is None:
            if exported is not None:
                raise ValueError(f"R3 export {name} must explicitly be null")
        elif Path(str(exported)).resolve() != value.resolve():
            raise ValueError(f"R3 export {name} mismatch")
    input_reports = _validate_optional_export_report_paths(
        export,
        r2a_enabled=r2a_enabled,
        r2b_enabled=r2b_enabled,
        r2a_export_report=args.r2a_export_report,
        r2b_export_report=args.r2b_export_report,
    )
    evidence_hashes = export.get("parent_evidence_hashes", {})
    prefix_rows = load_prefix_manifest(args.prefix_manifest.resolve(), prefix_id=args.prefix_id)
    rows = {str(row["scene_id"]): row for row in export["scenes"]}
    scene_payloads: list[SceneCounterfactual] = []
    for scene_id in scenes:
        if bool(rows[scene_id].get("parent_r2a_available")) != r2a_enabled or bool(
            rows[scene_id].get("parent_r2b_available")
        ) != r2b_enabled:
            raise ValueError(f"{scene_id}: R3 export optional evidence drift")
        cache_path = tr3d_r3_cache_path(args.r3_cache_root.resolve(), scene_id, args.prefix_id)
        if sha256_file(cache_path) != rows[scene_id].get("r3_sidecar_sha256"):
            raise ValueError(f"{scene_id}: R3 cache bytes changed after export")
        anchor_path = frozen_root / f"{scene_id}_boxes.pkl"
        anchor_corners, anchor_scores = _load_b6(anchor_path)
        metadata_path = args.scans_root.resolve() / scene_id / f"{scene_id}.txt"
        parent_path = tr3d_residual_cache_path(
            args.parent_cache_root.resolve(), scene_id, args.prefix_id
        )
        r2a_path = (
            tr3d_r2_cache_path(args.r2a_cache_root.resolve(), scene_id, args.prefix_id)
            if r2a_enabled
            else None
        )
        r2b_path = (
            tr3d_r2b_cache_path(args.r2b_cache_root.resolve(), scene_id, args.prefix_id)
            if r2b_enabled
            else None
        )
        manifest_row_sha = frame_tree_sha = ""
        if r2a_enabled:
            manifest_row = prefix_rows[scene_id]
            manifest_row_sha = canonical_json_sha256(manifest_row)
            bundle = discover_frame_bundle(args.frames_root.resolve(), scene_id)
            frame_tree_sha, _ = frame_artifact_tree(manifest_row, bundle)
        cache = load_tr3d_r3_cache(
            cache_path,
            parent_tr3d_cache_path=parent_path,
            frozen_anchor_manifest_path=args.frozen_manifest.resolve(),
            anchor_prediction_path=anchor_path,
            anchor_corners_world=anchor_corners,
            anchor_scores=anchor_scores,
            axis_alignment_metadata_path=metadata_path,
            expected_checkpoint_sha256=args.expected_parent_checkpoint_sha256,
            expected_config_sha256=args.expected_parent_config_sha256,
            expected_scene_id=scene_id,
            expected_prefix_id=args.prefix_id,
            expected_r3_config_sha256=str(export["r3_config_sha256"]),
            expected_r3_code_sha256=str(export["r3_code_sha256"]),
            parent_r2a_cache_path=r2a_path,
            parent_r2b_cache_path=r2b_path,
            expected_prefix_manifest_row_sha256=manifest_row_sha,
            expected_frame_artifact_tree_sha256=frame_tree_sha,
            expected_r2_config_sha256=str(evidence_hashes.get("r2_config_sha256", "")),
            expected_r2_code_sha256=str(evidence_hashes.get("r2_code_sha256", "")),
            expected_feature_checkpoint_sha256=str(
                evidence_hashes.get("feature_checkpoint_sha256", "")
            ),
            expected_feature_config_sha256=str(
                evidence_hashes.get("feature_config_sha256", "")
            ),
            expected_feature_code_sha256=str(
                evidence_hashes.get("feature_code_sha256", "")
            ),
        )
        if cache.parent_r2a_available != r2a_enabled or cache.parent_r2b_available != r2b_enabled:
            raise ValueError(f"{scene_id}: R3 optional evidence availability drift")
        scene_payloads.append(
            _scene_from_r3_cache(
                scene_id=scene_id,
                frozen_root=frozen_root,
                gt_root=args.gt_root.resolve(),
                scans_root=args.scans_root.resolve(),
                cache=cache,
            )
        )
    development_ids = (
        read_scene_list(args.development_scene_list.resolve())
        if args.development_scene_list is not None
        else None
    )
    available_rules = [PRIMARY_RULE, "tr3d_score", "score_anchor_iou"]
    if r2a_enabled:
        available_rules.append("score_depth_quality")
    if r2b_enabled:
        available_rules.append("fixed_joint")
    counterfactual = _partition_counterfactuals(
        scene_payloads,
        development_ids,
        available_rules=available_rules,
    )
    after = verify_frozen_anchor_manifest(args.frozen_manifest.resolve())
    after_snapshot = _snapshot(after)
    if before_snapshot != after_snapshot:
        raise RuntimeError("frozen G0 anchor changed during R3 audit")
    return {
        "schema": REPORT_SCHEMA,
        "observer_contract": {
            "observer_only": True,
            "mutation_enabled": False,
            "applied_count": 0,
            "ground_truth_used_by_audit_only": True,
            "clip_semantics_unchanged": True,
            "frozen_anchor_verified_before_and_after": True,
            "before": before_snapshot,
            "after": after_snapshot,
        },
        "purpose": (
            "validation veto only; full100 decisions use the strict heldout90 "
            "difference; no threshold/rule fitting or direct activation is permitted"
        ),
        "anchor": {
            "name": before["anchor_name"],
            "metrics_percent": before["anchor_metrics_percent"],
        },
        "counts": {
            "scenes": len(scene_payloads),
            "anchors": sum(len(scene.anchor_boxes) for scene in scene_payloads),
            "near_candidates": sum(len(scene.candidate_boxes) for scene in scene_payloads),
            "ground_truth": sum(len(scene.gt_boxes) for scene in scene_payloads),
        },
        "fixed_joint_formula": (
            "score*(0.5+0.5*anchor_iou)*(0.75+0.25*depth_quality)*"
            "(0.75+0.25*feature_quality); unavailable feature_quality=0.5"
        ),
        "counterfactual": counterfactual,
        "lineage": {
            "r3_export_report_sha256": sha256_file(args.r3_export_report.resolve()),
            "r2a_export_report_sha256": (
                sha256_file(args.r2a_export_report.resolve()) if r2a_enabled else None
            ),
            "r2b_export_report_sha256": (
                sha256_file(args.r2b_export_report.resolve()) if r2b_enabled else None
            ),
            "scene_list_sha256": sha256_file(args.scene_list.resolve()),
            "development_scene_list_sha256": (
                sha256_file(args.development_scene_list.resolve())
                if args.development_scene_list is not None
                else None
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--parent-cache-root", type=Path, required=True)
    parser.add_argument("--prefix-manifest", type=Path, required=True)
    parser.add_argument("--r2a-cache-root", type=Path)
    parser.add_argument("--r2b-cache-root", type=Path)
    parser.add_argument("--r3-cache-root", type=Path, required=True)
    parser.add_argument("--r2a-export-report", type=Path)
    parser.add_argument("--r2b-export-report", type=Path)
    parser.add_argument("--frames-root", type=Path)
    parser.add_argument("--r3-export-report", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--development-scene-list", type=Path)
    parser.add_argument("--prefix-id", default="p100")
    parser.add_argument("--expected-parent-checkpoint-sha256", required=True)
    parser.add_argument("--expected-parent-config-sha256", required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--scans-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_path = args.report.resolve()
    manifest = verify_frozen_anchor_manifest(args.frozen_manifest.resolve())
    frozen_root = Path(manifest["reference_result_root"]).resolve()
    if report_path == frozen_root or frozen_root in report_path.parents:
        raise ValueError("audit report must not be written inside frozen G0")
    report = audit(args)
    _create_only(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
