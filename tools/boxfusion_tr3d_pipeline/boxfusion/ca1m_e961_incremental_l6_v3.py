"""Fail-closed CA-only E961 causal LightweightAsync stage-6/L6 helpers.

This independent v3 revision contains no runner, GT loader, GPU entrypoint,
fold1 path, official-validation path, READY writer, or authorization writer.
It freezes only pure numerical and identity contracts for a future preregistered
execution.  The invalid v2 module is not imported.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PENDING_SCHEMA = "boxfusion.ca1m_e961_incremental_l6_pending_config.v3"
PROTOCOL_SCHEMA = "boxfusion.ca1m_e961_incremental_l6_preregistration_protocol.v3"
NAMESPACE = "ca1m_e961_incremental_l6_v3"

FIT_FOLDS = (2, 3, 4)
REUSED_DIAGNOSTIC_FOLDS = (0,)
OOF_PLAN = {2: (3, 4), 3: (2, 4), 4: (2, 3)}

R4_PROTOCOL_SCHEMA = (
    "boxfusion.ca1m_tr3d_terminal_gate_preregistration_protocol.v5.final.r4"
)
R4_PROTOCOL_SHA256 = (
    "c45f7ebdb20342a8c17e8b43f2964d70b9afba70a079478d2b02599597573399"
)
R6_PREREGISTRATION_SCHEMA = (
    "boxfusion.ca1m_tr3d_e961_terminal_inputs_preregistration.v5.r6"
)
R6_PREREGISTRATION_SHA256 = (
    "60267f07a77773f3e83e05e5a223a70f4e031b30f90724b02f80fecbc0b9edaa"
)
V2_INVALID_SCHEMA = "boxfusion.ca1m_e961_incremental_l6_revision_invalid.v2"
V2_INVALID_SHA256 = (
    "831a86154b9ac1d2f9b0ac97c1ca09ea3d6b3d04c3b28049f69b55afee2e5441"
)

TERMINAL_PASS_STATUS = "PASS_EXPLORATORY_FOLD0_DIAGNOSTIC_COMPLETE"
TERMINAL_SCIENTIFIC_STOP_STATUS = "STOP_FOLD234_OOF_GATE_FAIL"

INCREMENTAL_OBSERVER_CONFIG = {
    "voxel_size_m": 0.03,
    "pixel_stride": 4,
    "min_depth_m": 0.10,
    "max_depth_m": 6.0,
    "warmup_keyframes": 3,
    "inference_interval_keyframes": 5,
    "max_memory_voxels": 300_000,
    "max_snapshot_points": 200_000,
    "track_iou_threshold": 0.15,
    "track_center_threshold_m": 0.30,
    "min_track_hits": 2,
}
LIGHTWEIGHT_FUSION_CONFIG = {
    "stage": 6,
    "top_k_views": 5,
    "diversity_weight": 0.30,
    "min_view_angle_deg": 12.0,
    "depth_pixel_stride": 6,
    "depth_margin_m": 0.05,
    "support_weight": 0.55,
    "occlusion_weight": 0.10,
    "free_space_weight": 0.75,
    "invalid_weight": 0.15,
    "fused_choice_margin": 0.02,
    "max_pending_snapshots": 1,
    "drain_on_finalize": False,
}
INCREMENTAL_PROVIDER_CONFIG = {"score_threshold": 0.01, "max_proposals": 256}

FEATURE_NAMES = (
    "best_score", "score_mean", "score_std", "log_hit_count",
    "lifespan_fraction", "hit_rate", "center_jitter_m",
    "extent_jitter_relative", "mean_match_iou", "log_point_support",
    "log_point_density", "post_terminal_anchor_iou_max",
    "post_terminal_anchor_center_distance_m",
    "matched_post_terminal_anchor_score", "log_volume", "log_aspect_ratio",
    "first_call_fraction", "last_call_fraction", "visibility_quality_mean",
    "support_ratio_mean", "free_space_ratio_mean", "invalid_ratio_mean",
    "selected_geometry_fused",
)
FEATURE_FORMULAS = (
    "float(best_score)", "float(score_mean)", "float(score_std)",
    "log1p(hit_count)", "lifespan_calls/max(provider_calls,1)",
    "float(hit_rate)", "float(center_jitter_m)",
    "float(extent_jitter_relative)", "float(mean_match_iou)",
    "log1p(point_support)", "log1p(point_density)",
    "max_AABB_IoU(selected_corners_world,post_terminal_anchor_corners)",
    "min(nearest_center_distance(selected_corners_world,post_terminal_anchor_corners),5.0)",
    "base_finalize_matched_anchor_score_using_raw_best_corners_against_post_terminal_anchors",
    "log(max(product(ptp(raw_best_corners_world,axis=0)),1e-6))",
    "log(max(raw_best_corners_max_extent/raw_best_corners_min_extent,1.0))",
    "first_call/max(provider_calls,1)", "last_call/max(provider_calls,1)",
    "float(visibility_quality_mean)", "float(support_ratio_mean)",
    "float(free_space_ratio_mean)", "float(invalid_ratio_mean)",
    "1.0_if_selected_geometry_fused_else_0.0",
)

TRAINING_HYPERPARAMETERS = {
    "iterations": 1800,
    "learning_rate": 0.06,
    "learning_rate_schedule": "lr/sqrt(1+step/200)",
    "l2": 0.003,
    "normalization": "training_partition_mean_std_ddof0_scale_lt_1e-6_to_1",
    "class_weight": "balanced_inverse_frequency",
    "iou50_row_weight_multiplier": 2.0,
}
SELECTION_REQUIREMENTS = {
    "threshold_grid_start": 0.05,
    "threshold_grid_stop": 0.95,
    "threshold_grid_step": 0.005,
    "threshold_grid_count_per_head": 181,
    "joint_grid_count": 32761,
    "min_selected": 20,
    "min_precision_novel25": 0.70,
    "min_precision_quality50": 0.45,
    "min_recall_novel50": 0.15,
    "min_positive_folds": 3,
    "hard_max_post_terminal_anchor_iou": 0.10,
    "free_space_veto": 0.45,
    "candidate_nms_iou": 0.25,
    "max_candidates_per_scene": 6,
}
SOURCE_RANK_FORMULA = (
    "novelty_probability+0.10*visibility01+0.08*support"
    "-0.18*free_space+0.04*fused_geometry"
)
SCORE_POLICY = "global_source_rank_float32_below_every_post_terminal_anchor_v3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_integer_scalar(value: Any, name: str, *, minimum: int = 0) -> int:
    source = np.asarray(value)
    if source.shape != () or source.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer scalar")
    result = int(source.item())
    if result < minimum:
        raise ValueError(f"{name} is below its minimum")
    return result


def _strict_int_vector(value: Any, name: str) -> np.ndarray:
    source = np.asarray(value)
    if source.ndim != 1 or source.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return np.ascontiguousarray(source, dtype=np.int64)


def _strict_bool_vector(value: Any, name: str) -> np.ndarray:
    source = np.asarray(value)
    if source.ndim != 1 or source.dtype.kind != "b":
        raise ValueError(f"{name} must be a one-dimensional bool array")
    return np.ascontiguousarray(source, dtype=np.bool_)


def _strict_probability_vector(value: Any, name: str, length: int) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind == "b":
        raise ValueError(f"{name} must not be bool")
    result = np.asarray(source, dtype=np.float64)
    if (
        result.shape != (length,) or not np.isfinite(result).all()
        or np.any((result < 0.0) | (result > 1.0))
    ):
        raise ValueError(f"{name} must be finite probabilities")
    return result


def validate_fold234_oof_plan(
    scene_folds: Any, scoring_train_folds: Sequence[Sequence[int]],
) -> None:
    held = _strict_int_vector(scene_folds, "held folds")
    if len(held) != len(scoring_train_folds):
        raise ValueError("L6 OOF fold arrays differ")
    if set(held.tolist()) != set(FIT_FOLDS):
        raise ValueError("L6 OOF rows must cover folds 2,3,4 only")
    for fold, raw_train in zip(held.tolist(), scoring_train_folds):
        train = tuple(_strict_int_vector(raw_train, "scoring train folds").tolist())
        if train != OOF_PLAN.get(int(fold)) or int(fold) in train:
            raise ValueError("L6 row is not scene-grouped fold234 OOF")


def validate_r4_gate_oof_scoring(
    held_folds: Any, scoring_train_fold_json: Sequence[str],
) -> None:
    train_folds: list[np.ndarray] = []
    for value in scoring_train_fold_json:
        if not isinstance(value, str):
            raise ValueError("R4 gate OOF train-fold JSON must be text")
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("R4 gate OOF train-fold JSON differs") from error
        train_folds.append(_strict_int_vector(decoded, "R4 gate train folds"))
    validate_fold234_oof_plan(held_folds, train_folds)


def threshold_grid() -> np.ndarray:
    values = np.linspace(0.05, 0.95, 181, dtype=np.float64)
    if values.shape != (181,) or not np.allclose(np.diff(values), 0.005):
        raise RuntimeError("L6 threshold grid construction changed")
    values.setflags(write=False)
    return values


def terminal_selection_for_state(
    state: str, r4_candidate_rows: Any, selected_near: Any,
    candidate_count: Any,
) -> np.ndarray:
    count = _strict_integer_scalar(candidate_count, "candidate count")
    if state == TERMINAL_PASS_STATUS:
        rows = _strict_int_vector(r4_candidate_rows, "R4 candidate rows")
        selected = _strict_bool_vector(selected_near, "R4 selected near")
        if (
            selected.shape != rows.shape
            or np.any((rows < 0) | (rows >= count))
            or len(rows) != len(set(rows.tolist()))
        ):
            raise ValueError("R4 near-to-raw candidate mapping differs")
        raw = np.zeros(count, dtype=np.bool_)
        raw[rows] = selected
        return raw
    if state == TERMINAL_SCIENTIFIC_STOP_STATUS:
        rows = _strict_int_vector(r4_candidate_rows, "R4 STOP candidate rows")
        selected = _strict_bool_vector(selected_near, "R4 STOP selected near")
        if len(rows) or len(selected):
            raise ValueError("R4 scientific STOP must not claim selected rows")
        return np.zeros(count, dtype=np.bool_)
    raise PermissionError("R4 result is neither scientific PASS nor scientific STOP")


def post_terminal_gt_coverage(
    *, anchor_best_gt: Any, anchor_best_iou: Any,
    candidate_anchor_positions: Any, candidate_best_gt: Any,
    candidate_max_gt_iou: Any, terminal_selected: Any,
) -> dict[int, float]:
    anchor_gt = _strict_int_vector(anchor_best_gt, "anchor best GT").copy()
    anchor_iou = _strict_probability_vector(
        anchor_best_iou, "anchor best IoU", len(anchor_gt),
    ).copy()
    owner = _strict_int_vector(candidate_anchor_positions, "candidate anchor positions")
    candidate_gt = _strict_int_vector(candidate_best_gt, "candidate best GT")
    candidate_iou = _strict_probability_vector(
        candidate_max_gt_iou, "candidate best IoU", len(owner),
    )
    selected = _strict_bool_vector(terminal_selected, "terminal selected")
    if (
        candidate_gt.shape != owner.shape or selected.shape != owner.shape
        or np.any((owner < 0) | (owner >= len(anchor_gt)))
    ):
        raise ValueError("post-terminal coverage arrays differ")
    selected_rows = np.flatnonzero(selected)
    selected_owners = owner[selected_rows]
    if len(selected_owners) != len(set(selected_owners.tolist())):
        raise ValueError("R4 selected more than one candidate for one anchor")
    anchor_gt[selected_owners] = candidate_gt[selected_rows]
    anchor_iou[selected_owners] = candidate_iou[selected_rows]
    coverage: dict[int, float] = {}
    for gt_index, iou in zip(anchor_gt.tolist(), anchor_iou.tolist()):
        if gt_index >= 0:
            coverage[gt_index] = max(coverage.get(gt_index, 0.0), float(iou))
    return coverage


def track_targets(
    *, track_best_gt: Any, track_max_gt_iou: Any,
    post_terminal_coverage: Mapping[int, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    best_gt = _strict_int_vector(track_best_gt, "track best GT")
    best_iou = _strict_probability_vector(
        track_max_gt_iou, "track best IoU", len(best_gt),
    )
    coverage_checked: dict[int, float] = {}
    for key, raw_value in post_terminal_coverage.items():
        gt_index = _strict_integer_scalar(key, "coverage GT index")
        source = np.asarray(raw_value)
        if source.shape != () or source.dtype.kind == "b":
            raise ValueError("coverage value must be a numeric scalar")
        value = float(source)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("coverage value must be in [0,1]")
        coverage_checked[gt_index] = value
    novel25 = np.zeros(len(best_gt), dtype=np.bool_)
    quality50 = np.zeros(len(best_gt), dtype=np.bool_)
    novel50 = np.zeros(len(best_gt), dtype=np.bool_)
    for row, (gt_index, iou) in enumerate(zip(best_gt.tolist(), best_iou.tolist())):
        coverage = coverage_checked.get(gt_index, 0.0)
        novel25[row] = gt_index >= 0 and iou >= 0.25 and coverage < 0.25
        quality50[row] = gt_index >= 0 and iou >= 0.50
        novel50[row] = quality50[row] and coverage < 0.50
    for value in (novel25, quality50, novel50):
        value.setflags(write=False)
    return novel25, quality50, novel50


def source_aware_rank(
    *, novelty_probability: float, visibility_quality_mean: float,
    support_ratio_mean: float, free_space_ratio_mean: float,
    selected_geometry: str,
) -> float:
    raw_values = (
        novelty_probability, visibility_quality_mean, support_ratio_mean,
        free_space_ratio_mean,
    )
    if any(np.asarray(value).shape != () or np.asarray(value).dtype.kind == "b" for value in raw_values):
        raise ValueError("L6 rank scalars must be numeric non-bool")
    probability, visibility, support, free_space = (
        float(value) for value in raw_values
    )
    if not all(math.isfinite(value) for value in (
        probability, visibility, support, free_space,
    )):
        raise ValueError("L6 rank inputs must be finite")
    if not 0.0 <= probability <= 1.0 or not -1.0 <= visibility <= 1.0:
        raise ValueError("L6 probability/visibility range differs")
    if not 0.0 <= support <= 1.0 or not 0.0 <= free_space <= 1.0:
        raise ValueError("L6 support/free-space range differs")
    if selected_geometry not in {"raw", "fused"}:
        raise ValueError("L6 geometry source differs")
    visibility01 = float(np.clip((visibility + 1.0) * 0.5, 0.0, 1.0))
    return float(
        probability + 0.10 * visibility01 + 0.08 * support
        - 0.18 * free_space + (0.04 if selected_geometry == "fused" else 0.0)
    )


def _aabb_iou_one_to_many(corners: np.ndarray, others: np.ndarray) -> np.ndarray:
    left = np.asarray(corners, dtype=np.float32).reshape(1, 8, 3)
    right = np.asarray(others, dtype=np.float32).reshape(-1, 8, 3)
    left_min, left_max = left.min(axis=1), left.max(axis=1)
    right_min, right_max = right.min(axis=1), right.max(axis=1)
    extent = np.maximum(
        np.minimum(left_max[:, None], right_max[None])
        - np.maximum(left_min[:, None], right_min[None]), 0.0,
    )
    intersection = np.prod(extent, axis=2)
    left_volume = np.prod(left_max - left_min, axis=1)
    right_volume = np.prod(right_max - right_min, axis=1)
    union = left_volume[:, None] + right_volume[None] - intersection
    return np.divide(
        intersection, union, out=np.zeros_like(intersection), where=union > 0.0,
    )


def _strict_unit_scalar(value: Any, name: str) -> float:
    source = np.asarray(value)
    if source.shape != () or source.dtype.kind == "b":
        raise ValueError(f"{name} must be a numeric non-bool scalar")
    result = float(source)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0,1]")
    return result


def select_stage6_candidates(
    rows: Sequence[Mapping[str, Any]], novelty_probability: Any,
    quality_probability: Any, *, novelty_threshold: float,
    quality_threshold: float,
) -> tuple[int, ...]:
    n = len(rows)
    novelty = _strict_probability_vector(
        novelty_probability, "novelty probability", n,
    )
    quality = _strict_probability_vector(
        quality_probability, "quality probability", n,
    )
    novelty_gate = _strict_unit_scalar(novelty_threshold, "novelty threshold")
    quality_gate = _strict_unit_scalar(quality_threshold, "quality threshold")
    identities: set[int] = set()
    ranked: list[tuple[float, float, float, int, int, np.ndarray]] = []
    for index, row in enumerate(rows):
        track_id = _strict_integer_scalar(row.get("track_id"), "track ID")
        if track_id in identities:
            raise ValueError("duplicate scene-local L6 track ID")
        identities.add(track_id)
        anchor_iou = _strict_unit_scalar(
            row.get("post_terminal_anchor_iou_max"), "post-terminal anchor IoU",
        )
        free_space = _strict_unit_scalar(
            row.get("free_space_ratio_mean"), "free-space ratio",
        )
        best_score = _strict_unit_scalar(row.get("best_score"), "best score")
        if novelty[index] < novelty_gate or quality[index] < quality_gate:
            continue
        if anchor_iou > 0.10 or free_space > 0.45:
            continue
        corners = np.ascontiguousarray(row.get("selected_corners_world"), dtype=np.float32)
        if corners.shape != (8, 3) or not np.isfinite(corners).all():
            raise ValueError("L6 selected geometry differs")
        rank = source_aware_rank(
            novelty_probability=float(novelty[index]),
            visibility_quality_mean=row.get("visibility_quality_mean"),
            support_ratio_mean=row.get("support_ratio_mean"),
            free_space_ratio_mean=free_space,
            selected_geometry=str(row.get("selected_geometry")),
        )
        ranked.append((-rank, -float(novelty[index]), -best_score, track_id, index, corners))
    ranked.sort(key=lambda item: item[:4])
    kept: list[tuple[float, float, float, int, int, np.ndarray]] = []
    for item in ranked:
        if kept:
            others = np.stack([saved[5] for saved in kept])
            if float(_aabb_iou_one_to_many(item[5], others).max(initial=0.0)) > 0.25:
                continue
        kept.append(item)
        if len(kept) >= 6:
            break
    return tuple(item[4] for item in kept)


def assign_low_scores(
    entries: Sequence[tuple[int, int, float]], anchor_score_floor: float,
) -> dict[tuple[int, int], float]:
    floor = _strict_unit_scalar(anchor_score_floor, "anchor score floor")
    if floor <= 0.0:
        raise ValueError("anchor score floor must be positive")
    checked: list[tuple[int, int, float]] = []
    for raw_scene, raw_local, raw_rank in entries:
        scene = _strict_integer_scalar(raw_scene, "scene-local score scene index")
        local = _strict_integer_scalar(raw_local, "scene-local score row index")
        rank_source = np.asarray(raw_rank)
        if rank_source.shape != () or rank_source.dtype.kind == "b":
            raise ValueError("source rank must be numeric non-bool")
        rank = float(rank_source)
        if not math.isfinite(rank):
            raise ValueError("source rank must be finite")
        checked.append((scene, local, rank))
    identities = [(scene, local) for scene, local, _ in checked]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate L6 candidate identity")
    ordered = sorted(checked, key=lambda row: (-row[2], row[0], row[1]))
    if not ordered:
        return {}
    cap = np.float32(floor * 0.5)
    if not 0.0 < float(cap) < floor:
        raise ValueError("cannot form a score band below terminal anchors")
    result: dict[tuple[int, int], float] = {}
    previous = float("inf")
    count = len(ordered)
    for rank_index, (scene, local, _) in enumerate(ordered):
        score = float(np.float32(float(cap) * (count - rank_index) / (count + 1.0)))
        if not 0.0 < score < floor or score >= previous:
            raise ValueError("float32 quantization changed L6 source order")
        result[(scene, local)] = score
        previous = score
    return result


__all__ = [
    "FEATURE_FORMULAS", "FEATURE_NAMES", "FIT_FOLDS",
    "INCREMENTAL_OBSERVER_CONFIG", "INCREMENTAL_PROVIDER_CONFIG",
    "LIGHTWEIGHT_FUSION_CONFIG", "NAMESPACE", "OOF_PLAN", "PENDING_SCHEMA",
    "PROTOCOL_SCHEMA", "R4_PROTOCOL_SCHEMA", "R4_PROTOCOL_SHA256",
    "R6_PREREGISTRATION_SCHEMA", "R6_PREREGISTRATION_SHA256",
    "REUSED_DIAGNOSTIC_FOLDS", "SCORE_POLICY", "SELECTION_REQUIREMENTS",
    "SOURCE_RANK_FORMULA", "TERMINAL_PASS_STATUS",
    "TERMINAL_SCIENTIFIC_STOP_STATUS", "TRAINING_HYPERPARAMETERS",
    "V2_INVALID_SCHEMA", "V2_INVALID_SHA256", "assign_low_scores",
    "post_terminal_gt_coverage", "select_stage6_candidates", "sha256_file",
    "source_aware_rank", "terminal_selection_for_state", "threshold_grid",
    "track_targets", "validate_fold234_oof_plan",
    "validate_r4_gate_oof_scoring",
]
