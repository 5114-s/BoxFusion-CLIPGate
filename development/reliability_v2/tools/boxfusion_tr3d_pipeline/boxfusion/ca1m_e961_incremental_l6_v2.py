"""Static CA-only E961 causal LightweightAsync stage-6/L6 helpers.

This module is deliberately non-executable.  It contains the small, pure
pieces that can be frozen before the terminal R4 result exists: the fold234
cross-fit plan, reconstruction of post-terminal GT coverage from OOF rows,
novelty targets, and the append-only low-score contract.  It has no GT loader,
model runner, CUDA import, fold1 path, or official-validation path.

The L6 candidate universe is the confirmed-track output of the causal
``LightweightAsyncTR3DObserver`` stage-6 rerun with each held-out E961 detector. Neither
the one-shot raw P proposals nor R4's strict ``best_anchor_iou > 0.15`` near
subset is accepted as a substitute for the temporal track collector.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PENDING_SCHEMA = "boxfusion.ca1m_e961_incremental_l6_pending_config.v2"
PROTOCOL_SCHEMA = "boxfusion.ca1m_e961_incremental_l6_preregistration_protocol.v2"
DATASET_SCHEMA = "boxfusion.ca1m_e961_incremental_l6_dataset.v2"
OOF_SCHEMA = "boxfusion.ca1m_e961_incremental_l6_oof_predictions.v2"
THRESHOLD_SCHEMA = "boxfusion.ca1m_e961_incremental_l6_threshold_receipt.v2"
POLICY_SCHEMA = "boxfusion.ca1m_e961_incremental_l6_policy.non_deployable.v2"

NAMESPACE = "ca1m_e961_incremental_l6_v2"
FIT_FOLDS = (2, 3, 4)
REUSED_DIAGNOSTIC_FOLDS = (0,)
OOF_PLAN = {2: (3, 4), 3: (2, 4), 4: (2, 3)}

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
INCREMENTAL_PROVIDER_CONFIG = {"score_threshold": 0.01, "max_proposals": 256}
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
    "float(best_score)",
    "float(score_mean)",
    "float(score_std)",
    "log1p(hit_count)",
    "lifespan_calls/max(provider_calls,1)",
    "float(hit_rate)",
    "float(center_jitter_m)",
    "float(extent_jitter_relative)",
    "float(mean_match_iou)",
    "log1p(point_support)",
    "log1p(point_density)",
    "max_AABB_IoU(selected_corners_world,post_terminal_anchor_corners)",
    "min(nearest_center_distance(selected_corners_world,post_terminal_anchor_corners),5.0)",
    "base_finalize_matched_anchor_score_using_raw_best_corners_against_post_terminal_anchors",
    "log(max(product(ptp(raw_best_corners_world,axis=0)),1e-6))",
    "log(max(raw_best_corners_max_extent/raw_best_corners_min_extent,1.0))",
    "first_call/max(provider_calls,1)",
    "last_call/max(provider_calls,1)",
    "float(visibility_quality_mean)",
    "float(support_ratio_mean)",
    "float(free_space_ratio_mean)",
    "float(invalid_ratio_mean)",
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
    "joint_grid_count": 181 * 181,
    "min_selected": 20,
    "min_precision_novel25": 0.70,
    "min_precision_quality50": 0.45,
    "min_recall_novel50": 0.15,
    "min_positive_folds": 3,
    "max_candidates_per_scene": 6,
    "hard_max_post_terminal_anchor_iou": 0.10,
    "candidate_nms_iou": 0.25,
}

R4_PROTOCOL_SCHEMA = (
    "boxfusion.ca1m_tr3d_terminal_gate_preregistration_protocol.v5.final.r4"
)
R4_PROTOCOL_SHA256 = (
    "c45f7ebdb20342a8c17e8b43f2964d70b9afba70a079478d2b02599597573399"
)
R4_RUN_SCHEMA = "boxfusion.ca1m_tr3d_terminal_gate_run.v5.final.r4"
R4_STOP_SCHEMA = "boxfusion.ca1m_tr3d_terminal_gate_stop.v5.final.r4"
R6_PREREGISTRATION_SCHEMA = (
    "boxfusion.ca1m_tr3d_e961_terminal_inputs_preregistration.v5.r6"
)
R6_PREREGISTRATION_SHA256 = (
    "60267f07a77773f3e83e05e5a223a70f4e031b30f90724b02f80fecbc0b9edaa"
)

SOURCE_RANK_FORMULA = (
    "novelty_probability+0.10*visibility01+0.08*support"
    "-0.18*free_space+0.04*fused_geometry"
)
SCORE_POLICY = "global_source_rank_float32_below_every_post_terminal_anchor_v2"

TERMINAL_PASS_STATUS = "PASS_EXPLORATORY_FOLD0_DIAGNOSTIC_COMPLETE"
TERMINAL_SCIENTIFIC_STOP_STATUS = "STOP_FOLD234_OOF_GATE_FAIL"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_int_vector(value: Any, name: str) -> np.ndarray:
    source = np.asarray(value)
    if source.ndim != 1 or source.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return np.asarray(source, dtype=np.int64)


def _strict_bool_vector(value: Any, name: str) -> np.ndarray:
    source = np.asarray(value)
    if source.ndim != 1 or source.dtype.kind != "b":
        raise ValueError(f"{name} must be a one-dimensional bool array")
    return np.asarray(source, dtype=np.bool_)


def validate_fold234_oof_plan(
    scene_folds: Any, scoring_train_folds: Sequence[Sequence[int]],
) -> None:
    """Require every scored row to be produced by the other two fit folds."""

    held = np.asarray(scene_folds)
    if held.ndim != 1 or len(held) != len(scoring_train_folds):
        raise ValueError("L6 OOF fold arrays differ")
    if set(held.astype(int).tolist()) != set(FIT_FOLDS):
        raise ValueError("L6 OOF rows must cover folds 2,3,4 only")
    for fold, train in zip(held.tolist(), scoring_train_folds):
        fold_id = int(fold)
        train_tuple = tuple(int(value) for value in train)
        if train_tuple != OOF_PLAN.get(fold_id) or fold_id in train_tuple:
            raise ValueError("L6 row is not scene-grouped fold234 OOF")


def validate_r4_gate_oof_scoring(
    held_folds: Any, scoring_train_fold_json: Sequence[str],
) -> None:
    """Prove R4 PASS anchor selection uses heldout gate predictions only."""

    import json

    train_folds: list[tuple[int, ...]] = []
    for value in scoring_train_fold_json:
        try:
            decoded = json.loads(str(value))
        except json.JSONDecodeError as error:
            raise ValueError("R4 gate OOF train-fold JSON differs") from error
        if not isinstance(decoded, list):
            raise ValueError("R4 gate OOF train-fold JSON differs")
        train_folds.append(tuple(int(item) for item in decoded))
    validate_fold234_oof_plan(held_folds, train_folds)


def threshold_grid() -> np.ndarray:
    """Return the exact frozen per-head threshold grid (181 float64 rows)."""

    values = np.linspace(0.05, 0.95, 181, dtype=np.float64)
    if values.shape != (181,) or not np.allclose(np.diff(values), 0.005):
        raise RuntimeError("L6 threshold grid construction changed")
    values.setflags(write=False)
    return values


def terminal_selection_for_state(
    state: str, r4_candidate_rows: Any, selected_near: Any,
    candidate_count: int,
) -> np.ndarray:
    """Resolve the two scientific R4 states without hiding failures.

    PASS uses the frozen R4 fold234 OOF choice.  Scientific STOP means the
    terminal stage is inactive and therefore preserves identity B6 anchors.
    Any other state is a provenance/implementation failure and blocks L6.
    """

    count = int(candidate_count)
    if count < 0:
        raise ValueError("candidate count must be non-negative")
    if state == TERMINAL_PASS_STATUS:
        near_rows = _strict_int_vector(r4_candidate_rows, "R4 candidate rows")
        selected = _strict_bool_vector(selected_near, "R4 selected near")
        if (
            selected.shape != near_rows.shape
            or np.any((near_rows < 0) | (near_rows >= count))
            or len(near_rows) != len(set(near_rows.astype(int).tolist()))
        ):
            raise ValueError("R4 near-to-raw candidate mapping differs")
        raw = np.zeros(count, dtype=np.bool_)
        raw[near_rows.astype(np.int64, copy=False)] = selected
        return raw
    if state == TERMINAL_SCIENTIFIC_STOP_STATUS:
        near_rows = np.asarray(r4_candidate_rows)
        selected = np.asarray(selected_near)
        if near_rows.size or selected.size:
            raise ValueError("R4 scientific STOP must not claim selected rows")
        return np.zeros(count, dtype=np.bool_)
    raise PermissionError("R4 result is neither scientific PASS nor scientific STOP")


def post_terminal_gt_coverage(
    *,
    anchor_best_gt: Any,
    anchor_best_iou: Any,
    candidate_anchor_positions: Any,
    candidate_best_gt: Any,
    candidate_max_gt_iou: Any,
    terminal_selected: Any,
) -> dict[int, float]:
    """Reconstruct GT coverage after R4 OOF geometry replacements.

    R4 preserves anchor row count and permits at most one selected candidate
    per anchor.  A selected candidate replaces that anchor's GT identity/IoU;
    all other anchors retain their all-fold B6 OOF geometry.
    """

    anchor_gt = _strict_int_vector(anchor_best_gt, "anchor best GT").copy()
    anchor_iou = np.asarray(anchor_best_iou, dtype=np.float64).copy()
    owner = _strict_int_vector(candidate_anchor_positions, "candidate anchor positions")
    candidate_gt = _strict_int_vector(candidate_best_gt, "candidate best GT")
    candidate_iou = np.asarray(candidate_max_gt_iou, dtype=np.float64)
    selected = _strict_bool_vector(terminal_selected, "terminal selected")
    n = len(owner)
    if (
        anchor_gt.ndim != 1
        or anchor_iou.shape != anchor_gt.shape
        or owner.shape != (n,)
        or candidate_gt.shape != (n,)
        or candidate_iou.shape != (n,)
        or selected.shape != (n,)
        or np.any((owner < 0) | (owner >= len(anchor_gt)))
        or not np.isfinite(anchor_iou).all()
        or not np.isfinite(candidate_iou).all()
        or np.any((anchor_iou < 0.0) | (anchor_iou > 1.0))
        or np.any((candidate_iou < 0.0) | (candidate_iou > 1.0))
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
        if int(gt_index) >= 0:
            coverage[int(gt_index)] = max(coverage.get(int(gt_index), 0.0), float(iou))
    return coverage


def track_targets(
    *, track_best_gt: Any, track_max_gt_iou: Any,
    post_terminal_coverage: Mapping[int, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label the independent confirmed-track universe after anchor rebuild."""

    best_gt = _strict_int_vector(track_best_gt, "track best GT")
    best_iou = np.asarray(track_max_gt_iou, dtype=np.float64)
    coverage_checked: dict[int, float] = {}
    for key, value in post_terminal_coverage.items():
        key_scalar = np.asarray(key)
        score = float(value)
        if (
            key_scalar.shape != () or key_scalar.dtype.kind not in "iu"
            or int(key) < 0 or not math.isfinite(score) or not 0.0 <= score <= 1.0
        ):
            raise ValueError("post-terminal GT coverage mapping differs")
        coverage_checked[int(key)] = score
    n = len(best_gt)
    if (
        best_iou.shape != (n,)
        or not np.isfinite(best_iou).all()
        or np.any((best_iou < 0.0) | (best_iou > 1.0))
    ):
        raise ValueError("L6 confirmed-track target inputs differ")
    novel25 = np.zeros(n, dtype=np.bool_)
    quality50 = np.zeros(n, dtype=np.bool_)
    novel50 = np.zeros(n, dtype=np.bool_)
    for row in range(n):
        gt_index = int(best_gt[row])
        coverage = float(coverage_checked.get(gt_index, 0.0))
        novel25[row] = (
            gt_index >= 0
            and float(best_iou[row]) >= 0.25
            and coverage < 0.25
        )
        quality50[row] = gt_index >= 0 and float(best_iou[row]) >= 0.50
        novel50[row] = quality50[row] and coverage < 0.50
    for value in (novel25, quality50, novel50):
        value.setflags(write=False)
    return novel25, quality50, novel50


def source_aware_rank(
    *, novelty_probability: float, visibility_quality_mean: float,
    support_ratio_mean: float, free_space_ratio_mean: float,
    selected_geometry: str,
) -> float:
    values = tuple(float(value) for value in (
        novelty_probability, visibility_quality_mean, support_ratio_mean,
        free_space_ratio_mean,
    ))
    probability, visibility, support, free_space = values
    if not all(math.isfinite(value) for value in values):
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
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0.0)


def select_stage6_candidates(
    rows: Sequence[Mapping[str, Any]], novelty_probability: Any,
    quality_probability: Any, *, novelty_threshold: float,
    quality_threshold: float,
) -> tuple[int, ...]:
    """Freeze dual-head admission and the original L6 ordering/NMS semantics.

    The quality head is an admission gate only.  It is deliberately absent
    from ranking so source-aware L6 remains ordered by novelty-derived source
    rank, novelty probability, detector score, then track identity.
    """

    novelty_source = np.asarray(novelty_probability)
    quality_source = np.asarray(quality_probability)
    novelty = np.asarray(novelty_source, dtype=np.float64)
    quality = np.asarray(quality_source, dtype=np.float64)
    n = len(rows)
    thresholds = (novelty_threshold, quality_threshold)
    if (
        novelty_source.dtype.kind == "b" or quality_source.dtype.kind == "b"
        or novelty.shape != (n,) or quality.shape != (n,)
        or not np.isfinite(novelty).all() or not np.isfinite(quality).all()
        or np.any((novelty < 0.0) | (novelty > 1.0))
        or np.any((quality < 0.0) | (quality > 1.0))
        or any(isinstance(value, (bool, np.bool_)) for value in thresholds)
        or not all(math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0 for value in thresholds)
    ):
        raise ValueError("L6 probability rows differ")
    ranked: list[tuple[float, float, float, int, int, np.ndarray]] = []
    identities: set[int] = set()
    for index, row in enumerate(rows):
        track_id = row["track_id"]
        track_scalar = np.asarray(track_id)
        numeric_sources = (
            np.asarray(row["post_terminal_anchor_iou_max"]),
            np.asarray(row["free_space_ratio_mean"]),
            np.asarray(row["best_score"]),
        )
        anchor_iou, free_space, best_score = (
            float(value) for value in numeric_sources
        )
        if (
            track_scalar.shape != () or track_scalar.dtype.kind not in "iu"
            or int(track_id) < 0 or int(track_id) in identities
            or any(value.shape != () or value.dtype.kind == "b" for value in numeric_sources)
            or not all(math.isfinite(value) for value in (anchor_iou, free_space, best_score))
            or not 0.0 <= anchor_iou <= 1.0
            or not 0.0 <= free_space <= 1.0
            or not 0.0 <= best_score <= 1.0
        ):
            raise ValueError("L6 stage6 row identity/numeric contract differs")
        identities.add(int(track_id))
        if novelty[index] < float(novelty_threshold) or quality[index] < float(quality_threshold):
            continue
        if anchor_iou > 0.10:
            continue
        if free_space > 0.45:
            continue
        corners = np.ascontiguousarray(row["selected_corners_world"], dtype=np.float32)
        if corners.shape != (8, 3) or not np.isfinite(corners).all():
            raise ValueError("L6 selected geometry differs")
        rank = source_aware_rank(
            novelty_probability=float(novelty[index]),
            visibility_quality_mean=float(row["visibility_quality_mean"]),
            support_ratio_mean=float(row["support_ratio_mean"]),
            free_space_ratio_mean=float(row["free_space_ratio_mean"]),
            selected_geometry=str(row["selected_geometry"]),
        )
        ranked.append((
            -rank, -float(novelty[index]), -float(row["best_score"]),
            int(track_id), index, corners,
        ))
    ranked.sort(key=lambda item: item[:4])
    kept: list[tuple[float, float, float, int, int, np.ndarray]] = []
    for item in ranked:
        if kept:
            other = np.stack([saved[5] for saved in kept])
            if float(_aabb_iou_one_to_many(item[5], other).max(initial=0.0)) > 0.25:
                continue
        kept.append(item)
        if len(kept) >= 6:
            break
    return tuple(int(item[4]) for item in kept)


def assign_low_scores(
    entries: Sequence[tuple[int, int, float]], anchor_score_floor: float,
) -> dict[tuple[int, int], float]:
    """Assign distinct positive float32 scores below every terminal anchor."""

    floor = float(anchor_score_floor)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("post-terminal anchor score floor must be positive")
    ordered = sorted(entries, key=lambda row: (-float(row[2]), int(row[0]), int(row[1])))
    identities = [(int(scene), int(local)) for scene, local, _ in ordered]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate L6 candidate identity")
    if any(
        scene < 0 or local < 0 or not math.isfinite(float(rank))
        for scene, local, rank in ordered
    ):
        raise ValueError("invalid L6 rank entry")
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
        result[(int(scene), int(local))] = score
        previous = score
    return result


__all__ = [
    "DATASET_SCHEMA", "FEATURE_FORMULAS", "FEATURE_NAMES", "FIT_FOLDS",
    "INCREMENTAL_OBSERVER_CONFIG",
    "INCREMENTAL_PROVIDER_CONFIG", "LIGHTWEIGHT_FUSION_CONFIG", "NAMESPACE",
    "OOF_PLAN", "OOF_SCHEMA",
    "PENDING_SCHEMA", "POLICY_SCHEMA", "PROTOCOL_SCHEMA",
    "R4_PROTOCOL_SCHEMA", "R4_PROTOCOL_SHA256", "R4_RUN_SCHEMA",
    "R4_STOP_SCHEMA", "R6_PREREGISTRATION_SCHEMA",
    "R6_PREREGISTRATION_SHA256", "REUSED_DIAGNOSTIC_FOLDS", "SCORE_POLICY",
    "SOURCE_RANK_FORMULA", "TERMINAL_PASS_STATUS",
    "TERMINAL_SCIENTIFIC_STOP_STATUS", "THRESHOLD_SCHEMA",
    "TRAINING_HYPERPARAMETERS", "SELECTION_REQUIREMENTS", "assign_low_scores",
    "post_terminal_gt_coverage", "sha256_file", "track_targets",
    "select_stage6_candidates", "source_aware_rank",
    "terminal_selection_for_state", "threshold_grid",
    "validate_fold234_oof_plan", "validate_r4_gate_oof_scoring",
]
