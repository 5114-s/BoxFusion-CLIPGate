"""Bounded causal probabilistic TSDF and held-out 7D BoxBA.

This module is deliberately training-free and identity-agnostic.  A track's
chronological views are split once: ``views[:-1]`` builds a sparse 5 cm
probabilistic TSDF and fits the candidate, while ``views[-1]`` is used only to
accept that already-frozen candidate or roll back to the supplied baseline.
Consequently held-out pixels can affect ``output_corners`` but can never affect
``fit_candidate_corners``.

The implementation reuses the sealed RGB-D projection/evidence primitives
from :mod:`boxfusion.sam2_tsdf_mv3dis_shadow`.  It does not inspect labels,
scores, predictions, evaluator output, or ground truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from boxfusion import fastsam_openbox_f3_shadow as f3
from boxfusion.sam2_tsdf_mv3dis_shadow import (
    LiftedMaskView,
    _rectangle_mask_iou,
    _surface_free_space_evidence,
)
from boxfusion.target_masklift import robust_yaw_obb


SCHEMA = "boxfusion.probabilistic_tsdf_boxba.v1"
PROTOCOL_ID = "CAUSAL-PI-TSDF-FREESPACE-7D-BOXBA-HELDOUT-V1"
VOXEL_SIZE_M = 0.05
TSDF_TRUNCATION_M = 0.10
BETA_PRIOR_ALPHA = 1.0
BETA_PRIOR_BETA = 1.0
MAX_VOXELS = 8_192
MIN_TRACK_VIEWS = 3
MIN_CONSENSUS_VOXELS = 16
SEARCH_LAYERS = 5
MAX_CENTER_DELTA_M = 0.20
MIN_SIZE_RATIO = 0.70
MAX_SIZE_RATIO = 1.40
MAX_YAW_DELTA_RAD = math.radians(20.0)
HELDOUT_MIN_LOSS_IMPROVEMENT = 0.01
HELDOUT_MIN_DEPTH_CONTAINMENT = 0.45
HELDOUT_MIN_MASK_IOU = 0.10
HELDOUT_MAX_FREE_RATIO = 0.05

_EPS = 1.0e-12
_CORNER_SIGNS = np.asarray(
    (
        (-1.0, -1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, -1.0),
        (1.0, 1.0, 1.0),
    ),
    dtype=np.float64,
)


class PITSDFBoxBAError(ValueError):
    """An input violates the causal PI-TSDF/BoxBA contract."""


@dataclass(frozen=True)
class PITSDFBoxBAConfig:
    """Frozen route constants and held-out acceptance gates."""

    voxel_size_m: float = VOXEL_SIZE_M
    tsdf_truncation_m: float = TSDF_TRUNCATION_M
    beta_prior_alpha: float = BETA_PRIOR_ALPHA
    beta_prior_beta: float = BETA_PRIOR_BETA
    minimum_track_views: int = MIN_TRACK_VIEWS
    minimum_fit_surface_views: int = 2
    minimum_consensus_voxels: int = MIN_CONSENSUS_VOXELS
    maximum_voxels: int = MAX_VOXELS
    minimum_surface_posterior: float = 0.60
    search_layers: int = SEARCH_LAYERS
    maximum_center_delta_m: float = MAX_CENTER_DELTA_M
    minimum_size_ratio: float = MIN_SIZE_RATIO
    maximum_size_ratio: float = MAX_SIZE_RATIO
    maximum_yaw_delta_rad: float = MAX_YAW_DELTA_RAD
    heldout_minimum_loss_improvement: float = HELDOUT_MIN_LOSS_IMPROVEMENT
    heldout_minimum_depth_containment: float = HELDOUT_MIN_DEPTH_CONTAINMENT
    heldout_minimum_mask_iou: float = HELDOUT_MIN_MASK_IOU
    heldout_maximum_free_ratio: float = HELDOUT_MAX_FREE_RATIO

    def __post_init__(self) -> None:
        # These values define the requested protocol and must not silently
        # drift while materializing a run.
        frozen = (
            ("voxel_size_m", self.voxel_size_m, VOXEL_SIZE_M),
            ("tsdf_truncation_m", self.tsdf_truncation_m, TSDF_TRUNCATION_M),
            ("beta_prior_alpha", self.beta_prior_alpha, BETA_PRIOR_ALPHA),
            ("beta_prior_beta", self.beta_prior_beta, BETA_PRIOR_BETA),
            ("maximum_voxels", self.maximum_voxels, MAX_VOXELS),
            ("search_layers", self.search_layers, SEARCH_LAYERS),
            ("maximum_center_delta_m", self.maximum_center_delta_m, MAX_CENTER_DELTA_M),
            ("minimum_size_ratio", self.minimum_size_ratio, MIN_SIZE_RATIO),
            ("maximum_size_ratio", self.maximum_size_ratio, MAX_SIZE_RATIO),
            ("maximum_yaw_delta_rad", self.maximum_yaw_delta_rad, MAX_YAW_DELTA_RAD),
            (
                "heldout_minimum_loss_improvement",
                self.heldout_minimum_loss_improvement,
                HELDOUT_MIN_LOSS_IMPROVEMENT,
            ),
            (
                "heldout_minimum_depth_containment",
                self.heldout_minimum_depth_containment,
                HELDOUT_MIN_DEPTH_CONTAINMENT,
            ),
            ("heldout_minimum_mask_iou", self.heldout_minimum_mask_iou, HELDOUT_MIN_MASK_IOU),
            ("heldout_maximum_free_ratio", self.heldout_maximum_free_ratio, HELDOUT_MAX_FREE_RATIO),
        )
        for name, observed, expected in frozen:
            if isinstance(expected, int):
                valid = not isinstance(observed, bool) and int(observed) == expected
            else:
                valid = math.isfinite(float(observed)) and math.isclose(
                    float(observed), float(expected), rel_tol=0.0, abs_tol=1.0e-15
                )
            if not valid:
                raise PITSDFBoxBAError(f"{name} is frozen to {expected}")
        if self.minimum_track_views < 3:
            raise PITSDFBoxBAError("minimum_track_views must be at least three")
        if self.minimum_fit_surface_views < 2:
            raise PITSDFBoxBAError("minimum_fit_surface_views must be at least two")
        if self.minimum_consensus_voxels < 16:
            raise PITSDFBoxBAError("minimum_consensus_voxels must be at least sixteen")
        if not 0.5 < self.minimum_surface_posterior < 1.0:
            raise PITSDFBoxBAError("minimum_surface_posterior must be in (0.5,1)")


@dataclass(frozen=True)
class _BoxParameters:
    center: np.ndarray
    log_size: np.ndarray
    yaw: float

    @property
    def size(self) -> np.ndarray:
        return np.exp(self.log_size)


def _finite_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        result = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    except (TypeError, ValueError, OverflowError) as error:
        raise PITSDFBoxBAError(f"{name} must be numeric") from error
    if result.shape != shape or not np.isfinite(result).all():
        raise PITSDFBoxBAError(f"{name} must be finite with shape {shape}")
    return result


def _wrap_pi_periodic(value: float) -> float:
    """Wrap a yaw modulo pi because an OBB is unchanged after 180 degrees."""

    return float((value + math.pi * 0.5) % math.pi - math.pi * 0.5)


def _yaw_delta(value: float, reference: float) -> float:
    return _wrap_pi_periodic(value - reference)


def _rotation(yaw: float) -> np.ndarray:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def _corners(parameters: _BoxParameters) -> np.ndarray:
    return (
        parameters.center[None, :]
        + (_CORNER_SIGNS * (parameters.size[None, :] * 0.5))
        @ _rotation(parameters.yaw).T
    )


def _parameters_from_corners(value: object, name: str) -> _BoxParameters:
    corners = _finite_array(value, (8, 3), name)
    center0 = corners.mean(axis=0)
    centered_xy = corners[:, :2] - center0[None, :2]
    covariance = centered_xy.T @ centered_xy / len(corners)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if float(eigenvalues[-1] - eigenvalues[0]) <= 1.0e-12:
        yaw = 0.0
    else:
        axis = eigenvectors[:, -1]
        if axis[0] < 0.0 or (axis[0] == 0.0 and axis[1] < 0.0):
            axis = -axis
        yaw = _wrap_pi_periodic(math.atan2(float(axis[1]), float(axis[0])))
    rotation = _rotation(yaw)
    local = corners @ rotation
    lower, upper = local.min(axis=0), local.max(axis=0)
    size = upper - lower
    if np.any(size <= 1.0e-6):
        raise PITSDFBoxBAError(f"{name} has a degenerate extent")
    local_center = 0.5 * (lower + upper)
    center = local_center @ rotation.T
    return _BoxParameters(center=center, log_size=np.log(size), yaw=yaw)


def _constrain(
    parameters: _BoxParameters,
    baseline: _BoxParameters,
    config: PITSDFBoxBAConfig,
) -> _BoxParameters:
    delta = parameters.center - baseline.center
    norm = float(np.linalg.norm(delta))
    if norm > config.maximum_center_delta_m:
        delta *= config.maximum_center_delta_m / norm
    lower_log = baseline.log_size + math.log(config.minimum_size_ratio)
    upper_log = baseline.log_size + math.log(config.maximum_size_ratio)
    log_size = np.clip(parameters.log_size, lower_log, upper_log)
    delta_yaw = float(
        np.clip(
            _yaw_delta(parameters.yaw, baseline.yaw),
            -config.maximum_yaw_delta_rad,
            config.maximum_yaw_delta_rad,
        )
    )
    return _BoxParameters(
        center=baseline.center + delta,
        log_size=log_size,
        yaw=_wrap_pi_periodic(baseline.yaw + delta_yaw),
    )


def _deterministic_cap(values: np.ndarray, maximum: int) -> np.ndarray:
    if len(values) <= maximum:
        return values
    indices = (np.arange(maximum, dtype=np.int64) * (len(values) - 1)) // (maximum - 1)
    return values[indices]


def _depth_residuals(
    centers_world: np.ndarray,
    view: LiftedMaskView,
) -> tuple[np.ndarray, np.ndarray]:
    """Return measured-minus-query depth and a valid-projection mask."""

    world_to_camera = np.linalg.inv(view.camera_to_world)
    camera = centers_world @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    z = camera[:, 2]
    front = z > 1.0e-4
    safe_z = np.where(front, z, 1.0)
    columns = np.rint(
        view.intrinsic[0, 0] * camera[:, 0] / safe_z + view.intrinsic[0, 2]
    ).astype(np.int64)
    rows = np.rint(
        view.intrinsic[1, 1] * camera[:, 1] / safe_z + view.intrinsic[1, 2]
    ).astype(np.int64)
    inside = (
        front
        & (columns >= 0)
        & (columns < view.mask.shape[1])
        & (rows >= 0)
        & (rows < view.mask.shape[0])
    )
    measured = np.zeros(len(centers_world), dtype=np.float64)
    if np.any(inside):
        selected = np.flatnonzero(inside)
        measured[selected] = view.depth_m[rows[selected], columns[selected]]
    valid = inside & (measured >= 0.10) & (measured <= 6.0)
    return measured - z, valid


def _build_pi_tsdf(
    fit_views: Sequence[LiftedMaskView],
    config: PITSDFBoxBAConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    concatenated = np.concatenate([view.voxel_keys for view in fit_views], axis=0)
    union = np.unique(concatenated, axis=0)
    union_before_cap = int(len(union))
    # Bounding the candidate universe bounds all posterior arrays, not merely
    # the final serialized output.
    union = _deterministic_cap(union, config.maximum_voxels)
    centers = (union.astype(np.float64) + 0.5) * config.voxel_size_m
    alpha = np.full(len(union), config.beta_prior_alpha, dtype=np.float64)
    beta = np.full(len(union), config.beta_prior_beta, dtype=np.float64)
    surface_support = np.zeros(len(union), dtype=np.int16)
    free_support = np.zeros(len(union), dtype=np.int16)
    tsdf_sum = np.zeros(len(union), dtype=np.float64)
    tsdf_weight = np.zeros(len(union), dtype=np.int16)
    for view in fit_views:
        neighbouring = f3._view_neighbourhood_support_tree(union, view.voxel_keys)
        surface, free, occluded = _surface_free_space_evidence(centers, view)
        positive = neighbouring & surface
        negative = free
        surface_support += positive.astype(np.int16)
        free_support += negative.astype(np.int16)
        alpha += positive.astype(np.float64)
        beta += negative.astype(np.float64)

        residual, valid = _depth_residuals(centers, view)
        # Preserve the sealed evidence categories: unknown projections receive
        # no TSDF weight, and occluded samples provide signed distance but not
        # a negative Beta update.
        classified = valid & (surface | free | occluded)
        tsdf_sum[classified] += np.clip(
            residual[classified] / config.tsdf_truncation_m, -1.0, 1.0
        )
        tsdf_weight += classified.astype(np.int16)
    posterior = alpha / (alpha + beta)
    mean_tsdf = np.divide(
        tsdf_sum,
        tsdf_weight,
        out=np.zeros(len(union), dtype=np.float64),
        where=tsdf_weight > 0,
    )
    keep = (
        (surface_support >= config.minimum_fit_surface_views)
        & (posterior >= config.minimum_surface_posterior)
        & (free_support == 0)
        & (np.abs(mean_tsdf) <= 0.5)
    )
    retained_keys = union[keep]
    retained_alpha = alpha[keep]
    retained_beta = beta[keep]
    retained_posterior = posterior[keep]
    retained_tsdf = mean_tsdf[keep]
    retained_points = (retained_keys.astype(np.float64) + 0.5) * config.voxel_size_m
    state_hash = hashlib.sha256()
    for value, dtype in (
        (retained_keys, "<i8"),
        (retained_alpha, "<f8"),
        (retained_beta, "<f8"),
        (retained_tsdf, "<f8"),
    ):
        state_hash.update(np.ascontiguousarray(value, dtype=dtype).tobytes())
    receipt = {
        "union_voxel_count_before_cap": union_before_cap,
        "union_voxel_count": int(len(union)),
        "consensus_voxel_count": int(len(retained_keys)),
        "surface_support_histogram": {
            str(index): int(np.count_nonzero(surface_support == index))
            for index in range(len(fit_views) + 1)
        },
        "free_supported_voxel_count": int(np.count_nonzero(free_support)),
        "free_rejected_voxel_count": int(np.count_nonzero(free_support > 0)),
        "tsdf_rejected_voxel_count": int(np.count_nonzero(np.abs(mean_tsdf) > 0.5)),
        "posterior_min": float(retained_posterior.min()) if len(retained_posterior) else None,
        "posterior_median": (
            float(np.median(retained_posterior)) if len(retained_posterior) else None
        ),
        "posterior_max": float(retained_posterior.max()) if len(retained_posterior) else None,
        "mean_tsdf": float(retained_tsdf.mean()) if len(retained_tsdf) else None,
        "state_sha256": state_hash.hexdigest(),
    }
    return retained_points, receipt


def _contains(points: np.ndarray, parameters: _BoxParameters, margin: float) -> float:
    if not len(points):
        return 0.0
    local = (points - parameters.center[None, :]) @ _rotation(parameters.yaw)
    inside = np.all(
        np.abs(local) <= parameters.size[None, :] * 0.5 + margin,
        axis=1,
    )
    return float(np.mean(inside))


def _surface_samples(parameters: _BoxParameters) -> np.ndarray:
    levels = np.asarray((-0.5, 0.0, 0.5), dtype=np.float64)
    samples: list[tuple[float, float, float]] = []
    for fixed_axis in range(3):
        variable = [axis for axis in range(3) if axis != fixed_axis]
        for fixed in (-0.5, 0.5):
            for first in levels:
                for second in levels:
                    point = [0.0, 0.0, 0.0]
                    point[fixed_axis] = fixed
                    point[variable[0]] = float(first)
                    point[variable[1]] = float(second)
                    samples.append((point[0], point[1], point[2]))
    local = np.asarray(samples, dtype=np.float64) * parameters.size[None, :]
    return parameters.center[None, :] + local @ _rotation(parameters.yaw).T


def _free_ratio(parameters: _BoxParameters, view: LiftedMaskView) -> float:
    surface, free, occluded = _surface_free_space_evidence(
        _surface_samples(parameters), view
    )
    classified = surface | free | occluded
    count = int(np.count_nonzero(classified))
    return 0.0 if count == 0 else float(np.count_nonzero(free) / count)


def _view_metrics(parameters: _BoxParameters, view: LiftedMaskView) -> dict[str, float]:
    return {
        "mask_iou": _rectangle_mask_iou(_corners(parameters), view),
        "depth_containment": _contains(view.points_world, parameters, VOXEL_SIZE_M * 0.5),
        "free_ratio": _free_ratio(parameters, view),
    }


def _fit_metrics(
    parameters: _BoxParameters,
    fit_views: Sequence[LiftedMaskView],
    consensus_points: np.ndarray,
) -> dict[str, Any]:
    rows = [_view_metrics(parameters, view) for view in fit_views]
    mask = float(np.median([row["mask_iou"] for row in rows]))
    depth = float(np.median([row["depth_containment"] for row in rows]))
    free = float(np.median([row["free_ratio"] for row in rows]))
    consensus = _contains(consensus_points, parameters, VOXEL_SIZE_M * 0.5)
    loss = 0.40 * (1.0 - consensus) + 0.25 * (1.0 - depth) + 0.30 * (1.0 - mask) + 0.05 * free
    return {
        "loss": float(loss),
        "consensus_containment": consensus,
        "mask_iou_median": mask,
        "depth_containment_median": depth,
        "free_ratio_median": free,
        "per_view": rows,
    }


def _fit_only_reference(
    fit_views: Sequence[LiftedMaskView],
    consensus_points: np.ndarray,
    boxer_corners_by_source: Mapping[str, object],
) -> tuple[str, _BoxParameters]:
    """Choose a search anchor without reading the held-out-derived baseline."""

    candidates: list[tuple[float, str, _BoxParameters]] = []
    for ordinal, view in enumerate(fit_views):
        corners = _extract_boxer_corners(
            boxer_corners_by_source.get(view.source_id), view.source_id
        )
        if corners is None:
            continue
        parameters = _parameters_from_corners(corners, f"boxer[{view.source_id}]")
        loss = float(_fit_metrics(parameters, fit_views, consensus_points)["loss"])
        candidates.append((loss, f"boxer:{ordinal}:{view.source_id}", parameters))
    if candidates:
        _, name, parameters = min(candidates, key=lambda row: (row[0], row[1]))
        return name, parameters

    centered = consensus_points[:, :2] - np.median(
        consensus_points[:, :2], axis=0, keepdims=True
    )
    covariance = centered.T @ centered / max(len(centered), 1)
    _, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, -1]
    if axis[0] < 0.0 or (axis[0] == 0.0 and axis[1] < 0.0):
        axis = -axis
    yaw = _wrap_pi_periodic(math.atan2(float(axis[1]), float(axis[0])))
    robust = robust_yaw_obb(consensus_points, yaw_rad=yaw)
    return (
        "pi_tsdf_pca",
        _BoxParameters(
            center=np.asarray(robust.center, dtype=np.float64),
            log_size=np.log(np.maximum(robust.extent, 1.0e-4)),
            yaw=float(robust.yaw_rad),
        ),
    )


def _extract_boxer_corners(raw: object, source_id: str) -> np.ndarray | None:
    if raw is None:
        return None
    value = raw.get("world_corners") if isinstance(raw, Mapping) else raw
    try:
        return _finite_array(value, (8, 3), f"boxer[{source_id}]")
    except PITSDFBoxBAError:
        # A shadow Boxer abstention is not an input-contract failure.
        if isinstance(raw, Mapping) and raw.get("valid") is not True:
            return None
        raise


def _initial_hypotheses(
    fit_views: Sequence[LiftedMaskView],
    consensus_points: np.ndarray,
    boxer_corners_by_source: Mapping[str, object],
    baseline: _BoxParameters,
    config: PITSDFBoxBAConfig,
) -> list[tuple[str, _BoxParameters]]:
    hypotheses: list[tuple[str, _BoxParameters]] = [("baseline", baseline)]
    centered = consensus_points[:, :2] - np.median(
        consensus_points[:, :2], axis=0, keepdims=True
    )
    covariance = centered.T @ centered / max(len(centered), 1)
    _, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, -1]
    if axis[0] < 0.0 or (axis[0] == 0.0 and axis[1] < 0.0):
        axis = -axis
    pca_yaw = _wrap_pi_periodic(math.atan2(float(axis[1]), float(axis[0])))
    for name, yaw in (("pi_tsdf_pca", pca_yaw), ("pi_tsdf_baseline_yaw", baseline.yaw)):
        robust = robust_yaw_obb(consensus_points, yaw_rad=yaw)
        size = np.maximum(robust.extent, 1.0e-4)
        parameters = _BoxParameters(
            center=np.asarray(robust.center, dtype=np.float64),
            log_size=np.log(size),
            yaw=float(robust.yaw_rad),
        )
        hypotheses.append((name, _constrain(parameters, baseline, config)))
    # Only fitting sources are inspected.  A mapping entry for the held-out
    # source is intentionally unreachable from candidate generation.
    for ordinal, view in enumerate(fit_views):
        corners = _extract_boxer_corners(
            boxer_corners_by_source.get(view.source_id), view.source_id
        )
        if corners is None:
            continue
        parameters = _parameters_from_corners(corners, f"boxer[{view.source_id}]")
        hypotheses.append(
            (f"boxer:{ordinal}:{view.source_id}", _constrain(parameters, baseline, config))
        )
    return hypotheses


def _search(
    fit_views: Sequence[LiftedMaskView],
    consensus_points: np.ndarray,
    hypotheses: Sequence[tuple[str, _BoxParameters]],
    baseline: _BoxParameters,
    config: PITSDFBoxBAConfig,
) -> tuple[_BoxParameters, dict[str, Any]]:
    evaluated: list[tuple[float, str, _BoxParameters, dict[str, Any]]] = []
    evaluation_count = 0
    for name, parameters in hypotheses:
        metrics = _fit_metrics(parameters, fit_views, consensus_points)
        evaluation_count += 1
        evaluated.append((float(metrics["loss"]), name, parameters, metrics))
    _, initial_name, current, current_metrics = min(
        evaluated, key=lambda row: (row[0], row[1])
    )
    center_steps = (0.10, 0.05, 0.025, 0.0125, 0.00625)
    log_size_steps = tuple(math.log(value) for value in (1.20, 1.10, 1.05, 1.025, 1.0125))
    yaw_steps = tuple(math.radians(value) for value in (10.0, 5.0, 2.5, 1.25, 0.625))
    layer_receipts: list[dict[str, Any]] = []
    for layer in range(config.search_layers):
        before = float(current_metrics["loss"])
        for dimension in range(7):
            step = (
                center_steps[layer]
                if dimension < 3
                else log_size_steps[layer]
                if dimension < 6
                else yaw_steps[layer]
            )
            choices: list[tuple[float, int, _BoxParameters, dict[str, Any]]] = []
            for direction_order, direction in enumerate((0.0, -1.0, 1.0)):
                center = current.center.copy()
                log_size = current.log_size.copy()
                yaw = current.yaw
                if dimension < 3:
                    center[dimension] += direction * step
                elif dimension < 6:
                    log_size[dimension - 3] += direction * step
                else:
                    yaw += direction * step
                candidate = _constrain(
                    _BoxParameters(center=center, log_size=log_size, yaw=yaw),
                    baseline,
                    config,
                )
                metrics = _fit_metrics(candidate, fit_views, consensus_points)
                evaluation_count += 1
                choices.append(
                    (float(metrics["loss"]), direction_order, candidate, metrics)
                )
            _, _, current, current_metrics = min(
                choices, key=lambda row: (row[0], row[1])
            )
        layer_receipts.append(
            {
                "layer": layer,
                "loss_before": before,
                "loss_after": float(current_metrics["loss"]),
                "center_step_m": center_steps[layer],
                "log_size_step": log_size_steps[layer],
                "yaw_step_rad": yaw_steps[layer],
            }
        )
    return current, {
        "initial_hypothesis": initial_name,
        "initial_hypothesis_count": len(hypotheses),
        "layers": layer_receipts,
        "evaluation_count": evaluation_count,
        "final_fit_metrics": current_metrics,
    }


def _heldout_metrics(parameters: _BoxParameters, heldout: LiftedMaskView) -> dict[str, float]:
    row = _view_metrics(parameters, heldout)
    row["loss"] = float(
        0.45 * (1.0 - row["mask_iou"])
        + 0.45 * (1.0 - row["depth_containment"])
        + 0.10 * row["free_ratio"]
    )
    return row


def _base_receipt(
    *,
    fit_views: Sequence[LiftedMaskView],
    heldout: LiftedMaskView,
    config: PITSDFBoxBAConfig,
    pi_tsdf: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "config": asdict(config),
        "causal_split": {
            "fit_source_ids": [view.source_id for view in fit_views],
            "fit_frame_ids": [view.frame_id for view in fit_views],
            "heldout_source_id": heldout.source_id,
            "heldout_frame_id": heldout.frame_id,
            "fit_is_ordered_prefix": True,
        },
        "pi_tsdf": dict(pi_tsdf),
        "contracts": {
            "ground_truth_access": False,
            "semantic_or_clip_access": False,
            "score_access": False,
            "training": False,
            "online_learning": False,
            "past_only_fit": True,
            "heldout_used_for_fit": False,
            "rollback_on_gate_failure": True,
        },
    }


def refine_causal_track(
    *,
    views: Sequence[LiftedMaskView],
    boxer_corners_by_source: Mapping[str, object],
    baseline_corners: object,
    config: PITSDFBoxBAConfig = PITSDFBoxBAConfig(),
) -> dict[str, Any]:
    """Fit a bounded PI-TSDF 7D box and accept it on one later view.

    ``fit_candidate_corners`` and ``consensus_points`` depend exclusively on
    ``views[:-1]``, the baseline, fitting-source Boxer hypotheses, and config.
    The last view is read only after candidate generation.  A failed gate
    returns byte-equivalent baseline values in ``output_corners``.
    """

    if not isinstance(config, PITSDFBoxBAConfig):
        raise PITSDFBoxBAError("config must be PITSDFBoxBAConfig")
    ordered = tuple(views)
    if len(ordered) < config.minimum_track_views:
        raise PITSDFBoxBAError(
            f"at least {config.minimum_track_views} chronological views are required"
        )
    if any(not isinstance(view, LiftedMaskView) for view in ordered):
        raise PITSDFBoxBAError("every view must be LiftedMaskView")
    frame_ids = [view.frame_id for view in ordered]
    if frame_ids != sorted(frame_ids) or len(set(frame_ids)) != len(frame_ids):
        raise PITSDFBoxBAError("views must have distinct chronological frame_ids")
    if not isinstance(boxer_corners_by_source, Mapping):
        raise PITSDFBoxBAError("boxer_corners_by_source must be a mapping")
    baseline_array = _finite_array(baseline_corners, (8, 3), "baseline_corners")
    baseline = _parameters_from_corners(baseline_array, "baseline_corners")
    fit_views, heldout = ordered[:-1], ordered[-1]
    if len(fit_views) < config.minimum_fit_surface_views:
        raise PITSDFBoxBAError("too few fitting views for two-view consensus")

    consensus_points, pi_receipt = _build_pi_tsdf(fit_views, config)
    receipt = _base_receipt(
        fit_views=fit_views,
        heldout=heldout,
        config=config,
        pi_tsdf=pi_receipt,
    )
    if len(consensus_points) < config.minimum_consensus_voxels:
        receipt["acceptance"] = {
            "accepted": False,
            "reason": "too_few_consensus_voxels",
        }
        return {
            "fit_candidate_corners": baseline_array.tolist(),
            "output_corners": baseline_array.tolist(),
            "accepted": False,
            "reason": "too_few_consensus_voxels",
            "receipt": receipt,
            "consensus_points": consensus_points.tolist(),
        }

    reference_name, fit_reference = _fit_only_reference(
        fit_views,
        consensus_points,
        boxer_corners_by_source,
    )
    hypotheses = _initial_hypotheses(
        fit_views,
        consensus_points,
        boxer_corners_by_source,
        fit_reference,
        config,
    )
    candidate, search_receipt = _search(
        fit_views, consensus_points, hypotheses, fit_reference, config
    )
    candidate_corners = _corners(candidate)
    candidate_hash = hashlib.sha256(
        np.ascontiguousarray(candidate_corners, dtype="<f8").tobytes()
    ).hexdigest()
    receipt["boxba"] = {
        **search_receipt,
        "fit_candidate_sha256": candidate_hash,
        "fit_reference": reference_name,
        "center_delta_m": float(np.linalg.norm(candidate.center - fit_reference.center)),
        "size_ratios": (candidate.size / fit_reference.size).tolist(),
        "yaw_delta_rad": _yaw_delta(candidate.yaw, fit_reference.yaw),
    }

    # The held-out view is first consumed here, after the candidate and its
    # hash have been frozen.
    baseline_heldout = _heldout_metrics(baseline, heldout)
    candidate_heldout = _heldout_metrics(candidate, heldout)
    improvement = float(baseline_heldout["loss"] - candidate_heldout["loss"])
    output_center_delta = float(np.linalg.norm(candidate.center - baseline.center))
    output_size_ratios = candidate.size / baseline.size
    output_yaw_delta = abs(_yaw_delta(candidate.yaw, baseline.yaw))
    checks = {
        "loss_improves_by_0.01": improvement + _EPS >= config.heldout_minimum_loss_improvement,
        "depth_containment_at_least_0.45": (
            candidate_heldout["depth_containment"] + _EPS
            >= config.heldout_minimum_depth_containment
        ),
        "mask_iou_at_least_0.10": (
            candidate_heldout["mask_iou"] + _EPS >= config.heldout_minimum_mask_iou
        ),
        "mask_iou_does_not_decrease": (
            candidate_heldout["mask_iou"] + _EPS >= baseline_heldout["mask_iou"]
        ),
        "free_ratio_at_most_0.05": (
            candidate_heldout["free_ratio"] <= config.heldout_maximum_free_ratio + _EPS
        ),
        "free_ratio_does_not_increase": (
            candidate_heldout["free_ratio"] <= baseline_heldout["free_ratio"] + _EPS
        ),
        "output_center_within_0.20m": (
            output_center_delta <= config.maximum_center_delta_m + _EPS
        ),
        "output_size_ratio_within_bounds": bool(
            np.all(output_size_ratios >= config.minimum_size_ratio - _EPS)
            and np.all(output_size_ratios <= config.maximum_size_ratio + _EPS)
        ),
        "output_yaw_within_20deg": (
            output_yaw_delta <= config.maximum_yaw_delta_rad + _EPS
        ),
    }
    reason_by_check = (
        ("loss_improves_by_0.01", "heldout_loss_improvement_below_0.01"),
        ("depth_containment_at_least_0.45", "heldout_depth_containment_below_0.45"),
        ("mask_iou_at_least_0.10", "heldout_mask_iou_below_0.10"),
        ("mask_iou_does_not_decrease", "heldout_mask_iou_decreased"),
        ("free_ratio_at_most_0.05", "heldout_free_ratio_above_0.05"),
        ("free_ratio_does_not_increase", "heldout_free_ratio_increased"),
        ("output_center_within_0.20m", "output_center_delta_above_0.20m"),
        ("output_size_ratio_within_bounds", "output_size_ratio_outside_bounds"),
        ("output_yaw_within_20deg", "output_yaw_delta_above_20deg"),
    )
    accepted = all(checks.values())
    reason = "accepted" if accepted else next(
        reason for check, reason in reason_by_check if not checks[check]
    )
    receipt["acceptance"] = {
        "accepted": accepted,
        "reason": reason,
        "baseline": baseline_heldout,
        "candidate": candidate_heldout,
        "loss_improvement": improvement,
        "output_safety": {
            "center_delta_m": output_center_delta,
            "size_ratios": output_size_ratios.tolist(),
            "yaw_delta_rad": output_yaw_delta,
        },
        "checks": checks,
    }
    return {
        "fit_candidate_corners": candidate_corners.tolist(),
        "output_corners": (
            candidate_corners.tolist() if accepted else baseline_array.tolist()
        ),
        "accepted": accepted,
        "reason": reason,
        "receipt": receipt,
        "consensus_points": consensus_points.tolist(),
    }


def policy_receipt(config: PITSDFBoxBAConfig = PITSDFBoxBAConfig()) -> dict[str, Any]:
    """Return the static protocol receipt without consuming observations."""

    return {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "config": asdict(config),
        "fit_views": "ordered[:-1]",
        "heldout_view": "ordered[-1]",
        "parameterization": "7D(cx,cy,cz,log_sx,log_sy,log_sz,yaw)",
        "contracts": {
            "ground_truth_access": False,
            "training": False,
            "online_learning": False,
            "past_only": True,
            "deterministic": True,
        },
    }
