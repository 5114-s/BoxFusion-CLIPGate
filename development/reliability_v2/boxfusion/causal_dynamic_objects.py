"""Bounded causal state for dynamic 3D objects.

This module is deliberately independent of the current BoxFusion pipeline.  It
contains no learned model and performs no image inference: callers provide 7D
box observations, optional cached appearance descriptors, and a conservative
visibility verdict.  The state manager then supplies:

* constant-velocity prediction and short-window static/dynamic box updates;
* deterministic, null-aware one-to-one association;
* bounded appearance prototypes and observation history;
* visibility-conditioned lifecycle updates (occlusion is not negative evidence);
* relocation/reactivation and separate persistent/current object views; and
* an exact-token query-before-commit protocol.

The 7D box convention is ``(cx, cy, cz, sx, sy, sz, yaw)`` in world space.
Sizes must be positive and yaw is represented in radians.  Computation stays
on the CPU (NumPy plus SciPy's assignment solver when available; a deterministic
NumPy fallback is retained) and all externally exposed arrays are immutable
copies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _scipy_linear_sum_assignment
except Exception:  # pragma: no cover - the deterministic NumPy fallback is tested.
    _scipy_linear_sum_assignment = None


BOX_DIMENSION = 7
VELOCITY_DIMENSION = 4  # vx, vy, vz, yaw_rate
SCHEMA = "boxfusion.causal_dynamic_objects.v1"


class DynamicObjectContractError(ValueError):
    """Raised before mutation when the causal-state contract is violated."""


class Visibility(str, Enum):
    """Caller-supplied visibility of an existing track in the current frame."""

    EXPECTED_VISIBLE = "expected_visible"
    OCCLUDED = "occluded"
    OUT_OF_VIEW = "out_of_view"
    UNKNOWN = "unknown"


class Lifecycle(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    OCCLUDED = "occluded"
    DORMANT = "dormant"
    RETIRED = "retired"


class MotionState(str, Enum):
    STATIC = "static"
    UNCERTAIN = "uncertain"
    DYNAMIC = "dynamic"


@dataclass(frozen=True)
class DynamicObjectConfig:
    """Configuration with hard memory and per-frame compute bounds."""

    enabled: bool = False
    # Ablation only: disable *all* miss-driven score/lifecycle updates, including
    # age/coast retirement. Association, hit updates and memory bounds remain.
    miss_lifecycle_updates: bool = True
    max_tracks: int = 256
    max_observations_per_frame: int = 64
    history_size: int = 5
    max_appearance_prototypes: int = 4
    association_candidate_top_k: int = 4
    min_confirmed_hits: int = 2
    dormant_after_visible_misses: int = 2
    retire_after_visible_misses: int = 3
    max_unobserved_coast_frames: int = 3
    max_reactivation_age: int = 30
    max_center_distance_m: float = 1.20
    max_relocation_distance_m: float = 4.00
    max_mean_log_size_residual: float = 0.70
    max_yaw_residual_rad: float = math.pi * 0.75
    min_reactivation_appearance: float = 0.72
    null_track_cost: float = 0.34
    null_observation_cost: float = 0.34
    weight_center: float = 0.42
    weight_size: float = 0.18
    weight_yaw: float = 0.08
    weight_appearance: float = 0.32
    velocity_update_rate: float = 0.65
    dynamic_probability_update_rate: float = 0.45
    dynamic_speed_low_m_per_frame: float = 0.025
    dynamic_speed_high_m_per_frame: float = 0.10
    dynamic_probability_threshold: float = 0.60
    static_probability_threshold: float = 0.35
    dynamic_measurement_gain: float = 0.80
    current_hit_momentum: float = 0.90
    visible_miss_score_decay: float = 0.55
    current_output_threshold: float = 0.05
    relocation_distance_m: float = 0.75
    prototype_merge_cosine: float = 0.96

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, (bool, np.bool_)):
            raise DynamicObjectContractError("enabled must be boolean")
        if not isinstance(self.miss_lifecycle_updates, (bool, np.bool_)):
            raise DynamicObjectContractError("miss_lifecycle_updates must be boolean")
        for name in (
            "max_tracks",
            "max_observations_per_frame",
            "history_size",
            "max_appearance_prototypes",
            "association_candidate_top_k",
            "min_confirmed_hits",
            "dormant_after_visible_misses",
            "retire_after_visible_misses",
            "max_unobserved_coast_frames",
            "max_reactivation_age",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ) or value < 1:
                raise DynamicObjectContractError(f"{name} must be a positive integer")
        if self.history_size > 5:
            raise DynamicObjectContractError("history_size must be <= 5")
        if self.max_appearance_prototypes > 4:
            raise DynamicObjectContractError(
                "max_appearance_prototypes must be <= 4"
            )
        if self.association_candidate_top_k > 16:
            raise DynamicObjectContractError(
                "association_candidate_top_k must be <= 16"
            )
        if self.retire_after_visible_misses < self.dormant_after_visible_misses:
            raise DynamicObjectContractError(
                "retire_after_visible_misses must be >= dormant threshold"
            )
        if self.max_unobserved_coast_frames > self.max_reactivation_age:
            raise DynamicObjectContractError(
                "max_unobserved_coast_frames must be <= max_reactivation_age"
            )
        for name in (
            "max_center_distance_m",
            "max_relocation_distance_m",
            "max_mean_log_size_residual",
            "max_yaw_residual_rad",
            "null_track_cost",
            "null_observation_cost",
            "dynamic_speed_low_m_per_frame",
            "dynamic_speed_high_m_per_frame",
            "relocation_distance_m",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise DynamicObjectContractError(f"{name} must be finite and positive")
        if self.max_relocation_distance_m < self.max_center_distance_m:
            raise DynamicObjectContractError(
                "max_relocation_distance_m must not be smaller than normal gate"
            )
        if self.dynamic_speed_high_m_per_frame <= self.dynamic_speed_low_m_per_frame:
            raise DynamicObjectContractError(
                "dynamic speed high threshold must exceed low threshold"
            )
        for name in (
            "min_reactivation_appearance",
            "velocity_update_rate",
            "dynamic_probability_update_rate",
            "dynamic_probability_threshold",
            "static_probability_threshold",
            "dynamic_measurement_gain",
            "current_hit_momentum",
            "visible_miss_score_decay",
            "current_output_threshold",
            "prototype_merge_cosine",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise DynamicObjectContractError(f"{name} must be in [0, 1]")
        if self.static_probability_threshold >= self.dynamic_probability_threshold:
            raise DynamicObjectContractError(
                "static probability threshold must be below dynamic threshold"
            )
        weights = (
            self.weight_center,
            self.weight_size,
            self.weight_yaw,
            self.weight_appearance,
        )
        if any(not np.isfinite(row) or row < 0.0 for row in weights):
            raise DynamicObjectContractError("association weights must be non-negative")
        if not np.isclose(sum(weights), 1.0, atol=1.0e-9):
            raise DynamicObjectContractError("association weights must sum to 1")


def resolve_causal_dynamic_object_config(cfg: Mapping | None) -> DynamicObjectConfig:
    """Resolve ``cfg['dynamic_objects']`` without enabling it by default."""

    if cfg is None:
        return DynamicObjectConfig()
    if not isinstance(cfg, Mapping):
        raise DynamicObjectContractError("config must be a mapping")
    section = cfg.get("dynamic_objects", {})
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        raise DynamicObjectContractError("dynamic_objects must be a mapping")
    known = set(DynamicObjectConfig.__dataclass_fields__)
    unknown = sorted(set(section) - known)
    if unknown:
        raise DynamicObjectContractError(
            f"unknown dynamic_objects option(s): {', '.join(unknown)}"
        )
    return DynamicObjectConfig(**dict(section))


def _readonly_vector(value, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,):
        raise DynamicObjectContractError(f"{name} must have shape ({length},)")
    if not np.all(np.isfinite(result)):
        raise DynamicObjectContractError(f"{name} must contain only finite values")
    result = np.array(result, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _normalized_vector(value, length: int | None, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or (length is not None and result.shape != (length,)):
        suffix = "a vector" if length is None else f"shape ({length},)"
        raise DynamicObjectContractError(f"{name} must have {suffix}")
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise DynamicObjectContractError(f"{name} must be a non-empty finite vector")
    norm = float(np.linalg.norm(result))
    if norm <= 1.0e-12:
        raise DynamicObjectContractError(f"{name} must have non-zero norm")
    result = np.array(result / norm, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _validate_box(box7) -> np.ndarray:
    result = _readonly_vector(box7, BOX_DIMENSION, "box7")
    if np.any(result[3:6] <= 0.0):
        raise DynamicObjectContractError("box7 sizes must be positive")
    normalized = np.array(result, copy=True)
    normalized[6] = _wrap_angle(float(normalized[6]))
    normalized.setflags(write=False)
    return normalized


def _strict_frame(value: object, name: str = "frame_ordinal") -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise DynamicObjectContractError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise DynamicObjectContractError(f"{name} must be non-negative")
    return result


def _nonempty_id(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise DynamicObjectContractError(f"{name} must be a non-empty string")
    return value


def _wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def _angle_delta(target: float, source: float) -> float:
    return _wrap_angle(target - source)


def _blend_angle(left: float, right: float, right_weight: float) -> float:
    return _wrap_angle(left + right_weight * _angle_delta(right, left))


@dataclass(frozen=True)
class BoxObservation:
    """One current-frame observation; descriptors are optional cached features."""

    observation_id: str
    box7: np.ndarray
    score: float
    appearance: np.ndarray | None = None
    view_direction: np.ndarray | None = None
    appearance_quality: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observation_id", _nonempty_id(self.observation_id, "observation_id")
        )
        object.__setattr__(self, "box7", _validate_box(self.box7))
        score = float(self.score)
        quality = float(self.appearance_quality)
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise DynamicObjectContractError("score must be in [0, 1]")
        if not np.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise DynamicObjectContractError("appearance_quality must be in [0, 1]")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "appearance_quality", quality)
        if self.appearance is not None:
            object.__setattr__(
                self,
                "appearance",
                _normalized_vector(self.appearance, None, "appearance"),
            )
        if self.view_direction is not None:
            object.__setattr__(
                self,
                "view_direction",
                _normalized_vector(self.view_direction, 3, "view_direction"),
            )


@dataclass(frozen=True)
class TrackVisibility:
    track_id: str
    visibility: Visibility

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", _nonempty_id(self.track_id, "track_id"))
        try:
            value = Visibility(self.visibility)
        except ValueError as exc:
            raise DynamicObjectContractError("invalid visibility value") from exc
        object.__setattr__(self, "visibility", value)


@dataclass(frozen=True)
class Association:
    observation_id: str
    track_id: str
    cost: float
    center_distance_m: float
    appearance_similarity: float | None
    reactivated: bool
    relocated: bool


@dataclass(frozen=True)
class ObjectSnapshot:
    """Immutable lightweight output row for either map view."""

    track_id: str
    box7: np.ndarray
    score: float
    persistent_score: float
    current_score: float
    lifecycle: Lifecycle
    motion_state: MotionState
    dynamic_probability: float
    velocity4: np.ndarray
    hit_count: int
    consecutive_visible_misses: int
    last_observed_frame: int
    recent_observation_ids: tuple[str, ...]
    appearance_prototype_count: int


@dataclass(frozen=True)
class DynamicFrameQuery:
    frame_ordinal: int
    associations: tuple[Association, ...]
    birth_observation_ids: tuple[str, ...]
    unmatched_track_ids: tuple[str, ...]
    evicted_track_ids: tuple[str, ...]
    capacity_dropped_observation_ids: tuple[str, ...]
    claimed_observation_ids: tuple[str, ...]
    observation_to_track: tuple[tuple[str, str], ...]
    reactivated_track_ids: tuple[str, ...]
    relocated_track_ids: tuple[str, ...]
    memory_version_before: int
    maximum_accessed_frame_ordinal: int
    token: str


@dataclass(frozen=True)
class DynamicFrameCommit:
    frame_ordinal: int
    associations: tuple[Association, ...]
    birth_observation_ids: tuple[str, ...]
    evicted_track_ids: tuple[str, ...]
    claimed_observation_ids: tuple[str, ...]
    observation_to_track: tuple[tuple[str, str], ...]
    reactivated_track_ids: tuple[str, ...]
    relocated_track_ids: tuple[str, ...]
    memory_version_after: int
    live_track_count: int
    persistent_count: int
    current_count: int
    query_before_commit: bool
    token: str


@dataclass(frozen=True)
class _Prototype:
    embedding: np.ndarray
    view_direction: np.ndarray | None
    quality: float
    frame_ordinal: int
    observation_id: str


@dataclass(frozen=True)
class _HistoryRow:
    frame_ordinal: int
    observation_id: str
    box7: np.ndarray
    score: float


@dataclass(frozen=True)
class _Track:
    track_id: str
    current_box7: np.ndarray
    persistent_box7: np.ndarray
    velocity4: np.ndarray
    dynamic_probability: float
    persistent_score: float
    current_score: float
    lifecycle: Lifecycle
    ever_confirmed: bool
    hit_count: int
    consecutive_visible_misses: int
    first_observed_frame: int
    last_observed_frame: int
    inactive_since_frame: int | None
    history: tuple[_HistoryRow, ...]
    prototypes: tuple[_Prototype, ...]


@dataclass(frozen=True)
class _PairMetric:
    valid: bool
    cost: float
    center_distance_m: float
    appearance_similarity: float | None
    predicted_box7: np.ndarray


_APPEARANCE_NOT_PRECOMPUTED = object()


def _copy_readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _motion_state(probability: float, cfg: DynamicObjectConfig) -> MotionState:
    if probability >= cfg.dynamic_probability_threshold:
        return MotionState.DYNAMIC
    if probability <= cfg.static_probability_threshold:
        return MotionState.STATIC
    return MotionState.UNCERTAIN


def _predict_box(
    track: _Track, frame_ordinal: int, cfg: DynamicObjectConfig
) -> np.ndarray:
    result = np.array(track.current_box7, copy=True)
    dt = min(
        max(frame_ordinal - track.last_observed_frame, 0),
        cfg.max_unobserved_coast_frames,
    )
    # A probability-weighted prediction avoids moving a probably-static object
    # merely because one noisy frame produced a non-zero velocity estimate.
    scale = float(track.dynamic_probability) * float(dt)
    result[:3] += track.velocity4[:3] * scale
    result[6] = _wrap_angle(result[6] + track.velocity4[3] * scale)
    result.setflags(write=False)
    return result


def _prototype_similarity(
    track: _Track, observation: BoxObservation
) -> float | None:
    if observation.appearance is None or not track.prototypes:
        return None
    best = -1.0
    found = False
    for prototype in track.prototypes:
        if prototype.embedding.shape != observation.appearance.shape:
            continue
        found = True
        cosine = float(np.clip(prototype.embedding @ observation.appearance, -1.0, 1.0))
        if observation.view_direction is not None and prototype.view_direction is not None:
            view_compatibility = float(
                np.clip(prototype.view_direction @ observation.view_direction, -1.0, 1.0)
            )
            cosine -= 0.10 * (1.0 - (view_compatibility + 1.0) * 0.5)
        best = max(best, cosine)
    return float(np.clip(best, -1.0, 1.0)) if found else None


def _coarse_appearance_similarity(
    tracks: Sequence[_Track], observations: Sequence[BoxObservation]
) -> np.ndarray:
    """Vectorize best-prototype similarity for candidate preselection.

    Descriptor dimensions are handled independently because cached appearance
    encoders can differ across runs.  This mirrors ``_prototype_similarity``,
    including its view-direction penalty, while leaving incomparable pairs as
    ``NaN``.
    """

    track_count = len(tracks)
    observation_count = len(observations)
    best = np.full((track_count, observation_count), -np.inf, dtype=np.float64)
    observations_by_dimension: dict[int, list[int]] = {}
    for observation_index, observation in enumerate(observations):
        if observation.appearance is not None:
            observations_by_dimension.setdefault(
                int(observation.appearance.shape[0]), []
            ).append(observation_index)

    prototypes_by_dimension: dict[int, list[tuple[int, _Prototype]]] = {}
    for track_index, track in enumerate(tracks):
        for prototype in track.prototypes:
            prototypes_by_dimension.setdefault(
                int(prototype.embedding.shape[0]), []
            ).append((track_index, prototype))

    common_dimensions = sorted(
        set(observations_by_dimension) & set(prototypes_by_dimension)
    )
    for dimension in common_dimensions:
        observation_indices = np.asarray(
            observations_by_dimension[dimension], dtype=np.int64
        )
        prototype_rows = prototypes_by_dimension[dimension]
        prototype_track_indices = np.asarray(
            [row[0] for row in prototype_rows], dtype=np.int64
        )
        observation_embeddings = np.stack(
            [observations[int(index)].appearance for index in observation_indices],
            axis=0,
        )
        prototype_embeddings = np.stack(
            [row[1].embedding for row in prototype_rows], axis=0
        )
        similarities = np.clip(
            prototype_embeddings @ observation_embeddings.T, -1.0, 1.0
        )

        prototype_has_view = np.asarray(
            [row[1].view_direction is not None for row in prototype_rows], dtype=bool
        )
        observation_has_view = np.asarray(
            [
                observations[int(index)].view_direction is not None
                for index in observation_indices
            ],
            dtype=bool,
        )
        if np.any(prototype_has_view) and np.any(observation_has_view):
            prototype_views = np.zeros((len(prototype_rows), 3), dtype=np.float64)
            observation_views = np.zeros(
                (len(observation_indices), 3), dtype=np.float64
            )
            for index, (_, prototype) in enumerate(prototype_rows):
                if prototype.view_direction is not None:
                    prototype_views[index] = prototype.view_direction
            for local_index, observation_index in enumerate(observation_indices):
                view = observations[int(observation_index)].view_direction
                if view is not None:
                    observation_views[local_index] = view
            joint_view = prototype_has_view[:, None] & observation_has_view[None, :]
            view_compatibility = np.clip(
                prototype_views @ observation_views.T, -1.0, 1.0
            )
            similarities -= np.where(
                joint_view, 0.05 * (1.0 - view_compatibility), 0.0
            )

        # Multiple prototypes can map to one track.  ``maximum.at`` performs
        # that reduction without a Python loop over track/observation pairs.
        track_grid = np.broadcast_to(
            prototype_track_indices[:, None], similarities.shape
        )
        observation_grid = np.broadcast_to(
            observation_indices[None, :], similarities.shape
        )
        np.maximum.at(best, (track_grid, observation_grid), similarities)

    finite = np.isfinite(best)
    best[finite] = np.clip(best[finite], -1.0, 1.0)
    best[~finite] = np.nan
    return best


def _symmetric_top_k_mask(
    valid_mask: np.ndarray, ranking_cost: np.ndarray, top_k: int
) -> np.ndarray:
    """Select a bounded union of both directed Top-K rankings.

    Each observation selects no more than ``top_k`` tracks and each track
    selects no more than ``top_k`` observations.  The resulting candidate set
    therefore contains at most ``top_k * (tracks + observations)`` edges.
    """

    valid_mask = np.asarray(valid_mask, dtype=bool)
    ranking_cost = np.asarray(ranking_cost, dtype=np.float64)
    if valid_mask.ndim != 2 or ranking_cost.shape != valid_mask.shape:
        raise DynamicObjectContractError("Top-K matrices must have equal 2D shape")
    track_count, observation_count = valid_mask.shape
    selected = np.zeros_like(valid_mask)
    ranked = np.where(valid_mask, ranking_cost, np.inf)

    column_limit = min(top_k, track_count)
    if column_limit:
        # Stable sorting gives deterministic row-index tie breaking.
        column_order = np.argsort(ranked, axis=0, kind="stable")[:column_limit]
        column_indices = np.broadcast_to(
            np.arange(observation_count, dtype=np.int64)[None, :],
            column_order.shape,
        )
        rows = column_order.ravel()
        columns = column_indices.ravel()
        keep = valid_mask[rows, columns]
        selected[rows[keep], columns[keep]] = True

    row_limit = min(top_k, observation_count)
    if row_limit:
        # Stable sorting gives deterministic column-index tie breaking.
        row_order = np.argsort(ranked, axis=1, kind="stable")[:, :row_limit]
        row_indices = np.broadcast_to(
            np.arange(track_count, dtype=np.int64)[:, None], row_order.shape
        )
        rows = row_indices.ravel()
        columns = row_order.ravel()
        keep = valid_mask[rows, columns]
        selected[rows[keep], columns[keep]] = True
    return selected


def _pair_metric(
    track: _Track,
    observation: BoxObservation,
    frame_ordinal: int,
    cfg: DynamicObjectConfig,
    *,
    precomputed_appearance: float | None | object = _APPEARANCE_NOT_PRECOMPUTED,
) -> _PairMetric:
    inactive = track.lifecycle in (Lifecycle.DORMANT, Lifecycle.RETIRED)
    predicted = (
        track.current_box7 if inactive else _predict_box(track, frame_ordinal, cfg)
    )
    distance = float(np.linalg.norm(observation.box7[:3] - predicted[:3]))
    size_residual = float(
        np.mean(np.abs(np.log(observation.box7[3:6] / predicted[3:6])))
    )
    yaw_residual = abs(_angle_delta(observation.box7[6], predicted[6]))
    if precomputed_appearance is _APPEARANCE_NOT_PRECOMPUTED:
        appearance = _prototype_similarity(track, observation)
    elif precomputed_appearance is None:
        appearance = None
    else:
        appearance = float(precomputed_appearance)
    age = frame_ordinal - track.last_observed_frame
    coast_age = min(max(age, 0), cfg.max_unobserved_coast_frames)

    normal_gate = cfg.max_center_distance_m * math.sqrt(max(coast_age, 1))
    valid = (
        distance <= normal_gate
        and size_residual <= cfg.max_mean_log_size_residual
        and yaw_residual <= cfg.max_yaw_residual_rad
    )
    gate_distance = normal_gate
    if inactive:
        normal_reactivation = valid
        appearance_relocation = (
            appearance is not None
            and appearance >= cfg.min_reactivation_appearance
            and distance <= cfg.max_relocation_distance_m
            and size_residual <= cfg.max_mean_log_size_residual
        )
        valid = (
            age <= cfg.max_reactivation_age
            and (normal_reactivation or appearance_relocation)
        )
        gate_distance = (
            cfg.max_relocation_distance_m if appearance_relocation else normal_gate
        )

    center_cost = min(distance / max(gate_distance, 1.0e-9), 1.0)
    size_cost = min(size_residual / cfg.max_mean_log_size_residual, 1.0)
    yaw_cost = min(yaw_residual / cfg.max_yaw_residual_rad, 1.0)
    appearance_cost = 0.5 if appearance is None else (1.0 - appearance) * 0.5
    cost = (
        cfg.weight_center * center_cost
        + cfg.weight_size * size_cost
        + cfg.weight_yaw * yaw_cost
        + cfg.weight_appearance * appearance_cost
    )
    return _PairMetric(valid, float(cost), distance, appearance, predicted)


def _hungarian_min_cost(cost: np.ndarray) -> np.ndarray:
    """Deterministic square assignment with a dependency-free fallback."""

    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise DynamicObjectContractError("Hungarian cost must be square")
    n = cost.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if not np.all(np.isfinite(cost)):
        raise DynamicObjectContractError("Hungarian cost must be finite")
    if _scipy_linear_sum_assignment is not None:
        rows, columns = _scipy_linear_sum_assignment(cost)
        assignment = np.full(n, -1, dtype=np.int64)
        assignment[np.asarray(rows, dtype=np.int64)] = np.asarray(
            columns, dtype=np.int64
        )
        return assignment
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(n + 1, dtype=np.float64)
    p = np.zeros(n + 1, dtype=np.int64)
    way = np.zeros(n + 1, dtype=np.int64)
    eps = 1.0e-12
    for row in range(1, n + 1):
        p[0] = row
        col0 = 0
        minv = np.full(n + 1, np.inf, dtype=np.float64)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[col0] = True
            row0 = p[col0]
            delta = np.inf
            col1 = 0
            for col in range(1, n + 1):
                if used[col]:
                    continue
                current = cost[row0 - 1, col - 1] - u[row0] - v[col]
                if current < minv[col] - eps:
                    minv[col] = current
                    way[col] = col0
                if minv[col] < delta - eps:
                    delta = minv[col]
                    col1 = col
            for col in range(n + 1):
                if used[col]:
                    u[p[col]] += delta
                    v[col] -= delta
                else:
                    minv[col] -= delta
            col0 = col1
            if p[col0] == 0:
                break
        while True:
            col1 = way[col0]
            p[col0] = p[col1]
            col0 = col1
            if col0 == 0:
                break
    assignment = np.full(n, -1, dtype=np.int64)
    for col in range(1, n + 1):
        if p[col] != 0:
            assignment[p[col] - 1] = col - 1
    return assignment


def _null_aware_assignment(
    tracks: Sequence[_Track],
    observations: Sequence[BoxObservation],
    frame_ordinal: int,
    cfg: DynamicObjectConfig,
) -> tuple[list[tuple[int, int, _PairMetric]], set[int], set[int]]:
    track_count = len(tracks)
    observation_count = len(observations)
    if track_count == 0 or observation_count == 0:
        return [], set(range(track_count)), set(range(observation_count))

    predicted_boxes = np.stack(
        [
            track.current_box7
            if track.lifecycle in (Lifecycle.DORMANT, Lifecycle.RETIRED)
            else _predict_box(track, frame_ordinal, cfg)
            for track in tracks
        ],
        axis=0,
    )
    observation_boxes = np.stack([row.box7 for row in observations], axis=0)
    center_delta = (
        observation_boxes[None, :, :3] - predicted_boxes[:, None, :3]
    )
    center_distance = np.linalg.norm(center_delta, axis=2)
    log_size_residual = np.mean(
        np.abs(
            np.log(
                observation_boxes[None, :, 3:6]
                / predicted_boxes[:, None, 3:6]
            )
        ),
        axis=2,
    )
    yaw_delta = (
        observation_boxes[None, :, 6]
        - predicted_boxes[:, None, 6]
        + math.pi
    ) % (2.0 * math.pi) - math.pi
    yaw_residual = np.abs(yaw_delta)
    ages = np.asarray(
        [frame_ordinal - track.last_observed_frame for track in tracks],
        dtype=np.int64,
    )
    coast_ages = np.minimum(
        np.maximum(ages, 0), cfg.max_unobserved_coast_frames
    )
    normal_distance_gate = cfg.max_center_distance_m * np.sqrt(
        np.maximum(coast_ages, 1)
    )
    epsilon = 1.0e-12
    size_valid = log_size_residual <= cfg.max_mean_log_size_residual + epsilon
    normal_valid = (
        (center_distance <= normal_distance_gate[:, None] + epsilon)
        & size_valid
        & (yaw_residual <= cfg.max_yaw_residual_rad + epsilon)
    )
    inactive = np.asarray(
        [
            track.lifecycle in (Lifecycle.DORMANT, Lifecycle.RETIRED)
            for track in tracks
        ],
        dtype=bool,
    )

    # Relocation needs a comparable descriptor on both sides.  Grouping by
    # descriptor dimension constructs that coarse mask without any pair-wise
    # prototype/cosine loop; the exact similarity threshold remains in
    # ``_pair_metric`` and is evaluated only for surviving candidates.
    comparable_appearance = np.zeros(
        (track_count, observation_count), dtype=bool
    )
    observations_by_dimension: dict[int, list[int]] = {}
    for observation_index, observation in enumerate(observations):
        if observation.appearance is not None:
            observations_by_dimension.setdefault(
                int(observation.appearance.shape[0]), []
            ).append(observation_index)
    for track_index, track in enumerate(tracks):
        dimensions = {int(row.embedding.shape[0]) for row in track.prototypes}
        for dimension in sorted(dimensions):
            indices = observations_by_dimension.get(dimension)
            if indices:
                comparable_appearance[track_index, indices] = True

    reactivation_age_valid = ages <= cfg.max_reactivation_age
    relocation_coarse = (
        (center_distance <= cfg.max_relocation_distance_m + epsilon)
        & size_valid
        & comparable_appearance
    )
    candidate_mask = (
        ((~inactive)[:, None]) & normal_valid
    ) | (
        inactive[:, None]
        & reactivation_age_valid[:, None]
        & (normal_valid | relocation_coarse)
    )

    # Bound descriptor comparisons and ambiguity-component density.  Geometry
    # alone can delete an identity-preserving edge when objects cross, so keep
    # the union of geometry Top-K and appearance Top-K in both directions.
    # Each source contributes at most K edges per directed node: the final hard
    # bound is 2*K*(tracks + observations) exact pair evaluations.  Exact costs
    # and null decisions are still evaluated below.
    top_k = cfg.association_candidate_top_k
    precomputed_appearance: np.ndarray | None = None
    if np.count_nonzero(candidate_mask) > top_k * max(
        track_count, observation_count
    ):
        distance_denominator = np.where(
            inactive,
            cfg.max_relocation_distance_m,
            normal_distance_gate,
        )[:, None]
        coarse_cost = (
            cfg.weight_center
            * np.minimum(center_distance / np.maximum(distance_denominator, epsilon), 1.0)
            + cfg.weight_size
            * np.minimum(
                log_size_residual / cfg.max_mean_log_size_residual, 1.0
            )
            + cfg.weight_yaw
            * np.minimum(yaw_residual / cfg.max_yaw_residual_rad, 1.0)
            + cfg.weight_appearance * 0.5
        )
        geometry_bounded = _symmetric_top_k_mask(
            candidate_mask, coarse_cost, top_k
        )
        appearance_similarity = _coarse_appearance_similarity(
            tracks, observations
        )
        appearance_valid = candidate_mask & np.isfinite(appearance_similarity)
        appearance_bounded = _symmetric_top_k_mask(
            appearance_valid, -appearance_similarity, top_k
        )
        candidate_mask &= geometry_bounded | appearance_bounded
        precomputed_appearance = appearance_similarity

    metrics: dict[tuple[int, int], _PairMetric] = {}
    pair_cost = np.full((track_count, observation_count), 1.0e6, dtype=np.float64)
    for track_index, observation_index in np.argwhere(candidate_mask):
        track_index = int(track_index)
        observation_index = int(observation_index)
        if precomputed_appearance is None:
            metric = _pair_metric(
                tracks[track_index],
                observations[observation_index],
                frame_ordinal,
                cfg,
            )
        else:
            similarity = precomputed_appearance[track_index, observation_index]
            metric = _pair_metric(
                tracks[track_index],
                observations[observation_index],
                frame_ordinal,
                cfg,
                precomputed_appearance=(
                    float(similarity) if np.isfinite(similarity) else None
                ),
            )
        if metric.valid:
            metrics[(track_index, observation_index)] = metric
            pair_cost[track_index, observation_index] = metric.cost

    matched = []
    used_tracks: set[int] = set()
    used_observations: set[int] = set()
    track_neighbors = [
        set(np.flatnonzero(pair_cost[index] < 1.0e6).tolist())
        for index in range(track_count)
    ]
    observation_neighbors = [set() for _ in range(observation_count)]
    for track_index, neighbors in enumerate(track_neighbors):
        for observation_index in neighbors:
            observation_neighbors[observation_index].add(track_index)

    # The gated bipartite graph is normally very sparse.  Solving each connected
    # ambiguity component separately is exactly equivalent to one global solve,
    # while avoiding a large (tracks + observations)^3 matrix in online scenes.
    unseen_tracks = {index for index, row in enumerate(track_neighbors) if row}
    while unseen_tracks:
        seed = min(unseen_tracks)
        component_tracks: set[int] = set()
        component_observations: set[int] = set()
        pending_tracks = [seed]
        while pending_tracks:
            track_index = pending_tracks.pop()
            if track_index in component_tracks:
                continue
            component_tracks.add(track_index)
            unseen_tracks.discard(track_index)
            for observation_index in sorted(track_neighbors[track_index]):
                if observation_index in component_observations:
                    continue
                component_observations.add(observation_index)
                pending_tracks.extend(
                    sorted(observation_neighbors[observation_index] - component_tracks)
                )

        local_tracks = sorted(component_tracks)
        local_observations = sorted(component_observations)
        local_track_count = len(local_tracks)
        local_observation_count = len(local_observations)
        dimension = local_track_count + local_observation_count
        augmented = np.full((dimension, dimension), 1.0e6, dtype=np.float64)
        augmented[:local_track_count, :local_observation_count] = pair_cost[
            np.ix_(local_tracks, local_observations)
        ]
        for local_track in range(local_track_count):
            augmented[
                local_track, local_observation_count + local_track
            ] = cfg.null_track_cost
        for local_observation in range(local_observation_count):
            augmented[
                local_track_count + local_observation, local_observation
            ] = cfg.null_observation_cost
        augmented[local_track_count:, local_observation_count:] = 0.0
        assignment = _hungarian_min_cost(augmented)
        for local_track, track_index in enumerate(local_tracks):
            local_observation = int(assignment[local_track])
            if local_observation >= local_observation_count:
                continue
            observation_index = local_observations[local_observation]
            metric = metrics[(track_index, observation_index)]
            matched.append((track_index, observation_index, metric))
            used_tracks.add(track_index)
            used_observations.add(observation_index)
    return (
        matched,
        set(range(track_count)) - used_tracks,
        set(range(observation_count)) - used_observations,
    )


def _weighted_static_box(history: Sequence[_HistoryRow]) -> np.ndarray:
    weights = np.asarray([max(row.score, 1.0e-3) for row in history], dtype=np.float64)
    weights /= weights.sum()
    boxes = np.stack([row.box7 for row in history], axis=0)
    result = np.sum(boxes * weights[:, None], axis=0)
    sin_yaw = float(np.sum(np.sin(boxes[:, 6]) * weights))
    cos_yaw = float(np.sum(np.cos(boxes[:, 6]) * weights))
    result[6] = math.atan2(sin_yaw, cos_yaw)
    return _copy_readonly(result)


def _dynamic_box(
    predicted: np.ndarray,
    observation: BoxObservation,
    history: Sequence[_HistoryRow],
    gain: float,
) -> np.ndarray:
    result = np.array(predicted, copy=True)
    result[:3] = predicted[:3] + gain * (observation.box7[:3] - predicted[:3])
    recent_sizes = np.stack([row.box7[3:6] for row in history[-3:]], axis=0)
    result[3:6] = np.median(recent_sizes, axis=0)
    result[6] = _blend_angle(predicted[6], observation.box7[6], gain)
    return _copy_readonly(result)


def _add_prototype(
    prototypes: tuple[_Prototype, ...],
    observation: BoxObservation,
    frame_ordinal: int,
    cfg: DynamicObjectConfig,
) -> tuple[_Prototype, ...]:
    if observation.appearance is None:
        return prototypes
    candidate = _Prototype(
        embedding=observation.appearance,
        view_direction=observation.view_direction,
        quality=observation.appearance_quality,
        frame_ordinal=frame_ordinal,
        observation_id=observation.observation_id,
    )
    rows = list(prototypes)
    closest = None
    closest_similarity = -1.0
    for index, row in enumerate(rows):
        if row.embedding.shape != candidate.embedding.shape:
            continue
        similarity = float(row.embedding @ candidate.embedding)
        if similarity > closest_similarity:
            closest = index
            closest_similarity = similarity
    if closest is not None and closest_similarity >= cfg.prototype_merge_cosine:
        prior = rows[closest]
        if (candidate.quality, candidate.frame_ordinal, candidate.observation_id) >= (
            prior.quality,
            prior.frame_ordinal,
            prior.observation_id,
        ):
            rows[closest] = candidate
        return tuple(rows)
    rows.append(candidate)
    if len(rows) <= cfg.max_appearance_prototypes:
        return tuple(rows)

    # Retain high-quality and view-diverse representatives deterministically.
    first = max(
        range(len(rows)),
        key=lambda index: (
            rows[index].quality,
            rows[index].frame_ordinal,
            rows[index].observation_id,
        ),
    )
    selected = [first]
    while len(selected) < cfg.max_appearance_prototypes:
        candidates = [index for index in range(len(rows)) if index not in selected]

        def utility(index: int):
            row = rows[index]
            if row.view_direction is None:
                diversity = 0.0
            else:
                comparable = [
                    rows[chosen].view_direction
                    for chosen in selected
                    if rows[chosen].view_direction is not None
                ]
                diversity = (
                    0.0
                    if not comparable
                    else min(1.0 - float(row.view_direction @ view) for view in comparable)
                )
            return (
                row.quality + 0.25 * diversity,
                row.frame_ordinal,
                row.observation_id,
            )

        selected.append(max(candidates, key=utility))
    return tuple(rows[index] for index in sorted(selected))


def _new_track(
    track_id: str,
    observation: BoxObservation,
    frame_ordinal: int,
    cfg: DynamicObjectConfig,
) -> _Track:
    confirmed = cfg.min_confirmed_hits <= 1
    history = (
        _HistoryRow(
            frame_ordinal,
            observation.observation_id,
            observation.box7,
            observation.score,
        ),
    )
    return _Track(
        track_id=track_id,
        current_box7=observation.box7,
        persistent_box7=observation.box7,
        velocity4=_copy_readonly(np.zeros(VELOCITY_DIMENSION)),
        dynamic_probability=0.10,
        persistent_score=observation.score,
        current_score=observation.score,
        lifecycle=Lifecycle.CONFIRMED if confirmed else Lifecycle.TENTATIVE,
        ever_confirmed=confirmed,
        hit_count=1,
        consecutive_visible_misses=0,
        first_observed_frame=frame_ordinal,
        last_observed_frame=frame_ordinal,
        inactive_since_frame=None,
        history=history,
        prototypes=_add_prototype((), observation, frame_ordinal, cfg),
    )


def _update_hit(
    track: _Track,
    observation: BoxObservation,
    frame_ordinal: int,
    metric: _PairMetric,
    cfg: DynamicObjectConfig,
) -> tuple[_Track, bool, bool]:
    was_inactive = track.lifecycle in (Lifecycle.DORMANT, Lifecycle.RETIRED)
    relocated = was_inactive and metric.center_distance_m >= cfg.relocation_distance_m
    dt = max(frame_ordinal - track.last_observed_frame, 1)
    measured = np.empty(VELOCITY_DIMENSION, dtype=np.float64)
    measured[:3] = (observation.box7[:3] - track.current_box7[:3]) / dt
    measured[3] = _angle_delta(observation.box7[6], track.current_box7[6]) / dt
    rate = cfg.velocity_update_rate
    velocity = (1.0 - rate) * track.velocity4 + rate * measured
    speed = float(np.linalg.norm(measured[:3]))
    motion_evidence = np.clip(
        (speed - cfg.dynamic_speed_low_m_per_frame)
        / (cfg.dynamic_speed_high_m_per_frame - cfg.dynamic_speed_low_m_per_frame),
        0.0,
        1.0,
    )
    probability_rate = cfg.dynamic_probability_update_rate
    dynamic_probability = float(
        np.clip(
            (1.0 - probability_rate) * track.dynamic_probability
            + probability_rate * motion_evidence,
            0.0,
            1.0,
        )
    )
    # A strong appearance-supported relocation is direct motion evidence.  In
    # particular, it must not be averaged into the old persistent map anchor as
    # an "uncertain static" observation on the first reactivation frame.
    if relocated:
        dynamic_probability = max(
            dynamic_probability, cfg.dynamic_probability_threshold
        )
    history = (
        track.history
        + (
            _HistoryRow(
                frame_ordinal,
                observation.observation_id,
                observation.box7,
                observation.score,
            ),
        )
    )[-cfg.history_size :]
    state = _motion_state(dynamic_probability, cfg)
    if state == MotionState.DYNAMIC:
        current_box = _dynamic_box(
            metric.predicted_box7,
            observation,
            history,
            cfg.dynamic_measurement_gain,
        )
    else:
        current_box = _weighted_static_box(history)
    hit_count = track.hit_count + 1
    ever_confirmed = track.ever_confirmed or hit_count >= cfg.min_confirmed_hits
    lifecycle = Lifecycle.CONFIRMED if ever_confirmed else Lifecycle.TENTATIVE
    persistent_box = track.persistent_box7
    if not track.ever_confirmed and ever_confirmed:
        persistent_box = current_box
    elif state != MotionState.DYNAMIC:
        persistent_box = current_box
    persistent_score = max(track.persistent_score, observation.score)
    current_score = max(
        observation.score,
        cfg.current_hit_momentum * track.current_score,
    )
    return (
        _Track(
            track_id=track.track_id,
            current_box7=current_box,
            persistent_box7=persistent_box,
            velocity4=_copy_readonly(velocity),
            dynamic_probability=dynamic_probability,
            persistent_score=float(np.clip(persistent_score, 0.0, 1.0)),
            current_score=float(np.clip(current_score, 0.0, 1.0)),
            lifecycle=lifecycle,
            ever_confirmed=ever_confirmed,
            hit_count=hit_count,
            consecutive_visible_misses=0,
            first_observed_frame=track.first_observed_frame,
            last_observed_frame=frame_ordinal,
            inactive_since_frame=None,
            history=history,
            prototypes=_add_prototype(track.prototypes, observation, frame_ordinal, cfg),
        ),
        was_inactive,
        relocated,
    )


def _update_miss(
    track: _Track,
    visibility: Visibility,
    frame_ordinal: int,
    cfg: DynamicObjectConfig,
) -> _Track:
    if not cfg.miss_lifecycle_updates:
        return track
    unobserved_age = max(frame_ordinal - track.last_observed_frame, 0)
    if visibility == Visibility.EXPECTED_VISIBLE:
        misses = track.consecutive_visible_misses + 1
        score = track.current_score * cfg.visible_miss_score_decay
        if misses >= cfg.retire_after_visible_misses:
            lifecycle = Lifecycle.RETIRED
        elif misses >= cfg.dormant_after_visible_misses:
            lifecycle = Lifecycle.DORMANT
        else:
            lifecycle = track.lifecycle
        inactive_since = (
            track.inactive_since_frame
            if track.inactive_since_frame is not None
            else (frame_ordinal if lifecycle in (Lifecycle.DORMANT, Lifecycle.RETIRED) else None)
        )
    elif visibility == Visibility.OCCLUDED:
        misses = track.consecutive_visible_misses
        score = track.current_score
        if track.lifecycle == Lifecycle.RETIRED:
            lifecycle = Lifecycle.RETIRED
        elif unobserved_age > cfg.max_reactivation_age:
            lifecycle = Lifecycle.RETIRED
        elif unobserved_age >= cfg.max_unobserved_coast_frames:
            lifecycle = Lifecycle.DORMANT
        else:
            lifecycle = (
                track.lifecycle
                if track.lifecycle in (Lifecycle.DORMANT, Lifecycle.TENTATIVE)
                else Lifecycle.OCCLUDED
            )
        inactive_since = (
            track.inactive_since_frame
            if track.inactive_since_frame is not None
            else (
                frame_ordinal
                if lifecycle in (Lifecycle.DORMANT, Lifecycle.RETIRED)
                else None
            )
        )
    else:
        misses = track.consecutive_visible_misses
        score = track.current_score
        if track.lifecycle == Lifecycle.RETIRED:
            lifecycle = Lifecycle.RETIRED
        elif unobserved_age > cfg.max_reactivation_age:
            lifecycle = Lifecycle.RETIRED
        elif unobserved_age >= cfg.max_unobserved_coast_frames:
            lifecycle = Lifecycle.DORMANT
        else:
            lifecycle = track.lifecycle
        inactive_since = (
            track.inactive_since_frame
            if track.inactive_since_frame is not None
            else (
                frame_ordinal
                if lifecycle in (Lifecycle.DORMANT, Lifecycle.RETIRED)
                else None
            )
        )
    if unobserved_age > cfg.max_reactivation_age:
        lifecycle = Lifecycle.RETIRED
        inactive_since = (
            track.inactive_since_frame
            if track.inactive_since_frame is not None
            else frame_ordinal
        )
    return _Track(
        track_id=track.track_id,
        current_box7=track.current_box7,
        persistent_box7=track.persistent_box7,
        velocity4=track.velocity4,
        dynamic_probability=track.dynamic_probability,
        persistent_score=track.persistent_score,
        current_score=float(np.clip(score, 0.0, 1.0)),
        lifecycle=lifecycle,
        ever_confirmed=track.ever_confirmed,
        hit_count=track.hit_count,
        consecutive_visible_misses=misses,
        first_observed_frame=track.first_observed_frame,
        last_observed_frame=track.last_observed_frame,
        inactive_since_frame=inactive_since,
        history=track.history,
        prototypes=track.prototypes,
    )


def _eviction_candidate(
    tracks: Mapping[str, _Track], protected_track_ids: set[str]
) -> str | None:
    """Select one safely reclaimable track using a stable total order.

    Unconfirmed tentative rows have no persistent-map commitment and are
    reclaimed first.  Retired rows are the only confirmed rows eligible for
    bounded-map eviction.  Dormant and active confirmed rows are preserved.
    """

    candidates = []
    for track_id, track in tracks.items():
        if track_id in protected_track_ids:
            continue
        if track.lifecycle == Lifecycle.TENTATIVE and not track.ever_confirmed:
            priority = 0
        elif track.lifecycle == Lifecycle.RETIRED:
            priority = 1
        else:
            continue
        candidates.append(
            (
                priority,
                track.current_score,
                track.last_observed_frame,
                track_id,
            )
        )
    if not candidates:
        return None
    return min(candidates)[3]


def _state_digest(tracks: Mapping[str, _Track], version: int, next_id: int) -> str:
    digest = hashlib.sha256()
    digest.update(f"{SCHEMA}|{version}|{next_id}".encode("ascii"))
    for track_id in sorted(tracks):
        row = tracks[track_id]
        digest.update(track_id.encode("utf-8"))
        for array in (row.current_box7, row.persistent_box7, row.velocity4):
            digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(
            (
                f"|{row.dynamic_probability.hex()}|{row.persistent_score.hex()}|"
                f"{row.current_score.hex()}|{row.lifecycle.value}|{row.hit_count}|"
                f"{row.consecutive_visible_misses}|{row.last_observed_frame}"
            ).encode("ascii")
        )
        for history in row.history:
            digest.update(history.observation_id.encode("utf-8"))
            digest.update(str(history.frame_ordinal).encode("ascii"))
            digest.update(np.ascontiguousarray(history.box7).tobytes())
            digest.update(float(history.score).hex().encode("ascii"))
        for prototype in row.prototypes:
            digest.update(prototype.observation_id.encode("utf-8"))
            digest.update(np.ascontiguousarray(prototype.embedding).tobytes())
            digest.update(float(prototype.quality).hex().encode("ascii"))
            digest.update(str(prototype.frame_ordinal).encode("ascii"))
            if prototype.view_direction is None:
                digest.update(b"|no-view|")
            else:
                digest.update(np.ascontiguousarray(prototype.view_direction).tobytes())
    return digest.hexdigest()


def _query_token(
    *,
    state_digest: str,
    frame_ordinal: int,
    observations: Sequence[BoxObservation],
    visibilities: Sequence[TrackVisibility],
) -> str:
    digest = hashlib.sha256()
    digest.update(f"{SCHEMA}|{state_digest}|{frame_ordinal}".encode("ascii"))
    for observation in observations:
        digest.update(observation.observation_id.encode("utf-8"))
        digest.update(np.ascontiguousarray(observation.box7).tobytes())
        digest.update(float(observation.score).hex().encode("ascii"))
        digest.update(float(observation.appearance_quality).hex().encode("ascii"))
        if observation.appearance is not None:
            digest.update(str(observation.appearance.shape).encode("ascii"))
            digest.update(np.ascontiguousarray(observation.appearance).tobytes())
        else:
            digest.update(b"|no-appearance|")
        if observation.view_direction is not None:
            digest.update(np.ascontiguousarray(observation.view_direction).tobytes())
        else:
            digest.update(b"|no-view|")
    for visibility in visibilities:
        digest.update(
            f"{visibility.track_id}|{visibility.visibility.value}".encode("utf-8")
        )
    return digest.hexdigest()


class CausalDynamicObjectMap:
    """Bounded dynamic-object state machine with atomic causal updates."""

    def __init__(self, config: DynamicObjectConfig | Mapping | None = None) -> None:
        if isinstance(config, DynamicObjectConfig):
            self.config = config
        else:
            self.config = resolve_causal_dynamic_object_config(config)
        self._tracks: dict[str, _Track] = {}
        self._next_track_id = 0
        self._last_committed_frame = -1
        self._memory_version = 0
        self._pending: DynamicFrameQuery | None = None
        self._pending_tracks: dict[str, _Track] | None = None
        self._pending_next_track_id: int | None = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def memory_version(self) -> int:
        return self._memory_version

    @property
    def last_committed_frame(self) -> int:
        return self._last_committed_frame

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    @property
    def claimed_observation_ids(self) -> tuple[str, ...]:
        return tuple(
            row.observation_id
            for track_id in sorted(self._tracks)
            for row in self._tracks[track_id].history
        )

    def observation_track_map(self) -> dict[str, str]:
        """Map every retained observation ID to its current owner track."""

        return {
            row.observation_id: track_id
            for track_id in sorted(self._tracks)
            for row in self._tracks[track_id].history
        }

    def recent_observation_ids(self, track_id: str) -> tuple[str, ...]:
        key = _nonempty_id(track_id, "track_id")
        if key not in self._tracks:
            raise DynamicObjectContractError("unknown track_id")
        return tuple(row.observation_id for row in self._tracks[key].history)

    def _snapshot(self, track: _Track, *, persistent: bool) -> ObjectSnapshot:
        box7 = track.persistent_box7 if persistent else track.current_box7
        if (
            not persistent
            and self._last_committed_frame > track.last_observed_frame
            and _motion_state(track.dynamic_probability, self.config)
            == MotionState.DYNAMIC
            and track.lifecycle not in (Lifecycle.DORMANT, Lifecycle.RETIRED)
        ):
            box7 = _predict_box(track, self._last_committed_frame, self.config)
        return ObjectSnapshot(
            track_id=track.track_id,
            box7=_copy_readonly(box7),
            score=track.persistent_score if persistent else track.current_score,
            persistent_score=track.persistent_score,
            current_score=track.current_score,
            lifecycle=track.lifecycle,
            motion_state=_motion_state(track.dynamic_probability, self.config),
            dynamic_probability=track.dynamic_probability,
            velocity4=_copy_readonly(track.velocity4),
            hit_count=track.hit_count,
            consecutive_visible_misses=track.consecutive_visible_misses,
            last_observed_frame=track.last_observed_frame,
            recent_observation_ids=tuple(row.observation_id for row in track.history),
            appearance_prototype_count=len(track.prototypes),
        )

    def track_snapshot(self, track_id: str, *, persistent: bool = False) -> ObjectSnapshot:
        key = _nonempty_id(track_id, "track_id")
        if key not in self._tracks:
            raise DynamicObjectContractError("unknown track_id")
        return self._snapshot(self._tracks[key], persistent=persistent)

    def persistent_snapshots(self) -> tuple[ObjectSnapshot, ...]:
        """Objects ever confirmed; disappearance never lowers this score."""

        return tuple(
            self._snapshot(track, persistent=True)
            for _, track in sorted(self._tracks.items())
            if track.ever_confirmed
        )

    def current_snapshots(
        self, *, include_tentative: bool = False
    ) -> tuple[ObjectSnapshot, ...]:
        """Currently active objects; dormant and retired rows are omitted."""

        allowed = {Lifecycle.CONFIRMED, Lifecycle.OCCLUDED}
        if include_tentative:
            allowed.add(Lifecycle.TENTATIVE)
        return tuple(
            self._snapshot(track, persistent=False)
            for _, track in sorted(self._tracks.items())
            if track.lifecycle in allowed
            and track.current_score >= self.config.current_output_threshold
        )

    def query_frame(
        self,
        *,
        frame_ordinal: int,
        observations: Sequence[BoxObservation] = (),
        visibilities: Sequence[TrackVisibility] = (),
    ) -> DynamicFrameQuery:
        """Compute a decision using only the committed past, without mutation."""

        frame = _strict_frame(frame_ordinal)
        if self._pending is not None:
            raise DynamicObjectContractError("previous query must be committed first")
        if frame <= self._last_committed_frame:
            raise DynamicObjectContractError("frame_ordinal must be strictly increasing")
        observations = tuple(observations)
        visibilities = tuple(visibilities)
        if len(observations) > self.config.max_observations_per_frame:
            raise DynamicObjectContractError("per-frame observation capacity exceeded")
        if not all(isinstance(row, BoxObservation) for row in observations):
            raise DynamicObjectContractError("observations must contain BoxObservation")
        if not all(isinstance(row, TrackVisibility) for row in visibilities):
            raise DynamicObjectContractError("visibilities must contain TrackVisibility")
        observation_ids = [row.observation_id for row in observations]
        if len(set(observation_ids)) != len(observation_ids):
            raise DynamicObjectContractError("duplicate observation_id in one frame")
        retained_claims = set(self.claimed_observation_ids)
        reused = sorted(retained_claims.intersection(observation_ids))
        if reused:
            raise DynamicObjectContractError("observation_id is already retained")
        visibility_ids = [row.track_id for row in visibilities]
        if len(set(visibility_ids)) != len(visibility_ids):
            raise DynamicObjectContractError("duplicate track visibility")
        unknown = sorted(set(visibility_ids) - set(self._tracks))
        if unknown:
            raise DynamicObjectContractError("visibility references unknown track")
        visibility_by_track = {row.track_id: row.visibility for row in visibilities}

        track_rows = [row for _, row in sorted(self._tracks.items())]
        matches, unmatched_tracks, unmatched_observations = _null_aware_assignment(
            track_rows, observations, frame, self.config
        )
        next_tracks = dict(self._tracks)
        associations = []
        observation_to_track: dict[str, str] = {}
        reactivated = []
        relocated = []
        for track_index, observation_index, metric in matches:
            track = track_rows[track_index]
            observation = observations[observation_index]
            updated, was_reactivated, was_relocated = _update_hit(
                track, observation, frame, metric, self.config
            )
            next_tracks[track.track_id] = updated
            observation_to_track[observation.observation_id] = track.track_id
            associations.append(
                Association(
                    observation_id=observation.observation_id,
                    track_id=track.track_id,
                    cost=metric.cost,
                    center_distance_m=metric.center_distance_m,
                    appearance_similarity=metric.appearance_similarity,
                    reactivated=was_reactivated,
                    relocated=was_relocated,
                )
            )
            if was_reactivated:
                reactivated.append(track.track_id)
            if was_relocated:
                relocated.append(track.track_id)

        for track_index in sorted(unmatched_tracks):
            track = track_rows[track_index]
            next_tracks[track.track_id] = _update_miss(
                track,
                visibility_by_track.get(track.track_id, Visibility.UNKNOWN),
                frame,
                self.config,
            )

        next_id = self._next_track_id
        births = []
        evicted_track_ids = []
        capacity_dropped = []
        protected_track_ids = set(observation_to_track.values())
        for observation_index in sorted(unmatched_observations):
            observation = observations[observation_index]
            if len(next_tracks) >= self.config.max_tracks:
                evicted = _eviction_candidate(next_tracks, protected_track_ids)
                if evicted is None:
                    capacity_dropped.append(observation.observation_id)
                    continue
                del next_tracks[evicted]
                evicted_track_ids.append(evicted)
            track_id = f"dyn-{next_id:06d}"
            next_id += 1
            next_tracks[track_id] = _new_track(
                track_id, observation, frame, self.config
            )
            births.append(observation.observation_id)
            observation_to_track[observation.observation_id] = track_id
            # A track created from another observation in this same frame is a
            # committed current-frame claim, not spare capacity for later births.
            protected_track_ids.add(track_id)

        unmatched_track_ids = tuple(
            sorted(track_rows[index].track_id for index in unmatched_tracks)
        )
        state_digest = _state_digest(
            self._tracks, self._memory_version, self._next_track_id
        )
        token = _query_token(
            state_digest=state_digest,
            frame_ordinal=frame,
            observations=observations,
            visibilities=visibilities,
        )
        ordered_mapping = tuple(sorted(observation_to_track.items()))
        query = DynamicFrameQuery(
            frame_ordinal=frame,
            associations=tuple(sorted(associations, key=lambda row: row.observation_id)),
            birth_observation_ids=tuple(births),
            unmatched_track_ids=unmatched_track_ids,
            evicted_track_ids=tuple(evicted_track_ids),
            capacity_dropped_observation_ids=tuple(capacity_dropped),
            claimed_observation_ids=tuple(sorted(observation_to_track)),
            observation_to_track=ordered_mapping,
            reactivated_track_ids=tuple(sorted(reactivated)),
            relocated_track_ids=tuple(sorted(relocated)),
            memory_version_before=self._memory_version,
            maximum_accessed_frame_ordinal=self._last_committed_frame,
            token=token,
        )
        self._pending = query
        self._pending_tracks = next_tracks
        self._pending_next_track_id = next_id
        return query

    def commit_frame(
        self, query: DynamicFrameQuery, *, token: str
    ) -> DynamicFrameCommit:
        """Atomically install the exact pending decision."""

        if query is not self._pending:
            raise DynamicObjectContractError("commit requires the exact pending query")
        if token != query.token:
            raise DynamicObjectContractError("commit token differs from query token")
        if self._pending_tracks is None or self._pending_next_track_id is None:
            raise DynamicObjectContractError("pending state is incomplete")
        self._tracks = self._pending_tracks
        self._next_track_id = self._pending_next_track_id
        self._last_committed_frame = query.frame_ordinal
        self._memory_version += 1
        self._pending = None
        self._pending_tracks = None
        self._pending_next_track_id = None
        return DynamicFrameCommit(
            frame_ordinal=query.frame_ordinal,
            associations=query.associations,
            birth_observation_ids=query.birth_observation_ids,
            evicted_track_ids=query.evicted_track_ids,
            claimed_observation_ids=query.claimed_observation_ids,
            observation_to_track=query.observation_to_track,
            reactivated_track_ids=query.reactivated_track_ids,
            relocated_track_ids=query.relocated_track_ids,
            memory_version_after=self._memory_version,
            live_track_count=len(self._tracks),
            persistent_count=len(self.persistent_snapshots()),
            current_count=len(self.current_snapshots()),
            query_before_commit=True,
            token=token,
        )


__all__ = [
    "Association",
    "BOX_DIMENSION",
    "BoxObservation",
    "CausalDynamicObjectMap",
    "DynamicFrameCommit",
    "DynamicFrameQuery",
    "DynamicObjectConfig",
    "DynamicObjectContractError",
    "Lifecycle",
    "MotionState",
    "ObjectSnapshot",
    "SCHEMA",
    "TrackVisibility",
    "Visibility",
    "resolve_causal_dynamic_object_config",
]
