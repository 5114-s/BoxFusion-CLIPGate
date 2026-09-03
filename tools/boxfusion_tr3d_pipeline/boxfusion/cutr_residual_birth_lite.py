"""Bounded training-free tracking of low-score CuTR residual proposals.

This module is deliberately synchronous and prediction-neutral.  It partitions
raw CuTR scores, tracks only the residual partition with deterministic AABB
geometry, and returns *counterfactual* terminal candidates.  It never appends,
removes, or changes a native prediction.

All thresholds except ``score_ceiling`` are frozen.  The caller must set
``score_ceiling`` to the native CuTR score threshold, so the residual interval
is exactly ``[0.10, score_ceiling)`` and cannot overlap the native partition.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from math import fsum, isfinite
from numbers import Integral, Real
import time
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA = "boxfusion.cutr_residual_birth_lite_shadow.v1"

_SCORE_FLOOR = 0.10
_MIN_HITS = 3
_TTL_KEYFRAMES = 10
_HARD_MAX_TRACKS = 1024
_HARD_MAX_OBSERVATIONS_PER_FRAME = 64
_MAX_STORED_OBSERVATIONS = 5
_MATCH_IOU = 0.10
_MATCH_CENTER_M = 0.50
_DEDUP_IOU = 0.50
_FINAL_MEDIAN_PAIRWISE_IOU = 0.25
_FINAL_CENTER_RMS_M = 0.25
_NOVELTY_IOU = 0.10
_SELF_NMS_IOU = 0.25
_MAX_OUTPUTS = 6
_MIN_EXTENT_M = 0.30


def _strict_int(name: str, value: object, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _readonly_corners(value: object, name: str = "corners") -> np.ndarray:
    try:
        corners = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite [8,3]") from error
    if corners.shape != (8, 3) or not np.isfinite(corners).all():
        raise ValueError(f"{name} must be finite [8,3]")
    extent = np.ptp(corners, axis=0)
    if np.any(extent <= 0.0):
        raise ValueError(f"{name} must have positive AABB extent")
    # A bytes-backed array cannot be made writable again with setflags(),
    # unlike an owning ndarray whose WRITEABLE flag is merely advisory.
    packed = np.array(corners, dtype=np.float64, order="C", copy=True).tobytes()
    result = np.frombuffer(packed, dtype=np.float64).reshape(8, 3)
    return result


def _native_corners(value: object) -> Tuple[np.ndarray, ...]:
    try:
        boxes = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("native_corners must be finite [N,8,3]") from error
    if boxes.size == 0:
        if boxes.ndim == 1 and boxes.shape == (0,):
            return ()
        if boxes.ndim != 3 or boxes.shape[1:] != (8, 3):
            raise ValueError("native_corners must be finite [N,8,3]")
        return ()
    if boxes.ndim != 3 or boxes.shape[1:] != (8, 3):
        raise ValueError("native_corners must be finite [N,8,3]")
    return tuple(
        _readonly_corners(boxes[row], f"native_corners[{row}]")
        for row in range(len(boxes))
    )


def _score_vector(value: object, name: str = "scores") -> np.ndarray:
    try:
        scores = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite one-dimensional sequence") from error
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError(f"{name} must be a finite one-dimensional sequence")
    return np.array(scores, dtype=np.float64, order="C", copy=True)


def _validate_ceiling(value: object) -> float:
    ceiling = _finite_float("score_ceiling", value)
    if not _SCORE_FLOOR < ceiling <= 1.0:
        raise ValueError("score_ceiling must be in (0.10, 1.0]")
    return ceiling


@dataclass(frozen=True)
class RawScorePartition:
    """Immutable row indices for the disjoint raw-score partitions."""

    native_indices: Tuple[int, ...]
    residual_indices: Tuple[int, ...]
    dropped_indices: Tuple[int, ...]
    score_floor: float
    score_ceiling: float


def partition_scores(scores: object, *, score_ceiling: object) -> RawScorePartition:
    """Partition finite raw scores without reordering their row indices."""

    values = _score_vector(scores)
    ceiling = _validate_ceiling(score_ceiling)
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("scores must be in [0,1]")
    rows = np.arange(len(values), dtype=np.int64)
    return RawScorePartition(
        native_indices=tuple(int(row) for row in rows[values >= ceiling]),
        residual_indices=tuple(
            int(row)
            for row in rows[(values >= _SCORE_FLOOR) & (values < ceiling)]
        ),
        dropped_indices=tuple(int(row) for row in rows[values < _SCORE_FLOOR]),
        score_floor=_SCORE_FLOOR,
        score_ceiling=ceiling,
    )


@dataclass(frozen=True)
class ResidualObservation:
    """One copied, immutable CuTR proposal from a true online keyframe."""

    frame_id: int
    raw_index: int
    score: float
    corners: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _strict_int("frame_id", self.frame_id))
        object.__setattr__(self, "raw_index", _strict_int("raw_index", self.raw_index))
        score = _finite_float("score", self.score)
        if score < 0.0 or score > 1.0:
            raise ValueError("score must be in [0,1]")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "corners", _readonly_corners(self.corners))


@dataclass(frozen=True)
class ResidualBirthLiteConfig:
    enabled: bool = False
    observer_only: bool = True
    score_floor: float = _SCORE_FLOOR
    score_ceiling: Optional[float] = None
    max_tracks: int = _HARD_MAX_TRACKS
    max_observations_per_frame: int = _HARD_MAX_OBSERVATIONS_PER_FRAME

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, (bool, np.bool_)):
            raise ValueError("enabled must be a boolean")
        if not isinstance(self.observer_only, (bool, np.bool_)):
            raise ValueError("observer_only must be a boolean")
        if not self.observer_only:
            raise ValueError("observer_only must remain true")
        score_floor = _finite_float("score_floor", self.score_floor)
        if score_floor != _SCORE_FLOOR:
            raise ValueError("score_floor is frozen at 0.10")
        object.__setattr__(self, "score_floor", score_floor)
        if self.score_ceiling is not None:
            object.__setattr__(
                self, "score_ceiling", _validate_ceiling(self.score_ceiling)
            )
        elif self.enabled:
            raise ValueError(
                "enabled residual tracking requires score_ceiling equal to "
                "the native CuTR threshold"
            )
        max_tracks = _strict_int("max_tracks", self.max_tracks, 1)
        if max_tracks > _HARD_MAX_TRACKS:
            raise ValueError(f"max_tracks must not exceed {_HARD_MAX_TRACKS}")
        object.__setattr__(self, "max_tracks", max_tracks)
        per_frame = _strict_int(
            "max_observations_per_frame", self.max_observations_per_frame, 1
        )
        if per_frame > _HARD_MAX_OBSERVATIONS_PER_FRAME:
            raise ValueError(
                "max_observations_per_frame must not exceed "
                f"{_HARD_MAX_OBSERVATIONS_PER_FRAME}"
            )
        object.__setattr__(self, "max_observations_per_frame", per_frame)


@dataclass(frozen=True)
class ResidualAssignment:
    """Committed row-to-track assignment for one accepted residual row."""

    raw_index: int
    track_id: int
    action: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "raw_index", _strict_int("raw_index", self.raw_index)
        )
        object.__setattr__(
            self, "track_id", _strict_int("track_id", self.track_id)
        )
        if not isinstance(self.action, str) or self.action not in (
            "matched",
            "created",
        ):
            raise ValueError("action must be matched or created")


@dataclass(frozen=True)
class ResidualKeyframeResult:
    frame_id: int
    accepted_raw_indices: Tuple[int, ...]
    assignments: Tuple[ResidualAssignment, ...]
    matched_track_ids: Tuple[int, ...]
    created_track_ids: Tuple[int, ...]
    newly_confirmed_track_ids: Tuple[int, ...]
    newly_retired_track_ids: Tuple[int, ...]
    duplicate_dropped_raw_indices: Tuple[int, ...]
    capacity_dropped_raw_indices: Tuple[int, ...]
    track_capacity_dropped_raw_indices: Tuple[int, ...]
    active_track_ids: Tuple[int, ...]
    audit_complete: bool


@dataclass(frozen=True)
class ResidualCandidate:
    track_id: int
    corners: np.ndarray
    raw_mean_score: float
    appended_score: float
    evidence_frame_ids: Tuple[int, ...]
    evidence_raw_indices: Tuple[int, ...]
    median_pairwise_iou: float
    center_rms_m: float
    max_native_iou: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "corners", _readonly_corners(self.corners))

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "track_id": self.track_id,
            "corners": self.corners.tolist(),
            "raw_mean_score": self.raw_mean_score,
            "appended_score": self.appended_score,
            "evidence_frame_ids": list(self.evidence_frame_ids),
            "evidence_raw_indices": list(self.evidence_raw_indices),
            "median_pairwise_iou": self.median_pairwise_iou,
            "center_rms_m": self.center_rms_m,
            "max_native_iou": self.max_native_iou,
        }


@dataclass(frozen=True)
class ResidualCloseResult:
    candidates: Tuple[ResidualCandidate, ...]
    eligible_track_ids: Tuple[int, ...]
    unstable_track_ids: Tuple[int, ...]
    too_small_track_ids: Tuple[int, ...]
    native_overlap_rejected_track_ids: Tuple[int, ...]
    self_nms_rejected_track_ids: Tuple[int, ...]
    output_cap_rejected_track_ids: Tuple[int, ...]
    audit_complete: bool = True
    observer_only: bool = True
    active_authorized: bool = False
    native_mutation_applied: bool = False

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "candidates": [row.to_json_dict() for row in self.candidates],
            "eligible_track_ids": list(self.eligible_track_ids),
            "unstable_track_ids": list(self.unstable_track_ids),
            "too_small_track_ids": list(self.too_small_track_ids),
            "native_overlap_rejected_track_ids": list(
                self.native_overlap_rejected_track_ids
            ),
            "self_nms_rejected_track_ids": list(
                self.self_nms_rejected_track_ids
            ),
            "output_cap_rejected_track_ids": list(
                self.output_cap_rejected_track_ids
            ),
            "audit_complete": self.audit_complete,
            "observer_only": self.observer_only,
            "active_authorized": self.active_authorized,
            "native_mutation_applied": self.native_mutation_applied,
        }


@dataclass(frozen=True)
class ResidualSnapshot:
    last_frame_id: Optional[int]
    keyframes: int
    total_tracks: int
    active_track_ids: Tuple[int, ...]
    confirmed_track_ids: Tuple[int, ...]
    audit_complete: bool
    closed: bool


@dataclass(frozen=True)
class _Track:
    track_id: int
    observations: Tuple[ResidualObservation, ...]
    first_frame_id: int
    last_frame_id: int
    last_keyframe_step: int
    confirmed_frame_id: Optional[int]
    retired_frame_id: Optional[int] = None

    @property
    def active(self) -> bool:
        return self.retired_frame_id is None

    @property
    def association_observation(self) -> ResidualObservation:
        return self.observations[-1]


def _bounds(corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return corners.min(axis=0), corners.max(axis=0)


def _aabb_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_min, left_max = _bounds(left)
    right_min, right_max = _bounds(right)
    intersection_extent = np.maximum(
        np.minimum(left_max, right_max) - np.maximum(left_min, right_min), 0.0
    )
    intersection = float(np.prod(intersection_extent))
    left_volume = float(np.prod(left_max - left_min))
    right_volume = float(np.prod(right_max - right_min))
    union = left_volume + right_volume - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _center(corners: np.ndarray) -> np.ndarray:
    lower, upper = _bounds(corners)
    return 0.5 * (lower + upper)


def _geometry_many(
    corners: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return vectorized AABB bounds, centers, and volumes."""

    if not corners:
        empty = np.empty((0, 3), dtype=np.float64)
        return empty, empty.copy(), empty.copy(), np.empty((0,), dtype=np.float64)
    boxes = np.stack(corners, axis=0)
    lower = boxes.min(axis=1)
    upper = boxes.max(axis=1)
    centers = 0.5 * (lower + upper)
    volumes = np.prod(upper - lower, axis=1)
    return lower, upper, centers, volumes


def _aabb_iou_matrix(
    left_lower: np.ndarray,
    left_upper: np.ndarray,
    left_volume: np.ndarray,
    right_lower: np.ndarray,
    right_upper: np.ndarray,
    right_volume: np.ndarray,
) -> np.ndarray:
    """Compute all left/right AABB IoUs without Python pair loops."""

    if len(left_lower) == 0 or len(right_lower) == 0:
        return np.empty((len(left_lower), len(right_lower)), dtype=np.float64)
    intersection_extent = np.maximum(
        np.minimum(left_upper[:, None, :], right_upper[None, :, :])
        - np.maximum(left_lower[:, None, :], right_lower[None, :, :]),
        0.0,
    )
    intersection = np.prod(intersection_extent, axis=2)
    union = left_volume[:, None] + right_volume[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


def _pairwise_metrics(
    observations: Tuple[ResidualObservation, ...]
) -> Tuple[float, float, np.ndarray]:
    count = len(observations)
    ious = np.eye(count, dtype=np.float64)
    pair_values = []
    for left in range(count):
        for right in range(left + 1, count):
            overlap = _aabb_iou(
                observations[left].corners, observations[right].corners
            )
            ious[left, right] = ious[right, left] = overlap
            pair_values.append(overlap)
    median = float(np.median(np.asarray(pair_values, dtype=np.float64)))
    centers = np.asarray([_center(row.corners) for row in observations])
    centroid = centers.mean(axis=0)
    rms = float(np.sqrt(np.mean(np.sum((centers - centroid) ** 2, axis=1))))
    return median, rms, ious


def _medoid(
    observations: Tuple[ResidualObservation, ...], ious: np.ndarray
) -> ResidualObservation:
    # AABB-IoU medoid; chronological/raw-index tie breaking is explicit.
    costs = np.sum(1.0 - ious, axis=1)
    best = min(
        range(len(observations)),
        key=lambda row: (
            float(costs[row]),
            observations[row].frame_id,
            observations[row].raw_index,
        ),
    )
    return observations[best]


class CuTRResidualBirthLite:
    """Causal geometry-only observer for the low-score CuTR partition."""

    def __init__(self, config: ResidualBirthLiteConfig):
        if not isinstance(config, ResidualBirthLiteConfig):
            raise ValueError("config must be ResidualBirthLiteConfig")
        self.config = config
        self.enabled = bool(config.enabled)
        self.observer_only = True
        self._tracks: Dict[int, _Track] = {}
        self._next_track_id = 0
        self._last_frame_id: Optional[int] = None
        self._keyframe_count = 0
        self._audit_complete = True
        self._closed = False
        self._close_result: Optional[ResidualCloseResult] = None
        self._observe_times_ms = deque(maxlen=2048)
        self._close_times_ms = deque(maxlen=1)
        self._stats = {
            "observations_received": 0,
            "observations_accepted": 0,
            "duplicate_drops": 0,
            "proposal_capacity_drops": 0,
            "track_capacity_drops": 0,
            "tracks_created": 0,
            "tracks_confirmed": 0,
            "tracks_retired": 0,
            "retired_unconfirmed_reclaimed": 0,
            "retired_confirmed_archive_drops": 0,
        }

    def _require_open(self) -> None:
        if not self.config.enabled:
            raise RuntimeError("cutr_residual_birth_lite is disabled")
        if self._closed:
            raise RuntimeError("cutr_residual_birth_lite is closed")

    def observe(
        self, frame_id: object, observations: Sequence[ResidualObservation]
    ) -> ResidualKeyframeResult:
        """Commit one true keyframe as an all-validated transaction."""

        self._require_open()
        normalized_frame = _strict_int("frame_id", frame_id)
        if isinstance(observations, (str, bytes)) or not isinstance(
            observations, Sequence
        ):
            raise ValueError("observations must be a sequence")
        rows = tuple(observations)
        ceiling = self.config.score_ceiling
        if ceiling is None:  # Constructor invariant, kept explicit under -O.
            raise RuntimeError("enabled residual tracking has no score_ceiling")
        for index, row in enumerate(rows):
            if not isinstance(row, ResidualObservation):
                raise ValueError(
                    f"observations[{index}] must be ResidualObservation"
                )
            if row.frame_id != normalized_frame:
                raise ValueError("every observation.frame_id must equal frame_id")
            if not _SCORE_FLOOR <= row.score < ceiling:
                raise ValueError(
                    "observation score is outside the frozen residual interval"
                )
        raw_indices = tuple(row.raw_index for row in rows)
        if len(set(raw_indices)) != len(raw_indices):
            raise ValueError("observation raw_index values must be unique per frame")
        if self._last_frame_id is not None and normalized_frame <= self._last_frame_id:
            raise ValueError("frame_id must be strictly increasing")

        started = time.perf_counter()
        ranked = tuple(sorted(rows, key=lambda row: (-row.score, row.raw_index)))
        selected = ranked[: self.config.max_observations_per_frame]
        proposal_capacity_drops = ranked[self.config.max_observations_per_frame :]

        selected_lower, selected_upper, selected_centers, selected_volumes = (
            _geometry_many(tuple(row.corners for row in selected))
        )
        selected_ious = _aabb_iou_matrix(
            selected_lower,
            selected_upper,
            selected_volumes,
            selected_lower,
            selected_upper,
            selected_volumes,
        )
        deduplicated_indices = []
        duplicate_indices = []
        for row_index in range(len(selected)):
            if deduplicated_indices and np.any(
                selected_ious[row_index, deduplicated_indices] >= _DEDUP_IOU
            ):
                duplicate_indices.append(row_index)
            else:
                deduplicated_indices.append(row_index)
        deduplicated = [selected[index] for index in deduplicated_indices]
        duplicate_drops = [selected[index] for index in duplicate_indices]

        tracks = dict(self._tracks)
        step = self._keyframe_count
        newly_retired = []
        retired_unconfirmed_reclaimed = []
        for track_id in sorted(tracks):
            track = tracks[track_id]
            if track.active and step - track.last_keyframe_step > _TTL_KEYFRAMES:
                newly_retired.append(track_id)
                if track.confirmed_frame_id is None:
                    # A one/two-hit track can never yield a terminal candidate,
                    # so retaining it after TTL would consume the lifetime
                    # birth budget forever.  Reclaim it deterministically.
                    del tracks[track_id]
                    retired_unconfirmed_reclaimed.append(track_id)
                else:
                    tracks[track_id] = replace(
                        track, retired_frame_id=normalized_frame
                    )

        # Retired confirmed tracks remain useful terminal evidence, but their
        # archive is independently bounded.  Overflow drops the oldest rows
        # and marks the supplemental audit incomplete; native output is still
        # untouched.  Together with the active cap this bounds state by
        # 2 * max_tracks instead of imposing a lifetime birth cap.
        retired_confirmed = sorted(
            (
                track
                for track in tracks.values()
                if not track.active and track.confirmed_frame_id is not None
            ),
            key=lambda track: (
                int(track.retired_frame_id),
                track.track_id,
            ),
        )
        retired_confirmed_archive_drops = []
        overflow = len(retired_confirmed) - self.config.max_tracks
        if overflow > 0:
            for track in retired_confirmed[:overflow]:
                retired_confirmed_archive_drops.append(track.track_id)
                del tracks[track.track_id]

        active_ids = tuple(
            track_id for track_id in sorted(tracks) if tracks[track_id].active
        )
        active_id_array = np.asarray(active_ids, dtype=np.int64)
        active_observations = tuple(
            tracks[track_id].association_observation for track_id in active_ids
        )
        track_lower, track_upper, track_centers, track_volumes = _geometry_many(
            tuple(row.corners for row in active_observations)
        )
        if deduplicated_indices:
            observation_lower = selected_lower[deduplicated_indices]
            observation_upper = selected_upper[deduplicated_indices]
            observation_centers = selected_centers[deduplicated_indices]
            observation_volumes = selected_volumes[deduplicated_indices]
        else:
            observation_lower = np.empty((0, 3), dtype=np.float64)
            observation_upper = np.empty((0, 3), dtype=np.float64)
            observation_centers = np.empty((0, 3), dtype=np.float64)
            observation_volumes = np.empty((0,), dtype=np.float64)
        association_ious = _aabb_iou_matrix(
            observation_lower,
            observation_upper,
            observation_volumes,
            track_lower,
            track_upper,
            track_volumes,
        )
        if len(observation_centers) and len(track_centers):
            association_distances = np.linalg.norm(
                observation_centers[:, None, :] - track_centers[None, :, :],
                axis=2,
            )
        else:
            association_distances = np.empty(
                (len(observation_centers), len(track_centers)),
                dtype=np.float64,
            )
        association_valid = (
            (association_ious >= _MATCH_IOU)
            & (association_distances <= _MATCH_CENTER_M)
        )
        used_track_columns = np.zeros(len(active_ids), dtype=np.bool_)
        matched = []
        created = []
        assignments = []
        newly_confirmed = []
        track_capacity_drops = []
        next_track_id = self._next_track_id

        for row_index, row in enumerate(deduplicated):
            candidate_columns = np.flatnonzero(
                association_valid[row_index] & ~used_track_columns
            )
            if len(candidate_columns):
                order = np.lexsort(
                    (
                        active_id_array[candidate_columns],
                        association_distances[row_index, candidate_columns],
                        -association_ious[row_index, candidate_columns],
                    )
                )
                track_column = int(candidate_columns[int(order[0])])
                track_id = int(active_id_array[track_column])
                used_track_columns[track_column] = True
                previous_track = tracks[track_id]
                evidence = previous_track.observations
                if len(evidence) < _MAX_STORED_OBSERVATIONS:
                    evidence = evidence + (row,)
                else:
                    evidence = evidence[1:] + (row,)
                confirmation = previous_track.confirmed_frame_id
                if confirmation is None and len(evidence) >= _MIN_HITS:
                    confirmation = normalized_frame
                    newly_confirmed.append(track_id)
                tracks[track_id] = replace(
                    previous_track,
                    observations=evidence,
                    last_frame_id=normalized_frame,
                    last_keyframe_step=step,
                    confirmed_frame_id=confirmation,
                )
                matched.append(track_id)
                assignments.append(
                    ResidualAssignment(
                        raw_index=row.raw_index,
                        track_id=track_id,
                        action="matched",
                    )
                )
            elif sum(track.active for track in tracks.values()) < self.config.max_tracks:
                track_id = next_track_id
                next_track_id += 1
                tracks[track_id] = _Track(
                    track_id=track_id,
                    observations=(row,),
                    first_frame_id=normalized_frame,
                    last_frame_id=normalized_frame,
                    last_keyframe_step=step,
                    confirmed_frame_id=None,
                )
                created.append(track_id)
                assignments.append(
                    ResidualAssignment(
                        raw_index=row.raw_index,
                        track_id=track_id,
                        action="created",
                    )
                )
            else:
                track_capacity_drops.append(row)

        capacity_drops = (
            proposal_capacity_drops
            or track_capacity_drops
            or retired_confirmed_archive_drops
        )
        audit_complete = self._audit_complete and not bool(capacity_drops)

        track_capacity_drop_indices = {
            row.raw_index for row in track_capacity_drops
        }
        accepted_raw_indices = tuple(
            row.raw_index
            for row in deduplicated
            if row.raw_index not in track_capacity_drop_indices
        )
        assignment_raw_indices = tuple(row.raw_index for row in assignments)
        if (
            assignment_raw_indices != accepted_raw_indices
            or len(set(assignment_raw_indices)) != len(assignment_raw_indices)
        ):
            raise RuntimeError(
                "CuTR residual assignments lost accepted-row alignment"
            )

        # Commit only after every validation and deterministic transition above.
        self._tracks = tracks
        self._next_track_id = next_track_id
        self._last_frame_id = normalized_frame
        self._keyframe_count += 1
        self._audit_complete = audit_complete
        self._stats["observations_received"] += len(rows)
        self._stats["observations_accepted"] += len(deduplicated) - len(track_capacity_drops)
        self._stats["duplicate_drops"] += len(duplicate_drops)
        self._stats["proposal_capacity_drops"] += len(proposal_capacity_drops)
        self._stats["track_capacity_drops"] += len(track_capacity_drops)
        self._stats["tracks_created"] += len(created)
        self._stats["tracks_confirmed"] += len(newly_confirmed)
        self._stats["tracks_retired"] += len(newly_retired)
        self._stats["retired_unconfirmed_reclaimed"] += len(
            retired_unconfirmed_reclaimed
        )
        self._stats["retired_confirmed_archive_drops"] += len(
            retired_confirmed_archive_drops
        )
        self._observe_times_ms.append((time.perf_counter() - started) * 1000.0)

        active_after = tuple(
            track_id for track_id in sorted(tracks) if tracks[track_id].active
        )
        return ResidualKeyframeResult(
            frame_id=normalized_frame,
            accepted_raw_indices=accepted_raw_indices,
            assignments=tuple(assignments),
            matched_track_ids=tuple(matched),
            created_track_ids=tuple(created),
            newly_confirmed_track_ids=tuple(newly_confirmed),
            newly_retired_track_ids=tuple(newly_retired),
            duplicate_dropped_raw_indices=tuple(row.raw_index for row in duplicate_drops),
            capacity_dropped_raw_indices=tuple(
                row.raw_index for row in proposal_capacity_drops
            ),
            track_capacity_dropped_raw_indices=tuple(
                row.raw_index for row in track_capacity_drops
            ),
            active_track_ids=active_after,
            audit_complete=audit_complete,
        )

    def close(
        self, native_corners: object, native_scores: object
    ) -> ResidualCloseResult:
        """Return immutable counterfactual candidates after fixed safety gates."""

        self._require_open()
        native_boxes = _native_corners(native_corners)
        scores = _score_vector(native_scores, "native_scores")
        if len(scores) != len(native_boxes):
            raise ValueError("native_scores must align with native_corners")
        # Native rows came from score >= score_ceiling > 0.10.  Positivity is
        # also necessary for an appended score to be strictly lower while
        # remaining non-negative.
        if np.any((scores <= 0.0) | (scores > 1.0)):
            raise ValueError("native_scores must be in (0,1]")

        started = time.perf_counter()
        unstable = []
        too_small = []
        native_rejected = []
        stable_rows = []
        for track_id in sorted(self._tracks):
            track = self._tracks[track_id]
            if track.confirmed_frame_id is None:
                continue
            median_iou, center_rms, pair_ious = _pairwise_metrics(track.observations)
            medoid = _medoid(track.observations, pair_ious)
            if median_iou < _FINAL_MEDIAN_PAIRWISE_IOU or center_rms > _FINAL_CENTER_RMS_M:
                unstable.append(track_id)
                continue
            if float(np.min(np.ptp(medoid.corners, axis=0))) < _MIN_EXTENT_M:
                too_small.append(track_id)
                continue
            raw_mean = float(
                fsum(row.score for row in track.observations)
                / len(track.observations)
            )
            stable_rows.append(
                (track, medoid, raw_mean, median_iou, center_rms)
            )

        # Batch all terminal native-novelty comparisons.  The prior scalar
        # track-by-native loop was mathematically identical but could take
        # seconds at the frozen 1024 x 100 bounds.
        stable_lower, stable_upper, _, stable_volumes = _geometry_many(
            tuple(row[1].corners for row in stable_rows)
        )
        native_lower, native_upper, _, native_volumes = _geometry_many(
            native_boxes
        )
        native_iou_matrix = _aabb_iou_matrix(
            stable_lower,
            stable_upper,
            stable_volumes,
            native_lower,
            native_upper,
            native_volumes,
        )
        if len(native_boxes):
            max_native_ious = native_iou_matrix.max(axis=1)
        else:
            max_native_ious = np.zeros(len(stable_rows), dtype=np.float64)

        eligible_rows = []
        for row, max_native_iou_value in zip(stable_rows, max_native_ious):
            track, medoid, raw_mean, median_iou, center_rms = row
            max_native_iou = float(max_native_iou_value)
            if max_native_iou >= _NOVELTY_IOU:
                native_rejected.append(track.track_id)
                continue
            eligible_rows.append(
                (track, medoid, raw_mean, median_iou, center_rms, max_native_iou)
            )

        ranked = sorted(
            eligible_rows,
            key=lambda row: (-row[2], -len(row[0].observations), row[0].track_id),
        )
        ranked_lower, ranked_upper, _, ranked_volumes = _geometry_many(
            tuple(row[1].corners for row in ranked)
        )
        ranked_ious = _aabb_iou_matrix(
            ranked_lower,
            ranked_upper,
            ranked_volumes,
            ranked_lower,
            ranked_upper,
            ranked_volumes,
        )
        nms_kept_indices = []
        nms_rejected = []
        for row_index, row in enumerate(ranked):
            if nms_kept_indices and np.any(
                ranked_ious[row_index, nms_kept_indices] >= _SELF_NMS_IOU
            ):
                nms_rejected.append(row[0].track_id)
            else:
                nms_kept_indices.append(row_index)
        nms_kept = [ranked[index] for index in nms_kept_indices]
        output_rows = nms_kept[:_MAX_OUTPUTS]
        cap_rejected = tuple(row[0].track_id for row in nms_kept[_MAX_OUTPUTS:])

        min_native_score = float(np.min(scores)) if len(scores) else None
        candidates = []
        for track, medoid, raw_mean, median_iou, center_rms, max_native_iou in output_rows:
            score_cap = (
                raw_mean
                if min_native_score is None
                else float(np.nextafter(min_native_score, 0.0))
            )
            candidates.append(
                ResidualCandidate(
                    track_id=track.track_id,
                    corners=medoid.corners,
                    raw_mean_score=raw_mean,
                    appended_score=min(raw_mean, score_cap),
                    evidence_frame_ids=tuple(row.frame_id for row in track.observations),
                    evidence_raw_indices=tuple(row.raw_index for row in track.observations),
                    median_pairwise_iou=median_iou,
                    center_rms_m=center_rms,
                    max_native_iou=max_native_iou,
                )
            )

        result = ResidualCloseResult(
            candidates=tuple(candidates),
            eligible_track_ids=tuple(row[0].track_id for row in ranked),
            unstable_track_ids=tuple(unstable),
            too_small_track_ids=tuple(too_small),
            native_overlap_rejected_track_ids=tuple(native_rejected),
            self_nms_rejected_track_ids=tuple(nms_rejected),
            output_cap_rejected_track_ids=cap_rejected,
            audit_complete=self._audit_complete,
        )
        self._closed = True
        self._close_result = result
        self._close_times_ms.append((time.perf_counter() - started) * 1000.0)
        return result

    def snapshot(self) -> ResidualSnapshot:
        active = tuple(
            track_id for track_id in sorted(self._tracks) if self._tracks[track_id].active
        )
        confirmed = tuple(
            track_id
            for track_id in sorted(self._tracks)
            if self._tracks[track_id].confirmed_frame_id is not None
        )
        return ResidualSnapshot(
            last_frame_id=self._last_frame_id,
            keyframes=self._keyframe_count,
            total_tracks=len(self._tracks),
            active_track_ids=active,
            confirmed_track_ids=confirmed,
            audit_complete=self._audit_complete,
            closed=self._closed,
        )

    @staticmethod
    def _timing(values: Sequence[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if not values:
            return None, None, None
        array = np.asarray(values, dtype=np.float64)
        return float(array.mean()), float(np.percentile(array, 95)), float(array.max())

    def summary(self) -> Dict[str, object]:
        observe_mean, observe_p95, observe_max = self._timing(self._observe_times_ms)
        close_mean, close_p95, close_max = self._timing(self._close_times_ms)
        snapshot = self.snapshot()
        return {
            "schema": SCHEMA,
            "enabled": bool(self.config.enabled),
            "observer_only": True,
            "active_authorized": False,
            "native_mutation_applied": False,
            "training_free": True,
            "online_learning": False,
            "geometry_only_association": True,
            "gt_access": False,
            "clip_access": False,
            "detector_label_access": False,
            "detector_score_access": True,
            "detector_score_mutation": False,
            "score_floor": _SCORE_FLOOR,
            "score_ceiling": self.config.score_ceiling,
            "min_hits": _MIN_HITS,
            "ttl_keyframes": _TTL_KEYFRAMES,
            "max_tracks": self.config.max_tracks,
            "max_retired_confirmed_archive": self.config.max_tracks,
            "max_observations_per_frame": self.config.max_observations_per_frame,
            "max_stored_observations": _MAX_STORED_OBSERVATIONS,
            "match_iou": _MATCH_IOU,
            "match_center_m": _MATCH_CENTER_M,
            "dedup_iou": _DEDUP_IOU,
            "final_median_pairwise_iou": _FINAL_MEDIAN_PAIRWISE_IOU,
            "final_center_rms_m": _FINAL_CENTER_RMS_M,
            "novelty_iou": _NOVELTY_IOU,
            "self_nms_iou": _SELF_NMS_IOU,
            "max_outputs": _MAX_OUTPUTS,
            "min_extent_m": _MIN_EXTENT_M,
            "audit_complete": snapshot.audit_complete,
            "closed": snapshot.closed,
            "keyframes": snapshot.keyframes,
            "total_tracks": snapshot.total_tracks,
            "active_tracks": len(snapshot.active_track_ids),
            "confirmed_tracks": len(snapshot.confirmed_track_ids),
            **self._stats,
            "observe_time_mean_ms": observe_mean,
            "observe_time_p95_ms": observe_p95,
            "observe_time_max_ms": observe_max,
            "close_time_mean_ms": close_mean,
            "close_time_p95_ms": close_p95,
            "close_time_max_ms": close_max,
            "close_result": (
                None
                if self._close_result is None
                else self._close_result.to_json_dict()
            ),
        }


def build_cutr_residual_birth_lite(
    root_config: Optional[Mapping[str, object]] = None,
) -> CuTRResidualBirthLite:
    """Build from the strict ``cutr_residual_birth_lite`` root section."""

    if root_config is None:
        root_config = {}
    if not isinstance(root_config, Mapping):
        raise ValueError("root config must be a mapping")
    section = root_config.get("cutr_residual_birth_lite", {})
    if not isinstance(section, Mapping):
        raise ValueError("cutr_residual_birth_lite config must be a mapping")
    allowed = {
        "enabled",
        "observer_only",
        "score_floor",
        "score_ceiling",
        "max_tracks",
        "max_observations_per_frame",
    }
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(
            "Unknown cutr_residual_birth_lite config key(s): "
            + ", ".join(unknown)
        )
    resolved = ResidualBirthLiteConfig(**dict(section))
    if resolved.enabled:
        detection = root_config.get("detection")
        if not isinstance(detection, Mapping) or "score_thresh" not in detection:
            raise ValueError(
                "enabled cutr_residual_birth_lite requires "
                "detection.score_thresh"
            )
        native_threshold = _validate_ceiling(detection["score_thresh"])
        if resolved.score_ceiling != native_threshold:
            raise ValueError(
                "cutr_residual_birth_lite.score_ceiling must equal "
                "detection.score_thresh"
            )
    return CuTRResidualBirthLite(resolved)


__all__ = [
    "CuTRResidualBirthLite",
    "RawScorePartition",
    "ResidualAssignment",
    "ResidualBirthLiteConfig",
    "ResidualCandidate",
    "ResidualCloseResult",
    "ResidualKeyframeResult",
    "ResidualObservation",
    "ResidualSnapshot",
    "partition_scores",
    "build_cutr_residual_birth_lite",
]
