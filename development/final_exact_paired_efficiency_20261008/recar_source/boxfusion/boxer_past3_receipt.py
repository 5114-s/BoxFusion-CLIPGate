"""Bounded past-only OBB receipts for frozen per-view Boxer proposals.

This module is an output-inert observer.  It does not import or inspect native
BoxFusion predictions, detector labels, CLIP features, depth, ground truth, or
training state.  The frozen detector score is accepted only to reproduce the
S0 per-frame ordering; every association and stability decision is geometric.

Each keyframe is an explicit two-phase transaction::

    query = tracker.query(frame_id, observations)  # prior state only
    result = tracker.commit(query)                 # current state becomes past

The query snapshot never contains observations from the queried frame.  This
prevents same-frame proposals from confirming one another.  Empty transactions
are required and advance the keyframe TTL.  When an active track first has at
least three distinct frames and passes the frozen S0 stability rule, its OBB
medoid and complete evidence provenance are copied into an immutable receipt.
Later observations may keep that track alive but cannot change the receipt.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import fsum, isfinite
from numbers import Integral, Real
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA = "boxfusion.boxer_past3_receipt.v1"

# Frozen S0 science and operational bounds.  Association deliberately uses
# both gates (AND), matching the transferred executable used by S0.
_F_MIN_DISTINCT_FRAMES = 3
_F_TTL_KEYFRAMES = 10
_F_MAX_TRACKS = 1024
_F_MAX_RECEIPTS = 1024
_F_MAX_OBSERVATIONS_PER_FRAME = 64
_F_MAX_STORED_OBSERVATIONS = 5
_F_MATCH_AABB_IOU = 0.10
_F_MATCH_CENTER_M = 0.50
_F_DEDUP_AABB_IOU = 0.50
_F_STABLE_MEDIAN_PAIRWISE_AABB_IOU = 0.25
_F_STABLE_CENTER_RMS_M = 0.25
_F_MIN_MEDOID_AABB_EXTENT_M = 0.30
_F_MAX_ID = (1 << 63) - 1


def _strict_int(name: str, value: object, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or result > _F_MAX_ID:
        raise ValueError(f"{name} must be in [{minimum}, {_F_MAX_ID}]")
    return result


def _finite_score(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("score must be a finite number in [0,1]")
    result = float(value)
    if not isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError("score must be a finite number in [0,1]")
    return result


def _readonly_corners(value: object, name: str = "corners") -> np.ndarray:
    try:
        corners = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite [8,3] array") from error
    if corners.shape != (8, 3) or not np.isfinite(corners).all():
        raise ValueError(f"{name} must be a finite [8,3] array")
    if np.any(np.ptp(corners, axis=0) <= 0.0):
        raise ValueError(f"{name} must have positive AABB extent")
    # A bytes-backed ndarray cannot be made writeable again by merely toggling
    # its flag, so a caller cannot mutate committed evidence through aliasing.
    packed = np.array(corners, dtype=np.float64, order="C", copy=True).tobytes()
    return np.frombuffer(packed, dtype=np.float64).reshape(8, 3)


def _bounds(corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return corners.min(axis=0), corners.max(axis=0)


def _center(corners: np.ndarray) -> np.ndarray:
    lower, upper = _bounds(corners)
    return 0.5 * (lower + upper)


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


def _geometry_many(
    corners: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


@dataclass(frozen=True)
class BoxerObservation:
    """One immutable OBB observation from one real keyframe."""

    frame_id: int
    source_row: int
    score: float
    corners: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _strict_int("frame_id", self.frame_id))
        object.__setattr__(
            self, "source_row", _strict_int("source_row", self.source_row)
        )
        object.__setattr__(self, "score", _finite_score(self.score))
        object.__setattr__(self, "corners", _readonly_corners(self.corners))


@dataclass(frozen=True)
class BoxerAssignment:
    """One deterministic current-row assignment planned from prior state."""

    source_row: int
    track_id: int
    action: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_row", _strict_int("source_row", self.source_row)
        )
        object.__setattr__(self, "track_id", _strict_int("track_id", self.track_id))
        if self.action not in ("matched", "created"):
            raise ValueError("action must be matched or created")


@dataclass(frozen=True)
class BoxerPast3Receipt:
    """Geometry and provenance frozen at the first stable past-three event."""

    track_id: int
    confirmation_frame_id: int
    corners: np.ndarray
    evidence_frame_ids: Tuple[int, ...]
    evidence_source_rows: Tuple[int, ...]
    evidence_scores: Tuple[float, ...]
    raw_mean_score: float
    median_pairwise_aabb_iou: float
    center_rms_m: float
    min_medoid_aabb_extent_m: float
    observer_only: bool = True
    active_authorized: bool = False
    native_mutation_applied: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", _strict_int("track_id", self.track_id))
        object.__setattr__(
            self,
            "confirmation_frame_id",
            _strict_int("confirmation_frame_id", self.confirmation_frame_id),
        )
        object.__setattr__(self, "corners", _readonly_corners(self.corners))
        frames = tuple(
            _strict_int("evidence_frame_id", value)
            for value in self.evidence_frame_ids
        )
        rows = tuple(
            _strict_int("evidence_source_row", value)
            for value in self.evidence_source_rows
        )
        scores = tuple(_finite_score(value) for value in self.evidence_scores)
        if (
            len(frames) < _F_MIN_DISTINCT_FRAMES
            or len(frames) > _F_MAX_STORED_OBSERVATIONS
            or len(set(frames)) != len(frames)
            or frames != tuple(sorted(frames))
            or len(rows) != len(frames)
            or len(scores) != len(frames)
            or self.confirmation_frame_id != frames[-1]
        ):
            raise ValueError("receipt evidence must be aligned distinct ordered frames")
        object.__setattr__(self, "evidence_frame_ids", frames)
        object.__setattr__(self, "evidence_source_rows", rows)
        object.__setattr__(self, "evidence_scores", scores)
        for name in (
            "raw_mean_score",
            "median_pairwise_aabb_iou",
            "center_rms_m",
            "min_medoid_aabb_extent_m",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(f"{name} must be finite")
            normalized = float(value)
            if not isfinite(normalized) or normalized < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, normalized)
        if not self.observer_only or self.active_authorized or self.native_mutation_applied:
            raise ValueError("a Boxer-Past3 receipt must remain output-inert")

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "track_id": self.track_id,
            "confirmation_frame_id": self.confirmation_frame_id,
            "corners": self.corners.tolist(),
            "evidence_frame_ids": list(self.evidence_frame_ids),
            "evidence_source_rows": list(self.evidence_source_rows),
            "evidence_scores": list(self.evidence_scores),
            "raw_mean_score": self.raw_mean_score,
            "median_pairwise_aabb_iou": self.median_pairwise_aabb_iou,
            "center_rms_m": self.center_rms_m,
            "min_medoid_aabb_extent_m": self.min_medoid_aabb_extent_m,
            "observer_only": True,
            "active_authorized": False,
            "native_mutation_applied": False,
        }


@dataclass(frozen=True)
class BoxerFrameQuery:
    """Opaque, one-use plan computed exclusively from committed prior state."""

    serial: int
    frame_id: int
    history_max_frame_id: Optional[int]
    prior_track_ids: Tuple[int, ...]
    observations_received: int
    selected_source_rows: Tuple[int, ...]
    accepted_source_rows: Tuple[int, ...]
    duplicate_dropped_source_rows: Tuple[int, ...]
    observation_capacity_dropped_source_rows: Tuple[int, ...]
    track_capacity_dropped_source_rows: Tuple[int, ...]
    receipt_capacity_dropped_track_ids: Tuple[int, ...]
    assignments: Tuple[BoxerAssignment, ...]
    newly_retired_track_ids: Tuple[int, ...]
    prospective_receipt_track_ids: Tuple[int, ...]
    audit_complete: bool


@dataclass(frozen=True)
class BoxerFrameCommit:
    """Committed result for one keyframe transaction."""

    frame_id: int
    assignments: Tuple[BoxerAssignment, ...]
    matched_track_ids: Tuple[int, ...]
    created_track_ids: Tuple[int, ...]
    newly_retired_track_ids: Tuple[int, ...]
    newly_frozen_receipts: Tuple[BoxerPast3Receipt, ...]
    active_track_ids: Tuple[int, ...]
    audit_complete: bool
    observer_only: bool = True
    active_authorized: bool = False
    native_mutation_applied: bool = False


@dataclass(frozen=True)
class BoxerPast3Snapshot:
    """Small immutable view of committed observer state."""

    last_frame_id: Optional[int]
    keyframes: int
    active_track_ids: Tuple[int, ...]
    receipt_track_ids: Tuple[int, ...]
    pending_frame_id: Optional[int]
    audit_complete: bool


@dataclass(frozen=True)
class _Track:
    track_id: int
    first_frame_id: int
    last_frame_id: int
    last_keyframe_step: int
    association_observation: BoxerObservation
    evidence: Tuple[BoxerObservation, ...]
    receipt: Optional[BoxerPast3Receipt] = None


@dataclass(frozen=True)
class _Pending:
    public: BoxerFrameQuery
    tracks: Mapping[int, _Track]
    receipts: Mapping[int, BoxerPast3Receipt]
    next_track_id: int
    audit_complete: bool
    stats_delta: Mapping[str, int]
    matched_track_ids: Tuple[int, ...]
    created_track_ids: Tuple[int, ...]
    newly_frozen_receipts: Tuple[BoxerPast3Receipt, ...]


def _pairwise_metrics(
    observations: Tuple[BoxerObservation, ...]
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
    observations: Tuple[BoxerObservation, ...], ious: np.ndarray
) -> BoxerObservation:
    costs = np.sum(1.0 - ious, axis=1)
    best = min(
        range(len(observations)),
        key=lambda index: (
            float(costs[index]),
            observations[index].frame_id,
            observations[index].source_row,
        ),
    )
    return observations[best]


def _stable_receipt(track: _Track, frame_id: int) -> Optional[BoxerPast3Receipt]:
    evidence = track.evidence
    frames = tuple(row.frame_id for row in evidence)
    if len(evidence) < _F_MIN_DISTINCT_FRAMES or len(set(frames)) < _F_MIN_DISTINCT_FRAMES:
        return None
    median_iou, center_rms, pair_ious = _pairwise_metrics(evidence)
    medoid = _medoid(evidence, pair_ious)
    min_extent = float(np.min(np.ptp(medoid.corners, axis=0)))
    if (
        median_iou < _F_STABLE_MEDIAN_PAIRWISE_AABB_IOU
        or center_rms > _F_STABLE_CENTER_RMS_M
        or min_extent < _F_MIN_MEDOID_AABB_EXTENT_M
    ):
        return None
    return BoxerPast3Receipt(
        track_id=track.track_id,
        confirmation_frame_id=frame_id,
        corners=medoid.corners,
        evidence_frame_ids=frames,
        evidence_source_rows=tuple(row.source_row for row in evidence),
        evidence_scores=tuple(row.score for row in evidence),
        raw_mean_score=float(fsum(row.score for row in evidence) / len(evidence)),
        median_pairwise_aabb_iou=median_iou,
        center_rms_m=center_rms,
        min_medoid_aabb_extent_m=min_extent,
    )


class BoxerPast3ReceiptTracker:
    """Training-free two-phase tracker that can only emit shadow receipts."""

    def __init__(self) -> None:
        self.observer_only = True
        self._tracks: Dict[int, _Track] = {}
        self._receipts: Dict[int, BoxerPast3Receipt] = {}
        self._next_track_id = 0
        self._last_frame_id: Optional[int] = None
        self._keyframe_count = 0
        self._serial = 0
        self._pending: Optional[_Pending] = None
        self._audit_complete = True
        self._stats = {
            "observations_received": 0,
            "observations_accepted": 0,
            "duplicate_drops": 0,
            "observation_capacity_drops": 0,
            "track_capacity_drops": 0,
            "receipt_capacity_drops": 0,
            "tracks_created": 0,
            "tracks_matched": 0,
            "tracks_retired": 0,
            "receipts_frozen": 0,
            "empty_keyframes": 0,
        }

    def query(
        self, frame_id: object, observations: Sequence[BoxerObservation]
    ) -> BoxerFrameQuery:
        """Plan one frame against committed history without committing it."""

        if self._pending is not None:
            raise RuntimeError("the previous Boxer-Past3 query must be committed")
        normalized_frame = _strict_int("frame_id", frame_id)
        if self._last_frame_id is not None and normalized_frame <= self._last_frame_id:
            raise ValueError("frame_id must be strictly increasing")
        if isinstance(observations, (str, bytes)) or not isinstance(
            observations, Sequence
        ):
            raise ValueError("observations must be a sequence")
        rows = tuple(observations)
        for index, row in enumerate(rows):
            if not isinstance(row, BoxerObservation):
                raise ValueError(f"observations[{index}] must be BoxerObservation")
            if row.frame_id != normalized_frame:
                raise ValueError("every observation.frame_id must equal frame_id")
        source_rows = tuple(row.source_row for row in rows)
        if len(set(source_rows)) != len(source_rows):
            raise ValueError("observation source_row values must be unique per frame")

        # Reproduce S0's fixed detector-score ordering without using the score
        # in association, confirmation, or receipt stability.
        ranked = tuple(sorted(rows, key=lambda row: (-row.score, row.source_row)))
        selected = ranked[:_F_MAX_OBSERVATIONS_PER_FRAME]
        observation_capacity_drops = ranked[_F_MAX_OBSERVATIONS_PER_FRAME:]

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
                selected_ious[row_index, deduplicated_indices] >= _F_DEDUP_AABB_IOU
            ):
                duplicate_indices.append(row_index)
            else:
                deduplicated_indices.append(row_index)
        deduplicated = tuple(selected[index] for index in deduplicated_indices)
        duplicate_drops = tuple(selected[index] for index in duplicate_indices)

        # TTL is planned before association.  The copy is not installed until
        # commit, so query still cannot expose current rows as historical state.
        tracks = dict(self._tracks)
        step = self._keyframe_count
        retired = []
        for track_id in sorted(tuple(tracks)):
            track = tracks[track_id]
            if step - track.last_keyframe_step > _F_TTL_KEYFRAMES:
                retired.append(track_id)
                del tracks[track_id]

        prior_track_ids = tuple(sorted(tracks))
        prior_id_array = np.asarray(prior_track_ids, dtype=np.int64)
        prior_observations = tuple(
            tracks[track_id].association_observation for track_id in prior_track_ids
        )
        track_lower, track_upper, track_centers, track_volumes = _geometry_many(
            tuple(row.corners for row in prior_observations)
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
                observation_centers[:, None, :] - track_centers[None, :, :], axis=2
            )
        else:
            association_distances = np.empty(
                (len(observation_centers), len(track_centers)), dtype=np.float64
            )
        association_valid = (
            (association_ious >= _F_MATCH_AABB_IOU)
            & (association_distances <= _F_MATCH_CENTER_M)
        )

        receipts = dict(self._receipts)
        used_track_columns = np.zeros(len(prior_track_ids), dtype=np.bool_)
        assignments = []
        matched = []
        created = []
        track_capacity_drops = []
        receipt_capacity_drops = []
        newly_frozen = []
        next_track_id = self._next_track_id

        # The candidate columns are fixed to prior_track_ids.  Newly created
        # current-frame tracks are never inserted into this query matrix.
        for row_index, row in enumerate(deduplicated):
            candidate_columns = np.flatnonzero(
                association_valid[row_index] & ~used_track_columns
            )
            if len(candidate_columns):
                order = np.lexsort(
                    (
                        prior_id_array[candidate_columns],
                        association_distances[row_index, candidate_columns],
                        -association_ious[row_index, candidate_columns],
                    )
                )
                track_column = int(candidate_columns[int(order[0])])
                track_id = int(prior_id_array[track_column])
                used_track_columns[track_column] = True
                previous = tracks[track_id]
                evidence = previous.evidence
                if previous.receipt is None:
                    evidence = evidence + (row,)
                    if len(evidence) > _F_MAX_STORED_OBSERVATIONS:
                        evidence = evidence[-_F_MAX_STORED_OBSERVATIONS:]
                updated = replace(
                    previous,
                    last_frame_id=normalized_frame,
                    last_keyframe_step=step,
                    association_observation=row,
                    evidence=evidence,
                )
                if updated.receipt is None:
                    receipt = _stable_receipt(updated, normalized_frame)
                    if receipt is not None:
                        if len(receipts) < _F_MAX_RECEIPTS:
                            updated = replace(updated, receipt=receipt)
                            receipts[track_id] = receipt
                            newly_frozen.append(receipt)
                        else:
                            receipt_capacity_drops.append(track_id)
                tracks[track_id] = updated
                matched.append(track_id)
                assignments.append(
                    BoxerAssignment(row.source_row, track_id, "matched")
                )
            elif len(tracks) < _F_MAX_TRACKS and next_track_id <= _F_MAX_ID:
                track_id = next_track_id
                next_track_id += 1
                tracks[track_id] = _Track(
                    track_id=track_id,
                    first_frame_id=normalized_frame,
                    last_frame_id=normalized_frame,
                    last_keyframe_step=step,
                    association_observation=row,
                    evidence=(row,),
                )
                created.append(track_id)
                assignments.append(
                    BoxerAssignment(row.source_row, track_id, "created")
                )
            else:
                track_capacity_drops.append(row)

        capacity_drop = bool(
            observation_capacity_drops
            or track_capacity_drops
            or receipt_capacity_drops
        )
        audit_complete = self._audit_complete and not capacity_drop
        track_capacity_rows = {row.source_row for row in track_capacity_drops}
        accepted_source_rows = tuple(
            row.source_row
            for row in deduplicated
            if row.source_row not in track_capacity_rows
        )
        if tuple(row.source_row for row in assignments) != accepted_source_rows:
            raise RuntimeError("Boxer-Past3 assignment alignment failure")

        self._serial += 1
        public = BoxerFrameQuery(
            serial=self._serial,
            frame_id=normalized_frame,
            history_max_frame_id=self._last_frame_id,
            prior_track_ids=prior_track_ids,
            observations_received=len(rows),
            selected_source_rows=tuple(row.source_row for row in selected),
            accepted_source_rows=accepted_source_rows,
            duplicate_dropped_source_rows=tuple(
                row.source_row for row in duplicate_drops
            ),
            observation_capacity_dropped_source_rows=tuple(
                row.source_row for row in observation_capacity_drops
            ),
            track_capacity_dropped_source_rows=tuple(
                row.source_row for row in track_capacity_drops
            ),
            receipt_capacity_dropped_track_ids=tuple(receipt_capacity_drops),
            assignments=tuple(assignments),
            newly_retired_track_ids=tuple(retired),
            prospective_receipt_track_ids=tuple(
                row.track_id for row in newly_frozen
            ),
            audit_complete=audit_complete,
        )
        stats_delta = {
            "observations_received": len(rows),
            "observations_accepted": len(accepted_source_rows),
            "duplicate_drops": len(duplicate_drops),
            "observation_capacity_drops": len(observation_capacity_drops),
            "track_capacity_drops": len(track_capacity_drops),
            "receipt_capacity_drops": len(receipt_capacity_drops),
            "tracks_created": len(created),
            "tracks_matched": len(matched),
            "tracks_retired": len(retired),
            "receipts_frozen": len(newly_frozen),
            "empty_keyframes": int(len(rows) == 0),
        }
        self._pending = _Pending(
            public=public,
            tracks=tracks,
            receipts=receipts,
            next_track_id=next_track_id,
            audit_complete=audit_complete,
            stats_delta=stats_delta,
            matched_track_ids=tuple(matched),
            created_track_ids=tuple(created),
            newly_frozen_receipts=tuple(newly_frozen),
        )
        return public

    def commit(self, query: BoxerFrameQuery) -> BoxerFrameCommit:
        """Atomically make one exact query plan the new committed past."""

        pending = self._pending
        if pending is None:
            raise RuntimeError("there is no pending Boxer-Past3 query")
        if query is not pending.public:
            raise ValueError("commit requires the exact pending query token")
        self._tracks = dict(pending.tracks)
        self._receipts = dict(pending.receipts)
        self._next_track_id = pending.next_track_id
        self._last_frame_id = query.frame_id
        self._keyframe_count += 1
        self._audit_complete = pending.audit_complete
        for name, value in pending.stats_delta.items():
            self._stats[name] += int(value)
        self._pending = None
        return BoxerFrameCommit(
            frame_id=query.frame_id,
            assignments=query.assignments,
            matched_track_ids=pending.matched_track_ids,
            created_track_ids=pending.created_track_ids,
            newly_retired_track_ids=query.newly_retired_track_ids,
            newly_frozen_receipts=pending.newly_frozen_receipts,
            active_track_ids=tuple(sorted(self._tracks)),
            audit_complete=self._audit_complete,
        )

    def receipts(self) -> Tuple[BoxerPast3Receipt, ...]:
        """Return committed receipts in stable track-ID order."""

        return tuple(self._receipts[track_id] for track_id in sorted(self._receipts))

    def snapshot(self) -> BoxerPast3Snapshot:
        return BoxerPast3Snapshot(
            last_frame_id=self._last_frame_id,
            keyframes=self._keyframe_count,
            active_track_ids=tuple(sorted(self._tracks)),
            receipt_track_ids=tuple(sorted(self._receipts)),
            pending_frame_id=(
                None if self._pending is None else self._pending.public.frame_id
            ),
            audit_complete=self._audit_complete,
        )

    def summary(self) -> Dict[str, object]:
        snapshot = self.snapshot()
        return {
            "schema": SCHEMA,
            "observer_only": True,
            "active_authorized": False,
            "native_mutation_applied": False,
            "training_free": True,
            "online_learning": False,
            "past_only": True,
            "query_before_commit": True,
            "same_frame_confirmation": False,
            "geometry_only_association": True,
            "gt_access": False,
            "clip_access": False,
            "native_prediction_access": False,
            "depth_access": False,
            "detector_label_access": False,
            "detector_score_access": True,
            "detector_score_used_for_ranking_only": True,
            "minimum_distinct_frames": _F_MIN_DISTINCT_FRAMES,
            "ttl_keyframes": _F_TTL_KEYFRAMES,
            "max_tracks": _F_MAX_TRACKS,
            "max_receipts": _F_MAX_RECEIPTS,
            "max_observations_per_frame": _F_MAX_OBSERVATIONS_PER_FRAME,
            "max_stored_observations": _F_MAX_STORED_OBSERVATIONS,
            "match_aabb_iou": _F_MATCH_AABB_IOU,
            "match_center_m": _F_MATCH_CENTER_M,
            "match_rule": "aabb_iou_gte_AND_center_distance_lte",
            "dedup_aabb_iou": _F_DEDUP_AABB_IOU,
            "stable_median_pairwise_aabb_iou": (
                _F_STABLE_MEDIAN_PAIRWISE_AABB_IOU
            ),
            "stable_center_rms_m": _F_STABLE_CENTER_RMS_M,
            "min_medoid_aabb_extent_m": _F_MIN_MEDOID_AABB_EXTENT_M,
            "receipt_freezes_on_first_stable_past3": True,
            "last_frame_id": snapshot.last_frame_id,
            "keyframes": snapshot.keyframes,
            "active_tracks": len(snapshot.active_track_ids),
            "receipts": len(snapshot.receipt_track_ids),
            "pending_frame_id": snapshot.pending_frame_id,
            "audit_complete": snapshot.audit_complete,
            **self._stats,
        }


__all__ = [
    "BoxerAssignment",
    "BoxerFrameCommit",
    "BoxerFrameQuery",
    "BoxerObservation",
    "BoxerPast3Receipt",
    "BoxerPast3ReceiptTracker",
    "BoxerPast3Snapshot",
    "SCHEMA",
]
