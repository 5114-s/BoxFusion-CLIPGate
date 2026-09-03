"""Bounded, training-free S3R receipts for frozen raw Boxer proposals.

This module is deliberately output-inert.  It has no ground-truth, semantic,
CLIP, native-prediction, depth, optimizer, or training API.  A caller supplies
at most eight already-frozen raw Boxer rows for one real keyframe and advances
the tracker through an explicit two-phase transaction::

    query = tracker.query(frame_id, observations)  # committed past only
    commit = tracker.commit(query)                 # current becomes past

Rows in one query can never associate with tracks created by that same query.
Consequently every track receives at most one observation per frame and a
receipt can freeze only on its first third distinct committed frame.

The receipt geometry is the AABB-IoU medoid of exactly those first three rows.
Pairwise geometry metrics are exported for later output-inert diagnostics but
are not gates.  Later observations can update the live association anchor but
can never change the frozen receipt or its three-row provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import fsum, isfinite
from numbers import Integral, Real
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA = "boxfusion.s3r_receipt_tracker.v1"

_MIN_DISTINCT_FRAMES = 3
_TTL_KEYFRAMES = 10
_MAX_LIVE_TRACKS = 1024
_MAX_RECEIPTS = 1024
_MAX_OBSERVATIONS_PER_FRAME = 8
_MAX_STORED_EVIDENCE = 3
_MATCH_AABB_IOU = 0.10
_MATCH_CENTER_M = 0.50
_MAX_ID = (1 << 63) - 1


def _strict_int(name: str, value: object, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or result > _MAX_ID:
        raise ValueError(f"{name} must be in [{minimum}, {_MAX_ID}]")
    return result


def _finite_score(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("score must be a finite number in [0,1]")
    result = float(value)
    if not isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError("score must be a finite number in [0,1]")
    return result


def _readonly_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite array with shape {shape}") from error
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    packed = np.array(array, dtype=np.float64, order="C", copy=True).tobytes()
    return np.frombuffer(packed, dtype=np.float64).reshape(shape)


def _readonly_corners(value: object, name: str = "corners") -> np.ndarray:
    corners = _readonly_array(value, (8, 3), name)
    if np.any(np.ptp(corners, axis=0) <= 0.0):
        raise ValueError(f"{name} must have positive AABB extent")
    return corners


def _bounds(corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return corners.min(axis=0), corners.max(axis=0)


def _center(corners: np.ndarray) -> np.ndarray:
    lower, upper = _bounds(corners)
    return 0.5 * (lower + upper)


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


def _evidence_metrics(
    evidence: Tuple["S3RObservation", ...],
) -> tuple[np.ndarray, np.ndarray, float, float, int, float]:
    if len(evidence) != _MIN_DISTINCT_FRAMES:
        raise ValueError("S3R receipt metrics require exactly three observations")
    corners = tuple(row.corners for row in evidence)
    lower, upper, centers, volumes = _geometry_many(corners)
    pairwise_iou = _aabb_iou_matrix(lower, upper, volumes, lower, upper, volumes)
    pairwise_center = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    upper_triangle = pairwise_iou[np.triu_indices(len(evidence), 1)]
    median_iou = float(np.median(upper_triangle))
    centroid = centers.mean(axis=0)
    center_rms = float(np.sqrt(np.mean(np.sum((centers - centroid) ** 2, axis=1))))
    costs = np.sum(1.0 - pairwise_iou, axis=1)
    medoid_index = min(
        range(len(evidence)),
        key=lambda index: (
            float(costs[index]),
            evidence[index].frame_id,
            evidence[index].source_row,
            evidence[index].sealed_npz_row,
        ),
    )
    min_medoid_extent = float(np.min(upper[medoid_index] - lower[medoid_index]))
    return (
        pairwise_iou,
        pairwise_center,
        median_iou,
        center_rms,
        medoid_index,
        min_medoid_extent,
    )


@dataclass(frozen=True)
class S3RObservation:
    """One copied raw Boxer OBB from one scheduled keyframe."""

    frame_id: int
    source_row: int
    sealed_npz_row: int
    source_instance_id: int
    score: float
    corners: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _strict_int("frame_id", self.frame_id))
        object.__setattr__(
            self, "source_row", _strict_int("source_row", self.source_row)
        )
        object.__setattr__(
            self,
            "sealed_npz_row",
            _strict_int("sealed_npz_row", self.sealed_npz_row),
        )
        object.__setattr__(
            self,
            "source_instance_id",
            _strict_int("source_instance_id", self.source_instance_id),
        )
        object.__setattr__(self, "score", _finite_score(self.score))
        object.__setattr__(self, "corners", _readonly_corners(self.corners))


@dataclass(frozen=True)
class S3RAssignment:
    """One deterministic assignment planned against committed history."""

    source_row: int
    sealed_npz_row: int
    source_instance_id: int
    track_id: int
    action: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_row", _strict_int("source_row", self.source_row)
        )
        object.__setattr__(
            self,
            "sealed_npz_row",
            _strict_int("sealed_npz_row", self.sealed_npz_row),
        )
        object.__setattr__(
            self,
            "source_instance_id",
            _strict_int("source_instance_id", self.source_instance_id),
        )
        object.__setattr__(self, "track_id", _strict_int("track_id", self.track_id))
        if self.action not in ("matched", "created"):
            raise ValueError("action must be matched or created")


@dataclass(frozen=True)
class S3RReceipt:
    """Immutable first-three receipt and complete raw geometry diagnostics."""

    track_id: int
    confirmation_frame_id: int
    corners: np.ndarray
    medoid_evidence_index: int
    evidence_frame_ids: Tuple[int, ...]
    evidence_source_rows: Tuple[int, ...]
    evidence_sealed_npz_rows: Tuple[int, ...]
    evidence_source_instance_ids: Tuple[int, ...]
    evidence_scores: Tuple[float, ...]
    evidence_corners: np.ndarray
    pairwise_aabb_iou: np.ndarray
    pairwise_center_distance_m: np.ndarray
    raw_mean_score: float
    median_pairwise_aabb_iou: float
    center_rms_m: float
    min_medoid_aabb_extent_m: float
    observer_only: bool = True
    active_authorized: bool = False
    output_mutation_applied: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", _strict_int("track_id", self.track_id))
        object.__setattr__(
            self,
            "confirmation_frame_id",
            _strict_int("confirmation_frame_id", self.confirmation_frame_id),
        )
        medoid = _strict_int("medoid_evidence_index", self.medoid_evidence_index)
        if medoid >= _MIN_DISTINCT_FRAMES:
            raise ValueError("medoid_evidence_index must index the first three rows")
        object.__setattr__(self, "medoid_evidence_index", medoid)
        frames = tuple(
            _strict_int("evidence_frame_id", value) for value in self.evidence_frame_ids
        )
        rows = tuple(
            _strict_int("evidence_source_row", value)
            for value in self.evidence_source_rows
        )
        sealed_rows = tuple(
            _strict_int("evidence_sealed_npz_row", value)
            for value in self.evidence_sealed_npz_rows
        )
        instance_ids = tuple(
            _strict_int("evidence_source_instance_id", value)
            for value in self.evidence_source_instance_ids
        )
        scores = tuple(_finite_score(value) for value in self.evidence_scores)
        if (
            len(frames) != _MIN_DISTINCT_FRAMES
            or frames != tuple(sorted(frames))
            or len(set(frames)) != len(frames)
            or len(rows) != len(frames)
            or len(sealed_rows) != len(frames)
            or len(instance_ids) != len(frames)
            or len(scores) != len(frames)
            or self.confirmation_frame_id != frames[-1]
        ):
            raise ValueError("receipt evidence must be exactly three ordered frames")
        object.__setattr__(self, "evidence_frame_ids", frames)
        object.__setattr__(self, "evidence_source_rows", rows)
        object.__setattr__(self, "evidence_sealed_npz_rows", sealed_rows)
        object.__setattr__(self, "evidence_source_instance_ids", instance_ids)
        object.__setattr__(self, "evidence_scores", scores)

        evidence_corners = _readonly_array(
            self.evidence_corners,
            (_MIN_DISTINCT_FRAMES, 8, 3),
            "evidence_corners",
        )
        for index in range(_MIN_DISTINCT_FRAMES):
            if np.any(np.ptp(evidence_corners[index], axis=0) <= 0.0):
                raise ValueError(
                    "every receipt evidence OBB must have positive AABB extent"
                )
        object.__setattr__(self, "evidence_corners", evidence_corners)
        corners = _readonly_corners(self.corners)
        if not np.array_equal(corners, evidence_corners[medoid]):
            raise ValueError(
                "receipt primary corners must equal the selected raw medoid"
            )
        object.__setattr__(self, "corners", corners)

        synthetic = tuple(
            S3RObservation(
                frames[index],
                rows[index],
                sealed_rows[index],
                instance_ids[index],
                scores[index],
                evidence_corners[index],
            )
            for index in range(_MIN_DISTINCT_FRAMES)
        )
        expected = _evidence_metrics(synthetic)
        pairwise_iou = _readonly_array(
            self.pairwise_aabb_iou,
            (_MIN_DISTINCT_FRAMES, _MIN_DISTINCT_FRAMES),
            "pairwise_aabb_iou",
        )
        pairwise_center = _readonly_array(
            self.pairwise_center_distance_m,
            (_MIN_DISTINCT_FRAMES, _MIN_DISTINCT_FRAMES),
            "pairwise_center_distance_m",
        )
        if not np.allclose(pairwise_iou, expected[0], rtol=0.0, atol=1e-12):
            raise ValueError("pairwise_aabb_iou differs from receipt evidence")
        if not np.allclose(pairwise_center, expected[1], rtol=0.0, atol=1e-12):
            raise ValueError("pairwise_center_distance_m differs from receipt evidence")
        object.__setattr__(self, "pairwise_aabb_iou", pairwise_iou)
        object.__setattr__(self, "pairwise_center_distance_m", pairwise_center)
        if medoid != expected[4]:
            raise ValueError("medoid_evidence_index differs from AABB-IoU medoid")

        expected_scalars = {
            "raw_mean_score": float(fsum(scores) / len(scores)),
            "median_pairwise_aabb_iou": expected[2],
            "center_rms_m": expected[3],
            "min_medoid_aabb_extent_m": expected[5],
        }
        for name, expected_value in expected_scalars.items():
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite diagnostic")
            normalized = float(value)
            if not isfinite(normalized) or not np.isclose(
                normalized, expected_value, rtol=0.0, atol=1e-12
            ):
                raise ValueError(f"{name} differs from receipt evidence")
            object.__setattr__(self, name, normalized)
        if (
            not self.observer_only
            or self.active_authorized
            or self.output_mutation_applied
        ):
            raise ValueError("S3R receipts must remain output-inert")

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "track_id": self.track_id,
            "confirmation_frame_id": self.confirmation_frame_id,
            "corners": self.corners.tolist(),
            "medoid_evidence_index": self.medoid_evidence_index,
            "evidence_frame_ids": list(self.evidence_frame_ids),
            "evidence_source_rows": list(self.evidence_source_rows),
            "evidence_sealed_npz_rows": list(self.evidence_sealed_npz_rows),
            "evidence_source_instance_ids": list(self.evidence_source_instance_ids),
            "evidence_scores": list(self.evidence_scores),
            "evidence_corners": self.evidence_corners.tolist(),
            "pairwise_aabb_iou": self.pairwise_aabb_iou.tolist(),
            "pairwise_center_distance_m": self.pairwise_center_distance_m.tolist(),
            "raw_mean_score": self.raw_mean_score,
            "median_pairwise_aabb_iou": self.median_pairwise_aabb_iou,
            "center_rms_m": self.center_rms_m,
            "min_medoid_aabb_extent_m": self.min_medoid_aabb_extent_m,
            "observer_only": True,
            "active_authorized": False,
            "output_mutation_applied": False,
        }


@dataclass(frozen=True)
class S3RFrameQuery:
    """Opaque one-use query plan built only from committed past state."""

    serial: int
    frame_id: int
    history_max_frame_id: Optional[int]
    prior_track_ids: Tuple[int, ...]
    observations_received: int
    selected_source_rows: Tuple[int, ...]
    selected_sealed_npz_rows: Tuple[int, ...]
    accepted_source_rows: Tuple[int, ...]
    accepted_sealed_npz_rows: Tuple[int, ...]
    observation_capacity_dropped_source_rows: Tuple[int, ...]
    observation_capacity_dropped_sealed_npz_rows: Tuple[int, ...]
    track_capacity_dropped_source_rows: Tuple[int, ...]
    track_capacity_dropped_sealed_npz_rows: Tuple[int, ...]
    receipt_capacity_dropped_track_ids: Tuple[int, ...]
    assignments: Tuple[S3RAssignment, ...]
    newly_retired_track_ids: Tuple[int, ...]
    prospective_receipt_track_ids: Tuple[int, ...]
    audit_complete: bool


@dataclass(frozen=True)
class S3RFrameCommit:
    """Result after atomically committing one exact pending query."""

    frame_id: int
    assignments: Tuple[S3RAssignment, ...]
    matched_track_ids: Tuple[int, ...]
    created_track_ids: Tuple[int, ...]
    newly_retired_track_ids: Tuple[int, ...]
    newly_frozen_receipts: Tuple[S3RReceipt, ...]
    active_track_ids: Tuple[int, ...]
    audit_complete: bool
    observer_only: bool = True
    active_authorized: bool = False
    output_mutation_applied: bool = False


@dataclass(frozen=True)
class S3RSnapshot:
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
    association_observation: S3RObservation
    recent_evidence: Tuple[S3RObservation, ...]
    receipt: Optional[S3RReceipt] = None
    receipt_attempted: bool = False


@dataclass(frozen=True)
class _Pending:
    public: S3RFrameQuery
    tracks: Mapping[int, _Track]
    receipts: Mapping[int, S3RReceipt]
    next_track_id: int
    audit_complete: bool
    stats_delta: Mapping[str, int]
    matched_track_ids: Tuple[int, ...]
    created_track_ids: Tuple[int, ...]
    newly_frozen_receipts: Tuple[S3RReceipt, ...]


def _freeze_first_three(track: _Track, frame_id: int) -> S3RReceipt:
    evidence = track.recent_evidence
    if len(evidence) != _MIN_DISTINCT_FRAMES:
        raise RuntimeError("S3R first-three freeze did not receive exactly three rows")
    frames = tuple(row.frame_id for row in evidence)
    if len(set(frames)) != _MIN_DISTINCT_FRAMES or frames[-1] != frame_id:
        raise RuntimeError("S3R first-three evidence is not distinct and causal")
    metrics = _evidence_metrics(evidence)
    medoid_index = metrics[4]
    evidence_corners = np.stack([row.corners for row in evidence], axis=0)
    return S3RReceipt(
        track_id=track.track_id,
        confirmation_frame_id=frame_id,
        corners=evidence[medoid_index].corners,
        medoid_evidence_index=medoid_index,
        evidence_frame_ids=frames,
        evidence_source_rows=tuple(row.source_row for row in evidence),
        evidence_sealed_npz_rows=tuple(row.sealed_npz_row for row in evidence),
        evidence_source_instance_ids=tuple(row.source_instance_id for row in evidence),
        evidence_scores=tuple(row.score for row in evidence),
        evidence_corners=evidence_corners,
        pairwise_aabb_iou=metrics[0],
        pairwise_center_distance_m=metrics[1],
        raw_mean_score=float(fsum(row.score for row in evidence) / len(evidence)),
        median_pairwise_aabb_iou=metrics[2],
        center_rms_m=metrics[3],
        min_medoid_aabb_extent_m=metrics[5],
    )


class S3RReceiptTracker:
    """Fixed, causal raw-Boxer receipt tracker with no output API."""

    def __init__(self) -> None:
        self.observer_only = True
        self._tracks: Dict[int, _Track] = {}
        self._receipts: Dict[int, S3RReceipt] = {}
        self._next_track_id = 0
        self._last_frame_id: Optional[int] = None
        self._keyframe_count = 0
        self._serial = 0
        self._pending: Optional[_Pending] = None
        self._audit_complete = True
        self._stats = {
            "observations_received": 0,
            "observations_accepted": 0,
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
        self, frame_id: object, observations: Sequence[S3RObservation]
    ) -> S3RFrameQuery:
        """Plan one keyframe against committed history without committing it."""

        if self._pending is not None:
            raise RuntimeError("the previous S3R query must be committed")
        normalized_frame = _strict_int("frame_id", frame_id)
        if self._last_frame_id is not None and normalized_frame <= self._last_frame_id:
            raise ValueError("frame_id must be strictly increasing")
        if isinstance(observations, (str, bytes)) or not isinstance(
            observations, Sequence
        ):
            raise ValueError("observations must be a sequence")
        rows = tuple(observations)
        for index, row in enumerate(rows):
            if not isinstance(row, S3RObservation):
                raise ValueError(f"observations[{index}] must be S3RObservation")
            if row.frame_id != normalized_frame:
                raise ValueError("every observation.frame_id must equal frame_id")
        source_rows = tuple(row.source_row for row in rows)
        if len(set(source_rows)) != len(source_rows):
            raise ValueError("observation source_row values must be unique per frame")
        sealed_rows = tuple(row.sealed_npz_row for row in rows)
        if len(set(sealed_rows)) != len(sealed_rows):
            raise ValueError(
                "observation sealed_npz_row values must be unique per frame"
            )
        instance_ids = tuple(row.source_instance_id for row in rows)
        if len(set(instance_ids)) != len(instance_ids):
            raise ValueError(
                "observation source_instance_id values must be unique per frame"
            )

        ranked = tuple(
            sorted(
                rows,
                key=lambda row: (
                    -row.score,
                    row.source_row,
                    row.sealed_npz_row,
                ),
            )
        )
        selected = ranked[:_MAX_OBSERVATIONS_PER_FRAME]
        observation_capacity_drops = ranked[_MAX_OBSERVATIONS_PER_FRAME:]

        # TTL retirement is part of the pending plan.  Nothing here becomes
        # observable as committed history until commit() receives this exact
        # query token.
        tracks = dict(self._tracks)
        step = self._keyframe_count
        retired = []
        for track_id in sorted(tuple(tracks)):
            track = tracks[track_id]
            if step - track.last_keyframe_step > _TTL_KEYFRAMES:
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
        obs_lower, obs_upper, obs_centers, obs_volumes = _geometry_many(
            tuple(row.corners for row in selected)
        )
        association_ious = _aabb_iou_matrix(
            obs_lower,
            obs_upper,
            obs_volumes,
            track_lower,
            track_upper,
            track_volumes,
        )
        if len(obs_centers) and len(track_centers):
            association_distances = np.linalg.norm(
                obs_centers[:, None, :] - track_centers[None, :, :], axis=2
            )
        else:
            association_distances = np.empty(
                (len(obs_centers), len(track_centers)), dtype=np.float64
            )
        association_valid = (association_ious >= _MATCH_AABB_IOU) & (
            association_distances <= _MATCH_CENTER_M
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

        # Newly-created tracks are intentionally absent from prior_track_ids,
        # so same-frame rows cannot associate or confirm one another.
        for row_index, row in enumerate(selected):
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
                if previous.last_frame_id >= normalized_frame:
                    raise RuntimeError("one S3R track received two rows from one frame")
                # Once the first-three receipt was frozen (or permanently
                # rejected by the receipt cap), provenance is locked forever.
                # Later rows update only the association anchor and TTL.
                if previous.receipt_attempted:
                    recent = previous.recent_evidence
                else:
                    recent = previous.recent_evidence + (row,)
                    if len(recent) > _MAX_STORED_EVIDENCE:
                        recent = recent[-_MAX_STORED_EVIDENCE:]
                updated = replace(
                    previous,
                    last_frame_id=normalized_frame,
                    last_keyframe_step=step,
                    association_observation=row,
                    recent_evidence=recent,
                )
                if (
                    not previous.receipt_attempted
                    and len(recent) >= _MIN_DISTINCT_FRAMES
                ):
                    first_three = recent[:_MIN_DISTINCT_FRAMES]
                    attempt_track = replace(updated, recent_evidence=first_three)
                    if len(receipts) < _MAX_RECEIPTS:
                        receipt = _freeze_first_three(attempt_track, normalized_frame)
                        updated = replace(
                            updated, receipt=receipt, receipt_attempted=True
                        )
                        receipts[track_id] = receipt
                        newly_frozen.append(receipt)
                    else:
                        updated = replace(updated, receipt_attempted=True)
                        receipt_capacity_drops.append(track_id)
                tracks[track_id] = updated
                matched.append(track_id)
                assignments.append(
                    S3RAssignment(
                        row.source_row,
                        row.sealed_npz_row,
                        row.source_instance_id,
                        track_id,
                        "matched",
                    )
                )
            elif len(tracks) < _MAX_LIVE_TRACKS and next_track_id <= _MAX_ID:
                track_id = next_track_id
                next_track_id += 1
                tracks[track_id] = _Track(
                    track_id=track_id,
                    first_frame_id=normalized_frame,
                    last_frame_id=normalized_frame,
                    last_keyframe_step=step,
                    association_observation=row,
                    recent_evidence=(row,),
                )
                created.append(track_id)
                assignments.append(
                    S3RAssignment(
                        row.source_row,
                        row.sealed_npz_row,
                        row.source_instance_id,
                        track_id,
                        "created",
                    )
                )
            else:
                track_capacity_drops.append(row)

        capacity_drop = bool(
            observation_capacity_drops or track_capacity_drops or receipt_capacity_drops
        )
        audit_complete = self._audit_complete and not capacity_drop
        dropped_track_rows = {row.source_row for row in track_capacity_drops}
        accepted_source_rows = tuple(
            row.source_row
            for row in selected
            if row.source_row not in dropped_track_rows
        )
        accepted_sealed_npz_rows = tuple(
            row.sealed_npz_row
            for row in selected
            if row.source_row not in dropped_track_rows
        )
        if (
            tuple(row.source_row for row in assignments) != accepted_source_rows
            or tuple(row.sealed_npz_row for row in assignments)
            != accepted_sealed_npz_rows
        ):
            raise RuntimeError("S3R assignment alignment failure")

        self._serial += 1
        public = S3RFrameQuery(
            serial=self._serial,
            frame_id=normalized_frame,
            history_max_frame_id=self._last_frame_id,
            prior_track_ids=prior_track_ids,
            observations_received=len(rows),
            selected_source_rows=tuple(row.source_row for row in selected),
            selected_sealed_npz_rows=tuple(row.sealed_npz_row for row in selected),
            accepted_source_rows=accepted_source_rows,
            accepted_sealed_npz_rows=accepted_sealed_npz_rows,
            observation_capacity_dropped_source_rows=tuple(
                row.source_row for row in observation_capacity_drops
            ),
            observation_capacity_dropped_sealed_npz_rows=tuple(
                row.sealed_npz_row for row in observation_capacity_drops
            ),
            track_capacity_dropped_source_rows=tuple(
                row.source_row for row in track_capacity_drops
            ),
            track_capacity_dropped_sealed_npz_rows=tuple(
                row.sealed_npz_row for row in track_capacity_drops
            ),
            receipt_capacity_dropped_track_ids=tuple(receipt_capacity_drops),
            assignments=tuple(assignments),
            newly_retired_track_ids=tuple(retired),
            prospective_receipt_track_ids=tuple(
                receipt.track_id for receipt in newly_frozen
            ),
            audit_complete=audit_complete,
        )
        stats_delta = {
            "observations_received": len(rows),
            "observations_accepted": len(accepted_source_rows),
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

    def commit(self, query: S3RFrameQuery) -> S3RFrameCommit:
        """Atomically install the exact pending query as committed past."""

        pending = self._pending
        if pending is None:
            raise RuntimeError("there is no pending S3R query")
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
        return S3RFrameCommit(
            frame_id=query.frame_id,
            assignments=query.assignments,
            matched_track_ids=pending.matched_track_ids,
            created_track_ids=pending.created_track_ids,
            newly_retired_track_ids=query.newly_retired_track_ids,
            newly_frozen_receipts=pending.newly_frozen_receipts,
            active_track_ids=tuple(sorted(self._tracks)),
            audit_complete=self._audit_complete,
        )

    def receipts(self) -> Tuple[S3RReceipt, ...]:
        """Return committed receipts in stable track-ID order."""

        return tuple(self._receipts[track_id] for track_id in sorted(self._receipts))

    def snapshot(self) -> S3RSnapshot:
        return S3RSnapshot(
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
            "output_inert": True,
            "active_authorized": False,
            "birth": False,
            "output_mutation_applied": False,
            "training_free": True,
            "online_learning": False,
            "optimizer_access": False,
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
            "detector_score_used_for_row_order_only": True,
            "per_frame_row_order": ("score_desc_source_row_asc_sealed_npz_row_asc"),
            "within_frame_deduplication": False,
            "minimum_distinct_frames": _MIN_DISTINCT_FRAMES,
            "ttl_keyframes": _TTL_KEYFRAMES,
            "max_live_tracks": _MAX_LIVE_TRACKS,
            "max_receipts": _MAX_RECEIPTS,
            "max_observations_per_frame": _MAX_OBSERVATIONS_PER_FRAME,
            "max_stored_evidence": _MAX_STORED_EVIDENCE,
            "receipt_evidence_count": _MIN_DISTINCT_FRAMES,
            "post_receipt_evidence_updates": False,
            "sealed_npz_row_used_for_identity_only": True,
            "source_instance_id_used_for_identity_only": True,
            "match_aabb_iou": _MATCH_AABB_IOU,
            "match_center_m": _MATCH_CENTER_M,
            "match_rule": "aabb_iou_gte_AND_center_distance_lte",
            "receipt_geometry": "first_three_aabb_iou_medoid",
            "receipt_has_stability_gate": False,
            "receipt_has_extent_gate": False,
            "receipt_has_depth_gate": False,
            "receipt_has_native_gate": False,
            "receipt_has_nms_gate": False,
            "last_frame_id": snapshot.last_frame_id,
            "keyframes": snapshot.keyframes,
            "active_tracks": len(snapshot.active_track_ids),
            "receipts": len(snapshot.receipt_track_ids),
            "pending_frame_id": snapshot.pending_frame_id,
            "audit_complete": snapshot.audit_complete,
            **self._stats,
        }


__all__ = [
    "S3RAssignment",
    "S3RFrameCommit",
    "S3RFrameQuery",
    "S3RObservation",
    "S3RReceipt",
    "S3RReceiptTracker",
    "S3RSnapshot",
    "SCHEMA",
]
