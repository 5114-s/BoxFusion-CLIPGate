"""Training-free causal voxel-hash instance proposals for OAS-P1.

The module consumes class-agnostic automatic-mask RGB-D lifts.  It owns no
dataset, prediction, evaluator, annotation, semantic label, or trainable model
API.  Observations from one frame are queried against a snapshot of committed
past state, assigned one-to-one, and only then committed.  All state is
bounded and deterministic.

This is an OnlineAnySeg-inspired lightweight adapter, not a reproduction of
the complete method: the frozen input has geometry but no FCGF/CLIP instance
descriptor, so association uses mutual 5 cm voxel support and conservative
spatial gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree


VOXEL_SIZE_M = 0.05
MAX_VOXELS_PER_OBSERVATION = 512
MAX_HASH_VOXELS_PER_MEMORY = 2_048
MAX_RETAINED_VIEWS = 5
MIN_CONFIRMATION_VIEWS = 5
TTL_KEYFRAMES = 10
MAX_LIVE_MEMORIES = 1_024
MAX_TRACKS_PER_HASH_BUCKET = 8
RADIUS_QUERY_KEY_CAP = 64

MAX_CENTER_DISTANCE_M = 0.50
MIN_NEAR_OVERLAP_VOXELS = 8
MIN_MUTUAL_COVERAGE = 0.30
MIN_DOMINANT_COVERAGE = 0.80
MIN_SECONDARY_COVERAGE = 0.15
MIN_AABB_IOU_FOR_MATCH = 0.02
MIN_SMALLER_CONTAINMENT_FOR_MATCH = 0.10

MIN_CONSENSUS_VOXELS = 16
MAX_CONSENSUS_VOXELS = 2_048
CONSENSUS_MIN_VIEW_SUPPORT = 2
MIN_MEDIAN_ASSOCIATION_HARMONIC = 0.30
MIN_MEAN_AUTOMASK_CONFIDENCE = 0.50
CONSENSUS_MIN_VIEW_AABB_IOU = 0.10
CONSENSUS_MIN_LOO_STABILITY_IOU = 0.25
CONSENSUS_MAX_CENTER_SHIFT_M = 0.50
CONSENSUS_EXTENT_RATIO_RANGE = (0.50, 2.00)
CONSENSUS_VOLUME_RATIO_RANGE = (0.25, 4.00)
MIN_OUTPUT_EXTENT_M = 0.05
MAX_OUTPUT_EXTENT_M = 3.50
MAX_OUTPUT_VOLUME_M3 = 12.0

_NEIGHBOR_OFFSETS = np.asarray(
    [
        (x, y, z)
        for x in (-1, 0, 1)
        for y in (-1, 0, 1)
        for z in (-1, 0, 1)
    ],
    dtype=np.int64,
)
_CORNER_BITS = np.asarray(
    [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (1.0, 1.0, 1.0),
    ],
    dtype=np.float64,
)


def _finite_points(value: object) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("points_world must be finite [N,3]")
    if len(points) < MIN_CONSENSUS_VOXELS:
        raise ValueError("automatic-mask lift has too few points")
    return points


def _cap_rows(rows: np.ndarray, maximum: int) -> np.ndarray:
    if len(rows) <= maximum:
        return np.ascontiguousarray(rows)
    indices = (np.arange(maximum, dtype=np.int64) * (len(rows) - 1)) // (maximum - 1)
    return np.ascontiguousarray(rows[indices])


def _voxel_keys(points: np.ndarray, maximum: int) -> np.ndarray:
    keys = np.unique(np.floor(points / VOXEL_SIZE_M).astype(np.int64), axis=0)
    return _cap_rows(keys, maximum)


def _robust_bounds(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = np.quantile(points, [0.02, 0.98], axis=0)
    center = (lower + upper) * 0.5
    extent = np.maximum(upper - lower, MIN_OUTPUT_EXTENT_M)
    return center - extent * 0.5, center + extent * 0.5


def bounds_to_corners(lower: object, upper: object) -> np.ndarray:
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if (
        lo.shape != (3,)
        or hi.shape != (3,)
        or not np.isfinite(lo).all()
        or not np.isfinite(hi).all()
        or np.any(hi <= lo)
    ):
        raise ValueError("bounds must be finite positive xyz vectors")
    return np.ascontiguousarray((lo + _CORNER_BITS * (hi - lo)).astype(np.float32))


def aabb_overlap(
    left_lower: object,
    left_upper: object,
    right_lower: object,
    right_upper: object,
) -> tuple[float, float, float]:
    ll = np.asarray(left_lower, dtype=np.float64)
    lu = np.asarray(left_upper, dtype=np.float64)
    rl = np.asarray(right_lower, dtype=np.float64)
    ru = np.asarray(right_upper, dtype=np.float64)
    intersection = float(np.prod(np.maximum(np.minimum(lu, ru) - np.maximum(ll, rl), 0.0)))
    left_volume = float(np.prod(np.maximum(lu - ll, 0.0)))
    right_volume = float(np.prod(np.maximum(ru - rl, 0.0)))
    union = left_volume + right_volume - intersection
    return (
        0.0 if union <= 0.0 else intersection / union,
        0.0 if left_volume <= 0.0 else intersection / left_volume,
        0.0 if right_volume <= 0.0 else intersection / right_volume,
    )


@dataclass(frozen=True)
class AutomaticMaskObservation:
    source_id: str
    frame_id: int
    frame_ordinal: int
    confidence: float
    points_world: np.ndarray
    voxel_keys: np.ndarray = field(init=False, repr=False)
    lower: np.ndarray = field(init=False, repr=False)
    upper: np.ndarray = field(init=False, repr=False)
    center: np.ndarray = field(init=False, repr=False)
    voxel_tree: cKDTree = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be non-empty")
        if (
            isinstance(self.frame_id, bool)
            or not isinstance(self.frame_id, (int, np.integer))
            or int(self.frame_id) < 0
            or isinstance(self.frame_ordinal, bool)
            or not isinstance(self.frame_ordinal, (int, np.integer))
            or int(self.frame_ordinal) < 0
        ):
            raise ValueError("frame identity must contain non-negative integers")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must lie in [0,1]")
        points = _finite_points(self.points_world)
        keys = _voxel_keys(points, MAX_VOXELS_PER_OBSERVATION)
        if len(keys) < MIN_CONSENSUS_VOXELS:
            raise ValueError("automatic-mask lift has too few 5 cm voxels")
        lower, upper = _robust_bounds(points)
        for value in (points, keys, lower, upper):
            value.setflags(write=False)
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "frame_ordinal", int(self.frame_ordinal))
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "points_world", points)
        object.__setattr__(self, "voxel_keys", keys)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "center", (lower + upper) * 0.5)
        object.__setattr__(self, "voxel_tree", cKDTree(keys.astype(np.float64)))


@dataclass(frozen=True)
class AssociationEvidence:
    harmonic: float
    observation_coverage: float
    memory_view_coverage: float
    near_overlap_voxels: int
    aabb_iou: float
    smaller_containment: float
    center_distance_m: float


def _near_overlap(
    left: AutomaticMaskObservation,
    right: AutomaticMaskObservation,
) -> tuple[int, float, float, float]:
    left_supported = right.voxel_tree.query_ball_point(
        left.voxel_keys.astype(np.float64), r=1.0, p=np.inf, return_length=True
    )
    right_supported = left.voxel_tree.query_ball_point(
        right.voxel_keys.astype(np.float64), r=1.0, p=np.inf, return_length=True
    )
    left_count = int(np.count_nonzero(left_supported))
    right_count = int(np.count_nonzero(right_supported))
    left_fraction = left_count / len(left.voxel_keys)
    right_fraction = right_count / len(right.voxel_keys)
    harmonic = (
        0.0
        if left_fraction + right_fraction <= 0.0
        else 2.0 * left_fraction * right_fraction / (left_fraction + right_fraction)
    )
    return min(left_count, right_count), left_fraction, right_fraction, harmonic


@dataclass
class VoxelMemory:
    memory_id: int
    first_frame_ordinal: int
    last_frame_ordinal: int
    retained: list[AutomaticMaskObservation] = field(default_factory=list)
    source_count_total: int = 0
    association_evidence: list[AssociationEvidence] = field(default_factory=list)
    hash_keys: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.int64), repr=False
    )
    retired_at_frame_ordinal: int | None = None
    dropped_view_count: int = 0

    @property
    def latest(self) -> AutomaticMaskObservation:
        if not self.retained:
            raise ValueError("empty voxel memory")
        return self.retained[-1]

    @property
    def confirmed(self) -> bool:
        return len({row.frame_id for row in self.retained}) >= MIN_CONFIRMATION_VIEWS

    def commit(
        self,
        observation: AutomaticMaskObservation,
        evidence: AssociationEvidence | None,
    ) -> None:
        if self.retained and observation.frame_ordinal <= self.last_frame_ordinal:
            raise ValueError("memory commit is not strictly causal")
        self.retained.append(observation)
        self.source_count_total += 1
        self.last_frame_ordinal = observation.frame_ordinal
        if evidence is not None:
            self.association_evidence.append(evidence)
        if len(self.retained) > MAX_RETAINED_VIEWS:
            drop = len(self.retained) - MAX_RETAINED_VIEWS
            self.retained = self.retained[drop:]
            self.dropped_view_count += drop
        union = np.unique(
            np.concatenate([row.voxel_keys for row in self.retained], axis=0), axis=0
        )
        self.hash_keys = _cap_rows(union, MAX_HASH_VOXELS_PER_MEMORY)


@dataclass(frozen=True)
class TrackerAudit:
    observation_count: int
    created_count: int
    associated_count: int
    retired_ttl_count: int
    retired_capacity_count: int
    ignored_hot_bucket_queries: int
    maximum_live_memories: int
    maximum_hash_bucket_size: int
    query_before_commit: bool
    same_frame_self_confirmation_count: int


class CausalVoxelHashTracker:
    """Bounded automatic-mask instance memory with batch causal commits."""

    def __init__(self) -> None:
        self._memories: list[VoxelMemory] = []
        self._active: set[int] = set()
        self._index: dict[tuple[int, int, int], set[int]] = {}
        self._seen_sources: set[str] = set()
        self._last_frame_ordinal: int | None = None
        self._counts = {
            "observation": 0,
            "created": 0,
            "associated": 0,
            "retired_ttl": 0,
            "retired_capacity": 0,
            "ignored_hot": 0,
            "same_frame": 0,
            "max_live": 0,
            "max_bucket": 0,
        }

    @staticmethod
    def _key_rows(keys: np.ndarray) -> Iterable[tuple[int, int, int]]:
        for key in keys:
            yield int(key[0]), int(key[1]), int(key[2])

    def _remove_from_index(self, memory: VoxelMemory) -> None:
        for key in self._key_rows(memory.hash_keys):
            bucket = self._index.get(key)
            if bucket is None:
                continue
            bucket.discard(memory.memory_id)
            if not bucket:
                del self._index[key]

    def _add_to_index(self, memory: VoxelMemory) -> None:
        for key in self._key_rows(memory.hash_keys):
            bucket = self._index.setdefault(key, set())
            bucket.add(memory.memory_id)
            self._counts["max_bucket"] = max(self._counts["max_bucket"], len(bucket))

    def _retire(self, memory_id: int, frame_ordinal: int, *, capacity: bool) -> None:
        if memory_id not in self._active:
            return
        memory = self._memories[memory_id]
        self._remove_from_index(memory)
        self._active.remove(memory_id)
        memory.retired_at_frame_ordinal = frame_ordinal
        self._counts["retired_capacity" if capacity else "retired_ttl"] += 1

    def _candidate_memory_ids(self, observation: AutomaticMaskObservation) -> set[int]:
        candidates: set[int] = set()
        exact_keys = observation.voxel_keys
        for key in self._key_rows(exact_keys):
            bucket = self._index.get(key)
            if bucket is None:
                continue
            if len(bucket) > MAX_TRACKS_PER_HASH_BUCKET:
                self._counts["ignored_hot"] += 1
                continue
            candidates.update(bucket)
        sampled = _cap_rows(exact_keys, RADIUS_QUERY_KEY_CAP)
        expanded = np.unique((sampled[:, None, :] + _NEIGHBOR_OFFSETS[None, :, :]).reshape(-1, 3), axis=0)
        for key in self._key_rows(expanded):
            bucket = self._index.get(key)
            if bucket is None:
                continue
            if len(bucket) > MAX_TRACKS_PER_HASH_BUCKET:
                self._counts["ignored_hot"] += 1
                continue
            candidates.update(bucket)
        return candidates & self._active

    @staticmethod
    def _association(
        observation: AutomaticMaskObservation,
        memory: VoxelMemory,
    ) -> AssociationEvidence | None:
        latest = memory.latest
        distance = float(np.linalg.norm(observation.center - latest.center))
        if distance > MAX_CENTER_DISTANCE_M:
            return None
        best: AssociationEvidence | None = None
        for prior in memory.retained:
            near, left_fraction, right_fraction, harmonic = _near_overlap(observation, prior)
            iou, left_in_right, right_in_left = aabb_overlap(
                observation.lower,
                observation.upper,
                prior.lower,
                prior.upper,
            )
            smaller_containment = max(left_in_right, right_in_left)
            coverage_gate = (
                left_fraction >= MIN_MUTUAL_COVERAGE
                and right_fraction >= MIN_MUTUAL_COVERAGE
            ) or (
                max(left_fraction, right_fraction) >= MIN_DOMINANT_COVERAGE
                and min(left_fraction, right_fraction) >= MIN_SECONDARY_COVERAGE
            )
            if (
                near < MIN_NEAR_OVERLAP_VOXELS
                or not coverage_gate
                or (
                    iou < MIN_AABB_IOU_FOR_MATCH
                    and smaller_containment < MIN_SMALLER_CONTAINMENT_FOR_MATCH
                )
            ):
                continue
            row = AssociationEvidence(
                harmonic=harmonic,
                observation_coverage=left_fraction,
                memory_view_coverage=right_fraction,
                near_overlap_voxels=near,
                aabb_iou=iou,
                smaller_containment=smaller_containment,
                center_distance_m=distance,
            )
            if best is None or (
                row.harmonic,
                row.aabb_iou,
                -row.center_distance_m,
                row.near_overlap_voxels,
            ) > (
                best.harmonic,
                best.aabb_iou,
                -best.center_distance_m,
                best.near_overlap_voxels,
            ):
                best = row
        return best

    def process_frame(
        self,
        frame_id: int,
        frame_ordinal: int,
        observations: Sequence[AutomaticMaskObservation],
    ) -> tuple[dict[str, int | None], dict[str, AssociationEvidence | None]]:
        """Query one whole frame from prior state and then commit the batch."""

        if self._last_frame_ordinal is not None and frame_ordinal <= self._last_frame_ordinal:
            raise ValueError("frame_ordinal must increase strictly")
        rows = tuple(sorted(observations, key=lambda row: row.source_id))
        if any(row.frame_id != frame_id or row.frame_ordinal != frame_ordinal for row in rows):
            raise ValueError("observation belongs to a different frame")
        source_ids = [row.source_id for row in rows]
        if len(source_ids) != len(set(source_ids)) or self._seen_sources.intersection(source_ids):
            raise ValueError("source identities must be globally unique")

        for memory_id in tuple(sorted(self._active)):
            memory = self._memories[memory_id]
            if frame_ordinal - memory.last_frame_ordinal > TTL_KEYFRAMES:
                self._retire(memory_id, frame_ordinal, capacity=False)

        # All edges are computed from the same committed-past snapshot.
        edges: list[tuple[float, float, float, int, str, AssociationEvidence]] = []
        for observation in rows:
            for memory_id in sorted(self._candidate_memory_ids(observation)):
                memory = self._memories[memory_id]
                if memory.last_frame_ordinal >= frame_ordinal:
                    self._counts["same_frame"] += 1
                    continue
                evidence = self._association(observation, memory)
                if evidence is not None:
                    edges.append(
                        (
                            -evidence.harmonic,
                            -evidence.aabb_iou,
                            evidence.center_distance_m,
                            memory_id,
                            observation.source_id,
                            evidence,
                        )
                    )
        edges.sort(key=lambda row: row[:5])
        used_memories: set[int] = set()
        used_sources: set[str] = set()
        assignments: dict[str, int] = {}
        evidence_by_source: dict[str, AssociationEvidence] = {}
        for _, _, _, memory_id, source_id, evidence in edges:
            if memory_id in used_memories or source_id in used_sources:
                continue
            used_memories.add(memory_id)
            used_sources.add(source_id)
            assignments[source_id] = memory_id
            evidence_by_source[source_id] = evidence

        result: dict[str, int | None] = {}
        result_evidence: dict[str, AssociationEvidence | None] = {}
        # Commit matches first; none of these updates influenced the edge graph.
        for observation in rows:
            memory_id = assignments.get(observation.source_id)
            if memory_id is None:
                continue
            memory = self._memories[memory_id]
            self._remove_from_index(memory)
            memory.commit(observation, evidence_by_source[observation.source_id])
            self._add_to_index(memory)
            self._counts["associated"] += 1
            result[observation.source_id] = memory_id
            result_evidence[observation.source_id] = evidence_by_source[observation.source_id]

        for observation in rows:
            if observation.source_id in assignments:
                continue
            if len(self._active) >= MAX_LIVE_MEMORIES:
                evict = min(
                    self._active,
                    key=lambda index: (
                        self._memories[index].last_frame_ordinal,
                        self._memories[index].first_frame_ordinal,
                        index,
                    ),
                )
                self._retire(evict, frame_ordinal, capacity=True)
            memory_id = len(self._memories)
            memory = VoxelMemory(memory_id, frame_ordinal, frame_ordinal)
            memory.commit(observation, None)
            self._memories.append(memory)
            self._active.add(memory_id)
            self._add_to_index(memory)
            self._counts["created"] += 1
            result[observation.source_id] = memory_id
            result_evidence[observation.source_id] = None

        self._seen_sources.update(source_ids)
        self._counts["observation"] += len(rows)
        self._counts["max_live"] = max(self._counts["max_live"], len(self._active))
        self._last_frame_ordinal = frame_ordinal
        return result, result_evidence

    @property
    def memories(self) -> tuple[VoxelMemory, ...]:
        return tuple(self._memories)

    @property
    def audit(self) -> TrackerAudit:
        return TrackerAudit(
            observation_count=self._counts["observation"],
            created_count=self._counts["created"],
            associated_count=self._counts["associated"],
            retired_ttl_count=self._counts["retired_ttl"],
            retired_capacity_count=self._counts["retired_capacity"],
            ignored_hot_bucket_queries=self._counts["ignored_hot"],
            maximum_live_memories=self._counts["max_live"],
            maximum_hash_bucket_size=self._counts["max_bucket"],
            query_before_commit=True,
            same_frame_self_confirmation_count=self._counts["same_frame"],
        )


def _consensus_keys(views: Sequence[AutomaticMaskObservation]) -> np.ndarray:
    union = np.unique(np.concatenate([row.voxel_keys for row in views], axis=0), axis=0)
    support = np.zeros(len(union), dtype=np.int16)
    query = union.astype(np.float64)
    for view in views:
        counts = view.voxel_tree.query_ball_point(
            query, r=1.0, p=np.inf, return_length=True
        )
        support += (counts > 0).astype(np.int16)
    selected = union[support >= CONSENSUS_MIN_VIEW_SUPPORT]
    return _cap_rows(selected, MAX_CONSENSUS_VOXELS)


def _box_from_keys(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(keys) < MIN_CONSENSUS_VOXELS:
        return None
    centers = (keys.astype(np.float64) + 0.5) * VOXEL_SIZE_M
    return _robust_bounds(centers)


def _medoid_view(views: Sequence[AutomaticMaskObservation]) -> AutomaticMaskObservation:
    costs = []
    for left in views:
        cost = sum(
            1.0 - aabb_overlap(left.lower, left.upper, right.lower, right.upper)[0]
            for right in views
        )
        costs.append((cost, -left.confidence, left.source_id, left))
    return min(costs, key=lambda row: row[:3])[3]


@dataclass(frozen=True)
class InstanceProposal:
    memory_id: int
    corners: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    geometry_source: str
    source_ids: tuple[str, ...]
    frame_ids: tuple[int, ...]
    source_count_total: int
    median_association_harmonic: float
    mean_automask_confidence: float
    consensus_voxel_count: int
    consensus_view_iou_median: float
    consensus_loo_stability_iou_median: float
    consensus_used: bool
    admissible: bool
    rejection_reason: str | None

    @property
    def quality_key(self) -> tuple[float, float, float, float, int]:
        return (
            -self.median_association_harmonic,
            -self.mean_automask_confidence,
            -self.consensus_view_iou_median,
            -float(self.consensus_voxel_count),
            self.memory_id,
        )


def build_instance_proposal(memory: VoxelMemory) -> InstanceProposal:
    views = tuple(memory.retained)
    source_ids = tuple(row.source_id for row in views)
    frame_ids = tuple(row.frame_id for row in views)
    medoid = _medoid_view(views)
    medoid_lower, medoid_upper = medoid.lower.copy(), medoid.upper.copy()
    consensus_keys = _consensus_keys(views) if len(views) >= 2 else np.empty((0, 3), dtype=np.int64)
    consensus = _box_from_keys(consensus_keys)
    view_iou = 0.0
    loo_stability = 0.0
    consensus_usable = False
    if consensus is not None:
        consensus_lower, consensus_upper = consensus
        view_iou = float(
            np.median(
                [
                    aabb_overlap(
                        consensus_lower,
                        consensus_upper,
                        row.lower,
                        row.upper,
                    )[0]
                    for row in views
                ]
            )
        )
        loo_boxes = []
        for held_out in range(len(views)):
            loo = _box_from_keys(_consensus_keys(views[:held_out] + views[held_out + 1 :]))
            if loo is not None:
                loo_boxes.append(loo)
        pairwise = [
            aabb_overlap(left[0], left[1], right[0], right[1])[0]
            for index, left in enumerate(loo_boxes)
            for right in loo_boxes[index + 1 :]
        ]
        loo_stability = float(np.median(pairwise)) if pairwise else 0.0
        medoid_center = (medoid_lower + medoid_upper) * 0.5
        consensus_center = (consensus_lower + consensus_upper) * 0.5
        medoid_extent = medoid_upper - medoid_lower
        consensus_extent = consensus_upper - consensus_lower
        extent_ratio = consensus_extent / np.maximum(medoid_extent, 1.0e-9)
        volume_ratio = float(np.prod(consensus_extent) / max(np.prod(medoid_extent), 1.0e-9))
        consensus_usable = (
            view_iou >= CONSENSUS_MIN_VIEW_AABB_IOU
            and loo_stability >= CONSENSUS_MIN_LOO_STABILITY_IOU
            and float(np.linalg.norm(consensus_center - medoid_center))
            <= CONSENSUS_MAX_CENTER_SHIFT_M
            and np.all(extent_ratio >= CONSENSUS_EXTENT_RATIO_RANGE[0])
            and np.all(extent_ratio <= CONSENSUS_EXTENT_RATIO_RANGE[1])
            and CONSENSUS_VOLUME_RATIO_RANGE[0]
            <= volume_ratio
            <= CONSENSUS_VOLUME_RATIO_RANGE[1]
        )
    if consensus_usable and consensus is not None:
        lower, upper = consensus
        geometry_source = "two_view_voxel_consensus"
    else:
        lower, upper = medoid_lower, medoid_upper
        geometry_source = "best_source_medoid"
    extent = upper - lower
    volume = float(np.prod(extent))
    harmonics = [row.harmonic for row in memory.association_evidence]
    median_harmonic = float(np.median(harmonics)) if harmonics else 0.0
    mean_confidence = float(np.mean([row.confidence for row in views])) if views else 0.0
    reason = None
    if len(set(frame_ids)) < MIN_CONFIRMATION_VIEWS:
        reason = "too_few_distinct_views"
    elif len(consensus_keys) < MIN_CONSENSUS_VOXELS:
        reason = "too_few_consensus_voxels"
    elif median_harmonic < MIN_MEDIAN_ASSOCIATION_HARMONIC:
        reason = "association_support_below_threshold"
    elif mean_confidence < MIN_MEAN_AUTOMASK_CONFIDENCE:
        reason = "automask_confidence_below_threshold"
    elif np.any(extent < MIN_OUTPUT_EXTENT_M) or np.any(extent > MAX_OUTPUT_EXTENT_M):
        reason = "output_extent_out_of_range"
    elif volume > MAX_OUTPUT_VOLUME_M3:
        reason = "output_volume_out_of_range"
    corners = bounds_to_corners(lower, upper)
    return InstanceProposal(
        memory_id=memory.memory_id,
        corners=corners,
        lower=np.ascontiguousarray(lower, dtype=np.float64),
        upper=np.ascontiguousarray(upper, dtype=np.float64),
        geometry_source=geometry_source,
        source_ids=source_ids,
        frame_ids=frame_ids,
        source_count_total=memory.source_count_total,
        median_association_harmonic=median_harmonic,
        mean_automask_confidence=mean_confidence,
        consensus_voxel_count=len(consensus_keys),
        consensus_view_iou_median=view_iou,
        consensus_loo_stability_iou_median=loo_stability,
        consensus_used=consensus_usable,
        admissible=reason is None,
        rejection_reason=reason,
    )


def policy_receipt() -> dict[str, object]:
    return {
        "route": "OnlineAnySeg-inspired-lite automatic-mask voxel-hash",
        "training": False,
        "online_learning": False,
        "semantics": False,
        "query_before_commit": True,
        "voxel_size_m": VOXEL_SIZE_M,
        "max_voxels_per_observation": MAX_VOXELS_PER_OBSERVATION,
        "max_hash_voxels_per_memory": MAX_HASH_VOXELS_PER_MEMORY,
        "max_retained_views": MAX_RETAINED_VIEWS,
        "minimum_confirmation_views": MIN_CONFIRMATION_VIEWS,
        "ttl_keyframes": TTL_KEYFRAMES,
        "max_live_memories": MAX_LIVE_MEMORIES,
        "max_tracks_per_hash_bucket": MAX_TRACKS_PER_HASH_BUCKET,
        "radius_query_key_cap": RADIUS_QUERY_KEY_CAP,
        "association": "5cm_hash_exact_plus_bounded_chebyshev1_mutual_support",
        "max_center_distance_m": MAX_CENTER_DISTANCE_M,
        "minimum_near_overlap_voxels": MIN_NEAR_OVERLAP_VOXELS,
        "minimum_mutual_coverage": MIN_MUTUAL_COVERAGE,
        "dominant_secondary_coverage": [MIN_DOMINANT_COVERAGE, MIN_SECONDARY_COVERAGE],
        "minimum_aabb_iou_for_match": MIN_AABB_IOU_FOR_MATCH,
        "minimum_smaller_containment_for_match": MIN_SMALLER_CONTAINMENT_FOR_MATCH,
        "minimum_consensus_voxels": MIN_CONSENSUS_VOXELS,
        "maximum_consensus_voxels": MAX_CONSENSUS_VOXELS,
        "minimum_median_association_harmonic": MIN_MEDIAN_ASSOCIATION_HARMONIC,
        "minimum_mean_automask_confidence": MIN_MEAN_AUTOMASK_CONFIDENCE,
        "consensus_minimum_view_aabb_iou": CONSENSUS_MIN_VIEW_AABB_IOU,
        "consensus_minimum_loo_stability_iou": CONSENSUS_MIN_LOO_STABILITY_IOU,
    }


__all__ = [
    "AssociationEvidence",
    "AutomaticMaskObservation",
    "CausalVoxelHashTracker",
    "InstanceProposal",
    "TrackerAudit",
    "VoxelMemory",
    "aabb_overlap",
    "bounds_to_corners",
    "build_instance_proposal",
    "policy_receipt",
]
