"""Training-free causal sparse query memory for BoxFusion tracks.

This module borrows only the sparse 3D key-to-query indexing idea from
MoonSeg3R.  It deliberately contains no neural network, learned parameter, or
semantic update.  The first integration stage is observer-only: current CuTR
proposals query an index built from *previous* keyframes, native BoxFusion
association supplies a paired diagnostic target, and the index is updated only
after native association has completed.

Boxes use the BoxFusion convention ``[N, 8, 3]`` in metric world coordinates.
Stable track ids, rather than mutable row indices, are stored in postings.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from time import perf_counter_ns
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_MOON_QIM_LITE_CONFIG = {
    "enabled": False,
    # Active association is intentionally unavailable in the first stage.
    "observer_only": True,
    "voxel_size_m": 0.30,
    "samples_per_axis": 3,
    "neighbor_radius": 1,
    "max_candidates_per_query": 8,
    "max_tracks": 1024,
    "track_ttl_keyframes": 80,
    "max_postings_per_key": 32,
}


def _strict_bool(name: str, value: object) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _strict_int(name: str, value: object, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_float(name: str, value: object, minimum: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return result


def resolve_moon_qim_lite_config(
    config: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Return a strictly validated QIM-lite configuration."""

    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("moon_qim_lite config must be a mapping")
    unknown = sorted(set(config) - set(DEFAULT_MOON_QIM_LITE_CONFIG))
    if unknown:
        raise ValueError(
            "Unknown moon_qim_lite config key(s): " + ", ".join(unknown)
        )

    resolved = dict(DEFAULT_MOON_QIM_LITE_CONFIG)
    resolved.update(config)
    resolved["enabled"] = _strict_bool(
        "moon_qim_lite.enabled", resolved["enabled"]
    )
    resolved["observer_only"] = _strict_bool(
        "moon_qim_lite.observer_only", resolved["observer_only"]
    )
    if resolved["enabled"] and not resolved["observer_only"]:
        raise ValueError(
            "moon_qim_lite active association is not authorized; "
            "observer_only must remain true"
        )
    resolved["voxel_size_m"] = _finite_float(
        "moon_qim_lite.voxel_size_m", resolved["voxel_size_m"], 1e-6
    )
    resolved["samples_per_axis"] = _strict_int(
        "moon_qim_lite.samples_per_axis", resolved["samples_per_axis"], 1
    )
    if resolved["samples_per_axis"] > 5:
        raise ValueError("moon_qim_lite.samples_per_axis must not exceed 5")
    for key, minimum in (
        ("neighbor_radius", 0),
        ("max_candidates_per_query", 1),
        ("max_tracks", 1),
        ("track_ttl_keyframes", 0),
        ("max_postings_per_key", 1),
    ):
        resolved[key] = _strict_int(
            f"moon_qim_lite.{key}", resolved[key], minimum
        )
    if resolved["neighbor_radius"] > 2:
        raise ValueError("moon_qim_lite.neighbor_radius must not exceed 2")
    return resolved


def _as_numpy(value: object, name: str) -> np.ndarray:
    candidate = value
    if hasattr(candidate, "detach"):
        candidate = candidate.detach()
    if hasattr(candidate, "cpu"):
        candidate = candidate.cpu()
    if hasattr(candidate, "numpy"):
        candidate = candidate.numpy()
    try:
        return np.asarray(candidate)
    except Exception as error:
        raise ValueError(f"{name} cannot be converted to NumPy") from error


def _validated_ids(value: object, count: int, name: str) -> np.ndarray:
    ids = _as_numpy(value, name)
    if ids.ndim != 1 or len(ids) != count:
        raise ValueError(f"{name} must have shape [{count}]")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f"{name} must contain integers")
    ids = ids.astype(np.int64, copy=False)
    if np.any(ids < 0):
        raise ValueError(f"{name} must be non-negative")
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{name} must be unique")
    return np.array(ids, dtype=np.int64, order="C", copy=True)


def _validated_corners(value: object, name: str) -> np.ndarray:
    corners = _as_numpy(value, name).astype(np.float64, copy=False)
    if corners.ndim != 3 or corners.shape[1:] != (8, 3):
        raise ValueError(f"{name} must have shape [N, 8, 3]")
    if not np.isfinite(corners).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.array(corners, dtype=np.float64, order="C", copy=True)


def _frame_id(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("frame_id must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError("frame_id must be a non-negative integer")
    return result


def _aabb_iou(
    lower_a: np.ndarray,
    upper_a: np.ndarray,
    lower_b: np.ndarray,
    upper_b: np.ndarray,
) -> float:
    intersection = np.maximum(
        np.minimum(upper_a, upper_b) - np.maximum(lower_a, lower_b), 0.0
    )
    intersection_volume = float(np.prod(intersection))
    volume_a = float(np.prod(np.maximum(upper_a - lower_a, 0.0)))
    volume_b = float(np.prod(np.maximum(upper_b - lower_b, 0.0)))
    union = volume_a + volume_b - intersection_volume
    return intersection_volume / union if union > 0.0 else 0.0


def _normalized_fusion_groups(
    fusion_groups: Sequence[Iterable[int]], name: str
) -> Tuple[Tuple[int, ...], ...]:
    if isinstance(fusion_groups, (str, bytes)) or not isinstance(
        fusion_groups, Sequence
    ):
        raise ValueError(f"{name} must be a sequence of integer sequences")
    normalized = []
    for index, group in enumerate(fusion_groups):
        if isinstance(group, (str, bytes)):
            raise ValueError(f"{name}[{index}] must be an integer sequence")
        try:
            values = tuple(group)
        except TypeError as error:
            raise ValueError(
                f"{name}[{index}] must be an integer sequence"
            ) from error
        if not values:
            raise ValueError(f"{name}[{index}] must not be empty")
        row = []
        for value in values:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, Integral
            ):
                raise ValueError(f"{name}[{index}] must contain integers")
            value = int(value)
            if value < 0:
                raise ValueError(f"{name}[{index}] must be non-negative")
            row.append(value)
        normalized.append(tuple(sorted(set(row))))
    return tuple(normalized)


def derive_native_target_track_ids(
    *,
    proposal_ids: object,
    previous_fusion_groups: Sequence[Iterable[int]],
    previous_stable_ids: object,
    current_fusion_groups: Sequence[Iterable[int]],
    association_events: Sequence[Mapping[str, object]] = (),
) -> Tuple[Optional[Tuple[int, ...]], ...]:
    """Map current proposal ids to native pre-keyframe stable track ids.

    ``None`` denotes a proposal whose source id disappeared from all retained
    fusion groups, so it cannot be scored without changing native association.
    An empty tuple denotes a retained native birth.  A non-empty tuple contains
    the stable ids of pre-keyframe groups merged with the proposal.
    """

    previous_groups = _normalized_fusion_groups(
        previous_fusion_groups, "previous_fusion_groups"
    )
    current_groups = _normalized_fusion_groups(
        current_fusion_groups, "current_fusion_groups"
    )
    stable_ids = _validated_ids(
        previous_stable_ids,
        len(previous_groups),
        "previous_stable_ids",
    )
    proposal_array = _as_numpy(proposal_ids, "proposal_ids")
    if proposal_array.ndim != 1 or not np.issubdtype(
        proposal_array.dtype, np.integer
    ):
        raise ValueError("proposal_ids must be a one-dimensional integer array")
    proposal_array = proposal_array.astype(np.int64, copy=False)
    if np.any(proposal_array < 0) or len(np.unique(proposal_array)) != len(
        proposal_array
    ):
        raise ValueError("proposal_ids must be unique and non-negative")

    previous_member_to_stable: Dict[int, set[int]] = {}
    for group, stable_id in zip(previous_groups, stable_ids):
        for member in group:
            previous_member_to_stable.setdefault(member, set()).add(
                int(stable_id)
            )
    if isinstance(association_events, (str, bytes)) or not isinstance(
        association_events, Sequence
    ):
        raise ValueError("association_events must be a sequence of mappings")

    # Native fusion groups stop recording source observations once their
    # fixed five-view budget is full.  Explicit association events therefore
    # provide the authoritative proposal-to-winner trace.  A tiny union-find
    # also handles chained new-new then new-old merges in one NMS pass.
    parent: Dict[int, int] = {}

    def find(value: int) -> int:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    evidenced_members = set()
    for group in current_groups:
        evidenced_members.update(group)
        for member in group[1:]:
            union(group[0], member)
    for index, event in enumerate(association_events):
        if not isinstance(event, Mapping):
            raise ValueError(
                f"association_events[{index}] must be a mapping"
            )
        unknown = set(event) - {"stage", "winner_members", "loser_members"}
        if unknown:
            raise ValueError(
                f"association_events[{index}] has unknown keys: "
                + ", ".join(sorted(unknown))
            )
        members = []
        for key in ("winner_members", "loser_members"):
            groups = _normalized_fusion_groups(
                [event.get(key, ())],
                f"association_events[{index}].{key}",
            )
            members.extend(groups[0])
        evidenced_members.update(members)
        for member in members[1:]:
            union(members[0], member)

    components: Dict[int, set[int]] = {}
    for member in parent:
        components.setdefault(find(member), set()).add(member)

    targets = []
    for proposal_id in proposal_array:
        proposal_id = int(proposal_id)
        if proposal_id not in evidenced_members:
            targets.append(None)
            continue
        matched = set()
        component = components.get(find(proposal_id), {proposal_id})
        for member in component:
            matched.update(previous_member_to_stable.get(member, ()))
        targets.append(tuple(sorted(matched)))
    return tuple(targets)


class CausalFusionIdRegistry:
    """Assign row-aligned track IDs without ever transferring an old ID.

    Normal BoxFusion groups have unique, immutable minimum source IDs.  In a
    rare collision, however, recomputing a stateless collision repair can make
    an ID jump to a newly inserted group.  This registry inherits IDs through
    maximum source-member overlap and allocates never-reused synthetic IDs to
    unmatched groups.
    """

    _SYNTHETIC_START = 1 << 62

    def __init__(self) -> None:
        self._groups: Tuple[Tuple[int, ...], ...] = ()
        self._ids = np.empty((0,), dtype=np.int64)
        self._used_ids: set[int] = set()
        self._next_synthetic_id = self._SYNTHETIC_START

    def reset(self) -> None:
        self.__init__()

    def ids_for(
        self, fusion_groups: Sequence[Iterable[int]]
    ) -> np.ndarray:
        groups = _normalized_fusion_groups(fusion_groups, "fusion_groups")
        if groups != self._groups:
            raise ValueError(
                "fusion groups changed without a causal registry update"
            )
        return np.array(self._ids, dtype=np.int64, order="C", copy=True)

    def _allocate(self, preferred: int) -> int:
        preferred = int(preferred)
        if preferred not in self._used_ids:
            result = preferred
        else:
            while self._next_synthetic_id in self._used_ids:
                self._next_synthetic_id += 1
            if self._next_synthetic_id >= (1 << 63):
                raise RuntimeError("QIM stable ID space is exhausted")
            result = self._next_synthetic_id
            self._next_synthetic_id += 1
        self._used_ids.add(result)
        return result

    def update(
        self, fusion_groups: Sequence[Iterable[int]]
    ) -> np.ndarray:
        current = _normalized_fusion_groups(fusion_groups, "fusion_groups")
        assigned: list[Optional[int]] = [None] * len(current)
        used_previous = set()
        candidates = []
        previous_sets = [set(group) for group in self._groups]
        current_sets = [set(group) for group in current]
        for current_index, current_members in enumerate(current_sets):
            for previous_index, previous_members in enumerate(previous_sets):
                intersection = len(current_members & previous_members)
                if not intersection:
                    continue
                union_size = len(current_members | previous_members)
                candidates.append(
                    (
                        -intersection,
                        -(intersection / union_size),
                        int(self._ids[previous_index]),
                        current[current_index],
                        current_index,
                        previous_index,
                    )
                )
        for *_, current_index, previous_index in sorted(candidates):
            if assigned[current_index] is not None:
                continue
            if previous_index in used_previous:
                continue
            assigned[current_index] = int(self._ids[previous_index])
            used_previous.add(previous_index)
        for index, group in enumerate(current):
            if assigned[index] is None:
                assigned[index] = self._allocate(group[0])

        self._groups = current
        self._ids = np.asarray(assigned, dtype=np.int64)
        if len(np.unique(self._ids)) != len(self._ids):
            raise RuntimeError("causal QIM registry produced duplicate IDs")
        return np.array(self._ids, dtype=np.int64, order="C", copy=True)


@dataclass(frozen=True)
class QIMCandidate:
    """One sparse-memory candidate returned for a current proposal."""

    track_id: int
    shared_key_count: int
    shared_key_fraction: float
    center_distance_m: float
    aabb_iou: float
    age_keyframes: int
    active_at_last_commit: bool


@dataclass(frozen=True)
class QIMQueryBatch:
    """Immutable result of querying the history available before a keyframe."""

    scene_id: str
    frame_id: int
    proposal_ids: Tuple[int, ...]
    candidates: Tuple[Tuple[QIMCandidate, ...], ...]
    history_max_frame_id: Optional[int]
    query_ms: float


@dataclass
class _TrackRecord:
    track_id: int
    keys: Tuple[Tuple[int, int, int], ...]
    indexed_keys: set[Tuple[int, int, int]]
    lower: Tuple[float, float, float]
    upper: Tuple[float, float, float]
    center: Tuple[float, float, float]
    last_seen_step: int
    last_seen_frame_id: int


class MoonQIMLiteObserver:
    """Bounded sparse world-key index with paired native-association metrics."""

    def __init__(self, config: Optional[Mapping[str, object]] = None):
        self.config = resolve_moon_qim_lite_config(config)
        self.enabled = bool(self.config["enabled"])
        self.observer_only = bool(self.config["observer_only"])
        fractions = np.linspace(
            0.0, 1.0, int(self.config["samples_per_axis"]), dtype=np.float64
        )
        self._sample_grid = np.stack(
            np.meshgrid(fractions, fractions, fractions, indexing="ij"),
            axis=-1,
        ).reshape(-1, 3)
        radius = int(self.config["neighbor_radius"])
        self._neighbor_offsets = tuple(
            (x, y, z)
            for x in range(-radius, radius + 1)
            for y in range(-radius, radius + 1)
            for z in range(-radius, radius + 1)
        )
        self._scene_id: Optional[str] = None
        self._records: Dict[int, _TrackRecord] = {}
        self._postings: Dict[Tuple[int, int, int], set[int]] = {}
        self._step = -1
        self._last_query_frame_id: Optional[int] = None
        self._last_update_frame_id: Optional[int] = None
        self._pending_batch: Optional[QIMQueryBatch] = None
        self._pending_batch_observed = False
        self._stats = self._new_stats()

    @staticmethod
    def _new_stats() -> Dict[str, object]:
        return {
            "queries": 0,
            "proposals": 0,
            "proposals_with_candidates": 0,
            "candidate_total": 0,
            "native_matches": 0,
            "native_births": 0,
            "native_unresolved": 0,
            "recall_at_1": 0,
            "recall_at_3": 0,
            "recall_at_k": 0,
            "updates": 0,
            "evicted_tracks": 0,
            "expired_tracks": 0,
            "query_ms_total": 0.0,
            "query_ms_max": 0.0,
            "update_ms_total": 0.0,
            "update_ms_max": 0.0,
            "pipeline_query_calls": 0,
            "pipeline_query_ms_total": 0.0,
            "pipeline_query_ms_max": 0.0,
            "pipeline_update_calls": 0,
            "pipeline_update_ms_total": 0.0,
            "pipeline_update_ms_max": 0.0,
            "max_tracks_observed": 0,
            "max_keys_observed": 0,
            "max_postings_observed": 0,
        }

    @property
    def scene_id(self) -> Optional[str]:
        return self._scene_id

    def reset_scene(self, scene_id: str) -> None:
        scene_id = str(scene_id)
        if not scene_id:
            raise ValueError("scene_id must not be empty")
        self._scene_id = scene_id
        self._records.clear()
        self._postings.clear()
        self._step = -1
        self._last_query_frame_id = None
        self._last_update_frame_id = None
        self._pending_batch = None
        self._pending_batch_observed = False
        self._stats = self._new_stats()

    def _bind_scene(self, scene_id: str) -> str:
        scene_id = str(scene_id)
        if not scene_id:
            raise ValueError("scene_id must not be empty")
        if self._scene_id is None:
            self.reset_scene(scene_id)
        elif self._scene_id != scene_id:
            raise ValueError(
                f"moon_qim_lite is bound to {self._scene_id}, not {scene_id}"
            )
        return scene_id

    def _box_keys(
        self, lower: np.ndarray, upper: np.ndarray
    ) -> Tuple[Tuple[int, int, int], ...]:
        points = lower[None, :] + self._sample_grid * (
            upper - lower
        )[None, :]
        # Even sampling counts do not contain 0.5.  Always include the center
        # because queries use a proposal-center key for constant work per box.
        points = np.concatenate(
            (points, ((lower + upper) / 2.0)[None, :]), axis=0
        )
        quantized = np.floor(
            points / float(self.config["voxel_size_m"])
        ).astype(np.int64)
        return tuple(
            sorted({tuple(int(value) for value in row) for row in quantized})
        )

    def _remove_track(self, track_id: int) -> None:
        record = self._records.pop(int(track_id), None)
        if record is None:
            return
        for key in tuple(record.indexed_keys):
            posting = self._postings.get(key)
            if posting is None:
                continue
            posting.discard(int(track_id))
            if not posting:
                del self._postings[key]

    def _trim_posting(self, key: Tuple[int, int, int]) -> None:
        posting = self._postings[key]
        limit = int(self.config["max_postings_per_key"])
        if len(posting) <= limit:
            return
        keep = set(
            sorted(
                posting,
                key=lambda track_id: (
                    -self._records[track_id].last_seen_step,
                    track_id,
                ),
            )[:limit]
        )
        dropped = posting - keep
        posting.intersection_update(keep)
        for track_id in dropped:
            record = self._records.get(track_id)
            if record is not None:
                record.indexed_keys.discard(key)

    def query(
        self,
        *,
        scene_id: str,
        frame_id: int,
        proposal_ids: object,
        proposal_corners_world: object,
    ) -> QIMQueryBatch:
        """Retrieve candidate stable ids without updating the index."""

        start = perf_counter_ns()
        if not self.enabled:
            raise RuntimeError("moon_qim_lite observer is disabled")
        if self._pending_batch is not None:
            raise ValueError(
                "previous QIM query must be closed by an update first"
            )
        scene_id = self._bind_scene(scene_id)
        frame_id = _frame_id(frame_id)
        if (
            self._last_query_frame_id is not None
            and frame_id <= self._last_query_frame_id
        ):
            raise ValueError("QIM query frame ids must be strictly increasing")
        if (
            self._last_update_frame_id is not None
            and frame_id <= self._last_update_frame_id
        ):
            raise ValueError("QIM query must precede the same-frame update")

        corners = _validated_corners(
            proposal_corners_world, "proposal_corners_world"
        )
        ids = _validated_ids(proposal_ids, len(corners), "proposal_ids")
        rows = []
        candidate_total = 0
        rows_with_candidates = 0
        limit = int(self.config["max_candidates_per_query"])
        for box in corners:
            lower = np.min(box, axis=0)
            upper = np.max(box, axis=0)
            center = np.mean(box, axis=0)
            # Track records retain sparse volume keys.  A current proposal
            # probes only three bounded samples (center and diagonal bounds)
            # plus their fixed local neighborhoods.  This keeps query work
            # O(P * neighborhood), independent of samples_per_axis.
            center_key_array = np.floor(
                center / float(self.config["voxel_size_m"])
            ).astype(np.int64)
            bound_keys = np.floor(
                np.stack((lower, upper), axis=0)
                / float(self.config["voxel_size_m"])
            ).astype(np.int64)
            query_keys = tuple(
                sorted(
                    {
                        tuple(int(value) for value in center_key_array),
                        tuple(int(value) for value in bound_keys[0]),
                        tuple(int(value) for value in bound_keys[1]),
                    }
                )
            )
            shared_counts: Dict[int, int] = {}
            for key in query_keys:
                per_key_tracks = set()
                for offset in self._neighbor_offsets:
                    neighbor = (
                        key[0] + offset[0],
                        key[1] + offset[1],
                        key[2] + offset[2],
                    )
                    per_key_tracks.update(self._postings.get(neighbor, ()))
                for track_id in per_key_tracks:
                    shared_counts[track_id] = shared_counts.get(track_id, 0) + 1

            candidates = []
            denominator = max(len(query_keys), 1)
            for track_id, shared_count in shared_counts.items():
                record = self._records.get(track_id)
                if record is None:
                    continue
                record_lower = np.asarray(record.lower, dtype=np.float64)
                record_upper = np.asarray(record.upper, dtype=np.float64)
                record_center = np.asarray(record.center, dtype=np.float64)
                candidates.append(
                    QIMCandidate(
                        track_id=int(track_id),
                        shared_key_count=int(shared_count),
                        shared_key_fraction=float(shared_count / denominator),
                        center_distance_m=float(
                            np.linalg.norm(center - record_center)
                        ),
                        aabb_iou=float(
                            _aabb_iou(
                                lower,
                                upper,
                                record_lower,
                                record_upper,
                            )
                        ),
                        age_keyframes=max(
                            int(self._step - record.last_seen_step), 0
                        ),
                        active_at_last_commit=(
                            record.last_seen_step == self._step
                        ),
                    )
                )
            candidates.sort(
                key=lambda candidate: (
                    -candidate.shared_key_count,
                    -candidate.aabb_iou,
                    candidate.center_distance_m,
                    candidate.track_id,
                )
            )
            row = tuple(candidates[:limit])
            rows.append(row)
            candidate_total += len(row)
            rows_with_candidates += int(bool(row))

        elapsed_ms = (perf_counter_ns() - start) / 1e6
        self._last_query_frame_id = frame_id
        self._stats["queries"] += 1
        self._stats["proposals"] += len(corners)
        self._stats["proposals_with_candidates"] += rows_with_candidates
        self._stats["candidate_total"] += candidate_total
        self._stats["query_ms_total"] += elapsed_ms
        self._stats["query_ms_max"] = max(
            float(self._stats["query_ms_max"]), elapsed_ms
        )
        batch = QIMQueryBatch(
            scene_id=scene_id,
            frame_id=frame_id,
            proposal_ids=tuple(int(value) for value in ids),
            candidates=tuple(rows),
            history_max_frame_id=self._last_update_frame_id,
            query_ms=float(elapsed_ms),
        )
        self._pending_batch = batch
        self._pending_batch_observed = False
        return batch

    def observe_native_targets(
        self,
        batch: QIMQueryBatch,
        native_target_track_ids: Sequence[Optional[Iterable[int]]],
    ) -> None:
        """Compare retrieval with unmodified BoxFusion association.

        Each target entry has one of three meanings: a non-empty iterable is a
        native match to one or more pre-keyframe tracks, an empty iterable is a
        native birth, and ``None`` means the native path did not retain enough
        identity information to score this proposal.
        """

        if batch is not self._pending_batch:
            raise ValueError("native targets require the pending QIM batch")
        if self._pending_batch_observed:
            raise ValueError("pending QIM batch was already observed")
        if batch.scene_id != self._scene_id:
            raise ValueError("QIM batch belongs to a different scene")
        if len(native_target_track_ids) != len(batch.proposal_ids):
            raise ValueError("native targets must align with QIM proposals")
        for candidates, raw_targets in zip(
            batch.candidates, native_target_track_ids
        ):
            if raw_targets is None:
                self._stats["native_unresolved"] += 1
                continue
            targets = {int(value) for value in raw_targets}
            if any(value < 0 for value in targets):
                raise ValueError("native target ids must be non-negative")
            if not targets:
                self._stats["native_births"] += 1
                continue
            self._stats["native_matches"] += 1
            ranked = [candidate.track_id for candidate in candidates]
            self._stats["recall_at_1"] += int(
                bool(targets.intersection(ranked[:1]))
            )
            self._stats["recall_at_3"] += int(
                bool(targets.intersection(ranked[:3]))
            )
            self._stats["recall_at_k"] += int(
                bool(targets.intersection(ranked))
            )
        self._pending_batch_observed = True

    def update(
        self,
        *,
        scene_id: str,
        frame_id: int,
        track_ids: object,
        track_corners_world: object,
    ) -> None:
        """Synchronize current tracks after all native association is done."""

        start = perf_counter_ns()
        if not self.enabled:
            raise RuntimeError("moon_qim_lite observer is disabled")
        self._bind_scene(scene_id)
        frame_id = _frame_id(frame_id)
        if (
            self._pending_batch is not None
            and frame_id != self._pending_batch.frame_id
        ):
            raise ValueError(
                "QIM update must close the pending query at the same frame"
            )
        if (
            self._last_query_frame_id is not None
            and frame_id < self._last_query_frame_id
        ):
            raise ValueError("QIM update cannot precede the latest query")
        if (
            self._last_update_frame_id is not None
            and frame_id <= self._last_update_frame_id
        ):
            raise ValueError("QIM update frame ids must be strictly increasing")
        corners = _validated_corners(track_corners_world, "track_corners_world")
        ids = _validated_ids(track_ids, len(corners), "track_ids")
        self._step += 1

        lowers = np.min(corners, axis=1)
        uppers = np.max(corners, axis=1)
        centers = np.mean(corners, axis=1)
        for track_id, lower, upper, center in zip(
            ids, lowers, uppers, centers
        ):
            track_id = int(track_id)
            lower_tuple = tuple(float(value) for value in lower)
            upper_tuple = tuple(float(value) for value in upper)
            center_tuple = tuple(float(value) for value in center)
            previous = self._records.get(track_id)
            if (
                previous is not None
                and previous.lower == lower_tuple
                and previous.upper == upper_tuple
            ):
                # Most global tracks are byte-identical on a keyframe.  Merely
                # refresh liveness; no quantization or posting write is needed.
                previous.center = center_tuple
                previous.last_seen_step = self._step
                previous.last_seen_frame_id = frame_id
                # A crowded cell may previously have evicted this track from
                # one or more capped postings.  A new observation makes it
                # recent again, so restore only those missing memberships.
                if len(previous.indexed_keys) != len(previous.keys):
                    for key in previous.keys:
                        if key in previous.indexed_keys:
                            continue
                        previous.indexed_keys.add(key)
                        self._postings.setdefault(key, set()).add(track_id)
                        self._trim_posting(key)
                continue
            keys = self._box_keys(lower, upper)
            record = _TrackRecord(
                track_id=track_id,
                keys=keys,
                indexed_keys=(
                    set(previous.indexed_keys)
                    if previous is not None and previous.keys == keys
                    else set()
                ),
                lower=lower_tuple,
                upper=upper_tuple,
                center=center_tuple,
                last_seen_step=self._step,
                last_seen_frame_id=frame_id,
            )
            if previous is not None and previous.keys == keys:
                # Most global tracks are unchanged on a keyframe.  Refresh
                # geometry/liveness without rebuilding identical postings.
                self._records[track_id] = record
                continue
            self._remove_track(track_id)
            self._records[track_id] = record
            for key in record.keys:
                record.indexed_keys.add(key)
                self._postings.setdefault(key, set()).add(track_id)
                self._trim_posting(key)

        ttl = int(self.config["track_ttl_keyframes"])
        expired = [
            track_id
            for track_id, record in self._records.items()
            if self._step - record.last_seen_step > ttl
        ]
        for track_id in sorted(expired):
            self._remove_track(track_id)
        self._stats["expired_tracks"] += len(expired)

        overflow = max(0, len(self._records) - int(self.config["max_tracks"]))
        if overflow:
            victims = sorted(
                self._records.values(),
                key=lambda record: (record.last_seen_step, record.track_id),
            )[:overflow]
            for record in victims:
                self._remove_track(record.track_id)
            self._stats["evicted_tracks"] += len(victims)

        elapsed_ms = (perf_counter_ns() - start) / 1e6
        self._last_update_frame_id = frame_id
        self._pending_batch = None
        self._pending_batch_observed = False
        self._stats["updates"] += 1
        self._stats["update_ms_total"] += elapsed_ms
        self._stats["update_ms_max"] = max(
            float(self._stats["update_ms_max"]), elapsed_ms
        )
        posting_count = sum(len(values) for values in self._postings.values())
        self._stats["max_tracks_observed"] = max(
            int(self._stats["max_tracks_observed"]), len(self._records)
        )
        self._stats["max_keys_observed"] = max(
            int(self._stats["max_keys_observed"]), len(self._postings)
        )
        self._stats["max_postings_observed"] = max(
            int(self._stats["max_postings_observed"]), posting_count
        )

    def record_pipeline_timing(
        self,
        *,
        query_ms: Optional[float] = None,
        update_ms: Optional[float] = None,
    ) -> None:
        """Record wrapper time including tensor copies and ID bookkeeping."""

        if query_ms is None and update_ms is None:
            raise ValueError("at least one pipeline timing value is required")
        for stage, value in (("query", query_ms), ("update", update_ms)):
            if value is None:
                continue
            value = _finite_float(f"pipeline_{stage}_ms", value, 0.0)
            self._stats[f"pipeline_{stage}_calls"] += 1
            self._stats[f"pipeline_{stage}_ms_total"] += value
            self._stats[f"pipeline_{stage}_ms_max"] = max(
                float(self._stats[f"pipeline_{stage}_ms_max"]), value
            )

    def snapshot(self) -> Dict[str, object]:
        """Return a deterministic read-only diagnostic representation."""

        return {
            "scene_id": self._scene_id,
            "history_max_frame_id": self._last_update_frame_id,
            "track_ids": tuple(sorted(self._records)),
            "key_count": len(self._postings),
            "posting_count": sum(
                len(values) for values in self._postings.values()
            ),
        }

    def summary(self) -> Dict[str, object]:
        matches = int(self._stats["native_matches"])
        queries = int(self._stats["queries"])
        updates = int(self._stats["updates"])
        pipeline_queries = int(self._stats["pipeline_query_calls"])
        pipeline_updates = int(self._stats["pipeline_update_calls"])
        result = dict(self._stats)
        result.update(
            {
                "schema": "boxfusion.moon_qim_lite_observer.v1",
                "enabled": self.enabled,
                "observer_only": self.observer_only,
                "training_free": True,
                "causal": True,
                "semantic_access": False,
                "semantic_mutation": False,
                "scene_id": self._scene_id,
                "tracks_retained": len(self._records),
                "keys_retained": len(self._postings),
                "postings_retained": sum(
                    len(values) for values in self._postings.values()
                ),
                "recall_at_1_rate": (
                    int(self._stats["recall_at_1"]) / matches
                    if matches
                    else None
                ),
                "recall_at_3_rate": (
                    int(self._stats["recall_at_3"]) / matches
                    if matches
                    else None
                ),
                "recall_at_k_rate": (
                    int(self._stats["recall_at_k"]) / matches
                    if matches
                    else None
                ),
                "query_ms_mean": (
                    float(self._stats["query_ms_total"]) / queries
                    if queries
                    else 0.0
                ),
                "update_ms_mean": (
                    float(self._stats["update_ms_total"]) / updates
                    if updates
                    else 0.0
                ),
                "pipeline_query_ms_mean": (
                    float(self._stats["pipeline_query_ms_total"])
                    / pipeline_queries
                    if pipeline_queries
                    else 0.0
                ),
                "pipeline_update_ms_mean": (
                    float(self._stats["pipeline_update_ms_total"])
                    / pipeline_updates
                    if pipeline_updates
                    else 0.0
                ),
            }
        )
        return result

    def summary_line(self) -> str:
        summary = self.summary()
        def rate(value: object) -> str:
            return "nan" if value is None else f"{float(value):.4f}"

        return (
            "Moon-QIM-lite observer summary | "
            f"queries/proposals={summary['queries']}/{summary['proposals']}, "
            f"native_matches/births/unresolved="
            f"{summary['native_matches']}/{summary['native_births']}/"
            f"{summary['native_unresolved']}, "
            f"R@1/R@3/R@K={rate(summary['recall_at_1_rate'])}/"
            f"{rate(summary['recall_at_3_rate'])}/"
            f"{rate(summary['recall_at_k_rate'])}, "
            f"query_mean/max_ms={summary['query_ms_mean']:.3f}/"
            f"{summary['query_ms_max']:.3f}, "
            f"update_mean/max_ms={summary['update_ms_mean']:.3f}/"
            f"{summary['update_ms_max']:.3f}, "
            f"pipeline_query/update_mean_ms="
            f"{summary['pipeline_query_ms_mean']:.3f}/"
            f"{summary['pipeline_update_ms_mean']:.3f}, "
            f"tracks/keys/postings={summary['tracks_retained']}/"
            f"{summary['keys_retained']}/{summary['postings_retained']}"
        )


def build_moon_qim_lite(config: Mapping[str, object]) -> MoonQIMLiteObserver:
    if not isinstance(config, Mapping):
        raise ValueError("application config must be a mapping")
    return MoonQIMLiteObserver(config.get("moon_qim_lite", {}))


__all__ = [
    "DEFAULT_MOON_QIM_LITE_CONFIG",
    "CausalFusionIdRegistry",
    "MoonQIMLiteObserver",
    "QIMCandidate",
    "QIMQueryBatch",
    "build_moon_qim_lite",
    "derive_native_target_track_ids",
    "resolve_moon_qim_lite_config",
]
