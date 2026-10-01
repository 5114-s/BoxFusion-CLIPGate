"""Training-free geometry primitives for the R15 novel-object branch.

The module deliberately contains no dataset, annotation, evaluator, or model
API.  It turns already-confirmed, past-only mask-depth point fragments into a
bounded causal instance memory, refines each memory against local RGB-D depth
components, and selects one of several robust OBB hypotheses using only
cross-view geometric evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree


SIGNS = np.asarray(
    [
        (-1.0, -1.0, -1.0),
        (-1.0, -1.0, +1.0),
        (-1.0, +1.0, -1.0),
        (-1.0, +1.0, +1.0),
        (+1.0, -1.0, -1.0),
        (+1.0, -1.0, +1.0),
        (+1.0, +1.0, -1.0),
        (+1.0, +1.0, +1.0),
    ],
    dtype=np.float64,
)


def _points(value: object, label: str, *, allow_empty: bool = False) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1:] != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite [N,3]")
    if not allow_empty and len(result) == 0:
        raise ValueError(f"{label} must not be empty")
    return result


def _box(value: object, label: str = "corners") -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (8, 3) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite [8,3]")
    if np.any(np.ptp(result, axis=0) <= 0):
        raise ValueError(f"{label} must have positive extent")
    return result


def voxel_downsample(points: object, voxel_m: float) -> np.ndarray:
    """Return deterministic centroid points for signed-floor voxels."""

    values = _points(points, "points", allow_empty=True)
    if not math.isfinite(voxel_m) or voxel_m <= 0:
        raise ValueError("voxel_m must be positive")
    if not len(values):
        return np.empty((0, 3), dtype=np.float32)
    keys = np.floor(values / voxel_m).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys = keys[order]
    values = values[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1))]
    counts = np.diff(np.r_[starts, len(keys)])
    sums = np.add.reduceat(values, starts, axis=0)
    return np.ascontiguousarray((sums / counts[:, None]).astype(np.float32))


def aabb_bounds(corners: object) -> tuple[np.ndarray, np.ndarray]:
    box = _box(corners)
    return box.min(axis=0), box.max(axis=0)


def aabb_overlap(left: object, right: object) -> tuple[float, float, float]:
    ll, lu = aabb_bounds(left)
    rl, ru = aabb_bounds(right)
    intersection = float(np.prod(np.maximum(np.minimum(lu, ru) - np.maximum(ll, rl), 0)))
    lv = float(np.prod(lu - ll))
    rv = float(np.prod(ru - rl))
    union = lv + rv - intersection
    return (
        0.0 if union <= 0 else intersection / union,
        0.0 if lv <= 0 else intersection / lv,
        0.0 if rv <= 0 else intersection / rv,
    )


def aabb_corners(lower: object, upper: object) -> np.ndarray:
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if lo.shape != (3,) or hi.shape != (3,) or not np.all(hi > lo):
        raise ValueError("AABB bounds must be positive [3]")
    center = (lo + hi) * 0.5
    return np.ascontiguousarray((center + SIGNS * ((hi - lo) * 0.5)).astype(np.float32))


def _mutual_near(left: np.ndarray, right: np.ndarray, radius_m: float) -> tuple[float, float, float]:
    if not len(left) or not len(right):
        return 0.0, 0.0, 0.0
    left_to_right = cKDTree(right).query(left, k=1)[0]
    right_to_left = cKDTree(left).query(right, k=1)[0]
    lfrac = float(np.mean(left_to_right <= radius_m))
    rfrac = float(np.mean(right_to_left <= radius_m))
    harmonic = 0.0 if lfrac + rfrac <= 0 else 2.0 * lfrac * rfrac / (lfrac + rfrac)
    return lfrac, rfrac, harmonic


def _voxel_components(points: np.ndarray, voxel_m: float) -> list[np.ndarray]:
    samples = voxel_downsample(points, voxel_m).astype(np.float64, copy=False)
    if not len(samples):
        return []
    keys = np.floor(samples / voxel_m).astype(np.int64)
    lookup = {tuple(key.tolist()): index for index, key in enumerate(keys)}
    visited = np.zeros(len(keys), dtype=bool)
    offsets = tuple(
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    )
    components: list[np.ndarray] = []
    for seed in range(len(keys)):
        if visited[seed]:
            continue
        visited[seed] = True
        stack = [seed]
        members: list[int] = []
        while stack:
            index = stack.pop()
            members.append(index)
            key = keys[index]
            for offset in offsets:
                neighbor = lookup.get((int(key[0] + offset[0]), int(key[1] + offset[1]), int(key[2] + offset[2])))
                if neighbor is not None and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        components.append(samples[np.asarray(members, dtype=np.int64)])
    return sorted(components, key=lambda row: (-len(row), tuple(row.min(axis=0))))


@dataclass(frozen=True)
class OpenBoxRefinement:
    points: np.ndarray
    context_point_count: int
    component_count: int
    accepted_component_count: int
    mask_to_context_fraction: float
    context_to_mask_fraction: float
    mutual_harmonic: float
    mask_retained_fraction: float
    used_context: bool


def openbox_refine(
    mask_points: object,
    context_points: object,
    *,
    voxel_m: float = 0.05,
    proximity_m: float = 0.075,
    min_component_voxels: int = 4,
    min_component_to_mask: float = 0.35,
    min_mask_to_component: float = 0.50,
    min_mutual_harmonic: float = 0.40,
    min_mask_retained_fraction: float = 0.50,
    max_components: int = 2,
    context_frame_ids: Sequence[int] | None = None,
    cutoff_frame_id: int | None = None,
) -> OpenBoxRefinement:
    """OpenBox-style component refinement with an explicit causal cutoff.

    Only the part of a context component lying near the mask support can enter
    the refined cloud.  Mask voxels unsupported by the accepted context are
    removed, so this operation can clean mask spill instead of only expanding
    it.  If the context cannot support at least half the mask, the input mask
    is returned unchanged.
    """

    scalar_values = (
        voxel_m,
        proximity_m,
        min_component_to_mask,
        min_mask_to_component,
        min_mutual_harmonic,
        min_mask_retained_fraction,
    )
    if not all(math.isfinite(value) for value in scalar_values):
        raise ValueError("OpenBox thresholds must be finite")
    if voxel_m <= 0 or proximity_m <= 0 or min_component_voxels < 1 or max_components < 1:
        raise ValueError("invalid OpenBox geometry limits")
    if not all(0.0 <= value <= 1.0 for value in scalar_values[2:]):
        raise ValueError("OpenBox fractions must lie in [0,1]")
    if context_frame_ids is not None:
        frames = tuple(int(value) for value in context_frame_ids)
        if tuple(sorted(set(frames))) != frames:
            raise ValueError("context_frame_ids must be sorted and unique")
        if cutoff_frame_id is None or any(value > int(cutoff_frame_id) for value in frames):
            raise ValueError("context points are not past-only at the supplied cutoff")

    mask = voxel_downsample(mask_points, voxel_m).astype(np.float64, copy=False)
    context = voxel_downsample(context_points, voxel_m).astype(np.float64, copy=False)
    if not len(mask):
        raise ValueError("mask_points must not be empty")
    candidates: list[tuple[float, int, np.ndarray, float, float]] = []
    components = _voxel_components(context, voxel_m)
    mask_lower = mask.min(axis=0) - proximity_m
    mask_upper = mask.max(axis=0) + proximity_m
    for index, component in enumerate(components):
        local = component[np.all((component >= mask_lower) & (component <= mask_upper), axis=1)]
        if len(local) < min_component_voxels:
            continue
        distance_to_mask = cKDTree(mask).query(local, k=1)[0]
        near_component = local[distance_to_mask <= proximity_m]
        if len(near_component) < min_component_voxels:
            continue
        component_to_mask = float(len(near_component) / len(local))
        mask_to_component = float(
            np.mean(cKDTree(near_component).query(mask, k=1)[0] <= proximity_m)
        )
        harmonic = (
            0.0
            if component_to_mask + mask_to_component <= 0
            else 2.0
            * component_to_mask
            * mask_to_component
            / (component_to_mask + mask_to_component)
        )
        if (
            component_to_mask >= min_component_to_mask
            and mask_to_component >= min_mask_to_component
            and harmonic >= min_mutual_harmonic
        ):
            candidates.append(
                (harmonic, index, near_component, component_to_mask, mask_to_component)
            )
    candidates.sort(key=lambda row: (-row[0], row[1]))
    accepted = candidates[:max_components]
    if accepted:
        accepted_context = voxel_downsample(
            np.concatenate([row[2] for row in accepted], axis=0), voxel_m
        ).astype(np.float64, copy=False)
        context_to_mask, mask_to_context, harmonic = _mutual_near(
            accepted_context, mask, proximity_m
        )
        supported_mask = mask[
            cKDTree(accepted_context).query(mask, k=1)[0] <= proximity_m
        ]
        mask_retained = float(len(supported_mask) / len(mask))
        if mask_retained >= min_mask_retained_fraction:
            refined = voxel_downsample(
                np.concatenate([supported_mask, accepted_context], axis=0), voxel_m
            )
            used_context = True
        else:
            refined = mask.astype(np.float32, copy=False)
            mask_retained = 1.0
            used_context = False
    else:
        accepted_context = np.empty((0, 3), dtype=np.float64)
        refined = mask.astype(np.float32, copy=False)
        context_to_mask = mask_to_context = harmonic = 0.0
        mask_retained = 1.0
        used_context = False
    return OpenBoxRefinement(
        points=np.ascontiguousarray(refined, dtype=np.float32),
        context_point_count=int(len(context)),
        component_count=len(components),
        accepted_component_count=len(accepted),
        mask_to_context_fraction=mask_to_context,
        context_to_mask_fraction=context_to_mask,
        mutual_harmonic=harmonic,
        mask_retained_fraction=mask_retained,
        used_context=used_context,
    )


@dataclass(frozen=True)
class CausalObservation:
    source_index: int
    confirmation_frame_id: int
    target_group: str
    points: np.ndarray
    corners: np.ndarray
    evidence_frame_ids: tuple[int, ...]
    view_corners: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_index, (bool, np.bool_))
            or not isinstance(self.source_index, (int, np.integer))
            or int(self.source_index) < 0
        ):
            raise ValueError("source_index must be a non-negative integer")
        if (
            isinstance(self.confirmation_frame_id, (bool, np.bool_))
            or not isinstance(self.confirmation_frame_id, (int, np.integer))
            or int(self.confirmation_frame_id) < 0
        ):
            raise ValueError("confirmation_frame_id must be a non-negative integer")
        if not isinstance(self.target_group, str) or not self.target_group.strip():
            raise ValueError("target_group must not be empty")
        group = self.target_group.strip()
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in self.evidence_frame_ids
        ):
            raise ValueError("evidence frame ids must be integers")
        frames = tuple(int(value) for value in self.evidence_frame_ids)
        if not frames or tuple(sorted(set(frames))) != frames:
            raise ValueError("evidence_frame_ids must be non-empty, sorted, and unique")
        if frames[-1] > int(self.confirmation_frame_id):
            raise ValueError("evidence cannot be newer than confirmation")
        if len(frames) != len(self.view_corners):
            raise ValueError("each evidence frame must have exactly one view box")
        points = np.ascontiguousarray(_points(self.points, "observation points"), dtype=np.float32)
        corners = np.ascontiguousarray(_box(self.corners, "observation corners"), dtype=np.float32)
        views = tuple(
            np.ascontiguousarray(_box(value, "observation view corners"), dtype=np.float32)
            for value in self.view_corners
        )
        for value in (points, corners, *views):
            value.setflags(write=False)
        object.__setattr__(self, "source_index", int(self.source_index))
        object.__setattr__(self, "confirmation_frame_id", int(self.confirmation_frame_id))
        object.__setattr__(self, "target_group", group)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "corners", corners)
        object.__setattr__(self, "evidence_frame_ids", frames)
        object.__setattr__(self, "view_corners", views)


@dataclass
class CausalMemory:
    memory_id: int
    target_group: str
    first_confirmation_frame_id: int
    last_confirmation_frame_id: int
    source_indices: list[int] = field(default_factory=list)
    evidence_frame_ids: list[int] = field(default_factory=list)
    view_corners: list[np.ndarray] = field(default_factory=list)
    source_corners: list[np.ndarray] = field(default_factory=list)
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    source_count_total: int = 0
    retired_frame_id: int | None = None
    dropped_point_count: int = 0
    dropped_view_count: int = 0
    dropped_source_count: int = 0

    def commit(
        self,
        observation: CausalObservation,
        voxel_m: float,
        *,
        max_points: int,
        max_views: int,
        max_sources: int,
    ) -> None:
        if observation.confirmation_frame_id < self.last_confirmation_frame_id:
            raise ValueError("causal memory commit moved backwards")
        self.last_confirmation_frame_id = observation.confirmation_frame_id
        if observation.source_index in self.source_indices:
            raise ValueError("duplicate source observation committed")
        self.source_count_total += 1
        self.source_indices.append(observation.source_index)
        self.source_corners.append(observation.corners)
        if len(self.source_indices) > max_sources:
            excess = len(self.source_indices) - max_sources
            self.source_indices = self.source_indices[excess:]
            self.source_corners = self.source_corners[excess:]
            self.dropped_source_count += excess

        frame_to_view = dict(zip(self.evidence_frame_ids, self.view_corners))
        for frame_id, corners in zip(observation.evidence_frame_ids, observation.view_corners):
            frame_to_view.setdefault(frame_id, corners)
        ordered_views = sorted(frame_to_view.items())
        if len(ordered_views) > max_views:
            self.dropped_view_count += len(ordered_views) - max_views
            ordered_views = ordered_views[-max_views:]
        self.evidence_frame_ids = [row[0] for row in ordered_views]
        self.view_corners = [row[1] for row in ordered_views]

        merged = voxel_downsample(
            np.concatenate([self.points, observation.points], axis=0), voxel_m
        )
        if len(merged) > max_points:
            indices = np.linspace(0, len(merged) - 1, max_points, dtype=np.int64)
            self.dropped_point_count += len(merged) - max_points
            merged = merged[indices]
        self.points = np.ascontiguousarray(merged, dtype=np.float32)

    @property
    def corners(self) -> np.ndarray:
        if not self.source_corners:
            raise ValueError("empty causal memory")
        if len(self.source_corners) == 1:
            return self.source_corners[0]
        costs = []
        for left in self.source_corners:
            costs.append(sum(1.0 - aabb_overlap(left, right)[0] for right in self.source_corners))
        return self.source_corners[min(range(len(costs)), key=lambda index: (costs[index], index))]


def _association_score(
    observation: CausalObservation,
    memory: CausalMemory,
    *,
    voxel_m: float,
    min_score: float,
) -> float | None:
    if observation.target_group != memory.target_group:
        return None
    left_lo, left_hi = aabb_bounds(observation.corners)
    right_lo, right_hi = aabb_bounds(memory.corners)
    center_distance = float(np.linalg.norm((left_lo + left_hi - right_lo - right_hi) * 0.5))
    if center_distance > 0.75:
        return None
    iou, _, _ = aabb_overlap(observation.corners, memory.corners)
    _, _, point_harmonic = _mutual_near(
        voxel_downsample(observation.points, voxel_m),
        voxel_downsample(memory.points, voxel_m),
        2.0 * voxel_m,
    )
    # Containment alone is deliberately insufficient: it is exactly the
    # wall-window/table-cup failure mode of geometry-only grouping.
    if iou < 0.15 and point_harmonic < 0.20:
        return None
    score = 0.35 * iou + 0.65 * point_harmonic - 0.05 * min(
        center_distance / 0.75, 1.0
    )
    return score if score >= min_score else None


def build_causal_memories(
    observations: Sequence[CausalObservation],
    *,
    voxel_m: float = 0.05,
    max_memories: int = 64,
    max_idle_frames: int = 1000,
    max_points_per_memory: int = 32768,
    max_views_per_memory: int = 12,
    max_sources_per_memory: int = 16,
    min_association_score: float = 0.12,
) -> tuple[list[CausalMemory], list[dict[str, object]]]:
    """Associate observations query-before-commit, batching equal timestamps."""

    if (
        voxel_m <= 0
        or max_memories < 1
        or max_idle_frames < 0
        or max_points_per_memory < 1
        or max_views_per_memory < 1
        or max_sources_per_memory < 1
        or not math.isfinite(min_association_score)
    ):
        raise ValueError("invalid causal-memory bounds")
    source_indices = [row.source_index for row in observations]
    if len(set(source_indices)) != len(source_indices):
        raise ValueError("source_index values must be globally unique")
    ordered = sorted(
        observations,
        key=lambda row: (row.confirmation_frame_id, row.source_index),
    )
    memories: list[CausalMemory] = []
    active_memory_indices: set[int] = set()
    audit: list[dict[str, object]] = []
    cursor = 0
    while cursor < len(ordered):
        frame_id = ordered[cursor].confirmation_frame_id
        end = cursor
        while end < len(ordered) and ordered[end].confirmation_frame_id == frame_id:
            end += 1
        batch = ordered[cursor:end]
        retired_for_idle = []
        for memory_index in sorted(active_memory_indices):
            memory = memories[memory_index]
            if frame_id - memory.last_confirmation_frame_id > max_idle_frames:
                memory.retired_frame_id = frame_id
                retired_for_idle.append(memory_index)
        active_memory_indices.difference_update(retired_for_idle)
        history_memory_count = len(active_memory_indices)
        proposals: list[tuple[float, int, int]] = []
        for observation_index, observation in enumerate(batch):
            for memory_index in sorted(active_memory_indices):
                memory = memories[memory_index]
                if memory.last_confirmation_frame_id >= frame_id:
                    continue
                score = _association_score(
                    observation,
                    memory,
                    voxel_m=voxel_m,
                    min_score=min_association_score,
                )
                if score is not None:
                    proposals.append((score, observation_index, memory_index))
        proposals.sort(key=lambda row: (-row[0], row[1], row[2]))
        assigned_observations: set[int] = set()
        assigned_memories: set[int] = set()
        assignments: dict[int, tuple[int, float]] = {}
        for score, observation_index, memory_index in proposals:
            if observation_index in assigned_observations or memory_index in assigned_memories:
                continue
            assigned_observations.add(observation_index)
            assigned_memories.add(memory_index)
            assignments[observation_index] = (memory_index, score)
        for observation_index, observation in enumerate(batch):
            evicted_memory_id = None
            if observation_index in assignments:
                memory_index, score = assignments[observation_index]
                memory = memories[memory_index]
                prior_last = memory.last_confirmation_frame_id
                memory.commit(
                    observation,
                    voxel_m,
                    max_points=max_points_per_memory,
                    max_views=max_views_per_memory,
                    max_sources=max_sources_per_memory,
                )
                decision = "associate"
            else:
                if len(active_memory_indices) >= max_memories:
                    evicted_index = min(
                        active_memory_indices,
                        key=lambda index: (
                            memories[index].last_confirmation_frame_id,
                            memories[index].first_confirmation_frame_id,
                            memories[index].memory_id,
                        ),
                    )
                    active_memory_indices.remove(evicted_index)
                    memories[evicted_index].retired_frame_id = frame_id
                    evicted_memory_id = memories[evicted_index].memory_id
                memory = CausalMemory(
                    memory_id=len(memories),
                    target_group=observation.target_group,
                    first_confirmation_frame_id=frame_id,
                    last_confirmation_frame_id=frame_id,
                )
                memory.commit(
                    observation,
                    voxel_m,
                    max_points=max_points_per_memory,
                    max_views=max_views_per_memory,
                    max_sources=max_sources_per_memory,
                )
                memories.append(memory)
                active_memory_indices.add(len(memories) - 1)
                score = None
                prior_last = None
                decision = "new"
            audit.append(
                {
                    "source_index": observation.source_index,
                    "confirmation_frame_id": frame_id,
                    "history_memory_count": history_memory_count,
                    "active_memory_count_after_commit": len(active_memory_indices),
                    "evicted_memory_id": evicted_memory_id,
                    "decision": decision,
                    "memory_id": memory.memory_id,
                    "association_score": score,
                    "memory_last_frame_before_commit": prior_last,
                    "query_before_commit": prior_last is None or prior_last < frame_id,
                }
            )
        cursor = end
    return memories, audit


def _yaw_from_corners(corners: np.ndarray) -> float:
    box = _box(corners)
    vector = box[4] - box[0]
    if np.linalg.norm(vector[:2]) <= 1e-8:
        return 0.0
    return math.atan2(float(vector[1]), float(vector[0]))


def _pca_yaw(points: np.ndarray) -> float | None:
    xy = points[:, :2] - points[:, :2].mean(axis=0, keepdims=True)
    covariance = xy.T @ xy / max(len(xy), 1)
    values, vectors = np.linalg.eigh(covariance)
    if values[-1] <= 1e-10 or values[-1] / max(values[-2], 1e-10) < 1.15:
        return None
    vector = vectors[:, int(np.argmax(values))]
    return math.atan2(float(vector[1]), float(vector[0]))


def fit_quantile_obb(points: object, yaw_rad: float, *, quantile: float = 0.02) -> np.ndarray:
    values = _points(points, "fit points")
    if not math.isfinite(yaw_rad):
        raise ValueError("yaw_rad must be finite")
    if not math.isfinite(quantile) or not 0.0 <= quantile < 0.5:
        raise ValueError("quantile must lie in [0,0.5)")
    cosine, sine = math.cos(yaw_rad), math.sin(yaw_rad)
    rotation = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    local = values @ rotation
    lower, upper = np.quantile(local, [quantile, 1.0 - quantile], axis=0)
    extent = np.maximum(upper - lower, 0.05)
    center = ((lower + upper) * 0.5) @ rotation.T
    corners = center + (SIGNS * (extent * 0.5)) @ rotation.T
    return np.ascontiguousarray(corners.astype(np.float32))


def _oriented_containment(points: np.ndarray, corners: np.ndarray) -> float:
    center, unit, lengths, _ = _obb_geometry(corners)
    local = (points - center) @ unit.T
    return float(np.mean(np.all(np.abs(local) <= lengths * 0.5 + 1e-6, axis=1)))


def _obb_geometry(corners: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    box = _box(corners)
    center = box.mean(axis=0)
    axes = np.stack([box[4] - box[0], box[2] - box[0], box[1] - box[0]])
    lengths = np.linalg.norm(axes, axis=1)
    if np.any(lengths <= 1e-8):
        raise ValueError("degenerate OBB")
    unit = axes / lengths[:, None]
    if np.max(np.abs(unit @ unit.T - np.eye(3))) > 1e-3:
        raise ValueError("corners do not encode an orthogonal box")
    return center, unit, lengths, float(np.prod(lengths))


def _sample_box_interior(corners: np.ndarray, steps: int = 5) -> np.ndarray:
    center, unit, lengths, _ = _obb_geometry(corners)
    coordinates = np.linspace(-0.4, 0.4, steps, dtype=np.float64)
    grid = np.stack(np.meshgrid(coordinates, coordinates, coordinates, indexing="ij"), axis=-1)
    local = grid.reshape(-1, 3) * lengths
    return center + local @ unit


def _approx_oriented_iou(left: np.ndarray, right: np.ndarray) -> float:
    """Deterministic symmetric interior-grid approximation to oriented IoU."""

    _, _, _, left_volume = _obb_geometry(left)
    _, _, _, right_volume = _obb_geometry(right)
    left_fraction = _oriented_containment(_sample_box_interior(left), right)
    right_fraction = _oriented_containment(_sample_box_interior(right), left)
    intersection = 0.5 * (
        left_fraction * left_volume + right_fraction * right_volume
    )
    intersection = min(intersection, left_volume, right_volume)
    union = left_volume + right_volume - intersection
    return 0.0 if union <= 0 else float(intersection / union)


@dataclass(frozen=True)
class HypothesisSelection:
    name: str
    corners: np.ndarray
    diagnostics: tuple[dict[str, float | str | bool], ...]


def select_multi_hypothesis_obb(
    mask_points: object,
    refined_points: object,
    original_corners: object,
    view_corners: Sequence[np.ndarray],
) -> HypothesisSelection:
    """Choose original/raw-yaw/PCA/axis hypotheses without learned weights."""

    mask = _points(mask_points, "mask points")
    refined = _points(refined_points, "refined points")
    original = _box(original_corners, "original corners")
    views = tuple(_box(value, "view corners") for value in view_corners)
    candidate_rows: list[tuple[str, np.ndarray]] = [
        ("r15_original", original),
        ("raw_yaw_refined", fit_quantile_obb(refined, _yaw_from_corners(original))),
    ]
    pca_yaw = _pca_yaw(refined)
    if pca_yaw is not None:
        candidate_rows.append(("pca_yaw_refined", fit_quantile_obb(refined, pca_yaw)))
    candidate_rows.append(("axis_aligned_refined", fit_quantile_obb(refined, 0.0)))
    candidates = tuple(candidate_rows)
    original_center, _, _, original_volume = _obb_geometry(original)
    diagnostics: list[dict[str, float | str | bool]] = []
    ranking: list[tuple[float, float, float, float, int]] = []
    for index, (name, corners) in enumerate(candidates):
        center, _, _, volume = _obb_geometry(corners)
        mask_containment = _oriented_containment(mask, corners)
        refined_containment = _oriented_containment(refined, corners)
        view_ious = [_approx_oriented_iou(corners, view) for view in views]
        median_view_iou = float(np.median(view_ious)) if view_ious else 0.0
        center_shift = float(np.linalg.norm(center - original_center))
        volume_ratio = volume / max(original_volume, 1e-9)
        admissible = name == "r15_original" or (
            len(views) >= 2
            and mask_containment >= 0.90
            and refined_containment >= 0.90
            and center_shift <= 0.50
            and 0.35 <= volume_ratio <= 2.00
        )
        # Cross-view agreement is primary.  Containment and compactness only
        # break geometric ties; no value is fitted from target annotations.
        evidence = (
            median_view_iou
            + 0.20 * mask_containment
            + 0.10 * refined_containment
            - 0.05 * abs(math.log(max(volume_ratio, 1e-9)))
            - 0.05 * min(center_shift / 0.50, 1.0)
        )
        diagnostics.append(
            {
                "name": name,
                "admissible": admissible,
                "evidence_score": evidence,
                "median_view_oriented_iou_approx": median_view_iou,
                "mask_point_containment": mask_containment,
                "refined_point_containment": refined_containment,
                "obb_volume_m3": volume,
                "obb_volume_ratio_to_r15": volume_ratio,
                "center_shift_from_r15_m": center_shift,
            }
        )
        ranking.append((1.0 if admissible else 0.0, evidence, mask_containment, -volume, -index))
    admissible_indices = [
        index for index, row in enumerate(diagnostics) if bool(row["admissible"])
    ]
    chosen_index = (
        max(admissible_indices, key=lambda index: ranking[index])
        if admissible_indices
        else 0
    )
    name, corners = candidates[chosen_index]
    return HypothesisSelection(
        name=name,
        corners=np.ascontiguousarray(corners, dtype=np.float32),
        diagnostics=tuple(diagnostics),
    )


__all__ = [
    "CausalMemory",
    "CausalObservation",
    "HypothesisSelection",
    "OpenBoxRefinement",
    "aabb_bounds",
    "aabb_corners",
    "aabb_overlap",
    "build_causal_memories",
    "fit_quantile_obb",
    "openbox_refine",
    "select_multi_hypothesis_obb",
    "voxel_downsample",
]
