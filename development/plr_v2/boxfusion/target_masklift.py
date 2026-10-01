"""Training-free geometry for target-first three-view mask lifting.

The functions in this module deliberately have no detector, semantic model,
ground-truth, or training dependency.  They turn three already-cleaned sets of
world points into a deterministic 5 cm voxel consensus, and fit a robust
yaw-only OBB using the circular medoid of the three raw Boxer orientations.

Voxel coordinates use signed floor quantization.  In particular, world point
``(-eps, 0, 0)`` belongs to voxel ``(-1, 0, 0)``.  An observed voxel is retained
when at least two *different* views contain an observed voxel within Chebyshev
distance one.  Multiple points in one view can therefore never manufacture
cross-view support.

``PastOnlyTargetTracker`` is an optional, small causal tracker for raw target
observations.  Every call is one valid keyframe transaction: current-frame
observations are matched only to state committed by earlier calls, and tracks
confirm on their first three distinct frames.  Association thresholds and the
ten-keyframe TTL are fixed rather than learned.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


SCHEMA = "boxfusion.target_masklift.v1"
VOXEL_SIZE_METERS = 0.05
MIN_SUPPORT_VIEWS = 2
CHEBYSHEV_RADIUS = 1

MATCH_AABB_IOU = 0.10
MATCH_CENTER_DISTANCE_METERS = 0.50
CONFIRM_DISTINCT_FRAMES = 3
TTL_VALID_KEYFRAMES = 10
MAX_LIVE_TRACKS = 1024

_COORDINATE_LIMIT = 1 << 52
_MAX_ID = (1 << 63) - 1
_OFFSETS = np.asarray(
    [
        (x, y, z)
        for x in range(-CHEBYSHEV_RADIUS, CHEBYSHEV_RADIUS + 1)
        for y in range(-CHEBYSHEV_RADIUS, CHEBYSHEV_RADIUS + 1)
        for z in range(-CHEBYSHEV_RADIUS, CHEBYSHEV_RADIUS + 1)
    ],
    dtype=np.int64,
)
_CORNER_SIGNS = np.asarray(
    [
        (-1.0, -1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, -1.0),
        (1.0, 1.0, 1.0),
    ],
    dtype=np.float64,
)
_ROW_DTYPE = np.dtype([("x", "<i8"), ("y", "<i8"), ("z", "<i8")])


def _readonly(value: object, dtype: np.dtype, shape: Optional[tuple[int, ...]] = None) -> np.ndarray:
    """Return an owned bytes-backed array that callers cannot make writeable."""

    array = np.asarray(value, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError(f"array must have shape {shape}")
    packed = np.array(array, dtype=dtype, order="C", copy=True).tobytes()
    return np.frombuffer(packed, dtype=dtype).reshape(array.shape)


def _strict_id(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < 0 or result > _MAX_ID:
        raise ValueError(f"{name} must be in [0, {_MAX_ID}]")
    return result


def _points(value: object, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != 2 or value.shape[1:] != (3,):
        raise ValueError(f"{name} must be a numeric numpy array with shape [N,3]")
    if value.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a numeric numpy array with shape [N,3]")
    try:
        points = np.array(value, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a numeric numpy array with shape [N,3]") from error
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    scaled = points / VOXEL_SIZE_METERS
    if scaled.size and np.any(np.abs(scaled) > _COORDINATE_LIMIT):
        raise ValueError(f"{name} coordinate magnitude is unsafe for voxel quantization")
    return points


def signed_floor_voxels(points_world: object) -> np.ndarray:
    """Quantize finite ``[N,3]`` world points into sorted unique 5 cm keys."""

    points = _points(points_world, "points_world")
    if len(points) == 0:
        return _readonly(np.empty((0, 3), dtype=np.int64), np.int64)
    keys = np.floor(points / VOXEL_SIZE_METERS).astype(np.int64)
    keys = np.unique(keys, axis=0)
    return _readonly(keys, np.int64)


def _row_records(rows: np.ndarray) -> np.ndarray:
    records = np.empty(len(rows), dtype=_ROW_DTYPE)
    if len(rows):
        records["x"] = rows[:, 0]
        records["y"] = rows[:, 1]
        records["z"] = rows[:, 2]
    return records


def _rows_present(queries: np.ndarray, sorted_rows: np.ndarray) -> np.ndarray:
    """Vectorized exact row membership for lexicographically sorted int64 rows."""

    if len(queries) == 0 or len(sorted_rows) == 0:
        return np.zeros(len(queries), dtype=bool)
    haystack = _row_records(sorted_rows)
    needles = _row_records(queries)
    positions = np.searchsorted(haystack, needles)
    present = positions < len(haystack)
    if np.any(present):
        valid = np.flatnonzero(present)
        present[valid] = haystack[positions[valid]] == needles[valid]
    return present


def _neighborhood_present(candidates: np.ndarray, view_voxels: np.ndarray) -> np.ndarray:
    present = np.zeros(len(candidates), dtype=bool)
    for offset in _OFFSETS:
        if present.all():
            break
        pending = np.flatnonzero(~present)
        present[pending] = _rows_present(candidates[pending] + offset, view_voxels)
    return present


@dataclass(frozen=True)
class VoxelConsensus:
    """Deterministic three-view consensus and contribution diagnostics."""

    voxel_keys: np.ndarray
    voxel_centers: np.ndarray
    support_view_count: np.ndarray
    support_view_matrix: np.ndarray
    exact_view_matrix: np.ndarray
    supported_points: np.ndarray
    supported_point_view_ids: np.ndarray
    input_point_counts: np.ndarray
    input_voxel_counts: np.ndarray
    supported_point_counts: np.ndarray
    exact_supported_voxel_counts: np.ndarray
    neighborhood_supported_voxel_counts: np.ndarray

    def __post_init__(self) -> None:
        count = len(np.asarray(self.voxel_keys))
        specifications = (
            ("voxel_keys", np.int64, (count, 3)),
            ("voxel_centers", np.float64, (count, 3)),
            ("support_view_count", np.int64, (count,)),
            ("support_view_matrix", np.bool_, (count, 3)),
            ("exact_view_matrix", np.bool_, (count, 3)),
            ("supported_points", np.float64, (len(np.asarray(self.supported_points)), 3)),
            (
                "supported_point_view_ids",
                np.int64,
                (len(np.asarray(self.supported_points)),),
            ),
            ("input_point_counts", np.int64, (3,)),
            ("input_voxel_counts", np.int64, (3,)),
            ("supported_point_counts", np.int64, (3,)),
            ("exact_supported_voxel_counts", np.int64, (3,)),
            ("neighborhood_supported_voxel_counts", np.int64, (3,)),
        )
        for name, dtype, shape in specifications:
            object.__setattr__(self, name, _readonly(getattr(self, name), dtype, shape))
        if count and (
            np.any(self.support_view_count < MIN_SUPPORT_VIEWS)
            or not np.array_equal(self.support_view_count, self.support_view_matrix.sum(axis=1))
        ):
            raise ValueError("consensus support counts are inconsistent")
        if len(self.supported_point_view_ids) and (
            np.min(self.supported_point_view_ids) < 0
            or np.max(self.supported_point_view_ids) >= 3
        ):
            raise ValueError("supported point view ids must be in [0,2]")

    @property
    def voxel_count(self) -> int:
        return len(self.voxel_keys)

    @property
    def point_count(self) -> int:
        return len(self.supported_points)


def fuse_three_view_points(view_points_world: Sequence[np.ndarray]) -> VoxelConsensus:
    """Fuse exactly three cleaned point sets using cross-view voxel support.

    Output voxel rows are lexicographically ordered.  Output points are sorted
    by voxel key, view id, and coordinate, so shuffling points within an input
    view cannot alter the result.
    """

    if (
        isinstance(view_points_world, (str, bytes))
        or not isinstance(view_points_world, Sequence)
        or len(view_points_world) != 3
    ):
        raise ValueError("view_points_world must contain exactly three views")
    points = tuple(
        _points(value, f"view_points_world[{index}]")
        for index, value in enumerate(view_points_world)
    )
    point_keys = tuple(
        np.floor(value / VOXEL_SIZE_METERS).astype(np.int64)
        for value in points
    )
    voxels = tuple(
        np.unique(value, axis=0) if len(value) else np.empty((0, 3), dtype=np.int64)
        for value in point_keys
    )
    if not any(len(value) for value in voxels):
        union = np.empty((0, 3), dtype=np.int64)
    else:
        union = np.unique(
            np.concatenate([value for value in voxels if len(value)], axis=0), axis=0
        )

    support = np.stack(
        [_neighborhood_present(union, value) for value in voxels], axis=1
    )
    keep = support.sum(axis=1) >= MIN_SUPPORT_VIEWS
    kept_keys = union[keep]
    kept_support = support[keep]
    exact = np.stack([_rows_present(kept_keys, value) for value in voxels], axis=1)

    retained_points = []
    retained_views = []
    retained_keys = []
    point_counts = np.zeros(3, dtype=np.int64)
    for view_id, (view, keys) in enumerate(zip(points, point_keys)):
        mask = _rows_present(keys, kept_keys)
        point_counts[view_id] = int(mask.sum())
        if np.any(mask):
            retained_points.append(view[mask])
            retained_views.append(np.full(int(mask.sum()), view_id, dtype=np.int64))
            retained_keys.append(keys[mask])
    if retained_points:
        output_points = np.concatenate(retained_points, axis=0)
        output_views = np.concatenate(retained_views, axis=0)
        output_keys = np.concatenate(retained_keys, axis=0)
        order = np.lexsort(
            (
                output_points[:, 2],
                output_points[:, 1],
                output_points[:, 0],
                output_views,
                output_keys[:, 2],
                output_keys[:, 1],
                output_keys[:, 0],
            )
        )
        output_points = output_points[order]
        output_views = output_views[order]
    else:
        output_points = np.empty((0, 3), dtype=np.float64)
        output_views = np.empty((0,), dtype=np.int64)

    return VoxelConsensus(
        voxel_keys=kept_keys,
        voxel_centers=(kept_keys.astype(np.float64) + 0.5) * VOXEL_SIZE_METERS,
        support_view_count=kept_support.sum(axis=1, dtype=np.int64),
        support_view_matrix=kept_support,
        exact_view_matrix=exact,
        supported_points=output_points,
        supported_point_view_ids=output_views,
        input_point_counts=np.asarray([len(value) for value in points], dtype=np.int64),
        input_voxel_counts=np.asarray([len(value) for value in voxels], dtype=np.int64),
        supported_point_counts=point_counts,
        exact_supported_voxel_counts=exact.sum(axis=0, dtype=np.int64),
        neighborhood_supported_voxel_counts=kept_support.sum(axis=0, dtype=np.int64),
    )


def _wrap_pi(angle: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def quaternion_yaws_wxyz(quaternions_wxyz: object) -> np.ndarray:
    """Return normalized ZYX yaw for finite Hamilton ``[w,x,y,z]`` rows."""

    try:
        quaternions = np.asarray(quaternions_wxyz, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("quaternions_wxyz must be a finite [N,4] array") from error
    if (
        quaternions.ndim != 2
        or quaternions.shape[1:] != (4,)
        or not np.isfinite(quaternions).all()
    ):
        raise ValueError("quaternions_wxyz must be a finite [N,4] array")
    squared_norm = np.einsum("ij,ij->i", quaternions, quaternions)
    if np.any(squared_norm <= 1e-12):
        raise ValueError("quaternions_wxyz contains a degenerate quaternion")
    q = quaternions / np.sqrt(squared_norm)[:, None]
    w, x, y, z = q.T
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return _readonly(_wrap_pi(yaw), np.float64)


def circular_medoid_yaw(yaws_rad: object) -> float:
    """Choose a sample yaw minimizing total circular L1 distance.

    Ties are resolved by the wrapped numeric yaw and then sample index.  This
    makes the selected angle independent of input order except for duplicate
    samples, which are geometrically identical.
    """

    try:
        yaws = np.asarray(yaws_rad, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("yaws_rad must be a nonempty finite vector") from error
    if yaws.ndim != 1 or len(yaws) == 0 or not np.isfinite(yaws).all():
        raise ValueError("yaws_rad must be a nonempty finite vector")
    wrapped = np.asarray(_wrap_pi(yaws), dtype=np.float64)
    distances = np.abs(_wrap_pi(wrapped[:, None] - wrapped[None, :]))
    costs = distances.sum(axis=1)
    best = min(
        range(len(wrapped)),
        key=lambda index: (float(costs[index]), float(wrapped[index]), index),
    )
    return float(wrapped[best])


def yaw_medoid_wxyz(quaternions_wxyz: object) -> float:
    """Circular-medoid yaw from exactly three raw Boxer wxyz quaternions."""

    quaternions = np.asarray(quaternions_wxyz)
    if quaternions.shape != (3, 4):
        raise ValueError("quaternions_wxyz must have shape [3,4]")
    return circular_medoid_yaw(quaternion_yaws_wxyz(quaternions))


@dataclass(frozen=True)
class RobustYawOBB:
    center: np.ndarray
    extent: np.ndarray
    yaw_rad: float
    local_lower_q02: np.ndarray
    local_upper_q98: np.ndarray
    corners: np.ndarray
    aabb_lower: np.ndarray
    aabb_upper: np.ndarray
    point_count: int

    def __post_init__(self) -> None:
        for name, shape in (
            ("center", (3,)),
            ("extent", (3,)),
            ("local_lower_q02", (3,)),
            ("local_upper_q98", (3,)),
            ("corners", (8, 3)),
            ("aabb_lower", (3,)),
            ("aabb_upper", (3,)),
        ):
            array = np.asarray(getattr(self, name), dtype=np.float64)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
            object.__setattr__(self, name, _readonly(array, np.float64, shape))
        yaw = float(self.yaw_rad)
        if not isfinite(yaw):
            raise ValueError("yaw_rad must be finite")
        object.__setattr__(self, "yaw_rad", float(_wrap_pi(yaw)))
        object.__setattr__(self, "point_count", _strict_id("point_count", self.point_count))
        if self.point_count < 1:
            raise ValueError("point_count must be positive")
        if np.any(self.extent < 0.0):
            raise ValueError("extent must be nonnegative")


def robust_yaw_obb(
    points_world: object,
    *,
    yaw_rad: Optional[float] = None,
    quaternions_wxyz: Optional[object] = None,
) -> RobustYawOBB:
    """Fit the fixed local-frame q02/q98 yaw-only robust OBB.

    Exactly one orientation source is required.  Passing raw quaternions uses
    their circular-medoid yaw; a caller may instead pass an already frozen yaw.
    """

    points = _points(points_world, "points_world")
    if len(points) == 0:
        raise ValueError("points_world must contain at least one point")
    if (yaw_rad is None) == (quaternions_wxyz is None):
        raise ValueError("provide exactly one of yaw_rad or quaternions_wxyz")
    if quaternions_wxyz is not None:
        yaw = yaw_medoid_wxyz(quaternions_wxyz)
    else:
        try:
            yaw = float(yaw_rad)  # type: ignore[arg-type]
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("yaw_rad must be finite") from error
        if not isfinite(yaw):
            raise ValueError("yaw_rad must be finite")
        yaw = float(_wrap_pi(yaw))

    cosine, sine = np.cos(yaw), np.sin(yaw)
    local_to_world = np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    # Row-vector convention: world = local @ R.T, local = world @ R.
    local = points @ local_to_world
    quantiles = np.quantile(local, (0.02, 0.98), axis=0)
    lower, upper = quantiles[0], quantiles[1]
    local_center = 0.5 * (lower + upper)
    extent = np.maximum(upper - lower, 0.0)
    center = local_center @ local_to_world.T
    corners = (local_center + _CORNER_SIGNS * (extent / 2.0)) @ local_to_world.T
    return RobustYawOBB(
        center=center,
        extent=extent,
        yaw_rad=yaw,
        local_lower_q02=lower,
        local_upper_q98=upper,
        corners=corners,
        aabb_lower=corners.min(axis=0),
        aabb_upper=corners.max(axis=0),
        point_count=len(points),
    )


@dataclass(frozen=True)
class AABBOverlap:
    intersection_volume: float
    left_volume: float
    right_volume: float
    union_volume: float
    iou: float
    left_containment: float
    right_containment: float


def _bounds_pair(lower: object, upper: object, label: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        lo = np.asarray(lower, dtype=np.float64)
        hi = np.asarray(upper, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} bounds must be finite shape-[3] arrays") from error
    if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError(f"{label} bounds must be finite shape-[3] arrays")
    if np.any(hi < lo):
        raise ValueError(f"{label} upper bounds must not be below lower bounds")
    return lo, hi


def aabb_overlap(
    left_lower: object,
    left_upper: object,
    right_lower: object,
    right_upper: object,
) -> AABBOverlap:
    """Return IoU and containment in both directions for two world AABBs."""

    left_lo, left_hi = _bounds_pair(left_lower, left_upper, "left")
    right_lo, right_hi = _bounds_pair(right_lower, right_upper, "right")
    intersection_extent = np.maximum(
        np.minimum(left_hi, right_hi) - np.maximum(left_lo, right_lo), 0.0
    )
    intersection = float(np.prod(intersection_extent))
    left_volume = float(np.prod(left_hi - left_lo))
    right_volume = float(np.prod(right_hi - right_lo))
    union = left_volume + right_volume - intersection
    return AABBOverlap(
        intersection_volume=intersection,
        left_volume=left_volume,
        right_volume=right_volume,
        union_volume=union,
        iou=0.0 if union <= 0.0 else intersection / union,
        left_containment=0.0 if left_volume <= 0.0 else intersection / left_volume,
        right_containment=0.0 if right_volume <= 0.0 else intersection / right_volume,
    )


@dataclass(frozen=True)
class TargetObservation:
    """One immutable target-vocabulary AABB from one valid keyframe."""

    observation_id: int
    frame_id: int
    target_group: str
    aabb_lower: np.ndarray
    aabb_upper: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation_id", _strict_id("observation_id", self.observation_id))
        object.__setattr__(self, "frame_id", _strict_id("frame_id", self.frame_id))
        if not isinstance(self.target_group, str) or not self.target_group.strip():
            raise ValueError("target_group must be a nonempty string")
        object.__setattr__(self, "target_group", self.target_group.strip())
        lower, upper = _bounds_pair(self.aabb_lower, self.aabb_upper, "observation")
        if np.any(upper <= lower):
            raise ValueError("observation AABB must have positive extent")
        object.__setattr__(self, "aabb_lower", _readonly(lower, np.float64, (3,)))
        object.__setattr__(self, "aabb_upper", _readonly(upper, np.float64, (3,)))

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.aabb_lower + self.aabb_upper)


@dataclass(frozen=True)
class TargetAssignment:
    observation_id: int
    track_id: int
    action: str
    aabb_iou: float
    center_distance_m: float


@dataclass(frozen=True)
class TargetTrackSnapshot:
    track_id: int
    target_group: str
    first_frame_id: int
    last_frame_id: int
    evidence_frame_ids: Tuple[int, ...]
    evidence_observation_ids: Tuple[int, ...]
    aabb_lower: np.ndarray
    aabb_upper: np.ndarray
    confirmed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "aabb_lower", _readonly(self.aabb_lower, np.float64, (3,)))
        object.__setattr__(self, "aabb_upper", _readonly(self.aabb_upper, np.float64, (3,)))


@dataclass(frozen=True)
class TargetTrackerUpdate:
    frame_id: int
    assignments: Tuple[TargetAssignment, ...]
    newly_confirmed_tracks: Tuple[TargetTrackSnapshot, ...]
    retired_track_ids: Tuple[int, ...]
    active_tracks: Tuple[TargetTrackSnapshot, ...]


@dataclass(frozen=True)
class _TargetTrack:
    track_id: int
    target_group: str
    first_frame_id: int
    last_frame_id: int
    last_keyframe_step: int
    anchor: TargetObservation
    evidence: Tuple[TargetObservation, ...]
    confirmed: bool


def _snapshot(track: _TargetTrack) -> TargetTrackSnapshot:
    return TargetTrackSnapshot(
        track_id=track.track_id,
        target_group=track.target_group,
        first_frame_id=track.first_frame_id,
        last_frame_id=track.last_frame_id,
        evidence_frame_ids=tuple(row.frame_id for row in track.evidence),
        evidence_observation_ids=tuple(row.observation_id for row in track.evidence),
        aabb_lower=track.anchor.aabb_lower,
        aabb_upper=track.anchor.aabb_upper,
        confirmed=track.confirmed,
    )


class PastOnlyTargetTracker:
    """Fixed-threshold greedy tracker with first-three confirmation.

    ``update`` frame ids must be strictly increasing, including calls with no
    observations.  Such empty calls are valid keyframes and advance the TTL.
    Current observations are associated against a frozen copy of prior tracks;
    tracks created in the same call cannot match one another.
    """

    def __init__(self) -> None:
        self._tracks: Dict[int, _TargetTrack] = {}
        self._confirmed: Dict[int, TargetTrackSnapshot] = {}
        self._next_track_id = 0
        self._last_frame_id: Optional[int] = None
        self._keyframe_step = 0
        self._seen_observation_ids: set[int] = set()

    @property
    def last_frame_id(self) -> Optional[int]:
        return self._last_frame_id

    @property
    def confirmed_tracks(self) -> Tuple[TargetTrackSnapshot, ...]:
        return tuple(self._confirmed[key] for key in sorted(self._confirmed))

    def update(
        self, frame_id: object, observations: Sequence[TargetObservation]
    ) -> TargetTrackerUpdate:
        frame = _strict_id("frame_id", frame_id)
        if self._last_frame_id is not None and frame <= self._last_frame_id:
            raise ValueError("frame_id must be strictly increasing")
        if isinstance(observations, (str, bytes)) or not isinstance(observations, Sequence):
            raise ValueError("observations must be a sequence")
        rows = tuple(observations)
        if any(not isinstance(row, TargetObservation) for row in rows):
            raise ValueError("every observation must be a TargetObservation")
        if any(row.frame_id != frame for row in rows):
            raise ValueError("every observation frame_id must equal update frame_id")
        ids = [row.observation_id for row in rows]
        if len(ids) != len(set(ids)) or any(value in self._seen_observation_ids for value in ids):
            raise ValueError("observation_id must be globally unique")
        ordered = tuple(sorted(rows, key=lambda row: (row.target_group, row.observation_id)))

        step = self._keyframe_step + 1
        retired = tuple(
            sorted(
                track_id
                for track_id, track in self._tracks.items()
                if step - track.last_keyframe_step > TTL_VALID_KEYFRAMES
            )
        )
        prior: Dict[int, _TargetTrack] = {
            track_id: track
            for track_id, track in self._tracks.items()
            if track_id not in set(retired)
        }

        edges = []
        for observation in ordered:
            for track_id in sorted(prior):
                track = prior[track_id]
                if track.target_group != observation.target_group:
                    continue
                overlap = aabb_overlap(
                    observation.aabb_lower,
                    observation.aabb_upper,
                    track.anchor.aabb_lower,
                    track.anchor.aabb_upper,
                )
                distance = float(np.linalg.norm(observation.center - track.anchor.center))
                if overlap.iou >= MATCH_AABB_IOU and distance <= MATCH_CENTER_DISTANCE_METERS:
                    edges.append(
                        (-overlap.iou, distance, track_id, observation.observation_id, observation)
                    )
        edges.sort(key=lambda value: (value[0], value[1], value[2], value[3]))
        used_tracks: set[int] = set()
        used_observations: set[int] = set()
        matched: Dict[int, tuple[int, float, float]] = {}
        for negative_iou, distance, track_id, observation_id, _observation in edges:
            if track_id in used_tracks or observation_id in used_observations:
                continue
            used_tracks.add(track_id)
            used_observations.add(observation_id)
            matched[observation_id] = (track_id, -negative_iou, distance)

        next_tracks = dict(prior)
        assignments = []
        newly_confirmed = []
        for observation in ordered:
            match = matched.get(observation.observation_id)
            if match is None:
                if len(next_tracks) >= MAX_LIVE_TRACKS:
                    raise ValueError("live track capacity exceeded")
                track_id = self._next_track_id
                self._next_track_id += 1
                track = _TargetTrack(
                    track_id=track_id,
                    target_group=observation.target_group,
                    first_frame_id=frame,
                    last_frame_id=frame,
                    last_keyframe_step=step,
                    anchor=observation,
                    evidence=(observation,),
                    confirmed=False,
                )
                next_tracks[track_id] = track
                assignments.append(TargetAssignment(observation.observation_id, track_id, "created", 0.0, 0.0))
                continue

            track_id, overlap_iou, distance = match
            old = next_tracks[track_id]
            evidence = old.evidence
            if len(evidence) < CONFIRM_DISTINCT_FRAMES:
                evidence = evidence + (observation,)
            confirmed = old.confirmed or len(evidence) >= CONFIRM_DISTINCT_FRAMES
            track = _TargetTrack(
                track_id=track_id,
                target_group=old.target_group,
                first_frame_id=old.first_frame_id,
                last_frame_id=frame,
                last_keyframe_step=step,
                anchor=observation,
                evidence=evidence,
                confirmed=confirmed,
            )
            next_tracks[track_id] = track
            assignments.append(
                TargetAssignment(
                    observation.observation_id,
                    track_id,
                    "matched",
                    overlap_iou,
                    distance,
                )
            )
            if confirmed and not old.confirmed:
                snap = _snapshot(track)
                self._confirmed[track_id] = snap
                newly_confirmed.append(snap)

        self._tracks = next_tracks
        self._last_frame_id = frame
        self._keyframe_step = step
        self._seen_observation_ids.update(ids)
        assignments.sort(key=lambda row: row.observation_id)
        active = tuple(_snapshot(next_tracks[key]) for key in sorted(next_tracks))
        return TargetTrackerUpdate(
            frame_id=frame,
            assignments=tuple(assignments),
            newly_confirmed_tracks=tuple(sorted(newly_confirmed, key=lambda row: row.track_id)),
            retired_track_ids=retired,
            active_tracks=active,
        )

