"""Portable, training-free fragment shadow for box-crop depth observations.

The module is deliberately limited to causal depth-fragment extraction and a
bounded per-track memory.  It never changes native predictions and has no
semantic, model, image, or annotation dependency.  Proposal failures produce
diagnostics and an abstention instead of interrupting the native pipeline.

Coordinate contract
-------------------
``box_xyxy`` uses continuous, zero-based ``x=column, y=row`` coordinates in a
source image already registered to the depth image.  The caller must provide
the registration as a homogeneous 3x3 ``proposal_to_depth_affine`` acting on
column vectors ``[x, y, 1]``.  Only positive-scale, axis-aligned affine maps
are accepted; this helper does not calibrate independent RGB/depth cameras.
After mapping, the box is clipped to depth pixel-center coordinates
``[0, W-1] x [0, H-1]`` and integer centers from ``ceil(min)`` through
``floor(max)`` are sampled inclusively.  ``aligned_resize_affine`` constructs
the declared resize-only convention, including non-equal x/y scales.

Cleaned world points are quantized exactly once to signed ``int64`` voxel
keys with ``floor(world_coordinate / 0.05)``.  Voxel ``k`` therefore denotes
the half-open interval ``[0.05*k, 0.05*(k+1))`` on positive and negative
axes alike.  Matcher-facing code consumes these keys directly and never
re-quantizes the retained floating-point centroids.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, replace
from enum import Enum
import json
from numbers import Integral, Real
import os
import tempfile
import time
from types import MappingProxyType
from typing import Mapping, Optional, Sequence

import numpy as np


SCHEMA = "boxfusion.smov_fragment_shadow.v1"
VOXEL_SIZE_METERS = 0.05

MAX_INPUT_DEPTH_PIXELS = 4_194_304
MAX_INPUT_PROPOSALS = 4_096

# Private executable backstops.  The public names above are useful audit
# aliases, but rebinding one of them must not silently change an experiment.
_F_VOXEL_SIZE_METERS = 0.05
_F_MAX_INPUT_DEPTH_PIXELS = 4_194_304
_F_MAX_INPUT_PROPOSALS = 4_096
_F_MAX_DIAGNOSTIC_BYTES = 32 * 1024 * 1024
_F_MAX_DIAGNOSTIC_FRAMES = 4_096


class AbortReason(str, Enum):
    """Fixed transaction-abort vocabulary; values are stable diagnostic keys."""

    WRAPPER_FAILURE = "wrapper_failure"
    ASSOCIATION_UNAVAILABLE = "association_unavailable"
    SHUTDOWN = "shutdown"

_DEFAULT_CONFIG: Mapping[str, object] = MappingProxyType({
    "pixel_stride": 4,
    "max_rays_per_proposal": 1024,
    "min_depth_m": 0.10,
    "max_depth_m": 8.0,
    "depth_edge_m": 0.15,
    "component_jump_m": 0.15,
    "min_fragment_points": 16,
    "voxel_size_m": _F_VOXEL_SIZE_METERS,
    "max_points_per_view": 512,
    "max_views_per_track": 5,
    "max_points_per_track": 1024,
    "max_tracks": 1024,
    "max_proposals_per_keyframe": 64,
    "timing_window": 4096,
})
DEFAULT_CONFIG = _DEFAULT_CONFIG

_FROZEN_GEOMETRY = (
    "pixel_stride",
    "min_depth_m",
    "max_depth_m",
    "depth_edge_m",
    "component_jump_m",
    "min_fragment_points",
    "voxel_size_m",
)

_HARD_CAPS = (
    "max_rays_per_proposal",
    "max_points_per_view",
    "max_views_per_track",
    "max_points_per_track",
    "max_tracks",
    "max_proposals_per_keyframe",
    "timing_window",
)


def _strict_int(name: str, value: object, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _strict_real(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def resolve_config(value: Optional[Mapping[str, object]] = None) -> Mapping[str, object]:
    """Return a validated copy of the shadow configuration.

    Geometry choices are frozen to make comparisons reproducible.  Resource
    caps may be reduced, but never raised above the declared hard limits.
    """

    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("SMOV-lite config must be a mapping")
    unknown = sorted(set(value) - set(_DEFAULT_CONFIG))
    if unknown:
        raise ValueError("unknown SMOV-lite config key(s): " + ", ".join(unknown))
    cfg = dict(_DEFAULT_CONFIG)
    cfg.update(value)

    for name in (
        "pixel_stride",
        "max_rays_per_proposal",
        "min_fragment_points",
        "max_points_per_view",
        "max_views_per_track",
        "max_points_per_track",
        "max_tracks",
        "max_proposals_per_keyframe",
        "timing_window",
    ):
        cfg[name] = _strict_int(name, cfg[name])
    for name in (
        "min_depth_m",
        "max_depth_m",
        "depth_edge_m",
        "component_jump_m",
        "voxel_size_m",
    ):
        cfg[name] = _strict_real(name, cfg[name])

    changed = [name for name in _FROZEN_GEOMETRY if cfg[name] != _DEFAULT_CONFIG[name]]
    if changed:
        raise ValueError("SMOV-lite geometry fields are frozen; changed: " + ", ".join(changed))
    exceeded = [name for name in _HARD_CAPS if cfg[name] > _DEFAULT_CONFIG[name]]
    if exceeded:
        raise ValueError("SMOV-lite resource caps exceed hard limits: " + ", ".join(exceeded))
    minimum = int(cfg["min_fragment_points"])
    if int(cfg["max_rays_per_proposal"]) < minimum:
        raise ValueError("max_rays_per_proposal cannot be below min_fragment_points")
    if int(cfg["max_points_per_view"]) < minimum:
        raise ValueError("max_points_per_view cannot be below min_fragment_points")
    if int(cfg["max_points_per_track"]) < minimum * int(cfg["max_views_per_track"]):
        raise ValueError(
            "max_points_per_track must reserve min_fragment_points for every retained view"
        )
    if int(cfg["max_points_per_view"]) > int(cfg["max_points_per_track"]):
        raise ValueError("max_points_per_view cannot exceed max_points_per_track")
    return MappingProxyType(cfg)


def _readonly(value: object, dtype: np.dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FragmentCoverage:
    effective_stride: int = 0
    sampled_rays: int = 0
    usable_rays: int = 0
    edge_pixels: int = 0
    component_pixels: int = 0
    unique_voxels: int = 0
    output_voxels: int = 0
    output_points: int = 0
    valid_depth_ratio: float = 0.0
    component_ratio: float = 0.0


@dataclass(frozen=True)
class ViewFragment:
    proposal_id: int
    frame_id: int
    score: float
    crop_xyxy_depth: np.ndarray
    depth_shape: tuple[int, int]
    proposal_to_depth_affine: np.ndarray
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    points_world: np.ndarray
    voxel_keys: np.ndarray
    coverage: FragmentCoverage

    def __post_init__(self) -> None:
        object.__setattr__(self, "crop_xyxy_depth", _readonly(self.crop_xyxy_depth, np.float32))
        object.__setattr__(
            self,
            "proposal_to_depth_affine",
            _readonly(self.proposal_to_depth_affine, np.float64),
        )
        object.__setattr__(self, "intrinsics", _readonly(self.intrinsics, np.float64))
        object.__setattr__(self, "camera_to_world", _readonly(self.camera_to_world, np.float64))
        points = _readonly(self.points_world, np.float32)
        keys = _readonly(self.voxel_keys, np.int64)
        if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
            raise ValueError("points_world must contain finite [N,3] rows")
        if keys.ndim != 2 or keys.shape[1:] != (3,) or len(keys) != len(points):
            raise ValueError("voxel_keys must align with points_world as [N,3]")
        object.__setattr__(self, "points_world", points)
        object.__setattr__(self, "voxel_keys", keys)

    @property
    def voxels(self) -> np.ndarray:
        """Read-only matcher-facing alias for the direct signed voxel keys."""

        return self.voxel_keys


@dataclass(frozen=True)
class ProposalDiagnostic:
    proposal_id: int
    selected: bool
    reason: Optional[str]
    coverage: FragmentCoverage
    elapsed_ms: float
    fragment: Optional[ViewFragment]

    @property
    def accepted(self) -> bool:
        return self.fragment is not None


@dataclass(frozen=True)
class PreparedKeyframe:
    scene_id: str
    frame_id: int
    proposal_ids: tuple[int, ...]
    diagnostics: tuple[ProposalDiagnostic, ...]
    elapsed_ms: float


@dataclass(frozen=True)
class CommitDecision:
    proposal_id: int
    track_id: Optional[int]
    accepted: bool
    reason: str


@dataclass(frozen=True)
class CommitResult:
    frame_id: int
    decisions: tuple[CommitDecision, ...]
    accepted_track_ids: tuple[int, ...]
    elapsed_ms: float


@dataclass(frozen=True)
class TrackSnapshot:
    track_id: int
    views: tuple[ViewFragment, ...]
    points_world: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "points_world", _readonly(self.points_world, np.float32))


@dataclass(frozen=True)
class TerminalSnapshot:
    scene_id: str
    frame_id: int
    tracks: tuple[TrackSnapshot, ...]
    elapsed_ms: float


class _Abstain(ValueError):
    def __init__(self, reason: str, coverage: Optional[FragmentCoverage] = None):
        super().__init__(reason)
        self.reason = reason
        self.coverage = coverage or FragmentCoverage()


def _validate_depth(value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise _Abstain("depth_m_must_be_numpy")
    if value.ndim != 2 or min(value.shape, default=0) < 1:
        raise _Abstain("invalid_depth_m")
    if int(value.shape[0]) * int(value.shape[1]) > _F_MAX_INPUT_DEPTH_PIXELS:
        raise _Abstain("depth_pixel_cap")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise _Abstain("invalid_depth_m") from error
    if raw.dtype.kind not in "iuf":
        raise _Abstain("invalid_depth_m")
    try:
        return np.array(raw, dtype=np.float32, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise _Abstain("invalid_depth_m") from error


def _validate_image_shape(value: Sequence[int]) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) < 2:
        raise ValueError("proposal_image_shape must contain positive integer H,W")
    height = _strict_int("proposal_image_shape[0]", value[0])
    width = _strict_int("proposal_image_shape[1]", value[1])
    return height, width


def aligned_resize_affine(
    proposal_image_shape: Sequence[int], depth_shape: Sequence[int]
) -> np.ndarray:
    """Construct the explicit axis-aligned registration used for resize-only input."""

    source_height, source_width = _validate_image_shape(proposal_image_shape)
    depth_height, depth_width = _validate_image_shape(depth_shape)
    result = np.array(
        [
            [depth_width / source_width, 0.0, 0.0],
            [0.0, depth_height / source_height, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    result.setflags(write=False)
    return result


def _validate_registration_affine(
    value: object,
    proposal_image_shape: tuple[int, int],
    depth_shape: tuple[int, int],
) -> np.ndarray:
    try:
        affine = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise _Abstain("invalid_proposal_to_depth_affine") from error
    if affine.shape != (3, 3) or not np.isfinite(affine).all():
        raise _Abstain("invalid_proposal_to_depth_affine")
    tolerance = 1e-9
    if np.max(np.abs(affine[2] - [0.0, 0.0, 1.0])) > tolerance:
        raise _Abstain("invalid_proposal_to_depth_affine")
    if abs(float(affine[0, 1])) > tolerance or abs(float(affine[1, 0])) > tolerance:
        raise _Abstain("proposal_to_depth_affine_must_be_axis_aligned")
    if affine[0, 0] <= 0.0 or affine[1, 1] <= 0.0:
        raise _Abstain("proposal_to_depth_affine_must_have_positive_scale")

    source_height, source_width = proposal_image_shape
    depth_height, depth_width = depth_shape
    mapped_x = np.asarray(
        [affine[0, 2], affine[0, 0] * source_width + affine[0, 2]]
    )
    mapped_y = np.asarray(
        [affine[1, 2], affine[1, 1] * source_height + affine[1, 2]]
    )
    if (
        mapped_x[1] <= 0.0
        or mapped_x[0] >= depth_width
        or mapped_y[1] <= 0.0
        or mapped_y[0] >= depth_height
    ):
        raise _Abstain("proposal_to_depth_affine_has_no_overlap")
    return np.array(affine, dtype=np.float64, order="C", copy=True)


def _validate_intrinsics(value: object, depth_shape: tuple[int, int]) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise _Abstain("invalid_intrinsics") from error
    if matrix.shape == (4, 4):
        matrix = matrix[:3, :3]
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise _Abstain("invalid_intrinsics")
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise _Abstain("invalid_intrinsics")
    if abs(float(np.linalg.det(matrix))) <= 1e-12:
        raise _Abstain("invalid_intrinsics")
    height, width = depth_shape
    if not (0.0 <= matrix[0, 2] < width and 0.0 <= matrix[1, 2] < height):
        raise _Abstain("invalid_intrinsics")
    return np.array(matrix, dtype=np.float64, order="C", copy=True)


def _validate_pose(value: object) -> np.ndarray:
    try:
        pose = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise _Abstain("invalid_camera_to_world") from error
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise _Abstain("invalid_camera_to_world")
    if np.max(np.abs(pose[3] - np.array([0.0, 0.0, 0.0, 1.0]))) > 1e-7:
        raise _Abstain("invalid_camera_to_world")
    rotation = pose[:3, :3]
    if (
        np.max(np.abs(rotation.T @ rotation - np.eye(3))) > 1e-4
        or abs(float(np.linalg.det(rotation)) - 1.0) > 1e-4
    ):
        raise _Abstain("invalid_camera_to_world")
    return np.array(pose, dtype=np.float64, order="C", copy=True)


def _full_resolution_edge_mask(
    depth: np.ndarray, cfg: Mapping[str, object]
) -> np.ndarray:
    usable = (
        np.isfinite(depth)
        & (depth >= float(cfg["min_depth_m"]))
        & (depth <= float(cfg["max_depth_m"]))
    )
    edge = np.zeros_like(usable)
    threshold = float(cfg["depth_edge_m"])
    horizontal = usable[:, :-1] & usable[:, 1:] & (
        np.abs(depth[:, :-1] - depth[:, 1:]) > threshold
    )
    vertical = usable[:-1, :] & usable[1:, :] & (
        np.abs(depth[:-1, :] - depth[1:, :]) > threshold
    )
    edge[:, :-1] |= horizontal
    edge[:, 1:] |= horizontal
    edge[:-1, :] |= vertical
    edge[1:, :] |= vertical
    return edge


def _voxelize(points: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned centroids and direct signed-floor voxel keys.

    ``np.unique(..., axis=0)`` provides a deterministic lexicographic key
    order.  Keys are computed from the original float64 world points; the
    float32 centroids retained for the legacy fragment-memory API are never
    used to derive matcher geometry.
    """

    if len(points) == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.int64),
        )
    scaled = points.astype(np.float64, copy=False) / voxel_size
    if (
        not np.isfinite(scaled).all()
        or np.max(np.abs(scaled), initial=0.0) > np.iinfo(np.int64).max / 4
    ):
        raise _Abstain("point_range_overflow")
    raw_keys = np.floor(scaled).astype(np.int64)
    keys, inverse, counts = np.unique(
        raw_keys, axis=0, return_inverse=True, return_counts=True
    )
    centroids = np.empty((len(counts), 3), dtype=np.float64)
    for axis in range(3):
        centroids[:, axis] = np.bincount(
            inverse, weights=points[:, axis], minlength=len(counts)
        ) / counts
    return centroids.astype(np.float32), keys.astype(np.int64, copy=False)


def _direct_voxel_keys(points_world: np.ndarray, voxel_size: float) -> np.ndarray:
    """Return the lexicographic key set under the signed half-open rule."""

    _, keys = _voxelize(np.asarray(points_world, dtype=np.float64), voxel_size)
    return keys


def _bounded_voxel_fragment(
    points: np.ndarray, keys: np.ndarray, maximum: int
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one deterministic cap while preserving point/key alignment."""

    if len(points) != len(keys):
        raise ValueError("voxel centroid/key rows must align")
    if len(keys) <= maximum:
        positions = np.arange(len(keys), dtype=np.int64)
    else:
        # Keys returned by _voxelize are already lexicographically sorted.
        positions = np.linspace(0, len(keys) - 1, maximum, dtype=np.int64)
    return (
        np.array(points[positions], dtype=np.float32, order="C", copy=True),
        np.array(keys[positions], dtype=np.int64, order="C", copy=True),
    )


def _bounded_sample(points: np.ndarray, maximum: int) -> np.ndarray:
    if len(points) <= maximum:
        return np.array(points, dtype=np.float32, order="C", copy=True)
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    positions = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
    return np.array(points[order[positions]], dtype=np.float32, order="C", copy=True)


def _extract_validated(
    *,
    proposal_id: int,
    frame_id: int,
    score: float,
    box_xyxy: object,
    proposal_image_shape: tuple[int, int],
    proposal_to_depth_affine: np.ndarray,
    depth: np.ndarray,
    depth_edge_mask: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    cfg: Mapping[str, object],
) -> ViewFragment:
    try:
        raw_box = np.asarray(box_xyxy, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise _Abstain("invalid_box_xyxy") from error
    if raw_box.shape != (4,) or not np.isfinite(raw_box).all():
        raise _Abstain("invalid_box_xyxy")
    depth_height, depth_width = depth.shape
    source_corners = np.asarray(
        [
            [raw_box[0], raw_box[1], 1.0],
            [raw_box[2], raw_box[1], 1.0],
            [raw_box[0], raw_box[3], 1.0],
            [raw_box[2], raw_box[3], 1.0],
        ],
        dtype=np.float64,
    )
    mapped_corners = source_corners @ proposal_to_depth_affine.T
    box = np.asarray(
        [
            np.min(mapped_corners[:, 0]),
            np.min(mapped_corners[:, 1]),
            np.max(mapped_corners[:, 0]),
            np.max(mapped_corners[:, 1]),
        ],
        dtype=np.float64,
    )
    box[[0, 2]] = np.clip(box[[0, 2]], 0.0, float(depth_width - 1))
    box[[1, 3]] = np.clip(box[[1, 3]], 0.0, float(depth_height - 1))
    if box[2] <= box[0] or box[3] <= box[1]:
        raise _Abstain("empty_mapped_crop")

    row_min, row_max = int(np.ceil(box[1])), int(np.floor(box[3]))
    col_min, col_max = int(np.ceil(box[0])), int(np.floor(box[2]))
    if row_min > row_max or col_min > col_max:
        raise _Abstain("empty_mapped_crop")
    stride = int(cfg["pixel_stride"])
    maximum_rays = int(cfg["max_rays_per_proposal"])
    rows = np.arange(row_min, row_max + 1, stride, dtype=np.int64)
    cols = np.arange(col_min, col_max + 1, stride, dtype=np.int64)
    while len(rows) * len(cols) > maximum_rays:
        stride += 1
        rows = np.arange(row_min, row_max + 1, stride, dtype=np.int64)
        cols = np.arange(col_min, col_max + 1, stride, dtype=np.int64)
    if not len(rows) or not len(cols):
        raise _Abstain("no_sampled_rays")

    grid_cols, grid_rows = np.meshgrid(cols, rows)
    sampled = depth[grid_rows, grid_cols]
    usable = (
        np.isfinite(sampled)
        & (sampled >= float(cfg["min_depth_m"]))
        & (sampled <= float(cfg["max_depth_m"]))
    )
    edge = depth_edge_mask[grid_rows, grid_cols]
    component_valid = usable & ~edge

    sampled_count = int(sampled.size)
    usable_count = int(np.count_nonzero(usable))
    edge_count = int(np.count_nonzero(edge))

    def coverage(
        component_pixels: int = 0,
        unique_voxels: int = 0,
        output_voxels: int = 0,
    ) -> FragmentCoverage:
        return FragmentCoverage(
            effective_stride=stride,
            sampled_rays=sampled_count,
            usable_rays=usable_count,
            edge_pixels=edge_count,
            component_pixels=component_pixels,
            unique_voxels=unique_voxels,
            output_voxels=output_voxels,
            output_points=output_voxels,
            valid_depth_ratio=usable_count / sampled_count,
            component_ratio=component_pixels / sampled_count,
        )

    target_row = int(np.argmin(np.abs(rows - (box[1] + box[3]) * 0.5)))
    target_col = int(np.argmin(np.abs(cols - (box[0] + box[2]) * 0.5)))
    if not component_valid[target_row, target_col]:
        raise _Abstain("center_seed_unusable", coverage())

    visited = np.zeros_like(component_valid)
    visited[target_row, target_col] = True
    queue: deque[tuple[int, int]] = deque([(target_row, target_col)])
    component: list[tuple[int, int]] = []
    jump = float(cfg["component_jump_m"])

    def full_resolution_path_connected(
        row: int, col: int, next_row: int, next_col: int
    ) -> bool:
        source_y, source_x = int(rows[row]), int(cols[col])
        target_y, target_x = int(rows[next_row]), int(cols[next_col])
        if source_y == target_y:
            left, right = sorted((source_x, target_x))
            path = depth[source_y, left : right + 1]
        else:
            top, bottom = sorted((source_y, target_y))
            path = depth[top : bottom + 1, source_x]
        path_usable = (
            np.isfinite(path)
            & (path >= float(cfg["min_depth_m"]))
            & (path <= float(cfg["max_depth_m"]))
        )
        return bool(
            np.all(path_usable)
            and np.all(np.abs(np.diff(path.astype(np.float64))) <= jump)
        )

    while queue:
        row, col = queue.popleft()
        component.append((row, col))
        for next_row, next_col in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if not (
                0 <= next_row < len(rows)
                and 0 <= next_col < len(cols)
                and not visited[next_row, next_col]
                and component_valid[next_row, next_col]
            ):
                continue
            if full_resolution_path_connected(row, col, next_row, next_col):
                visited[next_row, next_col] = True
                queue.append((next_row, next_col))

    component_count = len(component)
    if component_count < int(cfg["min_fragment_points"]):
        raise _Abstain("insufficient_component_pixels", coverage(component_count))

    indices = np.asarray(component, dtype=np.int64)
    component_rows = rows[indices[:, 0]]
    component_cols = cols[indices[:, 1]]
    component_depth = sampled[indices[:, 0], indices[:, 1]].astype(np.float64)
    pixels = np.column_stack(
        (component_cols, component_rows, np.ones(component_count, dtype=np.float64))
    ).astype(np.float64)
    rays_camera = pixels @ np.linalg.inv(intrinsics).T
    rays_camera /= rays_camera[:, 2:3]
    points_camera = rays_camera * component_depth[:, None]
    points_world = points_camera @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    points_world, voxel_keys = _voxelize(points_world, float(cfg["voxel_size_m"]))
    unique_count = len(voxel_keys)
    points_world, voxel_keys = _bounded_voxel_fragment(
        points_world, voxel_keys, int(cfg["max_points_per_view"])
    )
    if len(voxel_keys) < int(cfg["min_fragment_points"]):
        raise _Abstain(
            "insufficient_points_after_voxel",
            coverage(component_count, unique_count, len(voxel_keys)),
        )
    final_coverage = coverage(component_count, unique_count, len(voxel_keys))
    return ViewFragment(
        proposal_id=proposal_id,
        frame_id=frame_id,
        score=score,
        crop_xyxy_depth=box,
        depth_shape=(depth_height, depth_width),
        proposal_to_depth_affine=proposal_to_depth_affine,
        intrinsics=intrinsics,
        camera_to_world=camera_to_world,
        points_world=points_world,
        voxel_keys=voxel_keys,
        coverage=final_coverage,
    )


def extract_fragment(
    *,
    proposal_id: int,
    frame_id: int,
    score: float,
    box_xyxy: object,
    proposal_image_shape: Sequence[int],
    proposal_to_depth_affine: object,
    depth_m: object,
    intrinsics: object,
    camera_to_world: object,
    config: Optional[Mapping[str, object]] = None,
) -> ProposalDiagnostic:
    """Extract one fragment, returning an abstention diagnostic on data failure."""

    cfg = resolve_config(config)
    proposal_id = _strict_int("proposal_id", proposal_id, 0)
    frame_id = _strict_int("frame_id", frame_id, 0)
    score = _strict_real("score", score)
    image_shape = _validate_image_shape(proposal_image_shape)
    started = time.perf_counter_ns()
    try:
        depth = _validate_depth(depth_m)
        matrix = _validate_intrinsics(intrinsics, depth.shape)
        pose = _validate_pose(camera_to_world)
        registration = _validate_registration_affine(
            proposal_to_depth_affine, image_shape, depth.shape
        )
        depth_edge_mask = _full_resolution_edge_mask(depth, cfg)
        fragment = _extract_validated(
            proposal_id=proposal_id,
            frame_id=frame_id,
            score=score,
            box_xyxy=box_xyxy,
            proposal_image_shape=image_shape,
            proposal_to_depth_affine=registration,
            depth=depth,
            depth_edge_mask=depth_edge_mask,
            intrinsics=matrix,
            camera_to_world=pose,
            cfg=cfg,
        )
        reason = None
        coverage = fragment.coverage
    except (_Abstain, FloatingPointError, np.linalg.LinAlgError) as error:
        fragment = None
        if isinstance(error, _Abstain):
            reason, coverage = error.reason, error.coverage
        else:
            reason, coverage = "numeric_failure", FragmentCoverage()
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    return ProposalDiagnostic(
        proposal_id=proposal_id,
        selected=True,
        reason=reason,
        coverage=coverage,
        elapsed_ms=elapsed_ms,
        fragment=fragment,
    )


def _prepare_keyframe_batch(
    *,
    config: Mapping[str, object],
    scene_id: object,
    frame_id: object,
    proposal_ids: object,
    boxes_xyxy: object,
    proposal_scores: object,
    proposal_image_shape: Sequence[int],
    proposal_to_depth_affine: object,
    depth_m: object,
    intrinsics: object,
    camera_to_world: object,
) -> PreparedKeyframe:
    """Pure current-keyframe extraction shared by both public adapters."""

    started = time.perf_counter_ns()
    if not isinstance(scene_id, str) or not scene_id:
        raise ValueError("scene_id must be a non-empty string")
    validated_frame_id = _strict_int("frame_id", frame_id, 0)
    for name, value in (
        ("proposal_ids", proposal_ids),
        ("boxes_xyxy", boxes_xyxy),
        ("proposal_scores", proposal_scores),
    ):
        try:
            input_count = len(value)  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError(f"{name} must be a sized row sequence") from error
        if input_count > _F_MAX_INPUT_PROPOSALS:
            raise ValueError(
                f"{name} exceeds the hard input proposal cap of "
                f"{_F_MAX_INPUT_PROPOSALS}"
            )

    ids_raw = np.asarray(proposal_ids)
    if ids_raw.ndim != 1 or ids_raw.dtype.kind not in "iu":
        raise ValueError("proposal_ids must be a one-dimensional integer array")
    ids = ids_raw.astype(np.int64, copy=True)
    if np.any(ids < 0) or len(np.unique(ids)) != len(ids):
        raise ValueError("proposal_ids must be unique and nonnegative")
    try:
        boxes = np.asarray(boxes_xyxy)
        scores = np.asarray(proposal_scores)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("proposal arrays must be rectangular and row aligned") from error
    if boxes.shape != (len(ids), 4) or scores.shape != (len(ids),):
        raise ValueError("proposal arrays must be row aligned")
    if scores.dtype.kind not in "iuf" or not np.isfinite(scores).all():
        raise ValueError("proposal_scores must be finite numeric values")
    boxes = np.array(boxes, order="C", copy=True)
    scores = np.array(scores, dtype=np.float64, order="C", copy=True)
    image_shape = _validate_image_shape(proposal_image_shape)

    cap = int(config["max_proposals_per_keyframe"])
    order = np.lexsort((ids, -scores))
    selected_indices = set(int(index) for index in order[:cap])
    diagnostics: list[Optional[ProposalDiagnostic]] = [None] * len(ids)
    for index, proposal_id in enumerate(ids):
        if index not in selected_indices:
            diagnostics[index] = ProposalDiagnostic(
                proposal_id=int(proposal_id),
                selected=False,
                reason="proposal_cap",
                coverage=FragmentCoverage(),
                elapsed_ms=0.0,
                fragment=None,
            )

    try:
        depth = _validate_depth(depth_m)
        matrix = _validate_intrinsics(intrinsics, depth.shape)
        pose = _validate_pose(camera_to_world)
        registration = _validate_registration_affine(
            proposal_to_depth_affine, image_shape, depth.shape
        )
        depth_edge_mask = _full_resolution_edge_mask(depth, config)
        global_failure: Optional[_Abstain] = None
    except (_Abstain, FloatingPointError, np.linalg.LinAlgError) as error:
        depth = np.empty((0, 0), dtype=np.float32)
        matrix = np.eye(3, dtype=np.float64)
        pose = np.eye(4, dtype=np.float64)
        registration = np.eye(3, dtype=np.float64)
        depth_edge_mask = np.empty((0, 0), dtype=bool)
        global_failure = (
            error
            if isinstance(error, _Abstain)
            else _Abstain("numeric_failure")
        )

    for index in sorted(selected_indices):
        proposal_started = time.perf_counter_ns()
        fragment: Optional[ViewFragment] = None
        try:
            if global_failure is not None:
                raise global_failure
            fragment = _extract_validated(
                proposal_id=int(ids[index]),
                frame_id=validated_frame_id,
                score=float(scores[index]),
                box_xyxy=boxes[index],
                proposal_image_shape=image_shape,
                proposal_to_depth_affine=registration,
                depth=depth,
                depth_edge_mask=depth_edge_mask,
                intrinsics=matrix,
                camera_to_world=pose,
                cfg=config,
            )
            reason, coverage = None, fragment.coverage
        except (_Abstain, FloatingPointError, np.linalg.LinAlgError) as error:
            if isinstance(error, _Abstain):
                reason, coverage = error.reason, error.coverage
            else:
                reason, coverage = "numeric_failure", FragmentCoverage()
        diagnostics[index] = ProposalDiagnostic(
            proposal_id=int(ids[index]),
            selected=True,
            reason=reason,
            coverage=coverage,
            elapsed_ms=(time.perf_counter_ns() - proposal_started) / 1e6,
            fragment=fragment,
        )

    completed = tuple(item for item in diagnostics if item is not None)
    return PreparedKeyframe(
        scene_id=scene_id,
        frame_id=validated_frame_id,
        proposal_ids=tuple(int(value) for value in ids),
        diagnostics=completed,
        elapsed_ms=(time.perf_counter_ns() - started) / 1e6,
    )


def smov_batch_to_dict(batch: PreparedKeyframe) -> dict[str, object]:
    """Convert one prepared keyframe into a deterministic JSON-ready record."""

    if not isinstance(batch, PreparedKeyframe):
        raise TypeError("batch must be a PreparedKeyframe")
    selected = tuple(item for item in batch.diagnostics if item.selected)
    accepted = tuple(item for item in selected if item.accepted)
    abstained = tuple(item for item in selected if not item.accepted)
    capped = tuple(item for item in batch.diagnostics if not item.selected)
    failures = Counter(
        item.reason or "unknown" for item in batch.diagnostics if not item.accepted
    )
    unique_counts = [item.coverage.unique_voxels for item in accepted]
    output_counts = [item.coverage.output_voxels for item in accepted]

    def voxel_summary(values: Sequence[int]) -> dict[str, object]:
        return {
            "total": int(sum(values)),
            "min": int(min(values)) if values else 0,
            "mean": float(np.mean(values)) if values else 0.0,
            "max": int(max(values)) if values else 0,
        }

    proposals: list[dict[str, object]] = []
    for item in batch.diagnostics:
        coverage = item.coverage
        proposals.append(
            {
                "proposal_id": int(item.proposal_id),
                "selected": bool(item.selected),
                "accepted": bool(item.accepted),
                "reason": item.reason,
                "elapsed_ms": float(item.elapsed_ms),
                "sampled_rays": int(coverage.sampled_rays),
                "usable_rays": int(coverage.usable_rays),
                "edge_pixels": int(coverage.edge_pixels),
                "component_pixels": int(coverage.component_pixels),
                "unique_voxels": int(coverage.unique_voxels),
                "output_voxels": int(coverage.output_voxels),
                "valid_depth_ratio": float(coverage.valid_depth_ratio),
                "component_ratio": float(coverage.component_ratio),
            }
        )

    return {
        "scene_id": batch.scene_id,
        "frame_id": int(batch.frame_id),
        "prepare_elapsed_ms": float(batch.elapsed_ms),
        "proposal_count": len(batch.proposal_ids),
        "selected_count": len(selected),
        "accepted_count": len(accepted),
        "abstained_count": len(abstained),
        "capped_count": len(capped),
        "selected_proposal_ids": [int(item.proposal_id) for item in selected],
        "accepted_proposal_ids": [int(item.proposal_id) for item in accepted],
        "abstained_proposal_ids": [int(item.proposal_id) for item in abstained],
        "capped_proposal_ids": [int(item.proposal_id) for item in capped],
        "failure_reasons": dict(sorted(failures.items())),
        "voxel_statistics": {
            "unique": voxel_summary(unique_counts),
            "output": voxel_summary(output_counts),
        },
        "proposals": proposals,
    }


def _plain_json_value(value: object, *, depth: int = 0) -> object:
    """Copy bounded diagnostic data into JSON-native containers."""

    if depth > 16:
        raise ValueError("SMOV diagnostic nesting exceeds the depth cap")
    if isinstance(value, Mapping):
        if len(value) > 131_072:
            raise ValueError("SMOV diagnostic mapping exceeds the item cap")
        return {
            str(key): _plain_json_value(item, depth=depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        if len(value) > 131_072:
            raise ValueError("SMOV diagnostic sequence exceeds the item cap")
        return [_plain_json_value(item, depth=depth + 1) for item in value]
    if isinstance(value, np.generic):
        return _plain_json_value(value.item(), depth=depth + 1)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(
        "SMOV diagnostics contain a non-JSON value: " + type(value).__name__
    )


def write_smov_shadow_diagnostics(
    path: os.PathLike[str] | str,
    scene_id: str,
    records: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    trace_valid: bool,
) -> str:
    """Atomically write one bounded ``.smov_shadow.json`` scene trace."""

    if not isinstance(scene_id, str) or not scene_id:
        raise ValueError("scene_id must be a non-empty string")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError("records must be a sequence")
    if len(records) > _F_MAX_DIAGNOSTIC_FRAMES:
        raise ValueError("SMOV diagnostic frame count exceeds the hard cap")
    if not isinstance(summary, Mapping):
        raise ValueError("summary must be a mapping")
    if not isinstance(trace_valid, (bool, np.bool_)):
        raise ValueError("trace_valid must be a boolean")
    destination = os.path.abspath(os.fspath(path))
    payload = {
        "schema": SCHEMA,
        "scene_id": scene_id,
        "trace_valid": bool(trace_valid),
        "frame_count": len(records),
        "frames": _plain_json_value(records),
        "summary": _plain_json_value(summary),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(encoded) > _F_MAX_DIAGNOSTIC_BYTES:
        raise ValueError("SMOV diagnostic exceeds the 32 MiB cap")

    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + os.path.basename(destination) + ".",
        suffix=".tmp",
        dir=parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


class SMOVFragmentExtractor:
    """Stateless, batch-only clean fragment extractor for Gclean sidecars.

    The instance owns only an immutable configuration.  It has no pending
    transaction, track memory, scene memory, or frame history; every call is a
    deterministic function of that call's RGB-D geometry inputs.
    """

    def __setattr__(self, name: str, value: object) -> None:
        if name == "_config":
            try:
                object.__getattribute__(self, "_config")
            except AttributeError:
                pass
            else:
                raise AttributeError("_config is write-once")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if name == "_config":
            raise AttributeError("_config is write-once and cannot be deleted")
        object.__delattr__(self, name)

    def __init__(self, config: Optional[Mapping[str, object]] = None):
        self._config = resolve_config(config)

    @property
    def config(self) -> Mapping[str, object]:
        return self._config

    def prepare_keyframe(
        self,
        *,
        scene_id: str,
        frame_id: int,
        proposal_ids: object,
        boxes_xyxy: object,
        proposal_scores: object,
        proposal_image_shape: Sequence[int],
        proposal_to_depth_affine: object,
        depth_m: object,
        intrinsics: object,
        camera_to_world: object,
    ) -> PreparedKeyframe:
        return _prepare_keyframe_batch(
            config=self._config,
            scene_id=scene_id,
            frame_id=frame_id,
            proposal_ids=proposal_ids,
            boxes_xyxy=boxes_xyxy,
            proposal_scores=proposal_scores,
            proposal_image_shape=proposal_image_shape,
            proposal_to_depth_affine=proposal_to_depth_affine,
            depth_m=depth_m,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
        )


class FragmentShadow:
    """Causal prepare/commit adapter with bounded fragment-only track state."""

    def __setattr__(self, name: str, value: object) -> None:
        if name == "_config":
            try:
                object.__getattribute__(self, "_config")
            except AttributeError:
                pass
            else:
                raise AttributeError("_config is write-once")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if name == "_config":
            raise AttributeError("_config is write-once and cannot be deleted")
        object.__delattr__(self, name)

    def __init__(self, config: Optional[Mapping[str, object]] = None):
        self._config = resolve_config(config)
        self._scene_id: Optional[str] = None
        self._pending: Optional[PreparedKeyframe] = None
        self._last_prepared_frame: Optional[int] = None
        self._last_committed_frame: Optional[int] = None
        self._closed = False
        self._tracks: dict[int, list[ViewFragment]] = {}
        window = int(self._config["timing_window"])
        self._prepare_timings: deque[float] = deque(maxlen=window)
        self._commit_timings: deque[float] = deque(maxlen=window)
        self._terminal_timings: deque[float] = deque(maxlen=window)
        self._failure_reasons: Counter[str] = Counter()
        self._coverages: deque[FragmentCoverage] = deque(maxlen=window)
        self._stats: Counter[str] = Counter()

    @property
    def config(self) -> Mapping[str, object]:
        """Immutable effective configuration; rebinding is intentionally forbidden."""

        return self._config

    def _bind_scene(self, scene_id: object) -> str:
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError("scene_id must be a non-empty string")
        if self._scene_id is None:
            self._scene_id = scene_id
        elif self._scene_id != scene_id:
            raise ValueError("one FragmentShadow instance cannot span scenes")
        return scene_id

    def prepare_keyframe(
        self,
        *,
        scene_id: str,
        frame_id: int,
        proposal_ids: object,
        boxes_xyxy: object,
        proposal_scores: object,
        proposal_image_shape: Sequence[int],
        proposal_to_depth_affine: object,
        depth_m: object,
        intrinsics: object,
        camera_to_world: object,
    ) -> PreparedKeyframe:
        if self._closed or self._pending is not None:
            raise RuntimeError("fragment prepare/commit transaction is not idle")
        scene_id = self._bind_scene(scene_id)
        frame_id = _strict_int("frame_id", frame_id, 0)
        if self._last_prepared_frame is not None and frame_id <= self._last_prepared_frame:
            raise ValueError("frame_id must be strictly increasing")
        batch = _prepare_keyframe_batch(
            config=self._config,
            scene_id=scene_id,
            frame_id=frame_id,
            proposal_ids=proposal_ids,
            boxes_xyxy=boxes_xyxy,
            proposal_scores=proposal_scores,
            proposal_image_shape=proposal_image_shape,
            proposal_to_depth_affine=proposal_to_depth_affine,
            depth_m=depth_m,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
        )
        self._pending = batch
        self._last_prepared_frame = frame_id
        self._prepare_timings.append(batch.elapsed_ms)
        self._stats["keyframes"] += 1
        self._stats["proposals"] += len(batch.proposal_ids)
        for diagnostic in batch.diagnostics:
            if diagnostic.accepted:
                self._stats["valid_fragments"] += 1
                self._coverages.append(diagnostic.coverage)
            else:
                self._stats["abstained_fragments"] += 1
                self._failure_reasons[diagnostic.reason or "unknown"] += 1
        return batch

    @staticmethod
    def _validate_track_ids(value: Sequence[Optional[int]], count: int) -> tuple[Optional[int], ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != count:
            raise ValueError("track_ids must align with the prepared proposals")
        result: list[Optional[int]] = []
        for item in value:
            result.append(None if item is None else _strict_int("track_id", item, 0))
        return tuple(result)

    def _validate_track_aliases(
        self,
        value: Optional[Mapping[int, int]],
        active: Optional[set[int]],
    ) -> dict[int, int]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("track_aliases must be a mapping")
        if len(value) > int(self._config["max_tracks"]):
            raise ValueError("track_aliases exceeds the configured track cap")
        if value and active is None:
            raise ValueError("track_aliases requires active_track_ids")
        aliases: dict[int, int] = {}
        for raw_source, raw_target in value.items():
            source = _strict_int("track_alias source", raw_source, 0)
            target = _strict_int("track_alias target", raw_target, 0)
            aliases[source] = target

        for start in aliases:
            seen: set[int] = set()
            current = start
            while current in aliases:
                if current in seen:
                    raise ValueError("track_aliases must be acyclic")
                seen.add(current)
                current = aliases[current]

        active_ids = active or set()
        if any(source in active_ids for source in aliases):
            raise ValueError("track_alias sources must not remain active")
        if any(target not in active_ids for target in aliases.values()):
            raise ValueError("track_alias targets must be active")
        return aliases

    def _bounded_views(self, views: Sequence[ViewFragment]) -> list[ViewFragment]:
        per_frame: dict[int, ViewFragment] = {}
        for view in views:
            current = per_frame.get(view.frame_id)
            candidate_key = (
                -len(view.points_world),
                -view.coverage.valid_depth_ratio,
                -view.score,
                view.proposal_id,
            )
            if current is None:
                per_frame[view.frame_id] = view
            else:
                current_key = (
                    -len(current.points_world),
                    -current.coverage.valid_depth_ratio,
                    -current.score,
                    current.proposal_id,
                )
                if candidate_key < current_key:
                    per_frame[view.frame_id] = view
        ordered = [per_frame[key] for key in sorted(per_frame)]
        limit = int(self._config["max_views_per_track"])
        ordered = ordered[-limit:]
        if not ordered:
            return []
        total_budget = int(self._config["max_points_per_track"])
        base, remainder = divmod(total_budget, len(ordered))
        bounded: list[ViewFragment] = []
        for index, view in enumerate(ordered):
            allowance = base + (1 if index < remainder else 0)
            points, voxel_keys = _bounded_voxel_fragment(
                view.points_world, view.voxel_keys, allowance
            )
            coverage = replace(
                view.coverage,
                output_points=len(points),
                output_voxels=len(voxel_keys),
            )
            bounded.append(
                replace(
                    view,
                    points_world=points,
                    voxel_keys=voxel_keys,
                    coverage=coverage,
                )
            )
        return bounded

    def commit_keyframe(
        self,
        batch: PreparedKeyframe,
        *,
        track_ids: Sequence[Optional[int]],
        active_track_ids: Optional[Sequence[int]] = None,
        track_aliases: Optional[Mapping[int, int]] = None,
    ) -> CommitResult:
        started = time.perf_counter_ns()
        if batch is not self._pending:
            raise RuntimeError("commit must consume the exact pending batch")
        try:
            aligned_tracks = self._validate_track_ids(
                track_ids, len(batch.proposal_ids)
            )
            active: Optional[set[int]] = None
            if active_track_ids is not None:
                if isinstance(active_track_ids, (str, bytes)) or not isinstance(
                    active_track_ids, Sequence
                ):
                    raise ValueError("active_track_ids must be a sequence")
                if len(active_track_ids) > int(self._config["max_tracks"]):
                    raise ValueError(
                        "active_track_ids exceeds the configured track cap"
                    )
                active_list = [
                    _strict_int("active_track_id", item, 0)
                    for item in active_track_ids
                ]
                if len(set(active_list)) != len(active_list):
                    raise ValueError("active_track_ids must be unique")
                active = set(active_list)
            aliases = self._validate_track_aliases(track_aliases, active)

            staged_tracks = {
                track_id: list(views) for track_id, views in self._tracks.items()
            }
            alias_merge_inputs: dict[int, list[ViewFragment]] = {}
            absorbed_aliases = 0
            for source, target in sorted(aliases.items()):
                source_views = staged_tracks.get(source)
                if source_views:
                    alias_merge_inputs.setdefault(target, []).extend(source_views)
                    absorbed_aliases += 1
            for source in aliases:
                staged_tracks.pop(source, None)
            for target, source_views in sorted(alias_merge_inputs.items()):
                staged_tracks[target] = self._bounded_views(
                    [*staged_tracks.get(target, []), *source_views]
                )

            retired_count = 0
            if active is not None:
                for track_id in tuple(staged_tracks):
                    if track_id not in active:
                        staged_tracks.pop(track_id)
                        retired_count += 1

            decisions: list[Optional[CommitDecision]] = [None] * len(
                batch.proposal_ids
            )
            candidates: dict[int, list[tuple[int, ViewFragment]]] = {}
            for index, (diagnostic, track_id) in enumerate(
                zip(batch.diagnostics, aligned_tracks)
            ):
                if track_id is None:
                    decisions[index] = CommitDecision(
                        diagnostic.proposal_id, None, False, "no_track_id"
                    )
                elif active is not None and track_id not in active:
                    decisions[index] = CommitDecision(
                        diagnostic.proposal_id,
                        track_id,
                        False,
                        "inactive_track_id",
                    )
                elif diagnostic.fragment is None:
                    decisions[index] = CommitDecision(
                        diagnostic.proposal_id,
                        track_id,
                        False,
                        diagnostic.reason or "fragment_abstained",
                    )
                else:
                    candidates.setdefault(track_id, []).append(
                        (index, diagnostic.fragment)
                    )

            accepted_tracks: list[int] = []
            for track_id in sorted(candidates):
                choices = candidates[track_id]
                winner_index, winner = min(
                    choices,
                    key=lambda item: (
                        -len(item[1].points_world),
                        -item[1].coverage.valid_depth_ratio,
                        -item[1].score,
                        item[1].proposal_id,
                    ),
                )
                for index, fragment in choices:
                    if index != winner_index:
                        decisions[index] = CommitDecision(
                            fragment.proposal_id,
                            track_id,
                            False,
                            "same_frame_duplicate",
                        )
                if (
                    track_id not in staged_tracks
                    and len(staged_tracks) >= int(self._config["max_tracks"])
                ):
                    decisions[winner_index] = CommitDecision(
                        winner.proposal_id,
                        track_id,
                        False,
                        "track_capacity",
                    )
                    continue
                staged_tracks[track_id] = self._bounded_views(
                    [*staged_tracks.get(track_id, []), winner]
                )
                decisions[winner_index] = CommitDecision(
                    winner.proposal_id, track_id, True, "accepted"
                )
                accepted_tracks.append(track_id)
        except Exception:
            self._pending = None
            self._stats["failed_commits"] += 1
            self._failure_reasons["commit:transaction_failure"] += 1
            raise

        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        completed_decisions = tuple(item for item in decisions if item is not None)
        result = CommitResult(
            frame_id=batch.frame_id,
            decisions=completed_decisions,
            accepted_track_ids=tuple(accepted_tracks),
            elapsed_ms=elapsed_ms,
        )
        for decision in completed_decisions:
            self._stats["accepted_views" if decision.accepted else "commit_abstentions"] += 1
            if not decision.accepted:
                self._failure_reasons["commit:" + decision.reason] += 1
        self._tracks = staged_tracks
        self._stats["retired_tracks"] += retired_count
        self._stats["absorbed_track_aliases"] += absorbed_aliases
        self._pending = None
        self._last_committed_frame = batch.frame_id
        self._commit_timings.append(elapsed_ms)
        return result

    def abort_keyframe(
        self, batch: PreparedKeyframe, *, reason: AbortReason
    ) -> None:
        """Explicitly discard the exact pending transaction without state changes."""

        if batch is not self._pending:
            raise RuntimeError("abort must consume the exact pending batch")
        if not isinstance(reason, AbortReason):
            raise ValueError("abort reason must be an AbortReason")
        self._pending = None
        self._stats["aborted_keyframes"] += 1
        self._failure_reasons["abort:" + reason.value] += 1

    def terminal_snapshot(self, *, frame_id: int, close: bool = False) -> TerminalSnapshot:
        """Snapshot only the most recently committed frame; stale reads fail."""

        started = time.perf_counter_ns()
        frame_id = _strict_int("frame_id", frame_id, 0)
        if not isinstance(close, (bool, np.bool_)):
            raise ValueError("close must be a boolean")
        if self._pending is not None:
            raise RuntimeError("terminal observation cannot bypass a pending keyframe")
        if self._last_committed_frame is None or frame_id != self._last_committed_frame:
            raise ValueError("terminal frame_id must equal the most recently committed frame")
        tracks: list[TrackSnapshot] = []
        for track_id in sorted(self._tracks):
            views = tuple(self._tracks[track_id])
            points = (
                np.concatenate([view.points_world for view in views], axis=0)
                if views
                else np.empty((0, 3), dtype=np.float32)
            )
            points = _bounded_sample(points, int(self._config["max_points_per_track"]))
            tracks.append(TrackSnapshot(track_id=track_id, views=views, points_world=points))
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        self._terminal_timings.append(elapsed_ms)
        if close:
            self._closed = True
        return TerminalSnapshot(
            scene_id=self._scene_id or "",
            frame_id=frame_id,
            tracks=tuple(tracks),
            elapsed_ms=elapsed_ms,
        )

    def diagnostics(self) -> dict[str, object]:
        """Return copied counters, coverage summaries, and bounded timings."""

        coverages = tuple(self._coverages)

        def timing(values: deque[float]) -> dict[str, object]:
            copied = tuple(float(value) for value in values)
            return {
                "samples_ms": copied,
                "mean_ms": float(np.mean(copied)) if copied else 0.0,
                "max_ms": float(np.max(copied)) if copied else 0.0,
            }

        return {
            "schema": SCHEMA,
            "scene_id": self._scene_id,
            "last_prepared_frame": self._last_prepared_frame,
            "last_committed_frame": self._last_committed_frame,
            "pending": self._pending is not None,
            "closed": self._closed,
            "tracks": len(self._tracks),
            "stats": dict(self._stats),
            "failure_reasons": dict(self._failure_reasons),
            "coverage": {
                "samples": len(coverages),
                "mean_valid_depth_ratio": (
                    float(np.mean([item.valid_depth_ratio for item in coverages]))
                    if coverages
                    else 0.0
                ),
                "mean_component_ratio": (
                    float(np.mean([item.component_ratio for item in coverages]))
                    if coverages
                    else 0.0
                ),
            },
            "timing": {
                "prepare": timing(self._prepare_timings),
                "commit": timing(self._commit_timings),
                "terminal": timing(self._terminal_timings),
            },
        }


__all__ = [
    "SCHEMA",
    "VOXEL_SIZE_METERS",
    "DEFAULT_CONFIG",
    "MAX_INPUT_DEPTH_PIXELS",
    "MAX_INPUT_PROPOSALS",
    "AbortReason",
    "FragmentCoverage",
    "ViewFragment",
    "ProposalDiagnostic",
    "PreparedKeyframe",
    "CommitDecision",
    "CommitResult",
    "TrackSnapshot",
    "TerminalSnapshot",
    "SMOVFragmentExtractor",
    "FragmentShadow",
    "aligned_resize_affine",
    "extract_fragment",
    "resolve_config",
    "smov_batch_to_dict",
    "write_smov_shadow_diagnostics",
]
