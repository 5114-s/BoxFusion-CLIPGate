"""Deterministic unexplained-depth box prompts for frozen MobileSAM.

This module is a deliberately small, training-free bridge between a current
RGB-D keyframe and a box-prompted, frozen segmentation model.  It accepts only
the current depth image, calibrated camera geometry, and current-frame CuTR
boxes that already explain image regions.  It has no ground-truth, semantic,
history, detector, or model dependency.

Coordinate contract
-------------------
``depth_m`` is an ``H x W`` numeric NumPy array in metres.  CuTR
``explained_boxes_xyxy`` are continuous, zero-based ``(x1, y1, x2, y2)``
coordinates registered to that same depth image.  Box endpoints are treated
as inclusive when masking stride-four sample centres.  Returned prompt boxes
are inclusive depth-image coordinates suitable for a 640 x 480 MobileSAM
box-prompt caller; no historical 960 x 960 Boxer resize is applied here.

The fixed policy first samples pixel centres ``0, 4, 8, ...``.  Invalid depth,
full-resolution four-neighbour depth jumps greater than 0.15 m (expanded by a
7 x 7 barrier before sampling), stride-grid neighbour jumps greater than the
same threshold, and samples inside a CuTR box expanded by four pixels are
removed.  Eight-connected components are then built with OpenCV.  Components
pass fixed pixel, border, and robust world-AABB filters before the largest two
are returned.  Equal-area ties are resolved top-to-bottom, then left-to-right,
independently of input box order.  Inputs are copied before processing and are
never made writeable or modified by the implementation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping, Optional

import cv2
import numpy as np


SCHEMA = "boxfusion.udc_mobilesam.v1"

# Public audit aliases.  Executable policy uses the private constants below so
# rebinding a public name cannot silently alter a sealed experiment.
PIXEL_STRIDE = 4
MIN_DEPTH_M = 0.10
MAX_DEPTH_M = 6.0
DEPTH_EDGE_JUMP_M = 0.15
EXPLAINED_BOX_EXPAND_PX = 4.0
MIN_COMPONENT_GRID_PIXELS = 24
MAX_COMPONENT_GRID_PIXELS = 5_000
MIN_BBOX_GRID_SIZE = 5
MAX_TOUCHING_BORDERS = 1
WORLD_QUANTILE_LOW = 0.02
WORLD_QUANTILE_HIGH = 0.98
MIN_WORLD_EXTENT_M = 0.05
MAX_WORLD_EXTENT_M = 2.5
MAX_WORLD_DIAGONAL_M = 3.0
MAX_WORLD_VOLUME_M3 = 4.0
TOP_K = 2
VOXEL_SIZE_METERS = 0.05

_F_PIXEL_STRIDE = 4
_F_MIN_DEPTH_M = 0.10
_F_MAX_DEPTH_M = 6.0
_F_DEPTH_EDGE_JUMP_M = 0.15
_F_EXPLAINED_BOX_EXPAND_PX = 4.0
_F_MIN_COMPONENT_GRID_PIXELS = 24
_F_MAX_COMPONENT_GRID_PIXELS = 5_000
_F_MIN_BBOX_GRID_SIZE = 5
_F_MAX_TOUCHING_BORDERS = 1
_F_WORLD_QUANTILE_LOW = 0.02
_F_WORLD_QUANTILE_HIGH = 0.98
_F_MIN_WORLD_EXTENT_M = 0.05
_F_MAX_WORLD_EXTENT_M = 2.5
_F_MAX_WORLD_DIAGONAL_M = 3.0
_F_MAX_WORLD_VOLUME_M3 = 4.0
_F_TOP_K = 2
_F_VOXEL_SIZE_METERS = 0.05

MAX_INPUT_DEPTH_PIXELS = 4_194_304
MAX_EXPLAINED_BOXES = 4_096
_F_MAX_INPUT_DEPTH_PIXELS = 4_194_304
_F_MAX_EXPLAINED_BOXES = 4_096

POLICY: Mapping[str, object] = MappingProxyType(
    {
        "pixel_stride": _F_PIXEL_STRIDE,
        "min_depth_m": _F_MIN_DEPTH_M,
        "max_depth_m": _F_MAX_DEPTH_M,
        "depth_edge_jump_m": _F_DEPTH_EDGE_JUMP_M,
        "explained_box_expand_px": _F_EXPLAINED_BOX_EXPAND_PX,
        "component_connectivity": 8,
        "min_component_grid_pixels": _F_MIN_COMPONENT_GRID_PIXELS,
        "max_component_grid_pixels": _F_MAX_COMPONENT_GRID_PIXELS,
        "min_bbox_grid_size": _F_MIN_BBOX_GRID_SIZE,
        "reject_touches_at_least_borders": _F_MAX_TOUCHING_BORDERS + 1,
        "world_quantiles": (_F_WORLD_QUANTILE_LOW, _F_WORLD_QUANTILE_HIGH),
        "min_world_extent_m": _F_MIN_WORLD_EXTENT_M,
        "max_world_extent_m": _F_MAX_WORLD_EXTENT_M,
        "max_world_diagonal_m": _F_MAX_WORLD_DIAGONAL_M,
        "max_world_volume_m3": _F_MAX_WORLD_VOLUME_M3,
        "top_k": _F_TOP_K,
        "voxel_size_m": _F_VOXEL_SIZE_METERS,
    }
)


def _readonly(
    value: object,
    dtype: np.dtype,
    shape: Optional[tuple[int, ...]] = None,
) -> np.ndarray:
    """Return a detached array backed by immutable bytes."""

    array = np.asarray(value, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError(f"array must have shape {shape}, got {array.shape}")
    packed = np.ascontiguousarray(array).tobytes()
    return np.frombuffer(packed, dtype=dtype).reshape(array.shape)


@dataclass(frozen=True)
class UDCPrompt:
    """One accepted MobileSAM box prompt and its auditable geometry."""

    rank: int
    component_id: int
    box_xyxy: np.ndarray
    grid_pixel_count: int
    source_pixels_yx: np.ndarray
    voxel_keys: np.ndarray
    world_q02: np.ndarray
    world_q98: np.ndarray
    world_extent: np.ndarray
    world_diagonal_m: float
    world_volume_m3: float

    def __post_init__(self) -> None:
        if self.rank < 0 or self.rank >= _F_TOP_K:
            raise ValueError("rank must be in the frozen Top-K range")
        if self.component_id < 1 or self.grid_pixel_count < 1:
            raise ValueError("component identifiers and counts must be positive")
        object.__setattr__(self, "box_xyxy", _readonly(self.box_xyxy, np.float32, (4,)))
        object.__setattr__(
            self,
            "source_pixels_yx",
            _readonly(
                self.source_pixels_yx,
                np.int64,
                (self.grid_pixel_count, 2),
            ),
        )
        voxel_count = len(np.asarray(self.voxel_keys))
        object.__setattr__(
            self,
            "voxel_keys",
            _readonly(self.voxel_keys, np.int64, (voxel_count, 3)),
        )
        object.__setattr__(self, "world_q02", _readonly(self.world_q02, np.float64, (3,)))
        object.__setattr__(self, "world_q98", _readonly(self.world_q98, np.float64, (3,)))
        object.__setattr__(
            self, "world_extent", _readonly(self.world_extent, np.float64, (3,))
        )


@dataclass(frozen=True)
class UDCComponentDiagnostic:
    """Disposition and measurements for every residual connected component."""

    component_id: int
    grid_pixel_count: int
    grid_bbox_xywh: np.ndarray
    box_xyxy: np.ndarray
    touching_borders: int
    depth_q02_q98_m: np.ndarray
    world_q02: np.ndarray
    world_q98: np.ndarray
    world_extent: np.ndarray
    world_diagonal_m: float
    world_volume_m3: float
    eligible: bool
    selected: bool
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "grid_bbox_xywh", _readonly(self.grid_bbox_xywh, np.int64, (4,))
        )
        object.__setattr__(self, "box_xyxy", _readonly(self.box_xyxy, np.float32, (4,)))
        object.__setattr__(
            self,
            "depth_q02_q98_m",
            _readonly(self.depth_q02_q98_m, np.float64, (2,)),
        )
        object.__setattr__(self, "world_q02", _readonly(self.world_q02, np.float64, (3,)))
        object.__setattr__(self, "world_q98", _readonly(self.world_q98, np.float64, (3,)))
        object.__setattr__(
            self, "world_extent", _readonly(self.world_extent, np.float64, (3,))
        )
        if self.selected and not self.eligible:
            raise ValueError("a selected component must be eligible")

    @property
    def accepted(self) -> bool:
        """Compatibility alias: accepted means all geometric filters passed."""

        return self.eligible


@dataclass(frozen=True)
class UDCFrameDiagnostics:
    """Frame-level capacity and rejection accounting without wall-clock noise."""

    schema: str
    depth_shape: tuple[int, int]
    grid_shape: tuple[int, int]
    input_explained_box_count: int
    sampled_grid_pixels: int
    valid_depth_grid_pixels: int
    edge_rejected_grid_pixels: int
    explained_valid_grid_pixels: int
    residual_grid_pixels: int
    component_count: int
    eligible_component_count: int
    selected_component_count: int
    rejection_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        counts = {
            str(key): int(value)
            for key, value in sorted(dict(self.rejection_counts).items())
        }
        if any(value < 0 for value in counts.values()):
            raise ValueError("rejection counts cannot be negative")
        object.__setattr__(self, "rejection_counts", MappingProxyType(counts))


@dataclass(frozen=True)
class UDCResult:
    """Immutable prompts plus complete per-component and frame diagnostics."""

    prompts: tuple[UDCPrompt, ...]
    components: tuple[UDCComponentDiagnostic, ...]
    diagnostics: UDCFrameDiagnostics

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompts", tuple(self.prompts))
        object.__setattr__(self, "components", tuple(self.components))
        if len(self.prompts) > _F_TOP_K:
            raise ValueError("result exceeds the frozen Top-K")
        if tuple(prompt.rank for prompt in self.prompts) != tuple(range(len(self.prompts))):
            raise ValueError("prompt ranks must be contiguous and zero based")

    @property
    def boxes_xyxy(self) -> np.ndarray:
        """Return all selected boxes as an immutable ``[K,4]`` float32 array."""

        if not self.prompts:
            return _readonly(np.empty((0, 4)), np.float32, (0, 4))
        return _readonly(
            np.stack([prompt.box_xyxy for prompt in self.prompts]),
            np.float32,
            (len(self.prompts), 4),
        )


def _validate_depth(value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError("depth_m must be a numeric numpy array with shape [H,W]")
    if value.ndim != 2 or min(value.shape, default=0) < 1 or value.dtype.kind not in "iuf":
        raise ValueError("depth_m must be a numeric numpy array with shape [H,W]")
    if int(value.shape[0]) * int(value.shape[1]) > _F_MAX_INPUT_DEPTH_PIXELS:
        raise ValueError("depth_m exceeds the fixed pixel cap")
    with np.errstate(over="ignore", invalid="ignore"):
        return np.array(value, dtype=np.float64, order="C", copy=True)


def _validate_boxes(value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError("explained_boxes_xyxy must be a numeric numpy array with shape [N,4]")
    if value.ndim != 2 or value.shape[1:] != (4,) or value.dtype.kind not in "iuf":
        raise ValueError("explained_boxes_xyxy must be a numeric numpy array with shape [N,4]")
    if len(value) > _F_MAX_EXPLAINED_BOXES:
        raise ValueError("explained_boxes_xyxy exceeds the fixed box cap")
    boxes = np.array(value, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(boxes).all():
        raise ValueError("explained_boxes_xyxy must contain finite values")
    if len(boxes) and (np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1])):
        raise ValueError("each explained box must have x2>x1 and y2>y1")
    return boxes


def _validate_intrinsics(value: object, depth_shape: tuple[int, int]) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("intrinsics must have finite shape [3,3] or [4,4]") from error
    if matrix.shape == (4, 4):
        matrix = matrix[:3, :3]
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("intrinsics must have finite shape [3,3] or [4,4]")
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0 or abs(float(np.linalg.det(matrix))) <= 1e-12:
        raise ValueError("intrinsics must be invertible with positive focal lengths")
    height, width = depth_shape
    if not (0.0 <= matrix[0, 2] < width and 0.0 <= matrix[1, 2] < height):
        raise ValueError("intrinsics principal point must lie inside depth_m")
    return np.array(matrix, dtype=np.float64, order="C", copy=True)


def _validate_pose(value: object) -> np.ndarray:
    try:
        pose = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("camera_to_world must be a finite rigid [4,4] matrix") from error
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("camera_to_world must be a finite rigid [4,4] matrix")
    if np.max(np.abs(pose[3] - [0.0, 0.0, 0.0, 1.0])) > 1e-7:
        raise ValueError("camera_to_world must be a finite rigid [4,4] matrix")
    rotation = pose[:3, :3]
    if (
        np.max(np.abs(rotation.T @ rotation - np.eye(3))) > 1e-4
        or abs(float(np.linalg.det(rotation)) - 1.0) > 1e-4
    ):
        raise ValueError("camera_to_world rotation must be orthonormal and right handed")
    return np.array(pose, dtype=np.float64, order="C", copy=True)


def _full_resolution_edge_mask(depth: np.ndarray) -> np.ndarray:
    usable = (
        np.isfinite(depth)
        & (depth >= _F_MIN_DEPTH_M)
        & (depth <= _F_MAX_DEPTH_M)
    )
    edge = np.zeros(depth.shape, dtype=bool)
    horizontal = usable[:, :-1] & usable[:, 1:] & (
        np.abs(depth[:, :-1] - depth[:, 1:]) > _F_DEPTH_EDGE_JUMP_M
    )
    vertical = usable[:-1, :] & usable[1:, :] & (
        np.abs(depth[:-1, :] - depth[1:, :]) > _F_DEPTH_EDGE_JUMP_M
    )
    edge[:, :-1] |= horizontal
    edge[:, 1:] |= horizontal
    edge[:-1, :] |= vertical
    edge[1:, :] |= vertical
    # A raw discontinuity can fall between two stride-four sample centres,
    # leaving neither sampled endpoint marked by the one-pixel mask.  Dilating
    # the full-resolution endpoint mask by the fixed 7 x 7 footprint creates a
    # phase-independent barrier before downsampling.
    return cv2.dilate(
        edge.astype(np.uint8, copy=False),
        np.ones((7, 7), dtype=np.uint8),
        iterations=1,
    ).astype(bool, copy=False)


def _stride_grid_edge_mask(sampled_depth: np.ndarray) -> np.ndarray:
    """Mark both endpoints of metric jumps between stride-grid neighbours."""

    usable = (
        np.isfinite(sampled_depth)
        & (sampled_depth >= _F_MIN_DEPTH_M)
        & (sampled_depth <= _F_MAX_DEPTH_M)
    )
    edge = np.zeros(sampled_depth.shape, dtype=bool)
    horizontal = usable[:, :-1] & usable[:, 1:] & (
        np.abs(sampled_depth[:, :-1] - sampled_depth[:, 1:])
        > _F_DEPTH_EDGE_JUMP_M
    )
    vertical = usable[:-1, :] & usable[1:, :] & (
        np.abs(sampled_depth[:-1, :] - sampled_depth[1:, :])
        > _F_DEPTH_EDGE_JUMP_M
    )
    edge[:, :-1] |= horizontal
    edge[:, 1:] |= horizontal
    edge[:-1, :] |= vertical
    edge[1:, :] |= vertical
    return edge


def _explained_grid_mask(
    rows: np.ndarray,
    cols: np.ndarray,
    boxes: np.ndarray,
) -> np.ndarray:
    explained = np.zeros((len(rows), len(cols)), dtype=bool)
    if not len(boxes):
        return explained
    for x1, y1, x2, y2 in boxes:
        # searchsorted implements the declared inclusive test without forming
        # an [N_boxes,H_grid,W_grid] temporary or scanning the whole grid once
        # per box.
        col_begin = int(
            np.searchsorted(
                cols, x1 - _F_EXPLAINED_BOX_EXPAND_PX, side="left"
            )
        )
        col_end = int(
            np.searchsorted(
                cols, x2 + _F_EXPLAINED_BOX_EXPAND_PX, side="right"
            )
        )
        row_begin = int(
            np.searchsorted(
                rows, y1 - _F_EXPLAINED_BOX_EXPAND_PX, side="left"
            )
        )
        row_end = int(
            np.searchsorted(
                rows, y2 + _F_EXPLAINED_BOX_EXPAND_PX, side="right"
            )
        )
        explained[row_begin:row_end, col_begin:col_end] = True
    return explained


def _image_box_from_grid_stats(
    left: int,
    top: int,
    width: int,
    height: int,
    image_shape: tuple[int, int],
) -> np.ndarray:
    image_height, image_width = image_shape
    return np.asarray(
        [
            left * _F_PIXEL_STRIDE,
            top * _F_PIXEL_STRIDE,
            min(image_width - 1, (left + width) * _F_PIXEL_STRIDE - 1),
            min(image_height - 1, (top + height) * _F_PIXEL_STRIDE - 1),
        ],
        dtype=np.float32,
    )


def _component_world_points(
    grid_rows: np.ndarray,
    grid_cols: np.ndarray,
    sampled_depth: np.ndarray,
    intrinsics_inverse: np.ndarray,
    camera_to_world: np.ndarray,
) -> np.ndarray:
    source_y = grid_rows.astype(np.float64) * _F_PIXEL_STRIDE
    source_x = grid_cols.astype(np.float64) * _F_PIXEL_STRIDE
    pixels = np.column_stack((source_x, source_y, np.ones(len(source_x))))
    rays = pixels @ intrinsics_inverse.T
    rays /= rays[:, 2:3]
    points_camera = rays * sampled_depth[grid_rows, grid_cols, None]
    return points_camera @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]


def _signed_floor_voxels(points_world: np.ndarray) -> np.ndarray:
    """Return deterministic unique 5 cm keys under signed half-open bins."""

    scaled = points_world / _F_VOXEL_SIZE_METERS
    if (
        not np.isfinite(scaled).all()
        or np.max(np.abs(scaled), initial=0.0) > np.iinfo(np.int64).max / 4
    ):
        raise ValueError("world point range is unsafe for 5 cm voxel quantization")
    return np.unique(np.floor(scaled).astype(np.int64), axis=0)


def _pixel_filter_reason(
    *,
    grid_pixel_count: int,
    bbox_width: int,
    bbox_height: int,
    touching_borders: int,
) -> Optional[str]:
    if grid_pixel_count < _F_MIN_COMPONENT_GRID_PIXELS:
        return "too_few_grid_pixels"
    if grid_pixel_count > _F_MAX_COMPONENT_GRID_PIXELS:
        return "too_many_grid_pixels"
    if bbox_width < _F_MIN_BBOX_GRID_SIZE or bbox_height < _F_MIN_BBOX_GRID_SIZE:
        return "bbox_too_small"
    if touching_borders > _F_MAX_TOUCHING_BORDERS:
        return "touches_multiple_borders"
    return None


def _metric_filter_reason(
    *,
    world_extent: np.ndarray,
    world_diagonal_m: float,
    world_volume_m3: float,
) -> Optional[str]:
    if np.any(world_extent < _F_MIN_WORLD_EXTENT_M):
        return "world_extent_too_small"
    if np.any(world_extent > _F_MAX_WORLD_EXTENT_M):
        return "world_extent_too_large"
    if world_diagonal_m > _F_MAX_WORLD_DIAGONAL_M:
        return "world_diagonal_too_large"
    if world_volume_m3 > _F_MAX_WORLD_VOLUME_M3:
        return "world_volume_too_large"
    return None


def generate_residual_box_prompts(
    *,
    depth_m: np.ndarray,
    explained_boxes_xyxy: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> UDCResult:
    """Generate at most two fixed-policy unexplained-depth MobileSAM prompts.

    The call is stateless and current-frame only.  Invalid input structure
    raises :class:`ValueError`; ordinary absence or rejection of residual
    components returns an empty result with complete diagnostics.
    """

    depth = _validate_depth(depth_m)
    boxes = _validate_boxes(explained_boxes_xyxy)
    depth_shape = (int(depth.shape[0]), int(depth.shape[1]))
    intrinsic_matrix = _validate_intrinsics(intrinsics, depth_shape)
    pose = _validate_pose(camera_to_world)

    rows = np.arange(0, depth_shape[0], _F_PIXEL_STRIDE, dtype=np.int64)
    cols = np.arange(0, depth_shape[1], _F_PIXEL_STRIDE, dtype=np.int64)
    sampled_depth = depth[np.ix_(rows, cols)]
    valid = (
        np.isfinite(sampled_depth)
        & (sampled_depth >= _F_MIN_DEPTH_M)
        & (sampled_depth <= _F_MAX_DEPTH_M)
    )
    sampled_edge = _full_resolution_edge_mask(depth)[np.ix_(rows, cols)]
    sampled_edge |= _stride_grid_edge_mask(sampled_depth)
    cleaned = valid & ~sampled_edge
    explained = _explained_grid_mask(rows, cols, boxes)
    residual = np.ascontiguousarray(cleaned & ~explained, dtype=np.uint8)

    component_total, labels, stats, _ = cv2.connectedComponentsWithStats(
        residual,
        connectivity=8,
        ltype=cv2.CV_32S,
    )
    grid_height, grid_width = residual.shape
    inverse_intrinsics = np.linalg.inv(intrinsic_matrix)
    components: list[UDCComponentDiagnostic] = []
    # Row-aligned private payloads are retained only for geometrically eligible
    # components.  Voxelization itself is delayed until after Top-2 selection.
    # This keeps isolated/noisy residual pixels inexpensive in live use.
    component_payloads: list[Optional[tuple[np.ndarray, np.ndarray]]] = []

    for component_id in range(1, int(component_total)):
        left = int(stats[component_id, cv2.CC_STAT_LEFT])
        top = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        count = int(stats[component_id, cv2.CC_STAT_AREA])
        touching_borders = int(left == 0) + int(top == 0) + int(
            left + width == grid_width
        ) + int(top + height == grid_height)
        reason = _pixel_filter_reason(
            grid_pixel_count=count,
            bbox_width=width,
            bbox_height=height,
            touching_borders=touching_borders,
        )
        depth_q02_q98 = np.full(2, np.nan, dtype=np.float64)
        world_q02 = np.full(3, np.nan, dtype=np.float64)
        world_q98 = np.full(3, np.nan, dtype=np.float64)
        world_extent = np.full(3, np.nan, dtype=np.float64)
        world_diagonal = float("nan")
        world_volume = float("nan")
        payload: Optional[tuple[np.ndarray, np.ndarray]] = None

        if reason is None:
            local_rows, local_cols = np.nonzero(
                labels[top : top + height, left : left + width] == component_id
            )
            component_rows = local_rows + top
            component_cols = local_cols + left
            points_world = _component_world_points(
                component_rows,
                component_cols,
                sampled_depth,
                inverse_intrinsics,
                pose,
            )
            source_pixels_yx = np.column_stack(
                (
                    component_rows * _F_PIXEL_STRIDE,
                    component_cols * _F_PIXEL_STRIDE,
                )
            ).astype(np.int64, copy=False)
            world_q02, world_q98 = np.quantile(
                points_world,
                [_F_WORLD_QUANTILE_LOW, _F_WORLD_QUANTILE_HIGH],
                axis=0,
            )
            world_extent = world_q98 - world_q02
            world_diagonal = float(np.linalg.norm(world_extent))
            world_volume = float(np.prod(world_extent))
            depth_q02_q98 = np.quantile(
                sampled_depth[component_rows, component_cols],
                [_F_WORLD_QUANTILE_LOW, _F_WORLD_QUANTILE_HIGH],
            )
            reason = _metric_filter_reason(
                world_extent=world_extent,
                world_diagonal_m=world_diagonal,
                world_volume_m3=world_volume,
            )
            if reason is None:
                payload = (source_pixels_yx, points_world)
        components.append(
            UDCComponentDiagnostic(
                component_id=component_id,
                grid_pixel_count=count,
                grid_bbox_xywh=np.asarray([left, top, width, height]),
                box_xyxy=_image_box_from_grid_stats(
                    left, top, width, height, depth_shape
                ),
                touching_borders=touching_borders,
                depth_q02_q98_m=depth_q02_q98,
                world_q02=world_q02,
                world_q98=world_q98,
                world_extent=world_extent,
                world_diagonal_m=world_diagonal,
                world_volume_m3=world_volume,
                eligible=reason is None,
                selected=False,
                reason=reason or "eligible",
            )
        )
        component_payloads.append(payload)

    eligible_positions = [index for index, item in enumerate(components) if item.eligible]
    eligible_positions.sort(
        key=lambda index: (
            -components[index].grid_pixel_count,
            int(components[index].grid_bbox_xywh[1]),
            int(components[index].grid_bbox_xywh[0]),
            int(components[index].grid_bbox_xywh[3]),
            int(components[index].grid_bbox_xywh[2]),
            components[index].component_id,
        )
    )
    selected_positions = eligible_positions[:_F_TOP_K]
    selected_set = set(selected_positions)
    for index in eligible_positions:
        if index in selected_set:
            components[index] = replace(components[index], selected=True, reason="selected")
        else:
            components[index] = replace(components[index], reason="top_k_cap")

    prompts_list = []
    for rank, index in enumerate(selected_positions):
        payload = component_payloads[index]
        if payload is None:  # Defensive invariant; never an ordinary abstention.
            raise RuntimeError("eligible UDC component is missing its geometry payload")
        source_pixels_yx, points_world = payload
        prompts_list.append(
            UDCPrompt(
                rank=rank,
                component_id=components[index].component_id,
                box_xyxy=components[index].box_xyxy,
                grid_pixel_count=components[index].grid_pixel_count,
                source_pixels_yx=source_pixels_yx,
                voxel_keys=_signed_floor_voxels(points_world),
                world_q02=components[index].world_q02,
                world_q98=components[index].world_q98,
                world_extent=components[index].world_extent,
                world_diagonal_m=components[index].world_diagonal_m,
                world_volume_m3=components[index].world_volume_m3,
            )
        )
    prompts = tuple(prompts_list)
    reason_counts = Counter(item.reason for item in components)
    frame_diagnostics = UDCFrameDiagnostics(
        schema=SCHEMA,
        depth_shape=depth_shape,
        grid_shape=(grid_height, grid_width),
        input_explained_box_count=len(boxes),
        sampled_grid_pixels=int(residual.size),
        valid_depth_grid_pixels=int(np.count_nonzero(valid)),
        edge_rejected_grid_pixels=int(np.count_nonzero(valid & sampled_edge)),
        explained_valid_grid_pixels=int(np.count_nonzero(cleaned & explained)),
        residual_grid_pixels=int(np.count_nonzero(residual)),
        component_count=len(components),
        eligible_component_count=len(eligible_positions),
        selected_component_count=len(prompts),
        rejection_counts=reason_counts,
    )
    return UDCResult(
        prompts=prompts,
        components=tuple(components),
        diagnostics=frame_diagnostics,
    )


# Concise integration alias used by the upcoming full100 shadow runner.
generate_udc_prompts = generate_residual_box_prompts


__all__ = [
    "SCHEMA",
    "POLICY",
    "UDCPrompt",
    "UDCComponentDiagnostic",
    "UDCFrameDiagnostics",
    "UDCResult",
    "VOXEL_SIZE_METERS",
    "generate_residual_box_prompts",
    "generate_udc_prompts",
]
