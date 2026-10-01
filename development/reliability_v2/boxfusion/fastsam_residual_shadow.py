"""Frozen FastSAM residual-mask shadow geometry.

This module is the provider-independent, training-free core of the F0
experiment.  A caller supplies binary automatic masks produced by a *frozen*
FastSAM model, current-frame CuTR boxes, registered metric depth, intrinsics,
and the current camera pose.  The implementation only measures, filters,
deduplicates, and lifts masks.  It never emits a birth, changes native output,
uses history, or consumes labels, semantics, CLIP features, or ground truth.

Coordinate contract
-------------------
Masks and depth are exactly 480 x 640.  Mask pixels and CuTR ``xyxy`` box
coordinates share that image coordinate system; box endpoints are inclusive.
CuTR boxes are expanded by four pixels before their union is removed for the
residual *eligibility* tests.  A selected mask is nevertheless lifted from its
full valid support, not from only the residual pixels.

All thresholds and tie breaks are fixed below.  Structural input errors raise
``ValueError`` (fail closed).  Ordinary mask rejection is represented in the
returned immutable diagnostics.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from types import MappingProxyType
from typing import Mapping, Optional

import cv2
import numpy as np


SCHEMA = "boxfusion.fastsam_residual_shadow.f0.v1"

# Public audit constants.  Executable policy uses private constants so that
# rebinding a public module attribute cannot alter a sealed run.
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
MIN_MASK_PIXELS = 200
MAX_MASK_PIXELS = 122_880
MIN_TIGHT_BOX_SIDE_PX = 16
MAX_TIGHT_BOX_ASPECT = 6.0
MIN_VALID_DEPTH_RATIO = 0.50
MIN_RESIDUAL_PIXELS = 200
MIN_RESIDUAL_RATIO = 0.20
EXPLAINED_BOX_EXPAND_PX = 4.0
DEDUP_MASK_IOU = 0.80
DEDUP_SMALLER_CONTAINMENT = 0.90
TOP_K = 16
MIN_DEPTH_M = 0.10
MAX_DEPTH_M = 6.0
MASK_EDGE_MARGIN_PX = 1
DEPTH_EDGE_JUMP_M = 0.15
VOXEL_SIZE_M = 0.02
MIN_UNIQUE_VOXELS = 16
MAX_STORED_POINTS = 2_048
WORLD_QUANTILES = (0.02, 0.98)
MIN_WORLD_AABB_EXTENT_M = 0.02

_F_IMAGE_HEIGHT = 480
_F_IMAGE_WIDTH = 640
_F_MIN_MASK_PIXELS = 200
_F_MAX_MASK_PIXELS = 122_880
_F_MIN_TIGHT_BOX_SIDE_PX = 16
_F_MAX_TIGHT_BOX_ASPECT = 6.0
_F_MIN_VALID_DEPTH_RATIO = 0.50
_F_MIN_RESIDUAL_PIXELS = 200
_F_MIN_RESIDUAL_RATIO = 0.20
_F_EXPLAINED_BOX_EXPAND_PX = 4.0
_F_DEDUP_MASK_IOU = 0.80
_F_DEDUP_SMALLER_CONTAINMENT = 0.90
_F_TOP_K = 16
_F_MIN_DEPTH_M = 0.10
_F_MAX_DEPTH_M = 6.0
_F_MASK_EDGE_MARGIN_PX = 1
_F_DEPTH_EDGE_JUMP_M = 0.15
_F_VOXEL_SIZE_M = 0.02
_F_MIN_UNIQUE_VOXELS = 16
_F_MAX_STORED_POINTS = 2_048
_F_WORLD_QUANTILE_LOW = 0.02
_F_WORLD_QUANTILE_HIGH = 0.98
_F_MIN_WORLD_AABB_EXTENT_M = 0.02
_F_MAX_INPUT_MASKS = 512
_F_MAX_EXPLAINED_BOXES = 4_096

POLICY: Mapping[str, object] = MappingProxyType(
    {
        "mask_shape": (_F_IMAGE_HEIGHT, _F_IMAGE_WIDTH),
        "mask_pixels": (_F_MIN_MASK_PIXELS, _F_MAX_MASK_PIXELS),
        "min_tight_box_side_px": _F_MIN_TIGHT_BOX_SIDE_PX,
        "max_tight_box_aspect": _F_MAX_TIGHT_BOX_ASPECT,
        "min_valid_depth_ratio": _F_MIN_VALID_DEPTH_RATIO,
        "min_residual_pixels": _F_MIN_RESIDUAL_PIXELS,
        "min_residual_ratio": _F_MIN_RESIDUAL_RATIO,
        "explained_box_expand_px": _F_EXPLAINED_BOX_EXPAND_PX,
        "sort": (
            "-confidence",
            "-residual_ratio",
            "-residual_pixels",
            "tight_box_xyxy",
            "mask_sha256",
        ),
        "dedup_mask_iou": _F_DEDUP_MASK_IOU,
        "dedup_smaller_containment": _F_DEDUP_SMALLER_CONTAINMENT,
        "top_k": _F_TOP_K,
        "depth_m": (_F_MIN_DEPTH_M, _F_MAX_DEPTH_M),
        "mask_edge_margin_px": _F_MASK_EDGE_MARGIN_PX,
        "depth_edge_connectivity": 4,
        "depth_edge_jump_m": _F_DEPTH_EDGE_JUMP_M,
        "voxel_size_m": _F_VOXEL_SIZE_M,
        "min_unique_voxels": _F_MIN_UNIQUE_VOXELS,
        "max_stored_points": _F_MAX_STORED_POINTS,
        "world_quantiles": (_F_WORLD_QUANTILE_LOW, _F_WORLD_QUANTILE_HIGH),
        "min_world_aabb_extent_m": _F_MIN_WORLD_AABB_EXTENT_M,
        "lift_support": "full_mask_not_residual",
    }
)


def _readonly(
    value: object,
    dtype: np.dtype,
    shape: Optional[tuple[int, ...]] = None,
) -> np.ndarray:
    """Return a detached NumPy view backed by immutable bytes."""

    array = np.asarray(value, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError(f"array must have shape {shape}, got {array.shape}")
    packed = np.ascontiguousarray(array).tobytes()
    return np.frombuffer(packed, dtype=dtype).reshape(array.shape)


@dataclass(frozen=True)
class FastSAMResidualCandidate:
    """One selected shadow candidate with bounded, auditable 3D geometry."""

    raw_index: int
    rank: int
    confidence: float
    mask_sha256: str
    tight_box_xyxy: np.ndarray
    pixel_count: int
    valid_pixel_count: int
    residual_pixel_count: int
    residual_ratio: float
    valid_ratio: float
    support_pixel_count: int
    voxel_count: int
    points_world: np.ndarray
    voxel_keys: np.ndarray
    points_sha256: str
    world_q02: np.ndarray
    world_q98: np.ndarray
    world_center: np.ndarray
    world_extent: np.ndarray

    def __post_init__(self) -> None:
        if self.raw_index < 0 or self.rank < 0 or self.rank >= _F_TOP_K:
            raise ValueError("candidate indices are outside the frozen policy")
        if self.pixel_count < 1 or self.voxel_count < _F_MIN_UNIQUE_VOXELS:
            raise ValueError("selected candidate has invalid support counts")
        point_count = int(np.asarray(self.points_world).shape[0])
        if not (1 <= point_count <= _F_MAX_STORED_POINTS):
            raise ValueError("stored point count is outside the frozen cap")
        object.__setattr__(
            self, "tight_box_xyxy", _readonly(self.tight_box_xyxy, np.int64, (4,))
        )
        object.__setattr__(
            self,
            "points_world",
            _readonly(self.points_world, np.float64, (point_count, 3)),
        )
        object.__setattr__(
            self, "voxel_keys", _readonly(self.voxel_keys, np.int64, (point_count, 3))
        )
        object.__setattr__(self, "world_q02", _readonly(self.world_q02, np.float64, (3,)))
        object.__setattr__(self, "world_q98", _readonly(self.world_q98, np.float64, (3,)))
        object.__setattr__(
            self, "world_center", _readonly(self.world_center, np.float64, (3,))
        )
        object.__setattr__(
            self, "world_extent", _readonly(self.world_extent, np.float64, (3,))
        )
        if np.any(self.world_extent < _F_MIN_WORLD_AABB_EXTENT_M - 1e-12):
            raise ValueError("world AABB violates the frozen minimum extent")

    @property
    def stored_point_count(self) -> int:
        return int(self.points_world.shape[0])

    @property
    def tight_box(self) -> np.ndarray:
        """Compatibility alias used by JSON sidecar writers."""

        return self.tight_box_xyxy


@dataclass(frozen=True)
class FastSAMMaskDiagnostic:
    """Disposition and measurements for one input automatic mask."""

    raw_index: int
    confidence: float
    mask_sha256: str
    tight_box_xyxy: np.ndarray
    pixel_count: int
    valid_pixel_count: int
    residual_pixel_count: int
    residual_ratio: float
    valid_ratio: float
    support_pixel_count: int
    voxel_count: int
    pre_dedup_eligible: bool
    deduplicated: bool
    lifted: bool
    selected: bool
    rank: Optional[int]
    duplicate_of_raw_index: Optional[int]
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tight_box_xyxy", _readonly(self.tight_box_xyxy, np.int64, (4,))
        )
        if self.selected and (not self.lifted or self.rank is None):
            raise ValueError("a selected diagnostic must be lifted and ranked")
        if self.deduplicated and self.duplicate_of_raw_index is None:
            raise ValueError("a deduplicated mask must identify its representative")


@dataclass(frozen=True)
class FastSAMFrameDiagnostics:
    """Frame-level accounting without timing or output-affecting state."""

    schema: str
    mask_shape: tuple[int, int]
    input_mask_count: int
    input_explained_box_count: int
    explained_union_pixels: int
    pre_dedup_eligible_count: int
    deduplicated_count: int
    post_dedup_count: int
    lifting_eligible_count: int
    selected_count: int
    cap_rejected_count: int
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
class FastSAMResidualResult:
    """Immutable selected candidates and complete per-input diagnostics."""

    candidates: tuple[FastSAMResidualCandidate, ...]
    masks: tuple[FastSAMMaskDiagnostic, ...]
    diagnostics: FastSAMFrameDiagnostics

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "masks", tuple(self.masks))
        if len(self.candidates) > _F_TOP_K:
            raise ValueError("result exceeds the frozen Top-K")
        if tuple(item.rank for item in self.candidates) != tuple(
            range(len(self.candidates))
        ):
            raise ValueError("candidate ranks must be contiguous and zero based")


def _validate_masks(value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError("masks must be a binary numpy array with shape [N,480,640]")
    if value.ndim != 3 or value.shape[1:] != (_F_IMAGE_HEIGHT, _F_IMAGE_WIDTH):
        raise ValueError("masks must have shape [N,480,640]")
    if len(value) > _F_MAX_INPUT_MASKS or value.dtype.kind not in "biuf":
        raise ValueError("masks exceed the fixed cap or are not numeric")
    try:
        array = np.array(value, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("masks must be finite and exactly binary") from error
    if array.dtype.kind in "uf" and not np.isfinite(array).all():
        raise ValueError("masks must be finite and exactly binary")
    if np.any((array != 0) & (array != 1)):
        raise ValueError("masks must be finite and exactly binary")
    return array.astype(bool, copy=False)


def _validate_confidences(value: object, count: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError("confidences must be a finite numpy array with shape [N]")
    if value.ndim != 1 or value.shape != (count,) or value.dtype.kind not in "iuf":
        raise ValueError("confidences must be a finite numpy array with shape [N]")
    confidences = np.array(value, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(confidences).all() or np.any(confidences < 0.0) or np.any(
        confidences > 1.0
    ):
        raise ValueError("confidences must be finite values in [0,1]")
    return confidences


def _validate_depth(value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError("depth_m must be a numeric numpy array with shape [480,640]")
    if value.shape != (_F_IMAGE_HEIGHT, _F_IMAGE_WIDTH) or value.dtype.kind not in "iuf":
        raise ValueError("depth_m must be a numeric numpy array with shape [480,640]")
    with np.errstate(over="ignore", invalid="ignore"):
        return np.array(value, dtype=np.float64, order="C", copy=True)


def _validate_boxes(value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError("explained_boxes_xyxy must be a numpy array with shape [N,4]")
    if value.ndim != 2 or value.shape[1:] != (4,) or value.dtype.kind not in "iuf":
        raise ValueError("explained_boxes_xyxy must be a numpy array with shape [N,4]")
    if len(value) > _F_MAX_EXPLAINED_BOXES:
        raise ValueError("explained_boxes_xyxy exceeds the fixed box cap")
    boxes = np.array(value, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(boxes).all():
        raise ValueError("explained_boxes_xyxy must contain finite values")
    if len(boxes) and (
        np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1])
    ):
        raise ValueError("each explained box must have x2>x1 and y2>y1")
    return boxes


def _validate_intrinsics(value: object) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("intrinsics must have finite shape [3,3] or [4,4]") from error
    if matrix.shape == (4, 4):
        matrix = matrix[:3, :3]
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("intrinsics must have finite shape [3,3] or [4,4]")
    if (
        matrix[0, 0] <= 0.0
        or matrix[1, 1] <= 0.0
        or abs(float(np.linalg.det(matrix))) <= 1e-12
        or not (0.0 <= matrix[0, 2] < _F_IMAGE_WIDTH)
        or not (0.0 <= matrix[1, 2] < _F_IMAGE_HEIGHT)
    ):
        raise ValueError("intrinsics must be invertible and registered to 480x640")
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


def _explained_union(boxes: np.ndarray) -> np.ndarray:
    union = np.zeros((_F_IMAGE_HEIGHT, _F_IMAGE_WIDTH), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        left = max(0, int(np.ceil(x1 - _F_EXPLAINED_BOX_EXPAND_PX)))
        top = max(0, int(np.ceil(y1 - _F_EXPLAINED_BOX_EXPAND_PX)))
        right = min(
            _F_IMAGE_WIDTH - 1,
            int(np.floor(x2 + _F_EXPLAINED_BOX_EXPAND_PX)),
        )
        bottom = min(
            _F_IMAGE_HEIGHT - 1,
            int(np.floor(y2 + _F_EXPLAINED_BOX_EXPAND_PX)),
        )
        if left <= right and top <= bottom:
            union[top : bottom + 1, left : right + 1] = True
    return union


def _mask_sha256(mask: np.ndarray) -> str:
    packed = np.packbits(mask.reshape(-1), bitorder="little")
    return hashlib.sha256(packed.tobytes()).hexdigest()


def _tight_box(mask: np.ndarray) -> tuple[np.ndarray, int, int, float]:
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return np.full(4, -1, dtype=np.int64), 0, 0, float("inf")
    left = int(np.min(cols))
    top = int(np.min(rows))
    right = int(np.max(cols))
    bottom = int(np.max(rows))
    width = right - left + 1
    height = bottom - top + 1
    aspect = float(max(width, height) / min(width, height))
    return np.asarray([left, top, right, bottom], dtype=np.int64), width, height, aspect


def _depth_edge_mask(depth: np.ndarray, valid_depth: np.ndarray) -> np.ndarray:
    """Mark both endpoints of every valid four-neighbour jump over 0.15 m."""

    edge = np.zeros(depth.shape, dtype=bool)
    horizontal = valid_depth[:, :-1] & valid_depth[:, 1:] & (
        np.abs(depth[:, :-1] - depth[:, 1:]) > _F_DEPTH_EDGE_JUMP_M
    )
    vertical = valid_depth[:-1, :] & valid_depth[1:, :] & (
        np.abs(depth[:-1, :] - depth[1:, :]) > _F_DEPTH_EDGE_JUMP_M
    )
    edge[:, :-1] |= horizontal
    edge[:, 1:] |= horizontal
    edge[:-1, :] |= vertical
    edge[1:, :] |= vertical
    return edge


def _overlap_is_duplicate(
    mask: np.ndarray,
    area: int,
    box: np.ndarray,
    other_mask: np.ndarray,
    other_area: int,
    other_box: np.ndarray,
) -> bool:
    left = max(int(box[0]), int(other_box[0]))
    top = max(int(box[1]), int(other_box[1]))
    right = min(int(box[2]), int(other_box[2]))
    bottom = min(int(box[3]), int(other_box[3]))
    if left > right or top > bottom:
        return False
    intersection = int(
        np.count_nonzero(
            mask[top : bottom + 1, left : right + 1]
            & other_mask[top : bottom + 1, left : right + 1]
        )
    )
    if intersection == 0:
        return False
    union = area + other_area - intersection
    iou = intersection / union
    smaller_containment = intersection / min(area, other_area)
    return bool(
        iou >= _F_DEDUP_MASK_IOU
        or smaller_containment >= _F_DEDUP_SMALLER_CONTAINMENT
    )


def _lift_mask(
    *,
    mask: np.ndarray,
    depth: np.ndarray,
    valid_depth: np.ndarray,
    depth_edge: np.ndarray,
    inverse_intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> Optional[dict[str, object]]:
    # A 3 x 3 erosion removes exactly the one-pixel mask-boundary ring.
    interior = cv2.erode(
        mask.astype(np.uint8, copy=False),
        np.ones((3, 3), dtype=np.uint8),
        iterations=_F_MASK_EDGE_MARGIN_PX,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool, copy=False)
    support = interior & valid_depth & ~depth_edge
    rows, cols = np.nonzero(support)
    support_count = int(len(rows))
    if not support_count:
        return None

    pixels = np.column_stack(
        (cols.astype(np.float64), rows.astype(np.float64), np.ones(support_count))
    )
    rays = pixels @ inverse_intrinsics.T
    rays /= rays[:, 2:3]
    points_camera = rays * depth[rows, cols, None]
    points_world = (
        points_camera @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    )
    scaled = points_world / _F_VOXEL_SIZE_M
    if (
        not np.isfinite(scaled).all()
        or np.max(np.abs(scaled), initial=0.0) > np.iinfo(np.int64).max / 4
    ):
        raise ValueError("world point range is unsafe for 2 cm voxel quantization")
    keys = np.floor(scaled).astype(np.int64)
    unique_keys, first_indices = np.unique(keys, axis=0, return_index=True)
    voxel_count = int(len(unique_keys))
    if voxel_count < _F_MIN_UNIQUE_VOXELS:
        return {
            "support_pixel_count": support_count,
            "voxel_count": voxel_count,
        }

    representatives = points_world[first_indices]
    raw_q02, raw_q98 = np.quantile(
        representatives,
        [_F_WORLD_QUANTILE_LOW, _F_WORLD_QUANTILE_HIGH],
        axis=0,
    )
    center = (raw_q02 + raw_q98) * 0.5
    extent = np.maximum(raw_q98 - raw_q02, _F_MIN_WORLD_AABB_EXTENT_M)
    world_q02 = center - extent * 0.5
    world_q98 = center + extent * 0.5

    if voxel_count <= _F_MAX_STORED_POINTS:
        stored_indices = np.arange(voxel_count, dtype=np.int64)
    else:
        # Evenly cover the lexicographically sorted unique voxel sequence,
        # including both endpoints.  The integer formula is version-stable.
        stored_indices = (
            np.arange(_F_MAX_STORED_POINTS, dtype=np.int64) * (voxel_count - 1)
        ) // (_F_MAX_STORED_POINTS - 1)
    stored_points = representatives[stored_indices]
    stored_keys = unique_keys[stored_indices]
    digest = hashlib.sha256()
    digest.update(np.asarray(stored_points, dtype="<f8").tobytes())
    digest.update(np.asarray(stored_keys, dtype="<i8").tobytes())
    return {
        "support_pixel_count": support_count,
        "voxel_count": voxel_count,
        "points_world": stored_points,
        "voxel_keys": stored_keys,
        "points_sha256": digest.hexdigest(),
        "world_q02": world_q02,
        "world_q98": world_q98,
        "world_center": center,
        "world_extent": extent,
    }


def select_and_lift_residual_masks(
    *,
    masks: np.ndarray,
    confidences: np.ndarray,
    depth_m: np.ndarray,
    explained_boxes_xyxy: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> FastSAMResidualResult:
    """Apply the sealed F0 residual-mask policy to one current RGB-D frame.

    ``masks`` must already be binary automatic masks from the frozen provider;
    this core deliberately has no model or image threshold argument.  It is
    stateless and current-frame only.  It returns metadata/geometry sidecars
    and cannot modify a detector's native predictions.
    """

    mask_array = _validate_masks(masks)
    confidence_array = _validate_confidences(confidences, len(mask_array))
    depth = _validate_depth(depth_m)
    boxes = _validate_boxes(explained_boxes_xyxy)
    intrinsic_matrix = _validate_intrinsics(intrinsics)
    pose = _validate_pose(camera_to_world)

    valid_depth = (
        np.isfinite(depth)
        & (depth >= _F_MIN_DEPTH_M)
        & (depth <= _F_MAX_DEPTH_M)
    )
    explained = _explained_union(boxes)

    # Mutable private rows are converted to immutable diagnostics only after
    # deduplication, lifting, and Top-16 accounting are complete.
    rows: list[dict[str, object]] = []
    basic_eligible: list[int] = []
    for raw_index, (mask, confidence) in enumerate(zip(mask_array, confidence_array)):
        mask_hash = _mask_sha256(mask)
        pixel_count = int(np.count_nonzero(mask))
        box, width, height, aspect = _tight_box(mask)
        valid_support = mask & valid_depth
        valid_pixel_count = int(np.count_nonzero(valid_support))
        # Residual membership is defined only on valid metric support.  Invalid
        # depth pixels cannot make a mask appear unexplained.
        residual_pixel_count = int(np.count_nonzero(valid_support & ~explained))
        valid_ratio = valid_pixel_count / pixel_count if pixel_count else 0.0
        residual_ratio = (
            residual_pixel_count / valid_pixel_count if valid_pixel_count else 0.0
        )
        reason = "pre_dedup_eligible"
        if pixel_count < _F_MIN_MASK_PIXELS:
            reason = "area_too_small"
        elif pixel_count > _F_MAX_MASK_PIXELS:
            reason = "area_too_large"
        elif width < _F_MIN_TIGHT_BOX_SIDE_PX or height < _F_MIN_TIGHT_BOX_SIDE_PX:
            reason = "tight_box_side_too_small"
        elif aspect > _F_MAX_TIGHT_BOX_ASPECT:
            reason = "tight_box_aspect_too_large"
        elif valid_ratio < _F_MIN_VALID_DEPTH_RATIO:
            reason = "valid_depth_ratio_too_low"
        elif residual_pixel_count < _F_MIN_RESIDUAL_PIXELS:
            reason = "residual_pixels_too_few"
        elif residual_ratio < _F_MIN_RESIDUAL_RATIO:
            reason = "residual_ratio_too_low"

        row: dict[str, object] = {
            "raw_index": raw_index,
            "confidence": float(confidence),
            "mask_sha256": mask_hash,
            "tight_box_xyxy": box,
            "pixel_count": pixel_count,
            "valid_pixel_count": valid_pixel_count,
            "residual_pixel_count": residual_pixel_count,
            "residual_ratio": float(residual_ratio),
            "valid_ratio": float(valid_ratio),
            "support_pixel_count": 0,
            "voxel_count": 0,
            "pre_dedup_eligible": reason == "pre_dedup_eligible",
            "deduplicated": False,
            "lifted": False,
            "selected": False,
            "rank": None,
            "duplicate_of_raw_index": None,
            "reason": reason,
        }
        rows.append(row)
        if reason == "pre_dedup_eligible":
            basic_eligible.append(raw_index)

    basic_eligible.sort(
        key=lambda index: (
            -float(rows[index]["confidence"]),
            -float(rows[index]["residual_ratio"]),
            -int(rows[index]["residual_pixel_count"]),
            *tuple(int(item) for item in np.asarray(rows[index]["tight_box_xyxy"])),
            str(rows[index]["mask_sha256"]),
        )
    )

    unique_indices: list[int] = []
    for raw_index in basic_eligible:
        row = rows[raw_index]
        duplicate_of: Optional[int] = None
        for representative_index in unique_indices:
            representative = rows[representative_index]
            if _overlap_is_duplicate(
                mask_array[raw_index],
                int(row["pixel_count"]),
                np.asarray(row["tight_box_xyxy"]),
                mask_array[representative_index],
                int(representative["pixel_count"]),
                np.asarray(representative["tight_box_xyxy"]),
            ):
                duplicate_of = representative_index
                break
        if duplicate_of is not None:
            row["deduplicated"] = True
            row["duplicate_of_raw_index"] = duplicate_of
            row["reason"] = "duplicate"
        else:
            unique_indices.append(raw_index)

    depth_edge = _depth_edge_mask(depth, valid_depth) if unique_indices else np.zeros(
        depth.shape, dtype=bool
    )
    # The bounded online policy permits geometry work for at most the first
    # sixteen deduplicated masks.  Lower-ranked masks are accounted as cap
    # drops without expensive backprojection, and geometry failures do not
    # pull later masks across the frozen boundary.
    attempted_indices = unique_indices[:_F_TOP_K]
    for raw_index in unique_indices[_F_TOP_K:]:
        rows[raw_index]["reason"] = "top_k_cap"

    lifted_payloads: dict[int, dict[str, object]] = {}
    lifted_indices: list[int] = []
    inverse_intrinsics = np.linalg.inv(intrinsic_matrix)
    for raw_index in attempted_indices:
        payload = _lift_mask(
            mask=mask_array[raw_index],
            depth=depth,
            valid_depth=valid_depth,
            depth_edge=depth_edge,
            inverse_intrinsics=inverse_intrinsics,
            camera_to_world=pose,
        )
        if payload is None:
            rows[raw_index]["reason"] = "too_few_unique_voxels"
            continue
        rows[raw_index]["support_pixel_count"] = int(payload["support_pixel_count"])
        rows[raw_index]["voxel_count"] = int(payload["voxel_count"])
        if int(payload["voxel_count"]) < _F_MIN_UNIQUE_VOXELS:
            rows[raw_index]["reason"] = "too_few_unique_voxels"
            continue
        rows[raw_index]["lifted"] = True
        rows[raw_index]["reason"] = "lifted"
        lifted_payloads[raw_index] = payload
        lifted_indices.append(raw_index)

    selected_indices = lifted_indices
    for rank, raw_index in enumerate(selected_indices):
        rows[raw_index]["selected"] = True
        rows[raw_index]["rank"] = rank
        rows[raw_index]["reason"] = "selected"
    candidates: list[FastSAMResidualCandidate] = []
    for rank, raw_index in enumerate(selected_indices):
        row = rows[raw_index]
        payload = lifted_payloads[raw_index]
        candidates.append(
            FastSAMResidualCandidate(
                raw_index=raw_index,
                rank=rank,
                confidence=float(row["confidence"]),
                mask_sha256=str(row["mask_sha256"]),
                tight_box_xyxy=np.asarray(row["tight_box_xyxy"]),
                pixel_count=int(row["pixel_count"]),
                valid_pixel_count=int(row["valid_pixel_count"]),
                residual_pixel_count=int(row["residual_pixel_count"]),
                residual_ratio=float(row["residual_ratio"]),
                valid_ratio=float(row["valid_ratio"]),
                support_pixel_count=int(payload["support_pixel_count"]),
                voxel_count=int(payload["voxel_count"]),
                points_world=np.asarray(payload["points_world"]),
                voxel_keys=np.asarray(payload["voxel_keys"]),
                points_sha256=str(payload["points_sha256"]),
                world_q02=np.asarray(payload["world_q02"]),
                world_q98=np.asarray(payload["world_q98"]),
                world_center=np.asarray(payload["world_center"]),
                world_extent=np.asarray(payload["world_extent"]),
            )
        )

    diagnostics = tuple(
        FastSAMMaskDiagnostic(
            raw_index=int(row["raw_index"]),
            confidence=float(row["confidence"]),
            mask_sha256=str(row["mask_sha256"]),
            tight_box_xyxy=np.asarray(row["tight_box_xyxy"]),
            pixel_count=int(row["pixel_count"]),
            valid_pixel_count=int(row["valid_pixel_count"]),
            residual_pixel_count=int(row["residual_pixel_count"]),
            residual_ratio=float(row["residual_ratio"]),
            valid_ratio=float(row["valid_ratio"]),
            support_pixel_count=int(row["support_pixel_count"]),
            voxel_count=int(row["voxel_count"]),
            pre_dedup_eligible=bool(row["pre_dedup_eligible"]),
            deduplicated=bool(row["deduplicated"]),
            lifted=bool(row["lifted"]),
            selected=bool(row["selected"]),
            rank=None if row["rank"] is None else int(row["rank"]),
            duplicate_of_raw_index=(
                None
                if row["duplicate_of_raw_index"] is None
                else int(row["duplicate_of_raw_index"])
            ),
            reason=str(row["reason"]),
        )
        for row in rows
    )
    reason_counts = Counter(item.reason for item in diagnostics)
    frame = FastSAMFrameDiagnostics(
        schema=SCHEMA,
        mask_shape=(_F_IMAGE_HEIGHT, _F_IMAGE_WIDTH),
        input_mask_count=len(mask_array),
        input_explained_box_count=len(boxes),
        explained_union_pixels=int(np.count_nonzero(explained)),
        pre_dedup_eligible_count=len(basic_eligible),
        deduplicated_count=sum(item.deduplicated for item in diagnostics),
        post_dedup_count=len(unique_indices),
        lifting_eligible_count=len(lifted_indices),
        selected_count=len(candidates),
        cap_rejected_count=int(reason_counts.get("top_k_cap", 0)),
        rejection_counts=reason_counts,
    )
    return FastSAMResidualResult(
        candidates=tuple(candidates),
        masks=diagnostics,
        diagnostics=frame,
    )


# Concise integration aliases for the 200-scene shadow runner.
generate_fastsam_residual_shadow = select_and_lift_residual_masks
process_fastsam_residual_masks = select_and_lift_residual_masks


__all__ = [
    "SCHEMA",
    "POLICY",
    "FastSAMResidualCandidate",
    "FastSAMMaskDiagnostic",
    "FastSAMFrameDiagnostics",
    "FastSAMResidualResult",
    "select_and_lift_residual_masks",
    "generate_fastsam_residual_shadow",
    "process_fastsam_residual_masks",
]
