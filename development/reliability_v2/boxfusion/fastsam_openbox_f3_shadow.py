"""Causal OpenBox-lite projection shadow for sealed FastSAM H0 candidates.

F3 is deliberately an observer.  It associates the sealed, class-agnostic
H0 candidates using only committed world AABBs, retains bounded mask/pose/
5 cm voxel evidence, and evaluates two terminal geometry hypotheses:

``B``
    The retained single-view H0 AABB with the best strict leave-one-view-out
    projected-box/mask IoU.
``C``
    A world-axis q02/q98 AABB fitted to a two-view-supported 5 cm voxel
    consensus.  Every leave-one-view-out fold is fit without the held-out
    observation and scored only on that observation's raw FastSAM mask.

The module has no detector, RGB, depth-pixel, semantic, label, evaluator,
ground-truth, training, or prediction-mutation API.  Tracks are queried
against a snapshot of committed past state and become visible only after an
exact-token commit.  ``finalize`` merely seals the latest causal receipts; it
does not create a prediction or a birth.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import math
from numbers import Integral
import time
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


SCHEMA = "boxfusion.fastsam_openbox_f3_shadow.v1"
MODE = "shadow"
PROTOCOL_ID = "F3-FASTSAM-OPENBOX-PROJECTION-SHADOW-PAPER100"

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
MASK_PACKED_BYTES = IMAGE_HEIGHT * IMAGE_WIDTH // 8
VOXEL_SIZE_M = 0.05
MAX_VOXELS_PER_OBSERVATION = 512
MAX_CONSENSUS_VOXELS = 2_048
MIN_CONSENSUS_VOXELS = 16
MIN_AABB_EXTENT_M = 0.02
MATCH_AABB_IOU = 0.10
MATCH_CENTER_DISTANCE_M = 0.50
TTL_KEYFRAMES = 10
MAX_LIVE_TRACKS = 1_024
MAX_RETAINED_OBSERVATIONS = 5
MIN_DISTINCT_FRAMES = 3
NEAR_PLANE_M = 1e-3
B_MIN_PROJECTION_IOU = 0.10
C_MIN_STABILITY_IOU = 0.25
C_MIN_GAIN_OVER_B = 0.03
C_MAX_CENTER_SHIFT_M = 0.50
C_EXTENT_RATIO_RANGE = (0.5, 2.0)
C_VOLUME_RATIO_RANGE = (0.25, 4.0)

# Private execution literals prevent rebinding a public audit constant from
# changing the preregistered experiment.
_F_IMAGE_HEIGHT = 480
_F_IMAGE_WIDTH = 640
_F_MASK_PACKED_BYTES = 38_400
_F_VOXEL_SIZE_M = 0.05
_F_MAX_VOXELS_PER_OBSERVATION = 512
_F_MAX_CONSENSUS_VOXELS = 2_048
_F_MIN_CONSENSUS_VOXELS = 16
_F_MIN_AABB_EXTENT_M = 0.02
_F_MATCH_AABB_IOU = 0.10
_F_MATCH_CENTER_DISTANCE_M = 0.50
_F_TTL_KEYFRAMES = 10
_F_MAX_LIVE_TRACKS = 1_024
_F_MAX_RETAINED_OBSERVATIONS = 5
_F_MIN_DISTINCT_FRAMES = 3
_F_NEAR_PLANE_M = 1e-3
_F_B_MIN_PROJECTION_IOU = 0.10
_F_C_MIN_STABILITY_IOU = 0.25
_F_C_MIN_GAIN_OVER_B = 0.03
_F_C_MAX_CENTER_SHIFT_M = 0.50
_F_C_MIN_EXTENT_RATIO = 0.5
_F_C_MAX_EXTENT_RATIO = 2.0
_F_C_MIN_VOLUME_RATIO = 0.25
_F_C_MAX_VOLUME_RATIO = 4.0
_MAX_ID = (1 << 63) - 1
_SAFE_VOXEL_COORDINATE = (1 << 62) - 2
_FLOAT64_EXACT_INTEGER_SPAN = 1 << 52

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
_OFFSETS = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
)
_ROW_DTYPE = np.dtype([("x", "<i8"), ("y", "<i8"), ("z", "<i8")])
_BYTE_POPCOUNT = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None], axis=1
).sum(axis=1, dtype=np.uint8)

POLICY: Mapping[str, object] = MappingProxyType(
    {
        "protocol_id": PROTOCOL_ID,
        "input_hypothesis": "F1/H0_only",
        "observer_only": True,
        "birth": False,
        "native_mutation": False,
        "training": False,
        "online_learning": False,
        "ground_truth": False,
        "rgb": False,
        "depth_pixels": False,
        "semantics": False,
        "association": "committed_past_aabb_iou_and_center",
        "match_aabb_iou": _F_MATCH_AABB_IOU,
        "match_center_distance_m": _F_MATCH_CENTER_DISTANCE_M,
        "ttl_keyframes": _F_TTL_KEYFRAMES,
        "max_live_tracks": _F_MAX_LIVE_TRACKS,
        "max_retained_observations": _F_MAX_RETAINED_OBSERVATIONS,
        "minimum_distinct_frames": _F_MIN_DISTINCT_FRAMES,
        "voxel_size_m": _F_VOXEL_SIZE_M,
        "max_voxels_per_observation": _F_MAX_VOXELS_PER_OBSERVATION,
        "max_consensus_voxels": _F_MAX_CONSENSUS_VOXELS,
        "min_consensus_voxels": _F_MIN_CONSENSUS_VOXELS,
        "consensus_support": "at_least_two_distinct_views_chebyshev_radius_one",
        "world_quantiles": (0.02, 0.98),
        "min_aabb_extent_m": _F_MIN_AABB_EXTENT_M,
        "near_plane_m": _F_NEAR_PLANE_M,
        "b_min_projection_iou": _F_B_MIN_PROJECTION_IOU,
        "c_min_stability_iou": _F_C_MIN_STABILITY_IOU,
        "c_min_gain_over_b": _F_C_MIN_GAIN_OVER_B,
        "c_max_center_shift_m": _F_C_MAX_CENTER_SHIFT_M,
        "c_extent_ratio_range": (
            _F_C_MIN_EXTENT_RATIO,
            _F_C_MAX_EXTENT_RATIO,
        ),
        "c_volume_ratio_range": (
            _F_C_MIN_VOLUME_RATIO,
            _F_C_MAX_VOLUME_RATIO,
        ),
        "projection_image_shape": (_F_IMAGE_HEIGHT, _F_IMAGE_WIDTH),
        "mask_bitorder": "little",
        "query_before_commit": True,
    }
)


def _strict_int(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < 0 or result > _MAX_ID:
        raise ValueError(f"{name} must lie in [0, {_MAX_ID}]")
    return result


def _source_id(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("source_id must be a nonempty stripped string")
    if any(ord(char) < 32 for char in value):
        raise ValueError("source_id must not contain control characters")
    return value


def _finite_float(name: str, value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _readonly(
    value: object,
    dtype: np.dtype,
    shape: Optional[tuple[int, ...]] = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError(f"array must have shape {shape}, got {array.shape}")
    packed = np.ascontiguousarray(array).tobytes()
    return np.frombuffer(packed, dtype=dtype).reshape(array.shape)


def _bounds(
    world_q02: object,
    world_q98: object,
    *,
    require_min_extent: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        lower = np.asarray(world_q02, dtype=np.float64)
        upper = np.asarray(world_q98, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("world_q02/world_q98 must be finite shape-[3] arrays") from error
    if (
        lower.shape != (3,)
        or upper.shape != (3,)
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
    ):
        raise ValueError("world_q02/world_q98 must be finite shape-[3] arrays")
    extent = upper - lower
    minimum = _F_MIN_AABB_EXTENT_M if require_min_extent else 0.0
    if np.any(extent < minimum - 1e-12):
        raise ValueError("AABB extent violates the frozen minimum")
    return (
        np.array(lower, dtype=np.float64, order="C", copy=True),
        np.array(upper, dtype=np.float64, order="C", copy=True),
    )


def _intrinsics(value: object) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("intrinsics must be finite [3,3] or [4,4]") from error
    if matrix.shape == (4, 4):
        matrix = matrix[:3, :3]
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("intrinsics must be finite [3,3] or [4,4]")
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise ValueError("intrinsics focal lengths must be positive")
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("intrinsics must be invertible") from error
    if not np.isfinite(inverse).all():
        raise ValueError("intrinsics inverse must be finite")
    return np.array(matrix, dtype=np.float64, order="C", copy=True)


def _pose_pair(value: object) -> tuple[np.ndarray, np.ndarray]:
    try:
        pose = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("camera_to_world must be a finite affine [4,4] matrix") from error
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("camera_to_world must be a finite affine [4,4] matrix")
    if not np.allclose(
        pose[3], np.asarray([0.0, 0.0, 0.0, 1.0]), rtol=0.0, atol=1e-7
    ):
        raise ValueError("camera_to_world must be a finite affine [4,4] matrix")
    try:
        inverse = np.linalg.inv(pose)
    except np.linalg.LinAlgError as error:
        raise ValueError("camera_to_world must be invertible") from error
    if not np.isfinite(inverse).all():
        raise ValueError("camera_to_world inverse must be finite")
    return (
        np.array(pose, dtype=np.float64, order="C", copy=True),
        np.array(inverse, dtype=np.float64, order="C", copy=True),
    )


def _pose(value: object) -> np.ndarray:
    return _pose_pair(value)[0]


def _mask_packbits(
    *, mask: Optional[object], mask_packbits: Optional[object]
) -> np.ndarray:
    if (mask is None) == (mask_packbits is None):
        raise ValueError("provide exactly one of mask or mask_packbits")
    if mask is not None:
        array = np.asarray(mask)
        if array.shape != (_F_IMAGE_HEIGHT, _F_IMAGE_WIDTH):
            raise ValueError("mask must have exact shape [480,640]")
        if array.dtype.kind not in "buif" or (
            array.dtype.kind in "uif" and not np.isfinite(array).all()
        ):
            raise ValueError("mask must be finite and binary")
        if not np.all((array == 0) | (array == 1)):
            raise ValueError("mask must be binary")
        packed = np.packbits(np.asarray(array, dtype=np.bool_).reshape(-1), bitorder="little")
    else:
        packed = np.asarray(mask_packbits)
        if packed.dtype != np.uint8 or packed.shape != (_F_MASK_PACKED_BYTES,):
            raise ValueError("mask_packbits must be uint8 with 38400 bytes")
    if packed.shape != (_F_MASK_PACKED_BYTES,):
        raise ValueError("packed mask byte count is invalid")
    return np.array(packed, dtype=np.uint8, order="C", copy=True)


def _bounded_voxel_keys(value: object) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1:] != (3,) or not np.issubdtype(
        array.dtype, np.signedinteger
    ):
        raise ValueError("voxel_keys must be a signed integer [N,3] array")
    keys = np.array(array, dtype=np.int64, order="C", copy=True)
    # Do not use ``abs(int64)`` here: abs(INT64_MIN) itself overflows.
    if len(keys) and (
        np.any(keys < -_SAFE_VOXEL_COORDINATE)
        or np.any(keys > _SAFE_VOXEL_COORDINATE)
    ):
        raise ValueError("voxel_keys exceed the safe neighbourhood range")
    keys = np.unique(keys, axis=0)
    if len(keys) > _F_MAX_VOXELS_PER_OBSERVATION:
        indices = np.linspace(
            0,
            len(keys) - 1,
            num=_F_MAX_VOXELS_PER_OBSERVATION,
            endpoint=True,
            dtype=np.int64,
        )
        keys = keys[indices]
    return np.array(keys, dtype=np.int64, order="C", copy=True)


def _build_exact_voxel_tree(
    keys: np.ndarray,
) -> tuple[Optional[cKDTree], Optional[np.ndarray]]:
    """Build a float64 tree only after an exact common integer translation."""

    if len(keys) == 0:
        return None, None
    anchor = np.array(keys[0], dtype=np.int64, copy=True)
    shifted = keys - anchor[None, :]
    if np.any(shifted < -_FLOAT64_EXACT_INTEGER_SPAN) or np.any(
        shifted > _FLOAT64_EXACT_INTEGER_SPAN
    ):
        return None, _readonly(anchor, np.int64, (3,))
    tree = cKDTree(
        shifted,
        leafsize=16,
        compact_nodes=True,
        copy_data=True,
        balanced_tree=True,
    )
    return tree, _readonly(anchor, np.int64, (3,))


def _observation_digest(
    source_id: str,
    frame_id: int,
    frame_ordinal: int,
    confidence: float,
    lower: np.ndarray,
    upper: np.ndarray,
    keys: np.ndarray,
    packed_mask: np.ndarray,
    pose: np.ndarray,
    intrinsic: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    encoded = source_id.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)
    digest.update(np.asarray([frame_id, frame_ordinal], dtype="<i8").tobytes())
    digest.update(np.asarray([confidence], dtype="<f8").tobytes())
    for value, dtype in (
        (lower, "<f8"),
        (upper, "<f8"),
        (keys, "<i8"),
        (packed_mask, "u1"),
        (pose, "<f8"),
        (intrinsic, "<f8"),
    ):
        digest.update(np.asarray(value, dtype=dtype).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class F3Observation:
    """One exact H0 source with bounded, immutable projection evidence."""

    source_id: str
    frame_id: int
    frame_ordinal: int
    confidence: float
    world_q02: np.ndarray
    world_q98: np.ndarray
    voxel_keys: np.ndarray
    mask_packbits: np.ndarray
    camera_to_world: np.ndarray
    intrinsics: np.ndarray
    observation_sha256: str
    world_to_camera: np.ndarray = field(init=False, repr=False, compare=False)
    mask_pixel_count: int = field(init=False, repr=False, compare=False)
    voxel_tree: Optional[cKDTree] = field(init=False, repr=False, compare=False)
    voxel_tree_anchor: Optional[np.ndarray] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        source_id = _source_id(self.source_id)
        frame_id = _strict_int("frame_id", self.frame_id)
        frame_ordinal = _strict_int("frame_ordinal", self.frame_ordinal)
        confidence = _finite_float("confidence", self.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must lie in [0,1]")
        lower, upper = _bounds(self.world_q02, self.world_q98)
        keys = _bounded_voxel_keys(self.voxel_keys)
        packed = _mask_packbits(mask=None, mask_packbits=self.mask_packbits)
        pose, world_to_camera = _pose_pair(self.camera_to_world)
        intrinsic = _intrinsics(self.intrinsics)
        expected = _observation_digest(
            source_id,
            frame_id,
            frame_ordinal,
            confidence,
            lower,
            upper,
            keys,
            packed,
            pose,
            intrinsic,
        )
        if self.observation_sha256 != expected:
            raise ValueError("observation_sha256 does not match normalized evidence")
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "frame_ordinal", frame_ordinal)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "world_q02", _readonly(lower, np.float64, (3,)))
        object.__setattr__(self, "world_q98", _readonly(upper, np.float64, (3,)))
        object.__setattr__(self, "voxel_keys", _readonly(keys, np.int64))
        object.__setattr__(
            self, "mask_packbits", _readonly(packed, np.uint8, (_F_MASK_PACKED_BYTES,))
        )
        object.__setattr__(self, "camera_to_world", _readonly(pose, np.float64, (4, 4)))
        object.__setattr__(self, "intrinsics", _readonly(intrinsic, np.float64, (3, 3)))
        object.__setattr__(
            self,
            "world_to_camera",
            _readonly(world_to_camera, np.float64, (4, 4)),
        )
        object.__setattr__(
            self,
            "mask_pixel_count",
            int(_BYTE_POPCOUNT[packed].sum(dtype=np.int64)),
        )
        voxel_tree, voxel_tree_anchor = _build_exact_voxel_tree(keys)
        object.__setattr__(self, "voxel_tree", voxel_tree)
        object.__setattr__(self, "voxel_tree_anchor", voxel_tree_anchor)

    @property
    def world_center(self) -> np.ndarray:
        return (self.world_q02 + self.world_q98) * 0.5

    @property
    def world_extent(self) -> np.ndarray:
        return self.world_q98 - self.world_q02

    def unpack_mask(self) -> np.ndarray:
        flat = np.unpackbits(self.mask_packbits, bitorder="little", count=_F_IMAGE_HEIGHT * _F_IMAGE_WIDTH)
        return flat.reshape(_F_IMAGE_HEIGHT, _F_IMAGE_WIDTH).astype(np.bool_, copy=False)


def make_observation(
    *,
    source_id: str,
    frame_id: object,
    frame_ordinal: object,
    confidence: object,
    world_q02: object,
    world_q98: object,
    voxel_keys: object,
    camera_to_world: object,
    intrinsics: object,
    mask: Optional[object] = None,
    mask_packbits: Optional[object] = None,
) -> F3Observation:
    """Normalize one runner row without exposing any forbidden input field."""

    normalized_source = _source_id(source_id)
    normalized_frame = _strict_int("frame_id", frame_id)
    normalized_ordinal = _strict_int("frame_ordinal", frame_ordinal)
    normalized_confidence = _finite_float("confidence", confidence)
    if not 0.0 <= normalized_confidence <= 1.0:
        raise ValueError("confidence must lie in [0,1]")
    lower, upper = _bounds(world_q02, world_q98)
    keys = _bounded_voxel_keys(voxel_keys)
    packed = _mask_packbits(mask=mask, mask_packbits=mask_packbits)
    pose, world_to_camera = _pose_pair(camera_to_world)
    intrinsic = _intrinsics(intrinsics)
    digest = _observation_digest(
        normalized_source,
        normalized_frame,
        normalized_ordinal,
        normalized_confidence,
        lower,
        upper,
        keys,
        packed,
        pose,
        intrinsic,
    )
    # All fields have just passed the same normalization performed by the
    # public dataclass constructor.  Install the immutable normalized values
    # directly so the high-rate runner path does not repeat unique/sort,
    # matrix inversion, mask copying, and hashing in ``__post_init__``.
    result = object.__new__(F3Observation)
    object.__setattr__(result, "source_id", normalized_source)
    object.__setattr__(result, "frame_id", normalized_frame)
    object.__setattr__(result, "frame_ordinal", normalized_ordinal)
    object.__setattr__(result, "confidence", normalized_confidence)
    object.__setattr__(result, "world_q02", _readonly(lower, np.float64, (3,)))
    object.__setattr__(result, "world_q98", _readonly(upper, np.float64, (3,)))
    object.__setattr__(result, "voxel_keys", _readonly(keys, np.int64))
    object.__setattr__(
        result, "mask_packbits", _readonly(packed, np.uint8, (_F_MASK_PACKED_BYTES,))
    )
    object.__setattr__(result, "camera_to_world", _readonly(pose, np.float64, (4, 4)))
    object.__setattr__(result, "intrinsics", _readonly(intrinsic, np.float64, (3, 3)))
    object.__setattr__(
        result, "world_to_camera", _readonly(world_to_camera, np.float64, (4, 4))
    )
    object.__setattr__(
        result,
        "mask_pixel_count",
        int(_BYTE_POPCOUNT[packed].sum(dtype=np.int64)),
    )
    voxel_tree, voxel_tree_anchor = _build_exact_voxel_tree(keys)
    object.__setattr__(result, "voxel_tree", voxel_tree)
    object.__setattr__(result, "voxel_tree_anchor", voxel_tree_anchor)
    object.__setattr__(result, "observation_sha256", digest)
    return result


def _aabb_iou(
    left_q02: np.ndarray,
    left_q98: np.ndarray,
    right_q02: np.ndarray,
    right_q98: np.ndarray,
) -> float:
    intersection_extent = np.maximum(
        np.minimum(left_q98, right_q98) - np.maximum(left_q02, right_q02), 0.0
    )
    intersection = float(np.prod(intersection_extent))
    left_volume = float(np.prod(np.maximum(left_q98 - left_q02, 0.0)))
    right_volume = float(np.prod(np.maximum(right_q98 - right_q02, 0.0)))
    union = left_volume + right_volume - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def _project_aabb(
    lower: np.ndarray,
    upper: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: Optional[np.ndarray] = None,
    *,
    world_to_camera: Optional[np.ndarray] = None,
) -> tuple[Optional[np.ndarray], str]:
    corners_world = lower[None, :] + _CORNER_BITS * (upper - lower)[None, :]
    if world_to_camera is None:
        if camera_to_world is None:
            raise ValueError("one camera transform is required")
        try:
            world_to_camera = np.linalg.inv(camera_to_world)
        except np.linalg.LinAlgError:
            return None, "noninvertible_pose"
    homogeneous = np.column_stack((corners_world, np.ones(8, dtype=np.float64)))
    camera = (homogeneous @ world_to_camera.T)[:, :3]
    if not np.isfinite(camera).all():
        return None, "nonfinite_camera_corners"
    if not np.all(camera[:, 2] > _F_NEAR_PLANE_M):
        return None, "corner_at_or_behind_near_plane"
    projected = camera @ intrinsics.T
    pixels = projected[:, :2] / projected[:, 2:3]
    if not np.isfinite(pixels).all():
        return None, "nonfinite_projection"
    x = np.clip(pixels[:, 0], 0.0, float(_F_IMAGE_WIDTH))
    y = np.clip(pixels[:, 1], 0.0, float(_F_IMAGE_HEIGHT))
    box = np.asarray([x.min(), y.min(), x.max(), y.max()], dtype=np.float64)
    if box[2] <= box[0] or box[3] <= box[1]:
        return None, "degenerate_after_clipping"
    return box, "valid"


def _box_mask_iou_unpack_reference(box: np.ndarray, packed_mask: np.ndarray) -> float:
    """Literal raster/unpack implementation retained for equivalence tests."""

    x_start = max(0, min(_F_IMAGE_WIDTH, int(np.floor(box[0]))))
    y_start = max(0, min(_F_IMAGE_HEIGHT, int(np.floor(box[1]))))
    x_stop = max(0, min(_F_IMAGE_WIDTH, int(np.ceil(box[2]))))
    y_stop = max(0, min(_F_IMAGE_HEIGHT, int(np.ceil(box[3]))))
    box_area = max(x_stop - x_start, 0) * max(y_stop - y_start, 0)
    mask = np.unpackbits(
        packed_mask,
        bitorder="little",
        count=_F_IMAGE_HEIGHT * _F_IMAGE_WIDTH,
    ).reshape(_F_IMAGE_HEIGHT, _F_IMAGE_WIDTH)
    mask_area = int(np.count_nonzero(mask))
    intersection = int(np.count_nonzero(mask[y_start:y_stop, x_start:x_stop]))
    union = box_area + mask_area - intersection
    return 0.0 if union <= 0 else float(intersection / union)


def _box_mask_iou(
    box: np.ndarray,
    packed_mask: np.ndarray,
    mask_pixel_count: Optional[int] = None,
) -> float:
    """Exact rectangle IoU counted directly in the little-endian bitstream."""

    x_start = max(0, min(_F_IMAGE_WIDTH, int(np.floor(box[0]))))
    y_start = max(0, min(_F_IMAGE_HEIGHT, int(np.floor(box[1]))))
    x_stop = max(0, min(_F_IMAGE_WIDTH, int(np.ceil(box[2]))))
    y_stop = max(0, min(_F_IMAGE_HEIGHT, int(np.ceil(box[3]))))
    box_width = max(x_stop - x_start, 0)
    box_height = max(y_stop - y_start, 0)
    box_area = box_width * box_height
    if mask_pixel_count is None:
        mask_area = int(_BYTE_POPCOUNT[packed_mask].sum(dtype=np.int64))
    else:
        mask_area = int(mask_pixel_count)
    intersection = 0
    if box_width and box_height:
        rows = packed_mask.reshape(_F_IMAGE_HEIGHT, _F_IMAGE_WIDTH // 8)[
            y_start:y_stop
        ]
        first_byte = x_start // 8
        last_byte = (x_stop - 1) // 8
        first_bit = x_start % 8
        last_bit_count = (x_stop - 1) % 8 + 1
        if first_byte == last_byte:
            bit_count = x_stop - x_start
            byte_mask = ((1 << bit_count) - 1) << first_bit
            intersection = int(
                _BYTE_POPCOUNT[
                    np.bitwise_and(rows[:, first_byte], np.uint8(byte_mask))
                ].sum(dtype=np.int64)
            )
        else:
            first_mask = (0xFF << first_bit) & 0xFF
            last_mask = (1 << last_bit_count) - 1
            intersection = int(
                _BYTE_POPCOUNT[
                    np.bitwise_and(rows[:, first_byte], np.uint8(first_mask))
                ].sum(dtype=np.int64)
            )
            intersection += int(
                _BYTE_POPCOUNT[
                    np.bitwise_and(rows[:, last_byte], np.uint8(last_mask))
                ].sum(dtype=np.int64)
            )
            if last_byte > first_byte + 1:
                intersection += int(
                    _BYTE_POPCOUNT[rows[:, first_byte + 1 : last_byte]].sum(
                        dtype=np.int64
                    )
                )
    union = box_area + mask_area - intersection
    return 0.0 if union <= 0 else float(intersection / union)


def projected_aabb_mask_iou(
    *,
    world_q02: object,
    world_q98: object,
    intrinsics: object,
    camera_to_world: object,
    mask: Optional[object] = None,
    mask_packbits: Optional[object] = None,
) -> tuple[bool, Optional[float], Optional[np.ndarray], str]:
    """Project with the exact F3 convention and score one raw 480x640 mask."""

    lower, upper = _bounds(world_q02, world_q98, require_min_extent=False)
    intrinsic = _intrinsics(intrinsics)
    pose = _pose(camera_to_world)
    packed = _mask_packbits(mask=mask, mask_packbits=mask_packbits)
    box, reason = _project_aabb(lower, upper, intrinsic, pose)
    if box is None:
        return False, None, None, reason
    readonly_box = _readonly(box, np.float64, (4,))
    return True, _box_mask_iou(box, packed), readonly_box, "valid"


def _row_records(rows: np.ndarray) -> np.ndarray:
    records = np.empty(len(rows), dtype=_ROW_DTYPE)
    if len(rows):
        records["x"] = rows[:, 0]
        records["y"] = rows[:, 1]
        records["z"] = rows[:, 2]
    return records


def _rows_present(queries: np.ndarray, sorted_rows: np.ndarray) -> np.ndarray:
    if len(queries) == 0 or len(sorted_rows) == 0:
        return np.zeros(len(queries), dtype=np.bool_)
    haystack = _row_records(sorted_rows)
    needles = _row_records(queries)
    positions = np.searchsorted(haystack, needles)
    present = positions < len(haystack)
    if np.any(present):
        valid = np.flatnonzero(present)
        present[valid] = haystack[positions[valid]] == needles[valid]
    return present


def _view_neighbourhood_support(union: np.ndarray, view: np.ndarray) -> np.ndarray:
    present = np.zeros(len(union), dtype=np.bool_)
    for offset in _OFFSETS:
        if present.all():
            break
        pending = np.flatnonzero(~present)
        query = union[pending] + np.asarray(offset, dtype=np.int64)
        present[pending] = _rows_present(query, view)
    return present


@dataclass(frozen=True)
class _ConsensusBox:
    valid: bool
    reason: str
    world_q02: Optional[np.ndarray]
    world_q98: Optional[np.ndarray]
    consensus_voxel_count_before_cap: int
    consensus_voxel_count: int

    def __post_init__(self) -> None:
        if self.world_q02 is None or self.world_q98 is None:
            if self.world_q02 is not None or self.world_q98 is not None:
                raise ValueError("consensus bounds must both be present or absent")
        else:
            lower, upper = _bounds(self.world_q02, self.world_q98)
            object.__setattr__(self, "world_q02", _readonly(lower, np.float64, (3,)))
            object.__setattr__(self, "world_q98", _readonly(upper, np.float64, (3,)))


def _fit_consensus_keys(retained: np.ndarray) -> _ConsensusBox:
    """Fit one frozen q02/q98 box from already-supported lexical keys."""

    before_cap = int(len(retained))
    if before_cap < _F_MIN_CONSENSUS_VOXELS:
        return _ConsensusBox(
            False,
            "too_few_consensus_voxels",
            None,
            None,
            before_cap,
            before_cap,
        )
    if len(retained) > _F_MAX_CONSENSUS_VOXELS:
        indices = np.linspace(
            0,
            len(retained) - 1,
            num=_F_MAX_CONSENSUS_VOXELS,
            endpoint=True,
            dtype=np.int64,
        )
        retained = retained[indices]
    centers = (retained.astype(np.float64) + 0.5) * _F_VOXEL_SIZE_M
    try:
        quantiles = np.quantile(centers, (0.02, 0.98), axis=0, method="linear")
    except (FloatingPointError, TypeError, ValueError, OverflowError):
        return _ConsensusBox(
            False,
            "invalid_consensus_quantiles",
            None,
            None,
            before_cap,
            len(retained),
        )
    lower = np.asarray(quantiles[0], dtype=np.float64)
    upper = np.asarray(quantiles[1], dtype=np.float64)
    if (
        lower.shape != (3,)
        or upper.shape != (3,)
        or not np.isfinite(quantiles).all()
        or np.any(upper - lower < _F_MIN_AABB_EXTENT_M - 1e-12)
    ):
        return _ConsensusBox(
            False,
            "consensus_extent_below_0.02m",
            None,
            None,
            before_cap,
            len(retained),
        )
    return _ConsensusBox(True, "valid", lower, upper, before_cap, len(retained))


def _consensus_box_reference(observations: Sequence[F3Observation]) -> _ConsensusBox:
    """Literal per-fit reference retained for equivalence auditing."""

    views = tuple(observation.voxel_keys for observation in observations)
    if len(views) < 2:
        return _ConsensusBox(False, "fewer_than_two_fitting_views", None, None, 0, 0)
    nonempty = tuple(view for view in views if len(view))
    if not nonempty:
        return _ConsensusBox(False, "too_few_consensus_voxels", None, None, 0, 0)
    union = np.unique(np.concatenate(nonempty, axis=0), axis=0)
    support_count = np.zeros(len(union), dtype=np.int16)
    for view in views:
        support_count += _view_neighbourhood_support(union, view).astype(np.int16)
    retained = union[support_count >= 2]
    return _fit_consensus_keys(retained)


def _view_neighbourhood_support_tree(
    union: np.ndarray,
    view: np.ndarray,
    tree: Optional[cKDTree] = None,
    tree_anchor: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Exact compiled Chebyshev-radius-one membership for integer keys.

    Both the indexed keys and queries are first translated by the same int64
    anchor.  The compiled path is used only while every translated coordinate
    is in float64's exact integer domain; otherwise the literal integer
    reference is used.  ``eps=0`` disables approximation and ``p=inf`` is the
    frozen Chebyshev metric.
    """

    if len(union) == 0 or len(view) == 0:
        return np.zeros(len(union), dtype=np.bool_)
    index = tree
    anchor = tree_anchor
    if index is None or anchor is None:
        index, anchor = _build_exact_voxel_tree(view)
    if index is None or anchor is None:
        return _view_neighbourhood_support(union, view)
    shifted_queries = union - anchor[None, :]
    if np.any(shifted_queries < -_FLOAT64_EXACT_INTEGER_SPAN) or np.any(
        shifted_queries > _FLOAT64_EXACT_INTEGER_SPAN
    ):
        return _view_neighbourhood_support(union, view)
    counts = index.query_ball_point(
        shifted_queries,
        r=1.0,
        p=np.inf,
        eps=0.0,
        return_length=True,
    )
    return np.asarray(counts, dtype=np.int64) > 0


def _consensus_boxes_all_loo(
    observations: Tuple[F3Observation, ...],
    support_cache: Optional[dict[tuple[str, str], np.ndarray]] = None,
) -> tuple[Tuple[_ConsensusBox, ...], _ConsensusBox]:
    """Build all LOO and full boxes from one exact support matrix.

    A fold's candidate universe is still only the union of its fitting views:
    ``exact_count_without_heldout >= 1`` removes keys introduced solely by the
    held-out view.  ``support_count_without_heldout >= 2`` then applies the
    unchanged cross-view neighbourhood rule.  Consequently each retained-key
    vector is byte-identical to independently recomputing that fold, while
    each view/union neighbourhood query is executed only once.
    """

    views = tuple(observation.voxel_keys for observation in observations)
    nonempty = tuple(view for view in views if len(view))
    if not nonempty:
        empty = _ConsensusBox(False, "too_few_consensus_voxels", None, None, 0, 0)
        return tuple(empty for _ in views), empty
    concatenated = np.concatenate(views, axis=0)
    union, inverse = np.unique(concatenated, axis=0, return_inverse=True)
    exact = np.zeros((len(union), len(views)), dtype=np.bool_)
    support = np.zeros((len(union), len(views)), dtype=np.bool_)
    offsets = np.zeros(len(views) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(
        np.asarray([len(view) for view in views], dtype=np.int64)
    )
    mutable_cache = {} if support_cache is None else support_cache
    for origin_index, origin in enumerate(observations):
        positions = inverse[offsets[origin_index] : offsets[origin_index + 1]]
        exact[positions, origin_index] = True
        # An exact occurrence is necessarily radius-one support in its own
        # view, without a tree query.
        support[positions, origin_index] = True
        for support_index, support_observation in enumerate(observations):
            if support_index == origin_index:
                continue
            key = (origin.source_id, support_observation.source_id)
            values = mutable_cache.get(key)
            if values is None:
                values = _view_neighbourhood_support_tree(
                    origin.voxel_keys,
                    support_observation.voxel_keys,
                    support_observation.voxel_tree,
                    support_observation.voxel_tree_anchor,
                )
                values = _readonly(
                    values, np.bool_, (len(origin.voxel_keys),)
                )
                mutable_cache[key] = values
            elif values.shape != (len(origin.voxel_keys),):
                raise RuntimeError("C pair-support cache shape differs")
            support[positions, support_index] |= values
    exact_sum = exact.sum(axis=1, dtype=np.int16)
    support_sum = support.sum(axis=1, dtype=np.int16)
    folds = []
    for heldout_index in range(len(views)):
        fold_union = (exact_sum - exact[:, heldout_index]) >= 1
        fold_support = (support_sum - support[:, heldout_index]) >= 2
        folds.append(_fit_consensus_keys(union[fold_union & fold_support]))
    full = _fit_consensus_keys(union[support_sum >= 2])
    return tuple(folds), full


@dataclass(frozen=True)
class ProjectionFold:
    heldout_source_id: str
    heldout_frame_id: int
    heldout_frame_ordinal: int
    fitting_source_ids: Tuple[str, ...]
    valid: bool
    reason: str
    projection_iou: Optional[float]
    projected_xyxy: Optional[np.ndarray]
    world_q02: Optional[np.ndarray]
    world_q98: Optional[np.ndarray]
    consensus_voxel_count_before_cap: Optional[int] = None
    consensus_voxel_count: Optional[int] = None

    def __post_init__(self) -> None:
        _source_id(self.heldout_source_id)
        _strict_int("heldout_frame_id", self.heldout_frame_id)
        _strict_int("heldout_frame_ordinal", self.heldout_frame_ordinal)
        for item in self.fitting_source_ids:
            _source_id(item)
        if self.valid:
            if self.projection_iou is None or self.projected_xyxy is None:
                raise ValueError("valid projection fold requires an IoU and image box")
            iou = _finite_float("projection_iou", self.projection_iou)
            if not 0.0 <= iou <= 1.0:
                raise ValueError("projection_iou must lie in [0,1]")
            object.__setattr__(self, "projection_iou", iou)
        elif self.projection_iou is not None or self.projected_xyxy is not None:
            raise ValueError("invalid projection fold cannot carry a projection score")
        if self.projected_xyxy is not None:
            object.__setattr__(
                self,
                "projected_xyxy",
                _readonly(self.projected_xyxy, np.float64, (4,)),
            )
        if self.world_q02 is None or self.world_q98 is None:
            if self.world_q02 is not None or self.world_q98 is not None:
                raise ValueError("fold bounds must both be present or absent")
        else:
            lower, upper = _bounds(self.world_q02, self.world_q98)
            object.__setattr__(self, "world_q02", _readonly(lower, np.float64, (3,)))
            object.__setattr__(self, "world_q98", _readonly(upper, np.float64, (3,)))


@dataclass(frozen=True)
class BCandidateEvaluation:
    source_id: str
    frame_id: int
    frame_ordinal: int
    world_q02: np.ndarray
    world_q98: np.ndarray
    folds: Tuple[ProjectionFold, ...]
    valid_fold_count: int
    score: Optional[float]

    def __post_init__(self) -> None:
        lower, upper = _bounds(self.world_q02, self.world_q98)
        object.__setattr__(self, "world_q02", _readonly(lower, np.float64, (3,)))
        object.__setattr__(self, "world_q98", _readonly(upper, np.float64, (3,)))
        if self.valid_fold_count != sum(fold.valid for fold in self.folds):
            raise ValueError("B valid_fold_count is inconsistent")
        if self.score is None:
            if self.valid_fold_count >= 2:
                raise ValueError("B candidate with two valid folds requires a score")
        else:
            score = _finite_float("B candidate score", self.score)
            if not 0.0 <= score <= 1.0 or self.valid_fold_count < 2:
                raise ValueError("B candidate score is inconsistent")
            object.__setattr__(self, "score", score)


@dataclass(frozen=True)
class F3Hypothesis:
    name: str
    available: bool
    valid: bool
    reason: str
    world_q02: Optional[np.ndarray]
    world_q98: Optional[np.ndarray]
    score: Optional[float]
    fold_ious: Tuple[float, ...]
    valid_fold_count: int
    folds: Tuple[ProjectionFold, ...]
    stability_ious: Tuple[float, ...] = ()
    stability_median_iou: Optional[float] = None
    source_id: Optional[str] = None
    consensus_voxel_count_before_cap: Optional[int] = None
    consensus_voxel_count: Optional[int] = None
    b_candidates: Tuple[BCandidateEvaluation, ...] = ()

    def __post_init__(self) -> None:
        if self.name not in {"B", "C"}:
            raise ValueError("hypothesis name must be B or C")
        if self.valid and not self.available:
            raise ValueError("valid hypothesis must be available")
        if self.world_q02 is None or self.world_q98 is None:
            if self.world_q02 is not None or self.world_q98 is not None:
                raise ValueError("hypothesis bounds must both be present or absent")
            if self.available:
                raise ValueError("available hypothesis requires bounds")
        else:
            lower, upper = _bounds(self.world_q02, self.world_q98)
            object.__setattr__(self, "world_q02", _readonly(lower, np.float64, (3,)))
            object.__setattr__(self, "world_q98", _readonly(upper, np.float64, (3,)))
        if self.score is None:
            if self.available:
                raise ValueError("available hypothesis requires a score")
        else:
            score = _finite_float("hypothesis score", self.score)
            if not 0.0 <= score <= 1.0:
                raise ValueError("hypothesis score must lie in [0,1]")
            object.__setattr__(self, "score", score)
        normalized_fold_ious = tuple(_finite_float("fold_iou", value) for value in self.fold_ious)
        if any(not 0.0 <= value <= 1.0 for value in normalized_fold_ious):
            raise ValueError("fold IoUs must lie in [0,1]")
        if self.valid_fold_count != len(normalized_fold_ious):
            raise ValueError("valid_fold_count must equal len(fold_ious)")
        if self.valid_fold_count != sum(fold.valid for fold in self.folds):
            raise ValueError("valid_fold_count is inconsistent with folds")
        object.__setattr__(self, "fold_ious", normalized_fold_ious)
        stability = tuple(_finite_float("stability_iou", value) for value in self.stability_ious)
        if any(not 0.0 <= value <= 1.0 for value in stability):
            raise ValueError("stability IoUs must lie in [0,1]")
        object.__setattr__(self, "stability_ious", stability)
        if self.stability_median_iou is not None:
            median = _finite_float("stability_median_iou", self.stability_median_iou)
            if not 0.0 <= median <= 1.0:
                raise ValueError("stability median must lie in [0,1]")
            object.__setattr__(self, "stability_median_iou", median)


@dataclass(frozen=True)
class F3Selector:
    chosen: Optional[str]
    reason: str
    world_q02: Optional[np.ndarray]
    world_q98: Optional[np.ndarray]
    score: Optional[float]

    def __post_init__(self) -> None:
        if self.chosen not in {None, "B", "C"}:
            raise ValueError("selector choice must be B, C, or None")
        if self.chosen is None:
            if self.world_q02 is not None or self.world_q98 is not None or self.score is not None:
                raise ValueError("abstaining selector cannot carry geometry or score")
            return
        if self.world_q02 is None or self.world_q98 is None or self.score is None:
            raise ValueError("selected hypothesis requires geometry and score")
        lower, upper = _bounds(self.world_q02, self.world_q98)
        score = _finite_float("selector score", self.score)
        if not 0.0 <= score <= 1.0:
            raise ValueError("selector score must lie in [0,1]")
        object.__setattr__(self, "world_q02", _readonly(lower, np.float64, (3,)))
        object.__setattr__(self, "world_q98", _readonly(upper, np.float64, (3,)))
        object.__setattr__(self, "score", score)


@dataclass(frozen=True)
class F3TrackReceipt:
    track_id: int
    # Complete, lightweight provenance.  These vectors are never truncated,
    # so their union can authenticate the exact sealed F1/H0 source universe.
    source_ids: Tuple[str, ...]
    frame_ids: Tuple[int, ...]
    frame_ordinals: Tuple[int, ...]
    observation_count: int
    # Only these last five views contribute masks/poses/voxels to B/C.
    retained_source_ids: Tuple[str, ...]
    retained_frame_ids: Tuple[int, ...]
    retained_frame_ordinals: Tuple[int, ...]
    retained_view_count: int
    total_observation_count: int
    confirmed: bool
    hypothesis_b: F3Hypothesis
    hypothesis_c: F3Hypothesis
    selector: F3Selector
    max_logical_accessed_ordinal: int
    seal_reason: str

    def __post_init__(self) -> None:
        _strict_int("track_id", self.track_id)
        count = len(self.source_ids)
        if not (
            count == len(self.frame_ids) == len(self.frame_ordinals) == self.observation_count
        ):
            raise ValueError("track receipt identity vectors must align")
        retained_count = len(self.retained_source_ids)
        if not (
            retained_count
            == len(self.retained_frame_ids)
            == len(self.retained_frame_ordinals)
            == self.retained_view_count
        ):
            raise ValueError("retained track identity vectors must align")
        if retained_count > _F_MAX_RETAINED_OBSERVATIONS:
            raise ValueError("track receipt exceeds bounded observation memory")
        if tuple(sorted(self.frame_ordinals)) != self.frame_ordinals or len(set(self.frame_ordinals)) != count:
            raise ValueError("track receipt frame ordinals must be increasing and distinct")
        if (
            tuple(sorted(self.retained_frame_ordinals)) != self.retained_frame_ordinals
            or len(set(self.retained_frame_ordinals)) != retained_count
        ):
            raise ValueError("retained frame ordinals must be increasing and distinct")
        if self.total_observation_count != count:
            raise ValueError("total_observation_count must equal complete lineage count")
        if self.confirmed != (retained_count >= _F_MIN_DISTINCT_FRAMES):
            raise ValueError("confirmed must reflect retained distinct views")
        if self.max_logical_accessed_ordinal != max(self.frame_ordinals, default=0):
            raise ValueError("track max logical access must match retained evidence")
        if self.seal_reason not in {"live", "ttl", "terminal"}:
            raise ValueError("invalid track seal reason")


@dataclass(frozen=True)
class F3Assignment:
    source_id: str
    track_id: Optional[int]
    action: str

    def __post_init__(self) -> None:
        _source_id(self.source_id)
        if self.action not in {"matched", "created", "dropped_capacity"}:
            raise ValueError("invalid F3 assignment action")
        if self.action == "dropped_capacity":
            if self.track_id is not None:
                raise ValueError("capacity drop cannot have a track_id")
        elif self.track_id is None:
            raise ValueError("accepted assignment requires track_id")
        else:
            _strict_int("track_id", self.track_id)


@dataclass(frozen=True)
class F3FrameQuery:
    serial: int
    frame_id: int
    frame_ordinal: int
    history_max_frame_ordinal: Optional[int]
    max_logical_accessed_ordinal: int
    prior_track_ids: Tuple[int, ...]
    assignments: Tuple[F3Assignment, ...]
    retired_track_ids: Tuple[int, ...]
    elapsed_ms: float
    audit_complete: bool


@dataclass(frozen=True)
class F3FrameCommit:
    frame_id: int
    frame_ordinal: int
    history_max_frame_ordinal: Optional[int]
    max_logical_accessed_ordinal: int
    assignments: Tuple[F3Assignment, ...]
    retired_track_ids: Tuple[int, ...]
    active_track_ids: Tuple[int, ...]
    elapsed_ms: float
    audit_complete: bool


@dataclass(frozen=True)
class F3TerminalSeal:
    tracks: Tuple[F3TrackReceipt, ...]
    last_frame_id: Optional[int]
    last_frame_ordinal: Optional[int]
    max_logical_accessed_ordinal: Optional[int]
    keyframe_count: int
    audit_complete: bool
    schema: str = SCHEMA
    mode: str = MODE
    protocol_id: str = PROTOCOL_ID


@dataclass(frozen=True)
class _Track:
    track_id: int
    last_keyframe_step: int
    retained: Tuple[F3Observation, ...]
    source_ids: Tuple[str, ...]
    frame_ids: Tuple[int, ...]
    frame_ordinals: Tuple[int, ...]
    b_fold_cache: Mapping[tuple[str, str], ProjectionFold]
    c_support_cache: Mapping[tuple[str, str], np.ndarray]
    latest_receipt: F3TrackReceipt


@dataclass(frozen=True)
class _Pending:
    public: F3FrameQuery
    tracks: Mapping[int, _Track]
    sealed: Mapping[int, F3TrackReceipt]
    next_track_id: int
    seen_source_ids: frozenset[str]
    audit_complete: bool


def _association_edges_reference(
    rows: Tuple[F3Observation, ...],
    prior_ids: Tuple[int, ...],
    tracks: Mapping[int, _Track],
) -> list[tuple[float, float, int, str, F3Observation]]:
    """Literal scalar association graph retained for equivalence auditing."""

    edges = []
    for row in rows:
        for track_id in prior_ids:
            anchor = tracks[track_id].retained[-1]
            iou = _aabb_iou(
                row.world_q02,
                row.world_q98,
                anchor.world_q02,
                anchor.world_q98,
            )
            distance = float(np.linalg.norm(row.world_center - anchor.world_center))
            if iou >= _F_MATCH_AABB_IOU and distance <= _F_MATCH_CENTER_DISTANCE_M:
                edges.append((-iou, distance, track_id, row.source_id, row))
    edges.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return edges


def _association_edges(
    rows: Tuple[F3Observation, ...],
    prior_ids: Tuple[int, ...],
    tracks: Mapping[int, _Track],
) -> list[tuple[float, float, int, str, F3Observation]]:
    """Build the exact frozen edge ledger with bounded vectorized geometry."""

    if not rows or not prior_ids:
        return []
    row_lower = np.stack([row.world_q02 for row in rows], axis=0)
    row_upper = np.stack([row.world_q98 for row in rows], axis=0)
    anchors = tuple(tracks[track_id].retained[-1] for track_id in prior_ids)
    anchor_lower = np.stack([row.world_q02 for row in anchors], axis=0)
    anchor_upper = np.stack([row.world_q98 for row in anchors], axis=0)
    intersection_extent = np.maximum(
        np.minimum(row_upper[:, None, :], anchor_upper[None, :, :])
        - np.maximum(row_lower[:, None, :], anchor_lower[None, :, :]),
        0.0,
    )
    intersection = np.prod(intersection_extent, axis=2)
    row_volume = np.prod(row_upper - row_lower, axis=1)
    anchor_volume = np.prod(anchor_upper - anchor_lower, axis=1)
    union = row_volume[:, None] + anchor_volume[None, :] - intersection
    iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )
    row_center = (row_lower + row_upper) * 0.5
    anchor_center = (anchor_lower + anchor_upper) * 0.5
    center_delta = row_center[:, None, :] - anchor_center[None, :, :]
    distance = np.sqrt(np.sum(center_delta * center_delta, axis=2))
    row_indices, track_indices = np.nonzero(
        (iou >= _F_MATCH_AABB_IOU)
        & (distance <= _F_MATCH_CENTER_DISTANCE_M)
    )
    edges = [
        (
            -float(iou[row_index, track_index]),
            float(distance[row_index, track_index]),
            prior_ids[int(track_index)],
            rows[int(row_index)].source_id,
            rows[int(row_index)],
        )
        for row_index, track_index in zip(row_indices, track_indices)
    ]
    edges.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return edges


def _projection_fold(
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    heldout: F3Observation,
    fitting_source_ids: Tuple[str, ...],
    consensus: Optional[_ConsensusBox] = None,
) -> ProjectionFold:
    box, reason = _project_aabb(
        lower,
        upper,
        heldout.intrinsics,
        world_to_camera=heldout.world_to_camera,
    )
    before_cap = None if consensus is None else consensus.consensus_voxel_count_before_cap
    count = None if consensus is None else consensus.consensus_voxel_count
    if box is None:
        return ProjectionFold(
            heldout.source_id,
            heldout.frame_id,
            heldout.frame_ordinal,
            fitting_source_ids,
            False,
            reason,
            None,
            None,
            lower,
            upper,
            before_cap,
            count,
        )
    iou = _box_mask_iou(
        box, heldout.mask_packbits, heldout.mask_pixel_count
    )
    return ProjectionFold(
        heldout.source_id,
        heldout.frame_id,
        heldout.frame_ordinal,
        fitting_source_ids,
        True,
        "valid",
        iou,
        box,
        lower,
        upper,
        before_cap,
        count,
    )


def _unavailable_hypothesis(name: str, reason: str) -> F3Hypothesis:
    return F3Hypothesis(
        name=name,
        available=False,
        valid=False,
        reason=reason,
        world_q02=None,
        world_q98=None,
        score=None,
        fold_ious=(),
        valid_fold_count=0,
        folds=(),
    )


def _best_single_view(
    observations: Tuple[F3Observation, ...],
    fold_cache: Optional[dict[tuple[str, str], ProjectionFold]] = None,
) -> F3Hypothesis:
    if len(observations) < _F_MIN_DISTINCT_FRAMES:
        return _unavailable_hypothesis("B", "fewer_than_three_distinct_frames")
    candidates = []
    for candidate in observations:
        folds_list = []
        for heldout in observations:
            if heldout.source_id == candidate.source_id:
                continue
            key = (candidate.source_id, heldout.source_id)
            fold = None if fold_cache is None else fold_cache.get(key)
            if fold is None:
                fold = _projection_fold(
                    lower=candidate.world_q02,
                    upper=candidate.world_q98,
                    heldout=heldout,
                    fitting_source_ids=(candidate.source_id,),
                )
                if fold_cache is not None:
                    fold_cache[key] = fold
            folds_list.append(fold)
        folds = tuple(folds_list)
        fold_ious = tuple(
            float(fold.projection_iou) for fold in folds if fold.valid
        )
        score = float(np.median(fold_ious)) if len(fold_ious) >= 2 else None
        candidates.append(
            BCandidateEvaluation(
                source_id=candidate.source_id,
                frame_id=candidate.frame_id,
                frame_ordinal=candidate.frame_ordinal,
                world_q02=candidate.world_q02,
                world_q98=candidate.world_q98,
                folds=folds,
                valid_fold_count=len(fold_ious),
                score=score,
            )
        )
    available = tuple(candidate for candidate in candidates if candidate.score is not None)
    if not available:
        return F3Hypothesis(
            name="B",
            available=False,
            valid=False,
            reason="fewer_than_two_valid_folds",
            world_q02=None,
            world_q98=None,
            score=None,
            fold_ious=(),
            valid_fold_count=0,
            folds=(),
            b_candidates=tuple(candidates),
        )
    winner = sorted(
        available,
        key=lambda item: (-float(item.score), item.frame_ordinal, item.source_id),
    )[0]
    fold_ious = tuple(float(fold.projection_iou) for fold in winner.folds if fold.valid)
    valid = float(winner.score) >= _F_B_MIN_PROJECTION_IOU
    return F3Hypothesis(
        name="B",
        available=True,
        valid=valid,
        reason="valid" if valid else "projection_iou_below_0.10",
        world_q02=winner.world_q02,
        world_q98=winner.world_q98,
        score=winner.score,
        fold_ious=fold_ious,
        valid_fold_count=len(fold_ious),
        folds=winner.folds,
        source_id=winner.source_id,
        b_candidates=tuple(candidates),
    )


def _multi_view_consensus(
    observations: Tuple[F3Observation, ...],
    b: F3Hypothesis,
    support_cache: Optional[dict[tuple[str, str], np.ndarray]] = None,
) -> F3Hypothesis:
    if len(observations) < _F_MIN_DISTINCT_FRAMES:
        return _unavailable_hypothesis("C", "fewer_than_three_distinct_frames")
    loo_consensus, full = _consensus_boxes_all_loo(observations, support_cache)
    folds = []
    geometry_boxes: list[tuple[np.ndarray, np.ndarray]] = []
    for heldout_index, heldout in enumerate(observations):
        fitting = observations[:heldout_index] + observations[heldout_index + 1 :]
        consensus = loo_consensus[heldout_index]
        fitting_ids = tuple(item.source_id for item in fitting)
        if not consensus.valid:
            folds.append(
                ProjectionFold(
                    heldout.source_id,
                    heldout.frame_id,
                    heldout.frame_ordinal,
                    fitting_ids,
                    False,
                    consensus.reason,
                    None,
                    None,
                    None,
                    None,
                    consensus.consensus_voxel_count_before_cap,
                    consensus.consensus_voxel_count,
                )
            )
            continue
        assert consensus.world_q02 is not None and consensus.world_q98 is not None
        geometry_boxes.append((consensus.world_q02, consensus.world_q98))
        folds.append(
            _projection_fold(
                lower=consensus.world_q02,
                upper=consensus.world_q98,
                heldout=heldout,
                fitting_source_ids=fitting_ids,
                consensus=consensus,
            )
        )
    if not full.valid:
        fold_ious = tuple(float(fold.projection_iou) for fold in folds if fold.valid)
        return F3Hypothesis(
            name="C",
            available=False,
            valid=False,
            reason=f"full_{full.reason}",
            world_q02=None,
            world_q98=None,
            score=None,
            fold_ious=fold_ious,
            valid_fold_count=len(fold_ious),
            folds=tuple(folds),
            consensus_voxel_count_before_cap=full.consensus_voxel_count_before_cap,
            consensus_voxel_count=full.consensus_voxel_count,
        )
    assert full.world_q02 is not None and full.world_q98 is not None
    fold_ious = tuple(float(fold.projection_iou) for fold in folds if fold.valid)
    if len(fold_ious) < 2:
        return F3Hypothesis(
            name="C",
            available=False,
            valid=False,
            reason="fewer_than_two_valid_folds",
            world_q02=None,
            world_q98=None,
            score=None,
            fold_ious=fold_ious,
            valid_fold_count=len(fold_ious),
            folds=tuple(folds),
            consensus_voxel_count_before_cap=full.consensus_voxel_count_before_cap,
            consensus_voxel_count=full.consensus_voxel_count,
        )
    score = float(np.median(fold_ious))
    stability_boxes = geometry_boxes + [(full.world_q02, full.world_q98)]
    stability_ious = tuple(
        _aabb_iou(left[0], left[1], right[0], right[1])
        for left_index, left in enumerate(stability_boxes)
        for right in stability_boxes[left_index + 1 :]
    )
    stability_median = (
        float(np.median(stability_ious)) if stability_ious else None
    )
    reason = "valid"
    valid = True
    if score < _F_B_MIN_PROJECTION_IOU:
        valid = False
        reason = "projection_iou_below_0.10"
    elif stability_median is None or stability_median < _F_C_MIN_STABILITY_IOU:
        valid = False
        reason = "stability_iou_below_0.25"
    elif b.available:
        assert b.world_q02 is not None and b.world_q98 is not None
        b_center = (b.world_q02 + b.world_q98) * 0.5
        c_center = (full.world_q02 + full.world_q98) * 0.5
        b_extent = b.world_q98 - b.world_q02
        c_extent = full.world_q98 - full.world_q02
        center_shift = float(np.linalg.norm(c_center - b_center))
        extent_ratio = c_extent / b_extent
        volume_ratio = float(np.prod(c_extent) / np.prod(b_extent))
        if center_shift > _F_C_MAX_CENTER_SHIFT_M:
            valid = False
            reason = "center_shift_above_0.50m"
        elif np.any(extent_ratio < _F_C_MIN_EXTENT_RATIO) or np.any(
            extent_ratio > _F_C_MAX_EXTENT_RATIO
        ):
            valid = False
            reason = "extent_ratio_outside_0.5_to_2.0"
        elif not _F_C_MIN_VOLUME_RATIO <= volume_ratio <= _F_C_MAX_VOLUME_RATIO:
            valid = False
            reason = "volume_ratio_outside_0.25_to_4.0"
    return F3Hypothesis(
        name="C",
        available=True,
        valid=valid,
        reason=reason,
        world_q02=full.world_q02,
        world_q98=full.world_q98,
        score=score,
        fold_ious=fold_ious,
        valid_fold_count=len(fold_ious),
        folds=tuple(folds),
        stability_ious=stability_ious,
        stability_median_iou=stability_median,
        consensus_voxel_count_before_cap=full.consensus_voxel_count_before_cap,
        consensus_voxel_count=full.consensus_voxel_count,
    )


def _select(b: F3Hypothesis, c: F3Hypothesis) -> F3Selector:
    if b.valid and c.valid and float(c.score) >= float(b.score) + _F_C_MIN_GAIN_OVER_B:
        chosen, reason = c, "C_valid_and_gain_at_least_0.03"
    elif b.valid:
        chosen, reason = b, "B_valid_C_not_better_by_0.03"
    elif c.valid:
        chosen, reason = c, "B_invalid_C_valid"
    else:
        return F3Selector(None, "neither_hypothesis_valid", None, None, None)
    assert chosen.world_q02 is not None and chosen.world_q98 is not None and chosen.score is not None
    return F3Selector(
        chosen.name,
        reason,
        chosen.world_q02,
        chosen.world_q98,
        chosen.score,
    )


def _track_receipt(
    track_id: int,
    observations: Tuple[F3Observation, ...],
    source_ids: Tuple[str, ...],
    frame_ids: Tuple[int, ...],
    frame_ordinals: Tuple[int, ...],
    seal_reason: str,
    b_fold_cache: Optional[Mapping[tuple[str, str], ProjectionFold]] = None,
    c_support_cache: Optional[Mapping[tuple[str, str], np.ndarray]] = None,
) -> tuple[
    F3TrackReceipt,
    Mapping[tuple[str, str], ProjectionFold],
    Mapping[tuple[str, str], np.ndarray],
]:
    mutable_b_cache = dict(b_fold_cache or {})
    mutable_c_cache = dict(c_support_cache or {})
    b = _best_single_view(observations, mutable_b_cache)
    c = _multi_view_consensus(observations, b, mutable_c_cache)
    selector = _select(b, c)
    receipt = F3TrackReceipt(
        track_id=track_id,
        source_ids=source_ids,
        frame_ids=frame_ids,
        frame_ordinals=frame_ordinals,
        observation_count=len(source_ids),
        retained_source_ids=tuple(item.source_id for item in observations),
        retained_frame_ids=tuple(item.frame_id for item in observations),
        retained_frame_ordinals=tuple(item.frame_ordinal for item in observations),
        retained_view_count=len(observations),
        total_observation_count=len(source_ids),
        confirmed=len(observations) >= _F_MIN_DISTINCT_FRAMES,
        hypothesis_b=b,
        hypothesis_c=c,
        selector=selector,
        max_logical_accessed_ordinal=max(
            frame_ordinals, default=0
        ),
        seal_reason=seal_reason,
    )
    return (
        receipt,
        MappingProxyType(mutable_b_cache),
        MappingProxyType(mutable_c_cache),
    )


class FastSAMOpenBoxF3ShadowTracker:
    """Fixed, bounded H0 tracker with exact query-before-commit semantics."""

    observer_only = True
    active_authorized = False

    def __init__(self) -> None:
        self._tracks: Dict[int, _Track] = {}
        self._sealed: Dict[int, F3TrackReceipt] = {}
        self._next_track_id = 0
        self._seen_source_ids: set[str] = set()
        self._last_frame_id: Optional[int] = None
        self._last_frame_ordinal: Optional[int] = None
        self._max_logical_accessed_ordinal: Optional[int] = None
        self._keyframe_count = 0
        self._serial = 0
        self._pending: Optional[_Pending] = None
        self._finalized: Optional[F3TerminalSeal] = None
        self._audit_complete = True

    def query(
        self,
        frame_id: object,
        frame_ordinal: object,
        observations: Sequence[F3Observation],
        *,
        max_logical_accessed_ordinal: object,
    ) -> F3FrameQuery:
        if self._finalized is not None:
            raise RuntimeError("F3 tracker has already been finalized")
        if self._pending is not None:
            raise RuntimeError("the previous F3 query must be committed")
        started = time.perf_counter_ns()
        normalized_frame = _strict_int("frame_id", frame_id)
        normalized_ordinal = _strict_int("frame_ordinal", frame_ordinal)
        max_access = _strict_int(
            "max_logical_accessed_ordinal", max_logical_accessed_ordinal
        )
        if self._last_frame_ordinal is not None and normalized_ordinal <= self._last_frame_ordinal:
            raise ValueError("frame_ordinal must be strictly increasing")
        if max_access > normalized_ordinal:
            raise ValueError("logical access cannot reach a future frame ordinal")
        if isinstance(observations, (str, bytes)) or not isinstance(observations, Sequence):
            raise ValueError("observations must be a sequence")
        rows = tuple(observations)
        for index, row in enumerate(rows):
            if not isinstance(row, F3Observation):
                raise ValueError(f"observations[{index}] must be F3Observation")
            if row.frame_id != normalized_frame or row.frame_ordinal != normalized_ordinal:
                raise ValueError("every observation must belong to the queried frame")
        source_ids = tuple(row.source_id for row in rows)
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("source_id values must be unique within a frame")
        if set(source_ids) & self._seen_source_ids:
            raise ValueError("source_id values must be globally unique")
        if rows and max_access < normalized_ordinal:
            raise ValueError("logical access receipt omits current source evidence")

        # Retire from a copy first.  Query cannot expose this prospective state.
        tracks = dict(self._tracks)
        sealed = dict(self._sealed)
        retired_ids = []
        step = self._keyframe_count
        for track_id in tuple(sorted(tracks)):
            track = tracks[track_id]
            if step - track.last_keyframe_step > _F_TTL_KEYFRAMES:
                retired_ids.append(track_id)
                sealed[track_id] = replace(track.latest_receipt, seal_reason="ttl")
                del tracks[track_id]

        prior_ids = tuple(sorted(tracks))
        edges = _association_edges(rows, prior_ids, tracks)
        used_tracks: set[int] = set()
        used_sources: set[str] = set()
        matched: dict[str, int] = {}
        for _, _, track_id, source_id, _ in edges:
            if track_id in used_tracks or source_id in used_sources:
                continue
            used_tracks.add(track_id)
            used_sources.add(source_id)
            matched[source_id] = track_id

        assignments = []
        next_track_id = self._next_track_id
        capacity_drop = False
        # Stable source order also makes capacity behaviour independent of
        # provider array order.
        for row in sorted(rows, key=lambda item: item.source_id):
            if row.source_id in matched:
                track_id = matched[row.source_id]
                previous = tracks[track_id]
                retained = previous.retained + (row,)
                if len(retained) > _F_MAX_RETAINED_OBSERVATIONS:
                    retained = retained[-_F_MAX_RETAINED_OBSERVATIONS:]
                source_lineage = previous.source_ids + (row.source_id,)
                frame_lineage = previous.frame_ids + (row.frame_id,)
                ordinal_lineage = previous.frame_ordinals + (row.frame_ordinal,)
                retained_ids = {item.source_id for item in retained}
                retained_b_cache = {
                    key: value
                    for key, value in previous.b_fold_cache.items()
                    if key[0] in retained_ids and key[1] in retained_ids
                }
                retained_c_cache = {
                    key: value
                    for key, value in previous.c_support_cache.items()
                    if key[0] in retained_ids and key[1] in retained_ids
                }
                receipt, b_fold_cache, c_support_cache = _track_receipt(
                    track_id,
                    retained,
                    source_lineage,
                    frame_lineage,
                    ordinal_lineage,
                    "live",
                    retained_b_cache,
                    retained_c_cache,
                )
                tracks[track_id] = _Track(
                    track_id,
                    step,
                    retained,
                    source_lineage,
                    frame_lineage,
                    ordinal_lineage,
                    b_fold_cache,
                    c_support_cache,
                    receipt,
                )
                assignments.append(F3Assignment(row.source_id, track_id, "matched"))
            elif len(tracks) < _F_MAX_LIVE_TRACKS and next_track_id <= _MAX_ID:
                track_id = next_track_id
                next_track_id += 1
                retained = (row,)
                source_lineage = (row.source_id,)
                frame_lineage = (row.frame_id,)
                ordinal_lineage = (row.frame_ordinal,)
                receipt, b_fold_cache, c_support_cache = _track_receipt(
                    track_id,
                    retained,
                    source_lineage,
                    frame_lineage,
                    ordinal_lineage,
                    "live",
                )
                tracks[track_id] = _Track(
                    track_id,
                    step,
                    retained,
                    source_lineage,
                    frame_lineage,
                    ordinal_lineage,
                    b_fold_cache,
                    c_support_cache,
                    receipt,
                )
                assignments.append(F3Assignment(row.source_id, track_id, "created"))
            else:
                capacity_drop = True
                assignments.append(F3Assignment(row.source_id, None, "dropped_capacity"))

        audit_complete = self._audit_complete and not capacity_drop
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        self._serial += 1
        public = F3FrameQuery(
            serial=self._serial,
            frame_id=normalized_frame,
            frame_ordinal=normalized_ordinal,
            history_max_frame_ordinal=self._last_frame_ordinal,
            max_logical_accessed_ordinal=max_access,
            prior_track_ids=prior_ids,
            assignments=tuple(assignments),
            retired_track_ids=tuple(retired_ids),
            elapsed_ms=elapsed_ms,
            audit_complete=audit_complete,
        )
        self._pending = _Pending(
            public=public,
            tracks=tracks,
            sealed=sealed,
            next_track_id=next_track_id,
            seen_source_ids=frozenset(self._seen_source_ids | set(source_ids)),
            audit_complete=audit_complete,
        )
        return public

    def commit(self, query: F3FrameQuery) -> F3FrameCommit:
        pending = self._pending
        if pending is None:
            raise RuntimeError("there is no pending F3 query")
        if query is not pending.public:
            raise ValueError("commit requires the exact pending query token")
        self._tracks = dict(pending.tracks)
        self._sealed = dict(pending.sealed)
        self._next_track_id = pending.next_track_id
        self._seen_source_ids = set(pending.seen_source_ids)
        self._last_frame_id = query.frame_id
        self._last_frame_ordinal = query.frame_ordinal
        self._max_logical_accessed_ordinal = (
            query.max_logical_accessed_ordinal
            if self._max_logical_accessed_ordinal is None
            else max(self._max_logical_accessed_ordinal, query.max_logical_accessed_ordinal)
        )
        self._keyframe_count += 1
        self._audit_complete = pending.audit_complete
        self._pending = None
        return F3FrameCommit(
            frame_id=query.frame_id,
            frame_ordinal=query.frame_ordinal,
            history_max_frame_ordinal=query.history_max_frame_ordinal,
            max_logical_accessed_ordinal=query.max_logical_accessed_ordinal,
            assignments=query.assignments,
            retired_track_ids=query.retired_track_ids,
            active_track_ids=tuple(sorted(self._tracks)),
            elapsed_ms=query.elapsed_ms,
            audit_complete=self._audit_complete,
        )

    def update(
        self,
        frame_id: object,
        frame_ordinal: object,
        observations: Sequence[F3Observation],
        *,
        max_logical_accessed_ordinal: object,
    ) -> F3FrameCommit:
        query = self.query(
            frame_id,
            frame_ordinal,
            observations,
            max_logical_accessed_ordinal=max_logical_accessed_ordinal,
        )
        return self.commit(query)

    def finalize(self) -> F3TerminalSeal:
        if self._pending is not None:
            raise RuntimeError("pending F3 query must be committed before finalize")
        if self._finalized is not None:
            return self._finalized
        sealed = dict(self._sealed)
        for track_id, track in self._tracks.items():
            sealed[track_id] = replace(track.latest_receipt, seal_reason="terminal")
        terminal = F3TerminalSeal(
            tracks=tuple(sealed[track_id] for track_id in sorted(sealed)),
            last_frame_id=self._last_frame_id,
            last_frame_ordinal=self._last_frame_ordinal,
            max_logical_accessed_ordinal=self._max_logical_accessed_ordinal,
            keyframe_count=self._keyframe_count,
            audit_complete=self._audit_complete,
        )
        self._finalized = terminal
        return terminal

    def summary(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "mode": MODE,
            "protocol_id": PROTOCOL_ID,
            "policy": dict(POLICY),
            "observer_only": True,
            "active_authorized": False,
            "native_mutation_applied": False,
            "birth_applied": False,
            "query_before_commit": True,
            "future_access": False,
            "last_frame_id": self._last_frame_id,
            "last_frame_ordinal": self._last_frame_ordinal,
            "max_logical_accessed_ordinal": self._max_logical_accessed_ordinal,
            "keyframe_count": self._keyframe_count,
            "active_track_count": len(self._tracks),
            "retired_track_count": len(self._sealed),
            "seen_source_count": len(self._seen_source_ids),
            "pending_frame_ordinal": (
                None if self._pending is None else self._pending.public.frame_ordinal
            ),
            "finalized": self._finalized is not None,
            "audit_complete": self._audit_complete,
        }


# Short alias for runners whose experiment name is already in their module.
OpenBoxProjectionTracker = FastSAMOpenBoxF3ShadowTracker


def _fold_to_dict(value: ProjectionFold) -> dict[str, object]:
    return {
        "heldout_source_id": value.heldout_source_id,
        "heldout_frame_id": value.heldout_frame_id,
        "heldout_frame_ordinal": value.heldout_frame_ordinal,
        "fitting_source_ids": list(value.fitting_source_ids),
        "valid": value.valid,
        "reason": value.reason,
        "projection_iou": value.projection_iou,
        "projected_xyxy": (
            None if value.projected_xyxy is None else value.projected_xyxy.tolist()
        ),
        "q02": None if value.world_q02 is None else value.world_q02.tolist(),
        "q98": None if value.world_q98 is None else value.world_q98.tolist(),
        "consensus_voxel_count_before_cap": value.consensus_voxel_count_before_cap,
        "consensus_voxel_count": value.consensus_voxel_count,
    }


def _b_candidate_to_dict(value: BCandidateEvaluation) -> dict[str, object]:
    return {
        "source_id": value.source_id,
        "frame_id": value.frame_id,
        "frame_ordinal": value.frame_ordinal,
        "q02": value.world_q02.tolist(),
        "q98": value.world_q98.tolist(),
        "score": value.score,
        "valid_fold_count": value.valid_fold_count,
        "fold_ious": [
            fold.projection_iou for fold in value.folds if fold.valid
        ],
        "folds": [_fold_to_dict(fold) for fold in value.folds],
    }


def _hypothesis_to_dict(value: F3Hypothesis) -> dict[str, object]:
    expose_geometry = value.valid
    q02 = None if not expose_geometry else value.world_q02.tolist()  # type: ignore[union-attr]
    q98 = None if not expose_geometry else value.world_q98.tolist()  # type: ignore[union-attr]
    center = (
        None
        if not expose_geometry
        else ((value.world_q02 + value.world_q98) * 0.5).tolist()  # type: ignore[operator]
    )
    extent = (
        None
        if not expose_geometry
        else (value.world_q98 - value.world_q02).tolist()  # type: ignore[operator]
    )
    return {
        "valid": value.valid,
        "available": value.available,
        "reason": value.reason,
        "q02": q02,
        "q98": q98,
        "center": center,
        "extent": extent,
        "score": value.score,
        "fold_ious": list(value.fold_ious),
        "valid_fold_count": value.valid_fold_count,
        "folds": [_fold_to_dict(fold) for fold in value.folds],
        "stability": {
            "pairwise_aabb_ious": list(value.stability_ious),
            "median_aabb_iou": value.stability_median_iou,
        },
        "source_id": value.source_id,
        "consensus_voxel_count_before_cap": value.consensus_voxel_count_before_cap,
        "consensus_voxel_count": value.consensus_voxel_count,
        "candidate_evaluations": [
            _b_candidate_to_dict(candidate) for candidate in value.b_candidates
        ],
    }


def track_receipt_to_dict(value: F3TrackReceipt) -> dict[str, object]:
    selector_center = (
        None
        if value.selector.world_q02 is None
        else ((value.selector.world_q02 + value.selector.world_q98) * 0.5).tolist()
    )
    selector_extent = (
        None
        if value.selector.world_q02 is None
        else (value.selector.world_q98 - value.selector.world_q02).tolist()
    )
    return {
        "track_id": value.track_id,
        "source_ids": list(value.source_ids),
        "frame_ids": list(value.frame_ids),
        "frame_ordinals": list(value.frame_ordinals),
        "observation_count": value.observation_count,
        "retained_source_ids": list(value.retained_source_ids),
        "retained_frame_ids": list(value.retained_frame_ids),
        "retained_frame_ordinals": list(value.retained_frame_ordinals),
        "retained_view_count": value.retained_view_count,
        "total_observation_count": value.total_observation_count,
        "confirmed": value.confirmed,
        "hypotheses": {
            "B": _hypothesis_to_dict(value.hypothesis_b),
            "C": _hypothesis_to_dict(value.hypothesis_c),
        },
        "selector": {
            "chosen": value.selector.chosen,
            "reason": value.selector.reason,
            "q02": (
                None if value.selector.world_q02 is None else value.selector.world_q02.tolist()
            ),
            "q98": (
                None if value.selector.world_q98 is None else value.selector.world_q98.tolist()
            ),
            "center": selector_center,
            "extent": selector_extent,
            "score": value.selector.score,
        },
        "max_logical_accessed_ordinal": value.max_logical_accessed_ordinal,
        "seal_reason": value.seal_reason,
    }


def frame_commit_to_dict(value: F3FrameCommit) -> dict[str, object]:
    return {
        "frame_id": value.frame_id,
        "frame_ordinal": value.frame_ordinal,
        "history_max_frame_ordinal": value.history_max_frame_ordinal,
        "max_logical_accessed_ordinal": value.max_logical_accessed_ordinal,
        "assignments": [
            {
                "source_id": row.source_id,
                "track_id": row.track_id,
                "action": row.action,
            }
            for row in value.assignments
        ],
        "retired_track_ids": list(value.retired_track_ids),
        "active_track_ids": list(value.active_track_ids),
        "elapsed_ms": value.elapsed_ms,
        "audit_complete": value.audit_complete,
    }


def terminal_seal_to_dict(value: F3TerminalSeal) -> dict[str, object]:
    return {
        "schema": value.schema,
        "mode": value.mode,
        "protocol_id": value.protocol_id,
        "policy": dict(POLICY),
        "complete": True,
        "observer_only": True,
        "active_authorized": False,
        "native_mutation_applied": False,
        "birth_applied": False,
        "last_frame_id": value.last_frame_id,
        "last_frame_ordinal": value.last_frame_ordinal,
        "max_logical_accessed_ordinal": value.max_logical_accessed_ordinal,
        "keyframe_count": value.keyframe_count,
        "track_count": len(value.tracks),
        "audit_complete": value.audit_complete,
        "tracks": [track_receipt_to_dict(track) for track in value.tracks],
    }


__all__ = [
    "B_MIN_PROJECTION_IOU",
    "C_EXTENT_RATIO_RANGE",
    "C_MAX_CENTER_SHIFT_M",
    "C_MIN_GAIN_OVER_B",
    "C_MIN_STABILITY_IOU",
    "C_VOLUME_RATIO_RANGE",
    "F3Assignment",
    "F3FrameCommit",
    "F3FrameQuery",
    "F3Hypothesis",
    "F3Observation",
    "F3Selector",
    "F3TerminalSeal",
    "F3TrackReceipt",
    "FastSAMOpenBoxF3ShadowTracker",
    "IMAGE_HEIGHT",
    "IMAGE_WIDTH",
    "MASK_PACKED_BYTES",
    "MAX_CONSENSUS_VOXELS",
    "MAX_LIVE_TRACKS",
    "MAX_RETAINED_OBSERVATIONS",
    "MAX_VOXELS_PER_OBSERVATION",
    "MIN_CONSENSUS_VOXELS",
    "MODE",
    "NEAR_PLANE_M",
    "OpenBoxProjectionTracker",
    "POLICY",
    "PROTOCOL_ID",
    "ProjectionFold",
    "SCHEMA",
    "TTL_KEYFRAMES",
    "VOXEL_SIZE_M",
    "frame_commit_to_dict",
    "make_observation",
    "projected_aabb_mask_iou",
    "terminal_seal_to_dict",
    "track_receipt_to_dict",
]
