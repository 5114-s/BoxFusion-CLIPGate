"""Pure-NumPy SAM2 mask lifting for the N0a shadow experiment.

This module converts one already-selected, current-frame SAM2 binary mask
into a bounded world-space geometry hypothesis.  It deliberately contains no
model provider, detector, semantic feature, ground truth, history, tracking,
or output mutation dependency.  The caller supplies and receives the exact
opaque F0 source identity so that a later runner can prove lineage without
changing the frozen F0 implementation.

The geometry contract mirrors the established F0 sensor geometry where
applicable, with one intentional N0a change: every signed 2 cm voxel is
represented by the centroid of all of its supporting world points rather than
its first point.  Quantiles are computed over *all* lexicographically ordered
voxel centroids.  The 2,048 point cap limits only sealed evidence and later
bounded state; it cannot change the fitted HS box.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


SCHEMA = "boxfusion.sam2_image_masklift_n0a.v1"
PROTOCOL_ID = "N0A-FROZEN-SAM2-IMAGE-BOXPROMPT-MASKLIFT-EXTRA100-SHADOW"

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
MIN_DEPTH_M = 0.10
MAX_DEPTH_M = 6.00
MASK_BOUNDARY_PX = 1
DEPTH_JUMP_M = 0.15
MIN_MASK_PIXELS = 200
MAX_MASK_PIXELS = 122_880
MIN_TIGHT_BOX_SIDE_PX = 16
MAX_TIGHT_BOX_ASPECT = 6.0
MIN_VALID_DEPTH_RATIO = 0.50
VOXEL_SIZE_M = 0.02
MIN_UNIQUE_VOXELS = 16
MAX_STORED_POINTS = 2_048
WORLD_QUANTILES = (0.02, 0.98)
MIN_AABB_EXTENT_M = 0.02
MASK_BITORDER = "little"
MASK_PACKED_BYTES = IMAGE_HEIGHT * IMAGE_WIDTH // 8

_MAX_SAFE_SCALED_COORDINATE = np.iinfo(np.int64).max / 4

POLICY: Mapping[str, object] = MappingProxyType(
    {
        "input_mask_shape": (IMAGE_HEIGHT, IMAGE_WIDTH),
        "mask_thresholded_upstream": True,
        "mask_pixels_inclusive": (MIN_MASK_PIXELS, MAX_MASK_PIXELS),
        "minimum_tight_box_side_px": MIN_TIGHT_BOX_SIDE_PX,
        "maximum_tight_box_aspect_inclusive": MAX_TIGHT_BOX_ASPECT,
        "minimum_valid_depth_ratio_inclusive": MIN_VALID_DEPTH_RATIO,
        "mask_boundary_px": MASK_BOUNDARY_PX,
        "mask_boundary_connectivity": 8,
        "valid_depth_m_inclusive": (MIN_DEPTH_M, MAX_DEPTH_M),
        "depth_jump_connectivity": 4,
        "depth_jump_m_strict": DEPTH_JUMP_M,
        "depth_jump_reject_both_endpoints": True,
        "voxel_size_m": VOXEL_SIZE_M,
        "voxel_quantization": "signed_floor",
        "voxel_representative": "centroid",
        "minimum_unique_voxels": MIN_UNIQUE_VOXELS,
        "maximum_stored_points": MAX_STORED_POINTS,
        "stored_point_order": "lexicographic_voxel_key_then_centroid",
        "stored_point_sampling": "integer_even_endpoints",
        "world_quantiles": WORLD_QUANTILES,
        "quantile_population": "all_lexicographic_voxel_centroids_before_cap",
        "minimum_aabb_extent_m": MIN_AABB_EXTENT_M,
        "mask_packbits_bitorder": MASK_BITORDER,
        "ground_truth": False,
        "semantics_or_clip": False,
        "native_prediction_access": False,
        "history_or_state": False,
        "training": False,
        "online_learning": False,
        "birth": False,
        "shadow_only": True,
    }
)


class N0AMaskLiftContractError(ValueError):
    """Raised when structural N0a input violates the fail-closed contract."""


def _readonly(
    value: object, dtype: np.dtype, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    """Return an immutable array backed by immutable owned bytes."""

    array = np.asarray(value, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise N0AMaskLiftContractError(
            f"array must have shape {shape}, got {array.shape}"
        )
    packed = np.array(array, dtype=dtype, order="C", copy=True).tobytes()
    return np.frombuffer(packed, dtype=dtype).reshape(array.shape)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise N0AMaskLiftContractError(
            "F0 source identity and H0 metadata must be finite JSON values"
        ) from error


def canonical_json_sha256(value: object) -> str:
    """Hash a finite JSON-compatible value with deterministic encoding."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _json_mapping_copy(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise N0AMaskLiftContractError(f"{label} must be a JSON mapping")
    if any(not isinstance(key, str) for key in value):
        raise N0AMaskLiftContractError(f"{label} keys must be strings")
    payload = _canonical_json_bytes(dict(value))
    clone = json.loads(payload.decode("utf-8"))
    if not isinstance(clone, dict):  # defensive: the input was checked above
        raise N0AMaskLiftContractError(f"{label} must be a JSON mapping")
    return MappingProxyType(clone)


def _sha256_array(value: np.ndarray, dtype: str) -> str:
    array = np.ascontiguousarray(value, dtype=np.dtype(dtype))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _joint_points_sha256(points: np.ndarray, keys: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(points, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(keys, dtype="<i8").tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class N0AAABB:
    """One normalized world AABB hypothesis."""

    name: str
    valid: bool
    q02: np.ndarray | None
    q98: np.ndarray | None
    center: np.ndarray | None
    extent: np.ndarray | None
    abstention_reason: str | None = None

    def __post_init__(self) -> None:
        if self.name not in {"H0", "HS"}:
            raise N0AMaskLiftContractError("AABB hypothesis name must be H0 or HS")
        if self.valid:
            if self.abstention_reason is not None:
                raise N0AMaskLiftContractError(
                    "a valid AABB cannot carry an abstention reason"
                )
            arrays = {}
            for field in ("q02", "q98", "center", "extent"):
                raw = getattr(self, field)
                if raw is None:
                    raise N0AMaskLiftContractError(
                        f"valid {self.name} is missing {field}"
                    )
                array = _readonly(raw, np.float64, (3,))
                if not np.isfinite(array).all():
                    raise N0AMaskLiftContractError(
                        f"valid {self.name}.{field} must be finite"
                    )
                arrays[field] = array
                object.__setattr__(self, field, array)
            lower = arrays["q02"]
            upper = arrays["q98"]
            if np.any(upper <= lower):
                raise N0AMaskLiftContractError(
                    f"valid {self.name} must have q98 > q02"
                )
            if not np.allclose(
                arrays["center"], (lower + upper) * 0.5, rtol=0.0, atol=1.0e-12
            ) or not np.allclose(
                arrays["extent"], upper - lower, rtol=0.0, atol=1.0e-12
            ):
                raise N0AMaskLiftContractError(
                    f"valid {self.name} center/extent must match q02/q98"
                )
            if np.any(arrays["extent"] < MIN_AABB_EXTENT_M - 1.0e-12):
                raise N0AMaskLiftContractError(
                    f"valid {self.name} violates the minimum AABB extent"
                )
        else:
            if not isinstance(self.abstention_reason, str) or not self.abstention_reason:
                raise N0AMaskLiftContractError(
                    "an invalid AABB requires an abstention reason"
                )
            if any(
                getattr(self, field) is not None
                for field in ("q02", "q98", "center", "extent")
            ):
                raise N0AMaskLiftContractError(
                    "an invalid AABB cannot expose fitted geometry"
                )

    def to_json(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "name": self.name,
            "valid": self.valid,
            "abstention_reason": self.abstention_reason,
        }
        if self.valid:
            row.update(
                {
                    "q02": self.q02.tolist(),
                    "q98": self.q98.tolist(),
                    "center": self.center.tolist(),
                    "extent": self.extent.tolist(),
                }
            )
        return row


@dataclass(frozen=True)
class N0AMaskLiftResult:
    """Immutable N0a shadow receipt and bounded point evidence."""

    f0_source_identity: Mapping[str, Any]
    f0_source_identity_sha256: str
    h0_input: Mapping[str, Any]
    h0_input_sha256: str
    h0: N0AAABB
    hs: N0AAABB
    mask_packbits: np.ndarray
    mask_sha256: str
    tight_box_xyxy: np.ndarray
    mask_pixel_count: int
    valid_depth_ratio: float
    interior_pixel_count: int
    metric_depth_pixel_count: int
    depth_jump_pixel_count: int
    support_pixel_count: int
    voxel_count: int
    stored_point_count: int
    quantile_point_count: int
    points_world: np.ndarray
    voxel_keys: np.ndarray
    points_and_voxel_keys_sha256: str
    input_sha256: str
    result_sha256: str
    valid: bool
    abstention_reason: str | None

    def __post_init__(self) -> None:
        identity = _json_mapping_copy(self.f0_source_identity, "f0_source_identity")
        h0_input = _json_mapping_copy(self.h0_input, "h0_input")
        object.__setattr__(self, "f0_source_identity", identity)
        object.__setattr__(self, "h0_input", h0_input)
        if identity.get("source_id") is None or not isinstance(
            identity.get("source_id"), str
        ):
            raise N0AMaskLiftContractError(
                "f0_source_identity.source_id must be a string"
            )
        count = int(self.stored_point_count)
        if count < 0 or count > MAX_STORED_POINTS:
            raise N0AMaskLiftContractError("stored point count is outside the cap")
        object.__setattr__(
            self,
            "mask_packbits",
            _readonly(self.mask_packbits, np.uint8, (MASK_PACKED_BYTES,)),
        )
        object.__setattr__(
            self, "tight_box_xyxy", _readonly(self.tight_box_xyxy, np.int64, (4,))
        )
        object.__setattr__(
            self, "points_world", _readonly(self.points_world, np.float64, (count, 3))
        )
        object.__setattr__(
            self, "voxel_keys", _readonly(self.voxel_keys, np.int64, (count, 3))
        )
        integer_counts = (
            "mask_pixel_count",
            "interior_pixel_count",
            "metric_depth_pixel_count",
            "depth_jump_pixel_count",
            "support_pixel_count",
            "voxel_count",
            "stored_point_count",
            "quantile_point_count",
        )
        for name in integer_counts:
            raw = getattr(self, name)
            if isinstance(raw, (bool, np.bool_)) or not isinstance(
                raw, (int, np.integer)
            ):
                raise N0AMaskLiftContractError(f"{name} must be an integer")
            if int(raw) < 0:
                raise N0AMaskLiftContractError(f"{name} cannot be negative")
            object.__setattr__(self, name, int(raw))
        if not isinstance(self.valid_depth_ratio, (int, float, np.integer, np.floating)):
            raise N0AMaskLiftContractError("valid_depth_ratio must be finite")
        ratio = float(self.valid_depth_ratio)
        if not math.isfinite(ratio) or not (0.0 <= ratio <= 1.0):
            raise N0AMaskLiftContractError("valid_depth_ratio must be in [0,1]")
        object.__setattr__(self, "valid_depth_ratio", ratio)
        if self.quantile_point_count != self.voxel_count:
            raise N0AMaskLiftContractError(
                "quantile population must equal the complete voxel centroid count"
            )
        if self.stored_point_count != len(self.points_world):
            raise N0AMaskLiftContractError("stored point count differs from evidence")
        if self.valid != self.hs.valid or self.abstention_reason != self.hs.abstention_reason:
            raise N0AMaskLiftContractError("top-level and HS validity differ")
        if self.valid and self.voxel_count < MIN_UNIQUE_VOXELS:
            raise N0AMaskLiftContractError("valid HS has too few unique voxels")
        if not self.valid and not self.abstention_reason:
            raise N0AMaskLiftContractError("invalid result requires an abstention reason")
        for name in (
            "f0_source_identity_sha256",
            "h0_input_sha256",
            "mask_sha256",
            "points_and_voxel_keys_sha256",
            "input_sha256",
            "result_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise N0AMaskLiftContractError(f"{name} must be a SHA-256 digest")

    @property
    def source_id(self) -> str:
        return str(self.f0_source_identity["source_id"])

    def to_receipt(self) -> dict[str, Any]:
        """Return a JSON receipt; raw packed masks and points remain sidecar arrays."""

        return {
            "schema": SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "mode": "shadow",
            "contracts": {
                "f0_source_identity_preserved": True,
                "ground_truth_access": False,
                "semantic_or_clip_access": False,
                "native_prediction_access": False,
                "history_or_state": False,
                "training": False,
                "online_learning": False,
                "birth_enabled": False,
                "native_output_mutation": False,
            },
            "f0_source_identity": dict(self.f0_source_identity),
            "f0_source_identity_sha256": self.f0_source_identity_sha256,
            "h0_input": dict(self.h0_input),
            "h0_input_sha256": self.h0_input_sha256,
            "hypotheses": {"H0": self.h0.to_json(), "HS": self.hs.to_json()},
            "mask": {
                "shape": [IMAGE_HEIGHT, IMAGE_WIDTH],
                "bitorder": MASK_BITORDER,
                "packed_byte_count": MASK_PACKED_BYTES,
                "sha256": self.mask_sha256,
                "tight_box_xyxy": self.tight_box_xyxy.tolist(),
                "pixel_count": self.mask_pixel_count,
                "valid_depth_ratio": self.valid_depth_ratio,
                "interior_pixel_count": self.interior_pixel_count,
                "metric_depth_pixel_count": self.metric_depth_pixel_count,
                "depth_jump_pixel_count": self.depth_jump_pixel_count,
                "support_pixel_count": self.support_pixel_count,
            },
            "points": {
                "voxel_size_m": VOXEL_SIZE_M,
                "voxel_representative": "centroid",
                "voxel_count": self.voxel_count,
                "quantile_point_count": self.quantile_point_count,
                "stored_point_count": self.stored_point_count,
                "maximum_stored_point_count": MAX_STORED_POINTS,
                "points_and_voxel_keys_sha256": self.points_and_voxel_keys_sha256,
            },
            "valid": self.valid,
            "abstention_reason": self.abstention_reason,
            "input_sha256": self.input_sha256,
            "result_sha256": self.result_sha256,
        }


def _validate_mask(value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.shape != (
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    ):
        raise N0AMaskLiftContractError(
            "selected_mask must be a binary numpy array with shape [480,640]"
        )
    if value.dtype.kind not in "biuf":
        raise N0AMaskLiftContractError(
            "selected_mask must be a binary numpy array with shape [480,640]"
        )
    try:
        mask = np.array(value, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise N0AMaskLiftContractError("selected_mask must be exactly binary") from error
    if mask.dtype.kind in "uf" and not np.isfinite(mask).all():
        raise N0AMaskLiftContractError("selected_mask must be exactly binary")
    if np.any((mask != 0) & (mask != 1)):
        raise N0AMaskLiftContractError("selected_mask must be exactly binary")
    return mask.astype(bool, copy=False)


def _validate_depth(value: object) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != (IMAGE_HEIGHT, IMAGE_WIDTH)
        or value.dtype.kind not in "iuf"
    ):
        raise N0AMaskLiftContractError(
            "depth_m must be a numeric numpy array with shape [480,640]"
        )
    try:
        return np.array(value, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise N0AMaskLiftContractError(
            "depth_m must be a numeric numpy array with shape [480,640]"
        ) from error


def _validate_intrinsics(value: object) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise N0AMaskLiftContractError(
            "intrinsics must have finite shape [3,3] or [4,4]"
        ) from error
    if matrix.shape == (4, 4):
        matrix = matrix[:3, :3]
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise N0AMaskLiftContractError(
            "intrinsics must have finite shape [3,3] or [4,4]"
        )
    if (
        matrix[0, 0] <= 0.0
        or matrix[1, 1] <= 0.0
        or abs(float(np.linalg.det(matrix))) <= 1.0e-12
        or not (0.0 <= matrix[0, 2] < IMAGE_WIDTH)
        or not (0.0 <= matrix[1, 2] < IMAGE_HEIGHT)
    ):
        raise N0AMaskLiftContractError(
            "intrinsics must be invertible and registered to 480x640"
        )
    return np.array(matrix, dtype=np.float64, order="C", copy=True)


def _validate_pose(value: object) -> np.ndarray:
    try:
        pose = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise N0AMaskLiftContractError(
            "camera_to_world must be a finite rigid [4,4] matrix"
        ) from error
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise N0AMaskLiftContractError(
            "camera_to_world must be a finite rigid [4,4] matrix"
        )
    if np.max(np.abs(pose[3] - [0.0, 0.0, 0.0, 1.0])) > 1.0e-7:
        raise N0AMaskLiftContractError(
            "camera_to_world must be a finite rigid [4,4] matrix"
        )
    rotation = pose[:3, :3]
    if (
        np.max(np.abs(rotation.T @ rotation - np.eye(3))) > 1.0e-4
        or abs(float(np.linalg.det(rotation)) - 1.0) > 1.0e-4
    ):
        raise N0AMaskLiftContractError(
            "camera_to_world rotation must be orthonormal and right handed"
        )
    return np.array(pose, dtype=np.float64, order="C", copy=True)


def _aabb_from_bounds(name: str, lower: np.ndarray, upper: np.ndarray) -> N0AAABB:
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    center = (lower + upper) * 0.5
    extent = upper - lower
    return N0AAABB(
        name=name,
        valid=True,
        q02=lower,
        q98=upper,
        center=center,
        extent=extent,
    )


def _normalize_h0(value: object) -> tuple[Mapping[str, Any], N0AAABB]:
    source = _json_mapping_copy(value, "h0")
    if source.get("valid", True) is not True:
        raise N0AMaskLiftContractError("H0 must be valid")
    lower_raw = source.get("q02", source.get("world_q02"))
    upper_raw = source.get("q98", source.get("world_q98"))
    try:
        lower = np.asarray(lower_raw, dtype=np.float64)
        upper = np.asarray(upper_raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise N0AMaskLiftContractError("H0 must contain finite q02/q98 vectors") from error
    if (
        lower.shape != (3,)
        or upper.shape != (3,)
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(upper - lower < MIN_AABB_EXTENT_M - 1.0e-12)
    ):
        raise N0AMaskLiftContractError(
            "H0 must contain finite q02/q98 with minimum extent 0.02 m"
        )
    expected_center = (lower + upper) * 0.5
    expected_extent = upper - lower
    for keys, expected, label in (
        (("center", "world_center"), expected_center, "center"),
        (("extent", "world_extent"), expected_extent, "extent"),
    ):
        raw = next((source[key] for key in keys if key in source), None)
        if raw is None:
            continue
        try:
            array = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise N0AMaskLiftContractError(f"H0 {label} is invalid") from error
        if array.shape != (3,) or not np.allclose(
            array, expected, rtol=0.0, atol=1.0e-12
        ):
            raise N0AMaskLiftContractError(f"H0 {label} differs from q02/q98")
    return source, _aabb_from_bounds("H0", lower, upper)


def _erode_one_pixel(mask: np.ndarray) -> np.ndarray:
    """Binary 3x3 erosion with false outside the image, implemented in NumPy."""

    interior = np.zeros(mask.shape, dtype=bool)
    if mask.shape[0] < 3 or mask.shape[1] < 3:
        return interior
    core = mask[1:-1, 1:-1].copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            core &= mask[1 + dy : mask.shape[0] - 1 + dy, 1 + dx : mask.shape[1] - 1 + dx]
    interior[1:-1, 1:-1] = core
    return interior


def _depth_edge_mask(depth: np.ndarray, valid_depth: np.ndarray) -> np.ndarray:
    """Mark both endpoints of every valid 4-neighbour jump strictly over 15 cm."""

    edge = np.zeros(depth.shape, dtype=bool)
    with np.errstate(invalid="ignore", over="ignore"):
        horizontal = valid_depth[:, :-1] & valid_depth[:, 1:] & (
            np.abs(depth[:, :-1] - depth[:, 1:]) > DEPTH_JUMP_M
        )
        vertical = valid_depth[:-1, :] & valid_depth[1:, :] & (
            np.abs(depth[:-1, :] - depth[1:, :]) > DEPTH_JUMP_M
        )
    edge[:, :-1] |= horizontal
    edge[:, 1:] |= horizontal
    edge[:-1, :] |= vertical
    edge[1:, :] |= vertical
    return edge


def _voxel_centroids(
    points_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lexicographic signed-floor keys and deterministic centroids."""

    scaled = points_world / VOXEL_SIZE_M
    if not np.isfinite(scaled).all() or (
        scaled.size
        and float(np.max(np.abs(scaled))) > _MAX_SAFE_SCALED_COORDINATE
    ):
        raise N0AMaskLiftContractError(
            "world point range is unsafe for signed 2 cm voxel quantization"
        )
    keys = np.floor(scaled).astype(np.int64)
    # ``lexsort`` is stable.  Sorting only by the voxel key therefore makes
    # keys lexicographic while preserving the original row-major support-point
    # order inside every key.  The latter is part of the frozen float64
    # centroid contract; adding coordinate tie-breaks would change last-bit
    # sums for some masks.
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    sorted_points = points_world[order]
    starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)).astype(
                np.int64
            )
            + 1,
        )
    )
    unique_keys = sorted_keys[starts]
    counts = np.diff(np.append(starts, len(sorted_keys))).astype(np.float64)
    sums = np.add.reduceat(sorted_points, starts, axis=0)
    centroids = sums / counts[:, None]
    return unique_keys, centroids


def _stored_indices(count: int) -> np.ndarray:
    if count <= MAX_STORED_POINTS:
        return np.arange(count, dtype=np.int64)
    return (
        np.arange(MAX_STORED_POINTS, dtype=np.int64) * (count - 1)
    ) // (MAX_STORED_POINTS - 1)


def _invalid_hs(reason: str) -> N0AAABB:
    return N0AAABB(
        name="HS",
        valid=False,
        q02=None,
        q98=None,
        center=None,
        extent=None,
        abstention_reason=reason,
    )


def _fit_hs(centroids: np.ndarray) -> N0AAABB:
    raw_q02, raw_q98 = np.quantile(
        centroids,
        WORLD_QUANTILES,
        axis=0,
        method="linear",
    )
    center = (raw_q02 + raw_q98) * 0.5
    extent = np.maximum(raw_q98 - raw_q02, MIN_AABB_EXTENT_M)
    return _aabb_from_bounds("HS", center - extent * 0.5, center + extent * 0.5)


def _input_hash(
    *,
    identity_sha: str,
    h0_sha: str,
    mask_sha: str,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    pose: np.ndarray,
) -> str:
    return canonical_json_sha256(
        {
            "f0_source_identity_sha256": identity_sha,
            "h0_input_sha256": h0_sha,
            "mask_sha256": mask_sha,
            "depth_sha256": _sha256_array(depth, "<f8"),
            "intrinsics_sha256": _sha256_array(intrinsics, "<f8"),
            "camera_to_world_sha256": _sha256_array(pose, "<f8"),
        }
    )


def _result_payload(
    *,
    identity: Mapping[str, Any],
    identity_sha: str,
    h0_input: Mapping[str, Any],
    h0_sha: str,
    h0: N0AAABB,
    hs: N0AAABB,
    mask_sha: str,
    tight_box_xyxy: np.ndarray,
    mask_pixel_count: int,
    valid_depth_ratio: float,
    interior_pixel_count: int,
    metric_depth_pixel_count: int,
    depth_jump_pixel_count: int,
    support_pixel_count: int,
    voxel_count: int,
    stored_point_count: int,
    points_sha: str,
    input_sha: str,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "f0_source_identity": dict(identity),
        "f0_source_identity_sha256": identity_sha,
        "h0_input": dict(h0_input),
        "h0_input_sha256": h0_sha,
        "hypotheses": {"H0": h0.to_json(), "HS": hs.to_json()},
        "mask": {
            "shape": [IMAGE_HEIGHT, IMAGE_WIDTH],
            "bitorder": MASK_BITORDER,
            "packed_byte_count": MASK_PACKED_BYTES,
            "sha256": mask_sha,
            "tight_box_xyxy": tight_box_xyxy.tolist(),
            "pixel_count": mask_pixel_count,
            "valid_depth_ratio": valid_depth_ratio,
            "interior_pixel_count": interior_pixel_count,
            "metric_depth_pixel_count": metric_depth_pixel_count,
            "depth_jump_pixel_count": depth_jump_pixel_count,
            "support_pixel_count": support_pixel_count,
        },
        "points": {
            "voxel_count": voxel_count,
            "quantile_point_count": voxel_count,
            "stored_point_count": stored_point_count,
            "points_and_voxel_keys_sha256": points_sha,
        },
        "valid": hs.valid,
        "abstention_reason": hs.abstention_reason,
        "input_sha256": input_sha,
    }


def lift_sam2_mask(
    *,
    f0_source_identity: Mapping[str, Any],
    selected_mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    h0: Mapping[str, Any],
) -> N0AMaskLiftResult:
    """Lift one selected SAM2 mask into an output-inert HS hypothesis.

    Structural input errors raise :class:`N0AMaskLiftContractError`.  Ordinary
    absence of sufficient sensor support returns a complete invalid HS receipt
    whose ``abstention_reason`` explains the failure.  H0 and the opaque F0
    source identity are always copied into the result without mutation.
    """

    identity = _json_mapping_copy(f0_source_identity, "f0_source_identity")
    if not isinstance(identity.get("source_id"), str) or not identity["source_id"]:
        raise N0AMaskLiftContractError(
            "f0_source_identity.source_id must be a non-empty string"
        )
    identity_sha = canonical_json_sha256(dict(identity))
    h0_input, h0_geometry = _normalize_h0(h0)
    h0_sha = canonical_json_sha256(dict(h0_input))
    mask = _validate_mask(selected_mask)
    depth = _validate_depth(depth_m)
    intrinsic = _validate_intrinsics(intrinsics)
    pose = _validate_pose(camera_to_world)

    packed_mask = np.packbits(mask.reshape(-1), bitorder=MASK_BITORDER).astype(
        np.uint8, copy=False
    )
    if packed_mask.shape != (MASK_PACKED_BYTES,):
        raise N0AMaskLiftContractError("packed SAM2 mask has an invalid size")
    mask_sha = hashlib.sha256(packed_mask.tobytes()).hexdigest()
    input_sha = _input_hash(
        identity_sha=identity_sha,
        h0_sha=h0_sha,
        mask_sha=mask_sha,
        depth=depth,
        intrinsics=intrinsic,
        pose=pose,
    )

    mask_pixel_count = int(np.count_nonzero(mask))
    if mask_pixel_count:
        mask_rows, mask_cols = np.nonzero(mask)
        tight_box = np.asarray(
            [
                int(mask_cols.min()),
                int(mask_rows.min()),
                int(mask_cols.max()),
                int(mask_rows.max()),
            ],
            dtype=np.int64,
        )
        tight_width = int(tight_box[2] - tight_box[0] + 1)
        tight_height = int(tight_box[3] - tight_box[1] + 1)
        tight_aspect = float(
            max(tight_width, tight_height) / min(tight_width, tight_height)
        )
    else:
        tight_box = np.full(4, -1, dtype=np.int64)
        tight_width = 0
        tight_height = 0
        tight_aspect = math.inf
    interior = _erode_one_pixel(mask)
    interior_pixel_count = int(np.count_nonzero(interior))
    with np.errstate(invalid="ignore"):
        valid_depth = (
            np.isfinite(depth) & (depth >= MIN_DEPTH_M) & (depth <= MAX_DEPTH_M)
        )
    metric_depth_pixel_count = int(np.count_nonzero(mask & valid_depth))
    valid_depth_ratio = (
        metric_depth_pixel_count / mask_pixel_count if mask_pixel_count else 0.0
    )
    depth_edges = _depth_edge_mask(depth, valid_depth)
    depth_jump_pixel_count = int(np.count_nonzero(interior & depth_edges))
    support = interior & valid_depth & ~depth_edges
    rows, cols = np.nonzero(support)
    support_pixel_count = int(len(rows))

    if mask_pixel_count < MIN_MASK_PIXELS:
        reason = "mask_pixel_count_below_200"
    elif mask_pixel_count > MAX_MASK_PIXELS:
        reason = "mask_pixel_count_above_122880"
    elif min(tight_width, tight_height) < MIN_TIGHT_BOX_SIDE_PX:
        reason = "tight_box_side_below_16"
    elif tight_aspect > MAX_TIGHT_BOX_ASPECT:
        reason = "tight_box_aspect_above_6"
    elif valid_depth_ratio < MIN_VALID_DEPTH_RATIO:
        reason = "valid_depth_ratio_below_0_50"
    elif interior_pixel_count == 0:
        reason = "empty_after_one_pixel_boundary_removal"
    elif metric_depth_pixel_count == 0:
        reason = "no_metric_depth_in_mask"
    elif support_pixel_count == 0:
        reason = "empty_after_depth_jump_removal"
    else:
        reason = None

    all_keys = np.empty((0, 3), dtype=np.int64)
    all_centroids = np.empty((0, 3), dtype=np.float64)
    if reason is None:
        pixels = np.column_stack(
            (
                cols.astype(np.float64),
                rows.astype(np.float64),
                np.ones(support_pixel_count, dtype=np.float64),
            )
        )
        inverse_intrinsic = np.linalg.inv(intrinsic)
        rays = pixels @ inverse_intrinsic.T
        if not np.isfinite(rays).all() or np.any(np.abs(rays[:, 2]) <= 1.0e-12):
            raise N0AMaskLiftContractError(
                "intrinsics produced invalid backprojection rays"
            )
        rays /= rays[:, 2:3]
        points_camera = rays * depth[rows, cols, None]
        points_world = points_camera @ pose[:3, :3].T + pose[:3, 3]
        all_keys, all_centroids = _voxel_centroids(points_world)
        if len(all_keys) < MIN_UNIQUE_VOXELS:
            reason = "fewer_than_16_unique_voxels"

    indices = _stored_indices(len(all_keys))
    stored_keys = all_keys[indices]
    stored_points = all_centroids[indices]
    points_sha = _joint_points_sha256(stored_points, stored_keys)
    hs = _invalid_hs(reason) if reason is not None else _fit_hs(all_centroids)

    payload = _result_payload(
        identity=identity,
        identity_sha=identity_sha,
        h0_input=h0_input,
        h0_sha=h0_sha,
        h0=h0_geometry,
        hs=hs,
        mask_sha=mask_sha,
        tight_box_xyxy=tight_box,
        mask_pixel_count=mask_pixel_count,
        valid_depth_ratio=valid_depth_ratio,
        interior_pixel_count=interior_pixel_count,
        metric_depth_pixel_count=metric_depth_pixel_count,
        depth_jump_pixel_count=depth_jump_pixel_count,
        support_pixel_count=support_pixel_count,
        voxel_count=len(all_keys),
        stored_point_count=len(stored_keys),
        points_sha=points_sha,
        input_sha=input_sha,
    )
    result_sha = canonical_json_sha256(payload)
    return N0AMaskLiftResult(
        f0_source_identity=identity,
        f0_source_identity_sha256=identity_sha,
        h0_input=h0_input,
        h0_input_sha256=h0_sha,
        h0=h0_geometry,
        hs=hs,
        mask_packbits=packed_mask,
        mask_sha256=mask_sha,
        tight_box_xyxy=tight_box,
        mask_pixel_count=mask_pixel_count,
        valid_depth_ratio=valid_depth_ratio,
        interior_pixel_count=interior_pixel_count,
        metric_depth_pixel_count=metric_depth_pixel_count,
        depth_jump_pixel_count=depth_jump_pixel_count,
        support_pixel_count=support_pixel_count,
        voxel_count=len(all_keys),
        stored_point_count=len(stored_keys),
        quantile_point_count=len(all_keys),
        points_world=stored_points,
        voxel_keys=stored_keys,
        points_and_voxel_keys_sha256=points_sha,
        input_sha256=input_sha,
        result_sha256=result_sha,
        valid=hs.valid,
        abstention_reason=hs.abstention_reason,
    )


__all__ = [
    "DEPTH_JUMP_M",
    "IMAGE_HEIGHT",
    "IMAGE_WIDTH",
    "MASK_BITORDER",
    "MASK_PACKED_BYTES",
    "MAX_MASK_PIXELS",
    "MAX_DEPTH_M",
    "MAX_STORED_POINTS",
    "MAX_TIGHT_BOX_ASPECT",
    "MIN_AABB_EXTENT_M",
    "MIN_DEPTH_M",
    "MIN_MASK_PIXELS",
    "MIN_TIGHT_BOX_SIDE_PX",
    "MIN_UNIQUE_VOXELS",
    "MIN_VALID_DEPTH_RATIO",
    "N0AAABB",
    "N0AMaskLiftContractError",
    "N0AMaskLiftResult",
    "POLICY",
    "PROTOCOL_ID",
    "SCHEMA",
    "VOXEL_SIZE_M",
    "WORLD_QUANTILES",
    "canonical_json_sha256",
    "lift_sam2_mask",
]
