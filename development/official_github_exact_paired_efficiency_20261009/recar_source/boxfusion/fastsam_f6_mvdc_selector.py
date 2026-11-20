"""Frozen GT-free past-only multi-view depth/projection selector for F6.

F6 observes the sealed F2/F4 evidence for one source, associates its H0 box
with sources in the previous three *committed* successful frames, and, when
two distinct past views are available, compares the copied H0/HL/HLG/HB
geometries using the preregistered depth, face-residual and convex-hull mask
metrics.  It is a shadow observer: it preserves the source census and formal
score and exposes no birth or native-prediction mutation API.

The public state deliberately uses query-before-commit semantics.  A query is
computed from an immutable past snapshot; only the exact returned token may be
committed.  Committed array payload is bounded to three frames, sixteen
sources per frame and 2.5 MiB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re
import time
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import numpy as np


SCHEMA = "boxfusion.fastsam_f6_mvdc_selector.v1"
PROTOCOL_ID = "F6-GT-FREE-PAST-ONLY-MULTIVIEW-DEPTH-PROJECTION-SELECTOR-PAPER100"
MODE = "shadow"

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
MASK_PACKED_BYTES = IMAGE_HEIGHT * IMAGE_WIDTH // 8
MAX_SAMPLES = 256
MAX_BUFFERED_SUCCESSFUL_FRAMES = 3
MAX_SOURCES_PER_FRAME = 16
MAX_RAW_ARRAY_PAYLOAD_BYTES = int(2.5 * 1024 * 1024)
NEAR_PLANE_M = 1.0e-4
ROBUST_MARGIN_M = 0.05

BASE_MIN_RETAINED_POINTS = 16
BASE_MIN_RETAINED_FRACTION = 0.55
BASE_VOLUME_RATIO = (0.25, 1.05)
BASE_EXTENT_RATIO = (0.35, 1.05)
BASE_CENTER_DIAGONAL_FRACTION = 0.20
BASE_CENTER_ABSOLUTE_MARGIN_M = 0.05

ASSOCIATION_ND_MAX = 0.50
ASSOCIATION_IOU_MIN = 0.15
ASSOCIATION_CONTAINMENT_MIN = 0.60
MIN_PAST_MATCHES = 2

CANDIDATE_ND_MAX = 0.50
CANDIDATE_VOLUME_RATIO = (0.25, 4.00)
CANDIDATE_IOU_MIN = 0.20
CANDIDATE_CONTAINMENT_MIN = 0.70
MIN_POINTS_PER_VIEW = 16
EXACT_SUPPORT_MIN = 0.60
EXPANDED_SUPPORT_MIN = 0.80
MIN_SUPPORTING_VIEWS = 2

DEPTH_WIN_MARGIN_M = 0.05
PROJECTION_WIN_MARGIN = 0.10
CONTAINMENT_WIN_MARGIN = 0.10
DEPTH_MAX_REGRESSION_M = 0.025
PROJECTION_MAX_REGRESSION = 0.05
CONTAINMENT_MAX_REGRESSION = 0.05
MIN_METRIC_WINS = 2

_SOURCE_RE = re.compile(
    r"^(?P<scene>scene[0-9]{4}_[0-9]{2})/"
    r"frame_(?P<frame>[0-9]{6})/raw_(?P<raw>[0-9]{3})$"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CORNER_SIGNS = np.asarray(
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
_CORNER_SIGNS.setflags(write=False)
_TIE_PRIORITY = {"H0": 0, "HL": 1, "HLG": 2, "HB": 3}
_BYTE_BITS_LITTLE = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None], axis=1, bitorder="little"
)
_BYTE_POPCOUNT = _BYTE_BITS_LITTLE.sum(axis=1, dtype=np.uint8)
_BYTE_SET_BIT_POSITIONS = np.full((256, 8), -1, dtype=np.int8)
for _byte_value in range(256):
    _positions = np.flatnonzero(_BYTE_BITS_LITTLE[_byte_value])
    _BYTE_SET_BIT_POSITIONS[_byte_value, : len(_positions)] = _positions
_BYTE_BITS_LITTLE.setflags(write=False)
_BYTE_POPCOUNT.setflags(write=False)
_BYTE_SET_BIT_POSITIONS.setflags(write=False)
_GRID_CENTRES_16 = np.arange(16, dtype=np.float64) + 0.5
_GRID_CENTRES_16.setflags(write=False)
_GRID_CELL_CENTRES_16X16 = np.column_stack(
    (
        np.tile(_GRID_CENTRES_16, 16),
        np.repeat(_GRID_CENTRES_16, 16),
    )
)
_GRID_CELL_CENTRES_16X16.setflags(write=False)

POLICY: Mapping[str, object] = MappingProxyType(
    {
        "protocol_id": PROTOCOL_ID,
        "mode": MODE,
        "hypotheses": ("H0", "HL", "HLG", "HB"),
        "base_candidate_order": ("HLG", "HL", "H0"),
        "exact_tie_priority": ("H0", "HL", "HLG", "HB"),
        "sample_cap": MAX_SAMPLES,
        "sampling": "floor((j+0.5)*N/256)",
        "mask_shape": (IMAGE_HEIGHT, IMAGE_WIDTH),
        "mask_bitorder": "little",
        "association": "H0_only_per_past_frame_mutual_best",
        "association_affinity": ("IoU3D", "SC", "-ND"),
        "association_nd_max": ASSOCIATION_ND_MAX,
        "association_iou_min": ASSOCIATION_IOU_MIN,
        "association_containment_min": ASSOCIATION_CONTAINMENT_MIN,
        "past_matches": MIN_PAST_MATCHES,
        "max_buffered_successful_frames": MAX_BUFFERED_SUCCESSFUL_FRAMES,
        "max_sources_per_frame": MAX_SOURCES_PER_FRAME,
        "max_raw_array_payload_bytes": MAX_RAW_ARRAY_PAYLOAD_BYTES,
        "robust_margin_m": ROBUST_MARGIN_M,
        "candidate_nd_max": CANDIDATE_ND_MAX,
        "candidate_volume_ratio": CANDIDATE_VOLUME_RATIO,
        "candidate_iou_min": CANDIDATE_IOU_MIN,
        "candidate_containment_min": CANDIDATE_CONTAINMENT_MIN,
        "minimum_points_per_view": MIN_POINTS_PER_VIEW,
        "exact_support_min": EXACT_SUPPORT_MIN,
        "expanded_support_min": EXPANDED_SUPPORT_MIN,
        "minimum_supporting_views": MIN_SUPPORTING_VIEWS,
        "depth_win_margin_m": DEPTH_WIN_MARGIN_M,
        "projection_win_margin": PROJECTION_WIN_MARGIN,
        "containment_win_margin": CONTAINMENT_WIN_MARGIN,
        "depth_max_regression_m": DEPTH_MAX_REGRESSION_M,
        "projection_max_regression": PROJECTION_MAX_REGRESSION,
        "containment_max_regression": CONTAINMENT_MAX_REGRESSION,
        "minimum_metric_wins": MIN_METRIC_WINS,
        "formal_score": 1.0,
        "maximum_lookahead_frames": 0,
        "observer_only": True,
        "birth": False,
        "native_output_mutation": False,
        "ground_truth": False,
        "annotation": False,
        "evaluator": False,
        "native_prediction_access": False,
        "semantics": False,
        "training": False,
        "online_learning": False,
        "cuda_allocation": False,
        "query_before_commit": True,
    }
)


class F6ContractError(RuntimeError):
    """Raised on any sealed-input, causality or bounded-state violation."""


def _json_value(value: object) -> object:
    """Convert NumPy and read-only containers to finite JSON primitives."""

    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        result = float(value)
        if not math.isfinite(result):
            raise F6ContractError("canonical JSON contains a non-finite number")
        return result
    if isinstance(value, float):
        if not math.isfinite(value):
            raise F6ContractError("canonical JSON contains a non-finite number")
        return value
    if value is None or isinstance(value, (str, int)):
        return value
    raise F6ContractError(f"unsupported canonical JSON type: {type(value).__name__}")


def canonical_json_sha256(value: object) -> str:
    # Results and tokens are built exclusively from ordinary JSON containers;
    # let the C encoder take that high-rate path.  Read-only MappingProxyType
    # evidence takes the audited normalization fallback below.
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError):
        try:
            payload = json.dumps(
                _json_value(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        except (TypeError, ValueError, UnicodeEncodeError) as error:
            raise F6ContractError("value is not canonical finite ASCII JSON") from error
    return hashlib.sha256(payload).hexdigest()


def canonical_result_sha256(row: Mapping[str, Any]) -> str:
    payload = dict(row)
    payload.pop("result_sha256", None)
    return canonical_json_sha256(payload)


def _strict_int(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise F6ContractError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise F6ContractError(f"{label} must be >= {minimum}")
    return result


def _finite_scalar(value: object, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise F6ContractError(f"{label} must be one finite scalar")
    result = float(value)
    if not math.isfinite(result):
        raise F6ContractError(f"{label} must be one finite scalar")
    return result


def _readonly_array(value: object, dtype: np.dtype, shape: Optional[tuple[int, ...]], label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as error:
        raise F6ContractError(f"{label} has an invalid array domain") from error
    if shape is not None and array.shape != shape:
        raise F6ContractError(f"{label} must have shape {shape}")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise F6ContractError(f"{label} must be finite")
    packed = np.ascontiguousarray(array).tobytes()
    return np.frombuffer(packed, dtype=dtype).reshape(array.shape)


def _deep_freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, np.ndarray):
        return _deep_freeze_json(value.tolist())
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze_json(item) for item in value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        if not math.isfinite(result):
            raise F6ContractError("hypothesis JSON contains a non-finite number")
        return result
    if value is None or isinstance(value, (str, int)):
        return value
    raise F6ContractError(f"unsupported hypothesis JSON type: {type(value).__name__}")


def _sample_indices(count: int) -> np.ndarray:
    if count <= MAX_SAMPLES:
        return np.arange(count, dtype=np.int64)
    j = np.arange(MAX_SAMPLES, dtype=np.float64)
    indices = np.floor((j + 0.5) * float(count) / float(MAX_SAMPLES)).astype(np.int64)
    return indices


def _sample_points(value: object) -> tuple[np.ndarray, int]:
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise F6ContractError("points_world must be a finite [N,3] array") from error
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise F6ContractError("points_world must be a finite [N,3] array")
    count = int(len(points))
    sampled = points[_sample_indices(count)]
    return _readonly_array(sampled, np.float64, None, "sampled points"), count


def _mask_and_pixels(value: object) -> tuple[np.ndarray, np.ndarray, int]:
    packed = np.asarray(value)
    if packed.dtype != np.uint8 or packed.shape != (MASK_PACKED_BYTES,):
        raise F6ContractError("mask_packbits must be uint8 with exactly 38400 bytes")
    immutable = _readonly_array(packed, np.uint8, (MASK_PACKED_BYTES,), "mask_packbits")
    nonzero_byte_indices = np.flatnonzero(immutable)
    nonzero_bytes = immutable[nonzero_byte_indices]
    byte_counts = _BYTE_POPCOUNT[nonzero_bytes].astype(np.int64, copy=False)
    cumulative = np.cumsum(byte_counts, dtype=np.int64)
    count = int(cumulative[-1]) if len(cumulative) else 0
    if count == 0:
        raise F6ContractError("mask_packbits must contain at least one positive pixel")
    retained_ordinals = _sample_indices(count)
    selected_nonzero_positions = np.searchsorted(
        cumulative, retained_ordinals, side="right"
    )
    preceding = np.zeros_like(retained_ordinals)
    has_preceding = selected_nonzero_positions > 0
    preceding[has_preceding] = cumulative[
        selected_nonzero_positions[has_preceding] - 1
    ]
    within_byte_rank = retained_ordinals - preceding
    selected_bytes = nonzero_bytes[selected_nonzero_positions]
    bit_offsets = _BYTE_SET_BIT_POSITIONS[selected_bytes, within_byte_rank]
    if np.any(bit_offsets < 0):  # defensive table-integrity assertion
        raise F6ContractError("packed-mask set-bit lookup failed")
    retained = (
        nonzero_byte_indices[selected_nonzero_positions] * 8
        + bit_offsets.astype(np.int64)
    )
    y = retained // IMAGE_WIDTH
    x = retained % IMAGE_WIDTH
    pixels_yx = np.column_stack((y, x)).astype(np.int16, copy=False)
    return immutable, _readonly_array(pixels_yx, np.int16, None, "sampled mask pixels"), count


def _validate_pose(value: object) -> np.ndarray:
    pose = _readonly_array(value, np.float64, (4, 4), "camera_to_world")
    if not np.allclose(pose[3], np.asarray([0.0, 0.0, 0.0, 1.0]), rtol=0.0, atol=1.0e-7):
        raise F6ContractError("camera_to_world must be affine")
    try:
        inverse = np.linalg.inv(pose)
    except np.linalg.LinAlgError as error:
        raise F6ContractError("camera_to_world must be invertible") from error
    if not np.isfinite(inverse).all():
        raise F6ContractError("camera_to_world inverse must be finite")
    return pose


def _validate_intrinsic(value: object) -> np.ndarray:
    intrinsic = np.asarray(value, dtype=np.float64)
    if intrinsic.shape == (4, 4):
        intrinsic = intrinsic[:3, :3]
    result = _readonly_array(intrinsic, np.float64, (3, 3), "intrinsic")
    if result[0, 0] <= 0.0 or result[1, 1] <= 0.0:
        raise F6ContractError("intrinsic focal lengths must be positive")
    try:
        inverse = np.linalg.inv(result)
    except np.linalg.LinAlgError as error:
        raise F6ContractError("intrinsic must be invertible") from error
    if not np.isfinite(inverse).all():
        raise F6ContractError("intrinsic inverse must be finite")
    return result


def _aabb_from_row(row: object, label: str) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(row, Mapping) or row.get("valid") is not True:
        raise F6ContractError(f"{label} must be a valid sealed AABB hypothesis")
    lower = _readonly_array(row.get("q02"), np.float64, (3,), f"{label}.q02")
    upper = _readonly_array(row.get("q98"), np.float64, (3,), f"{label}.q98")
    if np.any(upper <= lower):
        raise F6ContractError(f"{label} must have strictly positive extents")
    center = _readonly_array(row.get("center"), np.float64, (3,), f"{label}.center")
    extent = _readonly_array(row.get("extent"), np.float64, (3,), f"{label}.extent")
    if float(np.max(np.abs(center - (lower + upper) * 0.5))) > 1.0e-9:
        raise F6ContractError(f"{label}.center differs from q02/q98")
    if float(np.max(np.abs(extent - (upper - lower)))) > 1.0e-9:
        raise F6ContractError(f"{label}.extent differs from q02/q98")
    return lower, upper


def _optional_aabb(row: object, label: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    try:
        return _aabb_from_row(row, label)
    except F6ContractError:
        return None


def _aabb_metrics(
    left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]
) -> tuple[float, float, float]:
    left_lower, left_upper = left
    right_lower, right_upper = right
    intersection_extent = np.maximum(
        np.minimum(left_upper, right_upper) - np.maximum(left_lower, right_lower), 0.0
    )
    intersection = float(np.prod(intersection_extent))
    left_volume = float(np.prod(left_upper - left_lower))
    right_volume = float(np.prod(right_upper - right_lower))
    union = left_volume + right_volume - intersection
    iou = 0.0 if union <= 0.0 else intersection / union
    containment = intersection / min(left_volume, right_volume)
    scale = max(
        float(np.linalg.norm(left_upper - left_lower)),
        float(np.linalg.norm(right_upper - right_lower)),
        0.02,
    )
    nd = float(np.linalg.norm((left_lower + left_upper - right_lower - right_upper) * 0.5)) / scale
    return float(iou), float(containment), float(nd)


def _aabb_geometry(name: str, bounds: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
    lower, upper = bounds
    center = (lower + upper) * 0.5
    extent = upper - lower
    corners = center[None, :] + _CORNER_SIGNS * (extent[None, :] * 0.5)
    return {
        "kind": "world_aabb",
        "hypothesis": name,
        "q02": lower.tolist(),
        "q98": upper.tolist(),
        "center": center.tolist(),
        "extent": extent.tolist(),
        "world_rotation": np.eye(3, dtype=np.float64).tolist(),
        "world_corners": corners.tolist(),
        "envelope_q02": lower.tolist(),
        "envelope_q98": upper.tolist(),
    }


def _hb_geometry(row: object) -> dict[str, Any]:
    if not isinstance(row, Mapping) or row.get("valid") is not True:
        raise F6ContractError("HB validity is false or absent")
    center = _readonly_array(row.get("world_center"), np.float64, (3,), "HB.world_center")
    extent = _readonly_array(row.get("local_extent"), np.float64, (3,), "HB.local_extent")
    rotation = _readonly_array(row.get("world_rotation"), np.float64, (3, 3), "HB.world_rotation")
    corners = _readonly_array(row.get("world_corners"), np.float64, (8, 3), "HB.world_corners")
    if np.any(extent <= 0.0):
        raise F6ContractError("HB.local_extent must be positive")
    if float(np.linalg.det(rotation)) <= 0.0 or not np.allclose(
        rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1.0e-3
    ):
        raise F6ContractError("HB.world_rotation is not right-handed orthonormal")
    expected = center[None, :] + (_CORNER_SIGNS * (extent[None, :] * 0.5)) @ rotation.T
    if not np.allclose(corners, expected, rtol=0.0, atol=2.0e-6):
        raise F6ContractError("HB.world_corners differ from center/extent/rotation")
    if _finite_scalar(row.get("camera_depth"), "HB.camera_depth") <= NEAR_PLANE_M:
        raise F6ContractError("HB.camera_depth is not positive")
    lower = corners.min(axis=0)
    upper = corners.max(axis=0)
    if np.any(upper <= lower):
        raise F6ContractError("HB envelope is degenerate")
    return {
        "kind": "world_obb",
        "hypothesis": "HB",
        "world_center": center.tolist(),
        "local_extent": extent.tolist(),
        "world_rotation": rotation.tolist(),
        "world_corners": corners.tolist(),
        "envelope_q02": lower.tolist(),
        "envelope_q98": upper.tolist(),
    }


def _geometry_arrays(geometry: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    if geometry.get("kind") == "world_aabb":
        center = np.asarray(geometry["center"], dtype=np.float64)
        extent = np.asarray(geometry["extent"], dtype=np.float64)
        rotation = np.eye(3, dtype=np.float64)
        corners = np.asarray(geometry["world_corners"], dtype=np.float64)
    elif geometry.get("kind") == "world_obb":
        center = np.asarray(geometry["world_center"], dtype=np.float64)
        extent = np.asarray(geometry["local_extent"], dtype=np.float64)
        rotation = np.asarray(geometry["world_rotation"], dtype=np.float64)
        corners = np.asarray(geometry["world_corners"], dtype=np.float64)
    else:
        raise F6ContractError("geometry kind is invalid")
    lower = np.asarray(geometry["envelope_q02"], dtype=np.float64)
    upper = np.asarray(geometry["envelope_q98"], dtype=np.float64)
    return center, extent, rotation, corners, (lower, upper)


def _base_hypothesis(source: "F6SourceEvidence") -> tuple[str, dict[str, Any], dict[str, Any]]:
    h0_row = source.hypotheses["H0"]
    h0 = _aabb_from_row(h0_row, "H0")
    stored_count = h0_row.get("stored_point_count") if isinstance(h0_row, Mapping) else None
    if stored_count is not None and _strict_int(stored_count, "H0.stored_point_count") != source.original_point_count:
        raise F6ContractError("H0 stored point count differs from sealed evidence")
    h0_extent = h0[1] - h0[0]
    h0_volume = float(np.prod(h0_extent))
    h0_diagonal = float(np.linalg.norm(h0_extent))
    attempts: dict[str, Any] = {}
    for name in ("HLG", "HL"):
        row = source.hypotheses.get(name)
        bounds = _optional_aabb(row, name)
        reason = "missing_or_invalid"
        metrics: dict[str, Any] = {}
        if bounds is not None and isinstance(row, Mapping):
            diagnostics = row.get("diagnostics")
            if not isinstance(diagnostics, Mapping):
                reason = "diagnostics"
            elif diagnostics.get("applied") is not True or diagnostics.get("fallback") is not False:
                reason = "applied_or_fallback"
            else:
                try:
                    retained = _strict_int(diagnostics.get("retained_point_count"), f"{name}.retained_point_count")
                except F6ContractError:
                    retained = -1
                required = max(BASE_MIN_RETAINED_POINTS, int(math.ceil(BASE_MIN_RETAINED_FRACTION * source.original_point_count)))
                extent = bounds[1] - bounds[0]
                volume_ratio = float(np.prod(extent) / h0_volume)
                extent_ratio = extent / h0_extent
                center_shift = float(np.linalg.norm((bounds[0] + bounds[1] - h0[0] - h0[1]) * 0.5))
                center_limit = BASE_CENTER_DIAGONAL_FRACTION * h0_diagonal + BASE_CENTER_ABSOLUTE_MARGIN_M
                metrics = {
                    "retained_point_count": retained,
                    "required_retained_point_count": required,
                    "volume_ratio_to_h0": volume_ratio,
                    "extent_ratios_to_h0": extent_ratio.tolist(),
                    "center_shift_m": center_shift,
                    "center_shift_limit_m": center_limit,
                }
                if retained < required:
                    reason = "retained_count"
                elif not BASE_VOLUME_RATIO[0] <= volume_ratio <= BASE_VOLUME_RATIO[1]:
                    reason = "volume_ratio"
                elif not np.all((extent_ratio >= BASE_EXTENT_RATIO[0]) & (extent_ratio <= BASE_EXTENT_RATIO[1])):
                    reason = "extent_ratio"
                elif center_shift > center_limit:
                    reason = "center_shift"
                else:
                    attempts[name] = {"eligible": True, "reason": "eligible", **metrics}
                    return name, _aabb_geometry(name, bounds), {
                        "n0": source.original_point_count,
                        "h0_volume": h0_volume,
                        "attempts": attempts,
                    }
        attempts[name] = {"eligible": False, "reason": reason, **metrics}
    return "H0", _aabb_geometry("H0", h0), {
        "n0": source.original_point_count,
        "h0_volume": h0_volume,
        "attempts": attempts,
    }


def _evidence_digest(source: "F6SourceEvidence") -> str:
    digest = hashlib.sha256()
    digest.update(source.source_id.encode("ascii"))
    digest.update(np.asarray([source.frame_id, source.frame_ordinal, source.rank, source.original_point_count, source.mask_pixel_count], dtype="<i8").tobytes())
    digest.update(source.source_lineage_sha256.encode("ascii"))
    digest.update(canonical_json_sha256(source.hypotheses).encode("ascii"))
    for array, dtype in (
        (source.points_world, "<f8"),
        (source.sampled_mask_pixels_yx, "<i2"),
        (source.mask_packbits, "u1"),
        (source.tight_box_xyxy, "<f8"),
        (source.camera_to_world, "<f8"),
        (source.intrinsic, "<f8"),
    ):
        digest.update(np.asarray(array, dtype=dtype).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class F6SourceEvidence:
    """One normalized sealed source; large point/mask lists are sampled once."""

    source_id: str
    frame_id: int
    frame_ordinal: int
    rank: int
    hypotheses: Mapping[str, Any]
    points_world: np.ndarray
    mask_packbits: np.ndarray
    tight_box_xyxy: np.ndarray
    camera_to_world: np.ndarray
    intrinsic: np.ndarray
    source_lineage_sha256: str
    original_point_count: int = field(init=False)
    sampled_mask_pixels_yx: np.ndarray = field(init=False, repr=False, compare=False)
    mask_pixel_count: int = field(init=False)
    input_evidence_sha256: str = field(init=False)
    input_hypothesis_sha256: Mapping[str, str] = field(init=False, repr=False, compare=False)
    audit_hash_ns: int = field(init=False, repr=False, compare=False)
    audit_serialization_ns: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        match = _SOURCE_RE.fullmatch(self.source_id) if isinstance(self.source_id, str) else None
        if match is None:
            raise F6ContractError("source_id is not canonical")
        frame_id = _strict_int(self.frame_id, "frame_id")
        frame_ordinal = _strict_int(self.frame_ordinal, "frame_ordinal")
        rank = _strict_int(self.rank, "rank")
        if int(match.group("frame")) != frame_id:
            raise F6ContractError("source_id frame differs from frame_id")
        if rank >= MAX_SOURCES_PER_FRAME:
            raise F6ContractError("rank exceeds the sealed per-frame cap")
        if not isinstance(self.hypotheses, Mapping) or "H0" not in self.hypotheses:
            raise F6ContractError("hypotheses must contain H0")
        serialization_started = time.perf_counter_ns()
        hypotheses = _deep_freeze_json(self.hypotheses)
        audit_serialization_ns = time.perf_counter_ns() - serialization_started
        assert isinstance(hypotheses, Mapping)
        _aabb_from_row(hypotheses["H0"], "H0")
        points, original_count = _sample_points(self.points_world)
        packed, pixels, mask_count = _mask_and_pixels(self.mask_packbits)
        tight = _readonly_array(self.tight_box_xyxy, np.float64, (4,), "tight_box_xyxy")
        if tight[2] <= tight[0] or tight[3] <= tight[1] or tight[0] < 0.0 or tight[1] < 0.0 or tight[2] > IMAGE_WIDTH or tight[3] > IMAGE_HEIGHT:
            raise F6ContractError("tight_box_xyxy lies outside the sealed frame")
        pose = _validate_pose(self.camera_to_world)
        intrinsic = _validate_intrinsic(self.intrinsic)
        if not isinstance(self.source_lineage_sha256, str) or _SHA256_RE.fullmatch(self.source_lineage_sha256) is None:
            raise F6ContractError("source_lineage_sha256 is invalid")
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "frame_ordinal", frame_ordinal)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "hypotheses", hypotheses)
        object.__setattr__(self, "points_world", points)
        object.__setattr__(self, "original_point_count", original_count)
        object.__setattr__(self, "mask_packbits", packed)
        object.__setattr__(self, "sampled_mask_pixels_yx", pixels)
        object.__setattr__(self, "mask_pixel_count", mask_count)
        object.__setattr__(self, "tight_box_xyxy", tight)
        object.__setattr__(self, "camera_to_world", pose)
        object.__setattr__(self, "intrinsic", intrinsic)
        hash_started = time.perf_counter_ns()
        input_hypothesis_sha256 = MappingProxyType(
            {
                name: canonical_json_sha256(hypotheses.get(name))
                for name in ("H0", "HL", "HLG", "HB")
            }
        )
        input_evidence_sha256 = _evidence_digest(self)
        audit_hash_ns = time.perf_counter_ns() - hash_started
        object.__setattr__(self, "input_hypothesis_sha256", input_hypothesis_sha256)
        object.__setattr__(self, "input_evidence_sha256", input_evidence_sha256)
        object.__setattr__(self, "audit_hash_ns", audit_hash_ns)
        object.__setattr__(self, "audit_serialization_ns", audit_serialization_ns)


@dataclass(frozen=True)
class _View:
    source_id: str
    frame_id: int
    frame_ordinal: int
    rank: int
    h0: tuple[np.ndarray, np.ndarray]
    points_world: np.ndarray
    original_point_count: int
    mask_packbits: np.ndarray
    sampled_mask_pixels_yx: np.ndarray
    mask_pixel_count: int
    camera_to_world: np.ndarray
    intrinsic: np.ndarray
    source_lineage_sha256: str
    state_sha256: str

    @property
    def raw_array_payload_bytes(self) -> int:
        return int(
            self.h0[0].nbytes
            + self.h0[1].nbytes
            + self.points_world.nbytes
            + self.mask_packbits.nbytes
            + self.sampled_mask_pixels_yx.nbytes
            + self.camera_to_world.nbytes
            + self.intrinsic.nbytes
        )


def _view_digest(
    source_id: str,
    frame_id: int,
    frame_ordinal: int,
    rank: int,
    h0: tuple[np.ndarray, np.ndarray],
    points: np.ndarray,
    original_point_count: int,
    mask: np.ndarray,
    pixels: np.ndarray,
    mask_pixel_count: int,
    pose: np.ndarray,
    intrinsic: np.ndarray,
    lineage: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(source_id.encode("ascii"))
    digest.update(np.asarray([frame_id, frame_ordinal, rank, original_point_count, mask_pixel_count], dtype="<i8").tobytes())
    digest.update(lineage.encode("ascii"))
    for array, dtype in ((h0[0], "<f8"), (h0[1], "<f8"), (points, "<f8"), (mask, "u1"), (pixels, "<i2"), (pose, "<f8"), (intrinsic, "<f8")):
        digest.update(np.asarray(array, dtype=dtype).tobytes())
    return digest.hexdigest()


def _view_from_source(source: F6SourceEvidence) -> _View:
    h0 = _aabb_from_row(source.hypotheses["H0"], "H0")
    digest = _view_digest(
        source.source_id, source.frame_id, source.frame_ordinal, source.rank,
        h0, source.points_world, source.original_point_count, source.mask_packbits,
        source.sampled_mask_pixels_yx, source.mask_pixel_count,
        source.camera_to_world, source.intrinsic, source.source_lineage_sha256,
    )
    return _View(
        source.source_id, source.frame_id, source.frame_ordinal, source.rank,
        h0, source.points_world, source.original_point_count, source.mask_packbits,
        source.sampled_mask_pixels_yx, source.mask_pixel_count,
        source.camera_to_world, source.intrinsic, source.source_lineage_sha256, digest,
    )


@dataclass(frozen=True)
class _Prepared:
    source: F6SourceEvidence
    view: _View
    base_name: str
    base_geometry: Mapping[str, Any]
    base_diagnostics: Mapping[str, Any]
    geometries: Mapping[str, Mapping[str, Any]]
    geometry_errors: Mapping[str, Optional[str]]
    input_hypothesis_sha256: Mapping[str, str]
    geometry_arrays: Mapping[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]],
    ]
    geometry_sha256: Mapping[str, str]
    audit_hash_ns: int


def _prepare(source: F6SourceEvidence) -> _Prepared:
    base_name, base_geometry, base_diagnostics = _base_hypothesis(source)
    geometries: dict[str, Mapping[str, Any]] = {}
    errors: dict[str, Optional[str]] = {}
    for name in ("H0", "HL", "HLG"):
        try:
            bounds = _aabb_from_row(source.hypotheses.get(name), name)
            geometries[name] = _aabb_geometry(name, bounds)
            errors[name] = None
        except F6ContractError as error:
            if name == "H0":
                raise
            errors[name] = str(error)
    try:
        geometries["HB"] = _hb_geometry(source.hypotheses.get("HB"))
        errors["HB"] = None
    except F6ContractError as error:
        errors["HB"] = str(error)
    geometry_arrays = {
        name: _geometry_arrays(geometry) for name, geometry in geometries.items()
    }
    hash_started = time.perf_counter_ns()
    view = _view_from_source(source)
    geometry_sha256 = {
        name: canonical_json_sha256(geometry) for name, geometry in geometries.items()
    }
    audit_hash_ns = time.perf_counter_ns() - hash_started
    return _Prepared(
        source=source,
        view=view,
        base_name=base_name,
        base_geometry=MappingProxyType(dict(base_geometry)),
        base_diagnostics=MappingProxyType(dict(base_diagnostics)),
        geometries=MappingProxyType(geometries),
        geometry_errors=MappingProxyType(errors),
        input_hypothesis_sha256=source.input_hypothesis_sha256,
        geometry_arrays=MappingProxyType(geometry_arrays),
        geometry_sha256=MappingProxyType(geometry_sha256),
        audit_hash_ns=audit_hash_ns,
    )


@dataclass(frozen=True)
class _PastFrame:
    frame_id: int
    frame_ordinal: int
    rows: tuple[_View, ...]

    @property
    def raw_array_payload_bytes(self) -> int:
        return sum(row.raw_array_payload_bytes for row in self.rows)


def _mutual_best(current: Sequence[_Prepared], past: _PastFrame) -> dict[int, tuple[_View, tuple[float, float, float]]]:
    edges: list[tuple[int, int, tuple[float, float, float]]] = []
    if not current or not past.rows:
        return {}
    current_centers = tuple((row.view.h0[0] + row.view.h0[1]) * 0.5 for row in current)
    current_diagonals = tuple(
        float(np.linalg.norm(row.view.h0[1] - row.view.h0[0])) for row in current
    )
    past_centers = tuple((row.h0[0] + row.h0[1]) * 0.5 for row in past.rows)
    past_diagonals = tuple(
        float(np.linalg.norm(row.h0[1] - row.h0[0])) for row in past.rows
    )
    center_delta = (
        np.stack(current_centers, axis=0)[:, None, :]
        - np.stack(past_centers, axis=0)[None, :, :]
    )
    squared_distance = np.sum(center_delta * center_delta, axis=2)
    scale = np.maximum(
        np.maximum(
            np.asarray(current_diagonals, dtype=np.float64)[:, None],
            np.asarray(past_diagonals, dtype=np.float64)[None, :],
        ),
        0.02,
    )
    squared_limit = (ASSOCIATION_ND_MAX * scale) ** 2
    # This is only a conservative rejection.  Boundary pairs always take the
    # original exact scalar metric path, preserving every emitted affinity.
    possible_pairs = np.argwhere(
        squared_distance <= squared_limit * (1.0 + 1.0e-12)
    )
    for current_index_raw, past_index_raw in possible_pairs:
        current_index = int(current_index_raw)
        past_index = int(past_index_raw)
        row = current[current_index]
        prior = past.rows[past_index]
        iou, containment, nd = _aabb_metrics(row.view.h0, prior.h0)
        if nd <= ASSOCIATION_ND_MAX and (iou >= ASSOCIATION_IOU_MIN or containment >= ASSOCIATION_CONTAINMENT_MIN):
            edges.append((current_index, past_index, (iou, containment, -nd)))
    current_best: dict[int, tuple[int, tuple[float, float, float]]] = {}
    past_best: dict[int, tuple[int, tuple[float, float, float]]] = {}
    for current_index, past_index, affinity in edges:
        prior = past.rows[past_index]
        old = current_best.get(current_index)
        if old is None or affinity > old[1] or (
            affinity == old[1] and (prior.rank, prior.source_id) < (past.rows[old[0]].rank, past.rows[old[0]].source_id)
        ):
            current_best[current_index] = (past_index, affinity)
        current_row = current[current_index].source
        old_past = past_best.get(past_index)
        if old_past is None or affinity > old_past[1] or (
            affinity == old_past[1]
            and (current_row.rank, current_row.source_id)
            < (current[old_past[0]].source.rank, current[old_past[0]].source.source_id)
        ):
            past_best[past_index] = (current_index, affinity)
    result: dict[int, tuple[_View, tuple[float, float, float]]] = {}
    for current_index, (past_index, affinity) in current_best.items():
        if past_best.get(past_index, (-1, ()))[0] == current_index:
            result[current_index] = (past.rows[past_index], affinity)
    return result


def _convex_hull(points: np.ndarray) -> Optional[np.ndarray]:
    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique) < 3:
        return None

    def cross(origin: tuple[float, float], left: tuple[float, float], right: tuple[float, float]) -> float:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (right[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    if len(hull) < 3:
        return None
    area2 = float(np.sum(hull[:, 0] * np.roll(hull[:, 1], -1) - hull[:, 1] * np.roll(hull[:, 0], -1)))
    return None if area2 <= 0.0 else hull


def _inside_convex(
    points: np.ndarray,
    hull: np.ndarray,
    edges: Optional[np.ndarray] = None,
) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=np.bool_)
    if edges is None:
        edges = np.roll(hull, -1, axis=0) - hull
    relative = points[:, None, :] - hull[None, :, :]
    cross = edges[None, :, 0] * relative[:, :, 1] - edges[None, :, 1] * relative[:, :, 0]
    return np.all(cross >= -1.0e-12, axis=1)


def _mask_bits_at(packed: np.ndarray, xy: np.ndarray) -> np.ndarray:
    x = np.floor(xy[:, 0]).astype(np.int64)
    y = np.floor(xy[:, 1]).astype(np.int64)
    valid = (x >= 0) & (x < IMAGE_WIDTH) & (y >= 0) & (y < IMAGE_HEIGHT)
    result = np.zeros(len(x), dtype=np.bool_)
    indices = y[valid] * IMAGE_WIDTH + x[valid]
    result[valid] = ((packed[indices // 8] >> (indices % 8)) & np.uint8(1)).astype(np.bool_)
    return result


def _project_metric(
    corners_world: np.ndarray,
    view: _View,
    projection_cache: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    camera_rotation, camera_translation, sampled_xy = projection_cache
    camera = corners_world @ camera_rotation.T + camera_translation[None, :]
    if not np.isfinite(camera).all() or not np.all(camera[:, 2] > NEAR_PLANE_M):
        return {"valid": False, "reason": "corner_at_or_behind_near_plane", "minimum_corner_depth_m": None, "R": None, "P": None, "J": None}
    projected = camera @ view.intrinsic.T
    pixels = projected[:, :2] / projected[:, 2:3]
    if not np.isfinite(pixels).all():
        return {"valid": False, "reason": "nonfinite_projection", "minimum_corner_depth_m": float(camera[:, 2].min()), "R": None, "P": None, "J": None}
    hull = _convex_hull(pixels)
    if hull is None:
        return {"valid": False, "reason": "degenerate_convex_hull", "minimum_corner_depth_m": float(camera[:, 2].min()), "R": None, "P": None, "J": None}
    hull_edges = np.roll(hull, -1, axis=0) - hull
    x0 = max(0.0, float(hull[:, 0].min()))
    x1 = min(float(IMAGE_WIDTH), float(hull[:, 0].max()))
    y0 = max(0.0, float(hull[:, 1].min()))
    y1 = min(float(IMAGE_HEIGHT), float(hull[:, 1].max()))
    grid_inside_count = 0
    grid_mask_hit_count = 0
    p_value = 0.0
    if x1 > x0 and y1 > y0:
        grid = np.empty_like(_GRID_CELL_CENTRES_16X16)
        # Preserve the original multiply-then-divide operation order exactly;
        # only the repeated meshgrid/column_stack allocations are removed.
        grid[:, 0] = (
            x0
            + _GRID_CELL_CENTRES_16X16[:, 0] * (x1 - x0) / 16.0
        )
        grid[:, 1] = (
            y0
            + _GRID_CELL_CENTRES_16X16[:, 1] * (y1 - y0) / 16.0
        )
        combined_inside = _inside_convex(
            np.concatenate((sampled_xy, grid), axis=0), hull, hull_edges
        )
        inside_mask = combined_inside[: len(sampled_xy)]
        inside = combined_inside[len(sampled_xy) :]
        probes = grid[inside]
        grid_inside_count = int(len(probes))
        if grid_inside_count:
            grid_mask_hit_count = int(np.count_nonzero(_mask_bits_at(view.mask_packbits, probes)))
            p_value = grid_mask_hit_count / grid_inside_count
    else:
        inside_mask = _inside_convex(sampled_xy, hull, hull_edges)
    inside_positive_count = int(np.count_nonzero(inside_mask))
    r_value = inside_positive_count / len(sampled_xy)
    if p_value * r_value == 0.0:
        j_value = 0.0
    else:
        j_value = 1.0 / (1.0 / p_value + 1.0 / r_value - 1.0)
    return {
        "valid": True,
        "reason": "valid",
        "minimum_corner_depth_m": float(camera[:, 2].min()),
        "sampled_positive_pixel_count": int(len(sampled_xy)),
        "positive_pixels_inside_hull": inside_positive_count,
        "R": float(r_value),
        "grid_inside_hull_count": grid_inside_count,
        "grid_mask_hit_count": grid_mask_hit_count,
        "P": float(p_value),
        "J": float(j_value),
    }


def _score_geometry(
    geometry_arrays: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        tuple[np.ndarray, np.ndarray],
    ],
    views: Sequence[_View],
    projection_caches: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    point_cache: tuple[np.ndarray, np.ndarray],
    *,
    world_aabb: bool,
) -> dict[str, Any]:
    center, extent, rotation, corners, _ = geometry_arrays
    per_view: list[dict[str, Any]] = []
    all_local: list[np.ndarray] = []
    concatenated_world, point_offsets = point_cache
    concatenated_aabb_local = (
        concatenated_world - center[None, :] if world_aabb else None
    )
    for view_index, (view, projection_cache) in enumerate(
        zip(views, projection_caches, strict=True)
    ):
        if concatenated_aabb_local is None:
            local = (view.points_world - center[None, :]) @ rotation
        else:
            local = concatenated_aabb_local[
                point_offsets[view_index] : point_offsets[view_index + 1]
            ]
        exact = np.all(np.abs(local) <= extent[None, :] * 0.5, axis=1)
        expanded = np.all(np.abs(local) <= extent[None, :] * 0.5 + ROBUST_MARGIN_M, axis=1)
        c0 = 0.0 if len(local) == 0 else float(np.count_nonzero(exact) / len(local))
        c5 = 0.0 if len(local) == 0 else float(np.count_nonzero(expanded) / len(local))
        projection = _project_metric(corners, view, projection_cache)
        per_view.append(
            {
                "source_id": view.source_id,
                "frame_id": view.frame_id,
                "frame_ordinal": view.frame_ordinal,
                "rank": view.rank,
                "source_lineage_sha256": view.source_lineage_sha256,
                "state_sha256": view.state_sha256,
                "original_point_count": view.original_point_count,
                "sampled_point_count": int(len(local)),
                "C0": c0,
                "C5": c5,
                "support_gate_passed": bool(c0 >= EXACT_SUPPORT_MIN and c5 >= EXPANDED_SUPPORT_MIN),
                "projection": projection,
            }
        )
        all_local.append(local)
    concatenated = (
        concatenated_aabb_local
        if concatenated_aabb_local is not None
        else np.concatenate(all_local, axis=0)
    )
    if len(concatenated):
        quantiles = np.quantile(concatenated, (0.02, 0.98), axis=0, method="linear")
        candidate_faces = np.stack((-extent * 0.5, extent * 0.5), axis=0)
        d_value: Optional[float] = float(np.mean(np.abs(quantiles - candidate_faces)))
        q02_local: Optional[list[float]] = quantiles[0].tolist()
        q98_local: Optional[list[float]] = quantiles[1].tolist()
    else:
        d_value = None
        q02_local = None
        q98_local = None
    c0_median = float(np.median([row["C0"] for row in per_view]))
    c5_median = float(np.median([row["C5"] for row in per_view]))
    projections_valid = all(bool(row["projection"]["valid"]) for row in per_view)
    j_median = None
    if projections_valid:
        j_median = float(np.median([row["projection"]["J"] for row in per_view]))
    return {
        "valid": bool(
            d_value is not None
            and all(len(view.points_world) >= MIN_POINTS_PER_VIEW for view in views)
            and projections_valid
        ),
        "all_views_have_minimum_points": bool(all(len(view.points_world) >= MIN_POINTS_PER_VIEW for view in views)),
        "all_projections_valid": projections_valid,
        "C0": c0_median,
        "C5": c5_median,
        "D_m": d_value,
        "J": j_median,
        "q02_local": q02_local,
        "q98_local": q98_local,
        "supporting_view_count": sum(bool(row["support_gate_passed"]) for row in per_view),
        "per_view": per_view,
    }


def _unavailable_candidate(name: str, reason: str, is_base: bool = False) -> dict[str, Any]:
    return {
        "hypothesis": name,
        "available": False,
        "availability_reason": reason,
        "is_base": is_base,
        "geometry": None,
        "geometry_sha256": None,
        "metrics": None,
        "gate": None,
        "comparison": None,
        "selector_passed": False,
    }


def _select_one(
    prepared: _Prepared,
    matches: Sequence[tuple[_View, tuple[float, float, float]]],
) -> tuple[dict[str, Any], int]:
    source = prepared.source
    selected_matches = tuple(sorted(matches, key=lambda item: item[0].frame_ordinal, reverse=True)[:MIN_PAST_MATCHES])
    selected_matches = tuple(sorted(selected_matches, key=lambda item: (item[0].frame_ordinal, item[0].rank, item[0].source_id)))
    matched_rows = [
        {
            "source_id": view.source_id,
            "frame_id": view.frame_id,
            "frame_ordinal": view.frame_ordinal,
            "rank": view.rank,
            "source_lineage_sha256": view.source_lineage_sha256,
            "state_sha256": view.state_sha256,
            "affinity": {"iou3d": affinity[0], "symmetric_containment": affinity[1], "normalized_center_distance": -affinity[2]},
        }
        for view, affinity in selected_matches
    ]
    evaluations: dict[str, Any] = {}
    base_metrics: Optional[dict[str, Any]] = None
    chosen_name = prepared.base_name
    chosen_geometry = dict(prepared.base_geometry)
    selection_reason = "fewer_than_two_past_matches"

    if len(selected_matches) >= MIN_PAST_MATCHES:
        views = (prepared.view,) + tuple(item[0] for item in selected_matches)
        # The same three poses are used by every copied geometry.  Inverses
        # are transient query-local values and are never retained in state.
        try:
            world_to_camera = tuple(
                np.linalg.inv(view.camera_to_world) for view in views
            )
        except np.linalg.LinAlgError as error:  # guarded at evidence ingest
            raise F6ContractError("committed camera pose became noninvertible") from error
        projection_caches = tuple(
            (
                inverse[:3, :3],
                inverse[:3, 3],
                np.column_stack(
                    (
                        view.sampled_mask_pixels_yx[:, 1],
                        view.sampled_mask_pixels_yx[:, 0],
                    )
                ).astype(np.float64)
                + 0.5,
            )
            for view, inverse in zip(views, world_to_camera, strict=True)
        )
        point_offsets = np.zeros(len(views) + 1, dtype=np.int64)
        point_offsets[1:] = np.cumsum(
            np.asarray([len(view.points_world) for view in views], dtype=np.int64)
        )
        point_cache = (
            np.concatenate(tuple(view.points_world for view in views), axis=0),
            point_offsets,
        )
        base_arrays = prepared.geometry_arrays[prepared.base_name]
        base_metrics = _score_geometry(
            base_arrays,
            views,
            projection_caches,
            point_cache,
            world_aabb=True,
        )
        base_envelope = base_arrays[4]
        passers: list[tuple[str, dict[str, Any]]] = []
        for name in ("H0", "HL", "HLG", "HB"):
            geometry = prepared.geometries.get(name)
            if geometry is None:
                evaluations[name] = _unavailable_candidate(name, prepared.geometry_errors.get(name) or "missing_or_invalid")
                continue
            metrics = (
                base_metrics
                if name == prepared.base_name
                else _score_geometry(
                    prepared.geometry_arrays[name],
                    views,
                    projection_caches,
                    point_cache,
                    world_aabb=name != "HB",
                )
            )
            geometry_hash = prepared.geometry_sha256[name]
            if name == prepared.base_name:
                evaluations[name] = {
                    "hypothesis": name,
                    "available": True,
                    "availability_reason": "valid_base",
                    "is_base": True,
                    "geometry": dict(geometry),
                    "geometry_sha256": geometry_hash,
                    "metrics": metrics,
                    "gate": {"passed": True, "reason": "base_not_gated"},
                    "comparison": {"passed": False, "reason": "base_not_compared"},
                    "selector_passed": False,
                }
                continue
            envelope = prepared.geometry_arrays[name][4]
            iou, containment, nd = _aabb_metrics(envelope, base_envelope)
            volume_ratio = float(np.prod(envelope[1] - envelope[0]) / np.prod(base_envelope[1] - base_envelope[0]))
            current_view = metrics["per_view"][0]
            support_count = int(metrics["supporting_view_count"])
            gate_checks = {
                "metrics_valid": bool(metrics["valid"]),
                "all_projections_valid": bool(metrics["all_projections_valid"]),
                "all_views_have_minimum_points": bool(metrics["all_views_have_minimum_points"]),
                "nd_passed": bool(nd <= CANDIDATE_ND_MAX),
                "volume_ratio_passed": bool(CANDIDATE_VOLUME_RATIO[0] <= volume_ratio <= CANDIDATE_VOLUME_RATIO[1]),
                "overlap_passed": bool(iou >= CANDIDATE_IOU_MIN or containment >= CANDIDATE_CONTAINMENT_MIN),
                "current_exact_support_passed": bool(current_view["C0"] >= EXACT_SUPPORT_MIN),
                "current_expanded_support_passed": bool(current_view["C5"] >= EXPANDED_SUPPORT_MIN),
                "two_of_three_support_passed": bool(support_count >= MIN_SUPPORTING_VIEWS),
            }
            gate_passed = all(gate_checks.values())
            gate = {
                "passed": gate_passed,
                "candidate_base_iou3d": iou,
                "candidate_base_symmetric_containment": containment,
                "candidate_base_normalized_center_distance": nd,
                "candidate_base_envelope_volume_ratio": volume_ratio,
                "supporting_view_count": support_count,
                "checks": gate_checks,
            }
            comparison: dict[str, Any]
            selector_passed = False
            if not gate_passed or not base_metrics["valid"] or base_metrics["J"] is None or metrics["J"] is None:
                comparison = {
                    "passed": False,
                    "reason": "candidate_gate_failed" if not gate_passed else "base_or_candidate_metrics_invalid",
                    "depth_win": False,
                    "projection_win": False,
                    "containment_win": False,
                    "win_count": 0,
                    "depth_non_regression": False,
                    "projection_non_regression": False,
                    "containment_non_regression": False,
                }
            else:
                depth_win = bool(metrics["D_m"] <= base_metrics["D_m"] - DEPTH_WIN_MARGIN_M)
                projection_win = bool(metrics["J"] >= base_metrics["J"] + PROJECTION_WIN_MARGIN)
                containment_win = bool(metrics["C0"] >= base_metrics["C0"] + CONTAINMENT_WIN_MARGIN)
                win_count = int(depth_win) + int(projection_win) + int(containment_win)
                depth_nr = bool(metrics["D_m"] <= base_metrics["D_m"] + DEPTH_MAX_REGRESSION_M)
                projection_nr = bool(metrics["J"] >= base_metrics["J"] - PROJECTION_MAX_REGRESSION)
                containment_nr = bool(metrics["C0"] >= base_metrics["C0"] - CONTAINMENT_MAX_REGRESSION)
                selector_passed = bool(win_count >= MIN_METRIC_WINS and depth_nr and projection_nr and containment_nr)
                comparison = {
                    "passed": selector_passed,
                    "reason": "passed" if selector_passed else "wins_or_non_regression_failed",
                    "depth_win": depth_win,
                    "projection_win": projection_win,
                    "containment_win": containment_win,
                    "win_count": win_count,
                    "depth_non_regression": depth_nr,
                    "projection_non_regression": projection_nr,
                    "containment_non_regression": containment_nr,
                    "candidate_minus_base": {
                        "D_m": float(metrics["D_m"] - base_metrics["D_m"]),
                        "J": float(metrics["J"] - base_metrics["J"]),
                        "C0": float(metrics["C0"] - base_metrics["C0"]),
                    },
                }
            evaluation = {
                "hypothesis": name,
                "available": True,
                "availability_reason": "valid",
                "is_base": False,
                "geometry": dict(geometry),
                "geometry_sha256": geometry_hash,
                "metrics": metrics,
                "gate": gate,
                "comparison": comparison,
                "selector_passed": selector_passed,
            }
            evaluations[name] = evaluation
            if selector_passed:
                passers.append((name, evaluation))
        if passers:
            passers.sort(
                key=lambda item: (
                    -int(item[1]["comparison"]["win_count"]),
                    float(item[1]["metrics"]["D_m"]),
                    -float(item[1]["metrics"]["J"]),
                    -float(item[1]["metrics"]["C0"]),
                    _TIE_PRIORITY[item[0]],
                )
            )
            chosen_name, chosen_eval = passers[0]
            chosen_geometry = dict(chosen_eval["geometry"])
            selection_reason = "non_base_candidate_won"
        else:
            selection_reason = "no_non_base_candidate_passed"
    else:
        for name in ("H0", "HL", "HLG", "HB"):
            geometry = prepared.geometries.get(name)
            if geometry is None:
                evaluations[name] = _unavailable_candidate(name, prepared.geometry_errors.get(name) or "missing_or_invalid", name == prepared.base_name)
            else:
                evaluations[name] = {
                    **_unavailable_candidate(name, "fewer_than_two_past_matches", name == prepared.base_name),
                    "available": True,
                    "geometry": dict(geometry),
                    "geometry_sha256": prepared.geometry_sha256[name],
                }

    selected_hash = prepared.geometry_sha256[chosen_name]
    row: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "mode": MODE,
        "source_id": source.source_id,
        "source_lineage_sha256": source.source_lineage_sha256,
        "input_evidence_sha256": source.input_evidence_sha256,
        "frame_id": source.frame_id,
        "frame_ordinal": source.frame_ordinal,
        "rank": source.rank,
        "input_hypothesis_sha256": dict(prepared.input_hypothesis_sha256),
        "base_hypothesis": prepared.base_name,
        "base_geometry": dict(prepared.base_geometry),
        "base_geometry_sha256": prepared.geometry_sha256[prepared.base_name],
        "base_diagnostics": dict(prepared.base_diagnostics),
        "base_metrics": base_metrics,
        "matched_past": matched_rows,
        "matched_past_frame_count": len(selected_matches),
        "candidate_evaluations": evaluations,
        "selected_hypothesis": chosen_name,
        "selected_geometry": chosen_geometry,
        "selected_geometry_sha256": selected_hash,
        "switched_from_base": chosen_name != prepared.base_name,
        "selection_reason": selection_reason,
        "formal_score": 1.0,
        "maximum_lookahead_frames": 0,
        "observer_only": True,
        "birth_applied": False,
        "native_output_mutation_applied": False,
    }
    hash_started = time.perf_counter_ns()
    row["result_sha256"] = canonical_result_sha256(row)
    return row, time.perf_counter_ns() - hash_started


@dataclass(frozen=True)
class F6FrameQuery:
    frame_id: int
    frame_ordinal: int
    rows: tuple[Mapping[str, Any], ...]
    buffer_before: tuple[Mapping[str, Any], ...]
    maximum_accessed_frame_ordinal: int
    state_raw_array_payload_bytes: int
    audit_hash_ns: int
    audit_serialization_ns: int
    token: str


@dataclass(frozen=True)
class F6FrameCommit:
    frame_id: int
    frame_ordinal: int
    source_count: int
    buffer_after: tuple[Mapping[str, Any], ...]
    state_raw_array_payload_bytes: int
    audit_hash_ns: int
    audit_serialization_ns: int
    token: str


@dataclass(frozen=True)
class _Pending:
    query: F6FrameQuery
    prepared: tuple[_Prepared, ...]


class F6SelectorState:
    """Bounded causal F6 selector with exact-token query/commit semantics."""

    observer_only = True
    active_authorized = False

    def __init__(self) -> None:
        self._buffer: list[_PastFrame] = []
        self._seen_source_ids: set[str] = set()
        self._pending: Optional[_Pending] = None
        self._last_committed_ordinal = -1
        self._scene_id: Optional[str] = None

    @property
    def buffered_frame_count(self) -> int:
        return len(self._buffer)

    @property
    def seen_source_count(self) -> int:
        return len(self._seen_source_ids)

    @property
    def raw_array_payload_bytes(self) -> int:
        return sum(frame.raw_array_payload_bytes for frame in self._buffer)

    def _buffer_receipt(self, frames: Optional[Sequence[_PastFrame]] = None) -> tuple[Mapping[str, Any], ...]:
        selected = self._buffer if frames is None else frames
        return tuple(
            {
                "frame_id": frame.frame_id,
                "frame_ordinal": frame.frame_ordinal,
                "source_ids": [row.source_id for row in frame.rows],
                "state_sha256": [row.state_sha256 for row in frame.rows],
                "raw_array_payload_bytes": frame.raw_array_payload_bytes,
            }
            for frame in selected
        )

    def query_frame(self, *, frame_id: int, frame_ordinal: int, sources: Sequence[F6SourceEvidence]) -> F6FrameQuery:
        if self._pending is not None:
            raise F6ContractError("the previous F6 query has not been committed")
        frame_id = _strict_int(frame_id, "frame_id")
        frame_ordinal = _strict_int(frame_ordinal, "frame_ordinal")
        if frame_ordinal <= self._last_committed_ordinal:
            raise F6ContractError("frame ordinals must be strictly increasing")
        source_rows = tuple(sources)
        if len(source_rows) > MAX_SOURCES_PER_FRAME:
            raise F6ContractError("source count exceeds the sealed per-frame cap")
        if any(not isinstance(row, F6SourceEvidence) for row in source_rows):
            raise F6ContractError("sources must contain F6SourceEvidence")
        if any(row.frame_id != frame_id or row.frame_ordinal != frame_ordinal for row in source_rows):
            raise F6ContractError("source frame identity differs from query frame")
        if tuple(sorted(source_rows, key=lambda row: (row.rank, row.source_id))) != source_rows:
            raise F6ContractError("sources must use frozen rank/source order")
        if tuple(row.rank for row in source_rows) != tuple(range(len(source_rows))):
            raise F6ContractError("sources must use contiguous frozen ranks")
        current_ids = tuple(row.source_id for row in source_rows)
        if len(set(current_ids)) != len(current_ids):
            raise F6ContractError("duplicate source identity in current frame")
        if any(source_id in self._seen_source_ids for source_id in current_ids):
            raise F6ContractError("source identity was already committed")
        scenes = {_SOURCE_RE.fullmatch(source_id).group("scene") for source_id in current_ids}  # type: ignore[union-attr]
        if len(scenes) > 1:
            raise F6ContractError("one F6 state cannot mix scenes")
        if scenes and self._scene_id is not None and next(iter(scenes)) != self._scene_id:
            raise F6ContractError("one F6 state cannot cross scene boundaries")

        snapshot = tuple(self._buffer[-MAX_BUFFERED_SUCCESSFUL_FRAMES:])
        payload_before = sum(frame.raw_array_payload_bytes for frame in snapshot)
        if payload_before > MAX_RAW_ARRAY_PAYLOAD_BYTES:
            raise F6ContractError("F6 past-state raw array payload exceeds 2.5 MiB")
        buffer_before = self._buffer_receipt(snapshot)
        prepared = tuple(_prepare(row) for row in source_rows)
        matches: dict[int, list[tuple[_View, tuple[float, float, float]]]] = {index: [] for index in range(len(prepared))}
        for past in snapshot:
            for current_index, match in _mutual_best(prepared, past).items():
                matches[current_index].append(match)
        selected = tuple(
            _select_one(row, matches[index]) for index, row in enumerate(prepared)
        )
        rows = tuple(item[0] for item in selected)
        audit_hash_ns = (
            sum(row.audit_hash_ns for row in source_rows)
            + sum(row.audit_hash_ns for row in prepared)
            + sum(item[1] for item in selected)
        )
        audit_serialization_ns = sum(
            row.audit_serialization_ns for row in source_rows
        )
        maximum_accessed = max((frame.frame_ordinal for frame in snapshot), default=-1)
        token_hash_started = time.perf_counter_ns()
        token = canonical_json_sha256(
            {
                "protocol_id": PROTOCOL_ID,
                "frame_id": frame_id,
                "frame_ordinal": frame_ordinal,
                "buffer_before": buffer_before,
                "maximum_accessed_frame_ordinal": maximum_accessed,
                "state_raw_array_payload_bytes": payload_before,
                "result_sha256": [row["result_sha256"] for row in rows],
            }
        )
        audit_hash_ns += time.perf_counter_ns() - token_hash_started
        query = F6FrameQuery(
            frame_id=frame_id,
            frame_ordinal=frame_ordinal,
            rows=rows,
            buffer_before=buffer_before,
            maximum_accessed_frame_ordinal=maximum_accessed,
            state_raw_array_payload_bytes=payload_before,
            audit_hash_ns=audit_hash_ns,
            audit_serialization_ns=audit_serialization_ns,
            token=token,
        )
        self._pending = _Pending(query, prepared)
        return query

    def commit_frame(self, query: F6FrameQuery) -> F6FrameCommit:
        pending = self._pending
        if pending is None or query is not pending.query:
            raise F6ContractError("commit requires the exact pending F6 query")
        if query.buffer_before != self._buffer_receipt(tuple(self._buffer[-MAX_BUFFERED_SUCCESSFUL_FRAMES:])):
            raise F6ContractError("F6 past buffer changed after query")
        if query.state_raw_array_payload_bytes != self.raw_array_payload_bytes:
            raise F6ContractError("F6 state payload changed after query")
        audit_hash_ns = 0
        for row in query.rows:
            if row.get("formal_score") != 1.0 or row.get("selected_hypothesis") not in {"H0", "HL", "HLG", "HB"}:
                raise F6ContractError("F6 result selection changed after query")
            geometry = row.get("selected_geometry")
            geometry_hash_started = time.perf_counter_ns()
            geometry_sha256 = (
                canonical_json_sha256(geometry)
                if isinstance(geometry, Mapping)
                else None
            )
            audit_hash_ns += time.perf_counter_ns() - geometry_hash_started
            if not isinstance(geometry, Mapping) or geometry_sha256 != row.get("selected_geometry_sha256"):
                raise F6ContractError("F6 selected geometry changed after query")
            result_hash_started = time.perf_counter_ns()
            result_sha256 = canonical_result_sha256(row)
            audit_hash_ns += time.perf_counter_ns() - result_hash_started
            if result_sha256 != row.get("result_sha256"):
                raise F6ContractError("F6 result hash changed after query")
        token_hash_started = time.perf_counter_ns()
        recomputed = canonical_json_sha256(
            {
                "protocol_id": PROTOCOL_ID,
                "frame_id": query.frame_id,
                "frame_ordinal": query.frame_ordinal,
                "buffer_before": query.buffer_before,
                "maximum_accessed_frame_ordinal": query.maximum_accessed_frame_ordinal,
                "state_raw_array_payload_bytes": query.state_raw_array_payload_bytes,
                "result_sha256": [row["result_sha256"] for row in query.rows],
            }
        )
        audit_hash_ns += time.perf_counter_ns() - token_hash_started
        if recomputed != query.token:
            raise F6ContractError("F6 query token changed after query")
        views = tuple(item.view for item in pending.prepared)
        self._buffer.append(_PastFrame(query.frame_id, query.frame_ordinal, views))
        self._buffer = self._buffer[-MAX_BUFFERED_SUCCESSFUL_FRAMES:]
        payload_after = self.raw_array_payload_bytes
        if payload_after > MAX_RAW_ARRAY_PAYLOAD_BYTES:
            raise F6ContractError("F6 committed raw array payload exceeds 2.5 MiB")
        self._seen_source_ids.update(row.source.source_id for row in pending.prepared)
        if pending.prepared and self._scene_id is None:
            match = _SOURCE_RE.fullmatch(pending.prepared[0].source.source_id)
            assert match is not None
            self._scene_id = match.group("scene")
        self._last_committed_ordinal = query.frame_ordinal
        self._pending = None
        return F6FrameCommit(
            frame_id=query.frame_id,
            frame_ordinal=query.frame_ordinal,
            source_count=len(query.rows),
            buffer_after=self._buffer_receipt(),
            state_raw_array_payload_bytes=payload_after,
            audit_hash_ns=audit_hash_ns,
            audit_serialization_ns=0,
            token=query.token,
        )

    def select_frame(self, *, frame_id: int, frame_ordinal: int, sources: Sequence[F6SourceEvidence]) -> tuple[F6FrameQuery, F6FrameCommit]:
        query = self.query_frame(frame_id=frame_id, frame_ordinal=frame_ordinal, sources=sources)
        return query, self.commit_frame(query)


def select_frame(state: F6SelectorState, *, frame_id: int, frame_ordinal: int, sources: Sequence[F6SourceEvidence]) -> tuple[F6FrameQuery, F6FrameCommit]:
    if not isinstance(state, F6SelectorState):
        raise F6ContractError("state must be an F6SelectorState")
    return state.select_frame(frame_id=frame_id, frame_ordinal=frame_ordinal, sources=sources)


__all__ = [
    "F6ContractError",
    "F6FrameCommit",
    "F6FrameQuery",
    "F6SelectorState",
    "F6SourceEvidence",
    "MAX_RAW_ARRAY_PAYLOAD_BYTES",
    "POLICY",
    "PROTOCOL_ID",
    "SCHEMA",
    "canonical_json_sha256",
    "canonical_result_sha256",
    "select_frame",
]
