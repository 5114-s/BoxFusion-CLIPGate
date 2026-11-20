"""Deterministic GT-free past-only geometry selector for F5.

The selector consumes only sealed F2/F4 source evidence.  It chooses one of
``H0``, ``HL``, ``HLG`` or ``HB`` for every source identity, while retaining a
bounded buffer containing the previous three successful frames.  Decisions
are queried against a snapshot of committed past state and become visible to
later frames only after an exact-token commit.

There is deliberately no annotation, evaluator, native-prediction, semantic,
training, score calibration, proposal birth or output-mutation dependency in
this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import numpy as np


SCHEMA = "boxfusion.fastsam_f5_gtfree_selector.v1"
PROTOCOL_ID = "F5-GT-FREE-PAST-ONLY-GEOMETRY-SELECTOR-PAPER100"
MODE = "shadow"

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
NEAR_PLANE_M = 1.0e-4
ROBUST_MARGIN_M = 0.05
MAX_BUFFERED_SUCCESSFUL_FRAMES = 3
MAX_SOURCES_PER_FRAME = 16

BASE_MIN_RETAINED_POINTS = 16
BASE_MIN_RETAINED_FRACTION = 0.55
BASE_VOLUME_RATIO = (0.25, 1.05)
BASE_EXTENT_RATIO = (0.35, 1.05)
BASE_CENTER_DIAGONAL_FRACTION = 0.20
BASE_CENTER_ABSOLUTE_MARGIN_M = 0.05

HB_CONFIDENCE_MIN = 0.55
HB_MIN_POINT_COUNT = 16
HB_EXACT_SUPPORT_MIN = 0.60
HB_EXPANDED_SUPPORT_MIN = 0.80
HB_PROJECTION_IOU_MIN = 0.50
HB_BASE_ND_MAX = 0.50
HB_BASE_VOLUME_RATIO = (0.25, 4.00)
HB_BASE_IOU_MIN = 0.20
HB_BASE_CONTAINMENT_MIN = 0.70

HISTORY_ND_MAX = 0.50
HISTORY_IOU_MIN = 0.15
HISTORY_CONTAINMENT_MIN = 0.60
HB_HISTORY_IOU_MIN = 0.20
HB_HISTORY_CONTAINMENT_MIN = 0.60
MIN_CONFIRMING_PAST_FRAMES = 2

_SOURCE_RE = re.compile(
    r"^(?P<scene>scene[0-9]{4}_[0-9]{2})/"
    r"frame_(?P<frame>[0-9]{6})/raw_(?P<raw>[0-9]{3})$"
)
_CORNER_SIGNS = np.asarray(
    [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, +1.0],
        [-1.0, +1.0, -1.0],
        [-1.0, +1.0, +1.0],
        [+1.0, -1.0, -1.0],
        [+1.0, -1.0, +1.0],
        [+1.0, +1.0, -1.0],
        [+1.0, +1.0, +1.0],
    ],
    dtype=np.float64,
)
_CORNER_SIGNS.setflags(write=False)

POLICY: Mapping[str, object] = MappingProxyType(
    {
        "protocol_id": PROTOCOL_ID,
        "mode": MODE,
        "hypothesis_priority": ("HB", "HLG", "HL", "H0"),
        "base_candidate_order": ("HLG", "HL"),
        "base_min_retained_points": BASE_MIN_RETAINED_POINTS,
        "base_min_retained_fraction": BASE_MIN_RETAINED_FRACTION,
        "base_volume_ratio": BASE_VOLUME_RATIO,
        "base_extent_ratio": BASE_EXTENT_RATIO,
        "base_center_diagonal_fraction": BASE_CENTER_DIAGONAL_FRACTION,
        "base_center_absolute_margin_m": BASE_CENTER_ABSOLUTE_MARGIN_M,
        "hb_confidence_min": HB_CONFIDENCE_MIN,
        "hb_min_point_count": HB_MIN_POINT_COUNT,
        "hb_exact_support_min": HB_EXACT_SUPPORT_MIN,
        "hb_expanded_support_min": HB_EXPANDED_SUPPORT_MIN,
        "hb_projection_iou_min": HB_PROJECTION_IOU_MIN,
        "hb_base_nd_max": HB_BASE_ND_MAX,
        "hb_base_volume_ratio": HB_BASE_VOLUME_RATIO,
        "hb_base_iou_min": HB_BASE_IOU_MIN,
        "hb_base_containment_min": HB_BASE_CONTAINMENT_MIN,
        "history_nd_max": HISTORY_ND_MAX,
        "history_iou_min": HISTORY_IOU_MIN,
        "history_containment_min": HISTORY_CONTAINMENT_MIN,
        "hb_history_iou_min": HB_HISTORY_IOU_MIN,
        "hb_history_containment_min": HB_HISTORY_CONTAINMENT_MIN,
        "min_confirming_past_frames": MIN_CONFIRMING_PAST_FRAMES,
        "robust_margin_m": ROBUST_MARGIN_M,
        "max_buffered_successful_frames": MAX_BUFFERED_SUCCESSFUL_FRAMES,
        "max_sources_per_frame": MAX_SOURCES_PER_FRAME,
        "maximum_lookahead_frames": 0,
        "ground_truth": False,
        "annotation": False,
        "evaluator": False,
        "native_prediction_access": False,
        "training": False,
        "online_learning": False,
        "birth": False,
        "native_output_mutation": False,
        "formal_score": 1.0,
    }
)


class F5ContractError(RuntimeError):
    """Raised when sealed input or causal state violates the F5 contract."""


def canonical_json_sha256(value: object) -> str:
    """Hash finite ASCII JSON using one canonical encoding."""

    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise F5ContractError("value is not canonical finite ASCII JSON") from error
    return hashlib.sha256(payload).hexdigest()


def canonical_result_sha256(row: Mapping[str, Any]) -> str:
    """Hash one result row after removing its self-referential digest."""

    payload = dict(row)
    payload.pop("result_sha256", None)
    return canonical_json_sha256(payload)


def _strict_int(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise F5ContractError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise F5ContractError(f"{label} must be >= {minimum}")
    return result


def _finite_scalar(value: object, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise F5ContractError(f"{label} must be one scalar finite number")
    result = float(value)
    if not math.isfinite(result):
        raise F5ContractError(f"{label} must be one scalar finite number")
    return result


def _array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise F5ContractError(f"{label} must be a finite array of shape {shape}") from error
    if result.shape != shape or not np.isfinite(result).all():
        raise F5ContractError(f"{label} must be a finite array of shape {shape}")
    return np.array(result, dtype=np.float64, order="C", copy=True)


def _points(value: object) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise F5ContractError("points_world must be a finite [N,3] array") from error
    if result.ndim != 2 or result.shape[1:] != (3,) or not np.isfinite(result).all():
        raise F5ContractError("points_world must be a finite [N,3] array")
    return np.array(result, dtype=np.float64, order="C", copy=True)


def _aabb_from_row(row: object, label: str) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(row, Mapping) or row.get("valid") is not True:
        raise F5ContractError(f"{label} must be a valid sealed AABB hypothesis")
    lower = _array(row.get("q02"), (3,), f"{label}.q02")
    upper = _array(row.get("q98"), (3,), f"{label}.q98")
    if np.any(upper <= lower):
        raise F5ContractError(f"{label} must have strictly positive extents")
    expected_center = (lower + upper) * 0.5
    expected_extent = upper - lower
    center = _array(row.get("center"), (3,), f"{label}.center")
    extent = _array(row.get("extent"), (3,), f"{label}.extent")
    if not np.allclose(center, expected_center, rtol=0.0, atol=1.0e-9):
        raise F5ContractError(f"{label}.center differs from q02/q98")
    if not np.allclose(extent, expected_extent, rtol=0.0, atol=1.0e-9):
        raise F5ContractError(f"{label}.extent differs from q02/q98")
    return lower, upper


def _optional_aabb(row: object, label: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    try:
        return _aabb_from_row(row, label)
    except F5ContractError:
        return None


def _aabb_metrics(
    left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]
) -> tuple[float, float, float]:
    left_lower, left_upper = left
    right_lower, right_upper = right
    intersection_extent = np.maximum(
        np.minimum(left_upper, right_upper) - np.maximum(left_lower, right_lower),
        0.0,
    )
    intersection = float(np.prod(intersection_extent))
    left_volume = float(np.prod(left_upper - left_lower))
    right_volume = float(np.prod(right_upper - right_lower))
    union = left_volume + right_volume - intersection
    iou = 0.0 if union <= 0.0 else intersection / union
    containment = intersection / min(left_volume, right_volume)
    left_center = (left_lower + left_upper) * 0.5
    right_center = (right_lower + right_upper) * 0.5
    scale = max(
        float(np.linalg.norm(left_upper - left_lower)),
        float(np.linalg.norm(right_upper - right_lower)),
        0.02,
    )
    nd = float(np.linalg.norm(left_center - right_center)) / scale
    return float(iou), float(containment), float(nd)


def _xyxy_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.maximum(
        np.minimum(left[2:], right[2:]) - np.maximum(left[:2], right[:2]), 0.0
    )
    intersection_area = float(intersection[0] * intersection[1])
    left_area = float(np.prod(left[2:] - left[:2]))
    right_area = float(np.prod(right[2:] - right[:2]))
    union = left_area + right_area - intersection_area
    return 0.0 if union <= 0.0 else float(intersection_area / union)


def _geometry_aabb(geometry: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if geometry.get("kind") == "world_aabb":
        lower = _array(geometry.get("q02"), (3,), "selected geometry q02")
        upper = _array(geometry.get("q98"), (3,), "selected geometry q98")
    elif geometry.get("kind") == "world_obb":
        lower = _array(geometry.get("envelope_q02"), (3,), "HB envelope q02")
        upper = _array(geometry.get("envelope_q98"), (3,), "HB envelope q98")
    else:
        raise F5ContractError("selected geometry kind is invalid")
    if np.any(upper <= lower):
        raise F5ContractError("selected geometry envelope is degenerate")
    return lower, upper


def _aabb_geometry(name: str, row: Mapping[str, Any]) -> dict[str, Any]:
    lower, upper = _aabb_from_row(row, name)
    return {
        "kind": "world_aabb",
        "hypothesis": name,
        "q02": lower.tolist(),
        "q98": upper.tolist(),
        "center": ((lower + upper) * 0.5).tolist(),
        "extent": (upper - lower).tolist(),
    }


def _validate_hb_geometry(row: object) -> dict[str, Any]:
    if not isinstance(row, Mapping) or row.get("valid") is not True:
        raise F5ContractError("HB validity is false or absent")
    center = _array(row.get("world_center"), (3,), "HB.world_center")
    extent = _array(row.get("local_extent"), (3,), "HB.local_extent")
    rotation = _array(row.get("world_rotation"), (3, 3), "HB.world_rotation")
    corners = _array(row.get("world_corners"), (8, 3), "HB.world_corners")
    if np.any(extent <= 0.0):
        raise F5ContractError("HB.local_extent must be positive")
    if float(np.linalg.det(rotation)) <= 0.0 or not np.allclose(
        rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1.0e-3
    ):
        raise F5ContractError("HB.world_rotation is not right-handed orthonormal")
    expected = center[None, :] + (_CORNER_SIGNS * (extent[None, :] * 0.5)) @ rotation.T
    # F4 stores float32 model geometry serialized as JSON.  Its own validity
    # tolerance is reproduced here without refitting or changing the corners.
    if not np.allclose(corners, expected, rtol=0.0, atol=2.0e-6):
        raise F5ContractError("HB.world_corners differ from center/extent/rotation")
    camera_depth = _finite_scalar(row.get("camera_depth"), "HB.camera_depth")
    if camera_depth <= NEAR_PLANE_M:
        raise F5ContractError("HB.camera_depth is not in front of the camera")
    lower = corners.min(axis=0)
    upper = corners.max(axis=0)
    if np.any(upper <= lower):
        raise F5ContractError("HB world envelope is degenerate")
    return {
        "kind": "world_obb",
        "hypothesis": "HB",
        "world_center": center.tolist(),
        "local_extent": extent.tolist(),
        "world_rotation": rotation.tolist(),
        "world_corners": corners.tolist(),
        "envelope_q02": lower.tolist(),
        "envelope_q98": upper.tolist(),
        "envelope_center": ((lower + upper) * 0.5).tolist(),
        "envelope_extent": (upper - lower).tolist(),
    }


def _project_corners(
    corners_world: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[Optional[np.ndarray], Optional[float], str]:
    if camera_to_world.shape != (4, 4) or not np.isfinite(camera_to_world).all():
        raise F5ContractError("camera_to_world must be a finite [4,4] matrix")
    if not np.allclose(
        camera_to_world[3], np.asarray([0.0, 0.0, 0.0, 1.0]),
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise F5ContractError("camera_to_world must be affine")
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise F5ContractError("intrinsic must be a finite [3,3] matrix")
    if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
        raise F5ContractError("intrinsic focal lengths must be positive")
    try:
        world_to_camera = np.linalg.inv(camera_to_world)
    except np.linalg.LinAlgError as error:
        raise F5ContractError("camera_to_world is not invertible") from error
    homogeneous = np.column_stack((corners_world, np.ones(8, dtype=np.float64)))
    camera = (homogeneous @ world_to_camera.T)[:, :3]
    if not np.isfinite(camera).all() or not np.all(camera[:, 2] > NEAR_PLANE_M):
        return None, None, "projection_depth"
    projected = camera @ intrinsic.T
    pixels = projected[:, :2] / projected[:, 2:3]
    if not np.isfinite(pixels).all():
        return None, None, "projection_depth"
    x = np.clip(pixels[:, 0], 0.0, float(IMAGE_WIDTH))
    y = np.clip(pixels[:, 1], 0.0, float(IMAGE_HEIGHT))
    box = np.asarray([x.min(), y.min(), x.max(), y.max()], dtype=np.float64)
    if box[2] <= box[0] or box[3] <= box[1]:
        return None, None, "projection_iou"
    return box, float(camera[:, 2].min()), "valid"


@dataclass(frozen=True)
class F5SourceEvidence:
    """One sealed source and the current-frame evidence allowed by F5."""

    source_id: str
    frame_id: int
    frame_ordinal: int
    rank: int
    hypotheses: Mapping[str, Any]
    points_world: np.ndarray
    tight_box_xyxy: np.ndarray
    camera_to_world: np.ndarray
    intrinsic: np.ndarray
    source_lineage_sha256: str

    def __post_init__(self) -> None:
        match = _SOURCE_RE.fullmatch(self.source_id) if isinstance(self.source_id, str) else None
        if match is None:
            raise F5ContractError("source_id is not canonical")
        object.__setattr__(self, "frame_id", _strict_int(self.frame_id, "frame_id"))
        object.__setattr__(self, "frame_ordinal", _strict_int(self.frame_ordinal, "frame_ordinal"))
        object.__setattr__(self, "rank", _strict_int(self.rank, "rank"))
        if int(match.group("frame")) != self.frame_id:
            raise F5ContractError("source_id frame differs from frame_id")
        if self.rank >= MAX_SOURCES_PER_FRAME:
            raise F5ContractError("rank exceeds the sealed per-frame cap")
        if not isinstance(self.hypotheses, Mapping):
            raise F5ContractError("hypotheses must be a mapping")
        if "H0" not in self.hypotheses:
            raise F5ContractError("H0 is absent")
        _aabb_from_row(self.hypotheses["H0"], "H0")
        object.__setattr__(self, "points_world", _points(self.points_world))
        tight = _array(self.tight_box_xyxy, (4,), "tight_box_xyxy")
        if (
            tight[2] <= tight[0]
            or tight[3] <= tight[1]
            or tight[0] < 0.0
            or tight[1] < 0.0
            or tight[2] > IMAGE_WIDTH
            or tight[3] > IMAGE_HEIGHT
        ):
            raise F5ContractError("tight_box_xyxy lies outside the sealed frame")
        object.__setattr__(self, "tight_box_xyxy", tight)
        pose = _array(self.camera_to_world, (4, 4), "camera_to_world")
        if not np.allclose(
            pose[3], np.asarray([0.0, 0.0, 0.0, 1.0]), rtol=0.0, atol=1.0e-7
        ):
            raise F5ContractError("camera_to_world must be affine")
        try:
            np.linalg.inv(pose)
        except np.linalg.LinAlgError as error:
            raise F5ContractError("camera_to_world is not invertible") from error
        object.__setattr__(self, "camera_to_world", pose)
        intrinsic = _array(self.intrinsic, (3, 3), "intrinsic")
        if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
            raise F5ContractError("intrinsic focal lengths must be positive")
        object.__setattr__(self, "intrinsic", intrinsic)
        if (
            not isinstance(self.source_lineage_sha256, str)
            or len(self.source_lineage_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_lineage_sha256)
        ):
            raise F5ContractError("source_lineage_sha256 is invalid")


@dataclass(frozen=True)
class F5FrameQuery:
    """Immutable query result that must be committed with its exact token."""

    frame_id: int
    frame_ordinal: int
    rows: tuple[Mapping[str, Any], ...]
    buffer_before: tuple[Mapping[str, Any], ...]
    maximum_accessed_frame_ordinal: int
    token: str


@dataclass(frozen=True)
class F5FrameCommit:
    frame_id: int
    frame_ordinal: int
    source_count: int
    buffer_after: tuple[Mapping[str, Any], ...]
    token: str


def _base_hypothesis(source: F5SourceEvidence) -> tuple[str, dict[str, Any], dict[str, Any]]:
    h0_row = source.hypotheses["H0"]
    h0_lower, h0_upper = _aabb_from_row(h0_row, "H0")
    n0 = len(source.points_world)
    stored_count = h0_row.get("stored_point_count")
    if stored_count is not None and _strict_int(stored_count, "H0.stored_point_count") != n0:
        raise F5ContractError("H0 stored point count differs from sealed evidence")
    h0_extent = h0_upper - h0_lower
    h0_volume = float(np.prod(h0_extent))
    h0_diagonal = float(np.linalg.norm(h0_extent))
    attempts: dict[str, Any] = {}
    for name in ("HLG", "HL"):
        row = source.hypotheses.get(name)
        reason = "missing_or_invalid"
        metrics: dict[str, Any] = {}
        bounds = _optional_aabb(row, name)
        if bounds is not None and isinstance(row, Mapping):
            diagnostics = row.get("diagnostics")
            if not isinstance(diagnostics, Mapping):
                reason = "diagnostics"
            elif diagnostics.get("applied") is not True or diagnostics.get("fallback") is not False:
                reason = "applied_or_fallback"
            else:
                try:
                    retained = _strict_int(
                        diagnostics.get("retained_point_count"),
                        f"{name}.retained_point_count",
                    )
                except F5ContractError:
                    retained = -1
                required = max(
                    BASE_MIN_RETAINED_POINTS,
                    int(math.ceil(BASE_MIN_RETAINED_FRACTION * n0)),
                )
                lower, upper = bounds
                extent = upper - lower
                volume_ratio = float(np.prod(extent) / h0_volume)
                extent_ratio = extent / h0_extent
                center_shift = float(
                    np.linalg.norm((lower + upper) * 0.5 - (h0_lower + h0_upper) * 0.5)
                )
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
                elif not np.all(
                    (extent_ratio >= BASE_EXTENT_RATIO[0])
                    & (extent_ratio <= BASE_EXTENT_RATIO[1])
                ):
                    reason = "extent_ratio"
                elif center_shift > center_limit:
                    reason = "center_shift"
                else:
                    reason = "eligible"
                    attempts[name] = {"eligible": True, "reason": reason, **metrics}
                    return name, _aabb_geometry(name, row), {
                        "n0": n0,
                        "h0_volume": h0_volume,
                        "attempts": attempts,
                    }
        attempts[name] = {"eligible": False, "reason": reason, **metrics}
    return "H0", _aabb_geometry("H0", h0_row), {
        "n0": n0,
        "h0_volume": h0_volume,
        "attempts": attempts,
    }


@dataclass(frozen=True)
class _PreparedSource:
    evidence: F5SourceEvidence
    base_name: str
    base_geometry: Mapping[str, Any]
    base_diagnostics: Mapping[str, Any]
    base_aabb: tuple[np.ndarray, np.ndarray]
    input_hypothesis_sha256: Mapping[str, str]


@dataclass(frozen=True)
class _PastRow:
    source_id: str
    frame_id: int
    frame_ordinal: int
    rank: int
    selected_hypothesis: str
    geometry: Mapping[str, Any]
    aabb: tuple[np.ndarray, np.ndarray]
    result_sha256: str


@dataclass(frozen=True)
class _PastFrame:
    frame_id: int
    frame_ordinal: int
    rows: tuple[_PastRow, ...]


def _prepare(source: F5SourceEvidence) -> _PreparedSource:
    name, geometry, diagnostics = _base_hypothesis(source)
    hashes = {
        hypothesis: canonical_json_sha256(source.hypotheses.get(hypothesis))
        for hypothesis in ("H0", "HL", "HLG", "HB")
    }
    return _PreparedSource(
        evidence=source,
        base_name=name,
        base_geometry=geometry,
        base_diagnostics=diagnostics,
        base_aabb=_geometry_aabb(geometry),
        input_hypothesis_sha256=MappingProxyType(hashes),
    )


def _mutual_best_matches(
    current: Sequence[_PreparedSource], past: _PastFrame
) -> dict[int, _PastRow]:
    edges: list[tuple[int, int, tuple[float, float, float]]] = []
    for current_index, row in enumerate(current):
        for past_index, prior in enumerate(past.rows):
            iou, containment, nd = _aabb_metrics(row.base_aabb, prior.aabb)
            if nd <= HISTORY_ND_MAX and (
                iou >= HISTORY_IOU_MIN or containment >= HISTORY_CONTAINMENT_MIN
            ):
                edges.append((current_index, past_index, (iou, containment, -nd)))
    if not edges:
        return {}

    current_best: dict[int, tuple[int, tuple[float, float, float]]] = {}
    past_best: dict[int, tuple[int, tuple[float, float, float]]] = {}
    for current_index, past_index, affinity in edges:
        prior = past.rows[past_index]
        existing = current_best.get(current_index)
        if existing is None:
            current_best[current_index] = (past_index, affinity)
        else:
            old_prior = past.rows[existing[0]]
            if affinity > existing[1] or (
                affinity == existing[1]
                and (prior.rank, prior.source_id) < (old_prior.rank, old_prior.source_id)
            ):
                current_best[current_index] = (past_index, affinity)

        existing_past = past_best.get(past_index)
        current_row = current[current_index].evidence
        if existing_past is None:
            past_best[past_index] = (current_index, affinity)
        else:
            old_row = current[existing_past[0]].evidence
            if affinity > existing_past[1] or (
                affinity == existing_past[1]
                and (current_row.rank, current_row.source_id)
                < (old_row.rank, old_row.source_id)
            ):
                past_best[past_index] = (current_index, affinity)

    matches: dict[int, _PastRow] = {}
    for current_index, (past_index, _) in current_best.items():
        if past_best.get(past_index, (-1, ()))[0] == current_index:
            matches[current_index] = past.rows[past_index]
    return matches


def _hb_current_gates(prepared: _PreparedSource) -> tuple[Optional[dict[str, Any]], dict[str, Any], str]:
    source = prepared.evidence
    hb_row = source.hypotheses.get("HB")
    diagnostics: dict[str, Any] = {}
    try:
        hb_geometry = _validate_hb_geometry(hb_row)
    except F5ContractError as error:
        diagnostics["validity_error"] = str(error)
        return None, diagnostics, "validity"
    assert isinstance(hb_row, Mapping)

    confidence_raw = hb_row.get("confidence")
    try:
        confidence = _finite_scalar(confidence_raw, "HB.confidence")
    except F5ContractError:
        return None, diagnostics, "confidence_domain"
    diagnostics["boxer_confidence"] = confidence
    if not 0.0 <= confidence <= 1.0:
        return None, diagnostics, "confidence_domain"
    if confidence < HB_CONFIDENCE_MIN:
        return None, diagnostics, "confidence_threshold"

    points = source.points_world
    diagnostics["point_count"] = len(points)
    if len(points) < HB_MIN_POINT_COUNT:
        return None, diagnostics, "point_count"
    center = np.asarray(hb_geometry["world_center"], dtype=np.float64)
    extent = np.asarray(hb_geometry["local_extent"], dtype=np.float64)
    rotation = np.asarray(hb_geometry["world_rotation"], dtype=np.float64)
    local = (points - center[None, :]) @ rotation
    exact_support = float(
        np.count_nonzero(np.all(np.abs(local) <= extent[None, :] * 0.5, axis=1))
        / len(points)
    )
    diagnostics["exact_point_support"] = exact_support
    if exact_support < HB_EXACT_SUPPORT_MIN:
        return None, diagnostics, "exact_depth_support"
    expanded_support = float(
        np.count_nonzero(
            np.all(
                np.abs(local) <= extent[None, :] * 0.5 + ROBUST_MARGIN_M,
                axis=1,
            )
        )
        / len(points)
    )
    diagnostics["expanded_point_support"] = expanded_support
    if expanded_support < HB_EXPANDED_SUPPORT_MIN:
        return None, diagnostics, "expanded_depth_support"

    corners = np.asarray(hb_geometry["world_corners"], dtype=np.float64)
    projected, min_depth, projection_reason = _project_corners(
        corners, source.camera_to_world, source.intrinsic
    )
    diagnostics["minimum_projected_corner_depth_m"] = min_depth
    if projected is None:
        return None, diagnostics, projection_reason
    projection_iou = _xyxy_iou(projected, source.tight_box_xyxy)
    diagnostics["projected_box_xyxy"] = projected.tolist()
    diagnostics["projection_iou"] = projection_iou
    if projection_iou < HB_PROJECTION_IOU_MIN:
        return None, diagnostics, "projection_iou"

    hb_aabb = _geometry_aabb(hb_geometry)
    iou, containment, nd = _aabb_metrics(hb_aabb, prepared.base_aabb)
    hb_volume = float(np.prod(hb_aabb[1] - hb_aabb[0]))
    base_volume = float(np.prod(prepared.base_aabb[1] - prepared.base_aabb[0]))
    volume_ratio = hb_volume / base_volume
    diagnostics.update(
        {
            "hb_base_iou3d": iou,
            "hb_base_symmetric_containment": containment,
            "hb_base_normalized_center_distance": nd,
            "hb_base_envelope_volume_ratio": volume_ratio,
        }
    )
    if nd > HB_BASE_ND_MAX:
        return None, diagnostics, "center"
    if not HB_BASE_VOLUME_RATIO[0] <= volume_ratio <= HB_BASE_VOLUME_RATIO[1]:
        return None, diagnostics, "volume"
    if iou < HB_BASE_IOU_MIN and containment < HB_BASE_CONTAINMENT_MIN:
        return None, diagnostics, "base_overlap"
    return hb_geometry, diagnostics, "current_gates_passed"


def _select_prepared(
    prepared: _PreparedSource,
    matched_past: Sequence[_PastRow],
) -> dict[str, Any]:
    hb_geometry, hb_diagnostics, reason = _hb_current_gates(prepared)
    matched_rows = sorted(matched_past, key=lambda row: (row.frame_ordinal, row.rank, row.source_id))
    history_rows: list[dict[str, Any]] = []
    for past_row in matched_rows:
        base_iou, base_containment, base_nd = _aabb_metrics(
            prepared.base_aabb, past_row.aabb
        )
        history_rows.append(
            {
                "source_id": past_row.source_id,
                "frame_id": past_row.frame_id,
                "frame_ordinal": past_row.frame_ordinal,
                "rank": past_row.rank,
                "selected_hypothesis": past_row.selected_hypothesis,
                "result_sha256": past_row.result_sha256,
                "base_iou3d": base_iou,
                "base_symmetric_containment": base_containment,
                "base_normalized_center_distance": base_nd,
                "hb_consistency_evaluated": False,
                "hb_iou3d": None,
                "hb_symmetric_containment": None,
                "hb_normalized_center_distance": None,
                "passed_hb_consistency": None,
            }
        )
    history_pass_count = 0
    if hb_geometry is not None:
        if len({row.frame_ordinal for row in matched_rows}) < MIN_CONFIRMING_PAST_FRAMES:
            reason = "history_count"
        else:
            hb_aabb = _geometry_aabb(hb_geometry)
            for history_row, past_row in zip(history_rows, matched_rows, strict=True):
                iou, containment, nd = _aabb_metrics(hb_aabb, past_row.aabb)
                passed = nd <= HISTORY_ND_MAX and (
                    iou >= HB_HISTORY_IOU_MIN
                    or containment >= HB_HISTORY_CONTAINMENT_MIN
                )
                history_pass_count += int(passed)
                history_row.update(
                    {
                        "hb_consistency_evaluated": True,
                        "hb_iou3d": iou,
                        "hb_symmetric_containment": containment,
                        "hb_normalized_center_distance": nd,
                        "passed_hb_consistency": passed,
                    }
                )
            if history_pass_count < MIN_CONFIRMING_PAST_FRAMES:
                reason = "past_consistency"
            else:
                reason = "selected_hb"

    selected_name = "HB" if reason == "selected_hb" else prepared.base_name
    selected_geometry = hb_geometry if selected_name == "HB" else dict(prepared.base_geometry)
    assert selected_geometry is not None
    row: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "source_id": prepared.evidence.source_id,
        "source_lineage_sha256": prepared.evidence.source_lineage_sha256,
        "frame_id": prepared.evidence.frame_id,
        "frame_ordinal": prepared.evidence.frame_ordinal,
        "rank": prepared.evidence.rank,
        "input_hypothesis_sha256": dict(prepared.input_hypothesis_sha256),
        "base_hypothesis": prepared.base_name,
        "base_diagnostics": dict(prepared.base_diagnostics),
        "selected_hypothesis": selected_name,
        "selected_geometry": selected_geometry,
        "selected_geometry_sha256": canonical_json_sha256(selected_geometry),
        "formal_score": 1.0,
        "hb_abstention_reason": None if selected_name == "HB" else reason,
        "hb_diagnostics": hb_diagnostics,
        "matched_past": history_rows,
        "matched_past_frame_count": len({row.frame_ordinal for row in matched_rows}),
        "hb_consistent_past_frame_count": history_pass_count,
        "maximum_lookahead_frames": 0,
    }
    row["result_sha256"] = canonical_result_sha256(row)
    return row


class F5SelectorState:
    """Bounded causal selector state with query-before-commit semantics."""

    def __init__(self) -> None:
        self._buffer: list[_PastFrame] = []
        self._seen_source_ids: set[str] = set()
        self._pending: Optional[F5FrameQuery] = None
        self._last_committed_ordinal = -1
        self._scene_id: Optional[str] = None

    @property
    def buffered_frame_count(self) -> int:
        return len(self._buffer)

    @property
    def seen_source_count(self) -> int:
        return len(self._seen_source_ids)

    def _buffer_receipt(
        self, frames: Optional[Sequence[_PastFrame]] = None
    ) -> tuple[Mapping[str, Any], ...]:
        selected_frames = self._buffer if frames is None else frames
        return tuple(
            {
                "frame_id": frame.frame_id,
                "frame_ordinal": frame.frame_ordinal,
                "source_ids": [row.source_id for row in frame.rows],
                "result_sha256": [row.result_sha256 for row in frame.rows],
            }
            for frame in selected_frames
        )

    def query_frame(
        self,
        *,
        frame_id: int,
        frame_ordinal: int,
        sources: Sequence[F5SourceEvidence],
    ) -> F5FrameQuery:
        if self._pending is not None:
            raise F5ContractError("the previous F5 query has not been committed")
        frame_id = _strict_int(frame_id, "frame_id")
        frame_ordinal = _strict_int(frame_ordinal, "frame_ordinal")
        if frame_ordinal <= self._last_committed_ordinal:
            raise F5ContractError("frame ordinals must be strictly increasing")
        source_rows = tuple(sources)
        if len(source_rows) > MAX_SOURCES_PER_FRAME:
            raise F5ContractError("source count exceeds the sealed per-frame cap")
        if any(
            row.frame_id != frame_id or row.frame_ordinal != frame_ordinal
            for row in source_rows
        ):
            raise F5ContractError("source frame identity differs from query frame")
        ordered = tuple(sorted(source_rows, key=lambda row: (row.rank, row.source_id)))
        if ordered != source_rows:
            raise F5ContractError("sources must use frozen rank/source order")
        current_ids = [row.source_id for row in source_rows]
        if len(set(current_ids)) != len(current_ids):
            raise F5ContractError("duplicate source identity in current frame")
        if tuple(row.rank for row in source_rows) != tuple(range(len(source_rows))):
            raise F5ContractError("sources must use unique contiguous frozen ranks")
        if any(source_id in self._seen_source_ids for source_id in current_ids):
            raise F5ContractError("source identity was already committed")
        current_scenes = {
            _SOURCE_RE.fullmatch(source_id).group("scene")  # type: ignore[union-attr]
            for source_id in current_ids
        }
        if len(current_scenes) > 1:
            raise F5ContractError("one F5 state cannot mix scenes")
        if current_scenes:
            current_scene = next(iter(current_scenes))
            if self._scene_id is not None and current_scene != self._scene_id:
                raise F5ContractError("one F5 state cannot cross scene boundaries")

        # Expiry is based only on the current ordinal and already committed
        # state.  It cannot expose the current rows or a future frame.
        eligible_buffer = [
            frame
            for frame in self._buffer
            if frame_ordinal - frame.frame_ordinal <= MAX_BUFFERED_SUCCESSFUL_FRAMES
        ][-MAX_BUFFERED_SUCCESSFUL_FRAMES:]
        eligible_receipt = self._buffer_receipt(eligible_buffer)
        prepared = tuple(_prepare(row) for row in source_rows)
        matches_by_current: dict[int, list[_PastRow]] = {
            index: [] for index in range(len(prepared))
        }
        for past_frame in eligible_buffer:
            for current_index, past_row in _mutual_best_matches(prepared, past_frame).items():
                matches_by_current[current_index].append(past_row)
        rows = tuple(
            _select_prepared(row, matches_by_current[index])
            for index, row in enumerate(prepared)
        )
        maximum_accessed = max(
            (frame.frame_ordinal for frame in eligible_buffer), default=-1
        )
        token = canonical_json_sha256(
            {
                "protocol_id": PROTOCOL_ID,
                "frame_id": frame_id,
                "frame_ordinal": frame_ordinal,
                "buffer_before": eligible_receipt,
                "result_sha256": [row["result_sha256"] for row in rows],
            }
        )
        query = F5FrameQuery(
            frame_id=frame_id,
            frame_ordinal=frame_ordinal,
            rows=rows,
            buffer_before=eligible_receipt,
            maximum_accessed_frame_ordinal=maximum_accessed,
            token=token,
        )
        self._pending = query
        return query

    def commit_frame(self, query: F5FrameQuery) -> F5FrameCommit:
        if self._pending is None or query is not self._pending:
            raise F5ContractError("commit requires the exact pending F5 query")
        eligible_before_commit = [
            frame
            for frame in self._buffer
            if query.frame_ordinal - frame.frame_ordinal <= MAX_BUFFERED_SUCCESSFUL_FRAMES
        ][-MAX_BUFFERED_SUCCESSFUL_FRAMES:]
        if query.buffer_before != self._buffer_receipt(eligible_before_commit):
            raise F5ContractError("F5 past buffer changed after query")
        for row in query.rows:
            if not isinstance(row, Mapping):
                raise F5ContractError("F5 result row must be a mapping")
            if row.get("formal_score") != 1.0:
                raise F5ContractError("F5 formal score changed after query")
            if row.get("selected_hypothesis") not in {"H0", "HL", "HLG", "HB"}:
                raise F5ContractError("F5 selected hypothesis changed after query")
            geometry = row.get("selected_geometry")
            if not isinstance(geometry, Mapping):
                raise F5ContractError("F5 selected geometry changed after query")
            if canonical_json_sha256(geometry) != row.get("selected_geometry_sha256"):
                raise F5ContractError("F5 selected geometry hash changed after query")
            if canonical_result_sha256(row) != row.get("result_sha256"):
                raise F5ContractError("F5 result hash changed after query")
        recomputed_token = canonical_json_sha256(
            {
                "protocol_id": PROTOCOL_ID,
                "frame_id": query.frame_id,
                "frame_ordinal": query.frame_ordinal,
                "buffer_before": query.buffer_before,
                "result_sha256": [row["result_sha256"] for row in query.rows],
            }
        )
        if recomputed_token != query.token:
            raise F5ContractError("F5 query token changed after query")
        past_rows: list[_PastRow] = []
        for row in query.rows:
            geometry = row["selected_geometry"]
            assert isinstance(geometry, Mapping)
            past_rows.append(
                _PastRow(
                    source_id=str(row["source_id"]),
                    frame_id=int(row["frame_id"]),
                    frame_ordinal=int(row["frame_ordinal"]),
                    rank=int(row["rank"]),
                    selected_hypothesis=str(row["selected_hypothesis"]),
                    geometry=geometry,
                    aabb=_geometry_aabb(geometry),
                    result_sha256=str(row["result_sha256"]),
                )
            )
        self._buffer = [
            frame
            for frame in self._buffer
            if query.frame_ordinal - frame.frame_ordinal <= MAX_BUFFERED_SUCCESSFUL_FRAMES
        ]
        self._buffer.append(
            _PastFrame(query.frame_id, query.frame_ordinal, tuple(past_rows))
        )
        self._buffer = self._buffer[-MAX_BUFFERED_SUCCESSFUL_FRAMES:]
        self._seen_source_ids.update(str(row["source_id"]) for row in query.rows)
        if query.rows and self._scene_id is None:
            match = _SOURCE_RE.fullmatch(str(query.rows[0]["source_id"]))
            assert match is not None
            self._scene_id = match.group("scene")
        self._last_committed_ordinal = query.frame_ordinal
        self._pending = None
        return F5FrameCommit(
            frame_id=query.frame_id,
            frame_ordinal=query.frame_ordinal,
            source_count=len(query.rows),
            buffer_after=self._buffer_receipt(),
            token=query.token,
        )

    def select_frame(
        self,
        *,
        frame_id: int,
        frame_ordinal: int,
        sources: Sequence[F5SourceEvidence],
    ) -> tuple[F5FrameQuery, F5FrameCommit]:
        query = self.query_frame(
            frame_id=frame_id,
            frame_ordinal=frame_ordinal,
            sources=sources,
        )
        return query, self.commit_frame(query)


def select_frame(
    state: F5SelectorState,
    *,
    frame_id: int,
    frame_ordinal: int,
    sources: Sequence[F5SourceEvidence],
) -> tuple[F5FrameQuery, F5FrameCommit]:
    """Functional convenience wrapper around :class:`F5SelectorState`."""

    if not isinstance(state, F5SelectorState):
        raise F5ContractError("state must be an F5SelectorState")
    return state.select_frame(
        frame_id=frame_id,
        frame_ordinal=frame_ordinal,
        sources=sources,
    )


__all__ = [
    "F5ContractError",
    "F5FrameCommit",
    "F5FrameQuery",
    "F5SelectorState",
    "F5SourceEvidence",
    "POLICY",
    "PROTOCOL_ID",
    "SCHEMA",
    "canonical_json_sha256",
    "canonical_result_sha256",
    "select_frame",
]
