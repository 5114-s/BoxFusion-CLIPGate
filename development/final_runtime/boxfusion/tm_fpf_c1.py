"""Terminal target-mask face-wise probabilistic fusion (TM-FPF-C1).

The refiner is deliberately terminal-only and geometry-only.  It consumes
already associated target-mask RGB-D observations, chooses the single most
uncertain directed face of each final native box, and validates every proposed
face value on views other than the proposing view.  A failed validation returns
the native row bit-for-bit.  Scores are accepted only to enforce the terminal
row contract; they never participate in a geometry decision.

This module owns no online state and exposes no annotation/evaluator input.
It reuses CAPF's bounded face candidates and its three-state
surface/occluded/free-space ray comparison, but intentionally does *not* use
CAPF's rectangular-box depth sampler: TM-FPF requires explicit target masks.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from boxfusion.capf import (
    CAPF,
    CAPFFaceUpdate,
    DEFAULT_CAPF_CONFIG,
    box_to_local_faces,
)


SCHEMA = "boxfusion.tm_fpf_c1.v1"
PROTOCOL_ID = "TERMINAL-TARGET-MASK-FPF-C1-HELDOUT-V1"

_OWN_DEFAULTS = {
    "enabled": False,
    "minimum_mask_pixels": 32,
    "minimum_face_observations": 2,
    "minimum_normalized_face_uncertainty": 0.01,
    "maximum_views": 5,
    "mask_erosion_pixels": 1,
    "mask_match_min_box_iou": 0.20,
    "mask_match_min_containment": 0.60,
    "mask_match_min_native_coverage": 0.10,
    "mask_match_min_confidence": 0.0,
}

# These CAPF controls define the candidate and held-out evidence protocol.
# ``enabled`` and ``max_accepted_faces`` are fixed by TM-FPF-C1 itself.
_CAPF_KEYS = frozenset(DEFAULT_CAPF_CONFIG) - {"enabled", "max_accepted_faces"}
_ALLOWED_KEYS = frozenset(_OWN_DEFAULTS) | _CAPF_KEYS | {"max_accepted_faces"}


class TMFPFC1ContractError(ValueError):
    """An input or configuration violates the terminal C1 contract."""


@dataclass(frozen=True)
class TMFPFC1Config:
    enabled: bool
    minimum_mask_pixels: int
    minimum_face_observations: int
    minimum_normalized_face_uncertainty: float
    maximum_views: int
    mask_erosion_pixels: int
    mask_match_min_box_iou: float
    mask_match_min_containment: float
    mask_match_min_native_coverage: float
    mask_match_min_confidence: float
    capf: Mapping[str, object]


def resolve_tm_fpf_c1_config(box_fusion_cfg: Mapping) -> TMFPFC1Config:
    """Resolve ``box_fusion.tm_fpf_c1`` and freeze the C1 invariants."""

    if not isinstance(box_fusion_cfg, Mapping):
        raise TMFPFC1ContractError("box_fusion configuration must be a mapping")
    raw = box_fusion_cfg.get("tm_fpf_c1", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TMFPFC1ContractError("box_fusion.tm_fpf_c1 must be a mapping")
    raw = dict(raw)
    unknown = sorted(set(raw) - _ALLOWED_KEYS)
    if unknown:
        raise TMFPFC1ContractError(
            "unknown tm_fpf_c1 option(s): " + ", ".join(unknown)
        )
    if "max_accepted_faces" in raw and int(raw["max_accepted_faces"]) != 1:
        raise TMFPFC1ContractError("TM-FPF-C1 fixes max_accepted_faces to one")

    own = dict(_OWN_DEFAULTS)
    for key in own:
        if key in raw:
            own[key] = raw[key]
    own["enabled"] = bool(own["enabled"])
    for key in (
        "minimum_mask_pixels",
        "minimum_face_observations",
        "maximum_views",
    ):
        if isinstance(own[key], (bool, np.bool_)):
            raise TMFPFC1ContractError(f"tm_fpf_c1.{key} must be an integer")
        own[key] = int(own[key])
        if own[key] < 1:
            raise TMFPFC1ContractError(f"tm_fpf_c1.{key} must be positive")
    if isinstance(own["mask_erosion_pixels"], (bool, np.bool_)):
        raise TMFPFC1ContractError(
            "tm_fpf_c1.mask_erosion_pixels must be an integer"
        )
    own["mask_erosion_pixels"] = int(own["mask_erosion_pixels"])
    if own["mask_erosion_pixels"] < 0:
        raise TMFPFC1ContractError(
            "tm_fpf_c1.mask_erosion_pixels must be non-negative"
        )
    own["minimum_normalized_face_uncertainty"] = float(
        own["minimum_normalized_face_uncertainty"]
    )
    if (
        not math.isfinite(own["minimum_normalized_face_uncertainty"])
        or own["minimum_normalized_face_uncertainty"] < 0.0
    ):
        raise TMFPFC1ContractError(
            "tm_fpf_c1.minimum_normalized_face_uncertainty must be finite and non-negative"
        )
    for key in (
        "mask_match_min_box_iou",
        "mask_match_min_containment",
        "mask_match_min_native_coverage",
        "mask_match_min_confidence",
    ):
        own[key] = float(own[key])
        if not math.isfinite(own[key]) or not 0.0 <= own[key] <= 1.0:
            raise TMFPFC1ContractError(f"tm_fpf_c1.{key} must be in [0,1]")

    capf_raw = {
        key: raw[key]
        for key in _CAPF_KEYS
        if key in raw
    }
    capf_raw.update(
        {
            "enabled": own["enabled"],
            "max_accepted_faces": 1,
        }
    )
    # CAPF's resolver performs the remaining numerical/protocol validation.
    try:
        capf = CAPF({"capf": capf_raw}).cfg
    except (TypeError, ValueError) as error:
        raise TMFPFC1ContractError(str(error)) from error
    if own["maximum_views"] < int(capf["min_views"]):
        raise TMFPFC1ContractError(
            "tm_fpf_c1.maximum_views cannot be smaller than min_views"
        )
    if own["minimum_face_observations"] > own["maximum_views"]:
        raise TMFPFC1ContractError(
            "minimum_face_observations cannot exceed maximum_views"
        )
    return TMFPFC1Config(capf=MappingProxyType(dict(capf)), **own)


def _readonly(value: object, dtype, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        result = np.array(value, dtype=dtype, copy=True, order="C")
    except (TypeError, ValueError, OverflowError) as error:
        raise TMFPFC1ContractError(f"{label} must be numeric") from error
    if result.shape != shape or not np.isfinite(result).all():
        raise TMFPFC1ContractError(f"{label} must be finite with shape {shape}")
    result.setflags(write=False)
    return result


def _strict_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TMFPFC1ContractError(f"{label} must be an integer")
    result = int(value)
    if result < 0:
        raise TMFPFC1ContractError(f"{label} must be non-negative")
    return result


def _erode_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Dependency-free 3x3 binary erosion used to suppress mixed-depth edges."""

    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        eroded = np.ones_like(result)
        for dy in range(3):
            for dx in range(3):
                eroded &= padded[dy : dy + result.shape[0], dx : dx + result.shape[1]]
        result = eroded
    return result


def _boxes_xyxy(value: object, count: int, label: str) -> np.ndarray:
    try:
        boxes = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise TMFPFC1ContractError(f"{label} must be numeric") from error
    if boxes.shape != (count, 4) or not np.isfinite(boxes).all():
        raise TMFPFC1ContractError(f"{label} must be finite with shape [{count},4]")
    if count and np.any((boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])):
        raise TMFPFC1ContractError(f"{label} must have positive area")
    return boxes


def _clip_xyxy(boxes: np.ndarray, height: int, width: int) -> np.ndarray:
    result = boxes.copy()
    result[:, (0, 2)] = np.clip(result[:, (0, 2)], 0.0, float(width))
    result[:, (1, 3)] = np.clip(result[:, (1, 3)], 0.0, float(height))
    return result


def _box_iou_xyxy(left: np.ndarray, right: np.ndarray) -> float:
    lower = np.maximum(left[:2], right[:2])
    upper = np.minimum(left[2:], right[2:])
    intersection = float(np.prod(np.maximum(upper - lower, 0.0)))
    left_area = float(np.prod(np.maximum(left[2:] - left[:2], 0.0)))
    right_area = float(np.prod(np.maximum(right[2:] - right[:2], 0.0)))
    union = left_area + right_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def match_fastsam_target_masks(
    *,
    native_boxes_xyxy: object,
    automatic_masks: object,
    automatic_boxes_xyxy: object,
    automatic_confidences: object,
    config: TMFPFC1Config,
) -> tuple[int | None, ...]:
    """Deterministically match at most one automatic target mask per native row.

    Matching is one-to-one.  Both bounding-box overlap and the fraction of the
    mask contained by the native box must pass; native-pixel coverage prevents
    tiny mask fragments from being treated as a complete target.  Consequently
    a large background mask or a residual mask mostly outside native boxes has
    no eligible edge and produces ``None``.
    """

    masks = np.asarray(automatic_masks)
    if masks.ndim != 3:
        raise TMFPFC1ContractError("automatic_masks must have shape [M,H,W]")
    mask_count, height, width = masks.shape
    if height < 1 or width < 1:
        raise TMFPFC1ContractError("automatic_masks must have positive image size")
    if masks.dtype != np.bool_:
        if not np.issubdtype(masks.dtype, np.number) or not np.isfinite(masks).all():
            raise TMFPFC1ContractError("automatic_masks must be boolean or finite binary")
        if not np.all((masks == 0) | (masks == 1)):
            raise TMFPFC1ContractError("automatic_masks must be binary")
        masks = masks.astype(bool)
    native_array = np.asarray(native_boxes_xyxy)
    if native_array.ndim != 2 or native_array.shape[1:] != (4,):
        raise TMFPFC1ContractError("native_boxes_xyxy must have shape [N,4]")
    native_count = native_array.shape[0]
    native = _boxes_xyxy(native_boxes_xyxy, native_count, "native_boxes_xyxy")
    automatic = _boxes_xyxy(
        automatic_boxes_xyxy, mask_count, "automatic_boxes_xyxy"
    )
    try:
        confidences = np.asarray(automatic_confidences, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise TMFPFC1ContractError("automatic_confidences must be numeric") from error
    if (
        confidences.shape != (mask_count,)
        or not np.isfinite(confidences).all()
        or np.any((confidences < 0.0) | (confidences > 1.0))
    ):
        raise TMFPFC1ContractError(
            "automatic_confidences must be finite [M] values in [0,1]"
        )
    if native_count == 0 or mask_count == 0:
        return tuple(None for _ in range(native_count))

    clipped_native = _clip_xyxy(native, height, width)
    clipped_automatic = _clip_xyxy(automatic, height, width)
    edges = []
    for mask_index in range(mask_count):
        if confidences[mask_index] + 1e-12 < config.mask_match_min_confidence:
            continue
        mask = masks[mask_index]
        pixels = int(np.count_nonzero(mask))
        if pixels < config.minimum_mask_pixels:
            continue
        for native_index in range(native_count):
            box = clipped_native[native_index]
            x0 = int(np.floor(box[0]))
            y0 = int(np.floor(box[1]))
            x1 = int(np.ceil(box[2]))
            y1 = int(np.ceil(box[3]))
            if x1 <= x0 or y1 <= y0:
                continue
            inside = int(np.count_nonzero(mask[y0:y1, x0:x1]))
            containment = inside / max(pixels, 1)
            native_coverage = inside / max((x1 - x0) * (y1 - y0), 1)
            iou = _box_iou_xyxy(box, clipped_automatic[mask_index])
            if (
                iou + 1e-12 < config.mask_match_min_box_iou
                or containment + 1e-12 < config.mask_match_min_containment
                or native_coverage + 1e-12 < config.mask_match_min_native_coverage
            ):
                continue
            harmonic = 2.0 * iou * containment / max(iou + containment, 1e-12)
            edges.append(
                (
                    -harmonic,
                    -containment,
                    -iou,
                    -native_coverage,
                    -float(confidences[mask_index]),
                    native_index,
                    mask_index,
                )
            )
    edges.sort()
    result: list[int | None] = [None] * native_count
    used_masks: set[int] = set()
    for *_, native_index, mask_index in edges:
        if result[native_index] is not None or mask_index in used_masks:
            continue
        result[native_index] = mask_index
        used_masks.add(mask_index)
    return tuple(result)


@dataclass(frozen=True)
class TMFPFC1View:
    """One immutable target-mask RGB-D observation and its 3D proposal."""

    source_id: str
    frame_id: int
    observation_box_xyzlhw: np.ndarray
    observation_rotation: np.ndarray
    camera_to_world: np.ndarray
    target_surface_points_world: np.ndarray
    target_surface_valid: np.ndarray
    target_mask_pixel_count: int
    target_valid_depth_pixel_count: int
    target_mask_sha256: str
    evidence_kind: str = "target_mask_rgbd"

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise TMFPFC1ContractError("source_id must be a non-empty string")
        if self.source_id != self.source_id.strip():
            raise TMFPFC1ContractError("source_id must not contain surrounding whitespace")
        frame_id = _strict_nonnegative_int(self.frame_id, "frame_id")
        if self.evidence_kind != "target_mask_rgbd":
            raise TMFPFC1ContractError(
                "TM-FPF-C1 accepts only explicit target_mask_rgbd evidence"
            )
        box = _readonly(
            self.observation_box_xyzlhw,
            np.float64,
            (6,),
            "observation_box_xyzlhw",
        )
        if np.any(box[3:] <= 0.0):
            raise TMFPFC1ContractError("observation box extents must be positive")
        rotation = _readonly(
            self.observation_rotation,
            np.float64,
            (3, 3),
            "observation_rotation",
        )
        if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=5e-3):
            raise TMFPFC1ContractError("observation_rotation must be orthonormal")
        pose = _readonly(
            self.camera_to_world, np.float64, (4, 4), "camera_to_world"
        )
        points = np.array(
            self.target_surface_points_world,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        valid = np.array(self.target_surface_valid, dtype=bool, copy=True, order="C")
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise TMFPFC1ContractError(
                "target_surface_points_world must have shape [S,3]"
            )
        if valid.shape != points.shape[:1]:
            raise TMFPFC1ContractError(
                "target_surface_valid must align with target surface points"
            )
        if not np.isfinite(points[valid]).all():
            raise TMFPFC1ContractError("valid target surface points must be finite")
        points[~valid] = 0.0
        points.setflags(write=False)
        valid.setflags(write=False)
        mask_count = _strict_nonnegative_int(
            self.target_mask_pixel_count, "target_mask_pixel_count"
        )
        depth_count = _strict_nonnegative_int(
            self.target_valid_depth_pixel_count,
            "target_valid_depth_pixel_count",
        )
        if depth_count > mask_count:
            raise TMFPFC1ContractError(
                "target valid-depth pixels cannot exceed target-mask pixels"
            )
        if not isinstance(self.target_mask_sha256, str) or len(self.target_mask_sha256) != 64:
            raise TMFPFC1ContractError("target_mask_sha256 must be a SHA-256 hex digest")
        try:
            bytes.fromhex(self.target_mask_sha256)
        except ValueError as error:
            raise TMFPFC1ContractError(
                "target_mask_sha256 must be a SHA-256 hex digest"
            ) from error
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "observation_box_xyzlhw", box)
        object.__setattr__(self, "observation_rotation", rotation)
        object.__setattr__(self, "camera_to_world", pose)
        object.__setattr__(self, "target_surface_points_world", points)
        object.__setattr__(self, "target_surface_valid", valid)
        object.__setattr__(self, "target_mask_pixel_count", mask_count)
        object.__setattr__(self, "target_valid_depth_pixel_count", depth_count)


def make_target_mask_view(
    *,
    source_id: str,
    frame_id: int,
    observation_box_xyzlhw: object,
    observation_rotation: object,
    target_mask: object,
    depth_m: object,
    intrinsics: object,
    camera_to_world: object,
    config: TMFPFC1Config,
) -> TMFPFC1View:
    """Build bounded target-only ray evidence from one mask and RGB-D frame."""

    mask = np.asarray(target_mask)
    depth = np.asarray(depth_m, dtype=np.float64)
    if mask.ndim != 2 or depth.shape != mask.shape:
        raise TMFPFC1ContractError("target_mask and depth_m must share shape [H,W]")
    if mask.dtype != np.bool_:
        if not np.issubdtype(mask.dtype, np.number) or not np.isfinite(mask).all():
            raise TMFPFC1ContractError("target_mask must be boolean or finite binary")
        if not np.all((mask == 0) | (mask == 1)):
            raise TMFPFC1ContractError("target_mask must be binary")
        mask = mask.astype(bool)
    else:
        mask = mask.copy()
    raw_mask_count = int(np.count_nonzero(mask))
    if raw_mask_count < config.minimum_mask_pixels:
        raise TMFPFC1ContractError("target mask has too few pixels")
    sampled_mask = _erode_mask(mask, config.mask_erosion_pixels)
    valid_depth = (
        sampled_mask
        & np.isfinite(depth)
        & (depth >= float(config.capf["min_depth_m"]))
        & (depth <= float(config.capf["max_depth_m"]))
    )
    valid_depth_count = int(np.count_nonzero(valid_depth))
    if valid_depth_count < int(config.capf["min_valid_depth_samples"]):
        raise TMFPFC1ContractError("target mask has too few valid depth pixels")

    intrinsic = _readonly(intrinsics, np.float64, (3, 3), "intrinsics")
    pose = _readonly(camera_to_world, np.float64, (4, 4), "camera_to_world")
    if abs(float(intrinsic[0, 0])) <= 1e-12 or abs(float(intrinsic[1, 1])) <= 1e-12:
        raise TMFPFC1ContractError("intrinsics focal lengths must be non-zero")
    pixels = np.argwhere(valid_depth)
    maximum = int(config.capf["max_ray_samples"])
    if len(pixels) > maximum:
        keep = np.linspace(0, len(pixels) - 1, maximum, dtype=np.int64)
        pixels = pixels[keep]
    rows, columns = pixels[:, 0], pixels[:, 1]
    z = depth[rows, columns]
    camera_points = np.stack(
        (
            (columns.astype(np.float64) - intrinsic[0, 2]) * z / intrinsic[0, 0],
            (rows.astype(np.float64) - intrinsic[1, 2]) * z / intrinsic[1, 1],
            z,
        ),
        axis=1,
    )
    world_points = camera_points @ pose[:3, :3].T + pose[:3, 3]
    if not np.isfinite(world_points).all():
        raise TMFPFC1ContractError("target-mask backprojection produced non-finite points")
    points = np.zeros((maximum, 3), dtype=np.float64)
    point_valid = np.zeros(maximum, dtype=bool)
    points[: len(world_points)] = world_points
    point_valid[: len(world_points)] = True
    digest = hashlib.sha256(np.packbits(mask, bitorder="little").tobytes()).hexdigest()
    return TMFPFC1View(
        source_id=source_id,
        frame_id=frame_id,
        observation_box_xyzlhw=observation_box_xyzlhw,
        observation_rotation=observation_rotation,
        camera_to_world=pose,
        target_surface_points_world=points,
        target_surface_valid=point_valid,
        target_mask_pixel_count=raw_mask_count,
        target_valid_depth_pixel_count=valid_depth_count,
        target_mask_sha256=digest,
    )


@dataclass(frozen=True)
class TMFPFC1Decision:
    accepted: bool
    reason: str
    face_index: int | None
    normalized_face_uncertainty: float
    attempted_candidates: int
    update: CAPFFaceUpdate | None


@dataclass(frozen=True)
class TMFPFC1TerminalResult:
    boxes_xyzlhw: np.ndarray
    rotations: np.ndarray
    scores: np.ndarray
    decisions: tuple[TMFPFC1Decision, ...]
    accepted_count: int
    schema: str = SCHEMA
    protocol_id: str = PROTOCOL_ID
    online_writeback: bool = False


class TMFPFC1:
    """Stateless one-face refiner for final native rows."""

    def __init__(self, box_fusion_cfg: Mapping):
        self.config = resolve_tm_fpf_c1_config(box_fusion_cfg)
        self.enabled = self.config.enabled
        # The helper supplies bounded candidates and three-state held-out loss.
        self._capf = CAPF({"capf": dict(self.config.capf)})

    def _select_views(
        self, views: Sequence[TMFPFC1View]
    ) -> tuple[TMFPFC1View, ...]:
        if not views or any(not isinstance(row, TMFPFC1View) for row in views):
            return ()
        # Multiple masks from one frame are not independent views.  Retain the
        # one with the most complete metric target evidence deterministically.
        per_frame: dict[int, TMFPFC1View] = {}
        for row in views:
            previous = per_frame.get(row.frame_id)
            rank = (
                row.target_valid_depth_pixel_count,
                row.target_mask_pixel_count,
                row.source_id,
            )
            previous_rank = (
                -1,
                -1,
                "",
            ) if previous is None else (
                previous.target_valid_depth_pixel_count,
                previous.target_mask_pixel_count,
                previous.source_id,
            )
            if previous is None or rank > previous_rank:
                per_frame[row.frame_id] = row
        rows = sorted(per_frame.values(), key=lambda row: (row.frame_id, row.source_id))
        if len(rows) > self.config.maximum_views:
            # Keep the strongest target-depth observations, then restore time
            # order so leave-one-view-out behaviour is deterministic.
            rows = sorted(
                rows,
                key=lambda row: (
                    -row.target_valid_depth_pixel_count,
                    -row.target_mask_pixel_count,
                    row.frame_id,
                    row.source_id,
                ),
            )[: self.config.maximum_views]
            rows.sort(key=lambda row: (row.frame_id, row.source_id))
        return tuple(rows)

    def _reject(
        self,
        reason: str,
        *,
        face_index: int | None = None,
        uncertainty: float = 0.0,
        attempted: int = 0,
    ) -> TMFPFC1Decision:
        return TMFPFC1Decision(
            accepted=False,
            reason=reason,
            face_index=face_index,
            normalized_face_uncertainty=float(uncertainty),
            attempted_candidates=int(attempted),
            update=None,
        )

    def _refine_row(
        self,
        anchor_raw: np.ndarray,
        rotation_raw: np.ndarray,
        views: Sequence[TMFPFC1View],
    ) -> tuple[np.ndarray, TMFPFC1Decision]:
        fallback = anchor_raw.copy()
        if not self.enabled:
            return fallback, self._reject("disabled")
        selected = self._select_views(views)
        if len(selected) < int(self.config.capf["min_views"]):
            return fallback, self._reject("no_target_mask_evidence")
        try:
            anchor = np.asarray(anchor_raw, dtype=np.float64)
            rotation = np.asarray(rotation_raw, dtype=np.float64)
            if (
                anchor.shape != (6,)
                or rotation.shape != (3, 3)
                or not np.isfinite(anchor).all()
                or not np.isfinite(rotation).all()
                or np.any(anchor[3:] <= 0.0)
            ):
                return fallback, self._reject("invalid_anchor")
            boxes = np.stack([row.observation_box_xyzlhw for row in selected])
            rotations = np.stack([row.observation_rotation for row in selected])
            poses = np.stack([row.camera_to_world for row in selected])
            points = np.stack([row.target_surface_points_world for row in selected])
            valid = np.stack([row.target_surface_valid for row in selected])

            proposals: list[tuple[int, int, float]] = []
            for source_view, row in enumerate(selected):
                proposals.extend(
                    self._capf._source_face_candidates(
                        anchor,
                        rotation,
                        boxes[source_view],
                        rotations[source_view],
                        poses[source_view, :3, 3],
                        source_view,
                    )
                )
            if not proposals:
                return fallback, self._reject("no_visible_face_candidates")

            groups: dict[int, list[tuple[int, float]]] = {}
            for face_index, source_view, value in proposals:
                groups.setdefault(face_index, []).append((source_view, value))
            uncertainty_rows = []
            for face_index, rows in groups.items():
                distinct_sources = {source for source, _ in rows}
                if len(distinct_sources) < self.config.minimum_face_observations:
                    continue
                values = np.asarray([value for _, value in rows], dtype=np.float64)
                low, high = np.quantile(values, [0.25, 0.75])
                extent = max(float(anchor[3 + face_index // 2]), 1e-12)
                normalized = float((high - low) / extent)
                uncertainty_rows.append(
                    (-normalized, face_index, normalized, tuple(rows))
                )
            if not uncertainty_rows:
                return fallback, self._reject("insufficient_face_observations")
            uncertainty_rows.sort(key=lambda row: (row[0], row[1]))
            _, face_index, uncertainty, face_rows = uncertainty_rows[0]
            if (
                uncertainty + 1e-12
                < self.config.minimum_normalized_face_uncertainty
            ):
                return fallback, self._reject(
                    "insufficient_face_uncertainty",
                    face_index=face_index,
                    uncertainty=uncertainty,
                )

            anchor_faces = box_to_local_faces(anchor)
            accepted = []
            attempted = 0
            for source_view, proposed_value in face_rows:
                bounded = self._capf._bounded_candidate(
                    anchor,
                    rotation,
                    anchor_faces,
                    face_index,
                    proposed_value,
                )
                if bounded is None:
                    continue
                candidate, face_value = bounded
                attempted += 1
                score = self._capf._heldout_score(
                    anchor,
                    candidate,
                    rotation,
                    poses,
                    points,
                    valid,
                    source_view,
                )
                if score is None:
                    continue
                median, worst, heldout = score
                accepted.append(
                    (
                        -median,
                        -worst,
                        source_view,
                        face_value,
                        candidate,
                        median,
                        worst,
                        heldout,
                    )
                )
            if not accepted:
                return fallback, self._reject(
                    "no_heldout_improvement",
                    face_index=face_index,
                    uncertainty=uncertainty,
                    attempted=attempted,
                )
            accepted.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
            (
                _,
                _,
                source_view,
                face_value,
                candidate,
                median,
                worst,
                heldout,
            ) = accepted[0]
            output = np.asarray(candidate, dtype=anchor_raw.dtype)
            if (
                output.shape != fallback.shape
                or not np.isfinite(output).all()
                or np.any(output[3:] < float(self.config.capf["min_extent_m"]))
            ):
                return fallback, self._reject(
                    "invalid_final_candidate",
                    face_index=face_index,
                    uncertainty=uncertainty,
                    attempted=attempted,
                )
            update = CAPFFaceUpdate(
                face_index=face_index,
                source_view=source_view,
                heldout_views=heldout,
                face_value=face_value,
                median_loss_improvement=median,
                worst_loss_improvement=worst,
            )
            return output, TMFPFC1Decision(
                accepted=True,
                reason="accepted",
                face_index=face_index,
                normalized_face_uncertainty=uncertainty,
                attempted_candidates=attempted,
                update=update,
            )
        except (FloatingPointError, np.linalg.LinAlgError, TypeError, ValueError):
            return fallback, self._reject("invalid_or_incomparable_evidence")

    def refine_terminal(
        self,
        *,
        boxes_xyzlhw: object,
        rotations: object,
        scores: object,
        track_views: Sequence[Sequence[TMFPFC1View]],
    ) -> TMFPFC1TerminalResult:
        """Return terminal geometry copies while preserving row and score identity."""

        raw_boxes = np.asarray(boxes_xyzlhw)
        raw_rotations = np.asarray(rotations)
        raw_scores = np.asarray(scores)
        if raw_boxes.ndim != 2 or raw_boxes.shape[1:] != (6,):
            raise TMFPFC1ContractError("boxes_xyzlhw must have shape [N,6]")
        count = raw_boxes.shape[0]
        if raw_rotations.shape != (count, 3, 3):
            raise TMFPFC1ContractError("rotations must have shape [N,3,3]")
        if raw_scores.shape != (count,):
            raise TMFPFC1ContractError("scores must have shape [N]")
        if len(track_views) != count:
            raise TMFPFC1ContractError("track_views must align with terminal rows")
        if not np.issubdtype(raw_boxes.dtype, np.floating):
            raise TMFPFC1ContractError("boxes_xyzlhw must use a floating dtype")
        if not np.issubdtype(raw_rotations.dtype, np.floating):
            raise TMFPFC1ContractError("rotations must use a floating dtype")
        if not np.issubdtype(raw_scores.dtype, np.number):
            raise TMFPFC1ContractError("scores must use a numeric dtype")
        if (
            not np.isfinite(raw_boxes).all()
            or not np.isfinite(raw_rotations).all()
            or not np.isfinite(raw_scores).all()
            or np.any(raw_boxes[:, 3:] <= 0.0)
        ):
            raise TMFPFC1ContractError("terminal boxes, rotations and scores must be valid")

        output_boxes = np.array(raw_boxes, copy=True, order="C")
        output_rotations = np.array(raw_rotations, copy=True, order="C")
        output_scores = np.array(raw_scores, copy=True, order="C")
        decisions = []
        for index in range(count):
            refined, decision = self._refine_row(
                raw_boxes[index], raw_rotations[index], track_views[index]
            )
            output_boxes[index] = refined
            decisions.append(decision)

        # These invariants are checked at the module boundary, not left to an
        # evaluator or downstream caller.
        if output_boxes.shape != raw_boxes.shape:
            raise RuntimeError("TM-FPF-C1 changed terminal row count")
        if not np.array_equal(output_rotations, raw_rotations):
            raise RuntimeError("TM-FPF-C1 changed terminal rotations or row order")
        if not np.array_equal(output_scores, raw_scores):
            raise RuntimeError("TM-FPF-C1 changed a terminal score or row order")
        return TMFPFC1TerminalResult(
            boxes_xyzlhw=output_boxes,
            rotations=output_rotations,
            scores=output_scores,
            decisions=tuple(decisions),
            accepted_count=sum(row.accepted for row in decisions),
        )


__all__ = [
    "PROTOCOL_ID",
    "SCHEMA",
    "TMFPFC1",
    "TMFPFC1Config",
    "TMFPFC1ContractError",
    "TMFPFC1Decision",
    "TMFPFC1TerminalResult",
    "TMFPFC1View",
    "make_target_mask_view",
    "match_fastsam_target_masks",
    "resolve_tm_fpf_c1_config",
]
