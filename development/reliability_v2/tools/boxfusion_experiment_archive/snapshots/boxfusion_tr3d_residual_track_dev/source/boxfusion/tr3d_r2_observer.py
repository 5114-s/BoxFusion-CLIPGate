"""Pure-compute R2a scene observer for immutable TR3D proposals.

The observer is deliberately separated from cache serialization and every
active BoxFusion path.  It consumes parent TR3D yaw boxes, a strict causal
prefix manifest, and RGB-D frame resources.  Every manifest frame is first
used for projection only.  Depth is decoded lazily, once per frame, and only
when that frame belongs to at least one proposal's stable projected-area
Top-K.

No ground truth, CLIP feature, G0 prediction, score mutation, or proposal
mutation is accepted by this module.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
import time
from typing import Any, Optional

import numpy as np

from .tr3d_r2_geometry import (
    classify_depth_rays,
    compose_depth_camera_to_world,
    project_yaw_obb_to_depth,
    stable_top_k_view_indices,
)


R2_DEPTH_CLASS_NAMES = (
    "support",
    "occluded",
    "free_space",
    "invalid",
)
_DEPTH_CLASS_COUNT = len(R2_DEPTH_CLASS_NAMES)


def _readonly(value: object, dtype: np.dtype) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


def _finite_real(
    name: str,
    value: object,
    *,
    minimum: float,
    strict: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    valid_bound = result > minimum if strict else result >= minimum
    if not np.isfinite(result) or not valid_bound:
        relation = "greater than" if strict else "at least"
        raise ValueError(f"{name} must be finite and {relation} {minimum}")
    return result


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _image_shape(value: object) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("image_shape must be (height, width)")
    if len(value) != 2:
        raise ValueError("image_shape must be (height, width)")
    return (
        _positive_int("image height", value[0]),
        _positive_int("image width", value[1]),
    )


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class TR3DR2ObserverConfig:
    """Numerical and causal configuration for one R2a observer run."""

    image_shape: tuple[int, int]
    pose_source: str
    top_k: int = 3
    pixel_stride: int = 4
    depth_scale: float = 1000.0
    margin: float = 0.05
    min_depth: float = 0.10
    max_depth: float = 8.0
    near_clip: float = 1e-3

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_shape", _image_shape(self.image_shape))
        object.__setattr__(self, "pose_source", _text("pose_source", self.pose_source))
        object.__setattr__(self, "top_k", _positive_int("top_k", self.top_k))
        object.__setattr__(
            self,
            "pixel_stride",
            _positive_int("pixel_stride", self.pixel_stride),
        )
        object.__setattr__(
            self,
            "depth_scale",
            _finite_real("depth_scale", self.depth_scale, minimum=0.0, strict=True),
        )
        object.__setattr__(
            self,
            "margin",
            _finite_real("margin", self.margin, minimum=0.0),
        )
        object.__setattr__(
            self,
            "min_depth",
            _finite_real("min_depth", self.min_depth, minimum=0.0),
        )
        object.__setattr__(
            self,
            "max_depth",
            _finite_real("max_depth", self.max_depth, minimum=0.0),
        )
        if self.max_depth <= self.min_depth:
            raise ValueError("max_depth must exceed min_depth")
        object.__setattr__(
            self,
            "near_clip",
            _finite_real("near_clip", self.near_clip, minimum=0.0, strict=True),
        )


@dataclass(frozen=True)
class TR3DR2FrameBundle:
    """Depth/pose resources and depth-camera calibration for one scene.

    Values in ``depth`` are passed untouched to the injected depth decoder.
    Values in ``pose`` may be finite 4x4 arrays or paths accepted by the
    injected pose loader.  ``intrinsic_depth`` and ``extrinsic_depth`` may be
    arrays or text-matrix paths.
    """

    scene_id: str
    pose_source: str
    depth: Mapping[int, Any]
    pose: Mapping[int, Any]
    intrinsic_depth: Any
    extrinsic_depth: Any

    def __post_init__(self) -> None:
        _text("scene_id", self.scene_id)
        _text("frame bundle pose_source", self.pose_source)
        if not isinstance(self.depth, Mapping) or not isinstance(
            self.pose, Mapping
        ):
            raise ValueError("frame bundle depth and pose must be mappings")


@dataclass(frozen=True)
class TR3DR2Observation:
    """Explicit per-view and aggregate R2a pixel evidence.

    The four-count axis follows :data:`R2_DEPTH_CLASS_NAMES`.  Evidence is
    derived strictly as ``counts / point_count`` and therefore sums to one for
    every valid view/proposal with samples.  Invalid Top-K slots are ``-1`` in
    ``topk_frame_ids`` and zero everywhere else.
    """

    scene_id: str
    pose_source: str
    proposal_ids: np.ndarray
    used_frame_ids: np.ndarray
    decoded_frame_ids: np.ndarray
    topk_frame_ids: np.ndarray
    topk_view_valid: np.ndarray
    topk_projected_area_pixels: np.ndarray
    topk_projected_area_fraction: np.ndarray
    per_view_depth_counts: np.ndarray
    per_view_depth_evidence: np.ndarray
    per_view_point_count: np.ndarray
    aggregate_depth_counts: np.ndarray
    aggregate_depth_evidence: np.ndarray
    aggregate_view_count: np.ndarray
    aggregate_point_count: np.ndarray
    runtime_s: float

    @property
    def proposal_count(self) -> int:
        return int(self.proposal_ids.shape[0])

    @property
    def topk(self) -> int:
        return int(self.topk_frame_ids.shape[1])


def _resource_matrix(resource: object, name: str) -> np.ndarray:
    if isinstance(resource, (str, Path)):
        path = Path(resource)
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            value = np.loadtxt(path, dtype=np.float64)
        except (OSError, ValueError) as error:
            raise ValueError(f"{name}: failed to load 4x4 matrix") from error
    else:
        try:
            value = np.asarray(resource, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a finite 4x4 matrix") from error
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    return np.ascontiguousarray(value)


def _default_pose_loader(resource: object) -> np.ndarray:
    return _resource_matrix(resource, "pose")


def _default_depth_decoder(resource: object) -> np.ndarray:
    if not isinstance(resource, (str, Path)):
        raise ValueError(
            "default depth decoder requires a path; inject decode_depth for "
            "in-memory resources"
        )
    path = Path(resource)
    if not path.is_file():
        raise FileNotFoundError(path)
    from PIL import Image

    return np.asarray(Image.open(path))


def _normalized_resource_map(
    name: str, resources: Mapping[int, Any]
) -> dict[int, Any]:
    output: dict[int, Any] = {}
    for raw_frame_id, resource in resources.items():
        if isinstance(raw_frame_id, (bool, np.bool_)) or not isinstance(
            raw_frame_id, Integral
        ):
            raise ValueError(f"{name} frame ids must be non-negative integers")
        frame_id = int(raw_frame_id)
        if frame_id < 0 or frame_id in output:
            raise ValueError(f"{name} frame ids must be unique and non-negative")
        output[frame_id] = resource
    return output


def _strict_manifest_frame_ids(
    manifest: Mapping[str, Any],
    *,
    bundle: TR3DR2FrameBundle,
    config: TR3DR2ObserverConfig,
    depth_resources: Mapping[int, Any],
    pose_resources: Mapping[int, Any],
) -> np.ndarray:
    if not isinstance(manifest, Mapping):
        raise ValueError("prefix_manifest must be a mapping")
    if "used_frame_ids" not in manifest or "pose_source" not in manifest:
        raise ValueError(
            "prefix_manifest requires used_frame_ids and pose_source"
        )
    raw_ids = manifest["used_frame_ids"]
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError("used_frame_ids must be a non-empty JSON list")
    frame_ids: list[int] = []
    for value in raw_ids:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, Integral
        ):
            raise ValueError("used_frame_ids must contain non-negative integers")
        frame_ids.append(int(value))
    if any(value < 0 for value in frame_ids):
        raise ValueError("used_frame_ids must contain non-negative integers")
    if any(left >= right for left, right in zip(frame_ids, frame_ids[1:])):
        raise ValueError("used_frame_ids must be unique and strictly increasing")

    pose_source = _text("manifest pose_source", manifest["pose_source"])
    if pose_source != config.pose_source or pose_source != bundle.pose_source:
        raise ValueError("pose_source provenance mismatch")
    if "scene_id" in manifest and manifest["scene_id"] != bundle.scene_id:
        raise ValueError("prefix_manifest scene_id mismatch")

    missing_depth = [value for value in frame_ids if value not in depth_resources]
    missing_pose = [value for value in frame_ids if value not in pose_resources]
    if missing_depth or missing_pose:
        raise ValueError(
            "manifest frame is absent from frame bundle; "
            f"missing_depth={missing_depth[:8]}, missing_pose={missing_pose[:8]}"
        )
    available_ids = set(depth_resources) | set(pose_resources)
    if available_ids:
        lower, upper = min(available_ids), max(available_ids)
        if frame_ids[0] < lower or frame_ids[-1] > upper:
            raise ValueError("manifest frame id lies outside frame bundle bounds")
    return np.asarray(frame_ids, dtype=np.int64)


def _validated_parent(
    boxes_world: object, proposal_ids: object
) -> tuple[np.ndarray, np.ndarray]:
    try:
        boxes = np.asarray(boxes_world, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("boxes_world must be finite [P,7]") from error
    if boxes.ndim != 2 or boxes.shape[1] != 7 or not np.isfinite(boxes).all():
        raise ValueError("boxes_world must be finite [P,7]")
    if np.any(boxes[:, 3:6] <= 0.0):
        raise ValueError("boxes_world dimensions must be positive")
    raw_ids = np.asarray(proposal_ids)
    if raw_ids.dtype.kind not in "iu" or raw_ids.shape != (len(boxes),):
        raise ValueError("proposal_ids must be an integer [P] array")
    ids = raw_ids.astype(np.int64, copy=False)
    if np.any(ids < 0) or len(np.unique(ids)) != len(ids):
        raise ValueError("proposal_ids must be unique and non-negative")
    return np.ascontiguousarray(boxes), np.ascontiguousarray(ids)


def _decoded_depth(
    frame_id: int,
    resource: object,
    decoder: Callable[[object], object],
    config: TR3DR2ObserverConfig,
) -> np.ndarray:
    try:
        raw = np.asarray(decoder(resource))
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"frame {frame_id}: depth decode failed") from error
    if raw.shape != config.image_shape or raw.ndim != 2:
        raise ValueError(
            f"frame {frame_id}: depth shape {raw.shape} disagrees with "
            f"configured {config.image_shape}"
        )
    if not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"frame {frame_id}: depth image must be numeric")
    depth = raw.astype(np.float64, copy=False) / config.depth_scale
    # Non-finite pixels are legitimate sensor holes and are explicitly counted
    # as invalid by classify_depth_rays.  The array/scale itself remains strict.
    return np.ascontiguousarray(depth)


def _fractions_from_counts(
    counts: np.ndarray, totals: np.ndarray
) -> np.ndarray:
    evidence = np.zeros(counts.shape, dtype=np.float32)
    np.divide(
        counts,
        totals[..., None],
        out=evidence,
        where=totals[..., None] > 0,
        casting="unsafe",
    )
    return evidence


def observe_tr3d_r2_scene(
    *,
    boxes_world: object,
    proposal_ids: object,
    prefix_manifest: Mapping[str, Any],
    frame_bundle: TR3DR2FrameBundle,
    config: TR3DR2ObserverConfig,
    decode_depth: Optional[Callable[[object], object]] = None,
    load_pose: Optional[Callable[[object], object]] = None,
) -> TR3DR2Observation:
    """Observe immutable parent proposals using causal real-depth evidence."""

    started = time.perf_counter()
    if not isinstance(frame_bundle, TR3DR2FrameBundle):
        raise ValueError("frame_bundle must be TR3DR2FrameBundle")
    if not isinstance(config, TR3DR2ObserverConfig):
        raise ValueError("config must be TR3DR2ObserverConfig")
    boxes, ids = _validated_parent(boxes_world, proposal_ids)
    depth_resources = _normalized_resource_map("depth", frame_bundle.depth)
    pose_resources = _normalized_resource_map("pose", frame_bundle.pose)
    frame_ids = _strict_manifest_frame_ids(
        prefix_manifest,
        bundle=frame_bundle,
        config=config,
        depth_resources=depth_resources,
        pose_resources=pose_resources,
    )
    intrinsic = _resource_matrix(
        frame_bundle.intrinsic_depth, "intrinsic_depth"
    )
    extrinsic = _resource_matrix(
        frame_bundle.extrinsic_depth, "extrinsic_depth"
    )
    pose_decoder = _default_pose_loader if load_pose is None else load_pose
    depth_decoder = _default_depth_decoder if decode_depth is None else decode_depth

    depth_camera_to_world: dict[int, np.ndarray] = {}
    for frame_id in frame_ids.tolist():
        try:
            pose = np.asarray(pose_decoder(pose_resources[frame_id]), dtype=np.float64)
        except (OSError, TypeError, ValueError) as error:
            raise ValueError(f"frame {frame_id}: pose decode failed") from error
        depth_camera_to_world[frame_id] = compose_depth_camera_to_world(
            pose, extrinsic
        )

    proposal_count = len(boxes)
    frame_count = len(frame_ids)
    projected_area = np.zeros((proposal_count, frame_count), dtype=np.float64)
    projected_fraction = np.zeros(
        (proposal_count, frame_count), dtype=np.float64
    )
    projected_valid = np.zeros((proposal_count, frame_count), dtype=np.bool_)
    # Projection phase: no depth resource is decoded here.
    for frame_index, frame_id in enumerate(frame_ids.tolist()):
        camera_to_world = depth_camera_to_world[frame_id]
        for proposal_index, box in enumerate(boxes):
            projection = project_yaw_obb_to_depth(
                box,
                intrinsic,
                camera_to_world,
                config.image_shape,
                near_clip=config.near_clip,
            )
            if projection is not None:
                projected_area[proposal_index, frame_index] = (
                    projection.area_pixels
                )
                projected_fraction[proposal_index, frame_index] = (
                    projection.area_ratio
                )
                projected_valid[proposal_index, frame_index] = True

    topk_frame_ids = np.full(
        (proposal_count, config.top_k), -1, dtype=np.int64
    )
    topk_valid = np.zeros(
        (proposal_count, config.top_k), dtype=np.bool_
    )
    topk_area = np.zeros(
        (proposal_count, config.top_k), dtype=np.float32
    )
    topk_fraction = np.zeros_like(topk_area)
    selected_frame_indices = np.full(
        (proposal_count, config.top_k), -1, dtype=np.int64
    )
    for proposal_index in range(proposal_count):
        selected = stable_top_k_view_indices(
            projected_area[proposal_index],
            config.top_k,
            frame_ids=frame_ids,
            valid_mask=projected_valid[proposal_index],
        )
        valid_count = len(selected)
        if valid_count:
            selected_frame_indices[proposal_index, :valid_count] = selected
            topk_frame_ids[proposal_index, :valid_count] = frame_ids[selected]
            topk_valid[proposal_index, :valid_count] = True
            topk_area[proposal_index, :valid_count] = projected_area[
                proposal_index, selected
            ]
            topk_fraction[proposal_index, :valid_count] = projected_fraction[
                proposal_index, selected
            ]

    per_view_counts = np.zeros(
        (proposal_count, config.top_k, _DEPTH_CLASS_COUNT), dtype=np.int32
    )
    decoded: dict[int, np.ndarray] = {}
    decoded_order: list[int] = []
    # Classification phase: only stable Top-K frame ids reach the decoder.
    for proposal_index, box in enumerate(boxes):
        for slot in range(config.top_k):
            if not topk_valid[proposal_index, slot]:
                continue
            frame_id = int(topk_frame_ids[proposal_index, slot])
            if frame_id not in decoded:
                decoded[frame_id] = _decoded_depth(
                    frame_id,
                    depth_resources[frame_id],
                    depth_decoder,
                    config,
                )
                decoded_order.append(frame_id)
            classification = classify_depth_rays(
                decoded[frame_id],
                box,
                intrinsic,
                depth_camera_to_world[frame_id],
                pixel_stride=config.pixel_stride,
                margin=config.margin,
                min_depth=config.min_depth,
                max_depth=config.max_depth,
                near_clip=config.near_clip,
            )
            if classification is None or classification.sample_count < 1:
                raise ValueError(
                    f"proposal {int(ids[proposal_index])}/frame {frame_id}: "
                    "selected projection has no classifiable samples"
                )
            counts = np.asarray(
                [
                    classification.support_count,
                    classification.occluded_count,
                    classification.free_space_count,
                    classification.invalid_count,
                ],
                dtype=np.int64,
            )
            if np.any(counts > np.iinfo(np.int32).max):
                raise OverflowError("per-view pixel count exceeds int32")
            if int(counts.sum()) != classification.sample_count:
                raise AssertionError("R2a depth classes do not partition samples")
            per_view_counts[proposal_index, slot] = counts.astype(np.int32)

    per_view_point_count = per_view_counts.sum(axis=2, dtype=np.int32)
    per_view_evidence = _fractions_from_counts(
        per_view_counts, per_view_point_count
    )
    aggregate_counts = per_view_counts.sum(axis=1, dtype=np.int64)
    aggregate_point_count = aggregate_counts.sum(axis=1, dtype=np.int64)
    aggregate_evidence = _fractions_from_counts(
        aggregate_counts, aggregate_point_count
    )
    aggregate_view_count = topk_valid.sum(axis=1, dtype=np.int32)

    return TR3DR2Observation(
        scene_id=frame_bundle.scene_id,
        pose_source=frame_bundle.pose_source,
        proposal_ids=_readonly(ids, np.int64),
        used_frame_ids=_readonly(frame_ids, np.int64),
        decoded_frame_ids=_readonly(decoded_order, np.int64),
        topk_frame_ids=_readonly(topk_frame_ids, np.int64),
        topk_view_valid=_readonly(topk_valid, np.bool_),
        topk_projected_area_pixels=_readonly(topk_area, np.float32),
        topk_projected_area_fraction=_readonly(topk_fraction, np.float32),
        per_view_depth_counts=_readonly(per_view_counts, np.int32),
        per_view_depth_evidence=_readonly(per_view_evidence, np.float32),
        per_view_point_count=_readonly(per_view_point_count, np.int32),
        aggregate_depth_counts=_readonly(aggregate_counts, np.int64),
        aggregate_depth_evidence=_readonly(aggregate_evidence, np.float32),
        aggregate_view_count=_readonly(aggregate_view_count, np.int32),
        aggregate_point_count=_readonly(aggregate_point_count, np.int64),
        runtime_s=float(time.perf_counter() - started),
    )


__all__ = [
    "R2_DEPTH_CLASS_NAMES",
    "TR3DR2FrameBundle",
    "TR3DR2Observation",
    "TR3DR2ObserverConfig",
    "observe_tr3d_r2_scene",
]
