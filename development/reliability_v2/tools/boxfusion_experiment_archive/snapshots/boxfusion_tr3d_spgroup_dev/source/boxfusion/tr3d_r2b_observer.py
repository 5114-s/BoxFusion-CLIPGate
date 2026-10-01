"""Pure-compute multi-view DINO feature observer for TR3D R2b.

R2b consumes the immutable Top-K view assignment produced by R2a.  For each
proposal/view it recomputes the R2a depth classification, maps only metric
depth *support* points into the RGB camera, and pools the corresponding cells
from one dense feature map per frame.  It never reads ground truth or CLIP
text features and it has no access to the active BoxFusion prediction path.

The feature extractor is injected.  This keeps the geometry and aggregation
contract testable without PyTorch and allows the online implementation to
reuse the exact ``dino0`` tensor already computed by Selective Boxer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Optional

import numpy as np

from .tr3d_r2_geometry import (
    classify_depth_rays,
    compose_depth_camera_to_world,
)
from .tr3d_r2_observer import TR3DR2ObserverConfig


def _readonly(value: object, dtype: np.dtype) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


def _matrix(resource: object, name: str) -> np.ndarray:
    if isinstance(resource, (str, Path)):
        path = Path(resource)
        if not path.is_file():
            raise FileNotFoundError(path)
        value = np.loadtxt(path, dtype=np.float64)
    else:
        value = np.asarray(resource, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite [4,4] matrix")
    return np.ascontiguousarray(value)


def _decode_image(resource: object, *, mode: str) -> np.ndarray:
    if not isinstance(resource, (str, Path)):
        value = np.asarray(resource)
    else:
        path = Path(resource)
        if not path.is_file():
            raise FileNotFoundError(path)
        from PIL import Image

        value = np.asarray(Image.open(path))
    if mode == "depth":
        if value.ndim != 2 or not np.issubdtype(value.dtype, np.number):
            raise ValueError("depth image must be numeric [H,W]")
    elif mode == "rgb":
        if value.ndim != 3 or value.shape[2] < 3:
            raise ValueError("RGB image must have shape [H,W,>=3]")
        value = value[..., :3]
    else:  # pragma: no cover - internal programming error
        raise AssertionError(mode)
    return np.ascontiguousarray(value)


def _normalize_feature_map(value: object) -> np.ndarray:
    """Return a finite contiguous ``[D,H,W]`` float32 dense feature map."""

    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    features = np.asarray(value)
    if features.ndim == 4 and features.shape[0] == 1:
        features = features[0]
    if (
        features.ndim != 3
        or min(features.shape) < 1
        or not np.issubdtype(features.dtype, np.number)
        or not np.isfinite(features).all()
    ):
        raise ValueError("dense features must be finite [D,H,W] or [1,D,H,W]")
    return np.ascontiguousarray(features, dtype=np.float32)


def _normalize_vector(value: np.ndarray, epsilon: float) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(value.astype(np.float64, copy=False)))
    if not np.isfinite(norm) or norm <= epsilon:
        return None
    return np.asarray(value / norm, dtype=np.float32)


def project_world_points_to_rgb(
    points_world: object,
    intrinsic_color: object,
    color_camera_to_world: object,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points into RGB pixels and return only in-frame rows.

    Returns ``(u, v, source_indices)``.  Pixel coordinates are continuous;
    no rounding is performed before mapping them onto the dense feature grid.
    """

    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_world must be finite [N,3]")
    intrinsic = np.asarray(intrinsic_color, dtype=np.float64)
    if intrinsic.shape == (4, 4):
        intrinsic = intrinsic[:3, :3]
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("intrinsic_color must be finite [3,3] or [4,4]")
    c2w = _matrix(color_camera_to_world, "color_camera_to_world")
    if len(points) == 0:
        empty = _readonly(np.zeros(0), np.float64)
        return empty, empty, _readonly(np.zeros(0), np.int64)
    w2c = np.linalg.inv(c2w)
    camera = points @ w2c[:3, :3].T + w2c[:3, 3]
    positive = camera[:, 2] > 1e-6
    u = intrinsic[0, 0] * camera[:, 0] / camera[:, 2] + intrinsic[0, 2]
    v = intrinsic[1, 1] * camera[:, 1] / camera[:, 2] + intrinsic[1, 2]
    height, width = image_shape
    inside = (
        positive
        & np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0.0)
        & (u < float(width))
        & (v >= 0.0)
        & (v < float(height))
    )
    indices = np.flatnonzero(inside)
    return (
        _readonly(u[indices], np.float64),
        _readonly(v[indices], np.float64),
        _readonly(indices, np.int64),
    )


def pool_supported_dense_features(
    features: object,
    support_u: object,
    support_v: object,
    *,
    source_image_shape: tuple[int, int],
    min_unique_cells: int = 1,
    norm_epsilon: float = 1e-12,
) -> tuple[Optional[np.ndarray], int]:
    """Pool unique dense cells touched by depth-supported RGB pixels.

    Mapping by normalized image coordinates makes this correct for both the
    native 640x480 image and Boxer's deliberately stretched 960x960 input.
    Every dense cell receives equal weight, preventing the R2a sampling stride
    from turning into an accidental appearance weight.
    """

    fmap = _normalize_feature_map(features)
    u = np.asarray(support_u, dtype=np.float64)
    v = np.asarray(support_v, dtype=np.float64)
    if u.ndim != 1 or v.shape != u.shape or not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError("support_u/support_v must be finite equal-length vectors")
    if min_unique_cells < 1:
        raise ValueError("min_unique_cells must be positive")
    source_h, source_w = source_image_shape
    if source_h < 1 or source_w < 1:
        raise ValueError("source_image_shape must be positive")
    if len(u) == 0:
        return None, 0
    _, feature_h, feature_w = fmap.shape
    cols = np.floor(u / float(source_w) * feature_w).astype(np.int64)
    rows = np.floor(v / float(source_h) * feature_h).astype(np.int64)
    valid = (
        (cols >= 0)
        & (cols < feature_w)
        & (rows >= 0)
        & (rows < feature_h)
    )
    linear = np.unique(rows[valid] * feature_w + cols[valid])
    count = int(len(linear))
    if count < min_unique_cells:
        return None, count
    pooled = fmap.reshape(fmap.shape[0], -1)[:, linear].mean(axis=1)
    normalized = _normalize_vector(pooled, norm_epsilon)
    return normalized, count


FEATURE_STAT_NAMES = (
    "pairwise_mean",
    "pairwise_median",
    "pairwise_minimum",
    "pairwise_maximum",
    "pairwise_std",
    "medoid_mean",
)


def feature_consistency_statistics(
    vectors: object, valid: object
) -> tuple[np.ndarray, int]:
    """Return deterministic pairwise cosine statistics for normalized views."""

    features = np.asarray(vectors, dtype=np.float64)
    mask = np.asarray(valid)
    if features.ndim != 2 or mask.shape != (features.shape[0],) or mask.dtype != np.bool_:
        raise ValueError("vectors/valid must be [K,D] and boolean [K]")
    selected = features[mask]
    if len(selected) < 2:
        return _readonly(np.zeros(len(FEATURE_STAT_NAMES)), np.float32), 0
    norms = np.linalg.norm(selected, axis=1)
    if not np.isfinite(selected).all() or not np.allclose(norms, 1.0, atol=1e-4, rtol=0.0):
        raise ValueError("valid feature vectors must be finite and L2-normalized")
    cosine = np.clip(selected @ selected.T, -1.0, 1.0)
    left, right = np.triu_indices(len(selected), k=1)
    pairs = cosine[left, right]
    row_mean = (cosine.sum(axis=1) - 1.0) / float(len(selected) - 1)
    result = np.asarray(
        [
            pairs.mean(),
            np.median(pairs),
            pairs.min(),
            pairs.max(),
            pairs.std(),
            row_mean.max(),
        ],
        dtype=np.float32,
    )
    return _readonly(result, np.float32), int(len(pairs))


@dataclass(frozen=True)
class TR3DR2BFrameBundle:
    scene_id: str
    pose_source: str
    color: Mapping[int, Any]
    depth: Mapping[int, Any]
    pose: Mapping[int, Any]
    intrinsic_depth: Any
    intrinsic_color: Any
    extrinsic_depth: Any
    extrinsic_color: Any


@dataclass(frozen=True)
class TR3DR2BObservation:
    scene_id: str
    proposal_ids: np.ndarray
    topk_frame_ids: np.ndarray
    topk_view_valid: np.ndarray
    feature_view_valid: np.ndarray
    per_view_feature_count: np.ndarray
    per_view_support_point_count: np.ndarray
    per_view_features: np.ndarray
    aggregate_feature_statistics: np.ndarray
    aggregate_feature_view_count: np.ndarray
    aggregate_feature_pair_count: np.ndarray
    decoded_rgb_frame_ids: np.ndarray
    decoded_depth_frame_ids: np.ndarray
    encoded_frame_ids: np.ndarray
    feature_runtime_s: float
    geometry_runtime_s: float
    total_runtime_s: float


def observe_tr3d_r2b_scene(
    *,
    boxes_world: object,
    proposal_ids: object,
    topk_frame_ids: object,
    topk_view_valid: object,
    frame_bundle: TR3DR2BFrameBundle,
    depth_config: TR3DR2ObserverConfig,
    encode_rgb: Callable[[np.ndarray], object],
    min_support_points: int = 2,
    min_feature_cells: int = 1,
    decode_rgb: Optional[Callable[[object], object]] = None,
    decode_depth: Optional[Callable[[object], object]] = None,
) -> TR3DR2BObservation:
    """Observe one scene without mutating proposals or any active output."""

    started = time.perf_counter()
    boxes = np.asarray(boxes_world, dtype=np.float64)
    ids = np.asarray(proposal_ids)
    frames = np.asarray(topk_frame_ids)
    valid = np.asarray(topk_view_valid)
    if boxes.ndim != 2 or boxes.shape[1] != 7 or not np.isfinite(boxes).all() or np.any(boxes[:, 3:6] <= 0):
        raise ValueError("boxes_world must be finite positive [P,7]")
    if ids.shape != (len(boxes),) or not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != len(ids):
        raise ValueError("proposal_ids must be unique integer [P]")
    if frames.ndim != 2 or frames.shape[0] != len(boxes):
        raise ValueError("topk_frame_ids must have shape [P,K]")
    if valid.shape != frames.shape or valid.dtype != np.bool_:
        raise ValueError("topk_view_valid must be boolean [P,K]")
    if np.any(frames[valid] < 0) or np.any(frames[~valid] != -1):
        raise ValueError("invalid Top-K slots must use frame id -1")
    if min_support_points < 1 or min_feature_cells < 1:
        raise ValueError("support/cell minima must be positive")

    color_resources = {int(k): v for k, v in frame_bundle.color.items()}
    depth_resources = {int(k): v for k, v in frame_bundle.depth.items()}
    pose_resources = {int(k): v for k, v in frame_bundle.pose.items()}
    needed = sorted(set(int(value) for value in frames[valid]))
    for frame_id in needed:
        if frame_id not in color_resources or frame_id not in depth_resources or frame_id not in pose_resources:
            raise ValueError(f"frame {frame_id}: missing RGB/depth/pose resource")

    intrinsic_depth = _matrix(frame_bundle.intrinsic_depth, "intrinsic_depth")
    intrinsic_color = _matrix(frame_bundle.intrinsic_color, "intrinsic_color")
    extrinsic_depth = _matrix(frame_bundle.extrinsic_depth, "extrinsic_depth")
    extrinsic_color = _matrix(frame_bundle.extrinsic_color, "extrinsic_color")
    rgb_decoder = (lambda x: _decode_image(x, mode="rgb")) if decode_rgb is None else decode_rgb
    depth_decoder = (lambda x: _decode_image(x, mode="depth")) if decode_depth is None else decode_depth

    proposal_count, topk = frames.shape
    per_view: Optional[np.ndarray] = None
    feature_dim: Optional[int] = None
    feature_valid = np.zeros((proposal_count, topk), dtype=np.bool_)
    feature_counts = np.zeros((proposal_count, topk), dtype=np.int32)
    support_counts = np.zeros((proposal_count, topk), dtype=np.int32)
    feature_runtime = 0.0
    geometry_runtime = 0.0
    for frame_id in needed:
        pose = _matrix(pose_resources[frame_id], f"pose[{frame_id}]")
        depth_c2w = compose_depth_camera_to_world(pose, extrinsic_depth)
        color_c2w = compose_depth_camera_to_world(pose, extrinsic_color)
        rgb = _decode_image(rgb_decoder(color_resources[frame_id]), mode="rgb")
        depth_raw = _decode_image(depth_decoder(depth_resources[frame_id]), mode="depth")
        depth = depth_raw.astype(np.float64, copy=False)
        if np.issubdtype(depth_raw.dtype, np.integer):
            depth = depth / float(depth_config.depth_scale)
        depth[~np.isfinite(depth)] = 0.0
        encoded_started = time.perf_counter()
        fmap = _normalize_feature_map(encode_rgb(rgb))
        feature_runtime += time.perf_counter() - encoded_started
        if feature_dim is None:
            feature_dim = int(fmap.shape[0])
            per_view = np.zeros(
                (proposal_count, topk, feature_dim), dtype=np.float32
            )
        elif fmap.shape[0] != feature_dim:
            raise ValueError("feature dimension changed between frames")

        # Process and release one dense map at a time.  A 960x960 DINO map is
        # about 5.5 MB in float32, whereas retaining a long ScanNet trajectory
        # would consume hundreds of MB and would not match online reuse.
        geometry_started = time.perf_counter()
        assigned = np.argwhere(valid & (frames == frame_id))
        for proposal_index, slot in assigned.tolist():
            box = boxes[proposal_index]
            classification = classify_depth_rays(
                depth,
                box,
                intrinsic_depth,
                depth_c2w,
                pixel_stride=depth_config.pixel_stride,
                margin=depth_config.margin,
                min_depth=depth_config.min_depth,
                max_depth=depth_config.max_depth,
                near_clip=depth_config.near_clip,
            )
            if classification is None:
                raise ValueError("R2b Top-K view no longer projects under R2a geometry")
            support_counts[proposal_index, slot] = classification.support_count
            if classification.support_count < min_support_points:
                continue
            u, v, _ = project_world_points_to_rgb(
                classification.support_points_world,
                intrinsic_color,
                color_c2w,
                rgb.shape[:2],
            )
            vector, count = pool_supported_dense_features(
                fmap,
                u,
                v,
                source_image_shape=rgb.shape[:2],
                min_unique_cells=min_feature_cells,
            )
            feature_counts[proposal_index, slot] = count
            if vector is not None:
                assert per_view is not None
                per_view[proposal_index, slot] = vector
                feature_valid[proposal_index, slot] = True
        geometry_runtime += time.perf_counter() - geometry_started

    if per_view is None:
        per_view = np.zeros((proposal_count, topk, 0), dtype=np.float32)

    stats = np.zeros((proposal_count, len(FEATURE_STAT_NAMES)), dtype=np.float32)
    view_counts = feature_valid.sum(axis=1, dtype=np.int32)
    pair_counts = np.zeros(proposal_count, dtype=np.int32)
    for proposal_index in range(proposal_count):
        row_stats, pair_count = feature_consistency_statistics(
            per_view[proposal_index], feature_valid[proposal_index]
        )
        stats[proposal_index] = row_stats
        pair_counts[proposal_index] = pair_count

    total_runtime = time.perf_counter() - started
    needed_array = np.asarray(needed, dtype=np.int64)
    return TR3DR2BObservation(
        scene_id=frame_bundle.scene_id,
        proposal_ids=_readonly(ids, np.int64),
        topk_frame_ids=_readonly(frames, np.int64),
        topk_view_valid=_readonly(valid, np.bool_),
        feature_view_valid=_readonly(feature_valid, np.bool_),
        per_view_feature_count=_readonly(feature_counts, np.int32),
        per_view_support_point_count=_readonly(support_counts, np.int32),
        per_view_features=_readonly(per_view, np.float32),
        aggregate_feature_statistics=_readonly(stats, np.float32),
        aggregate_feature_view_count=_readonly(view_counts, np.int32),
        aggregate_feature_pair_count=_readonly(pair_counts, np.int32),
        decoded_rgb_frame_ids=_readonly(needed_array, np.int64),
        decoded_depth_frame_ids=_readonly(needed_array, np.int64),
        encoded_frame_ids=_readonly(needed_array, np.int64),
        feature_runtime_s=float(feature_runtime),
        geometry_runtime_s=float(geometry_runtime),
        total_runtime_s=float(total_runtime),
    )


__all__ = [
    "FEATURE_STAT_NAMES",
    "TR3DR2BFrameBundle",
    "TR3DR2BObservation",
    "feature_consistency_statistics",
    "observe_tr3d_r2b_scene",
    "pool_supported_dense_features",
    "project_world_points_to_rgb",
]
