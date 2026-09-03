"""Online Mask-RGBD object geometry memory.

The implementation in this module deliberately depends only on NumPy.  Inputs
from PyTorch are accepted through a small duck-typed conversion helper, but
PyTorch is never imported.  The geometry convention is the ScanNet convention
used by BoxFusion: metric XYZ points in world coordinates and axis-aligned
boxes represented by ``center`` and positive ``dims``.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


DEFAULT_OBJECT_MEMORY_CONFIG = {
    # Safe integration default.  Constructing the classes explicitly still
    # works when disabled; callers use this flag to preserve their baseline.
    "enabled": False,
    # Metric depth filtering.
    "min_depth": 0.10,
    "max_depth": 6.00,
    "depth_scale": 1.0,
    "mask_threshold": 0.50,
    "mask_edge_margin": 1,
    "depth_edge_threshold": 0.15,
    # Per-observation and persistent point budgets.
    "voxel_size": 0.02,
    "max_points_per_observation": 2048,
    "max_points_per_object": 8192,
    # B3 uses a separate, per-view point memory for geometry refinement.
    # A value of zero preserves the legacy all-view geometry exactly.
    "top_k_views": 0,
    "max_view_candidates": 12,
    "view_diversity_weight": 0.25,
    "minimum_view_quality": 0.0,
    # Robust ScanNet-style world AABB.
    "aabb_lower_quantile": 0.02,
    "aabb_upper_quantile": 0.98,
    "min_points_for_aabb": 8,
    "minimum_aabb_dimension": 0.02,
    # Candidate-track lifecycle and geometry-only association.
    "min_confirmations": 2,
    "track_ttl": 10,
    "association_iou_threshold": 0.05,
    "association_center_distance": 0.75,
    "association_inside_fraction": 0.25,
}


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _strict_int(name: str, value: object, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _validated_pair_compatibility(
    pair_compatibility: Optional[Mapping[Tuple[int, int], object]],
) -> Optional[Dict[Tuple[int, int], float]]:
    """Return a strict, detached track/observation compatibility mapping."""

    if pair_compatibility is None:
        return None
    if not isinstance(pair_compatibility, Mapping):
        raise ValueError("pair_compatibility must be a mapping or None")

    normalized: Dict[Tuple[int, int], float] = {}
    for pair, value in pair_compatibility.items():
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(
                "pair_compatibility keys must be "
                "(track_id, observation_index) pairs"
            )
        track_id = _strict_int(
            "pair_compatibility track_id", pair[0], 0
        )
        observation_index = _strict_int(
            "pair_compatibility observation_index", pair[1], 0
        )
        normalized[(track_id, observation_index)] = _finite_float(
            "pair_compatibility value", value
        )
    return normalized


def resolve_object_memory_config(
    config: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Validate an object-memory configuration without silently accepting typos.

    Args:
        config: The ``object_memory`` subsection, not the complete application
            configuration.  Unknown keys are rejected.
    """

    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("object_memory config must be a mapping")

    unknown = sorted(set(config) - set(DEFAULT_OBJECT_MEMORY_CONFIG))
    if unknown:
        raise ValueError(
            "Unknown object_memory config key(s): " + ", ".join(unknown)
        )

    resolved = dict(DEFAULT_OBJECT_MEMORY_CONFIG)
    resolved.update(config)

    if not isinstance(resolved["enabled"], (bool, np.bool_)):
        raise ValueError("object_memory.enabled must be a boolean")
    resolved["enabled"] = bool(resolved["enabled"])

    for key in (
        "min_depth",
        "max_depth",
        "depth_scale",
        "mask_threshold",
        "voxel_size",
        "view_diversity_weight",
        "minimum_view_quality",
        "aabb_lower_quantile",
        "aabb_upper_quantile",
        "minimum_aabb_dimension",
        "association_iou_threshold",
        "association_center_distance",
        "association_inside_fraction",
    ):
        resolved[key] = _finite_float(f"object_memory.{key}", resolved[key])

    depth_edge_threshold = resolved["depth_edge_threshold"]
    if depth_edge_threshold is not None:
        depth_edge_threshold = _finite_float(
            "object_memory.depth_edge_threshold", depth_edge_threshold
        )
        if depth_edge_threshold <= 0.0:
            raise ValueError(
                "object_memory.depth_edge_threshold must be positive or null"
            )
    resolved["depth_edge_threshold"] = depth_edge_threshold

    resolved["mask_edge_margin"] = _strict_int(
        "object_memory.mask_edge_margin",
        resolved["mask_edge_margin"],
        0,
    )
    for key, minimum in (
        ("max_points_per_observation", 1),
        ("max_points_per_object", 1),
        ("top_k_views", 0),
        ("max_view_candidates", 1),
        ("min_points_for_aabb", 1),
        ("min_confirmations", 2),
        ("track_ttl", 0),
    ):
        resolved[key] = _strict_int(
            f"object_memory.{key}", resolved[key], minimum
        )

    if resolved["min_depth"] < 0.0:
        raise ValueError("object_memory.min_depth must be non-negative")
    if resolved["max_depth"] <= resolved["min_depth"]:
        raise ValueError(
            "object_memory.max_depth must exceed object_memory.min_depth"
        )
    if resolved["depth_scale"] <= 0.0:
        raise ValueError("object_memory.depth_scale must be positive")
    if not 0.0 <= resolved["mask_threshold"] <= 1.0:
        raise ValueError("object_memory.mask_threshold must lie in [0, 1]")
    if resolved["voxel_size"] < 0.0:
        raise ValueError("object_memory.voxel_size must be non-negative")
    if not 0.0 <= resolved["view_diversity_weight"] <= 1.0:
        raise ValueError(
            "object_memory.view_diversity_weight must lie in [0, 1]"
        )
    if not 0.0 <= resolved["minimum_view_quality"] <= 1.0:
        raise ValueError(
            "object_memory.minimum_view_quality must lie in [0, 1]"
        )
    if (
        resolved["top_k_views"] > 0
        and resolved["max_view_candidates"] < resolved["top_k_views"]
    ):
        raise ValueError(
            "object_memory.max_view_candidates must be at least top_k_views"
        )

    lower = resolved["aabb_lower_quantile"]
    upper = resolved["aabb_upper_quantile"]
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError(
            "object_memory AABB quantiles must satisfy 0 <= lower < upper <= 1"
        )
    if resolved["minimum_aabb_dimension"] <= 0.0:
        raise ValueError(
            "object_memory.minimum_aabb_dimension must be positive"
        )
    if (
        resolved["max_points_per_object"]
        < resolved["min_points_for_aabb"]
    ):
        raise ValueError(
            "object_memory.max_points_per_object must be at least "
            "min_points_for_aabb"
        )
    if not 0.0 <= resolved["association_iou_threshold"] <= 1.0:
        raise ValueError(
            "object_memory.association_iou_threshold must lie in [0, 1]"
        )
    if resolved["association_center_distance"] <= 0.0:
        raise ValueError(
            "object_memory.association_center_distance must be positive"
        )
    if not 0.0 <= resolved["association_inside_fraction"] <= 1.0:
        raise ValueError(
            "object_memory.association_inside_fraction must lie in [0, 1]"
        )

    return resolved


def _as_numpy(value: object, name: str) -> np.ndarray:
    """Convert NumPy or optional torch-like values without importing torch."""

    candidate = value
    if hasattr(candidate, "detach"):
        candidate = candidate.detach()
    if hasattr(candidate, "cpu"):
        candidate = candidate.cpu()
    if hasattr(candidate, "numpy"):
        candidate = candidate.numpy()
    try:
        return np.asarray(candidate)
    except Exception as error:
        raise ValueError(f"{name} cannot be converted to a NumPy array") from error


def _as_2d_image(value: object, name: str) -> np.ndarray:
    array = _as_numpy(value, name)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2 or min(array.shape) <= 0:
        raise ValueError(f"{name} must have shape [H, W]")
    return array


def _validate_image_shape(output_shape: Sequence[int]) -> Tuple[int, int]:
    if not isinstance(output_shape, Sequence) or len(output_shape) != 2:
        raise ValueError("output_shape must be (height, width)")
    height = _strict_int("output height", output_shape[0], 1)
    width = _strict_int("output width", output_shape[1], 1)
    return height, width


def resize_mask_nearest(
    mask: object,
    output_shape: Sequence[int],
    threshold: float = 0.5,
) -> np.ndarray:
    """Resize a mask with deterministic nearest-neighbour sampling.

    The returned mask is always boolean.  Integer and floating masks are
    thresholded after resize; non-finite mask values are rejected.
    """

    source = _as_2d_image(mask, "mask")
    if not (
        np.issubdtype(source.dtype, np.bool_)
        or np.issubdtype(source.dtype, np.number)
    ):
        raise ValueError("mask must contain boolean or numeric values")
    if np.issubdtype(source.dtype, np.number) and not np.isfinite(source).all():
        raise ValueError("mask must contain only finite values")

    threshold = _finite_float("mask threshold", threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("mask threshold must lie in [0, 1]")

    output_height, output_width = _validate_image_shape(output_shape)
    input_height, input_width = source.shape
    rows = (
        np.arange(output_height, dtype=np.int64) * input_height
    ) // output_height
    cols = (
        np.arange(output_width, dtype=np.int64) * input_width
    ) // output_width
    resized = source[rows[:, None], cols[None, :]]
    if np.issubdtype(resized.dtype, np.bool_):
        return resized.astype(bool, copy=True)
    return np.asarray(resized >= threshold, dtype=bool)


def erode_mask_edges(mask: object, margin: int) -> np.ndarray:
    """Remove a square ``margin``-pixel band along all mask boundaries."""

    binary = resize_mask_nearest(
        mask,
        _as_2d_image(mask, "mask").shape,
    )
    margin = _strict_int("mask edge margin", margin, 0)
    if margin == 0:
        return binary

    height, width = binary.shape
    padded = np.pad(binary, margin, mode="constant", constant_values=False)
    eroded = np.ones_like(binary, dtype=bool)
    diameter = 2 * margin + 1
    for row_offset in range(diameter):
        for col_offset in range(diameter):
            eroded &= padded[
                row_offset : row_offset + height,
                col_offset : col_offset + width,
            ]
    return eroded


def depth_discontinuity_mask(
    depth_meters: object,
    threshold: float,
) -> np.ndarray:
    """Mark both sides of four-neighbour metric-depth discontinuities."""

    depth = _as_2d_image(depth_meters, "depth").astype(
        np.float64, copy=False
    )
    threshold = _finite_float("depth edge threshold", threshold)
    if threshold <= 0.0:
        raise ValueError("depth edge threshold must be positive")

    usable = np.isfinite(depth) & (depth > 0.0)
    edges = np.zeros(depth.shape, dtype=bool)

    horizontal = (
        usable[:, :-1]
        & usable[:, 1:]
        & (np.abs(depth[:, :-1] - depth[:, 1:]) > threshold)
    )
    edges[:, :-1] |= horizontal
    edges[:, 1:] |= horizontal

    vertical = (
        usable[:-1, :]
        & usable[1:, :]
        & (np.abs(depth[:-1, :] - depth[1:, :]) > threshold)
    )
    edges[:-1, :] |= vertical
    edges[1:, :] |= vertical
    return edges


def _validated_intrinsics(intrinsics: object) -> np.ndarray:
    matrix = _as_numpy(intrinsics, "intrinsics").astype(
        np.float64, copy=False
    )
    if matrix.shape == (4, 4):
        matrix = matrix[:3, :3]
    if matrix.shape != (3, 3):
        raise ValueError("intrinsics must have shape [3, 3] or [4, 4]")
    if not np.isfinite(matrix).all():
        raise ValueError("intrinsics must contain only finite values")
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise ValueError("intrinsics focal lengths must be positive")
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("intrinsics must be invertible") from error
    if not np.isfinite(inverse).all():
        raise ValueError("intrinsics inverse must be finite")
    return matrix


def _validated_camera_to_world(camera_to_world: object) -> np.ndarray:
    pose = _as_numpy(camera_to_world, "camera_to_world").astype(
        np.float64, copy=False
    )
    if pose.ndim == 3 and pose.shape[0] == 1:
        pose = pose[0]
    if pose.shape != (4, 4):
        raise ValueError("camera_to_world must have shape [4, 4]")
    if not np.isfinite(pose).all():
        raise ValueError("camera_to_world must contain only finite values")
    if not np.allclose(
        pose[3],
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError("camera_to_world must be an affine transform")
    linear_determinant = float(np.linalg.det(pose[:3, :3]))
    if not np.isfinite(linear_determinant) or abs(linear_determinant) <= 1e-12:
        raise ValueError("camera_to_world must be invertible")
    return pose


def _validated_points(
    points: object,
    name: str = "points",
    allow_empty: bool = True,
) -> np.ndarray:
    array = _as_numpy(points, name).astype(np.float64, copy=False)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3]")
    if not allow_empty and array.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def voxel_downsample(points: object, voxel_size: float) -> np.ndarray:
    """Replace points in each voxel by their centroid, in sorted voxel order."""

    array = _validated_points(points)
    voxel_size = _finite_float("voxel_size", voxel_size)
    if voxel_size < 0.0:
        raise ValueError("voxel_size must be non-negative")
    if array.shape[0] == 0 or voxel_size == 0.0:
        return array.astype(np.float32, copy=True)

    scaled = array / voxel_size
    if np.max(np.abs(scaled), initial=0.0) > np.iinfo(np.int64).max / 4:
        raise ValueError("points are too large for the requested voxel_size")
    voxel_keys = np.floor(scaled).astype(np.int64)
    _, inverse, counts = np.unique(
        voxel_keys,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    centroids = np.empty((counts.shape[0], 3), dtype=np.float64)
    for axis in range(3):
        centroids[:, axis] = np.bincount(
            inverse,
            weights=array[:, axis],
            minlength=counts.shape[0],
        ) / counts
    return centroids.astype(np.float32)


def deterministic_bounded_sample(
    points: object,
    max_points: int,
) -> np.ndarray:
    """Return at most ``max_points`` in a deterministic spatial ordering."""

    array = _validated_points(points)
    max_points = _strict_int("max_points", max_points, 1)
    if array.shape[0] <= max_points:
        return array.astype(np.float32, copy=True)

    order = np.lexsort((array[:, 2], array[:, 1], array[:, 0]))
    sorted_points = array[order]
    positions = np.linspace(
        0,
        sorted_points.shape[0] - 1,
        num=max_points,
        dtype=np.int64,
    )
    return sorted_points[positions].astype(np.float32)


@dataclass(frozen=True)
class DepthPointObservation:
    """Result of one Mask-RGBD backprojection."""

    points_world: np.ndarray
    valid_pixel_mask: np.ndarray
    mask_pixels: int
    valid_depth_pixels: int
    raw_point_count: int
    voxel_point_count: int
    retained_point_count: int
    median_depth: Optional[float]

    @property
    def valid_depth_ratio(self) -> float:
        if self.mask_pixels <= 0:
            return 0.0
        return float(self.valid_depth_pixels) / float(self.mask_pixels)


def extract_masked_world_points(
    depth: object,
    mask: object,
    intrinsics: object,
    camera_to_world: object,
    config: Optional[Mapping[str, object]] = None,
) -> DepthPointObservation:
    """Backproject filtered, real-depth mask pixels into metric world XYZ."""

    cfg = resolve_object_memory_config(config)
    depth_array = _as_2d_image(depth, "depth").astype(
        np.float64, copy=False
    )
    intrinsics_array = _validated_intrinsics(intrinsics)
    pose = _validated_camera_to_world(camera_to_world)

    resized_mask = resize_mask_nearest(
        mask,
        depth_array.shape,
        threshold=float(cfg["mask_threshold"]),
    )
    mask_pixels = int(np.count_nonzero(resized_mask))
    filtered_mask = erode_mask_edges(
        resized_mask,
        int(cfg["mask_edge_margin"]),
    )

    depth_meters = depth_array / float(cfg["depth_scale"])
    valid = (
        filtered_mask
        & np.isfinite(depth_meters)
        & (depth_meters >= float(cfg["min_depth"]))
        & (depth_meters <= float(cfg["max_depth"]))
    )
    if cfg["depth_edge_threshold"] is not None:
        valid &= ~depth_discontinuity_mask(
            depth_meters,
            float(cfg["depth_edge_threshold"]),
        )

    rows, cols = np.nonzero(valid)
    accepted_depth = depth_meters[rows, cols]
    valid_depth_pixels = int(accepted_depth.shape[0])
    if accepted_depth.shape[0] == 0:
        empty = np.empty((0, 3), dtype=np.float32)
        return DepthPointObservation(
            points_world=empty,
            valid_pixel_mask=valid,
            mask_pixels=mask_pixels,
            valid_depth_pixels=0,
            raw_point_count=0,
            voxel_point_count=0,
            retained_point_count=0,
            median_depth=None,
        )

    pixels = np.column_stack(
        (
            cols.astype(np.float64),
            rows.astype(np.float64),
            np.ones(rows.shape[0], dtype=np.float64),
        )
    )
    rays = pixels @ np.linalg.inv(intrinsics_array).T
    camera_points = rays * accepted_depth[:, None]
    world_points = (
        camera_points @ pose[:3, :3].T
        + pose[:3, 3][None, :]
    )
    if not np.isfinite(world_points).all():
        raise ValueError("backprojection produced non-finite world points")

    voxel_points = voxel_downsample(
        world_points,
        float(cfg["voxel_size"]),
    )
    retained_points = deterministic_bounded_sample(
        voxel_points,
        int(cfg["max_points_per_observation"]),
    )
    return DepthPointObservation(
        points_world=retained_points,
        valid_pixel_mask=valid,
        mask_pixels=mask_pixels,
        valid_depth_pixels=valid_depth_pixels,
        raw_point_count=int(world_points.shape[0]),
        voxel_point_count=int(voxel_points.shape[0]),
        retained_point_count=int(retained_points.shape[0]),
        median_depth=float(np.median(accepted_depth)),
    )


def robust_quantile_aabb(
    points: object,
    lower_quantile: float = 0.02,
    upper_quantile: float = 0.98,
    min_points: int = 1,
    minimum_dimension: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate a robust axis-aligned box as ``(center, dims)``."""

    array = _validated_points(points, allow_empty=False)
    lower_quantile = _finite_float("lower_quantile", lower_quantile)
    upper_quantile = _finite_float("upper_quantile", upper_quantile)
    min_points = _strict_int("min_points", min_points, 1)
    minimum_dimension = _finite_float(
        "minimum_dimension", minimum_dimension
    )
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError(
            "quantiles must satisfy 0 <= lower < upper <= 1"
        )
    if minimum_dimension <= 0.0:
        raise ValueError("minimum_dimension must be positive")
    if array.shape[0] < min_points:
        raise ValueError(
            f"at least {min_points} points are required for an AABB"
        )

    bounds = np.quantile(
        array,
        [lower_quantile, upper_quantile],
        axis=0,
    )
    center = (bounds[0] + bounds[1]) * 0.5
    dims = np.maximum(bounds[1] - bounds[0], minimum_dimension)
    return center.astype(np.float32), dims.astype(np.float32)


def _validated_aabb(
    center: object,
    dims: object,
    prefix: str = "AABB",
) -> Tuple[np.ndarray, np.ndarray]:
    center_array = _as_numpy(center, f"{prefix} center").astype(
        np.float64, copy=False
    ).reshape(-1)
    dims_array = _as_numpy(dims, f"{prefix} dims").astype(
        np.float64, copy=False
    ).reshape(-1)
    if center_array.shape != (3,) or dims_array.shape != (3,):
        raise ValueError(f"{prefix} center and dims must each have shape [3]")
    if not np.isfinite(center_array).all() or not np.isfinite(dims_array).all():
        raise ValueError(f"{prefix} must contain only finite values")
    if np.any(dims_array <= 0.0):
        raise ValueError(f"{prefix} dims must be positive")
    return center_array, dims_array


def aabb_bounds(
    center: object,
    dims: object,
) -> Tuple[np.ndarray, np.ndarray]:
    center_array, dims_array = _validated_aabb(center, dims)
    half = dims_array * 0.5
    return (
        (center_array - half).astype(np.float32),
        (center_array + half).astype(np.float32),
    )


def aabb_iou(
    center_a: object,
    dims_a: object,
    center_b: object,
    dims_b: object,
) -> float:
    """Compute three-dimensional IoU between two positive-volume AABBs."""

    center_a, dims_a = _validated_aabb(center_a, dims_a, "first AABB")
    center_b, dims_b = _validated_aabb(center_b, dims_b, "second AABB")
    minimum_a = center_a - dims_a * 0.5
    maximum_a = center_a + dims_a * 0.5
    minimum_b = center_b - dims_b * 0.5
    maximum_b = center_b + dims_b * 0.5
    intersection_dims = np.maximum(
        np.minimum(maximum_a, maximum_b)
        - np.maximum(minimum_a, minimum_b),
        0.0,
    )
    intersection = float(np.prod(intersection_dims))
    volume_a = float(np.prod(dims_a))
    volume_b = float(np.prod(dims_b))
    union = volume_a + volume_b - intersection
    if union <= 0.0:
        return 0.0
    return float(np.clip(intersection / union, 0.0, 1.0))


def points_inside_aabb(
    points: object,
    center: object,
    dims: object,
    tolerance: float = 0.0,
) -> np.ndarray:
    """Return an inclusive point-in-AABB mask."""

    array = _validated_points(points)
    center_array, dims_array = _validated_aabb(center, dims)
    tolerance = _finite_float("tolerance", tolerance)
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    half = dims_array * 0.5 + tolerance
    return np.all(
        (array >= center_array[None, :] - half[None, :])
        & (array <= center_array[None, :] + half[None, :]),
        axis=1,
    )


def points_inside_aabb_fraction(
    points: object,
    center: object,
    dims: object,
    tolerance: float = 0.0,
) -> float:
    array = _validated_points(points)
    if array.shape[0] == 0:
        return 0.0
    return float(
        np.mean(points_inside_aabb(array, center, dims, tolerance))
    )


def aabb_corners(center: object, dims: object) -> np.ndarray:
    center_array, dims_array = _validated_aabb(center, dims)
    signs = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    return (
        center_array[None, :] + signs * dims_array[None, :] * 0.5
    ).astype(np.float32)


def project_aabb_to_image(
    center: object,
    dims: object,
    intrinsics: object,
    camera_to_world: object,
    image_shape: Sequence[int],
    near_clip: float = 1e-3,
    require_all_in_front: bool = True,
) -> Optional[np.ndarray]:
    """Project a world AABB to a clipped ``[x1, y1, x2, y2]`` image box."""

    height, width = _validate_image_shape(image_shape)
    intrinsic = _validated_intrinsics(intrinsics)
    pose = _validated_camera_to_world(camera_to_world)
    near_clip = _finite_float("near_clip", near_clip)
    if near_clip <= 0.0:
        raise ValueError("near_clip must be positive")
    if not isinstance(require_all_in_front, (bool, np.bool_)):
        raise ValueError("require_all_in_front must be a boolean")

    corners_world = aabb_corners(center, dims).astype(np.float64)
    world_to_camera = np.linalg.inv(pose)
    corners_homogeneous = np.column_stack(
        (corners_world, np.ones(8, dtype=np.float64))
    )
    corners_camera = (
        corners_homogeneous @ world_to_camera.T
    )[:, :3]
    in_front = corners_camera[:, 2] > near_clip
    if not np.any(in_front) or (
        bool(require_all_in_front) and not np.all(in_front)
    ):
        return None

    corners_camera = corners_camera[in_front]
    projected = corners_camera @ intrinsic.T
    pixels = projected[:, :2] / projected[:, 2:3]
    x = np.clip(pixels[:, 0], 0.0, float(width))
    y = np.clip(pixels[:, 1], 0.0, float(height))
    box = np.asarray(
        [x.min(), y.min(), x.max(), y.max()],
        dtype=np.float32,
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def bbox_mask_iou(
    box_xyxy: object,
    mask: object,
    threshold: float = 0.5,
) -> float:
    """Rasterize a continuous XYXY box and compute IoU with a binary mask."""

    mask_array = _as_2d_image(mask, "mask")
    binary = resize_mask_nearest(
        mask_array,
        mask_array.shape,
        threshold=threshold,
    )
    box = _as_numpy(box_xyxy, "box_xyxy").astype(
        np.float64, copy=False
    ).reshape(-1)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("box_xyxy must contain four finite values")
    if box[2] < box[0] or box[3] < box[1]:
        raise ValueError("box_xyxy maximums must not be below minimums")

    height, width = binary.shape
    x_start = max(0, min(width, int(np.floor(box[0]))))
    y_start = max(0, min(height, int(np.floor(box[1]))))
    x_stop = max(0, min(width, int(np.ceil(box[2]))))
    y_stop = max(0, min(height, int(np.ceil(box[3]))))
    box_area = max(x_stop - x_start, 0) * max(y_stop - y_start, 0)
    mask_area = int(np.count_nonzero(binary))
    if box_area == 0 and mask_area == 0:
        return 0.0
    intersection = int(
        np.count_nonzero(binary[y_start:y_stop, x_start:x_stop])
    )
    union = box_area + mask_area - intersection
    if union <= 0:
        return 0.0
    return float(intersection / union)


def projected_aabb_mask_iou(
    center: object,
    dims: object,
    intrinsics: object,
    camera_to_world: object,
    mask: object,
    threshold: float = 0.5,
    near_clip: float = 1e-3,
) -> float:
    mask_array = _as_2d_image(mask, "mask")
    projected_box = project_aabb_to_image(
        center,
        dims,
        intrinsics,
        camera_to_world,
        mask_array.shape,
        near_clip=near_clip,
    )
    if projected_box is None:
        return 0.0
    return bbox_mask_iou(projected_box, mask_array, threshold=threshold)


@dataclass(frozen=True)
class ObjectObservation:
    """A finite world-point observation and its bounded quality metadata."""

    points_world: np.ndarray
    confidence: float = 1.0
    mask_pixels: Optional[int] = None
    valid_depth_pixels: Optional[int] = None
    projection_mask_iou: float = 1.0
    camera_position: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        points = _validated_points(self.points_world)
        object.__setattr__(
            self,
            "points_world",
            points.astype(np.float32, copy=True),
        )

        confidence = _finite_float("observation confidence", self.confidence)
        projection_iou = _finite_float(
            "observation projection_mask_iou",
            self.projection_mask_iou,
        )
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("observation confidence must lie in [0, 1]")
        if not 0.0 <= projection_iou <= 1.0:
            raise ValueError(
                "observation projection_mask_iou must lie in [0, 1]"
            )
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "projection_mask_iou", projection_iou)

        camera_position = self.camera_position
        if camera_position is not None:
            camera_position = _as_numpy(
                camera_position, "camera_position"
            ).astype(np.float32, copy=False)
            if camera_position.shape != (3,):
                raise ValueError("camera_position must have shape [3]")
            if not np.isfinite(camera_position).all():
                raise ValueError("camera_position must contain finite values")
            camera_position = camera_position.copy()
        object.__setattr__(self, "camera_position", camera_position)

        mask_pixels = (
            points.shape[0]
            if self.mask_pixels is None
            else _strict_int("observation mask_pixels", self.mask_pixels, 0)
        )
        valid_depth_pixels = (
            points.shape[0]
            if self.valid_depth_pixels is None
            else _strict_int(
                "observation valid_depth_pixels",
                self.valid_depth_pixels,
                0,
            )
        )
        if valid_depth_pixels > mask_pixels:
            raise ValueError(
                "observation valid_depth_pixels cannot exceed mask_pixels"
            )
        object.__setattr__(self, "mask_pixels", mask_pixels)
        object.__setattr__(self, "valid_depth_pixels", valid_depth_pixels)

    @property
    def valid_depth_ratio(self) -> float:
        if not self.mask_pixels:
            return 0.0
        return float(self.valid_depth_pixels) / float(self.mask_pixels)

    @property
    def quality(self) -> float:
        return float(
            self.confidence
            * self.valid_depth_ratio
            * self.projection_mask_iou
        )

    @classmethod
    def from_depth_observation(
        cls,
        observation: DepthPointObservation,
        confidence: float = 1.0,
        projection_mask_iou: float = 1.0,
        camera_position: Optional[object] = None,
    ) -> "ObjectObservation":
        return cls(
            points_world=observation.points_world,
            confidence=confidence,
            mask_pixels=observation.mask_pixels,
            valid_depth_pixels=observation.valid_depth_pixels,
            projection_mask_iou=projection_mask_iou,
            camera_position=camera_position,
        )


@dataclass(frozen=True)
class MemoryViewRecord:
    """One independently retained Mask-RGBD view for B3 geometry."""

    frame_id: int
    points_world: np.ndarray
    quality: float
    confidence: float
    valid_depth_ratio: float
    projection_mask_iou: float
    camera_position: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        frame_id = _strict_int("memory view frame_id", self.frame_id, 0)
        points = _validated_points(self.points_world)
        quality = _finite_float("memory view quality", self.quality)
        confidence = _finite_float(
            "memory view confidence", self.confidence
        )
        valid_depth_ratio = _finite_float(
            "memory view valid_depth_ratio", self.valid_depth_ratio
        )
        projection_iou = _finite_float(
            "memory view projection_mask_iou",
            self.projection_mask_iou,
        )
        for name, value in (
            ("quality", quality),
            ("confidence", confidence),
            ("valid_depth_ratio", valid_depth_ratio),
            ("projection_mask_iou", projection_iou),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"memory view {name} must lie in [0, 1]")

        camera_position = self.camera_position
        if camera_position is not None:
            camera_position = _as_numpy(
                camera_position, "memory view camera_position"
            ).astype(np.float32, copy=False)
            if camera_position.shape != (3,):
                raise ValueError(
                    "memory view camera_position must have shape [3]"
                )
            if not np.isfinite(camera_position).all():
                raise ValueError(
                    "memory view camera_position must contain finite values"
                )
            camera_position = camera_position.copy()

        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(
            self, "points_world", points.astype(np.float32, copy=True)
        )
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "valid_depth_ratio", valid_depth_ratio)
        object.__setattr__(self, "projection_mask_iou", projection_iou)
        object.__setattr__(self, "camera_position", camera_position)

    @property
    def view_direction(self) -> Optional[np.ndarray]:
        """Unit camera-to-object direction, when a camera pose is available."""

        if self.camera_position is None or not len(self.points_world):
            return None
        object_center = np.median(self.points_world, axis=0)
        direction = object_center - self.camera_position
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            return None
        return (direction / norm).astype(np.float32)

    def copy(self) -> "MemoryViewRecord":
        return MemoryViewRecord(
            frame_id=self.frame_id,
            points_world=self.points_world,
            quality=self.quality,
            confidence=self.confidence,
            valid_depth_ratio=self.valid_depth_ratio,
            projection_mask_iou=self.projection_mask_iou,
            camera_position=self.camera_position,
        )


def _view_diversity(
    candidate: MemoryViewRecord,
    selected: Sequence[MemoryViewRecord],
) -> float:
    direction = candidate.view_direction
    if direction is None or not selected:
        return 0.0
    distances = []
    for record in selected:
        other = record.view_direction
        if other is None:
            continue
        cosine = float(np.clip(np.dot(direction, other), -1.0, 1.0))
        distances.append(0.5 * (1.0 - cosine))
    return float(min(distances)) if distances else 0.0


def select_diverse_view_records(
    records: Sequence[MemoryViewRecord],
    count: int,
    diversity_weight: float,
) -> Tuple[MemoryViewRecord, ...]:
    """Greedily select deterministic quality-and-view-diverse records."""

    count = _strict_int("view selection count", count, 1)
    diversity_weight = _finite_float(
        "view diversity weight", diversity_weight
    )
    if not 0.0 <= diversity_weight <= 1.0:
        raise ValueError("view diversity weight must lie in [0, 1]")
    remaining = sorted(
        records,
        key=lambda item: (
            -item.quality,
            -item.confidence,
            -item.valid_depth_ratio,
            item.frame_id,
        ),
    )
    selected: List[MemoryViewRecord] = []
    while remaining and len(selected) < count:
        if not selected:
            best = remaining.pop(0)
        else:
            best_index = min(
                range(len(remaining)),
                key=lambda index: (
                    -(
                        (1.0 - diversity_weight)
                        * remaining[index].quality
                        + diversity_weight
                        * _view_diversity(remaining[index], selected)
                    ),
                    -remaining[index].quality,
                    -remaining[index].confidence,
                    remaining[index].frame_id,
                ),
            )
            best = remaining.pop(best_index)
        selected.append(best)
    return tuple(selected)


class ObjectGeometryMemory:
    """Bounded per-instance world-point memory with aggregate quality stats."""

    _STAT_NAMES = (
        "confidence",
        "valid_depth_ratio",
        "projection_mask_iou",
        "quality",
        "input_points",
    )

    def __init__(
        self,
        track_id: Optional[int] = None,
        config: Optional[Mapping[str, object]] = None,
    ) -> None:
        self.config = resolve_object_memory_config(config)
        if track_id is not None:
            track_id = _strict_int("track_id", track_id, 0)
        self.track_id = track_id
        self._points = np.empty((0, 3), dtype=np.float32)
        self._view_candidates: List[MemoryViewRecord] = []
        self._geometry_points = np.empty((0, 3), dtype=np.float32)
        self.observation_count = 0
        self.unique_view_count = 0
        self.first_frame_id: Optional[int] = None
        self.last_frame_id: Optional[int] = None
        self.total_mask_pixels = 0
        self.total_valid_depth_pixels = 0
        self.total_input_points = 0
        self._stat_sums = {name: 0.0 for name in self._STAT_NAMES}
        self._stat_mins = {name: np.inf for name in self._STAT_NAMES}
        self._stat_maxs = {name: -np.inf for name in self._STAT_NAMES}

    @property
    def points(self) -> np.ndarray:
        return self._points.copy()

    @property
    def num_points(self) -> int:
        return int(self._points.shape[0])

    @property
    def top_k_enabled(self) -> bool:
        return int(self.config["top_k_views"]) > 0

    @property
    def view_candidate_count(self) -> int:
        return len(self._view_candidates)

    def _selected_view_records(self) -> Tuple[MemoryViewRecord, ...]:
        if not self.top_k_enabled or not self._view_candidates:
            return ()
        return select_diverse_view_records(
            self._view_candidates,
            min(
                int(self.config["top_k_views"]),
                len(self._view_candidates),
            ),
            float(self.config["view_diversity_weight"]),
        )

    @property
    def selected_view_records(self) -> Tuple[MemoryViewRecord, ...]:
        return tuple(record.copy() for record in self._selected_view_records())

    @property
    def selected_view_count(self) -> int:
        return len(self._selected_view_records())

    @property
    def selected_view_frame_ids(self) -> Tuple[int, ...]:
        return tuple(
            int(record.frame_id) for record in self._selected_view_records()
        )

    @property
    def geometry_points(self) -> np.ndarray:
        """Points used by B3 refit, without changing legacy B6 features."""

        if not self.top_k_enabled:
            return self.points
        return self._geometry_points.copy()

    @property
    def geometry_num_points(self) -> int:
        if not self.top_k_enabled:
            return self.num_points
        return int(self._geometry_points.shape[0])

    @property
    def geometry_unique_view_count(self) -> int:
        if not self.top_k_enabled:
            return self.unique_view_count
        return self.selected_view_count

    @property
    def aabb(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.num_points < int(self.config["min_points_for_aabb"]):
            return None
        return robust_quantile_aabb(
            self._points,
            lower_quantile=float(self.config["aabb_lower_quantile"]),
            upper_quantile=float(self.config["aabb_upper_quantile"]),
            min_points=int(self.config["min_points_for_aabb"]),
            minimum_dimension=float(
                self.config["minimum_aabb_dimension"]
            ),
        )

    @property
    def geometry_aabb(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        points = self.geometry_points
        if points.shape[0] < int(self.config["min_points_for_aabb"]):
            return None
        return robust_quantile_aabb(
            points,
            lower_quantile=float(self.config["aabb_lower_quantile"]),
            upper_quantile=float(self.config["aabb_upper_quantile"]),
            min_points=int(self.config["min_points_for_aabb"]),
            minimum_dimension=float(
                self.config["minimum_aabb_dimension"]
            ),
        )

    def _rebuild_geometry_points(self) -> None:
        if not self.top_k_enabled:
            self._geometry_points = np.empty((0, 3), dtype=np.float32)
            return
        selected = self._selected_view_records()
        if not selected:
            self._geometry_points = np.empty((0, 3), dtype=np.float32)
            return
        combined = np.concatenate(
            [record.points_world for record in selected], axis=0
        )
        combined = voxel_downsample(
            combined, float(self.config["voxel_size"])
        )
        self._geometry_points = deterministic_bounded_sample(
            combined, int(self.config["max_points_per_object"])
        )

    @staticmethod
    def _record_priority(record: MemoryViewRecord) -> Tuple[float, ...]:
        return (
            float(record.quality),
            float(record.confidence),
            float(record.valid_depth_ratio),
            float(record.projection_mask_iou),
            float(record.points_world.shape[0]),
            -float(record.frame_id),
        )

    def _set_view_candidates(
        self, records: Sequence[MemoryViewRecord]
    ) -> None:
        if not self.top_k_enabled:
            return
        best_by_frame: Dict[int, MemoryViewRecord] = {}
        for record in records:
            existing = best_by_frame.get(int(record.frame_id))
            if (
                existing is None
                or self._record_priority(record)
                > self._record_priority(existing)
            ):
                best_by_frame[int(record.frame_id)] = record
        candidates = list(best_by_frame.values())
        maximum = int(self.config["max_view_candidates"])
        if len(candidates) > maximum:
            candidates = list(
                select_diverse_view_records(
                    candidates,
                    maximum,
                    float(self.config["view_diversity_weight"]),
                )
            )
        self._view_candidates = sorted(
            candidates, key=lambda record: int(record.frame_id)
        )
        self._rebuild_geometry_points()

    def _record_view_candidate(
        self,
        observation: ObjectObservation,
        frame_id: int,
        points: np.ndarray,
    ) -> None:
        if (
            not self.top_k_enabled
            or not len(points)
            or observation.quality
            < float(self.config["minimum_view_quality"])
        ):
            return
        record = MemoryViewRecord(
            frame_id=frame_id,
            points_world=points,
            quality=float(np.clip(observation.quality, 0.0, 1.0)),
            confidence=observation.confidence,
            valid_depth_ratio=observation.valid_depth_ratio,
            projection_mask_iou=observation.projection_mask_iou,
            camera_position=observation.camera_position,
        )
        self._set_view_candidates([*self._view_candidates, record])

    def merge_view_candidates_from(
        self,
        other: "ObjectGeometryMemory",
        *,
        crop_center: Optional[object] = None,
        crop_dims: Optional[object] = None,
        minimum_points: Optional[int] = None,
    ) -> None:
        """Merge B3 per-view candidates without changing legacy statistics."""

        if not isinstance(other, ObjectGeometryMemory):
            raise TypeError("other must be an ObjectGeometryMemory")
        if not self.top_k_enabled:
            return
        if (crop_center is None) != (crop_dims is None):
            raise ValueError(
                "crop_center and crop_dims must be provided together"
            )
        crop = None
        if crop_center is not None:
            crop = _validated_aabb(crop_center, crop_dims)
        if minimum_points is None:
            minimum_points = int(self.config["min_points_for_aabb"])
        else:
            minimum_points = _strict_int(
                "minimum merge points", minimum_points, 1
            )
        incoming_records = []
        for source in other._view_candidates:
            points = source.points_world
            if crop is not None:
                inside = points_inside_aabb(points, crop[0], crop[1])
                points = points[inside]
                if len(points) < minimum_points:
                    continue
            incoming_records.append(
                MemoryViewRecord(
                    frame_id=source.frame_id,
                    points_world=points,
                    quality=source.quality,
                    confidence=source.confidence,
                    valid_depth_ratio=source.valid_depth_ratio,
                    projection_mask_iou=source.projection_mask_iou,
                    camera_position=source.camera_position,
                )
            )
        self._set_view_candidates(
            [
                *self._view_candidates,
                *incoming_records,
            ]
        )

    def _record_stat(self, name: str, value: float) -> None:
        if not np.isfinite(value):
            raise ValueError(f"quality statistic {name} must be finite")
        self._stat_sums[name] += float(value)
        self._stat_mins[name] = min(self._stat_mins[name], float(value))
        self._stat_maxs[name] = max(self._stat_maxs[name], float(value))

    def add_observation(
        self,
        observation: Union[ObjectObservation, DepthPointObservation, object],
        frame_id: int,
        *,
        confidence: float = 1.0,
        mask_pixels: Optional[int] = None,
        valid_depth_pixels: Optional[int] = None,
        projection_mask_iou: float = 1.0,
        camera_position: Optional[object] = None,
        record_view_candidate: bool = True,
    ) -> ObjectObservation:
        """Merge one observation while respecting both point-memory budgets."""

        frame_id = _strict_int("frame_id", frame_id, 0)
        if not isinstance(record_view_candidate, (bool, np.bool_)):
            raise ValueError("record_view_candidate must be a boolean")
        if self.last_frame_id is not None and frame_id < self.last_frame_id:
            raise ValueError("frame_id must be non-decreasing")

        if isinstance(observation, ObjectObservation):
            if camera_position is None:
                normalized = observation
            else:
                normalized = ObjectObservation(
                    points_world=observation.points_world,
                    confidence=observation.confidence,
                    mask_pixels=observation.mask_pixels,
                    valid_depth_pixels=observation.valid_depth_pixels,
                    projection_mask_iou=observation.projection_mask_iou,
                    camera_position=camera_position,
                )
        elif isinstance(observation, DepthPointObservation):
            normalized = ObjectObservation.from_depth_observation(
                observation,
                confidence=confidence,
                projection_mask_iou=projection_mask_iou,
                camera_position=camera_position,
            )
        else:
            normalized = ObjectObservation(
                points_world=observation,
                confidence=confidence,
                mask_pixels=mask_pixels,
                valid_depth_pixels=valid_depth_pixels,
                projection_mask_iou=projection_mask_iou,
                camera_position=camera_position,
            )

        incoming = deterministic_bounded_sample(
            normalized.points_world,
            int(self.config["max_points_per_observation"]),
        )
        if record_view_candidate:
            self._record_view_candidate(normalized, frame_id, incoming)
        if incoming.shape[0]:
            combined = np.concatenate((self._points, incoming), axis=0)
            combined = voxel_downsample(
                combined,
                float(self.config["voxel_size"]),
            )
            self._points = deterministic_bounded_sample(
                combined,
                int(self.config["max_points_per_object"]),
            )

        self.observation_count += 1
        if self.last_frame_id is None or frame_id > self.last_frame_id:
            self.unique_view_count += 1
        if self.first_frame_id is None:
            self.first_frame_id = frame_id
        self.last_frame_id = frame_id

        self.total_mask_pixels += int(normalized.mask_pixels)
        self.total_valid_depth_pixels += int(
            normalized.valid_depth_pixels
        )
        self.total_input_points += int(normalized.points_world.shape[0])
        values = {
            "confidence": normalized.confidence,
            "valid_depth_ratio": normalized.valid_depth_ratio,
            "projection_mask_iou": normalized.projection_mask_iou,
            "quality": normalized.quality,
            "input_points": float(normalized.points_world.shape[0]),
        }
        for name, value in values.items():
            self._record_stat(name, float(value))
        return normalized

    def add_depth_observation(
        self,
        depth: object,
        mask: object,
        intrinsics: object,
        camera_to_world: object,
        frame_id: int,
        confidence: float = 1.0,
    ) -> DepthPointObservation:
        """Extract and merge a Mask-RGBD observation in one call."""

        depth_observation = extract_masked_world_points(
            depth,
            mask,
            intrinsics,
            camera_to_world,
            self.config,
        )
        pose = _validated_camera_to_world(camera_to_world)
        projection_iou = 1.0
        current_aabb = self.aabb
        if current_aabb is not None:
            depth_shape = _as_2d_image(depth, "depth").shape
            resized_mask = resize_mask_nearest(
                mask,
                depth_shape,
                threshold=float(self.config["mask_threshold"]),
            )
            projection_iou = projected_aabb_mask_iou(
                current_aabb[0],
                current_aabb[1],
                intrinsics,
                camera_to_world,
                resized_mask,
                threshold=float(self.config["mask_threshold"]),
            )
        self.add_observation(
            depth_observation,
            frame_id,
            confidence=confidence,
            projection_mask_iou=projection_iou,
            camera_position=pose[:3, 3],
        )
        return depth_observation

    def quality_summary(self) -> Dict[str, Union[int, float, None]]:
        summary: Dict[str, Union[int, float, None]] = {
            "observations": self.observation_count,
            "unique_views": self.unique_view_count,
            "stored_points": self.num_points,
            "view_candidates": self.view_candidate_count,
            "selected_views": self.selected_view_count,
            "geometry_stored_points": self.geometry_num_points,
            "total_input_points": self.total_input_points,
            "total_mask_pixels": self.total_mask_pixels,
            "total_valid_depth_pixels": self.total_valid_depth_pixels,
            "aggregate_valid_depth_ratio": (
                float(self.total_valid_depth_pixels)
                / float(self.total_mask_pixels)
                if self.total_mask_pixels
                else 0.0
            ),
        }
        for name in self._STAT_NAMES:
            if self.observation_count == 0:
                summary[f"mean_{name}"] = None
                summary[f"min_{name}"] = None
                summary[f"max_{name}"] = None
            else:
                summary[f"mean_{name}"] = (
                    self._stat_sums[name] / self.observation_count
                )
                summary[f"min_{name}"] = self._stat_mins[name]
                summary[f"max_{name}"] = self._stat_maxs[name]
        return summary


@dataclass
class CandidateTrack:
    track_id: int
    memory: ObjectGeometryMemory
    created_frame: int
    last_frame: int
    created_lifecycle_step: Optional[int] = None
    last_lifecycle_step: Optional[int] = None
    hit_count: int = 0
    view_count: int = 0
    confirmed: bool = False

    def add(
        self,
        observation: ObjectObservation,
        frame_id: int,
        min_confirmations: int,
        *,
        lifecycle_step: Optional[int] = None,
    ) -> bool:
        if lifecycle_step is None:
            lifecycle_step = frame_id
        previous_frame = self.last_frame if self.hit_count else None
        self.memory.add_observation(observation, frame_id)
        self.hit_count += 1
        if previous_frame is None or frame_id > previous_frame:
            self.view_count += 1
        self.last_frame = frame_id
        self.last_lifecycle_step = lifecycle_step
        became_confirmed = (
            not self.confirmed
            and self.view_count >= min_confirmations
        )
        self.confirmed = (
            self.confirmed
            or self.view_count >= min_confirmations
        )
        return became_confirmed


@dataclass(frozen=True)
class TrackUpdateResult:
    assignments: Dict[int, int]
    created_track_ids: Tuple[int, ...]
    newly_confirmed_track_ids: Tuple[int, ...]
    expired_track_ids: Tuple[int, ...]
    skipped_observation_indices: Tuple[int, ...]
    archived_track_ids: Tuple[int, ...] = ()
    discarded_track_ids: Tuple[int, ...] = ()


class CandidateTrackManager:
    """Deterministic, geometry-only manager for online candidate objects."""

    def __init__(
        self,
        config: Optional[Mapping[str, object]] = None,
        *,
        archive_confirmed: bool = False,
    ) -> None:
        if not isinstance(archive_confirmed, (bool, np.bool_)):
            raise ValueError("archive_confirmed must be a boolean")
        self.config = resolve_object_memory_config(config)
        self.archive_confirmed = bool(archive_confirmed)
        self.tracks: Dict[int, CandidateTrack] = {}
        # Confirmed tracks are frozen here after leaving the active association
        # window.  They are never considered by ``update`` again, but callers
        # may retain them for final supplemental output.
        self.archived_tracks: Dict[int, CandidateTrack] = {}
        self.next_track_id = 0
        self.last_update_frame: Optional[int] = None
        self.last_lifecycle_step: Optional[int] = None

    def _coerce_observation(
        self,
        observation: Union[ObjectObservation, DepthPointObservation, object],
    ) -> ObjectObservation:
        if isinstance(observation, ObjectObservation):
            return observation
        if isinstance(observation, DepthPointObservation):
            return ObjectObservation.from_depth_observation(observation)
        return ObjectObservation(points_world=observation)

    def _observation_aabb(
        self,
        observation: ObjectObservation,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        minimum_points = int(self.config["min_points_for_aabb"])
        if observation.points_world.shape[0] < minimum_points:
            return None
        return robust_quantile_aabb(
            observation.points_world,
            lower_quantile=float(self.config["aabb_lower_quantile"]),
            upper_quantile=float(self.config["aabb_upper_quantile"]),
            min_points=minimum_points,
            minimum_dimension=float(
                self.config["minimum_aabb_dimension"]
            ),
        )

    def _association_metrics(
        self,
        track: CandidateTrack,
        observation: ObjectObservation,
        observation_box: Tuple[np.ndarray, np.ndarray],
    ) -> Optional[Dict[str, float]]:
        track_box = track.memory.aabb
        if track_box is None:
            return None
        track_center, track_dims = track_box
        observation_center, observation_dims = observation_box
        overlap = aabb_iou(
            track_center,
            track_dims,
            observation_center,
            observation_dims,
        )
        center_distance = float(
            np.linalg.norm(track_center - observation_center)
        )
        observation_inside = points_inside_aabb_fraction(
            observation.points_world,
            track_center,
            track_dims,
            tolerance=float(self.config["minimum_aabb_dimension"]),
        )
        memory_inside = points_inside_aabb_fraction(
            track.memory.points,
            observation_center,
            observation_dims,
            tolerance=float(self.config["minimum_aabb_dimension"]),
        )
        inside_fraction = max(observation_inside, memory_inside)
        maximum_distance = float(
            self.config["association_center_distance"]
        )
        valid = (
            (
                overlap > 0.0
                and overlap
                >= float(self.config["association_iou_threshold"])
            )
            or (
                center_distance <= maximum_distance
                and inside_fraction
                >= float(self.config["association_inside_fraction"])
            )
        )
        if not valid:
            return None
        center_quality = max(
            0.0,
            1.0 - center_distance / maximum_distance,
        )
        score = 2.0 * overlap + inside_fraction + 0.25 * center_quality
        return {
            "score": float(score),
            "iou": float(overlap),
            "inside_fraction": float(inside_fraction),
            "center_distance": center_distance,
        }

    def _expire(
        self, lifecycle_step: int
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        ttl = int(self.config["track_ttl"])
        expired = tuple(
            track_id
            for track_id, track in sorted(self.tracks.items())
            if lifecycle_step
            - (
                track.last_lifecycle_step
                if track.last_lifecycle_step is not None
                else track.last_frame
            )
            > ttl
        )
        archived = []
        discarded = []
        for track_id in expired:
            track = self.tracks.pop(track_id)
            if self.archive_confirmed and track.confirmed:
                self.archived_tracks[track_id] = track
                archived.append(track_id)
            else:
                discarded.append(track_id)
        return expired, tuple(archived), tuple(discarded)

    def _create_track(
        self,
        observation: ObjectObservation,
        frame_id: int,
        lifecycle_step: int,
    ) -> CandidateTrack:
        track_id = self.next_track_id
        self.next_track_id += 1
        memory = ObjectGeometryMemory(
            track_id=track_id,
            config=self.config,
        )
        track = CandidateTrack(
            track_id=track_id,
            memory=memory,
            created_frame=frame_id,
            last_frame=frame_id,
            created_lifecycle_step=lifecycle_step,
            last_lifecycle_step=lifecycle_step,
        )
        track.add(
            observation,
            frame_id,
            min_confirmations=int(self.config["min_confirmations"]),
            lifecycle_step=lifecycle_step,
        )
        self.tracks[track_id] = track
        return track

    def update(
        self,
        observations: Iterable[
            Union[ObjectObservation, DepthPointObservation, object]
        ],
        frame_id: int,
        *,
        lifecycle_step: Optional[int] = None,
        pair_compatibility: Optional[
            Mapping[Tuple[int, int], object]
        ] = None,
    ) -> TrackUpdateResult:
        """Associate one lifecycle step of observations.

        ``frame_id`` remains the real keyframe id stored in object memory.
        ``lifecycle_step`` selects the TTL clock; when omitted it defaults to
        ``frame_id`` for backward compatibility.

        When ``pair_compatibility`` is supplied, it acts as a strict whitelist
        over geometry-compatible ``(track_id, observation_index)`` pairs.
        Missing pairs cannot associate.  Each finite mapped value is added to
        the geometry association score before deterministic sorting.  A track
        already updated in ``frame_id`` cannot consume another observation
        from that same frame.  Passing ``None`` preserves the legacy matching
        and tie-breaking path exactly.
        """

        compatibility = _validated_pair_compatibility(pair_compatibility)
        frame_id = _strict_int("frame_id", frame_id, 0)
        if lifecycle_step is None:
            lifecycle_step = frame_id
        lifecycle_step = _strict_int(
            "lifecycle_step", lifecycle_step, 0
        )
        if (
            self.last_update_frame is not None
            and frame_id < self.last_update_frame
        ):
            raise ValueError("track-manager frame_id must be non-decreasing")
        if (
            self.last_lifecycle_step is not None
            and lifecycle_step < self.last_lifecycle_step
        ):
            raise ValueError(
                "track-manager lifecycle_step must be non-decreasing"
            )
        self.last_update_frame = frame_id
        self.last_lifecycle_step = lifecycle_step
        expired, archived, discarded = self._expire(lifecycle_step)

        normalized = [
            self._coerce_observation(observation)
            for observation in observations
        ]
        observation_boxes: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        skipped = []
        for index, observation in enumerate(normalized):
            observation_box = self._observation_aabb(observation)
            if observation_box is None:
                skipped.append(index)
            else:
                observation_boxes[index] = observation_box

        candidate_pairs = []
        for track_id, track in sorted(self.tracks.items()):
            if (
                compatibility is not None
                and track.hit_count
                and track.last_frame == frame_id
            ):
                # Graph compatibility represents one observation edge per
                # track and view.  Repeated provider calls for the same frame
                # must not add the same view to an existing track twice.
                continue
            for observation_index, observation_box in observation_boxes.items():
                metrics = self._association_metrics(
                    track,
                    normalized[observation_index],
                    observation_box,
                )
                if metrics is not None:
                    association_score = metrics["score"]
                    if compatibility is not None:
                        pair = (track_id, observation_index)
                        if pair not in compatibility:
                            continue
                        association_score += compatibility[pair]
                    candidate_pairs.append(
                        (
                            -association_score,
                            -metrics["iou"],
                            metrics["center_distance"],
                            track_id,
                            observation_index,
                        )
                    )
        candidate_pairs.sort()

        assignments: Dict[int, int] = {}
        used_tracks = set()
        newly_confirmed = []
        for _, _, _, track_id, observation_index in candidate_pairs:
            if track_id in used_tracks or observation_index in assignments:
                continue
            track = self.tracks[track_id]
            became_confirmed = track.add(
                normalized[observation_index],
                frame_id,
                min_confirmations=int(self.config["min_confirmations"]),
                lifecycle_step=lifecycle_step,
            )
            assignments[observation_index] = track_id
            used_tracks.add(track_id)
            if became_confirmed:
                newly_confirmed.append(track_id)

        created = []
        for observation_index in sorted(observation_boxes):
            if observation_index in assignments:
                continue
            track = self._create_track(
                normalized[observation_index],
                frame_id,
                lifecycle_step,
            )
            assignments[observation_index] = track.track_id
            created.append(track.track_id)
            if track.confirmed:
                newly_confirmed.append(track.track_id)

        return TrackUpdateResult(
            assignments=dict(sorted(assignments.items())),
            created_track_ids=tuple(created),
            newly_confirmed_track_ids=tuple(sorted(newly_confirmed)),
            expired_track_ids=expired,
            skipped_observation_indices=tuple(skipped),
            archived_track_ids=archived,
            discarded_track_ids=discarded,
        )

    def active_tracks(
        self,
        confirmed_only: bool = False,
    ) -> Tuple[CandidateTrack, ...]:
        if not isinstance(confirmed_only, (bool, np.bool_)):
            raise ValueError("confirmed_only must be a boolean")
        return tuple(
            track
            for _, track in sorted(self.tracks.items())
            if not confirmed_only or track.confirmed
        )

    def confirmed_tracks(
        self,
        include_archived: bool = False,
    ) -> Tuple[CandidateTrack, ...]:
        if not isinstance(include_archived, (bool, np.bool_)):
            raise ValueError("include_archived must be a boolean")
        active = self.active_tracks(confirmed_only=True)
        if not include_archived:
            return active
        combined = {
            track.track_id: track
            for track in active
        }
        combined.update(self.archived_tracks)
        return tuple(combined[key] for key in sorted(combined))


# Short aliases make the intended integration surface explicit while retaining
# descriptive class names in diagnostics and reprs.
ObjectMemory = ObjectGeometryMemory
TrackManager = CandidateTrackManager


__all__ = [
    "DEFAULT_OBJECT_MEMORY_CONFIG",
    "CandidateTrack",
    "CandidateTrackManager",
    "DepthPointObservation",
    "MemoryViewRecord",
    "ObjectGeometryMemory",
    "ObjectMemory",
    "ObjectObservation",
    "TrackUpdateResult",
    "TrackManager",
    "aabb_bounds",
    "aabb_corners",
    "aabb_iou",
    "bbox_mask_iou",
    "depth_discontinuity_mask",
    "deterministic_bounded_sample",
    "erode_mask_edges",
    "extract_masked_world_points",
    "points_inside_aabb",
    "points_inside_aabb_fraction",
    "project_aabb_to_image",
    "projected_aabb_mask_iou",
    "resize_mask_nearest",
    "resolve_object_memory_config",
    "robust_quantile_aabb",
    "select_diverse_view_records",
    "voxel_downsample",
]
