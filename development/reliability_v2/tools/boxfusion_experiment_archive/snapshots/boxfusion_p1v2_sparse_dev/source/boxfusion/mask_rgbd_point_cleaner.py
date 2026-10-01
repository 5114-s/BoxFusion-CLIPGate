"""Deterministic Mask-RGBD point cleaning for isolated BoxFusion ablations.

The functions in this module are NumPy-only and deliberately contain no
network, checkpoint, file-system, or controller dependency.  They implement
the two geometry-cleaning ideas used by the YiDu route:

* an object-scale and depth-boundary aware mask erosion margin; and
* DFU3D-inspired local-radius and global-statistical point filtering.

The filters are fail-open.  If a stage would leave fewer than
``minimum_points`` points, the input to that stage is retained and the
fallback is recorded.  This matters for a diagnostics-only ablation: a noisy
mask may produce an unhelpful candidate, but cleaning must never silently
erase the evidence stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from numbers import Integral, Real
from typing import Dict, Mapping, Optional, Tuple

import numpy as np


DEFAULT_MASK_RGBD_POINT_CLEANER_CONFIG = {
    "radius_filter_enabled": False,
    "radius": 0.05,
    "minimum_neighbors": 3,
    "statistical_filter_enabled": False,
    "statistical_k": 8,
    "statistical_std_ratio": 2.0,
    "maximum_input_points": 4096,
    "minimum_points": 16,
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


def resolve_mask_rgbd_point_cleaner_config(
    config: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Return a detached and strictly validated point-cleaner config."""

    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("Mask-RGBD point-cleaner config must be a mapping")
    unknown = sorted(
        set(config) - set(DEFAULT_MASK_RGBD_POINT_CLEANER_CONFIG)
    )
    if unknown:
        raise ValueError(
            "Unknown Mask-RGBD point-cleaner key(s): " + ", ".join(unknown)
        )
    resolved = dict(DEFAULT_MASK_RGBD_POINT_CLEANER_CONFIG)
    resolved.update(config)
    for key in ("radius_filter_enabled", "statistical_filter_enabled"):
        if not isinstance(resolved[key], (bool, np.bool_)):
            raise ValueError(f"point_cleaner.{key} must be Boolean")
        resolved[key] = bool(resolved[key])
    resolved["radius"] = _finite_float(
        "point_cleaner.radius", resolved["radius"]
    )
    resolved["statistical_std_ratio"] = _finite_float(
        "point_cleaner.statistical_std_ratio",
        resolved["statistical_std_ratio"],
    )
    if float(resolved["radius"]) <= 0.0:
        raise ValueError("point_cleaner.radius must be positive")
    if float(resolved["statistical_std_ratio"]) < 0.0:
        raise ValueError(
            "point_cleaner.statistical_std_ratio must be non-negative"
        )
    for key, minimum in (
        ("minimum_neighbors", 1),
        ("statistical_k", 1),
        ("maximum_input_points", 1),
        ("minimum_points", 1),
    ):
        resolved[key] = _strict_int(
            f"point_cleaner.{key}", resolved[key], minimum
        )
    if int(resolved["minimum_points"]) > int(
        resolved["maximum_input_points"]
    ):
        raise ValueError(
            "point_cleaner.minimum_points exceeds maximum_input_points"
        )
    return resolved


def _validated_points(points: object) -> np.ndarray:
    array = np.asarray(points)
    if (
        array.ndim != 2
        or array.shape[1:] != (3,)
        or not np.issubdtype(array.dtype, np.number)
        or not np.isfinite(array).all()
    ):
        raise ValueError("points must have finite numeric shape [N, 3]")
    return np.asarray(array, dtype=np.float64)


def _immutable(value: object, dtype: np.dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _bounded_spatial_sample(points: np.ndarray, limit: int) -> np.ndarray:
    if len(points) <= limit:
        return np.asarray(points, dtype=np.float64).copy()
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    sorted_points = np.asarray(points[order], dtype=np.float64)
    positions = np.linspace(
        0, len(sorted_points) - 1, limit, dtype=np.int64
    )
    return sorted_points[positions]


def adaptive_erosion_margin(
    mask: object,
    depth_meters: Optional[object] = None,
    *,
    minimum_margin: int = 0,
    maximum_margin: int = 3,
    radius_fraction: float = 0.02,
    depth_edge_weight: float = 4.0,
    depth_edge_threshold: float = 0.15,
) -> int:
    """Choose a deterministic erosion margin from object scale and bad edges.

    ``radius_fraction`` is applied to the equivalent circular mask radius.
    The optional depth term increases the margin only when discontinuities
    occur on the *interior mask boundary*.  The returned value is clamped to
    ``[minimum_margin, maximum_margin]``.
    """

    binary = np.asarray(mask)
    if binary.ndim != 2 or min(binary.shape) <= 0:
        raise ValueError("mask must have shape [H, W]")
    if np.issubdtype(binary.dtype, np.number):
        if not np.isfinite(binary).all():
            raise ValueError("mask must be finite")
        binary = binary >= 0.5
    elif not np.issubdtype(binary.dtype, np.bool_):
        raise ValueError("mask must be boolean or numeric")
    binary = np.asarray(binary, dtype=bool)

    minimum = _strict_int("minimum_margin", minimum_margin, 0)
    maximum = _strict_int("maximum_margin", maximum_margin, 0)
    if maximum < minimum:
        raise ValueError("maximum_margin must be at least minimum_margin")
    fraction = _finite_float("radius_fraction", radius_fraction)
    edge_weight = _finite_float("depth_edge_weight", depth_edge_weight)
    edge_threshold = _finite_float(
        "depth_edge_threshold", depth_edge_threshold
    )
    if fraction < 0.0 or edge_weight < 0.0 or edge_threshold <= 0.0:
        raise ValueError(
            "erosion fractions/weights must be non-negative and the depth "
            "threshold must be positive"
        )
    area = int(np.count_nonzero(binary))
    if area == 0:
        return minimum
    equivalent_radius = float(np.sqrt(area / np.pi))
    scale_margin = int(np.rint(equivalent_radius * fraction))

    edge_fraction = 0.0
    if depth_meters is not None:
        depth = np.asarray(depth_meters, dtype=np.float64)
        if depth.shape != binary.shape:
            raise ValueError("depth_meters must have the same shape as mask")
        finite = np.isfinite(depth) & (depth > 0.0)
        edge = np.zeros_like(binary, dtype=bool)
        horizontal = (
            finite[:, :-1]
            & finite[:, 1:]
            & (np.abs(depth[:, :-1] - depth[:, 1:]) > edge_threshold)
        )
        vertical = (
            finite[:-1, :]
            & finite[1:, :]
            & (np.abs(depth[:-1, :] - depth[1:, :]) > edge_threshold)
        )
        edge[:, :-1] |= horizontal
        edge[:, 1:] |= horizontal
        edge[:-1, :] |= vertical
        edge[1:, :] |= vertical

        padded = np.pad(binary, 1, mode="constant", constant_values=False)
        interior_boundary = np.zeros_like(binary, dtype=bool)
        height, width = binary.shape
        for row_offset, col_offset in (
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 1),
        ):
            interior_boundary |= binary & ~padded[
                row_offset : row_offset + height,
                col_offset : col_offset + width,
            ]
        boundary_count = int(np.count_nonzero(interior_boundary))
        if boundary_count:
            edge_fraction = float(
                np.count_nonzero(edge & interior_boundary)
            ) / float(boundary_count)
    edge_margin = int(np.rint(edge_fraction * edge_weight))
    return int(np.clip(scale_margin + edge_margin, minimum, maximum))


def radius_neighbor_mask(
    points: object,
    *,
    radius: float,
    minimum_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return local-radius inlier mask and neighbour counts.

    Counts include the point itself.  A voxel hash limits each distance query
    to the 27 cells that can intersect the requested radius.
    """

    array = _validated_points(points)
    radius_value = _finite_float("radius", radius)
    minimum = _strict_int("minimum_neighbors", minimum_neighbors, 1)
    if radius_value <= 0.0:
        raise ValueError("radius must be positive")
    if len(array) == 0:
        return (
            np.zeros(0, dtype=bool),
            np.zeros(0, dtype=np.int64),
        )
    keys = np.floor(array / radius_value).astype(np.int64)
    buckets: Dict[Tuple[int, int, int], list] = {}
    for index, raw_key in enumerate(keys):
        key = tuple(int(value) for value in raw_key)
        buckets.setdefault(key, []).append(index)
    squared_radius = radius_value * radius_value
    counts = np.zeros(len(array), dtype=np.int64)
    offsets = tuple(product((-1, 0, 1), repeat=3))
    for index, raw_key in enumerate(keys):
        key = tuple(int(value) for value in raw_key)
        candidate_indices = []
        for offset in offsets:
            candidate_indices.extend(
                buckets.get(
                    (
                        key[0] + offset[0],
                        key[1] + offset[1],
                        key[2] + offset[2],
                    ),
                    (),
                )
            )
        candidates = array[np.asarray(candidate_indices, dtype=np.int64)]
        squared = np.sum((candidates - array[index]) ** 2, axis=1)
        counts[index] = int(np.count_nonzero(squared <= squared_radius))
    return counts >= minimum, counts


def statistical_inlier_mask(
    points: object,
    *,
    k: int,
    std_ratio: float,
    chunk_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return a global kNN-distance statistical inlier mask.

    Pairwise distances are evaluated in bounded chunks.  This is deterministic
    and keeps peak memory small for the route's default 4096-point cap.
    """

    array = _validated_points(points)
    neighbours = _strict_int("k", k, 1)
    ratio = _finite_float("std_ratio", std_ratio)
    chunk = _strict_int("chunk_size", chunk_size, 1)
    if ratio < 0.0:
        raise ValueError("std_ratio must be non-negative")
    if len(array) <= 1:
        means = np.zeros(len(array), dtype=np.float64)
        return np.ones(len(array), dtype=bool), means, 0.0
    effective_k = min(neighbours, len(array) - 1)
    mean_distances = np.empty(len(array), dtype=np.float64)
    for start in range(0, len(array), chunk):
        stop = min(start + chunk, len(array))
        squared = np.sum(
            (array[start:stop, None, :] - array[None, :, :]) ** 2,
            axis=2,
        )
        row_indices = np.arange(stop - start)
        squared[row_indices, np.arange(start, stop)] = np.inf
        nearest = np.partition(
            squared, kth=effective_k - 1, axis=1
        )[:, :effective_k]
        mean_distances[start:stop] = np.mean(np.sqrt(nearest), axis=1)
    threshold = float(
        np.mean(mean_distances) + ratio * np.std(mean_distances)
    )
    return mean_distances <= threshold, mean_distances, threshold


@dataclass(frozen=True)
class PointCleaningResult:
    """Immutable filtered points and audit statistics."""

    points: np.ndarray
    input_count: int
    bounded_count: int
    radius_retained_count: int
    statistical_retained_count: int
    radius_applied: bool
    statistical_applied: bool
    radius_fallback: bool
    statistical_fallback: bool
    statistical_threshold: float
    neighbor_count_quantiles: np.ndarray
    mean_distance_quantiles: np.ndarray

    def __post_init__(self) -> None:
        points = _validated_points(self.points)
        object.__setattr__(self, "points", _immutable(points, np.float32))
        for name in (
            "input_count",
            "bounded_count",
            "radius_retained_count",
            "statistical_retained_count",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or int(value) != value:
                raise ValueError(f"{name} must be an integer")
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, int(value))
        for name in (
            "radius_applied",
            "statistical_applied",
            "radius_fallback",
            "statistical_fallback",
        ):
            object.__setattr__(self, name, bool(getattr(self, name)))
        threshold = float(self.statistical_threshold)
        if not (np.isfinite(threshold) or np.isnan(threshold)):
            raise ValueError("statistical_threshold must be finite or NaN")
        object.__setattr__(self, "statistical_threshold", threshold)
        object.__setattr__(
            self,
            "neighbor_count_quantiles",
            _immutable(self.neighbor_count_quantiles, np.float32),
        )
        object.__setattr__(
            self,
            "mean_distance_quantiles",
            _immutable(self.mean_distance_quantiles, np.float32),
        )


def clean_mask_rgbd_points(
    points: object,
    config: Optional[Mapping[str, object]] = None,
) -> PointCleaningResult:
    """Apply bounded local and global filtering with fail-open stages."""

    resolved = resolve_mask_rgbd_point_cleaner_config(config)
    original = _validated_points(points)
    bounded = _bounded_spatial_sample(
        original, int(resolved["maximum_input_points"])
    )
    current = bounded
    minimum_points = int(resolved["minimum_points"])

    radius_counts = np.zeros(len(current), dtype=np.int64)
    radius_retained = len(current)
    radius_fallback = False
    if bool(resolved["radius_filter_enabled"]) and len(current):
        radius_mask, radius_counts = radius_neighbor_mask(
            current,
            radius=float(resolved["radius"]),
            minimum_neighbors=int(resolved["minimum_neighbors"]),
        )
        radius_retained = int(np.count_nonzero(radius_mask))
        if radius_retained >= minimum_points:
            current = current[radius_mask]
        else:
            radius_fallback = True

    mean_distances = np.zeros(len(current), dtype=np.float64)
    statistical_threshold = np.nan
    statistical_retained = len(current)
    statistical_fallback = False
    if bool(resolved["statistical_filter_enabled"]) and len(current):
        statistical_mask, mean_distances, statistical_threshold = (
            statistical_inlier_mask(
                current,
                k=int(resolved["statistical_k"]),
                std_ratio=float(resolved["statistical_std_ratio"]),
            )
        )
        statistical_retained = int(np.count_nonzero(statistical_mask))
        if statistical_retained >= minimum_points:
            current = current[statistical_mask]
        else:
            statistical_fallback = True

    def quantiles(values: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return np.full(3, np.nan, dtype=np.float32)
        return np.quantile(values, (0.10, 0.50, 0.90)).astype(np.float32)

    return PointCleaningResult(
        points=current,
        input_count=len(original),
        bounded_count=len(bounded),
        radius_retained_count=radius_retained,
        statistical_retained_count=statistical_retained,
        radius_applied=bool(resolved["radius_filter_enabled"]),
        statistical_applied=bool(
            resolved["statistical_filter_enabled"]
        ),
        radius_fallback=radius_fallback,
        statistical_fallback=statistical_fallback,
        statistical_threshold=statistical_threshold,
        neighbor_count_quantiles=quantiles(radius_counts),
        mean_distance_quantiles=quantiles(mean_distances),
    )


__all__ = [
    "DEFAULT_MASK_RGBD_POINT_CLEANER_CONFIG",
    "PointCleaningResult",
    "adaptive_erosion_margin",
    "clean_mask_rgbd_points",
    "radius_neighbor_mask",
    "resolve_mask_rgbd_point_cleaner_config",
    "statistical_inlier_mask",
]
