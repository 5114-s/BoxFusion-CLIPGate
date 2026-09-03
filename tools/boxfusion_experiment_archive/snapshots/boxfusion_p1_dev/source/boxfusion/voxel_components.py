"""Deterministic NumPy primitives for geometric voxel components.

The module intentionally stops below proposal generation.  It canonicalizes
points, voxelizes them against an explicit local origin, builds deterministic
Chebyshev-connected components, and exposes independent selectors for the
three ranking policies used by BoxFusion geometry refiners:

* largest: point count, then occupied-voxel count;
* densest: point density inside the raw point AABB;
* inside anchor: support inside an existing box, then multi-view support and
  distance to the box centre.

``neighbor_radius=1`` is exact 26-connectivity.  ``dilation_radius`` performs
a deterministic morphological dilation only for connectivity; returned voxel
keys and point indices always refer to the original occupied voxels.  All
arrays returned by this module are detached and read-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from numbers import Integral, Real
from typing import Dict, Optional, Tuple

import numpy as np


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


def _readonly(
    value: object,
    *,
    dtype: Optional[np.dtype] = None,
    shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError(f"array must have shape {shape}, got {array.shape}")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _points(value: object) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception as error:
        raise ValueError("points cannot be converted to an array") from error
    if (
        array.ndim != 2
        or array.shape[1:] != (3,)
        or not np.issubdtype(array.dtype, np.number)
    ):
        raise ValueError("points must have numeric shape [N, 3]")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("points must contain only finite values")
    return result


def _vector3(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.shape != (3,)
        or not np.issubdtype(array.dtype, np.number)
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must have finite numeric shape [3]")
    return np.asarray(array, dtype=np.float64)


def _point_view_ids(
    value: Optional[object], point_count: int
) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value)
    if (
        array.shape != (point_count,)
        or not np.issubdtype(array.dtype, np.integer)
    ):
        raise ValueError(
            "point_view_ids must have integer shape [N] aligned with points"
        )
    result = np.asarray(array, dtype=np.int64)
    if np.any(result < 0):
        raise ValueError("point_view_ids must be non-negative")
    return result


def _canonical_order(
    points: np.ndarray, point_view_ids: Optional[np.ndarray]
) -> np.ndarray:
    if len(points) == 0:
        return np.empty(0, dtype=np.int64)
    if point_view_ids is None:
        return np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    return np.lexsort(
        (
            point_view_ids,
            points[:, 2],
            points[:, 1],
            points[:, 0],
        )
    )


def _density(points: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    spans = np.max(points, axis=0) - np.min(points, axis=0)
    volume = float(np.prod(spans))
    if not np.isfinite(volume):
        raise ValueError("component AABB volume must be finite")
    if volume <= 0.0:
        return float("inf")
    return float(len(points)) / volume


def _key_tuple(key: np.ndarray) -> Tuple[int, int, int]:
    return tuple(int(value) for value in key)


def _neighbor_offsets(radius: int) -> Tuple[Tuple[int, int, int], ...]:
    return tuple(
        offset
        for offset in product(range(-radius, radius + 1), repeat=3)
        if offset != (0, 0, 0)
    )


def _component_labels(
    keys: np.ndarray, neighbor_radius: int
) -> np.ndarray:
    """Label lexicographically sorted unique keys deterministically."""

    if len(keys) == 0:
        return np.empty(0, dtype=np.int64)
    key_to_index = {
        _key_tuple(key): index for index, key in enumerate(keys)
    }
    offsets = _neighbor_offsets(neighbor_radius)
    labels = np.full(len(keys), -1, dtype=np.int64)
    next_label = 0
    for seed in range(len(keys)):
        if labels[seed] >= 0:
            continue
        labels[seed] = next_label
        queue = [seed]
        cursor = 0
        while cursor < len(queue):
            index = queue[cursor]
            cursor += 1
            key = keys[index]
            for offset in offsets:
                neighbor_key = (
                    int(key[0]) + offset[0],
                    int(key[1]) + offset[1],
                    int(key[2]) + offset[2],
                )
                neighbor = key_to_index.get(neighbor_key)
                if neighbor is None or labels[neighbor] >= 0:
                    continue
                labels[neighbor] = next_label
                queue.append(neighbor)
        next_label += 1
    return labels


def _voxel_component_labels(
    keys: np.ndarray,
    *,
    neighbor_radius: int,
    dilation_radius: int,
) -> np.ndarray:
    """Return component labels for original occupied voxel keys."""

    if len(keys) == 0:
        return np.empty(0, dtype=np.int64)
    if dilation_radius == 0:
        return _component_labels(keys, neighbor_radius)

    dilation_offsets = tuple(
        product(
            range(-dilation_radius, dilation_radius + 1),
            repeat=3,
        )
    )
    expanded = set()
    for key_array in keys:
        key = _key_tuple(key_array)
        for offset in dilation_offsets:
            expanded.add(
                (
                    key[0] + offset[0],
                    key[1] + offset[1],
                    key[2] + offset[2],
                )
            )
    expanded_keys = np.asarray(sorted(expanded), dtype=np.int64)
    expanded_labels = _component_labels(
        expanded_keys, neighbor_radius
    )
    expanded_key_to_label = {
        _key_tuple(key): int(label)
        for key, label in zip(expanded_keys, expanded_labels)
    }
    raw_labels = np.asarray(
        [expanded_key_to_label[_key_tuple(key)] for key in keys],
        dtype=np.int64,
    )

    # Reindex by the first original voxel key in each dilated component.  This
    # makes public component IDs independent of dilation-only cells.
    label_to_indices: Dict[int, list] = {}
    for voxel_index, label in enumerate(raw_labels):
        label_to_indices.setdefault(int(label), []).append(voxel_index)
    ordered_labels = sorted(
        label_to_indices,
        key=lambda label: _key_tuple(
            keys[min(label_to_indices[label])]
        ),
    )
    remap = {
        old_label: new_label
        for new_label, old_label in enumerate(ordered_labels)
    }
    return np.asarray(
        [remap[int(label)] for label in raw_labels], dtype=np.int64
    )


@dataclass(frozen=True)
class VoxelComponent:
    """One canonical, immutable connected component."""

    component_id: int
    point_indices: np.ndarray
    voxel_indices: np.ndarray
    points: np.ndarray
    point_view_ids: Optional[np.ndarray]
    voxel_keys: np.ndarray
    point_fraction: float
    view_count: int
    density: float
    stable_key: Tuple[int, int, int]

    def __post_init__(self) -> None:
        component_id = _strict_int(
            "component_id", self.component_id, 0
        )
        point_indices = np.asarray(self.point_indices)
        if (
            point_indices.ndim != 1
            or not np.issubdtype(point_indices.dtype, np.integer)
            or np.any(point_indices < 0)
        ):
            raise ValueError(
                "point_indices must be a non-negative integer vector"
            )
        voxel_indices = np.asarray(self.voxel_indices)
        if (
            voxel_indices.ndim != 1
            or not np.issubdtype(voxel_indices.dtype, np.integer)
            or np.any(voxel_indices < 0)
        ):
            raise ValueError(
                "voxel_indices must be a non-negative integer vector"
            )
        points = _points(self.points)
        if len(points) != len(point_indices):
            raise ValueError("points must align with point_indices")
        voxel_keys = np.asarray(self.voxel_keys)
        if (
            voxel_keys.shape != (len(voxel_indices), 3)
            or not np.issubdtype(voxel_keys.dtype, np.integer)
        ):
            raise ValueError(
                "voxel_keys must have integer shape [V, 3]"
            )
        view_ids = _point_view_ids(self.point_view_ids, len(points))
        view_count = _strict_int("view_count", self.view_count, 0)
        expected_views = (
            0 if view_ids is None else len(np.unique(view_ids))
        )
        if view_count != expected_views:
            raise ValueError("view_count disagrees with point_view_ids")
        fraction = _finite_float(
            "point_fraction", self.point_fraction
        )
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("point_fraction must lie in [0, 1]")
        density = float(self.density)
        if np.isnan(density) or density < 0.0:
            raise ValueError("density must be non-negative and not NaN")
        stable_key = tuple(int(value) for value in self.stable_key)
        if len(stable_key) != 3:
            raise ValueError("stable_key must contain three integers")
        if len(voxel_keys) == 0:
            raise ValueError("a component must contain at least one voxel")
        if stable_key != min(_key_tuple(key) for key in voxel_keys):
            raise ValueError(
                "stable_key must equal the lexicographically first voxel"
            )

        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(
            self,
            "point_indices",
            _readonly(point_indices, dtype=np.int64),
        )
        object.__setattr__(
            self,
            "voxel_indices",
            _readonly(voxel_indices, dtype=np.int64),
        )
        object.__setattr__(
            self, "points", _readonly(points, dtype=np.float64)
        )
        object.__setattr__(
            self,
            "point_view_ids",
            (
                None
                if view_ids is None
                else _readonly(view_ids, dtype=np.int64)
            ),
        )
        object.__setattr__(
            self,
            "voxel_keys",
            _readonly(voxel_keys, dtype=np.int64),
        )
        object.__setattr__(self, "point_fraction", fraction)
        object.__setattr__(self, "view_count", view_count)
        object.__setattr__(self, "density", density)
        object.__setattr__(self, "stable_key", stable_key)

    @property
    def point_count(self) -> int:
        return int(len(self.point_indices))

    @property
    def voxel_count(self) -> int:
        return int(len(self.voxel_indices))


@dataclass(frozen=True)
class VoxelComponentSet:
    """Canonical points, retained voxels, and their connected components."""

    points: np.ndarray
    point_view_ids: Optional[np.ndarray]
    voxel_keys: np.ndarray
    point_to_voxel: np.ndarray
    point_component_ids: np.ndarray
    components: Tuple[VoxelComponent, ...]
    origin: np.ndarray
    voxel_size: float
    boundary_epsilon: float
    neighbor_radius: int
    dilation_radius: int
    min_points_per_voxel: int

    def __post_init__(self) -> None:
        points = _points(self.points)
        view_ids = _point_view_ids(self.point_view_ids, len(points))
        voxel_keys = np.asarray(self.voxel_keys)
        if (
            voxel_keys.ndim != 2
            or voxel_keys.shape[1:] != (3,)
            or not np.issubdtype(voxel_keys.dtype, np.integer)
        ):
            raise ValueError("voxel_keys must have integer shape [V, 3]")
        point_to_voxel = np.asarray(self.point_to_voxel)
        point_component_ids = np.asarray(self.point_component_ids)
        for name, array in (
            ("point_to_voxel", point_to_voxel),
            ("point_component_ids", point_component_ids),
        ):
            if (
                array.shape != (len(points),)
                or not np.issubdtype(array.dtype, np.integer)
                or np.any(array < -1)
            ):
                raise ValueError(
                    f"{name} must have integer shape [N] with values >= -1"
                )
        if len(voxel_keys):
            if np.any(point_to_voxel >= len(voxel_keys)):
                raise ValueError("point_to_voxel index is out of range")
        elif np.any(point_to_voxel >= 0):
            raise ValueError("points cannot map to an empty voxel table")

        components = tuple(self.components)
        if any(
            component.component_id != index
            for index, component in enumerate(components)
        ):
            raise ValueError(
                "components must have contiguous ordered component IDs"
            )
        if len(components):
            if np.any(point_component_ids >= len(components)):
                raise ValueError(
                    "point_component_ids index is out of range"
                )
        elif np.any(point_component_ids >= 0):
            raise ValueError(
                "points cannot map to an empty component table"
            )

        origin = _vector3(self.origin, "origin")
        voxel_size = _finite_float("voxel_size", self.voxel_size)
        if voxel_size <= 0.0:
            raise ValueError("voxel_size must be positive")
        epsilon = _finite_float(
            "boundary_epsilon", self.boundary_epsilon
        )
        if not 0.0 <= epsilon < 0.5:
            raise ValueError("boundary_epsilon must lie in [0, 0.5)")
        neighbor_radius = _strict_int(
            "neighbor_radius", self.neighbor_radius, 1
        )
        dilation_radius = _strict_int(
            "dilation_radius", self.dilation_radius, 0
        )
        minimum = _strict_int(
            "min_points_per_voxel", self.min_points_per_voxel, 1
        )

        object.__setattr__(
            self, "points", _readonly(points, dtype=np.float64)
        )
        object.__setattr__(
            self,
            "point_view_ids",
            (
                None
                if view_ids is None
                else _readonly(view_ids, dtype=np.int64)
            ),
        )
        object.__setattr__(
            self,
            "voxel_keys",
            _readonly(voxel_keys, dtype=np.int64),
        )
        object.__setattr__(
            self,
            "point_to_voxel",
            _readonly(point_to_voxel, dtype=np.int64),
        )
        object.__setattr__(
            self,
            "point_component_ids",
            _readonly(point_component_ids, dtype=np.int64),
        )
        object.__setattr__(self, "components", components)
        object.__setattr__(
            self, "origin", _readonly(origin, dtype=np.float64)
        )
        object.__setattr__(self, "voxel_size", voxel_size)
        object.__setattr__(self, "boundary_epsilon", epsilon)
        object.__setattr__(self, "neighbor_radius", neighbor_radius)
        object.__setattr__(self, "dilation_radius", dilation_radius)
        object.__setattr__(self, "min_points_per_voxel", minimum)

    @property
    def component_count(self) -> int:
        return len(self.components)

    @property
    def retained_point_count(self) -> int:
        return int(np.sum(self.point_component_ids >= 0))


def build_voxel_components(
    points: object,
    *,
    origin: object,
    voxel_size: object,
    boundary_epsilon: object,
    neighbor_radius: object,
    dilation_radius: object,
    point_view_ids: Optional[object] = None,
    min_points_per_voxel: object = 1,
) -> VoxelComponentSet:
    """Build deterministic components from metric points.

    Args:
        points: Numeric ``[N, 3]`` points.  Empty input is valid.
        origin: Explicit metric voxel-grid origin ``[3]``.
        voxel_size: Positive metric edge length.
        boundary_epsilon: Non-negative *dimensionless* value added before
            ``floor``.  It must be below ``0.5`` and is intended only to
            stabilize floating-point values on voxel boundaries.
        neighbor_radius: Chebyshev graph radius.  ``1`` is exact
            26-connectivity.
        dilation_radius: Morphological dilation radius in voxel cells used
            only to determine connectivity.  ``0`` disables dilation.
        point_view_ids: Optional non-negative integer view index per point.
        min_points_per_voxel: Occupied voxels below this count are discarded;
            their canonical points receive ``-1`` mappings.

    Component point indices refer to ``result.points``, which is sorted
    lexicographically by XYZ and then by view ID.  They deliberately do not
    refer to the incoming point order.
    """

    point_array = _points(points)
    view_ids = _point_view_ids(point_view_ids, len(point_array))
    grid_origin = _vector3(origin, "origin")
    size = _finite_float("voxel_size", voxel_size)
    if size <= 0.0:
        raise ValueError("voxel_size must be positive")
    epsilon = _finite_float("boundary_epsilon", boundary_epsilon)
    if not 0.0 <= epsilon < 0.5:
        raise ValueError("boundary_epsilon must lie in [0, 0.5)")
    radius = _strict_int("neighbor_radius", neighbor_radius, 1)
    dilation = _strict_int("dilation_radius", dilation_radius, 0)
    minimum = _strict_int(
        "min_points_per_voxel", min_points_per_voxel, 1
    )

    order = _canonical_order(point_array, view_ids)
    canonical_points = np.asarray(point_array[order], dtype=np.float64)
    canonical_views = (
        None
        if view_ids is None
        else np.asarray(view_ids[order], dtype=np.int64)
    )
    if len(canonical_points) == 0:
        return VoxelComponentSet(
            points=np.empty((0, 3), dtype=np.float64),
            point_view_ids=canonical_views,
            voxel_keys=np.empty((0, 3), dtype=np.int64),
            point_to_voxel=np.empty(0, dtype=np.int64),
            point_component_ids=np.empty(0, dtype=np.int64),
            components=(),
            origin=grid_origin,
            voxel_size=size,
            boundary_epsilon=epsilon,
            neighbor_radius=radius,
            dilation_radius=dilation,
            min_points_per_voxel=minimum,
        )

    scaled = (canonical_points - grid_origin[None, :]) / size
    if (
        np.max(np.abs(scaled), initial=0.0)
        > np.iinfo(np.int64).max / 4
    ):
        raise ValueError("points are too large for the requested voxel grid")
    raw_point_keys = np.floor(scaled + epsilon).astype(np.int64)
    all_keys, all_inverse, all_counts = np.unique(
        raw_point_keys,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    retained = all_counts >= minimum
    voxel_keys = np.asarray(all_keys[retained], dtype=np.int64)
    old_to_new = np.full(len(all_keys), -1, dtype=np.int64)
    old_to_new[np.flatnonzero(retained)] = np.arange(
        len(voxel_keys), dtype=np.int64
    )
    point_to_voxel = old_to_new[all_inverse]
    voxel_component_ids = _voxel_component_labels(
        voxel_keys,
        neighbor_radius=radius,
        dilation_radius=dilation,
    )
    point_component_ids = np.full(
        len(canonical_points), -1, dtype=np.int64
    )
    retained_points = point_to_voxel >= 0
    if np.any(retained_points):
        point_component_ids[retained_points] = voxel_component_ids[
            point_to_voxel[retained_points]
        ]

    components = []
    component_count = (
        int(np.max(voxel_component_ids)) + 1
        if len(voxel_component_ids)
        else 0
    )
    for component_id in range(component_count):
        voxel_indices = np.flatnonzero(
            voxel_component_ids == component_id
        ).astype(np.int64)
        point_indices = np.flatnonzero(
            point_component_ids == component_id
        ).astype(np.int64)
        component_points = canonical_points[point_indices]
        component_views = (
            None
            if canonical_views is None
            else canonical_views[point_indices]
        )
        component_keys = voxel_keys[voxel_indices]
        components.append(
            VoxelComponent(
                component_id=component_id,
                point_indices=point_indices,
                voxel_indices=voxel_indices,
                points=component_points,
                point_view_ids=component_views,
                voxel_keys=component_keys,
                point_fraction=(
                    len(point_indices) / float(len(canonical_points))
                ),
                view_count=(
                    0
                    if component_views is None
                    else len(np.unique(component_views))
                ),
                density=_density(component_points),
                stable_key=min(
                    _key_tuple(key) for key in component_keys
                ),
            )
        )

    return VoxelComponentSet(
        points=canonical_points,
        point_view_ids=canonical_views,
        voxel_keys=voxel_keys,
        point_to_voxel=point_to_voxel,
        point_component_ids=point_component_ids,
        components=tuple(components),
        origin=grid_origin,
        voxel_size=size,
        boundary_epsilon=epsilon,
        neighbor_radius=radius,
        dilation_radius=dilation,
        min_points_per_voxel=minimum,
    )


def _eligible_components(
    component_set: VoxelComponentSet,
    *,
    min_points: object,
    min_voxels: object,
    min_views: object,
) -> Tuple[VoxelComponent, ...]:
    if not isinstance(component_set, VoxelComponentSet):
        raise ValueError("component_set must be a VoxelComponentSet")
    points = _strict_int("min_points", min_points, 1)
    voxels = _strict_int("min_voxels", min_voxels, 1)
    views = _strict_int("min_views", min_views, 0)
    return tuple(
        component
        for component in component_set.components
        if component.point_count >= points
        and component.voxel_count >= voxels
        and component.view_count >= views
    )


def select_largest_component(
    component_set: VoxelComponentSet,
    *,
    min_points: object = 1,
    min_voxels: object = 1,
    min_views: object = 0,
) -> Optional[VoxelComponent]:
    """Select by points, voxels, then the stable lexicographic key."""

    eligible = _eligible_components(
        component_set,
        min_points=min_points,
        min_voxels=min_voxels,
        min_views=min_views,
    )
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda component: (
            -component.point_count,
            -component.voxel_count,
            component.stable_key,
        ),
    )


def select_densest_component(
    component_set: VoxelComponentSet,
    *,
    min_points: object = 1,
    min_voxels: object = 1,
    min_views: object = 0,
) -> Optional[VoxelComponent]:
    """Select by raw point-AABB density with deterministic tie-breaks."""

    eligible = _eligible_components(
        component_set,
        min_points=min_points,
        min_voxels=min_voxels,
        min_views=min_views,
    )
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda component: (
            -component.density,
            -component.point_count,
            -component.voxel_count,
            component.stable_key,
        ),
    )


def select_inside_anchor(
    component_set: VoxelComponentSet,
    *,
    lower: object,
    upper: object,
    min_points: object = 1,
    min_voxels: object = 1,
    min_views: object = 0,
    min_inside_points: object = 1,
    min_inside_fraction: object = 0.0,
    normalization_dimensions: Optional[object] = None,
) -> Optional[VoxelComponent]:
    """Select the component most strongly anchored inside an existing box.

    Eligible components are ranked by descending inside-point count,
    descending distinct-view count, ascending normalized median-centre
    distance, descending total points, descending occupied voxels, and the
    stable lexicographic voxel key.
    """

    box_lower = _vector3(lower, "lower")
    box_upper = _vector3(upper, "upper")
    if np.any(box_upper <= box_lower):
        raise ValueError("upper must be strictly greater than lower")
    inside_points = _strict_int(
        "min_inside_points", min_inside_points, 0
    )
    inside_fraction = _finite_float(
        "min_inside_fraction", min_inside_fraction
    )
    if not 0.0 <= inside_fraction <= 1.0:
        raise ValueError("min_inside_fraction must lie in [0, 1]")
    dimensions = (
        box_upper - box_lower
        if normalization_dimensions is None
        else _vector3(
            normalization_dimensions, "normalization_dimensions"
        )
    )
    if np.any(dimensions <= 0.0):
        raise ValueError("normalization_dimensions must be positive")
    box_center = 0.5 * (box_lower + box_upper)

    eligible = _eligible_components(
        component_set,
        min_points=min_points,
        min_voxels=min_voxels,
        min_views=min_views,
    )
    ranked = []
    for component in eligible:
        inside = np.logical_and(
            component.points >= box_lower[None, :],
            component.points <= box_upper[None, :],
        ).all(axis=1)
        count = int(np.sum(inside))
        fraction = count / float(component.point_count)
        if count < inside_points or fraction < inside_fraction:
            continue
        normalized_center = (
            np.median(component.points, axis=0) - box_center
        ) / dimensions
        center_distance = float(np.linalg.norm(normalized_center))
        ranked.append(
            (
                (
                    -count,
                    -component.view_count,
                    center_distance,
                    -component.point_count,
                    -component.voxel_count,
                    component.stable_key,
                ),
                component,
            )
        )
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0])
    return ranked[0][1]


__all__ = [
    "VoxelComponent",
    "VoxelComponentSet",
    "build_voxel_components",
    "select_densest_component",
    "select_inside_anchor",
    "select_largest_component",
]
