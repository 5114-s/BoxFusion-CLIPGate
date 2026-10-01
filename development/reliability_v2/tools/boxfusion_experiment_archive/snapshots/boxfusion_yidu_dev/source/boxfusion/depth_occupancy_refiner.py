"""Deterministic NumPy geometry proposals from Mask-RGBD point memory.

This module deliberately contains no model, file-system, or ground-truth
dependency.  It only proposes geometry; the online controller remains
responsible for deciding whether a proposal is safe to export.

Two shape-specific branches are provided:

* compact/solid tracks use all available memory points and their raw envelope,
  retaining observations that may cover different object surfaces;
* planar tracks suppress sparse voxels, form 26-connected components, and
  select the sufficiently large component with the greatest point density.

The standalone :func:`largest_voxel_connected_component` helper implements the
more conservative point-count-first component rule.  It is intentionally kept
separate from the active planar proposal branch so both algorithms can be
tested and ablated independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from numbers import Integral, Real
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_DEPTH_OCCUPANCY_REFINER_CONFIG = {
    # Evidence requirements.
    "min_views": 3,
    "min_points": 192,
    # Standalone conservative largest-component settings.
    "voxel_size": 0.02,
    "neighbor_radius": 2,
    "min_component_fraction": 0.90,
    # Shape classification.  The ratio is the thinnest robust extent divided
    # by the middle robust extent, so rod-like tracks are not called planar.
    "planar_ratio_threshold": 0.10,
    # Oracle-selected planar occupancy settings: 4 cm voxels, at least five
    # points per occupied voxel, 26 connectivity, and a permissive component
    # fraction followed by a density-first choice.
    "planar_voxel_size": 0.04,
    "planar_min_points_per_voxel": 5,
    "planar_min_component_fraction": 0.10,
    "planar_thin_axis_minimum": 0.034,
    # Only guards numerically degenerate envelopes.  The planar thin-axis
    # clamp above is the meaningful geometric minimum.
    "minimum_dimension": 1e-4,
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


def resolve_depth_occupancy_refiner_config(
    config: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Return a detached, strictly validated refiner configuration.

    Unknown keys are rejected.  This prevents a misspelled experimental knob
    from silently changing an ablation into the default algorithm.
    """

    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("depth occupancy refiner config must be a mapping")

    unknown = sorted(
        set(config) - set(DEFAULT_DEPTH_OCCUPANCY_REFINER_CONFIG)
    )
    if unknown:
        raise ValueError(
            "Unknown depth occupancy refiner config key(s): "
            + ", ".join(unknown)
        )

    resolved = dict(DEFAULT_DEPTH_OCCUPANCY_REFINER_CONFIG)
    resolved.update(config)

    for key, minimum in (
        ("min_views", 1),
        ("min_points", 1),
        ("neighbor_radius", 1),
        ("planar_min_points_per_voxel", 1),
    ):
        resolved[key] = _strict_int(
            f"depth_occupancy_refiner.{key}",
            resolved[key],
            minimum,
        )

    for key in (
        "voxel_size",
        "min_component_fraction",
        "planar_ratio_threshold",
        "planar_voxel_size",
        "planar_min_component_fraction",
        "planar_thin_axis_minimum",
        "minimum_dimension",
    ):
        resolved[key] = _finite_float(
            f"depth_occupancy_refiner.{key}", resolved[key]
        )

    for key in (
        "voxel_size",
        "planar_voxel_size",
        "planar_thin_axis_minimum",
        "minimum_dimension",
    ):
        if float(resolved[key]) <= 0.0:
            raise ValueError(
                f"depth_occupancy_refiner.{key} must be positive"
            )

    for key in (
        "min_component_fraction",
        "planar_ratio_threshold",
        "planar_min_component_fraction",
    ):
        if not 0.0 <= float(resolved[key]) <= 1.0:
            raise ValueError(
                f"depth_occupancy_refiner.{key} must lie in [0, 1]"
            )

    return resolved


def _readonly_array(
    value: object,
    *,
    name: str,
    shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(array, dtype=np.float32, copy=True)
    result.setflags(write=False)
    return result


def _validated_points(
    points: object,
    *,
    name: str = "points",
    allow_empty: bool = False,
) -> np.ndarray:
    try:
        array = np.asarray(points)
    except Exception as error:
        raise ValueError(f"{name} cannot be converted to an array") from error
    if (
        array.ndim != 2
        or array.shape[1:] != (3,)
        or not np.issubdtype(array.dtype, np.number)
    ):
        raise ValueError(f"{name} must have numeric shape [N, 3]")
    array = np.asarray(array, dtype=np.float64)
    if not allow_empty and len(array) == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validated_box(box: object) -> np.ndarray:
    array = np.asarray(box, dtype=np.float64)
    if array.shape != (6,) or not np.isfinite(array).all():
        raise ValueError("original_box must have finite shape [6]")
    if np.any(array[3:6] <= 0.0):
        raise ValueError("original_box dimensions must be positive")
    return array.astype(np.float32)


def _sorted_points(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    return np.asarray(points[order], dtype=np.float32)


@dataclass(frozen=True)
class VoxelComponent:
    """One immutable deterministic voxel component."""

    points: np.ndarray
    point_fraction: float
    voxel_count: int
    density: float
    stable_index: int

    def __post_init__(self) -> None:
        points = _readonly_array(self.points, name="component points")
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("component points must have shape [N, 3]")
        fraction = _finite_float(
            "component point_fraction", self.point_fraction
        )
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("component point_fraction must lie in [0, 1]")
        voxel_count = _strict_int(
            "component voxel_count", self.voxel_count, 0
        )
        density = float(self.density)
        if np.isnan(density) or density < 0.0:
            raise ValueError(
                "component density must be non-negative and not NaN"
            )
        stable_index = _strict_int(
            "component stable_index", self.stable_index, 0
        )
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "point_fraction", fraction)
        object.__setattr__(self, "voxel_count", voxel_count)
        object.__setattr__(self, "density", density)
        object.__setattr__(self, "stable_index", stable_index)


@dataclass(frozen=True)
class DepthOccupancyProposal:
    """Immutable geometry proposal or an explicit identity fallback.

    Attributes:
        candidate: Proposed ``[cx, cy, cz, dx, dy, dz]`` AABB.  For every
            ``identity_*`` reason this is exactly the supplied original box.
        component_fraction: Fraction of active input points retained by the
            selected component.  Solid tracks use all memory points and report
            one.
        points: Deterministically sorted points used to form ``candidate``.
        planar: Whether the planar occupancy branch was selected.
        reason: ``candidate`` or a precise ``identity_*`` fallback reason.
        component_density: Selected points divided by their raw AABB volume.
        second_component_density: Density of the runner-up eligible planar
            component, or zero when there is no runner-up.
        density_ratio: Selected density divided by the runner-up eligible
            component density.  A unique eligible component reports infinity.
        branch: ``planar``, ``solid``, or ``identity`` when shape selection was
            impossible.
    """

    candidate: np.ndarray
    component_fraction: float
    points: np.ndarray
    planar: bool
    reason: str
    component_density: float
    second_component_density: float
    density_ratio: float
    branch: str

    def __post_init__(self) -> None:
        candidate = _readonly_array(
            self.candidate, name="proposal candidate", shape=(6,)
        )
        if np.any(candidate[3:6] <= 0.0):
            raise ValueError("proposal candidate dimensions must be positive")
        points = _readonly_array(self.points, name="proposal points")
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("proposal points must have shape [N, 3]")
        fraction = _finite_float(
            "proposal component_fraction", self.component_fraction
        )
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(
                "proposal component_fraction must lie in [0, 1]"
            )
        if not isinstance(self.planar, (bool, np.bool_)):
            raise ValueError("proposal planar must be Boolean")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("proposal reason must be a non-empty string")
        density = float(self.component_density)
        if np.isnan(density) or density < 0.0:
            raise ValueError(
                "proposal component_density must be non-negative and not NaN"
            )
        second_density = float(self.second_component_density)
        if np.isnan(second_density) or second_density < 0.0:
            raise ValueError(
                "proposal second_component_density must be non-negative "
                "and not NaN"
            )
        density_ratio = float(self.density_ratio)
        if np.isnan(density_ratio) or density_ratio < 0.0:
            raise ValueError(
                "proposal density_ratio must be non-negative and not NaN"
            )
        if self.branch not in {"identity", "planar", "solid"}:
            raise ValueError(
                "proposal branch must be identity, planar, or solid"
            )
        object.__setattr__(self, "candidate", candidate)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "component_fraction", fraction)
        object.__setattr__(self, "planar", bool(self.planar))
        object.__setattr__(self, "component_density", density)
        object.__setattr__(
            self, "second_component_density", second_density
        )
        object.__setattr__(self, "density_ratio", density_ratio)

    @property
    def proposed(self) -> bool:
        return self.reason == "candidate"


@dataclass(frozen=True)
class _Voxelization:
    keys: np.ndarray
    inverse: np.ndarray
    counts: np.ndarray


def _voxelize(points: np.ndarray, voxel_size: float) -> _Voxelization:
    # A track-local origin makes occupancy invariant to world translation and
    # avoids a component changing voxel assignment merely because it straddles
    # a global metric-grid boundary.
    scaled = (points - np.min(points, axis=0)[None, :]) / float(voxel_size)
    if np.max(np.abs(scaled), initial=0.0) > np.iinfo(np.int64).max / 4:
        raise ValueError("points are too large for the requested voxel size")
    # Float32 metric points can represent an exact voxel boundary a few ulps
    # below its mathematical value (for example 0.08 / 0.04).  A tiny
    # dimensionless nudge makes the specified floor rule stable without moving
    # any point that is meaningfully inside an adjacent voxel.
    point_keys = np.floor(scaled + 1e-6).astype(np.int64)
    keys, inverse, counts = np.unique(
        point_keys,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    return _Voxelization(keys=keys, inverse=inverse, counts=counts)


def _connected_components(
    keys: np.ndarray,
    *,
    neighbor_radius: int,
) -> List[np.ndarray]:
    if len(keys) == 0:
        return []
    key_to_index = {
        tuple(int(value) for value in key): index
        for index, key in enumerate(keys)
    }
    offsets = tuple(
        offset
        for offset in product(
            range(-neighbor_radius, neighbor_radius + 1), repeat=3
        )
        if offset != (0, 0, 0)
    )
    visited = np.zeros(len(keys), dtype=bool)
    components: List[np.ndarray] = []
    for seed in range(len(keys)):
        if visited[seed]:
            continue
        visited[seed] = True
        queue = [seed]
        component = []
        cursor = 0
        while cursor < len(queue):
            index = queue[cursor]
            cursor += 1
            component.append(index)
            key = keys[index]
            for offset in offsets:
                neighbor_key = (
                    int(key[0]) + offset[0],
                    int(key[1]) + offset[1],
                    int(key[2]) + offset[2],
                )
                neighbor = key_to_index.get(neighbor_key)
                if neighbor is None or visited[neighbor]:
                    continue
                visited[neighbor] = True
                queue.append(neighbor)
        components.append(np.asarray(sorted(component), dtype=np.int64))
    return components


def _point_aabb_density(points: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    spans = np.max(points, axis=0) - np.min(points, axis=0)
    volume = float(np.prod(spans))
    if not np.isfinite(volume):
        raise ValueError("component AABB volume must be finite")
    if volume <= 0.0:
        return float("inf")
    return float(len(points)) / volume


def _component_from_indices(
    points: np.ndarray,
    voxelization: _Voxelization,
    component_indices: np.ndarray,
    *,
    denominator: int,
) -> VoxelComponent:
    selected = np.isin(voxelization.inverse, component_indices)
    selected_points = points[selected]
    component_points = _sorted_points(selected_points)
    voxel_count = int(len(component_indices))
    stable_index = int(np.min(component_indices))
    density = _point_aabb_density(selected_points)
    return VoxelComponent(
        points=component_points,
        point_fraction=(
            float(len(component_points)) / float(denominator)
            if denominator > 0
            else 0.0
        ),
        voxel_count=voxel_count,
        density=density,
        stable_index=stable_index,
    )


def largest_voxel_connected_component(
    points: object,
    *,
    voxel_size: float = 0.02,
    neighbor_radius: int = 2,
) -> VoxelComponent:
    """Select a deterministic Chebyshev-connected voxel component.

    Components are ranked by descending point count, descending voxel count,
    then ascending stable voxel index.  ``np.unique`` lexicographically orders
    voxel keys, making the final tie-break independent of input point order.
    """

    array = _validated_points(points)
    voxel_size = _finite_float("voxel_size", voxel_size)
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")
    neighbor_radius = _strict_int(
        "neighbor_radius", neighbor_radius, 1
    )
    voxelization = _voxelize(array, voxel_size)
    components = _connected_components(
        voxelization.keys, neighbor_radius=neighbor_radius
    )
    records = [
        _component_from_indices(
            array,
            voxelization,
            indices,
            denominator=len(array),
        )
        for indices in components
    ]
    return min(
        records,
        key=lambda item: (
            -len(item.points),
            -item.voxel_count,
            item.stable_index,
        ),
    )


def _raw_bounds(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.min(points, axis=0).astype(np.float64),
        np.max(points, axis=0).astype(np.float64),
    )


def _envelope(
    points: np.ndarray,
    *,
    minimum_dimension: float,
    thin_axis_minimum: Optional[float] = None,
    original_dimensions: Optional[np.ndarray] = None,
) -> np.ndarray:
    lower, upper = _raw_bounds(points)
    center = 0.5 * (lower + upper)
    dimensions = upper - lower
    if thin_axis_minimum is not None:
        thin_axis = int(np.argmin(dimensions))
        original_floor = 0.0
        if original_dimensions is not None:
            original = np.asarray(original_dimensions, dtype=np.float64)
            if original.shape != (3,) or not np.isfinite(original).all():
                raise ValueError(
                    "original_dimensions must have finite shape [3]"
                )
            # The conservative clamp is defined by the original box's thin
            # extent, not merely the dimension that happens to align with the
            # point envelope's thinnest axis.
            original_floor = 0.65 * float(np.min(original))
        dimensions[thin_axis] = max(
            float(dimensions[thin_axis]),
            float(thin_axis_minimum),
            original_floor,
        )
    dimensions = np.maximum(dimensions, float(minimum_dimension))
    return np.concatenate((center, dimensions)).astype(np.float32)


def _planar_shape(
    points: np.ndarray,
    *,
    ratio_threshold: float,
) -> bool:
    lower, upper = _raw_bounds(points)
    extents = np.sort(np.maximum(upper - lower, 0.0))
    middle = float(extents[1])
    if middle <= 1e-12:
        return False
    ratio = float(extents[0]) / middle
    return bool(ratio <= float(ratio_threshold))


def _planar_density_component(
    points: np.ndarray,
    *,
    voxel_size: float,
    minimum_points_per_voxel: int,
    minimum_component_fraction: float,
) -> Tuple[Optional[VoxelComponent], float, float]:
    voxelization = _voxelize(points, voxel_size)
    dense_global_indices = np.flatnonzero(
        voxelization.counts >= int(minimum_points_per_voxel)
    )
    if len(dense_global_indices) == 0:
        return None, 0.0, 0.0
    dense_keys = voxelization.keys[dense_global_indices]
    dense_components = _connected_components(
        dense_keys,
        # Chebyshev radius one is exactly 26-connectivity.
        neighbor_radius=1,
    )
    records = []
    for dense_component in dense_components:
        global_indices = dense_global_indices[dense_component]
        record = _component_from_indices(
            points,
            voxelization,
            global_indices,
            denominator=len(points),
        )
        if (
            record.point_fraction + 1e-12
            >= float(minimum_component_fraction)
        ):
            records.append(record)
    if not records:
        return None, 0.0, 0.0
    ordered = sorted(
        records,
        key=lambda item: (
            -item.density,
            -len(item.points),
            -item.voxel_count,
            item.stable_index,
        ),
    )
    selected = ordered[0]
    if len(ordered) == 1:
        second_density = 0.0
        density_ratio = float("inf")
    else:
        second_density = float(ordered[1].density)
        if np.isinf(selected.density) and np.isinf(second_density):
            density_ratio = 1.0
        elif second_density <= 0.0:
            density_ratio = float("inf")
        else:
            density_ratio = float(selected.density) / second_density
    return selected, second_density, density_ratio


def _identity(
    original_box: np.ndarray,
    *,
    reason: str,
    branch: str = "identity",
    planar: bool = False,
    points: Optional[np.ndarray] = None,
    component_fraction: float = 0.0,
    component_density: float = 0.0,
    second_component_density: float = 0.0,
    density_ratio: float = 0.0,
) -> DepthOccupancyProposal:
    retained = (
        np.empty((0, 3), dtype=np.float32)
        if points is None
        else _sorted_points(points)
    )
    return DepthOccupancyProposal(
        candidate=original_box,
        component_fraction=component_fraction,
        points=retained,
        planar=planar,
        reason=reason,
        component_density=component_density,
        second_component_density=second_component_density,
        density_ratio=density_ratio,
        branch=branch,
    )


def propose_depth_occupancy_refinement(
    original_box: object,
    geometry_points: object,
    view_count: object,
    *,
    full_memory_points: Optional[object] = None,
    branch_hint: Optional[str] = None,
    config: Optional[Mapping[str, object]] = None,
) -> DepthOccupancyProposal:
    """Build a deterministic shape-adaptive AABB proposal.

    ``geometry_points`` should be the selected Top-K Mask-RGBD points.  When
    ``full_memory_points`` is supplied, those points form the active envelope
    and occupancy components; the cleaner Top-K points still determine the
    shape branch when they are available.  This permits compact objects such as
    sinks to retain surface coverage accumulated outside the selected Top-K.
    ``branch_hint`` may be ``"solid"`` or ``"planar"`` when an upstream
    semantic allowlist supplies a stricter shape contract; ``None`` retains
    the fully geometry-driven default.

    Invalid evidence and insufficient views/points return an explicit identity
    proposal.  An invalid ``original_box`` raises because no meaningful
    identity geometry exists in that case.
    """

    cfg = resolve_depth_occupancy_refiner_config(config)
    original = _validated_box(original_box)
    if branch_hint is not None:
        if not isinstance(branch_hint, str):
            return _identity(
                original, reason="identity_invalid_branch_hint"
            )
        branch_hint = branch_hint.strip().casefold()
        if branch_hint not in {"solid", "planar"}:
            return _identity(
                original, reason="identity_invalid_branch_hint"
            )

    if isinstance(view_count, (bool, np.bool_)) or not isinstance(
        view_count, Integral
    ):
        return _identity(original, reason="identity_invalid_view_count")
    views = int(view_count)
    if views < 0:
        return _identity(original, reason="identity_invalid_view_count")
    if views < int(cfg["min_views"]):
        return _identity(original, reason="identity_insufficient_views")

    try:
        geometry = _validated_points(
            geometry_points,
            name="geometry_points",
            allow_empty=True,
        )
    except ValueError:
        return _identity(original, reason="identity_invalid_geometry_points")

    if full_memory_points is None:
        active = geometry
    else:
        try:
            active = _validated_points(
                full_memory_points,
                name="full_memory_points",
                allow_empty=True,
            )
        except ValueError:
            return _identity(
                original, reason="identity_invalid_full_memory_points"
            )

    if len(active) < int(cfg["min_points"]):
        return _identity(
            original,
            reason="identity_insufficient_points",
            points=active,
        )

    classification = geometry if len(geometry) >= 3 else active
    planar = (
        branch_hint == "planar"
        if branch_hint is not None
        else _planar_shape(
            classification,
            ratio_threshold=float(cfg["planar_ratio_threshold"]),
        )
    )

    if planar:
        (
            component,
            second_component_density,
            density_ratio,
        ) = _planar_density_component(
            active,
            voxel_size=float(cfg["planar_voxel_size"]),
            minimum_points_per_voxel=int(
                cfg["planar_min_points_per_voxel"]
            ),
            minimum_component_fraction=float(
                cfg["planar_min_component_fraction"]
            ),
        )
        if component is None:
            return _identity(
                original,
                reason="identity_no_planar_component",
                branch="planar",
                planar=True,
                points=active,
            )
        candidate = _envelope(
            component.points,
            minimum_dimension=float(cfg["minimum_dimension"]),
            thin_axis_minimum=float(
                cfg["planar_thin_axis_minimum"]
            ),
            original_dimensions=original[3:6],
        )
        return DepthOccupancyProposal(
            candidate=candidate,
            component_fraction=component.point_fraction,
            points=component.points,
            planar=True,
            reason="candidate",
            component_density=component.density,
            second_component_density=second_component_density,
            density_ratio=density_ratio,
            branch="planar",
        )

    candidate = _envelope(
        active,
        minimum_dimension=float(cfg["minimum_dimension"]),
    )
    density = _point_aabb_density(active)
    sorted_active = _sorted_points(active)
    return DepthOccupancyProposal(
        candidate=candidate,
        component_fraction=1.0,
        points=sorted_active,
        planar=False,
        reason="candidate",
        component_density=density,
        second_component_density=0.0,
        density_ratio=1.0,
        branch="solid",
    )


__all__ = [
    "DEFAULT_DEPTH_OCCUPANCY_REFINER_CONFIG",
    "DepthOccupancyProposal",
    "VoxelComponent",
    "largest_voxel_connected_component",
    "propose_depth_occupancy_refinement",
    "resolve_depth_occupancy_refiner_config",
]
