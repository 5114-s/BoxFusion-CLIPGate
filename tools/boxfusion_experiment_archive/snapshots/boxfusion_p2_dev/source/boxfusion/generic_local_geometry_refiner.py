"""Deterministic generic Mask-RGBD local-geometry proposals.

The module is deliberately small and self contained: it has no model, file
system, ground-truth, or torch dependency.  It consumes the bounded per-view
point clouds already stored by :class:`ObjectGeometryMemory` and proposes a
conservative AABB refinement.  The online controller is responsible for
deciding whether a proposal may affect exported detections.

The algorithm has four safety layers:

* select at most five reliable views and deterministically subsample them;
* retain fine voxels observed by at least two views;
* choose a 26-connected coarse component anchored by points inside the
  original box (with a conservative nearby-part merge);
* update only box faces observed from at least two cameras, then enforce
  extent, centre-shift, and point-support gates.

For every ``identity_*`` result, ``candidate`` is an exact copy of
``original_box``.  Inputs are never modified and all diagnostic arrays in the
returned proposal are immutable.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from numbers import Integral, Real
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_GENERIC_LOCAL_GEOMETRY_CONFIG = {
    # View and point evidence.
    "max_views": 5,
    "max_points_per_view": 512,
    "min_views": 2,
    "min_points_per_view": 48,
    "min_total_points": 192,
    "crop_scale": 1.20,
    # Cross-view fine occupancy followed by coarse 26-connectivity.
    "fine_voxel_size": 0.04,
    "fine_min_view_consensus": 2,
    "coarse_voxel_size": 0.06,
    "min_component_views": 2,
    "min_component_inside_fraction": 0.50,
    "component_merge_gap": 0.10,
    # Per-face, camera-side-aware robust boundary estimation.
    "boundary_min_views": 2,
    "boundary_min_points_per_view": 12,
    "lower_quantile": 0.02,
    "upper_quantile": 0.98,
    "boundary_max_spread_ratio": 0.25,
    "boundary_max_spread_floor": 0.04,
    "boundary_max_spread_cap": 0.16,
    "boundary_blend": 0.50,
    "boundary_padding": 0.005,
    "maximum_face_shift_ratio": 0.15,
    # Final proposal safety gates.
    "minimum_extent_ratio": 0.75,
    "maximum_extent_ratio": 1.25,
    "maximum_center_shift_ratio": 0.15,
    "maximum_support_drop": 0.05,
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


def resolve_generic_local_geometry_config(
    config: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Return a detached, strictly validated generic-refiner config."""

    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("generic local geometry config must be a mapping")
    unknown = sorted(set(config) - set(DEFAULT_GENERIC_LOCAL_GEOMETRY_CONFIG))
    if unknown:
        raise ValueError(
            "Unknown generic local geometry config key(s): "
            + ", ".join(unknown)
        )

    resolved = dict(DEFAULT_GENERIC_LOCAL_GEOMETRY_CONFIG)
    resolved.update(config)

    for key, minimum in (
        ("max_views", 1),
        ("max_points_per_view", 1),
        ("min_views", 1),
        ("min_points_per_view", 1),
        ("min_total_points", 1),
        ("fine_min_view_consensus", 1),
        ("min_component_views", 1),
        ("boundary_min_views", 1),
        ("boundary_min_points_per_view", 1),
    ):
        resolved[key] = _strict_int(
            f"generic_local_geometry.{key}", resolved[key], minimum
        )

    for key in (
        "crop_scale",
        "fine_voxel_size",
        "coarse_voxel_size",
        "min_component_inside_fraction",
        "component_merge_gap",
        "lower_quantile",
        "upper_quantile",
        "boundary_max_spread_ratio",
        "boundary_max_spread_floor",
        "boundary_max_spread_cap",
        "boundary_blend",
        "boundary_padding",
        "maximum_face_shift_ratio",
        "minimum_extent_ratio",
        "maximum_extent_ratio",
        "maximum_center_shift_ratio",
        "maximum_support_drop",
    ):
        resolved[key] = _finite_float(
            f"generic_local_geometry.{key}", resolved[key]
        )

    for key in (
        "crop_scale",
        "fine_voxel_size",
        "coarse_voxel_size",
        "boundary_max_spread_floor",
        "boundary_max_spread_cap",
        "minimum_extent_ratio",
        "maximum_extent_ratio",
    ):
        if float(resolved[key]) <= 0.0:
            raise ValueError(f"generic_local_geometry.{key} must be positive")

    for key in (
        "min_component_inside_fraction",
        "lower_quantile",
        "upper_quantile",
        "boundary_max_spread_ratio",
        "boundary_blend",
        "maximum_face_shift_ratio",
        "maximum_center_shift_ratio",
        "maximum_support_drop",
    ):
        if not 0.0 <= float(resolved[key]) <= 1.0:
            raise ValueError(
                f"generic_local_geometry.{key} must lie in [0, 1]"
            )

    for key in ("component_merge_gap", "boundary_padding"):
        if float(resolved[key]) < 0.0:
            raise ValueError(
                f"generic_local_geometry.{key} must be non-negative"
            )

    if int(resolved["min_views"]) > int(resolved["max_views"]):
        raise ValueError("generic_local_geometry.min_views exceeds max_views")
    if int(resolved["fine_min_view_consensus"]) > int(
        resolved["max_views"]
    ):
        raise ValueError(
            "generic_local_geometry.fine_min_view_consensus exceeds max_views"
        )
    if int(resolved["min_component_views"]) > int(resolved["max_views"]):
        raise ValueError(
            "generic_local_geometry.min_component_views exceeds max_views"
        )
    if int(resolved["boundary_min_views"]) > int(resolved["max_views"]):
        raise ValueError(
            "generic_local_geometry.boundary_min_views exceeds max_views"
        )
    if float(resolved["crop_scale"]) < 1.0:
        raise ValueError("generic_local_geometry.crop_scale must be >= 1")
    if not float(resolved["lower_quantile"]) < float(
        resolved["upper_quantile"]
    ):
        raise ValueError(
            "generic_local_geometry.lower_quantile must be below "
            "upper_quantile"
        )
    if float(resolved["boundary_max_spread_floor"]) > float(
        resolved["boundary_max_spread_cap"]
    ):
        raise ValueError(
            "generic_local_geometry boundary spread floor exceeds cap"
        )
    if float(resolved["minimum_extent_ratio"]) > float(
        resolved["maximum_extent_ratio"]
    ):
        raise ValueError(
            "generic_local_geometry minimum extent ratio exceeds maximum"
        )
    return resolved


def _immutable_array(
    value: object,
    *,
    dtype: Optional[np.dtype] = None,
    shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError(f"diagnostic array must have shape {shape}")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class GenericLocalGeometryProposal:
    """Immutable proposal plus diagnostics needed by an observer/safety gate."""

    candidate: np.ndarray
    reason: str
    selected_frame_ids: Tuple[str, ...]
    eligible_view_count: int
    selected_view_count: int
    input_point_count: int
    cropped_point_count: int
    consensus_point_count: int
    component_count: int
    eligible_component_count: int
    anchor_point_count: int
    merged_component_count: int
    component_view_count: int
    component_inside_fraction: float
    original_support: float
    candidate_support: float
    support_drop: float
    boundary_values: np.ndarray
    boundary_view_counts: np.ndarray
    boundary_spreads: np.ndarray
    boundary_visible: np.ndarray
    extent_ratios: np.ndarray
    center_shift_ratios: np.ndarray
    points: np.ndarray

    def __post_init__(self) -> None:
        candidate = np.asarray(self.candidate)
        if candidate.shape != (6,) or not np.issubdtype(
            candidate.dtype, np.number
        ):
            raise ValueError("candidate must have numeric shape [6]")
        if not np.isfinite(candidate).all():
            raise ValueError("candidate must be finite")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a non-empty string")

        for name in (
            "eligible_view_count",
            "selected_view_count",
            "input_point_count",
            "cropped_point_count",
            "consensus_point_count",
            "component_count",
            "eligible_component_count",
            "anchor_point_count",
            "merged_component_count",
            "component_view_count",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or int(value) != value:
                raise ValueError(f"{name} must be a non-negative integer")
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, int(value))

        for name in (
            "component_inside_fraction",
            "original_support",
            "candidate_support",
            "support_drop",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)

        points = np.asarray(self.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("points must have shape [N, 3]")
        if not np.isfinite(points).all():
            raise ValueError("points must be finite")

        object.__setattr__(self, "candidate", _immutable_array(candidate))
        object.__setattr__(
            self,
            "selected_frame_ids",
            tuple(str(value) for value in self.selected_frame_ids),
        )
        object.__setattr__(
            self,
            "boundary_values",
            _immutable_array(self.boundary_values, dtype=np.float64, shape=(3, 2)),
        )
        object.__setattr__(
            self,
            "boundary_view_counts",
            _immutable_array(
                self.boundary_view_counts, dtype=np.int64, shape=(3, 2)
            ),
        )
        object.__setattr__(
            self,
            "boundary_spreads",
            _immutable_array(
                self.boundary_spreads, dtype=np.float64, shape=(3, 2)
            ),
        )
        object.__setattr__(
            self,
            "boundary_visible",
            _immutable_array(
                self.boundary_visible, dtype=np.bool_, shape=(3, 2)
            ),
        )
        object.__setattr__(
            self,
            "extent_ratios",
            _immutable_array(self.extent_ratios, dtype=np.float64, shape=(3,)),
        )
        object.__setattr__(
            self,
            "center_shift_ratios",
            _immutable_array(
                self.center_shift_ratios, dtype=np.float64, shape=(3,)
            ),
        )
        object.__setattr__(
            self, "points", _immutable_array(points, dtype=np.float64)
        )

    @property
    def is_candidate(self) -> bool:
        return self.reason == "candidate"


@dataclass(frozen=True)
class _View:
    frame_id: str
    points: np.ndarray
    camera: np.ndarray
    weight: float


def _exact_original(original_box: object) -> Tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(original_box)
    if (
        raw.shape != (6,)
        or not np.issubdtype(raw.dtype, np.number)
        or not np.isfinite(raw).all()
    ):
        raise ValueError("original_box must have finite numeric shape [6]")
    calc = np.asarray(raw, dtype=np.float64)
    if np.any(calc[3:] <= 0.0):
        raise ValueError("original_box dimensions must be positive")
    return np.array(raw, copy=True), calc


def _empty_diagnostics(original: np.ndarray) -> Dict[str, object]:
    lower = original[:3] - 0.5 * original[3:]
    upper = original[:3] + 0.5 * original[3:]
    return {
        "selected_frame_ids": (),
        "eligible_view_count": 0,
        "selected_view_count": 0,
        "input_point_count": 0,
        "cropped_point_count": 0,
        "consensus_point_count": 0,
        "component_count": 0,
        "eligible_component_count": 0,
        "anchor_point_count": 0,
        "merged_component_count": 0,
        "component_view_count": 0,
        "component_inside_fraction": 0.0,
        "original_support": 0.0,
        "candidate_support": 0.0,
        "support_drop": 0.0,
        "boundary_values": np.stack((lower, upper), axis=1),
        "boundary_view_counts": np.zeros((3, 2), dtype=np.int64),
        "boundary_spreads": np.full((3, 2), np.nan, dtype=np.float64),
        "boundary_visible": np.zeros((3, 2), dtype=np.bool_),
        "extent_ratios": np.ones(3, dtype=np.float64),
        "center_shift_ratios": np.zeros(3, dtype=np.float64),
        "points": np.empty((0, 3), dtype=np.float64),
    }


def _identity(
    original_raw: np.ndarray,
    original: np.ndarray,
    reason: str,
    diagnostics: Optional[Mapping[str, object]] = None,
) -> GenericLocalGeometryProposal:
    values = _empty_diagnostics(original)
    if diagnostics:
        values.update(diagnostics)
    return GenericLocalGeometryProposal(
        candidate=np.array(original_raw, copy=True),
        reason=reason,
        **values,
    )


def _record_value(record: object, name: str) -> object:
    if isinstance(record, Mapping):
        if name not in record:
            raise ValueError(f"view record is missing {name}")
        return record[name]
    if not hasattr(record, name):
        raise ValueError(f"view record is missing {name}")
    return getattr(record, name)


def _sorted_sample(points: np.ndarray, limit: int) -> np.ndarray:
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    sorted_points = np.asarray(points[order], dtype=np.float64)
    if len(sorted_points) <= limit:
        return sorted_points
    # Endpoint-inclusive positions preserve the full geometric envelope while
    # remaining exactly invariant to the incoming point order.
    indices = np.linspace(0, len(sorted_points) - 1, limit, dtype=np.int64)
    return sorted_points[indices]


def _prepare_views(
    view_records: object,
    config: Mapping[str, object],
    crop_lower: np.ndarray,
    crop_upper: np.ndarray,
) -> Tuple[Tuple[_View, ...], int, int]:
    if isinstance(view_records, (str, bytes)) or not isinstance(
        view_records, Sequence
    ):
        raise ValueError("view_records must be a sequence")

    prepared = []
    input_count = 0
    eligible_count = 0
    minimum = int(config["min_points_per_view"])
    for record in view_records:
        points_value = _record_value(record, "points_world")
        points = np.asarray(points_value)
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or not np.issubdtype(points.dtype, np.number)
            or not np.isfinite(points).all()
        ):
            raise ValueError("view points_world must have finite numeric shape [N, 3]")
        points = np.asarray(points, dtype=np.float64)
        input_count += len(points)

        quality = _finite_float("view quality", _record_value(record, "quality"))
        valid_depth = _finite_float(
            "view valid_depth_ratio",
            _record_value(record, "valid_depth_ratio"),
        )
        if quality < 0.0 or not 0.0 <= valid_depth <= 1.0:
            raise ValueError(
                "view quality must be non-negative and valid_depth_ratio "
                "must lie in [0, 1]"
            )
        camera = np.asarray(_record_value(record, "camera_position"))
        if (
            camera.shape != (3,)
            or not np.issubdtype(camera.dtype, np.number)
            or not np.isfinite(camera).all()
        ):
            raise ValueError("view camera_position must have finite shape [3]")
        frame_id = str(_record_value(record, "frame_id"))

        inside_crop = np.logical_and(
            points >= crop_lower[None, :], points <= crop_upper[None, :]
        ).all(axis=1)
        cropped = points[inside_crop]
        if len(cropped) < minimum:
            continue
        eligible_count += 1
        sampled = _sorted_sample(
            cropped, int(config["max_points_per_view"])
        )
        # Point count is deliberately absent from the ranking weight: a dense
        # depth frame must not crowd all other camera sides out of Top-K.
        weight = max(quality, 1e-6) * max(valid_depth, 1e-6)
        stable_point_key = tuple(
            np.round(
                np.concatenate(
                    (sampled[0], sampled[-1], np.mean(sampled, axis=0))
                ),
                decimals=9,
            ).tolist()
        )
        rank_key = (
            -weight,
            frame_id,
            tuple(np.round(camera.astype(np.float64), 9).tolist()),
            len(sampled),
            stable_point_key,
        )
        prepared.append(
            (
                rank_key,
                _View(
                    frame_id=frame_id,
                    points=sampled,
                    camera=np.asarray(camera, dtype=np.float64),
                    weight=float(weight),
                ),
            )
        )

    prepared.sort(key=lambda item: item[0])
    selected = tuple(
        item[1] for item in prepared[: int(config["max_views"])]
    )
    return selected, input_count, eligible_count


def _voxel_keys(
    points: np.ndarray, origin: np.ndarray, voxel_size: float
) -> np.ndarray:
    return np.floor((points - origin[None, :]) / voxel_size).astype(np.int64)


def _connected_components(keys: np.ndarray) -> Tuple[np.ndarray, ...]:
    voxel_to_points: Dict[Tuple[int, int, int], list] = {}
    for index, key_array in enumerate(keys):
        key = tuple(int(value) for value in key_array)
        voxel_to_points.setdefault(key, []).append(index)
    remaining = set(voxel_to_points)
    offsets = tuple(
        value
        for value in product((-1, 0, 1), repeat=3)
        if value != (0, 0, 0)
    )
    result = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        stack = [seed]
        voxels = []
        while stack:
            current = stack.pop()
            voxels.append(current)
            for offset in offsets:
                neighbor = (
                    current[0] + offset[0],
                    current[1] + offset[1],
                    current[2] + offset[2],
                )
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        indices = []
        for voxel in sorted(voxels):
            indices.extend(voxel_to_points[voxel])
        result.append(np.asarray(sorted(indices), dtype=np.int64))
    return tuple(result)


def _inside_box(
    points: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    return np.logical_and(
        points >= lower[None, :], points <= upper[None, :]
    ).all(axis=1)


def _aabb_gap(first: np.ndarray, second: np.ndarray) -> float:
    first_lower = np.min(first, axis=0)
    first_upper = np.max(first, axis=0)
    second_lower = np.min(second, axis=0)
    second_upper = np.max(second, axis=0)
    gap = np.maximum(
        np.maximum(first_lower - second_upper, second_lower - first_upper),
        0.0,
    )
    return float(np.max(gap))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.lexsort((weights, values))
    sorted_values = values[order]
    sorted_weights = weights[order]
    threshold = 0.5 * float(np.sum(sorted_weights))
    index = int(
        np.searchsorted(np.cumsum(sorted_weights), threshold, side="left")
    )
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _support(
    points: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    if len(points) == 0:
        return 0.0
    return float(np.mean(_inside_box(points, lower, upper)))


def propose_generic_local_geometry(
    original_box: object,
    view_records: object,
    config: Optional[Mapping[str, object]] = None,
) -> GenericLocalGeometryProposal:
    """Propose a conservative generic local AABB from multi-view RGB-D points."""

    resolved = resolve_generic_local_geometry_config(config)
    original_raw, original = _exact_original(original_box)
    center = original[:3]
    extent = original[3:]
    original_lower = center - 0.5 * extent
    original_upper = center + 0.5 * extent
    crop_extent = extent * float(resolved["crop_scale"])
    crop_lower = center - 0.5 * crop_extent
    crop_upper = center + 0.5 * crop_extent

    try:
        views, input_count, eligible_count = _prepare_views(
            view_records, resolved, crop_lower, crop_upper
        )
    except (TypeError, ValueError):
        return _identity(
            original_raw,
            original,
            "identity_invalid_view_record",
        )

    base = {
        "selected_frame_ids": tuple(view.frame_id for view in views),
        "eligible_view_count": eligible_count,
        "selected_view_count": len(views),
        "input_point_count": input_count,
        "cropped_point_count": int(sum(len(view.points) for view in views)),
    }
    if len(views) < int(resolved["min_views"]):
        return _identity(
            original_raw, original, "identity_insufficient_views", base
        )

    total_points = int(sum(len(view.points) for view in views))
    if total_points < int(resolved["min_total_points"]):
        return _identity(
            original_raw, original, "identity_insufficient_points", base
        )

    points = np.concatenate(tuple(view.points for view in views), axis=0)
    point_views = np.concatenate(
        tuple(
            np.full(len(view.points), index, dtype=np.int64)
            for index, view in enumerate(views)
        )
    )
    fine_keys = _voxel_keys(
        points, original_lower, float(resolved["fine_voxel_size"])
    )
    fine_view_sets: Dict[Tuple[int, int, int], set] = {}
    for key_array, view_index in zip(fine_keys, point_views):
        key = tuple(int(value) for value in key_array)
        fine_view_sets.setdefault(key, set()).add(int(view_index))
    consensus_mask = np.asarray(
        [
            len(fine_view_sets[tuple(int(value) for value in key)])
            >= int(resolved["fine_min_view_consensus"])
            for key in fine_keys
        ],
        dtype=np.bool_,
    )
    consensus_points = points[consensus_mask]
    consensus_views = point_views[consensus_mask]
    diagnostics = dict(base)
    diagnostics["consensus_point_count"] = len(consensus_points)
    if len(consensus_points) < int(resolved["min_total_points"]):
        return _identity(
            original_raw,
            original,
            "identity_insufficient_consensus_points",
            diagnostics,
        )

    coarse_keys = _voxel_keys(
        consensus_points,
        original_lower,
        float(resolved["coarse_voxel_size"]),
    )
    components = _connected_components(coarse_keys)
    component_details = []
    minimum_views = int(resolved["min_component_views"])
    minimum_inside = float(resolved["min_component_inside_fraction"])
    for stable_index, indices in enumerate(components):
        component_points = consensus_points[indices]
        component_views = consensus_views[indices]
        inside_count = int(
            np.sum(
                _inside_box(
                    component_points, original_lower, original_upper
                )
            )
        )
        inside_fraction = inside_count / float(len(indices))
        view_count = len(set(component_views.tolist()))
        if view_count >= minimum_views and inside_fraction >= minimum_inside:
            component_details.append(
                {
                    "indices": indices,
                    "points": component_points,
                    "views": component_views,
                    "inside_count": inside_count,
                    "inside_fraction": inside_fraction,
                    "view_count": view_count,
                    "stable_index": stable_index,
                }
            )
    diagnostics.update(
        {
            "component_count": len(components),
            "eligible_component_count": len(component_details),
        }
    )
    if not component_details:
        return _identity(
            original_raw, original, "identity_no_anchor_component", diagnostics
        )

    component_details.sort(
        key=lambda item: (
            -int(item["inside_count"]),
            -int(item["view_count"]),
            -len(item["indices"]),
            int(item["stable_index"]),
        )
    )
    anchor = component_details[0]
    merged = [anchor]
    merge_gap = float(resolved["component_merge_gap"])
    # Merge only other independently multi-view, mostly-in-box components and
    # only when their physical gap to the growing anchor is small.
    merged_points = np.asarray(anchor["points"], dtype=np.float64)
    for other in component_details[1:]:
        if _aabb_gap(merged_points, np.asarray(other["points"])) <= merge_gap:
            merged.append(other)
            merged_points = np.concatenate(
                (merged_points, np.asarray(other["points"])), axis=0
            )

    selected_indices = np.concatenate(
        tuple(np.asarray(item["indices"], dtype=np.int64) for item in merged)
    )
    selected_indices = np.unique(selected_indices)
    selected_points = consensus_points[selected_indices]
    selected_point_views = consensus_views[selected_indices]
    sort_order = np.lexsort(
        (
            selected_point_views,
            selected_points[:, 2],
            selected_points[:, 1],
            selected_points[:, 0],
        )
    )
    selected_points = selected_points[sort_order]
    selected_point_views = selected_point_views[sort_order]
    selected_view_count = len(set(selected_point_views.tolist()))
    selected_inside_fraction = float(
        np.mean(
            _inside_box(selected_points, original_lower, original_upper)
        )
    )
    diagnostics.update(
        {
            "anchor_point_count": len(anchor["indices"]),
            "merged_component_count": len(merged),
            "component_view_count": selected_view_count,
            "component_inside_fraction": selected_inside_fraction,
            "points": selected_points,
        }
    )

    boundaries = np.stack((original_lower, original_upper), axis=1)
    boundary_values = np.array(boundaries, copy=True)
    boundary_counts = np.zeros((3, 2), dtype=np.int64)
    boundary_spreads = np.full((3, 2), np.nan, dtype=np.float64)
    boundary_visible = np.zeros((3, 2), dtype=np.bool_)
    low_quantile = float(resolved["lower_quantile"])
    high_quantile = float(resolved["upper_quantile"])
    min_boundary_points = int(resolved["boundary_min_points_per_view"])
    minimum_boundary_views = int(resolved["boundary_min_views"])
    padding = float(resolved["boundary_padding"])
    blend = float(resolved["boundary_blend"])
    max_face_shift = float(resolved["maximum_face_shift_ratio"])

    for axis in range(3):
        spread_limit = np.clip(
            float(resolved["boundary_max_spread_ratio"]) * extent[axis],
            float(resolved["boundary_max_spread_floor"]),
            float(resolved["boundary_max_spread_cap"]),
        )
        for face in range(2):
            values = []
            weights = []
            for view_index, view in enumerate(views):
                is_visible_side = (
                    view.camera[axis] < center[axis]
                    if face == 0
                    else view.camera[axis] > center[axis]
                )
                if not is_visible_side:
                    continue
                view_points = selected_points[
                    selected_point_views == view_index
                ]
                if len(view_points) < min_boundary_points:
                    continue
                quantile = low_quantile if face == 0 else high_quantile
                values.append(
                    float(np.quantile(view_points[:, axis], quantile))
                )
                weights.append(float(view.weight))
            boundary_counts[axis, face] = len(values)
            if len(values) < minimum_boundary_views:
                continue
            value_array = np.asarray(values, dtype=np.float64)
            weight_array = np.asarray(weights, dtype=np.float64)
            spread = float(np.max(value_array) - np.min(value_array))
            boundary_spreads[axis, face] = spread
            if spread > spread_limit:
                continue
            estimate = _weighted_median(value_array, weight_array)
            estimate += -padding if face == 0 else padding
            original_face = boundaries[axis, face]
            proposed = original_face + blend * (estimate - original_face)
            shift_limit = max_face_shift * extent[axis]
            proposed = float(
                np.clip(
                    proposed,
                    original_face - shift_limit,
                    original_face + shift_limit,
                )
            )
            # The crop is a hard locality guard.
            proposed = float(
                np.clip(proposed, crop_lower[axis], crop_upper[axis])
            )
            boundary_values[axis, face] = proposed
            boundary_visible[axis, face] = True

    diagnostics.update(
        {
            "boundary_values": boundary_values,
            "boundary_view_counts": boundary_counts,
            "boundary_spreads": boundary_spreads,
            "boundary_visible": boundary_visible,
        }
    )
    if not bool(np.any(boundary_visible)):
        return _identity(
            original_raw,
            original,
            "identity_insufficient_boundary_consensus",
            diagnostics,
        )
    if np.any(boundary_values[:, 1] <= boundary_values[:, 0]):
        return _identity(
            original_raw, original, "identity_invalid_boundaries", diagnostics
        )

    candidate_extent = boundary_values[:, 1] - boundary_values[:, 0]
    candidate_center = 0.5 * (
        boundary_values[:, 1] + boundary_values[:, 0]
    )
    extent_ratios = candidate_extent / extent
    center_shift_ratios = np.abs(candidate_center - center) / extent
    diagnostics.update(
        {
            "extent_ratios": extent_ratios,
            "center_shift_ratios": center_shift_ratios,
        }
    )
    if np.any(
        extent_ratios < float(resolved["minimum_extent_ratio"])
    ) or np.any(extent_ratios > float(resolved["maximum_extent_ratio"])):
        return _identity(
            original_raw, original, "identity_extent_ratio", diagnostics
        )
    if np.any(
        center_shift_ratios
        > float(resolved["maximum_center_shift_ratio"])
    ):
        return _identity(
            original_raw, original, "identity_center_shift", diagnostics
        )

    original_support = _support(
        selected_points, original_lower, original_upper
    )
    candidate_support = _support(
        selected_points, boundary_values[:, 0], boundary_values[:, 1]
    )
    support_drop = max(0.0, original_support - candidate_support)
    diagnostics.update(
        {
            "original_support": original_support,
            "candidate_support": candidate_support,
            "support_drop": support_drop,
        }
    )
    if support_drop > float(resolved["maximum_support_drop"]):
        return _identity(
            original_raw, original, "identity_support_drop", diagnostics
        )

    candidate_float = np.concatenate((candidate_center, candidate_extent))
    output_dtype = (
        original_raw.dtype
        if np.issubdtype(original_raw.dtype, np.floating)
        else np.dtype(np.float64)
    )
    return GenericLocalGeometryProposal(
        candidate=np.asarray(candidate_float, dtype=output_dtype),
        reason="candidate",
        **diagnostics,
    )


__all__ = [
    "DEFAULT_GENERIC_LOCAL_GEOMETRY_CONFIG",
    "GenericLocalGeometryProposal",
    "propose_generic_local_geometry",
    "resolve_generic_local_geometry_config",
]
