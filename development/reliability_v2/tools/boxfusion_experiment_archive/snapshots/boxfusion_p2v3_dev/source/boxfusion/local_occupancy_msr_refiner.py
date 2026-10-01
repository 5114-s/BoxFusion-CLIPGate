"""Deterministic local occupancy/MSR refinement for oriented boxes.

This module is a standalone, NumPy-only geometry proposal.  It borrows the
useful geometric ideas from sparse-grid detectors without requiring SGCDet,
PyTorch, a checkpoint, ground truth, or the online refinement controller:

* Mask-depth points from every view are transformed into the *original* OBB
  frame, so the upstream basis (and therefore yaw) is never re-estimated.
* Cross-view fine occupancy removes single-view leakage.  A coarse sparse
  grid and deterministic 26-connectivity select the component anchored by
  the original box.
* Camera-to-surface rays accumulate explicit empty-space evidence.
* A coarse/fine multi-scale response (MSR) estimates each of the six faces.
  A face changes only when its own view, occupancy, uncertainty, and empty
  space evidence are sufficient.
* Face, centre, extent, locality, and point-support limits are hard clamps.

The proposal deliberately does not decide whether AP50 improves.  It returns
``gate_features`` with a fixed, documented schema for an AP50-aware observer
or learned gate.  Every failure after the original OBB has been validated is
fail-open: ``candidate_corners`` is an exact copy of ``original_corners`` and
``reason`` explains why.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from numbers import Integral, Real
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


LOCAL_OCCUPANCY_MSR_SOURCE = "occupancy_msr"

_FACE_NAMES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
LOCAL_OCCUPANCY_MSR_GATE_FEATURE_NAMES = (
    tuple(f"face_residual_ratio_{name}" for name in _FACE_NAMES)
    + tuple(f"face_support_{name}" for name in _FACE_NAMES)
    + tuple(f"face_uncertainty_ratio_{name}" for name in _FACE_NAMES)
    + tuple(f"face_empty_evidence_{name}" for name in _FACE_NAMES)
    + tuple(f"face_visibility_{name}" for name in _FACE_NAMES)
    + (
        "selected_view_fraction",
        "component_view_fraction",
        "consensus_point_fraction",
        "component_point_fraction",
        "component_inside_fraction",
        "coarse_occupied_voxel_fraction",
        "fine_occupied_voxel_fraction",
        "supported_face_fraction",
        "mean_absolute_face_residual_ratio",
        "maximum_absolute_face_residual_ratio",
        "mean_face_uncertainty_ratio",
        "maximum_face_uncertainty_ratio",
        "extent_ratio_x",
        "extent_ratio_y",
        "extent_ratio_z",
        "center_shift_ratio_x",
        "center_shift_ratio_y",
        "center_shift_ratio_z",
    )
)
LOCAL_OCCUPANCY_MSR_GATE_FEATURE_DIM = len(
    LOCAL_OCCUPANCY_MSR_GATE_FEATURE_NAMES
)
assert LOCAL_OCCUPANCY_MSR_GATE_FEATURE_DIM == 48
# Concise aliases used by diagnostic and AP50-gate dataset adapters.
OCCUPANCY_MSR_FEATURE_NAMES = LOCAL_OCCUPANCY_MSR_GATE_FEATURE_NAMES
OCCUPANCY_MSR_FEATURE_DIM = LOCAL_OCCUPANCY_MSR_GATE_FEATURE_DIM


DEFAULT_LOCAL_OCCUPANCY_MSR_CONFIG = {
    # Bounded, deterministic multi-view evidence.
    "max_views": 5,
    "max_points_per_view": 768,
    "min_views": 2,
    "min_points_per_view": 32,
    "min_total_points": 128,
    "crop_scale": 1.35,
    # Fine cross-view occupancy and coarse 26-connected anchoring.
    "fine_voxel_size": 0.025,
    "fine_min_view_consensus": 2,
    "coarse_voxel_size": 0.075,
    "min_component_views": 2,
    "min_component_points": 64,
    "min_component_inside_fraction": 0.40,
    # Empty-space ray carving.  The step is a multiple of fine_voxel_size.
    "empty_ray_step_ratio": 0.50,
    "empty_ray_surface_margin_ratio": 0.60,
    "empty_ray_max_samples": 96,
    "empty_shell_voxels": 2,
    # Per-face coarse/fine multi-scale response.
    "lower_quantile": 0.02,
    "upper_quantile": 0.98,
    "face_min_views": 2,
    "face_min_points_per_view": 10,
    "face_max_uncertainty_ratio": 0.15,
    "face_min_support": 0.25,
    "face_min_empty_evidence": 0.01,
    "msr_fine_weight": 0.80,
    "boundary_blend": 0.65,
    "boundary_padding": 0.005,
    "minimum_face_residual": 1e-5,
    # Hard geometric clamps.  Scaling a residual never moves an unsupported
    # face, which is important for partial-view safety.
    "maximum_face_shift_ratio": 0.18,
    "minimum_extent_ratio": 0.70,
    "maximum_extent_ratio": 1.25,
    "maximum_center_shift_ratio": 0.15,
    "maximum_support_drop": 0.08,
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


def resolve_local_occupancy_msr_config(
    config: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Return a detached, strictly validated occupancy/MSR configuration."""

    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("local occupancy/MSR config must be a mapping")
    unknown = sorted(set(config) - set(DEFAULT_LOCAL_OCCUPANCY_MSR_CONFIG))
    if unknown:
        raise ValueError(
            "Unknown local occupancy/MSR config key(s): " + ", ".join(unknown)
        )

    resolved = dict(DEFAULT_LOCAL_OCCUPANCY_MSR_CONFIG)
    resolved.update(config)
    for key, minimum in (
        ("max_views", 1),
        ("max_points_per_view", 1),
        ("min_views", 1),
        ("min_points_per_view", 1),
        ("min_total_points", 1),
        ("fine_min_view_consensus", 1),
        ("min_component_views", 1),
        ("min_component_points", 1),
        ("empty_ray_max_samples", 1),
        ("empty_shell_voxels", 1),
        ("face_min_views", 1),
        ("face_min_points_per_view", 1),
    ):
        resolved[key] = _strict_int(
            f"local_occupancy_msr.{key}", resolved[key], minimum
        )

    float_keys = (
        "crop_scale",
        "fine_voxel_size",
        "coarse_voxel_size",
        "min_component_inside_fraction",
        "empty_ray_step_ratio",
        "empty_ray_surface_margin_ratio",
        "lower_quantile",
        "upper_quantile",
        "face_max_uncertainty_ratio",
        "face_min_support",
        "face_min_empty_evidence",
        "msr_fine_weight",
        "boundary_blend",
        "boundary_padding",
        "minimum_face_residual",
        "maximum_face_shift_ratio",
        "minimum_extent_ratio",
        "maximum_extent_ratio",
        "maximum_center_shift_ratio",
        "maximum_support_drop",
    )
    for key in float_keys:
        resolved[key] = _finite_float(
            f"local_occupancy_msr.{key}", resolved[key]
        )

    for key in (
        "crop_scale",
        "fine_voxel_size",
        "coarse_voxel_size",
        "empty_ray_step_ratio",
        "minimum_extent_ratio",
        "maximum_extent_ratio",
    ):
        if float(resolved[key]) <= 0.0:
            raise ValueError(f"local_occupancy_msr.{key} must be positive")
    for key in ("boundary_padding", "minimum_face_residual"):
        if float(resolved[key]) < 0.0:
            raise ValueError(
                f"local_occupancy_msr.{key} must be non-negative"
            )
    for key in (
        "min_component_inside_fraction",
        "empty_ray_surface_margin_ratio",
        "lower_quantile",
        "upper_quantile",
        "face_max_uncertainty_ratio",
        "face_min_support",
        "face_min_empty_evidence",
        "msr_fine_weight",
        "boundary_blend",
        "maximum_face_shift_ratio",
        "maximum_center_shift_ratio",
        "maximum_support_drop",
    ):
        if not 0.0 <= float(resolved[key]) <= 1.0:
            raise ValueError(f"local_occupancy_msr.{key} must lie in [0, 1]")

    if int(resolved["min_views"]) > int(resolved["max_views"]):
        raise ValueError("local_occupancy_msr.min_views exceeds max_views")
    for key in (
        "fine_min_view_consensus",
        "min_component_views",
        "face_min_views",
    ):
        if int(resolved[key]) > int(resolved["max_views"]):
            raise ValueError(f"local_occupancy_msr.{key} exceeds max_views")
    if float(resolved["crop_scale"]) < 1.0:
        raise ValueError("local_occupancy_msr.crop_scale must be >= 1")
    if not float(resolved["lower_quantile"]) < float(
        resolved["upper_quantile"]
    ):
        raise ValueError(
            "local_occupancy_msr.lower_quantile must be below upper_quantile"
        )
    if float(resolved["minimum_extent_ratio"]) > float(
        resolved["maximum_extent_ratio"]
    ):
        raise ValueError(
            "local_occupancy_msr minimum extent ratio exceeds maximum"
        )
    if not (
        float(resolved["minimum_extent_ratio"])
        <= 1.0
        <= float(resolved["maximum_extent_ratio"])
    ):
        raise ValueError(
            "local_occupancy_msr extent ratio interval must contain identity"
        )
    if float(resolved["coarse_voxel_size"]) < float(
        resolved["fine_voxel_size"]
    ):
        raise ValueError(
            "local_occupancy_msr.coarse_voxel_size must be >= "
            "fine_voxel_size"
        )
    return resolved


def _immutable(
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


def _validate_finite_array(
    value: object, name: str, shape: Tuple[int, ...]
) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.shape != shape
        or not np.issubdtype(array.dtype, np.number)
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must have finite numeric shape {shape}")
    return array


@dataclass(frozen=True)
class LocalOccupancyMSRProposal:
    """Immutable OBB candidate and diagnostics for an external safety gate."""

    candidate_corners: np.ndarray
    original_corners: np.ndarray
    reason: str
    detail_reasons: Tuple[str, ...]
    selected_frame_ids: Tuple[str, ...]
    eligible_view_count: int
    selected_view_count: int
    input_point_count: int
    cropped_point_count: int
    consensus_point_count: int
    fine_occupied_voxel_count: int
    coarse_occupied_voxel_count: int
    component_count: int
    component_point_count: int
    component_view_count: int
    component_inside_fraction: float
    frame_center: np.ndarray
    frame_basis: np.ndarray
    original_local_box: np.ndarray
    candidate_local_box: np.ndarray
    local_points: np.ndarray
    face_residuals: np.ndarray
    face_support: np.ndarray
    face_uncertainty: np.ndarray
    face_empty_evidence: np.ndarray
    face_supported: np.ndarray
    face_view_counts: np.ndarray
    face_reasons: Tuple[str, ...]
    extent_ratios: np.ndarray
    center_shift_ratios: np.ndarray
    original_support: float
    candidate_support: float
    support_drop: float
    gate_features: np.ndarray
    source: str = LOCAL_OCCUPANCY_MSR_SOURCE

    def __post_init__(self) -> None:
        for name in (
            "eligible_view_count",
            "selected_view_count",
            "input_point_count",
            "cropped_point_count",
            "consensus_point_count",
            "fine_occupied_voxel_count",
            "coarse_occupied_voxel_count",
            "component_count",
            "component_point_count",
            "component_view_count",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or int(value) != value
                or int(value) < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
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
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a non-empty string")
        if self.source != LOCAL_OCCUPANCY_MSR_SOURCE:
            raise ValueError("source must identify the occupancy/MSR route")

        object.__setattr__(
            self,
            "candidate_corners",
            _immutable(self.candidate_corners, shape=(8, 3)),
        )
        object.__setattr__(
            self,
            "original_corners",
            _immutable(self.original_corners, shape=(8, 3)),
        )
        object.__setattr__(
            self,
            "detail_reasons",
            tuple(str(value) for value in self.detail_reasons),
        )
        object.__setattr__(
            self,
            "selected_frame_ids",
            tuple(str(value) for value in self.selected_frame_ids),
        )
        object.__setattr__(
            self, "frame_center", _immutable(self.frame_center, shape=(3,))
        )
        object.__setattr__(
            self, "frame_basis", _immutable(self.frame_basis, shape=(3, 3))
        )
        object.__setattr__(
            self,
            "original_local_box",
            _immutable(self.original_local_box, shape=(6,)),
        )
        object.__setattr__(
            self,
            "candidate_local_box",
            _immutable(self.candidate_local_box, shape=(6,)),
        )
        points = np.asarray(self.local_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("local_points must have shape [N, 3]")
        if not np.isfinite(points).all():
            raise ValueError("local_points must be finite")
        object.__setattr__(
            self, "local_points", _immutable(points, dtype=np.float64)
        )
        for name, dtype in (
            ("face_residuals", np.float64),
            ("face_support", np.float64),
            ("face_uncertainty", np.float64),
            ("face_empty_evidence", np.float64),
            ("face_supported", np.bool_),
            ("face_view_counts", np.int64),
        ):
            object.__setattr__(
                self,
                name,
                _immutable(getattr(self, name), dtype=dtype, shape=(3, 2)),
            )
        reasons = tuple(str(value) for value in self.face_reasons)
        if len(reasons) != 6 or any(not value for value in reasons):
            raise ValueError("face_reasons must contain six non-empty strings")
        object.__setattr__(self, "face_reasons", reasons)
        object.__setattr__(
            self,
            "extent_ratios",
            _immutable(self.extent_ratios, dtype=np.float64, shape=(3,)),
        )
        object.__setattr__(
            self,
            "center_shift_ratios",
            _immutable(
                self.center_shift_ratios, dtype=np.float64, shape=(3,)
            ),
        )
        features = _immutable(
            self.gate_features,
            dtype=np.float32,
            shape=(LOCAL_OCCUPANCY_MSR_GATE_FEATURE_DIM,),
        )
        if not np.isfinite(features).all():
            raise ValueError("gate_features must be finite")
        object.__setattr__(self, "gate_features", features)

    @property
    def is_candidate(self) -> bool:
        return self.reason == "candidate"

    @property
    def changed_face_count(self) -> int:
        threshold = np.finfo(np.float64).eps
        return int(np.sum(np.abs(self.face_residuals) > threshold))

    @property
    def gate_feature_names(self) -> Tuple[str, ...]:
        return LOCAL_OCCUPANCY_MSR_GATE_FEATURE_NAMES

    @property
    def feature_vector(self) -> np.ndarray:
        """Fixed finite vector consumed by an external AP50-aware gate."""

        return self.gate_features

    @property
    def candidate(self) -> np.ndarray:
        """Concise compatibility alias for ``candidate_corners``."""

        return self.candidate_corners

    @property
    def face_visibility(self) -> np.ndarray:
        """Per-face fraction of selected views that observed that face."""

        visibility = self.face_view_counts.astype(np.float64) / float(
            max(self.selected_view_count, 1)
        )
        visibility.setflags(write=False)
        return visibility


# Short public alias for callers that do not need the route qualifier.
OccupancyMSRProposal = LocalOccupancyMSRProposal


@dataclass(frozen=True)
class _View:
    frame_id: str
    points: np.ndarray
    camera: np.ndarray
    weight: float


def _oriented_frame(
    original_corners: object,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = _validate_finite_array(
        original_corners, "original_corners", (8, 3)
    )
    values = np.asarray(raw, dtype=np.float64)
    center = values.mean(axis=0)
    edges = np.stack(
        (
            values[1] - values[0],
            values[3] - values[0],
            values[4] - values[0],
        ),
        axis=1,
    )
    dimensions = np.linalg.norm(edges, axis=0)
    if np.any(dimensions <= 1e-8):
        raise ValueError("original_corners must define positive-volume edges")
    basis = edges / dimensions[None, :]
    if not np.allclose(
        basis.T @ basis, np.eye(3), atol=1e-3, rtol=0.0
    ):
        raise ValueError("original_corners edges must be orthogonal")
    if float(np.linalg.det(basis)) <= 0.0:
        raise ValueError("original_corners basis must be right-handed")
    reconstructed = _local_box_to_world(
        np.concatenate((np.zeros(3), dimensions)), center, basis
    )
    tolerance = max(float(np.max(dimensions)) * 1e-4, 1e-5)
    if not np.allclose(reconstructed, values, atol=tolerance, rtol=0.0):
        raise ValueError(
            "original_corners do not follow BoxFusion corner ordering"
        )
    return np.array(raw, copy=True), center, dimensions, basis


def _local_corners(box: np.ndarray) -> np.ndarray:
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
    return box[None, :3] + signs * (0.5 * box[None, 3:6])


def _local_box_to_world(
    box: np.ndarray, frame_center: np.ndarray, frame_basis: np.ndarray
) -> np.ndarray:
    return _local_corners(np.asarray(box, dtype=np.float64)) @ (
        np.asarray(frame_basis, dtype=np.float64).T
    ) + np.asarray(frame_center, dtype=np.float64)[None, :]


def _record_has(record: object, name: str) -> bool:
    if isinstance(record, Mapping):
        return name in record
    return hasattr(record, name)


def _record_value(
    record: object, name: str, *, default: object = None, required: bool = True
) -> object:
    if isinstance(record, Mapping):
        if name in record:
            return record[name]
    elif hasattr(record, name):
        return getattr(record, name)
    if required:
        raise ValueError(f"view record is missing {name}")
    return default


def _sorted_sample(points: np.ndarray, limit: int) -> np.ndarray:
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    ordered = np.asarray(points[order], dtype=np.float64)
    if len(ordered) <= limit:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, limit, dtype=np.int64)
    return ordered[indices]


def _prepare_views(
    view_records: object,
    center: np.ndarray,
    basis: np.ndarray,
    crop_lower: np.ndarray,
    crop_upper: np.ndarray,
    config: Mapping[str, object],
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
        point_name = (
            "mask_depth_points_world"
            if _record_has(record, "mask_depth_points_world")
            else "points_world"
        )
        raw_points = _record_value(record, point_name)
        points_world = np.asarray(raw_points)
        if (
            points_world.ndim != 2
            or points_world.shape[1:] != (3,)
            or not np.issubdtype(points_world.dtype, np.number)
            or not np.isfinite(points_world).all()
        ):
            raise ValueError(
                f"view {point_name} must have finite numeric shape [N, 3]"
            )
        input_count += len(points_world)
        local = (
            np.asarray(points_world, dtype=np.float64) - center[None, :]
        ) @ basis
        in_crop = np.logical_and(
            local >= crop_lower[None, :],
            local <= crop_upper[None, :],
        ).all(axis=1)
        cropped = local[in_crop]
        if len(cropped) < minimum:
            continue

        camera_world = np.asarray(_record_value(record, "camera_position"))
        if (
            camera_world.shape != (3,)
            or not np.issubdtype(camera_world.dtype, np.number)
            or not np.isfinite(camera_world).all()
        ):
            raise ValueError("view camera_position must have finite shape [3]")
        camera = (
            np.asarray(camera_world, dtype=np.float64) - center
        ) @ basis
        quality = _finite_float(
            "view quality",
            _record_value(
                record, "quality", default=1.0, required=False
            ),
        )
        valid_depth = _finite_float(
            "view valid_depth_ratio",
            _record_value(
                record,
                "valid_depth_ratio",
                default=1.0,
                required=False,
            ),
        )
        if quality < 0.0 or not 0.0 <= valid_depth <= 1.0:
            raise ValueError(
                "view quality must be non-negative and valid_depth_ratio "
                "must lie in [0, 1]"
            )
        frame_id = str(
            _record_value(record, "frame_id", default="", required=False)
        )
        sampled = _sorted_sample(
            cropped, int(config["max_points_per_view"])
        )
        eligible_count += 1
        weight = max(quality, 1e-6) * max(valid_depth, 1e-6)
        stable_points = np.concatenate(
            (sampled[0], sampled[-1], np.mean(sampled, axis=0))
        )
        rank_key = (
            -weight,
            frame_id,
            tuple(np.round(camera, 9).tolist()),
            len(sampled),
            tuple(np.round(stable_points, 9).tolist()),
        )
        prepared.append(
            (
                rank_key,
                _View(
                    frame_id=frame_id,
                    points=sampled,
                    camera=camera,
                    weight=float(weight),
                ),
            )
        )
    prepared.sort(key=lambda item: item[0])
    views = tuple(
        item[1] for item in prepared[: int(config["max_views"])]
    )
    return views, input_count, eligible_count


def _voxel_keys(
    points: np.ndarray, origin: np.ndarray, voxel_size: float
) -> np.ndarray:
    return np.floor(
        (points - origin[None, :]) / float(voxel_size) + 1e-10
    ).astype(np.int64)


def _key_tuple(key: np.ndarray) -> Tuple[int, int, int]:
    return tuple(int(value) for value in key)


def _connected_components(keys: np.ndarray) -> Tuple[np.ndarray, ...]:
    voxel_to_points: Dict[Tuple[int, int, int], list] = {}
    for point_index, key in enumerate(keys):
        voxel_to_points.setdefault(_key_tuple(key), []).append(point_index)
    remaining = set(voxel_to_points)
    offsets = tuple(
        offset
        for offset in product((-1, 0, 1), repeat=3)
        if offset != (0, 0, 0)
    )
    components = []
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
        components.append(np.asarray(sorted(indices), dtype=np.int64))
    return tuple(components)


def _dilated_fine_components(
    keys: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    """Return original-point components after one-cell fine-grid dilation.

    Raw depth samples commonly land every second or third fine voxel.  A
    one-cell dilation joins those samples without letting a genuinely empty
    four-cell gap between neighboring objects disappear.  Connectivity of
    the dilated sparse grid itself is still exactly 26-connected.
    """

    unique = sorted({_key_tuple(key) for key in keys})
    expanded = set()
    for key in unique:
        for offset in product((-1, 0, 1), repeat=3):
            expanded.add(
                (
                    key[0] + offset[0],
                    key[1] + offset[1],
                    key[2] + offset[2],
                )
            )
    expanded_array = np.asarray(sorted(expanded), dtype=np.int64)
    expanded_components = _connected_components(expanded_array)
    key_to_component: Dict[Tuple[int, int, int], int] = {}
    for component_index, indices in enumerate(expanded_components):
        for index in indices:
            key_to_component[_key_tuple(expanded_array[index])] = (
                component_index
            )
    grouped: Dict[int, list] = {}
    for point_index, key in enumerate(keys):
        grouped.setdefault(key_to_component[_key_tuple(key)], []).append(
            point_index
        )
    return tuple(
        np.asarray(grouped[index], dtype=np.int64)
        for index in sorted(grouped)
    )


def _inside(
    points: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    return np.logical_and(
        points >= lower[None, :], points <= upper[None, :]
    ).all(axis=1)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    # Values are the primary stable key.  Weight is a deterministic tie-break.
    order = np.lexsort((weights, values))
    sorted_values = values[order]
    sorted_weights = weights[order]
    threshold = 0.5 * float(np.sum(sorted_weights))
    index = int(
        np.searchsorted(np.cumsum(sorted_weights), threshold, side="left")
    )
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _segment_crop_interval(
    start: np.ndarray,
    end: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Optional[Tuple[float, float]]:
    direction = end - start
    t_min = 0.0
    t_max = 1.0
    for axis in range(3):
        if abs(float(direction[axis])) <= 1e-12:
            if start[axis] < lower[axis] or start[axis] > upper[axis]:
                return None
            continue
        first = (lower[axis] - start[axis]) / direction[axis]
        second = (upper[axis] - start[axis]) / direction[axis]
        entering = min(float(first), float(second))
        leaving = max(float(first), float(second))
        t_min = max(t_min, entering)
        t_max = min(t_max, leaving)
        if t_max < t_min:
            return None
    return max(0.0, t_min), min(1.0, t_max)


def _free_voxel_views(
    views: Tuple[_View, ...],
    selected_points: np.ndarray,
    selected_point_views: np.ndarray,
    crop_lower: np.ndarray,
    crop_upper: np.ndarray,
    fine_size: float,
    config: Mapping[str, object],
) -> Dict[Tuple[int, int, int], set]:
    free: Dict[Tuple[int, int, int], set] = {}
    step = fine_size * float(config["empty_ray_step_ratio"])
    margin = fine_size * float(config["empty_ray_surface_margin_ratio"])
    max_samples = int(config["empty_ray_max_samples"])
    for view_index, view in enumerate(views):
        endpoints = selected_points[selected_point_views == view_index]
        if len(endpoints) == 0:
            continue
        endpoint_keys = _voxel_keys(endpoints, crop_lower, fine_size)
        # One stable endpoint per occupied cell is enough to establish a ray,
        # and keeps the work bounded independently of repeated depth samples.
        first_by_key: Dict[Tuple[int, int, int], np.ndarray] = {}
        for point, key in zip(endpoints, endpoint_keys):
            first_by_key.setdefault(_key_tuple(key), point)
        for key in sorted(first_by_key):
            endpoint = first_by_key[key]
            direction = endpoint - view.camera
            length = float(np.linalg.norm(direction))
            if length <= margin + 1e-12:
                continue
            interval = _segment_crop_interval(
                view.camera, endpoint, crop_lower, crop_upper
            )
            if interval is None:
                continue
            start_t, crop_end_t = interval
            end_t = min(crop_end_t, 1.0 - margin / length)
            if end_t <= start_t:
                continue
            count = min(
                max_samples,
                max(2, int(np.ceil((end_t - start_t) * length / step)) + 1),
            )
            samples = view.camera[None, :] + np.linspace(
                start_t, end_t, count, dtype=np.float64
            )[:, None] * direction[None, :]
            keys = _voxel_keys(samples, crop_lower, fine_size)
            for free_key in keys:
                free.setdefault(_key_tuple(free_key), set()).add(view_index)
    return free


def _face_empty_evidence(
    occupied_keys: np.ndarray,
    free_views: Mapping[Tuple[int, int, int], set],
    axis: int,
    face: int,
    visible_views: set,
    shell_voxels: int,
) -> float:
    if len(occupied_keys) == 0 or not visible_views:
        return 0.0
    unique_keys = sorted({_key_tuple(key) for key in occupied_keys})
    extreme = (
        min(key[axis] for key in unique_keys)
        if face == 0
        else max(key[axis] for key in unique_keys)
    )
    boundary = [key for key in unique_keys if key[axis] == extreme]
    direction = -1 if face == 0 else 1
    observed = 0
    total = len(boundary) * shell_voxels
    for key in boundary:
        for distance in range(1, shell_voxels + 1):
            neighbor = list(key)
            neighbor[axis] += direction * distance
            free_for_neighbor = free_views.get(tuple(neighbor), set())
            if visible_views.intersection(free_for_neighbor):
                observed += 1
    return float(observed / max(total, 1))


def _support(
    points: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    if len(points) == 0:
        return 0.0
    return float(np.mean(_inside(points, lower, upper)))


def _axis_residual_scale(
    original_faces: np.ndarray,
    residuals: np.ndarray,
    original_extent: float,
    config: Mapping[str, object],
) -> float:
    minimum_extent = float(config["minimum_extent_ratio"]) * original_extent
    maximum_extent = float(config["maximum_extent_ratio"]) * original_extent
    maximum_center = (
        float(config["maximum_center_shift_ratio"]) * original_extent
    )

    def valid(scale: float) -> bool:
        faces = original_faces + scale * residuals
        extent = float(faces[1] - faces[0])
        center = 0.5 * float(faces[1] + faces[0])
        return (
            minimum_extent <= extent <= maximum_extent
            and abs(center) <= maximum_center
        )

    if valid(1.0):
        return 1.0
    low = 0.0
    high = 1.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        if valid(middle):
            low = middle
        else:
            high = middle
    # Move one tiny step toward the valid identity point so subsequent
    # floating-point reconstruction cannot cross a hard bound.
    return low * (1.0 - 4.0 * np.finfo(np.float64).eps)


def _gate_features(
    *,
    face_residuals: np.ndarray,
    face_support: np.ndarray,
    face_uncertainty: np.ndarray,
    face_empty_evidence: np.ndarray,
    face_supported: np.ndarray,
    face_view_counts: np.ndarray,
    dimensions: np.ndarray,
    selected_view_count: int,
    max_views: int,
    component_view_count: int,
    cropped_point_count: int,
    consensus_point_count: int,
    component_point_count: int,
    fine_occupied_voxel_count: int,
    coarse_occupied_voxel_count: int,
    component_inside_fraction: float,
    extent_ratios: np.ndarray,
    center_shift_ratios: np.ndarray,
) -> np.ndarray:
    scale = dimensions[:, None]
    residual_ratio = face_residuals / scale
    uncertainty_ratio = np.clip(face_uncertainty / scale, 0.0, 1.0)
    absolute_residual = np.abs(residual_ratio)
    face_visibility = np.clip(
        face_view_counts.astype(np.float64)
        / float(max(selected_view_count, 1)),
        0.0,
        1.0,
    )
    globals_ = np.asarray(
        [
            selected_view_count / float(max(max_views, 1)),
            component_view_count / float(max(selected_view_count, 1)),
            consensus_point_count / float(max(cropped_point_count, 1)),
            component_point_count / float(max(consensus_point_count, 1)),
            component_inside_fraction,
            coarse_occupied_voxel_count
            / float(max(component_point_count, 1)),
            fine_occupied_voxel_count / float(max(component_point_count, 1)),
            float(np.mean(face_supported)),
            float(np.mean(absolute_residual)),
            float(np.max(absolute_residual)),
            float(np.mean(uncertainty_ratio)),
            float(np.max(uncertainty_ratio)),
            *extent_ratios.tolist(),
            *center_shift_ratios.tolist(),
        ],
        dtype=np.float64,
    )
    features = np.concatenate(
        (
            residual_ratio.reshape(-1),
            np.clip(face_support, 0.0, 1.0).reshape(-1),
            uncertainty_ratio.reshape(-1),
            np.clip(face_empty_evidence, 0.0, 1.0).reshape(-1),
            face_visibility.reshape(-1),
            globals_,
        )
    )
    return np.asarray(features, dtype=np.float32)


def _empty_values(
    original_raw: np.ndarray,
    center: np.ndarray,
    dimensions: np.ndarray,
    basis: np.ndarray,
    config: Mapping[str, object],
) -> Dict[str, object]:
    original_local_box = np.concatenate((np.zeros(3), dimensions))
    uncertainty = np.repeat(dimensions[:, None], 2, axis=1)
    features = _gate_features(
        face_residuals=np.zeros((3, 2), dtype=np.float64),
        face_support=np.zeros((3, 2), dtype=np.float64),
        face_uncertainty=uncertainty,
        face_empty_evidence=np.zeros((3, 2), dtype=np.float64),
        face_supported=np.zeros((3, 2), dtype=np.bool_),
        face_view_counts=np.zeros((3, 2), dtype=np.int64),
        dimensions=dimensions,
        selected_view_count=0,
        max_views=int(config["max_views"]),
        component_view_count=0,
        cropped_point_count=0,
        consensus_point_count=0,
        component_point_count=0,
        fine_occupied_voxel_count=0,
        coarse_occupied_voxel_count=0,
        component_inside_fraction=0.0,
        extent_ratios=np.ones(3, dtype=np.float64),
        center_shift_ratios=np.zeros(3, dtype=np.float64),
    )
    return {
        "candidate_corners": np.array(original_raw, copy=True),
        "original_corners": np.array(original_raw, copy=True),
        "detail_reasons": (),
        "selected_frame_ids": (),
        "eligible_view_count": 0,
        "selected_view_count": 0,
        "input_point_count": 0,
        "cropped_point_count": 0,
        "consensus_point_count": 0,
        "fine_occupied_voxel_count": 0,
        "coarse_occupied_voxel_count": 0,
        "component_count": 0,
        "component_point_count": 0,
        "component_view_count": 0,
        "component_inside_fraction": 0.0,
        "frame_center": center,
        "frame_basis": basis,
        "original_local_box": original_local_box,
        "candidate_local_box": original_local_box,
        "local_points": np.empty((0, 3), dtype=np.float64),
        "face_residuals": np.zeros((3, 2), dtype=np.float64),
        "face_support": np.zeros((3, 2), dtype=np.float64),
        "face_uncertainty": uncertainty,
        "face_empty_evidence": np.zeros((3, 2), dtype=np.float64),
        "face_supported": np.zeros((3, 2), dtype=np.bool_),
        "face_view_counts": np.zeros((3, 2), dtype=np.int64),
        "face_reasons": ("not_evaluated",) * 6,
        "extent_ratios": np.ones(3, dtype=np.float64),
        "center_shift_ratios": np.zeros(3, dtype=np.float64),
        "original_support": 0.0,
        "candidate_support": 0.0,
        "support_drop": 0.0,
        "gate_features": features,
    }


def _identity(
    original_raw: np.ndarray,
    center: np.ndarray,
    dimensions: np.ndarray,
    basis: np.ndarray,
    config: Mapping[str, object],
    reason: str,
    diagnostics: Optional[Mapping[str, object]] = None,
    detail: Sequence[str] = (),
) -> LocalOccupancyMSRProposal:
    values = _empty_values(original_raw, center, dimensions, basis, config)
    if diagnostics:
        values.update(diagnostics)
    values["candidate_corners"] = np.array(original_raw, copy=True)
    values["candidate_local_box"] = np.concatenate(
        (np.zeros(3), dimensions)
    )
    values["face_residuals"] = np.zeros((3, 2), dtype=np.float64)
    values["extent_ratios"] = np.ones(3, dtype=np.float64)
    values["center_shift_ratios"] = np.zeros(3, dtype=np.float64)
    values["candidate_support"] = values.get("original_support", 0.0)
    values["support_drop"] = 0.0
    values["detail_reasons"] = tuple(detail)
    values["gate_features"] = _gate_features(
        face_residuals=np.asarray(values["face_residuals"]),
        face_support=np.asarray(values["face_support"]),
        face_uncertainty=np.asarray(values["face_uncertainty"]),
        face_empty_evidence=np.asarray(values["face_empty_evidence"]),
        face_supported=np.asarray(values["face_supported"]),
        face_view_counts=np.asarray(values["face_view_counts"]),
        dimensions=dimensions,
        selected_view_count=int(values["selected_view_count"]),
        max_views=int(config["max_views"]),
        component_view_count=int(values["component_view_count"]),
        cropped_point_count=int(values["cropped_point_count"]),
        consensus_point_count=int(values["consensus_point_count"]),
        component_point_count=int(values["component_point_count"]),
        fine_occupied_voxel_count=int(values["fine_occupied_voxel_count"]),
        coarse_occupied_voxel_count=int(
            values["coarse_occupied_voxel_count"]
        ),
        component_inside_fraction=float(
            values["component_inside_fraction"]
        ),
        extent_ratios=np.ones(3, dtype=np.float64),
        center_shift_ratios=np.zeros(3, dtype=np.float64),
    )
    return LocalOccupancyMSRProposal(reason=reason, **values)


def propose_local_occupancy_msr(
    original_corners: object,
    view_records: object,
    config: Optional[Mapping[str, object]] = None,
) -> LocalOccupancyMSRProposal:
    """Propose an orientation-preserving OBB from multi-view mask-depth points.

    ``original_corners`` must use BoxFusion ordering.  View records may be
    mappings or objects.  ``mask_depth_points_world`` is preferred when
    present; otherwise ``points_world`` is used.  ``camera_position`` is
    required, while ``frame_id``, ``quality``, and ``valid_depth_ratio`` have
    deterministic defaults.
    """

    resolved = resolve_local_occupancy_msr_config(config)
    original_raw, center, dimensions, basis = _oriented_frame(
        original_corners
    )
    original_lower = -0.5 * dimensions
    original_upper = 0.5 * dimensions
    original_faces = np.stack((original_lower, original_upper), axis=1)
    crop_extent = dimensions * float(resolved["crop_scale"])
    crop_lower = -0.5 * crop_extent
    crop_upper = 0.5 * crop_extent

    try:
        views, input_count, eligible_count = _prepare_views(
            view_records,
            center,
            basis,
            crop_lower,
            crop_upper,
            resolved,
        )
    except (TypeError, ValueError, FloatingPointError) as error:
        return _identity(
            original_raw,
            center,
            dimensions,
            basis,
            resolved,
            "identity_invalid_view_record",
            detail=(f"{type(error).__name__}: {error}",),
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
            original_raw,
            center,
            dimensions,
            basis,
            resolved,
            "identity_insufficient_views",
            base,
            detail=(
                f"selected_views={len(views)}",
                f"required_views={int(resolved['min_views'])}",
            ),
        )
    total_points = int(sum(len(view.points) for view in views))
    if total_points < int(resolved["min_total_points"]):
        return _identity(
            original_raw,
            center,
            dimensions,
            basis,
            resolved,
            "identity_insufficient_points",
            base,
            detail=(
                f"cropped_points={total_points}",
                f"required_points={int(resolved['min_total_points'])}",
            ),
        )

    try:
        points = np.concatenate(tuple(view.points for view in views), axis=0)
        point_views = np.concatenate(
            tuple(
                np.full(len(view.points), index, dtype=np.int64)
                for index, view in enumerate(views)
            )
        )
        fine_size = float(resolved["fine_voxel_size"])
        fine_keys = _voxel_keys(points, crop_lower, fine_size)
        fine_view_sets: Dict[Tuple[int, int, int], set] = {}
        for key, view_index in zip(fine_keys, point_views):
            fine_view_sets.setdefault(_key_tuple(key), set()).add(
                int(view_index)
            )
        consensus_mask = np.asarray(
            [
                len(fine_view_sets[_key_tuple(key)])
                >= int(resolved["fine_min_view_consensus"])
                for key in fine_keys
            ],
            dtype=np.bool_,
        )
        consensus_points = points[consensus_mask]
        consensus_views = point_views[consensus_mask]
        consensus_fine_keys = fine_keys[consensus_mask]
        fine_voxel_count = len(
            {_key_tuple(key) for key in consensus_fine_keys}
        )
        diagnostics = {
            **base,
            "consensus_point_count": len(consensus_points),
            "fine_occupied_voxel_count": fine_voxel_count,
        }
        if len(consensus_points) < int(resolved["min_total_points"]):
            return _identity(
                original_raw,
                center,
                dimensions,
                basis,
                resolved,
                "identity_insufficient_consensus_points",
                diagnostics,
                detail=(
                    f"consensus_points={len(consensus_points)}",
                    f"required_points={int(resolved['min_total_points'])}",
                ),
            )

        # First split leakage at fine scale.  One-cell dilation tolerates
        # ordinary depth sampling gaps while preserving a genuinely empty
        # gap between nearby objects.
        fine_components = _dilated_fine_components(consensus_fine_keys)
        eligible_fine_components = []
        for stable_index, indices in enumerate(fine_components):
            component_points = consensus_points[indices]
            component_views = consensus_views[indices]
            inside_count = int(
                np.sum(
                    _inside(
                        component_points, original_lower, original_upper
                    )
                )
            )
            inside_fraction = inside_count / float(len(indices))
            view_count = len(set(component_views.tolist()))
            if (
                len(indices) >= int(resolved["min_component_points"])
                and view_count >= int(resolved["min_component_views"])
                and inside_fraction
                >= float(resolved["min_component_inside_fraction"])
            ):
                normalized_center = np.median(
                    component_points / dimensions[None, :], axis=0
                )
                center_distance = float(np.linalg.norm(normalized_center))
                eligible_fine_components.append(
                    (
                        (
                            -inside_count,
                            -view_count,
                            center_distance,
                            -len(indices),
                            stable_index,
                        ),
                        indices,
                        inside_fraction,
                        view_count,
                    )
                )
        diagnostics["component_count"] = len(fine_components)
        if not eligible_fine_components:
            return _identity(
                original_raw,
                center,
                dimensions,
                basis,
                resolved,
                "identity_no_anchor_component",
                diagnostics,
                detail=(
                    f"fine_components={len(fine_components)}",
                    "no fine component met point/view/inside requirements",
                ),
            )
        eligible_fine_components.sort(key=lambda item: item[0])
        _, fine_indices, _, _ = eligible_fine_components[0]
        fine_anchor_points = consensus_points[fine_indices]
        fine_anchor_views = consensus_views[fine_indices]
        fine_anchor_keys = consensus_fine_keys[fine_indices]

        # The coarse response supplies stable surfaces and performs a second
        # exact 26-connected check, but it cannot reconnect an object rejected
        # by the fine-scale empty gap above.
        coarse_size = float(resolved["coarse_voxel_size"])
        coarse_keys = _voxel_keys(
            fine_anchor_points, crop_lower, coarse_size
        )
        coarse_components = _connected_components(coarse_keys)
        eligible_coarse_components = []
        for stable_index, indices in enumerate(coarse_components):
            component_points = fine_anchor_points[indices]
            component_views = fine_anchor_views[indices]
            inside_count = int(
                np.sum(
                    _inside(
                        component_points, original_lower, original_upper
                    )
                )
            )
            inside_fraction = inside_count / float(len(indices))
            view_count = len(set(component_views.tolist()))
            if (
                len(indices) >= int(resolved["min_component_points"])
                and view_count >= int(resolved["min_component_views"])
                and inside_fraction
                >= float(resolved["min_component_inside_fraction"])
            ):
                normalized_center = np.median(
                    component_points / dimensions[None, :], axis=0
                )
                center_distance = float(np.linalg.norm(normalized_center))
                eligible_coarse_components.append(
                    (
                        (
                            -inside_count,
                            -view_count,
                            center_distance,
                            -len(indices),
                            stable_index,
                        ),
                        indices,
                        inside_fraction,
                        view_count,
                    )
                )
        if not eligible_coarse_components:
            return _identity(
                original_raw,
                center,
                dimensions,
                basis,
                resolved,
                "identity_no_anchor_component",
                diagnostics,
                detail=(
                    f"fine_components={len(fine_components)}",
                    f"coarse_components={len(coarse_components)}",
                    "fine anchor had no eligible coarse component",
                ),
            )
        eligible_coarse_components.sort(key=lambda item: item[0])
        _, selected_indices, inside_fraction, component_view_count = (
            eligible_coarse_components[0]
        )
        selected_points = fine_anchor_points[selected_indices]
        selected_point_views = fine_anchor_views[selected_indices]
        selected_fine_keys = fine_anchor_keys[selected_indices]
        selected_coarse_keys = coarse_keys[selected_indices]
        order = np.lexsort(
            (
                selected_point_views,
                selected_points[:, 2],
                selected_points[:, 1],
                selected_points[:, 0],
            )
        )
        selected_points = selected_points[order]
        selected_point_views = selected_point_views[order]
        selected_fine_keys = selected_fine_keys[order]
        selected_coarse_keys = selected_coarse_keys[order]
        fine_voxel_count = len(
            {_key_tuple(key) for key in selected_fine_keys}
        )
        coarse_voxel_count = len(
            {_key_tuple(key) for key in selected_coarse_keys}
        )
        diagnostics.update(
            {
                "fine_occupied_voxel_count": fine_voxel_count,
                "coarse_occupied_voxel_count": coarse_voxel_count,
                "component_point_count": len(selected_points),
                "component_view_count": component_view_count,
                "component_inside_fraction": inside_fraction,
                "local_points": selected_points,
            }
        )

        free_views = _free_voxel_views(
            views,
            selected_points,
            selected_point_views,
            crop_lower,
            crop_upper,
            fine_size,
            resolved,
        )

        face_residuals = np.zeros((3, 2), dtype=np.float64)
        face_support = np.zeros((3, 2), dtype=np.float64)
        face_uncertainty = np.repeat(dimensions[:, None], 2, axis=1)
        face_empty = np.zeros((3, 2), dtype=np.float64)
        face_supported = np.zeros((3, 2), dtype=np.bool_)
        face_view_counts = np.zeros((3, 2), dtype=np.int64)
        face_reasons = ["not_evaluated"] * 6
        low_quantile = float(resolved["lower_quantile"])
        high_quantile = float(resolved["upper_quantile"])

        unique_coarse = np.asarray(
            sorted({_key_tuple(key) for key in selected_coarse_keys}),
            dtype=np.float64,
        )
        coarse_centers = (
            crop_lower[None, :]
            + (unique_coarse + 0.5) * coarse_size
        )
        for axis in range(3):
            for face in range(2):
                flat_index = axis * 2 + face
                visible_view_indices = []
                estimates = []
                weights = []
                for view_index, view in enumerate(views):
                    visible = (
                        view.camera[axis] < original_lower[axis]
                        if face == 0
                        else view.camera[axis] > original_upper[axis]
                    )
                    if not visible:
                        continue
                    view_points = selected_points[
                        selected_point_views == view_index
                    ]
                    if len(view_points) < int(
                        resolved["face_min_points_per_view"]
                    ):
                        continue
                    quantile = low_quantile if face == 0 else high_quantile
                    estimates.append(
                        float(
                            np.quantile(
                                view_points[:, axis],
                                quantile,
                                method="linear",
                            )
                        )
                    )
                    weights.append(view.weight)
                    visible_view_indices.append(view_index)
                face_view_counts[axis, face] = len(estimates)
                if len(estimates) < int(resolved["face_min_views"]):
                    face_reasons[flat_index] = (
                        "unsupported_insufficient_face_views"
                    )
                    continue

                estimate_values = np.asarray(estimates, dtype=np.float64)
                estimate_weights = np.asarray(weights, dtype=np.float64)
                fine_estimate = _weighted_median(
                    estimate_values, estimate_weights
                )
                coarse_estimate = (
                    float(np.min(coarse_centers[:, axis]) - 0.5 * coarse_size)
                    if face == 0
                    else float(
                        np.max(coarse_centers[:, axis]) + 0.5 * coarse_size
                    )
                )
                fine_weight = float(resolved["msr_fine_weight"])
                msr_estimate = (
                    fine_weight * fine_estimate
                    + (1.0 - fine_weight) * coarse_estimate
                )
                msr_estimate += (
                    -float(resolved["boundary_padding"])
                    if face == 0
                    else float(resolved["boundary_padding"])
                )
                spread = float(
                    max(
                        np.max(estimate_values) - np.min(estimate_values),
                        0.5 * abs(coarse_estimate - fine_estimate),
                        0.5 * fine_size,
                    )
                )
                face_uncertainty[axis, face] = spread
                uncertainty_ratio = spread / dimensions[axis]

                visible_set = set(visible_view_indices)
                empty_score = _face_empty_evidence(
                    selected_fine_keys,
                    free_views,
                    axis,
                    face,
                    visible_set,
                    int(resolved["empty_shell_voxels"]),
                )
                face_empty[axis, face] = empty_score
                view_consensus = min(
                    1.0,
                    len(estimates)
                    / float(max(int(resolved["face_min_views"]), 1)),
                )
                stability = max(
                    0.0,
                    1.0
                    - uncertainty_ratio
                    / max(
                        float(resolved["face_max_uncertainty_ratio"]),
                        1e-12,
                    ),
                )
                support = (
                    view_consensus
                    * stability
                    * (0.5 + 0.5 * empty_score)
                )
                face_support[axis, face] = support
                if uncertainty_ratio > float(
                    resolved["face_max_uncertainty_ratio"]
                ):
                    face_reasons[flat_index] = (
                        "unsupported_high_face_uncertainty"
                    )
                    continue
                if support < float(resolved["face_min_support"]):
                    face_reasons[flat_index] = "unsupported_low_face_support"
                    continue
                if empty_score < float(
                    resolved["face_min_empty_evidence"]
                ):
                    face_reasons[flat_index] = (
                        "unsupported_no_adjacent_empty_space"
                    )
                    continue

                original_face = original_faces[axis, face]
                residual = float(resolved["boundary_blend"]) * (
                    msr_estimate - original_face
                )
                maximum_shift = (
                    float(resolved["maximum_face_shift_ratio"])
                    * dimensions[axis]
                )
                residual = float(
                    np.clip(residual, -maximum_shift, maximum_shift)
                )
                proposed = float(
                    np.clip(
                        original_face + residual,
                        crop_lower[axis],
                        crop_upper[axis],
                    )
                )
                face_residuals[axis, face] = proposed - original_face
                face_supported[axis, face] = True
                face_reasons[flat_index] = "supported"

        diagnostics.update(
            {
                "face_support": face_support,
                "face_uncertainty": face_uncertainty,
                "face_empty_evidence": face_empty,
                "face_supported": face_supported,
                "face_view_counts": face_view_counts,
                "face_reasons": tuple(face_reasons),
            }
        )
        if not bool(np.any(face_supported)):
            return _identity(
                original_raw,
                center,
                dimensions,
                basis,
                resolved,
                "identity_no_supported_faces",
                diagnostics,
                detail=tuple(
                    f"{name}:{face_reasons[index]}"
                    for index, name in enumerate(_FACE_NAMES)
                ),
            )

        clamped_axes = []
        for axis in range(3):
            scale = _axis_residual_scale(
                original_faces[axis],
                face_residuals[axis],
                dimensions[axis],
                resolved,
            )
            if scale < 1.0 - 1e-12:
                clamped_axes.append("xyz"[axis])
                face_residuals[axis] *= scale

        candidate_faces = original_faces + face_residuals
        original_support = _support(
            selected_points, original_lower, original_upper
        )
        candidate_support = _support(
            selected_points,
            candidate_faces[:, 0],
            candidate_faces[:, 1],
        )
        support_drop = max(0.0, original_support - candidate_support)
        support_clamped = False
        maximum_drop = float(resolved["maximum_support_drop"])
        if support_drop > maximum_drop:
            # Point support is monotonic enough for a deterministic safety
            # bisection.  Crucially, this scales only already-supported faces.
            low = 0.0
            high = 1.0
            for _ in range(60):
                middle = 0.5 * (low + high)
                trial_faces = original_faces + middle * face_residuals
                trial_support = _support(
                    selected_points,
                    trial_faces[:, 0],
                    trial_faces[:, 1],
                )
                trial_drop = max(0.0, original_support - trial_support)
                if trial_drop <= maximum_drop + 1e-12:
                    low = middle
                else:
                    high = middle
            face_residuals *= low * (
                1.0 - 4.0 * np.finfo(np.float64).eps
            )
            candidate_faces = original_faces + face_residuals
            candidate_support = _support(
                selected_points,
                candidate_faces[:, 0],
                candidate_faces[:, 1],
            )
            support_drop = max(0.0, original_support - candidate_support)
            support_clamped = True

        if float(np.max(np.abs(face_residuals))) <= float(
            resolved["minimum_face_residual"]
        ):
            diagnostics.update(
                {
                    "original_support": original_support,
                    "candidate_support": original_support,
                    "support_drop": 0.0,
                }
            )
            return _identity(
                original_raw,
                center,
                dimensions,
                basis,
                resolved,
                "identity_no_effect",
                diagnostics,
                detail=("all supported residuals were clamped to zero",),
            )

        candidate_extent = (
            candidate_faces[:, 1] - candidate_faces[:, 0]
        )
        candidate_center = 0.5 * (
            candidate_faces[:, 1] + candidate_faces[:, 0]
        )
        extent_ratios = candidate_extent / dimensions
        center_shift_ratios = np.abs(candidate_center) / dimensions
        candidate_local_box = np.concatenate(
            (candidate_center, candidate_extent)
        )
        candidate_float = _local_box_to_world(
            candidate_local_box, center, basis
        )
        output_dtype = (
            original_raw.dtype
            if np.issubdtype(original_raw.dtype, np.floating)
            else np.dtype(np.float64)
        )
        candidate_corners = np.asarray(candidate_float, dtype=output_dtype)
        details = [
            f"supported_faces={int(np.sum(face_supported))}",
            f"changed_faces={int(np.sum(np.abs(face_residuals) > 0.0))}",
        ]
        if clamped_axes:
            details.append("axis_clamped=" + ",".join(clamped_axes))
        if support_clamped:
            details.append("point_support_clamped")

        gate_features = _gate_features(
            face_residuals=face_residuals,
            face_support=face_support,
            face_uncertainty=face_uncertainty,
            face_empty_evidence=face_empty,
            face_supported=face_supported,
            face_view_counts=face_view_counts,
            dimensions=dimensions,
            selected_view_count=len(views),
            max_views=int(resolved["max_views"]),
            component_view_count=component_view_count,
            cropped_point_count=total_points,
            consensus_point_count=len(consensus_points),
            component_point_count=len(selected_points),
            fine_occupied_voxel_count=fine_voxel_count,
            coarse_occupied_voxel_count=coarse_voxel_count,
            component_inside_fraction=inside_fraction,
            extent_ratios=extent_ratios,
            center_shift_ratios=center_shift_ratios,
        )
        return LocalOccupancyMSRProposal(
            candidate_corners=candidate_corners,
            original_corners=np.array(original_raw, copy=True),
            reason="candidate",
            detail_reasons=tuple(details),
            selected_frame_ids=tuple(view.frame_id for view in views),
            eligible_view_count=eligible_count,
            selected_view_count=len(views),
            input_point_count=input_count,
            cropped_point_count=total_points,
            consensus_point_count=len(consensus_points),
            fine_occupied_voxel_count=fine_voxel_count,
            coarse_occupied_voxel_count=coarse_voxel_count,
            component_count=len(fine_components),
            component_point_count=len(selected_points),
            component_view_count=component_view_count,
            component_inside_fraction=inside_fraction,
            frame_center=center,
            frame_basis=basis,
            original_local_box=np.concatenate((np.zeros(3), dimensions)),
            candidate_local_box=candidate_local_box,
            local_points=selected_points,
            face_residuals=face_residuals,
            face_support=face_support,
            face_uncertainty=face_uncertainty,
            face_empty_evidence=face_empty,
            face_supported=face_supported,
            face_view_counts=face_view_counts,
            face_reasons=tuple(face_reasons),
            extent_ratios=extent_ratios,
            center_shift_ratios=center_shift_ratios,
            original_support=original_support,
            candidate_support=candidate_support,
            support_drop=support_drop,
            gate_features=gate_features,
        )
    except (TypeError, ValueError, FloatingPointError, OverflowError) as error:
        return _identity(
            original_raw,
            center,
            dimensions,
            basis,
            resolved,
            "identity_numerical_failure",
            base,
            detail=(f"{type(error).__name__}: {error}",),
        )


def propose_occupancy_msr_refinement(
    original_corners: object,
    view_records: object,
    config: Optional[Mapping[str, object]] = None,
) -> LocalOccupancyMSRProposal:
    """Compatibility spelling for :func:`propose_local_occupancy_msr`."""

    return propose_local_occupancy_msr(
        original_corners, view_records, config
    )


def propose_occupancy_msr(
    original_corners: object,
    view_records: object,
    config: Optional[Mapping[str, object]] = None,
) -> LocalOccupancyMSRProposal:
    """Short compatibility spelling for the standalone proposal."""

    return propose_local_occupancy_msr(
        original_corners, view_records, config
    )


__all__ = [
    "DEFAULT_LOCAL_OCCUPANCY_MSR_CONFIG",
    "LOCAL_OCCUPANCY_MSR_GATE_FEATURE_DIM",
    "LOCAL_OCCUPANCY_MSR_GATE_FEATURE_NAMES",
    "LOCAL_OCCUPANCY_MSR_SOURCE",
    "OCCUPANCY_MSR_FEATURE_DIM",
    "OCCUPANCY_MSR_FEATURE_NAMES",
    "LocalOccupancyMSRProposal",
    "OccupancyMSRProposal",
    "propose_local_occupancy_msr",
    "propose_occupancy_msr",
    "propose_occupancy_msr_refinement",
    "resolve_local_occupancy_msr_config",
]
