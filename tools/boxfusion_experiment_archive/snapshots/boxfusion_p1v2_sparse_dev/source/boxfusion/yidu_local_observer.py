"""Observer-only local geometry route assembled from the YiDu survey.

This module composes already isolated evidence into strictly cumulative
diagnostic stages:

``A1/A2``
    Consume the secondary Mask-RGBD memory (adaptive erosion and DFU-style
    filtering happen during backprojection).
``A3``
    Build deterministic local geometric voxel components and select the
    multi-view component anchored by the original box.
``A4``
    Add the orientation-preserving occupancy/MSR candidate.
``A5``
    Compare original/raw-mask/component/occupancy candidates through a
    raw/fused query table.
``A6``
    Evaluate an optional train-only AP50 safety gate.

No function in this file accepts a destination output array, and every result
explicitly reports ``mutation_enabled=False`` and ``applied=False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from boxfusion.ap50_safety_gate import (
    AP50SafetyDecision,
    AP50SafetyGate,
    AP50SafetyGateConfig,
)
from boxfusion.local_occupancy_msr_refiner import (
    DEFAULT_LOCAL_OCCUPANCY_MSR_CONFIG,
    OCCUPANCY_MSR_FEATURE_DIM,
    OCCUPANCY_MSR_FEATURE_NAMES,
    LocalOccupancyMSRProposal,
    propose_local_occupancy_msr,
)
from boxfusion.quality_score import (
    QUALITY_FEATURE_DIM,
    QUALITY_FEATURE_NAMES,
)
from boxfusion.raw_fused_query import (
    RAW_FUSED_INPUT_QUALITY_NAMES,
    RAW_FUSED_QUERY_FEATURE_DIM,
    RAW_FUSED_QUERY_FEATURE_NAMES,
    RawFusedQueryObservation,
    observe_raw_fused_query,
)
from boxfusion.voxel_components import (
    VoxelComponent,
    VoxelComponentSet,
    build_voxel_components,
    select_inside_anchor,
)
from boxfusion.yidu_ablation import YIDU_STAGES, resolve_yidu_stage


YIDU_LOCAL_OBSERVER_SCHEMA = "boxfusion.yidu.local_observer.v1"
YIDU_COMPONENT_FEATURE_NAMES = (
    "input_points",
    "cropped_points",
    "component_count",
    "selected_points",
    "selected_voxels",
    "selected_views",
    "selected_point_fraction",
    "selected_inside_fraction",
    "selected_density_log1p",
    "candidate_volume_ratio",
    "candidate_center_shift_ratio",
    "candidate_extent_l1_ratio",
)
YIDU_COMPONENT_FEATURE_DIM = len(YIDU_COMPONENT_FEATURE_NAMES)
assert YIDU_COMPONENT_FEATURE_DIM == 12

YIDU_GATE_FEATURE_NAMES = (
    tuple(f"b6_original_{name}" for name in QUALITY_FEATURE_NAMES)
    + ("occupancy_features_available",)
    + tuple(f"occupancy_msr_{name}" for name in OCCUPANCY_MSR_FEATURE_NAMES)
    + tuple(
        f"raw_fused_selected_{name}"
        for name in RAW_FUSED_QUERY_FEATURE_NAMES
    )
)
YIDU_GATE_FEATURE_DIM = len(YIDU_GATE_FEATURE_NAMES)
assert YIDU_GATE_FEATURE_DIM == (
    QUALITY_FEATURE_DIM
    + 1
    + OCCUPANCY_MSR_FEATURE_DIM
    + RAW_FUSED_QUERY_FEATURE_DIM
)

DEFAULT_YIDU_LOCAL_OBSERVER_CONFIG = {
    "crop_scale": 1.35,
    "max_views": 5,
    "max_points_per_view": 768,
    "voxel_size": 0.04,
    "boundary_epsilon": 1e-8,
    "neighbor_radius": 1,
    "dilation_radius": 0,
    "min_points_per_voxel": 1,
    "minimum_component_points": 64,
    "minimum_component_voxels": 8,
    "minimum_component_views": 2,
    "minimum_inside_points": 16,
    "minimum_inside_fraction": 0.20,
    "lower_quantile": 0.02,
    "upper_quantile": 0.98,
    "minimum_dimension": 0.02,
    "occupancy_msr": dict(DEFAULT_LOCAL_OCCUPANCY_MSR_CONFIG),
    "raw_fused_scorer_checkpoint": None,
    "quality_gate": {
        "minimum_improvement_probability": 0.75,
        "maximum_harm_probability": 0.15,
        "uncertainty_multiplier": 1.64,
        "maximum_delta_std": 0.15,
        "minimum_delta_lower_bound": 0.005,
        "minimum_predicted_iou_margin": 0.005,
        "require_iou50_crossing": False,
        "minimum_iou50_crossing_probability": 0.60,
    },
}


def _readonly(
    value: object,
    *,
    dtype: np.dtype,
    shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError(f"array must have shape {shape}, got {array.shape}")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _record_value(
    record: object, name: str, default: object = None
) -> object:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _finite_scalar(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def resolve_yidu_local_observer_config(
    config: Optional[Mapping[str, object]] = None,
) -> dict:
    """Resolve a strict local-observer config without mutating its input."""

    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise TypeError("YiDu local-observer config must be a mapping")
    unknown = sorted(set(config) - set(DEFAULT_YIDU_LOCAL_OBSERVER_CONFIG))
    if unknown:
        raise ValueError(
            "Unknown YiDu local-observer key(s): " + ", ".join(unknown)
        )
    resolved = {
        key: (
            dict(value)
            if isinstance(value, Mapping)
            else value
        )
        for key, value in DEFAULT_YIDU_LOCAL_OBSERVER_CONFIG.items()
    }
    for key, value in config.items():
        if key in {"occupancy_msr", "quality_gate"}:
            if not isinstance(value, Mapping):
                raise TypeError(f"YiDu {key} must be a mapping")
            resolved[key].update(value)
        else:
            resolved[key] = value

    for key in (
        "crop_scale",
        "voxel_size",
        "boundary_epsilon",
        "minimum_inside_fraction",
        "lower_quantile",
        "upper_quantile",
        "minimum_dimension",
    ):
        resolved[key] = _finite_scalar(key, resolved[key])
    if resolved["crop_scale"] < 1.0:
        raise ValueError("crop_scale must be at least 1")
    if resolved["voxel_size"] <= 0.0:
        raise ValueError("voxel_size must be positive")
    if not 0.0 <= resolved["boundary_epsilon"] < 0.5:
        raise ValueError("boundary_epsilon must lie in [0, 0.5)")
    if not 0.0 <= resolved["minimum_inside_fraction"] <= 1.0:
        raise ValueError("minimum_inside_fraction must lie in [0, 1]")
    if not (
        0.0 <= resolved["lower_quantile"]
        < resolved["upper_quantile"]
        <= 1.0
    ):
        raise ValueError("YiDu quantiles must satisfy 0 <= lower < upper <= 1")
    if resolved["minimum_dimension"] <= 0.0:
        raise ValueError("minimum_dimension must be positive")
    for key, minimum in (
        ("max_views", 1),
        ("max_points_per_view", 1),
        ("neighbor_radius", 1),
        ("dilation_radius", 0),
        ("min_points_per_voxel", 1),
        ("minimum_component_points", 1),
        ("minimum_component_voxels", 1),
        ("minimum_component_views", 1),
        ("minimum_inside_points", 1),
    ):
        value = resolved[key]
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < minimum
        ):
            raise ValueError(f"{key} must be an integer >= {minimum}")
        resolved[key] = int(value)
    checkpoint = resolved["raw_fused_scorer_checkpoint"]
    if checkpoint is not None and (
        not isinstance(checkpoint, str) or not checkpoint.strip()
    ):
        raise ValueError(
            "raw_fused_scorer_checkpoint must be a non-empty path or None"
        )
    return resolved


def _aabb_from_corners(corners: object) -> Tuple[np.ndarray, np.ndarray]:
    array = np.asarray(corners)
    if (
        array.shape != (8, 3)
        or not np.issubdtype(array.dtype, np.number)
        or not np.isfinite(array).all()
    ):
        raise ValueError("corners must have finite numeric shape [8, 3]")
    lower = np.min(array.astype(np.float64), axis=0)
    upper = np.max(array.astype(np.float64), axis=0)
    dimensions = upper - lower
    if np.any(dimensions <= 0.0):
        raise ValueError("corners must span positive dimensions")
    return 0.5 * (lower + upper), dimensions


def _aabb_corners(center: np.ndarray, dimensions: np.ndarray) -> np.ndarray:
    lower = center - 0.5 * dimensions
    upper = center + 0.5 * dimensions
    return np.asarray(
        [
            [lower[0], lower[1], lower[2]],
            [upper[0], lower[1], lower[2]],
            [upper[0], upper[1], lower[2]],
            [lower[0], upper[1], lower[2]],
            [lower[0], lower[1], upper[2]],
            [upper[0], lower[1], upper[2]],
            [upper[0], upper[1], upper[2]],
            [lower[0], upper[1], upper[2]],
        ],
        dtype=np.float32,
    )


def _sorted_sample(points: np.ndarray, limit: int) -> np.ndarray:
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    sorted_points = np.asarray(points[order], dtype=np.float64)
    if len(sorted_points) <= limit:
        return sorted_points
    positions = np.linspace(
        0, len(sorted_points) - 1, limit, dtype=np.int64
    )
    return sorted_points[positions]


def _prepare_component_points(
    records: Sequence[object],
    *,
    center: np.ndarray,
    dimensions: np.ndarray,
    config: Mapping[str, object],
) -> Tuple[np.ndarray, np.ndarray, int, int, float, float]:
    crop_dimensions = dimensions * float(config["crop_scale"])
    lower = center - 0.5 * crop_dimensions
    upper = center + 0.5 * crop_dimensions
    ranked = []
    input_count = 0
    projection_values = []
    depth_values = []
    for source_index, record in enumerate(records):
        points = np.asarray(_record_value(record, "points_world"))
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or not np.issubdtype(points.dtype, np.number)
            or not np.isfinite(points).all()
        ):
            raise ValueError("view points_world must have finite shape [N,3]")
        points = np.asarray(points, dtype=np.float64)
        input_count += len(points)
        inside = np.logical_and(
            points >= lower[None, :], points <= upper[None, :]
        ).all(axis=1)
        cropped = points[inside]
        if not len(cropped):
            continue
        quality = _finite_scalar(
            "view quality", _record_value(record, "quality", 1.0)
        )
        valid_depth = _finite_scalar(
            "view valid_depth_ratio",
            _record_value(record, "valid_depth_ratio", 1.0),
        )
        projection = _finite_scalar(
            "view projection_mask_iou",
            _record_value(record, "projection_mask_iou", 1.0),
        )
        if (
            quality < 0.0
            or not 0.0 <= valid_depth <= 1.0
            or not 0.0 <= projection <= 1.0
        ):
            raise ValueError("invalid view quality metadata")
        sampled = _sorted_sample(
            cropped, int(config["max_points_per_view"])
        )
        frame_id = str(_record_value(record, "frame_id", source_index))
        ranked.append(
            (
                -quality * max(valid_depth, 1e-6),
                frame_id,
                source_index,
                sampled,
                valid_depth,
                projection,
            )
        )
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    selected = ranked[: int(config["max_views"])]
    if not selected:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.int64),
            input_count,
            0,
            0.0,
            0.0,
        )
    points = np.concatenate([item[3] for item in selected], axis=0)
    view_ids = np.concatenate(
        [
            np.full(len(item[3]), index, dtype=np.int64)
            for index, item in enumerate(selected)
        ]
    )
    depth_values.extend(float(item[4]) for item in selected)
    projection_values.extend(float(item[5]) for item in selected)
    return (
        points,
        view_ids,
        input_count,
        len(selected),
        float(np.mean(depth_values)),
        float(np.mean(projection_values)),
    )


def _component_candidate(
    component: VoxelComponent,
    *,
    original_center: np.ndarray,
    original_dimensions: np.ndarray,
    config: Mapping[str, object],
) -> Tuple[np.ndarray, np.ndarray]:
    bounds = np.quantile(
        component.points,
        (float(config["lower_quantile"]), float(config["upper_quantile"])),
        axis=0,
    )
    center = 0.5 * (bounds[0] + bounds[1])
    dimensions = np.maximum(
        bounds[1] - bounds[0], float(config["minimum_dimension"])
    )
    candidate = _aabb_corners(center, dimensions)
    diagonal = max(float(np.linalg.norm(original_dimensions)), 1e-6)
    volume_ratio = float(
        np.prod(dimensions) / max(np.prod(original_dimensions), 1e-9)
    )
    center_shift = float(
        np.linalg.norm(center - original_center) / diagonal
    )
    extent_l1 = float(
        np.mean(
            np.abs(
                dimensions / np.maximum(original_dimensions, 1e-6) - 1.0
            )
        )
    )
    lower = original_center - 0.5 * original_dimensions
    upper = original_center + 0.5 * original_dimensions
    inside = np.logical_and(
        component.points >= lower[None, :],
        component.points <= upper[None, :],
    ).all(axis=1)
    inside_fraction = float(np.mean(inside)) if len(inside) else 0.0
    features = np.asarray(
        [
            0.0,  # filled by caller
            0.0,  # filled by caller
            0.0,  # filled by caller
            component.point_count,
            component.voxel_count,
            component.view_count,
            component.point_fraction,
            inside_fraction,
            np.log1p(
                min(
                    float(component.density)
                    if np.isfinite(component.density)
                    else 1.0e6,
                    1.0e6,
                )
            ),
            volume_ratio,
            center_shift,
            extent_l1,
        ],
        dtype=np.float32,
    )
    return candidate, features


def _quality_row(
    proposal_score: float,
    mask_quality: float,
    depth_quality: float,
    geometry_quality: float,
    view_quality: float,
) -> np.ndarray:
    return np.clip(
        np.asarray(
            [
                proposal_score,
                mask_quality,
                depth_quality,
                geometry_quality,
                view_quality,
            ],
            dtype=np.float32,
        ),
        0.0,
        1.0,
    )


@dataclass(frozen=True)
class YiDuLocalObservation:
    """Immutable observer output.  It is not an exportable detection result."""

    stage: str
    reason: str
    original_corners: np.ndarray
    raw_candidate_corners: np.ndarray
    superpoint_candidate_corners: np.ndarray
    occupancy_candidate_corners: np.ndarray
    selected_candidate_corners: np.ndarray
    selected_source: str
    input_point_count: int
    cropped_point_count: int
    selected_view_count: int
    component_set: Optional[VoxelComponentSet]
    selected_component_id: int
    component_features: np.ndarray
    occupancy_proposal: Optional[LocalOccupancyMSRProposal]
    occupancy_features: np.ndarray
    raw_fused_observation: Optional[RawFusedQueryObservation]
    raw_fused_selected_features: np.ndarray
    gate_features: np.ndarray
    gate_decision: Optional[AP50SafetyDecision]
    mutation_enabled: bool = False
    applied: bool = False

    def __post_init__(self) -> None:
        stage = resolve_yidu_stage(self.stage)
        object.__setattr__(self, "stage", stage)
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be non-empty")
        for name in (
            "original_corners",
            "raw_candidate_corners",
            "superpoint_candidate_corners",
            "occupancy_candidate_corners",
            "selected_candidate_corners",
        ):
            object.__setattr__(
                self,
                name,
                _readonly(getattr(self, name), dtype=np.float32, shape=(8, 3)),
            )
        for name in (
            "input_point_count",
            "cropped_point_count",
            "selected_view_count",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        component_id = int(self.selected_component_id)
        if component_id < -1:
            raise ValueError("selected_component_id must be >= -1")
        object.__setattr__(self, "selected_component_id", component_id)
        object.__setattr__(
            self,
            "component_features",
            _readonly(
                self.component_features,
                dtype=np.float32,
                shape=(YIDU_COMPONENT_FEATURE_DIM,),
            ),
        )
        object.__setattr__(
            self,
            "occupancy_features",
            _readonly(
                self.occupancy_features,
                dtype=np.float32,
                shape=(OCCUPANCY_MSR_FEATURE_DIM,),
            ),
        )
        object.__setattr__(
            self,
            "raw_fused_selected_features",
            _readonly(
                self.raw_fused_selected_features,
                dtype=np.float32,
                shape=(RAW_FUSED_QUERY_FEATURE_DIM,),
            ),
        )
        object.__setattr__(
            self,
            "gate_features",
            _readonly(
                self.gate_features,
                dtype=np.float32,
                shape=(YIDU_GATE_FEATURE_DIM,),
            ),
        )
        if self.mutation_enabled or self.applied:
            raise ValueError("YiDu released route is observer-only")
        object.__setattr__(self, "mutation_enabled", False)
        object.__setattr__(self, "applied", False)


def observe_yidu_local_geometry(
    *,
    stage: str,
    original_corners: object,
    view_records: Sequence[object],
    detector_score: float,
    b6_quality_features: object,
    raw_candidate_corners: Optional[object] = None,
    raw_candidate_verified: bool = False,
    config: Optional[Mapping[str, object]] = None,
    quality_gate: Optional[AP50SafetyGate] = None,
) -> YiDuLocalObservation:
    """Run one cumulative YiDu observer stage without mutating B6 output."""

    canonical_stage = resolve_yidu_stage(stage)
    if canonical_stage == "B0":
        raise ValueError("B0 has no YiDu observer execution")
    stage_index = YIDU_STAGES.index(canonical_stage)
    resolved = resolve_yidu_local_observer_config(config)
    original = np.asarray(original_corners, dtype=np.float32)
    original_center, original_dimensions = _aabb_from_corners(original)
    score = _finite_scalar("detector_score", detector_score)
    if not 0.0 <= score <= 1.0:
        raise ValueError("detector_score must lie in [0, 1]")
    quality = np.asarray(b6_quality_features, dtype=np.float32)
    if (
        quality.shape != (QUALITY_FEATURE_DIM,)
        or not np.isfinite(quality).all()
    ):
        raise ValueError(
            f"b6_quality_features must be finite [{QUALITY_FEATURE_DIM}]"
        )
    raw = original.copy()
    if raw_candidate_corners is not None:
        candidate = np.asarray(raw_candidate_corners, dtype=np.float32)
        if candidate.shape != (8, 3) or not np.isfinite(candidate).all():
            raise ValueError("raw_candidate_corners must be finite [8,3]")
        if raw_candidate_verified:
            raw = candidate.copy()

    (
        points,
        point_views,
        input_count,
        selected_views,
        mean_depth,
        mean_projection,
    ) = _prepare_component_points(
        view_records,
        center=original_center,
        dimensions=original_dimensions,
        config=resolved,
    )
    component_set: Optional[VoxelComponentSet] = None
    selected_component: Optional[VoxelComponent] = None
    superpoint = raw.copy()
    component_features = np.zeros(
        YIDU_COMPONENT_FEATURE_DIM, dtype=np.float32
    )
    reason_parts = ["mask_rgbd_observed"]
    if stage_index >= YIDU_STAGES.index("A3") and len(points):
        crop_dimensions = original_dimensions * float(
            resolved["crop_scale"]
        )
        origin = original_center - 0.5 * crop_dimensions
        component_set = build_voxel_components(
            points,
            origin=origin,
            voxel_size=resolved["voxel_size"],
            boundary_epsilon=resolved["boundary_epsilon"],
            neighbor_radius=resolved["neighbor_radius"],
            dilation_radius=resolved["dilation_radius"],
            point_view_ids=point_views,
            min_points_per_voxel=resolved["min_points_per_voxel"],
        )
        selected_component = select_inside_anchor(
            component_set,
            lower=original_center - 0.5 * original_dimensions,
            upper=original_center + 0.5 * original_dimensions,
            min_points=resolved["minimum_component_points"],
            min_voxels=resolved["minimum_component_voxels"],
            min_views=resolved["minimum_component_views"],
            min_inside_points=resolved["minimum_inside_points"],
            min_inside_fraction=resolved["minimum_inside_fraction"],
            normalization_dimensions=original_dimensions,
        )
        if selected_component is not None:
            superpoint, component_features = _component_candidate(
                selected_component,
                original_center=original_center,
                original_dimensions=original_dimensions,
                config=resolved,
            )
            component_features[:3] = (
                input_count,
                len(points),
                component_set.component_count,
            )
            reason_parts.append("voxel_component_candidate")
        else:
            reason_parts.append("identity_no_voxel_component")

    occupancy = superpoint.copy()
    occupancy_proposal: Optional[LocalOccupancyMSRProposal] = None
    occupancy_features = np.zeros(
        OCCUPANCY_MSR_FEATURE_DIM, dtype=np.float32
    )
    if stage_index >= YIDU_STAGES.index("A4"):
        occupancy_proposal = propose_local_occupancy_msr(
            original,
            view_records,
            config=resolved["occupancy_msr"],
        )
        occupancy_features = np.asarray(
            occupancy_proposal.feature_vector, dtype=np.float32
        ).copy()
        if occupancy_proposal.is_candidate:
            occupancy = np.asarray(
                occupancy_proposal.candidate_corners, dtype=np.float32
            ).copy()
            reason_parts.append("occupancy_msr_candidate")
        else:
            reason_parts.append(str(occupancy_proposal.reason))

    raw_fused: Optional[RawFusedQueryObservation] = None
    selected = occupancy.copy()
    selected_source = (
        "occupancy"
        if occupancy_proposal is not None
        and occupancy_proposal.is_candidate
        else (
            "superpoint"
            if selected_component is not None
            else ("raw_mask" if raw_candidate_verified else "original")
        )
    )
    selected_query_features = np.zeros(
        RAW_FUSED_QUERY_FEATURE_DIM, dtype=np.float32
    )
    if stage_index >= YIDU_STAGES.index("A5"):
        view_fraction = min(
            selected_views / float(max(int(resolved["max_views"]), 1)),
            1.0,
        )
        raw_geometry = (
            1.0 if raw_candidate_verified else 0.0
        )
        super_geometry = (
            0.0
            if selected_component is None
            else float(
                np.clip(
                    0.5
                    * (
                        selected_component.point_fraction
                        + component_features[7]
                    ),
                    0.0,
                    1.0,
                )
            )
        )
        occupancy_geometry = (
            0.0
            if occupancy_proposal is None
            else float(
                np.clip(
                    occupancy_proposal.candidate_support,
                    0.0,
                    1.0,
                )
            )
        )
        quality_rows = {
            "original": _quality_row(score, 0.0, 0.0, 1.0, 0.0),
            "raw_mask": _quality_row(
                score,
                mean_projection,
                mean_depth,
                raw_geometry,
                view_fraction,
            ),
        }
        superpoint_input = None
        occupancy_input = None
        if selected_component is not None:
            superpoint_input = superpoint
            quality_rows["superpoint"] = _quality_row(
                score,
                mean_projection,
                mean_depth,
                super_geometry,
                view_fraction,
            )
        if occupancy_proposal is not None and occupancy_proposal.is_candidate:
            occupancy_input = occupancy
            quality_rows["occupancy"] = _quality_row(
                score,
                mean_projection,
                mean_depth,
                occupancy_geometry,
                view_fraction,
            )
        raw_fused = observe_raw_fused_query(
            original=original,
            raw_mask=raw,
            superpoint=superpoint_input,
            occupancy=occupancy_input,
            quality_features=quality_rows,
            scorer_checkpoint=resolved["raw_fused_scorer_checkpoint"],
        )
        selected = np.asarray(
            raw_fused.selected.corners, dtype=np.float32
        ).copy()
        selected_source = str(raw_fused.selected.source)
        selected_query_features = np.asarray(
            raw_fused.selected.feature_vector, dtype=np.float32
        ).copy()
        reason_parts.append(
            f"raw_fused_{raw_fused.selection_mode}:{selected_source}"
        )

    gate_features = np.concatenate(
        (
            quality,
            np.asarray(
                [
                    float(
                        occupancy_proposal is not None
                        and occupancy_proposal.is_candidate
                    )
                ],
                dtype=np.float32,
            ),
            occupancy_features,
            selected_query_features,
        )
    ).astype(np.float32, copy=False)
    gate_decision: Optional[AP50SafetyDecision] = None
    if stage_index >= YIDU_STAGES.index("A6"):
        if quality_gate is None:
            raise ValueError("A6 requires a train-only YiDu AP50 safety gate")
        if tuple(quality_gate.feature_names) != YIDU_GATE_FEATURE_NAMES:
            raise ValueError(
                "YiDu gate checkpoint feature schema does not match "
                f"the fixed {YIDU_GATE_FEATURE_DIM}-D runtime schema"
            )
        gate_cfg = AP50SafetyGateConfig(
            **resolved["quality_gate"]
        ).validated()
        gate_decision = quality_gate.decide(
            gate_features,
            geometry_verified=(
                selected_source != "original"
                and np.isfinite(selected).all()
            ),
            config=gate_cfg,
        )
        reason_parts.append(
            "gate_accept"
            if gate_decision.accepted
            else f"gate_reject:{gate_decision.reason}"
        )

    return YiDuLocalObservation(
        stage=canonical_stage,
        reason="|".join(reason_parts),
        original_corners=original,
        raw_candidate_corners=raw,
        superpoint_candidate_corners=superpoint,
        occupancy_candidate_corners=occupancy,
        selected_candidate_corners=selected,
        selected_source=selected_source,
        input_point_count=input_count,
        cropped_point_count=len(points),
        selected_view_count=selected_views,
        component_set=component_set,
        selected_component_id=(
            -1
            if selected_component is None
            else int(selected_component.component_id)
        ),
        component_features=component_features,
        occupancy_proposal=occupancy_proposal,
        occupancy_features=occupancy_features,
        raw_fused_observation=raw_fused,
        raw_fused_selected_features=selected_query_features,
        gate_features=gate_features,
        gate_decision=gate_decision,
        mutation_enabled=False,
        applied=False,
    )


__all__ = [
    "DEFAULT_YIDU_LOCAL_OBSERVER_CONFIG",
    "YIDU_COMPONENT_FEATURE_DIM",
    "YIDU_COMPONENT_FEATURE_NAMES",
    "YIDU_GATE_FEATURE_DIM",
    "YIDU_GATE_FEATURE_NAMES",
    "YIDU_LOCAL_OBSERVER_SCHEMA",
    "YiDuLocalObservation",
    "observe_yidu_local_geometry",
    "resolve_yidu_local_observer_config",
]
