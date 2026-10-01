"""Observer-only P2-v2 geometry from unmatched Mask-RGBD components.

P2-v2 keeps the frozen B6 -> P1 -> P2 proposal path intact.  It consumes
only data already computed during the scheduled provider call:

* frozen P2 occupancy-selected residual anchors; and
* unmatched YOLOE masks lifted with the sensor's real depth.

Each mask point cloud is split into deterministic 3D connected components.
A component becomes a candidate only when it contains a selected P2 anchor
voxel.  The component's robust AABB replaces the generic P1-regressed
geometry in a detached diagnostic stream.  No label, CLIP feature, ground
truth, additional model call, or formal-output mutation is permitted.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from boxfusion.object_memory import robust_quantile_aabb
from boxfusion.occupancy_topk import (
    OccupancyTopKObservation,
)
from boxfusion.residual_proposal import (
    center_size_to_corners,
    pairwise_aabb_iou,
    stable_nms_aabb,
)
from boxfusion.voxel_components import build_voxel_components


P2V2_DIAGNOSTIC_SCHEMA = "boxfusion.p2v2.local_mask_geometry_observer.v1"
P2V2_SOURCE = "p2v2_local_mask_rgbd_component"


def _finite_float(
    name: str,
    value: Any,
    *,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    strict_lower: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if lower is not None:
        invalid = result <= lower if strict_lower else result < lower
        if invalid:
            relation = "greater than" if strict_lower else "at least"
            raise ValueError(f"{name} must be {relation} {lower}")
    if upper is not None and result > upper:
        raise ValueError(f"{name} must be at most {upper}")
    return result


def _integer(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _readonly(
    value: Any,
    *,
    dtype: Any,
    shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError(f"array must have shape {shape}, got {array.shape}")
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class P2LocalMaskGeometryConfig:
    """Strict, bounded configuration for the P2-v2 observer."""

    enabled: bool = False
    observer_only: bool = True
    mutate: bool = False
    collect_diagnostics: bool = False
    occupancy_voxel_size: float = 0.08
    component_voxel_size: float = 0.04
    component_neighbor_radius: int = 1
    component_dilation_radius: int = 0
    minimum_component_points: int = 12
    minimum_component_voxels: int = 4
    maximum_components_per_mask: int = 8
    maximum_masks_per_step: int = 64
    minimum_mask_score: float = 0.25
    minimum_valid_depth_ratio: float = 0.05
    lower_quantile: float = 0.02
    upper_quantile: float = 0.98
    minimum_box_extent: float = 0.08
    maximum_box_extent: float = 3.00
    anchor_margin: float = 0.08
    minimum_selected_voxels_inside: int = 1
    maximum_normalized_center_distance: float = 2.0
    maximum_absolute_center_distance: float = 1.0
    minimum_parent_iou: float = 0.0
    minimum_extent_ratio: float = 0.15
    maximum_extent_ratio: float = 4.0
    max_candidates_per_step: int = 16
    max_scene_candidates: int = 64
    step_nms_iou: float = 0.25
    scene_nms_iou: float = 0.25
    max_history_steps: int = 64

    def validated(self) -> "P2LocalMaskGeometryConfig":
        for name in (
            "enabled",
            "observer_only",
            "mutate",
            "collect_diagnostics",
        ):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"p2_local_mask_geometry.{name} "
                                 "must be Boolean")
        if not bool(self.observer_only):
            raise ValueError("P2-v2 must remain observer_only")
        if bool(self.mutate):
            raise ValueError("P2-v2 cannot mutate formal detections")
        result = P2LocalMaskGeometryConfig(
            enabled=bool(self.enabled),
            observer_only=True,
            mutate=False,
            collect_diagnostics=bool(self.collect_diagnostics),
            occupancy_voxel_size=_finite_float(
                "p2v2.occupancy_voxel_size",
                self.occupancy_voxel_size,
                lower=0.0,
                strict_lower=True,
            ),
            component_voxel_size=_finite_float(
                "p2v2.component_voxel_size",
                self.component_voxel_size,
                lower=0.0,
                strict_lower=True,
            ),
            component_neighbor_radius=_integer(
                "p2v2.component_neighbor_radius",
                self.component_neighbor_radius,
                1,
            ),
            component_dilation_radius=_integer(
                "p2v2.component_dilation_radius",
                self.component_dilation_radius,
                0,
            ),
            minimum_component_points=_integer(
                "p2v2.minimum_component_points",
                self.minimum_component_points,
                1,
            ),
            minimum_component_voxels=_integer(
                "p2v2.minimum_component_voxels",
                self.minimum_component_voxels,
                1,
            ),
            maximum_components_per_mask=_integer(
                "p2v2.maximum_components_per_mask",
                self.maximum_components_per_mask,
                1,
            ),
            maximum_masks_per_step=_integer(
                "p2v2.maximum_masks_per_step",
                self.maximum_masks_per_step,
                1,
            ),
            minimum_mask_score=_finite_float(
                "p2v2.minimum_mask_score",
                self.minimum_mask_score,
                lower=0.0,
                upper=1.0,
            ),
            minimum_valid_depth_ratio=_finite_float(
                "p2v2.minimum_valid_depth_ratio",
                self.minimum_valid_depth_ratio,
                lower=0.0,
                upper=1.0,
            ),
            lower_quantile=_finite_float(
                "p2v2.lower_quantile",
                self.lower_quantile,
                lower=0.0,
                upper=1.0,
            ),
            upper_quantile=_finite_float(
                "p2v2.upper_quantile",
                self.upper_quantile,
                lower=0.0,
                upper=1.0,
            ),
            minimum_box_extent=_finite_float(
                "p2v2.minimum_box_extent",
                self.minimum_box_extent,
                lower=0.0,
                strict_lower=True,
            ),
            maximum_box_extent=_finite_float(
                "p2v2.maximum_box_extent",
                self.maximum_box_extent,
                lower=0.0,
                strict_lower=True,
            ),
            anchor_margin=_finite_float(
                "p2v2.anchor_margin", self.anchor_margin, lower=0.0
            ),
            minimum_selected_voxels_inside=_integer(
                "p2v2.minimum_selected_voxels_inside",
                self.minimum_selected_voxels_inside,
                1,
            ),
            maximum_normalized_center_distance=_finite_float(
                "p2v2.maximum_normalized_center_distance",
                self.maximum_normalized_center_distance,
                lower=0.0,
            ),
            maximum_absolute_center_distance=_finite_float(
                "p2v2.maximum_absolute_center_distance",
                self.maximum_absolute_center_distance,
                lower=0.0,
            ),
            minimum_parent_iou=_finite_float(
                "p2v2.minimum_parent_iou",
                self.minimum_parent_iou,
                lower=0.0,
                upper=1.0,
            ),
            minimum_extent_ratio=_finite_float(
                "p2v2.minimum_extent_ratio",
                self.minimum_extent_ratio,
                lower=0.0,
                strict_lower=True,
            ),
            maximum_extent_ratio=_finite_float(
                "p2v2.maximum_extent_ratio",
                self.maximum_extent_ratio,
                lower=0.0,
                strict_lower=True,
            ),
            max_candidates_per_step=_integer(
                "p2v2.max_candidates_per_step",
                self.max_candidates_per_step,
                1,
            ),
            max_scene_candidates=_integer(
                "p2v2.max_scene_candidates",
                self.max_scene_candidates,
                1,
            ),
            step_nms_iou=_finite_float(
                "p2v2.step_nms_iou",
                self.step_nms_iou,
                lower=0.0,
                upper=1.0,
            ),
            scene_nms_iou=_finite_float(
                "p2v2.scene_nms_iou",
                self.scene_nms_iou,
                lower=0.0,
                upper=1.0,
            ),
            max_history_steps=_integer(
                "p2v2.max_history_steps", self.max_history_steps, 1
            ),
        )
        if result.lower_quantile >= result.upper_quantile:
            raise ValueError("P2-v2 quantiles must be strictly ordered")
        if result.minimum_box_extent > result.maximum_box_extent:
            raise ValueError("P2-v2 minimum extent exceeds maximum")
        if result.minimum_extent_ratio > result.maximum_extent_ratio:
            raise ValueError("P2-v2 extent-ratio interval is invalid")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_p2_local_mask_geometry_config(
    config: Optional[Mapping[str, Any] | P2LocalMaskGeometryConfig] = None,
) -> P2LocalMaskGeometryConfig:
    if config is None:
        return P2LocalMaskGeometryConfig().validated()
    if isinstance(config, P2LocalMaskGeometryConfig):
        return config.validated()
    if not isinstance(config, Mapping):
        raise TypeError("p2_local_mask_geometry must be a mapping")
    known = set(P2LocalMaskGeometryConfig.__dataclass_fields__)
    unknown = sorted(set(config) - known)
    if unknown:
        raise ValueError(
            "Unknown p2_local_mask_geometry key(s): "
            + ", ".join(unknown)
        )
    return P2LocalMaskGeometryConfig(**dict(config)).validated()


@dataclass(frozen=True)
class P2MaskRGBDInput:
    """Detached adapter for one already-lifted unmatched mask."""

    source_id: str
    frame_index: int
    provider_step: int
    score: float
    valid_depth_ratio: float
    points_world: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("source_id must be non-empty")
        object.__setattr__(
            self, "frame_index", _integer("frame_index", self.frame_index, 0)
        )
        object.__setattr__(
            self,
            "provider_step",
            _integer("provider_step", self.provider_step, 0),
        )
        object.__setattr__(
            self,
            "score",
            _finite_float("mask score", self.score, lower=0.0, upper=1.0),
        )
        object.__setattr__(
            self,
            "valid_depth_ratio",
            _finite_float(
                "valid_depth_ratio",
                self.valid_depth_ratio,
                lower=0.0,
                upper=1.0,
            ),
        )
        points = np.asarray(self.points_world)
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or not np.issubdtype(points.dtype, np.number)
            or not np.isfinite(points).all()
        ):
            raise ValueError("points_world must have finite shape [N,3]")
        object.__setattr__(
            self, "points_world", _readonly(points, dtype=np.float32)
        )


@dataclass(frozen=True)
class P2MaskGeometryCandidate:
    candidate_id: str
    parent_p2_candidate_id: str
    mask_source_id: str
    frame_index: int
    provider_step: int
    box: np.ndarray
    corners: np.ndarray
    parent_box: np.ndarray
    score: float
    parent_objectness: float
    occupancy_score: float
    mask_score: float
    valid_depth_ratio: float
    component_point_count: int
    component_voxel_count: int
    selected_voxels_inside: int
    anchor_inside: bool
    parent_iou: float
    normalized_center_distance: float
    extent_ratios: np.ndarray
    center_shift_ratios: np.ndarray
    source: str = P2V2_SOURCE

    def __post_init__(self) -> None:
        for name, shape in (
            ("box", (6,)),
            ("parent_box", (6,)),
            ("extent_ratios", (3,)),
            ("center_shift_ratios", (3,)),
        ):
            array = _readonly(
                getattr(self, name), dtype=np.float32, shape=shape
            )
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, array)
        corners = _readonly(
            self.corners, dtype=np.float32, shape=(8, 3)
        )
        if not np.isfinite(corners).all():
            raise ValueError("corners must be finite")
        object.__setattr__(self, "corners", corners)
        if np.any(self.box[3:] <= 0.0) or np.any(
            self.parent_box[3:] <= 0.0
        ):
            raise ValueError("candidate and parent extents must be positive")
        for name in (
            "score",
            "parent_objectness",
            "occupancy_score",
            "mask_score",
            "valid_depth_ratio",
            "parent_iou",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(name, getattr(self, name), lower=0.0, upper=1.0),
            )
        object.__setattr__(
            self,
            "normalized_center_distance",
            _finite_float(
                "normalized_center_distance",
                self.normalized_center_distance,
                lower=0.0,
            ),
        )
        for name in (
            "component_point_count",
            "component_voxel_count",
            "selected_voxels_inside",
        ):
            object.__setattr__(
                self, name, _integer(name, getattr(self, name), 0)
            )
        object.__setattr__(self, "anchor_inside", bool(self.anchor_inside))
        if not self.candidate_id or not self.parent_p2_candidate_id:
            raise ValueError("candidate IDs must be non-empty")
        if not self.mask_source_id:
            raise ValueError("mask_source_id must be non-empty")
        if self.source != P2V2_SOURCE:
            raise ValueError("invalid P2-v2 source")


@dataclass(frozen=True)
class P2MaskGeometryStep:
    frame_index: int
    provider_step: int
    selected_voxel_count: int
    occupancy_component_count: int
    mask_observation_count: int
    mask_component_count: int
    eligible_pair_count: int
    candidates: Tuple[P2MaskGeometryCandidate, ...]
    seconds: float
    failed: bool = False
    error: str = ""


@dataclass(frozen=True)
class _MaskComponent:
    source: P2MaskRGBDInput
    component_id: int
    points: np.ndarray
    voxel_count: int
    box: np.ndarray


def _candidate_coordinate(candidate_id: str) -> Tuple[int, int, int]:
    fields = str(candidate_id).rsplit(":", 3)
    if len(fields) != 4:
        raise ValueError(f"invalid P2 candidate id: {candidate_id!r}")
    return tuple(int(value) for value in fields[1:])


def _mask_components(
    observation: P2MaskRGBDInput,
    cfg: P2LocalMaskGeometryConfig,
) -> Tuple[_MaskComponent, ...]:
    if (
        observation.score < cfg.minimum_mask_score
        or observation.valid_depth_ratio < cfg.minimum_valid_depth_ratio
        or len(observation.points_world) < cfg.minimum_component_points
    ):
        return ()
    components = build_voxel_components(
        observation.points_world,
        origin=np.zeros(3, dtype=np.float64),
        voxel_size=cfg.component_voxel_size,
        boundary_epsilon=1e-7,
        neighbor_radius=cfg.component_neighbor_radius,
        dilation_radius=cfg.component_dilation_radius,
        min_points_per_voxel=1,
    )
    eligible = [
        component
        for component in components.components
        if component.point_count >= cfg.minimum_component_points
        and component.voxel_count >= cfg.minimum_component_voxels
    ]
    eligible.sort(
        key=lambda row: (
            -row.point_count,
            -row.voxel_count,
            row.stable_key,
        )
    )
    rows = []
    for component in eligible[: cfg.maximum_components_per_mask]:
        center, extent = robust_quantile_aabb(
            component.points,
            lower_quantile=cfg.lower_quantile,
            upper_quantile=cfg.upper_quantile,
            min_points=cfg.minimum_component_points,
            minimum_dimension=cfg.minimum_box_extent,
        )
        if (
            np.any(extent < cfg.minimum_box_extent)
            or np.any(extent > cfg.maximum_box_extent)
        ):
            continue
        rows.append(
            _MaskComponent(
                source=observation,
                component_id=int(component.component_id),
                points=_readonly(component.points, dtype=np.float32),
                voxel_count=int(component.voxel_count),
                box=_readonly(
                    np.concatenate((center, extent)),
                    dtype=np.float32,
                    shape=(6,),
                ),
            )
        )
    return tuple(rows)


def _bounds(box: np.ndarray, margin: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    center = np.asarray(box[:3], dtype=np.float64)
    half = 0.5 * np.asarray(box[3:], dtype=np.float64) + float(margin)
    return center - half, center + half


def _inside(points: np.ndarray, box: np.ndarray, margin: float) -> np.ndarray:
    lower, upper = _bounds(box, margin)
    return np.all(
        (points >= lower[None]) & (points <= upper[None]), axis=1
    )


def _fit_step(
    p2: OccupancyTopKObservation,
    masks: Sequence[P2MaskRGBDInput],
    cfg: P2LocalMaskGeometryConfig,
) -> Tuple[
    Tuple[P2MaskGeometryCandidate, ...],
    int,
    int,
    int,
]:
    batch = p2.base.voxel_batch
    selected_indices = np.asarray(
        p2.selected_voxel_indices, dtype=np.int64
    )
    selected_scores = np.asarray(
        p2.selected_voxel_scores, dtype=np.float64
    )
    if len(selected_indices) != len(selected_scores):
        raise ValueError("P2 selected voxel arrays do not align")
    selected_centers = np.asarray(batch.centers)[selected_indices]
    selected_coordinates = np.asarray(batch.coordinates)[selected_indices]
    coordinate_to_center = {
        tuple(int(value) for value in coordinate): center
        for coordinate, center in zip(selected_coordinates, selected_centers)
    }
    occupancy_components = build_voxel_components(
        selected_centers,
        origin=np.zeros(3, dtype=np.float64),
        voxel_size=cfg.occupancy_voxel_size,
        boundary_epsilon=1e-7,
        neighbor_radius=1,
        dilation_radius=0,
        min_points_per_voxel=1,
    )
    component_rows = []
    for mask in sorted(
        masks[: cfg.maximum_masks_per_step],
        key=lambda row: (-row.score, row.source_id),
    ):
        component_rows.extend(_mask_components(mask, cfg))
    pairs = []
    for component_index, component in enumerate(component_rows):
        inside_selected = _inside(
            selected_centers, component.box, cfg.anchor_margin
        )
        selected_inside_count = int(np.sum(inside_selected))
        if selected_inside_count < cfg.minimum_selected_voxels_inside:
            continue
        for anchor_index, anchor in enumerate(p2.selected):
            coordinate = _candidate_coordinate(anchor.candidate_id)
            anchor_voxel_center = coordinate_to_center.get(coordinate)
            if anchor_voxel_center is None:
                continue
            anchor_inside = bool(
                _inside(
                    np.asarray(anchor_voxel_center)[None],
                    component.box,
                    cfg.anchor_margin,
                )[0]
            )
            parent_box = np.asarray(anchor.box, dtype=np.float64)
            parent_iou = float(
                pairwise_aabb_iou(
                    parent_box[None], component.box[None]
                )[0, 0]
            )
            absolute_distance = float(
                np.linalg.norm(parent_box[:3] - component.box[:3])
            )
            normalized_distance = float(
                np.linalg.norm(
                    (parent_box[:3] - component.box[:3])
                    / np.maximum(component.box[3:], cfg.minimum_box_extent)
                )
            )
            extent_ratio = np.asarray(
                component.box[3:] / parent_box[3:], dtype=np.float64
            )
            if (
                not anchor_inside
                or absolute_distance > cfg.maximum_absolute_center_distance
                or normalized_distance
                > cfg.maximum_normalized_center_distance
                or parent_iou < cfg.minimum_parent_iou
                or np.any(extent_ratio < cfg.minimum_extent_ratio)
                or np.any(extent_ratio > cfg.maximum_extent_ratio)
            ):
                continue
            pairs.append(
                (
                    (
                        -selected_inside_count,
                        normalized_distance,
                        -float(anchor.occupancy_score),
                        -float(anchor.objectness),
                        -float(component.source.score),
                        anchor.candidate_id,
                        component.source.source_id,
                        component.component_id,
                    ),
                    anchor_index,
                    component_index,
                    selected_inside_count,
                    parent_iou,
                    normalized_distance,
                    extent_ratio,
                )
            )
    used_anchors = set()
    used_components = set()
    candidates = []
    for (
        _,
        anchor_index,
        component_index,
        selected_inside_count,
        parent_iou,
        normalized_distance,
        extent_ratio,
    ) in sorted(pairs, key=lambda row: row[0]):
        if anchor_index in used_anchors or component_index in used_components:
            continue
        used_anchors.add(anchor_index)
        used_components.add(component_index)
        anchor = p2.selected[anchor_index]
        component = component_rows[component_index]
        parent_box = np.asarray(anchor.box, dtype=np.float32)
        box = np.asarray(component.box, dtype=np.float32)
        center_shift_ratios = np.abs(
            box[:3] - parent_box[:3]
        ) / np.maximum(parent_box[3:], cfg.minimum_box_extent)
        identity = (
            f"{anchor.candidate_id}|{component.source.source_id}|"
            f"{component.component_id}"
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        candidates.append(
            P2MaskGeometryCandidate(
                candidate_id=f"p2v2:{digest}",
                parent_p2_candidate_id=anchor.candidate_id,
                mask_source_id=component.source.source_id,
                frame_index=int(p2.base.frame_index),
                provider_step=int(p2.base.provider_step),
                box=box,
                corners=center_size_to_corners(box)[0],
                parent_box=parent_box,
                # Occupancy remains the primary order: this ablation changes
                # geometry, not the learned proposal score contract.
                score=float(anchor.occupancy_score),
                parent_objectness=float(anchor.objectness),
                occupancy_score=float(anchor.occupancy_score),
                mask_score=float(component.source.score),
                valid_depth_ratio=float(
                    component.source.valid_depth_ratio
                ),
                component_point_count=len(component.points),
                component_voxel_count=component.voxel_count,
                selected_voxels_inside=selected_inside_count,
                anchor_inside=True,
                parent_iou=parent_iou,
                normalized_center_distance=normalized_distance,
                extent_ratios=extent_ratio,
                center_shift_ratios=center_shift_ratios,
            )
        )
    if candidates:
        keep = stable_nms_aabb(
            np.stack([row.box for row in candidates]),
            np.asarray([row.score for row in candidates]),
            cfg.step_nms_iou,
            tie_breakers=[row.candidate_id for row in candidates],
            max_output=cfg.max_candidates_per_step,
        )
        candidates = [candidates[int(index)] for index in keep]
    return (
        tuple(candidates),
        occupancy_components.component_count,
        len(component_rows),
        len(pairs),
    )


class P2LocalMaskGeometryObserver:
    """Bounded observer that cannot affect BoxFusion's result object."""

    def __init__(
        self,
        config: Mapping[str, Any] | P2LocalMaskGeometryConfig,
        *,
        parent_p2_checkpoint_sha256: str,
        provider_name: str,
    ) -> None:
        self.config = resolve_p2_local_mask_geometry_config(config)
        if not self.config.enabled:
            raise ValueError("P2-v2 observer requires enabled configuration")
        if not self.config.collect_diagnostics:
            raise ValueError("P2-v2 requires collect_diagnostics")
        checkpoint_sha = str(parent_p2_checkpoint_sha256).strip().lower()
        if checkpoint_sha != "injected" and (
            len(checkpoint_sha) != 64
            or any(character not in "0123456789abcdef"
                   for character in checkpoint_sha)
        ):
            raise ValueError("P2-v2 requires the frozen P2 checkpoint SHA")
        self.parent_p2_checkpoint_sha256 = checkpoint_sha
        self.provider_name = str(provider_name)
        self.scene_id: Optional[str] = None
        self.steps: list[P2MaskGeometryStep] = []

    def reset(self, scene_id: str) -> None:
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError("scene_id must be non-empty")
        self.scene_id = scene_id.strip()
        self.steps.clear()

    def observe(
        self,
        *,
        scene_id: str,
        p2_observation: OccupancyTopKObservation,
        masks: Sequence[P2MaskRGBDInput],
    ) -> P2MaskGeometryStep:
        requested = str(scene_id).strip()
        if self.scene_id != requested:
            self.reset(requested)
        if not isinstance(p2_observation, OccupancyTopKObservation):
            raise TypeError("p2_observation has the wrong type")
        if any(
            row.frame_index != p2_observation.base.frame_index
            or row.provider_step != p2_observation.base.provider_step
            for row in masks
        ):
            raise ValueError("mask inputs are not aligned with the P2 step")
        started = time.perf_counter()
        candidates, occupancy_count, mask_count, pair_count = _fit_step(
            p2_observation, masks, self.config
        )
        step = P2MaskGeometryStep(
            frame_index=int(p2_observation.base.frame_index),
            provider_step=int(p2_observation.base.provider_step),
            selected_voxel_count=len(
                p2_observation.selected_voxel_indices
            ),
            occupancy_component_count=int(occupancy_count),
            mask_observation_count=len(masks),
            mask_component_count=int(mask_count),
            eligible_pair_count=int(pair_count),
            candidates=candidates,
            seconds=float(time.perf_counter() - started),
        )
        self.steps.append(step)
        del self.steps[: -self.config.max_history_steps]
        return step

    def record_failure(
        self,
        *,
        scene_id: str,
        frame_index: int,
        provider_step: int,
        selected_voxel_count: int,
        mask_observation_count: int,
        elapsed_seconds: float,
        error: Exception,
    ) -> None:
        requested = str(scene_id).strip()
        if self.scene_id != requested:
            self.reset(requested)
        self.steps.append(
            P2MaskGeometryStep(
                frame_index=int(frame_index),
                provider_step=int(provider_step),
                selected_voxel_count=int(selected_voxel_count),
                occupancy_component_count=0,
                mask_observation_count=int(mask_observation_count),
                mask_component_count=0,
                eligible_pair_count=0,
                candidates=(),
                seconds=max(float(elapsed_seconds), 0.0),
                failed=True,
                error=f"{type(error).__name__}: {error}",
            )
        )
        del self.steps[: -self.config.max_history_steps]

    def scene_candidates(self) -> Tuple[P2MaskGeometryCandidate, ...]:
        rows = [candidate for step in self.steps for candidate in step.candidates]
        if not rows:
            return ()
        keep = stable_nms_aabb(
            np.stack([row.box for row in rows]),
            np.asarray([row.score for row in rows]),
            self.config.scene_nms_iou,
            tie_breakers=[row.candidate_id for row in rows],
            max_output=self.config.max_scene_candidates,
        )
        return tuple(rows[int(index)] for index in keep)

    def diagnostic_payload(self) -> dict[str, np.ndarray]:
        steps = tuple(self.steps)
        candidates = self.scene_candidates()
        config_json = json.dumps(
            self.config.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )

        def values(
            attribute: str,
            empty_shape: Tuple[int, ...],
            dtype: Any,
        ) -> np.ndarray:
            if not candidates:
                return np.empty(empty_shape, dtype=dtype)
            return np.asarray(
                [getattr(row, attribute) for row in candidates],
                dtype=dtype,
            )

        return {
            "p2v2_schema": np.asarray(P2V2_DIAGNOSTIC_SCHEMA),
            "p2v2_stage": np.asarray("P2V2"),
            "p2v2_profile": np.asarray(
                "p2v2_local_component_mask_rgbd_observer"
            ),
            "p2v2_enabled": np.asarray(True, dtype=bool),
            "p2v2_observer_only": np.asarray(True, dtype=bool),
            "p2v2_uses_ground_truth": np.asarray(False, dtype=bool),
            "p2v2_reads_semantic_labels": np.asarray(False, dtype=bool),
            "p2v2_mutation_enabled": np.asarray(False, dtype=bool),
            "p2v2_applied_count": np.asarray(0, dtype=np.int64),
            "p2v2_complete": np.asarray(
                bool(steps) and not any(step.failed for step in steps),
                dtype=bool,
            ),
            "p2v2_source": np.asarray(P2V2_SOURCE),
            "p2v2_mask_provider": np.asarray(self.provider_name),
            "p2v2_parent_p2_checkpoint_sha256": np.asarray(
                self.parent_p2_checkpoint_sha256
            ),
            "p2v2_config_json": np.asarray(config_json),
            "p2v2_step_frame_ids": np.asarray(
                [step.frame_index for step in steps], dtype=np.int64
            ),
            "p2v2_step_provider_steps": np.asarray(
                [step.provider_step for step in steps], dtype=np.int64
            ),
            "p2v2_step_selected_voxel_counts": np.asarray(
                [step.selected_voxel_count for step in steps],
                dtype=np.int64,
            ),
            "p2v2_step_occupancy_component_counts": np.asarray(
                [step.occupancy_component_count for step in steps],
                dtype=np.int64,
            ),
            "p2v2_step_mask_observation_counts": np.asarray(
                [step.mask_observation_count for step in steps],
                dtype=np.int64,
            ),
            "p2v2_step_mask_component_counts": np.asarray(
                [step.mask_component_count for step in steps],
                dtype=np.int64,
            ),
            "p2v2_step_eligible_pair_counts": np.asarray(
                [step.eligible_pair_count for step in steps],
                dtype=np.int64,
            ),
            "p2v2_step_candidate_counts": np.asarray(
                [len(step.candidates) for step in steps], dtype=np.int64
            ),
            "p2v2_step_seconds": np.asarray(
                [step.seconds for step in steps], dtype=np.float64
            ),
            "p2v2_step_failed": np.asarray(
                [step.failed for step in steps], dtype=bool
            ),
            "p2v2_step_errors": np.asarray(
                [step.error for step in steps], dtype=np.str_
            ),
            "p2v2_candidate_ids": values(
                "candidate_id", (0,), np.str_
            ),
            "p2v2_parent_p2_candidate_ids": values(
                "parent_p2_candidate_id", (0,), np.str_
            ),
            "p2v2_mask_source_ids": values(
                "mask_source_id", (0,), np.str_
            ),
            "p2v2_candidate_boxes": values(
                "box", (0, 6), np.float32
            ),
            "p2v2_candidate_corners": values(
                "corners", (0, 8, 3), np.float32
            ),
            "p2v2_candidate_parent_boxes": values(
                "parent_box", (0, 6), np.float32
            ),
            "p2v2_candidate_scores": values(
                "score", (0,), np.float32
            ),
            "p2v2_candidate_parent_objectness": values(
                "parent_objectness", (0,), np.float32
            ),
            "p2v2_candidate_occupancy_scores": values(
                "occupancy_score", (0,), np.float32
            ),
            "p2v2_candidate_mask_scores": values(
                "mask_score", (0,), np.float32
            ),
            "p2v2_candidate_valid_depth_ratios": values(
                "valid_depth_ratio", (0,), np.float32
            ),
            "p2v2_candidate_component_point_counts": values(
                "component_point_count", (0,), np.int64
            ),
            "p2v2_candidate_component_voxel_counts": values(
                "component_voxel_count", (0,), np.int64
            ),
            "p2v2_candidate_selected_voxels_inside": values(
                "selected_voxels_inside", (0,), np.int64
            ),
            "p2v2_candidate_anchor_inside": values(
                "anchor_inside", (0,), bool
            ),
            "p2v2_candidate_parent_iou": values(
                "parent_iou", (0,), np.float32
            ),
            "p2v2_candidate_normalized_center_distance": values(
                "normalized_center_distance", (0,), np.float32
            ),
            "p2v2_candidate_extent_ratios": values(
                "extent_ratios", (0, 3), np.float32
            ),
            "p2v2_candidate_center_shift_ratios": values(
                "center_shift_ratios", (0, 3), np.float32
            ),
            "p2v2_candidate_applied": np.zeros(
                len(candidates), dtype=bool
            ),
        }


__all__ = [
    "P2LocalMaskGeometryConfig",
    "P2LocalMaskGeometryObserver",
    "P2MaskGeometryCandidate",
    "P2MaskGeometryStep",
    "P2MaskRGBDInput",
    "P2V2_DIAGNOSTIC_SCHEMA",
    "P2V2_SOURCE",
    "resolve_p2_local_mask_geometry_config",
]
