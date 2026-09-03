"""Observer-only reliability fusion for P2-v2 Mask-RGBD geometry.

P2-v3 adds exactly one operation to the frozen P2-v2 stream: a convex,
axis-aware fusion between the P2-v2 component box and its frozen P2 parent
box.  The fusion weight is computed only from evidence already present in
the P2-v2 candidate:

* mask and valid-depth confidence;
* bounded point, voxel, and anchor support;
* agreement between the component and parent geometry; and
* frozen P2 objectness/occupancy confidence.

No labels, CLIP features, ground truth, extra provider call, learned
checkpoint, or formal-output mutation are available to this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from boxfusion.p2_local_mask_geometry import (
    P2MaskGeometryCandidate,
    P2MaskGeometryStep,
    P2V2_DIAGNOSTIC_SCHEMA,
    P2V2_SOURCE,
)
from boxfusion.residual_proposal import (
    center_size_to_corners,
    stable_nms_aabb,
)


P2V3_DIAGNOSTIC_SCHEMA = (
    "boxfusion.p2v3.reliability_geometry_fusion_observer.v1"
)
P2V3_PROFILE = "p2v3_reliability_geometry_fusion_observer"
P2V3_SOURCE = "p2v3_reliability_geometry_fusion"
P2V3_RELIABILITY_CONTRACT = "analytic_observable_evidence_v1"


def _finite(
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
    if not math.isfinite(result):
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
class P2ReliabilityFusionConfig:
    """Strict bounds for the analytic, checkpoint-free P2-v3 observer."""

    enabled: bool = False
    observer_only: bool = True
    mutate: bool = False
    collect_diagnostics: bool = False
    reliability_epsilon: float = 1e-6
    component_prior_precision: float = 1.0
    parent_prior_precision: float = 1.0
    point_support_saturation: float = 64.0
    voxel_support_saturation: float = 16.0
    anchor_support_saturation: float = 2.0
    minimum_component_weight: float = 0.35
    maximum_component_weight: float = 0.85
    center_weight_scale: float = 1.0
    extent_weight_scale: float = 1.0
    max_candidates_per_step: int = 16
    max_scene_candidates: int = 64
    step_nms_iou: float = 0.25
    scene_nms_iou: float = 0.25
    max_history_steps: int = 64

    def validated(self) -> "P2ReliabilityFusionConfig":
        for name in (
            "enabled",
            "observer_only",
            "mutate",
            "collect_diagnostics",
        ):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"p2_reliability_fusion.{name} "
                                 "must be Boolean")
        if not bool(self.observer_only):
            raise ValueError("P2-v3 must remain observer_only")
        if bool(self.mutate):
            raise ValueError("P2-v3 cannot mutate formal detections")
        result = P2ReliabilityFusionConfig(
            enabled=bool(self.enabled),
            observer_only=True,
            mutate=False,
            collect_diagnostics=bool(self.collect_diagnostics),
            reliability_epsilon=_finite(
                "p2v3.reliability_epsilon",
                self.reliability_epsilon,
                lower=0.0,
                strict_lower=True,
            ),
            component_prior_precision=_finite(
                "p2v3.component_prior_precision",
                self.component_prior_precision,
                lower=0.0,
                strict_lower=True,
            ),
            parent_prior_precision=_finite(
                "p2v3.parent_prior_precision",
                self.parent_prior_precision,
                lower=0.0,
                strict_lower=True,
            ),
            point_support_saturation=_finite(
                "p2v3.point_support_saturation",
                self.point_support_saturation,
                lower=0.0,
                strict_lower=True,
            ),
            voxel_support_saturation=_finite(
                "p2v3.voxel_support_saturation",
                self.voxel_support_saturation,
                lower=0.0,
                strict_lower=True,
            ),
            anchor_support_saturation=_finite(
                "p2v3.anchor_support_saturation",
                self.anchor_support_saturation,
                lower=0.0,
                strict_lower=True,
            ),
            minimum_component_weight=_finite(
                "p2v3.minimum_component_weight",
                self.minimum_component_weight,
                lower=0.0,
                upper=1.0,
            ),
            maximum_component_weight=_finite(
                "p2v3.maximum_component_weight",
                self.maximum_component_weight,
                lower=0.0,
                upper=1.0,
            ),
            center_weight_scale=_finite(
                "p2v3.center_weight_scale",
                self.center_weight_scale,
                lower=0.0,
                strict_lower=True,
            ),
            extent_weight_scale=_finite(
                "p2v3.extent_weight_scale",
                self.extent_weight_scale,
                lower=0.0,
                strict_lower=True,
            ),
            max_candidates_per_step=_integer(
                "p2v3.max_candidates_per_step",
                self.max_candidates_per_step,
                1,
            ),
            max_scene_candidates=_integer(
                "p2v3.max_scene_candidates",
                self.max_scene_candidates,
                1,
            ),
            step_nms_iou=_finite(
                "p2v3.step_nms_iou",
                self.step_nms_iou,
                lower=0.0,
                upper=1.0,
            ),
            scene_nms_iou=_finite(
                "p2v3.scene_nms_iou",
                self.scene_nms_iou,
                lower=0.0,
                upper=1.0,
            ),
            max_history_steps=_integer(
                "p2v3.max_history_steps", self.max_history_steps, 1
            ),
        )
        if result.minimum_component_weight > result.maximum_component_weight:
            raise ValueError("P2-v3 component-weight interval is invalid")
        return result

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_p2_reliability_fusion_config(
    config: Optional[Mapping[str, Any] | P2ReliabilityFusionConfig] = None,
) -> P2ReliabilityFusionConfig:
    if config is None:
        return P2ReliabilityFusionConfig().validated()
    if isinstance(config, P2ReliabilityFusionConfig):
        return config.validated()
    if not isinstance(config, Mapping):
        raise TypeError("p2_reliability_fusion must be a mapping")
    known = set(P2ReliabilityFusionConfig.__dataclass_fields__)
    unknown = sorted(set(config) - known)
    if unknown:
        raise ValueError(
            "Unknown p2_reliability_fusion key(s): "
            + ", ".join(unknown)
        )
    return P2ReliabilityFusionConfig(**dict(config)).validated()


def _saturating_support(count: int, scale: float) -> float:
    value = max(float(count), 0.0)
    return float(value / (value + float(scale)))


def _geometric_mean(values: Sequence[float], epsilon: float) -> float:
    array = np.clip(np.asarray(values, dtype=np.float64), epsilon, 1.0)
    return float(np.exp(np.mean(np.log(array))))


@dataclass(frozen=True)
class P2ReliabilityFusedCandidate:
    candidate_id: str
    parent_p2v2_candidate_id: str
    parent_p2_candidate_id: str
    mask_source_id: str
    frame_index: int
    provider_step: int
    fused_box: np.ndarray
    fused_corners: np.ndarray
    component_box: np.ndarray
    component_corners: np.ndarray
    parent_box: np.ndarray
    parent_corners: np.ndarray
    score: float
    component_weight: float
    center_component_weights: np.ndarray
    extent_component_weights: np.ndarray
    component_reliability: float
    parent_reliability: float
    mask_reliability: float
    depth_reliability: float
    support_reliability: float
    agreement_reliability: float
    source: str = P2V3_SOURCE

    def __post_init__(self) -> None:
        for name in ("fused_box", "component_box", "parent_box"):
            value = _readonly(
                getattr(self, name), dtype=np.float32, shape=(6,)
            )
            if not np.isfinite(value).all() or np.any(value[3:] <= 0.0):
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        for name in (
            "fused_corners",
            "component_corners",
            "parent_corners",
        ):
            value = _readonly(
                getattr(self, name), dtype=np.float32, shape=(8, 3)
            )
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        for name in (
            "center_component_weights",
            "extent_component_weights",
        ):
            value = _readonly(
                getattr(self, name), dtype=np.float32, shape=(3,)
            )
            if (
                not np.isfinite(value).all()
                or np.any(value < 0.0)
                or np.any(value > 1.0)
            ):
                raise ValueError(f"{name} must lie in [0,1]")
            object.__setattr__(self, name, value)
        for name in (
            "score",
            "component_weight",
            "component_reliability",
            "parent_reliability",
            "mask_reliability",
            "depth_reliability",
            "support_reliability",
            "agreement_reliability",
        ):
            object.__setattr__(
                self,
                name,
                _finite(
                    name, getattr(self, name), lower=0.0, upper=1.0
                ),
            )
        for name in (
            "candidate_id",
            "parent_p2v2_candidate_id",
            "parent_p2_candidate_id",
            "mask_source_id",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(
                self, name
            ):
                raise ValueError(f"{name} must be non-empty")
        object.__setattr__(
            self, "frame_index", _integer("frame_index", self.frame_index, 0)
        )
        object.__setattr__(
            self,
            "provider_step",
            _integer("provider_step", self.provider_step, 0),
        )
        if self.source != P2V3_SOURCE:
            raise ValueError("invalid P2-v3 source")

    @property
    def box(self) -> np.ndarray:
        return self.fused_box

    @property
    def corners(self) -> np.ndarray:
        return self.fused_corners


@dataclass(frozen=True)
class P2ReliabilityFusionStep:
    frame_index: int
    provider_step: int
    input_candidate_count: int
    eligible_candidate_count: int
    candidates: Tuple[P2ReliabilityFusedCandidate, ...]
    seconds: float
    failed: bool = False
    error: str = ""


def _fuse_candidate(
    row: P2MaskGeometryCandidate,
    cfg: P2ReliabilityFusionConfig,
) -> P2ReliabilityFusedCandidate:
    epsilon = cfg.reliability_epsilon
    mask_reliability = float(row.mask_score)
    depth_reliability = float(row.valid_depth_ratio)
    point_support = _saturating_support(
        row.component_point_count, cfg.point_support_saturation
    )
    voxel_support = _saturating_support(
        row.component_voxel_count, cfg.voxel_support_saturation
    )
    anchor_support = _saturating_support(
        row.selected_voxels_inside, cfg.anchor_support_saturation
    )
    support_reliability = _geometric_mean(
        (point_support, voxel_support, anchor_support), epsilon
    )
    center_agreement = 1.0 / (
        1.0 + float(row.normalized_center_distance)
    )
    extent_ratios = np.asarray(row.extent_ratios, dtype=np.float64)
    axis_extent_agreement = np.exp(
        -np.abs(np.log(np.maximum(extent_ratios, epsilon)))
    )
    extent_agreement = float(np.mean(axis_extent_agreement))
    agreement_reliability = _geometric_mean(
        (
            center_agreement,
            extent_agreement,
            max(float(row.parent_iou), epsilon),
        ),
        epsilon,
    )
    component_reliability = _geometric_mean(
        (
            mask_reliability,
            depth_reliability,
            support_reliability,
            agreement_reliability,
        ),
        epsilon,
    )
    parent_reliability = _geometric_mean(
        (float(row.parent_objectness), float(row.occupancy_score)),
        epsilon,
    )
    component_precision = (
        cfg.component_prior_precision * component_reliability
    )
    parent_precision = cfg.parent_prior_precision * parent_reliability
    normalized_precision = component_precision / (
        component_precision + parent_precision + epsilon
    )
    minimum = cfg.minimum_component_weight
    maximum = cfg.maximum_component_weight
    component_weight = float(
        minimum + (maximum - minimum) * normalized_precision
    )

    component_box = np.asarray(row.box, dtype=np.float64)
    parent_box = np.asarray(row.parent_box, dtype=np.float64)
    center_shift = np.asarray(
        row.center_shift_ratios, dtype=np.float64
    )
    axis_center_agreement = 1.0 / (1.0 + center_shift)
    center_weights = minimum + (
        np.clip(
            (component_weight - minimum)
            * cfg.center_weight_scale,
            0.0,
            maximum - minimum,
        )
        * axis_center_agreement
    )
    extent_weights = minimum + (
        np.clip(
            (component_weight - minimum)
            * cfg.extent_weight_scale,
            0.0,
            maximum - minimum,
        )
        * axis_extent_agreement
    )
    center_weights = np.clip(center_weights, minimum, maximum)
    extent_weights = np.clip(extent_weights, minimum, maximum)
    fused_center = (
        center_weights * component_box[:3]
        + (1.0 - center_weights) * parent_box[:3]
    )
    fused_extent = (
        extent_weights * component_box[3:]
        + (1.0 - extent_weights) * parent_box[3:]
    )
    fused_box = np.concatenate((fused_center, fused_extent)).astype(
        np.float32
    )
    identity = (
        f"{row.candidate_id}|{P2V3_RELIABILITY_CONTRACT}|"
        f"{component_weight:.9f}"
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return P2ReliabilityFusedCandidate(
        candidate_id=f"p2v3:{digest}",
        parent_p2v2_candidate_id=row.candidate_id,
        parent_p2_candidate_id=row.parent_p2_candidate_id,
        mask_source_id=row.mask_source_id,
        frame_index=row.frame_index,
        provider_step=row.provider_step,
        fused_box=fused_box,
        fused_corners=center_size_to_corners(fused_box)[0],
        component_box=row.box,
        component_corners=row.corners,
        parent_box=row.parent_box,
        parent_corners=center_size_to_corners(row.parent_box)[0],
        # Score preservation isolates geometry from ranking calibration.
        score=float(row.score),
        component_weight=component_weight,
        center_component_weights=center_weights,
        extent_component_weights=extent_weights,
        component_reliability=component_reliability,
        parent_reliability=parent_reliability,
        mask_reliability=mask_reliability,
        depth_reliability=depth_reliability,
        support_reliability=support_reliability,
        agreement_reliability=agreement_reliability,
    )


class P2ReliabilityFusionObserver:
    """Detached analytic fusion of one P2-v2 step at a time."""

    def __init__(
        self,
        config: Mapping[str, Any] | P2ReliabilityFusionConfig,
        *,
        parent_p2_checkpoint_sha256: str,
    ) -> None:
        self.config = resolve_p2_reliability_fusion_config(config)
        if not self.config.enabled:
            raise ValueError("P2-v3 observer requires enabled configuration")
        if not self.config.collect_diagnostics:
            raise ValueError("P2-v3 requires collect_diagnostics")
        checkpoint_sha = str(parent_p2_checkpoint_sha256).strip().lower()
        if checkpoint_sha != "injected" and (
            len(checkpoint_sha) != 64
            or any(
                character not in "0123456789abcdef"
                for character in checkpoint_sha
            )
        ):
            raise ValueError("P2-v3 requires the frozen P2 checkpoint SHA")
        self.parent_p2_checkpoint_sha256 = checkpoint_sha
        self.scene_id: Optional[str] = None
        self.steps: list[P2ReliabilityFusionStep] = []

    def reset(self, scene_id: str) -> None:
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError("scene_id must be non-empty")
        self.scene_id = scene_id.strip()
        self.steps.clear()

    def observe(
        self,
        *,
        scene_id: str,
        p2v2_step: P2MaskGeometryStep,
    ) -> P2ReliabilityFusionStep:
        requested = str(scene_id).strip()
        if self.scene_id != requested:
            self.reset(requested)
        if not isinstance(p2v2_step, P2MaskGeometryStep):
            raise TypeError("p2v2_step has the wrong type")
        started = time.perf_counter()
        candidates = [
            _fuse_candidate(row, self.config)
            for row in p2v2_step.candidates
        ]
        if candidates:
            keep = stable_nms_aabb(
                np.stack([row.fused_box for row in candidates]),
                np.asarray([row.score for row in candidates]),
                self.config.step_nms_iou,
                tie_breakers=[row.candidate_id for row in candidates],
                max_output=self.config.max_candidates_per_step,
            )
            candidates = [candidates[int(index)] for index in keep]
        step = P2ReliabilityFusionStep(
            frame_index=int(p2v2_step.frame_index),
            provider_step=int(p2v2_step.provider_step),
            input_candidate_count=len(p2v2_step.candidates),
            eligible_candidate_count=len(p2v2_step.candidates),
            candidates=tuple(candidates),
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
        input_candidate_count: int,
        elapsed_seconds: float,
        error: Exception,
    ) -> None:
        requested = str(scene_id).strip()
        if self.scene_id != requested:
            self.reset(requested)
        self.steps.append(
            P2ReliabilityFusionStep(
                frame_index=int(frame_index),
                provider_step=int(provider_step),
                input_candidate_count=int(input_candidate_count),
                eligible_candidate_count=0,
                candidates=(),
                seconds=max(float(elapsed_seconds), 0.0),
                failed=True,
                error=f"{type(error).__name__}: {error}",
            )
        )
        del self.steps[: -self.config.max_history_steps]

    def scene_candidates(
        self,
    ) -> Tuple[P2ReliabilityFusedCandidate, ...]:
        rows = [candidate for step in self.steps for candidate in step.candidates]
        if not rows:
            return ()
        keep = stable_nms_aabb(
            np.stack([row.fused_box for row in rows]),
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
            "p2v3_schema": np.asarray(P2V3_DIAGNOSTIC_SCHEMA),
            "p2v3_stage": np.asarray("P2V3"),
            "p2v3_profile": np.asarray(P2V3_PROFILE),
            "p2v3_source": np.asarray(P2V3_SOURCE),
            "p2v3_enabled": np.asarray(True, dtype=bool),
            "p2v3_observer_only": np.asarray(True, dtype=bool),
            "p2v3_uses_ground_truth": np.asarray(False, dtype=bool),
            "p2v3_reads_semantic_labels": np.asarray(False, dtype=bool),
            "p2v3_mutation_enabled": np.asarray(False, dtype=bool),
            "p2v3_applied_count": np.asarray(0, dtype=np.int64),
            "p2v3_complete": np.asarray(
                bool(steps) and not any(step.failed for step in steps),
                dtype=bool,
            ),
            "p2v3_parent_p2_checkpoint_sha256": np.asarray(
                self.parent_p2_checkpoint_sha256
            ),
            "p2v3_parent_p2v2_schema": np.asarray(
                P2V2_DIAGNOSTIC_SCHEMA
            ),
            "p2v3_parent_p2v2_source": np.asarray(P2V2_SOURCE),
            "p2v3_reliability_contract": np.asarray(
                P2V3_RELIABILITY_CONTRACT
            ),
            "p2v3_config_json": np.asarray(config_json),
            "p2v3_step_frame_ids": np.asarray(
                [step.frame_index for step in steps], dtype=np.int64
            ),
            "p2v3_step_provider_steps": np.asarray(
                [step.provider_step for step in steps], dtype=np.int64
            ),
            "p2v3_step_input_candidate_counts": np.asarray(
                [step.input_candidate_count for step in steps],
                dtype=np.int64,
            ),
            "p2v3_step_eligible_candidate_counts": np.asarray(
                [step.eligible_candidate_count for step in steps],
                dtype=np.int64,
            ),
            "p2v3_step_candidate_counts": np.asarray(
                [len(step.candidates) for step in steps], dtype=np.int64
            ),
            "p2v3_step_seconds": np.asarray(
                [step.seconds for step in steps], dtype=np.float64
            ),
            "p2v3_step_failed": np.asarray(
                [step.failed for step in steps], dtype=bool
            ),
            "p2v3_step_errors": np.asarray(
                [step.error for step in steps], dtype=np.str_
            ),
            "p2v3_candidate_ids": values(
                "candidate_id", (0,), np.str_
            ),
            "p2v3_parent_p2v2_candidate_ids": values(
                "parent_p2v2_candidate_id", (0,), np.str_
            ),
            "p2v3_parent_p2_candidate_ids": values(
                "parent_p2_candidate_id", (0,), np.str_
            ),
            "p2v3_mask_source_ids": values(
                "mask_source_id", (0,), np.str_
            ),
            "p2v3_candidate_component_boxes": values(
                "component_box", (0, 6), np.float32
            ),
            "p2v3_candidate_component_corners": values(
                "component_corners", (0, 8, 3), np.float32
            ),
            "p2v3_candidate_parent_boxes": values(
                "parent_box", (0, 6), np.float32
            ),
            "p2v3_candidate_parent_corners": values(
                "parent_corners", (0, 8, 3), np.float32
            ),
            "p2v3_candidate_fused_boxes": values(
                "fused_box", (0, 6), np.float32
            ),
            "p2v3_candidate_fused_corners": values(
                "fused_corners", (0, 8, 3), np.float32
            ),
            "p2v3_candidate_scores": values(
                "score", (0,), np.float32
            ),
            "p2v3_candidate_component_weights": values(
                "component_weight", (0,), np.float32
            ),
            "p2v3_candidate_center_component_weights": values(
                "center_component_weights", (0, 3), np.float32
            ),
            "p2v3_candidate_extent_component_weights": values(
                "extent_component_weights", (0, 3), np.float32
            ),
            "p2v3_candidate_component_reliabilities": values(
                "component_reliability", (0,), np.float32
            ),
            "p2v3_candidate_parent_reliabilities": values(
                "parent_reliability", (0,), np.float32
            ),
            "p2v3_candidate_mask_reliabilities": values(
                "mask_reliability", (0,), np.float32
            ),
            "p2v3_candidate_depth_reliabilities": values(
                "depth_reliability", (0,), np.float32
            ),
            "p2v3_candidate_support_reliabilities": values(
                "support_reliability", (0,), np.float32
            ),
            "p2v3_candidate_agreement_reliabilities": values(
                "agreement_reliability", (0,), np.float32
            ),
            "p2v3_candidate_applied": np.zeros(
                len(candidates), dtype=bool
            ),
        }


__all__ = [
    "P2ReliabilityFusedCandidate",
    "P2ReliabilityFusionConfig",
    "P2ReliabilityFusionObserver",
    "P2ReliabilityFusionStep",
    "P2V3_DIAGNOSTIC_SCHEMA",
    "P2V3_PROFILE",
    "P2V3_RELIABILITY_CONTRACT",
    "P2V3_SOURCE",
    "resolve_p2_reliability_fusion_config",
]
