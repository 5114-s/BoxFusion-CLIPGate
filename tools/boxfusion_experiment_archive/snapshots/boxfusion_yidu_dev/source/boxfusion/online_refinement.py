"""Online orchestration for supplemental masks, RGB-D memory, and box refinement.

The controller in this module is intentionally external to BoxFusion's own
association state.  It observes the fused objects at each keyframe and changes
only the final exported detections.  This makes the feature opt-in and keeps a
disabled run on the exact legacy path.

Runtime inputs never include ground truth.  The only metric geometry source is
the sensor depth supplied by the caller.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from boxfusion.ap50_safety_gate import (
    AP50SafetyGate,
    AP50SafetyGateConfig,
    load_ap50_safety_gate,
)
from boxfusion.box_refiner import (
    BoxRefinerConfig,
    apply_box_residual_numpy,
    build_box_refiner,
)
from boxfusion.depth_occupancy_refiner import (
    DEFAULT_DEPTH_OCCUPANCY_REFINER_CONFIG,
    DepthOccupancyProposal,
    propose_depth_occupancy_refinement,
    resolve_depth_occupancy_refiner_config,
)
from boxfusion.fragment_stitch import (
    DEFAULT_FRAGMENT_STITCH_CONFIG,
    FragmentStitchCandidate,
    build_fragment_stitch_candidates,
    resolve_fragment_stitch_config,
)
from boxfusion.generic_local_geometry_refiner import (
    DEFAULT_GENERIC_LOCAL_GEOMETRY_CONFIG,
    GenericLocalGeometryProposal,
    propose_generic_local_geometry,
    resolve_generic_local_geometry_config,
)
from boxfusion.joint_local_head import (
    JOINT_LOCAL_HEAD_INPUT_SCHEMA,
    JOINT_LOCAL_HEAD_OUTPUT_SCHEMA,
    JOINT_VIEW_FEATURE_DIM,
    JointLocalHeadConfig,
    build_joint_local_head,
    prepare_joint_view_inputs,
)
from boxfusion.local_occupancy_msr_refiner import (
    DEFAULT_LOCAL_OCCUPANCY_MSR_CONFIG,
    OCCUPANCY_MSR_FEATURE_DIM,
    OCCUPANCY_MSR_FEATURE_NAMES,
    propose_local_occupancy_msr,
    resolve_local_occupancy_msr_config,
)
from boxfusion.mask_graph import (
    DEFAULT_MASK_GRAPH_CONFIG,
    MaskGraphEdge,
    MaskGraphNode,
    MaskGraphState,
    MaskGraphTrackEvidence,
    build_projection_context,
    coerce_mask_graph_node,
    coerce_track_evidence,
    evaluate_edge,
    resolve_mask_graph_config,
    update_mask_graph,
)
from boxfusion.missing_instance_graph import (
    DEFAULT_MISSING_INSTANCE_GRAPH_CONFIG,
    MISSING_GRAPH_FEATURE_NAMES,
    MISSING_INSTANCE_GRAPH_SCHEMA,
    MissingCandidateDecision,
    MissingInstanceGraphObserver,
    OrientedMissingCandidate,
    resolve_missing_instance_graph_config,
)
from boxfusion.object_memory import (
    CandidateTrackManager,
    MemoryViewRecord,
    ObjectGeometryMemory,
    ObjectObservation,
    aabb_corners,
    aabb_iou,
    deterministic_bounded_sample,
    extract_masked_world_points,
    points_inside_aabb,
    points_inside_aabb_fraction,
    project_aabb_to_image,
    projected_aabb_mask_iou,
    resolve_object_memory_config,
    robust_quantile_aabb,
)
from boxfusion.oriented_box_refiner import (
    OrientedBoxRefinerConfig,
    apply_local_box_residual_numpy,
    build_oriented_box_refiner,
)
from boxfusion.quality_score import (
    QUALITY_FEATURE_DIM,
    QUALITY_FEATURE_NAMES,
    load_quality_scorer,
    make_quality_scorer,
    quality_feature_vector,
    soft_nms_aabb_3d,
)
from boxfusion.raw_fused_query import (
    RAW_FUSED_QUERY_FEATURE_DIM,
    RAW_FUSED_QUERY_FEATURE_NAMES,
)
from boxfusion.supplemental_proposals import (
    ProposalProvider,
    SupplementalProposal,
    build_provider,
    resolve_supplemental_proposal_config,
)
from boxfusion.yidu_ablation import (
    YIDU_MODULES,
    YIDU_PROFILE_TO_STAGE,
    YIDU_SCHEMA,
    YIDU_STAGE_ADDED_MODULE,
    YIDU_STAGE_MODULE_MATRIX,
    YIDU_STAGE_TO_PROFILE,
)
from boxfusion.yidu_local_observer import (
    DEFAULT_YIDU_LOCAL_OBSERVER_CONFIG,
    YIDU_COMPONENT_FEATURE_DIM,
    YIDU_COMPONENT_FEATURE_NAMES,
    YIDU_GATE_FEATURE_DIM,
    YIDU_GATE_FEATURE_NAMES,
    YIDU_LOCAL_OBSERVER_SCHEMA,
    observe_yidu_local_geometry,
    resolve_yidu_local_observer_config,
)


TRIFUSION_GATE_FEATURE_NAMES = tuple(
    f"b6_original_{name}" for name in QUALITY_FEATURE_NAMES
) + ("b6_original_features_available",) + tuple(
    f"occupancy_msr_{name}"
    for name in OCCUPANCY_MSR_FEATURE_NAMES
)
TRIFUSION_GATE_FEATURE_DIM = len(TRIFUSION_GATE_FEATURE_NAMES)
assert TRIFUSION_GATE_FEATURE_DIM == (
    QUALITY_FEATURE_DIM + 1 + OCCUPANCY_MSR_FEATURE_DIM
)


_OBSERVER_ZERO_WRITE_ARRAY_FIELDS: Tuple[str, ...] = (
    "corners",
    "boxes",
    "scores",
    "source_indices",
    "stable_ids",
    "quality_features",
    "refit_original_boxes",
    "refit_original_corners",
    "refit_applied",
    "refit_changed_axes",
    "refit_boundary_delta",
    "refit_local_original_boxes",
    "refit_local_candidate_boxes",
    "refit_local_basis",
    "refit_local_frame_valid",
)


def _observer_zero_write_snapshot(
    arrays: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Copy the formal output arrays before read-only observers run."""

    missing = [
        name
        for name in _OBSERVER_ZERO_WRITE_ARRAY_FIELDS
        if name not in arrays
    ]
    if missing:
        raise RuntimeError(
            "observer zero-write snapshot is missing formal output arrays: "
            + ", ".join(missing)
        )
    return {
        name: np.array(
            np.asarray(arrays[name]),
            copy=True,
            order="C",
        )
        for name in _OBSERVER_ZERO_WRITE_ARRAY_FIELDS
    }


def _observer_zero_write_sha256(
    arrays: Mapping[str, np.ndarray],
) -> str:
    """Hash array name, dtype, shape, and logical C-order value bytes."""

    digest = hashlib.sha256()

    def update_token(value: bytes) -> None:
        digest.update(len(value).to_bytes(8, byteorder="little"))
        digest.update(value)

    for name in _OBSERVER_ZERO_WRITE_ARRAY_FIELDS:
        array = np.asarray(arrays[name])
        update_token(name.encode("utf-8"))
        update_token(array.dtype.str.encode("ascii"))
        update_token(
            json.dumps(
                list(array.shape), separators=(",", ":")
            ).encode("ascii")
        )
        update_token(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _observer_zero_write_changed_fields(
    snapshot: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
) -> Tuple[str, ...]:
    """Return formal arrays whose post-observer bytes differ exactly."""

    changed: List[str] = []
    for name in _OBSERVER_ZERO_WRITE_ARRAY_FIELDS:
        before = np.asarray(snapshot[name])
        after = np.asarray(arrays[name])
        if (
            before.shape != after.shape
            or before.dtype != after.dtype
            or before.tobytes(order="C")
            != np.ascontiguousarray(after).tobytes(order="C")
        ):
            changed.append(name)
    return tuple(changed)


def _empty_observer_zero_write_audit() -> Dict[str, Any]:
    return {
        "enabled": False,
        "verified": False,
        "pre_sha256": "",
        "post_sha256": "",
        "array_names": _OBSERVER_ZERO_WRITE_ARRAY_FIELDS,
        "changed_fields": (),
    }


DEFAULT_ONLINE_REFINEMENT_CONFIG: Dict[str, Any] = {
    "enabled": False,
    # Set by ``apply_online_ablation_profile``.  Keeping it in the resolved
    # config makes diagnostics self-describing instead of relying on a shell
    # command that may no longer be available when a dataset is built.
    "ablation_profile": "config-default",
    "scannet_axis_aligned_only": True,
    "inference_every_keyframes": 1,
    "candidate_lifecycle": {
        # Legacy-compatible default.  Stage-3 configs explicitly switch this
        # to ``provider_call`` so TTL counts missed proposal-provider updates.
        "ttl_clock": "keyframe",
        "archive_confirmed": False,
    },
    "supplemental_proposals": {},
    "appearance_memory": {
        "enabled": True,
        "masked_crop": True,
    },
    "object_memory": {},
    "matching": {
        "global_match_iou": 0.05,
        "global_match_2d_iou": 0.20,
        "max_center_distance": 0.75,
        "crop_to_global_expansion": 1.50,
        "rekey_iou": 0.50,
        "absorb_supplemental_iou": 0.35,
    },
    "missing_track_identity": {
        # A weak global match may be a nearby object swallowed by a coarse
        # BoxFusion box.  Shadow routing keeps such proposals in the missing
        # branch while still recording them as global evidence.  Final 3D
        # deduplication remains the hard guard against duplicate output.
        "enabled": False,
        "shadow_weak_global_matches": True,
        "strong_global_iou": 0.25,
        "strong_projection_iou": 0.50,
        "strong_point_support": 0.60,
    },
    "mask_graph": DEFAULT_MASK_GRAPH_CONFIG,
    # C3 observes cross-lifecycle Mask Graph fragments only after the graph
    # and C1/C2 contracts are frozen.  It is deliberately non-mutating.
    "fragment_stitch": DEFAULT_FRAGMENT_STITCH_CONFIG,
    "refit": {
        "enabled": True,
        # ``quantile_blend`` is the legacy all-surface AABB refit.
        # ``visibility_aware`` updates an axis only when selected cameras
        # observe that axis from both sides.
        "strategy": "quantile_blend",
        # B3-v2 performs visibility reasoning in the upstream oriented box's
        # local frame and restores the original orientation on export.
        "preserve_box_orientation": False,
        "min_views": 2,
        "min_points": 192,
        "blend": 0.70,
        "extent_padding": 0.02,
        "minimum_view_separation_degrees": 0.0,
        "minimum_axis_cosine": 0.0,
        "minimum_bilateral_axes": 1,
        "minimum_side_views": 1,
        "max_boundary_shift_ratio": 1.0,
        "minimum_boundary_change_ratio": 0.0,
        "visibility_boundary_quantile": 0.02,
        "visibility_point_crop_expansion": 1.20,
        "minimum_camera_outside_ratio": 0.0,
        "maximum_boundary_measurement_spread_ratio": 1.0,
        "enable_silhouette_axes": False,
        "select_best_silhouette_pair": False,
        "maximum_silhouette_axis_cosine": 0.0,
        "minimum_silhouette_views": 2,
        "minimum_silhouette_separation_degrees": 0.0,
        "max_center_shift_ratio": 0.60,
        "min_extent_ratio": 0.35,
        "max_extent_ratio": 2.50,
        "min_original_point_support": 0.20,
        "min_candidate_point_support": 0.0,
        "max_candidate_support_drop": 1.0,
        "min_reprojection_iou": 0.15,
        "min_reprojection_improvement": -0.02,
    },
    "box_refiner": {
        "enabled": False,
        "checkpoint": None,
        "device": None,
        # ``world_aabb`` retains the legacy refiner.  B5-v2 uses the
        # upstream OBB's local frame and restores its original basis/yaw.
        "coordinate_frame": "world_aabb",
        "preserve_orientation": False,
        # Legacy B5 profiles refine global detections. Missing-track profiles
        # can restrict B5 to graph-confirmed supplemental rows.
        "apply_scope": "global",
        "point_count": 512,
        "min_quality": 0.20,
        # B5-v2 predicts P(candidate improves evaluator IoU).  Keep
        # ``min_quality`` above as the legacy world-AABB threshold.
        "quality_threshold": 0.50,
        "architecture": {},
    },
    "joint_local_head": {
        # B3 -> B5 + B6-v2 is a strict alternative to all legacy final-output
        # mutations.  The observer may collect exact inputs with enabled=false
        # and collect_diagnostics=true, without loading PyTorch/checkpoints.
        "enabled": False,
        "checkpoint": None,
        "device": None,
        "max_views": 5,
        "points_per_view": 128,
        "improvement_threshold": 0.50,
        "max_candidate_uncertainty": 1.0,
        "detector_blend": 0.60,
        "preserve_original_floor": False,
        "mutate_geometry": False,
        "mutate_scores": False,
        "collect_diagnostics": False,
        "architecture": {},
    },
    "quality": {
        "enabled": True,
        "mode": "heuristic",
        "checkpoint": None,
        # Learned B6 checkpoints were trained on the unrefined BoxFusion
        # geometry.  B3-v2 can request ``original`` for a causal geometry-only
        # combination while legacy profiles retain ``refined``.
        "feature_geometry": "refined",
        # Optional strict input-contract override for legacy B6 checkpoints.
        # The released B6 training diagnostics used 0.5 for refiner_quality
        # on every row.  Geometry-only refiners must not inject their learned
        # gate confidence into that effectively constant feature.
        "refiner_quality_override": None,
        "blend_with_detector": 0.60,
        "preserve_original_floor": False,
        "apply_to_unobserved": False,
        "apply_to_supplemental": True,
        "support_reference_points": 8192,
        "target_views": 3,
        "max_view_records": 5,
        "soft_nms": {
            "enabled": True,
            "method": "gaussian",
            "iou_threshold": 0.30,
            "sigma": 0.50,
            "score_threshold": 0.05,
            "max_detections": None,
        },
    },
    "supplemental_output": {
        "enabled": True,
        "min_confirmations": 2,
        "min_score": 0.15,
        # Backward-compatible default.  Conservative B1 experiments require
        # the final 3D AABB to agree with its stored multi-view 2D boxes.
        "min_projection_iou": 0.0,
        "drop_if_global_iou": 0.70,
        "drop_if_supplemental_iou": 0.70,
        "require_mask_graph_confirmation": False,
        # ``None`` preserves the legacy contract: supplemental rows use the
        # global output extent filter. Missing-track profiles set a dedicated
        # threshold so observer/global detections remain exact identity.
        "minimum_extent": None,
        # C1-only lifecycle recovery.  The default is deliberately disabled
        # so every released observer/supplemental/B5/B6 profile retains its
        # exact candidate set.
        "recover_absorbed_confirmed": False,
        # A single uniform minimum extent removes legitimate thin doors and
        # small sinks.  C1 replaces it only for explicitly listed semantic
        # classes; unknown labels still use ``minimum_extent``.
        "class_aware_extent": False,
        "planar_extent_labels": ("door", "window"),
        "planar_min_extent": 0.04,
        "planar_middle_extent": 0.50,
        "planar_max_extent": 0.50,
        "small_extent_labels": ("sink",),
        "small_min_extent": 0.12,
        "small_middle_extent": 0.20,
        "small_max_extent": 0.30,
        # Full 3D IoU misses a duplicate whose height/depth is noisy but whose
        # floor-plane footprint is already occupied by a global detection.
        "bev_duplicate_enabled": False,
        "bev_duplicate_iou": 0.50,
        "bev_duplicate_containment": 0.80,
        "bev_duplicate_min_z_containment": 0.25,
        # Same-label thin tracks can fragment into two overlapping doors or
        # windows even when neither overlaps a global box strongly.
        "planar_duplicate_enabled": False,
        "planar_duplicate_bev_iou": 0.35,
        "planar_duplicate_containment": 0.70,
        "planar_duplicate_min_z_containment": 0.50,
        # Missing-track geometry is weaker than an upstream BoxFusion row.
        # C1 maps every supplemental row into a fixed score band below the
        # score-preserving 0.40 global anchor.  A projection term provides a
        # deterministic cross-scene quality rank without a learned head.
        "rank_after_globals": False,
        "rank_score_floor": 0.25,
        "rank_score_ceiling": 0.399,
        "rank_projection_weight": 0.50,
        "rank_recovered_bonus": 0.10,
    },
    "output_filter": {
        "minimum_extent": 0.0,
        # Populated by demo.py from the exact ScanNet final post-process
        # threshold. C1 uses it for global-reference and source-aware export
        # consistency; legacy profiles retain ``minimum_extent`` semantics.
        "final_minimum_extent": None,
    },
    # C2 is a deterministic, model-free child of C1.  It proposes geometry
    # from the bounded real Mask-RGBD point memory only after C1 has fixed the
    # supplemental candidate set, rank, and duplicate representatives.
    "supplemental_geometry_refiner": {
        "enabled": False,
        "mutate": False,
        "collect_diagnostics": False,
        "small_labels": ("sink",),
        "planar_labels": ("door", "window"),
        "proposal": dict(DEFAULT_DEPTH_OCCUPANCY_REFINER_CONFIG),
        "minimum_points": 256,
        "minimum_views": 3,
        "maximum_center_shift_ratio": 0.25,
        "minimum_extent_ratio": 0.65,
        "maximum_extent_ratio": 1.50,
        "minimum_absolute_projection_iou": 0.50,
        "minimum_projection_view_iou": 0.35,
        "minimum_projection_views": 2,
        "small_minimum_candidate_support": 0.95,
        "small_maximum_projection_drop": 0.18,
        "small_minimum_projection_views": 3,
        "planar_minimum_component_fraction": 0.20,
        "planar_minimum_density_ratio": 1.50,
        "planar_minimum_candidate_support": 0.40,
        "planar_minimum_view_point_support": 0.25,
        "planar_minimum_point_support_views": 2,
        "planar_maximum_projection_drop": 0.28,
        "refined_planar_minimum_extent": 0.034,
    },
    # C4 is a diagnostics-only geometry child of the frozen ``quality_only``
    # B6 anchor.  Its SAM3 proposal replay and Mask-RGBD memory are completely
    # independent of the primary YOLOE stream that forms B6 quality features.
    # This separation is what makes the observer bit-exact to the B6 output.
    "generic_local_geometry_refiner": {
        "enabled": False,
        "mutate": False,
        "collect_diagnostics": False,
        "fail_open": True,
        # When false, the detached provider is used only to decide which raw
        # masks remain unexplained by frozen B6.  No per-global C4 memory or
        # local-geometry candidate is built.
        "observe_existing_globals": True,
        "scope": "observed_global",
        "target_labels": (
            "chair",
            "desk",
            "table",
            "bookshelf",
            "cabinet",
            "bed",
            "sofa",
            "counter",
            "refrigerator",
            "toilet",
            "bathtub",
            "otherfurniture",
        ),
        "allow_unknown_label": False,
        # Filled by the dedicated runner.  Keeping this as an extension
        # mapping avoids hard-coding a machine-local teacher-cache path.
        "secondary_proposals": {},
        # Overrides are applied to a detached copy of the primary depth-memory
        # settings.  They never change ``self.object_config`` or B6 features.
        # This is an extension mapping because a resolved configuration is
        # intentionally accepted as controller input as well.  C4 defaults
        # are applied explicitly during validation before user overrides.
        "secondary_object_memory": {},
        "proposal": dict(DEFAULT_GENERIC_LOCAL_GEOMETRY_CONFIG),
        "minimum_mean_valid_depth_ratio": 0.65,
        "minimum_projection_views": 2,
        "minimum_projection_view_iou": 0.30,
        # Projection is deliberately a weak safety signal in the first
        # observer.  The full-100 audit will determine whether a positive
        # absolute floor generalizes before any active profile is released.
        "minimum_weighted_projection_iou": 0.0,
        "maximum_projection_drop": 0.03,
        "minimum_raw_candidate_support": 0.70,
        "maximum_raw_support_drop": 0.05,
        "maximum_center_shift_ratio": 0.15,
        "minimum_extent_ratio": 0.75,
        "maximum_extent_ratio": 1.25,
        "minimum_original_candidate_iou": 0.60,
        "maximum_overlap_increase": 0.10,
        "maximum_new_overlap": 0.70,
    },
    # TriFusion is a second observer layered on the isolated C4 per-global
    # memory.  It proposes an occupancy/MSR OBB and a fixed 48-D gate vector,
    # but has no mutation path by construction.
    "trifusion_observer": {
        "enabled": False,
        "mutate": False,
        "collect_diagnostics": False,
        "proposal": dict(DEFAULT_LOCAL_OCCUPANCY_MSR_CONFIG),
        # The missing-instance branch consumes only secondary SAM3 proposals
        # which did not associate with a frozen B6 global.  It is a separate
        # observer graph and can never enter supplemental output.
        "missing_instance_graph": dict(
            DEFAULT_MISSING_INSTANCE_GRAPH_CONFIG
        ),
        # M4 is optional and remains diagnostics-only.  A checkpoint may
        # classify a hard-verified M3 proposal, but cannot mutate geometry or
        # scores in this observer profile.
        "safety_gate": {
            "enabled": False,
            "checkpoint": None,
            "mutate": False,
            "collect_diagnostics": False,
            "minimum_improvement_probability": 0.75,
            "maximum_harm_probability": 0.15,
            "uncertainty_multiplier": 1.64,
            "maximum_delta_std": 0.15,
            "minimum_delta_lower_bound": 0.005,
            "minimum_predicted_iou_margin": 0.005,
            "require_iou50_crossing": False,
            "minimum_iou50_crossing_probability": 0.60,
        },
    },
    # One isolated proposal-recall experiment: raw masks which were not
    # claimed by frozen B6 are lifted with aligned sensor depth and associated
    # across views by ``MissingInstanceGraphObserver``.  It deliberately has
    # no output/supplemental mutation path.  ``source_mode`` can select the
    # immutable SAM3 teacher cache, the online YOLOE stream, or both sources
    # in one graph update per scheduled provider call.
    "residual_track_observer": {
        "enabled": False,
        "observer_only": True,
        "mutate": False,
        "collect_diagnostics": False,
        "source_mode": "sam3",
        "primary_provider_name": "yoloe",
        "secondary_provider_name": "sam3_teacher_cache",
        "missing_instance_graph": dict(
            DEFAULT_MISSING_INSTANCE_GRAPH_CONFIG
        ),
    },
    # Strictly cumulative YiDu observers.  Profiles B0->A6 change exactly one
    # module at a time, never expose a mutation switch, and reuse the isolated
    # C4 SAM3 evidence stream rather than the primary B6 memory.
    "yidu_ablation": {
        "schema": YIDU_SCHEMA,
        "stage": "B0",
        "profile": YIDU_STAGE_TO_PROFILE["B0"],
        "enabled": False,
        "observer_only": True,
        "mutate": False,
        "collect_diagnostics": False,
        "frozen_b6": True,
        "added_module": None,
        "modules": dict(YIDU_STAGE_MODULE_MATRIX["B0"]),
        "adaptive_erosion": {
            "enabled": False,
            "observer_only": True,
            "mutate": False,
        },
        "dfu_filter": {
            "enabled": False,
            "observer_only": True,
            "mutate": False,
        },
        "voxel_components": {
            "enabled": False,
            "observer_only": True,
            "mutate": False,
        },
        "occupancy_msr": {
            "enabled": False,
            "observer_only": True,
            "mutate": False,
        },
        "raw_fused_query": {
            "enabled": False,
            "observer_only": True,
            "mutate": False,
            "scorer_checkpoint": None,
        },
        "quality_gate": {
            "enabled": False,
            "observer_only": True,
            "mutate": False,
            "checkpoint": None,
        },
        "local_observer": dict(DEFAULT_YIDU_LOCAL_OBSERVER_CONFIG),
    },
    "diagnostics": {
        "enabled": False,
        "dump_track_memory": False,
        "root": None,
        "point_count": 512,
    },
}


def _deep_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _deep_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_deep_copy(item) for item in value)
    return value


def _merge_known(
    defaults: Mapping[str, Any],
    updates: Optional[Mapping[str, Any]],
    *,
    prefix: str,
) -> Dict[str, Any]:
    if updates is None:
        updates = {}
    if not isinstance(updates, Mapping):
        raise TypeError(f"{prefix} must be a mapping")
    # Empty mappings are deliberate extension points whose keys are validated
    # by the owning component (proposal provider, object memory, or model).
    if not defaults:
        return _deep_copy(updates)
    unknown = sorted(set(updates) - set(defaults))
    if unknown:
        raise ValueError(
            f"Unknown {prefix} key(s): " + ", ".join(str(key) for key in unknown)
        )
    output = _deep_copy(defaults)
    for key, value in updates.items():
        if isinstance(defaults[key], Mapping):
            output[key] = _merge_known(
                defaults[key],
                value,
                prefix=f"{prefix}.{key}",
            )
        else:
            output[key] = value
    return output


def _finite_float(
    config: Mapping[str, Any],
    key: str,
    *,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    strict_lower: bool = False,
) -> float:
    value = config[key]
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{key} must be a finite scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{key} must be finite")
    if lower is not None:
        invalid = result <= lower if strict_lower else result < lower
        if invalid:
            relation = "greater than" if strict_lower else "at least"
            raise ValueError(f"{key} must be {relation} {lower}")
    if upper is not None and result > upper:
        raise ValueError(f"{key} must be at most {upper}")
    return result


def _positive_int(config: Mapping[str, Any], key: str, minimum: int = 1) -> int:
    value = config[key]
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{key} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{key} must be at least {minimum}")
    return value


def resolve_online_refinement_config(
    cfg: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Resolve and validate the opt-in controller configuration.

    ``cfg`` may be the full BoxFusion configuration or the
    ``online_refinement`` subsection.  A disabled subsection is returned
    without validating or touching any optional provider/model settings.
    """

    if cfg is None:
        raw: Mapping[str, Any] = {}
    elif not isinstance(cfg, Mapping):
        raise TypeError("online refinement config must be a mapping")
    elif "online_refinement" in cfg:
        nested = cfg.get("online_refinement")
        if nested is None:
            raw = {}
        elif not isinstance(nested, Mapping):
            raise TypeError("online_refinement must be a mapping")
        else:
            raw = nested
    else:
        raw = cfg

    enabled = raw.get("enabled", False)
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("online_refinement.enabled must be Boolean")
    if not bool(enabled):
        disabled = _deep_copy(DEFAULT_ONLINE_REFINEMENT_CONFIG)
        disabled["enabled"] = False
        return disabled

    resolved = _merge_known(
        DEFAULT_ONLINE_REFINEMENT_CONFIG,
        raw,
        prefix="online_refinement",
    )
    resolved["enabled"] = True
    profile = resolved["ablation_profile"]
    if not isinstance(profile, str) or not profile.strip():
        raise ValueError("ablation_profile must be a non-empty string")
    resolved["ablation_profile"] = profile.strip()
    if not isinstance(
        resolved["scannet_axis_aligned_only"], (bool, np.bool_)
    ):
        raise ValueError("scannet_axis_aligned_only must be Boolean")
    resolved["scannet_axis_aligned_only"] = bool(
        resolved["scannet_axis_aligned_only"]
    )
    resolved["inference_every_keyframes"] = _positive_int(
        resolved, "inference_every_keyframes"
    )
    lifecycle = resolved["candidate_lifecycle"]
    lifecycle["ttl_clock"] = str(lifecycle["ttl_clock"]).strip().lower()
    if lifecycle["ttl_clock"] not in {"keyframe", "provider_call"}:
        raise ValueError(
            "candidate_lifecycle.ttl_clock must be keyframe or provider_call"
        )
    if not isinstance(lifecycle["archive_confirmed"], (bool, np.bool_)):
        raise ValueError(
            "candidate_lifecycle.archive_confirmed must be Boolean"
        )
    lifecycle["archive_confirmed"] = bool(
        lifecycle["archive_confirmed"]
    )
    appearance = resolved["appearance_memory"]
    for key in ("enabled", "masked_crop"):
        if not isinstance(appearance[key], (bool, np.bool_)):
            raise ValueError(f"appearance_memory.{key} must be Boolean")
        appearance[key] = bool(appearance[key])

    matching = resolved["matching"]
    for key in (
        "global_match_iou",
        "global_match_2d_iou",
        "rekey_iou",
        "absorb_supplemental_iou",
    ):
        matching[key] = _finite_float(matching, key, lower=0.0, upper=1.0)
    matching["max_center_distance"] = _finite_float(
        matching, "max_center_distance", lower=0.0, strict_lower=True
    )
    matching["crop_to_global_expansion"] = _finite_float(
        matching, "crop_to_global_expansion", lower=1.0
    )
    missing_identity = resolved["missing_track_identity"]
    for key in ("enabled", "shadow_weak_global_matches"):
        if not isinstance(missing_identity[key], (bool, np.bool_)):
            raise ValueError(f"missing_track_identity.{key} must be Boolean")
        missing_identity[key] = bool(missing_identity[key])
    for key in (
        "strong_global_iou",
        "strong_projection_iou",
        "strong_point_support",
    ):
        missing_identity[key] = _finite_float(
            missing_identity, key, lower=0.0, upper=1.0
        )
    resolved["mask_graph"] = resolve_mask_graph_config(
        resolved["mask_graph"]
    )
    if (
        resolved["mask_graph"]["enabled"]
        and not missing_identity["enabled"]
    ):
        raise ValueError(
            "enabled mask_graph requires missing_track_identity.enabled"
        )
    resolved["fragment_stitch"] = resolve_fragment_stitch_config(
        resolved["fragment_stitch"]
    )
    if (
        resolved["fragment_stitch"]["enabled"]
        and not resolved["mask_graph"]["enabled"]
    ):
        raise ValueError(
            "enabled fragment_stitch requires an enabled mask_graph"
        )

    refit = resolved["refit"]
    if not isinstance(refit["enabled"], (bool, np.bool_)):
        raise ValueError("refit.enabled must be Boolean")
    refit["enabled"] = bool(refit["enabled"])
    refit["strategy"] = str(refit["strategy"]).strip().lower()
    if refit["strategy"] not in {
        "quantile_blend",
        "visibility_aware",
    }:
        raise ValueError(
            "refit.strategy must be quantile_blend or visibility_aware"
        )
    if not isinstance(
        refit["preserve_box_orientation"], (bool, np.bool_)
    ):
        raise ValueError("preserve_box_orientation must be Boolean")
    refit["preserve_box_orientation"] = bool(
        refit["preserve_box_orientation"]
    )
    refit["min_views"] = _positive_int(refit, "min_views")
    refit["min_points"] = _positive_int(refit, "min_points")
    refit["minimum_bilateral_axes"] = _positive_int(
        refit, "minimum_bilateral_axes"
    )
    if refit["minimum_bilateral_axes"] > 3:
        raise ValueError("minimum_bilateral_axes must be at most 3")
    refit["minimum_side_views"] = _positive_int(
        refit, "minimum_side_views"
    )
    for key in (
        "blend",
        "min_original_point_support",
        "min_candidate_point_support",
        "min_reprojection_iou",
    ):
        refit[key] = _finite_float(refit, key, lower=0.0, upper=1.0)
    refit["minimum_view_separation_degrees"] = _finite_float(
        refit,
        "minimum_view_separation_degrees",
        lower=0.0,
        upper=180.0,
    )
    refit["minimum_axis_cosine"] = _finite_float(
        refit, "minimum_axis_cosine", lower=0.0, upper=1.0
    )
    refit["max_boundary_shift_ratio"] = _finite_float(
        refit, "max_boundary_shift_ratio", lower=0.0, upper=1.0
    )
    refit["minimum_boundary_change_ratio"] = _finite_float(
        refit,
        "minimum_boundary_change_ratio",
        lower=0.0,
        upper=1.0,
    )
    refit["visibility_boundary_quantile"] = _finite_float(
        refit,
        "visibility_boundary_quantile",
        lower=0.0,
        upper=0.5,
    )
    if refit["visibility_boundary_quantile"] >= 0.5:
        raise ValueError("visibility_boundary_quantile must be below 0.5")
    refit["visibility_point_crop_expansion"] = _finite_float(
        refit,
        "visibility_point_crop_expansion",
        lower=1.0,
    )
    refit["minimum_camera_outside_ratio"] = _finite_float(
        refit,
        "minimum_camera_outside_ratio",
        lower=0.0,
        upper=1.0,
    )
    refit["maximum_boundary_measurement_spread_ratio"] = _finite_float(
        refit,
        "maximum_boundary_measurement_spread_ratio",
        lower=0.0,
        upper=1.0,
    )
    if not isinstance(refit["enable_silhouette_axes"], (bool, np.bool_)):
        raise ValueError("enable_silhouette_axes must be Boolean")
    refit["enable_silhouette_axes"] = bool(
        refit["enable_silhouette_axes"]
    )
    if not isinstance(
        refit["select_best_silhouette_pair"], (bool, np.bool_)
    ):
        raise ValueError("select_best_silhouette_pair must be Boolean")
    refit["select_best_silhouette_pair"] = bool(
        refit["select_best_silhouette_pair"]
    )
    refit["maximum_silhouette_axis_cosine"] = _finite_float(
        refit,
        "maximum_silhouette_axis_cosine",
        lower=0.0,
        upper=1.0,
    )
    refit["minimum_silhouette_views"] = _positive_int(
        refit, "minimum_silhouette_views"
    )
    refit["minimum_silhouette_separation_degrees"] = _finite_float(
        refit,
        "minimum_silhouette_separation_degrees",
        lower=0.0,
        upper=180.0,
    )
    refit["max_candidate_support_drop"] = _finite_float(
        refit,
        "max_candidate_support_drop",
        lower=0.0,
        upper=1.0,
    )
    refit["extent_padding"] = _finite_float(
        refit, "extent_padding", lower=0.0
    )
    refit["max_center_shift_ratio"] = _finite_float(
        refit, "max_center_shift_ratio", lower=0.0, strict_lower=True
    )
    refit["min_extent_ratio"] = _finite_float(
        refit, "min_extent_ratio", lower=0.0, strict_lower=True
    )
    refit["max_extent_ratio"] = _finite_float(
        refit, "max_extent_ratio", lower=0.0, strict_lower=True
    )
    if refit["max_extent_ratio"] < refit["min_extent_ratio"]:
        raise ValueError("refit max_extent_ratio must exceed min_extent_ratio")
    refit["min_reprojection_improvement"] = _finite_float(
        refit, "min_reprojection_improvement"
    )

    box_refiner = resolved["box_refiner"]
    if not isinstance(box_refiner["enabled"], (bool, np.bool_)):
        raise ValueError("box_refiner.enabled must be Boolean")
    box_refiner["enabled"] = bool(box_refiner["enabled"])
    box_refiner["coordinate_frame"] = str(
        box_refiner["coordinate_frame"]
    ).strip().lower()
    if box_refiner["coordinate_frame"] not in {
        "world_aabb",
        "box_local",
    }:
        raise ValueError(
            "box_refiner.coordinate_frame must be world_aabb or box_local"
        )
    box_refiner["apply_scope"] = str(
        box_refiner["apply_scope"]
    ).strip().lower()
    if box_refiner["apply_scope"] not in {
        "none",
        "global",
        "confirmed_supplemental",
        "global_and_confirmed_supplemental",
    }:
        raise ValueError(
            "box_refiner.apply_scope must be none, global, "
            "confirmed_supplemental, or global_and_confirmed_supplemental"
        )
    if not isinstance(
        box_refiner["preserve_orientation"], (bool, np.bool_)
    ):
        raise ValueError("box_refiner.preserve_orientation must be Boolean")
    box_refiner["preserve_orientation"] = bool(
        box_refiner["preserve_orientation"]
    )
    if (
        box_refiner["coordinate_frame"] == "box_local"
        and not box_refiner["preserve_orientation"]
    ):
        raise ValueError(
            "box_local refinement must preserve the original orientation"
        )
    if (
        box_refiner["coordinate_frame"] == "world_aabb"
        and box_refiner["preserve_orientation"]
    ):
        raise ValueError(
            "preserve_orientation requires coordinate_frame=box_local"
        )
    box_refiner["point_count"] = _positive_int(
        box_refiner, "point_count"
    )
    box_refiner["min_quality"] = _finite_float(
        box_refiner, "min_quality", lower=0.0, upper=1.0
    )
    box_refiner["quality_threshold"] = _finite_float(
        box_refiner, "quality_threshold", lower=0.0, upper=1.0
    )
    if not isinstance(box_refiner["architecture"], Mapping):
        raise TypeError("box_refiner.architecture must be a mapping")
    if (
        refit["preserve_box_orientation"]
        and refit["strategy"] != "visibility_aware"
    ):
        raise ValueError(
            "preserve_box_orientation requires visibility_aware refit"
        )
    if (
        refit["preserve_box_orientation"]
        and refit["enabled"]
        and box_refiner["enabled"]
    ):
        raise ValueError(
            "oriented visibility refit cannot be combined with the "
            "neural box refiner in the same ablation"
        )

    joint = resolved["joint_local_head"]
    for key in (
        "enabled",
        "mutate_geometry",
        "mutate_scores",
        "collect_diagnostics",
        "preserve_original_floor",
    ):
        if not isinstance(joint[key], (bool, np.bool_)):
            raise ValueError(f"joint_local_head.{key} must be Boolean")
        joint[key] = bool(joint[key])
    joint["max_views"] = _positive_int(joint, "max_views")
    joint["points_per_view"] = _positive_int(
        joint, "points_per_view"
    )
    joint["improvement_threshold"] = _finite_float(
        joint,
        "improvement_threshold",
        lower=0.0,
        upper=1.0,
    )
    joint["max_candidate_uncertainty"] = _finite_float(
        joint,
        "max_candidate_uncertainty",
        lower=0.0,
        strict_lower=True,
    )
    joint["detector_blend"] = _finite_float(
        joint, "detector_blend", lower=0.0, upper=1.0
    )
    if joint["device"] is not None and (
        not isinstance(joint["device"], str)
        or not joint["device"].strip()
    ):
        raise ValueError(
            "joint_local_head.device must be null or a non-empty string"
        )
    if joint["device"] is not None:
        joint["device"] = joint["device"].strip()
    if not isinstance(joint["architecture"], Mapping):
        raise TypeError("joint_local_head.architecture must be a mapping")
    # Validate the exact architecture before any checkpoint or optional torch
    # work.  Strict checkpoint loading repeats this contract at model load.
    JointLocalHeadConfig(
        **dict(joint["architecture"])
    ).validated()
    if not joint["enabled"] and (
        joint["mutate_geometry"] or joint["mutate_scores"]
    ):
        raise ValueError(
            "joint_local_head mutations require joint_local_head.enabled"
        )
    if joint["enabled"] and not (
        joint["mutate_geometry"] or joint["mutate_scores"]
    ):
        raise ValueError(
            "enabled joint_local_head must mutate geometry or scores"
        )

    quality = resolved["quality"]
    for key in (
        "enabled",
        "preserve_original_floor",
        "apply_to_unobserved",
        "apply_to_supplemental",
    ):
        if not isinstance(quality[key], (bool, np.bool_)):
            raise ValueError(f"quality.{key} must be Boolean")
        quality[key] = bool(quality[key])
    quality["mode"] = str(quality["mode"]).strip().lower()
    if quality["mode"] not in {"heuristic", "linear", "mlp", "iou_mlp"}:
        raise ValueError(
            "quality.mode must be heuristic, linear, mlp, or iou_mlp"
        )
    quality["feature_geometry"] = str(
        quality["feature_geometry"]
    ).strip().lower()
    if quality["feature_geometry"] not in {"original", "refined"}:
        raise ValueError(
            "quality.feature_geometry must be original or refined"
        )
    if quality["refiner_quality_override"] is not None:
        quality["refiner_quality_override"] = _finite_float(
            quality,
            "refiner_quality_override",
            lower=0.0,
            upper=1.0,
        )
    quality["blend_with_detector"] = _finite_float(
        quality, "blend_with_detector", lower=0.0, upper=1.0
    )
    quality["support_reference_points"] = _positive_int(
        quality, "support_reference_points"
    )
    quality["target_views"] = _positive_int(quality, "target_views")
    quality["max_view_records"] = _positive_int(
        quality, "max_view_records"
    )
    soft_nms = quality["soft_nms"]
    if not isinstance(soft_nms["enabled"], (bool, np.bool_)):
        raise ValueError("quality.soft_nms.enabled must be Boolean")
    soft_nms["enabled"] = bool(soft_nms["enabled"])
    soft_nms["method"] = str(soft_nms["method"]).strip().lower()
    if soft_nms["method"] not in {"linear", "gaussian", "hard"}:
        raise ValueError("quality.soft_nms.method is invalid")
    for key in ("iou_threshold", "score_threshold"):
        soft_nms[key] = _finite_float(
            soft_nms, key, lower=0.0, upper=1.0
        )
    soft_nms["sigma"] = _finite_float(
        soft_nms, "sigma", lower=0.0, strict_lower=True
    )
    if soft_nms["max_detections"] is not None:
        soft_nms["max_detections"] = _positive_int(
            soft_nms, "max_detections"
        )

    supplemental = resolved["supplemental_output"]
    for key in (
        "enabled",
        "require_mask_graph_confirmation",
        "recover_absorbed_confirmed",
        "class_aware_extent",
        "bev_duplicate_enabled",
        "planar_duplicate_enabled",
        "rank_after_globals",
    ):
        if not isinstance(supplemental[key], (bool, np.bool_)):
            raise ValueError(f"supplemental_output.{key} must be Boolean")
        supplemental[key] = bool(supplemental[key])
    supplemental["min_confirmations"] = _positive_int(
        supplemental, "min_confirmations", minimum=2
    )
    if supplemental["minimum_extent"] is not None:
        supplemental["minimum_extent"] = _finite_float(
            supplemental, "minimum_extent", lower=0.0
        )
    for key in (
        "min_score",
        "min_projection_iou",
        "drop_if_global_iou",
        "drop_if_supplemental_iou",
    ):
        supplemental[key] = _finite_float(
            supplemental, key, lower=0.0, upper=1.0
        )
    for key in ("planar_extent_labels", "small_extent_labels"):
        labels_value = supplemental[key]
        if not isinstance(labels_value, (list, tuple)):
            raise ValueError(f"supplemental_output.{key} must be a sequence")
        normalized_labels = []
        for label in labels_value:
            if not isinstance(label, str) or not label.strip():
                raise ValueError(
                    f"supplemental_output.{key} entries must be strings"
                )
            normalized_labels.append(label.strip().casefold())
        if len(set(normalized_labels)) != len(normalized_labels):
            raise ValueError(
                f"supplemental_output.{key} contains duplicate labels"
            )
        supplemental[key] = tuple(normalized_labels)
    if set(supplemental["planar_extent_labels"]) & set(
        supplemental["small_extent_labels"]
    ):
        raise ValueError(
            "planar_extent_labels and small_extent_labels must be disjoint"
        )
    for prefix in ("planar", "small"):
        thresholds = []
        for suffix in ("min_extent", "middle_extent", "max_extent"):
            key = f"{prefix}_{suffix}"
            thresholds.append(
                _finite_float(supplemental, key, lower=0.0)
            )
        if thresholds != sorted(thresholds):
            raise ValueError(
                f"supplemental_output.{prefix} extent thresholds must be "
                "non-decreasing"
            )
    for key in (
        "bev_duplicate_iou",
        "bev_duplicate_containment",
        "bev_duplicate_min_z_containment",
        "planar_duplicate_bev_iou",
        "planar_duplicate_containment",
        "planar_duplicate_min_z_containment",
        "rank_score_floor",
        "rank_score_ceiling",
        "rank_projection_weight",
        "rank_recovered_bonus",
    ):
        supplemental[key] = _finite_float(
            supplemental, key, lower=0.0, upper=1.0
        )
    if (
        supplemental["rank_score_floor"]
        >= supplemental["rank_score_ceiling"]
    ):
        raise ValueError(
            "supplemental_output rank_score_floor must be below "
            "rank_score_ceiling"
        )

    output_filter = resolved["output_filter"]
    output_filter["minimum_extent"] = _finite_float(
        output_filter, "minimum_extent", lower=0.0
    )
    if output_filter["final_minimum_extent"] is not None:
        output_filter["final_minimum_extent"] = _finite_float(
            output_filter, "final_minimum_extent", lower=0.0
        )

    supplemental_geometry = resolved[
        "supplemental_geometry_refiner"
    ]
    for key in ("enabled", "mutate", "collect_diagnostics"):
        if not isinstance(
            supplemental_geometry[key], (bool, np.bool_)
        ):
            raise ValueError(
                f"supplemental_geometry_refiner.{key} must be Boolean"
            )
        supplemental_geometry[key] = bool(
            supplemental_geometry[key]
        )
    if (
        supplemental_geometry["mutate"]
        and not supplemental_geometry["enabled"]
    ):
        raise ValueError(
            "supplemental_geometry_refiner.mutate requires enabled"
        )
    for key in ("small_labels", "planar_labels"):
        labels_value = supplemental_geometry[key]
        if not isinstance(labels_value, (list, tuple)):
            raise ValueError(
                f"supplemental_geometry_refiner.{key} "
                "must be a sequence"
            )
        normalized_labels = []
        for label in labels_value:
            if not isinstance(label, str) or not label.strip():
                raise ValueError(
                    f"supplemental_geometry_refiner.{key} entries "
                    "must be strings"
                )
            normalized_labels.append(label.strip().casefold())
        if len(set(normalized_labels)) != len(normalized_labels):
            raise ValueError(
                f"supplemental_geometry_refiner.{key} contains "
                "duplicate labels"
            )
        supplemental_geometry[key] = tuple(normalized_labels)
    if set(supplemental_geometry["small_labels"]) & set(
        supplemental_geometry["planar_labels"]
    ):
        raise ValueError(
            "supplemental_geometry_refiner small_labels and "
            "planar_labels must be disjoint"
        )
    supplemental_geometry["proposal"] = (
        resolve_depth_occupancy_refiner_config(
            supplemental_geometry["proposal"]
        )
    )
    for key in (
        "minimum_points",
        "minimum_views",
        "minimum_projection_views",
        "small_minimum_projection_views",
        "planar_minimum_point_support_views",
    ):
        supplemental_geometry[key] = _positive_int(
            supplemental_geometry, key
        )
    for key in (
        "maximum_center_shift_ratio",
        "minimum_extent_ratio",
        "minimum_absolute_projection_iou",
        "minimum_projection_view_iou",
        "small_minimum_candidate_support",
        "small_maximum_projection_drop",
        "planar_minimum_component_fraction",
        "planar_minimum_candidate_support",
        "planar_minimum_view_point_support",
        "planar_maximum_projection_drop",
    ):
        supplemental_geometry[key] = _finite_float(
            supplemental_geometry, key, lower=0.0, upper=1.0
        )
    supplemental_geometry["maximum_extent_ratio"] = _finite_float(
        supplemental_geometry,
        "maximum_extent_ratio",
        lower=1.0,
    )
    supplemental_geometry["planar_minimum_density_ratio"] = (
        _finite_float(
            supplemental_geometry,
            "planar_minimum_density_ratio",
            lower=1.0,
        )
    )
    supplemental_geometry["refined_planar_minimum_extent"] = (
        _finite_float(
            supplemental_geometry,
            "refined_planar_minimum_extent",
            lower=0.0,
            strict_lower=True,
        )
    )
    if (
        supplemental_geometry["maximum_extent_ratio"]
        < supplemental_geometry["minimum_extent_ratio"]
    ):
        raise ValueError(
            "supplemental_geometry_refiner maximum_extent_ratio "
            "must be at least minimum_extent_ratio"
        )
    if supplemental_geometry["enabled"]:
        if not supplemental["enabled"]:
            raise ValueError(
                "enabled supplemental_geometry_refiner requires "
                "supplemental_output.enabled"
            )
        if not supplemental["class_aware_extent"]:
            raise ValueError(
                "enabled supplemental_geometry_refiner requires the "
                "C1 class-aware supplemental contract"
            )
        if (
            supplemental_geometry["mutate"]
            and box_refiner["enabled"]
            and box_refiner["apply_scope"]
            in {
                "confirmed_supplemental",
                "global_and_confirmed_supplemental",
            }
        ):
            raise ValueError(
                "mutating supplemental_geometry_refiner cannot be "
                "combined with a supplemental box_refiner"
            )

    generic_geometry = resolved["generic_local_geometry_refiner"]
    for key in (
        "enabled",
        "mutate",
        "collect_diagnostics",
        "fail_open",
        "observe_existing_globals",
        "allow_unknown_label",
    ):
        if not isinstance(generic_geometry[key], (bool, np.bool_)):
            raise ValueError(
                f"generic_local_geometry_refiner.{key} must be Boolean"
            )
        generic_geometry[key] = bool(generic_geometry[key])
    if generic_geometry["mutate"]:
        raise ValueError(
            "generic_local_geometry_refiner is observer-only; an active "
            "geometry profile must be released only after held-out gates "
            "have been frozen"
        )
    generic_geometry["scope"] = str(
        generic_geometry["scope"]
    ).strip().lower()
    if generic_geometry["scope"] != "observed_global":
        raise ValueError(
            "generic_local_geometry_refiner.scope must be observed_global"
        )
    labels_value = generic_geometry["target_labels"]
    if not isinstance(labels_value, (list, tuple)):
        raise ValueError(
            "generic_local_geometry_refiner.target_labels must be a sequence"
        )
    normalized_labels = []
    for label in labels_value:
        if not isinstance(label, str) or not label.strip():
            raise ValueError(
                "generic_local_geometry_refiner.target_labels entries "
                "must be non-empty strings"
            )
        normalized_labels.append(label.strip().casefold())
    if len(set(normalized_labels)) != len(normalized_labels):
        raise ValueError(
            "generic_local_geometry_refiner.target_labels contains "
            "duplicates"
        )
    generic_geometry["target_labels"] = tuple(normalized_labels)
    generic_geometry["proposal"] = resolve_generic_local_geometry_config(
        generic_geometry["proposal"]
    )
    generic_geometry["minimum_projection_views"] = _positive_int(
        generic_geometry, "minimum_projection_views"
    )
    for key in (
        "minimum_mean_valid_depth_ratio",
        "minimum_projection_view_iou",
        "minimum_weighted_projection_iou",
        "maximum_projection_drop",
        "minimum_raw_candidate_support",
        "maximum_raw_support_drop",
        "maximum_center_shift_ratio",
        "minimum_extent_ratio",
        "minimum_original_candidate_iou",
        "maximum_overlap_increase",
        "maximum_new_overlap",
    ):
        generic_geometry[key] = _finite_float(
            generic_geometry, key, lower=0.0, upper=1.0
        )
    generic_geometry["maximum_extent_ratio"] = _finite_float(
        generic_geometry,
        "maximum_extent_ratio",
        lower=1.0,
    )
    if (
        generic_geometry["maximum_extent_ratio"]
        < generic_geometry["minimum_extent_ratio"]
    ):
        raise ValueError(
            "generic_local_geometry_refiner maximum_extent_ratio must be "
            "at least minimum_extent_ratio"
        )
    secondary_proposals = resolve_supplemental_proposal_config(
        generic_geometry["secondary_proposals"]
    )
    generic_geometry["secondary_proposals"] = secondary_proposals
    secondary_memory_updates = generic_geometry[
        "secondary_object_memory"
    ]
    if not isinstance(secondary_memory_updates, Mapping):
        raise TypeError(
            "generic_local_geometry_refiner.secondary_object_memory must "
            "be a mapping"
        )
    secondary_memory = _deep_copy(resolved["object_memory"])
    secondary_memory.update(
        {
            "top_k_views": 5,
            "max_view_candidates": 12,
            "view_diversity_weight": 0.40,
            "max_points_per_object": 8192,
        }
    )
    secondary_memory.update(_deep_copy(secondary_memory_updates))
    generic_geometry["secondary_object_memory"] = (
        resolve_object_memory_config(secondary_memory)
    )
    if generic_geometry["enabled"]:
        if not generic_geometry["collect_diagnostics"]:
            raise ValueError(
                "enabled generic_local_geometry_refiner requires "
                "collect_diagnostics"
            )
        if not secondary_proposals["enabled"]:
            raise ValueError(
                "enabled generic_local_geometry_refiner requires an enabled "
                "secondary proposal provider"
            )
        if secondary_proposals["provider"] != "cache_only":
            raise ValueError(
                "generic_local_geometry_refiner secondary provider must be "
                "cache_only so the observer cannot add online model latency"
            )

    trifusion = resolved["trifusion_observer"]
    for key in ("enabled", "mutate", "collect_diagnostics"):
        if not isinstance(trifusion[key], (bool, np.bool_)):
            raise ValueError(
                f"trifusion_observer.{key} must be Boolean"
            )
        trifusion[key] = bool(trifusion[key])
    if trifusion["mutate"]:
        raise ValueError(
            "trifusion_observer is strictly observer-only; mutate must be "
            "false"
        )
    trifusion["proposal"] = resolve_local_occupancy_msr_config(
        trifusion["proposal"]
    )
    trifusion["missing_instance_graph"] = (
        resolve_missing_instance_graph_config(
            trifusion["missing_instance_graph"]
        )
    )
    safety_gate = trifusion["safety_gate"]
    for key in (
        "enabled",
        "mutate",
        "collect_diagnostics",
        "require_iou50_crossing",
    ):
        if not isinstance(safety_gate[key], (bool, np.bool_)):
            raise ValueError(
                f"trifusion_observer.safety_gate.{key} must be Boolean"
            )
        safety_gate[key] = bool(safety_gate[key])
    if safety_gate["mutate"]:
        raise ValueError(
            "trifusion AP50 safety gate is observer-only; mutate must be "
            "false"
        )
    checkpoint = safety_gate["checkpoint"]
    if checkpoint is not None:
        if not isinstance(checkpoint, (str, os.PathLike)):
            raise TypeError(
                "trifusion_observer.safety_gate.checkpoint must be a path"
            )
        checkpoint = os.fspath(checkpoint)
        if not checkpoint.strip():
            raise ValueError(
                "trifusion_observer.safety_gate.checkpoint cannot be empty"
            )
        safety_gate["checkpoint"] = checkpoint
    AP50SafetyGateConfig(
        minimum_improvement_probability=safety_gate[
            "minimum_improvement_probability"
        ],
        maximum_harm_probability=safety_gate[
            "maximum_harm_probability"
        ],
        uncertainty_multiplier=safety_gate["uncertainty_multiplier"],
        maximum_delta_std=safety_gate["maximum_delta_std"],
        minimum_delta_lower_bound=safety_gate[
            "minimum_delta_lower_bound"
        ],
        minimum_predicted_iou_margin=safety_gate[
            "minimum_predicted_iou_margin"
        ],
        require_iou50_crossing=safety_gate[
            "require_iou50_crossing"
        ],
        minimum_iou50_crossing_probability=safety_gate[
            "minimum_iou50_crossing_probability"
        ],
    ).validated()
    profile_is_trifusion = (
        resolved["ablation_profile"] == "trifusion_plus10_observer"
    )
    if trifusion["enabled"] != profile_is_trifusion:
        raise ValueError(
            "trifusion_observer may be enabled only by the exact "
            "trifusion_plus10_observer profile"
        )
    if (
        bool(trifusion["missing_instance_graph"]["enabled"])
        != profile_is_trifusion
    ):
        raise ValueError(
            "trifusion missing-instance graph may be enabled only by the "
            "exact trifusion_plus10_observer profile"
        )
    if trifusion["collect_diagnostics"] != trifusion["enabled"]:
        raise ValueError(
            "trifusion_observer.collect_diagnostics must exactly match "
            "enabled"
        )
    if safety_gate["enabled"] and not profile_is_trifusion:
        raise ValueError(
            "trifusion AP50 safety gate may be enabled only by the exact "
            "trifusion_plus10_observer profile"
        )
    if safety_gate["collect_diagnostics"] != safety_gate["enabled"]:
        raise ValueError(
            "trifusion AP50 safety gate collect_diagnostics must exactly "
            "match enabled"
        )
    if safety_gate["enabled"] and safety_gate["checkpoint"] is None:
        raise ValueError(
            "enabled trifusion AP50 safety gate requires checkpoint"
        )
    if trifusion["enabled"]:
        if (
            not generic_geometry["enabled"]
            or not generic_geometry["collect_diagnostics"]
            or generic_geometry["mutate"]
        ):
            raise ValueError(
                "trifusion observer requires the non-mutating C4 secondary "
                "geometry stream"
            )
        observer_conflicts = []
        if refit["enabled"]:
            observer_conflicts.append("refit")
        if box_refiner["enabled"]:
            observer_conflicts.append("box_refiner")
        if joint["enabled"] or joint["mutate_geometry"] or joint[
            "mutate_scores"
        ]:
            observer_conflicts.append("joint_local_head")
        if supplemental["enabled"]:
            observer_conflicts.append("supplemental_output")
        if soft_nms["enabled"]:
            observer_conflicts.append("quality.soft_nms")
        if not quality["enabled"]:
            observer_conflicts.append("quality.disabled")
        if missing_identity["enabled"]:
            observer_conflicts.append("missing_track_identity")
        if resolved["mask_graph"]["enabled"]:
            observer_conflicts.append("mask_graph")
        if observer_conflicts:
            raise ValueError(
                "trifusion observer must remain a strict quality-only B6 "
                "child; conflicts: " + ", ".join(observer_conflicts)
            )

    residual = resolved["residual_track_observer"]
    for key in (
        "enabled",
        "observer_only",
        "mutate",
        "collect_diagnostics",
    ):
        if not isinstance(residual[key], (bool, np.bool_)):
            raise ValueError(
                f"residual_track_observer.{key} must be Boolean"
            )
        residual[key] = bool(residual[key])
    if not residual["observer_only"] or residual["mutate"]:
        raise ValueError(
            "residual_track_observer is strictly observer-only"
        )
    source_mode = str(residual["source_mode"]).strip().lower()
    if source_mode not in {"sam3", "yoloe", "dual"}:
        raise ValueError(
            "residual_track_observer.source_mode must be sam3, yoloe, "
            "or dual"
        )
    residual["source_mode"] = source_mode
    for key in ("primary_provider_name", "secondary_provider_name"):
        value = residual[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"residual_track_observer.{key} must be a non-empty string"
            )
        residual[key] = value.strip().casefold()
    if (
        residual["primary_provider_name"]
        == residual["secondary_provider_name"]
    ):
        raise ValueError(
            "residual track primary and secondary provider names must differ"
        )
    residual["missing_instance_graph"] = (
        resolve_missing_instance_graph_config(
            residual["missing_instance_graph"]
        )
    )
    profile_is_residual = (
        resolved["ablation_profile"] == "residual_track_observer"
    )
    if residual["enabled"] != profile_is_residual:
        raise ValueError(
            "residual_track_observer may be enabled only by the exact "
            "residual_track_observer profile"
        )
    if (
        bool(residual["missing_instance_graph"]["enabled"])
        != profile_is_residual
    ):
        raise ValueError(
            "residual missing-instance graph may be enabled only by the "
            "exact residual_track_observer profile"
        )
    if residual["collect_diagnostics"] != residual["enabled"]:
        raise ValueError(
            "residual_track_observer.collect_diagnostics must exactly "
            "match enabled"
        )
    if residual["enabled"]:
        if (
            not generic_geometry["enabled"]
            or not generic_geometry["collect_diagnostics"]
            or generic_geometry["mutate"]
        ):
            raise ValueError(
                "residual track observer requires the non-mutating detached "
                "secondary provider stream"
            )
        if generic_geometry["observe_existing_globals"]:
            raise ValueError(
                "residual track observer must not build C4 per-global "
                "geometry memory"
            )
        observer_conflicts = []
        if refit["enabled"]:
            observer_conflicts.append("refit")
        if box_refiner["enabled"]:
            observer_conflicts.append("box_refiner")
        if joint["enabled"] or joint["mutate_geometry"] or joint[
            "mutate_scores"
        ]:
            observer_conflicts.append("joint_local_head")
        if supplemental["enabled"]:
            observer_conflicts.append("supplemental_output")
        if soft_nms["enabled"]:
            observer_conflicts.append("quality.soft_nms")
        if not quality["enabled"]:
            observer_conflicts.append("quality.disabled")
        if missing_identity["enabled"] or resolved["mask_graph"]["enabled"]:
            observer_conflicts.append("primary_missing_graph")
        if trifusion["enabled"]:
            observer_conflicts.append("trifusion_observer")
        if observer_conflicts:
            raise ValueError(
                "residual track observer must remain a strict quality-only "
                "B6 child; conflicts: " + ", ".join(observer_conflicts)
            )

    yidu = resolved["yidu_ablation"]
    for key in (
        "enabled",
        "observer_only",
        "mutate",
        "collect_diagnostics",
        "frozen_b6",
    ):
        if not isinstance(yidu[key], (bool, np.bool_)):
            raise ValueError(f"yidu_ablation.{key} must be Boolean")
        yidu[key] = bool(yidu[key])
    if (
        not yidu["observer_only"]
        or yidu["mutate"]
        or not yidu["frozen_b6"]
    ):
        raise ValueError(
            "released YiDu stages must be observer-only, non-mutating "
            "children of frozen B6"
        )
    profile_is_yidu = resolved["ablation_profile"] in YIDU_PROFILE_TO_STAGE
    expected_stage = (
        YIDU_PROFILE_TO_STAGE[resolved["ablation_profile"]]
        if profile_is_yidu
        else "B0"
    )
    if str(yidu["schema"]) != YIDU_SCHEMA:
        raise ValueError("yidu_ablation.schema mismatch")
    if str(yidu["stage"]) != expected_stage:
        raise ValueError(
            "yidu_ablation.stage disagrees with the exact ablation profile"
        )
    expected_profile = YIDU_STAGE_TO_PROFILE[expected_stage]
    if str(yidu["profile"]) != expected_profile:
        raise ValueError(
            "yidu_ablation.profile disagrees with the canonical stage"
        )
    expected_enabled = profile_is_yidu and expected_stage != "B0"
    if yidu["enabled"] != expected_enabled:
        raise ValueError("yidu_ablation.enabled disagrees with stage")
    if yidu["collect_diagnostics"] != expected_enabled:
        raise ValueError(
            "yidu_ablation.collect_diagnostics must exactly match enabled"
        )
    if yidu["added_module"] != YIDU_STAGE_ADDED_MODULE[expected_stage]:
        raise ValueError("yidu_ablation.added_module disagrees with stage")
    modules = yidu["modules"]
    if not isinstance(modules, Mapping):
        raise TypeError("yidu_ablation.modules must be a mapping")
    expected_modules = dict(YIDU_STAGE_MODULE_MATRIX[expected_stage])
    if set(modules) != set(YIDU_MODULES):
        raise ValueError("yidu_ablation.modules has an invalid schema")
    normalized_modules = {}
    for module in YIDU_MODULES:
        value = modules[module]
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(
                f"yidu_ablation.modules.{module} must be Boolean"
            )
        normalized_modules[module] = bool(value)
    if normalized_modules != expected_modules:
        raise ValueError(
            "yidu_ablation.modules disagrees with the cumulative stage"
        )
    yidu["modules"] = normalized_modules
    for module in YIDU_MODULES:
        section = yidu[module]
        for key in ("enabled", "observer_only", "mutate"):
            if not isinstance(section[key], (bool, np.bool_)):
                raise ValueError(
                    f"yidu_ablation.{module}.{key} must be Boolean"
                )
            section[key] = bool(section[key])
        if (
            section["enabled"] != expected_modules[module]
            or not section["observer_only"]
            or section["mutate"]
        ):
            raise ValueError(
                f"yidu_ablation.{module} violates its stage contract"
            )
    for section_name, checkpoint_key in (
        ("raw_fused_query", "scorer_checkpoint"),
        ("quality_gate", "checkpoint"),
    ):
        checkpoint = yidu[section_name][checkpoint_key]
        if checkpoint is not None:
            if not isinstance(checkpoint, (str, os.PathLike)):
                raise TypeError(
                    f"yidu_ablation.{section_name}.{checkpoint_key} "
                    "must be a path or None"
                )
            checkpoint = os.fspath(checkpoint)
            if not checkpoint.strip():
                raise ValueError(
                    f"yidu_ablation.{section_name}.{checkpoint_key} "
                    "cannot be empty"
                )
            yidu[section_name][checkpoint_key] = checkpoint
    local_observer_updates = dict(yidu["local_observer"])
    local_observer_updates["raw_fused_scorer_checkpoint"] = yidu[
        "raw_fused_query"
    ]["scorer_checkpoint"]
    yidu["local_observer"] = resolve_yidu_local_observer_config(
        local_observer_updates
    )
    if expected_stage == "A6" and yidu["quality_gate"]["checkpoint"] is None:
        raise ValueError(
            "YiDu A6 requires a train-only AP50 gate checkpoint"
        )
    if expected_stage != "A6" and yidu["quality_gate"]["checkpoint"] is not None:
        raise ValueError(
            "only YiDu A6 may carry an AP50 gate checkpoint"
        )
    if yidu["enabled"]:
        if (
            not generic_geometry["enabled"]
            or not generic_geometry["collect_diagnostics"]
            or generic_geometry["mutate"]
        ):
            raise ValueError(
                "YiDu observer requires the non-mutating C4 evidence stream"
            )
        yidu_conflicts = []
        if refit["enabled"]:
            yidu_conflicts.append("refit")
        if box_refiner["enabled"]:
            yidu_conflicts.append("box_refiner")
        if joint["enabled"] or joint["mutate_geometry"] or joint[
            "mutate_scores"
        ]:
            yidu_conflicts.append("joint_local_head")
        if supplemental["enabled"]:
            yidu_conflicts.append("supplemental_output")
        if soft_nms["enabled"]:
            yidu_conflicts.append("quality.soft_nms")
        if not quality["enabled"]:
            yidu_conflicts.append("quality.disabled")
        if missing_identity["enabled"] or resolved["mask_graph"]["enabled"]:
            yidu_conflicts.append("primary_missing_graph")
        if trifusion["enabled"]:
            yidu_conflicts.append("trifusion_observer")
        if yidu_conflicts:
            raise ValueError(
                "YiDu observer must remain a strict quality-only B6 child; "
                "conflicts: " + ", ".join(yidu_conflicts)
            )
    if joint["enabled"]:
        conflicts = []
        if refit["enabled"]:
            conflicts.append("refit")
        if box_refiner["enabled"]:
            conflicts.append("box_refiner")
        if quality["enabled"]:
            conflicts.append("quality")
        if supplemental["enabled"]:
            conflicts.append("supplemental_output")
        if soft_nms["enabled"]:
            conflicts.append("quality.soft_nms")
        if conflicts:
            raise ValueError(
                "joint_local_head is mutually exclusive with legacy "
                "mutations: " + ", ".join(conflicts)
            )
        top_k_views = resolved["object_memory"].get("top_k_views", 0)
        if (
            isinstance(top_k_views, bool)
            or not isinstance(top_k_views, (int, np.integer))
            or int(top_k_views) < int(joint["max_views"])
        ):
            raise ValueError(
                "joint_local_head requires object_memory.top_k_views >= "
                "joint_local_head.max_views"
            )

    diagnostics = resolved["diagnostics"]
    for key in ("enabled", "dump_track_memory"):
        if not isinstance(diagnostics[key], (bool, np.bool_)):
            raise ValueError(f"diagnostics.{key} must be Boolean")
        diagnostics[key] = bool(diagnostics[key])
    diagnostics["point_count"] = _positive_int(
        diagnostics, "point_count"
    )
    if diagnostics["enabled"] and diagnostics["root"] is None:
        raise ValueError("diagnostics.root is required when diagnostics are enabled")
    return resolved


def corners_to_center_size(corners: Any) -> np.ndarray:
    """Convert arbitrary 8-corner boxes to enclosing world AABBs."""

    values = np.asarray(corners, dtype=np.float32)
    if values.size == 0:
        return np.empty((0, 6), dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (8, 3):
        raise ValueError("corners must have shape [N, 8, 3]")
    if not np.isfinite(values).all():
        raise ValueError("corners must be finite")
    minimum = values.min(axis=1)
    maximum = values.max(axis=1)
    dims = maximum - minimum
    if np.any(dims <= 0.0):
        raise ValueError("corners must define positive-volume boxes")
    return np.concatenate(((minimum + maximum) * 0.5, dims), axis=1)


def center_size_to_corners(boxes: Any) -> np.ndarray:
    values = np.asarray(boxes, dtype=np.float32)
    if values.size == 0:
        return np.empty((0, 8, 3), dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("boxes must have shape [N, 6]")
    return np.stack([aabb_corners(box[:3], box[3:6]) for box in values])


def _oriented_box_frame(
    corners: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(center, dimensions, world_from_local_basis)``.

    BoxFusion's corner ordering has the three edges incident to corner zero at
    indices 1, 3, and 4.  Using those edges retains the upstream yaw instead
    of replacing an oriented box with its enclosing world AABB.
    """

    values = np.asarray(corners, dtype=np.float64)
    if values.shape != (8, 3) or not np.isfinite(values).all():
        raise ValueError("oriented corners must have finite shape [8, 3]")
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
        raise ValueError("oriented corners must define positive edges")
    basis = edges / dimensions[None, :]
    gram = basis.T @ basis
    if not np.allclose(gram, np.eye(3), atol=1e-3, rtol=0.0):
        raise ValueError("oriented box edges must be orthogonal")
    if float(np.linalg.det(basis)) <= 0.0:
        raise ValueError("oriented box basis must be right-handed")
    reconstructed = (
        aabb_corners(np.zeros(3), dimensions).astype(np.float64)
        @ basis.T
        + center[None, :]
    )
    tolerance = max(float(np.max(dimensions)) * 1e-4, 1e-5)
    if not np.allclose(
        reconstructed, values, atol=tolerance, rtol=0.0
    ):
        raise ValueError("corners do not follow BoxFusion box ordering")
    return (
        center.astype(np.float64),
        dimensions.astype(np.float64),
        basis.astype(np.float64),
    )


def _points_to_box_local(
    points: Any,
    center: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    return ((values - center[None, :]) @ basis).astype(np.float32)


@dataclass(frozen=True)
class _OrientedBoxRefinerInputs:
    """Exact NumPy inputs shared by online B5-v2 and its diagnostics."""

    frame_center: np.ndarray
    frame_basis: np.ndarray
    local_box: np.ndarray
    gate_points_local: np.ndarray
    points_local: np.ndarray
    point_mask: np.ndarray


def _prepare_oriented_box_refiner_inputs(
    original_corners: Any,
    geometry_points: Any,
    point_count: int,
) -> _OrientedBoxRefinerInputs:
    """Create the precise local-frame tensors passed to B5-v2.

    Keeping this as the single implementation for both runtime inference and
    diagnostic export prevents subtle disagreement in OBB frame construction
    or deterministic bounded sampling.
    """

    center, dimensions, basis = _oriented_box_frame(original_corners)
    gate_points_local = _points_to_box_local(
        geometry_points, center, basis
    )
    sampled = deterministic_bounded_sample(
        gate_points_local, int(point_count)
    )
    points = np.zeros((int(point_count), 3), dtype=np.float32)
    mask = np.zeros(int(point_count), dtype=bool)
    points[: sampled.shape[0]] = sampled
    mask[: sampled.shape[0]] = True
    local_box = np.concatenate(
        (
            np.zeros(3, dtype=np.float32),
            dimensions.astype(np.float32),
        )
    )
    return _OrientedBoxRefinerInputs(
        frame_center=center.copy(),
        frame_basis=basis.copy(),
        local_box=local_box,
        gate_points_local=gate_points_local,
        points_local=points,
        point_mask=mask,
    )


def _local_box_to_world_corners(
    box: Any,
    center: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    value = np.asarray(box, dtype=np.float64)
    if value.shape != (6,) or np.any(value[3:6] <= 0.0):
        raise ValueError("local box must be a positive [6] array")
    local_corners = aabb_corners(value[:3], value[3:6])
    return (
        local_corners.astype(np.float64) @ basis.T
        + center[None, :]
    ).astype(np.float32)


def bbox_iou_2d(box_a: Any, box_b: Any) -> float:
    a = np.asarray(box_a, dtype=np.float64).reshape(-1)
    b = np.asarray(box_b, dtype=np.float64).reshape(-1)
    if a.shape != (4,) or b.shape != (4,):
        raise ValueError("2D boxes must each have shape [4]")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("2D boxes must be finite")
    intersection_size = np.maximum(
        np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]),
        0.0,
    )
    intersection = float(np.prod(intersection_size))
    area_a = float(np.prod(np.maximum(a[2:] - a[:2], 0.0)))
    area_b = float(np.prod(np.maximum(b[2:] - b[:2], 0.0)))
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def supplemental_extent_is_valid(
    dimensions: Any,
    label: Optional[str],
    config: Mapping[str, Any],
    *,
    default_minimum_extent: float,
) -> bool:
    """Apply C1's source-aware semantic extent policy.

    The policy is intentionally an allow-list.  Unknown labels retain the
    caller's uniform minimum extent, while a door/window may have one thin
    axis and a sink may be small on all axes.  Sorting the dimensions makes
    the rule independent of ScanNet/world axis conventions.
    """

    dims = np.asarray(dimensions, dtype=np.float64)
    if dims.shape != (3,):
        return False
    fallback = float(default_minimum_extent)
    if not np.isfinite(fallback) or fallback < 0.0:
        raise ValueError("default_minimum_extent must be finite and non-negative")
    if not bool(config.get("class_aware_extent", False)):
        # Preserve the released legacy expression exactly on non-C1 paths.
        return bool(
            not (
                fallback > 0.0
                and np.any(dims < fallback)
            )
        )
    if not np.isfinite(dims).all() or np.any(dims <= 0.0):
        return False

    normalized_label = (
        "" if label is None else str(label).strip().casefold()
    )
    ordered = np.sort(dims)
    if normalized_label in set(config.get("planar_extent_labels", ())):
        thresholds = np.asarray(
            (
                config["planar_min_extent"],
                config["planar_middle_extent"],
                config["planar_max_extent"],
            ),
            dtype=np.float64,
        )
        return bool(np.all(ordered >= thresholds))
    if normalized_label in set(config.get("small_extent_labels", ())):
        thresholds = np.asarray(
            (
                config["small_min_extent"],
                config["small_middle_extent"],
                config["small_max_extent"],
            ),
            dtype=np.float64,
        )
        return bool(np.all(ordered >= thresholds))
    return bool(np.all(dims >= fallback))


def bev_iou_and_containment(box_a: Any, box_b: Any) -> Tuple[float, float]:
    """Return XY footprint IoU and intersection-over-smaller-area."""

    a = np.asarray(box_a, dtype=np.float64)
    b = np.asarray(box_b, dtype=np.float64)
    if (
        a.shape != (6,)
        or b.shape != (6,)
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or np.any(a[3:6] <= 0.0)
        or np.any(b[3:6] <= 0.0)
    ):
        raise ValueError("BEV boxes must be finite positive [6] arrays")
    a_min = a[:2] - 0.5 * a[3:5]
    a_max = a[:2] + 0.5 * a[3:5]
    b_min = b[:2] - 0.5 * b[3:5]
    b_max = b[:2] + 0.5 * b[3:5]
    intersection_dims = np.maximum(
        np.minimum(a_max, b_max) - np.maximum(a_min, b_min),
        0.0,
    )
    intersection = float(np.prod(intersection_dims))
    area_a = float(np.prod(a[3:5]))
    area_b = float(np.prod(b[3:5]))
    union = area_a + area_b - intersection
    iou = 0.0 if union <= 0.0 else intersection / union
    smaller = min(area_a, area_b)
    containment = 0.0 if smaller <= 0.0 else intersection / smaller
    return float(iou), float(containment)


def _axis_overlap_containment(
    box_a: Any,
    box_b: Any,
    *,
    axis: int,
) -> float:
    """Intersection length divided by the smaller extent on one axis."""

    a = np.asarray(box_a, dtype=np.float64)
    b = np.asarray(box_b, dtype=np.float64)
    if (
        a.shape != (6,)
        or b.shape != (6,)
        or axis not in (0, 1, 2)
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or a[axis + 3] <= 0.0
        or b[axis + 3] <= 0.0
    ):
        raise ValueError("axis-containment boxes must be finite positive [6]")
    lower = max(
        a[axis] - 0.5 * a[axis + 3],
        b[axis] - 0.5 * b[axis + 3],
    )
    upper = min(
        a[axis] + 0.5 * a[axis + 3],
        b[axis] + 0.5 * b[axis + 3],
    )
    intersection = max(float(upper - lower), 0.0)
    return float(
        intersection / min(a[axis + 3], b[axis + 3])
    )


@dataclass(frozen=True)
class ViewEvidence:
    frame_index: int
    score: float
    bbox: np.ndarray
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    image_shape: Tuple[int, int]
    area_ratio: float

    def __post_init__(self) -> None:
        bbox = np.asarray(self.bbox, dtype=np.float32)
        intrinsics = np.asarray(self.intrinsics, dtype=np.float32)
        pose = np.asarray(self.camera_to_world, dtype=np.float32)
        if bbox.shape != (4,) or intrinsics.shape != (3, 3) or pose.shape != (4, 4):
            raise ValueError("invalid view-evidence array shape")
        for value in (bbox, intrinsics, pose):
            if not np.isfinite(value).all():
                raise ValueError("view evidence must be finite")
        object.__setattr__(self, "bbox", bbox.copy())
        object.__setattr__(self, "intrinsics", intrinsics.copy())
        object.__setattr__(self, "camera_to_world", pose.copy())


@dataclass
class EvidenceStats:
    scores: List[float] = field(default_factory=list)
    view_records: List[ViewEvidence] = field(default_factory=list)
    box_records: List[Tuple[float, int, np.ndarray]] = field(
        default_factory=list
    )
    label_votes: Counter = field(default_factory=Counter)
    feature_sum: Optional[np.ndarray] = None
    feature_count: int = 0
    absorbed_views: int = 0

    def record(
        self,
        proposal: SupplementalProposal,
        view: ViewEvidence,
        *,
        max_views: int,
        box: Optional[np.ndarray] = None,
    ) -> None:
        self.scores.append(float(proposal.score))
        self.scores = self.scores[-64:]
        self.view_records.append(view)
        self.view_records.sort(key=lambda item: (-item.score, item.frame_index))
        del self.view_records[max_views:]
        if box is not None:
            box_value = np.asarray(box, dtype=np.float32)
            if (
                box_value.shape != (6,)
                or not np.isfinite(box_value).all()
                or np.any(box_value[3:6] <= 0.0)
            ):
                raise ValueError(
                    "evidence box must be a finite positive [6] AABB"
                )
            self.box_records.append(
                (float(proposal.score), int(view.frame_index), box_value.copy())
            )
            self.box_records.sort(key=lambda item: (-item[0], item[1]))
            del self.box_records[max_views:]
        if proposal.label is not None:
            self.label_votes[proposal.label] += float(proposal.score)
        if proposal.feature is not None:
            feature = np.asarray(proposal.feature, dtype=np.float32)
            norm = float(np.linalg.norm(feature))
            if norm > 1e-8:
                feature = feature / norm
                if self.feature_sum is None:
                    self.feature_sum = np.zeros_like(feature)
                if self.feature_sum.shape == feature.shape:
                    self.feature_sum += feature
                    self.feature_count += 1

    def merge_from(
        self,
        other: "EvidenceStats",
        *,
        max_views: int,
    ) -> None:
        """Merge frozen candidate evidence without dropping appearance state."""

        self.scores = (list(other.scores) + self.scores)[-64:]
        self.view_records.extend(other.view_records)
        self.view_records.sort(
            key=lambda item: (-item.score, item.frame_index)
        )
        del self.view_records[max_views:]
        self.box_records.extend(
            (score, frame, box.copy())
            for score, frame, box in other.box_records
        )
        self.box_records.sort(key=lambda item: (-item[0], item[1]))
        del self.box_records[max_views:]
        self.label_votes.update(other.label_votes)
        self.absorbed_views += int(other.absorbed_views)
        if other.feature_sum is None or other.feature_count <= 0:
            return
        if self.feature_sum is None:
            self.feature_sum = other.feature_sum.copy()
            self.feature_count = int(other.feature_count)
        elif self.feature_sum.shape == other.feature_sum.shape:
            self.feature_sum += other.feature_sum
            self.feature_count += int(other.feature_count)

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.scores)) if self.scores else 0.0

    @property
    def label(self) -> Optional[str]:
        if not self.label_votes:
            return None
        return sorted(
            self.label_votes.items(), key=lambda item: (-item[1], item[0])
        )[0][0]

    @property
    def appearance_consistency(self) -> float:
        if self.feature_sum is None or self.feature_count == 0:
            return 0.5
        if self.feature_count == 1:
            return 0.5
        return float(
            np.clip(
                np.linalg.norm(self.feature_sum) / float(self.feature_count),
                0.0,
                1.0,
            )
        )

    def temporal_box_stability(self, reference_box: np.ndarray) -> float:
        """Top-K lifted-box stability from center and log-size dispersion."""

        reference = np.asarray(reference_box, dtype=np.float64)
        if reference.shape != (6,) or np.any(reference[3:6] <= 0.0):
            raise ValueError("reference_box must be a positive [6] AABB")
        if len(self.box_records) < 2:
            return 0.5
        boxes = np.stack(
            [record[2] for record in self.box_records]
        ).astype(np.float64)
        weights = np.asarray(
            [max(record[0], 1e-4) for record in self.box_records],
            dtype=np.float64,
        )
        weights = weights / weights.sum()
        center_mean = np.sum(boxes[:, :3] * weights[:, None], axis=0)
        center_variance = np.sum(
            np.sum((boxes[:, :3] - center_mean) ** 2, axis=1) * weights
        )
        diagonal = max(float(np.linalg.norm(reference[3:6])), 1e-6)
        center_dispersion = float(np.sqrt(center_variance) / diagonal)
        log_dimensions = np.log(np.maximum(boxes[:, 3:6], 1e-6))
        log_mean = np.sum(log_dimensions * weights[:, None], axis=0)
        size_variance = np.sum(
            np.mean((log_dimensions - log_mean) ** 2, axis=1) * weights
        )
        size_dispersion = float(np.sqrt(size_variance))
        return float(
            np.clip(
                np.exp(
                    -(
                        center_dispersion / 0.20
                        + size_dispersion / 0.15
                    )
                ),
                0.0,
                1.0,
            )
        )


@dataclass
class GlobalEvidence:
    stable_id: int
    memory: ObjectGeometryMemory
    stats: EvidenceStats
    detector_score: float
    last_box: np.ndarray


@dataclass
class SupplementalEvidence:
    track_id: int
    stats: EvidenceStats = field(default_factory=EvidenceStats)
    graph: Optional[MaskGraphState] = None
    graph_rejections: Counter = field(default_factory=Counter)


@dataclass(frozen=True)
class _JointPreparedInstance:
    """One valid row in the single batched joint-head invocation."""

    index: int
    stable_id: int
    evidence: GlobalEvidence
    original_box: np.ndarray
    original_corners: np.ndarray
    detector_score: float
    quality_mapping: Mapping[str, float]
    quality_features: np.ndarray
    frame_center: np.ndarray
    frame_basis: np.ndarray
    local_box: np.ndarray
    gate_points_local: np.ndarray
    points_local: np.ndarray
    point_mask: np.ndarray
    view_features: np.ndarray
    view_mask: np.ndarray


@dataclass(frozen=True)
class _JointPrediction:
    """Validated CPU output for one row of the joint-head batch."""

    prepared: _JointPreparedInstance
    center_residual: np.ndarray
    center_residual_fraction: np.ndarray
    log_dimension_residual: np.ndarray
    improvement_probability: float
    quality_components: np.ndarray
    ranking_scores: np.ndarray
    quality_log_variance: np.ndarray
    quality_uncertainty: np.ndarray
    view_attention: np.ndarray


@dataclass(frozen=True)
class LiftedProposal:
    proposal: SupplementalProposal
    observation: ObjectObservation
    box: np.ndarray
    depth_ratio: float
    view: ViewEvidence


@dataclass(frozen=True)
class GlobalProposalMatch:
    """The selected global association and its observable evidence."""

    global_index: int
    overlap_3d: float
    projection_iou: float
    point_support: float
    center_distance: float
    score: float
    strong: bool


@dataclass(frozen=True)
class _AbsorbedSupplementalRecord:
    """Frozen confirmed candidate retained only by the opt-in C1 profile."""

    track: Any
    metadata: SupplementalEvidence
    absorbed_global_stable_id: int
    event_frame: int
    trigger_overlap: float
    match: GlobalProposalMatch


@dataclass(frozen=True)
class _SupplementalMaterialized:
    """A graph-confirmed row after structural gates but before B6/B5."""

    track_id: int
    box: np.ndarray
    detector_score: float
    label: Optional[str]
    memory: ObjectGeometryMemory
    stats: EvidenceStats
    view_count: int
    recovered_absorbed: bool


@dataclass(frozen=True)
class _SupplementalOutput:
    box: np.ndarray
    corners: np.ndarray
    score: float
    stable_id: int
    label: Optional[str]
    quality_features: np.ndarray
    memory: ObjectGeometryMemory
    stats: EvidenceStats
    original_box: np.ndarray
    original_corners: np.ndarray
    refit_applied: bool
    refit_reason: str


@dataclass(frozen=True)
class FinalRefinementResult:
    corners: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    source_indices: np.ndarray
    stable_ids: np.ndarray
    labels: Tuple[Optional[str], ...]
    quality_features: np.ndarray
    refit_original_boxes: np.ndarray
    refit_original_corners: np.ndarray
    refit_applied: np.ndarray
    refit_reasons: Tuple[str, ...]
    refit_changed_axes: np.ndarray
    refit_boundary_delta: np.ndarray
    refit_local_original_boxes: np.ndarray
    refit_local_candidate_boxes: np.ndarray
    refit_local_basis: np.ndarray
    refit_local_frame_valid: np.ndarray
    summary: Mapping[str, Any]


def _empty_runtime_stats() -> Dict[str, Any]:
    return {
        "keyframes": 0,
        "provider_calls": 0,
        "provider_seconds": 0.0,
        "appearance_seconds": 0.0,
        "geometry_seconds": 0.0,
        "proposals": 0,
        "lifted": 0,
        "matched_global": 0,
        "strong_global_matches": 0,
        "weak_global_matches": 0,
        "weak_shadow_candidates": 0,
        "unmatched_candidates": 0,
        "candidate_updates": 0,
        "candidate_archived": 0,
        "candidate_discarded": 0,
        "mask_graph_edges_evaluated": 0,
        "mask_graph_edges_accepted": 0,
        "mask_graph_nodes": 0,
        "mask_graph_confirmed": 0,
        "mask_graph_rejected": Counter(),
        "absorbed_recovery_stored": 0,
        "absorbed_recovery_considered": 0,
        "absorbed_recovery_eligible": 0,
        "absorbed_recovery_output": 0,
        "supplemental_considered": 0,
        "supplemental_rejected_graph": 0,
        "supplemental_rejected_extent": 0,
        "supplemental_rejected_class_extent": 0,
        "supplemental_rejected_refined_extent": 0,
        "supplemental_rejected_score": 0,
        "supplemental_rejected_projection": 0,
        "supplemental_rejected_global": 0,
        "supplemental_rejected_bev_global": 0,
        "supplemental_rejected_refined_global": 0,
        "supplemental_rejected_refined_bev_global": 0,
        "supplemental_scores_rank_mapped": 0,
        "supplemental_planar_deduplicated": 0,
        "supplemental_refined_deduplicated": 0,
        "supplemental_output": 0,
        "supplemental_deduplicated": 0,
        "supplemental_b5_attempted": 0,
        "supplemental_b5_accepted": 0,
        "refits_attempted": 0,
        "refits_accepted": 0,
        "neural_refits_attempted": 0,
        "neural_refits_accepted": 0,
        "neural_refits_quality_rejected": 0,
        "neural_refits_gate_rejected": 0,
        "neural_refits_invalid_orientation": 0,
        "joint_instances": 0,
        "joint_unobserved_identity": 0,
        "joint_invalid_identity": 0,
        "joint_inputs_valid": 0,
        "joint_batches": 0,
        "joint_forward_boxes": 0,
        "joint_improvement_rejected": 0,
        "joint_uncertainty_rejected": 0,
        "joint_gate_rejected": 0,
        "joint_accepted": 0,
        "joint_original_quality_branch": 0,
        "joint_candidate_quality_branch": 0,
        "joint_prepare_seconds": 0.0,
        "joint_forward_seconds": 0.0,
        "joint_gate_seconds": 0.0,
        "joint_rejected": Counter(),
        "c2_attempted": 0,
        "c2_proposed": 0,
        "c2_verified": 0,
        "c2_applied": 0,
        "c2_seconds": 0.0,
        "c2_rejected": Counter(),
        "fragment_stitch_seconds": 0.0,
        "fragment_stitch_invalid_snapshots": 0,
        "fragment_stitch_fail_open": 0,
        "c4_provider_calls": 0,
        "c4_provider_seconds": 0.0,
        "c4_geometry_seconds": 0.0,
        "c4_proposals": 0,
        "c4_lifted": 0,
        "c4_matched_global": 0,
        "c4_attempted": 0,
        "c4_proposed": 0,
        "c4_verified": 0,
        "c4_applied": 0,
        "c4_refiner_seconds": 0.0,
        "c4_fail_open": 0,
        "c4_rejected": Counter(),
        "trifusion_attempted": 0,
        "trifusion_valid": 0,
        "trifusion_candidates": 0,
        "trifusion_verified": 0,
        "trifusion_applied": 0,
        "trifusion_seconds": 0.0,
        "trifusion_rejected": Counter(),
        "trifusion_gate_evaluated": 0,
        "trifusion_gate_accepted": 0,
        "trifusion_gate_rejected": Counter(),
        "trifusion_missing_provider_calls": 0,
        "trifusion_missing_unmatched": 0,
        "trifusion_missing_components": 0,
        "trifusion_missing_candidates": 0,
        "trifusion_missing_errors": Counter(),
        "residual_track_provider_calls": 0,
        "residual_track_primary_unmatched": 0,
        "residual_track_secondary_unmatched": 0,
        "residual_track_components": 0,
        "residual_track_candidates": 0,
        "residual_track_errors": Counter(),
        "residual_track_observation_reasons": Counter(),
        "residual_track_association_reasons": Counter(),
        "residual_track_provider_nodes": Counter(),
        "residual_track_fail_closed": 0,
        "yidu_attempted": 0,
        "yidu_valid": 0,
        "yidu_component_candidates": 0,
        "yidu_occupancy_candidates": 0,
        "yidu_query_candidates": 0,
        "yidu_gate_evaluated": 0,
        "yidu_gate_accepted": 0,
        "yidu_applied": 0,
        "yidu_seconds": 0.0,
        "yidu_rejected": Counter(),
        "rejected": Counter(),
    }


def _validate_runtime_arrays(
    image: Any,
    depth: Any,
    intrinsics: Any,
    camera_to_world: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image = np.asarray(image)
    depth = np.asarray(depth)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    camera_to_world = np.asarray(camera_to_world, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape [H, W, 3]")
    if depth.ndim == 3 and 1 in (depth.shape[0], depth.shape[-1]):
        depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError("depth must have shape [H, W]")
    if intrinsics.shape == (4, 4):
        intrinsics = intrinsics[:3, :3]
    if intrinsics.shape != (3, 3):
        raise ValueError("intrinsics must have shape [3,3] or [4,4]")
    if camera_to_world.shape != (4, 4):
        raise ValueError("camera_to_world must have shape [4,4]")
    if not np.isfinite(intrinsics).all() or not np.isfinite(camera_to_world).all():
        raise ValueError("camera matrices must be finite")
    return image, depth, intrinsics, camera_to_world


def _deterministic_weighted_median(
    values: np.ndarray,
    weights: np.ndarray,
    tie_breakers: np.ndarray,
) -> float:
    """Return a stable weighted median with explicit tie ordering."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    tie_breakers = np.asarray(tie_breakers, dtype=np.int64).reshape(-1)
    if not len(values) or not (
        len(values) == len(weights) == len(tie_breakers)
    ):
        raise ValueError(
            "weighted median arrays must be non-empty and aligned"
        )
    if (
        not np.isfinite(values).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0.0)
    ):
        raise ValueError(
            "weighted median values and weights must be finite and "
            "weights non-negative"
        )
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("weighted median requires positive total weight")
    order = np.lexsort((tie_breakers, values))
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, 0.5 * total, side="left"))
    return float(values[order[min(index, len(order) - 1)]])


class OnlineRefinementController:
    """Opt-in final-output refinement controller."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        device: str = "cpu",
        provider: Optional[ProposalProvider] = None,
        generic_geometry_provider: Optional[ProposalProvider] = None,
        appearance_encoder: Any = None,
        box_refiner: Any = None,
        quality_scorer: Any = None,
        joint_local_head: Any = None,
        trifusion_ap50_gate: Optional[AP50SafetyGate] = None,
        yidu_ap50_gate: Optional[AP50SafetyGate] = None,
    ) -> None:
        self.config = resolve_online_refinement_config(config)
        self.enabled = bool(self.config["enabled"])
        self.device = str(device)
        if (
            self.enabled
            and self.config["scannet_axis_aligned_only"]
            and "dataset" in config
            and str(config["dataset"]).lower() != "scannet"
        ):
            raise ValueError(
                "this refinement configuration is restricted to ScanNet "
                "axis-aligned evaluation"
            )
        self.provider: Optional[ProposalProvider] = None
        self.generic_geometry_provider: Optional[ProposalProvider] = None
        self.appearance_encoder = None
        self.box_refiner = None
        self.box_refiner_coordinate_frame = "world_aabb"
        self.quality_scorer = None
        self.joint_local_head = None
        self._last_joint_runtime: Dict[int, Dict[str, Any]] = {}
        self._last_c2_runtime: Dict[int, Dict[str, Any]] = {}
        self._last_c4_runtime: Dict[int, Dict[str, Any]] = {}
        self._last_c4_error = ""
        self._c4_failed = False
        self._last_trifusion_runtime: Dict[int, Dict[str, Any]] = {}
        self.trifusion_ap50_gate: Optional[AP50SafetyGate] = None
        self._last_yidu_runtime: Dict[int, Dict[str, Any]] = {}
        self.yidu_ap50_gate: Optional[AP50SafetyGate] = None
        self._last_observer_zero_write_audit = (
            _empty_observer_zero_write_audit()
        )
        self.trifusion_missing_graph: Optional[
            MissingInstanceGraphObserver
        ] = None
        self._last_trifusion_missing_candidates: Tuple[
            OrientedMissingCandidate, ...
        ] = ()
        self._last_trifusion_missing_update: Any = None
        self.residual_track_graph: Optional[
            MissingInstanceGraphObserver
        ] = None
        self._last_residual_track_candidates: Tuple[
            OrientedMissingCandidate, ...
        ] = ()
        self._last_residual_track_decisions: Tuple[
            MissingCandidateDecision, ...
        ] = ()
        self._last_residual_track_update: Any = None
        self._residual_track_provider_counts: Dict[
            int, Counter
        ] = {}
        self._residual_track_failed = False
        self._last_residual_track_error = ""
        self._last_fragment_stitch_candidates: Tuple[
            FragmentStitchCandidate, ...
        ] = ()
        self._last_fragment_stitch_graph_snapshots: Tuple[
            Dict[str, Any], ...
        ] = ()
        self._fragment_stitch_refresh_complete = False
        self._last_fragment_stitch_error = ""
        self.object_config: Dict[str, Any] = {}
        self.generic_geometry_object_config: Dict[str, Any] = {}
        self.track_manager: Optional[CandidateTrackManager] = None
        self.global_tracks: Dict[int, GlobalEvidence] = {}
        self.generic_geometry_global_tracks: Dict[int, GlobalEvidence] = {}
        self.supplemental_metadata: Dict[int, SupplementalEvidence] = {}
        self.absorbed_supplemental_records: Dict[
            int, _AbsorbedSupplementalRecord
        ] = {}
        self.retired_mask_graph_snapshots: List[Dict[str, Any]] = []
        self.keyframe_count = 0
        self.scene_id: Optional[str] = None
        self.stats: Dict[str, Any] = _empty_runtime_stats()
        if not self.enabled:
            return

        trifusion_missing_cfg = self.config["trifusion_observer"][
            "missing_instance_graph"
        ]
        if trifusion_missing_cfg["enabled"]:
            self.trifusion_missing_graph = MissingInstanceGraphObserver(
                trifusion_missing_cfg
            )
        residual_graph_cfg = self.config["residual_track_observer"][
            "missing_instance_graph"
        ]
        if residual_graph_cfg["enabled"]:
            self.residual_track_graph = MissingInstanceGraphObserver(
                residual_graph_cfg
            )
        trifusion_gate_cfg = self.config["trifusion_observer"][
            "safety_gate"
        ]
        if trifusion_gate_cfg["enabled"]:
            self.trifusion_ap50_gate = (
                trifusion_ap50_gate
                if trifusion_ap50_gate is not None
                else load_ap50_safety_gate(
                    trifusion_gate_cfg["checkpoint"]
                )
            )
            if tuple(self.trifusion_ap50_gate.feature_names) != (
                TRIFUSION_GATE_FEATURE_NAMES
            ):
                raise ValueError(
                    "TriFusion AP50 safety checkpoint feature schema "
                    "does not match the fixed 61-D runtime schema"
                )
        elif trifusion_ap50_gate is not None:
            raise ValueError(
                "injected trifusion_ap50_gate requires an enabled safety "
                "gate configuration"
            )

        yidu_gate_cfg = self.config["yidu_ablation"]["quality_gate"]
        if yidu_gate_cfg["enabled"]:
            self.yidu_ap50_gate = (
                yidu_ap50_gate
                if yidu_ap50_gate is not None
                else load_ap50_safety_gate(yidu_gate_cfg["checkpoint"])
            )
            if tuple(self.yidu_ap50_gate.feature_names) != (
                YIDU_GATE_FEATURE_NAMES
            ):
                raise ValueError(
                    "YiDu AP50 safety checkpoint feature schema does not "
                    f"match the fixed {YIDU_GATE_FEATURE_DIM}-D runtime schema"
                )
        elif yidu_ap50_gate is not None:
            raise ValueError(
                "injected yidu_ap50_gate requires the exact A6 profile"
            )

        proposals_cfg = self.config["supplemental_proposals"]
        provider_device = proposals_cfg.get("device", self.device)
        self.provider = (
            provider
            if provider is not None
            else build_provider(proposals_cfg, str(provider_device))
        )
        generic_cfg = self.config["generic_local_geometry_refiner"]
        if generic_cfg["enabled"]:
            secondary_cfg = generic_cfg["secondary_proposals"]
            secondary_device = secondary_cfg.get("device", self.device)
            self.generic_geometry_provider = (
                generic_geometry_provider
                if generic_geometry_provider is not None
                else build_provider(
                    secondary_cfg, str(secondary_device)
                )
            )
            self.generic_geometry_object_config = dict(
                generic_cfg["secondary_object_memory"]
            )
        if self.config["appearance_memory"]["enabled"]:
            self.appearance_encoder = appearance_encoder
        self.object_config = resolve_object_memory_config(
            self.config["object_memory"]
        )
        self.track_manager = CandidateTrackManager(
            self.object_config,
            archive_confirmed=self.config["candidate_lifecycle"][
                "archive_confirmed"
            ],
        )

        refiner_cfg = self.config["box_refiner"]
        self.box_refiner_coordinate_frame = refiner_cfg[
            "coordinate_frame"
        ]
        architecture_type = (
            OrientedBoxRefinerConfig
            if self.box_refiner_coordinate_frame == "box_local"
            else BoxRefinerConfig
        )
        architecture = architecture_type(
            **dict(refiner_cfg["architecture"])
        ).validated()
        if architecture.quality_feature_dim != QUALITY_FEATURE_DIM:
            raise ValueError(
                "BoxRefiner quality_feature_dim must match the fixed "
                f"quality schema ({QUALITY_FEATURE_DIM})"
            )
        refiner_device = refiner_cfg["device"] or self.device
        if box_refiner is not None:
            self.box_refiner = box_refiner
        elif self.box_refiner_coordinate_frame == "box_local":
            self.box_refiner = build_oriented_box_refiner(
                enabled=refiner_cfg["enabled"],
                checkpoint_path=refiner_cfg["checkpoint"],
                config=architecture,
                device=refiner_device,
            )
        else:
            self.box_refiner = build_box_refiner(
                enabled=refiner_cfg["enabled"],
                checkpoint_path=refiner_cfg["checkpoint"],
                config=architecture,
                device=refiner_device,
            )

        quality_cfg = self.config["quality"]
        if quality_scorer is not None:
            self.quality_scorer = quality_scorer
        elif quality_cfg["enabled"]:
            if quality_cfg["mode"] == "heuristic":
                self.quality_scorer = make_quality_scorer("heuristic")
            else:
                if quality_cfg["checkpoint"] is None:
                    raise ValueError(
                        "learned quality scoring requires quality.checkpoint"
                    )
                self.quality_scorer = load_quality_scorer(
                    quality_cfg["checkpoint"],
                    method=quality_cfg["mode"],
                )

        joint_cfg = self.config["joint_local_head"]
        joint_architecture = JointLocalHeadConfig(
            **dict(joint_cfg["architecture"])
        ).validated()
        if joint_cfg["enabled"]:
            if joint_local_head is not None:
                injected_config = getattr(joint_local_head, "config", None)
                if (
                    injected_config is None
                    or not hasattr(injected_config, "architecture_dict")
                    or injected_config.architecture_dict()
                    != joint_architecture.architecture_dict()
                ):
                    raise ValueError(
                        "injected joint_local_head architecture does not "
                        "match joint_local_head.architecture"
                    )
                self.joint_local_head = joint_local_head
            else:
                self.joint_local_head = build_joint_local_head(
                    enabled=True,
                    checkpoint_path=joint_cfg["checkpoint"],
                    config=joint_architecture,
                    device=joint_cfg["device"] or self.device,
                )

    def reset_scene(self, scene_id: str) -> None:
        """Clear all geometry/track state while retaining loaded models."""

        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError("scene_id must be a non-empty string")
        self.scene_id = scene_id.strip()
        self.keyframe_count = 0
        self.global_tracks.clear()
        self.generic_geometry_global_tracks.clear()
        self.supplemental_metadata.clear()
        self.absorbed_supplemental_records.clear()
        self.retired_mask_graph_snapshots.clear()
        self._last_joint_runtime.clear()
        self._last_c2_runtime.clear()
        self._last_c4_runtime.clear()
        self._last_c4_error = ""
        self._c4_failed = False
        self._last_trifusion_runtime.clear()
        self._last_yidu_runtime.clear()
        self._last_observer_zero_write_audit = (
            _empty_observer_zero_write_audit()
        )
        if self.trifusion_missing_graph is not None:
            self.trifusion_missing_graph.reset()
        self._last_trifusion_missing_candidates = ()
        self._last_trifusion_missing_update = None
        if self.residual_track_graph is not None:
            self.residual_track_graph.reset()
        self._last_residual_track_candidates = ()
        self._last_residual_track_decisions = ()
        self._last_residual_track_update = None
        self._residual_track_provider_counts.clear()
        self._residual_track_failed = False
        self._last_residual_track_error = ""
        self._last_fragment_stitch_candidates = ()
        self._last_fragment_stitch_graph_snapshots = ()
        self._fragment_stitch_refresh_complete = False
        self._last_fragment_stitch_error = ""
        if self.enabled:
            self.track_manager = CandidateTrackManager(
                self.object_config,
                archive_confirmed=self.config["candidate_lifecycle"][
                    "archive_confirmed"
                ],
            )
        self.stats = _empty_runtime_stats()

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, Any],
        *,
        device: str = "cpu",
    ) -> "OnlineRefinementController":
        return cls(cfg, device=device)

    def _global_inputs(
        self,
        global_corners: Any,
        global_scores: Any,
        stable_ids: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        corners = np.asarray(global_corners, dtype=np.float32)
        if corners.size == 0:
            corners = np.empty((0, 8, 3), dtype=np.float32)
        boxes = corners_to_center_size(corners)
        scores = np.asarray(global_scores, dtype=np.float32).reshape(-1)
        ids = np.asarray(stable_ids, dtype=np.int64).reshape(-1)
        if len(boxes) != len(scores) or len(boxes) != len(ids):
            raise ValueError("global boxes, scores, and stable_ids must align")
        if not np.isfinite(scores).all():
            raise ValueError("global scores must be finite")
        if ((scores < 0.0) | (scores > 1.0)).any():
            raise ValueError("global scores must lie in [0,1]")
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("stable_ids must be unique")
        if np.any(ids < 0):
            raise ValueError("global stable_ids must be non-negative")
        return boxes, scores, ids

    def _sync_global_tracks(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        stable_ids: np.ndarray,
    ) -> None:
        current_ids = set(stable_ids.tolist())
        matching_cfg = self.config["matching"]
        for stable_id, box, score in zip(stable_ids, boxes, scores):
            key = int(stable_id)
            if key not in self.global_tracks:
                candidates = []
                for old_key, evidence in self.global_tracks.items():
                    if old_key in current_ids:
                        continue
                    overlap = aabb_iou(
                        box[:3], box[3:6],
                        evidence.last_box[:3], evidence.last_box[3:6],
                    )
                    if overlap >= matching_cfg["rekey_iou"]:
                        candidates.append((-overlap, old_key))
                if candidates:
                    _, old_key = min(candidates)
                    evidence = self.global_tracks.pop(old_key)
                    evidence.stable_id = key
                    evidence.memory.track_id = key
                    self.global_tracks[key] = evidence
            if key in self.global_tracks:
                evidence = self.global_tracks[key]
                evidence.last_box = box.copy()
                evidence.detector_score = float(score)

    def _new_global_evidence(
        self,
        stable_id: int,
        box: np.ndarray,
        score: float,
    ) -> GlobalEvidence:
        evidence = GlobalEvidence(
            stable_id=int(stable_id),
            memory=ObjectGeometryMemory(
                track_id=int(stable_id), config=self.object_config
            ),
            stats=EvidenceStats(),
            detector_score=float(score),
            last_box=np.asarray(box, dtype=np.float32).copy(),
        )
        self.global_tracks[int(stable_id)] = evidence
        return evidence

    def _lift_proposals(
        self,
        proposals: Sequence[SupplementalProposal],
        *,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        camera_to_world: np.ndarray,
        frame_index: int,
        image_shape: Tuple[int, int],
    ) -> List[LiftedProposal]:
        lifted: List[LiftedProposal] = []
        started = time.perf_counter()
        minimum_points = int(self.object_config["min_points_for_aabb"])
        for proposal in proposals:
            depth_observation = extract_masked_world_points(
                depth,
                proposal.mask,
                intrinsics,
                camera_to_world,
                self.object_config,
            )
            if depth_observation.retained_point_count < minimum_points:
                continue
            center, dims = robust_quantile_aabb(
                depth_observation.points_world,
                lower_quantile=float(
                    self.object_config["aabb_lower_quantile"]
                ),
                upper_quantile=float(
                    self.object_config["aabb_upper_quantile"]
                ),
                min_points=minimum_points,
                minimum_dimension=float(
                    self.object_config["minimum_aabb_dimension"]
                ),
            )
            box = np.concatenate((center, dims)).astype(np.float32)
            height, width = image_shape
            area = max(
                0.0,
                float(
                    (proposal.bbox[2] - proposal.bbox[0])
                    * (proposal.bbox[3] - proposal.bbox[1])
                ),
            )
            view = ViewEvidence(
                frame_index=frame_index,
                score=float(proposal.score),
                bbox=proposal.bbox,
                intrinsics=intrinsics,
                camera_to_world=camera_to_world,
                image_shape=image_shape,
                area_ratio=float(
                    np.clip(area / max(height * width, 1), 0.0, 1.0)
                ),
            )
            observation = ObjectObservation.from_depth_observation(
                depth_observation,
                confidence=float(proposal.score),
                projection_mask_iou=1.0,
                camera_position=camera_to_world[:3, 3],
            )
            lifted.append(
                LiftedProposal(
                    proposal=proposal,
                    observation=observation,
                    box=box,
                    depth_ratio=depth_observation.valid_depth_ratio,
                    view=view,
                )
            )
        self.stats["geometry_seconds"] += time.perf_counter() - started
        self.stats["lifted"] += len(lifted)
        return lifted

    def _match_to_globals_detailed(
        self,
        lifted: Sequence[LiftedProposal],
        boxes: np.ndarray,
        intrinsics: np.ndarray,
        camera_to_world: np.ndarray,
    ) -> Tuple[Dict[int, int], Dict[int, GlobalProposalMatch]]:
        cfg = self.config["matching"]
        identity_cfg = self.config["missing_track_identity"]
        candidates: List[
            Tuple[
                float,
                float,
                float,
                int,
                int,
                GlobalProposalMatch,
            ]
        ] = []
        for proposal_index, item in enumerate(lifted):
            for global_index, box in enumerate(boxes):
                overlap = aabb_iou(
                    item.box[:3],
                    item.box[3:6],
                    box[:3],
                    box[3:6],
                )
                center_distance = float(
                    np.linalg.norm(item.box[:3] - box[:3])
                )
                projection_iou = projected_aabb_mask_iou(
                    box[:3],
                    box[3:6],
                    intrinsics,
                    camera_to_world,
                    item.proposal.mask,
                    threshold=float(self.object_config["mask_threshold"]),
                )
                valid = overlap >= cfg["global_match_iou"] or (
                    projection_iou >= cfg["global_match_2d_iou"]
                    and center_distance <= cfg["max_center_distance"]
                )
                if not valid:
                    continue
                center_quality = max(
                    0.0, 1.0 - center_distance / cfg["max_center_distance"]
                )
                point_support = points_inside_aabb_fraction(
                    item.observation.points_world,
                    box[:3],
                    box[3:6] * cfg["crop_to_global_expansion"],
                )
                score = (
                    2.0 * overlap
                    + projection_iou
                    + 0.50 * point_support
                    + 0.25 * center_quality
                )
                strong = bool(
                    overlap >= identity_cfg["strong_global_iou"]
                    or (
                        projection_iou
                        >= identity_cfg["strong_projection_iou"]
                        and point_support
                        >= identity_cfg["strong_point_support"]
                    )
                )
                evidence = GlobalProposalMatch(
                    global_index=int(global_index),
                    overlap_3d=float(overlap),
                    projection_iou=float(projection_iou),
                    point_support=float(point_support),
                    center_distance=float(center_distance),
                    score=float(score),
                    strong=strong,
                )
                candidates.append(
                    (
                        -float(score),
                        -float(overlap),
                        -float(projection_iou),
                        proposal_index,
                        global_index,
                        evidence,
                    )
                )
        candidates.sort()
        assignments: Dict[int, int] = {}
        details: Dict[int, GlobalProposalMatch] = {}
        used_globals = set()
        for (
            _,
            _,
            _,
            proposal_index,
            global_index,
            evidence,
        ) in candidates:
            if proposal_index in assignments or global_index in used_globals:
                continue
            assignments[proposal_index] = global_index
            details[proposal_index] = evidence
            used_globals.add(global_index)
        return assignments, details

    def _match_to_globals(
        self,
        lifted: Sequence[LiftedProposal],
        boxes: np.ndarray,
        intrinsics: np.ndarray,
        camera_to_world: np.ndarray,
    ) -> Dict[int, int]:
        """Backward-compatible assignment-only view used by existing tests."""

        assignments, _ = self._match_to_globals_detailed(
            lifted,
            boxes,
            intrinsics,
            camera_to_world,
        )
        return assignments

    def _projection_iou_for_view(
        self, box: np.ndarray, view: ViewEvidence
    ) -> float:
        projected = project_aabb_to_image(
            box[:3],
            box[3:6],
            view.intrinsics,
            view.camera_to_world,
            view.image_shape,
            require_all_in_front=False,
        )
        if projected is None:
            return 0.0
        return bbox_iou_2d(projected, view.bbox)

    @staticmethod
    def _projection_iou_for_corners(
        corners: np.ndarray,
        view: ViewEvidence,
    ) -> float:
        values = np.asarray(corners, dtype=np.float64)
        if values.shape != (8, 3) or not np.isfinite(values).all():
            raise ValueError("corners must have finite shape [8, 3]")
        world_to_camera = np.linalg.inv(view.camera_to_world)
        homogeneous = np.column_stack(
            (values, np.ones(8, dtype=np.float64))
        )
        camera = (homogeneous @ world_to_camera.T)[:, :3]
        in_front = camera[:, 2] > 1e-3
        if not np.any(in_front):
            return 0.0
        camera = camera[in_front]
        projected = camera @ view.intrinsics.T
        pixels = projected[:, :2] / projected[:, 2:3]
        height, width = view.image_shape
        x = np.clip(pixels[:, 0], 0.0, float(width))
        y = np.clip(pixels[:, 1], 0.0, float(height))
        box = np.asarray(
            [x.min(), y.min(), x.max(), y.max()],
            dtype=np.float32,
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            return 0.0
        return bbox_iou_2d(box, view.bbox)

    def _add_global_observation(
        self,
        evidence: GlobalEvidence,
        item: LiftedProposal,
        *,
        box: np.ndarray,
        frame_index: int,
    ) -> None:
        expansion = float(
            self.config["matching"]["crop_to_global_expansion"]
        )
        points = item.observation.points_world
        inside = points_inside_aabb(
            points, box[:3], box[3:6] * expansion
        )
        minimum_points = int(self.object_config["min_points_for_aabb"])
        retained = points[inside]
        if retained.shape[0] < minimum_points:
            retained = points
        projection_iou = self._projection_iou_for_view(box, item.view)
        observation = ObjectObservation(
            points_world=retained,
            confidence=float(item.proposal.score),
            mask_pixels=item.observation.mask_pixels,
            valid_depth_pixels=item.observation.valid_depth_pixels,
            projection_mask_iou=projection_iou,
            camera_position=item.observation.camera_position,
        )
        evidence.memory.add_observation(observation, frame_index)
        evidence.stats.record(
            item.proposal,
            item.view,
            max_views=int(self.config["quality"]["max_view_records"]),
            box=item.box,
        )

    def _sync_generic_geometry_global_tracks(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        stable_ids: np.ndarray,
    ) -> None:
        """Rekey the isolated C4 memory without touching B6 evidence."""

        current_ids = set(stable_ids.tolist())
        matching_cfg = self.config["matching"]
        tracks = self.generic_geometry_global_tracks
        for stable_id, box, score in zip(stable_ids, boxes, scores):
            key = int(stable_id)
            if key not in tracks:
                candidates = []
                for old_key, evidence in tracks.items():
                    if old_key in current_ids:
                        continue
                    overlap = aabb_iou(
                        box[:3],
                        box[3:6],
                        evidence.last_box[:3],
                        evidence.last_box[3:6],
                    )
                    if overlap >= matching_cfg["rekey_iou"]:
                        candidates.append((-overlap, old_key))
                if candidates:
                    _, old_key = min(candidates)
                    evidence = tracks.pop(old_key)
                    evidence.stable_id = key
                    evidence.memory.track_id = key
                    tracks[key] = evidence
            if key in tracks:
                evidence = tracks[key]
                evidence.last_box = box.copy()
                evidence.detector_score = float(score)

    def _new_generic_geometry_global_evidence(
        self,
        stable_id: int,
        box: np.ndarray,
        score: float,
    ) -> GlobalEvidence:
        evidence = GlobalEvidence(
            stable_id=int(stable_id),
            memory=ObjectGeometryMemory(
                track_id=int(stable_id),
                config=self.generic_geometry_object_config,
            ),
            stats=EvidenceStats(),
            detector_score=float(score),
            last_box=np.asarray(box, dtype=np.float32).copy(),
        )
        self.generic_geometry_global_tracks[int(stable_id)] = evidence
        return evidence

    def _lift_generic_geometry_proposals(
        self,
        proposals: Sequence[SupplementalProposal],
        *,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        camera_to_world: np.ndarray,
        frame_index: int,
        image_shape: Tuple[int, int],
    ) -> List[LiftedProposal]:
        """Lift SAM3 teacher masks through a C4-only depth-memory config."""

        lifted: List[LiftedProposal] = []
        started = time.perf_counter()
        object_cfg = self.generic_geometry_object_config
        minimum_points = int(object_cfg["min_points_for_aabb"])
        for proposal in proposals:
            depth_observation = extract_masked_world_points(
                depth,
                proposal.mask,
                intrinsics,
                camera_to_world,
                object_cfg,
            )
            if depth_observation.retained_point_count < minimum_points:
                continue
            center, dims = robust_quantile_aabb(
                depth_observation.points_world,
                lower_quantile=float(object_cfg["aabb_lower_quantile"]),
                upper_quantile=float(object_cfg["aabb_upper_quantile"]),
                min_points=minimum_points,
                minimum_dimension=float(
                    object_cfg["minimum_aabb_dimension"]
                ),
            )
            box = np.concatenate((center, dims)).astype(np.float32)
            height, width = image_shape
            area = max(
                0.0,
                float(
                    (proposal.bbox[2] - proposal.bbox[0])
                    * (proposal.bbox[3] - proposal.bbox[1])
                ),
            )
            view = ViewEvidence(
                frame_index=frame_index,
                score=float(proposal.score),
                bbox=proposal.bbox,
                intrinsics=intrinsics,
                camera_to_world=camera_to_world,
                image_shape=image_shape,
                area_ratio=float(
                    np.clip(area / max(height * width, 1), 0.0, 1.0)
                ),
            )
            observation = ObjectObservation.from_depth_observation(
                depth_observation,
                confidence=float(proposal.score),
                projection_mask_iou=1.0,
                camera_position=camera_to_world[:3, 3],
            )
            lifted.append(
                LiftedProposal(
                    proposal=proposal,
                    observation=observation,
                    box=box,
                    depth_ratio=depth_observation.valid_depth_ratio,
                    view=view,
                )
            )
        self.stats["c4_geometry_seconds"] += (
            time.perf_counter() - started
        )
        self.stats["c4_lifted"] += len(lifted)
        return lifted

    def _add_generic_geometry_observation(
        self,
        evidence: GlobalEvidence,
        item: LiftedProposal,
        *,
        box: np.ndarray,
        frame_index: int,
    ) -> None:
        expansion = float(
            self.config["matching"]["crop_to_global_expansion"]
        )
        points = item.observation.points_world
        inside = points_inside_aabb(
            points, box[:3], box[3:6] * expansion
        )
        minimum_points = int(
            self.generic_geometry_object_config["min_points_for_aabb"]
        )
        retained = points[inside]
        if retained.shape[0] < minimum_points:
            retained = points
        projection_iou = self._projection_iou_for_view(box, item.view)
        observation = ObjectObservation(
            points_world=retained,
            confidence=float(item.proposal.score),
            mask_pixels=item.observation.mask_pixels,
            valid_depth_pixels=item.observation.valid_depth_pixels,
            projection_mask_iou=projection_iou,
            camera_position=item.observation.camera_position,
        )
        evidence.memory.add_observation(observation, frame_index)
        evidence.stats.record(
            item.proposal,
            item.view,
            max_views=5,
            box=item.box,
        )

    def _update_trifusion_missing_graph(
        self,
        *,
        proposals: Sequence[SupplementalProposal],
        lifted: Sequence[LiftedProposal],
        assignments: Mapping[int, int],
        depth: np.ndarray,
        intrinsics: np.ndarray,
        camera_to_world: np.ndarray,
        frame_index: int,
        cache_key: str,
        boxes: np.ndarray,
    ) -> None:
        """Feed secondary proposals not claimed by B6 into M1/M2.

        Raw masks and the aligned sensor depth are supplied to the standalone
        observer so its depth-aware connected-components stage remains
        active.  The resulting tracks are diagnostic state only; this method
        has no reference to any output array or supplemental-output manager.
        """

        observer = self.trifusion_missing_graph
        if observer is None:
            return
        assigned_proposal_objects = {
            id(lifted[index].proposal) for index in assignments
        }
        unmatched = [
            (proposal_index, proposal)
            for proposal_index, proposal in enumerate(proposals)
            if id(proposal) not in assigned_proposal_objects
        ]
        observations = [
            {
                "frame_id": int(frame_index),
                "proposal_id": f"{cache_key}:{proposal_index}",
                "mask": proposal.mask,
                "depth": depth,
                "intrinsics": intrinsics,
                "camera_to_world": camera_to_world,
                "score": float(proposal.score),
                "label": proposal.label,
                "feature": proposal.feature,
                "provider": "sam3_teacher_cache",
            }
            for proposal_index, proposal in unmatched
        ]
        provider_call_index = int(self.stats["c4_provider_calls"]) - 1
        update = observer.update(
            observations,
            global_boxes=boxes,
            provider_call_index=provider_call_index,
        )
        self._last_trifusion_missing_update = update
        self.stats["trifusion_missing_provider_calls"] += 1
        self.stats["trifusion_missing_unmatched"] += len(unmatched)
        self.stats["trifusion_missing_components"] += sum(
            int(audit.accepted) for audit in update.observations
        )
        self.stats["trifusion_missing_candidates"] = len(
            update.candidates
        )
        for error in update.errors:
            self.stats["trifusion_missing_errors"][str(error)] += 1

    @staticmethod
    def _raw_unmatched_mask_observations(
        *,
        proposals: Sequence[SupplementalProposal],
        lifted: Sequence[LiftedProposal],
        assignments: Mapping[int, int],
        depth: np.ndarray,
        intrinsics: np.ndarray,
        camera_to_world: np.ndarray,
        frame_index: int,
        cache_key: str,
        provider: str,
    ) -> Tuple[Dict[str, Any], ...]:
        """Adapt raw proposals not claimed by a frozen global.

        Association indices refer to ``lifted`` rather than the raw proposal
        list, so object identity is used to retain proposals which failed the
        primary lifting path.  The residual graph performs its own
        depth-aware lifting and therefore must be allowed to inspect those
        masks as well.
        """

        assigned_objects = {
            id(lifted[index].proposal) for index in assignments
        }
        normalized_provider = str(provider).strip().casefold()
        return tuple(
            {
                "frame_id": int(frame_index),
                "proposal_id": (
                    f"{normalized_provider}:{cache_key}:{proposal_index}"
                ),
                "mask": proposal.mask,
                "depth": depth,
                "intrinsics": intrinsics,
                "camera_to_world": camera_to_world,
                "score": float(proposal.score),
                "label": proposal.label,
                "feature": proposal.feature,
                "provider": normalized_provider,
            }
            for proposal_index, proposal in enumerate(proposals)
            if id(proposal) not in assigned_objects
        )

    def _update_residual_track_graph(
        self,
        observations: Sequence[Mapping[str, Any]],
        *,
        boxes: np.ndarray,
        provider_call_index: int,
    ) -> None:
        """Apply one combined SAM3/YOLOE residual-graph transaction."""

        observer = self.residual_track_graph
        if observer is None or self._residual_track_failed:
            return
        provider_by_proposal = {
            str(observation["proposal_id"]): str(
                observation["provider"]
            ).strip().casefold()
            for observation in observations
        }
        update = observer.update(
            observations,
            global_boxes=boxes,
            provider_call_index=int(provider_call_index),
        )
        self._last_residual_track_update = update
        self.stats["residual_track_provider_calls"] += 1
        for audit in update.observations:
            self.stats["residual_track_observation_reasons"][
                str(audit.reason)
            ] += 1
            if not audit.accepted or audit.track_id is None:
                continue
            provider = provider_by_proposal.get(
                str(audit.proposal_id), "unknown"
            )
            counter = self._residual_track_provider_counts.setdefault(
                int(audit.track_id), Counter()
            )
            counter[provider] += 1
            self.stats["residual_track_provider_nodes"][provider] += 1
            self.stats["residual_track_components"] += 1
        for audit in update.associations:
            self.stats["residual_track_association_reasons"][
                str(audit.reason)
            ] += 1
        self.stats["residual_track_candidates"] = len(
            update.candidates
        )
        for error in update.errors:
            self.stats["residual_track_errors"][str(error)] += 1

    def _process_generic_geometry_keyframe(
        self,
        *,
        image: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        camera_to_world: np.ndarray,
        frame_index: int,
        cache_key: str,
        boxes: np.ndarray,
        scores: np.ndarray,
        stable_ids: np.ndarray,
    ) -> Tuple[Dict[str, Any], ...]:
        """Replay one SAM3 cache frame and return its unmatched raw masks."""

        cfg = self.config["generic_local_geometry_refiner"]
        if not cfg["enabled"] or self._c4_failed:
            return ()
        if self.generic_geometry_provider is None:
            raise RuntimeError(
                "enabled generic geometry observer has no secondary provider"
            )
        if cfg["observe_existing_globals"]:
            self._sync_generic_geometry_global_tracks(
                boxes, scores, stable_ids
            )
        started = time.perf_counter()
        batches = self.generic_geometry_provider.predict(
            [image], frame_ids=[cache_key]
        )
        self.stats["c4_provider_seconds"] += (
            time.perf_counter() - started
        )
        self.stats["c4_provider_calls"] += 1
        if len(batches) != 1:
            raise RuntimeError(
                "C4 secondary provider returned the wrong batch size"
            )
        proposals = batches[0]
        self.stats["c4_proposals"] += len(proposals)
        lifted = self._lift_generic_geometry_proposals(
            proposals,
            depth=depth,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            frame_index=frame_index,
            image_shape=image.shape[:2],
        )
        assignments, _ = self._match_to_globals_detailed(
            lifted, boxes, intrinsics, camera_to_world
        )
        if cfg["observe_existing_globals"]:
            for proposal_index, global_index in assignments.items():
                stable_id = int(stable_ids[global_index])
                evidence = self.generic_geometry_global_tracks.get(
                    stable_id
                )
                if evidence is None:
                    evidence = self._new_generic_geometry_global_evidence(
                        stable_id,
                        boxes[global_index],
                        scores[global_index],
                    )
                self._add_generic_geometry_observation(
                    evidence,
                    lifted[proposal_index],
                    box=boxes[global_index],
                    frame_index=frame_index,
                )
        self.stats["c4_matched_global"] += len(assignments)
        self._update_trifusion_missing_graph(
            proposals=proposals,
            lifted=lifted,
            assignments=assignments,
            depth=depth,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            frame_index=frame_index,
            cache_key=cache_key,
            boxes=boxes,
        )
        residual_cfg = self.config["residual_track_observer"]
        return self._raw_unmatched_mask_observations(
            proposals=proposals,
            lifted=lifted,
            assignments=assignments,
            depth=depth,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            frame_index=frame_index,
            cache_key=cache_key,
            provider=residual_cfg["secondary_provider_name"],
        )

    @staticmethod
    def _mask_graph_snapshot(
        track: Any,
        metadata: Optional[SupplementalEvidence],
        *,
        lifecycle_state: str,
        event_frame: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        if metadata is None or metadata.graph is None:
            return None
        graph = metadata.graph
        track_box = track.memory.aabb
        box = (
            np.full(6, np.nan, dtype=np.float32)
            if track_box is None
            else np.concatenate(track_box).astype(np.float32)
        )
        edges = tuple(graph.edges.values())

        def mean_metric(name: str) -> float:
            values = [
                float(getattr(edge, name))
                for edge in edges
                if getattr(edge, name) is not None
            ]
            return float(np.mean(values)) if values else float("nan")

        return {
            "track_id": int(track.track_id),
            "lifecycle_state": str(lifecycle_state),
            "event_frame": (
                -1 if event_frame is None else int(event_frame)
            ),
            "box": box,
            "hit_count": int(track.hit_count),
            "view_count": int(track.view_count),
            "track_confirmed": bool(track.confirmed),
            "node_count": int(graph.node_count),
            "edge_count": int(graph.edge_count),
            "unique_frame_count": int(graph.unique_frame_count),
            "graph_confirmed": bool(graph.confirmed),
            "confirmation_frame_id": (
                ""
                if graph.confirmation_frame_id is None
                else str(graph.confirmation_frame_id)
            ),
            "mean_edge_score": mean_metric("score"),
            "mean_geometry_score": mean_metric("geometry_score"),
            "mean_iou_3d": mean_metric("iou_3d"),
            "mean_mutual_inside": mean_metric("mutual_inside"),
            "mean_projection_iou": mean_metric("projection_iou"),
            "mean_appearance_cosine": mean_metric(
                "appearance_cosine"
            ),
            "mean_detector_score": float(metadata.stats.mean_score),
            "label": metadata.stats.label or "",
            "rejections": dict(
                sorted(metadata.graph_rejections.items())
            ),
            "memory_view_candidates": int(
                track.memory.view_candidate_count
            ),
            "memory_selected_views": int(
                track.memory.selected_view_count
            ),
            "memory_geometry_points": int(
                track.memory.geometry_num_points
            ),
        }

    def _live_mask_graph_snapshots(self) -> List[Dict[str, Any]]:
        snapshots = [
            {
                **snapshot,
                "box": np.asarray(
                    snapshot["box"], dtype=np.float32
                ).copy(),
                "rejections": dict(snapshot["rejections"]),
            }
            for snapshot in self.retired_mask_graph_snapshots
        ]
        if self.track_manager is None:
            return snapshots
        for track_id, metadata in sorted(
            self.supplemental_metadata.items()
        ):
            if track_id in self.track_manager.tracks:
                track = self.track_manager.tracks[track_id]
                state = "active"
            elif track_id in self.track_manager.archived_tracks:
                track = self.track_manager.archived_tracks[track_id]
                state = "archived"
            else:
                continue
            snapshot = self._mask_graph_snapshot(
                track,
                metadata,
                lifecycle_state=state,
                event_frame=track.last_frame,
            )
            if snapshot is not None:
                snapshots.append(snapshot)
        return snapshots

    def _refresh_fragment_stitch_candidates(
        self,
    ) -> Tuple[FragmentStitchCandidate, ...]:
        """Refresh C3 observer candidates without touching detection state."""

        if not self.config["fragment_stitch"]["enabled"]:
            self._last_fragment_stitch_candidates = ()
            self._last_fragment_stitch_graph_snapshots = ()
            self._fragment_stitch_refresh_complete = False
            self._last_fragment_stitch_error = ""
            self.stats["fragment_stitch_seconds"] = 0.0
            self.stats["fragment_stitch_invalid_snapshots"] = 0
            self.stats["fragment_stitch_fail_open"] = 0
            return ()
        started = time.perf_counter()
        self._last_fragment_stitch_candidates = ()
        self._last_fragment_stitch_graph_snapshots = ()
        self._fragment_stitch_refresh_complete = False
        self._last_fragment_stitch_error = ""
        self.stats["fragment_stitch_invalid_snapshots"] = 0
        self.stats["fragment_stitch_fail_open"] = 0
        try:
            snapshots = self._live_mask_graph_snapshots()
            valid_snapshots = []
            for snapshot in snapshots:
                try:
                    # The pure builder owns the strict diagnostics contract.
                    # A singleton invocation validates one snapshot without
                    # ever producing a stitch candidate.
                    build_fragment_stitch_candidates(
                        [snapshot],
                        self.config["fragment_stitch"],
                    )
                except Exception:
                    self.stats[
                        "fragment_stitch_invalid_snapshots"
                    ] += 1
                    continue
                valid_snapshots.append(snapshot)
            # Summary and NPZ diagnostics reuse only rows that passed the
            # strict fragment contract.  Caching malformed rows here would
            # reintroduce a failure later during dtype conversion.
            self._last_fragment_stitch_graph_snapshots = tuple(
                valid_snapshots
            )
            self._last_fragment_stitch_candidates = (
                build_fragment_stitch_candidates(
                    valid_snapshots,
                    self.config["fragment_stitch"],
                )
            )
        except Exception as error:
            # C3 is observer-only.  Diagnostics must fail open so malformed
            # memory or a future clustering regression can never suppress an
            # otherwise valid C2 result.
            self._last_fragment_stitch_candidates = ()
            self.stats["fragment_stitch_fail_open"] = 1
            self._last_fragment_stitch_error = (
                f"{type(error).__name__}: {error}"
            )
        finally:
            self._fragment_stitch_refresh_complete = True
            self.stats["fragment_stitch_seconds"] = (
                time.perf_counter() - started
            )
        return self._last_fragment_stitch_candidates

    def _diagnostic_mask_graph_snapshots(
        self,
    ) -> List[Dict[str, Any]]:
        """Reuse C3's fail-open lifecycle snapshot instead of rebuilding it."""

        if (
            self.config["fragment_stitch"]["enabled"]
            and self._fragment_stitch_refresh_complete
        ):
            return list(self._last_fragment_stitch_graph_snapshots)
        return self._live_mask_graph_snapshots()

    def _absorb_candidate_track(
        self,
        evidence: GlobalEvidence,
        item: LiftedProposal,
        *,
        frame_index: int,
        match: GlobalProposalMatch,
    ) -> None:
        if self.track_manager is None:
            return
        threshold = float(
            self.config["matching"]["absorb_supplemental_iou"]
        )
        candidates = []
        for archived, tracks in (
            (False, self.track_manager.tracks),
            (True, self.track_manager.archived_tracks),
        ):
            for track_id, track in tracks.items():
                track_box = track.memory.aabb
                if track_box is None:
                    continue
                overlap = aabb_iou(
                    item.box[:3],
                    item.box[3:6],
                    track_box[0],
                    track_box[1],
                )
                if overlap >= threshold:
                    # Prefer the highest overlap, then an active track, then
                    # the stable track id.
                    candidates.append((-overlap, archived, track_id))
        if not candidates:
            return
        negative_overlap, archived, track_id = min(candidates)
        trigger_overlap = float(-negative_overlap)
        source = (
            self.track_manager.archived_tracks
            if archived
            else self.track_manager.tracks
        )
        track = source.pop(track_id)
        metadata = self.supplemental_metadata.pop(track_id, None)
        snapshot = self._mask_graph_snapshot(
            track,
            metadata,
            lifecycle_state="absorbed",
            event_frame=frame_index,
        )
        if snapshot is not None:
            self.retired_mask_graph_snapshots.append(snapshot)
        recovery_cfg = self.config["supplemental_output"]
        graph = None if metadata is None else metadata.graph
        if (
            recovery_cfg["recover_absorbed_confirmed"]
            and bool(track.confirmed)
            and metadata is not None
            and graph is not None
            and bool(graph.confirmed)
        ):
            self.absorbed_supplemental_records[int(track_id)] = (
                _AbsorbedSupplementalRecord(
                    track=track,
                    metadata=metadata,
                    absorbed_global_stable_id=int(evidence.stable_id),
                    event_frame=int(frame_index),
                    trigger_overlap=trigger_overlap,
                    match=match,
                )
            )
            self.stats["absorbed_recovery_stored"] += 1
        if track.memory.num_points:
            summary = track.memory.quality_summary()
            evidence.memory.add_observation(
                ObjectObservation(
                    points_world=track.memory.points,
                    confidence=float(
                        summary.get("mean_confidence") or 0.5
                    ),
                    projection_mask_iou=float(
                        summary.get("mean_projection_mask_iou") or 0.5
                    ),
                ),
                frame_index,
                record_view_candidate=False,
            )
            expansion = float(
                self.config["matching"]["crop_to_global_expansion"]
            )
            evidence.memory.merge_view_candidates_from(
                track.memory,
                crop_center=evidence.last_box[:3],
                crop_dims=evidence.last_box[3:6] * expansion,
                minimum_points=int(
                    self.object_config["min_points_for_aabb"]
                ),
            )
        evidence.stats.absorbed_views += int(track.view_count)
        if metadata is not None:
            evidence.stats.merge_from(
                metadata.stats,
                max_views=int(
                    self.config["quality"]["max_view_records"]
                ),
            )

    def _record_candidate_retirement(
        self,
        result: Any,
        prior_tracks: Optional[Mapping[int, Any]] = None,
    ) -> None:
        """Keep archived metadata and discard only unconfirmed-track metadata."""

        archived = set(result.archived_track_ids)
        discarded = set(result.discarded_track_ids)
        prior_tracks = {} if prior_tracks is None else prior_tracks
        for discarded_id in discarded:
            metadata = self.supplemental_metadata.get(discarded_id)
            track = prior_tracks.get(discarded_id)
            if track is not None:
                snapshot = self._mask_graph_snapshot(
                    track,
                    metadata,
                    lifecycle_state="discarded",
                    event_frame=track.last_frame,
                )
                if snapshot is not None:
                    self.retired_mask_graph_snapshots.append(snapshot)
            self.supplemental_metadata.pop(discarded_id, None)
        # Defensive compatibility: an implementation returning only
        # ``expired_track_ids`` must not leak metadata for discarded tracks.
        for expired_id in result.expired_track_ids:
            if expired_id not in archived and expired_id not in discarded:
                metadata = self.supplemental_metadata.get(expired_id)
                track = prior_tracks.get(expired_id)
                if track is not None:
                    snapshot = self._mask_graph_snapshot(
                        track,
                        metadata,
                        lifecycle_state="expired",
                        event_frame=track.last_frame,
                    )
                    if snapshot is not None:
                        self.retired_mask_graph_snapshots.append(
                            snapshot
                        )
                self.supplemental_metadata.pop(expired_id, None)
        self.stats["candidate_archived"] += len(result.archived_track_ids)
        self.stats["candidate_discarded"] += len(
            result.discarded_track_ids
        )

    def _advance_candidate_lifecycle(
        self,
        *,
        frame_index: int,
        lifecycle_step: int,
    ) -> None:
        if self.track_manager is None:
            return
        prior_tracks = dict(self.track_manager.tracks)
        result = self.track_manager.update(
            [],
            frame_index,
            lifecycle_step=lifecycle_step,
        )
        self._record_candidate_retirement(result, prior_tracks)

    def _update_candidates(
        self,
        unmatched: Sequence[LiftedProposal],
        *,
        frame_index: int,
        lifecycle_step: int,
    ) -> None:
        if self.track_manager is None:
            return
        graph_cfg = self.config["mask_graph"]
        graph_enabled = bool(graph_cfg["enabled"])
        pair_compatibility: Optional[Dict[Tuple[int, int], float]] = None
        graph_edges: Dict[
            Tuple[int, int], Tuple[MaskGraphEdge, MaskGraphNode]
        ] = {}
        graph_nodes: List[MaskGraphNode] = []
        projection_contexts: List[Any] = []
        projection_context_ready: List[bool] = []
        if graph_enabled:
            pair_compatibility = {}
            for local_index, item in enumerate(unmatched):
                node = coerce_mask_graph_node(
                    item,
                    node_id=(
                        f"frame:{frame_index}:proposal:{local_index}"
                    ),
                    frame_id=frame_index,
                )
                graph_nodes.append(node)
                projection_contexts.append(None)
                projection_context_ready.append(False)
            for track_id, track in sorted(
                self.track_manager.tracks.items()
            ):
                metadata = self.supplemental_metadata.get(track_id)
                graph = (
                    metadata.graph if metadata is not None else None
                )
                if graph is None or not graph.nodes:
                    continue
                track_evidence: MaskGraphTrackEvidence = (
                    coerce_track_evidence(track, graph)
                )
                for local_index, node in enumerate(graph_nodes):
                    if not projection_context_ready[local_index]:
                        projection_contexts[local_index] = (
                            build_projection_context(node, graph_cfg)
                        )
                        projection_context_ready[local_index] = True
                    edge = evaluate_edge(
                        track_evidence,
                        graph,
                        node,
                        projection_context=(
                            projection_contexts[local_index]
                        ),
                    )
                    self.stats["mask_graph_edges_evaluated"] += 1
                    if edge.accepted:
                        pair_compatibility[
                            (track_id, local_index)
                        ] = float(edge.score)
                        graph_edges[(track_id, local_index)] = (
                            edge,
                            node,
                        )
                        self.stats["mask_graph_edges_accepted"] += 1
                    else:
                        self.stats["mask_graph_rejected"][
                            edge.reason
                        ] += 1
                        metadata.graph_rejections[edge.reason] += 1
        prior_tracks = dict(self.track_manager.tracks)
        result = self.track_manager.update(
            [item.observation for item in unmatched],
            frame_index,
            lifecycle_step=lifecycle_step,
            pair_compatibility=pair_compatibility,
        )
        self._record_candidate_retirement(result, prior_tracks)
        for local_index, track_id in result.assignments.items():
            item = unmatched[local_index]
            metadata = self.supplemental_metadata.get(track_id)
            if metadata is None:
                metadata = SupplementalEvidence(
                    track_id=track_id,
                    graph=(
                        MaskGraphState(track_id, graph_cfg)
                        if graph_enabled
                        else None
                    ),
                )
                self.supplemental_metadata[track_id] = metadata
            if graph_enabled:
                graph = metadata.graph
                if graph is None:
                    raise RuntimeError(
                        "enabled mask graph has no per-track state"
                    )
                track = self.track_manager.tracks[track_id]
                if track_id in result.created_track_ids:
                    update = update_mask_graph(
                        track,
                        graph,
                        graph_nodes[local_index],
                        projection_context=(
                            projection_contexts[local_index]
                        ),
                    )
                    if not update.accepted or not update.seeded:
                        raise RuntimeError(
                            "new mask-graph track failed to seed"
                        )
                    self.stats["mask_graph_nodes"] += 1
                    if update.became_confirmed:
                        self.stats["mask_graph_confirmed"] += 1
                else:
                    edge_record = graph_edges.get(
                        (track_id, local_index)
                    )
                    if edge_record is None:
                        raise RuntimeError(
                            "mask-graph assignment has no accepted edge"
                        )
                    edge, node = edge_record
                    became_confirmed = graph.add_node(node)
                    graph.add_edge(edge)
                    self.stats["mask_graph_nodes"] += 1
                    if became_confirmed:
                        self.stats["mask_graph_confirmed"] += 1
            metadata.stats.record(
                item.proposal,
                item.view,
                max_views=int(self.config["quality"]["max_view_records"]),
                box=item.box,
            )
        self.stats["candidate_updates"] += len(result.assignments)

    def process_keyframe(
        self,
        *,
        image: Any,
        depth: Any,
        intrinsics: Any,
        camera_to_world: Any,
        frame_id: int,
        scene_id: str,
        cache_frame_id: Optional[str] = None,
        global_corners: Any,
        global_scores: Any,
        stable_ids: Any,
    ) -> None:
        """Run one scheduled proposal/memory update.

        The method never mutates the supplied global arrays.
        """

        if not self.enabled:
            return
        if not isinstance(frame_id, (int, np.integer)) or int(frame_id) < 0:
            raise ValueError("frame_id must be a non-negative integer")
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError("scene_id must be a non-empty string")
        if cache_frame_id is not None and (
            not isinstance(cache_frame_id, str)
            or not cache_frame_id.strip()
        ):
            raise ValueError("cache_frame_id must be a non-empty string")
        image, depth, intrinsics, camera_to_world = _validate_runtime_arrays(
            image, depth, intrinsics, camera_to_world
        )
        boxes, scores, ids = self._global_inputs(
            global_corners, global_scores, stable_ids
        )
        requested_scene = scene_id.strip()
        if self.scene_id is None:
            self.scene_id = requested_scene
        elif self.scene_id != requested_scene:
            self.reset_scene(requested_scene)
        frame_index = self.keyframe_count
        self.keyframe_count += 1
        self.stats["keyframes"] += 1
        self._sync_global_tracks(boxes, scores, ids)

        interval = int(self.config["inference_every_keyframes"])
        if frame_index % interval != 0:
            if (
                self.config["candidate_lifecycle"]["ttl_clock"]
                == "keyframe"
            ):
                self._advance_candidate_lifecycle(
                    frame_index=frame_index,
                    lifecycle_step=frame_index,
                )
            return

        if self.provider is None:
            raise RuntimeError("enabled controller has no proposal provider")
        provider_step = int(self.stats["provider_calls"])
        provider_cache_key = (
            f"{self.scene_id}:"
            + (
                cache_frame_id.strip()
                if cache_frame_id is not None
                else f"{int(frame_id):06d}"
            )
        )
        started = time.perf_counter()
        batches = self.provider.predict(
            [image],
            frame_ids=[provider_cache_key],
        )
        self.stats["provider_seconds"] += time.perf_counter() - started
        self.stats["provider_calls"] += 1
        if len(batches) != 1:
            raise RuntimeError("proposal provider returned the wrong batch size")
        lifecycle_step = (
            provider_step
            if self.config["candidate_lifecycle"]["ttl_clock"]
            == "provider_call"
            else frame_index
        )
        self._advance_candidate_lifecycle(
            frame_index=frame_index,
            lifecycle_step=lifecycle_step,
        )
        proposals = batches[0]
        if self.appearance_encoder is not None and proposals:
            appearance_started = time.perf_counter()
            encoded = self.appearance_encoder(image, proposals)
            encoded = list(encoded)
            self.stats["appearance_seconds"] += (
                time.perf_counter() - appearance_started
            )
            if len(encoded) != len(proposals):
                raise RuntimeError(
                    "appearance encoder returned the wrong number of features"
                )
            proposals = [
                replace(
                    proposal,
                    feature=(
                        proposal.feature
                        if proposal.feature is not None
                        else np.asarray(feature, dtype=np.float32)
                    ),
                )
                for proposal, feature in zip(proposals, encoded)
            ]
        self.stats["proposals"] += len(proposals)
        lifted = self._lift_proposals(
            proposals,
            depth=depth,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            frame_index=frame_index,
            image_shape=image.shape[:2],
        )
        assignments, match_details = self._match_to_globals_detailed(
            lifted, boxes, intrinsics, camera_to_world
        )
        for proposal_index, global_index in assignments.items():
            stable_id = int(ids[global_index])
            evidence = self.global_tracks.get(stable_id)
            if evidence is None:
                evidence = self._new_global_evidence(
                    stable_id, boxes[global_index], scores[global_index]
                )
            item = lifted[proposal_index]
            self._add_global_observation(
                evidence,
                item,
                box=boxes[global_index],
                frame_index=frame_index,
            )
            match = match_details[proposal_index]
            if match.strong:
                self.stats["strong_global_matches"] += 1
            else:
                self.stats["weak_global_matches"] += 1
            identity_cfg = self.config["missing_track_identity"]
            shadow_weak = bool(
                identity_cfg["enabled"]
                and identity_cfg["shadow_weak_global_matches"]
                and not match.strong
            )
            if not shadow_weak:
                self._absorb_candidate_track(
                    evidence,
                    item,
                    frame_index=frame_index,
                    match=match,
                )
        unmatched_indices = [
            index for index in range(len(lifted)) if index not in assignments
        ]
        self.stats["unmatched_candidates"] += len(unmatched_indices)
        candidate_indices = list(unmatched_indices)
        identity_cfg = self.config["missing_track_identity"]
        if (
            identity_cfg["enabled"]
            and identity_cfg["shadow_weak_global_matches"]
        ):
            weak_indices = [
                index
                for index in sorted(assignments)
                if not match_details[index].strong
            ]
            candidate_indices.extend(weak_indices)
            self.stats["weak_shadow_candidates"] += len(weak_indices)
        candidate_index_set = set(candidate_indices)
        missing_candidates = [
            item
            for index, item in enumerate(lifted)
            if index in candidate_index_set
        ]
        residual_cfg = self.config["residual_track_observer"]
        residual_source_mode = residual_cfg["source_mode"]
        primary_residual_observations: Tuple[Dict[str, Any], ...] = ()
        if (
            self.residual_track_graph is not None
            and residual_source_mode in {"yoloe", "dual"}
        ):
            primary_residual_observations = (
                self._raw_unmatched_mask_observations(
                    proposals=proposals,
                    lifted=lifted,
                    assignments=assignments,
                    depth=depth,
                    intrinsics=intrinsics,
                    camera_to_world=camera_to_world,
                    frame_index=frame_index,
                    cache_key=provider_cache_key,
                    provider=residual_cfg["primary_provider_name"],
                )
            )
            self.stats["residual_track_primary_unmatched"] += len(
                primary_residual_observations
            )
        self._update_candidates(
            missing_candidates,
            frame_index=frame_index,
            lifecycle_step=lifecycle_step,
        )
        self.stats["matched_global"] += len(assignments)
        secondary_residual_observations: Tuple[Dict[str, Any], ...] = ()
        try:
            secondary_residual_observations = (
                self._process_generic_geometry_keyframe(
                    image=image,
                    depth=depth,
                    intrinsics=intrinsics,
                    camera_to_world=camera_to_world,
                    frame_index=frame_index,
                    cache_key=provider_cache_key,
                    boxes=boxes,
                    scores=scores,
                    stable_ids=ids,
                )
            )
        except Exception as error:
            cfg = self.config["generic_local_geometry_refiner"]
            if not cfg["enabled"] or not cfg["fail_open"]:
                raise
            # An incomplete secondary stream must never become a partial
            # geometry experiment.  Drop all C4 evidence for this scene while
            # preserving the already-computed B6 state and output exactly.
            self.generic_geometry_global_tracks.clear()
            self._last_c4_runtime.clear()
            self._c4_failed = True
            self.stats["c4_fail_open"] = 1
            self._last_c4_error = (
                f"{type(error).__name__}: {error}"
            )
            if (
                self.residual_track_graph is not None
                and residual_source_mode in {"sam3", "dual"}
            ):
                # A partial SAM3 scene is not a valid residual-track
                # experiment.  Fail closed to an empty observer while the
                # frozen B6 output continues unchanged.
                self.residual_track_graph.reset()
                self._last_residual_track_candidates = ()
                self._last_residual_track_decisions = ()
                self._last_residual_track_update = None
                self._residual_track_provider_counts.clear()
                self._residual_track_failed = True
                self._last_residual_track_error = self._last_c4_error
                self.stats["residual_track_fail_closed"] = 1
        if (
            self.residual_track_graph is not None
            and not self._residual_track_failed
        ):
            selected_secondary = (
                secondary_residual_observations
                if residual_source_mode in {"sam3", "dual"}
                else ()
            )
            self.stats["residual_track_secondary_unmatched"] += len(
                selected_secondary
            )
            # Both providers enter one transaction.  This is essential:
            # MissingInstanceGraphObserver requires a strictly increasing
            # provider-call index and same-view cross-provider NMS must see
            # the two sources together.
            self._update_residual_track_graph(
                (
                    *primary_residual_observations,
                    *selected_secondary,
                ),
                boxes=boxes,
                provider_call_index=provider_step,
            )

    def _mean_projection_iou(
        self,
        box: np.ndarray,
        stats: EvidenceStats,
        frame_ids: Optional[Sequence[int]] = None,
    ) -> float:
        records = stats.view_records
        if frame_ids is not None:
            selected_frames = {int(frame_id) for frame_id in frame_ids}
            records = [
                view
                for view in records
                if int(view.frame_index) in selected_frames
            ]
        if not records:
            return 0.0
        values = np.asarray(
            [
                self._projection_iou_for_view(box, view)
                for view in records
            ],
            dtype=np.float32,
        )
        weights = np.asarray(
            [max(view.score, 1e-4) for view in records],
            dtype=np.float32,
        )
        return float(np.average(values, weights=weights))

    def _mean_projection_iou_corners(
        self,
        corners: np.ndarray,
        stats: EvidenceStats,
        frame_ids: Optional[Sequence[int]] = None,
    ) -> float:
        records = stats.view_records
        if frame_ids is not None:
            selected_frames = {int(frame_id) for frame_id in frame_ids}
            records = [
                view
                for view in records
                if int(view.frame_index) in selected_frames
            ]
        if not records:
            return 0.0
        values = np.asarray(
            [
                self._projection_iou_for_corners(corners, view)
                for view in records
            ],
            dtype=np.float32,
        )
        weights = np.asarray(
            [max(view.score, 1e-4) for view in records],
            dtype=np.float32,
        )
        return float(np.average(values, weights=weights))

    @staticmethod
    def _box_stability(original: np.ndarray, final: np.ndarray) -> float:
        diagonal = max(float(np.linalg.norm(original[3:6])), 1e-6)
        center_shift = float(np.linalg.norm(final[:3] - original[:3]))
        log_scale = float(
            np.mean(
                np.abs(
                    np.log(
                        np.maximum(final[3:6], 1e-6)
                        / np.maximum(original[3:6], 1e-6)
                    )
                )
            )
        )
        return float(np.exp(-(center_shift / diagonal + log_scale)))

    def _quality_mapping(
        self,
        *,
        original_box: np.ndarray,
        final_box: np.ndarray,
        detector_score: float,
        memory: Optional[ObjectGeometryMemory],
        stats: Optional[EvidenceStats],
        supplemental: bool,
        refiner_quality: float,
    ) -> Dict[str, float]:
        quality_cfg = self.config["quality"]
        quality_refiner_override = quality_cfg[
            "refiner_quality_override"
        ]
        if quality_refiner_override is not None:
            # Keep B5's predicted quality available to its geometry gate, but
            # preserve the frozen B6 scorer's training-time feature contract.
            refiner_quality = float(quality_refiner_override)
        if memory is None or stats is None:
            return {
                "detector_score": float(np.clip(detector_score, 0.0, 1.0)),
                "mask_confidence": 0.0,
                "valid_depth_ratio": 0.0,
                "depth_support": 0.0,
                "projection_iou": 0.0,
                "geometry_consistency": 1.0,
                "appearance_consistency": 0.5,
                "view_count_quality": 0.0,
                "box_stability": 0.5,
                "source_agreement": 0.0,
                "area_quality": 0.0,
                "refiner_quality": float(np.clip(refiner_quality, 0.0, 1.0)),
            }
        summary = memory.quality_summary()
        view_count = memory.unique_view_count + stats.absorbed_views
        point_support = points_inside_aabb_fraction(
            memory.points, final_box[:3], final_box[3:6]
        )
        geometry_consistency = 0.5 * (
            aabb_iou(
                original_box[:3],
                original_box[3:6],
                final_box[:3],
                final_box[3:6],
            )
            + point_support
        )
        area_quality = (
            float(np.mean([view.area_ratio for view in stats.view_records]))
            if stats.view_records
            else 0.0
        )
        source_agreement = (
            min(view_count / 2.0, 1.0)
            if supplemental
            else min(0.5 + view_count / 4.0, 1.0)
        )
        refit_stability = self._box_stability(original_box, final_box)
        temporal_stability = stats.temporal_box_stability(final_box)
        combined_stability = float(
            np.sqrt(refit_stability * temporal_stability)
        )
        return {
            "detector_score": float(np.clip(detector_score, 0.0, 1.0)),
            "mask_confidence": float(np.clip(stats.mean_score, 0.0, 1.0)),
            "valid_depth_ratio": float(
                np.clip(summary.get("mean_valid_depth_ratio") or 0.0, 0.0, 1.0)
            ),
            "depth_support": float(
                np.clip(
                    np.log1p(memory.num_points)
                    / np.log1p(
                        float(quality_cfg["support_reference_points"])
                    ),
                    0.0,
                    1.0,
                )
            ),
            "projection_iou": float(
                np.clip(self._mean_projection_iou(final_box, stats), 0.0, 1.0)
            ),
            "geometry_consistency": float(
                np.clip(geometry_consistency, 0.0, 1.0)
            ),
            "appearance_consistency": stats.appearance_consistency,
            "view_count_quality": float(
                np.clip(
                    view_count / float(quality_cfg["target_views"]),
                    0.0,
                    1.0,
                )
            ),
            "box_stability": combined_stability,
            "source_agreement": float(np.clip(source_agreement, 0.0, 1.0)),
            "area_quality": float(np.clip(area_quality / 0.10, 0.0, 1.0)),
            "refiner_quality": float(np.clip(refiner_quality, 0.0, 1.0)),
        }

    def _visibility_aware_candidate(
        self,
        original: np.ndarray,
        evidence: GlobalEvidence,
        *,
        selected_view_records: Optional[
            Sequence[MemoryViewRecord]
        ] = None,
    ) -> Tuple[np.ndarray, str]:
        """Build a conservative box from independently observed boundaries.

        A masked depth image normally contains only the object surfaces facing
        that camera.  Treating its full point AABB as six complete object
        boundaries therefore causes systematic shrinkage.  This strategy
        leaves an axis untouched unless selected Top-K cameras either observe
        the object from both sides of that axis or provide sufficiently
        separated silhouette views nearly perpendicular to it.  Opposing
        views estimate each boundary independently; silhouette views jointly
        estimate both boundaries.
        """

        cfg = self.config["refit"]
        original_lower = (
            np.asarray(original[:3], dtype=np.float64)
            - 0.5 * np.asarray(original[3:6], dtype=np.float64)
        )
        original_upper = (
            np.asarray(original[:3], dtype=np.float64)
            + 0.5 * np.asarray(original[3:6], dtype=np.float64)
        )
        crop_expansion = float(
            cfg["visibility_point_crop_expansion"]
        )
        minimum_record_points = max(
            8,
            int(self.object_config["min_points_for_aabb"]) // 4,
        )
        records: List[MemoryViewRecord] = []
        record_points: List[np.ndarray] = []
        directions = []
        source_records = (
            evidence.memory.selected_view_records
            if selected_view_records is None
            else selected_view_records
        )
        for record in source_records:
            if record.camera_position is None:
                continue
            camera_position = np.asarray(
                record.camera_position, dtype=np.float64
            )
            camera_inside = np.all(
                camera_position >= original_lower
            ) and np.all(camera_position <= original_upper)
            if camera_inside:
                # A camera centre inside the current object box cannot be
                # trusted as boundary-visibility evidence.  This can happen
                # for an oversized or erroneous upstream box.
                continue
            vector = np.asarray(
                camera_position - original[:3],
                dtype=np.float64,
            )
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(norm) or norm <= 1e-8:
                continue
            inside = points_inside_aabb(
                record.points_world,
                original[:3],
                original[3:6] * crop_expansion,
            )
            cropped = record.points_world[inside]
            if cropped.shape[0] < minimum_record_points:
                continue
            records.append(record)
            record_points.append(cropped)
            directions.append(vector / norm)
        if len(records) < int(cfg["min_views"]):
            return original.copy(), "visibility_views"

        directions_array = np.asarray(directions, dtype=np.float64)
        candidate_lower = original_lower.copy()
        candidate_upper = original_upper.copy()
        minimum_axis_cosine = float(cfg["minimum_axis_cosine"])
        minimum_side_views = int(cfg["minimum_side_views"])
        minimum_separation = float(
            cfg["minimum_view_separation_degrees"]
        )
        boundary_blend = float(cfg["blend"])
        maximum_shift_ratio = float(cfg["max_boundary_shift_ratio"])
        minimum_change_ratio = float(
            cfg["minimum_boundary_change_ratio"]
        )
        boundary_quantile = float(
            cfg["visibility_boundary_quantile"]
        )
        lower_quantile = boundary_quantile
        upper_quantile = 1.0 - boundary_quantile
        camera_outside_ratio = float(
            cfg["minimum_camera_outside_ratio"]
        )
        maximum_measurement_spread_ratio = float(
            cfg["maximum_boundary_measurement_spread_ratio"]
        )
        enable_silhouette_axes = bool(cfg["enable_silhouette_axes"])
        maximum_silhouette_axis_cosine = float(
            cfg["maximum_silhouette_axis_cosine"]
        )
        minimum_silhouette_views = int(
            cfg["minimum_silhouette_views"]
        )
        minimum_silhouette_separation = float(
            cfg["minimum_silhouette_separation_degrees"]
        )
        select_best_silhouette_pair = bool(
            cfg["select_best_silhouette_pair"]
        )
        padding = float(cfg["extent_padding"])
        changed_axes = 0

        for axis in range(3):
            dimension = max(float(original[3 + axis]), 1e-6)
            camera_margin = camera_outside_ratio * dimension
            lower_indices = np.flatnonzero(
                directions_array[:, axis] <= -minimum_axis_cosine
            )
            lower_indices = np.asarray(
                [
                    int(index)
                    for index in lower_indices
                    if float(records[int(index)].camera_position[axis])
                    <= original_lower[axis] - camera_margin
                ],
                dtype=np.int64,
            )
            upper_indices = np.flatnonzero(
                directions_array[:, axis] >= minimum_axis_cosine
            )
            upper_indices = np.asarray(
                [
                    int(index)
                    for index in upper_indices
                    if float(records[int(index)].camera_position[axis])
                    >= original_upper[axis] + camera_margin
                ],
                dtype=np.int64,
            )
            use_silhouette = False
            opposing_valid = (
                len(lower_indices) >= minimum_side_views
                and len(upper_indices) >= minimum_side_views
            )
            if opposing_valid:
                pair_dots = np.clip(
                    directions_array[lower_indices]
                    @ directions_array[upper_indices].T,
                    -1.0,
                    1.0,
                )
                maximum_separation = float(
                    np.degrees(np.arccos(np.min(pair_dots)))
                )
                opposing_valid = (
                    maximum_separation + 1e-8 >= minimum_separation
                )

            if not opposing_valid and enable_silhouette_axes:
                silhouette_indices = np.flatnonzero(
                    np.abs(directions_array[:, axis])
                    <= maximum_silhouette_axis_cosine
                )
                if len(silhouette_indices) >= minimum_silhouette_views:
                    plane_directions = directions_array[
                        silhouette_indices
                    ].copy()
                    plane_directions[:, axis] = 0.0
                    plane_norms = np.linalg.norm(
                        plane_directions, axis=1
                    )
                    valid_plane = plane_norms > 1e-8
                    silhouette_indices = silhouette_indices[valid_plane]
                    plane_directions = plane_directions[valid_plane]
                    plane_norms = plane_norms[valid_plane]
                    if len(silhouette_indices) >= minimum_silhouette_views:
                        plane_directions /= plane_norms[:, None]
                        if select_best_silhouette_pair:
                            best_pair = None
                            for first in range(
                                len(silhouette_indices) - 1
                            ):
                                for second in range(
                                    first + 1,
                                    len(silhouette_indices),
                                ):
                                    pair_dot = float(
                                        np.clip(
                                            plane_directions[first]
                                            @ plane_directions[second],
                                            -1.0,
                                            1.0,
                                        )
                                    )
                                    separation = float(
                                        np.degrees(np.arccos(pair_dot))
                                    )
                                    if (
                                        separation + 1e-8
                                        < minimum_silhouette_separation
                                    ):
                                        continue
                                    pair = np.asarray(
                                        [
                                            silhouette_indices[first],
                                            silhouette_indices[second],
                                        ],
                                        dtype=np.int64,
                                    )
                                    pair_lower = np.asarray(
                                        [
                                            np.quantile(
                                                record_points[
                                                    int(index)
                                                ][:, axis],
                                                lower_quantile,
                                            )
                                            for index in pair
                                        ],
                                        dtype=np.float64,
                                    )
                                    pair_upper = np.asarray(
                                        [
                                            np.quantile(
                                                record_points[
                                                    int(index)
                                                ][:, axis],
                                                upper_quantile,
                                            )
                                            for index in pair
                                        ],
                                        dtype=np.float64,
                                    )
                                    spread = max(
                                        float(np.ptp(pair_lower)),
                                        float(np.ptp(pair_upper)),
                                    ) / dimension
                                    if (
                                        spread
                                        > maximum_measurement_spread_ratio
                                    ):
                                        continue
                                    qualities = [
                                        float(
                                            records[int(index)].quality
                                        )
                                        for index in pair
                                    ]
                                    frame_ids = sorted(
                                        int(records[int(index)].frame_id)
                                        for index in pair
                                    )
                                    rank = (
                                        spread,
                                        -min(qualities),
                                        -separation,
                                        frame_ids[0],
                                        frame_ids[1],
                                    )
                                    if (
                                        best_pair is None
                                        or rank < best_pair[0]
                                    ):
                                        best_pair = (rank, pair)
                            if best_pair is not None:
                                lower_indices = best_pair[1]
                                upper_indices = best_pair[1]
                                use_silhouette = True
                        else:
                            pair_dots = np.clip(
                                plane_directions
                                @ plane_directions.T,
                                -1.0,
                                1.0,
                            )
                            off_diagonal = ~np.eye(
                                len(silhouette_indices), dtype=bool
                            )
                            maximum_separation = float(
                                np.degrees(
                                    np.arccos(
                                        np.min(
                                            pair_dots[off_diagonal]
                                        )
                                    )
                                )
                            )
                            if (
                                maximum_separation + 1e-8
                                >= minimum_silhouette_separation
                            ):
                                lower_indices = silhouette_indices
                                upper_indices = silhouette_indices
                                use_silhouette = True

            if not opposing_valid and not use_silhouette:
                continue

            if (
                len(lower_indices) < minimum_side_views
                or len(upper_indices) < minimum_side_views
            ):
                continue

            lower_samples = np.asarray(
                [
                    np.quantile(
                        record_points[int(index)][:, axis],
                        lower_quantile,
                    )
                    for index in lower_indices
                ],
                dtype=np.float64,
            )
            upper_samples = np.asarray(
                [
                    np.quantile(
                        record_points[int(index)][:, axis],
                        upper_quantile,
                    )
                    for index in upper_indices
                ],
                dtype=np.float64,
            )
            if (
                len(lower_samples) > 1
                and float(np.ptp(lower_samples))
                > maximum_measurement_spread_ratio * dimension
            ) or (
                len(upper_samples) > 1
                and float(np.ptp(upper_samples))
                > maximum_measurement_spread_ratio * dimension
            ):
                continue
            if use_silhouette:
                lower_weights = np.asarray(
                    [
                        max(float(records[int(index)].quality), 1e-6)
                        * (
                            1.0
                            - float(
                                abs(
                                    directions_array[int(index), axis]
                                )
                            )
                        )
                        ** 2
                        for index in lower_indices
                    ],
                    dtype=np.float64,
                )
                upper_weights = lower_weights.copy()
            else:
                lower_weights = np.asarray(
                    [
                        max(float(records[int(index)].quality), 1e-6)
                        * float(
                            abs(directions_array[int(index), axis])
                        )
                        ** 2
                        for index in lower_indices
                    ],
                    dtype=np.float64,
                )
                upper_weights = np.asarray(
                    [
                        max(float(records[int(index)].quality), 1e-6)
                        * float(
                            abs(directions_array[int(index), axis])
                        )
                        ** 2
                        for index in upper_indices
                    ],
                    dtype=np.float64,
                )
            target_lower = float(
                _deterministic_weighted_median(
                    lower_samples,
                    lower_weights,
                    np.asarray(
                        [
                            records[int(index)].frame_id
                            for index in lower_indices
                        ],
                        dtype=np.int64,
                    ),
                )
                - padding
            )
            target_upper = float(
                _deterministic_weighted_median(
                    upper_samples,
                    upper_weights,
                    np.asarray(
                        [
                            records[int(index)].frame_id
                            for index in upper_indices
                        ],
                        dtype=np.int64,
                    ),
                )
                + padding
            )
            if (
                not np.isfinite(target_lower)
                or not np.isfinite(target_upper)
                or target_lower >= target_upper
            ):
                continue

            maximum_shift = maximum_shift_ratio * dimension
            # First B3-v2 is deliberately shrink-only.  Depth points outside
            # the original box may be foreground leakage; expansion requires
            # a separately validated consensus rule.
            lower_shift = min(
                boundary_blend
                * max(target_lower - original_lower[axis], 0.0),
                maximum_shift,
            )
            upper_shift = min(
                boundary_blend
                * max(original_upper[axis] - target_upper, 0.0),
                maximum_shift,
            )
            proposed_lower = original_lower[axis] + lower_shift
            proposed_upper = original_upper[axis] - upper_shift
            if proposed_lower >= proposed_upper:
                continue
            relative_change = max(
                abs(float(proposed_lower - original_lower[axis])),
                abs(float(proposed_upper - original_upper[axis])),
            ) / dimension
            if relative_change < minimum_change_ratio:
                continue
            candidate_lower[axis] = proposed_lower
            candidate_upper[axis] = proposed_upper
            changed_axes += 1

        if changed_axes < int(cfg["minimum_bilateral_axes"]):
            return original.copy(), "visibility_axes"

        candidate = np.concatenate(
            (
                0.5 * (candidate_lower + candidate_upper),
                candidate_upper - candidate_lower,
            )
        ).astype(np.float32)
        return candidate, "candidate"

    def _refit_gate(
        self,
        original: np.ndarray,
        candidate: np.ndarray,
        evidence: GlobalEvidence,
        *,
        geometry_points: Optional[np.ndarray] = None,
        projection_corners: Optional[
            Tuple[np.ndarray, np.ndarray]
        ] = None,
        filter_boxes: Optional[
            Tuple[np.ndarray, np.ndarray]
        ] = None,
    ) -> Tuple[bool, str]:
        cfg = self.config["refit"]
        geometry_views = evidence.memory.geometry_unique_view_count
        if not evidence.memory.top_k_enabled:
            geometry_views += evidence.stats.absorbed_views
        if geometry_views < cfg["min_views"]:
            return False, "views"
        if evidence.memory.geometry_num_points < cfg["min_points"]:
            return False, "points"
        if not np.isfinite(candidate).all() or np.any(candidate[3:6] <= 0.0):
            return False, "invalid"
        minimum_extent = float(
            self.config["output_filter"]["minimum_extent"]
        )
        if minimum_extent > 0.0:
            filter_original, filter_candidate = (
                (original, candidate)
                if filter_boxes is None
                else filter_boxes
            )
            original_survives = bool(
                np.all(filter_original[3:6] >= minimum_extent)
            )
            candidate_survives = bool(
                np.all(filter_candidate[3:6] >= minimum_extent)
            )
            if original_survives != candidate_survives:
                return False, "extent_filter"
        diagonal = max(float(np.linalg.norm(original[3:6])), 1e-6)
        shift_ratio = float(
            np.linalg.norm(candidate[:3] - original[:3]) / diagonal
        )
        if shift_ratio > cfg["max_center_shift_ratio"]:
            return False, "center_shift"
        extent_ratio = candidate[3:6] / np.maximum(original[3:6], 1e-6)
        if np.any(extent_ratio < cfg["min_extent_ratio"]) or np.any(
            extent_ratio > cfg["max_extent_ratio"]
        ):
            return False, "extent"
        support_points = (
            evidence.memory.geometry_points
            if geometry_points is None
            else geometry_points
        )
        support = points_inside_aabb_fraction(
            support_points,
            original[:3],
            original[3:6],
        )
        if support < cfg["min_original_point_support"]:
            return False, "support"
        candidate_support = points_inside_aabb_fraction(
            support_points,
            candidate[:3],
            candidate[3:6],
        )
        if candidate_support < cfg["min_candidate_point_support"]:
            return False, "candidate_support"
        if (
            support - candidate_support
            > cfg["max_candidate_support_drop"]
        ):
            return False, "candidate_support_drop"
        selected_frames = (
            evidence.memory.selected_view_frame_ids
            if evidence.memory.top_k_enabled
            else None
        )
        if projection_corners is None:
            original_projection = self._mean_projection_iou(
                original, evidence.stats, selected_frames
            )
            candidate_projection = self._mean_projection_iou(
                candidate, evidence.stats, selected_frames
            )
        else:
            original_projection = self._mean_projection_iou_corners(
                projection_corners[0],
                evidence.stats,
                selected_frames,
            )
            candidate_projection = self._mean_projection_iou_corners(
                projection_corners[1],
                evidence.stats,
                selected_frames,
            )
        if candidate_projection < cfg["min_reprojection_iou"]:
            return False, "reprojection"
        if (
            candidate_projection - original_projection
            < cfg["min_reprojection_improvement"]
        ):
            return False, "reprojection_delta"
        return True, "accepted"

    def _robust_refit(
        self,
        original: np.ndarray,
        evidence: GlobalEvidence,
    ) -> Tuple[np.ndarray, bool, str]:
        cfg = self.config["refit"]
        if not cfg["enabled"]:
            return original.copy(), False, "disabled"
        if cfg["strategy"] == "visibility_aware":
            candidate, reason = self._visibility_aware_candidate(
                original, evidence
            )
            if reason != "candidate":
                return original.copy(), False, reason
        else:
            memory_box = evidence.memory.geometry_aabb
            if memory_box is None:
                return original.copy(), False, "points"
            memory_center, memory_dims = memory_box
            memory_dims = memory_dims + 2.0 * float(
                cfg["extent_padding"]
            )
            blend = float(cfg["blend"])
            candidate = np.concatenate(
                (
                    (1.0 - blend) * original[:3]
                    + blend * memory_center,
                    (1.0 - blend) * original[3:6]
                    + blend * memory_dims,
                )
            ).astype(np.float32)
        accepted, reason = self._refit_gate(original, candidate, evidence)
        return (
            candidate if accepted else original.copy(),
            bool(accepted),
            reason,
        )

    def _oriented_visibility_refit(
        self,
        original_corners: np.ndarray,
        evidence: GlobalEvidence,
    ) -> Tuple[np.ndarray, np.ndarray, bool, str]:
        """Refit in the upstream OBB frame and restore its orientation."""

        original_corners = np.asarray(
            original_corners, dtype=np.float32
        )
        original_world_box = corners_to_center_size(
            original_corners[None, ...]
        )[0]
        cfg = self.config["refit"]
        if not cfg["enabled"]:
            return (
                original_world_box,
                original_corners.copy(),
                False,
                "disabled",
            )
        try:
            center, dimensions, basis = _oriented_box_frame(
                original_corners
            )
        except ValueError:
            return (
                original_world_box,
                original_corners.copy(),
                False,
                "orientation",
            )

        local_original = np.concatenate(
            (np.zeros(3, dtype=np.float64), dimensions)
        ).astype(np.float32)
        local_records = []
        for record in evidence.memory.selected_view_records:
            local_camera = (
                None
                if record.camera_position is None
                else (
                    (
                        np.asarray(
                            record.camera_position,
                            dtype=np.float64,
                        )
                        - center
                    )
                    @ basis
                ).astype(np.float32)
            )
            local_records.append(
                replace(
                    record,
                    points_world=_points_to_box_local(
                        record.points_world,
                        center,
                        basis,
                    ),
                    camera_position=local_camera,
                )
            )
        local_candidate, reason = self._visibility_aware_candidate(
            local_original,
            evidence,
            selected_view_records=local_records,
        )
        if reason != "candidate":
            return (
                original_world_box,
                original_corners.copy(),
                False,
                reason,
            )

        candidate_corners = _local_box_to_world_corners(
            local_candidate,
            center,
            basis,
        )
        candidate_world_box = corners_to_center_size(
            candidate_corners[None, ...]
        )[0]
        local_geometry_points = _points_to_box_local(
            evidence.memory.geometry_points,
            center,
            basis,
        )
        accepted, reason = self._refit_gate(
            local_original,
            local_candidate,
            evidence,
            geometry_points=local_geometry_points,
            projection_corners=(
                original_corners,
                candidate_corners,
            ),
            filter_boxes=(
                original_world_box,
                candidate_world_box,
            ),
        )
        if not accepted:
            return (
                original_world_box,
                original_corners.copy(),
                False,
                reason,
            )
        return candidate_world_box, candidate_corners, True, reason

    def _run_neural_refiner(
        self,
        box: np.ndarray,
        evidence: GlobalEvidence,
        feature_mapping: Mapping[str, float],
    ) -> Tuple[np.ndarray, float, bool]:
        if self.box_refiner is None:
            return box, 0.5, False
        import torch

        point_count = int(self.config["box_refiner"]["point_count"])
        sampled = deterministic_bounded_sample(
            evidence.memory.geometry_points, point_count
        )
        valid_count = sampled.shape[0]
        if valid_count == 0:
            return box, 0.0, False
        points = np.zeros((point_count, 3), dtype=np.float32)
        mask = np.zeros(point_count, dtype=bool)
        points[:valid_count] = sampled
        mask[:valid_count] = True
        parameter = next(self.box_refiner.parameters())
        with torch.no_grad():
            output = self.box_refiner(
                torch.from_numpy(points[None]).to(parameter.device),
                torch.from_numpy(box[None].astype(np.float32)).to(
                    parameter.device
                ),
                torch.from_numpy(
                    quality_feature_vector(feature_mapping)[None]
                ).to(parameter.device),
                torch.from_numpy(mask[None]).to(parameter.device),
            )
        center_residual = (
            output["center_residual"].detach().float().cpu().numpy()
        )
        dimension_residual = (
            output["log_dimension_residual"].detach().float().cpu().numpy()
        )
        quality = float(
            output["quality"].detach().float().cpu().numpy().reshape(-1)[0]
        )
        if quality < float(
            self.config["box_refiner"]["quality_threshold"]
        ):
            return box, quality, False
        refined = apply_box_residual_numpy(
            box,
            center_residual[0],
            dimension_residual[0],
            max_center_fraction=self.box_refiner.config.max_center_fraction,
            max_abs_log_dimension_residual=(
                self.box_refiner.config.max_log_dimension_residual
            ),
            minimum_dimension=self.box_refiner.config.minimum_dimension,
        )
        accepted, _ = self._refit_gate(box, refined, evidence)
        return (refined if accepted else box), quality, bool(accepted)

    def _run_oriented_neural_refiner(
        self,
        original_corners: np.ndarray,
        evidence: GlobalEvidence,
        feature_mapping: Mapping[str, float],
    ) -> Tuple[np.ndarray, np.ndarray, float, bool, str]:
        """Run B5-v2 in the upstream OBB frame and retain its basis/yaw.

        The learned model never receives raw-world axes.  Points, box
        dimensions, and predicted residuals all live in the original box's
        local coordinate frame.  The candidate is transformed back to world
        corners only after inference, then passed through the same point
        support and real multi-view reprojection gates as the hand-written
        refit.
        """

        original_corners = np.asarray(original_corners, dtype=np.float32)
        original_world_box = corners_to_center_size(
            original_corners[None, ...]
        )[0]
        if self.box_refiner is None:
            return (
                original_world_box,
                original_corners.copy(),
                0.5,
                False,
                "neural_disabled",
            )
        self.stats["neural_refits_attempted"] += 1
        try:
            refiner_inputs = _prepare_oriented_box_refiner_inputs(
                original_corners,
                evidence.memory.geometry_points,
                int(self.config["box_refiner"]["point_count"]),
            )
        except ValueError:
            self.stats["neural_refits_invalid_orientation"] += 1
            return (
                original_world_box,
                original_corners.copy(),
                0.0,
                False,
                "neural_orientation",
            )
        if not np.any(refiner_inputs.point_mask):
            self.stats["neural_refits_gate_rejected"] += 1
            return (
                original_world_box,
                original_corners.copy(),
                0.0,
                False,
                "neural_points",
            )

        points = refiner_inputs.points_local
        mask = refiner_inputs.point_mask
        local_original = refiner_inputs.local_box
        local_geometry_points = refiner_inputs.gate_points_local
        center = refiner_inputs.frame_center
        basis = refiner_inputs.frame_basis
        quality_features = np.array(
            quality_feature_vector(feature_mapping),
            dtype=np.float32,
            copy=True,
        )

        import torch

        parameter = next(self.box_refiner.parameters())
        with torch.no_grad():
            output = self.box_refiner(
                torch.from_numpy(points[None]).to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
                torch.from_numpy(local_original[None]).to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
                torch.from_numpy(
                    quality_features[None]
                ).to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
                torch.from_numpy(mask[None]).to(parameter.device),
            )
        center_residual = (
            output["center_residual"].detach().float().cpu().numpy()[0]
        )
        dimension_residual = (
            output["log_dimension_residual"]
            .detach()
            .float()
            .cpu()
            .numpy()[0]
        )
        quality = float(
            output["quality"].detach().float().cpu().numpy().reshape(-1)[0]
        )
        if quality < float(self.config["box_refiner"]["min_quality"]):
            self.stats["neural_refits_quality_rejected"] += 1
            return (
                original_world_box,
                original_corners.copy(),
                quality,
                False,
                "neural_quality",
            )

        local_candidate = apply_local_box_residual_numpy(
            local_original,
            center_residual,
            dimension_residual,
            max_center_fraction=(
                self.box_refiner.config.max_center_fraction
            ),
            max_abs_log_dimension_residual=(
                self.box_refiner.config.max_log_dimension_residual
            ),
            minimum_dimension=self.box_refiner.config.minimum_dimension,
        )
        candidate_corners = _local_box_to_world_corners(
            local_candidate,
            center,
            basis,
        )
        candidate_world_box = corners_to_center_size(
            candidate_corners[None, ...]
        )[0]
        accepted, gate_reason = self._refit_gate(
            local_original,
            local_candidate,
            evidence,
            geometry_points=local_geometry_points,
            projection_corners=(
                original_corners,
                candidate_corners,
            ),
            filter_boxes=(
                original_world_box,
                candidate_world_box,
            ),
        )
        if not accepted:
            self.stats["neural_refits_gate_rejected"] += 1
            return (
                original_world_box,
                original_corners.copy(),
                quality,
                False,
                f"neural_{gate_reason}",
            )
        return (
            candidate_world_box,
            candidate_corners,
            quality,
            True,
            "neural_accepted",
        )

    def _run_joint_local_head_batch(
        self,
        *,
        corners: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        stable_ids: np.ndarray,
    ) -> Dict[int, _JointPrediction]:
        """Collect valid rows, then invoke the joint model exactly once.

        Invalid and unobserved rows are recorded for diagnostics but are not
        sent to the model.  Their caller-visible fallback remains the exact
        BoxFusion geometry and detector score.
        """

        self._last_joint_runtime.clear()
        cfg = self.config["joint_local_head"]
        if not cfg["enabled"]:
            return {}
        if self.joint_local_head is None:
            raise RuntimeError(
                "enabled joint_local_head has no strictly loaded model"
            )
        prepared_rows: List[_JointPreparedInstance] = []
        self.stats["joint_instances"] += int(len(boxes))
        started = time.perf_counter()
        minimum_views = int(self.config["refit"]["min_views"])
        minimum_points = int(self.config["refit"]["min_points"])
        for index, (original, original_corners, detector_score, stable_id) in (
            enumerate(zip(boxes, corners, scores, stable_ids))
        ):
            stable_id = int(stable_id)
            evidence = self.global_tracks.get(stable_id)
            if (
                evidence is None
                or evidence.memory.observation_count <= 0
            ):
                self.stats["joint_unobserved_identity"] += 1
                self._last_joint_runtime[stable_id] = {
                    "input_valid": False,
                    "reason": "joint_unobserved",
                }
                continue
            geometry_views = evidence.memory.geometry_unique_view_count
            if not evidence.memory.top_k_enabled:
                geometry_views += evidence.stats.absorbed_views
            if geometry_views < minimum_views:
                self.stats["joint_invalid_identity"] += 1
                self.stats["joint_rejected"]["views"] += 1
                self._last_joint_runtime[stable_id] = {
                    "input_valid": False,
                    "reason": "joint_views",
                }
                continue
            if evidence.memory.geometry_num_points < minimum_points:
                self.stats["joint_invalid_identity"] += 1
                self.stats["joint_rejected"]["points"] += 1
                self._last_joint_runtime[stable_id] = {
                    "input_valid": False,
                    "reason": "joint_points",
                }
                continue
            try:
                center, dimensions, basis = _oriented_box_frame(
                    original_corners
                )
                view_inputs = prepare_joint_view_inputs(
                    evidence.memory.selected_view_records,
                    frame_center=center,
                    frame_basis=basis,
                    max_views=int(cfg["max_views"]),
                    points_per_view=int(cfg["points_per_view"]),
                )
            except ValueError:
                self.stats["joint_invalid_identity"] += 1
                self.stats["joint_rejected"]["orientation"] += 1
                self._last_joint_runtime[stable_id] = {
                    "input_valid": False,
                    "reason": "joint_orientation",
                }
                continue
            if int(np.count_nonzero(view_inputs.view_mask)) < minimum_views:
                self.stats["joint_invalid_identity"] += 1
                self.stats["joint_rejected"]["valid_views"] += 1
                self._last_joint_runtime[stable_id] = {
                    "input_valid": False,
                    "reason": "joint_valid_views",
                }
                continue
            local_box = np.concatenate(
                (
                    np.zeros(3, dtype=np.float32),
                    dimensions.astype(np.float32),
                )
            )
            gate_points_local = _points_to_box_local(
                evidence.memory.geometry_points,
                center,
                basis,
            )
            quality_mapping = self._quality_mapping(
                original_box=original,
                final_box=original,
                detector_score=float(detector_score),
                memory=evidence.memory,
                stats=evidence.stats,
                supplemental=False,
                refiner_quality=0.5,
            )
            quality_features = np.array(
                quality_feature_vector(quality_mapping),
                dtype=np.float32,
                copy=True,
            )
            prepared = _JointPreparedInstance(
                index=int(index),
                stable_id=stable_id,
                evidence=evidence,
                original_box=np.asarray(
                    original, dtype=np.float32
                ).copy(),
                original_corners=np.asarray(
                    original_corners, dtype=np.float32
                ).copy(),
                detector_score=float(detector_score),
                quality_mapping=quality_mapping,
                quality_features=quality_features,
                frame_center=center.copy(),
                frame_basis=basis.copy(),
                local_box=local_box,
                gate_points_local=gate_points_local,
                points_local=view_inputs.points_local.copy(),
                point_mask=view_inputs.point_mask.copy(),
                view_features=view_inputs.view_features.copy(),
                view_mask=view_inputs.view_mask.copy(),
            )
            prepared_rows.append(prepared)
            self._last_joint_runtime[stable_id] = {
                "input_valid": True,
                "reason": "joint_pending",
                "prepared": prepared,
            }
        self.stats["joint_prepare_seconds"] += (
            time.perf_counter() - started
        )
        self.stats["joint_inputs_valid"] += len(prepared_rows)
        if not prepared_rows:
            return {}

        points_local = np.stack(
            [row.points_local for row in prepared_rows]
        ).astype(np.float32)
        point_mask = np.stack(
            [row.point_mask for row in prepared_rows]
        ).astype(bool)
        view_features = np.stack(
            [row.view_features for row in prepared_rows]
        ).astype(np.float32)
        view_mask = np.stack(
            [row.view_mask for row in prepared_rows]
        ).astype(bool)
        local_boxes = np.stack(
            [row.local_box for row in prepared_rows]
        ).astype(np.float32)
        quality_features = np.stack(
            [row.quality_features for row in prepared_rows]
        ).astype(np.float32)

        import torch

        try:
            parameter = next(self.joint_local_head.parameters())
        except StopIteration as error:
            raise RuntimeError(
                "joint_local_head must expose at least one parameter"
            ) from error
        forward_started = time.perf_counter()
        with torch.no_grad():
            output = self.joint_local_head(
                torch.from_numpy(points_local).to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
                torch.from_numpy(point_mask).to(parameter.device),
                torch.from_numpy(view_features).to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
                torch.from_numpy(view_mask).to(parameter.device),
                torch.from_numpy(local_boxes).to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
                torch.from_numpy(quality_features).to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                ),
            )
        self.stats["joint_forward_seconds"] += (
            time.perf_counter() - forward_started
        )
        self.stats["joint_batches"] += 1
        self.stats["joint_forward_boxes"] += len(prepared_rows)
        required_outputs = {
            "center_residual",
            "center_residual_fraction",
            "log_dimension_residual",
            "improvement_probability",
            "quality_components",
            "ranking_scores",
            "quality_log_variance",
            "quality_uncertainty",
            "view_attention",
        }
        if not isinstance(output, Mapping) or set(output) != required_outputs:
            raise RuntimeError(
                "joint_local_head output schema does not exactly match "
                f"{JOINT_LOCAL_HEAD_OUTPUT_SCHEMA}"
            )

        def numpy_output(name: str) -> np.ndarray:
            value = output[name]
            if not torch.is_tensor(value):
                raise RuntimeError(
                    f"joint_local_head output {name} must be a tensor"
                )
            result = value.detach().float().cpu().numpy()
            if not np.isfinite(result).all():
                raise RuntimeError(
                    f"joint_local_head output {name} must be finite"
                )
            return result

        batch = len(prepared_rows)
        center_residual = numpy_output("center_residual")
        center_residual_fraction = numpy_output(
            "center_residual_fraction"
        )
        log_dimension_residual = numpy_output(
            "log_dimension_residual"
        )
        improvement = numpy_output("improvement_probability")
        components = numpy_output("quality_components")
        rankings = numpy_output("ranking_scores")
        log_variance = numpy_output("quality_log_variance")
        uncertainty = numpy_output("quality_uncertainty")
        attention = numpy_output("view_attention")
        expected_shapes = {
            "center_residual": (center_residual, (batch, 3)),
            "center_residual_fraction": (
                center_residual_fraction,
                (batch, 3),
            ),
            "log_dimension_residual": (
                log_dimension_residual,
                (batch, 3),
            ),
            "improvement_probability": (improvement, (batch,)),
            "quality_components": (components, (batch, 2, 4)),
            "ranking_scores": (rankings, (batch, 2)),
            "quality_log_variance": (log_variance, (batch, 2)),
            "quality_uncertainty": (uncertainty, (batch, 2)),
            "view_attention": (
                attention,
                (batch, int(cfg["max_views"])),
            ),
        }
        for name, (value, expected) in expected_shapes.items():
            if value.shape != expected:
                raise RuntimeError(
                    f"joint_local_head output {name} must have shape "
                    f"{expected}, received {value.shape}"
                )
        for name, value in (
            ("improvement_probability", improvement),
            ("quality_components", components),
            ("ranking_scores", rankings),
            ("view_attention", attention),
        ):
            if ((value < 0.0) | (value > 1.0)).any():
                raise RuntimeError(
                    f"joint_local_head output {name} must lie in [0,1]"
                )
        if (uncertainty <= 0.0).any():
            raise RuntimeError(
                "joint_local_head quality_uncertainty must be positive"
            )
        architecture = self.joint_local_head.config
        if (
            np.abs(center_residual_fraction)
            > float(architecture.max_center_fraction) + 1e-6
        ).any():
            raise RuntimeError(
                "joint center residual fraction exceeds architecture bound"
            )
        if (
            np.abs(log_dimension_residual)
            > float(architecture.max_log_dimension_residual) + 1e-6
        ).any():
            raise RuntimeError(
                "joint log-dimension residual exceeds architecture bound"
            )
        expected_center_residual = (
            center_residual_fraction * local_boxes[:, 3:6]
        )
        if not np.allclose(
            center_residual,
            expected_center_residual,
            atol=1e-6,
            rtol=1e-5,
        ):
            raise RuntimeError(
                "joint center residual is inconsistent with its fraction"
            )
        if (
            (components[:, :, 2] > components[:, :, 1] + 1e-6).any()
            or (
                components[:, :, 3]
                > components[:, :, 2] + 1e-6
            ).any()
        ):
            raise RuntimeError(
                "joint IoU exceedance probabilities must be monotonic"
            )
        if not np.allclose(
            attention.sum(axis=1), 1.0, atol=1e-5, rtol=0.0
        ):
            raise RuntimeError(
                "joint view attention must sum to one per instance"
            )

        predictions: Dict[int, _JointPrediction] = {}
        for row_index, prepared in enumerate(prepared_rows):
            prediction = _JointPrediction(
                prepared=prepared,
                center_residual=center_residual[row_index].copy(),
                center_residual_fraction=(
                    center_residual_fraction[row_index].copy()
                ),
                log_dimension_residual=(
                    log_dimension_residual[row_index].copy()
                ),
                improvement_probability=float(
                    improvement[row_index]
                ),
                quality_components=components[row_index].copy(),
                ranking_scores=rankings[row_index].copy(),
                quality_log_variance=log_variance[row_index].copy(),
                quality_uncertainty=uncertainty[row_index].copy(),
                view_attention=attention[row_index].copy(),
            )
            predictions[prepared.index] = prediction
            self._last_joint_runtime[prepared.stable_id].update(
                {
                    "reason": "joint_inferred",
                    "prediction": prediction,
                }
            )
        return predictions

    def _apply_joint_prediction(
        self,
        prediction: _JointPrediction,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        float,
        Mapping[str, float],
        bool,
        str,
    ]:
        """Apply one inferred candidate and select its matching score branch."""

        cfg = self.config["joint_local_head"]
        prepared = prediction.prepared
        refined = prepared.original_box.copy()
        corners = prepared.original_corners.copy()
        accepted = False
        reason = "joint_geometry_disabled"
        branch_index = 0
        if cfg["mutate_geometry"]:
            if (
                prediction.improvement_probability
                < float(cfg["improvement_threshold"])
            ):
                reason = "joint_improvement"
                self.stats["joint_improvement_rejected"] += 1
                self.stats["joint_rejected"]["improvement"] += 1
            elif (
                float(prediction.quality_uncertainty[1])
                > float(cfg["max_candidate_uncertainty"])
            ):
                reason = "joint_uncertainty"
                self.stats["joint_uncertainty_rejected"] += 1
                self.stats["joint_rejected"]["uncertainty"] += 1
            else:
                local_candidate = apply_local_box_residual_numpy(
                    prepared.local_box,
                    prediction.center_residual,
                    prediction.log_dimension_residual,
                    max_center_fraction=(
                        self.joint_local_head.config.max_center_fraction
                    ),
                    max_abs_log_dimension_residual=(
                        self.joint_local_head.config
                        .max_log_dimension_residual
                    ),
                    minimum_dimension=(
                        self.joint_local_head.config.minimum_dimension
                    ),
                )
                candidate_corners = _local_box_to_world_corners(
                    local_candidate,
                    prepared.frame_center,
                    prepared.frame_basis,
                )
                candidate_world_box = corners_to_center_size(
                    candidate_corners[None, ...]
                )[0]
                gate_started = time.perf_counter()
                accepted, gate_reason = self._refit_gate(
                    prepared.local_box,
                    local_candidate,
                    prepared.evidence,
                    geometry_points=prepared.gate_points_local,
                    projection_corners=(
                        prepared.original_corners,
                        candidate_corners,
                    ),
                    filter_boxes=(
                        prepared.original_box,
                        candidate_world_box,
                    ),
                )
                self.stats["joint_gate_seconds"] += (
                    time.perf_counter() - gate_started
                )
                if accepted:
                    refined = candidate_world_box
                    corners = candidate_corners
                    branch_index = 1
                    reason = "joint_accepted"
                    self.stats["joint_accepted"] += 1
                else:
                    reason = f"joint_{gate_reason}"
                    self.stats["joint_gate_rejected"] += 1
                    self.stats["joint_rejected"][gate_reason] += 1

        if cfg["mutate_scores"]:
            quality_score = float(
                prediction.ranking_scores[branch_index]
            )
            blend = float(cfg["detector_blend"])
            score = (
                blend * prepared.detector_score
                + (1.0 - blend) * quality_score
            )
            if cfg["preserve_original_floor"]:
                score = max(score, prepared.detector_score)
            score = float(np.clip(score, 0.0, 1.0))
            if branch_index == 1:
                self.stats["joint_candidate_quality_branch"] += 1
            else:
                self.stats["joint_original_quality_branch"] += 1
        else:
            score = prepared.detector_score

        runtime = self._last_joint_runtime[prepared.stable_id]
        runtime.update(
            {
                "reason": reason,
                "accepted": bool(accepted),
                "quality_branch": (
                    "candidate" if branch_index == 1 else "original"
                ),
                "selected_ranking_score": float(
                    prediction.ranking_scores[branch_index]
                ),
                "final_score": score,
            }
        )
        return (
            refined,
            corners,
            score,
            prepared.quality_mapping,
            bool(accepted),
            reason,
        )

    def _score(
        self,
        detector_score: float,
        mapping: Mapping[str, float],
        *,
        observed: bool,
    ) -> float:
        cfg = self.config["quality"]
        if (
            not cfg["enabled"]
            or self.quality_scorer is None
            or (not observed and not cfg["apply_to_unobserved"])
        ):
            return float(detector_score)
        quality_score = float(self.quality_scorer(mapping))
        blend = float(cfg["blend_with_detector"])
        score = blend * float(detector_score) + (1.0 - blend) * quality_score
        if cfg["preserve_original_floor"]:
            score = max(score, float(detector_score))
        return float(np.clip(score, 0.0, 1.0))

    def _c2_projection_metrics(
        self,
        original: np.ndarray,
        candidate: np.ndarray,
        memory: ObjectGeometryMemory,
        stats: EvidenceStats,
    ) -> Tuple[float, float, int]:
        """Return weighted old/new projection and supporting-view count."""

        selected_frames = {
            int(frame_id)
            for frame_id in memory.selected_view_frame_ids
        }
        records = [
            view
            for view in stats.view_records
            if not selected_frames
            or int(view.frame_index) in selected_frames
        ]
        if not records:
            return 0.0, 0.0, 0
        original_values = np.asarray(
            [
                self._projection_iou_for_view(original, view)
                for view in records
            ],
            dtype=np.float64,
        )
        candidate_values = np.asarray(
            [
                self._projection_iou_for_view(candidate, view)
                for view in records
            ],
            dtype=np.float64,
        )
        weights = np.asarray(
            [max(float(view.score), 1e-4) for view in records],
            dtype=np.float64,
        )
        minimum_view_iou = float(
            self.config["supplemental_geometry_refiner"][
                "minimum_projection_view_iou"
            ]
        )
        return (
            float(np.average(original_values, weights=weights)),
            float(np.average(candidate_values, weights=weights)),
            int(np.count_nonzero(candidate_values >= minimum_view_iou)),
        )

    @staticmethod
    def _c2_point_support_views(
        candidate: np.ndarray,
        memory: ObjectGeometryMemory,
        minimum_support: float,
    ) -> int:
        count = 0
        for record in memory.selected_view_records:
            support = points_inside_aabb_fraction(
                record.points_world,
                candidate[:3],
                candidate[3:6],
            )
            count += int(support >= minimum_support)
        return count

    def _c2_structural_rejection(
        self,
        candidate: np.ndarray,
        *,
        label: Optional[str],
        branch: str,
        global_boxes: np.ndarray,
        other_outputs: Sequence[_SupplementalOutput],
    ) -> Optional[str]:
        """Reject a C2 box that would cross a frozen C1 output boundary."""

        supplemental_cfg = self.config["supplemental_output"]
        geometry_cfg = self.config["supplemental_geometry_refiner"]
        normalized_label = (
            "" if label is None else str(label).strip().casefold()
        )
        if branch == "planar":
            dimensions = np.sort(
                np.asarray(candidate[3:6], dtype=np.float64)
            )
            if (
                dimensions[0]
                < float(geometry_cfg["refined_planar_minimum_extent"])
                or dimensions[1]
                < float(supplemental_cfg["planar_middle_extent"])
                or dimensions[2]
                < float(supplemental_cfg["planar_max_extent"])
            ):
                return "structural_extent"
        else:
            output_cfg = self.config["output_filter"]
            configured_extent = supplemental_cfg["minimum_extent"]
            final_extent = output_cfg["final_minimum_extent"]
            default_extent = (
                float(final_extent)
                if supplemental_cfg["class_aware_extent"]
                and final_extent is not None
                else float(
                    output_cfg["minimum_extent"]
                    if configured_extent is None
                    else configured_extent
                )
            )
            if not supplemental_extent_is_valid(
                candidate[3:6],
                label,
                supplemental_cfg,
                default_minimum_extent=default_extent,
            ):
                return "structural_extent"

        for global_box in np.asarray(global_boxes, dtype=np.float32):
            overlap = aabb_iou(
                candidate[:3],
                candidate[3:6],
                global_box[:3],
                global_box[3:6],
            )
            if overlap >= supplemental_cfg["drop_if_global_iou"]:
                return "structural_global"
            if supplemental_cfg["bev_duplicate_enabled"]:
                bev_iou, containment = bev_iou_and_containment(
                    candidate, global_box
                )
                z_containment = _axis_overlap_containment(
                    candidate, global_box, axis=2
                )
                if (
                    bev_iou
                    >= supplemental_cfg["bev_duplicate_iou"]
                    and containment
                    >= supplemental_cfg["bev_duplicate_containment"]
                    and z_containment
                    >= supplemental_cfg["bev_duplicate_min_z_containment"]
                ):
                    return "structural_bev_global"

        for other in other_outputs:
            overlap = aabb_iou(
                candidate[:3],
                candidate[3:6],
                other.box[:3],
                other.box[3:6],
            )
            if overlap >= supplemental_cfg["drop_if_supplemental_iou"]:
                return "structural_supplemental"
            other_label = (
                ""
                if other.label is None
                else str(other.label).strip().casefold()
            )
            if (
                branch == "planar"
                and supplemental_cfg["planar_duplicate_enabled"]
                and normalized_label == other_label
                and normalized_label
                in set(supplemental_cfg["planar_extent_labels"])
            ):
                bev_iou, containment = bev_iou_and_containment(
                    candidate, other.box
                )
                z_containment = _axis_overlap_containment(
                    candidate, other.box, axis=2
                )
                if (
                    bev_iou
                    >= supplemental_cfg["planar_duplicate_bev_iou"]
                    and containment
                    >= supplemental_cfg["planar_duplicate_containment"]
                    and z_containment
                    >= supplemental_cfg[
                        "planar_duplicate_min_z_containment"
                    ]
                ):
                    return "structural_planar_supplemental"
        return None

    def _apply_c2_geometry_refinement(
        self,
        c1_outputs: Sequence[_SupplementalOutput],
        global_boxes: np.ndarray,
    ) -> List[_SupplementalOutput]:
        """Safely refine retained C1 geometry without changing its row set."""

        self._last_c2_runtime.clear()
        for key in (
            "c2_attempted",
            "c2_proposed",
            "c2_verified",
            "c2_applied",
        ):
            self.stats[key] = 0
        self.stats["c2_seconds"] = 0.0
        self.stats["c2_rejected"] = Counter()
        cfg = self.config["supplemental_geometry_refiner"]
        if not cfg["enabled"] or not c1_outputs:
            return list(c1_outputs)

        started = time.perf_counter()
        small_labels = set(cfg["small_labels"])
        planar_labels = set(cfg["planar_labels"])
        refined_outputs: List[_SupplementalOutput] = []
        for output_index, item in enumerate(c1_outputs):
            original = np.asarray(item.box, dtype=np.float32)
            normalized_label = (
                ""
                if item.label is None
                else str(item.label).strip().casefold()
            )
            runtime: Dict[str, Any] = {
                "attempted": False,
                "proposed": False,
                "verified": False,
                "applied": False,
                "reason": "label_not_supported",
                "branch": "identity",
                "original_box": original.copy(),
                "candidate_box": original.copy(),
                "component_fraction": np.nan,
                "component_density": np.nan,
                "density_ratio": np.nan,
                "point_count": int(item.memory.geometry_num_points),
                "view_count": int(item.memory.selected_view_count),
                "original_support": np.nan,
                "candidate_support": np.nan,
                "original_projection": np.nan,
                "candidate_projection": np.nan,
                "projection_delta": np.nan,
                "projection_views": 0,
                "point_support_views": 0,
                "center_shift_ratio": np.nan,
                "extent_ratios": np.full(3, np.nan, dtype=np.float32),
            }
            self._last_c2_runtime[int(item.stable_id)] = runtime

            expected_branch: Optional[str] = None
            if normalized_label in small_labels:
                expected_branch = "solid"
            elif normalized_label in planar_labels:
                expected_branch = "planar"
            if expected_branch is None:
                refined_outputs.append(item)
                continue

            runtime["attempted"] = True
            self.stats["c2_attempted"] += 1

            geometry_points = item.memory.geometry_points
            full_memory_points = item.memory.points
            selected_views = int(item.memory.selected_view_count)
            if selected_views < int(cfg["minimum_views"]):
                reason = "views"
                runtime["reason"] = reason
                self.stats["c2_rejected"][reason] += 1
                refined_outputs.append(item)
                continue
            if len(geometry_points) < int(cfg["minimum_points"]):
                reason = "points"
                runtime["reason"] = reason
                self.stats["c2_rejected"][reason] += 1
                refined_outputs.append(item)
                continue

            try:
                proposal: DepthOccupancyProposal = (
                    propose_depth_occupancy_refinement(
                        original,
                        geometry_points,
                        selected_views,
                        full_memory_points=full_memory_points,
                        branch_hint=expected_branch,
                        config=cfg["proposal"],
                    )
                )
            except (TypeError, ValueError, FloatingPointError):
                reason = "proposal_error"
                runtime["reason"] = reason
                self.stats["c2_rejected"][reason] += 1
                refined_outputs.append(item)
                continue

            candidate = np.asarray(
                proposal.candidate, dtype=np.float32
            ).copy()
            runtime.update(
                {
                    "proposed": bool(proposal.proposed),
                    "reason": str(proposal.reason),
                    "branch": str(proposal.branch),
                    "candidate_box": candidate.copy(),
                    "component_fraction": float(
                        proposal.component_fraction
                    ),
                    "component_density": float(
                        proposal.component_density
                    ),
                    "density_ratio": float(proposal.density_ratio),
                }
            )
            if not proposal.proposed:
                reason = str(proposal.reason)
                self.stats["c2_rejected"][reason] += 1
                refined_outputs.append(item)
                continue
            self.stats["c2_proposed"] += 1

            def reject(reason: str) -> None:
                runtime["reason"] = reason
                self.stats["c2_rejected"][reason] += 1

            if proposal.branch != expected_branch:
                reject("label_branch")
                refined_outputs.append(item)
                continue
            if (
                candidate.shape != (6,)
                or not np.isfinite(candidate).all()
                or np.any(candidate[3:6] <= 0.0)
            ):
                reject("invalid")
                refined_outputs.append(item)
                continue
            if np.array_equal(candidate, original):
                reject("unchanged")
                refined_outputs.append(item)
                continue

            diagonal = max(
                float(np.linalg.norm(original[3:6])), 1e-6
            )
            shift_ratio = float(
                np.linalg.norm(candidate[:3] - original[:3])
                / diagonal
            )
            extent_ratios = (
                candidate[3:6]
                / np.maximum(original[3:6], 1e-6)
            ).astype(np.float32)
            runtime["center_shift_ratio"] = shift_ratio
            runtime["extent_ratios"] = extent_ratios.copy()
            if shift_ratio > float(
                cfg["maximum_center_shift_ratio"]
            ):
                reject("center_shift")
                refined_outputs.append(item)
                continue
            if np.any(
                extent_ratios < float(cfg["minimum_extent_ratio"])
            ) or np.any(
                extent_ratios > float(cfg["maximum_extent_ratio"])
            ):
                reject("extent_ratio")
                refined_outputs.append(item)
                continue

            original_support = points_inside_aabb_fraction(
                geometry_points, original[:3], original[3:6]
            )
            candidate_support = points_inside_aabb_fraction(
                geometry_points, candidate[:3], candidate[3:6]
            )
            (
                original_projection,
                candidate_projection,
                projection_views,
            ) = self._c2_projection_metrics(
                original, candidate, item.memory, item.stats
            )
            projection_delta = (
                candidate_projection - original_projection
            )
            point_support_views = self._c2_point_support_views(
                candidate,
                item.memory,
                float(cfg["planar_minimum_view_point_support"]),
            )
            runtime.update(
                {
                    "original_support": float(original_support),
                    "candidate_support": float(candidate_support),
                    "original_projection": float(
                        original_projection
                    ),
                    "candidate_projection": float(
                        candidate_projection
                    ),
                    "projection_delta": float(projection_delta),
                    "projection_views": int(projection_views),
                    "point_support_views": int(point_support_views),
                }
            )
            if candidate_projection < float(
                cfg["minimum_absolute_projection_iou"]
            ):
                reject("projection")
                refined_outputs.append(item)
                continue
            if projection_views < int(cfg["minimum_projection_views"]):
                reject("projection_views")
                refined_outputs.append(item)
                continue

            if proposal.branch == "solid":
                if candidate_support < float(
                    cfg["small_minimum_candidate_support"]
                ):
                    reject("solid_support")
                    refined_outputs.append(item)
                    continue
                if projection_views < int(
                    cfg["small_minimum_projection_views"]
                ):
                    reject("solid_projection_views")
                    refined_outputs.append(item)
                    continue
                if projection_delta < -float(
                    cfg["small_maximum_projection_drop"]
                ):
                    reject("solid_projection_drop")
                    refined_outputs.append(item)
                    continue
            else:
                if proposal.component_fraction < float(
                    cfg["planar_minimum_component_fraction"]
                ):
                    reject("planar_component_fraction")
                    refined_outputs.append(item)
                    continue
                if proposal.density_ratio < float(
                    cfg["planar_minimum_density_ratio"]
                ):
                    reject("planar_density_ratio")
                    refined_outputs.append(item)
                    continue
                if candidate_support < float(
                    cfg["planar_minimum_candidate_support"]
                ):
                    reject("planar_support")
                    refined_outputs.append(item)
                    continue
                if point_support_views < int(
                    cfg["planar_minimum_point_support_views"]
                ):
                    reject("planar_point_support_views")
                    refined_outputs.append(item)
                    continue
                if projection_delta < -float(
                    cfg["planar_maximum_projection_drop"]
                ):
                    reject("planar_projection_drop")
                    refined_outputs.append(item)
                    continue

            structural_reason = self._c2_structural_rejection(
                candidate,
                label=item.label,
                branch=proposal.branch,
                global_boxes=global_boxes,
                # Earlier rows use their accepted C2 geometry while later
                # rows retain their frozen C1 geometry.  This deterministic
                # prefix rule prevents two individually valid refinements
                # from converging into a duplicate pair.
                other_outputs=(
                    *refined_outputs,
                    *c1_outputs[output_index + 1 :],
                ),
            )
            if structural_reason is not None:
                reject(structural_reason)
                refined_outputs.append(item)
                continue

            runtime["verified"] = True
            self.stats["c2_verified"] += 1
            if not cfg["mutate"]:
                runtime["reason"] = "verified_observer"
                refined_outputs.append(item)
                continue

            runtime["applied"] = True
            runtime["reason"] = "accepted"
            self.stats["c2_applied"] += 1
            refined_outputs.append(
                replace(
                    item,
                    box=candidate,
                    corners=aabb_corners(
                        candidate[:3], candidate[3:6]
                    ),
                    refit_applied=True,
                    refit_reason=f"c2_{proposal.branch}_accepted",
                )
            )

        self.stats["c2_seconds"] += time.perf_counter() - started
        return refined_outputs

    def _c4_projection_metrics(
        self,
        original_corners: np.ndarray,
        candidate_corners: np.ndarray,
        evidence: GlobalEvidence,
    ) -> Tuple[float, float, int]:
        """Return weighted OBB projection agreement for SAM3 views."""

        selected_frames = {
            int(frame_id)
            for frame_id in evidence.memory.selected_view_frame_ids
        }
        records = [
            view
            for view in evidence.stats.view_records
            if not selected_frames
            or int(view.frame_index) in selected_frames
        ]
        if not records:
            return 0.0, 0.0, 0
        original_values = np.asarray(
            [
                self._projection_iou_for_corners(
                    original_corners, view
                )
                for view in records
            ],
            dtype=np.float64,
        )
        candidate_values = np.asarray(
            [
                self._projection_iou_for_corners(
                    candidate_corners, view
                )
                for view in records
            ],
            dtype=np.float64,
        )
        weights = np.asarray(
            [max(float(view.score), 1e-4) for view in records],
            dtype=np.float64,
        )
        minimum_view_iou = float(
            self.config["generic_local_geometry_refiner"][
                "minimum_projection_view_iou"
            ]
        )
        return (
            float(np.average(original_values, weights=weights)),
            float(np.average(candidate_values, weights=weights)),
            int(np.count_nonzero(candidate_values >= minimum_view_iou)),
        )

    @staticmethod
    def _normalized_c4_label(label: Optional[str]) -> str:
        value = "" if label is None else str(label).strip().casefold()
        aliases = {
            "bookcase": "bookshelf",
            "couch": "sofa",
            "cupboard": "cabinet",
            "wardrobe": "cabinet",
            "dining table": "table",
            "dining_table": "table",
            "other furniture": "otherfurniture",
        }
        return aliases.get(value, value)

    def _observe_generic_geometry_candidates(
        self,
        *,
        boxes: np.ndarray,
        corners: np.ndarray,
        source_indices: np.ndarray,
        stable_ids: np.ndarray,
    ) -> None:
        """Build verified C4 candidates without mutating any B6 output."""

        self._last_c4_runtime.clear()
        for key in (
            "c4_attempted",
            "c4_proposed",
            "c4_verified",
            "c4_applied",
        ):
            self.stats[key] = 0
        self.stats["c4_refiner_seconds"] = 0.0
        self.stats["c4_rejected"] = Counter()
        cfg = self.config["generic_local_geometry_refiner"]
        if not cfg["enabled"] or self._c4_failed:
            return

        started = time.perf_counter()
        targets = set(cfg["target_labels"])
        all_boxes = np.asarray(boxes, dtype=np.float32)
        all_corners = np.asarray(corners, dtype=np.float32)
        if all_corners.shape != (len(all_boxes), 8, 3):
            raise ValueError(
                "C4 final corners must align with final boxes"
            )
        for row_index, (
            original,
            original_corners,
            source_index,
            stable_id,
        ) in enumerate(
            zip(all_boxes, all_corners, source_indices, stable_ids)
        ):
            stable_key = int(stable_id)
            evidence = self.generic_geometry_global_tracks.get(stable_key)
            raw_label = (
                None if evidence is None else evidence.stats.label
            )
            normalized_label = self._normalized_c4_label(raw_label)
            runtime: Dict[str, Any] = {
                "attempted": False,
                "proposed": False,
                "verified": False,
                "applied": False,
                "reason": "unobserved",
                "source": (
                    "global" if int(source_index) >= 0 else "supplemental"
                ),
                "label": "" if raw_label is None else str(raw_label),
                "normalized_label": normalized_label,
                "original_box": np.asarray(
                    original, dtype=np.float32
                ).copy(),
                "candidate_box": np.asarray(
                    original, dtype=np.float32
                ).copy(),
                "original_corners": np.asarray(
                    original_corners, dtype=np.float32
                ).copy(),
                "candidate_corners": np.asarray(
                    original_corners, dtype=np.float32
                ).copy(),
                "local_original_box": np.full(
                    6, np.nan, dtype=np.float32
                ),
                "local_candidate_box": np.full(
                    6, np.nan, dtype=np.float32
                ),
                "local_basis": np.full(
                    (3, 3), np.nan, dtype=np.float32
                ),
                "local_frame_valid": False,
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
                "component_inside_fraction": np.nan,
                "mean_valid_depth_ratio": np.nan,
                "clean_original_support": np.nan,
                "clean_candidate_support": np.nan,
                "clean_support_drop": np.nan,
                "raw_original_support": np.nan,
                "raw_candidate_support": np.nan,
                "raw_support_drop": np.nan,
                "original_projection": np.nan,
                "candidate_projection": np.nan,
                "projection_delta": np.nan,
                "projection_views": 0,
                "center_shift_ratio": np.nan,
                "center_shift_ratios": np.full(
                    3, np.nan, dtype=np.float32
                ),
                "extent_ratios": np.full(
                    3, np.nan, dtype=np.float32
                ),
                "original_candidate_iou": np.nan,
                "original_max_other_iou": np.nan,
                "candidate_max_other_iou": np.nan,
                "overlap_increase": np.nan,
                "boundary_values": np.full(
                    (3, 2), np.nan, dtype=np.float32
                ),
                "boundary_view_counts": np.zeros(
                    (3, 2), dtype=np.int64
                ),
                "boundary_spreads": np.full(
                    (3, 2), np.nan, dtype=np.float32
                ),
                "boundary_visible": np.zeros(
                    (3, 2), dtype=bool
                ),
            }
            self._last_c4_runtime[stable_key] = runtime

            def reject(reason: str) -> None:
                runtime["reason"] = reason
                self.stats["c4_rejected"][reason] += 1

            if int(source_index) < 0:
                reject("scope")
                continue
            if evidence is None:
                reject("unobserved")
                continue
            if (
                normalized_label not in targets
                and not cfg["allow_unknown_label"]
            ):
                reject("label_scope")
                continue

            records = evidence.memory.selected_view_records
            valid_depth_values = [
                float(record.valid_depth_ratio) for record in records
            ]
            mean_valid_depth = (
                float(np.mean(valid_depth_values))
                if valid_depth_values
                else 0.0
            )
            runtime["mean_valid_depth_ratio"] = mean_valid_depth
            runtime["attempted"] = True
            self.stats["c4_attempted"] += 1
            if mean_valid_depth < float(
                cfg["minimum_mean_valid_depth_ratio"]
            ):
                reject("valid_depth")
                continue

            try:
                (
                    frame_center,
                    frame_dimensions,
                    frame_basis,
                ) = _oriented_box_frame(original_corners)
                local_original = np.concatenate(
                    (
                        np.zeros(3, dtype=np.float32),
                        frame_dimensions.astype(np.float32),
                    )
                )
                local_records = tuple(
                    MemoryViewRecord(
                        frame_id=record.frame_id,
                        points_world=_points_to_box_local(
                            record.points_world,
                            frame_center,
                            frame_basis,
                        ),
                        quality=record.quality,
                        confidence=record.confidence,
                        valid_depth_ratio=record.valid_depth_ratio,
                        projection_mask_iou=(
                            record.projection_mask_iou
                        ),
                        camera_position=(
                            None
                            if record.camera_position is None
                            else _points_to_box_local(
                                np.asarray(
                                    record.camera_position,
                                    dtype=np.float32,
                                )[None, :],
                                frame_center,
                                frame_basis,
                            )[0]
                        ),
                    )
                    for record in records
                )
                runtime.update(
                    {
                        "local_original_box": local_original.copy(),
                        "local_candidate_box": local_original.copy(),
                        "local_basis": frame_basis.astype(
                            np.float32
                        ).copy(),
                        "local_frame_valid": True,
                    }
                )
            except (TypeError, ValueError, FloatingPointError) as error:
                reject(f"local_frame_error:{type(error).__name__}")
                continue

            try:
                proposal: GenericLocalGeometryProposal = (
                    propose_generic_local_geometry(
                        local_original,
                        local_records,
                        config=cfg["proposal"],
                    )
                )
            except (TypeError, ValueError, FloatingPointError) as error:
                reject(f"proposal_error:{type(error).__name__}")
                continue

            local_candidate = np.asarray(
                proposal.candidate, dtype=np.float32
            ).copy()
            try:
                candidate_corners = _local_box_to_world_corners(
                    local_candidate, frame_center, frame_basis
                )
                candidate = corners_to_center_size(
                    candidate_corners[None, ...]
                )[0]
            except (TypeError, ValueError, FloatingPointError) as error:
                reject(f"candidate_frame_error:{type(error).__name__}")
                continue
            runtime.update(
                {
                    "proposed": bool(proposal.is_candidate),
                    "reason": str(proposal.reason),
                    "candidate_box": candidate,
                    "candidate_corners": candidate_corners.copy(),
                    "local_candidate_box": local_candidate.copy(),
                    "selected_frame_ids": tuple(
                        proposal.selected_frame_ids
                    ),
                    "eligible_view_count": int(
                        proposal.eligible_view_count
                    ),
                    "selected_view_count": int(
                        proposal.selected_view_count
                    ),
                    "input_point_count": int(
                        proposal.input_point_count
                    ),
                    "cropped_point_count": int(
                        proposal.cropped_point_count
                    ),
                    "consensus_point_count": int(
                        proposal.consensus_point_count
                    ),
                    "component_count": int(
                        proposal.component_count
                    ),
                    "eligible_component_count": int(
                        proposal.eligible_component_count
                    ),
                    "anchor_point_count": int(
                        proposal.anchor_point_count
                    ),
                    "merged_component_count": int(
                        proposal.merged_component_count
                    ),
                    "component_view_count": int(
                        proposal.component_view_count
                    ),
                    "component_inside_fraction": float(
                        proposal.component_inside_fraction
                    ),
                    "clean_original_support": float(
                        proposal.original_support
                    ),
                    "clean_candidate_support": float(
                        proposal.candidate_support
                    ),
                    "clean_support_drop": float(
                        proposal.support_drop
                    ),
                    "center_shift_ratios": np.asarray(
                        proposal.center_shift_ratios,
                        dtype=np.float32,
                    ).copy(),
                    "extent_ratios": np.asarray(
                        proposal.extent_ratios, dtype=np.float32
                    ).copy(),
                    "boundary_values": np.asarray(
                        proposal.boundary_values, dtype=np.float32
                    ).copy(),
                    "boundary_view_counts": np.asarray(
                        proposal.boundary_view_counts,
                        dtype=np.int64,
                    ).copy(),
                    "boundary_spreads": np.asarray(
                        proposal.boundary_spreads,
                        dtype=np.float32,
                    ).copy(),
                    "boundary_visible": np.asarray(
                        proposal.boundary_visible, dtype=bool
                    ).copy(),
                }
            )
            if not proposal.is_candidate:
                reject(str(proposal.reason))
                continue
            self.stats["c4_proposed"] += 1

            if (
                local_candidate.shape != (6,)
                or not np.isfinite(local_candidate).all()
                or np.any(local_candidate[3:6] <= 0.0)
                or candidate.shape != (6,)
                or not np.isfinite(candidate).all()
                or np.any(candidate[3:6] <= 0.0)
                or candidate_corners.shape != (8, 3)
                or not np.isfinite(candidate_corners).all()
            ):
                reject("invalid_candidate")
                continue
            diagonal = max(
                float(np.linalg.norm(local_original[3:6])), 1e-6
            )
            center_shift_ratio = float(
                np.linalg.norm(
                    local_candidate[:3] - local_original[:3]
                )
                / diagonal
            )
            extent_ratios = local_candidate[3:6] / np.maximum(
                local_original[3:6], 1e-6
            )
            original_candidate_iou = aabb_iou(
                local_original[:3],
                local_original[3:6],
                local_candidate[:3],
                local_candidate[3:6],
            )
            runtime["center_shift_ratio"] = center_shift_ratio
            runtime["extent_ratios"] = extent_ratios.astype(
                np.float32
            )
            runtime[
                "original_candidate_iou"
            ] = original_candidate_iou
            if center_shift_ratio > float(
                cfg["maximum_center_shift_ratio"]
            ):
                reject("center_shift")
                continue
            if np.any(
                extent_ratios < float(cfg["minimum_extent_ratio"])
            ) or np.any(
                extent_ratios > float(cfg["maximum_extent_ratio"])
            ):
                reject("extent_ratio")
                continue
            if original_candidate_iou < float(
                cfg["minimum_original_candidate_iou"]
            ):
                reject("original_iou")
                continue

            final_extent = self.config["output_filter"][
                "final_minimum_extent"
            ]
            if (
                final_extent is not None
                and np.any(
                    local_candidate[3:6] < float(final_extent)
                )
            ):
                reject("extent_survival")
                continue

            raw_points = _points_to_box_local(
                evidence.memory.geometry_points,
                frame_center,
                frame_basis,
            )
            raw_original_support = points_inside_aabb_fraction(
                raw_points,
                local_original[:3],
                local_original[3:6],
            )
            raw_candidate_support = points_inside_aabb_fraction(
                raw_points,
                local_candidate[:3],
                local_candidate[3:6],
            )
            raw_support_drop = max(
                0.0, raw_original_support - raw_candidate_support
            )
            runtime.update(
                {
                    "raw_original_support": raw_original_support,
                    "raw_candidate_support": raw_candidate_support,
                    "raw_support_drop": raw_support_drop,
                }
            )
            if raw_candidate_support < float(
                cfg["minimum_raw_candidate_support"]
            ):
                reject("raw_support")
                continue
            if raw_support_drop > float(
                cfg["maximum_raw_support_drop"]
            ):
                reject("raw_support_drop")
                continue

            (
                original_projection,
                candidate_projection,
                projection_views,
            ) = self._c4_projection_metrics(
                original_corners, candidate_corners, evidence
            )
            projection_delta = (
                candidate_projection - original_projection
            )
            runtime.update(
                {
                    "original_projection": original_projection,
                    "candidate_projection": candidate_projection,
                    "projection_delta": projection_delta,
                    "projection_views": projection_views,
                }
            )
            if projection_views < int(
                cfg["minimum_projection_views"]
            ):
                reject("projection_views")
                continue
            if candidate_projection < float(
                cfg["minimum_weighted_projection_iou"]
            ):
                reject("projection_iou")
                continue
            if projection_delta < -float(
                cfg["maximum_projection_drop"]
            ):
                reject("projection_drop")
                continue

            other_indices = [
                index
                for index in range(len(all_boxes))
                if index != row_index
            ]
            if other_indices:
                original_other = max(
                    aabb_iou(
                        original[:3],
                        original[3:6],
                        all_boxes[index, :3],
                        all_boxes[index, 3:6],
                    )
                    for index in other_indices
                )
                candidate_other = max(
                    aabb_iou(
                        candidate[:3],
                        candidate[3:6],
                        all_boxes[index, :3],
                        all_boxes[index, 3:6],
                    )
                    for index in other_indices
                )
            else:
                original_other = 0.0
                candidate_other = 0.0
            overlap_increase = candidate_other - original_other
            runtime.update(
                {
                    "original_max_other_iou": original_other,
                    "candidate_max_other_iou": candidate_other,
                    "overlap_increase": overlap_increase,
                }
            )
            if (
                candidate_other > float(cfg["maximum_new_overlap"])
                and overlap_increase
                > float(cfg["maximum_overlap_increase"])
            ):
                reject("neighbor_overlap")
                continue

            runtime["verified"] = True
            runtime["reason"] = "verified_observer"
            self.stats["c4_verified"] += 1

        self.stats["c4_refiner_seconds"] = (
            time.perf_counter() - started
        )

    def _observe_trifusion_candidates(
        self,
        *,
        boxes: np.ndarray,
        corners: np.ndarray,
        source_indices: np.ndarray,
        stable_ids: np.ndarray,
        quality_features: np.ndarray,
    ) -> None:
        """Observe occupancy/MSR OBB candidates without touching output.

        The refiner consumes only the isolated C4 per-global Mask-RGBD
        records.  Candidate verification is deterministic and uses no ground
        truth: it reuses the frozen C4 support, projection, extent, and
        neighbour-overlap gates.  Even a verified row remains diagnostic.
        """

        self._last_trifusion_runtime.clear()
        for key in (
            "trifusion_attempted",
            "trifusion_valid",
            "trifusion_candidates",
            "trifusion_verified",
            "trifusion_applied",
            "trifusion_gate_evaluated",
            "trifusion_gate_accepted",
        ):
            self.stats[key] = 0
        self.stats["trifusion_seconds"] = 0.0
        self.stats["trifusion_rejected"] = Counter()
        self.stats["trifusion_gate_rejected"] = Counter()
        if (
            self.residual_track_graph is not None
            and not self._residual_track_failed
        ):
            self._last_residual_track_candidates = (
                self.residual_track_graph.candidates(
                    global_boxes=boxes
                )
            )
            self._last_residual_track_decisions = (
                self.residual_track_graph.candidate_decisions(
                    global_boxes=boxes
                )
            )
            self.stats["residual_track_candidates"] = len(
                self._last_residual_track_candidates
            )
        else:
            self._last_residual_track_candidates = ()
            self._last_residual_track_decisions = ()
            self.stats["residual_track_candidates"] = 0
        cfg = self.config["trifusion_observer"]
        if not cfg["enabled"]:
            return

        started = time.perf_counter()
        gate_cfg = self.config["generic_local_geometry_refiner"]
        safety_cfg = cfg["safety_gate"]
        safety_policy = AP50SafetyGateConfig(
            minimum_improvement_probability=safety_cfg[
                "minimum_improvement_probability"
            ],
            maximum_harm_probability=safety_cfg[
                "maximum_harm_probability"
            ],
            uncertainty_multiplier=safety_cfg[
                "uncertainty_multiplier"
            ],
            maximum_delta_std=safety_cfg["maximum_delta_std"],
            minimum_delta_lower_bound=safety_cfg[
                "minimum_delta_lower_bound"
            ],
            minimum_predicted_iou_margin=safety_cfg[
                "minimum_predicted_iou_margin"
            ],
            require_iou50_crossing=safety_cfg[
                "require_iou50_crossing"
            ],
            minimum_iou50_crossing_probability=safety_cfg[
                "minimum_iou50_crossing_probability"
            ],
        ).validated()
        all_boxes = np.asarray(boxes, dtype=np.float32)
        all_corners = np.asarray(corners, dtype=np.float32)
        all_quality_features = np.asarray(
            quality_features, dtype=np.float32
        )
        if all_corners.shape != (len(all_boxes), 8, 3):
            raise ValueError(
                "TriFusion final corners must align with final boxes"
            )
        if all_quality_features.shape != (
            len(all_boxes),
            QUALITY_FEATURE_DIM,
        ):
            raise ValueError(
                "TriFusion quality features must align with final boxes "
                f"and have width {QUALITY_FEATURE_DIM}"
            )
        if not np.isfinite(all_quality_features).all():
            raise ValueError(
                "TriFusion quality features must be finite"
            )

        for row_index, (
            original,
            original_corners,
            source_index,
            stable_id,
        ) in enumerate(
            zip(all_boxes, all_corners, source_indices, stable_ids)
        ):
            if int(source_index) < 0:
                continue
            stable_key = int(stable_id)
            evidence = self.generic_geometry_global_tracks.get(stable_key)
            if evidence is None:
                continue

            combined_gate_features = np.concatenate(
                (
                    all_quality_features[row_index],
                    np.ones(1, dtype=np.float32),
                    np.zeros(
                        OCCUPANCY_MSR_FEATURE_DIM,
                        dtype=np.float32,
                    ),
                )
            ).astype(np.float32, copy=False)
            runtime: Dict[str, Any] = {
                "stable_id": stable_key,
                "attempted": True,
                "candidate_valid": False,
                "is_candidate": False,
                "candidate_verified": False,
                "applied": False,
                "reason": "proposal_not_run",
                "original_corners": np.asarray(
                    original_corners, dtype=np.float32
                ).copy(),
                "candidate_corners": np.asarray(
                    original_corners, dtype=np.float32
                ).copy(),
                "features": np.zeros(
                    OCCUPANCY_MSR_FEATURE_DIM, dtype=np.float32
                ),
                "gate_features": combined_gate_features.copy(),
                "gate_enabled": bool(
                    self.trifusion_ap50_gate is not None
                ),
                "gate_evaluated": False,
                "gate_accepted": False,
                "gate_reason": (
                    "pending_geometry"
                    if self.trifusion_ap50_gate is not None
                    else "disabled"
                ),
                "gate_lower_confidence_delta": np.nan,
                "gate_delta_mean": np.nan,
                "gate_delta_std": np.nan,
                "gate_improvement_probability": np.nan,
                "gate_harm_probability": np.nan,
                "gate_original_iou": np.nan,
                "gate_candidate_iou": np.nan,
                "gate_cross_iou25_probability": np.nan,
                "gate_cross_iou50_probability": np.nan,
                "source": "occupancy_msr",
                "projection_original": np.nan,
                "projection_candidate": np.nan,
                "projection_views": 0,
                "candidate_support": np.nan,
                "support_drop": np.nan,
                "original_candidate_iou": np.nan,
                "original_max_other_iou": np.nan,
                "candidate_max_other_iou": np.nan,
                "overlap_increase": np.nan,
            }
            self._last_trifusion_runtime[stable_key] = runtime
            self.stats["trifusion_attempted"] += 1

            def reject(reason: str) -> None:
                runtime["reason"] = reason
                self.stats["trifusion_rejected"][reason] += 1

            try:
                proposal = propose_local_occupancy_msr(
                    original_corners,
                    evidence.memory.selected_view_records,
                    config=cfg["proposal"],
                )
                candidate_corners = np.asarray(
                    proposal.candidate_corners, dtype=np.float32
                )
                features = np.asarray(
                    proposal.feature_vector, dtype=np.float32
                )
            except (TypeError, ValueError, FloatingPointError) as error:
                reject(f"proposal_error:{type(error).__name__}")
                continue

            if (
                candidate_corners.shape != (8, 3)
                or not np.isfinite(candidate_corners).all()
                or features.shape != (OCCUPANCY_MSR_FEATURE_DIM,)
                or not np.isfinite(features).all()
            ):
                reject("invalid_candidate")
                continue

            runtime.update(
                {
                    "candidate_valid": True,
                    "is_candidate": bool(proposal.is_candidate),
                    "reason": str(proposal.reason),
                    "candidate_corners": candidate_corners.copy(),
                    "features": features.copy(),
                    "gate_features": np.concatenate(
                        (
                            all_quality_features[row_index],
                            np.ones(1, dtype=np.float32),
                            features,
                        )
                    ).astype(np.float32, copy=False),
                    "candidate_support": float(
                        proposal.candidate_support
                    ),
                    "support_drop": float(proposal.support_drop),
                }
            )
            self.stats["trifusion_valid"] += 1
            if not proposal.is_candidate:
                reject(str(proposal.reason))
                continue
            self.stats["trifusion_candidates"] += 1

            original_local = np.asarray(
                proposal.original_local_box, dtype=np.float32
            )
            candidate_local = np.asarray(
                proposal.candidate_local_box, dtype=np.float32
            )
            original_candidate_iou = aabb_iou(
                original_local[:3],
                original_local[3:6],
                candidate_local[:3],
                candidate_local[3:6],
            )
            runtime["original_candidate_iou"] = (
                original_candidate_iou
            )
            if original_candidate_iou < float(
                gate_cfg["minimum_original_candidate_iou"]
            ):
                reject("original_iou")
                continue
            if float(proposal.candidate_support) < float(
                gate_cfg["minimum_raw_candidate_support"]
            ):
                reject("raw_support")
                continue
            if float(proposal.support_drop) > float(
                gate_cfg["maximum_raw_support_drop"]
            ):
                reject("raw_support_drop")
                continue
            final_extent = self.config["output_filter"][
                "final_minimum_extent"
            ]
            if final_extent is not None and np.any(
                candidate_local[3:6] < float(final_extent)
            ):
                reject("extent_survival")
                continue

            (
                original_projection,
                candidate_projection,
                projection_views,
            ) = self._c4_projection_metrics(
                original_corners, candidate_corners, evidence
            )
            runtime.update(
                {
                    "projection_original": original_projection,
                    "projection_candidate": candidate_projection,
                    "projection_views": projection_views,
                }
            )
            if projection_views < int(
                gate_cfg["minimum_projection_views"]
            ):
                reject("projection_views")
                continue
            if candidate_projection < float(
                gate_cfg["minimum_weighted_projection_iou"]
            ):
                reject("projection_iou")
                continue
            if (
                candidate_projection - original_projection
                < -float(gate_cfg["maximum_projection_drop"])
            ):
                reject("projection_drop")
                continue

            try:
                candidate_box = corners_to_center_size(
                    candidate_corners[None, ...]
                )[0]
            except (TypeError, ValueError, FloatingPointError):
                reject("invalid_candidate_box")
                continue
            other_indices = [
                index
                for index in range(len(all_boxes))
                if index != row_index
            ]
            if other_indices:
                original_other = max(
                    aabb_iou(
                        original[:3],
                        original[3:6],
                        all_boxes[index, :3],
                        all_boxes[index, 3:6],
                    )
                    for index in other_indices
                )
                candidate_other = max(
                    aabb_iou(
                        candidate_box[:3],
                        candidate_box[3:6],
                        all_boxes[index, :3],
                        all_boxes[index, 3:6],
                    )
                    for index in other_indices
                )
            else:
                original_other = 0.0
                candidate_other = 0.0
            overlap_increase = candidate_other - original_other
            runtime.update(
                {
                    "original_max_other_iou": original_other,
                    "candidate_max_other_iou": candidate_other,
                    "overlap_increase": overlap_increase,
                }
            )
            if (
                candidate_other
                > float(gate_cfg["maximum_new_overlap"])
                and overlap_increase
                > float(gate_cfg["maximum_overlap_increase"])
            ):
                reject("neighbor_overlap")
                continue

            runtime["candidate_verified"] = True
            runtime["reason"] = "verified_observer"
            self.stats["trifusion_verified"] += 1
            if self.trifusion_ap50_gate is not None:
                runtime["gate_evaluated"] = True
                self.stats["trifusion_gate_evaluated"] += 1
                try:
                    decision = self.trifusion_ap50_gate.decide(
                        runtime["gate_features"],
                        geometry_verified=True,
                        config=safety_policy,
                    )
                except (
                    TypeError,
                    ValueError,
                    FloatingPointError,
                ) as error:
                    gate_reason = (
                        f"gate_error:{type(error).__name__}"
                    )
                    runtime["gate_reason"] = gate_reason
                    self.stats["trifusion_gate_rejected"][
                        gate_reason
                    ] += 1
                else:
                    prediction = decision.prediction
                    runtime.update(
                        {
                            "gate_accepted": bool(
                                decision.accepted
                            ),
                            "gate_reason": str(decision.reason),
                            "gate_lower_confidence_delta": float(
                                decision.lower_confidence_delta
                            ),
                            "gate_delta_mean": float(
                                prediction.delta_mean
                            ),
                            "gate_delta_std": float(
                                prediction.delta_std
                            ),
                            "gate_improvement_probability": float(
                                prediction.improvement_probability
                            ),
                            "gate_harm_probability": float(
                                prediction.harm_probability
                            ),
                            "gate_original_iou": float(
                                prediction.original_iou
                            ),
                            "gate_candidate_iou": float(
                                prediction.candidate_iou
                            ),
                            "gate_cross_iou25_probability": float(
                                prediction.cross_iou25_probability
                            ),
                            "gate_cross_iou50_probability": float(
                                prediction.cross_iou50_probability
                            ),
                        }
                    )
                    if decision.accepted:
                        self.stats["trifusion_gate_accepted"] += 1
                    else:
                        self.stats["trifusion_gate_rejected"][
                            str(decision.reason)
                        ] += 1
            # There is deliberately no geometry write and no route that can
            # increment trifusion_applied.

        self.stats["trifusion_seconds"] = (
            time.perf_counter() - started
        )
        if self.trifusion_missing_graph is not None:
            self._last_trifusion_missing_candidates = (
                self.trifusion_missing_graph.candidates(
                    global_boxes=all_boxes
                )
            )
            self.stats["trifusion_missing_candidates"] = len(
                self._last_trifusion_missing_candidates
            )

    def _observe_yidu_candidates(
        self,
        *,
        corners: np.ndarray,
        scores: np.ndarray,
        source_indices: np.ndarray,
        stable_ids: np.ndarray,
        quality_features: np.ndarray,
    ) -> None:
        """Run the exact cumulative YiDu stage without touching output."""

        self._last_yidu_runtime.clear()
        for key in (
            "yidu_attempted",
            "yidu_valid",
            "yidu_component_candidates",
            "yidu_occupancy_candidates",
            "yidu_query_candidates",
            "yidu_gate_evaluated",
            "yidu_gate_accepted",
            "yidu_applied",
        ):
            self.stats[key] = 0
        self.stats["yidu_seconds"] = 0.0
        self.stats["yidu_rejected"] = Counter()
        cfg = self.config["yidu_ablation"]
        if not cfg["enabled"]:
            return

        started = time.perf_counter()
        all_corners = np.asarray(corners, dtype=np.float32)
        all_scores = np.asarray(scores, dtype=np.float32)
        all_features = np.asarray(quality_features, dtype=np.float32)
        if all_corners.shape != (len(all_scores), 8, 3):
            raise ValueError("YiDu corners must align with final scores")
        if all_features.shape != (
            len(all_scores),
            QUALITY_FEATURE_DIM,
        ):
            raise ValueError("YiDu B6 features must align with final rows")

        stage = str(cfg["stage"])
        for row_index, (
            original_corners,
            detector_score,
            source_index,
            stable_id,
        ) in enumerate(
            zip(all_corners, all_scores, source_indices, stable_ids)
        ):
            if int(source_index) < 0:
                continue
            stable_key = int(stable_id)
            evidence = self.generic_geometry_global_tracks.get(stable_key)
            runtime: Dict[str, Any] = {
                "attempted": False,
                "valid": False,
                "applied": False,
                "reason": "unobserved",
                "stage": stage,
                "selected_source": "original",
                "original_corners": np.asarray(
                    original_corners, dtype=np.float32
                ).copy(),
                "raw_candidate_corners": np.asarray(
                    original_corners, dtype=np.float32
                ).copy(),
                "superpoint_candidate_corners": np.asarray(
                    original_corners, dtype=np.float32
                ).copy(),
                "occupancy_candidate_corners": np.asarray(
                    original_corners, dtype=np.float32
                ).copy(),
                "selected_candidate_corners": np.asarray(
                    original_corners, dtype=np.float32
                ).copy(),
                "input_point_count": 0,
                "cropped_point_count": 0,
                "selected_view_count": 0,
                "component_count": 0,
                "selected_component_id": -1,
                "component_features": np.zeros(
                    YIDU_COMPONENT_FEATURE_DIM, dtype=np.float32
                ),
                "occupancy_features": np.zeros(
                    OCCUPANCY_MSR_FEATURE_DIM, dtype=np.float32
                ),
                "raw_fused_features": np.zeros(
                    RAW_FUSED_QUERY_FEATURE_DIM, dtype=np.float32
                ),
                "gate_features": np.zeros(
                    YIDU_GATE_FEATURE_DIM, dtype=np.float32
                ),
                "gate_evaluated": False,
                "gate_accepted": False,
                "gate_reason": "disabled",
                "gate_delta_mean": np.nan,
                "gate_delta_std": np.nan,
                "gate_improvement_probability": np.nan,
                "gate_harm_probability": np.nan,
                "gate_original_iou": np.nan,
                "gate_candidate_iou": np.nan,
                "gate_cross_iou25_probability": np.nan,
                "gate_cross_iou50_probability": np.nan,
            }
            self._last_yidu_runtime[stable_key] = runtime
            if evidence is None:
                self.stats["yidu_rejected"]["unobserved"] += 1
                continue
            runtime["attempted"] = True
            self.stats["yidu_attempted"] += 1
            raw_runtime = self._last_c4_runtime.get(stable_key, {})
            raw_candidate = raw_runtime.get(
                "candidate_corners", original_corners
            )
            raw_verified = bool(raw_runtime.get("verified", False))
            try:
                observation = observe_yidu_local_geometry(
                    stage=stage,
                    original_corners=original_corners,
                    view_records=evidence.memory.selected_view_records,
                    detector_score=float(detector_score),
                    b6_quality_features=all_features[row_index],
                    raw_candidate_corners=raw_candidate,
                    raw_candidate_verified=raw_verified,
                    config=cfg["local_observer"],
                    quality_gate=self.yidu_ap50_gate,
                )
            except (
                TypeError,
                ValueError,
                FloatingPointError,
            ) as error:
                reason = f"observer_error:{type(error).__name__}"
                runtime["reason"] = reason
                self.stats["yidu_rejected"][reason] += 1
                continue

            component_count = (
                0
                if observation.component_set is None
                else observation.component_set.component_count
            )
            gate = observation.gate_decision
            runtime.update(
                {
                    "valid": True,
                    "reason": str(observation.reason),
                    "selected_source": str(
                        observation.selected_source
                    ),
                    "raw_candidate_corners": np.asarray(
                        observation.raw_candidate_corners,
                        dtype=np.float32,
                    ).copy(),
                    "superpoint_candidate_corners": np.asarray(
                        observation.superpoint_candidate_corners,
                        dtype=np.float32,
                    ).copy(),
                    "occupancy_candidate_corners": np.asarray(
                        observation.occupancy_candidate_corners,
                        dtype=np.float32,
                    ).copy(),
                    "selected_candidate_corners": np.asarray(
                        observation.selected_candidate_corners,
                        dtype=np.float32,
                    ).copy(),
                    "input_point_count": int(
                        observation.input_point_count
                    ),
                    "cropped_point_count": int(
                        observation.cropped_point_count
                    ),
                    "selected_view_count": int(
                        observation.selected_view_count
                    ),
                    "component_count": int(component_count),
                    "selected_component_id": int(
                        observation.selected_component_id
                    ),
                    "component_features": np.asarray(
                        observation.component_features,
                        dtype=np.float32,
                    ).copy(),
                    "occupancy_features": np.asarray(
                        observation.occupancy_features,
                        dtype=np.float32,
                    ).copy(),
                    "raw_fused_features": np.asarray(
                        observation.raw_fused_selected_features,
                        dtype=np.float32,
                    ).copy(),
                    "gate_features": np.asarray(
                        observation.gate_features,
                        dtype=np.float32,
                    ).copy(),
                    "gate_evaluated": gate is not None,
                    "gate_accepted": bool(
                        False if gate is None else gate.accepted
                    ),
                    "gate_reason": (
                        "disabled" if gate is None else str(gate.reason)
                    ),
                }
            )
            if gate is not None:
                prediction = gate.prediction
                runtime.update(
                    {
                        "gate_delta_mean": float(
                            prediction.delta_mean
                        ),
                        "gate_delta_std": float(prediction.delta_std),
                        "gate_improvement_probability": float(
                            prediction.improvement_probability
                        ),
                        "gate_harm_probability": float(
                            prediction.harm_probability
                        ),
                        "gate_original_iou": float(
                            prediction.original_iou
                        ),
                        "gate_candidate_iou": float(
                            prediction.candidate_iou
                        ),
                        "gate_cross_iou25_probability": float(
                            prediction.cross_iou25_probability
                        ),
                        "gate_cross_iou50_probability": float(
                            prediction.cross_iou50_probability
                        ),
                    }
                )
            self.stats["yidu_valid"] += 1
            if observation.selected_component_id >= 0:
                self.stats["yidu_component_candidates"] += 1
            if (
                observation.occupancy_proposal is not None
                and observation.occupancy_proposal.is_candidate
            ):
                self.stats["yidu_occupancy_candidates"] += 1
            if observation.raw_fused_observation is not None:
                self.stats["yidu_query_candidates"] += 1
            if gate is not None:
                self.stats["yidu_gate_evaluated"] += 1
                if gate.accepted:
                    self.stats["yidu_gate_accepted"] += 1
            # Deliberately no write to corners/scores/count/order/IDs.

        self.stats["yidu_seconds"] = time.perf_counter() - started

    def _supplemental_outputs(
        self,
        global_boxes: np.ndarray,
    ) -> List[_SupplementalOutput]:
        """Materialize, score once with B6, then optionally refine with B5."""

        output: List[_SupplementalOutput] = []
        deduplicated = 0
        cfg = self.config["supplemental_output"]
        output_cfg = self.config["output_filter"]
        configured_supplemental_extent = cfg["minimum_extent"]
        final_minimum_extent = output_cfg["final_minimum_extent"]
        if cfg["class_aware_extent"] and final_minimum_extent is not None:
            # C1's unknown labels follow the exact final ScanNet contract.
            # The explicit sink/door/window rules remain source-specific.
            supplemental_minimum_extent = float(final_minimum_extent)
        else:
            supplemental_minimum_extent = float(
                output_cfg["minimum_extent"]
                if configured_supplemental_extent is None
                else configured_supplemental_extent
            )

        def duplicates_global_in_bev(candidate_box: np.ndarray) -> bool:
            if not cfg["bev_duplicate_enabled"]:
                return False
            for global_box in global_boxes:
                bev_iou, containment = bev_iou_and_containment(
                    candidate_box, global_box
                )
                z_containment = _axis_overlap_containment(
                    candidate_box, global_box, axis=2
                )
                if (
                    bev_iou >= cfg["bev_duplicate_iou"]
                    and containment >= cfg["bev_duplicate_containment"]
                    and z_containment
                    >= cfg["bev_duplicate_min_z_containment"]
                ):
                    return True
            return False

        for key in (
            "absorbed_recovery_considered",
            "absorbed_recovery_eligible",
            "absorbed_recovery_output",
            "supplemental_considered",
            "supplemental_rejected_graph",
            "supplemental_rejected_extent",
            "supplemental_rejected_class_extent",
            "supplemental_rejected_refined_extent",
            "supplemental_rejected_score",
            "supplemental_rejected_projection",
            "supplemental_rejected_global",
            "supplemental_rejected_bev_global",
            "supplemental_rejected_refined_global",
            "supplemental_rejected_refined_bev_global",
            "supplemental_scores_rank_mapped",
            "supplemental_planar_deduplicated",
            "supplemental_refined_deduplicated",
            "supplemental_b5_attempted",
            "supplemental_b5_accepted",
            "supplemental_output",
        ):
            self.stats[key] = 0
        if not cfg["enabled"] or self.track_manager is None:
            self.stats["supplemental_deduplicated"] = 0
            return output
        candidates: List[_SupplementalMaterialized] = []
        recovered_track_ids = set()
        track_rows = []
        for track in self.track_manager.confirmed_tracks(include_archived=True):
            track_rows.append(
                (
                    track,
                    self.supplemental_metadata.get(track.track_id),
                    False,
                )
            )
        if cfg["recover_absorbed_confirmed"]:
            live_track_ids = {
                int(track.track_id) for track, _, _ in track_rows
            }
            for track_id, record in sorted(
                self.absorbed_supplemental_records.items()
            ):
                if int(track_id) in live_track_ids:
                    raise RuntimeError(
                        "absorbed recovery track unexpectedly remains live"
                    )
                track_rows.append((record.track, record.metadata, True))
                recovered_track_ids.add(int(track_id))

        for track, metadata, recovered_absorbed in track_rows:
            if recovered_absorbed:
                self.stats["absorbed_recovery_considered"] += 1
            if metadata is None or track.view_count < cfg["min_confirmations"]:
                continue
            graph = getattr(metadata, "graph", None)
            if (
                cfg["require_mask_graph_confirmation"]
                and (graph is None or not bool(graph.confirmed))
            ):
                self.stats["supplemental_rejected_graph"] += 1
                continue
            track_box = track.memory.aabb
            if track_box is None:
                continue
            self.stats["supplemental_considered"] += 1
            box = np.concatenate(track_box).astype(np.float32)
            if not supplemental_extent_is_valid(
                box[3:6],
                metadata.stats.label,
                cfg,
                default_minimum_extent=supplemental_minimum_extent,
            ):
                if cfg["class_aware_extent"]:
                    self.stats["supplemental_rejected_class_extent"] += 1
                else:
                    self.stats["supplemental_rejected_extent"] += 1
                continue
            detector_score = metadata.stats.mean_score
            if detector_score < cfg["min_score"]:
                self.stats["supplemental_rejected_score"] += 1
                continue
            structural_mapping = self._quality_mapping(
                original_box=box,
                final_box=box,
                detector_score=detector_score,
                memory=track.memory,
                stats=metadata.stats,
                supplemental=True,
                refiner_quality=0.5,
            )
            if (
                structural_mapping["projection_iou"]
                < cfg["min_projection_iou"]
            ):
                self.stats["supplemental_rejected_projection"] += 1
                continue
            if len(global_boxes):
                maximum_overlap = max(
                    aabb_iou(
                        box[:3],
                        box[3:6],
                        global_box[:3],
                        global_box[3:6],
                    )
                    for global_box in global_boxes
                )
                if maximum_overlap >= cfg["drop_if_global_iou"]:
                    self.stats["supplemental_rejected_global"] += 1
                    continue
                if duplicates_global_in_bev(box):
                    self.stats["supplemental_rejected_bev_global"] += 1
                    continue
            candidates.append(
                _SupplementalMaterialized(
                    track_id=int(track.track_id),
                    box=box,
                    detector_score=float(detector_score),
                    label=metadata.stats.label,
                    memory=track.memory,
                    stats=metadata.stats,
                    view_count=int(track.view_count),
                    recovered_absorbed=bool(recovered_absorbed),
                )
            )
            if recovered_absorbed:
                self.stats["absorbed_recovery_eligible"] += 1

        quality_cfg = self.config["quality"]
        refiner_scope = self.config["box_refiner"]["apply_scope"]
        apply_supplemental_refiner = (
            self.box_refiner is not None
            and refiner_scope
            in {
                "confirmed_supplemental",
                "global_and_confirmed_supplemental",
            }
        )
        post_candidates: List[
            Tuple[_SupplementalOutput, int, int]
        ] = []
        for candidate in sorted(candidates, key=lambda item: item.track_id):
            original_box = candidate.box.copy()
            original_corners = aabb_corners(
                original_box[:3], original_box[3:6]
            )
            mapping = self._quality_mapping(
                original_box=original_box,
                final_box=original_box,
                detector_score=candidate.detector_score,
                memory=candidate.memory,
                stats=candidate.stats,
                supplemental=True,
                refiner_quality=0.5,
            )
            if quality_cfg["apply_to_supplemental"]:
                score = self._score(
                    candidate.detector_score,
                    mapping,
                    observed=True,
                )
            else:
                score = candidate.detector_score
            if score < cfg["min_score"]:
                self.stats["supplemental_rejected_score"] += 1
                continue
            if cfg["rank_after_globals"]:
                projection_weight = float(
                    cfg["rank_projection_weight"]
                )
                rank_quality = (
                    (1.0 - projection_weight) * float(score)
                    + projection_weight
                    * float(mapping["projection_iou"])
                )
                if candidate.recovered_absorbed:
                    rank_quality += float(
                        cfg["rank_recovered_bonus"]
                    )
                score = (
                    float(cfg["rank_score_floor"])
                    + (
                        float(cfg["rank_score_ceiling"])
                        - float(cfg["rank_score_floor"])
                    )
                    * float(np.clip(rank_quality, 0.0, 1.0))
                )
                self.stats["supplemental_scores_rank_mapped"] += 1

            final_box = original_box.copy()
            final_corners = original_corners.copy()
            refit_applied = False
            refit_reason = "supplemental_identity"
            if apply_supplemental_refiner:
                self.stats["supplemental_b5_attempted"] += 1
                evidence = GlobalEvidence(
                    stable_id=-(candidate.track_id + 1),
                    memory=candidate.memory,
                    stats=candidate.stats,
                    detector_score=candidate.detector_score,
                    last_box=original_box.copy(),
                )
                if self.box_refiner_coordinate_frame == "box_local":
                    (
                        refined_box,
                        refined_corners,
                        _,
                        accepted_refit,
                        refit_reason,
                    ) = self._run_oriented_neural_refiner(
                        original_corners,
                        evidence,
                        mapping,
                    )
                else:
                    refined_box, _, accepted_refit = (
                        self._run_neural_refiner(
                            original_box,
                            evidence,
                            mapping,
                        )
                    )
                    refined_corners = aabb_corners(
                        refined_box[:3], refined_box[3:6]
                    )
                    refit_reason = (
                        "neural_accepted"
                        if accepted_refit
                        else "neural_rejected"
                    )
                if accepted_refit:
                    final_box = refined_box
                    final_corners = refined_corners
                    refit_applied = True
                    self.stats["supplemental_b5_accepted"] += 1
                    self.stats["neural_refits_accepted"] += 1

            # A learned residual may cross a structural boundary even though
            # the pre-B5 track box passed every gate. Recheck only the final
            # geometry; B6 remains the single frozen score computed above.
            if not supplemental_extent_is_valid(
                final_box[3:6],
                candidate.label,
                cfg,
                default_minimum_extent=supplemental_minimum_extent,
            ):
                self.stats[
                    "supplemental_rejected_refined_extent"
                ] += 1
                continue
            if len(global_boxes):
                maximum_final_global_overlap = max(
                    aabb_iou(
                        final_box[:3],
                        final_box[3:6],
                        global_box[:3],
                        global_box[3:6],
                    )
                    for global_box in global_boxes
                )
                if (
                    maximum_final_global_overlap
                    >= cfg["drop_if_global_iou"]
                ):
                    self.stats[
                        "supplemental_rejected_refined_global"
                    ] += 1
                    continue
                if duplicates_global_in_bev(final_box):
                    self.stats[
                        "supplemental_rejected_refined_bev_global"
                    ] += 1
                    continue
            post_candidates.append(
                (
                    _SupplementalOutput(
                    box=np.asarray(final_box, dtype=np.float32),
                    corners=np.asarray(final_corners, dtype=np.float32),
                    score=float(score),
                    stable_id=-(candidate.track_id + 1),
                    label=candidate.label,
                    quality_features=quality_feature_vector(mapping),
                    memory=candidate.memory,
                    stats=candidate.stats,
                    original_box=original_box,
                    original_corners=original_corners,
                    refit_applied=bool(refit_applied),
                    refit_reason=refit_reason,
                    ),
                    int(candidate.view_count),
                    int(candidate.track_id),
                )
            )

        # Score every candidate once before suppression. This ensures the
        # frozen B6 rank, rather than the raw detector rank, selects the
        # representative retained from a re-entered/duplicate graph track.
        for candidate_output, _, _ in sorted(
            post_candidates,
            key=lambda item: (
                -item[0].score,
                -item[1],
                item[2],
            ),
        ):
            duplicate = None
            planar_duplicate = False
            for previous in output:
                normalized_label = (
                    ""
                    if candidate_output.label is None
                    else str(candidate_output.label).strip().casefold()
                )
                previous_label = (
                    ""
                    if previous.label is None
                    else str(previous.label).strip().casefold()
                )
                if (
                    cfg["planar_duplicate_enabled"]
                    and normalized_label
                    == previous_label
                    and normalized_label
                    in set(cfg["planar_extent_labels"])
                ):
                    bev_iou, bev_containment = (
                        bev_iou_and_containment(
                            candidate_output.box, previous.box
                        )
                    )
                    z_containment = _axis_overlap_containment(
                        candidate_output.box,
                        previous.box,
                        axis=2,
                    )
                    if (
                        bev_iou
                        >= cfg["planar_duplicate_bev_iou"]
                        and bev_containment
                        >= cfg["planar_duplicate_containment"]
                        and z_containment
                        >= cfg[
                            "planar_duplicate_min_z_containment"
                        ]
                    ):
                        duplicate = previous
                        planar_duplicate = True
                        break
                final_overlap = aabb_iou(
                    candidate_output.box[:3],
                    candidate_output.box[3:6],
                    previous.box[:3],
                    previous.box[3:6],
                )
                if final_overlap >= cfg["drop_if_supplemental_iou"]:
                    duplicate = previous
                    break
            if duplicate is not None:
                if planar_duplicate:
                    self.stats[
                        "supplemental_planar_deduplicated"
                    ] += 1
                    continue
                original_overlap = aabb_iou(
                    candidate_output.original_box[:3],
                    candidate_output.original_box[3:6],
                    duplicate.original_box[:3],
                    duplicate.original_box[3:6],
                )
                if original_overlap >= cfg["drop_if_supplemental_iou"]:
                    deduplicated += 1
                else:
                    self.stats[
                        "supplemental_refined_deduplicated"
                    ] += 1
                continue
            output.append(candidate_output)
        self.stats["supplemental_deduplicated"] = deduplicated
        self.stats["supplemental_output"] = len(output)
        self.stats["absorbed_recovery_output"] = sum(
            int(-(item.stable_id + 1)) in recovered_track_ids
            for item in output
        )
        return self._apply_c2_geometry_refinement(
            output, global_boxes
        )

    def finalize(
        self,
        *,
        global_corners: Any,
        global_scores: Any,
        stable_ids: Any,
        scene_id: Optional[str] = None,
    ) -> FinalRefinementResult:
        """Return refined and supplemental detections without mutating inputs."""

        self._last_observer_zero_write_audit = (
            _empty_observer_zero_write_audit()
        )
        corners_input = np.asarray(global_corners, dtype=np.float32)
        if corners_input.size == 0:
            corners_input = np.empty((0, 8, 3), dtype=np.float32)
        boxes, scores, ids = self._global_inputs(
            corners_input, global_scores, stable_ids
        )
        if not self.enabled:
            empty_features = np.zeros(
                (len(boxes), QUALITY_FEATURE_DIM), dtype=np.float32
            )
            return FinalRefinementResult(
                corners=corners_input.copy(),
                boxes=boxes,
                scores=scores.copy(),
                source_indices=np.arange(len(boxes), dtype=np.int64),
                stable_ids=ids.copy(),
                labels=tuple(None for _ in boxes),
                quality_features=empty_features,
                refit_original_boxes=boxes.copy(),
                refit_original_corners=corners_input.copy(),
                refit_applied=np.zeros(len(boxes), dtype=bool),
                refit_reasons=tuple("disabled" for _ in boxes),
                refit_changed_axes=np.zeros((len(boxes), 3), dtype=bool),
                refit_boundary_delta=np.zeros(
                    (len(boxes), 6), dtype=np.float32
                ),
                refit_local_original_boxes=np.full(
                    (len(boxes), 6), np.nan, dtype=np.float32
                ),
                refit_local_candidate_boxes=np.full(
                    (len(boxes), 6), np.nan, dtype=np.float32
                ),
                refit_local_basis=np.full(
                    (len(boxes), 3, 3), np.nan, dtype=np.float32
                ),
                refit_local_frame_valid=np.zeros(
                    len(boxes), dtype=bool
                ),
                summary={"enabled": False},
            )

        self._sync_global_tracks(boxes, scores, ids)
        joint_enabled = bool(
            self.config["joint_local_head"]["enabled"]
        )
        joint_predictions = self._run_joint_local_head_batch(
            corners=corners_input,
            boxes=boxes,
            scores=scores,
            stable_ids=ids,
        )
        final_corners: List[np.ndarray] = []
        final_boxes: List[np.ndarray] = []
        final_scores: List[float] = []
        source_indices: List[int] = []
        result_ids: List[int] = []
        labels: List[Optional[str]] = []
        feature_rows: List[np.ndarray] = []
        memories: List[Optional[ObjectGeometryMemory]] = []
        evidence_stats_rows: List[Optional[EvidenceStats]] = []
        refit_original_boxes: List[np.ndarray] = []
        refit_original_corners: List[np.ndarray] = []
        refit_applied: List[bool] = []
        refit_reasons: List[str] = []
        refit_changed_axes: List[np.ndarray] = []
        refit_boundary_delta: List[np.ndarray] = []
        refit_local_original_boxes: List[np.ndarray] = []
        refit_local_candidate_boxes: List[np.ndarray] = []
        refit_local_basis: List[np.ndarray] = []
        refit_local_frame_valid: List[bool] = []

        for index, (original, detector_score, stable_id) in enumerate(
            zip(boxes, scores, ids)
        ):
            evidence = self.global_tracks.get(int(stable_id))
            refined = original.copy()
            refined_oriented_corners: Optional[np.ndarray] = None
            robust_accepted = False
            refit_reason = "unobserved"
            refiner_quality = 0.5
            if joint_enabled:
                joint_prediction = joint_predictions.get(index)
                if joint_prediction is None:
                    # Unobserved/invalid rows are exact identity fallbacks:
                    # no learned score may modify their detector confidence.
                    refined_oriented_corners = corners_input[index].copy()
                    score = float(detector_score)
                    runtime = self._last_joint_runtime.get(
                        int(stable_id), {}
                    )
                    refit_reason = str(
                        runtime.get("reason", "joint_invalid")
                    )
                    mapping = self._quality_mapping(
                        original_box=original,
                        final_box=original,
                        detector_score=float(detector_score),
                        memory=(
                            evidence.memory
                            if evidence is not None
                            else None
                        ),
                        stats=(
                            evidence.stats
                            if evidence is not None
                            else None
                        ),
                        supplemental=False,
                        refiner_quality=0.5,
                    )
                else:
                    (
                        refined,
                        refined_oriented_corners,
                        score,
                        mapping,
                        robust_accepted,
                        refit_reason,
                    ) = self._apply_joint_prediction(
                        joint_prediction
                    )
            else:
                if evidence is not None:
                    self.stats["refits_attempted"] += 1
                    if (
                        self.config["refit"]["strategy"]
                        == "visibility_aware"
                        and self.config["refit"][
                            "preserve_box_orientation"
                        ]
                    ):
                        (
                            refined,
                            refined_oriented_corners,
                            robust_accepted,
                            reason,
                        ) = self._oriented_visibility_refit(
                            corners_input[index],
                            evidence,
                        )
                    else:
                        refined, robust_accepted, reason = (
                            self._robust_refit(original, evidence)
                        )
                    if robust_accepted:
                        self.stats["refits_accepted"] += 1
                    elif reason != "disabled":
                        self.stats["rejected"][reason] += 1
                    refit_reason = reason
                    quality_geometry = (
                        original
                        if self.config["quality"]["feature_geometry"]
                        == "original"
                        else refined
                    )
                    preliminary = self._quality_mapping(
                        original_box=original,
                        final_box=quality_geometry,
                        detector_score=float(detector_score),
                        memory=evidence.memory,
                        stats=evidence.stats,
                        supplemental=False,
                        refiner_quality=0.5,
                    )
                    neural_corners = None
                    neural = refined
                    neural_accepted = False
                    refiner_scope = self.config["box_refiner"][
                        "apply_scope"
                    ]
                    apply_global_refiner = refiner_scope in {
                        "global",
                        "global_and_confirmed_supplemental",
                    }
                    neural_reason = (
                        "neural_disabled"
                        if apply_global_refiner
                        else "neural_scope"
                    )
                    if (
                        apply_global_refiner
                        and self.box_refiner_coordinate_frame == "box_local"
                    ):
                        (
                            neural,
                            neural_corners,
                            refiner_quality,
                            neural_accepted,
                            neural_reason,
                        ) = self._run_oriented_neural_refiner(
                            corners_input[index],
                            evidence,
                            preliminary,
                        )
                    elif apply_global_refiner:
                        (
                            neural,
                            refiner_quality,
                            neural_accepted,
                        ) = self._run_neural_refiner(
                            refined, evidence, preliminary
                        )
                        neural_reason = (
                            "neural_accepted"
                            if neural_accepted
                            else "neural_rejected"
                        )
                    if neural_accepted:
                        refined = neural
                        refined_oriented_corners = neural_corners
                        self.stats["neural_refits_accepted"] += 1
                        refit_reason = "neural_accepted"
                    elif self.box_refiner is not None:
                        refit_reason = neural_reason
                quality_geometry = (
                    original
                    if self.config["quality"]["feature_geometry"]
                    == "original"
                    else refined
                )
                mapping = self._quality_mapping(
                    original_box=original,
                    final_box=quality_geometry,
                    detector_score=float(detector_score),
                    memory=(
                        evidence.memory if evidence is not None else None
                    ),
                    stats=(
                        evidence.stats if evidence is not None else None
                    ),
                    supplemental=False,
                    refiner_quality=refiner_quality,
                )
                score = self._score(
                    float(detector_score),
                    mapping,
                    observed=evidence is not None
                    and evidence.memory.observation_count > 0,
                )
            if refined_oriented_corners is not None:
                corners = refined_oriented_corners
            elif robust_accepted or not np.array_equal(refined, original):
                corners = aabb_corners(refined[:3], refined[3:6])
            else:
                corners = corners_input[index].copy()
            final_corners.append(corners)
            final_boxes.append(refined)
            final_scores.append(score)
            source_indices.append(index)
            result_ids.append(int(stable_id))
            labels.append(evidence.stats.label if evidence is not None else None)
            feature_rows.append(quality_feature_vector(mapping))
            memories.append(evidence.memory if evidence is not None else None)
            evidence_stats_rows.append(
                evidence.stats if evidence is not None else None
            )
            original_lower = original[:3] - 0.5 * original[3:6]
            original_upper = original[:3] + 0.5 * original[3:6]
            refined_lower = refined[:3] - 0.5 * refined[3:6]
            refined_upper = refined[:3] + 0.5 * refined[3:6]
            boundary_delta = np.concatenate(
                (
                    refined_lower - original_lower,
                    refined_upper - original_upper,
                )
            ).astype(np.float32)
            local_original_box = np.full(6, np.nan, dtype=np.float32)
            local_candidate_box = np.full(6, np.nan, dtype=np.float32)
            local_basis = np.full((3, 3), np.nan, dtype=np.float32)
            local_frame_valid = False
            if (
                self.config["refit"]["preserve_box_orientation"]
                or (
                    self.config["box_refiner"]["enabled"]
                    and self.box_refiner_coordinate_frame == "box_local"
                )
                or self.config["joint_local_head"]["enabled"]
                or self.config["joint_local_head"][
                    "collect_diagnostics"
                ]
            ):
                try:
                    (
                        oriented_center,
                        oriented_dimensions,
                        oriented_basis,
                    ) = _oriented_box_frame(corners_input[index])
                    local_original_box = np.concatenate(
                        (
                            np.zeros(3, dtype=np.float64),
                            oriented_dimensions,
                        )
                    ).astype(np.float32)
                    local_candidate_corners = _points_to_box_local(
                        corners,
                        oriented_center,
                        oriented_basis,
                    )
                    local_candidate_box = corners_to_center_size(
                        local_candidate_corners[None, ...]
                    )[0]
                    local_basis = oriented_basis.astype(np.float32)
                    local_frame_valid = True
                except ValueError:
                    pass
            refit_original_boxes.append(original.copy())
            refit_original_corners.append(corners_input[index].copy())
            refit_applied.append(bool(np.any(boundary_delta != 0.0)))
            refit_reasons.append(refit_reason)
            refit_changed_axes.append(
                (
                    (boundary_delta[:3] != 0.0)
                    | (boundary_delta[3:] != 0.0)
                )
            )
            refit_boundary_delta.append(boundary_delta)
            refit_local_original_boxes.append(local_original_box)
            refit_local_candidate_boxes.append(local_candidate_box)
            refit_local_basis.append(local_basis)
            refit_local_frame_valid.append(local_frame_valid)

        output_filter_cfg = self.config["output_filter"]
        minimum_extent = float(output_filter_cfg["minimum_extent"])
        source_aware_extent = bool(
            self.config["supplemental_output"]["class_aware_extent"]
        )
        configured_final_extent = output_filter_cfg[
            "final_minimum_extent"
        ]
        reference_minimum_extent = (
            float(configured_final_extent)
            if source_aware_extent
            and configured_final_extent is not None
            else minimum_extent
        )
        eligible_global_boxes = np.asarray(
            final_boxes, dtype=np.float32
        )
        if (
            len(eligible_global_boxes)
            and reference_minimum_extent > 0.0
        ):
            eligible_mask = np.all(
                eligible_global_boxes[:, 3:6]
                >= reference_minimum_extent,
                axis=1,
            )
            eligible_global_boxes = eligible_global_boxes[eligible_mask]
        for supplemental in self._supplemental_outputs(
            eligible_global_boxes,
        ):
            box = supplemental.box
            final_corners.append(supplemental.corners)
            final_boxes.append(box)
            final_scores.append(supplemental.score)
            source_indices.append(-1)
            result_ids.append(supplemental.stable_id)
            labels.append(supplemental.label)
            feature_rows.append(supplemental.quality_features)
            memories.append(supplemental.memory)
            evidence_stats_rows.append(supplemental.stats)
            refit_original_boxes.append(supplemental.original_box.copy())
            refit_original_corners.append(
                supplemental.original_corners.copy()
            )
            refit_applied.append(supplemental.refit_applied)
            refit_reasons.append(supplemental.refit_reason)
            original_lower = (
                supplemental.original_box[:3]
                - 0.5 * supplemental.original_box[3:6]
            )
            original_upper = (
                supplemental.original_box[:3]
                + 0.5 * supplemental.original_box[3:6]
            )
            refined_lower = box[:3] - 0.5 * box[3:6]
            refined_upper = box[:3] + 0.5 * box[3:6]
            boundary_delta = np.concatenate(
                (
                    refined_lower - original_lower,
                    refined_upper - original_upper,
                )
            ).astype(np.float32)
            refit_boundary_delta.append(boundary_delta)
            refit_changed_axes.append(
                (
                    (boundary_delta[:3] != 0.0)
                    | (boundary_delta[3:] != 0.0)
                )
            )
            local_original_box = np.full(6, np.nan, dtype=np.float32)
            local_candidate_box = np.full(6, np.nan, dtype=np.float32)
            local_basis = np.full((3, 3), np.nan, dtype=np.float32)
            local_frame_valid = False
            try:
                center, dimensions, basis = _oriented_box_frame(
                    supplemental.original_corners
                )
                local_original_box = np.concatenate(
                    (
                        np.zeros(3, dtype=np.float32),
                        dimensions.astype(np.float32),
                    )
                )
                local_candidate_corners = _points_to_box_local(
                    supplemental.corners,
                    center,
                    basis,
                )
                local_candidate_box = corners_to_center_size(
                    local_candidate_corners[None, ...]
                )[0]
                local_basis = basis.astype(np.float32)
                local_frame_valid = True
            except ValueError:
                pass
            refit_local_original_boxes.append(local_original_box)
            refit_local_candidate_boxes.append(local_candidate_box)
            refit_local_basis.append(local_basis)
            refit_local_frame_valid.append(local_frame_valid)

        if final_boxes:
            boxes_array = np.asarray(final_boxes, dtype=np.float32)
            corners_array = np.asarray(final_corners, dtype=np.float32)
            scores_array = np.asarray(final_scores, dtype=np.float32)
            source_array = np.asarray(source_indices, dtype=np.int64)
            ids_array = np.asarray(result_ids, dtype=np.int64)
            features_array = np.asarray(feature_rows, dtype=np.float32)
            refit_original_array = np.asarray(
                refit_original_boxes, dtype=np.float32
            )
            refit_original_corners_array = np.asarray(
                refit_original_corners, dtype=np.float32
            )
            refit_applied_array = np.asarray(refit_applied, dtype=bool)
            refit_changed_axes_array = np.asarray(
                refit_changed_axes, dtype=bool
            )
            refit_boundary_delta_array = np.asarray(
                refit_boundary_delta, dtype=np.float32
            )
            refit_local_original_array = np.asarray(
                refit_local_original_boxes, dtype=np.float32
            )
            refit_local_candidate_array = np.asarray(
                refit_local_candidate_boxes, dtype=np.float32
            )
            refit_local_basis_array = np.asarray(
                refit_local_basis, dtype=np.float32
            )
            refit_local_frame_valid_array = np.asarray(
                refit_local_frame_valid, dtype=bool
            )
        else:
            boxes_array = np.empty((0, 6), dtype=np.float32)
            corners_array = np.empty((0, 8, 3), dtype=np.float32)
            scores_array = np.empty(0, dtype=np.float32)
            source_array = np.empty(0, dtype=np.int64)
            ids_array = np.empty(0, dtype=np.int64)
            features_array = np.empty(
                (0, QUALITY_FEATURE_DIM), dtype=np.float32
            )
            refit_original_array = np.empty((0, 6), dtype=np.float32)
            refit_original_corners_array = np.empty(
                (0, 8, 3), dtype=np.float32
            )
            refit_applied_array = np.empty(0, dtype=bool)
            refit_changed_axes_array = np.empty((0, 3), dtype=bool)
            refit_boundary_delta_array = np.empty(
                (0, 6), dtype=np.float32
            )
            refit_local_original_array = np.empty(
                (0, 6), dtype=np.float32
            )
            refit_local_candidate_array = np.empty(
                (0, 6), dtype=np.float32
            )
            refit_local_basis_array = np.empty(
                (0, 3, 3), dtype=np.float32
            )
            refit_local_frame_valid_array = np.empty(0, dtype=bool)

        apply_source_aware_filter = bool(
            source_aware_extent
            and configured_final_extent is not None
        )
        if len(boxes_array) and (
            minimum_extent > 0.0 or apply_source_aware_filter
        ):
            if apply_source_aware_filter:
                export_dimensions = (
                    np.max(corners_array, axis=1)
                    - np.min(corners_array, axis=1)
                )
                # C2 validates its authoritative float32 center/size box
                # before constructing corners.  Reconstructing a small
                # boundary-size box from large world coordinates can lose a
                # few ulps (for example 0.12 -> 0.119999885) and must not
                # delete an already verified C1 row.  Use the authoritative
                # dimensions for C2 rows only; all upstream and legacy
                # supplemental rows keep the original corner-span contract.
                c2_extent_dimensions = export_dimensions.copy()
                for row_index, reason in enumerate(refit_reasons):
                    if str(reason).startswith("c2_") and str(reason).endswith(
                        "_accepted"
                    ):
                        c2_extent_dimensions[row_index] = boxes_array[
                            row_index, 3:6
                        ]
                valid_output = np.asarray(
                    [
                        (
                            bool(
                                np.all(
                                    dimensions
                                    >= reference_minimum_extent
                                )
                            )
                            if int(source_index) >= 0
                            else (
                                bool(
                                    np.all(
                                        np.sort(dimensions) + 1e-6
                                        >= np.asarray(
                                            [
                                                self.config[
                                                    "supplemental_geometry_refiner"
                                                ][
                                                    "refined_planar_minimum_extent"
                                                ],
                                                self.config[
                                                    "supplemental_output"
                                                ][
                                                    "planar_middle_extent"
                                                ],
                                                self.config[
                                                    "supplemental_output"
                                                ][
                                                    "planar_max_extent"
                                                ],
                                            ],
                                            dtype=np.float64,
                                        )
                                    )
                                )
                                if reason == "c2_planar_accepted"
                                else supplemental_extent_is_valid(
                                    dimensions,
                                    label,
                                    self.config[
                                        "supplemental_output"
                                    ],
                                    default_minimum_extent=(
                                        reference_minimum_extent
                                    ),
                                )
                            )
                        )
                        for (
                            dimensions,
                            source_index,
                            label,
                            reason,
                        ) in zip(
                            c2_extent_dimensions,
                            source_array,
                            labels,
                            refit_reasons,
                        )
                    ],
                    dtype=bool,
                )
            else:
                valid_output = np.all(
                    boxes_array[:, 3:6] >= minimum_extent, axis=1
                )
            boxes_array = boxes_array[valid_output]
            corners_array = corners_array[valid_output]
            scores_array = scores_array[valid_output]
            source_array = source_array[valid_output]
            ids_array = ids_array[valid_output]
            features_array = features_array[valid_output]
            refit_original_array = refit_original_array[valid_output]
            refit_original_corners_array = (
                refit_original_corners_array[valid_output]
            )
            refit_applied_array = refit_applied_array[valid_output]
            refit_changed_axes_array = refit_changed_axes_array[
                valid_output
            ]
            refit_boundary_delta_array = refit_boundary_delta_array[
                valid_output
            ]
            refit_local_original_array = refit_local_original_array[
                valid_output
            ]
            refit_local_candidate_array = refit_local_candidate_array[
                valid_output
            ]
            refit_local_basis_array = refit_local_basis_array[
                valid_output
            ]
            refit_local_frame_valid_array = (
                refit_local_frame_valid_array[valid_output]
            )
            labels = [
                label
                for label, keep in zip(labels, valid_output)
                if bool(keep)
            ]
            memories = [
                memory
                for memory, keep in zip(memories, valid_output)
                if bool(keep)
            ]
            evidence_stats_rows = [
                stats
                for stats, keep in zip(
                    evidence_stats_rows, valid_output
                )
                if bool(keep)
            ]
            refit_reasons = [
                reason
                for reason, keep in zip(refit_reasons, valid_output)
                if bool(keep)
            ]

        soft_cfg = self.config["quality"]["soft_nms"]
        if (
            self.config["quality"]["enabled"]
            and soft_cfg["enabled"]
            and len(boxes_array)
        ):
            keep, decayed = soft_nms_aabb_3d(
                boxes_array,
                scores_array,
                method=soft_cfg["method"],
                iou_threshold=soft_cfg["iou_threshold"],
                sigma=soft_cfg["sigma"],
                score_threshold=soft_cfg["score_threshold"],
                max_detections=soft_cfg["max_detections"],
            )
            boxes_array = boxes_array[keep]
            corners_array = corners_array[keep]
            source_array = source_array[keep]
            ids_array = ids_array[keep]
            features_array = features_array[keep]
            scores_array = decayed
            labels = [labels[int(index)] for index in keep]
            memories = [memories[int(index)] for index in keep]
            evidence_stats_rows = [
                evidence_stats_rows[int(index)] for index in keep
            ]
            refit_original_array = refit_original_array[keep]
            refit_original_corners_array = (
                refit_original_corners_array[keep]
            )
            refit_applied_array = refit_applied_array[keep]
            refit_changed_axes_array = refit_changed_axes_array[keep]
            refit_boundary_delta_array = refit_boundary_delta_array[keep]
            refit_local_original_array = refit_local_original_array[keep]
            refit_local_candidate_array = refit_local_candidate_array[
                keep
            ]
            refit_local_basis_array = refit_local_basis_array[keep]
            refit_local_frame_valid_array = (
                refit_local_frame_valid_array[keep]
            )
            refit_reasons = [
                refit_reasons[int(index)] for index in keep
            ]

        if joint_enabled:
            expected_keep = (
                np.all(boxes[:, 3:6] >= minimum_extent, axis=1)
                if minimum_extent > 0.0
                else np.ones(len(boxes), dtype=bool)
            )
            expected_sources = np.flatnonzero(expected_keep).astype(
                np.int64
            )
            if (
                not np.array_equal(source_array, expected_sources)
                or not np.array_equal(ids_array, ids[expected_keep])
            ):
                raise RuntimeError(
                    "joint_local_head violated detection count/order/stable "
                    "ID conservation"
                )

        observer_output_arrays = {
            "corners": corners_array,
            "boxes": boxes_array,
            "scores": scores_array,
            "source_indices": source_array,
            "stable_ids": ids_array,
            "quality_features": features_array,
            "refit_original_boxes": refit_original_array,
            "refit_original_corners": refit_original_corners_array,
            "refit_applied": refit_applied_array,
            "refit_changed_axes": refit_changed_axes_array,
            "refit_boundary_delta": refit_boundary_delta_array,
            "refit_local_original_boxes": refit_local_original_array,
            "refit_local_candidate_boxes": refit_local_candidate_array,
            "refit_local_basis": refit_local_basis_array,
            "refit_local_frame_valid": refit_local_frame_valid_array,
        }
        c4_cfg = self.config["generic_local_geometry_refiner"]
        trifusion_cfg = self.config["trifusion_observer"]
        residual_cfg = self.config["residual_track_observer"]
        yidu_cfg = self.config["yidu_ablation"]
        fragment_cfg = self.config["fragment_stitch"]
        observer_zero_write_enabled = bool(
            (
                c4_cfg["enabled"]
                and c4_cfg["collect_diagnostics"]
                and not c4_cfg["mutate"]
            )
            or (
                trifusion_cfg["enabled"]
                and trifusion_cfg["collect_diagnostics"]
                and not trifusion_cfg["mutate"]
            )
            or (
                residual_cfg["enabled"]
                and residual_cfg["collect_diagnostics"]
                and residual_cfg["observer_only"]
                and not residual_cfg["mutate"]
            )
            or (
                yidu_cfg["enabled"]
                and yidu_cfg["collect_diagnostics"]
                and yidu_cfg["observer_only"]
                and not yidu_cfg["mutate"]
            )
            or fragment_cfg["enabled"]
        )
        observer_snapshot: Optional[Dict[str, np.ndarray]] = None
        if observer_zero_write_enabled:
            observer_snapshot = _observer_zero_write_snapshot(
                observer_output_arrays
            )
            self._last_observer_zero_write_audit.update(
                {
                    "enabled": True,
                    "pre_sha256": _observer_zero_write_sha256(
                        observer_snapshot
                    ),
                }
            )

        # C4 consumes only the arrays that survived the frozen B6 extent
        # filter.  It writes diagnostics into an isolated runtime table and
        # cannot feed candidate geometry back into B6 scoring or export.
        self._observe_generic_geometry_candidates(
            boxes=boxes_array,
            corners=corners_array,
            source_indices=source_array,
            stable_ids=ids_array,
        )
        self._observe_trifusion_candidates(
            boxes=boxes_array,
            corners=corners_array,
            source_indices=source_array,
            stable_ids=ids_array,
            quality_features=features_array,
        )
        self._observe_yidu_candidates(
            corners=corners_array,
            scores=scores_array,
            source_indices=source_array,
            stable_ids=ids_array,
            quality_features=features_array,
        )

        # C3 is finalized only after the complete fragment lifecycle is known.
        # It is observer-only, so this cannot alter any array assembled above.
        self._refresh_fragment_stitch_candidates()
        if observer_snapshot is not None:
            post_sha256 = _observer_zero_write_sha256(
                observer_output_arrays
            )
            changed_fields = _observer_zero_write_changed_fields(
                observer_snapshot,
                observer_output_arrays,
            )
            self._last_observer_zero_write_audit.update(
                {
                    "verified": not changed_fields,
                    "post_sha256": post_sha256,
                    "changed_fields": changed_fields,
                }
            )
            if changed_fields:
                raise RuntimeError(
                    "observer zero-write contract violated; formal output "
                    "array bytes changed: "
                    + ", ".join(changed_fields)
                )
        summary = self.summary()
        result = FinalRefinementResult(
            corners=corners_array,
            boxes=boxes_array,
            scores=scores_array,
            source_indices=source_array,
            stable_ids=ids_array,
            labels=tuple(labels),
            quality_features=features_array,
            refit_original_boxes=refit_original_array,
            refit_original_corners=refit_original_corners_array,
            refit_applied=refit_applied_array,
            refit_reasons=tuple(refit_reasons),
            refit_changed_axes=refit_changed_axes_array,
            refit_boundary_delta=refit_boundary_delta_array,
            refit_local_original_boxes=refit_local_original_array,
            refit_local_candidate_boxes=refit_local_candidate_array,
            refit_local_basis=refit_local_basis_array,
            refit_local_frame_valid=refit_local_frame_valid_array,
            summary=summary,
        )
        selected_scene = scene_id or self.scene_id
        if selected_scene is not None:
            self._dump_diagnostics(
                result,
                memories,
                evidence_stats_rows,
                selected_scene,
            )
        return result

    def _c4_diagnostic_payload(
        self,
        result: FinalRefinementResult,
        point_count: int,
    ) -> Dict[str, np.ndarray]:
        """Serialize the independent C4 observer without mixing B6 memory."""

        enabled = bool(
            self.config["generic_local_geometry_refiner"]["enabled"]
        )
        result_indices = np.asarray(
            [
                index
                for index, stable_id in enumerate(result.stable_ids)
                if enabled
                and int(stable_id) in self._last_c4_runtime
            ],
            dtype=np.int64,
        )
        rows = [
            self._last_c4_runtime[int(result.stable_ids[index])]
            for index in result_indices
        ]
        count = len(rows)

        def boolean(name: str) -> np.ndarray:
            return np.asarray(
                [bool(row.get(name, False)) for row in rows], dtype=bool
            )

        def integer(name: str) -> np.ndarray:
            return np.asarray(
                [int(row.get(name, 0)) for row in rows], dtype=np.int64
            )

        def floating(name: str) -> np.ndarray:
            return np.asarray(
                [float(row.get(name, np.nan)) for row in rows],
                dtype=np.float32,
            )

        def matrix(
            name: str,
            shape: Tuple[int, ...],
            *,
            dtype: Any,
            fill: Any,
        ) -> np.ndarray:
            output = np.full((count, *shape), fill, dtype=dtype)
            for index, row in enumerate(rows):
                value = np.asarray(row.get(name, output[index]), dtype=dtype)
                if value.shape == shape:
                    output[index] = value
            return output

        max_views = int(
            self.config["generic_local_geometry_refiner"][
                "secondary_object_memory"
            ].get("top_k_views", 5)
        )
        memory_points = np.zeros(
            (count, point_count, 3), dtype=np.float32
        )
        memory_point_mask = np.zeros(
            (count, point_count), dtype=bool
        )
        view_points = np.zeros(
            (count, max_views, point_count, 3), dtype=np.float32
        )
        view_point_mask = np.zeros(
            (count, max_views, point_count), dtype=bool
        )
        view_valid = np.zeros((count, max_views), dtype=bool)
        view_frame_ids = np.full(
            (count, max_views), -1, dtype=np.int64
        )
        view_quality = np.full(
            (count, max_views), np.nan, dtype=np.float32
        )
        view_valid_depth_ratio = np.full_like(
            view_quality, np.nan
        )
        view_projection_iou = np.full_like(view_quality, np.nan)
        view_camera_position = np.full(
            (count, max_views, 3), np.nan, dtype=np.float32
        )
        memory_valid = np.zeros(count, dtype=bool)
        for output_index, result_index in enumerate(result_indices):
            stable_id = int(result.stable_ids[int(result_index)])
            evidence = self.generic_geometry_global_tracks.get(stable_id)
            if evidence is None:
                continue
            memory_valid[output_index] = True
            sampled = deterministic_bounded_sample(
                evidence.memory.geometry_points, point_count
            )
            memory_points[output_index, : len(sampled)] = sampled
            memory_point_mask[output_index, : len(sampled)] = True
            for view_index, record in enumerate(
                evidence.memory.selected_view_records[:max_views]
            ):
                sampled_view = deterministic_bounded_sample(
                    record.points_world, point_count
                )
                view_points[
                    output_index, view_index, : len(sampled_view)
                ] = sampled_view
                view_point_mask[
                    output_index, view_index, : len(sampled_view)
                ] = True
                view_valid[output_index, view_index] = True
                view_frame_ids[
                    output_index, view_index
                ] = int(record.frame_id)
                view_quality[
                    output_index, view_index
                ] = float(record.quality)
                view_valid_depth_ratio[
                    output_index, view_index
                ] = float(record.valid_depth_ratio)
                view_projection_iou[
                    output_index, view_index
                ] = float(record.projection_mask_iou)
                if record.camera_position is not None:
                    view_camera_position[
                        output_index, view_index
                    ] = record.camera_position

        return {
            "c4_diagnostics_schema": np.asarray(
                "generic_mask_rgbd_local_geometry_v2"
            ),
            "c4_enabled": np.asarray(enabled, dtype=bool),
            "c4_mutation_enabled": np.asarray(False, dtype=bool),
            "c4_fail_open": np.asarray(
                bool(self.stats["c4_fail_open"]), dtype=bool
            ),
            "c4_error": np.asarray(self._last_c4_error),
            "c4_config_json": np.asarray(
                json.dumps(
                    self.config["generic_local_geometry_refiner"],
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
            ),
            "c4_result_indices": result_indices,
            "c4_stable_ids": result.stable_ids[result_indices],
            "c4_source_indices": result.source_indices[result_indices],
            "c4_scores": result.scores[result_indices],
            "c4_attempted": boolean("attempted"),
            "c4_proposed": boolean("proposed"),
            "c4_verified": boolean("verified"),
            "c4_applied": boolean("applied"),
            "c4_reason": np.asarray(
                [str(row.get("reason", "")) for row in rows],
                dtype=np.str_,
            ),
            "c4_source": np.asarray(
                [str(row.get("source", "")) for row in rows],
                dtype=np.str_,
            ),
            "c4_label": np.asarray(
                [str(row.get("label", "")) for row in rows],
                dtype=np.str_,
            ),
            "c4_normalized_label": np.asarray(
                [str(row.get("normalized_label", "")) for row in rows],
                dtype=np.str_,
            ),
            "c4_original_boxes": matrix(
                "original_box", (6,), dtype=np.float32, fill=np.nan
            ),
            "c4_candidate_boxes": matrix(
                "candidate_box", (6,), dtype=np.float32, fill=np.nan
            ),
            "c4_original_corners": matrix(
                "original_corners",
                (8, 3),
                dtype=np.float32,
                fill=np.nan,
            ),
            "c4_candidate_corners": matrix(
                "candidate_corners",
                (8, 3),
                dtype=np.float32,
                fill=np.nan,
            ),
            "c4_local_original_boxes": matrix(
                "local_original_box",
                (6,),
                dtype=np.float32,
                fill=np.nan,
            ),
            "c4_local_candidate_boxes": matrix(
                "local_candidate_box",
                (6,),
                dtype=np.float32,
                fill=np.nan,
            ),
            "c4_local_basis": matrix(
                "local_basis",
                (3, 3),
                dtype=np.float32,
                fill=np.nan,
            ),
            "c4_local_frame_valid": boolean("local_frame_valid"),
            "c4_eligible_view_count": integer(
                "eligible_view_count"
            ),
            "c4_selected_view_count": integer(
                "selected_view_count"
            ),
            "c4_input_point_count": integer("input_point_count"),
            "c4_cropped_point_count": integer(
                "cropped_point_count"
            ),
            "c4_consensus_point_count": integer(
                "consensus_point_count"
            ),
            "c4_component_count": integer("component_count"),
            "c4_eligible_component_count": integer(
                "eligible_component_count"
            ),
            "c4_anchor_point_count": integer("anchor_point_count"),
            "c4_merged_component_count": integer(
                "merged_component_count"
            ),
            "c4_component_view_count": integer(
                "component_view_count"
            ),
            "c4_component_inside_fraction": floating(
                "component_inside_fraction"
            ),
            "c4_mean_valid_depth_ratio": floating(
                "mean_valid_depth_ratio"
            ),
            "c4_clean_original_support": floating(
                "clean_original_support"
            ),
            "c4_clean_candidate_support": floating(
                "clean_candidate_support"
            ),
            "c4_clean_support_drop": floating(
                "clean_support_drop"
            ),
            "c4_raw_original_support": floating(
                "raw_original_support"
            ),
            "c4_raw_candidate_support": floating(
                "raw_candidate_support"
            ),
            "c4_raw_support_drop": floating("raw_support_drop"),
            "c4_original_projection": floating(
                "original_projection"
            ),
            "c4_candidate_projection": floating(
                "candidate_projection"
            ),
            "c4_projection_delta": floating("projection_delta"),
            "c4_projection_views": integer("projection_views"),
            "c4_center_shift_ratio": floating(
                "center_shift_ratio"
            ),
            "c4_center_shift_ratios": matrix(
                "center_shift_ratios",
                (3,),
                dtype=np.float32,
                fill=np.nan,
            ),
            "c4_extent_ratios": matrix(
                "extent_ratios",
                (3,),
                dtype=np.float32,
                fill=np.nan,
            ),
            "c4_original_candidate_iou": floating(
                "original_candidate_iou"
            ),
            "c4_original_max_other_iou": floating(
                "original_max_other_iou"
            ),
            "c4_candidate_max_other_iou": floating(
                "candidate_max_other_iou"
            ),
            "c4_overlap_increase": floating("overlap_increase"),
            "c4_boundary_values": matrix(
                "boundary_values",
                (3, 2),
                dtype=np.float32,
                fill=np.nan,
            ),
            "c4_boundary_view_counts": matrix(
                "boundary_view_counts",
                (3, 2),
                dtype=np.int64,
                fill=0,
            ),
            "c4_boundary_spreads": matrix(
                "boundary_spreads",
                (3, 2),
                dtype=np.float32,
                fill=np.nan,
            ),
            "c4_boundary_visible": matrix(
                "boundary_visible",
                (3, 2),
                dtype=bool,
                fill=False,
            ),
            "c4_memory_valid": memory_valid,
            "c4_memory_points": memory_points,
            "c4_memory_point_mask": memory_point_mask,
            "c4_view_points": view_points,
            "c4_view_point_mask": view_point_mask,
            "c4_view_valid": view_valid,
            "c4_view_frame_ids": view_frame_ids,
            "c4_view_quality": view_quality,
            "c4_view_valid_depth_ratio": view_valid_depth_ratio,
            "c4_view_projection_iou": view_projection_iou,
            "c4_view_camera_position": view_camera_position,
        }

    def _trifusion_diagnostic_payload(
        self,
        result: FinalRefinementResult,
    ) -> Dict[str, np.ndarray]:
        """Serialize the strict occupancy/MSR observer contract."""

        enabled = bool(self.config["trifusion_observer"]["enabled"])
        result_indices = np.asarray(
            [
                index
                for index, stable_id in enumerate(result.stable_ids)
                if enabled
                and int(stable_id) in self._last_trifusion_runtime
            ],
            dtype=np.int64,
        )
        rows = [
            self._last_trifusion_runtime[int(result.stable_ids[index])]
            for index in result_indices
        ]
        count = len(rows)

        def matrix(
            name: str,
            shape: Tuple[int, ...],
            *,
            fill: float,
        ) -> np.ndarray:
            output = np.full(
                (count, *shape), fill, dtype=np.float32
            )
            for index, row in enumerate(rows):
                value = np.asarray(
                    row.get(name, output[index]), dtype=np.float32
                )
                if value.shape == shape:
                    output[index] = value
            return output

        return {
            "trifusion_diagnostics_schema": np.asarray(
                "boxfusion.trifusion.occupancy_msr_observer.v1"
            ),
            "trifusion_enabled": np.asarray(enabled, dtype=bool),
            "trifusion_mutation_enabled": np.asarray(False, dtype=bool),
            "trifusion_config_json": np.asarray(
                json.dumps(
                    self.config["trifusion_observer"],
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
            ),
            "trifusion_result_indices": result_indices,
            "trifusion_stable_ids": result.stable_ids[result_indices],
            "trifusion_feature_names": np.asarray(
                OCCUPANCY_MSR_FEATURE_NAMES, dtype=np.str_
            ),
            "trifusion_original_corners": matrix(
                "original_corners", (8, 3), fill=np.nan
            ),
            "trifusion_candidate_corners": matrix(
                "candidate_corners", (8, 3), fill=np.nan
            ),
            "trifusion_features": matrix(
                "features",
                (OCCUPANCY_MSR_FEATURE_DIM,),
                fill=0.0,
            ),
            "trifusion_gate_diagnostics_schema": np.asarray(
                "boxfusion.trifusion.ap50_safety_observer.v1"
            ),
            "trifusion_gate_enabled": np.asarray(
                self.trifusion_ap50_gate is not None, dtype=bool
            ),
            "trifusion_gate_mutation_enabled": np.asarray(
                False, dtype=bool
            ),
            "trifusion_gate_feature_names": np.asarray(
                TRIFUSION_GATE_FEATURE_NAMES, dtype=np.str_
            ),
            "trifusion_gate_features": matrix(
                "gate_features",
                (TRIFUSION_GATE_FEATURE_DIM,),
                fill=0.0,
            ),
            "trifusion_gate_evaluated": np.asarray(
                [
                    bool(row.get("gate_evaluated", False))
                    for row in rows
                ],
                dtype=bool,
            ),
            "trifusion_gate_accepted": np.asarray(
                [
                    bool(row.get("gate_accepted", False))
                    for row in rows
                ],
                dtype=bool,
            ),
            "trifusion_gate_reason": np.asarray(
                [
                    str(row.get("gate_reason", "disabled"))
                    for row in rows
                ],
                dtype=np.str_,
            ),
            "trifusion_gate_lower_confidence_delta": np.asarray(
                [
                    float(
                        row.get(
                            "gate_lower_confidence_delta", np.nan
                        )
                    )
                    for row in rows
                ],
                dtype=np.float32,
            ),
            "trifusion_gate_delta_mean": np.asarray(
                [
                    float(row.get("gate_delta_mean", np.nan))
                    for row in rows
                ],
                dtype=np.float32,
            ),
            "trifusion_gate_delta_std": np.asarray(
                [
                    float(row.get("gate_delta_std", np.nan))
                    for row in rows
                ],
                dtype=np.float32,
            ),
            "trifusion_gate_improvement_probability": np.asarray(
                [
                    float(
                        row.get(
                            "gate_improvement_probability", np.nan
                        )
                    )
                    for row in rows
                ],
                dtype=np.float32,
            ),
            "trifusion_gate_harm_probability": np.asarray(
                [
                    float(
                        row.get("gate_harm_probability", np.nan)
                    )
                    for row in rows
                ],
                dtype=np.float32,
            ),
            "trifusion_gate_original_iou": np.asarray(
                [
                    float(row.get("gate_original_iou", np.nan))
                    for row in rows
                ],
                dtype=np.float32,
            ),
            "trifusion_gate_candidate_iou": np.asarray(
                [
                    float(row.get("gate_candidate_iou", np.nan))
                    for row in rows
                ],
                dtype=np.float32,
            ),
            "trifusion_gate_cross_iou25_probability": np.asarray(
                [
                    float(
                        row.get(
                            "gate_cross_iou25_probability", np.nan
                        )
                    )
                    for row in rows
                ],
                dtype=np.float32,
            ),
            "trifusion_gate_cross_iou50_probability": np.asarray(
                [
                    float(
                        row.get(
                            "gate_cross_iou50_probability", np.nan
                        )
                    )
                    for row in rows
                ],
                dtype=np.float32,
            ),
            "trifusion_candidate_valid": np.asarray(
                [
                    bool(row.get("candidate_valid", False))
                    for row in rows
                ],
                dtype=bool,
            ),
            "trifusion_is_candidate": np.asarray(
                [
                    bool(row.get("is_candidate", False))
                    for row in rows
                ],
                dtype=bool,
            ),
            "trifusion_candidate_verified": np.asarray(
                [
                    bool(row.get("candidate_verified", False))
                    for row in rows
                ],
                dtype=bool,
            ),
            "trifusion_applied": np.zeros(count, dtype=bool),
            "trifusion_reason": np.asarray(
                [str(row.get("reason", "")) for row in rows],
                dtype=np.str_,
            ),
            "trifusion_source": np.asarray(
                [str(row.get("source", "")) for row in rows],
                dtype=np.str_,
            ),
        }

    def _yidu_diagnostic_payload(
        self,
        result: FinalRefinementResult,
    ) -> Dict[str, np.ndarray]:
        """Serialize one strict incremental YiDu observer stage."""

        cfg = self.config["yidu_ablation"]
        enabled = bool(cfg["enabled"])
        result_indices = np.asarray(
            [
                index
                for index, stable_id in enumerate(result.stable_ids)
                if enabled
                and int(stable_id) in self._last_yidu_runtime
            ],
            dtype=np.int64,
        )
        rows = [
            self._last_yidu_runtime[int(result.stable_ids[index])]
            for index in result_indices
        ]
        count = len(rows)

        def matrix(
            name: str,
            shape: Tuple[int, ...],
            *,
            fill: float,
        ) -> np.ndarray:
            output = np.full(
                (count, *shape), fill, dtype=np.float32
            )
            for index, row in enumerate(rows):
                value = np.asarray(
                    row.get(name, output[index]), dtype=np.float32
                )
                if value.shape == shape:
                    output[index] = value
            return output

        def integer(name: str, fill: int = 0) -> np.ndarray:
            return np.asarray(
                [int(row.get(name, fill)) for row in rows],
                dtype=np.int64,
            )

        def floating(name: str) -> np.ndarray:
            return np.asarray(
                [float(row.get(name, np.nan)) for row in rows],
                dtype=np.float32,
            )

        return {
            "yidu_diagnostics_schema": np.asarray(
                YIDU_LOCAL_OBSERVER_SCHEMA
            ),
            "yidu_ablation_schema": np.asarray(YIDU_SCHEMA),
            "yidu_stage": np.asarray(str(cfg["stage"])),
            "yidu_profile": np.asarray(str(cfg["profile"])),
            "yidu_enabled": np.asarray(enabled, dtype=bool),
            "yidu_mutation_enabled": np.asarray(False, dtype=bool),
            "yidu_applied_count": np.asarray(0, dtype=np.int64),
            "yidu_zero_write_check_enabled": np.asarray(
                bool(self._last_observer_zero_write_audit["enabled"]),
                dtype=bool,
            ),
            "yidu_zero_write_verified": np.asarray(
                bool(self._last_observer_zero_write_audit["verified"]),
                dtype=bool,
            ),
            "yidu_zero_write_pre_sha256": np.asarray(
                str(
                    self._last_observer_zero_write_audit[
                        "pre_sha256"
                    ]
                )
            ),
            "yidu_zero_write_post_sha256": np.asarray(
                str(
                    self._last_observer_zero_write_audit[
                        "post_sha256"
                    ]
                )
            ),
            "yidu_zero_write_array_names": np.asarray(
                self._last_observer_zero_write_audit["array_names"],
                dtype=np.str_,
            ),
            "yidu_zero_write_changed_fields": np.asarray(
                self._last_observer_zero_write_audit["changed_fields"],
                dtype=np.str_,
            ),
            "yidu_modules_json": np.asarray(
                json.dumps(
                    cfg["modules"],
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            "yidu_config_json": np.asarray(
                json.dumps(
                    cfg,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
            ),
            "yidu_result_indices": result_indices,
            "yidu_stable_ids": result.stable_ids[result_indices],
            "yidu_attempted": np.asarray(
                [bool(row.get("attempted", False)) for row in rows],
                dtype=bool,
            ),
            "yidu_valid": np.asarray(
                [bool(row.get("valid", False)) for row in rows],
                dtype=bool,
            ),
            "yidu_applied": np.zeros(count, dtype=bool),
            "yidu_reason": np.asarray(
                [str(row.get("reason", "")) for row in rows],
                dtype=np.str_,
            ),
            "yidu_selected_source": np.asarray(
                [
                    str(row.get("selected_source", "original"))
                    for row in rows
                ],
                dtype=np.str_,
            ),
            "yidu_original_corners": matrix(
                "original_corners", (8, 3), fill=np.nan
            ),
            "yidu_raw_candidate_corners": matrix(
                "raw_candidate_corners", (8, 3), fill=np.nan
            ),
            "yidu_superpoint_candidate_corners": matrix(
                "superpoint_candidate_corners", (8, 3), fill=np.nan
            ),
            "yidu_occupancy_candidate_corners": matrix(
                "occupancy_candidate_corners", (8, 3), fill=np.nan
            ),
            "yidu_selected_candidate_corners": matrix(
                "selected_candidate_corners", (8, 3), fill=np.nan
            ),
            "yidu_input_point_count": integer("input_point_count"),
            "yidu_cropped_point_count": integer(
                "cropped_point_count"
            ),
            "yidu_selected_view_count": integer(
                "selected_view_count"
            ),
            "yidu_component_count": integer("component_count"),
            "yidu_selected_component_id": integer(
                "selected_component_id", -1
            ),
            "yidu_component_feature_names": np.asarray(
                YIDU_COMPONENT_FEATURE_NAMES, dtype=np.str_
            ),
            "yidu_component_features": matrix(
                "component_features",
                (YIDU_COMPONENT_FEATURE_DIM,),
                fill=0.0,
            ),
            "yidu_occupancy_feature_names": np.asarray(
                OCCUPANCY_MSR_FEATURE_NAMES, dtype=np.str_
            ),
            "yidu_occupancy_features": matrix(
                "occupancy_features",
                (OCCUPANCY_MSR_FEATURE_DIM,),
                fill=0.0,
            ),
            "yidu_raw_fused_features": matrix(
                "raw_fused_features",
                (RAW_FUSED_QUERY_FEATURE_DIM,),
                fill=0.0,
            ),
            "yidu_raw_fused_feature_names": np.asarray(
                RAW_FUSED_QUERY_FEATURE_NAMES, dtype=np.str_
            ),
            "yidu_gate_feature_names": np.asarray(
                YIDU_GATE_FEATURE_NAMES, dtype=np.str_
            ),
            "yidu_gate_features": matrix(
                "gate_features",
                (YIDU_GATE_FEATURE_DIM,),
                fill=0.0,
            ),
            "yidu_gate_evaluated": np.asarray(
                [
                    bool(row.get("gate_evaluated", False))
                    for row in rows
                ],
                dtype=bool,
            ),
            "yidu_gate_accepted": np.asarray(
                [
                    bool(row.get("gate_accepted", False))
                    for row in rows
                ],
                dtype=bool,
            ),
            "yidu_gate_reason": np.asarray(
                [
                    str(row.get("gate_reason", "disabled"))
                    for row in rows
                ],
                dtype=np.str_,
            ),
            "yidu_gate_delta_mean": floating("gate_delta_mean"),
            "yidu_gate_delta_std": floating("gate_delta_std"),
            "yidu_gate_improvement_probability": floating(
                "gate_improvement_probability"
            ),
            "yidu_gate_harm_probability": floating(
                "gate_harm_probability"
            ),
            "yidu_gate_original_iou": floating(
                "gate_original_iou"
            ),
            "yidu_gate_candidate_iou": floating(
                "gate_candidate_iou"
            ),
            "yidu_gate_cross_iou25_probability": floating(
                "gate_cross_iou25_probability"
            ),
            "yidu_gate_cross_iou50_probability": floating(
                "gate_cross_iou50_probability"
            ),
        }

    def _trifusion_missing_diagnostic_payload(
        self,
    ) -> Dict[str, np.ndarray]:
        """Serialize M1/M2 confirmed candidates without object arrays."""

        enabled = self.trifusion_missing_graph is not None
        candidates = (
            self._last_trifusion_missing_candidates if enabled else ()
        )
        count = len(candidates)

        def stacked(
            values: Sequence[np.ndarray],
            shape: Tuple[int, ...],
        ) -> np.ndarray:
            if not values:
                return np.empty((0, *shape), dtype=np.float32)
            output = np.stack(
                [np.asarray(value, dtype=np.float32) for value in values]
            )
            if output.shape != (count, *shape):
                raise ValueError(
                    "TriFusion missing diagnostic shape mismatch"
                )
            return output

        frame_ids_json = [
            json.dumps(
                list(candidate.frame_ids),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for candidate in candidates
        ]
        audit_reasons_json = [
            json.dumps(
                list(candidate.audit_reasons),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for candidate in candidates
        ]
        last_errors = (
            ()
            if self._last_trifusion_missing_update is None
            else tuple(self._last_trifusion_missing_update.errors)
        )
        return {
            "trifusion_missing_diagnostics_schema": np.asarray(
                "boxfusion.trifusion.missing_graph_observer.v1"
            ),
            "trifusion_missing_graph_schema": np.asarray(
                MISSING_INSTANCE_GRAPH_SCHEMA
            ),
            "trifusion_missing_enabled": np.asarray(
                enabled, dtype=bool
            ),
            "trifusion_missing_mutation_enabled": np.asarray(
                False, dtype=bool
            ),
            "trifusion_missing_candidate_ids": np.asarray(
                [candidate.candidate_id for candidate in candidates],
                dtype=np.int64,
            ),
            "trifusion_missing_sources": np.asarray(
                ["missing_graph"] * count, dtype=np.str_
            ),
            "trifusion_missing_corners": stacked(
                [candidate.corners for candidate in candidates],
                (8, 3),
            ),
            "trifusion_missing_oriented_boxes": stacked(
                [candidate.oriented_box for candidate in candidates],
                (7,),
            ),
            "trifusion_missing_aabbs": stacked(
                [candidate.aabb for candidate in candidates],
                (6,),
            ),
            "trifusion_missing_scores": np.asarray(
                [candidate.score for candidate in candidates],
                dtype=np.float32,
            ),
            "trifusion_missing_labels": np.asarray(
                [
                    ""
                    if candidate.label is None
                    else str(candidate.label)
                    for candidate in candidates
                ],
                dtype=np.str_,
            ),
            "trifusion_missing_lifecycle_states": np.asarray(
                [
                    candidate.lifecycle_state
                    for candidate in candidates
                ],
                dtype=np.str_,
            ),
            "trifusion_missing_feature_names": np.asarray(
                MISSING_GRAPH_FEATURE_NAMES, dtype=np.str_
            ),
            "trifusion_missing_features": stacked(
                [
                    candidate.feature_vector
                    for candidate in candidates
                ],
                (len(MISSING_GRAPH_FEATURE_NAMES),),
            ),
            "trifusion_missing_valid": np.ones(count, dtype=bool),
            "trifusion_missing_verified": np.ones(count, dtype=bool),
            "trifusion_missing_confirmed": np.ones(count, dtype=bool),
            "trifusion_missing_applied": np.zeros(count, dtype=bool),
            "trifusion_missing_reasons": np.asarray(
                [candidate.reason for candidate in candidates],
                dtype=np.str_,
            ),
            "trifusion_missing_unique_views": np.asarray(
                [len(candidate.frame_ids) for candidate in candidates],
                dtype=np.int64,
            ),
            "trifusion_missing_node_counts": np.asarray(
                [candidate.node_count for candidate in candidates],
                dtype=np.int64,
            ),
            "trifusion_missing_edge_counts": np.asarray(
                [candidate.edge_count for candidate in candidates],
                dtype=np.int64,
            ),
            "trifusion_missing_point_counts": np.asarray(
                [candidate.point_count for candidate in candidates],
                dtype=np.int64,
            ),
            "trifusion_missing_provider_call_first": np.asarray(
                [
                    candidate.provider_call_first
                    for candidate in candidates
                ],
                dtype=np.int64,
            ),
            "trifusion_missing_provider_call_last": np.asarray(
                [
                    candidate.provider_call_last
                    for candidate in candidates
                ],
                dtype=np.int64,
            ),
            "trifusion_missing_frame_ids_json": np.asarray(
                frame_ids_json, dtype=np.str_
            ),
            "trifusion_missing_audit_reasons_json": np.asarray(
                audit_reasons_json, dtype=np.str_
            ),
            "trifusion_missing_last_errors": np.asarray(
                [str(error) for error in last_errors], dtype=np.str_
            ),
        }

    def _residual_track_audit_metrics(self) -> Dict[str, Any]:
        """Summarize cumulative, last-call, and final graph diagnostics."""

        observation_reasons = dict(
            sorted(
                self.stats[
                    "residual_track_observation_reasons"
                ].items()
            )
        )
        association_reasons = dict(
            sorted(
                self.stats[
                    "residual_track_association_reasons"
                ].items()
            )
        )
        last_update = self._last_residual_track_update
        last_observation_reasons = Counter()
        last_association_reasons = Counter()
        if last_update is not None:
            last_observation_reasons.update(
                str(audit.reason)
                for audit in last_update.observations
            )
            last_association_reasons.update(
                str(audit.reason)
                for audit in last_update.associations
            )
        final_decision_reasons = dict(
            sorted(
                Counter(
                    str(decision.reason)
                    for decision in self._last_residual_track_decisions
                ).items()
            )
        )
        return {
            "observation_reasons": observation_reasons,
            "association_reasons": association_reasons,
            "associations_evaluated": int(
                sum(association_reasons.values())
            ),
            "associations_accepted": int(
                association_reasons.get("accepted", 0)
            ),
            # An accepted pair can lose the deterministic one-to-one
            # assignment.  "associated" observations are the selected edges.
            "associations_selected": int(
                observation_reasons.get("associated", 0)
            ),
            "tracks_seeded": int(
                observation_reasons.get("seeded", 0)
            ),
            "last_provider_call_index": (
                -1
                if last_update is None
                else int(last_update.provider_call_index)
            ),
            "last_observation_reasons": dict(
                sorted(last_observation_reasons.items())
            ),
            "last_association_reasons": dict(
                sorted(last_association_reasons.items())
            ),
            "last_associations_evaluated": (
                0
                if last_update is None
                else len(last_update.associations)
            ),
            "last_associations_accepted": (
                0
                if last_update is None
                else sum(
                    int(audit.accepted)
                    for audit in last_update.associations
                )
            ),
            "last_associations_selected": int(
                last_observation_reasons.get("associated", 0)
            ),
            "last_tracks_seeded": int(
                last_observation_reasons.get("seeded", 0)
            ),
            "final_decision_reasons": final_decision_reasons,
            "final_decisions": len(
                self._last_residual_track_decisions
            ),
            "final_confirmed": sum(
                int(decision.confirmed)
                for decision in self._last_residual_track_decisions
            ),
            "final_accepted": sum(
                int(decision.accepted)
                for decision in self._last_residual_track_decisions
            ),
        }

    def _residual_track_diagnostic_payload(
        self,
    ) -> Dict[str, np.ndarray]:
        """Serialize frozen-B6 residual tracks with provider provenance."""

        cfg = self.config["residual_track_observer"]
        enabled = self.residual_track_graph is not None
        candidates = (
            self._last_residual_track_candidates if enabled else ()
        )
        count = len(candidates)
        decisions = (
            self._last_residual_track_decisions if enabled else ()
        )
        audit_metrics = self._residual_track_audit_metrics()

        def stacked(
            values: Sequence[np.ndarray],
            shape: Tuple[int, ...],
        ) -> np.ndarray:
            if not values:
                return np.empty((0, *shape), dtype=np.float32)
            output = np.stack(
                [np.asarray(value, dtype=np.float32) for value in values]
            )
            if output.shape != (count, *shape):
                raise ValueError(
                    "residual-track diagnostic shape mismatch"
                )
            return output

        primary_name = str(cfg["primary_provider_name"])
        secondary_name = str(cfg["secondary_provider_name"])
        provider_counts = []
        primary_nodes = []
        secondary_nodes = []
        unknown_nodes = []
        mixed_provider = []
        for candidate in candidates:
            counts = Counter(
                self._residual_track_provider_counts.get(
                    int(candidate.track_id), Counter()
                )
            )
            known_nodes = sum(int(value) for value in counts.values())
            if known_nodes > int(candidate.node_count):
                raise RuntimeError(
                    "residual provider provenance exceeds candidate nodes"
                )
            if known_nodes < int(candidate.node_count):
                counts["unknown"] += int(candidate.node_count) - known_nodes
            normalized_counts = {
                str(key): int(value)
                for key, value in sorted(counts.items())
                if int(value) > 0
            }
            provider_counts.append(
                json.dumps(
                    normalized_counts,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            primary_nodes.append(normalized_counts.get(primary_name, 0))
            secondary_nodes.append(
                normalized_counts.get(secondary_name, 0)
            )
            unknown_nodes.append(normalized_counts.get("unknown", 0))
            mixed_provider.append(
                sum(value > 0 for value in normalized_counts.values()) > 1
            )

        last_errors = (
            ()
            if self._last_residual_track_update is None
            else tuple(self._last_residual_track_update.errors)
        )
        final_decisions_json = json.dumps(
            [
                {
                    "accepted": bool(decision.accepted),
                    "confirmed": bool(decision.confirmed),
                    "duplicate_of_track_id": (
                        None
                        if decision.duplicate_of_track_id is None
                        else int(decision.duplicate_of_track_id)
                    ),
                    "lifecycle_state": str(
                        decision.lifecycle_state
                    ),
                    "maximum_candidate_containment": float(
                        decision.maximum_candidate_containment
                    ),
                    "maximum_candidate_iou": float(
                        decision.maximum_candidate_iou
                    ),
                    "maximum_global_containment": float(
                        decision.maximum_global_containment
                    ),
                    "maximum_global_iou": float(
                        decision.maximum_global_iou
                    ),
                    "reason": str(decision.reason),
                    "track_id": int(decision.track_id),
                    "unique_views": int(decision.unique_views),
                }
                for decision in decisions
            ],
            sort_keys=True,
            separators=(",", ":"),
        )
        return {
            "residual_track_diagnostics_schema": np.asarray(
                "boxfusion.residual_mask_track_observer.v1"
            ),
            "residual_track_graph_schema": np.asarray(
                MISSING_INSTANCE_GRAPH_SCHEMA
            ),
            "residual_track_enabled": np.asarray(enabled, dtype=bool),
            "residual_track_observer_only": np.asarray(True, dtype=bool),
            "residual_track_mutation_enabled": np.asarray(
                False, dtype=bool
            ),
            "residual_track_source_mode": np.asarray(
                str(cfg["source_mode"])
            ),
            "residual_track_failed": np.asarray(
                self._residual_track_failed, dtype=bool
            ),
            "residual_track_failure": np.asarray(
                self._last_residual_track_error
            ),
            "residual_track_config_json": np.asarray(
                json.dumps(
                    cfg,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
            ),
            "residual_track_candidate_ids": np.asarray(
                [candidate.candidate_id for candidate in candidates],
                dtype=np.int64,
            ),
            "residual_track_sources": np.asarray(
                ["residual_mask_track"] * count, dtype=np.str_
            ),
            "residual_track_corners": stacked(
                [candidate.corners for candidate in candidates], (8, 3)
            ),
            "residual_track_oriented_boxes": stacked(
                [candidate.oriented_box for candidate in candidates], (7,)
            ),
            "residual_track_aabbs": stacked(
                [candidate.aabb for candidate in candidates], (6,)
            ),
            "residual_track_scores": np.asarray(
                [candidate.score for candidate in candidates],
                dtype=np.float32,
            ),
            "residual_track_labels": np.asarray(
                [
                    "" if candidate.label is None else str(candidate.label)
                    for candidate in candidates
                ],
                dtype=np.str_,
            ),
            "residual_track_lifecycle_states": np.asarray(
                [candidate.lifecycle_state for candidate in candidates],
                dtype=np.str_,
            ),
            "residual_track_feature_names": np.asarray(
                MISSING_GRAPH_FEATURE_NAMES, dtype=np.str_
            ),
            "residual_track_features": stacked(
                [candidate.feature_vector for candidate in candidates],
                (len(MISSING_GRAPH_FEATURE_NAMES),),
            ),
            # These fields mean that the graph construction contract passed;
            # they are not claims about ground-truth IoU or AP50 quality.
            "residual_track_graph_contract_valid": np.ones(
                count, dtype=bool
            ),
            "residual_track_graph_confirmed": np.ones(
                count, dtype=bool
            ),
            "residual_track_applied": np.zeros(count, dtype=bool),
            "residual_track_reasons": np.asarray(
                [candidate.reason for candidate in candidates],
                dtype=np.str_,
            ),
            "residual_track_unique_views": np.asarray(
                [len(candidate.frame_ids) for candidate in candidates],
                dtype=np.int64,
            ),
            "residual_track_node_counts": np.asarray(
                [candidate.node_count for candidate in candidates],
                dtype=np.int64,
            ),
            "residual_track_edge_counts": np.asarray(
                [candidate.edge_count for candidate in candidates],
                dtype=np.int64,
            ),
            "residual_track_point_counts": np.asarray(
                [candidate.point_count for candidate in candidates],
                dtype=np.int64,
            ),
            "residual_track_provider_call_first": np.asarray(
                [
                    candidate.provider_call_first
                    for candidate in candidates
                ],
                dtype=np.int64,
            ),
            "residual_track_provider_call_last": np.asarray(
                [
                    candidate.provider_call_last
                    for candidate in candidates
                ],
                dtype=np.int64,
            ),
            "residual_track_frame_ids_json": np.asarray(
                [
                    json.dumps(
                        list(candidate.frame_ids),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    for candidate in candidates
                ],
                dtype=np.str_,
            ),
            "residual_track_audit_reasons_json": np.asarray(
                [
                    json.dumps(
                        list(candidate.audit_reasons),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    for candidate in candidates
                ],
                dtype=np.str_,
            ),
            "residual_track_provider_counts_json": np.asarray(
                provider_counts, dtype=np.str_
            ),
            "residual_track_primary_nodes": np.asarray(
                primary_nodes, dtype=np.int64
            ),
            "residual_track_secondary_nodes": np.asarray(
                secondary_nodes, dtype=np.int64
            ),
            "residual_track_unknown_nodes": np.asarray(
                unknown_nodes, dtype=np.int64
            ),
            "residual_track_mixed_provider": np.asarray(
                mixed_provider, dtype=bool
            ),
            "residual_track_final_decision_reasons_json": np.asarray(
                json.dumps(
                    audit_metrics["final_decision_reasons"],
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            "residual_track_final_decisions_json": np.asarray(
                final_decisions_json
            ),
            "residual_track_final_decision_track_ids": np.asarray(
                [decision.track_id for decision in decisions],
                dtype=np.int64,
            ),
            "residual_track_final_decision_lifecycle_states": np.asarray(
                [
                    decision.lifecycle_state
                    for decision in decisions
                ],
                dtype=np.str_,
            ),
            "residual_track_final_decision_confirmed": np.asarray(
                [decision.confirmed for decision in decisions],
                dtype=bool,
            ),
            "residual_track_final_decision_unique_views": np.asarray(
                [decision.unique_views for decision in decisions],
                dtype=np.int64,
            ),
            "residual_track_final_decision_accepted": np.asarray(
                [decision.accepted for decision in decisions],
                dtype=bool,
            ),
            "residual_track_final_decision_reasons": np.asarray(
                [decision.reason for decision in decisions],
                dtype=np.str_,
            ),
            "residual_track_final_decision_duplicate_of_track_ids": (
                np.asarray(
                    [
                        (
                            -1
                            if decision.duplicate_of_track_id is None
                            else decision.duplicate_of_track_id
                        )
                        for decision in decisions
                    ],
                    dtype=np.int64,
                )
            ),
            "residual_track_final_decision_maximum_global_iou": (
                np.asarray(
                    [
                        decision.maximum_global_iou
                        for decision in decisions
                    ],
                    dtype=np.float32,
                )
            ),
            "residual_track_final_decision_maximum_global_containment": (
                np.asarray(
                    [
                        decision.maximum_global_containment
                        for decision in decisions
                    ],
                    dtype=np.float32,
                )
            ),
            "residual_track_final_decision_maximum_candidate_iou": (
                np.asarray(
                    [
                        decision.maximum_candidate_iou
                        for decision in decisions
                    ],
                    dtype=np.float32,
                )
            ),
            "residual_track_final_decision_maximum_candidate_containment": (
                np.asarray(
                    [
                        decision.maximum_candidate_containment
                        for decision in decisions
                    ],
                    dtype=np.float32,
                )
            ),
            "residual_track_observation_reasons_json": np.asarray(
                json.dumps(
                    audit_metrics["observation_reasons"],
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            "residual_track_association_reasons_json": np.asarray(
                json.dumps(
                    audit_metrics["association_reasons"],
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            "residual_track_associations_evaluated": np.asarray(
                audit_metrics["associations_evaluated"],
                dtype=np.int64,
            ),
            "residual_track_associations_accepted": np.asarray(
                audit_metrics["associations_accepted"],
                dtype=np.int64,
            ),
            "residual_track_associations_selected": np.asarray(
                audit_metrics["associations_selected"],
                dtype=np.int64,
            ),
            "residual_track_tracks_seeded": np.asarray(
                audit_metrics["tracks_seeded"], dtype=np.int64
            ),
            "residual_track_last_provider_call_index": np.asarray(
                audit_metrics["last_provider_call_index"],
                dtype=np.int64,
            ),
            "residual_track_last_observation_reasons_json": np.asarray(
                json.dumps(
                    audit_metrics["last_observation_reasons"],
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            "residual_track_last_association_reasons_json": np.asarray(
                json.dumps(
                    audit_metrics["last_association_reasons"],
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            "residual_track_last_associations_evaluated": np.asarray(
                audit_metrics["last_associations_evaluated"],
                dtype=np.int64,
            ),
            "residual_track_last_associations_accepted": np.asarray(
                audit_metrics["last_associations_accepted"],
                dtype=np.int64,
            ),
            "residual_track_last_associations_selected": np.asarray(
                audit_metrics["last_associations_selected"],
                dtype=np.int64,
            ),
            "residual_track_last_tracks_seeded": np.asarray(
                audit_metrics["last_tracks_seeded"], dtype=np.int64
            ),
            "residual_track_last_errors": np.asarray(
                [str(error) for error in last_errors], dtype=np.str_
            ),
        }

    def _dump_diagnostics(
        self,
        result: FinalRefinementResult,
        memories: Sequence[Optional[ObjectGeometryMemory]],
        evidence_stats_rows: Sequence[Optional[EvidenceStats]],
        scene_id: str,
    ) -> None:
        cfg = self.config["diagnostics"]
        if not cfg["enabled"] or not cfg["dump_track_memory"]:
            return
        if (
            len(memories) != len(result.boxes)
            or len(evidence_stats_rows) != len(result.boxes)
        ):
            raise RuntimeError(
                "diagnostic memory/evidence rows must align with results"
            )
        root = Path(cfg["root"])
        root.mkdir(parents=True, exist_ok=True)
        point_count = int(cfg["point_count"])
        graph_snapshots = self._diagnostic_mask_graph_snapshots()
        graph_component_track_ids = np.asarray(
            [row["track_id"] for row in graph_snapshots],
            dtype=np.int64,
        )
        graph_component_states = np.asarray(
            [row["lifecycle_state"] for row in graph_snapshots],
            dtype=np.str_,
        )
        graph_component_event_frames = np.asarray(
            [row["event_frame"] for row in graph_snapshots],
            dtype=np.int64,
        )
        graph_component_boxes = (
            np.stack([row["box"] for row in graph_snapshots]).astype(
                np.float32
            )
            if graph_snapshots
            else np.empty((0, 6), dtype=np.float32)
        )

        def graph_integer(name: str) -> np.ndarray:
            return np.asarray(
                [row[name] for row in graph_snapshots], dtype=np.int64
            )

        def graph_float(name: str) -> np.ndarray:
            return np.asarray(
                [row[name] for row in graph_snapshots],
                dtype=np.float32,
            )

        graph_component_hit_counts = graph_integer("hit_count")
        graph_component_view_counts = graph_integer("view_count")
        graph_component_node_counts = graph_integer("node_count")
        graph_component_edge_counts = graph_integer("edge_count")
        graph_component_unique_frame_counts = graph_integer(
            "unique_frame_count"
        )
        graph_component_track_confirmed = np.asarray(
            [row["track_confirmed"] for row in graph_snapshots],
            dtype=bool,
        )
        graph_component_confirmed = np.asarray(
            [row["graph_confirmed"] for row in graph_snapshots],
            dtype=bool,
        )
        graph_component_confirmation_frames = np.asarray(
            [row["confirmation_frame_id"] for row in graph_snapshots],
            dtype=np.str_,
        )
        graph_component_mean_edge_score = graph_float(
            "mean_edge_score"
        )
        graph_component_mean_geometry_score = graph_float(
            "mean_geometry_score"
        )
        graph_component_mean_iou_3d = graph_float("mean_iou_3d")
        graph_component_mean_mutual_inside = graph_float(
            "mean_mutual_inside"
        )
        graph_component_mean_projection_iou = graph_float(
            "mean_projection_iou"
        )
        graph_component_mean_appearance_cosine = graph_float(
            "mean_appearance_cosine"
        )
        graph_component_mean_detector_score = graph_float(
            "mean_detector_score"
        )
        graph_component_labels = np.asarray(
            [row["label"] for row in graph_snapshots], dtype=np.str_
        )
        graph_component_rejections_json = np.asarray(
            [
                json.dumps(
                    row["rejections"], sort_keys=True, separators=(",", ":")
                )
                for row in graph_snapshots
            ],
            dtype=np.str_,
        )
        graph_component_memory_view_candidates = graph_integer(
            "memory_view_candidates"
        )
        graph_component_memory_selected_views = graph_integer(
            "memory_selected_views"
        )
        graph_component_memory_geometry_points = graph_integer(
            "memory_geometry_points"
        )
        stitch_candidates = self._last_fragment_stitch_candidates
        stitch_count = len(stitch_candidates)
        stitch_max_tracks = max(
            (len(candidate.track_ids) for candidate in stitch_candidates),
            default=0,
        )
        fragment_stitch_candidate_track_ids = np.full(
            (stitch_count, stitch_max_tracks),
            -1,
            dtype=np.int64,
        )
        fragment_stitch_candidate_event_frames = np.full(
            (stitch_count, stitch_max_tracks),
            -1,
            dtype=np.int64,
        )
        fragment_stitch_candidate_track_mask = np.zeros(
            (stitch_count, stitch_max_tracks),
            dtype=bool,
        )
        for candidate_index, candidate in enumerate(stitch_candidates):
            member_count = len(candidate.track_ids)
            fragment_stitch_candidate_track_ids[
                candidate_index, :member_count
            ] = candidate.track_ids
            fragment_stitch_candidate_event_frames[
                candidate_index, :member_count
            ] = candidate.event_frames
            fragment_stitch_candidate_track_mask[
                candidate_index, :member_count
            ] = True
        fragment_stitch_candidate_representative_track_ids = np.asarray(
            [
                candidate.representative_track_id
                for candidate in stitch_candidates
            ],
            dtype=np.int64,
        )
        fragment_stitch_candidate_boxes = (
            np.stack(
                [
                    np.asarray(candidate.box, dtype=np.float32)
                    for candidate in stitch_candidates
                ]
            )
            if stitch_candidates
            else np.empty((0, 6), dtype=np.float32)
        )
        fragment_stitch_candidate_labels = np.asarray(
            [candidate.label for candidate in stitch_candidates],
            dtype=np.str_,
        )
        fragment_stitch_candidate_states_json = np.asarray(
            [
                json.dumps(
                    candidate.states,
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
                for candidate in stitch_candidates
            ],
            dtype=np.str_,
        )
        fragment_stitch_candidate_total_views = np.asarray(
            [candidate.total_views for candidate in stitch_candidates],
            dtype=np.int64,
        )
        fragment_stitch_candidate_edge_counts = np.asarray(
            [candidate.edge_count for candidate in stitch_candidates],
            dtype=np.int64,
        )

        def stitch_float(name: str) -> np.ndarray:
            return np.asarray(
                [
                    float(getattr(candidate, name))
                    for candidate in stitch_candidates
                ],
                dtype=np.float32,
            )

        fragment_stitch_candidate_min_pair_iou = stitch_float(
            "min_pair_iou"
        )
        fragment_stitch_candidate_min_pair_containment = stitch_float(
            "min_pair_containment"
        )
        fragment_stitch_candidate_max_pair_center_distance = stitch_float(
            "max_pair_center_distance"
        )
        fragment_stitch_candidate_max_detector_score = stitch_float(
            "max_detector_score"
        )
        fragment_stitch_candidate_mean_detector_score = stitch_float(
            "mean_detector_score"
        )
        observed_indices = np.asarray(
            [
                index
                for index, memory in enumerate(memories)
                if memory is not None and memory.num_points > 0
            ],
            dtype=np.int64,
        )
        points = np.zeros(
            (len(observed_indices), point_count, 3), dtype=np.float32
        )
        point_mask = np.zeros(
            (len(observed_indices), point_count), dtype=bool
        )
        geometry_points = np.zeros(
            (len(observed_indices), point_count, 3), dtype=np.float32
        )
        geometry_point_mask = np.zeros(
            (len(observed_indices), point_count), dtype=bool
        )
        view_candidate_counts = np.zeros(
            len(observed_indices), dtype=np.int64
        )
        selected_view_counts = np.zeros(
            len(observed_indices), dtype=np.int64
        )
        top_k_views = int(self.object_config["top_k_views"])
        selected_view_frame_ids = np.full(
            (len(observed_indices), top_k_views),
            -1,
            dtype=np.int64,
        )
        top_k_view_points = np.zeros(
            (
                len(observed_indices),
                top_k_views,
                point_count,
                3,
            ),
            dtype=np.float32,
        )
        top_k_view_point_mask = np.zeros(
            (len(observed_indices), top_k_views, point_count),
            dtype=bool,
        )
        top_k_view_valid = np.zeros(
            (len(observed_indices), top_k_views), dtype=bool
        )
        top_k_view_quality = np.zeros(
            (len(observed_indices), top_k_views), dtype=np.float32
        )
        top_k_view_confidence = np.zeros_like(top_k_view_quality)
        top_k_view_valid_depth_ratio = np.zeros_like(top_k_view_quality)
        top_k_view_projection_iou = np.zeros_like(top_k_view_quality)
        top_k_view_camera_position = np.full(
            (len(observed_indices), top_k_views, 3),
            np.nan,
            dtype=np.float32,
        )
        top_k_view_camera_valid = np.zeros(
            (len(observed_indices), top_k_views), dtype=bool
        )
        box_refiner_point_count = int(
            self.config["box_refiner"]["point_count"]
        )
        box_refiner_gate_point_count = int(
            self.object_config["max_points_per_object"]
        )
        box_refiner_max_view_records = int(
            self.config["quality"]["max_view_records"]
        )
        box_refiner_points_local = np.zeros(
            (
                len(observed_indices),
                box_refiner_point_count,
                3,
            ),
            dtype=np.float32,
        )
        box_refiner_point_mask = np.zeros(
            (len(observed_indices), box_refiner_point_count),
            dtype=bool,
        )
        box_refiner_local_boxes = np.full(
            (len(observed_indices), 6),
            np.nan,
            dtype=np.float32,
        )
        box_refiner_frame_valid = np.zeros(
            len(observed_indices), dtype=bool
        )
        box_refiner_frame_centers = np.full(
            (len(observed_indices), 3),
            np.nan,
            dtype=np.float64,
        )
        box_refiner_frame_basis = np.full(
            (len(observed_indices), 3, 3),
            np.nan,
            dtype=np.float64,
        )
        box_refiner_gate_points_local = np.zeros(
            (
                len(observed_indices),
                box_refiner_gate_point_count,
                3,
            ),
            dtype=np.float32,
        )
        box_refiner_gate_point_mask = np.zeros(
            (len(observed_indices), box_refiner_gate_point_count),
            dtype=bool,
        )
        box_refiner_view_valid = np.zeros(
            (len(observed_indices), box_refiner_max_view_records),
            dtype=bool,
        )
        box_refiner_view_frame_ids = np.full(
            (len(observed_indices), box_refiner_max_view_records),
            -1,
            dtype=np.int64,
        )
        box_refiner_view_scores = np.full(
            (len(observed_indices), box_refiner_max_view_records),
            np.nan,
            dtype=np.float32,
        )
        box_refiner_view_bboxes = np.full(
            (
                len(observed_indices),
                box_refiner_max_view_records,
                4,
            ),
            np.nan,
            dtype=np.float32,
        )
        box_refiner_view_intrinsics = np.full(
            (
                len(observed_indices),
                box_refiner_max_view_records,
                3,
                3,
            ),
            np.nan,
            dtype=np.float32,
        )
        box_refiner_view_camera_to_world = np.full(
            (
                len(observed_indices),
                box_refiner_max_view_records,
                4,
                4,
            ),
            np.nan,
            dtype=np.float32,
        )
        box_refiner_view_image_shapes = np.full(
            (
                len(observed_indices),
                box_refiner_max_view_records,
                2,
            ),
            -1,
            dtype=np.int64,
        )
        joint_cfg = self.config["joint_local_head"]
        joint_max_views = int(joint_cfg["max_views"])
        joint_points_per_view = int(joint_cfg["points_per_view"])
        joint_points_local = np.zeros(
            (
                len(observed_indices),
                joint_max_views,
                joint_points_per_view,
                3,
            ),
            dtype=np.float32,
        )
        joint_point_mask = np.zeros(
            (
                len(observed_indices),
                joint_max_views,
                joint_points_per_view,
            ),
            dtype=bool,
        )
        joint_view_features = np.zeros(
            (
                len(observed_indices),
                joint_max_views,
                JOINT_VIEW_FEATURE_DIM,
            ),
            dtype=np.float32,
        )
        joint_view_mask = np.zeros(
            (len(observed_indices), joint_max_views), dtype=bool
        )
        joint_local_boxes = np.full(
            (len(observed_indices), 6), np.nan, dtype=np.float32
        )
        joint_quality_features = np.zeros(
            (len(observed_indices), QUALITY_FEATURE_DIM),
            dtype=np.float32,
        )
        joint_frame_center = np.full(
            (len(observed_indices), 3), np.nan, dtype=np.float64
        )
        joint_frame_basis = np.full(
            (len(observed_indices), 3, 3),
            np.nan,
            dtype=np.float64,
        )
        joint_input_valid = np.zeros(
            len(observed_indices), dtype=bool
        )
        joint_output_valid = np.zeros(
            len(observed_indices), dtype=bool
        )
        joint_runtime_reason = np.full(
            len(observed_indices), "joint_not_run", dtype="<U64"
        )
        joint_accepted = np.zeros(len(observed_indices), dtype=bool)
        joint_quality_branch = np.full(
            len(observed_indices), -1, dtype=np.int8
        )
        joint_improvement_probability = np.full(
            len(observed_indices), np.nan, dtype=np.float32
        )
        joint_center_residual = np.full(
            (len(observed_indices), 3), np.nan, dtype=np.float32
        )
        joint_center_residual_fraction = np.full(
            (len(observed_indices), 3), np.nan, dtype=np.float32
        )
        joint_log_dimension_residual = np.full(
            (len(observed_indices), 3), np.nan, dtype=np.float32
        )
        joint_quality_components = np.full(
            (len(observed_indices), 2, 4), np.nan, dtype=np.float32
        )
        joint_ranking_scores = np.full(
            (len(observed_indices), 2), np.nan, dtype=np.float32
        )
        joint_quality_log_variance = np.full(
            (len(observed_indices), 2), np.nan, dtype=np.float32
        )
        joint_quality_uncertainty = np.full(
            (len(observed_indices), 2), np.nan, dtype=np.float32
        )
        joint_view_attention = np.zeros(
            (len(observed_indices), joint_max_views),
            dtype=np.float32,
        )
        joint_final_score = np.zeros(
            len(observed_indices), dtype=np.float32
        )
        c2_attempted = np.zeros(len(observed_indices), dtype=bool)
        c2_proposed = np.zeros(len(observed_indices), dtype=bool)
        c2_verified = np.zeros(len(observed_indices), dtype=bool)
        c2_applied = np.zeros(len(observed_indices), dtype=bool)
        c2_reason = np.full(
            len(observed_indices), "not_supplemental", dtype="<U64"
        )
        c2_branch = np.full(
            len(observed_indices), "identity", dtype="<U16"
        )
        c2_original_boxes = np.full(
            (len(observed_indices), 6), np.nan, dtype=np.float32
        )
        c2_candidate_boxes = np.full(
            (len(observed_indices), 6), np.nan, dtype=np.float32
        )
        c2_component_fraction = np.full(
            len(observed_indices), np.nan, dtype=np.float32
        )
        c2_component_density = np.full(
            len(observed_indices), np.nan, dtype=np.float32
        )
        c2_density_ratio = np.full(
            len(observed_indices), np.nan, dtype=np.float32
        )
        c2_point_count = np.zeros(
            len(observed_indices), dtype=np.int64
        )
        c2_view_count = np.zeros(
            len(observed_indices), dtype=np.int64
        )
        c2_original_support = np.full(
            len(observed_indices), np.nan, dtype=np.float32
        )
        c2_candidate_support = np.full(
            len(observed_indices), np.nan, dtype=np.float32
        )
        c2_original_projection = np.full(
            len(observed_indices), np.nan, dtype=np.float32
        )
        c2_candidate_projection = np.full(
            len(observed_indices), np.nan, dtype=np.float32
        )
        c2_projection_delta = np.full(
            len(observed_indices), np.nan, dtype=np.float32
        )
        c2_projection_views = np.zeros(
            len(observed_indices), dtype=np.int64
        )
        c2_point_support_views = np.zeros(
            len(observed_indices), dtype=np.int64
        )
        c2_center_shift_ratio = np.full(
            len(observed_indices), np.nan, dtype=np.float32
        )
        c2_extent_ratios = np.full(
            (len(observed_indices), 3), np.nan, dtype=np.float32
        )
        for output_index, result_index in enumerate(observed_indices):
            memory = memories[int(result_index)]
            assert memory is not None
            c2_runtime = self._last_c2_runtime.get(
                int(result.stable_ids[int(result_index)])
            )
            if c2_runtime is not None:
                c2_attempted[output_index] = bool(
                    c2_runtime.get("attempted", False)
                )
                c2_proposed[output_index] = bool(
                    c2_runtime.get("proposed", False)
                )
                c2_verified[output_index] = bool(
                    c2_runtime.get("verified", False)
                )
                c2_applied[output_index] = bool(
                    c2_runtime.get("applied", False)
                )
                c2_reason[output_index] = str(
                    c2_runtime.get("reason", "unknown")
                )
                c2_branch[output_index] = str(
                    c2_runtime.get("branch", "identity")
                )
                for destination_array, key in (
                    (c2_original_boxes, "original_box"),
                    (c2_candidate_boxes, "candidate_box"),
                ):
                    value = np.asarray(
                        c2_runtime.get(
                            key, np.full(6, np.nan)
                        ),
                        dtype=np.float32,
                    )
                    if value.shape == (6,):
                        destination_array[output_index] = value
                for destination_array, key in (
                    (
                        c2_component_fraction,
                        "component_fraction",
                    ),
                    (
                        c2_component_density,
                        "component_density",
                    ),
                    (c2_density_ratio, "density_ratio"),
                    (c2_original_support, "original_support"),
                    (c2_candidate_support, "candidate_support"),
                    (
                        c2_original_projection,
                        "original_projection",
                    ),
                    (
                        c2_candidate_projection,
                        "candidate_projection",
                    ),
                    (c2_projection_delta, "projection_delta"),
                    (
                        c2_center_shift_ratio,
                        "center_shift_ratio",
                    ),
                ):
                    destination_array[output_index] = float(
                        c2_runtime.get(key, np.nan)
                    )
                c2_point_count[output_index] = int(
                    c2_runtime.get("point_count", 0)
                )
                c2_view_count[output_index] = int(
                    c2_runtime.get("view_count", 0)
                )
                c2_projection_views[output_index] = int(
                    c2_runtime.get("projection_views", 0)
                )
                c2_point_support_views[output_index] = int(
                    c2_runtime.get("point_support_views", 0)
                )
                extent_ratios = np.asarray(
                    c2_runtime.get(
                        "extent_ratios", np.full(3, np.nan)
                    ),
                    dtype=np.float32,
                )
                if extent_ratios.shape == (3,):
                    c2_extent_ratios[output_index] = extent_ratios
            sampled = deterministic_bounded_sample(
                memory.points, point_count
            )
            points[output_index, : len(sampled)] = sampled
            point_mask[output_index, : len(sampled)] = True
            geometry_sampled = deterministic_bounded_sample(
                memory.geometry_points, point_count
            )
            geometry_points[
                output_index, : len(geometry_sampled)
            ] = geometry_sampled
            geometry_point_mask[
                output_index, : len(geometry_sampled)
            ] = True
            view_candidate_counts[output_index] = (
                memory.view_candidate_count
            )
            selected_view_counts[output_index] = (
                memory.selected_view_count
            )
            frame_ids = memory.selected_view_frame_ids
            selected_view_frame_ids[
                output_index, : len(frame_ids)
            ] = frame_ids
            for view_index, record in enumerate(
                memory.selected_view_records
            ):
                view_sampled = deterministic_bounded_sample(
                    record.points_world, point_count
                )
                top_k_view_points[
                    output_index,
                    view_index,
                    : len(view_sampled),
                ] = view_sampled
                top_k_view_point_mask[
                    output_index,
                    view_index,
                    : len(view_sampled),
                ] = True
                top_k_view_valid[output_index, view_index] = True
                top_k_view_quality[
                    output_index, view_index
                ] = record.quality
                top_k_view_confidence[
                    output_index, view_index
                ] = record.confidence
                top_k_view_valid_depth_ratio[
                    output_index, view_index
                ] = record.valid_depth_ratio
                top_k_view_projection_iou[
                    output_index, view_index
                ] = record.projection_mask_iou
                if record.camera_position is not None:
                    top_k_view_camera_position[
                        output_index, view_index
                    ] = record.camera_position
                    top_k_view_camera_valid[
                        output_index, view_index
                    ] = True
            try:
                refiner_inputs = _prepare_oriented_box_refiner_inputs(
                    result.refit_original_corners[int(result_index)],
                    memory.geometry_points,
                    box_refiner_point_count,
                )
            except ValueError:
                # This is the same invalid-OBB condition that prevents the
                # online refiner from being invoked.  Keep all masks false
                # rather than manufacturing a world-AABB training example.
                refiner_inputs = None
            if refiner_inputs is not None:
                box_refiner_points_local[output_index] = (
                    refiner_inputs.points_local
                )
                box_refiner_point_mask[output_index] = (
                    refiner_inputs.point_mask
                )
                box_refiner_local_boxes[output_index] = (
                    refiner_inputs.local_box
                )
                box_refiner_frame_valid[output_index] = True
                box_refiner_frame_centers[output_index] = (
                    refiner_inputs.frame_center
                )
                box_refiner_frame_basis[output_index] = (
                    refiner_inputs.frame_basis
                )
                gate_count = refiner_inputs.gate_points_local.shape[0]
                if gate_count > box_refiner_gate_point_count:
                    raise RuntimeError(
                        "geometry memory exceeds its configured point budget"
                    )
                box_refiner_gate_points_local[
                    output_index, :gate_count
                ] = refiner_inputs.gate_points_local
                box_refiner_gate_point_mask[
                    output_index, :gate_count
                ] = True
            evidence = self.global_tracks.get(
                int(result.stable_ids[int(result_index)])
            )
            row_stats = evidence_stats_rows[int(result_index)]
            if row_stats is not None:
                for view_index, view in enumerate(
                    row_stats.view_records[
                        :box_refiner_max_view_records
                    ]
                ):
                    box_refiner_view_valid[
                        output_index, view_index
                    ] = True
                    box_refiner_view_frame_ids[
                        output_index, view_index
                    ] = int(view.frame_index)
                    box_refiner_view_scores[
                        output_index, view_index
                    ] = float(view.score)
                    box_refiner_view_bboxes[
                        output_index, view_index
                    ] = view.bbox
                    box_refiner_view_intrinsics[
                        output_index, view_index
                    ] = view.intrinsics
                    box_refiner_view_camera_to_world[
                        output_index, view_index
                    ] = view.camera_to_world
                    box_refiner_view_image_shapes[
                        output_index, view_index
                    ] = np.asarray(view.image_shape, dtype=np.int64)
            stable_id = int(result.stable_ids[int(result_index)])
            runtime = self._last_joint_runtime.get(stable_id)
            joint_final_score[output_index] = result.scores[
                int(result_index)
            ]
            if runtime is not None:
                joint_runtime_reason[output_index] = str(
                    runtime.get("reason", "joint_not_run")
                )
            prepared = (
                runtime.get("prepared")
                if runtime is not None
                and bool(runtime.get("input_valid", False))
                else None
            )
            # Observer and legacy B5 profiles do not run the joint model, but
            # must serialize the exact VxP joint inputs used by the active
            # runtime.  Reuse the same public preparation helper.
            if runtime is None and evidence is not None:
                geometry_views = (
                    evidence.memory.geometry_unique_view_count
                )
                if not evidence.memory.top_k_enabled:
                    geometry_views += evidence.stats.absorbed_views
                if (
                    geometry_views
                    >= int(self.config["refit"]["min_views"])
                    and evidence.memory.geometry_num_points
                    >= int(self.config["refit"]["min_points"])
                ):
                    try:
                        (
                            frame_center,
                            dimensions,
                            frame_basis,
                        ) = _oriented_box_frame(
                            result.refit_original_corners[
                                int(result_index)
                            ]
                        )
                        exact_inputs = prepare_joint_view_inputs(
                            evidence.memory.selected_view_records,
                            frame_center=frame_center,
                            frame_basis=frame_basis,
                            max_views=joint_max_views,
                            points_per_view=joint_points_per_view,
                        )
                    except ValueError:
                        exact_inputs = None
                    if (
                        exact_inputs is not None
                        and int(
                            np.count_nonzero(exact_inputs.view_mask)
                        )
                        >= int(self.config["refit"]["min_views"])
                    ):
                        original_box = result.refit_original_boxes[
                            int(result_index)
                        ]
                        detector_score = float(evidence.detector_score)
                        quality_mapping = self._quality_mapping(
                            original_box=original_box,
                            final_box=original_box,
                            detector_score=detector_score,
                            memory=evidence.memory,
                            stats=evidence.stats,
                            supplemental=False,
                            refiner_quality=0.5,
                        )
                        prepared = _JointPreparedInstance(
                            index=int(result_index),
                            stable_id=stable_id,
                            evidence=evidence,
                            original_box=original_box.copy(),
                            original_corners=(
                                result.refit_original_corners[
                                    int(result_index)
                                ].copy()
                            ),
                            detector_score=detector_score,
                            quality_mapping=quality_mapping,
                            quality_features=np.array(
                                quality_feature_vector(
                                    quality_mapping
                                ),
                                dtype=np.float32,
                                copy=True,
                            ),
                            frame_center=frame_center.copy(),
                            frame_basis=frame_basis.copy(),
                            local_box=np.concatenate(
                                (
                                    np.zeros(3, dtype=np.float32),
                                    dimensions.astype(np.float32),
                                )
                            ),
                            gate_points_local=_points_to_box_local(
                                evidence.memory.geometry_points,
                                frame_center,
                                frame_basis,
                            ),
                            points_local=(
                                exact_inputs.points_local.copy()
                            ),
                            point_mask=exact_inputs.point_mask.copy(),
                            view_features=(
                                exact_inputs.view_features.copy()
                            ),
                            view_mask=exact_inputs.view_mask.copy(),
                        )
                        joint_runtime_reason[
                            output_index
                        ] = "joint_observer"
            if prepared is not None:
                joint_points_local[output_index] = (
                    prepared.points_local
                )
                joint_point_mask[output_index] = prepared.point_mask
                joint_view_features[output_index] = (
                    prepared.view_features
                )
                joint_view_mask[output_index] = prepared.view_mask
                joint_local_boxes[output_index] = prepared.local_box
                joint_quality_features[output_index] = (
                    prepared.quality_features
                )
                joint_frame_center[output_index] = (
                    prepared.frame_center
                )
                joint_frame_basis[output_index] = prepared.frame_basis
                joint_input_valid[output_index] = True
            prediction = (
                runtime.get("prediction")
                if runtime is not None
                else None
            )
            if isinstance(prediction, _JointPrediction):
                joint_output_valid[output_index] = True
                joint_center_residual[output_index] = (
                    prediction.center_residual
                )
                joint_center_residual_fraction[output_index] = (
                    prediction.center_residual_fraction
                )
                joint_log_dimension_residual[output_index] = (
                    prediction.log_dimension_residual
                )
                joint_improvement_probability[output_index] = (
                    prediction.improvement_probability
                )
                joint_quality_components[output_index] = (
                    prediction.quality_components
                )
                joint_ranking_scores[output_index] = (
                    prediction.ranking_scores
                )
                joint_quality_log_variance[output_index] = (
                    prediction.quality_log_variance
                )
                joint_quality_uncertainty[output_index] = (
                    prediction.quality_uncertainty
                )
                joint_view_attention[output_index] = (
                    prediction.view_attention
                )
                joint_accepted[output_index] = bool(
                    runtime.get("accepted", False)
                )
                branch = runtime.get("quality_branch")
                joint_quality_branch[output_index] = (
                    1 if branch == "candidate" else 0
                )
        output_is_supplemental = (
            result.source_indices[observed_indices] < 0
        )
        confirmed_graph_track_ids = {
            int(row["track_id"])
            for row in graph_snapshots
            if bool(row["graph_confirmed"])
        }
        output_mask_graph_confirmed = np.asarray(
            [
                bool(
                    result.source_indices[int(index)] < 0
                    and (
                        -int(result.stable_ids[int(index)]) - 1
                        in confirmed_graph_track_ids
                    )
                )
                for index in observed_indices
            ],
            dtype=bool,
        )
        labels = np.asarray(
            [
                result.labels[int(index)]
                if result.labels[int(index)] is not None
                else ""
                for index in observed_indices
            ],
            dtype=np.str_,
        )
        summary_json = json.dumps(
            dict(result.summary), sort_keys=True, default=str
        )
        c4_payload = self._c4_diagnostic_payload(
            result, point_count
        )
        trifusion_payload = self._trifusion_diagnostic_payload(result)
        yidu_payload = self._yidu_diagnostic_payload(result)
        trifusion_missing_payload = (
            self._trifusion_missing_diagnostic_payload()
        )
        residual_track_payload = (
            self._residual_track_diagnostic_payload()
        )
        destination = root / f"{scene_id}_tracks.npz"
        with tempfile.NamedTemporaryFile(
            dir=root, suffix=".npz", delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            np.savez_compressed(
                temporary,
                scene_id=np.asarray(scene_id),
                graph_component_track_ids=graph_component_track_ids,
                graph_component_states=graph_component_states,
                graph_component_event_frames=(
                    graph_component_event_frames
                ),
                graph_component_boxes=graph_component_boxes,
                graph_component_hit_counts=graph_component_hit_counts,
                graph_component_view_counts=graph_component_view_counts,
                graph_component_node_counts=graph_component_node_counts,
                graph_component_edge_counts=graph_component_edge_counts,
                graph_component_unique_frame_counts=(
                    graph_component_unique_frame_counts
                ),
                graph_component_track_confirmed=(
                    graph_component_track_confirmed
                ),
                graph_component_confirmed=graph_component_confirmed,
                graph_component_confirmation_frames=(
                    graph_component_confirmation_frames
                ),
                graph_component_mean_edge_score=(
                    graph_component_mean_edge_score
                ),
                graph_component_mean_geometry_score=(
                    graph_component_mean_geometry_score
                ),
                graph_component_mean_iou_3d=(
                    graph_component_mean_iou_3d
                ),
                graph_component_mean_mutual_inside=(
                    graph_component_mean_mutual_inside
                ),
                graph_component_mean_projection_iou=(
                    graph_component_mean_projection_iou
                ),
                graph_component_mean_appearance_cosine=(
                    graph_component_mean_appearance_cosine
                ),
                graph_component_mean_detector_score=(
                    graph_component_mean_detector_score
                ),
                graph_component_labels=graph_component_labels,
                graph_component_rejections_json=(
                    graph_component_rejections_json
                ),
                graph_component_memory_view_candidates=(
                    graph_component_memory_view_candidates
                ),
                graph_component_memory_selected_views=(
                    graph_component_memory_selected_views
                ),
                graph_component_memory_geometry_points=(
                    graph_component_memory_geometry_points
                ),
                fragment_stitch_candidate_track_ids=(
                    fragment_stitch_candidate_track_ids
                ),
                fragment_stitch_candidate_event_frames=(
                    fragment_stitch_candidate_event_frames
                ),
                fragment_stitch_candidate_track_mask=(
                    fragment_stitch_candidate_track_mask
                ),
                fragment_stitch_candidate_representative_track_ids=(
                    fragment_stitch_candidate_representative_track_ids
                ),
                fragment_stitch_candidate_boxes=(
                    fragment_stitch_candidate_boxes
                ),
                fragment_stitch_candidate_labels=(
                    fragment_stitch_candidate_labels
                ),
                fragment_stitch_candidate_states_json=(
                    fragment_stitch_candidate_states_json
                ),
                fragment_stitch_candidate_total_views=(
                    fragment_stitch_candidate_total_views
                ),
                fragment_stitch_candidate_edge_counts=(
                    fragment_stitch_candidate_edge_counts
                ),
                fragment_stitch_candidate_min_pair_iou=(
                    fragment_stitch_candidate_min_pair_iou
                ),
                fragment_stitch_candidate_min_pair_containment=(
                    fragment_stitch_candidate_min_pair_containment
                ),
                fragment_stitch_candidate_max_pair_center_distance=(
                    fragment_stitch_candidate_max_pair_center_distance
                ),
                fragment_stitch_candidate_max_detector_score=(
                    fragment_stitch_candidate_max_detector_score
                ),
                fragment_stitch_candidate_mean_detector_score=(
                    fragment_stitch_candidate_mean_detector_score
                ),
                fragment_stitch_diagnostics_schema=np.asarray(
                    "mask_graph_fragment_stitch_v2"
                ),
                boxes=result.boxes[observed_indices],
                scores=result.scores[observed_indices],
                quality_features=result.quality_features[observed_indices],
                c2_attempted=c2_attempted,
                c2_proposed=c2_proposed,
                c2_verified=c2_verified,
                c2_applied=c2_applied,
                c2_reason=c2_reason,
                c2_branch=c2_branch,
                c2_original_boxes=c2_original_boxes,
                c2_candidate_boxes=c2_candidate_boxes,
                c2_component_fraction=c2_component_fraction,
                c2_component_density=c2_component_density,
                c2_density_ratio=c2_density_ratio,
                c2_point_count=c2_point_count,
                c2_view_count=c2_view_count,
                c2_original_support=c2_original_support,
                c2_candidate_support=c2_candidate_support,
                c2_original_projection=c2_original_projection,
                c2_candidate_projection=c2_candidate_projection,
                c2_projection_delta=c2_projection_delta,
                c2_projection_views=c2_projection_views,
                c2_point_support_views=c2_point_support_views,
                c2_center_shift_ratio=c2_center_shift_ratio,
                c2_extent_ratios=c2_extent_ratios,
                supplemental_geometry_diagnostics_schema=np.asarray(
                    result.summary[
                        "supplemental_geometry_diagnostics_schema"
                    ]
                ),
                joint_points_local=joint_points_local,
                joint_point_mask=joint_point_mask,
                joint_view_features=joint_view_features,
                joint_view_mask=joint_view_mask,
                joint_local_boxes=joint_local_boxes,
                joint_quality_features=joint_quality_features,
                # Keep both spellings for strict dataset builders created
                # during the migration.  They contain the same [N,3] array.
                joint_frame_center=joint_frame_center,
                joint_frame_centers=joint_frame_center,
                joint_frame_basis=joint_frame_basis,
                joint_input_valid=joint_input_valid,
                joint_output_valid=joint_output_valid,
                joint_runtime_reason=joint_runtime_reason,
                joint_accepted=joint_accepted,
                joint_quality_branch=joint_quality_branch,
                joint_center_residual=joint_center_residual,
                joint_center_residual_fraction=(
                    joint_center_residual_fraction
                ),
                joint_log_dimension_residual=(
                    joint_log_dimension_residual
                ),
                joint_improvement_probability=(
                    joint_improvement_probability
                ),
                joint_quality_components=joint_quality_components,
                joint_ranking_scores=joint_ranking_scores,
                joint_quality_log_variance=(
                    joint_quality_log_variance
                ),
                joint_quality_uncertainty=joint_quality_uncertainty,
                joint_view_attention=joint_view_attention,
                joint_final_score=joint_final_score,
                joint_input_schema=np.asarray(
                    JOINT_LOCAL_HEAD_INPUT_SCHEMA
                ),
                joint_output_schema=np.asarray(
                    JOINT_LOCAL_HEAD_OUTPUT_SCHEMA
                ),
                points=points,
                point_mask=point_mask,
                geometry_points=geometry_points,
                geometry_point_mask=geometry_point_mask,
                view_candidate_counts=view_candidate_counts,
                selected_view_counts=selected_view_counts,
                selected_view_frame_ids=selected_view_frame_ids,
                top_k_view_points=top_k_view_points,
                top_k_view_point_mask=top_k_view_point_mask,
                top_k_view_valid=top_k_view_valid,
                top_k_view_quality=top_k_view_quality,
                top_k_view_confidence=top_k_view_confidence,
                top_k_view_valid_depth_ratio=(
                    top_k_view_valid_depth_ratio
                ),
                top_k_view_projection_iou=top_k_view_projection_iou,
                top_k_view_camera_position=top_k_view_camera_position,
                top_k_view_camera_valid=top_k_view_camera_valid,
                box_refiner_points_local=box_refiner_points_local,
                box_refiner_point_mask=box_refiner_point_mask,
                box_refiner_local_boxes=box_refiner_local_boxes,
                box_refiner_frame_valid=box_refiner_frame_valid,
                box_refiner_frame_centers=box_refiner_frame_centers,
                box_refiner_frame_basis=box_refiner_frame_basis,
                box_refiner_gate_points_local=(
                    box_refiner_gate_points_local
                ),
                box_refiner_gate_point_mask=(
                    box_refiner_gate_point_mask
                ),
                box_refiner_view_valid=box_refiner_view_valid,
                box_refiner_view_frame_ids=box_refiner_view_frame_ids,
                box_refiner_view_scores=box_refiner_view_scores,
                box_refiner_view_bboxes=box_refiner_view_bboxes,
                box_refiner_view_intrinsics=box_refiner_view_intrinsics,
                box_refiner_view_camera_to_world=(
                    box_refiner_view_camera_to_world
                ),
                box_refiner_view_image_shapes=(
                    box_refiner_view_image_shapes
                ),
                top_k_views=np.asarray(top_k_views, dtype=np.int64),
                runtime_diagnostics_schema=np.asarray(
                    result.summary["runtime_diagnostics_schema"]
                ),
                mask_graph_diagnostics_schema=np.asarray(
                    result.summary["mask_graph_diagnostics_schema"]
                ),
                box_refiner_input_schema=np.asarray(
                    result.summary["box_refiner_input_schema"]
                ),
                online_ablation_profile=np.asarray(
                    result.summary["online_ablation_profile"]
                ),
                candidate_ttl_clock=np.asarray(
                    result.summary["candidate_ttl_clock"]
                ),
                candidate_track_ttl=np.asarray(
                    result.summary["candidate_track_ttl"],
                    dtype=np.int64,
                ),
                archive_confirmed_tracks=np.asarray(
                    result.summary["archive_confirmed_tracks"],
                    dtype=bool,
                ),
                mutation_refit_enabled=np.asarray(
                    result.summary["mutation_refit_enabled"],
                    dtype=bool,
                ),
                mutation_box_refiner_enabled=np.asarray(
                    result.summary["mutation_box_refiner_enabled"],
                    dtype=bool,
                ),
                mutation_quality_enabled=np.asarray(
                    result.summary["mutation_quality_enabled"],
                    dtype=bool,
                ),
                mutation_joint_local_head_enabled=np.asarray(
                    result.summary[
                        "mutation_joint_local_head_enabled"
                    ],
                    dtype=bool,
                ),
                mutation_joint_geometry_enabled=np.asarray(
                    result.summary[
                        "mutation_joint_geometry_enabled"
                    ],
                    dtype=bool,
                ),
                mutation_joint_scores_enabled=np.asarray(
                    result.summary["mutation_joint_scores_enabled"],
                    dtype=bool,
                ),
                mutation_supplemental_output_enabled=np.asarray(
                    result.summary[
                        "mutation_supplemental_output_enabled"
                    ],
                    dtype=bool,
                ),
                fragment_stitch_enabled=np.asarray(
                    result.summary["fragment_stitch_enabled"],
                    dtype=bool,
                ),
                mutation_fragment_stitch_enabled=np.asarray(
                    result.summary["mutation_fragment_stitch_enabled"],
                    dtype=bool,
                ),
                fragment_stitch_invalid_snapshots=np.asarray(
                    result.summary[
                        "fragment_stitch_invalid_snapshots"
                    ],
                    dtype=np.int64,
                ),
                fragment_stitch_fail_open=np.asarray(
                    result.summary["fragment_stitch_fail_open"],
                    dtype=bool,
                ),
                fragment_stitch_error=np.asarray(
                    result.summary["fragment_stitch_error"],
                ),
                fragment_stitch_config_json=np.asarray(
                    result.summary["fragment_stitch_config_json"],
                ),
                mutation_soft_nms_enabled=np.asarray(
                    result.summary["mutation_soft_nms_enabled"],
                    dtype=bool,
                ),
                output_minimum_extent=np.asarray(
                    result.summary["output_minimum_extent"],
                    dtype=np.float64,
                ),
                final_output_minimum_extent=np.asarray(
                    (
                        np.nan
                        if result.summary[
                            "final_output_minimum_extent"
                        ]
                        is None
                        else result.summary[
                            "final_output_minimum_extent"
                        ]
                    ),
                    dtype=np.float64,
                ),
                supplemental_minimum_extent=np.asarray(
                    result.summary["supplemental_minimum_extent"],
                    dtype=np.float64,
                ),
                joint_max_views=np.asarray(
                    result.summary["joint_max_views"],
                    dtype=np.int64,
                ),
                joint_points_per_view=np.asarray(
                    result.summary["joint_points_per_view"],
                    dtype=np.int64,
                ),
                joint_improvement_threshold=np.asarray(
                    result.summary["joint_improvement_threshold"],
                    dtype=np.float64,
                ),
                joint_max_candidate_uncertainty=np.asarray(
                    result.summary[
                        "joint_max_candidate_uncertainty"
                    ],
                    dtype=np.float64,
                ),
                joint_detector_blend=np.asarray(
                    result.summary["joint_detector_blend"],
                    dtype=np.float64,
                ),
                joint_preserve_original_floor=np.asarray(
                    result.summary[
                        "joint_preserve_original_floor"
                    ],
                    dtype=bool,
                ),
                box_refiner_point_count=np.asarray(
                    result.summary["box_refiner_point_count"],
                    dtype=np.int64,
                ),
                box_refiner_gate_point_count=np.asarray(
                    result.summary["box_refiner_gate_point_count"],
                    dtype=np.int64,
                ),
                box_refiner_max_view_records=np.asarray(
                    result.summary["box_refiner_max_view_records"],
                    dtype=np.int64,
                ),
                box_refiner_coordinate_frame=np.asarray(
                    result.summary["box_refiner_coordinate_frame"]
                ),
                refit_gate_min_views=np.asarray(
                    result.summary["refit_gate_min_views"],
                    dtype=np.int64,
                ),
                refit_gate_min_points=np.asarray(
                    result.summary["refit_gate_min_points"],
                    dtype=np.int64,
                ),
                refit_gate_max_center_shift_ratio=np.asarray(
                    result.summary[
                        "refit_gate_max_center_shift_ratio"
                    ],
                    dtype=np.float64,
                ),
                refit_gate_min_extent_ratio=np.asarray(
                    result.summary["refit_gate_min_extent_ratio"],
                    dtype=np.float64,
                ),
                refit_gate_max_extent_ratio=np.asarray(
                    result.summary["refit_gate_max_extent_ratio"],
                    dtype=np.float64,
                ),
                refit_gate_min_original_point_support=np.asarray(
                    result.summary[
                        "refit_gate_min_original_point_support"
                    ],
                    dtype=np.float64,
                ),
                refit_gate_min_candidate_point_support=np.asarray(
                    result.summary[
                        "refit_gate_min_candidate_point_support"
                    ],
                    dtype=np.float64,
                ),
                refit_gate_max_candidate_support_drop=np.asarray(
                    result.summary[
                        "refit_gate_max_candidate_support_drop"
                    ],
                    dtype=np.float64,
                ),
                refit_gate_min_reprojection_iou=np.asarray(
                    result.summary["refit_gate_min_reprojection_iou"],
                    dtype=np.float64,
                ),
                refit_gate_min_reprojection_improvement=np.asarray(
                    result.summary[
                        "refit_gate_min_reprojection_improvement"
                    ],
                    dtype=np.float64,
                ),
                b3_schema=np.asarray("topk_mask_rgbd_v1"),
                b5_schema=np.asarray(
                    (
                        "oriented_local_refiner_v1"
                        if self.config["box_refiner"]["enabled"]
                        and self.box_refiner_coordinate_frame == "box_local"
                        else "disabled"
                    )
                ),
                b3_refit_schema=np.asarray(
                    (
                        (
                            "visibility_aware_oriented_v3"
                            if self.config["refit"][
                                "preserve_box_orientation"
                            ]
                            else "visibility_aware_v2"
                        )
                        if self.config["refit"]["strategy"]
                        == "visibility_aware"
                        else "quantile_blend_v1"
                    )
                ),
                refit_original_boxes=result.refit_original_boxes[
                    observed_indices
                ],
                refit_original_corners=result.refit_original_corners[
                    observed_indices
                ],
                refit_candidate_boxes=result.boxes[observed_indices],
                refit_candidate_corners=result.corners[
                    observed_indices
                ],
                refit_applied=result.refit_applied[observed_indices],
                refit_reason=np.asarray(
                    result.refit_reasons, dtype=np.str_
                )[observed_indices],
                refit_changed_axes=result.refit_changed_axes[
                    observed_indices
                ],
                refit_boundary_delta=result.refit_boundary_delta[
                    observed_indices
                ],
                refit_local_original_boxes=(
                    result.refit_local_original_boxes[observed_indices]
                ),
                refit_local_candidate_boxes=(
                    result.refit_local_candidate_boxes[observed_indices]
                ),
                refit_local_basis=result.refit_local_basis[
                    observed_indices
                ],
                refit_local_frame_valid=(
                    result.refit_local_frame_valid[observed_indices]
                ),
                source_indices=result.source_indices[observed_indices],
                output_is_supplemental=output_is_supplemental,
                output_mask_graph_confirmed=(
                    output_mask_graph_confirmed
                ),
                track_ids=result.stable_ids[observed_indices],
                result_indices=observed_indices,
                labels=labels,
                quality_feature_names=np.asarray(
                    QUALITY_FEATURE_NAMES, dtype=np.str_
                ),
                summary_json=np.asarray(summary_json),
                **c4_payload,
                **trifusion_payload,
                **yidu_payload,
                **trifusion_missing_payload,
                **residual_track_payload,
            )
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    def summary(self) -> Dict[str, Any]:
        rejected = dict(sorted(self.stats["rejected"].items()))
        joint_rejected = dict(
            sorted(self.stats["joint_rejected"].items())
        )
        mask_graph_rejected = dict(
            sorted(self.stats["mask_graph_rejected"].items())
        )
        c2_rejected = dict(
            sorted(self.stats["c2_rejected"].items())
        )
        c4_rejected = dict(
            sorted(self.stats["c4_rejected"].items())
        )
        trifusion_rejected = dict(
            sorted(self.stats["trifusion_rejected"].items())
        )
        trifusion_gate_rejected = dict(
            sorted(self.stats["trifusion_gate_rejected"].items())
        )
        trifusion_missing_errors = dict(
            sorted(self.stats["trifusion_missing_errors"].items())
        )
        residual_track_errors = dict(
            sorted(self.stats["residual_track_errors"].items())
        )
        residual_track_audit = self._residual_track_audit_metrics()
        residual_track_provider_nodes = dict(
            sorted(self.stats["residual_track_provider_nodes"].items())
        )
        global_memories = [
            evidence.memory for evidence in self.global_tracks.values()
        ]
        c4_memories = [
            evidence.memory
            for evidence in self.generic_geometry_global_tracks.values()
        ]
        graph_snapshots = self._diagnostic_mask_graph_snapshots()
        supplemental_memories: List[ObjectGeometryMemory] = []
        if self.track_manager is not None:
            supplemental_memories = [
                track.memory
                for _, track in sorted(
                    {
                        **self.track_manager.tracks,
                        **self.track_manager.archived_tracks,
                    }.items()
                )
            ]
        if self.config["supplemental_output"][
            "recover_absorbed_confirmed"
        ]:
            supplemental_memories.extend(
                record.track.memory
                for _, record in sorted(
                    self.absorbed_supplemental_records.items()
                )
            )
        top_k_views = int(self.object_config.get("top_k_views", 0))
        final_minimum_extent = self.config["output_filter"][
            "final_minimum_extent"
        ]
        configured_supplemental_extent = self.config[
            "supplemental_output"
        ]["minimum_extent"]
        effective_supplemental_extent = (
            float(final_minimum_extent)
            if self.config["supplemental_output"][
                "class_aware_extent"
            ]
            and final_minimum_extent is not None
            else float(
                self.config["output_filter"]["minimum_extent"]
                if configured_supplemental_extent is None
                else configured_supplemental_extent
            )
        )
        return {
            "enabled": self.enabled,
            "keyframes": int(self.stats["keyframes"]),
            "provider_calls": int(self.stats["provider_calls"]),
            "proposal_provider": (
                str(
                    self.config["supplemental_proposals"].get(
                        "provider", type(self.provider).__name__
                    )
                )
                if self.config["supplemental_proposals"].get(
                    "enabled", False
                )
                else "disabled"
            ),
            "proposal_cache_hits": int(
                getattr(self.provider, "hits", 0)
            ),
            "proposal_cache_misses": int(
                getattr(self.provider, "misses", 0)
            ),
            "provider_seconds": float(self.stats["provider_seconds"]),
            "appearance_seconds": float(self.stats["appearance_seconds"]),
            "geometry_seconds": float(self.stats["geometry_seconds"]),
            "proposals": int(self.stats["proposals"]),
            "lifted": int(self.stats["lifted"]),
            "matched_global": int(self.stats["matched_global"]),
            "strong_global_matches": int(
                self.stats["strong_global_matches"]
            ),
            "weak_global_matches": int(
                self.stats["weak_global_matches"]
            ),
            "weak_shadow_candidates": int(
                self.stats["weak_shadow_candidates"]
            ),
            "unmatched_candidates": int(
                self.stats["unmatched_candidates"]
            ),
            "candidate_updates": int(self.stats["candidate_updates"]),
            "missing_track_identity_enabled": bool(
                self.config["missing_track_identity"]["enabled"]
            ),
            "mask_graph_enabled": bool(
                self.config["mask_graph"]["enabled"]
            ),
            "mask_graph_edges_evaluated": int(
                self.stats["mask_graph_edges_evaluated"]
            ),
            "mask_graph_edges_accepted": int(
                self.stats["mask_graph_edges_accepted"]
            ),
            "mask_graph_nodes_added": int(
                self.stats["mask_graph_nodes"]
            ),
            "mask_graph_newly_confirmed": int(
                self.stats["mask_graph_confirmed"]
            ),
            "mask_graph_components": len(graph_snapshots),
            "mask_graph_confirmed_components": int(
                sum(
                    bool(row["graph_confirmed"])
                    for row in graph_snapshots
                )
            ),
            "mask_graph_retired_components": int(
                sum(
                    row["lifecycle_state"]
                    in {"absorbed", "discarded", "expired"}
                    for row in graph_snapshots
                )
            ),
            "mask_graph_rejected": mask_graph_rejected,
            "fragment_stitch_enabled": bool(
                self.config["fragment_stitch"]["enabled"]
            ),
            "fragment_stitch_candidates": len(
                self._last_fragment_stitch_candidates
            ),
            "fragment_stitch_candidate_tracks": int(
                sum(
                    len(candidate.track_ids)
                    for candidate in self._last_fragment_stitch_candidates
                )
            ),
            "fragment_stitch_seconds": float(
                self.stats["fragment_stitch_seconds"]
            ),
            "fragment_stitch_invalid_snapshots": int(
                self.stats["fragment_stitch_invalid_snapshots"]
            ),
            "fragment_stitch_fail_open": bool(
                self.stats["fragment_stitch_fail_open"]
            ),
            "fragment_stitch_error": self._last_fragment_stitch_error,
            "fragment_stitch_config_json": json.dumps(
                self.config["fragment_stitch"],
                sort_keys=True,
                separators=(",", ":"),
            ),
            "fragment_stitch_diagnostics_schema": (
                "mask_graph_fragment_stitch_v2"
            ),
            "candidate_ttl_clock": (
                self.config["candidate_lifecycle"]["ttl_clock"]
                if self.enabled
                else "disabled"
            ),
            "online_ablation_profile": self.config[
                "ablation_profile"
            ],
            "runtime_diagnostics_schema": (
                "box_refiner_k5_runtime_v1"
            ),
            "mask_graph_diagnostics_schema": (
                "missing_track_mask_graph_v1"
            ),
            "box_refiner_input_schema": (
                "oriented_local_refiner_input_v1"
            ),
            "joint_input_schema": JOINT_LOCAL_HEAD_INPUT_SCHEMA,
            "joint_output_schema": JOINT_LOCAL_HEAD_OUTPUT_SCHEMA,
            "candidate_track_ttl": int(
                self.object_config.get("track_ttl", 0)
            ),
            "archive_confirmed_tracks": bool(
                self.config["candidate_lifecycle"][
                    "archive_confirmed"
                ]
            ),
            "mutation_refit_enabled": bool(
                self.config["refit"]["enabled"]
            ),
            "mutation_box_refiner_enabled": bool(
                self.config["box_refiner"]["enabled"]
            ),
            "mutation_quality_enabled": bool(
                self.config["quality"]["enabled"]
            ),
            "mutation_joint_local_head_enabled": bool(
                self.config["joint_local_head"]["enabled"]
            ),
            "mutation_joint_geometry_enabled": bool(
                self.config["joint_local_head"]["enabled"]
                and self.config["joint_local_head"]["mutate_geometry"]
            ),
            "mutation_joint_scores_enabled": bool(
                self.config["joint_local_head"]["enabled"]
                and self.config["joint_local_head"]["mutate_scores"]
            ),
            "mutation_supplemental_output_enabled": bool(
                self.config["supplemental_output"]["enabled"]
            ),
            # C3 is intentionally diagnostics-only.  Keeping this explicit in
            # every NPZ prevents a later active exporter from being confused
            # with the observer experiment.
            "mutation_fragment_stitch_enabled": False,
            "generic_local_geometry_refiner_enabled": bool(
                self.config["generic_local_geometry_refiner"]["enabled"]
            ),
            # C4 is deliberately observer-only.  This remains a separate
            # field from ``enabled`` so run manifests can reject any future
            # experiment that accidentally changes the frozen B6 outputs.
            "mutation_generic_local_geometry_enabled": False,
            "generic_local_geometry_diagnostics_schema": (
                "generic_mask_rgbd_local_geometry_v2"
            ),
            "generic_local_geometry_config_json": json.dumps(
                self.config["generic_local_geometry_refiner"],
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ),
            "trifusion_observer_enabled": bool(
                self.config["trifusion_observer"]["enabled"]
            ),
            "mutation_trifusion_enabled": False,
            "trifusion_diagnostics_schema": (
                "boxfusion.trifusion.occupancy_msr_observer.v1"
            ),
            "trifusion_gate_diagnostics_schema": (
                "boxfusion.trifusion.ap50_safety_observer.v1"
            ),
            "trifusion_gate_feature_names": list(
                TRIFUSION_GATE_FEATURE_NAMES
            ),
            "trifusion_config_json": json.dumps(
                self.config["trifusion_observer"],
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ),
            "residual_track_observer_enabled": bool(
                self.config["residual_track_observer"]["enabled"]
            ),
            "residual_track_observer_only": True,
            "mutation_residual_track_enabled": False,
            "residual_track_diagnostics_schema": (
                "boxfusion.residual_mask_track_observer.v1"
            ),
            "residual_track_source_mode": str(
                self.config["residual_track_observer"]["source_mode"]
            ),
            "residual_track_config_json": json.dumps(
                self.config["residual_track_observer"],
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ),
            "yidu_observer_enabled": bool(
                self.config["yidu_ablation"]["enabled"]
            ),
            "yidu_stage": str(
                self.config["yidu_ablation"]["stage"]
            ),
            "yidu_added_module": self.config["yidu_ablation"][
                "added_module"
            ],
            "yidu_modules": dict(
                self.config["yidu_ablation"]["modules"]
            ),
            "mutation_yidu_enabled": False,
            "observer_zero_write_check_enabled": bool(
                self._last_observer_zero_write_audit["enabled"]
            ),
            "observer_zero_write_verified": bool(
                self._last_observer_zero_write_audit["verified"]
            ),
            "observer_zero_write_pre_sha256": str(
                self._last_observer_zero_write_audit["pre_sha256"]
            ),
            "observer_zero_write_post_sha256": str(
                self._last_observer_zero_write_audit["post_sha256"]
            ),
            "observer_zero_write_changed_fields": list(
                self._last_observer_zero_write_audit[
                    "changed_fields"
                ]
            ),
            "yidu_attempted": int(self.stats["yidu_attempted"]),
            "yidu_valid": int(self.stats["yidu_valid"]),
            "yidu_component_candidates": int(
                self.stats["yidu_component_candidates"]
            ),
            "yidu_occupancy_candidates": int(
                self.stats["yidu_occupancy_candidates"]
            ),
            "yidu_query_candidates": int(
                self.stats["yidu_query_candidates"]
            ),
            "yidu_gate_evaluated": int(
                self.stats["yidu_gate_evaluated"]
            ),
            "yidu_gate_accepted": int(
                self.stats["yidu_gate_accepted"]
            ),
            "yidu_applied": int(self.stats["yidu_applied"]),
            "yidu_seconds": float(self.stats["yidu_seconds"]),
            "yidu_rejected": dict(
                sorted(self.stats["yidu_rejected"].items())
            ),
            "yidu_gate_feature_names": list(
                YIDU_GATE_FEATURE_NAMES
            ),
            "yidu_config_json": json.dumps(
                self.config["yidu_ablation"],
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ),
            "supplemental_geometry_refiner_enabled": bool(
                self.config["supplemental_geometry_refiner"]["enabled"]
            ),
            "mutation_supplemental_geometry_enabled": bool(
                self.config["supplemental_geometry_refiner"]["enabled"]
                and self.config["supplemental_geometry_refiner"]["mutate"]
            ),
            "supplemental_geometry_diagnostics_schema": (
                "c2_depth_occupancy_v1"
            ),
            "absorbed_recovery_enabled": bool(
                self.config["supplemental_output"][
                    "recover_absorbed_confirmed"
                ]
            ),
            "class_aware_extent_enabled": bool(
                self.config["supplemental_output"][
                    "class_aware_extent"
                ]
            ),
            "bev_duplicate_enabled": bool(
                self.config["supplemental_output"][
                    "bev_duplicate_enabled"
                ]
            ),
            "planar_duplicate_enabled": bool(
                self.config["supplemental_output"][
                    "planar_duplicate_enabled"
                ]
            ),
            "supplemental_rank_after_globals": bool(
                self.config["supplemental_output"][
                    "rank_after_globals"
                ]
            ),
            "quality_apply_to_supplemental": bool(
                self.config["quality"]["apply_to_supplemental"]
            ),
            "box_refiner_apply_scope": self.config["box_refiner"][
                "apply_scope"
            ],
            "mutation_soft_nms_enabled": bool(
                self.config["quality"]["soft_nms"]["enabled"]
            ),
            "output_minimum_extent": float(
                self.config["output_filter"]["minimum_extent"]
            ),
            "final_output_minimum_extent": (
                None
                if final_minimum_extent is None
                else float(final_minimum_extent)
            ),
            "supplemental_minimum_extent": (
                effective_supplemental_extent
            ),
            "box_refiner_point_count": int(
                self.config["box_refiner"]["point_count"]
            ),
            "box_refiner_gate_point_count": int(
                self.object_config.get("max_points_per_object", 0)
            ),
            "box_refiner_max_view_records": int(
                self.config["quality"]["max_view_records"]
            ),
            "refit_gate_min_views": int(
                self.config["refit"]["min_views"]
            ),
            "refit_gate_min_points": int(
                self.config["refit"]["min_points"]
            ),
            "refit_gate_max_center_shift_ratio": float(
                self.config["refit"]["max_center_shift_ratio"]
            ),
            "refit_gate_min_extent_ratio": float(
                self.config["refit"]["min_extent_ratio"]
            ),
            "refit_gate_max_extent_ratio": float(
                self.config["refit"]["max_extent_ratio"]
            ),
            "refit_gate_min_original_point_support": float(
                self.config["refit"][
                    "min_original_point_support"
                ]
            ),
            "refit_gate_min_candidate_point_support": float(
                self.config["refit"][
                    "min_candidate_point_support"
                ]
            ),
            "refit_gate_max_candidate_support_drop": float(
                self.config["refit"][
                    "max_candidate_support_drop"
                ]
            ),
            "refit_gate_min_reprojection_iou": float(
                self.config["refit"]["min_reprojection_iou"]
            ),
            "refit_gate_min_reprojection_improvement": float(
                self.config["refit"][
                    "min_reprojection_improvement"
                ]
            ),
            "candidate_archived_total": int(
                self.stats["candidate_archived"]
            ),
            "candidate_discarded_total": int(
                self.stats["candidate_discarded"]
            ),
            "global_memories": len(self.global_tracks),
            "refit_strategy": (
                self.config["refit"]["strategy"]
                if self.enabled
                else "disabled"
            ),
            "refit_coordinate_frame": (
                "box_local"
                if self.enabled
                and self.config["refit"]["preserve_box_orientation"]
                else ("world_aabb" if self.enabled else "disabled")
            ),
            "box_refiner_coordinate_frame": (
                self.box_refiner_coordinate_frame
                if self.enabled
                else "disabled"
            ),
            "quality_feature_geometry": (
                self.config["quality"]["feature_geometry"]
                if self.enabled
                else "disabled"
            ),
            "quality_refiner_quality_override": (
                self.config["quality"]["refiner_quality_override"]
                if self.enabled
                else None
            ),
            "joint_max_views": int(
                self.config["joint_local_head"]["max_views"]
            ),
            "joint_points_per_view": int(
                self.config["joint_local_head"]["points_per_view"]
            ),
            "joint_improvement_threshold": float(
                self.config["joint_local_head"][
                    "improvement_threshold"
                ]
            ),
            "joint_max_candidate_uncertainty": float(
                self.config["joint_local_head"][
                    "max_candidate_uncertainty"
                ]
            ),
            "joint_detector_blend": float(
                self.config["joint_local_head"]["detector_blend"]
            ),
            "joint_preserve_original_floor": bool(
                self.config["joint_local_head"][
                    "preserve_original_floor"
                ]
            ),
            "top_k_memory_enabled": bool(top_k_views),
            "top_k_views": top_k_views,
            "top_k_candidate_views": int(
                sum(
                    memory.view_candidate_count
                    for memory in global_memories
                )
            ),
            "top_k_selected_views": int(
                sum(
                    memory.selected_view_count
                    for memory in global_memories
                )
            ),
            "top_k_geometry_points": int(
                sum(
                    memory.geometry_num_points
                    for memory in global_memories
                )
            )
            if top_k_views
            else 0,
            "supplemental_top_k_candidate_views": int(
                sum(
                    memory.view_candidate_count
                    for memory in supplemental_memories
                )
            ),
            "supplemental_top_k_selected_views": int(
                sum(
                    memory.selected_view_count
                    for memory in supplemental_memories
                )
            ),
            "supplemental_top_k_geometry_points": int(
                sum(
                    memory.geometry_num_points
                    for memory in supplemental_memories
                )
            )
            if top_k_views
            else 0,
            "active_supplemental_tracks": (
                len(self.track_manager.tracks)
                if self.track_manager is not None
                else 0
            ),
            "archived_supplemental_tracks": (
                len(self.track_manager.archived_tracks)
                if self.track_manager is not None
                else 0
            ),
            "confirmed_supplemental_tracks": (
                len(
                    self.track_manager.confirmed_tracks(
                        include_archived=True
                    )
                )
                if self.track_manager is not None
                else 0
            ),
            "absorbed_recovery_records": len(
                self.absorbed_supplemental_records
            ),
            "absorbed_recovery_stored": int(
                self.stats["absorbed_recovery_stored"]
            ),
            "absorbed_recovery_considered": int(
                self.stats["absorbed_recovery_considered"]
            ),
            "absorbed_recovery_eligible": int(
                self.stats["absorbed_recovery_eligible"]
            ),
            "absorbed_recovery_output": int(
                self.stats["absorbed_recovery_output"]
            ),
            "supplemental_considered": int(
                self.stats["supplemental_considered"]
            ),
            "supplemental_rejected_graph": int(
                self.stats["supplemental_rejected_graph"]
            ),
            "supplemental_rejected_extent": int(
                self.stats["supplemental_rejected_extent"]
            ),
            "supplemental_rejected_class_extent": int(
                self.stats["supplemental_rejected_class_extent"]
            ),
            "supplemental_rejected_refined_extent": int(
                self.stats["supplemental_rejected_refined_extent"]
            ),
            "supplemental_rejected_score": int(
                self.stats["supplemental_rejected_score"]
            ),
            "supplemental_rejected_projection": int(
                self.stats["supplemental_rejected_projection"]
            ),
            "supplemental_rejected_global": int(
                self.stats["supplemental_rejected_global"]
            ),
            "supplemental_rejected_bev_global": int(
                self.stats["supplemental_rejected_bev_global"]
            ),
            "supplemental_rejected_refined_global": int(
                self.stats["supplemental_rejected_refined_global"]
            ),
            "supplemental_rejected_refined_bev_global": int(
                self.stats[
                    "supplemental_rejected_refined_bev_global"
                ]
            ),
            "supplemental_scores_rank_mapped": int(
                self.stats["supplemental_scores_rank_mapped"]
            ),
            "supplemental_output": int(
                self.stats["supplemental_output"]
            ),
            "supplemental_deduplicated": int(
                self.stats["supplemental_deduplicated"]
            ),
            "supplemental_planar_deduplicated": int(
                self.stats["supplemental_planar_deduplicated"]
            ),
            "supplemental_refined_deduplicated": int(
                self.stats["supplemental_refined_deduplicated"]
            ),
            "supplemental_b5_attempted": int(
                self.stats["supplemental_b5_attempted"]
            ),
            "supplemental_b5_accepted": int(
                self.stats["supplemental_b5_accepted"]
            ),
            "c2_attempted": int(self.stats["c2_attempted"]),
            "c2_proposed": int(self.stats["c2_proposed"]),
            "c2_verified": int(self.stats["c2_verified"]),
            "c2_applied": int(self.stats["c2_applied"]),
            "c2_seconds": float(self.stats["c2_seconds"]),
            "c2_rejections": c2_rejected,
            "c4_proposal_provider": (
                str(
                    self.config["generic_local_geometry_refiner"][
                        "secondary_proposals"
                    ].get("provider", "disabled")
                )
                if self.config["generic_local_geometry_refiner"]["enabled"]
                else "disabled"
            ),
            "c4_proposal_cache_hits": int(
                getattr(self.generic_geometry_provider, "hits", 0)
            ),
            "c4_proposal_cache_misses": int(
                getattr(self.generic_geometry_provider, "misses", 0)
            ),
            "c4_provider_calls": int(
                self.stats["c4_provider_calls"]
            ),
            "c4_provider_seconds": float(
                self.stats["c4_provider_seconds"]
            ),
            "c4_geometry_seconds": float(
                self.stats["c4_geometry_seconds"]
            ),
            "c4_refiner_seconds": float(
                self.stats["c4_refiner_seconds"]
            ),
            "c4_proposals": int(self.stats["c4_proposals"]),
            "c4_lifted": int(self.stats["c4_lifted"]),
            "c4_matched_global": int(
                self.stats["c4_matched_global"]
            ),
            "c4_attempted": int(self.stats["c4_attempted"]),
            "c4_proposed": int(self.stats["c4_proposed"]),
            "c4_verified": int(self.stats["c4_verified"]),
            "c4_applied": int(self.stats["c4_applied"]),
            "c4_rejections": c4_rejected,
            "c4_fail_open": bool(self.stats["c4_fail_open"]),
            "c4_error": self._last_c4_error,
            "c4_global_memories": len(c4_memories),
            "c4_top_k_candidate_views": int(
                sum(
                    memory.view_candidate_count
                    for memory in c4_memories
                )
            ),
            "c4_top_k_selected_views": int(
                sum(
                    memory.selected_view_count
                    for memory in c4_memories
                )
            ),
            "c4_top_k_geometry_points": int(
                sum(
                    memory.geometry_num_points
                    for memory in c4_memories
                )
            ),
            "trifusion_attempted": int(
                self.stats["trifusion_attempted"]
            ),
            "trifusion_valid": int(
                self.stats["trifusion_valid"]
            ),
            "trifusion_candidates": int(
                self.stats["trifusion_candidates"]
            ),
            "trifusion_verified": int(
                self.stats["trifusion_verified"]
            ),
            "trifusion_applied": int(
                self.stats["trifusion_applied"]
            ),
            "trifusion_seconds": float(
                self.stats["trifusion_seconds"]
            ),
            "trifusion_rejections": trifusion_rejected,
            "trifusion_gate_enabled": bool(
                self.trifusion_ap50_gate is not None
            ),
            "trifusion_gate_evaluated": int(
                self.stats["trifusion_gate_evaluated"]
            ),
            "trifusion_gate_accepted": int(
                self.stats["trifusion_gate_accepted"]
            ),
            "trifusion_gate_rejections": trifusion_gate_rejected,
            "trifusion_missing_provider_calls": int(
                self.stats["trifusion_missing_provider_calls"]
            ),
            "trifusion_missing_unmatched": int(
                self.stats["trifusion_missing_unmatched"]
            ),
            "trifusion_missing_components": int(
                self.stats["trifusion_missing_components"]
            ),
            "trifusion_missing_candidates": int(
                self.stats["trifusion_missing_candidates"]
            ),
            "trifusion_missing_errors": trifusion_missing_errors,
            "residual_track_provider_calls": int(
                self.stats["residual_track_provider_calls"]
            ),
            "residual_track_primary_unmatched": int(
                self.stats["residual_track_primary_unmatched"]
            ),
            "residual_track_secondary_unmatched": int(
                self.stats["residual_track_secondary_unmatched"]
            ),
            "residual_track_components": int(
                self.stats["residual_track_components"]
            ),
            "residual_track_candidates": int(
                self.stats["residual_track_candidates"]
            ),
            "residual_track_errors": residual_track_errors,
            "residual_track_observation_reasons": (
                residual_track_audit["observation_reasons"]
            ),
            "residual_track_association_reasons": (
                residual_track_audit["association_reasons"]
            ),
            "residual_track_associations_evaluated": (
                residual_track_audit["associations_evaluated"]
            ),
            "residual_track_associations_accepted": (
                residual_track_audit["associations_accepted"]
            ),
            "residual_track_associations_selected": (
                residual_track_audit["associations_selected"]
            ),
            "residual_track_tracks_seeded": (
                residual_track_audit["tracks_seeded"]
            ),
            "residual_track_last_provider_call_index": (
                residual_track_audit["last_provider_call_index"]
            ),
            "residual_track_last_observation_reasons": (
                residual_track_audit["last_observation_reasons"]
            ),
            "residual_track_last_association_reasons": (
                residual_track_audit["last_association_reasons"]
            ),
            "residual_track_last_associations_evaluated": (
                residual_track_audit[
                    "last_associations_evaluated"
                ]
            ),
            "residual_track_last_associations_accepted": (
                residual_track_audit[
                    "last_associations_accepted"
                ]
            ),
            "residual_track_last_associations_selected": (
                residual_track_audit[
                    "last_associations_selected"
                ]
            ),
            "residual_track_last_tracks_seeded": (
                residual_track_audit["last_tracks_seeded"]
            ),
            "residual_track_final_decisions": (
                residual_track_audit["final_decisions"]
            ),
            "residual_track_final_confirmed": (
                residual_track_audit["final_confirmed"]
            ),
            "residual_track_final_accepted": (
                residual_track_audit["final_accepted"]
            ),
            "residual_track_final_decision_reasons": (
                residual_track_audit["final_decision_reasons"]
            ),
            "residual_track_provider_nodes": (
                residual_track_provider_nodes
            ),
            "residual_track_fail_closed": bool(
                self.stats["residual_track_fail_closed"]
            ),
            "residual_track_failure": str(
                self._last_residual_track_error
            ),
            "refits_attempted": int(self.stats["refits_attempted"]),
            "refits_accepted": int(self.stats["refits_accepted"]),
            "neural_refits_accepted": int(
                self.stats["neural_refits_accepted"]
            ),
            "neural_refits_attempted": int(
                self.stats["neural_refits_attempted"]
            ),
            "neural_refits_quality_rejected": int(
                self.stats["neural_refits_quality_rejected"]
            ),
            "neural_refits_gate_rejected": int(
                self.stats["neural_refits_gate_rejected"]
            ),
            "neural_refits_invalid_orientation": int(
                self.stats["neural_refits_invalid_orientation"]
            ),
            "joint_instances": int(self.stats["joint_instances"]),
            "joint_unobserved_identity": int(
                self.stats["joint_unobserved_identity"]
            ),
            "joint_invalid_identity": int(
                self.stats["joint_invalid_identity"]
            ),
            "joint_inputs_valid": int(
                self.stats["joint_inputs_valid"]
            ),
            "joint_batches": int(self.stats["joint_batches"]),
            "joint_forward_boxes": int(
                self.stats["joint_forward_boxes"]
            ),
            "joint_improvement_rejected": int(
                self.stats["joint_improvement_rejected"]
            ),
            "joint_uncertainty_rejected": int(
                self.stats["joint_uncertainty_rejected"]
            ),
            "joint_gate_rejected": int(
                self.stats["joint_gate_rejected"]
            ),
            "joint_accepted": int(self.stats["joint_accepted"]),
            "joint_original_quality_branch": int(
                self.stats["joint_original_quality_branch"]
            ),
            "joint_candidate_quality_branch": int(
                self.stats["joint_candidate_quality_branch"]
            ),
            "joint_prepare_seconds": float(
                self.stats["joint_prepare_seconds"]
            ),
            "joint_forward_seconds": float(
                self.stats["joint_forward_seconds"]
            ),
            "joint_gate_seconds": float(
                self.stats["joint_gate_seconds"]
            ),
            "joint_rejections": joint_rejected,
            "refit_rejections": rejected,
        }

    def summary_text(self) -> str:
        summary = self.summary()
        return (
            "Online refinement summary | "
            f"keyframes={summary['keyframes']}, "
            f"provider_calls={summary['provider_calls']}, "
            f"provider={summary['proposal_provider']} "
            f"(cache={summary['proposal_cache_hits']}/"
            f"{summary['proposal_cache_misses']}), "
            f"proposals={summary['proposals']}, "
            f"lifted={summary['lifted']}, "
            f"matched_global={summary['matched_global']}, "
            "global_match(strong/weak/shadow)="
            f"{summary['strong_global_matches']}/"
            f"{summary['weak_global_matches']}/"
            f"{summary['weak_shadow_candidates']}, "
            f"candidate_updates={summary['candidate_updates']}, "
            f"candidate_clock={summary['candidate_ttl_clock']}, "
            f"active/archived="
            f"{summary['active_supplemental_tracks']}/"
            f"{summary['archived_supplemental_tracks']}, "
            "mask_graph(edges/nodes/confirmed/retired)="
            f"{summary['mask_graph_edges_accepted']}/"
            f"{summary['mask_graph_nodes_added']}/"
            f"{summary['mask_graph_confirmed_components']}/"
            f"{summary['mask_graph_retired_components']}, "
            "c1_recovery(stored/considered/eligible/output)="
            f"{summary['absorbed_recovery_stored']}/"
            f"{summary['absorbed_recovery_considered']}/"
            f"{summary['absorbed_recovery_eligible']}/"
            f"{summary['absorbed_recovery_output']}, "
            "topk(global|supp)="
            f"{summary['top_k_candidate_views']}/"
            f"{summary['top_k_selected_views']}|"
            f"{summary['supplemental_top_k_candidate_views']}/"
            f"{summary['supplemental_top_k_selected_views']} views, "
            f"refit_strategy={summary['refit_strategy']}, "
            f"refit_frame={summary['refit_coordinate_frame']}, "
            "box_refiner_frame="
            f"{summary['box_refiner_coordinate_frame']}, "
            "supp_filter="
            f"{summary['supplemental_considered']}->"
            f"{summary['supplemental_output']} "
            "(graph/extent/class_extent/refined_extent/score/proj/"
            "global/bev_global/refined_global/refined_bev_global/"
            "dedup/planar_dedup/refined_dedup="
            f"{summary['supplemental_rejected_graph']}/"
            f"{summary['supplemental_rejected_extent']}/"
            f"{summary['supplemental_rejected_class_extent']}/"
            f"{summary['supplemental_rejected_refined_extent']}/"
            f"{summary['supplemental_rejected_score']}/"
            f"{summary['supplemental_rejected_projection']}/"
            f"{summary['supplemental_rejected_global']}/"
            f"{summary['supplemental_rejected_bev_global']}/"
            f"{summary['supplemental_rejected_refined_global']}/"
            f"{summary['supplemental_rejected_refined_bev_global']}/"
            f"{summary['supplemental_deduplicated']}/"
            f"{summary['supplemental_planar_deduplicated']}/"
            f"{summary['supplemental_refined_deduplicated']}), "
            "supp_rank_mapped="
            f"{summary['supplemental_scores_rank_mapped']}, "
            "c2(applied/verified/proposed/attempted)="
            f"{summary['c2_applied']}/"
            f"{summary['c2_verified']}/"
            f"{summary['c2_proposed']}/"
            f"{summary['c2_attempted']} "
            f"(rejects={summary['c2_rejections']}), "
            "c4(verified/proposed/attempted/applied)="
            f"{summary['c4_verified']}/"
            f"{summary['c4_proposed']}/"
            f"{summary['c4_attempted']}/"
            f"{summary['c4_applied']} "
            f"(cache={summary['c4_proposal_cache_hits']}/"
            f"{summary['c4_proposal_cache_misses']}, "
            f"fail_open={int(summary['c4_fail_open'])}, "
            f"rejects={summary['c4_rejections']}), "
            "trifusion(m3_verified/m3_candidates/missing_components/"
            "missing_candidates)="
            f"{summary['trifusion_verified']}/"
            f"{summary['trifusion_candidates']}/"
            f"{summary['trifusion_missing_components']}/"
            f"{summary['trifusion_missing_candidates']} "
            f"(errors={summary['trifusion_missing_errors']}), "
            "residual_track(mode/calls/primary/secondary/components/"
            "candidates)="
            f"{summary['residual_track_source_mode']}/"
            f"{summary['residual_track_provider_calls']}/"
            f"{summary['residual_track_primary_unmatched']}/"
            f"{summary['residual_track_secondary_unmatched']}/"
            f"{summary['residual_track_components']}/"
            f"{summary['residual_track_candidates']} "
            "(assoc_selected/accepted/evaluated="
            f"{summary['residual_track_associations_selected']}/"
            f"{summary['residual_track_associations_accepted']}/"
            f"{summary['residual_track_associations_evaluated']}, "
            "final_accepted/confirmed/decisions="
            f"{summary['residual_track_final_accepted']}/"
            f"{summary['residual_track_final_confirmed']}/"
            f"{summary['residual_track_final_decisions']}, "
            "fail_closed/errors="
            f"{int(summary['residual_track_fail_closed'])}/"
            f"{summary['residual_track_errors']}), "
            "m4_gate(accepted/evaluated/enabled)="
            f"{summary['trifusion_gate_accepted']}/"
            f"{summary['trifusion_gate_evaluated']}/"
            f"{int(summary['trifusion_gate_enabled'])} "
            f"(rejects={summary['trifusion_gate_rejections']}), "
            "c3_stitch(candidates/tracks)="
            f"{summary['fragment_stitch_candidates']}/"
            f"{summary['fragment_stitch_candidate_tracks']} "
            "(invalid/fail_open="
            f"{summary['fragment_stitch_invalid_snapshots']}/"
            f"{int(summary['fragment_stitch_fail_open'])}), "
            f"refits={summary['refits_accepted']}/"
            f"{summary['refits_attempted']}, "
            f"neural_refits={summary['neural_refits_accepted']}/"
            f"{summary['neural_refits_attempted']} "
            "(quality/gate/orientation="
            f"{summary['neural_refits_quality_rejected']}/"
            f"{summary['neural_refits_gate_rejected']}/"
            f"{summary['neural_refits_invalid_orientation']}), "
            "joint="
            f"{summary['joint_accepted']}/"
            f"{summary['joint_forward_boxes']} "
            "(improve/uncertainty/gate/invalid/unobserved="
            f"{summary['joint_improvement_rejected']}/"
            f"{summary['joint_uncertainty_rejected']}/"
            f"{summary['joint_gate_rejected']}/"
            f"{summary['joint_invalid_identity']}/"
            f"{summary['joint_unobserved_identity']}), "
            "joint_s="
            f"{summary['joint_prepare_seconds']:.3f}/"
            f"{summary['joint_forward_seconds']:.3f}/"
            f"{summary['joint_gate_seconds']:.3f}, "
            f"c2_s={summary['c2_seconds']:.3f}, "
            "c4_s="
            f"{summary['c4_provider_seconds']:.3f}/"
            f"{summary['c4_geometry_seconds']:.3f}/"
            f"{summary['c4_refiner_seconds']:.3f}, "
            "c3_s="
            f"{summary['fragment_stitch_seconds']:.3f}, "
            f"provider_s={summary['provider_seconds']:.3f}, "
            f"appearance_s={summary['appearance_seconds']:.3f}, "
            f"geometry_s={summary['geometry_seconds']:.3f}"
        )


def build_online_refinement_controller(
    cfg: Mapping[str, Any],
    *,
    device: str = "cpu",
    appearance_encoder: Any = None,
) -> OnlineRefinementController:
    """Public factory used by ``demo.py``."""

    return OnlineRefinementController(
        cfg,
        device=device,
        appearance_encoder=appearance_encoder,
    )


__all__ = [
    "DEFAULT_ONLINE_REFINEMENT_CONFIG",
    "TRIFUSION_GATE_FEATURE_DIM",
    "TRIFUSION_GATE_FEATURE_NAMES",
    "FinalRefinementResult",
    "OnlineRefinementController",
    "bev_iou_and_containment",
    "bbox_iou_2d",
    "build_online_refinement_controller",
    "center_size_to_corners",
    "corners_to_center_size",
    "resolve_online_refinement_config",
    "supplemental_extent_is_valid",
]
