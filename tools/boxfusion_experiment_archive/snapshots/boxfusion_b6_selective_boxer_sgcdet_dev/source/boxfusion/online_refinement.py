"""Online orchestration for supplemental masks, RGB-D memory, and box refinement.

The controller in this module is intentionally external to BoxFusion's own
association state.  It observes the fused objects at each keyframe and changes
only the final exported detections.  This makes the feature opt-in and keeps a
disabled run on the exact legacy path.

Runtime inputs never include ground truth.  The only metric geometry source is
the sensor depth supplied by the caller.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from boxfusion.box_refiner import (
    BoxRefinerConfig,
    apply_box_residual_numpy,
    build_box_refiner,
)
from boxfusion.joint_local_head import (
    JOINT_LOCAL_HEAD_INPUT_SCHEMA,
    JOINT_LOCAL_HEAD_OUTPUT_SCHEMA,
    JOINT_VIEW_FEATURE_DIM,
    JointLocalHeadConfig,
    build_joint_local_head,
    prepare_joint_view_inputs,
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
from boxfusion.sgcdet_local_sparse_refiner import (
    SGCDET_SPARSE_REFINER_SCHEMA,
    SGCDetLocalSparseRefinerConfig,
    apply_sgcdet_sparse_residual_numpy,
    build_sgcdet_sparse_refiner,
)
from boxfusion.supplemental_proposals import (
    ProposalProvider,
    SupplementalProposal,
    build_provider,
)


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
    "sgcdet_sparse_refiner": {
        # Clean object-local adaptation of SGCDet occupancy + hard Top-K.
        # The observer prepares exact inputs without loading a checkpoint.
        "enabled": False,
        "checkpoint": None,
        "device": None,
        "max_views": 5,
        "points_per_view": 128,
        "improvement_threshold": 0.55,
        "max_candidate_uncertainty": 0.50,
        "mutate_geometry": False,
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
    },
    "output_filter": {
        "minimum_extent": 0.0,
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

    sparse = resolved["sgcdet_sparse_refiner"]
    for key in (
        "enabled",
        "mutate_geometry",
        "collect_diagnostics",
    ):
        if not isinstance(sparse[key], (bool, np.bool_)):
            raise ValueError(
                f"sgcdet_sparse_refiner.{key} must be Boolean"
            )
        sparse[key] = bool(sparse[key])
    sparse["max_views"] = _positive_int(sparse, "max_views")
    sparse["points_per_view"] = _positive_int(
        sparse, "points_per_view"
    )
    sparse["improvement_threshold"] = _finite_float(
        sparse,
        "improvement_threshold",
        lower=0.0,
        upper=1.0,
    )
    sparse["max_candidate_uncertainty"] = _finite_float(
        sparse,
        "max_candidate_uncertainty",
        lower=0.0,
        strict_lower=True,
    )
    if sparse["device"] is not None and (
        not isinstance(sparse["device"], str)
        or not sparse["device"].strip()
    ):
        raise ValueError(
            "sgcdet_sparse_refiner.device must be null or a non-empty string"
        )
    if sparse["device"] is not None:
        sparse["device"] = sparse["device"].strip()
    if not isinstance(sparse["architecture"], Mapping):
        raise TypeError(
            "sgcdet_sparse_refiner.architecture must be a mapping"
        )
    sparse_architecture = SGCDetLocalSparseRefinerConfig(
        **dict(sparse["architecture"])
    ).validated()
    if sparse_architecture.quality_feature_dim != QUALITY_FEATURE_DIM:
        raise ValueError(
            "sgcdet sparse quality_feature_dim must match the frozen B6 "
            f"quality schema ({QUALITY_FEATURE_DIM})"
        )
    if sparse_architecture.view_feature_dim != JOINT_VIEW_FEATURE_DIM:
        raise ValueError(
            "sgcdet sparse view_feature_dim must match the exact Top-K "
            f"view schema ({JOINT_VIEW_FEATURE_DIM})"
        )
    if sparse["mutate_geometry"] and not sparse["enabled"]:
        raise ValueError(
            "sgcdet sparse geometry mutation requires an enabled model"
        )

    quality = resolved["quality"]
    for key in (
        "enabled",
        "preserve_original_floor",
        "apply_to_unobserved",
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
    if not isinstance(supplemental["enabled"], (bool, np.bool_)):
        raise ValueError("supplemental_output.enabled must be Boolean")
    supplemental["enabled"] = bool(supplemental["enabled"])
    supplemental["min_confirmations"] = _positive_int(
        supplemental, "min_confirmations", minimum=2
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

    output_filter = resolved["output_filter"]
    output_filter["minimum_extent"] = _finite_float(
        output_filter, "minimum_extent", lower=0.0
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
    if sparse["enabled"] or sparse["collect_diagnostics"]:
        conflicts = []
        if refit["enabled"]:
            conflicts.append("refit")
        if box_refiner["enabled"]:
            conflicts.append("box_refiner")
        if joint["enabled"] or joint["collect_diagnostics"]:
            conflicts.append("joint_local_head")
        if supplemental["enabled"]:
            conflicts.append("supplemental_output")
        if soft_nms["enabled"]:
            conflicts.append("quality.soft_nms")
        if conflicts:
            raise ValueError(
                "sgcdet_sparse_refiner is mutually exclusive with other "
                "geometry/count mutations: " + ", ".join(conflicts)
            )
        top_k_views = resolved["object_memory"].get("top_k_views", 0)
        if (
            isinstance(top_k_views, bool)
            or not isinstance(top_k_views, (int, np.integer))
            or int(top_k_views) < int(sparse["max_views"])
        ):
            raise ValueError(
                "sgcdet_sparse_refiner requires object_memory.top_k_views "
                ">= sgcdet_sparse_refiner.max_views"
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
class _SparsePrediction:
    """Validated CPU output for one object-local sparse-refiner row."""

    prepared: _JointPreparedInstance
    center_residual: np.ndarray
    center_residual_fraction: np.ndarray
    log_dimension_residual: np.ndarray
    candidate_iou: float
    improvement_probability: float
    uncertainty: float
    coarse_occupancy_logits: np.ndarray
    coarse_occupancy_targets: np.ndarray
    occupancy_logits: np.ndarray
    occupancy_targets: np.ndarray
    selected_indices: np.ndarray
    selected_mask: np.ndarray
    selected_count: int
    selected_fraction: float
    selected_occupancy_mean: float
    selected_occupancy_maximum: float
    selected_target_fraction: float
    valid_point_count: int


@dataclass(frozen=True)
class LiftedProposal:
    proposal: SupplementalProposal
    observation: ObjectObservation
    box: np.ndarray
    depth_ratio: float
    view: ViewEvidence


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
        "candidate_updates": 0,
        "candidate_archived": 0,
        "candidate_discarded": 0,
        "supplemental_considered": 0,
        "supplemental_rejected_extent": 0,
        "supplemental_rejected_score": 0,
        "supplemental_rejected_projection": 0,
        "supplemental_rejected_global": 0,
        "supplemental_output": 0,
        "supplemental_deduplicated": 0,
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
        "sparse_instances": 0,
        "sparse_unobserved_identity": 0,
        "sparse_invalid_identity": 0,
        "sparse_inputs_valid": 0,
        "sparse_batches": 0,
        "sparse_forward_boxes": 0,
        "sparse_improvement_rejected": 0,
        "sparse_uncertainty_rejected": 0,
        "sparse_gate_rejected": 0,
        "sparse_accepted": 0,
        "sparse_prepare_seconds": 0.0,
        "sparse_forward_seconds": 0.0,
        "sparse_gate_seconds": 0.0,
        "sparse_rejected": Counter(),
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
        appearance_encoder: Any = None,
        box_refiner: Any = None,
        quality_scorer: Any = None,
        joint_local_head: Any = None,
        sgcdet_sparse_refiner: Any = None,
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
        self.appearance_encoder = None
        self.box_refiner = None
        self.box_refiner_coordinate_frame = "world_aabb"
        self.quality_scorer = None
        self.joint_local_head = None
        self._last_joint_runtime: Dict[int, Dict[str, Any]] = {}
        self.sgcdet_sparse_refiner = None
        self._last_sparse_runtime: Dict[int, Dict[str, Any]] = {}
        self.object_config: Dict[str, Any] = {}
        self.track_manager: Optional[CandidateTrackManager] = None
        self.global_tracks: Dict[int, GlobalEvidence] = {}
        self.supplemental_metadata: Dict[int, SupplementalEvidence] = {}
        self.keyframe_count = 0
        self.scene_id: Optional[str] = None
        self.stats: Dict[str, Any] = _empty_runtime_stats()
        if not self.enabled:
            return

        proposals_cfg = self.config["supplemental_proposals"]
        provider_device = proposals_cfg.get("device", self.device)
        self.provider = (
            provider
            if provider is not None
            else build_provider(proposals_cfg, str(provider_device))
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

        sparse_cfg = self.config["sgcdet_sparse_refiner"]
        sparse_architecture = SGCDetLocalSparseRefinerConfig(
            **dict(sparse_cfg["architecture"])
        ).validated()
        if sparse_cfg["enabled"]:
            if sgcdet_sparse_refiner is not None:
                injected_config = getattr(
                    sgcdet_sparse_refiner, "config", None
                )
                if (
                    injected_config is None
                    or not hasattr(injected_config, "architecture_dict")
                    or injected_config.architecture_dict()
                    != sparse_architecture.architecture_dict()
                ):
                    raise ValueError(
                        "injected sgcdet sparse-refiner architecture does "
                        "not match sgcdet_sparse_refiner.architecture"
                    )
                self.sgcdet_sparse_refiner = sgcdet_sparse_refiner
            else:
                self.sgcdet_sparse_refiner = build_sgcdet_sparse_refiner(
                    enabled=True,
                    checkpoint_path=sparse_cfg["checkpoint"],
                    config=sparse_architecture,
                    device=sparse_cfg["device"] or self.device,
                )

    def reset_scene(self, scene_id: str) -> None:
        """Clear all geometry/track state while retaining loaded models."""

        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError("scene_id must be a non-empty string")
        self.scene_id = scene_id.strip()
        self.keyframe_count = 0
        self.global_tracks.clear()
        self.supplemental_metadata.clear()
        self._last_joint_runtime.clear()
        self._last_sparse_runtime.clear()
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

    def _match_to_globals(
        self,
        lifted: Sequence[LiftedProposal],
        boxes: np.ndarray,
        intrinsics: np.ndarray,
        camera_to_world: np.ndarray,
    ) -> Dict[int, int]:
        cfg = self.config["matching"]
        candidates: List[Tuple[float, float, float, int, int]] = []
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
                candidates.append(
                    (
                        -float(score),
                        -float(overlap),
                        -float(projection_iou),
                        proposal_index,
                        global_index,
                    )
                )
        candidates.sort()
        assignments: Dict[int, int] = {}
        used_globals = set()
        for _, _, _, proposal_index, global_index in candidates:
            if proposal_index in assignments or global_index in used_globals:
                continue
            assignments[proposal_index] = global_index
            used_globals.add(global_index)
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

    def _absorb_candidate_track(
        self,
        evidence: GlobalEvidence,
        item: LiftedProposal,
        *,
        frame_index: int,
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
        _, archived, track_id = min(candidates)
        source = (
            self.track_manager.archived_tracks
            if archived
            else self.track_manager.tracks
        )
        track = source.pop(track_id)
        metadata = self.supplemental_metadata.pop(track_id, None)
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

    def _record_candidate_retirement(self, result: Any) -> None:
        """Keep archived metadata and discard only unconfirmed-track metadata."""

        archived = set(result.archived_track_ids)
        for discarded_id in result.discarded_track_ids:
            self.supplemental_metadata.pop(discarded_id, None)
        # Defensive compatibility: an implementation returning only
        # ``expired_track_ids`` must not leak metadata for discarded tracks.
        for expired_id in result.expired_track_ids:
            if expired_id not in archived:
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
        result = self.track_manager.update(
            [],
            frame_index,
            lifecycle_step=lifecycle_step,
        )
        self._record_candidate_retirement(result)

    def _update_candidates(
        self,
        unmatched: Sequence[LiftedProposal],
        *,
        frame_index: int,
        lifecycle_step: int,
    ) -> None:
        if self.track_manager is None:
            return
        result = self.track_manager.update(
            [item.observation for item in unmatched],
            frame_index,
            lifecycle_step=lifecycle_step,
        )
        self._record_candidate_retirement(result)
        for local_index, track_id in result.assignments.items():
            item = unmatched[local_index]
            metadata = self.supplemental_metadata.setdefault(
                track_id, SupplementalEvidence(track_id)
            )
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
        started = time.perf_counter()
        batches = self.provider.predict(
            [image],
            frame_ids=[
                f"{self.scene_id}:"
                + (
                    cache_frame_id.strip()
                    if cache_frame_id is not None
                    else f"{int(frame_id):06d}"
                )
            ],
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
        assignments = self._match_to_globals(
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
            self._absorb_candidate_track(
                evidence, item, frame_index=frame_index
            )
        unmatched = [
            item
            for index, item in enumerate(lifted)
            if index not in assignments
        ]
        self._update_candidates(
            unmatched,
            frame_index=frame_index,
            lifecycle_step=lifecycle_step,
        )
        self.stats["matched_global"] += len(assignments)

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

    def _run_sgcdet_sparse_refiner_batch(
        self,
        *,
        corners: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        stable_ids: np.ndarray,
    ) -> Dict[int, _SparsePrediction]:
        """Prepare all valid local volumes and run one sparse-head batch.

        ``collect_diagnostics`` without ``enabled`` is a true observer: the
        exact tensors are built and serialized, but PyTorch and a checkpoint
        are not required. Invalid or unobserved rows remain exact identity.
        """

        self._last_sparse_runtime.clear()
        cfg = self.config["sgcdet_sparse_refiner"]
        if not (cfg["enabled"] or cfg["collect_diagnostics"]):
            return {}
        prepared_rows: List[_JointPreparedInstance] = []
        self.stats["sparse_instances"] += int(len(boxes))
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
                self.stats["sparse_unobserved_identity"] += 1
                self._last_sparse_runtime[stable_id] = {
                    "input_valid": False,
                    "reason": "sparse_unobserved",
                }
                continue
            geometry_views = evidence.memory.geometry_unique_view_count
            if not evidence.memory.top_k_enabled:
                geometry_views += evidence.stats.absorbed_views
            if geometry_views < minimum_views:
                self.stats["sparse_invalid_identity"] += 1
                self.stats["sparse_rejected"]["views"] += 1
                self._last_sparse_runtime[stable_id] = {
                    "input_valid": False,
                    "reason": "sparse_views",
                }
                continue
            if evidence.memory.geometry_num_points < minimum_points:
                self.stats["sparse_invalid_identity"] += 1
                self.stats["sparse_rejected"]["points"] += 1
                self._last_sparse_runtime[stable_id] = {
                    "input_valid": False,
                    "reason": "sparse_points",
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
                self.stats["sparse_invalid_identity"] += 1
                self.stats["sparse_rejected"]["orientation"] += 1
                self._last_sparse_runtime[stable_id] = {
                    "input_valid": False,
                    "reason": "sparse_orientation",
                }
                continue
            if int(np.count_nonzero(view_inputs.view_mask)) < minimum_views:
                self.stats["sparse_invalid_identity"] += 1
                self.stats["sparse_rejected"]["valid_views"] += 1
                self._last_sparse_runtime[stable_id] = {
                    "input_valid": False,
                    "reason": "sparse_valid_views",
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
            prepared = _JointPreparedInstance(
                index=int(index),
                stable_id=stable_id,
                evidence=evidence,
                original_box=np.asarray(original, dtype=np.float32).copy(),
                original_corners=np.asarray(
                    original_corners, dtype=np.float32
                ).copy(),
                detector_score=float(detector_score),
                quality_mapping=quality_mapping,
                quality_features=np.array(
                    quality_feature_vector(quality_mapping),
                    dtype=np.float32,
                    copy=True,
                ),
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
            self._last_sparse_runtime[stable_id] = {
                "input_valid": True,
                "reason": (
                    "sparse_pending"
                    if cfg["enabled"]
                    else "sparse_observer"
                ),
                "prepared": prepared,
            }
        self.stats["sparse_prepare_seconds"] += (
            time.perf_counter() - started
        )
        self.stats["sparse_inputs_valid"] += len(prepared_rows)
        if not cfg["enabled"] or not prepared_rows:
            return {}
        if self.sgcdet_sparse_refiner is None:
            raise RuntimeError(
                "enabled sgcdet_sparse_refiner has no strictly loaded model"
            )

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
            parameter = next(self.sgcdet_sparse_refiner.parameters())
        except StopIteration as error:
            raise RuntimeError(
                "sgcdet sparse refiner must expose a parameter"
            ) from error
        forward_started = time.perf_counter()
        with torch.no_grad():
            output = self.sgcdet_sparse_refiner(
                torch.from_numpy(points_local).to(
                    device=parameter.device, dtype=parameter.dtype
                ),
                torch.from_numpy(point_mask).to(parameter.device),
                torch.from_numpy(local_boxes).to(
                    device=parameter.device, dtype=parameter.dtype
                ),
                torch.from_numpy(quality_features).to(
                    device=parameter.device, dtype=parameter.dtype
                ),
                torch.from_numpy(view_features).to(
                    device=parameter.device, dtype=parameter.dtype
                ),
                torch.from_numpy(view_mask).to(parameter.device),
            )
        self.stats["sparse_forward_seconds"] += (
            time.perf_counter() - forward_started
        )
        self.stats["sparse_batches"] += 1
        self.stats["sparse_forward_boxes"] += len(prepared_rows)
        required = {
            "center_residual",
            "center_residual_fraction",
            "log_dimension_residual",
            "candidate_iou",
            "improvement_probability",
            "uncertainty",
            "coarse_occupancy_logits",
            "coarse_occupancy_targets",
            "occupancy_logits",
            "occupancy_targets",
            "selected_indices",
            "selected_mask",
            "selected_stats",
        }
        if not isinstance(output, Mapping) or set(output) != required:
            raise RuntimeError(
                "sgcdet sparse-refiner output schema does not match "
                f"{SGCDET_SPARSE_REFINER_SCHEMA}"
            )

        def numpy_output(name: str) -> np.ndarray:
            value = output[name]
            if not torch.is_tensor(value):
                raise RuntimeError(
                    f"sparse-refiner output {name} must be a tensor"
                )
            array = value.detach().float().cpu().numpy()
            if not np.isfinite(array).all():
                raise RuntimeError(
                    f"sparse-refiner output {name} must be finite"
                )
            return array

        batch = len(prepared_rows)
        center_residual = numpy_output("center_residual")
        center_fraction = numpy_output("center_residual_fraction")
        log_dimensions = numpy_output("log_dimension_residual")
        candidate_iou = numpy_output("candidate_iou")
        improvement = numpy_output("improvement_probability")
        uncertainty = numpy_output("uncertainty")
        coarse_logits = numpy_output("coarse_occupancy_logits")
        coarse_targets = numpy_output("coarse_occupancy_targets")
        occupancy_logits = numpy_output("occupancy_logits")
        occupancy_targets = numpy_output("occupancy_targets")
        selected_indices_tensor = output["selected_indices"]
        selected_mask_tensor = output["selected_mask"]
        if (
            not torch.is_tensor(selected_indices_tensor)
            or selected_indices_tensor.dtype != torch.long
            or not torch.is_tensor(selected_mask_tensor)
            or selected_mask_tensor.dtype != torch.bool
        ):
            raise RuntimeError(
                "selected_indices/mask must be long/Boolean tensors"
            )
        selected_indices = (
            selected_indices_tensor.detach().cpu().numpy()
        )
        selected_mask = selected_mask_tensor.detach().cpu().numpy()
        architecture = self.sgcdet_sparse_refiner.config
        coarse_count = int(np.prod(architecture.coarse_grid_size))
        fine_count = int(np.prod(architecture.fine_grid_size))
        selected_count = int(architecture.selected_token_count)
        expected_shapes = {
            "center_residual": (center_residual, (batch, 3)),
            "center_residual_fraction": (center_fraction, (batch, 3)),
            "log_dimension_residual": (log_dimensions, (batch, 3)),
            "candidate_iou": (candidate_iou, (batch,)),
            "improvement_probability": (improvement, (batch,)),
            "uncertainty": (uncertainty, (batch,)),
            "coarse_occupancy_logits": (
                coarse_logits,
                (batch, coarse_count),
            ),
            "coarse_occupancy_targets": (
                coarse_targets,
                (batch, coarse_count),
            ),
            "occupancy_logits": (occupancy_logits, (batch, fine_count)),
            "occupancy_targets": (
                occupancy_targets,
                (batch, fine_count),
            ),
            "selected_indices": (
                selected_indices,
                (batch, selected_count),
            ),
            "selected_mask": (selected_mask, (batch, fine_count)),
        }
        for name, (value, shape) in expected_shapes.items():
            if value.shape != shape:
                raise RuntimeError(
                    f"sparse-refiner output {name} must have shape {shape}, "
                    f"received {value.shape}"
                )
        for name, value in (
            ("candidate_iou", candidate_iou),
            ("improvement_probability", improvement),
            ("uncertainty", uncertainty),
            ("coarse_occupancy_targets", coarse_targets),
            ("occupancy_targets", occupancy_targets),
        ):
            if ((value < 0.0) | (value > 1.0)).any():
                raise RuntimeError(
                    f"sparse-refiner output {name} must lie in [0,1]"
                )
        if not np.all(selected_mask.sum(axis=1) == selected_count):
            raise RuntimeError("sparse hard Top-K count is inconsistent")
        if (
            np.abs(center_fraction)
            > float(architecture.max_center_fraction) + 1e-6
        ).any():
            raise RuntimeError("sparse center residual exceeds its bound")
        if (
            np.abs(log_dimensions)
            > float(architecture.max_log_dimension_residual) + 1e-6
        ).any():
            raise RuntimeError("sparse size residual exceeds its bound")
        if not np.allclose(
            center_residual,
            center_fraction * local_boxes[:, 3:6],
            atol=1e-6,
            rtol=1e-5,
        ):
            raise RuntimeError(
                "sparse center residual is inconsistent with its fraction"
            )
        selected_stats = output["selected_stats"]
        required_stats = {
            "count",
            "fraction",
            "occupancy_mean",
            "occupancy_maximum",
            "target_fraction",
            "valid_point_count",
        }
        if not isinstance(selected_stats, Mapping) or set(
            selected_stats
        ) != required_stats:
            raise RuntimeError("invalid sparse selected_stats schema")

        def stat_array(name: str) -> np.ndarray:
            value = selected_stats[name]
            if not torch.is_tensor(value) or value.shape != (batch,):
                raise RuntimeError(
                    f"selected_stats.{name} must have shape [{batch}]"
                )
            result = value.detach().float().cpu().numpy()
            if not np.isfinite(result).all():
                raise RuntimeError(
                    f"selected_stats.{name} must be finite"
                )
            return result

        stat_count = stat_array("count")
        stat_fraction = stat_array("fraction")
        stat_mean = stat_array("occupancy_mean")
        stat_maximum = stat_array("occupancy_maximum")
        stat_target = stat_array("target_fraction")
        stat_valid_points = stat_array("valid_point_count")
        if not np.allclose(
            stat_count, float(selected_count), atol=0.0, rtol=0.0
        ):
            raise RuntimeError("selected_stats.count disagrees with hard Top-K")
        if not np.allclose(
            stat_fraction,
            float(architecture.topk_fraction),
            atol=1e-7,
            rtol=0.0,
        ):
            raise RuntimeError(
                "selected_stats.fraction disagrees with the architecture"
            )
        for name, value in (
            ("occupancy_mean", stat_mean),
            ("occupancy_maximum", stat_maximum),
            ("target_fraction", stat_target),
        ):
            if ((value < 0.0) | (value > 1.0)).any():
                raise RuntimeError(
                    f"selected_stats.{name} must lie in [0,1]"
                )
        if (stat_valid_points < 0.0).any():
            raise RuntimeError(
                "selected_stats.valid_point_count must be non-negative"
            )
        expected_masks = np.zeros_like(selected_mask, dtype=bool)
        for row_index, row_indices in enumerate(selected_indices):
            if (
                (row_indices < 0).any()
                or (row_indices >= fine_count).any()
                or np.unique(row_indices).size != selected_count
            ):
                raise RuntimeError(
                    "selected_indices must be unique in-range voxel indices"
                )
            expected_masks[row_index, row_indices] = True
        if not np.array_equal(selected_mask, expected_masks):
            raise RuntimeError(
                "selected_mask does not exactly match selected_indices"
            )
        predictions: Dict[int, _SparsePrediction] = {}
        for row_index, prepared in enumerate(prepared_rows):
            prediction = _SparsePrediction(
                prepared=prepared,
                center_residual=center_residual[row_index].copy(),
                center_residual_fraction=center_fraction[row_index].copy(),
                log_dimension_residual=log_dimensions[row_index].copy(),
                candidate_iou=float(candidate_iou[row_index]),
                improvement_probability=float(improvement[row_index]),
                uncertainty=float(uncertainty[row_index]),
                coarse_occupancy_logits=coarse_logits[row_index].copy(),
                coarse_occupancy_targets=coarse_targets[row_index].copy(),
                occupancy_logits=occupancy_logits[row_index].copy(),
                occupancy_targets=occupancy_targets[row_index].copy(),
                selected_indices=selected_indices[row_index].copy(),
                selected_mask=selected_mask[row_index].copy(),
                selected_count=int(round(float(stat_count[row_index]))),
                selected_fraction=float(stat_fraction[row_index]),
                selected_occupancy_mean=float(stat_mean[row_index]),
                selected_occupancy_maximum=float(
                    stat_maximum[row_index]
                ),
                selected_target_fraction=float(stat_target[row_index]),
                valid_point_count=int(
                    round(float(stat_valid_points[row_index]))
                ),
            )
            predictions[prepared.index] = prediction
            self._last_sparse_runtime[prepared.stable_id].update(
                {
                    "reason": "sparse_inferred",
                    "prediction": prediction,
                }
            )
        return predictions

    def _apply_sgcdet_sparse_prediction(
        self, prediction: _SparsePrediction
    ) -> Tuple[np.ndarray, np.ndarray, bool, str]:
        """Gate one sparse candidate while preserving B6 ranking state."""

        cfg = self.config["sgcdet_sparse_refiner"]
        prepared = prediction.prepared
        refined = prepared.original_box.copy()
        corners = prepared.original_corners.copy()
        accepted = False
        reason = "sparse_geometry_disabled"
        if cfg["mutate_geometry"]:
            if prediction.improvement_probability < float(
                cfg["improvement_threshold"]
            ):
                reason = "sparse_improvement"
                self.stats["sparse_improvement_rejected"] += 1
                self.stats["sparse_rejected"]["improvement"] += 1
            elif prediction.uncertainty > float(
                cfg["max_candidate_uncertainty"]
            ):
                reason = "sparse_uncertainty"
                self.stats["sparse_uncertainty_rejected"] += 1
                self.stats["sparse_rejected"]["uncertainty"] += 1
            else:
                local_candidate = apply_sgcdet_sparse_residual_numpy(
                    prepared.local_box,
                    prediction.center_residual,
                    prediction.log_dimension_residual,
                    config=self.sgcdet_sparse_refiner.config,
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
                self.stats["sparse_gate_seconds"] += (
                    time.perf_counter() - gate_started
                )
                if accepted:
                    refined = candidate_world_box
                    corners = candidate_corners
                    reason = "sparse_accepted"
                    self.stats["sparse_accepted"] += 1
                else:
                    reason = f"sparse_{gate_reason}"
                    self.stats["sparse_gate_rejected"] += 1
                    self.stats["sparse_rejected"][gate_reason] += 1
        self._last_sparse_runtime[prepared.stable_id].update(
            {
                "reason": reason,
                "accepted": bool(accepted),
            }
        )
        return refined, corners, bool(accepted), reason

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

    def _supplemental_outputs(
        self,
        global_boxes: np.ndarray,
    ) -> List[
        Tuple[
            np.ndarray,
            float,
            int,
            Optional[str],
            np.ndarray,
            ObjectGeometryMemory,
        ]
    ]:
        output = []
        deduplicated = 0
        cfg = self.config["supplemental_output"]
        for key in (
            "supplemental_considered",
            "supplemental_rejected_extent",
            "supplemental_rejected_score",
            "supplemental_rejected_projection",
            "supplemental_rejected_global",
            "supplemental_output",
        ):
            self.stats[key] = 0
        if not cfg["enabled"] or self.track_manager is None:
            self.stats["supplemental_deduplicated"] = 0
            return output
        candidates = []
        for track in self.track_manager.confirmed_tracks(
            include_archived=True
        ):
            metadata = self.supplemental_metadata.get(track.track_id)
            if metadata is None or track.view_count < cfg["min_confirmations"]:
                continue
            track_box = track.memory.aabb
            if track_box is None:
                continue
            self.stats["supplemental_considered"] += 1
            box = np.concatenate(track_box).astype(np.float32)
            minimum_extent = float(
                self.config["output_filter"]["minimum_extent"]
            )
            if minimum_extent > 0.0 and np.any(
                box[3:6] < minimum_extent
            ):
                self.stats["supplemental_rejected_extent"] += 1
                continue
            detector_score = metadata.stats.mean_score
            if detector_score < cfg["min_score"]:
                self.stats["supplemental_rejected_score"] += 1
                continue
            mapping = self._quality_mapping(
                original_box=box,
                final_box=box,
                detector_score=detector_score,
                memory=track.memory,
                stats=metadata.stats,
                supplemental=True,
                refiner_quality=0.5,
            )
            if mapping["projection_iou"] < cfg["min_projection_iou"]:
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
            score = self._score(detector_score, mapping, observed=True)
            if score < cfg["min_score"]:
                self.stats["supplemental_rejected_score"] += 1
                continue
            candidates.append(
                (
                    -score,
                    -track.view_count,
                    track.track_id,
                    box,
                    score,
                    -(track.track_id + 1),
                    metadata.stats.label,
                    quality_feature_vector(mapping),
                    track.memory,
                )
            )

        # A long-lived archive can contain a new track for an object that
        # re-entered the camera after its active TTL elapsed.  Keep the
        # strongest representative for high-overlap duplicates while leaving
        # neighbouring/nested objects untouched at the conservative threshold.
        accepted_boxes = []
        for (
            _,
            _,
            _,
            box,
            score,
            supplemental_id,
            label,
            features,
            memory,
        ) in sorted(candidates):
            if accepted_boxes and max(
                aabb_iou(
                    box[:3],
                    box[3:6],
                    accepted[:3],
                    accepted[3:6],
                )
                for accepted in accepted_boxes
            ) >= cfg["drop_if_supplemental_iou"]:
                deduplicated += 1
                continue
            accepted_boxes.append(box)
            output.append(
                (
                    box,
                    score,
                    supplemental_id,
                    label,
                    features,
                    memory,
                )
            )
        self.stats["supplemental_deduplicated"] = deduplicated
        self.stats["supplemental_output"] = len(output)
        return output

    def finalize(
        self,
        *,
        global_corners: Any,
        global_scores: Any,
        stable_ids: Any,
        scene_id: Optional[str] = None,
    ) -> FinalRefinementResult:
        """Return refined and supplemental detections without mutating inputs."""

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
        sparse_cfg = self.config["sgcdet_sparse_refiner"]
        sparse_route = bool(
            sparse_cfg["enabled"] or sparse_cfg["collect_diagnostics"]
        )
        # Sparse inference deliberately precedes the joint branch.  The two
        # routes are config-mutually-exclusive, but keeping the ordering
        # explicit prevents a future joint-head edit from silently changing
        # the frozen B6 inputs consumed by the sparse observer/active pair.
        sparse_predictions = self._run_sgcdet_sparse_refiner_batch(
            corners=corners_input,
            boxes=boxes,
            scores=scores,
            stable_ids=ids,
        )
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
            if sparse_route:
                sparse_prediction = sparse_predictions.get(index)
                if sparse_prediction is None:
                    # Observer, unobserved, and invalid-input rows are strict
                    # geometry identity.  Their B6 score is still computed
                    # below from the frozen original-geometry feature path.
                    refined_oriented_corners = corners_input[index].copy()
                    runtime = self._last_sparse_runtime.get(
                        int(stable_id), {}
                    )
                    refit_reason = str(
                        runtime.get("reason", "sparse_invalid")
                    )
                else:
                    (
                        refined,
                        refined_oriented_corners,
                        robust_accepted,
                        refit_reason,
                    ) = self._apply_sgcdet_sparse_prediction(
                        sparse_prediction
                    )
                # The sparse module is geometry-only.  Both observer and
                # active runs use the exact frozen B6 mapping from the
                # original BoxFusion geometry, never candidate IoU or sparse
                # uncertainty, so score/order changes cannot be attributed to
                # the local refiner.
                mapping = self._quality_mapping(
                    original_box=original,
                    final_box=original,
                    detector_score=float(detector_score),
                    memory=(
                        evidence.memory if evidence is not None else None
                    ),
                    stats=(
                        evidence.stats if evidence is not None else None
                    ),
                    supplemental=False,
                    refiner_quality=0.5,
                )
                score = self._score(
                    float(detector_score),
                    mapping,
                    observed=evidence is not None
                    and evidence.memory.observation_count > 0,
                )
            elif joint_enabled:
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
                    neural_reason = "neural_disabled"
                    if self.box_refiner_coordinate_frame == "box_local":
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
                    else:
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
                or self.config["sgcdet_sparse_refiner"]["enabled"]
                or self.config["sgcdet_sparse_refiner"][
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

        minimum_extent = float(
            self.config["output_filter"]["minimum_extent"]
        )
        eligible_global_boxes = np.asarray(
            final_boxes, dtype=np.float32
        )
        if len(eligible_global_boxes) and minimum_extent > 0.0:
            eligible_global_boxes = eligible_global_boxes[
                np.all(
                    eligible_global_boxes[:, 3:6] >= minimum_extent,
                    axis=1,
                )
            ]
        for (
            box,
            score,
            supplemental_id,
            label,
            features,
            memory,
        ) in self._supplemental_outputs(eligible_global_boxes):
            supplemental_corners = aabb_corners(box[:3], box[3:6])
            final_corners.append(supplemental_corners)
            final_boxes.append(box)
            final_scores.append(score)
            source_indices.append(-1)
            result_ids.append(supplemental_id)
            labels.append(label)
            feature_rows.append(features)
            memories.append(memory)
            refit_original_boxes.append(box.copy())
            refit_original_corners.append(supplemental_corners.copy())
            refit_applied.append(False)
            refit_reasons.append("supplemental")
            refit_changed_axes.append(np.zeros(3, dtype=bool))
            refit_boundary_delta.append(
                np.zeros(6, dtype=np.float32)
            )
            refit_local_original_boxes.append(
                np.full(6, np.nan, dtype=np.float32)
            )
            refit_local_candidate_boxes.append(
                np.full(6, np.nan, dtype=np.float32)
            )
            refit_local_basis.append(
                np.full((3, 3), np.nan, dtype=np.float32)
            )
            refit_local_frame_valid.append(False)

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

        if len(boxes_array) and minimum_extent > 0.0:
            # The sparse route is geometry-only and must conserve the frozen
            # B6 detection set.  Eligibility therefore follows the original
            # BoxFusion extent, not a learned candidate that could otherwise
            # add/drop a row at the minimum-extent boundary.
            extent_boxes = (
                refit_original_array if sparse_route else boxes_array
            )
            valid_output = np.all(
                extent_boxes[:, 3:6] >= minimum_extent, axis=1
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

        if joint_enabled or sparse_route:
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
                    (
                        "sgcdet_sparse_refiner"
                        if sparse_route
                        else "joint_local_head"
                    )
                    + " violated detection count/order/stable ID "
                    "conservation"
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
            self._dump_diagnostics(result, memories, selected_scene)
        return result

    def _dump_diagnostics(
        self,
        result: FinalRefinementResult,
        memories: Sequence[Optional[ObjectGeometryMemory]],
        scene_id: str,
    ) -> None:
        cfg = self.config["diagnostics"]
        if not cfg["enabled"] or not cfg["dump_track_memory"]:
            return
        root = Path(cfg["root"])
        root.mkdir(parents=True, exist_ok=True)
        point_count = int(cfg["point_count"])
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
        sparse_cfg = self.config["sgcdet_sparse_refiner"]
        sparse_architecture = (
            self.sgcdet_sparse_refiner.config
            if self.sgcdet_sparse_refiner is not None
            else SGCDetLocalSparseRefinerConfig(
                **dict(sparse_cfg["architecture"])
            ).validated()
        )
        sparse_max_views = int(sparse_cfg["max_views"])
        sparse_points_per_view = int(sparse_cfg["points_per_view"])
        sparse_view_feature_dim = int(
            sparse_architecture.view_feature_dim
        )
        sparse_coarse_count = int(
            np.prod(sparse_architecture.coarse_grid_size)
        )
        sparse_fine_count = int(
            np.prod(sparse_architecture.fine_grid_size)
        )
        sparse_selected_count = int(
            sparse_architecture.selected_token_count
        )
        sparse_points_local = np.zeros(
            (
                len(observed_indices),
                sparse_max_views,
                sparse_points_per_view,
                3,
            ),
            dtype=np.float32,
        )
        sparse_point_mask = np.zeros(
            (
                len(observed_indices),
                sparse_max_views,
                sparse_points_per_view,
            ),
            dtype=bool,
        )
        sparse_view_features = np.zeros(
            (
                len(observed_indices),
                sparse_max_views,
                sparse_view_feature_dim,
            ),
            dtype=np.float32,
        )
        sparse_view_mask = np.zeros(
            (len(observed_indices), sparse_max_views), dtype=bool
        )
        sparse_local_boxes = np.full(
            (len(observed_indices), 6), np.nan, dtype=np.float32
        )
        sparse_quality_features = np.zeros(
            (len(observed_indices), QUALITY_FEATURE_DIM),
            dtype=np.float32,
        )
        sparse_frame_center = np.full(
            (len(observed_indices), 3), np.nan, dtype=np.float64
        )
        sparse_frame_basis = np.full(
            (len(observed_indices), 3, 3), np.nan, dtype=np.float64
        )
        sparse_input_valid = np.zeros(
            len(observed_indices), dtype=bool
        )
        sparse_output_valid = np.zeros(
            len(observed_indices), dtype=bool
        )
        sparse_runtime_reason = np.full(
            len(observed_indices), "sparse_not_run", dtype="<U96"
        )
        sparse_accepted = np.zeros(len(observed_indices), dtype=bool)
        sparse_center_residual = np.full(
            (len(observed_indices), 3), np.nan, dtype=np.float32
        )
        sparse_center_residual_fraction = np.full(
            (len(observed_indices), 3), np.nan, dtype=np.float32
        )
        sparse_log_dimension_residual = np.full(
            (len(observed_indices), 3), np.nan, dtype=np.float32
        )
        sparse_candidate_iou = np.full(
            len(observed_indices), np.nan, dtype=np.float32
        )
        sparse_improvement_probability = np.full(
            len(observed_indices), np.nan, dtype=np.float32
        )
        sparse_uncertainty = np.full(
            len(observed_indices), np.nan, dtype=np.float32
        )
        sparse_coarse_occupancy_logits = np.full(
            (len(observed_indices), sparse_coarse_count),
            np.nan,
            dtype=np.float32,
        )
        sparse_coarse_occupancy_targets = np.zeros(
            (len(observed_indices), sparse_coarse_count),
            dtype=np.float32,
        )
        sparse_occupancy_logits = np.full(
            (len(observed_indices), sparse_fine_count),
            np.nan,
            dtype=np.float32,
        )
        sparse_occupancy_targets = np.zeros(
            (len(observed_indices), sparse_fine_count),
            dtype=np.float32,
        )
        sparse_selected_indices = np.full(
            (len(observed_indices), sparse_selected_count),
            -1,
            dtype=np.int64,
        )
        sparse_selected_mask = np.zeros(
            (len(observed_indices), sparse_fine_count), dtype=bool
        )
        sparse_selected_stats = np.full(
            (len(observed_indices), 6), np.nan, dtype=np.float32
        )
        sparse_original_boxes = result.refit_original_boxes[
            observed_indices
        ].astype(np.float32, copy=True)
        sparse_original_corners = result.refit_original_corners[
            observed_indices
        ].astype(np.float32, copy=True)
        sparse_active_boxes = result.boxes[observed_indices].astype(
            np.float32, copy=True
        )
        sparse_active_corners = result.corners[observed_indices].astype(
            np.float32, copy=True
        )
        sparse_final_scores = result.scores[observed_indices].astype(
            np.float32, copy=True
        )
        sparse_detector_scores = np.full(
            len(observed_indices), np.nan, dtype=np.float32
        )
        sparse_pair_keys = np.full(
            len(observed_indices), "", dtype="<U160"
        )
        for output_index, result_index in enumerate(observed_indices):
            memory = memories[int(result_index)]
            assert memory is not None
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
            if evidence is not None:
                for view_index, view in enumerate(
                    evidence.stats.view_records[
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
            sparse_pair_keys[output_index] = (
                f"{scene_id}:{int(result.source_indices[int(result_index)])}:"
                f"{stable_id}"
            )
            if evidence is not None:
                sparse_detector_scores[output_index] = float(
                    evidence.detector_score
                )
            sparse_runtime = self._last_sparse_runtime.get(stable_id)
            if sparse_runtime is not None:
                sparse_runtime_reason[output_index] = str(
                    sparse_runtime.get("reason", "sparse_not_run")
                )
                sparse_accepted[output_index] = bool(
                    sparse_runtime.get("accepted", False)
                )
            sparse_prepared = (
                sparse_runtime.get("prepared")
                if sparse_runtime is not None
                and bool(sparse_runtime.get("input_valid", False))
                else None
            )
            if isinstance(sparse_prepared, _JointPreparedInstance):
                sparse_points_local[output_index] = (
                    sparse_prepared.points_local
                )
                sparse_point_mask[output_index] = (
                    sparse_prepared.point_mask
                )
                sparse_view_features[output_index] = (
                    sparse_prepared.view_features
                )
                sparse_view_mask[output_index] = (
                    sparse_prepared.view_mask
                )
                sparse_local_boxes[output_index] = (
                    sparse_prepared.local_box
                )
                sparse_quality_features[output_index] = (
                    sparse_prepared.quality_features
                )
                sparse_frame_center[output_index] = (
                    sparse_prepared.frame_center
                )
                sparse_frame_basis[output_index] = (
                    sparse_prepared.frame_basis
                )
                sparse_input_valid[output_index] = True
            sparse_prediction = (
                sparse_runtime.get("prediction")
                if sparse_runtime is not None
                else None
            )
            if isinstance(sparse_prediction, _SparsePrediction):
                sparse_output_valid[output_index] = True
                sparse_center_residual[output_index] = (
                    sparse_prediction.center_residual
                )
                sparse_center_residual_fraction[output_index] = (
                    sparse_prediction.center_residual_fraction
                )
                sparse_log_dimension_residual[output_index] = (
                    sparse_prediction.log_dimension_residual
                )
                sparse_candidate_iou[output_index] = (
                    sparse_prediction.candidate_iou
                )
                sparse_improvement_probability[output_index] = (
                    sparse_prediction.improvement_probability
                )
                sparse_uncertainty[output_index] = (
                    sparse_prediction.uncertainty
                )
                sparse_coarse_occupancy_logits[output_index] = (
                    sparse_prediction.coarse_occupancy_logits
                )
                sparse_coarse_occupancy_targets[output_index] = (
                    sparse_prediction.coarse_occupancy_targets
                )
                sparse_occupancy_logits[output_index] = (
                    sparse_prediction.occupancy_logits
                )
                sparse_occupancy_targets[output_index] = (
                    sparse_prediction.occupancy_targets
                )
                sparse_selected_indices[output_index] = (
                    sparse_prediction.selected_indices
                )
                sparse_selected_mask[output_index] = (
                    sparse_prediction.selected_mask
                )
                sparse_selected_stats[output_index] = np.asarray(
                    (
                        sparse_prediction.selected_count,
                        sparse_prediction.selected_fraction,
                        sparse_prediction.selected_occupancy_mean,
                        sparse_prediction.selected_occupancy_maximum,
                        sparse_prediction.selected_target_fraction,
                        sparse_prediction.valid_point_count,
                    ),
                    dtype=np.float32,
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
        destination = root / f"{scene_id}_tracks.npz"
        with tempfile.NamedTemporaryFile(
            dir=root, suffix=".npz", delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            np.savez_compressed(
                temporary,
                scene_id=np.asarray(scene_id),
                # Full final-output pre/post geometry is intentionally kept
                # separate from the observed-memory rows below.  Identity
                # controls can therefore prove that even unobserved outputs
                # were not changed, without comparing independent GPU runs.
                output_geometry_schema=np.asarray(
                    "boxfusion.full_output_geometry_prepost.v1"
                ),
                output_pre_geometry_boxes=result.refit_original_boxes,
                output_pre_geometry_corners=(
                    result.refit_original_corners
                ),
                output_post_geometry_boxes=result.boxes,
                output_post_geometry_corners=result.corners,
                output_source_indices=result.source_indices,
                output_stable_ids=result.stable_ids,
                output_refit_applied=result.refit_applied,
                boxes=result.boxes[observed_indices],
                scores=result.scores[observed_indices],
                quality_features=result.quality_features[observed_indices],
                sparse_pair_schema=np.asarray(
                    "sgcdet_observer_active_pair_v1"
                ),
                sparse_input_schema=np.asarray(
                    "sgcdet_local_sparse_8x8x4_to_16x16x8_v1"
                ),
                sparse_output_schema=np.asarray(
                    SGCDET_SPARSE_REFINER_SCHEMA
                ),
                sparse_pair_keys=sparse_pair_keys,
                sparse_pair_source_indices=result.source_indices[
                    observed_indices
                ],
                sparse_pair_stable_ids=result.stable_ids[
                    observed_indices
                ],
                sparse_points_local=sparse_points_local,
                sparse_point_mask=sparse_point_mask,
                sparse_view_features=sparse_view_features,
                sparse_view_mask=sparse_view_mask,
                sparse_local_boxes=sparse_local_boxes,
                sparse_quality_features=sparse_quality_features,
                sparse_frame_center=sparse_frame_center,
                sparse_frame_basis=sparse_frame_basis,
                sparse_input_valid=sparse_input_valid,
                sparse_output_valid=sparse_output_valid,
                sparse_runtime_reason=sparse_runtime_reason,
                sparse_gate_reason=sparse_runtime_reason,
                sparse_accepted=sparse_accepted,
                sparse_center_residual=sparse_center_residual,
                sparse_center_residual_fraction=(
                    sparse_center_residual_fraction
                ),
                sparse_log_dimension_residual=(
                    sparse_log_dimension_residual
                ),
                sparse_candidate_iou=sparse_candidate_iou,
                sparse_improvement_probability=(
                    sparse_improvement_probability
                ),
                sparse_uncertainty=sparse_uncertainty,
                sparse_coarse_occupancy_logits=(
                    sparse_coarse_occupancy_logits
                ),
                sparse_coarse_occupancy_targets=(
                    sparse_coarse_occupancy_targets
                ),
                sparse_occupancy_logits=sparse_occupancy_logits,
                sparse_occupancy_targets=sparse_occupancy_targets,
                sparse_selected_indices=sparse_selected_indices,
                sparse_selected_mask=sparse_selected_mask,
                sparse_selected_stats=sparse_selected_stats,
                sparse_selected_stat_names=np.asarray(
                    (
                        "count",
                        "fraction",
                        "occupancy_mean",
                        "occupancy_maximum",
                        "target_fraction",
                        "valid_point_count",
                    ),
                    dtype=np.str_,
                ),
                sparse_original_boxes=sparse_original_boxes,
                sparse_original_corners=sparse_original_corners,
                sparse_active_boxes=sparse_active_boxes,
                sparse_active_corners=sparse_active_corners,
                sparse_detector_scores=sparse_detector_scores,
                sparse_final_b6_scores=sparse_final_scores,
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
                mutation_sparse_refiner_enabled=np.asarray(
                    result.summary[
                        "mutation_sparse_refiner_enabled"
                    ],
                    dtype=bool,
                ),
                mutation_sparse_geometry_enabled=np.asarray(
                    result.summary[
                        "mutation_sparse_geometry_enabled"
                    ],
                    dtype=bool,
                ),
                sparse_collect_diagnostics=np.asarray(
                    result.summary["sparse_collect_diagnostics"],
                    dtype=bool,
                ),
                sparse_max_views=np.asarray(
                    result.summary["sparse_max_views"],
                    dtype=np.int64,
                ),
                sparse_points_per_view=np.asarray(
                    result.summary["sparse_points_per_view"],
                    dtype=np.int64,
                ),
                sparse_improvement_threshold=np.asarray(
                    result.summary[
                        "sparse_improvement_threshold"
                    ],
                    dtype=np.float64,
                ),
                sparse_max_candidate_uncertainty=np.asarray(
                    result.summary[
                        "sparse_max_candidate_uncertainty"
                    ],
                    dtype=np.float64,
                ),
                sparse_topk_fraction=np.asarray(
                    result.summary["sparse_topk_fraction"],
                    dtype=np.float64,
                ),
                sparse_selected_token_count=np.asarray(
                    result.summary[
                        "sparse_selected_token_count"
                    ],
                    dtype=np.int64,
                ),
                mutation_supplemental_output_enabled=np.asarray(
                    result.summary[
                        "mutation_supplemental_output_enabled"
                    ],
                    dtype=bool,
                ),
                mutation_soft_nms_enabled=np.asarray(
                    result.summary["mutation_soft_nms_enabled"],
                    dtype=bool,
                ),
                output_minimum_extent=np.asarray(
                    result.summary["output_minimum_extent"],
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
                track_ids=result.stable_ids[observed_indices],
                result_indices=observed_indices,
                labels=labels,
                quality_feature_names=np.asarray(
                    QUALITY_FEATURE_NAMES, dtype=np.str_
                ),
                summary_json=np.asarray(summary_json),
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
        sparse_rejected = dict(
            sorted(self.stats["sparse_rejected"].items())
        )
        sparse_cfg = self.config["sgcdet_sparse_refiner"]
        sparse_architecture = (
            self.sgcdet_sparse_refiner.config
            if self.sgcdet_sparse_refiner is not None
            else SGCDetLocalSparseRefinerConfig(
                **dict(sparse_cfg["architecture"])
            ).validated()
        )
        global_memories = [
            evidence.memory for evidence in self.global_tracks.values()
        ]
        top_k_views = int(self.object_config.get("top_k_views", 0))
        return {
            "enabled": self.enabled,
            "keyframes": int(self.stats["keyframes"]),
            "provider_calls": int(self.stats["provider_calls"]),
            "provider_seconds": float(self.stats["provider_seconds"]),
            "appearance_seconds": float(self.stats["appearance_seconds"]),
            "geometry_seconds": float(self.stats["geometry_seconds"]),
            "proposals": int(self.stats["proposals"]),
            "lifted": int(self.stats["lifted"]),
            "matched_global": int(self.stats["matched_global"]),
            "candidate_updates": int(self.stats["candidate_updates"]),
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
            "mutation_sparse_refiner_enabled": bool(
                sparse_cfg["enabled"]
            ),
            "mutation_sparse_geometry_enabled": bool(
                sparse_cfg["enabled"]
                and sparse_cfg["mutate_geometry"]
            ),
            "sparse_collect_diagnostics": bool(
                sparse_cfg["collect_diagnostics"]
            ),
            "mutation_supplemental_output_enabled": bool(
                self.config["supplemental_output"]["enabled"]
            ),
            "mutation_soft_nms_enabled": bool(
                self.config["quality"]["soft_nms"]["enabled"]
            ),
            "output_minimum_extent": float(
                self.config["output_filter"]["minimum_extent"]
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
            "sparse_input_schema": (
                "sgcdet_local_sparse_8x8x4_to_16x16x8_v1"
            ),
            "sparse_output_schema": SGCDET_SPARSE_REFINER_SCHEMA,
            "sparse_pair_schema": "sgcdet_observer_active_pair_v1",
            "sparse_max_views": int(sparse_cfg["max_views"]),
            "sparse_points_per_view": int(
                sparse_cfg["points_per_view"]
            ),
            "sparse_improvement_threshold": float(
                sparse_cfg["improvement_threshold"]
            ),
            "sparse_max_candidate_uncertainty": float(
                sparse_cfg["max_candidate_uncertainty"]
            ),
            "sparse_topk_fraction": float(
                sparse_architecture.topk_fraction
            ),
            "sparse_selected_token_count": int(
                sparse_architecture.selected_token_count
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
            "supplemental_considered": int(
                self.stats["supplemental_considered"]
            ),
            "supplemental_rejected_extent": int(
                self.stats["supplemental_rejected_extent"]
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
            "supplemental_output": int(
                self.stats["supplemental_output"]
            ),
            "supplemental_deduplicated": int(
                self.stats["supplemental_deduplicated"]
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
            "sparse_instances": int(self.stats["sparse_instances"]),
            "sparse_unobserved_identity": int(
                self.stats["sparse_unobserved_identity"]
            ),
            "sparse_invalid_identity": int(
                self.stats["sparse_invalid_identity"]
            ),
            "sparse_inputs_valid": int(
                self.stats["sparse_inputs_valid"]
            ),
            "sparse_batches": int(self.stats["sparse_batches"]),
            "sparse_forward_boxes": int(
                self.stats["sparse_forward_boxes"]
            ),
            "sparse_improvement_rejected": int(
                self.stats["sparse_improvement_rejected"]
            ),
            "sparse_uncertainty_rejected": int(
                self.stats["sparse_uncertainty_rejected"]
            ),
            "sparse_gate_rejected": int(
                self.stats["sparse_gate_rejected"]
            ),
            "sparse_accepted": int(self.stats["sparse_accepted"]),
            "sparse_prepare_seconds": float(
                self.stats["sparse_prepare_seconds"]
            ),
            "sparse_forward_seconds": float(
                self.stats["sparse_forward_seconds"]
            ),
            "sparse_gate_seconds": float(
                self.stats["sparse_gate_seconds"]
            ),
            "sparse_rejections": sparse_rejected,
            "refit_rejections": rejected,
        }

    def summary_text(self) -> str:
        summary = self.summary()
        return (
            "Online refinement summary | "
            f"keyframes={summary['keyframes']}, "
            f"provider_calls={summary['provider_calls']}, "
            f"proposals={summary['proposals']}, "
            f"lifted={summary['lifted']}, "
            f"matched_global={summary['matched_global']}, "
            f"candidate_updates={summary['candidate_updates']}, "
            f"candidate_clock={summary['candidate_ttl_clock']}, "
            f"active/archived="
            f"{summary['active_supplemental_tracks']}/"
            f"{summary['archived_supplemental_tracks']}, "
            f"topk={summary['top_k_candidate_views']}/"
            f"{summary['top_k_selected_views']} views, "
            f"refit_strategy={summary['refit_strategy']}, "
            f"refit_frame={summary['refit_coordinate_frame']}, "
            "box_refiner_frame="
            f"{summary['box_refiner_coordinate_frame']}, "
            "supp_filter="
            f"{summary['supplemental_considered']}->"
            f"{summary['supplemental_output']} "
            "(extent/score/proj/global/dedup="
            f"{summary['supplemental_rejected_extent']}/"
            f"{summary['supplemental_rejected_score']}/"
            f"{summary['supplemental_rejected_projection']}/"
            f"{summary['supplemental_rejected_global']}/"
            f"{summary['supplemental_deduplicated']}), "
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
            "sparse="
            f"{summary['sparse_accepted']}/"
            f"{summary['sparse_forward_boxes']} "
            "(improve/uncertainty/gate/invalid/unobserved="
            f"{summary['sparse_improvement_rejected']}/"
            f"{summary['sparse_uncertainty_rejected']}/"
            f"{summary['sparse_gate_rejected']}/"
            f"{summary['sparse_invalid_identity']}/"
            f"{summary['sparse_unobserved_identity']}), "
            "sparse_s="
            f"{summary['sparse_prepare_seconds']:.3f}/"
            f"{summary['sparse_forward_seconds']:.3f}/"
            f"{summary['sparse_gate_seconds']:.3f}, "
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
    "FinalRefinementResult",
    "OnlineRefinementController",
    "bbox_iou_2d",
    "build_online_refinement_controller",
    "center_size_to_corners",
    "corners_to_center_size",
    "resolve_online_refinement_config",
]
