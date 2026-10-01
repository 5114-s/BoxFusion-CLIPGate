"""Pure configuration profiles for online-refinement ablations.

The profiles isolate final-output mutations while leaving the configured
supplemental proposal provider, RGB-D memory parameters, matching parameters,
and diagnostics untouched.  This module deliberately has no model or runtime
imports, so profiles can be prepared before any optional dependency is
loaded.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, MutableMapping


ONLINE_ABLATION_PROFILES = (
    "observer",
    "quality_observer",
    "refit_only",
    "supplemental_only",
    "supplemental_conservative",
    "quality_only",
    "b3_memory_observer",
    "b3_topk_refit_only",
    "b3_b6",
    "b3v2_memory_observer",
    "b3v2_visibility_refit_only",
    "b3v2_b6",
    "b5v2_memory_observer",
    "b5v2_refiner_only",
    "b5v2_b6",
    "full",
)

_REQUIRED_SECTIONS = (
    "appearance_memory",
    "supplemental_proposals",
    "object_memory",
    "refit",
    "box_refiner",
    "quality",
    "supplemental_output",
    "output_filter",
)


def _require_mapping(
    parent: Mapping[str, Any],
    key: str,
    *,
    qualified_parent: str,
) -> Mapping[str, Any]:
    if key not in parent:
        raise ValueError(f"Missing {qualified_parent}.{key} section")
    value = parent[key]
    if not isinstance(value, Mapping):
        raise TypeError(f"{qualified_parent}.{key} must be a mapping")
    return value


def _validate_enabled(section: Mapping[str, Any], qualified_name: str) -> None:
    if "enabled" not in section:
        raise ValueError(f"Missing {qualified_name}.enabled")
    if not isinstance(section["enabled"], bool):
        raise TypeError(f"{qualified_name}.enabled must be Boolean")


def _validate_config(config: Mapping[str, Any]) -> None:
    if "online_refinement" not in config:
        raise ValueError("Missing online_refinement section")
    online = config["online_refinement"]
    if not isinstance(online, Mapping):
        raise TypeError("online_refinement must be a mapping")
    if "enabled" not in online:
        raise ValueError("Missing online_refinement.enabled")
    if not isinstance(online["enabled"], bool):
        raise TypeError("online_refinement.enabled must be Boolean")

    sections = {
        name: _require_mapping(
            online, name, qualified_parent="online_refinement"
        )
        for name in _REQUIRED_SECTIONS
    }
    for name in (
        "appearance_memory",
        "supplemental_proposals",
        "object_memory",
        "refit",
        "box_refiner",
        "quality",
        "supplemental_output",
    ):
        _validate_enabled(
            sections[name], f"online_refinement.{name}"
        )
    soft_nms = _require_mapping(
        sections["quality"],
        "soft_nms",
        qualified_parent="online_refinement.quality",
    )
    _validate_enabled(soft_nms, "online_refinement.quality.soft_nms")
    minimum_extent = sections["output_filter"].get("minimum_extent")
    if minimum_extent is None:
        raise ValueError(
            "Missing online_refinement.output_filter.minimum_extent"
        )
    if isinstance(minimum_extent, bool) or not isinstance(
        minimum_extent, (int, float)
    ):
        raise TypeError(
            "online_refinement.output_filter.minimum_extent "
            "must be numeric"
        )


def _mutable_section(
    online: MutableMapping[str, Any], name: str
) -> MutableMapping[str, Any]:
    section = online[name]
    if not isinstance(section, MutableMapping):
        # ``_validate_config`` accepts general mappings as input.  Convert a
        # copied read-only/custom mapping before applying profile overrides.
        section = dict(section)
        online[name] = section
    return section


def apply_online_ablation_profile(
    config: Mapping[str, Any],
    profile: str,
) -> Dict[str, Any]:
    """Return an independent configuration with one ablation profile applied.

    ``full`` is a deep-copy-only profile: no value is changed.  The other
    profiles keep proposal and object-memory configuration unchanged:

    ``observer``
        Collect proposal/memory evidence and diagnostics without changing
        exported geometry, scores, or detection count.
    ``quality_observer``
        Same no-op output contract as ``observer``, but retain masked CLIP
        appearance extraction so B6 training diagnostics match inference.
    ``refit_only``
        Enable only the gated robust geometry refit.
    ``supplemental_only``
        Enable only confirmed supplemental-track output.
    ``supplemental_conservative``
        Enable confirmed supplemental-track output with the fixed B1 quality
        gates used by the ScanNet experiment: score/projection/global-IoU
        thresholds of 0.25/0.30/0.30 and a 0.30-m output extent filter.
    ``quality_only``
        Change only observed detections' scores with the configured learned
        quality model.  Geometry, detection count, and Soft-NMS stay disabled
        so this is a pure B6 ablation rather than a B6+B7 mixture.
    ``b3_memory_observer``
        Build the per-view Top-K Mask-RGBD memory and diagnostics while
        preserving exported boxes, scores, and detection count.
    ``b3_topk_refit_only``
        Apply only the conservative robust refit from Top-K Mask-RGBD memory.
        Detector scores and detection count remain unchanged.
    ``b3_b6``
        Combine the same conservative Top-K memory refit with learned B6
        quality scoring.  Supplemental output, BoxRefiner, and Soft-NMS stay
        disabled so the two intended mutations remain isolated.
    ``b3v2_visibility_refit_only``
        Apply only the visibility-aware, per-axis two-sided boundary refit.
        Legacy B3 profiles remain available for exact reproduction.
    ``b3v2_memory_observer``
        Construct the exact K=5 B3-v2 memory without changing any output.
    ``b3v2_b6``
        Combine B3-v2 geometry with B6 scores computed from the original
        BoxFusion geometry, matching the B6 checkpoint's training features.
    ``b5v2_refiner_only``
        Apply only the learned, object-local BoxRefiner to K=5 Mask-RGBD
        memory.  Hand-written refit, quality scoring, supplemental output,
        and Soft-NMS stay disabled.
    ``b5v2_memory_observer``
        Collect the exact object-local model inputs, full support-gate points,
        and reprojection evidence consumed by ``b5v2_refiner_only`` while
        preserving identity geometry, scores, and detection count.  The
        profile fixes K=5, provider-call TTL=3, no confirmed-track archive,
        the 0.40-m output contract, and all B5-v2 gate thresholds.
    ``b5v2_b6``
        Combine the same object-local BoxRefiner with the learned B6 IoU
        scorer.  B6 features are computed from the original BoxFusion
        geometry and all other output mutations remain disabled.

    Args:
        config: Full BoxFusion configuration containing an
            ``online_refinement`` mapping.
        profile: One of :data:`ONLINE_ABLATION_PROFILES`.
    """

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    if not isinstance(profile, str):
        raise TypeError("profile must be a string")
    if profile not in ONLINE_ABLATION_PROFILES:
        choices = ", ".join(ONLINE_ABLATION_PROFILES)
        raise ValueError(
            f"Unknown online ablation profile {profile!r}; choose {choices}"
        )
    _validate_config(config)

    result: Dict[str, Any] = deepcopy(dict(config))
    if profile == "full":
        return result
    result_online = result["online_refinement"]
    if not isinstance(result_online, MutableMapping):
        result_online = dict(result_online)
        result["online_refinement"] = result_online
    result_online["ablation_profile"] = profile

    online = result_online
    if not isinstance(online, MutableMapping):
        online = dict(online)
        result["online_refinement"] = online
    online["enabled"] = True

    appearance = _mutable_section(online, "appearance_memory")
    object_memory = _mutable_section(online, "object_memory")
    refit = _mutable_section(online, "refit")
    box_refiner = _mutable_section(online, "box_refiner")
    quality = _mutable_section(online, "quality")
    supplemental = _mutable_section(online, "supplemental_output")
    output_filter = _mutable_section(online, "output_filter")
    soft_nms = quality["soft_nms"]
    if not isinstance(soft_nms, MutableMapping):
        soft_nms = dict(soft_nms)
        quality["soft_nms"] = soft_nms

    appearance["enabled"] = profile in {
        "quality_observer",
        "quality_only",
        "b3_b6",
        "b3v2_b6",
        "b5v2_b6",
    }
    if profile not in {
        "b3_memory_observer",
        "b3_topk_refit_only",
        "b3_b6",
        "b3v2_memory_observer",
        "b3v2_visibility_refit_only",
        "b3v2_b6",
        "b5v2_memory_observer",
        "b5v2_refiner_only",
        "b5v2_b6",
    } and "top_k_views" in object_memory:
        object_memory["top_k_views"] = 0
    if profile in {
        "b3v2_memory_observer",
        "b3v2_visibility_refit_only",
        "b3v2_b6",
        "b5v2_memory_observer",
        "b5v2_refiner_only",
        "b5v2_b6",
    }:
        object_memory["top_k_views"] = 5
        object_memory["max_view_candidates"] = max(
            int(object_memory.get("max_view_candidates", 12)),
            12,
        )
        object_memory["view_diversity_weight"] = 0.40
    if profile == "b3v2_memory_observer":
        # Refit remains disabled; this value only makes diagnostics identify
        # the exact downstream B3-v2 strategy the memory is intended for.
        refit["strategy"] = "visibility_aware"
    b5v2_profiles = {
        "b5v2_memory_observer",
        "b5v2_refiner_only",
        "b5v2_b6",
    }
    box_refiner["enabled"] = profile in {
        "b5v2_refiner_only",
        "b5v2_b6",
    }
    if profile in b5v2_profiles:
        box_refiner["coordinate_frame"] = "box_local"
        box_refiner["preserve_orientation"] = True
        box_refiner["point_count"] = 512
        # Keep the legacy name in sync while the runtime migrates to the
        # explicit improvement-probability threshold.
        box_refiner["min_quality"] = 0.50
        box_refiner["quality_threshold"] = 0.50
        architecture = box_refiner.get("architecture", {})
        if not isinstance(architecture, MutableMapping):
            architecture = dict(architecture)
            box_refiner["architecture"] = architecture
        architecture["max_center_fraction"] = 0.15
        architecture["max_log_dimension_residual"] = (
            0.22314355131420976
        )
        # Neural candidates use the same point-support and reprojection
        # safeguards that made B3-v2 non-destructive. The learned head, not
        # the hand-written refit, remains the only geometry mutation.
        refit.update(
            {
                "min_views": 2,
                "min_points": 128,
                "visibility_point_crop_expansion": 1.20,
                "max_center_shift_ratio": 0.16,
                "min_extent_ratio": 0.80,
                "max_extent_ratio": 1.25,
                "min_original_point_support": 0.55,
                "min_candidate_point_support": 0.55,
                "max_candidate_support_drop": 0.08,
                "min_reprojection_iou": 0.20,
                "min_reprojection_improvement": 0.0,
            }
        )
    if profile == "b5v2_memory_observer":
        lifecycle = online.get("candidate_lifecycle", {})
        if not isinstance(lifecycle, MutableMapping):
            lifecycle = dict(lifecycle)
        online["candidate_lifecycle"] = lifecycle
        lifecycle["ttl_clock"] = "provider_call"
        lifecycle["archive_confirmed"] = False
        object_memory["track_ttl"] = 3
        object_memory["max_points_per_object"] = 8192
        quality["max_view_records"] = 5
    quality["enabled"] = profile in {
        "quality_only",
        "b3_b6",
        "b3v2_b6",
        "b5v2_b6",
    }
    soft_nms["enabled"] = False
    if profile != "supplemental_conservative":
        output_filter["minimum_extent"] = 0.0
    if profile == "b5v2_memory_observer":
        output_filter["minimum_extent"] = 0.40

    refit["enabled"] = profile in {
        "refit_only",
        "b3_topk_refit_only",
        "b3_b6",
        "b3v2_visibility_refit_only",
        "b3v2_b6",
    }
    if profile in {"b3_topk_refit_only", "b3_b6"}:
        refit["strategy"] = "quantile_blend"
    elif profile in {
        "b3v2_memory_observer",
        "b3v2_visibility_refit_only",
        "b3v2_b6",
    }:
        refit.update(
            {
                "strategy": "visibility_aware",
                "preserve_box_orientation": True,
                "min_views": 2,
                "blend": 0.50,
                "minimum_view_separation_degrees": 45.0,
                "minimum_axis_cosine": 0.35,
                "minimum_bilateral_axes": 1,
                "minimum_side_views": 1,
                "max_boundary_shift_ratio": 0.03,
                "minimum_boundary_change_ratio": 0.005,
                "visibility_boundary_quantile": 0.05,
                "visibility_point_crop_expansion": 1.20,
                "minimum_camera_outside_ratio": 0.02,
                "maximum_boundary_measurement_spread_ratio": 0.10,
                "enable_silhouette_axes": True,
                "select_best_silhouette_pair": True,
                "maximum_silhouette_axis_cosine": 0.40,
                "minimum_silhouette_views": 2,
                "minimum_silhouette_separation_degrees": 30.0,
                "max_center_shift_ratio": 0.08,
                "min_extent_ratio": 0.92,
                "max_extent_ratio": 1.0,
                "min_original_point_support": 0.70,
                "min_candidate_point_support": 0.70,
                "max_candidate_support_drop": 0.03,
                "min_reprojection_iou": 0.20,
                "min_reprojection_improvement": 0.0,
            }
        )
    supplemental["enabled"] = profile in {
        "supplemental_only",
        "supplemental_conservative",
    }
    if profile in {
        "quality_only",
        "b3_b6",
        "b3v2_b6",
        "b5v2_b6",
    }:
        quality["apply_to_unobserved"] = False
    if profile in {"b3v2_b6", "b5v2_b6"}:
        quality["feature_geometry"] = "original"
    if profile == "b5v2_b6":
        quality["mode"] = "iou_mlp"
    if profile == "supplemental_conservative":
        supplemental["min_score"] = 0.25
        supplemental["min_projection_iou"] = 0.30
        supplemental["drop_if_global_iou"] = 0.30
        output_filter["minimum_extent"] = 0.30
    return result


__all__ = [
    "ONLINE_ABLATION_PROFILES",
    "apply_online_ablation_profile",
]
