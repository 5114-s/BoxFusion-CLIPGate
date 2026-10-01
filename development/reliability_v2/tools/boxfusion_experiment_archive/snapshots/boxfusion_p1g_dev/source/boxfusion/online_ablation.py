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

from boxfusion.yidu_ablation import (
    YIDU_PROFILE_TO_STAGE,
    apply_yidu_ablation,
)
from boxfusion.p_ablation import (
    P_PROFILE_TO_STAGE,
    apply_p_ablation,
)


ONLINE_ABLATION_PROFILES = (
    "observer",
    "quality_observer",
    "refit_only",
    "supplemental_only",
    "supplemental_conservative",
    "quality_only",
    *tuple(P_PROFILE_TO_STAGE),
    "b6_c4_mask_rgbd_observer",
    "trifusion_plus10_observer",
    *tuple(YIDU_PROFILE_TO_STAGE),
    "b3_memory_observer",
    "b3_topk_refit_only",
    "b3_b6",
    "b3v2_memory_observer",
    "b3v2_visibility_refit_only",
    "b3v2_b6",
    "b5v2_memory_observer",
    "b5v2_refiner_only",
    "b5v2_b6",
    "missing_mask_graph_observer",
    "missing_mask_graph_supplemental",
    "missing_mask_graph_c1_recovery",
    "missing_mask_graph_c2_geometry_observer",
    "missing_mask_graph_c2_geometry",
    "missing_mask_graph_c3_stitch_observer",
    "missing_mask_graph_b6",
    "missing_mask_graph_b5_b6",
    "joint_b3_b5_b6v2_observer",
    "joint_b3_b5_b6v2",
    "full",
)

_JOINT_PROFILE_DEFAULTS = {
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
}

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
    joint = online.get("joint_local_head")
    if joint is not None:
        if not isinstance(joint, Mapping):
            raise TypeError(
                "online_refinement.joint_local_head must be a mapping"
            )
        _validate_enabled(
            joint, "online_refinement.joint_local_head"
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
    ``p0_frozen_b6`` / ``p1_residual_proposal_observer`` /
    ``p2_occupancy_topk_observer``
        Freeze the same B6 quality-only output. P1 adds exactly one detached,
        class-agnostic residual RGB-D proposal observer; P2 adds only a
        detached foreground-occupancy Top-K selector over frozen P1 voxels.
        P0 keeps both disabled.
    ``b6_c4_mask_rgbd_observer``
        Preserve the historical B6 quality-only output bit-for-bit while a
        second, read-only SAM3 Mask-RGBD stream observes generic local
        geometry candidates.  The secondary stream collects diagnostics only:
        it cannot mutate geometry, scores, count, order, the primary
        appearance memory, or the primary object memory.
    ``trifusion_plus10_observer``
        Preserve the same historical B6 quality-only output bit-for-bit and
        replay the same cache-only C4 evidence stream, then run the
        deterministic local occupancy/MSR OBB proposal on every observed
        global.  Both geometry branches are diagnostics-only: no candidate
        may change exported geometry, scores, count, IDs, or order.
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
    ``missing_mask_graph_observer``
        Shadow weak global matches into an incremental Mask Graph and collect
        missing-track diagnostics without changing any exported detection.
    ``missing_mask_graph_supplemental``
        Add only graph-confirmed missing tracks. Global geometry and scores
        remain unchanged and the B6/B5 heads stay disabled.
    ``missing_mask_graph_c1_recovery``
        Recover graph-confirmed tracks absorbed by transient strong global
        matches, then apply class-aware supplemental extent, BEV duplicate,
        and conservative supplemental-ranking gates. This is an isolated C1
        output branch; global geometry/scores and the B5/B6 heads remain
        unchanged.
    ``missing_mask_graph_c2_geometry_observer``
        Build C2 depth-occupancy proposals and online verification
        diagnostics on top of C1 while preserving every C1 output bit.
    ``missing_mask_graph_c2_geometry``
        Apply only verified C2 geometry to retained supplemental rows.
        Detection count, IDs, labels, scores, and upstream/global rows remain
        identical to C1.
    ``missing_mask_graph_c3_stitch_observer``
        Preserve the active C2 output exactly while observing deterministic
        cross-lifecycle, same-label fragment clusters.  The clusters are
        diagnostics only and cannot enter the exported row set.
    ``missing_mask_graph_b6``
        Preserve the frozen global B6 anchor and score graph-confirmed
        supplemental rows once with the same explicitly frozen feature
        contract. B5 and Soft-NMS remain disabled.
    ``missing_mask_graph_b5_b6``
        Run the same frozen B6 scoring, then apply B5 only to graph-confirmed
        supplemental rows as a final identity-fallback geometry correction.
    ``joint_b3_b5_b6v2_observer``
        Collect the exact V=5, P=128 joint B3 inputs without loading a
        checkpoint or changing geometry, scores, count, or order.
    ``joint_b3_b5_b6v2``
        Run the strict shared B3 -> B5 + B6-v2 local head. Legacy refit/B5/B6,
        supplemental output, CLIP appearance, and Soft-NMS remain disabled.

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

    if profile in P_PROFILE_TO_STAGE:
        return apply_p_ablation(config, P_PROFILE_TO_STAGE[profile])

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
    missing_identity = online.get("missing_track_identity")
    if not isinstance(missing_identity, MutableMapping):
        missing_identity = (
            {} if missing_identity is None else dict(missing_identity)
        )
        online["missing_track_identity"] = missing_identity
    mask_graph = online.get("mask_graph")
    if not isinstance(mask_graph, MutableMapping):
        mask_graph = {} if mask_graph is None else dict(mask_graph)
        online["mask_graph"] = mask_graph
    mask_graph_profiles = {
        "missing_mask_graph_observer",
        "missing_mask_graph_supplemental",
        "missing_mask_graph_c1_recovery",
        "missing_mask_graph_c2_geometry_observer",
        "missing_mask_graph_c2_geometry",
        "missing_mask_graph_c3_stitch_observer",
        "missing_mask_graph_b6",
        "missing_mask_graph_b5_b6",
    }
    yidu_profiles = set(YIDU_PROFILE_TO_STAGE)
    yidu_observer_profiles = {
        profile_name
        for profile_name, stage_name in YIDU_PROFILE_TO_STAGE.items()
        if stage_name != "B0"
    }
    joint_profiles = {
        "joint_b3_b5_b6v2_observer",
        "joint_b3_b5_b6v2",
    }
    joint = online.get("joint_local_head")
    if joint is not None and not isinstance(joint, MutableMapping):
        joint = dict(joint)
        online["joint_local_head"] = joint
    if profile in joint_profiles and joint is None:
        joint = deepcopy(_JOINT_PROFILE_DEFAULTS)
        online["joint_local_head"] = joint
    if joint is not None:
        joint["enabled"] = profile == "joint_b3_b5_b6v2"
        joint["mutate_geometry"] = profile == "joint_b3_b5_b6v2"
        joint["mutate_scores"] = profile == "joint_b3_b5_b6v2"
        joint["collect_diagnostics"] = profile in joint_profiles
    supplemental_geometry = online.get(
        "supplemental_geometry_refiner"
    )
    if not isinstance(supplemental_geometry, MutableMapping):
        supplemental_geometry = (
            {}
            if supplemental_geometry is None
            else dict(supplemental_geometry)
        )
        online["supplemental_geometry_refiner"] = (
            supplemental_geometry
        )
    c2_profiles = {
        "missing_mask_graph_c2_geometry_observer",
        "missing_mask_graph_c2_geometry",
        "missing_mask_graph_c3_stitch_observer",
    }
    supplemental_geometry["enabled"] = profile in c2_profiles
    supplemental_geometry["mutate"] = (
        profile
        in {
            "missing_mask_graph_c2_geometry",
            "missing_mask_graph_c3_stitch_observer",
        }
    )
    supplemental_geometry["collect_diagnostics"] = (
        profile in c2_profiles
    )
    fragment_stitch = online.get("fragment_stitch")
    if not isinstance(fragment_stitch, MutableMapping):
        fragment_stitch = (
            {} if fragment_stitch is None else dict(fragment_stitch)
        )
        online["fragment_stitch"] = fragment_stitch
    fragment_stitch["enabled"] = (
        profile == "missing_mask_graph_c3_stitch_observer"
    )
    generic_geometry = online.get("generic_local_geometry_refiner")
    if not isinstance(generic_geometry, MutableMapping):
        generic_geometry = (
            {} if generic_geometry is None else dict(generic_geometry)
        )
        online["generic_local_geometry_refiner"] = generic_geometry
    c4_observer_profiles = {
        "b6_c4_mask_rgbd_observer",
        "trifusion_plus10_observer",
        *yidu_observer_profiles,
    }
    is_c4_observer = profile in c4_observer_profiles
    # Every non-``full`` profile receives an explicit value so a stale YAML
    # or shell override cannot silently activate C4.  C4 is observer-only by
    # construction; a future active profile must be named and reviewed
    # separately rather than toggling this mutation bit.
    generic_geometry["enabled"] = is_c4_observer
    generic_geometry["collect_diagnostics"] = is_c4_observer
    generic_geometry["mutate"] = False
    if is_c4_observer:
        # C4 is a strict child of the historical quality-only B6 anchor.  Its
        # SAM3 evidence is maintained by a secondary controller-owned memory,
        # so Mask Graph/C1 lifecycle state must not leak into the primary
        # B6 path.
        missing_identity["enabled"] = False
        mask_graph["enabled"] = False
    if profile in yidu_profiles:
        # The YiDu route owns a detached SAM3 Mask-RGBD memory.  A1 adds only
        # adaptive erosion; A2 adds only the DFU-style 3D filters.  Later
        # stages keep the exact same evidence settings so adjacent profiles
        # differ by one downstream module.
        stage = YIDU_PROFILE_TO_STAGE[profile]
        secondary_memory = generic_geometry.get("secondary_object_memory")
        if not isinstance(secondary_memory, MutableMapping):
            secondary_memory = (
                {}
                if secondary_memory is None
                else dict(secondary_memory)
            )
            generic_geometry["secondary_object_memory"] = secondary_memory
        if stage != "B0":
            secondary_memory.update(
                {
                    "top_k_views": 5,
                    "max_view_candidates": 12,
                    "view_diversity_weight": 0.40,
                    "max_points_per_object": 8192,
                    "mask_edge_margin": 0,
                    "adaptive_mask_erosion": True,
                    "adaptive_erosion_min_margin": 0,
                    "adaptive_erosion_max_margin": 3,
                    "adaptive_erosion_radius_fraction": 0.02,
                    "adaptive_erosion_depth_edge_weight": 4.0,
                    "radius_filter_enabled": stage
                    in {"A2", "A3", "A4", "A5", "A6"},
                    "radius_filter_radius": 0.05,
                    "radius_filter_min_neighbors": 3,
                    "statistical_filter_enabled": stage
                    in {"A2", "A3", "A4", "A5", "A6"},
                    "statistical_filter_k": 8,
                    "statistical_filter_std_ratio": 2.0,
                    "point_filter_max_points": 4096,
                    "point_filter_min_points": 16,
                }
            )
    trifusion = online.get("trifusion_observer")
    if not isinstance(trifusion, MutableMapping):
        trifusion = {} if trifusion is None else dict(trifusion)
        online["trifusion_observer"] = trifusion
    is_trifusion_observer = profile == "trifusion_plus10_observer"
    # This profile is intentionally impossible to activate by carrying stale
    # YAML state into another non-full ablation.
    trifusion["enabled"] = is_trifusion_observer
    trifusion["collect_diagnostics"] = is_trifusion_observer
    trifusion["mutate"] = False
    trifusion_missing = trifusion.get("missing_instance_graph")
    if not isinstance(trifusion_missing, MutableMapping):
        trifusion_missing = (
            {}
            if trifusion_missing is None
            else dict(trifusion_missing)
        )
        trifusion["missing_instance_graph"] = trifusion_missing
    # Explicitly pin the graph to the exact observer profile.  A stale YAML
    # ``enabled`` bit therefore cannot leak missing-instance state into any
    # historical B6/C4 ablation.
    trifusion_missing["enabled"] = is_trifusion_observer
    trifusion_gate = trifusion.get("safety_gate")
    if not isinstance(trifusion_gate, MutableMapping):
        trifusion_gate = (
            {} if trifusion_gate is None else dict(trifusion_gate)
        )
        trifusion["safety_gate"] = trifusion_gate
    # M4 is opt-in only after the exact profile has been applied and a
    # train-only checkpoint has explicitly been supplied by the CLI.  Profile
    # selection itself must never activate a stale checkpoint.
    trifusion_gate["enabled"] = False
    trifusion_gate["collect_diagnostics"] = False
    trifusion_gate["mutate"] = False
    soft_nms = quality["soft_nms"]
    if not isinstance(soft_nms, MutableMapping):
        soft_nms = dict(soft_nms)
        quality["soft_nms"] = soft_nms

    appearance["enabled"] = profile in {
        "quality_observer",
        "quality_only",
        "b6_c4_mask_rgbd_observer",
        "trifusion_plus10_observer",
        "b3_b6",
        "b3v2_b6",
        "b5v2_b6",
        *mask_graph_profiles,
        *yidu_profiles,
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
        *mask_graph_profiles,
        "joint_b3_b5_b6v2_observer",
        "joint_b3_b5_b6v2",
    } and "top_k_views" in object_memory:
        object_memory["top_k_views"] = 0
    if profile in {
        "b3v2_memory_observer",
        "b3v2_visibility_refit_only",
        "b3v2_b6",
        "b5v2_memory_observer",
        "b5v2_refiner_only",
        "b5v2_b6",
        *mask_graph_profiles,
        "joint_b3_b5_b6v2_observer",
        "joint_b3_b5_b6v2",
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
        "missing_mask_graph_b5_b6",
    }
    local_neural_profiles = b5v2_profiles | joint_profiles
    box_refiner["enabled"] = profile in {
        "b5v2_refiner_only",
        "b5v2_b6",
        "missing_mask_graph_b5_b6",
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
    if profile in local_neural_profiles:
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
    if profile == "b5v2_memory_observer" or profile in joint_profiles:
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
        "b6_c4_mask_rgbd_observer",
        "trifusion_plus10_observer",
        "b3_b6",
        "b3v2_b6",
        "b5v2_b6",
        "missing_mask_graph_b6",
        "missing_mask_graph_b5_b6",
        *yidu_profiles,
    }
    soft_nms["enabled"] = False
    if profile != "supplemental_conservative":
        output_filter["minimum_extent"] = 0.0
    if profile == "b5v2_memory_observer":
        output_filter["minimum_extent"] = 0.40
    if profile in joint_profiles:
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
        "missing_mask_graph_supplemental",
        "missing_mask_graph_c1_recovery",
        "missing_mask_graph_c2_geometry_observer",
        "missing_mask_graph_c2_geometry",
        "missing_mask_graph_c3_stitch_observer",
        "missing_mask_graph_b6",
        "missing_mask_graph_b5_b6",
    }
    # C2 is a geometry-only child of the independently reproducible C1
    # candidate set. Explicitly disabling these gates for every other
    # non-``full`` profile prevents inherited YAML/shell configuration from
    # changing an established ablation.
    is_c1_family = profile in {
        "missing_mask_graph_c1_recovery",
        "missing_mask_graph_c2_geometry_observer",
        "missing_mask_graph_c2_geometry",
        "missing_mask_graph_c3_stitch_observer",
    }
    supplemental["recover_absorbed_confirmed"] = is_c1_family
    supplemental["class_aware_extent"] = is_c1_family
    supplemental["bev_duplicate_enabled"] = is_c1_family
    supplemental["planar_duplicate_enabled"] = is_c1_family
    supplemental["rank_after_globals"] = is_c1_family
    if profile in {
        "quality_only",
        "b6_c4_mask_rgbd_observer",
        "trifusion_plus10_observer",
        "b3_b6",
        "b3v2_b6",
        "b5v2_b6",
        *yidu_profiles,
    }:
        quality["apply_to_unobserved"] = False
    if profile in {
        "b3v2_b6",
        "b5v2_b6",
        "b6_c4_mask_rgbd_observer",
        "trifusion_plus10_observer",
        *yidu_profiles,
    }:
        quality["feature_geometry"] = "original"
    if profile in {
        "b5v2_b6",
        "b6_c4_mask_rgbd_observer",
        "trifusion_plus10_observer",
        *yidu_profiles,
    }:
        # The CLI supplies the frozen ``iou_mlp`` checkpoint/mode.  Pin the
        # remaining feature contract so C4 receives exactly the inputs used
        # by B6 even while its independent SAM3 observer is running.
        quality["refiner_quality_override"] = 0.5
    if profile == "b5v2_b6":
        quality["mode"] = "iou_mlp"
        # B6 was trained with refiner_quality identically equal to 0.5.  The
        # B5-v2 head may use its quality prediction to gate geometry, but it
        # must not feed that out-of-distribution value into the frozen B6 MLP.
        quality["refiner_quality_override"] = 0.5
    if profile == "supplemental_conservative":
        supplemental["min_score"] = 0.25
        supplemental["min_projection_iou"] = 0.30
        supplemental["drop_if_global_iou"] = 0.30
        output_filter["minimum_extent"] = 0.30
    if profile in mask_graph_profiles:
        # The Mask Graph stages share proposal, lifecycle, graph, and Top-K
        # evidence settings.  Only their final output mutations differ.
        lifecycle = online.get("candidate_lifecycle", {})
        if not isinstance(lifecycle, MutableMapping):
            lifecycle = dict(lifecycle)
            online["candidate_lifecycle"] = lifecycle
        lifecycle["ttl_clock"] = "provider_call"
        lifecycle["archive_confirmed"] = True

        object_memory["track_ttl"] = 4
        object_memory["min_confirmations"] = 3
        object_memory["top_k_views"] = 5
        object_memory["max_view_candidates"] = max(
            int(object_memory.get("max_view_candidates", 12)),
            12,
        )
        object_memory["view_diversity_weight"] = 0.40
        object_memory["max_points_per_object"] = max(
            int(object_memory.get("max_points_per_object", 8192)),
            8192,
        )
        quality["max_view_records"] = 5

        missing_identity.update(
            {
                "enabled": True,
                "shadow_weak_global_matches": True,
                "strong_global_iou": 0.25,
                "strong_projection_iou": 0.50,
                "strong_point_support": 0.60,
            }
        )
        mask_graph.update(
            {
                "enabled": True,
                "min_unique_frames": 3,
                "minimum_edge_score": 0.38,
                "minimum_iou_3d": 0.02,
                "minimum_mutual_inside": 0.10,
                "minimum_projection_iou": 0.05,
                "minimum_geometry_matches": 2,
                "require_projection": True,
                "max_nodes": 32,
                "max_edges": 128,
            }
        )

        supplemental.update(
            {
                "min_confirmations": 3,
                "require_mask_graph_confirmation": True,
                "minimum_extent": 0.30,
                "min_score": 0.25,
                "min_projection_iou": 0.20,
                "drop_if_global_iou": 0.25,
                "drop_if_supplemental_iou": 0.50,
            }
        )
        # Keep every original/global row untouched. Only newly materialized
        # missing tracks use the supplemental-specific 0.30-m extent gate.
        output_filter["minimum_extent"] = 0.0
        quality["apply_to_unobserved"] = False
        quality["apply_to_supplemental"] = profile in {
            "missing_mask_graph_b6",
            "missing_mask_graph_b5_b6",
        }
        box_refiner["apply_scope"] = (
            "confirmed_supplemental"
            if profile == "missing_mask_graph_b5_b6"
            else "none"
        )

        if profile in {
            "missing_mask_graph_b6",
            "missing_mask_graph_b5_b6",
        }:
            # Freeze the existing B6 checkpoint contract.  Supplemental B5
            # runs after this one score pass and never feeds back into B6.
            quality["mode"] = "iou_mlp"
            quality["feature_geometry"] = "original"
            quality["refiner_quality_override"] = 0.5
    if profile in yidu_profiles:
        result = apply_yidu_ablation(result, profile)
    return result


__all__ = [
    "ONLINE_ABLATION_PROFILES",
    "apply_online_ablation_profile",
]
