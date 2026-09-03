from copy import deepcopy

import pytest

from boxfusion.online_ablation import (
    ONLINE_ABLATION_PROFILES,
    apply_online_ablation_profile,
)


def complete_config():
    return {
        "dataset": "scannet",
        "association": {"appearance_gate": {"enabled": True}},
        "online_refinement": {
            "enabled": False,
            "inference_every_keyframes": 5,
            "candidate_lifecycle": {
                "ttl_clock": "provider_call",
                "archive_confirmed": True,
            },
            "appearance_memory": {
                "enabled": True,
                "masked_crop": True,
            },
            "supplemental_proposals": {
                "enabled": True,
                "provider": "yoloe",
                "confidence": 0.15,
            },
            "object_memory": {
                "enabled": True,
                "voxel_size": 0.02,
            },
            "matching": {"global_match_iou": 0.05},
            "refit": {
                "enabled": True,
                "blend": 0.70,
                "min_views": 2,
                "sentinel": ["preserve", {"nested": True}],
            },
            "box_refiner": {
                "enabled": True,
                "checkpoint": "refiner.pt",
                "coordinate_frame": "world_aabb",
                "preserve_orientation": False,
                "min_quality": 0.20,
                "quality_threshold": 0.50,
                "architecture": {
                    "max_center_fraction": 0.25,
                    "max_log_dimension_residual": 0.4054651081081644,
                },
            },
            "quality": {
                "enabled": True,
                "mode": "heuristic",
                "feature_geometry": "refined",
                "blend_with_detector": 0.60,
                "apply_to_unobserved": False,
                "soft_nms": {
                    "enabled": True,
                    "method": "gaussian",
                    "sigma": 0.5,
                },
            },
            "supplemental_output": {
                "enabled": True,
                "min_confirmations": 2,
                "min_score": 0.15,
                "min_projection_iou": 0.0,
                "drop_if_global_iou": 0.70,
                "drop_if_supplemental_iou": 0.70,
            },
            "output_filter": {"minimum_extent": 0.30},
            "diagnostics": {"enabled": True, "root": "diagnostics"},
        },
    }


@pytest.mark.parametrize(
    "profile,refit_enabled,supplemental_enabled,minimum_extent",
    [
        ("observer", False, False, 0.0),
        ("refit_only", True, False, 0.0),
        ("supplemental_only", False, True, 0.0),
        ("supplemental_conservative", False, True, 0.30),
    ],
)
def test_isolated_profiles_set_only_intended_output_switches(
    profile, refit_enabled, supplemental_enabled, minimum_extent
):
    source = complete_config()
    original = deepcopy(source)
    result = apply_online_ablation_profile(source, profile)
    online = result["online_refinement"]

    assert source == original
    assert result is not source
    assert online["enabled"] is True
    assert online["appearance_memory"]["enabled"] is False
    assert online["refit"]["enabled"] is refit_enabled
    assert online["box_refiner"]["enabled"] is False
    assert online["quality"]["enabled"] is False
    assert online["quality"]["soft_nms"]["enabled"] is False
    assert (
        online["supplemental_output"]["enabled"]
        is supplemental_enabled
    )
    assert online["output_filter"]["minimum_extent"] == minimum_extent

    # Proposal/memory and non-toggle refit parameters are not rewritten.
    assert (
        online["supplemental_proposals"]
        == original["online_refinement"]["supplemental_proposals"]
    )
    assert (
        online["object_memory"]
        == original["online_refinement"]["object_memory"]
    )
    assert (
        online["candidate_lifecycle"]
        == original["online_refinement"]["candidate_lifecycle"]
    )
    assert online["refit"]["blend"] == 0.70
    assert online["refit"]["min_views"] == 2
    assert online["refit"]["sentinel"] == [
        "preserve",
        {"nested": True},
    ]
    assert online["matching"] == original["online_refinement"]["matching"]
    assert online["diagnostics"] == original["online_refinement"]["diagnostics"]
    assert result["association"] == original["association"]


def test_conservative_supplemental_profile_sets_only_fixed_output_gates():
    source = complete_config()
    result = apply_online_ablation_profile(
        source, "supplemental_conservative"
    )
    online = result["online_refinement"]
    supplemental = online["supplemental_output"]

    assert supplemental["enabled"] is True
    assert supplemental["min_score"] == 0.25
    assert supplemental["min_projection_iou"] == 0.30
    assert supplemental["drop_if_global_iou"] == 0.30
    assert supplemental["drop_if_supplemental_iou"] == 0.70
    assert online["output_filter"]["minimum_extent"] == 0.30
    assert online["candidate_lifecycle"] == source["online_refinement"][
        "candidate_lifecycle"
    ]
    assert online["object_memory"] == source["online_refinement"][
        "object_memory"
    ]


@pytest.mark.parametrize(
    "profile,quality_enabled",
    [
        ("quality_observer", False),
        ("quality_only", True),
    ],
)
def test_quality_profiles_keep_appearance_but_never_enable_b7(
    profile, quality_enabled
):
    result = apply_online_ablation_profile(complete_config(), profile)
    online = result["online_refinement"]
    assert online["appearance_memory"]["enabled"] is True
    assert online["refit"]["enabled"] is False
    assert online["box_refiner"]["enabled"] is False
    assert online["quality"]["enabled"] is quality_enabled
    assert online["quality"]["soft_nms"]["enabled"] is False
    assert online["supplemental_output"]["enabled"] is False
    assert online["output_filter"]["minimum_extent"] == 0.0
    if quality_enabled:
        assert online["quality"]["apply_to_unobserved"] is False


def test_c4_is_a_bit_exact_observer_child_of_cli_prepared_b6():
    source = complete_config()
    online = source["online_refinement"]
    # These values reproduce the quality-only command-line contract before
    # profile application.  C4 must neither replace nor perturb that model.
    online["quality"].update(
        {
            "mode": "iou_mlp",
            "feature_geometry": "original",
            "refiner_quality_override": 0.5,
        }
    )
    online["missing_track_identity"] = {
        "enabled": False,
        "sentinel": "identity",
    }
    online["mask_graph"] = {
        "enabled": False,
        "sentinel": "graph",
    }
    online["generic_local_geometry_refiner"] = {
        "enabled": False,
        "collect_diagnostics": False,
        "mutate": True,
        "sentinel": ["preserve", 4],
    }

    b6 = apply_online_ablation_profile(source, "quality_only")[
        "online_refinement"
    ]
    c4 = apply_online_ablation_profile(
        source, "b6_c4_mask_rgbd_observer"
    )["online_refinement"]

    # Apart from its explicitly read-only diagnostics stream and profile
    # label, C4 is the same configuration as the historical B6 anchor.
    b6_comparable = deepcopy(b6)
    c4_comparable = deepcopy(c4)
    for comparable in (b6_comparable, c4_comparable):
        comparable.pop("ablation_profile")
        comparable.pop("generic_local_geometry_refiner")
    assert c4_comparable == b6_comparable

    observer = c4["generic_local_geometry_refiner"]
    assert observer == {
        "enabled": True,
        "collect_diagnostics": True,
        "mutate": False,
        "sentinel": ["preserve", 4],
    }
    assert c4["appearance_memory"]["enabled"] is True
    assert c4["object_memory"] == b6["object_memory"]
    assert c4["refit"]["enabled"] is False
    assert c4["box_refiner"]["enabled"] is False
    assert c4["supplemental_output"]["enabled"] is False
    assert c4["missing_track_identity"]["enabled"] is False
    assert c4["mask_graph"]["enabled"] is False
    assert c4["supplemental_geometry_refiner"] == {
        "enabled": False,
        "mutate": False,
        "collect_diagnostics": False,
    }
    assert c4["fragment_stitch"]["enabled"] is False
    assert c4["quality"]["enabled"] is True
    assert c4["quality"]["mode"] == "iou_mlp"
    assert c4["quality"]["feature_geometry"] == "original"
    assert c4["quality"]["refiner_quality_override"] == 0.5
    assert c4["quality"]["apply_to_unobserved"] is False
    assert c4["quality"]["soft_nms"]["enabled"] is False


def test_trifusion_profile_is_a_non_mutating_child_of_c4_and_b6():
    source = complete_config()
    source["online_refinement"]["quality"].update(
        {
            "mode": "iou_mlp",
            "feature_geometry": "original",
            "refiner_quality_override": 0.5,
        }
    )
    source["online_refinement"]["trifusion_observer"] = {
        "enabled": True,
        "collect_diagnostics": False,
        "mutate": True,
        "proposal": {"max_views": 3},
        "sentinel": "preserved",
    }
    source["online_refinement"]["missing_track_identity"] = {
        "enabled": False
    }
    source["online_refinement"]["mask_graph"] = {"enabled": False}

    b6 = apply_online_ablation_profile(source, "quality_only")[
        "online_refinement"
    ]
    trifusion = apply_online_ablation_profile(
        source, "trifusion_plus10_observer"
    )["online_refinement"]

    b6_comparable = deepcopy(b6)
    trifusion_comparable = deepcopy(trifusion)
    for comparable in (b6_comparable, trifusion_comparable):
        comparable.pop("ablation_profile")
        comparable.pop("generic_local_geometry_refiner")
        comparable.pop("trifusion_observer")
    assert trifusion_comparable == b6_comparable
    assert trifusion["generic_local_geometry_refiner"]["enabled"] is True
    assert (
        trifusion["generic_local_geometry_refiner"][
            "collect_diagnostics"
        ]
        is True
    )
    assert trifusion["generic_local_geometry_refiner"]["mutate"] is False
    assert trifusion["trifusion_observer"] == {
        "enabled": True,
        "collect_diagnostics": True,
        "mutate": False,
            "proposal": {"max_views": 3},
            "missing_instance_graph": {"enabled": True},
            "safety_gate": {
                "enabled": False,
                "collect_diagnostics": False,
                "mutate": False,
            },
            "sentinel": "preserved",
        }
    assert trifusion["quality"]["enabled"] is True
    assert trifusion["quality"]["feature_geometry"] == "original"
    assert trifusion["quality"]["refiner_quality_override"] == 0.5
    assert trifusion["refit"]["enabled"] is False
    assert trifusion["box_refiner"]["enabled"] is False
    assert trifusion["supplemental_output"]["enabled"] is False
    assert trifusion["quality"]["soft_nms"]["enabled"] is False


@pytest.mark.parametrize(
    "profile",
    tuple(
        profile
        for profile in ONLINE_ABLATION_PROFILES
        if profile != "full"
    ),
)
def test_non_full_profiles_explicitly_isolate_c4_yaml_state(profile):
    source = complete_config()
    source["online_refinement"]["generic_local_geometry_refiner"] = {
        "enabled": True,
        "collect_diagnostics": True,
        "mutate": True,
        "sentinel": {"preserved": True},
    }

    section = apply_online_ablation_profile(source, profile)[
        "online_refinement"
    ]["generic_local_geometry_refiner"]
    expected = profile in {
        "b6_c4_mask_rgbd_observer",
        "trifusion_plus10_observer",
        "yidu_a1_adaptive_erosion_observer",
        "yidu_a2_dfu_filter_observer",
        "yidu_a3_voxel_components_observer",
        "yidu_a4_occupancy_msr_observer",
        "yidu_a5_raw_fused_query_observer",
        "yidu_a6_quality_gate_observer",
    }
    assert section["enabled"] is expected
    assert section["collect_diagnostics"] is expected
    assert section["mutate"] is False
    assert section["sentinel"] == {"preserved": True}


@pytest.mark.parametrize(
    "profile",
    tuple(
        profile
        for profile in ONLINE_ABLATION_PROFILES
        if profile != "full"
    ),
)
def test_non_full_profiles_explicitly_isolate_trifusion_yaml_state(
    profile,
):
    source = complete_config()
    source["online_refinement"]["trifusion_observer"] = {
        "enabled": True,
        "collect_diagnostics": True,
        "mutate": True,
        "proposal": {"max_views": 4},
        "sentinel": {"preserved": True},
    }

    section = apply_online_ablation_profile(source, profile)[
        "online_refinement"
    ]["trifusion_observer"]
    expected = profile == "trifusion_plus10_observer"
    assert section["enabled"] is expected
    assert section["collect_diagnostics"] is expected
    assert section["mutate"] is False
    assert section["proposal"] == {"max_views": 4}
    assert section["sentinel"] == {"preserved": True}


@pytest.mark.parametrize(
    (
        "profile",
        "appearance_enabled",
        "refit_enabled",
        "quality_enabled",
    ),
    [
        ("b3_memory_observer", False, False, False),
        ("b3_topk_refit_only", False, True, False),
        ("b3_b6", True, True, True),
        ("b3v2_memory_observer", False, False, False),
        ("b3v2_visibility_refit_only", False, True, False),
        ("b3v2_b6", True, True, True),
    ],
)
def test_b3_profiles_are_strictly_isolated(
    profile,
    appearance_enabled,
    refit_enabled,
    quality_enabled,
):
    source = complete_config()
    result = apply_online_ablation_profile(source, profile)
    online = result["online_refinement"]

    assert online["enabled"] is True
    assert online["appearance_memory"]["enabled"] is appearance_enabled
    assert online["refit"]["enabled"] is refit_enabled
    assert online["quality"]["enabled"] is quality_enabled
    assert online["quality"]["soft_nms"]["enabled"] is False
    assert online["box_refiner"]["enabled"] is False
    assert online["supplemental_output"]["enabled"] is False
    assert online["output_filter"]["minimum_extent"] == 0.0
    if profile.startswith("b3v2_"):
        assert online["object_memory"]["top_k_views"] == 5
        assert online["object_memory"]["max_view_candidates"] == 12
        assert online["object_memory"]["view_diversity_weight"] == 0.40
    else:
        assert online["object_memory"] == source["online_refinement"][
            "object_memory"
        ]
    assert online["supplemental_proposals"] == source[
        "online_refinement"
    ]["supplemental_proposals"]
    if quality_enabled:
        assert online["quality"]["apply_to_unobserved"] is False
    if profile in {"b3_topk_refit_only", "b3_b6"}:
        assert online["refit"]["strategy"] == "quantile_blend"
    if profile in {
        "b3v2_visibility_refit_only",
        "b3v2_b6",
    }:
        assert online["refit"]["strategy"] == "visibility_aware"
        assert online["refit"]["preserve_box_orientation"] is True
        assert online["refit"]["select_best_silhouette_pair"] is True
        assert online["refit"]["min_views"] == 2
        assert online["refit"]["max_boundary_shift_ratio"] == 0.03
        assert online["refit"]["min_extent_ratio"] == 0.92
        assert online["refit"]["min_candidate_point_support"] == 0.70
        assert online["refit"]["max_candidate_support_drop"] == 0.03
    if profile == "b3v2_b6":
        assert online["quality"]["feature_geometry"] == "original"


def test_b3_memory_observer_preserves_all_output_mutations_disabled():
    online = apply_online_ablation_profile(
        complete_config(), "b3_memory_observer"
    )["online_refinement"]

    assert online["appearance_memory"]["enabled"] is False
    assert online["refit"]["enabled"] is False
    assert online["box_refiner"]["enabled"] is False
    assert online["quality"]["enabled"] is False
    assert online["quality"]["soft_nms"]["enabled"] is False
    assert online["supplemental_output"]["enabled"] is False


@pytest.mark.parametrize(
    "profile,appearance_enabled,quality_enabled",
    [
        ("b5v2_refiner_only", False, False),
        ("b5v2_b6", True, True),
    ],
)
def test_b5v2_profiles_isolate_object_local_neural_refinement(
    profile,
    appearance_enabled,
    quality_enabled,
):
    source = complete_config()
    original = deepcopy(source)
    online = apply_online_ablation_profile(source, profile)[
        "online_refinement"
    ]

    assert source == original
    assert online["enabled"] is True
    assert online["appearance_memory"]["enabled"] is appearance_enabled
    assert online["object_memory"]["top_k_views"] == 5
    assert online["object_memory"]["max_view_candidates"] == 12
    assert online["object_memory"]["view_diversity_weight"] == 0.40
    assert online["refit"]["enabled"] is False
    assert online["box_refiner"]["enabled"] is True
    assert online["box_refiner"]["coordinate_frame"] == "box_local"
    assert online["box_refiner"]["preserve_orientation"] is True
    assert online["box_refiner"]["min_quality"] == 0.50
    assert online["box_refiner"]["quality_threshold"] == 0.50
    architecture = online["box_refiner"]["architecture"]
    assert architecture["max_center_fraction"] == 0.15
    assert (
        architecture["max_log_dimension_residual"]
        == 0.22314355131420976
    )
    assert online["refit"]["min_views"] == 2
    assert online["refit"]["min_points"] == 128
    assert online["refit"]["max_center_shift_ratio"] == 0.16
    assert online["refit"]["min_extent_ratio"] == 0.80
    assert online["refit"]["max_extent_ratio"] == 1.25
    assert online["refit"]["min_original_point_support"] == 0.55
    assert online["refit"]["min_candidate_point_support"] == 0.55
    assert online["refit"]["max_candidate_support_drop"] == 0.08
    assert online["refit"]["min_reprojection_iou"] == 0.20
    assert online["refit"]["min_reprojection_improvement"] == 0.0
    assert online["quality"]["enabled"] is quality_enabled
    assert online["quality"]["soft_nms"]["enabled"] is False
    assert online["supplemental_output"]["enabled"] is False
    assert online["output_filter"]["minimum_extent"] == 0.0

    if profile == "b5v2_b6":
        assert online["quality"]["mode"] == "iou_mlp"
        assert online["quality"]["feature_geometry"] == "original"
        assert online["quality"]["refiner_quality_override"] == 0.5
        assert online["quality"]["apply_to_unobserved"] is False
        # The run script/CLI supplies the explicit comparison blend.
        assert online["quality"]["blend_with_detector"] == 0.60
    else:
        assert online["quality"] == {
            **original["online_refinement"]["quality"],
            "enabled": False,
            "soft_nms": {
                **original["online_refinement"]["quality"]["soft_nms"],
                "enabled": False,
            },
        }


@pytest.mark.parametrize(
    "profile,enabled",
    [
        ("joint_b3_b5_b6v2_observer", False),
        ("joint_b3_b5_b6v2", True),
    ],
)
def test_joint_b356_profiles_share_exact_k5_runtime_contract(
    profile, enabled
):
    source = complete_config()
    original = deepcopy(source)
    online = apply_online_ablation_profile(source, profile)[
        "online_refinement"
    ]
    joint = online["joint_local_head"]

    assert source == original
    assert online["enabled"] is True
    assert online["appearance_memory"]["enabled"] is False
    assert online["object_memory"]["top_k_views"] == 5
    assert online["object_memory"]["max_view_candidates"] == 12
    assert online["object_memory"]["view_diversity_weight"] == 0.40
    assert online["object_memory"]["track_ttl"] == 3
    assert online["object_memory"]["max_points_per_object"] == 8192
    assert online["candidate_lifecycle"]["ttl_clock"] == "provider_call"
    assert online["candidate_lifecycle"]["archive_confirmed"] is False

    assert online["refit"]["enabled"] is False
    assert online["box_refiner"]["enabled"] is False
    assert online["quality"]["enabled"] is False
    assert online["quality"]["soft_nms"]["enabled"] is False
    assert online["supplemental_output"]["enabled"] is False
    assert online["output_filter"]["minimum_extent"] == 0.40

    assert online["refit"]["min_views"] == 2
    assert online["refit"]["min_points"] == 128
    assert online["refit"]["max_center_shift_ratio"] == 0.16
    assert online["refit"]["min_extent_ratio"] == 0.80
    assert online["refit"]["max_extent_ratio"] == 1.25
    assert online["refit"]["min_original_point_support"] == 0.55
    assert online["refit"]["min_candidate_point_support"] == 0.55
    assert online["refit"]["max_candidate_support_drop"] == 0.08
    assert online["refit"]["min_reprojection_iou"] == 0.20
    assert online["refit"]["min_reprojection_improvement"] == 0.0

    assert joint["enabled"] is enabled
    assert joint["max_views"] == 5
    assert joint["points_per_view"] == 128
    assert joint["mutate_geometry"] is enabled
    assert joint["mutate_scores"] is enabled
    assert joint["collect_diagnostics"] is True


@pytest.mark.parametrize(
    (
        "profile",
        "supplemental_enabled",
        "quality_enabled",
        "box_refiner_enabled",
        "quality_on_supplemental",
        "box_refiner_scope",
    ),
    [
        (
            "missing_mask_graph_observer",
            False,
            False,
            False,
            False,
            "none",
        ),
        (
            "missing_mask_graph_supplemental",
            True,
            False,
            False,
            False,
            "none",
        ),
        (
            "missing_mask_graph_c1_recovery",
            True,
            False,
            False,
            False,
            "none",
        ),
        (
            "missing_mask_graph_c2_geometry_observer",
            True,
            False,
            False,
            False,
            "none",
        ),
        (
            "missing_mask_graph_c2_geometry",
            True,
            False,
            False,
            False,
            "none",
        ),
        (
            "missing_mask_graph_c3_stitch_observer",
            True,
            False,
            False,
            False,
            "none",
        ),
        (
            "missing_mask_graph_b6",
            True,
            True,
            False,
            True,
            "none",
        ),
        (
            "missing_mask_graph_b5_b6",
            True,
            True,
            True,
            True,
            "confirmed_supplemental",
        ),
    ],
)
def test_missing_mask_graph_profiles_have_a_strict_source_specific_contract(
    profile,
    supplemental_enabled,
    quality_enabled,
    box_refiner_enabled,
    quality_on_supplemental,
    box_refiner_scope,
):
    source = complete_config()
    original = deepcopy(source)
    online = apply_online_ablation_profile(source, profile)[
        "online_refinement"
    ]

    assert source == original
    assert online["enabled"] is True
    assert online["appearance_memory"]["enabled"] is True
    assert online["refit"]["enabled"] is False
    assert online["quality"]["soft_nms"]["enabled"] is False

    identity = online["missing_track_identity"]
    assert identity == {
        "enabled": True,
        "shadow_weak_global_matches": True,
        "strong_global_iou": 0.25,
        "strong_projection_iou": 0.50,
        "strong_point_support": 0.60,
    }
    graph = online["mask_graph"]
    assert graph["enabled"] is True
    assert graph["min_unique_frames"] == 3

    assert online["candidate_lifecycle"]["ttl_clock"] == "provider_call"
    assert online["candidate_lifecycle"]["archive_confirmed"] is True
    assert online["object_memory"]["track_ttl"] == 4
    assert online["object_memory"]["min_confirmations"] == 3
    assert online["object_memory"]["top_k_views"] == 5
    assert online["object_memory"]["max_view_candidates"] >= 12

    supplemental = online["supplemental_output"]
    assert supplemental["enabled"] is supplemental_enabled
    assert supplemental["min_confirmations"] == 3
    assert supplemental["require_mask_graph_confirmation"] is True
    c1_enabled = profile in {
        "missing_mask_graph_c1_recovery",
        "missing_mask_graph_c2_geometry_observer",
        "missing_mask_graph_c2_geometry",
        "missing_mask_graph_c3_stitch_observer",
    }
    for key in (
        "recover_absorbed_confirmed",
        "class_aware_extent",
        "bev_duplicate_enabled",
        "planar_duplicate_enabled",
        "rank_after_globals",
    ):
        assert supplemental[key] is c1_enabled
    # The extent gate is source-specific: existing global detections retain
    # exact baseline membership while only newly added tracks use 0.30 m.
    assert online["output_filter"]["minimum_extent"] == 0.0
    assert supplemental["minimum_extent"] == 0.30

    assert online["quality"]["enabled"] is quality_enabled
    assert (
        online["quality"]["apply_to_supplemental"]
        is quality_on_supplemental
    )
    assert online["box_refiner"]["enabled"] is box_refiner_enabled
    assert online["box_refiner"]["apply_scope"] == box_refiner_scope
    c2 = online["supplemental_geometry_refiner"]
    assert c2["enabled"] is (
        profile
        in {
            "missing_mask_graph_c2_geometry_observer",
            "missing_mask_graph_c2_geometry",
            "missing_mask_graph_c3_stitch_observer",
        }
    )
    assert c2["mutate"] is (
        profile
        in {
            "missing_mask_graph_c2_geometry",
            "missing_mask_graph_c3_stitch_observer",
        }
    )
    assert c2["collect_diagnostics"] is c2["enabled"]
    assert online["fragment_stitch"]["enabled"] is (
        profile == "missing_mask_graph_c3_stitch_observer"
    )

    if quality_enabled:
        assert online["quality"]["mode"] == "iou_mlp"
        assert online["quality"]["feature_geometry"] == "original"
        assert online["quality"]["refiner_quality_override"] == 0.5
        assert online["quality"]["apply_to_unobserved"] is False


def test_full_is_an_exact_deep_copy():
    source = complete_config()
    result = apply_online_ablation_profile(source, "full")

    assert result == source
    assert result is not source
    assert result["online_refinement"] is not source["online_refinement"]
    assert (
        result["online_refinement"]["refit"]
        is not source["online_refinement"]["refit"]
    )
    result["online_refinement"]["refit"]["sentinel"][1]["nested"] = False
    assert source["online_refinement"]["refit"]["sentinel"][1]["nested"] is True


def test_returned_profile_never_aliases_nested_input_values():
    source = complete_config()
    result = apply_online_ablation_profile(source, "refit_only")
    result["online_refinement"]["supplemental_proposals"]["confidence"] = 0.9
    result["online_refinement"]["refit"]["sentinel"].append("changed")

    assert (
        source["online_refinement"]["supplemental_proposals"]["confidence"]
        == 0.15
    )
    assert source["online_refinement"]["refit"]["sentinel"] == [
        "preserve",
        {"nested": True},
    ]


def test_profile_names_are_stable_and_strictly_validated():
    assert ONLINE_ABLATION_PROFILES == (
        "observer",
        "quality_observer",
        "refit_only",
        "supplemental_only",
        "supplemental_conservative",
        "quality_only",
        "p0_frozen_b6",
        "p1_residual_proposal_observer",
        "p2_occupancy_topk_observer",
        "b6_c4_mask_rgbd_observer",
        "trifusion_plus10_observer",
        "yidu_b0_frozen_b6",
        "yidu_a1_adaptive_erosion_observer",
        "yidu_a2_dfu_filter_observer",
        "yidu_a3_voxel_components_observer",
        "yidu_a4_occupancy_msr_observer",
        "yidu_a5_raw_fused_query_observer",
        "yidu_a6_quality_gate_observer",
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
    with pytest.raises(ValueError, match="Unknown online ablation profile"):
        apply_online_ablation_profile(complete_config(), "refit")
    with pytest.raises(TypeError, match="string"):
        apply_online_ablation_profile(complete_config(), None)
    with pytest.raises(TypeError, match="mapping"):
        apply_online_ablation_profile([], "observer")


@pytest.mark.parametrize(
    "mutation,error_type,message",
    [
        (
            lambda cfg: cfg.pop("online_refinement"),
            ValueError,
            "Missing online_refinement",
        ),
        (
            lambda cfg: cfg.__setitem__("online_refinement", []),
            TypeError,
            "online_refinement must be a mapping",
        ),
        (
            lambda cfg: cfg["online_refinement"].pop("refit"),
            ValueError,
            "online_refinement.refit",
        ),
        (
            lambda cfg: cfg["online_refinement"].__setitem__(
                "object_memory", []
            ),
            TypeError,
            "object_memory must be a mapping",
        ),
        (
            lambda cfg: cfg["online_refinement"]["quality"].pop("soft_nms"),
            ValueError,
            "quality.soft_nms",
        ),
        (
            lambda cfg: cfg["online_refinement"]["quality"][
                "soft_nms"
            ].__setitem__("enabled", 1),
            TypeError,
            "soft_nms.enabled must be Boolean",
        ),
        (
            lambda cfg: cfg["online_refinement"]["output_filter"].__setitem__(
                "minimum_extent", "0.3"
            ),
            TypeError,
            "minimum_extent",
        ),
    ],
)
def test_malformed_online_refinement_structure_fails_fast(
    mutation, error_type, message
):
    source = complete_config()
    mutation(source)
    with pytest.raises(error_type, match=message):
        apply_online_ablation_profile(source, "observer")
