"""Strict incremental P-stage contracts layered on the frozen B6 anchor."""

from __future__ import annotations

from copy import deepcopy

import pytest

from boxfusion.p_ablation import (
    P_STAGE_MODULE_MATRIX,
    P_STAGE_TO_PROFILE,
    apply_p_ablation,
)


def _config():
    return {
        "dataset": "scannet",
        "association": {"appearance_gate": {"enabled": False}},
        "online_refinement": {
            "enabled": True,
            "ablation_profile": "quality_only",
            "appearance_memory": {"enabled": True, "sentinel": "b6"},
            "quality": {
                "enabled": True,
                "mode": "iou_mlp",
                "checkpoint": "models/scannet_b6_iou_mlp.npz",
                "feature_geometry": "original",
                "blend_with_detector": 0.40,
                "apply_to_unobserved": False,
                "soft_nms": {"enabled": False},
            },
            "refit": {"enabled": True},
            "box_refiner": {"enabled": True},
            "supplemental_output": {"enabled": True},
            "missing_track_identity": {"enabled": True},
            "mask_graph": {"enabled": True},
            "fragment_stitch": {"enabled": True},
            "generic_local_geometry_refiner": {
                "enabled": True,
                "collect_diagnostics": True,
                "mutate": True,
            },
            "supplemental_geometry_refiner": {
                "enabled": True,
                "collect_diagnostics": True,
                "mutate": True,
            },
            "trifusion_observer": {
                "enabled": True,
                "collect_diagnostics": True,
                "mutate": True,
            },
            "yidu_ablation": {
                "enabled": True,
                "collect_diagnostics": True,
                "mutate": True,
            },
            "residual_proposal": {
                "enabled": True,
                "observer_only": False,
                "mutate": True,
                "mode": "infer",
                "checkpoint": "models/p1_train_only.pt",
                "voxel_size": 0.05,
            },
            "occupancy_topk": {
                "enabled": True,
                "observer_only": False,
                "mutate": True,
                "collect_diagnostics": True,
                "checkpoint": "models/p2_train_only.pt",
            },
            "p2_local_mask_geometry": {
                "enabled": True,
                "observer_only": False,
                "mutate": True,
                "collect_diagnostics": True,
                "component_voxel_size": 0.03,
            },
            "p2_reliability_fusion": {
                "enabled": True,
                "observer_only": False,
                "mutate": True,
                "collect_diagnostics": True,
                "minimum_component_weight": 0.25,
                "maximum_component_weight": 0.90,
            },
            "sentinel": {"nested": [1, 2, 3]},
        },
    }


def test_public_p_profiles_and_matrix_are_incremental():
    assert P_STAGE_TO_PROFILE["P0"] == "p0_frozen_b6"
    assert P_STAGE_TO_PROFILE["P1"] == "p1_residual_proposal_observer"
    assert P_STAGE_TO_PROFILE["P2"] == "p2_occupancy_topk_observer"
    assert (
        P_STAGE_TO_PROFILE["P2V2"]
        == "p2v2_local_component_mask_rgbd_observer"
    )
    assert (
        P_STAGE_TO_PROFILE["P2V3"]
        == "p2v3_reliability_geometry_fusion_observer"
    )
    assert set(P_STAGE_MODULE_MATRIX["P0"]) == set(
        P_STAGE_MODULE_MATRIX["P1"]
    ) == set(P_STAGE_MODULE_MATRIX["P2"]) == set(
        P_STAGE_MODULE_MATRIX["P2V2"]
    ) == set(
        P_STAGE_MODULE_MATRIX["P2V3"]
    )
    assert not any(P_STAGE_MODULE_MATRIX["P0"].values())
    added = [
        name
        for name, enabled in P_STAGE_MODULE_MATRIX["P1"].items()
        if enabled and not P_STAGE_MODULE_MATRIX["P0"][name]
    ]
    assert added == ["residual_proposal"]
    added_p2 = [
        name
        for name, enabled in P_STAGE_MODULE_MATRIX["P2"].items()
        if enabled and not P_STAGE_MODULE_MATRIX["P1"][name]
    ]
    assert added_p2 == ["occupancy_topk"]
    added_p2v2 = [
        name
        for name, enabled in P_STAGE_MODULE_MATRIX["P2V2"].items()
        if enabled and not P_STAGE_MODULE_MATRIX["P2"][name]
    ]
    assert added_p2v2 == ["p2_local_mask_geometry"]
    added_p2v3 = [
        name
        for name, enabled in P_STAGE_MODULE_MATRIX["P2V3"].items()
        if enabled and not P_STAGE_MODULE_MATRIX["P2V2"][name]
    ]
    assert added_p2v3 == ["p2_reliability_fusion"]


@pytest.mark.parametrize("stage", ("P0", "P1", "P2", "P2V2", "P2V3"))
def test_p_stages_are_deep_copied_children_of_quality_only_b6(stage):
    source = _config()
    original = deepcopy(source)
    result = apply_p_ablation(source, stage)
    online = result["online_refinement"]

    assert source == original
    assert result is not source
    assert online is not source["online_refinement"]
    assert online["ablation_profile"] == "quality_only"
    assert online["quality"] == original["online_refinement"]["quality"]
    assert online["appearance_memory"] == original["online_refinement"][
        "appearance_memory"
    ]
    assert online["quality"]["enabled"] is True
    assert online["quality"]["mode"] == "iou_mlp"
    assert online["quality"]["soft_nms"]["enabled"] is False
    assert online["quality"]["apply_to_unobserved"] is False

    residual = online["residual_proposal"]
    assert residual["enabled"] is (stage in {"P1", "P2", "P2V2", "P2V3"})
    assert residual["observer_only"] is True
    assert residual["mutate"] is False
    assert residual["collect_diagnostics"] is (
        stage in {"P1", "P2", "P2V2", "P2V3"}
    )
    assert residual["checkpoint"] == "models/p1_train_only.pt"
    assert residual["voxel_size"] == 0.05
    if stage in {"P1", "P2", "P2V2", "P2V3"}:
        assert residual["mode"] == "infer"
        assert residual["device"] == "cpu"
    occupancy = online["occupancy_topk"]
    assert occupancy["enabled"] is (stage in {"P2", "P2V2", "P2V3"})
    assert occupancy["observer_only"] is True
    assert occupancy["mutate"] is False
    assert occupancy["collect_diagnostics"] is (
        stage in {"P2", "P2V2", "P2V3"}
    )
    assert occupancy["device"] == "cpu"
    local_geometry = online["p2_local_mask_geometry"]
    assert local_geometry["enabled"] is (stage in {"P2V2", "P2V3"})
    assert local_geometry["observer_only"] is True
    assert local_geometry["mutate"] is False
    assert local_geometry["collect_diagnostics"] is (
        stage in {"P2V2", "P2V3"}
    )
    assert local_geometry["occupancy_voxel_size"] == residual["voxel_size"]
    assert local_geometry["component_voxel_size"] == 0.03
    fusion = online["p2_reliability_fusion"]
    assert fusion["enabled"] is (stage == "P2V3")
    assert fusion["observer_only"] is True
    assert fusion["mutate"] is False
    assert fusion["collect_diagnostics"] is (stage == "P2V3")
    assert fusion["minimum_component_weight"] == 0.25
    assert fusion["maximum_component_weight"] == 0.90

    # P0/P1 are deliberately not combinations with any previous negative or
    # experimental route.  B6 ranking remains the sole output mutation.
    for name in (
        "refit",
        "box_refiner",
        "supplemental_output",
        "missing_track_identity",
        "mask_graph",
        "fragment_stitch",
    ):
        assert online[name]["enabled"] is False
    for name in (
        "generic_local_geometry_refiner",
        "supplemental_geometry_refiner",
        "trifusion_observer",
        "yidu_ablation",
    ):
        section = online[name]
        assert section["enabled"] is False
        assert section["collect_diagnostics"] is False
        assert section["mutate"] is False

    result["online_refinement"]["sentinel"]["nested"].append(4)
    assert source == original


def test_p1_differs_from_p0_only_by_read_only_residual_stream():
    source = _config()
    p0 = apply_p_ablation(source, "P0")["online_refinement"]
    p1 = apply_p_ablation(source, "P1")["online_refinement"]

    p0_comparable = deepcopy(p0)
    p1_comparable = deepcopy(p1)
    p0_residual = p0_comparable.pop("residual_proposal")
    p1_residual = p1_comparable.pop("residual_proposal")
    for key in ("p_ablation_stage", "p_ablation_profile", "p_added_module"):
        p0_comparable.pop(key)
        p1_comparable.pop(key)
    assert p1_comparable == p0_comparable

    ignored_lifecycle = {"enabled", "collect_diagnostics"}
    assert {
        key: value
        for key, value in p0_residual.items()
        if key not in ignored_lifecycle
    } == {
        key: value
        for key, value in p1_residual.items()
        if key not in ignored_lifecycle
    }
    assert p0_residual["enabled"] is False
    assert p0_residual["collect_diagnostics"] is False
    assert p1_residual["enabled"] is True
    assert p1_residual["collect_diagnostics"] is True
    assert p0_residual["mutate"] is p1_residual["mutate"] is False
    assert p0_residual["observer_only"] is p1_residual["observer_only"] is True


def test_p2_differs_from_p1_only_by_read_only_occupancy_topk():
    source = _config()
    source["online_refinement"]["occupancy_topk"] = {
        "enabled": False,
        "observer_only": True,
        "mutate": False,
        "collect_diagnostics": False,
        "checkpoint": "models/p2_train_only.pt",
        "device": "cpu",
    }
    p1 = apply_p_ablation(source, "P1")["online_refinement"]
    p2 = apply_p_ablation(source, "P2")["online_refinement"]
    p1_comparable = deepcopy(p1)
    p2_comparable = deepcopy(p2)
    p1_occupancy = p1_comparable.pop("occupancy_topk")
    p2_occupancy = p2_comparable.pop("occupancy_topk")
    for key in ("p_ablation_stage", "p_ablation_profile", "p_added_module"):
        p1_comparable.pop(key)
        p2_comparable.pop(key)
    assert p1_comparable == p2_comparable
    ignored = {"enabled", "collect_diagnostics"}
    assert {
        key: value
        for key, value in p1_occupancy.items()
        if key not in ignored
    } == {
        key: value
        for key, value in p2_occupancy.items()
        if key not in ignored
    }
    assert p1_occupancy["enabled"] is False
    assert p2_occupancy["enabled"] is True
    assert p1_occupancy["collect_diagnostics"] is False
    assert p2_occupancy["collect_diagnostics"] is True


def test_p2v2_differs_from_p2_only_by_read_only_local_mask_geometry():
    source = _config()
    p2 = apply_p_ablation(source, "P2")["online_refinement"]
    p2v2 = apply_p_ablation(source, "P2V2")["online_refinement"]
    p2_comparable = deepcopy(p2)
    p2v2_comparable = deepcopy(p2v2)
    p2_local = p2_comparable.pop("p2_local_mask_geometry")
    p2v2_local = p2v2_comparable.pop("p2_local_mask_geometry")
    for key in ("p_ablation_stage", "p_ablation_profile", "p_added_module"):
        p2_comparable.pop(key)
        p2v2_comparable.pop(key)
    assert p2_comparable == p2v2_comparable
    ignored = {"enabled", "collect_diagnostics"}
    assert {
        key: value
        for key, value in p2_local.items()
        if key not in ignored
    } == {
        key: value
        for key, value in p2v2_local.items()
        if key not in ignored
    }
    assert p2_local["enabled"] is False
    assert p2v2_local["enabled"] is True
    assert p2_local["collect_diagnostics"] is False
    assert p2v2_local["collect_diagnostics"] is True
    assert p2_local["observer_only"] is p2v2_local["observer_only"] is True
    assert p2_local["mutate"] is p2v2_local["mutate"] is False


def test_p2v3_differs_from_p2v2_only_by_read_only_reliability_fusion():
    source = _config()
    p2v2 = apply_p_ablation(source, "P2V2")["online_refinement"]
    p2v3 = apply_p_ablation(source, "P2V3")["online_refinement"]
    p2v2_comparable = deepcopy(p2v2)
    p2v3_comparable = deepcopy(p2v3)
    p2v2_fusion = p2v2_comparable.pop("p2_reliability_fusion")
    p2v3_fusion = p2v3_comparable.pop("p2_reliability_fusion")
    for key in ("p_ablation_stage", "p_ablation_profile", "p_added_module"):
        p2v2_comparable.pop(key)
        p2v3_comparable.pop(key)
    assert p2v2_comparable == p2v3_comparable
    ignored = {"enabled", "collect_diagnostics"}
    assert {
        key: value
        for key, value in p2v2_fusion.items()
        if key not in ignored
    } == {
        key: value
        for key, value in p2v3_fusion.items()
        if key not in ignored
    }
    assert p2v2_fusion["enabled"] is False
    assert p2v3_fusion["enabled"] is True
    assert p2v2_fusion["collect_diagnostics"] is False
    assert p2v3_fusion["collect_diagnostics"] is True
    assert (
        p2v2_fusion["observer_only"]
        is p2v3_fusion["observer_only"]
        is True
    )
    assert p2v2_fusion["mutate"] is p2v3_fusion["mutate"] is False


@pytest.mark.parametrize(
    "stage,error",
    [
        ("", ValueError),
        ("P3", ValueError),
        ("p1_residual_proposal_active", ValueError),
        (None, TypeError),
        (1, TypeError),
    ],
)
def test_unknown_or_unimplemented_p_stages_fail_closed(stage, error):
    with pytest.raises(error):
        apply_p_ablation(_config(), stage)


def test_malformed_config_fails_before_any_profile_is_applied():
    with pytest.raises(TypeError, match="config must be a mapping"):
        apply_p_ablation([], "P0")
    with pytest.raises(ValueError, match="online_refinement"):
        apply_p_ablation({}, "P0")
    with pytest.raises(TypeError, match="online_refinement"):
        apply_p_ablation({"online_refinement": []}, "P0")
