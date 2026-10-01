"""Strict P0/P1 ablation contracts layered on the frozen B6 anchor."""

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
            "sentinel": {"nested": [1, 2, 3]},
        },
    }


def test_public_p0_p1_profiles_and_matrix_are_fixed():
    assert P_STAGE_TO_PROFILE["P0"] == "p0_frozen_b6"
    assert P_STAGE_TO_PROFILE["P1"] == "p1_residual_proposal_observer"
    assert set(P_STAGE_MODULE_MATRIX["P0"]) == set(
        P_STAGE_MODULE_MATRIX["P1"]
    )
    assert not any(P_STAGE_MODULE_MATRIX["P0"].values())
    added = [
        name
        for name, enabled in P_STAGE_MODULE_MATRIX["P1"].items()
        if enabled and not P_STAGE_MODULE_MATRIX["P0"][name]
    ]
    assert added == ["residual_proposal"]


@pytest.mark.parametrize("stage", ("P0", "P1"))
def test_p0_p1_are_deep_copied_children_of_quality_only_b6(stage):
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
    assert residual["enabled"] is (stage == "P1")
    assert residual["observer_only"] is True
    assert residual["mutate"] is False
    assert residual["collect_diagnostics"] is (stage == "P1")
    assert residual["checkpoint"] == "models/p1_train_only.pt"
    assert residual["voxel_size"] == 0.05
    if stage == "P1":
        assert residual["mode"] == "infer"

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


@pytest.mark.parametrize(
    "stage,error",
    [
        ("", ValueError),
        ("P2", ValueError),
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
