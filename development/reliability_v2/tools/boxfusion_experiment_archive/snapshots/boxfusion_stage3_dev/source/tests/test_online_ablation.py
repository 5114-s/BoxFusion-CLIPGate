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
            },
            "quality": {
                "enabled": True,
                "mode": "heuristic",
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
        "refit_only",
        "supplemental_only",
        "supplemental_conservative",
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
