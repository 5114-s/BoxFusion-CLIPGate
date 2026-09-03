"""CPU-only contracts for strict incremental YiDu observers."""

from __future__ import annotations

from copy import deepcopy

import pytest

from boxfusion.yidu_ablation import (
    YIDU_MODULES,
    YIDU_MODULE_MATRIX,
    YIDU_PROFILE_STAGE_MAP,
    YIDU_SCHEMA,
    YIDU_STAGES,
    YIDU_STAGE_ADDED_MODULE,
    YIDU_STAGE_PROFILE_MAP,
    adjacent_yidu_added_module,
    apply_yidu_ablation,
    apply_yidu_ablation_profile,
    apply_yidu_ablation_stage,
    profile_for_yidu_stage,
    resolve_yidu_stage,
    validate_yidu_stage_matrix,
)


def _config():
    return {
        "dataset": "scannet",
        "online_refinement": {
            "enabled": True,
            "ablation_profile": "quality_only",
            "quality": {"enabled": True, "checkpoint": "frozen-b6.npz"},
            "sentinel": {"keep": [1, 2, 3]},
        },
    }


def test_public_stage_profile_and_module_axes_are_fixed():
    assert YIDU_STAGES == ("B0", "A1", "A2", "A3", "A4", "A5", "A6")
    assert YIDU_MODULES == (
        "adaptive_erosion",
        "dfu_filter",
        "voxel_components",
        "occupancy_msr",
        "raw_fused_query",
        "quality_gate",
    )
    assert tuple(YIDU_STAGE_PROFILE_MAP) == YIDU_STAGES
    assert set(YIDU_PROFILE_STAGE_MAP.values()) == set(YIDU_STAGES)
    assert len(YIDU_PROFILE_STAGE_MAP) == len(YIDU_STAGES)
    for stage in YIDU_STAGES:
        profile = YIDU_STAGE_PROFILE_MAP[stage]
        assert YIDU_PROFILE_STAGE_MAP[profile] == stage
        assert profile_for_yidu_stage(stage) == profile
        assert resolve_yidu_stage(profile) == stage
        assert resolve_yidu_stage(stage.lower()) == stage


def test_builtin_matrix_is_cumulative_and_adds_exactly_one_module():
    validate_yidu_stage_matrix()
    assert not any(YIDU_MODULE_MATRIX["B0"].values())

    for index, stage in enumerate(YIDU_STAGES[1:], start=1):
        previous = YIDU_STAGES[index - 1]
        expected = YIDU_STAGE_ADDED_MODULE[stage]
        assert adjacent_yidu_added_module(previous, stage) == expected
        assert sum(YIDU_MODULE_MATRIX[stage].values()) == index
        for module in YIDU_MODULES:
            assert (
                YIDU_MODULE_MATRIX[previous][module]
                <= YIDU_MODULE_MATRIX[stage][module]
            )


@pytest.mark.parametrize(
    "previous,current",
    [
        ("B0", "A2"),
        ("A2", "A2"),
        ("A3", "A2"),
        ("A6", "B0"),
    ],
)
def test_transition_helper_rejects_skips_repeats_and_reversals(
    previous, current
):
    with pytest.raises(ValueError, match="exactly one adjacent stage"):
        adjacent_yidu_added_module(previous, current)


def test_matrix_validator_rejects_non_incremental_edits():
    removal = {
        stage: dict(YIDU_MODULE_MATRIX[stage]) for stage in YIDU_STAGES
    }
    removal["A3"]["adaptive_erosion"] = False
    with pytest.raises(ValueError, match="disables earlier"):
        validate_yidu_stage_matrix(removal)

    multiple_adds = {
        stage: dict(YIDU_MODULE_MATRIX[stage]) for stage in YIDU_STAGES
    }
    multiple_adds["A1"]["dfu_filter"] = True
    with pytest.raises(ValueError, match="must add only"):
        validate_yidu_stage_matrix(multiple_adds)

    non_boolean = {
        stage: dict(YIDU_MODULE_MATRIX[stage]) for stage in YIDU_STAGES
    }
    non_boolean["A2"]["dfu_filter"] = 1
    with pytest.raises(TypeError, match="must be Boolean"):
        validate_yidu_stage_matrix(non_boolean)


@pytest.mark.parametrize("stage", YIDU_STAGES)
def test_config_application_is_deep_copied_and_strictly_observer_only(stage):
    source = _config()
    original = deepcopy(source)

    result = apply_yidu_ablation(source, stage)

    assert source == original
    assert result is not source
    assert result["online_refinement"] is not source["online_refinement"]
    # The YiDu observer is layered on, not substituted for, frozen B6.
    assert result["online_refinement"]["ablation_profile"] == "quality_only"
    assert result["online_refinement"]["quality"] == {
        "enabled": True,
        "checkpoint": "frozen-b6.npz",
    }
    section = result["online_refinement"]["yidu_ablation"]
    assert section["schema"] == YIDU_SCHEMA
    assert section["stage"] == stage
    assert section["profile"] == YIDU_STAGE_PROFILE_MAP[stage]
    assert section["frozen_b6"] is True
    assert section["observer_only"] is True
    assert section["mutate"] is False
    assert section["enabled"] is (stage != "B0")
    assert section["collect_diagnostics"] is (stage != "B0")
    assert section["added_module"] == YIDU_STAGE_ADDED_MODULE[stage]
    assert section["modules"] == dict(YIDU_MODULE_MATRIX[stage])
    for module in YIDU_MODULES:
        module_config = section[module]
        assert module_config["enabled"] is YIDU_MODULE_MATRIX[stage][module]
        assert module_config["observer_only"] is True
        assert module_config["mutate"] is False

    result["online_refinement"]["sentinel"]["keep"].append(4)
    assert source == original


def test_existing_hyperparameters_survive_but_stale_mutation_flags_do_not():
    source = _config()
    source["online_refinement"]["yidu_ablation"] = {
        "mutate": True,
        "enabled": True,
        "collect_diagnostics": False,
        "custom_manifest": {"teacher": "immutable-v1"},
        "adaptive_erosion": {
            "radius": 3,
            "enabled": False,
            "observer_only": False,
            "mutate": True,
        },
        "quality_gate": {
            "checkpoint": "gate.npz",
            "enabled": True,
            "mutate": True,
        },
    }
    original = deepcopy(source)

    result = apply_yidu_ablation_profile(source, "A1")
    section = result["online_refinement"]["yidu_ablation"]

    assert source == original
    assert section["custom_manifest"] == {"teacher": "immutable-v1"}
    assert section["adaptive_erosion"]["radius"] == 3
    assert section["adaptive_erosion"]["enabled"] is True
    assert section["adaptive_erosion"]["observer_only"] is True
    assert section["adaptive_erosion"]["mutate"] is False
    assert section["quality_gate"]["checkpoint"] == "gate.npz"
    assert section["quality_gate"]["enabled"] is False
    assert section["quality_gate"]["observer_only"] is True
    assert section["quality_gate"]["mutate"] is False
    assert section["mutate"] is False
    assert section["collect_diagnostics"] is True


def test_stage_and_profile_appliers_are_equivalent():
    config = _config()
    profile = YIDU_STAGE_PROFILE_MAP["A4"]
    expected = apply_yidu_ablation(config, "A4")
    assert apply_yidu_ablation_stage(config, "a4") == expected
    assert apply_yidu_ablation_profile(config, profile) == expected


@pytest.mark.parametrize(
    "value,error",
    [
        (None, TypeError),
        (1, TypeError),
        ("", ValueError),
        ("A7", ValueError),
        ("yidu_unknown", ValueError),
    ],
)
def test_stage_resolution_fails_closed(value, error):
    with pytest.raises(error):
        resolve_yidu_stage(value)


def test_config_application_rejects_malformed_sections():
    with pytest.raises(TypeError, match="config must be a mapping"):
        apply_yidu_ablation([], "B0")
    with pytest.raises(ValueError, match="Missing online_refinement"):
        apply_yidu_ablation({}, "B0")
    with pytest.raises(TypeError, match="online_refinement must be a mapping"):
        apply_yidu_ablation({"online_refinement": []}, "B0")

    config = _config()
    config["online_refinement"]["yidu_ablation"] = []
    with pytest.raises(TypeError, match="yidu_ablation must be a mapping"):
        apply_yidu_ablation(config, "A1")

    config = _config()
    config["online_refinement"]["yidu_ablation"] = {
        "adaptive_erosion": []
    }
    with pytest.raises(TypeError, match="adaptive_erosion must be a mapping"):
        apply_yidu_ablation(config, "A1")
