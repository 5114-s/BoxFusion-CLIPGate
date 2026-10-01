from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_g0_config_has_opt_in_uncertainty_block():
    cfg = yaml.safe_load(
        (ROOT / "config" / "scannet_b6_selective_boxer.yaml").read_text()
    )
    assert cfg["detection"]["score_thresh"] == 0.4
    assert cfg["data"]["post_process_min_extent"] == 0.4
    assert cfg["lifting"]["backend"] == "boxer"
    assert cfg["lifting"]["boxer"]["mode"] == "active"
    assert cfg["lifting"]["boxer"]["selective_gate"] == {
        "enabled": True,
        "max_center_shift_m": 0.10,
        "min_volume_ratio": 0.50,
        "max_volume_ratio": 2.00,
    }
    reliable = cfg["box_fusion"]["reliable_views"]
    assert reliable["enabled"] is True
    assert reliable["top_k"] == 3
    assert reliable["boxer_uncertainty"]["mode"] == "disabled"
    assert reliable["boxer_uncertainty"]["confidence_power"] == 1.0
    assert reliable["boxer_uncertainty"]["minimum_confidence"] == 0.05


def test_runner_profiles_change_only_uncertainty_mode():
    wrapper = (
        ROOT / "scripts" / "run_scannet_b6_boxer_uncertainty.sh"
    ).read_text()
    for contract in (
        "u0_control)",
        'UNCERTAINTY_MODE="disabled"',
        "u1_observer)",
        'UNCERTAINTY_MODE="observer"',
        "u2_active)",
        'UNCERTAINTY_MODE="active"',
        'BOXFUSION_QUALITY_DETECTOR_BLEND="0.40"',
        'BOXFUSION_SCANNET_MIN_EXTENT="0.40"',
        'BOXFUSION_BOXER_GATE_MAX_CENTER_SHIFT_M="0.10"',
        'BOXFUSION_BOXER_GATE_MIN_VOLUME_RATIO="0.50"',
        'BOXFUSION_BOXER_GATE_MAX_VOLUME_RATIO="2.00"',
    ):
        assert contract in wrapper

    runner = (
        ROOT / "scripts" / "run_scannet_online_refinement.sh"
    ).read_text()
    assert "boxer_uncertainty_mode=$BOXER_UNCERTAINTY_MODE" in runner
    assert '"$ROOT/boxfusion/boxer_uncertainty.py"' in runner
    assert '"$ROOT/boxfusion/reliable_views.py"' in runner
    assert '"$ROOT/boxfusion/instances.py"' in runner
    assert "--boxer-uncertainty-mode" in runner
    assert "--boxer-uncertainty-diagnostics-root" in runner
    assert 'export BOXFUSION_DISABLE_ONLINE_REFINEMENT="0"' in wrapper
    assert "unset BOXFUSION_REFINER_CHECKPOINT" in wrapper
