from pathlib import Path

import yaml

from boxfusion.boxer_uncertainty import (
    resolve_final_boxer_uncertainty_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_parent_config_freezes_g0_and_score04():
    with (ROOT / "config" / "scannet_b6_selective_boxer.yaml").open(
        "r", encoding="utf-8"
    ) as handle:
        cfg = yaml.safe_load(handle)
    assert cfg["detection"]["score_thresh"] == 0.4
    assert cfg["data"]["post_process_min_extent"] == 0.4
    assert cfg["lifting"]["boxer"]["mode"] == "active"
    assert cfg["lifting"]["boxer"]["selective_gate"] == {
        "enabled": True,
        "max_center_shift_m": 0.10,
        "min_volume_ratio": 0.50,
        "max_volume_ratio": 2.00,
    }
    reliable = cfg["box_fusion"]["reliable_views"]
    assert reliable["top_k"] == 3
    assert reliable["boxer_uncertainty"]["mode"] == "disabled"
    assert resolve_final_boxer_uncertainty_config(reliable)["mode"] == "disabled"


def test_final_runner_varies_only_post_b6_mode():
    runner = (
        ROOT / "scripts" / "run_scannet_b6_boxer_uncertainty_final.sh"
    ).read_text(encoding="utf-8")
    assert "f0_control" in runner
    assert "f1_observer" in runner
    assert "f2_active" in runner
    assert 'BOXFUSION_BOXER_UNCERTAINTY_MODE="disabled"' in runner
    assert 'BOXFUSION_BOXER_FINAL_UNCERTAINTY_MODE="$FINAL_MODE"' in runner
    assert 'BOXFUSION_QUALITY_DETECTOR_BLEND="0.40"' in runner
    assert 'BOXFUSION_SCANNET_MIN_EXTENT="0.40"' in runner
    assert 'BOXFUSION_ONLINE_ABLATION_PROFILE="quality_only"' in runner
    assert "fixed G0 Top-K=3; reweight only" in runner


def test_generic_runner_tracks_final_diagnostics_in_fingerprint():
    runner = (
        ROOT / "scripts" / "run_scannet_online_refinement.sh"
    ).read_text(encoding="utf-8")
    assert "boxer_final_uncertainty_mode=$BOXER_FINAL_UNCERTAINTY_MODE" in runner
    assert "--boxer-final-uncertainty-mode" in runner
    assert "--boxer-final-uncertainty-diagnostics-root" in runner
    assert "final uncertainty diagnostic is missing" in runner
