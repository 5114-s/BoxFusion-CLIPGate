from pathlib import Path

import yaml

from boxfusion.online_ablation import apply_online_ablation_profile


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "scannet_b6_selective_boxer_sgcdet.yaml"
RUNNER = ROOT / "scripts" / "run_scannet_b6_g0_sgcdet_combo.sh"
DRIVER = ROOT / "scripts" / "run_scannet_online_refinement.sh"


def _config():
    with CONFIG.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_combined_config_freezes_selective_boxer_g0():
    config = _config()
    assert config["detection"]["score_thresh"] == 0.4
    assert config["box_fusion"]["reliable_views"]["top_k"] == 3

    lifting = config["lifting"]
    assert lifting["backend"] == "boxer"
    assert lifting["proposal_cache"]["mode"] == "replay"
    assert lifting["boxer"]["mode"] == "active"
    assert lifting["boxer"]["apply_stage"] == "post_filter"
    assert lifting["boxer"]["selective_gate"] == {
        "enabled": True,
        "max_center_shift_m": 0.10,
        "min_volume_ratio": 0.50,
        "max_volume_ratio": 2.00,
    }


def test_combined_sparse_profiles_change_only_intended_final_head():
    config = _config()
    observer = apply_online_ablation_profile(
        config, "sgcdet_sparse_observer"
    )["online_refinement"]
    identity = apply_online_ablation_profile(
        config, "sgcdet_sparse_identity"
    )["online_refinement"]
    active = apply_online_ablation_profile(
        config, "sgcdet_sparse_active"
    )["online_refinement"]

    for profile in (observer, identity, active):
        assert profile["object_memory"]["top_k_views"] == 5
        assert profile["sgcdet_sparse_refiner"]["max_views"] == 5
        assert profile["sgcdet_sparse_refiner"]["points_per_view"] == 128
        assert profile["quality"]["enabled"] is True
        assert profile["quality"]["mode"] == "iou_mlp"
        assert profile["quality"]["soft_nms"]["enabled"] is False
        assert profile["refit"]["enabled"] is False
        assert profile["box_refiner"]["enabled"] is False
        assert profile["supplemental_output"]["enabled"] is False
        assert profile["output_filter"]["minimum_extent"] == 0.40

    assert observer["sgcdet_sparse_refiner"]["enabled"] is False
    assert observer["sgcdet_sparse_refiner"]["mutate_geometry"] is False
    assert observer["sgcdet_sparse_refiner"]["collect_diagnostics"] is True
    assert identity["sgcdet_sparse_refiner"]["enabled"] is True
    assert identity["sgcdet_sparse_refiner"]["mutate_geometry"] is False
    assert active["sgcdet_sparse_refiner"]["enabled"] is True
    assert active["sgcdet_sparse_refiner"]["mutate_geometry"] is True


def test_combined_runner_freezes_both_modules_and_requires_explicit_full100():
    wrapper = RUNNER.read_text(encoding="utf-8")
    driver = DRIVER.read_text(encoding="utf-8")

    for value in (
        'BOXFUSION_BOXER_GATE_MAX_CENTER_SHIFT_M="0.10"',
        'BOXFUSION_BOXER_GATE_MIN_VOLUME_RATIO="0.50"',
        'BOXFUSION_BOXER_GATE_MAX_VOLUME_RATIO="2.00"',
        'BOXFUSION_QUALITY_DETECTOR_BLEND="0.40"',
        'BOXFUSION_SCANNET_MIN_EXTENT="0.40"',
        "BOXFUSION_COMBO_SCENE_LIST",
        "BOXFUSION_COMBO_RUN_TAG",
        "sgcdet_sparse_observer",
        "sgcdet_sparse_identity",
        "sgcdet_sparse_active",
    ):
        assert value in wrapper

    assert "--boxer-selective-max-center-shift-m" in driver
    assert "--online-sgcdet-sparse-checkpoint" in driver
    assert 'sha256sum "$SGCDET_SPARSE_CHECKPOINT"' in driver

