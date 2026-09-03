from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import yaml

from boxfusion.online_ablation import apply_online_ablation_profile


ROOT = Path(__file__).resolve().parents[1]
OBSERVER = "scannet_b6_boxer_observer.yaml"
ACTIVE = "scannet_b6_selective_boxer.yaml"
CONTROL = "scannet_b6_cutr_replay.yaml"


def _load(name):
    return yaml.safe_load((ROOT / "config" / name).read_text())


def test_observer_and_active_change_only_mode_and_artifact_roots():
    observer = _load(OBSERVER)
    active = _load(ACTIVE)
    normalized = deepcopy(active)
    normalized["data"]["output_dir"] = observer["data"]["output_dir"]
    normalized["lifting"]["boxer"]["mode"] = "observer"
    normalized["lifting"]["boxer"]["diagnostics_dir"] = observer[
        "lifting"
    ]["boxer"]["diagnostics_dir"]
    normalized["online_refinement"]["diagnostics"]["root"] = observer[
        "online_refinement"
    ]["diagnostics"]["root"]
    assert normalized == observer


def test_frozen_b6_protocol_and_selective_gate():
    for name in (OBSERVER, ACTIVE):
        cfg = _load(name)
        assert cfg["detection"]["score_thresh"] == 0.4
        assert cfg["data"]["post_process_min_extent"] == 0.4
        assert cfg["association"]["appearance_gate"]["enabled"] is True
        reliable = cfg["box_fusion"]["reliable_views"]
        assert reliable["enabled"] is True
        assert reliable["top_k"] == 3

        cache = cfg["lifting"]["proposal_cache"]
        assert cache["mode"] == "replay"
        assert cache["namespace"] == "scannet-score04-gap25-postfilter-v1"
        assert cache["baseline_prediction_root"].endswith(
            "/boxer_lifting/score04/x0_cutr"
        )

        boxer = cfg["lifting"]["boxer"]
        assert boxer["apply_stage"] == "post_filter"
        gate = boxer["selective_gate"]
        assert gate == {
            "enabled": True,
            "max_center_shift_m": 0.10,
            "min_volume_ratio": 0.50,
            "max_volume_ratio": 2.00,
        }
        assert cfg["online_refinement"]["supplemental_proposals"]["cache"][
            "write"
        ] is False


def test_cutr_control_replays_the_same_frozen_inputs_without_boxer():
    control = _load(CONTROL)
    observer = _load(OBSERVER)
    assert control["lifting"]["backend"] == "cutr"
    assert "boxer" not in control["lifting"]
    assert control["lifting"]["proposal_cache"] == observer["lifting"][
        "proposal_cache"
    ]
    for key_path in (
        ("detection", "score_thresh"),
        ("data", "post_process_min_extent"),
        ("association", "appearance_gate"),
        ("box_fusion", "reliable_views"),
        ("online_refinement", "supplemental_proposals"),
        ("online_refinement", "quality"),
        ("online_refinement", "output_filter"),
    ):
        left = control
        right = observer
        for key in key_path:
            left = left[key]
            right = right[key]
        assert left == right


def test_quality_only_disables_non_score_output_mutations():
    cfg = apply_online_ablation_profile(_load(ACTIVE), "quality_only")
    online = cfg["online_refinement"]
    assert online["refit"]["enabled"] is False
    assert online["box_refiner"]["enabled"] is False
    assert online["supplemental_output"]["enabled"] is False
    assert online["quality"]["enabled"] is True
    assert online["quality"]["soft_nms"]["enabled"] is False


def test_wrapper_freezes_quality_checkpoint_and_runtime_overrides():
    checkpoint = ROOT / "models" / "scannet_b6_iou_mlp.npz"
    assert sha256(checkpoint.read_bytes()).hexdigest() == (
        "d60abf798edbfa3d7902b42651be7d6053727948f740e05795de6feed60a7071"
    )
    wrapper = (ROOT / "scripts" / "run_scannet_b6_selective_boxer.sh").read_text()
    for contract in (
        "s0_control)",
        "s1_selective|g0)",
        "g1)",
        "g2)",
        "g3)",
        'GATE_CENTER="0.075"',
        'GATE_CENTER="0.05"',
        'GATE_MIN_VOLUME="0.67"',
        'GATE_MAX_VOLUME="1.50"',
        'BOXFUSION_ONLINE_ABLATION_PROFILE="quality_only"',
        'BOXFUSION_QUALITY_MODE="iou_mlp"',
        'BOXFUSION_QUALITY_DETECTOR_BLEND="0.40"',
        'BOXFUSION_SCANNET_MIN_EXTENT="0.40"',
    ):
        assert contract in wrapper
    assert 'unset BOXFUSION_BOXER_DIAGNOSTICS_ROOT' in wrapper

    runner = (
        ROOT / "scripts" / "run_scannet_online_refinement.sh"
    ).read_text()
    for contract in (
        "BOXFUSION_BOXER_GATE_MAX_CENTER_SHIFT_M",
        "BOXFUSION_BOXER_GATE_MIN_VOLUME_RATIO",
        "BOXFUSION_BOXER_GATE_MAX_VOLUME_RATIO",
        "boxer_gate_center=$BOXER_GATE_CENTER",
        "--boxer-selective-max-center-shift-m",
        "--boxer-selective-min-volume-ratio",
        "--boxer-selective-max-volume-ratio",
    ):
        assert contract in runner

    paired = (
        ROOT / "scripts" / "run_scannet_b6_selective_boxer_paired.sh"
    ).read_text()
    assert "s0_control" in paired
    assert "s0_observer" in paired
    assert "s1_selective" in paired
    assert "audit_scannet_b6_selective_boxer.sh" in paired
