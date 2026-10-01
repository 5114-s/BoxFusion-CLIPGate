from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "config/scannet_boxer_tm_fpf_c1_stream3dv2_lightweight_score05.yaml"
)
RUNNER = (
    ROOT
    / "scripts/run_scannet_boxer_tm_fpf_c1_stream3dv2_lightweight_official100.sh"
)


def test_official100_config_seals_only_the_requested_route() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert cfg["data"]["output_dir"].endswith(
        "/results/scannet_boxer_tm_fpf_c1_stream3dv2_lightweight_score05"
    )
    assert cfg["lifting"]["backend"] == "boxer"
    assert cfg["lifting"]["boxer"]["mode"] == "active"
    assert cfg["lifting"]["boxer"]["cache_image_features"] is True
    views = cfg["box_fusion"]["reliable_views"]
    assert views["enabled"] is True
    assert views["top_k"] == 3
    assert cfg["box_fusion"]["capf"]["enabled"] is False
    assert cfg["box_fusion"]["vapf_lite"]["enabled"] is False
    tm = cfg["box_fusion"]["tm_fpf_c1"]
    assert tm["enabled"] is True
    assert tm["max_accepted_faces"] == 1

    route = cfg["online_stream3dv2"]
    light = route["lightweight"]
    assert route["enabled"] is True
    assert light["enabled"] is True
    assert light["depth_trigger"]["enabled"] is True
    assert light["fastsam_top_k"] == {"box_shortlist": 12, "mask_cap": 8}
    assert light["conditional_f2"] is True
    assert light["f4_top_m_tracks"] == 8
    assert light["terminal_clip"] == {"enabled": True, "batch_size": 32}
    assert route["sam3"]["enabled"] is False
    assert cfg["online_stream3dv3"]["enabled"] is False


def test_runner_is_strict_official100_two_gpu_and_reports_ap_fps() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert '"$1" != "0,1"' in text
    assert '[[ "${#SCENES[@]}" -eq 100 ]]' in text
    assert '[[ "$shard_zero" -eq 50 && "$shard_one" -eq 50 ]]' in text
    assert "run_worker 0 0 &" in text
    assert "run_worker 1 1 &" in text
    assert '[[ "$completed" -eq 50 ]]' in text
    assert "eval_scannet_official100_real_score.sh" in text
    assert "summarize_stream3dv2_live_official100.py" in text
    assert '--scene-log-root "$SCENE_LOG_ROOT"' in text
    assert "--require-complete" in text
    assert "scannet_cbest_f4_stream3dv2_live_score05" not in text
    assert "scannet_cbest_f4_stream3dv3_live_score05" not in text
