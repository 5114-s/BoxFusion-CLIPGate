"""CPU-only contracts for the TriFusion AP50 gate CLI hand-off."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from boxfusion.trifusion_cli import configure_trifusion_ap50_gate


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return {
        "online_refinement": {
            "enabled": True,
            "ablation_profile": "trifusion_plus10_observer",
            "trifusion_observer": {
                "safety_gate": {"threshold_sentinel": 0.75}
            },
        }
    }


def test_cli_helper_sets_exact_non_mutating_safety_gate_contract(tmp_path):
    checkpoint = tmp_path / "gate.npz"
    checkpoint.write_bytes(b"strict-checkpoint-placeholder")
    config = _config()

    returned = configure_trifusion_ap50_gate(
        config,
        str(checkpoint),
        cli_profile="trifusion_plus10_observer",
    )

    assert returned is config
    assert config["online_refinement"]["trifusion_observer"][
        "safety_gate"
    ] == {
        "threshold_sentinel": 0.75,
        "enabled": True,
        "checkpoint": str(checkpoint),
        "collect_diagnostics": True,
        "mutate": False,
    }


@pytest.mark.parametrize(
    "config,profile,message",
    [
        (
            {
                "online_refinement": {
                    "enabled": True,
                    "ablation_profile": "quality_only",
                }
            },
            "quality_only",
            "allowed only",
        ),
        (
            {
                "online_refinement": {
                    "enabled": False,
                    "ablation_profile": "trifusion_plus10_observer",
                }
            },
            None,
            "requires enabled",
        ),
        (
            {
                "online_refinement": {
                    "enabled": True,
                    "ablation_profile": "trifusion_plus10_observer",
                }
            },
            "quality_only",
            "allowed only",
        ),
    ],
)
def test_cli_helper_rejects_non_trifusion_or_disabled_profiles(
    tmp_path, config, profile, message
):
    checkpoint = tmp_path / "gate.npz"
    checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match=message):
        configure_trifusion_ap50_gate(
            config, str(checkpoint), cli_profile=profile
        )


def test_cli_helper_rejects_missing_or_non_npz_checkpoint(tmp_path):
    with pytest.raises(ValueError, match=r"\.npz"):
        configure_trifusion_ap50_gate(
            _config(), str(tmp_path / "gate.pt")
        )
    with pytest.raises(FileNotFoundError, match="not found"):
        configure_trifusion_ap50_gate(
            _config(), str(tmp_path / "missing.npz")
        )


def test_demo_and_shell_runners_expose_one_consistent_gate_interface():
    demo = (ROOT / "demo.py").read_text(encoding="utf-8")
    generic = (
        ROOT / "scripts" / "run_scannet_online_refinement.sh"
    ).read_text(encoding="utf-8")
    dedicated = (
        ROOT / "scripts" / "run_scannet_trifusion_observer.sh"
    ).read_text(encoding="utf-8")

    assert "--trifusion-ap50-gate-checkpoint" in demo
    assert "configure_trifusion_ap50_gate(" in demo
    assert (
        demo.index("apply_online_ablation_profile(")
        < demo.index("configure_trifusion_ap50_gate(")
    )
    assert "BOXFUSION_TRIFUSION_GATE_CHECKPOINT" in generic
    assert "--trifusion-ap50-gate-checkpoint" in generic
    assert "trifusion_plus10_observer" in generic
    assert "BOXFUSION_TRIFUSION_AP50_GATE_CHECKPOINT" in dedicated
    assert "BOXFUSION_TRIFUSION_GATE_CHECKPOINT" in dedicated

    for script in (
        ROOT / "scripts" / "run_scannet_online_refinement.sh",
        ROOT / "scripts" / "run_scannet_trifusion_observer.sh",
    ):
        subprocess.run(
            ["bash", "-n", str(script)],
            check=True,
            capture_output=True,
            text=True,
        )
