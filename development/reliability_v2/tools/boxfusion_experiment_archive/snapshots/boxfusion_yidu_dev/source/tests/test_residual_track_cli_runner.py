"""CPU-only contracts for the residual-track semantic-gate ablation."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess

import pytest

from boxfusion.residual_track_cli import (
    configure_residual_min_semantic_score,
)


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return {
        "online_refinement": {
            "enabled": True,
            "ablation_profile": "residual_track_observer",
            "residual_track_observer": {
                "enabled": True,
                "observer_only": True,
                "mutate": False,
                "source_mode": "sam3",
                "missing_instance_graph": {
                    "enabled": True,
                    "minimum_semantic_score": 0.50,
                    "minimum_iou_3d": 0.02,
                    "global_reject_iou": 0.30,
                },
            },
        }
    }


def test_semantic_override_changes_only_one_residual_graph_value():
    config = _config()
    before = copy.deepcopy(config)

    returned = configure_residual_min_semantic_score(
        config,
        0.0,
        cli_profile="residual_track_observer",
    )

    assert returned is config
    graph = config["online_refinement"]["residual_track_observer"][
        "missing_instance_graph"
    ]
    assert graph["minimum_semantic_score"] == 0.0
    before["online_refinement"]["residual_track_observer"][
        "missing_instance_graph"
    ]["minimum_semantic_score"] = 0.0
    assert config == before


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
def test_semantic_override_rejects_non_finite_or_out_of_range(value):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        configure_residual_min_semantic_score(
            _config(),
            value,
            cli_profile="residual_track_observer",
        )


@pytest.mark.parametrize("value", [False, "0.0", object()])
def test_semantic_override_rejects_non_real_values(value):
    with pytest.raises(TypeError, match="real number"):
        configure_residual_min_semantic_score(
            _config(),
            value,
            cli_profile="residual_track_observer",
        )


def test_semantic_override_is_scoped_to_exact_enabled_profile():
    with pytest.raises(ValueError, match="allowed only"):
        configure_residual_min_semantic_score(
            _config(), 0.0, cli_profile="quality_only"
        )
    config = _config()
    config["online_refinement"]["enabled"] = False
    with pytest.raises(ValueError, match="requires enabled"):
        configure_residual_min_semantic_score(
            config, 0.0, cli_profile="residual_track_observer"
        )


def test_demo_and_shell_runners_expose_one_scoped_override_chain():
    demo = (ROOT / "demo.py").read_text(encoding="utf-8")
    generic = (
        ROOT / "scripts" / "run_scannet_online_refinement.sh"
    ).read_text(encoding="utf-8")
    dedicated = (
        ROOT / "scripts" / "run_scannet_residual_track_observer.sh"
    ).read_text(encoding="utf-8")

    assert "--residual-track-min-semantic-score" in demo
    assert "configure_residual_min_semantic_score(" in demo
    assert (
        demo.index("apply_online_ablation_profile(")
        < demo.index("configure_residual_min_semantic_score(")
    )
    assert "BOXFUSION_RESIDUAL_MIN_SEMANTIC_SCORE" in generic
    assert "--residual-track-min-semantic-score" in generic
    assert (
        "is valid only with residual_track_observer" in generic
    )
    assert "BOXFUSION_RESIDUAL_MIN_SEMANTIC_SCORE" in dedicated
    assert "graph minimum semantic score:" in dedicated

    for script in (
        ROOT / "scripts" / "run_scannet_online_refinement.sh",
        ROOT / "scripts" / "run_scannet_residual_track_observer.sh",
    ):
        subprocess.run(
            ["bash", "-n", str(script)],
            check=True,
            capture_output=True,
            text=True,
        )


def test_dedicated_runner_rejects_invalid_override_before_launch():
    environment = os.environ.copy()
    environment.update(
        {
            "BOXFUSION_RESIDUAL_MIN_SEMANTIC_SCORE": "1.01",
            "BOXFUSION_RESIDUAL_DRY_RUN": "1",
        }
    )
    result = subprocess.run(
        [
            "bash",
            str(
                ROOT
                / "scripts"
                / "run_scannet_residual_track_observer.sh"
            ),
            "0",
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "must lie in [0, 1]" in result.stderr
