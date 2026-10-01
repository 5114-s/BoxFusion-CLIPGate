from dataclasses import replace
import random

import numpy as np
import torch

from boxfusion.boxer_mvpr import BoxerMVPR, BoxerMVPRConfig, isolated_rng
from boxfusion.boxes import GeneralInstance3DBoxes
from boxfusion.instances import Instances3D


def _cfg(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    return {
        "lifting": {
            "boxer_mvpr": {
                "enabled": True,
                "diagnostics_dir": str(tmp_path / "diagnostics"),
                "proposal_cache": {
                    "mode": "replay",
                    "root": str(tmp_path / "cache"),
                    "namespace": "low",
                    "expected_fingerprint": "test",
                    "baseline_prediction_root": str(baseline),
                },
            }
        }
    }


def _instances(center_x=0.0, score=0.45):
    instances = Instances3D((480, 640))
    instances.scores = torch.tensor([score], dtype=torch.float32)
    instances.pred_boxes_3d = GeneralInstance3DBoxes(
        torch.tensor([[center_x, 0.0, 2.0, 1.0, 1.0, 1.0]]),
        torch.eye(3).unsqueeze(0),
    )
    return instances


def test_three_stable_views_promote_current_real_score(tmp_path):
    mvpr = BoxerMVPR(_cfg(tmp_path), device="cpu")
    pose = torch.eye(4)

    assert len(mvpr.recover(0, _instances(0.00), camera_to_world=pose)) == 0
    assert len(mvpr.recover(25, _instances(0.03), camera_to_world=pose)) == 0
    promoted = mvpr.recover(50, _instances(0.01), camera_to_world=pose)

    assert len(promoted) == 1
    assert promoted.scores.item() == torch.tensor(0.45).item()
    assert mvpr.summary()["stats"]["tracks_confirmed"] == 1
    assert mvpr.summary()["stats"]["promoted_observations"] == 1


def test_unstable_first_three_views_are_not_promoted(tmp_path):
    mvpr = BoxerMVPR(_cfg(tmp_path), device="cpu")
    pose = torch.eye(4)
    mvpr.config = BoxerMVPRConfig(
        enabled=True,
        match_aabb_iou=0.0,
        match_center_m=3.0,
        diagnostics_dir=str(tmp_path / "diagnostics"),
    )

    assert len(mvpr.recover(0, _instances(0.0), camera_to_world=pose)) == 0
    assert len(mvpr.recover(25, _instances(0.8), camera_to_world=pose)) == 0
    assert len(mvpr.recover(50, _instances(1.6), camera_to_world=pose)) == 0
    assert mvpr.summary()["stats"]["tracks_rejected"] == 1


def test_rolling_consensus_can_reconfirm_after_an_unstable_window(tmp_path):
    mvpr = BoxerMVPR(_cfg(tmp_path), device="cpu")
    mvpr.config = replace(
        mvpr.config,
        rolling_consensus=True,
        match_aabb_iou=0.0,
        match_center_m=3.0,
    )
    pose = torch.eye(4)

    assert len(mvpr.recover(0, _instances(0.0), camera_to_world=pose)) == 0
    assert len(mvpr.recover(25, _instances(0.8), camera_to_world=pose)) == 0
    assert len(mvpr.recover(50, _instances(1.6), camera_to_world=pose)) == 0
    assert len(mvpr.recover(75, _instances(1.62), camera_to_world=pose)) == 0
    promoted = mvpr.recover(100, _instances(1.61), camera_to_world=pose)

    assert len(promoted) == 1
    assert mvpr.summary()["schema"] == "boxfusion.boxer_mvpr.v2"
    assert mvpr.summary()["stats"]["tracks_reconfirmed"] == 1


def test_first_promotion_can_use_consensus_medoid_geometry(tmp_path):
    mvpr = BoxerMVPR(_cfg(tmp_path), device="cpu")
    mvpr.config = replace(
        mvpr.config,
        rolling_consensus=True,
        promote_consensus_medoid=True,
    )
    pose = torch.eye(4)

    assert len(mvpr.recover(0, _instances(0.00), camera_to_world=pose)) == 0
    assert len(mvpr.recover(25, _instances(0.03), camera_to_world=pose)) == 0
    promoted = mvpr.recover(50, _instances(0.10), camera_to_world=pose)

    assert len(promoted) == 1
    assert promoted.pred_boxes_3d.tensor[0, 0].item() == torch.tensor(0.03).item()
    assert mvpr.summary()["stats"]["consensus_geometry_promotions"] == 1


def test_strong_two_view_track_can_promote_without_fixed_score(tmp_path):
    mvpr = BoxerMVPR(_cfg(tmp_path), device="cpu")
    mvpr.config = replace(
        mvpr.config,
        strong_two_view_confirmation=True,
        strong_two_view_min_median_score=0.44,
    )
    pose = torch.eye(4)

    assert len(mvpr.recover(0, _instances(0.00, 0.44), camera_to_world=pose)) == 0
    promoted = mvpr.recover(25, _instances(0.03, 0.48), camera_to_world=pose)

    assert len(promoted) == 1
    assert promoted.scores.item() == torch.tensor(0.48).item()
    summary = mvpr.summary()
    assert summary["schema"] == "boxfusion.boxer_mvpr.v2"
    assert summary["minimum_distinct_views"] == 2
    assert summary["stats"]["tracks_confirmed_two_view"] == 1


def test_auxiliary_rng_is_restored():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    expected = (random.random(), np.random.rand(), torch.rand(1).item())

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    with isolated_rng("cpu"):
        random.random()
        np.random.rand()
        torch.rand(4)
    actual = (random.random(), np.random.rand(), torch.rand(1).item())

    assert actual == expected


def test_score_interval_validation():
    try:
        BoxerMVPRConfig.from_mapping(
            {"enabled": True, "low_score_min": 0.5, "native_score_min": 0.5}
        )
    except ValueError as error:
        assert "score interval" in str(error)
    else:
        raise AssertionError("invalid closed score interval was accepted")
