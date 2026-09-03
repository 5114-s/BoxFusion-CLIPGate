from dataclasses import replace

import torch

from boxfusion.boxer_gsa import BoxerGSA, BoxerGSAConfig
from boxfusion.boxes import GeneralInstance3DBoxes
from boxfusion.instances import Instances3D


def _cfg(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    return {
        "lifting": {
            "boxer_gsa": {
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


def _instances(center_x=0.0, score=0.45, size=1.0):
    instances = Instances3D((480, 640))
    instances.scores = torch.tensor([score], dtype=torch.float32)
    instances.pred_boxes_3d = GeneralInstance3DBoxes(
        torch.tensor([[center_x, 0.0, 2.0, size, size, size]]),
        torch.eye(3).unsqueeze(0),
    )
    return instances


def test_stable_group_promotes_source_geometry_with_dynamic_score(tmp_path):
    gsa = BoxerGSA(_cfg(tmp_path), device="cpu")
    pose = torch.eye(4)

    assert len(gsa.recover(0, _instances(0.00, 0.44), camera_to_world=pose)) == 0
    promoted = gsa.recover(25, _instances(0.02, 0.46), camera_to_world=pose)

    assert len(promoted) == 1
    assert promoted.pred_boxes_3d.tensor[0, 0].item() == torch.tensor(0.02).item()
    assert 0.4 <= promoted.scores.item() < 0.5
    assert promoted.scores.item() != torch.tensor(0.46).item()
    summary = gsa.summary()
    assert summary["source_geometry_preserved"] is True
    assert summary["stats"]["groups_confirmed_two_view"] == 1


def test_group_can_associate_small_iou_when_center_and_size_agree(tmp_path):
    gsa = BoxerGSA(_cfg(tmp_path), device="cpu")
    gsa.config = replace(
        gsa.config,
        two_view_quality=0.0,
        minimum_pair_iou=0.20,
        minimum_affinity=0.35,
    )
    pose = torch.eye(4)

    assert len(gsa.recover(0, _instances(0.00, size=0.40), camera_to_world=pose)) == 0
    promoted = gsa.recover(25, _instances(0.35, size=0.40), camera_to_world=pose)

    assert len(promoted) == 1
    event = gsa.summary()["events"][-1]["assignments"][0]
    assert event["pair"]["aabb_iou"] < 0.20
    assert event["action"] == "matched"


def test_distant_observations_do_not_form_a_group(tmp_path):
    gsa = BoxerGSA(_cfg(tmp_path), device="cpu")
    pose = torch.eye(4)

    assert len(gsa.recover(0, _instances(0.0), camera_to_world=pose)) == 0
    assert len(gsa.recover(25, _instances(2.0), camera_to_world=pose)) == 0
    assert gsa.summary()["stats"]["groups_created"] == 2
    assert gsa.summary()["stats"]["groups_confirmed"] == 0


def test_invalid_score_interval_is_rejected():
    try:
        BoxerGSAConfig.from_mapping(
            {"enabled": True, "low_score_min": 0.5, "native_score_min": 0.5}
        )
    except ValueError as error:
        assert "score interval" in str(error)
    else:
        raise AssertionError("invalid Boxer-GSA score interval was accepted")
