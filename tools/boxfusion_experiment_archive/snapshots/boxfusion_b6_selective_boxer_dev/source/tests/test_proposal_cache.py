import json

import numpy as np
import pytest
import torch

from boxfusion.boxer_lifter import geometry_hash, protected_proposal_hashes
from boxfusion.boxes import BoxDOF, GeneralInstance3DBoxes
from boxfusion.instances import Instances3D
from boxfusion.proposal_cache import (
    ProposalCache,
    ProposalCacheConfig,
    ProposalCacheError,
    build_proposal_cache,
)


def _instances(count=2):
    instances = Instances3D((480, 640))
    # Match the exact field insertion order produced by CuTR.
    instances.scores = torch.tensor([0.81, 0.63])[:count]
    instances.pred_classes = torch.tensor([0, 0], dtype=torch.int64)[:count]
    instances.pred_boxes = torch.tensor(
        [[10.0, 20.0, 110.0, 220.0], [30.0, 40.0, 130.0, 240.0]]
    )[:count]
    instances.pred_logits = torch.tensor([[1.2], [0.8]])[:count]
    instances.pred_boxes_3d = GeneralInstance3DBoxes(
        torch.tensor(
            [
                [0.1, 0.2, 2.0, 1.0, 1.2, 0.8],
                [-0.3, 0.1, 3.0, 0.6, 0.7, 0.9],
            ]
        )[:count],
        torch.eye(3).repeat(2, 1, 1)[:count],
        dof=BoxDOF.All,
    )
    instances.object_desc = torch.arange(
        8, dtype=torch.float32
    ).reshape(2, 4)[:count]
    instances.pred_proj_xy = torch.tensor(
        [[100.0, 120.0], [200.0, 220.0]]
    )[:count]
    return instances


def _inputs():
    return {
        "image": np.zeros((4, 6, 3), dtype=np.uint8),
        "depth": torch.ones((2, 3), dtype=torch.float32),
        "image_K": torch.eye(3),
        "depth_K": torch.eye(3),
        "camera_to_world": torch.eye(4),
    }


def _config(tmp_path, mode):
    return ProposalCacheConfig(
        mode=mode,
        root=tmp_path / "cache",
        namespace="unit-v1",
    )


def test_record_roundtrip_and_replay_are_exact(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "BOXFUSION_PROPOSAL_CACHE_PRODUCER_FINGERPRINT", "producer-1"
    )
    source = _instances()
    protected = protected_proposal_hashes(source)
    geometry = geometry_hash(source)
    recorder = ProposalCache(_config(tmp_path, "record"), torch.device("cpu"))
    recorder.bind_scene("scene0000_00", dataset_length=51, gap=25)

    canonical = recorder.record(
        "scene0000_00",
        0,
        source,
        attempt_id="retry",
        inputs=_inputs(),
    )
    assert canonical is not source
    assert tuple(canonical.get_fields()) == tuple(source.get_fields())
    assert protected_proposal_hashes(canonical) == protected
    assert geometry_hash(canonical) == geometry
    torch.testing.assert_close(
        canonical.pred_proj_xy, source.pred_proj_xy, rtol=0.0, atol=0.0
    )

    prediction = tmp_path / "scene0000_00_boxes.pkl"
    prediction.write_bytes(b"frozen-prediction")
    manifest_path = recorder.finalize("scene0000_00", prediction)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["record_count"] == 1
    assert manifest["proposal_count"] == 2
    assert manifest["records"][0]["attempt_id"] == "retry"

    monkeypatch.setenv(
        "BOXFUSION_PROPOSAL_CACHE_EXPECTED_FINGERPRINT", "producer-1"
    )
    replayer = ProposalCache(_config(tmp_path, "replay"), torch.device("cpu"))
    replayer.bind_scene("scene0000_00", dataset_length=51, gap=25)
    replayed, attempt_id = replayer.replay(
        "scene0000_00", 0, inputs=_inputs()
    )
    assert attempt_id == "retry"
    assert protected_proposal_hashes(replayed) == protected
    assert geometry_hash(replayed) == geometry
    replayer.verify_replay_complete(
        "scene0000_00", baseline_prediction_path=prediction
    )


def test_empty_event_is_recorded_and_replayed(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "BOXFUSION_PROPOSAL_CACHE_PRODUCER_FINGERPRINT", "producer-empty"
    )
    recorder = ProposalCache(_config(tmp_path, "record"), torch.device("cpu"))
    recorder.bind_scene("scene0001_00", dataset_length=1, gap=25)
    canonical = recorder.record(
        "scene0001_00",
        0,
        _instances(count=0),
        attempt_id="primary",
        inputs=_inputs(),
    )
    assert len(canonical) == 0
    prediction = tmp_path / "scene0001_00_boxes.pkl"
    prediction.write_bytes(b"empty-event-prediction")
    recorder.finalize("scene0001_00", prediction)

    monkeypatch.setenv(
        "BOXFUSION_PROPOSAL_CACHE_EXPECTED_FINGERPRINT", "producer-empty"
    )
    replayer = ProposalCache(_config(tmp_path, "replay"), torch.device("cpu"))
    replayer.bind_scene("scene0001_00", dataset_length=1, gap=25)
    replayed, attempt_id = replayer.replay(
        "scene0001_00", 0, inputs=_inputs()
    )
    assert len(replayed) == 0
    assert attempt_id == "primary"
    replayer.verify_replay_complete(
        "scene0001_00", baseline_prediction_path=prediction
    )


def test_replay_rejects_changed_input_and_incomplete_schedule(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(
        "BOXFUSION_PROPOSAL_CACHE_PRODUCER_FINGERPRINT", "producer-2"
    )
    recorder = ProposalCache(_config(tmp_path, "record"), torch.device("cpu"))
    recorder.bind_scene("scene0002_00", dataset_length=51, gap=25)
    recorder.record(
        "scene0002_00",
        0,
        _instances(),
        attempt_id="primary",
        inputs=_inputs(),
    )
    recorder.record(
        "scene0002_00",
        25,
        _instances(count=1),
        attempt_id="primary",
        inputs=_inputs(),
    )
    prediction = tmp_path / "scene0002_00_boxes.pkl"
    prediction.write_bytes(b"prediction")
    recorder.finalize("scene0002_00", prediction)

    monkeypatch.setenv(
        "BOXFUSION_PROPOSAL_CACHE_EXPECTED_FINGERPRINT", "producer-2"
    )
    changed = _inputs()
    changed["depth"] = torch.ones((2, 3)) * 2
    replayer = ProposalCache(_config(tmp_path, "replay"), torch.device("cpu"))
    replayer.bind_scene("scene0002_00", dataset_length=51, gap=25)
    with pytest.raises(ProposalCacheError, match="input differs"):
        replayer.replay("scene0002_00", 0, inputs=changed)

    replayer = ProposalCache(_config(tmp_path, "replay"), torch.device("cpu"))
    replayer.bind_scene("scene0002_00", dataset_length=51, gap=25)
    replayer.replay("scene0002_00", 0, inputs=_inputs())
    with pytest.raises(ProposalCacheError, match="Incomplete"):
        replayer.verify_replay_complete(
            "scene0002_00", baseline_prediction_path=prediction
        )


@pytest.mark.parametrize("backend", ["cutr", "boxer"])
def test_replay_builder_supports_control_and_boxer_backends(
    tmp_path, monkeypatch, backend
):
    monkeypatch.setenv(
        "BOXFUSION_PROPOSAL_CACHE_EXPECTED_FINGERPRINT", "paired-test"
    )
    cfg = {
        "lifting": {
            "backend": backend,
            "proposal_cache": {
                "mode": "replay",
                "root": str(tmp_path / "cache"),
                "namespace": "paired-v1",
            },
        }
    }
    cache = build_proposal_cache(cfg, device=torch.device("cpu"))
    assert cache is not None
    assert cache.is_replay
