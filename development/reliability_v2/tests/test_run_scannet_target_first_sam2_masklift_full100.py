from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_scannet_target_first_sam2_masklift_full100 as module


class _Timing:
    encoder_ms = 1.25
    decoder_and_host_mask_ms = 2.75
    complete_ms = 4.0


class _Result:
    masks = np.ones((2, 480, 640), dtype=np.bool_)
    predicted_ious = np.asarray([0.8, 0.7], dtype=np.float32)
    selected_hypothesis_indices = np.asarray([1, 2], dtype=np.int64)
    timing = _Timing()


class _Provider:
    last_boxes = None

    def __init__(self, *, config):
        self.config = config

    def predict(self, image, boxes):
        assert image.shape == (480, 640, 3)
        assert boxes.shape == (2, 4)
        type(self).last_boxes = boxes.copy()
        return _Result()


def test_engine_adapts_sam2_result_without_reordering() -> None:
    engine = module.SAM2TargetFirstEngine("cuda:7", provider_factory=_Provider)
    result = engine.predict(
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32),
    )
    masks, ious, hypotheses, timing = result
    assert masks.shape == (2, 480, 640)
    assert ious.tolist() == pytest.approx([0.8, 0.7])
    assert hypotheses.tolist() == [1, 2]
    assert timing == {
        "encoder_ms": 1.25,
        "decoder_and_host_mask_ms": 2.75,
        "provider_ms": 4.0,
    }
    assert engine.runtime_metadata()["sam2_device"] == "cuda:7"


def test_engine_converts_legacy_half_open_upper_image_bounds() -> None:
    engine = module.SAM2TargetFirstEngine("cuda:0", provider_factory=_Provider)
    engine.predict(
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.asarray([[1, 2, 640, 480], [5, 6, 7, 8]], dtype=np.float32),
    )
    assert _Provider.last_boxes[0].tolist() == [1.0, 2.0, 639.0, 479.0]


def test_run_restores_base_identity_after_failure(tmp_path: Path) -> None:
    before = (module.base.SCHEMA, module.base.OUTPUT_JSON, module.base.OUTPUT_NPZ, module.base.s3a.MOBILESAM_CHECKPOINT, module.base._scene_order, module.base._process_scene, module.base._canonical_scene_receipts)
    try:
        module.run_shadow(
            receipt_manifest_path=tmp_path / "absent.json",
            raw_log_root=tmp_path,
            schedule_root=tmp_path,
            scene_root=tmp_path,
            scene_list_path=tmp_path / "absent.txt",
            baseline_root=tmp_path,
            output_root=tmp_path / "out",
            device="cpu",
            expected_scene_count=1,
            plan_only=True,
        )
    except module.base.TargetFirstMaskLiftError:
        pass
    else:
        raise AssertionError("fixture must fail before inference")
    after = (module.base.SCHEMA, module.base.OUTPUT_JSON, module.base.OUTPUT_NPZ, module.base.s3a.MOBILESAM_CHECKPOINT, module.base._scene_order, module.base._process_scene, module.base._canonical_scene_receipts)
    assert after == before


def test_nonfrozen_checkpoint_fails_closed(tmp_path: Path) -> None:
    try:
        module.run_shadow(checkpoint=tmp_path / "other.pt")
    except module.base.TargetFirstMaskLiftError as error:
        assert "checkpoint path differs" in str(error)
    else:
        raise AssertionError("nonfrozen checkpoint must fail closed")


def test_scene_start_validation_is_fail_closed() -> None:
    for value in (-1, True):
        with pytest.raises(module.base.TargetFirstMaskLiftError, match="scene-start"):
            module.run_shadow(scene_start=value)
    with pytest.raises(module.base.TargetFirstMaskLiftError, match="cannot be combined"):
        module.run_shadow(scene_start=1, scene="scene0000_00")


def test_scene_start_preserves_official_global_index(monkeypatch, tmp_path: Path) -> None:
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text("s0\ns1\ns2\ns3\n", encoding="utf-8")

    def fake_process_scene(**kwargs):
        return kwargs["scene_index"]

    def fake_canonical(scene_order, tracks):
        return [row["scene_index"] for row in tracks]

    def fake_base_run(**kwargs):
        _full, selected = module.base._scene_order(
            scene_list, 4, None, kwargs["max_scenes"]
        )
        indices = [
            module.base._process_scene(scene_index=index)
            for index, _scene in enumerate(selected)
        ]
        receipts = module.base._canonical_scene_receipts(
            selected, [{"scene_index": index} for index in indices]
        )
        return {"selected": selected, "indices": indices, "receipts": receipts}

    monkeypatch.setattr(module.base, "_process_scene", fake_process_scene)
    monkeypatch.setattr(module.base, "_canonical_scene_receipts", fake_canonical)
    monkeypatch.setattr(module.base, "run_shadow", fake_base_run)
    result = module.run_shadow(scene_start=2, max_scenes=2)
    assert result == {
        "selected": ["s2", "s3"],
        "indices": [2, 3],
        "receipts": [0, 1],
    }


import pytest
