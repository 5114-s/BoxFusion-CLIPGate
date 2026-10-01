from __future__ import annotations

import torch

from boxfusion.boxer_lifter import BoxerLiftingAdapter, BoxerLiftingConfig


def _config(tmp_path) -> BoxerLiftingConfig:
    return BoxerLiftingConfig(
        mode="active",
        apply_stage="post_filter",
        official_root="/unused",
        checkpoint="/unused/model.ckpt",
        expected_commit="test",
        checkpoint_sha256="",
        dinov3_sha256="",
        precision="float32",
        use_sdp=True,
        sdp_samples=8,
        seed=0,
        diagnostics_dir=str(tmp_path),
        cache_image_features=True,
    )


class _CacheModel:
    def __init__(self) -> None:
        self.encode_calls = 0
        self.query_calls = 0

    def prepare_inputs(self, datum):
        return {
            "img0": datum["img0"].clone(),
            "bb2d": datum["bb2d"].clone(),
        }

    def encode(self, inputs):
        self.encode_calls += 1
        inputs["T_world_voxel0"] = torch.tensor([7.0])
        return {"input_enc": inputs["img0"].sum().reshape(1, 1, 1)}

    def query(self, inputs, output):
        self.query_calls += 1
        result = dict(output)
        result["query_value"] = result["input_enc"] + inputs["bb2d"].sum()
        result["pose_value"] = inputs["T_world_voxel0"]
        return result


def _datum(box_value: float):
    return {
        "img0": torch.ones((1, 3, 2, 2)),
        "bb2d": torch.full((1, 1, 4), box_value),
    }


def test_exact_frame_cache_reuses_encode_but_queries_each_box_batch(tmp_path):
    adapter = BoxerLiftingAdapter(_config(tmp_path), device="cpu")
    model = _CacheModel()
    adapter.model = model

    first, _, first_hit = adapter.forward_raw_with_feature_cache(
        _datum(1.0),
        scene_id="scene0000_00",
        frame_id=25,
        encoder_input_sha256="a" * 64,
    )
    second, _, second_hit = adapter.forward_raw_with_feature_cache(
        _datum(2.0),
        scene_id="scene0000_00",
        frame_id=25,
        encoder_input_sha256="a" * 64,
    )

    assert first_hit is False and second_hit is True
    assert model.encode_calls == 1
    assert model.query_calls == 2
    assert float(first["query_value"]) != float(second["query_value"])
    assert adapter._stats["feature_cache_hits"] == 1
    assert adapter._stats["feature_cache_misses"] == 1


def test_same_frame_with_changed_encoder_digest_is_a_safe_cache_miss(tmp_path):
    adapter = BoxerLiftingAdapter(_config(tmp_path), device="cpu")
    model = _CacheModel()
    adapter.model = model

    adapter.forward_raw_with_feature_cache(
        _datum(1.0),
        scene_id="scene0000_00",
        frame_id=25,
        encoder_input_sha256="a" * 64,
    )
    _, _, cache_hit = adapter.forward_raw_with_feature_cache(
        _datum(2.0),
        scene_id="scene0000_00",
        frame_id=25,
        encoder_input_sha256="b" * 64,
    )

    assert cache_hit is False
    assert model.encode_calls == 2
    assert model.query_calls == 2
    assert adapter._stats["feature_cache_hits"] == 0
    assert adapter._stats["feature_cache_misses"] == 2
