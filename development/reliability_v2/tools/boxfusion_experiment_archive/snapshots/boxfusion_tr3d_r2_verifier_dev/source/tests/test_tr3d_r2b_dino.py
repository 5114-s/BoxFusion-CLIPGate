from __future__ import annotations

import numpy as np
import pytest
import torch

from boxfusion.tr3d_r2b_dino import (
    BOXER_DINO_SHA256,
    BOXER_OFFICIAL_COMMIT,
    BoxerDINOv3Config,
    BoxerDINOv3DenseEncoder,
)


def _config(**overrides) -> BoxerDINOv3Config:
    values = {
        "official_root": "/not-used-by-fake-model",
        "expected_commit": BOXER_OFFICIAL_COMMIT,
        "checkpoint_sha256": BOXER_DINO_SHA256,
        "input_height": 32,
        "input_width": 48,
        "precision": "float32",
        "device": "cpu",
    }
    values.update(overrides)
    return BoxerDINOv3Config(**values)


def test_config_requires_patch_aligned_input() -> None:
    with pytest.raises(ValueError, match="multiples of 16"):
        _config(input_width=47)
    with pytest.raises(ValueError, match="precision"):
        _config(precision="float16")


def test_encoder_matches_boxer_stretch_and_returns_chw_float32() -> None:
    observed = {}

    class FakeModel:
        def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
            observed["shape"] = tuple(tensor.shape)
            observed["range"] = (float(tensor.min()), float(tensor.max()))
            return torch.ones((1, 7, 2, 3), dtype=torch.float64)

    encoder = BoxerDINOv3DenseEncoder(_config())
    encoder.model = FakeModel()
    image = np.full((9, 11, 3), 128, dtype=np.uint8)
    result = encoder(image)
    assert observed["shape"] == (1, 3, 32, 48)
    assert observed["range"] == pytest.approx((128 / 255, 128 / 255))
    assert result.shape == (7, 2, 3)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous


def test_float_unit_range_is_converted_to_uint8_like_boxer() -> None:
    encoder = BoxerDINOv3DenseEncoder(_config())
    converted = encoder._uint8_rgb(
        np.asarray([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    )
    np.testing.assert_array_equal(converted, [[[0, 128, 255]]])
