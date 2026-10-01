from __future__ import annotations

import numpy as np
import pytest
import torch

from boxfusion.clip_instance_features import ClipInstanceFeatureEncoder
from boxfusion.supplemental_proposals import SupplementalProposal


class FakeClip(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.last_batch = None

    def encode_image(self, batch):
        self.last_batch = batch.detach().cpu()
        means = batch.mean(dim=(1, 2, 3))
        return torch.stack(
            (means + 1.0, means + 2.0, means + 3.0), dim=1
        )


def preprocess(image):
    value = np.asarray(image, dtype=np.float32)
    return torch.from_numpy(value).permute(2, 0, 1) / 255.0


def proposal(mask, bbox=(1.0, 1.0, 5.0, 5.0)):
    return SupplementalProposal(
        bbox=np.asarray(bbox, dtype=np.float32),
        score=0.8,
        mask=mask,
    )


def test_clip_encoder_batches_normalized_masked_crops():
    image = np.zeros((6, 6, 3), dtype=np.uint8)
    image[1:5, 1:5] = 100
    mask = np.zeros((6, 6), dtype=bool)
    mask[2:4, 2:4] = True
    model = FakeClip()
    encoder = ClipInstanceFeatureEncoder(
        model, preprocess, masked_crop=True, background_value=255
    )

    features = encoder(image, [proposal(mask), proposal(mask)])

    assert len(features) == 2
    assert features[0].shape == (3,)
    assert np.linalg.norm(features[0]) == pytest.approx(1.0)
    assert model.last_batch.shape == (2, 3, 4, 4)
    assert torch.any(model.last_batch == 1.0)


def test_clip_encoder_unmasked_crop_and_empty_batch():
    image = np.full((6, 6, 3), 64, dtype=np.uint8)
    mask = np.zeros((6, 6), dtype=bool)
    model = FakeClip()
    encoder = ClipInstanceFeatureEncoder(
        model, preprocess, masked_crop=False
    )
    assert encoder(image, []) == []
    features = encoder(image, [proposal(mask)])
    assert len(features) == 1
    assert torch.allclose(
        model.last_batch,
        torch.full_like(model.last_batch, 64.0 / 255.0),
    )


def test_clip_encoder_rejects_mask_shape_and_zero_features():
    image = np.zeros((6, 6, 3), dtype=np.uint8)
    wrong_mask = np.ones((5, 6), dtype=bool)
    encoder = ClipInstanceFeatureEncoder(FakeClip(), preprocess)
    with pytest.raises(ValueError, match="mask must match"):
        encoder(image, [proposal(wrong_mask)])

    class ZeroClip(FakeClip):
        def encode_image(self, batch):
            return torch.zeros((len(batch), 3), device=batch.device)

    with pytest.raises(ValueError, match="non-zero norm"):
        ClipInstanceFeatureEncoder(ZeroClip(), preprocess)(
            image, [proposal(np.ones((6, 6), dtype=bool))]
        )
