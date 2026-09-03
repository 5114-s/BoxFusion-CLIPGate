from contextlib import nullcontext

import numpy as np
import pytest

from boxfusion.sam2_video_track_provider import (
    FrozenSAM2VideoTrackProvider,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)


class _FakeCuda:
    @staticmethod
    def is_available():
        return False


class _FakeTorch:
    cuda = _FakeCuda()

    @staticmethod
    def inference_mode():
        return nullcontext()


class _FakePredictor:
    def __init__(self):
        self.events = []

    def init_state(self, **kwargs):
        self.events.append(("init", kwargs))
        return {"num_frames": 0, "commits": []}

    def add_new_frame(self, state, image):
        index = state["num_frames"]
        state["num_frames"] += 1
        self.events.append(("add", index, int(image[0, 0, 0])))
        return index

    def infer_single_frame(self, state, index):
        # The output encodes how many masks were already committed.  A leaked
        # current correction would therefore change this deterministic mask.
        prior_count = len(state["commits"])
        self.events.append(("infer", index, prior_count))
        logits = np.full((1, 1, IMAGE_HEIGHT, IMAGE_WIDTH), -1.0, dtype=np.float32)
        logits[:, :, :prior_count, :prior_count] = 1.0
        return index, [1], logits

    def add_new_mask(self, state, index, object_id, mask):
        self.events.append(("commit", index, object_id))
        state["commits"].append((index, np.array(mask, copy=True)))
        return index, [object_id], np.zeros(
            (1, 1, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32
        )

    def reset_state(self, state):
        self.events.append(("reset", len(state["commits"])))


def _inputs(count=3):
    images = np.zeros((count, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    masks = np.zeros((count, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.bool_)
    for index in range(count):
        images[index, 0, 0, 0] = index
        masks[index, index : index + 2, index : index + 2] = True
    return images, masks


def test_query_happens_before_current_commit_and_seed_is_exact():
    fake = _FakePredictor()
    provider = FrozenSAM2VideoTrackProvider(
        predictor_factory=lambda config: (fake, _FakeTorch())
    )
    images, masks = _inputs(3)
    result = provider.predict_track(images_rgb=images, frozen_masks=masks)

    assert np.array_equal(result.masks[0], masks[0])
    assert np.count_nonzero(result.masks[1]) == 1
    assert np.count_nonzero(result.masks[2]) == 4
    assert result.predicted_flags == (False, True, True)
    assert result.maximum_lookahead_observations == 0
    assert fake.events[1:] == [
        ("add", 0, 0),
        ("commit", 0, 1),
        ("add", 1, 1),
        ("infer", 1, 1),
        ("commit", 1, 1),
        ("add", 2, 2),
        ("infer", 2, 2),
        ("commit", 2, 1),
        ("reset", 3),
    ]


@pytest.mark.parametrize("count", [0, 2, 6])
def test_track_bound_fails_closed(count):
    provider = FrozenSAM2VideoTrackProvider(
        predictor_factory=lambda config: (_FakePredictor(), _FakeTorch())
    )
    images = np.zeros((count, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    masks = np.zeros((count, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.bool_)
    with pytest.raises(ValueError):
        provider.predict_track(images_rgb=images, frozen_masks=masks)


def test_empty_observation_mask_fails_before_model_load():
    called = False

    def factory(config):
        nonlocal called
        called = True
        return _FakePredictor(), _FakeTorch()

    provider = FrozenSAM2VideoTrackProvider(predictor_factory=factory)
    images, masks = _inputs(3)
    masks[1] = False
    with pytest.raises(ValueError):
        provider.predict_track(images_rgb=images, frozen_masks=masks)
    assert not called
