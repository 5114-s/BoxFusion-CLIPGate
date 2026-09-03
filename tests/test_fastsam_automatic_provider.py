from __future__ import annotations

from contextlib import nullcontext
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from boxfusion import fastsam_automatic_provider as provider_module
from boxfusion.fastsam_automatic_provider import (
    EXPECTED_CHECKPOINT_BYTES,
    EXPECTED_CHECKPOINT_SHA256,
    FastSAMCheckpointIdentity,
    FastSAMProviderError,
    FrozenFastSAMAutomaticMaskProvider,
    PREDICT_POLICY,
)


HEIGHT = 480
WIDTH = 640


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeParameter:
    def __init__(self):
        self.requires_grad = True


class FakeModel:
    def __init__(self, result, *, thaw_during_predict=False):
        self.result = result
        self.training = True
        self.params = [FakeParameter(), FakeParameter()]
        self.predict_calls = []
        self.thaw_during_predict = thaw_during_predict

    def eval(self):
        self.training = False
        return self

    def requires_grad_(self, enabled):
        for parameter in self.params:
            parameter.requires_grad = bool(enabled)
        return self

    def parameters(self):
        return iter(self.params)

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        if self.thaw_during_predict:
            self.params[0].requires_grad = True
        return self.result


class FakeCuda:
    def __init__(self, *, available=True):
        self.available = available
        self.synchronizations = []
        self.peak_resets = []

    def is_available(self):
        return self.available

    def synchronize(self, device):
        self.synchronizations.append(device)

    def reset_peak_memory_stats(self, device):
        self.peak_resets.append(device)

    def memory_allocated(self, device):
        return 101

    def memory_reserved(self, device):
        return 202

    def max_memory_allocated(self, device):
        return 303

    def max_memory_reserved(self, device):
        return 404


class FakeTorch:
    def __init__(self, *, cuda_available=True):
        self.cuda = FakeCuda(available=cuda_available)
        self.inference_context_count = 0

    def device(self, value):
        return f"torch-device:{value}"

    def inference_mode(self):
        self.inference_context_count += 1
        return nullcontext()


def make_result(*, count=2, masks=True):
    boxes = np.asarray(
        [[10.0, 20.0, 30.0, 40.0], [100.0, 120.0, 150.0, 180.0]],
        dtype=np.float32,
    )[:count]
    conf = np.asarray([0.75, 0.50], dtype=np.float32)[:count]
    boxes_object = SimpleNamespace(
        xyxy=FakeTensor(boxes),
        conf=FakeTensor(conf),
        # These fields must be ignored by the provider.
        cls=FakeTensor(np.arange(count)),
    )
    if masks:
        mask_values = np.zeros((count, HEIGHT, WIDTH), dtype=np.float32)
        if count:
            mask_values[0, 10, 10:13] = [0.4999, 0.5, 1.0]
        masks_object = SimpleNamespace(data=FakeTensor(mask_values))
    else:
        masks_object = None
    return [
        SimpleNamespace(
            orig_shape=(HEIGHT, WIDTH),
            boxes=boxes_object,
            masks=masks_object,
            names={0: "must-not-escape"},
        )
    ]


@pytest.fixture
def verified_checkpoint(monkeypatch):
    identity = FastSAMCheckpointIdentity(
        path="/verified/FastSAM.pt",
        byte_count=EXPECTED_CHECKPOINT_BYTES,
        sha256=EXPECTED_CHECKPOINT_SHA256,
    )
    monkeypatch.setattr(provider_module, "_validate_checkpoint", lambda path: identity)
    return identity


def test_exact_frozen_fastsam_call_outputs_and_synchronized_timing(verified_checkpoint):
    model = FakeModel(make_result())
    factory_inputs = []

    def factory(path):
        factory_inputs.append(path)
        return model

    torch = FakeTorch()
    clock = iter([1_000_000_000, 1_250_000_000, 1_300_000_000])
    provider = FrozenFastSAMAutomaticMaskProvider(
        "ignored-by-fixture.pt",
        device="cuda:1",
        model_factory=factory,
        torch_module=torch,
        clock_ns=lambda: next(clock),
    )
    image = np.arange(HEIGHT * WIDTH * 3, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
    original = image.copy()
    result = provider.predict(image)

    assert provider.checkpoint == verified_checkpoint
    assert provider.device == "cuda:1"
    assert factory_inputs == ["/verified/FastSAM.pt"]
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.params)
    assert torch.inference_context_count == 1

    assert len(model.predict_calls) == 1
    call = model.predict_calls[0]
    assert frozenset(call) == {
        "source",
        "imgsz",
        "conf",
        "iou",
        "max_det",
        "agnostic_nms",
        "retina_masks",
        "classes",
        "augment",
        "half",
        "batch",
        "device",
        "verbose",
        "save",
        "stream",
    }
    assert call | {"source": None} == {
        "source": None,
        "imgsz": 1024,
        "conf": 0.25,
        "iou": 0.90,
        "max_det": 100,
        "agnostic_nms": True,
        "retina_masks": True,
        "classes": None,
        "augment": False,
        "half": False,
        "batch": 1,
        "device": "cuda:1",
        "verbose": False,
        "save": False,
        "stream": False,
    }
    assert isinstance(call["source"], list) and len(call["source"]) == 1
    assert call["source"][0] is not image
    np.testing.assert_array_equal(call["source"][0], original)
    np.testing.assert_array_equal(image, original)

    assert result.masks.dtype == np.bool_
    assert result.masks.shape == (2, HEIGHT, WIDTH)
    assert not result.masks[0, 10, 10]
    assert result.masks[0, 10, 11]
    assert result.masks[0, 10, 12]
    np.testing.assert_array_equal(result.conf, [0.75, 0.50])
    np.testing.assert_array_equal(
        result.boxes,
        [[10.0, 20.0, 30.0, 40.0], [100.0, 120.0, 150.0, 180.0]],
    )
    assert result.count == 2
    for array in (result.masks, result.confidences, result.boxes_xyxy):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    assert not hasattr(result, "classes")
    assert not hasattr(result, "names")

    assert torch.cuda.synchronizations == [
        "torch-device:cuda:1",
        "torch-device:cuda:1",
    ]
    assert torch.cuda.peak_resets == ["torch-device:cuda:1"]
    timing = result.timing
    assert timing.cuda_synchronized
    assert timing.prediction_seconds == pytest.approx(0.25)
    assert timing.extraction_seconds == pytest.approx(0.05)
    assert timing.total_seconds == pytest.approx(0.30)
    assert timing.memory_allocated_before_bytes == 101
    assert timing.memory_allocated_after_bytes == 101
    assert timing.memory_reserved_before_bytes == 202
    assert timing.memory_reserved_after_bytes == 202
    assert timing.max_memory_allocated_bytes == 303
    assert timing.max_memory_reserved_bytes == 404


def test_empty_none_masks_are_allowed_only_with_zero_boxes(verified_checkpoint):
    empty_result = [
        SimpleNamespace(
            orig_shape=(HEIGHT, WIDTH),
            boxes=SimpleNamespace(
                xyxy=FakeTensor(np.empty((0, 4))),
                conf=FakeTensor(np.empty((0,))),
            ),
            masks=None,
        )
    ]
    provider = FrozenFastSAMAutomaticMaskProvider(
        "unused.pt",
        device="cpu",
        model_factory=lambda path: FakeModel(empty_result),
        torch_module=FakeTorch(),
        clock_ns=iter([1, 2, 3]).__next__,
    )
    output = provider(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
    assert output.masks.shape == (0, HEIGHT, WIDTH)
    assert output.confidences.shape == (0,)
    assert output.boxes_xyxy.shape == (0, 4)
    assert not output.timing.cuda_synchronized
    assert output.timing.max_memory_allocated_bytes == 0

    with pytest.raises(FastSAMProviderError, match="boxes without masks"):
        provider_module._extract_result(make_result(count=1, masks=False))


def test_checkpoint_primitive_rejects_symlink_size_and_digest(tmp_path):
    payload = b"frozen-fastsam-test-checkpoint"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    identity = provider_module._validate_checkpoint(
        checkpoint,
        expected_bytes=len(payload),
        expected_sha256=digest,
    )
    assert identity.path == str(checkpoint.resolve())
    assert identity.byte_count == len(payload)
    assert identity.sha256 == digest

    with pytest.raises(FastSAMProviderError, match="byte count differs"):
        provider_module._validate_checkpoint(
            checkpoint,
            expected_bytes=len(payload) + 1,
            expected_sha256=digest,
        )
    with pytest.raises(FastSAMProviderError, match="SHA-256 differs"):
        provider_module._validate_checkpoint(
            checkpoint,
            expected_bytes=len(payload),
            expected_sha256="0" * 64,
        )

    link = tmp_path / "link.pt"
    link.symlink_to(checkpoint)
    with pytest.raises(FastSAMProviderError, match="non-symlink regular file"):
        provider_module._validate_checkpoint(
            link,
            expected_bytes=len(payload),
            expected_sha256=digest,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda result: result.append(result[0]), "exactly one result"),
        (lambda result: setattr(result[0], "orig_shape", (479, 640)), "orig_shape"),
        (
            lambda result: setattr(
                result[0].boxes,
                "xyxy",
                FakeTensor([[10.0, 20.0, 641.0, 40.0], [1.0, 2.0, 3.0, 4.0]]),
            ),
            "outside image bounds",
        ),
        (
            lambda result: setattr(
                result[0].boxes, "conf", FakeTensor([np.nan, 0.5])
            ),
            "finite",
        ),
        (
            lambda result: setattr(result[0].boxes, "conf", FakeTensor([1.1, 0.5])),
            r"within \[0,1\]",
        ),
        (
            lambda result: setattr(
                result[0].masks,
                "data",
                FakeTensor(np.zeros((1, HEIGHT, WIDTH), dtype=np.float32)),
            ),
            "matching boxes",
        ),
        (
            lambda result: setattr(
                result[0].masks,
                "data",
                FakeTensor(np.full((2, HEIGHT, WIDTH), np.nan, dtype=np.float32)),
            ),
            "finite",
        ),
        (
            lambda result: setattr(
                result[0].masks,
                "data",
                FakeTensor(np.full((2, HEIGHT, WIDTH), 1.01, dtype=np.float32)),
            ),
            r"within \[0,1\]",
        ),
    ],
)
def test_malformed_fastsam_results_fail_closed(mutate, message):
    result = make_result()
    mutate(result)
    with pytest.raises(FastSAMProviderError, match=message):
        provider_module._extract_result(result)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32),
        np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8),
        np.zeros((HEIGHT - 1, WIDTH, 3), dtype=np.uint8),
        [[[]]],
    ],
)
def test_invalid_bgr_input_fails_before_model_call(verified_checkpoint, image):
    model = FakeModel(make_result())
    provider = FrozenFastSAMAutomaticMaskProvider(
        "unused.pt",
        device="cpu",
        model_factory=lambda path: model,
        torch_module=FakeTorch(),
    )
    with pytest.raises(ValueError, match="image_bgr"):
        provider.predict(image)
    assert not model.predict_calls


def test_model_that_becomes_trainable_during_predict_fails_closed(verified_checkpoint):
    model = FakeModel(make_result(), thaw_during_predict=True)
    provider = FrozenFastSAMAutomaticMaskProvider(
        "unused.pt",
        device="cpu",
        model_factory=lambda path: model,
        torch_module=FakeTorch(),
    )
    with pytest.raises(FastSAMProviderError, match="trainable parameters"):
        provider.predict(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))


def test_cuda_request_fails_closed_if_cuda_is_unavailable(verified_checkpoint):
    with pytest.raises(FastSAMProviderError, match="CUDA is unavailable"):
        FrozenFastSAMAutomaticMaskProvider(
            "unused.pt",
            device="cuda:0",
            model_factory=lambda path: FakeModel(make_result()),
            torch_module=FakeTorch(cuda_available=False),
        )


def test_public_policy_freezes_exact_nonsemantic_prediction_contract():
    assert PREDICT_POLICY == {
        "imgsz": 1024,
        "conf": 0.25,
        "iou": 0.90,
        "max_det": 100,
        "agnostic_nms": True,
        "retina_masks": True,
        "classes": None,
        "augment": False,
        "half": False,
        "batch": 1,
        "verbose": False,
        "save": False,
        "stream": False,
        "source_container": "one_element_list",
        "source_color_order": "BGR",
        "source_dtype": "uint8",
        "source_shape": (480, 640, 3),
        "mask_threshold": 0.5,
        "semantic_outputs": None,
    }
    with pytest.raises(TypeError):
        PREDICT_POLICY["conf"] = 0.26
