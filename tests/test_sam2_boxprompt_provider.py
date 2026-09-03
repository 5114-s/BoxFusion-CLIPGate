from __future__ import annotations

from contextlib import nullcontext
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest

from boxfusion.sam2_boxprompt_provider import (
    EXPECTED_SAM2_CHECKPOINT_BYTES,
    EXPECTED_SAM2_CHECKPOINT_SHA256,
    EXPECTED_SAM2_CONFIG_SHA256,
    EXPECTED_SAM2_SOURCE_FILE_COUNT,
    EXPECTED_SAM2_SOURCE_TREE_SHA256,
    EXPECTED_SAM2_TRANSFORMER_RELATIVE_PATH,
    EXPECTED_SAM2_TRANSFORMER_SHA256,
    EXPECTED_TORCH_ATTENTION_SHA256,
    EXPECTED_TORCH_VERSION,
    MAX_BOXES_PER_FRAME,
    MULTIMASK_HYPOTHESES,
    PRODUCTION_CONFIG,
    SDPA_COMPATIBILITY_POLICY_ID,
    FrozenSAM2BoxPromptProvider,
    SAM2BoxPromptError,
    SAM2BoxPromptTiming,
    _install_sdpa_compatibility_patch,
    _install_verified_production_sdpa_patch,
    _sdpa_backend_names,
)


HEIGHT = 480
WIDTH = 640


class FakePredictor:
    def __init__(self, output, *, fail_predict=False, fail_reset=False):
        self.output = output
        self.fail_predict = fail_predict
        self.fail_reset = fail_reset
        self.set_images = []
        self.predict_calls = []
        self.reset_calls = 0

    def set_image(self, image):
        self.set_images.append(image)

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        if self.fail_predict:
            raise RuntimeError("synthetic decoder failure")
        return self.output

    def reset_predictor(self):
        self.reset_calls += 1
        if self.fail_reset:
            raise RuntimeError("synthetic reset failure")


class FakeCuda:
    def __init__(self):
        self.synchronizations = []
        self.peak_resets = []
        self.peak_reads = []

    def synchronize(self, device):
        self.synchronizations.append(device)

    def reset_peak_memory_stats(self, device):
        self.peak_resets.append(device)

    def max_memory_allocated(self, device):
        self.peak_reads.append(device)
        return 987_654_321


class FakeTorch:
    bfloat16 = object()

    def __init__(self):
        self.cuda = FakeCuda()
        self.inference_contexts = 0
        self.autocast_calls = []

    def inference_mode(self):
        self.inference_contexts += 1
        return nullcontext()

    def autocast(self, **kwargs):
        self.autocast_calls.append(kwargs)
        return nullcontext()


def _image() -> np.ndarray:
    return (
        np.arange(HEIGHT * WIDTH * 3, dtype=np.uint32) % 251
    ).astype(np.uint8).reshape(HEIGHT, WIDTH, 3)


def _two_box_output():
    masks = np.zeros((2, 3, HEIGHT, WIDTH), dtype=np.uint8)
    masks[0, 1, 2, 3] = 1
    masks[0, 2, 2, 4] = 1
    masks[1, 0, 4, 5] = 1
    masks[1, 0, 4, 6] = 0
    ious = np.asarray([[0.2, 0.8, 0.8], [0.9, 0.1, 0.2]], dtype=np.float32)
    low_res = np.zeros((2, 3, 256, 256), dtype=np.float32)
    return masks, ious, low_res


def test_lazy_single_embedding_batched_decode_tie_break_and_source_order() -> None:
    fake = FakePredictor(_two_box_output())
    factory_calls = []

    def factory(config):
        factory_calls.append(config)
        return fake

    provider = FrozenSAM2BoxPromptProvider(predictor_factory=factory)
    assert not provider.loaded
    image = _image()
    original_image = image.copy()
    boxes = np.asarray([[1, 2, 20, 21], [2, 3, 22, 23]], dtype=np.float32)
    original_boxes = boxes.copy()

    result = provider.predict(image, boxes)

    assert provider.loaded
    assert factory_calls == [PRODUCTION_CONFIG]
    assert len(fake.set_images) == 1
    assert len(fake.predict_calls) == 1
    assert fake.reset_calls == 1
    assert fake.set_images[0] is not image
    np.testing.assert_array_equal(fake.set_images[0], original_image)
    np.testing.assert_array_equal(image, original_image)
    call = fake.predict_calls[0]
    assert frozenset(call) == {
        "point_coords",
        "point_labels",
        "box",
        "mask_input",
        "multimask_output",
        "return_logits",
        "normalize_coords",
    }
    assert call["box"] is not boxes
    assert call["point_coords"] is None
    assert call["point_labels"] is None
    assert call["mask_input"] is None
    np.testing.assert_array_equal(call["box"], original_boxes)
    np.testing.assert_array_equal(boxes, original_boxes)
    assert call["multimask_output"] is True
    assert call["return_logits"] is False
    assert call["normalize_coords"] is True

    # Prompt 0 has an exact IoU tie at hypotheses 1 and 2: lowest index wins.
    np.testing.assert_array_equal(result.selected_indices, [1, 0])
    np.testing.assert_allclose(result.predicted_ious, [0.8, 0.9])
    np.testing.assert_allclose(
        result.all_predicted_ious,
        [[0.2, 0.8, 0.8], [0.9, 0.1, 0.2]],
    )
    assert result.masks.shape == (2, HEIGHT, WIDTH)
    assert result.masks[0, 2, 3]
    assert not result.masks[0, 2, 4]
    assert result.masks[1, 4, 5]
    # return_logits=False supplies exact binary masks, so zero is background.
    assert not result.masks[1, 4, 6]
    assert result.count == 2
    assert result.timing.encoder_ms >= 0.0
    assert result.timing.decoder_and_host_mask_ms >= 0.0
    assert result.timing.complete_ms >= 0.0
    assert result.timing.complete_ms == pytest.approx(
        result.timing.encoder_ms + result.timing.decoder_and_host_mask_ms
    )
    assert result.timing.cuda_synchronized is False
    assert result.timing.peak_allocated_memory_bytes == 0
    with pytest.raises(FrozenInstanceError):
        result.timing.complete_ms = 1.0
    for array in (
        result.masks,
        result.selected_hypothesis_indices,
        result.predicted_ious,
        result.all_predicted_ious,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_single_box_accepts_squeezed_sam2_output_and_reuses_only_model() -> None:
    masks = np.zeros((3, HEIGHT, WIDTH), dtype=np.bool_)
    masks[2, 1, 1] = True
    output = (masks, np.asarray([0.1, 0.2, 0.3]), np.zeros((3, 256, 256)))
    fake = FakePredictor(output)
    factory_count = 0

    def factory(config):
        nonlocal factory_count
        factory_count += 1
        return fake

    provider = FrozenSAM2BoxPromptProvider(predictor_factory=factory)
    box = np.asarray([[1, 1, 10, 10]], dtype=np.float32)
    first = provider(_image(), box)
    second = provider(_image(), box)

    assert factory_count == 1
    assert len(fake.set_images) == 2
    assert len(fake.predict_calls) == 2
    assert fake.reset_calls == 2
    assert first.masks.shape == (1, HEIGHT, WIDTH)
    assert first.masks[0, 1, 1]
    np.testing.assert_array_equal(first.selected_indices, [2])
    np.testing.assert_array_equal(second.selected_indices, [2])


@pytest.mark.parametrize(
    "empty_boxes",
    [None, [], np.empty((0, 4), dtype=np.float32)],
)
def test_empty_boxes_are_explicit_and_do_not_load_or_embed(empty_boxes) -> None:
    factory_calls = 0

    def factory(config):
        nonlocal factory_calls
        factory_calls += 1
        return FakePredictor(_two_box_output())

    def forbidden_clock():
        raise AssertionError("empty box batches must not start per-frame timing")

    provider = FrozenSAM2BoxPromptProvider(
        predictor_factory=factory,
        clock_ns=forbidden_clock,
    )
    result = provider(_image(), empty_boxes)

    assert not provider.loaded
    assert factory_calls == 0
    assert result.masks.shape == (0, HEIGHT, WIDTH)
    assert result.selected_indices.shape == (0,)
    assert result.predicted_ious.shape == (0,)
    assert result.all_predicted_ious.shape == (0, MULTIMASK_HYPOTHESES)
    assert result.timing == SAM2BoxPromptTiming(0.0, 0.0, 0.0, False, 0)


def test_cuda_timing_boundaries_cover_reset_and_host_selection() -> None:
    fake = FakePredictor(_two_box_output())
    torch = FakeTorch()
    clock = iter([1_000_000_000, 1_250_000_000, 1_400_000_000])
    provider = FrozenSAM2BoxPromptProvider(
        predictor_factory=lambda config: fake,
        torch_module=torch,
        clock_ns=lambda: next(clock),
    )

    result = provider(_image(), [[1, 2, 20, 21], [2, 3, 22, 23]])

    assert torch.cuda.synchronizations == ["cuda", "cuda", "cuda"]
    assert torch.cuda.peak_resets == ["cuda"]
    assert torch.cuda.peak_reads == ["cuda"]
    assert torch.inference_contexts == 1
    assert torch.autocast_calls == [
        {"device_type": "cuda", "dtype": torch.bfloat16}
    ]
    assert len(fake.set_images) == 1
    assert len(fake.predict_calls) == 1
    assert fake.reset_calls == 1
    assert result.timing == SAM2BoxPromptTiming(
        encoder_ms=250.0,
        decoder_and_host_mask_ms=150.0,
        complete_ms=400.0,
        cuda_synchronized=True,
        peak_allocated_memory_bytes=987_654_321,
    )


@pytest.mark.parametrize(
    ("boxes", "message"),
    [
        (np.zeros((1, 5)), r"shape \[N,4\]"),
        (np.zeros((MAX_BOXES_PER_FRAME + 1, 4)), "at most 16"),
        ([[1, 2, 1, 4]], "non-empty original-image"),
        ([[-1, 2, 4, 5]], "non-empty original-image"),
        ([[1, 2, WIDTH, 5]], "non-empty original-image"),
        ([[1, 2, 4, HEIGHT]], "non-empty original-image"),
        ([[1, 2, WIDTH + 1, 5]], "non-empty original-image"),
        ([[1, np.nan, 4, 5]], "finite"),
    ],
)
def test_invalid_boxes_fail_before_lazy_load(boxes, message) -> None:
    calls = 0

    def factory(config):
        nonlocal calls
        calls += 1
        return FakePredictor(_two_box_output())

    provider = FrozenSAM2BoxPromptProvider(predictor_factory=factory)
    with pytest.raises(ValueError, match=message):
        provider(_image(), boxes)
    assert calls == 0
    assert not provider.loaded


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32),
        np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8),
        np.zeros((HEIGHT - 1, WIDTH, 3), dtype=np.uint8),
        np.zeros((HEIGHT, WIDTH - 1, 3), dtype=np.uint8),
        [[[]]],
    ],
)
def test_invalid_image_fails_before_lazy_load(image) -> None:
    provider = FrozenSAM2BoxPromptProvider(
        predictor_factory=lambda config: FakePredictor(_two_box_output())
    )
    with pytest.raises(ValueError, match="image_rgb"):
        provider(image, [[1, 2, 3, 4]])
    assert not provider.loaded


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda output: None, "masks, IoUs"),
        (lambda output: output[:2], "masks, IoUs"),
        (
            lambda output: (
                np.zeros((1, 3, HEIGHT, WIDTH)),
                output[1],
                output[2],
            ),
            "output shape differs",
        ),
        (
            lambda output: (
                np.full((2, 3, HEIGHT, WIDTH), np.nan),
                output[1],
                output[2],
            ),
            "masks must be finite",
        ),
        (
            lambda output: (
                np.full((2, 3, HEIGHT, WIDTH), 0.5, dtype=np.float32),
                output[1],
                output[2],
            ),
            r"exact binary \{0,1\}",
        ),
        (
            lambda output: (
                np.full((2, 3, HEIGHT, WIDTH), -1, dtype=np.int8),
                output[1],
                output[2],
            ),
            r"exact binary \{0,1\}",
        ),
        (
            lambda output: (
                output[0],
                np.full((2, 3), np.nan),
                output[2],
            ),
            "IoUs must be finite",
        ),
    ],
)
def test_malformed_sam2_outputs_fail_closed_after_state_reset(mutate, message) -> None:
    fake = FakePredictor(mutate(_two_box_output()))
    provider = FrozenSAM2BoxPromptProvider(predictor_factory=lambda config: fake)
    with pytest.raises(SAM2BoxPromptError, match=message):
        provider(_image(), [[1, 2, 20, 21], [2, 3, 22, 23]])
    assert fake.reset_calls == 1


def test_decoder_failure_resets_state_and_reset_failure_poisons_provider() -> None:
    failing_predict = FakePredictor(_two_box_output(), fail_predict=True)
    provider = FrozenSAM2BoxPromptProvider(
        predictor_factory=lambda config: failing_predict
    )
    with pytest.raises(SAM2BoxPromptError, match="inference failed"):
        provider(_image(), [[1, 2, 20, 21]])
    assert failing_predict.reset_calls == 1

    failing_reset = FakePredictor(_two_box_output(), fail_reset=True)
    poisoned = FrozenSAM2BoxPromptProvider(
        predictor_factory=lambda config: failing_reset
    )
    with pytest.raises(SAM2BoxPromptError, match="cleanup failed"):
        poisoned(_image(), [[1, 2, 20, 21], [2, 3, 22, 23]])
    with pytest.raises(SAM2BoxPromptError, match="poisoned"):
        poisoned(_image(), [[1, 2, 20, 21]])


def test_predictor_contract_is_checked_lazily() -> None:
    provider = FrozenSAM2BoxPromptProvider(predictor_factory=lambda config: object())
    assert not provider.loaded
    with pytest.raises(SAM2BoxPromptError, match="lacks callable set_image"):
        provider(_image(), [[1, 2, 20, 21]])


def test_production_config_pins_source_config_checkpoint_and_policy() -> None:
    assert PRODUCTION_CONFIG.source_file_glob == "sam2/**/*.py"
    assert PRODUCTION_CONFIG.source_file_count == EXPECTED_SAM2_SOURCE_FILE_COUNT == 23
    assert (
        PRODUCTION_CONFIG.source_tree_sha256
        == EXPECTED_SAM2_SOURCE_TREE_SHA256
        == "cc5a594bab1508ab69cbedfbb83ba8e226f848dd142a3deba8c195ee1e2469cf"
    )
    assert (
        PRODUCTION_CONFIG.config_sha256
        == EXPECTED_SAM2_CONFIG_SHA256
        == "545e4325aa5c19a1615d43c946b07276ed4c57214eacf1437e38fa3d9374f636"
    )
    assert PRODUCTION_CONFIG.checkpoint_bytes == EXPECTED_SAM2_CHECKPOINT_BYTES
    assert (
        PRODUCTION_CONFIG.checkpoint_sha256
        == EXPECTED_SAM2_CHECKPOINT_SHA256
        == "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
    )
    assert PRODUCTION_CONFIG.config_name == "configs/sam2.1/sam2.1_hiera_l.yaml"
    assert PRODUCTION_CONFIG.multimask_output is True
    assert PRODUCTION_CONFIG.return_logits is False
    assert PRODUCTION_CONFIG.normalize_coords is True
    assert PRODUCTION_CONFIG.mask_threshold == 0.0
    assert PRODUCTION_CONFIG.max_boxes_per_frame == 16
    assert PRODUCTION_CONFIG.multimask_hypotheses == 3
    assert (
        PRODUCTION_CONFIG.sdpa_compatibility_policy_id
        == SDPA_COMPATIBILITY_POLICY_ID
    )
    assert PRODUCTION_CONFIG.torch_version == EXPECTED_TORCH_VERSION
    assert (
        PRODUCTION_CONFIG.transformer_relative_path
        == EXPECTED_SAM2_TRANSFORMER_RELATIVE_PATH
    )
    assert PRODUCTION_CONFIG.transformer_sha256 == EXPECTED_SAM2_TRANSFORMER_SHA256
    assert PRODUCTION_CONFIG.torch_attention_sha256 == EXPECTED_TORCH_ATTENTION_SHA256

    with pytest.raises(ValueError, match="selection policy differs"):
        FrozenSAM2BoxPromptProvider(
            config=replace(PRODUCTION_CONFIG, return_logits=True),
            predictor_factory=lambda config: FakePredictor(_two_box_output()),
        )


@pytest.mark.parametrize(
    ("old_gpu", "flash", "math", "dropout", "expected"),
    [
        (False, True, False, 0.0, ("FLASH_ATTENTION", "CUDNN_ATTENTION")),
        (
            True,
            False,
            False,
            0.0,
            ("EFFICIENT_ATTENTION", "CUDNN_ATTENTION"),
        ),
        (
            True,
            False,
            False,
            0.1,
            ("EFFICIENT_ATTENTION", "MATH", "CUDNN_ATTENTION"),
        ),
        (False, False, True, 0.0, ("MATH", "CUDNN_ATTENTION")),
    ],
)
def test_sdpa_compatibility_mapping_preserves_all_old_flags(
    old_gpu, flash, math, dropout, expected
) -> None:
    assert _sdpa_backend_names(
        old_gpu=old_gpu,
        use_flash_attention=flash,
        math_kernel_on=math,
        dropout_p=dropout,
    ) == expected


class _FakeSDPBackend:
    FLASH_ATTENTION = "flash"
    EFFICIENT_ATTENTION = "efficient"
    MATH = "math"
    CUDNN_ATTENTION = "cudnn"


class _FakeAttentionModule:
    SDPBackend = _FakeSDPBackend

    def __init__(self) -> None:
        self.calls = []

    def sdpa_kernel(self, backends):
        self.calls.append(tuple(backends))
        return nullcontext()


def _fake_transformer(**overrides):
    values = {
        "OLD_GPU": False,
        "USE_FLASH_ATTN": True,
        "MATH_KERNEL_ON": False,
        "ALLOW_ALL_KERNELS": False,
        "sdp_kernel_context": lambda dropout_p: nullcontext(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_sdpa_compatibility_patch_is_idempotent_and_keeps_allow_all_fallback() -> None:
    transformer = _fake_transformer()
    attention = _FakeAttentionModule()

    first = _install_sdpa_compatibility_patch(transformer, attention)
    second = _install_sdpa_compatibility_patch(transformer, attention)
    assert first is second is transformer.sdp_kernel_context
    with first(0.0):
        pass
    assert attention.calls == [("flash", "cudnn")]

    transformer.ALLOW_ALL_KERNELS = True
    with first(0.0):
        pass
    assert attention.calls == [("flash", "cudnn")]


def test_sdpa_compatibility_patch_fails_closed_on_api_flags_and_foreign_patch() -> None:
    with pytest.raises(SAM2BoxPromptError, match="OLD_GPU flag differs"):
        _install_sdpa_compatibility_patch(
            _fake_transformer(OLD_GPU=1), _FakeAttentionModule()
        )

    missing_api = SimpleNamespace(SDPBackend=_FakeSDPBackend)
    with pytest.raises(SAM2BoxPromptError, match="API is absent"):
        _install_sdpa_compatibility_patch(_fake_transformer(), missing_api)

    foreign = lambda dropout_p: nullcontext()
    setattr(
        foreign,
        "__boxfusion_n0a_sdpa_compatibility_policy_id__",
        "foreign-policy",
    )
    with pytest.raises(SAM2BoxPromptError, match="incompatible"):
        _install_sdpa_compatibility_patch(
            _fake_transformer(sdp_kernel_context=foreign), _FakeAttentionModule()
        )

    with pytest.raises(SAM2BoxPromptError, match="frozen torch 2.5.1"):
        _install_verified_production_sdpa_patch(
            source_root=PRODUCTION_CONFIG.source_root,
            torch_module=SimpleNamespace(__version__="2.5.2"),
        )


def test_sdpa_compatibility_patch_revalidates_mutated_runtime_flags() -> None:
    transformer = _fake_transformer()
    installed = _install_sdpa_compatibility_patch(
        transformer, _FakeAttentionModule()
    )
    transformer.MATH_KERNEL_ON = "false"
    with pytest.raises(SAM2BoxPromptError, match="runtime SDP flag"):
        installed(0.0)
