import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "boxfusion"
    / "supplemental_proposals.py"
)
spec = importlib.util.spec_from_file_location(
    "boxfusion_supplemental_proposals", SOURCE
)
supplemental = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = supplemental
spec.loader.exec_module(supplemental)


def proposal(shape=(6, 8), *, offset=0.0, label="chair", feature=True):
    mask = np.zeros(shape, dtype=np.bool_)
    mask[1:5, 2:7] = True
    return supplemental.SupplementalProposal(
        bbox=np.asarray([1.0 + offset, 1.0, 7.0, 5.0]),
        score=0.8,
        mask=mask,
        label=label,
        feature=np.asarray([1.0, 2.0]) if feature else None,
    )


def test_proposal_normalizes_and_freezes_arrays():
    item = proposal()
    assert item.bbox.dtype == np.float32
    assert item.mask.dtype == np.bool_
    assert item.feature.dtype == np.float32
    assert item.label == "chair"
    assert not item.bbox.flags.writeable
    assert not item.mask.flags.writeable
    assert not item.feature.flags.writeable


@pytest.mark.parametrize(
    "kwargs, exception",
    [
        ({"bbox": [0.0, 0.0, 1.0]}, ValueError),
        ({"bbox": [0.0, 0.0, np.nan, 1.0]}, ValueError),
        ({"bbox": [1.0, 0.0, 1.0, 1.0]}, ValueError),
        ({"score": np.inf}, ValueError),
        ({"score": 1.1}, ValueError),
        ({"mask": np.zeros((2, 2, 1))}, ValueError),
        ({"mask": np.asarray([[0.0, np.nan]])}, ValueError),
        ({"mask": np.asarray([[0, 2]])}, ValueError),
        ({"label": ""}, ValueError),
        ({"feature": np.asarray([[1.0]])}, ValueError),
        ({"feature": np.asarray([np.inf])}, ValueError),
    ],
)
def test_proposal_rejects_invalid_fields(kwargs, exception):
    valid = {
        "bbox": [0.0, 0.0, 2.0, 2.0],
        "score": 0.5,
        "mask": np.ones((2, 2), dtype=np.bool_),
        "label": None,
        "feature": None,
    }
    valid.update(kwargs)
    with pytest.raises(exception):
        supplemental.SupplementalProposal(**valid)


def test_missing_config_builds_a_true_noop_without_importing_yoloe(monkeypatch):
    imported = []
    original_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == "ultralytics":
            imported.append(name)
            raise AssertionError("disabled provider imported YOLOE")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    provider = supplemental.build_provider({}, "cuda:0")
    images = [object(), object()]
    assert provider.predict(images, frame_ids=["a", "b"]) == [[], []]
    assert imported == []


def test_enabled_provider_is_still_lazy_and_missing_dependency_is_clear(
    monkeypatch,
):
    provider = supplemental.build_provider(
        {
            "enabled": True,
            "checkpoint": "model.pt",
            "mode": "prompt_free",
        },
        "cpu",
    )
    assert isinstance(provider, supplemental.YOLOEProposalProvider)
    assert provider._model is None

    def missing():
        raise RuntimeError(
            "The YOLOE supplemental proposal provider is enabled, but "
            "YOLOE is unavailable."
        )

    monkeypatch.setattr(provider, "_import_yoloe", missing)
    with pytest.raises(RuntimeError, match="YOLOE is unavailable"):
        provider.predict([np.zeros((4, 5, 3), dtype=np.uint8)])


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


def fake_result(index, *, source_shape=(6, 8), mask_shape=(3, 4)):
    boxes = types.SimpleNamespace(
        xyxy=FakeTensor([[1.0, 1.0, 7.0, 5.0]]),
        conf=FakeTensor([0.75 + 0.01 * index]),
        cls=FakeTensor([1.0]),
    )
    mask = np.zeros((1,) + mask_shape, dtype=np.float32)
    mask[:, 1:, 1:3] = 0.9
    masks = types.SimpleNamespace(data=FakeTensor(mask))
    return types.SimpleNamespace(
        boxes=boxes,
        masks=masks,
        names={0: "table", 1: "chair"},
        orig_shape=source_shape,
    )


class FakeYOLOE:
    instances = []

    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.text_requests = []
        self.predict_calls = []
        self.__class__.instances.append(self)

    def get_text_pe(self, prompts):
        self.text_requests.append(("embedding", list(prompts)))
        return np.ones((1, len(prompts), 4), dtype=np.float32)

    def set_classes(self, prompts, embeddings):
        self.text_requests.append(("classes", list(prompts), embeddings.shape))

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        return [
            fake_result(index, source_shape=image.shape[:2])
            for index, image in enumerate(kwargs["source"])
        ]


def install_fake_yoloe(monkeypatch):
    FakeYOLOE.instances.clear()
    module = types.ModuleType("ultralytics")
    module.YOLOE = FakeYOLOE
    monkeypatch.setitem(sys.modules, "ultralytics", module)


def test_text_mode_batches_images_and_extracts_boxes_masks_labels(monkeypatch):
    install_fake_yoloe(monkeypatch)
    provider = supplemental.YOLOEProposalProvider(
        checkpoint="yoloe-seg.pt",
        device="cuda:1",
        mode="text",
        prompts=["chair", "table"],
        confidence=0.2,
        iou=0.6,
        image_size=320,
        max_detections=25,
        mask_threshold=0.5,
    )
    image_a = np.zeros((6, 8, 3), dtype=np.uint8)
    image_a[..., 0] = 10
    image_a[..., 2] = 30
    image_b = np.zeros((10, 12, 3), dtype=np.uint8)
    batches = provider.predict([image_a, image_b], frame_ids=["0", "1"])

    assert len(FakeYOLOE.instances) == 1
    model = FakeYOLOE.instances[0]
    assert model.text_requests[0] == (
        "embedding",
        ["chair", "table"],
    )
    assert model.text_requests[1][:2] == (
        "classes",
        ["chair", "table"],
    )
    call = model.predict_calls[0]
    assert call["device"] == "cuda:1"
    assert call["conf"] == 0.2
    assert call["iou"] == 0.6
    assert call["imgsz"] == 320
    assert call["max_det"] == 25
    assert call["source"][0][0, 0].tolist() == [30, 0, 10]

    assert len(batches) == 2
    assert batches[0][0].label == "chair"
    assert batches[0][0].mask.shape == (6, 8)
    assert batches[1][0].mask.shape == (10, 12)
    assert batches[0][0].score == pytest.approx(0.75)
    assert batches[0][0].feature is None


def test_prompt_free_mode_does_not_compute_or_set_text_prompts(monkeypatch):
    install_fake_yoloe(monkeypatch)
    provider = supplemental.YOLOEProposalProvider(
        checkpoint="yoloe-seg-pf.pt",
        device="cpu",
        mode="prompt-free",
    )
    provider.predict([np.zeros((6, 8, 3), dtype=np.uint8)])
    assert FakeYOLOE.instances[0].text_requests == []


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"mode": "text", "prompts": []}, "requires at least one"),
        (
            {"mode": "prompt_free", "prompts": ["chair"]},
            "must not define",
        ),
        ({"mode": "unknown"}, "either 'text' or 'prompt_free'"),
        ({"confidence": np.nan}, "confidence"),
        ({"iou": 1.1}, "iou"),
        ({"image_size": 0}, "image_size"),
        ({"max_detections": 0}, "max_detections"),
    ],
)
def test_yoloe_configuration_fails_fast(kwargs, message):
    defaults = {
        "checkpoint": "model.pt",
        "device": "cpu",
        "mode": "prompt_free",
    }
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=message):
        supplemental.YOLOEProposalProvider(**defaults)


def test_detector_requires_one_mask_per_box():
    class BadModel(FakeYOLOE):
        def predict(self, **kwargs):
            result = fake_result(0)
            result.masks = None
            return [result]

    provider = supplemental.YOLOEProposalProvider(
        checkpoint="detect-only.pt",
        device="cpu",
        model_factory=BadModel,
    )
    with pytest.raises(ValueError, match="boxes but no masks"):
        provider.predict([np.zeros((6, 8, 3), dtype=np.uint8)])


def test_npz_cache_round_trip_is_pickle_free_and_preserves_optional_fields(
    tmp_path,
):
    cache = supplemental.NpzProposalCache(tmp_path / "cache")
    values = [
        proposal(),
        proposal(offset=0.1, label=None, feature=False),
    ]
    assert cache.store("scene/frame", values, image_shape=(6, 8))
    path = cache.path_for_key("scene/frame")
    assert path.is_file()
    assert list(path.parent.glob("*.tmp")) == []

    with np.load(path, allow_pickle=False) as archive:
        assert archive["labels"].dtype.kind == "U"
        assert archive["masks"].dtype == np.bool_
    loaded = cache.load("scene/frame", expected_image_shape=(6, 8))
    assert len(loaded) == 2
    np.testing.assert_allclose(loaded[0].bbox, values[0].bbox)
    np.testing.assert_allclose(loaded[0].feature, values[0].feature)
    np.testing.assert_array_equal(loaded[0].mask, values[0].mask)
    assert loaded[0].label == "chair"
    assert loaded[1].label is None
    assert loaded[1].feature is None
    assert cache.load(
        "scene/frame", expected_image_shape=(8, 6)
    ) is None


def test_npz_cache_can_disable_writes_without_creating_a_directory(tmp_path):
    directory = tmp_path / "must-not-exist"
    cache = supplemental.NpzProposalCache(
        directory, write_enabled=False
    )
    assert not cache.store(
        "frame", [proposal()], image_shape=(6, 8)
    )
    assert not directory.exists()


def test_npz_cache_supports_empty_proposal_lists(tmp_path):
    cache = supplemental.NpzProposalCache(tmp_path)
    cache.store("empty", [], image_shape=(6, 8))
    assert cache.load("empty", expected_image_shape=(6, 8)) == []


class RecordingProvider:
    def __init__(self):
        self.calls = []

    def predict(self, images, *, frame_ids=None):
        self.calls.append((len(images), list(frame_ids or [])))
        return [
            [proposal(shape=image.shape[:2], offset=0.2)]
            for image in images
        ]


def test_cached_provider_batches_only_misses_and_preserves_order(tmp_path):
    backend = RecordingProvider()
    cache = supplemental.NpzProposalCache(tmp_path)
    wrapped = supplemental.CachedProposalProvider(
        backend, cache, namespace="config-a"
    )
    images = [
        np.zeros((6, 8, 3), dtype=np.uint8),
        np.zeros((6, 8, 3), dtype=np.uint8),
    ]
    cache.store(
        wrapped._key("hit", images[0]),
        [proposal(offset=0.5)],
        image_shape=(6, 8),
    )

    output = wrapped.predict(images, frame_ids=["hit", "miss"])
    assert backend.calls == [(1, ["miss"])]
    assert output[0][0].bbox[0] == pytest.approx(1.5)
    assert output[1][0].bbox[0] == pytest.approx(1.2)

    output_again = wrapped.predict(images, frame_ids=["hit", "miss"])
    assert backend.calls == [(1, ["miss"])]
    assert len(output_again) == 2


def test_cached_provider_does_not_reuse_same_id_for_different_rgb(tmp_path):
    backend = RecordingProvider()
    cache = supplemental.NpzProposalCache(tmp_path)
    wrapped = supplemental.CachedProposalProvider(
        backend, cache, namespace="config-a"
    )
    first = np.zeros((6, 8, 3), dtype=np.uint8)
    second = first.copy()
    second[0, 0, 0] = 1

    wrapped.predict([first], frame_ids=["same-logical-id"])
    wrapped.predict([second], frame_ids=["same-logical-id"])

    assert backend.calls == [
        (1, ["same-logical-id"]),
        (1, ["same-logical-id"]),
    ]


def test_build_provider_supports_nested_config_and_read_only_cache(tmp_path):
    provider = supplemental.build_provider(
        {
            "supplemental_proposals": {
                "enabled": True,
                "provider": "yoloe",
                "checkpoint": "model.pt",
                "mode": "text",
                "prompts": ["chair"],
                "cache": {
                    "enabled": True,
                    "directory": str(tmp_path),
                    "write": False,
                },
            }
        },
        "cuda:0",
    )
    assert isinstance(provider, supplemental.CachedProposalProvider)
    assert isinstance(provider.provider, supplemental.YOLOEProposalProvider)
    assert provider.cache.write_enabled is False
    assert provider.provider._model is None


def test_classes_is_a_supported_alias_for_text_prompts():
    resolved = supplemental.resolve_supplemental_proposal_config(
        {
            "enabled": True,
            "mode": "text",
            "classes": ["chair", "table"],
        }
    )
    assert resolved["prompts"] == ("chair", "table")


def test_cache_key_is_path_traversal_safe(tmp_path):
    cache = supplemental.NpzProposalCache(tmp_path)
    path = cache.path_for_key("../../outside")
    assert path.parent == tmp_path
    assert path.suffix == ".npz"
    assert ".." not in path.name


def test_public_cache_key_preserves_read_through_key_and_binds_rgb(tmp_path):
    backend = RecordingProvider()
    cache = supplemental.NpzProposalCache(tmp_path)
    wrapped = supplemental.CachedProposalProvider(
        backend, cache, namespace="teacher-v1"
    )
    image = np.zeros((6, 8, 3), dtype=np.uint8)

    key = supplemental.proposal_cache_key(
        "teacher-v1", "scene0000_00:0", image
    )
    assert key == wrapped._key("scene0000_00:0", image)

    changed = image.copy()
    changed[0, 0, 0] = 1
    assert supplemental.proposal_cache_key(
        "teacher-v1", "scene0000_00:0", changed
    ) != key


def test_strict_cache_replays_hits_without_writing_or_fallback(tmp_path):
    namespace = "sam3-scannet-train-v1"
    frame_id = "scene0000_00:0"
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    writable = supplemental.NpzProposalCache(tmp_path)
    writable.store(
        supplemental.proposal_cache_key(namespace, frame_id, image),
        [proposal()],
        image_shape=image.shape[:2],
    )
    files_before = set(tmp_path.iterdir())

    provider = supplemental.StrictCacheProposalProvider(
        writable,
        namespace=namespace,
        missing_policy="error",
    )
    output = provider.predict([image], frame_ids=[frame_id])

    assert len(output) == 1
    assert len(output[0]) == 1
    assert output[0][0].label == "chair"
    assert provider.hits == 1
    assert provider.misses == 0
    assert provider.cache.write_enabled is False
    assert set(tmp_path.iterdir()) == files_before


def test_strict_cache_error_policy_fails_closed_on_miss(tmp_path):
    provider = supplemental.StrictCacheProposalProvider(
        supplemental.NpzProposalCache(tmp_path),
        namespace="sam3-v1",
        missing_policy="error",
    )
    image = np.zeros((6, 8, 3), dtype=np.uint8)

    with pytest.raises(FileNotFoundError, match="scene0000_00:0"):
        provider.predict([image], frame_ids=["scene0000_00:0"])

    assert provider.hits == 0
    assert provider.misses == 1
    assert list(tmp_path.iterdir()) == []


def test_strict_cache_empty_policy_returns_empty_without_storing(tmp_path):
    provider = supplemental.StrictCacheProposalProvider(
        supplemental.NpzProposalCache(tmp_path),
        namespace="sam3-v1",
        missing_policy="empty",
    )
    images = [
        np.zeros((6, 8, 3), dtype=np.uint8),
        np.zeros((6, 8, 3), dtype=np.uint8),
    ]

    assert provider.predict(
        images,
        frame_ids=["scene0000_00:0", "scene0000_00:125"],
    ) == [[], []]
    assert provider.hits == 0
    assert provider.misses == 2
    assert list(tmp_path.iterdir()) == []


def test_strict_cache_requires_frame_ids_and_valid_configuration(tmp_path):
    cache = supplemental.NpzProposalCache(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        supplemental.StrictCacheProposalProvider(
            cache, namespace="", missing_policy="error"
        )
    with pytest.raises(ValueError, match="error.*empty"):
        supplemental.StrictCacheProposalProvider(
            cache, namespace="teacher", missing_policy="fallback"
        )

    provider = supplemental.StrictCacheProposalProvider(
        cache, namespace="teacher", missing_policy="empty"
    )
    with pytest.raises(ValueError, match="requires frame_ids"):
        provider.predict([np.zeros((6, 8, 3), dtype=np.uint8)])


def test_build_cache_only_provider_never_imports_or_constructs_models(
    tmp_path, monkeypatch
):
    imported = []
    original_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == "ultralytics" or name.startswith("sam3"):
            imported.append(name)
            raise AssertionError(f"cache-only provider imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    provider = supplemental.build_provider(
        {
            "enabled": True,
            "provider": "cache_only",
            "cache": {
                "enabled": True,
                "directory": str(tmp_path),
                "write": True,
                "namespace": "sam3-scannet-v1",
                "missing_policy": "empty",
            },
        },
        "cuda:0",
    )

    assert isinstance(provider, supplemental.StrictCacheProposalProvider)
    assert provider.cache.write_enabled is False
    assert provider.predict(
        [np.zeros((6, 8, 3), dtype=np.uint8)],
        frame_ids=["scene0000_00:0"],
    ) == [[]]
    assert imported == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "cache, message",
    [
        (
            {
                "enabled": False,
                "directory": "/unused",
                "namespace": "teacher",
            },
            "cache.enabled",
        ),
        (
            {
                "enabled": True,
                "directory": None,
                "namespace": "teacher",
            },
            "cache.directory",
        ),
        (
            {
                "enabled": True,
                "directory": "/unused",
                "namespace": None,
            },
            "cache.namespace",
        ),
    ],
)
def test_build_cache_only_provider_requires_complete_cache_config(
    cache, message
):
    with pytest.raises(ValueError, match=message):
        supplemental.build_provider(
            {
                "enabled": True,
                "provider": "cache_only",
                "cache": cache,
            },
            "cpu",
        )


@pytest.mark.parametrize(
    "cache, exception, message",
    [
        ({"namespace": 3}, TypeError, "namespace"),
        ({"namespace": " "}, ValueError, "namespace"),
        (
            {"missing_policy": "fallback"},
            ValueError,
            "missing_policy",
        ),
    ],
)
def test_cache_config_rejects_invalid_strict_replay_fields(
    cache, exception, message
):
    with pytest.raises(exception, match=message):
        supplemental.resolve_supplemental_proposal_config(
            {"cache": cache}
        )


def test_explicit_namespace_is_supported_by_legacy_yoloe_cache(tmp_path):
    provider = supplemental.build_provider(
        {
            "enabled": True,
            "provider": "yoloe",
            "checkpoint": "model.pt",
            "cache": {
                "enabled": True,
                "directory": str(tmp_path),
                "write": False,
                "namespace": "explicit-yoloe-v1",
                "missing_policy": "error",
            },
        },
        "cpu",
    )

    assert isinstance(provider, supplemental.CachedProposalProvider)
    assert provider.namespace == "explicit-yoloe-v1"
    assert isinstance(provider.provider, supplemental.YOLOEProposalProvider)
