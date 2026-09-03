from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from boxfusion.s3r_h10_provider_core import (
    ExactScheduleBundle,
    FrameTransaction,
    SceneSchedule,
    ScheduledFrame,
)
from tools import run_scannet_s3r_h10_fresh_boxer_provider as fresh


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_matrix(path: Path, value: np.ndarray) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = []
    for row in value:
        stream.append(" ".join(str(float(item)) for item in row))
    payload = ("\n".join(stream) + "\n").encode("ascii")
    path.write_bytes(payload)
    return payload


def _tiny_bundle(tmp_path: Path) -> tuple[ExactScheduleBundle, Path]:
    scene_id = "scene0001_00"
    scene_root = tmp_path / "scenes"
    scene_dir = scene_root / scene_id
    intrinsic = np.eye(4, dtype=np.float64)
    intrinsic[0, 0] = 500.0
    intrinsic[1, 1] = 501.0
    intrinsic[0, 2] = 320.0
    intrinsic[1, 2] = 240.0
    intrinsic_payload = _write_matrix(
        scene_dir / "frames/intrinsic/intrinsic_color.txt", intrinsic
    )
    frames = []
    for frame_id, translation in ((0, [10.0, 20.0, 30.0]), (25, [11, 20, 30])):
        color = f"color-{frame_id}".encode("ascii")
        depth = f"depth-{frame_id}".encode("ascii")
        color_path = scene_dir / f"frames/color/{frame_id}.jpg"
        depth_path = scene_dir / f"frames/depth/{frame_id}.png"
        color_path.parent.mkdir(parents=True, exist_ok=True)
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        color_path.write_bytes(color)
        depth_path.write_bytes(depth)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = translation
        pose_path = scene_dir / f"frames/pose/{frame_id}.txt"
        pose_payload = _write_matrix(pose_path, pose)
        frames.append(
            ScheduledFrame(
                frame_id=frame_id,
                color_relpath=f"frames/color/{frame_id}.jpg",
                color_sha256=_digest(color),
                depth_relpath=f"frames/depth/{frame_id}.png",
                depth_sha256=_digest(depth),
                pose_relpath=f"frames/pose/{frame_id}.txt",
                pose_sha256=_digest(pose_payload),
            )
        )
    scene = SceneSchedule(
        scene_id=scene_id,
        source_schedule_manifest_relpath="source/manifest.json",
        source_schedule_manifest_sha256="1" * 64,
        formal_t05_relpath="formal/scene0001_00_boxes.pkl",
        formal_t05_sha256="2" * 64,
        intrinsic_color_relpath="frames/intrinsic/intrinsic_color.txt",
        intrinsic_color_sha256=_digest(intrinsic_payload),
        raw_frame_ids=(0, 25),
        valid_frame_ids=(0, 25),
        excluded_frames=(),
        frames=tuple(frames),
    )
    bundle = ExactScheduleBundle(
        schema="boxfusion.s3r_h10_exact_schedule.v1",
        scene_order=(scene_id,),
        raw_frame_count=2,
        valid_frame_count=2,
        holdout_list_sha256="3" * 64,
        scenes=(scene,),
        sha256="4" * 64,
    )
    return bundle, scene_root


def test_frozen_v2_schedule_is_exact_and_core_accepts_it() -> None:
    assert fresh._sha256_path(fresh.SCHEDULE_PATH) == fresh.EXPECTED_SCHEDULE_SHA256
    bundle = fresh.parse_exact_schedule_bundle(fresh.SCHEDULE_PATH)
    assert bundle.sha256 == fresh.EXPECTED_SCHEDULE_SHA256
    assert bundle.valid_frame_count == 769
    assert bundle.raw_frame_count == 770


def test_reader_is_exact_order_synchronous_and_intrinsic_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle, scene_root = _tiny_bundle(tmp_path)
    events: list[str] = []
    original_read = fresh._read_exact_bytes

    def recording_read(path, expected_sha256, *, max_bytes, label):
        events.append(f"open:{label}")
        return original_read(path, expected_sha256, max_bytes=max_bytes, label=label)

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("directory enumeration is forbidden")

    monkeypatch.setattr(fresh, "_read_exact_bytes", recording_read)
    monkeypatch.setattr(os, "listdir", forbidden)
    monkeypatch.setattr(Path, "iterdir", forbidden)
    monkeypatch.setattr(Path, "glob", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)

    offsets: list[np.ndarray] = []

    def builder(**kwargs):
        offsets.append(kwargs["world_offset_absolute"].copy())
        events.append(f"build:{kwargs['frame_id']}")
        return {"frame_id": kwargs["frame_id"]}

    reader = fresh.ManifestScanNetFrameReader(bundle, scene_root, datum_builder=builder)
    scene = bundle.scenes[0]
    with pytest.raises(fresh.FreshProviderError, match="exact next"):
        reader.read(scene, scene.frames[1])
    assert events == []

    first = reader.read(scene, scene.frames[0])
    second = reader.read(scene, scene.frames[1])
    assert first.frame_id == 0 and second.frame_id == 25
    assert reader.completed_frame_count == 2
    np.testing.assert_array_equal(offsets[0], [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(offsets[1], [10.0, 20.0, 30.0])
    assert events == [
        "open:scene0001_00 intrinsic",
        "open:scene0001_00/0 color",
        "open:scene0001_00/0 depth",
        "open:scene0001_00/0 pose",
        "build:0",
        "open:scene0001_00/25 color",
        "open:scene0001_00/25 depth",
        "open:scene0001_00/25 pose",
        "build:25",
    ]


def test_reader_rejects_current_bundle_hash_mismatch_before_build(
    tmp_path: Path,
) -> None:
    bundle, scene_root = _tiny_bundle(tmp_path)
    scene = bundle.scenes[0]
    (scene_root / scene.scene_id / scene.frames[0].color_relpath).write_bytes(
        b"modified"
    )
    called = False

    def builder(**kwargs):
        nonlocal called
        del kwargs
        called = True
        return {}

    reader = fresh.ManifestScanNetFrameReader(bundle, scene_root, datum_builder=builder)
    with pytest.raises(fresh.FreshProviderError, match="SHA-256 mismatch"):
        reader.read(scene, scene.frames[0])
    assert called is False
    assert reader.completed_frame_count == 0


def test_post_stream_rehash_covers_every_exact_input_and_rejects_change(
    tmp_path: Path,
) -> None:
    bundle, scene_root = _tiny_bundle(tmp_path)
    result = fresh._rehash_exact_frame_inputs(bundle, scene_root)
    assert result["verified_file_count"] == 7
    assert result["expected_file_count"] == 7
    assert len(result["exact_input_ledger_sha256"]) == 64

    scene = bundle.scenes[0]
    changed = scene_root / scene.scene_id / scene.frames[1].depth_relpath
    changed.write_bytes(b"post-stream mutation")
    with pytest.raises(fresh.FreshProviderError, match="SHA-256 mismatch"):
        fresh._rehash_exact_frame_inputs(bundle, scene_root)


class _FakeReader:
    def __init__(self, bundle: ExactScheduleBundle, events: list[str]):
        self.bundle = bundle
        self.events = events
        self.completed_frame_count = 0

    def read(self, scene: SceneSchedule, frame: ScheduledFrame) -> fresh.FrameDatum:
        self.events.append(f"read:{frame.frame_id}")
        self.completed_frame_count += 1
        offset = np.asarray([10.0, 20.0, 30.0], dtype=np.float64)
        return fresh.FrameDatum(scene.scene_id, frame.frame_id, {}, offset)


class _FakeProvider:
    image_hw = 960

    def __init__(self, events: list[str]):
        self.events = events

    def synchronize(self) -> None:
        self.events.append("synchronize")

    def reset_scene_seed(self, scene_id: str) -> None:
        self.events.append(f"seed:{scene_id}")

    def infer(self, frame: fresh.FrameDatum) -> fresh.RawBoxRows:
        self.events.append(f"infer:{frame.frame_id}")
        if frame.frame_id == 25:
            return fresh.RawBoxRows.empty()
        return fresh.RawBoxRows(
            center=np.asarray([[10.0, 20.0, 31.0]], dtype=np.float64),
            extent=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
            quaternion=np.asarray([[1.0001, 0.0, 0.0, 0.0]], dtype=np.float64),
            score=np.asarray([0.7], dtype=np.float64),
        )

    def provenance(self):
        return {
            "owl_instance_count": 1,
            "boxernet_instance_count": 1,
            "fake": True,
        }


class _RecordingTransaction:
    events: list[str] = []

    def __init__(self, output_root: Path, bundle: ExactScheduleBundle):
        self.inner = FrameTransaction(output_root, bundle)

    @property
    def completed_frame_count(self):
        return self.inner.completed_frame_count

    def begin(self, scene_id, frame_id):
        self.events.append(f"begin:{frame_id}")
        return self.inner.begin(scene_id, frame_id)

    def commit(self, token, **kwargs):
        result = self.inner.commit(token, **kwargs)
        self.events.append(f"commit:{result.frame_id}")
        return result

    def publish_run_provenance(self, payload):
        self.events.append("publish-provenance")
        return self.inner.publish_run_provenance(payload)

    def seal(self, **kwargs):
        self.events.append("seal")
        return self.inner.seal(**kwargs)

    def close(self):
        self.inner.close()


def test_stream_loads_models_once_commits_empty_and_seals_provenance(
    tmp_path: Path,
) -> None:
    bundle, _ = _tiny_bundle(tmp_path)
    output = tmp_path / "fresh-output"
    events: list[str] = []
    factory_calls = 0
    _RecordingTransaction.events = events

    def provider_factory():
        nonlocal factory_calls
        factory_calls += 1
        events.append("load-models")
        return _FakeProvider(events)

    t05 = {bundle.scene_order[0]: "a" * 64}
    assets = {
        "fixture": {
            "path": "/fixture",
            "sha256_before": "b" * 64,
            "expected_sha256": None,
        }
    }
    result = fresh._execute_stream(
        bundle=bundle,
        output_root=output,
        provider_factory=provider_factory,
        reader_factory=lambda provider: _FakeReader(bundle, events),
        environment={"fixture": "pinned"},
        asset_ledger=assets,
        t05_before=t05,
        t05_after_fn=lambda: t05,
        immutable_recheck_fn=lambda: {"fixture": "b" * 64},
        frame_input_recheck_fn=lambda: {
            "verified_file_count": 7,
            "expected_file_count": 7,
            "exact_input_ledger_sha256": "d" * 64,
        },
        transaction_factory=_RecordingTransaction,
    )
    assert factory_calls == 1
    assert events.index("commit:0") < events.index("begin:25")
    assert events.index("commit:25") < events.index("publish-provenance")
    assert events.index("publish-provenance") < events.index("seal")
    assert result["final_seal"]["completed_frame_count"] == 2
    assert result["final_seal"]["run_provenance_sha256"] == result["provenance_sha256"]
    assert result["provenance"]["output"] == {
        "committed_frame_count": 2,
        "raw_row_count": 1,
        "empty_frame_count": 1,
        "native_prediction_mutation": False,
        "tracked_csv_created": False,
    }
    assert result["provenance"]["runtime"]["deadline_uses"] == (
        "warm_frame_end_to_end_summary_after_global_first_committed_frame"
    )
    assert result["provenance"]["runtime"]["warm_frame_count"] == 1
    assert result["provenance"]["runtime"]["cold_first_frame"]["frame_id"] == 0
    assert result["provenance"]["runtime"]["process_peak_rss_bytes"] > 0
    assert result["provenance"]["runtime"]["integrated_realtime_qualified"] is False
    assert (output / "RUN_PROVENANCE.json").is_file()
    assert (output / "FINAL_SEAL.json").is_file()
    assert not (output / "boxer_3dbbs_tracked.csv").exists()

    with np.load(output / "frames/scene0001_00.000000.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["center"], [[10.0, 20.0, 31.0]])
        np.testing.assert_array_equal(data["quaternion"], [[1.0, 0.0, 0.0, 0.0]])
        np.testing.assert_array_equal(data["source_row"], [0])
    with np.load(output / "frames/scene0001_00.000025.npz", allow_pickle=False) as data:
        assert data["center"].shape == (0, 3)
        assert data["score"].shape == (0,)

    provenance_bytes = (output / "RUN_PROVENANCE.json").read_bytes()
    assert _digest(provenance_bytes) == result["provenance_sha256"]
    decoded = json.loads(provenance_bytes)
    assert decoded["formal_t05"]["deserialized"] is False
    assert decoded["provider_contract"]["temporal_state"] is False


def test_changed_native_hash_leaves_run_unsealed(tmp_path: Path) -> None:
    bundle, _ = _tiny_bundle(tmp_path)
    output = tmp_path / "changed-native"
    events: list[str] = []
    t05 = {bundle.scene_order[0]: "a" * 64}
    with pytest.raises(fresh.FreshProviderError, match="formal T05 files changed"):
        fresh._execute_stream(
            bundle=bundle,
            output_root=output,
            provider_factory=lambda: _FakeProvider(events),
            reader_factory=lambda provider: _FakeReader(bundle, events),
            environment={},
            asset_ledger={},
            t05_before=t05,
            t05_after_fn=lambda: {bundle.scene_order[0]: "c" * 64},
            immutable_recheck_fn=lambda: {},
            frame_input_recheck_fn=lambda: {},
        )
    assert not (output / "RUN_PROVENANCE.json").exists()
    assert not (output / "FINAL_SEAL.json").exists()


def test_provider_factory_failure_creates_no_output_namespace(tmp_path: Path) -> None:
    bundle, _ = _tiny_bundle(tmp_path)
    output = tmp_path / "model-failure"

    def fail():
        raise fresh.FreshProviderError("load failed")

    with pytest.raises(fresh.FreshProviderError, match="load failed"):
        fresh._execute_stream(
            bundle=bundle,
            output_root=output,
            provider_factory=fail,
            reader_factory=lambda provider: pytest.fail("reader must not load"),
            environment={},
            asset_ledger={},
            t05_before={},
            t05_after_fn=lambda: {},
            immutable_recheck_fn=lambda: {},
            frame_input_recheck_fn=lambda: {},
        )
    assert not output.exists()


def test_boxer_infer_filters_3d_means_scores_and_adds_world_offset() -> None:
    torch = pytest.importorskip("torch")

    class TorchShim:
        bfloat16 = torch.bfloat16

        @staticmethod
        def autocast(**kwargs):
            del kwargs
            return nullcontext()

    class FakePose:
        def __init__(self, centers, quaternions):
            self.t = torch.tensor(centers, dtype=torch.float32)
            self.q = torch.tensor(quaternions, dtype=torch.float32)

    class FakeObbs:
        def __init__(self, centers, extents, quaternions, scores):
            self._centers = np.asarray(centers, dtype=np.float32)
            self._extents = np.asarray(extents, dtype=np.float32)
            self._quaternions = np.asarray(quaternions, dtype=np.float32)
            self._scores = np.asarray(scores, dtype=np.float32)

        def __len__(self):
            return len(self._scores)

        @property
        def prob(self):
            return torch.tensor(self._scores[:, None])

        def __getitem__(self, mask):
            index = np.asarray(mask.numpy(), dtype=bool)
            return FakeObbs(
                self._centers[index],
                self._extents[index],
                self._quaternions[index],
                self._scores[index],
            )

        def clone(self):
            return self

        @property
        def T_world_object(self):
            return FakePose(self._centers, self._quaternions)

        @property
        def bb3_diagonal(self):
            return torch.tensor(self._extents)

    predicted = FakeObbs(
        centers=[[1, 2, 3], [4, 5, 6]],
        extents=[[1, 1, 1], [2, 3, 4]],
        quaternions=[[1, 0, 0, 0], [1, 0, 0, 0]],
        scores=[0.49, 0.6],
    )

    class FakeBatch:
        def cpu(self):
            return self

        def __getitem__(self, index):
            assert index == 0
            return predicted

    class FakeOwl:
        def forward(self, image, resize_to_HW):
            assert resize_to_HW == (960, 960)
            assert float(image.max()) == 255.0
            return (
                torch.zeros((2, 4)),
                torch.tensor([0.3, 0.8]),
                torch.tensor([0, 1]),
                None,
            )

    class FakeBoxer:
        def forward(self, datum):
            assert datum["bb2d"].shape == (2, 4)
            return {"obbs_pr_w": FakeBatch()}

    provider = object.__new__(fresh.FrozenBoxerProvider)
    provider.torch = TorchShim()
    provider.owl = FakeOwl()
    provider.boxernet = FakeBoxer()
    frame = fresh.FrameDatum(
        "scene0001_00",
        0,
        {"img0": torch.ones((1, 3, 2, 2))},
        np.asarray([10.0, 20.0, 30.0]),
    )
    rows = provider.infer(frame)
    np.testing.assert_array_equal(rows.center, [[14.0, 25.0, 36.0]])
    np.testing.assert_array_equal(rows.extent, [[2.0, 3.0, 4.0]])
    np.testing.assert_array_equal(rows.quaternion, [[1.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(rows.score, [0.7], rtol=0.0, atol=1e-7)


def test_empty_owl_frame_never_calls_boxer() -> None:
    torch = pytest.importorskip("torch")

    class FakeOwl:
        def forward(self, image, resize_to_HW):
            del image, resize_to_HW
            return torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0), None

    provider = object.__new__(fresh.FrozenBoxerProvider)
    provider.torch = SimpleNamespace()
    provider.owl = FakeOwl()
    provider.boxernet = SimpleNamespace(
        forward=lambda datum: pytest.fail("Boxer must not run on an empty OWL frame")
    )
    frame = fresh.FrameDatum(
        "scene0001_00",
        0,
        {"img0": torch.ones((1, 3, 2, 2))},
        np.zeros(3),
    )
    rows = provider.infer(frame)
    assert rows.center.shape == (0, 3)
    assert rows.score.shape == (0,)


def test_environment_and_output_scope_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name, value in fresh.REQUIRED_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    assert fresh._validate_environment() == dict(fresh.REQUIRED_ENVIRONMENT)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(fresh.FreshProviderError, match="pinned before Python"):
        fresh._validate_environment()
    with pytest.raises(fresh.FreshProviderError, match="shadow log root"):
        fresh._validated_shadow_output(tmp_path / "outside")


def test_provider_contract_is_required_hash_bound_and_not_parsed(
    tmp_path: Path,
) -> None:
    contract = tmp_path / "contract.bin"
    payload = b"not even JSON; contract bytes are opaque\n"
    contract.write_bytes(payload)
    expected = _digest(payload)
    assert fresh._validate_provider_contract(contract, expected) == contract.absolute()
    with pytest.raises(fresh.FreshProviderError, match="SHA-256 mismatch"):
        fresh._validate_provider_contract(contract, "0" * 64)
    with pytest.raises(fresh.FreshProviderError, match="64 lowercase"):
        fresh._validate_provider_contract(contract, "A" * 64)
    with pytest.raises(fresh.FreshProviderError, match="missing provider contract"):
        fresh._validate_provider_contract(tmp_path / "missing.md", expected)


def test_source_has_no_dataset_loader_prefetch_or_frame_enumeration_surface() -> None:
    source = Path(fresh.__file__).read_text(encoding="utf-8")
    forbidden = (
        "from loaders.scannet_loader import",
        "from loaders.base_loader import",
        "run_boxer.main(",
        "os.listdir(",
        ".iterdir(",
        ".glob(",
        ".rglob(",
        "threading.Thread(",
        "boxer_3dbbs_tracked.csv",
    )
    assert all(token not in source for token in forbidden)
