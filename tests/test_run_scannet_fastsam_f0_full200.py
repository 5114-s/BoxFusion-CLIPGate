from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import cv2
import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_scannet_fastsam_f0_full200 as runner


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class _Timing:
    device: str = "cpu"
    total_seconds: float = 0.004
    max_memory_allocated_bytes: int = 0
    cuda_synchronized: bool = False


@dataclass(frozen=True)
class _Checkpoint:
    path: str
    byte_count: int
    sha256: str


class _FakeProvider:
    def __init__(self, checkpoint: Path, calls: list[np.ndarray]) -> None:
        self.checkpoint = _Checkpoint(
            path=str(checkpoint.resolve()),
            byte_count=checkpoint.stat().st_size,
            sha256=_hash(checkpoint),
        )
        self.device = "cpu"
        self.calls = calls

    def predict(self, bgr: np.ndarray) -> object:
        self.calls.append(bgr.copy())
        mask = np.zeros((480, 640), dtype=bool)
        mask[100:140, 100:140] = True
        return SimpleNamespace(
            masks=np.stack([mask]),
            confidences=np.asarray([0.9], dtype=np.float32),
            boxes_xyxy=np.asarray([[100, 100, 139, 139]], dtype=np.float32),
            timing=_Timing(),
        )


def _write_pose(path: Path, valid: bool) -> None:
    pose = np.eye(4, dtype=np.float64)
    if not valid:
        pose[0, 0] = np.nan
    np.savetxt(path, pose)


def _write_scene(
    scene_root: Path,
    scene: str,
    frames: tuple[int, ...] = (0, 25),
    pose_valid: tuple[bool, ...] = (True, True),
) -> None:
    root = scene_root / scene
    for name in ("color", "depth", "pose", "intrinsic"):
        (root / name).mkdir(parents=True, exist_ok=True)
    intrinsic = np.eye(4, dtype=np.float64)
    intrinsic[0, 0] = intrinsic[1, 1] = 500.0
    intrinsic[0, 2] = 320.0
    intrinsic[1, 2] = 240.0
    np.savetxt(root / "intrinsic/intrinsic_depth.txt", intrinsic)
    validity = dict(zip(frames, pose_valid))
    # CuTR's producer authenticates a scheduled frame with the nearest finite
    # raw pose, not merely the previous scheduled keyframe.  Materialize the
    # intervening pose table so synthetic receipts mirror that data surface.
    for frame_id in range(max(frames) + 1):
        _write_pose(root / "pose" / f"{frame_id}.txt", validity.get(frame_id, True))
    for frame_id in frames:
        # Different B/G/R bytes let the fake assert that the runner did not
        # silently convert the provider's required BGR input to RGB.
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        bgr[:, :, 0] = min(250, 11 + frame_id)
        bgr[:, :, 1] = 22
        bgr[:, :, 2] = 233
        assert cv2.imwrite(str(root / "color" / f"{frame_id}.jpg"), bgr)
        depth = np.full((480, 640), 2_000, dtype=np.uint16)
        assert cv2.imwrite(str(root / "depth" / f"{frame_id}.png"), depth)


def _cache_payload(
    boxes: torch.Tensor,
    *,
    schema: str,
    payload_count: int,
    input_signature: dict[str, str],
) -> dict:
    protected = hashlib.sha256(b"protected-pred-boxes").hexdigest()
    geometry = hashlib.sha256(b"geometry").hexdigest()
    return {
        "schema": schema,
        "image_size": (480, 640),
        "field_names": ["pred_boxes"],
        "fields": {"pred_boxes": boxes},
        "field_metadata": {
            "pred_boxes": {
                "dtype": str(boxes.dtype),
                "shape": list(boxes.shape),
                "sha256": runner._tensor_sha256(boxes),
            }
        },
        "count": payload_count,
        "attempt_id": "primary",
        "input_signature": input_signature,
        "protected_hashes": {"pred_boxes": protected},
        "geometry_sha256": geometry,
    }


def _write_schedule(
    root: Path,
    scene: str,
    frames: tuple[int, ...] = (0, 25),
    *,
    scene_root: Path | None = None,
    schema: str = runner.EXPECTED_CACHE_SCHEMA,
    payload_count: int = 0,
    namespace: str = runner.EXPECTED_CACHE_NAMESPACE,
    producer_fingerprint: str | None = None,
) -> None:
    scene_dir = root / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for frame_id in frames:
        if scene_root is None:
            input_signature = {
                key: hashlib.sha256(f"{scene}/{frame_id}/{key}".encode()).hexdigest()
                for key in (
                    "camera_to_world",
                    "depth",
                    "depth_K",
                    "image",
                    "image_K",
                )
            }
        else:
            bgr, depth_mm, _paths = runner._decode_frame(
                scene_root, scene, frame_id
            )
            _intrinsic_path, intrinsic = runner._load_intrinsic(scene_root, scene)
            input_signature, _metadata = runner._reconstruct_cutr_input_signature(
                bgr=bgr,
                depth_mm=depth_mm,
                intrinsic=intrinsic,
                scene_root=scene_root,
                scene=scene,
                frame_id=frame_id,
            )
        boxes = torch.empty((0, 4), dtype=torch.float32)
        payload = _cache_payload(
            boxes,
            schema=schema,
            payload_count=payload_count,
            input_signature=input_signature,
        )
        path = scene_dir / f"frame_{frame_id:06d}.pt"
        torch.save(payload, path)
        records.append(
            {
                "frame_id": frame_id,
                "count": 0,
                "sha256": _hash(path),
                "attempt_id": payload["attempt_id"],
                "input_signature": payload["input_signature"],
                "protected_hashes": payload["protected_hashes"],
                "geometry_sha256": payload["geometry_sha256"],
            }
        )
    (scene_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": runner.EXPECTED_CACHE_SCHEMA,
                "scene_id": scene,
                "namespace": namespace,
                "producer_fingerprint": (
                    producer_fingerprint
                    or runner.OLD100_PRODUCER_FINGERPRINT
                ),
                "schedule": {
                    "dataset_length": max(frames) + 50,
                    "gap": 25,
                    "terminal_policy": "upstream_boxfusion_early_exit_v1",
                },
                "recorded_frame_ids": list(frames),
                "records": records,
                "record_count": len(records),
                "proposal_count": 0,
            }
        ),
        encoding="utf-8",
    )


def _fixture(
    tmp_path: Path,
    *,
    scenes: tuple[str, ...] = ("scene0000_00", "scene0001_00"),
) -> dict:
    scene_list = tmp_path / "full200.txt"
    scene_list.write_text("\n".join(scenes) + "\n", encoding="utf-8")
    scene_root = tmp_path / "scans"
    roots = (tmp_path / "old", tmp_path / "delta")
    for index, scene in enumerate(scenes):
        _write_scene(scene_root, scene)
        _write_schedule(roots[index % 2], scene, scene_root=scene_root)
    checkpoint = tmp_path / "FastSAM.pt"
    checkpoint.write_bytes(b"frozen-test-checkpoint")
    return {
        "scenes": scenes,
        "scene_list": scene_list,
        "scene_root": scene_root,
        "roots": roots,
        "checkpoint": checkpoint,
        "output": tmp_path / "output",
    }


def _factory(checkpoint: Path, calls: list[np.ndarray]):
    def create(_sealed_path: Path, _device: str) -> _FakeProvider:
        return _FakeProvider(checkpoint, calls)

    return create


def _run(data: dict, calls: list[np.ndarray], **overrides):
    values = {
        "schedule_roots": data["roots"],
        "scene_root": data["scene_root"],
        "scene_list_path": data["scene_list"],
        "output_root": data["output"],
        "device": "cpu",
        "provider_factory": _factory(data["checkpoint"], calls),
        "_expected_scene_count": len(data["scenes"]),
    }
    values.update(overrides)
    return runner.run_shadow(**values)


def _all_keys(value: object) -> list[str]:
    if isinstance(value, dict):
        return list(value) + [key for item in value.values() for key in _all_keys(item)]
    if isinstance(value, list):
        return [key for item in value for key in _all_keys(item)]
    return []


def test_full_shadow_is_causal_bgr_only_and_writes_complete_geometry_sidecar(
    tmp_path: Path,
) -> None:
    data = _fixture(tmp_path, scenes=("scene0000_00",))
    calls: list[np.ndarray] = []
    manifest = _run(data, calls)

    assert manifest["contracts"]["shadow_only"]
    assert manifest["contracts"]["no_output_affecting"]
    assert not manifest["contracts"]["birth_enabled"]
    assert manifest["totals"]["accepted_lifts"] == 2
    assert len(calls) == 2
    assert tuple(calls[0][0, 0]) == (11, 22, 233)

    scene_row = manifest["scenes"][0]
    sidecar = json.loads(Path(scene_row["sidecar_path"]).read_text())
    assert _hash(Path(scene_row["sidecar_path"])) == scene_row["sidecar_sha256"]
    assert sidecar["complete"]
    assert [row["frame_id"] for row in sidecar["frames"]] == [0, 25]
    assert all(frame["inputs"]["current_pose_valid"] for frame in sidecar["frames"])
    assert all(
        not frame["inputs"]["f0_pose_forward_filled"] for frame in sidecar["frames"]
    )
    assert [
        row["runtime"]["provider_call_index_in_shard"] for row in sidecar["frames"]
    ] == [0, 1]
    assert all(row["runtime"]["warmup_excluded"] for row in sidecar["frames"])
    for frame in sidecar["frames"]:
        runtime = frame["runtime"]
        assert runtime["complete_ms"] >= runtime["provider_ms"] + runtime["core_ms"]
        assert runtime["receipt_total_ms"] >= runtime["complete_ms"]
    candidate = sidecar["frames"][0]["funnel"]["candidates"][0]
    assert len(candidate["points_and_voxel_keys_sha256"]) == 64
    assert len(candidate["world_center"]) == len(candidate["world_extent"]) == 3
    keys = _all_keys(sidecar)
    assert "points_world" not in keys
    assert "voxel_keys" not in keys


def test_invalid_current_pose_abstains_without_past_forward_fill(tmp_path: Path) -> None:
    data = _fixture(tmp_path, scenes=("scene0000_00",))
    _write_scene(
        data["scene_root"],
        "scene0000_00",
        pose_valid=(True, False),
    )
    _write_schedule(
        data["roots"][0],
        "scene0000_00",
        scene_root=data["scene_root"],
    )
    calls: list[np.ndarray] = []
    manifest = _run(data, calls)
    sidecar = json.loads(Path(manifest["scenes"][0]["sidecar_path"]).read_text())
    assert len(calls) == 1
    assert sidecar["frames"][0]["successful"]
    invalid = sidecar["frames"][1]
    assert not invalid["successful"]
    assert invalid["abstention"] == "invalid_current_pose"
    assert not invalid["inputs"]["current_pose_valid"]
    assert invalid["inputs"]["f0_pose_source_frame_id"] is None
    assert not invalid["inputs"]["f0_pose_forward_filled"]
    assert invalid["runtime"]["complete_ms"] == 0.0
    assert invalid["runtime"]["receipt_total_ms"] > 0.0
    assert invalid["runtime"]["provider_call_index_in_shard"] is None


def test_composite_roots_require_exactly_one_manifest_per_scene(tmp_path: Path) -> None:
    data = _fixture(tmp_path, scenes=("scene0000_00",))
    _write_schedule(data["roots"][1], "scene0000_00")
    with pytest.raises(runner.F0RunnerError, match="exactly one"):
        _run(data, [], plan_only=True)


def test_delta_cache_has_its_own_exact_frozen_namespace(tmp_path: Path) -> None:
    scene = "scene0568_01"
    data = _fixture(tmp_path, scenes=(scene,))
    first_manifest = data["roots"][0] / scene / "manifest.json"
    first_manifest.unlink()
    _write_schedule(
        data["roots"][1],
        scene,
        namespace=runner.EXPECTED_DELTA_CACHE_NAMESPACE,
        producer_fingerprint=(
            "1589802fd762b69015f6fd06f8ad88826888874540be7b8ad4b27b2d566cd316"
        ),
    )
    plan = _run(data, [], plan_only=True)
    assert plan["shard_keyframe_count"] == 2


def test_plan_only_needs_no_rgbd_checkpoint_provider_or_output(tmp_path: Path) -> None:
    scene = "scene0000_00"
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n", encoding="utf-8")
    root = tmp_path / "cache"
    _write_schedule(root, scene)

    def forbidden_factory(*_args):
        raise AssertionError("plan-only must not instantiate FastSAM")

    plan = runner.run_shadow(
        schedule_roots=(root,),
        scene_root=tmp_path / "missing-scans",
        scene_list_path=scene_list,
        output_root=tmp_path / "not-created",
        device="cuda:0",
        plan_only=True,
        provider_factory=forbidden_factory,
        _expected_scene_count=1,
    )
    assert plan["mode"] == "plan_only"
    assert plan["shard_keyframe_count"] == 2
    assert not (tmp_path / "not-created").exists()


def test_two_shards_are_disjoint_and_resume_is_create_only(tmp_path: Path) -> None:
    data = _fixture(tmp_path)
    first_calls: list[np.ndarray] = []
    second_calls: list[np.ndarray] = []
    shard0 = _run(data, first_calls, shard_index=0, num_shards=2)
    shard1 = _run(data, second_calls, shard_index=1, num_shards=2)
    assert shard0["shard"]["scene_order"] == ["scene0000_00"]
    assert shard1["shard"]["scene_order"] == ["scene0001_00"]
    assert {row["scene_id"] for row in shard0["scenes"]}.isdisjoint(
        row["scene_id"] for row in shard1["scenes"]
    )

    resumed_calls: list[np.ndarray] = []
    provider_instantiations = 0

    def forbidden_factory(*_args):
        nonlocal provider_instantiations
        provider_instantiations += 1
        raise AssertionError("a completed shard must not instantiate FastSAM")

    resumed = _run(
        data,
        resumed_calls,
        shard_index=0,
        num_shards=2,
        resume=True,
        provider_factory=forbidden_factory,
    )
    assert resumed["run_signature_sha256"] == shard0["run_signature_sha256"]
    assert resumed_calls == []
    assert provider_instantiations == 0
    with pytest.raises(runner.F0RunnerError, match="overwrite"):
        _run(data, [], shard_index=0, num_shards=2)


def test_first_three_successful_provider_calls_are_warmup_per_shard_not_scene(
    tmp_path: Path,
) -> None:
    data = _fixture(tmp_path)
    manifest = _run(data, [], shard_index=0, num_shards=1)
    ledgers = [
        json.loads(Path(row["sidecar_path"]).read_text())["frames"]
        for row in manifest["scenes"]
    ]
    successful = [frame for ledger in ledgers for frame in ledger if frame["successful"]]
    assert [
        frame["runtime"]["provider_call_index_in_shard"] for frame in successful
    ] == [0, 1, 2, 3]
    assert [frame["runtime"]["warmup_excluded"] for frame in successful] == [
        True,
        True,
        True,
        False,
    ]
    assert manifest["totals"]["warmup_excluded_successful_frames"] == 3
    assert manifest["scenes"][1]["runtime"]["complete_ms"]["sample_count"] == 1


def test_partial_resume_restores_shard_global_provider_call_index(tmp_path: Path) -> None:
    data = _fixture(tmp_path)
    original = _run(data, [], shard_index=0, num_shards=1)
    # Simulate a crash after scene 0 was atomically published but before scene
    # 1 and the shard manifest survived.  These are isolated tmp fixtures.
    Path(original["scenes"][1]["sidecar_path"]).unlink()
    (
        data["output"] / "shards/shard-000-of-001.json"
    ).unlink()

    resumed_calls: list[np.ndarray] = []
    resumed = _run(data, resumed_calls, shard_index=0, num_shards=1, resume=True)
    assert resumed["scenes"][0]["resumed"] is True
    assert resumed["scenes"][1]["resumed"] is False
    assert len(resumed_calls) == 5
    assert resumed["resume_rewarm_calls"] == 3
    assert resumed["resume_rewarm"]["all_successful"]
    assert resumed["resume_rewarm"]["scene_id"] == "scene0001_00"
    assert resumed["resume_rewarm"]["frame_id"] == 0
    assert all(
        call["success"] for call in resumed["resume_rewarm"]["calls"]
    )
    # Three unrecorded physical warmup calls precede the two recorded calls,
    # all using the pending scene's first RGB for rewarm.
    assert [int(call[0, 0, 0]) for call in resumed_calls] == [11, 11, 11, 11, 36]
    second = json.loads(Path(resumed["scenes"][1]["sidecar_path"]).read_text())
    assert [
        row["runtime"]["provider_call_index_in_shard"] for row in second["frames"]
    ] == [2, 3]
    assert [row["runtime"]["warmup_excluded"] for row in second["frames"]] == [
        True,
        False,
    ]


def test_partial_resume_fails_closed_if_any_physical_rewarm_call_fails(
    tmp_path: Path,
) -> None:
    data = _fixture(tmp_path)
    original = _run(data, [], shard_index=0, num_shards=1)
    Path(original["scenes"][1]["sidecar_path"]).unlink()
    (data["output"] / "shards/shard-000-of-001.json").unlink()

    attempted: list[np.ndarray] = []

    class FailingProvider(_FakeProvider):
        def predict(self, bgr: np.ndarray) -> object:
            attempted.append(bgr.copy())
            raise RuntimeError("synthetic rewarm failure")

    def factory(_sealed_path: Path, _device: str) -> FailingProvider:
        return FailingProvider(data["checkpoint"], [])

    with pytest.raises(runner.F0RunnerError, match="resume rewarm provider call 0"):
        _run(
            data,
            [],
            shard_index=0,
            num_shards=1,
            resume=True,
            provider_factory=factory,
        )
    assert len(attempted) == 1
    assert not Path(original["scenes"][1]["sidecar_path"]).exists()
    assert not (data["output"] / "shards/shard-000-of-001.json").exists()


def test_core_topk_and_provider_max_det_saturation_are_distinct() -> None:
    counts: Counter[str] = Counter()
    base = {
        "pre_dedup_eligible_count": 16,
        "deduplicated_count": 0,
        "lifting_eligible_count": 16,
        "selected_count": 16,
    }
    # Exactly Top-16 with no rejected mask is not saturated.
    runner._accumulate_funnel_counts(
        counts,
        {**base, "input_mask_count": 16, "cap_rejected_count": 0},
    )
    assert counts["cap_saturated_frames"] == 0
    assert counts["provider_max_det_saturated_frames"] == 0

    # A diagnosed Top-K rejection and a raw provider max_det hit are tracked
    # independently, even though they happen in the same synthetic frame.
    runner._accumulate_funnel_counts(
        counts,
        {**base, "input_mask_count": 100, "cap_rejected_count": 2},
    )
    assert counts["cap_saturated_frames"] == 1
    assert counts["provider_max_det_saturated_frames"] == 1
    assert counts["cap_rejected_masks"] == 2


@pytest.mark.parametrize(
    ("schema", "payload_count", "message"),
    [
        ("wrong.schema", 0, "schema differs"),
        (runner.EXPECTED_CACHE_SCHEMA, 1, "count differs"),
    ],
)
def test_each_cutr_pt_schema_and_count_are_verified(
    tmp_path: Path, schema: str, payload_count: int, message: str
) -> None:
    data = _fixture(tmp_path, scenes=("scene0000_00",))
    _write_schedule(
        data["roots"][0],
        "scene0000_00",
        schema=schema,
        payload_count=payload_count,
    )
    with pytest.raises(runner.F0RunnerError, match=message):
        _run(data, [])


def test_each_cutr_pt_file_hash_is_verified(tmp_path: Path) -> None:
    data = _fixture(tmp_path, scenes=("scene0000_00",))
    path = data["roots"][0] / "scene0000_00/frame_000000.pt"
    with path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(runner.F0RunnerError, match="hash changed"):
        _run(data, [])


def test_each_cutr_input_signature_is_reconstructed_and_verified(
    tmp_path: Path,
) -> None:
    data = _fixture(tmp_path, scenes=("scene0000_00",))
    cache_path = data["roots"][0] / "scene0000_00/frame_000000.pt"
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    payload["input_signature"]["image"] = "f" * 64
    torch.save(payload, cache_path)
    manifest_path = data["roots"][0] / "scene0000_00/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["records"][0]["input_signature"] = payload["input_signature"]
    manifest["records"][0]["sha256"] = _hash(cache_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(runner.F0RunnerError, match="input signature differs"):
        _run(data, [])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "wrong.manifest.schema", "manifest schema differs"),
        ("scene_id", "scene9999_99", "manifest scene identity differs"),
        ("producer_fingerprint", "0" * 64, "producer fingerprint differs"),
    ],
)
def test_schedule_manifest_identity_is_fail_closed(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    data = _fixture(tmp_path, scenes=("scene0000_00",))
    path = data["roots"][0] / "scene0000_00/manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(runner.F0RunnerError, match=message):
        _run(data, [], plan_only=True)


def test_non_upright_full200_abstention_census_is_hard_locked() -> None:
    assert runner.EXPECTED_NON_UPRIGHT_KEYFRAMES == {
        ("scene0246_00", 1900): 1,
        ("scene0426_00", 2200): 3,
    }


def test_default_provider_production_rejects_cpu_before_instantiation(
    tmp_path: Path,
) -> None:
    data = _fixture(tmp_path, scenes=("scene0000_00",))
    with pytest.raises(runner.F0RunnerError, match="explicit cuda:N"):
        runner.run_shadow(
            schedule_roots=data["roots"],
            scene_root=data["scene_root"],
            scene_list_path=data["scene_list"],
            output_root=data["output"],
            device="cpu",
            _expected_scene_count=1,
        )


def test_real_full200_mode_fails_closed_on_keyframe_total(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenes = tuple(f"scene{index:04d}_00" for index in range(200))
    scene_list = tmp_path / "full200.txt"
    scene_list.write_text("\n".join(scenes) + "\n", encoding="utf-8")
    root = tmp_path / "cache"
    for scene in scenes:
        _write_schedule(root, scene, frames=(0,))
    monkeypatch.setattr(runner, "EXPECTED_SCENE_LIST_SHA256", _hash(scene_list))
    with pytest.raises(runner.F0RunnerError, match="keyframe count differs"):
        runner.run_shadow(
            schedule_roots=(root,),
            scene_root=tmp_path / "unused",
            scene_list_path=scene_list,
            output_root=tmp_path / "unused-output",
            device="cpu",
            plan_only=True,
        )


def test_public_cli_has_no_active_or_supervised_surface() -> None:
    parameters = set(inspect.signature(runner.run_shadow).parameters)
    assert not parameters & {
        "gt",
        "annotation",
        "evaluator",
        "native_predictions",
        "birth",
        "clip",
        "semantic",
        "training_data",
    }
    options = {
        option
        for action in runner._parser()._actions
        for option in action.option_strings
    }
    assert not any(
        forbidden in option
        for option in options
        for forbidden in (
            "--gt",
            "annot",
            "eval",
            "native",
            "birth",
            "clip",
            "semantic",
            "train",
            "score",
        )
    )
