from __future__ import annotations

import contextlib
from copy import deepcopy
import hashlib
import importlib.machinery
import importlib.util
import inspect
import json
from multiprocessing import spawn as mp_spawn
import os
from pathlib import Path
import py_compile
import struct
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import tools.benchmark_scannet_s3r_h10_integrated_runtime as runtime


def _manifest() -> dict[str, object]:
    sha = "1" * 64
    frames = [
        {
            "frame_id": 0,
            "color_relpath": "frames/color/0.jpg",
            "color_sha256": sha,
            "depth_relpath": "frames/depth/0.png",
            "depth_sha256": sha,
            "pose_relpath": "frames/pose/0.txt",
            "pose_sha256": sha,
            "raw_pose_finite": True,
            "effective_pose_frame_id": 0,
            "effective_pose_relpath": "frames/pose/0.txt",
            "effective_pose_sha256": sha,
            "pose_resolution": "raw_finite",
            "intrinsic_color_relpath": "frames/intrinsic/intrinsic_color.txt",
            "intrinsic_color_sha256": sha,
            "provider_status": runtime.PROVIDER_MEMBER,
        },
        {
            "frame_id": 1,
            "color_relpath": "frames/color/1.jpg",
            "color_sha256": sha,
            "depth_relpath": "frames/depth/1.png",
            "depth_sha256": sha,
            "pose_relpath": "frames/pose/1.txt",
            "pose_sha256": sha,
            "raw_pose_finite": False,
            "effective_pose_frame_id": 0,
            "effective_pose_relpath": "frames/pose/0.txt",
            "effective_pose_sha256": sha,
            "pose_resolution": "past_most_recent_valid_native_pose",
            "intrinsic_color_relpath": "frames/intrinsic/intrinsic_color.txt",
            "intrinsic_color_sha256": sha,
            "provider_status": runtime.PROVIDER_ABSTAIN,
        },
        {
            "frame_id": 2,
            "color_relpath": "frames/color/2.jpg",
            "color_sha256": sha,
            "depth_relpath": "frames/depth/2.png",
            "depth_sha256": sha,
            "pose_relpath": "frames/pose/2.txt",
            "pose_sha256": sha,
            "raw_pose_finite": True,
            "effective_pose_frame_id": 2,
            "effective_pose_relpath": "frames/pose/2.txt",
            "effective_pose_sha256": sha,
            "pose_resolution": "raw_finite",
            "intrinsic_color_relpath": "frames/intrinsic/intrinsic_color.txt",
            "intrinsic_color_sha256": sha,
            "provider_status": runtime.OUTSIDE_PROVIDER,
        },
    ]
    return {
        "schema": runtime.EXPECTED_NATIVE_MANIFEST_SCHEMA,
        "scene_order": ["scene_test"],
        "scene_count": 1,
        "native_frame_count": 3,
        "provider_schedule": {
            "schema": "boxfusion.s3r_h10_exact_schedule.v2",
            "sha256": runtime.EXPECTED_PROVIDER_SCHEDULE_SHA256,
            "raw_frame_count": 4,
            "valid_frame_count": 1,
            "excluded_frame_count": 1,
        },
        "scenes": [
            {
                "scene_id": "scene_test",
                "native_frame_count": 3,
                "intrinsic_color_relpath": "frames/intrinsic/intrinsic_color.txt",
                "intrinsic_color_sha256": sha,
                "intrinsic_depth_relpath": "frames/intrinsic/intrinsic_depth.txt",
                "intrinsic_depth_sha256": sha,
                "role_mounts": {
                    "color": {"synthetic": True},
                    "depth": {"synthetic": True},
                    "pose": {"synthetic": True},
                    "intrinsic": {"synthetic": True},
                },
                "frame_ids": [0, 1, 2],
                "frames": frames,
            }
        ],
    }


@pytest.fixture(autouse=True)
def _single_visible_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    for name in tuple(os.environ):
        if name.startswith(("GIT_", "LD_")) and name not in runtime.REQUIRED_ENVIRONMENT:
            monkeypatch.delenv(name, raising=False)
    # Spawned test children must exercise the same interpreter-start guard as
    # formal workers.  Changing this environment value in the already-running
    # pytest parent intentionally does not mutate the parent's prefix.
    monkeypatch.setenv(
        "PYTHONPYCACHEPREFIX", runtime.FROZEN_PYCACHE_PREFIX
    )


def _run(tmp_path: Path, **kwargs):
    output = tmp_path / "integrated_timing.json"
    result = runtime._run_injected_harness(
        native_manifest=kwargs.pop("manifest", _manifest()),
        scene_root=tmp_path / "scenes",
        output=output,
        factories=runtime.SYNTHETIC_FACTORIES,
        native_factory_config=kwargs.pop(
            "native_config", {"gpu_uuid": "GPU-TEST-0"}
        ),
        provider_factory_config=kwargs.pop(
            "provider_config", {"gpu_uuid": "GPU-TEST-0"}
        ),
        cuda_visible_devices="7",
        ready_timeout_seconds=kwargs.pop("ready_timeout_seconds", 10.0),
        frame_timeout_seconds=kwargs.pop("frame_timeout_seconds", 10.0),
    )
    assert not kwargs
    return output, result


def _run_control(tmp_path: Path, **kwargs):
    output = tmp_path / "control_timing.json"
    result = runtime._run_injected_control(
        native_manifest=kwargs.pop("manifest", _manifest()),
        scene_root=tmp_path / "scenes",
        output=output,
        native_factory=runtime.SYNTHETIC_FACTORIES.native,
        native_factory_config=kwargs.pop(
            "native_config", {"gpu_uuid": "GPU-TEST-0"}
        ),
        cuda_visible_devices="7",
        ready_timeout_seconds=kwargs.pop("ready_timeout_seconds", 10.0),
        frame_timeout_seconds=kwargs.pop("frame_timeout_seconds", 10.0),
    )
    assert not kwargs
    return output, result


def test_two_spawn_workers_are_causal_bounded_and_timing_only(tmp_path):
    output, result = _run(tmp_path)

    assert output.is_file()
    assert json.loads(output.read_text("ascii")) == result
    assert result["spawn_start_method"] == "spawn"
    assert result["spawn_worker_count"] == 2
    assert result["workers"]["native"]["pid"] != result["workers"]["provider"]["pid"]
    assert result["same_cuda_visible_devices"] is True
    assert result["same_gpu_uuid"] is True
    assert result["gpu_uuid"] == "GPU-TEST-0"
    assert result["model_load_count_per_worker"] == 1
    assert result["queue_maxsize"] == 1
    assert result["queue_max_observed"] == 1
    assert result["backlog_events"] == 0
    assert result["native_frame_count"] == 3
    assert result["provider_call_count"] == 1
    assert result["provider_abstention_count"] == 1
    assert result["provider_outside_schedule_count"] == 1
    assert result["opaque_t05_identity_hashing"] is True
    assert result["coordinator_native_prediction_semantic_access"] is False
    assert result["native_prediction_deserialization"] is False
    assert result["native_prediction_geometry_access"] is False
    assert result["native_prediction_serialized"] is False
    assert result["native_prediction_mutation"] is False
    assert result["native_prediction_write"] is False
    assert result["coordinator_preflight_opaque_input_hashing"] is True
    assert result["online_worker_prefetch"] is False
    assert result["online_worker_future_frame_semantic_access"] is False
    assert result["gt_access"] is False
    assert result["evaluation"] is False
    assert result["ap_computation"] is False
    assert result["geometry_serialized"] is False
    assert result["labels_serialized"] is False
    assert result["birth"] is False
    assert result["provider"]["tracker_execution_device"] == "cpu"
    assert result["provider"]["tracker_gpu_execution"] is False
    assert result["provider"]["tracker_gpu_bytes"] == 0
    assert result["resources"][
        "torch_allocator_role_peak_upper_sum_allocated_bytes"
    ] == 2048
    assert result["resources"][
        "torch_allocator_role_peak_upper_sum_reserved_bytes"
    ] == 4096
    assert result["resources"]["device_wide_used_at_sync_max_bytes"] == 4096
    assert result["resources"][
        "device_wide_sampling_scope"
    ] == "cuda_synchronization_boundaries_only_not_continuous_peak"
    assert result["resources"][
        "device_wide_samples_include_non_torch_allocations"
    ] is True
    assert result["resources"][
        "device_wide_samples_cover_both_same_gpu_workers"
    ] is True
    assert result["resources"]["continuous_device_memory_peak_measured"] is False
    assert result["resources"]["numerical_vram_cap_preregistered"] is False
    assert result["resources"][
        "same_gpu_models_simultaneously_resident_and_full_stream_completed"
    ] is True
    assert result["resources"]["oom_failure_reported"] is False
    assert result["resources"][
        "full_stream_completed_without_oom_failure"
    ] is True

    ledger = result["causal_frame_ledger"]
    assert [row["frame_id"] for row in ledger] == [0, 1, 2]
    for previous, row in zip([None] + ledger[:-1], ledger):
        assert row["provider_request_ns"] <= row["provider_ack_ns"]
        assert row["provider_ack_ns"] <= row["native_request_ns"]
        assert row["native_request_ns"] <= row["native_ack_ns"]
        if previous is not None:
            assert previous["native_ack_ns"] <= row["provider_request_ns"]
    assert result["stream_clock"]["prestream_initialization_excluded"] is True
    assert result["stream_clock"][
        "component_constructor_warmup_disclosed_in_workers"
    ] is True
    assert result["stream_clock"]["full_pipeline_warmup"] is False
    assert result["stream_clock"]["first_real_forward_included"] is True
    assert all(
        worker["prestream_initialization_ns"] >= 0
        for worker in result["workers"].values()
    )
    assert result["workers"]["native"]["owl_constructor_dummy_warmup"] is False
    assert result["workers"]["provider"]["owl_constructor_dummy_warmup"] is True
    assert result["workers"]["provider"][
        "first_real_owl_kernels_pre_warmed"
    ] is True
    assert result["workers"]["provider"][
        "first_real_boxer_forward_included"
    ] is True


def test_control_arm_has_one_native_worker_and_no_provider_ack(tmp_path):
    output, result = _run_control(tmp_path)
    assert output.is_file()
    assert result["arm"] == "control"
    assert result["spawn_worker_count"] == 1
    assert result["provider_process_present"] is False
    assert result["provider_ack_count"] == 0
    assert result["provider_call_count"] == 0
    assert result["labels_serialized"] is False
    assert result["geometry_serialized"] is False
    assert result["native_frame_count"] == 3
    assert result["native_gap25_scheduled_keyframe_slot_count"] == 1
    assert set(result["workers"]) == {"native"}
    assert all("provider_request_ns" not in row for row in result["causal_frame_ledger"])


def test_entire_child_lifecycle_stdout_and_stderr_are_suppressed(tmp_path, capfd):
    _, result = _run(
        tmp_path,
        native_config={
            "gpu_uuid": "GPU-TEST-0",
            "emit_child_text": True,
            "emit_native_fd_bytes": True,
        },
        provider_config={
            "gpu_uuid": "GPU-TEST-0",
            "emit_child_text": True,
            "emit_native_fd_bytes": True,
        },
    )
    captured = capfd.readouterr()
    assert "suppressed-native" not in captured.out + captured.err
    assert "suppressed-provider" not in captured.out + captured.err
    assert "native-fd-secret-native" not in captured.out + captured.err
    assert "native-fd-secret-provider" not in captured.out + captured.err
    terminal = result["terminal_output"]
    assert terminal["model_lifecycle_fd1_fd2_redirected_to_devnull"] is True
    assert terminal["suppression_begins_at_spawn_target_before_factory"] is True
    assert terminal["spawn_bootstrap_stdio_suppression_not_claimed"] is True
    assert terminal["fd_redirection_scope"] == (
        "spawn_target_before_factory_through_worker_close"
    )
    assert terminal["stdio_content_retained"] is False
    assert terminal["stdio_character_counts_retained"] is False
    assert terminal["prediction_derived_text_reaches_coordinator_terminal"] is False
    for role in ("native", "provider"):
        assert result["workers"][role][
            "model_lifecycle_fd1_fd2_redirected_to_devnull"
        ] is True


@pytest.mark.parametrize(
    "role,config",
    [
        ("provider", {"gpu_uuid": "GPU-TEST-0", "fail_frame_id": 1}),
        ("native", {"gpu_uuid": "GPU-TEST-0", "fail_frame_id": 1}),
        ("provider", {"gpu_uuid": "GPU-TEST-0", "oom": True}),
        ("native", {"gpu_uuid": "GPU-TEST-0", "cap_violation": True}),
        (
            "provider",
            {"gpu_uuid": "GPU-TEST-0", "inject_forbidden_extra_key": True},
        ),
    ],
)
def test_worker_failure_oom_cap_or_nontiming_payload_publishes_nothing(
    tmp_path, role, config
):
    output = tmp_path / "integrated_timing.json"
    native_config = {"gpu_uuid": "GPU-TEST-0"}
    provider_config = {"gpu_uuid": "GPU-TEST-0"}
    if role == "native":
        native_config = config
    else:
        provider_config = config
    with pytest.raises(runtime.IntegratedRuntimeError):
        _run(
            tmp_path,
            native_config=native_config,
            provider_config=provider_config,
        )
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_worker_error_ipc_and_terminal_do_not_retain_exception_secret(tmp_path, capfd):
    secret = "prediction-derived-secret-obb-center-12345"
    with pytest.raises(runtime.IntegratedRuntimeError) as raised:
        _run(
            tmp_path,
            provider_config={
                "gpu_uuid": "GPU-TEST-0",
                "fail_frame_id": 1,
                "failure_secret": secret,
            },
        )
    captured = capfd.readouterr()
    assert secret not in str(raised.value)
    assert secret not in captured.out + captured.err
    assert not (tmp_path / "integrated_timing.json").exists()


def test_formal_cli_reports_only_content_free_failure(monkeypatch, capsys, tmp_path):
    secret = "tensor-or-path-secret-from-deep-runtime"

    def fail_closed(**_kwargs):
        raise runtime.IntegratedRuntimeError(secret)

    monkeypatch.setattr(runtime, "run_formal_h10_runtime_arm", fail_closed)
    status = runtime.main(
        [
            "--arm",
            "control",
            "--output",
            os.fspath(tmp_path / "output.json"),
            "--runtime-contract",
            os.fspath(tmp_path / "contract.md"),
            "--expected-runtime-contract-sha256",
            "1" * 64,
            "--expected-runner-sha256",
            "2" * 64,
            "--expected-runner-test-sha256",
            "3" * 64,
        ]
    )
    captured = capsys.readouterr()
    assert status == 2
    assert captured.out == ""
    assert captured.err == "ERROR: integrated runtime failed closed\n"
    assert secret not in captured.err


def test_mismatched_gpu_uuid_fails_before_stream_and_publishes_nothing(tmp_path):
    with pytest.raises(runtime.IntegratedRuntimeError, match="GPU UUID"):
        _run(
            tmp_path,
            native_config={"gpu_uuid": "GPU-A"},
            provider_config={"gpu_uuid": "GPU-B"},
        )
    assert not (tmp_path / "integrated_timing.json").exists()


def test_performance_gate_failure_is_published_as_measurement(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "NATIVE_MIN_FPS", 1e12)
    output, result = _run(tmp_path)
    assert output.exists()
    assert result["native"]["minimum_fps_met"] is False
    assert result["performance_gates"]["all_met"] is False


def test_timeout_terminates_workers_and_publishes_nothing(tmp_path):
    with pytest.raises(runtime.IntegratedRuntimeError, match="timed out"):
        _run(
            tmp_path,
            provider_config={
                "gpu_uuid": "GPU-TEST-0",
                "frame_delay_seconds": 0.25,
            },
            frame_timeout_seconds=0.02,
        )
    assert not (tmp_path / "integrated_timing.json").exists()


def test_jpg_only_and_past_only_manifest_guards_fail_before_spawn(tmp_path):
    png = _manifest()
    png["scenes"][0]["frames"][0]["color_relpath"] = "frames/color/0.png"
    with pytest.raises(runtime.IntegratedRuntimeError, match="JPG-only"):
        _run(tmp_path, manifest=png)
    assert not (tmp_path / "integrated_timing.json").exists()

    future = _manifest()
    future_frame = future["scenes"][0]["frames"][1]
    future_frame["effective_pose_frame_id"] = 2
    future_frame["effective_pose_relpath"] = "frames/pose/2.txt"
    with pytest.raises(runtime.IntegratedRuntimeError, match="past-most-recent"):
        _run(tmp_path, manifest=future)
    assert not (tmp_path / "integrated_timing.json").exists()


def test_create_only_output_refuses_overwrite(tmp_path):
    output, first = _run(tmp_path)
    before = output.read_bytes()
    with pytest.raises(runtime.IntegratedRuntimeError, match="publication"):
        _run(tmp_path)
    assert output.read_bytes() == before
    assert json.loads(before.decode("ascii")) == first


def test_minimal_view_never_copies_a_future_frame(tmp_path):
    view = runtime._minimal_manifest_view(deepcopy(_manifest()))
    first = runtime._frame_command_payload(view["scenes"][0]["frames"][0])
    assert first["frame_id"] == 0
    assert "frames" not in first
    assert "next_frame" not in first
    assert first["effective_pose_frame_id"] <= first["frame_id"]


@pytest.mark.parametrize("stage", ["ready", "frame", "end", "stop"])
def test_bool_cannot_smuggle_worker_protocol_integers(tmp_path, stage):
    with pytest.raises(runtime.IntegratedRuntimeError):
        _run(
            tmp_path,
            provider_config={
                "gpu_uuid": "GPU-TEST-0",
                "bool_smuggle_stage": stage,
            },
        )
    assert not (tmp_path / "integrated_timing.json").exists()


@pytest.mark.parametrize("stage", ["constructor", "ready", "frame", "end-scene", "close"])
def test_worker_lifecycle_rejects_post_factory_sys_path_drift(tmp_path, stage):
    with pytest.raises(runtime.IntegratedRuntimeError):
        _run(
            tmp_path,
            provider_config={
                "gpu_uuid": "GPU-TEST-0",
                "mutate_sys_path_stage": stage,
            },
        )
    assert not (tmp_path / "integrated_timing.json").exists()


@pytest.mark.parametrize(
    "mutation_key",
    ("mutate_pycache_prefix_stage", "mutate_pycache_environment_stage"),
)
@pytest.mark.parametrize("stage", ["constructor", "ready", "frame", "end-scene", "close"])
def test_worker_lifecycle_rejects_pycache_prefix_drift(
    tmp_path, mutation_key, stage
):
    with pytest.raises(runtime.IntegratedRuntimeError):
        _run(
            tmp_path,
            provider_config={
                "gpu_uuid": "GPU-TEST-0",
                mutation_key: stage,
            },
        )
    assert not (tmp_path / "integrated_timing.json").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("scene_count", True),
        lambda value: value["scenes"][0]["frames"][1].__setitem__("frame_id", True),
        lambda value: value["provider_schedule"].__setitem__(
            "valid_frame_count", True
        ),
        lambda value: value["provider_schedule"].__setitem__(
            "excluded_frame_count", True
        ),
    ],
)
def test_bool_cannot_smuggle_manifest_integers(tmp_path, mutation):
    manifest = deepcopy(_manifest())
    mutation(manifest)
    with pytest.raises(runtime.IntegratedRuntimeError, match="integer|identity"):
        _run(tmp_path, manifest=manifest)
    assert not (tmp_path / "integrated_timing.json").exists()


class _NoOpEngine:
    def __init__(self):
        self.gpu_uuid = "GPU-NOOP"
        self.gpu_device_name = "no-op-gpu"
        self.gpu_total_memory_bytes = 1024
        self.gpu_driver_version = "no-op-driver"
        self.torch = SimpleNamespace(
            __version__="synthetic",
            __file__="/synthetic/torch/__init__.py",
            version=SimpleNamespace(cuda="synthetic"),
        )
        self.memory_calls = 0
        self.sync_calls = 0
        self.infer_calls = 0
        self.reset_calls = 0

    def memory(self):
        self.memory_calls += 1
        return (11, 22, 33)

    def synchronize(self):
        self.sync_calls += 1

    def reset_scene(self, _scene_id):
        self.reset_calls += 1

    def infer(self, *_args, **_kwargs):
        self.infer_calls += 1
        raise AssertionError("no-op frame invoked provider inference")


class _NoOpReader:
    intrinsic_color_sha256 = "1" * 64
    intrinsic_depth_sha256 = "1" * 64

    def __init__(self):
        self.skip_ids = []
        self.read_calls = 0

    def skip_current(self, frame):
        self.skip_ids.append(frame["frame_id"])

    def read_current(self, _frame):
        self.read_calls += 1
        raise AssertionError("no-op frame read input bytes")


class _NoOpTracker:
    def __init__(self):
        self.snapshot_calls = 0
        self.query_calls = 0
        self.commit_calls = 0

    def snapshot(self):
        self.snapshot_calls += 1
        raise AssertionError("no-op frame touched tracker")

    def query(self, *_args, **_kwargs):
        self.query_calls += 1
        raise AssertionError("no-op frame queried tracker")

    def commit(self, *_args, **_kwargs):
        self.commit_calls += 1
        raise AssertionError("no-op frame committed tracker")


def test_provider_outside_and_abstain_are_strict_zero_work():
    engine = _NoOpEngine()
    reader = _NoOpReader()
    tracker = _NoOpTracker()
    worker = runtime.ProviderS3RRuntimeWorker(
        {},
        engine=engine,
        reader_factory=lambda *_args, **_kwargs: reader,
        tracker_factory=lambda: tracker,
    )
    worker.ready()
    worker.start_scene(
        {
            "scene_id": "scene_test",
            "intrinsic_color_sha256": "1" * 64,
            "intrinsic_depth_sha256": "1" * 64,
        }
    )
    for frame_id, status in ((1, runtime.PROVIDER_ABSTAIN), (2, runtime.OUTSIDE_PROVIDER)):
        ack = worker.process_frame(
            {"scene_id": "scene_test", "frame_id": frame_id, "provider_status": status}
        )
        assert ack["input_read"] is False
        assert ack["call_executed"] is False
        assert ack["raw_row_count"] == 0
        assert ack["k8_row_count"] == 0
        assert ack["tracker_ns"] == 0
        assert ack["tracker_audit_complete"] is True
        assert ack["device_wide_memory_sampled_at_sync"] is False
        assert ack["device_wide_used_at_sync_bytes"] is None
    assert reader.skip_ids == [1, 2]
    assert reader.read_calls == 0
    assert engine.infer_calls == 0
    assert engine.sync_calls == 0
    assert engine.memory_calls == 1  # READY only; no no-op CUDA-stat query.
    assert tracker.snapshot_calls == 0
    assert tracker.query_calls == 0
    assert tracker.commit_calls == 0


def test_cuda_memory_separates_torch_allocator_from_device_wide_sample():
    fake_cuda = SimpleNamespace(
        max_memory_allocated=lambda: 101,
        max_memory_reserved=lambda: 202,
        mem_get_info=lambda: (600, 1000),
    )
    assert runtime._cuda_memory(SimpleNamespace(cuda=fake_cuda)) == (101, 202, 400)


@pytest.mark.parametrize("status", [runtime.PROVIDER_ABSTAIN, runtime.OUTSIDE_PROVIDER])
def test_provider_noop_cannot_claim_device_wide_memory_sample(status):
    worker = runtime._SyntheticRuntimeWorker("provider", {})
    worker.start_scene(
        {
            "scene_id": "scene_test",
            "intrinsic_color_sha256": "1" * 64,
            "intrinsic_depth_sha256": "1" * 64,
        }
    )
    frame = runtime._minimal_manifest_view(_manifest())["scenes"][0]["frames"][
        1 if status == runtime.PROVIDER_ABSTAIN else 2
    ]
    ack = dict(worker.process_frame(frame))
    ack["device_wide_used_at_sync_bytes"] = 123
    ack["device_wide_memory_sampled_at_sync"] = True
    with pytest.raises(runtime.IntegratedRuntimeError, match="sampling contract"):
        runtime._validate_frame_payload(ack, role="provider", frame=frame)


def test_provider_member_must_report_device_wide_memory_sample():
    worker = runtime._SyntheticRuntimeWorker("provider", {})
    worker.start_scene(
        {
            "scene_id": "scene_test",
            "intrinsic_color_sha256": "1" * 64,
            "intrinsic_depth_sha256": "1" * 64,
        }
    )
    frame = runtime._minimal_manifest_view(_manifest())["scenes"][0]["frames"][0]
    ack = dict(worker.process_frame(frame))
    ack["device_wide_used_at_sync_bytes"] = None
    ack["device_wide_memory_sampled_at_sync"] = False
    with pytest.raises(runtime.IntegratedRuntimeError, match="sampling contract"):
        runtime._validate_frame_payload(ack, role="provider", frame=frame)


def test_native_pst_is_forced_to_exact_absolute_hashed_asset(tmp_path):
    cfg = {"box_fusion": {"pst_path": "./data/pst_1024_0.tiff"}}
    runtime._force_native_runtime_paths(cfg, runtime.NATIVE_PST)
    assert cfg["box_fusion"]["pst_path"] == os.fspath(
        runtime.NATIVE_PST.resolve(strict=True)
    )
    wrong = tmp_path / "pst.tiff"
    wrong.write_bytes(b"not-the-frozen-pst")
    with pytest.raises(runtime.IntegratedRuntimeError, match="PST path differs"):
        runtime._force_native_runtime_paths(
            {"box_fusion": {"pst_path": "relative"}}, wrong
        )


def test_native_primary_context_is_scoped_to_demo_thread_and_popped_on_error():
    events = []

    class ContextApi:
        current = None

        @classmethod
        def get_current(cls):
            return cls.current

        @classmethod
        def pop(cls):
            events.append("pop")
            cls.current = None

    class PrimaryContext:
        def push(self):
            events.append("push")
            ContextApi.current = self

    engine = runtime._FrozenNativeT05Engine.__new__(
        runtime._FrozenNativeT05Engine
    )
    engine._pycuda_driver = SimpleNamespace(Context=ContextApi)
    engine._pycuda_primary_context = PrimaryContext()

    with pytest.raises(RuntimeError, match="injected demo failure"):
        with engine.thread_context():
            assert ContextApi.get_current() is engine._pycuda_primary_context
            raise RuntimeError("injected demo failure")

    assert events == ["push", "pop"]
    assert ContextApi.get_current() is None


def test_native_primary_context_is_popped_after_normal_demo_body():
    events = []

    class ContextApi:
        current = None

        @classmethod
        def get_current(cls):
            return cls.current

        @classmethod
        def pop(cls):
            events.append("pop")
            cls.current = None

    class PrimaryContext:
        def push(self):
            events.append("push")
            ContextApi.current = self

    engine = runtime._FrozenNativeT05Engine.__new__(
        runtime._FrozenNativeT05Engine
    )
    engine._pycuda_driver = SimpleNamespace(Context=ContextApi)
    engine._pycuda_primary_context = PrimaryContext()

    with engine.thread_context():
        assert ContextApi.get_current() is engine._pycuda_primary_context

    assert events == ["push", "pop"]
    assert ContextApi.get_current() is None


def test_native_primary_context_missing_after_push_fails_and_still_pops():
    events = []

    class ContextApi:
        @staticmethod
        def get_current():
            return None

        @staticmethod
        def pop():
            events.append("pop")

    class PrimaryContext:
        def push(self):
            events.append("push")

    engine = runtime._FrozenNativeT05Engine.__new__(
        runtime._FrozenNativeT05Engine
    )
    engine._pycuda_driver = SimpleNamespace(Context=ContextApi)
    engine._pycuda_primary_context = PrimaryContext()

    with pytest.raises(runtime.IntegratedRuntimeError, match="context is absent"):
        with engine.thread_context():
            raise AssertionError("unreachable body")

    assert events == ["push", "pop"]


def test_native_pre_yield_failure_wakes_first_submit_without_outer_timeout():
    dataset = runtime._CausalNativeDataset(
        reader=None,
        engine=None,
        frame_count=1,
        sample_builder=lambda value: value,
    )
    dataset.fail_pending(RuntimeError("injected pre-yield failure"))

    completion = dataset.submit({"scene_id": "scene_test", "frame_id": 0})

    assert completion["ok"] is False
    assert completion["error_code"] == "worker_failure"


def test_native_submit_uses_one_end_to_end_local_deadline(monkeypatch):
    observed = {}

    class InputQueue:
        @staticmethod
        def put(_value, *, timeout):
            observed["put_timeout"] = timeout

    class CompletionQueue:
        @staticmethod
        def get(*, timeout):
            observed["get_timeout"] = timeout
            return {"ok": False, "error_code": "worker_failure"}

    ticks = iter((10.0, 80.0))
    monkeypatch.setattr(runtime.time, "monotonic", lambda: next(ticks))
    dataset = runtime._CausalNativeDataset(
        reader=None,
        engine=None,
        frame_count=1,
        sample_builder=lambda value: value,
    )
    dataset.input_queue = InputQueue()
    dataset.completion_queue = CompletionQueue()

    completion = dataset.submit({"scene_id": "scene_test", "frame_id": 0})

    assert completion["ok"] is False
    assert observed["put_timeout"] == runtime.NATIVE_LOCAL_COMPLETION_TIMEOUT_SECONDS
    assert observed["get_timeout"] == pytest.approx(40.0)


def test_native_thread_context_wraps_pre_yield_failure_and_publishes_completion():
    events = []

    class Engine:
        @contextlib.contextmanager
        def thread_context(self):
            events.append("enter")
            try:
                yield
            finally:
                events.append("exit")

        def run_scene(self, _dataset, **_kwargs):
            events.append("run")
            raise RuntimeError("injected pre-yield failure")

    dataset = runtime._CausalNativeDataset(
        reader=None,
        engine=Engine(),
        frame_count=1,
        sample_builder=lambda value: value,
    )
    worker = runtime.NativeT05RuntimeWorker({}, engine=dataset.engine)
    worker._dataset = dataset
    worker._scene = {
        "scene_id": "scene_test",
        "scene_directory": "/not-read",
    }

    worker._thread_main()
    completion = dataset.submit({"scene_id": "scene_test", "frame_id": 0})

    assert events == ["enter", "run", "exit"]
    assert len(worker._thread_errors) == 1
    assert completion["ok"] is False
    assert completion["error_code"] == "worker_failure"


def test_native_dependency_ledger_covers_reviewed_import_closure():
    required = {
        "native_boxfusion_package_init": "boxfusion/__init__.py",
        "native_tools_package_init": "tools/__init__.py",
        "native_vit": "boxfusion/vit.py",
        "native_batching": "boxfusion/batching.py",
        "native_imagelist": "boxfusion/imagelist.py",
        "native_pos": "boxfusion/pos.py",
        "native_transforms": "boxfusion/transforms.py",
        "native_color": "boxfusion/color.py",
        "native_proposal_cache": "boxfusion/proposal_cache.py",
        "native_graw_fragments": "boxfusion/graw_fragments.py",
        "native_graw_shadow": "boxfusion/graw_shadow.py",
        "native_gclean_shadow": "boxfusion/gclean_shadow.py",
        "native_puf_gclean_shadow": "boxfusion/puf_gclean_shadow.py",
        "native_observer_track_adapter": "boxfusion/observer_track_adapter.py",
        "native_observer_track_registry": "boxfusion/observer_track_registry.py",
        "native_smov_fragments": "boxfusion/smov_fragments.py",
        "native_group3d_lite": "boxfusion/group3d_lite.py",
        "native_group3d_lite_oracle": "boxfusion/group3d_lite_oracle.py",
        "native_puf_lite": "boxfusion/puf_lite.py",
        "provider_external_owlv2_model": (
            "/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer/"
            "owl/owlv2_model.py"
        ),
        "provider_external_dinov3_wrapper": (
            "/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer/"
            "boxernet/dinov3_wrapper.py"
        ),
        "provider_external_demo_utils": (
            "/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer/"
            "utils/demo_utils.py"
        ),
        "provider_external_gravity": (
            "/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer/"
            "utils/gravity.py"
        ),
        "provider_external_tw_obb": (
            "/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer/"
            "utils/tw/obb.py"
        ),
        "provider_external_tw_tensor_utils": (
            "/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer/"
            "utils/tw/tensor_utils.py"
        ),
        "provider_external_tw_tensor_wrapper": (
            "/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer/"
            "utils/tw/tensor_wrapper.py"
        ),
        "provider_external_boxernet_init": (
            "/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer/"
            "boxernet/__init__.py"
        ),
        "provider_external_loaders_init": (
            "/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer/"
            "loaders/__init__.py"
        ),
        "provider_external_owl_init": (
            "/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer/"
            "owl/__init__.py"
        ),
        "provider_external_utils_init": (
            "/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer/"
            "utils/__init__.py"
        ),
        "provider_external_tw_init": (
            "/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer/"
            "utils/tw/__init__.py"
        ),
        "system_git": "/usr/bin/git",
        "system_nvidia_smi": "/usr/bin/nvidia-smi",
    }
    paths = runtime._native_static_asset_paths()
    assert set(required) <= set(paths)
    assert set(required) <= set(runtime.NATIVE_ASSET_EXPECTED_SHA256)
    for name, relative in required.items():
        expected_path = Path(relative)
        if not expected_path.is_absolute():
            expected_path = runtime.REPOSITORY_ROOT / expected_path
        assert paths[name].resolve() == expected_path.resolve()
        assert hashlib.sha256(paths[name].read_bytes()).hexdigest() == (
            runtime.NATIVE_ASSET_EXPECTED_SHA256[name]
        )


def test_external_command_binaries_are_absolute_and_hash_pinned():
    expected = {
        "system_git": runtime.GIT_EXECUTABLE,
        "system_nvidia_smi": runtime.NVIDIA_SMI_EXECUTABLE,
    }
    paths = runtime._native_static_asset_paths()
    for name, path in expected.items():
        assert path.is_absolute()
        assert paths[name] == path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == (
            runtime.NATIVE_ASSET_EXPECTED_SHA256[name]
        )


def test_asset_snapshot_binds_absent_import_shadow_candidates(tmp_path):
    source = tmp_path / "frozen_module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    snapshot = runtime._snapshot_asset_paths(
        {"source": source}, {"source": digest}
    )
    assert snapshot["import_shadow_candidates_absent"] is True
    assert snapshot["import_shadow_candidate_count"] >= 2
    assert len(snapshot["import_shadow_candidates_identity_sha256"]) == 64

    extension = source.with_name(
        f"{source.stem}{importlib.machinery.EXTENSION_SUFFIXES[0]}"
    )
    extension.write_bytes(b"shadow")
    with pytest.raises(runtime.IntegratedRuntimeError, match="import-shadow"):
        runtime._snapshot_asset_paths({"source": source}, {"source": digest})


def test_import_shadow_guard_rejects_legacy_adjacent_pyc(tmp_path):
    source = tmp_path / "frozen_module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    source.with_suffix(".pyc").write_bytes(b"legacy-shadow")
    with pytest.raises(RuntimeError, match="must be absent"):
        runtime._assert_import_shadow_candidates_absent((source,))


def test_pathfinder_prefers_same_stem_package_and_guard_rejects_it(tmp_path):
    source = tmp_path / "victim.py"
    source.write_text("VALUE = 'pinned-source'\n", encoding="utf-8")
    package = tmp_path / "victim"
    package.mkdir()
    (package / "__init__.py").write_text(
        "VALUE = 'shadow-package'\n", encoding="utf-8"
    )
    spec = importlib.machinery.PathFinder.find_spec("victim", [os.fspath(tmp_path)])
    assert spec is not None
    assert Path(str(spec.origin)).resolve() == (package / "__init__.py").resolve()
    with pytest.raises(RuntimeError, match="must be absent"):
        runtime._assert_import_shadow_candidates_absent((source,))


def test_import_shadow_guard_rejects_symlinked_parent_component(tmp_path):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    source = real_parent / "victim.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(RuntimeError, match="parent"):
        runtime._assert_import_shadow_candidates_absent((alias / "victim.py",))


def test_provider_checkout_snapshot_binds_clean_commit_tree_and_inode():
    snapshot = runtime._snapshot_provider_checkout(runtime.PROVIDER_BOXER_ROOT)
    assert snapshot["commit"] == runtime.EXPECTED_PROVIDER_BOXER_COMMIT
    assert snapshot["tree"] == runtime.EXPECTED_PROVIDER_BOXER_TREE
    assert snapshot["status_porcelain_v1_empty"] is True
    assert snapshot["ordinary_untracked_files_absent"] is True
    assert snapshot["ignored_files_allowlist_enforced"] is True
    assert snapshot["ignored_files"]["entry_count"] == 21
    assert snapshot["ignored_files"]["checkpoint_count"] == 2
    assert snapshot["ignored_files"]["pycache_count"] == 19
    dino = next(
        row
        for row in snapshot["ignored_files"]["entries"]
        if row["relative_path"] == runtime.PROVIDER_IGNORED_DINO_RELPATH
    )
    assert dino["type"] == "symlink_to_hash_bound_regular_file"
    assert dino["symlink_target"] == (
        runtime.PROVIDER_IGNORED_DINO_SYMLINK_TARGET
    )
    assert len(snapshot["identity_sha256"]) == 64


def _run_fixture_git(*arguments, **kwargs):
    return subprocess.run(
        [os.fspath(runtime.GIT_EXECUTABLE), *arguments],
        env=runtime._minimal_external_command_environment(git=True),
        **kwargs,
    )


def _make_ignored_checkout(tmp_path: Path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".gitignore").write_text(
        "ckpts/\n__pycache__/\n*.pyc\n*.so\nignored-*\n",
        encoding="utf-8",
    )
    (checkout / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
    _run_fixture_git("init", "-q", checkout, check=True)
    _run_fixture_git(
        "-C", checkout, "config", "user.email", "tests@example.invalid",
        check=True,
    )
    _run_fixture_git(
        "-C", checkout, "config", "user.name", "Runtime Tests",
        check=True,
    )
    _run_fixture_git(
        "-C", checkout, "add", ".gitignore", "tracked.py", check=True
    )
    _run_fixture_git(
        "-C", checkout, "commit", "-q", "-m", "fixture", check=True
    )
    checkpoint_dir = checkout / "ckpts"
    checkpoint_dir.mkdir()
    boxer = checkpoint_dir / Path(
        next(
            path
            for path in runtime.PROVIDER_IGNORED_CHECKPOINT_SHA256
            if path != runtime.PROVIDER_IGNORED_DINO_RELPATH
        )
    ).name
    boxer.write_bytes(b"small-boxer-checkpoint")
    dino_target = tmp_path / "frozen-dino-target.pth"
    dino_target.write_bytes(b"small-dino-checkpoint")
    dino = checkpoint_dir / Path(runtime.PROVIDER_IGNORED_DINO_RELPATH).name
    dino.symlink_to(dino_target)
    pycache = checkout / "module" / "__pycache__"
    pycache.mkdir(parents=True)
    (pycache / "allowed.cpython-310.pyc").write_bytes(b"allowed-pyc")
    expected_checkpoints = {
        f"ckpts/{boxer.name}": hashlib.sha256(boxer.read_bytes()).hexdigest(),
        runtime.PROVIDER_IGNORED_DINO_RELPATH: hashlib.sha256(
            dino_target.read_bytes()
        ).hexdigest(),
    }
    commit = _run_fixture_git(
        "-C", checkout, "rev-parse", "HEAD",
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    tree = _run_fixture_git(
        "-C", checkout, "rev-parse", "HEAD^{tree}",
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    return checkout, commit, tree, expected_checkpoints, dino_target


def _snapshot_test_checkout(tmp_path: Path):
    checkout, commit, tree, checkpoints, dino_target = _make_ignored_checkout(
        tmp_path
    )
    snapshot = runtime._snapshot_provider_checkout(
        checkout,
        expected_commit=commit,
        expected_tree=tree,
        expected_ignored_checkpoints=checkpoints,
        expected_ignored_dino_symlink_target=os.fspath(dino_target),
    )
    return checkout, commit, tree, checkpoints, dino_target, snapshot


@pytest.mark.parametrize("dirty_status,tree_changed", [(True, False), (False, True)])
def test_provider_checkout_snapshot_rejects_untracked_or_tree_tamper(
    monkeypatch, tmp_path, dirty_status, tree_changed
):
    checkout, commit, tree, checkpoints, dino_target = _make_ignored_checkout(
        tmp_path
    )
    if dirty_status:
        (checkout / "secret.py").write_text("SECRET = True\n", encoding="utf-8")
    with pytest.raises(runtime.IntegratedRuntimeError):
        runtime._snapshot_provider_checkout(
            checkout,
            expected_commit=commit,
            expected_tree="c" * 40 if tree_changed else tree,
            expected_ignored_checkpoints=checkpoints,
            expected_ignored_dino_symlink_target=os.fspath(dino_target),
        )


def test_provider_checkout_snapshot_rejects_ignored_same_name_extension(tmp_path):
    checkout, commit, tree, checkpoints, dino_target = _make_ignored_checkout(
        tmp_path
    )
    package = checkout / "module"
    extension = package / f"allowed{importlib.machinery.EXTENSION_SUFFIXES[0]}"
    extension.write_bytes(b"not-a-real-extension")
    with pytest.raises(runtime.IntegratedRuntimeError, match="extension module"):
        runtime._snapshot_provider_checkout(
            checkout,
            expected_commit=commit,
            expected_tree=tree,
            expected_ignored_checkpoints=checkpoints,
            expected_ignored_dino_symlink_target=os.fspath(dino_target),
        )


@pytest.mark.parametrize("extra_kind", ["regular", "symlink"])
def test_provider_checkout_snapshot_rejects_extra_ignored_entry(
    tmp_path, extra_kind
):
    checkout, commit, tree, checkpoints, dino_target = _make_ignored_checkout(
        tmp_path
    )
    extra = checkout / "ignored-extra"
    if extra_kind == "regular":
        extra.write_bytes(b"unexpected")
    else:
        extra.symlink_to(dino_target)
    with pytest.raises(runtime.IntegratedRuntimeError, match="non-allowlisted"):
        runtime._snapshot_provider_checkout(
            checkout,
            expected_commit=commit,
            expected_tree=tree,
            expected_ignored_checkpoints=checkpoints,
            expected_ignored_dino_symlink_target=os.fspath(dino_target),
        )


def test_provider_checkout_snapshot_binds_allowed_pyc_and_checkpoint_drift(tmp_path):
    checkout, commit, tree, checkpoints, dino_target, before = (
        _snapshot_test_checkout(tmp_path)
    )
    pyc = checkout / "module" / "__pycache__" / "allowed.cpython-310.pyc"
    pyc.write_bytes(b"changed-pyc")
    after = runtime._snapshot_provider_checkout(
        checkout,
        expected_commit=commit,
        expected_tree=tree,
        expected_ignored_checkpoints=checkpoints,
        expected_ignored_dino_symlink_target=os.fspath(dino_target),
    )
    assert after != before
    assert after["ignored_files"]["identity_sha256"] != (
        before["ignored_files"]["identity_sha256"]
    )

    boxer_relative = next(
        path
        for path in checkpoints
        if path != runtime.PROVIDER_IGNORED_DINO_RELPATH
    )
    (checkout / boxer_relative).write_bytes(b"changed-checkpoint")
    with pytest.raises(runtime.IntegratedRuntimeError, match="checkpoint hash"):
        runtime._snapshot_provider_checkout(
            checkout,
            expected_commit=commit,
            expected_tree=tree,
            expected_ignored_checkpoints=checkpoints,
            expected_ignored_dino_symlink_target=os.fspath(dino_target),
        )


def test_provider_checkout_snapshot_rejects_dino_symlink_or_target_drift(tmp_path):
    checkout, commit, tree, checkpoints, dino_target = _make_ignored_checkout(
        tmp_path
    )
    dino = checkout / runtime.PROVIDER_IGNORED_DINO_RELPATH
    alternate_target = tmp_path / "alternate-dino-target.pth"
    alternate_target.write_bytes(dino_target.read_bytes())
    dino.unlink()
    dino.symlink_to(alternate_target)
    with pytest.raises(runtime.IntegratedRuntimeError, match="symlink identity"):
        runtime._snapshot_provider_checkout(
            checkout,
            expected_commit=commit,
            expected_tree=tree,
            expected_ignored_checkpoints=checkpoints,
            expected_ignored_dino_symlink_target=os.fspath(dino_target),
        )


def test_ignored_regular_file_walk_rejects_symlink_parent(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    real_parent = tmp_path / "real-parent"
    cache = real_parent / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "probe.pyc").write_bytes(b"probe")
    (checkout / "alias").symlink_to(real_parent, target_is_directory=True)
    descriptor = os.open(checkout, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(runtime.IntegratedRuntimeError, match="parent"):
            runtime._stream_checkout_regular_file_identity(
                descriptor,
                "alias/__pycache__/probe.pyc",
                maximum=1024,
                label="ignored provider file",
            )
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    "payload",
    [
        b"absolute/path\0/escape\0",
        b"a\0a\0",
        b"a\0\0",
        b"./a\0",
        b"a/../b\0",
        b"\xff\0",
        b"not-terminated",
    ],
)
def test_git_nul_path_parser_rejects_noncanonical_input(payload):
    with pytest.raises(runtime.IntegratedRuntimeError):
        runtime._parse_git_nul_relative_paths(payload)


def test_git_audit_uses_absolute_binary_and_noninheriting_environment(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", "/host/git-object-injection")
    monkeypatch.setenv("GIT_DIR", "/host/git-dir-injection")
    monkeypatch.setenv("LD_AUDIT", "/host/loader-audit-injection.so")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/host/library-injection")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        return SimpleNamespace(stdout=b"frozen-output\n")

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        observed = runtime._git_checkout_bytes_at_descriptor(
            descriptor, "rev-parse", "HEAD"
        )
    finally:
        os.close(descriptor)
    assert observed == b"frozen-output\n"
    assert captured["command"][0] == "/usr/bin/git"
    environment = captured["environment"]
    assert environment["PATH"] == runtime.TRUSTED_EXECUTABLE_PATH
    assert environment["LD_LIBRARY_PATH"] == ""
    assert environment["LD_PRELOAD"] == ""
    assert environment["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert environment["GIT_CONFIG_SYSTEM"] == "/dev/null"
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert "GIT_OBJECT_DIRECTORY" not in environment
    assert "GIT_DIR" not in environment
    assert "LD_AUDIT" not in environment


def test_fresh_provider_git_smoke_with_all_git_names_absent(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("GIT_"):
            monkeypatch.delenv(name, raising=False)
    assert not any(name.startswith("GIT_") for name in os.environ)
    from tools import run_scannet_s3r_h10_fresh_boxer_provider as fresh

    assert fresh._git_text(fresh.BOXER_ROOT, "rev-parse", "HEAD") == (
        runtime.EXPECTED_PROVIDER_BOXER_COMMIT
    )


def test_nvidia_smi_uses_absolute_binary_and_noninheriting_environment(
    monkeypatch,
):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", "/host/git-injection")
    monkeypatch.setenv("LD_AUDIT", "/host/loader-audit-injection.so")
    monkeypatch.setenv("LD_PRELOAD", "/host/preload-injection.so")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs["env"]))
        stdout = "GPU-TEST-ABSOLUTE\n" if "--query-gpu=uuid" in command else "999.1\n"
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def get_device_properties(_index):
            return SimpleNamespace(uuid=None, name="test-gpu", total_memory=1024)

    identity = runtime._cuda_device_identity(SimpleNamespace(cuda=FakeCuda()))
    assert identity == ("GPU-TEST-ABSOLUTE", "test-gpu", 1024, "999.1")
    assert len(calls) == 2
    for command, environment in calls:
        assert command[0] == "/usr/bin/nvidia-smi"
        assert environment == runtime._minimal_external_command_environment()
        assert "GIT_CEILING_DIRECTORIES" not in environment
        assert "LD_AUDIT" not in environment


def test_python_probe_environment_does_not_inherit_git_or_loader_injections(
    monkeypatch,
):
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", "/host/git-object-injection")
    monkeypatch.setenv("LD_AUDIT", "/host/loader-audit-injection.so")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/host/library-injection")
    observed = runtime._minimal_python_probe_environment()
    assert observed["PATH"] == runtime.TRUSTED_EXECUTABLE_PATH
    assert observed["LD_LIBRARY_PATH"] == ""
    assert observed["LD_PRELOAD"] == ""
    assert not any(name.startswith("GIT_") for name in observed)
    assert "GIT_OBJECT_DIRECTORY" not in observed
    assert "LD_AUDIT" not in observed


def test_second_worker_spawn_failure_terminates_first(monkeypatch, tmp_path):
    spawned = []
    terminated = []
    first = SimpleNamespace(role="native", process=object(), command_queue=object(), response_queue=object())

    def fake_spawn(*_args, role, **_kwargs):
        if role == "native":
            spawned.append(first)
            return first
        raise RuntimeError("second spawn failed")

    monkeypatch.setattr(runtime.mp, "get_context", lambda _method: object())
    monkeypatch.setattr(runtime, "_spawn_runtime_endpoint", fake_spawn)
    monkeypatch.setattr(
        runtime,
        "_terminate_endpoints",
        lambda endpoints: terminated.extend(endpoints),
    )
    with pytest.raises(RuntimeError, match="second spawn failed"):
        runtime._execute_integrated_runtime(
            manifest_view=_manifest(),
            scene_root=tmp_path,
            factories=runtime.SYNTHETIC_FACTORIES,
            native_factory_config={},
            provider_factory_config={},
            cuda_visible_devices="7",
        )
    assert spawned == [first]
    assert terminated == [first]


def test_partial_current_worker_start_failure_cleans_process_and_both_queues(
    monkeypatch,
):
    events = []

    class FakeQueue:
        def __init__(self, name):
            self.name = name

        def close(self):
            events.append((self.name, "close"))

        def join_thread(self):
            events.append((self.name, "join_thread"))

    queues = [FakeQueue("command"), FakeQueue("response")]

    class PartialProcess:
        def __init__(self):
            self.alive = False

        def start(self):
            self.alive = True
            events.append(("process", "start"))
            raise RuntimeError("partial second worker start")

        def is_alive(self):
            return self.alive

        def terminate(self):
            events.append(("process", "terminate"))
            self.alive = False

        def join(self, timeout):
            assert timeout == runtime.WORKER_STOP_TIMEOUT_SECONDS
            events.append(("process", "join"))

        def kill(self):
            events.append(("process", "kill"))
            self.alive = False

    process = PartialProcess()

    class FakeContext:
        def Queue(self, *, maxsize):
            assert maxsize == runtime.QUEUE_MAXSIZE
            return queues.pop(0)

        def Process(self, **_kwargs):
            return process

    parent_sys_path = list(sys.path)
    role_sys_path = [os.fspath(runtime.REPOSITORY_ROOT), "/role-specific-only"]
    role_post_path = [
        os.fspath(runtime.PROVIDER_BOXER_ROOT.resolve(strict=True)),
        *role_sys_path,
    ]
    original_executable = os.fsdecode(mp_spawn.get_executable())
    monkeypatch.setattr(
        runtime.mp,
        "set_executable",
        lambda value: events.append(("exe", os.fspath(value))),
    )
    with pytest.raises(RuntimeError, match="partial second worker start"):
        runtime._spawn_runtime_endpoint(
            FakeContext(),
            role="provider",
            factory=runtime.SYNTHETIC_FACTORIES.provider,
            config={
                "python_executable": sys.executable,
                "python_sys_path": role_sys_path,
                "expected_post_factory_python_sys_path": role_post_path,
                "expected_runtime_identity": {
                    "python_pycache_prefix_environment": (
                        runtime.FROZEN_PYCACHE_PREFIX
                    ),
                    "python_pycache_prefix": runtime.FROZEN_PYCACHE_PREFIX,
                    "spawn_entry_python_sys_path_sha256": runtime._sys_path_sha256(
                        role_sys_path
                    ),
                    "post_factory_python_sys_path_sha256": runtime._sys_path_sha256(
                        role_post_path
                    ),
                },
            },
        )
    assert sys.path == parent_sys_path
    assert events[-1] == ("response", "join_thread")
    assert ("exe", original_executable) in events
    assert ("process", "terminate") in events
    assert ("process", "join") in events
    assert ("process", "kill") not in events
    assert ("command", "close") in events
    assert ("command", "join_thread") in events
    assert ("response", "close") in events
    assert ("response", "join_thread") in events


def test_concurrent_spawn_configuration_is_serialized_and_restored(monkeypatch):
    parent_sys_path = list(sys.path)
    original_executable = os.fsdecode(mp_spawn.get_executable())
    executable_state = {"value": original_executable}
    native_entered = threading.Event()
    provider_entered = threading.Event()
    release_native = threading.Event()
    observations = {}
    results = {}
    errors = []

    class FakeQueue:
        def close(self):
            pass

        def join_thread(self):
            pass

    class FakeProcess:
        def __init__(self, role):
            self.role = role

        def start(self):
            observations[self.role] = (
                tuple(sys.path),
                executable_state["value"],
            )
            if self.role == "native":
                native_entered.set()
                assert release_native.wait(2.0)
            else:
                provider_entered.set()

        def is_alive(self):
            return False

    class FakeContext:
        def Queue(self, *, maxsize):
            assert maxsize == runtime.QUEUE_MAXSIZE
            return FakeQueue()

        def Process(self, *, name, **_kwargs):
            return FakeProcess(name.rsplit("-", 1)[-1])

    monkeypatch.setattr(
        runtime.mp,
        "set_executable",
        lambda value: executable_state.__setitem__("value", os.fspath(value)),
    )
    monkeypatch.setattr(
        runtime.mp_spawn,
        "get_executable",
        lambda: os.fsencode(executable_state["value"]),
    )

    def config(role):
        if role == "native":
            executable = runtime.NATIVE_PYTHON_EXECUTABLE
            entry = runtime.NATIVE_FROZEN_SYS_PATH
            post = runtime.NATIVE_POST_FACTORY_SYS_PATH
        else:
            executable = runtime.PROVIDER_PYTHON_EXECUTABLE
            entry = runtime.PROVIDER_FROZEN_SYS_PATH
            post = runtime.PROVIDER_POST_FACTORY_SYS_PATH
        return {
            "python_executable": os.fspath(executable),
            "python_sys_path": list(entry),
            "expected_post_factory_python_sys_path": list(post),
            "expected_runtime_identity": {
                "python_pycache_prefix_environment": (
                    runtime.FROZEN_PYCACHE_PREFIX
                ),
                "python_pycache_prefix": runtime.FROZEN_PYCACHE_PREFIX,
                "spawn_entry_python_sys_path_sha256": runtime._sys_path_sha256(
                    entry
                ),
                "post_factory_python_sys_path_sha256": runtime._sys_path_sha256(
                    post
                ),
            },
        }

    def launch(role):
        try:
            results[role] = runtime._spawn_runtime_endpoint(
                FakeContext(),
                role=role,
                factory=(
                    runtime.SYNTHETIC_FACTORIES.native
                    if role == "native"
                    else runtime.SYNTHETIC_FACTORIES.provider
                ),
                config=config(role),
            )
        except BaseException as error:
            errors.append(error)

    native_thread = threading.Thread(target=launch, args=("native",))
    provider_thread = threading.Thread(target=launch, args=("provider",))
    native_thread.start()
    assert native_entered.wait(2.0)
    provider_thread.start()
    assert not provider_entered.wait(0.05)
    release_native.set()
    native_thread.join(2.0)
    provider_thread.join(2.0)
    assert not native_thread.is_alive() and not provider_thread.is_alive()
    assert errors == []
    assert set(results) == {"native", "provider"}
    assert observations["native"] == (
        tuple(runtime.NATIVE_FROZEN_SYS_PATH),
        os.fspath(runtime.NATIVE_PYTHON_EXECUTABLE.resolve(strict=True)),
    )
    assert observations["provider"] == (
        tuple(runtime.PROVIDER_FROZEN_SYS_PATH),
        os.fspath(runtime.PROVIDER_PYTHON_EXECUTABLE.resolve(strict=True)),
    )
    assert sys.path == parent_sys_path
    assert executable_state["value"] == original_executable


def _install_trusted_native_cubify_import_side_effects(
    monkeypatch, tmp_path: Path
) -> SimpleNamespace:
    """Model the two exact path appends made by frozen Cubify dependencies."""

    remote_template_dir = tmp_path / "tmp-test-remote-template"
    remote_template_dir.mkdir()
    remote_template_dir.chmod(0o700)
    remote_template_payload = b"# frozen test remote-module template\n"
    remote_template_file = (
        remote_template_dir / runtime.NATIVE_TORCH_REMOTE_TEMPLATE_BASENAME
    )
    remote_template_file.write_bytes(remote_template_payload)
    remote_template_file.chmod(0o600)
    monkeypatch.setattr(
        runtime, "NATIVE_TORCH_REMOTE_TEMPLATE_PARENT", tmp_path
    )
    monkeypatch.setattr(
        runtime,
        "NATIVE_TORCH_REMOTE_TEMPLATE_SIZE",
        len(remote_template_payload),
    )
    monkeypatch.setattr(
        runtime,
        "NATIVE_TORCH_REMOTE_TEMPLATE_SHA256",
        hashlib.sha256(remote_template_payload).hexdigest(),
    )
    setuptools_module = SimpleNamespace(
        __file__=os.fspath(runtime.NATIVE_SETUPTOOLS_INIT),
        __spec__=SimpleNamespace(
            origin=os.fspath(runtime.NATIVE_SETUPTOOLS_INIT)
        ),
    )
    instantiator_module = SimpleNamespace(
        __file__=os.fspath(runtime.NATIVE_TORCH_REMOTE_INSTANTIATOR),
        __spec__=SimpleNamespace(
            origin=os.fspath(runtime.NATIVE_TORCH_REMOTE_INSTANTIATOR)
        ),
        INSTANTIATED_TEMPLATE_DIR_PATH=os.fspath(remote_template_dir),
    )
    generated_module = SimpleNamespace(
        __file__=os.fspath(remote_template_file),
        __spec__=SimpleNamespace(
            name="_remote_module_non_scriptable",
            origin=os.fspath(remote_template_file),
        ),
    )
    monkeypatch.setitem(sys.modules, "setuptools", setuptools_module)
    monkeypatch.setitem(
        sys.modules,
        "torch.distributed.nn.jit.instantiator",
        instantiator_module,
    )
    monkeypatch.setitem(
        sys.modules, "_remote_module_non_scriptable", generated_module
    )
    observed = [
        *runtime.NATIVE_FROZEN_SYS_PATH,
        os.fspath(runtime.NATIVE_SETUPTOOLS_VENDOR),
        os.fspath(remote_template_dir),
    ]
    monkeypatch.setattr(sys, "path", observed)
    return SimpleNamespace(
        observed=observed,
        setuptools=setuptools_module,
        instantiator=instantiator_module,
        generated=generated_module,
        remote_dir=remote_template_dir,
        remote_file=remote_template_file,
        remote_payload=remote_template_payload,
    )


def test_native_cubify_import_path_restore_is_exact_and_in_place(
    monkeypatch, tmp_path
):
    state = _install_trusted_native_cubify_import_side_effects(
        monkeypatch, tmp_path
    )
    path_object = sys.path

    runtime._restore_native_cubify_import_sys_path()

    assert sys.path is path_object
    assert tuple(sys.path) == runtime.NATIVE_FROZEN_SYS_PATH
    assert runtime._sys_path_sha256(sys.path) == runtime._sys_path_sha256(
        runtime.NATIVE_POST_FACTORY_SYS_PATH
    )
    assert state.observed is path_object


@pytest.mark.parametrize(
    "mutation",
    (
        "missing-vendor",
        "missing-instantiator-dir",
        "reordered-appends",
        "duplicate-vendor",
        "unknown-extra",
        "frozen-prefix-drift",
    ),
)
def test_native_cubify_import_path_restore_rejects_unexpected_drift_atomically(
    monkeypatch, tmp_path, mutation
):
    state = _install_trusted_native_cubify_import_side_effects(
        monkeypatch, tmp_path
    )
    vendor = os.fspath(runtime.NATIVE_SETUPTOOLS_VENDOR)
    remote = os.fspath(state.remote_dir)
    if mutation == "missing-vendor":
        candidate = [*runtime.NATIVE_FROZEN_SYS_PATH, remote]
    elif mutation == "missing-instantiator-dir":
        candidate = [*runtime.NATIVE_FROZEN_SYS_PATH, vendor]
    elif mutation == "reordered-appends":
        candidate = [*runtime.NATIVE_FROZEN_SYS_PATH, remote, vendor]
    elif mutation == "duplicate-vendor":
        candidate = [*state.observed, vendor]
    elif mutation == "unknown-extra":
        candidate = [*state.observed, os.fspath(tmp_path / "unknown")]
    else:
        candidate = ["/untrusted-prefix", *state.observed[1:]]
    monkeypatch.setattr(sys, "path", candidate)
    path_object = sys.path
    before = list(sys.path)

    with pytest.raises(runtime.IntegratedRuntimeError):
        runtime._restore_native_cubify_import_sys_path()

    assert sys.path is path_object
    assert sys.path == before


@pytest.mark.parametrize(
    "mutation",
    (
        "missing-setuptools-module",
        "setuptools-origin",
        "missing-instantiator-module",
        "instantiator-origin",
        "instantiator-directory-identity",
        "remote-directory-symlink",
        "remote-directory-mode",
        "remote-directory-extra-entry",
        "remote-loaded-module-closure",
        "remote-template-symlink",
        "missing-generated-module",
        "generated-module-origin",
        "generated-module-name",
        "generated-module-hash",
        "generated-module-owner",
        "generated-module-group",
    ),
)
def test_native_cubify_import_path_restore_rejects_untrusted_sources_atomically(
    monkeypatch, tmp_path, mutation
):
    state = _install_trusted_native_cubify_import_side_effects(
        monkeypatch, tmp_path
    )
    fake_source = tmp_path / "untrusted.py"
    fake_source.write_text("# test-only untrusted origin\n", encoding="ascii")
    if mutation == "missing-setuptools-module":
        monkeypatch.delitem(sys.modules, "setuptools")
    elif mutation == "setuptools-origin":
        state.setuptools.__file__ = os.fspath(fake_source)
        state.setuptools.__spec__.origin = os.fspath(fake_source)
    elif mutation == "missing-instantiator-module":
        monkeypatch.delitem(
            sys.modules, "torch.distributed.nn.jit.instantiator"
        )
    elif mutation == "instantiator-origin":
        state.instantiator.__file__ = os.fspath(fake_source)
        state.instantiator.__spec__.origin = os.fspath(fake_source)
    elif mutation == "instantiator-directory-identity":
        other_directory = tmp_path / "other-remote-template"
        other_directory.mkdir()
        state.instantiator.INSTANTIATED_TEMPLATE_DIR_PATH = os.fspath(
            other_directory
        )
    elif mutation == "remote-directory-symlink":
        real_directory = tmp_path / "real-remote-template"
        state.remote_dir.rename(real_directory)
        state.remote_dir.symlink_to(real_directory, target_is_directory=True)
    elif mutation == "remote-directory-mode":
        state.remote_dir.chmod(0o755)
    elif mutation == "remote-directory-extra-entry":
        (state.remote_dir / "unexpected.py").write_text(
            "# unexpected\n", encoding="ascii"
        )
    elif mutation == "remote-loaded-module-closure":
        monkeypatch.setitem(
            sys.modules,
            "unexpected_remote_module",
            SimpleNamespace(
                __file__=os.fspath(state.remote_file),
                __spec__=SimpleNamespace(origin=os.fspath(state.remote_file)),
            ),
        )
    elif mutation == "remote-template-symlink":
        real_template = tmp_path / "real-remote-template.py"
        real_template.write_bytes(state.remote_payload)
        state.remote_file.unlink()
        state.remote_file.symlink_to(real_template)
    elif mutation == "missing-generated-module":
        monkeypatch.delitem(sys.modules, "_remote_module_non_scriptable")
    elif mutation == "generated-module-origin":
        state.generated.__file__ = os.fspath(fake_source)
        state.generated.__spec__.origin = os.fspath(fake_source)
    elif mutation == "generated-module-name":
        state.generated.__spec__.name = "untrusted_remote_module"
    elif mutation == "generated-module-hash":
        state.remote_file.write_bytes(b"# corrupt remote template\n")
    elif mutation == "generated-module-owner":
        real_euid = os.geteuid()
        monkeypatch.setattr(runtime.os, "geteuid", lambda: real_euid + 1)
    else:
        real_egid = os.getegid()
        monkeypatch.setattr(runtime.os, "getegid", lambda: real_egid + 1)
    path_object = sys.path
    before = list(sys.path)

    with pytest.raises(runtime.IntegratedRuntimeError):
        runtime._restore_native_cubify_import_sys_path()

    assert sys.path is path_object
    assert sys.path == before


def test_native_cubify_import_path_restore_rolls_back_if_cache_invalidation_fails(
    monkeypatch, tmp_path
):
    _install_trusted_native_cubify_import_side_effects(monkeypatch, tmp_path)
    path_object = sys.path
    before = list(sys.path)

    def fail_cache_invalidation():
        raise RuntimeError("test cache invalidation failure")

    monkeypatch.setattr(
        runtime.importlib, "invalidate_caches", fail_cache_invalidation
    )
    with pytest.raises(RuntimeError, match="test cache invalidation failure"):
        runtime._restore_native_cubify_import_sys_path()

    assert sys.path is path_object
    assert sys.path == before


def test_held_file_rejects_public_name_swap_and_keeps_original_bytes(tmp_path):
    path = tmp_path / "manifest.json"
    payload = b'{"schema":"held"}\n'
    path.write_bytes(payload)
    held = runtime._HeldPinnedRegularFile(
        path,
        maximum=1024,
        label="test manifest",
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )
    original = tmp_path / "manifest.original.json"
    path.rename(original)
    path.write_bytes(payload)
    assert held.payload == payload
    with pytest.raises(runtime.IntegratedRuntimeError, match="identity changed"):
        held.verify_after_stream()
    held.close()


def test_held_file_rejects_in_place_byte_tamper(tmp_path):
    path = tmp_path / "schedule.json"
    payload = b'{"schedule":"original"}\n'
    path.write_bytes(payload)
    held = runtime._HeldPinnedRegularFile(
        path,
        maximum=1024,
        label="test schedule",
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )
    path.write_bytes(b'{"schedule":"tampered"}\n')
    with pytest.raises(runtime.IntegratedRuntimeError, match="changed during stream"):
        held.verify_after_stream()
    held.close()


def test_formal_schedule_is_parsed_from_the_exact_held_payload():
    with runtime._HeldPinnedRegularFile(
        runtime.DEFAULT_PROVIDER_SCHEDULE,
        maximum=runtime.MAX_ASSET_BYTES,
        label="provider schedule",
        expected_sha256=runtime.EXPECTED_PROVIDER_SCHEDULE_SHA256,
    ) as held:
        bundle = runtime._parse_held_provider_schedule(held)
        assert bundle.sha256 == held.sha256
        assert bundle.valid_frame_count == runtime.EXPECTED_PROVIDER_VALID_CALLS
        held.verify_after_stream()


def test_native_manifest_validation_consumes_held_bytes_not_a_reopened_path(
    tmp_path, monkeypatch
):
    path = tmp_path / "native.json"
    payload = b'{"schema":"same-held-bytes","value":7}\n'
    path.write_bytes(payload)
    observed = []

    def validate(value, **kwargs):
        observed.append((deepcopy(value), dict(kwargs)))
        return dict(value)

    monkeypatch.setattr(
        runtime.native_manifest_builder, "validate_native_manifest", validate
    )
    monkeypatch.setattr(
        runtime.native_manifest_builder,
        "verify_manifest_files",
        lambda value, **kwargs: observed.append((deepcopy(value), dict(kwargs))),
    )
    with runtime._HeldPinnedRegularFile(
        path,
        maximum=1024,
        label="native manifest",
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    ) as held:
        result = runtime._parse_held_native_manifest(
            held, provider_bundle=object(), scene_root=tmp_path
        )
        assert result == {"schema": "same-held-bytes", "value": 7}
        assert observed[0][0] == result
        assert observed[1][0] == result
        held.verify_after_stream()


def test_formal_paths_are_exact_and_arm_specific(tmp_path, monkeypatch):
    control = tmp_path / "control.json"
    integrated = tmp_path / "integrated.json"
    monkeypatch.setattr(runtime, "FORMAL_CONTROL_OUTPUT", control)
    monkeypatch.setattr(runtime, "FORMAL_INTEGRATED_OUTPUT", integrated)
    assert runtime._require_formal_paths(
        arm="control", output=control, control_receipt=None
    ) == (control, None)
    assert runtime._require_formal_paths(
        arm="integrated", output=integrated, control_receipt=control
    ) == (integrated, control)
    with pytest.raises(runtime.IntegratedRuntimeError, match="output path differs"):
        runtime._require_formal_paths(
            arm="control", output=tmp_path / "alternative.json", control_receipt=None
        )
    with pytest.raises(runtime.IntegratedRuntimeError, match="absolute path"):
        runtime._require_formal_paths(
            arm="control",
            output=Path("logs/scannet_s3r_h10_runtime_control_v2.json"),
            control_receipt=None,
        )
    with pytest.raises(runtime.IntegratedRuntimeError, match="receipt path differs"):
        runtime._require_formal_paths(
            arm="integrated",
            output=integrated,
            control_receipt=tmp_path / "picked-control.json",
        )
    with pytest.raises(runtime.IntegratedRuntimeError, match="absolute path"):
        runtime._require_formal_paths(
            arm="integrated",
            output=integrated,
            control_receipt=Path("logs/scannet_s3r_h10_runtime_control_v2.json"),
        )


def test_formal_public_api_has_no_factory_injection_and_bad_pin_is_prepublish(
    tmp_path, monkeypatch
):
    assert "factory" not in inspect.signature(
        runtime.run_formal_h10_runtime_arm
    ).parameters
    contract = tmp_path / "contract.md"
    contract.write_text("opaque runtime contract\n", encoding="utf-8")
    output = tmp_path / "formal.json"
    monkeypatch.setattr(runtime, "FORMAL_CONTROL_OUTPUT", output)
    monkeypatch.setattr(
        runtime,
        "_validate_parent_runtime_identity",
        lambda: {
            "python_executable": os.fspath(runtime.NATIVE_PYTHON_EXECUTABLE),
            "python_version": runtime.NATIVE_RUNTIME_IDENTITY["python_version"],
            "numpy_version": runtime.NATIVE_RUNTIME_IDENTITY["numpy_version"],
            "torch_imported": False,
            "python_pycache_prefix_environment": (
                runtime.FROZEN_PYCACHE_PREFIX
            ),
            "python_pycache_prefix": runtime.FROZEN_PYCACHE_PREFIX,
        },
    )
    with pytest.raises(runtime.IntegratedRuntimeError, match="byte pin"):
        runtime.run_formal_h10_runtime_arm(
            arm="control",
            output=output,
            runtime_contract=contract,
            expected_runtime_contract_sha256=hashlib.sha256(
                contract.read_bytes()
            ).hexdigest(),
            expected_runner_sha256="0" * 64,
            expected_runner_test_sha256="0" * 64,
        )
    assert not output.exists()


def test_wrong_parent_runtime_fails_before_formal_input_or_spawn(tmp_path, monkeypatch):
    wrong_python = tmp_path / "wrong-python"
    wrong_python.write_bytes(b"not-the-frozen-parent")
    output = tmp_path / "formal.json"
    touched = []
    monkeypatch.setattr(runtime, "NATIVE_PYTHON_EXECUTABLE", wrong_python)
    monkeypatch.setattr(runtime, "FORMAL_CONTROL_OUTPUT", output)
    monkeypatch.setattr(
        runtime,
        "_validate_formal_self_and_contract",
        lambda **_kwargs: touched.append("formal-input"),
    )
    with pytest.raises(runtime.IntegratedRuntimeError, match="parent runtime identity"):
        runtime.run_formal_h10_runtime_arm(
            arm="control",
            output=output,
            runtime_contract=tmp_path / "not-read-contract.md",
            expected_runtime_contract_sha256="1" * 64,
            expected_runner_sha256="2" * 64,
            expected_runner_test_sha256="3" * 64,
        )
    assert touched == []
    assert not output.exists()


def test_provider_ignored_guard_precedes_provider_runtime_probe(tmp_path, monkeypatch):
    output = tmp_path / "formal.json"
    events = []
    monkeypatch.setattr(runtime, "FORMAL_CONTROL_OUTPUT", output)
    monkeypatch.setattr(
        runtime,
        "_validate_parent_runtime_identity",
        lambda: {"parent": "frozen"},
    )
    monkeypatch.setattr(
        runtime,
        "_validate_formal_self_and_contract",
        lambda **_kwargs: {"pins": "frozen"},
    )
    monkeypatch.setattr(runtime, "_formal_output_preflight", lambda path, **_kwargs: path)
    monkeypatch.setattr(
        runtime,
        "_validate_environment",
        lambda **_kwargs: {"CUDA_VISIBLE_DEVICES": "7"},
    )
    monkeypatch.setattr(
        runtime,
        "_snapshot_external_command_binaries",
        lambda: events.append("external-binary-pin") or {},
    )
    monkeypatch.setattr(
        runtime,
        "_assert_import_shadow_candidates_absent",
        lambda _paths: events.append("local-shadow-guard") or (),
    )

    def fail_guard(_root):
        events.append("ignored-guard")
        raise runtime.IntegratedRuntimeError("guard stopped probe")

    def forbidden_probe(**_kwargs):
        events.append("provider-probe")
        raise AssertionError("provider probe ran before ignored guard")

    monkeypatch.setattr(runtime, "_snapshot_provider_checkout", fail_guard)
    monkeypatch.setattr(runtime, "_probe_frozen_child_runtime", forbidden_probe)
    with pytest.raises(runtime.IntegratedRuntimeError, match="guard stopped probe"):
        runtime.run_formal_h10_runtime_arm(
            arm="control",
            output=output,
            runtime_contract=tmp_path / "contract.md",
            expected_runtime_contract_sha256="1" * 64,
            expected_runner_sha256="2" * 64,
            expected_runner_test_sha256="3" * 64,
        )
    assert events == [
        "external-binary-pin",
        "local-shadow-guard",
        "ignored-guard",
    ]


def test_post_provider_probe_precedes_final_guards_and_publication(
    tmp_path, monkeypatch
):
    events = []
    schedule_sha = "1" * 64
    manifest_sha = "2" * 64
    self_pins = {
        "runtime_contract": "3" * 64,
        "runner": "4" * 64,
        "runner_test": "5" * 64,
    }
    parent_identity = {"parent": "frozen"}
    child_probes = {
        role: {
            "python_executable": f"/{role}/python",
            "python_sys_path": [f"/{role}/path"],
            "expected_post_factory_python_sys_path": [f"/{role}/post"],
        }
        for role in ("native", "provider")
    }
    checkout = {
        "identity_sha256": "6" * 64,
        "commit": "a" * 40,
        "tree": "b" * 40,
    }
    static = {"identity_sha256": "7" * 64}
    inputs = {"identity_sha256": "8" * 64}
    t05 = {"identity_sha256": "9" * 64}
    provider_assets = {"asset": {"sha256_before": "c" * 64}}
    bundle = SimpleNamespace(sha256=schedule_sha, scene_order=())

    class Held:
        def __init__(self, sha256):
            self.sha256 = sha256
            self.payload = b"{}"

        def verify_after_stream(self):
            events.append("held-input-verify")

    held_schedule = Held(schedule_sha)
    held_manifest = Held(manifest_sha)
    fresh = SimpleNamespace(
        BOXER_ROOT=runtime.PROVIDER_BOXER_ROOT,
        BOXER_CHECKPOINT_RELPATH="ckpts/boxer.ckpt",
        OWL_CHECKPOINT=tmp_path / "owl.pt",
        _validate_frozen_assets=lambda *_args: (provider_assets, {}),
        _rehash_assets=lambda *_args: {"asset": "c" * 64},
    )

    monkeypatch.setattr(runtime, "_parse_held_provider_schedule", lambda _held: bundle)
    monkeypatch.setattr(runtime, "_parse_held_native_manifest", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runtime, "_minimal_manifest_view", lambda _manifest: {})
    monkeypatch.setattr(runtime, "_assert_formal_manifest_view", lambda *_args: None)
    checkout_calls = 0

    def snapshot_checkout(_root):
        nonlocal checkout_calls
        checkout_calls += 1
        events.append(
            "provider-checkout-before"
            if checkout_calls == 1
            else "final-provider-guard"
        )
        return checkout

    static_calls = 0

    def snapshot_static(**_kwargs):
        nonlocal static_calls
        static_calls += 1
        events.append(
            "local-static-before"
            if static_calls == 1
            else "final-local-guard"
        )
        return static

    input_calls = 0

    def snapshot_inputs(*_args, **_kwargs):
        nonlocal input_calls
        input_calls += 1
        events.append("inputs-before" if input_calls == 1 else "inputs-after")
        return inputs

    monkeypatch.setattr(runtime, "_snapshot_provider_checkout", snapshot_checkout)
    monkeypatch.setattr(runtime, "_formal_static_snapshot", snapshot_static)
    monkeypatch.setattr(runtime, "_snapshot_manifest_inputs", snapshot_inputs)
    monkeypatch.setattr(runtime, "_snapshot_t05_opaque", lambda _bundle: t05)
    monkeypatch.setattr(runtime.importlib, "import_module", lambda _name: fresh)
    monkeypatch.setattr(runtime, "_child_runtime_identity_sha256", lambda _value: "d" * 64)
    monkeypatch.setattr(
        runtime,
        "_execute_control_runtime",
        lambda **_kwargs: events.append("stream")
        or {"performance_gates": {"all_met": True}},
    )
    monkeypatch.setattr(runtime, "_assert_formal_runtime_result", lambda *_args, **_kwargs: None)

    def post_probe(*, boxer_root=None, executable, **_kwargs):
        role = "provider" if boxer_root is not None else "native"
        events.append(f"post-{role}-probe")
        return child_probes[role]

    monkeypatch.setattr(runtime, "_probe_frozen_child_runtime", post_probe)
    monkeypatch.setattr(
        runtime.native_manifest_builder,
        "verify_manifest_files",
        lambda *_args, **_kwargs: events.append("manifest-verify"),
    )
    monkeypatch.setattr(
        runtime,
        "_validate_formal_self_and_contract",
        lambda **_kwargs: events.append("self-pin-verify") or self_pins,
    )
    monkeypatch.setattr(
        runtime,
        "_validate_parent_runtime_identity",
        lambda: events.append("parent-identity-verify") or parent_identity,
    )
    monkeypatch.setattr(
        runtime,
        "_publish_timing_receipt",
        lambda *_args: events.append("publication"),
    )

    result = runtime._run_formal_h10_runtime_arm_with_held_inputs(
        arm="control",
        output=tmp_path / "formal.json",
        runtime_contract=tmp_path / "contract.md",
        expected_runtime_contract_sha256=self_pins["runtime_contract"],
        expected_runner_sha256=self_pins["runner"],
        expected_runner_test_sha256=self_pins["runner_test"],
        self_pins=self_pins,
        environment={"CUDA_VISIBLE_DEVICES": "7"},
        parent_runtime_identity=parent_identity,
        child_runtime_probes=child_probes,
        provider_checkout_preprobe=checkout,
        held_schedule=held_schedule,
        held_manifest=held_manifest,
        held_control=None,
    )
    assert result["immutable_before_after_verified"] is True
    assert events.index("post-provider-probe") < events.index("final-local-guard")
    assert events.index("post-provider-probe") < events.index("final-provider-guard")
    assert events.index("final-local-guard") < events.index("final-provider-guard")
    assert events[-1] == "publication"


def test_formal_environment_rejects_required_value_drift(
    monkeypatch,
):
    for name, expected in runtime.REQUIRED_ENVIRONMENT.items():
        monkeypatch.setenv(name, expected)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONPYCACHEPREFIX",
        "PATH",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
    ):
        monkeypatch.setenv(name, "/unfrozen/injection")
        with pytest.raises(runtime.IntegratedRuntimeError, match="environment differs"):
            runtime._validate_environment(require_cuda=True)
        monkeypatch.setenv(name, runtime.REQUIRED_ENVIRONMENT[name])
    observed = runtime._validate_environment(require_cuda=True)
    assert observed["git_environment_names_absent"] is True


@pytest.mark.parametrize(
    "name",
    [
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
    ],
)
def test_formal_environment_rejects_git_name_even_when_empty(monkeypatch, name):
    for required_name, expected in runtime.REQUIRED_ENVIRONMENT.items():
        monkeypatch.setenv(required_name, expected)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    assert not any(key.startswith("GIT_") for key in os.environ)
    monkeypatch.setenv(name, "")
    with pytest.raises(runtime.IntegratedRuntimeError, match="environment differs"):
        runtime._validate_environment(require_cuda=True)


@pytest.mark.parametrize("name", ["GIT_OBJECT_DIRECTORY", "LD_AUDIT"])
def test_formal_environment_rejects_unknown_git_or_loader_variable(
    monkeypatch, name
):
    for required_name, expected in runtime.REQUIRED_ENVIRONMENT.items():
        monkeypatch.setenv(required_name, expected)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.delenv("GIT_PAGER", raising=False)
    monkeypatch.setenv(name, "/unfrozen/injection")
    with pytest.raises(runtime.IntegratedRuntimeError, match="environment differs"):
        runtime._validate_environment(require_cuda=True)


def test_parent_runtime_binds_environment_and_interpreter_pycache_prefix(
    monkeypatch,
):
    assert "torch" not in sys.modules
    monkeypatch.setattr(runtime, "NATIVE_PYTHON_EXECUTABLE", Path(sys.executable))
    identity = dict(runtime.NATIVE_RUNTIME_IDENTITY)
    identity["python_version"] = sys.version.split()[0]
    identity["numpy_version"] = str(np.__version__)
    monkeypatch.setattr(runtime, "NATIVE_RUNTIME_IDENTITY", identity)
    monkeypatch.setenv(
        "PYTHONPYCACHEPREFIX", runtime.FROZEN_PYCACHE_PREFIX
    )
    monkeypatch.setattr(
        sys, "pycache_prefix", runtime.FROZEN_PYCACHE_PREFIX
    )

    observed = runtime._validate_parent_runtime_identity()
    assert observed["python_pycache_prefix_environment"] == "/dev/null"
    assert observed["python_pycache_prefix"] == "/dev/null"

    monkeypatch.setenv("PYTHONPYCACHEPREFIX", "/tmp/not-frozen")
    with pytest.raises(runtime.IntegratedRuntimeError, match="parent runtime identity"):
        runtime._validate_parent_runtime_identity()
    monkeypatch.setenv(
        "PYTHONPYCACHEPREFIX", runtime.FROZEN_PYCACHE_PREFIX
    )
    monkeypatch.setattr(sys, "pycache_prefix", "/tmp/not-frozen")
    with pytest.raises(runtime.IntegratedRuntimeError, match="parent runtime identity"):
        runtime._validate_parent_runtime_identity()


def test_devnull_pycache_prefix_ignores_valid_adjacent_timestamp_pyc(
    tmp_path,
):
    module = tmp_path / "pycache_identity_probe.py"
    cached_source = b"VALUE = 'cached'\n"
    live_source = b"VALUE = 'source'\n"
    assert len(cached_source) == len(live_source)
    fixed_seconds = 1_700_000_000
    fixed_ns = fixed_seconds * 1_000_000_000

    module.write_bytes(cached_source)
    os.utime(module, ns=(fixed_ns, fixed_ns))
    # Construct the adjacent cache path explicitly.  The pytest parent may
    # itself already have PYTHONPYCACHEPREFIX=/dev/null, in which case
    # importlib.util.cache_from_source() intentionally points under /dev/null.
    cache = (
        module.parent
        / "__pycache__"
        / f"{module.stem}.{sys.implementation.cache_tag}.pyc"
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    py_compile.compile(
        os.fspath(module),
        cfile=os.fspath(cache),
        doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
    )
    malicious_pyc = cache.read_bytes()
    magic, flags, header_mtime, header_size = struct.unpack(
        "<4sIII", malicious_pyc[:16]
    )
    assert magic == importlib.util.MAGIC_NUMBER
    assert flags == 0
    assert header_mtime == fixed_seconds
    assert header_size == len(live_source)

    # Preserve the valid timestamp/size header while changing source behavior.
    module.write_bytes(live_source)
    os.utime(module, ns=(fixed_ns, fixed_ns))
    probe = (
        "import json,sys,pycache_identity_probe as module;"
        "print(json.dumps([sys.pycache_prefix,module.VALUE]))"
    )
    ordinary_environment = dict(os.environ)
    ordinary_environment.pop("PYTHONHOME", None)
    ordinary_environment.pop("PYTHONPATH", None)
    ordinary_environment.pop("PYTHONPYCACHEPREFIX", None)
    ordinary_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    ordinary_environment["PYTHONNOUSERSITE"] = "1"
    ordinary = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=ordinary_environment,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert json.loads(ordinary.stdout) == [None, "cached"]

    hardened_environment = dict(ordinary_environment)
    hardened_environment["PYTHONPYCACHEPREFIX"] = (
        runtime.FROZEN_PYCACHE_PREFIX
    )
    hardened = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=hardened_environment,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert json.loads(hardened.stdout) == ["/dev/null", "source"]
    assert cache.read_bytes() == malicious_pyc


def test_cli_missing_pins_fails_before_public_runner(monkeypatch):
    called = []
    monkeypatch.setattr(
        runtime,
        "run_formal_h10_runtime_arm",
        lambda **_kwargs: called.append(True),
    )
    with pytest.raises(SystemExit) as raised:
        runtime.main(["--arm", "control", "--output", "/tmp/never-created"])
    assert raised.value.code == 2
    assert called == []


def test_heterogeneous_spawn_interpreters_exchange_causal_queue_acks(tmp_path):
    parent_sys_path = list(sys.path)
    parent_spawn_executable = mp_spawn.get_executable()
    native_executable = runtime.NATIVE_PYTHON_EXECUTABLE.resolve(strict=True)
    provider_executable = runtime.PROVIDER_PYTHON_EXECUTABLE.resolve(strict=True)
    native_probe = runtime._probe_frozen_child_runtime(
        executable=native_executable,
        expected_spawn_entry_sys_path=runtime.NATIVE_FROZEN_SYS_PATH,
        expected_post_factory_sys_path=runtime.NATIVE_POST_FACTORY_SYS_PATH,
        expected_identity=runtime.NATIVE_RUNTIME_IDENTITY,
    )
    provider_probe = runtime._probe_frozen_child_runtime(
        executable=provider_executable,
        expected_spawn_entry_sys_path=runtime.PROVIDER_FROZEN_SYS_PATH,
        expected_post_factory_sys_path=runtime.PROVIDER_POST_FACTORY_SYS_PATH,
        expected_identity=runtime.PROVIDER_RUNTIME_IDENTITY,
        boxer_root=runtime.PROVIDER_BOXER_ROOT,
    )
    for probe in (native_probe, provider_probe):
        assert probe["runtime_identity"][
            "python_pycache_prefix_environment"
        ] == runtime.FROZEN_PYCACHE_PREFIX
        assert probe["runtime_identity"][
            "python_pycache_prefix"
        ] == runtime.FROZEN_PYCACHE_PREFIX
    assert len(
        runtime._child_runtime_identity_sha256(
            {"native": native_probe, "provider": provider_probe}
        )
    ) == 64
    output, result = _run(
        tmp_path,
        native_config={
            "gpu_uuid": "GPU-TEST-0",
            **native_probe,
            "expected_runtime_identity": native_probe["runtime_identity"],
            "probe_real_runtime_identity": True,
        },
        provider_config={
            "gpu_uuid": "GPU-TEST-0",
            **provider_probe,
            "expected_runtime_identity": provider_probe["runtime_identity"],
            "probe_real_runtime_identity": True,
            "apply_real_provider_sys_path_transform": True,
        },
    )
    assert output.is_file()
    assert sys.path == parent_sys_path
    assert mp_spawn.get_executable() == parent_spawn_executable
    assert Path(result["workers"]["native"]["python_executable"]).resolve() == (
        native_executable
    )
    assert Path(result["workers"]["provider"]["python_executable"]).resolve() == (
        provider_executable
    )
    assert result["workers"]["native"]["python_version"] == "3.10.13"
    assert result["workers"]["provider"]["python_version"] == "3.10.19"
    assert result["workers"]["native"]["torch_version"] == "2.6.0+cu124"
    assert result["workers"]["provider"]["torch_version"] == "2.2.0+cu121"
    assert result["workers"]["native"]["cuda_version"] == "12.4"
    assert result["workers"]["provider"]["cuda_version"] == "12.1"
    assert result["workers"]["native"]["numpy_origin"].startswith(
        "/home/admin1/miniconda3/envs/boxfusion-online/"
    )
    assert result["workers"]["native"]["torch_origin"].startswith(
        "/home/admin1/miniconda3/envs/boxfusion-online/"
    )
    assert result["workers"]["provider"]["numpy_origin"].startswith(
        "/home/admin1/miniconda3/envs/ovm3d-1/"
    )
    assert result["workers"]["provider"]["torch_origin"].startswith(
        "/home/admin1/miniconda3/envs/ovm3d-1/"
    )
    assert result["workers"]["native"][
        "spawn_entry_python_sys_path_sha256"
    ] == runtime._sys_path_sha256(runtime.NATIVE_FROZEN_SYS_PATH)
    assert result["workers"]["native"][
        "post_factory_python_sys_path_sha256"
    ] == runtime._sys_path_sha256(runtime.NATIVE_POST_FACTORY_SYS_PATH)
    assert result["workers"]["provider"][
        "spawn_entry_python_sys_path_sha256"
    ] == runtime._sys_path_sha256(runtime.PROVIDER_FROZEN_SYS_PATH)
    assert result["workers"]["provider"][
        "post_factory_python_sys_path_sha256"
    ] == runtime._sys_path_sha256(runtime.PROVIDER_POST_FACTORY_SYS_PATH)
    assert result["workers"]["provider"][
        "post_factory_python_sys_path_sha256"
    ] != result["workers"]["provider"][
        "spawn_entry_python_sys_path_sha256"
    ]
    for role in ("native", "provider"):
        assert result["workers"][role][
            "python_pycache_prefix_environment"
        ] == runtime.FROZEN_PYCACHE_PREFIX
        assert result["workers"][role][
            "python_pycache_prefix"
        ] == runtime.FROZEN_PYCACHE_PREFIX
    assert [row["frame_id"] for row in result["causal_frame_ledger"]] == [0, 1, 2]


def _held_reader_fixture(tmp_path: Path):
    scene_directory = tmp_path / "scene_reader"
    frames = scene_directory / "frames"
    for role in ("color", "depth", "pose", "intrinsic"):
        (frames / role).mkdir(parents=True, exist_ok=True)
    intrinsic = np.eye(4, dtype=np.float64)
    intrinsic_payload = "\n".join(
        " ".join(str(value) for value in row) for row in intrinsic
    ).encode("ascii") + b"\n"
    color_intrinsic = frames / "intrinsic" / "intrinsic_color.txt"
    depth_intrinsic = frames / "intrinsic" / "intrinsic_depth.txt"
    color_intrinsic.write_bytes(intrinsic_payload)
    depth_intrinsic.write_bytes(intrinsic_payload)
    finite_pose = np.eye(4, dtype=np.float64)
    finite_pose[0, 3] = 1.25
    finite_payload = "\n".join(
        " ".join(str(value) for value in row) for row in finite_pose
    ).encode("ascii") + b"\n"
    inf_pose = np.eye(4, dtype=np.float64)
    inf_pose[1, 3] = np.inf
    inf_payload = "\n".join(
        " ".join(str(value) for value in row) for row in inf_pose
    ).encode("ascii") + b"\n"
    payloads = {
        ("color", "0.jpg"): b"jpeg-current-0",
        ("color", "1.jpg"): b"jpeg-current-1",
        ("depth", "0.png"): b"depth-current-0",
        ("depth", "1.png"): b"depth-current-1",
        ("pose", "0.txt"): finite_payload,
        ("pose", "1.txt"): inf_payload,
    }
    for (role, name), payload in payloads.items():
        (frames / role / name).write_bytes(payload)
    digest = lambda payload: hashlib.sha256(payload).hexdigest()
    intrinsic_sha = digest(intrinsic_payload)
    scene = {
        "scene_id": "scene_reader",
        "scene_index": 0,
        "native_frame_count": 2,
        "scene_directory": os.fspath(scene_directory),
        "intrinsic_color_relpath": "frames/intrinsic/intrinsic_color.txt",
        "intrinsic_color_sha256": intrinsic_sha,
        "intrinsic_depth_relpath": "frames/intrinsic/intrinsic_depth.txt",
        "intrinsic_depth_sha256": intrinsic_sha,
        "role_mounts": runtime.native_manifest_builder._scene_mounts(
            frames, scene_id="scene_reader"
        ),
    }
    frame0 = {
        "scene_id": "scene_reader",
        "scene_index": 0,
        "scene_frame_index": 0,
        "global_frame_index": 0,
        "frame_id": 0,
        "color_relpath": "frames/color/0.jpg",
        "color_sha256": digest(payloads[("color", "0.jpg")]),
        "depth_relpath": "frames/depth/0.png",
        "depth_sha256": digest(payloads[("depth", "0.png")]),
        "pose_relpath": "frames/pose/0.txt",
        "pose_sha256": digest(finite_payload),
        "raw_pose_finite": True,
        "effective_pose_frame_id": 0,
        "effective_pose_relpath": "frames/pose/0.txt",
        "effective_pose_sha256": digest(finite_payload),
        "pose_resolution": "current_finite",
        "provider_status": runtime.PROVIDER_MEMBER,
    }
    frame1 = {
        **frame0,
        "scene_frame_index": 1,
        "global_frame_index": 1,
        "frame_id": 1,
        "color_relpath": "frames/color/1.jpg",
        "color_sha256": digest(payloads[("color", "1.jpg")]),
        "depth_relpath": "frames/depth/1.png",
        "depth_sha256": digest(payloads[("depth", "1.png")]),
        "pose_relpath": "frames/pose/1.txt",
        "pose_sha256": digest(inf_payload),
        "raw_pose_finite": False,
        "effective_pose_frame_id": 0,
        "effective_pose_relpath": "frames/pose/0.txt",
        "effective_pose_sha256": digest(finite_payload),
        "pose_resolution": "past_most_recent_valid",
        "provider_status": runtime.PROVIDER_ABSTAIN,
    }
    return scene, frame0, frame1


def test_current_only_reader_uses_actual_bytes_and_cached_past_pose(
    tmp_path, monkeypatch
):
    scene, frame0, frame1 = _held_reader_fixture(tmp_path)
    monkeypatch.setattr(os, "listdir", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("enumeration")))
    monkeypatch.setattr(os, "scandir", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("enumeration")))
    monkeypatch.setattr(Path, "glob", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("enumeration")))
    monkeypatch.setattr(Path, "iterdir", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("enumeration")))
    reader = runtime.HeldManifestSceneReader(scene, mode="native")
    first = reader.read_current(frame0)
    second = reader.read_current(frame1)
    assert first.input_identity_sha256 == runtime._expected_frame_input_identity(frame0)
    assert second.input_identity_sha256 == runtime._expected_frame_input_identity(frame1)
    assert np.isinf(second.pose_raw).any()
    assert np.array_equal(second.pose_effective, first.pose_effective)
    reader.close()


def test_current_only_reader_rejects_hash_mismatch(tmp_path):
    scene, frame0, _frame1 = _held_reader_fixture(tmp_path)
    reader = runtime.HeldManifestSceneReader(scene, mode="native")
    (Path(scene["scene_directory"]) / frame0["color_relpath"]).write_bytes(
        b"tampered-current"
    )
    with pytest.raises(runtime.IntegratedRuntimeError, match="hash differs"):
        reader.read_current(frame0)
    reader.close(abort=True)


def _control_receipt_fixture(tmp_path, bindings):
    _, receipt = _run_control(
        tmp_path,
        native_config={
            "gpu_uuid": "GPU-CROSS-ARM",
            "gpu_device_name": "cross-arm-gpu",
            "gpu_total_memory_bytes": 24 * 1024**3,
            "gpu_driver_version": "999.1",
            "python_executable": os.fspath(runtime.NATIVE_PYTHON_EXECUTABLE),
            "python_sys_path": list(runtime.NATIVE_FROZEN_SYS_PATH),
            "expected_post_factory_python_sys_path": list(
                runtime.NATIVE_POST_FACTORY_SYS_PATH
            ),
            "expected_runtime_identity": dict(runtime.NATIVE_RUNTIME_IDENTITY),
            "probe_real_runtime_identity": True,
        },
    )
    receipt.update(
        {
            "formal_h10": True,
            "synthetic_worker_injection": False,
            "timing_only": True,
            "runtime_only": True,
            "original_terminal_exact": False,
            "full100_not_authorized": True,
            "h10_gt_oracle_authorized": False,
            "gt_access_authorized": False,
            "integrated_provider_runtime_qualified": False,
            "integrated_realtime_qualified": False,
            "native_fps_protocol_equivalent": False,
            "bindings": dict(bindings),
            "immutable_before_after_verified": True,
            "environment": {
                **dict(runtime.REQUIRED_ENVIRONMENT),
                "CUDA_VISIBLE_DEVICES": "7",
                "git_environment_names_absent": True,
            },
            "parent_runtime_identity": {
                "python_executable": os.fspath(
                    runtime.NATIVE_PYTHON_EXECUTABLE.resolve(strict=True)
                ),
                "python_version": runtime.NATIVE_RUNTIME_IDENTITY[
                    "python_version"
                ],
                "numpy_version": runtime.NATIVE_RUNTIME_IDENTITY["numpy_version"],
                "torch_imported": False,
                "python_pycache_prefix_environment": (
                    runtime.FROZEN_PYCACHE_PREFIX
                ),
                "python_pycache_prefix": runtime.FROZEN_PYCACHE_PREFIX,
            },
        }
    )
    return receipt


@pytest.mark.parametrize(
    "tamper", ["binding", "driver", "cvd", "worker_pycache", "parent_pycache"]
)
def test_control_receipt_cross_arm_identity_is_strict(
    tmp_path, monkeypatch, tamper
):
    monkeypatch.setattr(runtime, "EXPECTED_NATIVE_FRAME_COUNT", 3)
    monkeypatch.setattr(runtime, "EXPECTED_NATIVE_SCHEDULED_KEYFRAME_SLOTS", 1)
    monkeypatch.setattr(runtime, "EXPECTED_SCENE_COUNT", 1)
    monkeypatch.setattr(
        runtime, "EXPECTED_PROVIDER_RAW_ROWS_PER_SCENE", {"scene_test": 0}
    )
    bindings = {
        "native_manifest_sha256": runtime.EXPECTED_NATIVE_MANIFEST_SHA256,
        "runner_sha256": "2" * 64,
        "manifest_inputs_identity_sha256": "3" * 64,
    }
    receipt = _control_receipt_fixture(tmp_path, bindings)
    if tamper == "binding":
        receipt["bindings"]["runner_sha256"] = "4" * 64
    elif tamper == "driver":
        receipt["workers"]["native"]["gpu_driver_version"] = "wrong"
    elif tamper == "worker_pycache":
        receipt["workers"]["native"]["python_pycache_prefix"] = "/tmp/wrong"
    elif tamper == "parent_pycache":
        receipt["parent_runtime_identity"][
            "python_pycache_prefix_environment"
        ] = "/tmp/wrong"
    else:
        receipt["workers"]["native"]["cuda_visible_devices"] = "0"
    path = tmp_path / "control.json"
    payload = runtime._canonical_json_bytes(receipt)
    path.write_bytes(payload)
    expected_bindings = dict(bindings)
    if tamper == "driver":
        validated = runtime._validate_control_receipt(
            receipt,
            hashlib.sha256(payload).hexdigest(),
            expected_bindings=expected_bindings,
            expected_cuda_visible_devices="7",
            expected_manifest_view=runtime._minimal_manifest_view(_manifest()),
        )
        integrated = dict(validated["worker_identity"])
        integrated["gpu_driver_version"] = "999.1"
        with pytest.raises(runtime.IntegratedRuntimeError, match="GPU/runtime"):
            runtime._assert_cross_arm_worker_identity(
                validated["worker_identity"], integrated
            )
        return
    with pytest.raises(
        runtime.IntegratedRuntimeError,
        match=(
            "binding values differ|CUDA_VISIBLE_DEVICES differs|"
            "frozen runtime identity differs|parent runtime identity differs"
        ),
    ):
        runtime._validate_control_receipt(
            receipt,
            hashlib.sha256(payload).hexdigest(),
            expected_bindings=expected_bindings,
            expected_cuda_visible_devices="7",
            expected_manifest_view=runtime._minimal_manifest_view(_manifest()),
        )


def test_control_receipt_requires_complete_formal_stopping_schema(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runtime, "EXPECTED_NATIVE_FRAME_COUNT", 3)
    monkeypatch.setattr(runtime, "EXPECTED_NATIVE_SCHEDULED_KEYFRAME_SLOTS", 1)
    monkeypatch.setattr(runtime, "EXPECTED_SCENE_COUNT", 1)
    monkeypatch.setattr(
        runtime, "EXPECTED_PROVIDER_RAW_ROWS_PER_SCENE", {"scene_test": 0}
    )
    bindings = {
        "native_manifest_sha256": runtime.EXPECTED_NATIVE_MANIFEST_SHA256,
        "runner_sha256": "2" * 64,
        "manifest_inputs_identity_sha256": "3" * 64,
    }
    receipt = _control_receipt_fixture(tmp_path, bindings)
    mutations = []
    missing_geometry = deepcopy(receipt)
    missing_geometry.pop("geometry_serialized")
    mutations.append(missing_geometry)
    for field, value in (
        ("labels_serialized", True),
        ("synthetic_worker_injection", True),
        ("timing_only", False),
        ("immutable_before_after_verified", False),
        ("full100_not_authorized", False),
    ):
        mutated = deepcopy(receipt)
        mutated[field] = value
        mutations.append(mutated)
    for mutated in mutations:
        with pytest.raises(runtime.IntegratedRuntimeError):
            runtime._validate_control_receipt(
                mutated,
                hashlib.sha256(runtime._canonical_json_bytes(mutated)).hexdigest(),
                expected_bindings=bindings,
                expected_cuda_visible_devices="7",
                expected_manifest_view=runtime._minimal_manifest_view(_manifest()),
            )


@pytest.mark.parametrize("tamper", ["causal_identity", "runtime_summary"])
def test_control_receipt_recomputes_timing_and_binds_manifest_rows(
    tmp_path, monkeypatch, tamper
):
    monkeypatch.setattr(runtime, "EXPECTED_NATIVE_FRAME_COUNT", 3)
    monkeypatch.setattr(runtime, "EXPECTED_NATIVE_SCHEDULED_KEYFRAME_SLOTS", 1)
    monkeypatch.setattr(runtime, "EXPECTED_SCENE_COUNT", 1)
    monkeypatch.setattr(
        runtime, "EXPECTED_PROVIDER_RAW_ROWS_PER_SCENE", {"scene_test": 0}
    )
    bindings = {
        "native_manifest_sha256": runtime.EXPECTED_NATIVE_MANIFEST_SHA256,
        "runner_sha256": "2" * 64,
        "manifest_inputs_identity_sha256": "3" * 64,
    }
    receipt = _control_receipt_fixture(tmp_path, bindings)
    manifest_view = runtime._minimal_manifest_view(_manifest())
    runtime._validate_control_receipt(
        receipt,
        hashlib.sha256(runtime._canonical_json_bytes(receipt)).hexdigest(),
        expected_bindings=bindings,
        expected_cuda_visible_devices="7",
        expected_manifest_view=manifest_view,
    )
    if tamper == "causal_identity":
        receipt["causal_frame_ledger"][1]["frame_id"] = 99
    else:
        receipt["native"]["frame_runtime"]["p95_ns"] += 1
    with pytest.raises(
        runtime.IntegratedRuntimeError,
        match="causal manifest identity differs|frame runtime summary differs",
    ):
        runtime._validate_control_receipt(
            receipt,
            hashlib.sha256(runtime._canonical_json_bytes(receipt)).hexdigest(),
            expected_bindings=bindings,
            expected_cuda_visible_devices="7",
            expected_manifest_view=manifest_view,
        )
