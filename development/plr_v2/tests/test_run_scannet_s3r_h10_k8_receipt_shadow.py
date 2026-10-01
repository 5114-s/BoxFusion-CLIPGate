from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle

import numpy as np
import pytest

import tools.run_scannet_s3r_h10_k8_receipt_shadow as h10


@pytest.fixture(autouse=True)
def _pin_numeric_threads(monkeypatch, request):
    if request.node.name == (
        "test_formal_numeric_source_preflight_recomputes_exact_content_and_k8"
    ):
        return
    for name, value in h10.REQUIRED_NUMERIC_THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    # Unit tests verify accounting/guards, not host-load timing variance.
    monkeypatch.setattr(h10, "TRACKER_CPU_P95_LIMIT_NS", 1_000_000_000)
    monkeypatch.setattr(h10, "TRACKER_CPU_MAX_LIMIT_NS", 1_000_000_000)
    monkeypatch.setattr(h10, "MAX_TRACKER_INCREMENTAL_MEMORY_BYTES", 1024**3)


def _raw_arrays() -> dict[str, np.ndarray]:
    # The valid-frame ledger contains one empty frame.  A single spatially
    # stable proposal appears in four distinct frames and confirms at frame 75.
    frame_id = np.asarray([0, 50, 75, 100], dtype=np.int64)
    count = len(frame_id)
    return {
        "scene_ids": np.asarray(["scene_test"], dtype="<U10"),
        "per_view_scene_index": np.zeros(count, dtype=np.int16),
        "per_view_frame_id": frame_id,
        "per_view_source_row": np.zeros(count, dtype=np.int64),
        "per_view_source_instance_id": np.asarray(
            [0, 2 * 2048, 3 * 2048, 4 * 2048], dtype=np.int64
        ),
        "per_view_source_score": np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float64),
        "per_view_center_world": np.asarray(
            [[1.0, 2.0, 3.0]] * count, dtype=np.float64
        ),
        "per_view_extent_xyz": np.ones((count, 3), dtype=np.float64),
        "per_view_quaternion_wxyz": np.asarray(
            [[1.0, 0.0, 0.0, 0.0]] * count, dtype=np.float64
        ),
    }


def _manifest(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    ledger = [[0, 0, 1], [0, 25, 0], [0, 50, 1], [0, 75, 1], [0, 100, 1]]
    membership = [
        [0, 0, 0, 0, 0],
        [0, 50, 0, 2 * 2048, 1],
        [0, 75, 0, 3 * 2048, 2],
        [0, 100, 0, 4 * 2048, 3],
    ]
    matrix = np.asarray(membership, dtype=np.int64)
    any_hash = "1" * 64
    return {
        "schema": h10.EXPECTED_SOURCE_SCHEMA,
        "mode": "sealed_raw_observer_source",
        "create_only": True,
        "association_applied": False,
        "tracking_enabled": False,
        "tracked_artifact_present": False,
        "coordinate_frame": "scannet_world",
        "coordinate_contract_sha256": "2" * 64,
        "scene_ids": ["scene_test"],
        "scene_count": 1,
        "exact_frame_count": 5,
        "raw_frame_count": 5,
        "raw_row_count": 4,
        "empty_frame_count": 1,
        "empty_frame_identities": [[0, 25]],
        "frame_row_ledger": ledger,
        "scene_row_counts": [4],
        "source_instance_id_rule": (
            "global_exact_schedule_index*2048+per_frame_source_row"
        ),
        "provider_bindings": {
            "schedule_sha256": h10.EXPECTED_SCHEDULE_SHA256,
            "run_provenance_sha256": "3" * 64,
            "final_seal_sha256": "4" * 64,
            "journal_sha256": "5" * 64,
            "provider_contract_sha256": "6" * 64,
            "frozen_assets_sha256": "7" * 64,
            "exact_input_ledger_sha256": "8" * 64,
            "code_hashes": {"runner": "9" * 64},
            "model_hashes": {"boxer": "a" * 64},
            "protocol_hashes": {"provider_contract": "b" * 64},
        },
        "input_identity": {
            "snapshot_entry_count": 1,
            "snapshot_sha256_before": any_hash,
            "snapshot_sha256_after": any_hash,
            "byte_identical": True,
        },
        "k8": {
            "top_k": 8,
            "sort_key": list(h10.K8_SORT_KEY),
            "identity_columns": list(h10.K8_COLUMNS),
            "membership_identities": membership,
            "membership_count": 4,
            "membership_per_scene": [4],
            "membership_sha256": h10._numeric_matrix_sha256(
                "k8_membership_identity", matrix
            ),
        },
        "array_names": sorted(h10.SOURCE_ARRAYS),
        "array_content_sha256": h10._array_content_sha256(arrays),
        "npz_file": h10.SOURCE_NPZ_NAME,
        "npz_sha256": "0" * 64,
    }


def _write_source(
    tmp_path: Path,
    monkeypatch,
    *,
    mutate_manifest=None,
    mutate_arrays=None,
) -> tuple[Path, Path, Path, dict[str, np.ndarray], dict[str, object]]:
    arrays = _raw_arrays()
    if mutate_arrays is not None:
        mutate_arrays(arrays)
    manifest = _manifest(arrays)
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    source = tmp_path / "source"
    source.mkdir()
    npz_payload = h10._deterministic_npz_bytes(arrays)
    manifest["npz_sha256"] = hashlib.sha256(npz_payload).hexdigest()
    json_payload = h10._canonical_json_bytes(manifest)
    (source / h10.SOURCE_NPZ_NAME).write_bytes(npz_payload)
    (source / h10.SOURCE_JSON_NAME).write_bytes(json_payload)
    contract = tmp_path / "receipt_contract.md"
    contract.write_bytes(b"synthetic no-GT receipt contract\n")
    output = tmp_path / "receipt"

    membership = np.asarray(
        manifest["k8"]["membership_identities"], dtype=np.int64
    ).reshape((-1, 5))
    monkeypatch.setattr(h10, "EXPECTED_SCENE_COUNT", 1)
    monkeypatch.setattr(h10, "EXPECTED_EXACT_FRAME_COUNT", 5)
    monkeypatch.setattr(h10, "EXPECTED_RAW_FRAME_COUNT", 5)
    monkeypatch.setattr(h10, "EXPECTED_SOURCE_JSON_SHA256", hashlib.sha256(json_payload).hexdigest())
    monkeypatch.setattr(h10, "EXPECTED_SOURCE_NPZ_SHA256", hashlib.sha256(npz_payload).hexdigest())
    monkeypatch.setattr(h10, "EXPECTED_SOURCE_ARRAY_CONTENT_SHA256", h10._array_content_sha256(arrays))
    monkeypatch.setattr(h10, "EXPECTED_K8_MEMBERSHIP_COUNT", len(membership))
    monkeypatch.setattr(
        h10,
        "EXPECTED_K8_MEMBERSHIP_SHA256",
        h10._numeric_matrix_sha256("k8_membership_identity", membership),
    )
    return source, contract, output, arrays, manifest


def _run(source: Path, contract: Path, output: Path, **kwargs):
    assert not kwargs
    return h10.run_h10_receipt_shadow(
        source_root=source,
        receipt_contract=contract,
        expected_receipt_contract_sha256=hashlib.sha256(contract.read_bytes()).hexdigest(),
        expected_runner_sha256=hashlib.sha256(Path(h10.__file__).read_bytes()).hexdigest(),
        expected_runner_test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        output_root=output,
    )


def test_final_sealer_and_tracker_bytes_are_pinned():
    assert h10._read_regular_bytes(
        h10.SOURCE_SEALER,
        max_bytes=h10.MAX_CODE_BYTES,
        label="sealer",
    )[1]["sha256"] == h10.EXPECTED_SOURCE_SEALER_SHA256
    assert h10._read_regular_bytes(
        h10.SOURCE_SEALER_TEST,
        max_bytes=h10.MAX_CODE_BYTES,
        label="sealer test",
    )[1]["sha256"] == h10.EXPECTED_SOURCE_SEALER_TEST_SHA256
    assert h10._read_regular_bytes(
        h10.TRACKER_SOURCE,
        max_bytes=h10.MAX_CODE_BYTES,
        label="tracker",
    )[1]["sha256"] == h10.EXPECTED_TRACKER_SHA256
    assert h10._read_regular_bytes(
        h10.TRACKER_TEST,
        max_bytes=h10.MAX_CODE_BYTES,
        label="tracker test",
    )[1]["sha256"] == h10.EXPECTED_TRACKER_TEST_SHA256


def test_formal_numeric_source_preflight_recomputes_exact_content_and_k8():
    # This is deliberately only a numeric source decode/validation preflight:
    # no tracker replay, output publication, native prediction, annotation,
    # evaluator, or GT surface is called.
    (
        manifest,
        arrays,
        scenes,
        frames_by_scene,
        membership,
        selections,
        membership_hash,
        snapshot,
    ) = h10._load_sealed_source(h10.DEFAULT_SOURCE_ROOT)
    assert manifest["schema"] == h10.EXPECTED_SOURCE_SCHEMA
    assert len(scenes) == 10
    assert sum(len(frames) for frames in frames_by_scene) == 769
    assert len(membership) == 4557
    assert sum(len(selection) for selection in selections) == 4557
    assert h10._array_content_sha256(arrays) == (
        "a5efdb8d0d2c7b95f63368a3249229659a1052c400539321ce461da32732b862"
    )
    assert membership_hash == (
        "a2a94b11461e8c1bdd15d6a4ad99d058f42db6fd73690c69269ff1b89deb6391"
    )
    assert snapshot["json"]["sha256"] == h10.EXPECTED_SOURCE_JSON_SHA256
    assert snapshot["npz"]["sha256"] == h10.EXPECTED_SOURCE_NPZ_SHA256
    assert all(value.flags.writeable is False for value in arrays.values())


def test_synthetic_end_to_end_replays_empty_frame_and_three_distinct_frames(
    tmp_path, monkeypatch
):
    source, contract, output, _, _ = _write_source(tmp_path, monkeypatch)
    source_before = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in (h10.SOURCE_JSON_NAME, h10.SOURCE_NPZ_NAME)
    }
    manifest = _run(source, contract, output)
    assert manifest["schema"] == h10.SCHEMA
    assert manifest["H10_shadow_complete"] is True
    assert manifest["gt_access"] is False
    assert manifest["gt_access_authorized"] is False
    assert manifest["ap_evaluation"] is False
    assert manifest["birth"] is False
    assert manifest["H10_oracle_authorized"] is False
    assert manifest["full100_not_authorized"] is True
    assert manifest["native_prediction_access"] is False
    assert manifest["selected_row_count"] == 4
    assert manifest["valid_frame_count"] == 5
    assert manifest["receipt_count"] == 1
    assert manifest["evidence_count"] == 3
    assert manifest["selection"]["membership_consumed_not_reselected"] is True
    assert manifest["input_hash_identity"] is True
    assert source_before == {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in (h10.SOURCE_JSON_NAME, h10.SOURCE_NPZ_NAME)
    }
    with np.load(output / h10.OUTPUT_NPZ_NAME, allow_pickle=False) as trace:
        np.testing.assert_array_equal(trace["schedule_frame_id"], [0, 25, 50, 75, 100])
        np.testing.assert_array_equal(trace["frame_selected_offsets"], [0, 1, 1, 2, 3, 4])
        np.testing.assert_array_equal(trace["evidence_frame_id"], [0, 50, 75])
        assert len(set(trace["evidence_frame_id"].tolist())) == 3
        assert trace["receipt_confirmation_frame_id"].tolist() == [75]


def test_no_directory_enumeration_pickle_or_native_access(tmp_path, monkeypatch):
    source, contract, output, _, _ = _write_source(tmp_path, monkeypatch)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("directory enumeration/pickle access is forbidden")

    real_listdir = os.listdir

    def guarded_listdir(path):
        # Exact-two-entry verification on a held output dirfd is required;
        # enumeration of any input path remains forbidden.
        if isinstance(path, int):
            return real_listdir(path)
        return forbidden(path)

    monkeypatch.setattr(os, "listdir", guarded_listdir)
    monkeypatch.setattr(os, "scandir", forbidden)
    monkeypatch.setattr(Path, "iterdir", forbidden)
    monkeypatch.setattr(Path, "glob", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)
    monkeypatch.setattr(pickle, "load", forbidden)
    monkeypatch.setattr(pickle, "loads", forbidden)
    manifest = _run(source, contract, output)
    assert manifest["pickle_deserialization"] is False
    assert manifest["source"]["only_sealed_numeric_source_consumed"] is True


def test_tracker_is_constructed_once_per_scene(tmp_path, monkeypatch):
    _, _, _, arrays, manifest = _write_source(tmp_path, monkeypatch)
    calls = []

    def factory():
        calls.append(1)
        return h10.S3RReceiptTracker()

    ledger, frames = h10._validate_frame_ledger(manifest, arrays, ("scene_test",))
    _, selections, _ = h10._verify_frozen_membership(
        manifest, arrays, ledger, 1
    )
    h10._run_tracking(
        scene_ids=("scene_test",),
        frames_by_scene=frames,
        source_arrays=arrays,
        selections=selections,
        tracker_factory=factory,
    )
    assert len(calls) == 1


def test_membership_is_verified_not_trusted(tmp_path, monkeypatch):
    def mutate(manifest):
        manifest["k8"]["membership_identities"][0][4] = 1
        matrix = np.asarray(manifest["k8"]["membership_identities"], dtype=np.int64)
        manifest["k8"]["membership_sha256"] = h10._numeric_matrix_sha256(
            "k8_membership_identity", matrix
        )

    source, contract, output, _, _ = _write_source(
        tmp_path, monkeypatch, mutate_manifest=mutate
    )
    with pytest.raises(h10.H10ReceiptShadowError, match="independent verification"):
        _run(source, contract, output)
    assert not output.exists()


def test_source_byte_hash_mismatch_fails_before_decode(tmp_path, monkeypatch):
    source, contract, output, _, _ = _write_source(tmp_path, monkeypatch)
    monkeypatch.setattr(h10, "EXPECTED_SOURCE_NPZ_SHA256", "f" * 64)
    called = False

    def decode(_payload):
        nonlocal called
        called = True
        raise AssertionError("must not decode hash-mismatched source")

    monkeypatch.setattr(h10, "_load_npz_bytes", decode)
    with pytest.raises(h10.H10ReceiptShadowError, match="NPZ differs"):
        _run(source, contract, output)
    assert called is False


def test_contract_is_hash_only_and_mismatch_fails(tmp_path, monkeypatch):
    source, contract, output, _, _ = _write_source(tmp_path, monkeypatch)
    with pytest.raises(h10.H10ReceiptShadowError, match="contract SHA-256 mismatch"):
        h10.run_h10_receipt_shadow(
            source_root=source,
            receipt_contract=contract,
            expected_receipt_contract_sha256="f" * 64,
            expected_runner_sha256=hashlib.sha256(Path(h10.__file__).read_bytes()).hexdigest(),
            expected_runner_test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            output_root=output,
        )
    assert not output.exists()


def test_create_only_refuses_existing_output(tmp_path, monkeypatch):
    source, contract, output, _, _ = _write_source(tmp_path, monkeypatch)
    output.mkdir()
    marker = output / "keep"
    marker.write_text("owned", encoding="utf-8")
    with pytest.raises(h10.H10ReceiptShadowError, match="refusing to overwrite"):
        _run(source, contract, output)
    assert marker.read_text(encoding="utf-8") == "owned"


def test_source_changed_after_replay_fails_without_publication(tmp_path, monkeypatch):
    source, contract, output, _, _ = _write_source(tmp_path, monkeypatch)
    original = h10._run_tracking

    def changing(**kwargs):
        result = original(**kwargs)
        with (source / h10.SOURCE_JSON_NAME).open("ab") as handle:
            handle.write(b" ")
        return result

    monkeypatch.setattr(h10, "_run_tracking", changing)
    with pytest.raises(h10.H10ReceiptShadowError, match="changed during receipt replay"):
        _run(source, contract, output)
    assert not output.exists()


def test_numeric_environment_fails_closed(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(h10.H10ReceiptShadowError, match="pinned exactly"):
        h10._validate_numeric_thread_environment()


@pytest.mark.parametrize("which", ["runner", "runner_test"])
def test_public_api_rejects_unbound_runner_or_test_hash(
    tmp_path, monkeypatch, which
):
    source, contract, output, _, _ = _write_source(tmp_path, monkeypatch)
    runner_hash = hashlib.sha256(Path(h10.__file__).read_bytes()).hexdigest()
    test_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if which == "runner":
        runner_hash = "f" * 64
    else:
        test_hash = "f" * 64
    with pytest.raises(h10.H10ReceiptShadowError, match="runner.*SHA-256 mismatch"):
        h10.run_h10_receipt_shadow(
            source_root=source,
            receipt_contract=contract,
            expected_receipt_contract_sha256=hashlib.sha256(contract.read_bytes()).hexdigest(),
            expected_runner_sha256=runner_hash,
            expected_runner_test_sha256=test_hash,
            output_root=output,
        )
    assert not output.exists()


def test_cpu_only_tracker_runtime_is_explicit():
    arrays = _raw_arrays()
    trace, summary = h10._run_tracking(
        scene_ids=("scene_test",),
        frames_by_scene=((0, 50, 75, 100),),
        source_arrays=arrays,
        selections=(np.arange(4, dtype=np.int64),),
    )
    assert len(trace["receipt_track_id"]) == 1
    runtime = summary["runtime"]
    assert runtime["tracker_execution_device"] == "cpu"
    assert runtime["tracker_gpu_execution"] is False
    assert runtime["tracker_cuda_api_access"] is False
    assert runtime["tracker_gpu_allocation_bytes"] == 0
    assert runtime["gpu_memory_measurement_claimed"] is False
    assert runtime["cpu_only_implementation"]["audit_method"] == (
        "static_AST_import_audit"
    )


def test_same_frame_rows_cannot_confirm_one_another():
    arrays = _raw_arrays()
    # Three identical rows in one frame create three independent tracks.
    arrays = {
        name: np.repeat(value[:1], 3, axis=0) if name != "scene_ids" else value
        for name, value in arrays.items()
    }
    arrays["per_view_source_row"] = np.arange(3, dtype=np.int64)
    arrays["per_view_source_instance_id"] = np.arange(3, dtype=np.int64)
    selection = (np.arange(3, dtype=np.int64),)
    trace, summary = h10._run_tracking(
        scene_ids=("scene_test",),
        frames_by_scene=((0,),),
        source_arrays=arrays,
        selections=selection,
    )
    assert summary["receipt_count"] == 0
    np.testing.assert_array_equal(trace["assignment_action"], [0, 0, 0])
    assert len(set(trace["assignment_track_id"].tolist())) == 3


def test_cli_requires_opaque_contract_binding():
    parser = h10._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--output-root", "/tmp/out"])


def _publication_fixture(tmp_path: Path):
    source = tmp_path / "sealed_source"
    source.mkdir()
    output = tmp_path / "parent" / "receipt"
    output.parent.mkdir()
    arrays = {"value": np.asarray([1, 2, 3], dtype=np.int64)}
    payload = h10._deterministic_npz_bytes(arrays)
    manifest = {
        "audit_complete": True,
        "cap_event_count": 0,
        "npz_sha256": hashlib.sha256(payload).hexdigest(),
        "trace_array_content_sha256": h10._array_content_sha256(arrays),
    }
    return source, output, arrays, payload, manifest


def test_publish_fails_closed_on_output_parent_path_swap(tmp_path, monkeypatch):
    source, output, arrays, payload, manifest = _publication_fixture(tmp_path)
    real_write = h10._write_exclusive_fsync_at
    calls = 0

    def swapping_write(directory_fd, name, data):
        nonlocal calls
        real_write(directory_fd, name, data)
        calls += 1
        if calls == 2:
            displaced = output.parent.with_name("parent_displaced")
            os.rename(output.parent, displaced)
            output.parent.mkdir()

    monkeypatch.setattr(h10, "_write_exclusive_fsync_at", swapping_write)
    with pytest.raises(h10.H10ReceiptShadowError, match="output parent identity changed"):
        h10._publish_create_only(
            output_root=output,
            source_root=source,
            arrays=arrays,
            manifest=manifest,
            npz_payload=payload,
        )
    assert not output.exists()


def test_publish_detects_staging_stat_to_open_swap(tmp_path, monkeypatch):
    source, output, arrays, payload, manifest = _publication_fixture(tmp_path)
    real_open = os.open
    attacked = False

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal attacked
        if (
            not attacked
            and isinstance(path, str)
            and path.startswith(f".{output.name}.stage.")
            and dir_fd is not None
        ):
            attacked = True
            os.rename(path, f"{path}.held", src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
            os.mkdir(path, mode=0o700, dir_fd=dir_fd)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swapping_open)
    with pytest.raises(h10.H10ReceiptShadowError, match="staging identity changed while opening"):
        h10._publish_create_only(
            output_root=output,
            source_root=source,
            arrays=arrays,
            manifest=manifest,
            npz_payload=payload,
        )
    assert attacked is True
    assert not output.exists()


def test_publish_detects_staging_replacement_immediately_before_rename(
    tmp_path, monkeypatch
):
    source, output, arrays, payload, manifest = _publication_fixture(tmp_path)
    real_rename = h10._rename_noreplace

    def replacing_rename(source_fd, source_name, destination_fd, destination_name):
        os.rename(
            source_name,
            f"{source_name}.stolen",
            src_dir_fd=source_fd,
            dst_dir_fd=source_fd,
        )
        os.mkdir(source_name, mode=0o700, dir_fd=source_fd)
        real_rename(source_fd, source_name, destination_fd, destination_name)

    monkeypatch.setattr(h10, "_rename_noreplace", replacing_rename)
    with pytest.raises(h10.H10ReceiptShadowError, match="published output directory identity differs"):
        h10._publish_create_only(
            output_root=output,
            source_root=source,
            arrays=arrays,
            manifest=manifest,
            npz_payload=payload,
        )


def test_publish_detects_output_replacement_immediately_after_rename(
    tmp_path, monkeypatch
):
    source, output, arrays, payload, manifest = _publication_fixture(tmp_path)
    real_rename = h10._rename_noreplace

    def replacing_after(source_fd, source_name, destination_fd, destination_name):
        real_rename(source_fd, source_name, destination_fd, destination_name)
        os.rename(
            destination_name,
            f"{destination_name}.stolen",
            src_dir_fd=destination_fd,
            dst_dir_fd=destination_fd,
        )
        os.mkdir(destination_name, mode=0o700, dir_fd=destination_fd)

    monkeypatch.setattr(h10, "_rename_noreplace", replacing_after)
    with pytest.raises(h10.H10ReceiptShadowError, match="published output directory identity differs"):
        h10._publish_create_only(
            output_root=output,
            source_root=source,
            arrays=arrays,
            manifest=manifest,
            npz_payload=payload,
        )
