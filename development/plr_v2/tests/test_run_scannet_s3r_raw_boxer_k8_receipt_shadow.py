from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

import tools.run_scannet_s3r_raw_boxer_k8_receipt_shadow as s3r_run


@pytest.fixture(autouse=True)
def _pin_numeric_thread_environment(monkeypatch):
    for name, value in s3r_run.REQUIRED_NUMERIC_THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)


def _synthetic_source_and_schedule():
    # scene0 confirms on its first three frames and receives a fourth update;
    # scene1 has an empty valid frame between its two observations; scene2 is
    # entirely empty.  All rows are already in exact K8 order.
    scene_index = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int16)
    frame_id = np.asarray([0, 25, 50, 75, 0, 50], dtype=np.int64)
    count = len(frame_id)
    arrays = {
        "per_view_scene_index": scene_index,
        "per_view_frame_id": frame_id,
        "per_view_source_row": np.arange(10, 10 + count, dtype=np.int32),
        "per_view_source_instance_id": np.arange(20, 20 + count, dtype=np.int32),
        "per_view_source_score": np.asarray(
            [0.9, 0.8, 0.7, 0.6, 0.95, 0.75], dtype=np.float32
        ),
        "per_view_center_world": np.asarray(
            [[0.0, 0.0, 2.0]] * 4 + [[5.0, 0.0, 2.0]] * 2,
            dtype=np.float32,
        ),
        "per_view_extent_xyz": np.ones((count, 3), dtype=np.float32),
        "per_view_quaternion_wxyz": np.asarray(
            [[1.0, 0.0, 0.0, 0.0]] * count, dtype=np.float32
        ),
    }
    selections = (
        np.asarray([0, 1, 2, 3], dtype=np.int64),
        np.asarray([4, 5], dtype=np.int64),
        np.empty((0,), dtype=np.int64),
    )
    schedules = {
        "scene0568_00": {
            "valid_frame_ids": (0, 25, 50, 75),
            "candidate_frame_ids": (0, 25, 50, 75),
            "empty_valid_frame_ids": (),
            "sha256": "a" * 64,
        },
        "scene0606_01": {
            "valid_frame_ids": (0, 25, 50),
            "candidate_frame_ids": (0, 50),
            "empty_valid_frame_ids": (25,),
            "sha256": "b" * 64,
        },
        "scene0377_02": {
            "valid_frame_ids": (0,),
            "candidate_frame_ids": (),
            "empty_valid_frame_ids": (0,),
            "sha256": "c" * 64,
        },
    }
    return arrays, selections, schedules


def _publish_fixture():
    arrays = {
        "scene_ids": np.asarray(["scene0568_00"], dtype="<U12"),
        "value": np.asarray([1, 2, 3], dtype=np.int32),
    }
    payload = s3r_run._deterministic_npz_bytes(arrays)
    manifest = {
        "schema": s3r_run.SCHEMA,
        "audit_complete": True,
        "cap_event_count": 0,
        "runtime": {
            "tracker_cpu_budget_pass": True,
            "tracker_memory_upper_bound_pass": True,
            "resource_budget_pass": True,
            "numeric_thread_environment": dict(
                s3r_run.REQUIRED_NUMERIC_THREAD_ENVIRONMENT
            ),
            "numeric_thread_environment_pinned": True,
        },
        "npz_file": s3r_run.OUTPUT_NPZ_NAME,
        "npz_sha256": hashlib.sha256(payload).hexdigest(),
        "candidate_content_sha256": s3r_run._array_content_sha256(arrays),
    }
    return arrays, payload, manifest


def _reverse_confirmation_source_and_schedule():
    # Tracks 0 and 1 are born together.  Track 1 receives its third frame at
    # frame 50, while lower-ID track 0 confirms only at frame 100.
    scene_index = np.zeros(6, dtype=np.int16)
    frame_id = np.asarray([0, 0, 25, 50, 75, 100], dtype=np.int64)
    center = np.asarray(
        [
            [0.0, 0.0, 2.0],
            [5.0, 0.0, 2.0],
            [5.0, 0.0, 2.0],
            [5.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )
    arrays = {
        "per_view_scene_index": scene_index,
        "per_view_frame_id": frame_id,
        "per_view_source_row": np.arange(100, 106, dtype=np.int32),
        "per_view_source_instance_id": np.arange(200, 206, dtype=np.int32),
        "per_view_source_score": np.linspace(0.9, 0.4, 6, dtype=np.float32),
        "per_view_center_world": center,
        "per_view_extent_xyz": np.ones((6, 3), dtype=np.float32),
        "per_view_quaternion_wxyz": np.tile(
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (6, 1)
        ),
    }
    selections = (
        np.arange(6, dtype=np.int64),
        np.empty((0,), dtype=np.int64),
        np.empty((0,), dtype=np.int64),
    )
    schedules = {
        "scene0568_00": {
            "valid_frame_ids": (0, 25, 50, 75, 100),
            "candidate_frame_ids": (0, 25, 50, 75, 100),
            "empty_valid_frame_ids": (),
            "sha256": "a" * 64,
        },
        "scene0606_01": {
            "valid_frame_ids": (0,),
            "candidate_frame_ids": (),
            "empty_valid_frame_ids": (0,),
            "sha256": "b" * 64,
        },
        "scene0377_02": {
            "valid_frame_ids": (0,),
            "candidate_frame_ids": (),
            "empty_valid_frame_ids": (0,),
            "sha256": "c" * 64,
        },
    }
    return arrays, selections, schedules


def test_frozen_prereg_tracker_and_tracker_tests_are_exact_final_bytes():
    assert s3r_run._sha256(s3r_run.PREREGISTRATION) == (
        "14f29a50dd65ee791be2df519e0000cf22bfc94a0209880f3539159acf4f7df3"
    )
    assert s3r_run._sha256(s3r_run.TRACKER_SOURCE) == (
        "277316c36b7a7fcb8005a24e907e0f232e41f6b5874411293eb26b0744df9628"
    )
    assert s3r_run._sha256(s3r_run.TRACKER_TEST) == (
        "f08fd59ee2888c936e5b783de668fd789ba6b676bc4864e001b000ea287b1e3c"
    )
    assert "117c289e" not in Path(s3r_run.__file__).read_text(encoding="utf-8")
    assert "e8325ed7" not in Path(s3r_run.__file__).read_text(encoding="utf-8")


def test_numeric_thread_environment_is_exact_and_fails_closed(monkeypatch):
    assert s3r_run._validate_numeric_thread_environment() == dict(
        s3r_run.REQUIRED_NUMERIC_THREAD_ENVIRONMENT
    )
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    with pytest.raises(s3r_run.S3RShadowError, match="pinned exactly to one thread"):
        s3r_run._validate_numeric_thread_environment()
    monkeypatch.delenv("OPENBLAS_NUM_THREADS")
    with pytest.raises(s3r_run.S3RShadowError, match="pinned exactly to one thread"):
        s3r_run._validate_numeric_thread_environment()


def test_real_frozen_source_k8_counts_selection_and_empty_schedule_are_exact():
    ledger = s3r_run._validate_fixed_assets()
    assert ledger["topk_receipt"]["sha256"] == (
        "d4ba67b37d362842333ac525abe32f6807c4fba90af83b699bbfc1494aa5ea1f"
    )
    manifest, arrays = s3r_run._load_sealed_candidates()
    selections, digest = s3r_run._select_k8(arrays)
    assert digest == s3r_run.EXPECTED_SELECTION_SHA256
    assert [len(value) for value in selections] == [501, 854, 216]
    schedules = s3r_run._load_schedules(manifest, arrays)
    assert [len(schedules[scene]["valid_frame_ids"]) for scene in s3r_run.DEV3_SCENES] == [
        66,
        112,
        30,
    ]
    assert schedules["scene0606_01"]["invalid_pose_frame_ids"] == (1325,)
    assert schedules["scene0606_01"]["empty_valid_frame_ids"] == (1300, 1350)
    # Tracked/native-like numeric arrays exist in the frozen ZIP schema, but
    # the S3R loader deliberately does not decode or return them.
    assert set(arrays) == set(s3r_run.ALLOWED_SOURCE_ARRAYS)
    assert not any(name.startswith("tracked_") for name in arrays)
    assert all(array.flags.writeable is False for array in arrays.values())


def test_hamilton_golden_corner_sign_order_scale_and_direction_are_exact():
    identity = s3r_run._obb_corners(
        np.asarray([10.0, 20.0, 30.0]),
        np.asarray([2.0, 4.0, 6.0]),
        np.asarray([2.0, 0.0, 0.0, 0.0]),  # deliberately non-unit
    )
    expected_identity = s3r_run.SIGNS * [1.0, 2.0, 3.0] + [10.0, 20.0, 30.0]
    np.testing.assert_array_equal(identity, expected_identity)

    # Positive Hamilton z-rotation maps local (x,y,z) -> (-y,x,z).  Scaling
    # the quaternion must not alter the result because the formula uses 2/n.
    q = 3.0 * np.asarray([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    rotated = s3r_run._obb_corners(
        np.asarray([10.0, 20.0, 30.0]), np.asarray([2.0, 4.0, 6.0]), q
    )
    local = s3r_run.SIGNS * [1.0, 2.0, 3.0]
    expected_rotated = np.column_stack((-local[:, 1], local[:, 0], local[:, 2]))
    expected_rotated += [10.0, 20.0, 30.0]
    np.testing.assert_allclose(rotated, expected_rotated, rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(
        s3r_run._obb_corners([10, 20, 30], [2, 4, 6], -q),
        rotated,
        rtol=0.0,
        atol=0.0,
    )

    with pytest.raises(s3r_run.S3RShadowError):
        s3r_run._obb_corners([0, 0, 0], [1, 1, 1], [0, 0, 0, 0])
    with pytest.raises(s3r_run.S3RShadowError):
        s3r_run._obb_corners([0, 0, 0], [1, 0, 1], [1, 0, 0, 0])


def test_synthetic_exact_query_commit_empty_frames_and_complete_trace():
    source, selections, schedules = _synthetic_source_and_schedule()
    arrays, summary = s3r_run._run_tracking(
        source_arrays=source, selections=selections, schedules=schedules
    )
    assert summary["audit_complete"] is True
    assert summary["cap_event_count"] == 0
    assert summary["selected_row_count"] == 6
    assert summary["assignment_count"] == 6
    assert summary["valid_frame_count"] == 8
    assert summary["receipt_count"] == 1
    assert summary["evidence_count"] == 3
    assert summary["tracker_summaries"]["scene0606_01"]["empty_keyframes"] == 1
    np.testing.assert_array_equal(
        arrays["schedule_frame_id"], [0, 25, 50, 75, 0, 25, 50, 0]
    )
    np.testing.assert_array_equal(
        arrays["frame_selected_offsets"], [0, 1, 2, 3, 4, 5, 5, 6, 6]
    )
    np.testing.assert_array_equal(arrays["selected_sealed_npz_row"], [0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(arrays["selected_rank_in_frame"], [0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(arrays["assignment_action"], [0, 1, 1, 1, 0, 1])
    np.testing.assert_array_equal(arrays["evidence_selected_index"], [0, 1, 2])
    np.testing.assert_array_equal(arrays["evidence_frame_id"], [0, 25, 50])
    assert arrays["receipt_medoid_evidence_index"].tolist() == [0]
    np.testing.assert_array_equal(
        arrays["receipt_corners_world"][0], arrays["evidence_corners_world"][0]
    )
    assert arrays["frame_cap_event_count"].sum() == 0
    assert arrays["frame_audit_complete"].all()


def test_receipt_export_is_track_id_stable_when_confirmation_order_is_reversed():
    source, selections, schedules = _reverse_confirmation_source_and_schedule()
    arrays, summary = s3r_run._run_tracking(
        source_arrays=source, selections=selections, schedules=schedules
    )
    assert summary["audit_complete"] is True
    assert summary["receipt_count"] == 2
    # Confirmation chronology is track 1 then track 0, visible in frame data.
    assert arrays["frame_new_receipt_count"].tolist() == [0, 0, 1, 0, 1, 0, 0]
    # The sealed receipt table is deterministic scene/track order.
    assert arrays["receipt_track_id"].tolist() == [0, 1]
    assert arrays["receipt_confirmation_frame_id"].tolist() == [100, 50]
    assert arrays["evidence_offsets"].tolist() == [0, 3, 6]


def test_trace_schema_has_complete_provenance_and_no_forbidden_payload_arrays():
    source, selections, schedules = _synthetic_source_and_schedule()
    arrays, _summary = s3r_run._run_tracking(
        source_arrays=source, selections=selections, schedules=schedules
    )
    required = {
        "selected_center_world",
        "selected_extent_xyz",
        "selected_quaternion_wxyz",
        "selected_corners_world",
        "assignment_track_id",
        "assignment_action",
        "assignment_aabb_iou",
        "assignment_center_distance_m",
        "receipt_corners_world",
        "receipt_medoid_evidence_index",
        "receipt_pairwise_aabb_iou",
        "evidence_offsets",
        "evidence_selected_index",
        "evidence_corners_world",
        "retired_track_id",
    }
    assert required.issubset(arrays)
    forbidden = ("label", "class", "clip", "depth", "native", "oracle", "match_gt")
    assert not any(token in name.lower() for name in arrays for token in forbidden)
    assert not any(name.lower().startswith("ap_") for name in arrays)
    assert arrays["selected_corners_world"].dtype == np.float64
    assert arrays["selected_corners_world"].shape == (6, 8, 3)
    assert arrays["evidence_offsets"].tolist() == [0, 3]


def test_deterministic_npz_bytes_and_content_hash_roundtrip(tmp_path):
    arrays = {
        "z": np.asarray([3.0, 4.0], dtype=np.float64),
        "a": np.asarray([1, 2], dtype=np.int16),
    }
    first = s3r_run._deterministic_npz_bytes(arrays)
    second = s3r_run._deterministic_npz_bytes(dict(reversed(tuple(arrays.items()))))
    assert first == second
    path = tmp_path / "trace.npz"
    path.write_bytes(first)
    with np.load(path, allow_pickle=False) as source:
        loaded = {name: np.array(source[name], copy=True) for name in source.files}
    assert s3r_run._array_content_sha256(loaded) == s3r_run._array_content_sha256(arrays)


def test_manifest_binds_complete_trace_runtime_inputs_and_native_identity():
    source, selections, schedules = _synthetic_source_and_schedule()
    arrays, summary = s3r_run._run_tracking(
        source_arrays=source, selections=selections, schedules=schedules
    )
    payload = s3r_run._deterministic_npz_bytes(arrays)
    native = {
        scene: {
            "path": f"/frozen/{scene}_boxes.pkl",
            "sha256": s3r_run.EXPECTED_FORMAL_T05_SHA256[scene],
            "bytes": 1,
        }
        for scene in s3r_run.DEV3_SCENES
    }
    frozen_inputs = {"opaque": "same"}
    manifest = s3r_run._build_manifest(
        arrays=arrays,
        summary=summary,
        selection_sha256=s3r_run.EXPECTED_SELECTION_SHA256,
        schedules=schedules,
        input_before=frozen_inputs,
        input_after=dict(frozen_inputs),
        native_before=native,
        native_after=native,
        npz_sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert manifest["schema"] == s3r_run.SCHEMA
    assert manifest["audit_complete"] is True
    assert manifest["output_inert"] is True
    assert manifest["past_only"] is True
    assert manifest["gt_access"] is False
    assert manifest["ap_evaluation"] is False
    assert manifest["birth"] is False
    assert manifest["H10_not_authorized"] is True
    assert manifest["C87_not_authorized"] is True
    assert manifest["full100_not_authorized"] is True
    assert manifest["contracts"]["tracker_source_sha256"].startswith("277316c3")
    assert manifest["selection"]["selection_sha256"] == s3r_run.EXPECTED_SELECTION_SHA256
    assert manifest["native_prediction_hash_identity"] is True
    assert manifest["candidate_content_sha256"] == s3r_run._array_content_sha256(arrays)
    assert manifest["runtime"]["integrated_provider_runtime_qualified"] is False
    assert manifest["runtime"]["numeric_thread_environment"] == dict(
        s3r_run.REQUIRED_NUMERIC_THREAD_ENVIRONMENT
    )
    # The real seal must be strict JSON: no NaN, bytes, or non-string map keys.
    assert s3r_run._json_bytes(manifest).endswith(b"\n")

    with pytest.raises(s3r_run.S3RShadowError, match="frozen inputs changed"):
        s3r_run._build_manifest(
            arrays=arrays,
            summary=summary,
            selection_sha256=s3r_run.EXPECTED_SELECTION_SHA256,
            schedules=schedules,
            input_before={"opaque": "before"},
            input_after={"opaque": "after"},
            native_before=native,
            native_after=native,
            npz_sha256=hashlib.sha256(payload).hexdigest(),
        )

    forged_summary = dict(summary)
    forged_summary["runtime"] = dict(summary["runtime"])
    forged_summary["runtime"]["tracker_cpu_budget_pass"] = False
    forged_summary["runtime"]["resource_budget_pass"] = False
    forged_summary["audit_complete"] = True
    with pytest.raises(s3r_run.S3RShadowError, match="resource-budget failure"):
        s3r_run._build_manifest(
            arrays=arrays,
            summary=forged_summary,
            selection_sha256=s3r_run.EXPECTED_SELECTION_SHA256,
            schedules=schedules,
            input_before=frozen_inputs,
            input_after=dict(frozen_inputs),
            native_before=native,
            native_after=native,
            npz_sha256=hashlib.sha256(payload).hexdigest(),
        )


def test_atomic_publish_is_create_only_and_existing_empty_dir_is_preserved(tmp_path):
    arrays, payload, manifest = _publish_fixture()
    output = tmp_path / "sealed"
    s3r_run._publish_create_only(
        output_root=output, arrays=arrays, manifest=manifest, npz_payload=payload
    )
    before_npz = s3r_run._sha256(output / s3r_run.OUTPUT_NPZ_NAME)
    with pytest.raises(s3r_run.S3RShadowError, match="refusing to overwrite"):
        s3r_run._publish_create_only(
            output_root=output, arrays=arrays, manifest=manifest, npz_payload=payload
        )
    assert s3r_run._sha256(output / s3r_run.OUTPUT_NPZ_NAME) == before_npz

    empty = tmp_path / "already_empty"
    empty.mkdir()
    with pytest.raises(s3r_run.S3RShadowError, match="refusing to overwrite"):
        s3r_run._publish_create_only(
            output_root=empty, arrays=arrays, manifest=manifest, npz_payload=payload
        )
    assert empty.is_dir() and list(empty.iterdir()) == []


def test_atomic_publish_race_uses_kernel_noreplace_and_leaves_competing_root(
    tmp_path, monkeypatch
):
    arrays, payload, manifest = _publish_fixture()
    output = tmp_path / "race"
    real_rename = s3r_run._rename_noreplace

    def competing_rename(source, destination):
        destination.mkdir()
        real_rename(source, destination)

    monkeypatch.setattr(s3r_run, "_rename_noreplace", competing_rename)
    with pytest.raises(s3r_run.S3RShadowError, match="refusing to overwrite"):
        s3r_run._publish_create_only(
            output_root=output, arrays=arrays, manifest=manifest, npz_payload=payload
        )
    assert output.is_dir() and list(output.iterdir()) == []
    assert not list(tmp_path.glob(".race.stage.*"))


def test_symlink_output_and_renameat2_enosys_fail_closed(tmp_path, monkeypatch):
    arrays, payload, manifest = _publish_fixture()
    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "link"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(s3r_run.S3RShadowError, match="refusing to overwrite"):
        s3r_run._publish_create_only(
            output_root=symlink, arrays=arrays, manifest=manifest, npz_payload=payload
        )

    class _FakeRename:
        argtypes = None
        restype = None

        def __call__(self, *_args):
            s3r_run.ctypes.set_errno(errno.ENOSYS)
            return -1

    class _FakeLibc:
        renameat2 = _FakeRename()

    monkeypatch.setattr(s3r_run.ctypes, "CDLL", lambda *_args, **_kwargs: _FakeLibc())
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "destination"
    with pytest.raises(s3r_run.S3RShadowError, match="unsupported"):
        s3r_run._rename_noreplace(source, destination)
    assert source.is_dir() and not destination.exists()


def test_cap_or_incomplete_trace_cannot_publish_valid_schema(tmp_path):
    arrays, payload, manifest = _publish_fixture()
    invalid = dict(manifest)
    invalid["audit_complete"] = False
    invalid["cap_event_count"] = 1
    output = tmp_path / "invalid"
    with pytest.raises(s3r_run.S3RShadowError, match="incomplete or capped"):
        s3r_run._publish_create_only(
            output_root=output, arrays=arrays, manifest=invalid, npz_payload=payload
        )
    assert not os.path.lexists(output)


def test_resource_budget_failure_cannot_publish_valid_schema(tmp_path):
    arrays, payload, manifest = _publish_fixture()
    invalid = dict(manifest)
    invalid["runtime"] = dict(manifest["runtime"])
    invalid["runtime"]["tracker_cpu_budget_pass"] = False
    invalid["runtime"]["resource_budget_pass"] = False
    output = tmp_path / "runtime_invalid"
    with pytest.raises(s3r_run.S3RShadowError, match="resource-budget failure"):
        s3r_run._publish_create_only(
            output_root=output, arrays=arrays, manifest=invalid, npz_payload=payload
        )
    assert not os.path.lexists(output)


@pytest.mark.parametrize(
    ("p95_limit_ns", "max_limit_ns"),
    ((-1, 10**18), (10**18, -1)),
)
def test_cpu_p95_or_max_budget_failure_invalidates_audit(
    monkeypatch, p95_limit_ns, max_limit_ns
):
    source, selections, schedules = _synthetic_source_and_schedule()
    monkeypatch.setattr(s3r_run, "TRACKER_CPU_P95_LIMIT_NS", p95_limit_ns)
    monkeypatch.setattr(s3r_run, "TRACKER_CPU_MAX_LIMIT_NS", max_limit_ns)
    _arrays, summary = s3r_run._run_tracking(
        source_arrays=source, selections=selections, schedules=schedules
    )
    assert summary["runtime"]["tracker_cpu_budget_pass"] is False
    assert summary["runtime"]["resource_budget_pass"] is False
    assert summary["audit_complete"] is False


def test_memory_over_budget_or_unmeasurable_invalidates_audit(monkeypatch):
    source, selections, schedules = _synthetic_source_and_schedule()

    calls = 0

    def over_budget_rss():
        nonlocal calls
        calls += 1
        if calls == 1:
            return 0
        return s3r_run.MAX_TRACKER_INCREMENTAL_MEMORY_BYTES + 1

    monkeypatch.setattr(s3r_run, "_rss_bytes", over_budget_rss)
    _arrays, over_summary = s3r_run._run_tracking(
        source_arrays=source, selections=selections, schedules=schedules
    )
    assert over_summary["runtime"]["rss_measurement_complete"] is True
    assert over_summary["runtime"]["tracker_memory_upper_bound_pass"] is False
    assert over_summary["runtime"]["resource_budget_pass"] is False
    assert over_summary["audit_complete"] is False

    monkeypatch.setattr(s3r_run, "_rss_bytes", lambda: -1)
    _arrays, unknown_summary = s3r_run._run_tracking(
        source_arrays=source, selections=selections, schedules=schedules
    )
    assert unknown_summary["runtime"]["rss_measurement_complete"] is False
    assert unknown_summary["runtime"]["runner_incremental_rss_upper_bound_bytes"] == -1
    assert unknown_summary["runtime"]["tracker_memory_upper_bound_pass"] is False
    assert unknown_summary["runtime"]["resource_budget_pass"] is False
    assert unknown_summary["audit_complete"] is False


def test_trace_byte_cap_fails_before_a_valid_result(monkeypatch):
    source, selections, schedules = _synthetic_source_and_schedule()
    monkeypatch.setattr(s3r_run, "MAX_TRACE_UNCOMPRESSED_BYTES", 1)
    with pytest.raises(s3r_run.S3RShadowError, match="trace cap"):
        s3r_run._run_tracking(
            source_arrays=source, selections=selections, schedules=schedules
        )


def test_native_identity_is_direct_hash_only_and_formal_root_is_fixed():
    native = s3r_run._hash_formal_t05_predictions()
    assert tuple(native) == s3r_run.DEV3_SCENES
    assert {scene: native[scene]["sha256"] for scene in s3r_run.DEV3_SCENES} == dict(
        s3r_run.EXPECTED_FORMAL_T05_SHA256
    )
    assert {
        Path(native[scene]["path"]).parent.resolve() for scene in s3r_run.DEV3_SCENES
    } == {s3r_run.FORMAL_T05_ROOT.resolve()}


def test_old_or_wrong_formal_root_and_hashes_fail_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(s3r_run, "FORMAL_T05_ROOT", tmp_path)
    with pytest.raises(s3r_run.S3RShadowError, match="formal T05 root mismatch"):
        s3r_run._hash_formal_t05_predictions()


def test_cli_and_import_surface_have_no_target_or_active_inputs():
    options = {
        option
        for action in s3r_run._build_parser()._actions
        for option in action.option_strings
    }
    assert options == {"-h", "--help", "--output-root"}
    source = Path(s3r_run.__file__).read_text(encoding="utf-8")
    forbidden_imports = (
        "from evaluation",
        "import evaluation",
        "from tools.audit_scannet",
        "import tools.audit_scannet",
        "import pickle",
        "from pickle",
        "import cv2",
        "import torch",
        "BoxManager",
    )
    assert not any(token in source for token in forbidden_imports)
    assert "_read_json(TOPK_RECEIPT" not in source
    assert "renameat2" in source and "_RENAME_NOREPLACE = 1" in source
    assert "engineering-smoke" not in source
