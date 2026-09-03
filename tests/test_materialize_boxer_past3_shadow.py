import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from tools.materialize_boxer_past3_shadow import (
    SCHEMA,
    ShadowError,
    materialize_boxer_past3_shadow,
)
from tools.audit_scannet_boxer_past3_oracle import (
    audit_scannet_boxer_past3_oracle,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    scene = "scene0000_00"
    input_dir = tmp_path / "input"
    baseline = tmp_path / "baseline"
    schedules = tmp_path / "schedules"
    rgbd = tmp_path / "rgbd"
    input_dir.mkdir()
    baseline.mkdir()
    (schedules / scene).mkdir(parents=True)
    pose_dir = rgbd / scene / "frames" / "pose"
    pose_dir.mkdir(parents=True)

    frames = [0, 25, 50, 75]
    schedule = {
        "schema": "boxfusion.cutr_postfilter_cache.v3",
        "scene_id": scene,
        "namespace": "synthetic-past3-test",
        "record_count": len(frames),
        "recorded_frame_ids": frames,
    }
    schedule_path = schedules / scene / "manifest.json"
    schedule_path.write_text(json.dumps(schedule), encoding="utf-8")

    camera_x = {0: -0.5, 25: 0.0, 50: 0.5, 75: 0.75}
    for frame_id in frames:
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = camera_x[frame_id]
        np.savetxt(pose_dir / f"{frame_id}.txt", pose)

    centers = np.asarray(
        [[0.00, 0.00, 2.00], [0.02, 0.00, 2.00], [-0.01, 0.01, 2.00]],
        dtype=np.float32,
    )
    arrays = {
        "scene_ids": np.asarray([scene], dtype="<U12"),
        "per_view_scene_index": np.zeros(3, dtype=np.int16),
        "per_view_frame_id": np.asarray([0, 25, 50], dtype=np.int64),
        "per_view_source_row": np.asarray([0, 1, 2], dtype=np.int32),
        "per_view_source_instance_id": np.asarray([0, 0, 0], dtype=np.int32),
        "per_view_center_world": centers,
        "per_view_quaternion_wxyz": np.tile(
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (3, 1)
        ),
        "per_view_extent_xyz": np.ones((3, 3), dtype=np.float32),
        "per_view_source_score": np.asarray([0.7, 0.65, 0.6], dtype=np.float32),
        "tracked_scene_index": np.empty((0,), dtype=np.int16),
        "tracked_source_row": np.empty((0,), dtype=np.int32),
        "tracked_instance_id": np.empty((0,), dtype=np.int32),
        "tracked_center_world": np.empty((0, 3), dtype=np.float32),
        "tracked_quaternion_wxyz": np.empty((0, 4), dtype=np.float32),
        "tracked_extent_xyz": np.empty((0, 3), dtype=np.float32),
        "tracked_source_score": np.empty((0,), dtype=np.float32),
    }
    input_npz = input_dir / "boxer_shadow_candidates.npz"
    np.savez(input_npz, **arrays)
    input_json = input_dir / "boxer_shadow_candidates.json"
    manifest = {
        "schema": "boxfusion.owl_boxer_shadow_candidates.v1",
        "profile": "clean_in2",
        "mode": "shadow",
        "output_inert": True,
        "birth": False,
        "gt_access": False,
        "gt_access_guard_verified": True,
        "native_before_after_identity": True,
        "native_clip_unchanged": True,
        "semantic_source_exported": False,
        "coordinate_frame": "scannet_world",
        "npz_file": input_npz.name,
        "npz_sha256": _sha256(input_npz),
        "candidate_content_sha256": "synthetic",
        "scene_count": 1,
        "per_view_candidate_count": 3,
        "scenes": [
            {
                "scene_id": scene,
                "scene_index": 0,
                "gt_access_guard_verified": True,
                "per_view_extra_schedule_rows_excluded": 0,
                "sealed_schedule_manifest_sha256": _sha256(schedule_path),
                "sealed_schedule_mode": "valid_recorded_frames",
                "sealed_schedule_invalid_pose_frame_ids_excluded": [],
                "sealed_schedule_frame_count": 4,
            }
        ],
    }
    input_json.write_text(json.dumps(manifest), encoding="utf-8")
    with (baseline / f"{scene}_boxes.pkl").open("wb") as handle:
        pickle.dump([[]], handle)
    prereg = tmp_path / "preregistered.md"
    prereg.write_text("frozen before oracle\n", encoding="utf-8")
    return {
        "input_json": input_json,
        "input_npz": input_npz,
        "baseline_root": baseline,
        "schedule_root": schedules,
        "scene_rgbd_root": rgbd,
        "preregistration": prereg,
    }


def _run(inputs: dict[str, Path], output_dir: Path):
    output_dir.mkdir()
    return materialize_boxer_past3_shadow(
        **inputs,
        output_json=output_dir / "boxer_past3_shadow.json",
        output_npz=output_dir / "boxer_past3_shadow.npz",
    )


def test_three_distinct_views_produce_one_output_inert_candidate(tmp_path):
    inputs = _fixture(tmp_path)
    baseline_path = inputs["baseline_root"] / "scene0000_00_boxes.pkl"
    before = _sha256(baseline_path)
    report = _run(inputs, tmp_path / "output")

    assert report["schema"] == SCHEMA
    assert report["candidate_count"] == 1
    assert report["gt_access"] is False
    assert report["birth"] is False
    assert report["native_before_after_identity"] is True
    assert _sha256(baseline_path) == before
    scene = report["scenes"]["scene0000_00"]
    assert scene["processed_keyframes"] == 4
    assert scene["nonempty_candidate_keyframes"] == 3
    assert scene["zero_candidate_keyframes"] == 1
    assert scene["view_gate_accepted_candidates"] == 1
    row = scene["accepted_candidates"][0]
    assert row["confirmation_frame_id"] == 50
    assert row["evidence_frame_ids"] == [0, 25, 50]
    assert row["max_camera_baseline_m"] == pytest.approx(1.0)
    assert row["max_view_ray_span_deg"] > 10.0

    with np.load(tmp_path / "output" / "boxer_past3_shadow.npz") as arrays:
        assert arrays["candidate_corners_world"].shape == (1, 8, 3)
        assert arrays["candidate_evidence_offsets"].tolist() == [0, 3]
        assert arrays["evidence_frame_id"].tolist() == [0, 25, 50]


def test_input_npz_hash_mismatch_is_rejected(tmp_path):
    inputs = _fixture(tmp_path)
    manifest = json.loads(inputs["input_json"].read_text(encoding="utf-8"))
    manifest["npz_sha256"] = "0" * 64
    inputs["input_json"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ShadowError, match="SHA-256 mismatch"):
        _run(inputs, tmp_path / "output")


def test_existing_output_is_never_overwritten(tmp_path):
    inputs = _fixture(tmp_path)
    output = tmp_path / "output"
    _run(inputs, output)
    json_before = (output / "boxer_past3_shadow.json").read_bytes()
    npz_before = (output / "boxer_past3_shadow.npz").read_bytes()
    with pytest.raises(ShadowError, match="refusing to overwrite"):
        materialize_boxer_past3_shadow(
            **inputs,
            output_json=output / "boxer_past3_shadow.json",
            output_npz=output / "boxer_past3_shadow.npz",
        )
    assert (output / "boxer_past3_shadow.json").read_bytes() == json_before
    assert (output / "boxer_past3_shadow.npz").read_bytes() == npz_before


def test_candidate_npz_is_byte_deterministic(tmp_path):
    inputs = _fixture(tmp_path)
    _run(inputs, tmp_path / "output_a")
    _run(inputs, tmp_path / "output_b")
    assert (
        (tmp_path / "output_a" / "boxer_past3_shadow.npz").read_bytes()
        == (tmp_path / "output_b" / "boxer_past3_shadow.npz").read_bytes()
    )


def test_fixed_shadow_candidate_is_evaluated_without_gt_selection(tmp_path):
    inputs = _fixture(tmp_path)
    output = tmp_path / "output"
    _run(inputs, output)
    gt_root = tmp_path / "gt"
    scan_root = tmp_path / "scans"
    gt_root.mkdir()
    scene_dir = scan_root / "scene0000_00"
    scene_dir.mkdir(parents=True)
    np.save(
        gt_root / "scene0000_00_bbox.npy",
        np.asarray([[0.0, 0.0, 2.0, 1.0, 1.0, 1.0]], dtype=np.float32),
    )
    identity = " ".join(str(value) for value in np.eye(4).reshape(-1))
    (scene_dir / "scene0000_00.txt").write_text(
        f"axisAlignment = {identity}\n", encoding="utf-8"
    )
    report = audit_scannet_boxer_past3_oracle(
        shadow_json=output / "boxer_past3_shadow.json",
        shadow_npz=output / "boxer_past3_shadow.npz",
        baseline_root=inputs["baseline_root"],
        gt_root=gt_root,
        scan_root=scan_root,
    )
    assert report["candidate_selection_used_gt"] is False
    assert report["evaluation_used_gt"] is True
    assert report["totals"]["fixed_candidate_count"] == 1
    for threshold in ("0.15", "0.25", "0.50"):
        row = report["per_threshold"][threshold]
        assert row["additional_union_matching_over_native"] == 1
        assert row["fixed_suffix_delta_ap_points"] > 99.0
    assert report["promotion"]["passes_three_scene_active_counterfactual_gate"]
