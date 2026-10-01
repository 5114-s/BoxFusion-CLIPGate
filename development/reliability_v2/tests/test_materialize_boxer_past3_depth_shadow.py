import hashlib
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.audit_scannet_boxer_past3_oracle import _load_shadow
from tools.materialize_boxer_past3_depth_shadow import (
    FrameAccessError,
    SCHEMA,
    _SceneFrameStore,
    _choose_qualifying_component,
    materialize_boxer_past3_depth_shadow,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(
    tmp_path: Path,
    *,
    third_depth_valid: bool = False,
    schedule_schema: str = "boxfusion.cutr_postfilter_cache.v3",
    include_later_candidate: bool = True,
) -> dict[str, Path]:
    scene = "scene0000_00"
    input_dir = tmp_path / "input"
    baseline = tmp_path / "baseline"
    schedules = tmp_path / "schedules"
    rgbd = tmp_path / "rgbd"
    input_dir.mkdir()
    baseline.mkdir()
    (schedules / scene).mkdir(parents=True)
    frames_root = rgbd / scene / "frames"
    for name in ("pose", "depth", "intrinsic"):
        (frames_root / name).mkdir(parents=True)

    frame_ids = [0, 25, 50, 75]
    schedule = {
        "schema": schedule_schema,
        "scene_id": scene,
        "namespace": "synthetic-depth-s1",
        "record_count": len(frame_ids),
        "recorded_frame_ids": frame_ids,
    }
    schedule_path = schedules / scene / "manifest.json"
    schedule_path.write_text(json.dumps(schedule), encoding="utf-8")

    K = np.eye(4, dtype=np.float64)
    K[0, 0] = K[1, 1] = 100.0
    K[0, 2], K[1, 2] = 32.0, 24.0
    np.savetxt(frames_root / "intrinsic" / "intrinsic_depth.txt", K)
    camera_x = {0: -0.5, 25: 0.0, 50: 0.5, 75: 0.75}
    for frame_id in frame_ids:
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = camera_x[frame_id]
        np.savetxt(frames_root / "pose" / f"{frame_id}.txt", pose)
        valid = frame_id != 50 or third_depth_valid
        depth = np.full((48, 64), 2000 if valid else 0, dtype=np.uint16)
        assert cv2.imwrite(str(frames_root / "depth" / f"{frame_id}.png"), depth)

    candidate_frames = frame_ids if include_later_candidate else frame_ids[:3]
    count = len(candidate_frames)
    arrays = {
        "scene_ids": np.asarray([scene], dtype="<U12"),
        "per_view_scene_index": np.zeros(count, dtype=np.int16),
        "per_view_frame_id": np.asarray(candidate_frames, dtype=np.int64),
        "per_view_source_row": np.asarray([10, 11, 12, 13][:count], dtype=np.int32),
        "per_view_source_instance_id": np.zeros(count, dtype=np.int32),
        "per_view_center_world": np.tile(
            np.asarray([[0.0, 0.0, 2.0]], dtype=np.float32), (count, 1)
        ),
        "per_view_quaternion_wxyz": np.tile(
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (count, 1)
        ),
        "per_view_extent_xyz": np.ones((count, 3), dtype=np.float32),
        "per_view_source_score": np.asarray([0.9, 0.8, 0.7, 0.6][:count], dtype=np.float32),
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
    input_json.write_text(
        json.dumps(
            {
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
                "per_view_candidate_count": count,
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
        ),
        encoding="utf-8",
    )
    with (baseline / f"{scene}_boxes.pkl").open("wb") as handle:
        pickle.dump([[]], handle)
    preregistration = tmp_path / "preregistered.md"
    preregistration.write_text("frozen before held-out GT\n", encoding="utf-8")
    return {
        "input_json": input_json,
        "input_npz": input_npz,
        "baseline_root": baseline,
        "schedule_root": schedules,
        "scene_rgbd_root": rgbd,
        "preregistration": preregistration,
    }


def _run(inputs: dict[str, Path], output: Path):
    output.mkdir()
    return materialize_boxer_past3_depth_shadow(
        **inputs,
        output_json=output / "boxer_past3_depth_shadow.json",
        output_npz=output / "boxer_past3_depth_shadow.npz",
    )


def test_later_past_only_hit_qualifies_without_changing_receipt(tmp_path):
    inputs = _fixture(tmp_path)
    native = inputs["baseline_root"] / "scene0000_00_boxes.pkl"
    native_before = _sha256(native)
    report = _run(inputs, tmp_path / "output")

    assert report["schema"] == SCHEMA
    assert report["birth"] is False
    assert report["gt_access"] is False
    assert report["candidate_count"] == 1
    assert report["native_before_after_identity"] is True
    assert _sha256(native) == native_before
    row = report["scenes"]["scene0000_00"]["accepted_candidates"][0]
    assert row["confirmation_frame_id"] == 50
    assert row["qualification_frame_id"] == 75
    assert row["receipt_evidence_frame_ids"] == [0, 25, 50]
    assert row["receipt_evidence_source_rows"] == [10, 11, 12]
    qualification = row["depth_qualification"]
    assert qualification["node_frame_ids"] == [0, 25, 75]
    assert qualification["qualifying_component"]["frame_ids"] == [0, 25, 75]
    assert qualification["qualifying_component"]["support_edge_count"] >= 2
    assert all(
        edge["v_f"] > 0.30 and edge["v_b"] > 0.90
        for edge in qualification["qualifying_component"]["support_edges"]
    )

    with np.load(tmp_path / "output" / "boxer_past3_depth_shadow.npz") as arrays:
        assert arrays["candidate_corners_world"].shape == (1, 8, 3)
        assert arrays["receipt_evidence_frame_id"].tolist() == [0, 25, 50]
        assert arrays["depth_node_frame_id"].tolist() == [0, 25, 75]
        assert arrays["candidate_support_edge_offsets"][1] >= 2


@pytest.mark.parametrize(
    "schedule_schema",
    ["boxfusion.cutr_postfilter_cache.v2", "boxfusion.cutr_postfilter_cache.v3"],
)
def test_v2_v3_schedules_and_initial_nodes_qualify_at_receipt_time(
    tmp_path, schedule_schema
):
    inputs = _fixture(
        tmp_path, third_depth_valid=True, schedule_schema=schedule_schema
    )
    report = _run(inputs, tmp_path / "output")
    row = report["scenes"]["scene0000_00"]["accepted_candidates"][0]
    assert row["confirmation_frame_id"] == 50
    assert row["qualification_frame_id"] == 50
    assert 75 not in row["depth_qualification"]["node_frame_ids"]


def test_disconnected_global_edge_count_does_not_pass_component_gate():
    components = [
        {"frame_ids": [0, 1], "support_edge_count": 1, "support_edges": [{}]},
        {"frame_ids": [2, 3], "support_edge_count": 1, "support_edges": [{}]},
    ]
    assert sum(row["support_edge_count"] for row in components) == 2
    assert _choose_qualifying_component(components) is None
    connected = [
        {"frame_ids": [0, 1, 2], "support_edge_count": 2, "support_edges": [{}, {}]}
    ]
    assert _choose_qualifying_component(connected) is connected[0]


def test_zero_candidate_keyframe_still_advances_rgbd_ring(tmp_path):
    inputs = _fixture(
        tmp_path, third_depth_valid=True, include_later_candidate=False
    )
    report = _run(inputs, tmp_path / "output")
    scene = report["scenes"]["scene0000_00"]
    assert scene["zero_candidate_keyframes"] == 1
    assert scene["depth_frame_store"]["zero_candidate_frames_advanced"] == 1
    assert scene["depth_frame_store"]["frames_advanced"] == 4
    assert scene["depth_frame_store"]["arbitrary_historical_reload"] is False


def test_rgbd_ring_rejects_future_and_evicted_access(tmp_path):
    inputs = _fixture(tmp_path)
    scene_root = inputs["scene_rgbd_root"]
    frames_root = scene_root / "scene0000_00" / "frames"
    K = np.eye(4, dtype=np.float64)
    K[0, 0] = K[1, 1] = 100.0
    K[0, 2], K[1, 2] = 32.0, 24.0
    np.savetxt(frames_root / "intrinsic" / "intrinsic_depth.txt", K)
    frame_ids = list(range(12))
    for frame_id in frame_ids:
        np.savetxt(frames_root / "pose" / f"{frame_id}.txt", np.eye(4))
        assert cv2.imwrite(
            str(frames_root / "depth" / f"{frame_id}.png"),
            np.full((48, 64), 2000, dtype=np.uint16),
        )
    store = _SceneFrameStore(scene_root, "scene0000_00", frame_ids)
    store.advance(0)
    with pytest.raises(FrameAccessError, match="future"):
        store.get(1)
    for frame_id in frame_ids[1:]:
        store.advance(frame_id)
    assert store.peak_cached_frames == 11
    with pytest.raises(FrameAccessError, match="evicted"):
        store.get(0)
    assert store.get(1).frame_id == 1


def test_outputs_are_byte_deterministic_and_not_overwritten(tmp_path):
    inputs = _fixture(tmp_path)
    _run(inputs, tmp_path / "output_a")
    _run(inputs, tmp_path / "output_b")
    left = tmp_path / "output_a" / "boxer_past3_depth_shadow.npz"
    right = tmp_path / "output_b" / "boxer_past3_depth_shadow.npz"
    assert left.read_bytes() == right.read_bytes()
    with pytest.raises(ValueError, match="refusing to overwrite"):
        materialize_boxer_past3_depth_shadow(
            **inputs,
            output_json=tmp_path / "output_a" / "boxer_past3_depth_shadow.json",
            output_npz=left,
        )


def test_depth_shadow_is_accepted_by_fixed_candidate_oracle_loader(tmp_path):
    inputs = _fixture(tmp_path)
    _run(inputs, tmp_path / "output")
    manifest, arrays, scenes = _load_shadow(
        tmp_path / "output" / "boxer_past3_depth_shadow.json",
        tmp_path / "output" / "boxer_past3_depth_shadow.npz",
    )
    assert manifest["schema"] == SCHEMA
    assert scenes == ("scene0000_00",)
    assert arrays["candidate_corners_world"].shape == (1, 8, 3)
