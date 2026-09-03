import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from boxfusion.tr3d_r2_provenance import (
    R2A_CLOCK_POLICY,
    R2A_POSE_POLICY,
    R2A_TIMESTAMP_SEMANTICS,
    canonical_json_sha256,
    frame_artifact_tree,
    load_resolved_poses,
    validate_prefix_manifest_row,
)
from tools.tr3d_data import FrameBundle, PREFIX_SCHEMA


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix(path: Path, translation: float = 0.0) -> None:
    value = np.eye(4, dtype=np.float64)
    value[0, 3] = translation
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, value)


def _fixture(tmp_path: Path) -> tuple[FrameBundle, dict]:
    scene = "scene0001_00"
    frames = tmp_path / scene / "frames"
    for directory in ("color", "depth", "pose", "intrinsic"):
        (frames / directory).mkdir(parents=True, exist_ok=True)
    for name in (
        "intrinsic_depth.txt",
        "intrinsic_color.txt",
        "extrinsic_depth.txt",
        "extrinsic_color.txt",
    ):
        _matrix(frames / "intrinsic" / name)
    color: dict[int, Path] = {}
    depth: dict[int, Path] = {}
    pose: dict[int, Path] = {}
    for frame_id in (0, 25):
        color[frame_id] = frames / "color" / f"{frame_id}.jpg"
        depth[frame_id] = frames / "depth" / f"{frame_id}.png"
        pose[frame_id] = frames / "pose" / f"{frame_id}.txt"
        color[frame_id].write_bytes(f"color-{frame_id}".encode())
        depth[frame_id].write_bytes(f"depth-{frame_id}".encode())
        _matrix(pose[frame_id], float(frame_id))
    bundle = FrameBundle(
        scene_id=scene,
        frame_root=frames,
        color=color,
        depth=depth,
        pose=pose,
        intrinsic_depth=frames / "intrinsic" / "intrinsic_depth.txt",
        intrinsic_color=frames / "intrinsic" / "intrinsic_color.txt",
        extrinsic_depth=frames / "intrinsic" / "extrinsic_depth.txt",
        extrinsic_color=frames / "intrinsic" / "extrinsic_color.txt",
    )
    row = {
        "schema": PREFIX_SCHEMA,
        "status": "exported",
        "scene_id": scene,
        "tag": "p100",
        "fraction": 1.0,
        "clock_policy": R2A_CLOCK_POLICY,
        "pose_policy": R2A_POSE_POLICY,
        "source_timestamp_semantics": R2A_TIMESTAMP_SEMANTICS,
        "frame_stride": 25,
        "tail_guard_frames": 25,
        "source_frame_count": 51,
        "processed_frame_count": 26,
        "frame_ids": [0, 25],
        "used_frame_ids": [0, 25],
        "source_timestamps": [0, 25],
        "used_source_timestamps": [0, 25],
        "sampled_frame_count": 2,
        "last_frame_id": 25,
        "last_source_timestamp": 25,
        "pose_provenance": [
            {
                "source_timestamp": 0,
                "frame_id": 0,
                "input_pose_frame_id": 0,
                "input_pose_sha256": _sha(pose[0]),
                "pose_resolution": "direct",
                "resolved_pose_source_timestamp": 0,
                "resolved_pose_frame_id": 0,
                "resolved_pose_sha256": _sha(pose[0]),
            },
            {
                "source_timestamp": 25,
                "frame_id": 25,
                "input_pose_frame_id": 25,
                "input_pose_sha256": _sha(pose[25]),
                "pose_resolution": "carry_forward",
                "resolved_pose_source_timestamp": 0,
                "resolved_pose_frame_id": 0,
                "resolved_pose_sha256": _sha(pose[0]),
            },
        ],
    }
    return bundle, row


def test_manifest_and_artifact_tree_are_deterministic(tmp_path: Path) -> None:
    bundle, row = _fixture(tmp_path)
    checked = validate_prefix_manifest_row(
        row, expected_scene_id=bundle.scene_id, expected_prefix_id="p100"
    )
    assert canonical_json_sha256(checked) == canonical_json_sha256(row)
    first, records = frame_artifact_tree(checked, bundle)
    second, repeated = frame_artifact_tree(checked, bundle)
    assert first == second
    assert records == repeated
    assert len(records) == 12  # 4 calibration + 4 artifacts per frame.

    poses = load_resolved_poses(checked, bundle)
    assert set(poses) == {0, 25}
    np.testing.assert_array_equal(poses[0], poses[25])


def test_artifact_tree_detects_post_export_pose_change(tmp_path: Path) -> None:
    bundle, row = _fixture(tmp_path)
    _matrix(Path(bundle.pose[0]), 123.0)
    with pytest.raises(ValueError, match="pose content changed"):
        frame_artifact_tree(row, bundle)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("clock_policy", "legacy", "frozen-G0 clock"),
        ("used_frame_ids", [0], "lists disagree"),
        ("source_timestamps", [0, 24], "lists disagree"),
    ],
)
def test_manifest_contract_fails_closed(
    tmp_path: Path, field: str, value, message: str
) -> None:
    _, row = _fixture(tmp_path)
    row[field] = value
    with pytest.raises(ValueError, match=message):
        validate_prefix_manifest_row(row, expected_prefix_id="p100")


def test_manifest_rejects_off_clock_timestamp(tmp_path: Path) -> None:
    _, row = _fixture(tmp_path)
    row["source_timestamps"] = [0, 24]
    row["used_source_timestamps"] = [0, 24]
    row["last_source_timestamp"] = 24
    row["pose_provenance"][1]["source_timestamp"] = 24
    with pytest.raises(ValueError, match="keyframe stride"):
        validate_prefix_manifest_row(row, expected_prefix_id="p100")


def test_manifest_rejects_stride_aligned_future_frame(tmp_path: Path) -> None:
    _, row = _fixture(tmp_path)
    row["frame_ids"].append(50)
    row["used_frame_ids"].append(50)
    row["source_timestamps"].append(50)
    row["used_source_timestamps"].append(50)
    row["sampled_frame_count"] = 3
    row["last_frame_id"] = 50
    row["last_source_timestamp"] = 50
    item = copy.deepcopy(row["pose_provenance"][0])
    item.update({
        "source_timestamp": 50,
        "frame_id": 50,
        "input_pose_frame_id": 50,
        "resolved_pose_source_timestamp": 50,
        "resolved_pose_frame_id": 50,
    })
    row["pose_provenance"].append(item)
    with pytest.raises(ValueError, match="frozen-G0 schedule"):
        validate_prefix_manifest_row(row, expected_prefix_id="p100")


def test_manifest_rejects_forged_tail_contract(tmp_path: Path) -> None:
    _, row = _fixture(tmp_path)
    row["processed_frame_count"] = 51
    with pytest.raises(ValueError, match="processed_frame_count"):
        validate_prefix_manifest_row(row, expected_prefix_id="p100")

    _, row = _fixture(tmp_path)
    row["tail_guard_frames"] = 1
    with pytest.raises(ValueError, match="tail guard"):
        validate_prefix_manifest_row(row, expected_prefix_id="p100")


def test_manifest_rejects_noncanonical_frame_id_map(tmp_path: Path) -> None:
    _, row = _fixture(tmp_path)
    row["frame_ids"] = [0, 24]
    row["used_frame_ids"] = [0, 24]
    row["last_frame_id"] = 24
    row["pose_provenance"][1]["frame_id"] = 24
    row["pose_provenance"][1]["input_pose_frame_id"] = 24
    with pytest.raises(ValueError, match="timestamp map"):
        validate_prefix_manifest_row(row, expected_prefix_id="p100")
