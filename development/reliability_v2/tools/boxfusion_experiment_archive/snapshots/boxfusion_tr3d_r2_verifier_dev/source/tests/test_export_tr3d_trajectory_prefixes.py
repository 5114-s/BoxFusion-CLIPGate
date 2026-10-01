from pathlib import Path

import numpy as np
import pytest

from tools.export_tr3d_trajectory_prefixes import (
    BOXFUSION_CLOCK_POLICY,
    BOXFUSION_POSE_POLICY,
    boxfusion_prefix_schedule,
    export_boxfusion_scene_prefixes,
)


def _write_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix)


def _make_manifest_only_scene(
        root: Path,
        *,
        scene_id: str,
        frame_count: int,
        infinite_pose_timestamps=(),
) -> Path:
    frames = root / scene_id / "frames"
    for name in ("color", "depth", "pose", "intrinsic"):
        (frames / name).mkdir(parents=True, exist_ok=True)
    identity = np.eye(4, dtype=np.float64)
    for name in (
        "intrinsic_depth.txt",
        "intrinsic_color.txt",
        "extrinsic_depth.txt",
        "extrinsic_color.txt",
    ):
        _write_matrix(frames / "intrinsic" / name, identity)
    infinite = set(infinite_pose_timestamps)
    for source_timestamp in range(frame_count):
        # Manifest-only export discovers the modalities but never decodes RGB-D.
        (frames / "color" / f"{source_timestamp}.jpg").touch()
        (frames / "depth" / f"{source_timestamp}.png").touch()
        pose = identity.copy()
        pose[0, 3] = float(source_timestamp)
        if source_timestamp in infinite:
            pose[0, 0] = np.inf
        _write_matrix(frames / "pose" / f"{source_timestamp}.txt", pose)
    return frames


def test_scene0435_g0_clock_excludes_protected_tail_and_final_frame() -> None:
    # scene0435_00 has 3273 source frames.  Frozen G0 logs 130 keyframes:
    # timestamps 0..3225.  The legacy exporter incorrectly appended 3272.
    schedule = boxfusion_prefix_schedule(
        list(range(3273)), fractions=(1.0,), frame_stride=25)
    assert len(schedule) == 1
    item = schedule[0]
    assert item["source_frame_count"] == 3273
    assert item["processed_frame_count"] == 3248
    assert item["sampled_frame_count"] == 130
    assert item["source_timestamps"] == list(range(0, 3248, 25))
    assert item["frame_ids"] == list(range(0, 3248, 25))
    assert item["last_source_timestamp"] == 3225
    assert item["last_frame_id"] == 3225
    assert 3250 not in item["frame_ids"]
    assert 3272 not in item["frame_ids"]


def test_source_timestamp_is_sequence_position_not_numeric_frame_id() -> None:
    frame_ids = [1000 + 10 * index for index in range(76)]
    item = boxfusion_prefix_schedule(
        frame_ids, fractions=(1.0,), frame_stride=25)[0]
    assert item["source_timestamps"] == [0, 25, 50]
    assert item["frame_ids"] == [1000, 1250, 1500]
    assert item["last_frame_id"] != item["last_source_timestamp"]


def test_manifest_records_non_keyframe_pose_carry_forward(tmp_path: Path) -> None:
    scene = "scene0496_00"
    frames_root = tmp_path / "frames_root"
    _make_manifest_only_scene(
        frames_root,
        scene_id=scene,
        frame_count=51,
        infinite_pose_timestamps=(25,),
    )
    rows, manifests = export_boxfusion_scene_prefixes(
        scene_id=scene,
        frame_root=frames_root,
        source_row={"instances": [], "axis_align_matrix": np.eye(4).tolist()},
        output_root=tmp_path / "prepared",
        fractions=(1.0,),
        frame_stride=25,
        manifest_only=True,
    )
    assert rows == []
    assert len(manifests) == 1
    manifest = manifests[0]
    assert manifest["status"] == "planned"
    assert manifest["clock_policy"] == BOXFUSION_CLOCK_POLICY
    assert manifest["pose_policy"] == BOXFUSION_POSE_POLICY
    assert manifest["source_timestamps"] == [0, 25]
    assert manifest["used_source_timestamps"] == [0, 25]
    assert manifest["frame_ids"] == [0, 25]
    assert manifest["used_frame_ids"] == [0, 25]
    assert manifest["last_source_timestamp"] == 25
    assert manifest["last_frame_id"] == 25

    direct, carried = manifest["pose_provenance"]
    assert direct["source_timestamp"] == 0
    assert direct["pose_resolution"] == "direct"
    assert direct["resolved_pose_source_timestamp"] == 0
    assert direct["resolved_pose_frame_id"] == 0
    assert carried["source_timestamp"] == 25
    assert carried["input_pose_frame_id"] == 25
    assert carried["pose_resolution"] == "carry_forward"
    # The immediately preceding valid *raw* frame is 24, although it is not a
    # selected keyframe.  This is the behavior of ScannetDataset.load_poses.
    assert carried["resolved_pose_source_timestamp"] == 24
    assert carried["resolved_pose_frame_id"] == 24
    assert carried["input_pose_sha256"] != carried["resolved_pose_sha256"]
    assert Path(carried["resolved_pose_path"]).name == "24.txt"


def test_first_infinite_pose_fails_like_unresolvable_g0_input(
        tmp_path: Path) -> None:
    scene = "scene0000_00"
    frames_root = tmp_path / "frames_root"
    _make_manifest_only_scene(
        frames_root,
        scene_id=scene,
        frame_count=30,
        infinite_pose_timestamps=(0,),
    )
    with pytest.raises(ValueError, match="no previous valid pose"):
        export_boxfusion_scene_prefixes(
            scene_id=scene,
            frame_root=frames_root,
            source_row={
                "instances": [],
                "axis_align_matrix": np.eye(4).tolist(),
            },
            output_root=tmp_path / "prepared",
            fractions=(1.0,),
            frame_stride=25,
            manifest_only=True,
        )
