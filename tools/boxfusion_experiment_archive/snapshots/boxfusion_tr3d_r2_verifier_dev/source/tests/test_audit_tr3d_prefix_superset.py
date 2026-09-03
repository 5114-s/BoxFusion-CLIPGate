import copy
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from boxfusion.tr3d_residual_cache import (
    make_tr3d_residual_cache_from_aligned,
    tr3d_residual_cache_path,
    write_tr3d_residual_cache,
)
from tools.audit_tr3d_prefix_superset import (
    audit_parent_cache_subset,
    audit_prefix_superset,
)


FIELDS = {
    "schema": "boxfusion.tr3d.trajectory_prefix.v1",
    "tag": "p100",
    "fraction": 1.0,
    "frame_stride": 25,
    "tail_guard_frames": 25,
    "clock_policy": "g0_post_frame_tail_guard_v1",
    "pose_policy": "previous_valid_inf_only_v1",
    "source_timestamp_semantics": "zero_based_scannet_dataset_index",
    "source_frame_count": 51,
    "processed_frame_count": 26,
    "pixel_stride": 4,
    "voxel_size": 0.01,
    "depth_scale": 1000.0,
    "coordinate_frame": "world_unaligned",
    "network_frame_after_pipeline": "scannet_axis_aligned",
    "sampled_frame_count": 2,
    "first_frame_id": 0,
    "last_frame_id": 25,
    "frame_ids": [0, 25],
    "source_timestamps": [0, 25],
    "last_source_timestamp": 25,
    "used_frame_ids": [0, 25],
    "used_source_timestamps": [0, 25],
    "axis_align_matrix": np.eye(4).tolist(),
    "min_observed_points": 20,
    "min_visibility_fraction": 0.0,
    "status": "exported",
    "source_instance_count": 1,
    "kept_instance_count": 1,
    "instance_support": [
        {
            "instance_index": 0,
            "observed_point_count": 1,
            "full_point_count": None,
            "visibility_fraction": None,
            "accepted": True,
        }
    ],
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_scene_artifacts(
    root: Path,
    scene: str,
    *,
    point_value: float,
    pose_value: float,
    calibration_value: float,
) -> dict:
    source = root / "source_namespace" / scene / "frames"
    intrinsic = source / "intrinsic"
    pose_root = source / "pose"
    intrinsic.mkdir(parents=True)
    pose_root.mkdir(parents=True)
    for index, name in enumerate(
        (
            "intrinsic_depth.txt",
            "intrinsic_color.txt",
            "extrinsic_depth.txt",
            "extrinsic_color.txt",
        )
    ):
        (intrinsic / name).write_text(
            f"{calibration_value + index}\n", encoding="utf-8"
        )
    pose = pose_root / "0.txt"
    pose.write_text(f"{pose_value}\n", encoding="utf-8")
    points = root / "points" / "prefixes" / scene / f"{scene}__p100.bin"
    points.parent.mkdir(parents=True, exist_ok=True)
    np.full((2, 6), point_value, dtype=np.float32).tofile(points)
    row = copy.deepcopy(FIELDS)
    row.update(
        {
            "scene_id": scene,
            "source_frames_root": str((root / "source_namespace").resolve()),
            "source_scene_frame_root": str(source.resolve()),
            "point_path": str(points.resolve()),
            "point_count": 2,
            "pose_provenance": [
                {
                    "source_timestamp": 0,
                    "frame_id": 0,
                    "input_pose_frame_id": 0,
                    "input_pose_path": str(pose.resolve()),
                    "input_pose_sha256": _sha(pose),
                    "pose_resolution": "direct",
                    "resolved_pose_source_timestamp": 0,
                    "resolved_pose_frame_id": 0,
                    "resolved_pose_path": str(pose.resolve()),
                    "resolved_pose_sha256": _sha(pose),
                }
            ],
        }
    )
    return row


def _write_export(
    root: Path,
    scenes: list[str],
    *,
    point_value: float = 1.0,
    pose_value: float = 2.0,
    calibration_value: float = 3.0,
) -> tuple[Path, Path, list[dict]]:
    rows = [
        _write_scene_artifacts(
            root,
            scene,
            point_value=point_value,
            pose_value=pose_value,
            calibration_value=calibration_value,
        )
        for scene in scenes
    ]
    manifest = root / "manifests" / "prefix.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    info = root / "annotations" / "info.pkl"
    info.parent.mkdir(parents=True)
    data_list = []
    for row in rows:
        scene = row["scene_id"]
        data_list.append(
            {
                "lidar_points": {
                    "num_pts_feats": 6,
                    "lidar_path": f"prefixes/{scene}/{scene}__p100.bin",
                },
                "instances": [
                    {
                        "bbox_3d": [0, 0, 0, 1, 1, 1],
                        "bbox_label_3d": 0,
                        "prefix_observed_point_count": 1,
                    }
                ],
                "axis_align_matrix": np.eye(4).tolist(),
                "coordinate_frame": "world_unaligned",
                "box_coordinate_frame": "scannet_axis_aligned",
                "trajectory_prefix": copy.deepcopy(row),
            }
        )
    with info.open("wb") as handle:
        pickle.dump({"metainfo": {"classes": ("foreground",)}, "data_list": data_list}, handle)
    return manifest, info, rows


def _scene_list(path: Path, scenes: list[str]) -> Path:
    path.write_text("\n".join(scenes) + "\n", encoding="utf-8")
    return path


def _fixture(tmp_path: Path):
    full_scenes = ["scene0000_00", "scene0001_00"]
    fixed_scenes = ["scene0001_00"]
    full_manifest, full_info, _ = _write_export(
        tmp_path / "full", full_scenes
    )
    fixed_manifest, fixed_info, _ = _write_export(
        tmp_path / "fixed", fixed_scenes
    )
    full_list = _scene_list(tmp_path / "full.txt", full_scenes)
    fixed_list = _scene_list(tmp_path / "fixed.txt", fixed_scenes)
    return {
        "full_manifest": full_manifest,
        "fixed_manifest": fixed_manifest,
        "full_info": full_info,
        "fixed_info": fixed_info,
        "full_scene_list": full_list,
        "fixed_scene_list": fixed_list,
        "expected_full_scene_count": 2,
    }


def test_accepts_relocated_content_identical_fixed_subset(tmp_path: Path) -> None:
    report = audit_prefix_superset(**_fixture(tmp_path))
    assert report["ok"] is True
    assert report["full_scene_count"] == 2
    assert report["fixed_scene_count"] == 1
    assert report["content_identical_fixed_subset"] is True


@pytest.mark.parametrize(
    "kind,match",
    [
        ("clock", "clock/content"),
        ("point", "point content"),
        ("pose", "pose provenance/content"),
        ("calibration", "calibration content"),
    ],
)
def test_rejects_fixed_subset_content_drift(
    tmp_path: Path, kind: str, match: str
) -> None:
    args = _fixture(tmp_path)
    manifest = args["fixed_manifest"]
    row = json.loads(manifest.read_text(encoding="utf-8"))
    if kind == "clock":
        row["frame_stride"] = 10
    elif kind == "point":
        np.full((2, 6), 9.0, dtype=np.float32).tofile(row["point_path"])
    elif kind == "pose":
        path = Path(row["pose_provenance"][0]["input_pose_path"])
        path.write_text("99\n", encoding="utf-8")
        digest = _sha(path)
        row["pose_provenance"][0]["input_pose_sha256"] = digest
        row["pose_provenance"][0]["resolved_pose_sha256"] = digest
    else:
        path = Path(row["source_scene_frame_root"]) / "intrinsic" / "intrinsic_depth.txt"
        path.write_text("99\n", encoding="utf-8")
    manifest.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    # Keep the fixed info internally consistent; the cross-export audit must
    # still reject the content drift itself.
    with args["fixed_info"].open("rb") as handle:
        payload = pickle.load(handle)
    payload["data_list"][0]["trajectory_prefix"] = copy.deepcopy(row)
    with args["fixed_info"].open("wb") as handle:
        pickle.dump(payload, handle)
    with pytest.raises(ValueError, match=match):
        audit_prefix_superset(**args)


def test_rejects_duplicate_full_scene(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    text = args["full_manifest"].read_text(encoding="utf-8").splitlines()
    args["full_manifest"].write_text(
        text[0] + "\n" + text[0] + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unique p100 scenes"):
        audit_prefix_superset(**args)


def _cache(root: Path, scene: str, *, center: float) -> None:
    sha = "a" * 64
    config = "b" * 64
    cache = make_tr3d_residual_cache_from_aligned(
        scene_id=scene,
        boxes_aligned=np.asarray([[center, 0, 0, 1, 1, 1, 0]], dtype=float),
        scores_3d=np.asarray([0.8]),
        unaligned_to_aligned=np.eye(4),
        checkpoint_sha256=sha,
        config_sha256=config,
        source_scene_sha256="c" * 64,
        prefix_id="p100",
        voxel_size=0.01,
        num_input_points=2,
    )
    write_tr3d_residual_cache(
        tr3d_residual_cache_path(root, scene, "p100"), cache
    )


def test_parent_cache_subset_requires_exact_arrays(tmp_path: Path) -> None:
    scene = "scene0001_00"
    scene_list = _scene_list(tmp_path / "fixed.txt", [scene])
    _cache(tmp_path / "full", scene, center=0.0)
    _cache(tmp_path / "fixed", scene, center=0.0)
    report = audit_parent_cache_subset(
        full_cache_root=tmp_path / "full",
        fixed_cache_root=tmp_path / "fixed",
        fixed_scene_list=scene_list,
        prefix_id="p100",
        checkpoint_sha256="a" * 64,
        config_sha256="b" * 64,
    )
    assert report["arrays_exact"] is True

    _cache(tmp_path / "drift", scene, center=0.1)
    with pytest.raises(ValueError, match="cache array"):
        audit_parent_cache_subset(
            full_cache_root=tmp_path / "drift",
            fixed_cache_root=tmp_path / "fixed",
            fixed_scene_list=scene_list,
            prefix_id="p100",
            checkpoint_sha256="a" * 64,
            config_sha256="b" * 64,
        )
