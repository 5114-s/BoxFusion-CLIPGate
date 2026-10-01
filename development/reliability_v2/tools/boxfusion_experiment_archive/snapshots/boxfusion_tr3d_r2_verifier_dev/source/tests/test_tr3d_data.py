from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.tr3d_data import (
    backproject_rgbd,
    build_foreground_info,
    deterministic_partition,
    filter_prefix_instances,
    prefix_schedule,
    scene_id_from_info,
    voxel_downsample_first,
)
from tools.prepare_tr3d_scannet import discover_frame_scenes


def _scene(index):
    return f"scene{index:04d}_00"


def _info_row(scene, label=7):
    return {
        "lidar_points": {
            "num_pts_feats": 6,
            "lidar_path": f"{scene}.bin",
        },
        "instances": [{
            "bbox_3d": [1, 2, 3, 4, 5, 6],
            "bbox_label_3d": label,
        }],
        "pts_semantic_mask_path": f"{scene}.bin",
        "pts_instance_mask_path": f"{scene}.bin",
        "axis_align_matrix": np.eye(4).tolist(),
    }


def test_deterministic_partition_is_order_invariant_and_leak_free():
    train = [_scene(index) for index in range(30)]
    forbidden = [_scene(index) for index in range(100, 105)]
    first = deterministic_partition(
        train,
        forbidden_scenes=forbidden,
        calibration_size=5,
        audit_size=7,
        seed="unit-test",
    )
    second = deterministic_partition(
        list(reversed(train)),
        forbidden_scenes=forbidden,
        calibration_size=5,
        audit_size=7,
        seed="unit-test",
    )
    assert first == second
    assert {key: len(value) for key, value in first.items()} == {
        "train": 18,
        "calibration": 5,
        "audit": 7,
    }
    assert not set().union(*map(set, first.values())) & set(forbidden)
    assert set().union(*map(set, first.values())) == set(train)


def test_deterministic_partition_rejects_val_overlap():
    train = [_scene(index) for index in range(10)]
    with pytest.raises(ValueError, match="forbidden validation"):
        deterministic_partition(
            train,
            forbidden_scenes=[train[-1]],
            calibration_size=2,
            audit_size=2,
        )


def test_foreground_transform_maps_every_instance_and_rewrites_point_path():
    scenes = [_scene(2), _scene(1)]
    rows = [_info_row(scenes[0], 3), _info_row(scenes[1], 17)]
    value = build_foreground_info(
        {
            "dataset": "scannet",
            "categories": {
                "chair": 0,
                "table": 1,
            },
        },
        rows,
        scenes,
        point_path_prefix="full",
    )
    assert value["metainfo"]["categories"] == {"foreground": 0}
    assert value["metainfo"]["classes"] == ("foreground",)
    assert [
        scene_id_from_info(row) for row in value["data_list"]
    ] == sorted(scenes)
    for row in value["data_list"]:
        assert row["coordinate_frame"] == "world_unaligned"
        assert row["box_coordinate_frame"] == "scannet_axis_aligned"
        assert row["lidar_points"]["lidar_path"].startswith("full/")
        instance = row["instances"][0]
        assert instance["bbox_label_3d"] == 0
        assert instance["source_category_id"] in {3, 17}
    # The source object must remain untouched.
    assert rows[0]["instances"][0]["bbox_label_3d"] == 3


def test_prefix_schedule_is_monotonic_and_always_includes_last_frame():
    schedule = prefix_schedule(
        list(range(101)), fractions=(0.25, 0.5, 0.75, 1.0),
        frame_stride=25)
    assert [item["sampled_frame_count"] for item in schedule] == [2, 3, 4, 5]
    assert schedule[-1]["frame_ids"] == [0, 25, 50, 75, 100]
    assert schedule[-1]["last_frame_id"] == 100


def test_frame_scene_discovery_accepts_direct_and_nested_layouts(tmp_path):
    direct = tmp_path / _scene(1)
    nested = tmp_path / _scene(2) / "frames"
    incomplete = tmp_path / _scene(3)
    for root in (direct, nested):
        for name in ("color", "depth", "pose", "intrinsic"):
            (root / name).mkdir(parents=True, exist_ok=True)
    (incomplete / "color").mkdir(parents=True)
    assert discover_frame_scenes(tmp_path) == [_scene(1), _scene(2)]


def test_prefix_gt_filter_counts_points_after_axis_alignment():
    axis = np.eye(4)
    axis[0, 3] = 10.0
    # Stored prefix XYZ is unaligned around x=0. Axis alignment moves it to
    # x=10, where the first aligned GT box lives.
    prefix = np.zeros((25, 6), dtype=np.float32)
    prefix[:, :3] = np.linspace(-0.1, 0.1, 75).reshape(25, 3)
    instances = [
        {
            "bbox_3d": [10, 0, 0, 1, 1, 1],
            "bbox_label_3d": 2,
        },
        {
            "bbox_3d": [20, 0, 0, 1, 1, 1],
            "bbox_label_3d": 4,
        },
    ]
    kept, diagnostics = filter_prefix_instances(
        instances, prefix, axis, min_observed_points=20)
    assert len(kept) == 1
    assert kept[0]["bbox_label_3d"] == 0
    assert kept[0]["source_category_id"] == 2
    assert diagnostics[0]["observed_point_count"] == 25
    assert diagnostics[0]["accepted"] is True
    assert diagnostics[1]["accepted"] is False


def test_prefix_gt_filter_supports_optional_visibility_fraction():
    axis = np.eye(4)
    instances = [{
        "bbox_3d": [0, 0, 0, 2, 2, 2],
        "bbox_label_3d": 0,
    }]
    prefix = np.zeros((20, 6), dtype=np.float32)
    full = np.zeros((100, 6), dtype=np.float32)
    kept, diagnostics = filter_prefix_instances(
        instances,
        prefix,
        axis,
        min_observed_points=10,
        full_world_points=full,
        min_visibility_fraction=0.25,
    )
    assert kept == []
    assert diagnostics[0]["visibility_fraction"] == pytest.approx(0.2)


def test_backprojection_emits_unaligned_world_xyzrgb(tmp_path):
    depth = np.full((2, 2), 1000, dtype=np.uint16)
    color = np.asarray(
        [[[10, 20, 30], [40, 50, 60]],
         [[70, 80, 90], [100, 110, 120]]],
        dtype=np.uint8,
    )
    depth_path = tmp_path / "depth.png"
    color_path = tmp_path / "color.png"
    Image.fromarray(depth).save(depth_path)
    Image.fromarray(color).save(color_path)
    intrinsic = np.eye(4)
    pose = np.eye(4)
    pose[0, 3] = 10
    points = backproject_rgbd(
        depth_path=depth_path,
        color_path=color_path,
        pose=pose,
        intrinsic_depth=intrinsic,
        intrinsic_color=intrinsic,
        extrinsic_depth=np.eye(4),
        extrinsic_color=np.eye(4),
        pixel_stride=1,
    )
    assert points.shape == (4, 6)
    np.testing.assert_allclose(points[:, 0], [10, 11, 10, 11])
    np.testing.assert_allclose(points[:, 1], [0, 0, 1, 1])
    np.testing.assert_allclose(points[:, 2], 1)
    np.testing.assert_array_equal(points[:, 3:], color.reshape(-1, 3))


def test_voxel_downsample_keeps_first_observation():
    points = np.asarray([
        [0.01, 0.01, 0.01, 1, 2, 3],
        [0.02, 0.02, 0.02, 9, 9, 9],
        [1.01, 0.00, 0.00, 4, 5, 6],
    ], dtype=np.float32)
    result = voxel_downsample_first(points, 0.1)
    np.testing.assert_array_equal(result, points[[0, 2]])


def test_configs_inherit_pinned_official_tr3d_and_use_one_class_head():
    root = Path(__file__).resolve().parents[1]
    config = (
        root / "config" / "tr3d" / "tr3d_scannet_foreground.py"
    ).read_text()
    base = (
        root / "third_party" / "mmdetection3d"
        / "projects" / "TR3D" / "configs" / "tr3d.py"
    ).read_text()
    assert "projects/TR3D/configs/tr3d.py" in config
    assert "configs/_base_/datasets/scannet-3d.py" in config
    assert "tr3d_1xb16_scannet-3d-18class.py" in config
    assert "TR3DMinkResNet" in base
    assert "TR3DNeck" in base
    assert "TR3DClassAgnosticHead" in config
    assert "projects.TR3D.tr3d" in config
    assert "tr3d_plugin" in config
    assert "label2level" not in config
