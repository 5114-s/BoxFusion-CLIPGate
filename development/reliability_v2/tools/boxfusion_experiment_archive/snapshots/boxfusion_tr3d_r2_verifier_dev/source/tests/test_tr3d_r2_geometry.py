from __future__ import annotations

import numpy as np
import pytest

from boxfusion.tr3d_r2_geometry import (
    classify_depth_rays,
    compose_depth_camera_to_world,
    intersect_rays_with_yaw_obb,
    project_yaw_obb_to_depth,
    stable_top_k_view_indices,
    yaw_obb_corners_world,
)


def intrinsics(height: int, width: int, focal: float = 10.0) -> np.ndarray:
    return np.asarray(
        [
            [focal, 0.0, (width - 1) / 2.0],
            [0.0, focal, (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_compose_depth_camera_to_world_uses_pose_then_depth_extrinsic():
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [10.0, 0.0, 0.0]
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, 3] = [0.0, 2.0, 0.0]

    result = compose_depth_camera_to_world(pose, extrinsic)

    assert np.array_equal(result[:3, 3], [10.0, 2.0, 0.0])
    assert not result.flags.writeable


def test_yaw_obb_corners_and_depth_projection_preserve_rotation():
    box = np.asarray([0.0, 0.0, 5.0, 2.0, 4.0, 2.0, np.pi / 2])
    corners = yaw_obb_corners_world(box)

    assert np.ptp(corners[:, 0]) == pytest.approx(4.0)
    assert np.ptp(corners[:, 1]) == pytest.approx(2.0)
    assert np.ptp(corners[:, 2]) == pytest.approx(2.0)
    projection = project_yaw_obb_to_depth(
        box, intrinsics(101, 101, 100.0), np.eye(4), (101, 101)
    )
    assert projection is not None
    assert projection.pixels.shape == (8, 2)
    assert projection.area_pixels > 0.0
    assert 0.0 < projection.area_ratio <= 1.0
    assert not projection.pixels.flags.writeable


def test_projection_returns_none_for_near_plane_or_outside_image():
    assert (
        project_yaw_obb_to_depth(
            [0, 0, 0.5, 1, 1, 2, 0],
            intrinsics(11, 11),
            np.eye(4),
            (11, 11),
        )
        is None
    )
    assert (
        project_yaw_obb_to_depth(
            [100, 0, 5, 1, 1, 1, 0],
            intrinsics(11, 11),
            np.eye(4),
            (11, 11),
        )
        is None
    )


def test_ray_obb_slab_intersection_handles_hit_miss_and_inside_origin():
    box = [0, 0, 5, 2, 2, 2, 0]
    result = intersect_rays_with_yaw_obb(
        [0, 0, 0],
        [[0, 0, 1], [2, 0, 1]],
        box,
    )
    assert result.intersects.tolist() == [True, False]
    assert result.t_near.tolist() == pytest.approx([4.0, 0.0])
    assert result.t_far.tolist() == pytest.approx([6.0, 0.0])

    inside = intersect_rays_with_yaw_obb(
        [0, 0, 5], [[1, 0, 0]], box
    )
    assert inside.intersects.tolist() == [True]
    assert inside.t_near[0] == pytest.approx(0.0)
    assert inside.t_far[0] == pytest.approx(1.0)


def test_rotated_yaw_changes_ray_intersection_extent():
    origin = [0, 0, 0]
    # At z=5 this ray is at x=1.5.  It misses the narrow x extent before
    # rotation and hits after the 4m local-x extent rotates onto world y,
    # leaving the 1m local-y extent on world x.  Reverse the boxes to test it.
    direction = [[0.3, 0.0, 1.0]]
    wide_x = intersect_rays_with_yaw_obb(
        origin, direction, [0, 0, 5, 4, 1, 2, 0]
    )
    narrow_x = intersect_rays_with_yaw_obb(
        origin, direction, [0, 0, 5, 4, 1, 2, np.pi / 2]
    )
    assert wide_x.intersects[0]
    assert not narrow_x.intersects[0]


def test_depth_classification_partitions_support_occlusion_free_and_invalid():
    depth = np.asarray([[5.0, 3.0], [7.0, 0.0]], dtype=np.float32)
    result = classify_depth_rays(
        depth,
        [0, 0, 5, 20, 20, 2, 0],
        intrinsics(2, 2, 1.0),
        np.eye(4),
        pixel_stride=1,
        margin=0.0,
        min_depth=0.1,
        max_depth=8.0,
    )
    assert result is not None
    assert result.sample_count == 4
    assert result.support_count == 1
    assert result.occluded_count == 1
    assert result.free_space_count == 1
    assert result.invalid_count == 1
    assert result.support_ratio == pytest.approx(1.0 / 3.0)
    assert result.free_space_ratio == pytest.approx(1.0 / 3.0)
    assert result.invalid_ratio == pytest.approx(0.25)
    assert result.support_points_world.shape == (1, 3)
    assert result.support_points_world[0, 2] == pytest.approx(5.0)


def test_depth_margin_and_pixel_stride_are_respected():
    depth = np.full((5, 5), 3.95, dtype=np.float32)
    result = classify_depth_rays(
        depth,
        [0, 0, 5, 20, 20, 2, 0],
        intrinsics(5, 5, 2.0),
        np.eye(4),
        pixel_stride=2,
        margin=0.1,
    )
    assert result is not None
    assert result.sample_count == 9
    assert result.support_count == 9
    assert np.all(result.rows % 2 == 0)
    assert np.all(result.cols % 2 == 0)


def test_subpixel_projection_uses_deterministic_center_fallback():
    result = classify_depth_rays(
        np.full((9, 9), 100.0, dtype=np.float32),
        [0.2, 0.0, 100.0, 0.01, 0.01, 0.01, 0.0],
        intrinsics(9, 9, 10.0),
        np.eye(4),
        pixel_stride=4,
        min_depth=0.1,
        max_depth=200.0,
    )
    assert result is not None
    assert result.sample_count == 1
    assert result.rows.tolist() == [4]
    assert result.cols.tolist() == [4]


def test_nonfinite_depth_is_invalid_but_nonfinite_geometry_fails_closed():
    depth = np.asarray([[np.nan, np.inf], [5.0, 5.0]])
    result = classify_depth_rays(
        depth,
        [0, 0, 5, 20, 20, 2, 0],
        intrinsics(2, 2, 1.0),
        np.eye(4),
        pixel_stride=1,
    )
    assert result is not None
    assert result.invalid_count == 2
    assert result.support_count == 2

    with pytest.raises(ValueError, match="finite"):
        classify_depth_rays(
            np.ones((2, 2)),
            [0, 0, np.nan, 1, 1, 1, 0],
            intrinsics(2, 2),
            np.eye(4),
        )
    bad_pose = np.eye(4)
    bad_pose[0, 0] = 2.0
    with pytest.raises(ValueError, match="rigid"):
        classify_depth_rays(
            np.ones((2, 2)),
            [0, 0, 5, 1, 1, 1, 0],
            intrinsics(2, 2),
            bad_pose,
        )
    with pytest.raises(ValueError, match="positive integer"):
        classify_depth_rays(
            np.ones((2, 2)),
            [0, 0, 5, 1, 1, 1, 0],
            intrinsics(2, 2),
            np.eye(4),
            pixel_stride=0,
        )


def test_stable_top_k_uses_score_then_frame_then_original_index():
    selected = stable_top_k_view_indices(
        [0.8, 0.9, 0.9, 0.9, 0.7],
        3,
        frame_ids=np.asarray([5, 8, 3, 3, 1], dtype=np.int64),
        valid_mask=np.asarray([True, True, True, True, False]),
    )
    assert selected.tolist() == [2, 3, 1]
    assert not selected.flags.writeable


def test_stable_top_k_rejects_nonfinite_scores_and_bad_masks():
    with pytest.raises(ValueError, match="finite"):
        stable_top_k_view_indices([0.5, np.nan], 1)
    with pytest.raises(ValueError, match="Boolean"):
        stable_top_k_view_indices([0.5], 1, valid_mask=[1])
