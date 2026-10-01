import itertools

import numpy as np
import pytest

from boxfusion.depth_guide_geometry import (
    DEPTH_ALPHA,
    GRID_SIZE,
    MAX_BATCH_PROPOSALS,
    MAX_DEPTH_M,
    MAX_GUIDE_POINTS,
    MIN_DEPTH_M,
    MIN_GUIDE_POINTS,
    DepthGuideMetrics,
    DepthGuideSample,
    project_guide_metrics,
    sample_depth_guide_points,
    sample_depth_guide_points_batch,
)


SIGNS = np.asarray(list(itertools.product((-1.0, 1.0), repeat=3)))


def obb_corners(
    center=(0.0, 0.0, 2.0),
    size=(2.0, 2.0, 1.0),
    yaw=0.0,
):
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    return (SIGNS * (np.asarray(size) / 2.0)) @ rotation.T + np.asarray(center)


def sample_camera():
    depth = np.full((21, 21), 1.5, dtype=np.float32)
    intrinsic = np.asarray(
        [[6.0, 0.0, 10.0], [0.0, 6.0, 10.0], [0.0, 0.0, 1.0]]
    )
    return depth, intrinsic, np.eye(4)


def metric_fixture():
    depth = np.full((17, 17), 2.0, dtype=np.float64)
    intrinsic = np.asarray(
        [[4.0, 0.0, 8.0], [0.0, 4.0, 8.0], [0.0, 0.0, 1.0]]
    )
    pixels = np.asarray(
        [(u, v) for v in (4.0, 6.0, 8.0, 10.0) for u in (4.0, 6.0, 8.0, 10.0)]
    )
    rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(intrinsic).T
    points = rays * 2.0
    return points, depth, intrinsic, np.eye(4), pixels


def test_frozen_constants_and_public_result_types():
    assert GRID_SIZE == 8
    assert MIN_GUIDE_POINTS == 16
    assert MAX_GUIDE_POINTS == 64
    assert MAX_BATCH_PROPOSALS == 256
    assert MIN_DEPTH_M == 0.10
    assert MAX_DEPTH_M == 8.0
    assert DEPTH_ALPHA == 0.05

    depth, intrinsic, pose = sample_camera()
    sample = sample_depth_guide_points(
        depth, intrinsic, pose, [0, 0, 21, 21], obb_corners()
    )
    assert isinstance(sample, DepthGuideSample)
    metrics = project_guide_metrics(
        sample.points_world, depth, intrinsic, pose, [0, 0, 21, 21]
    )
    assert isinstance(metrics, DepthGuideMetrics)


def test_axis_aligned_sampler_is_fixed_grid_deterministic_and_read_only():
    depth, intrinsic, pose = sample_camera()
    corners = obb_corners()
    # Corner order is deliberately scrambled; it is not part of the API.
    corners = corners[[3, 1, 7, 0, 6, 4, 2, 5]]
    inputs = [value.copy() for value in (depth, intrinsic, pose, corners)]

    first = sample_depth_guide_points(
        depth, intrinsic, pose, [-5, -7, 30, 40], corners
    )
    second = sample_depth_guide_points(
        depth, intrinsic, pose, [-5, -7, 30, 40], corners
    )

    assert first is not None and second is not None
    assert first.sampled_cell_count == 64
    assert first.unique_pixel_count == 64
    assert first.valid_depth_count == 64
    assert len(first.points_world) == 64
    assert np.array_equal(first.pixels_xy, second.pixels_xy)
    assert np.array_equal(first.points_world, second.points_world)
    assert first.pixels_xy[0].tolist() == [7, 7]
    assert first.pixels_xy[-1].tolist() == [14, 14]
    assert len(np.unique(first.pixels_xy, axis=0)) == len(first.pixels_xy)
    for array in (
        first.pixels_xy,
        first.points_camera,
        first.points_world,
        first.intersection_polygon_xy,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 99
    for before, after in zip(inputs, (depth, intrinsic, pose, corners)):
        assert np.array_equal(before, after)


def test_rotated_obb_uses_true_polygon_intersection_and_oriented_inside_test():
    depth, intrinsic, pose = sample_camera()
    yaw = np.pi / 4.0
    corners = obb_corners(yaw=yaw)[[5, 2, 0, 7, 3, 6, 1, 4]]
    sample = sample_depth_guide_points(
        depth, intrinsic, pose, [0, 0, 21, 21], corners
    )

    assert sample is not None
    # A projected AABB would retain all 64 cells.  The exact diamond-shaped
    # projected hull removes its corners before depth lookup.
    assert MIN_GUIDE_POINTS <= sample.sampled_cell_count < 64
    assert MIN_GUIDE_POINTS <= len(sample.points_world) < 64
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    local = (sample.points_world - [0.0, 0.0, 2.0]) @ rotation
    assert np.all(np.abs(local[:, 0]) <= 1.0 + 1e-6)
    assert np.all(np.abs(local[:, 1]) <= 1.0 + 1e-6)
    assert np.all(np.abs(local[:, 2]) <= 0.5 + 1e-6)


def test_raw_box_is_clipped_against_polygon_not_projected_aabb():
    depth, intrinsic, pose = sample_camera()
    corners = obb_corners(yaw=np.pi / 4.0)
    # This keeps the upper-right quadrant of the projected diamond.  Its true
    # intersection is triangular; intersecting two AABBs would be rectangular.
    sample = sample_depth_guide_points(
        depth, intrinsic, pose, [10.0, 4.0, 16.0, 10.0], corners
    )
    assert sample is not None
    polygon = sample.intersection_polygon_xy
    assert len(polygon) == 3
    assert np.all(polygon[:, 0] >= 10.0 - 1e-8)
    assert np.all(polygon[:, 1] <= 10.0 + 1e-8)
    # Projected diamond equation for this camera/OBB front face.
    assert np.all(
        np.abs(polygon[:, 0] - 10.0) + np.abs(polygon[:, 1] - 10.0)
        <= 4.0 * np.sqrt(2.0) + 1e-7
    )


def test_sampler_filters_depth_range_and_obb_exterior_then_fails_closed():
    depth, intrinsic, pose = sample_camera()
    corners = obb_corners()
    assert sample_depth_guide_points(
        np.full_like(depth, 0.099), intrinsic, pose, [0, 0, 21, 21], corners
    ) is None
    assert sample_depth_guide_points(
        np.full_like(depth, 8.001), intrinsic, pose, [0, 0, 21, 21], corners
    ) is None
    # Metric depth remains sensor-valid but lies behind the OBB.
    assert sample_depth_guide_points(
        np.full_like(depth, 4.0), intrinsic, pose, [0, 0, 21, 21], corners
    ) is None
    # The narrow intersection collapses to fewer than 16 unique pixels.
    assert sample_depth_guide_points(
        depth, intrinsic, pose, [9.8, 9.8, 10.2, 10.2], corners
    ) is None
    # A near-plane crossing is not partially projected.
    crossing = obb_corners(center=(0, 0, 0.25), size=(1, 1, 1))
    assert sample_depth_guide_points(
        depth, intrinsic, pose, [0, 0, 21, 21], crossing
    ) is None


def test_invalid_sensor_depth_is_legal_and_filtered_per_pixel():
    depth, intrinsic, pose = sample_camera()
    depth[7, 7] = np.nan
    depth[7, 8] = np.inf
    depth[7, 9] = 0.0
    depth[7, 10] = 8.01
    sample = sample_depth_guide_points(
        depth, intrinsic, pose, [0, 0, 21, 21], obb_corners()
    )
    assert sample is not None
    assert sample.unique_pixel_count == 64
    assert sample.valid_depth_count == 60
    assert len(sample.points_world) == 60
    assert np.isfinite(sample.points_world).all()


def test_frozen_one_millimetre_near_plane_fails_closed():
    depth, intrinsic, pose = sample_camera()
    # The nearest four corners are exactly z=1e-3 and must be rejected by the
    # repository's fixed 1 mm projection convention (strictly greater only).
    corners = obb_corners(
        center=(0.0, 0.0, 0.0015), size=(0.001, 0.001, 0.001)
    )
    assert np.min(corners[:, 2]) == pytest.approx(1e-3)
    assert sample_depth_guide_points(
        depth, intrinsic, pose, [0, 0, 21, 21], corners
    ) is None


def test_sampler_depth_endpoints_are_inclusive():
    intrinsic = np.asarray(
        [[6.0, 0.0, 10.0], [0.0, 6.0, 10.0], [0.0, 0.0, 1.0]]
    )
    near_box = obb_corners(center=(0, 0, 0.15), size=(0.02, 0.02, 0.10))
    near = sample_depth_guide_points(
        np.full((21, 21), MIN_DEPTH_M), intrinsic, np.eye(4), [0, 0, 21, 21], near_box
    )
    # Its sub-pixel projection cannot meet min16; endpoint acceptance itself
    # is covered more directly by the metric path below.
    assert near is None
    points, depth, intrinsic, pose, _ = metric_fixture()
    depth.fill(MIN_DEPTH_M)
    near_points = points.copy()
    near_points[:, 2] = MIN_DEPTH_M
    near_points[:, :2] *= MIN_DEPTH_M / 2.0
    assert project_guide_metrics(near_points, depth, intrinsic, pose).v_f == 1.0
    depth.fill(MAX_DEPTH_M)
    far_points = points.copy()
    far_points[:, 2] = MAX_DEPTH_M
    far_points[:, :2] *= MAX_DEPTH_M / 2.0
    assert project_guide_metrics(far_points, depth, intrinsic, pose).v_f == 1.0


def test_perfect_projection_forward_and_backward_formulas():
    points, depth, intrinsic, pose, pixels = metric_fixture()
    metrics = project_guide_metrics(
        points,
        depth,
        intrinsic,
        pose,
        proposal_box_xyxy=[3.5, 3.5, 7.0, 10.5],
    )

    assert np.all(metrics.i_vis)
    assert np.all(metrics.w_d == 1.0)
    assert np.allclose(metrics.pixels_xy, pixels)
    assert metrics.v_f == 1.0
    assert metrics.d_f == 1.0
    assert metrics.q_f == 1.0
    assert metrics.inside_proposal is not None
    assert int(metrics.inside_proposal.sum()) == 8
    assert metrics.v_b == 0.5
    assert metrics.d_b == 1.0
    assert metrics.affinity_a == 0.5


def test_mv3dis_relative_visibility_and_depth_weight_are_exact():
    points, depth, intrinsic, pose, _ = metric_fixture()
    depth.fill(2.08)
    metrics = project_guide_metrics(points, depth, intrinsic, pose)
    expected_weight = 1.0 - 0.08 / (DEPTH_ALPHA * 2.08)

    assert np.all(metrics.i_vis)
    assert metrics.w_d == pytest.approx(np.full(16, expected_weight))
    assert metrics.v_f == 1.0
    assert metrics.d_f == pytest.approx(expected_weight)
    assert metrics.q_f == pytest.approx(expected_weight)
    assert metrics.inside_proposal is None
    assert metrics.v_b is None
    assert metrics.d_b is None
    assert metrics.affinity_a is None

    depth.fill(2.2)  # abs(2-2.2) > .05*2.2
    rejected = project_guide_metrics(points, depth, intrinsic, pose)
    assert not np.any(rejected.i_vis)
    assert np.all(rejected.w_d == 0.0)
    assert rejected.v_f == rejected.d_f == rejected.q_f == 0.0


def test_backward_denominators_follow_visible_and_inside_visible_counts():
    points, depth, intrinsic, pose, _ = metric_fixture()
    # Half the points are depth-inconsistent.  The proposal contains four of
    # the eight remaining visible points, so Vb=4/8, not 4/16.
    depth[8:, :] = 2.3
    metrics = project_guide_metrics(
        points, depth, intrinsic, pose, [3.5, 3.5, 7.0, 7.0]
    )
    assert int(metrics.i_vis.sum()) == 8
    assert int((metrics.i_vis & metrics.inside_proposal).sum()) == 4
    assert metrics.v_f == 0.5
    assert metrics.v_b == 0.5
    assert metrics.d_b == 1.0
    assert metrics.affinity_a == 0.5

    empty = project_guide_metrics(
        points, depth, intrinsic, pose, [12.0, 12.0, 16.0, 16.0]
    )
    assert empty.v_b == empty.d_b == empty.affinity_a == 0.0


def test_nearest_pixel_rule_tie_edge_clip_and_geometric_image_boundary():
    depth = np.full((3, 4), 7.0, dtype=np.float64)
    depth[1, 1] = 1.0
    depth[1, 3] = 1.0
    intrinsic = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]
    )
    # Four copies per location keep the guide within its 16-point contract.
    u_values = np.repeat([0.5, 3.9, -0.01, 4.0], 4)
    points = np.column_stack((u_values, np.zeros(16), np.ones(16)))
    metrics = project_guide_metrics(points, depth, intrinsic, np.eye(4))

    # Half-up tie: u=.5 samples x=1.  The in-image u=3.9 projection selects
    # the nearest image-edge pixel x=3 after clipping.  -0.01 and W are
    # geometrically outside even though their safe lookup indices are edges.
    assert np.all(metrics.pixel_indices_xy[:4, 0] == 1)
    assert np.all(metrics.pixel_indices_xy[4:8, 0] == 3)
    assert np.all(metrics.pixel_indices_xy[8:12, 0] == 0)
    assert np.all(metrics.pixel_indices_xy[12:, 0] == 3)
    assert metrics.i_vis.tolist() == [True] * 8 + [False] * 8
    assert metrics.v_f == 0.5


def test_T_wc_direction_maps_camera_guide_to_world_then_back_to_camera():
    camera_points, depth, intrinsic, _, pixels = metric_fixture()
    angle = np.pi / 2.0
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0.0],
         [np.sin(angle), np.cos(angle), 0.0],
         [0.0, 0.0, 1.0]]
    )
    T_wc = np.eye(4)
    T_wc[:3, :3] = rotation
    T_wc[:3, 3] = [1.5, -2.0, 0.75]
    world_points = camera_points @ rotation.T + T_wc[:3, 3]

    metrics = project_guide_metrics(
        world_points, depth, intrinsic, T_wc, [0, 0, 16, 16]
    )
    assert np.allclose(metrics.pixels_xy, pixels)
    assert np.allclose(metrics.projected_depth_m, 2.0)
    assert metrics.v_f == metrics.d_f == metrics.q_f == 1.0


def test_metric_outputs_are_copies_read_only_and_inputs_are_unchanged():
    points, depth, intrinsic, pose, _ = metric_fixture()
    snapshots = [value.copy() for value in (points, depth, intrinsic, pose)]
    metrics = project_guide_metrics(
        points, depth, intrinsic, pose, [0, 0, 16, 16]
    )
    arrays = (
        metrics.pixels_xy,
        metrics.pixel_indices_xy,
        metrics.projected_depth_m,
        metrics.measured_depth_m,
        metrics.valid_depth,
        metrics.i_vis,
        metrics.w_d,
        metrics.inside_proposal,
    )
    for array in arrays:
        assert array is not None and not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 0
    for before, after in zip(snapshots, (points, depth, intrinsic, pose)):
        assert np.array_equal(before, after)


@pytest.mark.parametrize(
    "call, message",
    [
        (
            lambda: project_guide_metrics(
                np.zeros((15, 3)), np.ones((3, 3)), np.eye(3), np.eye(4)
            ),
            "16..64",
        ),
        (
            lambda: project_guide_metrics(
                np.zeros((65, 3)), np.ones((3, 3)), np.eye(3), np.eye(4)
            ),
            "16..64",
        ),
        (
            lambda: project_guide_metrics(
                np.zeros((16, 3)), np.ones((3, 3)), np.eye(3), np.eye(4), alpha=0
            ),
            "alpha",
        ),
        (
            lambda: sample_depth_guide_points(
                np.ones((3, 3), dtype=bool), np.eye(3), np.eye(4), [0, 0, 2, 2], obb_corners()
            ),
            "numeric",
        ),
        (
            lambda: sample_depth_guide_points(
                np.ones((3, 3)), np.eye(3), np.eye(4), [2, 0, 1, 2], obb_corners()
            ),
            "x2>x1",
        ),
        (
            lambda: sample_depth_guide_points(
                np.ones((3, 3)), np.eye(3), np.diag([1, 1, -1, 1]), [0, 0, 2, 2], obb_corners()
            ),
            "proper rigid",
        ),
    ],
)
def test_strict_validation(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def test_non_obb_corners_are_rejected_instead_of_using_an_aabb_approximation():
    depth, intrinsic, pose = sample_camera()
    malformed = obb_corners()
    malformed[0] += [0.2, 0.1, 0.0]
    with pytest.raises(ValueError, match="does not define|faces must|axes must"):
        sample_depth_guide_points(
            depth, intrinsic, pose, [0, 0, 21, 21], malformed
        )


def test_real_float32_translated_boxfusion_corners_are_not_split_into_extra_faces():
    # A real fixed-10 BoxFusion output.  The former world-offset plane
    # de-duplication split this valid OBB into eight faces because float32
    # normal noise was multiplied by its ~6 m world translation.
    corners = np.asarray(
        [
            [3.40206194, 5.75479841, 0.99720836],
            [3.24027348, 6.72733593, 0.99720842],
            [3.24027371, 6.72733641, 0.07969195],
            [3.40206194, 5.75479889, 0.07969189],
            [2.51704240, 5.60756922, 0.99720818],
            [2.35525417, 6.58010674, 0.99720824],
            [2.35525417, 6.58010721, 0.07969177],
            [2.51704264, 5.60756969, 0.07969171],
        ],
        dtype=np.float32,
    )
    center = corners.mean(axis=0, dtype=np.float64)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = center - [0.0, 0.0, 2.0]
    intrinsic = np.asarray(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
    )
    sample = sample_depth_guide_points(
        np.full((101, 101), 2.0, dtype=np.float32),
        intrinsic,
        pose,
        [0.0, 0.0, 101.0, 101.0],
        corners,
    )
    assert sample is not None
    assert len(sample.points_world) >= MIN_GUIDE_POINTS


def test_structured_batch_matches_corner_sampler_and_preserves_order():
    depth, intrinsic, pose = sample_camera()
    centers = np.asarray([[0.0, 0.0, 2.0], [0.25, -0.1, 2.0]])
    dimensions = np.asarray([[2.0, 2.0, 1.0], [1.2, 0.8, 1.4]])
    rotations = []
    corners = []
    for center, dimension, yaw in zip(centers, dimensions, (0.4, -0.25)):
        cosine, sine = np.cos(yaw), np.sin(yaw)
        rotation = np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
        )
        rotations.append(rotation)
        corners.append((SIGNS * (dimension / 2.0)) @ rotation.T + center)
    rotations = np.asarray(rotations)
    boxes = np.asarray([[0.0, 0.0, 21.0, 21.0]] * 2)

    batch = sample_depth_guide_points_batch(
        depth, intrinsic, pose, boxes, centers, dimensions, rotations
    )
    scalar = tuple(
        sample_depth_guide_points(depth, intrinsic, pose, box, corner)
        for box, corner in zip(boxes, corners)
    )

    assert len(batch) == len(scalar) == 2
    for fast, reference in zip(batch, scalar):
        assert fast is not None and reference is not None
        assert np.array_equal(fast.pixels_xy, reference.pixels_xy)
        assert np.array_equal(fast.points_camera, reference.points_camera)
        assert np.array_equal(fast.points_world, reference.points_world)
        assert np.allclose(
            fast.intersection_polygon_xy,
            reference.intersection_polygon_xy,
            rtol=0.0,
            atol=1e-12,
        )
        assert fast.sampled_cell_count == reference.sampled_cell_count
        assert fast.unique_pixel_count == reference.unique_pixel_count
        assert fast.valid_depth_count == reference.valid_depth_count
        assert not fast.points_world.flags.writeable


def test_structured_batch_handles_float32_full_rotation_and_nonidentity_T_wc():
    def rotation_x(angle):
        cosine, sine = np.cos(angle), np.sin(angle)
        return np.asarray(
            [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]]
        )

    def rotation_y(angle):
        cosine, sine = np.cos(angle), np.sin(angle)
        return np.asarray(
            [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
        )

    def rotation_z(angle):
        cosine, sine = np.cos(angle), np.sin(angle)
        return np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
        )

    camera_rotation = rotation_z(0.31) @ rotation_y(-0.22) @ rotation_x(0.17)
    pose_rotation = rotation_z(-0.7) @ rotation_y(0.15) @ rotation_x(-0.12)
    pose = np.eye(4)
    pose[:3, :3] = pose_rotation
    pose[:3, 3] = [4.6, 3.2, 1.4]
    camera_center = np.asarray([0.0, 0.0, 2.0])
    world_center = camera_center @ pose_rotation.T + pose[:3, 3]
    world_rotation = pose_rotation @ camera_rotation
    dimensions = np.asarray([1.2, 0.8, 0.6])
    corners = (
        (SIGNS * (dimensions / 2.0)) @ world_rotation.T + world_center
    ).astype(np.float32)
    intrinsic = np.asarray(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
    )
    depth = np.full((101, 101), 2.0, dtype=np.float32)
    box = [0.0, 0.0, 101.0, 101.0]

    reference = sample_depth_guide_points(
        depth, intrinsic, pose, box, corners
    )
    fast = sample_depth_guide_points_batch(
        depth,
        intrinsic,
        pose,
        [box],
        np.asarray([world_center], dtype=np.float32),
        np.asarray([dimensions], dtype=np.float32),
        np.asarray([world_rotation], dtype=np.float32),
    )[0]
    assert reference is not None and fast is not None
    assert np.array_equal(fast.pixels_xy, reference.pixels_xy)
    assert np.allclose(fast.points_world, reference.points_world, atol=1e-7)
    local = (fast.points_world - world_center) @ world_rotation
    assert np.all(np.abs(local) <= dimensions / 2.0 + 2e-6)


def test_structured_batch_is_bounded_aligned_and_fail_closed():
    depth, intrinsic, pose = sample_camera()
    empty = sample_depth_guide_points_batch(
        depth,
        intrinsic,
        pose,
        np.empty((0, 4)),
        np.empty((0, 3)),
        np.empty((0, 3)),
        np.empty((0, 3, 3)),
    )
    assert empty == ()

    with pytest.raises(ValueError, match="same length"):
        sample_depth_guide_points_batch(
            depth,
            intrinsic,
            pose,
            [[0, 0, 20, 20]],
            np.empty((0, 3)),
            [[1, 1, 1]],
            [np.eye(3)],
        )
    with pytest.raises(ValueError, match="proper rigid rotations"):
        sample_depth_guide_points_batch(
            depth,
            intrinsic,
            pose,
            [[0, 0, 20, 20]],
            [[0, 0, 2]],
            [[1, 1, 1]],
            [np.diag([1.0, 1.0, -1.0])],
        )
    with pytest.raises(ValueError, match="must not exceed"):
        count = MAX_BATCH_PROPOSALS + 1
        sample_depth_guide_points_batch(
            depth,
            intrinsic,
            pose,
            np.tile([[0, 0, 20, 20]], (count, 1)),
            np.tile([[0, 0, 2]], (count, 1)),
            np.tile([[1, 1, 1]], (count, 1)),
            np.tile(np.eye(3), (count, 1, 1)),
        )
