import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "boxfusion"
    / "object_memory.py"
)
SPEC = importlib.util.spec_from_file_location(
    "boxfusion_object_memory", SOURCE
)
object_memory = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = object_memory
SPEC.loader.exec_module(object_memory)


def resolved_config(**overrides):
    base = {
        "mask_edge_margin": 0,
        "depth_edge_threshold": None,
        "voxel_size": 0.0,
        "max_points_per_observation": 128,
        "max_points_per_object": 256,
        "aabb_lower_quantile": 0.0,
        "aabb_upper_quantile": 1.0,
        "min_points_for_aabb": 4,
        "minimum_aabb_dimension": 0.01,
    }
    base.update(overrides)
    return object_memory.resolve_object_memory_config(base)


def camera_matrix(focal=100.0, center=50.0):
    return np.asarray(
        [
            [focal, 0.0, center],
            [0.0, focal, center],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def cube_points(center, half_extent=0.5):
    center = np.asarray(center, dtype=np.float32)
    signs = np.asarray(
        list(itertools.product((-1.0, 1.0), repeat=3)),
        dtype=np.float32,
    )
    return center[None, :] + signs * float(half_extent)


def observation(center, half_extent=0.5, **kwargs):
    return object_memory.ObjectObservation(
        cube_points(center, half_extent),
        **kwargs,
    )


def manager_config(**overrides):
    values = {
        "mask_edge_margin": 0,
        "depth_edge_threshold": None,
        "voxel_size": 0.0,
        "max_points_per_observation": 64,
        "max_points_per_object": 128,
        "aabb_lower_quantile": 0.0,
        "aabb_upper_quantile": 1.0,
        "min_points_for_aabb": 8,
        "minimum_aabb_dimension": 0.01,
        "min_confirmations": 2,
        "track_ttl": 2,
        "association_iou_threshold": 0.05,
        "association_center_distance": 1.0,
        "association_inside_fraction": 0.25,
    }
    values.update(overrides)
    return object_memory.resolve_object_memory_config(values)


def test_default_config_is_safe_and_scannet_metric():
    config = object_memory.resolve_object_memory_config()
    assert config["enabled"] is False
    assert config["depth_scale"] == 1.0
    assert config["min_confirmations"] == 2
    assert config["aabb_lower_quantile"] < config["aabb_upper_quantile"]
    assert object_memory.ObjectMemory is object_memory.ObjectGeometryMemory
    assert object_memory.TrackManager is object_memory.CandidateTrackManager


@pytest.mark.parametrize(
    "override",
    [
        {"enabled": 1},
        {"min_depth": -0.1},
        {"max_depth": 0.05},
        {"depth_scale": 0.0},
        {"mask_threshold": 1.1},
        {"mask_edge_margin": -1},
        {"depth_edge_threshold": 0.0},
        {"voxel_size": -0.1},
        {"max_points_per_observation": 0},
        {"max_points_per_object": 0},
        {"aabb_lower_quantile": 0.99, "aabb_upper_quantile": 0.98},
        {"minimum_aabb_dimension": 0.0},
        {"min_confirmations": 1},
        {"track_ttl": -1},
        {"association_iou_threshold": 1.1},
        {"association_center_distance": 0.0},
        {"association_inside_fraction": -0.1},
        {"min_depth": np.nan},
    ],
)
def test_invalid_config_fails_fast(override):
    with pytest.raises(ValueError):
        object_memory.resolve_object_memory_config(override)


def test_unknown_config_key_is_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        object_memory.resolve_object_memory_config({"max_pointz": 10})


def test_mask_resize_is_exact_nearest_neighbour():
    mask = np.asarray([[1, 0], [0, 1]], dtype=np.uint8)
    resized = object_memory.resize_mask_nearest(mask, (4, 4))
    expected = np.asarray(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(resized, expected)
    assert resized.dtype == np.bool_


def test_mask_resize_threshold_and_finite_validation():
    mask = np.asarray([[0.49, 0.50], [0.9, 0.1]], dtype=np.float32)
    np.testing.assert_array_equal(
        object_memory.resize_mask_nearest(mask, (2, 2)),
        [[False, True], [True, False]],
    )
    mask[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        object_memory.resize_mask_nearest(mask, (2, 2))


def test_mask_edge_erosion_removes_boundary_band():
    mask = np.ones((5, 5), dtype=bool)
    eroded = object_memory.erode_mask_edges(mask, 1)
    expected = np.zeros((5, 5), dtype=bool)
    expected[1:4, 1:4] = True
    np.testing.assert_array_equal(eroded, expected)


def test_depth_discontinuity_marks_both_sides():
    depth = np.asarray(
        [[1.0, 1.0, 3.0], [1.0, 1.0, 3.0]],
        dtype=np.float32,
    )
    edges = object_memory.depth_discontinuity_mask(depth, 0.5)
    np.testing.assert_array_equal(
        edges,
        [[False, True, True], [False, True, True]],
    )


def test_real_depth_backprojection_and_world_transform_are_exact():
    depth_mm = np.full((2, 2), 2000, dtype=np.uint16)
    mask = np.asarray([[1, 0], [0, 1]], dtype=bool)
    intrinsics = np.asarray(
        [[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    camera_to_world = np.eye(4, dtype=np.float32)
    camera_to_world[:3, 3] = [1.0, 2.0, 3.0]
    result = object_memory.extract_masked_world_points(
        depth_mm,
        mask,
        intrinsics,
        camera_to_world,
        resolved_config(depth_scale=1000.0),
    )
    np.testing.assert_allclose(
        result.points_world,
        [[0.5, 1.5, 5.0], [1.5, 2.5, 5.0]],
        atol=1e-6,
    )
    assert result.mask_pixels == 2
    assert result.valid_depth_pixels == 2
    assert result.median_depth == 2.0


def test_mask_is_resized_to_depth_before_backprojection():
    result = object_memory.extract_masked_world_points(
        np.ones((4, 4), dtype=np.float32),
        np.asarray([[1, 0], [0, 0]], dtype=bool),
        np.eye(3),
        np.eye(4),
        resolved_config(min_depth=0.1, max_depth=2.0),
    )
    assert result.mask_pixels == 4
    assert result.valid_depth_pixels == 4


def test_invalid_and_out_of_range_depths_are_filtered_not_projected():
    depth = np.asarray(
        [[0.05, 1.0, np.nan], [np.inf, 7.0, 2.0]],
        dtype=np.float32,
    )
    result = object_memory.extract_masked_world_points(
        depth,
        np.ones_like(depth, dtype=bool),
        np.eye(3),
        np.eye(4),
        resolved_config(min_depth=0.1, max_depth=6.0),
    )
    assert result.valid_depth_pixels == 2
    assert result.points_world.shape == (2, 3)
    assert np.isfinite(result.points_world).all()


def test_mask_and_depth_edges_are_filtered_in_extraction():
    depth = np.tile(
        np.asarray([1.0, 1.0, 3.0], dtype=np.float32),
        (3, 1),
    )
    result = object_memory.extract_masked_world_points(
        depth,
        np.ones_like(depth, dtype=bool),
        np.eye(3),
        np.eye(4),
        resolved_config(depth_edge_threshold=0.5),
    )
    assert result.valid_depth_pixels == 3
    np.testing.assert_array_equal(
        result.valid_pixel_mask,
        np.asarray(
            [
                [True, False, False],
                [True, False, False],
                [True, False, False],
            ]
        ),
    )

    eroded = object_memory.extract_masked_world_points(
        np.ones((5, 5), dtype=np.float32),
        np.ones((5, 5), dtype=bool),
        np.eye(3),
        np.eye(4),
        resolved_config(mask_edge_margin=1),
    )
    assert eroded.valid_depth_pixels == 9


def test_empty_valid_depth_returns_well_formed_empty_observation():
    result = object_memory.extract_masked_world_points(
        np.zeros((3, 3), dtype=np.float32),
        np.ones((3, 3), dtype=bool),
        np.eye(3),
        np.eye(4),
        resolved_config(),
    )
    assert result.points_world.shape == (0, 3)
    assert result.points_world.dtype == np.float32
    assert result.valid_depth_pixels == 0
    assert result.median_depth is None
    assert result.valid_depth_ratio == 0.0


@pytest.mark.parametrize(
    "intrinsics,pose",
    [
        (np.zeros((3, 3)), np.eye(4)),
        (np.eye(3), np.full((4, 4), np.nan)),
        (np.eye(3), np.zeros((4, 4))),
    ],
)
def test_invalid_camera_geometry_is_rejected(intrinsics, pose):
    with pytest.raises(ValueError):
        object_memory.extract_masked_world_points(
            np.ones((2, 2)),
            np.ones((2, 2)),
            intrinsics,
            pose,
            resolved_config(),
        )


def test_voxel_downsample_returns_deterministic_centroids():
    points = np.asarray(
        [
            [0.01, 0.0, 0.0],
            [0.03, 0.0, 0.0],
            [0.11, 0.0, 0.0],
        ]
    )
    downsampled = object_memory.voxel_downsample(points, 0.1)
    np.testing.assert_allclose(
        downsampled,
        [[0.02, 0.0, 0.0], [0.11, 0.0, 0.0]],
    )


def test_bounded_sampling_is_deterministic_and_permutation_invariant():
    points = np.column_stack(
        (
            np.arange(20, dtype=np.float32),
            np.zeros(20),
            np.zeros(20),
        )
    )
    first = object_memory.deterministic_bounded_sample(points, 5)
    second = object_memory.deterministic_bounded_sample(points[::-1], 5)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:, 0], [0, 4, 9, 14, 19])


def test_robust_quantile_aabb_ignores_extreme_outlier():
    points = np.column_stack(
        (
            np.asarray(list(range(10)) + [1000], dtype=np.float32),
            np.zeros(11, dtype=np.float32),
            np.ones(11, dtype=np.float32),
        )
    )
    center, dims = object_memory.robust_quantile_aabb(
        points,
        lower_quantile=0.1,
        upper_quantile=0.9,
        min_points=4,
        minimum_dimension=0.1,
    )
    np.testing.assert_allclose(center, [5.0, 0.0, 1.0])
    np.testing.assert_allclose(dims, [8.0, 0.1, 0.1])


def test_aabb_iou_is_exact_symmetric_and_zero_when_disjoint():
    center_a = np.asarray([0.0, 0.0, 0.0])
    center_b = np.asarray([1.0, 0.0, 0.0])
    dims = np.asarray([2.0, 2.0, 2.0])
    assert object_memory.aabb_iou(center_a, dims, center_a, dims) == 1.0
    assert object_memory.aabb_iou(center_a, dims, center_b, dims) == pytest.approx(
        1.0 / 3.0
    )
    assert object_memory.aabb_iou(center_b, dims, center_a, dims) == pytest.approx(
        1.0 / 3.0
    )
    assert object_memory.aabb_iou(center_a, dims, [10, 0, 0], dims) == 0.0


def test_points_inside_aabb_is_inclusive_and_fraction_is_exact():
    points = np.asarray(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.01, 0.0, 0.0],
        ]
    )
    inside = object_memory.points_inside_aabb(
        points,
        [0, 0, 0],
        [2, 2, 2],
    )
    np.testing.assert_array_equal(inside, [True, True, True, False])
    assert object_memory.points_inside_aabb_fraction(
        points, [0, 0, 0], [2, 2, 2]
    ) == 0.75
    assert object_memory.points_inside_aabb_fraction(
        np.empty((0, 3)), [0, 0, 0], [2, 2, 2]
    ) == 0.0


def test_nonfinite_points_and_invalid_aabb_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        object_memory.robust_quantile_aabb([[np.nan, 0.0, 0.0]])
    with pytest.raises(ValueError, match="positive"):
        object_memory.aabb_iou(
            [0, 0, 0],
            [1, 0, 1],
            [0, 0, 0],
            [1, 1, 1],
        )


def test_project_aabb_to_image_is_exact_and_pose_aware():
    expected = [50.0 / 3.0, 50.0 / 3.0, 250.0 / 3.0, 250.0 / 3.0]
    projected = object_memory.project_aabb_to_image(
        [0.0, 0.0, 2.0],
        [1.0, 1.0, 1.0],
        camera_matrix(),
        np.eye(4),
        (100, 100),
    )
    np.testing.assert_allclose(projected, expected, atol=1e-5)

    camera_to_world = np.eye(4)
    camera_to_world[0, 3] = 1.0
    shifted = object_memory.project_aabb_to_image(
        [1.0, 0.0, 2.0],
        [1.0, 1.0, 1.0],
        camera_matrix(),
        camera_to_world,
        (100, 100),
    )
    np.testing.assert_allclose(shifted, expected, atol=1e-5)


def test_project_aabb_rejects_behind_or_near_plane_crossing():
    assert (
        object_memory.project_aabb_to_image(
            [0, 0, -2],
            [1, 1, 1],
            camera_matrix(),
            np.eye(4),
            (100, 100),
        )
        is None
    )
    assert (
        object_memory.project_aabb_to_image(
            [0, 0, 0],
            [1, 1, 1],
            camera_matrix(),
            np.eye(4),
            (100, 100),
        )
        is None
    )


def test_bbox_mask_iou_is_exact():
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True
    assert object_memory.bbox_mask_iou([0, 0, 2, 2], mask) == 1.0
    assert object_memory.bbox_mask_iou([1, 0, 3, 2], mask) == pytest.approx(
        1.0 / 3.0
    )
    assert object_memory.bbox_mask_iou([3, 3, 4, 4], mask) == 0.0


def test_projected_aabb_mask_iou_matches_its_rasterized_box():
    mask = np.zeros((100, 100), dtype=bool)
    mask[16:84, 16:84] = True
    score = object_memory.projected_aabb_mask_iou(
        [0.0, 0.0, 2.0],
        [1.0, 1.0, 1.0],
        camera_matrix(),
        np.eye(4),
        mask,
    )
    assert score == 1.0


def test_optional_torch_inputs_are_accepted_without_gpu():
    torch = pytest.importorskip("torch")
    result = object_memory.extract_masked_world_points(
        torch.ones((1, 2, 2), dtype=torch.float32),
        torch.ones((1, 1, 1), dtype=torch.float32),
        torch.eye(3),
        torch.eye(4),
        resolved_config(),
    )
    assert result.points_world.shape == (4, 3)
    assert isinstance(result.points_world, np.ndarray)


def test_object_memory_enforces_both_point_budgets_deterministically():
    config = resolved_config(
        max_points_per_observation=5,
        max_points_per_object=8,
        min_points_for_aabb=4,
    )
    first_points = np.column_stack(
        (np.arange(20), np.zeros(20), np.zeros(20))
    )
    second_points = np.column_stack(
        (np.arange(20, 40), np.zeros(20), np.zeros(20))
    )

    first = object_memory.ObjectGeometryMemory(3, config)
    second = object_memory.ObjectGeometryMemory(3, config)
    for memory in (first, second):
        memory.add_observation(first_points, 0)
        assert memory.num_points == 5
        memory.add_observation(second_points, 1)
        assert memory.num_points == 8
    np.testing.assert_array_equal(first.points, second.points)


def test_object_memory_tracks_multi_observation_quality_statistics():
    memory = object_memory.ObjectGeometryMemory(
        0,
        resolved_config(max_points_per_object=32),
    )
    memory.add_observation(
        object_memory.ObjectObservation(
            cube_points([0, 0, 0]),
            confidence=0.8,
            mask_pixels=10,
            valid_depth_pixels=5,
            projection_mask_iou=0.5,
        ),
        4,
    )
    memory.add_observation(
        object_memory.ObjectObservation(
            cube_points([0.1, 0, 0]),
            confidence=1.0,
            mask_pixels=20,
            valid_depth_pixels=20,
            projection_mask_iou=1.0,
        ),
        5,
    )
    summary = memory.quality_summary()
    assert summary["observations"] == 2
    assert summary["unique_views"] == 2
    assert summary["total_mask_pixels"] == 30
    assert summary["total_valid_depth_pixels"] == 25
    assert summary["aggregate_valid_depth_ratio"] == pytest.approx(25.0 / 30.0)
    assert summary["mean_confidence"] == pytest.approx(0.9)
    assert summary["mean_valid_depth_ratio"] == pytest.approx(0.75)
    assert summary["mean_projection_mask_iou"] == pytest.approx(0.75)
    assert summary["mean_quality"] == pytest.approx(0.6)


def test_same_frame_observations_do_not_count_as_multiple_views():
    memory = object_memory.ObjectGeometryMemory(0, resolved_config())
    memory.add_observation(cube_points([0, 0, 0]), 7)
    memory.add_observation(cube_points([0.1, 0, 0]), 7)
    assert memory.observation_count == 2
    assert memory.unique_view_count == 1
    with pytest.raises(ValueError, match="non-decreasing"):
        memory.add_observation(cube_points([0, 0, 0]), 6)


def test_memory_points_are_defensive_copy_and_aabb_is_available():
    memory = object_memory.ObjectGeometryMemory(0, resolved_config())
    memory.add_observation(cube_points([2, 3, 4]), 0)
    center, dims = memory.aabb
    np.testing.assert_allclose(center, [2, 3, 4])
    np.testing.assert_allclose(dims, [1, 1, 1])
    external = memory.points
    external[:] = 1000
    assert not np.all(memory.points == 1000)


def test_candidate_requires_two_distinct_views_for_confirmation():
    manager = object_memory.CandidateTrackManager(manager_config())
    first = manager.update([observation([0, 0, 0])], frame_id=0)
    assert first.assignments == {0: 0}
    assert first.created_track_ids == (0,)
    assert manager.tracks[0].confirmed is False

    second = manager.update([observation([0.05, 0, 0])], frame_id=1)
    assert second.assignments == {0: 0}
    assert second.created_track_ids == ()
    assert second.newly_confirmed_track_ids == (0,)
    assert manager.tracks[0].confirmed is True
    assert tuple(track.track_id for track in manager.confirmed_tracks()) == (0,)


def test_repeated_update_in_same_frame_does_not_confirm_candidate():
    manager = object_memory.CandidateTrackManager(manager_config())
    manager.update([observation([0, 0, 0])], frame_id=0)
    result = manager.update([observation([0.02, 0, 0])], frame_id=0)
    assert result.assignments == {0: 0}
    assert result.newly_confirmed_track_ids == ()
    assert manager.tracks[0].view_count == 1
    assert manager.tracks[0].confirmed is False


def test_track_ttl_expires_only_after_allowed_gap():
    manager = object_memory.CandidateTrackManager(
        manager_config(track_ttl=2)
    )
    manager.update([observation([0, 0, 0])], frame_id=0)
    at_boundary = manager.update([], frame_id=2)
    assert at_boundary.expired_track_ids == ()
    assert 0 in manager.tracks
    expired = manager.update([], frame_id=3)
    assert expired.expired_track_ids == (0,)
    assert 0 not in manager.tracks


def test_confirmed_track_is_frozen_in_archive_after_ttl():
    manager = object_memory.CandidateTrackManager(
        manager_config(track_ttl=1),
        archive_confirmed=True,
    )
    manager.update([observation([0, 0, 0])], frame_id=0)
    manager.update([observation([0.02, 0, 0])], frame_id=1)
    assert manager.tracks[0].confirmed is True

    at_boundary = manager.update([], frame_id=2)
    assert at_boundary.expired_track_ids == ()
    expired = manager.update([], frame_id=3)

    assert expired.expired_track_ids == (0,)
    assert expired.archived_track_ids == (0,)
    assert expired.discarded_track_ids == ()
    assert manager.tracks == {}
    assert manager.archived_tracks[0].confirmed is True
    assert [
        track.track_id
        for track in manager.confirmed_tracks(include_archived=True)
    ] == [0]


def test_unconfirmed_track_is_discarded_instead_of_archived():
    manager = object_memory.CandidateTrackManager(
        manager_config(track_ttl=0),
        archive_confirmed=True,
    )
    manager.update([observation([0, 0, 0])], frame_id=0)
    expired = manager.update([], frame_id=1)

    assert expired.expired_track_ids == (0,)
    assert expired.archived_track_ids == ()
    assert expired.discarded_track_ids == (0,)
    assert manager.tracks == {}
    assert manager.archived_tracks == {}


def test_archive_flags_are_strict_booleans():
    with pytest.raises(ValueError, match="archive_confirmed"):
        object_memory.CandidateTrackManager(
            manager_config(),
            archive_confirmed=1,
        )
    manager = object_memory.CandidateTrackManager(manager_config())
    with pytest.raises(ValueError, match="include_archived"):
        manager.confirmed_tracks(include_archived=1)


def test_far_geometry_creates_a_new_track():
    manager = object_memory.CandidateTrackManager(manager_config())
    manager.update([observation([0, 0, 0])], frame_id=0)
    result = manager.update([observation([10, 0, 0])], frame_id=1)
    assert result.assignments == {0: 1}
    assert result.created_track_ids == (1,)
    assert sorted(manager.tracks) == [0, 1]


def test_inside_fraction_can_associate_contained_view_with_low_iou():
    manager = object_memory.CandidateTrackManager(
        manager_config(
            association_iou_threshold=0.20,
            association_inside_fraction=0.80,
        )
    )
    manager.update(
        [observation([0, 0, 0], half_extent=1.0)],
        frame_id=0,
    )
    result = manager.update(
        [observation([0, 0, 0], half_extent=0.2)],
        frame_id=1,
    )
    assert result.assignments == {0: 0}
    assert result.created_track_ids == ()


def test_association_is_one_to_one_with_deterministic_observation_tie_break():
    manager = object_memory.CandidateTrackManager(manager_config())
    manager.update([observation([0, 0, 0])], frame_id=0)
    result = manager.update(
        [observation([0, 0, 0]), observation([0, 0, 0])],
        frame_id=1,
    )
    assert result.assignments == {0: 0, 1: 1}
    assert result.created_track_ids == (1,)
    assert manager.tracks[0].confirmed is True
    assert manager.tracks[1].confirmed is False


def test_tied_track_association_uses_lower_track_id():
    manager = object_memory.CandidateTrackManager(
        manager_config(association_iou_threshold=0.0)
    )
    initial = manager.update(
        [
            observation([-1, 0, 0], half_extent=0.5),
            observation([1, 0, 0], half_extent=0.5),
        ],
        frame_id=0,
    )
    assert initial.created_track_ids == (0, 1)
    result = manager.update(
        [observation([0, 0, 0], half_extent=1.5)],
        frame_id=1,
    )
    assert result.assignments == {0: 0}


def test_too_few_points_are_skipped_without_creating_track():
    manager = object_memory.CandidateTrackManager(manager_config())
    result = manager.update(
        [np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]])],
        frame_id=0,
    )
    assert result.assignments == {}
    assert result.skipped_observation_indices == (0,)
    assert manager.tracks == {}


def test_track_manager_rejects_nonfinite_observation_and_time_reversal():
    manager = object_memory.CandidateTrackManager(manager_config())
    with pytest.raises(ValueError, match="finite"):
        manager.update([[[np.nan, 0.0, 0.0]]], frame_id=0)
    manager.update([], frame_id=2)
    with pytest.raises(ValueError, match="non-decreasing"):
        manager.update([], frame_id=1)


def test_lifecycle_clock_is_independent_and_failed_update_is_atomic():
    manager = object_memory.CandidateTrackManager(manager_config())
    manager.update([], frame_id=1, lifecycle_step=1)

    with pytest.raises(ValueError, match="lifecycle_step"):
        manager.update([], frame_id=2, lifecycle_step=0)

    assert manager.last_update_frame == 1
    assert manager.last_lifecycle_step == 1
    manager.update([], frame_id=2, lifecycle_step=2)
    assert manager.last_update_frame == 2
    assert manager.last_lifecycle_step == 2
