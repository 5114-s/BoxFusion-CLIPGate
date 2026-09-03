import numpy as np
import pytest

from boxfusion.mask_rgbd_point_cleaner import (
    adaptive_erosion_margin,
    clean_mask_rgbd_points,
    radius_neighbor_mask,
    resolve_mask_rgbd_point_cleaner_config,
    statistical_inlier_mask,
)


def test_adaptive_margin_tracks_object_scale_and_is_clamped():
    small = np.zeros((32, 32), dtype=bool)
    small[12:20, 12:20] = True
    large = np.zeros((32, 32), dtype=bool)
    large[4:28, 4:28] = True
    small_margin = adaptive_erosion_margin(
        small, minimum_margin=0, maximum_margin=4, radius_fraction=0.08
    )
    large_margin = adaptive_erosion_margin(
        large, minimum_margin=0, maximum_margin=4, radius_fraction=0.08
    )
    assert 0 <= small_margin <= large_margin <= 4


def test_depth_boundary_can_increase_adaptive_margin():
    mask = np.zeros((24, 24), dtype=bool)
    mask[4:20, 4:20] = True
    clean_depth = np.ones((24, 24), dtype=np.float32)
    bad_depth = clean_depth.copy()
    bad_depth[:, :4] = 2.0
    clean = adaptive_erosion_margin(
        mask,
        clean_depth,
        radius_fraction=0.0,
        depth_edge_weight=12.0,
        maximum_margin=4,
    )
    bad = adaptive_erosion_margin(
        mask,
        bad_depth,
        radius_fraction=0.0,
        depth_edge_weight=12.0,
        maximum_margin=4,
    )
    assert bad >= clean


def test_radius_filter_removes_an_isolated_point():
    cluster = np.asarray(
        [
            [0.00, 0.00, 0.00],
            [0.01, 0.00, 0.00],
            [0.00, 0.01, 0.00],
            [0.01, 0.01, 0.00],
        ],
        dtype=np.float64,
    )
    points = np.concatenate((cluster, [[2.0, 2.0, 2.0]]), axis=0)
    mask, counts = radius_neighbor_mask(
        points, radius=0.03, minimum_neighbors=3
    )
    assert mask.tolist() == [True, True, True, True, False]
    assert counts[-1] == 1


def test_statistical_filter_removes_far_outlier():
    line = np.column_stack(
        (
            np.linspace(0.0, 0.09, 10),
            np.zeros(10),
            np.zeros(10),
        )
    )
    points = np.concatenate((line, [[4.0, 4.0, 4.0]]), axis=0)
    mask, distances, threshold = statistical_inlier_mask(
        points, k=3, std_ratio=1.0
    )
    assert not bool(mask[-1])
    assert distances[-1] > threshold


def test_cleaner_is_deterministic_permutation_invariant_and_readonly():
    rng = np.random.default_rng(4)
    cluster = rng.normal(0.0, 0.01, size=(64, 3))
    points = np.concatenate((cluster, [[1.5, 1.5, 1.5]]), axis=0)
    config = {
        "radius_filter_enabled": True,
        "radius": 0.04,
        "minimum_neighbors": 3,
        "statistical_filter_enabled": True,
        "statistical_k": 5,
        "statistical_std_ratio": 1.5,
        "maximum_input_points": 128,
        "minimum_points": 8,
    }
    first = clean_mask_rgbd_points(points, config)
    second = clean_mask_rgbd_points(points[rng.permutation(len(points))], config)
    first_sorted = first.points[
        np.lexsort((first.points[:, 2], first.points[:, 1], first.points[:, 0]))
    ]
    second_sorted = second.points[
        np.lexsort(
            (second.points[:, 2], second.points[:, 1], second.points[:, 0])
        )
    ]
    np.testing.assert_array_equal(first_sorted, second_sorted)
    assert not first.points.flags.writeable
    with pytest.raises(ValueError):
        first.points[0, 0] = 2.0


def test_cleaner_fails_open_when_filter_is_too_aggressive():
    points = np.eye(3, dtype=np.float64)
    result = clean_mask_rgbd_points(
        points,
        {
            "radius_filter_enabled": True,
            "radius": 1e-4,
            "minimum_neighbors": 2,
            "minimum_points": 3,
            "maximum_input_points": 8,
        },
    )
    assert result.radius_fallback
    np.testing.assert_array_equal(result.points, points.astype(np.float32))


def test_config_rejects_unknown_and_invalid_values():
    with pytest.raises(ValueError, match="Unknown"):
        resolve_mask_rgbd_point_cleaner_config({"typo": True})
    with pytest.raises(ValueError, match="positive"):
        resolve_mask_rgbd_point_cleaner_config({"radius": 0.0})
