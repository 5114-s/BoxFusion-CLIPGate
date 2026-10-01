"""CPU contracts for deterministic reusable voxel components."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from boxfusion.voxel_components import (
    build_voxel_components,
    select_densest_component,
    select_inside_anchor,
    select_largest_component,
)


def _build(points, **updates):
    values = {
        "origin": np.zeros(3, dtype=np.float64),
        "voxel_size": 1.0,
        "boundary_epsilon": 1e-9,
        "neighbor_radius": 1,
        "dilation_radius": 0,
    }
    values.update(updates)
    return build_voxel_components(points, **values)


def _voxel_points(key, count, *, spread=0.12):
    key = np.asarray(key, dtype=np.float64)
    points = []
    for index in range(count):
        offset = np.asarray(
            [
                0.10 + spread * ((index * 3) % 7) / 7.0,
                0.12 + spread * ((index * 5) % 11) / 11.0,
                0.14 + spread * ((index * 7) % 13) / 13.0,
            ],
            dtype=np.float64,
        )
        points.append(key + offset)
    return np.asarray(points, dtype=np.float64)


def _assert_component_sets_equal(first, second):
    np.testing.assert_array_equal(first.points, second.points)
    if first.point_view_ids is None:
        assert second.point_view_ids is None
    else:
        np.testing.assert_array_equal(
            first.point_view_ids, second.point_view_ids
        )
    np.testing.assert_array_equal(first.voxel_keys, second.voxel_keys)
    np.testing.assert_array_equal(
        first.point_to_voxel, second.point_to_voxel
    )
    np.testing.assert_array_equal(
        first.point_component_ids, second.point_component_ids
    )
    assert first.component_count == second.component_count
    for left, right in zip(first.components, second.components):
        assert left.component_id == right.component_id
        assert left.stable_key == right.stable_key
        assert left.point_fraction == right.point_fraction
        assert left.view_count == right.view_count
        assert left.density == right.density
        np.testing.assert_array_equal(
            left.point_indices, right.point_indices
        )
        np.testing.assert_array_equal(
            left.voxel_indices, right.voxel_indices
        )
        np.testing.assert_array_equal(left.points, right.points)
        np.testing.assert_array_equal(
            left.voxel_keys, right.voxel_keys
        )


def test_exact_26_connectivity_and_immutable_outputs():
    points = np.concatenate(
        (
            _voxel_points((0, 0, 0), 4),
            _voxel_points((1, 1, 1), 4),
            _voxel_points((4, 4, 4), 3),
        )
    )
    result = _build(points)

    assert result.component_count == 2
    assert result.components[0].point_count == 8
    assert result.components[0].voxel_count == 2
    assert result.components[0].stable_key == (0, 0, 0)
    assert result.components[1].stable_key == (4, 4, 4)
    assert result.points.flags.writeable is False
    assert result.voxel_keys.flags.writeable is False
    assert result.point_component_ids.flags.writeable is False
    assert result.components[0].points.flags.writeable is False
    assert result.components[0].voxel_keys.flags.writeable is False

    with pytest.raises(ValueError):
        result.points[0, 0] = 99.0
    with pytest.raises(FrozenInstanceError):
        result.voxel_size = 2.0


def test_neighbor_radius_and_dilation_are_separate_explicit_controls():
    gap_two = np.concatenate(
        (
            _voxel_points((0, 0, 0), 3),
            _voxel_points((2, 2, 2), 3),
        )
    )
    radius_one = _build(gap_two, neighbor_radius=1)
    radius_two = _build(gap_two, neighbor_radius=2)
    assert radius_one.component_count == 2
    assert radius_two.component_count == 1

    gap_three = np.concatenate(
        (
            _voxel_points((0, 0, 0), 3),
            _voxel_points((3, 0, 0), 3),
        )
    )
    undilated = _build(gap_three, dilation_radius=0)
    dilated = _build(gap_three, dilation_radius=1)
    assert undilated.component_count == 2
    assert dilated.component_count == 1
    # Dilation affects connectivity only, never the public occupied keys.
    np.testing.assert_array_equal(
        undilated.voxel_keys, dilated.voxel_keys
    )
    assert dilated.components[0].voxel_count == 2

    gap_four = np.concatenate(
        (
            _voxel_points((0, 0, 0), 3),
            _voxel_points((4, 0, 0), 3),
        )
    )
    assert _build(gap_four, dilation_radius=1).component_count == 2


def test_origin_and_boundary_epsilon_are_explicit_and_translation_equivariant():
    below_boundary = np.nextafter(1.0, 0.0)
    points = np.asarray(
        [
            [below_boundary, 0.2, 0.2],
            [1.2, 0.2, 0.2],
        ],
        dtype=np.float64,
    )
    without_nudge = _build(points, boundary_epsilon=0.0)
    with_nudge = _build(points, boundary_epsilon=1e-9)
    np.testing.assert_array_equal(
        without_nudge.voxel_keys,
        np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        with_nudge.voxel_keys,
        np.asarray([[1, 0, 0]], dtype=np.int64),
    )

    shift = np.asarray([10.25, -3.75, 2.125])
    base = _build(
        np.concatenate(
            (
                _voxel_points((0, 0, 0), 3),
                _voxel_points((2, 0, 0), 2),
            )
        ),
        neighbor_radius=2,
    )
    shifted = _build(
        base.points + shift,
        origin=shift,
        neighbor_radius=2,
    )
    np.testing.assert_array_equal(base.voxel_keys, shifted.voxel_keys)
    np.testing.assert_allclose(
        shifted.points, base.points + shift, atol=1e-14
    )
    np.testing.assert_array_equal(
        base.point_component_ids, shifted.point_component_ids
    )


def test_sparse_voxel_filter_marks_discarded_points_without_reindexing_input():
    dense = _voxel_points((0, 0, 0), 3)
    sparse = _voxel_points((4, 0, 0), 1)
    result = _build(
        np.concatenate((sparse, dense)),
        min_points_per_voxel=2,
    )

    assert result.component_count == 1
    assert result.retained_point_count == 3
    assert len(result.points) == 4
    np.testing.assert_array_equal(
        result.voxel_keys, np.asarray([[0, 0, 0]], dtype=np.int64)
    )
    sparse_index = int(np.argmax(result.points[:, 0]))
    assert result.point_to_voxel[sparse_index] == -1
    assert result.point_component_ids[sparse_index] == -1
    assert result.components[0].point_fraction == pytest.approx(0.75)


def test_point_permutation_produces_exact_canonical_components():
    points = np.concatenate(
        (
            _voxel_points((0, 0, 0), 5),
            _voxel_points((1, 0, 0), 4),
            _voxel_points((8, 0, 0), 6),
        )
    )
    view_ids = np.asarray(
        [0, 1, 0, 2, 1, 2, 0, 1, 2, 3, 3, 2, 1, 0, 3],
        dtype=np.int64,
    )
    first = _build(points, point_view_ids=view_ids)
    permutation = np.random.default_rng(71).permutation(len(points))
    second = _build(
        points[permutation],
        point_view_ids=view_ids[permutation],
    )

    _assert_component_sets_equal(first, second)


def test_largest_selector_uses_points_then_voxels_then_stable_key():
    one_voxel = _voxel_points((0, 0, 0), 8)
    two_voxels = np.concatenate(
        (
            _voxel_points((10, 0, 0), 4),
            _voxel_points((11, 0, 0), 4),
        )
    )
    result = _build(np.concatenate((two_voxels, one_voxel)))
    selected = select_largest_component(result)

    assert selected is not None
    assert selected.point_count == 8
    assert selected.voxel_count == 2
    assert selected.stable_key == (10, 0, 0)

    equal_lower = np.concatenate(
        (
            _voxel_points((0, 0, 0), 3),
            _voxel_points((1, 0, 0), 3),
        )
    )
    equal_upper = np.concatenate(
        (
            _voxel_points((10, 0, 0), 3),
            _voxel_points((11, 0, 0), 3),
        )
    )
    tied = _build(np.concatenate((equal_upper, equal_lower)))
    assert select_largest_component(tied).stable_key == (0, 0, 0)
    assert (
        select_largest_component(tied, min_points=7) is None
    )


def test_densest_selector_is_independent_from_largest_selector():
    # A large diffuse component spans eight voxels.
    diffuse = np.concatenate(
        [
            _voxel_points((axis, side, height), 2, spread=0.50)
            for axis in (0, 1)
            for side in (0, 1)
            for height in (0, 1)
        ]
    )
    # A smaller compact component occupies a much smaller raw AABB.
    compact = _voxel_points((10, 0, 0), 10, spread=0.02)
    result = _build(np.concatenate((diffuse, compact)))

    largest = select_largest_component(result)
    densest = select_densest_component(result)
    assert largest is not None and densest is not None
    assert largest.stable_key == (0, 0, 0)
    assert densest.stable_key == (10, 0, 0)
    assert densest.density > largest.density


def test_inside_anchor_prefers_box_support_then_views_over_larger_distractor():
    anchored = _voxel_points((0, 0, 0), 8)
    distractor = np.concatenate(
        (
            _voxel_points((5, 0, 0), 8),
            _voxel_points((6, 0, 0), 8),
        )
    )
    points = np.concatenate((distractor, anchored))
    # The anchored component is seen by three views; the larger distractor is
    # outside the seed box and is observed by one.
    views = np.concatenate(
        (
            np.zeros(len(distractor), dtype=np.int64),
            np.arange(len(anchored), dtype=np.int64) % 3,
        )
    )
    result = _build(points, point_view_ids=views)
    selected = select_inside_anchor(
        result,
        lower=np.asarray([0.0, 0.0, 0.0]),
        upper=np.asarray([1.0, 1.0, 1.0]),
        min_points=4,
        min_views=2,
        min_inside_points=6,
        min_inside_fraction=0.75,
    )

    assert selected is not None
    assert selected.stable_key == (0, 0, 0)
    assert selected.view_count == 3
    assert np.max(selected.points[:, 0]) < 1.0

    rejected = select_inside_anchor(
        result,
        lower=np.asarray([0.0, 0.0, 0.0]),
        upper=np.asarray([1.0, 1.0, 1.0]),
        min_views=4,
    )
    assert rejected is None


@pytest.mark.parametrize(
    "updates",
    [
        {"voxel_size": 0.0},
        {"boundary_epsilon": -1e-8},
        {"boundary_epsilon": 0.5},
        {"neighbor_radius": 0},
        {"neighbor_radius": 1.5},
        {"dilation_radius": -1},
        {"dilation_radius": True},
        {"min_points_per_voxel": 0},
    ],
)
def test_invalid_grid_parameters_fail_fast(updates):
    with pytest.raises(ValueError):
        _build(np.zeros((1, 3), dtype=np.float64), **updates)


def test_invalid_points_views_bounds_and_empty_input_contract():
    with pytest.raises(ValueError, match="shape"):
        _build(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="finite"):
        _build(np.asarray([[np.nan, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="point_view_ids"):
        _build(
            np.zeros((2, 3)),
            point_view_ids=np.asarray([0], dtype=np.int64),
        )
    with pytest.raises(ValueError, match="point_view_ids"):
        _build(
            np.zeros((2, 3)),
            point_view_ids=np.asarray([0.0, 1.0]),
        )

    empty = _build(np.empty((0, 3)))
    assert empty.component_count == 0
    assert empty.retained_point_count == 0
    assert select_largest_component(empty) is None
    assert select_densest_component(empty) is None

    with pytest.raises(ValueError, match="upper"):
        select_inside_anchor(
            empty,
            lower=np.ones(3),
            upper=np.zeros(3),
        )
    with pytest.raises(ValueError, match="min_inside_fraction"):
        select_inside_anchor(
            empty,
            lower=np.zeros(3),
            upper=np.ones(3),
            min_inside_fraction=1.1,
        )
