"""Deterministic CPU tests for true Top-K Mask-RGBD object memory."""

from __future__ import annotations

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
    "boxfusion_b3_topk_object_memory",
    SOURCE,
)
object_memory = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = object_memory
SPEC.loader.exec_module(object_memory)


def memory_config(**overrides):
    values = {
        "mask_edge_margin": 0,
        "depth_edge_threshold": None,
        "voxel_size": 0.0,
        "max_points_per_observation": 64,
        "max_points_per_object": 512,
        "aabb_lower_quantile": 0.0,
        "aabb_upper_quantile": 1.0,
        "min_points_for_aabb": 4,
        "minimum_aabb_dimension": 0.01,
        "top_k_views": 2,
        "max_view_candidates": 8,
        "view_diversity_weight": 0.5,
        "minimum_view_quality": 0.0,
    }
    values.update(overrides)
    return object_memory.resolve_object_memory_config(values)


def cube_points(center, half_extent=0.5):
    center = np.asarray(center, dtype=np.float32)
    signs = np.asarray(
        list(itertools.product((-1.0, 1.0), repeat=3)),
        dtype=np.float32,
    )
    return center[None, :] + signs * float(half_extent)


def observation(
    center,
    *,
    confidence=1.0,
    camera_position=None,
    half_extent=0.5,
):
    return object_memory.ObjectObservation(
        cube_points(center, half_extent),
        confidence=confidence,
        mask_pixels=8,
        valid_depth_pixels=8,
        projection_mask_iou=1.0,
        camera_position=camera_position,
    )


def assert_optional_aabb_equal(first, second):
    assert (first is None) == (second is None)
    if first is not None:
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])


def test_observation_accepts_optional_defensive_camera_position():
    config = memory_config()
    assert config["top_k_views"] == 2
    assert config["max_view_candidates"] == 8
    assert config["view_diversity_weight"] == pytest.approx(0.5)
    assert config["minimum_view_quality"] == pytest.approx(0.0)

    camera = np.asarray([3.0, 2.0, 1.0], dtype=np.float32)
    with_camera = observation([0, 0, 0], camera_position=camera)
    camera[:] = 99.0
    np.testing.assert_array_equal(
        with_camera.camera_position,
        [3.0, 2.0, 1.0],
    )
    assert observation([0, 0, 0]).camera_position is None

    with pytest.raises(ValueError, match="camera_position"):
        observation([0, 0, 0], camera_position=[1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        observation([0, 0, 0], camera_position=[0.0, np.nan, 0.0])


def test_top_k_zero_preserves_legacy_geometry_exactly():
    memory = object_memory.ObjectGeometryMemory(
        7,
        memory_config(top_k_views=0),
    )
    memory.add_observation(
        observation(
            [0.0, 0.0, 0.0],
            confidence=0.9,
            camera_position=[3.0, 0.0, 0.0],
        ),
        frame_id=2,
    )
    memory.add_observation(
        observation(
            [0.2, 0.0, 0.0],
            confidence=0.8,
            camera_position=[0.0, 3.0, 0.0],
        ),
        frame_id=5,
    )

    np.testing.assert_array_equal(memory.geometry_points, memory.points)
    assert memory.geometry_num_points == memory.num_points
    assert_optional_aabb_equal(memory.geometry_aabb, memory.aabb)


def test_low_quality_far_outlier_is_excluded_and_geometry_aabb_is_cleaner():
    memory = object_memory.ObjectGeometryMemory(
        1,
        memory_config(
            top_k_views=2,
            minimum_view_quality=0.1,
            view_diversity_weight=0.0,
        ),
    )
    memory.add_observation(
        observation(
            [0.0, 0.0, 0.0],
            confidence=0.95,
            camera_position=[3.0, 0.0, 0.0],
        ),
        frame_id=0,
    )
    memory.add_observation(
        observation(
            [0.1, 0.0, 0.0],
            confidence=0.90,
            camera_position=[0.0, 3.0, 0.0],
        ),
        frame_id=1,
    )
    memory.add_observation(
        observation(
            [50.0, 0.0, 0.0],
            confidence=0.01,
            camera_position=[-3.0, 0.0, 0.0],
        ),
        frame_id=2,
    )

    assert set(memory.selected_view_frame_ids) == {0, 1}
    assert 2 not in memory.selected_view_frame_ids
    legacy_center, legacy_dims = memory.aabb
    geometry_center, geometry_dims = memory.geometry_aabb
    assert abs(float(geometry_center[0])) < abs(float(legacy_center[0]))
    assert float(geometry_dims[0]) < float(legacy_dims[0]) * 0.1
    assert float(np.max(memory.geometry_points[:, 0])) < 2.0
    assert float(np.max(memory.points[:, 0])) > 40.0


def test_similar_quality_prefers_distinct_camera_direction():
    memory = object_memory.ObjectGeometryMemory(
        2,
        memory_config(
            top_k_views=2,
            max_view_candidates=3,
            view_diversity_weight=1.0,
        ),
    )
    observations = (
        observation(
            [0, 0, 0],
            confidence=0.90,
            camera_position=[5.0, 0.0, 0.0],
        ),
        observation(
            [0, 0, 0],
            confidence=0.89,
            camera_position=[5.0, 0.02, 0.0],
        ),
        observation(
            [0, 0, 0],
            confidence=0.88,
            camera_position=[0.0, 5.0, 0.0],
        ),
    )
    for frame_id, current in enumerate(observations):
        memory.add_observation(current, frame_id)

    assert set(memory.selected_view_frame_ids) == {0, 2}
    assert 1 not in memory.selected_view_frame_ids


def test_candidate_pool_and_top_k_are_bounded_and_deterministic():
    config = memory_config(
        top_k_views=3,
        max_view_candidates=5,
        view_diversity_weight=0.4,
    )
    memories = (
        object_memory.ObjectGeometryMemory(3, config),
        object_memory.ObjectGeometryMemory(3, config),
    )

    for memory in memories:
        for frame_id in range(12):
            angle = frame_id * np.pi / 6.0
            memory.add_observation(
                observation(
                    [0.01 * frame_id, 0.0, 0.0],
                    confidence=0.70 + 0.02 * (frame_id % 5),
                    camera_position=[
                        4.0 * np.cos(angle),
                        4.0 * np.sin(angle),
                        0.0,
                    ],
                ),
                frame_id,
            )
            assert memory.view_candidate_count <= 5
            assert memory.selected_view_count <= 3

    assert memories[0].view_candidate_count == 5
    assert memories[0].selected_view_count == 3
    assert (
        memories[0].selected_view_frame_ids
        == memories[1].selected_view_frame_ids
    )
    np.testing.assert_array_equal(
        memories[0].geometry_points,
        memories[1].geometry_points,
    )


def test_same_frame_candidate_keeps_higher_quality_observation():
    memory = object_memory.ObjectGeometryMemory(
        4,
        memory_config(
            top_k_views=1,
            max_view_candidates=2,
            view_diversity_weight=0.0,
        ),
    )
    memory.add_observation(
        observation(
            [20.0, 0.0, 0.0],
            confidence=0.2,
            camera_position=[3.0, 0.0, 0.0],
        ),
        frame_id=9,
    )
    memory.add_observation(
        observation(
            [0.0, 0.0, 0.0],
            confidence=0.9,
            camera_position=[0.0, 3.0, 0.0],
        ),
        frame_id=9,
    )

    assert memory.view_candidate_count == 1
    assert memory.selected_view_frame_ids == (9,)
    assert float(np.max(memory.geometry_points[:, 0])) < 2.0


def test_geometry_accessors_return_defensive_arrays():
    memory = object_memory.ObjectGeometryMemory(
        5,
        memory_config(top_k_views=1),
    )
    memory.add_observation(
        observation(
            [1.0, 2.0, 3.0],
            camera_position=[4.0, 2.0, 3.0],
        ),
        frame_id=0,
    )

    expected_points = memory.geometry_points
    external_points = memory.geometry_points
    external_points[:] = 999.0
    np.testing.assert_array_equal(memory.geometry_points, expected_points)

    expected_center, expected_dims = memory.geometry_aabb
    external_center, external_dims = memory.geometry_aabb
    external_center[:] = 999.0
    external_dims[:] = 999.0
    actual_center, actual_dims = memory.geometry_aabb
    np.testing.assert_array_equal(actual_center, expected_center)
    np.testing.assert_array_equal(actual_dims, expected_dims)


def test_quality_summary_reports_top_k_memory_counts():
    memory = object_memory.ObjectGeometryMemory(
        6,
        memory_config(top_k_views=2, max_view_candidates=3),
    )
    for frame_id, confidence in enumerate((0.9, 0.8, 0.7)):
        memory.add_observation(
            observation(
                [0.05 * frame_id, 0.0, 0.0],
                confidence=confidence,
                camera_position=[3.0 - frame_id, frame_id, 0.0],
            ),
            frame_id,
        )

    summary = memory.quality_summary()
    assert summary["selected_views"] == memory.selected_view_count == 2
    assert summary["view_candidates"] == memory.view_candidate_count == 3
    assert (
        summary["geometry_stored_points"]
        == memory.geometry_num_points
    )


def test_candidate_merge_crops_each_view_to_the_global_box():
    config = memory_config(
        top_k_views=2,
        max_view_candidates=4,
        view_diversity_weight=0.0,
    )
    source = object_memory.ObjectGeometryMemory(10, config)
    source.add_observation(
        observation(
            [0.0, 0.0, 0.0],
            confidence=0.9,
            camera_position=[3.0, 0.0, 0.0],
        ),
        frame_id=1,
    )
    source.add_observation(
        observation(
            [25.0, 0.0, 0.0],
            confidence=0.99,
            camera_position=[28.0, 0.0, 0.0],
        ),
        frame_id=2,
    )

    destination = object_memory.ObjectGeometryMemory(11, config)
    destination.merge_view_candidates_from(
        source,
        crop_center=[0.0, 0.0, 0.0],
        crop_dims=[2.0, 2.0, 2.0],
        minimum_points=4,
    )

    assert destination.view_candidate_count == 1
    assert destination.selected_view_frame_ids == (1,)
    assert float(np.max(np.abs(destination.geometry_points))) <= 0.5
    # Per-view merge is deliberately B3-only: legacy B6 statistics stay empty.
    assert destination.num_points == 0
    assert destination.aabb is None
