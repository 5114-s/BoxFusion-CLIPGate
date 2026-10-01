import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest


DEFAULT_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "boxfusion"
    / "reliable_views.py"
)
SOURCE = Path(
    os.environ.get("BOXFUSION_RELIABLE_VIEWS", DEFAULT_SOURCE)
)
spec = importlib.util.spec_from_file_location(
    "boxfusion_reliable_views", SOURCE
)
reliable_views = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reliable_views)


def rectangle_corners(x1, y1, x2, y2):
    return np.asarray(
        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ],
        dtype=np.float32,
    )


def make_inputs(count):
    boxes = np.tile(
        np.asarray([[0.0, 0.0, 2.0, 2.0, 1.0, 1.0]]),
        (count, 1),
    )
    scores = np.full(count, 0.9, dtype=np.float32)
    detector_boxes = np.tile(
        np.asarray([[100.0, 100.0, 300.0, 300.0]]),
        (count, 1),
    )
    corners = np.stack(
        [rectangle_corners(100.0, 100.0, 300.0, 300.0)]
        * count
    )
    return boxes, scores, detector_boxes, corners


def enabled_cfg(**overrides):
    cfg = reliable_views.resolve_reliable_view_config(
        {"reliable_views": {"enabled": True, **overrides}}
    )
    return cfg


def select(boxes, scores, detector_boxes, corners, **cfg):
    return reliable_views.select_top_k_reliable_views(
        boxes,
        scores,
        detector_boxes,
        corners,
        image_height=480,
        image_width=640,
        cfg=enabled_cfg(**cfg),
    )


def test_missing_config_is_disabled_for_baseline_compatibility():
    cfg = reliable_views.resolve_reliable_view_config({})
    assert cfg["enabled"] is False
    assert cfg["top_k"] == 3
    assert cfg["min_views"] == 3


@pytest.mark.parametrize(
    "override",
    [
        {"top_k": 0},
        {"min_views": 0},
        {"area_reference_ratio": 0.0},
        {"center_sigma": 0.0},
        {"minimum_weight": 1.1},
        {"projection_iou_power": -1.0},
        {"geometry_consistency_power": -1.0},
    ],
)
def test_invalid_config_fails_fast(override):
    with pytest.raises(ValueError):
        enabled_cfg(**override)


def test_projected_area_and_projection_iou_are_exact():
    corners = np.stack(
        [
            rectangle_corners(0.0, 0.0, 320.0, 240.0),
            rectangle_corners(100.0, 100.0, 200.0, 200.0),
        ]
    )
    area = reliable_views.projected_area_ratio(
        corners, image_height=480, image_width=640
    )
    np.testing.assert_allclose(
        area,
        [0.25, 10000.0 / (480.0 * 640.0)],
    )

    detector_boxes = np.asarray(
        [
            [0.0, 0.0, 320.0, 240.0],
            [150.0, 100.0, 250.0, 200.0],
        ]
    )
    iou = reliable_views.detector_projection_iou(
        detector_boxes,
        corners,
        image_height=480,
        image_width=640,
    )
    np.testing.assert_allclose(iou, [1.0, 1.0 / 3.0])


def test_real_confidence_changes_reliability_ranking():
    boxes, scores, detector_boxes, corners = make_inputs(4)
    scores[:] = [0.55, 0.95, 0.80, 0.70]
    result = select(
        boxes,
        scores,
        detector_boxes,
        corners,
        top_k=3,
    )
    np.testing.assert_array_equal(
        result["selected_indices"], [1, 2, 3]
    )


def test_bad_2d_3d_projection_agreement_is_penalized():
    boxes, scores, detector_boxes, corners = make_inputs(4)
    detector_boxes[3] = [400.0, 350.0, 500.0, 450.0]
    result = select(
        boxes,
        scores,
        detector_boxes,
        corners,
        top_k=3,
    )
    assert result["projection_iou"][3] == 0.0
    assert result["weights"][3] < result["weights"][0]
    assert 3 not in result["selected_indices"]


def test_geometric_outlier_is_removed_even_with_high_confidence():
    boxes, scores, detector_boxes, corners = make_inputs(4)
    boxes[:, 0] = [0.0, 0.05, 0.10, 10.0]
    scores[:] = [0.85, 0.85, 0.85, 0.99]
    result = select(
        boxes,
        scores,
        detector_boxes,
        corners,
        top_k=3,
    )
    assert result["geometry_consistency"][3] < 0.01
    assert 3 not in result["selected_indices"]


def test_equal_reliability_uses_stable_temporal_order():
    boxes, scores, detector_boxes, corners = make_inputs(5)
    result = select(
        boxes,
        scores,
        detector_boxes,
        corners,
        top_k=3,
    )
    np.testing.assert_array_equal(
        result["selected_indices"], [0, 1, 2]
    )


def test_minimum_weight_ties_use_real_confidence_then_time():
    boxes, scores, detector_boxes, corners = make_inputs(4)
    scores[:] = [0.60, 0.90, 0.90, 0.70]
    detector_boxes[:] = [400.0, 350.0, 500.0, 450.0]
    result = select(
        boxes,
        scores,
        detector_boxes,
        corners,
        top_k=3,
    )
    np.testing.assert_allclose(result["weights"], 0.05)
    np.testing.assert_array_equal(
        result["selected_indices"], [1, 2, 3]
    )


def test_min_views_prevents_over_aggressive_top_k():
    boxes, scores, detector_boxes, corners = make_inputs(5)
    result = select(
        boxes,
        scores,
        detector_boxes,
        corners,
        top_k=2,
        min_views=4,
    )
    assert result["selected_indices"].shape[0] == 4


def test_nan_inputs_produce_finite_bounded_weights():
    boxes, scores, detector_boxes, corners = make_inputs(3)
    scores[0] = np.nan
    detector_boxes[1, 0] = np.inf
    corners[2, 0, 0] = np.nan
    result = select(
        boxes,
        scores,
        detector_boxes,
        corners,
    )
    assert np.isfinite(result["weights"]).all()
    assert (result["weights"] > 0.0).all()


def test_stable_unique_preserves_first_observation_order():
    unique = reliable_views.stable_unique(
        np.asarray([7, 3, 7, 5, 3, 9])
    )
    np.testing.assert_array_equal(unique, [7, 3, 5, 9])


def test_valid_view_mask_rejects_invalid_and_behind_camera_views():
    boxes, scores, detector_boxes, corners = make_inputs(4)
    rotations = np.tile(np.eye(3)[None], (4, 1, 1))
    camera_poses = np.tile(np.eye(4)[None], (4, 1, 1))
    boxes[:, 2] = 2.0
    boxes[1, 3] = 0.0
    rotations[2, 0, 0] = np.nan
    boxes[3, 2] = -2.0
    valid = reliable_views.valid_reliable_view_mask(
        boxes,
        rotations,
        scores,
        detector_boxes,
        corners,
        camera_poses,
    )
    np.testing.assert_array_equal(valid, [True, False, False, False])


def test_weighted_initialization_uses_best_rotation_and_weighted_center():
    boxes = np.asarray(
        [
            [0.0, 0.0, 2.0, 3.0, 2.0, 1.0],
            [4.0, 0.0, 2.0, 1.0, 3.0, 2.0],
        ],
        dtype=np.float32,
    )
    rotations = np.stack(
        [
            np.eye(3, dtype=np.float32),
            np.asarray(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
        ]
    )
    initial_box, rotation, best = (
        reliable_views.weighted_box_initialization(
            boxes,
            rotations,
            np.asarray([3.0, 1.0]),
        )
    )
    np.testing.assert_allclose(
        initial_box,
        [1.0, 0.0, 2.0, 3.0, 2.0, 1.0],
    )
    np.testing.assert_allclose(rotation, rotations[0])
    assert best == 0


def test_weighted_initialization_rejects_invalid_dimensions():
    boxes = np.asarray([[0.0, 0.0, 2.0, 1.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        reliable_views.weighted_box_initialization(
            boxes,
            np.eye(3)[None],
            np.ones(1),
        )
