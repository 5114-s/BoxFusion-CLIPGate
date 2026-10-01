from __future__ import annotations

import numpy as np
import pytest

from boxfusion.tr3d_r3_observer import (
    TR3D_R3_NEAR_ANCHOR_IOU,
    axis_aligned_minmax,
    observe_anchor_near_candidates,
    pairwise_aabb_iou,
)


_SIGNS = np.asarray(
    [
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1],
    ],
    dtype=np.float64,
)


def _corners(center, size):
    return np.asarray(center, dtype=np.float64) + _SIGNS * (
        np.asarray(size, dtype=np.float64) / 2
    )


def _observe():
    proposals = np.stack(
        (
            _corners([0, 0, 0], [2, 2, 2]),
            _corners([3.5, 0, 0], [2, 2, 2]),
            _corners([20, 0, 0], [1, 1, 1]),
        )
    )
    anchors = np.stack(
        (
            _corners([0, 0, 0], [2, 2, 2]),
            _corners([3, 0, 0], [2, 2, 2]),
        )
    )
    return observe_anchor_near_candidates(
        proposal_ids=np.asarray([10, 20, 30], dtype=np.int64),
        lineage_ids=np.asarray([100, 200, 300], dtype=np.int64),
        proposal_corners_world=proposals,
        tr3d_score=np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
        point_count=np.asarray([80, 40, 10], dtype=np.int32),
        anchor_corners_world=anchors,
        anchor_score=np.asarray([0.6, 0.7], dtype=np.float64),
        axis_alignment=np.eye(4, dtype=np.float64),
        r2a_evidence_available=np.asarray([True, False, True], dtype=np.bool_),
        r2a_depth_evidence=np.asarray(
            [[0.8, 0.1, 0.05, 0.05], [0, 0, 0, 0], [0.5, 0.2, 0.2, 0.1]],
            dtype=np.float32,
        ),
        r2a_view_count=np.asarray([2, 0, 1], dtype=np.int32),
        r2a_point_count=np.asarray([100, 0, 20], dtype=np.int64),
        r2b_feature_view_count=np.asarray([2, 0, 1], dtype=np.int32),
        r2b_pairwise_cosine_count=np.asarray([1, 0, 0], dtype=np.int32),
        r2b_pairwise_cosine_mean=np.asarray([0.8, 0, 0], dtype=np.float32),
        r2b_pairwise_cosine_median=np.asarray([0.8, 0, 0], dtype=np.float32),
        r2b_pairwise_cosine_min=np.asarray([0.8, 0, 0], dtype=np.float32),
        r2b_pairwise_cosine_max=np.asarray([0.8, 0, 0], dtype=np.float32),
        r2b_pairwise_cosine_std=np.asarray([0, 0, 0], dtype=np.float32),
    )


def test_near_observer_is_strict_stable_and_complete() -> None:
    result = _observe()
    assert TR3D_R3_NEAR_ANCHOR_IOU == 0.15
    np.testing.assert_array_equal(result.proposal_ids, [10, 20])
    np.testing.assert_array_equal(result.lineage_ids, [100, 200])
    np.testing.assert_array_equal(result.anchor_index, [0, 1])
    np.testing.assert_allclose(result.anchor_iou, [1.0, 0.6])
    np.testing.assert_allclose(result.center_distance_m, [0.0, 0.5])
    np.testing.assert_allclose(
        result.center_distance_over_anchor_diagonal,
        [0.0, 0.5 / np.sqrt(12)],
    )
    np.testing.assert_allclose(result.volume_ratio, [1.0, 1.0])
    np.testing.assert_allclose(result.point_density_m3, [10.0, 5.0])
    np.testing.assert_array_equal(result.r2a_evidence_available, [True, False])
    np.testing.assert_allclose(result.r2a_depth_quality, [0.8 / 0.95, 0.0])
    np.testing.assert_array_equal(result.r2b_feature_available, [True, False])
    np.testing.assert_array_equal(result.r2b_multiview_available, [True, False])
    assert not result.proposal_corners_world.flags.writeable


def test_fixed_split_and_missing_evidence_sentinels_fail_closed() -> None:
    with pytest.raises(ValueError, match="frozen at 0.15"):
        observe_anchor_near_candidates(
            proposal_ids=np.empty(0, dtype=np.int64),
            lineage_ids=np.empty(0, dtype=np.int64),
            proposal_corners_world=np.empty((0, 8, 3), dtype=np.float32),
            tr3d_score=np.empty(0, dtype=np.float32),
            point_count=np.empty(0, dtype=np.int32),
            anchor_corners_world=np.empty((0, 8, 3), dtype=np.float32),
            anchor_score=np.empty(0),
            axis_alignment=np.eye(4),
            r2a_evidence_available=np.empty(0, dtype=np.bool_),
            r2a_depth_evidence=np.empty((0, 4), dtype=np.float32),
            r2a_view_count=np.empty(0, dtype=np.int32),
            r2a_point_count=np.empty(0, dtype=np.int64),
            r2b_feature_view_count=np.empty(0, dtype=np.int32),
            r2b_pairwise_cosine_count=np.empty(0, dtype=np.int32),
            r2b_pairwise_cosine_mean=np.empty(0, dtype=np.float32),
            r2b_pairwise_cosine_median=np.empty(0, dtype=np.float32),
            r2b_pairwise_cosine_min=np.empty(0, dtype=np.float32),
            r2b_pairwise_cosine_max=np.empty(0, dtype=np.float32),
            r2b_pairwise_cosine_std=np.empty(0, dtype=np.float32),
            near_anchor_iou=0.2,
        )

    inputs = _observe()
    assert inputs.proposal_count == 2


def test_axis_transform_and_pairwise_iou() -> None:
    transform = np.eye(4)
    transform[:3, 3] = [5, -2, 1]
    boxes = axis_aligned_minmax(
        np.stack((_corners([0, 0, 0], [2, 2, 2]),)), transform
    )
    np.testing.assert_allclose(boxes, [[4, -3, 0, 6, -1, 2]])
    np.testing.assert_allclose(pairwise_aabb_iou(boxes, boxes), [[1.0]])
