from __future__ import annotations

import numpy as np
import pytest

from boxfusion.tr3d_r2_observer import TR3DR2ObserverConfig
from boxfusion.tr3d_r2b_observer import (
    FEATURE_STAT_NAMES,
    TR3DR2BFrameBundle,
    feature_consistency_statistics,
    observe_tr3d_r2b_scene,
    pool_supported_dense_features,
    project_world_points_to_rgb,
)


def test_project_world_points_to_rgb_keeps_source_indices() -> None:
    intrinsic = np.eye(4, dtype=np.float64)
    intrinsic[0, 0] = intrinsic[1, 1] = 10.0
    intrinsic[0, 2] = intrinsic[1, 2] = 5.0
    points = np.asarray(
        [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 0.0, -1.0]],
        dtype=np.float64,
    )
    u, v, indices = project_world_points_to_rgb(
        points, intrinsic, np.eye(4), (10, 10)
    )
    np.testing.assert_allclose(u, [5.0])
    np.testing.assert_allclose(v, [5.0])
    np.testing.assert_array_equal(indices, [0])
    assert not u.flags.writeable


def test_pooling_uses_unique_cells_and_normalizes() -> None:
    features = np.asarray(
        [
            [[1.0, 3.0], [5.0, 7.0]],
            [[0.0, 4.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    vector, count = pool_supported_dense_features(
        features,
        [0.1, 0.2, 3.0],
        [0.1, 0.2, 0.1],
        source_image_shape=(4, 4),
    )
    # The first two points share one cell; the third point hits the other top
    # cell.  Unique-cell pooling is therefore mean([1,0], [3,4])=[2,2].
    assert count == 2
    np.testing.assert_allclose(vector, [2**-0.5, 2**-0.5], atol=1e-6)

    missing, count = pool_supported_dense_features(
        features,
        [0.1],
        [0.1],
        source_image_shape=(4, 4),
        min_unique_cells=2,
    )
    assert missing is None and count == 1


def test_pairwise_statistics_and_single_view_sentinel() -> None:
    vectors = np.asarray(
        [[1.0, 0.0], [0.8, 0.6], [0.0, 0.0]], dtype=np.float32
    )
    stats, pairs = feature_consistency_statistics(
        vectors, np.asarray([True, True, False])
    )
    assert FEATURE_STAT_NAMES[0] == "pairwise_mean"
    assert pairs == 1
    np.testing.assert_allclose(stats, [0.8, 0.8, 0.8, 0.8, 0.0, 0.8], atol=1e-6)

    stats, pairs = feature_consistency_statistics(
        vectors, np.asarray([True, False, False])
    )
    assert pairs == 0
    np.testing.assert_array_equal(stats, np.zeros(len(FEATURE_STAT_NAMES)))


def _intrinsic() -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = matrix[1, 1] = 20.0
    matrix[0, 2] = matrix[1, 2] = 8.0
    return matrix


def test_scene_observer_encodes_each_unique_frame_once_and_is_observer_only() -> None:
    boxes = np.asarray([[0.0, 0.0, 3.0, 2.0, 2.0, 2.0, 0.0]], dtype=np.float32)
    ids = np.asarray([17], dtype=np.int64)
    topk_frames = np.asarray([[0, 1, -1]], dtype=np.int64)
    topk_valid = np.asarray([[True, True, False]], dtype=np.bool_)
    colors = {
        0: np.full((16, 16, 3), 1, dtype=np.uint8),
        1: np.full((16, 16, 3), 2, dtype=np.uint8),
    }
    depths = {
        0: np.full((16, 16), 3.0, dtype=np.float32),
        1: np.full((16, 16), 3.0, dtype=np.float32),
    }
    bundle = TR3DR2BFrameBundle(
        scene_id="scene0001_00",
        pose_source="resolved",
        color=colors,
        depth=depths,
        pose={0: np.eye(4), 1: np.eye(4)},
        intrinsic_depth=_intrinsic(),
        intrinsic_color=_intrinsic(),
        extrinsic_depth=np.eye(4),
        extrinsic_color=np.eye(4),
    )
    calls: list[int] = []

    def encode(image: np.ndarray) -> np.ndarray:
        marker = int(image[0, 0, 0])
        calls.append(marker)
        vector = [1.0, 0.0] if marker == 1 else [0.8, 0.6]
        return np.broadcast_to(
            np.asarray(vector, dtype=np.float32)[:, None, None],
            (2, 4, 4),
        ).copy()

    boxes_before = boxes.copy()
    result = observe_tr3d_r2b_scene(
        boxes_world=boxes,
        proposal_ids=ids,
        topk_frame_ids=topk_frames,
        topk_view_valid=topk_valid,
        frame_bundle=bundle,
        depth_config=TR3DR2ObserverConfig(
            image_shape=(16, 16),
            pose_source="resolved",
            top_k=3,
            pixel_stride=2,
        ),
        encode_rgb=encode,
        min_support_points=2,
    )

    assert calls == [1, 2]
    np.testing.assert_array_equal(boxes, boxes_before)
    np.testing.assert_array_equal(result.feature_view_valid, [[True, True, False]])
    np.testing.assert_array_equal(result.aggregate_feature_view_count, [2])
    np.testing.assert_array_equal(result.aggregate_feature_pair_count, [1])
    assert result.per_view_support_point_count[0, 0] > 1
    assert result.per_view_feature_count[0, 0] > 0
    assert result.aggregate_feature_statistics[0, 0] == pytest.approx(0.8, abs=1e-5)
    assert not result.per_view_features.flags.writeable
    np.testing.assert_array_equal(result.encoded_frame_ids, [0, 1])


def test_scene_observer_rejects_noncanonical_invalid_slots() -> None:
    bundle = TR3DR2BFrameBundle(
        scene_id="scene0001_00",
        pose_source="resolved",
        color={},
        depth={},
        pose={},
        intrinsic_depth=np.eye(4),
        intrinsic_color=np.eye(4),
        extrinsic_depth=np.eye(4),
        extrinsic_color=np.eye(4),
    )
    with pytest.raises(ValueError, match="invalid Top-K slots"):
        observe_tr3d_r2b_scene(
            boxes_world=np.asarray([[0, 0, 3, 1, 1, 1, 0]], dtype=float),
            proposal_ids=np.asarray([1]),
            topk_frame_ids=np.asarray([[0]]),
            topk_view_valid=np.asarray([[False]]),
            frame_bundle=bundle,
            depth_config=TR3DR2ObserverConfig(
                image_shape=(4, 4), pose_source="resolved"
            ),
            encode_rgb=lambda _: np.ones((2, 1, 1), dtype=np.float32),
        )
