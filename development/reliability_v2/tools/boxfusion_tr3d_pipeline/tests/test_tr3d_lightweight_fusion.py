import time

import numpy as np

from boxfusion.tr3d_incremental_online import IncrementalTR3DConfig, TR3DProviderResult
from boxfusion.tr3d_lightweight_fusion import (
    DepthViewEvidence,
    LightweightAsyncTR3DObserver,
    LightweightFusionConfig,
    depth_view_evidence,
    diverse_top_k_indices,
    fuse_yaw_boxes,
)
from boxfusion.tr3d_r2_geometry import yaw_obb_corners_world


def _view(frame, quality, direction):
    return DepthViewEvidence(
        frame, np.zeros(3), np.asarray(direction, np.float64), 0.02,
        0.7, 0.1, 0.1, 0.1, quality,
    )


def test_diverse_topk_does_not_select_only_duplicate_view_directions():
    values = [
        _view(1, 0.90, [1, 0, 0]),
        _view(2, 0.89, [0.999, 0.04, 0]),
        _view(3, 0.80, [0, 1, 0]),
    ]
    selected = diverse_top_k_indices(
        values, 2, diversity_weight=0.5, min_view_angle_deg=12.0
    )
    assert selected == [0, 2]


def test_depth_visibility_detects_support_without_free_space():
    box = np.asarray([0.0, 0.0, 2.0, 1.0, 1.0, 1.0, 0.0])
    corners = yaw_obb_corners_world(box)
    depth = np.full((80, 100), 2.0, np.float32)
    intrinsics = np.asarray([[80.0, 0.0, 50.0], [0.0, 80.0, 40.0], [0.0, 0.0, 1.0]])
    evidence = depth_view_evidence(
        corners, depth=depth, intrinsics=intrinsics,
        camera_to_world=np.eye(4), frame_id=7,
        config=LightweightFusionConfig(depth_pixel_stride=4),
    )
    assert evidence is not None
    # Projected rectangle sampling includes grazing rays; the important
    # invariant is that the real surface remains the majority evidence.
    assert evidence.support_ratio > 0.50
    assert evidence.support_ratio > evidence.free_space_ratio
    assert evidence.quality > 0.0


def test_fused_yaw_box_uses_circular_pi_periodic_mean():
    boxes = np.asarray([
        [0, 0, 2, 1, 0.5, 1, np.deg2rad(89)],
        [0.1, 0, 2, 1, 0.5, 1, np.deg2rad(-89)],
    ], np.float64)
    result = fuse_yaw_boxes(boxes, np.ones(2))
    assert np.isclose(result[0], 0.05)
    assert abs(abs(np.rad2deg(result[6])) - 90.0) < 1.1


class _FakeProvider:
    def __init__(self, corners):
        self.corners = corners
        self.calls = 0

    def infer(self, **_kwargs):
        self.calls += 1
        time.sleep(0.005)
        shift = 0.01 * self.calls
        corners = self.corners.copy()
        corners[:, :, 0] += shift
        return TR3DProviderResult(
            corners, np.asarray([0.8], np.float32), 0.005,
            np.asarray([100], np.int64),
        )


def test_async_observer_drains_and_emits_lightweight_features():
    corners = yaw_obb_corners_world(
        np.asarray([0, 0, 2, 1, 1, 1, 0], np.float64)
    )[None].astype(np.float32)
    provider = _FakeProvider(corners)
    observer = LightweightAsyncTR3DObserver(
        IncrementalTR3DConfig(
            pixel_stride=8, warmup_keyframes=1,
            inference_interval_keyframes=1, min_track_hits=2,
        ),
        provider,
        LightweightFusionConfig(drain_on_finalize=True, depth_pixel_stride=8),
    )
    observer.reset_scene("scene0000_00", np.eye(4))
    depth = np.full((40, 60), 2.0, np.float32)
    image = np.zeros((40, 60, 3), np.uint8)
    intrinsics = np.asarray([[50, 0, 30], [0, 50, 20], [0, 0, 1]], np.float64)
    for frame in range(3):
        observer.process_keyframe(
            scene_id="scene0000_00", depth=depth, image=image,
            intrinsics=intrinsics, camera_to_world=np.eye(4),
            source_timestamp=frame,
        )
    summary = observer.finalize(
        anchor_corners_world=np.empty((0, 8, 3)),
        anchor_scores=np.empty((0,)),
    )
    assert summary["schema"] == "boxfusion.tr3d_lightweight_online_observer.v1"
    assert summary["async_completed"] >= 2
    assert summary["confirmed_tracks"] == 1
    row = summary["confirmed"][0]
    assert row["lightweight_schema"] == "boxfusion.tr3d_lightweight_track.v1"
    assert row["diverse_topk_count"] >= 1
    assert np.asarray(row["selected_corners_world"]).shape == (8, 3)
