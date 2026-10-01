import numpy as np

from boxfusion.tr3d_incremental_online import (
    CausalVoxelMemory,
    IncrementalTR3DConfig,
    IncrementalTR3DObserver,
    TR3DProviderResult,
    backproject_rgbd,
)


def cube(center, size=0.4):
    signs = np.asarray([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
    ], dtype=np.float32)
    return signs * (size / 2) + np.asarray(center, dtype=np.float32)


class FakeProvider:
    def __init__(self):
        self.calls = 0

    def infer(self, **kwargs):
        self.calls += 1
        shift = 0.01 * self.calls
        return TR3DProviderResult(
            np.stack((cube([shift, 0, 1]), cube([2 + shift, 0, 1]))),
            np.asarray([0.9, 0.7], dtype=np.float32),
            0.002,
        )


def test_backproject_and_bounded_memory():
    depth = np.ones((8, 8), dtype=np.float32)
    image = np.full((8, 8, 3), 127, dtype=np.uint8)
    intrinsic = np.asarray([[4, 0, 4], [0, 4, 4], [0, 0, 1]], dtype=np.float64)
    points = backproject_rgbd(
        depth, image, intrinsic, np.eye(4), pixel_stride=2,
        min_depth_m=0.1, max_depth_m=6.0,
    )
    assert points.shape == (16, 6)
    config = IncrementalTR3DConfig(voxel_size_m=0.05, max_memory_voxels=8)
    memory = CausalVoxelMemory(config)
    memory.update(points, keyframe_index=0)
    assert len(memory.snapshot()) <= 8


def test_periodic_provider_and_cross_prefix_tracks():
    config = IncrementalTR3DConfig(
        pixel_stride=2, warmup_keyframes=1, inference_interval_keyframes=2,
        min_track_hits=2, max_memory_voxels=128, max_snapshot_points=128,
    )
    provider = FakeProvider()
    observer = IncrementalTR3DObserver(config, provider)
    observer.reset_scene("scene0000_00", np.eye(4))
    depth = np.ones((8, 8), dtype=np.float32)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    intrinsic = np.asarray([[4, 0, 4], [0, 4, 4], [0, 0, 1]], dtype=np.float64)
    for index in range(5):
        observer.process_keyframe(
            scene_id="scene0000_00", depth=depth, image=image,
            intrinsics=intrinsic, camera_to_world=np.eye(4),
            source_timestamp=index * 25,
        )
    report = observer.finalize()
    assert provider.calls == 3
    assert report["provider_calls"] == 3
    assert report["tracks"] == 2
    assert report["confirmed_tracks"] == 2
    assert all(row["hit_count"] == 3 for row in report["confirmed"])
    assert all(np.asarray(row["best_corners_world"]).shape == (8, 3)
               for row in report["confirmed"])
    assert report["ground_truth_access"] is False
    assert report["schema"] == "boxfusion.tr3d_incremental_online_observer.v3"
    assert report["anchor_count"] == 0
    for row in report["confirmed"]:
        assert row["score_mean"] > 0
        assert row["hit_rate"] == 1.0
        assert row["center_jitter_m"] >= 0
        assert row["point_support"] == 0
