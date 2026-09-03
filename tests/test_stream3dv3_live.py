from types import SimpleNamespace

import numpy as np

from boxfusion.stream3dv3_live import Stream3Dv3LiveConfig, Stream3Dv3LiveRoute


class _AlwaysGate:
    def __init__(self, run=True):
        self.run = run
        self.queries = 0
        self.commits = 0

    def query(self, **kwargs):
        self.queries += 1
        return SimpleNamespace(run_discovery=self.run, reason="test")

    def commit(self, query):
        self.commits += 1

    def summary(self):
        return {"queries": self.queries, "runs": self.commits if self.run else 0}


class _BombFastSAM:
    def infer_bgr(self, image):
        raise AssertionError("FastSAM must not run on a skipped keyframe")


class _FakeFastSAM:
    def __init__(self):
        mask = np.zeros((480, 640), dtype=np.bool_)
        mask[160:320, 210:430] = True
        self.result = SimpleNamespace(
            masks=np.stack([mask]),
            confidences=np.asarray([0.95], dtype=np.float32),
            boxes_xyxy=np.asarray([[210, 160, 429, 319]], dtype=np.float32),
            count=1,
        )

    def infer_bgr(self, image):
        return self.result


class _FakeF4:
    def __init__(self):
        self.calls = 0
        self.sources = []

    def infer_batch(self, scene_id, frame_id, image, depth, K, pose, boxes, source_ids):
        self.calls += 1
        self.sources.extend(source_ids)
        rows = tuple(
            SimpleNamespace(
                source_id=source_id,
                valid=True,
                world_center=np.asarray([0.0, 0.0, 2.0], dtype=np.float64),
                local_extent=np.asarray([0.90, 0.60, 0.60], dtype=np.float64),
                world_rotation=np.eye(3, dtype=np.float64),
                confidence=0.95,
            )
            for source_id in source_ids
        )
        return SimpleNamespace(rows=rows)


class _InvalidF4(_FakeF4):
    def infer_batch(self, scene_id, frame_id, image, depth, K, pose, boxes, source_ids):
        self.calls += 1
        self.sources.extend(source_ids)
        return SimpleNamespace(
            rows=tuple(
                SimpleNamespace(
                    source_id=source_id,
                    valid=False,
                    world_center=None,
                    local_extent=None,
                    world_rotation=None,
                    confidence=None,
                )
                for source_id in source_ids
            )
        )


def _config():
    return Stream3Dv3LiveConfig.from_mapping(
        {
            "enabled": True,
            "strict_fresh": True,
            "native_score_lower_bound": 0.125,
            "f0": {"box_shortlist": 4, "prelift_top_k": 2},
            "f4": {
                "max_views_per_track": 2,
                "max_sources_per_batch": 2,
                "min_baseline_m": 0.01,
                "min_view_angle_deg": 0.01,
            },
            "output": {"max_births_per_scene": 2},
            "acceptance": {
                "min_total_views": 5,
                "min_f4_views": 2,
                "min_view_ray_angle_deg": 0.0,
                "min_camera_baseline_m": 0.0,
                "max_center_rms_m": 1.0,
                "max_log_size_mad": 1.0,
                "max_yaw_mad_deg": 180.0,
                "max_normalized_center_std": 2.0,
                "max_center_std_m": 1.0,
                "max_log_size_std": 1.0,
                "min_mask_box_iou": 0.0,
                "min_mask_containment": 0.0,
                "min_point_inside": 0.0,
                "min_depth_support": 0.0,
                "max_free_space": 1.0,
                "min_quality": 0.0,
                "min_hypothesis_margin": 0.0,
            },
        }
    )


def _inputs(frame, camera_x=0.0):
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = camera_x
    return {
        "scene_id": "scene0000_00",
        "frame_id": frame,
        "rgb": np.zeros((480, 640, 3), dtype=np.uint8),
        "depth_m": np.full((480, 640), 2.0, dtype=np.float32),
        "intrinsics": np.asarray(
            [[574.0, 0.0, 320.0], [0.0, 577.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        "camera_to_world": pose,
        "native_boxes_xyxy": np.empty((0, 4), dtype=np.float64),
    }


def _far_native_box():
    lower = np.asarray([8.0, 8.0, 8.0])
    upper = np.asarray([9.0, 9.0, 9.0])
    return lower[None] + np.asarray(
        [
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
        ],
        dtype=np.float64,
    ) * (upper - lower)[None]


def test_skipped_keyframe_does_not_call_fastsam_or_f4():
    f4 = _FakeF4()
    route = Stream3Dv3LiveRoute(
        _config(),
        lifting_adapter=None,
        device="cpu",
        fastsam_provider=_BombFastSAM(),
        f4_provider=f4,
    )
    route._trigger = _AlwaysGate(run=False)
    route.start_pipeline_clock(dataset_frame_count=26, keyframe_gap=25)
    route.poll(0)
    route.process_keyframe(**_inputs(0))
    result = route.finalize(
        native_boxes_3d=np.stack([_far_native_box()]),
        native_scores=np.asarray([0.80]),
        final_frame_id=0,
    )
    assert f4.calls == 0
    assert result.birth_count == 0
    assert result.scores.tolist() == [np.float32(0.8)]
    assert result.diagnostics["counts"]["keyframes_skipped"] == 1


def test_track_first_runs_only_two_f4_batches_then_low_score_birth():
    f4 = _FakeF4()
    route = Stream3Dv3LiveRoute(
        _config(),
        lifting_adapter=None,
        device="cpu",
        fastsam_provider=_FakeFastSAM(),
        f4_provider=f4,
    )
    route._trigger = _AlwaysGate(run=True)
    route.start_pipeline_clock(dataset_frame_count=30, keyframe_gap=25)
    for ordinal, camera_x in enumerate((-0.20, -0.10, 0.00, 0.10, 0.20)):
        route.poll(ordinal)
        route.process_keyframe(**_inputs(ordinal, camera_x))
    result = route.finalize(
        native_boxes_3d=np.stack([_far_native_box()]),
        native_scores=np.asarray([0.80]),
        final_frame_id=4,
    )
    assert f4.calls == 2
    assert result.diagnostics["counts"]["f4_attempts"] == 2
    assert result.diagnostics["counts"]["tracks_frozen"] == 1
    assert result.diagnostics["counts"]["tracks_accepted"] == 1
    assert result.birth_count == 1
    assert result.boxes_3d.shape == (2, 8, 3)
    assert result.scores[0] == np.float32(0.8)
    assert 0.05 < result.scores[1] < result.scores[0]
    assert result.diagnostics["raw_frame_count"] == 5
    assert result.diagnostics["pipeline_seconds"] > 0.0
    assert result.diagnostics["score_audit"]["native_prefix_sha256"] == (
        result.diagnostics["score_audit"]["output_prefix_sha256"]
    )
    assert result.diagnostics["geometry_audit"]["native_prefix_sha256"] == (
        result.diagnostics["geometry_audit"]["output_prefix_sha256"]
    )


def test_invalid_f4_results_still_consume_the_per_track_attempt_budget():
    f4 = _InvalidF4()
    route = Stream3Dv3LiveRoute(
        _config(),
        lifting_adapter=None,
        device="cpu",
        fastsam_provider=_FakeFastSAM(),
        f4_provider=f4,
    )
    route._trigger = _AlwaysGate(run=True)
    route.start_pipeline_clock(dataset_frame_count=31, keyframe_gap=25)
    for ordinal, camera_x in enumerate((-0.20, -0.10, 0.00, 0.10, 0.20, 0.30)):
        route.poll(ordinal)
        route.process_keyframe(**_inputs(ordinal, camera_x))
    result = route.finalize(
        native_boxes_3d=np.stack([_far_native_box()]),
        native_scores=np.asarray([0.80]),
        final_frame_id=5,
    )
    assert f4.calls == 2
    assert result.birth_count == 0
    assert result.diagnostics["counts"]["f4_attempts"] == 2
    assert result.diagnostics["bounded"]["max_f4_attempts_observed"] == 2
    assert max(result.diagnostics["f4_per_track"].values()) == 2
