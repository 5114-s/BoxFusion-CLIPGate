"""Contract tests for exact K=5 B5-v2 training diagnostics."""

from __future__ import annotations

import math
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from boxfusion.object_memory import (
    MemoryViewRecord,
    ObjectGeometryMemory,
    aabb_corners,
)
from boxfusion.online_ablation import apply_online_ablation_profile
from boxfusion.online_refinement import (
    DEFAULT_ONLINE_REFINEMENT_CONFIG,
    EvidenceStats,
    GlobalEvidence,
    OnlineRefinementController,
    ViewEvidence,
    _oriented_box_frame,
)
from boxfusion.quality_score import QUALITY_FEATURE_NAMES


class _NoopProvider:
    def predict(self, images, *, frame_ids=None):
        return [[] for _ in images]


class _CaptureRefiner(torch.nn.Module):
    def __init__(self, *, output_quality: float = 0.0) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.output_quality = float(output_quality)
        self.config = SimpleNamespace(
            max_center_fraction=0.15,
            max_log_dimension_residual=math.log(1.25),
            minimum_dimension=1e-3,
        )
        self.captured = None

    def forward(self, points, boxes, quality_features, point_mask):
        self.captured = tuple(
            item.detach().cpu().numpy().copy()
            for item in (points, boxes, quality_features, point_mask)
        )
        batch = points.shape[0]
        zeros = torch.zeros(
            (batch, 3), dtype=points.dtype, device=points.device
        )
        quality = torch.full(
            (batch,),
            self.output_quality,
            dtype=points.dtype,
            device=points.device,
        )
        return {
            "center_residual": zeros,
            "log_dimension_residual": zeros,
            "quality": quality,
        }


def _base_config(tmp_path):
    online = deepcopy(DEFAULT_ONLINE_REFINEMENT_CONFIG)
    online["enabled"] = False
    online["supplemental_proposals"] = {"enabled": False}
    online["object_memory"] = {
        "enabled": True,
        "voxel_size": 0.0,
    }
    online["diagnostics"].update(
        {
            "enabled": True,
            "dump_track_memory": True,
            "root": str(tmp_path),
            "point_count": 512,
        }
    )
    return {"dataset": "scannet", "online_refinement": online}


def _rotated_geometry():
    yaw = math.radians(29.0)
    basis = np.asarray(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    center = np.asarray([0.25, -0.35, 2.5], dtype=np.float64)
    dimensions = np.asarray([1.4, 0.9, 0.7], dtype=np.float64)
    corners = (
        aabb_corners(np.zeros(3), dimensions) @ basis.T
        + center[None, :]
    ).astype(np.float32)
    rng = np.random.default_rng(17)
    local_points = (
        rng.uniform(-0.48, 0.48, size=(901, 3)) * dimensions[None, :]
    ).astype(np.float32)
    world_points = (
        local_points.astype(np.float64) @ basis.T + center[None, :]
    ).astype(np.float32)
    return corners, world_points


def _evidence(controller, corners, world_points):
    memory = ObjectGeometryMemory(7, controller.object_config)
    # Populate bounded state directly so the test isolates diagnostic
    # serialization rather than depth backprojection/voxelisation.
    memory._points = world_points.copy()
    memory._geometry_points = world_points.copy()
    memory.observation_count = 5
    memory.unique_view_count = 5
    memory.first_frame_id = 10
    memory.last_frame_id = 50
    memory._view_candidates = [
        MemoryViewRecord(
            frame_id=frame_id,
            points_world=world_points[index * 30 : index * 30 + 160],
            quality=0.95 - 0.05 * index,
            confidence=0.90 - 0.04 * index,
            valid_depth_ratio=0.85 - 0.03 * index,
            projection_mask_iou=0.80 - 0.02 * index,
            camera_position=np.asarray(
                [index - 2.0, 0.25 * index, 0.0], dtype=np.float32
            ),
        )
        for index, frame_id in enumerate((10, 20, 30, 40, 50))
    ]
    stats = EvidenceStats(scores=[0.9, 0.8])
    views = []
    for index, frame_id in enumerate((10, 20, 30, 40, 50)):
        intrinsics = np.asarray(
            [
                [100.0 + index, 0.0, 40.0],
                [0.0, 101.0 + index, 30.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = 0.1 * index
        views.append(
            ViewEvidence(
                frame_index=frame_id,
                score=0.95 - 0.05 * index,
                bbox=np.asarray(
                    [5.0 + index, 6.0, 45.0, 46.0 + index],
                    dtype=np.float32,
                ),
                intrinsics=intrinsics,
                camera_to_world=pose,
                image_shape=(60 + index, 80 + index),
                area_ratio=0.2,
            )
        )
    stats.view_records = views
    _, _, _ = _oriented_box_frame(corners)
    world_box_min = corners.min(axis=0)
    world_box_max = corners.max(axis=0)
    world_box = np.concatenate(
        (
            0.5 * (world_box_min + world_box_max),
            world_box_max - world_box_min,
        )
    ).astype(np.float32)
    return GlobalEvidence(
        stable_id=7,
        memory=memory,
        stats=stats,
        detector_score=0.8125,
        last_box=world_box,
    )


def test_b5v2_memory_observer_fixes_identity_runtime_contract(tmp_path):
    source = _base_config(tmp_path)
    original = deepcopy(source)
    profiled = apply_online_ablation_profile(
        source, "b5v2_memory_observer"
    )
    assert source == original
    online = profiled["online_refinement"]

    assert online["ablation_profile"] == "b5v2_memory_observer"
    assert online["object_memory"]["top_k_views"] == 5
    assert online["object_memory"]["max_points_per_object"] == 8192
    assert online["object_memory"]["track_ttl"] == 3
    assert online["candidate_lifecycle"] == {
        "ttl_clock": "provider_call",
        "archive_confirmed": False,
    }
    assert online["box_refiner"]["coordinate_frame"] == "box_local"
    assert online["box_refiner"]["point_count"] == 512
    assert online["output_filter"]["minimum_extent"] == pytest.approx(0.40)
    assert online["refit"]["min_views"] == 2
    assert online["refit"]["min_points"] == 128
    for enabled in (
        online["refit"]["enabled"],
        online["box_refiner"]["enabled"],
        online["quality"]["enabled"],
        online["supplemental_output"]["enabled"],
        online["quality"]["soft_nms"]["enabled"],
    ):
        assert enabled is False


def test_diagnostics_match_runtime_local_inputs_and_view_evidence(tmp_path):
    observer_config = apply_online_ablation_profile(
        _base_config(tmp_path), "b5v2_memory_observer"
    )
    observer = OnlineRefinementController(
        observer_config, provider=_NoopProvider()
    )
    corners, world_points = _rotated_geometry()
    evidence = _evidence(observer, corners, world_points)
    observer.global_tracks[7] = evidence
    score = np.asarray([0.8125], dtype=np.float32)
    result = observer.finalize(
        global_corners=corners[None],
        global_scores=score,
        stable_ids=np.asarray([7], dtype=np.int64),
        scene_id="scene0000_00",
    )
    np.testing.assert_array_equal(result.corners, corners[None])
    np.testing.assert_array_equal(result.scores, score)

    path = tmp_path / "scene0000_00_tracks.npz"
    with np.load(path, allow_pickle=False) as payload:
        assert payload["box_refiner_points_local"].shape == (1, 512, 3)
        assert payload["box_refiner_point_mask"].sum() == 512
        assert payload["box_refiner_gate_points_local"].shape == (
            1,
            8192,
            3,
        )
        assert payload["box_refiner_gate_point_mask"].sum() == 901
        assert payload["box_refiner_frame_valid"].tolist() == [True]
        assert payload["box_refiner_view_valid"].tolist() == [
            [True, True, True, True, True]
        ]
        assert payload["box_refiner_view_frame_ids"].tolist() == [
            [10, 20, 30, 40, 50]
        ]
        np.testing.assert_allclose(
            payload["box_refiner_view_scores"][0],
            np.asarray([0.95, 0.90, 0.85, 0.80, 0.75]),
        )
        np.testing.assert_array_equal(
            payload["box_refiner_view_image_shapes"][0],
            np.asarray(
                [[60, 80], [61, 81], [62, 82], [63, 83], [64, 84]]
            ),
        )
        np.testing.assert_allclose(
            payload["box_refiner_view_bboxes"][0, 2],
            np.asarray([7.0, 6.0, 45.0, 48.0]),
        )
        np.testing.assert_allclose(
            payload["box_refiner_view_intrinsics"][0, 3, 0, 0],
            103.0,
        )
        np.testing.assert_allclose(
            payload["box_refiner_view_camera_to_world"][0, 4, 0, 3],
            0.4,
        )
        diagnostic_points = payload["box_refiner_points_local"].copy()
        diagnostic_mask = payload["box_refiner_point_mask"].copy()
        diagnostic_boxes = payload["box_refiner_local_boxes"].copy()
        np.testing.assert_array_equal(
            payload["quality_features"],
            payload["joint_quality_features"],
        )
        refiner_quality_index = QUALITY_FEATURE_NAMES.index(
            "refiner_quality"
        )
        assert payload["quality_features"][
            0, refiner_quality_index
        ] == pytest.approx(0.5)
        for key, expected in {
            "runtime_diagnostics_schema": "box_refiner_k5_runtime_v1",
            "box_refiner_input_schema": (
                "oriented_local_refiner_input_v1"
            ),
            "online_ablation_profile": "b5v2_memory_observer",
            "candidate_ttl_clock": "provider_call",
            "box_refiner_coordinate_frame": "box_local",
        }.items():
            assert payload[key].item() == expected
        assert payload["candidate_track_ttl"].item() == 3
        assert payload["archive_confirmed_tracks"].item() is False
        assert payload["output_minimum_extent"].item() == pytest.approx(
            0.40
        )
        for key in (
            "mutation_refit_enabled",
            "mutation_box_refiner_enabled",
            "mutation_quality_enabled",
            "mutation_supplemental_output_enabled",
            "mutation_soft_nms_enabled",
        ):
            assert payload[key].item() is False

    active_config = apply_online_ablation_profile(
        _base_config(tmp_path / "active"), "b5v2_refiner_only"
    )
    capture = _CaptureRefiner()
    active = OnlineRefinementController(
        active_config,
        provider=_NoopProvider(),
        box_refiner=capture,
    )
    active_evidence = _evidence(active, corners, world_points)
    active._run_oriented_neural_refiner(
        corners,
        active_evidence,
        {name: 0.0 for name in QUALITY_FEATURE_NAMES},
    )
    runtime_points, runtime_boxes, _, runtime_mask = capture.captured
    np.testing.assert_array_equal(diagnostic_points, runtime_points)
    np.testing.assert_array_equal(diagnostic_mask, runtime_mask)
    np.testing.assert_array_equal(diagnostic_boxes, runtime_boxes)

    summary = result.summary
    assert summary["online_ablation_profile"] == "b5v2_memory_observer"
    assert summary["candidate_ttl_clock"] == "provider_call"
    assert summary["candidate_track_ttl"] == 3
    assert summary["top_k_views"] == 5
    assert summary["box_refiner_point_count"] == 512
    assert summary["box_refiner_gate_point_count"] == 8192
    assert summary["refit_gate_min_reprojection_iou"] == pytest.approx(
        0.20
    )


def test_b5v2_b6_freezes_refiner_quality_for_b6_scoring(tmp_path):
    config = apply_online_ablation_profile(
        _base_config(tmp_path), "b5v2_b6"
    )
    online = config["online_refinement"]
    online["quality"]["blend_with_detector"] = 0.0
    captured_mappings = []

    def capture_quality(mapping):
        captured_mappings.append(dict(mapping))
        return float(mapping["refiner_quality"])

    refiner = _CaptureRefiner(output_quality=0.8125)
    controller = OnlineRefinementController(
        config,
        provider=_NoopProvider(),
        box_refiner=refiner,
        quality_scorer=capture_quality,
    )
    corners, world_points = _rotated_geometry()
    controller.global_tracks[7] = _evidence(
        controller, corners, world_points
    )

    result = controller.finalize(
        global_corners=corners[None],
        global_scores=np.asarray([0.8125], dtype=np.float32),
        stable_ids=np.asarray([7], dtype=np.int64),
        scene_id="scene0000_00",
    )

    # The learned 0.8125 output still clears B5's geometry-quality gate.
    assert result.summary["neural_refits_accepted"] == 1
    # The frozen B6 scorer and exported diagnostics must nevertheless see the
    # exact constant feature used by the B6 training dataset.
    assert len(captured_mappings) == 1
    assert captured_mappings[0]["refiner_quality"] == pytest.approx(0.5)
    refiner_quality_index = QUALITY_FEATURE_NAMES.index(
        "refiner_quality"
    )
    assert result.quality_features[
        0, refiner_quality_index
    ] == pytest.approx(0.5)
    assert result.scores[0] == pytest.approx(0.5)
    assert (
        result.summary["quality_refiner_quality_override"]
        == pytest.approx(0.5)
    )
