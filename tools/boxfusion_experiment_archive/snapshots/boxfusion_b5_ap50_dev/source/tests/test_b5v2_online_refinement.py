"""Integration tests for the B5-v2 object-local online path."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from boxfusion.object_memory import (
    ObjectGeometryMemory,
    ObjectObservation,
    aabb_corners,
)
from boxfusion.online_refinement import (
    EvidenceStats,
    GlobalEvidence,
    OnlineRefinementController,
    _oriented_box_frame,
    corners_to_center_size,
    resolve_online_refinement_config,
)
from boxfusion.oriented_box_refiner import (
    OrientedBoxRefinerConfig,
    PointNetOrientedBoxRefiner,
)


class NoopProvider:
    def predict(self, images, *, frame_ids=None):
        return [[] for _ in images]


def _config() -> dict:
    return {
        "dataset": "scannet",
        "online_refinement": {
            "enabled": True,
            "supplemental_proposals": {"enabled": False},
            "object_memory": {
                "top_k_views": 0,
                "voxel_size": 0.0,
                "max_points_per_observation": 128,
                "max_points_per_object": 256,
                "aabb_lower_quantile": 0.0,
                "aabb_upper_quantile": 1.0,
                "min_points_for_aabb": 4,
            },
            "refit": {
                "enabled": False,
                "min_views": 1,
                "min_points": 4,
                "max_center_shift_ratio": 0.20,
                "min_extent_ratio": 0.80,
                "max_extent_ratio": 1.25,
                "min_original_point_support": 0.0,
                "min_candidate_point_support": 0.0,
                "max_candidate_support_drop": 1.0,
                "min_reprojection_iou": 0.0,
                "min_reprojection_improvement": -1.0,
            },
            "box_refiner": {
                "enabled": True,
                "coordinate_frame": "box_local",
                "preserve_orientation": True,
                "point_count": 32,
                "quality_threshold": 0.50,
                "architecture": {},
            },
            "quality": {"enabled": False},
            "supplemental_output": {"enabled": False},
            "diagnostics": {"enabled": False},
        },
    }


def _oriented_corners() -> np.ndarray:
    yaw = math.radians(31.0)
    basis = np.asarray(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    center = np.asarray([1.0, -0.5, 2.0], dtype=np.float32)
    dimensions = np.asarray([1.2, 0.8, 1.0], dtype=np.float32)
    return (
        aabb_corners(np.zeros(3), dimensions) @ basis.T
        + center[None, :]
    ).astype(np.float32)


def _model(*, quality_probability: float) -> PointNetOrientedBoxRefiner:
    model = PointNetOrientedBoxRefiner(OrientedBoxRefinerConfig())
    with torch.no_grad():
        model.output_layer.weight.zero_()
        model.output_layer.bias.zero_()
        # A deterministic non-identity candidate: +5% local-x centre and
        # 10% local-x shrink, both strictly within the architecture bounds.
        model.output_layer.bias[0] = math.atanh(0.05 / 0.15)
        model.output_layer.bias[3] = math.atanh(
            math.log(0.90) / math.log(1.25)
        )
        model.output_layer.bias[6] = math.log(
            quality_probability / (1.0 - quality_probability)
        )
    model.eval()
    return model


def _controller_with_evidence(
    model: PointNetOrientedBoxRefiner,
) -> tuple[OnlineRefinementController, np.ndarray]:
    controller = OnlineRefinementController(
        _config(),
        provider=NoopProvider(),
        box_refiner=model,
    )
    corners = _oriented_corners()
    center, dimensions, basis = _oriented_box_frame(corners)
    values = np.linspace(-0.35, 0.35, 3, dtype=np.float32)
    local_points = np.asarray(
        [
            [x * dimensions[0], y * dimensions[1], z * dimensions[2]]
            for x in values
            for y in values
            for z in values
        ],
        dtype=np.float32,
    )
    world_points = local_points @ basis.T + center[None, :]
    memory = ObjectGeometryMemory(7, controller.object_config)
    memory.add_observation(
        ObjectObservation(
            points_world=world_points,
            confidence=0.9,
            mask_pixels=len(world_points),
            valid_depth_pixels=len(world_points),
            projection_mask_iou=1.0,
        ),
        0,
    )
    world_box = corners_to_center_size(corners[None])[0]
    controller.global_tracks[7] = GlobalEvidence(
        stable_id=7,
        memory=memory,
        stats=EvidenceStats(),
        detector_score=0.8,
        last_box=world_box,
    )
    return controller, corners


def test_b5v2_accepted_candidate_changes_local_geometry_and_preserves_yaw():
    controller, corners = _controller_with_evidence(
        _model(quality_probability=0.99)
    )
    score = np.asarray([0.8125], dtype=np.float32)

    result = controller.finalize(
        global_corners=corners[None],
        global_scores=score,
        stable_ids=np.asarray([7], dtype=np.int64),
    )

    assert result.corners.shape == (1, 8, 3)
    assert not np.array_equal(result.corners[0], corners)
    np.testing.assert_array_equal(result.scores, score)
    _, input_dimensions, input_basis = _oriented_box_frame(corners)
    _, output_dimensions, output_basis = _oriented_box_frame(
        result.corners[0]
    )
    np.testing.assert_allclose(output_basis, input_basis, atol=1e-6)
    assert output_dimensions[0] == pytest.approx(
        0.90 * input_dimensions[0], rel=1e-5
    )
    assert result.summary["neural_refits_attempted"] == 1
    assert result.summary["neural_refits_accepted"] == 1


def test_b5v2_low_quality_candidate_is_exact_identity():
    controller, corners = _controller_with_evidence(
        _model(quality_probability=0.01)
    )
    score = np.asarray([0.8125], dtype=np.float32)

    result = controller.finalize(
        global_corners=corners[None],
        global_scores=score,
        stable_ids=np.asarray([7], dtype=np.int64),
    )

    np.testing.assert_array_equal(result.corners, corners[None])
    np.testing.assert_array_equal(result.scores, score)
    assert result.summary["neural_refits_attempted"] == 1
    assert result.summary["neural_refits_accepted"] == 0
    assert result.summary["neural_refits_quality_rejected"] == 1


def test_box_local_refiner_cannot_disable_orientation_preservation():
    config = _config()
    config["online_refinement"]["box_refiner"][
        "preserve_orientation"
    ] = False

    with pytest.raises(ValueError, match="must preserve"):
        resolve_online_refinement_config(config)
