"""CPU tests for B3-v2 visibility-constrained boundary refinement."""

from __future__ import annotations

import itertools

import numpy as np

from boxfusion.object_memory import (
    ObjectGeometryMemory,
    ObjectObservation,
    aabb_corners,
)
from boxfusion.online_refinement import (
    EvidenceStats,
    GlobalEvidence,
    OnlineRefinementController,
    _deterministic_weighted_median,
    _oriented_box_frame,
)


class NoopProvider:
    def predict(self, images, *, frame_ids=None):
        return [[] for _ in images]


def make_controller():
    config = {
        "dataset": "scannet",
        "online_refinement": {
            "enabled": True,
            "supplemental_proposals": {"enabled": False},
            "object_memory": {
                "top_k_views": 5,
                "max_view_candidates": 12,
                "view_diversity_weight": 0.4,
                "minimum_view_quality": 0.0,
                "voxel_size": 0.0,
                "max_points_per_observation": 256,
                "max_points_per_object": 2048,
                "aabb_lower_quantile": 0.02,
                "aabb_upper_quantile": 0.98,
                "min_points_for_aabb": 16,
            },
            "refit": {
                "enabled": True,
                "strategy": "visibility_aware",
                "min_views": 3,
                "min_points": 32,
                "blend": 0.5,
                "extent_padding": 0.0,
                "minimum_view_separation_degrees": 60.0,
                "minimum_axis_cosine": 0.45,
                "minimum_bilateral_axes": 1,
                "minimum_side_views": 1,
                "max_boundary_shift_ratio": 0.08,
                "minimum_boundary_change_ratio": 0.005,
                "visibility_boundary_quantile": 0.05,
                "visibility_point_crop_expansion": 1.2,
                "minimum_camera_outside_ratio": 0.02,
                "maximum_boundary_measurement_spread_ratio": 0.10,
                "enable_silhouette_axes": True,
                "maximum_silhouette_axis_cosine": 0.30,
                "minimum_silhouette_views": 2,
                "minimum_silhouette_separation_degrees": 45.0,
                "max_center_shift_ratio": 0.08,
                "min_extent_ratio": 0.90,
                "max_extent_ratio": 1.0,
                "min_original_point_support": 0.60,
                "min_candidate_point_support": 0.85,
                "max_candidate_support_drop": 0.05,
                "min_reprojection_iou": 0.0,
                "min_reprojection_improvement": -1.0,
            },
            "box_refiner": {"enabled": False},
            "quality": {"enabled": False},
            "supplemental_output": {"enabled": False},
            "diagnostics": {"enabled": False},
        },
    }
    return OnlineRefinementController(
        config,
        provider=NoopProvider(),
    )


def surface_points(axis, value):
    values = np.linspace(-0.8, 0.8, 8, dtype=np.float32)
    points = []
    for first, second in itertools.product(values, repeat=2):
        point = np.zeros(3, dtype=np.float32)
        point[axis] = float(value)
        other = [index for index in range(3) if index != axis]
        point[other[0]] = first
        point[other[1]] = second
        points.append(point)
    return np.asarray(points, dtype=np.float32)


def silhouette_points(axis):
    values = np.linspace(-0.8, 0.8, 8, dtype=np.float32)
    other = [index for index in range(3) if index != axis]
    points = []
    for boundary_value, transverse_value in itertools.product(
        values, repeat=2
    ):
        point = np.zeros(3, dtype=np.float32)
        point[axis] = boundary_value
        point[other[0]] = transverse_value
        points.append(point)
    return np.asarray(points, dtype=np.float32)


def make_evidence(controller, views):
    memory = ObjectGeometryMemory(7, controller.object_config)
    for frame_id, (points, camera) in enumerate(views):
        memory.add_observation(
            ObjectObservation(
                points_world=points,
                confidence=0.9,
                mask_pixels=len(points),
                valid_depth_pixels=len(points),
                projection_mask_iou=1.0,
                camera_position=camera,
            ),
            frame_id,
        )
    original = np.asarray(
        [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        dtype=np.float32,
    )
    return original, GlobalEvidence(
        stable_id=7,
        memory=memory,
        stats=EvidenceStats(),
        detector_score=0.8,
        last_box=original.copy(),
    )


def test_weighted_median_is_deterministic_at_exact_half():
    result = _deterministic_weighted_median(
        np.asarray([2.0, 1.0]),
        np.asarray([0.5, 0.5]),
        np.asarray([20, 10]),
    )
    assert result == 1.0


def test_same_side_views_cannot_shrink_an_axis():
    controller = make_controller()
    points = surface_points(0, -0.8)
    original, evidence = make_evidence(
        controller,
        [
            (points, np.asarray([-4.0, 0.0, 0.0])),
            (points, np.asarray([-4.0, 0.5, 0.0])),
            (points, np.asarray([-4.0, -0.5, 0.0])),
        ],
    )

    candidate, reason = controller._visibility_aware_candidate(
        original, evidence
    )

    assert reason in {"visibility_views", "visibility_axes"}
    np.testing.assert_array_equal(candidate, original)


def test_opposing_views_shrink_only_the_supported_axis():
    controller = make_controller()
    controller.config["refit"]["enable_silhouette_axes"] = False
    lower = surface_points(0, -0.8)
    upper = surface_points(0, 0.8)
    original, evidence = make_evidence(
        controller,
        [
            (lower, np.asarray([-4.0, 0.0, 0.0])),
            (upper, np.asarray([4.0, 0.0, 0.0])),
            (lower, np.asarray([-4.0, 0.5, 0.0])),
        ],
    )

    candidate, reason = controller._visibility_aware_candidate(
        original, evidence
    )

    assert reason == "candidate"
    assert candidate[3] < original[3]
    assert candidate[3] >= 0.90 * original[3] - 1e-6
    np.testing.assert_array_equal(candidate[1:3], original[1:3])
    np.testing.assert_array_equal(candidate[4:6], original[4:6])


def test_separated_silhouette_views_shrink_only_the_supported_axis():
    controller = make_controller()
    points = silhouette_points(0)
    original, evidence = make_evidence(
        controller,
        [
            (points, np.asarray([0.0, -4.0, 0.0])),
            (points, np.asarray([0.0, 0.0, 4.0])),
            (points, np.asarray([0.0, -4.0, 0.5])),
        ],
    )

    candidate, reason = controller._visibility_aware_candidate(
        original, evidence
    )

    assert reason == "candidate"
    assert candidate[3] < original[3]
    assert candidate[3] >= 0.90 * original[3] - 1e-6
    np.testing.assert_array_equal(candidate[1:3], original[1:3])
    np.testing.assert_array_equal(candidate[4:6], original[4:6])


def test_near_duplicate_silhouette_views_cannot_shrink_an_axis():
    controller = make_controller()
    points = silhouette_points(0)
    original, evidence = make_evidence(
        controller,
        [
            (points, np.asarray([0.0, -4.0, 0.0])),
            (points, np.asarray([0.0, -4.0, 0.2])),
            (points, np.asarray([0.0, -4.0, -0.2])),
        ],
    )

    candidate, reason = controller._visibility_aware_candidate(
        original, evidence
    )

    assert reason == "visibility_axes"
    np.testing.assert_array_equal(candidate, original)


def test_best_silhouette_pair_ignores_one_inconsistent_view():
    controller = make_controller()
    controller.config["refit"].update(
        {
            "minimum_axis_cosine": 1.0,
            "maximum_silhouette_axis_cosine": 0.40,
            "minimum_silhouette_separation_degrees": 45.0,
        }
    )
    consistent = silhouette_points(0)
    outlier = consistent.copy()
    outlier[:, 0] *= 0.25
    original, evidence = make_evidence(
        controller,
        [
            (consistent, np.asarray([0.0, -4.0, 0.0])),
            (consistent, np.asarray([0.0, -2.0, 3.464])),
            (outlier, np.asarray([0.0, -3.464, -2.0])),
        ],
    )

    controller.config["refit"]["select_best_silhouette_pair"] = False
    all_views, all_reason = controller._visibility_aware_candidate(
        original, evidence
    )
    controller.config["refit"]["select_best_silhouette_pair"] = True
    best_pair, pair_reason = controller._visibility_aware_candidate(
        original, evidence
    )

    assert all_reason == "visibility_axes"
    np.testing.assert_array_equal(all_views, original)
    assert pair_reason == "candidate"
    assert best_pair[3] < original[3]
    np.testing.assert_array_equal(best_pair[4:6], original[4:6])


def test_missing_pose_and_camera_inside_box_are_not_visibility_evidence():
    controller = make_controller()
    controller.config["refit"]["enable_silhouette_axes"] = False
    lower = surface_points(0, -0.8)
    upper = surface_points(0, 0.8)
    original, evidence = make_evidence(
        controller,
        [
            (lower, np.asarray([-4.0, 0.0, 0.0])),
            (upper, np.asarray([0.5, 0.0, 0.0])),
            (lower, np.asarray([-4.0, 0.5, 0.0])),
        ],
    )
    # A missing pose is retained in legacy memory but cannot provide a side.
    evidence.memory.add_observation(
        ObjectObservation(
            upper,
            confidence=1.0,
            camera_position=None,
        ),
        frame_id=9,
    )

    candidate, reason = controller._visibility_aware_candidate(
        original, evidence
    )

    assert reason in {"visibility_views", "visibility_axes"}
    np.testing.assert_array_equal(candidate, original)


def test_candidate_support_gate_rejects_a_box_that_discards_observed_points():
    controller = make_controller()
    lower = surface_points(0, -0.8)
    upper = surface_points(0, 0.8)
    original, evidence = make_evidence(
        controller,
        [
            (lower, np.asarray([-4.0, 0.0, 0.0])),
            (upper, np.asarray([4.0, 0.0, 0.0])),
            (lower, np.asarray([-4.0, 0.5, 0.0])),
        ],
    )
    controller.config["refit"]["min_extent_ratio"] = 0.4
    bad_candidate = original.copy()
    bad_candidate[3] = 1.0

    accepted, reason = controller._refit_gate(
        original, bad_candidate, evidence
    )

    assert accepted is False
    assert reason in {"candidate_support", "candidate_support_drop"}


def opposing_evidence(controller):
    return make_evidence(
        controller,
        [
            (
                surface_points(0, -0.8),
                np.asarray([-4.0, 0.0, 0.0]),
            ),
            (
                surface_points(0, 0.8),
                np.asarray([4.0, 0.0, 0.0]),
            ),
            (
                surface_points(0, -0.8),
                np.asarray([-4.0, 0.5, 0.0]),
            ),
        ],
    )


def test_finalize_refit_does_not_mutate_tracking_state_or_inputs():
    controller = make_controller()
    original, evidence = opposing_evidence(controller)
    controller.global_tracks[7] = evidence
    corners = aabb_corners(original[:3], original[3:6])[None]
    corners_before = corners.copy()
    last_box_before = evidence.last_box.copy()
    frames_before = evidence.memory.selected_view_frame_ids

    result = controller.finalize(
        global_corners=corners,
        global_scores=np.asarray([0.8], dtype=np.float32),
        stable_ids=np.asarray([7], dtype=np.int64),
    )

    assert result.summary["refits_accepted"] == 1
    assert result.boxes[0, 3] < original[3]
    np.testing.assert_array_equal(corners, corners_before)
    np.testing.assert_array_equal(evidence.last_box, last_box_before)
    assert evidence.memory.selected_view_frame_ids == frames_before


def test_oriented_finalize_refits_locally_and_preserves_input_orientation():
    controller = make_controller()
    controller.config["refit"]["preserve_box_orientation"] = True
    controller.config["refit"]["enable_silhouette_axes"] = False
    angle = np.deg2rad(37.0)
    basis = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    center = np.asarray([1.5, -0.75, 2.0], dtype=np.float32)
    local_original = np.asarray(
        [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        dtype=np.float32,
    )
    local_corners = aabb_corners(
        local_original[:3], local_original[3:6]
    )
    world_corners = local_corners @ basis.T + center
    local_views = [
        (surface_points(0, -0.8), np.asarray([-4.0, 0.0, 0.0])),
        (surface_points(0, 0.8), np.asarray([4.0, 0.0, 0.0])),
        (surface_points(0, -0.8), np.asarray([-4.0, 0.5, 0.0])),
    ]
    world_views = [
        (
            points @ basis.T + center,
            camera @ basis.T + center,
        )
        for points, camera in local_views
    ]
    _, evidence = make_evidence(controller, world_views)
    controller.global_tracks[7] = evidence

    result = controller.finalize(
        global_corners=world_corners[None],
        global_scores=np.asarray([0.8], dtype=np.float32),
        stable_ids=np.asarray([7], dtype=np.int64),
    )

    assert result.summary["refits_accepted"] == 1
    input_center, input_dims, input_basis = _oriented_box_frame(
        world_corners
    )
    output_center, output_dims, output_basis = _oriented_box_frame(
        result.corners[0]
    )
    np.testing.assert_allclose(output_basis, input_basis, atol=1e-6)
    assert output_dims[0] < input_dims[0]
    np.testing.assert_allclose(output_dims[1:], input_dims[1:], atol=1e-6)
    np.testing.assert_allclose(
        result.boxes,
        np.concatenate(
            (
                result.corners.mean(axis=1),
                result.corners.max(axis=1)
                - result.corners.min(axis=1),
            ),
            axis=1,
        ),
        atol=1e-6,
    )
    assert np.linalg.norm(output_center - input_center) > 0.0


def test_b3v2_b6_original_geometry_features_keep_scores_bit_exact():
    control = make_controller()
    refined = make_controller()
    original, control_evidence = opposing_evidence(control)
    _, refined_evidence = opposing_evidence(refined)
    control.global_tracks[7] = control_evidence
    refined.global_tracks[7] = refined_evidence
    control.config["refit"]["enabled"] = False
    for controller in (control, refined):
        controller.config["quality"]["enabled"] = True
        controller.config["quality"]["feature_geometry"] = "original"
        controller.config["quality"]["blend_with_detector"] = 0.0
        controller.config["quality"]["soft_nms"]["enabled"] = False
        controller.quality_scorer = (
            lambda mapping: 0.25
            + 0.5 * float(mapping["geometry_consistency"])
        )
    corners = aabb_corners(original[:3], original[3:6])[None]
    kwargs = {
        "global_corners": corners,
        "global_scores": np.asarray([0.8], dtype=np.float32),
        "stable_ids": np.asarray([7], dtype=np.int64),
    }

    control_result = control.finalize(**kwargs)
    refined_result = refined.finalize(**kwargs)

    np.testing.assert_array_equal(
        refined_result.scores, control_result.scores
    )
    np.testing.assert_array_equal(
        refined_result.quality_features,
        control_result.quality_features,
    )
    assert refined_result.boxes[0, 3] < control_result.boxes[0, 3]
