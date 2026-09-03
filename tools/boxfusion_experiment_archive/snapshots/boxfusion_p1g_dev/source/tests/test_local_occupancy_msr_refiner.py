"""CPU tests for the standalone orientation-preserving occupancy/MSR route."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from boxfusion.local_occupancy_msr_refiner import (
    LOCAL_OCCUPANCY_MSR_SOURCE,
    OCCUPANCY_MSR_FEATURE_DIM,
    OCCUPANCY_MSR_FEATURE_NAMES,
    propose_local_occupancy_msr,
    resolve_local_occupancy_msr_config,
)


_SIGNS = np.asarray(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class View:
    frame_id: int
    points_world: np.ndarray
    camera_position: np.ndarray
    quality: float
    valid_depth_ratio: float


def _rotation_z(degrees):
    angle = np.deg2rad(degrees)
    return np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _corners(center, dimensions, basis):
    return (
        np.asarray(center)[None, :]
        + (_SIGNS * (0.5 * np.asarray(dimensions)[None, :])) @ basis.T
    )


def _grid(
    lower=(-0.30, -0.20, -0.17),
    upper=(0.30, 0.20, 0.17),
    counts=(13, 11, 9),
):
    return np.stack(
        np.meshgrid(
            *[
                np.linspace(lower[axis], upper[axis], counts[axis])
                for axis in range(3)
            ],
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)


def _views(
    points_local,
    *,
    center=np.zeros(3),
    basis=np.eye(3),
    cameras=(
        (-2.0, -2.0, -2.0),
        (2.0, 2.0, 2.0),
        (-2.0, 2.0, 2.0),
        (2.0, -2.0, -2.0),
    ),
):
    result = []
    for index, camera_local in enumerate(cameras):
        shifted = np.array(points_local, dtype=np.float64, copy=True)
        shifted[:, (index + 1) % 3] += (index - 1.5) * 0.00025
        points_world = shifted @ basis.T + np.asarray(center)[None, :]
        camera_world = (
            np.asarray(camera_local, dtype=np.float64) @ basis.T
            + np.asarray(center)
        )
        result.append(
            View(
                frame_id=10 + index,
                points_world=points_world,
                camera_position=camera_world,
                quality=0.95 - 0.03 * index,
                valid_depth_ratio=0.92,
            )
        )
    return result


def _config(**updates):
    values = {
        "max_points_per_view": 512,
        "min_points_per_view": 20,
        "min_total_points": 80,
        "min_component_points": 48,
        "face_min_points_per_view": 8,
    }
    values.update(updates)
    return values


def test_public_schema_and_fail_open_identity_are_fixed_finite_and_read_only():
    assert OCCUPANCY_MSR_FEATURE_DIM == 48
    assert len(OCCUPANCY_MSR_FEATURE_NAMES) == 48
    assert len(set(OCCUPANCY_MSR_FEATURE_NAMES)) == 48
    assert "face_support_x_min" in OCCUPANCY_MSR_FEATURE_NAMES
    assert "face_uncertainty_ratio_z_max" in OCCUPANCY_MSR_FEATURE_NAMES
    assert "face_visibility_y_min" in OCCUPANCY_MSR_FEATURE_NAMES
    assert "component_inside_fraction" in OCCUPANCY_MSR_FEATURE_NAMES

    corners = _corners(np.zeros(3), (1.0, 0.8, 0.7), np.eye(3)).astype(
        np.float32
    )
    proposal = propose_local_occupancy_msr(
        corners, _views(_grid())[:1], _config()
    )

    assert proposal.source == LOCAL_OCCUPANCY_MSR_SOURCE
    assert proposal.reason == "identity_insufficient_views"
    assert not proposal.is_candidate
    np.testing.assert_array_equal(proposal.candidate_corners, corners)
    assert proposal.candidate_corners.dtype == corners.dtype
    assert proposal.feature_vector.shape == (48,)
    assert np.isfinite(proposal.feature_vector).all()
    assert proposal.feature_vector.flags.writeable is False
    assert proposal.candidate_corners.flags.writeable is False
    assert proposal.gate_feature_names == OCCUPANCY_MSR_FEATURE_NAMES

    with pytest.raises(ValueError, match="Unknown"):
        resolve_local_occupancy_msr_config({"fine_voxel_szie": 0.02})


def test_disconnected_background_leakage_does_not_move_object_faces():
    dimensions = np.asarray([1.0, 0.8, 0.7])
    corners = _corners(np.zeros(3), dimensions, np.eye(3))
    foreground = _grid()
    # Coherent, multi-view background leakage is within the local crop but
    # separated from the foreground by an empty fine-grid gap.
    background = _grid(
        lower=(0.54, -0.10, -0.10),
        upper=(0.63, 0.10, 0.10),
        counts=(7, 7, 7),
    )
    proposal = propose_local_occupancy_msr(
        corners,
        _views(np.concatenate((foreground, background))),
        _config(),
    )

    assert proposal.reason == "candidate"
    assert proposal.component_count >= 2
    assert proposal.component_inside_fraction > 0.99
    assert np.max(proposal.local_points[:, 0]) < 0.40
    # The selected +X surface is the foreground at x ~= 0.30, not leakage.
    assert proposal.face_residuals[0, 1] < 0.0


def test_close_neighbor_is_split_by_fine_then_coarse_26_connectivity():
    corners = _corners(
        np.zeros(3), np.asarray([1.0, 0.8, 0.7]), np.eye(3)
    )
    foreground = _grid()
    neighboring_object = _grid(
        lower=(0.40, -0.12, -0.12),
        upper=(0.53, 0.12, 0.12),
        counts=(9, 9, 9),
    )
    proposal = propose_local_occupancy_msr(
        corners,
        _views(np.concatenate((foreground, neighboring_object))),
        _config(),
    )

    assert proposal.reason == "candidate"
    assert proposal.component_count >= 2
    assert np.max(proposal.local_points[:, 0]) < 0.35
    assert proposal.component_view_count == 4


def test_partial_views_change_only_the_supported_faces():
    dimensions = np.asarray([1.0, 0.8, 0.7])
    corners = _corners(np.zeros(3), dimensions, np.eye(3))
    cameras = (
        (-2.0, -1.0, -1.0),
        (-2.0, 1.0, 1.0),
        (-2.0, -1.0, 1.0),
    )
    proposal = propose_local_occupancy_msr(
        corners,
        _views(_grid(), cameras=cameras),
        _config(),
    )

    assert proposal.reason == "candidate"
    assert proposal.face_supported[0, 0]
    assert not proposal.face_supported[0, 1]
    assert proposal.face_empty_evidence[0, 0] > 0.0
    assert proposal.face_residuals[0, 0] > 0.0
    assert proposal.face_residuals[0, 1] == 0.0
    candidate_upper_x = (
        proposal.candidate_local_box[0]
        + 0.5 * proposal.candidate_local_box[3]
    )
    assert candidate_upper_x == pytest.approx(0.5 * dimensions[0], abs=1e-12)
    for axis in range(3):
        for face in range(2):
            if not proposal.face_supported[axis, face]:
                assert proposal.face_residuals[axis, face] == 0.0


def test_rotated_obb_retains_original_basis_and_yaw_under_clamps():
    basis = _rotation_z(37.0)
    center = np.asarray([1.2, -0.7, 2.4])
    dimensions = np.asarray([1.0, 0.8, 0.7])
    corners = _corners(center, dimensions, basis)
    proposal = propose_local_occupancy_msr(
        corners,
        _views(_grid(), center=center, basis=basis),
        _config(),
    )

    assert proposal.reason == "candidate"
    np.testing.assert_allclose(proposal.frame_basis, basis, atol=1e-12)
    candidate_edges = np.stack(
        (
            proposal.candidate_corners[1] - proposal.candidate_corners[0],
            proposal.candidate_corners[3] - proposal.candidate_corners[0],
            proposal.candidate_corners[4] - proposal.candidate_corners[0],
        ),
        axis=1,
    )
    candidate_basis = candidate_edges / np.linalg.norm(
        candidate_edges, axis=0
    )[None, :]
    np.testing.assert_allclose(candidate_basis, basis, atol=2e-7)
    assert np.all(proposal.extent_ratios >= 0.70 - 1e-12)
    assert np.all(proposal.extent_ratios <= 1.25 + 1e-12)
    assert np.all(proposal.center_shift_ratios <= 0.15 + 1e-12)
    assert np.all(
        np.abs(proposal.face_residuals)
        <= 0.18 * dimensions[:, None] + 1e-12
    )


def test_view_and_point_permutations_are_exactly_deterministic():
    basis = _rotation_z(23.0)
    center = np.asarray([0.3, -0.4, 1.7])
    corners = _corners(center, (1.0, 0.8, 0.7), basis)
    views = _views(_grid(), center=center, basis=basis)
    first = propose_local_occupancy_msr(corners, views, _config())

    rng = np.random.default_rng(91)
    shuffled = [
        View(
            frame_id=view.frame_id,
            points_world=view.points_world[
                rng.permutation(len(view.points_world))
            ],
            camera_position=np.array(view.camera_position, copy=True),
            quality=view.quality,
            valid_depth_ratio=view.valid_depth_ratio,
        )
        for view in reversed(views)
    ]
    second = propose_local_occupancy_msr(corners, shuffled, _config())

    assert first.reason == second.reason == "candidate"
    np.testing.assert_array_equal(
        first.candidate_corners, second.candidate_corners
    )
    np.testing.assert_array_equal(first.local_points, second.local_points)
    np.testing.assert_array_equal(first.face_residuals, second.face_residuals)
    np.testing.assert_array_equal(first.feature_vector, second.feature_vector)
    assert first.face_reasons == second.face_reasons
    assert first.selected_frame_ids == second.selected_frame_ids


def test_invalid_mask_depth_record_fails_open_with_detailed_reason():
    corners = _corners(
        np.zeros(3), np.asarray([1.0, 0.8, 0.7]), np.eye(3)
    ).astype(np.float32)
    invalid = {
        "frame_id": 1,
        "mask_depth_points_world": np.asarray([[np.nan, 0.0, 0.0]]),
        "camera_position": np.zeros(3),
    }
    proposal = propose_local_occupancy_msr(
        corners, [invalid], _config()
    )

    assert proposal.reason == "identity_invalid_view_record"
    assert proposal.detail_reasons
    assert "finite numeric shape" in proposal.detail_reasons[0]
    np.testing.assert_array_equal(proposal.candidate_corners, corners)
    assert np.isfinite(proposal.feature_vector).all()
