import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "boxfusion"
    / "generic_local_geometry_refiner.py"
)
SPEC = importlib.util.spec_from_file_location(
    "boxfusion_generic_local_geometry_refiner", SOURCE
)
refiner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = refiner
SPEC.loader.exec_module(refiner)


@dataclass(frozen=True)
class View:
    frame_id: int
    points_world: np.ndarray
    quality: float
    valid_depth_ratio: float
    camera_position: np.ndarray


def object_points(
    lower=(-0.30, -0.20, -0.20),
    upper=(0.30, 0.20, 0.20),
    *,
    offset=(0.0, 0.0, 0.0),
):
    axes = [
        np.linspace(lower[index], upper[index], (9, 7, 7)[index])
        + offset[index]
        for index in range(3)
    ]
    return np.asarray(
        np.meshgrid(*axes, indexing="ij"), dtype=np.float64
    ).reshape(3, -1).T


def views_from_points(points, cameras=None):
    if cameras is None:
        cameras = (
            (-2.0, -2.0, -2.0),
            (2.0, 2.0, 2.0),
            (-2.0, 2.0, -2.0),
            (2.0, -2.0, 2.0),
        )
    result = []
    for index, camera in enumerate(cameras):
        # Tiny deterministic sub-voxel shifts model independent depth samples
        # while retaining fine-voxel cross-view consensus.
        shifted = np.array(points, copy=True)
        shifted[:, (index + 1) % 3] += (index - 1.5) * 0.0005
        result.append(
            View(
                frame_id=index,
                points_world=shifted,
                quality=0.9 - 0.03 * index,
                valid_depth_ratio=0.95,
                camera_position=np.asarray(camera, dtype=np.float64),
            )
        )
    return result


def permissive(**updates):
    config = {
        "min_points_per_view": 24,
        "min_total_points": 96,
        "boundary_min_points_per_view": 8,
    }
    config.update(updates)
    return config


def test_default_contract_and_strict_config():
    config = refiner.resolve_generic_local_geometry_config()
    assert config["max_views"] == 5
    assert config["max_points_per_view"] == 512
    assert config["crop_scale"] == pytest.approx(1.20)
    assert config["fine_voxel_size"] == pytest.approx(0.04)
    assert config["coarse_voxel_size"] == pytest.approx(0.06)
    assert config["min_views"] == 2
    assert config["min_points_per_view"] == 48
    assert config["min_total_points"] == 192
    assert config["min_component_views"] == 2
    assert config["min_component_inside_fraction"] == pytest.approx(0.50)
    assert config["boundary_min_views"] == 2
    assert config["minimum_extent_ratio"] == pytest.approx(0.75)
    assert config["maximum_extent_ratio"] == pytest.approx(1.25)
    assert config["maximum_center_shift_ratio"] == pytest.approx(0.15)
    assert config["maximum_support_drop"] == pytest.approx(0.05)

    with pytest.raises(ValueError, match="Unknown"):
        refiner.resolve_generic_local_geometry_config(
            {"fine_voxel_szie": 0.04}
        )
    with pytest.raises(ValueError, match="mapping"):
        refiner.resolve_generic_local_geometry_config([])

    source = {"min_views": 3}
    detached = refiner.resolve_generic_local_geometry_config(source)
    detached["min_views"] = 1
    assert source == {"min_views": 3}


@pytest.mark.parametrize(
    "bad",
    [
        {"max_views": True},
        {"max_views": 0},
        {"min_views": 6},
        {"crop_scale": 0.99},
        {"fine_voxel_size": 0.0},
        {"fine_min_view_consensus": 6},
        {"min_component_inside_fraction": 1.01},
        {"lower_quantile": 0.99, "upper_quantile": 0.98},
        {"boundary_max_spread_floor": 0.2},
        {"boundary_blend": -0.1},
        {"minimum_extent_ratio": 1.3, "maximum_extent_ratio": 1.2},
        {"maximum_support_drop": np.nan},
    ],
)
def test_invalid_config_fails_fast(bad):
    with pytest.raises(ValueError):
        refiner.resolve_generic_local_geometry_config(bad)


def test_insufficient_and_invalid_evidence_are_exact_identity():
    box = np.asarray([0, 0, 0, 0.8, 0.6, 0.5], dtype=np.float32)
    one = views_from_points(object_points())[:1]
    proposal = refiner.propose_generic_local_geometry(
        box, one, permissive()
    )
    assert proposal.reason == "identity_insufficient_views"
    np.testing.assert_array_equal(proposal.candidate, box)

    invalid = [
        View(
            frame_id=0,
            points_world=np.asarray([[np.nan, 0.0, 0.0]]),
            quality=1.0,
            valid_depth_ratio=1.0,
            camera_position=np.zeros(3),
        )
    ]
    proposal = refiner.propose_generic_local_geometry(
        box, invalid, permissive()
    )
    assert proposal.reason == "identity_invalid_view_record"
    np.testing.assert_array_equal(proposal.candidate, box)


def test_multiview_component_produces_conservative_candidate_and_diagnostics():
    box = np.asarray([0, 0, 0, 0.8, 0.6, 0.5], dtype=np.float32)
    views = views_from_points(object_points())
    proposal = refiner.propose_generic_local_geometry(
        box, views, permissive()
    )

    assert proposal.reason == "candidate"
    assert proposal.selected_view_count == 4
    assert proposal.component_view_count == 4
    assert proposal.component_inside_fraction > 0.99
    assert proposal.consensus_point_count >= 96
    assert proposal.anchor_point_count > 0
    assert np.any(proposal.boundary_visible)
    assert np.all(proposal.extent_ratios >= 0.75)
    assert np.all(proposal.extent_ratios <= 1.25)
    assert proposal.support_drop <= 0.05
    assert proposal.candidate.flags.writeable is False
    assert proposal.points.flags.writeable is False
    assert proposal.boundary_values.flags.writeable is False


def test_disconnected_background_outlier_is_not_used():
    box = np.asarray([0, 0, 0, 1.0, 0.8, 0.8], dtype=np.float64)
    foreground = object_points(
        lower=(-0.32, -0.22, -0.18),
        upper=(0.32, 0.22, 0.18),
    )
    # This coherent distractor lies inside the 1.2 crop, but is disconnected
    # and has fewer in-original-box points than the foreground anchor.
    distractor = object_points(
        lower=(0.51, -0.08, -0.08),
        upper=(0.57, 0.08, 0.08),
    )
    views = views_from_points(np.concatenate((foreground, distractor)))
    proposal = refiner.propose_generic_local_geometry(
        box,
        views,
        permissive(
            coarse_voxel_size=0.10,
            component_merge_gap=0.04,
            minimum_extent_ratio=0.70,
        ),
    )

    assert proposal.reason == "candidate"
    assert np.max(proposal.points[:, 0]) < 0.5
    assert proposal.boundary_values[0, 1] < 0.5


def test_point_and_view_permutations_are_exactly_deterministic():
    box = np.asarray([0, 0, 0, 0.8, 0.6, 0.5], dtype=np.float64)
    views = views_from_points(object_points())
    first = refiner.propose_generic_local_geometry(
        box, views, permissive()
    )
    rng = np.random.default_rng(19)
    shuffled = []
    for view in reversed(views):
        shuffled.append(
            View(
                frame_id=view.frame_id,
                points_world=view.points_world[
                    rng.permutation(len(view.points_world))
                ],
                quality=view.quality,
                valid_depth_ratio=view.valid_depth_ratio,
                camera_position=np.array(view.camera_position, copy=True),
            )
        )
    second = refiner.propose_generic_local_geometry(
        box, shuffled, permissive()
    )

    assert first.reason == second.reason == "candidate"
    np.testing.assert_array_equal(first.candidate, second.candidate)
    np.testing.assert_array_equal(first.points, second.points)
    np.testing.assert_array_equal(
        first.boundary_values, second.boundary_values
    )
    np.testing.assert_array_equal(
        first.boundary_view_counts, second.boundary_view_counts
    )
    assert first.selected_frame_ids == second.selected_frame_ids


def test_single_camera_side_never_changes_unseen_opposite_face():
    box = np.asarray([0, 0, 0, 0.8, 0.6, 0.5], dtype=np.float64)
    points = object_points(
        lower=(-0.34, -0.20, -0.18),
        upper=(0.24, 0.20, 0.18),
    )
    cameras = (
        (-2.0, -2.0, -2.0),
        (-2.0, 2.0, 2.0),
        (-2.0, -2.0, 2.0),
    )
    proposal = refiner.propose_generic_local_geometry(
        box, views_from_points(points, cameras), permissive()
    )

    assert proposal.reason == "candidate"
    original_upper_x = box[0] + 0.5 * box[3]
    assert proposal.boundary_visible[0, 0]
    assert not proposal.boundary_visible[0, 1]
    assert proposal.boundary_values[0, 1] == original_upper_x


def test_inputs_are_never_modified():
    box = np.asarray([0, 0, 0, 0.8, 0.6, 0.5], dtype=np.float32)
    views = views_from_points(object_points())
    original_box = box.copy()
    original_points = [view.points_world.copy() for view in views]
    original_cameras = [view.camera_position.copy() for view in views]

    refiner.propose_generic_local_geometry(box, views, permissive())

    np.testing.assert_array_equal(box, original_box)
    for view, points, camera in zip(
        views, original_points, original_cameras
    ):
        np.testing.assert_array_equal(view.points_world, points)
        np.testing.assert_array_equal(view.camera_position, camera)
