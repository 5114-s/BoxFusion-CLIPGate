import importlib.util
import itertools
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "boxfusion"
    / "depth_occupancy_refiner.py"
)
SPEC = importlib.util.spec_from_file_location(
    "boxfusion_depth_occupancy_refiner", SOURCE
)
refiner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = refiner
SPEC.loader.exec_module(refiner)


def dense_voxel_points(keys, points_per_voxel, voxel_size=0.04):
    """Create deterministic non-degenerate points inside requested voxels."""

    points = []
    for key in keys:
        base = np.asarray(key, dtype=np.float64) * float(voxel_size)
        for index in range(points_per_voxel):
            offset = np.asarray(
                [
                    (index % 5) * 0.002,
                    ((index // 5) % 4) * 0.002,
                    (index % 3) * 0.001,
                ],
                dtype=np.float64,
            )
            points.append(base + offset)
    return np.asarray(points, dtype=np.float64)


def solid_grid(center=(1.0, 2.0, 3.0), dims=(0.8, 0.6, 0.4)):
    center = np.asarray(center, dtype=np.float64)
    dims = np.asarray(dims, dtype=np.float64)
    axes = [
        np.linspace(
            center[axis] - 0.5 * dims[axis],
            center[axis] + 0.5 * dims[axis],
            7,
        )
        for axis in range(3)
    ]
    return np.asarray(
        list(itertools.product(*axes)), dtype=np.float32
    )


def planar_components():
    main_keys = list(itertools.product(range(4), range(4), (0,)))
    background_keys = list(
        itertools.product(range(30, 33), range(3), (0,))
    )
    main = dense_voxel_points(main_keys, 20)
    background = dense_voxel_points(background_keys, 5)
    return main, background


def test_default_config_matches_c2_frozen_geometry_contract():
    config = refiner.resolve_depth_occupancy_refiner_config()

    assert config["min_views"] == 3
    assert config["min_points"] == 192
    assert config["voxel_size"] == pytest.approx(0.02)
    assert config["neighbor_radius"] == 2
    assert config["min_component_fraction"] == pytest.approx(0.90)
    assert config["planar_ratio_threshold"] == pytest.approx(0.10)
    assert config["planar_voxel_size"] == pytest.approx(0.04)
    assert config["planar_min_points_per_voxel"] == 5
    assert config["planar_min_component_fraction"] == pytest.approx(0.10)
    assert config["planar_thin_axis_minimum"] == pytest.approx(0.034)
    assert "solid_quantile" not in config
    assert "planar_quantile" not in config


@pytest.mark.parametrize(
    "override",
    [
        {"min_views": True},
        {"min_views": 0},
        {"min_points": 0},
        {"voxel_size": 0.0},
        {"voxel_size": np.nan},
        {"neighbor_radius": 0},
        {"neighbor_radius": 1.5},
        {"min_component_fraction": 1.01},
        {"planar_ratio_threshold": -0.01},
        {"planar_voxel_size": 0.0},
        {"planar_min_points_per_voxel": 0},
        {"planar_min_component_fraction": -0.1},
        {"planar_thin_axis_minimum": 0.0},
        {"minimum_dimension": 0.0},
    ],
)
def test_invalid_config_fails_fast(override):
    with pytest.raises(ValueError):
        refiner.resolve_depth_occupancy_refiner_config(override)


def test_config_schema_is_strict_and_input_is_detached():
    with pytest.raises(ValueError, match="Unknown"):
        refiner.resolve_depth_occupancy_refiner_config(
            {"voxel_szie": 0.04}
        )
    with pytest.raises(ValueError, match="mapping"):
        refiner.resolve_depth_occupancy_refiner_config([])

    source = {"min_views": 4}
    resolved = refiner.resolve_depth_occupancy_refiner_config(source)
    resolved["min_views"] = 9
    assert source == {"min_views": 4}


def test_largest_component_removes_disconnected_background():
    main = dense_voxel_points([(0, 0, 0), (1, 0, 0), (2, 0, 0)], 10)
    background = dense_voxel_points([(20, 0, 0), (21, 0, 0)], 4)
    points = np.concatenate((background, main), axis=0)

    component = refiner.largest_voxel_connected_component(
        points, voxel_size=0.04, neighbor_radius=1
    )

    assert len(component.points) == len(main)
    assert component.voxel_count == 3
    assert component.point_fraction == pytest.approx(
        len(main) / len(points)
    )
    assert np.max(component.points[:, 0]) < 0.10
    assert component.points.flags.writeable is False


def test_largest_component_ranking_uses_points_then_voxels_then_stable_index():
    # Point-count priority.
    many_points = dense_voxel_points([(0, 0, 0)], 12)
    many_voxels = dense_voxel_points(
        [(20, 0, 0), (21, 0, 0), (22, 0, 0)], 3
    )
    selected = refiner.largest_voxel_connected_component(
        np.concatenate((many_voxels, many_points)),
        voxel_size=0.04,
        neighbor_radius=1,
    )
    assert len(selected.points) == 12
    assert selected.voxel_count == 1

    # Equal point count: voxel-count priority.
    one_voxel = dense_voxel_points([(0, 0, 0)], 12)
    three_voxels = dense_voxel_points(
        [(20, 0, 0), (21, 0, 0), (22, 0, 0)], 4
    )
    selected = refiner.largest_voxel_connected_component(
        np.concatenate((one_voxel, three_voxels)),
        voxel_size=0.04,
        neighbor_radius=1,
    )
    assert len(selected.points) == 12
    assert selected.voxel_count == 3
    assert np.min(selected.points[:, 0]) > 0.7

    # Equal points and voxels: lexicographically stable first component.
    lower = dense_voxel_points([(0, 0, 0), (1, 0, 0)], 5)
    upper = dense_voxel_points([(20, 0, 0), (21, 0, 0)], 5)
    selected = refiner.largest_voxel_connected_component(
        np.concatenate((upper, lower)),
        voxel_size=0.04,
        neighbor_radius=1,
    )
    lower_order = np.lexsort((lower[:, 2], lower[:, 1], lower[:, 0]))
    np.testing.assert_array_equal(
        selected.points, lower[lower_order].astype(np.float32)
    )
    assert selected.stable_index == 0


def test_chebyshev_radius_connects_a_two_voxel_gap_only_at_radius_two():
    points = np.concatenate(
        (
            dense_voxel_points([(0, 0, 0)], 6),
            dense_voxel_points([(2, 2, 2)], 6),
        )
    )

    radius_one = refiner.largest_voxel_connected_component(
        points, voxel_size=0.04, neighbor_radius=1
    )
    radius_two = refiner.largest_voxel_connected_component(
        points, voxel_size=0.04, neighbor_radius=2
    )

    assert len(radius_one.points) == 6
    assert len(radius_two.points) == 12


def test_largest_component_is_exactly_permutation_deterministic():
    main = dense_voxel_points(
        [(0, 0, 0), (1, 0, 0), (2, 0, 0)], 8
    )
    other = dense_voxel_points([(20, 0, 0), (21, 0, 0)], 6)
    points = np.concatenate((main, other))
    shuffled = points[np.random.default_rng(17).permutation(len(points))]

    first = refiner.largest_voxel_connected_component(
        points, voxel_size=0.04, neighbor_radius=1
    )
    second = refiner.largest_voxel_connected_component(
        shuffled, voxel_size=0.04, neighbor_radius=1
    )

    np.testing.assert_array_equal(first.points, second.points)
    assert first.point_fraction == second.point_fraction
    assert first.voxel_count == second.voxel_count
    assert first.density == second.density
    assert first.stable_index == second.stable_index


def test_planar_branch_removes_background_by_raw_aabb_density():
    main, background = planar_components()
    full_memory = np.concatenate((background, main))
    original = np.asarray(
        [0.06, 0.06, 0.00, 0.30, 0.30, 0.08], dtype=np.float32
    )

    proposal = refiner.propose_depth_occupancy_refinement(
        original,
        main,
        4,
        full_memory_points=full_memory,
    )

    assert proposal.proposed is True
    assert proposal.branch == "planar"
    assert proposal.planar is True
    assert len(proposal.points) == len(main)
    assert np.max(proposal.points[:, 0]) < np.min(background[:, 0])
    assert proposal.component_fraction == pytest.approx(
        len(main) / len(full_memory)
    )
    expected_density = len(main) / float(
        np.prod(np.ptp(main.astype(np.float64), axis=0))
    )
    expected_second = len(background) / float(
        np.prod(np.ptp(background.astype(np.float64), axis=0))
    )
    assert proposal.component_density == pytest.approx(expected_density)
    assert proposal.second_component_density == pytest.approx(
        expected_second
    )
    assert proposal.density_ratio == pytest.approx(
        expected_density / expected_second
    )
    assert proposal.density_ratio > 1.5

    raw_center = 0.5 * (
        np.min(main, axis=0) + np.max(main, axis=0)
    )
    raw_dims = np.ptp(main, axis=0)
    np.testing.assert_allclose(proposal.candidate[:3], raw_center)
    np.testing.assert_allclose(proposal.candidate[3:5], raw_dims[:2])
    # The raw Z thickness is smaller than both 3.4 cm and 65% of the
    # original 8 cm thin dimension, so the exact symmetric clamp is 5.2 cm.
    assert proposal.candidate[5] == pytest.approx(0.65 * original[5])


def test_semantic_branch_hint_can_force_planar_occupancy():
    main, background = planar_components()
    # Add a broad sparse envelope so automatic shape classification is solid,
    # while the dense planar component remains unambiguous.
    broad = solid_grid(
        center=(2.0, 2.0, 2.0), dims=(0.8, 0.8, 0.8)
    )
    full_memory = np.concatenate((main, background, broad))
    original = np.asarray(
        [0.06, 0.06, 0.00, 0.30, 0.30, 0.08], dtype=np.float32
    )

    automatic = refiner.propose_depth_occupancy_refinement(
        original,
        full_memory,
        4,
        full_memory_points=full_memory,
    )
    forced = refiner.propose_depth_occupancy_refinement(
        original,
        full_memory,
        4,
        full_memory_points=full_memory,
        branch_hint="planar",
    )

    assert automatic.branch == "solid"
    assert forced.branch == "planar"
    assert forced.proposed is True
    assert np.max(forced.points[:, 0]) < np.min(background[:, 0])


@pytest.mark.parametrize("hint", ["unknown", "", 1])
def test_invalid_semantic_branch_hint_is_identity(hint):
    points = solid_grid()
    original = np.asarray(
        [1.0, 2.0, 3.0, 1.0, 1.0, 1.0], dtype=np.float32
    )

    proposal = refiner.propose_depth_occupancy_refinement(
        original,
        points,
        3,
        branch_hint=hint,
    )

    assert proposal.proposed is False
    assert proposal.reason == "identity_invalid_branch_hint"
    np.testing.assert_array_equal(proposal.candidate, original)


def test_planar_branch_uses_track_local_voxel_origin():
    main, background = planar_components()
    full_memory = np.concatenate((main, background))
    original = np.asarray(
        [0.06, 0.06, 0.00, 0.30, 0.30, 0.08], dtype=np.float32
    )
    shift = np.asarray([10.013, -3.027, 1.019], dtype=np.float32)

    base = refiner.propose_depth_occupancy_refinement(
        original,
        main,
        3,
        full_memory_points=full_memory,
    )
    shifted_original = original.copy()
    shifted_original[:3] += shift
    shifted = refiner.propose_depth_occupancy_refinement(
        shifted_original,
        main + shift,
        3,
        full_memory_points=full_memory + shift,
    )

    np.testing.assert_allclose(
        shifted.candidate[:3], base.candidate[:3] + shift, atol=2e-6
    )
    np.testing.assert_allclose(
        shifted.candidate[3:6], base.candidate[3:6], atol=2e-6
    )
    np.testing.assert_allclose(
        shifted.points, base.points + shift, atol=2e-6
    )
    assert shifted.component_fraction == base.component_fraction
    assert shifted.density_ratio == pytest.approx(
        base.density_ratio, rel=2e-4
    )


def test_planar_clamp_uses_original_box_thinnest_dimension():
    main, background = planar_components()
    # The point envelope is thinnest on Z, while this deliberately permuted
    # original box is thinnest on X.  The clamp must use min(original_dims).
    original = np.asarray(
        [0.06, 0.06, 0.00, 0.08, 0.30, 0.12], dtype=np.float32
    )

    proposal = refiner.propose_depth_occupancy_refinement(
        original,
        main,
        3,
        full_memory_points=np.concatenate((main, background)),
    )

    assert proposal.branch == "planar"
    assert proposal.candidate[5] == pytest.approx(0.65 * 0.08)


def test_planar_proposal_is_permutation_deterministic():
    main, background = planar_components()
    full_memory = np.concatenate((main, background))
    generator = np.random.default_rng(23)
    geometry_shuffled = main[generator.permutation(len(main))]
    memory_shuffled = full_memory[
        generator.permutation(len(full_memory))
    ]
    original = np.asarray(
        [0.06, 0.06, 0.00, 0.30, 0.30, 0.08], dtype=np.float32
    )

    first = refiner.propose_depth_occupancy_refinement(
        original, main, 3, full_memory_points=full_memory
    )
    second = refiner.propose_depth_occupancy_refinement(
        original,
        geometry_shuffled,
        3,
        full_memory_points=memory_shuffled,
    )

    np.testing.assert_array_equal(first.candidate, second.candidate)
    np.testing.assert_array_equal(first.points, second.points)
    assert first.component_fraction == second.component_fraction
    assert first.component_density == second.component_density
    assert (
        first.second_component_density
        == second.second_component_density
    )
    assert first.density_ratio == second.density_ratio


def test_solid_branch_uses_full_memory_raw_min_max():
    full_memory = solid_grid()
    geometry = solid_grid(dims=(0.60, 0.40, 0.20))
    original = np.asarray(
        [1.0, 2.0, 3.0, 1.0, 1.0, 1.0], dtype=np.float32
    )

    proposal = refiner.propose_depth_occupancy_refinement(
        original,
        geometry,
        3,
        full_memory_points=full_memory,
    )

    assert proposal.proposed is True
    assert proposal.branch == "solid"
    assert proposal.planar is False
    expected_minimum = np.min(full_memory, axis=0)
    expected_maximum = np.max(full_memory, axis=0)
    expected = np.concatenate(
        (
            0.5 * (expected_minimum + expected_maximum),
            expected_maximum - expected_minimum,
        )
    )
    np.testing.assert_allclose(proposal.candidate, expected, atol=1e-7)
    assert proposal.component_fraction == 1.0
    assert len(proposal.points) == len(full_memory)
    assert proposal.component_density == pytest.approx(
        len(full_memory)
        / float(np.prod(np.ptp(full_memory.astype(np.float64), axis=0)))
    )
    assert proposal.second_component_density == 0.0
    assert proposal.density_ratio == 1.0


@pytest.mark.parametrize(
    "view_count,reason",
    [
        (2, "identity_insufficient_views"),
        (-1, "identity_invalid_view_count"),
        (True, "identity_invalid_view_count"),
        (3.0, "identity_invalid_view_count"),
    ],
)
def test_invalid_or_insufficient_view_count_is_explicit_identity(
    view_count, reason
):
    original = np.asarray(
        [1.0, 2.0, 3.0, 0.8, 0.6, 0.4], dtype=np.float32
    )
    proposal = refiner.propose_depth_occupancy_refinement(
        original, solid_grid(), view_count
    )

    assert proposal.reason == reason
    assert proposal.branch == "identity"
    assert proposal.proposed is False
    np.testing.assert_array_equal(proposal.candidate, original)


def test_invalid_and_insufficient_points_are_explicit_identity():
    original = np.asarray(
        [1.0, 2.0, 3.0, 0.8, 0.6, 0.4], dtype=np.float32
    )
    insufficient = refiner.propose_depth_occupancy_refinement(
        original, solid_grid()[:191], 3
    )
    assert insufficient.reason == "identity_insufficient_points"
    assert len(insufficient.points) == 191

    invalid_geometry = solid_grid()
    invalid_geometry[0, 0] = np.nan
    invalid = refiner.propose_depth_occupancy_refinement(
        original, invalid_geometry, 3
    )
    assert invalid.reason == "identity_invalid_geometry_points"

    invalid_full = solid_grid()
    invalid_full[0, 0] = np.inf
    invalid = refiner.propose_depth_occupancy_refinement(
        original,
        solid_grid(),
        3,
        full_memory_points=invalid_full,
    )
    assert invalid.reason == "identity_invalid_full_memory_points"


def test_planar_track_without_dense_voxels_is_explicit_identity():
    main, _ = planar_components()
    sparse = np.column_stack(
        (
            np.arange(220, dtype=np.float32) * 0.05,
            (np.arange(220, dtype=np.float32) % 2) * 0.20,
            (np.arange(220, dtype=np.float32) % 3) * 0.002,
        )
    )
    original = np.asarray(
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.08], dtype=np.float32
    )

    proposal = refiner.propose_depth_occupancy_refinement(
        original, main, 3, full_memory_points=sparse
    )

    assert proposal.reason == "identity_no_planar_component"
    assert proposal.branch == "planar"
    assert proposal.planar is True
    np.testing.assert_array_equal(proposal.candidate, original)


def test_proposal_is_deeply_immutable_and_invalid_box_raises():
    original = np.asarray(
        [1.0, 2.0, 3.0, 0.8, 0.6, 0.4], dtype=np.float32
    )
    proposal = refiner.propose_depth_occupancy_refinement(
        original, solid_grid(), 3
    )

    assert proposal.candidate.flags.writeable is False
    assert proposal.points.flags.writeable is False
    with pytest.raises(ValueError):
        proposal.candidate[0] = 99.0
    with pytest.raises(FrozenInstanceError):
        proposal.reason = "changed"

    with pytest.raises(ValueError, match="original_box"):
        refiner.propose_depth_occupancy_refinement(
            np.asarray([0.0, 0.0, 0.0, -1.0, 1.0, 1.0]),
            solid_grid(),
            3,
        )
