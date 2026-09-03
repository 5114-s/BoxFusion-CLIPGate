import json

import numpy as np
import pytest

from boxfusion import fastsam_dfu_lgf_shadow as f2


def _grid(origin=(0.0, 0.0, 0.0), shape=(4, 4, 4), spacing=0.02):
    origin = np.asarray(origin, dtype=np.float64)
    rows = []
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                rows.append(origin + spacing * np.asarray([x, y, z]))
    return np.asarray(rows, dtype=np.float64)


def _bounds(points):
    raw_q02, raw_q98 = np.quantile(points, [0.02, 0.98], axis=0)
    center = (raw_q02 + raw_q98) * 0.5
    extent = np.maximum(raw_q98 - raw_q02, 0.02)
    return center - extent * 0.5, center + extent * 0.5


def _run(points, q02=None, q98=None, include_keys=True):
    points = np.asarray(points, dtype=np.float64)
    if q02 is None or q98 is None:
        q02, q98 = _bounds(points)
    keys = np.floor(points / 0.02).astype(np.int64) if include_keys else None
    return f2.refine_fastsam_candidate(
        points_world=points,
        voxel_keys=keys,
        world_q02=np.asarray(q02, dtype=np.float64),
        world_q98=np.asarray(q98, dtype=np.float64),
    )


def test_policy_is_frozen_training_free_output_inert_and_exact():
    assert dict(f2.POLICY) == {
        "input": "f0_selected_2cm_voxel_representatives",
        "point_count": (16, 2048),
        "voxel_size_m": 0.02,
        "local_radius_m": 0.06,
        "local_min_other_neighbors": 3,
        "local_index": "scipy_ckdtree_query_pairs_then_exact_squared_predicate",
        "global_center": "coordinate_median",
        "global_distance": "euclidean_rho",
        "global_scale": "max(1.4826*MAD,0.02m)",
        "global_threshold": "median_rho+3.5*scale",
        "world_quantiles": (0.02, 0.98),
        "min_aabb_extent_m": 0.02,
        "stage_fail_open_min_points": 16,
        "training": False,
        "ground_truth": False,
        "history": False,
        "birth": False,
    }
    with pytest.raises(TypeError):
        f2.POLICY["local_radius_m"] = 1.0


def test_local_radius_keeps_dense_support_and_removes_isolated_points():
    dense = _grid(shape=(4, 4, 2))
    isolated = np.asarray(
        [[1.0 + 0.2 * index, 1.0, 1.0] for index in range(4)], dtype=np.float64
    )
    points = np.vstack((dense, isolated))
    result = _run(points)

    assert not result.hl.failed_open
    assert result.hl.reason == "accepted"
    np.testing.assert_array_equal(result.hl.retained_indices, np.arange(len(dense)))
    assert result.diagnostics.local_retained_before_fallback == len(dense)
    assert result.diagnostics.local_effective_count == len(dense)
    assert result.diagnostics.spatial_hash_bucket_count > 0
    assert result.diagnostics.spatial_hash_bucket_probes == (
        result.diagnostics.spatial_hash_bucket_count * 14
    )
    assert np.max(result.hl.world_q98) < 0.1


def test_hlg_indices_remain_in_original_candidate_index_space():
    # Put local outliers first so the effective HL subset is represented by
    # sparse/high original indices rather than a zero-based compact range.
    isolated = np.asarray(
        [[-2.0 - 0.2 * index, -2.0, -2.0] for index in range(4)],
        dtype=np.float64,
    )
    dense = _grid(shape=(4, 4, 2))
    points = np.vstack((isolated, dense))
    result = _run(points)

    expected = np.arange(4, len(points), dtype=np.int64)
    np.testing.assert_array_equal(result.hl.retained_indices, expected)
    np.testing.assert_array_equal(result.hlg.retained_indices, expected)
    assert result.hlg.source_point_count == len(points)
    assert result.hlg.retained_indices[-1] < result.hlg.source_point_count


def test_local_filter_uses_exact_radius_and_counts_other_points_only():
    # Sixteen four-point constellations.  For each anchor, two neighbours are
    # inside 6 cm and one is just outside, so no point can count itself as the
    # required third neighbour and the stage must fail open.
    rows = []
    for group in range(4):
        base = group * 1.0
        rows.extend(
            [
                [base + 0.00, 0.0, 0.0],
                [base + 0.02, 0.0, 0.0],
                [base + 0.04, 0.0, 0.0],
                [base + 0.08, 0.0, 0.0],
            ]
        )
    points = np.asarray(rows, dtype=np.float64)
    result = _run(points)
    assert result.diagnostics.local_retained_before_fallback < 16
    assert result.hl.failed_open
    assert result.hl.fallback_from == "h0"
    assert result.hl.reason == "too_few_points"
    np.testing.assert_array_equal(result.hl.retained_indices, np.arange(16))


def test_local_boundary_at_exact_six_centimetres_is_inclusive():
    # A 4x4 planar lattice has corners with exactly three other neighbours at
    # <=6 cm when spacing is 2 cm; every point therefore survives.
    points = _grid(shape=(4, 4, 1))
    result = _run(points)
    assert result.diagnostics.local_retained_before_fallback == 16
    assert not result.hl.failed_open


def test_global_coordinate_median_mad_removes_a_dense_far_cluster():
    main = _grid(shape=(4, 4, 2))
    far = _grid(origin=(2.0, 2.0, 2.0), shape=(2, 2, 2))
    points = np.vstack((main, far))
    result = _run(points)

    # Both clusters have >=3 local neighbours, while the radial robust stage
    # removes the eight-point remote component and keeps a valid main box.
    assert result.diagnostics.local_retained_before_fallback == len(points)
    assert not result.hl.failed_open
    assert result.diagnostics.rho_scale_m == pytest.approx(
        max(1.4826 * result.diagnostics.rho_mad_m, 0.02)
    )
    assert result.diagnostics.rho_threshold_m == pytest.approx(
        result.diagnostics.rho_median_m + 3.5 * result.diagnostics.rho_scale_m
    )
    assert result.diagnostics.global_retained_before_fallback == len(main)
    assert not result.hlg.failed_open
    np.testing.assert_array_equal(result.hlg.retained_indices, np.arange(len(main)))
    assert np.max(result.hlg.world_q98) < 0.1


def test_global_too_few_points_fails_open_to_effective_hl():
    inner = _grid(shape=(3, 2, 2))  # 12 locally connected points.
    outer = _grid(origin=(3.0, 3.0, 3.0), shape=(2, 2, 2))  # 8 points.
    points = np.vstack((inner, outer))
    result = _run(points)

    assert result.diagnostics.local_retained_before_fallback == 20
    assert result.diagnostics.global_retained_before_fallback < 16
    assert result.hlg.failed_open
    assert result.hlg.fallback_from == "hl"
    np.testing.assert_array_equal(result.hlg.world_q02, result.hl.world_q02)
    np.testing.assert_array_equal(result.hlg.world_q98, result.hl.world_q98)
    np.testing.assert_array_equal(result.hlg.retained_indices, result.hl.retained_indices)


def test_h0_is_exact_and_fitted_hypotheses_apply_two_centimetre_extent_floor():
    # Geometry has broad X/Y support but one constant Z plane.  H0 must remain
    # byte-exact, while fitted boxes are centered around a 2 cm Z extent.
    points = _grid(shape=(4, 4, 1))
    q02 = np.asarray([-0.2, -0.3, -0.01], dtype=np.float64)
    q98 = np.asarray([0.2, 0.3, 0.01], dtype=np.float64)
    result = _run(points, q02=q02, q98=q98)

    np.testing.assert_array_equal(result.h0.world_q02, q02)
    np.testing.assert_array_equal(result.h0.world_q98, q98)
    assert result.h0.reason == "native_f0"
    assert np.all(result.hl.world_extent >= 0.02 - 1e-12)
    assert result.hl.world_extent[2] == pytest.approx(0.02)
    assert result.hlg.world_extent[2] == pytest.approx(0.02)


def test_inputs_unchanged_outputs_deeply_readonly_and_hashes_stable():
    points = _grid(shape=(4, 4, 2))
    keys = np.floor(points / 0.02).astype(np.int64)
    q02, q98 = _bounds(points)
    originals = tuple(value.copy() for value in (points, keys, q02, q98))

    first = f2.refine_fastsam_candidate(
        points_world=points, voxel_keys=keys, world_q02=q02, world_q98=q98
    )
    second = f2.refine_fastsam_candidate(
        points_world=points, voxel_keys=keys, world_q02=q02, world_q98=q98
    )

    for value, original in zip((points, keys, q02, q98), originals):
        np.testing.assert_array_equal(value, original)
    assert first.input_sha256 == second.input_sha256
    assert first.result_sha256 == second.result_sha256
    for left, right in zip((first.h0, first.hl, first.hlg), (second.h0, second.hl, second.hlg)):
        np.testing.assert_array_equal(left.world_q02, right.world_q02)
        np.testing.assert_array_equal(left.world_q98, right.world_q98)
        np.testing.assert_array_equal(left.retained_indices, right.retained_indices)
        for array in (
            left.world_q02,
            left.world_q98,
            left.world_center,
            left.world_extent,
            left.retained_indices,
        ):
            assert not array.flags.writeable
            with pytest.raises(ValueError):
                array.flat[0] = 999


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda p, k, lo, hi: (p[:15], k[:15], lo, hi), "between 16 and 2048"),
        (lambda p, k, lo, hi: (np.vstack((p, p[0])), np.vstack((k, k[0])), lo, hi), "unique"),
        (lambda p, k, lo, hi: (p, k.astype(np.uint64), lo, hi), "signed integer"),
        (lambda p, k, lo, hi: (p, k + 1, lo, hi), "must equal"),
        (lambda p, k, lo, hi: (p, k, hi, lo), "minimum extent"),
    ],
)
def test_structural_contract_is_fail_closed(mutation, message):
    points = _grid(shape=(4, 4, 1))
    keys = np.floor(points / 0.02).astype(np.int64)
    q02, q98 = _bounds(points)
    bad_points, bad_keys, bad_q02, bad_q98 = mutation(points, keys, q02, q98)
    with pytest.raises(ValueError, match=message):
        f2.refine_fastsam_candidate(
            points_world=bad_points,
            voxel_keys=bad_keys,
            world_q02=bad_q02,
            world_q98=bad_q98,
        )


def test_derived_voxel_keys_path_matches_explicit_f0_keys():
    # Use interior representatives, as F0 does, rather than numerically
    # ambiguous synthetic points exactly on negative voxel boundaries.
    keys = np.asarray(
        [
            [x, y, z]
            for x in range(-8, -4)
            for y in range(-4, 0)
            for z in range(-2, 0)
        ],
        dtype=np.int64,
    )
    points = (keys.astype(np.float64) + 0.25) * 0.02
    explicit = _run(points, include_keys=True)
    derived = _run(points, include_keys=False)
    assert explicit.input_sha256 == derived.input_sha256
    assert explicit.result_sha256 == derived.result_sha256


def test_json_record_has_grouped_hypotheses_diagnostics_and_timing():
    result = _run(_grid(shape=(4, 4, 2)))
    record = f2.dfu_lgf_result_to_dict(result)
    encoded = json.dumps(record, sort_keys=True, allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["schema"] == "boxfusion.fastsam_dfu_lgf_shadow.f2.v1"
    assert decoded["mode"] == "shadow"
    assert list(decoded["hypotheses"]) == ["h0", "hl", "hlg"]
    assert decoded["policy"]["ground_truth"] is False
    assert decoded["policy"]["birth"] is False
    assert decoded["diagnostics"]["total_elapsed_ms"] >= 0.0
    assert len(decoded["input_sha256"]) == 64
    assert len(decoded["result_sha256"]) == 64


def test_spatial_hash_work_is_bounded_for_maximum_input():
    # A sparse 2 cm lattice across multiple hash buckets exercises the cap
    # without constructing an N x N distance matrix.
    indices = np.arange(2048, dtype=np.int64)
    voxel_keys = np.column_stack(
        (
            indices % 32,
            (indices // 32) % 16,
            indices // (32 * 16),
        )
    ).astype(np.int64)
    points = (voxel_keys.astype(np.float64) + 0.25) * 0.02
    result = _run(points)
    diagnostics = result.diagnostics
    assert diagnostics.input_point_count == 2048
    assert diagnostics.spatial_hash_bucket_probes == diagnostics.spatial_hash_bucket_count * 14
    # Unique 2 cm representatives and fixed 6 cm cells bound local pair work
    # by a small constant multiple of N, far below all-pairs N*(N-1)/2.
    assert diagnostics.local_distance_pair_tests < 2048 * 400


def test_compiled_exact_radius_query_matches_frozen_hash_reference():
    rng = np.random.default_rng(20260829)
    random_points = rng.uniform(-1.0, 1.0, size=(512, 3))
    # Include negative coordinates and exact/near 6 cm boundary cases.
    boundary = np.asarray(
        [
            [-0.12, 0.0, 0.0],
            [-0.06, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.06, 0.0, 0.0],
            [0.1200000001, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    points = np.concatenate((random_points, boundary), axis=0)
    fast, *_ = f2._local_neighbor_counts(points)
    reference, *_ = f2._local_neighbor_counts_fixed_hash_reference(points)
    assert np.array_equal(fast, reference)
