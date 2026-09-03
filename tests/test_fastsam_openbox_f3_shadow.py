import inspect
import json
import time

import numpy as np
import pytest

from boxfusion import fastsam_openbox_f3_shadow as f3


K = np.asarray(
    [[100.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
POSE = np.eye(4, dtype=np.float64)
H0_Q02 = np.asarray([-0.30, -0.30, 1.70], dtype=np.float64)
H0_Q98 = np.asarray([0.30, 0.30, 2.30], dtype=np.float64)


def _voxel_block(x=range(-4, 4), y=range(-4, 4), z=range(36, 44)):
    return np.asarray([(a, b, c) for a in x for b in y for c in z], dtype=np.int64)


def _projected_mask(q02=H0_Q02, q98=H0_Q98):
    valid, _, box, reason = f3.projected_aabb_mask_iou(
        world_q02=q02,
        world_q98=q98,
        intrinsics=K,
        camera_to_world=POSE,
        mask=np.zeros((480, 640), dtype=np.uint8),
    )
    assert valid and reason == "valid" and box is not None
    mask = np.zeros((480, 640), dtype=np.uint8)
    x1 = max(0, min(640, int(np.floor(box[0]))))
    y1 = max(0, min(480, int(np.floor(box[1]))))
    x2 = max(0, min(640, int(np.ceil(box[2]))))
    y2 = max(0, min(480, int(np.ceil(box[3]))))
    mask[y1:y2, x1:x2] = 1
    return mask


def _observation(
    frame_ordinal,
    *,
    source_id=None,
    frame_id=None,
    q02=H0_Q02,
    q98=H0_Q98,
    mask=None,
    voxel_keys=None,
    pose=POSE,
):
    if source_id is None:
        source_id = f"source-{frame_ordinal:03d}"
    if frame_id is None:
        frame_id = frame_ordinal * 25
    if mask is None:
        mask = _projected_mask(q02, q98)
    if voxel_keys is None:
        voxel_keys = _voxel_block()
    return f3.make_observation(
        source_id=source_id,
        frame_id=frame_id,
        frame_ordinal=frame_ordinal,
        confidence=0.8,
        world_q02=np.asarray(q02, dtype=np.float64),
        world_q98=np.asarray(q98, dtype=np.float64),
        voxel_keys=np.asarray(voxel_keys),
        camera_to_world=np.asarray(pose, dtype=np.float64),
        intrinsics=K,
        mask=np.asarray(mask),
    )


def _step(tracker, observation):
    return tracker.update(
        observation.frame_id,
        observation.frame_ordinal,
        [observation],
        max_logical_accessed_ordinal=observation.frame_ordinal,
    )


def test_policy_is_frozen_h0_only_training_free_and_output_inert():
    assert f3.POLICY["protocol_id"] == "F3-FASTSAM-OPENBOX-PROJECTION-SHADOW-PAPER100"
    assert f3.POLICY["input_hypothesis"] == "F1/H0_only"
    assert f3.POLICY["observer_only"] is True
    assert f3.POLICY["birth"] is False
    assert f3.POLICY["native_mutation"] is False
    assert f3.POLICY["training"] is False
    assert f3.POLICY["ground_truth"] is False
    assert f3.POLICY["depth_pixels"] is False
    assert f3.POLICY["semantics"] is False
    assert f3.POLICY["match_aabb_iou"] == 0.10
    assert f3.POLICY["match_center_distance_m"] == 0.50
    assert f3.POLICY["max_retained_observations"] == 5
    assert f3.POLICY["mask_bitorder"] == "little"
    with pytest.raises(TypeError):
        f3.POLICY["birth"] = True

    signature = inspect.signature(f3.make_observation)
    forbidden = {"gt", "ground_truth", "depth", "rgb", "label", "class_id", "semantic"}
    assert forbidden.isdisjoint(signature.parameters)


def test_observation_packs_exact_mask_caps_sorted_unique_5cm_keys_and_is_readonly():
    keys = np.asarray(
        [(index, -index, index % 7) for index in reversed(range(700))]
        + [(3, -3, 3)],
        dtype=np.int64,
    )
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[11:19, 23:41] = 1
    observation = _observation(0, mask=mask, voxel_keys=keys)

    assert observation.mask_packbits.shape == (38_400,)
    assert observation.mask_packbits.dtype == np.uint8
    np.testing.assert_array_equal(observation.unpack_mask(), mask.astype(bool))
    assert observation.voxel_keys.shape == (512, 3)
    np.testing.assert_array_equal(observation.voxel_keys[0], np.asarray([0, 0, 0]))
    np.testing.assert_array_equal(observation.voxel_keys[-1], np.asarray([699, -699, 6]))
    assert len(np.unique(observation.voxel_keys, axis=0)) == 512
    for value in (
        observation.world_q02,
        observation.world_q98,
        observation.voxel_keys,
        observation.mask_packbits,
        observation.camera_to_world,
        observation.intrinsics,
    ):
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.flat[0] = 9


def test_packed_mask_input_is_identical_and_bad_structures_fail_closed():
    mask = _projected_mask()
    packed = np.packbits(mask.reshape(-1), bitorder="little")
    raw = _observation(0, source_id="raw", mask=mask)
    direct = f3.make_observation(
        source_id="packed",
        frame_id=0,
        frame_ordinal=0,
        confidence=0.8,
        world_q02=H0_Q02,
        world_q98=H0_Q98,
        voxel_keys=_voxel_block(),
        camera_to_world=POSE,
        intrinsics=np.pad(K, ((0, 1), (0, 1))),
        mask_packbits=packed,
    )
    np.testing.assert_array_equal(raw.mask_packbits, direct.mask_packbits)
    np.testing.assert_array_equal(direct.intrinsics, K)

    with pytest.raises(ValueError, match="exactly one"):
        f3.make_observation(
            source_id="bad",
            frame_id=0,
            frame_ordinal=0,
            confidence=0.8,
            world_q02=H0_Q02,
            world_q98=H0_Q98,
            voxel_keys=_voxel_block(),
            camera_to_world=POSE,
            intrinsics=K,
        )
    with pytest.raises(ValueError, match="signed integer"):
        _observation(0, voxel_keys=_voxel_block().astype(np.uint64))
    with pytest.raises(ValueError, match="minimum"):
        _observation(0, q02=[0, 0, 2], q98=[0.01, 1, 3])


def test_projection_uses_all_corners_near_plane_and_floor_ceil_rasterization():
    mask = _projected_mask()
    valid, iou, box, reason = f3.projected_aabb_mask_iou(
        world_q02=H0_Q02,
        world_q98=H0_Q98,
        intrinsics=K,
        camera_to_world=POSE,
        mask=mask,
    )
    assert valid and reason == "valid"
    assert iou == pytest.approx(1.0)
    assert box is not None and box.dtype == np.float64 and not box.flags.writeable

    invalid, score, projected, reason = f3.projected_aabb_mask_iou(
        world_q02=[-0.2, -0.2, -0.1],
        world_q98=[0.2, 0.2, 2.0],
        intrinsics=K,
        camera_to_world=POSE,
        mask=mask,
    )
    assert not invalid and score is None and projected is None
    assert reason == "corner_at_or_behind_near_plane"


def test_direct_packed_rectangle_popcount_is_exactly_equal_to_unpack_reference():
    rng = np.random.default_rng(20260829)
    for _ in range(8):
        mask = rng.random((480, 640)) < rng.uniform(0.01, 0.75)
        packed = np.packbits(mask.reshape(-1), bitorder="little")
        mask_count = int(np.count_nonzero(mask))
        for _ in range(32):
            box = rng.uniform(
                np.asarray([-80.0, -60.0, -20.0, -20.0]),
                np.asarray([660.0, 500.0, 720.0, 540.0]),
            )
            box[2:] = np.maximum(box[2:], box[:2])
            fast = f3._box_mask_iou(box, packed, mask_count)
            reference = f3._box_mask_iou_unpack_reference(box, packed)
            assert fast == reference

    # Explicitly cover every little-endian start/end bit alignment, including
    # a one-pixel interval and the full image width.
    mask = rng.random((480, 640)) < 0.5
    packed = np.packbits(mask.reshape(-1), bitorder="little")
    mask_count = int(np.count_nonzero(mask))
    for start_bit in range(8):
        for stop_bit in range(1, 9):
            box = np.asarray(
                [16 + start_bit, 13, 32 + stop_bit, 27], dtype=np.float64
            )
            assert f3._box_mask_iou(box, packed, mask_count) == (
                f3._box_mask_iou_unpack_reference(box, packed)
            )
    full = np.asarray([0.0, 0.0, 640.0, 480.0], dtype=np.float64)
    assert f3._box_mask_iou(full, packed, mask_count) == (
        f3._box_mask_iou_unpack_reference(full, packed)
    )


def test_query_is_exact_token_prior_only_and_same_frame_sources_do_not_match():
    tracker = f3.FastSAMOpenBoxF3ShadowTracker()
    rows = [
        _observation(0, source_id="b"),
        _observation(0, source_id="a"),
    ]
    query = tracker.query(0, 0, rows, max_logical_accessed_ordinal=0)
    assert query.prior_track_ids == ()
    assert [(row.source_id, row.track_id, row.action) for row in query.assignments] == [
        ("a", 0, "created"),
        ("b", 1, "created"),
    ]
    assert tracker.summary()["active_track_count"] == 0
    with pytest.raises(RuntimeError, match="must be committed"):
        tracker.query(25, 1, [], max_logical_accessed_ordinal=1)
    forged = replace_query(query)
    with pytest.raises(ValueError, match="exact pending"):
        tracker.commit(forged)
    first = tracker.commit(query)
    assert first.active_track_ids == (0, 1)
    with pytest.raises(RuntimeError, match="no pending"):
        tracker.commit(query)

    # Every edge is tied.  The frozen global ordering is past track ID and
    # then lexical current source ID, independent of provider input order.
    second = tracker.update(
        25,
        1,
        [_observation(1, source_id="d"), _observation(1, source_id="c")],
        max_logical_accessed_ordinal=1,
    )
    assert [(row.source_id, row.track_id, row.action) for row in second.assignments] == [
        ("c", 0, "matched"),
        ("d", 1, "matched"),
    ]


def test_vectorized_association_is_receipt_exact_to_scalar_reference(monkeypatch):
    frames = []
    for ordinal in range(6):
        rows = []
        for object_id in range(9):
            center_x = object_id * 0.9 + ordinal * 0.015
            q02 = np.asarray([center_x - 0.25, -0.25, 1.75])
            q98 = np.asarray([center_x + 0.25, 0.25, 2.25])
            rows.append(
                _observation(
                    ordinal,
                    source_id=f"assoc-{ordinal}-{object_id}",
                    q02=q02,
                    q98=q98,
                )
            )
        # Provider order varies, while the frozen edge ledger remains stable.
        frames.append(tuple(reversed(rows)) if ordinal % 2 else tuple(rows))

    def run():
        tracker = f3.OpenBoxProjectionTracker()
        commits = []
        for ordinal, rows in enumerate(frames):
            commit = tracker.update(
                ordinal * 25,
                ordinal,
                rows,
                max_logical_accessed_ordinal=ordinal,
            )
            value = f3.frame_commit_to_dict(commit)
            value.pop("elapsed_ms")
            commits.append(value)
        return commits, f3.terminal_seal_to_dict(tracker.finalize())

    optimized = run()
    monkeypatch.setattr(f3, "_association_edges", f3._association_edges_reference)
    reference = run()
    assert optimized == reference


def replace_query(query):
    # A value-equal dataclass remains the wrong transaction capability.
    return f3.F3FrameQuery(**vars(query))


def test_future_access_duplicate_source_and_noncausal_order_are_rejected():
    tracker = f3.OpenBoxProjectionTracker()
    row = _observation(0)
    with pytest.raises(ValueError, match="future"):
        tracker.query(0, 0, [row], max_logical_accessed_ordinal=1)
    with pytest.raises(ValueError, match="omits current"):
        tracker.query(
            25,
            1,
            [_observation(1)],
            max_logical_accessed_ordinal=0,
        )

    _step(tracker, row)
    with pytest.raises(ValueError, match="globally unique"):
        tracker.query(25, 1, [_observation(1, source_id=row.source_id)], max_logical_accessed_ordinal=1)
    with pytest.raises(ValueError, match="strictly increasing"):
        tracker.query(25, 0, [], max_logical_accessed_ordinal=0)


def test_B_is_strict_loo_and_exact_tie_prefers_earlier_frame_then_source():
    tracker = f3.OpenBoxProjectionTracker()
    for ordinal in range(3):
        _step(tracker, _observation(ordinal))
    receipt = tracker.finalize().tracks[0]
    b = receipt.hypothesis_b
    assert b.valid and b.available
    assert b.source_id == "source-000"
    assert b.valid_fold_count == 2
    assert b.score == pytest.approx(1.0)
    for candidate in b.b_candidates:
        assert len(candidate.folds) == 2
        assert all(fold.heldout_source_id != candidate.source_id for fold in candidate.folds)
        assert all(fold.fitting_source_ids == (candidate.source_id,) for fold in candidate.folds)


def test_C_strict_loo_consensus_and_fixed_selector_choose_C_on_real_gain():
    keys = _voxel_block()
    centers = (keys.astype(np.float64) + 0.5) * 0.05
    c_q02, c_q98 = np.quantile(centers, (0.02, 0.98), axis=0)
    c_mask = _projected_mask(c_q02, c_q98)
    b_q02 = np.asarray([-0.25, -0.25, 1.75], dtype=np.float64)
    b_q98 = np.asarray([0.25, 0.25, 2.25], dtype=np.float64)

    tracker = f3.OpenBoxProjectionTracker()
    for ordinal in range(3):
        _step(
            tracker,
            _observation(
                ordinal,
                q02=b_q02,
                q98=b_q98,
                mask=c_mask,
                voxel_keys=keys,
            ),
        )
    receipt = tracker.finalize().tracks[0]
    b, c = receipt.hypothesis_b, receipt.hypothesis_c
    assert b.valid and c.valid
    assert c.valid_fold_count == 3
    assert c.score == pytest.approx(1.0)
    assert c.stability_median_iou == pytest.approx(1.0)
    assert c.consensus_voxel_count == 512
    assert c.score >= b.score + 0.03
    for fold in c.folds:
        assert fold.heldout_source_id not in fold.fitting_source_ids
        assert len(fold.fitting_source_ids) == 2
    assert receipt.selector.chosen == "C"
    np.testing.assert_array_equal(receipt.selector.world_q02, c.world_q02)
    np.testing.assert_array_equal(receipt.selector.world_q98, c.world_q98)
    assert receipt.selector.score == c.score


def test_C_loo_does_not_leak_heldout_voxels_into_its_fit():
    clean = _voxel_block()
    remote = clean + np.asarray([100, 100, 100], dtype=np.int64)
    tracker = f3.OpenBoxProjectionTracker()
    _step(tracker, _observation(0, voxel_keys=clean))
    _step(tracker, _observation(1, voxel_keys=clean))
    _step(tracker, _observation(2, voxel_keys=remote))
    c = tracker.finalize().tracks[0].hypothesis_c

    # Holding out the remote view leaves two clean fitting views and one valid
    # fold.  Holding out either clean view fits clean+remote and cannot form a
    # two-view consensus.  A leaky implementation would report three folds.
    assert c.valid_fold_count == 1
    assert not c.available and not c.valid
    assert c.reason == "fewer_than_two_valid_folds"
    valid_folds = [fold for fold in c.folds if fold.valid]
    assert [fold.heldout_source_id for fold in valid_folds] == ["source-002"]


def _assert_consensus_equal(left, right):
    assert left.valid == right.valid
    assert left.reason == right.reason
    assert left.consensus_voxel_count_before_cap == right.consensus_voxel_count_before_cap
    assert left.consensus_voxel_count == right.consensus_voxel_count
    if left.world_q02 is None:
        assert right.world_q02 is None and right.world_q98 is None
    else:
        np.testing.assert_array_equal(left.world_q02, right.world_q02)
        np.testing.assert_array_equal(left.world_q98, right.world_q98)


def test_compiled_all_loo_consensus_is_byte_exact_to_literal_reference():
    rng = np.random.default_rng(103_20260829)
    for view_count in (3, 4, 5):
        rows = []
        common = rng.integers(-12, 13, size=(80, 3), dtype=np.int64)
        # Exact +/-1 boundaries must count; distance two must not.
        boundary = np.asarray(
            [
                [-8, -8, -8],
                [-7, -8, -8],
                [-6, -8, -8],
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
            ],
            dtype=np.int64,
        )
        for view_id in range(view_count):
            jitter = rng.integers(-1, 2, size=common.shape, dtype=np.int64)
            unique = rng.integers(-30, 31, size=(35 + view_id, 3), dtype=np.int64)
            keys = np.vstack((common + jitter, boundary + view_id % 2, unique))
            rows.append(
                _observation(
                    view_id,
                    source_id=f"equiv-{view_count}-{view_id}",
                    voxel_keys=keys,
                )
            )
        observations = tuple(rows)
        optimized_folds, optimized_full = f3._consensus_boxes_all_loo(observations)
        reference_folds = tuple(
            f3._consensus_box_reference(
                observations[:heldout] + observations[heldout + 1 :]
            )
            for heldout in range(view_count)
        )
        reference_full = f3._consensus_box_reference(observations)
        for optimized, reference in zip(optimized_folds, reference_folds):
            _assert_consensus_equal(optimized, reference)
        _assert_consensus_equal(optimized_full, reference_full)


def test_extreme_int64_voxel_guards_and_anchored_tree_are_integer_exact():
    limit = (1 << 62) - 2
    with pytest.raises(ValueError, match="safe neighbourhood"):
        _observation(
            0,
            voxel_keys=np.asarray(
                [[np.iinfo(np.int64).min, 0, 0]], dtype=np.int64
            ),
        )
    with pytest.raises(ValueError, match="safe neighbourhood"):
        _observation(
            0,
            voxel_keys=np.asarray(
                [[np.iinfo(np.int64).max, 0, 0]], dtype=np.int64
            ),
        )

    near_view = np.asarray(
        [
            [limit - 20, limit - 10, limit - 5],
            [limit - 19, limit - 10, limit - 5],
            [limit - 10, limit - 3, limit - 2],
        ],
        dtype=np.int64,
    )
    observation = _observation(0, voxel_keys=near_view)
    assert observation.voxel_tree is not None
    queries = np.asarray(
        [
            near_view[0],
            near_view[0] + [1, 1, 1],
            near_view[0] + [2, 0, 0],
            [limit - 9, limit - 2, limit - 1],
        ],
        dtype=np.int64,
    )
    optimized = f3._view_neighbourhood_support_tree(
        queries,
        observation.voxel_keys,
        observation.voxel_tree,
        observation.voxel_tree_anchor,
    )
    reference = f3._view_neighbourhood_support(queries, observation.voxel_keys)
    np.testing.assert_array_equal(optimized, reference)

    # A view spanning both safe extremes is outside float64's exact relative
    # integer domain, so the implementation must select the integer fallback.
    wide_view = np.asarray(
        [[-limit, 0, 0], [limit, 0, 0]], dtype=np.int64
    )
    wide = _observation(1, voxel_keys=wide_view)
    assert wide.voxel_tree is None
    wide_queries = np.asarray(
        [[-limit + 1, 0, 0], [limit - 1, 0, 0], [0, 0, 0]],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(
        f3._view_neighbourhood_support_tree(
            wide_queries,
            wide.voxel_keys,
            wide.voxel_tree,
            wide.voxel_tree_anchor,
        ),
        f3._view_neighbourhood_support(wide_queries, wide.voxel_keys),
    )


def test_optimized_consensus_reuses_exact_directed_pair_support(monkeypatch):
    observations = tuple(_observation(index) for index in range(5))
    original = f3._view_neighbourhood_support_tree
    calls = []

    def counted(union, view, tree=None, tree_anchor=None):
        calls.append((len(union), len(view)))
        return original(union, view, tree, tree_anchor)

    monkeypatch.setattr(f3, "_view_neighbourhood_support_tree", counted)
    cache = {}
    # First evaluation seals all 5*4 directed pair-support vectors.
    f3._consensus_boxes_all_loo(observations, cache)
    assert len(calls) == 20
    started = time.perf_counter()
    optimized_folds, optimized_full = f3._consensus_boxes_all_loo(
        observations, cache
    )
    optimized_seconds = time.perf_counter() - started
    # Re-evaluation of unchanged retained evidence performs no tree query.
    assert len(calls) == 20

    started = time.perf_counter()
    reference_folds = tuple(
        f3._consensus_box_reference(observations[:index] + observations[index + 1 :])
        for index in range(5)
    )
    reference_full = f3._consensus_box_reference(observations)
    reference_seconds = time.perf_counter() - started
    for optimized, reference in zip(optimized_folds, reference_folds):
        _assert_consensus_equal(optimized, reference)
    _assert_consensus_equal(optimized_full, reference_full)
    # A generous structural performance guard: the compiled all-LOO path
    # should retain a material margin without depending on an absolute host
    # clock or the paper100 runtime gate.
    assert optimized_seconds < reference_seconds * 0.5


def test_terminal_receipt_is_exact_with_optimized_or_literal_consensus(monkeypatch):
    rows = tuple(_observation(index) for index in range(6))
    optimized_tracker = f3.OpenBoxProjectionTracker()
    for row in rows:
        _step(optimized_tracker, row)
    optimized = f3.terminal_seal_to_dict(optimized_tracker.finalize())

    def literal_all_loo(observations, support_cache=None):
        return (
            tuple(
                f3._consensus_box_reference(
                    observations[:index] + observations[index + 1 :]
                )
                for index in range(len(observations))
            ),
            f3._consensus_box_reference(observations),
        )

    monkeypatch.setattr(f3, "_consensus_boxes_all_loo", literal_all_loo)
    reference_tracker = f3.OpenBoxProjectionTracker()
    for row in rows:
        _step(reference_tracker, row)
    reference = f3.terminal_seal_to_dict(reference_tracker.finalize())
    assert optimized == reference


def test_five_view_geometry_cap_preserves_complete_lineage_and_terminal_coverage():
    tracker = f3.OpenBoxProjectionTracker()
    commits = []
    for ordinal in range(8):
        commits.append(_step(tracker, _observation(ordinal)))
    seal = tracker.finalize()
    assert len(seal.tracks) == 1
    receipt = seal.tracks[0]
    assert receipt.source_ids == tuple(f"source-{index:03d}" for index in range(8))
    assert receipt.frame_ids == tuple(index * 25 for index in range(8))
    assert receipt.frame_ordinals == tuple(range(8))
    assert receipt.observation_count == receipt.total_observation_count == 8
    assert receipt.retained_source_ids == tuple(
        f"source-{index:03d}" for index in range(3, 8)
    )
    assert receipt.retained_frame_ordinals == tuple(range(3, 8))
    assert receipt.retained_view_count == 5
    assert receipt.confirmed
    assert {row.source_id for commit in commits for row in commit.assignments} == set(
        receipt.source_ids
    )


def test_empty_scheduled_frames_advance_ttl_and_retired_track_survives_finalize():
    tracker = f3.OpenBoxProjectionTracker()
    _step(tracker, _observation(0))
    for ordinal in range(1, 11):
        commit = tracker.update(
            ordinal * 25,
            ordinal,
            [],
            max_logical_accessed_ordinal=ordinal,
        )
        assert commit.retired_track_ids == ()
        assert commit.active_track_ids == (0,)
    retired = tracker.update(275, 11, [], max_logical_accessed_ordinal=11)
    assert retired.retired_track_ids == (0,)
    assert retired.active_track_ids == ()
    seal = tracker.finalize()
    assert [track.track_id for track in seal.tracks] == [0]
    assert seal.tracks[0].seal_reason == "ttl"
    assert seal.tracks[0].source_ids == ("source-000",)
    assert seal.max_logical_accessed_ordinal == 11


def test_invalid_projection_abstains_without_nan_or_output_mutation():
    behind = np.eye(4, dtype=np.float64)
    behind[2, 3] = 3.0  # world box is behind/through this camera's near plane
    tracker = f3.OpenBoxProjectionTracker()
    for ordinal in range(3):
        _step(tracker, _observation(ordinal, pose=behind))
    receipt = tracker.finalize().tracks[0]
    assert not receipt.hypothesis_b.available
    assert receipt.hypothesis_b.reason == "fewer_than_two_valid_folds"
    assert not receipt.hypothesis_c.available
    assert receipt.selector.chosen is None
    assert receipt.selector.world_q02 is None
    assert receipt.selector.world_q98 is None


def test_terminal_json_schema_has_all_tracks_B_C_selector_and_causality_receipts():
    tracker = f3.OpenBoxProjectionTracker()
    for ordinal in range(3):
        _step(tracker, _observation(ordinal))
    terminal = tracker.finalize()
    # Idempotent sealing cannot recompute or alter a receipt.
    assert tracker.finalize() is terminal
    record = f3.terminal_seal_to_dict(terminal)
    encoded = json.dumps(record, sort_keys=True, allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["schema"] == "boxfusion.fastsam_openbox_f3_shadow.v1"
    assert decoded["protocol_id"] == "F3-FASTSAM-OPENBOX-PROJECTION-SHADOW-PAPER100"
    assert decoded["observer_only"] is True
    assert decoded["active_authorized"] is False
    assert decoded["native_mutation_applied"] is False
    assert decoded["birth_applied"] is False
    assert decoded["track_count"] == 1
    assert list(decoded["tracks"][0]["hypotheses"]) == ["B", "C"]
    assert decoded["tracks"][0]["source_ids"] == [
        "source-000",
        "source-001",
        "source-002",
    ]
    assert decoded["tracks"][0]["retained_view_count"] == 3
    assert decoded["tracks"][0]["max_logical_accessed_ordinal"] == 2
    assert decoded["max_logical_accessed_ordinal"] == 2
    assert decoded["tracks"][0]["selector"]["chosen"] in {"B", "C"}
    selected = decoded["tracks"][0]["selector"]
    chosen = decoded["tracks"][0]["hypotheses"][selected["chosen"]]
    assert selected["q02"] == chosen["q02"]
    assert selected["q98"] == chosen["q98"]
    assert selected["center"] == chosen["center"]
    assert selected["extent"] == chosen["extent"]


def test_invalid_hypothesis_serialization_hides_top_level_geometry():
    behind = np.eye(4, dtype=np.float64)
    behind[2, 3] = 3.0
    tracker = f3.OpenBoxProjectionTracker()
    for ordinal in range(3):
        _step(tracker, _observation(ordinal, pose=behind))
    record = f3.track_receipt_to_dict(tracker.finalize().tracks[0])
    for name in ("B", "C"):
        hypothesis = record["hypotheses"][name]
        assert hypothesis["valid"] is False
        assert hypothesis["q02"] is None
        assert hypothesis["q98"] is None
        assert hypothesis["center"] is None
        assert hypothesis["extent"] is None
    assert record["selector"]["q02"] is None
    assert record["selector"]["q98"] is None
    assert record["selector"]["center"] is None
    assert record["selector"]["extent"] is None


def test_finalize_requires_commit_and_blocks_future_updates():
    tracker = f3.OpenBoxProjectionTracker()
    query = tracker.query(0, 0, [_observation(0)], max_logical_accessed_ordinal=0)
    with pytest.raises(RuntimeError, match="must be committed"):
        tracker.finalize()
    tracker.commit(query)
    tracker.finalize()
    with pytest.raises(RuntimeError, match="already been finalized"):
        tracker.query(25, 1, [], max_logical_accessed_ordinal=1)
