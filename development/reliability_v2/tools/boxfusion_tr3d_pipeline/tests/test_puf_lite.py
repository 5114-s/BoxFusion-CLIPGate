import numpy as np
import pytest

from boxfusion.moon_qim_lite import QIMCandidate, QIMQueryBatch
from boxfusion.puf_lite import (
    PUFLiteShadowObserver,
    box_geometry_likelihood,
    build_puf_lite,
    normalize_puf_likelihoods,
    resolve_puf_lite_config,
)


SIGNS = np.asarray(
    [
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1],
    ],
    dtype=np.float32,
)


def cube(center, size=0.4):
    return SIGNS * (size / 2.0) + np.asarray(center, dtype=np.float32)


def candidate(track_id, q=1.0, *, active=True, rank_iou=1.0):
    return QIMCandidate(
        track_id=track_id,
        shared_key_count=3,
        shared_key_fraction=q,
        center_distance_m=0.0,
        aabb_iou=rank_iou,
        age_keyframes=0 if active else 1,
        active_at_last_commit=active,
    )


def qim_batch(proposal_ids, rows, *, frame_id=1, history=0):
    return QIMQueryBatch(
        scene_id="scene",
        frame_id=frame_id,
        proposal_ids=tuple(proposal_ids),
        candidates=tuple(tuple(row) for row in rows),
        history_max_frame_id=history,
        query_ms=0.0,
    )


def enabled_config(**overrides):
    config = {
        "enabled": True,
        "observer_only": True,
        "top_k": 3,
        "birth_likelihood": 0.4,
        "center_sigma": 0.5,
        "center_margin_m": 0.05,
        "shared_key_power": 1.0,
        "max_tracks": 32,
        "exhaustive_fallback": True,
        "probability_tolerance": 1e-12,
        "snapshot_tolerance": 1e-5,
        "epsilon": 1e-9,
        "max_diagnostic_examples": 8,
    }
    config.update(overrides)
    return config


def run_query(observer, proposals, active_ids, active_boxes, rows, frame_id=1):
    return observer.query(
        qim_batch=qim_batch(
            range(100, 100 + len(proposals)),
            rows,
            frame_id=frame_id,
            history=None if frame_id == 0 else frame_id - 1,
        ),
        proposal_corners_world=np.asarray(proposals),
        active_track_ids=np.asarray(active_ids, dtype=np.int64),
        active_track_corners_world=np.asarray(active_boxes).reshape(-1, 8, 3),
    )


def test_config_is_default_off_strict_and_shadow_only():
    resolved = resolve_puf_lite_config()
    assert resolved["enabled"] is False
    assert resolved["observer_only"] is True
    assert resolved["birth_likelihood"] == 0.4
    assert build_puf_lite({}).enabled is False
    with pytest.raises(ValueError, match="Unknown puf_lite"):
        resolve_puf_lite_config({"typo": 1})
    with pytest.raises(ValueError, match="active association is not authorized"):
        resolve_puf_lite_config({"enabled": True, "observer_only": False})
    with pytest.raises(ValueError, match="must not exceed 3"):
        resolve_puf_lite_config({"top_k": 4})


def test_eq6_is_direct_normalization_and_birth_threshold_is_strict():
    probabilities, birth = normalize_puf_likelihoods([0.4, 0.3], 0.5)
    assert probabilities == pytest.approx((1.0 / 3.0, 0.25))
    assert birth == pytest.approx(5.0 / 12.0)
    assert birth <= 0.5  # association, although birth is the largest class

    probabilities, birth = normalize_puf_likelihoods([0.2, 0.1], 0.5)
    assert probabilities == pytest.approx((0.25, 0.125))
    assert birth == pytest.approx(0.625)
    assert birth > 0.5

    _, boundary = normalize_puf_likelihoods([0.25, 0.15], 0.4)
    assert boundary == pytest.approx(0.5)
    assert not boundary > 0.5


def test_geometry_likelihood_exact_cases_and_monotonic_q():
    box = cube([0, 0, 1])
    identical = box_geometry_likelihood(box, box, shared_key_fraction=0.0)
    assert identical.containment == pytest.approx(1.0)
    assert identical.aabb_iou == pytest.approx(1.0)
    assert identical.likelihood == pytest.approx(1.0)

    containing = cube([0, 0, 1], size=0.8)
    nested = box_geometry_likelihood(box, containing, shared_key_fraction=0.0)
    assert nested.likelihood == pytest.approx(
        np.sqrt(nested.containment * nested.aabb_iou)
    )
    far_zero = box_geometry_likelihood(
        box, cube([4, 0, 1]), shared_key_fraction=0.0
    )
    far_key = box_geometry_likelihood(
        box, cube([4, 0, 1]), shared_key_fraction=1.0
    )
    assert far_zero.likelihood == 0.0
    assert far_key.likelihood >= far_zero.likelihood


def test_no_candidates_is_explicit_birth_with_normalized_probability():
    observer = PUFLiteShadowObserver(enabled_config())
    batch = run_query(observer, [cube([5, 0, 1])], [], [], [[]])
    row = batch.rows[0]
    assert row.valid and row.actionable
    assert row.candidates == ()
    assert row.birth_probability == 1.0
    assert row.predicted_birth is True
    assert row.predicted_track_id is None
    assert row.normalization_error == 0.0


def test_identical_candidate_associates_and_probabilities_sum_to_one():
    observer = PUFLiteShadowObserver(enabled_config())
    box = cube([0, 0, 1])
    batch = run_query(observer, [box], [7], [box], [[candidate(7)]])
    row = batch.rows[0]
    assert row.predicted_birth is False
    assert row.predicted_track_id == 7
    assert row.predicted_global_row == 0
    assert row.candidates[0].likelihood == pytest.approx(1.0)
    assert row.candidates[0].probability == pytest.approx(1.0 / 1.4)
    assert row.birth_probability == pytest.approx(0.4 / 1.4)
    assert sum(item.probability for item in row.candidates) + row.birth_probability == pytest.approx(1.0)


def test_qim_miss_triggers_bounded_fallback_and_rescues_overlap():
    observer = PUFLiteShadowObserver(enabled_config())
    box = cube([0, 0, 1])
    batch = run_query(
        observer,
        [box],
        [9],
        [box],
        [[candidate(88, active=False)]],
    )
    row = batch.rows[0]
    assert row.fallback_triggered is True
    assert row.fallback_rescued is True
    assert row.qim_candidate_track_ids == ()
    assert row.predicted_track_id == 9
    assert row.candidates[0].source == "fallback"
    summary = observer.summary()
    assert summary["stale_candidates_dropped"] == 1
    assert summary["exhaustive_tracks_scored"] == 1


def test_far_history_remains_birth_after_fallback():
    observer = PUFLiteShadowObserver(enabled_config())
    batch = run_query(
        observer,
        [cube([5, 0, 1])],
        [1],
        [cube([0, 0, 1])],
        [[]],
    )
    row = batch.rows[0]
    assert row.fallback_triggered is True
    assert row.fallback_rescued is False
    assert row.predicted_birth is True
    assert row.birth_probability == 1.0


def test_qim_unique_top3_then_puf_reranks_with_stable_id_tie():
    observer = PUFLiteShadowObserver(enabled_config())
    box = cube([0, 0, 1])
    batch = run_query(
        observer,
        [box],
        [1, 2, 3, 4],
        [box, box, box, box],
        [[
            candidate(4, q=0.2),
            candidate(3),
            candidate(4, q=0.9),
            candidate(2),
            candidate(1),
        ]],
    )
    row = batch.rows[0]
    # First three unique IDs by QIM rank are 4, 3, 2; identical likelihoods
    # are then ordered by stable ID for the hard PUF choice.
    assert row.qim_candidate_track_ids == (4, 3, 2)
    assert [item.track_id for item in row.candidates] == [2, 3, 4]
    assert row.predicted_track_id == 2


def test_registry_or_liveness_inconsistency_is_invalid_not_birth():
    observer = PUFLiteShadowObserver(enabled_config())
    box = cube([0, 0, 1])
    batch = run_query(
        observer, [box], [1], [box], [[candidate(2, active=True)]]
    )
    row = batch.rows[0]
    assert row.valid is False
    assert row.predicted_birth is None
    assert row.birth_probability is None
    assert "registry_mismatch" in row.invalid_reason


def test_candidate_snapshot_mismatch_is_invalid_not_partially_scored():
    observer = PUFLiteShadowObserver(enabled_config())
    box = cube([0, 0, 1])
    mismatched = candidate(1, rank_iou=0.25)
    batch = run_query(observer, [box], [1], [box], [[mismatched]])
    assert batch.rows[0].valid is False
    assert batch.rows[0].invalid_reason == "candidate_snapshot_mismatch"
    assert batch.rows[0].birth_probability is None


def test_nonfinite_proposal_and_track_cap_fail_closed_per_row():
    observer = PUFLiteShadowObserver(enabled_config())
    bad = np.full((8, 3), np.nan, dtype=np.float32)
    batch = run_query(observer, [bad], [], [], [[]])
    assert batch.rows[0].valid is False
    assert batch.rows[0].invalid_reason == "nonfinite_proposal_geometry"

    capped = PUFLiteShadowObserver(enabled_config(max_tracks=1))
    box = cube([0, 0, 1])
    batch = run_query(capped, [box], [1, 2], [box, box], [[]])
    assert batch.rows[0].valid is False
    assert batch.rows[0].invalid_reason == "active_track_cap_exceeded"


def test_same_frame_track_conflict_is_not_actionable():
    observer = PUFLiteShadowObserver(enabled_config())
    box = cube([0, 0, 1])
    batch = run_query(
        observer,
        [box, box],
        [1],
        [box],
        [[candidate(1)], [candidate(1)]],
    )
    assert all(row.valid for row in batch.rows)
    assert all(not row.actionable for row in batch.rows)
    assert all(row.conflict for row in batch.rows)
    assert observer.summary()["same_track_conflicts"] == 2


def test_native_metrics_separate_coverage_agreement_and_birth():
    observer = PUFLiteShadowObserver(enabled_config())
    box = cube([0, 0, 1])
    batch = run_query(
        observer,
        [box, cube([5, 0, 1]), cube([8, 0, 1])],
        [7],
        [box],
        [[candidate(7)], [], []],
    )
    observer.observe_native_targets(batch, [(7,), (), None])
    summary = observer.summary()
    assert summary["native_history_matches"] == 1
    assert summary["native_births"] == 1
    assert summary["native_unresolved"] == 1
    assert summary["qim_target_coverage_at_3_rate"] == 1.0
    assert summary["post_fallback_target_coverage_rate"] == 1.0
    assert summary["top1_native_agreement_rate"] == 1.0
    assert summary["birth_precision"] == 1.0
    assert summary["birth_recall"] == 1.0
    assert summary["nll_mean"] is not None
    assert summary["brier_mean"] is not None


def test_zero_likelihood_membership_is_not_reported_as_target_support():
    observer = PUFLiteShadowObserver(enabled_config())
    proposal = cube([100, 0, 1])
    tracks = [cube([0, 0, 1]), cube([2, 0, 1]), cube([4, 0, 1])]
    batch = run_query(observer, [proposal], [1, 2, 3], tracks, [[]])
    assert batch.rows[0].candidates == ()
    observer.observe_native_targets(batch, [(1,)])
    summary = observer.summary()
    assert summary["post_fallback_target_coverage_rate"] == 0.0
    assert summary["conditional_top1_denominator"] == 0
    assert summary["retrieval_misses"] == 0
    assert summary["false_births"] == 1


def test_ambiguous_native_target_set_is_reported_separately():
    observer = PUFLiteShadowObserver(enabled_config())
    box = cube([0, 0, 1])
    batch = run_query(
        observer,
        [box],
        [1, 2],
        [box, box],
        [[candidate(2), candidate(1)]],
    )
    observer.observe_native_targets(batch, [(1, 2)])
    summary = observer.summary()
    assert summary["native_ambiguous"] == 1
    assert summary["native_history_matches"] == 0
    assert summary["ambiguous_qim_coverage_any_rate"] == 1.0
    assert summary["ambiguous_final_support_any_rate"] == 1.0
    assert summary["ambiguous_top1_in_target_set_rate"] == 1.0


def test_ambiguous_history_match_predicted_birth_counts_as_false_birth():
    observer = PUFLiteShadowObserver(enabled_config())
    proposal = cube([20, 0, 1])
    tracks = [cube([0, 0, 1]), cube([2, 0, 1])]
    batch = run_query(observer, [proposal], [1, 2], tracks, [[]])
    observer.observe_native_targets(batch, [(1, 2)])
    summary = observer.summary()
    assert summary["native_ambiguous"] == 1
    assert summary["false_births"] == 1
    assert summary["predicted_births_evaluated"] == 1
    assert summary["birth_precision"] == 0.0


def test_invalid_native_target_validation_is_transactional():
    observer = PUFLiteShadowObserver(enabled_config())
    boxes = [cube([0, 0, 1]), cube([4, 0, 1])]
    batch = run_query(observer, boxes, [], [], [[], []])
    before = observer.summary()
    with pytest.raises(ValueError, match="must contain integers"):
        observer.observe_native_targets(batch, [(), ("bad",)])
    after = observer.summary()
    assert after["native_births"] == before["native_births"] == 0
    observer.observe_native_targets(batch, [(), ()])
    assert observer.summary()["native_births"] == 2


def test_native_observation_cannot_change_future_probabilities():
    left = PUFLiteShadowObserver(enabled_config())
    right = PUFLiteShadowObserver(enabled_config())
    box = cube([0, 0, 1])
    first_left = run_query(left, [box], [4], [box], [[candidate(4)]], frame_id=1)
    first_right = run_query(right, [box], [4], [box], [[candidate(4)]], frame_id=1)
    left.observe_native_targets(first_left, [(4,)])
    right.observe_native_targets(first_right, [()])

    second_left = run_query(left, [box], [4], [box], [[candidate(4)]], frame_id=2)
    second_right = run_query(right, [box], [4], [box], [[candidate(4)]], frame_id=2)
    assert second_left.rows == second_right.rows


def test_pending_duplicate_scene_and_causality_guards():
    observer = PUFLiteShadowObserver(enabled_config())
    box = cube([0, 0, 1])
    batch = run_query(observer, [box], [], [], [[]])
    with pytest.raises(ValueError, match="previous PUF query"):
        run_query(observer, [box], [], [], [[]], frame_id=2)
    observer.observe_native_targets(batch, [()])
    with pytest.raises(ValueError, match="already observed"):
        observer.observe_native_targets(batch, [()])
    with pytest.raises(ValueError, match="bound to scene"):
        observer.query(
            qim_batch=QIMQueryBatch(
                scene_id="other",
                frame_id=2,
                proposal_ids=(101,),
                candidates=((),),
                history_max_frame_id=1,
                query_ms=0.0,
            ),
            proposal_corners_world=np.stack([box]),
            active_track_ids=np.empty((0,), dtype=np.int64),
            active_track_corners_world=np.empty((0, 8, 3), dtype=np.float32),
        )


def test_inputs_unchanged_and_summary_declares_safety_contract():
    observer = PUFLiteShadowObserver(enabled_config())
    proposals = np.stack([cube([0, 0, 1])])
    active_ids = np.asarray([5], dtype=np.int64)
    active_boxes = proposals.copy()
    proposal_before = proposals.copy()
    ids_before = active_ids.copy()
    active_before = active_boxes.copy()
    batch = observer.query(
        qim_batch=qim_batch([100], [[candidate(5)]]),
        proposal_corners_world=proposals,
        active_track_ids=active_ids,
        active_track_corners_world=active_boxes,
    )
    observer.observe_native_targets(batch, [(5,)])
    assert np.array_equal(proposals, proposal_before)
    assert np.array_equal(active_ids, ids_before)
    assert np.array_equal(active_boxes, active_before)
    summary = observer.summary()
    for key in ("observer_only", "training_free", "causal"):
        assert summary[key] is True
    for key in (
        "online_update",
        "semantic_access",
        "semantic_mutation",
        "ground_truth_access",
        "detector_score_access",
    ):
        assert summary[key] is False
    assert summary["nonfinite_probability_rows"] == 0
    assert summary["max_normalization_error"] <= 1e-12
