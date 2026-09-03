from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from boxfusion.boxer_past3_receipt import (
    BoxerFrameQuery,
    BoxerObservation,
    BoxerPast3ReceiptTracker,
    SCHEMA,
)


_SIGNS = np.asarray(
    [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)


def _corners(x=0.0, *, extent=1.0):
    return _SIGNS * (extent / 2.0) + np.asarray([x, 0.0, 2.0])


def _observation(frame, source_row, *, x=0.0, score=0.8, extent=1.0):
    return BoxerObservation(
        frame_id=frame,
        source_row=source_row,
        score=score,
        corners=_corners(x, extent=extent),
    )


def _step(tracker, frame, observations=()):
    query = tracker.query(frame, tuple(observations))
    return query, tracker.commit(query)


def test_query_is_prior_only_and_commit_is_required():
    tracker = BoxerPast3ReceiptTracker()
    first_query = tracker.query(10, [_observation(10, 1)])
    assert first_query.history_max_frame_id is None
    assert first_query.prior_track_ids == ()
    assert tracker.snapshot().active_track_ids == ()
    assert tracker.snapshot().pending_frame_id == 10

    with pytest.raises(RuntimeError, match="must be committed"):
        tracker.query(11, [_observation(11, 2)])

    first_commit = tracker.commit(first_query)
    assert first_commit.created_track_ids == (0,)
    assert first_commit.newly_frozen_receipts == ()
    second_query = tracker.query(11, [_observation(11, 2, x=0.02)])
    assert second_query.history_max_frame_id == 10
    assert second_query.history_max_frame_id < second_query.frame_id
    assert second_query.prior_track_ids == (0,)
    assert second_query.assignments[0].action == "matched"
    tracker.commit(second_query)


def test_only_exact_pending_token_can_commit_and_token_cannot_replay():
    tracker = BoxerPast3ReceiptTracker()
    query = tracker.query(0, ())
    forged = replace(query)
    assert isinstance(forged, BoxerFrameQuery)
    with pytest.raises(ValueError, match="exact pending query token"):
        tracker.commit(forged)
    tracker.commit(query)
    with pytest.raises(RuntimeError, match="no pending"):
        tracker.commit(query)


def test_same_frame_observations_cannot_self_confirm():
    tracker = BoxerPast3ReceiptTracker()
    query, commit = _step(
        tracker,
        0,
        [
            _observation(0, 10, score=0.9),
            _observation(0, 11, x=0.01, score=0.8),
            _observation(0, 12, x=-0.01, score=0.7),
        ],
    )
    # The fixed within-frame deduplicator keeps only the highest ranked OBB.
    assert query.accepted_source_rows == (10,)
    assert query.duplicate_dropped_source_rows == (11, 12)
    assert commit.created_track_ids == (0,)
    assert commit.newly_frozen_receipts == ()
    assert tracker.receipts() == ()


def test_first_stable_third_frame_freezes_geometry_and_provenance_forever():
    tracker = BoxerPast3ReceiptTracker()
    _step(tracker, 0, [_observation(0, 100, x=0.00, score=0.9)])
    _step(tracker, 1, [_observation(1, 101, x=0.02, score=0.8)])
    query = tracker.query(2, [_observation(2, 102, x=-0.01, score=0.7)])
    assert query.prospective_receipt_track_ids == (0,)
    assert tracker.receipts() == ()
    commit = tracker.commit(query)

    assert len(commit.newly_frozen_receipts) == 1
    receipt = commit.newly_frozen_receipts[0]
    assert receipt.confirmation_frame_id == 2
    assert receipt.evidence_frame_ids == (0, 1, 2)
    assert receipt.evidence_source_rows == (100, 101, 102)
    assert receipt.raw_mean_score == pytest.approx(0.8)
    frozen_bytes = receipt.corners.tobytes()
    frozen_json = receipt.to_json_dict()

    # This fourth observation still matches the track but is deliberately far
    # enough to move a terminal/live tracker.  The already frozen receipt must
    # remain byte-identical and must not acquire fourth-frame provenance.
    _, fourth = _step(
        tracker, 3, [_observation(3, 103, x=0.30, score=1.0)]
    )
    assert fourth.matched_track_ids == (0,)
    assert fourth.newly_frozen_receipts == ()
    retained = tracker.receipts()[0]
    assert retained is receipt
    assert retained.corners.tobytes() == frozen_bytes
    assert retained.to_json_dict() == frozen_json
    assert retained.evidence_frame_ids == (0, 1, 2)


def test_unstable_or_too_small_third_hit_does_not_freeze_receipt():
    unstable = BoxerPast3ReceiptTracker()
    _step(unstable, 0, [_observation(0, 0, x=0.00)])
    _step(unstable, 1, [_observation(1, 1, x=0.45)])
    query, commit = _step(unstable, 2, [_observation(2, 2, x=0.90)])
    assert query.prospective_receipt_track_ids == ()
    assert commit.newly_frozen_receipts == ()
    assert unstable.receipts() == ()

    small = BoxerPast3ReceiptTracker()
    for frame in range(3):
        _step(
            small,
            frame,
            [_observation(frame, frame, x=0.01 * frame, extent=0.29)],
        )
    assert small.receipts() == ()


def test_empty_frames_advance_ttl_and_expire_before_query_association():
    tracker = BoxerPast3ReceiptTracker()
    _step(tracker, 0, [_observation(0, 0)])
    for frame in range(1, 11):
        query, commit = _step(tracker, frame, ())
        assert query.newly_retired_track_ids == ()
        assert commit.active_track_ids == (0,)

    # Ten missed keyframes are retained.  The eleventh empty keyframe expires
    # the track before any current-frame association is considered.
    query, commit = _step(tracker, 11, ())
    assert query.prior_track_ids == ()
    assert query.newly_retired_track_ids == (0,)
    assert commit.active_track_ids == ()
    summary = tracker.summary()
    assert summary["keyframes"] == 12
    assert summary["empty_keyframes"] == 11
    assert summary["tracks_retired"] == 1


def test_association_uses_both_frozen_s0_gates():
    tracker = BoxerPast3ReceiptTracker()
    _step(tracker, 0, [_observation(0, 0, x=0.0, extent=1.0)])

    # Center distance is exactly allowed, but IoU is below 0.10; AND means a
    # new track.  A literal interpretation of the old prereg typo as OR would
    # incorrectly match track 0 here.
    query, commit = _step(
        tracker, 1, [_observation(1, 1, x=0.5, extent=0.31)]
    )
    assert query.assignments[0].action == "created"
    assert commit.created_track_ids == (1,)
    assert tracker.summary()["match_rule"] == "aabb_iou_gte_AND_center_distance_lte"


def test_per_frame_cap_is_deterministic_bounded_and_marks_audit_incomplete():
    tracker = BoxerPast3ReceiptTracker()
    rows = [
        _observation(
            0,
            source_row=index,
            x=2.0 * index,
            score=index / 100.0,
        )
        for index in range(65)
    ]
    query, commit = _step(tracker, 0, rows)
    # Top-64 score ranking drops source row 0 and never exceeds the hard bound.
    assert query.observation_capacity_dropped_source_rows == (0,)
    assert len(query.selected_source_rows) == 64
    assert len(commit.active_track_ids) == 64
    assert not query.audit_complete
    assert not commit.audit_complete
    assert not tracker.summary()["audit_complete"]


def test_inputs_and_receipts_are_copied_frozen_and_output_inert():
    corners = _corners()
    observation = BoxerObservation(0, 0, 0.7, corners)
    corners[:] = 99.0
    np.testing.assert_allclose(observation.corners, _corners())
    assert not observation.corners.flags.writeable
    with pytest.raises(ValueError):
        observation.corners.setflags(write=True)

    tracker = BoxerPast3ReceiptTracker()
    _step(tracker, 0, [observation])
    _step(tracker, 1, [_observation(1, 1)])
    _, commit = _step(tracker, 2, [_observation(2, 2)])
    receipt = commit.newly_frozen_receipts[0]
    with pytest.raises(FrozenInstanceError):
        receipt.confirmation_frame_id = 99
    assert not receipt.corners.flags.writeable
    assert receipt.observer_only
    assert not receipt.active_authorized
    assert not receipt.native_mutation_applied

    summary = tracker.summary()
    assert summary["schema"] == SCHEMA
    assert summary["training_free"]
    assert summary["past_only"]
    assert summary["gt_access"] is False
    assert summary["clip_access"] is False
    assert summary["native_prediction_access"] is False
    assert summary["depth_access"] is False
    assert summary["detector_label_access"] is False
    assert summary["detector_score_used_for_ranking_only"] is True


@pytest.mark.parametrize(
    "observation",
    [
        lambda: BoxerObservation(True, 0, 0.5, _corners()),
        lambda: BoxerObservation(0, -1, 0.5, _corners()),
        lambda: BoxerObservation(0, 0, float("nan"), _corners()),
        lambda: BoxerObservation(0, 0, 0.5, np.zeros((8, 3))),
        lambda: BoxerObservation(0, 0, 0.5, np.zeros((7, 3))),
    ],
)
def test_malformed_observation_is_rejected_without_state(observation):
    tracker = BoxerPast3ReceiptTracker()
    with pytest.raises(ValueError):
        observation()
    assert tracker.snapshot().keyframes == 0
    assert tracker.snapshot().active_track_ids == ()


def test_invalid_query_is_transactional_and_does_not_advance_time():
    tracker = BoxerPast3ReceiptTracker()
    with pytest.raises(ValueError, match="must equal frame_id"):
        tracker.query(1, [_observation(0, 0)])
    assert tracker.snapshot().last_frame_id is None
    assert tracker.snapshot().keyframes == 0
    assert tracker.snapshot().pending_frame_id is None

    _step(tracker, 1, ())
    with pytest.raises(ValueError, match="strictly increasing"):
        tracker.query(1, ())
    assert tracker.snapshot().keyframes == 1
