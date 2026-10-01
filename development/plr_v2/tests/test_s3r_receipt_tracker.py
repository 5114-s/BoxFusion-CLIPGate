from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

import boxfusion.s3r_receipt_tracker as s3r
from boxfusion.s3r_receipt_tracker import (
    SCHEMA,
    S3RFrameQuery,
    S3RObservation,
    S3RReceiptTracker,
)


_SIGNS = np.asarray(
    [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
    dtype=np.float64,
)


def _corners(x=0.0, *, extent=1.0):
    extent_xyz = np.broadcast_to(np.asarray(extent, dtype=np.float64), (3,))
    return _SIGNS * (extent_xyz / 2.0) + np.asarray([x, 0.0, 2.0])


def _observation(
    frame,
    source_row,
    *,
    x=0.0,
    score=0.8,
    extent=1.0,
    sealed_npz_row=None,
    source_instance_id=None,
):
    return S3RObservation(
        frame_id=frame,
        source_row=source_row,
        sealed_npz_row=(source_row if sealed_npz_row is None else sealed_npz_row),
        source_instance_id=(
            source_row if source_instance_id is None else source_instance_id
        ),
        score=score,
        corners=_corners(x, extent=extent),
    )


def _step(tracker, frame, observations=()):
    query = tracker.query(frame, tuple(observations))
    return query, tracker.commit(query)


def test_query_is_prior_only_and_commit_requires_exact_one_use_token():
    tracker = S3RReceiptTracker()
    query = tracker.query(10, [_observation(10, 1)])
    assert query.history_max_frame_id is None
    assert query.prior_track_ids == ()
    assert tracker.snapshot().active_track_ids == ()
    assert tracker.snapshot().pending_frame_id == 10
    assert tracker.summary()["observations_received"] == 0

    with pytest.raises(RuntimeError, match="must be committed"):
        tracker.query(11, [_observation(11, 2)])
    forged = replace(query)
    assert isinstance(forged, S3RFrameQuery)
    with pytest.raises(ValueError, match="exact pending query token"):
        tracker.commit(forged)
    commit = tracker.commit(query)
    assert commit.created_track_ids == (0,)
    assert tracker.snapshot().active_track_ids == (0,)
    with pytest.raises(RuntimeError, match="no pending"):
        tracker.commit(query)


def test_no_within_frame_dedup_and_no_same_frame_confirmation():
    tracker = S3RReceiptTracker()
    rows = [_observation(0, row, score=0.9 - row * 0.1) for row in range(3)]
    query, commit = _step(tracker, 0, rows)
    assert query.selected_source_rows == (0, 1, 2)
    assert query.accepted_source_rows == (0, 1, 2)
    assert commit.created_track_ids == (0, 1, 2)
    assert commit.matched_track_ids == ()
    assert commit.newly_frozen_receipts == ()
    assert tracker.receipts() == ()
    assert tracker.summary()["within_frame_deduplication"] is False

    # Three identical current rows may consume the three distinct prior tracks,
    # but each prior track remains one-to-one within the frame.
    query, commit = _step(
        tracker,
        1,
        [_observation(1, row + 10, score=0.9 - row * 0.1) for row in range(3)],
    )
    assert [assignment.track_id for assignment in query.assignments] == [0, 1, 2]
    assert commit.matched_track_ids == (0, 1, 2)


def test_top8_row_order_is_score_then_source_and_cap_drop_is_sticky():
    tracker = S3RReceiptTracker()
    rows = [
        _observation(0, row, x=2.0 * row, score=(row % 4) / 10.0) for row in range(9)
    ]
    query, commit = _step(tracker, 0, list(reversed(rows)))
    expected = tuple(
        row.source_row
        for row in sorted(rows, key=lambda item: (-item.score, item.source_row))[:8]
    )
    dropped = tuple(
        row.source_row
        for row in sorted(rows, key=lambda item: (-item.score, item.source_row))[8:]
    )
    assert query.selected_source_rows == expected
    assert tuple(item.source_row for item in query.assignments) == expected
    assert query.observation_capacity_dropped_source_rows == dropped
    assert len(commit.active_track_ids) == 8
    assert query.audit_complete is False
    assert commit.audit_complete is False
    assert tracker.summary()["audit_complete"] is False
    _, later = _step(tracker, 1, ())
    assert later.audit_complete is False


def test_receipt_cap_failure_is_permanent_locks_first_three_and_is_sticky(
    monkeypatch,
):
    # Shrink only the defensive capacity in this white-box branch test; the
    # public tracker has no configurable science/cap API and remains fixed at
    # 1024 receipts.
    monkeypatch.setattr(s3r, "_MAX_RECEIPTS", 1)
    tracker = S3RReceiptTracker()
    for frame in range(3):
        _step(tracker, frame, [_observation(frame, frame, x=0.0)])
    assert len(tracker.receipts()) == 1

    for frame in range(3, 5):
        _step(tracker, frame, [_observation(frame, frame, x=10.0)])
    query, commit = _step(tracker, 5, [_observation(5, 5, x=10.0)])
    assert query.receipt_capacity_dropped_track_ids == (1,)
    assert commit.newly_frozen_receipts == ()
    assert commit.audit_complete is False
    assert tracker.summary()["receipt_capacity_drops"] == 1
    locked = tuple(
        row.frame_id for row in tracker._tracks[1].recent_evidence  # noqa: SLF001
    )
    assert locked == (3, 4, 5)

    query, commit = _step(tracker, 6, [_observation(6, 6, x=10.0)])
    assert query.receipt_capacity_dropped_track_ids == ()
    assert tracker.summary()["receipt_capacity_drops"] == 1
    assert (
        tuple(
            row.frame_id for row in tracker._tracks[1].recent_evidence  # noqa: SLF001
        )
        == locked
    )
    assert commit.audit_complete is False


def test_row_greedy_one_to_one_and_candidate_track_tie_breaks():
    tracker = S3RReceiptTracker()
    # Identical prior tracks make IoU and distance tie, so lower track_id wins.
    _step(tracker, 0, [_observation(0, 0), _observation(0, 1)])
    query, commit = _step(
        tracker,
        1,
        [
            _observation(1, 20, score=0.9),
            _observation(1, 10, score=0.8),
            _observation(1, 30, score=0.7),
        ],
    )
    assert [
        (row.source_row, row.track_id, row.action) for row in query.assignments
    ] == [
        (20, 0, "matched"),
        (10, 1, "matched"),
        (30, 2, "created"),
    ]
    assert commit.matched_track_ids == (0, 1)

    # Higher IoU is primary even when another track has the smaller center
    # distance.  Track 0: same size at x=.3 (IoU~.54, distance=.3).
    # Track 1: centered but extent=2 (IoU=.125, distance=0).
    tracker = S3RReceiptTracker()
    _step(
        tracker,
        0,
        [
            _observation(0, 0, x=0.3, extent=1.0),
            _observation(0, 1, x=0.0, extent=2.0),
        ],
    )
    query, _ = _step(tracker, 1, [_observation(1, 2, x=0.0, extent=1.0)])
    assert query.assignments[0].track_id == 0

    # Equal IoU with different distance uses the smaller distance.  The x-only
    # 2/3 extent centered track and unit extent shifted .2 both have IoU 2/3.
    tracker = S3RReceiptTracker()
    _step(
        tracker,
        0,
        [
            _observation(0, 0, x=0.2, extent=(1.0, 1.0, 1.0)),
            _observation(0, 1, x=0.0, extent=(2.0 / 3.0, 1.0, 1.0)),
        ],
    )
    query, _ = _step(tracker, 1, [_observation(1, 2)])
    assert query.assignments[0].track_id == 1


def test_association_is_exact_inclusive_aabb_iou_and_center_and_uses_last_row():
    # Same-center x-only extent ratio is exactly the AABB IoU, isolating the
    # strict production gate without adding an epsilon.
    exact_extent = (0.10, 1.0, 1.0)
    tracker = S3RReceiptTracker()
    _step(tracker, 0, [_observation(0, 0, extent=1.0)])
    query, _ = _step(tracker, 1, [_observation(1, 1, extent=exact_extent)])
    assert query.assignments[0].action == "matched"

    tracker = S3RReceiptTracker()
    _step(tracker, 0, [_observation(0, 0, extent=1.0)])
    below = (np.nextafter(0.10, 0.0), 1.0, 1.0)
    query, _ = _step(tracker, 1, [_observation(1, 1, extent=below)])
    assert query.assignments[0].action == "created"

    tracker = S3RReceiptTracker()
    _step(tracker, 0, [_observation(0, 0, x=0.0, extent=10.0)])
    query, _ = _step(tracker, 1, [_observation(1, 1, x=0.5, extent=10.0)])
    assert query.assignments[0].action == "matched"
    tracker = S3RReceiptTracker()
    _step(tracker, 0, [_observation(0, 0, x=0.0, extent=10.0)])
    # Use a representable margin that survives corner -> AABB-center recovery;
    # one ULP at x=.5 is rounded away when added to +/-5m corners.
    outside = 0.500000000001
    query, _ = _step(tracker, 1, [_observation(1, 1, x=outside, extent=10.0)])
    assert query.assignments[0].action == "created"

    # The association anchor is the last committed row, so a causal drift chain
    # 0 -> .4 -> .8 remains one track even though first-to-third is too far.
    tracker = S3RReceiptTracker()
    for frame, x in enumerate((0.0, 0.4, 0.8)):
        query, _ = _step(tracker, frame, [_observation(frame, frame, x=x)])
        assert query.assignments[0].track_id == 0
    assert len(tracker.receipts()) == 1


@pytest.mark.parametrize(
    "centers,extent",
    [((0.0, 0.4, 0.8), 1.0), ((0.0, 0.02, 0.04), 0.10)],
)
def test_third_distinct_frame_freezes_without_old_stability_or_extent_gates(
    centers, extent
):
    tracker = S3RReceiptTracker()
    for frame, x in enumerate(centers[:2]):
        _step(tracker, frame, [_observation(frame, frame, x=x, extent=extent)])
    query = tracker.query(2, [_observation(2, 2, x=centers[2], extent=extent)])
    assert query.prospective_receipt_track_ids == (0,)
    assert tracker.receipts() == ()
    commit = tracker.commit(query)
    assert len(commit.newly_frozen_receipts) == 1
    receipt = commit.newly_frozen_receipts[0]
    assert receipt.evidence_frame_ids == (0, 1, 2)
    assert receipt.min_medoid_aabb_extent_m == pytest.approx(extent)
    assert tracker.summary()["receipt_has_stability_gate"] is False
    assert tracker.summary()["receipt_has_extent_gate"] is False


def test_first_three_medoid_full_evidence_metrics_and_receipt_are_immutable():
    tracker = S3RReceiptTracker()
    scores = (0.9, 0.6, 0.3)
    centers = (-0.1, 0.0, 0.1)
    for frame, (x, score) in enumerate(zip(centers, scores)):
        _, commit = _step(
            tracker,
            frame,
            [
                _observation(
                    frame,
                    100 + frame,
                    x=x,
                    score=score,
                    sealed_npz_row=200 + frame,
                    source_instance_id=300 + frame,
                )
            ],
        )
    receipt = commit.newly_frozen_receipts[0]
    assert receipt.medoid_evidence_index == 1
    np.testing.assert_array_equal(receipt.corners, _corners(0.0))
    assert receipt.evidence_frame_ids == (0, 1, 2)
    assert receipt.evidence_source_rows == (100, 101, 102)
    assert receipt.evidence_sealed_npz_rows == (200, 201, 202)
    assert receipt.evidence_source_instance_ids == (300, 301, 302)
    assert receipt.evidence_scores == scores
    assert receipt.evidence_corners.shape == (3, 8, 3)
    assert receipt.pairwise_aabb_iou.shape == (3, 3)
    assert receipt.pairwise_center_distance_m.shape == (3, 3)
    assert receipt.raw_mean_score == pytest.approx(0.6)
    assert receipt.median_pairwise_aabb_iou == pytest.approx(9.0 / 11.0)
    expected_rms = np.sqrt(np.mean(np.square(centers)))
    assert receipt.center_rms_m == pytest.approx(expected_rms)

    frozen = receipt.to_json_dict()
    primary_bytes = receipt.corners.tobytes()
    evidence_bytes = receipt.evidence_corners.tobytes()
    for frame in range(3, 8):
        _, later = _step(
            tracker, frame, [_observation(frame, 100 + frame, x=0.1 * frame)]
        )
        assert later.newly_frozen_receipts == ()
    retained = tracker.receipts()[0]
    assert retained is receipt
    assert retained.to_json_dict() == frozen
    assert retained.corners.tobytes() == primary_bytes
    assert retained.evidence_corners.tobytes() == evidence_bytes
    with pytest.raises(FrozenInstanceError):
        receipt.confirmation_frame_id = 99
    for array in (
        receipt.corners,
        receipt.evidence_corners,
        receipt.pairwise_aabb_iou,
        receipt.pairwise_center_distance_m,
    ):
        assert array.flags.writeable is False
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_empty_commits_advance_ttl_and_receipt_survives_track_retirement():
    tracker = S3RReceiptTracker()
    for frame in range(3):
        _step(tracker, frame, [_observation(frame, frame)])
    assert tracker.snapshot().receipt_track_ids == (0,)
    for frame in range(3, 13):
        query, commit = _step(tracker, frame, ())
        assert query.newly_retired_track_ids == ()
        assert commit.active_track_ids == (0,)
    query, commit = _step(tracker, 13, ())
    assert query.newly_retired_track_ids == (0,)
    assert commit.active_track_ids == ()
    assert tracker.snapshot().receipt_track_ids == (0,)
    assert len(tracker.receipts()) == 1


def test_inputs_are_copied_and_summary_exposes_only_output_inert_contract():
    source = _corners()
    observation = S3RObservation(0, 0, 10, 20, 1.0, source)
    source[:] = 99.0
    np.testing.assert_array_equal(observation.corners, _corners())
    assert observation.corners.flags.writeable is False
    with pytest.raises(ValueError):
        observation.corners.setflags(write=True)

    summary = S3RReceiptTracker().summary()
    assert summary["schema"] == SCHEMA
    assert summary["observer_only"] is True
    assert summary["output_inert"] is True
    assert summary["birth"] is False
    assert summary["active_authorized"] is False
    assert summary["training_free"] is True
    assert summary["online_learning"] is False
    assert summary["optimizer_access"] is False
    assert summary["gt_access"] is False
    assert summary["clip_access"] is False
    assert summary["native_prediction_access"] is False
    assert summary["depth_access"] is False
    assert summary["detector_label_access"] is False
    assert summary["detector_score_used_for_row_order_only"] is True
    assert (
        summary["per_frame_row_order"] == "score_desc_source_row_asc_sealed_npz_row_asc"
    )
    assert summary["max_observations_per_frame"] == 8
    assert summary["max_live_tracks"] == 1024
    assert summary["max_receipts"] == 1024
    assert summary["max_stored_evidence"] == 3
    assert summary["receipt_evidence_count"] == 3
    assert summary["post_receipt_evidence_updates"] is False
    assert summary["sealed_npz_row_used_for_identity_only"] is True
    assert summary["source_instance_id_used_for_identity_only"] is True


@pytest.mark.parametrize(
    "factory",
    [
        lambda: S3RObservation(True, 0, 0, 0, 0.5, _corners()),
        lambda: S3RObservation(0, -1, 0, 0, 0.5, _corners()),
        lambda: S3RObservation(0, 0, -1, 0, 0.5, _corners()),
        lambda: S3RObservation(0, 0, 0, -1, 0.5, _corners()),
        lambda: S3RObservation(0, 0, 0, 0, float("nan"), _corners()),
        lambda: S3RObservation(0, 0, 0, 0, 1.1, _corners()),
        lambda: S3RObservation(0, 0, 0, 0, 0.5, np.zeros((8, 3))),
        lambda: S3RObservation(0, 0, 0, 0, 0.5, np.zeros((7, 3))),
    ],
)
def test_malformed_observation_is_rejected(factory):
    with pytest.raises(ValueError):
        factory()


def test_invalid_queries_are_transactional_and_validate_rows_before_top8_cap():
    tracker = S3RReceiptTracker()
    with pytest.raises(ValueError, match="must equal frame_id"):
        tracker.query(1, [_observation(0, 0)])
    assert tracker.snapshot().keyframes == 0
    assert tracker.snapshot().pending_frame_id is None

    with pytest.raises(ValueError, match=r"observations\[8\]"):
        tracker.query(0, [_observation(0, row) for row in range(8)] + [object()])
    assert tracker.snapshot().keyframes == 0

    duplicate_rows = [_observation(0, row) for row in range(8)] + [
        _observation(0, 7, score=0.0)
    ]
    with pytest.raises(ValueError, match="unique per frame"):
        tracker.query(0, duplicate_rows)
    assert tracker.snapshot().keyframes == 0

    duplicate_sealed = [
        _observation(0, 0, sealed_npz_row=7, source_instance_id=0),
        _observation(0, 1, sealed_npz_row=7, source_instance_id=1),
    ]
    with pytest.raises(ValueError, match="sealed_npz_row values must be unique"):
        tracker.query(0, duplicate_sealed)
    duplicate_instance = [
        _observation(0, 0, sealed_npz_row=7, source_instance_id=3),
        _observation(0, 1, sealed_npz_row=8, source_instance_id=3),
    ]
    with pytest.raises(ValueError, match="source_instance_id values must be unique"):
        tracker.query(0, duplicate_instance)

    _step(tracker, 1, ())
    with pytest.raises(ValueError, match="strictly increasing"):
        tracker.query(1, ())
    assert tracker.snapshot().keyframes == 1
