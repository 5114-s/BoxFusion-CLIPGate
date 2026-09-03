import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from boxfusion.cutr_residual_birth_lite import (
    CuTRResidualBirthLite,
    ResidualAssignment,
    ResidualBirthLiteConfig,
    ResidualObservation,
    build_cutr_residual_birth_lite,
    partition_scores,
)


def cube(center=(0.0, 0.0, 0.0), extent=(1.0, 1.0, 1.0)):
    center = np.asarray(center, dtype=np.float64)
    half = 0.5 * np.asarray(extent, dtype=np.float64)
    signs = np.asarray(
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
        dtype=np.float64,
    )
    return center + signs * half


def obs(frame, raw=0, score=0.30, center=(0.0, 0.0, 0.0), extent=(1, 1, 1)):
    return ResidualObservation(frame, raw, score, cube(center, extent))


def tracker(**kwargs):
    return CuTRResidualBirthLite(
        ResidualBirthLiteConfig(enabled=True, score_ceiling=0.50, **kwargs)
    )


def confirm(target, center=(0.0, 0.0, 0.0), raw=0, score=0.30):
    for frame, shift in enumerate((0.0, 0.01, 0.02)):
        target.observe(
            frame,
            [obs(frame, raw, score, (center[0] + shift, center[1], center[2]))],
        )


def test_partition_scores_preserves_rows_and_supports_both_native_thresholds():
    scores = np.asarray([0.09, 0.10, 0.3999, 0.40, 0.4999, 0.50])
    at_four = partition_scores(scores, score_ceiling=0.40)
    assert at_four.dropped_indices == (0,)
    assert at_four.residual_indices == (1, 2)
    assert at_four.native_indices == (3, 4, 5)
    assert at_four.score_floor == 0.10

    at_five = partition_scores(scores, score_ceiling=0.50)
    assert at_five.dropped_indices == (0,)
    assert at_five.residual_indices == (1, 2, 3, 4)
    assert at_five.native_indices == (5,)


@pytest.mark.parametrize("scores", [[0.1, np.nan], [[0.2]], [-0.1], [1.1]])
def test_partition_rejects_invalid_scores(scores):
    with pytest.raises(ValueError):
        partition_scores(scores, score_ceiling=0.5)


def test_config_is_strict_default_off_bounded_and_builder_resolves_section():
    assert build_cutr_residual_birth_lite({}).config.enabled is False
    target = build_cutr_residual_birth_lite(
        {
            "other": 1,
            "detection": {"score_thresh": 0.4},
            "cutr_residual_birth_lite": {
                "enabled": True,
                "observer_only": True,
                "score_ceiling": 0.4,
            },
        }
    )
    assert target.config.score_ceiling == 0.4
    with pytest.raises(ValueError, match="requires score_ceiling"):
        ResidualBirthLiteConfig(enabled=True)
    with pytest.raises(ValueError, match="observer_only must remain true"):
        ResidualBirthLiteConfig(observer_only=False)
    with pytest.raises(ValueError, match="must not exceed 1024"):
        ResidualBirthLiteConfig(max_tracks=1025)
    with pytest.raises(ValueError, match="Unknown"):
        build_cutr_residual_birth_lite(
            {"cutr_residual_birth_lite": {"typo": 1}}
        )
    with pytest.raises(ValueError, match="must equal detection.score_thresh"):
        build_cutr_residual_birth_lite(
            {
                "detection": {"score_thresh": 0.5},
                "cutr_residual_birth_lite": {
                    "enabled": True,
                    "score_ceiling": 0.4,
                },
            }
        )
    with pytest.raises(RuntimeError, match="disabled"):
        build_cutr_residual_birth_lite({}).observe(0, [])


def test_observation_copies_input_and_exposes_read_only_finite_corners():
    source = cube()
    row = ResidualObservation(0, 2, 0.2, source)
    source[:] = 99.0
    assert not np.all(row.corners == 99.0)
    assert row.corners.flags.writeable is False
    with pytest.raises(ValueError):
        row.corners[0, 0] = 1.0
    with pytest.raises(ValueError):
        row.corners.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        row.score = 0.3
    with pytest.raises(ValueError, match="finite"):
        ResidualObservation(0, 0, 0.2, np.full((8, 3), np.nan))


def test_assignment_is_frozen_strict_and_rejects_invalid_actions():
    row = ResidualAssignment(raw_index=2, track_id=7, action="created")
    assert (row.raw_index, row.track_id, row.action) == (2, 7, "created")
    with pytest.raises(FrozenInstanceError):
        row.track_id = 8
    with pytest.raises(ValueError, match="action must be matched or created"):
        ResidualAssignment(raw_index=2, track_id=7, action="dropped")
    with pytest.raises(ValueError, match="raw_index must be an integer"):
        ResidualAssignment(raw_index=True, track_id=7, action="matched")
    with pytest.raises(ValueError, match="track_id must be at least 0"):
        ResidualAssignment(raw_index=2, track_id=-1, action="matched")


def test_three_distinct_keyframes_confirm_and_evidence_is_bounded_to_five():
    target = tracker()
    results = []
    for frame in range(8):
        results.append(target.observe(frame, [obs(frame, frame, center=(0.01 * frame, 0, 0))]))
    assert results[1].newly_confirmed_track_ids == ()
    assert results[2].newly_confirmed_track_ids == (0,)
    assert all(row.newly_confirmed_track_ids == () for row in results[3:])
    closed = target.close(np.empty((0, 8, 3)), np.empty((0,)))
    assert len(closed.candidates) == 1
    assert closed.candidates[0].evidence_frame_ids == (3, 4, 5, 6, 7)


def test_matching_requires_both_iou_and_center_and_is_tie_stable():
    target = tracker()
    first = target.observe(
        0,
        [obs(0, 1, center=(0, 0, 0)), obs(0, 2, center=(3, 0, 0))],
    )
    assert first.created_track_ids == (0, 1)
    assert first.assignments == (
        ResidualAssignment(1, 0, "created"),
        ResidualAssignment(2, 1, "created"),
    )
    # The reversed caller order cannot change association order; raw_index is
    # the final tie breaker after score, IoU and center distance.
    second = target.observe(
        1,
        [obs(1, 2, center=(3.01, 0, 0)), obs(1, 1, center=(0.01, 0, 0))],
    )
    assert second.matched_track_ids == (0, 1)
    assert second.assignments == (
        ResidualAssignment(1, 0, "matched"),
        ResidualAssignment(2, 1, "matched"),
    )
    # Center is within 0.5 m but the tiny overlap is below 0.10: new track.
    third = target.observe(
        2,
        [obs(2, 3, center=(0.49, 0, 0), extent=(0.2, 0.2, 0.2))],
    )
    assert third.created_track_ids == (2,)
    assert third.assignments == (ResidualAssignment(3, 2, "created"),)


def test_same_frame_dedup_keeps_higher_score_and_never_counts_two_hits():
    target = tracker()
    result = target.observe(
        0,
        [obs(0, 4, 0.20), obs(0, 3, 0.30, center=(0.01, 0, 0))],
    )
    assert result.accepted_raw_indices == (3,)
    assert result.assignments == (ResidualAssignment(3, 0, "created"),)
    assert result.duplicate_dropped_raw_indices == (4,)
    assert target.snapshot().total_tracks == 1
    assert target.snapshot().confirmed_track_ids == ()


def test_invalid_observe_is_transactional_and_causal_order_is_strict():
    target = tracker()
    target.observe(2, [obs(2)])
    before = target.snapshot()
    with pytest.raises(ValueError, match="must equal"):
        target.observe(3, [obs(4)])
    assert target.snapshot() == before
    with pytest.raises(ValueError, match="unique"):
        target.observe(3, [obs(3, 1), obs(3, 1, center=(2, 0, 0))])
    assert target.snapshot() == before
    with pytest.raises(ValueError, match="strictly increasing"):
        target.observe(2, [])
    assert target.snapshot() == before
    target.observe(3, [obs(3)])


def test_ttl_survives_ten_misses_and_retires_before_the_eleventh_can_match():
    target = tracker()
    target.observe(0, [obs(0)])
    for frame in range(1, 11):
        result = target.observe(frame, [])
        assert result.newly_retired_track_ids == ()
    result = target.observe(11, [obs(11)])
    assert result.newly_retired_track_ids == (0,)
    assert result.created_track_ids == (1,)


def test_per_frame_capacity_drops_low_ranked_rows_without_crashing():
    target = tracker(max_observations_per_frame=2)
    result = target.observe(
        0,
        [
            obs(0, 0, 0.20, (0, 0, 0)),
            obs(0, 1, 0.35, (3, 0, 0)),
            obs(0, 2, 0.30, (6, 0, 0)),
        ],
    )
    assert result.accepted_raw_indices == (1, 2)
    assert result.assignments == (
        ResidualAssignment(1, 0, "created"),
        ResidualAssignment(2, 1, "created"),
    )
    assert result.capacity_dropped_raw_indices == (0,)
    assert result.audit_complete is False
    assert target.summary()["proposal_capacity_drops"] == 1
    target.observe(1, [obs(1, 1, 0.35, (3.01, 0, 0))])
    target.observe(2, [obs(2, 1, 0.35, (3.02, 0, 0))])
    assert target.close([], []).audit_complete is False


def test_track_capacity_drops_supplemental_birth_and_marks_incomplete():
    target = tracker(max_tracks=1)
    result = target.observe(
        0,
        [obs(0, 0, 0.30, (0, 0, 0)), obs(0, 1, 0.20, (3, 0, 0))],
    )
    assert result.created_track_ids == (0,)
    assert result.assignments == (ResidualAssignment(0, 0, "created"),)
    assert result.track_capacity_dropped_raw_indices == (1,)
    assert target.snapshot().total_tracks == 1
    assert target.snapshot().audit_complete is False


def test_assignment_covers_mixed_matches_and_creates_in_accepted_row_order():
    target = tracker()
    target.observe(0, [obs(0, 9, 0.30, (0, 0, 0))])
    result = target.observe(
        1,
        [
            obs(1, 8, 0.20, (3, 0, 0)),
            obs(1, 9, 0.30, (0.01, 0, 0)),
        ],
    )
    assert result.accepted_raw_indices == (9, 8)
    assert result.assignments == (
        ResidualAssignment(9, 0, "matched"),
        ResidualAssignment(8, 1, "created"),
    )
    assert tuple(row.raw_index for row in result.assignments) == (
        result.accepted_raw_indices
    )


def test_retired_unconfirmed_track_releases_active_capacity():
    target = tracker(max_tracks=1)
    target.observe(0, [obs(0, 0, 0.30, (0, 0, 0))])
    for frame in range(1, 11):
        target.observe(frame, [])
    result = target.observe(11, [obs(11, 1, 0.30, (3, 0, 0))])
    assert result.newly_retired_track_ids == (0,)
    assert result.created_track_ids == (1,)
    assert result.track_capacity_dropped_raw_indices == ()
    summary = target.summary()
    assert summary["retired_unconfirmed_reclaimed"] == 1
    assert summary["active_tracks"] == 1


def test_retired_confirmed_archive_is_bounded_and_overflow_is_fail_closed():
    target = tracker(max_tracks=1)
    confirm(target)
    for frame in range(3, 13):
        target.observe(frame, [])
    retired = target.observe(13, [obs(13, 1, center=(3, 0, 0))])
    assert retired.newly_retired_track_ids == (0,)
    assert retired.created_track_ids == (1,)
    target.observe(14, [obs(14, 1, center=(3.01, 0, 0))])
    target.observe(15, [obs(15, 1, center=(3.02, 0, 0))])
    for frame in range(16, 26):
        target.observe(frame, [])
    overflow = target.observe(26, [obs(26, 2, center=(6, 0, 0))])
    assert overflow.newly_retired_track_ids == (1,)
    assert overflow.created_track_ids == (2,)
    summary = target.summary()
    assert summary["retired_confirmed_archive_drops"] == 1
    assert summary["audit_complete"] is False
    assert summary["total_tracks"] <= 2 * summary["max_tracks"]


def test_close_uses_iou_medoid_and_strictly_lower_appended_score():
    target = tracker()
    for frame, x in enumerate((0.0, 0.02, 0.04)):
        target.observe(frame, [obs(frame, frame, 0.30, (x, 0, 0))])
    native = np.asarray([cube((10, 0, 0))])
    result = target.close(native, np.asarray([0.20]))
    candidate = result.candidates[0]
    assert np.allclose(candidate.corners.mean(axis=0), [0.02, 0, 0])
    assert candidate.raw_mean_score == pytest.approx(0.30)
    assert candidate.appended_score == np.nextafter(0.20, 0.0)
    assert candidate.corners.flags.writeable is False
    with pytest.raises(ValueError):
        candidate.corners.setflags(write=True)
    assert result.active_authorized is False


def test_terminal_stability_and_minimum_extent_fail_closed():
    jitter = tracker()
    for frame, x in enumerate((-0.49, 0.0, 0.49)):
        jitter.observe(frame, [obs(frame, frame, center=(x, 0, 0))])
    jitter_result = jitter.close([], [])
    assert jitter_result.candidates == ()
    assert jitter_result.unstable_track_ids == (0,)

    tiny = tracker()
    for frame in range(3):
        tiny.observe(frame, [obs(frame, frame, center=(0.01 * frame, 0, 0), extent=(0.2, 0.2, 0.2))])
    tiny_result = tiny.close([], [])
    assert tiny_result.candidates == ()
    assert tiny_result.too_small_track_ids == (0,)


def _two_confirmed_overlapping_tracks():
    target = tracker()
    for frame in range(3):
        target.observe(
            frame,
            [
                obs(frame, 0, 0.35, (0.01 * frame, 0, 0)),
                obs(frame, 1, 0.25, (0.46 + 0.01 * frame, 0, 0)),
            ],
        )
    return target


def test_native_novelty_and_self_nms_are_fixed_and_deterministic():
    novelty = _two_confirmed_overlapping_tracks()
    novelty_result = novelty.close(np.asarray([cube((0.02, 0, 0))]), [0.6])
    assert 0 in novelty_result.native_overlap_rejected_track_ids

    nms = _two_confirmed_overlapping_tracks()
    nms_result = nms.close([], [])
    assert tuple(row.track_id for row in nms_result.candidates) == (0,)
    assert nms_result.self_nms_rejected_track_ids == (1,)


def test_terminal_output_cap_is_six_with_score_then_track_tie_breaking():
    target = tracker()
    for frame in range(3):
        rows = [
            obs(frame, index, 0.20 + index * 0.01, (index * 2.0 + frame * 0.005, 0, 0))
            for index in range(7)
        ]
        target.observe(frame, rows)
    result = target.close([], [])
    assert len(result.candidates) == 6
    # Track ids were themselves allocated in descending-score order.
    assert tuple(row.track_id for row in result.candidates) == (0, 1, 2, 3, 4, 5)
    assert result.output_cap_rejected_track_ids == (6,)


def test_invalid_close_is_retryable_then_result_and_summary_are_json_safe():
    target = tracker()
    confirm(target)
    before = target.snapshot()
    with pytest.raises(ValueError, match="align"):
        target.close(np.asarray([cube((9, 0, 0))]), [])
    assert target.snapshot() == before
    with pytest.raises(ValueError, match=r"finite \[N,8,3\]"):
        target.close(np.empty((0, 7, 2)), [])
    assert target.snapshot() == before
    with pytest.raises(ValueError, match=r"\(0,1\]"):
        target.close(np.asarray([cube((9, 0, 0))]), [0.0])
    assert target.snapshot() == before
    result = target.close([], [])
    payload = result.to_json_dict()
    assert json.loads(json.dumps(payload))["candidates"][0]["track_id"] == 0
    summary = target.summary()
    json.dumps(summary)
    assert summary["close_result"] == payload
    with pytest.raises(RuntimeError, match="closed"):
        target.observe(4, [])


def test_summary_declares_training_free_shadow_contract_and_finite_timing():
    target = tracker()
    keyframe = target.observe(0, [])
    assert keyframe.accepted_raw_indices == ()
    assert keyframe.assignments == ()
    summary = target.summary()
    assert summary["training_free"] is True
    assert summary["online_learning"] is False
    assert summary["observer_only"] is True
    assert summary["active_authorized"] is False
    assert summary["gt_access"] is False
    assert summary["clip_access"] is False
    assert summary["detector_label_access"] is False
    assert summary["detector_score_access"] is True
    assert summary["detector_score_mutation"] is False
    assert summary["ttl_keyframes"] == 10
    assert summary["max_stored_observations"] == 5
    assert summary["observe_time_p95_ms"] >= 0.0
