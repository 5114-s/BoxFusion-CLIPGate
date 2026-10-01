import numpy as np
import pytest
from dataclasses import replace

from boxfusion.puf_arbitration_lite import (
    PUFArbitrationLiteObserver,
    build_puf_arbitration_lite,
    resolve_puf_arbitration_lite_config,
)
from boxfusion.puf_lite import (
    PUFCandidatePosterior,
    PUFProposalDecision,
    PUFQueryBatch,
)


def config(**overrides):
    values = {
        "enabled": True,
        "observer_only": True,
        "track_min_probability": 0.70,
        "track_min_margin": 0.20,
        "birth_min_probability": 0.70,
        "birth_min_margin": 0.20,
        "conflict_min_owner_gap": 0.10,
        "max_proposals": 32,
        "probability_tolerance": 1e-12,
        "max_diagnostic_examples": 8,
    }
    values.update(overrides)
    return values


def candidate(track_id, probability, *, source="qim", likelihood=1.0):
    return PUFCandidatePosterior(
        track_id=track_id,
        global_row=track_id,
        source=source,
        qim_rank=0 if source == "qim" else None,
        containment=1.0,
        aabb_iou=1.0,
        overlap_support=1.0,
        center_support=1.0,
        shared_key_fraction=1.0 if source == "qim" else 0.0,
        likelihood=likelihood,
        probability=probability,
    )


def track_row(
    proposal_id,
    track_id,
    probability,
    *,
    source="qim",
    conflict=False,
    likelihood=1.0,
):
    birth = 1.0 - probability
    return PUFProposalDecision(
        proposal_id=proposal_id,
        valid=True,
        actionable=not conflict,
        invalid_reason="same_track_conflict" if conflict else None,
        conflict=conflict,
        qim_candidate_track_ids=(track_id,) if source == "qim" else (),
        candidates=(
            candidate(
                track_id,
                probability,
                source=source,
                likelihood=likelihood,
            ),
        ),
        birth_probability=birth,
        predicted_birth=False,
        predicted_track_id=track_id,
        predicted_global_row=track_id,
        fallback_triggered=source != "qim",
        fallback_rescued=source != "qim",
        exhaustive_ms=0.0,
        normalization_error=0.0,
    )


def birth_row(proposal_id, probability=0.8):
    track_probability = 1.0 - probability
    return PUFProposalDecision(
        proposal_id=proposal_id,
        valid=True,
        actionable=True,
        invalid_reason=None,
        conflict=False,
        qim_candidate_track_ids=(9,),
        candidates=(candidate(9, track_probability),),
        birth_probability=probability,
        predicted_birth=True,
        predicted_track_id=None,
        predicted_global_row=None,
        fallback_triggered=False,
        fallback_rescued=False,
        exhaustive_ms=0.0,
        normalization_error=0.0,
    )


def invalid_row(proposal_id):
    return PUFProposalDecision(
        proposal_id=proposal_id,
        valid=False,
        actionable=False,
        invalid_reason="invalid_geometry",
        conflict=False,
        qim_candidate_track_ids=(),
        candidates=(),
        birth_probability=None,
        predicted_birth=None,
        predicted_track_id=None,
        predicted_global_row=None,
        fallback_triggered=False,
        fallback_rescued=False,
        exhaustive_ms=0.0,
        normalization_error=None,
    )


def puf_batch(rows, frame_id=1, scene_id="scene"):
    return PUFQueryBatch(
        scene_id=scene_id,
        frame_id=frame_id,
        history_max_frame_id=frame_id - 1,
        proposal_ids=tuple(row.proposal_id for row in rows),
        rows=tuple(rows),
        query_ms=0.0,
    )


def test_config_default_off_strict_shadow_and_frozen_thresholds():
    resolved = resolve_puf_arbitration_lite_config()
    assert resolved["enabled"] is False
    assert resolved["observer_only"] is True
    assert build_puf_arbitration_lite({}).enabled is False
    with pytest.raises(ValueError, match="Unknown puf_arbitration_lite"):
        resolve_puf_arbitration_lite_config({"typo": 1})
    with pytest.raises(ValueError, match="active association is not authorized"):
        resolve_puf_arbitration_lite_config(
            {"enabled": True, "observer_only": False}
        )
    with pytest.raises(ValueError, match="frozen thresholds"):
        resolve_puf_arbitration_lite_config(
            {"enabled": True, "track_min_probability": 0.69}
        )


def test_unique_high_confidence_qim_track_emits_one_directive():
    observer = PUFArbitrationLiteObserver(config())
    batch = observer.query(puf_batch=puf_batch([track_row(100, 1, 0.72)]))
    row = batch.rows[0]
    assert row.action == "track"
    assert row.selected_track_id == 1
    assert row.selected_source == "qim"
    assert row.confidence_eligible is True
    assert row.margin == pytest.approx(0.44)


def test_equal_conflict_abstains_instead_of_using_tie_break_as_confidence():
    observer = PUFArbitrationLiteObserver(config())
    rows = [
        track_row(100, 1, 0.72, conflict=True),
        track_row(101, 1, 0.72, conflict=True),
    ]
    batch = observer.query(puf_batch=puf_batch(rows))
    assert all(row.action == "native_fallback" for row in batch.rows)
    assert all(not row.conflict_winner for row in batch.rows)
    summary = observer.summary()
    assert summary["source_conflict_groups"] == 1
    assert summary["conflict_tie_abstentions"] == 1
    assert summary["duplicate_selected_tracks"] == 0


def test_clear_conflict_owner_wins_and_loser_is_never_rerouted():
    observer = PUFArbitrationLiteObserver(config())
    rows = [
        track_row(101, 4, 0.55, conflict=True),
        track_row(100, 4, 0.72, conflict=True),
    ]
    batch = observer.query(puf_batch=puf_batch(rows))
    by_id = {row.proposal_id: row for row in batch.rows}
    assert by_id[100].action == "track"
    assert by_id[100].conflict_winner is True
    assert by_id[101].action == "native_fallback"
    assert by_id[101].selected_track_id is None
    assert by_id[101].raw_predicted_birth is False
    assert by_id[101].reason == "conflict_loser_native_fallback"
    assert observer.summary()["duplicate_selected_tracks"] == 0


def test_row_permutation_does_not_change_clear_owner():
    rows = [
        track_row(101, 4, 0.55, conflict=True),
        track_row(100, 4, 0.72, conflict=True),
    ]
    left = PUFArbitrationLiteObserver(config()).query(
        puf_batch=puf_batch(rows)
    )
    right = PUFArbitrationLiteObserver(config()).query(
        puf_batch=puf_batch(rows[::-1])
    )
    left_actions = {
        row.proposal_id: (row.action, row.conflict_winner) for row in left.rows
    }
    right_actions = {
        row.proposal_id: (row.action, row.conflict_winner) for row in right.rows
    }
    assert left_actions == right_actions


def test_fallback_only_track_is_not_active_eligible():
    observer = PUFArbitrationLiteObserver(config())
    batch = observer.query(
        puf_batch=puf_batch([track_row(100, 1, 0.72, source="fallback")])
    )
    row = batch.rows[0]
    assert row.action == "native_fallback"
    assert row.confidence_eligible is False
    assert row.selected_track_id is None


def test_high_confidence_birth_is_selected_but_low_confidence_abstains():
    observer = PUFArbitrationLiteObserver(config())
    batch = observer.query(
        puf_batch=puf_batch([birth_row(100, 0.80), birth_row(101, 0.65)])
    )
    assert batch.rows[0].action == "birth"
    assert batch.rows[1].action == "native_fallback"
    assert all(row.selected_track_id is None for row in batch.rows)


def test_invalid_source_and_proposal_cap_fall_back_to_native():
    observer = PUFArbitrationLiteObserver(config())
    batch = observer.query(puf_batch=puf_batch([invalid_row(100)]))
    assert batch.rows[0].action == "native_fallback"
    assert batch.rows[0].source_valid is False

    capped = PUFArbitrationLiteObserver(config(max_proposals=1))
    batch = capped.query(
        puf_batch=puf_batch(
            [
                track_row(100, 1, 0.72, conflict=True),
                track_row(101, 1, 0.72, conflict=True),
            ]
        )
    )
    assert all(row.action == "native_fallback" for row in batch.rows)
    assert all(row.reason == "proposal_cap_native_fallback" for row in batch.rows)
    assert capped.summary()["proposal_cap_batches"] == 1


def test_selected_track_and_global_row_snapshot_must_agree():
    observer = PUFArbitrationLiteObserver(config())
    inconsistent = replace(
        track_row(100, 1, 0.72), predicted_global_row=99
    )
    with pytest.raises(ValueError, match="global-row snapshot"):
        observer.query(puf_batch=puf_batch([inconsistent]))


def test_native_group_metrics_distinguish_multi_positive_losers():
    observer = PUFArbitrationLiteObserver(config())
    rows = [
        track_row(100, 4, 0.72, conflict=True),
        track_row(101, 4, 0.55, conflict=True),
        birth_row(102, 0.80),
    ]
    batch = observer.query(puf_batch=puf_batch(rows))
    observer.observe_native_targets(batch, [(4,), (4,), ()])
    summary = observer.summary()
    assert summary["conflict_native_multi_positive_groups"] == 1
    assert summary["conflict_owner_group_precision"] == 1.0
    assert summary["conflict_loser_native_positive_rate"] == 1.0
    assert summary["selective_precision"] == 1.0
    assert summary["selected_track_precision"] == 1.0
    assert summary["selected_birth_precision"] == 1.0


def test_wrong_conflict_owner_is_visible_not_hidden_by_set_coverage():
    observer = PUFArbitrationLiteObserver(config())
    rows = [
        track_row(100, 4, 0.72, conflict=True),
        track_row(101, 4, 0.55, conflict=True),
    ]
    batch = observer.query(puf_batch=puf_batch(rows))
    observer.observe_native_targets(batch, [(), (4,)])
    summary = observer.summary()
    assert summary["conflict_native_unique_positive_groups"] == 1
    assert summary["conflict_owner_group_precision"] == 0.0
    assert summary["selected_wrong"] == 1
    assert summary["false_track_overrides"] == 1


def test_unresolved_is_excluded_from_precision_denominator():
    observer = PUFArbitrationLiteObserver(config())
    batch = observer.query(puf_batch=puf_batch([track_row(100, 1, 0.72)]))
    observer.observe_native_targets(batch, [None])
    summary = observer.summary()
    assert summary["native_unresolved"] == 1
    assert summary["selected_evaluable"] == 0
    assert summary["selective_precision"] is None


def test_observed_native_labels_cannot_change_future_arbitration():
    left = PUFArbitrationLiteObserver(config())
    right = PUFArbitrationLiteObserver(config())
    first_rows = [track_row(100, 1, 0.72)]
    left_first = left.query(puf_batch=puf_batch(first_rows, frame_id=1))
    right_first = right.query(puf_batch=puf_batch(first_rows, frame_id=1))
    left.observe_native_targets(left_first, [(1,)])
    right.observe_native_targets(right_first, [()])
    second_rows = [track_row(101, 1, 0.72)]
    left_second = left.query(puf_batch=puf_batch(second_rows, frame_id=2))
    right_second = right.query(puf_batch=puf_batch(second_rows, frame_id=2))
    assert left_second.rows == right_second.rows


def test_pending_duplicate_scene_and_transactional_target_validation():
    observer = PUFArbitrationLiteObserver(config())
    batch = observer.query(
        puf_batch=puf_batch([track_row(100, 1, 0.72), birth_row(101)])
    )
    with pytest.raises(ValueError, match="previous arbitration"):
        observer.query(
            puf_batch=puf_batch([track_row(102, 1, 0.72)], frame_id=2)
        )
    before = observer.summary()
    with pytest.raises(ValueError, match="must contain integers"):
        observer.observe_native_targets(batch, [(1,), ("bad",)])
    assert observer.summary()["native_history_unique"] == before[
        "native_history_unique"
    ]
    observer.observe_native_targets(batch, [(1,), ()])
    with pytest.raises(ValueError, match="already observed"):
        observer.observe_native_targets(batch, [(1,), ()])
    with pytest.raises(ValueError, match="bound to scene"):
        observer.query(
            puf_batch=puf_batch(
                [track_row(102, 1, 0.72)], frame_id=2, scene_id="other"
            )
        )


def test_summary_declares_no_mutation_training_or_reassignment():
    observer = PUFArbitrationLiteObserver(config())
    batch = observer.query(puf_batch=puf_batch([birth_row(100)]))
    observer.observe_native_targets(batch, [()])
    summary = observer.summary()
    for key in ("observer_only", "training_free", "causal"):
        assert summary[key] is True
    for key in (
        "active_authorized",
        "online_update",
        "semantic_access",
        "semantic_mutation",
        "ground_truth_access",
        "detector_score_access",
        "reassigns_losers",
        "suppresses_proposals",
    ):
        assert summary[key] is False
    assert summary["duplicate_selected_tracks"] == 0
