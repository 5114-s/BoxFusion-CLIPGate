import ast
import copy
import inspect
import json
from dataclasses import FrozenInstanceError

import pytest

import boxfusion.side_birth_probation_lite as module
from boxfusion.side_birth_probation_lite import (
    SCHEMA,
    SideBirthProbationLiteLedger,
    SideBirthSeedEvent,
    build_side_birth_probation_lite,
    resolve_side_birth_probation_lite_config,
)


def config(**overrides):
    result = {"enabled": True, "observer_only": True}
    result.update(overrides)
    return result


def ledger(**overrides):
    return SideBirthProbationLiteLedger(config(**overrides))


def seed(
    proposal_id=100,
    stable_id=7,
    *,
    kind="birth",
    target_ids=(),
    probability=0.91,
    margin=0.34,
):
    return SideBirthSeedEvent(
        proposal_id=proposal_id,
        committed_stable_id=stable_id,
        top_probability=probability,
        margin=margin,
        native_target_kind=kind,
        native_target_ids=target_ids,
    )


def observe(target, frame, step, events=(), active=(), scene="scene"):
    return target.observe_true_cutr_keyframe(
        scene_id=scene,
        frame_id=frame,
        keyframe_step=step,
        birth_events=events,
        observed_stable_ids=active,
    )


def test_config_is_default_off_strict_bounded_and_freezes_decision_values():
    assert resolve_side_birth_probation_lite_config() == {
        "enabled": False,
        "observer_only": True,
        "min_distinct_keyframes": 3,
        "max_missed_keyframes": 10,
        "max_pending_tracks": 256,
        "max_birth_events": 8192,
    }
    assert build_side_birth_probation_lite({}).enabled is False
    assert build_side_birth_probation_lite(
        {"dataset": "scannet", "data": {}}
    ).enabled is False
    assert build_side_birth_probation_lite(
        {"side_birth_probation_lite": config()}
    ).enabled is True
    with pytest.raises(ValueError, match="Unknown side_birth_probation_lite"):
        resolve_side_birth_probation_lite_config({"typo": 1})
    with pytest.raises(ValueError, match="observer_only must remain true"):
        resolve_side_birth_probation_lite_config({"observer_only": False})
    with pytest.raises(ValueError, match="three-keyframe threshold"):
        resolve_side_birth_probation_lite_config(
            config(min_distinct_keyframes=2)
        )
    with pytest.raises(ValueError, match="frozen TTL"):
        resolve_side_birth_probation_lite_config(
            config(max_missed_keyframes=9)
        )
    with pytest.raises(ValueError, match="must not exceed 256"):
        resolve_side_birth_probation_lite_config({"max_pending_tracks": 257})
    with pytest.raises(ValueError, match="must not exceed 8192"):
        resolve_side_birth_probation_lite_config({"max_birth_events": 8193})
    with pytest.raises(ValueError, match="must be a boolean"):
        resolve_side_birth_probation_lite_config({"enabled": 1})


def test_disabled_ledger_has_no_runtime_or_active_authority():
    target = SideBirthProbationLiteLedger()
    with pytest.raises(RuntimeError, match="disabled"):
        observe(target, 0, 0)
    assert target.summary()["active_authorized"] is False


def test_known_false_like_single_view_is_retired_without_confirmation():
    target = ledger()
    row = seed(kind="unique_history", target_ids=(41,))
    first = observe(target, 10, 0, [row], [7])
    assert first.pending_track_ids == (7,)
    receipt = target.close_scene(
        scene_id="scene", terminal_frame_id=11, active_stable_ids=[]
    )
    event = receipt.events[0]
    assert event.observed_frame_ids == (10,)
    assert event.observed_keyframe_steps == (0,)
    assert event.confirmed is False
    assert event.retired is True
    assert event.status == "retired_probationary"
    assert event.reason == "terminal_inactive"
    assert receipt.metrics.event_native_unique_history == 1
    assert receipt.metrics.event_confirmed_precision is None


def test_third_distinct_true_keyframe_confirms_once_and_is_sticky():
    target = ledger()
    observe(target, 5, 0, [seed()], [7])
    second = observe(target, 8, 1, [], [7])
    assert second.pending_track_ids == (7,)
    third = observe(target, 13, 2, [], [7])
    assert third.newly_confirmed_track_ids == (7,)
    assert third.confirmed_track_ids == (7,)
    fourth = observe(target, 21, 3, [], [7])
    assert fourth.newly_confirmed_track_ids == ()
    assert fourth.confirmed_track_ids == (7,)

    receipt = target.close_scene(
        scene_id="scene", terminal_frame_id=22, active_stable_ids=[7]
    )
    event = receipt.events[0]
    assert event.observed_frame_ids == (5, 8, 13, 21)
    assert event.confirmation_frame_id == 13
    assert event.confirmation_keyframe_step == 2
    assert event.latency_frames == 8
    assert event.latency_keyframes == 2
    assert event.status == "confirmed"
    assert receipt.metrics.event_confirmed_precision == 1.0
    assert receipt.metrics.event_confirmed_retention == 1.0
    assert receipt.metrics.unique_track_confirmed_precision == 1.0
    assert receipt.metrics.unique_track_confirmed_retention == 1.0


def test_same_frame_duplicates_collapse_and_multiple_events_share_track_result():
    target = ledger()
    rows = [seed(100, 7), seed(101, 7)]
    first = observe(target, 0, 0, rows, [7, 7, 7])
    assert first.observed_track_ids == (7,)
    assert first.pending_track_ids == (7,)
    observe(target, 1, 1, [], [7, 7])
    observe(target, 2, 2, [], [7])
    receipt = target.close_scene(
        scene_id="scene", terminal_frame_id=2, active_stable_ids=[7, 7]
    )
    assert len(receipt.events) == 2
    assert all(event.observed_frame_ids == (0, 1, 2) for event in receipt.events)
    assert all(event.confirmed for event in receipt.events)
    assert receipt.metrics.event_confirmed == 2
    assert receipt.metrics.unique_track_confirmed == 1
    assert receipt.metrics.unique_track_total == 1


def test_ttl_allows_ten_misses_and_retires_on_the_next_true_keyframe():
    target = ledger()
    observe(target, 0, 0, [seed()], [7])
    for step in range(1, 11):
        result = observe(target, step, step, [], [])
        assert result.retired_track_ids == ()
    expired = observe(target, 11, 11, [], [])
    assert expired.newly_retired_track_ids == (7,)
    assert expired.retired_track_ids == (7,)
    receipt = target.close_scene(
        scene_id="scene", terminal_frame_id=12, active_stable_ids=[]
    )
    event = receipt.events[0]
    assert event.status == "retired_probationary"
    assert event.retirement_frame_id == 11
    assert event.retirement_keyframe_step == 11
    assert event.reason == "ttl_expired"


def test_unresolved_committed_identity_is_retired_and_coverage_is_reported():
    target = ledger()
    unresolved = seed(
        55, None, kind="unresolved", target_ids=(), probability=0.5, margin=0.0
    )
    observe(target, 2, 0, [unresolved], [])
    assert target.snapshot().pending_track_ids == ()
    receipt = target.close_scene(
        scene_id="scene", terminal_frame_id=2, active_stable_ids=[]
    )
    event = receipt.events[0]
    assert event.status == "retired_unresolved"
    assert event.observed_frame_ids == ()
    assert event.reason == "unresolved_committed_stable_id"
    assert receipt.metrics.event_unresolved == 1
    assert receipt.metrics.event_evaluable == 0
    assert receipt.metrics.unique_track_total == 0


def test_diagnostic_labels_cannot_change_any_state_transition():
    positive = ledger()
    false_like = ledger()
    left = seed(kind="birth", target_ids=())
    right = seed(kind="ambiguous_history", target_ids=(8, 9))
    for target, event in ((positive, left), (false_like, right)):
        observe(target, 0, 0, [event], [7])
    assert positive.snapshot() == false_like.snapshot()
    for step in (1, 2, 3):
        observe(positive, step, step, [], [7])
        observe(false_like, step, step, [], [7])
        assert positive.snapshot() == false_like.snapshot()

    left_receipt = positive.close_scene(
        scene_id="scene", terminal_frame_id=3, active_stable_ids=[7]
    )
    right_receipt = false_like.close_scene(
        scene_id="scene", terminal_frame_id=3, active_stable_ids=[7]
    )
    left_event = left_receipt.events[0]
    right_event = right_receipt.events[0]
    operational_fields = {
        "observed_frame_ids",
        "observed_keyframe_steps",
        "confirmed",
        "confirmation_frame_id",
        "confirmation_keyframe_step",
        "retired",
        "retirement_frame_id",
        "retirement_keyframe_step",
        "status",
        "latency_frames",
        "latency_keyframes",
        "reason",
    }
    assert {
        key: getattr(left_event, key) for key in operational_fields
    } == {key: getattr(right_event, key) for key in operational_fields}
    assert left_receipt.metrics.event_confirmed_precision == 1.0
    assert right_receipt.metrics.event_confirmed_precision == 0.0


def test_event_and_unique_track_metrics_dedupe_shared_stable_identity():
    target = ledger()
    observe(
        target,
        0,
        0,
        [
            seed(1, 10, kind="birth"),
            seed(2, 10, kind="birth"),
            seed(3, 20, kind="unique_history", target_ids=(4,)),
        ],
        [10, 20],
    )
    observe(target, 1, 1, [], [10, 20])
    observe(target, 2, 2, [], [10, 20])
    receipt = target.close_scene(
        scene_id="scene", terminal_frame_id=2, active_stable_ids=[10, 20]
    )
    metrics = receipt.metrics
    assert metrics.event_confirmed == 3
    assert metrics.event_confirmed_precision == pytest.approx(2 / 3)
    assert metrics.event_confirmed_retention == 1.0
    assert metrics.unique_track_confirmed == 2
    assert metrics.unique_track_confirmed_precision == pytest.approx(1 / 2)
    assert metrics.unique_track_confirmed_retention == 1.0


def test_track_capacity_fails_closed_but_retains_complete_event_receipts():
    target = ledger(max_pending_tracks=1)
    result = observe(
        target,
        0,
        0,
        [seed(1, 10), seed(2, 20)],
        [10, 20],
    )
    assert result.pending_track_ids == (10,)
    assert result.capacity_rejected_event_indices == (1,)
    assert result.audit_complete is False
    receipt = target.close_scene(
        scene_id="scene", terminal_frame_id=0, active_stable_ids=[10, 20]
    )
    assert len(receipt.events) == 2
    assert receipt.events[1].status == "capacity_rejected"
    assert receipt.events[1].reason == "max_pending_tracks_exceeded"
    assert receipt.pending_track_cap_hits == 1
    assert receipt.pending_track_capacity_rejected_events == 1
    assert receipt.audit_complete is False


def test_event_capacity_rejects_whole_batch_but_keyframe_stays_live():
    target = ledger(max_birth_events=2)
    observe(target, 0, 0, [seed(1, 10)], [10])
    before = target.snapshot()
    overflow = observe(
        target,
        1,
        1,
        [seed(2, 20), seed(3, 30)],
        [10, 20, 30],
    )
    after = target.snapshot()
    assert after.birth_events == before.birth_events == 1
    assert after.last_frame_id == 1
    assert after.last_keyframe_step == 1
    assert after.pending_track_ids == (10,)
    assert after.birth_event_cap_hits == 1
    assert after.birth_event_capacity_rejected_events == 2
    assert after.audit_complete is False
    assert overflow.enrolled_events == ()
    assert overflow.birth_event_capacity_rejected_events == 2
    assert overflow.observed_track_ids == (10,)
    assert overflow.pending_track_ids == (10,)
    assert overflow.audit_complete is False

    # The overflow keyframe supplied the second causal view.  A later normal
    # keyframe remains contiguous and can still confirm the pre-existing track.
    continued = observe(target, 2, 2, [], [10])
    assert continued.newly_confirmed_track_ids == (10,)
    assert continued.confirmed_track_ids == (10,)
    receipt = target.close_scene(
        scene_id="scene", terminal_frame_id=2, active_stable_ids=[10]
    )
    assert len(receipt.events) == 1
    assert receipt.events[0].observed_keyframe_steps == (0, 1, 2)
    assert receipt.events[0].confirmed is True
    assert receipt.keyframes_observed == 3
    assert receipt.birth_event_cap_hits == 1
    assert receipt.birth_event_overflow_attempts == 2
    assert receipt.birth_event_capacity_rejected_events == 2
    assert receipt.audit_complete is False
    summary = target.summary()
    assert summary["event_count"] == 1
    assert summary["birth_event_cap_hits"] == 1
    assert summary["birth_event_overflow_attempts"] == 2
    assert summary["birth_event_capacity_rejected_events"] == 2
    assert summary["keyframes_observed"] == 3


def test_event_capacity_overflow_keyframe_advances_existing_track_ttl():
    target = ledger(max_birth_events=1)
    observe(target, 0, 0, [seed(1, 10)], [10])

    # This rejected seed cannot create track 20, but step 1 is still the first
    # miss for existing track 10.
    overflow = observe(target, 1, 1, [seed(2, 20)], [20])
    assert overflow.enrolled_events == ()
    assert overflow.observed_track_ids == ()
    assert overflow.pending_track_ids == (10,)
    assert overflow.birth_event_capacity_rejected_events == 1

    for step in range(2, 11):
        current = observe(target, step, step, [], [])
        assert current.retired_track_ids == ()
    expired = observe(target, 11, 11, [], [])
    assert expired.newly_retired_track_ids == (10,)
    assert expired.retired_track_ids == (10,)
    assert 20 not in expired.pending_track_ids
    assert 20 not in expired.confirmed_track_ids
    assert 20 not in expired.retired_track_ids

    receipt = target.close_scene(
        scene_id="scene", terminal_frame_id=11, active_stable_ids=[]
    )
    assert len(receipt.events) == 1
    assert receipt.events[0].retirement_keyframe_step == 11
    assert receipt.events[0].reason == "ttl_expired"
    assert receipt.birth_event_cap_hits == 1
    assert receipt.birth_event_capacity_rejected_events == 1
    assert receipt.keyframes_observed == 12


@pytest.mark.parametrize(
    "field,value",
    [
        ("top_probability", float("nan")),
        ("top_probability", float("inf")),
        ("margin", float("-inf")),
        ("margin", -0.1),
        ("top_probability", 1.1),
    ],
)
def test_nonfinite_and_out_of_range_diagnostics_are_rejected(field, value):
    kwargs = {
        "proposal_id": 1,
        "committed_stable_id": 2,
        "top_probability": 0.8,
        "margin": 0.2,
        "native_target_kind": "birth",
        "native_target_ids": (),
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match="finite number"):
        SideBirthSeedEvent(**kwargs)


@pytest.mark.parametrize(
    "kind,ids",
    [
        ("birth", (1,)),
        ("unique_history", ()),
        ("unique_history", (1, 2)),
        ("ambiguous_history", (1,)),
        ("unresolved", (1,)),
        ("other", ()),
    ],
)
def test_native_diagnostic_taxonomy_and_cardinality_are_strict(kind, ids):
    with pytest.raises(ValueError, match="native_target"):
        seed(kind=kind, target_ids=ids)


def test_validation_is_transactional_and_context_rows_are_supported():
    target = ledger()
    raw = {
        "scene_id": "scene",
        "frame_id": 4,
        "keyframe_step": 0,
        "proposal_id": 9,
        "committed_stable_id": 3,
        "top_probability": 0.7,
        "margin": 0.1,
        "native_target_kind": "birth",
        "native_target_ids": [],
    }
    original = copy.deepcopy(raw)
    with pytest.raises(ValueError, match="must be present"):
        observe(target, 4, 0, [raw], [])
    assert target.snapshot().scene_id is None
    assert raw == original
    result = observe(target, 4, 0, [raw], [3])
    assert result.enrolled_events[0].scene_id == "scene"
    assert result.enrolled_events[0].frame_id == 4
    assert result.enrolled_events[0].keyframe_step == 0
    before = target.snapshot()
    with pytest.raises(ValueError, match="contiguous"):
        observe(target, 5, 2, [], [3])
    assert target.snapshot() == before


def test_receipt_is_frozen_deeply_tupled_and_strictly_json_safe():
    target = ledger()
    event = seed(target_ids=[])
    observe(target, 0, 0, [event], [7, 7])
    observe(target, 1, 1, [], [7])
    observe(target, 2, 2, [], [7])
    receipt = target.close_scene(
        scene_id="scene", terminal_frame_id=2, active_stable_ids=[7]
    )
    with pytest.raises(FrozenInstanceError):
        receipt.scene_id = "other"
    with pytest.raises(FrozenInstanceError):
        receipt.events[0].status = "other"
    with pytest.raises(FrozenInstanceError):
        event.native_target_ids = (3,)
    assert isinstance(receipt.events, tuple)
    assert isinstance(receipt.events[0].observed_frame_ids, tuple)
    payload = receipt.to_json_dict()
    strict = json.loads(json.dumps(payload, allow_nan=False, sort_keys=True))
    assert strict["schema"] == SCHEMA
    assert strict["active_authorized"] is False
    assert strict["ground_truth_access"] is False
    assert strict["detector_score_access"] is False
    assert strict["clip_access"] is False
    assert strict["puf_access"] is False
    assert strict["puf_event_source"] is True
    assert strict["puf_state_access"] is False
    assert strict["event_source"] == "puf_arbitration_lite.action_birth"
    assert strict["puf_birth_event_input"] is True
    assert strict["puf_internal_state_access"] is False
    assert strict["puf_access_semantics"] == "direct_module_access"
    assert strict["module_training_free"] is True
    assert strict["no_additional_training"] is True
    assert strict["native_labels_used_for_state_transitions"] is False
    assert strict["native_labels_used_at_close_only"] is True
    payload["events"][0]["observed_frame_ids"].append(999)
    assert receipt.events[0].observed_frame_ids == (0, 1, 2)


def test_public_surface_and_imports_make_observer_contract_explicit():
    parameters = set(
        inspect.signature(
            SideBirthProbationLiteLedger.observe_true_cutr_keyframe
        ).parameters
    )
    assert parameters == {
        "self",
        "scene_id",
        "frame_id",
        "keyframe_step",
        "birth_events",
        "observed_stable_ids",
    }
    tree = ast.parse(inspect.getsource(module))
    imported_roots = {
        node.names[0].name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
    }
    imported_roots.update(
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert "torch" not in imported_roots
    assert "clip" not in imported_roots
    assert "puf_lite" not in imported_roots
    assert "puf_arbitration_lite" not in imported_roots
    summary = ledger().summary()
    assert summary["schema"] == SCHEMA
    assert summary["training_free"] is True
    assert summary["causal"] is True
    assert summary["observer_only"] is True
    assert summary["active_authorized"] is False


def test_pipeline_timing_is_finite_and_reported_after_close():
    target = ledger()
    target.record_pipeline_timing(0.125)
    observe(target, 0, 0, [], [])
    target.close_scene(
        scene_id="scene", terminal_frame_id=0, active_stable_ids=[]
    )
    summary = target.summary()
    assert summary["pipeline_observe_calls"] == 1
    assert summary["pipeline_observe_ms_mean"] == pytest.approx(0.125)
    assert summary["pipeline_observe_ms_p95"] == pytest.approx(0.125)
    for value in (True, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and non-negative"):
            target.record_pipeline_timing(value)


def test_ttl_retired_probation_can_remain_native_active_at_terminal():
    target = ledger()
    observe(target, 0, 0, [seed(stable_id=7)], [7])
    for step in range(1, 12):
        observe(target, step, step, [], [])
    assert 7 in target.snapshot().retired_track_ids
    receipt = target.close_scene(
        scene_id="scene", terminal_frame_id=11, active_stable_ids=[7]
    )
    event = receipt.events[0]
    assert event.status == "retired_probationary"
    assert event.active_at_terminal is True
    assert event.to_json_dict()["active_at_terminal"] is True


def test_ttl_retired_identity_can_reappear_without_crashing_or_revival():
    target = ledger()
    observe(target, 0, 0, [seed(stable_id=7)], [7])
    for step in range(1, 12):
        observe(target, step, step, [], [])
    result = observe(target, 12, 12, [], [7])
    assert 7 in result.retired_track_ids
    receipt = target.close_scene(
        scene_id="scene", terminal_frame_id=12, active_stable_ids=[7]
    )
    assert receipt.events[0].confirmed is False
    assert target.summary()["late_reappearances"] == 1


def test_later_event_does_not_inherit_pre_enrollment_confirmation_views():
    target = ledger()
    observe(target, 0, 0, [seed(proposal_id=1, stable_id=7)], [7])
    observe(target, 1, 1, [], [7])
    observe(target, 2, 2, [], [7])
    later = seed(
        proposal_id=2,
        stable_id=7,
        kind="unique_history",
        target_ids=(7,),
    )
    observe(target, 3, 3, [later], [7])
    receipt = target.close_scene(
        scene_id="scene", terminal_frame_id=3, active_stable_ids=[7]
    )
    first, second = receipt.events
    assert first.confirmed is True
    assert first.observed_keyframe_steps == (0, 1, 2, 3)
    assert second.confirmed is False
    assert second.observed_keyframe_steps == (3,)
    assert receipt.metrics.event_confirmed_native_birth == 1
