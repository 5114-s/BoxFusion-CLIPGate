import json
import random

import numpy as np

from boxfusion.observer_track_adapter import (
    ObserverAdapterConfig,
    ObserverBoundaryError,
    ObserverTrackAdapter,
    assert_native_rng_digest_unchanged,
    build_observer_track_adapter,
    capture_native_rng_digest,
)


class FakeBoxManager:
    def __init__(self):
        self.fusion_list = []
        self.observer_track_registry = None
        self.observer_track_token = None
        self.observer_track_error = None

    def append_rows(self, count):
        self.fusion_list.extend([[index] for index in range(count)])

    def attach_observer_track_registry(self, registry, token):
        registry.assert_native_row_count(token, len(self.fusion_list))
        self.observer_track_registry = registry
        self.observer_track_token = token
        self.observer_track_error = None
        return True

    def detach_observer_track_registry(self):
        result = (
            self.observer_track_registry,
            self.observer_track_token,
            self.observer_track_error,
        )
        self.observer_track_registry = None
        self.observer_track_token = None
        return result

    def associate(self, winner, losers, stage="spatial"):
        self.observer_track_registry.record_association(
            self.observer_track_token, winner, losers, stage=stage
        )

    def keep(self, indices):
        self.observer_track_registry.apply_keep(
            self.observer_track_token, indices
        )
        self.fusion_list = [self.fusion_list[index] for index in indices]
        self.observer_track_registry.assert_native_row_count(
            self.observer_track_token, len(self.fusion_list)
        )


def shadow_adapter(scene="scene0000_00", **kwargs):
    return ObserverTrackAdapter(
        ObserverAdapterConfig(mode="shadow", **kwargs), scene_id=scene
    )


def commit_first_frame(adapter, manager, proposals=(10, 11)):
    token = adapter.begin_keyframe(0, proposals)
    manager.append_rows(len(proposals))
    assert adapter.attach(manager, token)
    trace = adapter.finalize(manager, token)
    assert trace is not None
    return trace


def test_default_config_is_disabled_and_builder_does_not_mutate_input():
    cfg = {"association": {"group3d_lite": {}}}
    before = json.dumps(cfg, sort_keys=True)
    adapter = build_observer_track_adapter(cfg, scene_id="scene0000_00")
    assert not adapter.enabled
    assert adapter.begin_keyframe(0, [1]) is None
    assert json.dumps(cfg, sort_keys=True) == before


def test_only_disabled_and_shadow_modes_are_accepted():
    assert ObserverAdapterConfig(mode="disabled").mode == "disabled"
    assert ObserverAdapterConfig(mode="shadow").mode == "shadow"
    try:
        ObserverAdapterConfig(mode="active")
    except ValueError as error:
        assert "not implemented" in str(error)
    else:  # pragma: no cover
        raise AssertionError("active mode must remain gated")


def test_begin_attach_finalize_reports_reserved_and_unmatched_status():
    adapter = shadow_adapter()
    manager = FakeBoxManager()
    first = commit_first_frame(adapter, manager)
    assert first.native_status == ("unmatched_retained", "unmatched_retained")
    assert first.native_unmatched_retained_proposal_ids == (10, 11)

    token = adapter.begin_keyframe(25, [20, 21])
    manager.append_rows(2)
    assert adapter.attach(manager, token)
    manager.associate(2, [0], stage="spatial")
    manager.associate(3, [1], stage="correspondence")
    manager.keep([2, 3])
    second = adapter.finalize(manager, token)

    assert second is not None
    assert second.begin_past_track_ids == (10, 11)
    assert second.proposal_track_ids == (10, 11)
    assert second.native_target_track_ids == ((10,), (11,))
    assert second.native_status == (
        "matched_past_retained",
        "matched_past_retained",
    )
    assert second.reserved_past_track_ids == (10, 11)
    assert second.native_matched_proposal_ids == (20, 21)
    assert second.native_unmatched_retained_proposal_ids == ()
    assert second.active_track_ids == (10, 11)


def test_current_to_current_unmatched_candidates_are_deduplicated_by_track():
    adapter = shadow_adapter()
    manager = FakeBoxManager()
    token = adapter.begin_keyframe(0, [30, 31])
    manager.append_rows(2)
    assert adapter.attach(manager, token)
    manager.associate(0, [1])
    manager.keep([0])
    trace = adapter.finalize(manager, token)

    assert trace is not None
    assert trace.proposal_track_ids == (30, 30)
    assert trace.native_target_track_ids == ((), ())
    assert trace.native_status == ("unmatched_retained", "unmatched_retained")
    assert trace.native_unmatched_retained_proposal_ids == (30,)


def test_terminal_mapping_uses_final_native_keep_order():
    adapter = shadow_adapter()
    manager = FakeBoxManager()
    commit_first_frame(adapter, manager, proposals=(10, 11, 12))
    mapping = adapter.terminal_mapping(
        [2, 0], native_row_count=3, current_frame_id=40
    )
    assert mapping is not None
    assert mapping.snapshot_frame_id == 0
    assert mapping.kept_native_indices == (2, 0)
    assert mapping.output_track_ids == (12, 10)


def test_one_shot_proposal_container_is_not_consumed_and_trace_fails_open():
    adapter = shadow_adapter()
    proposals = iter([1, 2, 3])
    assert adapter.begin_keyframe(0, proposals) is None
    assert list(proposals) == [1, 2, 3]
    assert not adapter.trace_valid
    assert "bounded indexable" in adapter.errors[0]


def test_association_inputs_are_copied_and_never_modified():
    adapter = shadow_adapter()
    manager = FakeBoxManager()
    token = adapter.begin_keyframe(0, [10, 11])
    manager.append_rows(2)
    assert adapter.attach(manager, token)
    losers = [1]
    manager.associate(0, losers)
    assert losers == [1]


def test_abort_disables_only_observer_and_leaves_native_rows_untouched():
    adapter = shadow_adapter()
    manager = FakeBoxManager()
    token = adapter.begin_keyframe(0, [10])
    manager.append_rows(1)
    assert adapter.attach(manager, token)
    native_before = [list(row) for row in manager.fusion_list]
    adapter.abort_keyframe(token)
    assert manager.fusion_list == native_before
    assert not adapter.trace_valid
    assert adapter.begin_keyframe(25, [20]) is None


def test_native_rng_digest_assertion_and_fail_open_boundary_restore_rng():
    adapter = shadow_adapter()
    native = {
        "scores": np.array([0.4, 0.9], dtype=np.float32),
        "fusion_list": [[0], [1]],
    }
    before = capture_native_rng_digest(native)
    after = capture_native_rng_digest(native)
    assert_native_rng_digest_unchanged(before, after)

    with adapter.observer_boundary(native, label="rng_violation"):
        random.random()
        np.random.random()

    restored = capture_native_rng_digest(native)
    assert before == restored
    assert not adapter.trace_valid
    assert adapter.diagnostics()["boundary_violations"] == 1


def test_explicit_digest_assertion_detects_native_mutation():
    native = {"scores": np.array([1.0], dtype=np.float32)}
    before = capture_native_rng_digest(native)
    native["scores"][0] = 0.0
    after = capture_native_rng_digest(native)
    try:
        assert_native_rng_digest_unchanged(before, after)
    except ObserverBoundaryError as error:
        assert "native fields" in str(error)
    else:  # pragma: no cover
        raise AssertionError("native mutation was not detected")


def test_atomic_json_diagnostic_contains_stable_trace(tmp_path):
    adapter = shadow_adapter(diagnostics_root=str(tmp_path))
    manager = FakeBoxManager()
    commit_first_frame(adapter, manager)
    adapter.terminal_mapping([0, 1], native_row_count=2, current_frame_id=1)
    destination = adapter.write_diagnostics()

    assert destination == tmp_path / "scene0000_00.observer_tracks.json"
    payload = json.loads(destination.read_text())
    assert payload["trace_valid"] is True
    assert payload["frames"][0]["proposal_ids"] == [10, 11]
    assert payload["frames"][0]["native_status"] == [
        "unmatched_retained",
        "unmatched_retained",
    ]
    assert payload["terminal"]["output_track_ids"] == [10, 11]
    assert not list(tmp_path.glob("*.tmp"))


def test_trace_caps_fail_open_before_native_attachment():
    adapter = shadow_adapter(max_trace_proposals=1)
    proposals = [10, 11]
    assert adapter.begin_keyframe(0, proposals) is None
    assert proposals == [10, 11]
    assert not adapter.trace_valid
    assert "proposal trace cap" in adapter.errors[0]
