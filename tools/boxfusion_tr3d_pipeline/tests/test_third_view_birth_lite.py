import ast
import copy
import inspect
import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import boxfusion.third_view_birth_lite as module
from boxfusion.third_view_birth_lite import (
    ThirdViewBirthLiteObserver,
    build_third_view_birth_lite,
    resolve_third_view_birth_lite_config,
)


def enabled_config(**overrides):
    result = {"enabled": True, "observer_only": True}
    result.update(overrides)
    return result


def observer(**overrides):
    return ThirdViewBirthLiteObserver(enabled_config(**overrides))


def observe(
    target,
    frame_id,
    groups,
    stable_ids,
    source_frames,
    *,
    scene_id="scene",
):
    return target.observe(
        scene_id=scene_id,
        frame_id=frame_id,
        current_fusion_groups=groups,
        current_stable_ids=stable_ids,
        source_frame_ids=source_frames,
    )


def finalize(target, stable_ids):
    return target.finalize(final_stable_ids=stable_ids)


def test_config_is_default_off_strict_bounded_and_freezes_three_views():
    resolved = resolve_third_view_birth_lite_config()
    assert resolved == {
        "enabled": False,
        "observer_only": True,
        "min_distinct_source_frames": 3,
        "max_tracks": 1024,
        "max_sources_per_group": 5,
        "max_diagnostic_examples": 64,
    }
    assert build_third_view_birth_lite({}).enabled is False
    assert build_third_view_birth_lite(
        {"dataset": "scannet", "data": {}}
    ).enabled is False
    assert build_third_view_birth_lite(
        {"third_view_birth_lite": enabled_config()}
    ).enabled is True
    with pytest.raises(ValueError, match="Unknown third_view_birth_lite"):
        resolve_third_view_birth_lite_config({"typo": 1})
    with pytest.raises(ValueError, match="observer_only must remain true"):
        resolve_third_view_birth_lite_config(
            {"enabled": True, "observer_only": False}
        )
    with pytest.raises(ValueError, match="frozen three-view"):
        resolve_third_view_birth_lite_config(
            enabled_config(min_distinct_source_frames=2)
        )
    with pytest.raises(ValueError, match="must not exceed 1024"):
        resolve_third_view_birth_lite_config({"max_tracks": 1025})
    with pytest.raises(ValueError, match="must not exceed 5"):
        resolve_third_view_birth_lite_config({"max_sources_per_group": 6})
    with pytest.raises(ValueError, match="must be a boolean"):
        resolve_third_view_birth_lite_config({"enabled": 1})


def test_disabled_observer_has_no_implicit_runtime_authority():
    target = ThirdViewBirthLiteObserver()
    with pytest.raises(RuntimeError, match="disabled"):
        observe(target, 0, [[0]], [7], [0])
    with pytest.raises(RuntimeError, match="disabled"):
        finalize(target, [])


def test_three_distinct_source_frames_confirm_once_and_confirmation_is_sticky():
    target = observer()

    first = observe(target, 0, [[0]], [7], [0])
    assert first.confirmed_tracks == 0
    assert first.probationary_tracks == 1
    result = finalize(target, [7])
    # Native anchors are always preserved.  Only the side-candidate diagnostic
    # is probationary.
    assert result.keep_mask == (True,)
    assert result.would_admit_side_candidate_mask == (False,)

    second = observe(target, 1, [[0, 1]], [7], [0, 1])
    assert second.tracks[0].status == "probationary"
    finalize(target, [7])

    # Two init ids originate in frame 1 and count as one view.
    third = observe(target, 2, [[0, 1, 2, 3]], [7], [0, 1, 1, 2])
    row = third.tracks[0]
    assert row.source_frame_ids == (0, 1, 2)
    assert row.distinct_source_frames == 3
    assert row.status == "confirmed"
    assert row.confirmed_frame_id == 2
    assert row.birth_to_confirm_latency_frames == 2
    result = finalize(target, [7])
    assert result.keep_mask == (True,)
    assert result.would_admit_side_candidate_mask == (True,)

    # The current group can later be small without revoking confirmation.
    fourth = observe(target, 3, [[3]], [7], [0, 1, 1, 3])
    assert fourth.tracks[0].distinct_source_frames == 1
    assert fourth.tracks[0].status == "confirmed"
    assert fourth.tracks[0].confirmed_frame_id == 2
    finalize(target, [7])
    summary = target.summary()
    assert summary["confirmation_events"] == 1
    assert summary["birth_to_confirm_latency_count"] == 1
    assert summary["birth_to_confirm_latency_mean_frames"] == 2.0


def test_sparse_source_mapping_matches_dense_sequence():
    dense = observer()
    sparse = observer()
    dense_sources = [0] * 10
    dense_sources[2], dense_sources[5], dense_sources[9] = 0, 1, 2
    dense_batch = observe(dense, 2, [[2, 5, 9]], [7], dense_sources)
    sparse_batch = sparse.observe(
        scene_id="scene",
        frame_id=2,
        current_fusion_groups=[[2, 5, 9]],
        current_stable_ids=[7],
        source_frame_ids={2: 0, 5: 1, 9: 2},
    )
    assert dense_batch == sparse_batch
    assert finalize(dense, [7]) == finalize(sparse, [7])


def test_confirmation_requires_three_views_in_one_group_not_history_union():
    target = observer()
    observe(target, 1, [[0, 1]], [9], [0, 1])
    finalize(target, [9])
    batch = observe(target, 2, [[1, 2]], [9], [0, 1, 2])
    # Across calls {0,1,2} exists, but the current fusion group has only {1,2}.
    assert batch.tracks[0].status == "probationary"
    assert batch.tracks[0].confirmed_frame_id is None
    assert finalize(target, [9]).would_admit_side_candidate_mask == (False,)


def test_merge_inherits_earliest_birth_confirms_and_retires_absorbed_identity():
    target = observer(max_diagnostic_examples=16)
    first = observe(target, 0, [[0], [1]], [10, 20], [0, 0])
    assert first.new_tracks == 2
    finalize(target, [10, 20])

    merged = observe(target, 4, [[0, 1, 2]], [10], [0, 2, 4])
    row = merged.tracks[0]
    assert row.predecessor_stable_ids == (10, 20)
    assert row.merge_observed is True
    assert row.birth_frame_id == 0
    assert row.confirmed_frame_id == 4
    assert row.birth_to_confirm_latency_frames == 4
    assert merged.merged_predecessors == 1
    assert merged.retired_tracks == 1
    finalize(target, [10])

    summary = target.summary()
    assert summary["merge_events"] == 1
    assert summary["merged_predecessors"] == 1
    assert summary["retired_tracks_total"] == 1
    assert any(
        row["kind"] == "retired"
        and row["stable_id"] == 20
        and row["reason"] == "merged_or_remapped"
        for row in summary["diagnostic_examples"]
    )


def test_track_disappearance_retires_confirmed_and_probationary_states():
    target = observer()
    observe(target, 2, [[0, 1, 2], [3]], [7, 8], [0, 1, 2, 2])
    finalize(target, [7, 8])
    empty = observe(target, 3, [], [], [0, 1, 2, 2])
    assert empty.retired_tracks == 2
    assert empty.tracks == ()
    result = finalize(target, [])
    assert result.keep_mask == ()
    assert result.would_admit_side_candidate_mask == ()
    summary = target.summary()
    assert summary["retired_confirmed"] == 1
    assert summary["retired_probationary"] == 1
    assert summary["active_tracks"] == 0


def test_finalize_accepts_reordered_subset_but_unknown_and_duplicates_fail_closed():
    target = observer()
    observe(target, 0, [[0], [1], [2]], [4, 5, 6], [0, 0, 0])
    with pytest.raises(ValueError, match="unique"):
        finalize(target, [4, 4])
    assert target.snapshot().pending_frame_id == 0
    with pytest.raises(ValueError, match="absent from the latest"):
        finalize(target, [99])
    assert target.snapshot().pending_frame_id == 0
    result = finalize(target, np.asarray([6, 4], dtype=np.int64))
    assert result.stable_ids == (6, 4)
    assert result.keep_mask == (True, True)
    assert result.would_admit_side_candidate_mask == (False, False)
    assert tuple(row.stable_id for row in result.diagnostics) == (6, 4)


def test_transaction_scene_and_causal_order_are_strict_and_retryable():
    target = observer()
    observe(target, 2, [[0]], [7], [2])
    with pytest.raises(ValueError, match="closed by finalize"):
        observe(target, 3, [[1]], [7], [2, 3])
    with pytest.raises(ValueError, match="before reset"):
        target.reset_scene("other")
    finalize(target, [7])
    with pytest.raises(ValueError, match="strictly increasing"):
        observe(target, 2, [[0]], [7], [2])
    with pytest.raises(ValueError, match="bound to scene"):
        observe(target, 3, [[1]], [7], [2, 3], scene_id="other")
    # Failed calls did not open or alter a transaction.
    assert target.snapshot().pending_frame_id is None
    assert target.snapshot().last_observed_frame_id == 2
    target.reset_scene("other")
    assert target.snapshot().active_track_ids == ()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            dict(frame_id=True, groups=[[0]], stable_ids=[1], sources=[0]),
            "frame_id must be an integer",
        ),
        (
            dict(frame_id=0, groups=[[0]], stable_ids=[1.0], sources=[0]),
            "current_stable_ids.*integer",
        ),
        (
            dict(frame_id=0, groups=[[0]], stable_ids=[1], sources=[0.0]),
            "source_frame_ids.*integer",
        ),
        (
            dict(frame_id=0, groups=[[0, 0]], stable_ids=[1], sources=[0]),
            "unique init ids",
        ),
        (
            dict(frame_id=0, groups=[[0], [0]], stable_ids=[1, 2], sources=[0]),
            "must not share init ids",
        ),
        (
            dict(frame_id=0, groups=[[0], [1]], stable_ids=[1, 1], sources=[0, 0]),
            "unique ids",
        ),
        (
            dict(frame_id=0, groups=[[1]], stable_ids=[1], sources=[0]),
            "outside source_frame_ids",
        ),
        (
            dict(frame_id=0, groups=[[0]], stable_ids=[1], sources=[1]),
            "future evidence",
        ),
        (
            dict(frame_id=0, groups=[[0]], stable_ids=[], sources=[0]),
            "row-aligned",
        ),
    ],
)
def test_integer_alignment_partition_and_causality_validation(kwargs, match):
    target = observer()
    with pytest.raises(ValueError, match=match):
        observe(
            target,
            kwargs["frame_id"],
            kwargs["groups"],
            kwargs["stable_ids"],
            kwargs["sources"],
        )
    assert target.snapshot().scene_id is None
    assert target.snapshot().active_track_ids == ()


def test_group_and_track_caps_fail_before_mutating_existing_state():
    target = observer(max_tracks=2)
    observe(target, 0, [[0]], [1], [0])
    finalize(target, [1])
    before = target.snapshot()
    overflow = observe(
        target, 1, [[0, 1, 2, 3, 4, 5]], [1], [0, 1, 1, 1, 1, 1]
    )
    assert overflow.tracks[0].status == "abstained_source_overflow"
    overflow_result = finalize(target, [1])
    assert overflow_result.keep_mask == (True,)
    assert overflow_result.would_admit_side_candidate_mask == (False,)
    assert target.summary()["source_cap_overflow_rows"] == 1
    before = target.snapshot()
    with pytest.raises(ValueError, match="bounded capacity"):
        observe(target, 2, [[0], [1], [2]], [1, 2, 3], [0, 1, 2])
    assert target.snapshot() == before


def test_source_overflow_abstention_is_sticky_but_never_filters_native():
    target = observer()
    batch = observe(
        target,
        5,
        [[0, 1, 2, 3, 4, 5]],
        [8],
        [0, 1, 2, 3, 4, 5],
    )
    assert batch.tracks[0].status == "abstained_source_overflow"
    assert finalize(target, [8]).would_admit_side_candidate_mask == (False,)
    later = observe(target, 6, [[5]], [8], [0, 1, 2, 3, 4, 5])
    assert later.tracks[0].status == "abstained_source_overflow"
    result = finalize(target, [8])
    assert result.keep_mask == (True,)
    assert result.diagnostics[0].reason == "source_cap_overflow"


def test_ambiguous_split_does_not_clone_prior_evidence_or_mutate_state():
    target = observer()
    observe(target, 0, [[0, 1]], [10], [0, 0])
    finalize(target, [10])
    before = target.snapshot()
    with pytest.raises(ValueError, match="ambiguously split"):
        observe(target, 1, [[0], [1]], [20, 30], [0, 0])
    assert target.snapshot() == before


def test_inputs_are_not_mutated_and_results_are_deeply_immutable_scalars():
    target = observer()
    groups = [[2, 0, 1]]
    stable_ids = np.asarray([7], dtype=np.int64)
    sources = np.asarray([0, 1, 2], dtype=np.int64)
    original_groups = copy.deepcopy(groups)
    original_stable = stable_ids.copy()
    original_sources = sources.copy()
    batch = observe(target, 2, groups, stable_ids, sources)
    result = finalize(target, stable_ids)
    assert groups == original_groups
    assert np.array_equal(stable_ids, original_stable)
    assert np.array_equal(sources, original_sources)
    assert batch.tracks[0].source_init_ids == (0, 1, 2)
    with pytest.raises(FrozenInstanceError):
        batch.frame_id = 99
    with pytest.raises(FrozenInstanceError):
        result.keep_mask = ()
    assert isinstance(result.keep_mask, tuple)
    assert isinstance(result.would_admit_side_candidate_mask, tuple)
    assert not any(isinstance(value, np.ndarray) for value in vars(result).values())


def test_summary_is_json_safe_bounded_and_explicitly_not_a_native_filter():
    target = observer(max_diagnostic_examples=2)
    target.record_pipeline_timing(0.125)
    observe(target, 0, [[0]], [1], [0])
    finalize(target, [1])
    observe(target, 1, [[0, 1, 2]], [1], [0, 1, 1])
    finalize(target, [1])
    summary = target.summary()
    json.dumps(summary, sort_keys=True)
    assert summary["training_free"] is True
    assert summary["causal"] is True
    assert summary["bounded_memory"] is True
    assert summary["counterfactual_only"] is True
    assert summary["side_candidate_gate_only"] is True
    assert summary["active_authorized"] is False
    assert summary["native_filter_applied"] is False
    assert summary["native_keep_mask_identity"] is True
    assert summary["native_would_suppress"] == 0
    assert summary["would_suppress_scope"] == "side_candidates_only"
    assert summary["current_cutr_commit_only_contract"] is True
    assert summary["terminal_stale_frames_observed"] is False
    assert summary["native_outputs_mutated"] is False
    assert summary["ground_truth_access"] is False
    assert summary["detector_score_access"] is False
    assert summary["clip_access"] is False
    assert summary["puf_access"] is False
    assert summary["schema"] == "boxfusion.third_view_birth_lite_shadow.v1"
    assert summary["effective_config"]["min_distinct_source_frames"] == 3
    assert summary["pipeline_observe_calls"] == 1
    assert summary["pipeline_observe_ms_mean"] == pytest.approx(0.125)
    assert summary["pipeline_observe_ms_p95"] == pytest.approx(0.125)
    assert "pipeline_mean/p95/max_ms" in target.summary_line()
    assert summary["final_confirmed"] == 0
    assert summary["final_probationary"] == 2
    assert summary["would_suppress"] == 2
    assert len(summary["diagnostic_examples"]) <= 2


@pytest.mark.parametrize("value", [True, -1.0, float("nan"), float("inf")])
def test_pipeline_timing_is_finite_and_non_negative(value):
    target = observer()
    with pytest.raises(ValueError, match="finite non-negative"):
        target.record_pipeline_timing(value)


def test_public_observe_surface_cannot_receive_semantic_or_native_labels():
    parameters = set(inspect.signature(ThirdViewBirthLiteObserver.observe).parameters)
    assert parameters == {
        "self",
        "scene_id",
        "frame_id",
        "current_fusion_groups",
        "current_stable_ids",
        "source_frame_ids",
    }
    tree = ast.parse(inspect.getsource(module))
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "torch" not in imported_roots
    assert "sklearn" not in imported_roots
