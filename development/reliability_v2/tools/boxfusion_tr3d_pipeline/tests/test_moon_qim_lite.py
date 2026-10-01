import numpy as np
import pytest

from boxfusion.moon_qim_lite import (
    CausalFusionIdRegistry,
    MoonQIMLiteObserver,
    build_moon_qim_lite,
    derive_native_target_track_ids,
    resolve_moon_qim_lite_config,
)


def cube(center, size=0.4):
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
        dtype=np.float32,
    )
    return signs * (size / 2.0) + np.asarray(center, dtype=np.float32)


def enabled_config(**overrides):
    config = {
        "enabled": True,
        "observer_only": True,
        "voxel_size_m": 0.30,
        "samples_per_axis": 3,
        "neighbor_radius": 1,
        "max_candidates_per_query": 8,
        "max_tracks": 32,
        "track_ttl_keyframes": 4,
        "max_postings_per_key": 16,
    }
    config.update(overrides)
    return config


def test_config_is_default_off_strict_and_observer_only():
    resolved = resolve_moon_qim_lite_config()
    assert resolved["enabled"] is False
    assert resolved["observer_only"] is True
    assert build_moon_qim_lite({}).enabled is False

    with pytest.raises(ValueError, match="Unknown moon_qim_lite"):
        resolve_moon_qim_lite_config({"typo": 1})
    with pytest.raises(ValueError, match="active association is not authorized"):
        resolve_moon_qim_lite_config(
            {"enabled": True, "observer_only": False}
        )


def test_query_then_update_is_strictly_causal():
    observer = MoonQIMLiteObserver(enabled_config())
    first = observer.query(
        scene_id="scene0000_00",
        frame_id=0,
        proposal_ids=np.asarray([100], dtype=np.int64),
        proposal_corners_world=np.stack([cube([0, 0, 1])]),
    )
    assert first.history_max_frame_id is None
    assert first.candidates == ((),)

    observer.update(
        scene_id="scene0000_00",
        frame_id=0,
        track_ids=np.asarray([7], dtype=np.int64),
        track_corners_world=np.stack([cube([0, 0, 1])]),
    )
    second = observer.query(
        scene_id="scene0000_00",
        frame_id=25,
        proposal_ids=np.asarray([101], dtype=np.int64),
        proposal_corners_world=np.stack([cube([0.04, 0, 1])]),
    )
    assert second.history_max_frame_id == 0
    assert [candidate.track_id for candidate in second.candidates[0]][0] == 7
    observer.update(
        scene_id="scene0000_00",
        frame_id=25,
        track_ids=np.asarray([7], dtype=np.int64),
        track_corners_world=np.stack([cube([0, 0, 1])]),
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        observer.query(
            scene_id="scene0000_00",
            frame_id=25,
            proposal_ids=np.asarray([102], dtype=np.int64),
            proposal_corners_world=np.stack([cube([0, 0, 1])]),
        )


def test_pending_query_rejects_time_travel_and_duplicate_observation():
    observer = MoonQIMLiteObserver(enabled_config())
    observer.update(
        scene_id="scene",
        frame_id=0,
        track_ids=np.asarray([1]),
        track_corners_world=np.stack([cube([0, 0, 1])]),
    )
    batch = observer.query(
        scene_id="scene",
        frame_id=10,
        proposal_ids=np.asarray([100]),
        proposal_corners_world=np.stack([cube([0, 0, 1])]),
    )
    observer.observe_native_targets(batch, [(1,)])
    with pytest.raises(ValueError, match="already observed"):
        observer.observe_native_targets(batch, [(1,)])
    with pytest.raises(ValueError, match="same frame"):
        observer.update(
            scene_id="scene",
            frame_id=5,
            track_ids=np.asarray([1]),
            track_corners_world=np.stack([cube([0, 0, 1])]),
        )


def test_candidates_are_unique_ranked_and_input_order_independent():
    config = enabled_config(neighbor_radius=0)
    forward = MoonQIMLiteObserver(config)
    reverse = MoonQIMLiteObserver(config)
    track_ids = np.asarray([9, 3], dtype=np.int64)
    track_boxes = np.stack([cube([0.12, 0, 1]), cube([0.02, 0, 1])])
    forward.update(
        scene_id="scene", frame_id=0,
        track_ids=track_ids, track_corners_world=track_boxes,
    )
    reverse.update(
        scene_id="scene", frame_id=0,
        track_ids=track_ids[::-1], track_corners_world=track_boxes[::-1],
    )
    query = np.stack([cube([0, 0, 1])])
    first = forward.query(
        scene_id="scene", frame_id=1,
        proposal_ids=np.asarray([20]), proposal_corners_world=query,
    )
    second = reverse.query(
        scene_id="scene", frame_id=1,
        proposal_ids=np.asarray([20]), proposal_corners_world=query,
    )
    first_ids = [candidate.track_id for candidate in first.candidates[0]]
    second_ids = [candidate.track_id for candidate in second.candidates[0]]
    assert first_ids == second_ids
    assert len(first_ids) == len(set(first_ids))
    assert first_ids[0] == 3


def test_native_association_metrics_exclude_unresolved_and_births():
    observer = MoonQIMLiteObserver(enabled_config())
    observer.update(
        scene_id="scene",
        frame_id=0,
        track_ids=np.asarray([5, 8]),
        track_corners_world=np.stack(
            [cube([0, 0, 1]), cube([2, 0, 1])]
        ),
    )
    batch = observer.query(
        scene_id="scene",
        frame_id=1,
        proposal_ids=np.asarray([20, 21, 22, 23]),
        proposal_corners_world=np.stack(
            [
                cube([0.01, 0, 1]),
                cube([2.01, 0, 1]),
                cube([5, 0, 1]),
                cube([8, 0, 1]),
            ]
        ),
    )
    observer.observe_native_targets(batch, [(5,), (8,), (), None])
    summary = observer.summary()
    assert summary["native_matches"] == 2
    assert summary["native_births"] == 1
    assert summary["native_unresolved"] == 1
    assert summary["recall_at_1_rate"] == 1.0
    assert summary["recall_at_3_rate"] == 1.0
    assert summary["recall_at_k_rate"] == 1.0
    assert summary["training_free"] is True
    assert summary["causal"] is True
    assert summary["semantic_access"] is False
    assert summary["semantic_mutation"] is False


def test_native_targets_survive_row_compaction_and_mark_unknown_removals():
    targets = derive_native_target_track_ids(
        proposal_ids=np.asarray([100, 101, 102, 103]),
        previous_fusion_groups=[[4, 7], [20, 21]],
        previous_stable_ids=np.asarray([4, 20], dtype=np.int64),
        # Row order changed. Proposal 100 merged with old track 20, proposal
        # 101 is a retained birth, 102 merged with old track 4, and 103 was
        # suppressed without a retained source-id trace.
        current_fusion_groups=[[101], [20, 21, 100], [4, 7, 102]],
    )
    assert targets == ((20,), (), (4,), None)


def test_native_event_scores_match_when_five_view_group_cannot_append_id():
    kwargs = dict(
        proposal_ids=np.asarray([100]),
        previous_fusion_groups=[[1, 2, 3, 4, 5]],
        previous_stable_ids=np.asarray([1], dtype=np.int64),
        current_fusion_groups=[[1, 2, 3, 4, 5]],
    )
    assert derive_native_target_track_ids(**kwargs) == (None,)
    assert derive_native_target_track_ids(
        **kwargs,
        association_events=[
            {
                "stage": "spatial",
                "winner_members": (1, 2, 3, 4, 5),
                "loser_members": (100,),
            }
        ],
    ) == ((1,),)


def test_causal_registry_never_transfers_id_when_collision_set_changes():
    registry = CausalFusionIdRegistry()
    first_groups = [[1, 5], [1, 6]]
    first_ids = registry.update(first_groups)
    assert np.array_equal(registry.ids_for(first_groups), first_ids)
    assert len(np.unique(first_ids)) == 2

    second_groups = [[1, 4], [1, 6], [1, 5]]
    second_ids = registry.update(second_groups)
    by_group = {
        tuple(group): int(track_id)
        for group, track_id in zip(second_groups, second_ids)
    }
    assert by_group[(1, 5)] == int(first_ids[0])
    assert by_group[(1, 6)] == int(first_ids[1])
    assert by_group[(1, 4)] not in set(int(value) for value in first_ids)

    # Row permutation alone cannot change either inherited identity.
    third_groups = [[1, 5], [1, 4], [1, 6]]
    third_ids = registry.update(third_groups)
    third_by_group = {
        tuple(group): int(track_id)
        for group, track_id in zip(third_groups, third_ids)
    }
    assert third_by_group == by_group


def test_memory_track_and_candidate_caps_are_hard_bounds():
    observer = MoonQIMLiteObserver(
        enabled_config(max_tracks=3, max_candidates_per_query=2)
    )
    observer.update(
        scene_id="scene",
        frame_id=0,
        track_ids=np.arange(10, dtype=np.int64),
        track_corners_world=np.stack(
            [cube([0.01 * index, 0, 1]) for index in range(10)]
        ),
    )
    snapshot = observer.snapshot()
    assert len(snapshot["track_ids"]) == 3
    assert snapshot["posting_count"] <= (
        3 * 3 ** 3
    )
    batch = observer.query(
        scene_id="scene",
        frame_id=1,
        proposal_ids=np.asarray([100]),
        proposal_corners_world=np.stack([cube([0.05, 0, 1])]),
    )
    assert len(batch.candidates[0]) <= 2
    assert observer.summary()["evicted_tracks"] == 7


def test_recent_track_reenters_a_capped_crowded_posting():
    observer = MoonQIMLiteObserver(
        enabled_config(
            neighbor_radius=0,
            max_postings_per_key=1,
        )
    )
    shared_boxes = np.stack([cube([0, 0, 1]), cube([0, 0, 1])])
    observer.update(
        scene_id="scene",
        frame_id=0,
        track_ids=np.asarray([1, 2]),
        track_corners_world=shared_boxes,
    )
    first = observer.query(
        scene_id="scene",
        frame_id=1,
        proposal_ids=np.asarray([100]),
        proposal_corners_world=np.stack([cube([0, 0, 1])]),
    )
    assert [item.track_id for item in first.candidates[0]] == [1]

    # Only track 2 is observed at the next commit, making it the most recent
    # occupant of every capped cell.  It must become queryable again.
    observer.update(
        scene_id="scene",
        frame_id=1,
        track_ids=np.asarray([2]),
        track_corners_world=shared_boxes[1:],
    )
    second = observer.query(
        scene_id="scene",
        frame_id=2,
        proposal_ids=np.asarray([101]),
        proposal_corners_world=np.stack([cube([0, 0, 1])]),
    )
    assert [item.track_id for item in second.candidates[0]] == [2]


def test_scene_isolation_and_invalid_inputs_fail_closed():
    observer = MoonQIMLiteObserver(enabled_config())
    observer.update(
        scene_id="scene_a",
        frame_id=0,
        track_ids=np.asarray([1]),
        track_corners_world=np.stack([cube([0, 0, 1])]),
    )
    with pytest.raises(ValueError, match="bound to scene_a"):
        observer.query(
            scene_id="scene_b",
            frame_id=1,
            proposal_ids=np.asarray([2]),
            proposal_corners_world=np.stack([cube([0, 0, 1])]),
        )

    observer.reset_scene("scene_b")
    assert observer.snapshot()["track_ids"] == ()
    with pytest.raises(ValueError, match="finite"):
        observer.update(
            scene_id="scene_b",
            frame_id=0,
            track_ids=np.asarray([1]),
            track_corners_world=np.full((1, 8, 3), np.nan),
        )
    with pytest.raises(ValueError, match="unique"):
        observer.update(
            scene_id="scene_b",
            frame_id=0,
            track_ids=np.asarray([1, 1]),
            track_corners_world=np.stack(
                [cube([0, 0, 1]), cube([1, 0, 1])]
            ),
        )


def test_inputs_are_not_mutated():
    observer = MoonQIMLiteObserver(enabled_config())
    track_ids = np.asarray([4], dtype=np.int64)
    track_boxes = np.stack([cube([0, 0, 1])])
    ids_before = track_ids.copy()
    boxes_before = track_boxes.copy()
    observer.update(
        scene_id="scene", frame_id=0,
        track_ids=track_ids, track_corners_world=track_boxes,
    )
    observer.query(
        scene_id="scene", frame_id=1,
        proposal_ids=track_ids, proposal_corners_world=track_boxes,
    )
    assert np.array_equal(track_ids, ids_before)
    assert np.array_equal(track_boxes, boxes_before)
