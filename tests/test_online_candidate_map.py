import numpy as np
import pytest

from boxfusion.online_candidate_map import (
    EvidenceFrame,
    NativeFrame,
    OnlineCandidateMap,
    OnlineNativeReranker,
    OnlineProposalRecovery,
    unique_native_ids,
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
    dtype=np.float64,
)
K = np.asarray([[100.0, 0.0, 128.0], [0.0, 100.0, 128.0], [0.0, 0.0, 1.0]])
POSE = np.eye(4)


def corners(x=0.0, z=4.0, extent=1.0):
    return SIGNS * (extent / 2.0) + np.asarray([x, 0.0, z])


def projected_box(box):
    uvw = box @ K.T
    uv = uvw[:, :2] / uvw[:, 2:]
    return np.r_[uv.min(0), uv.max(0)]


def evidence(frame, box=None, *, anchors=False):
    box = corners(0.01 * frame) if box is None else np.asarray(box)
    return EvidenceFrame(
        proposal_ids=np.asarray([frame]),
        proposal_boxes_2d=np.asarray([projected_box(box)]),
        proposal_corners=np.asarray([box]),
        proposal_scores=np.asarray([0.8 - 0.01 * frame]),
        anchor_ids=np.asarray([frame]) if anchors else np.empty(0, dtype=np.int64),
        anchor_corners=np.asarray([box]) if anchors else np.empty((0, 8, 3)),
        anchor_scores=np.asarray([0.2]) if anchors else np.empty(0),
    )


def native(ids=(), boxes=(), scores=()):
    return NativeFrame(
        ids=np.asarray(ids, dtype=np.int64),
        corners=np.asarray(boxes, dtype=np.float64).reshape(-1, 8, 3),
        scores=np.asarray(scores, dtype=np.float64),
        camera_to_world=POSE,
        intrinsic=K,
        width=256,
        height=256,
    )


def test_unique_native_ids_preserves_disjoint_lineages_and_resolves_overlap():
    assert unique_native_ids([[5, 7], [2, 9]]).tolist() == [5, 2]

    ids = unique_native_ids([[10, 11], [10], [20], [20]])
    assert ids[:3].tolist() == [11, 10, 20]
    assert ids[3] < 0
    assert len(set(ids.tolist())) == 4


def test_m1p_births_on_third_distinct_keyframe_and_freezes_geometry():
    state = OnlineProposalRecovery()
    empty_native = np.empty((0, 8, 3))
    boxes = [corners(0.00), corners(0.02), corners(-0.01)]
    for ordinal in range(2):
        assert state.update(
            ordinal,
            ordinal,
            proposal_ids=[ordinal],
            proposal_corners=[boxes[ordinal]],
            proposal_scores=[0.9 - 0.1 * ordinal],
            native_corners=empty_native,
        ) == []
        assert state.rows() == []
    born = state.update(
        2,
        2,
        proposal_ids=[2],
        proposal_corners=[boxes[2]],
        proposal_scores=[0.7],
        native_corners=empty_native,
    )
    assert len(born) == 1
    assert born[0].evidence_frame_ids == (0, 1, 2)
    frozen = state.rows()[0]["box"].copy()

    state.update(
        3,
        3,
        proposal_ids=[3],
        proposal_corners=[corners(0.25)],
        proposal_scores=[1.0],
        native_corners=empty_native,
    )
    np.testing.assert_array_equal(state.rows()[0]["box"], frozen)


def test_m1p_native_dedup_is_causal_and_can_retire_a_prior_birth():
    state = OnlineProposalRecovery()
    box = corners()
    for frame in range(3):
        state.update(
            frame,
            frame,
            proposal_ids=[frame],
            proposal_corners=[box],
            proposal_scores=[0.8],
        )
    assert len(state.rows()) == 1
    state.update(
        3,
        3,
        proposal_ids=[],
        proposal_corners=np.empty((0, 8, 3)),
        proposal_scores=[],
        native_corners=[box],
    )
    assert state.rows() == []
    assert state.diagnostics()["births_retired_native_overlap"] == 1
    assert len(state.terminal_rows(np.empty((0, 8, 3)))) == 1
    assert state.terminal_rows(np.asarray([box])) == []


def test_m1p_output_budget_is_revisable_and_keeps_stronger_late_birth():
    state = OnlineProposalRecovery(max_births=1)
    empty_native = np.empty((0, 8, 3))
    early = corners(0.0)
    late = corners(10.0)
    for frame in range(3):
        state.update(
            frame,
            frame,
            proposal_ids=[frame],
            proposal_corners=[early],
            proposal_scores=[0.2],
            native_corners=empty_native,
        )
    assert state.rows()[0]["raw_mean_score"] == pytest.approx(0.2)

    for frame in range(3, 6):
        state.update(
            frame,
            frame,
            proposal_ids=[frame],
            proposal_corners=[late],
            proposal_scores=[0.9],
            native_corners=empty_native,
        )
    rows = state.rows()
    assert len(rows) == 1
    assert rows[0]["raw_mean_score"] == pytest.approx(0.9)
    np.testing.assert_array_equal(rows[0]["box"], late)
    assert state.diagnostics()["births_total"] == 2


def test_m1p_native_covered_birth_releases_current_output_slot():
    state = OnlineProposalRecovery(max_births=1)
    early = corners(0.0)
    late = corners(10.0)
    for frame in range(3):
        state.update(
            frame,
            frame,
            proposal_ids=[frame],
            proposal_corners=[early],
            proposal_scores=[0.9],
        )
    for frame in range(3, 6):
        state.update(
            frame,
            frame,
            proposal_ids=[frame],
            proposal_corners=[late],
            proposal_scores=[0.8],
            native_corners=[early],
        )
    rows = state.rows()
    assert len(rows) == 1
    np.testing.assert_array_equal(rows[0]["box"], late)
    assert len(state.terminal_rows(np.empty((0, 8, 3)))) == 1
    np.testing.assert_array_equal(
        state.terminal_rows(np.asarray([early]))[0]["box"], late
    )


def test_m1p_confirmed_track_keeps_updating_its_medoid():
    state = OnlineProposalRecovery()
    boxes = [
        corners(-0.20),
        corners(0.00),
        corners(0.20),
        corners(0.25),
        corners(0.30),
    ]
    for frame, box in enumerate(boxes[:3]):
        state.update(
            frame,
            frame,
            proposal_ids=[frame],
            proposal_corners=[box],
            proposal_scores=[0.8],
        )
    initial = state.rows()[0]["box"].copy()
    for frame, box in enumerate(boxes[3:], start=3):
        state.update(
            frame,
            frame,
            proposal_ids=[frame],
            proposal_corners=[box],
            proposal_scores=[0.8],
        )
    revised = state.rows()[0]
    assert revised["support_frames"] == 5
    assert not np.array_equal(revised["box"], initial)
    assert state.diagnostics()["birth_revisions"] == 2


def test_m1p_can_disable_child_evidence_completely():
    state = OnlineProposalRecovery(use_children=False, child_min_views=1)
    box = corners()
    born = state.update(
        0,
        0,
        proposal_ids=[],
        proposal_corners=np.empty((0, 8, 3)),
        proposal_scores=[],
        child_ids=[1],
        child_corners=[box],
        child_scores=[0.9],
    )
    assert born == []
    assert state.rows() == []
    assert state.diagnostics()["use_children"] is False


def test_same_frame_proposals_do_not_self_confirm_and_state_is_bounded():
    state = OnlineProposalRecovery(
        max_tracks=2, max_observations_per_frame=3, ttl_keyframes=0
    )
    state.update(
        0,
        0,
        proposal_ids=[0, 1, 2, 3],
        proposal_corners=[
            corners(0.00),
            corners(0.01),
            corners(10.0),
            corners(20.0),
        ],
        proposal_scores=[0.9, 0.8, 0.7, 0.6],
    )
    assert state.rows() == []
    assert len(state.tracks) <= 2
    assert state.diagnostics()["capacity_drops"] >= 1


def test_m2_running_max_changes_scores_only_and_never_needs_past_cache():
    state = OnlineNativeReranker()
    box = corners()
    current = native([7], [box], [0.2])
    first = state.update(0, 0, current, [projected_box(box)])
    assert first[0] > 0.2
    assert state.last_support[0] == pytest.approx(1.0)
    second = state.update(1, 1, current, np.empty((0, 4)))
    assert second[0] == pytest.approx(first[0])
    assert state.last_support[0] == pytest.approx(1.0)
    np.testing.assert_array_equal(current.corners, np.asarray([box]))
    assert len(second) == len(current.ids)
    final_scores, final_support = state.materialize([7], [0.2])
    assert final_scores[0] == pytest.approx(first[0])
    assert final_support[0] == pytest.approx(1.0)


def test_m2_exclusive_matching_assigns_one_proposal_to_one_native_row():
    state = OnlineNativeReranker(exclusive_matching=True)
    box = corners()
    current = native([1, 2], [box, box], [0.2, 0.2])
    state.update(0, 0, current, [projected_box(box)])
    assert sorted(state.last_support.tolist()) == pytest.approx([0.0, 1.0])


def test_composed_map_is_prefix_invariant_and_emits_online_m1a_m1p_m2():
    short = OnlineCandidateMap("scene-test")
    long = OnlineCandidateMap("scene-test")
    native_box = corners(3.0)
    current_native = native([11], [native_box], [0.2])
    short_snapshot = None
    long_prefix = None
    for frame in range(3):
        ev = evidence(frame, anchors=True)
        short_snapshot = short.update(frame, current_native, ev)
        long_prefix = long.update(frame, current_native, ev)
    assert short_snapshot is not None and long_prefix is not None
    np.testing.assert_array_equal(short_snapshot.boxes, long_prefix.boxes)
    np.testing.assert_array_equal(short_snapshot.scores, long_prefix.scores)
    assert short_snapshot.sources.count("native") == 1
    assert short_snapshot.sources.count("m1p") == 1
    assert short_snapshot.sources.count("m1a") == 1
    frozen_prefix = long_prefix.boxes.copy()

    for frame in range(3, 8):
        long.update(frame, current_native, evidence(frame, anchors=True))
    np.testing.assert_array_equal(long_prefix.boxes, frozen_prefix)
    diagnostics = long.diagnostics()
    assert diagnostics["strictly_causal"]
    assert diagnostics["online_incremental"]
    assert diagnostics["uses_terminal_map"] is False
    assert diagnostics["uses_full_scene_cache"] is False

    terminal = long.materialize_terminal(
        7,
        np.asarray([11]),
        np.asarray([native_box]),
        np.asarray([0.2]),
    )
    terminal_diagnostics = long.diagnostics()
    assert terminal_diagnostics["uses_terminal_map"] is True
    assert terminal_diagnostics["uses_terminal_map_for_inference"] is False
    assert terminal_diagnostics["terminal_counts"] == {
        "native": 1,
        "m1p": 1,
        "m1a": 1,
        "total": 3,
    }
    assert terminal.sources.count("native") == 1
    assert terminal.sources.count("m1p") == 1
    assert terminal.sources.count("m1a") == 1
    assert long.diagnostics()["terminal_native_map_readout"] is True


def test_repeated_or_future_reordered_frames_are_rejected():
    state = OnlineCandidateMap("scene-test")
    state.update(10, native(), evidence(10))
    with pytest.raises(ValueError, match="increase strictly"):
        state.update(10, native(), evidence(10))
