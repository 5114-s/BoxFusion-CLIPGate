import numpy as np

from boxfusion.plr_v2 import OnlineProposalRecoveryV2


SIGNS = np.asarray(
    [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
    dtype=np.float64,
)


def box(x, extent=1.0):
    return SIGNS * (0.5 * extent) + np.asarray([x, 0.0, 5.0])


def pose(x):
    value = np.eye(4)
    value[0, 3] = x
    return value


def update(state, ordinal, boxes, scores, camera_x=0.0):
    state.update(
        ordinal,
        ordinal * 25,
        proposal_ids=np.arange(len(boxes)) + ordinal * 100,
        proposal_corners=np.asarray(boxes),
        proposal_scores=np.asarray(scores),
        native_corners=np.empty((0, 8, 3)),
        camera_to_world=pose(camera_x),
    )


def test_assoc_is_query_before_commit_and_one_to_one():
    state = OnlineProposalRecoveryV2(stage="assoc", proposal_min_views=3)
    update(state, 0, [box(0.0)], [0.9])
    update(state, 1, [box(0.0), box(0.1)], [0.9, 0.8])
    assert len(state.tracks) == 2
    assert sorted(len(track.observations) for track in state.tracks.values()) == [1, 2]
    assert state.association_matches == 1


def test_reliability_stage_ranks_stable_high_quality_track_first():
    state = OnlineProposalRecoveryV2(
        stage="reliability", proposal_min_views=3, max_births=2
    )
    for ordinal, camera_x in enumerate((-3.0, 0.0, 3.0)):
        update(
            state,
            ordinal,
            [box(0.0), box(4.0)],
            [0.9, 0.5],
            camera_x,
        )
    rows = state.rows()
    assert len(rows) == 2
    assert rows[0]["reliability"] > rows[1]["reliability"]
    assert rows[0]["raw_mean_score"] == 0.9


def test_score_stage_uses_low_tail_reliability_rank_not_size_price():
    state = OnlineProposalRecoveryV2(stage="score", proposal_min_views=3, max_births=2)
    for ordinal, camera_x in enumerate((-3.0, 0.0, 3.0)):
        update(
            state,
            ordinal,
            [box(0.0, 0.4), box(4.0, 1.2)],
            [0.9, 0.5],
            camera_x,
        )
    rows = state.rows()
    assert len(rows) == 2
    assert rows[0]["score"] > rows[1]["score"]
    assert all(0.05 < row["score"] < 0.50 for row in rows)
    assert rows[0]["score"] != 0.10
    assert rows[1]["score"] != 0.50

