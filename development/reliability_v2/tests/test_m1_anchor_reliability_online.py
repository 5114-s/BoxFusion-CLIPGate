import numpy as np

from boxfusion.m1_anchor_reliability_online import OnlineReliableAnchorRecovery


SIGNS = np.asarray(
    [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
    dtype=np.float64,
)


def box(x=0.0, z=4.0, extent=1.0):
    return SIGNS * (extent / 2.0) + np.asarray([x, 0.0, z])


def pose(camera_x=0.0):
    value = np.eye(4)
    value[0, 3] = camera_x
    return value


def update(state, ordinal, corners, camera_x=0.0, score=0.2, anchor_id=None, **kwargs):
    return state.update(
        ordinal,
        ordinal,
        [ordinal if anchor_id is None else anchor_id],
        np.asarray([corners]),
        [score],
        pose(camera_x),
        **kwargs,
    )


def test_three_directionally_distinct_keyframes_confirm_one_track():
    state = OnlineReliableAnchorRecovery("scene", min_angular_separation_deg=20.0)
    for ordinal, camera_x in enumerate((-3.0, 0.0, 3.0)):
        rows = update(state, ordinal, box(0.01 * ordinal), camera_x)
    assert len(rows) == 1
    assert rows[0]["support_views"] == 3
    assert 0.04 < rows[0]["score"] < 0.05
    assert state.diagnostics()["created_tracks"] == 1
    assert state.diagnostics()["associations"] == 2


def test_repeated_near_identical_view_does_not_fake_multiview_confirmation():
    state = OnlineReliableAnchorRecovery("scene", min_angular_separation_deg=30.0)
    for ordinal in range(3):
        rows = update(state, ordinal, box(0.01 * ordinal), camera_x=0.0)
    assert rows == []
    assert state.reliability.summary(0)["positive_views"] == 1


def test_native_shadow_is_reversible_and_does_not_delete_confirmed_track():
    state = OnlineReliableAnchorRecovery("scene", min_angular_separation_deg=20.0)
    target = box()
    for ordinal, camera_x in enumerate((-3.0, 0.0, 3.0)):
        update(state, ordinal, target, camera_x)
    assert len(state.rows()) == 1
    assert state.rows(native_corners=np.asarray([target])) == []
    assert len(state.rows(native_corners=np.empty((0, 8, 3)))) == 1
    assert state.diagnostics()["confirmed_tracks"] == 1


def test_dynamic_output_cap_allows_later_stronger_track_to_replace_weak_one():
    state = OnlineReliableAnchorRecovery(
        "scene", max_births=1, min_angular_separation_deg=20.0
    )
    cameras = (-3.0, 0.0, 3.0)
    for ordinal, camera_x in enumerate(cameras):
        state.update(
            ordinal,
            ordinal,
            [100 + ordinal, 200 + ordinal],
            np.asarray([box(-1.5), box(1.5)]),
            [0.1, 0.9],
            pose(camera_x),
        )
    rows = state.rows()
    assert len(rows) == 1
    assert rows[0]["raw_score"] == 0.9

