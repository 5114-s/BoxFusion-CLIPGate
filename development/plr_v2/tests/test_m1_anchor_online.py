import numpy as np

from boxfusion.m1_anchor_online import OnlineAnchorRecovery, deterministic_tail_score


def cube(center=(0.0, 0.0, 0.0), side=0.2):
    center = np.asarray(center, dtype=float)
    signs = np.asarray([[x, y, z] for x in (-1, 1)
                        for y in (-1, 1) for z in (-1, 1)], dtype=float)
    return center + signs * side / 2


def test_birth_is_causal_on_third_distinct_frame_and_score_is_ranked():
    tracker = OnlineAnchorRecovery('scene0000_00', max_active=8, max_births=4)
    box = cube()
    assert tracker.update(0, 0, [7], [box], [.2]) == []
    # A second observation in the same frame cannot increase view support.
    assert tracker.update(1, 0, [8], [box], [.3]) == []
    assert tracker.update(2, 25, [9], [box], [.25]) == []
    events = tracker.update(3, 50, [10], [box], [.1])
    assert len(events) == 1
    row = tracker.rows()[0]
    assert row['support_frames'] == 3
    assert row['raw_score'] == .3
    assert 0.040001 <= row['score'] < .05
    assert tracker.diagnostics()['frames_seen'] == 4


def test_fixed_caps_and_deterministic_tie_break():
    tracker = OnlineAnchorRecovery('s', min_views=4, max_active=2, max_births=1)
    boxes = np.asarray([cube((i, 0, 0)) for i in range(3)])
    tracker.update(0, 0, [0, 1, 2], boxes, [.1, .2, .3])
    assert tracker.diagnostics()['active'] == 2
    assert tracker.diagnostics()['dropped_active'] == 1
    a = deterministic_tail_score(.2, 's', 1, 2)
    b = deterministic_tail_score(.2, 's', 1, 3)
    c = deterministic_tail_score(.3, 's', 1, 2)
    assert a != b
    assert c > max(a, b)
