import numpy as np
import pytest

from boxfusion.online_hypotheses import (
    OnlineHypotheses, Settings, View, corners, projection_quality,
)


ROT = np.eye(3)
BASE = np.array([0., 0., 5., 2., 2., 2.])
ALT = np.array([0., 0., 10., 4., 4., 4.])
K = np.array([[100., 0., 128.], [0., 100., 128.], [0., 0., 1.]])


def view(frame, target=ALT, x=0.):
    pose = np.eye(4)
    pose[0, 3] = x
    xyz = corners(target, ROT) - pose[:3, 3]
    uvw = xyz @ K.T
    uv = uvw[:, :2]/uvw[:, 2:]
    rect = np.r_[uv.min(0), uv.max(0)]
    return View.make(frame, pose, K, rect, 256, 256)


def seed(cfg=Settings(fit=False)):
    bank = OnlineHypotheses(cfg)
    bank.advance(0, BASE, ROT, .9, view=view(0), candidate=(ALT, ROT, .8, 7))
    return bank


def test_projection_and_depth_ambiguity():
    v = view(0)
    assert projection_quality(BASE, ROT, v)[0] == pytest.approx(1.)
    assert projection_quality(ALT, ROT, v)[0] == pytest.approx(1.)
    side = view(1, x=2.)
    assert projection_quality(ALT, ROT, side)[0] == pytest.approx(1.)
    assert projection_quality(BASE, ROT, side)[0] < .7
    behind = BASE.copy()
    behind[2] = -5
    assert projection_quality(behind, ROT, side)[0] == 0


def test_seed_cannot_validate_itself_and_two_future_views_are_required():
    bank = seed()
    assert list(bank.hypotheses[0].gains) == []
    assert bank.select() is None
    bank.advance(1, BASE, ROT, .9, view=view(1, x=2.))
    assert len(bank.hypotheses[0].gains) == 1
    assert bank.select() is None
    bank.advance(2, BASE, ROT, .9, view=view(2, x=2.5))
    assert bank.select().source_id == 7
    np.testing.assert_array_equal(bank.output()[0], ALT)


def test_later_contradiction_reverts_choice_to_native():
    bank = seed()
    for i in (1, 2):
        bank.advance(i, BASE, ROT, .9, view=view(i, x=2.))
    assert bank.select() is not None
    bank.advance(3, BASE, ROT, .9, view=view(3, target=BASE, x=2.))
    assert bank.select() is None
    assert bank.stats['reversals'] == 1
    np.testing.assert_array_equal(bank.output()[0], BASE)


def test_future_or_repeated_view_rejected_without_state_mutation():
    bank = seed()
    with pytest.raises(ValueError):
        bank.advance(1, BASE, ROT, .9, view=view(2))
    assert bank.frame == 0 and len(bank.views) == 1
    with pytest.raises(ValueError):
        bank.advance(0, BASE, ROT, .9, view=view(0))
    assert bank.hypotheses[0].gains == bank.hypotheses[0].gains.__class__()


def test_fitting_is_after_checking_and_each_branch_stays_in_seed_trust_region():
    bank = seed(Settings(fit=True))
    h = bank.hypotheses[0]
    h.box[0] += .5
    old_box = h.box.copy()
    v = view(1, x=2.)
    expected = projection_quality(old_box, ROT, v)[0] - projection_quality(BASE, ROT, v)[0]
    bank.advance(1, BASE, ROT, .9, view=v)
    assert h.gains[-1] == pytest.approx(expected)
    assert not np.array_equal(h.box, old_box)
    assert projection_quality(h.box, ROT, v)[0] >= projection_quality(old_box, ROT, v)[0]
    assert np.all(h.box[3:] >= .5*h.seed_box[3:])
    assert np.all(h.box[3:] <= 1.5*h.seed_box[3:])


def test_prefix_invariance_bounded_memory_and_inputs_unchanged():
    a, b = seed(), seed()
    original = ALT.copy()
    for i in range(1, 4):
        for bank in (a, b):
            bank.advance(i, BASE, ROT, .9, view=view(i, x=2.), candidate=(ALT, ROT, .8, i+10))
    snapshot = a.output()[0]
    np.testing.assert_array_equal(snapshot, b.output()[0])
    for i in range(4, 20):
        b.advance(i, BASE, ROT, .9, view=view(i, x=2.), candidate=(ALT, ROT, .8, i+10))
        assert len(b.hypotheses) <= 2 and len(b.views) <= 3
        assert all(len(h.gains) <= 3 for h in b.hypotheses)
    np.testing.assert_array_equal(a.output()[0], snapshot)
    np.testing.assert_array_equal(ALT, original)


def test_same_pool_score_control_and_returned_arrays_do_not_mutate_state():
    bank = seed()
    for i in (1, 2):
        bank.advance(i, BASE, ROT, .9, view=view(i, x=2.))
    assert bank.select('rank') is not None
    assert bank.select('score') is None
    output = bank.output('rank')
    output[0][:] = 0
    assert np.all(bank.hypotheses[0].box[3:] > 0)


def test_invalid_geometry_and_settings_rejected():
    with pytest.raises(ValueError):
        Settings(slots=1)
    with pytest.raises(ValueError):
        OnlineHypotheses().advance(0, np.zeros(6), ROT, .9)


def test_previous_native_geometry_survives_replacement_and_can_return():
    bank = OnlineHypotheses(Settings(fit=False))
    bank.advance(0, BASE, ROT, .9, view=view(0, target=BASE))
    assert not bank.hypotheses
    for i in (1, 2, 3):
        bank.advance(i, ALT, ROT, .8, view=view(i, target=BASE, x=2.))
    assert bank.select().source_id == -1
    np.testing.assert_array_equal(bank.output()[0], BASE)


def test_representative_id_change_inherits_but_merges_and_splits_reset():
    from tools.test_online_hypotheses_replay import inherit_lineage
    inherited, resets = inherit_lineage([99], [[1, 3, 9]], {1: 42, 3: 42})
    assert inherited == {99: 42} and resets == 0
    assert inherit_lineage([99], [[1, 3]], {1: 42, 3: 43}) == ({}, 1)
    assert inherit_lineage([98, 99], [[1], [3]], {1: 42, 3: 42}) == ({}, 2)
    assert inherit_lineage([98, 99], [[1, 9], [3, 9]], {1: 42, 3: 43}) == ({}, 2)
