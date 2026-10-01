"""Protocol checks: causal masking, shared baselines, missing references."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from run_bonn_fourarm import ARMS, replay, visible_aabb_center
from run_bonn_maskdiag import gap_positions, run_cases, summarize_cases


def fixture():
    kfs = list(range(0, 175, 25))
    ts = {k: i*.83 for i, k in enumerate(kfs)}
    cands = {k: {'center': np.array([2*ts[k], 1., 3.]), 'score': .8} for k in kfs}
    return kfs, ts, cands, {k: v['center'].copy() for k,v in cands.items()}, {k: {'person_sane': True} for k in kfs}


def test_statistic_matches_prediction_quantiles():
    points = np.zeros((101, 3)); points[-1] = 1000.
    center, extent = visible_aabb_center(points)
    np.testing.assert_array_equal(center, np.zeros(3))
    np.testing.assert_array_equal(extent, np.zeros(3))


def test_mask_removes_center_and_score_before_updates():
    kfs, ts, cands, _, _ = fixture()
    expected = replay(kfs, ts, {k:v for k,v in cands.items() if k != 75})
    corrupt = dict(cands); corrupt[75] = {'center': np.full(3, 1e9), 'score': 1e8}
    actual = replay(kfs, ts, corrupt, {75})
    for arm in ARMS:
        assert actual[arm].keys() == expected[arm].keys()
        for k in expected[arm]:
            np.testing.assert_allclose(actual[arm][k], expected[arm][k])
    assert 75 not in actual['A_latest']


def test_future_cannot_change_gap_prediction():
    kfs, ts, cands, _, _ = fixture()
    expected = replay(kfs, ts, cands, {75})
    changed = {k: ({'center': np.full(3, -1e9), 'score': 1e9} if k>75 else v) for k,v in cands.items()}
    actual = replay(kfs, ts, changed, {75})
    for arm in ARMS:
        for k in expected[arm]:
            if k <= 75:
                np.testing.assert_allclose(actual[arm][k], expected[arm][k])


def test_constant_velocity_gap_and_recovery():
    kfs, ts, cands, refs, sanity = fixture()
    assert gap_positions(kfs, cands, refs) == [(75,100), (100,125), (125,150)]
    for row in run_cases(kfs, ts, cands, refs, sanity):
        assert row['gap_err']['A'] is None
        assert row['gap_err']['HOLD'] > 1.
        for arm in ('C','D'):
            assert row['gap_err'][arm] < 1e-10
            assert row['recovery_next'][arm] < 1e-10


def test_missing_recovery_reference_is_not_zero_or_key_error():
    kfs, ts, cands, refs, sanity = fixture()
    del refs[100]
    rows = run_cases(kfs, ts, cands, refs, sanity)
    row = next(r for r in rows if r['mask']==75)
    assert not row['recovery_reference_available']
    assert all(v is None for v in row['recovery_next'].values())
    result = summarize_cases(rows)
    assert result['recovery_reference_missing_or_filtered']==1
    assert result['stop_rule_verdict']=='STOP'


def test_no_positions_is_not_pass():
    assert summarize_cases([])['stop_rule_verdict']=='INSUFFICIENT'
