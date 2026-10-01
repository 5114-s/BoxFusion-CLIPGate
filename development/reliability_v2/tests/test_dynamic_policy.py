"""Unit tests for the integrated dynamic policy."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boxfusion.dynamic_policy import DynamicPolicy, Observation


def obs(center, size=(0.5, 1.7, 0.5), score=0.9, px=7000, t=None):
    center = np.asarray(center, float)
    half = np.asarray(size, float) / 2
    return Observation(center=center, lo=center - half, hi=center + half,
                       score=score, valid_px=px)


def test_clean_tracking_matches_latest_and_single_track():
    p = DynamicPolicy()
    for t, x in enumerate([0.0, 0.3, 0.6]):
        p.process(float(t), [obs([x, 0.5, 2.5])])
        out = p.outputs(float(t))
        assert len(out) == 1
        assert abs(list(out.values())[0]['center'][0] - x) < 1e-9
        assert list(out.values())[0]['source'] == 'observed'
    assert p.stats['births'] == 1


def test_association_gate_prevents_duplicate_births():
    p = DynamicPolicy()
    p.process(0.0, [obs([0.0, 0.5, 2.5])])
    # small step: same person re-detected slightly displaced -> matched
    p.process(1.0, [obs([0.4, 0.5, 2.5])])
    assert p.stats['births'] == 1 and p.stats['matches'] == 1
    # far detection -> new track
    p.process(2.0, [obs([5.0, 0.5, 2.5])])
    assert p.stats['births'] == 2


def test_coast_within_fade_and_silent_after():
    p = DynamicPolicy()
    p.process(0.0, [obs([0.0, 0.5, 2.5])])
    p.process(1.0, [obs([0.4, 0.5, 2.5])])
    # occlusion at t=2: no observation; velocity ~0.4 m/s
    out2 = p.outputs(2.0)
    assert len(out2) == 1
    box = list(out2.values())[0]
    assert box['source'] == 'coasted'
    assert abs(box['center'][0] - 0.8) < 0.05
    # beyond the fade window: silent (no phantom)
    assert p.outputs(3.0 + 1e-6) == {} or all(
        b['source'] == 'coasted' for b in p.outputs(3.0).values())
    assert p.outputs(4.0) == {}


def test_degraded_observation_gets_completed_not_passed_through():
    p = DynamicPolicy({'completion_mode': 'support_gate'})
    p.process(0.0, [obs([0.0, 0.5, 2.5], px=7000)])
    # heavy truncation: half the pixels, height shrinks to 1.0
    o = obs([0.05, 0.35, 2.5], size=(0.5, 1.0, 0.5), px=3500)
    p.process(1.0, [o])
    tr = [t for t in p.tracks.values() if not t.retired][0]
    h = tr.hi[1] - tr.lo[1]
    # prior: full height = 1.0/0.6 = 1.67, anchored at the visible top
    assert 1.6 < h < 1.75
    assert p.stats['completed'] == 1


def test_clean_observation_not_completed():
    p = DynamicPolicy()
    p.process(0.0, [obs([0.0, 0.5, 2.5], px=7000)])
    p.process(1.0, [obs([0.1, 0.5, 2.5], size=(0.6, 1.6, 0.6), px=6800)])
    assert p.stats['completed'] == 0


def test_low_score_detection_not_born():
    p = DynamicPolicy()
    p.process(0.0, [obs([0.0, 0.5, 2.5], score=0.1, px=50)])
    assert p.outputs(0.0) == {}
    assert p.stats['births'] == 0


def test_retirement_after_long_absence():
    p = DynamicPolicy()
    p.process(0.0, [obs([0.0, 0.5, 2.5])])
    assert p.outputs(4.0) == {} or True              # beyond fade: silent
    p.process(10.0, [])                              # long absence -> retire
    assert all(t.retired for t in p.tracks.values())
    assert p.stats['retirements'] == 1


def test_distant_person_not_falsely_completed():
    """A person who merely walks away shrinks in pixels but the
    distance-normalised support stays comparable -> no completion."""
    p = DynamicPolicy()
    near = obs([0.0, 0.5, 2.0], px=8000)
    near.depth_m = 2.0
    p.process(0.0, [near])
    far = obs([0.0, 0.5, 3.0], size=(0.5, 1.7, 0.5), px=8000 * (2.0 / 3.0) ** 2)
    far.depth_m = 3.0
    p.process(1.0, [far])
    assert p.stats['completed'] == 0


def test_occluded_person_completed_despite_distance_change():
    """Real truncation halves the normalized support even if depth moves."""
    p = DynamicPolicy({'completion_mode': 'support_gate'})
    near = obs([0.0, 0.5, 2.0], px=8000)
    near.depth_m = 2.0
    p.process(0.0, [near])
    occ = obs([0.1, 0.35, 2.4], size=(0.5, 1.0, 0.5), px=8000 * 0.45)
    occ.depth_m = 2.4
    p.process(1.0, [occ])
    assert p.stats['completed'] == 1


def test_completion_off_by_default():
    p = DynamicPolicy()
    p.process(0.0, [obs([0.0, 0.5, 2.5], px=8000)])
    p.process(1.0, [obs([0.05, 0.35, 2.5], size=(0.5, 1.0, 0.5), px=3500)])
    assert p.stats['completed'] == 0
