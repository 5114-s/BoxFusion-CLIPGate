"""Integrated online dynamic-object policy (training-free, causal).

Assembles the four components validated in the 2026-09-13/14 rounds:

  1. frozen front-end observations (person candidates, provided by caller);
  2. causal constant-velocity tracking for position, association and
     occlusion coasting (validated: walk-bench 0.59->0.20 m, mask-diag);
  3. shape from the CURRENT observation plus a visibility-prior completion
     when the observation is objectively degraded (validated: Behave
     occlusion 0.301->0.424 IoU).  History is NEVER fused into geometry and
     NEVER retrieved as a replacement shape -- both were falsified;
  4. stale-fade lifecycle: outputs stop after ``fade_after_s`` without
     support and the track retires after ``retire_after_s``.

The policy consumes only current+past observations (no GT, no future).
History use is restricted to: association gate, velocity, and the
degradation quality gate (rolling valid-pixel median).  Completion assumes
the missing part is along gravity (bottom hidden by desk/table occlusion or
image crop); the prior fraction must eventually be estimated causally in
production -- here it is a fixed frozen parameter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Observation:
    center: np.ndarray          # (3,) world/camera frame
    lo: np.ndarray              # (3,) observed AABB low
    hi: np.ndarray              # (3,) observed AABB high
    score: float
    valid_px: int
    depth_m: Optional[float] = None   # median depth of the visible points;
    #                                    enables the distance-normalised gate
    trunc_below: bool = False         # occluder/border truncates the LOWER
    trunc_above: bool = False         # resp. UPPER part (front-end signals)
    track_hint: Optional[int] = None

    @property
    def support(self) -> float:
        """Occlusion-sensitive, distance-invariant support measure:
        valid pixels scaled by squared depth (a person seen from farther
        shrinks in pixels but keeps support)."""
        return float(self.valid_px) * (self.depth_m ** 2 if self.depth_m else 1.0)


@dataclass
class Track:
    tid: int
    center: np.ndarray
    lo: np.ndarray
    hi: np.ndarray              # last FULL (completed) extent
    score: float
    last_seen_t: float
    vel: Optional[np.ndarray] = None
    support_history: List[float] = field(default_factory=list)
    obs_count: int = 0
    retired: bool = False


class DynamicPolicy:
    def __init__(self, cfg=None):
        cfg = cfg or {}
        p = cfg.get('dynamic_policy', cfg)
        self.assoc_gate_m = float(p.get('assoc_gate_m', 1.0))
        self.vel_ema = float(p.get('vel_ema', 0.3))
        self.fade_after_s = float(p.get('fade_after_s', 1.5))
        self.retire_after_s = float(p.get('retire_after_s', 5.0))
        self.min_birth_score = float(p.get('min_birth_score', 0.3))
        self.min_birth_px = int(p.get('min_birth_px', 200))
        self.degrade_ratio = float(p.get('degrade_ratio', 0.75))
        # 'off'            - never complete (safe default; mild truncation
        #                   hurts per the real-occlusion probe)
        # 'support_gate'   - complete on low distance-normalised support
        #                   (validated ONLY for the heavy-truncation sim;
        #                   over-fires on side views in crowd)
        # 'truncation'     - complete only on truncation-specific signals
        #                   (occluder depth discontinuity at the mask
        #                   boundary / border crop), with direction
        self.completion_mode = str(p.get('completion_mode', 'off'))
        self.visible_fraction_prior = float(p.get('visible_fraction_prior', 0.6))
        self.height_clamp = tuple(p.get('height_clamp', (0.8, 2.2)))
        self.gravity_axis = int(p.get('gravity_axis', 1))     # camera y
        self.gravity_sign = int(p.get('gravity_sign', 1))     # +1 = down
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1
        self.stats = {'births': 0, 'matches': 0, 'coast_outputs': 0,
                      'retirements': 0, 'completed': 0}

    # ------------------------------------------------------------------ #
    def _complete(self, obs: Observation, ref_support: Optional[float]):
        """Return the full (lo, hi) for an observation.

        Degradation gate: distance-normalised support below ``degrade_ratio``
        x the track's quality reference (recent MAX support from non-degraded
        observations only).  Completion extends the observed extent along
        gravity with the fixed visibility prior; the visible top edge is the
        anchor.  Clean observations pass through unchanged."""
        lo, hi = obs.lo.copy(), obs.hi.copy()
        if self.completion_mode == 'truncation':
            if not (obs.trunc_below or obs.trunc_above):
                return lo, hi, False
        elif (self.completion_mode != 'support_gate'
                or ref_support is None
                or obs.support >= self.degrade_ratio * ref_support):
            return lo, hi, False
        a = self.gravity_axis
        h_v = float(hi[a] - lo[a])
        h_f = float(np.clip(h_v / max(self.visible_fraction_prior, 1e-3),
                            *self.height_clamp))
        extend_down = obs.trunc_below or not obs.trunc_above
        if (self.gravity_sign > 0) == extend_down:
            hi[a] = lo[a] + h_f
        else:
            lo[a] = hi[a] - h_f
        return lo, hi, True

    def _predicted(self, tr: Track, t: float):
        if tr.vel is None:
            return tr.center
        return tr.center + tr.vel * (t - tr.last_seen_t)

    # ------------------------------------------------------------------ #
    def process(self, t: float, observations: List[Observation]):
        """Advance the policy one keyframe. Observations must be causal
        (current frame front-end output only)."""
        # retire stale tracks
        for tr in self.tracks.values():
            if not tr.retired and t - tr.last_seen_t > self.retire_after_s:
                tr.retired = True
                self.stats['retirements'] += 1

        active = [tr for tr in self.tracks.values() if not tr.retired]
        preds = {tr.tid: self._predicted(tr, t) for tr in active}

        # greedy nearest matching inside the association gate (prediction-based)
        pairs = []
        used_obs = set()
        for tr in sorted(active, key=lambda x: x.tid):
            best, best_d = None, self.assoc_gate_m
            for i, obs in enumerate(observations):
                if i in used_obs:
                    continue
                d = float(np.linalg.norm(
                    (obs.center - preds[tr.tid])[:2]))
                if d < best_d:
                    best, best_d = i, d
            if best is not None:
                used_obs.add(best)
                pairs.append((tr, best))

        matched_tids = set()
        for tr, i in pairs:
            obs = observations[i]
            dt = max(t - tr.last_seen_t, 1e-3)
            if tr.vel is None:
                tr.vel = (obs.center - tr.center) / dt
            else:
                tr.vel = (1 - self.vel_ema) * tr.vel + \
                    self.vel_ema * (obs.center - tr.center) / dt
            ref_support = float(max(tr.support_history)) \
                if tr.support_history else None
            lo, hi, completed = self._complete(obs, ref_support)
            # after completion the centre must describe the FULL box
            tr.center = (lo + hi) / 2
            tr.lo, tr.hi = lo, hi
            tr.score = obs.score
            tr.last_seen_t = t
            if not completed:  # degraded frames never contaminate the reference
                tr.support_history.append(obs.support)
                if len(tr.support_history) > 8:
                    tr.support_history.pop(0)
            tr.obs_count += 1
            tr.retired = False
            matched_tids.add(tr.tid)
            self.stats['matches'] += 1
            if completed:
                self.stats['completed'] += 1

        # births
        for i, obs in enumerate(observations):
            if i in used_obs:
                continue
            if obs.score < self.min_birth_score or obs.valid_px < self.min_birth_px:
                continue
            lo, hi, completed = self._complete(obs, None)
            self.tracks[self._next_id] = Track(
                tid=self._next_id, center=(lo + hi) / 2,
                lo=lo, hi=hi, score=obs.score, last_seen_t=t,
                support_history=[obs.support])
            self._next_id += 1
            self.stats['births'] += 1
            if completed:
                self.stats['completed'] += 1

    # ------------------------------------------------------------------ #
    def outputs(self, t: float):
        """Current-frame box outputs (dict tid -> box dict) for tracks still
        within the fade window; coasted boxes use the predicted centre."""
        out = {}
        for tr in self.tracks.values():
            if tr.retired or t - tr.last_seen_t > self.fade_after_s:
                continue
            if t == tr.last_seen_t:
                center, source = tr.center, 'observed'
            else:
                center, source = self._predicted(tr, t), 'coasted'
                self.stats['coast_outputs'] += 1
            half = (tr.hi - tr.lo) / 2
            out[tr.tid] = {
                'center': center.copy(), 'lo': center - half, 'hi': center + half,
                'score': tr.score, 'source': source, 'age_s': t - tr.last_seen_t}
        return out
