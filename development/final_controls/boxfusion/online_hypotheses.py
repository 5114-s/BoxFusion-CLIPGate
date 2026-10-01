"""Bounded, training-free geometry hypotheses for an existing static track.

The caller supplies causal, GT-free associations and the native geometry. A new
view scores OLD hypotheses against the PREVIOUS native box before any fitting.
Only then are hypotheses updated. This is a prequential trajectory selector,
not an independently validated score for the exact post-update geometry.
No detection scores, identities, births, or association decisions are changed.
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np


SIGNS = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                  [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])


def corners(box, rotation):
    box = np.asarray(box, dtype=float)
    return (SIGNS * box[3:] * 0.5) @ np.asarray(rotation).T + box[:3]


def overlap(a, b):
    lo_a, hi_a = a.min(0), a.max(0)
    lo_b, hi_b = b.min(0), b.max(0)
    inter = np.maximum(np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b), 0).prod()
    return float(inter / max((hi_a-lo_a).prod() + (hi_b-lo_b).prod() - inter, 1e-12))


@dataclass(frozen=True)
class Settings:
    # Total geometry slots = native fallback + alternatives.
    slots: int = 3
    window: int = 3
    min_checks: int = 2
    gain_margin: float = 0.02
    distinct_iou: float = 0.90
    fit_steps: int = 2
    fit: bool = True

    def __post_init__(self):
        if not (2 <= self.slots <= 8 and 1 <= self.min_checks <= self.window <= 8):
            raise ValueError("Invalid bounded hypothesis configuration")
        if not (0 <= self.gain_margin <= 1 and 0 < self.distinct_iou < 1):
            raise ValueError("Invalid thresholds")
        if not 0 <= self.fit_steps <= 4:
            raise ValueError("Invalid fit budget")


@dataclass
class View:
    frame: int
    world_to_camera: np.ndarray
    K: np.ndarray
    box2d: np.ndarray
    height: int
    width: int

    @classmethod
    def make(cls, frame, camera_to_world, K, box2d, height, width):
        pose, K, rect = (np.asarray(x, dtype=float).copy()
                         for x in (camera_to_world, K, box2d))
        if (pose.shape != (4, 4) or K.shape not in ((3, 3), (4, 4))
                or rect.shape != (4,) or not all(np.isfinite(x).all() for x in (pose, K, rect))
                or np.any(rect[2:] <= rect[:2]) or height <= 0 or width <= 0):
            raise ValueError("Invalid observation")
        return cls(int(frame), np.linalg.inv(pose), K[:3, :3], rect, int(height), int(width))


def projection_quality(boxes, rotation, view):
    """2D IoU with the independent detector rectangle; invalid projections get 0."""
    boxes = np.atleast_2d(boxes)
    xyz = ((SIGNS[None] * boxes[:, None, 3:] * .5) @ rotation.T
           + boxes[:, None, :3])
    cam = xyz @ view.world_to_camera[:3, :3].T + view.world_to_camera[:3, 3]
    valid = (cam[..., 2] > .05).all(1)
    uvw = cam @ view.K.T
    uv = uvw[..., :2] / np.maximum(uvw[..., 2:], 1e-8)
    low = np.maximum(uv.min(1), 0)
    high = np.minimum(uv.max(1), [view.width, view.height])
    target_low = np.maximum(view.box2d[:2], 0)
    target_high = np.minimum(view.box2d[2:], [view.width, view.height])
    area = np.maximum(high-low, 0).prod(1)
    target_area = np.maximum(target_high-target_low, 0).prod()
    inter = np.maximum(np.minimum(high, target_high)-np.maximum(low, target_low), 0).prod(1)
    return np.where(valid & (area > 1) & (target_area > 1),
                    inter / np.maximum(area+target_area-inter, 1e-12), 0)


@dataclass
class Hypothesis:
    box: np.ndarray
    rotation: np.ndarray
    seed_box: np.ndarray
    seed_score: float
    source_id: int
    updated_frame: int
    gains: deque = field(default_factory=deque)

    def merit(self):
        return float(np.mean(self.gains)) if self.gains else 0.0


class OnlineHypotheses:
    """One instance per native track. Memory and work are bounded per update."""
    def __init__(self, settings=Settings()):
        self.cfg = settings
        self.hypotheses = []
        self.views = deque(maxlen=settings.window)
        self.native = None
        self.native_score = 0.
        self.frame = -1
        self.chosen_source = None
        self.stats = dict(checks=0, updates=0, births=0, selections=0,
                          reversals=0, fitted_hypotheses=0)

    @staticmethod
    def validate_geometry(box, rotation):
        box, rotation = np.asarray(box, dtype=float), np.asarray(rotation, dtype=float)
        if (box.shape != (6,) or rotation.shape != (3, 3)
                or not np.isfinite(box).all() or not np.isfinite(rotation).all()
                or np.any(box[3:] <= 0)):
            raise ValueError("Invalid box geometry")
        return box.copy(), rotation.copy()

    def _fit(self, hypothesis):
        box = hypothesis.box.copy()
        anchor = hypothesis.seed_box
        radius = max(np.linalg.norm(anchor[3:]), .1)
        lower = np.r_[anchor[:3] - .25*radius, np.maximum(.5*anchor[3:], .02)]
        upper = np.r_[anchor[:3] + .25*radius, np.maximum(1.5*anchor[3:], .02)]
        step = np.r_[np.full(3, min(.025*radius, .05)), .05*anchor[3:]]
        for _ in range(self.cfg.fit_steps):
            proposals = np.vstack([box, box + np.diag(step), box - np.diag(step)])
            proposals = np.clip(proposals, lower, upper)
            objective = np.mean([projection_quality(proposals, hypothesis.rotation, v)
                                 for v in self.views], axis=0)
            box = proposals[int(np.argmax(objective))].copy()
        hypothesis.box = box
        self.stats['fitted_hypotheses'] += 1

    def _admit(self, box, rotation, score, source, frame, native):
        candidate_corners = corners(box, rotation)
        existing = [corners(*native)] + [corners(h.box, h.rotation) for h in self.hypotheses]
        if any(overlap(candidate_corners, x) >= self.cfg.distinct_iou for x in existing):
            return
        h = Hypothesis(box.copy(), rotation.copy(), box.copy(), score, source, frame,
                       deque(maxlen=self.cfg.window))
        if len(self.hypotheses) < self.cfg.slots-1:
            self.hypotheses.append(h)
            self.stats['births'] += 1
        else:
            eligible = [j for j, old in enumerate(self.hypotheses)
                        if len(old.gains) >= self.cfg.min_checks]
            if eligible:
                worst = min(eligible, key=lambda j: (self.hypotheses[j].merit(),
                                                    self.hypotheses[j].seed_score))
                old = self.hypotheses[worst]
                if old.merit() <= 0 and score >= old.seed_score:
                    self.hypotheses[worst] = h
                    self.stats['births'] += 1

    def advance(self, frame, native_box, native_rotation, native_score,
                *, view=None, candidate=None):
        """candidate=(box6, rotation, detector_score, raw_source_id).

        A view must belong to THIS frame. Late-associated past observations are
        never retrospectively used as held-out evidence. Repeated frames fail.
        The current native box is supplied AFTER its native fusion update.
        """
        frame = int(frame)
        if frame <= self.frame:
            raise ValueError("Frames must increase strictly")
        native = self.validate_geometry(native_box, native_rotation)
        if not np.isfinite(native_score):
            raise ValueError("Invalid score")
        if view is not None and view.frame != frame:
            raise ValueError("View is not current; future/past evidence rejected")
        prepared = None
        if candidate is not None:
            box, rotation = self.validate_geometry(candidate[0], candidate[1])
            if view is None or not np.isfinite(candidate[2]):
                raise ValueError("Candidate requires a valid current observation")
            prepared = (box, rotation, float(candidate[2]), int(candidate[3]))

        # Preserve the previous native interpretation before it is overwritten.
        # It existed at the previous step, so the current view can test it.
        if self.native is not None:
            self._admit(*self.native, self.native_score, -(self.frame+1), self.frame, native)
        if view is not None:
            if self.native is not None:
                baseline_q = float(projection_quality(self.native[0], self.native[1], view)[0])
                for h in self.hypotheses:
                    assert h.updated_frame < frame
                    gain = float(projection_quality(h.box, h.rotation, view)[0]) - baseline_q
                    h.gains.append(gain)
                    self.stats['checks'] += 1
            self.views.append(view)
            for h in self.hypotheses:
                if self.cfg.fit:
                    self._fit(h)
                h.updated_frame = frame

        # Admission is after checking/fitting: a seed cannot validate itself.
        if prepared is not None:
            box, rotation, score, source = prepared
            self._admit(box, rotation, score, source, frame, native)
        self.native, self.native_score, self.frame = native, float(native_score), frame
        self.stats['updates'] += 1
        winner = self.select('rank')
        source = winner.source_id if winner is not None else None
        if source != self.chosen_source and self.chosen_source is not None:
            self.stats['reversals'] += 1
        self.chosen_source = source
        self.stats['selections'] += int(source is not None)
        assert len(self.hypotheses) < self.cfg.slots
        return self.output('rank')

    def select(self, rule='rank'):
        if rule == 'score':
            candidates = [h for h in self.hypotheses if h.seed_score > self.native_score]
            return max(candidates, key=lambda h: h.seed_score, default=None)
        if rule != 'rank':
            raise ValueError("Unknown selector")
        candidates = [h for h in self.hypotheses
                      if len(h.gains) >= self.cfg.min_checks
                      and h.merit() > self.cfg.gain_margin and h.gains[-1] > 0]
        return max(candidates, key=lambda h: (h.merit(), h.seed_score), default=None)

    def output(self, rule='rank'):
        h = self.select(rule)
        pair = (h.box, h.rotation) if h is not None else self.native
        return tuple(x.copy() for x in pair)
