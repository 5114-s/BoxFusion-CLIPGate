"""Causal, bounded state for M1 anchor-level tail recovery.

The state consumes one keyframe at a time.  A voxel becomes an output row only
after observations from ``min_views`` distinct past/current frames.  No future
frame, scene length, final map, or GT is required.  Active unconfirmed voxels
and emitted rows both have explicit caps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import struct
from typing import Dict, Iterable, Tuple

import numpy as np


Voxel = Tuple[int, int, int]


def deterministic_tail_score(raw_score: float, scene_id: str, frame_id: int,
                             anchor_id: int, low: float = 0.040001,
                             high: float = 0.049999) -> float:
    """Map a positive float32 score and stable identity below the 0.05 prefix.

    Positive IEEE-754 float32 bit patterns are monotone in value.  The upper
    bits therefore preserve the raw-score ordering; a deterministic 20-bit
    identity digest breaks exact raw-score ties.  The frozen full-dataset audit
    additionally checks that the resulting float64 values are unique.
    """
    value = np.float32(raw_score)
    if not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f'raw_score must be finite in [0,1], got {raw_score!r}')
    bits = struct.unpack('>I', struct.pack('>f', float(value)))[0]
    identity = f'{scene_id}\0{int(frame_id)}\0{int(anchor_id)}'.encode('utf-8')
    tie = int.from_bytes(hashlib.blake2s(identity, digest_size=4).digest(), 'big') \
        & ((1 << 20) - 1)
    composite = (bits << 20) | tie
    maximum = (0x3F800000 << 20) | ((1 << 20) - 1)
    score = low + (high - low) * (float(composite) / float(maximum))
    if not low <= score <= high or not score < 0.05:
        raise AssertionError((raw_score, score))
    return float(score)


@dataclass
class _Candidate:
    frames: set = field(default_factory=set)
    observations: int = 0
    last_ordinal: int = -1
    rank: tuple | None = None
    box: np.ndarray | None = None
    raw_score: float = 0.0
    frame_id: int = -1
    anchor_id: int = -1

    def update(self, ordinal: int, frame_id: int, anchor_id: int,
               box: np.ndarray, score: float) -> bool:
        self.frames.add(int(frame_id))
        self.observations += 1
        self.last_ordinal = int(ordinal)
        rank = (-float(score), int(frame_id), int(anchor_id))
        changed = self.rank is None or rank < self.rank
        if changed:
            self.rank = rank
            self.box = np.asarray(box, dtype=np.float64).copy()
            self.raw_score = float(score)
            self.frame_id = int(frame_id)
            self.anchor_id = int(anchor_id)
        return changed


class OnlineAnchorRecovery:
    """Bounded causal voxel confirmation and birth state."""

    def __init__(self, scene_id: str, voxel_m: float = 0.3,
                 min_views: int = 3, max_active: int = 4096,
                 max_births: int = 640):
        if voxel_m <= 0 or min_views < 1 or max_active < 1 or max_births < 1:
            raise ValueError('invalid online anchor-recovery bounds')
        self.scene_id = str(scene_id)
        self.voxel_m = float(voxel_m)
        self.min_views = int(min_views)
        self.max_active = int(max_active)
        self.max_births = int(max_births)
        self.active: Dict[Voxel, _Candidate] = {}
        self.births: Dict[Voxel, _Candidate] = {}
        self.events = []
        self.frames_seen = 0
        self.peak_active = 0
        self.peak_births = 0
        self.dropped_active = 0
        self.dropped_births = 0
        self.revisions = 0

    def _key(self, box: np.ndarray) -> Voxel:
        center = np.asarray(box, dtype=np.float64).mean(0)
        return tuple(np.floor(center / self.voxel_m).astype(np.int64).tolist())

    @staticmethod
    def _keep_priority(item) -> tuple:
        key, state = item
        # More distinct support, more recent evidence, then stronger exemplar.
        return (len(state.frames), state.last_ordinal,
                -state.rank[0], tuple(-v for v in key))

    def _enforce_active_cap(self) -> None:
        overflow = len(self.active) - self.max_active
        if overflow <= 0:
            return
        ordered = sorted(self.active.items(), key=self._keep_priority)
        for key, _ in ordered[:overflow]:
            del self.active[key]
        self.dropped_active += overflow

    def update(self, ordinal: int, frame_id: int, anchor_ids: Iterable[int],
               corners: np.ndarray, scores: Iterable[float]) -> list[dict]:
        """Consume exactly one keyframe and return births created now."""
        if ordinal != self.frames_seen:
            raise ValueError(f'expected ordinal {self.frames_seen}, got {ordinal}')
        ids = np.asarray(anchor_ids, dtype=np.int64)
        boxes = np.asarray(corners, dtype=np.float64).reshape(-1, 8, 3)
        values = np.asarray(scores, dtype=np.float64)
        if not (len(ids) == len(boxes) == len(values)):
            raise ValueError('anchor_ids, corners and scores must have equal length')
        if len(boxes) and (not np.isfinite(boxes).all()
                           or (np.ptp(boxes, axis=1) <= 0).any()):
            raise ValueError('invalid lifted 3D boxes')
        born_now = []
        # A frame contributes at most once to support, while all anchors may
        # improve the exemplar.  Stable input order is the last tie-break.
        for anchor, box, score in zip(ids, boxes, values):
            key = self._key(box)
            if key in self.births:
                state = self.births[key]
                if state.update(ordinal, frame_id, int(anchor), box, float(score)):
                    self.revisions += 1
                continue
            state = self.active.setdefault(key, _Candidate())
            state.update(ordinal, frame_id, int(anchor), box, float(score))
            if len(state.frames) >= self.min_views:
                if len(self.births) >= self.max_births:
                    self.dropped_births += 1
                    del self.active[key]
                    continue
                self.births[key] = state
                del self.active[key]
                event = {'event': 'birth', 'ordinal': int(ordinal),
                         'frame_id': int(frame_id), 'voxel_key': list(key),
                         'support_frames': len(state.frames),
                         'score': deterministic_tail_score(
                             state.raw_score, self.scene_id,
                             state.frame_id, state.anchor_id),
                         'raw_score': float(state.raw_score),
                         'exemplar_frame_id': int(state.frame_id),
                         'exemplar_anchor_id': int(state.anchor_id),
                         'box': state.box.tolist()}
                self.events.append(event)
                born_now.append(event)
        self._enforce_active_cap()
        self.frames_seen += 1
        self.peak_active = max(self.peak_active, len(self.active))
        self.peak_births = max(self.peak_births, len(self.births))
        return born_now

    def rows(self) -> list[dict]:
        """Return current outputs with deterministic, non-flat confidence."""
        rows = []
        for key, state in self.births.items():
            rows.append({
                'voxel_key': key,
                'box': state.box.copy(),
                'raw_score': state.raw_score,
                'frame_id': state.frame_id,
                'anchor_id': state.anchor_id,
                'score': deterministic_tail_score(
                    state.raw_score, self.scene_id,
                    state.frame_id, state.anchor_id),
                'support_frames': len(state.frames),
                'observations': state.observations,
            })
        rows.sort(key=lambda r: (-r['score'], r['voxel_key']))
        return rows

    def diagnostics(self) -> dict:
        return {
            'frames_seen': self.frames_seen,
            'active': len(self.active),
            'births': len(self.births),
            'peak_active': self.peak_active,
            'peak_births': self.peak_births,
            'max_active': self.max_active,
            'max_births': self.max_births,
            'dropped_active': self.dropped_active,
            'dropped_births': self.dropped_births,
            'revisions': self.revisions,
        }
