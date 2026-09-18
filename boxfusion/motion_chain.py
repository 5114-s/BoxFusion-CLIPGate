"""Causal motion-chain module for the synthetic walking benchmark.

Online, training-free, geometry-only.  It never touches native association,
PFO fusion, or scores while the stream runs; it only records per-row
observation history and builds successor chains between native rows
(row born at the position where an older, now-unsupported row's object should
have moved to).  At terminal time it can
  a) drop rows superseded by a confirmed motion chain (stale trail), and
  b) replace rows whose own observations span a horizontal drift by the
     geometry of their latest observation (motion compensation at output time).

Evidence rules (frozen before running either arm):
- link gates: horizontal centre distance in (min_link, max_link); sorted-extent
  size ratio within (size_ratio_min, 1/size_ratio_min);
- velocity consistency for chains with >= 2 links;
- a chain legitimises dropping only if it has >= 2 direction-consistent links,
  or a single link with a confirmed tip (>= tip_confirm_obs observations) and
  a short hop;
- a superseded row that receives new observations is reactivated and its
  chain edge is cancelled.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _horizontal(a, b):
    d = np.asarray(a) - np.asarray(b)
    d[1] = 0.0
    return float(np.linalg.norm(d))


class MotionChain:
    def __init__(self, cfg, scene_id=None):
        mc = cfg.get('motion_chain', {}) or {}
        self.enabled = bool(mc.get('enabled', False))
        self.dump_only = bool(mc.get('dump_only', False))
        dump_dir = mc.get('dump_dir')
        self.dump_path = (str(Path(dump_dir) / f'{scene_id}.rows.jsonl')
                          if (self.dump_only and dump_dir and scene_id)
                          else mc.get('dump_path'))
        self.min_link = float(mc.get('min_link_dist_m', 0.15))
        self.max_link = float(mc.get('max_link_dist_m', 1.20))
        self.size_ratio_min = float(mc.get('size_ratio_min', 0.55))
        self.velocity_gate = float(mc.get('velocity_gate_m', 0.45))
        self.drift_span = float(mc.get('drift_span_m', 0.35))
        self.tip_confirm_obs = int(mc.get('tip_confirm_obs', 2))
        self.short_hop = float(mc.get('short_hop_m', 0.80))
        # key = first observation init_id of the native row (stable under
        # update(keep_idx) because the survivor keeps its own first obs)
        self.rows = {}
        self.superseded_by = {}
        self.predecessor_of = {}
        # per-row full observation history [(frame, center, sorted size,
        # score)], keyed by the row's first-observation init_id
        self.history = {}
        self.stats = {
            'links': 0, 'drops': 0, 'drift_replacements': 0,
            'reactivations': 0, 'keyframes': 0,
        }

    # ------------------------------------------------------------------ #
    def process_keyframe(self, per_frame_ins, fusion_list, kf):
        """Record row states after native association of keyframe ``kf``."""
        if not self.enabled:
            return
        self.stats['keyframes'] += 1
        centers = per_frame_ins.pred_boxes_3d.tensor.detach().cpu().numpy()[:, :3]
        sizes = per_frame_ins.pred_boxes_3d.tensor.detach().cpu().numpy()[:, 3:]
        corners = per_frame_ins.pred_boxes_3d.corners.detach().cpu().numpy()
        frames = per_frame_ins.frame_id.detach().cpu().numpy().astype(int)
        init_ids = per_frame_ins.init_id.detach().cpu().numpy().astype(int)
        obs_scores = per_frame_ins.scores.detach().cpu().numpy()
        obs_cats = getattr(per_frame_ins, 'categories', None)
        if obs_cats is not None and len(obs_cats) != len(obs_scores):
            obs_cats = None

        live = {}
        for row_ids in fusion_list:
            if not row_ids:
                continue
            ids = np.asarray(sorted(set(int(i) for i in row_ids)), dtype=int)
            obs_frames = frames[ids]
            latest = int(ids[np.argmax(obs_frames)])
            key = int(init_ids[ids[np.argmin(obs_frames)]])
            obs_c = centers[ids]
            live[key] = {
                'center': centers[latest],
                'size': np.sort(sizes[latest]),
                'kf_last': int(obs_frames.max()),
                'kf_born': int(obs_frames.min()),
                'n_obs': int(len(ids)),
                'latest_obs_id': latest,
                'latest_corners': corners[latest],
                'drift': float(max(
                    (_horizontal(c, centers[latest]) for c in obs_c),
                    default=0.0)),
            }
            hist = self.history.setdefault(key, [])
            seen = {h[0] for h in hist}
            for oid in ids:
                fr = int(frames[oid])
                if fr not in seen:
                    cat = (str(obs_cats[oid]) if obs_cats is not None else '')
                    hist.append((fr, centers[oid].tolist(),
                                 np.sort(sizes[oid]).tolist(),
                                 float(obs_scores[oid]), cat))
                    seen.add(fr)
        # reactivation: a superseded row observed again cancels its chain edge
        for key, state in live.items():
            if key in self.superseded_by and state['kf_last'] == kf:
                del self.superseded_by[key]
                self.stats['reactivations'] += 1
        self.rows = live

        born_now = [k for k, s in live.items()
                    if s['kf_born'] == kf and s['n_obs'] == 1]
        for b in born_now:
            best = None
            for a, sa in live.items():
                if a == b or sa['kf_last'] >= kf:
                    continue
                dist = _horizontal(live[b]['center'], sa['center'])
                if not (self.min_link < dist <= self.max_link):
                    continue
                ratio = np.sort(sa['size']) / live[b]['size']
                if (ratio < self.size_ratio_min).any() or (ratio > 1 / self.size_ratio_min).any():
                    continue
                gate = self.velocity_gate
                p = self.predecessor_of.get(a)
                if p is not None and p in live:
                    dt = max(sa['kf_last'] - live[p]['kf_last'], 1)
                    v = (np.asarray(sa['center'])[:2] - np.asarray(live[p]['center'])[:2]) / dt
                    dtb = max(kf - sa['kf_last'], 1)
                    pred = np.asarray(sa['center'])[:2] + v * dtb
                    err = float(np.linalg.norm(pred - live[b]['center'][:2]))
                    if err > gate:
                        continue
                if best is None or dist < best[0]:
                    best = (dist, a)
            if best is not None:
                self.superseded_by[best[1]] = b
                self.predecessor_of[b] = best[1]
                self.stats['links'] += 1

    # ------------------------------------------------------------------ #
    def _chain_drop_set(self):
        """Rows whose stale trail is explained by a confirmed chain."""
        drop = set()
        for a, b in self.superseded_by.items():
            if a not in self.rows or b not in self.rows:
                continue
            chain = [a, b]
            while chain[-1] in self.superseded_by and self.superseded_by[chain[-1]] in self.rows:
                chain.append(self.superseded_by[chain[-1]])
            tip = chain[-1]
            links = len(chain) - 1
            if links >= 2:
                consistent = True
                steps = [(self.rows[chain[i + 1]]['kf_last'] - self.rows[chain[i]]['kf_last'],
                          self.rows[chain[i + 1]]['center'][:2] - self.rows[chain[i]]['center'][:2])
                         for i in range(links)]
                speeds = [np.linalg.norm(s) / max(t, 1) for t, s in steps]
                if min(speeds) > 0 and max(speeds) > 3 * min(speeds) + 1e-9:
                    consistent = False
                dirs = [s / max(np.linalg.norm(s), 1e-9) for _, s in steps]
                if any(float(d1 @ d2) < 0.5 for d1, d2 in zip(dirs, dirs[1:])):
                    consistent = False
                if consistent and self.rows[tip]['n_obs'] >= 1:
                    drop.update(chain[:-1])
            elif links == 1 and self.rows[tip]['n_obs'] >= self.tip_confirm_obs:
                dist = _horizontal(self.rows[b]['center'], self.rows[a]['center'])
                if dist <= self.short_hop:
                    drop.add(a)
        return drop

    def materialize(self, all_pred_box, fusion_list):
        """Terminal output transform; None means keep native output."""
        if not self.enabled:
            return None
        corners = all_pred_box.pred_boxes_3d.corners.detach().cpu().numpy()
        scores = all_pred_box.scores.detach().cpu().numpy()
        if self.dump_only:
            if self.dump_path:
                import json as _json
                with open(self.dump_path, 'a') as fh:
                    for i, row_ids in enumerate(fusion_list):
                        key = int(row_ids[0]) if row_ids else -1
                        fh.write(_json.dumps({
                            'key': key,
                            'obs': self.history.get(key, []),
                            'native_corners': corners[i].tolist(),
                            'native_score': float(scores[i]),
                        }) + '\n')
                print(f'MotionChain observer dump written: {self.dump_path}')
            return None

        n = len(fusion_list)
        if n != len(corners):
            raise RuntimeError('motion_chain: fusion_list out of sync with rows')
        mask = np.ones(n, dtype=bool)
        out_corners = corners.copy()
        # map row -> live key
        row_keys = []
        for row_ids in fusion_list:
            row_keys.append(int(row_ids[0]) if row_ids else None)
        drop = self._chain_drop_set()
        for i, key in enumerate(row_keys):
            if key is None or key not in self.rows:
                continue
            if key in drop:
                mask[i] = False
                self.stats['drops'] += 1
            elif self.rows[key]['drift'] > self.drift_span:
                out_corners[i] = self.rows[key]['latest_corners']
                self.stats['drift_replacements'] += 1
        print('MotionChain summary |',
              f"keyframes={self.stats['keyframes']}, links={self.stats['links']}, "
              f"reactivations={self.stats['reactivations']}, drops={self.stats['drops']}, "
              f"drift_replacements={self.stats['drift_replacements']}")
        return {'mask': mask, 'corners': out_corners, 'scores': scores.copy()}
