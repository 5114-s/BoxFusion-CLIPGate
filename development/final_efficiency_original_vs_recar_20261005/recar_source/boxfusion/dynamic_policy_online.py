"""Online wrapper binding DynamicPolicy to the demo pipeline via a
person front-end daemon (boxfusion-online env, JSONL over stdin/stdout).

One daemon per scene.  Per keyframe the wrapper sends the frame paths,
intrinsics, pose and depth scale; the daemon replies with world-frame
person observations (visible-surface AABB + causal truncation signals).
At terminal time the wrapper suppresses native rows overlapping any live
dynamic output and appends the dynamic outputs (current-state semantics).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np

from boxfusion.dynamic_policy import DynamicPolicy, Observation

ROOT = Path(__file__).resolve().parents[1]


class DynamicPolicyOnline:
    def __init__(self, cfg, scene_id):
        dp = cfg.get('dynamic_policy', {})
        self.policy = DynamicPolicy(dp)
        self.suppress_iou = float(dp.get('suppress_iou', 0.2))
        self.classes = list(dp.get('classes', ['person']))
        prov = dp.get('provider', {})
        python = prov.get('python', '/home/admin1/miniconda3/envs/boxfusion-online/bin/python')
        script = prov.get('script', str(ROOT / 'tools/dynamic_provider_daemon.py'))
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        self.proc = subprocess.Popen(
            [python, '-u', script], cwd=str(ROOT), env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True)
        ready = json.loads(self.proc.stdout.readline())
        if not ready.get('ready'):
            raise RuntimeError(f'dynamic policy provider failed to start: {ready}')
        self.calls = 0
        self.provider_ms = 0.0
        self.suppressed_rows = 0
        self.appended_rows = 0

    def process_keyframe(self, frame_idx, dataset, depth_scale, t_seconds,
                         verbose=False):
        import time as _time
        req = {
            'frame': int(frame_idx),
            'color': dataset.img_files[frame_idx],
            'depth': dataset.depth_paths[frame_idx],
            'depth_scale': float(depth_scale),
            'K': np.asarray(dataset.K, float).tolist(),
            'pose': np.asarray(dataset.poses[frame_idx], float).tolist(),
            'classes': self.classes,
        }
        t0 = _time.perf_counter()
        self.proc.stdin.write(json.dumps(req) + '\n')
        self.proc.stdin.flush()
        reply = json.loads(self.proc.stdout.readline())
        self.provider_ms += (_time.perf_counter() - t0) * 1000
        self.calls += 1
        obs_list = []
        for p in reply.get('persons', []):
            obs_list.append(Observation(
                center=np.asarray(p['center'], float),
                lo=np.asarray(p['lo'], float), hi=np.asarray(p['hi'], float),
                score=p['score'], valid_px=p['n_px'], depth_m=p.get('depth_m'),
                trunc_below=p.get('trunc_below', False),
                trunc_above=p.get('trunc_above', False)))
        self.policy.process(float(t_seconds), obs_list)
        if verbose and obs_list:
            print(f'DynamicPolicyOnline: frame={frame_idx} persons={len(obs_list)}')

    def materialize_terminal(self, native_corners, native_scores, t_end):
        outs = self.policy.outputs(float(t_end))
        dyn = [(np.asarray(b['lo']), np.asarray(b['hi']), b['score'])
               for b in outs.values()]
        keep_corners, keep_scores = [], []

        def aabb_iou(lo1, hi1, lo2, hi2):
            inter = np.maximum(0, np.minimum(hi1, hi2) - np.maximum(lo1, lo2)).prod()
            u = (hi1 - lo1).prod() + (hi2 - lo2).prod() - inter
            return float(inter / max(u, 1e-9))

        for i in range(len(native_corners)):
            c = np.asarray(native_corners[i], float)
            lo, hi = c.min(0), c.max(0)
            overlapped = any(aabb_iou(lo, hi, dl, dh) >= self.suppress_iou
                             for dl, dh, _ in dyn)
            if overlapped:
                self.suppressed_rows += 1
                continue
            keep_corners.append(native_corners[i])
            keep_scores.append(native_scores[i])
        for dl, dh, sc in dyn:
            corners = np.stack([np.array([sx, sy, sz])
                                for sx in (dl[0], dh[0])
                                for sy in (dl[1], dh[1])
                                for sz in (dl[2], dh[2])])
            keep_corners.append(corners)
            keep_scores.append(float(sc))
            self.appended_rows += 1
        print('DynamicPolicyOnline terminal | '
              f'provider_calls={self.calls} provider_ms_total={self.provider_ms:.0f} '
              f'policy_stats={self.policy.stats} '
              f'suppressed_native_rows={self.suppressed_rows} '
              f'appended_dynamic_rows={self.appended_rows}')
        return (np.asarray(keep_corners), np.asarray(keep_scores))

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.terminate()
