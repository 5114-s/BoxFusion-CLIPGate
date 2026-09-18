#!/usr/bin/env python3
"""End-to-end replay of the integrated DynamicPolicy on the two benchmarks.

Pre-registered acceptance bar:
  (a) Behave clean: mean IoU within 0.02 of the latest-frame baseline
      (do-no-harm on clean frames);
  (b) Behave occluded (top-60% mask clip): mean IoU >= occluded-latest + 0.08
      through the full ONLINE policy path (degradation gate + completion);
  (c) Bonn person_tracking: median centre error within 0.05 m of the
      latest-frame arm on unified references; phantom outputs on the 11
      person-absent keyframes <= 1; exactly one track (no duplicate births).

All inputs are cached causal observations (frozen YOLOE); no GT enters the
policy -- GT is used for evaluation only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT))

from boxfusion.dynamic_policy import DynamicPolicy, Observation
from boxfusion.truncation_signals import truncation_signals
from run_behave_occlusion_diag import aabb_from_mask, ensure_rectified
from run_bonn_fourarm import reference_center

PROBE = ROOT / 'reports/behave_motion_probe_20260914'
BEHAVE_DATA = ROOT / 'data_behave_probe/rectified'
OUT = ROOT / 'reports/dynamic_policy_replay_20260914'


def iou_aabb(a, b):
    inter = np.maximum(0, np.minimum(a[1], b[1]) - np.maximum(a[0], b[0])).prod()
    u = (a[1] - a[0]).prod() + (b[1] - b[0]).prod() - inter
    return float(inter / max(u, 1e-9))


def behave_obs(t, row, mask=None, frac=None, virtual_occluder=False):
    """Observation from a probe row; optionally recompute from a clipped mask.
    With ``virtual_occluder`` the depth below the clip line (within the mask's
    column span) is replaced by a NEARER surface so the occluder signature is
    physically present -- the simulated truncation then exercises the real
    'truncation' gate rather than bypassing it."""
    if mask is None:
        c = np.asarray(row['box'][:3]); s = np.asarray(row['box'][3:])
        return Observation(center=c, lo=c - s / 2, hi=c + s / 2,
                           score=row['score'], valid_px=row['valid_pixels'])
    depth = np.load(BEHAVE_DATA / f'{t}.npy')
    m = mask.copy()
    split = None
    if frac is not None:
        rows = np.nonzero(m)[0]
        split = rows.min() + frac * (rows.max() - rows.min())
        m[int(split):, :] = False
    (lo, hi), n = aabb_from_mask(m, depth)
    if lo is None:
        return None
    depth_eff = depth
    if virtual_occluder and split is not None:
        ys, xs = np.nonzero(mask)
        z_person = float(np.median(depth[ys, xs]))
        depth_eff = depth.copy()
        depth_eff[int(split):, xs.min():xs.max() + 1] = z_person * 0.6
        sig = truncation_signals(m, depth_eff,
                                 [float(xs.min()), float(ys.min()),
                                  float(xs.max()), float(ys.max())], z_person)
    elif virtual_occluder:
        sig = {'trunc_below': False, 'trunc_above': False}
    else:
        sig = {'trunc_below': False, 'trunc_above': False}
    return Observation(center=(lo + hi) / 2, lo=lo, hi=hi,
                       score=row['score'], valid_px=n,
                       trunc_below=sig.get('trunc_below', False),
                       trunc_above=sig.get('trunc_above', False))


def run_behave(occluded):
    obs_rows = json.loads((PROBE / 'observations.json').read_text())
    refs = {int(t): v for t, v in
            json.loads((PROBE / 'references.json').read_text()).items()}
    policy = DynamicPolicy({'degrade_ratio': 0.60, 'completion_mode': 'truncation'})
    per_frame = []
    for o in obs_rows:
        t = float(o['time'])
        if occluded and o['time'] >= 4:
            # occlusion starts after one clean seed frame, matching the
            # offline diagnostic design (clean history before occlusion);
            # a virtual occluder is placed in the depth so the causal
            # truncation gate itself must fire (no GT, no flag bypass)
            mask = cv2.imread(str(PROBE / f'predicted_mask_{o["time"]}.png'), 0) > 127
            ob = behave_obs(o['time'], o['selected'], mask=mask, frac=0.6,
                            virtual_occluder=True)
        else:
            ob = behave_obs(o['time'], o['selected'])
        policy.process(t, [ob] if ob is not None else [])
        outs = policy.outputs(t)
        gt = (np.asarray(refs[int(t)]['lo']), np.asarray(refs[int(t)]['hi']))
        pol = None
        for b in outs.values():
            box = (b['lo'], b['hi'])
            v = iou_aabb(box, gt)
            if pol is None or v > pol:
                pol = v
        base = iou_aabb((ob.lo, ob.hi), gt) if ob is not None else None
        per_frame.append({'t': int(t), 'policy_iou': round(pol, 3) if pol is not None else None,
                          'latest_iou': round(base, 3) if base is not None else None})
    mean_pol = np.mean([r['policy_iou'] for r in per_frame if r['policy_iou'] is not None])
    mean_base = np.mean([r['latest_iou'] for r in per_frame if r['latest_iou'] is not None])
    return {'per_frame': per_frame, 'mean_policy': round(float(mean_pol), 3),
            'mean_latest': round(float(mean_base), 3),
            'stats': dict(policy.stats)}


def run_bonn():
    ann = json.loads((ROOT / 'reports/bonn_fourarm_20260914/pt_annotation.json').read_text())
    ts = {int(k): v for k, v in
          json.loads((ROOT / 'reports/bonn_detect_20260914/pt_timestamps.json').read_text()).items()}
    recs = json.loads((ROOT / 'reports/bonn_detect_20260914/yoloe_candidates.json').read_text())
    policy = DynamicPolicy()
    refs, sanity = {}, {}
    for f, box in ann['boxes'].items():
        c, info = reference_center(box, int(f))
        if c is not None and info and info['person_sane']:
            refs[int(f)] = c
            sanity[int(f)] = info
    errs, misses = [], 0
    phantoms = 0
    absent = set(ann['no_person_frames'])
    for r in recs:
        if r['scene'] != 'scene0001_01':
            continue
        t = ts[r['frame']]
        people = [d for d in r['detections'] if d['label'] == 'person'
                  and 'world_aabb_lo' in d]
        obs_list = []
        for d in people:
            lo = np.asarray(d['world_aabb_lo']); hi = np.asarray(d['world_aabb_hi'])
            obs_list.append(Observation(center=(lo + hi) / 2, lo=lo, hi=hi,
                                        score=d['score'],
                                        valid_px=d['n_depth_points']))
        policy.process(t, obs_list)
        outs = policy.outputs(t)
        fr = r['frame']
        if fr in absent:
            phantoms += len(outs)
        if fr in refs:
            if outs:
                best = min(float(np.linalg.norm(b['center'][:2] - refs[fr][:2]))
                           for b in outs.values())
                errs.append(round(best, 3))
            else:
                misses += 1
    return {'median_err': round(float(np.median(errs)), 3),
            'mean_err': round(float(np.mean(errs)), 3),
            'n_eval': len(errs), 'miss': misses,
            'phantom_outputs_on_absent': phantoms,
            'n_tracks': len(policy.tracks), 'stats': dict(policy.stats)}


def main():
    OUT.mkdir(exist_ok=True)
    clean = run_behave(occluded=False)
    occ = run_behave(occluded=True)
    bonn = run_bonn()
    checks = {
        'a_behave_clean_within_0.02': clean['mean_policy'] >= clean['mean_latest'] - 0.02,
        'b_behave_occluded_gain_ge_0.08': occ['mean_policy'] >= occ['mean_latest'] + 0.08,
        'c_bonn_err_within_0.05': bonn['median_err'] <= 0.30 + 0.05,
        'c_bonn_phantom_le_1': bonn['phantom_outputs_on_absent'] <= 1,
        'c_bonn_single_track': bonn['n_tracks'] == 1,
    }
    result = {'behave_clean': clean, 'behave_occluded': occ, 'bonn': bonn,
              'acceptance': checks,
              'all_pass': all(checks.values())}
    (OUT / 'replay_results.json').write_text(json.dumps(result, indent=1))
    print(json.dumps({'behave_clean': {k: clean[k] for k in ('mean_policy', 'mean_latest')},
                      'behave_occluded': {k: occ[k] for k in ('mean_policy', 'mean_latest')},
                      'bonn': bonn, 'acceptance': checks, 'all_pass': result['all_pass']},
                     indent=1))


if __name__ == '__main__':
    main()
