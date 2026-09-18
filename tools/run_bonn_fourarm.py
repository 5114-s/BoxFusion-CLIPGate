#!/usr/bin/env python3
"""Independent 3D references + four-arm causal per-frame tracking experiment.

3D reference (eval-only, independent of YOLOE): for each VLM-annotated frame,
person pixels are the depth-modal cluster INSIDE the annotation box (nearest
coherent surface), backprojected with sensor depth and the frame pose into
world coordinates; reference uses the SAME 2%/98% per-axis bounds and AABB
centre as the cached predictions. Different point-selection methods remain
a limitation: the reference cluster is a proxy, not verified human 3D GT.

Four arms share the exact same frozen YOLOE person candidates
(yoloe_candidates.json), see real timestamps (pt_timestamps.json), emit one
3D centre per frame, and read only current+past observations:

A latest_single  - best-score candidate of frame t (miss if none)
B short_window   - median centre of candidates in the last 5 keyframes
C const_velocity - causal EMA-velocity tracker; coast when no observation
D motion_aware   - causal Theil-Sen velocity over the last 8 keyframes,
                   every observation centre compensated to t (score*recency
                   weighted mean); coast on empty window (centre-compensated
                   fusion; full projection-consistency PFO is the next step)

Single-person clip (person_tracking); identity is trivial by construction.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FRM = ROOT / 'data_bonn/scene0001_01/frames'
REP = ROOT / 'reports/bonn_detect_20260914'
OUT = ROOT / 'reports/bonn_fourarm_20260914'
ARMS = ('A_latest', 'B_window', 'C_cvel', 'D_maware', 'HOLD')


def visible_aabb_center(points):
    """Match run_yoloe_bonn_candidates.py exactly; no min/max reference."""
    points = np.asarray(points, dtype=float)
    lo, hi = np.quantile(points, (0.02, 0.98), axis=0)
    return (lo + hi) / 2, hi - lo


def reference_center(box, fr):
    """Unified reference: AABB CENTRE of the person's visible-surface world
    points (nearest coherent depth cluster inside the annotation box) --
    the same statistic the YOLOE prediction uses (mask-depth AABB centre).
    Size sanity is NOT a semantic person-membership test. Its original
    coordinate-specific thresholds are retained for a sensitivity subset."""
    K = np.loadtxt(FRM / 'intrinsic/intrinsic_depth.txt')[:3, :3]
    dep = np.asarray(Image.open(FRM / f'depth/{fr}.png')).astype(np.float64) / 5000.0
    pose = np.loadtxt(FRM / f'pose/{fr}.txt')
    x1, y1, x2, y2 = map(int, box)
    x1, x2 = np.clip([x1, x2], 0, dep.shape[1])
    y1, y2 = np.clip([y1, y2], 0, dep.shape[0])
    if x2 <= x1 or y2 <= y1:
        return None, {'failure': 'empty_annotation_box', 'person_sane': False}
    sub = dep[y1:y2, x1:x2]
    valid = sub[(sub > 0.3) & (sub < 8.0)]
    if len(valid) < 50:
        return None, {'failure': 'insufficient_valid_depth', 'person_sane': False}
    # nearest coherent cluster: depth within 0.25 m of the 20th percentile
    ref = np.quantile(valid, 0.2)
    m = (sub > 0.3) & (sub < 8.0) & (sub > ref - 0.25) & (sub < ref + 0.25)
    rows, cols = np.nonzero(m)
    z = sub[rows, cols]
    if len(z) < 8:
        return None, {'failure': 'insufficient_cluster_points', 'person_sane': False}
    xs = (cols + x1 - K[0, 2]) / K[0, 0] * z
    ys = (rows + y1 - K[1, 2]) / K[1, 1] * z
    pc = np.stack([xs, ys, z], 1)
    pw = pc @ pose[:3, :3].T + pose[:3, 3]
    centre, trimmed_size = visible_aabb_center(pw)
    size = np.ptp(pw, axis=0)  # Freeze the previous RAW-extent filter.
    sane = (0.5 < size[1] < 2.4) and (0.2 < size[0] < 1.2) and (0.2 < size[2] < 1.2)
    return centre, {'n_points': int(len(z)), 'raw_extent_xyz_m': size.tolist(),
                    'quantile_extent_xyz_m': trimmed_size.tolist(),
                    'person_sane': bool(sane),  # Legacy key, size sanity ONLY.
                    'person_membership_verified': False,
                    'center_statistic': 'per-axis 0.02/0.98 quantile AABB midpoint'}


def candidates():
    recs = json.loads((REP / 'yoloe_candidates.json').read_text())
    out = {}
    for r in recs:
        if r['scene'] != 'scene0001_01':
            continue
        people = [d for d in r['detections'] if d['label'] == 'person'
                  and 'world_aabb_lo' in d]
        if people:
            best = max(people, key=lambda d: d['score'])
            out[r['frame']] = {
                'center': (np.asarray(best['world_aabb_lo']) +
                           np.asarray(best['world_aabb_hi'])) / 2,
                'score': best['score'],
                'size': (np.asarray(best['world_aabb_hi']) -
                         np.asarray(best['world_aabb_lo'])),
                'n': len(people)}
    return out


def load_inputs():
    ann = json.loads((OUT / 'pt_annotation.json').read_text())
    ts = {int(k): v for k, v in
          json.loads((REP / 'pt_timestamps.json').read_text()).items()}
    cands = candidates()
    records = json.loads((REP / 'yoloe_candidates.json').read_text())
    # The timestamp file contains ALL RGB frames; replay only exported KFs.
    kfs = sorted(r['frame'] for r in records if r['scene'] == ann['scene'])
    if len(set(kfs)) != len(kfs) or any(k not in ts for k in kfs):
        raise ValueError('Duplicate keyframes or missing timestamps')
    if any(ts[b] <= ts[a] for a, b in zip(kfs, kfs[1:])):
        raise ValueError('Timestamps must be strictly increasing')

    refs = {}
    sanity = {}
    for f, box in ann['boxes'].items():
        c, info = reference_center(box, int(f))
        sanity[int(f)] = info
        if c is not None:
            refs[int(f)] = c
    return ann, ts, cands, kfs, refs, sanity


def replay(kfs, ts, cands, masked_frames=()):
    """Pure causal replay; masked observations are removed before ANY update.

    No reference, annotation or evaluation value is an input to this function.
    C/D algorithms and their hyperparameters are unchanged.
    """
    masked_frames = set(masked_frames)
    cands = {k: v for k, v in cands.items() if k not in masked_frames}
    # ---------------- four causal arms ----------------
    outs = {a: {} for a in ARMS}
    c_est = v_est = None
    last_obs = None
    for t in kfs:
        # A
        if t in cands:
            outs['A_latest'][t] = cands[t]['center'].copy()
        # B
        win = [cands[k]['center'] for k in kfs if k in cands
               and k <= t and k >= t - 100]
        if win:
            outs['B_window'][t] = np.median(np.stack(win), 0)
        # C
        if t in cands:
            c_obs = cands[t]['center']
            if last_obs is not None:
                dt = ts[t] - ts[last_obs]
                v_new = (c_obs - c_est) / max(dt, 1e-3)
                v_est = 0.7 * v_est + 0.3 * v_new if v_est is not None else v_new
            c_est = c_obs.copy()
            last_obs = t
            outs['C_cvel'][t] = c_est.copy()
        elif c_est is not None and v_est is not None:
            dt = ts[t] - ts[last_obs]
            outs['C_cvel'][t] = c_est + v_est * dt
        if c_est is not None:
            outs['HOLD'][t] = c_est.copy()
        # D
        win_kfs = [k for k in kfs if k in cands and k <= t and k >= t - 175]
        if win_kfs:
            cs = np.stack([cands[k]['center'] for k in win_kfs])
            tt = np.asarray([ts[k] for k in win_kfs])
            # causal Theil-Sen velocity (median pairwise slope), per axis
            v = np.zeros(3)
            for a in range(3):
                slopes = []
                for i in range(len(tt)):
                    for j in range(i + 1, len(tt)):
                        dtk = tt[j] - tt[i]
                        if dtk > 1e-3:
                            slopes.append((cs[j, a] - cs[i, a]) / dtk)
                v[a] = float(np.median(slopes)) if slopes else 0.0
            comp = cs + v[None, :] * (ts[t] - tt)[:, None]
            w = np.asarray([cands[k]['score'] * np.exp(-(ts[t] - ts[k]))
                            for k in win_kfs])
            w = w / w.sum()
            outs['D_maware'][t] = (comp * w[:, None]).sum(0)
        elif c_est is not None and v_est is not None:
            dt = ts[t] - ts[last_obs]
            outs['D_maware'][t] = c_est + v_est * dt
    return outs


def error_at(outputs, t, reference, axes=None):
    if reference is None or t not in outputs:
        return None
    delta = outputs[t] - reference
    if axes is not None:
        delta = delta[list(axes)]
    return float(np.linalg.norm(delta))


def summarize_errors(rows, arms=ARMS):
    summary = {}
    for arm in arms:
        errs = [r[arm] for r in rows if r[arm] is not None]
        miss = sum(1 for r in rows if r[arm] is None)
        summary[arm] = {'n_eval': len(errs), 'miss': miss,
                        'median_m': float(np.median(errs)) if errs else None,
                        'mean_m': float(np.mean(errs)) if errs else None,
                        'max_m': float(np.max(errs)) if errs else None}
    return summary


def input_hashes():
    paths = [Path(__file__), ROOT / 'tools/run_yoloe_bonn_candidates.py',
             OUT / 'pt_annotation.json', REP / 'pt_timestamps.json',
             REP / 'yoloe_candidates.json',
             FRM / 'intrinsic/intrinsic_depth.txt']
    ann = json.loads((OUT / 'pt_annotation.json').read_text())
    for frame in ann['boxes']:
        paths.extend([FRM / f'depth/{frame}.png', FRM / f'pose/{frame}.txt'])
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def main():
    ann, ts, cands, kfs, refs, sanity = load_inputs()
    outs = replay(kfs, ts, cands)
    rows, xy_rows = [], []
    for t, ref in sorted(refs.items()):
        row = {'frame': t, 'ref_stats': sanity[t], 'ref_xyz_m': ref.tolist()}
        row.update({a: error_at(outs[a], t, ref) for a in ARMS})
        rows.append(row)
        xy_rows.append({'frame': t, **{a: error_at(outs[a], t, ref, (0, 1)) for a in ARMS}})
    kept = [r for r in rows if sanity[r['frame']]['person_sane']]
    dropped = sorted(t for t in refs if not sanity[t]['person_sane'])
    result = {'summary': summarize_errors(kept), 'per_frame': rows,
              'summary_all_references': summarize_errors(rows),
              'summary_xy_projection_legacy': summarize_errors(
                  [r for r in xy_rows if sanity[r['frame']]['person_sane']]),
              'reference_stats': sanity,
              'refs_kept_size_filter': [r['frame'] for r in kept],
              'refs_dropped_unsane': dropped,
              'reference_failures': {t: info for t, info in sanity.items() if t not in refs},
              'keyframes': kfs, 'input_sha256': input_hashes(),
              'metric': 'Euclidean 3D centre error in metres; no gravity-axis assumption',
              'note': 'Both centres use 2%/98% world-coordinate AABB bounds. '
                      'Size filtering is a sensitivity analysis, not proof of person membership. '
                      'References are VLM-box/depth proxies, not full human-box ground truth.'}
    (OUT / 'fourarm_results_v3.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('References:', len(refs), 'size-filter kept:', len(kept), 'flagged:', dropped)
    print(json.dumps(result['summary'], indent=1))


if __name__ == '__main__':
    main()
