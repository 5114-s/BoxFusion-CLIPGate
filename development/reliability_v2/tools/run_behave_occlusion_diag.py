#!/usr/bin/env python3
"""Controlled-occlusion diagnostic on the Behave probe (quality-gated shape
retrieval vs degraded latest frame).

At every t with >=1 past observation: recompute the person observation with
the mask clipped to its TOP 60% rows (desk/chair occlusion hides the lower
body), using the probe's own geometry rule (masked valid depth -> 2/98
percentile AABB, camera-1 coordinates).  Compare:

  BASE   - latest-frame policy under occlusion: output the clipped AABB as-is
  RET-A  - quality gate (valid pixels < 0.75 x rolling past median) retrieves
           the most recent high-quality UNOCCLUDED shape from memory, placed
           at the causally velocity-predicted centre (history only)
  RET-B  - same retrieval, placed at the clipped AABB's observed centre

Pre-registered hypotheses: (H1) BASE IoU drops clearly below the clean-frame
level; (H2) RET-A beats BASE on mean IoU by >= 0.02 and on a majority of
frames.  Stop rule: H2 fails -> the shape-retrieval hypothesis is rejected
under controlled occlusion and fusion-side work stops.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / 'reports/behave_motion_probe_20260914'
DATA = ROOT / 'data_behave_probe/rectified'
K = np.array([[306.18263244628906, 0.0, 318.42260360717773],
              [0.0, 306.200008392334, 243.58959197998047],
              [0.0, 0.0, 1.0]])


def aabb_from_mask(mask, depth):
    valid = mask & (depth > 0.05) & (depth < 12)
    ys, xs = np.nonzero(valid)
    if len(xs) < 8:
        return None, 0
    z = depth[ys, xs]
    x = (xs - K[0, 2]) / K[0, 0] * z
    y = (ys - K[1, 2]) / K[1, 1] * z
    pts = np.stack([x, y, z], 1)
    lo, hi = np.quantile(pts, [.02, .98], axis=0)
    return (lo, hi), int(len(xs))


def iou_aabb(a, b):
    alo, ahi = a; blo, bhi = b
    inter = np.maximum(0, np.minimum(ahi, bhi) - np.maximum(alo, blo)).prod()
    u = (ahi - alo).prod() + (bhi - blo).prod() - inter
    return float(inter / max(u, 1e-9))


def ensure_rectified():
    """Rebuild the probe's rectified depth cache if it was cleaned."""
    if all((DATA / f'{t}.npy').is_file() for t in range(3, 14)):
        return
    seq = ROOT / 'data_behave_probe'
    cal = json.loads((seq / 'calibs/intrinsics/1/calibration.json').read_text())['color']
    ok = np.array([[cal['fx'], 0, cal['cx']], [0, cal['fy'], cal['cy']], [0, 0, 1.]])
    k = ok.copy(); k[0] *= 640 / cal['width']; k[1] *= 480 / cal['height']
    maps = cv2.initUndistortRectifyMap(ok, np.array(cal['opencv'][4:]), None,
                                       k, (640, 480), cv2.CV_32FC1)
    DATA.mkdir(parents=True, exist_ok=True)
    for t in range(3, 14):
        dep = cv2.imread(str(seq / f'Date04_Sub05_chairwood/t{t:04d}.000/k1.depth.png'), -1)
        np.save(DATA / f'{t}.npy',
                cv2.remap(dep, *maps, cv2.INTER_NEAREST).astype(np.float32) / 1000.)
    print('rectified depth rebuilt for t=3..13', flush=True)


def main():
    ensure_rectified()
    obs = json.loads((PROBE / 'observations.json').read_text())
    refs = {int(t): v for t, v in
            json.loads((PROBE / 'references.json').read_text()).items()}
    times = sorted(o['time'] for o in obs)
    by_time = {o['time']: o for o in obs}

    rows = []
    for t in times:
        if t == times[0]:
            continue
        depth = np.load(DATA / f'{t}.npy')
        mask = cv2.imread(str(PROBE / f'predicted_mask_{t}.png'), 0) > 127
        clean, _ = aabb_from_mask(mask, depth)
        rows_mask = np.nonzero(mask)[0]
        y0, y1 = rows_mask.min(), rows_mask.max()
        split = y0 + 0.6 * (y1 - y0)
        occ_mask = mask.copy()
        occ_mask[int(split):, :] = False
        occ, n_occ = aabb_from_mask(occ_mask, depth)
        gt = (np.asarray(refs[t]['lo']), np.asarray(refs[t]['hi']))

        past = [p for p in times if p < t]
        vps = [by_time[p]['selected']['valid_pixels'] for p in past]
        gate = n_occ < 0.75 * float(np.median(vps))
        best_past = max(past, key=lambda p: by_time[p]['selected']['valid_pixels'])
        s_ret = np.asarray(by_time[best_past]['selected']['box'][3:])
        c_ret = np.asarray(by_time[best_past]['selected']['box'][:3])
        # causal velocity from the last two unoccluded centres
        if len(past) >= 2:
            c_prev = np.asarray(by_time[past[-2]]['selected']['box'][:3])
            dt = max(past[-1] - past[-2], 1e-3)
            v = (c_ret - c_prev) / dt
            c_pred = c_ret + v * (t - past[-1])
        else:
            c_pred = c_ret
        ret_a = (c_pred - s_ret / 2, c_pred + s_ret / 2)
        c_occ = (occ[0] + occ[1]) / 2
        ret_b = (c_occ - s_ret / 2, c_occ + s_ret / 2)

        rows.append({
            't': t, 'n_occ_px': n_occ, 'gate_degraded': bool(gate),
            'retrieved_from': best_past,
            'iou_clean': round(iou_aabb(clean, gt), 3),
            'iou_base': round(iou_aabb(occ, gt), 3),
            'iou_ret_pred': round(iou_aabb(ret_a, gt), 3),
            'iou_ret_obs': round(iou_aabb(ret_b, gt), 3),
        })
        # COMPLETION variant: keep the occluded observation's own pose/extents,
        # extend the missing (lower) part with a fixed visibility prior
        # (top-60% visible -> full height = visible / 0.6, clamped 0.8-2.2 m)
        if occ is not None:
            h_v = occ[1][1] - occ[0][1]
            h_f = float(np.clip(h_v / 0.6, 0.8, 2.2))
            lo_c, hi_c = occ[0].copy(), occ[1].copy()
            hi_c[1] = lo_c[1] + h_f
            rows[-1]['iou_complete'] = round(iou_aabb((lo_c, hi_c), gt), 3)
        print(rows[-1], flush=True)

    def m(k):
        return round(float(np.mean([r[k] for r in rows])), 3)

    summary = {
        'n_frames': len(rows),
        'mean_iou_clean': m('iou_clean'),
        'mean_iou_base': m('iou_base'),
        'mean_iou_ret_pred': m('iou_ret_pred'),
        'mean_iou_ret_obs': m('iou_ret_obs'),
        'mean_iou_complete': m('iou_complete'),
        'ret_pred_beats_base': sum(r['iou_ret_pred'] > r['iou_base'] for r in rows),
        'ret_obs_beats_base': sum(r['iou_ret_obs'] > r['iou_base'] for r in rows),
        'complete_beats_base': sum(r['iou_complete'] > r['iou_base'] for r in rows),
    }
    summary['H1_drop'] = round(summary['mean_iou_clean'] - summary['mean_iou_base'], 3)
    summary['H2_pass'] = bool(
        summary['mean_iou_ret_pred'] - summary['mean_iou_base'] >= 0.02
        and summary['ret_pred_beats_base'] > len(rows) / 2)
    out = ROOT / 'reports/behave_occlusion_diag_20260914'
    out.mkdir(exist_ok=True)
    (out / 'occlusion_diag_results.json').write_text(
        json.dumps({'summary': summary, 'rows': rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
