#!/usr/bin/env python3
"""Real-occlusion evaluation of the completion prior on Behave chairwood t=14..45.

Pipeline per frame (camera k1): rectify -> frozen YOLOE person (retina masks,
probe settings) -> visible-surface AABB from masked depth (2/98 quantile);
GT = person_fit.ply mesh AABB.  For every frame compute

  r = observed height / GT height                (visible fraction, eval-only)
  BASE  IoU  - raw observation
  FIX   IoU  - completion with the frozen prior 0.6 (policy operator)
  ORACLE IoU - completion with r itself (upper bound of prior estimation)

Stratify by r: completion should help where r is genuinely low (real
occlusion/partial view) and not hurt where r ~ 1.  No GT enters the policy
operator; r is used only for stratification and the oracle bound.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEQ = ROOT / 'data_behave_probe/Date04_Sub05_chairwood'
RECT = ROOT / 'data_behave_probe/rectified'
OUT = ROOT / 'reports/behave_occlusion_diag_20260914'
K = np.array([[306.18263244628906, 0.0, 318.42260360717773],
              [0.0, 306.200008392334, 243.58959197998047],
              [0.0, 0.0, 1.0]])


def rectify(times):
    cal = json.loads((ROOT / 'data_behave_probe/calibs/intrinsics/1/calibration.json').read_text())['color']
    ok = np.array([[cal['fx'], 0, cal['cx']], [0, cal['fy'], cal['cy']], [0, 0, 1.]])
    k = ok.copy(); k[0] *= 640 / cal['width']; k[1] *= 480 / cal['height']
    maps = cv2.initUndistortRectifyMap(ok, np.array(cal['opencv'][4:]), None,
                                       k, (640, 480), cv2.CV_32FC1)
    RECT.mkdir(exist_ok=True)
    for t in times:
        p = SEQ / f't{t:04d}.000'
        dep = cv2.imread(str(p / 'k1.depth.png'), -1)
        np.save(RECT / f'{t}.npy',
                cv2.remap(dep, *maps, cv2.INTER_NEAREST).astype(np.float32) / 1000.)


def iou_aabb(a, b):
    inter = np.maximum(0, np.minimum(a[1], b[1]) - np.maximum(a[0], b[0])).prod()
    u = (a[1] - a[0]).prod() + (b[1] - b[0]).prod() - inter
    return float(inter / max(u, 1e-9))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--times', type=int, nargs='+', default=list(range(14, 46)))
    args = p.parse_args()
    rectify(args.times)

    from ultralytics import YOLOE
    import trimesh
    model = YOLOE('/data/ZhaoX/OVM3D-Dett/boxfusion_b6_selective_boxer_dev/models/yoloe-11s-seg-pf.pt')

    rows = []
    for t in args.times:
        depth = np.load(RECT / f'{t}.npy')
        rgb = cv2.imread(str(SEQ / f't{t:04d}.000/k1.color.jpg'))
        rgb = cv2.resize(rgb, (640, 480))
        res = model.predict(rgb, device='cuda:0', retina_masks=True, verbose=False,
                            conf=0.25, iou=0.7, imgsz=640, max_det=64,
                            agnostic_nms=True)[0]
        best = None
        if res.masks is not None and res.boxes is not None:
            names = res.names
            for j in range(len(res.boxes)):
                label = names[int(res.boxes.cls[j])] if isinstance(names, dict) else str(names[int(res.boxes.cls[j])])
                if label != 'person':
                    continue
                m = res.masks.data[j].cpu().numpy() > 0.5
                if m.shape != depth.shape:
                    m = cv2.resize(m.astype(np.uint8), (depth.shape[1], depth.shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                valid = m & (depth > 0.05) & (depth < 12)
                ys, xs = np.nonzero(valid)
                if len(xs) < 8:
                    continue
                z = depth[ys, xs]
                pts = np.stack([(xs - K[0, 2]) / K[0, 0] * z,
                                (ys - K[1, 2]) / K[1, 1] * z, z], 1)
                lo, hi = np.quantile(pts, [.02, .98], axis=0)
                cand = {'n': int(len(xs)), 'lo': lo, 'hi': hi,
                        'score': float(res.boxes.conf[j])}
                if best is None or cand['score'] > best['score']:
                    best = cand
        verts = np.asarray(trimesh.load(
            SEQ / f't{t:04d}.000/person/fit02/person_fit.ply', process=False).vertices)
        gt = (verts.min(0), verts.max(0))
        if best is None:
            rows.append({'t': t, 'detected': False})
            continue
        lo, hi, n = best['lo'], best['hi'], best['n']
        h_obs = float(hi[1] - lo[1])
        h_gt = float(gt[1][1] - gt[0][1])
        r = h_obs / max(h_gt, 1e-6)
        base = iou_aabb((lo, hi), gt)
        hi_f = hi.copy(); hi_f[1] = lo[1] + float(np.clip(h_obs / 0.6, 0.8, 2.2))
        fix = iou_aabb((lo, hi_f), gt)
        hi_o = hi.copy(); hi_o[1] = lo[1] + float(np.clip(h_gt, 0.8, 2.2))
        oracle = iou_aabb((lo, hi_o), gt)
        rows.append({'t': t, 'detected': True, 'n_px': n, 'r': round(r, 3),
                     'iou_base': round(base, 3), 'iou_fix': round(fix, 3),
                     'iou_oracle': round(oracle, 3)})
        print(rows[-1], flush=True)

    det = [x for x in rows if x.get('detected')]
    def band(lo_, hi_):
        sel = [x for x in det if lo_ <= x['r'] < hi_]
        if not sel:
            return None
        return {'n': len(sel),
                'mean_base': round(float(np.mean([x['iou_base'] for x in sel])), 3),
                'mean_fix': round(float(np.mean([x['iou_fix'] for x in sel])), 3),
                'mean_oracle': round(float(np.mean([x['iou_oracle'] for x in sel])), 3)}
    summary = {'n_frames': len(rows), 'n_detected': len(det),
               'bands': {f'{lo_}-{hi_}': band(lo_, hi_)
                         for lo_, hi_ in [(0.0, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.05), (1.05, 2.0)]},
               'overall': {'mean_base': round(float(np.mean([x['iou_base'] for x in det])), 3),
                           'mean_fix': round(float(np.mean([x['iou_fix'] for x in det])), 3),
                           'mean_oracle': round(float(np.mean([x['iou_oracle'] for x in det])), 3)}}
    (OUT / 'real_occlusion_results.json').write_text(
        json.dumps({'summary': summary, 'rows': rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
