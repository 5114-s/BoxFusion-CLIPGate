#!/usr/bin/env python3
"""Stage B (boxfusion2): history-prompted SAM2 recovery on blurred misses.

At each YOLOE miss: causal box prompt = last hit's world AABB propagated by
the velocity of the last two hits, projected into the current camera, widened
1.25x.  SAM2 decodes a mask on the CURRENT blurred RGB; the (unblurred) depth
turns it into a 3D observation.  No evaluation information enters the prompt.

Arms per miss frame:
  COAST - velocity-propagated box alone (no current image evidence)
  SAM2  - prompt + current RGB + current depth (history as PROMPT, not proof)

Pre-registered decision (user's table): continue iff >= 2/3 miss segments
recovered (matched IoU2D >= 0.3), no ghost recoveries (IoU2D < 0.1) on
attempted prompts, SAM2 latency acceptable (< 100 ms/frame).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
OUT = ROOT / 'reports/recovery_gate2_20260914'
FRAMES = ROOT / 'data_bonn/scene0001_01/frames'
MISS_SEGMENTS = [[25, 50, 75], [125], [225, 250, 275]]


def iou2d(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(u, 1e-9)


def motion_blur(img, dx, dy, k):
    n = np.hypot(dx, dy)
    if n < 1e-6:
        return img
    ux, uy = dx / n, dy / n
    c = (k - 1) / 2
    xs = np.linspace(-c, c, k) * ux
    ys = np.linspace(-c, c, k) * uy
    ker = np.zeros((k, k), np.float32)
    for x, y in zip(xs, ys):
        ker[int(round(y + c)), int(round(x + c))] += 1
    return cv2.filter2D(img, -1, ker / ker.sum())


def main():
    stageA = json.loads((OUT / 'stageA_yoloe_blur7.json').read_text())
    by_frame = {r['frame']: r for r in stageA}
    ann = json.loads((ROOT / 'reports/bonn_fourarm_20260914/pt_annotation.json').read_text())
    ann_boxes = {int(k): v for k, v in ann['boxes'].items()}
    K = np.loadtxt(FRAMES / 'intrinsic/intrinsic_depth.txt')[:3, :3]
    cands = json.loads((ROOT / 'reports/bonn_detect_20260914/yoloe_candidates_v2.json').read_text())
    centers = {}
    for r in cands:
        if r['scene'] != 'scene0001_01':
            continue
        ps = [d for d in r['detections'] if d['label'] == 'person']
        if ps:
            b = max(ps, key=lambda d: d['score'])['box2d']
            centers[r['frame']] = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

    def load_blurred(fr):
        img = cv2.imread(str(FRAMES / f'color/{fr}.jpg'))
        cx, cy = centers.get(fr, (320, 240))
        prev = centers.get(fr - 25, (cx - 1, cy))
        return motion_blur(img, cx - prev[0], cy - prev[1], 7)

    def obs_from_mask(mask, fr):
        depth = np.asarray(Image.open(FRAMES / f'depth/{fr}.png')).astype(np.float64) / 5000.0
        valid = mask & (depth > 0.05) & (depth < 12.0)
        ys, xs = np.nonzero(valid)
        if len(xs) < 8:
            return None, 0
        z = depth[ys, xs]
        pts = np.stack([(xs - K[0, 2]) / K[0, 0] * z,
                        (ys - K[1, 2]) / K[1, 1] * z, z], 1)
        pose = np.loadtxt(FRAMES / f'pose/{fr}.txt')
        pw = pts @ pose[:3, :3].T + pose[:3, 3]
        lo, hi = np.quantile(pw, [.02, .98], axis=0)
        return (lo, hi), int(len(xs))

    def project_box(box3d, fr):
        lo, hi = box3d
        pose = np.loadtxt(FRAMES / f'pose/{fr}.txt')
        Rt = np.linalg.inv(pose)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                            for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        pc = (Rt[:3, :3] @ corners.T).T + Rt[:3, 3]
        uv = (K @ pc.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        return [float(uv[:, 0].min()), float(uv[:, 1].min()),
                float(uv[:, 0].max()), float(uv[:, 1].max())]

    # hit observations (3D) in chronological order
    hits = sorted(r['frame'] for r in stageA if r['hit'])
    hit_obs = {}
    for h in hits:
        mask = np.asarray(Image.open(OUT / 'masks' / f'{h}.png')) > 127
        box3d, n = obs_from_mask(mask, h)
        if box3d is not None:
            hit_obs[h] = box3d

    from boxfusion.sam2_boxprompt_provider import FrozenSAM2BoxPromptProvider
    provider = FrozenSAM2BoxPromptProvider()

    rows = []
    for seg in MISS_SEGMENTS:
        for fr in seg:
            ref = ann_boxes[fr]
            last_hits = [h for h in hits if h < fr and h in hit_obs]
            if len(last_hits) < 1:
                rows.append({'frame': fr, 'error': 'no causal history'})
                continue
            h1 = last_hits[-1]
            c1 = (hit_obs[h1][0] + hit_obs[h1][1]) / 2
            if len(last_hits) >= 2:
                h2 = last_hits[-2]
                c2 = (hit_obs[h2][0] + hit_obs[h2][1]) / 2
                v = (c1 - c2) / (h1 - h2)
            else:
                v = np.zeros(3)
            pred_box3d = (hit_obs[h1][0] + v * (fr - h1),
                          hit_obs[h1][1] + v * (fr - h1))
            prompt2d = project_box(pred_box3d, fr)
            w = prompt2d[2] - prompt2d[0]
            hgt = prompt2d[3] - prompt2d[1]
            prompt2d = [max(0, prompt2d[0] - 0.125 * w), max(0, prompt2d[1] - 0.125 * hgt),
                        min(639.0, prompt2d[2] + 0.125 * w), min(479.0, prompt2d[3] + 0.125 * hgt)]
            coast2d = project_box(pred_box3d, fr)
            coast_iou = iou2d(coast2d, ref)
            print('  DEBUG fr', fr, 'prompt2d', [round(v,1) for v in prompt2d], flush=True)
            blurred = load_blurred(fr)
            t0 = time.perf_counter()
            res = provider.predict(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB),
                                   np.asarray([prompt2d], np.float32))
            sam2_ms = (time.perf_counter() - t0) * 1000
            mask = res.masks[0] if len(res.masks) else None
            rec_iou, rec_n, rec_box3d = 0.0, 0, None
            if mask is not None and mask.any():
                box3d, n = obs_from_mask(mask, fr)
                if box3d is not None:
                    rec_box3d = box3d
                    rec_n = n
                    rec_iou = iou2d(project_box(box3d, fr), ref)
            rows.append({'frame': fr, 'segment': seg[0], 'prompt_iou_vs_ref': round(iou2d(prompt2d, ref), 3),
                         'coast_iou2d': round(coast_iou, 3),
                         'sam2_recovered_iou2d': round(rec_iou, 3),
                         'sam2_depth_px': rec_n, 'sam2_ms': round(sam2_ms, 1),
                         'ghost': bool(0 < rec_iou < 0.1)})
            print(rows[-1], flush=True)

    seg_stats = []
    for seg in MISS_SEGMENTS:
        seg_rows = [r for r in rows if r.get('segment') == seg[0]]
        if not seg_rows:
            continue
        rec = [r for r in seg_rows if r['sam2_recovered_iou2d'] >= 0.3]
        coast = [r for r in seg_rows if r['coast_iou2d'] >= 0.3]
        ghosts = [r for r in seg_rows if r.get('ghost')]
        seg_stats.append({'segment': seg, 'frames': len(seg_rows),
                          'sam2_recovered': len(rec), 'coast_matched': len(coast),
                          'ghosts': len(ghosts)})
    lat = [r['sam2_ms'] for r in rows if 'sam2_ms' in r]
    verdict_continue = (sum(1 for s in seg_stats if s['sam2_recovered'] >= max(1, s['frames'] // 2 + s['frames'] % 2)) >= 2
                        and sum(s['ghosts'] for s in seg_stats) == 0
                        and float(np.median(lat)) < 100.0)
    summary = {'segments': seg_stats, 'median_sam2_ms': round(float(np.median(lat)), 1),
               'decision': 'CONTINUE' if verdict_continue else
                           ('INSUFFICIENT' if any(s['sam2_recovered'] > 0 for s in seg_stats) else 'PAUSE')}
    (OUT / 'recovery_results.json').write_text(
        json.dumps({'summary': summary, 'rows': rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
