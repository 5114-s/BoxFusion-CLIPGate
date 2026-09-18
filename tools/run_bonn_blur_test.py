#!/usr/bin/env python3
"""Motion-blur robustness test of the frozen person front-end (Bonn pt).

Real walking blur at ~30 fps exposure: person at ~2.5 m moving ~1 m/s
displaces ~7 px during exposure along the image motion direction (estimated
causally from consecutive person-box centres; falls back to horizontal).
A heavier 15 px variant probes robustness margin.  Gate: person coverage
>= 50% of the 13 visible keyframes + >= 3 consecutive hits; box quality is
compared against the independent VLM annotation (IoU2D).  Run in
boxfusion-online.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CKPT = '/data/ZhaoX/OVM3D-Dett/boxfusion_b6_selective_boxer_dev/models/yoloe-11s-seg-pf.pt'
VISIBLE = list(range(0, 325, 25))            # 13 VLM-verified visible kfs


def motion_blur(img, dx, dy, k):
    n = np.hypot(dx, dy)
    if n < 1e-6:
        return img
    ux, uy = dx / n, dy / n
    c = (k - 1) / 2
    xs = np.linspace(-c, c, k) * ux
    ys = np.linspace(-c, c, k) * uy
    kernel = np.zeros((k, k), np.float32)
    for x, y in zip(xs, ys):
        kernel[int(round(y + c)), int(round(x + c))] += 1
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    from ultralytics import YOLOE
    model = YOLOE(CKPT)
    frames = ROOT / 'data_bonn/scene0001_01/frames'
    cands = json.loads((ROOT / 'reports/bonn_detect_20260914/yoloe_candidates_v2.json').read_text())
    centers = {}
    for r in cands:
        if r['scene'] != 'scene0001_01':
            continue
        ps = [d for d in r['detections'] if d['label'] == 'person']
        if ps:
            b = max(ps, key=lambda d: d['score'])['box2d']
            centers[r['frame']] = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    ann = json.loads((ROOT / 'reports/bonn_fourarm_20260914/pt_annotation.json').read_text())
    ann_boxes = {int(k): v for k, v in ann['boxes'].items()}

    results = {}
    for k_px in (7, 15):
        rows = []
        prev = None
        for fr in sorted(centers):
            img = cv2.imread(str(frames / f'color/{fr}.jpg'))
            cx, cy = centers[fr]
            if prev is not None:
                dx, dy = cx - prev[0], cy - prev[1]
            else:
                dx, dy = 1.0, 0.0
            prev = (cx, cy)
            blurred = motion_blur(img, dx, dy, k_px)  # BGR throughout
            res = model.predict(source=[np.ascontiguousarray(blurred)], device='cuda:0',
                                conf=0.25, iou=0.7, imgsz=640, max_det=64,
                                agnostic_nms=True, verbose=False)[0]
            persons = []
            if res.boxes is not None:
                names = res.names
                for j in range(len(res.boxes)):
                    label = names[int(res.boxes.cls[j])] if isinstance(names, dict) else str(names[int(res.boxes.cls[j])])
                    if label == 'person':
                        persons.append((res.boxes.xyxy.cpu().numpy()[j].tolist(),
                                        float(res.boxes.conf[j])))
            def iou2d(a, b):
                ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
                iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
                inter = ix * iy
                u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
                return inter / max(u, 1e-9)
            best_iou = None
            if persons and fr in ann_boxes:
                ab = ann_boxes[fr]
                best_iou = round(max(iou2d(b[0], ab) for b in persons), 3)
            # hit requires an actual MATCH to the annotated person
            hit = best_iou is not None and best_iou >= 0.3
            if persons and fr in ann_boxes:
                ab = ann_boxes[fr]

                def iou2d(a, b):
                    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
                    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
                    inter = ix * iy
                    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
                    return inter / max(u, 1e-9)
                best_iou = round(max(iou2d(b[0], ab) for b in persons), 3)
            rows.append({'frame': fr, 'hit': hit,
                         'score': round(max((x[1] for x in persons), default=0.0), 2),
                         'iou2d_vs_vlm': best_iou})
        visible_rows = [r for r in rows if r['frame'] in VISIBLE]
        cov = sum(1 for r in visible_rows if r['hit'])
        run, best_run = 0, 0
        for r in visible_rows:
            run = run + 1 if r['hit'] else 0
            best_run = max(best_run, run)
        ious = [r['iou2d_vs_vlm'] for r in visible_rows if r['iou2d_vs_vlm'] is not None]
        results[f'{k_px}px'] = {
            'coverage': f'{cov}/13', 'coverage_frac': round(cov / 13, 3),
            'longest_consecutive': best_run,
            'gate_pass': bool(cov >= 7 and best_run >= 3),
            'mean_iou2d_vs_vlm': round(float(np.mean(ious)), 3) if ious else None,
            'rows': rows}
        print(k_px, 'px:', results[f'{k_px}px']['coverage'],
              'longest', best_run, 'mean_iou2d',
              results[f'{k_px}px']['mean_iou2d_vs_vlm'], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print('BLUR_TEST_DONE')


if __name__ == '__main__':
    main()
