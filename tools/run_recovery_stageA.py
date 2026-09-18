#!/usr/bin/env python3
"""Stage A (boxfusion-online): YOLOE on 7px-blurred pt keyframes.

Saves per-frame person box2d + mask (PNG) + matched flag (IoU2D >= 0.3 vs the
adjudicated reference), so stage B can build causal prompts without touching
the detector environment again.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLOE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/recovery_gate2_20260914'
FRAMES = ROOT / 'data_bonn/scene0001_01/frames'


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


def iou2d(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(u, 1e-9)


def main():
    model = YOLOE('/data/ZhaoX/OVM3D-Dett/boxfusion_b6_selective_boxer_dev/models/yoloe-11s-seg-pf.pt')
    ann = json.loads((ROOT / 'reports/bonn_fourarm_20260914/pt_annotation.json').read_text())
    ann_boxes = {int(k): v for k, v in ann['boxes'].items()}
    cands = json.loads((ROOT / 'reports/bonn_detect_20260914/yoloe_candidates_v2.json').read_text())
    centers = {}
    for r in cands:
        if r['scene'] != 'scene0001_01':
            continue
        ps = [d for d in r['detections'] if d['label'] == 'person']
        if ps:
            b = max(ps, key=lambda d: d['score'])['box2d']
            centers[r['frame']] = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    OUT.mkdir(exist_ok=True)
    (OUT / 'masks').mkdir(exist_ok=True)
    records = []
    prev = None
    for fr in sorted(ann_boxes):
        img = cv2.imread(str(FRAMES / f'color/{fr}.jpg'))
        cx, cy = centers.get(fr, (320, 240))
        dx, dy = (cx - prev[0], cy - prev[1]) if prev else (1.0, 0.0)
        prev = (cx, cy)
        blurred = motion_blur(img, dx, dy, 7)
        res = model.predict(source=[np.ascontiguousarray(blurred)], device='cuda:0',
                            conf=0.25, iou=0.7, imgsz=640, max_det=64,
                            agnostic_nms=True, retina_masks=True, verbose=False)[0]
        best, best_mask, best_iou = None, None, 0.0
        if res.boxes is not None and res.masks is not None:
            names = res.names
            H, W = img.shape[:2]
            for j in range(len(res.boxes)):
                label = names[int(res.boxes.cls[j])] if isinstance(names, dict) else str(names[int(res.boxes.cls[j])])
                if label != 'person':
                    continue
                box = [float(v) for v in res.boxes.xyxy.cpu().numpy()[j]]
                m = res.masks.data[j].cpu().numpy() > 0.5
                if m.shape != (H, W):
                    m = cv2.resize(m.astype(np.uint8), (W, H),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                v = iou2d(box, ann_boxes[fr])
                if best is None or v > best_iou:
                    best, best_mask, best_iou = box, m, v
        rec = {'frame': fr,
               'hit': bool(best is not None and best_iou >= 0.3),
               'iou2d': round(best_iou, 3) if best else 0.0}
        if best is not None:
            rec['box2d'] = [round(v, 1) for v in best]
            Image.fromarray((best_mask * 255).astype(np.uint8)).save(
                OUT / 'masks' / f'{fr}.png')
        records.append(rec)
        print(rec, flush=True)
    (OUT / 'stageA_yoloe_blur7.json').write_text(json.dumps(records, indent=1))
    print('STAGE_A_DONE')


if __name__ == '__main__':
    main()
