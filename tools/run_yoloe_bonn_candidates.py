#!/usr/bin/env python3
"""Frozen YOLOE single-frame candidate export on the Bonn keyframes.

Runs the S2-frozen YOLOE-11s-seg-pf checkpoint (prompt-free, conf 0.25,
iou 0.70, imgsz 640, agnostic NMS) on the 62 sampled keyframes and exports,
for every detection: 2D box, class, score, mask depth validity, and a
visible-surface world AABB built from masked depth points (2%/98% quantiles).

Single-frame only: no cross-frame confirmation, no static-map fusion.  Mask
depth yields the VISIBLE surface of the person, not a full 3D human box.
Run inside the boxfusion-online environment (ultralytics + YOLOE).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CKPT = Path('/data/ZhaoX/OVM3D-Dett/boxfusion_b6_selective_boxer_dev/models/yoloe-11s-seg-pf.pt')
CONF, IOU, IMGSZ, MAXDET, MASK_THR = 0.25, 0.70, 640, 64, 0.50


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()

    from ultralytics import YOLOE
    model = YOLOE(str(CKPT))

    scenes = {'scene0001_01': 580, 'scene0002_01': 927}
    records = []
    for scene, n in scenes.items():
        frames = Path(f'{ROOT}/data_bonn/{scene}/frames')
        K = np.loadtxt(frames / 'intrinsic/intrinsic_depth.txt')[:3, :3]
        for fr in range(0, n, 25):
            img = np.asarray(Image.open(frames / f'color/{fr}.jpg').convert('RGB'))
            dep = np.asarray(Image.open(frames / f'depth/{fr}.png')).astype(np.float64) / 5000.0
            pose = np.loadtxt(frames / f'pose/{fr}.txt')
            t0 = time.perf_counter()
            results = model.predict(source=[np.ascontiguousarray(img[..., ::-1])],
                                    device=args.device, conf=CONF, iou=IOU,
                                    imgsz=IMGSZ, max_det=MAXDET,
                                    agnostic_nms=True, verbose=False)
            dt = time.perf_counter() - t0
            r = results[0]
            H, W = img.shape[:2]
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
            scores = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
            cls = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.zeros(0, int)
            names = r.names
            masks = (r.masks.data.cpu().numpy() >= MASK_THR) if r.masks is not None else np.zeros((0, IMGSZ, IMGSZ), bool)
            dets = []
            for k in range(len(boxes)):
                label = names[int(cls[k])] if isinstance(names, dict) else str(names[cls[k]])
                m = cv2.resize(masks[k].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                z = dep[m]
                valid = z[(z > 0.05) & (z < 12.0)]
                entry = {
                    'label': str(label), 'score': float(scores[k]),
                    'box2d': [float(v) for v in boxes[k]],
                    'mask_area_px': int(m.sum()),
                    'valid_depth_ratio': float(len(valid) / max(m.sum(), 1)),
                    'n_depth_points': int(len(valid)),
                }
                if len(valid) >= 8:
                    rows, cols = np.nonzero(m)
                    zu_all = dep[rows, cols]
                    sel = (zu_all > 0.05) & (zu_all < 12.0)
                    zu = zu_all[sel]
                    xs = (cols[sel] - K[0, 2]) / K[0, 0] * zu
                    ys = (rows[sel] - K[1, 2]) / K[1, 1] * zu
                    pc = np.stack([xs, ys, zu], 1)
                    pw = pc @ pose[:3, :3].T + pose[:3, 3]
                    lo = np.quantile(pw, 0.02, axis=0)
                    hi = np.quantile(pw, 0.98, axis=0)
                    entry['world_aabb_lo'] = lo.round(4).tolist()
                    entry['world_aabb_hi'] = hi.round(4).tolist()
                    entry['depth_median_m'] = float(np.median(zu))
                dets.append(entry)
            records.append({'scene': scene, 'frame': fr, 'n_dets': len(dets),
                            'infer_ms': round(dt * 1000, 1), 'detections': dets})
            print(scene, fr, 'dets:', len(dets),
                  'person:', sum(1 for d in dets if d['label'] == 'person'),
                  f'{dt*1000:.0f}ms', flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=1))
    total = sum(r['infer_ms'] for r in records)
    print(f'YOLOE_EXPORTED {len(records)} frames, mean {total/len(records):.1f} ms/frame')


if __name__ == '__main__':
    main()
