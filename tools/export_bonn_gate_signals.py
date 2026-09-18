#!/usr/bin/env python3
"""Re-export frozen YOLOE Bonn candidates with truncation signals (v2).

Same frozen detector settings as v1; additionally computes, for every person
detection, the causal truncation signals (occluder depth discontinuity at the
mask boundary / border contact) used by the dynamic policy's 'truncation'
completion gate.  Run in the boxfusion-online environment.
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
sys.path.insert(0, str(ROOT))
from boxfusion.truncation_signals import truncation_signals

CKPT = '/data/ZhaoX/OVM3D-Dett/boxfusion_b6_selective_boxer_dev/models/yoloe-11s-seg-pf.pt'
SCENES = {'scene0001_01': 580, 'scene0002_01': 927}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    from ultralytics import YOLOE
    model = YOLOE(CKPT)
    records = []
    for scene, n in SCENES.items():
        frames = ROOT / 'data_bonn' / scene / 'frames'
        K = np.loadtxt(frames / 'intrinsic/intrinsic_depth.txt')[:3, :3]
        for fr in range(0, n, 25):
            img = np.asarray(Image.open(frames / f'color/{fr}.jpg').convert('RGB'))
            dep = np.asarray(Image.open(frames / f'depth/{fr}.png')).astype(np.float64) / 5000.0
            pose = np.loadtxt(frames / f'pose/{fr}.txt')
            t0 = time.perf_counter()
            r = model.predict(source=[np.ascontiguousarray(img[..., ::-1])],
                              device='cuda:0', conf=0.25, iou=0.7, imgsz=640,
                              max_det=64, agnostic_nms=True, verbose=False)[0]
            dt = (time.perf_counter() - t0) * 1000
            H, W = img.shape[:2]
            dets = []
            if r.boxes is not None and r.masks is not None:
                names = r.names
                boxes = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                masks = r.masks.data.cpu().numpy() > 0.5
                for k in range(len(boxes)):
                    label = names[int(cls[k])] if isinstance(names, dict) else str(names[cls[k]])
                    m = cv2.resize(masks[k].astype(np.uint8), (W, H),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                    valid = m & (dep > 0.05) & (dep < 12.0)
                    entry = {'label': str(label), 'score': float(scores[k]),
                             'box2d': [float(v) for v in boxes[k]],
                             'n_depth_points': int(valid.sum())}
                    if entry['n_depth_points'] >= 8 and label == 'person':
                        z = dep[valid]
                        pd = float(np.median(z))
                        entry['depth_median_m'] = pd
                        rows, cols = np.nonzero(valid)
                        zs = dep[rows, cols]
                        xs = (cols - K[0, 2]) / K[0, 0] * zs
                        ys2 = (rows - K[1, 2]) / K[1, 1] * zs
                        pc = np.stack([xs, ys2, zs], 1)
                        pw = pc @ pose[:3, :3].T + pose[:3, 3]
                        lo = np.quantile(pw, 0.02, axis=0)
                        hi = np.quantile(pw, 0.98, axis=0)
                        entry['world_aabb_lo'] = lo.round(4).tolist()
                        entry['world_aabb_hi'] = hi.round(4).tolist()
                        entry.update(truncation_signals(m, dep, entry['box2d'], pd, (W, H)))
                    dets.append(entry)
            records.append({'scene': scene, 'frame': fr, 'infer_ms': round(dt, 1),
                            'detections': dets})
            print(scene, fr, 'persons:',
                  sum(1 for d in dets if d['label'] == 'person'),
                  'trunc_below:',
                  sum(1 for d in dets if d.get('trunc_below')),
                  'trunc_above:', sum(1 for d in dets if d.get('trunc_above')),
                  flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=1))
    print('V2_EXPORTED', len(records))


if __name__ == '__main__':
    main()
