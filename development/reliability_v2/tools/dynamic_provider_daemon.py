#!/usr/bin/env python3
"""Person front-end daemon for the online pipeline (boxfusion-online env).

Reads JSON requests from stdin, one per keyframe:
  {"frame": int, "color": path, "depth": path, "depth_scale": float,
   "K": 3x3, "pose": 4x4, "classes": ["person"]}
Writes one JSON reply:
  {"persons": [{"center":[3], "lo":[3], "hi":[3], "score": float,
                "n_px": int, "depth_m": float,
                "trunc_below": bool, "trunc_above": bool}]}

Frozen YOLOE-11s-seg-pf, prompt-free, conf 0.25 / iou 0.70 / imgsz 640 /
max_det 64 / agnostic NMS.  Colour is resized onto the depth grid (the same
convention as the online loader).  Observations are world-frame visible-
surface AABBs (2/98 quantile) with causal truncation signals.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from boxfusion.truncation_signals import truncation_signals

CKPT = '/data/ZhaoX/OVM3D-Dett/boxfusion_b6_selective_boxer_dev/models/yoloe-11s-seg-pf.pt'


def main():
    from ultralytics import YOLOE
    model = YOLOE(CKPT)
    print(json.dumps({'ready': True}), flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        color = cv2.imread(req['color'])
        depth_raw = cv2.imread(req['depth'], cv2.IMREAD_UNCHANGED)
        if color is None or depth_raw is None:
            print(json.dumps({'persons': [], 'error': 'io'}), flush=True)
            continue
        depth = depth_raw.astype(np.float64) / float(req.get('depth_scale', 1000.0))
        H, W = depth.shape
        color = cv2.resize(color, (W, H))
        K = np.asarray(req['K'], float)
        pose = np.asarray(req['pose'], float)
        classes = set(req.get('classes', ['person']))
        res = model.predict(source=[np.ascontiguousarray(color)],  # NumPy input is BGR
                            device=req.get('device', 'cuda:0'), conf=0.25,
                            iou=0.7, imgsz=640, max_det=64,
                            agnostic_nms=True, verbose=False)[0]
        persons = []
        if res.boxes is not None and res.masks is not None:
            names = res.names
            boxes = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            masks = res.masks.data.cpu().numpy() > 0.5
            for k in range(len(boxes)):
                label = names[int(cls[k])] if isinstance(names, dict) else str(names[cls[k]])
                if str(label).lower() not in classes:
                    continue
                m = cv2.resize(masks[k].astype(np.uint8), (W, H),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
                valid = m & (depth > 0.05) & (depth < 12.0)
                n = int(valid.sum())
                if n < 8:
                    continue
                rows, cols = np.nonzero(valid)
                z = depth[rows, cols]
                xs = (cols - K[0, 2]) / K[0, 0] * z
                ys = (rows - K[1, 2]) / K[1, 1] * z
                pc = np.stack([xs, ys, z], 1)
                pw = pc @ pose[:3, :3].T + pose[:3, 3]
                lo = np.quantile(pw, 0.02, axis=0)
                hi = np.quantile(pw, 0.98, axis=0)
                pd = float(np.median(z))
                sig = truncation_signals(m, depth, [float(v) for v in boxes[k]], pd, (W, H))
                persons.append({
                    'center': ((lo + hi) / 2).tolist(),
                    'lo': lo.tolist(), 'hi': hi.tolist(),
                    'score': float(scores[k]), 'n_px': n, 'depth_m': pd,
                    'trunc_below': sig['trunc_below'],
                    'trunc_above': sig['trunc_above'],
                })
        print(json.dumps({'persons': persons}), flush=True)


if __name__ == '__main__':
    main()
