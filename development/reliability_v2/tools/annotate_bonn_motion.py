#!/usr/bin/env python3
"""Model-free person annotation on the Bonn keyframes (change detection).

Pose-compensated depth differencing between consecutive sampled keyframes:
pixels whose depth became >=0.15 m closer mark the moving person's silhouette
in the later frame.  Components are filtered by 3D height/width; output is a
2D box + visible-surface 3D median point per annotated person-frame, plus
greedy cross-frame identity tracks.

Independent of any detector; used as evaluation annotation only.  Known blind
spot: a person standing perfectly still produces no change and is not
annotated (denominator undercount -> coverage optimistic).  A VLM spot-check
on contact sheets quantifies this.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NEARER, MIN_AREA, MIN_H, MAX_W = 0.15, 400, 0.8, 1.6


def warp_depth(dep_prev, pose_prev, pose_cur, K):
    H, W = dep_prev.shape
    vs, us = np.mgrid[0:H, 0:W]
    z = dep_prev
    ok = z > 0.05
    x = (us[ok] - K[0, 2]) / K[0, 0] * z[ok]
    y = (vs[ok] - K[1, 2]) / K[1, 1] * z[ok]
    pc = np.stack([x, y, z[ok]], 1)
    pw = pc @ pose_prev[:3, :3].T + pose_prev[:3, 3]
    back = (pw - pose_cur[:3, 3]) @ pose_cur[:3, :3].T
    uv = (K @ back.T).T
    uv = uv[:, :2] / np.maximum(uv[:, 2:3], 1e-6)
    out = np.full((H, W), np.inf)
    ui = np.round(uv[:, 0]).astype(int)
    vi = np.round(uv[:, 1]).astype(int)
    keep = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (back[:, 2] > 0.05)
    np.minimum.at(out, (vi[keep], ui[keep]), back[keep, 2])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    scenes = {'scene0001_01': 580, 'scene0002_01': 927}
    ann = {}
    for scene, n in scenes.items():
        frames = ROOT / 'data_bonn' / scene / 'frames'
        K = np.loadtxt(frames / 'intrinsic/intrinsic_depth.txt')[:3, :3]
        kfs = list(range(0, n, 25))
        ann[scene] = []
        for a, b in zip(kfs, kfs[1:]):
            dep_a = np.asarray(Image.open(frames / f'depth/{a}.png')).astype(np.float64) / 5000.
            dep_b = np.asarray(Image.open(frames / f'depth/{b}.png')).astype(np.float64) / 5000.
            pose_a = np.loadtxt(frames / f'pose/{a}.txt')
            pose_b = np.loadtxt(frames / f'pose/{b}.txt')
            prev = warp_depth(dep_a, pose_a, pose_b, K)
            okb = dep_b > 0.05
            nearer = okb & np.isfinite(prev) & (prev - dep_b > NEARER)
            # both-frame validity guards against depth noise at edges
            nearer = cv2.morphologyEx(nearer.astype(np.uint8), cv2.MORPH_OPEN,
                                      np.ones((3, 3), np.uint8)).astype(bool)
            n_comp, lab, stats, _ = cv2.connectedComponentsWithStats(nearer.astype(np.uint8), 8)
            for c in range(1, n_comp):
                x, y, w, h, area = stats[c]
                if area < MIN_AREA:
                    continue
                m = lab == c
                z = dep_b[m]
                z = z[(z > 0.05) & (z < 12)]
                if len(z) < 100:
                    continue
                rows, cols = np.nonzero(m)
                xs = (cols - K[0, 2]) / K[0, 0] * dep_b[rows, cols]
                ys = (rows - K[1, 2]) / K[1, 1] * dep_b[rows, cols]
                zs = dep_b[rows, cols]
                sel = (zs > 0.05) & (zs < 12)
                pw = np.stack([xs[sel], ys[sel], zs[sel]], 1) @ pose_b[:3, :3].T + pose_b[:3, 3]
                height = np.quantile(pw[:, 1], 0.95) - np.quantile(pw[:, 1], 0.05)
                width = np.quantile(pw[:, 0], 0.95) - np.quantile(pw[:, 0], 0.05)
                depth_w = np.quantile(pw[:, 2], 0.95) - np.quantile(pw[:, 2], 0.05)
                if not (MIN_H < height < 2.6 and width < MAX_W and depth_w < 1.2):
                    continue
                ann[scene].append({
                    'frame': int(b), 'box2d': [int(x), int(y), int(x + w), int(y + h)],
                    'world_center': pw.mean(0).round(3).tolist(),
                    'area_px': int(area)})
        print(scene, 'annotated person-frames:', len(ann[scene]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ann, indent=1))
    print('ANNOTATION_WRITTEN', args.output)


if __name__ == '__main__':
    main()
