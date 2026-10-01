#!/usr/bin/env python3
"""Convert Bonn RGB-D Dynamic TUM-format sequences to the pipeline frames layout.

scene ids are assigned as scene0001_01 (person_tracking) / scene0002_01 (crowd)
because the online loader derives scene ids from a `scene\\d{4}_\\d{2}` regex in
the datadir.  Poses are interpolated from the mocap groundtruth onto every RGB
timestamp (nearest <= 20 ms, else linear/slerp interpolation between bracketing
samples).  Depth follows the TUM convention: 16-bit PNG, 5000 == 1 m.
"""
from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DL = ROOT / 'data_bonn_dl'
OUT = ROOT / 'data_bonn'
SEQS = {'rgbd_bonn_person_tracking': 'scene0001_01',
        'rgbd_bonn_crowd': 'scene0002_01'}
FX, FY, CX, CY = 542.822841, 542.576870, 315.593520, 237.756098


def quat_to_mat(qx, qy, qz, qw):
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
    return R


def quat_slerp(q0, q1, t):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.clip(q0 @ q1, -1.0, 1.0))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(dot)
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


def convert(seq, scene_id):
    src = DL / seq
    frames = OUT / scene_id / 'frames'
    if frames.exists():
        return f'{scene_id}: exists'
    rgb_dir, depth_dir = src / 'rgb', src / 'depth'
    rgb = sorted((float(p.stem), p) for p in rgb_dir.glob('*.png'))
    dep = sorted((float(p.stem), p) for p in depth_dir.glob('*.png'))
    dep_t = np.asarray([t for t, _ in dep])
    gt = []
    for line in (src / 'groundtruth.txt').read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        v = list(map(float, line.split()))
        gt.append(v)  # t tx ty tz qx qy qz qw
    gt = np.asarray(gt)
    gt_t = gt[:, 0]

    # nearest/interpolated pose per rgb timestamp
    def pose_at(t):
        i = int(np.searchsorted(gt_t, t))
        if i == 0:
            row0 = row1 = gt[0]; a = 0.0
        elif i >= len(gt_t):
            row0 = row1 = gt[-1]; a = 0.0
        else:
            row0, row1 = gt[i - 1], gt[i]
            a = (t - gt_t[i - 1]) / max(gt_t[i] - gt_t[i - 1], 1e-9)
        q = quat_slerp(row0[4:8], row1[4:8], a)
        pose = np.eye(4)
        pose[:3, :3] = quat_to_mat(*q)
        pose[:3, 3] = row0[1:4] * (1 - a) + row1[1:4] * a
        return pose

    def depth_for(t):
        j = int(np.searchsorted(dep_t, t))
        best, best_dt = None, 1e9
        for k in (j - 1, j):
            if 0 <= k < len(dep):
                dt = abs(dep_t[k] - t)
                if dt < best_dt:
                    best, best_dt = dep[k][1], dt
        return best if best_dt <= 0.02 else None

    for sub in ('color', 'depth', 'pose', 'intrinsic'):
        (frames / sub).mkdir(parents=True)
    K = np.array([[FX, 0, CX, 0], [0, FY, CY, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    np.savetxt(frames / 'intrinsic/intrinsic_color.txt', K, fmt='%.6f')
    np.savetxt(frames / 'intrinsic/intrinsic_depth.txt', K, fmt='%.6f')
    np.savetxt(frames / 'K_rgb.txt', K[:3, :3], fmt='%.10f')
    np.savetxt(frames / 'K_depth.txt', K[:3, :3], fmt='%.10f')
    n_used = 0
    skipped = 0
    for t, rp in rgb:
        dp = depth_for(t)
        if dp is None:
            skipped += 1
            continue
        i = n_used
        np.savetxt(frames / 'pose' / f'{i}.txt', pose_at(t), fmt='%.9f')
        img = Image.open(rp).convert('RGB')
        img.save(frames / 'color' / f'{i}.jpg', quality=92)
        shutil.copy(dp, frames / 'depth' / f'{i}.png')
        n_used += 1
    return (f'{scene_id}: {n_used} frames ({skipped} skipped without depth '
            f'within 10 ms), from {seq}')


def main():
    OUT.mkdir(exist_ok=True)
    mapping = {}
    for seq, scene_id in SEQS.items():
        zips = DL / f'{seq}.zip'
        if not (DL / seq).is_dir() and zips.is_file():
            with zipfile.ZipFile(zips) as z:
                z.extractall(DL)
        mapping[scene_id] = convert(seq, scene_id)
        print(mapping[scene_id], flush=True)
    (OUT / 'manifest.json').write_text(json.dumps(
        {'scene_to_sequence': {v: k for k, v in SEQS.items()},
         'log': mapping, 'depth_scale': 5000.0,
         'intrinsics': dict(fx=FX, fy=FY, cx=CX, cy=CY)}, indent=1))
    print('BONN_CONVERTED')


if __name__ == '__main__':
    main()
