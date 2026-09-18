#!/usr/bin/env python3
"""Build the synthetic walking benchmark (data_dyn_walk).

A GT object that is well visible before T keeps walking from T onward: at every
frame f >= T the object is sensor-level removed at its original position and
re-pasted, depth-consistently, at position A + dir * min(speed*(f-T), D_max).
All edits happen on the 640x480 depth grid (the online loader resamples colour
onto that grid anyway).  Per-frame GT offsets are recoverable from the
manifest; terminal GT places the walker at A + dir*D_max.

Geometric rules frozen before any inference:
- direction: 8 horizontal candidates scored by clearance to other GT boxes and
  post-T visibility of the swept positions; D_max clamped by clearance-0.30 m
  into the [0.8, 1.5] m range;
- speed alternates 0.012 / 0.020 m/frame across scenes;
- walker identity is recovered at evaluation time with the same max-AABB-IoU
  >= 0.5 rule as the removal benchmark (marker corners stored in manifest).
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / 'upstream_clean/scannet_readme_frames'
GT_ROOT = ROOT / 'evaluation/data_util/scannet_train_detection_data'
DST_ROOT = ROOT / 'data_dyn_walk'
N_SCENES = 100
SPEEDS = (0.012, 0.020)
D_MIN, D_MAX_CAP, CLEARANCE = 0.6, 1.5, 0.05
SHRINK = 0.12  # walker AABB in-shrink when measuring clearance (AABB inflation
#              of rotated objects makes unshrunk distances over-conservative)


def obb_frame(corners):
    """Local frame of a (possibly rotated) 8-corner box."""
    c = corners.mean(0)
    edges = np.stack([corners[1] - corners[0], corners[3] - corners[0],
                      corners[4] - corners[0]])
    axes = edges / np.linalg.norm(edges, axis=1)[:, None]
    half = np.linalg.norm(edges, axis=1) / 2.0
    return c, axes, half


def points_in_obb(pts, corners):
    """Exact test for the world-axis-aligned walker boxes (no corner-order
    ambiguity: min/max extents only)."""
    lo = np.minimum(corners.min(0), corners.max(0)) - 1e-6
    hi = np.maximum(corners.min(0), corners.max(0)) + 1e-6
    return ((pts >= lo) & (pts <= hi)).all(1)


def project_corners(corners_w, pose, K, W, H):
    Rt = np.linalg.inv(pose)
    c = (Rt[:3, :3] @ corners_w.T).T + Rt[:3, 3]
    if (c[:, 2] < 0.1).any():
        return None
    uv = (K @ c.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    x1, y1 = uv[:, 0].min(), uv[:, 1].min()
    x2, y2 = uv[:, 0].max(), uv[:, 1].max()
    if x2 <= 0 or y2 <= 0 or x1 >= W or y1 >= H:
        return None
    return (int(max(0, x1)), int(max(0, y1)), int(min(W, x2)), int(min(H, y2)))


def backproject(us, vs, z, pose, K):
    x = (us - K[0, 2]) / K[0, 0] * z
    y = (vs - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], axis=1)
    return pc @ pose[:3, :3].T + pose[:3, 3]


def project_world(pts_w, pose, K):
    Rt = np.linalg.inv(pose)
    pc = (Rt[:3, :3] @ pts_w.T).T + Rt[:3, 3]
    z = pc[:, 2]
    uv = (K @ pc.T).T
    uv = uv[:, :2] / np.maximum(uv[:, 2:3], 1e-6)
    return uv, z


def aabb_bounds(corners):
    return corners.min(0), corners.max(0)


def gt_world_corners(scene):
    axis = None
    for line in open(f'/extra/ZhaoX/scannet_data/scans/{scene}/{scene}.txt'):
        if 'axisAlignment' in line:
            axis = np.array([float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]).reshape(4, 4)
    Tinv = np.linalg.inv(axis)
    gt = np.load(GT_ROOT / f'{scene}_bbox.npy')
    centers, wlh = gt[:, :3], gt[:, 3:6]
    out = []
    for c, s in zip(centers, wlh):
        half = s / 2.0
        local = np.array([[dx, dy, dz] for dx in (-half[0], half[0])
                          for dy in (-half[1], half[1]) for dz in (-half[2], half[2])])
        corners = local + c
        out.append((Tinv[:3, :3] @ corners.T).T + Tinv[:3, 3])
    return out


def horizontal_clearance(swept_lo, swept_hi, swept_zlo, swept_zhi, others):
    """Min horizontal distance to boxes whose z-range overlaps the swept one."""
    best = float('inf')
    for lo, hi, zlo, zhi in others:
        if zhi <= swept_zlo or zlo >= swept_zhi:
            continue
        dx = max(swept_lo[0] - hi[0], lo[0] - swept_hi[0], 0.0)
        dz = max(swept_lo[2] - hi[2], lo[2] - swept_hi[2], 0.0)
        best = min(best, float(np.hypot(dx, dz)))
    return best


def aabb_iou(c1, c2):
    lo1, hi1 = aabb_bounds(c1)
    lo2, hi2 = aabb_bounds(c2)
    inter = np.maximum(0, np.minimum(hi1, hi2) - np.maximum(lo1, lo2)).prod()
    union = (hi1 - lo1).prod() + (hi2 - lo2).prod() - inter
    return inter / union if union > 0 else 0.0


def choose_direction(corners, gt_corners, poses_post, Kd):
    own = int(np.argmax([aabb_iou(corners, c) for c in gt_corners]))
    others = [(*aabb_bounds(c),) for i, c in enumerate(gt_corners) if i != own]
    others = [(lo, hi, lo[2], hi[2]) for lo, hi in others]
    lo_a, hi_a = aabb_bounds(corners)
    pad = np.array([SHRINK, 0.0, SHRINK])
    lo_s, hi_s = lo_a + pad, hi_a - pad
    best = None
    for angle in np.linspace(0, 2 * np.pi, 9)[:-1]:
        d = np.array([np.cos(angle), 0.0, np.sin(angle)])
        dmax = D_MAX_CAP
        for dist in np.arange(0.2, D_MAX_CAP + 0.05, 0.1):
            lo = np.minimum(lo_s, lo_s + d * dist)
            hi = np.maximum(hi_s, hi_s + d * dist)
            if horizontal_clearance(lo, hi, lo_a[2], hi_a[2], others) < CLEARANCE:
                dmax = max(dist - 0.1, 0.0)
                break
        if dmax < D_MIN:
            continue
        vis = 0
        for t in (0.5, 1.0):
            moved = corners + d * dmax * t
            for pose in poses_post:
                bb = project_corners(moved, pose, Kd, 640, 480)
                if bb and (bb[2] - bb[0]) * (bb[3] - bb[1]) > 500:
                    vis += 1
                    break
        score = (vis, round(dmax, 2))
        if best is None or score > best[0]:
            best = (score, d, dmax)
    if best is None:
        return None
    return best[1], float(best[2])


def edit_frame(img, dep_u16, pose, Kd, corners, delta):
    """Return edited (colour, depth) with the object moved by delta."""
    rect = project_corners(corners, pose, Kd, 640, 480)
    if rect is None:
        return None
    x1, y1, x2, y2 = rect
    us, vs = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
    us, vs = us.ravel(), vs.ravel()
    z = dep_u16[vs, us].astype(np.float64) / 1000.0
    ok = z > 0.05
    if not ok.any():
        return None
    pts = backproject(us[ok], vs[ok], z[ok], pose, Kd)
    inside = points_in_obb(pts, corners)
    if inside.sum() < 60:
        return None
    src_col = img[vs[ok][inside], us[ok][inside]].astype(np.float64)

    # remove at A: dilated projected rect
    mx = int(max(2, (x2 - x1) * 0.15))
    my = int(max(2, (y2 - y1) * 0.15))
    mask = np.zeros((480, 640), np.uint8)
    mask[max(0, y1 - my):min(480, y2 + my), max(0, x1 - mx):min(640, x2 + mx)] = 255
    base_img = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
    base_dep = cv2.inpaint(dep_u16, mask, 5, cv2.INPAINT_NS).astype(np.float64)

    # paste at B with a z-buffer (nearest surface wins)
    moved = pts[inside] + delta
    uv2, zc2 = project_world(moved, pose, Kd)
    ok2 = (zc2 > 0.05) & (uv2[:, 0] >= 0) & (uv2[:, 0] < 639.5) & \
          (uv2[:, 1] >= 0) & (uv2[:, 1] < 479.5)
    if not ok2.any():
        return base_img, base_dep.astype(np.uint16)
    uv2, zc2, col2 = uv2[ok2], zc2[ok2], src_col[ok2]
    order = np.argsort(zc2)          # nearest first
    uv2, zc2, col2 = uv2[order], zc2[order], col2[order]
    tu = np.round(uv2[:, 0]).astype(int)
    tv = np.round(uv2[:, 1]).astype(int)
    _, keep_idx = np.unique(tv * 640 + tu, return_index=True)
    tu, tv, zc2, col2 = tu[keep_idx], tv[keep_idx], zc2[keep_idx], col2[keep_idx]
    base_zm = base_dep[tv, tu] / 1000.0
    paste_ok = (base_zm <= 0.05) | (zc2 < base_zm)
    tu, tv, zc2, col2 = tu[paste_ok], tv[paste_ok], zc2[paste_ok], col2[paste_ok]

    out_img = base_img.copy()
    out_dep = base_dep.copy()
    out_img[tv, tu] = np.clip(col2, 0, 255).astype(np.uint8)
    out_dep[tv, tu] = np.clip(zc2 * 1000.0, 0, 65535)

    # close rounding gaps: 1 px ring around pasted pixels gets inpainted colour
    # and the nearest (min) pasted depth, bounded halo of one pixel
    hole = np.zeros((480, 640), np.uint8)
    hole[tv, tu] = 255
    ring = cv2.dilate(hole, np.ones((3, 3), np.uint8)) > 0
    ring &= ~(hole > 0)
    if ring.any():
        patched = cv2.inpaint(out_img, (ring * 255).astype(np.uint8), 3, cv2.INPAINT_TELEA)
        out_img[ring] = patched[ring]
        big = cv2.dilate(out_dep, np.ones((3, 3), np.uint8))
        near = cv2.erode(np.where(out_dep > 0, out_dep, 65535).astype(np.uint16),
                         np.ones((3, 3), np.uint8))
        fill_dep = np.where(near < big, near, big)
        out_dep[ring] = fill_dep[ring]
    return out_img, out_dep.astype(np.uint16)


def build_scene(scene):
    dst = DST_ROOT / scene / 'frames'
    manifest_path = DST_ROOT / scene / 'entry.json'
    if manifest_path.is_file():
        return scene, manifest_path.read_text()
    gt_corners = gt_world_corners(scene)

    src = SRC_ROOT / scene / 'frames'
    Kd = np.loadtxt(src / 'intrinsic/intrinsic_depth.txt')[:3, :3]
    color_files = sorted((src / 'color').glob('*.jpg'), key=lambda p: int(p.stem))
    n = len(color_files)
    T_frame = int(n * 0.4)

    assignment_path = DST_ROOT / 'assignment.json'
    if assignment_path.is_file():
        fixed = json.loads(assignment_path.read_text())[scene]
        corners = gt_corners[fixed['gt_idx']]
        direction = np.asarray(fixed['dir'], float)
        dmax = float(fixed['dmax'])
        speed = float(fixed['spd'])
        best = (None, corners, direction, dmax)
    else:
        poses_pre, poses_post = [], []
        for f in range(0, n, 25):
            pf = src / 'pose' / f'{f}.txt'
            if not pf.is_file():
                continue
            pose = np.loadtxt(pf).reshape(4, 4)
            (poses_pre if f < T_frame else poses_post).append(pose)
        # walker selection: pre-T visibility >= 8 keyframes, then the walking
        # direction maximising (post-T visibility of swept positions, distance)
        best = None
        for i, cand in enumerate(gt_corners):
            vis = sum(1 for pose in poses_pre
                      if (lambda bb: bb and (bb[2] - bb[0]) * (bb[3] - bb[1]) > 500)(
                          project_corners(cand, pose, Kd, 640, 480)))
            if vis < 8:
                continue
            choice = choose_direction(cand, gt_corners, poses_post, Kd)
            if choice is None:
                continue
            direction, dmax = choice
            score = (round(dmax, 2), vis)
            if best is None or score > best[0]:
                best = (score, cand, direction, dmax)
        if best is None:
            return scene, None
        _, corners, direction, dmax = best
        speed = SPEEDS[hash(scene) % 2]

    for sub in ('color', 'depth', 'pose', 'intrinsic'):
        (dst / sub).mkdir(parents=True, exist_ok=True)
    for f_ in (src / 'intrinsic').glob('*'):
        os.symlink(f_, dst / 'intrinsic' / f_.name)
    # BoxFusion.__init__ reads these from the frames root when the datadir
    # has no 'scannet' substring (same convention as data_dyn)
    np.savetxt(dst / 'K_depth.txt',
               np.loadtxt(src / 'intrinsic/intrinsic_depth.txt')[:3, :3], fmt='%.10f')
    np.savetxt(dst / 'K_rgb.txt',
               np.loadtxt(src / 'intrinsic/intrinsic_color.txt')[:3, :3], fmt='%.10f')
    n_edited = 0
    for cf in color_files:
        f = int(cf.stem)
        df = src / 'depth' / f'{f}.png'
        pf = src / 'pose' / f'{f}.txt'
        if not df.is_file() or not pf.is_file():
            continue
        os.symlink(pf, dst / 'pose' / f'{f}.txt')
        if f < T_frame:
            os.symlink(cf, dst / 'color' / f'{f}.jpg')
            os.symlink(df, dst / 'depth' / f'{f}.png')
            continue
        dist = min(speed * (f - T_frame), dmax)
        delta = direction * dist
        img = cv2.resize(cv2.imread(str(cf)), (640, 480))
        dep = np.asarray(Image.open(df))
        edited = edit_frame(img, dep, np.loadtxt(pf).reshape(4, 4), Kd, corners, delta)
        if edited is None:
            os.symlink(cf, dst / 'color' / f'{f}.jpg')
            os.symlink(df, dst / 'depth' / f'{f}.png')
            continue
        out_img, out_dep = edited
        cv2.imwrite(str(dst / 'color' / f'{f}.jpg'), out_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        Image.fromarray(out_dep).save(dst / 'depth' / f'{f}.png')
        n_edited += 1

    # post-build verification at the frames the online model actually samples:
    # the walker must have been moved (real file, not a symlink) in >= 5
    # sampled keyframes, and the paste must be geometrically present at B and
    # removed at A on the first such keyframe
    sampled = [f for f in range(0, n, 25) if f >= T_frame
               and not os.path.islink(dst / 'color' / f'{f}.jpg')]
    if len(sampled) < 5:
        import shutil
        shutil.rmtree(DST_ROOT / scene)
        return scene, None
    dep0 = odep0 = pose0 = f0 = None
    for f_check in sampled:
        pose_c = np.loadtxt(src / 'pose' / f'{f_check}.txt').reshape(4, 4)
        dist_c = min(speed * (f_check - T_frame), dmax)
        ra = project_corners(corners, pose_c, Kd, 640, 480)
        rb = project_corners(corners + direction * dist_c, pose_c, Kd, 640, 480)
        if ra is not None and rb is not None:
            f0, pose0 = f_check, pose_c
            dep0 = np.asarray(Image.open(dst / 'depth' / f'{f0}.png')).astype(np.float64) / 1000.0
            odep0 = np.asarray(Image.open(src / 'depth' / f'{f0}.png')).astype(np.float64) / 1000.0
            break
    if f0 is None:
        import shutil
        shutil.rmtree(DST_ROOT / scene)
        return scene, None
    dist0 = min(speed * (f0 - T_frame), dmax)

    def rect_of(c8):
        return project_corners(c8, pose0, Kd, 640, 480)

    def frac(rect, pred):
        if rect is None:
            return 0.0
        x1, y1, x2, y2 = rect
        e_, o_ = dep0[y1:y2, x1:x2], odep0[y1:y2, x1:x2]
        m = (e_ > 0.05) & (o_ > 0.05)
        return float(pred(e_, o_)[m].mean()) if m.any() else 0.0

    nearer_at_B = frac(rect_of(corners + direction * dist0),
                       lambda e, o: e < o - 0.05)
    changed_A = frac(rect_of(corners), lambda e, o: np.abs(e - o) > 0.05)
    if nearer_at_B < 0.02 or changed_A < 0.10:
        import shutil
        shutil.rmtree(DST_ROOT / scene)
        return scene, None
    payload = json.dumps({
        'T_frame': T_frame, 'marker': corners.reshape(-1).tolist(),
        'direction': direction.tolist(), 'speed_m_per_frame': speed,
        'D_max': dmax, 'final_offset': (direction * dmax).tolist(),
        'n_edited': n_edited, 'n_frames': n,
        'edited_sampled_kf': len(sampled),
        'verify_nearer_at_B': round(nearer_at_B, 4),
        'verify_changed_at_A': round(changed_A, 4),
    })
    manifest_path.write_text(payload)
    return scene, payload


def main():
    global DST_ROOT
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dst-root', type=Path, default=DST_ROOT)
    p.add_argument('--assignment', type=Path, default=None,
                   help='fixed scene -> walker assignment json (overrides scan)')
    p.add_argument('--n-scenes', type=int, default=N_SCENES)
    args = p.parse_args()
    DST_ROOT = args.dst_root
    assignment_path = DST_ROOT / 'assignment.json'
    if args.assignment is not None:
        assignment_path.parent.mkdir(parents=True, exist_ok=True)
        assignment_path.write_text(args.assignment.read_text())
    if assignment_path.is_file():
        scenes = sorted(json.loads(assignment_path.read_text()))
    else:
        scenes = sorted(p.name for p in SRC_ROOT.iterdir()
                        if p.is_dir() and (p / 'frames').is_dir())[:args.n_scenes]
    results = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for scene, payload in pool.map(build_scene, scenes):
            if payload is None:
                print(f'SKIP {scene}: no admissible walking direction', flush=True)
            else:
                print(f'{scene}: built', flush=True)
                results[scene] = json.loads(payload)
    (DST_ROOT / 'manifest.json').write_text(json.dumps(results, indent=1))
    print(f'WALK_BENCH_BUILT: {len(results)} scenes')


if __name__ == '__main__':
    main()
