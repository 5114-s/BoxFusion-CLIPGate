#!/usr/bin/env python3
"""Dense separation v2b: view-consistency model selection (GT-free).

For every final row with a tight multi-keyframe candidate cluster in its
neighbourhood, compare two 3D hypotheses by how well their PER-FRAME
projections explain the 2D observation boxes of the cluster members:

  H0 = the row's fused box;   H1 = the cluster representative box
       (highest-score member).

H1 wins by >= margin on mean per-frame 2D IoU  ->  replace the row geometry
(keep the row's score).  Split action as in v1 (mutually separated
multi-member clusters) with per-cluster representatives.

This is the fusion objective itself, evaluated causally as a post-hoc
model-selection test; no GT anywhere.
"""
from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / 'tools'))
from eval_ca1m import evaluate

CAPTURE = ROOT / 'reports/ca1m_lifting_factors_full107_20260908/capture'
GT_ROOT = Path('/extra/ZhaoX/boxfusion_ca1m'
               if False else '/extra/ZhaoX/boxfusion_ca1m')
M2 = ROOT / 'results/ca1m_thr15_m1_m2nl_full107'
OUT = ROOT / 'reports/dense_separation_v2b_20260915'
MARGIN = 0.05


def aabb(c):
    return c.min(1), c.max(1)


def iou_mat(lo_a, hi_a, lo_b, hi_b):
    inter = np.maximum(0, np.minimum(hi_a[:, None], hi_b[None]) -
                       np.maximum(lo_a[:, None], lo_b[None])).prod(2)
    va = (hi_a - lo_a).prod(1)
    vb = (hi_b - lo_b).prod(1)
    return inter / np.maximum(va[:, None] + vb[None] - inter, 1e-9)


def corners_from_aabb(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0])
                     for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)


def project_corners(c8, K, pose):
    Rt = np.linalg.inv(pose)
    pc = (Rt[:3, :3] @ c8.T).T + Rt[:3, 3]
    uv = (K @ pc.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    return [float(uv[:, 0].min()), float(uv[:, 1].min()),
            float(uv[:, 0].max()), float(uv[:, 1].max())]


def iou2d(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(u, 1e-9)


def main():
    scenes = sorted(s.stem for s in CAPTURE.glob('*.npz')
                    if (M2 / f'{s.stem}_boxes.pkl').is_file()
                    and (GT_ROOT / s.stem / 'after_filter_boxes.npy').is_file())
    OUT.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    for scene in scenes:
        with open(M2 / f'{scene}_boxes.pkl', 'rb') as f:
            rows = list(pickle.load(f)[0])
        z = np.load(CAPTURE / f'{scene}.npz')
        cand = z['corners']
        cand_scores = z['scores']
        cand_frames = z['frame_ids']
        boxes2d = z['boxes2d']
        K = z['K_rgb'].astype(np.float64)
        # poses: all_poses.npy rows are indexed by raw frame id (gap=20 kfs)
        ap = np.load(f'{GT_ROOT}/{scene}/all_poses.npy')
        poses = {int(f): ap[int(f)] for f in set(cand_frames.tolist())
                 if int(f) < len(ap)}
        if not len(cand):
            with open(OUT / f'{scene}_boxes.pkl', 'wb') as f:
                pickle.dump([rows], f)
            continue
        c_lo, c_hi = aabb(cand)
        centers = (c_lo + c_hi) / 2
        f_arr = np.asarray([r[1] for r in rows]) if rows else np.zeros((0, 8, 3))
        r_lo, r_hi = aabb(f_arr)
        iou_rc = iou_mat(r_lo, r_hi, c_lo, c_hi)

        new_rows = []
        for k, row in enumerate(rows):
            R_lo, R_hi = r_lo[k], r_hi[k]
            diag = float(np.linalg.norm(R_hi - R_lo))
            nb = np.flatnonzero(iou_rc[k] >= 0.3)
            if len(nb) < 2:
                new_rows.append(row)
                continue
            radius = 0.30 * diag
            order = sorted(nb.tolist(), key=lambda i: -cand_scores[i])
            clusters = []
            for i in order:
                for cl in clusters:
                    if any(np.linalg.norm(centers[i] - centers[m]) <= radius for m in cl):
                        cl.append(i)
                        break
                else:
                    clusters.append([i])
            big = [cl for cl in clusters if len(cl) >= 2
                   and len({int(cand_frames[m]) for m in cl}) >= 2]
            # SPLIT (same as v1)
            if len(big) >= 2:
                cents = [centers[cl].mean(0) for cl in big]
                if all(np.linalg.norm(cents[i] - cents[j]) >= 0.30 * diag
                       for i in range(len(big)) for j in range(i + 1, len(big))):
                    big.sort(key=len, reverse=True)
                    for ci, cl in enumerate(big):
                        rep = max(cl, key=lambda i: cand_scores[i])
                        sc = row[2] if ci == 0 else float(cand_scores[rep])
                        new_rows.append((row[0], cand[rep], sc))
                    counts['split_rows'] += len(big)
                    counts['split_events'] += 1
                    continue
            # RESCUE via view-consistency model selection
            main_cl = max(big or clusters, key=len)
            if len(main_cl) >= 2 and len({int(cand_frames[m]) for m in main_cl}) >= 2:
                spread = float(np.linalg.norm(
                    centers[main_cl] - centers[main_cl].mean(0), axis=1).max())
                if spread <= 0.35 * diag:
                    rep = max(main_cl, key=lambda i: cand_scores[i])
                    # per-frame evaluation of both hypotheses
                    per_frame = {}
                    for m in main_cl:
                        per_frame.setdefault(int(cand_frames[m]), []).append(m)
                    s0, s1, n_fr = 0.0, 0.0, 0
                    for fr, ms in per_frame.items():
                        pose = poses.get(fr)
                        if pose is None:
                            continue
                        n_fr += 1
                        b2 = boxes2d[ms[0]]
                        p0 = project_corners(f_arr[k], K, pose)
                        p1 = project_corners(cand[rep], K, pose)
                        s0 += iou2d(p0, b2)
                        s1 += iou2d(p1, b2)
                    if n_fr >= 2:
                        m0, m1 = s0 / n_fr, s1 / n_fr
                        if m1 > m0 + MARGIN:
                            new_rows.append((row[0], cand[rep], row[2]))
                            counts['rescue_rows'] += 1
                            counts['rescue_margin_sum'] += round(m1 - m0, 3)
                            continue
            new_rows.append(row)
        with open(OUT / f'{scene}_boxes.pkl', 'wb') as f:
            pickle.dump([new_rows], f)
        if (scenes.index(scene) + 1) % 30 == 0:
            print(f'{scenes.index(scene)+1}/{len(scenes)}', dict(counts), flush=True)

    aps = evaluate(str(GT_ROOT), str(OUT), scenes)
    result = {'AP15': round(aps[0] * 100, 2), 'AP25': round(aps[1] * 100, 2),
              'AP50': round(aps[2] * 100, 2), 'counts': dict(counts)}
    (OUT / 'results.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
