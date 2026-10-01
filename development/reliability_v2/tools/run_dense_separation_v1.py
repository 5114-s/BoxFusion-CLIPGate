#!/usr/bin/env python3
"""Dense-neighborhood observation-to-instance assignment, v1 (GT-free).

Runs on frozen artifacts (final M1+M2 rows + per-keyframe candidate capture),
mimicking what the online fusion stage could compute from its own row/observation
membership.  For every final row R:

  neighborhood N = candidates with AABB IoU >= 0.3 against R;
  cluster N greedily by world-centre distance (radius = 0.30 x R diagonal);
  SPLIT   - >= 2 clusters of >= 2 members each, mutually separated by
            >= 0.30 x R diagonal: emit one row per cluster (geometry =
            highest-score member; biggest cluster keeps R's score, others
            use their member max real score);
  RESCUE  - single tight cluster whose median extent < 0.65 x R extent on
            some axis (row inflated between neighbours): replace R geometry
            with the highest-score member box, keep R's score.

Acceptance (pre-registered): >= 30% of oracle-B's AP25/AP50 gains
(+0.72 AP25 / +0.99 AP50) with AP15 loss <= 0.5.
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
GT_ROOT = Path('/extra/ZhaoX/boxfusion_ca1m')
M2 = ROOT / 'results/ca1m_thr15_m1_m2nl_full107'
OUT = ROOT / 'reports/dense_separation_v1_20260915'


def aabb(c):
    return c.min(1), c.max(1)


def iou_mat(lo_a, hi_a, lo_b, hi_b):
    inter = np.maximum(0, np.minimum(hi_a[:, None], hi_b[None]) -
                       np.maximum(lo_a[:, None], lo_b[None])).prod(2)
    va = (hi_a - lo_a).prod(1)
    vb = (hi_b - lo_b).prod(1)
    return inter / np.maximum(va[:, None] + vb[None] - inter, 1e-9)


def cluster(cand_idx, centers, radius):
    """Greedy score-ordered clustering by centre distance."""
    order = sorted(cand_idx, key=lambda i: -centers_score[i])
    clusters, assigned = [], {}
    for i in order:
        placed = False
        for ci, members in enumerate(clusters):
            if any(np.linalg.norm(centers[i] - centers[m]) <= radius for m in members):
                members.append(i)
                assigned[i] = ci
                placed = True
                break
        if not placed:
            clusters.append([i])
            assigned[i] = len(clusters) - 1
    return clusters


centers_score = None  # module-level for the greedy ordering


def main():
    global centers_score
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
        if not len(cand):
            with open(OUT / f'{scene}_boxes.pkl', 'wb') as f:
                pickle.dump([rows], f)
            continue
        c_lo, c_hi = aabb(cand)
        centers = (c_lo + c_hi) / 2
        extents = c_hi - c_lo
        centers_score = cand_scores.tolist()
        f_arr = np.asarray([r[1] for r in rows]) if rows else np.zeros((0, 8, 3))
        r_lo, r_hi = aabb(f_arr)
        iou_rc = iou_mat(r_lo, r_hi, c_lo, c_hi)  # rows x cands

        new_rows = []
        for k, row in enumerate(rows):
            R_lo, R_hi = r_lo[k], r_hi[k]
            diag = float(np.linalg.norm(R_hi - R_lo))
            nb = np.flatnonzero(iou_rc[k] >= 0.3)
            if len(nb) < 2:
                new_rows.append(row)
                continue
            radius = 0.30 * diag
            clusters = cluster(nb.tolist(), centers, radius)
            big = [cl for cl in clusters if len(cl) >= 2]
            # SPLIT: >= 2 multi-member clusters, mutually separated
            if len(big) >= 2:
                cents = [centers[cl].mean(0) for cl in big]
                ok = all(np.linalg.norm(cents[i] - cents[j]) >= 0.30 * diag
                         for i in range(len(big)) for j in range(i + 1, len(big)))
                if ok:
                    big.sort(key=len, reverse=True)
                    for ci, cl in enumerate(big):
                        rep = max(cl, key=lambda i: cand_scores[i])
                        sc = row[2] if ci == 0 else float(cand_scores[rep])
                        new_rows.append((row[0], cand[rep], sc))
                    counts['split_rows'] += len(big)
                    counts['split_events'] += 1
                    continue
            # RESCUE: single tight cluster inside an inflated row
            main_cl = max(clusters, key=len)
            if len(main_cl) >= 2:
                med_ext = np.median(extents[main_cl], axis=0)
                if (med_ext < 0.65 * (R_hi - R_lo)).any():
                    spread = float(np.linalg.norm(
                        centers[main_cl] - centers[main_cl].mean(0), axis=1).max())
                    if spread <= 0.35 * diag:
                        rep = max(main_cl, key=lambda i: cand_scores[i])
                        new_rows.append((row[0], cand[rep], row[2]))
                        counts['rescue_rows'] += 1
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
