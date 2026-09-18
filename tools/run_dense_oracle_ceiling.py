#!/usr/bin/env python3
"""Oracle ceilings for the two dense-instance mechanisms (offline, cached).

Arm B (geometry replacement): for every fused-worse target (final row IoU in
[0.25, 0.5) and best candidate IoU > final + 0.05), replace that final row's
corners with the best candidate's corners (row score unchanged).  A row is
replaced at most once.

Arm A (single-frame birth): for every lost target with single-keyframe
candidate evidence and no M1 row >= 0.25, append a new row with the best
candidate's corners and the candidate's own real score.

Arms: base / A / B / AB, evaluated with the anchor CA-1M evaluator
(pooled greedy real-score AP15/25/50) plus dense-GT recall@0.25/0.50.
These are CEILINGS under oracle selection, not a method.
"""
from __future__ import annotations

import json
import pickle
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from eval_ca1m import box_iou, evaluate, load_gt

CAPTURE = ROOT / 'reports/ca1m_lifting_factors_full107_20260908/capture'
GT_ROOT = Path('/extra/ZhaoX/boxfusion_ca1m')
M2 = ROOT / 'results/ca1m_thr15_m1_m2nl_full107'
OUT = ROOT / 'reports/dense_oracle_ceiling_20260914'
PAIR_IOU2, CAND_MIN, TP = 0.25, 0.3, 0.5


def aabb(c):
    return c.min(1), c.max(1)


def iou_mat(lo_a, hi_a, lo_b, hi_b):
    inter = np.maximum(0, np.minimum(hi_a[:, None], hi_b[None]) -
                       np.maximum(lo_a[:, None], lo_b[None])).prod(2)
    va = (hi_a - lo_a).prod(1)
    vb = (hi_b - lo_b).prod(1)
    return inter / np.maximum(va[:, None] + vb[None] - inter, 1e-9)


def iou_sm(lo_a, hi_a, lo_b, hi_b):
    inter = np.maximum(0, np.minimum(hi_a[:, None], hi_b[None]) -
                       np.maximum(lo_a[:, None], lo_b[None])).prod(2)
    v = np.minimum((hi_a - lo_a).prod(1)[:, None], (hi_b - lo_b).prod(1)[None])
    return inter / np.maximum(v, 1e-9)


def main():
    trace = json.loads((ROOT / 'reports/dense_instance_diag_20260914/trace_lost_targets.json').read_text())
    lost = trace['lost']
    scenes = sorted({t['scene'] for t in lost} |
                    {s.stem for s in CAPTURE.glob('*.npz')
                     if (M2 / f'{s.stem}_boxes.pkl').is_file()
                     and (GT_ROOT / s.stem / 'after_filter_boxes.npy').is_file()})
    by_scene = {}
    for t in lost:
        by_scene.setdefault(t['scene'], []).append(t)

    for arm in ('base', 'A', 'B', 'AB'):
        (OUT / arm).mkdir(parents=True, exist_ok=True)

    counts = Counter()
    dense_matched = {arm: Counter() for arm in ('base', 'A', 'B', 'AB')}
    dense_total = 0
    for scene in scenes:
        with open(M2 / f'{scene}_boxes.pkl', 'rb') as f:
            rows = list(pickle.load(f)[0])
        z = np.load(CAPTURE / f'{scene}.npz')
        cand = z['corners']
        cand_scores = z['scores']
        cand_frames = z['frame_ids']
        gt = np.load(GT_ROOT / scene / 'after_filter_boxes.npy')
        g_lo, g_hi = aabb(gt)
        f_arr = np.asarray([r[1] for r in rows]) if rows else np.zeros((0, 8, 3))
        c_lo, c_hi = aabb(cand) if len(cand) else (np.zeros((0, 3)), np.zeros((0, 3)))
        p_lo, p_hi = aabb(f_arr) if len(f_arr) else (np.zeros((0, 3)), np.zeros((0, 3)))
        iou_pair = iou_sm(g_lo, g_hi, g_lo, g_hi)
        np.fill_diagonal(iou_pair, 0)
        dense_idx = set(np.flatnonzero((iou_pair >= PAIR_IOU2).any(1)).tolist())

        rows_A = list(rows)
        rows_B = list(rows)
        replaced = set()
        for t in by_scene.get(scene, []):
            gi = t['gt']
            best_c, best_f = t['best_cand_iou'], t['best_m2_iou']
            if not (best_c >= CAND_MIN and best_f < TP):
                continue
            # candidate selection (same rule as the trace)
            iou_gc = iou_mat(g_lo[gi:gi + 1], g_hi[gi:gi + 1], c_lo, c_hi)[0] if len(cand) else np.zeros(0)
            sel = np.flatnonzero(iou_gc >= CAND_MIN)
            if not len(sel):
                continue
            best_sel = int(sel[np.argmax(iou_gc[sel])])
            # Arm B: fused-worse geometry replacement
            if (0.25 <= best_f < TP and best_f < best_c - 0.05):
                row = int(np.argmax(iou_mat(g_lo[gi:gi + 1], g_hi[gi:gi + 1], p_lo, p_hi)[0])) if len(f_arr) else -1
                if row >= 0 and row not in replaced and row < len(rows_B):
                    rows_B[row] = (rows_B[row][0], cand[best_sel], rows_B[row][2])
                    replaced.add(row)
                    counts['B_replaced'] += 1
                else:
                    counts['B_skipped_conflict'] += 1
            # Arm A: single-keyframe evidence never confirmed
            frames_sel = cand_frames[sel]
            if t['best_m1_iou'] < 0.25 and len(set(frames_sel.tolist())) == 1:
                rows_A.append((0, cand[best_sel], float(cand_scores[best_sel])))
                counts['A_injected'] += 1
        rows_AB = list(rows_B) + [r for r in rows_A[len(rows):]]

        for arm, rr in (('base', rows), ('A', rows_A), ('B', rows_B), ('AB', rows_AB)):
            with open(OUT / arm / f'{scene}_boxes.pkl', 'wb') as f:
                pickle.dump([rr], f)
            if not dense_idx:
                continue
            arr = np.asarray([r[1] for r in rr]) if rr else np.zeros((0, 8, 3))
            a_lo, a_hi = aabb(arr) if len(arr) else (np.zeros((0, 3)), np.zeros((0, 3)))
            di = sorted(dense_idx)
            gm = iou_mat(g_lo[di], g_hi[di], a_lo, a_hi) if len(arr) else np.zeros((len(di), 0))
            dense_matched[arm]['total'] += len(di)
            dense_matched[arm]['hit25'] += int((gm >= 0.25).any(1).sum()) if gm.size else 0
            dense_matched[arm]['hit50'] += int((gm >= 0.5).any(1).sum()) if gm.size else 0
            dense_total += len(di)

    results = {}
    for arm in ('base', 'A', 'B', 'AB'):
        aps = evaluate(str(GT_ROOT), str(OUT / arm), scenes)
        dm = dense_matched[arm]
        results[arm] = {
            'AP15': round(aps[0] * 100, 2), 'AP25': round(aps[1] * 100, 2),
            'AP50': round(aps[2] * 100, 2),
            'dense_hit25': f"{dm['hit25']}/{dm['total']}",
            'dense_hit50': f"{dm['hit50']}/{dm['total']}",
        }
        print(arm, results[arm], flush=True)
    (OUT / 'oracle_ceiling_results.json').write_text(
        json.dumps({'counts': dict(counts), 'dense_gts': dense_total,
                    'results': results}, indent=1))
    print('counts:', dict(counts))


if __name__ == '__main__':
    main()
