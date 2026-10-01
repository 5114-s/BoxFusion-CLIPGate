#!/usr/bin/env python3
"""Trace the 1,490 'candidate-ok, final-missing' dense targets (audit v2).

Corrected counting (each dense GT counted once by its OWN status; all
qualifying dense pairs counted), then for every dense GT whose best captured
candidate reaches IoU>=0.3 while the M1+M2 final map stays below 0.5, trace
where the evidence was lost across frozen layers:

  A capture    - per-keyframe WeDetect+Boxer candidates (score>=0.05, top150)
  B m1_final   - M1-recycling stack terminal rows
  C m2_final   - M1+M2 terminal rows (the production map)

Buckets for each lost target:
  lost_before_m1      - no row >=0.25 in B: the candidate never became a track
                        (sub-bucket by candidate evidence: single-keyframe /
                        low-score / multi-keyframe-but-unconfirmed);
  lost_at_m2          - row >=0.5 existed in B but <0.5 in C (M2/NMS stage);
  geometry_shifted    - a C-row exists at >=0.25 but <0.5: fused geometry
                        moved off the GT (compare best candidate IoU vs C-row
                        IoU: fused_worse / fused_better_but_short).
"""
from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / 'reports/ca1m_lifting_factors_full107_20260908/capture'
GT_ROOT = Path('/extra/ZhaoX/boxfusion_ca1m')
M1 = ROOT / 'results/ca1m_thr15_m1_full107'
M2 = ROOT / 'results/ca1m_thr15_m1_m2nl_full107'
PAIR_IOU2, CAND_MIN, FINAL_TP = 0.25, 0.3, 0.5


def aabb(c):
    return c.min(1), c.max(1)


def iou_mat(lo_a, hi_a, lo_b, hi_b):
    inter = np.maximum(0, np.minimum(hi_a[:, None], hi_b[None]) -
                       np.maximum(lo_a[:, None], lo_b[None])).prod(2)
    va = (hi_a - lo_a).prod(1)
    vb = (hi_b - lo_b).prod(1)
    return inter / np.maximum(va[:, None] + vb[None] - inter, 1e-9)


def iou_smaller_mat(lo_a, hi_a, lo_b, hi_b):
    inter = np.maximum(0, np.minimum(hi_a[:, None], hi_b[None]) -
                       np.maximum(lo_a[:, None], lo_b[None])).prod(2)
    v = np.minimum((hi_a - lo_a).prod(1)[:, None], (hi_b - lo_b).prod(1)[None])
    return inter / np.maximum(v, 1e-9)


def main():
    scenes = sorted(s.stem for s in CAPTURE.glob('*.npz')
                    if (GT_ROOT / s.stem / 'after_filter_boxes.npy').is_file()
                    and (M2 / f'{s.stem}_boxes.pkl').is_file())
    print('scenes:', len(scenes), flush=True)

    n_pairs = 0
    dense_gts = set()
    no_cand, cand_ok, final_ok = set(), set(), set()
    lost = []            # dicts per lost dense GT
    for si, scene in enumerate(scenes):
        z = np.load(CAPTURE / f'{scene}.npz')
        gt = np.load(GT_ROOT / scene / 'after_filter_boxes.npy')
        with open(M2 / f'{scene}_boxes.pkl', 'rb') as f:
            fin2 = pickle.load(f)[0]
        with open(M1 / f'{scene}_boxes.pkl', 'rb') as f:
            fin1 = pickle.load(f)[0]
        f2 = np.asarray([r[1] for r in fin2]) if fin2 else np.zeros((0, 8, 3))
        f1 = np.asarray([r[1] for r in fin1]) if fin1 else np.zeros((0, 8, 3))
        cand = z['corners']
        g_lo, g_hi = aabb(gt)
        cand = cand if len(cand) else np.zeros((0, 8, 3))
        c_lo, c_hi = aabb(cand)
        p_lo2, p_hi2 = aabb(f2)
        p_lo1, p_hi1 = aabb(f1)

        iou_pair = iou_smaller_mat(g_lo, g_hi, g_lo, g_hi)
        np.fill_diagonal(iou_pair, 0)
        dense_mask = iou_pair >= PAIR_IOU2
        n_pairs += int(np.count_nonzero(np.triu(dense_mask, 1)))
        dense_idx = set(np.flatnonzero(dense_mask.any(1)).tolist())
        dense_gts.update((scene, i) for i in dense_idx)

        if len(dense_idx):
            di = sorted(dense_idx)
            iou_gc = iou_mat(g_lo[di], g_hi[di], c_lo, c_hi) if len(cand) else np.zeros((len(di), 0))
            iou_g2 = iou_mat(g_lo[di], g_hi[di], p_lo2, p_hi2) if len(f2) else np.zeros((len(di), 0))
            iou_g1 = iou_mat(g_lo[di], g_hi[di], p_lo1, p_hi1) if len(f1) else np.zeros((len(di), 0))
            for k, gi in enumerate(di):
                best_c = float(iou_gc[k].max()) if iou_gc[k].size else 0.0
                best_f = float(iou_g2[k].max()) if iou_g2[k].size else 0.0
                best_1 = float(iou_g1[k].max()) if iou_g1[k].size else 0.0
                if best_c < CAND_MIN:
                    no_cand.add((scene, gi))
                    continue
                cand_ok.add((scene, gi))
                if best_f >= FINAL_TP:
                    final_ok.add((scene, gi))
                    continue
                # lost target: assemble trace
                m = iou_gc[k] >= CAND_MIN
                sel = np.flatnonzero(m)
                frames_sel = z['frame_ids'][sel]
                scores_sel = z['scores'][sel]
                n_kfs = len(set(frames_sel.tolist()))
                lost.append({
                    'scene': scene, 'gt': gi,
                    'best_cand_iou': round(best_c, 3),
                    'best_m1_iou': round(best_1, 3),
                    'best_m2_iou': round(best_f, 3),
                    'n_candidates': int(len(sel)),
                    'n_keyframes': n_kfs,
                    'cand_score_max': round(float(scores_sel.max()), 3) if len(sel) else 0.0,
                    'cand_score_median': round(float(np.median(scores_sel)), 3) if len(sel) else 0.0,
                })
        if (si + 1) % 30 == 0:
            print(f'{si+1}/{len(scenes)} pairs={n_pairs} dense={len(dense_gts)} '
                  f'no_cand={len(no_cand)} cand_ok={len(cand_ok)} lost={len(lost)}', flush=True)

    buckets = Counter()
    sub = Counter()
    for t in lost:
        if t['best_m1_iou'] < 0.25:
            buckets['lost_before_m1'] += 1
            if t['n_keyframes'] == 1:
                sub['before_m1/single_keyframe'] += 1
            elif t['cand_score_max'] < 0.3:
                sub['before_m1/low_score(max<0.3)'] += 1
            else:
                sub['before_m1/multi_kf_unconfirmed'] += 1
        elif t['best_m1_iou'] >= FINAL_TP and t['best_m2_iou'] < FINAL_TP:
            buckets['lost_at_m2'] += 1
        elif t['best_m2_iou'] >= 0.25:
            buckets['geometry_shifted'] += 1
            if t['best_m2_iou'] < t['best_cand_iou'] - 0.05:
                sub['geometry/fused_worse'] += 1
            else:
                sub['geometry/better_but_below_0.5'] += 1
        else:
            buckets['m1_partial_lost_at_m2'] += 1
    summary = {
        'dense_pairs': n_pairs,
        'dense_gts_unique': len(dense_gts),
        'gt_no_candidate_ge_0.3': len(no_cand),
        'gt_candidate_ok': len(cand_ok),
        'gt_candidate_ok_final_ok': len(final_ok),
        'lost_targets_traced': len(lost),
        'buckets': dict(buckets),
        'sub_buckets': dict(sub),
        'note': 'AABB IoU; capture = WeDetect+Boxer score>=0.05 top150 (pre-pipeline)',
    }
    out = ROOT / 'reports/dense_instance_diag_20260914'
    (out / 'trace_lost_targets.json').write_text(
        json.dumps({'summary': summary, 'lost': lost}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
