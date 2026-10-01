#!/usr/bin/env python3
"""Dense-instance separation diagnostic on CA-1M (figure 10(b) failure class).

For every GT box that participates in a dense pair (AABB intersection over
the smaller box >= 0.25 with another GT), attribute the pipeline's behaviour
using three frozen artifact layers:

  L1 candidates  - per-keyframe WeDetect+Boxer capture (upper bound of what
                   the front end ever saw);
  L2 final map   - M1/M2-stack persistent rows.

Per dense GT:
  cand_only_alone  - some candidate matches this GT alone (IoU>=0.5) and no
                     candidate matches it together with its pair partner
                     (IoU>=0.3 with both)  -> separable evidence existed;
  cand_shared      - every matching candidate ALSO covers the partner
                     -> front end never separated the pair;
  no_candidate     - no candidate at IoU>=0.3 at all -> missing proposal.
Final-map outcome per dense PAIR:
  separated_ok     - two rows each matching one GT (IoU>=0.5);
  merged           - exactly one row matches both GTs (IoU>=0.3 each);
  lost             - fewer matches than objects.
Geometry degradation for merged pairs: final row IoU vs each GT compared with
the best single-candidate IoU for those GTs.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / 'reports/ca1m_lifting_factors_full107_20260908/capture'
GT_ROOT = Path('/extra/ZhaoX/boxfusion_ca1m')  # after_filter_boxes.npy = CA-1M GT
FINAL = ROOT / 'results/ca1m_thr15_m1_m2nl_full107'
PAIR_IOU2 = 0.25          # dense pair: intersection / smaller volume
MATCH = 0.5
SHARED = 0.3


def aabb(c):
    return c.min(1), c.max(1)


def iou_matrix_aabb(lo_a, hi_a, lo_b, hi_b):
    inter = np.maximum(0, np.minimum(hi_a[:, None], hi_b[None]) -
                       np.maximum(lo_a[:, None], lo_b[None])).prod(2)
    va = (hi_a - lo_a).prod(1)
    vb = (hi_b - lo_b).prod(1)
    return inter / np.maximum(va[:, None] + vb[None] - inter, 1e-9)


def iou_smaller_matrix(lo_a, hi_a, lo_b, hi_b):
    inter = np.maximum(0, np.minimum(hi_a[:, None], hi_b[None]) -
                       np.maximum(lo_a[:, None], lo_b[None])).prod(2)
    v = np.minimum((hi_a - lo_a).prod(1)[:, None], (hi_b - lo_b).prod(1)[None])
    return inter / np.maximum(v, 1e-9)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'reports/dense_instance_diag_20260914')
    args = p.parse_args()
    scenes = sorted(s.stem for s in CAPTURE.glob('*.npz')
                    if (GT_ROOT / s.stem / 'after_filter_boxes.npy').is_file()
                    and (FINAL / f'{s.stem}_boxes.pkl').is_file())
    print('scenes with all three layers:', len(scenes), flush=True)

    stats = {
        'dense_gts': 0, 'dense_pairs': 0,
        'gt_cand_only_alone': 0, 'gt_cand_shared': 0, 'gt_no_candidate': 0,
        'pair_separated_ok': 0, 'pair_merged': 0, 'pair_lost': 0,
        'merged_geometry_degraded': 0, 'merged_geometry_kept': 0,
    }
    examples = []
    for si, scene in enumerate(scenes):
        z = np.load(CAPTURE / f'{scene}.npz')
        cand = z['corners']
        if len(cand) == 0:
            continue
        gt = np.load(GT_ROOT / scene / 'after_filter_boxes.npy')
        with open(FINAL / f'{scene}_boxes.pkl', 'rb') as f:
            final = pickle.load(f)[0]
        fin = np.asarray([r[1] for r in final]) if final else np.zeros((0, 8, 3))
        c_lo, c_hi = aabb(cand)
        g_lo, g_hi = aabb(gt)
        f_lo, f_hi = aabb(fin) if len(fin) else (np.zeros((0, 3)), np.zeros((0, 3)))

        iou_gc = iou_matrix_aabb(g_lo, g_hi, c_lo, c_hi)
        iou_gf = iou_matrix_aabb(g_lo, g_hi, f_lo, f_hi)
        iou_pair = iou_smaller_matrix(g_lo, g_hi, g_lo, g_hi)
        np.fill_diagonal(iou_pair, 0)

        dense = iou_pair >= PAIR_IOU2
        partners = {i: int(np.argmax(dense[i])) for i in range(len(gt)) if dense[i].any()}
        # unique pairs
        pairs = set()
        for i in partners:
            j = partners[i]
            pairs.add((min(i, j), max(i, j)))
        stats['dense_pairs'] += len(pairs)
        stats['dense_gts'] += len(partners)

        for (i, j) in pairs:
            mi_mask = iou_gc[i] >= SHARED
            mj_mask = iou_gc[j] >= SHARED
            mi, mj = bool(mi_mask.any()), bool(mj_mask.any())
            shared = bool((mi_mask & mj_mask).any())
            alone_i = bool((mi_mask & ~mj_mask).any())
            alone_j = bool((mj_mask & ~mi_mask).any())
            for has, alone in ((mi, alone_i), (mj, alone_j)):
                if not has:
                    stats['gt_no_candidate'] += 1
                elif alone:
                    stats['gt_cand_only_alone'] += 1
                else:
                    stats['gt_cand_shared'] += 1

            fi = np.flatnonzero(iou_gf[i] >= MATCH)
            fj = np.flatnonzero(iou_gf[j] >= MATCH)
            both = np.intersect1d(fi, fj)
            if len(both) >= 1:
                stats['pair_merged'] += 1
                row = int(both[0])
                best_cand_i = float(iou_gc[i].max()) if mi else 0.0
                best_cand_j = float(iou_gc[j].max()) if mj else 0.0
                f_lo_r, f_hi_r = aabb(fin[row:row + 1])
                f_i = float(iou_matrix_aabb(g_lo[i:i + 1], g_hi[i:i + 1],
                                            f_lo_r, f_hi_r)[0, 0])
                f_j = float(iou_matrix_aabb(g_lo[j:j + 1], g_hi[j:j + 1],
                                            f_lo_r, f_hi_r)[0, 0])
                if f_i < best_cand_i - 0.05 and f_j < best_cand_j - 0.05:
                    stats['merged_geometry_degraded'] += 1
                    if len(examples) < 8:
                        examples.append({'scene': scene, 'pair': [i, j],
                                         'final_iou': [round(f_i, 3), round(f_j, 3)],
                                         'best_cand_iou': [round(best_cand_i, 3), round(best_cand_j, 3)]})
                else:
                    stats['merged_geometry_kept'] += 1
            elif len(fi) >= 1 and len(fj) >= 1 and not set(fi) & set(fj):
                stats['pair_separated_ok'] += 1
            else:
                stats['pair_lost'] += 1
        if (si + 1) % 20 == 0:
            print(f'{si+1}/{len(scenes)} scenes', json.dumps(stats), flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'dense_diag_stats.json').write_text(
        json.dumps({'stats': stats, 'merged_degraded_examples': examples,
                    'thresholds': {'pair_iou_smaller': PAIR_IOU2, 'match': MATCH,
                                   'shared': SHARED}}, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
