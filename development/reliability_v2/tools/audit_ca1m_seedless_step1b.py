"""Seedless pre-NMS Step 1b: trigger feasibility of a top-K raw-score pre-screen.

Step 1 established a GT-assisted ceiling (+3.47 AP15, 23/87 missed GTs with
>=3-frame support @0.15) over the untruncated dense anchor pools. The unsolved
piece is the GT-free trigger: a deployable mechanism must find those support
anchors among 8,400 raw anchors per frame whose scores are ~1e-5 noise.

This audit measures the simplest deployable pre-screen: keep only each frame's
top-K anchors by raw score (K = 100/300/1000) and redo the identical oracle
selection, lifting and coverage. It also reports where the unfiltered support
anchors sit in the per-frame score ranking — the single number that decides
whether any score-based trigger can work at all.

Variants share one pass: per frame the four eligible sets are selected, their
anchor union is lifted once, and coverage/AP arms are attributed per variant.
GT remains target/anchor selector (ceiling measurement, not a module).
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_ca1m_nms_child_headroom import sha256, valid_boxes
from tools.true_fusion_audit_core import aabb_iou, class_agnostic_ap
from tools.audit_m1m2_remaining_children import (
    BASE, GT_ROOT, THRESHOLDS, read_prediction,
)
from tools.ca1m_prenms_query_core import (
    box_iou2d, box_iou3d, project_box, _nms_ids,
)
from tools.validate_ca1m_prenms_query import build_lifter, lift, load_frame

PILOT = ROOT / 'reports/ca1m_prenms_query_pilot_20260908'
DATA_ROOT = Path('/extra/ZhaoX/boxfusion_ca1m')
SCENES = ('42446540', '42897501', '42897521')
TOP_K_2D = 3
MIN_IOU_2D = 0.2
PRESCREEN_KS = (100, 300, 1000)
VARIANTS = ('none',) + tuple(f'top{k}' for k in PRESCREEN_KS)
APPEND_SCORE = 0.10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f'Refusing to overwrite {args.output}')
    args.output.mkdir(parents=True)
    protocol = json.loads((PILOT / 'protocol.json').read_text())
    assert tuple(protocol['scenes']) == SCENES and protocol['gap'] == 20
    inputs = {str(p.resolve()): sha256(p) for p in (
        Path(__file__), PILOT / 'protocol.json',
        ROOT / 'tools/audit_ca1m_nms_child_headroom.py',
        ROOT / 'tools/true_fusion_audit_core.py',
        ROOT / 'tools/audit_m1m2_remaining_children.py',
        ROOT / 'tools/ca1m_prenms_query_core.py',
        ROOT / 'tools/validate_ca1m_prenms_query.py')}
    adapter = build_lifter(args.output)
    baseline, gts, arms = {}, {}, {}
    coverage = {v: [] for v in VARIANTS}
    totals = {v: Counter() for v in VARIANTS}
    support_ranks = []
    for scene in SCENES:
        pred_path = BASE / f'{scene}_boxes.pkl'
        gt_path = GT_ROOT / scene / 'after_filter_boxes.npy'
        inputs[str(pred_path.resolve())] = sha256(pred_path)
        inputs[str(gt_path.resolve())] = sha256(gt_path)
        boxes, scores = read_prediction(pred_path)
        gt = valid_boxes(np.load(gt_path, allow_pickle=False), str(gt_path))
        poses = np.load(DATA_ROOT / scene / 'all_poses.npy', allow_pickle=False)
        k_rgb = np.loadtxt(DATA_ROOT / scene / 'K_rgb.txt').reshape(3, 3)
        k_depth = np.loadtxt(DATA_ROOT / scene / 'K_depth.txt').reshape(3, 3)
        biou = aabb_iou(boxes, gt).max(0)
        missed = np.flatnonzero(biou <= .15)
        support = {v: {int(g): set() for g in missed} for v in VARIANTS}
        best_box = {v: {int(g): (0.0, None) for g in missed} for v in VARIANTS}
        pools = sorted((PILOT / scene).glob('raw_*.npz'))
        for pool_path in pools:
            frame = int(pool_path.stem.split('_')[1])
            inputs[str(pool_path.resolve())] = sha256(pool_path)
            raw = np.load(pool_path, allow_pickle=False)
            order = np.argsort(-raw['scores'], kind='mergesort')
            rank = np.empty(len(order), dtype=np.int64)
            rank[order] = np.arange(len(order))
            rgb_pil, rgb, depth = load_frame(DATA_ROOT / scene, frame)
            height, width = rgb.shape[:2]
            pose = poses[frame]
            plans = {v: [] for v in VARIANTS}
            for g in missed:
                proj = project_box(gt[g], pose, k_rgb, width, height)
                if proj is None:
                    continue
                ious2d = box_iou2d(raw['boxes'], proj[None])[:, 0]
                hit = ious2d > MIN_IOU_2D
                if not hit.any():
                    continue
                for v in VARIANTS:
                    eligible = np.flatnonzero(hit if v == 'none'
                                              else hit & (rank < int(v[3:])))
                    if not len(eligible):
                        continue
                    top = _nms_ids(raw['boxes'], eligible, ious2d[eligible], TOP_K_2D)
                    for anchor in top:
                        plans[v].append((int(anchor), int(g)))
            selected = [(a, g, v) for v in VARIANTS for a, g in plans[v]]
            if not selected:
                continue
            unique, inverse = np.unique([s[0] for s in selected], return_inverse=True)
            corners, _, _ = lift(adapter, scene, frame, rgb, depth, k_rgb, k_depth,
                                 pose, raw['boxes'][unique])
            for pos, (a, g, v) in zip(inverse, selected):
                iou3 = float(box_iou3d(corners[pos][None], gt[g][None])[0, 0])
                if iou3 > .15:
                    support[v][g].add(frame)
                    if v == 'none':
                        support_ranks.append(int(rank[a]))
                if iou3 > best_box[v][g][0]:
                    best_box[v][g] = (iou3, corners[pos])
        for v in VARIANTS:
            for g in missed:
                coverage[v].append({'scene': scene, 'gt': int(g),
                                    'frames_0.15': len(support[v][int(g)])})
                totals[v][f'ge3'] += int(len(support[v][int(g)]) >= 3)
                totals[v][f'ge2'] += int(len(support[v][int(g)]) >= 2)
        baseline[scene], gts[scene] = (boxes, scores), gt
        for v in VARIANTS:
            extra = [best_box[v][int(g)][1] for g in missed
                     if len(support[v][int(g)]) >= 3 and best_box[v][int(g)][1] is not None]
            arms.setdefault(v, {})[scene] = (
                np.concatenate([boxes, np.asarray(extra)]) if extra else boxes,
                np.r_[scores, np.full(len(extra), APPEND_SCORE)] if extra else scores)
        print(f'{scene}: missed={len(missed)} '
              + ' '.join(f'{v}:ge3={totals[v]["ge3"]}' for v in VARIANTS), flush=True)
    metrics = {'baseline': {str(t): class_agnostic_ap(baseline, gts, t) for t in THRESHOLDS}}
    for v in VARIANTS:
        metrics[v] = {}
        for t in THRESHOLDS:
            value = class_agnostic_ap(arms[v], gts, t)
            value['delta_ap'] = value['ap'] - metrics['baseline'][str(t)]['ap']
            value['delta_tp'] = value['tp'] - metrics['baseline'][str(t)]['tp']
            value['delta_fp'] = value['fp'] - metrics['baseline'][str(t)]['fp']
            metrics[v][str(t)] = value
    for path, digest in inputs.items():
        assert sha256(path) == digest, f'Input changed: {path}'
    ranks = np.asarray(support_ranks)
    result = {
        'schema': 'boxfusion.seedless_prenms_step1b.v1', 'completed': True,
        'scenes': list(SCENES), 'missed_total': len(coverage['none']),
        'prescreen': {v: (None if v == 'none' else int(v[3:])) for v in VARIANTS},
        'coverage_ge3_ge2': {v: dict(totals[v]) for v in VARIANTS},
        'support_anchor_score_ranks': {
            'n': int(len(ranks)),
            'median': float(np.median(ranks)) if len(ranks) else None,
            'p25': float(np.percentile(ranks, 25)) if len(ranks) else None,
            'p75': float(np.percentile(ranks, 75)) if len(ranks) else None,
            'p90': float(np.percentile(ranks, 90)) if len(ranks) else None,
            'within_100': float(np.mean(ranks < 100)) if len(ranks) else None,
            'within_300': float(np.mean(ranks < 300)) if len(ranks) else None,
            'within_1000': float(np.mean(ranks < 1000)) if len(ranks) else None,
            'mean_random_expectation': 4200.0,
        },
        'limits': [
            'GT-assisted ceilings under a score-rank pre-screen only; the trigger '
            'itself (selection without GT) is still unsolved',
            'top-K by raw score is the simplest pre-screen; score-NMS or size-aware '
            'variants are not tested',
            'Append arms use a fixed score 0.10; 3 dev scenes; no FP simulation '
            'beyond the appended qualifying boxes.',
        ],
        'metrics': metrics,
    }
    for name, data in (('results.json', json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'),
                       ('input_sha256.json', json.dumps(inputs, ensure_ascii=False, indent=2) + '\n')):
        with (args.output / name).open('x') as handle:
            handle.write(data)
    print(json.dumps({
        'coverage': {v: dict(totals[v]) for v in VARIANTS},
        'support_rank_stats': result['support_anchor_score_ranks'],
        'deltas': {v: [round(metrics[v][str(t)]['delta_ap'], 4) for t in THRESHOLDS]
                   for v in VARIANTS}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
