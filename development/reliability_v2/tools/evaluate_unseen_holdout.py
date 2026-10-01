#!/usr/bin/env python3
"""Evaluate the pre-registered unseen holdout (reports/unseen_holdout_extra50).

Arms per scene, all from the frozen run's own outputs:
  base    native pickle
  p       m1_only pickle minus M1-A birth rows (pre-M2 funnel)
  p_m2    persistent pickle minus M1-A birth rows (post-M2/M5 persistent view)
  final   persistent pickle as-is (funnel + strict-online M1-A births)

M1-A birth rows are identified by score < 0.05 (their unique-score interval
[0.040001, 0.049999]); the count must equal the causal trace's births for
every scene, and the non-birth rows of m1_only/persistent must be prefix
consistent with base geometry ordering.  Evaluation uses the semantic-table
coordinate convention (npy GT + axisAlignment) and the anchor AP convention.
No development anchors exist for these scenes by construction; instead the
internal consistency assertions above bind the arms.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from true_fusion_audit_core import aabb_iou, class_agnostic_ap
from run_scannet_semantic_table import ground_truth, alignment

OUT = ROOT / 'reports/unseen_holdout_extra50_20260916'
THRESHOLDS = (0.15, 0.25, 0.5)
B = 2_000
SEED = 20260916
BIRTH_MARK = 0.05


def read_rows(path):
    rows = pickle.loads(path.read_bytes())[0]
    if any(len(r) != 3 for r in rows):
        raise ValueError(f'unexpected row arity: {path}')
    boxes = np.asarray([r[1] for r in rows], np.float64).reshape(-1, 8, 3)
    scores = np.asarray([r[2] for r in rows], np.float64)
    return boxes, scores


def scene_pairs(corners, scores, gt, threshold):
    scores = np.asarray(scores, dtype=np.float64)
    matrix = aabb_iou(corners, gt) if len(corners) and len(gt) else np.zeros(
        (len(corners), len(gt)))
    matched = np.zeros(len(gt), dtype=bool)
    is_tp = np.zeros(len(scores), dtype=np.float64)
    for row in np.argsort(-scores, kind='mergesort'):
        values = matrix[row]
        if len(values) and values.max() > threshold:
            target = int(values.argmax())
            if not matched[target]:
                matched[target] = True
                is_tp[row] = 1.0
    return np.column_stack((scores, is_tp))


def ap_from_pairs(pairs, gt_count):
    order = np.argsort(-pairs[:, 0], kind='mergesort')
    tp = pairs[order, 1]
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1.0 - tp)
    recall = cum_tp / (gt_count + 1e-6)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)
    mr = np.r_[0.0, recall, 1.0]
    mp = np.r_[0.0, precision, 0.0]
    mp = np.maximum.accumulate(mp[::-1])[::-1]
    changes = np.flatnonzero(mr[1:] != mr[:-1])
    return 100.0 * float(np.sum((mr[changes + 1] - mr[changes])
                                * mp[changes + 1]))


def main():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    scenes = protocol['scenes']
    arms = {name: {} for name in ('base', 'p', 'a', 'p_m2', 'final')}
    gts = {}
    for scene in scenes:
        persistent = OUT / 'persistent' / f'{scene}_boxes.pkl'
        m1_only = OUT / 'm1_only' / f'{scene}_boxes.pkl'
        native = OUT / 'native' / f'{scene}_boxes.pkl'
        trace = json.loads((persistent.with_suffix(
            '').with_name(persistent.name + '.m1a_online.json')).read_text())
        fb, fs = read_rows(persistent)
        mb, ms = read_rows(m1_only)
        nb, ns = read_rows(native)
        is_birth = fs < BIRTH_MARK
        assert int(is_birth.sum()) == trace['tracker']['births'], scene
        is_birth_m = ms < BIRTH_MARK
        assert int(is_birth_m.sum()) == trace['tracker']['births'], scene
        assert np.array_equal(fb[is_birth], mb[is_birth_m]), scene
        align = alignment(scene)
        world = lambda b: b @ align[:3, :3].T + align[:3, 3]
        gts[scene] = ground_truth(scene)[0]
        arms['base'][scene] = (world(nb), ns)
        arms['p'][scene] = (world(mb[~is_birth_m]), ms[~is_birth_m])
        arms['a'][scene] = (world(np.concatenate([nb, fb[is_birth]])),
                            np.concatenate([ns, fs[is_birth]]))
        arms['p_m2'][scene] = (world(fb[~is_birth]), fs[~is_birth])
        arms['final'][scene] = (world(fb), fs)
    metrics = {arm: {str(t): class_agnostic_ap(preds, gts, t)
                     for t in THRESHOLDS}
               for arm, preds in arms.items()}
    deltas = {
        'M1P_over_base': ('p', 'base'),
        'M1A_over_base': ('a', 'base'),
        'M2_over_M1P': ('p_m2', 'p'),
        'M1A_over_M1PM2': ('final', 'p_m2'),
        'full_over_base': ('final', 'base'),
    }
    pairs = {arm: {t: {s: scene_pairs(*arms[arm][s], gts[s], t)
                       for s in scenes} for t in THRESHOLDS}
             for arm in arms}
    result = {'protocol': protocol['constants'],
              'scene_count': len(scenes),
              'metrics': metrics,
              'level_deltas': {name: {str(t):
                                      metrics[up][str(t)]['ap']
                                      - metrics[low][str(t)]['ap']
                                      for t in THRESHOLDS}
                               for name, (up, low) in deltas.items()},
              'per_scene_ap15': {arm: [round(ap_from_pairs(
                  np.concatenate([pairs[arm][0.15][s] for s in [sc]]),
                  len(gts[sc])), 4) for sc in scenes] for arm in arms},
              'bootstrap': {}}
    rng = np.random.default_rng(SEED)
    gt_counts = {s: len(gts[s]) for s in scenes}
    for name, (up, low) in deltas.items():
        result['bootstrap'][name] = {}
        for t in THRESHOLDS:
            up_pairs = {s: pairs[up][t][s] for s in scenes}
            low_pairs = {s: pairs[low][t][s] for s in scenes}
            samples = np.empty(B)
            for b in range(B):
                idx = rng.integers(0, len(scenes), len(scenes))
                gu = sum(gt_counts[scenes[i]] for i in idx)
                pu = np.concatenate([up_pairs[scenes[i]] for i in idx])
                pl = np.concatenate([low_pairs[scenes[i]] for i in idx])
                samples[b] = ap_from_pairs(pu, gu) - ap_from_pairs(pl, gu)
            result['bootstrap'][name][str(t)] = {
                'point': result['level_deltas'][name][str(t)],
                'mean': float(samples.mean()),
                'ci_lower': float(np.percentile(samples, 2.5)),
                'ci_upper': float(np.percentile(samples, 97.5)),
                'p_positive': float((samples > 0).mean())}
    (OUT / 'results.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('scenes:', len(scenes))
    for arm in arms:
        print(f"{arm:6s}", [round(metrics[arm][str(t)]['ap'], 4)
                            for t in THRESHOLDS],
              'boxes', metrics[arm]['0.15']['predictions'])
    for name, d in result['level_deltas'].items():
        print(f'{name:16s}', [round(v, 4) for v in d.values()])
    for name, by_t in result['bootstrap'].items():
        for t, v in by_t.items():
            print(f"  {name:16s} IoU{t}: {v['point']:+.4f} "
                  f"[{v['ci_lower']:+.4f}, {v['ci_upper']:+.4f}] "
                  f"p>0={v['p_positive']:.3f}")


if __name__ == '__main__':
    main()
