#!/usr/bin/env python3
"""Same-budget controls for the STRICT-ONLINE final arm (ScanNet, M300).

Re-binds the earlier terminal-arm controls to the current final version:
baseline = M1-P+M2 persistent map, final = strict-online M1-A output
(reports/m1a_strict_online_scannet_20260916), whose per-scene birth counts
all other arms must match exactly.  Arms differ only in where the added
boxes come from and at what score:

  final_online   the strict-online M1-A route (asserted to 44.0564/39.75/20.2437)
  low_threshold  ordinary-stream tail appended with REAL detection scores
                 (score-preserving threshold release, top-N per scene);
  topn_flat      tail top-N at the flat tail price;
  tail_dedup     same, after dropping tail boxes that duplicate a map row;
  tail_voxel     tail + the module's cross-frame voxel dedup, score-ranked;
  cluster_random same frozen anchor-cluster pool, N-scene clusters by seeded RNG;
  cluster_lowest N-scene clusters from the bottom of the exemplar ranking.

Also emits pooled precision-recall curves for baseline vs final at the three
thresholds (downsampled), for the paper's PR figure.  GT for evaluation only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_m1m2_remaining_children import read_prediction
from tools.audit_seedless_ablation_matrix import (
    BASELINES, RUNS, THRESHOLDS, build_scannet_eval)
from tools.audit_tier2_same_budget import (
    DEDUP_IOU, SEED, TAIL, cluster_states)
from tools.true_fusion_audit_core import aabb_iou, class_agnostic_ap

ONLINE_RUN = ROOT / 'reports/m1a_strict_online_scannet_20260916'
OUT = ROOT / 'reports/m1a_same_budget_20260916'
TAIL_PRICE = 0.05
EXPECTED = (44.0564, 39.75, 20.2437)
BASELINE_EXPECTED = (41.29890136134413, 37.302988324064586, 18.512383520854534)


def pr_curve(predictions, gts, threshold, points=200):
    """Pooled (recall, precision) envelope, downsampled for plotting."""
    entries = []
    for scene, (corners, scores) in predictions.items():
        gt = gts.get(scene, [])
        overlaps = aabb_iou(corners, gt) if len(corners) and len(gt) \
            else np.zeros((len(corners), len(gt)))
        for row, score in enumerate(np.asarray(scores, float)):
            entries.append((float(score), scene, row, overlaps[row]))
    entries.sort(key=lambda e: -e[0])
    gt_total = sum(len(g) for g in gts.values())
    taken = {s: np.zeros(len(gts.get(s, [])), bool) for s in predictions}
    curve = []
    tp = fp = 0
    for score, scene, row, values in entries:
        if len(values) and values.max() > threshold:
            target = int(values.argmax())
            if not taken[scene][target]:
                taken[scene][target] = True
                tp += 1
            else:
                fp += 1
        else:
            fp += 1
        curve.append((tp / (gt_total + 1e-6), tp / max(tp + fp, 1e-12)))
    step = max(1, len(curve) // points)
    return curve[::step] + [curve[-1]]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run = RUNS['scannet']
    protocol = json.loads((run / 'protocol.json').read_text())
    scenes = protocol['scenes']
    gts, aligns, transform = build_scannet_eval(protocol)
    rng = np.random.default_rng(SEED)

    arms = {name: {} for name in ('baseline', 'final_online', 'low_threshold',
                                  'topn_flat', 'tail_dedup', 'tail_voxel',
                                  'cluster_random', 'cluster_lowest')}
    plan = {}
    for scene in scenes:
        base_boxes, base_scores = read_prediction(
            BASELINES['scannet'] / f'{scene}_boxes.pkl')
        ob, os_ = read_prediction(ONLINE_RUN / 'predictions'
                                  / f'{scene}_boxes.pkl')
        births = ob[len(base_boxes):]
        n = len(births)
        plan[scene] = int(n)

        states = cluster_states(scene, run, 'score_m300')
        exemplars = np.asarray([c['box'] for c in states]) if states else \
            np.zeros((0, 8, 3))
        tail = np.load(TAIL / f'{scene}.npz', allow_pickle=True)
        tail_boxes, tail_scores = (np.asarray(tail['corners_raw'], np.float64),
                                   np.asarray(tail['scores'], np.float64))
        order = np.argsort(-tail_scores, kind='mergesort')[:n]
        matrix = aabb_iou(tail_boxes, base_boxes) if len(tail_boxes) \
            and len(base_boxes) else np.zeros((len(tail_boxes),
                                               len(base_boxes)))
        keep = np.flatnonzero(matrix.max(1) <= DEDUP_IOU) if len(matrix) \
            else np.arange(len(tail_boxes))
        keep_sorted = keep[np.argsort(-tail_scores[keep], kind='mergesort')]
        voxels = {}
        from tools.ca1m_seedless_trigger_core import add_clusters
        add_clusters(voxels, -1, np.arange(len(tail_boxes))[keep],
                     tail_boxes[keep], tail_scores[keep], .3)
        voxel_states = sorted(voxels.values(), key=lambda c: c['rank'])
        random_idx = rng.choice(len(states), size=min(n, len(states)),
                                replace=False) if len(states) else []

        picks = {
            'final_online': (births, os_[len(base_scores):]),
            'low_threshold': (tail_boxes[order], tail_scores[order]),
            'topn_flat': (tail_boxes[order],
                          np.full(len(order), TAIL_PRICE)),
            'tail_dedup': (tail_boxes[keep_sorted[:n]],
                           np.full(min(n, len(keep_sorted)), TAIL_PRICE)),
            'tail_voxel': (np.asarray([c['box'] for c in
                                       voxel_states[:n]]),
                           np.full(min(n, len(voxel_states)), TAIL_PRICE)),
            'cluster_random': (exemplars[sorted(random_idx)],
                               np.full(len(random_idx), TAIL_PRICE)),
            'cluster_lowest': (exemplars[::-1][:n],
                               np.full(min(n, len(exemplars)), TAIL_PRICE)),
        }
        arms['baseline'][scene] = (transform(base_boxes, aligns[scene]),
                                   base_scores)
        for name, value in picks.items():
            extra, extra_scores = value
            if name == 'final_online':
                merged = np.concatenate([base_boxes, extra])
                scores = np.concatenate([base_scores, extra_scores])
            else:
                extra = np.asarray(extra).reshape(-1, 8, 3)
                extra_scores = np.asarray(extra_scores, float)
                merged = np.concatenate([base_boxes, extra]) if len(extra) \
                    else base_boxes
                scores = np.concatenate([base_scores, extra_scores]) \
                    if len(extra) else base_scores
            arms[name][scene] = (transform(merged, aligns[scene]), scores)

    metrics = {}
    for name, predictions in arms.items():
        metrics[name] = {str(t): class_agnostic_ap(predictions, gts, t)
                         for t in THRESHOLDS}
    for t, expected in zip(THRESHOLDS, EXPECTED):
        assert abs(metrics['final_online'][str(t)]['ap'] - expected) <= 5e-4
    for t, expected in zip(THRESHOLDS, BASELINE_EXPECTED):
        assert abs(metrics['baseline'][str(t)]['ap'] - expected) <= 5e-4

    result = {
        'protocol': {
            'baseline': 'M1-P+M2 persistent map (results/scannet_m2nl_m5_dual_full100/persistent)',
            'final': 'strict-online M1-A route',
            'matched_counts': 'per-scene online birth counts',
            'tail_cache': str(TAIL), 'tail_price': TAIL_PRICE,
            'seed': SEED, 'dedup_iou': DEDUP_IOU,
            'online_births': int(sum(plan.values())),
        },
        'counts': plan,
        'metrics': metrics,
        'deltas': {name: {str(t): {k: metrics[name][str(t)][k] -
                                   metrics['baseline'][str(t)][k]
                                   for k in ('ap', 'tp', 'fp')}
                          for t in THRESHOLDS}
                   for name in metrics if name != 'baseline'},
        'pr_curves': {arm: {str(t): pr_curve(arms[arm], gts, t)
                            for t in THRESHOLDS}
                      for arm in ('baseline', 'final_online')},
        'limits': ['Tail cache predates the current baseline run; frozen '
                   'per-frame candidates are deterministic across same-config '
                   'runs and coordinate consistency is checked by the '
                   'map-overlap statistic.',
                   'GT is used for evaluation only.'],
    }
    (OUT / 'same_budget.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('online births:', result['protocol']['online_births'])
    for name in metrics:
        d = result['deltas'].get(name)
        if name == 'baseline':
            print(f"baseline       "
                  f"{[round(metrics[name][str(t)]['ap'], 4) for t in THRESHOLDS]}")
            continue
        print(f"{name:15s} dAP="
              f"{d['0.15']['ap']:+.4f}/{d['0.25']['ap']:+.4f}/"
              f"{d['0.5']['ap']:+.4f}  dTP15={d['0.15']['tp']:+.0f} "
              f"dFP15={d['0.15']['fp']:+.0f}")


if __name__ == '__main__':
    main()
