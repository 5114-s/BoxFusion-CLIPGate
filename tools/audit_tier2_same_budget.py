#!/usr/bin/env python3
"""Same-budget pure-score controls for the final tier-2 arm (ScanNet).

Question: does the tier-2 gain come from its design (anchor-level candidate
expansion + Boxer lifting + voxel dedup + score-ranked births) or from simply
adding more candidate boxes?  Every arm requests the same per-scene number at
the same flat 0.05 price over the same unified-stack baseline.  Arms whose
deduplicated source pool is too small append fewer boxes; the output records
the actual counts, so those arms are compact-budget controls rather than
strict same-count controls:

  tier2_score    final tier-2 (frozen seedless selection; reproduces 43.3503)
  tier2_mapdedup final design additionally skipping exemplars that duplicate
                 a baseline map row (IoU>0.5), refilled by score rank;
  tail_score     ordinary-stream tail: WeDetect proposals below the 0.5 map
                 threshold (wedetect_lifted_cache, Boxer-lifted, real scores),
                 top-N per scene by real detection score;
  tail_dedup     same, after dropping tail candidates with IoU>0.5 against a
                 baseline map row (anti-inflation variant);
  tail_voxel     strongest control: ordinary tail plus the module's own
                 cross-frame 0.3 m voxel dedup, exemplar-score-ranked;
  cluster_random same cluster pool, N3-scene clusters drawn by seeded RNG;
  cluster_lowest N3-scene clusters from the bottom of the exemplar ranking.

The tail cache predates the current baseline run (2026-09-03 Cbest prefix);
the frozen detector makes per-frame candidates deterministic, but map-level
differences are handled by reporting tail_dedup alongside the raw variant.
GT is used for evaluation only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_m1m2_remaining_children import read_prediction
from tools.ca1m_seedless_trigger_core import add_clusters
from tools.audit_final_ledger import TIER2_PRICE, OUT
from tools.audit_seedless_ablation_matrix import (
    BASELINES, RUNS, THRESHOLDS, build_scannet_eval)
from tools.true_fusion_audit_core import aabb_iou, class_agnostic_ap

TAIL = ROOT / 'results/wedetect_lifted_cache'
BASELINE = BASELINES['scannet']
SEED = 20260915
DEDUP_IOU = .5


def cluster_states(scene, run, pool):
    """Voxel clusters for one scene/pool (same construction as the final arm)."""
    directory = run / 'scenes' / scene
    selection = json.loads((directory / 'selection.json').read_text())
    clusters = {}
    for row in selection['frames']:
        frame = row['frame']
        with np.load(directory / f'lifted_{frame:06d}.npz',
                     allow_pickle=False) as values:
            union, corners = values['anchor_ids'], values['corners']
        with np.load(run / 'raw' / scene / f'raw_{frame:06d}.npz',
                     allow_pickle=False) as values:
            raw_scores = values['scores']
        ids = np.asarray(row['selected_anchor_ids'][pool], dtype=np.int64)
        positions = np.searchsorted(union, ids)
        if not np.array_equal(union[positions], ids):
            raise ValueError(f'Anchor mismatch: {scene}/{frame}/{pool}')
        add_clusters(clusters, frame, ids, corners[positions],
                     raw_scores[ids], .3)
    return sorted(clusters.values(), key=lambda c: c['rank'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT / 'same_budget.json')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run = RUNS['scannet']
    protocol = json.loads((run / 'protocol.json').read_text())
    scenes = protocol['scenes']
    gts, aligns, transform = build_scannet_eval(protocol)

    arms = {name: {} for name in ('baseline', 'tier2_score', 'tier2_mapdedup',
                                  'tail_score', 'tail_dedup', 'tail_voxel',
                                  'cluster_random', 'cluster_lowest')}
    plan = {'requested': {}, 'tail_pool': {}, 'tail_in_map': {},
            'tail_used': {}, 'tail_used_dedup': {}, 'actual_added': {
                name: {} for name in ('tier2_score', 'tier2_mapdedup',
                                      'tail_score', 'tail_dedup', 'tail_voxel',
                                      'cluster_random', 'cluster_lowest')}}
    rng = np.random.default_rng(SEED)
    for scene in scenes:
        base_boxes, base_scores = read_prediction(
            BASELINE / f'{scene}_boxes.pkl')
        states = cluster_states(scene, run, 'score_m300')
        n3 = sum(len(c['frames']) >= 3 for c in states)
        born = np.asarray([c['box'] for c in states[:n3]])
        plan['requested'][scene] = int(n3)

        tail = np.load(TAIL / f'{scene}.npz', allow_pickle=True)
        tail_boxes, tail_scores = (np.asarray(tail['corners_raw'], np.float64),
                                   np.asarray(tail['scores'], np.float64))
        plan['tail_pool'][scene] = int(len(tail_boxes))
        order = np.argsort(-tail_scores, kind='mergesort')
        top = order[:n3]
        plan['tail_used'][scene] = int(len(top))
        matrix = aabb_iou(tail_boxes, base_boxes) if len(tail_boxes) and \
            len(base_boxes) else np.zeros((len(tail_boxes), len(base_boxes)))
        plan['tail_in_map'][scene] = int((matrix.max(1) > DEDUP_IOU).sum()
                                         ) if len(matrix) else 0
        keep = np.flatnonzero(matrix.max(1) <= DEDUP_IOU) if len(matrix) else \
            np.arange(len(tail_boxes))
        order_keep = keep[np.argsort(-tail_scores[keep], kind='mergesort')]
        top_dedup = order_keep[:n3]
        plan['tail_used_dedup'][scene] = int(len(top_dedup))

        # strongest control: ordinary tail + the module's own cross-frame
        # 0.3 m voxel dedup, exemplar = highest raw score
        voxels = {}
        add_clusters(voxels, -1,
                     np.arange(len(tail_boxes))[keep],
                     tail_boxes[keep], tail_scores[keep], .3)
        voxel_states = sorted(voxels.values(), key=lambda c: c['rank'])
        # symmetry arm: final design additionally skipping exemplars that
        # duplicate a baseline map row (IoU>0.5), refilled by score rank
        exemplars = np.asarray([c['box'] for c in states])
        ex_map = aabb_iou(exemplars, base_boxes) if len(exemplars) and \
            len(base_boxes) else np.zeros((len(exemplars), len(base_boxes)))
        fresh = np.flatnonzero(ex_map.max(1) <= DEDUP_IOU) if len(ex_map) \
            else np.arange(len(exemplars))
        fresh = sorted(fresh, key=lambda i: states[i]['rank'])[:n3]
        picks = {
            'tier2_score': born,
            'tier2_mapdedup': np.asarray([states[i]['box'] for i in fresh]),
            'tail_score': tail_boxes[top],
            'tail_dedup': tail_boxes[top_dedup],
            'tail_voxel': np.asarray([c['box'] for c in
                                      voxel_states[:n3]]),
        }
        pick_low = states[::-1][:n3]
        picks['cluster_lowest'] = np.asarray([c['box'] for c in pick_low])
        random_idx = rng.choice(len(states), size=n3, replace=False) \
            if len(states) > n3 else np.arange(len(states))
        picks['cluster_random'] = np.asarray(
            [states[i]['box'] for i in random_idx])
        for name, extra in picks.items():
            plan['actual_added'][name][scene] = int(len(extra))
            merged = np.concatenate([base_boxes, extra]) if len(extra) \
                else base_boxes
            arms[name][scene] = (transform(merged, aligns[scene]),
                                 np.r_[base_scores, [TIER2_PRICE] * len(extra)]
                                 if len(extra) else base_scores)
        arms['baseline'][scene] = (transform(base_boxes, aligns[scene]),
                                   base_scores)
    metrics = {}
    for name, predictions in arms.items():
        metrics[name] = {str(t): class_agnostic_ap(predictions, gts, t)
                         for t in THRESHOLDS}
    expected = [43.350321, 39.147075, 19.561201]
    actual = [metrics['tier2_score'][str(t)]['ap'] for t in THRESHOLDS]
    assert np.allclose(actual, expected, atol=5e-4), (actual, expected)
    added_total = {name: int(sum(values.values()))
                   for name, values in plan['actual_added'].items()}
    result = {'protocol': {'seed': SEED, 'dedup_iou': DEDUP_IOU,
                           'price': TIER2_PRICE,
                           'baseline': str(BASELINE),
                           'tail_cache': str(TAIL),
                           'question': 'design vs merely more candidates',
                           'count_note': 'same requested count; deduplicated pools may '
                                         'yield fewer actual additions'},
              'plan': plan, 'metrics': metrics,
              'actual_added_total': added_total,
              'deltas': {name: {str(t): {k: metrics[name][str(t)][k] -
                                         metrics['baseline'][str(t)][k]
                                         for k in ('ap', 'tp', 'fp')}
                                for t in THRESHOLDS}
                         for name in metrics if name != 'baseline'},
              'input_bindings_no_hash': {'baseline': str(BASELINE),
                                         'run': str(run)}}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)
                           + '\n')
    print(f"baseline {['%.4f' % metrics['baseline'][str(t)]['ap'] for t in THRESHOLDS]}")
    for name in metrics:
        if name == 'baseline':
            continue
        d = result['deltas'][name]
        print(f"{name:15s} dAP={d['0.15']['ap']:+.4f}/{d['0.25']['ap']:+.4f}/"
              f"{d['0.5']['ap']:+.4f}  dTP15={d['0.15']['tp']:+.0f} "
              f"dFP15={d['0.15']['fp']:+.0f}  dTP50={d['0.5']['tp']:+.0f} "
              f"dFP50={d['0.5']['fp']:+.0f} added={added_total[name]}")
    req = np.asarray(list(plan['requested'].values()))
    pool = np.asarray(list(plan['tail_pool'].values()))
    used = np.asarray(list(plan['tail_used'].values()))
    print(f"births requested total={req.sum()} tail_pool median={np.median(pool)}"
          f" satisfied={int((used == req).sum())}/100 scenes")


if __name__ == '__main__':
    main()
