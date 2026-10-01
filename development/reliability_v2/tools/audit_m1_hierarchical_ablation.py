#!/usr/bin/env python3
"""Factorial audit for the hierarchical M1 proposal-recovery module.

M1-P is the existing post-NMS proposal/child funnel.  M1-A is the former
"tier-2" pre-NMS anchor-tail recovery.  This audit appends the *same frozen
M1-A births*, with the same score and order, to three prefixes so that the
paper can separate complementarity from a change in candidate budget:

  base, P, A, P+A, P+M2, P+A+M2.

The M1-A candidates are reconstructed from GT-free frozen caches.  GT is
accessed only by the evaluator.  This is a compositional offline ablation; it
does not establish strict-online execution of M1-A.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.audit_ca1m_nms_child_headroom import sha256, valid_boxes
from tools.audit_final_ledger import build_tier2_raw, scannet_inputs, TIER2_PRICE
from tools.audit_m1m2_remaining_children import GT_ROOT, read_prediction
from tools.audit_seedless_ablation_matrix import RUNS, THRESHOLDS
from tools.true_fusion_audit_core import class_agnostic_ap


PREFIXES = {
    'scannet': {
        'base': ROOT / 'results/scannet_t05_boxer_kfmap_score05',
        'p': ROOT / 'results/causal_m1_only_v9',
        'p_m2': ROOT / 'results/scannet_m2nl_m5_dual_full100/persistent',
    },
    'ca1m': {
        'base': ROOT / 'results/ca1m_thr15',
        'p': ROOT / 'results/ca1m_dual_nom2',
        'p_m2': ROOT / 'results/ca1m_dual',
    },
}

EXPECTED_FULL = {
    'scannet': (43.350321, 39.147075, 19.561201),
    'ca1m': (47.560329, 39.577718, 17.626631),
}


def evaluate(dataset: str, pool: str) -> dict:
    protocol = json.loads((RUNS[dataset] / 'protocol.json').read_text())
    scenes = protocol['scenes']
    if dataset == 'scannet':
        gts, aligns, transform = scannet_inputs()
    else:
        gts = {s: valid_boxes(np.load(GT_ROOT / s / 'after_filter_boxes.npy'), s)
               for s in scenes}
        aligns = {s: None for s in scenes}
        transform = lambda boxes, align: boxes

    # Rebuild the archived M1-A arm once, then isolate its appended rows.  The
    # main score pool is independent of the final-map residual mask: it is the
    # per-frame score top-M followed by Boxer lifting and voxel clustering.
    _, archived, births_total = build_tier2_raw(dataset, pool)
    frozen_baseline = PREFIXES[dataset]['p_m2']
    births = {}
    for scene in scenes:
        baseline_boxes, _ = read_prediction(
            frozen_baseline / f'{scene}_boxes.pkl')
        merged_boxes, merged_scores = archived[scene]
        n = len(baseline_boxes)
        if not np.array_equal(merged_boxes[:n], baseline_boxes):
            raise AssertionError(f'Archived prefix mismatch: {dataset}/{scene}')
        extra = merged_boxes[n:]
        extra_scores = merged_scores[n:]
        if len(extra) and not np.all(extra_scores == TIER2_PRICE):
            raise AssertionError(f'Unexpected M1-A price: {dataset}/{scene}')
        births[scene] = extra
    if sum(map(len, births.values())) != births_total:
        raise AssertionError('Birth count mismatch')

    arms = {name: {} for name in ('base', 'p', 'a', 'p_a', 'p_m2', 'p_a_m2',
                                  'base_m2', 'a_m2')}
    hashes = {str((RUNS[dataset] / 'protocol.json').resolve()):
              sha256(RUNS[dataset] / 'protocol.json')}
    counts = {name: {'prefix': 0, 'added_a': 0, 'total': 0}
              for name in arms}
    for scene in scenes:
        loaded = {}
        for key, directory in PREFIXES[dataset].items():
            path = directory / f'{scene}_boxes.pkl'
            loaded[key] = read_prediction(path)
            hashes[str(path.resolve())] = sha256(path)
        extra = births[scene]
        extra_scores = np.full(len(extra), TIER2_PRICE, dtype=float)
        n_base = len(loaded['base'][0])
        # base_m2/a_m2 need row identity between the base map and the p_m2
        # map. ScanNet keeps it bit-for-bit (paired prefix); the CA-1M funnel
        # rewrites map rows, so the two arms are ScanNet-only and the CA-1M
        # omission is disclosed rather than approximated.
        paired = np.array_equal(loaded['p_m2'][0][:n_base],
                                loaded['base'][0]) and \
            len(loaded['p_m2'][0]) >= n_base
        if not paired:
            for arm in ('base_m2', 'a_m2'):
                arms[arm][scene] = None
            extra_counts = {}
        else:
            m2_native = (loaded['p_m2'][0][:n_base],
                         loaded['p_m2'][1][:n_base])
            extra_counts = {
                'base_m2': (m2_native[0], m2_native[1]),
                'a_m2': (np.concatenate([m2_native[0], extra]) if len(extra)
                         else m2_native[0],
                         np.concatenate([m2_native[1], extra_scores])
                         if len(extra) else m2_native[1])}
        specs = {
            'base': ('base', False),
            'p': ('p', False),
            'a': ('base', True),
            'p_a': ('p', True),
            'p_m2': ('p_m2', False),
            'p_a_m2': ('p_m2', True),
        }
        for arm, (prefix, add_a) in specs.items():
            boxes, scores = loaded[prefix]
            if add_a and len(extra):
                boxes = np.concatenate([boxes, extra])
                scores = np.concatenate([scores, extra_scores])
            arms[arm][scene] = (transform(boxes, aligns[scene]), scores)
            counts[arm]['prefix'] += len(loaded[prefix][0])
            counts[arm]['added_a'] += len(extra) if add_a else 0
            counts[arm]['total'] += len(boxes)
        for arm, (boxes, scores) in extra_counts.items():
            arms[arm][scene] = (transform(boxes, aligns[scene]), scores)
            counts[arm]['prefix'] += len(boxes) - len(extra)
            counts[arm]['added_a'] += len(extra) if arm == 'a_m2' else 0
            counts[arm]['total'] += len(boxes)

    metrics = {}
    for arm, predictions in arms.items():
        if predictions[scenes[0]] is None:
            metrics[arm] = None
        else:
            metrics[arm] = {str(t): class_agnostic_ap(predictions, gts, t)
                            for t in THRESHOLDS}
    actual_full = tuple(metrics['p_a_m2'][str(t)]['ap'] for t in THRESHOLDS)
    if not np.allclose(actual_full, EXPECTED_FULL[dataset], atol=5e-4):
        raise AssertionError((dataset, actual_full, EXPECTED_FULL[dataset]))
    a_counts = {counts[name]['added_a'] for name in ('a', 'p_a', 'p_a_m2')}
    if metrics['base_m2'] is not None:
        a_counts.add(counts['a_m2']['added_a'])
    if a_counts != {births_total}:
        raise AssertionError((dataset, a_counts, births_total))
    ap = lambda arm, t: metrics[arm][str(t)]['ap']
    paired_m2 = metrics['base_m2'] is not None
    effects = {}
    for t in THRESHOLDS:
        k = str(t)
        effects[k] = {
            'P_over_base': ap('p', t) - ap('base', t),
            'A_over_base': ap('a', t) - ap('base', t),
            'A_over_P': ap('p_a', t) - ap('p', t),
            'P_over_A': ap('p_a', t) - ap('a', t),
            'P_A_interaction': (ap('p_a', t) - ap('p', t)
                                - ap('a', t) + ap('base', t)),
            'M2_over_P': ap('p_m2', t) - ap('p', t),
            'M2_over_P_A': ap('p_a_m2', t) - ap('p_a', t),
            'A_over_P_M2': ap('p_a_m2', t) - ap('p_m2', t),
            'full_over_base': ap('p_a_m2', t) - ap('base', t),
        }
        if paired_m2:
            effects[k].update({
                'M2_over_base': ap('base_m2', t) - ap('base', t),
                'M2_over_A': ap('a_m2', t) - ap('a', t),
                'A_over_base_m2': ap('a_m2', t) - ap('base_m2', t),
            })

    return {
        'dataset': dataset,
        'protocol': {
            'scenes': len(scenes),
            'pool': pool,
            'm1_a_name': 'anchor-level tail recovery (former tier-2)',
            'm1_p_name': 'proposal/child-level recovery (former M1)',
            'm1_a_price': TIER2_PRICE,
            'births_are_identical_across_a_arms': True,
            'evaluation_use_of_gt_only': True,
            'scope': 'offline compositional ablation; not a strict-online claim',
            'tie_policy': ('all M1-A births use flat 0.05; archived stable order '
                           'is retained, so equal-score order remains a disclosed limitation'),
        },
        'counts': counts,
        'metrics': metrics,
        'effects': effects,
        'input_sha256': hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'reports/m1_hierarchical_ablation_20260916/results.json')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        'scannet': evaluate('scannet', 'score_m300'),
        'ca1m': evaluate('ca1m', 'score_m300'),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for dataset, block in result.items():
        print(f'== {dataset} ==')
        for arm in ('base', 'p', 'a', 'p_a', 'p_m2', 'p_a_m2'):
            values = [block['metrics'][arm][str(t)]['ap'] for t in THRESHOLDS]
            c = block['counts'][arm]
            print(f"{arm:7s} {values[0]:8.4f}/{values[1]:8.4f}/{values[2]:8.4f} "
                  f"boxes={c['total']} added_A={c['added_a']}")
        for t in THRESHOLDS:
            e = block['effects'][str(t)]
            print(f"IoU{t}: A|base={e['A_over_base']:+.4f} A|P={e['A_over_P']:+.4f} "
                  f"interaction={e['P_A_interaction']:+.4f} "
                  f"M2|PA={e['M2_over_P_A']:+.4f}")


if __name__ == '__main__':
    main()
