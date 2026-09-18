#!/usr/bin/env python3
"""Audit deterministic M1-A scores and a strictly causal bounded replay.

No detector/lifter is rerun. Frozen per-keyframe raw scores and lifted boxes
are fed in temporal order to the same state class used by the live pipeline.
GT is loaded only after all prediction streams have been materialized.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boxfusion.m1_anchor_online import OnlineAnchorRecovery, deterministic_tail_score
from tools.audit_ca1m_nms_child_headroom import sha256, valid_boxes
from tools.audit_m1m2_remaining_children import GT_ROOT, read_prediction
from tools.audit_seedless_ablation_matrix import BASELINES, RUNS, THRESHOLDS, build_scannet_eval
from tools.audit_tier2_same_budget import cluster_states
from tools.true_fusion_audit_core import class_agnostic_ap


PRICE = 0.05
POOL = 'score_m300'
MAX_ACTIVE = 4096
MAX_BIRTHS = 640


def append(prefix, extra, scores, transform, align):
    boxes, prefix_scores = prefix
    merged = np.concatenate([boxes, extra]) if len(extra) else boxes
    values = np.concatenate([prefix_scores, scores]) if len(extra) else prefix_scores
    return transform(merged, align), values


def replay_scene(scene, run, protocol, bounded: bool):
    tracker = OnlineAnchorRecovery(
        scene, voxel_m=.3, min_views=3,
        max_active=MAX_ACTIVE if bounded else 10**9,
        max_births=MAX_BIRTHS if bounded else 10**9)
    directory = run / 'scenes' / scene
    selection = json.loads((directory / 'selection.json').read_text())
    elapsed = 0.0
    observations = 0
    for ordinal, row in enumerate(selection['frames']):
        frame = int(row['frame'])
        with np.load(directory / f'lifted_{frame:06d}.npz',
                     allow_pickle=False) as values:
            union, corners = values['anchor_ids'], values['corners']
        with np.load(run / 'raw' / scene / f'raw_{frame:06d}.npz',
                     allow_pickle=False) as values:
            raw_scores = values['scores']
        ids = np.asarray(row['selected_anchor_ids'][POOL], dtype=np.int64)
        positions = np.searchsorted(union, ids)
        if not np.array_equal(union[positions], ids):
            raise ValueError(f'anchor mismatch: {scene}/{frame}')
        started = time.perf_counter()
        tracker.update(ordinal, frame, ids, corners[positions], raw_scores[ids])
        elapsed += time.perf_counter() - started
        observations += len(ids)
    return tracker, elapsed, observations


def evaluate(dataset):
    run = RUNS[dataset]
    protocol = json.loads((run / 'protocol.json').read_text())
    scenes = protocol['scenes']
    if dataset == 'scannet':
        gts, aligns, transform = build_scannet_eval(protocol)
    else:
        gts = {s: valid_boxes(np.load(GT_ROOT / s / 'after_filter_boxes.npy'), s)
               for s in scenes}
        aligns = {s: None for s in scenes}
        transform = lambda boxes, align: boxes

    names = ('baseline', 'offline_flat', 'offline_rank',
             'online_unbounded_flat', 'online_unbounded_rank',
             'online_bounded_rank')
    predictions = {name: {} for name in names}
    score_lists = {name: [] for name in names if 'rank' in name}
    totals = {name: 0 for name in names if name != 'baseline'}
    timing = {'seconds': 0.0, 'observations': 0}
    diagnostics = {'unbounded': {}, 'bounded': {}}
    hashes = {str((run / 'protocol.json').resolve()): sha256(run / 'protocol.json')}

    for scene in scenes:
        prefix_path = BASELINES[dataset] / f'{scene}_boxes.pkl'
        prefix = read_prediction(prefix_path)
        hashes[str(prefix_path.resolve())] = sha256(prefix_path)
        predictions['baseline'][scene] = (transform(prefix[0], aligns[scene]), prefix[1])

        states = cluster_states(scene, run, POOL)
        n3 = sum(len(c['frames']) >= 3 for c in states)
        selected = states[:n3]
        offline_boxes = np.asarray([c['box'] for c in selected], dtype=float).reshape(-1, 8, 3)
        offline_scores = np.asarray([
            deterministic_tail_score(-c['rank'][0], scene, c['rank'][1], c['rank'][2])
            for c in selected], dtype=float)
        predictions['offline_flat'][scene] = append(
            prefix, offline_boxes, np.full(n3, PRICE), transform, aligns[scene])
        predictions['offline_rank'][scene] = append(
            prefix, offline_boxes, offline_scores, transform, aligns[scene])
        totals['offline_flat'] += n3
        totals['offline_rank'] += n3
        score_lists['offline_rank'].extend(offline_scores.tolist())

        unbounded, elapsed, observations = replay_scene(scene, run, protocol, False)
        bounded, elapsed_b, observations_b = replay_scene(scene, run, protocol, True)
        if observations_b != observations:
            raise AssertionError('replay observation mismatch')
        timing['seconds'] += elapsed_b
        timing['observations'] += observations_b
        diagnostics['unbounded'][scene] = unbounded.diagnostics()
        diagnostics['bounded'][scene] = bounded.diagnostics()
        for label, tracker in (('online_unbounded', unbounded),
                               ('online_bounded', bounded)):
            rows = tracker.rows()
            boxes = np.asarray([r['box'] for r in rows], dtype=float).reshape(-1, 8, 3)
            ranked = np.asarray([r['score'] for r in rows], dtype=float)
            flat_name = label + '_flat'
            rank_name = label + '_rank'
            if flat_name in predictions:
                predictions[flat_name][scene] = append(
                    prefix, boxes, np.full(len(rows), PRICE), transform, aligns[scene])
                totals[flat_name] += len(rows)
            predictions[rank_name][scene] = append(
                prefix, boxes, ranked, transform, aligns[scene])
            totals[rank_name] += len(rows)
            score_lists[rank_name].extend(ranked.tolist())

    uniqueness = {}
    for name, values in score_lists.items():
        array = np.asarray(values, dtype=np.float64)
        uniqueness[name] = {
            'scores': len(array), 'unique_scores': len(np.unique(array)),
            'ties': len(array) - len(np.unique(array)),
            'minimum': float(array.min()) if len(array) else None,
            'maximum': float(array.max()) if len(array) else None,
            'all_below_0.05': bool((array < .05).all()),
        }
        if uniqueness[name]['ties']:
            raise AssertionError((dataset, name, uniqueness[name]))

    metrics = {name: {str(t): class_agnostic_ap(pred, gts, t)
                      for t in THRESHOLDS}
               for name, pred in predictions.items()}
    losses = {
        'bounded_vs_unbounded_births':
            totals['online_bounded_rank'] - totals['online_unbounded_rank'],
        'bounded_scenes_with_active_pruning': sum(
            d['dropped_active'] > 0 for d in diagnostics['bounded'].values()),
        'bounded_scenes_hitting_birth_cap': sum(
            d['dropped_births'] > 0 for d in diagnostics['bounded'].values()),
        'peak_active': max(d['peak_active'] for d in diagnostics['bounded'].values()),
        'peak_births': max(d['peak_births'] for d in diagnostics['bounded'].values()),
    }
    return {
        'dataset': dataset,
        'protocol': {
            'pool': POOL, 'confirmation_distinct_frames': 3,
            'voxel_m': .3, 'max_active': MAX_ACTIVE,
            'max_births': MAX_BIRTHS,
            'online_definition': ('one pass in increasing keyframe order; birth occurs '
                                  'on the third distinct observed frame; no final-map, '
                                  'future-frame, scene-length, or GT query'),
            'rank_score': ('monotone float32 raw-score bits plus deterministic '
                           '20-bit scene/frame/anchor tie-break, mapped to [0.040001,0.049999]'),
            'gt_use': 'evaluation only after predictions are frozen',
        },
        'totals': totals,
        'metrics': metrics,
        'score_uniqueness': uniqueness,
        'bounded_losses': losses,
        'state_update_timing': {
            **timing,
            'microseconds_per_observation':
                timing['seconds'] * 1e6 / max(timing['observations'], 1),
        },
        'diagnostics': diagnostics,
        'input_sha256': hashes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'reports/m1a_online_ties_20260916/results.json')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {ds: evaluate(ds) for ds in ('scannet', 'ca1m')}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for ds, block in result.items():
        print(f'== {ds} ==')
        for name, metric in block['metrics'].items():
            values = [metric[str(t)]['ap'] for t in THRESHOLDS]
            print(f'{name:24s} {values[0]:8.4f}/{values[1]:8.4f}/{values[2]:8.4f} '
                  f'births={block["totals"].get(name, 0)}')
        print('bounded', block['bounded_losses'])
        print('timing', block['state_update_timing'])


if __name__ == '__main__':
    main()
