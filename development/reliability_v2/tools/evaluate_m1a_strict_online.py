#!/usr/bin/env python3
"""Evaluate completed strict-online M1-A predictions and aggregate traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.audit_ca1m_nms_child_headroom import sha256, valid_boxes
from tools.audit_m1m2_remaining_children import GT_ROOT, read_prediction
from tools.audit_m1_hierarchical_ablation import PREFIXES
from tools.audit_seedless_ablation_matrix import BASELINES, RUNS, THRESHOLDS, build_scannet_eval
from tools.true_fusion_audit_core import class_agnostic_ap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('scannet', 'ca1m'), required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((RUNS[args.dataset] / 'protocol.json').read_text())
    scenes = protocol['scenes']
    manifest = json.loads((args.run / 'manifest.json').read_text())
    if manifest['scenes'] != scenes:
        raise RuntimeError('manifest does not bind the full frozen scene list')
    if args.dataset == 'scannet':
        gts, aligns, transform = build_scannet_eval(protocol)
    else:
        gts = {s: valid_boxes(np.load(GT_ROOT / s / 'after_filter_boxes.npy'), s)
               for s in scenes}
        aligns = {s: None for s in scenes}
        transform = lambda boxes, align: boxes
    baseline, online = {}, {}
    factorial = {name: {} for name in ('base', 'p', 'a', 'p_a', 'p_m2',
                                       'p_a_m2', 'base_m2', 'a_m2')}
    traces, hashes, tail_scores = {}, {}, []
    for scene in scenes:
        bp = BASELINES[args.dataset] / f'{scene}_boxes.pkl'
        op = args.run / 'predictions' / f'{scene}_boxes.pkl'
        tp = args.run / 'traces' / f'{scene}.json'
        if not op.is_file() or not tp.is_file():
            raise RuntimeError(f'incomplete strict-online run: {scene}')
        bb, bs = read_prediction(bp)
        ob, os = read_prediction(op)
        trace = json.loads(tp.read_text())
        if (not trace['completed'] or trace['baseline_rows'] != len(bb)
                or trace['output_rows'] != len(ob)
                or len(ob) - len(bb) != trace['birth_rows']):
            raise RuntimeError(f'trace/output mismatch: {scene}')
        if not np.array_equal(ob[:len(bb)], bb) or not np.array_equal(os[:len(bs)], bs):
            raise RuntimeError(f'baseline prefix changed: {scene}')
        baseline[scene] = (transform(bb, aligns[scene]), bs)
        online[scene] = (transform(ob, aligns[scene]), os)
        birth_boxes, birth_scores = ob[len(bb):], os[len(bs):]
        native_boxes, native_scores = read_prediction(
            PREFIXES[args.dataset]['base'] / f'{scene}_boxes.pkl')
        p_boxes, p_scores = read_prediction(
            PREFIXES[args.dataset]['p'] / f'{scene}_boxes.pkl')
        raw_arms = {
            'base': (native_boxes, native_scores),
            'p': (p_boxes, p_scores),
            'a': (np.concatenate([native_boxes, birth_boxes]),
                  np.concatenate([native_scores, birth_scores])),
            'p_a': (np.concatenate([p_boxes, birth_boxes]),
                    np.concatenate([p_scores, birth_scores])),
            'p_m2': (bb, bs),
            'p_a_m2': (ob, os),
        }
        # M2-rescored native prefix of the online run's own baseline map;
        # a_m2 reuses those row scores with the M1-P rows removed. Row
        # identity holds on ScanNet (paired prefix); the CA-1M funnel
        # rewrites map rows, so the two arms stay ScanNet-only and the
        # CA-1M omission is disclosed rather than approximated.
        paired = np.array_equal(bb[:len(native_boxes)], native_boxes) and \
            len(bb) >= len(native_boxes)
        if paired:
            m2_native = (bb[:len(native_boxes)], bs[:len(native_scores)])
            raw_arms['base_m2'] = m2_native
            raw_arms['a_m2'] = (
                np.concatenate([m2_native[0], birth_boxes]),
                np.concatenate([m2_native[1], birth_scores]))
        for name, (boxes, scores) in raw_arms.items():
            factorial[name][scene] = (transform(boxes, aligns[scene]), scores)
        tail_scores.extend(os[len(bs):].tolist())
        traces[scene] = trace
        hashes[str(op.resolve())] = sha256(op)
        hashes[str(tp.resolve())] = sha256(tp)
    tail_scores = np.asarray(tail_scores, dtype=np.float64)
    if (len(np.unique(tail_scores)) != len(tail_scores)
            or (len(tail_scores) and tail_scores.max() >= .05)):
        raise RuntimeError('global strict-online scores are tied or leave tail interval')
    metrics = {
        'baseline': {str(t): class_agnostic_ap(baseline, gts, t)
                     for t in THRESHOLDS},
        'strict_online_m1a': {str(t): class_agnostic_ap(online, gts, t)
                              for t in THRESHOLDS},
    }
    factorial_metrics = {}
    for name, predictions in factorial.items():
        if predictions.get(scenes[0]) is None:
            factorial_metrics[name] = None
            continue
        factorial_metrics[name] = {str(t): class_agnostic_ap(predictions, gts,
                                                             t)
                                   for t in THRESHOLDS}
    deltas = {str(t): {k: metrics['strict_online_m1a'][str(t)][k]
                           - metrics['baseline'][str(t)][k]
                       for k in ('ap', 'tp', 'fp')}
              for t in THRESHOLDS}
    frames = sum(len(t['frames']) for t in traces.values())
    summary = {
        'schema': 'boxfusion.m1a.strict_online.evaluation.v1',
        'dataset': args.dataset, 'scene_count': len(scenes),
        'frame_count': frames,
        'births': int(len(tail_scores)),
        'score_uniqueness': {'scores': int(len(tail_scores)),
                             'unique': int(len(np.unique(tail_scores))),
                             'ties': 0,
                             'minimum': float(tail_scores.min()),
                             'maximum': float(tail_scores.max())},
        'metrics': metrics, 'deltas': deltas,
        'factorial_metrics': factorial_metrics,
        'online_state': {
            'max_active_bound': manifest['parameters']['max_active'],
            'max_births_bound': manifest['parameters']['max_births'],
            'observed_peak_active': max(t['tracker']['peak_active'] for t in traces.values()),
            'observed_peak_births': max(t['tracker']['peak_births'] for t in traces.values()),
            'scenes_with_active_drops': sum(t['tracker']['dropped_active'] > 0
                                            for t in traces.values()),
            'scenes_with_birth_drops': sum(t['tracker']['dropped_births'] > 0
                                           for t in traces.values()),
        },
        'runtime': {
            'sum_wall_seconds': sum(t['wall_seconds'] for t in traces.values()),
            'sum_lift_seconds': sum(t['lift_seconds'] for t in traces.values()),
            'sum_state_update_seconds': sum(t['state_update_seconds'] for t in traces.values()),
            'mean_wall_ms_per_keyframe': 1000 * sum(t['wall_seconds'] for t in traces.values()) / frames,
            'mean_lift_ms_per_keyframe': 1000 * sum(t['lift_seconds'] for t in traces.values()) / frames,
            'mean_state_ms_per_keyframe': 1000 * sum(t['state_update_seconds'] for t in traces.values()) / frames,
            'scope': 'raw detector outputs cached; Boxer single-pool lift and online state measured live',
        },
        'input_sha256': hashes,
        'limits': [
            'Development scene lists already used in method selection; not held out.',
            'Runtime excludes WeDetect forward because frozen raw outputs are replayed.',
            'Fixed birth cap bounds M1-A state but may limit longer/unseen sequences.',
        ],
    }
    write = args.run / 'evaluation.json'
    write.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({
        'dataset': args.dataset, 'births': summary['births'],
        'baseline': [metrics['baseline'][str(t)]['ap'] for t in THRESHOLDS],
        'online': [metrics['strict_online_m1a'][str(t)]['ap'] for t in THRESHOLDS],
        'delta': [deltas[str(t)]['ap'] for t in THRESHOLDS],
        'factorial': {name: ([rows[str(t)]['ap'] for t in THRESHOLDS]
                             if rows is not None else None)
                      for name, rows in factorial_metrics.items()},
        'state': summary['online_state'], 'runtime': summary['runtime']},
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
