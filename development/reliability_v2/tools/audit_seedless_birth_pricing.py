"""Birth-price calibration for the seedless module (training-free).

Open-Det-style quality-aware birth scoring, instantiated training-free: the
repo's own M1d size-price table (EDGES=(0.3,0.5,0.7,1.0) -> (0.05,0.10,0.25,
0.40,0.50), tools/integrated_online.py:46) versus flat prices. Selection is
frozen to the ablation winner (score_matched: per-scene count matched to ge3,
ranked by exemplar raw score); every price arm therefore appends the IDENTICAL
boxes and varies only the price — a clean isolation of the pricing question.

Pure offline post-processing of the full107/full100 frozen runs; GT used only
for evaluation. Prices are tuned on development scenes and must be disclosed
as such.
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
from tools.audit_m1m2_remaining_children import read_prediction
from tools.ca1m_seedless_trigger_core import add_clusters
from tools.true_fusion_audit_core import aabb_iou, class_agnostic_ap
from tools import audit_ca1m_seedless_step1c as pilot
from tools.audit_seedless_ablation_matrix import (
    RUNS, BASELINES, EXPECTED, POOLS, THRESHOLDS, build_scannet_eval)

M1D_EDGES = (0.3, 0.5, 0.7, 1.0)
M1D_TABLE = (0.05, 0.10, 0.25, 0.40, 0.50)
FLAT_PRICES = (0.05, 0.10, 0.15, 0.20)


def size_price(box):
    edge = float((box.max(0) - box.min(0)).max())
    for bound, value in zip(M1D_EDGES, M1D_TABLE):
        if edge < bound:
            return value
    return M1D_TABLE[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=tuple(RUNS), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f'Refusing to overwrite {args.output}')
    run = RUNS[args.dataset]
    protocol = json.loads((run / 'protocol.json').read_text())
    inputs = {str(p.resolve()): sha256(p) for p in (Path(__file__), run / 'protocol.json')}
    if args.dataset == 'ca1m':
        gts = {s: valid_boxes(np.load(pilot.GT_ROOT / s / 'after_filter_boxes.npy'), s)
               for s in protocol['scenes']}
        aligns = {s: None for s in protocol['scenes']}
        transform = lambda boxes, align: boxes
    else:
        gts, aligns, transform = build_scannet_eval(protocol)
    price_arms = [f'flat_{p:g}' for p in FLAT_PRICES] + ['size_m1d']
    baseline = {}
    predictions = {f'{pool}_{arm}': {} for pool in POOLS for arm in price_arms}
    stats = {name: Counter() for name in predictions}
    for ordinal, scene in enumerate(protocol['scenes'], 1):
        directory = run / 'scenes' / scene
        selection = json.loads((directory / 'selection.json').read_text())
        inputs[str((directory / 'selection.json').resolve())] = sha256(directory / 'selection.json')
        boxes, scores = read_prediction(BASELINES[args.dataset] / f'{scene}_boxes.pkl')
        inputs[str((BASELINES[args.dataset] / f'{scene}_boxes.pkl').resolve())] = \
            sha256(BASELINES[args.dataset] / f'{scene}_boxes.pkl')
        baseline[scene] = (transform(boxes, aligns[scene]), scores)
        for pool in POOLS:
            clusters = {}
            for row in selection['frames']:
                frame = row['frame']
                lift_path = directory / f'lifted_{frame:06d}.npz'
                raw_path = run / 'raw' / scene / f'raw_{frame:06d}.npz'
                with np.load(lift_path, allow_pickle=False) as values:
                    union, corners = values['anchor_ids'], values['corners']
                with np.load(raw_path, allow_pickle=False) as values:
                    raw_scores = values['scores']
                ids = np.asarray(row['selected_anchor_ids'][pool], dtype=np.int64)
                positions = np.searchsorted(union, ids)
                if not np.array_equal(union[positions], ids):
                    raise ValueError(f'Anchor mismatch: {scene}/{frame}/{pool}')
                add_clusters(clusters, frame, ids, corners[positions], raw_scores[ids], .3)
                for path in (lift_path, raw_path):
                    inputs[str(path.resolve())] = sha256(path)
            states = sorted(clusters.values(), key=lambda c: c['rank'])
            n3 = sum(len(c['frames']) >= 3 for c in states)
            born = [c['box'] for c in states[:n3]]
            for arm in price_arms:
                name = f'{pool}_{arm}'
                if arm == 'size_m1d':
                    prices = [size_price(box) for box in born]
                else:
                    prices = [float(arm.split('_')[1])] * len(born)
                stats[name]['births'] += len(born)
                stats[name]['mean_price'] += float(np.sum(prices))
                predictions[name][scene] = (
                    transform(np.concatenate([boxes, np.asarray(born)])
                              if len(born) else boxes, aligns[scene]),
                    np.r_[scores, prices] if len(born) else scores)
        if ordinal % 10 == 0 or ordinal == len(protocol['scenes']):
            print(f'{args.dataset} {ordinal}/{len(protocol["scenes"])} scenes', flush=True)
    metrics = {'baseline': {str(t): class_agnostic_ap(baseline, gts, t) for t in THRESHOLDS}}
    for name, arms in predictions.items():
        metrics[name] = {}
        for t in THRESHOLDS:
            value = class_agnostic_ap(arms, gts, t)
            for key in ('ap', 'tp', 'fp'):
                value['delta_' + key] = value[key] - metrics['baseline'][str(t)][key]
            metrics[name][str(t)] = value
    actual = [metrics['baseline'][str(t)]['ap'] for t in THRESHOLDS]
    assert np.allclose(actual, EXPECTED[args.dataset], atol=5e-4, rtol=0), actual
    for path, digest in inputs.items():
        assert sha256(path) == digest, f'Input changed: {path}'
    result = {
        'schema': 'boxfusion.seedless_birth_pricing.v1', 'completed': True,
        'dataset': args.dataset, 'run': str(run),
        'selection': 'score_matched (per-scene ge3 count, exemplar raw-score order) '
                     'from the ablation matrix; identical boxes across price arms',
        'prices': {'flat': list(FLAT_PRICES), 'size_m1d': {
            'edges': list(M1D_EDGES), 'table': list(M1D_TABLE)}},
        'stats': {k: dict(v) for k, v in stats.items()},
        'metrics': metrics,
        'limits': [
            'Prices tuned on development scenes; no held-out validation',
            'Selection frozen; only the appended price varies across arms',
            'GT used for evaluation only; offline post-processing of frozen runs',
        ],
    }
    args.output.mkdir(parents=True)
    for name, data in (('results.json', json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'),
                       ('input_sha256.json', json.dumps(inputs, ensure_ascii=False, indent=2) + '\n')):
        with (args.output / name).open('x') as handle:
            handle.write(data)
    print(json.dumps({
        'deltas': {name: [round(metrics[name][str(t)]['delta_ap'], 4) for t in THRESHOLDS]
                   for name in predictions}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
