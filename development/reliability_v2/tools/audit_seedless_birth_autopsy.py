"""Seedless birth autopsy + FSOD-VFM-inspired fragment competition (offline).

Question (user screening report, 2026-09-12): the tier-2 module births many
unmatched boxes — are they FRAGMENTS of larger births/map rows (chair back vs
chair), DUPLICATES of found objects, or isolated localization noise? And does
directed fragment suppression (in spirit of FSOD-VFM's graph competition) beat
simple score ranking at MATCHED birth counts?

Per dataset (frozen full107/full100 artifacts, no new inference):
  autopsy       classify every ge3 birth: matches-existing / matches-missed /
                duplicate-of-baseline (IoU>0.5 vs map row), fragment (>=0.8 of
                its AABB volume inside a larger birth or map row), isolated;
  arms (all at the same per-scene birth count as the score-ranked reference):
                score_top    top-N3 exemplars by raw score (reference);
                dedup05      drop IoU>0.5 baseline duplicates, top up by score;
                graphcomp    drop directed-fragment-dominated births (contained
                >=0.8 in a larger, higher-score birth or map row), top up.

Harm metric: suppressed births that matched MISSED GT (the cup-on-table
casualties the user flagged). GT only for evaluation. Pre-registered bar:
an arm wins only if it beats score_top by >= +0.1 AP15 at matched counts.
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
from tools.audit_m1m2_remaining_children import read_prediction
from tools.ca1m_seedless_trigger_core import add_clusters
from tools import audit_ca1m_seedless_step1c as pilot
from tools.audit_seedless_ablation_matrix import (
    RUNS, BASELINES, EXPECTED, THRESHOLDS, build_scannet_eval)

POOLS = ('score_m300',)
CONTAIN = 0.8
DUP_IOU = 0.5
APPEND_SCORE = 0.05
ARMS = ('score_top', 'dedup05', 'graphcomp', 'graph_only', 'score_cut_nG',
        'diffusion_cut')


def volumes(corners):
    return np.prod(corners.max(1) - corners.min(1), axis=1)


def iou_matrix(a, b):
    return aabb_iou(np.asarray(a), np.asarray(b))


def intersection_volumes(a, b):
    alo, ahi = a.min(1), a.max(1)
    blo, bhi = b.min(1), b.max(1)
    inter = np.maximum(0.0, np.minimum(ahi[:, None], bhi[None])
                       - np.maximum(alo[:, None], blo[None])).prod(2)
    return inter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=tuple(RUNS), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f'Refusing to overwrite {args.output}')
    run = RUNS[args.dataset]
    protocol = json.loads((run / 'protocol.json').read_text())
    inputs = {str(p.resolve()): sha256(p) for p in (
        Path(__file__), run / 'protocol.json')}
    if args.dataset == 'ca1m':
        gts = {s: valid_boxes(np.load(pilot.GT_ROOT / s / 'after_filter_boxes.npy'), s)
               for s in protocol['scenes']}
        aligns = {s: None for s in protocol['scenes']}
        transform = lambda boxes, align: boxes
    else:
        gts, aligns, transform = build_scannet_eval(protocol)
    baseline = {}
    predictions = {f'{pool}_{arm}': {} for pool in POOLS for arm in ARMS}
    autopsy = Counter()
    casualties = {f'{pool}_{arm}': Counter() for pool in POOLS for arm in ARMS}
    birth_counts = {f'{pool}_{arm}': Counter() for pool in POOLS for arm in ARMS}
    for ordinal, scene in enumerate(protocol['scenes'], 1):
        directory = run / 'scenes' / scene
        selection = json.loads((directory / 'selection.json').read_text())
        inputs[str((directory / 'selection.json').resolve())] = sha256(
            directory / 'selection.json')
        boxes, scores = read_prediction(BASELINES[args.dataset] / f'{scene}_boxes.pkl')
        inputs[str((BASELINES[args.dataset] / f'{scene}_boxes.pkl').resolve())] = sha256(
            BASELINES[args.dataset] / f'{scene}_boxes.pkl')
        baseline[scene] = (transform(boxes, aligns[scene]), scores)
        gt = gts[scene]
        biou = aabb_iou(transform(boxes, aligns[scene]), gt).max(0)
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
            ge3 = [c for c in states if len(c['frames']) >= 3]
            born = np.asarray([c['box'] for c in ge3], dtype=float).reshape(-1, 8, 3)
            exemplar_scores = np.asarray([-c['rank'][0] for c in ge3])
            n3 = len(ge3)
            gt_iou = iou_matrix(transform(born, aligns[scene]), gt) if len(gt) \
                else np.zeros((len(born), 0))
            matched = gt_iou.max(1) > .15 if len(gt) else np.zeros(len(born), bool)
            owner = gt_iou.argmax(1) if len(gt) else np.zeros(len(born), int)
            map_iou = iou_matrix(born, boxes)
            map_max = map_iou.max(1) if len(boxes) else np.zeros(len(born))
            owner_map = map_iou.argmax(1) if len(boxes) else np.zeros(len(born), int)
            # ---- autopsy of every birth ----
            inter_birth = intersection_volumes(born, born)
            vol = volumes(born)
            contained = np.zeros(len(born), dtype=bool)
            for i in range(len(born)):
                others = np.flatnonzero(
                    (inter_birth[i] >= CONTAIN * vol[i]) & (vol > vol[i]))
                if len(others):
                    contained[i] = exemplar_scores[others].max() > exemplar_scores[i]
                elif len(boxes):
                    inter_map = intersection_volumes(born[i:i + 1], boxes)[0]
                    big = np.flatnonzero(
                        (inter_map >= CONTAIN * vol[i])
                        & (volumes(boxes) > vol[i]))
                    if len(big):
                        contained[i] = True
            for i in range(len(born)):
                if matched[i]:
                    autopsy['matches_missed' if biou[owner[i]] <= .15
                            else 'matches_existing'] += 1
                elif map_max[i] > DUP_IOU:
                    autopsy['duplicate_of_baseline'] += 1
                elif contained[i]:
                    autopsy['fragment_of_larger'] += 1
                else:
                    autopsy['isolated_unmatched'] += 1
            # ---- count-matched arms ----
            order = np.argsort(-exemplar_scores, kind='mergesort')
            arms = {}
            arms['score_top'] = order[:n3]
            keep_dedup = order[map_max[order] <= DUP_IOU][:n3]
            arms['dedup05'] = keep_dedup if len(keep_dedup) >= n3 else np.concatenate(
                [keep_dedup, order[~np.isin(order, keep_dedup)]])[:n3]
            suppressed = contained | (map_max > DUP_IOU)
            casualties_mask = suppressed & matched & (biou[owner] <= .15)
            casualties[f'{pool}_graphcomp']['suppressed_total'] += int(suppressed.sum())
            casualties[f'{pool}_graphcomp']['suppressed_missed_tp'] += int(casualties_mask.sum())
            keep_graph_full = order[~suppressed]
            keep_graph = keep_graph_full[:n3]
            arms['graphcomp'] = keep_graph if len(keep_graph) >= n3 else np.concatenate(
                [keep_graph, order[~np.isin(order, keep_graph)]])[:n3]
            # reduced-count arms: suppression WITHOUT top-up vs score truncation
            # at the same reduced count.
            arms['graph_only'] = keep_graph_full
            arms['score_cut_nG'] = order[:len(keep_graph_full)]
            # FSOD-VFM-style directed-graph score diffusion: each fragment
            # donates its exemplar score mass to its dominating parent (one
            # propagation round), then rank by diffused score at the same
            # reduced count as graph_only.
            diffused = exemplar_scores.copy()
            for i in range(len(born)):
                parents = np.flatnonzero(
                    (inter_birth[i] >= CONTAIN * vol[i]) & (vol > vol[i]))
                if len(parents):
                    best = parents[np.argmax(exemplar_scores[parents])]
                    if exemplar_scores[best] > exemplar_scores[i]:
                        diffused[best] += exemplar_scores[i]
                        diffused[i] -= exemplar_scores[i]
            arms['diffusion_cut'] = np.argsort(-diffused, kind='mergesort')[
                :len(keep_graph_full)]
            for arm, sel in arms.items():
                name = f'{pool}_{arm}'
                extra = born[sel]
                birth_counts[name]['births'] += len(sel)
                predictions[name][scene] = (
                    transform(np.concatenate([boxes, extra])
                              if len(sel) else boxes, aligns[scene]),
                    np.r_[scores, np.full(len(sel), APPEND_SCORE)] if len(sel) else scores)
        if ordinal % 10 == 0 or ordinal == len(protocol['scenes']):
            print(f'{args.dataset} {ordinal}/{len(protocol["scenes"])} scenes '
                  f'autopsy={dict(autopsy)}', flush=True)
    metrics = {'baseline': {str(t): class_agnostic_ap(baseline, gts, t) for t in THRESHOLDS}}
    for name, arms_ in predictions.items():
        metrics[name] = {}
        for t in THRESHOLDS:
            value = class_agnostic_ap(arms_, gts, t)
            for key in ('ap', 'tp', 'fp'):
                value['delta_' + key] = value[key] - metrics['baseline'][str(t)][key]
            metrics[name][str(t)] = value
    actual = [metrics['baseline'][str(t)]['ap'] for t in THRESHOLDS]
    assert np.allclose(actual, EXPECTED[args.dataset], atol=5e-4, rtol=0), actual
    for path, digest in inputs.items():
        assert sha256(path) == digest, f'Input changed: {path}'
    result = {
        'schema': 'boxfusion.seedless_birth_autopsy.v1', 'completed': True,
        'dataset': args.dataset, 'autopsy': dict(autopsy),
        'casualties': {k: dict(v) for k, v in casualties.items()},
        'birth_counts': {k: dict(v) for k, v in birth_counts.items()},
        'definitions': {
            'duplicate_of_baseline': 'max IoU vs map row > 0.5',
            'fragment_of_larger': '>= 0.8 of own AABB volume inside a larger birth '
                                  'or map row (higher exemplar score for birth rivals)',
            'arms': 'all arms match the per-scene score_top birth count; '
                    'suppressed sets are topped up by score',
            'append_price': APPEND_SCORE,
        },
        'bar': 'arm wins only if >= +0.1 AP15 over score_top at matched counts',
        'metrics': metrics,
        'limits': [
            'AABB containment approximates FSOD-VFM mask containment; no masks',
            'Fragment/duplicate rules are fixed thresholds, sensitivity untested',
            'Offline post-processing of frozen runs; GT only for evaluation',
        ],
    }
    args.output.mkdir(parents=True)
    for name, data in (('results.json', json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'),
                       ('input_sha256.json', json.dumps(inputs, ensure_ascii=False, indent=2) + '\n')):
        with (args.output / name).open('x') as handle:
            handle.write(data)
    print(json.dumps({
        'autopsy': dict(autopsy),
        'casualties': {k: dict(v) for k, v in casualties.items()},
        'deltas_vs_baseline': {n: [round(metrics[n][str(t)]['delta_ap'], 4)
                                   for t in THRESHOLDS] for n in predictions}},
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
