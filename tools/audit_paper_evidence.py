"""Paper evidence pack A: size grouping, per-scene gains, paired bootstrap.

Three CPU experiments from the evidence plan (reports/recall_fp_budget line),
on the frozen CA-1M full107 paired artifacts (native / +M1 / +M1+M2nl):
  grouping      GT-volume terciles (pre-registered cuts over all 12,911 GT);
                per-group covered-GT recall at IoU .15/.25/.50 (max-IoU
                coverage convention of the audit ledger, disclosed);
  per-scene     per-scene coverage deltas with win/tie/loss counts;
  bootstrap     scene-level paired bootstrap (B=10,000, fixed seed) of pooled
                AP15/25/50 and their deltas, via precomputed per-scene
                (score, TP) pairs — exact per-scene matching of the anchor
                metric, validated to reproduce the pooled AP bit-for-bit
                before any resampling.
No new inference; GT used for evaluation only. All three baseline APs must
reproduce the published ledger values within 5e-4.
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
from tools.true_fusion_audit_core import aabb_iou, class_agnostic_ap
from tools.audit_m1m2_remaining_children import (
    GT_ROOT, SCENES, THRESHOLDS, read_prediction)

CONFIGS = {
    'native': ROOT / 'results/ca1m_thr15',
    'm1': ROOT / 'results/ca1m_thr15_m1_full107',
    'm1m2': ROOT / 'results/ca1m_thr15_m1_m2nl_full107',
}
LEDGER = {
    'native': [44.8526, 36.7678, 15.3999],
    'm1': [45.7277, 37.6384, 15.7733],
    'm1m2': [45.9762, 38.2087, 16.7660],
}
BOOTSTRAPS = 10_000
SEED = 20260915
GROUPS = ('small', 'mid', 'large')


def scene_pairs(corners, scores, gt, threshold):
    """Anchor-exact per-scene greedy matching -> (score, is_tp) pairs."""
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
    return 100.0 * float(np.sum((mr[changes + 1] - mr[changes]) * mp[changes + 1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f'Refusing to overwrite {args.output}')
    scenes = SCENES.read_text().split()
    assert len(scenes) == len(set(scenes)) == 107
    inputs = {str(p.resolve()): sha256(p) for p in (
        Path(__file__), SCENES,
        ROOT / 'tools/true_fusion_audit_core.py',
        ROOT / 'tools/audit_m1m2_remaining_children.py')}
    data, gts = {}, {}
    for name, root in CONFIGS.items():
        data[name] = {}
        for scene in scenes:
            path = root / f'{scene}_boxes.pkl'
            inputs[str(path.resolve())] = sha256(path)
            data[name][scene] = read_prediction(path)
    for scene in scenes:
        gt_path = GT_ROOT / scene / 'after_filter_boxes.npy'
        inputs[str(gt_path.resolve())] = sha256(gt_path)
        gts[scene] = valid_boxes(np.load(gt_path, allow_pickle=False), scene)
    # ---- pairs precompute + pooled-AP bit validation ----
    pairs = {name: {str(t): {} for t in THRESHOLDS} for name in CONFIGS}
    pooled = {name: {} for name in CONFIGS}
    for name in CONFIGS:
        predictions = {s: data[name][s] for s in scenes}
        reference = {str(t): class_agnostic_ap(predictions, gts, t)['ap']
                     for t in THRESHOLDS}
        assert np.allclose([reference[str(t)] for t in THRESHOLDS],
                           LEDGER[name], atol=5e-4, rtol=0), (name, reference)
        for t in THRESHOLDS:
            for scene in scenes:
                pairs[name][str(t)][scene] = scene_pairs(
                    data[name][scene][0], data[name][scene][1], gts[scene], t)
            stacked = np.concatenate([pairs[name][str(t)][s]
                                      for s in sorted(scenes)])
            gt_all = sum(len(gts[s]) for s in scenes)
            rebuilt = ap_from_pairs(stacked, gt_all)
            assert abs(rebuilt - reference[str(t)]) < 1e-6, (name, t, rebuilt)
            pooled[name][str(t)] = rebuilt
    # ---- size grouping (pre-registered terciles over all GT) ----
    volumes = np.concatenate([
        np.prod(gts[s].max(1) - gts[s].min(1), axis=1) for s in scenes])
    cuts = np.percentile(volumes, [100 / 3, 200 / 3])
    group_index = np.digitize(volumes, cuts)
    scene_offsets, cursor = {}, 0
    for scene in scenes:
        scene_offsets[scene] = (cursor, cursor + len(gts[scene]))
        cursor += len(gts[scene])
    grouping = {}
    for name in CONFIGS:
        grouping[name] = {}
        for t in THRESHOLDS:
            covered = np.zeros(len(volumes), dtype=bool)
            for scene in scenes:
                corners, _ = data[name][scene]
                if len(corners) and len(gts[scene]):
                    covered[scene_offsets[scene][0]:scene_offsets[scene][1]] = (
                        aabb_iou(corners, gts[scene]).max(0) > t)
            grouping[name][str(t)] = {
                g: {'gt': int((group_index == i).sum()),
                    'recall': float(covered[group_index == i].mean())}
                for i, g in enumerate(GROUPS)}
    grouping['volume_cuts_m3'] = [float(cuts[0]), float(cuts[1])]
    # ---- per-scene coverage deltas ----
    per_scene = {}
    for t in THRESHOLDS:
        cover = {}
        for name in CONFIGS:
            cover[name] = {}
            for scene in scenes:
                corners, _ = data[name][scene]
                cover[name][scene] = int((aabb_iou(corners, gts[scene]).max(0) > t
                                          ).sum()) if len(corners) and len(gts[scene]) else 0
        for pair in (('m1', 'native'), ('m1m2', 'native'), ('m1m2', 'm1')):
            delta = np.asarray([cover[pair[0]][s] - cover[pair[1]][s]
                                for s in scenes])
            per_scene[f'{pair[0]}-vs-{pair[1]}@{t:g}'] = {
                'win': int((delta > 0).sum()), 'tie': int((delta == 0).sum()),
                'lose': int((delta < 0).sum()),
                'mean_delta': float(delta.mean())}
    # ---- paired scene bootstrap of pooled AP deltas ----
    rng = np.random.default_rng(SEED)
    keys = ('m1-native', 'm1m2-native', 'm1m2-m1')
    boot = {k: {str(t): [] for t in THRESHOLDS} for k in keys}
    indices = np.arange(len(scenes))
    sorted_scenes = sorted(scenes)
    for _ in range(BOOTSTRAPS):
        sample = rng.choice(indices, size=len(scenes), replace=True)
        picked = [sorted_scenes[i] for i in sample]
        gt_count = sum(len(gts[s]) for s in picked)
        aps = {name: {} for name in CONFIGS}
        for name in CONFIGS:
            for t in THRESHOLDS:
                stacked = np.concatenate([pairs[name][str(t)][s] for s in picked])
                aps[name][str(t)] = ap_from_pairs(stacked, gt_count)
        for k, (a, b) in zip(keys, (('m1', 'native'), ('m1m2', 'native'),
                                    ('m1m2', 'm1'))):
            for t in THRESHOLDS:
                boot[k][str(t)].append(aps[a][str(t)] - aps[b][str(t)])
    bootstrap = {}
    for k in keys:
        bootstrap[k] = {}
        for t in THRESHOLDS:
            values = np.asarray(boot[k][str(t)])
            bootstrap[k][str(t)] = {
                'mean': float(values.mean()),
                'ci95': [float(np.percentile(values, 2.5)),
                         float(np.percentile(values, 97.5))],
                'p_positive': float((values > 0).mean())}
    for path, digest in inputs.items():
        assert sha256(path) == digest, f'Input changed: {path}'
    result = {
        'schema': 'boxfusion.paper_evidence_a.v1', 'completed': True,
        'configs': {k: str(v) for k, v in CONFIGS.items()},
        'pooled_ap': pooled, 'ledger_parity': 'all three configs within 5e-4; '
                     'pairs-rebuilt AP within 1e-6 of the anchor implementation',
        'grouping': grouping,
        'per_scene_coverage_deltas': per_scene,
        'bootstrap': bootstrap,
        'design': {'tertiles': 'GT AABB volume, cuts at 33.3/66.7 percentiles '
                   'over all 12,911 GT (pre-registered)',
                   'coverage': 'max-IoU coverage (audit convention), not '
                               'one-to-one matched recall',
                   'bootstrap': f'scene-level paired, B={BOOTSTRAPS}, seed {SEED}'},
        'limits': ['Development scenes (full107) used throughout; no '
                   'independent hold-out claim',
                   'Coverage grouping does not use one-to-one matching; '
                   'AP bootstrap does',
                   'Single dataset (CA-1M): the ScanNet ladder predictions for '
                   'native/M1 stages are not preserved as frozen dirs'],
    }
    args.output.mkdir(parents=True)
    for name, payload in (('results.json',
                           json.dumps(result, ensure_ascii=False, indent=2,
                                      allow_nan=False) + '\n'),
                          ('input_sha256.json',
                           json.dumps(inputs, ensure_ascii=False, indent=2) + '\n')):
        with (args.output / name).open('x') as handle:
            handle.write(payload)
    print(json.dumps({'pooled_ap': pooled, 'grouping_recall@0.5': {
        name: {g: round(grouping[name]['0.5'][g]['recall'], 4)
               for g in GROUPS} for name in CONFIGS},
        'per_scene': {k: v for k, v in per_scene.items() if '@0.5' in k},
        'bootstrap': bootstrap}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
