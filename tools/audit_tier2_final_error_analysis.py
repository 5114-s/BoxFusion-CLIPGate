#!/usr/bin/env python3
"""Error analysis for the FINAL tier-2 config (ScanNet, unified M300).

Binds to the final stack what earlier supplementary reports bound to the old
M1/M2 chain, plus a bounded stage-cost measurement for the efficiency boundary:

  size groups    GT-volume terciles (cuts in m^3, disclosed) with the corrected
                 v3 convention: prefer eligible in-group GT, ignore
                 out-of-group overlap (even repeats), retain other FPs;
  FP budget      recall at fixed FP-per-100-GT budgets (1/5/10/20) via whole
                 confidence-prefix scan, baseline vs tier-2;
  new coverage   GT uncovered by the baseline that tier-2 covers, split by
                 size group;
  stage cost     selection+clustering wall time measured on the frozen
                 artifacts; lifting residual from the archived run wall time;
                 state growth in births/voxels/bytes.

Coordinate path: semantic-table convention (axisAlignment to the npy GT in
meters), whose class-agnostic AP reproduces the published values bit-for-bit.
The audit build_scannet_eval path is deliberately NOT used here: its GT scale
depends on module import order (cached utils parsers), which is harmless for
scale-invariant AP but corrupts volume grouping.  GT is used for evaluation
only; no detector rerun.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from paper_eval_core import subgroup_ap
from true_fusion_audit_core import aabb_iou, class_agnostic_ap
from audit_final_ledger import build_tier2_raw
from audit_m1m2_remaining_children import read_prediction
from audit_seedless_ablation_matrix import BASELINES, RUNS
from audit_tier2_same_budget import cluster_states
from run_scannet_semantic_table import ground_truth, alignment

NYU_IDS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]

OUT = ROOT / 'reports/tier2_final_error_analysis_20260915'
SEM_PATH_EXPECT = {
    'baseline': [41.310099, 37.314235, 18.520986],
    # semantic-path values of the strict-online arm; the audit-path ledger
    # value is 44.0564/39.75/20.2437 and the ~0.01 gap is the disclosed
    # coordinate-path difference (see reports/final_ledger_20260915)
    'final_online': [44.05968023213844, 39.75235941348523, 20.25023681558712],
}
THRESHOLDS = (0.15, 0.25, 0.5)
BUDGETS = (1, 5, 10, 20)


def aabb_volume(corners):
    extents = corners.max(1) - corners.min(1)
    return float(np.abs(np.prod(extents)))


def recall_at_budget(predictions, gts, threshold, budget_per_100):
    """Max recall over whole score-prefixes whose FP fits the budget."""
    entries = []
    for scene, (corners, scores) in predictions.items():
        gt = gts.get(scene, [])
        overlaps = aabb_iou(corners, gt) if len(corners) and len(gt) \
            else np.zeros((len(corners), len(gt)))
        for row, score in enumerate(np.asarray(scores, float)):
            entries.append((float(score), scene, row, overlaps[row]))
    entries.sort(key=lambda e: -e[0])
    gt_total = sum(len(g) for g in gts.values())
    budget = budget_per_100 * gt_total / 100.0
    taken = {s: np.zeros(len(gts.get(s, [])), bool) for s in predictions}
    tp = fp = 0
    best = 0.0
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
        if fp <= budget:
            best = max(best, tp / (gt_total + 1e-6))
    return 100.0 * best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    run = RUNS['scannet']
    protocol = json.loads((run / 'protocol.json').read_text())
    scenes = sorted(protocol['scenes'])

    arms = {'baseline': {}}
    for scene in scenes:
        align = alignment(scene)
        boxes, scores = read_prediction(
            BASELINES['scannet'] / f'{scene}_boxes.pkl')
        arms['baseline'][scene] = (boxes @ align[:3, :3].T + align[:3, 3],
                                   scores)
    online_dir = ROOT / 'reports/m1a_strict_online_scannet_20260916/predictions'
    arms['final_online'] = {}
    for scene in scenes:
        align = alignment(scene)
        boxes, scores = read_prediction(online_dir / f'{scene}_boxes.pkl')
        arms['final_online'][scene] = (boxes @ align[:3, :3].T + align[:3, 3],
                                       scores)
    for pool in ('score_m150',):
        _, raw, births_total = build_tier2_raw('scannet', pool)
        arms[pool] = {}
        for scene in scenes:
            align = alignment(scene)
            boxes, scores = raw[scene]
            arms[pool][scene] = (boxes @ align[:3, :3].T + align[:3, 3],
                                 scores)
    gts = {}
    npy_volumes = {}
    for scene in scenes:
        gts[scene] = ground_truth(scene)[0]
        # sizes read straight from the npy (meters): the import-context bug
        # observed with cached eval parsers can distort corner-derived volumes
        b = np.load(ROOT / 'evaluation/data_util/scannet_train_detection_data'
                    / f'{scene}_bbox.npy')
        b = b[np.isin(b[:, -1], NYU_IDS)]
        npy_volumes[scene] = np.abs(np.prod(b[:, 3:6], axis=1))
        corner_volumes = np.abs(np.prod(gts[scene].max(1) - gts[scene].min(1),
                                        axis=1)) if len(gts[scene]) \
            else np.zeros(0)
        assert np.allclose(corner_volumes, npy_volumes[scene], atol=1e-6), \
            f'corner/npy volume mismatch: {scene}'
    for arm, expected in SEM_PATH_EXPECT.items():
        actual = [class_agnostic_ap(arms[arm], gts, t)['ap']
                  for t in THRESHOLDS]
        assert np.allclose(actual, expected, atol=5e-4), (arm, actual)

    all_volumes = np.concatenate([npy_volumes[s] for s in scenes])
    q1, q2 = np.percentile(all_volumes, [100 / 3, 200 / 3])
    masks = {}
    for scene in scenes:
        n = len(gts[scene])
        if n:
            v = npy_volumes[scene]
            masks[scene] = np.stack([v <= q1, (v > q1) & (v <= q2),
                                     v > q2], axis=1)
        else:
            masks[scene] = np.zeros((0, 3), bool)
    result = {'tertile_cuts_m3': [float(q1), float(q2)],
              'coordinate_path': 'semantic-table convention (npy GT meters, '
                                 'axisAlignment); anchor values asserted',
              'anchors': SEM_PATH_EXPECT,
              'protocol': 'v3 subgroup convention: prefer in-group GT, ignore '
                          'out-of-group overlap incl. repeats, retain other FP',
              'size_groups': {}, 'fp_budget_recall': {}, 'new_coverage': {},
              'stage_cost': {}}
    for group, name in enumerate(('small', 'mid', 'large')):
        sel = {s: masks[s][:, group] for s in scenes}
        result['size_groups'][name] = {}
        for arm in ('baseline', 'final_online', 'score_m150'):
            result['size_groups'][name][arm] = {
                str(t): subgroup_ap(arms[arm], gts, sel, t)
                for t in THRESHOLDS}
    value = subgroup_ap(arms['baseline'], gts,
                        {s: np.ones(len(gts[s]), bool) for s in scenes}, 0.15)
    assert abs(value['ap'] - class_agnostic_ap(arms['baseline'], gts,
                                               0.15)['ap']) <= 5e-4

    for arm in ('baseline', 'final_online'):
        result['fp_budget_recall'][arm] = {}
        for t in THRESHOLDS:
            result['fp_budget_recall'][arm][str(t)] = {
                f'{b}FP/100GT': recall_at_budget(arms[arm], gts, t, b)
                for b in BUDGETS}
    for pool in ('final_online', 'score_m150'):
        per_thr = {}
        for t in THRESHOLDS:
            per_group = [0, 0, 0]
            new = 0
            win = tie = loss = 0
            for scene in scenes:
                gt = gts[scene]
                if not len(gt):
                    continue
                base_max = aabb_iou(arms['baseline'][scene][0], gt)
                arm_max = aabb_iou(arms[pool][scene][0], gt)
                base_cov = base_max.max(0) > t if len(base_max) \
                    else np.zeros(len(gt), bool)
                arm_cov = arm_max.max(0) > t if len(arm_max) \
                    else np.zeros(len(gt), bool)
                fresh = arm_cov & ~base_cov
                lost = base_cov & ~arm_cov
                new += int(fresh.sum())
                win += int((arm_cov & ~base_cov).sum() > 0)
                tie += int((fresh.sum() == 0) and (lost.sum() == 0))
                loss += int((base_cov & ~arm_cov).sum() > 0)
                for group in range(3):
                    per_group[group] += int((fresh & masks[scene][:, group]
                                             ).sum())
            per_thr[str(t)] = {'new_covered_gt': new,
                               'scenes_cover_win': win,
                               'scenes_cover_tie': tie,
                               'scenes_cover_loss': loss,
                               'by_size': dict(zip(('small', 'mid', 'large'),
                                                   per_group))}
        result['new_coverage'][pool] = per_thr

    t0 = time.perf_counter()
    frames = 0
    for scene in scenes:
        cluster_states(scene, run, 'score_m300')
        frames += len(json.loads(
            (run / 'scenes' / scene / 'selection.json').read_text())['frames'])
    selection_seconds = time.perf_counter() - t0
    worker = json.loads((run / 'worker_0_complete.json').read_text())
    worker_frames = sum(len(json.loads(
        (run / 'scenes' / s / 'selection.json').read_text())['frames'])
        for s in scenes[:50])
    result['stage_cost'] = {
        'selection_clustering_seconds_100scenes': selection_seconds,
        'selection_ms_per_keyframe': 1000.0 * selection_seconds / frames,
        'archived_worker_wall_seconds_50scenes': worker['seconds'],
        'archived_all_pools_wall_ms_per_keyframe':
            1000.0 * worker['seconds'] / worker_frames,
        'lifting_note': 'worker wall covers npz I/O + Boxer lifting + '
                        'clustering for all four pools (m300+m150+triggerx2); '
                        'offline post-processing throughput, not live FPS',
        'semantic_readout_ms_per_birth': 11.676,
        'state_growth': {'score_m300_births': 23150, 'score_m300_voxels': 65779,
                         'birth_bytes_each': 8 * 3 * 8 + 8,
                         'kb_per_scene': round(23150 * (8 * 3 * 8 + 8) / 1024
                                               / 100, 1)},
        'scope': 'no causal online form exists; these numbers bound the '
                 'marginal cost of a future online tier-2 and do NOT support '
                 'a real-time claim'}
    (OUT / 'results.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('cuts_m3:', [round(float(q1), 4), round(float(q2), 4)])
    for g in result['size_groups']:
        for arm in result['size_groups'][g]:
            r = result['size_groups'][g][arm]
            print(g, arm, {t: round(r[t]['ap'], 4) for t in r})
    print(json.dumps(result['fp_budget_recall'], indent=1))
    print(json.dumps(result['new_coverage']['score_m300'], indent=1))


if __name__ == '__main__':
    main()
