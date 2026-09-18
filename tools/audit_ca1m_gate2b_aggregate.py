"""Gate 2b: decision-free aggregation audit (zero-selection variant).

The last decision-free variant of the wide-pool geometry route: replace a row's
geometry with a deterministic aggregate (component-wise mean / median / trimmed
mean of centers and axis-aligned sizes) over {row} U per-frame child
representatives. No oracle decision, no signal, no threshold — if any aggregate
clears the module bar, the route reopens without a third-gate solution.

Members use the same per-frame highest-score representatives as Gate 2 Phase A.
Aggregates are emitted as canonical axis-aligned corners (the evaluation
metric is world AABB IoU, so rotation is inert). Row count, order and scores
stay frozen. GT is used only for the reference oracle arm and diagnostics.

Pre-registered expectation (2026-09-11 session): all aggregates <= +0.3 AP50;
children-only mean negative (anti-selection endpoint).
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
from tools.audit_ca1m_nms_child_headroom import (
    event_batches, load_summary, sha256, valid_boxes,
)
from tools.true_fusion_audit_core import aabb_iou, class_agnostic_ap
from tools.audit_m1m2_remaining_children import (
    BASE, DIAG, GT_ROOT, SCENES, THRESHOLDS, read_prediction,
)

GATE2_ORACLE_PURE = (2.4530, 4.2247, 6.9382)
ASSIGN_IOU = .10
SIGNS = np.asarray([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                    for sz in (-1, 1)], dtype=np.float64)


def center_size(corners):
    corners = np.asarray(corners, dtype=np.float64)
    return corners.mean(0), corners.max(0) - corners.min(0)


def aggregate(members, kind):
    """members: list of (center, size); kind in mean/median/trimmed."""
    centers = np.stack([m[0] for m in members])
    sizes = np.stack([m[1] for m in members])
    if kind == 'mean':
        c, s = centers.mean(0), sizes.mean(0)
    elif kind == 'median':
        c, s = np.median(centers, 0), np.median(sizes, 0)
    elif kind == 'trimmed':
        if len(members) >= 3:
            c = np.asarray([np.sort(centers[:, i])[1:-1].mean()
                            for i in range(3)])
            s = np.asarray([np.sort(sizes[:, i])[1:-1].mean()
                            for i in range(3)])
        else:
            c, s = centers.mean(0), sizes.mean(0)
    else:
        raise ValueError(kind)
    return c, s


def canonical_corners(center, size):
    return center + SIGNS * (size / 2)


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
        ROOT / 'tools/audit_ca1m_nms_child_headroom.py',
        ROOT / 'tools/true_fusion_audit_core.py',
        ROOT / 'tools/audit_m1m2_remaining_children.py')}
    arm_names = ['agg_mean', 'agg_median', 'agg_trimmed', 'agg_mean_children',
                 'oracle_pure']
    arms = {name: {} for name in arm_names}
    baseline, gts = {}, {}
    stats = {name: Counter() for name in arm_names}
    for ordinal, scene in enumerate(scenes, 1):
        pred_path = BASE / f'{scene}_boxes.pkl'
        gt_path = GT_ROOT / scene / 'after_filter_boxes.npy'
        summary_path = DIAG / f'{scene}_pvq_ar_summary.json'
        ledger = DIAG / f'{scene}_pvq_nms.jsonl'
        for path in (pred_path, gt_path, summary_path):
            inputs[str(path.resolve())] = sha256(path)
        if ledger.exists():
            inputs[str(ledger.resolve())] = sha256(ledger)
        boxes, scores = read_prediction(pred_path)
        gt = valid_boxes(np.load(gt_path, allow_pickle=False), str(gt_path))
        biou_full = aabb_iou(boxes, gt)
        row_gt = biou_full.argmax(1)
        n = load_summary(summary_path, scene)
        records = [r for batch in event_batches(ledger, scene, n) for r in batch]
        child = valid_boxes([r['child_corners_world'] for r in records], scene)
        cscores = np.asarray([r['child_score'] for r in records], dtype=float)
        frames = np.asarray([r['child_frame_id'] for r in records], dtype=int)
        iou_rc = aabb_iou(boxes, child)
        assign = {int(j): int(iou_rc[:, j].argmax())
                  for j in np.flatnonzero(iou_rc.max(0) >= ASSIGN_IOU)}
        groups = {}
        for j, r in assign.items():
            groups.setdefault(r, []).append(j)
        ciou = aabb_iou(child, gt)
        new_boxes = {name: boxes.copy() for name in arm_names}
        for r, js in groups.items():
            g = int(row_gt[r])
            reps = {}
            for j in js:
                f = int(frames[j])
                if f not in reps or (cscores[j], -reps[f]) > (cscores[reps[f]], -j):
                    reps[f] = j
            rep_js = sorted(reps.values())
            members = [center_size(boxes[r])] + [center_size(child[j]) for j in rep_js]
            for name, kind, pool in (('agg_mean', 'mean', 'row+children'),
                                     ('agg_median', 'median', 'row+children'),
                                     ('agg_trimmed', 'trimmed', 'row+children'),
                                     ('agg_mean_children', 'mean', 'children')):
                chosen = members if pool == 'row+children' else members[1:]
                c, s = aggregate(chosen, kind)
                new_boxes[name][r] = canonical_corners(c, s)
                stats[name]['aggregated'] += 1
                new_iou = aabb_iou(new_boxes[name][r][None], gt[g:g + 1])[0, 0]
                old_iou = float(biou_full[r, g])
                stats[name]['iou_improved' if new_iou > old_iou else 'iou_worsened_or_equal'] += 1
            best_all = int(np.asarray(js)[ciou[np.asarray(js), g].argmax()])
            if ciou[best_all, g] > biou_full[r, g]:
                new_boxes['oracle_pure'][r] = child[best_all]
                stats['oracle_pure']['swaps'] += 1
        for name in arm_names:
            arms[name][scene] = (new_boxes[name], scores)
        baseline[scene], gts[scene] = (boxes, scores), gt
        if ordinal % 25 == 0 or ordinal == len(scenes):
            print(f'{ordinal}/107 scenes audited', flush=True)
    metrics = {'baseline': {str(t): class_agnostic_ap(baseline, gts, t) for t in THRESHOLDS}}
    assert np.allclose([metrics['baseline'][str(t)]['ap'] for t in THRESHOLDS],
                       [46.13, 38.34, 16.85], atol=.005, rtol=0)
    for name, predictions in arms.items():
        metrics[name] = {}
        for t in THRESHOLDS:
            value = class_agnostic_ap(predictions, gts, t)
            value['delta_ap'] = value['ap'] - metrics['baseline'][str(t)]['ap']
            value['delta_tp'] = value['tp'] - metrics['baseline'][str(t)]['tp']
            value['delta_fp'] = value['fp'] - metrics['baseline'][str(t)]['fp']
            metrics[name][str(t)] = value
    assert np.allclose([metrics['oracle_pure'][str(t)]['delta_ap'] for t in THRESHOLDS],
                       GATE2_ORACLE_PURE, atol=1e-3, rtol=0), 'oracle_pure diverges from Gate 2'
    for path, digest in inputs.items():
        assert sha256(path) == digest, f'Input changed: {path}'
    result = {
        'schema': 'boxfusion.gate2b_aggregate.v1', 'completed': True,
        'members': 'row geometry + per-frame highest-score child representatives '
                   f'(assignment IoU >= {ASSIGN_IOU}); aggregates over center and '
                   'axis-aligned size, emitted as canonical axis-aligned corners',
        'aggregates': {
            'agg_mean': 'component-wise mean over row + representatives',
            'agg_median': 'component-wise median over row + representatives',
            'agg_trimmed': 'drop one min and one max per component when >= 3 members, else mean',
            'agg_mean_children': 'mean over representatives only (anti-selection endpoint)',
            'oracle_pure': 'GT argmax-IoU child swap, replicates Gate 2 Phase A',
        },
        'pre_registered_expectation': 'all aggregates <= +0.3 AP50; children-only mean negative',
        'stats': {k: dict(v) for k, v in stats.items()},
        'limits': [
            'Aggregation is decision-free but still requires keeping per-row alternate '
            'candidates online; boundedness and FPS are not measured here.',
            'Canonical axis-aligned corners are AABB-equivalent under the world-AABB metric; '
            'any rotation information in the row geometry is discarded for aggregated rows.',
            'GT is used only for the oracle reference arm and improved/worsened diagnostics.',
            'CA-1M only; dev scenes; no held-out validation.',
        ],
        'metrics': metrics,
    }
    args.output.mkdir(parents=True)
    for name, data in (('results.json', json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'),
                       ('input_sha256.json', json.dumps(inputs, ensure_ascii=False, indent=2) + '\n')):
        with (args.output / name).open('x') as handle:
            handle.write(data)
    print(json.dumps({'deltas': {name: [round(metrics[name][str(t)]['delta_ap'], 4)
                                        for t in THRESHOLDS] for name in arm_names},
                      'stats': {k: dict(v) for k, v in stats.items()}},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
