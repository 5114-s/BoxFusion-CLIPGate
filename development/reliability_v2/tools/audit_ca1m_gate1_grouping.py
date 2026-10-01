"""Gate 1 grouping audit for wide-pool deferred geometry selection (CA-1M full107).

Question: how much of the Gate 0b replace-protocol oracle ceiling survives when
the child-to-incumbent assignment is made by GT-free geometric rules instead of
GT argmax identity, and how consistent is that grouping with the oracle?

Protocol: every saved NMS child snapshot is assigned to exactly one baseline
output row by a fixed rule (3D AABB IoU argmax above a threshold, or nearest
center within a radius). Within each group the decision to swap the row's
geometry to its best assigned child is oracle (swap iff IoU with the row's
argmax GT improves), optionally restricted to GT-measured center-error margins.
Row count, order and scores stay frozen. GT is never used for assignment.

Pure offline CPU audit of frozen artifacts; not a module and not deployable.
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

FROZEN = ROOT / 'reports/m1m2_remaining_children_20260908_v2/results.json'
GATE0 = ROOT / 'reports/ca1m_gate0_margin_20260911/results.json'
ASSIGN_RULES = (
    ('iou10', 'iou', .10),
    ('iou15', 'iou', .15),
    ('iou25', 'iou', .25),
    ('center50', 'center', .50),
)
MARGIN_GATES_CM = (0.0, 5.0, 10.0)


def arm_name(rule, margin):
    return f'{rule}_m{margin:g}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f'Refusing to overwrite {args.output}')
    frozen = json.loads(FROZEN.read_text())
    gate0 = json.loads(GATE0.read_text())
    gate0_index = {(e['scene'], e['event_line'], e['gt_index']): e
                   for e in gate0['events']['oracle50_any'] if e['covered15']}
    assert len(gate0_index) == 989
    scenes = SCENES.read_text().split()
    inputs = {str(p.resolve()): sha256(p) for p in (
        Path(__file__), SCENES, FROZEN, GATE0,
        ROOT / 'tools/audit_ca1m_nms_child_headroom.py',
        ROOT / 'tools/true_fusion_audit_core.py',
        ROOT / 'tools/audit_m1m2_remaining_children.py')}
    arms = {arm_name(rule, margin): {} for rule, _, _ in ASSIGN_RULES
            for margin in MARGIN_GATES_CM}
    baseline, gts = {}, {}
    stats = {arm: Counter() for arm in arms}
    grouping = {rule: Counter() for rule, _, _ in ASSIGN_RULES}
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
        biou_full = aabb_iou(boxes, gt)                     # rows x GT
        existing_row = biou_full.argmax(0)                  # GT -> oracle row
        row_gt = biou_full.argmax(1)                        # row -> oracle GT
        row_err = np.linalg.norm(boxes.mean(1) - gt[row_gt].mean(1), axis=1) * 100
        n = load_summary(summary_path, scene)
        records = [r for batch in event_batches(ledger, scene, n) for r in batch]
        child = valid_boxes([r['child_corners_world'] for r in records], scene)
        child_line = {r['event_line']: j for j, r in enumerate(records)}
        ciou = aabb_iou(child, gt)                          # children x GT
        iou_rc = aabb_iou(boxes, child)                     # rows x children
        dist_rc = np.linalg.norm(boxes.mean(1)[:, None, :] - child.mean(1)[None, :, :], axis=2)
        # GT-free assignment per rule: child -> single best row above threshold.
        assign = {}
        for rule, kind, cut in ASSIGN_RULES:
            matrix = iou_rc if kind == 'iou' else -dist_rc
            best = matrix.argmax(0)
            value = (iou_rc if kind == 'iou' else dist_rc)[best, np.arange(len(child))]
            keep = (value >= cut) if kind == 'iou' else (value <= cut)
            assign[rule] = {int(j): int(best[j]) for j in np.flatnonzero(keep)}
        # Oracle decisions within GT-free groups, with margin gates.
        groups = {rule: {} for rule, _, _ in ASSIGN_RULES}
        for rule, rows in assign.items():
            for j, r in rows.items():
                groups[rule].setdefault(r, []).append(j)
        covered = {(e['scene'], e['event_line'], e['gt_index']): e
                   for e in frozen['selected_events']['oracle50_any']
                   if e['scene'] == scene and e['baseline_best_iou'] > .15}
        for rule, rows in assign.items():
            for key, event in covered.items():
                j = child_line[event['event_line']]
                r_oracle = int(existing_row[event['gt_index']])
                grouping[rule]['total'] += 1
                got = rows.get(j)
                if got is None:
                    grouping[rule]['unassigned'] += 1
                elif got == r_oracle:
                    grouping[rule]['correct_row'] += 1
                    g_event = gt[event['gt_index']]
                    margin = (np.linalg.norm(boxes[r_oracle].mean(0) - g_event.mean(0))
                              - np.linalg.norm(child[j].mean(0) - g_event.mean(0))) * 100
                    reference = gate0_index[key]['delta_center_err_cm']
                    assert abs(margin - reference) < 1e-6, (scene, key, margin, reference)
                else:
                    grouping[rule]['wrong_row'] += 1
        for rule, _, _ in ASSIGN_RULES:
            for margin_gate in MARGIN_GATES_CM:
                arm = arm_name(rule, margin_gate)
                new_boxes = boxes.copy()
                swapped_pairs = set()
                for r, js in groups[rule].items():
                    g = int(row_gt[r])
                    best = max(js, key=lambda j: (ciou[j, g], -j))
                    if ciou[best, g] <= biou_full[r, g]:
                        continue
                    margin = row_err[r] - np.linalg.norm(child[best].mean(0) - gt[g].mean(0)) * 100
                    if margin < margin_gate:
                        continue
                    new_boxes[r] = child[best]
                    swapped_pairs.add((r, best))
                    stats[arm]['swaps'] += 1
                for key, event in covered.items():
                    j = child_line[event['event_line']]
                    r_oracle = int(existing_row[event['gt_index']])
                    if (r_oracle, j) in swapped_pairs:
                        stats[arm]['exact_989_pairs'] += 1
                    elif r_oracle in {r for r, _ in swapped_pairs}:
                        stats[arm]['row_swapped_other_child'] += 1
                    else:
                        stats[arm]['opportunity_missed'] += 1
                arms[arm][scene] = (new_boxes, scores)
        baseline[scene], gts[scene] = (boxes, scores), gt
        if ordinal % 25 == 0 or ordinal == len(scenes):
            print(f'{ordinal}/107 scenes audited', flush=True)
    metrics = {'baseline': {str(t): class_agnostic_ap(baseline, gts, t) for t in THRESHOLDS}}
    assert np.allclose([metrics['baseline'][str(t)]['ap'] for t in THRESHOLDS],
                       [46.13, 38.34, 16.85], atol=.005, rtol=0)
    for arm, predictions in arms.items():
        metrics[arm] = {}
        for t in THRESHOLDS:
            value = class_agnostic_ap(predictions, gts, t)
            value['delta_ap'] = value['ap'] - metrics['baseline'][str(t)]['ap']
            value['delta_tp'] = value['tp'] - metrics['baseline'][str(t)]['tp']
            value['delta_fp'] = value['fp'] - metrics['baseline'][str(t)]['fp']
            metrics[arm][str(t)] = value
    for path, digest in inputs.items():
        assert sha256(path) == digest, f'Input changed: {path}'
    result = {
        'schema': 'boxfusion.gate1_grouping.v1', 'completed': True,
        'assignment_rules': {
            'iou10/iou15/iou25': 'child assigned to argmax 3D-AABB-IoU baseline row if IoU >= 0.10/0.15/0.25',
            'center50': 'child assigned to nearest baseline row center if distance <= 0.50 m',
        },
        'decision': 'oracle within GT-free groups: swap row geometry to best assigned child '
                    'iff IoU with the row argmax GT improves; margin gates on GT center-error improvement',
        'margin_gates_cm': list(MARGIN_GATES_CM),
        'grouping_consistency_vs_oracle_989': {k: dict(v) for k, v in grouping.items()},
        'stats': {k: dict(v) for k, v in stats.items()},
        'limits': [
            'Assignment is GT-free, but swap decisions, margins and row identity use GT: optimistic ceiling.',
            'Grouping uses final fused row geometry vs the NMS-time child snapshot; no temporal/causal constraint.',
            'A wrong-row assignment can still swap that row to a better child for its own GT; harm is included in AP.',
            'Only saved NMS child snapshots; not exhaustive pre-NMS proposals. CA-1M only; no FPS or ScanNet.',
        ],
        'metrics': metrics,
    }
    args.output.mkdir(parents=True)
    for name, data in (('results.json', json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'),
                       ('input_sha256.json', json.dumps(inputs, ensure_ascii=False, indent=2) + '\n')):
        with (args.output / name).open('x') as handle:
            handle.write(data)
    print(json.dumps({
        'grouping': {k: dict(v) for k, v in grouping.items()},
        'deltas': {arm: [round(metrics[arm][str(t)]['delta_ap'], 4) for t in THRESHOLDS]
                   for arm in arms}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
