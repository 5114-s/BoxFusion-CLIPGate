"""Gate 2d: combined-signal policies (the untested variant after the sign fix).

Motivation: the sign-corrected recompute showed consensus (rowchild AUC 0.598,
childchild 0.695) and score (0.589/0.627) are individually below the actionable
band, but dismissing their COMBINATION by argument was a methodological gap —
two weak signals from different evidence families (geometric agreement vs
detector confidence) can combine above either alone. This audit measures:

  combo_cs   z-normalized sum of consensus (negative median distance, cm) and
             native score, pooled over all candidates and rows; swap the row to
             its argmax-combo child when the combo exceeds the row's by delta;
  and_*      AND-gated precision policies: swap only when the consensus
             advantage (cm) and the score advantage both clear their gates.

Same frozen baseline, iou10 grouping, per-frame representatives, fixed
row count/order/scores, oracle_pure cross-check. Bar unchanged: >= +1 AP50
with the other two thresholds not worse. Offline CPU audit, not a module.
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
COMBO_DELTAS = (0.0, .25, .5, 1.0)
AND_GATES = (('and_c2_s005', 2.0, .05), ('and_c2_s0', 2.0, 0.0),
             ('and_c5_s0', 5.0, 0.0), ('and_c0_s005', 0.0, .05))


def auc(diffs, labels):
    diffs = np.asarray(diffs, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    pos, neg = diffs[labels], diffs[~labels]
    if not len(pos) or not len(neg):
        return None
    order = np.argsort(diffs, kind='mergesort')
    ranks = np.empty(len(diffs), dtype=np.float64)
    sd = diffs[order]
    i = 0
    while i < len(sd):
        j = i
        while j + 1 < len(sd) and sd[j + 1] == sd[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def med_or_none(values):
    return float(np.median(values)) if len(values) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f'Refusing to overwrite {args.output}')
    scenes = SCENES.read_text().split()
    inputs = {str(p.resolve()): sha256(p) for p in (
        Path(__file__), SCENES,
        ROOT / 'tools/audit_ca1m_nms_child_headroom.py',
        ROOT / 'tools/true_fusion_audit_core.py',
        ROOT / 'tools/audit_m1m2_remaining_children.py')}
    states, scene_data, gts = [], {}, {}
    pooled_cons, pooled_score = [], []
    rowchild_raw, childchild_raw = [], []
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
        scene_data[scene] = (boxes, scores, child, ciou, biou_full)
        gts[scene] = gt
        for r, js in groups.items():
            g = int(row_gt[r])
            reps = {}
            for j in js:
                f = int(frames[j])
                if f not in reps or (cscores[j], -reps[f]) > (cscores[reps[f]], -j):
                    reps[f] = j
            rep_js = sorted(reps.values())
            centers = child[rep_js].mean(1)
            c_row = boxes[r].mean(0)
            row_cons = med_or_none(-(np.linalg.norm(centers - c_row, axis=1) * 100))
            if row_cons is None:
                continue
            row_iou = float(biou_full[r, g])
            row_err = float(np.linalg.norm(c_row - gt[g].mean(0)) * 100)
            best_all = int(np.asarray(js)[ciou[np.asarray(js), g].argmax()])
            entry = {
                'scene': scene, 'r': r, 'g': g,
                'row_cons': row_cons, 'row_score': float(scores[r]),
                'row_iou': row_iou,
                'child_cons': {}, 'child_score': {}, 'child_iou': {}, 'child_err': {},
                'oracle_swap': (r, best_all) if ciou[best_all, g] > row_iou else None,
            }
            for idx, j in enumerate(rep_js):
                others = centers[np.arange(len(rep_js)) != idx]
                s_cons = med_or_none(-(np.linalg.norm(others - centers[idx], axis=1) * 100))
                if s_cons is None:
                    continue
                entry['child_cons'][int(j)] = s_cons
                entry['child_score'][int(j)] = float(cscores[j])
                entry['child_iou'][int(j)] = float(ciou[j, g])
                entry['child_err'][int(j)] = float(np.linalg.norm(
                    child[j].mean(0) - gt[g].mean(0)) * 100)
                pooled_cons.append(s_cons)
                pooled_score.append(float(cscores[j]))
                rowchild_raw.append({
                    'cons': s_cons - row_cons, 'score': float(cscores[j]) - float(scores[r]),
                    'label': bool(ciou[j, g] > row_iou), 'margin': row_err - entry['child_err'][int(j)]})
            pooled_cons.append(row_cons)
            pooled_score.append(float(scores[r]))
            for a in range(len(rep_js)):
                for b in range(a + 1, len(rep_js)):
                    ja, jb = int(rep_js[a]), int(rep_js[b])
                    if ja not in entry['child_cons'] or jb not in entry['child_cons']:
                        continue
                    if entry['child_iou'][ja] == entry['child_iou'][jb]:
                        continue
                    childchild_raw.append({
                        'cons': entry['child_cons'][jb] - entry['child_cons'][ja],
                        'score': entry['child_score'][jb] - entry['child_score'][ja],
                        'label': bool(entry['child_iou'][jb] > entry['child_iou'][ja])})
            states.append(entry)
        if ordinal % 25 == 0 or ordinal == len(scenes):
            print(f'{ordinal}/107 scenes audited', flush=True)
    cons_std = float(np.std(pooled_cons)) or 1.0
    score_std = float(np.std(pooled_score)) or 1.0

    def combo(cons_diff, score_diff):
        return cons_diff / cons_std + score_diff / score_std

    arm_names = ([f'combo_cs_d{d:g}' for d in COMBO_DELTAS]
                 + [name for name, _, _ in AND_GATES] + ['oracle_pure'])
    swaps_per_arm = {name: [] for name in arm_names}
    for entry in states:
        if entry['oracle_swap'] is not None:
            swaps_per_arm['oracle_pure'].append(
                (entry['scene'], entry['r'], entry['oracle_swap'][1]))
        candidates = list(entry['child_cons'])
        if not candidates:
            continue
        for d in COMBO_DELTAS:
            best_j = max(candidates, key=lambda j: (
                combo(entry['child_cons'][j] - entry['row_cons'],
                      entry['child_score'][j] - entry['row_score']), -j))
            if combo(entry['child_cons'][best_j] - entry['row_cons'],
                    entry['child_score'][best_j] - entry['row_score']) >= d:
                swaps_per_arm[f'combo_cs_d{d:g}'].append((entry['scene'], entry['r'], best_j))
        for name, dc, ds in AND_GATES:
            eligible = [j for j in candidates
                        if (entry['child_cons'][j] - entry['row_cons']) >= dc
                        and (entry['child_score'][j] - entry['row_score']) >= ds]
            if not eligible:
                continue
            best_j = max(eligible, key=lambda j: (entry['child_cons'][j], -j))
            swaps_per_arm[name].append((entry['scene'], entry['r'], best_j))
    arms, stats = {name: {} for name in arm_names}, {name: Counter() for name in arm_names}
    for scene, (boxes, scores, child, ciou, biou_full) in scene_data.items():
        for name in arm_names:
            new_boxes = boxes.copy()
            for sw_scene, r, j in swaps_per_arm[name]:
                if sw_scene != scene:
                    continue
                new_boxes[r] = child[j]
                stats[name]['swaps'] += 1
            arms[name][scene] = (new_boxes, scores)
    for name in arm_names[:-1]:
        swapped = {(s, r): j for s, r, j in swaps_per_arm[name]}
        for entry in states:
            key = (entry['scene'], entry['r'])
            if key not in swapped:
                continue
            j = swapped[key]
            if entry['child_iou'].get(j, -1) > entry['row_iou']:
                stats[name]['oracle_positive'] += 1
            else:
                stats[name]['oracle_negative'] += 1

    def auc_block(pairs):
        if not pairs:
            return {'n': 0}
        return {'n': len(pairs),
                'auc': auc([combo(p['cons'], p['score']) for p in pairs],
                           [p['label'] for p in pairs])}
    bins = {'m0_5': [], 'm5_10': [], 'm10p': []}
    for pair in rowchild_raw:
        if pair['margin'] >= 10:
            bins['m10p'].append(pair)
        elif pair['margin'] >= 5:
            bins['m5_10'].append(pair)
        elif pair['margin'] >= 0:
            bins['m0_5'].append(pair)
    signal_summary = {
        'combo_cs': {
            'rowchild': auc_block(rowchild_raw),
            'childchild': auc_block(childchild_raw),
            'rowchild_margin_bins': {b: auc_block(bins[b]) for b in bins}},
        'normalization': {'cons_std': cons_std, 'score_std': score_std},
    }
    baseline = {s: (scene_data[s][0], scene_data[s][1]) for s in scene_data}
    metrics = {'baseline': {str(t): class_agnostic_ap(baseline, gts, t) for t in THRESHOLDS}}
    assert np.allclose([metrics['baseline'][str(t)]['ap'] for t in THRESHOLDS],
                       [46.13, 38.34, 16.85], atol=.005, rtol=0)
    for name in arm_names:
        metrics[name] = {}
        for t in THRESHOLDS:
            value = class_agnostic_ap(arms[name], gts, t)
            value['delta_ap'] = value['ap'] - metrics['baseline'][str(t)]['ap']
            value['delta_tp'] = value['tp'] - metrics['baseline'][str(t)]['tp']
            value['delta_fp'] = value['fp'] - metrics['baseline'][str(t)]['fp']
            metrics[name][str(t)] = value
    assert np.allclose([metrics['oracle_pure'][str(t)]['delta_ap'] for t in THRESHOLDS],
                       GATE2_ORACLE_PURE, atol=1e-3, rtol=0), 'oracle_pure diverges'
    for path, digest in inputs.items():
        assert sha256(path) == digest, f'Input changed: {path}'
    result = {
        'schema': 'boxfusion.gate2d_combo.v1', 'completed': True,
        'combo': 'z-normalized (pooled) consensus-negative-distance + native score; '
                 'swap to argmax child when the combo clears delta over the row',
        'and_gates': {name: f'consensus advantage >= {dc:g} cm AND score advantage >= {ds:g}'
                      for name, dc, ds in AND_GATES},
        'bar': '>= +1 AP50 with other thresholds not worse (unchanged)',
        'signal_summary': signal_summary,
        'stats': {k: dict(v) for k, v in stats.items()},
        'limits': [
            'z-normalization pooled over all candidates and rows of the dev set; '
            'combo weights fixed at 1:1, no GT-tuned weighting beyond arm selection',
            'Temporal signal excluded (overlaps consensus; undefined for many rows); '
            'depth signal not combined (per-pair values not persisted in Gate 2c)',
            'GT used for oracle arm, AUC labels/margins only; dev scenes; CA-1M only.',
        ],
        'metrics': metrics,
    }
    args.output.mkdir(parents=True)
    for name, data in (('results.json', json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'),
                       ('input_sha256.json', json.dumps(inputs, ensure_ascii=False, indent=2) + '\n')):
        with (args.output / name).open('x') as handle:
            handle.write(data)
    print(json.dumps({
        'auc': {'rowchild': signal_summary['combo_cs']['rowchild'].get('auc'),
                'childchild': signal_summary['combo_cs']['childchild'].get('auc'),
                'bins': {b: signal_summary['combo_cs']['rowchild_margin_bins'][b].get('auc')
                         for b in ('m0_5', 'm5_10', 'm10p')}},
        'stats': {k: dict(v) for k, v in stats.items()},
        'deltas': {name: [round(metrics[name][str(t)]['delta_ap'], 4)
                          for t in THRESHOLDS] for name in arm_names}},
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
