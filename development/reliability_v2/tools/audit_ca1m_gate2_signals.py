"""Gate 2 Phase A: pure-ledger selection signals for deferred geometry swap.

Question: can GT-free signals computed only from the saved NMS ledgers decide,
per output row, whether to swap the row geometry to one of its swallowed
children — well enough to realize a meaningful share of the Gate 1 oracle
ceiling?

Key structural fact: a swallowed child never entered its parent row's PFO
state, so the other-frame children assigned to the same row are out-of-sample
observations for both the row and for each other. Signals:

  consensus  -median distance from candidate center to same-group child
              centers from other frames (leave-one-out for children,
              all representatives for the incumbent row), in cm;
  temporal   same but validators restricted to strictly later frames
              (deferred validation; undefined when no later child exists);
  score      native child score vs the incumbent row's persistent score.

Policies swap the row to its argmax-signal child when the signal exceeds the
row's by a threshold delta. A seeded always-swap-random control bounds the
harm of unvalidated swapping; the GT oracle arm replicates Gate 1 iou10_m0.

Row count, order and scores stay frozen. GT is used only for oracle decisions,
AUC labels and margin stratification. Offline CPU audit, not a module.
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

GATE1_ORACLE = (2.0896, 3.5982, 6.0532)   # iou10_m0 deltas, for cross-check
ASSIGN_IOU = .10
DIST_DELTA_GRID = (0.0, 1.0, 2.0, 5.0, 10.0)      # cm, consensus/temporal
SCORE_DELTA_GRID = (0.0, .05, .10, .20, .30)      # score units
SIGNALS = ('consensus', 'temporal', 'score')
RNG_SEED = 20260911


def auc(diffs, labels):
    """Rank AUC of diff predicting boolean label; empty-safe."""
    diffs = np.asarray(diffs, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    pos, neg = diffs[labels], diffs[~labels]
    if not len(pos) or not len(neg):
        return None
    order = np.argsort(diffs, kind='mergesort')
    ranks = np.empty(len(diffs), dtype=np.float64)
    # Average ranks over ties.
    sorted_diffs = diffs[order]
    i = 0
    while i < len(sorted_diffs):
        j = i
        while j + 1 < len(sorted_diffs) and sorted_diffs[j + 1] == sorted_diffs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def med_or_none(values):
    return float(np.median(values)) if len(values) else None


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
    arm_names = (['oracle_m0', 'oracle_pure', 'random']
                 + [f'consensus_d{d:g}' for d in DIST_DELTA_GRID]
                 + [f'temporal_d{d:g}' for d in DIST_DELTA_GRID]
                 + [f'score_d{d:g}' for d in SCORE_DELTA_GRID])
    arms = {name: {} for name in arm_names}
    baseline, gts = {}, {}
    stats = {name: Counter() for name in arm_names}
    rng = np.random.default_rng(RNG_SEED)
    auc_pairs = {f'{kind}_{signal}': [] for kind in ('rowchild', 'childchild')
                 for signal in SIGNALS}
    strat = {f'rowchild_{signal}_{bin_}': [] for signal in SIGNALS
             for bin_ in ('m0_5', 'm5_10', 'm10p')}
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
        row_states = {}
        for r, js in groups.items():
            g = int(row_gt[r])
            # One representative per source frame: highest child score, ties to lower index.
            reps = {}
            for j in js:
                f = int(frames[j])
                if f not in reps or (cscores[j], -reps[f]) > (cscores[reps[f]], -j):
                    reps[f] = j
            rep_js = np.asarray(sorted(reps.values()), dtype=int)
            centers = child[rep_js].mean(1)
            c_row = boxes[r].mean(0)
            s_cons = {}
            s_temp = {}
            for idx, j in enumerate(rep_js):
                others = centers[np.arange(len(rep_js)) != idx]
                s_cons[int(j)] = med_or_none(np.linalg.norm(others - centers[idx], axis=1) * 100)
                later = centers[frames[rep_js] > frames[j]]
                s_temp[int(j)] = med_or_none(np.linalg.norm(later - centers[idx], axis=1) * 100) if len(later) else None
            row_cons = med_or_none(np.linalg.norm(centers - c_row, axis=1) * 100)
            row_err = float(np.linalg.norm(c_row - gt[g].mean(0)) * 100)
            child_err = np.linalg.norm(centers - gt[g].mean(0), axis=1) * 100
            row_states[r] = {
                'g': g, 'rep_js': rep_js, 'all_js': list(js),
                's_cons': s_cons, 's_temp': s_temp,
                's_row_cons': row_cons, 's_row_score': float(scores[r]),
                'child_scores': {int(j): float(cscores[j]) for j in rep_js},
            }
            row_iou = float(biou_full[r, g])
            child_ious = ciou[rep_js, g]
            # Row-vs-child AUC pairs and margin stratification.
            for idx, j in enumerate(rep_js):
                label = bool(child_ious[idx] > row_iou)
                margin = row_err - float(child_err[idx])
                key = ('consensus', s_cons[int(j)], row_cons)
                rec = {
                    'consensus': (s_cons[int(j)] - row_cons, label, margin) if s_cons[int(j)] is not None and row_cons is not None else None,
                    'temporal': (s_temp[int(j)] - row_cons, label, margin) if s_temp[int(j)] is not None and row_cons is not None else None,
                    'score': (float(cscores[j]) - float(scores[r]), label, margin),
                }
                for signal in SIGNALS:
                    if rec[signal] is not None:
                        auc_pairs[f'rowchild_{signal}'].append(rec[signal])
                        if margin >= 10:
                            strat[f'rowchild_{signal}_m10p'].append(rec[signal])
                        elif margin >= 5:
                            strat[f'rowchild_{signal}_m5_10'].append(rec[signal])
                        elif margin >= 0:
                            strat[f'rowchild_{signal}_m0_5'].append(rec[signal])
            # Child-vs-child AUC pairs.
            for a in range(len(rep_js)):
                for b in range(a + 1, len(rep_js)):
                    if child_ious[a] == child_ious[b]:
                        continue
                    label = bool(child_ious[b] > child_ious[a])
                    margin = float(child_err[a]) - float(child_err[b])
                    for signal, sj, sk in (('consensus', s_cons[int(rep_js[a])], s_cons[int(rep_js[b])]),
                                           ('temporal', s_temp[int(rep_js[a])], s_temp[int(rep_js[b])]),
                                           ('score', cscores[rep_js[a]], cscores[rep_js[b]])):
                        if sj is None or sk is None:
                            continue
                        auc_pairs[f'childchild_{signal}'].append((float(sk - sj), label, margin))
        # Build arm outputs for this scene.
        scene_swaps = {}
        for r, state in row_states.items():
            g, rep_js = state['g'], state['rep_js']
            all_js = np.asarray(state['all_js'], dtype=int)
            best_all = int(all_js[ciou[all_js, g].argmax()])
            row_err_g = float(np.linalg.norm(boxes[r].mean(0) - gt[g].mean(0)) * 100)
            best_err = float(np.linalg.norm(child[best_all].mean(0) - gt[g].mean(0)) * 100)
            if ciou[best_all, g] > biou_full[r, g]:
                scene_swaps.setdefault('oracle_pure', []).append((r, best_all))
                if row_err_g - best_err >= 0:
                    scene_swaps.setdefault('oracle_m0', []).append((r, best_all))
            if len(rep_js):
                scene_swaps.setdefault('random', []).append(
                    (r, int(rep_js[rng.integers(len(rep_js))])))
            for signal in ('consensus', 'temporal'):
                table = state['s_cons'] if signal == 'consensus' else state['s_temp']
                defined = [(table[int(j)], int(j)) for j in rep_js
                           if table[int(j)] is not None]
                if not defined:
                    continue
                best_s, best_j = max(defined, key=lambda p: (p[0], -p[1]))
                for d in DIST_DELTA_GRID:
                    if state['s_row_cons'] is not None and best_s - state['s_row_cons'] >= d:
                        scene_swaps.setdefault(f'{signal}_d{d:g}', []).append((r, best_j))
            best_s = max(float(cscores[j]) for j in rep_js)
            best_js = [int(j) for j in rep_js if cscores[j] == best_s]
            best_j = min(best_js)
            for d in SCORE_DELTA_GRID:
                if best_s - state['s_row_score'] >= d:
                    scene_swaps.setdefault(f'score_d{d:g}', []).append((r, best_j))
        for name in arm_names:
            pairs = [p for p in scene_swaps.get(name, []) if p]
            new_boxes = boxes.copy()
            for r, j in pairs:
                new_boxes[r] = child[j]
                stats[name]['swaps'] += 1
                g = int(row_gt[r])
                if ciou[j, g] > biou_full[r, g]:
                    stats[name]['oracle_positive'] += 1
                else:
                    stats[name]['oracle_negative'] += 1
            arms[name][scene] = (new_boxes, scores)
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
    oracle_deltas = [metrics['oracle_m0'][str(t)]['delta_ap'] for t in THRESHOLDS]
    print('oracle_m0 deltas:', oracle_deltas, 'expect:', GATE1_ORACLE,
          'swaps:', stats['oracle_m0']['swaps'], file=sys.stderr)
    assert np.allclose(oracle_deltas, GATE1_ORACLE, atol=5e-4, rtol=0), 'oracle arm diverges from Gate 1'
    def auc_block(pairs):
        if not pairs:
            return {'n': 0}
        diffs = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]
        return {'n': len(pairs), 'auc': auc(diffs, labels)}
    signal_summary = {}
    for signal in SIGNALS:
        signal_summary[signal] = {
            'rowchild': auc_block(auc_pairs[f'rowchild_{signal}']),
            'childchild': auc_block(auc_pairs[f'childchild_{signal}']),
            'rowchild_margin_bins': {b: auc_block(strat[f'rowchild_{signal}_{b}'])
                                     for b in ('m0_5', 'm5_10', 'm10p')},
        }
    for path, digest in inputs.items():
        assert sha256(path) == digest, f'Input changed: {path}'
    result = {
        'schema': 'boxfusion.gate2_phaseA_signals.v1', 'completed': True,
        'assignment': f'child -> argmax-IoU row, IoU >= {ASSIGN_IOU}; one representative '
                      'per source frame (highest child score) as swap candidates',
        'signals': {
            'consensus': 'child: -(median cm distance to other-frame same-group children); '
                         'row: -(median cm distance to all representatives)',
            'temporal': 'child: same with strictly later frames only; row: consensus value',
            'score': 'native child score vs incumbent row persistent score',
        },
        'policy': 'swap row geometry to argmax-signal child iff signal margin over the row >= delta',
        'controls': {'oracle_m0': 'GT argmax-IoU child with non-negative center margin; '
                                  'replicates Gate 1 iou10_m0 exactly (its margin gate at 0)',
                     'oracle_pure': 'GT argmax-IoU child, IoU improvement only — the true '
                                    'no-filter replace ceiling (Gate 1 m0 implicitly required margin >= 0)',
                     'random': f'seeded always-swap to a random child (seed {RNG_SEED})'},
        'delta_grids': {'consensus_temporal_cm': list(DIST_DELTA_GRID),
                        'score': list(SCORE_DELTA_GRID)},
        'signal_summary': signal_summary,
        'stats': {k: dict(v) for k, v in stats.items()},
        'limits': [
            'Signals read only saved NMS ledgers; no imagery, poses, depth or new forwards.',
            'GT is used for oracle decisions, AUC labels, margins and the oracle arm only.',
            'Policies choose among per-frame representatives; the oracle arm uses all children '
            '(matches Gate 1), a small candidate-set asymmetry in the oracle\'s favour.',
            'Delta grids are tuned on these 107 development scenes; no held-out validation.',
            'Row count/order/scores frozen; CA-1M only; no FPS or ScanNet.',
        ],
        'metrics': metrics,
    }
    args.output.mkdir(parents=True)
    for name, data in (('results.json', json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'),
                       ('input_sha256.json', json.dumps(inputs, ensure_ascii=False, indent=2) + '\n')):
        with (args.output / name).open('x') as handle:
            handle.write(data)
    print(json.dumps({
        'auc': {s: {'rowchild': signal_summary[s]['rowchild'].get('auc'),
                    'bins': {b: signal_summary[s]['rowchild_margin_bins'][b].get('auc')
                             for b in ('m0_5', 'm5_10', 'm10p')}}
                for s in SIGNALS},
        'deltas': {name: [round(metrics[name][str(t)]['delta_ap'], 4) for t in THRESHOLDS]
                   for name in arm_names}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
