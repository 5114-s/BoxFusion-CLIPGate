"""Gate 0b replace-protocol ceiling under margin restrictions (CA-1M full107).

Follow-up to audit_ca1m_gate0_margin: convert the measured child-vs-existing
margins into AP ceilings for the deferred geometry-selection module semantics.

Protocol: REPLACE, not append. For each frozen wide-pool ideal-selection event
in the replace population (baseline_best_iou > 0.15), the geometry of the
baseline argmax-IoU row for that GT is swapped to the NMS child snapshot.
Row count, row order, and all scores stay frozen. Arms restrict the swap to
events whose GT-measured center-error margin exceeds a noise-scale threshold.

GT is used to pick targets, margins, and the argmax row; no validator is
simulated. These are optimistic ceilings, not module results.
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
    BASE, DIAG, GT_ROOT, SCENES, THRESHOLDS, partition, pick, read_prediction,
)

FROZEN = ROOT / 'reports/m1m2_remaining_children_20260908_v2/results.json'
GATE0 = ROOT / 'reports/ca1m_gate0_margin_20260911/results.json'
PRIMARY = .50
MARGIN_KEYS = ('all', 5.0, 10.0, 15.0, 20.0)


def center_size(corners):
    corners = np.asarray(corners, dtype=np.float64)
    return corners.mean(0), corners.max(0) - corners.min(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f'Refusing to overwrite {args.output}')
    frozen = json.loads(FROZEN.read_text())
    gate0 = json.loads(GATE0.read_text())
    margin_index = {(e['scene'], e['event_line'], e['gt_index']): e
                    for e in gate0['events']['oracle50_any']}
    scenes = SCENES.read_text().split()
    inputs = {str(p.resolve()): sha256(p) for p in (
        Path(__file__), SCENES, FROZEN, GATE0,
        ROOT / 'tools/audit_ca1m_nms_child_headroom.py',
        ROOT / 'tools/true_fusion_audit_core.py',
        ROOT / 'tools/audit_m1m2_remaining_children.py')}
    arms = {f'replace_{key}': {} for key in MARGIN_KEYS}
    arms.update({f'append_replace_{key}': {} for key in MARGIN_KEYS})
    gts, baseline = {}, {}
    swapped, collided = Counter(), Counter()
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
        biou = biou_full.max(0)
        existing_row = biou_full.argmax(0)
        n = load_summary(summary_path, scene)
        records = [r for batch in event_batches(ledger, scene, n) for r in batch]
        child = valid_boxes([r['child_corners_world'] for r in records], scene)
        parent = valid_boxes([r['parent_corners_world'] for r in records], scene)
        cscores = np.asarray([r['child_score'] for r in records], dtype=float)
        frames = np.asarray([r['child_frame_id'] for r in records], dtype=int)
        ciou, piou = aabb_iou(child, gt), aabb_iou(parent, gt)
        best, _c, residual, _s = partition(ciou, piou, biou, PRIMARY)
        selected = pick(best, residual, cscores, frames, minimum_frames=1)
        mine = {(scene, records[i]['event_line'], int(best[i])) for i in selected}
        theirs = {(e['scene'], e['event_line'], e['gt_index'])
                  for e in frozen['selected_events']['oracle50_any'] if e['scene'] == scene}
        assert mine == theirs, (scene, len(mine), len(theirs))
        # Group selected replace events by margin gate; oracle identity grouping.
        gated = {key: [] for key in MARGIN_KEYS}
        for i in selected:
            g = int(best[i])
            if biou[g] <= .15:
                continue  # append population: no incumbent row to swap
            key3 = (scene, records[i]['event_line'], g)
            event = margin_index[key3]
            margin = event['delta_center_err_cm']
            for key in MARGIN_KEYS:
                if key == 'all' or margin >= key:
                    gated[key].append(i)
        used_rows = {key: set() for key in MARGIN_KEYS}
        for key in MARGIN_KEYS:
            new_boxes = boxes.copy()
            for i in gated[key]:
                g = int(best[i])
                row = int(existing_row[g])
                if row in used_rows[key]:
                    collided[key] += 1
                new_boxes[row] = child[i]
                used_rows[key].add(row)
                swapped[key] += 1
            arms[f'replace_{key}'][scene] = (new_boxes, scores)
            arms[f'append_replace_{key}'][scene] = (
                np.concatenate([boxes, child[gated[key]]]),
                np.r_[scores, cscores[gated[key]]])
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
        'schema': 'boxfusion.gate0b_replace_ceiling.v1', 'completed': True,
        'protocol': 'replace: swap argmax-row geometry to the selected child; '
                    'row count, order and scores frozen; no validator simulated',
        'population': 'oracle50_any replace subset (baseline_best_iou > 0.15); '
                      'GT-measured delta_center_err_cm gates are optimistic ceilings',
        'frozen_reference': str(FROZEN), 'gate0_reference': str(GATE0),
        'swapped_rows': dict(swapped), 'row_collisions': dict(collided),
        'limits': [
            'Oracle margins and argmax-row identity use GT; grouping error is not included.',
            'A real validator decides with noise-scale errors and may select wrong candidates.',
            'replace changes the geometry of rows that may also be the best row of another GT; '
            'collateral losses are included in the AP numbers.',
            'append_replace arms reprice nothing and reuse logged child scores; they bound the '
            'same population under the frozen v2 append protocol for comparison.',
            'No ScanNet audit, no inference, no FPS measurement, no production code change.',
        ],
        'metrics': metrics,
    }
    args.output.mkdir(parents=True)
    for name, data in (('results.json', json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'),
                       ('input_sha256.json', json.dumps(inputs, ensure_ascii=False, indent=2) + '\n')):
        with (args.output / name).open('x') as handle:
            handle.write(data)
    print(json.dumps({arm: {str(t): round(metrics[arm][str(t)]['delta_ap'], 4)
                            for t in THRESHOLDS}
                      for arm in arms}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
