"""Gate 0 margin audit for wide-pool deferred geometry selection (CA-1M full107).

Question: on the frozen wide-pool ideal-selection targets of
reports/m1m2_remaining_children_20260908_v2, is the geometric advantage of the
selected NMS child over the surviving baseline box large enough to be resolved
by a GT-free RGB-D consistency validator at centimetre noise scale?

Pure offline CPU audit of frozen artifacts. GT is used only to measure margins;
no prediction row, score, or candidate geometry is modified and no output set
is changed. This is not an AP result and not a module.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_ca1m_nms_child_headroom import (
    event_batches, load_summary, sha256, valid_boxes,
)
from tools.true_fusion_audit_core import aabb_iou
from tools.audit_m1m2_remaining_children import (
    BASE, DIAG, GT_ROOT, SCENES, THRESHOLDS, partition, pick, read_prediction,
)

FROZEN = ROOT / 'reports/m1m2_remaining_children_20260908_v2/results.json'
ARMS = ('oracle15_any', 'oracle25_any', 'oracle50_any')
NOISE_SCALES_CM = (5.0, 10.0, 15.0, 20.0)
POPULATIONS = ('all', 'replace', 'append')


def center_size(corners):
    corners = np.asarray(corners, dtype=np.float64)
    return corners.mean(0), corners.max(0) - corners.min(0)


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {'n': 0}
    quartiles = np.percentile(values, [10, 25, 50, 75, 90])
    return {
        'n': int(len(values)),
        'p10': float(quartiles[0]),
        'p25': float(quartiles[1]),
        'median': float(quartiles[2]),
        'p75': float(quartiles[3]),
        'p90': float(quartiles[4]),
        'mean': float(values.mean()),
    }


def summarize(rows):
    """Margin distributions and noise-scale discriminability fractions."""
    get = lambda key: [r[key] for r in rows]
    out = {f'dist_{key}': distribution(get(key)) for key in (
        'child_center_err_cm', 'existing_center_err_cm', 'delta_center_err_cm',
        'inter_candidate_cm', 'delta_size_err_cm', 'delta_iou')}
    out['frac_child_closer'] = float(np.mean(np.asarray(get('delta_center_err_cm')) > 0)) if rows else 0.0
    delta = np.asarray(get('delta_center_err_cm'), dtype=np.float64)
    inter = np.asarray(get('inter_candidate_cm'), dtype=np.float64)
    diou = np.asarray(get('delta_iou'), dtype=np.float64)
    out['noise_scale'] = {}
    for scale in NOISE_SCALES_CM:
        out['noise_scale'][f'{scale:g}cm'] = {
            'delta_center_err_ge': float(np.mean(delta >= scale)) if len(delta) else 0.0,
            'inter_candidate_ge': float(np.mean(inter >= scale)) if len(inter) else 0.0,
            'both_ge': float(np.mean((delta >= scale) & (inter >= scale))) if len(delta) else 0.0,
            'delta_ge_and_diou_ge_0.25': float(np.mean((delta >= scale) & (diou >= 0.25))) if len(delta) else 0.0,
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f'Refusing to overwrite {args.output}')
    frozen = json.loads(FROZEN.read_text())
    frozen_sel = frozen['selected_events']
    scenes = SCENES.read_text().split()
    assert len(scenes) == len(set(scenes)) == 107
    inputs = {str(p.resolve()): sha256(p) for p in (
        Path(__file__), SCENES, FROZEN,
        ROOT / 'tools/audit_ca1m_nms_child_headroom.py',
        ROOT / 'tools/true_fusion_audit_core.py',
        ROOT / 'tools/audit_m1m2_remaining_children.py')}
    events = {arm: [] for arm in ARMS}
    for ordinal, scene in enumerate(scenes, 1):
        pred_path = BASE / f'{scene}_boxes.pkl'
        gt_path = GT_ROOT / scene / 'after_filter_boxes.npy'
        summary_path = DIAG / f'{scene}_pvq_ar_summary.json'
        ledger = DIAG / f'{scene}_pvq_nms.jsonl'
        for path in (pred_path, gt_path, summary_path):
            inputs[str(path.resolve())] = sha256(path)
        if ledger.exists():
            inputs[str(ledger.resolve())] = sha256(ledger)
        boxes, _scores = read_prediction(pred_path)
        gt = valid_boxes(np.load(gt_path, allow_pickle=False), str(gt_path))
        assert len(gt), scene
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
        for t in THRESHOLDS:
            arm = f'oracle{int(t*100)}_any'
            best, _candidate, residual, _strict = partition(ciou, piou, biou, t)
            selected = pick(best, residual, cscores, frames, minimum_frames=1)
            mine = {(scene, records[i]['event_line'], int(best[i])) for i in selected}
            theirs = {(e['scene'], e['event_line'], e['gt_index'])
                      for e in frozen_sel[arm] if e['scene'] == scene}
            assert mine == theirs, (arm, scene, len(mine), len(theirs))
            for i in selected:
                g = int(best[i])
                child_center, child_size = center_size(child[i])
                existing_center, existing_size = center_size(boxes[existing_row[g]])
                gt_center, gt_size = center_size(gt[g])
                child_err = np.linalg.norm(child_center - gt_center) * 100
                existing_err = np.linalg.norm(existing_center - gt_center) * 100
                events[arm].append({
                    'scene': scene, 'gt_index': g, 'event_line': records[i]['event_line'],
                    'child_frame_id': int(frames[i]),
                    'covered15': bool(biou[g] > .15),
                    'child_iou': float(ciou[i, g]),
                    'parent_iou': float(piou[i, g]),
                    'baseline_best_iou': float(biou[g]),
                    'child_center_err_cm': float(child_err),
                    'existing_center_err_cm': float(existing_err),
                    'delta_center_err_cm': float(existing_err - child_err),
                    'inter_candidate_cm': float(np.linalg.norm(child_center - existing_center) * 100),
                    'child_axis_err_cm': [float(abs(v)) for v in (child_center - gt_center) * 100],
                    'existing_axis_err_cm': [float(abs(v)) for v in (existing_center - gt_center) * 100],
                    'child_size_err_cm': float(np.linalg.norm(child_size - gt_size) * 100),
                    'existing_size_err_cm': float(np.linalg.norm(existing_size - gt_size) * 100),
                    'delta_size_err_cm': float(np.linalg.norm(existing_size - gt_size) * 100
                                               - np.linalg.norm(child_size - gt_size) * 100),
                    'delta_iou': float(ciou[i, g] - biou[g]),
                })
        if ordinal % 25 == 0 or ordinal == len(scenes):
            print(f'{ordinal}/107 scenes audited', flush=True)
    # Population splits must reproduce the frozen audit's coverage table.
    for t in THRESHOLDS:
        arm = f'oracle{int(t*100)}_any'
        totals = frozen['totals'][str(t)]
        rows = events[arm]
        assert len(rows) == totals['remaining_targets'], (arm, len(rows))
        assert sum(r['covered15'] for r in rows) == totals['remaining_already_covered_at_15'], arm
        assert sum(not r['covered15'] for r in rows) == totals['remaining_uncovered_even_at_15'], arm
    summary = {}
    for arm in ARMS:
        rows = events[arm]
        summary[arm] = {pop: summarize([r for r in rows if pop == 'all'
                                        or (pop == 'replace') == r['covered15']])
                        for pop in POPULATIONS}
    for path, digest in inputs.items():
        assert sha256(path) == digest, f'Input changed: {path}'
    result = {
        'schema': 'boxfusion.gate0_wide_pool_margin.v1', 'completed': True,
        'question': 'Is the child-over-baseline geometric margin on wide-pool '
                    'ideal-selection targets resolvable at RGB-D noise scale?',
        'frozen_reference': str(FROZEN), 'baseline_root': str(BASE), 'gt_root': str(GT_ROOT),
        'nms_root': str(DIAG),
        'populations': {
            'replace': 'baseline_best_iou > 0.15: geometry swap of an already covered instance',
            'append': 'baseline_best_iou <= 0.15: effectively a new instance (M1b scenario)',
        },
        'noise_scale_cm': list(NOISE_SCALES_CM),
        'margin_definitions': {
            'delta_center_err_cm': 'existing center error minus child center error (positive = child closer to GT)',
            'inter_candidate_cm': 'distance between the two candidate centers; below noise they are indistinguishable',
            'delta_size_err_cm': 'existing size L2 error minus child size L2 error (cm)',
        },
        'limits': [
            'Margins are measured with GT only to characterize separability; no output set was changed.',
            'Populations come from the frozen v2 wide-pool selection; oracle15/25/50 select different targets.',
            'The surviving box is the baseline argmax-IoU row per GT, a proxy for the box an online module would replace.',
            'Axis errors are world-frame x/y/z; no camera line-of-sight decomposition is implied.',
            'Center/size margins do not by themselves predict AP; the reference AP ceilings live in the frozen v2 report.',
            'No ScanNet audit, no inference, no FPS measurement, no production code change.',
        ],
        'summary': summary, 'events': events,
    }
    args.output.mkdir(parents=True)
    for name, data in (('results.json', json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'),
                       ('input_sha256.json', json.dumps(inputs, ensure_ascii=False, indent=2) + '\n')):
        with (args.output / name).open('x') as handle:
            handle.write(data)
    digest = {arm: {pop: {key: round(summary[arm][pop]['dist_' + key]['median'], 2)
                          for key in ('delta_center_err_cm', 'inter_candidate_cm', 'delta_iou')
                          if summary[arm][pop]['dist_' + key]['n']}
                    for pop in POPULATIONS if summary[arm][pop]['dist_delta_center_err_cm']['n']}
              for arm in ARMS}
    print(json.dumps(digest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
