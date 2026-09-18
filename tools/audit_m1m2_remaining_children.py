"""Frozen CA-1M full107 NMS-child residual audit; CPU only, no inference.

GT is used solely for retrospective selection. Original prediction rows,
scores, and candidate geometry/scores are immutable. This is not an online
module or an upper bound on changed detection/association/fusion algorithms.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_ca1m_nms_child_headroom import (
    event_batches, load_summary, sha256, valid_boxes, verify_anchor,
)
from tools.true_fusion_audit_core import aabb_iou, class_agnostic_ap

THRESHOLDS = (.15, .25, .50)
ANCHOR = Path('/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/evaluation')
SCENES = ROOT / 'tools/boxfusion_tr3d_pipeline/evaluation/data_util/meta_data/ca1m_val_full107.txt'
GT_ROOT = Path('/tmp/ca1m_clean_root')
BASE = ROOT / 'results/ca1m_dual'
NATIVE = ROOT / 'results/ca1m_thr15_obs'
M1 = ROOT / 'results/ca1m_dual_nom2'
WD = ROOT / 'results/ca1m_wd_only'
DIAG = ROOT / 'diagnostics/ca1m_full_obs'


def read_prediction(path):
    with path.open('rb') as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)) or len(payload) != 1:
        raise ValueError(f'Invalid prediction container: {path}')
    rows = payload[0]
    if any(len(r) != 3 or int(r[0]) != 0 for r in rows):
        raise ValueError(f'Non-class-agnostic prediction: {path}')
    boxes = valid_boxes([r[1] for r in rows], str(path))
    scores = np.asarray([r[2] for r in rows], dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError(f'Invalid score: {path}')
    return boxes, scores


def partition(ciou, piou, baseline_best, threshold):
    """Use the evaluator's argmax identity, never match a second-best free GT."""
    best = ciou.argmax(1)
    quality = ciou[np.arange(len(ciou)), best]
    candidate = (quality > threshold) & (piou[np.arange(len(ciou)), best] <= threshold)
    residual = candidate & (baseline_best[best] <= threshold)
    strict = residual & ((ciou > threshold).sum(1) == 1) & ((piou > threshold).sum(1) == 1)
    return best, candidate, residual, strict


def pick(best, eligible, scores, frames, *, minimum_frames=1):
    groups = defaultdict(list)
    for i in np.flatnonzero(eligible):
        groups[int(best[i])].append(int(i))
    chosen = []
    for gt, indices in sorted(groups.items()):
        if len(set(int(frames[i]) for i in indices)) >= minimum_frames:
            # Fixed, GT-independent score ordering within a GT-assisted group.
            chosen.append(min(indices, key=lambda i: (-scores[i], i)))
    return chosen


def verify_metric():
    checked = verify_anchor(ANCHOR)
    sys.path.insert(0, str(ANCHOR / 'utils'))
    spec = importlib.util.spec_from_file_location('residual_anchor_eval', ANCHOR / 'utils/eval_det.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.tqdm = lambda values: values
    signs = np.asarray([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    cube = signs.astype(float)
    gt = {'a': np.asarray([cube, cube + [4, 0, 0]]), 'b': np.asarray([cube])}
    pred = {'a': (np.asarray([cube, cube, cube + [4.7, 0, 0], cube + [20, 0, 0]]),
                  np.asarray([.8, .8, .2, .9])),
            'b': (np.asarray([cube + [.8, 0, 0]]), np.asarray([.8]))}
    errors = []
    for t in THRESHOLDS:
        expected = module.eval_det_cls(
            {s: list(zip(*p)) for s, p in pred.items()}, gt, ovthresh=t,
            get_iou_func=module.get_iou_obb_v2)[2] * 100
        observed = class_agnostic_ap(pred, gt, t)['ap']
        errors.append(abs(expected - observed))
    assert max(errors) < 1e-10
    # Exact-threshold rejection; second-best GT cannot produce another TP;
    # repeated event timestamps must not manufacture independent source frames.
    cb, candidate, residual, strict = partition(
        np.asarray([[.25, .0], [.8, .6], [.0, .7], [.0, .8]]),
        np.asarray([[0., .8], [.0, .8], [.8, .0], [.8, .0]]),
        np.asarray([.8, .0]), .25)
    assert candidate.tolist() == [False, True, True, True]
    assert residual.tolist() == [False, False, True, True]
    assert strict.tolist() == [False, False, True, True]
    assert pick(cb, strict, np.asarray([.1, .2, .3, .4]), [0, 1, 7, 7], minimum_frames=2) == []
    assert pick(cb, strict, np.asarray([.1, .2, .3, .4]), [0, 1, 7, 8], minimum_frames=2) == [3]
    checked['AP_parity_max_error'] = max(errors)
    checked['synthetic_selection_checks'] = 'passed'
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f'Refusing to overwrite {args.output}')
    metric = verify_metric()
    scenes = SCENES.read_text().split()
    assert len(scenes) == len(set(scenes)) == 107
    expected = {s + '_boxes.pkl' for s in scenes}
    for root in (BASE, NATIVE, M1, WD):
        assert {p.name for p in root.glob('*_boxes.pkl')} == expected, root
    assert {p.name for p in DIAG.glob('*_pvq_ar_summary.json')} == {
        s + '_pvq_ar_summary.json' for s in scenes}
    assert {p.name for p in DIAG.glob('*_pvq_nms.jsonl')} <= {
        s + '_pvq_nms.jsonl' for s in scenes}
    inputs = {str(p.resolve()): sha256(p) for p in (
        Path(__file__), SCENES, ROOT / 'tools/audit_ca1m_nms_child_headroom.py',
        ROOT / 'tools/true_fusion_audit_core.py', ANCHOR / 'utils/eval_det.py',
        ANCHOR / 'utils/box_util.py', ROOT / 'config/ca1m_thr15_obs.yaml')}
    policies = {t: ('any', 'strict', 'strict_2frames', 'strict_3frames') +
                (('identity15', 'identity15_sameframe') if t > .15 else ())
                for t in THRESHOLDS}
    arms = {f'oracle{int(t*100)}_{policy}': {} for t in THRESHOLDS for policy in policies[t]}
    baseline, gts, scene_rows, selections = {}, {}, [], {a: [] for a in arms}
    totals = {str(t): Counter() for t in THRESHOLDS}
    for ordinal, scene in enumerate(scenes, 1):
        pred_path = BASE / f'{scene}_boxes.pkl'
        native_path = NATIVE / pred_path.name
        m1_path = M1 / pred_path.name
        wd_path = WD / pred_path.name
        side_path = Path(str(pred_path) + '.dual_state.json')
        summary_path = DIAG / f'{scene}_pvq_ar_summary.json'
        ledger = DIAG / f'{scene}_pvq_nms.jsonl'
        gt_path = GT_ROOT / scene / 'after_filter_boxes.npy'
        for path in (pred_path, native_path, m1_path, wd_path, side_path, summary_path, gt_path):
            inputs[str(path.resolve())] = sha256(path)
        if ledger.exists():
            inputs[str(ledger.resolve())] = sha256(ledger)
        base, native, m1, wd = map(read_prediction, (pred_path, native_path, m1_path, wd_path))
        assert np.array_equal(base[0][:len(native[0])], native[0]), scene
        assert np.array_equal(base[0], m1[0]), scene
        assert np.array_equal(wd[0][:len(native[0])], native[0]), scene
        side = json.loads(side_path.read_text())
        assert side['scene_id'] == scene and not side['m5_enabled']
        assert side['selected_score_view'] == 'persistent' and side['row_count'] == len(base[0])
        assert np.array_equal([r['persistent_score'] for r in side['rows']], base[1])
        gt = valid_boxes(np.load(gt_path, allow_pickle=False), str(gt_path))
        assert len(gt), scene
        baseline[scene], gts[scene] = base, gt
        n = load_summary(summary_path, scene)
        records = [r for batch in event_batches(ledger, scene, n) for r in batch]
        child = valid_boxes([r['child_corners_world'] for r in records], scene)
        parent = valid_boxes([r['parent_corners_world'] for r in records], scene)
        scores = np.asarray([r['child_score'] for r in records], dtype=float)
        assert np.isfinite(scores).all()
        frames = np.asarray([r['child_frame_id'] for r in records], dtype=int)
        ciou, piou = aabb_iou(child, gt), aabb_iou(parent, gt)
        cb = ciou.argmax(1)
        identity15 = ((ciou > .15).sum(1) == 1) & ((piou > .15).sum(1) == 1)
        identity15 &= piou[np.arange(len(ciou)), cb] <= .15
        sameframe = np.asarray([r['child_frame_id'] == r['parent_frame_id'] for r in records], dtype=bool)
        biou = aabb_iou(base[0], gt).max(0) if len(base[0]) else np.zeros(len(gt))
        niou = aabb_iou(native[0], gt).max(0) if len(native[0]) else np.zeros(len(gt))
        wiou = aabb_iou(wd[0], gt).max(0) if len(wd[0]) else np.zeros(len(gt))
        scene_row = {'scene': scene, 'GT': len(gt), 'baseline_rows': len(base[0]),
                     'native_rows': len(native[0]), 'nms_events': n, 'thresholds': {}}
        for t in THRESHOLDS:
            best, candidate, residual, strict = partition(ciou, piou, biou, t)
            candidate_gts = set(best[candidate].tolist())
            native_missing = {g for g in candidate_gts if niou[g] <= t}
            remaining = {g for g in candidate_gts if biou[g] <= t}
            recovered = native_missing - remaining
            row = {
                'GT': len(gt), 'baseline_any_covered': int((biou > t).sum()),
                'child_argmax_parent_does_not_cover_targets': len(candidate_gts),
                'native_missing_candidate_targets': len(native_missing),
                'already_recovered_by_dual_M1': len(recovered),
                'dual_recovered_not_covered_by_wd_only': sum(wiou[g] <= t for g in recovered),
                'remaining_targets': len(remaining),
                'remaining_events': int(residual.sum()),
                'strict_remaining_targets': len(set(best[strict].tolist())),
                'strict_remaining_events': int(strict.sum()),
                'remaining_uncovered_even_at_15': sum(biou[g] <= .15 for g in remaining),
                'remaining_already_covered_at_15': sum(biou[g] > .15 for g in remaining),
            }
            for policy in policies[t]:
                mask = residual if policy == 'any' else strict
                if policy.startswith('identity15'):
                    mask = residual & identity15
                    if policy.endswith('sameframe'):
                        mask &= sameframe
                required = 2 if policy.endswith('2frames') else 3 if policy.endswith('3frames') else 1
                selected = pick(best, mask, scores, frames, minimum_frames=required)
                arm = f'oracle{int(t*100)}_{policy}'
                arms[arm][scene] = (np.concatenate([base[0], child[selected]]),
                                    np.r_[base[1], scores[selected]])
                assert np.array_equal(arms[arm][scene][0][:len(base[0])], base[0])
                assert np.array_equal(arms[arm][scene][1][:len(base[0])], base[1])
                row[policy + '_selected'] = len(selected)
                for i in selected:
                    g = int(best[i])
                    support = np.flatnonzero(mask & (best == g))
                    selections[arm].append({
                        'scene': scene, 'gt_index': g, 'event_line': records[i]['event_line'],
                        'child_init_id': records[i]['child_init_id'], 'child_frame_id': int(frames[i]),
                        'parent_frame_id': records[i]['parent_frame_id'],
                        'keyframe_id': records[i]['keyframe_id'], 'source_score': float(scores[i]),
                        'child_iou': float(ciou[i, g]), 'parent_iou_for_target': float(piou[i, g]),
                        'baseline_best_iou': float(biou[g]),
                        'distinct_source_frames': sorted(set(frames[support].tolist())),
                        'unique_child_ids': len({(records[j]['child_init_id'], int(frames[j])) for j in support}),
                    })
            row = {key: int(value) for key, value in row.items()}
            scene_row['thresholds'][str(t)] = row
            totals[str(t)].update(row)
        scene_rows.append(scene_row)
        if ordinal % 25 == 0 or ordinal == len(scenes):
            print(f'{ordinal}/107 scenes verified', flush=True)
    metrics = {'baseline': {str(t): class_agnostic_ap(baseline, gts, t) for t in THRESHOLDS}}
    for arm, predictions in arms.items():
        metrics[arm] = {}
        primary = int(arm.split('_')[0][6:]) / 100
        for t in THRESHOLDS:
            value = class_agnostic_ap(predictions, gts, t)
            value['delta_ap'] = value['ap'] - metrics['baseline'][str(t)]['ap']
            value['delta_tp'] = value['tp'] - metrics['baseline'][str(t)]['tp']
            value['delta_fp'] = value['fp'] - metrics['baseline'][str(t)]['fp']
            if t == primary:
                assert value['delta_tp'] == len(selections[arm]), (arm, value)
                assert value['delta_fp'] == 0, (arm, value)
            metrics[arm][str(t)] = value
    # Historical full107 dual-source receipt, rounded to the recorded precision.
    assert np.allclose([metrics['baseline'][str(t)]['ap'] for t in THRESHOLDS],
                       [46.13, 38.34, 16.85], atol=.005, rtol=0)
    # Freeze input files throughout the audit, not just on first read.
    for path, digest in inputs.items():
        assert sha256(path) == digest, f'Input changed: {path}'
    result = {
        'schema': 'boxfusion.m1m2_remaining_nms_children.v1', 'completed': True,
        'dataset': 'CA1M full107', 'baseline_root': str(BASE), 'native_root': str(NATIVE),
        'wd_only_root': str(WD), 'nms_root': str(DIAG), 'gt_root': str(GT_ROOT),
        'metric_checks': metric, 'scenes': 107, 'events': sum(s['nms_events'] for s in scene_rows),
        'policies': {
            'any': 'Child argmax GT exceeds threshold, parent does not cover it, no baseline box covers it.',
            'strict': 'Additionally child and parent each overlap exactly one GT above threshold; they differ.',
            'strict_2frames': 'Strict with at least two distinct child_frame_id values, not event timestamps.',
            'strict_3frames': 'Strict with at least three distinct child_frame_id values, not event timestamps.',
            'identity15': 'Independent parent/child identity proxy fixed at unique IoU > .15; child geometry and baseline absence tested at target threshold.',
            'identity15_sameframe': 'identity15 plus equal recorded parent_frame_id and child_frame_id; not pixel-level proof.',
        },
        'selection': 'At most one original child per argmax GT, highest original child_score, event order tie break.',
        'ranking': 'Existing baseline scores frozen; appended child uses original logged score; anchor numpy argsort.',
        'limits': [
            'Retrospective GT-assisted selection on already-used development/evaluation scenes, not a deployable result.',
            'Each oracle15/25/50 selects a different output set; cross-threshold values are evaluated without reselection.',
            'Only saved NMS child snapshots, not all pre-detector-NMS proposals, are available.',
            'IoU assignments are identity proxies, not verified pixel-level instance identities or clutter labels.',
            'Frame support is a necessary opportunity count, not validated online association or birth acceptance.',
            'No rerun of M1/M2, no changed native geometry/score, no new lifting/fusion; not a global upper bound.',
            'Equal-score order follows the existing evaluator; no GT-based reranking of baseline rows.',
            'WD-only contrast is conditional coverage, not an isolated causal attribution of child recovery.',
            'No ScanNet experiment or inference/FPS experiment was run in this audit.',
        ],
        'totals': totals, 'metrics': metrics, 'scene_audit': scene_rows, 'selected_events': selections,
    }
    encoded = [(name, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
               for name, data in [('results.json', result), ('input_sha256.json', inputs)]]
    args.output.mkdir(parents=True)
    for name, data in encoded:
        with (args.output / name).open('x') as handle:
            handle.write(data + '\n')
    print(json.dumps({'baseline': metrics['baseline'], 'events': result['events'],
                      'totals': totals, 'oracle_deltas': {
                          a: [round(metrics[a][str(t)]['delta_ap'], 4) for t in THRESHOLDS]
                          for a in arms}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
