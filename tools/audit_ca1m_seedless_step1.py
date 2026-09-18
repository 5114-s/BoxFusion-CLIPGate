"""Seedless pre-NMS quantification, Step 1 (3-scene oracle on frozen captures).

Question: for CA-1M GTs missed by the frozen dual-source baseline (max output
IoU <= 0.15), does the UNTRUNCATED pre-NMS dense anchor pool contain liftable
geometry — i.e., could a seedless accumulate-and-confirm mechanism (M1-style
>=3-frame funnel fed by pre-NMS candidates) recover them?

Protocol per scene (pilot captures reports/ca1m_prenms_query_pilot_20260908):
for every saved keyframe pool, project each missed GT into the RGB frame,
select top-3 anchors by 2D IoU (NMS 0.7, 2D IoU > 0.2), batch-lift with the
frozen Boxer adapter (per-frame encoder cache), and score 3D AABB IoU against
the GT. Coverage = distinct frames with best lifted IoU >= 0.15/0.25/0.50.
Oracle-append arms add the best lifted box per qualifying GT (fixed low score)
to the untouched baseline and evaluate the anchor class-agnostic AP.

GT is used for target selection, anchor selection and reporting only; it is an
explicitly GT-assisted ceiling, not a module. Stop gate (pre-registered):
3-scene oracle-append AP15 potential < +0.5 closes the seedless route.
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
from tools.audit_m1m2_remaining_children import (
    BASE, GT_ROOT, THRESHOLDS, read_prediction,
)
from tools.ca1m_prenms_query_core import (
    box_iou2d, box_iou3d, project_box, _nms_ids,
)
from tools.validate_ca1m_prenms_query import build_lifter, lift, load_frame

PILOT = ROOT / 'reports/ca1m_prenms_query_pilot_20260908'
DATA_ROOT = Path('/extra/ZhaoX/boxfusion_ca1m')
SCENES = ('42446540', '42897501', '42897521')
TOP_K_2D = 3
MIN_IOU_2D = 0.2
IOU3D_LEVELS = (.15, .25, .50)
APPEND_SCORES = (0.05, 0.10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f'Refusing to overwrite {args.output}')
    args.output.mkdir(parents=True)
    protocol = json.loads((PILOT / 'protocol.json').read_text())
    assert tuple(protocol['scenes']) == SCENES and protocol['gap'] == 20
    inputs = {str(p.resolve()): sha256(p) for p in (
        Path(__file__), PILOT / 'protocol.json',
        ROOT / 'tools/audit_ca1m_nms_child_headroom.py',
        ROOT / 'tools/true_fusion_audit_core.py',
        ROOT / 'tools/audit_m1m2_remaining_children.py',
        ROOT / 'tools/ca1m_prenms_query_core.py',
        ROOT / 'tools/validate_ca1m_prenms_query.py')}
    adapter = build_lifter(args.output)
    baseline, gts, arms = {}, {}, {}
    arm_names = ([f'append3f_s{s:g}' for s in APPEND_SCORES]
                 + [f'append2f_s{s:g}' for s in APPEND_SCORES])
    coverage_rows = []
    totals = Counter()
    for scene in SCENES:
        pred_path = BASE / f'{scene}_boxes.pkl'
        gt_path = GT_ROOT / scene / 'after_filter_boxes.npy'
        inputs[str(pred_path.resolve())] = sha256(pred_path)
        inputs[str(gt_path.resolve())] = sha256(gt_path)
        boxes, scores = read_prediction(pred_path)
        gt = valid_boxes(np.load(gt_path, allow_pickle=False), str(gt_path))
        poses = np.load(DATA_ROOT / scene / 'all_poses.npy', allow_pickle=False)
        k_rgb = np.loadtxt(DATA_ROOT / scene / 'K_rgb.txt').reshape(3, 3)
        k_depth = np.loadtxt(DATA_ROOT / scene / 'K_depth.txt').reshape(3, 3)
        biou = aabb_iou(boxes, gt).max(0) if len(boxes) else np.zeros(len(gt))
        missed = np.flatnonzero(biou <= .15)
        # Frame support per missed GT at each 3D IoU level + best lifted box.
        support = {int(g): {t: set() for t in IOU3D_LEVELS} for g in missed}
        best_box = {int(g): (0.0, None) for g in missed}
        pools = sorted((PILOT / scene).glob('raw_*.npz'))
        assert pools, scene
        for pool_path in pools:
            frame = int(pool_path.stem.split('_')[1])
            inputs[str(pool_path.resolve())] = sha256(pool_path)
            raw = np.load(pool_path, allow_pickle=False)
            rgb_pil, rgb, depth = load_frame(DATA_ROOT / scene, frame)
            height, width = rgb.shape[:2]
            pose = poses[frame]
            selected, owner = [], []
            for g in missed:
                proj = project_box(gt[g], pose, k_rgb, width, height)
                if proj is None:
                    continue
                ious2d = box_iou2d(raw['boxes'], proj[None])[:, 0]
                eligible = np.flatnonzero(ious2d > MIN_IOU_2D)
                if not len(eligible):
                    continue
                top = _nms_ids(raw['boxes'], eligible, ious2d[eligible], TOP_K_2D)
                for anchor in top:
                    selected.append(int(anchor))
                    owner.append(int(g))
            if not selected:
                continue
            unique, inverse = np.unique(selected, return_inverse=True)
            corners, _, _ = lift(adapter, scene, frame, rgb, depth, k_rgb, k_depth,
                                 pose, raw['boxes'][unique])
            for ui, g in zip(inverse, owner):
                iou3 = float(box_iou3d(corners[ui][None], gt[g][None])[0, 0])
                for t in IOU3D_LEVELS:
                    if iou3 > t:
                        support[g][t].add(frame)
                if iou3 > best_box[g][0]:
                    best_box[g] = (iou3, corners[ui])
            print(f'{scene} frame {frame}: anchors={len(selected)} '
                  f'unique={len(unique)}', flush=True)
        for g in missed:
            row = {'scene': scene, 'gt': int(g), 'baseline_iou': float(biou[g])}
            for t in IOU3D_LEVELS:
                row[f'frames_{t:g}'] = len(support[g][t])
            coverage_rows.append(row)
            for t in IOU3D_LEVELS:
                totals[f'ge1_{t:g}'] += int(len(support[g][t]) >= 1)
                totals[f'ge2_{t:g}'] += int(len(support[g][t]) >= 2)
                totals[f'ge3_{t:g}'] += int(len(support[g][t]) >= 3)
        baseline[scene], gts[scene] = (boxes, scores), gt
        for name in arm_names:
            frames_needed = 3 if name.startswith('append3f') else 2
            score = float(name.split('_s')[1])
            extra, extra_scores = [], []
            for g in missed:
                if len(support[g][.15]) >= frames_needed and best_box[g][1] is not None:
                    extra.append(best_box[g][1])
                    extra_scores.append(score)
            arms.setdefault(name, {})[scene] = (
                np.concatenate([boxes, np.asarray(extra)]) if extra else boxes,
                np.r_[scores, extra_scores] if extra else scores)
        print(f'{scene}: missed={len(missed)} '
              f'ge3@0.15={sum(len(support[int(g)][.15]) >= 3 for g in missed)}',
              flush=True)
    assert len(coverage_rows) == sum(
        1 for scene in SCENES
        for _ in np.flatnonzero(aabb_iou(baseline[scene][0], gts[scene]).max(0) <= .15))
    metrics = {'baseline': {str(t): class_agnostic_ap(baseline, gts, t) for t in THRESHOLDS}}
    for name in arm_names:
        metrics[name] = {}
        for t in THRESHOLDS:
            value = class_agnostic_ap(arms[name], gts, t)
            value['delta_ap'] = value['ap'] - metrics['baseline'][str(t)]['ap']
            value['delta_tp'] = value['tp'] - metrics['baseline'][str(t)]['tp']
            value['delta_fp'] = value['fp'] - metrics['baseline'][str(t)]['fp']
            metrics[name][str(t)] = value
    for path, digest in inputs.items():
        assert sha256(path) == digest, f'Input changed: {path}'
    result = {
        'schema': 'boxfusion.seedless_prenms_step1.v1', 'completed': True,
        'scenes': list(SCENES), 'missed_total': len(coverage_rows),
        'selection': f'projected-GT top-{TOP_K_2D} anchors by 2D IoU '
                     f'(> {MIN_IOU_2D}, NMS 0.7), frozen Boxer lift with per-frame cache',
        'coverage_totals': dict(totals),
        'coverage_rows': coverage_rows,
        'arms': {name: {'score': float(name.split('_s')[1]),
                        'min_frames@0.15': 3 if name.startswith('append3f') else 2}
                 for name in arm_names},
        'stop_gate': '3-scene oracle-append AP15 potential < +0.5 closes the route',
        'limits': [
            'GT-assisted ceiling: target and anchor selection read GT; no GT-free '
            'trigger, memory bound, FPS or full107 validation is claimed',
            'Coverage counts distinct keyframes (gap-20) with best lifted IoU > level; '
            'frames are not de-duplicated by viewpoint',
            'Append arms use fixed scores; real funnel pricing (M1d size buckets) is '
            'not replicated; 3-scene AP is a gate estimate, not a full107 result',
            'Baseline is the frozen dual-source output; no production file changed.',
        ],
        'metrics': metrics,
    }
    for name, data in (('results.json', json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'),
                       ('input_sha256.json', json.dumps(inputs, ensure_ascii=False, indent=2) + '\n')):
        with (args.output / name).open('x') as handle:
            handle.write(data)
    print(json.dumps({
        'missed': len(coverage_rows),
        'coverage': dict(totals),
        'deltas': {name: [round(metrics[name][str(t)]['delta_ap'], 4)
                          for t in THRESHOLDS] for name in arm_names}},
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
