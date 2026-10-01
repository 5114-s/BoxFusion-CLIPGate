"""Focal canonicalization test for the frozen Boxer lifter (training-free).

Hypothesis (in spirit of 3D-MOOD's canonical image space): the Boxer adapter
already reproduces the training-time square-resize pipeline, so the remaining
CA-1M domain gap is the normalized focal. Boxer's ScanNet training optics
present per-axis normalized focals of about (0.903, 1.209); CA-1M's ARKit
optics present (0.824, 1.099) — ~10% wider FOV on both axes. If Boxer's
learned size/depth priors are calibrated to the training FOV, the mismatch
biases metric estimates by the same ratio (~20 cm at 2 m), matching the
diagnosed B5b failure mode (35% of misses: good 2D, poor lift).

Per frame the SAME frozen top-M anchor selection is re-lifted under a digital
zoom (centre crop by per-axis factor, rescaled to full size; boxes, RGB-K and
depth-K remapped identically; pose untouched — the zoomed camera is presented
consistently):
  control   untouched inputs, must reproduce the frozen pilot corners;
  c95 / canon / c88   zoom 0.95 / per-scene canonical (~0.912) / 0.88.

Reports paired per-anchor 3D IoU vs GT on anchors valid in all variants, each
variant's >=3-frame coverage and oracle-append AP. GT only for evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_ca1m_nms_child_headroom import sha256, valid_boxes
from tools.true_fusion_audit_core import aabb_iou, class_agnostic_ap
from tools.audit_m1m2_remaining_children import (
    BASE, GT_ROOT, THRESHOLDS, read_prediction,
)
from tools.ca1m_prenms_query_core import box_iou3d
from tools.validate_ca1m_prenms_query import build_lifter, lift, load_frame

RUN_ROOT = ROOT / 'reports/ca1m_seedless_full107_20260911'
PILOT = RUN_ROOT / 'scenes'
PILOT_RAW = RUN_ROOT / 'raw'
DATA_ROOT = Path('/extra/ZhaoX/boxfusion_ca1m')
SCENES = ('42446540', '42897501', '42897521')
TRAIN_FX_N, TRAIN_FY_N = 0.903, 1.209
VARIANTS = ('control', 'repeat', 'letter', 'c95', 'canon', 'c88')
ZOOM = {'c95': 0.95, 'c88': 0.88}
APPEND_SCORE = 0.10


def crop_factor(kind, k_rgb, width, height):
    if kind in ('control', 'repeat', 'letter'):
        return 1.0, 1.0
    if kind == 'canon':
        return ((k_rgb[0, 0] / width) / TRAIN_FX_N,
                (k_rgb[1, 1] / height) / TRAIN_FY_N)
    return ZOOM[kind], ZOOM[kind]


def apply_zoom(rgb, depth, k_rgb, k_depth, boxes, cx, cy):
    """Centre-crop the same FRACTION of each grid, resize back, remap K per grid."""
    h, w = depth.shape
    rh, rw = rgb.shape[:2]
    fx0, fy0 = (1.0 - cx) / 2.0, (1.0 - cy) / 2.0
    dx0, dy0 = int(round(fx0 * w)), int(round(fy0 * h))
    dw, dh = w - 2 * dx0, h - 2 * dy0
    rx0, ry0 = int(round(fx0 * rw)), int(round(fy0 * rh))
    rw_c, rh_c = rw - 2 * rx0, rh - 2 * ry0
    rgb_z = cv2.resize(rgb[ry0:ry0 + rh_c, rx0:rx0 + rw_c], (rw, rh),
                       interpolation=cv2.INTER_LINEAR)
    depth_z = cv2.resize(depth[dy0:dy0 + dh, dx0:dx0 + dw], (w, h),
                         interpolation=cv2.INTER_NEAREST)
    k_rgb_z = k_rgb.copy()
    k_rgb_z[0, 0] = k_rgb[0, 0] * (rw / rw_c)
    k_rgb_z[0, 2] = (k_rgb[0, 2] - rx0) * (rw / rw_c)
    k_rgb_z[1, 1] = k_rgb[1, 1] * (rh / rh_c)
    k_rgb_z[1, 2] = (k_rgb[1, 2] - ry0) * (rh / rh_c)
    k_depth_z = k_depth.copy()
    k_depth_z[0, 0] = k_depth[0, 0] * (w / dw)
    k_depth_z[0, 2] = (k_depth[0, 2] - dx0) * (w / dw)
    k_depth_z[1, 1] = k_depth[1, 1] * (h / dh)
    k_depth_z[1, 2] = (k_depth[1, 2] - dy0) * (h / dh)
    sx, sy = rw / rw_c, rh / rh_c
    remapped = np.stack((
        (boxes[:, 0] - rx0) * sx, (boxes[:, 1] - ry0) * sy,
        (boxes[:, 2] - rx0) * sx, (boxes[:, 3] - ry0) * sy), axis=1)
    inside = ((remapped[:, 0] >= 0) & (remapped[:, 1] >= 0)
              & (remapped[:, 2] <= rw) & (remapped[:, 3] <= rh)
              & (remapped[:, 2] > remapped[:, 0]) & (remapped[:, 3] > remapped[:, 1]))
    return rgb_z, depth_z, k_rgb_z, k_depth_z, remapped, inside


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f'Refusing to overwrite {args.output}')
    args.output.mkdir(parents=True)
    adapter = build_lifter(args.output)
    inputs = {str(p.resolve()): sha256(p) for p in (
        Path(__file__), RUN_ROOT / 'protocol.json',
        ROOT / 'tools/validate_ca1m_prenms_query.py',
        ROOT / 'tools/ca1m_prenms_query_core.py',
        ROOT / 'tools/true_fusion_audit_core.py',
        ROOT / 'tools/audit_m1m2_remaining_children.py')}
    baseline, gts, arms = {}, {}, {v: {} for v in VARIANTS}
    support = {v: {} for v in VARIANTS}
    best = {v: {} for v in VARIANTS}
    paired = {}
    control_parity = []
    for scene in SCENES:
        pred_path = BASE / f'{scene}_boxes.pkl'
        gt_path = GT_ROOT / scene / 'after_filter_boxes.npy'
        inputs[str(pred_path.resolve())] = sha256(pred_path)
        inputs[str(gt_path.resolve())] = sha256(gt_path)
        base_boxes, base_scores = read_prediction(pred_path)
        gt = valid_boxes(np.load(gt_path, allow_pickle=False), str(gt_path))
        baseline[scene], gts[scene] = (base_boxes, base_scores), gt
        selection = json.loads((PILOT / scene / 'selection.json').read_text())
        poses = np.load(DATA_ROOT / scene / 'all_poses.npy', allow_pickle=False)
        k_rgb = np.loadtxt(DATA_ROOT / scene / 'K_rgb.txt').reshape(3, 3)
        k_depth = np.loadtxt(DATA_ROOT / scene / 'K_depth.txt').reshape(3, 3)
        for row in selection['frames']:
            frame = row['frame']
            lift_path = PILOT / scene / f'lifted_{frame:06d}.npz'
            raw_path = PILOT_RAW / scene / f'raw_{frame:06d}.npz'
            for path in (lift_path, raw_path):
                inputs[str(path.resolve())] = sha256(path)
            with np.load(lift_path, allow_pickle=False) as data:
                frozen_ids, frozen_corners = data['anchor_ids'], data['corners']
            with np.load(raw_path, allow_pickle=False) as data:
                raw_boxes, raw_scores = data['boxes'], data['scores']
            ids = np.asarray(row['selected_anchor_ids']['score_m300'], dtype=np.int64)
            positions = np.searchsorted(frozen_ids, ids)
            assert np.array_equal(frozen_ids[positions], ids)
            rgb_pil, rgb, depth = load_frame(DATA_ROOT / scene, frame)
            pose = poses[frame]
            per_variant = {}
            for variant in VARIANTS:
                cx, cy = crop_factor(variant, k_rgb, rgb.shape[1], rgb.shape[0])
                if variant == 'repeat':
                    corners, _, _ = lift(adapter, scene, frame, rgb, depth,
                                         k_rgb, k_depth, pose, raw_boxes[ids])
                    keep_ids, keep_corners = ids, np.asarray(corners)
                elif variant == 'letter':
                    rh, rw = rgb.shape[:2]
                    pad = (rw - rh) // 2
                    rgb_l = np.pad(rgb, ((pad, rw - rh - pad), (0, 0), (0, 0)),
                                   mode='edge')
                    hq, wq = depth.shape
                    dpad = (wq - hq) // 2
                    depth_l = np.pad(depth, ((dpad, wq - hq - dpad), (0, 0)),
                                     mode='edge')
                    k_l = k_rgb.copy()
                    k_l[1, 2] += pad
                    kd_l = k_depth.copy()
                    kd_l[1, 2] += dpad
                    boxes_l = raw_boxes[ids].astype(np.float64).copy()
                    boxes_l[:, 1] += pad
                    boxes_l[:, 3] += pad
                    corners, _, _ = lift(adapter, scene, frame, rgb_l, depth_l,
                                         k_l, kd_l, pose, boxes_l)
                    keep_ids, keep_corners = ids, np.asarray(corners)
                elif variant == 'control':
                    corners, _, _ = lift(adapter, scene, frame, rgb, depth,
                                         k_rgb, k_depth, pose, raw_boxes[ids])
                    if len(corners):
                        frozen_sel = frozen_corners[positions]
                        control_parity.append(float(np.max(np.abs(
                            np.asarray(corners) - frozen_sel))))
                    keep_ids, keep_corners = ids, np.asarray(corners)
                else:
                    rgb_z, depth_z, kr_z, kd_z, remapped, inside = apply_zoom(
                        rgb, depth, k_rgb, k_depth, raw_boxes[ids], cx, cy)
                    keep = np.flatnonzero(inside)
                    if not len(keep):
                        per_variant[variant] = (np.empty(0, dtype=int),
                                                np.empty((0, 8, 3)))
                        continue
                    corners, _, _ = lift(adapter, scene, frame, rgb_z, depth_z,
                                         kr_z, kd_z, pose, remapped[keep])
                    keep_ids, keep_corners = ids[keep], np.asarray(corners)
                per_variant[variant] = (keep_ids, keep_corners)
            common = None
            for keep_ids, _ in per_variant.values():
                common = set(keep_ids.tolist()) if common is None \
                    else common & set(keep_ids.tolist())
            for variant, (keep_ids, keep_corners) in per_variant.items():
                index = {int(a): i for i, a in enumerate(keep_ids)}
                for anchor in common:
                    corners = keep_corners[index[anchor]]
                    iou = float(box_iou3d(corners[None], gt)[0].max()) \
                        if len(gt) else 0.0
                    key = (scene, frame, anchor)
                    paired.setdefault(key, {})[variant] = iou
                    if iou > .15:
                        support[variant].setdefault(scene, {}).setdefault(
                            'hit', []).append((frame, key))
                    b = best[variant]
                    if iou > b.get(key, (0.0, None))[0]:
                        b[key] = (iou, corners)
        # coverage + append arms per variant
        for variant in VARIANTS:
            per_anchor = {}
            for (sc, fr, anchor) in paired:
                if sc != scene:
                    continue
                entry = paired[(sc, fr, anchor)].get(variant)
                if entry is not None and entry > .15:
                    state = per_anchor.setdefault(anchor, {'frames': set(),
                                                           'best': (0.0, None)})
                    state['frames'].add(fr)
                    stored = best[variant].get((sc, fr, anchor))
                    if stored and stored[0] > state['best'][0]:
                        state['best'] = stored
            extra, extra_scores = [], []
            for anchor, state in per_anchor.items():
                if len(state['frames']) >= 3 and state['best'][1] is not None:
                    extra.append(state['best'][1])
            arms[variant][scene] = (
                np.concatenate([base_boxes, np.asarray(extra)]) if extra else base_boxes,
                np.r_[base_scores, np.full(len(extra), APPEND_SCORE)] if extra else base_scores)
        print(f'{scene}: paired anchors={sum(1 for k in paired if k[0] == scene)}',
              flush=True)
    metrics = {'baseline': {str(t): class_agnostic_ap(baseline, gts, t) for t in THRESHOLDS}}
    for variant in VARIANTS:
        metrics[variant] = {}
        for t in THRESHOLDS:
            value = class_agnostic_ap(arms[variant], gts, t)
            value['delta_ap'] = value['ap'] - metrics['baseline'][str(t)]['ap']
            value['delta_tp'] = value['tp'] - metrics['baseline'][str(t)]['tp']
            value['delta_fp'] = value['fp'] - metrics['baseline'][str(t)]['fp']
            metrics[variant][str(t)] = value
    stats = {}
    for variant in VARIANTS:
        values = [pair[variant] for pair in paired.values()
                  if variant in pair and 'control' in pair]
        control = [pair['control'] for pair in paired.values()
                   if variant in pair and 'control' in pair]
        stats[variant] = {
            'n_paired': len(values),
            'median_iou': float(np.median(values)) if values else None,
            'control_median': float(np.median(control)) if control else None,
            'improved': int(sum(v > c for v, c in zip(values, control))),
            'worsened': int(sum(v < c for v, c in zip(values, control))),
            'ge15': int(sum(v > .15 for v in values)),
            'control_ge15': int(sum(c > .15 for c in control)),
        }
    for path, digest in inputs.items():
        assert sha256(path) == digest, f'Input changed: {path}'
    result = {
        'schema': 'boxfusion.boxer_focal_canon.v1', 'completed': True,
        'train_normalized_focal': [TRAIN_FX_N, TRAIN_FY_N],
        'control_reproduction_max_abs_err': float(np.max(control_parity))
        if control_parity else None,
        'paired_stats': stats, 'metrics': metrics,
        'limits': [
            'Canonical crop factors derived from per-scene K vs fixed ScanNet '
            'training focals; training focals vary slightly across scenes',
            '3 dev scenes, top-M score_m300 anchors only; oracle-append AP '
            'suppresses the FP side as in the seedless Step-1 protocol',
            'Border anchors fall outside the crop and drop from every variant '
            'equally (paired set); GT used only for evaluation',
        ],
    }
    for name, data in (('results.json', json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'),
                       ('input_sha256.json', json.dumps(inputs, ensure_ascii=False, indent=2) + '\n')):
        with (args.output / name).open('x') as handle:
            handle.write(data)
    print(json.dumps({
        'control_parity': result['control_reproduction_max_abs_err'],
        'paired': stats,
        'deltas': {v: [round(metrics[v][str(t)]['delta_ap'], 4)
                       for t in THRESHOLDS] for v in VARIANTS}},
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
