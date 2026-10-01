#!/usr/bin/env python3
"""Scene-level paired bootstrap for ReCaR-3D full vs Base on extra50.

Reuses the official evaluator's exact pipeline: GT comes from
ScannetDetectionDataset + parse_groundtruths (labels forced to class 0),
predictions are axis-aligned, flipped, and converted with the same
helpers, per-scene greedy matching mirrors eval_det_cls with
get_iou_obb (box3d_iou), and the pooled AP tail (cumsum, rec=tp/(npos+1e-6),
voc_ap) is the official formula.  The aggregate pooled APs are asserted
against the official evaluator log values before bootstrapping.

Each bootstrap iteration resamples the 50 scene indices with replacement
and re-aggregates Base and Full with the same indices (paired), then
takes the AP difference at IoU 0.15/0.25/0.50.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

EVAL_ROOT = Path('/data/ZhaoX/BoxFusion/development/reliability_v2/evaluation_extra50')
DEV = Path('/data/ZhaoX/BoxFusion/development/reliability_v2')
FACTORIAL = DEV / 'results/scannet_recar3d_extra50_factorial'
SCENE_LIST = Path('/data/ZhaoX/BoxFusion/evaluation/data_util/meta_data/scannetv2_val_f0_extra50.txt')
SCANS = Path('/extra/ZhaoX/scannet_data/scans')
OUTPUT = Path('/data/ZhaoX/BoxFusion/reports/recar3d_extra50_20260928')
THRESHOLDS = (0.15, 0.25, 0.50)
B = 10_000
SEED = 20260928
OFFICIAL = {  # from logs/scannet_extra50_real_score (percent)
    "base": [33.3306, 29.2950, 12.4559],
    "full": [43.2482, 38.1948, 16.8572],
}

# __file__ can be relative at startup, so resolve it before chdir.
_here = os.path.dirname(os.path.abspath(__file__))
os.chdir(EVAL_ROOT)
# The tools/ directory of the dev tree contains its own utils.py which would
# shadow the evaluator's utils namespace package on sys.path.  Keep only
# EVAL_ROOT on the path; importing data_util.dataset appends the flat utils
# directory itself (same import chain as the official eval_scannet.py).
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _here]
sys.path.insert(0, str(EVAL_ROOT))

# Import order mirrors the official eval_scannet.py: binding the utils
# namespace package first prevents dataset.py's later sys.path append of the
# flat utils directory from resolving top-level ``utils`` to utils/utils.py.
from utils.utils import (  # noqa: E402
    flip_axis_to_camera, obb_to_aabb_corners, reorganize_obb_to_aabb)
from torch.utils.data import DataLoader  # noqa: E402
from data_util.dataset import ScannetDetectionDataset  # noqa: E402
from data_util.model_util_scannet import ScannetDatasetConfig  # noqa: E402
from utils.ap_helper import parse_groundtruths  # noqa: E402
from utils.eval_det import get_iou_obb, voc_ap  # noqa: E402


def load_predictions(scene: str, arm: str):
    import pickle
    with open(FACTORIAL / arm / f'{scene}_boxes.pkl', 'rb') as handle:
        payload = pickle.load(handle)
    rows = payload[0]
    meta_file = SCANS / scene / f'{scene}.txt'
    axis_align_matrix = None
    for line in meta_file.read_text().splitlines():
        if 'axisAlignment' in line:
            axis_align_matrix = [
                float(x) for x in
                line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    bbox = np.asarray([row[1] for row in rows], dtype=np.float64)
    scores = [float(row[2]) for row in rows]
    if len(rows) == 0:
        return np.zeros((0, 8, 3)), np.zeros(0)
    transformed = axis_align_matrix[None, :3, :3] @ np.transpose(bbox, (0, 2, 1))
    transformed = np.transpose(transformed, (0, 2, 1)) + axis_align_matrix[None, :3, 3]
    corners = reorganize_obb_to_aabb(obb_to_aabb_corners(flip_axis_to_camera(transformed)))
    return np.asarray(corners, dtype=np.float64), np.asarray(scores, dtype=np.float64)


def scene_tp_flags(pred_corners, pred_scores, gt_corners, threshold):
    """Mirror of eval_det_cls greedy matching for one scene.

    Returns (tp flags in original row order, ious unused)."""
    n = len(pred_scores)
    tp = np.zeros(n)
    if n == 0 or len(gt_corners) == 0:
        return tp
    order = np.argsort(-pred_scores)
    matched = np.zeros(len(gt_corners), dtype=bool)
    iou_matrix = np.zeros((n, len(gt_corners)))
    for i in range(n):
        for j in range(len(gt_corners)):
            iou_matrix[i, j] = get_iou_obb(pred_corners[i], gt_corners[j])
    for d in order:
        values = iou_matrix[d]
        jmax = int(np.argmax(values))
        if values[jmax] > threshold and not matched[jmax]:
            matched[jmax] = True
            tp[d] = 1.0
    return tp


def pooled_ap(pairs, npos):
    """Official eval_det_cls AP tail over pooled (score, tp) pairs."""
    scores = np.concatenate([p[0] for p in pairs]) if pairs else np.zeros(0)
    tp = np.concatenate([p[1] for p in pairs]) if pairs else np.zeros(0)
    order = np.argsort(-scores)
    tp = tp[order]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1.0 - tp)
    rec = ctp / float(npos + 1e-6)
    prec = ctp / np.maximum(ctp + cfp, np.finfo(np.float64).eps)
    return voc_ap(rec, prec, False) * 100.0


def main():
    scenes = [row.strip() for row in SCENE_LIST.read_text().splitlines()
              if row.strip() and not row.lstrip().startswith('#')]
    assert len(scenes) == 50

    dataset_config = ScannetDatasetConfig()
    config_dict = {
        'remove_empty_box': True,
        'use_3d_nms': True, 'nms_iou': 0.25, 'use_old_type_nms': False,
        'cls_nms': True, 'per_class_proposal': True, 'conf_thresh': 0.05,
        'dataset_config': dataset_config,
    }
    dataset = ScannetDetectionDataset(
        'val', num_points=40000, augment=False, use_color=False,
        use_height=True, data_path='./data_util/scannet_train_detection_data')
    print(f'kept {len(dataset)} scenes', flush=True)
    assert set(dataset.scan_names) == set(scenes)

    gt_corners = {}
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    import torch
    for batch in loader:
        scene = batch['scan_name'][0]
        gt_map = parse_groundtruths(batch, config_dict)
        boxes = np.asarray([row[1] for row in gt_map[0]], dtype=np.float64)
        gt_corners[scene] = boxes.reshape(-1, 8, 3) if boxes.size else np.zeros((0, 8, 3))
        print(f'GT {scene}: {len(gt_corners[scene])}', flush=True)

    data = {}
    for arm in ('base', 'full'):
        per_scene = {}
        for scene in scenes:
            corners, scores = load_predictions(scene, arm)
            gt = gt_corners[scene]
            entry = {'npos': len(gt), 'scores': scores}
            for threshold in THRESHOLDS:
                entry[f'tp_{threshold}'] = scene_tp_flags(corners, scores, gt, threshold)
            per_scene[scene] = entry
        data[arm] = per_scene

    aggregate = {}
    for arm in ('base', 'full'):
        values = []
        for threshold in THRESHOLDS:
            pairs = [(data[arm][s]['scores'], data[arm][s][f'tp_{threshold}'])
                     for s in scenes]
            npos = sum(data[arm][s]['npos'] for s in scenes)
            values.append(pooled_ap(pairs, npos))
        aggregate[arm] = values
        print(f'pooled {arm}: {values}', flush=True)
        for computed, reference in zip(values, OFFICIAL[arm]):
            # the full arm contains many exactly tied birth scores; the
            # per-scene vs pooled quicksort tie order differs, an
            # irreducible <=1e-3 AP offset vs the official aggregate
            if abs(computed - reference) > 1e-3:
                raise RuntimeError(
                    f'pooled AP mismatch for {arm}: {computed} vs official {reference}')

    rng = np.random.default_rng(SEED)
    deltas = np.zeros((B, len(THRESHOLDS)))
    for b in range(B):
        idx = rng.integers(0, len(scenes), size=len(scenes))
        for k, threshold in enumerate(THRESHOLDS):
            aps = {}
            for arm in ('base', 'full'):
                pairs = [(data[arm][scenes[i]]['scores'],
                          data[arm][scenes[i]][f'tp_{threshold}']) for i in idx]
                npos = sum(data[arm][scenes[i]]['npos'] for i in idx)
                aps[arm] = pooled_ap(pairs, npos)
            deltas[b, k] = aps['full'] - aps['base']
        if (b + 1) % 1000 == 0:
            print(f'bootstrap {b + 1}/{B}', flush=True)

    summary = {
        'schema': 'boxfusion.recar3d.extra50.bootstrap.v1',
        'method': 'scene-level paired bootstrap, identical indices for Base and Full',
        'iterations': B,
        'seed': SEED,
        'thresholds': list(THRESHOLDS),
        'aggregate_ap': aggregate,
        'point_estimates_delta': [aggregate['full'][k] - aggregate['base'][k]
                                  for k in range(len(THRESHOLDS))],
        'ci95': [[float(np.percentile(deltas[:, k], 2.5)),
                  float(np.percentile(deltas[:, k], 97.5))]
                 for k in range(len(THRESHOLDS))],
        'p_delta_positive': [float((deltas[:, k] > 0).mean())
                             for k in range(len(THRESHOLDS))],
        'note': ('aggregate pooled APs reproduce the official evaluator '
                 'within exact-tie ordering noise (<1e-3 AP); bootstrap '
                 'deltas use the same per-scene greedy matching and pooled '
                 'voc_ap formula.')
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'bootstrap.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in
                      ('point_estimates_delta', 'ci95', 'p_delta_positive')}, indent=2))


if __name__ == '__main__':
    main()
