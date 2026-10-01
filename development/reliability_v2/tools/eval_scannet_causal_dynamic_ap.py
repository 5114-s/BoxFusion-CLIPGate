#!/usr/bin/env python3
"""Evaluate saved dynamic maps with the existing ScanNet AP implementation.

No detector inference, prediction changes, threshold tuning, or GT writes.
The synthetic-removal manifest stores world-space *prediction* boxes; its
removed_ids index the builder's candidate list, NOT the ScanNet GT array.
Recover the intended GT by the builder's >=0.5 AABB-overlap eligibility rule,
choose the maximum-overlap GT, and collapse repeated references to the same GT.
This policy is frozen before evaluating either prediction view.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = (0.15, 0.25, 0.50)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def aabb_overlaps(boxes, gt):
    if not len(boxes) or not len(gt):
        return np.zeros((len(boxes), len(gt)))
    lo, hi = boxes.min(1), boxes.max(1)
    gl, gh = gt.min(1), gt.max(1)
    intersection = np.maximum(0, np.minimum(hi[:, None], gh) - np.maximum(lo[:, None], gl)).prod(2)
    union = (hi - lo).prod(1)[:, None] + (gh - gl).prod(1) - intersection
    return intersection / union


def removal_mapping(overlaps):
    """Never interpret manifest removed_ids as GT identities."""
    records, removed = [], set()
    for i, values in enumerate(overlaps):
        if not len(values) or not np.isfinite(values).all() or values.max() < 0.50:
            raise ValueError(f'Removal marker {i} has no GT at builder IoU >= 0.50')
        best = int(values.argmax())
        records.append({'marker_index': i, 'matched_gt_index': best,
                        'best_iou': float(values[best]),
                        'eligible_gt_indices': np.flatnonzero(values >= 0.50).tolist(),
                        'duplicate_gt_reference': best in removed})
        removed.add(best)
    return removed, records


def boxes_array(values, context):
    a = np.asarray(values)
    if not a.size:
        return np.empty((0, 8, 3), dtype=np.float64)
    if a.ndim != 3 or a.shape[1:] != (8, 3) or not np.isfinite(a).all():
        raise ValueError(f'{context}: invalid corners, expected finite [N,8,3]')
    if (np.ptp(a, axis=1) <= 0).any():
        raise ValueError(f'{context}: degenerate box')
    return a


def read_alignment(path):
    matches = [line.split('=', 1)[1] for line in path.read_text().splitlines()
               if line.strip().startswith('axisAlignment')]
    if len(matches) != 1:
        raise ValueError(f'{path}: expected exactly one axisAlignment')
    matrix = np.fromstring(matches[0], sep=' ').reshape(4, 4)
    if not np.isfinite(matrix).all():
        raise ValueError(f'{path}: invalid axisAlignment')
    return matrix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=ROOT / 'data_dyn/manifest.json')
    p.add_argument('--persistent-root', type=Path, default=ROOT / 'results/scannet_dyn_causal_dynamic_persistent')
    p.add_argument('--current-root', type=Path, default=ROOT / 'results/scannet_dyn_causal_dynamic_current')
    p.add_argument('--event-root', type=Path, default=ROOT / 'diagnostics/dynamic_objects/events')
    p.add_argument('--evaluation-root', type=Path, default=ROOT / 'evaluation')
    p.add_argument('--scans-root', type=Path, default=Path('/extra/ZhaoX/scannet_data/scans'))
    p.add_argument('--expected-scenes', type=int, default=75)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise ValueError(f'Refusing to overwrite existing evaluation: {args.output_dir}')

    sys.path.insert(0, str(ROOT))
    from tools.validate_dynamic_run_coverage import read_manifest_scenes, validate_dynamic_run_coverage
    coverage = validate_dynamic_run_coverage(
        args.manifest, persistent_root=args.persistent_root,
        current_root=args.current_root, event_roots=[args.event_root])
    scenes = sorted(read_manifest_scenes(args.manifest))
    if len(scenes) != args.expected_scenes:
        raise ValueError(f'Expected {args.expected_scenes} scenes, got {len(scenes)}')
    manifest = json.loads(args.manifest.read_text())
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / 'coverage.json', coverage)

    # Import the anchor implementation, not a new approximate AP calculator.
    # When launched as tools/foo.py, tools/utils.py otherwise shadows the
    # evaluator's namespace package named utils.
    sys.path[:] = [entry for entry in sys.path
                   if Path(entry or '.').resolve() != Path(__file__).resolve().parent]
    sys.path.insert(0, str(args.evaluation_root))
    # Load the utils namespace before legacy dataset modules append
    # evaluation/utils itself (which contains another utils.py) to sys.path.
    from utils.ap_helper import parse_groundtruths
    from utils.utils import flip_axis_to_camera, obb_to_aabb_corners, reorganize_obb_to_aabb
    from data_util.dataset import ScannetDetectionDataset
    from data_util.model_util_scannet import ScannetDatasetConfig
    from torch.utils.data._utils.collate import default_collate
    from eval_det import eval_det_cls, get_iou_obb

    def to_eval(corners, align):
        if not len(corners):
            return np.empty((0, 8, 3))
        value = np.transpose(align[None, :3, :3] @ np.transpose(corners, (0, 2, 1)), (0, 2, 1))
        value += align[None, :3, 3]
        return reorganize_obb_to_aabb(obb_to_aabb_corners(flip_axis_to_camera(value)))

    np.random.seed(0)
    gt_root = args.evaluation_root / 'data_util/scannet_train_detection_data'
    dataset = ScannetDetectionDataset('val', num_points=40000, augment=False,
                                     use_color=False, use_height=True, data_path=str(gt_root))
    if not set(scenes).issubset(dataset.scan_names):
        raise ValueError('Manifest scenes missing from anchor GT dataset')
    dataset.scan_names = scenes
    cfg = {'dataset_config': ScannetDatasetConfig()}
    source_paths = [Path(__file__), args.manifest,
                    args.evaluation_root / 'eval_scannet.py',
                    args.evaluation_root / 'utils/ap_helper.py',
                    args.evaluation_root / 'utils/eval_det.py',
                    args.evaluation_root / 'utils/box_util.py',
                    args.evaluation_root / 'utils/utils.py',
                    args.evaluation_root / 'data_util/dataset.py',
                    args.evaluation_root / 'data_util/model_util_scannet.py']
    inputs = {str(path.resolve()): sha256(path) for path in source_paths}
    predictions = {'persistent': {}, 'current': {}}
    full_gt, terminal_gt, audit = {}, {}, []
    metric_parity_error = 0.0
    for i, scene in enumerate(scenes):
        meta_path = args.scans_root / scene / f'{scene}.txt'
        inputs[str(meta_path.resolve())] = sha256(meta_path)
        bbox_path = gt_root / f'{scene}_bbox.npy'
        inputs[str(bbox_path.resolve())] = sha256(bbox_path)
        align = read_alignment(meta_path)
        batch = default_collate([dataset[i]])
        gt = boxes_array([row[1] for row in parse_groundtruths(batch, cfg)[0]], scene)
        markers = boxes_array(np.asarray(manifest[scene]['removed'], dtype=float).reshape(-1, 8, 3), scene)
        transformed_markers = to_eval(markers, align)
        overlaps = aabb_overlaps(transformed_markers, gt)
        removed, mapping = removal_mapping(overlaps)
        # Both arguments are canonical AABBs; verify mapping geometry against
        # the exact corner-based IoU used by the official ScanNet AP path.
        for r in mapping:
            marker, target = r['marker_index'], r['matched_gt_index']
            error = abs(get_iou_obb(transformed_markers[marker], gt[target]) - overlaps[marker, target])
            metric_parity_error = max(metric_parity_error, float(error))
        if metric_parity_error > 1e-6:
            raise ValueError(f'Anchor IoU mapping mismatch: {metric_parity_error}')
        full_gt[scene] = list(gt)
        terminal_gt[scene] = [g for j, g in enumerate(gt) if j not in removed]
        for view, root in [('persistent', args.persistent_root), ('current', args.current_root)]:
            path = root / f'{scene}_boxes.pkl'
            inputs[str(path.resolve())] = sha256(path)
            with path.open('rb') as f:
                payload = pickle.load(f)
            if not isinstance(payload, (tuple, list)) or len(payload) != 1:
                raise ValueError(f'{path}: expected exactly one final scene output')
            rows = payload[0]
            corners = boxes_array([r[1] for r in rows], str(path))
            scores = [float(r[2]) for r in rows]
            if not np.isfinite(scores).all():
                raise ValueError(f'{path}: nonfinite score')
            predictions[view][scene] = list(zip(to_eval(corners, align), scores))
        entry = {'scene': scene, 'full_gt': len(gt), 'terminal_gt': len(terminal_gt[scene]),
                 'removal_markers': len(markers), 'unique_removed_gt': len(removed),
                 'manifest_candidate_ids_not_gt_ids': manifest[scene].get('removed_ids'),
                 'mapping': mapping,
                 'persistent_predictions': len(predictions['persistent'][scene]),
                 'current_predictions': len(predictions['current'][scene])}
        audit.append(entry)
        print(f'PREPARED {i+1}/{len(scenes)} {scene} GT={len(gt)} removed={len(removed)} '
              f'persistent={entry["persistent_predictions"]} current={entry["current_predictions"]}', flush=True)

    write_json(args.output_dir / 'gt_identity_audit.json', {
        'policy': 'best AABB IoU >= 0.50, matching builder eligibility; duplicate GT references collapsed',
        'removed_ids_are_not_gt_ids': True, 'anchor_mapping_iou_max_error': metric_parity_error,
        'scenes': audit})
    write_json(args.output_dir / 'input_sha256.json', inputs)
    result = {'schema': 'boxfusion.causal_dynamic_ap.v1',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'scene_count': len(scenes), 'scene_order': scenes,
              'full_gt_count': sum(map(len, full_gt.values())),
              'terminal_gt_count': sum(map(len, terminal_gt.values())),
              'removal_marker_count': sum(x['removal_markers'] for x in audit),
              'unique_removed_gt_count': sum(x['unique_removed_gt'] for x in audit),
              'prediction_counts': {v: sum(map(len, scenes_.values())) for v, scenes_ in predictions.items()},
              'protocol': {'class_agnostic': True, 'real_score': True,
                           'gt': 'anchor ScannetDetectionDataset + parse_groundtruths',
                           'transform': 'axisAlignment -> camera axes -> AABB -> canonical corners',
                           'metric': 'anchor eval_det_cls(get_iou_obb), continuous VOC AP, strict IoU > t',
                           'extra_prediction_filtering': False,
                           'extra_nms': False, 'new_inference': False,
                           'persistent_is_disabled_baseline': False,
                           'current_gt': 'original GT minus unique manifest-matched removed instances'},
              'metrics': {}}
    for label, view, gt in [('persistent_terminal', 'persistent', terminal_gt),
                            ('current_terminal', 'current', terminal_gt),
                            ('persistent_full', 'persistent', full_gt)]:
        result['metrics'][label] = {}
        for threshold in THRESHOLDS:
            print(f'EVALUATING {label} IoU={threshold}', flush=True)
            rec, prec, ap = eval_det_cls(predictions[view], gt, ovthresh=threshold,
                                         use_07_metric=False, get_iou_func=get_iou_obb)
            values = {'AP': float(ap * 100), 'Recall': float(rec[-1] * 100) if len(rec) else 0.0,
                      'Precision': float(prec[-1] * 100) if len(prec) else 0.0}
            result['metrics'][label][f'{threshold:.2f}'] = values
            print(f'RESULT {label} IoU={threshold} AP={values["AP"]:.6f} '
                  f'Recall={values["Recall"]:.6f} Precision={values["Precision"]:.6f}', flush=True)
    for path, expected in inputs.items():
        if sha256(path) != expected:
            raise ValueError(f'Evaluation input changed during execution: {path}')
    result['input_integrity_passed'] = True
    write_json(args.output_dir / 'results.json', result)
    lines = ['# ScanNet-Dyn causal dynamic：75场保存输出AP', '',
             f'场景：{len(scenes)}；原始GT：{result["full_gt_count"]}；终态GT：{result["terminal_gt_count"]}。',
             f'移除框标记：{result["removal_marker_count"]}；匹配去重后移除GT：{result["unique_removed_gt_count"]}。', '',
             '| 预测 / GT口径 | AP15 | AP25 | AP50 |', '|---|---:|---:|---:|']
    for key, title in [('persistent_terminal', 'persistent / 终态GT'),
                       ('current_terminal', 'current / 同一终态GT'),
                       ('persistent_full', 'persistent / 原始完整GT')]:
        values = [result['metrics'][key][f'{t:.2f}']['AP'] for t in THRESHOLDS]
        lines.append('| ' + title + ' | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |')
    delta = [result['metrics']['current_terminal'][f'{t:.2f}']['AP'] -
             result['metrics']['persistent_terminal'][f'{t:.2f}']['AP'] for t in THRESHOLDS]
    lines.extend(['| current − persistent（同终态GT） | ' + ' | '.join(f'{v:+.4f}' for v in delta) + ' |', '',
                  '## 评测边界', '',
                  '- 仅评测已有75场预测；未重跑模型，未修改score、几何或GT源文件。',
                  '- 两种终态AP使用完全相同的GT；移除实例的旧位预测不会被忽略，按匹配规则计FP。',
                  '- 复用现有ScanNet的GT解析、坐标转换、IoU及AP函数。无额外NMS/score阈值。',
                  '- manifest removed_ids是生成器候选索引，不是GT索引。以生成器固定IoU>=0.5规则恢复最大重叠GT，重复映射合并；详见gt_identity_audit.json。',
                  '- 新分支persistent本身有几何/去重处理，不等于M5-off或未修改原版；两输出差值不是独立模块消融。',
                  '- 这是合成移除流的终态class-agnostic AP，不是逐帧动态AP、真实移动人物性能、语义AP或FPS。',
                  '- 不应将旧24场后处理M5的数字当作本75场的配对基线。', '',
                  '产物：results.json、coverage.json、gt_identity_audit.json、input_sha256.json。'])
    with (args.output_dir / 'REPORT.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'DYNAMIC_AP_COMPLETE {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
