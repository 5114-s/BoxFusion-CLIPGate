#!/usr/bin/env python3
"""Evaluate saved maps on the synthetic walking benchmark.

Terminal GT keeps every original GT object; the walker's GT box is translated
by the manifest's final_offset (the person still exists, at the final
position).  Walker identity is recovered with the frozen max-AABB-IoU >= 0.5
rule on the original-position marker, mirroring the removal benchmark.

Metrics:
- AP15/25/50 on terminal GT (overall) and on terminal GT minus the walker
  (static-only, collateral check), anchor ScanNet AP implementation;
- walker: best IoU vs final position (recall proxy), stale-at-origin flag,
  trail count = predictions matching some swept corridor position (IoU >= 0.25)
  but not the final position, and not explainable by other GT (IoU >= 0.5).
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


def read_alignment(path):
    matches = [line.split('=', 1)[1] for line in path.read_text().splitlines()
               if line.strip().startswith('axisAlignment')]
    matrix = np.fromstring(matches[0], sep=' ').reshape(4, 4)
    return matrix


def gt_world_corners(scene, gt_root):
    axis = read_alignment(
        Path(f'/extra/ZhaoX/scannet_data/scans/{scene}/{scene}.txt'))
    Tinv = np.linalg.inv(axis)
    gt = np.load(gt_root / f'{scene}_bbox.npy')
    out = []
    for c, s in zip(gt[:, :3], gt[:, 3:6]):
        half = s / 2.0
        local = np.array([[dx, dy, dz] for dx in (-half[0], half[0])
                          for dy in (-half[1], half[1]) for dz in (-half[2], half[2])])
        corners = local + c
        out.append((Tinv[:3, :3] @ corners.T).T + Tinv[:3, 3])
    return out, axis


def aabb_overlaps(boxes, gt):
    lo, hi = boxes.min(1), boxes.max(1)
    gl, gh = gt.min(1), gt.max(1)
    inter = np.maximum(0, np.minimum(hi[:, None], gh) - np.maximum(lo[:, None], gl)).prod(2)
    union = (hi - lo).prod(1)[:, None] + (gh - gl).prod(1) - inter
    return inter / union


def get_iou_aabb(box, gt):
    """Exact IoU for the axis-aligned geometry used by this benchmark.

    Avoid polygon clipping at coincident edges: the legacy OBB routine can
    return negative self-IoU for otherwise valid thin axis-aligned boxes.
    """
    return float(aabb_overlaps(np.asarray(box, dtype=float)[None],
                               np.asarray(gt, dtype=float)[None])[0, 0])


def boxes_array(values, context):
    a = np.asarray(values)
    if not a.size:
        return np.empty((0, 8, 3), dtype=np.float64)
    if a.ndim != 3 or a.shape[1:] != (8, 3) or not np.isfinite(a).all():
        raise ValueError(f'{context}: invalid corners, expected finite [N,8,3]')
    if (np.ptp(a, axis=1) <= 0).any():
        raise ValueError(f'{context}: degenerate box')
    return a


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=ROOT / 'data_dyn_walk/manifest.json')
    p.add_argument('--pred-root', type=Path, required=True)
    p.add_argument('--evaluation-root', type=Path, default=ROOT / 'evaluation')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--label', default='arm')
    args = p.parse_args()
    if args.output_dir.exists():
        raise ValueError(f'Refusing to overwrite: {args.output_dir}')

    sys.path.insert(0, str(ROOT))
    sys.path[:] = [e for e in sys.path
                   if Path(e or '.').resolve() != Path(__file__).resolve().parent]
    sys.path.insert(0, str(args.evaluation_root))
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
                                      use_color=False, use_height=True,
                                      data_path=str(gt_root))
    manifest = json.loads(args.manifest.read_text())
    scenes = sorted(manifest)
    dataset.scan_names = scenes
    cfg = {'dataset_config': ScannetDatasetConfig()}

    preds_all, gt_term, gt_static = {}, {}, {}
    walker_rows = []
    iou_parity_error = 0.0
    legacy_self_anomalies = []
    for i, scene in enumerate(scenes):
        entry = manifest[scene]
        corners_world, align = gt_world_corners(scene, gt_root)
        marker = np.asarray(entry['marker'], float).reshape(8, 3)
        overlaps = aabb_overlaps(marker[None], np.stack([c for c in corners_world]))[0]
        if overlaps.max() < 0.50:
            raise ValueError(f'{scene}: walker marker has no GT at IoU >= 0.5')
        walker_idx = int(overlaps.argmax())
        final_offset = np.asarray(entry['final_offset'], float)
        # Anchor IoU needs the GT parser's winding order. The prediction
        # reordering helper produces a different winding and is NOT a GT
        # constructor (using it for GT makes polygon intersection return zero).
        parsed_gt = boxes_array([r[1] for r in parse_groundtruths(
            default_collate([dataset[i]]), cfg)[0]], scene).astype(np.float64)
        bounds_check = to_eval(np.asarray(corners_world), align)
        if (len(parsed_gt) != len(corners_world)
                or not np.allclose(parsed_gt.min(1), bounds_check.min(1), atol=1e-5)
                or not np.allclose(parsed_gt.max(1), bounds_check.max(1), atol=1e-5)):
            raise ValueError(f'{scene}: GT identity/bounds mismatch')
        legacy_self = float(get_iou_obb(parsed_gt[walker_idx], parsed_gt[walker_idx]))
        if abs(legacy_self-1) > 1e-6:
            legacy_self_anomalies.append({'scene': scene, 'legacy_self_iou': legacy_self})
        if abs(get_iou_aabb(parsed_gt[walker_idx], parsed_gt[walker_idx])-1) > 1e-12:
            raise ValueError(f'{scene}: analytic AABB self-check failed')
        gt_term_eval = [g.copy() for g in parsed_gt]
        offset_eval = flip_axis_to_camera((align[:3, :3] @ final_offset)[None])[0]
        walker_final_eval = parsed_gt[walker_idx] + offset_eval
        gt_term_eval[walker_idx] = walker_final_eval
        gt_static_eval = [g for j, g in enumerate(gt_term_eval) if j != walker_idx]

        path = args.pred_root / f'{scene}_boxes.pkl'
        with path.open('rb') as f:
            payload = pickle.load(f)
        rows = payload[0]
        pred_corners = boxes_array([r[1] for r in rows], str(path))
        scores = np.asarray([float(r[2]) for r in rows])
        pred_eval = to_eval(pred_corners, align)
        # Canonical AABBs permit an independent order-free IoU parity check.
        quick = aabb_overlaps(pred_eval, np.asarray(gt_term_eval)) if len(pred_eval) else np.empty((0, len(gt_term_eval)))
        for k in range(len(pred_eval)):
            j = int(quick[k].argmax())
            iou_parity_error = max(iou_parity_error, abs(float(quick[k,j])-get_iou_obb(pred_eval[k], gt_term_eval[j])))

        # walker diagnostics in the eval frame
        direction = np.asarray(entry['direction'], float)
        dmax = float(entry['D_max'])
        corridor = []
        for t in np.arange(0.0, dmax + 1e-9, 0.1):
            delta_eval = flip_axis_to_camera((align[:3, :3] @ (direction*t))[None])[0]
            corridor.append(parsed_gt[walker_idx] + delta_eval)
        best_final = 0.0
        stale_origin = 0
        trail = []
        for k in range(len(pred_eval)):
            iou_final = get_iou_aabb(pred_eval[k], walker_final_eval)
            best_final = max(best_final, iou_final)
            iou_a = get_iou_aabb(pred_eval[k], corridor[0])
            if iou_a >= 0.25 and iou_final < 0.25:
                stale_origin += 1
            if iou_final >= 0.25:
                continue
            corridor_hit = max(
                get_iou_aabb(pred_eval[k], c) for c in corridor)
            if corridor_hit < 0.25:
                continue
            other = max((get_iou_aabb(pred_eval[k], g)
                         for j, g in enumerate(gt_term_eval) if j != walker_idx),
                        default=0.0)
            if other < 0.50:
                trail.append(float(corridor_hit))
        walker_rows.append({
            'scene': scene, 'best_iou_final': float(best_final),
            'recall25_final': int(best_final >= 0.25),
            'recall50_final': int(best_final >= 0.50),
            'stale_at_origin': stale_origin, 'trail_boxes': len(trail),
            'n_pred': int(len(pred_eval)),
        })
        preds_all[scene] = list(zip(pred_eval, scores.tolist()))
        gt_term[scene] = gt_term_eval
        gt_static[scene] = gt_static_eval
        print(f'PREP {i+1}/{len(scenes)} {scene} preds={len(pred_eval)} '
              f'walker_iou={best_final:.3f} trail={len(trail)} stale={stale_origin}', flush=True)

    result = {'schema': 'boxfusion.walk_bench.v1', 'label': args.label,
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'scene_count': len(scenes), 'walker': walker_rows, 'metrics': {},
              'legacy_prediction_iou_max_error': float(iou_parity_error),
              'legacy_self_iou_anomalies': legacy_self_anomalies,
              'iou_method': 'analytic AABB; native AP score sorting, greedy matching and integration',
              'gt_source': 'official parse_groundtruths order; translated only the walker'}
    for label, gt in [('terminal', gt_term), ('static_only', gt_static)]:
        result['metrics'][label] = {}
        for t in THRESHOLDS:
            rec, prec, ap = eval_det_cls(preds_all, gt, ovthresh=t,
                                         use_07_metric=False, get_iou_func=get_iou_aabb)
            result['metrics'][label][f'{t:.2f}'] = {
                'AP': float(ap * 100),
                'Recall': float(rec[-1] * 100) if len(rec) else 0.0}
            print(f'RESULT {label} IoU={t} AP={ap*100:.4f}', flush=True)

    w = result['walker']
    result['walker_summary'] = {
        'scenes': len(w),
        'mean_best_iou_final': float(np.mean([r['best_iou_final'] for r in w])),
        'recall25_final': int(sum(r['recall25_final'] for r in w)),
        'recall50_final': int(sum(r['recall50_final'] for r in w)),
        'total_trail_boxes': int(sum(r['trail_boxes'] for r in w)),
        'scenes_with_stale_origin': int(sum(r['stale_at_origin'] > 0 for r in w)),
        'total_stale_origin': int(sum(r['stale_at_origin'] for r in w)),
    }
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / 'results.json', result)
    m = result['metrics']
    ws = result['walker_summary']
    lines = [f'# 行走基准评估：{args.label}', '',
             f'场景：{len(scenes)}；行者最终位召回@25：{ws["recall25_final"]}/{ws["scenes"]}'
             f'（@50：{ws["recall50_final"]}/{ws["scenes"]}）；行者最终位平均最优IoU：'
             f'{ws["mean_best_iou_final"]:.4f}；走廊重复框总数：{ws["total_trail_boxes"]}'
             f'；原位残留场景数：{ws["scenes_with_stale_origin"]}', '',
             '| GT口径 | AP15 | AP25 | AP50 |', '|---|---:|---:|---:|']
    for key, title in [('terminal', '终态GT（行者在最终位）'), ('static_only', '静态GT（除行者）')]:
        v = [m[key][f'{t:.2f}']['AP'] for t in THRESHOLDS]
        lines.append(f'| {title} | ' + ' | '.join(f'{x:.4f}' for x in v) + ' |')
    lines += ['', '## 边界', '',
              '- 终态协议：行者仍在场景中，GT 移到最终位；走廊上的中间位预测按 FP 计入 AP。',
              '- 行者身份用生成器 marker 的最大 AABB IoU>=0.5 规则恢复。',
              '- trail 框定义：与走廊某位置 IoU>=0.25、与最终位 <0.25、与其他 GT <0.5。',
              '- static_only保留全部预测、仅移除移动目标GT，移动目标的正确预测也可能成为FP；不能单独据此判断静态物体误伤。',
              '- 这是合成刚性位移与终态地图评测，不含逐帧动态AP、身份保持或真人运动模糊结论。',
              '- 本评估不重跑推理；只读已保存的输出。']
    lines += ['- AP沿用原生排序、匹配、积分，IoU使用解析AABB；旧多边形IoU在个别GT自交测试出现负值，不能当作可靠几何实现。',
              f'- 实际预测各自最大重叠GT处，解析IoU与旧实现的最大差值：{iou_parity_error:.3g}。']
    with (args.output_dir / 'REPORT.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'WALK_EVAL_COMPLETE {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
