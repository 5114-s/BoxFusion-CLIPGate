#!/usr/bin/env python3
"""Paired terminal AP + stale/live coverage on the fixed 75-scene removal set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.eval_scannet_causal_dynamic_ap import (
    THRESHOLDS, aabb_overlaps, boxes_array, read_alignment, removal_mapping, sha256, write_json,
)
from tools.validate_dynamic_run_coverage import validate_dynamic_run_coverage

ARMS = ('native_off', 'full_on', 'no_miss_retirement')


def covered_gt(predicted_boxes, scores, gt_boxes, *, threshold=0.25, score_floor=0.30):
    """Auxiliary instance coverage, not one-to-one AP or an identity metric."""
    selected = np.asarray(scores) > score_floor
    overlaps = aabb_overlaps(predicted_boxes[selected], gt_boxes)
    return (overlaps > threshold).any(axis=0)


def coverage_delta(reference, variant):
    return {'reference_covered': int(reference.sum()), 'variant_covered': int(variant.sum()),
            'lost': int((reference & ~variant).sum()), 'gained': int((~reference & variant).sum())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    args = p.parse_args()
    run_dir = args.run_dir.resolve()
    output = run_dir / 'evaluation'
    if output.exists():
        raise ValueError(f'Refusing to overwrite evaluation {output}')
    protocol = json.loads((run_dir / 'protocol.json').read_text())
    scenes = protocol['scenes']
    if len(scenes) != 75 or scenes != sorted(json.loads(Path(protocol['manifest']).read_text())):
        raise ValueError('Not the frozen full75 cohort')
    for f, h in protocol['frozen_sha256'].items():
        if sha256(f) != h:
            raise ValueError(f'Frozen source/config changed: {f}')
    coverage = {}
    for arm in ARMS:
        target = run_dir / arm
        coverage[arm] = validate_dynamic_run_coverage(
            protocol['manifest'], persistent_root=target / 'persistent',
            current_root=target / 'current' if arm != 'native_off' else None,
            event_roots=[target / 'events'] if arm != 'native_off' else ())

    evaluation = ROOT / 'evaluation'
    # Keep the exact ScanNet anchor imports, avoiding tools/utils.py shadowing.
    sys.path[:] = [s for s in sys.path if Path(s or '.').resolve() != Path(__file__).resolve().parent]
    sys.path.insert(0, str(evaluation))
    from utils.ap_helper import parse_groundtruths
    from utils.utils import flip_axis_to_camera, obb_to_aabb_corners, reorganize_obb_to_aabb
    from data_util.dataset import ScannetDetectionDataset
    from data_util.model_util_scannet import ScannetDatasetConfig
    from torch.utils.data._utils.collate import default_collate
    from eval_det import eval_det_cls, get_iou_obb

    def to_eval(corners, alignment):
        if not len(corners):
            return np.empty((0, 8, 3))
        transformed = corners @ alignment[:3, :3].T + alignment[:3, 3]
        return reorganize_obb_to_aabb(obb_to_aabb_corners(flip_axis_to_camera(transformed)))

    sources = [Path(__file__), run_dir / 'protocol.json', Path(protocol['manifest']),
               evaluation / 'utils/ap_helper.py', evaluation / 'utils/eval_det.py',
               evaluation / 'utils/box_util.py', evaluation / 'utils/utils.py',
               evaluation / 'data_util/dataset.py', evaluation / 'data_util/model_util_scannet.py']
    hashes = {str(f): sha256(f) for f in sources}
    gt_root = evaluation / 'data_util/scannet_train_detection_data'
    np.random.seed(0)
    dataset = ScannetDetectionDataset('val', num_points=40000, augment=False,
                                     use_color=False, use_height=True, data_path=str(gt_root))
    if not set(scenes).issubset(dataset.scan_names):
        raise ValueError('Missing GT scenes')
    dataset.scan_names = scenes
    manifest = json.loads(Path(protocol['manifest']).read_text())
    gt_config = {'dataset_config': ScannetDatasetConfig()}
    variants = dict.fromkeys(ARMS)
    variants['full_on_persistent_diagnostic'] = None
    predictions = {k: {} for k in variants}
    terminal_gt, audit = {}, []
    metrics_aux = {k: {'stale': [], 'live': []} for k in variants}
    timing = {k: [] for k in ARMS}
    n_full, n_removed = 0, 0
    for i, scene in enumerate(scenes):
        metadata = Path('/extra/ZhaoX/scannet_data/scans') / scene / f'{scene}.txt'
        for path in (metadata, gt_root / f'{scene}_bbox.npy'):
            hashes[str(path)] = sha256(path)
        align = read_alignment(metadata)
        gt = boxes_array([r[1] for r in parse_groundtruths(default_collate([dataset[i]]), gt_config)[0]], scene)
        markers = boxes_array(np.asarray(manifest[scene]['removed']).reshape(-1, 8, 3), scene)
        transformed = to_eval(markers, align)
        removed, mapping = removal_mapping(aabb_overlaps(transformed, gt))
        for m in mapping:
            if abs(get_iou_obb(transformed[m['marker_index']], gt[m['matched_gt_index']]) - m['best_iou']) > 1e-6:
                raise ValueError('GT marker mapping IoU differs from anchor')
        live = np.asarray([g for j, g in enumerate(gt) if j not in removed]).reshape(-1, 8, 3)
        removed_boxes = gt[sorted(removed)]
        terminal_gt[scene] = list(live)
        n_full += len(gt)
        n_removed += len(removed)
        scene_record = {'scene': scene, 'full_gt': len(gt), 'live_gt': len(live), 'mapping': mapping, 'views': {}}
        for key in variants:
            arm = 'full_on' if key == 'full_on_persistent_diagnostic' else key
            view = 'persistent' if key in ('native_off', 'full_on_persistent_diagnostic') else 'current'
            path = run_dir / arm / view / f'{scene}_boxes.pkl'
            hashes[str(path)] = sha256(path)
            with path.open('rb') as f:
                payload = pickle.load(f)
            if not isinstance(payload, (tuple, list)) or len(payload) != 1:
                raise ValueError(f'Expected exactly one terminal output: {path}')
            rows = payload[0]
            corners = to_eval(boxes_array([r[1] for r in rows], str(path)), align)
            scores = np.asarray([float(r[2]) for r in rows])
            if not np.isfinite(scores).all():
                raise ValueError(f'Nonfinite score {path}')
            predictions[key][scene] = list(zip(corners, scores))
            stale = covered_gt(corners, scores, removed_boxes)
            retained = covered_gt(corners, scores, live)
            metrics_aux[key]['stale'].append(stale)
            metrics_aux[key]['live'].append(retained)
            scene_record['views'][key] = {'boxes': len(rows), 'stale_gt': int(stale.sum()),
                                        'live_gt_covered': int(retained.sum())}
        audit.append(scene_record)
        for arm in ARMS:
            path = run_dir / arm / 'logs' / f'{scene}.result.json'
            hashes[str(path)] = sha256(path)
            record = json.loads(path.read_text())
            if record['returncode'] or not record['loop_seconds']:
                raise ValueError(f'Incomplete or invalid trial: {path}')
            for f, h in record['artifacts'].items():
                if sha256(f) != h:
                    raise ValueError(f'Trial artifact changed: {f}')
                hashes[f] = h
            timing[arm].append(record)
        print(f'PREPARED {i+1}/75 {scene}', flush=True)

    results = {'schema': 'boxfusion.dynamic_pair75_results.v1', 'scenes': 75,
               'full_gt': n_full, 'removed_gt': n_removed, 'terminal_gt': n_full - n_removed,
               'metrics': {}, 'auxiliary': {}, 'timing': {}, 'delta_full_minus_native': {}}
    for key in variants:
        result = {'boxes': sum(len(p) for p in predictions[key].values())}
        for threshold in THRESHOLDS:
            rec, prec, ap = eval_det_cls(predictions[key], terminal_gt, ovthresh=threshold,
                                        use_07_metric=False, get_iou_func=get_iou_obb)
            result[f'{threshold:.2f}'] = {'AP': float(ap*100),
                                        'Recall': float(rec[-1]*100) if len(rec) else 0.,
                                        'Precision': float(prec[-1]*100) if len(prec) else 0.}
            print(f'RESULT {key} IoU={threshold} AP={ap*100:.4f}', flush=True)
        results['metrics'][key] = result
        stale = np.concatenate(metrics_aux[key]['stale'])
        live = np.concatenate(metrics_aux[key]['live'])
        results['auxiliary'][key] = {
            'stale_count': int(stale.sum()), 'stale_denominator': n_removed,
            'stale_percent': float(stale.mean()*100),
            'stale_vs_native': coverage_delta(np.concatenate(metrics_aux['native_off']['stale']), stale),
            'live_vs_native': coverage_delta(np.concatenate(metrics_aux['native_off']['live']), live),
        }
    for threshold in THRESHOLDS:
        key = f'{threshold:.2f}'
        results['delta_full_minus_native'][key] = (results['metrics']['full_on'][key]['AP'] -
                                                  results['metrics']['native_off'][key]['AP'])
    input_frames = sum(row['input_frames'] for row in protocol['inventory'].values())
    keyframes = sum(row['keyframes'] for row in protocol['inventory'].values())
    for arm, rows in timing.items():
        loop = sum(r['loop_seconds'] for r in rows)
        wall = sum(r['process_wall_seconds'] for r in rows)
        results['timing'][arm] = {
            'input_frames': input_frames, 'processed_keyframes': keyframes,
            'loop_seconds': loop, 'summed_process_wall_seconds': wall,
            'input_fps_loop': input_frames/loop, 'input_fps_process_wall': input_frames/wall,
            'keyframes_per_second_process_wall': keyframes/wall,
            'scene_reported_fps_min': min(r['reported_input_fps'] for r in rows),
            'scene_reported_fps_median': float(np.median([r['reported_input_fps'] for r in rows])),
        }
    for path, digest in hashes.items():
        if sha256(path) != digest:
            raise ValueError(f'Input changed during evaluation: {path}')
    results['input_integrity_passed'] = True
    output.mkdir()
    write_json(output / 'results.json', results)
    write_json(output / 'coverage.json', coverage)
    write_json(output / 'scene_audit.json', audit)
    write_json(output / 'input_sha256.json', hashes)
    lines = ['# 在线动态分支：同75场三臂配对测试', '',
             f'原始GT {n_full}；移除GT {n_removed}；三臂统一终态GT {n_full-n_removed}。', '',
             '| 配置 | 框数 | AP15 | AP25 | AP50 | stale % |', '|---|---:|---:|---:|---:|---:|']
    for key in variants:
        m = results['metrics'][key]
        aps = ' | '.join(f'{m[f"{t:.2f}"]["AP"]:.4f}' for t in THRESHOLDS)
        lines.append(f'| {key} | {m["boxes"]} | {aps} | {results["auxiliary"][key]["stale_percent"]:.2f} |')
    lines += ['', 'full_on − native_off AP：' + ' / '.join(f'{v:+.4f}' for v in results['delta_full_minus_native'].values()), '',
              '| 配置 | 存活GT：丢失/新增覆盖 | 原生旧位：清除/新增残留 | loop input FPS | 含进程启动 input FPS |',
              '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        a, t = results['auxiliary'][arm], results['timing'][arm]
        l, s = a['live_vs_native'], a['stale_vs_native']
        lines.append(f'| {arm} | {l["lost"]}/{l["gained"]} | {s["lost"]}/{s["gained"]} | {t["input_fps_loop"]:.2f} | {t["input_fps_process_wall"]:.2f} |')
    lines += ['', '## 解释边界', '',
              '- 原生底座是真正disabled重跑；full_on persistent仅作诊断，不充当off消融。',
              '- no_miss_retirement关闭漏观测造成的降分及遮挡/coast/age休眠退役；并非只在末尾恢复score。会连带影响再激活与容量占用。',
              '- AP保持真实分数，无额外阈值或NMS。辅助stale/live覆盖固定score>0.30、IoU>0.25，不是AP的一部分。',
              '- 存活GT覆盖损失不是自动归因于误退役，还可能来自关联、几何或排序变化。',
              '- 每个场景三臂同GPU顺序运行；不同场景可并行。FPS按gap25输入流归一化，不等于每帧运行检测器；进程耗时含模型初始化。',
              '- 这是合成移除场景的在线推理/终态评测，不能单独证明真实连续运动、身份保持、逐帧动态AP或开放词汇语义有效。',
              '- 旧位减少但AP/存活物体召回下降时，不应声称新版整体有效。所有三臂与所有阈值均保留，无按结果调参。']
    with (output / 'REPORT.md').open('x') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'PAIRED_AP_COMPLETE {output}', flush=True)


if __name__ == '__main__':
    main()
