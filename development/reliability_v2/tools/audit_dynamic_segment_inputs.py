"""Read-only input gate for the proposed motion-segment fusion diagnostic."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data_dyn_move'
OUT = ROOT / 'reports/dynamic_segment_diagnosis_20260912'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distances(box):
    a, b = np.triu_indices(8, 1)
    return np.sort(np.linalg.norm(box[a] - box[b], axis=1))


def main():
    moved = json.loads((DATA / 'manifest.json').read_text())
    removal = json.loads((ROOT / 'data_dyn/manifest.json').read_text())
    rows = []
    for scene, entry in sorted(moved.items()):
        assert entry is not None
        a = np.asarray(entry['A']).reshape(8, 3)
        b = np.asarray(entry['B']).reshape(8, 3)
        frames = DATA / scene / 'frames'
        after = sorted((p for p in (frames / 'color').glob('*.jpg')
                        if int(p.stem) >= entry['T_frame'] and int(p.stem) % 25 == 0),
                       key=lambda p: int(p.stem))
        original, edited, depth_delta = 0, 0, []
        for color in after:
            depth = frames / 'depth' / (color.stem + '.png')
            if color.is_symlink() and depth.is_symlink():
                original += 1
                continue
            edited += 1
            pose = np.loadtxt(frames / 'pose' / (color.stem + '.txt'))
            if np.isfinite(pose).all():
                world_to_camera = np.linalg.inv(pose)
                depth_delta.append(abs(float((world_to_camera[:3, :3]
                                              @ (b.mean(0) - a.mean(0)))[2])))
        old = removal.get(scene, {}).get('removed', [])
        existing_b = len(old) > 1 and np.allclose(b, np.asarray(old[1]).reshape(8, 3))
        rows.append(dict(scene=scene, post_event_keyframes=len(after),
                         unchanged_post_event_keyframes=original,
                         edited_post_event_keyframes=edited,
                         b_equals_second_original_gt=bool(existing_b),
                         pairwise_distance_max_difference_m=float(np.max(abs(distances(a)-distances(b)))),
                         translation_depth_offsets_m=depth_delta,
                         cached_base_files=sorted(p.name for p in
                             (ROOT / 'results/scannet_move_base').glob(scene + '*'))))
    offsets = [v for r in rows for v in r['translation_depth_offsets_m']]
    summary = dict(
        scenes=len(rows),
        b_is_original_object_scenes=sum(r['b_equals_second_original_gt'] for r in rows),
        noncongruent_ab_boxes_over_1cm=sum(r['pairwise_distance_max_difference_m'] > .01 for r in rows),
        post_event_keyframes=sum(r['post_event_keyframes'] for r in rows),
        unchanged_post_event_keyframes=sum(r['unchanged_post_event_keyframes'] for r in rows),
        edited_post_event_keyframes=sum(r['edited_post_event_keyframes'] for r in rows),
        scenes_with_unchanged_post_event_frames=sum(r['unchanged_post_event_keyframes'] > 0 for r in rows),
        valid_edited_poses=len(offsets),
        median_required_center_depth_translation_m=float(np.median(offsets)),
        edited_poses_requiring_over_20cm_depth_translation=sum(v > .2 for v in offsets),
    )
    paths = [DATA / 'manifest.json', ROOT / 'data_dyn/manifest.json', Path('/tmp/build_move.py')]
    result = dict(status='input_gate_failed', ap_experiment_run=False,
                  summary=summary, scenes=rows,
                  source_sha256={str(p): sha(p) for p in paths if p.exists()},
                  limits=[
                      'Depth-offset calculation diagnoses a proposed rigid translation; it is not a measured depth reconstruction error.',
                      'Unchanged symlinks after the event are not themselves false detections; they invalidate a globally persistent moved-object GT assumption.',
                      'B is a pre-existing GT object location. A detection at B alone does not establish relocation or recovered identity.',
                      'Source script copies and resizes source depth without translation correction; B box dimensions also come from a different object.',
                      'The inspected move baseline caches contain terminal boxes/logs, not a complete per-frame PFO replay dataset.',
                      'This diagnostic rejects these inputs, not the segmented-fusion hypothesis.',
                  ])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    s = summary
    report = f'''# 运动分段融合：输入诊断

状态：输入门不通过，未运行 AP 对比；不据此判定方法有效或无效。

- 现有搬移集：{s['scenes']} 场；B 位置在 {s['b_is_original_object_scenes']} 场均对应原场景第二个 GT 物体。
- A/B 框不满足同一刚体尺寸形状（角点两两距离最大差 >1cm）：{s['noncongruent_ab_boxes_over_1cm']} 场。
- 搬移后 gap25 关键帧：{s['post_event_keyframes']}；其中 {s['unchanged_post_event_keyframes']} 帧的 RGB/深度仍链接原始未搬移输入，涉及 {s['scenes_with_unchanged_post_event_frames']} 场。
- 已编辑帧有效位姿：{s['valid_edited_poses']}。若将 A 刚体平移到 B 中心，所需相机深度平移绝对值中位数为 {s['median_required_center_depth_translation_m']:.3f}m，其中 {s['edited_poses_requiring_over_20cm_depth_translation']} 帧超过 0.2m。生成脚本只缩放复制原深度，没有该修正；此数不是实测重建误差。
- 生成脚本还将 A 的图像缩放到另一物体 B 的投影矩形；因此不能直接将 B 框当作搬移后同一物体的准确三维 GT。
- 所查原生搬移结果为终图 PKL/日志，缺完整逐帧融合输入。纯移除 75 场不能替代搬移后恢复实验。

结论：现有“B 位 63/63 有框”只能证明该位置被检测，不能证明搬移成功、同一身份重捕获或几何正确。当前数据不适合用于“理想身份/分段下，短窗口 vs 分段 PFO”的几何优劣判断。

最低后续条件：一组几何一致、具有时序三维 GT 的搬移—再静止序列，并缓存相同逐帧候选和 PFO 所需观测；之后再做同框数同分数比较。此次只执行输入审计，没有训练、下载模型、改生产代码或启动全量推理。

复现：`python tools/audit_dynamic_segment_inputs.py`。逐场统计与输入哈希见 `results.json`。
'''
    (OUT / 'REPORT.md').write_text(report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
