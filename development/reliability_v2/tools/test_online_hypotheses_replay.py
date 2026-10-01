#!/usr/bin/env python3
"""Fixed-prefix CA-1M feasibility replay; not an end-to-end active AP run.

Inference consumes only captured chronological associations/observations.
All predictions are written before GT is opened. Native counts/scores/order
are preserved. Geometry feedback into subsequent native association is frozen.
"""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from boxfusion.online_hypotheses import OnlineHypotheses, Settings, View, corners
from tools.true_fusion_audit_core import aabb_iou, class_agnostic_ap

SCENES = ('42446540', '42897501', '42897521', '42897538', '42897545',
          '42897552', '42897561', '42897599', '42897647', '42897688')
ARMS = ('native', 'online_rank', 'same_pool_score', 'single_alternative', 'frozen_rank')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def inherit_lineage(ids, groups, previous_owners):
    parents = [{previous_owners[s] for s in group if s in previous_owners} for group in groups]
    usages = Counter(parent for group in parents for parent in group)
    source_usages = Counter(s for group in groups for s in set(group))
    inherited = {}
    for key, ancestors, group in zip(ids, parents, groups):
        if len(ancestors) == 1 and all(source_usages[s] == 1 for s in group):
            parent = next(iter(ancestors))
            if usages[parent] == 1:
                inherited[key] = parent
    resets = sum(bool(ancestors) and key not in inherited for key, ancestors in zip(ids, parents))
    return inherited, resets


def replay_scene(directory, destination, cfg, calibration_path):
    started = time.perf_counter()
    manifest = json.loads((directory/'scene.json').read_text())
    assert manifest['completed'] and manifest['raw_prefix_immutable']
    with np.load(directory/manifest['raw_file'], allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files}
    calls = json.loads((directory/manifest['calls_file']).read_text())
    events = json.loads((directory/manifest['events_file']).read_text())
    K = np.loadtxt(calibration_path)
    # Actual images may be portrait despite landscape defaults in the YAML.
    with Image.open(calibration_path.parent/'depth/0.png') as image:
        W, H = image.size
    assert all(np.allclose(K, np.asarray(e['K'])[:3, :3], rtol=0, atol=3e-5)
               and H == e['H'] and W == e['W'] for e in events)
    banks = {name: {} for name in ('online_rank', 'single_alternative', 'frozen_rank')}
    configurations = {'online_rank': cfg, 'single_alternative': replace(cfg, slots=2),
                      'frozen_rank': replace(cfg, fit=False)}
    total_stats = {name: Counter() for name in banks}
    timings = {name: [] for name in banks}
    receipt = dict(calls=len(calls), current_observation_updates=0, prefix_checks=0,
                   identity_resets=0, identity_transfers=0, merge_or_split_resets=0,
                   exact_terminal_rows=0, unsafe_terminal_rows=0,
                   selected_rows={name: 0 for name in ARMS}, max_alternatives=0,
                   projection_parity_max_px=0.0)
    previous_frame = -1
    previous_owners = {}
    for call in calls:
        frame = int(call['frame_id'])
        assert frame > previous_frame
        previous_frame = frame
        post = call['post']
        ids = [int(x) for x in post['init_ids']]
        assert len(set(ids)) == len(ids)
        lists = call['post_fusion_lists']
        assert len(lists) == len(ids)
        # The retained native init_id is a representative observation, not a
        # permanent instance ID. Transfer state through CAUSAL association
        # lineage only; ambiguous merges/splits start fresh.
        inherited, resets = inherit_lineage(ids, lists, previous_owners)
        receipt['identity_transfers'] += sum(key != parent for key, parent in inherited.items())
        receipt['merge_or_split_resets'] += resets
        for name, bank in banks.items():
            retained = set(inherited.values())
            for key in set(bank)-retained:
                total_stats[name].update(bank[key].stats)
            banks[name] = {key: bank[parent] for key, parent in inherited.items()}
        source_usages = Counter(s for group in lists for s in set(group))
        previous_owners = {s: key for key, group in zip(ids, lists) for s in group
                           if source_usages[s] == 1}
        for row, track_id in enumerate(ids):
            source = np.asarray(sorted(set(lists[row])), dtype=int)
            assert np.all((source >= 0) & (source < len(raw['scores'])))
            assert np.all(raw['frame_ids'][source] <= frame)
            receipt['prefix_checks'] += len(source)
            current = source[raw['frame_ids'][source] == frame]
            view = candidate = None
            if len(current):
                # One independent frame, irrespective of same-frame duplicates.
                s = int(current[np.argmax(raw['scores'][current])])
                view = View.make(frame, raw['cam_poses'][s], K, raw['boxes2d'][s], H, W)
                candidate = (raw['boxes_xyzlhw'][s], raw['rotations'][s], raw['scores'][s], s)
                cam = raw['corners'][s] @ view.world_to_camera[:3, :3].T + view.world_to_camera[:3, 3]
                p = cam @ view.K.T
                if np.all(p[:, 2] > .05):
                    projected = np.clip(p[:, :2]/p[:, 2:], [0, 0], [W, H])
                    error = float(np.abs(projected - raw['projected_boxes'][s]).max())
                    receipt['projection_parity_max_px'] = max(receipt['projection_parity_max_px'], error)
                receipt['current_observation_updates'] += 1
            for name, bank in banks.items():
                if track_id not in bank:
                    bank[track_id] = OnlineHypotheses(configurations[name])
                    if name == 'online_rank':
                        receipt['identity_resets'] += 1
                t0 = time.perf_counter()
                bank[track_id].advance(frame, post['boxes_xyzlhw'][row],
                                      post['rotations'][row], post['scores'][row],
                                      view=view, candidate=candidate)
                timings[name].append(time.perf_counter()-t0)
                receipt['max_alternatives'] = max(receipt['max_alternatives'],
                                                  len(bank[track_id].hypotheses))
    # EOF only maps already computed causal states to unchanged native output.
    with np.load(directory/manifest['final_file'], allow_pickle=False) as z:
        final = {k: z[k] for k in z.files}
    output = {arm: final['corners'].copy() for arm in ARMS}
    pools = []
    for row, track_id in enumerate(final['init_ids']):
        pool = [final['corners'][row].copy()]
        state = banks['online_rank'].get(int(track_id))
        exact = (state is not None
                 and np.array_equal(state.native[0].astype(final['boxes_xyzlhw'].dtype), final['boxes_xyzlhw'][row])
                 and np.array_equal(state.native[1].astype(final['rotations'].dtype), final['rotations'][row]))
        receipt['exact_terminal_rows' if exact else 'unsafe_terminal_rows'] += 1
        if exact:
            for arm in ARMS[1:]:
                name = 'online_rank' if arm == 'same_pool_score' else arm
                rule = 'score' if arm == 'same_pool_score' else 'rank'
                state_arm = banks[name][int(track_id)]
                h = state_arm.select(rule)
                if h is not None:
                    output[arm][row] = corners(h.box, h.rotation)
                    receipt['selected_rows'][arm] += 1
            pool.extend(corners(h.box, h.rotation) for h in state.hypotheses)
        pools.append(np.asarray(pool))
    for name, bank in banks.items():
        for state in bank.values():
            total_stats[name].update(state.stats)
    destination.mkdir()
    np.savez_compressed(destination/'predictions.npz', scores=final['scores'],
                        **output, pool_corners=np.concatenate(pools),
                        pool_offsets=np.cumsum([0]+[len(p) for p in pools]))
    receipt['state_stats'] = {k: dict(v) for k, v in total_stats.items()}
    receipt['time'] = {k: {'total_s': float(sum(v)),
                           'mean_ms_per_track_call': float(np.mean(v)*1000),
                           'p95_ms_per_track_call': float(np.percentile(v, 95)*1000)}
                       for k, v in timings.items()}
    receipt['wall_s'] = time.perf_counter()-started
    receipt['predictions'] = len(final['scores'])
    assert receipt['projection_parity_max_px'] < .1, 'Coordinate/projection convention mismatch'
    assert receipt['max_alternatives'] <= cfg.slots-1
    write_json(destination/'receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, default=ROOT/'reports/ca1m_true_fusion_pilot_v2_20260908')
    parser.add_argument('--data-root', type=Path, default=Path('/extra/ZhaoX/boxfusion_ca1m'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = Settings()
    input_hashes = {}
    for scene in SCENES:
        calibration = args.data_root/scene/'K_depth.txt'
        input_hashes[str(calibration.resolve())] = sha(calibration)
        dimensions = args.data_root/scene/'depth/0.png'
        input_hashes[str(dimensions.resolve())] = sha(dimensions)
        for filename in ('scene.json', 'observations.npz', 'fusion_calls.json', 'fusion_events.json', 'final.npz'):
            path = args.capture/scene/filename
            input_hashes[str(path.resolve())] = sha(path)
    source_hashes = {str(p): sha(p) for p in (Path(__file__).resolve(), ROOT/'boxfusion/online_hypotheses.py',
                                             ROOT/'tools/true_fusion_audit_core.py')}
    protocol = dict(scenes=SCENES, settings=asdict(cfg), arms=ARMS,
                    input_sha256=input_hashes, source_sha256=source_hashes,
                    baseline='CuTR + Boxer + reliable Top-K3, threshold .15, gap20; no M1/M2/tier2',
                    rule='Freeze configuration before AP; no threshold search or best-scene selection.',
                    scope='Causal geometry replay with frozen native association and scores; not end-to-end active inference.',
                    go_rule='online_rank AP50 must beat native and same-pool score by >=0.2 points, with no AP15/AP25 loss; descriptive pilot only.')
    write_json(args.output/'protocol.json', protocol)
    receipts = {}
    # All model predictions complete before ground truth is read.
    for scene in SCENES:
        receipts[scene] = replay_scene(args.capture/scene, args.output/scene, cfg,
                                       args.data_root/scene/'K_depth.txt')
        print(json.dumps({'scene': scene, 'selected': receipts[scene]['selected_rows'],
                          'seconds': round(receipts[scene]['wall_s'], 2)}), flush=True)
    write_json(args.output/'inference_complete.json', {'completed': True, 'ground_truth_used': False})
    gt, predictions = {}, {arm: {} for arm in (*ARMS, 'pool_oracle')}
    per_scene, improvements = {}, []
    for scene in SCENES:
        gt_path = args.data_root/scene/'after_filter_boxes.npy'
        input_hashes[str(gt_path.resolve())] = sha(gt_path)
        gt[scene] = np.load(gt_path, allow_pickle=False)
        with np.load(args.output/scene/'predictions.npz', allow_pickle=False) as z:
            scores = z['scores'].copy()
            for arm in ARMS:
                predictions[arm][scene] = (z[arm].copy(), scores.copy())
            oracle = z['native'].copy()
            offsets, pool = z['pool_offsets'], z['pool_corners']
            for row in range(len(scores)):
                candidates = pool[offsets[row]:offsets[row+1]]
                matrix = aabb_iou(candidates, gt[scene])
                if matrix.shape[1]:
                    best = int(matrix.max(1).argmax())
                    oracle[row] = candidates[best]
                    improvements.append(float(matrix[best].max()-matrix[0].max()))
            predictions['pool_oracle'][scene] = (oracle, scores.copy())
        per_scene[scene] = {arm: {str(t): class_agnostic_ap({scene: data[scene]}, {scene: gt[scene]}, t)['ap']
                                 for t in (.15, .25, .5)} for arm, data in predictions.items()}
    metrics = {arm: {str(t): class_agnostic_ap(data, gt, t) for t in (.15, .25, .5)}
               for arm, data in predictions.items()}
    baseline = metrics['native']
    deltas = {arm: {t: row['ap']-baseline[t]['ap'] for t, row in values.items()}
              for arm, values in metrics.items()}
    # Check the identical native replay anchor, not a different full107 stack.
    prior = json.loads((args.capture/'refusion_analysis.json').read_text())['sample_native_AP']
    parity = max(abs(baseline[str(float(t))]['ap']-value['ap']) for t, value in prior.items())
    assert parity < 1e-6, f'Native AP mismatch: {parity}'
    passed = (deltas['online_rank']['0.5'] >= .2
              and metrics['online_rank']['0.5']['ap']-metrics['same_pool_score']['0.5']['ap'] >= .2
              and all(deltas['online_rank'][t] >= 0 for t in ('0.15', '0.25')))
    results = dict(completed=True, scenes=list(SCENES), metrics=metrics, delta_AP=deltas,
                   per_scene_AP=per_scene, receipts=receipts, native_AP_parity_error=parity,
                   rows_with_better_pool_IoU=sum(x > 1e-6 for x in improvements),
                   decision='expand_only_after_active_validation' if passed else 'stop_this_configuration',
                   input_sha256=input_hashes, source_sha256=source_hashes,
                   limitations=[
                       'No feedback from selected geometry to native association, births, PFO or scores.',
                       'Native representative changes inherit state via unique causal lineage; ambiguous merges/splits/shared-source rows reset it.',
                       'Only exact terminal native-parameter links are patched; no retrospective group reconstruction.',
                       'Fitted trajectories are checked before each update, not the exact newly fitted box.',
                       'No depth/mask evidence, learned ranker, or full107 M1/M2 stack evaluated.',
                       'Pool oracle uses GT only after inference; best-IoU per row is not a global AP upper bound.',
                       'CPU component timing excludes detector/native association/PFO and is not end-to-end FPS.',
                       'Ten existing development scenes; no significance or held-out generalization claim.'])
    for path, expected in input_hashes.items():
        assert sha(path) == expected, f'Input mutated: {path}'
    write_json(args.output/'results.json', results)
    lines = ['# 在线多假设融合：固定前 10 场可行性回放', '',
             '**结论：' + ('通过预设小样本门槛，仍需 active 验证。' if passed else '当前配置未通过预设门槛，不扩大运行。') + '**', '',
             '底座为 CuTR + Boxer + Reliable Top-K3，gap20、阈值0.15；不是完整 M1/M2 栈，也不是全107场。', '',
             '最多3个几何槽（原生框+2个备用），每轨迹最多3帧。新帧先比较旧假设与上一原生框，再独立局部拟合；至少2次未来帧检验、平均IoU优势>0.02且最新检验为正才替换。',
             '拟合固定种子旋转，中心和尺寸在种子信赖域内优化。各假设持续存在、分别更新，允许选择回退。仅用当前帧检测矩形验证，不读取GT。', '',
             '| 配置 | AP15 | AP25 | AP50 | ΔAP50 |', '|---|---:|---:|---:|---:|']
    for arm in metrics:
        values = [metrics[arm][t]['ap'] for t in ('0.15', '0.25', '0.5')]
        lines.append(f'| {arm} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | {deltas[arm]["0.5"]:+.4f} |')
    lines += ['', 'online_rank 与 same_pool_score 使用完全相同的在线候选池；single_alternative 限为一个备用；frozen_rank 不更新候选几何、槽数与证据门槛相同。所有臂保持原生输出框数、分数、顺序。',
              '预设扩大门槛：AP50较原生及同池分数各提升至少0.2点，AP15/AP25不下降；并非统计显著性标准。', '',
              f'原生AP复现误差：{parity:.3g}。池中有更高GT IoU的终端行：{results["rows_with_better_pool_IoU"]}。', '',
              '## 边界', *['- '+x for x in results['limitations']], '',
              '## 复现', '```bash', f'python tools/test_online_hypotheses_replay.py --output /tmp/online_hypotheses_replay_new', '```',
              '完整参数、逐场结果、计时、输入及源码哈希见 protocol.json / results.json。']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'decision': results['decision'], 'delta_AP': deltas}), flush=True)


if __name__ == '__main__':
    main()
