"""Scale the frozen Step 1c protocol to a specified CA-1M scene list.

prepare checks inputs and freezes the protocol; worker performs GT-free
capture/selection/lifting/birth construction; evaluate requires every scene
complete and computes pooled AP with all births, including false positives.
Two workers may process disjoint scene-list slices on separate GPUs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_ca1m_seedless_step1c as pilot
from tools.audit_ca1m_nms_child_headroom import sha256, valid_boxes
from tools.audit_m1m2_remaining_children import SCENES as OFFICIAL_SCENES, verify_metric
from tools.ca1m_seedless_trigger_core import add_clusters
from tools.true_fusion_audit_core import aabb_iou, class_agnostic_ap
from tools.validate_ca1m_prenms_query import DenseCapture, forbid_gt

PILOT_OUTPUT = ROOT / 'reports/ca1m_seedless_step1c_20260911_fixed'
PILOT_RAW = pilot.PILOT
VARIANTS = pilot.VARIANTS


def check_hashes(hashes):
    for path, expected in hashes.items():
        if sha256(path) != expected:
            raise ValueError(f'Input changed: {path}')


def read_protocol(output):
    protocol = json.loads((output / 'protocol.json').read_text())
    check_hashes(protocol['source_sha256'])
    return protocol


def prepare(args):
    scenes = args.scenes.read_text().split()
    if len(scenes) != args.expected_scenes or len(scenes) != len(set(scenes)):
        raise ValueError('Unexpected or duplicate scene list')
    if not all(s.isdecimal() for s in scenes):
        raise ValueError('Expected numeric CA-1M scene IDs')
    frames = {}
    for scene in scenes:
        root = pilot.DATA_ROOT / scene
        for p in [root / 'K_rgb.txt', root / 'K_depth.txt',
                  pilot.BASE / f'{scene}_boxes.pkl', pilot.GT_ROOT / scene / 'after_filter_boxes.npy']:
            if not p.is_file():
                raise FileNotFoundError(p)
        poses = np.load(root / 'all_poses.npy', allow_pickle=False)
        frames[scene] = list(range(0, len(poses), 20))
        for frame in frames[scene]:
            if not np.isfinite(poses[frame]).all():
                raise ValueError(f'Invalid pose: {scene}/{frame}')
            for kind in ['rgb', 'depth']:
                path = root / kind / f'{frame}.png'
                if not path.is_file():
                    raise FileNotFoundError(path)
    paths = [Path(__file__), args.scenes, Path(pilot.__file__),
             ROOT / 'tools/ca1m_seedless_trigger_core.py',
             ROOT / 'tools/validate_ca1m_prenms_query.py',
             ROOT / 'tools/ca1m_prenms_query_core.py',
             ROOT / 'tools/true_fusion_audit_core.py',
             ROOT / 'tools/audit_m1m2_remaining_children.py',
             ROOT / 'tools/audit_ca1m_nms_child_headroom.py',
             ROOT / 'boxfusion/boxer_lifter.py',
             ROOT / 'third_party/WeDetect/wedetect_uni_infer.py',
             ROOT / 'third_party/WeDetect/wedetect_base_uni.pth']
    hashes = {str(p.resolve()): sha256(p) for p in paths}
    metric = verify_metric()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'raw').mkdir()
    (args.output / 'scenes').mkdir()
    (args.output / 'scenes.txt').write_text('\n'.join(scenes) + '\n')
    pilot.write_json(args.output / 'protocol.json', {
        'schema': 'boxfusion.seedless_full.v1', 'scenes': scenes, 'frames': frames,
        'scene_count': len(scenes), 'frame_count': sum(map(len, frames.values())),
        'variants': list(VARIANTS), 'gap': 20, 'source_sha256': hashes,
        'parameters': {'residual_iou': .2, 'radius_rgb_px': 30., 'other_frames': 2,
                       'budgets': [300, 150], 'cluster_voxel_m': .3,
                       'cluster_distinct_frames': 3, 'append_score': .1},
        'baseline': str(pilot.BASE), 'dataset': str(pilot.DATA_ROOT),
        'birth_selection': 'all >=3-frame voxels, highest raw-score representative; no GT filter',
        'production_outputs_modified': False, 'worker_gt_access': False,
        'offline_final_map_masks': True, 'all_frame_recurrence': True,
        'metric_check': metric, 'prepared_unix': time.time(),
    })
    print(f'Prepared {len(scenes)} scenes / {sum(map(len, frames.values()))} frames', flush=True)


def build_births(directory, selection, inputs):
    """Construct complete candidate births from frozen IDs/geometry without GT."""
    clusters = {v: {} for v in VARIANTS}
    for row in selection['frames']:
        frame = row['frame']
        lift_path = directory / f'lifted_{frame:06d}.npz'
        raw_path = directory.parents[1] / 'raw' / selection['scene'] / f'raw_{frame:06d}.npz'
        with np.load(lift_path, allow_pickle=False) as values:
            union, corners = values['anchor_ids'], values['corners']
        with np.load(raw_path, allow_pickle=False) as values:
            scores = values['scores']
        for v in VARIANTS:
            ids = np.asarray(row['selected_anchor_ids'][v], dtype=np.int64)
            positions = np.searchsorted(union, ids)
            if not np.array_equal(union[positions], ids):
                raise ValueError('Lifting anchor IDs do not match the frozen selection')
            add_clusters(clusters[v], frame, ids, corners[positions], scores[ids], .3)
        for path in [lift_path, raw_path]:
            inputs[str(path.resolve())] = sha256(path)
    summary = {}
    for v in VARIANTS:
        confirmed = [(key, c) for key, c in sorted(clusters[v].items()) if len(c['frames']) >= 3]
        boxes = np.asarray([c['box'] for _, c in confirmed], dtype=float).reshape(-1, 8, 3)
        path = directory / f'births_{v}.npz'
        np.savez_compressed(path, corners=boxes,
                            voxel_keys=np.asarray([key for key, _ in confirmed], dtype=int).reshape(-1, 3),
                            distinct_frames=np.asarray([len(c['frames']) for _, c in confirmed], dtype=int))
        inputs[str(path.resolve())] = sha256(path)
        summary[v] = {'all_voxels': len(clusters[v]), 'births': len(confirmed),
                      'selected_observations': sum(c['observations'] for c in clusters[v].values())}
    return summary


def worker(args):
    protocol = read_protocol(args.output)
    if not 0 <= args.worker_index < args.workers:
        raise ValueError('Invalid worker partition')
    scenes = protocol['scenes'][args.worker_index::args.workers]
    # No worker is allowed to open GT, including in third-party inference code.
    sys.addaudithook(forbid_gt)
    pilot.PILOT = args.output / 'raw'
    detector, adapter = None, None
    start = time.perf_counter()
    for ordinal, scene in enumerate(scenes, 1):
        directory = args.output / 'scenes' / scene
        complete = directory / 'complete.json'
        if complete.exists():
            saved = json.loads(complete.read_text())
            check_hashes(saved['sha256'])
            print(f'Worker {args.worker_index}: verified existing {scene}', flush=True)
            continue
        if directory.exists() or (pilot.PILOT / scene).exists():
            raise ValueError(f'Partial scene needs explicit recovery before reuse: {scene}')
        scene_start = time.perf_counter()
        frames = protocol['frames'][scene]
        rawdir = pilot.PILOT / scene
        rawdir.mkdir()
        reused_pilot = scene in pilot.SCENES
        capture_ms = 0.
        if reused_pilot:
            for frame in frames:
                source = PILOT_RAW / scene / f'raw_{frame:06d}.npz'
                shutil.copyfile(source, rawdir / source.name)
        else:
            if detector is None:
                detector = DenseCapture()
                # Confirm regeneration uses the exact same raw boxes/scores as the pilot.
                image, _, _ = pilot.load_frame(pilot.DATA_ROOT / pilot.SCENES[0], 0)
                probe = detector.forward(image)
                with np.load(PILOT_RAW / pilot.SCENES[0] / 'raw_000000.npz') as reference:
                    parity = {key: float(np.max(np.abs(probe[key] - reference[key])))
                              for key in ('boxes', 'scores')}
                    if not (np.allclose(probe['boxes'], reference['boxes'], atol=1e-4, rtol=0)
                            and np.allclose(probe['scores'], reference['scores'], atol=1e-10, rtol=1e-5)):
                        raise ValueError(f'Dense capture differs from pilot: {parity}')
                pilot.write_json(args.output / f'worker_{args.worker_index}_capture_parity.json', parity)
            for frame in frames:
                image, _, _ = pilot.load_frame(pilot.DATA_ROOT / scene, frame)
                value = detector.forward(image)
                if value['boxes'].shape != (8400, 4) or value['scores'].shape != (8400,):
                    raise ValueError(f'Unexpected dense pool: {scene}/{frame}')
                np.savez_compressed(rawdir / f'raw_{frame:06d}.npz',
                                    boxes=value['boxes'], scores=value['scores'])
                capture_ms += value['forward_ms'] + value['recovery_ms']
        inputs = {}
        state = pilot.select_scene(scene, args.output / 'scenes', inputs)
        if state['frames'] != frames:
            raise ValueError('Frame list differs from frozen protocol')
        if reused_pilot:
            old = json.loads((PILOT_OUTPUT / scene / 'selection.json').read_text())
            new = json.loads((directory / 'selection.json').read_text())
            if old != new:
                raise ValueError(f'Selection changed from three-scene run: {scene}')
            for frame in frames:
                source = PILOT_OUTPUT / scene / f'lifted_{frame:06d}.npz'
                shutil.copyfile(source, directory / source.name)
        else:
            if adapter is None:
                adapter = pilot.build_lifter(args.output / f'worker_{args.worker_index}')
            for frame in frames:
                union = np.unique(np.concatenate(list(state['plans'][frame].values())))
                _, rgb, depth = pilot.load_frame(pilot.DATA_ROOT / scene, frame)
                corners, _, _ = pilot.lift(adapter, scene, frame, rgb, depth,
                                           state['k_rgb'], state['k_depth'], state['poses'][frame],
                                           state['anchors'][frame][union])
                corners = valid_boxes(corners, f'{scene}/{frame}')
                if len(corners) != len(union):
                    raise ValueError('Lifting changed candidate cardinality')
                np.savez_compressed(directory / f'lifted_{frame:06d}.npz', anchor_ids=union, corners=corners)
        selection = json.loads((directory / 'selection.json').read_text())
        birth_stats = build_births(directory, selection, inputs)
        check_hashes(inputs)
        pilot.write_json(complete, {
            'completed': True, 'scene': scene, 'frames': len(frames), 'gt_access': False,
            'worker_index': args.worker_index, 'reused_pilot': reused_pilot,
            'capture_ms': capture_ms, 'seconds': time.perf_counter() - scene_start,
            'birth_stats': birth_stats, 'sha256': inputs,
        })
        print(f'Worker {args.worker_index} [{ordinal}/{len(scenes)}] {scene}: '
              f'frames={len(frames)}, births={ {v: birth_stats[v]["births"] for v in VARIANTS} }, '
              f'seconds={time.perf_counter()-scene_start:.1f}', flush=True)
    check_hashes(protocol['source_sha256'])
    pilot.write_json(args.output / f'worker_{args.worker_index}_complete.json', {
        'completed': True, 'scenes': scenes, 'gt_access': False,
        'seconds': time.perf_counter() - start,
    })


def evaluate(args):
    protocol = read_protocol(args.output)
    baseline, gts = {}, {}
    arms = {v: {} for v in VARIANTS}
    counts = {v: Counter() for v in VARIANTS}
    inputs, manifests, stats = {}, [], []
    predroot = args.output / 'predictions'
    predroot.mkdir(exist_ok=False)
    for v in ('baseline',) + VARIANTS:
        (predroot / v).mkdir()
    for scene in protocol['scenes']:
        directory = args.output / 'scenes' / scene
        completed = json.loads((directory / 'complete.json').read_text())
        if not completed['completed'] or completed['frames'] != len(protocol['frames'][scene]):
            raise ValueError(f'Incomplete scene: {scene}')
        check_hashes(completed['sha256'])
        inputs.update(completed['sha256'])
        manifests.append(completed)
        selection = json.loads((directory / 'selection.json').read_text())
        stats.extend(selection['frames'])
        path = pilot.GT_ROOT / scene / 'after_filter_boxes.npy'
        inputs[str(path.resolve())] = sha256(path)
        gt = valid_boxes(np.load(path, allow_pickle=False), str(path))
        boxes, scores = pilot.read_prediction(pilot.BASE / f'{scene}_boxes.pkl')
        baseline[scene], gts[scene] = (boxes, scores), gt
        scene_predictions = {'baseline': (boxes, scores)}
        for v in VARIANTS:
            with np.load(directory / f'births_{v}.npz', allow_pickle=False) as data:
                births = valid_boxes(data['corners'], v)
            arms[v][scene] = pilot.append_boxes(boxes, scores, births)
            scene_predictions[v] = arms[v][scene]
            counts[v].update(completed['birth_stats'][v])
            matrix = aabb_iou(births, gt)
            counts[v]['births_unmatched_gt15'] += int((matrix.max(1) <= .15).sum()) if len(gt) else len(births)
        for v, (corners, confidence) in scene_predictions.items():
            payload = [[(0, box, float(score)) for box, score in zip(corners, confidence)]]
            path = predroot / v / f'{scene}_boxes.pkl'
            with path.open('xb') as handle:
                pickle.dump(payload, handle)
            inputs[str(path.resolve())] = sha256(path)
    metric_check = verify_metric()
    metrics = {'baseline': {str(t): class_agnostic_ap(baseline, gts, t) for t in pilot.THRESHOLDS}}
    for v in VARIANTS:
        metrics[v] = {}
        for t in pilot.THRESHOLDS:
            value = class_agnostic_ap(arms[v], gts, t)
            for key in ('ap', 'tp', 'fp'):
                value['delta_' + key] = value[key] - metrics['baseline'][str(t)][key]
            metrics[v][str(t)] = value
    if protocol['scenes'] == OFFICIAL_SCENES.read_text().split():
        expected = [46.1315, 38.3400, 16.8523]
        actual = [metrics['baseline'][str(t)]['ap'] for t in pilot.THRESHOLDS]
        if not np.allclose(actual, expected, atol=5e-4, rtol=0):
            raise ValueError(f'Full107 baseline parity failed: {actual}')
    # Reproduce the three-scene matched comparison from the pooled predictions.
    pilot_parity = {}
    if set(pilot.SCENES) <= set(protocol['scenes']):
        reference = json.loads((PILOT_OUTPUT / 'results.json').read_text())
        for v in VARIANTS:
            pilot_parity[v] = []
            for t in pilot.THRESHOLDS:
                value = class_agnostic_ap({s: arms[v][s] for s in pilot.SCENES},
                                          {s: gts[s] for s in pilot.SCENES}, t)['ap']
                expected = reference['metrics'][v + '_all_clusters'][str(t)]['ap']
                pilot_parity[v].append(abs(value - expected))
                if abs(value - expected) > 1e-8:
                    raise ValueError('Three-scene birth output no longer matches the frozen pilot')
    check_hashes(inputs)
    result = {
        'schema': 'boxfusion.seedless_full_ap.v1', 'completed': True,
        'scenes': protocol['scenes'], 'scene_count': len(manifests), 'frame_count': len(stats),
        'metrics': metrics, 'birth_stats': {v: dict(counts[v]) for v in VARIANTS},
        'mean_budget': {v: float(np.mean([r['selected'][v] for r in stats])) for v in VARIANTS},
        'mean_residual': float(np.mean([r['residual'] for r in stats])),
        'mean_recurrent': float(np.mean([r['recurrent'] for r in stats])),
        'metric_check': metric_check, 'pilot_ap_parity_errors': pilot_parity,
        'limits': ['GT used only in evaluation, all confirmed births included in main AP.',
                   'Fixed final-map masks and all-frame recurrence are offline; no causal/FPS claim.',
                   'The official validation scenes have been used in development; not an untouched test set.',
                   'Voxel births may fragment objects or produce duplicates; no extra NMS or tuning.'],
    }
    pilot.write_json(args.output / 'results.json', result)
    pilot.write_json(args.output / 'input_sha256.json', inputs)
    print(json.dumps({'metrics': metrics, 'birth_stats': result['birth_stats']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'worker', 'evaluate'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenes', type=Path, default=OFFICIAL_SCENES)
    parser.add_argument('--expected-scenes', type=int, default=107)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--worker-index', type=int, default=0)
    args = parser.parse_args()
    args.output = args.output.resolve()
    {'prepare': prepare, 'worker': worker, 'evaluate': evaluate}[args.stage](args)


if __name__ == '__main__':
    main()
