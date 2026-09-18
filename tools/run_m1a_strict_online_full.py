#!/usr/bin/env python3
"""Run strict-online M1-A from frozen causal detector outputs.

Each keyframe reads only that frame's raw detector anchors, lifts its score
top-300 as a single Boxer query set, and immediately updates the bounded
``OnlineAnchorRecovery`` state.  It never reads future selections, final-map
masks, terminal cluster counts, or GT.  Per-scene outputs are resumable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boxfusion.m1_anchor_online import OnlineAnchorRecovery
from tools.audit_ca1m_nms_child_headroom import sha256
from tools.audit_m1m2_remaining_children import read_prediction
from tools.audit_seedless_ablation_matrix import BASELINES, RUNS
from tools.validate_ca1m_prenms_query import build_lifter, lift


TOP_M = 300
MAX_ACTIVE = 4096
MAX_BIRTHS = 640


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def frame_inputs(dataset, root, scene, frame, poses, k_rgb, k_depth):
    if dataset == 'scannet':
        color = root / scene / 'color' / f'{frame}.jpg'
        depth_path = root / scene / 'depth' / f'{frame}.png'
        pose = np.loadtxt(root / scene / 'pose' / f'{frame}.txt').reshape(4, 4)
    else:
        color = root / scene / 'rgb' / f'{frame}.png'
        depth_path = root / scene / 'depth' / f'{frame}.png'
        pose = poses[frame]
    with Image.open(color) as image:
        pil = image.convert('RGB')
        rgb = np.asarray(pil)
    with Image.open(depth_path) as image:
        depth = np.asarray(image).astype(np.float32) / 1000.0
    return pil, rgb, depth, pose, k_rgb, k_depth


def scene_calibration(dataset, root, scene):
    base = root / scene
    if dataset == 'scannet':
        k_rgb = np.loadtxt(base / 'intrinsic/intrinsic_color.txt')[:3, :3]
        k_depth = np.loadtxt(base / 'intrinsic/intrinsic_depth.txt')[:3, :3]
        poses = None
    else:
        k_rgb = np.loadtxt(base / 'K_rgb.txt').reshape(3, 3)
        k_depth = np.loadtxt(base / 'K_depth.txt').reshape(3, 3)
        poses = np.load(base / 'all_poses.npy', allow_pickle=False)
    return poses, k_rgb, k_depth


def process_scene(dataset, run, protocol, output, scene, adapter):
    output_pkl = output / 'predictions' / f'{scene}_boxes.pkl'
    trace_path = output / 'traces' / f'{scene}.json'
    if output_pkl.is_file() and trace_path.is_file():
        trace = json.loads(trace_path.read_text())
        if trace.get('completed') is True:
            return trace
        raise RuntimeError(f'partial scene: {scene}')
    root = Path(protocol.get('data') or protocol.get('dataset'))
    poses, k_rgb, k_depth = scene_calibration(dataset, root, scene)
    tracker = OnlineAnchorRecovery(
        scene, voxel_m=.3, min_views=3,
        max_active=MAX_ACTIVE, max_births=MAX_BIRTHS)
    scene_started = time.perf_counter()
    lift_seconds = 0.0
    state_seconds = 0.0
    observations = 0
    invalid_lifts = 0
    frame_rows = []
    for ordinal, frame in enumerate(protocol['frames'][scene]):
        raw_path = run / 'raw' / scene / f'raw_{frame:06d}.npz'
        with np.load(raw_path, allow_pickle=False) as values:
            raw_boxes = values['boxes']
            raw_scores = values['scores']
        ids = np.argsort(-raw_scores, kind='stable')[:TOP_M]
        _, rgb, depth, pose, kr, kd = frame_inputs(
            dataset, root, scene, frame, poses, k_rgb, k_depth)
        corners, lift_ms, _ = lift(
            adapter, scene, frame, rgb, depth, kr, kd, pose, raw_boxes[ids])
        valid = (np.isfinite(corners).all(axis=(1, 2))
                 & (np.ptp(corners, axis=1) > 0).all(1))
        invalid_lifts += int((~valid).sum())
        started = time.perf_counter()
        events = tracker.update(ordinal, frame, ids[valid], corners[valid],
                                raw_scores[ids][valid])
        state_seconds += time.perf_counter() - started
        lift_seconds += lift_ms / 1000.0
        observations += int(valid.sum())
        frame_rows.append({'ordinal': ordinal, 'frame_id': int(frame),
                           'valid_observations': int(valid.sum()),
                           'births_now': len(events),
                           'active_after': len(tracker.active),
                           'births_after': len(tracker.births)})

    rows = tracker.rows()
    scores = [float(r['score']) for r in rows]
    if len(scores) != len(set(scores)) or (scores and max(scores) >= .05):
        raise RuntimeError(f'invalid deterministic scores: {scene}')
    base_path = BASELINES[dataset] / f'{scene}_boxes.pkl'
    payload = pickle.load(open(base_path, 'rb'))
    base_boxes, base_scores = read_prediction(base_path)
    payload[0] = ([tuple(r) for r in payload[0]]
                  + [(0, r['box'], r['score']) for r in rows])
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open('xb') as handle:
        pickle.dump(payload, handle)
    trace = {
        'schema': 'boxfusion.m1a.strict_online.full.v1',
        'completed': True,
        'dataset': dataset,
        'scene_id': scene,
        'strictly_causal': True,
        'uses_future_frames': False,
        'uses_final_map_mask': False,
        'uses_gt': False,
        'single_pool_lift': True,
        'parameters': {'top_m': TOP_M, 'voxel_m': .3,
                       'confirmation_distinct_frames': 3,
                       'max_active': MAX_ACTIVE, 'max_births': MAX_BIRTHS,
                       'score_interval': [0.040001, 0.049999]},
        'baseline_path': str(base_path),
        'baseline_sha256': sha256(base_path),
        'baseline_rows': len(base_boxes),
        'birth_rows': len(rows),
        'output_rows': len(payload[0]),
        'score_unique': True,
        'invalid_lifts': invalid_lifts,
        'observations': observations,
        'lift_seconds': lift_seconds,
        'state_update_seconds': state_seconds,
        'wall_seconds': time.perf_counter() - scene_started,
        'tracker': tracker.diagnostics(),
        'frames': frame_rows,
        'birth_events': tracker.events,
    }
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(trace_path, trace)
    return trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('scannet', 'ca1m'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit-scenes', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--worker-index', type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1 or not 0 <= args.worker_index < args.workers:
        parser.error('require workers >= 1 and 0 <= worker-index < workers')
    run = RUNS[args.dataset]
    protocol = json.loads((run / 'protocol.json').read_text())
    all_scenes = protocol['scenes'][:args.limit_scenes or None]
    scenes = all_scenes[args.worker_index::args.workers]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest['dataset'] != args.dataset
                or manifest['scenes'] != all_scenes
                or manifest['parameters'] != {
                    'top_m': TOP_M, 'voxel_m': .3,
                    'confirmation_distinct_frames': 3,
                    'max_active': MAX_ACTIVE, 'max_births': MAX_BIRTHS}):
            raise RuntimeError('existing manifest does not match requested run')
    else:
        manifest = {
            'schema': 'boxfusion.m1a.strict_online.manifest.v1',
            'dataset': args.dataset, 'scenes': all_scenes,
            'source_run': str(run),
            'source_protocol_sha256': sha256(run / 'protocol.json'),
            'baseline': str(BASELINES[args.dataset]),
            'parameters': {'top_m': TOP_M, 'voxel_m': .3,
                           'confirmation_distinct_frames': 3,
                           'max_active': MAX_ACTIVE,
                           'max_births': MAX_BIRTHS},
            'gt_access': False,
        }
        write_json(manifest_path, manifest)
    worker_dir = args.output / 'workers' / f'{args.worker_index:02d}_of_{args.workers:02d}'
    worker_dir.mkdir(parents=True, exist_ok=True)
    adapter = build_lifter(worker_dir)
    started = time.perf_counter()
    for ordinal, scene in enumerate(scenes, 1):
        trace = process_scene(args.dataset, run, protocol, args.output,
                              scene, adapter)
        print(f'{args.dataset} worker={args.worker_index}/{args.workers} '
              f'{ordinal}/{len(scenes)} {scene}: '
              f'births={trace["birth_rows"]} active_peak='
              f'{trace["tracker"]["peak_active"]} wall={trace["wall_seconds"]:.1f}s',
              flush=True)
    summary = {
        'completed': True, 'dataset': args.dataset,
        'worker_index': args.worker_index, 'workers': args.workers,
        'scene_count_this_worker': len(scenes),
        'wall_seconds_this_invocation': time.perf_counter() - started,
        'predictions': str(args.output / 'predictions'),
    }
    write_json(args.output /
               f'complete.worker{args.worker_index:02d}of{args.workers:02d}.json',
               summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
