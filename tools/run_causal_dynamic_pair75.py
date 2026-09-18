#!/usr/bin/env python3
"""Frozen three-arm online inference. No GT is passed to demo.py; no tuning.

One GPU owns all three arms of each scene, sequentially, with rotated arm order.
Two GPUs may process different scenes concurrently. Completed artifacts are
never overwritten. The paired evaluator runs automatically after 225 successes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.eval_scannet_causal_dynamic_ap import sha256, write_json
from tools.validate_dynamic_run_coverage import validate_dynamic_run_coverage

ARMS = ('native_off', 'full_on', 'no_miss_retirement')


def arm_config(base, run_dir, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    cfg = deepcopy(base)
    target = run_dir / arm
    cfg['data']['output_dir'] = str(target / 'persistent')
    cfg['lifting']['boxer']['diagnostics_dir'] = str(target / 'boxer')
    branch = cfg['causal_dynamic_branch']
    branch['events_root'] = str(target / 'events')
    branch['current_output_root'] = str(target / 'current')
    branch['mode'] = 'disabled' if arm == 'native_off' else 'active'
    cfg['dynamic_objects']['enabled'] = arm != 'native_off'
    cfg['dynamic_objects']['miss_lifecycle_updates'] = arm != 'no_miss_retirement'
    return cfg


def fingerprints():
    paths = [ROOT / 'demo.py', ROOT / 'tools/utils.py', Path(__file__),
             ROOT / 'tools/eval_causal_dynamic_pair75.py',
             ROOT / 'tools/eval_scannet_causal_dynamic_ap.py',
             ROOT / 'tools/validate_dynamic_run_coverage.py']
    paths.extend((ROOT / 'boxfusion').rglob('*.py'))
    paths.extend((ROOT / 'datasets').rglob('*.py'))
    return {str(p): sha256(p) for p in sorted(set(paths))}


def verify_sources(protocol):
    for path, digest in protocol['frozen_sha256'].items():
        if sha256(path) != digest:
            raise ValueError(f'Frozen source/config/manifest changed: {path}')


def prepare(run_dir, config, gpus):
    manifest = ROOT / 'data_dyn/manifest.json'
    scenes = sorted(json.loads(manifest.read_text()))
    if len(scenes) != 75:
        raise ValueError(f'Require all 75 scenes, got {len(scenes)}')
    base = yaml.safe_load(config.read_text())
    inventory = {}
    for scene in scenes:
        frames = ROOT / 'data_dyn' / scene / 'frames'
        for name, original in [('K_depth.txt', 'intrinsic_depth.txt'),
                               ('K_rgb.txt', 'intrinsic_color.txt')]:
            actual = np.loadtxt(frames / name).reshape(3, 3)
            expected = np.loadtxt(frames / 'intrinsic' / original).reshape(4, 4)[:3, :3]
            if not np.allclose(actual, expected, rtol=0, atol=1e-5):
                raise ValueError(f'Calibration compatibility mismatch: {scene}/{name}')
        names = {folder: {p.stem for p in (frames / folder).glob(pattern)}
                 for folder, pattern in [('color', '*.jpg'), ('depth', '*.png'), ('pose', '*.txt')]}
        if not names['color'] or not names['color'] == names['depth'] == names['pose']:
            raise ValueError(f'Incomplete RGB-D/pose stream: {scene}')
        inventory[scene] = {'input_frames': len(names['color']),
                            'keyframes': len(range(0, len(names['color']), base['data']['gap']))}
    run_dir.mkdir(parents=True, exist_ok=False)
    configs = {}
    for arm in ARMS:
        path = run_dir / f'{arm}.yaml'
        with path.open('x') as f:
            yaml.safe_dump(arm_config(base, run_dir, arm), f, sort_keys=False)
        configs[arm] = str(path)
    frozen = fingerprints()
    for path in [manifest, config, *(Path(p) for p in configs.values())]:
        frozen[str(path.resolve())] = sha256(path)
    protocol = {
        'schema': 'boxfusion.dynamic_pair75.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'manifest': str(manifest), 'scenes': scenes, 'arms': list(ARMS),
        'configs': configs, 'gpus': gpus, 'inventory': inventory,
        'frozen_sha256': frozen,
        'primary_comparison': 'full_on current minus native_off output on identical terminal GT',
        'ablation': 'no_miss_retirement: disable all miss-driven score/occlusion/coast/age lifecycle changes; association/hit/geometry/capacity rules unchanged',
        'scope': '75 synthetic removal scenes, terminal class-agnostic AP; NOT real motion tracking or semantic AP',
        'stale_metric': 'fraction of removed GT with any predicted IoU > 0.25 and score > 0.30; fixed auxiliary threshold, NOT used for AP',
        'live_loss_metric': 'remaining GT covered by native at IoU > 0.25, score > 0.30 but not covered by variant; detection loss, not causal proof of false retirement',
        'timing': 'input-frame normalized throughput; keyframe gap=25. Report measured loop and process wall time separately, not keyframe Hz as input FPS.',
        'claim_rule': 'report every arm and all AP thresholds; no favorable-subset selection. Stale reduction alone does not establish overall benefit.',
    }
    write_json(run_dir / 'protocol.json', protocol)
    return protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT / 'config/scannet_dyn_causal_dynamic_active.yaml')
    p.add_argument('--gpus', default='0,1')
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    run_dir = args.run_dir.resolve()
    gpus = args.gpus.split(',')
    if not gpus or len(gpus) != len(set(gpus)) or any(not g.isdigit() for g in gpus):
        raise ValueError('GPUs must be distinct integer indices')
    protocol_path = run_dir / 'protocol.json'
    protocol = (json.loads(protocol_path.read_text()) if protocol_path.is_file()
                else prepare(run_dir, args.config.resolve(), gpus))
    if protocol['gpus'] != gpus:
        raise ValueError('Resume must use the frozen GPU assignment')
    verify_sources(protocol)
    if args.prepare_only:
        print(f'PREPARED {protocol_path}', flush=True)
        return
    # An exclusive lock prevents concurrent resumes from touching the same logs.
    import fcntl
    with (run_dir / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stop = threading.Event()
        progress_lock = threading.Lock()

        def worker(worker_index, gpu):
            for scene_index in range(worker_index, 75, len(gpus)):
                scene = protocol['scenes'][scene_index]
                order = ARMS[scene_index % 3:] + ARMS[:scene_index % 3]
                for arm in order:
                    if stop.is_set():
                        return
                    target = run_dir / arm
                    target.mkdir(exist_ok=True)
                    log_dir = target / 'logs'
                    log_dir.mkdir(exist_ok=True)
                    for directory in ('persistent', 'current', 'events', 'boxer'):
                        (target / directory).mkdir(exist_ok=True)
                    record_path = log_dir / f'{scene}.result.json'
                    if record_path.exists():
                        record = json.loads(record_path.read_text())
                        if record['returncode'] or any(sha256(f) != h for f, h in record['artifacts'].items()):
                            raise ValueError(f'Failed or changed prior trial; preserve and inspect {record_path}')
                        continue
                    verify_sources(protocol)
                    log_path = log_dir / f'{scene}.log'
                    env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONNOUSERSITE='1',
                               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                               MPLCONFIGDIR='/tmp/boxfusion_dynamic_pair_mpl')
                    prefix = Path(sys.executable).resolve().parents[1]
                    env['LD_LIBRARY_PATH'] = f'{prefix}/lib:{prefix}/lib/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64'
                    command = [sys.executable, str(ROOT / 'demo.py'), 'scannet',
                               '--model-path', str(ROOT / 'models/cutr_rgbd.pth'),
                               '--config', protocol['configs'][arm], '--device', 'cuda', '--seq', scene]
                    with progress_lock:
                        print(f'START scene={scene_index+1}/75 arm={arm} gpu={gpu} log={log_path}', flush=True)
                    start = time.monotonic()
                    with log_path.open('x') as log:
                        code = subprocess.call(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    elapsed = time.monotonic() - start
                    text = log_path.read_text(errors='replace')
                    timing = re.findall(r'Cost: ([\d.]+) s Average FPS: ([\d.]+)', text)
                    artifacts = [target / 'persistent' / f'{scene}_boxes.pkl']
                    if arm != 'native_off':
                        artifacts += [target / 'current' / f'{scene}_boxes.pkl', target / 'events' / f'{scene}.jsonl']
                    terminal_event_complete = True
                    if arm != 'native_off':
                        event_path = target / 'events' / f'{scene}.jsonl'
                        terminal_event_complete = False
                        if event_path.is_file():
                            event_lines = event_path.read_text().splitlines()
                            if event_lines:
                                final = json.loads(event_lines[-1])
                                terminal_event_complete = final.get('type') == 'summary' and final.get('scene_id') == scene
                    record = {'scene': scene, 'arm': arm, 'gpu': gpu, 'returncode': code,
                              'terminal_event_complete': terminal_event_complete,
                              'process_wall_seconds': elapsed,
                              'loop_seconds': float(timing[-1][0]) if timing else None,
                              'reported_input_fps': float(timing[-1][1]) if timing else None,
                              'artifacts': {str(f): sha256(f) for f in artifacts if f.is_file()}}
                    write_json(record_path, record)
                    with progress_lock:
                        print(f'DONE scene={scene_index+1}/75 arm={arm} rc={code} wall={elapsed:.1f}s', flush=True)
                    if code or not terminal_event_complete or not timing or len(record['artifacts']) != len(artifacts):
                        stop.set()
                        raise RuntimeError(f'Stopping on first failure; inspect {log_path}')
                    verify_sources(protocol)

        def guarded_worker(i, gpu):
            try:
                return worker(i, gpu)
            except BaseException:
                stop.set()
                raise

        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(guarded_worker, i, gpu) for i, gpu in enumerate(gpus)]
            for future in futures:
                future.result()
        for arm in ARMS:
            target = run_dir / arm
            if arm != 'native_off':
                validate_dynamic_run_coverage(
                    protocol['manifest'], persistent_root=target / 'persistent',
                    current_root=target / 'current',
                    event_roots=[target / 'events'])
        verify_sources(protocol)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        subprocess.run([sys.executable, str(ROOT / 'tools/eval_causal_dynamic_pair75.py'),
                        '--run-dir', str(run_dir)], cwd=ROOT, env=env, check=True)
        print(f'ALL_225_TRIALS_AND_AP_COMPLETE {run_dir}', flush=True)


if __name__ == '__main__':
    main()
