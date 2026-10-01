#!/usr/bin/env python3
"""Lean paired runner for the walking benchmark (dev stage).

Runs demo.py per scene per arm with a per-GPU worker; completed scenes are
skipped by their result record.  No frozen-source discipline (dev), but every
run keeps its log and artifact hashes for auditability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--gpus', default='0,1')
    p.add_argument('--manifest', type=Path, default=ROOT / 'data_dyn_walk/manifest.json')
    args = p.parse_args()
    scenes = sorted(json.loads(args.manifest.read_text()))
    gpus = args.gpus.split(',')
    arm = args.run_dir.name
    log_dir = args.run_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / 'persistent').mkdir(exist_ok=True)
    stop = threading.Event()
    lock = threading.Lock()
    prefix = Path(sys.executable).resolve().parents[1]

    def worker(index, gpu):
        for s_i in range(index, len(scenes), len(gpus)):
            if stop.is_set():
                return
            scene = scenes[s_i]
            record_path = log_dir / f'{scene}.result.json'
            if record_path.exists():
                rec = json.loads(record_path.read_text())
                if rec['returncode'] == 0 and all(
                        sha256(f) == h for f, h in rec['artifacts'].items()):
                    continue
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONNOUSERSITE='1',
                       PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1',
                       MPLCONFIGDIR='/tmp/boxfusion_walk_mpl')
            env['LD_LIBRARY_PATH'] = (
                f'{prefix}/lib:{prefix}/lib/opt/rviz_ogre_vendor/lib'
                ':/usr/local/cuda-12.1/lib64')
            command = [sys.executable, str(ROOT / 'demo.py'), 'scannet',
                       '--model-path', str(ROOT / 'models/cutr_rgbd.pth'),
                       '--config', str(args.config.resolve()),
                       '--device', 'cuda', '--seq', scene]
            log_path = log_dir / f'{scene}.log'
            with lock:
                print(f'START {arm} {s_i+1}/{len(scenes)} {scene} gpu={gpu}', flush=True)
            start = time.monotonic()
            with log_path.open('w') as log:
                code = subprocess.call(command, cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            elapsed = time.monotonic() - start
            text = log_path.read_text(errors='replace')
            timing = re.findall(r'Cost: ([\d.]+) s Average FPS: ([\d.]+)', text)
            artifacts = {}
            out = args.run_dir / 'persistent' / f'{scene}_boxes.pkl'
            if out.is_file():
                artifacts[str(out)] = sha256(out)
            json.dump({'scene': scene, 'arm': arm, 'gpu': gpu,
                       'returncode': code, 'wall_seconds': elapsed,
                       'loop_seconds': float(timing[-1][0]) if timing else None,
                       'artifacts': artifacts},
                      open(record_path, 'w'))
            with lock:
                print(f'DONE {arm} {scene} rc={code} wall={elapsed:.1f}s', flush=True)
            if code or not artifacts:
                stop.set()
                raise RuntimeError(f'failure; inspect {log_path}')

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, i, g) for i, g in enumerate(gpus)]
        for f in futures:
            f.result()
    print(f'WALK_RUN_COMPLETE {args.run_dir}', flush=True)


if __name__ == '__main__':
    main()
