#!/usr/bin/env python3
"""100-scene end-to-end timing for the final M1-P+A+M2 route.

Drives tools/benchmark_final_online_pipeline.py once per official val scene
(--final-only), extracts the warm end-to-end timing from each runtime.json,
then deletes the per-scene work directory (frames are copied per scene and
the disk has no room for 100 copies).  Reports mean raw FPS, scene-level
P95 latency, and peak CUDA memory.  runtime.json carries per-scene totals,
not per-frame latencies, so the P95 is over scenes and is disclosed as such.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/benchmark_100_20260916'
PYTHON = '/home/admin1/miniconda3/envs/boxfusion2/bin/python'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device-index', type=int, default=1)
    parser.add_argument('--limit', type=int, default=100)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = (ROOT / 'evaluation/data_util/meta_data/scannetv2_val.txt'
              ).read_text().split()[:args.limit]
    per_scene = {}
    for i, scene in enumerate(scenes, 1):
        work = OUT / 'work'
        if work.exists():
            shutil.rmtree(work)
        started = time.perf_counter()
        result = subprocess.run(
            [PYTHON, 'tools/benchmark_final_online_pipeline.py',
             '--final-only', '--scene', scene,
             '--device-index', str(args.device_index),
             '--output-root', str(work)],
            cwd=ROOT, capture_output=True, text=True)
        wall = time.perf_counter() - started
        runtime_file = work / 'runtime.json'
        if not runtime_file.is_file():
            per_scene[scene] = {'error': result.stderr[-2000:],
                                'wall_seconds': wall}
            (OUT / 'per_scene.json').write_text(
                json.dumps(per_scene, ensure_ascii=False, indent=2) + '\n')
            print(f'{i:3d}/{len(scenes)} {scene} FAILED', flush=True)
            continue
        runtime = json.loads(runtime_file.read_text())
        arm = runtime['arms']['m1pa_m2']
        per_scene[scene] = {
            'consumed_raw_frames': runtime['consumed_raw_frames'],
            'keyframes': runtime['keyframes'],
            'warm_end_to_end_seconds': arm['warm_end_to_end_seconds'],
            'native_seconds': arm['native_seconds'],
            'extension_seconds': arm['extension_seconds'],
            'semantic_readout_seconds': arm['semantic_readout_seconds'],
            'raw_fps': arm['raw_fps'],
            'effective_ms_per_raw_frame': arm['effective_ms_per_raw_frame'],
            'peak_allocated_bytes': runtime['peak_allocated_bytes'],
            'rows': runtime['rows'].get('m1pa_m2'),
            'model_load_seconds': runtime['model_load_seconds'],
            'benchmark_wall_seconds': wall,
        }
        print(f'{i:3d}/{len(scenes)} {scene} '
              f"fps={arm['raw_fps']:.2f} "
              f"ms/frame={arm['effective_ms_per_raw_frame']:.2f} "
              f"rows={per_scene[scene]['rows']}", flush=True)
        shutil.rmtree(work)
        (OUT / 'per_scene.json').write_text(
            json.dumps(per_scene, ensure_ascii=False, indent=2) + '\n')
    good = [v for v in per_scene.values() if 'raw_fps' in v]
    fps = [v['raw_fps'] for v in good]
    ms = [v['effective_ms_per_raw_frame'] for v in good]
    peaks = [v['peak_allocated_bytes'] for v in good]
    summary = {
        'protocol': 'tools/benchmark_final_online_pipeline.py --final-only '
                    'per scene; warm end-to-end includes native live run, '
                    'M1-P finalization, strict-online M1-A state machine, '
                    'M2, serialization and terminal semantic readout; '
                    'excludes model loading',
        'scene_count': len(good), 'failed': len(per_scene) - len(good),
        'mean_raw_fps': sum(fps) / len(fps),
        'min_raw_fps': min(fps), 'max_raw_fps': max(fps),
        'mean_ms_per_raw_frame': sum(ms) / len(ms),
        'p95_ms_per_raw_frame_scene_level':
            sorted(ms)[int(np.ceil(0.95 * len(ms))) - 1] if False else
            sorted(ms)[min(len(ms) - 1, int(np.ceil(0.95 * len(ms))) - 1)],
        'p50_ms_per_raw_frame_scene_level': sorted(ms)[len(ms) // 2],
        'peak_allocated_bytes_max': max(peaks),
        'peak_allocated_bytes_mean': sum(peaks) / len(peaks),
        'latency_granularity': 'per-scene totals; runtime.json has no '
                               'per-frame latency log',
    }
    import numpy as np  # noqa: F401  (kept out of the hot loop)
    summary['p95_ms_per_raw_frame_scene_level'] = float(
        np.percentile(ms, 95))
    summary['p50_ms_per_raw_frame_scene_level'] = float(np.percentile(ms, 50))
    (OUT / 'runtime_100.json').write_text(
        json.dumps({'summary': summary, 'per_scene': per_scene},
                   ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
