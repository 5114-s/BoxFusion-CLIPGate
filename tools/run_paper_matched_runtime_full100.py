#!/usr/bin/env python3
"""Paired full-100 live FPS benchmark: official BoxFusion vs M1-P+A+M2."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import time


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_COMMIT = '06ea629'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def ensure_official_source(output: Path) -> Path:
    source = output / 'official_source_06ea629'
    marker = source / '.source_receipt.json'
    if marker.is_file():
        receipt = json.loads(marker.read_text())
        if receipt.get('commit') != OFFICIAL_COMMIT:
            raise RuntimeError('official source receipt changed')
        return source
    source.mkdir(parents=True, exist_ok=False)
    archive = output / 'official_source_06ea629.tar'
    with archive.open('wb') as handle:
        subprocess.run(['git', 'archive', '--format=tar', OFFICIAL_COMMIT],
                       cwd=ROOT, check=True, stdout=handle)
    with tarfile.open(archive) as bundle:
        base = source.resolve()
        for member in bundle.getmembers():
            target = (source / member.name).resolve()
            if base not in target.parents and target != base:
                raise RuntimeError('unsafe archive member')
        bundle.extractall(source)
    archive.unlink()
    write_json(marker, {
        'commit': OFFICIAL_COMMIT,
        'demo_sha256': sha256(source / 'demo.py'),
        'config_sha256': sha256(source / 'config/scannet.yaml'),
    })
    return source


def completed(path: Path, scene: str) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text())
    except Exception:
        return False
    return value.get('scene') == scene


def run_child(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        tail = '\n'.join(log.read_text(errors='replace').splitlines()[-30:])
        raise RuntimeError(f'child failed ({result.returncode}): {command}\n{tail}')


def quantiles(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        'mean': statistics.fmean(values),
        'median': statistics.median(values),
        'p25': ordered[int(.25 * (len(ordered) - 1))],
        'minimum': ordered[0],
        'maximum': ordered[-1],
    }


def summarize(output: Path, scenes: list[str]) -> dict:
    pairs = []
    for scene in scenes:
        op = output / 'official' / scene / 'runtime.json'
        fp = output / 'final' / scene / 'runtime.json'
        if not op.is_file() or not fp.is_file():
            continue
        official = json.loads(op.read_text())
        final_root = json.loads(fp.read_text())
        final = final_root['arms']['m1pa_m2']
        if official['consumed_raw_frames'] != final_root['consumed_raw_frames']:
            raise RuntimeError(f'frame mismatch: {scene}')
        frames = int(official['consumed_raw_frames'])
        pairs.append({
            'scene': scene,
            'frames': frames,
            'official_seconds': float(official['run_seconds']),
            'official_fps': float(official['raw_fps']),
            'final_seconds': float(final['warm_end_to_end_seconds']),
            'final_fps': float(final['raw_fps']),
            'retention': float(final['raw_fps']) / float(official['raw_fps']),
        })
    result = {'completed_scenes': len(pairs), 'pairs': pairs}
    if pairs:
        frames = sum(row['frames'] for row in pairs)
        official_seconds = sum(row['official_seconds'] for row in pairs)
        final_seconds = sum(row['final_seconds'] for row in pairs)
        result['aggregate'] = {
            'frames': frames,
            'official_seconds': official_seconds,
            'final_seconds': final_seconds,
            'official_frame_weighted_fps': frames / official_seconds,
            'final_frame_weighted_fps': frames / final_seconds,
            'frame_weighted_retention': official_seconds / final_seconds,
            'official_scene_fps': quantiles([r['official_fps'] for r in pairs]),
            'final_scene_fps': quantiles([r['final_fps'] for r in pairs]),
            'paired_retention': quantiles([r['retention'] for r in pairs]),
        }
    write_json(output / 'summary.partial.json', result)
    return result


def write_report(output: Path, summary: dict) -> None:
    a = summary['aggregate']
    report = f"""# Paper-matched full100 FPS benchmark

Official BoxFusion control and M1-P+A+M2 were run live, serially, on the same
GPU and the same fixed ScanNet 100-scene list. Proposal replay and visualization
were disabled. Explicit model initialization and metric evaluation were
excluded; data I/O, inference, association/fusion, output serialization and all
method-specific stages were included. Arm order alternated by scene.

| Arm | Frame-weighted FPS | Mean scene FPS | Median | P25 | Minimum |
|---|---:|---:|---:|---:|---:|
| Official control | {a['official_frame_weighted_fps']:.4f} | {a['official_scene_fps']['mean']:.4f} | {a['official_scene_fps']['median']:.4f} | {a['official_scene_fps']['p25']:.4f} | {a['official_scene_fps']['minimum']:.4f} |
| **M1-P+A+M2** | **{a['final_frame_weighted_fps']:.4f}** | **{a['final_scene_fps']['mean']:.4f}** | **{a['final_scene_fps']['median']:.4f}** | **{a['final_scene_fps']['p25']:.4f}** | **{a['final_scene_fps']['minimum']:.4f}** |

Frame-weighted throughput retention: **{100*a['frame_weighted_retention']:.2f}%**.
Median paired scene retention: **{100*a['paired_retention']['median']:.2f}%**.

The control source is the frozen repository commit `{OFFICIAL_COMMIT}` with its
released ScanNet configuration (gap 25, score threshold 0.5, CuTR, association
and PFO). The current method adds its enhanced native route and M1-P+A+M2. Both
arms used the same RTX device recorded in the JSON receipts. This is a local
same-machine reproduction; it is not an RTX 3090 Ti reproduction of the paper's
published 20 FPS number.
"""
    (output / 'REPORT.md').write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--device-index', type=int, default=0)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = ensure_official_source(output)
    scene_list = ROOT / 'evaluation/data_util/meta_data/scannetv2_val.txt'
    scenes = scene_list.read_text().split()[:args.limit]
    protocol = {
        'schema': 'boxfusion.paper_matched.full100.v1',
        'scene_list': str(scene_list),
        'scene_list_sha256': sha256(scene_list),
        'scenes': scenes,
        'official_commit': OFFICIAL_COMMIT,
        'official_source_receipt': json.loads(
            (source / '.source_receipt.json').read_text()),
        'final_benchmark_sha256': sha256(
            ROOT / 'tools/benchmark_final_online_pipeline.py'),
        'official_benchmark_sha256': sha256(
            ROOT / 'tools/benchmark_official_boxfusion_scene.py'),
        'gap': 25,
        'proposal_replay': False,
        'visualization': False,
        'model_load_excluded': True,
        'device_index': args.device_index,
    }
    protocol_path = output / 'protocol.json'
    if protocol_path.is_file():
        if json.loads(protocol_path.read_text()) != protocol:
            raise RuntimeError('existing protocol differs')
    else:
        write_json(protocol_path, protocol)

    python = sys.executable
    data = ROOT / 'upstream_clean/scannet_readme_frames'
    started = time.perf_counter()
    for index, scene in enumerate(scenes):
        frame_count = len(list((data / scene / 'frames/color').glob('*.jpg')))
        if frame_count <= 0:
            raise RuntimeError(f'empty scene: {scene}')
        official_root = output / 'official' / scene
        final_root = output / 'final' / scene
        commands = {
            'official': [python, str(ROOT / 'tools/benchmark_official_boxfusion_scene.py'),
                         '--source-root', str(source), '--repository-root', str(ROOT),
                         '--scene', scene, '--output-root', str(official_root),
                         '--device-index', str(args.device_index)],
            'final': [python, str(ROOT / 'tools/benchmark_final_online_pipeline.py'),
                      '--frames', str(frame_count), '--scene', scene,
                      '--output-root', str(final_root), '--device-index',
                      str(args.device_index), '--final-only'],
        }
        order = ('official', 'final') if index % 2 == 0 else ('final', 'official')
        for arm in order:
            runtime = ((official_root if arm == 'official' else final_root)
                       / 'runtime.json')
            if completed(runtime, scene):
                continue
            print(f'RUN {index+1}/{len(scenes)} {scene} {arm}', flush=True)
            run_child(commands[arm], output / 'logs' / f'{scene}.{arm}.log')
        partial = summarize(output, scenes)
        pair = partial['pairs'][-1]
        print(f'DONE {index+1}/{len(scenes)} {scene} '
              f'official={pair["official_fps"]:.2f} '
              f'final={pair["final_fps"]:.2f} '
              f'retention={100*pair["retention"]:.1f}%', flush=True)

    summary = summarize(output, scenes)
    summary['completed'] = summary['completed_scenes'] == len(scenes)
    summary['wall_seconds_this_invocation'] = time.perf_counter() - started
    write_json(output / 'summary.json', summary)
    write_report(output, summary)
    print(json.dumps(summary['aggregate'], ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
