"""Supervise baseline validation, two GPU workers, and final pooled AP."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    output = args.output.resolve()
    root = Path(__file__).resolve().parents[1]
    runner = root / 'tools/run_scannet_seedless_full.py'
    base = [sys.executable, str(runner)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', HF_HUB_OFFLINE='1')

    def status(stage, **extra):
        value = {'stage': stage, 'supervisor_pid': os.getpid(), 'updated_unix': time.time(), **extra}
        temp = output / 'run_status.tmp'
        temp.write_text(json.dumps(value, indent=2) + '\n')
        temp.replace(output / 'run_status.json')
        print(json.dumps(value), flush=True)

    def command(stage, *extra):
        return base + [stage, '--output', str(output), *extra]

    status('baseline_validation')
    with (output / 'baseline.log').open('x') as log:
        done = subprocess.run(command('baseline'), cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    if done.returncode:
        status('failed', failed_stage='baseline', exit_code=done.returncode)
        raise SystemExit(done.returncode)
    workers, logs = [], []
    for i in range(2):
        log = (output / f'worker_{i}.log').open('x')
        logs.append(log)
        workers.append(subprocess.Popen(command('worker', '--workers', '2', '--worker-index', str(i)),
            cwd=root, env=dict(env, CUDA_VISIBLE_DEVICES=str(i)), stdout=log, stderr=subprocess.STDOUT))
    previous = -1
    while any(w.poll() is None for w in workers):
        count = len(list((output / 'scenes').glob('*/complete.json')))
        if count != previous:
            status('workers_running', completed_scenes=count, total_scenes=100,
                   worker_pids=[w.pid for w in workers], worker_exit_codes=[w.poll() for w in workers])
            previous = count
        time.sleep(10)
    for log in logs:
        log.close()
    codes = [w.returncode for w in workers]
    if any(codes):
        status('failed', failed_stage='workers', worker_exit_codes=codes,
               completed_scenes=len(list((output / 'scenes').glob('*/complete.json'))))
        raise SystemExit(1)
    status('evaluating', completed_scenes=100)
    with (output / 'eval.log').open('x') as log:
        done = subprocess.run(command('evaluate'), cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    if done.returncode:
        status('failed', failed_stage='evaluate', exit_code=done.returncode)
        raise SystemExit(done.returncode)
    results = json.loads((output / 'results.json').read_text())
    lines = ['# ScanNet Step 1c — full100', '',
             f"Completed {results['scene_count']} scenes / {results['frame_count']} keyframes.", '',
             'Baseline: `results/scannet_m2nl_m5_dual_full100/persistent`.', '',
             '| Variant | AP15 | AP25 | AP50 | ΔAP50 | Births | Unmatched at IoU15 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for variant, metrics in results['metrics'].items():
        births = results['birth_stats'][variant]
        values = [metrics[str(t)]['ap'] for t in (.15, .25, .5)]
        delta = metrics['0.5'].get('delta_ap', 0.)
        lines.append(f'| {variant} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | '
                     f'{delta:+.4f} | {births.get("births", 0)} | {births.get("births_unmatched_gt15", 0)} |')
    lines += ['', 'AP values are percentages; deltas are percentage points.', '',
              'All four arms were checked against the exact repository ScanNet evaluator.', '',
              '## Limits', ''] + ['- ' + x for x in results['limits']]
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    status('completed', completed_scenes=100, results=str(output / 'results.json'))


if __name__ == '__main__':
    main()
