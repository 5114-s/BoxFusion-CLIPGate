#!/usr/bin/env python3
"""Stage-1 gate analysis on Bonn RGB-D Dynamic: can the native pipeline see
the people at all?

Reads the observer dumps of a native run (post score-threshold, post-NMS
keyframe observations -- exactly what fusion would receive) and reports, per
sequence:

- person coverage: fraction of sampled keyframes with >= 1 person-category
  observation (CLIP category in PERSON_TAGS);
- consecutive miss run lengths over the keyframe series;
- person-box size sanity (world extents);
- center jump: nearest-neighbour linked centre displacement between
  consecutive keyframes that both have person observations.

Gate (dev entry): coverage >= 50% on visible frames AND at least one run of
>= 3 consecutive keyframes with detections.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PERSON_TAGS = {'person', 'man', 'woman', 'people', 'girl', 'boy', 'child'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dump-dir', type=Path, required=True)
    p.add_argument('--scenes', nargs='+', required=True)
    p.add_argument('--gap', type=int, default=25)
    p.add_argument('--n-frames', type=int, nargs='+', required=True,
                   help='total converted frames per scene, same order as --scenes')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()

    report = {}
    for scene, n in zip(args.scenes, args.n_frames):
        kfs = list(range(0, n, args.gap))
        rows = [json.loads(l) for l in
                (args.dump_dir / f'{scene}.rows.jsonl').read_text().splitlines()]
        persons = {}  # kf -> list of centers
        sizes = []
        for row in rows:
            for obs in row['obs']:
                fr, center, size, score, cat = obs[0], obs[1], obs[2], obs[3], (obs[4] if len(obs) > 4 else '')
                if str(cat).strip().lower() in PERSON_TAGS:
                    persons.setdefault(int(fr), []).append(np.asarray(center))
                    sizes.append(np.asarray(size))
        have = [1 if k in persons else 0 for k in kfs]
        coverage = float(np.mean(have)) if kfs else 0.0
        # consecutive miss runs
        miss_runs, run = [], 0
        for h in have:
            if h == 0:
                run += 1
            elif run:
                miss_runs.append(run)
                run = 0
        if run:
            miss_runs.append(run)
        # consecutive hit runs (longest)
        hit_runs, run = [], 0
        for h in have:
            if h == 1:
                run += 1
            elif run:
                hit_runs.append(run)
                run = 0
        if run:
            hit_runs.append(run)
        # centre jumps: link nearest person obs between consecutive kfs
        jumps = []
        kfs_with = [k for k in kfs if k in persons]
        for a, b in zip(kfs_with, kfs_with[1:]):
            if b - a != args.gap:
                continue
            ca, cb = persons[a], persons[b]
            d = [float(np.linalg.norm(x[:2] - y[:2])) for x in ca for y in cb]
            jumps.append(min(d))
        report[scene] = {
            'keyframes': len(kfs),
            'person_obs_total': sum(len(v) for v in persons.values()),
            'coverage': round(coverage, 4),
            'miss_run_lengths': miss_runs,
            'longest_hit_run_kf': max(hit_runs) if hit_runs else 0,
            'center_jump_min_q50_q90': ([round(float(np.quantile(jumps, q)), 3)
                                         for q in (0, 0.5, 0.9)] if jumps else None),
            'person_size_median_lwh': (np.median(np.stack(sizes), 0).round(3).tolist()
                                       if sizes else None),
            'gate_pass': bool(coverage >= 0.5 and (max(hit_runs) if hit_runs else 0) >= 3),
        }
        print(scene, json.dumps(report[scene]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1))
    print('GATE:', {s: r['gate_pass'] for s, r in report.items()})


if __name__ == '__main__':
    main()
