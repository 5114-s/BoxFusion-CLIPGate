#!/usr/bin/env python3
"""GT-identity / GT-motion offline diagnostic on the walking benchmark.

Reads the observer dumps (per native row: observation history) written by the
dump-only motion-chain mode and answers, per scene:

1. single-frame candidate quality: for every observation whose centre is near
   the walker's GT position at its frame, the centre error (环节一);
2. association: which native rows absorbed the walker's observations, and how
   the observations split across rows (环节二);
3. fusion: error of the native fused box vs three offline references computed
   from the same observations -- static mean, GT-motion-compensated mean, and
   the best single observation (环节三 / 补偿上限).

GT is used offline only; nothing feeds back into inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def gt_world_corners(scene, gt_root):
    lines = Path(f'/extra/ZhaoX/scannet_data/scans/{scene}/{scene}.txt').read_text().splitlines()
    axis = np.fromstring(
        [l for l in lines if l.strip().startswith('axisAlignment')][0].split('=', 1)[1],
        sep=' ').reshape(4, 4)
    Tinv = np.linalg.inv(axis)
    gt = np.load(gt_root / f'{scene}_bbox.npy')
    out = []
    for c, s in zip(gt[:, :3], gt[:, 3:6]):
        h = s / 2.0
        local = np.array([[dx, dy, dz] for dx in (-h[0], h[0])
                          for dy in (-h[1], h[1]) for dz in (-h[2], h[2])])
        out.append((Tinv[:3, :3] @ (local + c).T).T + Tinv[:3, 3])
    return out


def horiz(a, b):
    d = np.asarray(a)[:2] - np.asarray(b)[:2]
    return float(np.linalg.norm(d))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=ROOT / 'data_dyn_walk/manifest.json')
    p.add_argument('--dump-dir', type=Path, default=ROOT / 'reports/walk_dev_20260913/native/observer')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--match-radius', type=float, default=0.6)
    args = p.parse_args()
    gt_root = ROOT / 'evaluation/data_util/scannet_train_detection_data'
    manifest = json.loads(args.manifest.read_text())

    scenes = {}
    for scene, e in sorted(manifest.items()):
        T, spd = e['T_frame'], e['speed_m_per_frame']
        direction = np.asarray(e['direction'], float)
        marker = np.asarray(e['marker'], float).reshape(8, 3)
        gt = gt_world_corners(scene, gt_root)
        ious = [((np.minimum(marker.max(0), g.max(0)) - np.maximum(marker.min(0), g.min(0)))
                 .clip(min=0).prod()) /
                max((marker.max(0) - marker.min(0)).prod() +
                    (g.max(0) - g.min(0)).prod() -
                    ((np.minimum(marker.max(0), g.max(0)) - np.maximum(marker.min(0), g.min(0)))
                     .clip(min=0).prod()), 1e-9)
                for g in gt]
        walker_gt = gt[int(np.argmax(ious))]
        gt_center = walker_gt.mean(0)

        def walker_pos(frame):
            return gt_center + direction * min(spd * max(frame - T, 0), e['D_max'])

        rows = [json.loads(l) for l in
                (args.dump_dir / f'{scene}.rows.jsonl').read_text().splitlines()]
        walker_obs = []       # (frame, center, err_to_gt_at_frame, row_key, score)
        for row in rows:
            for fr, center, size, score in row['obs']:
                target = walker_pos(fr)
                err = horiz(center, target)
                if err <= args.match_radius:
                    walker_obs.append((fr, np.asarray(center), err, row['key'], score))
        walker_obs.sort(key=lambda x: x[0])

        final_pos = walker_pos(10 ** 9)
        native_err = None
        if walker_obs:
            # native fused box of the row with most walker observations
            keys = [o[3] for o in walker_obs]
            main_key = max(set(keys), key=keys.count)
            native = [r for r in rows if r['key'] == main_key][0]
            native_err = horiz(np.asarray(native['native_corners']).mean(0), final_pos)

        obs_at_final = [o for o in walker_obs
                        if horiz(o[1], final_pos) <= args.match_radius]
        report = {
            'scene': scene,
            'n_rows': len(rows),
            'walker_obs_count': len(walker_obs),
            'walker_obs_frames': sorted({o[0] for o in walker_obs}),
            'walker_obs_rows': sorted({o[3] for o in walker_obs}),
            'single_frame_err_median': (float(np.median([o[2] for o in walker_obs]))
                                        if walker_obs else None),
            'single_frame_err_final_phase': (float(np.median(
                [horiz(o[1], walker_pos(o[0])) for o in walker_obs if o[0] >= T]))
                if any(o[0] >= T for o in walker_obs) else None),
            'obs_in_final_region': len(obs_at_final),
            'native_fused_center_err_final': native_err,
        }
        if len(walker_obs) >= 2:
            centers = np.stack([o[1] for o in walker_obs])
            static_mean = centers.mean(0)
            report['static_mean_center_err_final'] = horiz(static_mean, final_pos)
            d_final = direction * e['D_max']

            def delta(t):
                return direction * min(spd * max(t - T, 0), e['D_max'])

            comp = np.stack([o[1] + (d_final - delta(o[0])) for o in walker_obs])
            report['compensated_mean_center_err_final'] = horiz(comp.mean(0), final_pos)
            best_single = min(horiz(o[1], walker_pos(o[0])) for o in walker_obs)
            report['best_single_obs_err_at_its_time'] = best_single
        report['walker_obs_in_motion_phase'] = sum(1 for o in walker_obs if o[0] >= T)
        scenes[scene] = report
        print(json.dumps(report, default=str), flush=True)

    summary = {k: float(np.mean([s[k] for s in scenes.values()
                                 if s[k] is not None]))
               for k in ('single_frame_err_median', 'native_fused_center_err_final')}
    out = {'scenes': scenes, 'summary': summary,
           'match_radius_m': args.match_radius}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))
    print('SUMMARY', json.dumps(summary))


if __name__ == '__main__':
    main()
