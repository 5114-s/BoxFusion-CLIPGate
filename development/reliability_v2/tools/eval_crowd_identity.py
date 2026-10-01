#!/usr/bin/env python3
"""Per-person identity evaluation of the DynamicPolicy on crowd.

Annotated persons are linked across the 6 annotated frames by image-x order
(left / mid / right slot); the policy's live track outputs are projected to
2D and matched one-to-one (IoU2D >= 0.2) to the annotated persons of each
frame.  Reported per annotated person-slot:

- coverage: annotated frames with a matched policy output;
- track consistency: the set of policy track ids matched over time and the
  number of id switches (changes of the majority/best-matching id between
  consecutive annotated frames);
- positional agreement: median IoU2D of matched outputs.

Caveat: x-order linking assumes no full x-crossing between the six sampled
frames; a real swap would be counted as an id switch for BOTH slots.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys_path = ROOT
import sys
sys.path.insert(0, str(ROOT))
from boxfusion.dynamic_policy import DynamicPolicy, Observation

ANN_FRAMES = [350, 525, 600, 675, 750, 875]


def iou2d(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(u, 1e-9)


def main():
    ann = json.loads((ROOT / 'reports/bonn_fourarm_20260914/crowd_annotation.json').read_text())
    recs = json.loads((ROOT / 'reports/bonn_detect_20260914/yoloe_candidates_v2.json').read_text())
    frames_dir = ROOT / 'data_bonn/scene0002_01/frames'
    K = np.loadtxt(frames_dir / 'intrinsic/intrinsic_depth.txt')[:3, :3]

    policy = DynamicPolicy({'completion_mode': 'truncation'})
    # replay online; capture outputs at annotated frames
    per_frame_out = {}
    for r in recs:
        if r['scene'] != 'scene0002_01':
            continue
        fr = r['frame']
        t = fr / 30.0
        obs_list = []
        for d in r['detections']:
            if d['label'] != 'person' or 'world_aabb_lo' not in d:
                continue
            lo = np.asarray(d['world_aabb_lo']); hi = np.asarray(d['world_aabb_hi'])
            obs_list.append(Observation(
                center=(lo + hi) / 2, lo=lo, hi=hi, score=d['score'],
                valid_px=d['n_depth_points'], depth_m=d.get('depth_median_m'),
                trunc_below=d.get('trunc_below', False),
                trunc_above=d.get('trunc_above', False)))
        policy.process(t, obs_list)
        if fr in ANN_FRAMES:
            pose = np.loadtxt(frames_dir / f'pose/{fr}.txt')
            Rt = np.linalg.inv(pose)
            boxes2d = {}
            for tid, b in policy.outputs(t).items():
                half = (b['hi'] - b['lo']) / 2
                local = np.array([[sx * half[0], sy * half[1], sz * half[2]]
                                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
                pc = (Rt[:3, :3] @ (b['center'] + local).T).T + Rt[:3, 3]
                uv = (K @ pc.T).T
                uv = uv[:, :2] / uv[:, 2:3]
                boxes2d[tid] = [float(uv[:, 0].min()), float(uv[:, 1].min()),
                                float(uv[:, 0].max()), float(uv[:, 1].max()), b['source']]
            per_frame_out[fr] = boxes2d

    # x-order slot assignment for annotations and matching
    rows = {slot: {'frames': [], 'tids': [], 'ious': []}
            for slot in ('left', 'mid', 'right')}
    for fr in ANN_FRAMES:
        gt_boxes = sorted(ann['frames'][str(fr)]['boxes'], key=lambda b: b[0])
        slots = ['left', 'mid', 'right'][:len(gt_boxes)]
        out = per_frame_out.get(fr, {})
        free = dict(out)
        for slot, gb in zip(slots, gt_boxes):
            best_tid, best_iou = None, 0.2
            for tid, pb in free.items():
                v = iou2d(pb[:4], gb)
                if v > best_iou:
                    best_tid, best_iou = tid, v
            if best_tid is not None:
                rows[slot]['frames'].append(fr)
                rows[slot]['tids'].append(best_tid)
                rows[slot]['ious'].append(round(best_iou, 3))
                del free[best_tid]

    report = {}
    for slot, r in rows.items():
        if not r['frames']:
            continue
        switches = sum(1 for a, b in zip(r['tids'], r['tids'][1:]) if a != b)
        report[slot] = {
            'matched_frames': f"{len(r['frames'])}/{sum(1 for fr in ANN_FRAMES if slot in ['left','mid','right'][:ann['frames'][str(fr)]['n']])}",
            'track_ids': r['tids'], 'id_switches': switches,
            'median_iou2d': round(float(np.median(r['ious'])), 3),
        }
    (ROOT / 'reports/bonn_fourarm_20260914/identity_results.json').write_text(
        json.dumps({'per_slot': report, 'stats': dict(policy.stats)}, indent=1))
    print(json.dumps({'per_slot': report, 'stats': dict(policy.stats)}, indent=1))


if __name__ == '__main__':
    main()
