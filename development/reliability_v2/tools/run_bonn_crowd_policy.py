#!/usr/bin/env python3
"""Multi-person (crowd) online sanity check of the integrated DynamicPolicy.

No per-person GT exists for crowd; evaluation anchors are the 8 VLM-verified
person-COUNT frames and the two VLM-verified empty stretches.  Measures:

- output count vs verified count at those frames (over/under-shoot);
- phantom outputs inside verified-empty stretches (0-225 s-range, 900-925);
- duplicate outputs (two live tracks with 3D box overlap > 0.3 IoU);
- track lifecycle timeline (births/retirements vs verified presence).

Policy is causal throughout; cached frozen YOLOE candidates only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys_path = ROOT
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from boxfusion.dynamic_policy import DynamicPolicy, Observation

VERIFIED_COUNTS = {0: 0, 75: 0, 150: 0, 225: 1, 600: 3, 875: 1, 900: 0, 925: 0}
EMPTY_STRETCH = [(0, 200), (900, 925)]


def iou_aabb(a, b):
    inter = np.maximum(0, np.minimum(a[1], b[1]) - np.maximum(a[0], b[0])).prod()
    u = (a[1] - a[0]).prod() + (b[1] - b[0]).prod() - inter
    return float(inter / max(u, 1e-9))


def main():
    ts = {int(k): v for k, v in
          json.loads((ROOT / 'reports/bonn_detect_20260914/pt_timestamps.json').read_text()).items()}
    recs = json.loads((ROOT / 'reports/bonn_detect_20260914/yoloe_candidates.json').read_text())
    # crowd timestamps: rebuilt analogously (927 frames kept)
    crowd_ts = json.loads((ROOT / 'reports/bonn_detect_20260914/crowd_timestamps.json').read_text()) \
        if (ROOT / 'reports/bonn_detect_20260914/crowd_timestamps.json').exists() else None
    policy = DynamicPolicy({'completion_mode': 'off'})
    timeline = []
    count_rows = []
    phantom = 0
    dup_frames = 0
    for r in recs:
        if r['scene'] != 'scene0002_01':
            continue
        fr = r['frame']
        t = float(fr) / 30.0 if crowd_ts is None else crowd_ts[str(fr)]
        obs_list = []
        for d in r['detections']:
            if d['label'] != 'person' or 'world_aabb_lo' not in d:
                continue
            lo = np.asarray(d['world_aabb_lo']); hi = np.asarray(d['world_aabb_hi'])
            obs_list.append(Observation(center=(lo + hi) / 2, lo=lo, hi=hi,
                                        score=d['score'], valid_px=d['n_depth_points'],
                                        depth_m=d.get('depth_median_m')))
        policy.process(t, obs_list)
        outs = policy.outputs(t)
        boxes = list(outs.values())
        # duplicates among live outputs
        dup = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if iou_aabb((boxes[i]['lo'], boxes[i]['hi']),
                            (boxes[j]['lo'], boxes[j]['hi'])) > 0.3:
                    dup = True
        if dup and len(boxes) > 1:
            dup_frames += 1
        if fr in VERIFIED_COUNTS:
            count_rows.append({'frame': fr, 'verified': VERIFIED_COUNTS[fr],
                               'policy_outputs': len(boxes)})
        for lo_, hi_ in EMPTY_STRETCH:
            if lo_ <= fr <= hi_:
                phantom += len(boxes)
        timeline.append({'frame': fr, 'n_obs': len(obs_list),
                         'n_out': len(boxes),
                         'live_tracks': sum(1 for x in policy.tracks.values() if not x.retired)})
    exact = sum(1 for c in count_rows if c['verified'] == c['policy_outputs'])
    result = {
        'count_check': {'rows': count_rows, 'exact_matches': f'{exact}/{len(count_rows)}'},
        'phantom_outputs_in_verified_empty_stretches': phantom,
        'frames_with_duplicate_outputs': dup_frames,
        'total_keyframes': len(timeline),
        'final_stats': dict(policy.stats),
        'max_concurrent_tracks': max(x['live_tracks'] for x in timeline),
        'timeline': timeline,
    }
    out = ROOT / 'reports/dynamic_policy_replay_20260914'
    out.mkdir(exist_ok=True)
    (out / 'crowd_policy_results.json').write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in
                      ('count_check', 'phantom_outputs_in_verified_empty_stretches',
                       'frames_with_duplicate_outputs', 'total_keyframes',
                       'final_stats', 'max_concurrent_tracks')}, indent=1))


if __name__ == '__main__':
    main()
