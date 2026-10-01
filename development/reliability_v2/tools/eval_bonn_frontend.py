#!/usr/bin/env python3
"""Unified gate evaluation: YOLOE vs native candidates against VLM annotations.

Annotation sources (eval-only):
- VLM contact-sheet checks on 10 keyframes (person visibility + boxes where
  reliable; crowd counts trusted, crowd box coords partially unreliable and
  used only where the quadrant convention was unambiguous).
- Visibility interpolation for person_tracking kfs 0..300 is justified by the
  tracked subject being continuously present (6 VLM-confirmed frames inside
  the run + 13 consecutive YOLOE hits); kfs >= 325 are VLM-confirmed absent.

Outputs per-candidate-source: matched-frame coverage, IoU on annotated boxes,
consecutive hit runs, person-class false positives on VLM-negative frames,
depth validity, timing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / 'reports/bonn_detect_20260914'

# frame-relative VLM boxes (pt) and counts (crowd)
PT_BOXES = {0: [296, 128, 386, 436], 25: [270, 128, 356, 436],
            100: [286, 130, 372, 438], 200: [291, 131, 374, 437],
            275: [282, 129, 366, 438]}
PT_VISIBLE_KFS = list(range(0, 325, 25))          # 13 kfs (6 VLM-confirmed)
CROWD_COUNTS = {0: 0, 75: 0, 150: 0, 225: 1, 600: 3, 875: 1, 900: 0, 925: 0}


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    u = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / max(u, 1)


def native_boxes_2d(scene, frame):
    rows = [json.loads(l) for l in
            (REP / 'native/observer' / f'{scene}.rows.jsonl').read_text().splitlines()]
    K = np.loadtxt(ROOT / f'data_bonn/{scene}/frames/intrinsic/intrinsic_depth.txt')[:3, :3]
    pose = np.loadtxt(ROOT / f'data_bonn/{scene}/frames/pose/{frame}.txt')
    Rt = np.linalg.inv(pose)
    out = []
    for r in rows:
        for o in r['obs']:
            if o[0] != frame:
                continue
            c = np.asarray(o[1]); s = np.asarray(o[2])
            half = s / 2
            local = np.array([[dx * half[0], dy * half[1], dz * half[2]]
                              for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)])
            pc = (Rt[:3, :3] @ (c + local).T).T + Rt[:3, 3]
            uv = (K @ pc.T).T
            uv = uv[:, :2] / uv[:, 2:3]
            out.append([float(uv[:, 0].min()), float(uv[:, 1].min()),
                        float(uv[:, 0].max()), float(uv[:, 1].max())])
    return out


def main():
    yoloe = json.loads((REP / 'yoloe_candidates.json').read_text())
    yoloe_by = {(r['scene'], r['frame']): r for r in yoloe}

    # person_tracking: per-frame YOLOE person vs VLM box
    pt_ious, pt_hits = [], 0
    for fr, box in PT_BOXES.items():
        dets = [d for d in yoloe_by[('scene0001_01', fr)]['detections']
                if d['label'] == 'person']
        if dets:
            best = max(iou(d['box2d'], box) for d in dets)
            pt_ious.append(best)
            pt_hits += 1
    pt_covered = sum(
        1 for fr in PT_VISIBLE_KFS
        if any(d['label'] == 'person'
               for d in yoloe_by[('scene0001_01', fr)]['detections']))
    run, best_run = 0, 0
    for fr in PT_VISIBLE_KFS:
        hit = any(d['label'] == 'person'
                  for d in yoloe_by[('scene0001_01', fr)]['detections'])
        run = run + 1 if hit else 0
        best_run = max(best_run, run)

    # native candidates vs the same VLM boxes
    native_best = {}
    for fr, box in PT_BOXES.items():
        boxes = native_boxes_2d('scene0001_01', fr)
        native_best[fr] = (max((iou(b, box) for b in boxes), default=0.0),
                           len(boxes))

    # crowd: count agreement on VLM-checked frames
    crowd_ok = []
    for fr, cnt in CROWD_COUNTS.items():
        y = sum(1 for d in yoloe_by[('scene0002_01', fr)]['detections']
                if d['label'] == 'person')
        crowd_ok.append((fr, cnt, y, cnt == y))
    cr_counts = [sum(1 for d in yoloe_by[('scene0002_01', fr)]['detections']
                     if d['label'] == 'person')
                 for fr in range(0, 927, 25)]
    run, cr_best = 0, 0
    for c in cr_counts:
        run = run + 1 if c > 0 else 0
        cr_best = max(cr_best, run)

    # person-class FPs on VLM-negative frames
    fp = sum(sum(1 for d in yoloe_by[(s, fr)]['detections'] if d['label'] == 'person')
             for s, frs in [('scene0001_01', [325, 350, 425, 500, 575]),
                            ('scene0002_01', [0, 75, 150, 900, 925])]
             for fr in frs)
    dv = [d['valid_depth_ratio'] for r in yoloe for d in r['detections']
          if d['label'] == 'person']
    ms = [r['infer_ms'] for r in yoloe]
    result = {
        'pt_coverage': f'{pt_covered}/{len(PT_VISIBLE_KFS)}',
        'pt_consecutive_hits': best_run,
        'pt_iou_vs_vlm': [round(v, 3) for v in pt_ious],
        'pt_iou_mean': round(float(np.mean(pt_ious)), 3) if pt_ious else None,
        'native_best_iou_vs_vlm': {str(k): round(v[0], 3) for k, v in native_best.items()},
        'native_obs_count_at_frames': {str(k): v[1] for k, v in native_best.items()},
        'crowd_count_agreement': crowd_ok,
        'crowd_longest_person_run_kf': cr_best,
        'yoloe_person_fps_on_vlm_negative': fp,
        'person_depth_validity_median': round(float(np.median(dv)), 3),
        'infer_ms_steady': round(float(np.median(ms)), 1),
    }
    (REP / 'frontend_gate_eval.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
