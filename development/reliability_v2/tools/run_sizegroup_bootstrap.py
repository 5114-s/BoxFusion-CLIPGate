#!/usr/bin/env python3
"""Size-group breakdown + per-scene paired bootstrap on the CA-1M full107 arms.

Pre-registered before scoring:
- GT volume tertiles pooled over all 12,911 GT (AABB volume); groups
  small/medium/large with the pooled cut points reported.
- Subgroup AP: predictions unchanged; the GT set is restricted to the group
  (predictions matching out-of-group GTs count as FP inside the subgroup
  evaluation -- conservative, standard subgroup protocol, declared).
- Per-scene AP per arm at IoU 0.15/0.25/0.5; paired deltas
  (M1-native, M1M2-M1, M1M2-native); 10,000-scene bootstrap CIs on the mean
  delta; fraction of scenes with negative delta reported alongside.

Arms and inputs are the frozen full107 artifacts already cross-checked
against the historical ledger; no new inference, no retuning.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / 'tools'))
from true_fusion_audit_core import class_agnostic_ap

METHODS = {'native': 'ca1m_thr15', 'M1': 'ca1m_thr15_m1_full107',
           'M1_M2': 'ca1m_thr15_m1_m2nl_full107'}
THRESHOLDS = (0.15, 0.25, 0.5)
OUT = ROOT / 'reports/sizegroup_bootstrap_20260915'
GT_ROOT = Path('/extra/ZhaoX/boxfusion_ca1m')
N_BOOT = 10000
rng = np.random.default_rng(0)


def main():
    scenes = sorted(p.name.replace('_boxes.pkl', '')
                    for p in (ROOT / 'results/ca1m_thr15').glob('*_boxes.pkl'))
    scenes = [s for s in scenes if (GT_ROOT / s / 'after_filter_boxes.npy').is_file()]
    assert len(scenes) == 107, len(scenes)
    OUT.mkdir(parents=True, exist_ok=True)
    gts, preds, hashes = {}, {}, {}
    for scene in scenes:
        path = GT_ROOT / scene / 'after_filter_boxes.npy'
        gts[scene] = np.load(path, allow_pickle=False)
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for method, directory in METHODS.items():
        preds[method] = {}
        for scene in scenes:
            path = ROOT / 'results' / directory / f'{scene}_boxes.pkl'
            with path.open('rb') as stream:
                rows = pickle.load(stream)[0]
            preds[method][scene] = (
                np.asarray([r[1] for r in rows], float).reshape(-1, 8, 3),
                np.asarray([float(r[2]) for r in rows]))
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    # ---- pre-registered pooled tertiles on GT AABB volume ----
    vols = np.concatenate([(g.max(1) - g.min(1)).prod(1) for g in gts.values()])
    q1, q2 = np.quantile(vols, [1 / 3, 2 / 3])
    group_names = ('small', 'medium', 'large')

    def group_of(scene):
        v = (gts[scene].max(1) - gts[scene].min(1)).prod(1)
        return {'small': gts[scene][v <= q1],
                'medium': gts[scene][(v > q1) & (v <= q2)],
                'large': gts[scene][v > q2]}

    protocol = {'tertile_cuts_m3': [float(q1), float(q2)],
                'gt_counts': {}, 'n_boot': N_BOOT, 'seed': 0,
                'thresholds': THRESHOLDS, 'methods': METHODS,
                'subgroup_rule': 'GT restricted to group; cross-group matches count as FP',
                'inputs': 'frozen full107 artifacts; no new inference'}
    for name in group_names:
        protocol['gt_counts'][name] = int(sum(len(group_of(s)[name]) for s in scenes))
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')

    # ---- subgroup AP + per-scene AP ----
    results = {'size_groups': {}, 'per_scene': {}, 'deltas': {}}
    for method in METHODS:
        results['size_groups'][method] = {}
        for thr in THRESHOLDS:
            results['size_groups'][method][str(thr)] = {}
            for name in group_names:
                sub_gts = {s: group_of(s)[name] for s in scenes
                           if len(group_of(s)[name])}
                m = class_agnostic_ap(preds[method], sub_gts, thr)
                results['size_groups'][method][str(thr)][name] = {
                    'AP': m['ap'], 'tp': m['tp'], 'fp': m['fp'],
                    'gt': sum(len(v) for v in sub_gts.values())}
        results['per_scene'][method] = {}
        for thr in THRESHOLDS:
            per = {}
            for s in scenes:
                m = class_agnostic_ap({s: preds[method][s]}, {s: gts[s]}, thr)
                per[s] = m['ap']
            results['per_scene'][method][str(thr)] = per
        print(method, 'size AP25:',
              {k: round(results['size_groups'][method]['0.25'][k]['AP'], 2)
               for k in group_names}, flush=True)

    # ---- paired deltas + bootstrap ----
    for pair in (('M1', 'native'), ('M1_M2', 'M1'), ('M1_M2', 'native')):
        a, b = pair
        key = f'{a}-minus-{b}'
        results['deltas'][key] = {}
        for thr in THRESHOLDS:
            da = np.array([results['per_scene'][a][str(thr)][s] for s in scenes])
            db = np.array([results['per_scene'][b][str(thr)][s] for s in scenes])
            d = da - db
            boots = []
            for _ in range(N_BOOT):
                idx = rng.integers(0, len(scenes), len(scenes))
                boots.append(d[idx].mean())
            results['deltas'][key][str(thr)] = {
                'mean_delta_ap': float(d.mean()),
                'ci95': [float(np.quantile(boots, .025)),
                         float(np.quantile(boots, .975))],
                'scenes_improved': int((d > 1e-9).sum()),
                'scenes_worsened': int((d < -1e-9).sum()),
                'scenes_equal': int((np.abs(d) <= 1e-9).sum()),
                'n_scenes': len(scenes)}
            print(key, thr, results['deltas'][key][str(thr)], flush=True)

    (OUT / 'results.json').write_text(json.dumps(results, indent=1))
    (OUT / 'input_sha256.json').write_text(json.dumps(hashes, indent=1))
    print('DONE', OUT)


if __name__ == '__main__':
    main()
