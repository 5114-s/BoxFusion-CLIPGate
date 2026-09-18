#!/usr/bin/env python3
"""Size-group AP v3: prefer eligible in-group GT, otherwise ignore any
out-of-group overlap, including duplicates; retain all other false positives.
Historical bootstrap CI describes scene-mean differences, not pooled AP.
Same frozen arms; no new inference. Filename retained for reproducibility.
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
from paper_eval_core import subgroup_ap

METHODS = {'native': 'ca1m_thr15', 'M1': 'ca1m_thr15_m1_full107',
           'M1_M2': 'ca1m_thr15_m1_m2nl_full107'}
THRESHOLDS = (0.15, 0.25, 0.5)
OUT = ROOT / 'reports/sizegroup_bootstrap_20260915'
GT_ROOT = Path('/extra/ZhaoX/boxfusion_ca1m')


def main():
    scenes = sorted(p.name.replace('_boxes.pkl', '')
                    for p in (ROOT / 'results/ca1m_thr15').glob('*_boxes.pkl'))
    scenes = [s for s in scenes if (GT_ROOT / s / 'after_filter_boxes.npy').is_file()]
    assert len(scenes) == 107
    gts, preds = {}, {}
    for scene in scenes:
        gts[scene] = np.load(GT_ROOT / scene / 'after_filter_boxes.npy')
    for method, directory in METHODS.items():
        preds[method] = {}
        for scene in scenes:
            with open(ROOT / 'results' / directory / f'{scene}_boxes.pkl', 'rb') as f:
                rows = pickle.load(f)[0]
            preds[method][scene] = (
                np.asarray([r[1] for r in rows], float).reshape(-1, 8, 3),
                np.asarray([float(r[2]) for r in rows]))

    vols_all = np.concatenate([(g.max(1) - g.min(1)).prod(1) for g in gts.values()])
    q1, q2 = np.quantile(vols_all, [1 / 3, 2 / 3])
    groups = {}
    for scene in scenes:
        v = (gts[scene].max(1) - gts[scene].min(1)).prod(1)
        groups[scene] = {'small': v <= q1, 'medium': (v > q1) & (v <= q2),
                         'large': v > q2}

    results = {'size_groups_v3_ignore_outgroup': {}, 'tertile_cuts_m3': [q1,q2],
               'protocol_note': 'in-group eligible GT prioritized; otherwise out-group overlap '
               'ignored including duplicates; unmatched FP retained',
               'bootstrap_note': 'existing CI is scene-mean difference, not pooled AP',
               'all_group_anchor_parity': {}, 'inputs_sha256': {}}
    (OUT/'sizegroup_v3_protocol.json').write_text(json.dumps(
        {k:results[k] for k in ('tertile_cuts_m3','protocol_note','bootstrap_note')},indent=2))
    for scene in scenes:
        for path in [GT_ROOT/scene/'after_filter_boxes.npy'] + [
                ROOT/'results'/d/f'{scene}_boxes.pkl' for d in METHODS.values()]:
            results['inputs_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for method in METHODS:
        results['size_groups_v3_ignore_outgroup'][method] = {}
        for thr in THRESHOLDS:
            full = subgroup_ap(preds[method],gts,{s:np.ones(len(g),bool) for s,g in gts.items()},thr)
            anchor = class_agnostic_ap(preds[method],gts,thr)
            assert abs(full['ap']-anchor['ap']) < 1e-10
            assert (full['tp'],full['fp']) == (anchor['tp'],anchor['fp'])
            results['all_group_anchor_parity'][f'{method}/{thr}'] = full['ap']
            results['size_groups_v3_ignore_outgroup'][method][str(thr)] = {}
            for name in ('small', 'medium', 'large'):
                m = subgroup_ap(preds[method],gts,{s:groups[s][name] for s in scenes},thr)
                results['size_groups_v3_ignore_outgroup'][method][str(thr)][name] = m
        print(method, 'AP25:',
              {k: round(results['size_groups_v3_ignore_outgroup'][method]['0.25'][k]['ap'], 2)
               for k in ('small', 'medium', 'large')}, flush=True)

    results['input_hashes_unchanged'] = all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
                                         for p,h in results['inputs_sha256'].items())
    assert results['input_hashes_unchanged']
    (OUT / 'sizegroup_v3_results.json').write_text(json.dumps(results, indent=1))
    print('DONE')


if __name__ == '__main__':
    main()
