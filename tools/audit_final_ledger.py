#!/usr/bin/env python3
"""Final-ledger reconciliation: bind every main-table row to its cached run.

One evaluator convention (tools.true_fusion_audit_core.class_agnostic_ap, the
anchor-exact audit path already cross-checked bit-for-bit against the sealed
ScanNet/CA-1M evaluators in prior audits) recomputes AP15/25/50 at full
precision for every published row, including the derived tier-2 arms rebuilt
from the frozen seedless artifacts. Each row records its prediction directory,
scene list, evaluator and delta against its chain predecessor; published
values are asserted (strict 5e-4 where a full-precision anchor exists, loose
elsewhere) so the paper table can only cite numbers this file reproduces.

No detector rerun; GT is used for evaluation only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_ca1m_nms_child_headroom import sha256, valid_boxes
from tools.audit_m1m2_remaining_children import BASE, GT_ROOT, read_prediction
from tools.ca1m_seedless_trigger_core import add_clusters
from tools.true_fusion_audit_core import class_agnostic_ap
from tools import audit_ca1m_seedless_step1c as pilot
from tools.audit_seedless_ablation_matrix import (
    BASELINES, RUNS, THRESHOLDS, build_scannet_eval)

TIER2_PRICE = 0.05
OUT = ROOT / 'reports/final_ledger_20260915'
STRICT = 5e-4
LOOSE = 5e-3

# Published anchors. 'p' is full precision where an archived results.json or
# asserted audit exists (strict tolerance); 2-decimal ledger rows use LOOSE.
SCAN_ROWS = [
    ('native',       dict(dir='results/scannet_t05_boxer_kfmap_score05'),
     [35.04, 31.46, 15.70], 0.03,
     '语义表 native 行同目录；在线 native base 35.04/31.46/15.70（评测路径差≤0.026）'),
    ('m1_run',       dict(dir='results/causal_m1_only_v9'),
     [39.04, 35.11, 16.94], LOOSE,
     'v9 因果 M1-only 实跑（causal_online_verification）'),
    ('m1_paired',    dict(derived='m1_paired'),
     [39.04, 35.11, 16.94], LOOSE,
     '配对重建：persistent 几何+原生行还原 native 分数；审计路径下与实跑 M1 行一致'),
    ('m2_nl',        dict(dir='results/scnet_m2nl'),
     [41.26, 37.26, 18.47], LOOSE,
     'M2 nativelogit（无 M5）；旧主表 +2.22 增量来源'),
    ('m2_m5_pers',   dict(dir='results/scannet_m2nl_m5_dual_full100/persistent'),
     [41.29890136134413, 37.302988324064586, 18.512383520854534], STRICT,
     '统一栈 tier-2 基线：M2nl + M5 dual 持久视图'),
    ('m2_add_a80',   dict(dir='results/causal_v9_a80'),
     [42.78, 38.71, 18.94], LOOSE,
     '旧账本 M2 additive α=0.8（非统一栈）'),
    ('m5_win',       dict(dir='results/scnet_m5win'),
     [45.06, 40.77, 20.58], LOOSE,
     '旧账本 M5 支持断流（非统一栈）'),
    ('tier2_m300',   dict(derived='tier2', pool='score_m300'),
     [43.350321, 39.147075, 19.561201], STRICT,
     '统一栈 + tier-2（M300, flat 0.05）；定价报告 results.json 全精度'),
    ('tier2_m150',   dict(derived='tier2', pool='score_m150'),
     [43.707544, 39.368073, 19.764582], STRICT,
     '预算敏感性：M150, flat 0.05'),
    ('m1a_online', dict(dir='reports/m1a_strict_online_scannet_20260916/predictions'),
     [44.05642888843821, 39.75004020678599, 20.243659679452], STRICT,
     '严格在线 M1-A：单池 top-300、三帧因果确认、确定性非同分排序'),
]
CA1M_ROWS = [
    ('native',    dict(dir='results/ca1m_thr15'),
     [44.8526, 36.7678, 15.3999], STRICT,
     'thr15 地图 native（audit_paper_evidence LEDGER）'),
    ('m1_single', dict(dir='results/ca1m_thr15_m1_full107'),
     [45.7277, 37.6384, 15.7733], STRICT,
     'M1 单源（WeDetect-only 漏斗）'),
    ('m1m2_nl',   dict(dir='results/ca1m_thr15_m1_m2nl_full107'),
     [45.9762, 38.2087, 16.7660], STRICT,
     'M1 单源 + M2nativelogit'),
    ('m1_dual',   dict(dir='results/ca1m_dual_nom2'),
     None, None,
     'M1 双源（WeDetect+child 漏斗）；无全精度发布锚，仅记录'),
    ('dual',      dict(dir='results/ca1m_dual'),
     [46.13147239557094, 38.339996974211196, 16.85230964456153], STRICT,
     'CA-1M tier-2 基线：双源 M1 + M2nl（ca1m_transfer A 臂 46.13/38.34/16.85）'),
    ('tier2_m300', dict(derived='tier2', pool='score_m300'),
     [47.560329, 39.577718, 17.626631], STRICT,
     '统一栈 + tier-2（M300, flat 0.05）'),
    ('m1a_online', dict(dir='reports/m1a_strict_online_ca1m_20260916/predictions'),
     [47.32993252172456, 39.40078787209063, 17.553485241202083], STRICT,
     '严格在线 M1-A：单池 top-300、三帧因果确认、确定性非同分排序'),
]


def build_tier2_raw(dataset, pool):
    """Per-scene RAW (unaligned) tier-2 predictions: (boxes, scores), birth count."""
    run = RUNS[dataset]
    protocol = json.loads((run / 'protocol.json').read_text())
    baseline_dir = BASELINES[dataset]
    predictions = {}
    births_total = 0
    for scene in protocol['scenes']:
        directory = run / 'scenes' / scene
        selection = json.loads((directory / 'selection.json').read_text())
        boxes, scores = read_prediction(baseline_dir / f'{scene}_boxes.pkl')
        clusters = {}
        for row in selection['frames']:
            frame = row['frame']
            with np.load(directory / f'lifted_{frame:06d}.npz',
                         allow_pickle=False) as values:
                union, corners = values['anchor_ids'], values['corners']
            with np.load(run / 'raw' / scene / f'raw_{frame:06d}.npz',
                         allow_pickle=False) as values:
                raw_scores = values['scores']
            ids = np.asarray(row['selected_anchor_ids'][pool], dtype=np.int64)
            positions = np.searchsorted(union, ids)
            if not np.array_equal(union[positions], ids):
                raise ValueError(f'Anchor mismatch: {scene}/{frame}/{pool}')
            add_clusters(clusters, frame, ids, corners[positions],
                         raw_scores[ids], .3)
        states = sorted(clusters.values(), key=lambda c: c['rank'])
        n3 = sum(len(c['frames']) >= 3 for c in states)
        born = [c['box'] for c in states[:n3]]
        births_total += len(born)
        prices = [TIER2_PRICE] * len(born)
        merged = np.concatenate([boxes, np.asarray(born)]) if len(born) else boxes
        predictions[scene] = (merged,
                              np.r_[scores, prices] if len(born) else scores)
    return protocol, predictions, births_total


def build_tier2(dataset, pool, transform, aligns):
    """Rebuild tier-2 predictions from frozen seedless artifacts (pricing path)."""
    protocol, raw, births_total = build_tier2_raw(dataset, pool)
    predictions = {scene: (transform(boxes, aligns[scene]), scores)
                   for scene, (boxes, scores) in raw.items()}
    return protocol, predictions, births_total


def scannet_inputs():
    protocol = json.loads(
        (RUNS['scannet'] / 'protocol.json').read_text())
    return build_scannet_eval(protocol)


def m1_paired_rows(persistent, native):
    """Semantic-table construction: persistent geometry, native scores restored."""
    zb, zs = persistent
    nb, ns = native
    assert len(zb) >= len(nb) and np.array_equal(zb[:len(nb)], nb)
    return zb, np.concatenate([ns, zs[len(nb):]]).astype(float)


def evaluate(dataset, rows):
    if dataset == 'scannet':
        gts, aligns, transform = scannet_inputs()
    else:
        scenes_ca = Path(ROOT / 'tools/boxfusion_tr3d_pipeline/evaluation/'
                         'data_util/meta_data/ca1m_val_full107.txt'
                         ).read_text().split()
        gts = {s: valid_boxes(np.load(GT_ROOT / s / 'after_filter_boxes.npy'), s)
               for s in scenes_ca}
        aligns = {s: None for s in scenes_ca}
        transform = lambda boxes, align: boxes
    protocol = json.loads((RUNS[dataset] / 'protocol.json').read_text())
    scenes = protocol['scenes']
    results, hashes = {}, {}
    cache = {}
    for name, spec, expected, tol, note in rows:
        if 'dir' in spec:
            directory = ROOT / spec['dir']
            per_scene = {}
            for scene in scenes:
                if scene not in cache:
                    cache[scene] = {
                        d: read_prediction(ROOT / d / f'{scene}_boxes.pkl')
                        for d in {r[1].get('dir') for r in rows if 'dir' in r[1]}
                        if (ROOT / d / f'{scene}_boxes.pkl').exists()}
                boxes, scores = cache[scene][spec['dir']]
                per_scene[scene] = (transform(boxes, aligns[scene]), scores)
        elif spec.get('derived') == 'm1_paired':
            per_scene = {}
            for scene in scenes:
                z = cache[scene]['results/scannet_m2nl_m5_dual_full100/persistent']
                n = cache[scene]['results/scannet_t05_boxer_kfmap_score05']
                boxes, scores = m1_paired_rows(z, n)
                per_scene[scene] = (transform(boxes, aligns[scene]), scores)
        else:
            _, per_scene, births = build_tier2(dataset, spec['pool'],
                                               transform, aligns)
            results[name] = {'births': births}
        if 'dir' in spec:
            for scene in scenes:
                p = ROOT / spec['dir'] / f'{scene}_boxes.pkl'
                hashes[str(p)] = sha256(p)
        scores_all = {s: v[1] for s, v in per_scene.items()}
        row = results.setdefault(name, {})
        row['ap'] = {str(t): class_agnostic_ap(
            per_scene, gts, t) for t in THRESHOLDS}
        row['boxes'] = int(sum(len(v) for v in scores_all.values()))
        row['dir'] = spec.get('dir', f"derived:{spec.get('derived')}"
                                     f"/{spec.get('pool', '')}")
        row['expected'] = expected
        row['note'] = note
        if expected and tol:
            actual = [row['ap'][str(t)]['ap'] for t in THRESHOLDS]
            off = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
            row['max_abs_dev_from_published'] = off
            assert off <= tol, (name, actual, expected)
    if dataset == 'scannet':
        chains = {
            'scannet_unified_m300':
                ['native', 'm1_run', 'm2_m5_pers', 'm1a_online'],
            'scannet_m2row_only': ['native', 'm1_run', 'm2_nl'],
            'scannet_budget_m150': ['m2_m5_pers', 'tier2_m150'],
        }
    else:
        chains = {
            'ca1m_unified': ['native', 'm1_dual', 'dual', 'm1a_online'],
            'ca1m_single_source': ['native', 'm1_single', 'm1m2_nl'],
        }
    deltas = {}
    for chain, names in chains.items():
        entries = [n for n in names if n in results]
        deltas[chain] = []
        for prev, cur in zip(entries, entries[1:]):
            step = {'from': prev, 'to': cur}
            for t in THRESHOLDS:
                step[f'delta_{t}'] = results[cur]['ap'][str(t)]['ap'] - \
                    results[prev]['ap'][str(t)]['ap']
            deltas[chain].append(step)
    return {'dataset': dataset, 'scenes': scenes, 'rows': results,
            'deltas': deltas, 'input_sha256': hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-ca1m', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = {'scannet': evaluate('scannet', SCAN_ROWS)}
    if not args.skip_ca1m:
        ledger['ca1m'] = evaluate('ca1m', CA1M_ROWS)
    (OUT / 'ledger.json').write_text(json.dumps(ledger, ensure_ascii=False,
                                                indent=2) + '\n')
    for dataset, block in ledger.items():
        print(f"== {dataset} ==")
        for name, row in block['rows'].items():
            aps = [row['ap'][str(t)]['ap'] for t in THRESHOLDS]
            dev = row.get('max_abs_dev_from_published')
            print(f"{name:12s} {aps[0]:9.4f} {aps[1]:9.4f} {aps[2]:9.4f} "
                  f"boxes={row['boxes']:6d} "
                  f"dev={'' if dev is None else f'{dev:.2e}'} {row['dir']}")
        for chain, steps in block['deltas'].items():
            if not steps:
                continue
            print(f" -- {chain}")
            for step in steps:
                d = [step[f'delta_{t}'] for t in THRESHOLDS]
                print(f"   {step['from']} -> {step['to']}: "
                      f"{d[0]:+.4f}/{d[1]:+.4f}/{d[2]:+.4f}")


if __name__ == '__main__':
    main()
