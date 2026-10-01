"""ScanNet tier-2 transfer onto the M5-free (m1+m2nl) baseline, 41.26 lineage.

The seedless ScanNet numbers were measured on scannet_m2nl_m5_dual_full100
(tier1 + M2nl + M5, 41.2989). The user's chosen stack baseline is the M5-free
41.26 lineage whose predictions were never saved. This audit reconstructs the
M5-free scores from the frozen persistent outputs (per-row m5_retired flags in
the dual_state sidecars; the retirement rule multiplies the persistent score
by 0.3 exactly once) and validates the reconstruction against the known
41.26/37.26/18.47 before appending anything. On validation failure it aborts.

Arms (births are baseline-independent frozen artifacts of the full100 run):
  ge3_flat010        frozen module form: >=3-frame voxel exemplars at 0.10;
  scorematched005    ablation-winning selection at flat 0.05.
GT is used only for evaluation. Offline post-processing, no new inference.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_ca1m_nms_child_headroom import sha256, valid_boxes
from tools.audit_m1m2_remaining_children import read_prediction
from tools.ca1m_seedless_trigger_core import add_clusters
from tools.true_fusion_audit_core import class_agnostic_ap
from tools.audit_seedless_ablation_matrix import RUNS, THRESHOLDS, build_scannet_eval
from tools.audit_seedless_birth_pricing import size_price  # noqa: F401 (reuse import path)

RUN = RUNS['scannet']
BASE = ROOT / 'results/scannet_m2nl_m5_dual_full100/persistent'
EXPECTED_M5FREE = (41.26, 37.26, 18.47)
M5_FACTOR = 0.3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f'Refusing to overwrite {args.output}')
    protocol = json.loads((RUN / 'protocol.json').read_text())
    inputs = {str(p.resolve()): sha256(p) for p in (Path(__file__), RUN / 'protocol.json')}
    gts, aligns, transform = build_scannet_eval(protocol)
    baseline, retired_counts = {}, Counter()
    for scene in protocol['scenes']:
        pkl = BASE / f'{scene}_boxes.pkl'
        side_path = Path(str(pkl) + '.dual_state.json')
        for path in (pkl, side_path):
            inputs[str(path.resolve())] = sha256(path)
        boxes, scores = read_prediction(pkl)
        side = json.loads(side_path.read_text())
        assert side['scene_id'] == scene and side['row_count'] == len(boxes)
        assert [r['persistent_score'] for r in side['rows']] == scores.tolist()
        restored = scores.copy()
        for index, row in enumerate(side['rows']):
            if row['m5_retired']:
                retired_counts['rows'] += 1
                restored[index] = scores[index] / M5_FACTOR
        retired_counts['scenes_with_retired'] += bool(side['rows']
                                                       and any(r['m5_retired'] for r in side['rows']))
        baseline[scene] = (transform(boxes, aligns[scene]), restored)
    metrics0 = {str(t): class_agnostic_ap(baseline, gts, t)['ap'] for t in THRESHOLDS}
    actual = [metrics0[str(t)] for t in THRESHOLDS]
    if not np.allclose(actual, EXPECTED_M5FREE, atol=.011, rtol=0):
        raise ValueError(f'M5-free reconstruction mismatch: {actual}')
    arms = {'ge3_flat010': {}, 'scorematched005': {}}
    stats = {name: Counter() for name in arms}
    for scene in protocol['scenes']:
        directory = RUN / 'scenes' / scene
        selection = json.loads((directory / 'selection.json').read_text())
        inputs[str((directory / 'selection.json').resolve())] = sha256(directory / 'selection.json')
        pkl = BASE / f'{scene}_boxes.pkl'
        boxes, scores = read_prediction(pkl)
        side = json.loads(Path(str(pkl) + '.dual_state.json').read_text())
        restored = scores.copy()
        for index, row in enumerate(side['rows']):
            if row['m5_retired']:
                restored[index] = scores[index] / M5_FACTOR
        for pool in ('score_m300',):
            clusters = {}
            for row in selection['frames']:
                frame = row['frame']
                lift_path = directory / f'lifted_{frame:06d}.npz'
                raw_path = RUN / 'raw' / scene / f'raw_{frame:06d}.npz'
                with np.load(lift_path, allow_pickle=False) as values:
                    union, corners = values['anchor_ids'], values['corners']
                with np.load(raw_path, allow_pickle=False) as values:
                    raw_scores = values['scores']
                ids = np.asarray(row['selected_anchor_ids'][pool], dtype=np.int64)
                positions = np.searchsorted(union, ids)
                if not np.array_equal(union[positions], ids):
                    raise ValueError(f'Anchor mismatch: {scene}/{frame}')
                add_clusters(clusters, frame, ids, corners[positions], raw_scores[ids], .3)
                for path in (lift_path, raw_path):
                    inputs[str(path.resolve())] = sha256(path)
            states = sorted(clusters.values(), key=lambda c: c['rank'])
            ge3 = [c['box'] for c in states if len(c['frames']) >= 3]
            matched = [c['box'] for c in states[:len(ge3)]]
            born = {'ge3_flat010': (ge3, 0.10), 'scorematched005': (matched, 0.05)}
            for name, (extra, price) in born.items():
                stats[name]['births'] += len(extra)
                arms[name][scene] = (
                    transform(np.concatenate([boxes, np.asarray(extra)])
                              if extra else boxes, aligns[scene]),
                    np.r_[restored, np.full(len(extra), price)] if extra else restored)
    metrics = {'baseline_m5free': {str(t): class_agnostic_ap(baseline, gts, t)
                                   for t in THRESHOLDS}}
    for name, predictions in arms.items():
        metrics[name] = {}
        for t in THRESHOLDS:
            value = class_agnostic_ap(predictions, gts, t)
            for key in ('ap', 'tp', 'fp'):
                value['delta_' + key] = value[key] - metrics['baseline_m5free'][str(t)][key]
            metrics[name][str(t)] = value
    for path, digest in inputs.items():
        assert sha256(path) == digest, f'Input changed: {path}'
    result = {
        'schema': 'boxfusion.seedless_scannet_nom5_transfer.v1', 'completed': True,
        'baseline_reconstruction': {
            'rule': f'persistent_score / {M5_FACTOR} on m5_retired rows',
            'validated_against': list(EXPECTED_M5FREE),
            'reconstructed': actual,
            'retired_rows_total': retired_counts['rows'],
            'scenes_with_retired': retired_counts['scenes_with_retired']},
        'arms': {'ge3_flat010': 'frozen module form (>=3-frame births at 0.10)',
                 'scorematched005': 'score-ranked same-count selection at 0.05'},
        'stats': {k: dict(v) for k, v in stats.items()},
        'metrics': metrics,
        'limits': ['M5-free baseline reconstructed by inverting the one-shot x0.3 '
                   'demotion and validated to 0.011 AP against the published 41.26',
                   'Offline post-processing of frozen artifacts; GT for evaluation only'],
    }
    args.output.mkdir(parents=True)
    for name, data in (('results.json', json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'),
                       ('input_sha256.json', json.dumps(inputs, ensure_ascii=False, indent=2) + '\n')):
        with (args.output / name).open('x') as handle:
            handle.write(data)
    print(json.dumps({
        'reconstruction': result['baseline_reconstruction'],
        'deltas': {name: [round(metrics[name][str(t)]['delta_ap'], 4) for t in THRESHOLDS]
                   for name in arms}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
