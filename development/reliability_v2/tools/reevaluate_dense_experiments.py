#!/usr/bin/env python3
"""Re-score saved dense experiments with the established full107 AABB/VOC AP."""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from true_fusion_audit_core import class_agnostic_ap

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
GT_ROOT = Path('/extra/ZhaoX/boxfusion_ca1m')
OUT = REPORTS / 'dense_eval_review_20260915'
THRESHOLDS = (.15, .25, .5)


def read_prediction(path):
    with path.open('rb') as stream:
        rows = pickle.load(stream)[0]
    boxes = np.asarray([r[1] for r in rows], dtype=np.float64).reshape(-1, 8, 3)
    scores = np.asarray([float(r[2]) for r in rows], dtype=np.float64)
    return boxes, scores


def main():
    capture = REPORTS / 'ca1m_lifting_factors_full107_20260908/capture/protocol.json'
    scenes = json.loads(capture.read_text())['scenes']
    assert len(scenes) == len(set(scenes)) == 107
    directories = {arm: REPORTS / 'dense_oracle_ceiling_20260914' / arm
                   for arm in ('base', 'A', 'B', 'AB')}
    directories.update({v: REPORTS / f'dense_separation_{v}_20260915'
                        for v in ('v1', 'v2b', 'v2c')})
    expected = (45.9762, 38.2087, 16.7660)
    ground_truth, hashes = {}, {}
    for scene in scenes:
        path = GT_ROOT / scene / 'after_filter_boxes.npy'
        ground_truth[scene] = np.load(path, allow_pickle=False)
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sum(map(len, ground_truth.values())) == 12911
    metrics, counts = {}, {}
    baseline = None
    for arm, directory in directories.items():
        predictions = {}
        for scene in scenes:
            path = directory / f'{scene}_boxes.pkl'
            predictions[scene] = read_prediction(path)
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            if arm == 'base':
                original_path = ROOT / 'results/ca1m_thr15_m1_m2nl_full107' / path.name
                original = read_prediction(original_path)
                hashes[str(original_path)] = hashlib.sha256(original_path.read_bytes()).hexdigest()
                assert all(np.array_equal(a, b) for a, b in zip(predictions[scene], original))
            elif arm == 'B':
                assert len(predictions[scene][0]) == len(baseline[scene][0])
                assert np.array_equal(predictions[scene][1], baseline[scene][1])
        if arm == 'base':
            baseline = predictions
        metrics[arm] = {str(t): class_agnostic_ap(predictions, ground_truth, t) for t in THRESHOLDS}
        if arm == 'base':
            assert np.allclose([metrics[arm][str(t)]['ap'] for t in THRESHOLDS],
                               expected, atol=.0001, rtol=0), metrics[arm]
        for t in THRESHOLDS:
            metrics[arm][str(t)]['delta_ap'] = metrics[arm][str(t)]['ap'] - metrics['base'][str(t)]['ap']
        counts[arm] = sum(len(p[0]) for p in predictions.values())
        print(arm, json.dumps({str(t): round(metrics[arm][str(t)]['ap'], 4) for t in THRESHOLDS}), flush=True)
    skipped = [s for s in scenes if not (GT_ROOT / s / 'instances.json').exists()]
    assert len(skipped) == 36 and all(s in ground_truth for s in skipped)
    for path, expected_hash in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected_hash, path
    result = dict(
        protocol=dict(scene_count=107, gt_count=12911, scenes_skipped=0,
                      iou='world-corner AABB; strict > threshold',
                      ap='pooled score sorting, best-GT greedy matching, VOC precision envelope',
                      source='tools/true_fusion_audit_core.py:class_agnostic_ap',
                      prediction_mutation=False, models_run=False),
        verification=dict(original_baseline_rows_scores_exact=True,
                          baseline_historical_AP_parity=True, B_count_and_score_preserved=True,
                          input_hashes_unchanged=True),
        incorrectly_skipped_despite_gt_arrays=skipped,
        predictions=counts, metrics=metrics,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       (Path(__file__), ROOT / 'tools/true_fusion_audit_core.py', ROOT / 'tools/eval_ca1m.py')},
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    (OUT / 'input_sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')


if __name__ == '__main__':
    main()
