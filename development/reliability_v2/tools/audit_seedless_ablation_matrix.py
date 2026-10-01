"""Seedless-module ablation matrix: confirmation value at matched birth counts.

The full-scale runs (ca1m_seedless_full107 / scannet_seedless_full100) bundle
candidate addition, Boxer lifting, voxel clustering and >=3-frame confirmation
into one positive number. This audit decomposes the confirmation's own
contribution with per-scene count-matched controls, reusing the runs' frozen
selections, lifted geometry and raw scores (no new inference):

  ge1 / ge2 / ge3    all voxels with >=1/2/3 distinct frames -> exemplar births
                     (dose response at natural counts);
  score_matched      >=1-frame voxels ranked by exemplar raw score, top-N3 per
                     scene where N3 = that scene's ge3 count — "pick by score"
                     at the same pool and the same birth count;
  ge3_dedup          ge3 minus clusters whose exemplar has IoU > 0.5 with a
                     baseline map row (drop near-duplicates, keep upgrades).

Cross-checks: per-scene ge3 must reproduce the run's frozen births_score_m*
files (count and corners), baseline AP parity against the published numbers,
input hashes frozen. GT is used only for evaluation/composition statistics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_ca1m_nms_child_headroom import sha256, valid_boxes
from tools.audit_m1m2_remaining_children import read_prediction
from tools.ca1m_seedless_trigger_core import add_clusters
from tools.true_fusion_audit_core import aabb_iou, class_agnostic_ap
from tools import audit_ca1m_seedless_step1c as pilot

RUNS = {
    'ca1m': ROOT / 'reports/ca1m_seedless_full107_20260911',
    'scannet': ROOT / 'reports/scannet_seedless_full100_20260911',
}
BASELINES = {
    'ca1m': pilot.BASE,
    'scannet': ROOT / 'results/scannet_m2nl_m5_dual_full100/persistent',
}
EXPECTED = {
    'ca1m': [46.1315, 38.3400, 16.8523],
    'scannet': [41.298901361344126, 37.302988324064584, 18.512383520854536],
}
POOLS = ('score_m300', 'score_m150')
ARMS = ('ge1', 'ge2', 'ge3', 'score_matched', 'ge3_dedup')
DEDUP_IOU = .5
APPEND_SCORE = .1
THRESHOLDS = pilot.THRESHOLDS


def build_scannet_eval(protocol):
    """Replicate the run's GT parsing and alignment exactly."""
    sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != ROOT / 'tools']
    sys.path.insert(0, str(ROOT / 'evaluation'))
    from utils.ap_helper import parse_groundtruths
    from utils.utils import flip_axis_to_camera, obb_to_aabb_corners, reorganize_obb_to_aabb
    from data_util.dataset import ScannetDetectionDataset
    from data_util.model_util_scannet import ScannetDatasetConfig
    from torch.utils.data._utils.collate import default_collate
    from tools.eval_scannet_causal_dynamic_ap import read_alignment

    def transform(boxes, align):
        if not len(boxes):
            return np.empty((0, 8, 3))
        value = np.transpose(align[None, :3, :3] @ np.transpose(boxes, (0, 2, 1)), (0, 2, 1))
        value += align[None, :3, 3]
        return reorganize_obb_to_aabb(obb_to_aabb_corners(flip_axis_to_camera(value)))

    dataset = ScannetDetectionDataset('val', num_points=40000, augment=False,
                                      use_color=False, use_height=True,
                                      data_path=str(ROOT / 'evaluation/data_util/scannet_train_detection_data'))
    scenes = protocol['scenes']
    if not set(scenes) <= set(dataset.scan_names):
        raise ValueError('Missing GT scenes')
    dataset.scan_names = scenes
    cfg = {'dataset_config': ScannetDatasetConfig()}
    gts, aligns = {}, {}
    for index, scene in enumerate(scenes):
        meta = Path(protocol['data']) / scene / f'{scene}.txt'
        aligns[scene] = read_alignment(meta)
        gts[scene] = valid_boxes(
            [r[1] for r in parse_groundtruths(default_collate([dataset[index]]), cfg)[0]], scene)
    return gts, aligns, transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=tuple(RUNS), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f'Refusing to overwrite {args.output}')
    run = RUNS[args.dataset]
    protocol = json.loads((run / 'protocol.json').read_text())
    inputs = {str(p.resolve()): sha256(p) for p in (
        Path(__file__), run / 'protocol.json')}
    if args.dataset == 'ca1m':
        gts = {s: valid_boxes(np.load(pilot.GT_ROOT / s / 'after_filter_boxes.npy'),
                              s) for s in protocol['scenes']}
        for s in gts:
            inputs[str((pilot.GT_ROOT / s / 'after_filter_boxes.npy').resolve())] = \
                sha256(pilot.GT_ROOT / s / 'after_filter_boxes.npy')
        transform = lambda boxes, align: boxes
        aligns = {s: None for s in protocol['scenes']}
    else:
        gts, aligns, transform = build_scannet_eval(protocol)
    baseline = {}
    predictions = {f'{pool}_{arm}': {} for pool in POOLS for arm in ARMS}
    composition = {f'{pool}_{arm}': Counter() for pool in POOLS for arm in ARMS}
    counts = {f'{pool}_{arm}': Counter() for pool in POOLS for arm in ARMS}
    for ordinal, scene in enumerate(protocol['scenes'], 1):
        directory = run / 'scenes' / scene
        selection = json.loads((directory / 'selection.json').read_text())
        inputs[str((directory / 'selection.json').resolve())] = sha256(directory / 'selection.json')
        complete = json.loads((directory / 'complete.json').read_text())
        boxes, scores = read_prediction(BASELINES[args.dataset] / f'{scene}_boxes.pkl')
        inputs[str((BASELINES[args.dataset] / f'{scene}_boxes.pkl').resolve())] = \
            sha256(BASELINES[args.dataset] / f'{scene}_boxes.pkl')
        baseline[scene] = (transform(boxes, aligns[scene]), scores)
        gt = gts[scene]
        biou = aabb_iou(transform(boxes, aligns[scene]), gt).max(0)
        for pool in POOLS:
            clusters = {}
            for row in selection['frames']:
                frame = row['frame']
                lift_path = directory / f'lifted_{frame:06d}.npz'
                raw_path = run / 'raw' / scene / f'raw_{frame:06d}.npz'
                with np.load(lift_path, allow_pickle=False) as values:
                    union, corners = values['anchor_ids'], values['corners']
                with np.load(raw_path, allow_pickle=False) as values:
                    raw_scores = values['scores']
                ids = np.asarray(row['selected_anchor_ids'][pool], dtype=np.int64)
                positions = np.searchsorted(union, ids)
                if not np.array_equal(union[positions], ids):
                    raise ValueError(f'Anchor mismatch: {scene}/{frame}/{pool}')
                add_clusters(clusters, frame, ids, corners[positions], raw_scores[ids], .3)
                for path in (lift_path, raw_path):
                    inputs[str(path.resolve())] = sha256(path)
            states = sorted(clusters.values(), key=lambda c: c['rank'])
            by_frames = {minimum: [c for c in states if len(c['frames']) >= minimum]
                         for minimum in (1, 2, 3)}
            with np.load(directory / f'births_{pool}.npz', allow_pickle=False) as values:
                frozen = valid_boxes(values['corners'], f'{scene}/{pool}')
                frozen_keys = [tuple(v) for v in values['voxel_keys'].tolist()]
            ge3_by_key = {key: c for key, c in clusters.items() if len(c['frames']) >= 3}
            if (sorted(ge3_by_key) != frozen_keys
                    or not np.array_equal(
                        np.asarray([ge3_by_key[k]['box'] for k in frozen_keys]
                                   ).reshape(-1, 8, 3), frozen)):
                raise ValueError(f'ge3 does not reproduce frozen births: {scene}/{pool}')
            ge3 = by_frames[3]
            arm_boxes = {
                'ge1': [c['box'] for c in by_frames[1]],
                'ge2': [c['box'] for c in by_frames[2]],
                'ge3': [c['box'] for c in ge3],
                'score_matched': [c['box'] for c in by_frames[1][:len(ge3)]],
                'ge3_dedup': [c['box'] for c in ge3
                              if not len(boxes) or aabb_iou(
                                  c['box'][None], boxes).max() <= DEDUP_IOU],
            }
            assert len(arm_boxes['score_matched']) == len(arm_boxes['ge3'])
            for arm, born in arm_boxes.items():
                born = np.asarray(born, dtype=float).reshape(-1, 8, 3)
                name = f'{pool}_{arm}'
                counts[name]['births'] += len(born)
                if arm != 'ge1' and arm != 'ge2':
                    iou = aabb_iou(transform(born, aligns[scene]), gt)
                    owner = iou.argmax(1)
                    matched = iou.max(1) > .15 if len(gt) else np.zeros(len(born), dtype=bool)
                    composition[name]['matches_existing'] += int((matched & (biou[owner] > .15)).sum())
                    composition[name]['matches_missed'] += int((matched & (biou[owner] <= .15)).sum())
                    composition[name]['unmatched'] += int((~matched).sum())
                predictions[name][scene] = (
                    transform(np.concatenate([boxes, born])
                              if len(born) else boxes, aligns[scene]),
                    np.r_[scores, np.full(len(born), APPEND_SCORE)] if len(born) else scores)
        if ordinal % 10 == 0 or ordinal == len(protocol['scenes']):
            print(f'{args.dataset} {ordinal}/{len(protocol["scenes"])} scenes', flush=True)
    metrics = {'baseline': {str(t): class_agnostic_ap(baseline, gts, t) for t in THRESHOLDS}}
    for name, arms in predictions.items():
        metrics[name] = {}
        for t in THRESHOLDS:
            value = class_agnostic_ap(arms, gts, t)
            for key in ('ap', 'tp', 'fp'):
                value['delta_' + key] = value[key] - metrics['baseline'][str(t)][key]
            metrics[name][str(t)] = value
    actual = [metrics['baseline'][str(t)]['ap'] for t in THRESHOLDS]
    assert np.allclose(actual, EXPECTED[args.dataset], atol=5e-4, rtol=0), actual
    for path, digest in inputs.items():
        assert sha256(path) == digest, f'Input changed: {path}'
    result = {
        'schema': 'boxfusion.seedless_ablation_matrix.v1', 'completed': True,
        'dataset': args.dataset, 'run': str(run),
        'design': {'pool': 'score-prefix top-M per frame (frozen run selections)',
                   'count_matching': 'score_matched uses each scene ge3 birth count',
                   'dedup': f'exemplar-vs-baseline max IoU <= {DEDUP_IOU}'},
        'counts': {k: dict(v) for k, v in counts.items()},
        'composition': {k: dict(v) for k, v in composition.items()},
        'metrics': metrics,
        'limits': [
            'Ablations are offline post-processing of the frozen runs; the causal '
            'online form, memory bounds and FPS are unchanged open items',
            'score_matched shares the voxel exemplar machinery; it removes only the '
            'distinct-frame requirement, so it also tests exemplar-vs-frames value',
            'GT used for evaluation and composition statistics only',
            'Development/validation scenes already used in tuning; not held out.',
        ],
    }
    args.output.mkdir(parents=True)
    for name, data in (('results.json', json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'),
                       ('input_sha256.json', json.dumps(inputs, ensure_ascii=False, indent=2) + '\n')):
        with (args.output / name).open('x') as handle:
            handle.write(data)
    print(json.dumps({
        'deltas': {name: [round(metrics[name][str(t)]['delta_ap'], 4) for t in THRESHOLDS]
                   for name in predictions},
        'births': {k: v['births'] for k, v in counts.items()},
        'composition': {k: dict(v) for k, v in composition.items()}},
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
