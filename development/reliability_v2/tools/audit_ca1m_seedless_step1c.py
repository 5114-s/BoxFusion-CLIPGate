"""Step 1c: final-map residual + 2D recurrence, with matched score controls.

Selection is GT-free: projected-final-map IoU < .2, depth in metres -> world
point -> residual centre within 30 RGB pixels in >=2 other frames; top-M
raw score AFTER the binary geometry gate. M=300/150 are nested prefixes.
All selections and lifted outputs are saved before GT is read. Evaluation
reports oracle coverage/AP and the full burden of >=3-frame voxel births.
Final-map masking and all-frame recurrence are offline, not causal online.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_ca1m_nms_child_headroom import sha256, valid_boxes
from tools.true_fusion_audit_core import aabb_iou, class_agnostic_ap
from tools.audit_m1m2_remaining_children import BASE, GT_ROOT, THRESHOLDS, read_prediction
from tools.ca1m_prenms_query_core import box_iou2d, project_box
from tools.validate_ca1m_prenms_query import build_lifter, lift, load_frame
from tools.ca1m_seedless_trigger_core import (
    add_clusters, anchor_world_points, recurrence_counts, top_m,
    update_coverage, validate_grids,
)

PILOT = ROOT / 'reports/ca1m_prenms_query_pilot_20260908'
DATA_ROOT = Path('/extra/ZhaoX/boxfusion_ca1m')
SCENES = ('42446540', '42897501', '42897521')
RESIDUAL_IOU = .2
RECURRENCE_RADIUS_PX = 30.0
RECURRENCE_OTHER_FRAMES = 2
BUDGETS = (300, 150)
VARIANTS = tuple(f'{kind}_m{m}' for kind in ('trigger', 'score') for m in BUDGETS)
APPEND_SCORE = .10
CLUSTER_VOXEL = .3


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def select_scene(scene, output, inputs):
    """Selection and provenance only. This function never loads GT."""
    start = time.perf_counter()
    root = DATA_ROOT / scene

    def remember(path):
        inputs[str(path.resolve())] = sha256(path)
        return path

    boxes, scores = read_prediction(remember(BASE / f'{scene}_boxes.pkl'))
    poses = np.load(remember(root / 'all_poses.npy'), allow_pickle=False)
    k_rgb = np.loadtxt(remember(root / 'K_rgb.txt')).reshape(3, 3)
    k_depth = np.loadtxt(remember(root / 'K_depth.txt')).reshape(3, 3)
    pools = sorted((PILOT / scene).glob('raw_*.npz'),
                   key=lambda p: int(p.stem.split('_')[1]))
    frames = [int(p.stem.split('_')[1]) for p in pools]
    if not frames or len(frames) != len(set(frames)):
        raise ValueError(f'Empty or duplicate frame list: {scene}')
    anchors, ascores, shapes, trees, points, stats = {}, {}, {}, {}, {}, {}
    world_to_camera = {}
    for path, frame in zip(pools, frames):
        with np.load(remember(path), allow_pickle=False) as raw:
            anchors[frame] = raw['boxes'].astype(float)
            ascores[frame] = raw['scores'].astype(float)
        if not np.isfinite(anchors[frame]).all() or not np.isfinite(ascores[frame]).all():
            raise ValueError('Nonfinite raw anchors/scores')
        remember(root / 'rgb' / f'{frame}.png')
        remember(root / 'depth' / f'{frame}.png')
        _, rgb, depth_m = load_frame(root, frame)
        height, width = validate_grids(rgb.shape, depth_m, k_rgb, k_depth)
        shapes[frame] = (height, width)
        world_to_camera[frame] = np.linalg.inv(poses[frame])
        centers = (anchors[frame][:, :2] + anchors[frame][:, 2:]) / 2
        valid = ((anchors[frame][:, 2:] > anchors[frame][:, :2]).all(1)
                 & (centers[:, 0] >= 0) & (centers[:, 0] < width)
                 & (centers[:, 1] >= 0) & (centers[:, 1] < height))
        projected = [project_box(row, poses[frame], k_rgb, width, height) for row in boxes]
        projected = np.asarray([p for p in projected if p is not None]).reshape(-1, 4)
        residual = valid & ((box_iou2d(anchors[frame], projected).max(1) < RESIDUAL_IOU)
                            if len(projected) else True)
        rc = centers[residual]
        trees[frame] = cKDTree(rc) if len(rc) else None
        ids, worlds, sampled_depths, depth_stats = anchor_world_points(
            centers, np.flatnonzero(residual), depth_m, poses[frame],
            k_rgb, k_depth, rgb.shape)
        points[frame] = (ids, worlds)
        stats[frame] = {
            'frame': frame, 'raw': len(centers), 'valid_2d': int(valid.sum()),
            'projected_map_rows': len(projected), 'residual': int(residual.sum()),
            'rgb_hw': [height, width], 'depth_hw': list(depth_m.shape),
            'depth_unit': 'metres', 'depth_sampling': dict(depth_stats),
            'sampled_depth_median_m': float(np.median(sampled_depths)) if len(sampled_depths) else None,
        }
    plans = {}
    for frame in frames:
        ids, worlds = points[frame]
        hits = recurrence_counts(worlds, frame, frames, world_to_camera, k_rgb,
                                 shapes, trees, RECURRENCE_RADIUS_PX)
        recurrent = ids[hits >= RECURRENCE_OTHER_FRAMES]
        ranked_trigger = top_m(recurrent, ascores[frame], max(BUDGETS))
        ranked_score = top_m(np.arange(len(ascores[frame])), ascores[frame], max(BUDGETS))
        plans[frame] = {f'{kind}_m{m}': ranked[:m]
                        for kind, ranked in [('trigger', ranked_trigger), ('score', ranked_score)]
                        for m in BUDGETS}
        for kind in ('trigger', 'score'):
            assert np.array_equal(plans[frame][f'{kind}_m150'], plans[frame][f'{kind}_m300'][:150])
        stats[frame].update({
            'recurrent': len(recurrent),
            'recurrence_other_frame_histogram': dict(Counter(map(int, hits))),
            'selected': {v: len(plans[frame][v]) for v in VARIANTS},
        })
    directory = output / scene
    directory.mkdir()
    write_json(directory / 'selection.json', {
        'scene': scene, 'gt_access': False, 'depth_unit': 'metres',
        'frames': [{**stats[f], 'selected_anchor_ids':
                    {v: plans[f][v].tolist() for v in VARIANTS}} for f in frames],
    })
    inputs[str((directory / 'selection.json').resolve())] = sha256(directory / 'selection.json')
    elapsed = time.perf_counter() - start
    print(f'{scene}: selected {len(frames)} frames; residual/frame='
          f'{np.mean([s["residual"] for s in stats.values()]):.1f}; '
          f'recurrent/frame={np.mean([s["recurrent"] for s in stats.values()]):.1f}; '
          f'selection={elapsed:.2f}s', flush=True)
    return {'boxes': boxes, 'scores': scores, 'poses': poses, 'k_rgb': k_rgb,
            'k_depth': k_depth, 'frames': frames, 'anchors': anchors,
            'ascores': ascores, 'plans': plans, 'frame_stats': stats,
            'selection_seconds': elapsed}


def append_boxes(boxes, scores, extra):
    extra = np.asarray(extra, dtype=float).reshape(-1, 8, 3)
    return np.concatenate([boxes, extra]), np.r_[scores, np.full(len(extra), APPEND_SCORE)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((PILOT / 'protocol.json').read_text())
    assert tuple(protocol['scenes']) == SCENES and protocol['gap'] == 20
    source_paths = [Path(__file__), PILOT / 'protocol.json'] + [ROOT / p for p in (
        'tools/ca1m_seedless_trigger_core.py', 'tools/audit_ca1m_nms_child_headroom.py',
        'tools/true_fusion_audit_core.py', 'tools/audit_m1m2_remaining_children.py',
        'tools/ca1m_prenms_query_core.py', 'tools/validate_ca1m_prenms_query.py',
        'boxfusion/boxer_lifter.py')]
    inputs = {str(p.resolve()): sha256(p) for p in source_paths}
    selected = {s: select_scene(s, args.output, inputs) for s in SCENES}
    assert sum(len(p['frames']) for p in selected.values()) == 105
    write_json(args.output / 'selection_complete.json', {
        'gt_access': False, 'scenes': list(SCENES), 'frames': 105,
        'selection_hashes': {s: sha256(args.output / s / 'selection.json') for s in SCENES},
    })

    adapter = build_lifter(args.output)
    lift_counts, lift_ms = Counter(), 0.0
    for scene, state in selected.items():
        for ordinal, frame in enumerate(state['frames'], 1):
            union = np.unique(np.concatenate(list(state['plans'][frame].values())))
            _, rgb, depth_m = load_frame(DATA_ROOT / scene, frame)
            corners, elapsed_ms, hit = lift(
                adapter, scene, frame, rgb, depth_m, state['k_rgb'], state['k_depth'],
                state['poses'][frame], state['anchors'][frame][union])
            corners = np.asarray(corners)
            if (corners.shape != (len(union), 8, 3) or not np.isfinite(corners).all()
                    or not (np.ptp(corners, axis=1) > 0).all()):
                raise ValueError(f'Lifter changed cardinality or returned invalid geometry: {scene}/{frame}')
            path = args.output / scene / f'lifted_{frame:06d}.npz'
            np.savez_compressed(path, anchor_ids=union, corners=corners)
            inputs[str(path.resolve())] = sha256(path)
            lift_counts.update(frames=1, unique_anchors=len(union), cache_hits=int(hit))
            lift_ms += elapsed_ms
            if ordinal % 10 == 0 or ordinal == len(state['frames']):
                print(f'{scene}: lifted {ordinal}/{len(state["frames"])} frames', flush=True)
    write_json(args.output / 'lifting_complete.json', {
        'gt_access': False, **dict(lift_counts), 'lifter_ms': lift_ms,
        'note': 'Combined diagnostic workload, not production FPS',
    })

    # Only now read GT: all trigger/control IDs and lifted geometry are frozen.
    baseline, gts, arms = {}, {}, {}
    totals = {v: Counter({'ge1': 0, 'ge2': 0, 'ge3': 0}) for v in VARIANTS}
    coverage_rows, cluster_stats, missed_total = {v: [] for v in VARIANTS}, {}, 0
    for scene, state in selected.items():
        path = GT_ROOT / scene / 'after_filter_boxes.npy'
        inputs[str(path.resolve())] = sha256(path)
        gt = valid_boxes(np.load(path, allow_pickle=False), str(path))
        boxes, scores = state['boxes'], state['scores']
        biou = aabb_iou(boxes, gt).max(0) if len(boxes) else np.zeros(len(gt))
        missed = np.flatnonzero(biou <= .15)
        missed_total += len(missed)
        support = {v: {int(g): set() for g in missed} for v in VARIANTS}
        best = {v: {int(g): (0.0, None) for g in missed} for v in VARIANTS}
        clusters = {v: {} for v in VARIANTS}
        for frame in state['frames']:
            with np.load(args.output / scene / f'lifted_{frame:06d}.npz') as lifted:
                union, all_corners = lifted['anchor_ids'], lifted['corners']
            for variant in VARIANTS:
                ids = state['plans'][frame][variant]
                positions = np.searchsorted(union, ids)
                assert np.array_equal(union[positions], ids)
                corners = all_corners[positions]
                update_coverage(support[variant], best[variant], missed, corners, gt, frame)
                add_clusters(clusters[variant], frame, ids, corners,
                             state['ascores'][frame][ids], CLUSTER_VOXEL)
        cluster_stats[scene] = {}
        for variant in VARIANTS:
            for g in missed:
                count = len(support[variant][int(g)])
                for minimum in (1, 2, 3):
                    totals[variant][f'ge{minimum}'] += int(count >= minimum)
                coverage_rows[variant].append({'scene': scene, 'gt': int(g),
                                               'frames': sorted(support[variant][int(g)])})
            oracle_extra = [best[variant][int(g)][1] for g in missed
                            if len(support[variant][int(g)]) >= 3]
            arms.setdefault(variant, {})[scene] = append_boxes(boxes, scores, oracle_extra)
            confirmed = [c for _, c in sorted(clusters[variant].items()) if len(c['frames']) >= 3]
            births = [c['box'] for c in confirmed]
            arms.setdefault(f'{variant}_all_clusters', {})[scene] = append_boxes(boxes, scores, births)
            ciou = aabb_iou(np.asarray(births).reshape(-1, 8, 3), gt)
            matched = ciou.max(1) > .15 if len(births) else np.zeros(0, dtype=bool)
            owner = ciou.argmax(1) if len(births) else np.zeros(0, dtype=int)
            cluster_stats[scene][variant] = {
                'all_voxels': len(clusters[variant]), 'confirmed_ge3': len(confirmed),
                'selected_observations': sum(c['observations'] for c in clusters[variant].values()),
                'confirmed_rep_unmatched_gt15': int((~matched).sum()),
                'confirmed_rep_matches_existing_gt15': int((matched & (biou[owner] > .15)).sum()),
                'confirmed_rep_matches_missed_gt15': int((matched & (biou[owner] <= .15)).sum()),
                'distinct_frame_histogram': dict(Counter(len(c['frames']) for c in clusters[variant].values())),
            }
        baseline[scene], gts[scene] = (boxes, scores), gt
    assert missed_total == 87
    metrics = {'baseline': {str(t): class_agnostic_ap(baseline, gts, t) for t in THRESHOLDS}}
    for name, predictions in arms.items():
        metrics[name] = {}
        for t in THRESHOLDS:
            value = class_agnostic_ap(predictions, gts, t)
            for metric in ('ap', 'tp', 'fp'):
                value[f'delta_{metric}'] = value[metric] - metrics['baseline'][str(t)][metric]
            metrics[name][str(t)] = value
    reference_path = ROOT / 'reports/ca1m_seedless_step1b_20260911/results.json'
    reference = json.loads(reference_path.read_text())
    inputs[str(reference_path.resolve())] = sha256(reference_path)
    for t in THRESHOLDS:
        assert abs(metrics['baseline'][str(t)]['ap'] - reference['metrics']['baseline'][str(t)]['ap']) < 1e-6
    rows = [r for s in selected.values() for r in s['frame_stats'].values()]
    budget_stats = {key: float(np.mean([r[key] for r in rows])) for key in ('raw', 'residual', 'recurrent')}
    budget_stats['selected'] = {v: float(np.mean([r['selected'][v] for r in rows])) for v in VARIANTS}
    cluster_totals = {v: {key: sum(cluster_stats[s][v][key] for s in SCENES) for key in (
        'all_voxels', 'confirmed_ge3', 'selected_observations',
        'confirmed_rep_unmatched_gt15', 'confirmed_rep_matches_existing_gt15',
        'confirmed_rep_matches_missed_gt15')} for v in VARIANTS}
    step1_path = ROOT / 'reports/ca1m_seedless_step1_20260911/results.json'
    inputs[str(step1_path.resolve())] = sha256(step1_path)
    step1 = json.loads(step1_path.read_text())
    reference_gt = {(r['scene'], r['gt']) for r in step1['coverage_rows'] if r['frames_0.15'] >= 3}
    retained = {v: len(reference_gt & {(r['scene'], r['gt']) for r in coverage_rows[v]
                                     if len(r['frames']) >= 3}) for v in VARIANTS}
    for path, digest in inputs.items():
        assert sha256(path) == digest, f'Input changed: {path}'
    result = {
        'schema': 'boxfusion.seedless_prenms_step1c.v2', 'completed': True,
        'scenes': list(SCENES), 'frames': len(rows), 'missed_total': missed_total,
        'trigger': {'residual_iou': RESIDUAL_IOU, 'recurrence_radius_rgb_px': RECURRENCE_RADIUS_PX,
                    'other_frames': RECURRENCE_OTHER_FRAMES, 'budgets': BUDGETS,
                    'gt_free_selection': True, 'depth_unit': 'metres',
                    'mask_source': 'projected frozen final-map 3D rows',
                    'budget_order': 'raw score descending after binary geometry gate; stable ties',
                    'cluster': '0.3m world-centre voxel, >=3 distinct frames, highest raw-score exemplar'},
        'coverage': {v: dict(totals[v]) for v in VARIANTS}, 'coverage_rows': coverage_rows,
        'step1_reference_gt_count': len(reference_gt), 'retained_step1_ge3': retained,
        'budget_per_frame': budget_stats, 'cluster_counts_per_scene': cluster_stats,
        'cluster_totals': cluster_totals, 'metrics': metrics,
        'step1b_reference': {'top300_coverage': reference['coverage_ge3_ge2']['top300'],
                            'top300_metrics': reference['metrics']['top300'],
                            'note': 'Step 1b GT-selected top-3 per target within score top-300; '
                                    'score_m300 lifts the entire score prefix without GT'},
        'timing': {'selection_seconds': sum(s['selection_seconds'] for s in selected.values()),
                   'combined_lifter_ms': lift_ms, **dict(lift_counts)},
        'limits': [
            'Final-map masks and all-frame recurrence use future observations; offline GT-free audit only.',
            'Oracle append metrics select qualifying missed GTs and best geometry; not deployable AP.',
            'all_clusters metrics include all confirmed voxel births at score .10, without GT filtering or extra NMS.',
            'Voxels are a budget proxy, not validated instance tracks; fragmentation/duplicates are possible.',
            'Unmatched GT does not distinguish wall/floor/background from unlabeled objects.',
            'Three development scenes; no full107, bounded-memory or end-to-end FPS claim.',
        ],
    }
    write_json(args.output / 'results.json', result)
    write_json(args.output / 'input_sha256.json', inputs)
    print(json.dumps({
        'coverage': result['coverage'], 'retained_step1_ge3': retained,
        'budget_per_frame': budget_stats, 'cluster_totals': cluster_totals,
        'deltas': {v: [round(metrics[v][str(t)]['delta_ap'], 4) for t in THRESHOLDS] for v in arms},
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
