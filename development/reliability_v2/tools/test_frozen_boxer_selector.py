"""CPU-only, GT-free discrete selection on existing native Top-K captures.

This is a fixed-history output counterfactual, not a new detector run. The
native PFO memberships and downstream association history are never changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_m1m2_remaining_children import read_prediction, verify_metric
from tools.true_fusion_audit_core import aabb_iou, class_agnostic_ap
from tools.verify_true_fusion_capture import Evidence

CAPTURE = ROOT / 'reports/ca1m_true_fusion_pilot_v2_20260908'
THRESHOLDS = (.15, .25, .5)


def choose(ids, values):
    """No GT, confidence fitting, thresholds, or future observations."""
    ids = sorted(set(int(i) for i in ids))
    if not ids or not np.isfinite(values[ids]).all():
        return None
    return min(ids, key=lambda i: (-float(values[i]), i))


def join_confidence(raw, records):
    """Join by frame and exact original row order/2D boxes/scores, not GT/IoU."""
    confidence = np.full(len(raw['scores']), np.nan)
    seen = set()
    errors = {'world_center': 0., 'world_rotation': 0., 'confidence_formula': 0.}
    for record in records:
        if record.get('schema') != 'boxfusion.boxer_lifting.frame.v1':
            continue
        frame = record['frame_id']
        if frame in seen:
            raise ValueError(f'Duplicate lifting frame {frame}')
        seen.add(frame)
        ids = np.flatnonzero(raw['frame_ids'] == frame)
        assert len(ids) == record['count']
        if not len(ids):
            continue
        assert record['mode'] == 'active' and record['apply_stage'] == 'post_filter'
        assert np.array_equal(raw['boxes2d'][ids], record['input_boxes_xyxy'])
        assert np.array_equal(raw['scores'][ids], record['detector_scores'])
        pose = np.asarray(record['camera_to_world'])
        assert np.array_equal(raw['cam_poses'][ids], np.repeat(pose[None], len(ids), axis=0))
        boxes = np.asarray(record['output_xyz_dims_camera'])
        rotations = np.asarray(record['output_rotation_camera_object'])
        assert np.array_equal(raw['boxes_xyzlhw'][ids, 3:], boxes[:, 3:])
        center_error = float(np.max(abs(boxes[:, :3] @ pose[:3, :3].T + pose[:3, 3]
                                         - raw['boxes_xyzlhw'][ids, :3]))) if len(ids) else 0.
        rotation_error = float(np.max(abs(pose[:3, :3] @ rotations - raw['rotations'][ids]))) if len(ids) else 0.
        # Pure numerical tolerance for camera-to-world float32 operations.
        assert center_error <= 1e-4 and rotation_error <= 1e-5
        q = np.asarray(record['confidence'], dtype=float).reshape(-1)
        v = np.asarray(record['logvar'], dtype=float).reshape(-1)
        assert q.shape == v.shape == (len(ids),)
        assert np.isfinite(q).all() and np.isfinite(v).all() and ((q >= 0) & (q <= 1)).all()
        error = float(np.max(abs(q - 1 / (1 + np.exp(v))))) if len(ids) else 0.
        assert error < 2e-7
        confidence[ids] = q
        errors['world_center'] = max(errors['world_center'], center_error)
        errors['world_rotation'] = max(errors['world_rotation'], rotation_error)
        errors['confidence_formula'] = max(errors['confidence_formula'], error)
    return confidence, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    # Deterministic selection and missing-data fallback checks.
    assert choose([2, 1, 1], np.asarray([.1, .8, .8])) == 1
    assert choose([0, 1], np.asarray([.1, .8])) == 1
    assert choose([0, 1], np.asarray([.1, np.nan])) is None
    evidence = Evidence()
    evidence.remember(__file__)
    for helper in ('tools/audit_m1m2_remaining_children.py', 'tools/true_fusion_audit_core.py',
                   'tools/verify_true_fusion_capture.py', 'tools/audit_ca1m_nms_child_headroom.py'):
        evidence.remember(ROOT / helper)
    integrity = evidence.json(CAPTURE / 'integrity.json')
    assert integrity['checkpass'] and integrity['matches_complete_planned_scene_list']
    scenes = integrity['scenes']
    assert len(scenes) == 10
    predictions = {a: {} for a in ('native_pfo', 'detector_select', 'boxer_select')}
    audit, event_choices, cache = [], [], {}
    elapsed = 0.
    # No GT file is opened until every scene's selections have been fixed.
    for scene in scenes:
        directory = CAPTURE / scene
        manifest = evidence.json(directory / 'scene.json')
        protocol = evidence.json(directory / 'protocol.json')
        assert manifest['completed'] and manifest['saved_row_mapping_exact']
        assert protocol['ground_truth_allowed'] is False
        raw = evidence.npz(directory / 'observations.npz')
        final = evidence.npz(directory / 'final.npz')
        birth = evidence.npz(directory / 'birth_native_corners.npz')['corners']
        assert np.array_equal(raw['init_ids'], np.arange(len(raw['init_ids'])))
        assert birth.shape == raw['corners'].shape
        assert np.max(abs(birth - raw['corners']), initial=0) < 1e-4
        events = evidence.json(directory / 'fusion_events.json')
        log = directory / 'diagnostics/lifting_boxer_diagnostics_dir' / f'{scene}_boxer_lifting.jsonl'
        records = [json.loads(line) for line in evidence.remember(log).open()] if log.exists() else []
        assert all(r.get('scene_id') == scene for r in records)
        q, errors = join_confidence(raw, records)
        path = evidence.remember(directory / 'predictions' / f'{scene}_boxes.pkl')
        base = read_prediction(path)
        assert np.array_equal(base[0], final['corners'])
        assert np.array_equal(base[1].astype(final['scores'].dtype), final['scores'])
        for arm in predictions:
            predictions[arm][scene] = (base[0].copy(), base[1].copy())
        latest, choices = {}, {}
        started = time.perf_counter()
        for event in events:
            ids = list(dict.fromkeys(event['selected_ids']))
            assert 0 < len(ids) <= 3
            assert set(ids) <= set(event['source_ids'])
            assert max(raw['frame_ids'][ids]) <= event['frame_id']
            assert np.array_equal(raw['frame_ids'][event['selected_ids']], event['selected_frame_ids'])
            detector = choose(ids, raw['scores'])
            boxer = choose(ids, q)
            if len(ids) < 2 or detector is None or boxer is None:
                continue
            choices[event['event_id']] = (detector, boxer)
            event_choices.append({'scene': scene, 'event_id': event['event_id'],
                                  'frame_id': event['frame_id'], 'ids': ids,
                                  'detector_id': detector, 'boxer_id': boxer,
                                  'disagree': detector != boxer})
        elapsed += time.perf_counter() - started
        for event in events:
            if event['updated']:
                latest[event['init_id']] = event
        changed, missing_links, disagreements = 0, 0, 0
        links = []
        for row, init in enumerate(final['init_ids']):
            event = latest.get(int(init))
            if event is None:
                continue
            if not (np.array_equal(np.asarray(event['post_box_xyzlhw'], dtype=final['boxes_xyzlhw'].dtype), final['boxes_xyzlhw'][row])
                    and np.array_equal(np.asarray(event['post_rotation'], dtype=final['rotations'].dtype), final['rotations'][row])):
                missing_links += 1
                continue
            if event['event_id'] not in choices:
                continue
            detector, boxer = choices[event['event_id']]
            predictions['detector_select'][scene][0][row] = birth[detector]
            predictions['boxer_select'][scene][0][row] = birth[boxer]
            changed += 1
            disagreements += int(detector != boxer)
            links.append({'row': row, 'init_id': int(init), 'event_id': event['event_id'],
                          'detector_id': detector, 'boxer_id': boxer})
        for arm in predictions:
            assert predictions[arm][scene][0].shape == base[0].shape
            assert np.array_equal(predictions[arm][scene][1], base[1])
        audit.append({'scene': scene, 'raw_rows': len(q), 'confidence_matched': int(np.isfinite(q).sum()),
                      'pfo_events': len(events), 'eligible_events': len(choices), 'terminal_rows': len(base[0]),
                      'terminal_replacements': changed, 'terminal_choice_disagreements': disagreements,
                      'unresolved_terminal_links_kept_native': missing_links,
                      'join_errors': errors, 'terminal_links': links})
        cache[scene] = (raw, birth, events, q)
    anchor = verify_metric()
    gt = {scene: np.load(evidence.remember(Path('/tmp/ca1m_clean_root') / scene / 'after_filter_boxes.npy'))
          for scene in scenes}
    metrics = {arm: {str(t): class_agnostic_ap(pred, gt, t) for t in THRESHOLDS}
               for arm, pred in predictions.items()}
    differences = {arm: {str(t): metrics[arm][str(t)]['ap'] - metrics['native_pfo'][str(t)]['ap']
                        for t in THRESHOLDS} for arm in ('detector_select', 'boxer_select')}
    quality_vs_detector = {str(t): metrics['boxer_select'][str(t)]['ap'] - metrics['detector_select'][str(t)]['ap']
                           for t in THRESHOLDS}
    # Diagnostic only: judge both picks against the same GT identity chosen by
    # the captured fused box. This never changes a selected ID or output row.
    comparisons = []
    for row in audit:
        scene = row['scene']
        raw, birth, events, q = cache[scene]
        overlaps = aabb_iou(birth, gt[scene])
        base_iou = aabb_iou(predictions['native_pfo'][scene][0], gt[scene])
        for link in row['terminal_links']:
            values = base_iou[link['row']]
            if not len(values) or values.max() <= .15:
                continue
            target = int(values.argmax())
            a, b = link['detector_id'], link['boxer_id']
            comparisons.append({'scene': scene, **link, 'gt_index': target,
                                'native_iou': float(values[target]),
                                'detector_pick_iou': float(overlaps[a, target]),
                                'boxer_pick_iou': float(overlaps[b, target]),
                                'detector_pick_q': float(q[a]), 'boxer_pick_q': float(q[b])})
    evidence.unchanged()
    result = {'schema': 'boxfusion.frozen_boxer_discrete_selector.v1', 'completed': True,
              'capture_root': str(CAPTURE), 'scenes': scenes, 'metric_checks': anchor,
              'protocol': {
                  'pool': 'Same native PFO selected_ids, at most three, from current/past frames only.',
                  'baseline': 'Same captured native PFO output; NOT current dual-source M1+M2.',
                  'detector_select': 'Maximum original detector score; smallest raw ID breaks ties.',
                  'boxer_select': 'Maximum frozen Boxer confidence; smallest raw ID breaks ties.',
                  'output': 'Only exact latest-PFO-to-final links replaced; counts/order/scores frozen.',
                  'fallback': 'Any missing confidence/ineligible pool/unresolved final link keeps native for both arms.',
                  'GT': 'Opened only after all selections were fixed; evaluation/diagnosis only.',
                  'no_inference': True, 'no_training': True, 'no_tuning': True,
              },
              'totals': {k: sum(row[k] for row in audit) for k in (
                  'raw_rows', 'confidence_matched', 'pfo_events', 'eligible_events', 'terminal_rows',
                  'terminal_replacements', 'terminal_choice_disagreements', 'unresolved_terminal_links_kept_native')},
              'selector_only_seconds': elapsed, 'metrics': metrics, 'delta_vs_native': differences,
              'boxer_minus_detector_ap': quality_vs_detector, 'scene_audit': audit,
              'event_choices': event_choices, 'same_target_geometry_diagnostics': comparisons,
              'limitations': [
                  'Ten existing development scenes; no independent generalization claim.',
                  'Native association supplies instance groups; they can contain identity errors, not GT-cleaned groups.',
                  'Fixed-history output counterfactual: no changed geometry feeds back into future association or PFO.',
                  'Eligible final rows only; this is not full replacement of every native/birth box.',
                  'Inference FPS not measured; selector CPU timing alone cannot prove end-to-end real time.',
                  'Current M1+M2 run differs from this capture and is not mixed into the comparison.',
              ]}
    encoded = {name: json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
               for name, value in [('results.json', result), ('input_sha256.json', evidence.read_hashes)]}
    args.output.mkdir(parents=True)
    for name, value in encoded.items():
        with (args.output / name).open('x') as handle:
            handle.write(value)
    print(json.dumps({k: result[k] for k in ('totals', 'metrics', 'delta_vs_native',
                                            'boxer_minus_detector_ap', 'selector_only_seconds')}, indent=2))


if __name__ == '__main__':
    main()
