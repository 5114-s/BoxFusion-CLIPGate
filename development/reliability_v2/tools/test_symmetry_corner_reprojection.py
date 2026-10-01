"""One frozen, CPU-only corner-reprojection pilot on existing native PFO captures.

No training/inference, GT selection, association changes, or parameter sweeps.
Only exactly linked terminal rows are changed; this is NOT an online rerun.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_m1m2_remaining_children import read_prediction, verify_metric
from tools.true_fusion_audit_core import aabb_iou, class_agnostic_ap
from tools.verify_true_fusion_capture import Evidence, canonical_corners

CAPTURE = ROOT / 'reports/ca1m_true_fusion_pilot_v2_20260908'
THRESHOLDS = (.15, .25, .5)
SIGNS = np.asarray([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                    [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]], dtype=float)
PROTOCOL = {
    'variant_count': 1,
    'baseline': 'Captured native PFO; NOT the current dual-source M1+M2 run.',
    'inputs': 'Actual native selected_ids (at most 3), original poses/targets/weights.',
    'alignment': 'Minimum SO(3) angle to PFO initial rotation among 4 local-z quarter turns; fixed during solve.',
    'ambiguity': 'Fall back if the best and second-best alignment angles differ by <=10 degrees.',
    'objective': 'View-weighted squared corner projection error normalized by image width/height; boundary targets are one-sided censored constraints.',
    'initialization': 'Captured pfo_init_box_xyzlhw and pfo_rotation, never native post or GT.',
    'parameters': 'Center offset / initial maximum dimension, log dimension ratios, local-z yaw delta.',
    'bounds': 'Center offsets +/-0.5 initial maximum dimension per axis; dimensions [0.5,2] x initial; yaw +/-45 degrees.',
    'solver': '10 LM trial steps maximum; initial damping 1e-3, divide by 3 on improvement, multiply by 10 otherwise; analytic Jacobian.',
    'fallback': 'Invalid input/depth<=1mm, ambiguous symmetry, initial Jacobian singular-value ratio<1e-8, or no loss improvement >1e-12: retain native post.',
    'terminal_scope': 'Only final rows exactly equal to the latest UPDATED native PFO event; other rows unchanged.',
    'frozen': 'All native association history, selected membership, score, output row count/order, and camera poses.',
    'gate': 'AP50 gain >=0.5 percentage points AND AP15/AP25 do not decrease.',
    'GT': 'Annotation reads forbidden until ALL proposals and final arrays have been saved and hashed.',
    'no_inference': True, 'no_training': True, 'no_parameter_sweep': True,
}


def rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c,-s,0], [s,c,0], [0,0,1.]])


QUARTERS = [rz(i * np.pi / 2) for i in range(4)]
PERMUTATIONS = [np.argmin(np.linalg.norm((SIGNS @ q.T)[:,None] - SIGNS[None], axis=2), axis=1)
                for q in QUARTERS]


def align(reference, observed):
    angles = np.asarray([np.arccos(np.clip((np.trace(reference.T @ observed @ q)-1)/2, -1, 1))
                         for q in QUARTERS])
    order = np.argsort(angles, kind='stable')
    if angles[order[1]] - angles[order[0]] <= np.deg2rad(10):
        return None
    return int(order[0])


def geometry(x, box, reference):
    scale = max(box[3:])
    local = SIGNS * (box[3:] * np.exp(x[3:6])) / 2
    rotation = reference @ rz(x[6])
    corners = local @ rotation.T + box[:3] + x[:3] * scale
    jac = np.zeros((8,3,7))
    jac[:,:,:3] = np.eye(3) * scale
    for j in range(3):
        jac[:,:,3+j] = local[:,j,None] * rotation[:,j]
    jac[:,:,6] = np.stack([-local[:,1], local[:,0], np.zeros(8)], axis=1) @ rotation.T
    return corners, jac


def residual(x, box, reference, inverse_poses, intrinsic, targets, image_size, weights):
    corners, world_jac = geometry(x, box, reference)
    cam = np.einsum('vij,kj->vki', inverse_poses[:,:3,:3], corners) + inverse_poses[:,None,:3,3]
    if not np.isfinite(cam).all() or (cam[:,:,2] <= .001).any():
        return None
    projected = cam @ intrinsic.T
    uv = projected[:,:,:2] / projected[:,:,2:]
    perspective = np.zeros((*cam.shape[:2], 2, 3))
    for j in range(2):
        perspective[:,:,j] = (intrinsic[j] - uv[:,:,j,None]*intrinsic[2]) / projected[:,:,2,None]
    cam_jac = np.einsum('vij,kjp->vkip', inverse_poses[:,:3,:3], world_jac)
    jac = np.einsum('vkij,vkjp->vkip', perspective, cam_jac)
    difference = uv - targets
    # Interior coordinates remain ordinary residuals even if prediction exits
    # the image; boundary targets only constrain the appropriate half-space.
    inactive = ((targets <= 0) & (uv <= 0)) | ((targets >= image_size) & (uv >= image_size))
    difference[inactive] = 0
    jac[inactive] = 0
    normalizer = np.sqrt(weights / weights.sum())[:,None,None] / image_size
    return (difference * normalizer).ravel(), (jac * normalizer[:,:,:,None]).reshape(-1,7)


def optimize(box, reference, poses, intrinsic, targets, image_size, weights):
    inverse = np.linalg.inv(poses)
    x = np.zeros(7)
    current = residual(x, box, reference, inverse, intrinsic, targets, image_size, weights)
    if current is None:
        return None, {'status': 'invalid_initial_depth'}
    values = np.linalg.svd(current[1], compute_uv=False)
    if values[0] == 0 or values[-1] < values[0]*1e-8:
        return None, {'status': 'rank_deficient'}
    initial = float(current[0] @ current[0])
    cost, damping, accepted = initial, 1e-3, 0
    lower = np.asarray([-.5]*3 + [np.log(.5)]*3 + [-np.pi/4])
    upper = -lower
    for iteration in range(10):
        r, jac = current
        normal = jac.T @ jac
        system = normal + damping*np.diag(np.maximum(np.diag(normal), 1e-12))
        try:
            step = np.linalg.solve(system, -jac.T @ r)
        except np.linalg.LinAlgError:
            return None, {'status': 'linear_solve_failed'}
        candidate = np.clip(x + step, lower, upper)
        proposed = residual(candidate, box, reference, inverse, intrinsic, targets, image_size, weights)
        proposed_cost = float(proposed[0] @ proposed[0]) if proposed is not None else float('inf')
        if proposed_cost < cost:
            x, current, cost = candidate, proposed, proposed_cost
            accepted += 1
            damping = max(damping/3, 1e-12)
        else:
            damping *= 10
        if np.linalg.norm(candidate-x) < 1e-10 and np.linalg.norm(step) < 1e-10:
            break
    info = {'status': 'accepted' if initial-cost > 1e-12 else 'no_improvement',
            'initial_loss': initial, 'final_loss': cost, 'iterations': iteration+1,
            'accepted_steps': accepted, 'initial_jacobian_condition': float(values[0]/values[-1]),
            'at_bound': bool(np.any(np.isclose(x, lower, atol=1e-7) | np.isclose(x, upper, atol=1e-7))),
            'parameters': x.tolist()}
    return (geometry(x, box, reference)[0] if info['status'] == 'accepted' else None), info


def proposal(raw, event):
    ids = event['selected_ids']
    assert 1 < len(ids) <= 3 and len(set(ids)) == len(ids)
    assert set(ids) <= set(event['source_ids'])
    assert max(raw['frame_ids'][ids]) <= event['frame_id']
    assert np.array_equal(raw['frame_ids'][ids], event['selected_frame_ids'])
    box = np.asarray(event['pfo_init_box_xyzlhw'], dtype=float)
    reference = np.asarray(event['pfo_rotation'], dtype=float)
    weights = np.asarray(event['scores_box'] if event['use_view_weights'] else np.ones(len(ids)), dtype=float)
    poses = raw['cam_poses'][ids]
    intrinsic = np.asarray(event['K'])[:3,:3]
    image_size = np.asarray([event['W'], event['H']])
    if not (np.isfinite(box).all() and (box[3:] > 0).all() and (weights > 0).all()):
        return None, {'status': 'invalid_inputs'}
    alignments, targets, projection_error = [], [], 0.
    for i, inverse in zip(ids, np.linalg.inv(poses)):
        cam = raw['corners'][i] @ inverse[:3,:3].T + inverse[:3,3]
        if (cam[:,2] <= .001).any():
            return None, {'status': 'invalid_source_depth'}
        projected = cam @ intrinsic.T
        clipped = np.clip(projected[:,:2]/projected[:,2:], 0, image_size)
        projection_error = max(projection_error, float(abs(clipped - raw['projected_boxes'][i]).max()))
        # Float64 re-expression versus native float32 world/camera operations.
        assert projection_error < .1, 'Projection convention mismatch (>0.1 pixel)'
        alignment = align(reference, raw['rotations'][i])
        if alignment is None:
            return None, {'status': 'ambiguous_symmetry'}
        alignments.append(alignment)
        targets.append(raw['projected_boxes'][i][PERMUTATIONS[alignment]])
    result, info = optimize(box, reference, poses, intrinsic, np.asarray(targets), image_size, weights)
    info.update({'quarter_turns': alignments, 'projection_check_max_px': projection_error})
    return result, info


def self_test():
    box = np.asarray([0.,0.,5.,1.2,.7,.9])
    reference = rz(.2)
    native = canonical_corners(box[None], reference[None])[0]
    assert np.allclose(geometry(np.zeros(7), box, reference)[0], native)
    for quarter in range(4):
        observed = reference @ QUARTERS[quarter]
        aligned = align(reference, observed)
        dims = np.abs(QUARTERS[quarter]).T @ box[3:]
        alternative = canonical_corners(np.r_[box[:3], dims][None], observed[None])[0]
        assert np.allclose(alternative[PERMUTATIONS[aligned]], native)
    assert align(reference, reference @ rz(np.pi/4)) is None
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:,0,3] = [-.8,0,.8]
    intrinsic = np.asarray([[420,0,256],[0,420,192],[0,0,1.]])
    size, weights = np.asarray([512,384]), np.ones(3)
    truth = np.asarray([.03,-.02,.04,.03,-.05,.02,.08])
    target_corners = geometry(truth, box, reference)[0]
    cam = target_corners[None] - poses[:,None,:3,3]
    uv = cam @ intrinsic.T
    targets = uv[:,:,:2] / uv[:,:,2:]
    inverse = np.linalg.inv(poses)
    args = (box, reference, inverse, intrinsic, targets, size, weights)
    r, jac = residual(np.zeros(7), *args)
    numeric = np.empty_like(jac)
    for j in range(7):
        perturb = np.zeros(7); perturb[j] = 1e-6
        numeric[:,j] = (residual(perturb, *args)[0] - residual(-perturb, *args)[0])/(2e-6)
    error = float(abs(numeric-jac).max())
    assert error < 1e-7
    result, info = optimize(box, reference, poses, intrinsic, targets, size, weights)
    assert info['status'] == 'accepted' and np.max(abs(result-target_corners)) < 1e-5
    zero = residual(truth, *args)[0]
    assert np.max(abs(zero)) < 1e-12
    # Upper-bound targets penalize predictions inside the image, but not
    # predictions beyond that boundary. Lower-bound targets are symmetric.
    for boundary, principal in [(size, 1000.), (np.zeros(2), -1000.)]:
        censored = np.broadcast_to(boundary, targets.shape).copy()
        rc, jc = residual(truth, box, reference, inverse, intrinsic, censored, size, weights)
        assert np.linalg.norm(rc) > 0 and np.linalg.norm(jc) > 0
        shifted = intrinsic.copy(); shifted[:2,2] = principal
        rc, jc = residual(truth, box, reference, inverse, shifted, censored, size, weights)
        assert np.count_nonzero(rc) == 0 and np.count_nonzero(jc) == 0
        assert optimize(box, reference, poses, shifted, censored, size, weights)[1]['status'] == 'rank_deficient'
    behind = box.copy(); behind[2] = -5
    assert optimize(behind, reference, poses, intrinsic, targets, size, weights)[1]['status'] == 'invalid_initial_depth'
    return {'corner_convention': True, 'four_symmetry_permutations': True, 'ambiguity_fallback': True,
            'synthetic_recovery': True, 'zero_residual': True, 'invalid_depth': True,
            'censored_boundary_constraints': True, 'rank_fallback': True,
            'analytic_jacobian_max_error': error}


def write_json(path, value):
    with path.open('x') as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    checks = self_test()
    if args.self_test:
        print(json.dumps(checks, indent=2)); return
    if args.output is None:
        parser.error('--output required')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'protocol.json', PROTOCOL)
    evidence = Evidence()
    for source in [Path(__file__), args.output/'protocol.json', ROOT/'tools/audit_m1m2_remaining_children.py',
                   ROOT/'tools/true_fusion_audit_core.py', ROOT/'tools/verify_true_fusion_capture.py',
                   ROOT/'tools/audit_ca1m_nms_child_headroom.py', ROOT/'boxfusion/instances.py']:
        evidence.remember(source)
    gt_allowed = [False]
    def guard(event, values):
        if event == 'open' and values and isinstance(values[0], (str, bytes)):
            name = str(values[0])
            if not gt_allowed[0] and ('after_filter_boxes.npy' in name or 'full_annotations.json' in name):
                raise RuntimeError('GT read forbidden before proposal freeze')
    sys.addaudithook(guard)
    integrity = evidence.json(CAPTURE/'integrity.json')
    assert integrity['checkpass'] and integrity['matches_complete_planned_scene_list']
    scenes = integrity['scenes']; assert len(scenes) == 10
    predictions = {'native_pfo': {}, 'corner_reprojection': {}}
    audits, event_audits, durations = [], [], []
    for scene in scenes:
        directory = CAPTURE/scene
        manifest = evidence.json(directory/'scene.json')
        assert manifest['completed'] and manifest['saved_row_mapping_exact']
        assert evidence.json(directory/'protocol.json')['ground_truth_allowed'] is False
        raw = evidence.npz(directory/'observations.npz')
        final = evidence.npz(directory/'final.npz')
        assert np.array_equal(raw['init_ids'], np.arange(len(raw['init_ids'])))
        base = read_prediction(evidence.remember(directory/'predictions'/f'{scene}_boxes.pkl'))
        assert np.array_equal(base[0], final['corners'])
        assert np.array_equal(base[1].astype(final['scores'].dtype), final['scores'])
        for arm in predictions:
            predictions[arm][scene] = (base[0].copy(), base[1].copy())
        events = evidence.json(directory/'fusion_events.json')
        proposals, latest, links = {}, {}, []
        for event in events:
            started = time.perf_counter()
            candidate, info = proposal(raw, event)
            durations.append(time.perf_counter()-started)
            proposals[event['event_id']] = candidate
            event_audits.append({'scene': scene, 'event_id': event['event_id'],
                                 'selected_ids': event['selected_ids'], **info})
            if event['updated']:
                latest[event['init_id']] = event
        unresolved, changed = 0, 0
        for row, init in enumerate(final['init_ids']):
            event = latest.get(int(init))
            if event is None:
                continue
            if not (np.array_equal(np.asarray(event['post_box_xyzlhw'], dtype=final['boxes_xyzlhw'].dtype), final['boxes_xyzlhw'][row])
                    and np.array_equal(np.asarray(event['post_rotation'], dtype=final['rotations'].dtype), final['rotations'][row])):
                unresolved += 1; continue
            candidate = proposals[event['event_id']]
            links.append({'row': row, 'event_id': event['event_id'], 'replaced': candidate is not None})
            if candidate is not None:
                predictions['corner_reprojection'][scene][0][row] = candidate
                changed += 1
        test = predictions['corner_reprojection'][scene]
        assert test[0].shape == base[0].shape and np.array_equal(test[1], base[1])
        with (args.output/f'{scene}_frozen_outputs.npz').open('xb') as handle:
            np.savez_compressed(handle, native=base[0], proposal=test[0], scores=base[1])
        evidence.remember(args.output/f'{scene}_frozen_outputs.npz')
        audits.append({'scene': scene, 'raw_rows': len(raw['scores']), 'events': len(events),
                       'final_rows': len(base[0]), 'linked_rows': len(links), 'replaced_rows': changed,
                       'unresolved_links': unresolved, 'links': links})
    write_json(args.output/'proposals_before_gt.json', event_audits)
    evidence.remember(args.output/'proposals_before_gt.json')
    write_json(args.output/'pre_evaluation_sha256.json', evidence.read_hashes)
    gt_allowed[0] = True
    anchor = verify_metric()
    gt = {s: np.load(evidence.remember(Path('/tmp/ca1m_clean_root')/s/'after_filter_boxes.npy')) for s in scenes}
    metrics = {arm: {str(t): class_agnostic_ap(pred, gt, t) for t in THRESHOLDS} for arm,pred in predictions.items()}
    delta = {str(t): metrics['corner_reprojection'][str(t)]['ap']-metrics['native_pfo'][str(t)]['ap'] for t in THRESHOLDS}
    diagnostics = []
    for audit in audits:
        scene = audit['scene']
        base_iou = aabb_iou(predictions['native_pfo'][scene][0], gt[scene])
        test_iou = aabb_iou(predictions['corner_reprojection'][scene][0], gt[scene])
        for link in audit['links']:
            row = link['row']
            if not link['replaced'] or not len(gt[scene]) or base_iou[row].max() <= .15:
                continue
            target = int(base_iou[row].argmax())
            diagnostics.append({'scene': scene, **link, 'gt_index': target,
                                'native_iou': float(base_iou[row,target]), 'proposal_iou': float(test_iou[row,target])})
        audit['ap_delta'] = {str(t): class_agnostic_ap({scene:predictions['corner_reprojection'][scene]}, {scene:gt[scene]}, t)['ap']
                            - class_agnostic_ap({scene:predictions['native_pfo'][scene]}, {scene:gt[scene]}, t)['ap'] for t in THRESHOLDS}
    evidence.unchanged()
    result = {'schema': 'boxfusion.symmetry_corner_reprojection.v1', 'completed': True,
              'protocol': PROTOCOL, 'self_tests': checks, 'metric_checks': anchor, 'scenes': scenes,
              'totals': {k: sum(a[k] for a in audits) for k in ('raw_rows','events','final_rows','linked_rows','replaced_rows','unresolved_links')},
              'event_statuses': dict(Counter(e['status'] for e in event_audits)),
              'accepted_at_bound': sum(e.get('at_bound',False) for e in event_audits if e['status']=='accepted'),
              'metrics': metrics, 'delta_vs_native_pp': delta,
              'promotion_gate_passed': delta['0.5'] >= .5 and delta['0.15'] >= 0 and delta['0.25'] >= 0,
              'proposal_cpu_timing_ms': {'total':sum(durations)*1000,'median':float(np.median(durations))*1000,
                                         'p95':float(np.quantile(durations,.95))*1000},
              'scene_audit': audits, 'same_target_geometry_diagnostics': diagnostics,
              'limitations': ['Development ten-scene fixed-history terminal counterfactual; no future association feedback.',
                              'Only exactly linked latest UPDATED native event rows are eligible; not all solver calls represented in final output.',
                              'Native groups may contain identity errors; symmetry alignment does not prove physical corner correspondence.',
                              'Targets are model-box projections, not new independent image-corner measurements.',
                              'Current M1+M2 run differs from this capture; no claim of gain over that run.',
                              'CPU proposal timing excludes inference and downstream work; no end-to-end 15 FPS claim.']}
    write_json(args.output/'results.json', result)
    write_json(args.output/'input_sha256.json', evidence.read_hashes)
    print(json.dumps({k:result[k] for k in ('totals','event_statuses','accepted_at_bound','metrics','delta_vs_native_pp','promotion_gate_passed','proposal_cpu_timing_ms')},indent=2))


if __name__ == '__main__':
    main()
