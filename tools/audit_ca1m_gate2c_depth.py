"""Gate 2 Phase B: depth/surface-consistency selection signal (last open branch).

The user-proposed validator, never tested by Phase A: check visible-surface and
depth consistency of each candidate geometry against sensor depth from frames
that did not generate the candidate. Pure offline audit over the frozen
ca1m_dual baseline, NMS ledgers, and the cached CA-1M frames/poses/depth.

Per output row group (IoU>=0.10 assignment, per-frame representatives), each
candidate (row geometry and every representative child) is projected into up to
5 validation keyframes of its group (a child's own source frame excluded for
that child). Signals per frame, averaged over valid frames:

  containment  fraction of valid region depth pixels whose camera-z lies in
               [zmin-0.10, zmax+0.10] of the candidate's projected corners;
  front        |median of the nearest 30% region depths - zmin| (metres);
  comb         containment - front / 0.5.

Policies swap the row geometry to its argmax-signal child when the signal
exceeds the row's by delta. GT is used only for the oracle reference arm, AUC
labels/margins and the wiring probe. A GT-box depth-consistency probe must pass
before any result is accepted. Row count/order/scores frozen; no new forwards.
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
from tools.audit_ca1m_nms_child_headroom import (
    event_batches, load_summary, sha256, valid_boxes,
)
from tools.true_fusion_audit_core import aabb_iou, class_agnostic_ap
from tools.audit_m1m2_remaining_children import (
    BASE, DIAG, GT_ROOT, SCENES, THRESHOLDS, read_prediction,
)

GATE2_ORACLE_PURE = (2.4530, 4.2247, 6.9382)
ASSIGN_IOU = .10
MAX_VALIDATION_FRAMES = 5
CONTAIN_TOL = .10
FRONT_SHARE = .30
COMB_FRONT_SCALE = .5
DELTA_GRID = (0.0, .02, .05, .10)
SIGNALS = ('cont', 'front', 'comb')
WIRING_FLOOR = .5


def auc(diffs, labels):
    diffs = np.asarray(diffs, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    pos, neg = diffs[labels], diffs[~labels]
    if not len(pos) or not len(neg):
        return None
    order = np.argsort(diffs, kind='mergesort')
    ranks = np.empty(len(diffs), dtype=np.float64)
    sd = diffs[order]
    i = 0
    while i < len(sd):
        j = i
        while j + 1 < len(sd) and sd[j + 1] == sd[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def camera_corners(corners_world, pose):
    """World corners -> camera frame under pose T (either convention probed)."""
    r, t = pose[:3, :3], pose[:3, 3]
    return corners_world @ r.T + t


def project(corners_cam, k):
    z = corners_cam[:, 2]
    u = corners_cam[:, 0] / z * k[0, 0] + k[0, 2]
    v = corners_cam[:, 1] / z * k[1, 1] + k[1, 2]
    return u, v, z


def frame_signals(corners_cam, k, depth):
    """(containment, front) or None when the projection/region is unusable."""
    if corners_cam[:, 2].min() <= .05:
        return None
    u, v, z = project(corners_cam, k)
    h, w = depth.shape
    u0, u1 = int(np.ceil(u.min())), int(np.floor(u.max()))
    v0, v1 = int(np.ceil(v.min())), int(np.floor(v.max()))
    u0, v0 = max(u0, 0), max(v0, 0)
    u1, v1 = min(u1, w), min(v1, h)
    if u1 - u0 < 3 or v1 - v0 < 3:
        return None
    zvals = depth[v0:v1, u0:u1]
    zvals = zvals[zvals > 0].astype(np.float64) / 1000.0
    if len(zvals) < 10:
        return None
    zmin, zmax = float(z.min()), float(z.max())
    containment = float(np.mean((zvals >= zmin - CONTAIN_TOL)
                                & (zvals <= zmax + CONTAIN_TOL)))
    nearest = np.sort(zvals)[:max(1, int(len(zvals) * FRONT_SHARE))]
    front = float(abs(np.median(nearest) - zmin))
    return containment, front


def pose_convention(poses, gt, k, depth):
    """0 = pose maps world->camera (R x + t); 1 = camera->world (R^T (x - t))."""
    mid = poses[len(poses) // 2]
    inverse = np.eye(4)
    inverse[:3, :3] = mid[:3, :3].T
    inverse[:3, 3] = -mid[:3, :3].T @ mid[:3, 3]
    best, best_ok = 0, -1
    for idx, cand in enumerate((mid, inverse)):
        ok = 0
        total = 0
        for box in gt[:50]:
            cam = camera_corners(box, cand)
            total += 1
            if cam[:, 2].min() > .05:
                u, v, _ = project(cam, k)
                h, w = depth.shape
                if (u >= -50).all() and (u <= w + 50).all() \
                        and (v >= -50).all() and (v <= h + 50).all():
                    ok += 1
        if ok > best_ok:
            best, best_ok = idx, ok
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f'Refusing to overwrite {args.output}')
    scenes = SCENES.read_text().split()
    assert len(scenes) == len(set(scenes)) == 107
    inputs = {str(p.resolve()): sha256(p) for p in (
        Path(__file__), SCENES,
        ROOT / 'tools/audit_ca1m_nms_child_headroom.py',
        ROOT / 'tools/true_fusion_audit_core.py',
        ROOT / 'tools/audit_m1m2_remaining_children.py')}
    arm_names = ([f'{s}_d{d:g}' for s in SIGNALS for d in DELTA_GRID]
                 + ['oracle_pure'])
    arms = {name: {} for name in arm_names}
    baseline, gts = {}, {}
    stats = {name: Counter() for name in arm_names}
    auc_pairs = {f'rowchild_{s}': [] for s in SIGNALS}
    strat = {f'rowchild_{s}_{b}': [] for s in SIGNALS
             for b in ('m0_5', 'm5_10', 'm10p')}
    wiring_hits, wiring_total = 0, 0
    conventions = Counter()

    def load_depth(scene, frame, cache, hashed):
        path = GT_ROOT / scene / 'depth' / f'{frame}.png'
        if frame not in cache:
            import cv2
            array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if array is None:
                raise ValueError(f'unreadable depth: {path}')
            cache[frame] = array
            key = str(path.resolve())
            if key not in hashed:
                hashed[key] = sha256(path)
        return cache[frame]

    for ordinal, scene in enumerate(scenes, 1):
        pred_path = BASE / f'{scene}_boxes.pkl'
        gt_path = GT_ROOT / scene / 'after_filter_boxes.npy'
        summary_path = DIAG / f'{scene}_pvq_ar_summary.json'
        ledger = DIAG / f'{scene}_pvq_nms.jsonl'
        poses_path = GT_ROOT / scene / 'all_poses.npy'
        k_path = GT_ROOT / scene / 'K_depth.txt'
        for path in (pred_path, gt_path, summary_path, poses_path, k_path):
            inputs[str(path.resolve())] = sha256(path)
        if ledger.exists():
            inputs[str(ledger.resolve())] = sha256(ledger)
        boxes, scores = read_prediction(pred_path)
        gt = valid_boxes(np.load(gt_path, allow_pickle=False), str(gt_path))
        poses = np.load(poses_path, allow_pickle=False)
        assert poses.ndim == 3 and poses.shape[1:] == (4, 4), scene
        k = np.loadtxt(k_path)
        assert k.shape == (3, 3), scene
        depth_cache, depth_hashed = {}, dict(inputs)
        n = load_summary(summary_path, scene)
        records = [r for batch in event_batches(ledger, scene, n) for r in batch]
        assert not records or max(
            max(r['keyframe_id'], r['child_frame_id']) for r in records) \
            < len(poses), scene
        probe_frame = len(poses) // 2
        probe_depth = load_depth(scene, probe_frame, depth_cache, depth_hashed)
        convention = pose_convention(poses, gt, k, probe_depth)
        conventions[convention] += 1

        def to_camera(corners_world, frame):
            pose = poses[frame]
            if convention == 1:
                inv = np.eye(4)
                inv[:3, :3] = pose[:3, :3].T
                inv[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
                pose = inv
            return camera_corners(corners_world, pose)

        # GT wiring probe: z-surface of GT boxes must sit in their z-range.
        for box in gt[::max(1, len(gt) // 30)][:30]:
            cam = to_camera(box, probe_frame)
            if cam[:, 2].min() <= .05:
                continue
            u, v, z = project(cam, k)
            h, w = probe_depth.shape
            u0, u1 = max(int(np.ceil(u.min())), 0), min(int(np.floor(u.max())), w)
            v0, v1 = max(int(np.ceil(v.min())), 0), min(int(np.floor(v.max())), h)
            if u1 - u0 < 3 or v1 - v0 < 3:
                continue
            zvals = probe_depth[v0:v1, u0:u1]
            zvals = zvals[zvals > 0].astype(np.float64) / 1000.0
            if len(zvals) < 10:
                continue
            nearest = np.sort(zvals)[:max(1, int(len(zvals) * FRONT_SHARE))]
            zs = np.median(nearest)
            wiring_total += 1
            if z.min() - .3 <= zs <= z.max() + .3:
                wiring_hits += 1

        child = valid_boxes([r['child_corners_world'] for r in records], scene)
        cscores = np.asarray([r['child_score'] for r in records], dtype=float)
        frames = np.asarray([r['child_frame_id'] for r in records], dtype=int)
        iou_rc = aabb_iou(boxes, child)
        assign = {int(j): int(iou_rc[:, j].argmax())
                  for j in np.flatnonzero(iou_rc.max(0) >= ASSIGN_IOU)}
        groups = {}
        for j, r in assign.items():
            groups.setdefault(r, []).append(j)
        ciou = aabb_iou(child, gt)
        biou_full = aabb_iou(boxes, gt)
        row_gt = biou_full.argmax(1)
        new_boxes = {name: boxes.copy() for name in arm_names}
        for r, js in groups.items():
            g = int(row_gt[r])
            reps = {}
            for j in js:
                f = int(frames[j])
                if f not in reps or (cscores[j], -reps[f]) > (cscores[reps[f]], -j):
                    reps[f] = j
            rep_js = sorted(reps.values())
            pool = sorted({int(records[j]['keyframe_id']) for j in js}
                          | {int(frames[j]) for j in rep_js})
            if len(pool) > MAX_VALIDATION_FRAMES:
                idx = np.linspace(0, len(pool) - 1, MAX_VALIDATION_FRAMES)
                pool = [pool[int(round(i))] for i in idx]
            scored = {}
            for cand_id, corners in [('row', boxes[r])] + [
                    (int(j), child[j]) for j in rep_js]:
                per_frame = {0: [], 1: []}
                for frame in pool:
                    if cand_id != 'row' and int(frames[cand_id]) == frame:
                        continue
                    result = frame_signals(
                        to_camera(corners, frame), k,
                        load_depth(scene, frame, depth_cache, depth_hashed))
                    if result is None:
                        continue
                    per_frame[0].append(result[0])
                    per_frame[1].append(result[1])
                if per_frame[0]:
                    cont = float(np.mean(per_frame[0]))
                    front = float(np.mean(per_frame[1]))
                    scored['row' if cand_id == 'row' else int(cand_id)] = {
                        'cont': cont, 'front': -front,
                        'comb': cont - front / COMB_FRONT_SCALE}
            row_scores = scored.pop('row', None)
            row_iou = float(biou_full[r, g])
            row_err = float(np.linalg.norm(
                boxes[r].mean(0) - gt[g].mean(0)) * 100)
            best_all = int(np.asarray(js)[ciou[np.asarray(js), g].argmax()])
            if ciou[best_all, g] > biou_full[r, g]:
                new_boxes['oracle_pure'][r] = child[best_all]
                stats['oracle_pure']['swaps'] += 1
            if row_scores is None:
                continue
            for j in rep_js:
                if int(j) not in scored:
                    continue
                label = bool(ciou[j, g] > row_iou)
                margin = row_err - float(np.linalg.norm(
                    child[j].mean(0) - gt[g].mean(0)) * 100)
                for signal in SIGNALS:
                    pair = (scored[int(j)][signal] - row_scores[signal],
                            label, margin)
                    auc_pairs[f'rowchild_{signal}'].append(pair)
                    if margin >= 10:
                        strat[f'rowchild_{signal}_m10p'].append(pair)
                    elif margin >= 5:
                        strat[f'rowchild_{signal}_m5_10'].append(pair)
                    elif margin >= 0:
                        strat[f'rowchild_{signal}_m0_5'].append(pair)
            defined = [(v[signal], int(j)) for j, v in scored.items()]
            if not defined:
                continue
            for signal in SIGNALS:
                best_s, best_j = max(defined, key=lambda p: (p[0], -p[1]))
                for d in DELTA_GRID:
                    if best_s - row_scores[signal] >= d:
                        new_boxes[f'{signal}_d{d:g}'][r] = child[best_j]
                        stats[f'{signal}_d{d:g}']['swaps'] += 1
                        if ciou[best_j, g] > row_iou:
                            stats[f'{signal}_d{d:g}']['oracle_positive'] += 1
                        else:
                            stats[f'{signal}_d{d:g}']['oracle_negative'] += 1
        inputs.update(depth_hashed)
        for name in arm_names:
            arms[name][scene] = (new_boxes[name], scores)
        baseline[scene], gts[scene] = (boxes, scores), gt
        depth_cache.clear()
        if ordinal % 10 == 0 or ordinal == len(scenes):
            print(f'{ordinal}/107 scenes audited '
                  f'(wiring {wiring_hits}/{wiring_total})', flush=True)
    wiring_rate = wiring_hits / max(wiring_total, 1)
    assert wiring_rate > WIRING_FLOOR, f'wiring probe failed: {wiring_rate:.3f}'
    metrics = {'baseline': {str(t): class_agnostic_ap(baseline, gts, t) for t in THRESHOLDS}}
    assert np.allclose([metrics['baseline'][str(t)]['ap'] for t in THRESHOLDS],
                       [46.13, 38.34, 16.85], atol=.005, rtol=0)
    for name, predictions in arms.items():
        metrics[name] = {}
        for t in THRESHOLDS:
            value = class_agnostic_ap(predictions, gts, t)
            value['delta_ap'] = value['ap'] - metrics['baseline'][str(t)]['ap']
            value['delta_tp'] = value['tp'] - metrics['baseline'][str(t)]['tp']
            value['delta_fp'] = value['fp'] - metrics['baseline'][str(t)]['fp']
            metrics[name][str(t)] = value
    assert np.allclose([metrics['oracle_pure'][str(t)]['delta_ap'] for t in THRESHOLDS],
                       GATE2_ORACLE_PURE, atol=1e-3, rtol=0), 'oracle_pure diverges'

    def auc_block(pairs):
        if not pairs:
            return {'n': 0}
        return {'n': len(pairs), 'auc': auc([p[0] for p in pairs],
                                            [p[1] for p in pairs])}
    signal_summary = {}
    for signal in SIGNALS:
        signal_summary[signal] = {
            'rowchild': auc_block(auc_pairs[f'rowchild_{signal}']),
            'rowchild_margin_bins': {
                b: auc_block(strat[f'rowchild_{signal}_{b}'])
                for b in ('m0_5', 'm5_10', 'm10p')},
        }
    for path, digest in inputs.items():
        assert sha256(path) == digest, f'Input changed: {path}'
    result = {
        'schema': 'boxfusion.gate2c_depth.v1', 'completed': True,
        'wiring_probe': {'gt_surface_in_range_rate': wiring_rate,
                         'floor': WIRING_FLOOR, 'pose_conventions': dict(conventions)},
        'signals': {
            'cont': 'fraction of region depth pixels within candidate z-range (+/-10cm)',
            'front': '-|median nearest-30% depth - zmin| (metres; higher better)',
            'comb': 'cont - front_penalty/0.5',
        },
        'validation_frames': f'up to {MAX_VALIDATION_FRAMES} group keyframes; '
                             'a child\'s own source frame excluded for that child; '
                             'the row uses all of them',
        'delta_grid': list(DELTA_GRID),
        'signal_summary': signal_summary,
        'stats': {k: dict(v) for k, v in stats.items()},
        'limits': [
            'Projection uses the axis-aligned rect of projected corners (not the '
            'hull): background pixels contaminate containment equally for all '
            'candidates but favour deeper boxes.',
            'Depth assumed z-depth in mm uint16; poses probed per scene, majority '
            'convention asserted by the GT wiring probe.',
            'The row is validated on group keyframes that may have contributed to '
            'its own PFO; strict out-of-sample holds only for the children.',
            'GT used for oracle arm, AUC labels/margins, wiring probe only; deltas '
            'tuned on dev scenes; CA-1M only; bounded FPS not measured.',
        ],
        'metrics': metrics,
    }
    args.output.mkdir(parents=True)
    for name, data in (('results.json', json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'),
                       ('input_sha256.json', json.dumps(inputs, ensure_ascii=False, indent=2) + '\n')):
        with (args.output / name).open('x') as handle:
            handle.write(data)
    print(json.dumps({
        'wiring': wiring_rate,
        'auc': {s: {'rowchild': signal_summary[s]['rowchild'].get('auc'),
                    'bins': {b: signal_summary[s]['rowchild_margin_bins'][b].get('auc')
                             for b in ('m0_5', 'm5_10', 'm10p')}}
                for s in SIGNALS},
        'deltas': {name: [round(metrics[name][str(t)]['delta_ap'], 4)
                          for t in THRESHOLDS] for name in arm_names}},
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
