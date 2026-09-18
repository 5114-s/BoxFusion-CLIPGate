"""Frozen Step 1c transfer to ScanNet paper100, with no-GT GPU workers.

Native RGB coordinates and calibrated RGB-to-depth rays are used. Both sensor
extrinsics must be identity (as in the exported ScanNet data). Final-map masks
and all-frame recurrence make this an offline experiment. No GT birth filter,
extra NMS, or score tuning is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_ca1m_nms_child_headroom import sha256, valid_boxes
from tools.audit_m1m2_remaining_children import read_prediction
from tools.ca1m_prenms_query_core import box_iou2d, project_box
from tools.ca1m_seedless_trigger_core import recurrence_counts, top_m, add_clusters
from tools.true_fusion_audit_core import aabb_iou, class_agnostic_ap
from tools.validate_ca1m_prenms_query import DenseCapture, build_lifter, lift

VARIANTS = ('trigger_m300', 'trigger_m150', 'score_m300', 'score_m150')
THRESHOLDS = (.15, .25, .5)


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def check_hashes(values):
    for path, expected in values.items():
        if sha256(path) != expected:
            raise ValueError(f'Input changed: {path}')


def forbid_gt(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes)):
        name = str(args[0])
        if any(x in name for x in ('_bbox.npy', '_vert.npy', '_ins_label.npy',
                                  '_sem_label.npy', '.aggregation.json',
                                  'after_filter_boxes.npy', 'full_annotations.json')):
            raise RuntimeError(f'GT access forbidden in worker: {name}')


def load_frame(root, frame):
    with Image.open(root / 'color' / f'{frame}.jpg') as image:
        pil = image.convert('RGB')
    with Image.open(root / 'depth' / f'{frame}.png') as image:
        depth = np.asarray(image).astype(np.float32) / 1000.0
    if depth.ndim != 2:
        raise ValueError('Expected HxW depth in metres')
    return pil, np.asarray(pil), depth


def anchor_points(centers, ids, depth, pose, kr, kd, rgb_shape):
    """Calibrated coincident optical frames; do not assume scaled intrinsics."""
    if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.floating):
        raise ValueError('Expected HxW floating point depth in metres')
    height, width = rgb_shape[:2]
    dh, dw = depth.shape
    inverse = np.linalg.inv(kr)
    mapping = kd @ inverse
    kept, worlds, depths = [], [], []
    counts = Counter(tested=len(ids), no_valid_depth=0, out_of_range=0, outside_image=0)
    for idx in ids:
        u, v = centers[idx]
        if not (np.isfinite([u, v]).all() and 0 <= u < width and 0 <= v < height):
            counts['outside_image'] += 1
            continue
        mapped = mapping @ [u, v, 1.0]
        du, dv = mapped[:2] / mapped[2]
        if not (0 <= du < dw and 0 <= dv < dh):
            counts['outside_image'] += 1
            continue
        x, y = int(du), int(dv)
        patch = depth[max(0, y-2):min(dh, y+3), max(0, x-2):min(dw, x+3)]
        values = patch[np.isfinite(patch) & (patch > 0)]
        if not len(values):
            counts['no_valid_depth'] += 1
            continue
        d = float(np.median(values))
        if not .2 < d < 12.0:
            counts['out_of_range'] += 1
            continue
        world = (pose @ np.r_[inverse @ [u, v, 1.] * d, 1.])[:3]
        if not np.isfinite(world).all():
            raise ValueError('Nonfinite world point')
        kept.append(idx)
        worlds.append(world)
        depths.append(d)
    counts['valid_world_points'] = len(kept)
    return (np.asarray(kept, dtype=np.int64), np.asarray(worlds).reshape(-1, 3),
            np.asarray(depths), dict(counts))


def prepare(args):
    scenes = args.scenes.read_text().split()
    if len(scenes) != args.expected_scenes or len(set(scenes)) != len(scenes):
        raise ValueError('Wrong scene count or duplicates')
    frames, excluded, hashes = {}, {}, {}
    for scene in scenes:
        root = args.data / scene
        hashes[str(args.baseline / f'{scene}_boxes.pkl')] = sha256(args.baseline / f'{scene}_boxes.pkl')
        for sensor in ('color', 'depth'):
            for kind in ('intrinsic', 'extrinsic'):
                path = root / 'intrinsic' / f'{kind}_{sensor}.txt'
                value = np.loadtxt(path).reshape(4, 4)
                if not np.isfinite(value).all():
                    raise ValueError(f'Invalid calibration {path}')
                if kind == 'extrinsic' and not np.allclose(value, np.eye(4), atol=1e-8, rtol=0):
                    raise ValueError(f'Nonidentity sensor extrinsic requires registration: {path}')
                hashes[str(path)] = sha256(path)
        frames[scene], excluded[scene] = [], []
        # Match the native integrated_online.py schedule; fail on missing files.
        n = len(list((root / 'color').glob('*.jpg')))
        if n == 0:
            raise ValueError(f'No RGB frames: {scene}')
        for frame in range(0, n, 25):
            pp = root / 'pose' / f'{frame}.txt'
            for path in (pp, root / 'color' / f'{frame}.jpg', root / 'depth' / f'{frame}.png'):
                if not path.is_file():
                    raise FileNotFoundError(path)
                hashes[str(path)] = sha256(path)
            pose = np.loadtxt(pp).reshape(4, 4)
            if not np.isfinite(pose).all():
                excluded[scene].append(frame)
                continue
            if not np.allclose(pose[3], [0, 0, 0, 1]):
                raise ValueError(f'Invalid homogeneous pose {pp}')
            np.linalg.inv(pose)
            frames[scene].append(frame)
        if len(frames[scene]) < 3:
            raise ValueError(f'Insufficient valid frames: {scene}')
        gt = ROOT / 'evaluation/data_util/scannet_train_detection_data' / f'{scene}_bbox.npy'
        if not gt.is_file() or not (root / f'{scene}.txt').is_file():
            raise FileNotFoundError(f'Missing evaluation inputs: {scene}')
    sources = [Path(__file__), args.scenes] + [ROOT / x for x in (
        'tools/ca1m_seedless_trigger_core.py', 'tools/ca1m_prenms_query_core.py',
        'tools/validate_ca1m_prenms_query.py', 'tools/audit_ca1m_nms_child_headroom.py',
        'tools/audit_m1m2_remaining_children.py', 'tools/true_fusion_audit_core.py',
        'boxfusion/boxer_lifter.py', 'third_party/WeDetect/wedetect_uni_infer.py',
        'third_party/WeDetect/wedetect_base_uni.pth', 'evaluation/eval_scannet.py',
        'evaluation/utils/eval_det.py', 'evaluation/utils/box_util.py',
        'evaluation/utils/ap_helper.py', 'evaluation/utils/utils.py',
        'evaluation/data_util/dataset.py', 'evaluation/data_util/model_util_scannet.py')]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'raw').mkdir()
    (args.output / 'scenes').mkdir()
    (args.output / 'scenes.txt').write_text('\n'.join(scenes) + '\n')
    write_json(args.output / 'protocol.json', {
        'schema': 'boxfusion.scannet_seedless.v1', 'scenes': scenes, 'frames': frames,
        'scene_count': len(scenes), 'frame_count': sum(map(len, frames.values())),
        'excluded_nonfinite_pose_frames': excluded, 'baseline': str(args.baseline),
        'data': str(args.data), 'source_sha256': {str(p.resolve()): sha256(p) for p in sources},
        'input_sha256': hashes, 'parameters': {'gap': 25, 'residual_iou': .2,
        'recurrence_radius_native_rgb_px': 30., 'recurrence_other_frames': 2,
        'budgets': [300, 150], 'voxel_m': .3, 'distinct_cluster_frames': 3, 'birth_score': .1},
        'rgb_depth_mapping': 'K_depth @ inverse(K_color); identity optical extrinsics checked',
        'rgb_resize': False, 'depth_unit': 'metres, source millimetres divided once',
        'birth_policy': 'all >=3-frame voxels; highest raw-score exemplar; no GT, NMS or cap',
        'offline_final_map_mask': True, 'all_frame_recurrence': True,
        'created_unix': time.time(),
    })
    print(f'PREPARED {len(scenes)} scenes, {sum(map(len, frames.values()))} frames', flush=True)


def select_scene(protocol, output, scene):
    root = Path(protocol['data']) / scene
    frames = protocol['frames'][scene]
    boxes, _ = read_prediction(Path(protocol['baseline']) / f'{scene}_boxes.pkl')
    kr = np.loadtxt(root / 'intrinsic/intrinsic_color.txt')[:3, :3]
    kd = np.loadtxt(root / 'intrinsic/intrinsic_depth.txt')[:3, :3]
    poses = {f: np.loadtxt(root / 'pose' / f'{f}.txt') for f in frames}
    inverses = {f: np.linalg.inv(p) for f, p in poses.items()}
    anchors, scores, trees, points, shapes, stats = {}, {}, {}, {}, {}, {}
    for frame in frames:
        with np.load(output / 'raw' / scene / f'raw_{frame:06d}.npz') as raw:
            anchors[frame], scores[frame] = raw['boxes'].astype(float), raw['scores'].astype(float)
        _, rgb, depth = load_frame(root, frame)
        h, w = rgb.shape[:2]
        shapes[frame] = (h, w)
        centers = (anchors[frame][:, :2] + anchors[frame][:, 2:]) / 2
        valid = ((anchors[frame][:, 2:] > anchors[frame][:, :2]).all(1)
                 & (centers[:, 0] >= 0) & (centers[:, 0] < w)
                 & (centers[:, 1] >= 0) & (centers[:, 1] < h))
        projected = [project_box(b, poses[frame], kr, w, h) for b in boxes]
        projected = np.asarray([b for b in projected if b is not None]).reshape(-1, 4)
        residual = valid & (box_iou2d(anchors[frame], projected).max(1) < .2 if len(projected) else True)
        trees[frame] = cKDTree(centers[residual]) if residual.any() else None
        ids, worlds, depths, sampling = anchor_points(centers, np.flatnonzero(residual),
                                                     depth, poses[frame], kr, kd, rgb.shape)
        points[frame] = ids, worlds
        stats[frame] = {'frame': frame, 'raw': len(centers), 'valid_2d': int(valid.sum()),
                        'projected_map_rows': len(projected), 'residual': int(residual.sum()),
                        'rgb_hw': [h, w], 'depth_hw': list(depth.shape),
                        'depth_sampling': sampling,
                        'sampled_depth_median_m': float(np.median(depths)) if len(depths) else None}
    plans = {}
    for frame in frames:
        ids, worlds = points[frame]
        hits = recurrence_counts(worlds, frame, frames, inverses, kr, shapes, trees, 30.)
        recurrent = ids[hits >= 2]
        trigger = top_m(recurrent, scores[frame], 300)
        score = top_m(np.arange(len(scores[frame])), scores[frame], 300)
        plans[frame] = {f'{kind}_m{m}': values[:m] for kind, values in
                        [('trigger', trigger), ('score', score)] for m in (300, 150)}
        stats[frame].update(recurrent=len(recurrent),
                            recurrence_other_frame_histogram=dict(Counter(map(int, hits))),
                            selected={v: len(ids) for v, ids in plans[frame].items()},
                            selected_anchor_ids={v: ids.tolist() for v, ids in plans[frame].items()})
    directory = output / 'scenes' / scene
    write_json(directory / 'selection.json', {'scene': scene, 'gt_access': False,
                                              'frames': [stats[f] for f in frames]})
    print(f'{scene}: selected {len(frames)} frames, recurrent/frame='
          f'{np.mean([r["recurrent"] for r in stats.values()]):.1f}', flush=True)
    return anchors, scores, plans, poses, kr, kd


def worker(args):
    protocol = json.loads((args.output / 'protocol.json').read_text())
    check_hashes(protocol['source_sha256'])
    if not 0 <= args.worker_index < args.workers:
        raise ValueError('Invalid worker partition')
    scenes = protocol['scenes'][args.worker_index::args.workers]
    sys.addaudithook(forbid_gt)
    detector, adapter = None, None
    start = time.time()
    for ordinal, scene in enumerate(scenes, 1):
        directory = args.output / 'scenes' / scene
        rawdir = args.output / 'raw' / scene
        complete = directory / 'complete.json'
        if complete.exists():
            check_hashes(json.loads(complete.read_text())['sha256'])
            continue
        if directory.exists() or rawdir.exists():
            raise ValueError(f'Partial scene requires explicit recovery: {scene}')
        check_hashes({p: h for p, h in protocol['input_sha256'].items()
                      if f'/{scene}/' in p or p.endswith(f'/{scene}_boxes.pkl')})
        directory.mkdir()
        rawdir.mkdir()
        if detector is None:
            detector = DenseCapture()
        frames = protocol['frames'][scene]
        root = Path(protocol['data']) / scene
        scene_start = time.time()
        for ordinal_frame, frame in enumerate(frames, 1):
            pil, _, _ = load_frame(root, frame)
            value = detector.forward(pil)
            if value['boxes'].shape != (8400, 4) or value['scores'].shape != (8400,):
                raise ValueError('Unexpected raw anchor cardinality')
            if not np.isfinite(value['boxes']).all() or not np.isfinite(value['scores']).all():
                raise ValueError('Nonfinite raw anchors')
            np.savez_compressed(rawdir / f'raw_{frame:06d}.npz', boxes=value['boxes'], scores=value['scores'])
            if ordinal_frame % 25 == 0 or ordinal_frame == len(frames):
                print(f'{scene}: capture {ordinal_frame}/{len(frames)}', flush=True)
        anchors, scores, plans, poses, kr, kd = select_scene(protocol, args.output, scene)
        if adapter is None:
            adapter = build_lifter(args.output / f'worker_{args.worker_index}')
        clusters = {v: {} for v in VARIANTS}
        for ordinal_frame, frame in enumerate(frames, 1):
            union = np.unique(np.concatenate(list(plans[frame].values())))
            _, rgb, depth = load_frame(root, frame)
            corners, _, _ = lift(adapter, scene, frame, rgb, depth, kr, kd,
                                  poses[frame], anchors[frame][union])
            corners = valid_boxes(corners, f'{scene}/{frame}')
            if len(corners) != len(union):
                raise ValueError('Lifter changed cardinality')
            np.savez_compressed(directory / f'lifted_{frame:06d}.npz', anchor_ids=union, corners=corners)
            for v in VARIANTS:
                ids = plans[frame][v]
                positions = np.searchsorted(union, ids)
                if not np.array_equal(union[positions], ids):
                    raise ValueError('Anchor identity mismatch')
                add_clusters(clusters[v], frame, ids, corners[positions], scores[frame][ids], .3)
            if ordinal_frame % 10 == 0 or ordinal_frame == len(frames):
                print(f'{scene}: lift {ordinal_frame}/{len(frames)}', flush=True)
        births = {}
        for v in VARIANTS:
            confirmed = [(k, c) for k, c in sorted(clusters[v].items()) if len(c['frames']) >= 3]
            np.savez_compressed(directory / f'births_{v}.npz',
                corners=np.asarray([c['box'] for _, c in confirmed]).reshape(-1, 8, 3),
                voxel_keys=np.asarray([k for k, _ in confirmed], dtype=int).reshape(-1, 3),
                distinct_frames=np.asarray([len(c['frames']) for _, c in confirmed], dtype=int))
            births[v] = {'births': len(confirmed), 'all_voxels': len(clusters[v]),
                         'selected_observations': sum(c['observations'] for c in clusters[v].values())}
        hashes = {str(p): sha256(p) for p in list(directory.iterdir()) + list(rawdir.iterdir()) if p.is_file()}
        write_json(complete, {'completed': True, 'scene': scene, 'frames': len(frames),
            'gt_access': False, 'worker_index': args.worker_index, 'birth_stats': births,
            'seconds': time.time() - scene_start, 'sha256': hashes})
        print(f'WORKER {args.worker_index} [{ordinal}/{len(scenes)}] {scene} COMPLETE '
              f'{time.time()-scene_start:.1f}s births={ {v: b["births"] for v, b in births.items()} }', flush=True)
        if args.max_scenes and ordinal >= args.max_scenes:
            return
    check_hashes(protocol['source_sha256'])
    write_json(args.output / f'worker_{args.worker_index}_complete.json',
               {'completed': True, 'scenes': scenes, 'seconds': time.time()-start})


def evaluate(args):
    protocol = json.loads((args.output / 'protocol.json').read_text())
    check_hashes(protocol['source_sha256'])
    check_hashes(protocol['input_sha256'])
    scenes = protocol['scenes']
    if args.stage != 'baseline':
        for s in scenes:
            complete = json.loads((args.output / 'scenes' / s / 'complete.json').read_text())
            if not complete['completed'] or complete['frames'] != len(protocol['frames'][s]):
                raise ValueError(f'Incomplete scene {s}')
            check_hashes(complete['sha256'])
    # Import the repository's exact ScanNet label parser and transformations.
    sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != ROOT / 'tools']
    sys.path.insert(0, str(ROOT / 'evaluation'))
    from utils.ap_helper import parse_groundtruths
    from utils.utils import flip_axis_to_camera, obb_to_aabb_corners, reorganize_obb_to_aabb
    from data_util.dataset import ScannetDetectionDataset
    from data_util.model_util_scannet import ScannetDatasetConfig
    from torch.utils.data._utils.collate import default_collate
    import eval_det
    from tools.eval_scannet_causal_dynamic_ap import read_alignment
    eval_det.tqdm = lambda x: x

    def transform(boxes, align):
        if not len(boxes):
            return np.empty((0, 8, 3))
        value = np.transpose(align[None, :3, :3] @ np.transpose(boxes, (0, 2, 1)), (0, 2, 1))
        value += align[None, :3, 3]
        return reorganize_obb_to_aabb(obb_to_aabb_corners(flip_axis_to_camera(value)))

    gtroot = ROOT / 'evaluation/data_util/scannet_train_detection_data'
    dataset = ScannetDetectionDataset('val', num_points=40000, augment=False,
                                      use_color=False, use_height=True, data_path=str(gtroot))
    if not set(scenes) <= set(dataset.scan_names):
        raise ValueError('Missing GT scenes')
    dataset.scan_names = scenes
    cfg = {'dataset_config': ScannetDatasetConfig()}
    variants = ('baseline',) if args.stage == 'baseline' else ('baseline',) + VARIANTS
    predictions = {v: {} for v in variants}
    gts, counts, stats, inputs = {}, {v: Counter() for v in variants}, [], {}
    predroot = args.output / 'predictions'
    if args.stage != 'baseline':
        predroot.mkdir(exist_ok=False)
        for v in variants:
            (predroot / v).mkdir()
    np.random.seed(0)
    for index, scene in enumerate(scenes):
        meta = Path(protocol['data']) / scene / f'{scene}.txt'
        align = read_alignment(meta)
        inputs[str(meta)] = sha256(meta)
        inputs[str(gtroot / f'{scene}_bbox.npy')] = sha256(gtroot / f'{scene}_bbox.npy')
        gt = valid_boxes([r[1] for r in parse_groundtruths(default_collate([dataset[index]]), cfg)[0]], scene)
        gts[scene] = gt
        native, confidence = read_prediction(Path(protocol['baseline']) / f'{scene}_boxes.pkl')
        directory = args.output / 'scenes' / scene
        if args.stage != 'baseline':
            complete = json.loads((directory / 'complete.json').read_text())
            stats.extend(json.loads((directory / 'selection.json').read_text())['frames'])
        for v in variants:
            boxes, scores = native, confidence
            if v != 'baseline':
                with np.load(directory / f'births_{v}.npz') as data:
                    births = valid_boxes(data['corners'], v)
                counts[v].update(complete['birth_stats'][v])
                iou = aabb_iou(transform(births, align), gt)
                counts[v]['births_unmatched_gt15'] += int((iou.max(1) <= .15).sum()) if len(gt) else len(births)
                boxes, scores = np.concatenate([native, births]), np.r_[confidence, np.full(len(births), .1)]
            predictions[v][scene] = transform(boxes, align), scores
            if args.stage != 'baseline':
                path = predroot / v / f'{scene}_boxes.pkl'
                with path.open('xb') as f:
                    pickle.dump([[(0, b, float(s)) for b, s in zip(boxes, scores)]], f)
                inputs[str(path)] = sha256(path)
    metrics, parity = {}, {}
    for v in variants:
        metrics[v], parity[v] = {}, {}
        # Run the exact anchor metric on every arm; vectorized metric supplies
        # counts only after its AP agrees with the anchor.
        for threshold in THRESHOLDS:
            value = class_agnostic_ap(predictions[v], gts, threshold)
            anchor_ap = eval_det.eval_det_cls({s: list(zip(*p)) for s, p in predictions[v].items()},
                {s: list(g) for s, g in gts.items()}, ovthresh=threshold,
                get_iou_func=eval_det.get_iou_obb)[2] * 100
            parity[v][str(threshold)] = abs(value['ap'] - anchor_ap)
            if parity[v][str(threshold)] > 1e-6:
                raise ValueError(f'AP metric mismatch: {v}/{threshold}')
            value['ap'] = float(anchor_ap)
            if v != 'baseline':
                value['delta_ap'] = value['ap'] - metrics['baseline'][str(threshold)]['ap']
            metrics[v][str(threshold)] = value
            print(f'AP {v} IoU={threshold}: {value}', flush=True)
    expected = [41.298901361344126, 37.302988324064584, 18.512383520854536]
    if (Path(protocol['baseline']) == ROOT / 'results/scannet_m2nl_m5_dual_full100/persistent'
            and scenes == (ROOT / 'evaluation/data_util/meta_data/scannetv2_val.txt').read_text().split()):
        actual = [metrics['baseline'][str(t)]['ap'] for t in THRESHOLDS]
        if not np.allclose(actual, expected, atol=1e-6, rtol=0):
            raise ValueError(f'Published baseline parity failed: {actual}')
    check_hashes(inputs)
    filename = 'baseline_check.json' if args.stage == 'baseline' else 'results.json'
    result = {'completed': True, 'scene_count': len(scenes), 'frame_count': protocol['frame_count'],
        'metrics': metrics, 'birth_stats': {v: dict(c) for v, c in counts.items()},
        'anchor_ap_parity_errors': parity,
        'mean_budget': {v: float(np.mean([r['selected'][v] for r in stats])) for v in VARIANTS} if stats else {},
        'limits': ['Offline final-map masks and all-frame recurrence; no causal or FPS claim.',
                   'Validation scenes used in development; not an untouched test set.',
                   'Unmatched births are not semantic background labels.',
                   '30px radius is in native RGB pixels; angular radius differs across datasets.']}
    write_json(args.output / filename, result)
    write_json(args.output / (filename + '.inputs.json'), inputs)
    print(f'SAVED {args.output / filename}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'worker', 'baseline', 'evaluate'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scenes', type=Path, default=ROOT / 'evaluation/data_util/meta_data/scannetv2_val.txt')
    p.add_argument('--data', type=Path, default=Path('/extra/ZhaoX/scannet_data/scans'))
    p.add_argument('--baseline', type=Path, default=ROOT / 'results/scannet_m2nl_m5_dual_full100/persistent')
    p.add_argument('--expected-scenes', type=int, default=100)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--worker-index', type=int, default=0)
    p.add_argument('--max-scenes', type=int, default=0, help='Smoke run; no worker-complete marker')
    args = p.parse_args()
    args.output = args.output.resolve()
    {'prepare': prepare, 'worker': worker, 'baseline': evaluate, 'evaluate': evaluate}[args.stage](args)


if __name__ == '__main__':
    main()
