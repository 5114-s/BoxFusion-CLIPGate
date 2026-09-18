#!/usr/bin/env python3
"""Semantic readout of the FINAL tier-2 stack (unified M300, flat 0.05).

Extends reports/scannet_semantic_20260915 to the tier-2 arm without re-running
the detector and without re-encoding the 2,948 prefix boxes: their frozen CLIP
features/labels are reused after a per-scene geometry-hash check, and only the
tier-2 births go through the identical automatic crop + frozen CLIP protocol.
The arm is bound to the reconciled ledger row by asserting the audit-path
class-agnostic AP reproduces 43.3503/39.1471/19.5612.

Development-set semantic detection under a fixed 18-class table; not a
novel-class or online claim.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from paper_eval_core import best_view
from true_fusion_audit_core import class_agnostic_ap
from audit_final_ledger import build_tier2_raw, build_scannet_eval
from audit_m1m2_remaining_children import read_prediction
from run_scannet_semantic_table import (NAMES, NYU, FRAMES, ground_truth,
                                        alignment)

BASELINE = ROOT / 'results/scannet_m2nl_m5_dual_full100/persistent'
SEM = ROOT / 'reports/scannet_semantic_20260915'
OUT = ROOT / 'reports/tier2_semantic_20260915'
EXPECTED = {'0.15': 43.350321, '0.25': 39.147075, '0.5': 19.561201}
SEM_PUBLISHED = {'mAP': {'0.15': 23.9586, '0.25': 21.1467, '0.5': 14.2563}}
THRESHOLDS = (0.15, 0.25, 0.5)


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def keyframes(scene):
    directory = FRAMES / scene / 'frames'
    colors = sorted((directory / 'color').glob('*.jpg'),
                    key=lambda p: int(p.stem))[::25]
    paths, poses = [], []
    for path in colors:
        pp = directory / 'pose' / f'{path.stem}.txt'
        if pp.exists():
            pose = np.loadtxt(pp).reshape(4, 4)
            if np.isfinite(pose).all():
                paths.append(path)
                poses.append(pose)
    if not paths:
        raise ValueError(f'No valid keyframes: {scene}')
    with Image_open(paths[0]) as im:
        width, height = im.size
    intrinsic = np.loadtxt(directory / 'intrinsic/intrinsic_color.txt')[:3, :3]
    return paths, poses, intrinsic, width, height


def Image_open(path):
    from PIL import Image
    return Image.open(path)


def classify(scenes, merged, batch_size):
    import torch
    import open_clip
    from PIL import Image
    cache = OUT / 'semantic_cache'
    cache.mkdir(exist_ok=True)
    model = prep = text = None
    startup = None
    for si, scene in enumerate(scenes):
        dest = cache / f'{scene}.npz'
        if dest.exists():
            print(f'CACHED {si + 1}/{len(scenes)} {scene}', flush=True)
            continue
        start = time.perf_counter()
        if model is None:
            t0 = time.perf_counter()
            model, _, prep = open_clip.create_model_and_transforms(
                'ViT-H-14', pretrained=str(ROOT / 'models/open_clip_pytorch_model.bin'))
            model.cuda().eval()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                text = model.encode_text(open_clip.tokenize(NAMES).cuda()).float()
                text /= text.norm(dim=-1, keepdim=True)
            torch.cuda.synchronize()
            startup = time.perf_counter() - t0
        boxes, scores = merged[scene]
        with np.load(SEM / 'semantic_cache' / f'{scene}.npz') as old:
            prefix_hash = str(old['geometry_hash'])
            labels = old['labels'].copy()
        n = int(labels.shape[0])
        assert n == len(read_prediction(BASELINE / f'{scene}_boxes.pkl')[1])
        prefix = np.asarray(boxes[:n], np.float64).reshape(-1, 8, 3)
        assert hashlib.sha256(prefix.tobytes()).hexdigest() == prefix_hash, \
            f'prefix geometry drift: {scene}'
        birth_boxes = np.asarray(boxes[n:], np.float64).reshape(-1, 8, 3)
        merged_hash = hashlib.sha256(np.asarray(boxes, np.float64)
                                     .tobytes()).hexdigest()
        paths, poses, intrinsic, width, height = keyframes(scene)
        indices, rectangles = best_view(birth_boxes, poses, intrinsic,
                                        width, height)
        birth_features = np.zeros((len(birth_boxes), text.shape[1]), np.float32)
        birth_labels = np.full(len(birth_boxes), -1, int)
        view_ids = np.full(len(birth_boxes), -1, int)
        valid = np.flatnonzero(indices >= 0)
        images = {}
        gpu_seconds = 0.
        batches = []
        crop_hashes = []
        for begin in range(0, len(valid), batch_size):
            ids = valid[begin:begin + batch_size]
            inputs = []
            for bi in ids:
                path = paths[indices[bi]]
                if path not in images:
                    with Image.open(path) as im:
                        images[path] = im.convert('RGB')
                x1, y1, x2, y2 = rectangles[bi]
                dx, dy = .15 * (x2 - x1), .15 * (y2 - y1)
                crop = images[path].crop((int(max(0, x1 - dx)),
                                          int(max(0, y1 - dy)),
                                          int(min(width, x2 + dx)),
                                          int(min(height, y2 + dy))))
                inputs.append(prep(crop))
                view_ids[bi] = int(path.stem)
                crop_hashes.append(hashlib.sha256(crop.tobytes()).hexdigest())
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                feat = model.encode_image(torch.stack(inputs).cuda()).float()
                feat /= feat.norm(dim=-1, keepdim=True)
                cls = (feat @ text.T).argmax(-1).cpu().numpy()
                birth_features[ids] = feat.cpu().numpy()
                birth_labels[ids] = cls
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            gpu_seconds += dt
            batches.append(dt)
        image_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in images}
        np.savez_compressed(dest, features=birth_features,
                            labels=birth_labels, view_ids=view_ids,
                            rectangles=rectangles,
                            merged_geometry_hash=merged_hash,
                            crop_hashes=np.asarray(crop_hashes))
        write(cache / f'{scene}.timing.json',
              {'scene': scene, 'births': len(birth_boxes),
               'classified': len(valid), 'abstained': int((birth_labels < 0).sum()),
               'clip_seconds': gpu_seconds, 'batch_seconds': batches,
               'clip_peak_allocated_bytes': int(torch.cuda.max_memory_allocated()),
               'image_sha256': image_hashes, 'startup_seconds': startup})
        print(f'CLASSIFIED {si + 1}/{len(scenes)} {scene} '
              f'{len(valid)}/{len(birth_boxes)} CLIP={gpu_seconds:.2f}s',
              flush=True)


def evaluate(scenes, merged, gts_audit, aligns, transform):
    all_gt, gt_classes = {}, {}
    predictions = {'M1_M2': {}, 'TIER2': {}}
    classes = {'M1_M2': {}, 'TIER2': {}}
    audit_predictions = {}
    # audit binding keeps the seedless protocol order (ledger convention);
    # the semantic table uses the sorted order of the published table.
    # 23,150 births share the flat 0.05 score, so pooled AP depends on the
    # tie order: the two orders differ by ~0.13 AP15 and both are disclosed.
    order = sorted(scenes)
    for scene in order:
        all_gt[scene], gt_classes[scene] = ground_truth(scene)
        align = alignment(scene)
        boxes, scores = merged[scene]
        with np.load(SEM / 'semantic_cache' / f'{scene}.npz') as old:
            prefix_labels = old['labels'].copy()
        with np.load(OUT / 'semantic_cache' / f'{scene}.npz') as new:
            birth_labels = new['labels'].copy()
        n = len(prefix_labels)
        assert len(scores) == n + len(birth_labels)
        predictions['M1_M2'][scene] = (boxes[:n] @ align[:3, :3].T
                                       + align[:3, 3], scores[:n])
        predictions['TIER2'][scene] = (boxes @ align[:3, :3].T + align[:3, 3],
                                       scores)
        classes['M1_M2'][scene] = prefix_labels
        classes['TIER2'][scene] = np.concatenate([prefix_labels, birth_labels])
    for scene in scenes:
        boxes, scores = merged[scene]
        audit_predictions[scene] = (transform(boxes, aligns[scene]), scores)
    audit = {str(t): class_agnostic_ap(audit_predictions, gts_audit, t)
             for t in THRESHOLDS}
    if len(scenes) == 100:
        for t in THRESHOLDS:
            assert abs(audit[str(t)]['ap'] - EXPECTED[str(t)]) <= 5e-4, audit
    result = {'arms': {}, 'audit_path_class_agnostic': audit,
              'ledger_binding': 'reproduces reports/final_ledger_20260915 '
                                'tier2_m300 row (audit evaluator path)'}
    for arm in ('M1_M2', 'TIER2'):
        per_class = {}
        for cid, name in enumerate(NAMES):
            gts = {s: all_gt[s][gt_classes[s] == cid] for s in order}
            preds = {s: (b[classes[arm][s] == cid], c[classes[arm][s] == cid])
                     for s, (b, c) in predictions[arm].items()}
            count = sum(len(g) for g in gts.values())
            per_class[name] = {'gt': count,
                               'predictions': sum(len(b) for b, c in preds.values())}
            for thr in THRESHOLDS:
                per_class[name][str(thr)] = \
                    class_agnostic_ap(preds, gts, thr) if count else None
        active = [v for v in per_class.values() if v['gt']]
        result['arms'][arm] = {
            'per_class': per_class,
            'mAP': {str(t): float(np.mean([v[str(t)]['ap'] for v in active]))
                    for t in THRESHOLDS},
            'class_agnostic': {str(t): class_agnostic_ap(predictions[arm],
                                                         all_gt, t)
                               for t in THRESHOLDS},
            'boxes': sum(len(b) for b, c in predictions[arm].values()),
            'abstained': sum(int((x < 0).sum()) for x in classes[arm].values())}
        print(arm, result['arms'][arm]['mAP'], flush=True)
    for t in THRESHOLDS:
        published = SEM_PUBLISHED['mAP'][str(t)]
        dev = abs(result['arms']['M1_M2']['mAP'][str(t)] - published)
        print(f'M1_M2 mAP{t}: recomputed '
              f"{result['arms']['M1_M2']['mAP'][str(t)]:.6f} vs published "
              f'{published} (dev {dev:.2e})', flush=True)
        if len(scenes) == 100:
            assert dev <= 5e-3, (t, dev)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    protocol, merged, births_total = build_tier2_raw('scannet', 'score_m300')
    scenes = protocol['scenes'][:args.limit]
    full = args.limit == 100
    hashes = {}
    for scene in scenes:
        p = BASELINE / f'{scene}_boxes.pkl'
        hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    protocol_out = {'arms': {'M1_M2': str(BASELINE),
                             'TIER2': 'persistent + tier-2 births (score_m300, '
                                      'flat 0.05, score-ranked cluster exemplars)'},
                    'scenes': protocol['scenes'], 'births_total': births_total,
                    'class_names': NAMES, 'nyu40ids': NYU,
                    'reuse': 'prefix CLIP features reused from '
                             'reports/scannet_semantic_20260915 after geometry-hash check; '
                             'only births encoded',
                    'protocol': 'identical to scannet_semantic_20260915: gap25 '
                                'largest-projection auto crop, 15% margin, frozen '
                                'ViT-H-14 fp16, argmax over 18 names, detection-score '
                                'ranking unchanged, births appended at 0.05',
                    'scope': 'terminal common semantic readout; development scenes, '
                             'not novel/online proof'}
    write(OUT / 'protocol.json', protocol_out)
    if not args.evaluate_only:
        classify(scenes, merged, args.batch_size)
    gts, aligns, transform = build_scannet_eval(protocol)
    result = evaluate(scenes, merged, gts, aligns, transform)
    write(OUT / ('smoke_results.json' if not full else 'semantic_table.json'),
          result)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
               for p, h in hashes.items())
    write(OUT / 'integrity.json', {'input_predictions_unchanged': True,
                                   'audit_path_bound': True})
    timings = []
    for scene in scenes:
        t = json.loads((OUT / 'semantic_cache' / f'{scene}.timing.json')
                       .read_text())
        timings.append(t)
    write(OUT / 'timing_summary.json', {
        'births_encoded': sum(t['births'] for t in timings),
        'classified': sum(t['classified'] for t in timings),
        'abstained': sum(t['abstained'] for t in timings),
        'clip_seconds': sum(t['clip_seconds'] for t in timings),
        'per_birth_ms': 1000.0 * sum(t['clip_seconds'] for t in timings) /
                        max(1, sum(t['classified'] for t in timings))})


if __name__ == '__main__':
    main()
