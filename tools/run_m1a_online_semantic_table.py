#!/usr/bin/env python3
"""Semantic readout of the STRICT-ONLINE final arm (ScanNet, M300).

Re-binds the semantic table to the current final configuration
(reports/m1a_strict_online_scannet_20260916): the M1-P+M2 prefix (2,948 rows)
reuses the frozen CLIP features/labels from
reports/scannet_semantic_20260915 after a per-scene geometry-hash check, and
only the 23,177 strict-online M1-A births (unique scores in [0.040001,
0.049999]) go through the identical automatic crop + frozen CLIP protocol.

Anchors: prefix semantic mAP must reproduce the published table
(23.9586/21.1467/14.2563) and the online arm's class-agnostic AP must
reproduce the semantic-path values (44.0597/39.7524/20.2502).  Development-set
semantic detection under a fixed 18-class table; not a novel-class or
per-frame online-semantics claim.
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
from audit_m1m2_remaining_children import read_prediction
from run_scannet_semantic_table import (NAMES, FRAMES, ground_truth,
                                        alignment)
from run_tier2_semantic_table import keyframes, Image_open

BASELINE = ROOT / 'results/scannet_m2nl_m5_dual_full100/persistent'
ONLINE = ROOT / 'reports/m1a_strict_online_scannet_20260916/predictions'
SEM = ROOT / 'reports/scannet_semantic_20260915'
OUT = ROOT / 'reports/m1a_online_semantic_20260916'
THRESHOLDS = (0.15, 0.25, 0.5)
PREFIX_SEM_EXPECT = (23.9586, 21.1467, 14.2563)
ONLINE_CLSAGN_EXPECT = (44.05968023213844, 39.75235941348523,
                        20.25023681558712)


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


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
                'ViT-H-14',
                pretrained=str(ROOT / 'models/open_clip_pytorch_model.bin'))
            model.cuda().eval()
            with torch.inference_mode(), torch.autocast('cuda',
                                                        dtype=torch.float16):
                text = model.encode_text(
                    open_clip.tokenize(NAMES).cuda()).float()
                text /= text.norm(dim=-1, keepdim=True)
            torch.cuda.synchronize()
            startup = time.perf_counter() - t0
        boxes, scores = merged[scene]
        with np.load(SEM / 'semantic_cache' / f'{scene}.npz') as old:
            prefix_hash = str(old['geometry_hash'])
            prefix_labels = old['labels'].copy()
        n = int(prefix_labels.shape[0])
        base_boxes, _ = read_prediction(BASELINE / f'{scene}_boxes.pkl')
        assert n == len(base_boxes)
        prefix = np.asarray(boxes[:n], np.float64).reshape(-1, 8, 3)
        assert hashlib.sha256(prefix.tobytes()).hexdigest() == prefix_hash, \
            f'prefix geometry drift: {scene}'
        births = np.asarray(boxes[n:], np.float64).reshape(-1, 8, 3)
        paths, poses, intrinsic, width, height = keyframes(scene)
        indices, rectangles = best_view(births, poses, intrinsic,
                                        width, height)
        birth_features = np.zeros((len(births), text.shape[1]), np.float32)
        birth_labels = np.full(len(births), -1, int)
        view_ids = np.full(len(births), -1, int)
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
            with torch.inference_mode(), torch.autocast(
                    'cuda', dtype=torch.float16):
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
                            crop_hashes=np.asarray(crop_hashes))
        write(cache / f'{scene}.timing.json',
              {'scene': scene, 'births': len(births),
               'classified': len(valid),
               'abstained': int((birth_labels < 0).sum()),
               'keyframes': len(paths),
               'semantic_stage_seconds': time.perf_counter() - start,
               'clip_seconds': gpu_seconds, 'batch_seconds': batches,
               'clip_peak_allocated_bytes':
                   int(torch.cuda.max_memory_allocated()),
               'image_sha256': image_hashes,
               'startup_seconds': startup})
        print(f'CLASSIFIED {si + 1}/{len(scenes)} {scene} '
              f'{len(valid)}/{len(births)} CLIP={gpu_seconds:.2f}s',
              flush=True)


def evaluate(scenes, merged):
    all_gt, gt_classes = {}, {}
    predictions = {'M1_M2': {}, 'ONLINE': {}}
    classes = {'M1_M2': {}, 'ONLINE': {}}
    for scene in scenes:
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
        predictions['ONLINE'][scene] = (boxes @ align[:3, :3].T
                                        + align[:3, 3], scores)
        classes['M1_M2'][scene] = prefix_labels
        classes['ONLINE'][scene] = np.concatenate([prefix_labels,
                                                   birth_labels])
    result = {'arms': {}}
    for arm in ('M1_M2', 'ONLINE'):
        per_class = {}
        for cid, name in enumerate(NAMES):
            gts = {s: all_gt[s][gt_classes[s] == cid] for s in scenes}
            preds = {s: (b[classes[arm][s] == cid], c[classes[arm][s] == cid])
                     for s, (b, c) in predictions[arm].items()}
            count = sum(len(g) for g in gts.values())
            per_class[name] = {'gt': count,
                               'predictions': sum(len(b) for b, c in
                                                  preds.values())}
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
            'abstained': sum(int((x < 0).sum())
                             for x in classes[arm].values())}
        print(arm, result['arms'][arm]['mAP'], flush=True)
    if len(scenes) == 100:
        for t, expected in zip(THRESHOLDS, PREFIX_SEM_EXPECT):
            assert abs(result['arms']['M1_M2']['mAP'][str(t)]
                       - expected) <= 5e-4, (t, expected)
        for t, expected in zip(THRESHOLDS, ONLINE_CLSAGN_EXPECT):
            assert abs(result['arms']['ONLINE']['class_agnostic'][str(t)]
                       ['ap'] - expected) <= 5e-4, (t, expected)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    manifest = json.loads((ONLINE.parent / 'manifest.json').read_text())
    scenes = sorted(manifest['scenes'])[:args.limit]
    merged = {}
    births_total = 0
    hashes = {}
    for scene in manifest['scenes']:
        base_boxes, base_scores = read_prediction(
            BASELINE / f'{scene}_boxes.pkl')
        ob, os_ = read_prediction(ONLINE / f'{scene}_boxes.pkl')
        n = len(base_boxes)
        assert np.array_equal(ob[:n], base_boxes), scene
        assert np.array_equal(os_[:n], base_scores), scene
        merged[scene] = (ob, os_)
        births_total += len(ob) - n
        hashes[str(ONLINE / f'{scene}_boxes.pkl')] = hashlib.sha256(
            (ONLINE / f'{scene}_boxes.pkl').read_bytes()).hexdigest()
    write(OUT / 'protocol.json', {
        'arms': {'M1_M2': str(BASELINE),
                 'ONLINE': 'strict-online M1-A output (prefix + births, '
                           'unique scores in [0.040001, 0.049999])'},
        'scenes': sorted(manifest['scenes']),
        'online_births_total': births_total,
        'class_names': NAMES,
        'reuse': 'prefix CLIP features reused from '
                 'reports/scannet_semantic_20260915 after geometry-hash check; '
                 'only online births encoded',
        'protocol': 'identical to scannet_semantic_20260915: gap25 '
                    'largest-projection auto crop, 15% margin, frozen '
                    'ViT-H-14 fp16, argmax over 18 names, detection-score '
                    'ranking unchanged',
        'anchors': {'prefix_semantic_mAP': list(PREFIX_SEM_EXPECT),
                    'online_class_agnostic_semantic_path':
                        list(ONLINE_CLSAGN_EXPECT)},
        'scope': 'terminal common semantic readout; development scenes, '
                 'not novel-class or per-frame online-semantics proof'})
    if not args.evaluate_only:
        classify(scenes, merged, args.batch_size)
    result = evaluate(scenes, merged)
    write(OUT / ('smoke_results.json' if len(scenes) < 100
                 else 'semantic_table.json'), result)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
               for p, h in hashes.items())
    write(OUT / 'integrity.json', {'online_predictions_unchanged': True,
                                   'prefix_reused_after_hash_check': True})
    if len(scenes) == 100:
        timings = [json.loads((OUT / 'semantic_cache' / f'{s}.timing.json')
                              .read_text()) for s in scenes]
        write(OUT / 'timing_summary.json', {
            'births_encoded': sum(t['births'] for t in timings),
            'classified': sum(t['classified'] for t in timings),
            'abstained': sum(t['abstained'] for t in timings),
            'clip_seconds': sum(t['clip_seconds'] for t in timings),
            'per_birth_ms': 1000.0 * sum(t['clip_seconds'] for t in timings)
                            / max(1, sum(t['classified'] for t in timings))})


if __name__ == '__main__':
    main()
