#!/usr/bin/env python3
"""Metadata-only admission audit; no detector, training, or dataset download."""
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/original_dynamic_admission_20260914'
CA = Path('/extra/ZhaoX/boxfusion_ca1m')
SC = ROOT / 'data/scannet_val_rgbfix'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def frame_ids(path):
    with os.scandir(path) as entries:
        return {Path(p.name).stem for p in entries if Path(p.name).stem.isdigit() and p.is_file()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ca, sc, raw = [], [], []
    for p in sorted(CA.iterdir()):
        if not (p.is_dir() and p.name.isdigit()):
            continue
        rgb, depth = frame_ids(p / 'rgb'), frame_ids(p / 'depth')
        poses = np.load(p / 'all_poses.npy', mmap_mode='r', allow_pickle=False)
        gt = np.load(p / 'after_filter_boxes.npy', allow_pickle=False)
        files = sorted((p / 'instances').glob('*.json'), key=lambda x: int(x.stem))
        sample = [files[i] for i in sorted(set(np.linspace(0, len(files)-1, 5, dtype=int)))] if files else []
        keys = Counter()
        samples = []
        for f in sample:
            rows = read(f)
            for row in rows:
                keys.update(row.keys())
            samples.append({'frame': int(f.stem), 'instances': len(rows), 'sha256': digest(f)})
        ca.append({'scene': p.name, 'rgb': len(rgb), 'depth': len(depth),
                   'rgb_depth_ids_equal': rgb == depth, 'pose_shape': list(poses.shape),
                   'pose_count_matches_rgb': len(poses) == len(rgb),
                   'gt_shape': list(gt.shape), 'gt_finite': bool(np.isfinite(gt).all()),
                   'frame_label_files': len(files), 'sampled_fields': dict(keys),
                   'samples': samples, 'world_label_sha256': digest(p / 'instances.json')
                   if (p / 'instances.json').exists() else None})
    gtroot = ROOT / 'evaluation/data_util/scannet_train_detection_data'
    classes = set()
    for p in sorted(SC.glob('scene*')):
        q = p / 'frames'
        color, depth, pose = [frame_ids(q / x) for x in ['color', 'depth', 'pose']]
        f = gtroot / (p.name + '_bbox.npy')
        a = np.load(f, allow_pickle=False) if f.exists() else None
        if a is not None and len(a):
            classes.update(map(int, a[:, -1]))
        sc.append({'scene': p.name, 'rgb': len(color), 'depth': len(depth), 'poses': len(pose),
                   'rgb_depth_pose_ids_equal': color == depth == pose,
                   'gt_path': str(f), 'gt_shape': list(a.shape) if a is not None else None,
                   'gt_finite': bool(np.isfinite(a).all()) if a is not None else False,
                   'frame_directory_entries': sorted(x.name for x in q.iterdir()),
                   'gt_sha256': digest(f) if f.exists() else None})
    for scene in ['42898811', '47334256']:
        p = Path('/extra/ZhaoX/ca1m_raw_verify') / scene
        if not (p / 'world.gt/instances.json').exists():
            p = p / scene
        world = read(p / 'world.gt/instances.json')
        world_ids = {x['id'] for x in world}
        files = sorted(p.glob('*.wide'))
        samples = []
        for i in sorted(set(np.linspace(0, len(files)-1, 5, dtype=int))):
            f = files[i] / 'instances.json'
            rows = read(f)
            samples.append({'frame': files[i].stem, 'instances': len(rows),
                            'ids_not_in_world': sum(x['id'] not in world_ids for x in rows),
                            'fields': sorted({k for x in rows for k in x}), 'sha256': digest(f)})
        a = files[0]
        processed = cv2.imread(str(CA / scene / 'depth/0.png'), -1)
        depths = {}
        for name, f in [('faro_gt', a.with_suffix('.gt') / 'depth.png'),
                        ('sensor_wide', a / 'depth.png')]:
            if not f.exists():
                depths[name] = {'available': False}
                continue
            z = cv2.imread(str(f), -1)
            depths[name] = {'available': True, 'path': str(f), 'shape': list(z.shape),
                            'processed_frame0_equal_rotations': [k for k in range(4)
                                if processed.shape == np.rot90(z, k).shape
                                and np.array_equal(processed, np.rot90(z, k))]}
        raw.append({'scene': scene, 'root': str(p), 'world_instances': len(world),
                    'world_categories': dict(Counter(x['category'] for x in world)),
                    'raw_frame_directories': len(files), 'samples': samples,
                    'processed_frame0_shape': list(processed.shape), 'depth_probe': depths})
    frozen_paths = [
        'tools/run_bonn_fullbox_loop.py', 'tools/run_bonn_surface_track_probe.py',
        'reports/bonn_fullbox_loop_20260914/protocol.json',
        'reports/bonn_surface_track_625_675_20260914/protocol.json',
        'reports/bonn_fullbox_loop_20260914/flow_latest_trace.json',
        'reports/bonn_detect_20260914/yoloe_candidates.json',
        'evaluation/data_util/model_util_scannet.py', 'data_process/process2slam.py',
        'data_process/process2slam_gtbox.py', 'tools/audit_original_dynamic_admission.py']
    result = {
        'scope': 'Metadata inventory all local original evaluation scenes; 5 annotation samples per CA-1M scene; 2 raw CA-1M captures. No RGB image review or detector execution. Counts do not establish absence of people in RGB.',
        'summary': {'ca1m_scenes': len(ca), 'ca1m_with_frame_labels': sum(x['frame_label_files'] > 0 for x in ca),
                    'ca1m_with_world_json': sum(x['world_label_sha256'] is not None for x in ca),
                    'ca1m_sampled_frames': sum(len(x['samples']) for x in ca),
                    'ca1m_missing_rgb_depth_alignment': sum(not x['rgb_depth_ids_equal'] for x in ca),
                    'ca1m_pose_count_mismatch': sum(not x['pose_count_matches_rgb'] for x in ca),
                    'ca1m_nonfinite_gt_scenes': sum(not x['gt_finite'] for x in ca),
                    'scannet_scenes': len(sc), 'scannet_missing_or_invalid_gt': sum(not x['gt_finite'] for x in sc),
                    'scannet_rgb_depth_pose_mismatch': sum(not x['rgb_depth_pose_ids_equal'] for x in sc),
                    'scannet_observed_nyu40_ids': sorted(classes), 'scannet_person_nyu40_31_in_eval': 31 in classes},
        'decision': {'original_scene_detection_AP': 'available; this audit does not rerun AP',
                     'moving_person_3D_benchmark': 'not admitted: inspected annotations/documented generation do not supply verified moving-person geometry trajectories',
                     'CA1M_per_frame_boxes': 'present, generated from scene annotation with view/frustum/occlusion processing; do not interpret changing camera-frame centers or view-dependent box extents as physical motion',
                     'detector_admission': 'not run because moving-object annotation gate is unfulfilled',
                     'geometry_three_arm_test': 'not run; cannot report dynamic AP from scene GT',
                     'frozen_baseline': 'existing flow_latest RGB-D geometry updates and support-loss suspension; new identity-gating variants not promoted'},
        'ca1m': ca, 'scannet': sc, 'raw_ca1m': raw,
        'frozen_sha256': {s: digest(ROOT / s) for s in frozen_paths},
        'sources': ['https://github.com/apple-aiml-research/ml-cubifyanything#data-format',
                    'https://arxiv.org/html/2412.04458v1#S3.SS2',
                    'https://github.com/ScanNet/ScanNet#data-formats',
                    'https://arxiv.org/html/2506.15610v3#S5']}
    (OUT / 'audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result['summary'], ensure_ascii=False))
    print('raw_depth_checks:', json.dumps([{'scene': x['scene'], 'depth': x['depth_probe']} for x in raw]))
    print('saved', OUT / 'audit.json')


if __name__ == '__main__':
    main()
