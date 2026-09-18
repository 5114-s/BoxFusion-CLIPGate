"""Verify frozen person proposals against calibrated sensor depth; no 3D GT."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/original_motion_clip_20260914'
protocol = json.loads((OUT / 'sparse_screen_protocol.json').read_text())
windows = {'scene0423_01': list(range(546, 619, 8)),
           'scene0207_02': [2258, 2283, 2308, 2333, 2358]}
(OUT / 'depth_protocol.json').write_text(json.dumps({
    'frames': windows, 'detector': protocol['frozen_settings'],
    'masks': 'retina_masks=True for original RGB pixel coordinates; erode 3x3 once for interior diagnostic',
    'depth': 'uint16 / 1000 metres; valid 0.2 < z < 12; project depth pixel through Kdepth and Kcolor (both extrinsics verified identity)',
    'limits': 'Detector masks and depth only check observable surface support; never full-body GT or AP. No shape fitting or identity tuning.'}, indent=2)+'\n')
ckpt = Path(protocol['checkpoint'])
assert hashlib.sha256(ckpt.read_bytes()).hexdigest() == protocol['checkpoint_sha256']
model = YOLOE(str(ckpt))
rows = []
for scene, frames in windows.items():
    root = ROOT / 'data/scannet_val_rgbfix' / scene / 'frames'
    kd = np.loadtxt(root / 'intrinsic/intrinsic_depth.txt')[:3, :3]
    kc = np.loadtxt(root / 'intrinsic/intrinsic_color.txt')[:3, :3]
    for kind in ['color', 'depth']:
        assert np.allclose(np.loadtxt(root / f'intrinsic/extrinsic_{kind}.txt'), np.eye(4))
    for frame in frames:
        path = root / f'color/{frame}.jpg'
        rgb = cv2.imread(str(path))
        depth = cv2.imread(str(root / f'depth/{frame}.png'), -1).astype(float) / 1000
        pose = np.loadtxt(root / f'pose/{frame}.txt')
        yy, xx = np.indices(depth.shape)
        rays = np.column_stack([xx.ravel(), yy.ravel(), np.ones(depth.size)]) @ np.linalg.inv(kd).T
        uv = rays @ kc.T
        uv = np.rint(uv[:, :2] / uv[:, 2:]).astype(int)
        in_image = (uv[:, 0] >= 0) & (uv[:, 0] < rgb.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < rgb.shape[0])
        result = model.predict(str(path), device='cuda:0', verbose=False,
                               retina_masks=True, **protocol['frozen_settings'])[0]
        people = []
        for k, box in enumerate(result.boxes):
            if result.names[int(box.cls.item())].lower() != 'person':
                continue
            mask = result.masks.data[k].cpu().numpy().astype(np.uint8)
            assert mask.shape == rgb.shape[:2]
            diagnostics = {}
            for label, m in [('mask', mask), ('interior', cv2.erode(mask, np.ones((3, 3), np.uint8)))]:
                selected = np.zeros(depth.size, bool)
                selected[in_image] = m[uv[in_image, 1], uv[in_image, 0]] > 0
                valid = selected & (depth.ravel() > .2) & (depth.ravel() < 12)
                z = depth.ravel()[valid]
                nonzero = depth.ravel()[selected & (depth.ravel() > 0)]
                diagnostics[label] = {'sampled_pixels': int(selected.sum()), 'valid_points': int(valid.sum()),
                                      'valid_ratio': float(valid.sum()/max(selected.sum(), 1)),
                                      'nonzero_points_without_range_filter': int(len(nonzero)),
                                      'nonzero_depth_minmax_m': [float(nonzero.min()), float(nonzero.max())] if len(nonzero) else None,
                                      'depth_m_p10_p50_p90': np.percentile(z, [10, 50, 90]).tolist() if len(z) else None}
            people.append({'score': float(box.conf.item()), 'box2d_original': box.xyxy[0].cpu().tolist(), **diagnostics})
            cv2.imwrite(str(OUT / f'{scene}_{frame}_person{k}_mask.png'), mask*255)
        rows.append({'scene': scene, 'frame': frame, 'person_detections': people, 'pose_finite': bool(np.isfinite(pose).all()),
                     'rgb_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'depth_sha256': hashlib.sha256((root / f'depth/{frame}.png').read_bytes()).hexdigest()})
        print(scene, frame, [(round(p['score'], 2), p['interior']['valid_points'], round(p['interior']['valid_ratio'], 3)) for p in people], flush=True)
(OUT / 'depth_verification.json').write_text(json.dumps(rows, indent=2)+'\n')
