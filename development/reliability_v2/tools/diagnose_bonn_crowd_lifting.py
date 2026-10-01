#!/usr/bin/env python3
"""CPU-only diagnosis of three frozen crowd lifting failures; no model changes."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from PIL import Image
from run_bonn_crowd_ablation import overlap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/bonn_crowd_lifting_diag_20260914'
DATA = ROOT/'data_bonn/scene0002_01/frames'
PREVIOUS = ROOT/'reports/bonn_crowd_native_dynamic_20260914'


def corners(lo, hi):
    return np.asarray(list(itertools.product(*zip(lo, hi))))


def image_box(camera_points, K):
    assert np.all(camera_points[:, 2] > .05)
    homogeneous = camera_points @ K.T
    uv = homogeneous[:, :2]/homogeneous[:, 2:]
    return np.clip(np.r_[uv.min(0), uv.max(0)], 0, [640,480,640,480])


def main():
    source = ROOT/'reports/bonn_detect_20260914/yoloe_candidates.json'
    cache = json.loads(source.read_text())
    references = json.loads((PREVIOUS/'annotations.json').read_text())['frames']
    baseline = json.loads((PREVIOUS/'results.json').read_text())
    K = np.loadtxt(DATA/'intrinsic/intrinsic_depth.txt')[:3,:3]
    inverse_K = np.linalg.inv(K)
    rows = []
    paths = [Path(__file__), source, PREVIOUS/'annotations.json', PREVIOUS/'results.json',
             DATA/'intrinsic/intrinsic_depth.txt', ROOT/'tools/run_yoloe_bonn_candidates.py',
             ROOT/'tools/run_bonn_native_dynamic_ablation.py']
    for frame in (500,525,725):
        detection = next(r for r in cache if r['scene']=='scene0002_01' and r['frame']==frame)['detections'][1]
        assert detection['label']=='person'
        box = np.asarray(detection['box2d'])
        lo, hi = np.asarray(detection['world_aabb_lo']), np.asarray(detection['world_aabb_hi'])
        pose_path, depth_path = DATA/f'pose/{frame}.txt', DATA/f'depth/{frame}.png'
        paths += [pose_path, depth_path]
        pose = np.loadtxt(pose_path); R,t = pose[:3,:3], pose[:3,3]
        depth = np.asarray(Image.open(depth_path), dtype=np.float64)/5000.
        assert depth.shape==(480,640)
        yy,xx = np.nonzero((depth>.05)&(depth<12))
        z = depth[yy,xx]
        camera_points = np.c_[xx,yy,np.ones(len(xx))]@inverse_K.T*z[:,None]
        world_points = camera_points@R.T+t
        back = (world_points-t)@R
        reprojected = back@K.T
        uv_back = reprojected[:,:2]/reprojected[:,2:]
        roundtrip_px = float(np.max(np.abs(uv_back-np.c_[xx,yy])))
        assert roundtrip_px < 1e-4
        cached_camera_corners = (corners(lo,hi)-t)@R
        cached_projection = image_box(cached_camera_corners,K)
        saved = next(r for r in baseline['per_frame']['bypass'] if r['frame']==frame)
        saved = next(p for p in saved['projected_predictions'] if p['source']==f'{frame}:1')
        np.testing.assert_allclose(cached_projection,saved['box'],atol=1e-8)
        inside_roi = (xx>=box[0])&(xx<=box[2])&(yy>=box[1])&(yy<=box[3])
        supported = np.all((world_points>=lo)&(world_points<=hi),axis=1)
        # These are observed depth samples inside the cached volume, not the
        # original segmentation mask, which the exporter did not retain.
        supported_uv = np.c_[xx[supported],yy[supported]]
        roi_depth = z[inside_roi]
        ref = next(g['box'] for g in references[str(frame)]['people'] if g['identity']=='P1')
        x1,y1,x2,y2 = box
        plane_uv = np.asarray([[x,y] for x in (x1,x2) for y in (y1,y2)])
        plane_camera = np.c_[plane_uv,np.ones(4)]@inverse_K.T*detection['depth_median_m']
        plane_world = plane_camera@R.T+t
        plane_world_aabb = corners(plane_world.min(0),plane_world.max(0))
        plane_aabb_projection = image_box((plane_world_aabb-t)@R,K)
        preserved_plane_projection = image_box((plane_world-t)@R,K)
        # Same noiseless points and pose; only taking a world-axis bounding box
        # changes. The oriented plane is a diagnostic control, not a human 3D box.
        np.testing.assert_allclose(preserved_plane_projection,box,atol=1e-5)
        assert overlap(plane_aabb_projection,ref)<.5
        rows.append({
            'frame':frame,'source':f'{frame}:1','world_extent_m':(hi-lo).tolist(),
            'rotation':R.tolist(),'rotation_orthogonality_max_error':float(np.max(abs(R.T@R-np.eye(3)))),
            'depth_roundtrip_max_pixel_error':roundtrip_px,
            'cached_corner_depth_min_max_m':[float(cached_camera_corners[:,2].min()),float(cached_camera_corners[:,2].max())],
            'cached_mask_depth_median_m':detection['depth_median_m'],
            'cached_mask_valid_depth_ratio':detection['valid_depth_ratio'],
            'raw_box':box.tolist(),'cached_aabb_projection':cached_projection.tolist(),
            'raw_reference_iou':overlap(box,ref),'aabb_reference_iou':overlap(cached_projection,ref),
            'projected_width_over_raw_width':float((cached_projection[2]-cached_projection[0])/(box[2]-box[0])),
            'roi_valid_depth_quantiles_m':np.quantile(roi_depth,[0,.02,.1,.5,.9,.98,1]).tolist(),
            'roi_depth_note':'Unsegmented rectangle includes background; its far tail is not proof of original mask contamination.',
            'raw_depth_samples_inside_cached_volume':int(supported.sum()),
            'supported_samples_outside_original_2d_box':int((supported&~inside_roi).sum()),
            'supported_samples_outside_fraction':float((supported&~inside_roi).sum()/max(supported.sum(),1)),
            'supported_pixel_bounds':np.r_[supported_uv.min(0),supported_uv.max(0)].tolist() if len(supported_uv) else None,
            'noiseless_plane_control':{'world_aabb_projection':plane_aabb_projection.tolist(),
                                      'world_aabb_reference_iou':overlap(plane_aabb_projection,ref),
                                      'orientation_preserved_reference_iou':overlap(preserved_plane_projection,ref)}})
    result={'frames':rows,'detector_forward_runs':0,'training_runs':0,'gpu_runs':0,
            'original_mask_available':False,'original_masked_points_available':False,
            'pipeline_modified':False,'eleven_frame_rerun':False,
            'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'scope':'Three frozen failures; independent numerical projection check, actual depth support, and noiseless representation control. No GT used to create or change predictions.'}
    OUT.mkdir(exist_ok=True)
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for r in rows:
        print(json.dumps({k:r[k] for k in ['frame','raw_reference_iou','aabb_reference_iou','projected_width_over_raw_width','cached_corner_depth_min_max_m','depth_roundtrip_max_pixel_error','raw_depth_samples_inside_cached_volume','supported_samples_outside_fraction','supported_pixel_bounds','noiseless_plane_control']}))


if __name__=='__main__':
    main()
