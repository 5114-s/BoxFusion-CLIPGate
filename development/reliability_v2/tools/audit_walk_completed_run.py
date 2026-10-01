#!/usr/bin/env python3
"""Read-only input/coverage audit for the completed synthetic walk dev run."""
import hashlib
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_walk_bench import points_in_obb, obb_frame, backproject, edit_frame, project_corners
from tools.eval_walk_bench import read_alignment


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    data = ROOT/'data_dyn_walk'
    run = ROOT/'reports/walk_dev_20260913/native'
    manifest = json.loads((data/'manifest.json').read_text())
    assignment = json.loads((data/'assignment.json').read_text())
    labels = dict(zip([3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39],
                      ['cabinet','bed','chair','sofa','table','door','window','bookshelf',
                       'picture','counter','desk','curtain','refrigerator','showercurtain',
                       'toilet','sink','bathtub','garbagebin']))
    records = []
    inputs = {str(p): sha(p) for p in [data/'manifest.json', data/'assignment.json',
              Path(__file__), ROOT/'tools/build_walk_bench.py', ROOT/'tools/eval_walk_bench.py',
              ROOT/'config/scannet_walk_native.yaml']}
    for scene, entry in sorted(manifest.items()):
        directory = data/scene/'frames'
        source = ROOT/'upstream_clean/scannet_readme_frames'/scene/'frames'
        record_path = run/'logs'/f'{scene}.result.json'
        record = json.loads(record_path.read_text())
        assert record['returncode'] == 0 and record['artifacts']
        assert all(sha(path) == digest for path, digest in record['artifacts'].items())
        inputs[str(record_path)] = sha(record_path)
        inputs.update(record['artifacts'])
        gt_path = ROOT/'evaluation/data_util/scannet_train_detection_data'/f'{scene}_bbox.npy'
        gt = np.load(gt_path)
        index = assignment[scene]['gt_idx']
        align_path = Path('/extra/ZhaoX/scannet_data/scans')/scene/f'{scene}.txt'
        align = read_alignment(align_path)
        inputs[str(gt_path)], inputs[str(align_path)] = sha(gt_path), sha(align_path)
        vertical = float((align[:3, :3] @ np.asarray(entry['final_offset']))[2])
        T = entry['T_frame']
        end = T + entry['D_max']/entry['speed_m_per_frame']
        files = sorted((directory/'depth').glob('*.png'), key=lambda x: int(x.stem))
        after = [f for f in files if int(f.stem) >= T]
        moving_kf = [f for f in after if int(f.stem) % 25 == 0 and int(f.stem) < end]
        edited = [f for f in after if not f.is_symlink()]
        corners = np.array(entry['marker']).reshape(8, 3)
        _, axes, _ = obb_frame(corners)
        orthogonality_error = float(np.max(np.abs(axes @ axes.T - np.eye(3))))
        text = (run/'logs'/f'{scene}.log').read_text(errors='replace')
        cost = re.findall(r'Cost: ([\d.]+) s Average FPS: ([\d.]+)', text)
        r = dict(scene=scene, target_label=labels.get(int(gt[index,-1]), str(gt[index,-1])),
                 final_vertical_displacement_m=vertical, post_frames=len(after),
                 post_unchanged_frames=sum(f.is_symlink() for f in after),
                 motion_phase_sampled_frames=len(moving_kf),
                 motion_phase_edited_sampled_frames=sum(not f.is_symlink() for f in moving_kf),
                 total_edited_frames=len(edited), obb_axes_nonorthogonality=orthogonality_error,
                 loop_seconds=record['loop_seconds'], process_seconds=record['wall_seconds'],
                 logged_input_fps=float(cost[-1][1]), rendering_check=None)
        # Three concrete sensor checks, without rerunning any detector.
        if len(records) < 3 and edited:
            f = int(edited[0].stem)
            K = np.loadtxt(source/'intrinsic/intrinsic_depth.txt')[:3,:3]
            pose = np.loadtxt(source/'pose'/f'{f}.txt')
            dep = np.asarray(Image.open(source/'depth'/f'{f}.png'))
            img = cv2.resize(cv2.imread(str(source/'color'/f'{f}.jpg')), (640,480))
            delta = np.array(entry['direction'])*min(entry['speed_m_per_frame']*(f-T), entry['D_max'])
            rebuilt = edit_frame(img, dep, pose, K, corners, delta)
            actual = np.asarray(Image.open(edited[0]))
            rect = project_corners(corners, pose, K,640,480)
            x1,y1,x2,y2=rect
            us,vs=np.meshgrid(np.arange(x1,x2),np.arange(y1,y2))
            us,vs=us.ravel(),vs.ravel()
            z=dep[vs,us]/1000.
            valid=z>.05
            points=backproject(us[valid],vs[valid],z[valid],pose,K)
            buggy=points_in_obb(points,corners)
            # gt_world_corners uses product order; true edges are 1,2,4.
            edges=corners[[1,2,4]]-corners[0]
            lengths=np.linalg.norm(edges,axis=1)
            local=(points-corners.mean(0)) @ (edges/lengths[:,None]).T
            correct=(np.abs(local)<=lengths/2+1e-6).all(1)
            r['rendering_check']=dict(frame=f,depth_exact_fraction=float(np.mean(rebuilt[1]==actual)),
                                     admitted_points=int(buggy.sum()),
                                     admitted_outside_true_box=int((buggy & ~correct).sum()))
        records.append(r)
    summary = dict(completed_scenes=len(records), target_labels=sorted(set(r['target_label'] for r in records)),
                   nonhorizontal_moves=sum(abs(r['final_vertical_displacement_m'])>.05 for r in records),
                   post_frames=sum(r['post_frames'] for r in records),
                   post_unchanged_frames=sum(r['post_unchanged_frames'] for r in records),
                   motion_phase_sampled_frames=sum(r['motion_phase_sampled_frames'] for r in records),
                   motion_phase_edited_sampled_frames=sum(r['motion_phase_edited_sampled_frames'] for r in records),
                   scenes_without_edited_motion_sample=sum(r['motion_phase_edited_sampled_frames']==0 for r in records),
                   nonorthogonal_obb_masks=sum(r['obb_axes_nonorthogonality']>1e-5 for r in records),
                   total_loop_seconds=sum(r['loop_seconds'] for r in records),
                   total_process_seconds=sum(r['process_seconds'] for r in records))
    output = dict(summary=summary, scenes=records, input_sha256=inputs,
                  limitations=['Unchanged post frames can be legitimate occlusion/out-of-view; their count alone is not a corruption rate.',
                               'Input FPS printed by demo is normalized by frame gap; not detector keyframes per second.',
                               'Current generator sampled for three saved depth images; exact agreement checks actual artifacts.',
                               'No optimized arm or temporal 3D GT evaluation is present in this run directory.'])
    path = ROOT/'reports/walk_dev_20260913/input_audit.json'
    path.write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summary,ensure_ascii=False))
    print('sensor_checks',json.dumps([r['rendering_check'] for r in records if r['rendering_check']]))


if __name__=='__main__':
    main()
