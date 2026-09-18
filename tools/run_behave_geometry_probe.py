"""Small geometry gate on official fitted full-body references, not a scene AP run."""
import argparse
import contextlib
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'reports/behave_motion_probe_20260914'
DATA = ROOT / 'data_behave_probe'
SEQ = 'Date04_Sub05_chairwood'
CKPT = Path('/data/ZhaoX/OVM3D-Dett/boxfusion_b6_selective_boxer_dev/models/yoloe-11s-seg-pf.pt')


def read(name):
    return json.loads((OUT / name).read_text())


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def prepare():
    import trimesh
    cal = json.loads((DATA / 'calibs/intrinsics/1/calibration.json').read_text())['color']
    original_k = np.array([[cal['fx'], 0, cal['cx']], [0, cal['fy'], cal['cy']], [0, 0, 1.]])
    k = original_k.copy(); k[0] *= 640/cal['width']; k[1] *= 480/cal['height']
    maps = cv2.initUndistortRectifyMap(original_k, np.array(cal['opencv'][4:]), None, k, (640, 480), cv2.CV_32FC1)
    protocol = {'sequence': SEQ, 'camera': 1, 'times_s': list(range(3, 14)), 'cadence_hz': 1,
                'K': k.tolist(), 'window': 3, 'detector': {'conf': .25, 'iou': .70, 'imgsz': 640, 'max_det': 64, 'agnostic_nms': True},
                'target_rule': 'Highest-score person with >=8 valid masked depth pixels; no reference used. All candidates saved. Single-target geometry diagnostic, not full association or duplicate evaluation.',
                'geometry': 'Undistort/resize RGB to 640x480; nearest remap depth, mm to m once; predicted mask depths -> 2/98 percentile AABB in camera-1/world coordinates. Reference -> full SMPL-H fitted mesh AABB in same coordinates.',
                'controls': 'Latest; last-3 mean; last-3 actual PFO. Oracle controls translate each past center by current-reference-center minus past-reference-center; no GT size/orientation replacement. PFO virtual cameras receive the same translation.',
                'flow': 'Reuse frozen Bonn LK/robust RGB-D translation on this 1-Hz input; any lost support falls back to latest. This is not a dense-video validation.',
                'gate': 'On common evaluable last-3 windows, at least one oracle control must improve mean 3D IoU over latest by >=0.02, not worsen mean center error, and not lower recall@0.5. Otherwise stop current fusion design before raw-video acquisition.',
                'limits': 'One development sequence, fixed camera, bending/sitting motions as well as translation. Reference is a fitted body model, not perfect physical extent. GT only in oracle/evaluation; no GT masks in prediction. No static regression or end-to-end FPS claim.'}
    save('geometry_protocol.json', protocol)
    refs, hashes = {}, {}
    frames = DATA / 'rectified'
    frames.mkdir(exist_ok=True)
    for t in protocol['times_s']:
        p = DATA / SEQ / f't{t:04d}.000'
        rgb = cv2.imread(str(p/'k1.color.jpg'))
        dep = cv2.imread(str(p/'k1.depth.png'), -1)
        cv2.imwrite(str(frames/f'{t}.jpg'), cv2.remap(rgb, *maps, cv2.INTER_LINEAR))
        np.save(frames/f'{t}.npy', cv2.remap(dep, *maps, cv2.INTER_NEAREST).astype(np.float32)/1000.)
        verts = np.asarray(trimesh.load(p/'person/fit02/person_fit.ply', process=False).vertices)
        lo, hi = verts.min(0), verts.max(0)
        refs[str(t)] = {'center': ((lo+hi)/2).tolist(), 'size': (hi-lo).tolist(), 'lo': lo.tolist(), 'hi': hi.tolist()}
        for f in [p/'k1.color.jpg', p/'k1.depth.png', p/'person/fit02/person_fit.ply']:
            hashes[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    save('references.json', refs); save('geometry_input_hashes.json', hashes)
    print('prepared', len(refs), 'frames', flush=True)


def infer():
    from ultralytics import YOLOE
    from tools.run_bonn_surface_track_probe import advance, lift, sample_depth
    cfg = read('geometry_protocol.json'); k = np.array(cfg['K'])
    flow_cfg = json.loads((ROOT/'reports/bonn_surface_track_625_675_20260914/protocol.json').read_text())
    model = YOLOE(str(CKPT)); records = []; states = {}; previous = None
    cv2.setNumThreads(1); cv2.setRNGSeed(0)
    for t in cfg['times_s']:
        rgb = cv2.imread(str(DATA/'rectified'/f'{t}.jpg')); gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        depth = np.load(DATA/'rectified'/f'{t}.npy'); started = time.perf_counter()
        if previous is not None:
            for source, state in list(states.items()):
                if source < t-cfg['window']+1:
                    del states[source]
                else:
                    advance(state, previous, gray, depth, np.eye(4), k, flow_cfg)
        flow_ms = 1000*(time.perf_counter()-started)
        started = time.perf_counter()
        result = model.predict(rgb, device='cuda:0', retina_masks=True, verbose=False, **cfg['detector'])[0]
        detector_ms = 1000*(time.perf_counter()-started)
        candidates = []; masks = {}
        for j, box in enumerate(result.boxes):
            if result.names[int(box.cls.item())].lower() != 'person':
                continue
            mask = result.masks.data[j].cpu().numpy() > .5
            valid = mask & (depth > .05) & (depth < 12)
            y, x = np.nonzero(valid)
            row = {'index': j, 'score': float(box.conf.item()), 'box2d': box.xyxy[0].cpu().tolist(), 'valid_pixels': len(x)}
            if len(x) >= 8:
                points = lift(np.c_[x, y], depth[y, x], k, np.eye(4)); lo, hi = np.quantile(points, [.02, .98], axis=0)
                row['box'] = np.r_[(lo+hi)/2, np.maximum(hi-lo, .01)].tolist(); masks[j] = valid
            candidates.append(row)
        eligible = [r for r in candidates if 'box' in r]
        selected = max(eligible, key=lambda r:r['score']) if eligible else None
        if selected is not None:
            mask = cv2.erode(masks[selected['index']].astype(np.uint8), np.ones((3, 3), np.uint8))
            uv = cv2.goodFeaturesToTrack(gray, maxCorners=flow_cfg['max_corners'], qualityLevel=flow_cfg['quality_level'], minDistance=flow_cfg['min_distance_px'], mask=mask)
            uv = uv.reshape(-1, 2) if uv is not None else np.empty((0, 2), np.float32)
            z, valid = sample_depth(depth, uv); uv = uv[valid]; world = lift(uv, z[valid], k, np.eye(4))
            anchor = world.mean(0) if len(world) else np.zeros(3)
            states[t] = {'uv': uv, 'world': world, 'anchor': anchor.copy(), 'initial_anchor': anchor.copy(), 'ids': np.arange(len(uv)), 'active': len(uv) >= 8}
            cv2.imwrite(str(OUT/f'predicted_mask_{t}.png'), masks[selected['index']].astype(np.uint8)*255)
        shifts = {str(s): {'valid': bool(v['active']), 'points': len(v['uv']), 'delta': (v['anchor']-v['initial_anchor']).tolist()} for s, v in states.items()}
        records.append({'time': t, 'candidates': candidates, 'selected': selected, 'flow_shifts': shifts, 'detector_ms': detector_ms, 'flow_ms': flow_ms})
        previous = gray
        print(t, 'persons', len(candidates), 'selected', None if selected is None else round(selected['score'], 3), 'flow', {s:v['valid'] for s,v in shifts.items()}, flush=True)
    save('observations.json', records)


def fuse():
    import torch
    import yaml
    from boxfusion.boxes import GeneralInstance3DBoxes
    from boxfusion.instances import Instances3D
    from boxfusion.box_manager import BoxManager
    from boxfusion.box_fusion import BoxFusion, GPU_MODE
    assert GPU_MODE and torch.cuda.is_available()
    protocol = read('geometry_protocol.json'); records = read('observations.json'); refs = read('references.json')
    k = np.array(protocol['K']); cfg = yaml.safe_load((ROOT/'config/bonn_native.yaml').read_text()); cfg['dataset'] = 'online'; cfg['motion_chain']['enabled'] = False
    cfg['cam'].update({'fx': float(k[0, 0]), 'fy': float(k[1, 1]), 'cx': float(k[0, 2]), 'cy': float(k[1, 2]), 'H': 480, 'W': 640})
    cfg['data']['datadir'] = str(DATA/'rectified')
    save('pfo_configuration.json', cfg); predictions = {}
    with (OUT/'pfo.log').open('w') as log, contextlib.redirect_stdout(log):
        fuser = BoxFusion(cfg)
        np.testing.assert_allclose(fuser.K[:3, :3], k)
        for i, row in enumerate(records):
            t = row['time']; selected = row['selected']; arms = {}; predictions[str(t)] = arms
            if selected is None:
                continue
            arms['latest'] = {'box': selected['box'], 'pfo_updates': 0}
            history = records[max(0, i-2):i+1]
            usable = len(history) == 3 and all(r['selected'] is not None for r in history)
            for mode in ['static', 'oracle', 'flow']:
                shifts = []; okay = usable
                for h in history:
                    source = h['time']
                    if mode == 'static':
                        delta = np.zeros(3)
                    elif mode == 'oracle':
                        delta = np.array(refs[str(t)]['center'])-refs[str(source)]['center']
                    else:
                        state = row['flow_shifts'].get(str(source)); okay &= state is not None and state['valid']
                        delta = np.array(state['delta']) if state else np.zeros(3)
                    shifts.append(delta)
                if not okay:
                    for f in ['mean', 'pfo']:
                        arms[mode+'_'+f] = dict(arms['latest'], fallback_latest=True)
                    continue
                boxes = np.array([h['selected']['box'] for h in history]); original = boxes.copy(); boxes[:, :3] += shifts
                arms[mode+'_mean'] = {'box': boxes.mean(0).tolist(), 'pfo_updates': 0}
                poses = np.tile(np.eye(4), (3, 1, 1)); poses[:, :3, 3] = shifts
                np.testing.assert_allclose(boxes[:, :3]-poses[:, :3, 3], original[:, :3], atol=1e-8)
                ins = Instances3D((480, 640)); ins.pred_boxes_3d = GeneralInstance3DBoxes(boxes.astype(np.float32), np.tile(np.eye(3, dtype=np.float32), (3, 1, 1)))
                ins.cam_pose = torch.tensor(poses, dtype=torch.float32); ins.scores = torch.tensor([h['selected']['score'] for h in history]); ins.pred_boxes = torch.tensor([h['selected']['box2d'] for h in history], dtype=torch.float32)
                ins.frame_id = torch.tensor([h['time'] for h in history]); ins.init_id = torch.arange(3); ins.valid_num = torch.zeros(3, dtype=torch.long); ins.categories = np.array(['person']*3)
                ins.project_3d_boxes(torch.tensor(k, dtype=torch.float32), H=480, W=640)
                current = ins[2:3]; manager = BoxManager(cfg); manager.init_new_predictions(1, 0); manager.fusion_list[0] = [0, 1, 2]
                np.random.seed(0); torch.manual_seed(0); started = time.perf_counter(); fuser.boxfusion(current, ins, manager); torch.cuda.synchronize()
                corners = current.pred_boxes_3d.corners[0].numpy(); lo, hi = corners.min(0), corners.max(0)
                arms[mode+'_pfo'] = {'box': np.r_[(lo+hi)/2, hi-lo].tolist(), 'pfo_updates': len(manager.already_fusion), 'seconds': time.perf_counter()-started}
    save('geometry_predictions.json', predictions); print('fused', len(predictions), 'timestamps', flush=True)


def evaluate():
    refs = read('references.json'); predictions = read('geometry_predictions.json'); observations = read('observations.json'); detail = {}; summary = {}
    names = ['latest', 'static_mean', 'oracle_mean', 'flow_mean', 'static_pfo', 'oracle_pfo', 'flow_pfo']
    for t, arms in predictions.items():
        gt = refs[t]; lo, hi = np.array(gt['lo']), np.array(gt['hi']); detail[t] = {}
        for name in names:
            if name not in arms:
                detail[t][name] = {'missing': True, 'iou': 0., 'center_error_m': None}; continue
            b = np.array(arms[name]['box']); a, z = b[:3]-b[3:]/2, b[:3]+b[3:]/2
            inter = np.maximum(0, np.minimum(z, hi)-np.maximum(a, lo)).prod(); union = np.prod(z-a)+np.prod(hi-lo)-inter
            detail[t][name] = {'missing': False, 'iou': float(inter/union), 'center_error_m': float(np.linalg.norm(b[:3]-gt['center'])), 'size_abs_error_m': np.abs(b[3:]-gt['size']).tolist()}
    evaluation_times = [str(r['time']) for i,r in enumerate(observations) if i>=2 and all(x['selected'] is not None for x in observations[i-2:i+1])]
    for name in names:
        x = [detail[t][name] for t in evaluation_times]; all_rows = [r[name] for r in detail.values()]
        errors = [r['center_error_m'] for r in x if r['center_error_m'] is not None]
        summary[name] = {'paired_windows': len(x), 'mean_iou': float(np.mean([r['iou'] for r in x])) if x else None,
                         'mean_center_error_m': float(np.mean(errors)) if errors else None, 'recall50_paired': float(np.mean([r['iou']>=.5 for r in x])) if x else None,
                         'all_frame_missing': sum(r['missing'] for r in all_rows), 'all_frame_recall50': float(np.mean([r['iou']>=.5 for r in all_rows])),
                         'fallback_frames': sum(v.get(name, {}).get('fallback_latest', False) for v in predictions.values()),
                         'pfo_updates': sum(v.get(name, {}).get('pfo_updates', 0) for v in predictions.values())}
    baseline = summary['latest']; passed = []
    if evaluation_times:
        for name in ['oracle_mean', 'oracle_pfo']:
            v = summary[name]
            if v['mean_iou'] >= baseline['mean_iou']+.02 and v['mean_center_error_m'] <= baseline['mean_center_error_m'] and v['recall50_paired'] >= baseline['recall50_paired']:
                passed.append(name)
    result = {'summary': summary, 'oracle_gate_passed_arms': passed, 'oracle_gate_pass': bool(passed), 'paired_times': evaluation_times, 'detail': detail,
              'scope': '11 annotated timestamps at 1 Hz; forced single-target geometry windows with frozen YOLOE frontend. Actual PFO kernel used, native association bypassed. No AP/duplicate/full-dynamic/30-Hz claim.',
              'input_hashes_unchanged': all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in read('geometry_input_hashes.json').items())}
    save('geometry_results.json', result); print(json.dumps({'summary': summary, 'oracle_gate': passed}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--phase', choices=['prepare', 'infer', 'fuse', 'evaluate'], required=True)
    globals()[p.parse_args().phase]()
