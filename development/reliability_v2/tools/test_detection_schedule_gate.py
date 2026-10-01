"""No-inference feasibility gate for causal fixed-budget detection scheduling.

This is a hindsight opportunity audit on a FIXED native history, not AP, not
an implemented live scheduler. All target seeds predate the audited interval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.verify_true_fusion_capture import Evidence

SCENES = ['42446540', '42897501', '42897521']
CAPTURE = ROOT/'reports/ca1m_true_fusion_pilot_v2_20260908'
DATA = Path('/extra/ZhaoX/boxfusion_ca1m')
PROTOCOL = {
    'schema': 'boxfusion.detection_schedule_opportunity_gate.v1',
    'scenes': SCENES, 'scene_selection': 'First 3 official CA1M scenes; no effect-based selection.',
    'GT_allowed': False, 'model_inference': False,
    'baseline': 'Native gap20 observation history, used ONLY for opportunity screening; NOT an M1+M2 AP comparison.',
    'interval': 'Each complete interval (previous gap20 frame, next gap20 frame]; exclude startup and trailing incomplete interval.',
    'seeds': 'At most 16 highest-score native boxes in previous gap20 frame; no current-interval detections or final map.',
    'past_history': 'At most top16 per frame in the preceding 10 gap20 frames; same-object support proxy AABB IoU>.25.',
    'deficit': 'Fewer than 3 distinct supporting frames OR angular spread relative to latest camera <15 degrees.',
    'image': 'Resize current RGB and depth to 512x384; scale each calibrated intrinsic independently.',
    'quality': 'ROI Laplacian energy / (ROI intensity variance+1); clipped/unclipped projection area; valid depth fraction inside predicted corner-z interval +/-0.1m.',
    'valid': 'All corners >1mm camera depth; image ROI >=16px each side; visible area fraction>=.5; >=16 valid depth pixels; depth-compatible fraction>=.2.',
    'opportunity': 'Compared with SAME seed at scheduled deadline: recover an invalid view, OR +25% sharpness, OR +.20 visible fraction, OR +5deg parallax; other quality ratios >=.9 and parallax no worse by >2deg.',
    'gate': 'At least 10% of deficit-bearing intervals have an opportunity, and at least 2 scenes individually reach 10%; at least 30 deficit-bearing intervals required.',
    'next_stage': 'Only if gate passes: freeze one causal policy then three-arm same-budget AP test versus current M1+M2; no threshold sweeps.',
    'limitations': ['Targets can be wrong; depth support and sharpness are proxies, not true detection quality.',
                    'Best alternate is chosen retrospectively only to test opportunity existence; it is NOT a causal policy.',
                    'Passing cannot establish AP gain, runtime, or generalization; failing only stops this fixed gate/configuration.'],
}


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def angles(center, cameras, reference_camera):
    rays = cameras-center
    reference = reference_camera-center
    denom = np.maximum(np.linalg.norm(rays, axis=-1)*np.linalg.norm(reference), 1e-9)
    return np.rad2deg(np.arccos(np.clip(rays@reference/denom, -1, 1)))


def seed_ids(raw, previous):
    selected = []
    for frame in range(max(0, previous-180), previous+1, 20):
        ids = np.flatnonzero(raw['frame_ids'] == frame)
        selected.extend(ids[np.argsort(-raw['scores'][ids], kind='stable')[:16]])
    history = np.asarray(selected, dtype=int)
    current = history[raw['frame_ids'][history] == previous]
    seeds = []
    for i in current:
        box = raw['corners'][i]
        low, high = box.min(0), box.max(0)
        hlow, hhigh = raw['corners'][history].min(1), raw['corners'][history].max(1)
        inter = np.maximum(0, np.minimum(hhigh,high)-np.maximum(hlow,low)).prod(1)
        union = (hhigh-hlow).prod(1)+(high-low).prod()-inter
        matched = history[inter/np.maximum(union,1e-9) > .25]
        frames = np.unique(raw['frame_ids'][matched])
        spread = float(angles(box.mean(0), raw['cam_poses'][matched,:3,3], raw['cam_poses'][i,:3,3]).max(initial=0))
        if len(frames)<3 or spread<15:
            seeds.append({'raw_id':int(i), 'support_frames':frames.tolist(), 'past_spread_deg':spread})
    return seeds


def image_metrics(corners, pose, reference_pose, gray, depth, intrinsic, depth_intrinsic):
    inverse = np.linalg.inv(pose)
    cam = corners@inverse[:3,:3].T+inverse[:3,3]
    if not np.isfinite(cam).all() or cam[:,2].min() <= .001:
        return {'valid':False, 'reason':'depth_or_projection'}
    projected = cam@intrinsic.T
    uv = projected[:,:2]/projected[:,2:]
    low, high = uv.min(0),uv.max(0)
    unclipped = np.maximum(high-low,0).prod()
    lo,hi = np.maximum(low,0),np.minimum(high,[512,384])
    width,height = np.maximum(hi-lo,0)
    visible = float(width*height/max(unclipped,1e-9))
    if width<16 or height<16 or visible<.5:
        return {'valid':False,'reason':'small_or_truncated','visible':visible}
    x0,y0 = np.floor(lo).astype(int); x1,y1 = np.ceil(hi).astype(int)
    crop = gray[y0:y1,x0:x1].astype(np.float32)
    lap = cv2.Laplacian(crop, cv2.CV_32F, ksize=1)
    sharp = float(np.mean(lap*lap)/(crop.var()+1.))
    duv = cam@depth_intrinsic.T; duv = duv[:,:2]/duv[:,2:]
    dlo = np.maximum(np.floor(duv.min(0)).astype(int),0)
    dhi = np.minimum(np.ceil(duv.max(0)).astype(int),[512,384])
    if np.any(dhi<=dlo):
        return {'valid':False,'reason':'depth_roi'}
    roi = depth[dlo[1]:dhi[1]:4,dlo[0]:dhi[0]:4]
    valid = roi[(roi>.1)&(roi<10)]
    compatibility = float(np.mean((valid>=cam[:,2].min()-.1)&(valid<=cam[:,2].max()+.1))) if len(valid) else 0.
    if len(valid)<16 or compatibility<.2:
        return {'valid':False,'reason':'depth_support','visible':visible,'sharpness':sharp,'depth_support':compatibility}
    parallax = float(angles(corners.mean(0),pose[None,:3,3],reference_pose[:3,3])[0])
    return {'valid':True,'visible':visible,'sharpness':sharp,'depth_support':compatibility,'parallax_deg':parallax}


def improvement(alternative, baseline):
    if not alternative['valid']:
        return None
    if not baseline['valid']:
        return 'view_recovered'
    sharp = alternative['sharpness']/max(baseline['sharpness'],1e-9)
    visible = alternative['visible']-baseline['visible']
    angle = alternative['parallax_deg']-baseline['parallax_deg']
    if (sharp<.9 or alternative['visible']<.9*baseline['visible']
            or alternative['depth_support']<.9*baseline['depth_support'] or angle < -2):
        return None
    if angle>=5: return 'parallax'
    if visible>=.20: return 'visibility'
    if sharp>=1.25: return 'sharpness'
    return None


def self_test():
    base={'valid':True,'visible':.8,'sharpness':1.,'depth_support':.8,'parallax_deg':5.}
    assert improvement(dict(base,sharpness=1.3),base)=='sharpness'
    assert improvement(dict(base,parallax_deg=11.),base)=='parallax'
    assert improvement(dict(base,sharpness=1.3,parallax_deg=0.),base) is None
    assert improvement(dict(base,sharpness=1.3,depth_support=.1),base) is None
    assert improvement(base,dict(valid=False))=='view_recovered'
    assert improvement(dict(valid=False),base) is None
    assert np.allclose(angles(np.zeros(3),np.array([[1.,0,0]]),np.array([0.,1,0])),90)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();self_test();args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'protocol.json',PROTOCOL)
    evidence=Evidence();evidence.remember(__file__);evidence.remember(args.output/'protocol.json')
    evidence.remember(ROOT/'tools/verify_true_fusion_capture.py')
    def guard(event,values):
        if event=='open' and values and isinstance(values[0],(str,bytes)):
            if any(name in str(values[0]) for name in ('after_filter_boxes.npy','full_annotations.json')):
                raise RuntimeError('GT forbidden in schedule opportunity gate')
    sys.addaudithook(guard)
    assert evidence.json(CAPTURE/'integrity.json')['checkpass']
    audits=[];started=time.perf_counter()
    for scene in SCENES:
        raw=evidence.npz(CAPTURE/scene/'observations.npz')
        directory=DATA/scene
        poses=np.load(evidence.remember(directory/'all_poses.npy'))
        kc=np.loadtxt(evidence.remember(directory/'K_rgb.txt')).reshape(3,3)
        kd=np.loadtxt(evidence.remember(directory/'K_depth.txt')).reshape(3,3)
        intervals=[];reason_counts={}
        for previous in range(0,len(poses)-20,20):
            seeds=seed_ids(raw,previous)
            if not seeds: continue
            assert all(max(seed['support_frames'])<=previous for seed in seeds)
            metrics={}
            for frame in range(previous+1,previous+21):
                if not np.isfinite(poses[frame]).all(): continue
                with Image.open(evidence.remember(directory/'rgb'/f'{frame}.png')) as rgb:
                    k=kc.copy();k[0]*=512/rgb.width;k[1]*=384/rgb.height
                    gray=np.asarray(rgb.convert('L').resize((512,384),Image.Resampling.BILINEAR))
                with Image.open(evidence.remember(directory/'depth'/f'{frame}.png')) as dep:
                    dk=kd.copy();dk[0]*=512/dep.width;dk[1]*=384/dep.height
                    depth=np.asarray(dep.resize((512,384),Image.Resampling.NEAREST),dtype=np.float32)/1000
                metrics[frame]=[image_metrics(raw['corners'][seed['raw_id']],poses[frame],poses[previous],gray,depth,k,dk) for seed in seeds]
            deadline=previous+20
            if deadline not in metrics:
                raise ValueError(f'Missing baseline deadline {scene}/{deadline}')
            opportunities=[]
            for index,seed in enumerate(seeds):
                for frame in range(previous+1,deadline):
                    if frame not in metrics:continue
                    reason=improvement(metrics[frame][index],metrics[deadline][index])
                    if reason is not None:
                        opportunities.append({'seed':seed,'frame':frame,'reason':reason,
                                              'alternative':metrics[frame][index],'baseline':metrics[deadline][index]})
                        reason_counts[reason]=reason_counts.get(reason,0)+1
                        break  # first opportunity per seed, never tune to the best one
            intervals.append({'previous':previous,'deadline':deadline,'deficit_seeds':len(seeds),
                              'opportunities':opportunities})
        count=sum(bool(x['opportunities']) for x in intervals)
        audit={'scene':scene,'input_frames':len(poses),'deficit_intervals':len(intervals),
               'opportunity_intervals':count,'opportunity_fraction':count/max(len(intervals),1),
               'opportunity_reasons_per_seed_interval':reason_counts,'intervals':intervals}
        audits.append(audit)
        write_json(args.output/f'{scene}.json',audit)
        print(json.dumps({k:v for k,v in audit.items() if k!='intervals'}),flush=True)
    total=sum(a['deficit_intervals'] for a in audits);hits=sum(a['opportunity_intervals'] for a in audits)
    passed=total>=30 and hits>=.1*total and sum(a['opportunity_fraction']>=.1 for a in audits)>=2
    evidence.unchanged()
    result={'completed':True,'gate_passed':passed,'deficit_intervals':total,'opportunity_intervals':hits,
            'opportunity_fraction':hits/max(total,1),'wall_seconds':time.perf_counter()-started,
            'GT_reads':0,'model_inferences':0,'self_tests':'passed','inputs_unchanged':True,
            'scene_summary':[{k:v for k,v in a.items() if k!='intervals'} for a in audits],
            'interpretation':'Opportunity proxy only; not AP or a causal selection result.'}
    write_json(args.output/'results.json',result);write_json(args.output/'input_sha256.json',evidence.read_hashes)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
