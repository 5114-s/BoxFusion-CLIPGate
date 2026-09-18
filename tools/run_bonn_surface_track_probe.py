#!/usr/bin/env python3
"""Causal CPU RGB-D point tracking during one frozen detector gap."""
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/bonn_surface_track_625_675_20260914'
DATA = ROOT/'data_bonn/scene0002_01/frames'


def lift(uv, z, K, pose):
    pc = np.c_[uv,np.ones(len(uv))]@np.linalg.inv(K).T*np.asarray(z)[:,None]
    return pc@pose[:3,:3].T+pose[:3,3]


def project(points, K, pose):
    pc = (np.asarray(points)-pose[:3,3])@pose[:3,:3]
    hp = pc@K.T
    return hp[:,:2]/hp[:,2:],pc[:,2]


def sample_depth(depth, uv):
    uv = np.asarray(uv).reshape(-1,2)
    finite = np.isfinite(uv).all(1)
    xy = np.rint(np.where(np.isfinite(uv),uv,0)).astype(int)
    valid = finite&(xy[:,0]>=0)&(xy[:,0]<depth.shape[1])&(xy[:,1]>=0)&(xy[:,1]<depth.shape[0])
    z = np.zeros(len(uv)); z[valid] = depth[xy[valid,1],xy[valid,0]]
    valid &= (z>.05)&(z<12)
    return z,valid


def robust_translation(displacements, cfg):
    if len(displacements)<cfg['min_points']:
        return None,np.zeros(len(displacements),bool)
    median = np.median(displacements,axis=0)
    residual = np.linalg.norm(displacements-median,axis=1)
    middle = np.median(residual)
    threshold = max(cfg['translation_residual_floor_m'],middle+cfg['robust_mad_multiplier']*1.4826*np.median(abs(residual-middle)))
    keep = residual<=threshold
    if keep.sum()<cfg['min_points'] or keep.mean()<cfg['min_motion_inlier_fraction']:
        return None,keep
    return np.median(displacements[keep],axis=0),keep


def timestamp_map():
    # Reproduce the converter's accepted RGB ordering, without image conversion.
    raw = ROOT/'data_bonn_dl/rgbd_bonn_crowd'
    rgb = sorted(float(p.stem) for p in (raw/'rgb').glob('*.png'))
    depth = np.asarray(sorted(float(p.stem) for p in (raw/'depth').glob('*.png')))
    accepted = []
    for t in rgb:
        i = np.searchsorted(depth,t)
        if min(abs(depth[j]-t) for j in (i-1,i) if 0<=j<len(depth))<=.02:
            accepted.append(t)
    assert len(accepted)==927
    return accepted


def load_frame(frame):
    gray = cv2.imread(str(DATA/f'color/{frame}.jpg'),cv2.IMREAD_GRAYSCALE)
    depth = cv2.imread(str(DATA/f'depth/{frame}.png'),cv2.IMREAD_UNCHANGED).astype(float)/5000.
    pose = np.loadtxt(DATA/f'pose/{frame}.txt')
    return gray,depth,pose


def snapshot(state,frame):
    return {'frame':frame,'active':state['active'],'points':state['uv'].tolist(),
            'anchor_world':state['anchor'].tolist(),'surviving_seed_ids':state['ids'].tolist(),
            'step_audit':state.get('step_audit',{})}


def advance(state, previous_gray, gray, depth, pose, K, cfg):
    if not state['active']:
        return
    old = state['uv'].astype(np.float32).reshape(-1,1,2)
    options = dict(winSize=(cfg['lk_window'],)*2,maxLevel=cfg['lk_max_level'],
                   criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    new,forward,_ = cv2.calcOpticalFlowPyrLK(previous_gray,gray,old,None,**options)
    if new is None:
        state['active']=False; state['step_audit']={'reason':'no_lk_output'}; return
    back,backward,_ = cv2.calcOpticalFlowPyrLK(gray,previous_gray,new,None,**options)
    if back is None:
        state['active']=False; state['step_audit']={'reason':'no_backward_output'}; return
    uv = new.reshape(-1,2)
    z,depth_valid = sample_depth(depth,uv)
    fb = np.linalg.norm(back.reshape(-1,2)-old.reshape(-1,2),axis=1)
    valid = forward.ravel().astype(bool)&backward.ravel().astype(bool)&depth_valid&(fb<=cfg['fb_max_px'])
    selected = np.flatnonzero(valid)
    world = lift(uv[valid],z[valid],K,pose)
    delta,inlier = robust_translation(world-state['world'][valid],cfg)
    state['step_audit']={'input_points':len(old),'lk_depth_fb_valid':len(selected),
                         'motion_inliers':int(inlier.sum()),'delta_world':None if delta is None else delta.tolist()}
    if delta is None:
        state['active']=False; return
    selected = selected[inlier]
    state['uv']=uv[selected];state['world']=world[inlier];state['ids']=state['ids'][selected]
    state['anchor']=state['anchor']+delta


def evaluate(predicted_world, K, pose, depth, polygon):
    uv,z = project(predicted_world,K,pose)
    measured,valid = sample_depth(depth,uv)
    xy = np.rint(np.where(np.isfinite(uv),uv,0)).astype(int)
    mask = np.zeros(depth.shape,np.uint8);cv2.fillPoly(mask,[np.asarray(polygon,np.int32)],1)
    on_person = np.zeros(len(uv),bool)
    on_person[valid]=mask[xy[valid,1],xy[valid,0]].astype(bool)
    supported=on_person&(np.abs(z-measured)<=.1)&(z>.05)
    return {'points':len(uv),'person_pixel_fraction':float(on_person.mean()),
            'person_depth_support_fraction':float(supported.mean()),
            'median_depth_residual_m':float(np.median(abs(z[on_person]-measured[on_person]))) if on_person.any() else None,
            'projected_uv':uv.tolist()}


def main():
    cv2.setNumThreads(1);cv2.setRNGSeed(0)
    cfg=json.loads((OUT/'protocol.json').read_text());K=np.loadtxt(DATA/'intrinsic/intrinsic_depth.txt')[:3,:3]
    paths=[Path(__file__),OUT/'protocol.json',OUT/'reference.json',ROOT/'reports/bonn_detect_20260914/yoloe_candidates.json']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    timestamps=timestamp_map()
    records=json.loads(paths[-1].read_text())
    initial=next(r for r in records if r['scene']==cfg['scene'] and r['frame']==cfg['start'])
    candidates=[(j,d) for j,d in enumerate(initial['detections']) if d['label']=='person']
    target=f"625:{max(candidates,key=lambda item:item[1]['score'])[0]}"
    previous_gray,depth,pose=load_frame(cfg['start'])
    yy,xx=np.indices(depth.shape);uv=np.c_[xx.ravel(),yy.ravel()]
    world=lift(uv,depth.ravel(),K,pose).reshape(480,640,3)
    states={};traces={};bootstrap={};started=time.perf_counter()
    for index,d in candidates:
        x1,y1,x2,y2=d['box2d']
        seed=(xx>=x1)&(xx<=x2)&(yy>=y1)&(yy<=y2)&(depth>.05)&(depth<12)
        seed &= np.all((world>=d['world_aabb_lo'])&(world<=d['world_aabb_hi']),axis=2)
        seed=cv2.erode(seed.astype(np.uint8),np.ones((3,3),np.uint8))
        points=cv2.goodFeaturesToTrack(previous_gray,maxCorners=cfg['max_corners'],qualityLevel=cfg['quality_level'],minDistance=cfg['min_distance_px'],mask=seed)
        points=np.zeros((0,2),np.float32) if points is None else points.reshape(-1,2)
        z,valid=sample_depth(depth,points);points=points[valid]
        pw=lift(points,z[valid],K,pose);source=f'625:{index}'
        if len(pw)<cfg['min_points']:
            raise RuntimeError(f'Insufficient initialization points: {source}')
        states[source]={'uv':points,'world':pw,'anchor':np.mean(pw,axis=0),'ids':np.arange(len(pw)),'active':True}
        traces[source]=[snapshot(states[source],625)]
    for frame in range(626,676):
        gray,depth,pose=load_frame(frame)
        for source,state in states.items():
            advance(state,previous_gray,gray,depth,pose,K,cfg)
            traces[source].append(snapshot(state,frame))
            if frame==cfg['bootstrap_end'] and state['active']:
                anchor=state['anchor'].copy()
                bootstrap[source]={'anchor':anchor,'cloud':state['world'].copy(),
                                   'velocity':(anchor-np.asarray(traces[source][0]['anchor_world']))/(timestamps[frame]-timestamps[625])}
        previous_gray=gray
    elapsed=time.perf_counter()-started
    (OUT/'tracking_trace.json').write_text(json.dumps(traces,allow_nan=False)+'\n')
    # Inference has finished. Reference polygons are first read here.
    reference=json.loads((OUT/'reference.json').read_text())
    assert target==reference['target_source']
    evaluations={};clouds={}
    for frame in cfg['evaluation_frames']:
        _,depth,pose=load_frame(frame);evaluations[str(frame)]={};clouds[str(frame)]={}
        for source in states:
            state=traces[source][frame-625]
            item={'flow_active':state['active'],'last_retained_points':len(state['points'])}
            if source==target and source in bootstrap:
                base=bootstrap[source];now=np.asarray(state['anchor_world'])
                movements={'hold':np.zeros(3),'constant_velocity':base['velocity']*(timestamps[frame]-timestamps[630]),
                           'surface_flow':now-base['anchor']}
                item['arms']={}
                for arm,delta in movements.items():
                    if arm=='surface_flow' and not state['active']:
                        item['arms'][arm]={'output':False,'person_depth_support_fraction':0.};continue
                    cloud=base['cloud']+delta
                    result=evaluate(cloud,K,pose,depth,reference['polygons'][str(frame)])
                    result['output']=True
                    anchor_test=evaluate((base['anchor']+delta)[None],K,pose,depth,reference['polygons'][str(frame)])
                    result['anchor']=anchor_test
                    item['arms'][arm]=result;clouds[str(frame)][arm]=cloud.tolist()
                if state['active']:
                    p=np.asarray(state['points']);z,valid=sample_depth(depth,p)
                    actual=evaluate(lift(p[valid],z[valid],K,pose),K,pose,depth,reference['polygons'][str(frame)])
                    item['tracked_pixel_person_fraction']=actual['person_pixel_fraction']
            if source!=target and state['active']:
                p=np.asarray(state['points']);z,valid=sample_depth(depth,p)
                item['control_points_on_target_fraction']=evaluate(lift(p[valid],z[valid],K,pose),K,pose,depth,reference['polygons'][str(frame)])['person_pixel_fraction']
            evaluations[str(frame)][source]=item
    focus=evaluations['650'][target];arms=focus.get('arms',{});flow=arms.get('surface_flow',{})
    gate=bool(focus['flow_active'] and focus.get('tracked_pixel_person_fraction',0)>=.8 and flow.get('person_depth_support_fraction',0)>=.5 and all(flow.get('person_depth_support_fraction',0)>=arms.get(a,{}).get('person_depth_support_fraction',0)+.1 for a in ['hold','constant_velocity']))
    artifact={'target':target,'evaluations':evaluations,'gate_pass':gate,'protocol':cfg,
              'bootstrap':{s:{k:v.tolist() for k,v in b.items()} for s,b in bootstrap.items()},
              'elapsed_cpu_seconds_including_frame_reads':elapsed,'timestamps':{str(f):timestamps[f] for f in range(625,676)},
              'input_sha256':hashes,'frozen_inputs_unchanged':all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items()),
              'detector_forward_runs':0,'training_runs':0,'gpu_runs':0}
    (OUT/'results.json').write_text(json.dumps(artifact,indent=2,allow_nan=False)+'\n')
    (OUT/'comparison_clouds.json').write_text(json.dumps(clouds,allow_nan=False)+'\n')
    for frame,items in evaluations.items():
        print(frame,{s:{k:v for k,v in item.items() if k!='arms'} for s,item in items.items()})
        print({a:{k:v for k,v in value.items() if k not in ['projected_uv','anchor']} for a,value in items[target].get('arms',{}).items()})
    print('gate',gate,'seconds',elapsed,flush=True)


if __name__=='__main__':
    main()
