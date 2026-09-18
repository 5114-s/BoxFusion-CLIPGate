#!/usr/bin/env python3
"""Isolated actual CUDA PFO test on causal RGB-D surface observations."""
import argparse
import contextlib
import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np

from run_bonn_surface_track_probe import advance, lift, load_frame, project, sample_depth

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/bonn_motion_pfo_probe_20260914'
PRE=ROOT/'reports/bonn_surface_track_625_675_20260914'
DATA=ROOT/'data_bonn/scene0002_01/frames'


def fit_box(points, rotation):
    local=np.asarray(points)@rotation
    lo,hi=np.quantile(local,[.02,.98],axis=0)
    return np.r_[((lo+hi)/2)@rotation.T,np.maximum(hi-lo,.01)]


def box_corners(box,rotation):
    signs=np.asarray(list(itertools.product([-1,1],repeat=3)))
    return (signs*box[3:]/2)@rotation.T+box[:3]


def align_observation(box,pose,delta):
    shifted=box.copy();shifted[:3]+=delta
    virtual=pose.copy();virtual[:3,3]+=delta
    return shifted,virtual


def transfer(points,source_box,source_rotation,target_box,target_rotation):
    normalized=(points-source_box[:3])@source_rotation/source_box[3:]
    return (normalized*target_box[3:])@target_rotation.T+target_box[:3]


def frozen_hashes():
    paths=[Path(__file__),OUT/'protocol.json',PRE/'tracking_trace.json',PRE/'protocol.json',PRE/'reference.json',
           ROOT/'tools/run_bonn_surface_track_probe.py',ROOT/'config/bonn_native.yaml',
           ROOT/'boxfusion/box_fusion.py',ROOT/'boxfusion/boxes.py',ROOT/'boxfusion/instances.py',
           ROOT/'boxfusion/reliable_views.py']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def prepare():
    cfg=json.loads((OUT/'protocol.json').read_text())
    tracker_cfg=json.loads((PRE/'protocol.json').read_text())
    initial=json.loads((PRE/'tracking_trace.json').read_text())[cfg['target_source']][0]
    times=json.loads((PRE/'results.json').read_text())['timestamps']
    K=np.loadtxt(DATA/'intrinsic/intrinsic_depth.txt')[:3,:3]
    gray,depth,pose=load_frame(625);orientation=pose[:3,:3]
    states={};traces={name:{} for name in ('fit','heldout')}
    cv2.setNumThreads(1);cv2.setRNGSeed(0)
    for name,parity in [('fit',0),('heldout',1)]:
        ids=np.asarray(initial['surviving_seed_ids']);choose=ids%2==parity
        uv=np.asarray(initial['points'],np.float32)[choose];ids=ids[choose]
        z,valid=sample_depth(depth,uv);uv=uv[valid];ids=ids[valid]
        world=lift(uv,z[valid],K,pose)
        states[name]={'uv':uv,'world':world,'anchor':world.mean(0),'ids':ids,'active':True}
    def save(frame):
        for name,s in states.items():
            traces[name][str(frame)]={'active':s['active'],'ids':s['ids'].tolist(),'uv':s['uv'].tolist(),
                                     'world':s['world'].tolist(),'anchor':s['anchor'].tolist()}
    save(625)
    for frame in range(626,676):
        new_gray,depth,pose=load_frame(frame)
        for state in states.values():
            advance(state,gray,new_gray,depth,pose,K,tracker_cfg)
        save(frame);gray=new_gray
    fit=traces['fit'];prepared={}
    velocity=(np.asarray(fit['630']['anchor'])-fit['625']['anchor'])/(times['630']-times['625'])
    for query,frames in cfg['queries'].items():
        needed=[str(f) for f in frames]+[query]
        if not all(fit[f]['active'] for f in needed):
            prepared[query]={'eligible':False,'reason':'fit_track_lost'};continue
        common=sorted(set.intersection(*(set(fit[str(f)]['ids']) for f in frames)))
        if len(common)<cfg['min_points']:
            prepared[query]={'eligible':False,'reason':'insufficient_common_fit_points'};continue
        observations=[]
        for frame in frames:
            r=fit[str(frame)];indices=[r['ids'].index(k) for k in common]
            world=np.asarray(r['world'])[indices];uv=np.asarray(r['uv'])[indices]
            box=fit_box(world,orientation);pose=np.loadtxt(DATA/f'pose/{frame}.txt')
            observations.append({'frame':frame,'box':box.tolist(),'pose':pose.tolist(),
                                 'box2d':np.r_[uv.min(0),uv.max(0)].tolist(),
                                 'flow_delta':(np.asarray(fit[query]['anchor'])-r['anchor']).tolist(),
                                 'cv_delta':(velocity*(times[query]-times[str(frame)])).tolist()})
        prepared[query]={'eligible':True,'common_fit_ids':common,'observations':observations}
    artifact={'queries':prepared,'orientation':orientation.tolist(),'K':K.tolist(),'traces':traces,
              'bootstrap_velocity':velocity.tolist(),'input_sha256':frozen_hashes()}
    (OUT/'prepared.json').write_text(json.dumps(artifact,allow_nan=False)+'\n')
    print('query eligibility',{q:{k:v for k,v in x.items() if k!='observations'} for q,x in prepared.items()})
    print('fit/heldout survivors',{f:{name:len(traces[name][f]['ids']) if traces[name][f]['active'] else 0 for name in traces} for f in ['625','650','675']})


def fuse():
    sys.path.insert(0,str(ROOT))
    import torch
    import yaml
    from boxfusion.boxes import GeneralInstance3DBoxes
    from boxfusion.instances import Instances3D
    from boxfusion.box_manager import BoxManager
    from boxfusion.box_fusion import BoxFusion,GPU_MODE
    assert GPU_MODE and torch.cuda.is_available(),'Actual CUDA required'
    data=json.loads((OUT/'prepared.json').read_text());R=np.asarray(data['orientation']);K=np.asarray(data['K'])
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in data['input_sha256'].items())
    cfg=yaml.safe_load((ROOT/'config/bonn_native.yaml').read_text());cfg['dataset']='online'
    (OUT/'config.yaml').write_text(yaml.safe_dump(cfg))
    outputs={}
    for query,entry in data['queries'].items():
        if not entry['eligible']:
            continue
        outputs[query]={}
        for arm in ('static_pfo','cv_pfo','flow_pfo','flow_latest'):
            observations=entry['observations'];boxes=[];poses=[]
            for o in observations:
                delta=np.zeros(3) if arm=='static_pfo' else np.asarray(o['cv_delta'] if arm=='cv_pfo' else o['flow_delta'])
                original=np.asarray(o['box']);pose=np.asarray(o['pose'])
                box,virtual=align_observation(original,pose,delta)
                before,_=project(box_corners(original,R),K,pose)
                after,_=project(box_corners(box,R),K,virtual)
                np.testing.assert_allclose(before,after,atol=1e-8)
                boxes.append(box);poses.append(virtual)
            if arm=='flow_latest':
                outputs[query][arm]={'box':boxes[-1].tolist(),'rotation':R.tolist(),'pfo_updates':0};continue
            np.random.seed(0);torch.manual_seed(0)
            with (OUT/f'{query}_{arm}.log').open('w') as log,contextlib.redirect_stdout(log):
                fuser=BoxFusion(cfg);manager=BoxManager(cfg)
                history=Instances3D((480,640))
                history.pred_boxes_3d=GeneralInstance3DBoxes(np.asarray(boxes,dtype=np.float32),np.tile(R.astype(np.float32),(3,1,1)))
                history.cam_pose=torch.tensor(np.asarray(poses),dtype=torch.float32)
                history.scores=torch.ones(3);history.pred_boxes=torch.tensor([o['box2d'] for o in observations],dtype=torch.float32)
                history.frame_id=torch.tensor([o['frame'] for o in observations],dtype=torch.long)
                history.init_id=torch.arange(3);history.valid_num=torch.zeros(3,dtype=torch.long)
                history.categories=np.array(['tracked_surface']*3)
                history.project_3d_boxes(torch.tensor(K,dtype=torch.float32),H=480,W=640)
                current=history[2:3]
                manager.init_new_predictions(1,0);manager.fusion_list[0]=[0,1,2]
                started=time.perf_counter();fuser.boxfusion(current,history,manager);torch.cuda.synchronize()
                outputs[query][arm]={'box':current.pred_boxes_3d.tensor[0].tolist(),
                                     'rotation':current.pred_boxes_3d.R[0].tolist(),
                                     'pfo_updates':len(manager.already_fusion),
                                     'reliable_views':fuser.reliable_view_stats,
                                     'seconds':time.perf_counter()-started}
            print(query,arm,'updates',outputs[query][arm]['pfo_updates'],flush=True)
    (OUT/'predictions.json').write_text(json.dumps(outputs,indent=2,allow_nan=False)+'\n')


def evaluate():
    data=json.loads((OUT/'prepared.json').read_text())
    predictions=json.loads((OUT/'predictions.json').read_text())
    reference=json.loads((PRE/'reference.json').read_text())
    R=np.asarray(data['orientation']);results={}
    for query,arms in predictions.items():
        latest=str(data['queries'][query]['observations'][-1]['frame'])
        start=data['traces']['heldout'][latest];end=data['traces']['heldout'][query]
        ids=sorted(set(start['ids'])&set(end['ids'])) if start['active'] and end['active'] else []
        ids=np.asarray(ids,dtype=int)
        if len(ids)<8:
            results[query]={'eligible':False,'reason':'heldout_track_insufficient'};continue
        assert all(ids%2==1) and all(np.asarray(data['queries'][query]['common_fit_ids'])%2==0)
        begin_idx=[start['ids'].index(int(i)) for i in ids];end_idx=[end['ids'].index(int(i)) for i in ids]
        old=np.asarray(start['world'])[begin_idx];current=np.asarray(end['world'])[end_idx]
        uv=np.rint(np.asarray(end['uv'])[end_idx]).astype(int)
        mask=np.zeros((480,640),np.uint8);cv2.fillPoly(mask,[np.asarray(reference['polygons'][query],np.int32)],1)
        membership=float(mask[uv[:,1],uv[:,0]].mean())
        source=np.asarray(data['queries'][query]['observations'][-1]['box'])
        details={}
        for arm,p in arms.items():
            estimate=transfer(old,source,R,np.asarray(p['box']),np.asarray(p['rotation']))
            error=np.linalg.norm(estimate-current,axis=1)
            details[arm]={'median_m':float(np.median(error)),'mean_m':float(error.mean()),
                          'p90_m':float(np.quantile(error,.9)),'per_point_errors_m':error.tolist(),
                          'predicted_points':estimate.tolist(),'pfo_updates':p['pfo_updates']}
        results[query]={'eligible':membership>=.8,'heldout_points':len(ids),'heldout_person_fraction':membership,
                        'latest_history_frame':int(latest),'point_ids':ids.tolist(),'current_points':current.tolist(),'arms':details}
    gate=False
    if all(results.get(q,{}).get('eligible',False) for q in ('650','675')):
        a=results['650']['arms'];b=results['675']['arms']
        gate=(a['flow_pfo']['median_m']<=.9*a['flow_latest']['median_m'] and
              a['flow_pfo']['median_m']<min(a['static_pfo']['median_m'],a['cv_pfo']['median_m']) and
              b['flow_pfo']['median_m']<=b['flow_latest']['median_m']+.02)
    result={'queries':results,'gate_pass':bool(gate),'input_hashes_unchanged':all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in data['input_sha256'].items()),
            'detector_forward_runs':0,'training_runs':0,'pfo_calls':sum(a.endswith('_pfo') for row in predictions.values() for a in row),
            'scope':'Surface-patch motion/shape transfer with inherited point identity; not whole-human 3D AP, native association, static regression or independent 3D ground truth.'}
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for q,item in results.items():
        print(q,'eligible',item['eligible'],'heldout',item.get('heldout_points'))
        print({a:{k:x for k,x in v.items() if k in ['median_m','mean_m','pfo_updates']} for a,v in item.get('arms',{}).items()})
    print('gate',gate)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','fuse','evaluate'],required=True)
    {'prepare':prepare,'fuse':fuse,'evaluate':evaluate}[p.parse_args().phase]()
