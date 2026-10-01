#!/usr/bin/env python3
"""Frozen, causal identity recovery; geometry and visibility remain unchanged."""
import copy
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from run_bonn_fullbox_loop import assign, hellinger, identity_purity
from run_bonn_surface_track_probe import lift, load_frame, timestamp_map
from run_bonn_crowd_ablation import match_frame, identity_diagnostic

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/bonn_identity_recovery_20260914'
BASE=ROOT/'reports/bonn_fullbox_loop_20260914'
OLD=ROOT/'reports/bonn_crowd_native_dynamic_20260914'
DATA=ROOT/'data_bonn/scene0002_01/frames'


def descriptor(d,color,depth,pose,K,cfg):
    yy,xx=np.indices(depth.shape); x1,y1,x2,y2=d['box2d']
    mask=(xx>=x1)&(xx<=x2)&(yy>=y1)&(yy<=y2)&(depth>.05)&(depth<12)
    y,x=np.nonzero(mask); world=lift(np.c_[x,y],depth[y,x],K,pose)
    valid=np.all((world>=d['world_aabb_lo'])&(world<=d['world_aabb_hi']),axis=1)
    mask[:]=False;mask[y[valid],x[valid]]=True
    mask=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    count=int(mask.sum())
    if count<cfg['min_descriptor_pixels']:return None,count
    hsv=cv2.cvtColor(color,cv2.COLOR_BGR2HSV)[mask]
    h,_=np.histogramdd(hsv,bins=(8,4,4),range=((0,180),(0,256),(0,256)))
    return h.ravel()/h.sum(),count


class Recovery:
    def __init__(self,mode,cfg):
        self.mode,self.cfg=mode,cfg
        self.mapping={};self.profiles={};self.next_id=0;self.events=[];self.pair_audit=[]

    def step(self,frame,now,rows,features):
        people=[r for r in rows if r['label']=='person']
        active={self.mapping[r['track_id']] for r in people if r['track_id'] in self.mapping}
        fresh=[r for r in people if r['track_id'] not in self.mapping]
        # A live identity may not be assigned to another currently visible tracklet.
        available=[k for k,p in self.profiles.items() if k not in active and 0<now-p['last_time']<=self.cfg['memory_seconds']]
        cost=np.full((len(available),len(fresh)),np.inf)
        pair={}
        for i,k in enumerate(available):
            p=self.profiles[k]; age=now-p['last_time']
            reach=min(self.cfg['maximum_reach_m'],self.cfg['reach_noise_m']+self.cfg['maximum_speed_m_s']*age)
            velocity=np.zeros(3)
            if len(p['history'])>=2:
                ta,ca=p['history'][0];tb,cb=p['history'][-1]
                velocity=(np.asarray(cb)-ca)/max(tb-ta,1e-6)
                velocity*=min(1.,self.cfg['maximum_speed_m_s']/max(np.linalg.norm(velocity),1e-9))
            prediction=np.asarray(p['center'])+velocity*age
            for j,row in enumerate(fresh):
                center=np.asarray(row['center']);distance=float(np.linalg.norm(center-p['center']))
                residual=float(np.linalg.norm(center-prediction)/reach)
                feature=features.get(row['source'])
                h=min((hellinger(proto,feature) for _,proto in p['bank']),default=1.)
                reachable=distance<=reach
                valid=reachable and (self.mode=='motion_only' or h<=self.cfg['appearance_hellinger_max'])
                if valid:
                    w=self.cfg['appearance_weight'] if self.mode=='appearance_motion' else 0.
                    cost[i,j]=w*h/self.cfg['appearance_hellinger_max']+(1-w)*residual
                detail={'frame':frame,'identity':k,'new_source':row['source'],'old_sources':list(p['sources']),
                        'age_seconds':age,'distance_m':distance,'reach_m':reach,'appearance_distance':h,
                        'reachable':bool(reachable),'eligible':bool(valid),'cost':float(cost[i,j]) if valid else None}
                self.pair_audit.append(detail);pair[i,j]=detail
        margin=self.cfg['ambiguity_margin'] if self.mode=='appearance_motion' else None
        matches={j:i for i,j in assign(cost,margin)}
        for j,row in enumerate(fresh):
            if j in matches:
                i=matches[j];k=available[i]
                self.events.append({**pair[i,j],'event':'recovered','base_track_id':row['track_id']})
                # Unsupported interval cannot be used as a short-term velocity sample.
                self.profiles[k]['history']=[]
            else:
                k=self.next_id;self.next_id+=1
                self.profiles[k]={'bank':[],'sources':[],'history':[]}
                self.events.append({'frame':frame,'identity':k,'new_source':row['source'],
                                    'base_track_id':row['track_id'],'event':'new_identity'})
            self.mapping[row['track_id']]=k
        outputs=[]
        for row in rows:
            if row['label']!='person':outputs.append(copy.deepcopy(row));continue
            k=self.mapping[row['track_id']];p=self.profiles[k]
            p.update(center=list(row['center']),last_time=now)
            p['history'].append((now,list(row['center'])))
            p['history']=p['history'][-self.cfg['velocity_history_frames']:]
            if row['observed_now'] and row['source'] not in p['sources']:
                p['sources'].append(row['source'])
                feature=features.get(row['source'])
                if feature is not None:
                    p['bank'].append((row['source'],feature.copy()))
                    if len(p['bank'])>self.cfg['max_prototypes']:
                        p['bank']=[p['bank'][0]]+p['bank'][-(self.cfg['max_prototypes']-1):]
            outputs.append({**copy.deepcopy(row),'base_track_id':row['track_id'],'track_id':k})
        used={r['track_id'] for r in outputs if r['label']=='person'}
        assert len(used)==len(people)
        expired=[k for k,p in self.profiles.items() if k not in used and now-p['last_time']>self.cfg['memory_seconds']]
        for k in expired:del self.profiles[k]
        while len(self.profiles)>self.cfg['max_profiles']:
            dormant=[k for k in self.profiles if k not in used]
            if not dormant:raise RuntimeError('Active profile capacity exceeded')
            del self.profiles[min(dormant,key=lambda k:self.profiles[k]['last_time'])]
        return outputs


def evaluate(trace,records,reference):
    frames=[];truth={}
    for state in trace:
        f=state['frame']
        if f not in records:continue
        by_source={r['source']:r['track_id'] for r in state['outputs'] if r['label']=='person' and r['observed_now']}
        ps=[{'box':d['box2d'],'track_id':by_source[f'{f}:{j}'],'source':f'{f}:{j}'}
            for j,d in enumerate(records[f]['detections']) if f'{f}:{j}' in by_source]
        metric=match_frame(ps,reference[str(f)])
        frames.append({'frame':f,**metric})
        for m in metric['matches']:
            truth[ps[m['prediction']]['source']]=m['identity']
    return {'continuity':identity_diagnostic(frames),'purity':identity_purity(frames),'per_frame':frames},truth


def main():
    cv2.setNumThreads(1);cfg=json.loads((OUT/'protocol.json').read_text())
    paths=[Path(__file__),OUT/'protocol.json',BASE/'flow_latest_trace.json',BASE/'results.json',BASE/'audit.json',
           ROOT/'reports/bonn_detect_20260914/yoloe_candidates.json',OLD/'annotations.json',
           ROOT/'tools/run_bonn_fullbox_loop.py',ROOT/'tools/run_bonn_surface_track_probe.py',ROOT/'tools/run_bonn_crowd_ablation.py']
    # The reused motion/geometry baseline must still correspond to its recorded inputs.
    baseline_audit=json.loads((BASE/'audit.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in baseline_audit['input_sha256'].items())
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    base=json.loads((BASE/'flow_latest_trace.json').read_text());assert [s['frame'] for s in base]==list(range(cfg['start'],cfg['end']+1))
    records={r['frame']:r for r in json.loads(paths[5].read_text()) if r['scene']==cfg['scene'] and cfg['start']<=r['frame']<=cfg['end']}
    K=np.loadtxt(DATA/'intrinsic/intrinsic_depth.txt')[:3,:3];times=timestamp_map()
    loops={a:Recovery(a,cfg) for a in cfg['arms'] if a!='existing_flow'};traces={a:[] for a in loops}
    timings={a:0. for a in loops};feature_time=0.;features={};feature_audit={}
    for state in base:
        f=state['frame']
        if f in records:
            start=time.perf_counter();_,depth,pose=load_frame(f);color=cv2.imread(str(DATA/f'color/{f}.jpg'))
            for j,d in enumerate(records[f]['detections']):
                if d['label']!='person' or 'world_aabb_lo' not in d or np.any(np.asarray(d['world_aabb_hi'])<=d['world_aabb_lo']):continue
                source=f'{f}:{j}';features[source],count=descriptor(d,color,depth,pose,K,cfg)
                feature_audit[source]={'pixels':count,'valid':features[source] is not None}
            feature_time+=time.perf_counter()-start
        for arm,loop in loops.items():
            start=time.perf_counter();rows=loop.step(f,times[f],state['outputs'],features)
            timings[arm]+=time.perf_counter()-start
            # Equality of every field except identity, at every raw frame.
            for new,old in zip(rows,state['outputs']):
                assert {k:v for k,v in new.items() if k not in ['track_id','base_track_id']}=={k:v for k,v in old.items() if k!='track_id'}
            assert len(rows)==len(state['outputs'])
            traces[arm].append({'frame':f,'outputs':rows})
    # Save inference before reading any evaluation identity labels.
    for arm,trace in traces.items():(OUT/f'{arm}_trace.json').write_text(json.dumps(trace,allow_nan=False)+'\n')
    (OUT/'events.json').write_text(json.dumps({a:l.events for a,l in loops.items()},indent=2)+'\n')
    (OUT/'pairs.json').write_text(json.dumps({a:l.pair_audit for a,l in loops.items()},indent=2)+'\n')
    reference=json.loads((OLD/'annotations.json').read_text())['frames']
    results={};details={}
    for arm,trace in {'existing_flow':base,**traces}.items():
        result,truth=evaluate(trace,records,reference);results[arm]=result
        events=[]
        for event in loops[arm].events if arm in loops else []:
            if event['event']!='recovered':continue
            current=truth.get(event['new_source']);old={truth[s] for s in event['old_sources'] if truth.get(s) is not None}
            verdict='unknown' if current is None or not old else 'correct' if old=={current} else 'wrong'
            events.append({**event,'reference_identity':current,'old_reference_identities':sorted(old),'verdict':verdict})
        result['recovery_counts']={key:sum(e['verdict']==key for e in events) for key in ['correct','wrong','unknown']}
        details[arm]=events
    b=results['existing_flow'];a=results['appearance_motion']
    gate=a['continuity']['switches']<b['continuity']['switches'] and a['purity']['tracks_containing_multiple_people']==0 and a['recovery_counts']['wrong']==0 and a['recovery_counts']['correct']>0
    result={'arms':results,'recovery_details':details,'gate_pass':bool(gate),'features':feature_audit,
            'all_frame_geometry_scores_visibility_exactly_unchanged':True,
            'shared_detection_projection_metrics':json.loads((BASE/'results.json').read_text())['summaries']['flow_latest'],
            'association_seconds':timings,'descriptor_seconds_including_keyframe_reads':feature_time,
            'new_model_forwards':0,'training_runs':0,'gpu_runs':0,'formal_mot_or_3d_ap':False}
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (OUT/'audit.json').write_text(json.dumps({'input_sha256':hashes,'frozen_inputs_unchanged':all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items()),'baseline_input_hashes_verified':True},indent=2)+'\n')
    print(json.dumps({'gate_pass':gate,'arms':{k:{'id_changes':v['continuity']['switches'],'mixed_tracks':v['purity']['tracks_containing_multiple_people'],'recovery':v['recovery_counts']} for k,v in results.items()},'seconds':timings,'descriptor_seconds':feature_time},indent=2))


if __name__=='__main__':main()
