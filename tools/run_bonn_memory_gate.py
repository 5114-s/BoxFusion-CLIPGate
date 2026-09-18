#!/usr/bin/env python3
"""Mature identity admission and causal appearance rejection, frozen two windows."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from run_bonn_identity_recovery import Recovery, descriptor, evaluate
from run_bonn_fullbox_loop import BoxLoop, assign, hellinger, identity_purity
from run_bonn_surface_track_probe import load_frame, timestamp_map

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/bonn_memory_gate_20260914'
DATA=ROOT/'data_bonn/scene0002_01/frames'


class GuardedRecovery:
    def __init__(self,mode,cfg,rule):
        self.mode,self.cfg,self.rule=mode,cfg,rule
        self.mapping={};self.profiles={};self.next_id=0
        self.events=[];self.pairs=[];self.calibration=[];self.calibration_audit=[]

    def appearance_limit(self):
        limit=self.cfg['appearance_hellinger_max']
        if len(self.calibration)>=self.rule['calibration_min_pairs']:
            limit=max(limit,float(np.quantile(self.calibration,self.rule['calibration_quantile']))+self.rule['calibration_slack'])
        return min(1.,limit)

    def step(self,frame,now,rows,features):
        limit=self.appearance_limit();calibration_count=len(self.calibration)
        people=[r for r in rows if r['label']=='person']
        known={r['track_id'] for r in people if r['track_id'] in self.mapping and self.mapping[r['track_id']] in self.profiles}
        active={self.mapping[k] for k in known};fresh=[r for r in people if r['track_id'] not in known]
        dormant=[k for k,p in self.profiles.items() if k not in active and 0<now-p['last_time']<=self.cfg['memory_seconds']]
        cost=np.full((len(dormant),len(fresh)),np.inf);pair={}
        for i,k in enumerate(dormant):
            p=self.profiles[k];age=now-p['last_time']
            reach=min(self.cfg['maximum_reach_m'],self.cfg['reach_noise_m']+self.cfg['maximum_speed_m_s']*age)
            velocity=np.zeros(3)
            if len(p['history'])>=2:
                ta,ca=p['history'][0];tb,cb=p['history'][-1]
                velocity=(np.asarray(cb)-ca)/max(tb-ta,1e-6)
                velocity*=min(1.,self.cfg['maximum_speed_m_s']/max(np.linalg.norm(velocity),1e-9))
            predicted=np.asarray(p['center'])+velocity*age
            for j,row in enumerate(fresh):
                distance=float(np.linalg.norm(np.asarray(row['center'])-p['center']))
                residual=float(np.linalg.norm(np.asarray(row['center'])-predicted)/reach)
                h=min((hellinger(proto,features.get(row['source'])) for _,proto in p['bank']),default=1.)
                mature=p['consecutive']>=self.rule['mature_consecutive_frames']
                supported=row.get('support_points',0)>=self.rule['min_support_points']
                eligible=mature and supported and distance<=reach
                score=residual
                if self.mode=='admission_reject':
                    w=self.cfg['appearance_weight'];score=w*h/limit+(1-w)*residual
                    eligible=eligible and bool(p['bank']) and features.get(row['source']) is not None and h<=limit and score<self.rule['null_cost']
                if eligible:cost[i,j]=score
                detail=dict(frame=frame,identity=k,new_source=row['source'],old_sources=list(p['sources']),
                            age_seconds=age,distance_m=distance,reach_m=reach,appearance_distance=h,appearance_limit=limit,
                            past_calibration_pairs=calibration_count,memory_mature=bool(mature),candidate_supported=bool(supported),
                            eligible=bool(eligible),cost=float(score))
                pair[i,j]=detail;self.pairs.append(detail)
        margin=self.cfg['ambiguity_margin'] if self.mode=='admission_reject' else None
        matches={j:i for i,j in assign(cost,margin)}
        for j,row in enumerate(fresh):
            if j in matches:
                i=matches[j];k=dormant[i];p=self.profiles[k]
                self.events.append({**pair[i,j],'event':'recovered','base_track_id':row['track_id']})
                p['history']=[];p['consecutive']=0;p['last_frame']=-1
            else:
                k=self.next_id;self.next_id+=1
                self.profiles[k]={'history':[],'consecutive':0,'last_frame':-1,'sources':[],'bank':[]}
                self.events.append(dict(frame=frame,identity=k,new_source=row['source'],base_track_id=row['track_id'],event='new_identity'))
            self.mapping[row['track_id']]=k
        outputs=[]
        # Calibration occurs only now, after current-frame identity decisions.
        for row in rows:
            if row['label']!='person':outputs.append(copy.deepcopy(row));continue
            k=self.mapping[row['track_id']];p=self.profiles[k]
            previous_mature=p['consecutive']>=self.rule['mature_consecutive_frames']
            continuing=row['track_id'] in known and p['last_frame']==frame-1
            feature=features.get(row['source'])
            if row['observed_now'] and row['source'] not in p['sources']:
                if continuing and previous_mature and feature is not None and p['bank']:
                    h=min(hellinger(proto,feature) for _,proto in p['bank'])
                    self.calibration.append(h);self.calibration=self.calibration[-self.rule['calibration_buffer']:]
                    self.calibration_audit.append(dict(frame=frame,source=row['source'],past_sources=list(p['sources']),distance=h))
                p['sources'].append(row['source'])
                if feature is not None and row.get('support_points',0)>=self.rule['min_support_points']:
                    p['bank'].append((row['source'],feature.copy()))
                    if len(p['bank'])>self.cfg['max_prototypes']:p['bank']=[p['bank'][0]]+p['bank'][-(self.cfg['max_prototypes']-1):]
            if row.get('support_points',0)>=self.rule['min_support_points']:
                p['consecutive']=p['consecutive']+1 if p['last_frame']==frame-1 else 1
            else:p['consecutive']=0
            p.update(center=list(row['center']),last_time=now,last_frame=frame)
            p['history'].append((now,list(row['center'])));p['history']=p['history'][-self.cfg['velocity_history_frames']:]
            outputs.append({**copy.deepcopy(row),'base_track_id':row['track_id'],'track_id':k})
        used={r['track_id'] for r in outputs if r['label']=='person'};assert len(used)==len(people)
        for k in [k for k,p in self.profiles.items() if k not in used and now-p['last_time']>self.cfg['memory_seconds']]:del self.profiles[k]
        while len(self.profiles)>self.cfg['max_profiles']:
            dormant=[k for k in self.profiles if k not in used]
            if not dormant:raise RuntimeError('Active identity capacity exceeded')
            del self.profiles[min(dormant,key=lambda k:self.profiles[k]['last_time'])]
        return outputs


def infer():
    cv2.setNumThreads(1);cv2.setRNGSeed(0)
    rule=json.loads((OUT/'protocol.json').read_text())
    cfg=json.loads((ROOT/'reports/bonn_identity_recovery_20260914/protocol.json').read_text())
    flowcfg=json.loads((ROOT/'reports/bonn_fullbox_loop_20260914/protocol.json').read_text())
    fcfg=json.loads((ROOT/'reports/bonn_surface_track_625_675_20260914/protocol.json').read_text())
    paths=[Path(__file__),OUT/'protocol.json',ROOT/'reports/bonn_identity_recovery_20260914/protocol.json',
           ROOT/'reports/bonn_fullbox_loop_20260914/protocol.json',ROOT/'reports/bonn_surface_track_625_675_20260914/protocol.json',
           ROOT/'reports/bonn_detect_20260914/yoloe_candidates.json',ROOT/'reports/bonn_fullbox_loop_20260914/flow_latest_trace.json',
           ROOT/'tools/run_bonn_identity_recovery.py',ROOT/'tools/run_bonn_fullbox_loop.py',ROOT/'tools/run_bonn_surface_track_probe.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    cached={r['frame']:r for r in json.loads(paths[5].read_text()) if r['scene']==rule['scene']}
    times=timestamp_map();K=np.loadtxt(DATA/'intrinsic/intrinsic_depth.txt')[:3,:3]
    timings={}
    for window in ['development_window','temporal_extension']:
        start,end=rule[window];out=OUT/window;out.mkdir(exist_ok=True)
        if window=='development_window':base=json.loads(paths[6].read_text())
        else:
            base=[];loop=BoxLoop('flow',flowcfg,fcfg,K);previous=None;started=time.perf_counter()
            for frame in range(start,end+1):
                gray,depth,pose=load_frame(frame);color=cv2.imread(str(DATA/f'color/{frame}.jpg'));dets=[]
                if frame in cached:
                    dets=[{**d,'source':f'{frame}:{j}'} for j,d in enumerate(cached[frame]['detections'])
                          if d['label']=='person' and 'world_aabb_lo' in d and np.all(np.asarray(d['world_aabb_hi'])>d['world_aabb_lo'])]
                rows=loop.step(frame,times[frame],previous,gray,color,depth,pose,dets)['latest']
                base.append({'frame':frame,'outputs':rows});previous=gray
            timings['extension_flow_seconds_including_reads']=time.perf_counter()-started
        assert [s['frame'] for s in base]==list(range(start,end+1))
        loops={a:Recovery(a,cfg) if a in ['motion_only','appearance_motion'] else GuardedRecovery(a,cfg,rule)
               for a in rule['arms'] if a!='existing_flow'}
        traces={a:[] for a in loops};features={};feature_audit={};started=time.perf_counter()
        for state in base:
            f=state['frame']
            if f in cached:
                _,depth,pose=load_frame(f);color=cv2.imread(str(DATA/f'color/{f}.jpg'))
                for j,d in enumerate(cached[f]['detections']):
                    if d['label']!='person' or 'world_aabb_lo' not in d or np.any(np.asarray(d['world_aabb_hi'])<=d['world_aabb_lo']):continue
                    source=f'{f}:{j}';features[source],n=descriptor(d,color,depth,pose,K,cfg);feature_audit[source]=n
            for arm,loop in loops.items():
                rows=loop.step(f,times[f],state['outputs'],features)
                assert len(rows)==len(state['outputs'])
                for new,old in zip(rows,state['outputs']):
                    assert {k:v for k,v in new.items() if k not in ['track_id','base_track_id']}=={k:v for k,v in old.items() if k!='track_id'}
                traces[arm].append({'frame':f,'outputs':rows})
        timings[window+'_recovery_all_arms_seconds_including_reads']=time.perf_counter()-started
        traces['existing_flow']=base
        for a,t in traces.items():(out/f'{a}_trace.json').write_text(json.dumps(t,allow_nan=False)+'\n')
        (out/'events.json').write_text(json.dumps({a:l.events for a,l in loops.items()},indent=2)+'\n')
        (out/'pairs.json').write_text(json.dumps({a:l.pairs if isinstance(l,GuardedRecovery) else l.pair_audit for a,l in loops.items()},indent=2)+'\n')
        (out/'calibration.json').write_text(json.dumps({a:l.calibration_audit for a,l in loops.items() if isinstance(l,GuardedRecovery)},indent=2)+'\n')
        (out/'features.json').write_text(json.dumps(feature_audit,indent=2)+'\n')
        print(window,'complete',len(base),'frames',len(feature_audit),'person candidates',flush=True)
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    (OUT/'inference_audit.json').write_text(json.dumps(dict(input_sha256=hashes,frozen_inputs_unchanged=True,
        all_geometry_scores_visibility_unchanged=True,timing=timings,training_runs=0,new_model_forwards=0,gpu_runs=0),indent=2)+'\n')


def score():
    rule=json.loads((OUT/'protocol.json').read_text());results={}
    records={r['frame']:r for r in json.loads((ROOT/'reports/bonn_detect_20260914/yoloe_candidates.json').read_text()) if r['scene']==rule['scene']}
    oldref=json.loads((ROOT/'reports/bonn_crowd_native_dynamic_20260914/annotations.json').read_text())['frames']
    devtrace=json.loads((OUT/'development_window/existing_flow_trace.json').read_text())
    _,truth=evaluate(devtrace,{f:r for f,r in records.items() if 500<=f<=750},oldref)
    truth={s:p for s,p in truth.items() if p is not None}
    truth.update(json.loads((ROOT/'reports/bonn_recovery_review_20260914/visual_review.json').read_text())['source_identity_additions'])
    extension=json.loads((OUT/'extension_visual_review.json').read_text())['source_identities']
    for window,labels in [('development_window',truth),('temporal_extension',extension)]:
        folder=OUT/window;events=json.loads((folder/'events.json').read_text());results[window]={}
        for arm in rule['arms']:
            trace=json.loads((folder/f'{arm}_trace.json').read_text())
            ids={r['source']:r['track_id'] for state in trace for r in state['outputs'] if r['label']=='person' and r['observed_now']}
            assert set(ids)==set(labels)
            purity=identity_purity([{'matches':[{'track_id':ids[s],'identity':p} for s,p in labels.items() if p is not None]}])
            evaluated=[]
            for e in events.get(arm,[]):
                if e['event']!='recovered':continue
                prior={labels[s] for s in e['old_sources'] if labels.get(s) is not None};current=labels.get(e['new_source'])
                verdict='unknown' if current is None or not prior else 'correct' if prior=={current} else 'wrong'
                evaluated.append({**e,'prior_visual_identities':sorted(prior),'current_visual_identity':current,'verdict':verdict})
            count={k:sum(e['verdict']==k for e in evaluated) for k in ['correct','wrong','unknown']}
            results[window][arm]=dict(counts=count,purity=purity,events=evaluated,unknown_candidate_labels=sum(p is None for p in labels.values()))
        a=results[window]['admission_reject'];c=a['counts']
        if c['wrong'] or a['purity']['tracks_containing_multiple_people']:gate='fail'
        elif window=='development_window':gate='pass' if c['correct']>results[window]['appearance_motion']['counts']['correct'] else 'fail'
        else:gate='pass' if c['correct']>=2 else 'inconclusive'
        results[window]['gate']=gate
    (OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({w:{a:({'counts':v['counts'],'mixed_tracks':v['purity']['tracks_containing_multiple_people']}) if a!='gate' else v for a,v in arms.items()} for w,arms in results.items()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['infer','score'],required=True)
    infer() if parser.parse_args().phase=='infer' else score()
