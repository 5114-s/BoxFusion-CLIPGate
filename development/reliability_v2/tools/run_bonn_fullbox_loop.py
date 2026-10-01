#!/usr/bin/env python3
"""Causal full-box lifecycle replay; frozen development protocol, no model forward.

Geometry is never fitted to evaluation polygons. The two flow geometry arms
share exactly the same association and image/depth evidence. Native/static
outputs are reused from the matching, actual CUDA backend replay.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from run_bonn_surface_track_probe import advance, lift, load_frame, sample_depth, timestamp_map
from run_bonn_crowd_ablation import overlap, match_frame, summarize, identity_diagnostic

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/bonn_fullbox_loop_20260914'
OLD = ROOT/'reports/bonn_crowd_native_dynamic_20260914'
DATA = ROOT/'data_bonn/scene0002_01/frames'


def box_row(d, track_id):
    lo, hi = np.asarray(d['world_aabb_lo']), np.asarray(d['world_aabb_hi'])
    return dict(label='person', score=d['score'], center=((lo+hi)/2).tolist(),
                corners=[[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])],
                source=d['source'], lineage=[d['source']], track_id=track_id)


def translate(row, delta):
    """Move the entire box; preserve its shape and orientation exactly."""
    return {**row, 'center':(np.asarray(row['center'])+delta).tolist(),
            'corners':(np.asarray(row['corners'])+delta).tolist()}


def projection(row, K, pose):
    # Same all-corners/near-plane and clipped-FOV policy as the native replay.
    pc = (np.asarray(row['corners'])-pose[:3,3])@pose[:3,:3]
    if not np.all(pc[:,2] > .05):
        return None
    uv = pc[:,:2]/pc[:,2:]@K[:2,:2].T+K[:2,2]
    lo, hi = uv.min(0), uv.max(0)
    if hi[0]<=0 or hi[1]<=0 or lo[0]>=640 or lo[1]>=480:
        return None
    return [float(max(0,lo[0])),float(max(0,lo[1])),float(min(640,hi[0])),float(min(480,hi[1]))]


def guarded_geometry(previous, observed, d, npoints, cfg):
    """Conservative AABB trial, distinct from the original oriented PFO."""
    old, new = np.asarray(previous['corners']), np.asarray(observed['corners'])
    ratio = np.ptp(new,axis=0)/np.maximum(np.ptp(old,axis=0),1e-9)
    x1,y1,x2,y2 = d['box2d']; b = cfg['geometry_image_border_px']
    checks = {
        'unclipped':x1>b and y1>b and x2<640-b and y2<480-b,
        'seed_support':npoints>=cfg['geometry_min_seed_points'],
        'center_close':np.linalg.norm(np.asarray(observed['center'])-previous['center'])<=cfg['geometry_center_gate_m'],
        'shape_close':bool(np.all((ratio>=cfg['geometry_dimension_ratio_bounds'][0]) &
                                 (ratio<=cfg['geometry_dimension_ratio_bounds'][1])))}
    w = cfg['geometry_blend'] if all(checks.values()) else 0.
    # Corner ordering is identical for these cached AABBs. No rotation averaging.
    result = {**observed, 'center':((1-w)*np.asarray(previous['center'])+w*np.asarray(observed['center'])).tolist(),
              'corners':((1-w)*old+w*new).tolist()}
    return result, {'accepted':bool(w), 'checks':{k:bool(v) for k,v in checks.items()}, 'dimension_ratio':ratio.tolist()}


def appearance(color, points):
    if not len(points):
        return None
    xy = np.rint(points).astype(int)
    hsv = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)[xy[:,1],xy[:,0]]
    hist,_ = np.histogramdd(hsv,bins=(8,4,4),range=((0,180),(0,256),(0,256)))
    return (hist.ravel()/hist.sum()).astype(float)


def hellinger(a,b):
    return float(np.sqrt(max(0.,1.-np.sqrt(a*b).sum()))) if a is not None and b is not None else 1.


def seed_detection(d, gray, color, depth, pose, K, cfg):
    h,w=depth.shape; yy,xx=np.indices((h,w)); x1,y1,x2,y2=d['box2d']
    mask=(xx>=x1)&(xx<=x2)&(yy>=y1)&(yy<=y2)&(depth>.05)&(depth<12)
    y,x=np.nonzero(mask); world=lift(np.c_[x,y],depth[y,x],K,pose)
    inside=np.all((world>=d['world_aabb_lo'])&(world<=d['world_aabb_hi']),axis=1)
    mask[:]=False; mask[y[inside],x[inside]]=True
    mask=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8))
    points=cv2.goodFeaturesToTrack(gray,maxCorners=cfg['max_corners'],qualityLevel=cfg['quality_level'],
                                   minDistance=cfg['min_distance_px'],mask=mask)
    points=np.empty((0,2),np.float32) if points is None else points.reshape(-1,2)
    z,valid=sample_depth(depth,points);points=points[valid]; pw=lift(points,z[valid],K,pose)
    return {'uv':points,'world':pw,'anchor':pw.mean(0) if len(pw) else np.zeros(3),
            'ids':np.arange(len(pw)),'active':len(pw)>=cfg['min_points']}, appearance(color,points)


def assign(cost, margin=None):
    """Maximum valid-cardinality one-to-one assignment; optional ambiguity veto."""
    cost=np.asarray(cost,dtype=float).copy()
    if not cost.size:
        return []
    if margin is not None:
        valid=np.isfinite(cost); keep=valid.copy()
        for i,j in zip(*np.nonzero(valid)):
            row=np.delete(cost[i],j); col=np.delete(cost[:,j],i)
            if (len(row) and row.min()<cost[i,j]+margin) or (len(col) and col.min()<cost[i,j]+margin):
                keep[i,j]=False
        cost[~keep]=np.inf
    finite=np.isfinite(cost)
    if not finite.any():
        return []
    penalty=(min(cost.shape)+1)*(float(cost[finite].max())+1)
    return [(int(i),int(j)) for i,j in zip(*linear_sum_assignment(np.where(finite,cost,penalty))) if finite[i,j]]


class BoxLoop:
    def __init__(self, mode, cfg, flow_cfg, K):
        self.mode,self.cfg,self.fc,self.K=mode,cfg,flow_cfg,K
        self.tracks={};self.next_id=0;self.events=[];self.last_time=None

    def emit(self, tr, frame, now):
        if tr['last_detection_frame']==frame:
            return True
        if self.mode=='flow':
            return bool(tr['flow']['active'])
        return now-tr['last_detection_time']<=self.cfg['cv_max_detection_age_seconds']

    def step(self, frame, now, previous_gray, gray, color, depth, pose, detections):
        for k,tr in self.tracks.items():
            active_before=tr['flow']['active']
            bootstrap=frame-tr['last_detection_frame']<=self.cfg['cv_bootstrap_frames']
            if self.mode=='flow' or bootstrap:
                old=tr['flow']['anchor'].copy()
                advance(tr['flow'],previous_gray,gray,depth,pose,self.K,self.fc)
                if tr['flow']['active']:
                    delta=tr['flow']['anchor']-old;tr['last_evidence']=now
                    tr['velocity']=(tr['flow']['anchor']-tr['seed_anchor'])/(now-tr['last_detection_time'])
                elif self.mode=='cv' and self.emit(tr,frame,now):
                    delta=tr['velocity']*(now-self.last_time)
                else:
                    delta=np.zeros(3)
                if active_before and not tr['flow']['active']:
                    self.events.append({'frame':frame,'track':k,'event':'flow_support_lost'})
            else:
                delta=tr['velocity']*(now-self.last_time) if self.emit(tr,frame,now) else np.zeros(3)
            for name in tr['boxes']:
                tr['boxes'][name]=translate(tr['boxes'][name],delta)
        seeds=[seed_detection(d,gray,color,depth,pose,self.K,self.fc) for d in detections]
        active=[k for k,t in self.tracks.items() if self.emit(t,frame,now)]
        cost=np.full((len(active),len(detections)),np.inf)
        for i,k in enumerate(active):
            tr=self.tracks[k]; row=tr['boxes']['latest']
            for j,d in enumerate(detections):
                c=(np.asarray(d['world_aabb_lo'])+d['world_aabb_hi'])/2
                dist=np.linalg.norm(c-row['center'])
                if dist>self.cfg['active_distance_gate_m']:
                    continue
                if self.mode=='flow':
                    uv=tr['flow']['uv']; pw=tr['flow']['world']; x1,y1,x2,y2=d['box2d']
                    m=self.cfg['candidate_depth_margin_m']
                    hit=(uv[:,0]>=x1)&(uv[:,0]<=x2)&(uv[:,1]>=y1)&(uv[:,1]<=y2)
                    hit &= np.all((pw>=np.asarray(d['world_aabb_lo'])-m)&(pw<=np.asarray(d['world_aabb_hi'])+m),axis=1)
                    fraction=float(hit.mean()) if len(hit) else 0.
                    if fraction>=self.cfg['flow_min_candidate_point_fraction']:
                        cost[i,j]=1.-fraction+.1*dist/self.cfg['active_distance_gate_m']
                else:
                    projected=projection(row,self.K,pose)
                    iou=overlap(projected,d['box2d']) if projected is not None else 0.
                    if iou>=self.cfg['cv_min_projected_iou']:
                        cost[i,j]=1.-iou+.1*dist/self.cfg['active_distance_gate_m']
        matches={j:(active[i],'active_match') for i,j in assign(cost)}
        # Retained memory is not automatically output or treated as physical absence.
        inactive=[k for k,t in self.tracks.items() if k not in active and now-t['last_evidence']<=self.cfg['inactive_memory_seconds']]
        left=[j for j in range(len(detections)) if j not in matches]
        cost=np.full((len(inactive),len(left)),np.inf)
        for i,k in enumerate(inactive):
            tr=self.tracks[k]
            for jj,j in enumerate(left):
                d=detections[j]; c=(np.asarray(d['world_aabb_lo'])+d['world_aabb_hi'])/2
                dist=np.linalg.norm(c-tr['boxes']['latest']['center']); h=hellinger(tr['appearance'],seeds[j][1])
                if dist<=self.cfg['inactive_distance_gate_m'] and h<=self.cfg['inactive_appearance_hellinger_max']:
                    cost[i,jj]=dist/self.cfg['inactive_distance_gate_m']+h/self.cfg['inactive_appearance_hellinger_max']
        for i,jj in assign(cost,self.cfg['inactive_cost_margin']):
            matches[left[jj]]=(inactive[i],'memory_revalidated')
        for j,d in enumerate(detections):
            flow,hist=seeds[j]
            if j in matches:
                k,event=matches[j];tr=self.tracks[k]; observed=box_row(d,k)
                lineage=tr['boxes']['latest']['lineage']+[d['source']];observed['lineage']=lineage
                if self.mode=='flow':
                    if event=='memory_revalidated':
                        guarded=copy.deepcopy(observed);audit={'reset_on_revalidation':True}
                    else:
                        guarded,audit=guarded_geometry(tr['boxes']['guarded'],observed,d,len(flow['uv']),self.cfg)
                    tr['boxes']['guarded']=guarded
                else:
                    audit={}
                tr['boxes']['latest']=observed
            else:
                k=self.next_id;self.next_id+=1;event='birth';audit={}
                row=box_row(d,k);tr={'boxes':{'latest':row},'velocity':np.zeros(3)}
                if self.mode=='flow':tr['boxes']['guarded']=copy.deepcopy(row)
                self.tracks[k]=tr
            tr.update(flow=flow,seed_anchor=flow['anchor'].copy(),appearance=hist,last_detection_frame=frame,
                      last_detection_time=now,last_evidence=now)
            self.events.append({'frame':frame,'track':k,'event':event,'source':d['source'],'seed_points':len(flow['uv']),
                                'geometry':audit})
        # Bound memory. Do not evict an emitting track just because detection is absent.
        expired=[k for k,t in self.tracks.items() if not self.emit(t,frame,now) and now-t['last_evidence']>self.cfg['inactive_memory_seconds']]
        for k in expired:del self.tracks[k]
        while len(self.tracks)>self.cfg['max_tracks']:
            del self.tracks[min(self.tracks,key=lambda k:self.tracks[k]['last_evidence'])]
        names=['latest','guarded'] if self.mode=='flow' else ['latest']
        outputs={n:[{**t['boxes'][n],'observed_now':t['last_detection_frame']==frame,
                      'state':'observed' if t['last_detection_frame']==frame else 'tracked' if self.mode=='flow' else 'predicted',
                      'support_points':len(t['flow']['uv']) if t['flow']['active'] else 0}
                     for t in self.tracks.values() if self.emit(t,frame,now)] for n in names}
        self.last_time=now
        return outputs


def static_ledger(a,b):
    missing=extra=changed=0; maximum=0.
    for x,y in zip(a,b):
        assert x['frame']==y['frame']
        maps=[{tuple(r['lineage']):r for r in state['outputs'] if r['label']!='person'} for state in (x,y)]
        missing+=len(maps[1].keys()-maps[0].keys());extra+=len(maps[0].keys()-maps[1].keys())
        for key in maps[0].keys() & maps[1].keys():
            diff=float(np.max(np.abs(np.asarray(maps[0][key]['corners'])-maps[1][key]['corners'])))
            maximum=max(maximum,diff);changed+=int(diff>1e-4)
    return dict(missing=missing,extra=extra,changed_over_0_1mm=changed,max_corner_difference_m=maximum)


def identity_purity(frames):
    counts={};by_person={}
    for frame in frames:
        for match in frame['matches']:
            person,track=match['identity'],match['track_id']
            if person is None or track is None:continue
            counts.setdefault(track,{})[person]=counts.get(track,{}).get(person,0)+1
            by_person.setdefault(person,set()).add(track)
    total=sum(sum(v.values()) for v in counts.values())
    return {'matched_detections_with_known_identity':total,
            'weighted_track_purity':sum(max(v.values()) for v in counts.values())/max(total,1),
            'tracks_containing_multiple_people':sum(len(v)>1 for v in counts.values()),
            'track_identity_counts':counts,'distinct_track_ids_per_person':{k:len(v) for k,v in by_person.items()}}


def main():
    cv2.setNumThreads(1);cv2.setRNGSeed(0)
    cfg=json.loads((OUT/'protocol.json').read_text())
    fc=json.loads((ROOT/'reports/bonn_surface_track_625_675_20260914/protocol.json').read_text())
    frozen=[Path(__file__), OUT/'protocol.json', ROOT/'tools/run_bonn_surface_track_probe.py',
            ROOT/'tools/run_bonn_crowd_ablation.py', ROOT/'reports/bonn_surface_track_625_675_20260914/protocol.json',
            ROOT/'reports/bonn_detect_20260914/yoloe_candidates.json',OLD/'native_trace.json',OLD/'bypass_trace.json',OLD/'annotations.json']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen}
    # Verify native replay really used the same cached candidates/config and core.
    old_hashes=json.loads((OLD/'input_sha256.json').read_text())
    old_checks={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in old_hashes.items()}
    if not all(old_checks.values()):raise RuntimeError(f'Baseline inputs changed: {old_checks}')
    times=timestamp_map(); K=np.loadtxt(DATA/'intrinsic/intrinsic_depth.txt')[:3,:3]
    records=json.loads(frozen[5].read_text()); selected={r['frame']:r for r in records if r['scene']==cfg['scene'] and cfg['start']<=r['frame']<=cfg['end']}
    frames=list(range(cfg['start'],cfg['end']+1,cfg['keyframe_gap'])); assert sorted(selected)==frames
    native=json.loads((OLD/'native_trace.json').read_text());current=json.loads((OLD/'bypass_trace.json').read_text())
    assert [s['frame'] for s in native]==frames and [s['frame'] for s in current]==frames
    static={s['frame']:[r for r in s['outputs'] if r['label']!='person'] for s in current}
    loops={name:BoxLoop(name,cfg,fc,K) for name in ['cv','flow']}; traces={a:[] for a in ['cv','flow_latest','flow_guarded']}
    runtime={k:[] for k in loops}; io_seconds=0.; previous=None; static_now=[]; excluded=[]; npeople=0
    for frame in range(cfg['start'],cfg['end']+1):
        start=time.perf_counter(); gray,depth,pose=load_frame(frame); color=cv2.imread(str(DATA/f'color/{frame}.jpg'))
        io_seconds+=time.perf_counter()-start
        dets=[]
        if frame in selected:
            static_now=static[frame]
            for j,d in enumerate(selected[frame]['detections']):
                if 'world_aabb_lo' not in d or np.any(np.asarray(d['world_aabb_hi'])<=d['world_aabb_lo']):
                    excluded.append([frame,j]);continue
                if d['label']=='person':dets.append({**d,'source':f'{frame}:{j}'})
            npeople+=len(dets)
        for name,loop in loops.items():
            start=time.perf_counter(); rows=loop.step(frame,times[frame],previous,gray,color,depth,pose,dets)
            runtime[name].append(time.perf_counter()-start)
            for geom,outputs in rows.items():
                arm=f'flow_{geom}' if name=='flow' else 'cv'
                traces[arm].append({'frame':frame,'outputs':static_now+outputs})
        previous=gray
    # Write all 251 full-box outputs before opening reference annotations.
    for arm,trace in traces.items():
        (OUT/f'{arm}_trace.json').write_text(json.dumps(trace,allow_nan=False)+'\n')
    (OUT/'events.json').write_text(json.dumps({k:l.events for k,l in loops.items()},indent=2)+'\n')
    reference=json.loads((OLD/'annotations.json').read_text())['frames']
    all_traces={'native':native,'current_only':current,**{k:[s for s in v if s['frame'] in selected] for k,v in traces.items()}}
    details={};summaries={};identities={}
    for arm,trace in all_traces.items():
        details[arm]=[]
        for state in trace:
            f=state['frame'];pose=np.loadtxt(DATA/f'pose/{f}.txt');predictions=[]
            for row in state['outputs']:
                if row['label']!='person':continue
                p=projection(row,K,pose)
                if p is not None:predictions.append({**row,'box':p})
            details[arm].append({'frame':f,**match_frame(predictions,reference[str(f)]),'projected_predictions':predictions})
        summaries[arm]=summarize(details[arm]);identities[arm]=identity_diagnostic(details[arm])
    # ID continuity conditional on matched raw detections, avoiding AABB projection inflation.
    raw_id={};raw_purity={}
    for name,loop in loops.items():
        mapping={e['source']:e['track'] for e in loop.events if 'source' in e};diag=[]
        for frame in frames:
            ps=[{'box':d['box2d'],'track_id':mapping[f'{frame}:{j}']} for j,d in enumerate(selected[frame]['detections']) if f'{frame}:{j}' in mapping]
            diag.append({'frame':frame,**match_frame(ps,reference[str(frame)])})
        raw_id[name]=identity_diagnostic(diag)
        raw_purity[name]=identity_purity(diag)
    a=summaries; flow=a['flow_latest'];cv=a['cv'];guard=a['flow_guarded']
    motion_gate=flow['tp']>=max(a['current_only']['tp'],cv['tp']) and flow['fp']<=cv['fp'] and (flow['tp']>cv['tp'] or flow['fp']<cv['fp'])
    geometry_gate=guard['tp']>=flow['tp'] and guard['fp']<=flow['fp'] and (guard['tp']>flow['tp'] or guard['fp']<flow['fp'])
    ledger={k:static_ledger(v,current) for k,v in all_traces.items()}
    exact={k:all([r for r in x['outputs'] if r['label']!='person']==[r for r in y['outputs'] if r['label']!='person'] for x,y in zip(v,current)) for k,v in all_traces.items()}
    artifact=dict(summaries=summaries,per_frame=details,identity_on_projected_boxes=identities,identity_on_raw_detections=raw_id,
                  identity_purity_on_raw_detections=raw_purity,
                  motion_gate=bool(motion_gate),geometry_gate=bool(geometry_gate),nonperson_ledger_vs_control=ledger,exact_nonperson_rows_vs_control=exact,
                  person_candidates=npeople,excluded_candidates=excluded,total_raw_frames=cfg['end']-cfg['start']+1,
                  timing={k:{'total_seconds':sum(v),'mean_ms':np.mean(v)*1000,'p95_ms':np.percentile(v,95)*1000,'max_ms':max(v)*1000} for k,v in runtime.items()},
                  frame_io_seconds=io_seconds,elapsed_sequence_seconds=times[cfg['end']]-times[cfg['start']],
                  model_forward_runs=0,training_runs=0,new_gpu_runs=0,reused_backend_stats=json.loads((OLD/'results.json').read_text())['backend_stats'],
                  formal_3d_ap=None,groundtruth_file_meaning='camera pose trajectory; no dynamic person boxes or identities present locally')
    (OUT/'results.json').write_text(json.dumps(artifact,indent=2,allow_nan=False)+'\n')
    (OUT/'audit.json').write_text(json.dumps({'input_sha256':hashes,'baseline_input_checks':old_checks,
        'frozen_inputs_unchanged':all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items()),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    print(json.dumps({k:artifact[k] for k in ['summaries','motion_gate','geometry_gate','identity_on_raw_detections','timing','exact_nonperson_rows_vs_control']},indent=2))


if __name__=='__main__':main()
