#!/usr/bin/env python3
"""Actual BoxFusion backend replay on frozen Bonn YOLOE candidates.

No detector forward. The native arm calls the repository's spatial association,
correspondence association and CUDA PFO, preserving demo.py's PFO dispatch.
The bypass arm routes non-person proposals through the SAME backend and emits
current person geometry separately. A non-person-only control checks isolation.
References are loaded ONLY after all three inference arms finish.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
from boxfusion.boxes import GeneralInstance3DBoxes
from boxfusion.instances import Instances3D
from boxfusion.box_manager import BoxManager
from boxfusion.box_fusion import BoxFusion, GPU_MODE

SOURCE = ROOT/'reports/bonn_detect_20260914/yoloe_candidates.json'
SCENE = 'scene0001_01'
FRAMES = ROOT/f'data_bonn/{SCENE}/frames'


def iou2(a, b):
    a, b = np.asarray(a), np.asarray(b)
    inter = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    va, vb = np.maximum(0, a[2:]-a[:2]).prod(), np.maximum(0, b[2:]-b[:2]).prod()
    return float(inter/max(va+vb-inter, 1e-12))


class NativeBackend:
    def __init__(self, cfg, K):
        self.cfg, self.K = cfg, K
        self.manager = BoxManager(cfg)
        self.fuser = BoxFusion(cfg)
        self.live = self.history = self.poses = None
        self.kf_poses = {}
        self.sources = []
        self.spatial_calls = self.correspondence_calls = self.pfo_calls = 0

    def make_instances(self, records, frame, pose):
        n = len(records)
        start = len(self.sources)
        self.sources.extend(r['source'] for r in records)
        lo = np.asarray([r['world_aabb_lo'] for r in records], np.float32)
        hi = np.asarray([r['world_aabb_hi'] for r in records], np.float32)
        ins = Instances3D((480, 640))
        ins.pred_boxes_3d = GeneralInstance3DBoxes(
            np.concatenate([(lo+hi)/2, hi-lo], axis=1), np.tile(np.eye(3), (n,1,1)))
        ins.pred_boxes = torch.tensor([r['box2d'] for r in records], dtype=torch.float32)
        ins.scores = torch.tensor([r['score'] for r in records], dtype=torch.float32)
        ins.categories = np.asarray([r['label'] for r in records])
        ins.init_id = torch.arange(start, start+n)
        ins.frame_id = torch.full((n,), frame, dtype=torch.long)
        ins.valid_num = torch.zeros(n, dtype=torch.int64)
        ins.cam_pose = torch.from_numpy(np.repeat(pose[None], n, axis=0)).float()
        ins.project_3d_boxes(torch.from_numpy(self.K).float(), H=480, W=640)
        # Verify parameter/corner convention once for EVERY injected batch.
        corners = ins.pred_boxes_3d.corners.numpy()
        np.testing.assert_allclose(corners.min(1), lo, atol=1e-5)
        np.testing.assert_allclose(corners.max(1), hi, atol=1e-5)
        return ins

    def step(self, records, frame, pose):
        self.kf_poses[frame] = pose
        if not records:
            self.manager.num_record[frame] = len(self.sources)
            return self.snapshot()
        pred = self.make_instances(records, frame, pose)
        self.manager.num_record[frame] = len(self.sources)
        poses = pred.cam_pose.numpy()
        if self.live is None:
            self.live = pred
            self.history = pred
            self.poses = poses
            self.manager.init_new_predictions(len(pred), 0)
        else:
            self.manager.init_new_predictions(len(pred), len(self.history))
            previous = self.live
            count_before = len(previous)
            self.live = Instances3D.cat([previous, pred])
            self.history = Instances3D.cat([self.history, pred])
            self.poses = np.concatenate([self.poses, poses])
            self.spatial_calls += 1
            keep, success = Instances3D.spatial_association(
                self.live, self.cfg['box_fusion']['nms_threshold'],
                self.manager, self.history.cam_pose)
            new_keep = [i-count_before for i in keep if i >= count_before]
            new_success = [i-count_before for i in success if i >= count_before]
            if new_keep:
                self.correspondence_calls += 1
                self.live, self.poses, keep = Instances3D.correspondence_association(
                    self.cfg, self.manager, new_keep, new_success, pred, previous,
                    self.live, self.poses, self.history.cam_pose, frame, keep,
                    torch.from_numpy(self.K).float(), self.kf_poses,
                    threshold=self.cfg['association']['small_threshold'], H=480, W=640)
                self.manager.update(keep)
                if self.cfg['box_fusion']['check_valid']:
                    self.live = self.manager.check_valid_num(self.live, frame, 25)
                # Exactly the native demo dispatch: no extra PFO on all-NMS frames.
                self.pfo_calls += 1
                self.fuser.boxfusion(self.live, self.history, self.manager)
            else:
                self.live = self.live[keep]
                self.poses = self.poses[keep]
                self.manager.update(np.asarray(keep))
        assert len(self.live) == len(self.manager.fusion_list)
        return self.snapshot()

    def snapshot(self):
        if self.live is None:
            return []
        boxes = self.live.pred_boxes_3d
        out = []
        for i, ids in enumerate(self.manager.fusion_list):
            out.append({'label': str(self.live.categories[i]),
                        'score': float(self.live.scores[i]),
                        'center': boxes.tensor[i, :3].tolist(),
                        'corners': boxes.corners[i].tolist(),
                        'source': self.sources[int(self.live.init_id[i])],
                        'lineage': sorted(self.sources[j] for j in ids)})
        return out

    def stats(self):
        return {'spatial_calls': self.spatial_calls,
                'correspondence_calls': self.correspondence_calls,
                'pfo_calls': self.pfo_calls,
                'successful_fusion_updates': len(self.manager.already_fusion),
                'reliable_views': self.fuser.reliable_view_stats}


class CurrentPeople:
    """Person-only baseline: one-to-one identity memory, current geometry only.

    Fixed 1.2m gate reuses bonn_native's dynamic_objects.max_center_distance_m.
    No velocity, extrapolation, GT identity, or future observations.
    Unmatched memories become unobserved; not declared physically absent.
    """
    def __init__(self, distance_gate=1.2, max_tracks=256):
        self.distance_gate, self.max_tracks = distance_gate, max_tracks
        self.memory = {}
        self.next_id = 0

    def step(self, records, frame):
        ids = list(self.memory)
        centers = [(np.asarray(r['world_aabb_lo'])+r['world_aabb_hi'])/2 for r in records]
        assignments = {}
        if ids and records:
            cost = np.asarray([[np.linalg.norm(self.memory[k]['center']-c)
                                for c in centers] for k in ids])
            gated = np.where(cost <= self.distance_gate, cost, 1e6)
            for i,j in zip(*linear_sum_assignment(gated)):
                if gated[i,j] < 1e6:
                    assignments[j] = ids[i]
        for memory in self.memory.values():
            memory['state'] = 'unobserved'
        outputs = []
        for j, (r,c) in enumerate(zip(records, centers)):
            if j not in assignments:
                assignments[j] = self.next_id; self.next_id += 1
            k = assignments[j]
            self.memory[k] = {'center': c, 'last_frame': frame, 'state': 'observed'}
            lo,hi = np.asarray(r['world_aabb_lo']),np.asarray(r['world_aabb_hi'])
            corners = [[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])]
            outputs.append({'label':'person','score':r['score'],'center':c.tolist(),
                            'corners':corners,'source':r['source'],
                            'lineage':[r['source']], 'track_id':k})
        while len(self.memory) > self.max_tracks:
            oldest = min(self.memory, key=lambda k:self.memory[k]['last_frame'])
            del self.memory[oldest]
        return outputs


def project(row, K, pose):
    points = np.asarray(row['corners'])
    pc = (points-pose[:3,3]) @ pose[:3,:3]
    if not np.all(pc[:,2] > .05):
        return None  # same conservative visibility policy for both arms
    uv = pc[:,:2]/pc[:,2:3] @ K[:2,:2].T + K[:2,2]
    lo,hi = uv.min(0),uv.max(0)
    if hi[0]<=0 or hi[1]<=0 or lo[0]>=640 or lo[1]>=480:
        return None
    return [float(max(0,lo[0])),float(max(0,lo[1])),float(min(640,hi[0])),float(min(480,hi[1]))]


def evaluate(traces, K, poses):
    # Inference is finished and saved before these evaluation inputs are read.
    from run_bonn_fourarm import load_inputs
    ann, _, _, _, refs, sanity = load_inputs()
    summaries, details = {}, {}
    for arm in ('native','bypass'):
        rows = []
        for state in traces[arm]:
            fr=state['frame']
            if str(fr) not in ann['boxes']:
                continue
            people=[]
            for prediction in state['outputs']:
                if prediction['label'] != 'person':
                    continue
                box=project(prediction,K,poses[fr])
                if box is not None:
                    people.append((prediction,box))
            matched=[r for r,b in people if iou2(b,ann['boxes'][str(fr)])>=.5]
            # Score-based selection uses NO reference geometry / GT IoU.
            chosen=max((r for r,_ in people),key=lambda r:r['score'],default=None)
            error=(float(np.linalg.norm(np.asarray(chosen['center'])-refs[fr]))
                   if chosen is not None and fr in refs else None)
            rows.append({'frame':fr,'person_rows_in_view':len(people),
                         'matched_at_2d_iou50':len(matched),'duplicates':max(0,len(matched)-1),
                         'top_score_center_error_3d':error,
                         'size_filter_pass':bool(sanity.get(fr,{}).get('person_sane'))})
        def stats(subset):
            errors=[r['top_score_center_error_3d'] for r in subset if r['top_score_center_error_3d'] is not None]
            return {'frames':len(subset),'frames_with_output':len(errors),
                    'median_m':float(np.median(errors)) if errors else None,
                    'mean_m':float(np.mean(errors)) if errors else None,
                    'recall_2d_iou50_frames':sum(r['matched_at_2d_iou50']>0 for r in subset),
                    'duplicate_boxes_iou50':sum(r['duplicates'] for r in subset),
                    'person_rows_in_view_sum':sum(r['person_rows_in_view'] for r in subset)}
        summaries[arm]={'size_filtered':stats([r for r in rows if r['size_filter_pass']]),
                        'all_references':stats(rows)}
        details[arm]=rows
    # Non-person ledger relative to independent native non-person control.
    ledger={}
    for arm in ('native','bypass'):
        missing=extra=changed=0; max_shift=0.
        for state,control in zip(traces[arm],traces['static_control']):
            def keyed(records):
                return {tuple(r['lineage']):r for r in records if r['label']!='person'}
            current=keyed(state['outputs']); base=keyed(control['outputs'])
            missing+=len(base.keys()-current.keys()); extra+=len(current.keys()-base.keys())
            for key in base.keys() & current.keys():
                shift=float(np.max(np.abs(np.asarray(base[key]['corners'])-current[key]['corners'])))
                max_shift=max(max_shift,shift); changed+=int(shift>1e-4)
        ledger[arm]={'missing_lineage_rows_over_frames':missing,'extra_lineage_rows_over_frames':extra,
                     'same_lineage_geometry_changed_gt_0_1mm':changed,'max_corner_difference_m':max_shift}
    return {'summaries':summaries,'per_frame':details,'nonperson_ledger':ledger,
            'limitations':['VLM/depth proxy references; not full human 3D GT or AP.',
                           'Non-person labels are not verified static ground truth.',
                           'Single-person clip; no evaluated crossing identities.',
                           'Current-frame visibility policy differs from persistent identity memory.',
                           'Native means actual repository backend with frozen Bonn Top-K3 config; front end is shared YOLOE.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'reports/bonn_native_dynamic_20260914')
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    if not GPU_MODE or not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA PFO is required; no substitute optimizer')
    cfg=yaml.safe_load((ROOT/'config/bonn_native.yaml').read_text())
    cfg['dataset']='online'  # Constructor uses explicit Bonn intrinsics, not path-based CA-1M loading.
    K=np.loadtxt(FRAMES/'intrinsic/intrinsic_depth.txt')[:3,:3]
    records=[r for r in json.loads(SOURCE.read_text()) if r['scene']==SCENE]
    poses={r['frame']:np.loadtxt(FRAMES/f"pose/{r['frame']}.txt") for r in records}
    excluded=[]
    for r in records:
        valid=[]
        for j,d in enumerate(r['detections']):
            if 'world_aabb_lo' not in d or np.any(np.asarray(d['world_aabb_hi'])<=d['world_aabb_lo']):
                excluded.append([r['frame'],j]); continue
            valid.append({**d,'source':f"{r['frame']}:{j}"})
        r['valid']=valid
    (args.out/'config.yaml').write_text(yaml.safe_dump(cfg))
    traces, backend_stats={},{}
    for arm in ('native','bypass','static_control'):
        np.random.seed(0); torch.manual_seed(0)
        with (args.out/f'{arm}.log').open('w') as log, contextlib.redirect_stdout(log):
            backend=NativeBackend(cfg,K)
            dynamic=CurrentPeople(cfg['dynamic_objects']['max_center_distance_m'])
            trace=[]
            for r in records:
                fr=r['frame']; before=time.perf_counter()
                inputs=r['valid'] if arm=='native' else [x for x in r['valid'] if x['label']!='person']
                rows=backend.step(inputs,fr,poses[fr])
                if arm=='bypass':
                    rows+=dynamic.step([x for x in r['valid'] if x['label']=='person'],fr)
                torch.cuda.synchronize()
                trace.append({'frame':fr,'outputs':rows,'backend_seconds':time.perf_counter()-before})
            traces[arm]=trace; backend_stats[arm]=backend.stats()
            if arm=='bypass':
                backend_stats[arm]['dynamic_memory']={k:{**v,'center':v['center'].tolist()} for k,v in dynamic.memory.items()}
        (args.out/f'{arm}_trace.json').write_text(json.dumps(trace,allow_nan=False))
        print(arm,'frames',len(trace),'fusion updates',backend_stats[arm]['successful_fusion_updates'],flush=True)
    result=evaluate(traces,K,poses)
    result.update({'backend_stats':backend_stats,'excluded_degenerate_candidates':excluded,
                   'frames':len(records),'candidate_records':sum(len(r['valid']) for r in records),
                   'person_candidate_records':sum(d['label']=='person' for r in records for d in r['valid']),
                   'model_forward_runs':0})
    from run_bonn_fourarm import input_hashes
    paths=['tools/run_bonn_native_dynamic_ablation.py','demo.py','boxfusion/instances.py',
           'boxfusion/boxes.py','boxfusion/box_manager.py','boxfusion/box_fusion.py',
           'boxfusion/reliable_views.py','config/bonn_native.yaml','data/pst_1024_0.tiff']
    result['input_sha256']=input_hashes()
    result['input_sha256'].update({path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in paths})
    (args.out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'summaries':result['summaries'],'nonperson_ledger':result['nonperson_ledger']},indent=1))


if __name__=='__main__':
    main()
