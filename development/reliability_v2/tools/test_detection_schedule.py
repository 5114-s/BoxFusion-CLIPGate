"""Frozen dev3 same-budget scheduling pilot; production sources stay untouched.

Native demo control-flow is transformed in memory at exactly its keyframe
predicates. Existing terminal-save behavior and native fusion are unchanged.
M1+M2 use the same selected timestamps, retaining the original M1-only tail.
"""
from __future__ import annotations

import argparse
import ast
from collections import deque
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.test_detection_schedule_gate import SCENES, angles, image_metrics, improvement, write_json
from tools.verify_true_fusion_capture import canonical_corners, digest

ARMS=('fixed','sharpness','object')
PROTOCOL={
    'schema':'boxfusion.detection_schedule_dev3.v1','scenes':SCENES,'arms':ARMS,
    'base_config':'config/ca1m_thr15_obs.yaml',
    'baseline':'Fresh native CuTR+Boxer+TopK3+real score(.15) + dual-source M1 + M2-nativelogit, exclusive support; M5 off.',
    'budget':'Frame0 then exactly one native detection per complete native gap20 period. All original frame0 retries unchanged. One WeDetect call per selected timestamp, plus unchanged original M1-only terminal tail.',
    'policy':'No action before period offset5; period deadline is fallback. Every fourth period fixed for ordinary discovery.',
    'sharpness':'Early trigger when current global Laplacian energy/(variance+1) >=1.25x previous selected frame.',
    'object':'Early trigger when >=2 deficient seeds improve by frozen gate quality rule relative to their own latest observation. Seed parallax baseline=0. Max16 seeds; top16 observations per detection in a deque of10 batches.',
    'deficit':'Fewer than3 distinct past support frames at AABB IoU>.25, or past angular spread<15deg.',
    'causality':'Decisions are computed on the current sample and already committed native observations; query-before-commit; no offline gate-selected schedule, future poses, GT, or final map used for scheduling.',
    'frozen':'Detector/model weights, detection thresholds, association rules, PFO, M1 admission/pricing and M2 coefficients; one policy per arm, no sweep.',
    'terminal_contract':'Native demo saves before EOF by its existing gap-based condition; do not fix that in this experiment. Native scheduling cannot spend a token from a period beyond this stop. Existing M1 fixed-gap tail remains identical across arms.',
    'gate':'object-fixed AP50>=.5pp, AP15/AP25>=0; object must also outperform sharpness in AP50 to support object-specific contribution. Full pipeline throughput >=15 input FPS required, measured separately.',
    'evaluation':'All9 inference outputs frozen before GT evaluation. Exact original AABB/AP anchor; real scores; no additional filters/NMS.',
    'runtime':'Serial same-GPU native and M1 processes; whole-process throughput includes initialization; native loop+M1 processing throughput also reported. No inference on every raw frame.',
    'limits':['Three previously used development scenes, not independent generalization.',
              'M1+M2 retain the existing scene-end processing semantics; terminal AP/throughput cannot establish bounded-latency arbitrary-prefix output.',
              'Changing detection times changes future association history: full native feedback is intentionally rerun.',
              'Quality and geometry proxies may select different but not better observations.'],
}


def sharpness(gray):
    value=np.asarray(gray,dtype=np.float32)
    lap=cv2.Laplacian(value,cv2.CV_32F,ksize=1)
    return float(np.mean(lap*lap)/(value.var()+1.))


def convert_sample(sample):
    info=sample['sensor_info']
    pose=info.gt.RT[-1].numpy().astype(float)
    rgb=np.moveaxis(sample['wide']['image'][-1].numpy(),0,-1)
    if rgb.dtype!=np.uint8:
        assert rgb.max()>1, 'Unexpected normalized RGB convention'
        rgb=np.clip(rgb,0,255).astype(np.uint8)
    h,w=rgb.shape[:2]
    gray=cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),(512,384),interpolation=cv2.INTER_LINEAR)
    depth=sample['wide']['depth'][-1].numpy()
    dh,dw=depth.shape
    depth=cv2.resize(depth.astype(np.float32),(512,384),interpolation=cv2.INTER_NEAREST)
    # Native dataset provides metric depth, never millimetres here.
    assert np.nanmedian(depth[depth>0])<100 if np.any(depth>0) else True
    k=info.wide.image.K[-1].numpy()[:3,:3].astype(float).copy()
    kd=info.gt.depth.K[-1].numpy()[:3,:3].astype(float).copy()
    k[0]*=512/w;k[1]*=384/h;kd[0]*=512/dw;kd[1]*=384/dh
    return pose,gray,depth,k,kd


class Scheduler:
    def __init__(self,arm):
        assert arm in ARMS
        self.arm=arm;self.history=deque(maxlen=10);self.seeds=[]
        self.selected=[];self.records=[];self.last_frame=-1;self.used=set()
        self.current=None;self.current_frame=-1;self.previous_sharp=None
        self.elapsed=0.;self.total=None

    def begin(self,frame,sample,total):
        started=time.perf_counter()
        assert frame==self.last_frame+1,'Nonsequential scheduler access'
        self.last_frame=frame;self.total=total;self.current_frame=frame;self.current=None
        period=(frame-1)//20+1 if frame else 0
        max_deadline=((total-21)//20)*20
        reason='skip';self.active=False;improved=0
        allowed=frame==0 or (period*20<=max_deadline and period not in self.used)
        if allowed:
            if frame==0 or frame%20==0:
                self.active=True;reason='startup' if frame==0 else 'deadline'
            elif self.arm!='fixed' and period%4!=0 and (frame-1)%20+1>=5:
                self.current=convert_sample(sample)
                if self.arm=='sharpness':
                    if self.previous_sharp is not None and sharpness(self.current[1])>=1.25*self.previous_sharp:
                        self.active=True;reason='sharpness'
                else:
                    pose,gray,depth,k,kd=self.current
                    for seed in self.seeds:
                        assert seed['frame']<frame
                        value=image_metrics(seed['corners'],pose,seed['pose'],gray,depth,k,kd)
                        improved+=improvement(value,seed['quality']) is not None
                    if improved>=2:
                        self.active=True;reason='object_evidence'
            if self.active:
                if self.arm!='fixed' and self.current is None:self.current=convert_sample(sample)
                if self.current is not None:self.previous_sharp=sharpness(self.current[1])
                self.used.add(period);self.selected.append(frame)
                self.records.append({'frame':frame,'period':period,'reason':reason,'improved_seeds':improved,
                                     'seed_count':len(self.seeds),'latest_committed_frame':max((s['frame'] for s in self.seeds),default=-1)})
        self.elapsed+=time.perf_counter()-started

    def record(self,frame,instances):
        if self.arm!='object':return
        started=time.perf_counter();assert self.active and frame==self.current_frame and self.current is not None
        boxes=instances.pred_boxes_3d.tensor.detach().cpu().numpy()
        rotations=instances.pred_boxes_3d.R.detach().cpu().numpy()
        scores=instances.scores.detach().cpu().numpy()
        ids=np.argsort(-scores,kind='stable')[:16]
        corners=canonical_corners(boxes[ids],rotations[ids])
        pose,gray,depth,k,kd=self.current
        self.history.append({'frame':frame,'corners':corners.copy(),'pose':pose.copy()})
        self.seeds=[]
        for box in corners:
            low,high=box.min(0),box.max(0);support=[];cameras=[]
            for batch in self.history:
                assert batch['frame']<=frame
                lo,hi=batch['corners'].min(1),batch['corners'].max(1)
                inter=np.maximum(0,np.minimum(hi,high)-np.maximum(lo,low)).prod(1)
                iou=inter/np.maximum((hi-lo).prod(1)+(high-low).prod()-inter,1e-9)
                if np.any(iou>.25):support.append(batch['frame']);cameras.append(batch['pose'][:3,3])
            spread=float(angles(box.mean(0),np.asarray(cameras),pose[:3,3]).max(initial=0)) if cameras else 0.
            if len(support)<3 or spread<15:
                quality=image_metrics(box,pose,pose,gray,depth,k,kd)
                self.seeds.append({'frame':frame,'pose':pose.copy(),'corners':box.copy(),'quality':quality})
        self.elapsed+=time.perf_counter()-started

    def finish(self):
        expected=list(range(0,self.total-20,20))
        assert len(self.selected)==len(expected),(self.selected,expected)
        assert [0 if f==0 else (f-1)//20+1 for f in self.selected]==list(range(len(expected)))
        if self.arm=='fixed':assert self.selected==expected
        return {'arm':self.arm,'selected_frames':self.selected,'original_native_frames':expected,
                'm1_tail_frames':list(range(expected[-1]+20,self.total,20)),
                'native_detection_count':len(self.selected),'input_frames':self.total,
                'scheduler_processed_frames':self.last_frame+1,'scheduler_seconds':self.elapsed,
                'causal_records':self.records,'max_history_batches':10,'max_observations_per_batch':16,
                'max_seeds':16,'budget_check_passed':True}


def transform_demo(source):
    class Rewrite(ast.NodeTransformer):
        def __init__(self):self.predicates=0;self.loops=0;self.commits=0
        def visit_Compare(self,node):
            text=ast.unparse(node)
            if text in ('count % gap == 0','count % gap != 0'):
                self.predicates+=1
                return ast.copy_location(ast.parse('SCHED.active' if '==' in text else 'not SCHED.active',mode='eval').body,node)
            return self.generic_visit(node)
        def visit_For(self,node):
            node=self.generic_visit(node)
            if isinstance(node.target,ast.Name) and node.target.id=='sample' and isinstance(node.iter,ast.Name) and node.iter.id=='dataset':
                self.loops+=1;node.body.insert(0,ast.parse('SCHED.begin(count, sample, len(dataset))').body[0])
            return node
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='pred_instances.pred_boxes_3d.transform2world':
                self.commits+=1
                return [node,ast.parse('SCHED.record(count, pred_instances)').body[0]]
            return self.generic_visit(node)
    visitor=Rewrite();tree=visitor.visit(ast.parse(source));ast.fix_missing_locations(tree)
    assert (visitor.predicates,visitor.loops,visitor.commits)==(5,1,1),(visitor.predicates,visitor.loops,visitor.commits)
    return tree,{'keyframe_predicates':visitor.predicates,'stream_hooks':visitor.loops,'commit_hooks':visitor.commits}


def gt_guard(event,values):
    if event=='open' and values and isinstance(values[0],(str,bytes)):
        if any(s in str(values[0]) for s in ('after_filter_boxes.npy','full_annotations.json')):
            raise RuntimeError('GT read forbidden during inference')


def worker(args):
    import torch
    import random
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    assert torch.cuda.is_available(),'CUDA required; never fall back to CPU inference'
    sys.addaudithook(gt_guard)
    target=args.output/args.scene/args.arm
    if args.stage=='native':
        (target/'native').mkdir(exist_ok=False)
        config=yaml.safe_load((ROOT/'config/ca1m_thr15_obs.yaml').read_text())
        config['data']['output_dir']=str(target/'native')
        config['association']['pvq_ar']['diagnostics_dir']=str(target/'nms')
        config['lifting']['boxer']['diagnostics_dir']=str(target/'boxer')
        path=target/'config.yaml'
        with path.open('x') as handle:yaml.safe_dump(config,handle,sort_keys=False)
        schedule=Scheduler(args.arm)
        source=(ROOT/'demo.py').read_text();tree,receipt=transform_demo(source)
        write_json(target/'runtime_transform.json',receipt)
        ns={'__name__':'__main__','__file__':str(ROOT/'demo.py'),'SCHED':schedule}
        sys.argv=[str(ROOT/'demo.py'),'CA1M','--seq',args.scene,'--config',str(path),
                  '--model-path',str(ROOT/'models/cutr_rgbd.pth'),'--device','cuda']
        try:exec(compile(tree,str(ROOT/'demo.py'),'exec'),ns)
        except SystemExit as error:
            if error.code not in (None,0):raise
        assert (target/'native'/f'{args.scene}_boxes.pkl').exists()
        write_json(target/'schedule.json',schedule.finish())
    else:
        import tools.integrated_online as online
        schedule=json.loads((target/'schedule.json').read_text())
        selected=schedule['selected_frames']+schedule['m1_tail_frames']
        selected=sorted(selected);assert len(selected)==len(range(0,schedule['input_frames'],20))
        online.M5_OFF=True;online.M5_CH2_OFF=True;online.M2_MODE='nativelogit';online.M2_EXCLUSIVE=True
        original_loader=online.load_ca1m_scene
        def loader(scene):
            k,kd,_,w,h,gap=original_loader(scene)
            directory=Path(online.CA1M_ROOT)/scene
            poses=np.load(directory/'all_poses.npy')
            kfs=[(f,poses[f],str(directory/'rgb'/f'{f}.png'),str(directory/'depth'/f'{f}.png')) for f in selected]
            return k,kd,kfs,w,h,gap
        online.load_ca1m_scene=loader
        model,adapter=online.load_models();calls=[];lift_calls=[]
        class Counted:
            def __call__(self,paths):
                assert len(paths)==1
                calls.append(int(Path(paths[0]).stem));return model(paths)
        original_forward=adapter.forward_raw_with_feature_cache
        def count_forward(*values,**kwargs):
            lift_calls.append(kwargs['frame_id']);return original_forward(*values,**kwargs)
        adapter.forward_raw_with_feature_cache=count_forward
        started=time.perf_counter()
        online.process_scene(args.scene,str(target/'native'/f'{args.scene}_boxes.pkl'),
                             str(target/'nms'/f'{args.scene}_pvq_nms.jsonl'),
                             str(target/'m1m2'/f'{args.scene}_boxes.pkl'),Counted(),adapter,score_view='persistent')
        assert calls==selected
        write_json(target/'m1m2_runtime.json',{'frames':calls,'detector_calls':len(calls),'boxer_calls':len(lift_calls),
                                             'processing_seconds':time.perf_counter()-started,'M5_off':True})


def checks():
    _,changes=transform_demo((ROOT/'demo.py').read_text())
    # Prefix determinism for control; exact same-budget period arithmetic.
    for n in (100,646,855):
        s=Scheduler('fixed')
        for f in range(n-20):s.begin(f,None,n)
        assert s.finish()['budget_check_passed']
    # Synthetic causal trigger, discovery quota and prefix determinism checks.
    original=globals()['convert_sample']
    pose=np.eye(4);k=np.array([[420.,0,256],[0,420,192],[0,0,1.]])
    gray=np.indices((384,512)).sum(0).astype(np.uint8)
    ctx=(pose,gray,np.full((384,512),3.,np.float32),k,k)
    globals()['convert_sample']=lambda sample:ctx
    try:
        traces=[]
        for _ in range(2):
            s=Scheduler('object')
            corners=canonical_corners(np.array([[0,0,3,1,1,1.]]),np.eye(3)[None])[0]
            s.seeds=[{'frame':0,'pose':pose,'corners':corners,'quality':{'valid':False}} for _ in range(2)]
            for f in range(101):s.begin(f,None,121)
            result=s.finish();traces.append(result['selected_frames'])
            assert result['selected_frames']==[0,5,25,45,80,85]
            assert max(r['latest_committed_frame'] for r in result['causal_records'])<=0
        assert traces[0]==traces[1]
    finally:globals()['convert_sample']=original
    return changes


def run(args):
    checks()
    args.output.mkdir(parents=True,exist_ok=False)
    source_paths=[Path(__file__),ROOT/'tools/test_detection_schedule_gate.py',ROOT/'demo.py',
                  ROOT/'tools/integrated_online.py',ROOT/'config/ca1m_thr15_obs.yaml']
    source_paths+=list((ROOT/'boxfusion').glob('*.py'))
    frozen={str(p.resolve()):digest(p) for p in source_paths}
    gate=json.loads((ROOT/'reports/detection_schedule_gate_dev3_20260908/results.json').read_text())
    assert gate['gate_passed']
    gate_hashes=json.loads((ROOT/'reports/detection_schedule_gate_dev3_20260908/input_sha256.json').read_text())
    assert all(digest(p)==h for p,h in gate_hashes.items())
    model_paths=[ROOT/'models/cutr_rgbd.pth',ROOT/'models/open_clip_pytorch_model.bin',ROOT/'data/class_features.pt',
                 ROOT/'third_party/WeDetect/wedetect_base_uni.pth']
    model_hashes={str(p.resolve()):digest(p) for p in model_paths}
    write_json(args.output/'protocol.json',{**PROTOCOL,'source_sha256':frozen,'model_sha256':model_hashes})
    runtime=[]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=args.gpu,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',
        M5_OFF='1',M5_CH2_OFF='1',M2_MODE='nativelogit',M2_EXCLUSIVE='1',CAUSAL_TAU='0.5',
        MPLCONFIGDIR=str(args.output/'mplconfig'),XDG_CACHE_HOME=str(args.output/'cache'))
    for index,scene in enumerate(SCENES):
        order=ARMS[index:]+ARMS[:index]
        for arm in order:
            target=args.output/scene/arm;target.mkdir(parents=True)
            times={}
            for stage in ('native','m1m2'):
                path=target/f'{stage}.log';started=time.perf_counter()
                print(f'{scene} {arm} {stage}: start',flush=True)
                with path.open('x') as log:
                    result=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(args.output),
                                           '--stage',stage,'--scene',scene,'--arm',arm],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                times[stage]=time.perf_counter()-started
                if result.returncode or re.search(r'\bf\d+ ERROR:',path.read_text(errors='replace')):
                    raise RuntimeError(f'Failed {scene}/{arm}/{stage}; preserve {path}')
                for source,expected in frozen.items():
                    assert digest(source)==expected,source
                print(f'{scene} {arm} {stage}: complete ({times[stage]:.1f}s)',flush=True)
            schedule=json.loads((target/'schedule.json').read_text())
            m1=json.loads((target/'m1m2_runtime.json').read_text())
            log=(target/'native.log').read_text(errors='replace')
            costs=re.findall(r'Cost:\s*([0-9.]+)',log)
            runtime.append({'scene':scene,'arm':arm,'process_seconds':times,
                            'input_frames':schedule['input_frames'],'native_frames_processed':schedule['scheduler_processed_frames'],
                            'native_loop_seconds':float(costs[-1]) if costs else None,'m1m2_processing_seconds':m1['processing_seconds'],
                            'native_calls':schedule['native_detection_count'],'wedetect_calls':m1['detector_calls'],
                            'scheduler_seconds':schedule['scheduler_seconds'],
                            'early_triggers':sum(r['reason'] in ('object_evidence','sharpness') for r in schedule['causal_records'])})
    # Freeze every output before opening GT; do not select arms/scenes by AP.
    files=list(args.output.glob('*/*/native/*_boxes.pkl'))+list(args.output.glob('*/*/m1m2/*_boxes.pkl'))
    outputs={str(p.resolve()):digest(p) for p in files};assert len(outputs)==18
    write_json(args.output/'outputs_before_gt.json',outputs);write_json(args.output/'runtime.json',runtime)
    from tools.audit_m1m2_remaining_children import read_prediction,verify_metric
    from tools.true_fusion_audit_core import class_agnostic_ap
    anchor=verify_metric()
    gt={s:np.load(Path('/tmp/ca1m_clean_root')/s/'after_filter_boxes.npy') for s in SCENES}
    predictions={a:{s:read_prediction(args.output/s/a/'m1m2'/f'{s}_boxes.pkl') for s in SCENES} for a in ARMS}
    metrics={a:{str(t):class_agnostic_ap(predictions[a],gt,t) for t in (.15,.25,.5)} for a in ARMS}
    deltas={a:{t:metrics[a][t]['ap']-metrics['fixed'][t]['ap'] for t in metrics[a]} for a in ('sharpness','object')}
    for s in SCENES:
        group=[r for r in runtime if r['scene']==s]
        assert len({r['native_calls'] for r in group})==len({r['wedetect_calls'] for r in group})==1
    speed={}
    for a in ARMS:
        group=[r for r in runtime if r['arm']==a]
        frames=sum(r['native_frames_processed'] for r in group)
        speed[a]={'whole_process_input_fps':frames/sum(sum(r['process_seconds'].values()) for r in group),
                  'loop_plus_m1_input_fps':frames/sum(r['native_loop_seconds']+r['m1m2_processing_seconds'] for r in group)
                  if all(r['native_loop_seconds'] is not None for r in group) else None}
    result={'completed':True,'metric_checks':anchor,'metrics':metrics,'delta_vs_fixed_pp':deltas,
            'object_minus_sharpness_ap50':metrics['object']['0.5']['ap']-metrics['sharpness']['0.5']['ap'],
            'accuracy_gate_passed':deltas['object']['0.5']>=.5 and deltas['object']['0.15']>=0 and deltas['object']['0.25']>=0,
            'timing':speed,'runtime':runtime,'budget_check_passed':True,'scope':PROTOCOL['limits']}
    assert all(digest(p)==h for p,h in outputs.items())
    assert all(digest(p)==h for p,h in model_hashes.items())
    write_json(args.output/'results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('runtime','scope')},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path);p.add_argument('--stage',choices=('native','m1m2'))
    p.add_argument('--scene',choices=SCENES);p.add_argument('--arm',choices=ARMS);p.add_argument('--gpu',default='0')
    p.add_argument('--self-test',action='store_true');args=p.parse_args()
    if args.self_test:print(checks());return
    if args.output is None:p.error('--output required')
    args.output=args.output.resolve()
    if args.stage:worker(args)
    else:run(args)


if __name__=='__main__':main()
