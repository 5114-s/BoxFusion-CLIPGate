#!/usr/bin/env python3
"""Bounded fresh native + M1/M2 timing. No replay or AP/GT selection.

Uses a declared prefix of the first listed semantic smoke scene.
Measures the existing scene-end implementation; does not claim live latency.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'reports/pipeline_runtime_20260915'

class CountedStream:
    def __init__(self,dataset):self.dataset=dataset;self.consumed=0
    def __len__(self):return len(self.dataset)
    def __getattr__(self,name):return getattr(self.dataset,name)
    def __iter__(self):
        for sample in self.dataset:
            self.consumed+=1
            yield sample

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--frames',type=int,default=100)
    parser.add_argument('--output-root',default=str(OUT))
    parser.add_argument('--semantic-only',action='store_true',help='Finish common readout timing from this run, without detector rerun')
    args=parser.parse_args();OUT=Path(args.output_root);OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'native').mkdir(exist_ok=True)
    if args.semantic_only:
        sys.path.insert(0,str(ROOT/'tools'))
        import run_scannet_semantic_table as semantic
        runtime=json.loads((OUT/'runtime.json').read_text());scene=runtime['scene']
        semantic.OUT=OUT/'semantic_profile';semantic.OUT.mkdir(exist_ok=True)
        semantic.FRAMES=OUT/'scannet_input'
        semantic.ARMS={'native':str(OUT/'native'),'M1':'paired reconstruction',
                       'M1_M2':str(OUT/'M1_M2')}
        start=time.perf_counter();semantic.classify([scene],16);elapsed=time.perf_counter()-start
        timing=json.loads((semantic.OUT/'semantic_cache'/f'{scene}.timing.json').read_text())
        runtime['semantic_profile']=timing
        runtime['semantic_cold_wall_seconds']=elapsed
        runtime['semantic_warm_stage_seconds']=timing['semantic_stage_seconds']-timing['startup_seconds']
        runtime['combined_warm_components_seconds']=runtime['combined_run_seconds_excluding_model_load']+runtime['semantic_warm_stage_seconds']
        runtime['raw_frames_per_second_with_readout']=runtime['native_consumed_raw_frames']/runtime['combined_warm_components_seconds']
        runtime['cold_components_seconds']=runtime['complete_cold_wall_seconds']+elapsed
        runtime['semantic_readout']='same 18-name CLIP readout, separate measured stage; model initialization subtracted for warm component sum'
        (OUT/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
        print('SEMANTIC RUNTIME',runtime['semantic_warm_stage_seconds'],flush=True)
        return
    scene='scene0011_01';source=ROOT/'upstream_clean/scannet_readme_frames'/scene/'frames'
    folder=OUT/'scannet_input'/scene/'frames';folder.mkdir(parents=True,exist_ok=True)
    for sub,suffix in [('color','.jpg'),('depth','.png'),('pose','.txt')]:
        destination=folder/sub;destination.mkdir(exist_ok=True)
        for fid in range(args.frames):
            p=source/sub/f'{fid}{suffix}';assert p.is_file(),p
            target=destination/p.name
            if not target.exists():target.symlink_to(p)
    if not (folder/'intrinsic').exists():(folder/'intrinsic').symlink_to(source/'intrinsic',target_is_directory=True)
    cfg=yaml.safe_load((ROOT/'config/scannet_t05_boxer_kfmap_score05.yaml').read_text())
    cfg['data']['datadir']=str(folder);cfg['data']['output_dir']=str(OUT/'native')
    cfg['lifting']['proposal_cache']['mode']='disabled'
    cfg['lifting']['boxer']['diagnostics_dir']=str(OUT/'boxer_diagnostics')
    cfg['association']['pvq_ar']['diagnostics_dir']=str(OUT/'native_diagnostics')
    cfg['vis']['rerun']=False
    (OUT/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    import torch
    assert torch.cuda.is_available()
    cold_start=time.perf_counter()
    import demo
    checkpoint=torch.load(ROOT/'models/cutr_rgbd.pth',map_location='cpu',weights_only=False)['model']
    dimension=checkpoint['backbone.0.patch_embed.proj.weight'].shape[0]
    model=demo.make_cubify_transformer(dimension=dimension,depth_model=True).eval().cuda()
    model.load_state_dict(checkpoint);del checkpoint
    clip_model,preprocess=demo.load_clip(str(ROOT/'models/open_clip_pytorch_model.bin'))
    categories=np.genfromtxt(ROOT/'data/panoptic_categories_nomerge.txt',delimiter='\n',dtype=str)
    features=torch.load(ROOT/'data/class_features.pt',weights_only=False).cuda()
    dataset=demo.get_dataset(cfg);dataset.load_arkit_depth=True;dataset=CountedStream(dataset)
    augmentor=demo.Augmentor(('wide/image','wide/depth'));preprocessor=demo.Preprocessor()
    torch.cuda.synchronize();loaded=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        demo.run(cfg,model,dataset,clip_model,preprocess,categories,features,augmentor,
                 preprocessor,score_thresh=.5,gap=25,re_vis=False)
    except SystemExit as exit_status:
        if exit_status.code not in (None,0):raise
    torch.cuda.synchronize();native_done=time.perf_counter()
    native_peak=int(torch.cuda.max_memory_allocated())
    del model,clip_model,preprocess,features,dataset.dataset
    gc.collect();torch.cuda.empty_cache()
    os.environ.update(M2_MODE='nativelogit',M2_EXCLUSIVE='1',CAUSAL_TAU='0.5',M5_OFF='1')
    import tools.integrated_online as integrated
    integrated.SCANS=str(OUT/'scannet_input')
    # demo finalizes before the configured last frame. Limit the provider to
    # the actual consumed prefix; it must not read later input frames.
    for fid in range(dataset.consumed,args.frames):
        for sub,suffix in [('color','.jpg'),('depth','.png'),('pose','.txt')]:
            (folder/sub/f'{fid}{suffix}').unlink()
    # The processor expects scene/{color,depth,pose,intrinsic}, not scene/frames.
    for sub in ('color','depth','pose','intrinsic'):
        target=OUT/'scannet_input'/scene/sub
        if not target.exists():target.symlink_to(folder/sub,target_is_directory=True)
    m1cold=time.perf_counter();wmodel,adapter=integrated.load_models()
    torch.cuda.synchronize();m1loaded=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    native_pkl=OUT/'native'/f'{scene}_boxes.pkl';final=OUT/'M1_M2'/f'{scene}_boxes.pkl'
    m1file=OUT/'M1'/f'{scene}_boxes.pkl'
    integrated.process_scene(scene,str(native_pkl),str(OUT/'native_diagnostics'/f'{scene}_pvq_nms.jsonl'),
        str(final),wmodel,adapter,gap=25,score_view='persistent',m1_out_pkl=str(m1file))
    torch.cuda.synchronize();finished=time.perf_counter()
    with native_pkl.open('rb') as f:native_rows=pickle.load(f)[0]
    with final.open('rb') as f:final_rows=pickle.load(f)[0]
    with m1file.open('rb') as f:m1_rows=pickle.load(f)[0]
    assert len(m1_rows)==len(final_rows)
    assert all(np.array_equal(a[1],b[1]) for a,b in zip(m1_rows,final_rows))
    consumed=dataset.consumed;keyframes=len(range(0,consumed,25))
    runseconds=native_done-loaded+finished-m1loaded
    result={'scene':scene,'configured_raw_frames':args.frames,'native_consumed_raw_frames':consumed,
        'provider_keyframes':keyframes,'native_model_import_load_seconds':loaded-cold_start,
        'native_run_seconds':native_done-loaded,'native_peak_allocated_bytes':native_peak,
        'm1_m2_model_load_seconds':m1loaded-m1cold,'m1_m2_run_seconds':finished-m1loaded,
        'm1_m2_peak_allocated_bytes':int(torch.cuda.max_memory_allocated()),
        'combined_run_seconds_excluding_model_load':runseconds,
        'raw_frames_per_second_excluding_model_load':consumed/runseconds,
        'keyframe_updates_per_second_excluding_model_load':keyframes/runseconds,
        'complete_cold_wall_seconds':finished-cold_start,
        'native_rows':len(native_rows),'M1_rows':len(m1_rows),'M1_M2_rows':len(final_rows),
        'paired_geometry':True,'proposal_replay':'disabled','M5':'disabled',
        'semantic_readout':'not included here; separate frozen-CLIP cost reported',
        'device':torch.cuda.get_device_name(),'torch':torch.__version__,
        'scope':'one truncated scene, fresh forward, existing scene-end processing; not live output latency',
        'native_output_sha256':hashlib.sha256(native_pkl.read_bytes()).hexdigest()}
    (OUT/'runtime.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RUNTIME',json.dumps(result),flush=True)

if __name__=='__main__':main()
