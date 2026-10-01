#!/usr/bin/env python3
"""Paired terminal semantic readout of frozen boxes; no detector rerun.

All three arms use identical frozen CLIP, automatic crops and detection scores.
This is development-set semantic detection, not novel-class or online proof.
"""
import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from paper_eval_core import best_view,corners_from_center_size
from true_fusion_audit_core import class_agnostic_ap
ARMS={'native':'scannet_t05_boxer_kfmap_score05',
      'M1':'derived_same_M2_geometry_restore_native_scores',
      'M1_M2':'scannet_m2nl_m5_dual_full100/persistent'}
NAMES=['cabinet','bed','chair','sofa','table','door','window','bookshelf','picture',
       'counter','desk','curtain','refrigerator','shower curtain','toilet','sink',
       'bathtub','garbage bin']
NYU=[3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]
FRAMES=ROOT/'upstream_clean/scannet_readme_frames'
SCANS=Path('/extra/ZhaoX/scannet_data/scans')
GTROOT=ROOT/'evaluation/data_util/scannet_train_detection_data'
OUT=ROOT/'reports/scannet_semantic_20260915'


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def load_scene(scene):
    rows={a:pickle.loads((ROOT/'results'/d/f'{scene}_boxes.pkl').read_bytes())[0]
          for a,d in ARMS.items() if a!='M1'}
    n,z=rows['native'],rows['M1_M2']
    rows['M1']=[(r[0],r[1],float(n[i][2]) if i<len(n) else float(r[2]))
                for i,r in enumerate(z)]
    n,m,z=rows['native'],rows['M1'],rows['M1_M2']
    assert len(m)==len(z) and len(n)<=len(m)
    assert all(np.array_equal(x[1],y[1]) and int(x[0])==int(y[0]) for x,y in zip(m,z))
    assert all(np.array_equal(x[1],y[1]) and float(x[2])==float(y[2]) for x,y in zip(n,m))
    assert all(float(x[2])==float(y[2]) for x,y in zip(m[len(n):],z[len(n):]))
    return rows


def ground_truth(scene):
    b=np.load(GTROOT/f'{scene}_bbox.npy');b=b[np.isin(b[:,-1],NYU)]
    return corners_from_center_size(b),np.array([NYU.index(int(v)) for v in b[:,-1]])


def alignment(scene):
    for line in (SCANS/scene/f'{scene}.txt').read_text().splitlines():
        if line.startswith('axisAlignment'):
            return np.fromstring(line.split('=',1)[1],sep=' ').reshape(4,4)
    raise ValueError(f'No alignment: {scene}')


def classify(scenes,batch_size):
    import torch
    import open_clip
    from PIL import Image
    cache=OUT/'semantic_cache';cache.mkdir(exist_ok=True)
    model=prep=text=None;startup=None
    for si,scene in enumerate(scenes):
        rows=load_scene(scene)['M1']
        boxes=np.asarray([r[1] for r in rows],float).reshape(-1,8,3)
        geometry_hash=hashlib.sha256(boxes.tobytes()).hexdigest()
        dest=cache/f'{scene}.npz'
        if dest.exists():
            with np.load(dest) as old:assert str(old['geometry_hash'])==geometry_hash
            print(f'CACHED {si+1}/{len(scenes)} {scene}',flush=True);continue
        start=time.perf_counter()
        if model is None:
            t0=time.perf_counter()
            model,_,prep=open_clip.create_model_and_transforms(
                'ViT-H-14',pretrained=str(ROOT/'models/open_clip_pytorch_model.bin'))
            model.cuda().eval()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                text=model.encode_text(open_clip.tokenize(NAMES).cuda()).float()
                text/=text.norm(dim=-1,keepdim=True)
            torch.cuda.synchronize();startup=time.perf_counter()-t0
        directory=FRAMES/scene/'frames'
        colors=sorted((directory/'color').glob('*.jpg'),key=lambda p:int(p.stem))[::25]
        paths,poses=[],[]
        for path in colors:
            pp=directory/'pose'/f'{path.stem}.txt'
            if pp.exists():
                pose=np.loadtxt(pp).reshape(4,4)
                if np.isfinite(pose).all():paths.append(path);poses.append(pose)
        if not paths:raise ValueError(f'No valid keyframes: {scene}')
        with Image.open(paths[0]) as im:width,height=im.size
        intrinsic=np.loadtxt(directory/'intrinsic/intrinsic_color.txt')[:3,:3]
        indices,rectangles=best_view(boxes,poses,intrinsic,width,height)
        features=np.zeros((len(boxes),text.shape[1]),np.float32)
        labels=np.full(len(boxes),-1,int);view_ids=np.full(len(boxes),-1,int)
        valid=np.flatnonzero(indices>=0);images={};gpu_seconds=0.;batches=[];crop_hashes=[]
        for begin in range(0,len(valid),batch_size):
            ids=valid[begin:begin+batch_size];inputs=[]
            for bi in ids:
                path=paths[indices[bi]]
                if path not in images:
                    with Image.open(path) as im:images[path]=im.convert('RGB')
                x1,y1,x2,y2=rectangles[bi];dx,dy=.15*(x2-x1),.15*(y2-y1)
                crop=images[path].crop((int(max(0,x1-dx)),int(max(0,y1-dy)),
                    int(min(width,x2+dx)),int(min(height,y2+dy))))
                inputs.append(prep(crop));view_ids[bi]=int(path.stem)
                crop_hashes.append(hashlib.sha256(crop.tobytes()).hexdigest())
            torch.cuda.synchronize();t0=time.perf_counter()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                feat=model.encode_image(torch.stack(inputs).cuda()).float()
                feat/=feat.norm(dim=-1,keepdim=True)
                cls=(feat@text.T).argmax(-1).cpu().numpy()
                features[ids]=feat.cpu().numpy();labels[ids]=cls
            torch.cuda.synchronize();dt=time.perf_counter()-t0
            gpu_seconds+=dt;batches.append(dt)
        image_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in images}
        np.savez_compressed(dest,features=features,labels=labels,view_ids=view_ids,
            rectangles=rectangles,geometry_hash=geometry_hash,crop_hashes=np.asarray(crop_hashes))
        write(cache/f'{scene}.timing.json',{'scene':scene,'boxes':len(boxes),
            'classified':len(valid),'abstained':int((labels<0).sum()),'keyframes':len(paths),
            'semantic_stage_seconds':time.perf_counter()-start,'clip_seconds':gpu_seconds,
            'batch_seconds':batches,'clip_peak_allocated_bytes':int(torch.cuda.max_memory_allocated()),
            'image_sha256':image_hashes,'startup_seconds':startup})
        print(f'CLASSIFIED {si+1}/{len(scenes)} {scene} {len(valid)}/{len(boxes)} '
              f'CLIP={gpu_seconds:.2f}s',flush=True)


def evaluate(scenes):
    result={'scenes':scenes,'arms':{},'pair_audit':{'passed':True}}
    all_gt,gt_classes={},{}
    predictions,classes={a:{} for a in ARMS},{a:{} for a in ARMS}
    for scene in scenes:
        all_gt[scene],gt_classes[scene]=ground_truth(scene);align=alignment(scene)
        with np.load(OUT/'semantic_cache'/f'{scene}.npz') as cache:
            labels=cache['labels'].copy();assert np.isfinite(cache['features']).all()
        for arm,rows in load_scene(scene).items():
            boxes=np.asarray([r[1] for r in rows],float).reshape(-1,8,3)
            predictions[arm][scene]=(boxes@align[:3,:3].T+align[:3,3],
                                    np.asarray([float(r[2]) for r in rows]))
            classes[arm][scene]=labels[:len(rows)]
    for arm in ARMS:
        per_class={}
        for cid,name in enumerate(NAMES):
            gts={s:all_gt[s][gt_classes[s]==cid] for s in scenes}
            preds={s:(b[classes[arm][s]==cid],c[classes[arm][s]==cid])
                   for s,(b,c) in predictions[arm].items()}
            count=sum(len(g) for g in gts.values())
            per_class[name]={'gt':count,'predictions':sum(len(b) for b,c in preds.values())}
            for thr in (.15,.25,.5):
                per_class[name][str(thr)]=class_agnostic_ap(preds,gts,thr) if count else None
        active=[v for v in per_class.values() if v['gt']]
        result['arms'][arm]={'per_class':per_class,
            'mAP':{str(t):float(np.mean([v[str(t)]['ap'] for v in active])) for t in (.15,.25,.5)},
            'class_agnostic':{str(t):class_agnostic_ap(predictions[arm],all_gt,t) for t in (.15,.25,.5)},
            'boxes':sum(len(b) for b,c in predictions[arm].values()),
            'abstained':sum(int((x<0).sum()) for x in classes[arm].values())}
        print(arm,result['arms'][arm]['mAP'],flush=True)
    write(OUT/('smoke_results.json' if len(scenes)<100 else 'semantic_table.json'),result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--evaluate-only',action='store_true')
    args=p.parse_args();OUT.mkdir(exist_ok=True)
    all_scenes=sorted(f.name.replace('_boxes.pkl','') for f in
        (ROOT/'results'/ARMS['native']).glob('*_boxes.pkl'))
    official=(ROOT/'evaluation/data_util/meta_data/scannetv2_val.txt').read_text().split()
    assert len(all_scenes)==100 and set(all_scenes)==set(official)
    scenes=all_scenes[:args.limit]
    paths=[ROOT/'results'/d/f'{s}_boxes.pkl' for s in all_scenes
           for a,d in ARMS.items() if a!='M1']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    protocol={'arms':ARMS,'scenes':all_scenes,'class_names':NAMES,'nyu40ids':NYU,
        'ranking_score':'unchanged detection score','class_assignment':'argmax normalized CLIP cosine',
        'view_rule':'largest valid projected area in own gap25 keyframes; 15% padding',
        'no_valid_projection':'semantic abstention -1; retained in class-agnostic AP',
        'representation':'M1 paired reconstruction: M2 geometry and birth scores unchanged, '
                         'native scores restored from native input; shared CLIP labels; '
                         'M2 nativelogit persistent score view, M5 current-score effect excluded',
        'precision':'frozen ViT-H-14 fp16 autocast; no new training or tuning',
        'scope':'terminal common semantic readout; development scenes, not novel/online proof',
        'metric':'axisAlignment to ScanNet GT; AABB strict >, VOC envelope; all negative scenes retained',
        'batch_size':args.batch_size}
    if (OUT/'protocol.json').exists() and list((OUT/'semantic_cache').glob('*.npz')):
        assert json.loads((OUT/'protocol.json').read_text())==protocol
    else:write(OUT/'protocol.json',protocol)
    for s in all_scenes:load_scene(s)
    write(OUT/'input_sha256.json',hashes)
    if not args.evaluate_only:classify(scenes,args.batch_size)
    evaluate(scenes)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    write(OUT/'integrity.json',{'input_predictions_unchanged':True,'paired_all100':True})


if __name__=='__main__':main()
