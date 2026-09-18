#!/usr/bin/env python3
"""Check cached semantic AP against the repository's original evaluator."""
import contextlib
import io
import json
import sys
from pathlib import Path
import numpy as np
import run_scannet_semantic_table as sem

sys.path.remove(str(sem.ROOT/'tools'))
sys.path.insert(0,str(sem.ROOT/'evaluation/utils'))
from eval_det import eval_det_cls,get_iou_obb_v2
from ap_helper import flip_axis_to_camera
from utils import obb_to_aabb_corners,reorganize_obb_to_aabb

def camera_aabb(boxes):
    return reorganize_obb_to_aabb(obb_to_aabb_corners(flip_axis_to_camera(boxes)))

def main():
    result=json.loads((sem.OUT/'smoke_results.json').read_text())
    scenes=result['scenes'];gts={};gt_classes={}
    for scene in scenes:gts[scene],gt_classes[scene]=sem.ground_truth(scene)
    checks={}
    for arm in sem.ARMS:
        predictions={};labels={}
        for scene in scenes:
            rows=sem.load_scene(scene)[arm];align=sem.alignment(scene)
            boxes=np.asarray([r[1] for r in rows],float).reshape(-1,8,3)
            boxes=boxes@align[:3,:3].T+align[:3,3]
            predictions[scene]=(camera_aabb(boxes),[float(r[2]) for r in rows])
            with np.load(sem.OUT/'semantic_cache'/f'{scene}.npz') as cache:
                labels[scene]=cache['labels'][:len(rows)]
        for cid,name in enumerate(sem.NAMES):
            if not result['arms'][arm]['per_class'][name]['gt']:continue
            gt={s:list(camera_aabb(gts[s][gt_classes[s]==cid])) for s in scenes}
            pred={s:[(b,c) for b,c,l in zip(*predictions[s],labels[s]) if l==cid] for s in scenes}
            for threshold in (.15,.25,.5):
                with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    _,_,ap=eval_det_cls(pred,gt,ovthresh=threshold,get_iou_func=get_iou_obb_v2)
                ours=result['arms'][arm]['per_class'][name][str(threshold)]['ap']
                checks[f'{arm}/{name}/{threshold}']=abs(100*ap-ours)
    maximum=max(checks.values());assert maximum<1e-4,maximum
    sem.write(sem.OUT/'anchor_ap_parity.json',{'scenes':scenes,'checks':checks,
        'max_AP_point_difference':maximum,'passed':True,
        'evaluator':'evaluation/utils/ap_helper.py APCalculator uses eval_det_cls + get_iou_obb_v2; original camera-AABB conversion'})
    print('ANCHOR AP PARITY',maximum,len(checks))

if __name__=='__main__':main()
