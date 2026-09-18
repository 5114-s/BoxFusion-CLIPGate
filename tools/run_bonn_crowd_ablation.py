#!/usr/bin/env python3
"""Frozen 11-frame crowd test: actual native backend versus current-person bypass."""
from __future__ import annotations
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/bonn_crowd_native_dynamic_20260914'


def overlap(a, b, prediction_area=False):
    a, b = np.asarray(a), np.asarray(b)
    intersection = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    area_a, area_b = np.maximum(0, a[2:]-a[:2]).prod(), np.maximum(0, b[2:]-b[:2]).prod()
    return float(intersection/max(area_a if prediction_area else area_a+area_b-intersection, 1e-12))


def match_frame(predictions, reference):
    """Evaluation only; no annotations are exposed to inference."""
    gt = reference['people']
    matched = []
    if predictions and gt:
        ious = np.asarray([[overlap(p['box'], g['box']) for g in gt] for p in predictions])
        # One extra valid match dominates every possible change in summed IoU.
        bonus = min(len(predictions), len(gt))+1
        weights = np.where(ious >= .5, bonus+ious, 0)
        for i, j in zip(*linear_sum_assignment(-weights)):
            if ious[i,j] >= .5:
                matched.append({'prediction':int(i), 'reference':int(j), 'iou':float(ious[i,j]),
                                'identity':gt[j]['identity'],
                                'track_id':predictions[i].get('track_id')})
    used = {m['prediction'] for m in matched}
    ignored, duplicates, other_fp = [], [], []
    for i, p in enumerate(predictions):
        if i in used:
            continue
        if any(overlap(p['box'], b, True) >= .5 for b in reference['ignore_regions']):
            ignored.append(i)
        elif any(overlap(p['box'], g['box']) >= .5 for g in gt):
            duplicates.append(i)
        else:
            other_fp.append(i)
    return {'references':len(gt), 'predictions':len(predictions), 'tp':len(matched),
            'fn':len(gt)-len(matched), 'fp':len(duplicates)+len(other_fp),
            'duplicates':len(duplicates), 'other_fp':len(other_fp), 'ignored':len(ignored),
            'matches':matched, 'duplicate_indices':duplicates,
            'other_fp_indices':other_fp, 'ignored_indices':ignored}


def summarize(frames):
    keys = ['references','predictions','tp','fn','fp','duplicates','other_fp','ignored']
    result = {k:sum(f[k] for f in frames) for k in keys}
    result['recall'] = result['tp']/max(result['references'], 1)
    result['precision'] = result['tp']/max(result['tp']+result['fp'], 1)
    result['f1'] = 2*result['tp']/max(2*result['tp']+result['fp']+result['fn'], 1)
    return result


def identity_diagnostic(frames):
    # Secondary diagnostic on raw 2D matches, to avoid conflating bad 3D lifting
    # with ID quality. Native BoxFusion has no directly comparable track IDs.
    last, switches, transitions = {}, [], 0
    for frame in frames:
        for m in frame['matches']:
            person, track = m['identity'], m['track_id']
            if person is None or track is None:
                continue
            if person in last:
                transitions += 1
                if last[person]['track'] != track:
                    switches.append({'identity':person, 'from':last[person],
                                     'to':{'frame':frame['frame'], 'track':track}})
            last[person] = {'frame':frame['frame'], 'track':track}
    return {'matched_identity_transitions':transitions, 'switches':len(switches),
            'details':switches, 'not_standard_mot_metrics':True}


def main():
    # Import the CUDA backend only for the actual experiment, not metric tests.
    sys.path.insert(0, str(ROOT/'tools'))
    import torch
    import yaml
    from run_bonn_native_dynamic_ablation import NativeBackend, CurrentPeople, project, GPU_MODE
    if not GPU_MODE or not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA PFO required')
    protocol = json.loads((OUT/'protocol.json').read_text())
    frozen_paths = [OUT/'protocol.json', OUT/'annotations.json', Path(__file__),
                    ROOT/'tools/run_bonn_native_dynamic_ablation.py', ROOT/'config/bonn_native.yaml',
                    ROOT/'reports/bonn_detect_20260914/yoloe_candidates.json']
    frozen_paths += [ROOT/'boxfusion'/f for f in ('instances.py','boxes.py','box_manager.py','box_fusion.py','reliable_views.py')]
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen_paths}
    (OUT/'input_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    cfg = yaml.safe_load((ROOT/'config/bonn_native.yaml').read_text())
    cfg['dataset'] = 'online'
    (OUT/'config.yaml').write_text(yaml.safe_dump(cfg))
    folder = ROOT/f"data_bonn/{protocol['scene']}/frames"
    K = np.loadtxt(folder/'intrinsic/intrinsic_depth.txt')[:3,:3]
    cached = json.loads((ROOT/'reports/bonn_detect_20260914/yoloe_candidates.json').read_text())
    records = sorted([r for r in cached if r['scene']==protocol['scene'] and r['frame'] in protocol['frames']], key=lambda r:r['frame'])
    assert [r['frame'] for r in records] == protocol['frames']
    poses = {r['frame']:np.loadtxt(folder/f"pose/{r['frame']}.txt") for r in records}
    excluded = []
    for r in records:
        r['valid'] = []
        for j,d in enumerate(r['detections']):
            if 'world_aabb_lo' not in d or np.any(np.asarray(d['world_aabb_hi'])<=d['world_aabb_lo']):
                excluded.append([r['frame'],j])
            else:
                r['valid'].append({**d,'source':f"{r['frame']}:{j}"})
    traces, stats = {}, {}
    for arm in protocol['arms']:
        np.random.seed(0); torch.manual_seed(0)
        with (OUT/f'{arm}.log').open('w') as log, contextlib.redirect_stdout(log):
            backend = NativeBackend(cfg,K)
            people = CurrentPeople(cfg['dynamic_objects']['max_center_distance_m'])
            trace = []
            for r in records:
                frame = r['frame']; started = time.perf_counter()
                inputs = r['valid'] if arm=='native' else [d for d in r['valid'] if d['label']!='person']
                outputs = backend.step(inputs,frame,poses[frame])
                if arm=='bypass':
                    outputs += people.step([d for d in r['valid'] if d['label']=='person'],frame)
                torch.cuda.synchronize()
                trace.append({'frame':frame,'outputs':outputs,'backend_seconds':time.perf_counter()-started})
            stats[arm] = backend.stats()
        traces[arm] = trace
        (OUT/f'{arm}_trace.json').write_text(json.dumps(trace,allow_nan=False)+'\n')
        print(arm,'completed',len(trace),'frames',flush=True)
    # Load reference annotations only after both inference runs are finished.
    annotations = json.loads((OUT/'annotations.json').read_text())['frames']
    results = {}; details = {}
    for arm,trace in traces.items():
        details[arm] = []
        for state in trace:
            frame = state['frame']; predictions = []
            for row in state['outputs']:
                if row['label']=='person':
                    box = project(row,K,poses[frame])
                    if box is not None:
                        predictions.append({**row,'box':box})
            metrics = match_frame(predictions,annotations[str(frame)])
            details[arm].append({'frame':frame,**metrics,'projected_predictions':predictions})
        results[arm] = summarize(details[arm])
    # Frozen raw 2D proposals are a diagnostic reference, not another backend arm.
    raw_frames = []
    for r,state in zip(records,traces['bypass']):
        tracks = {p['source']:p['track_id'] for p in state['outputs'] if p['label']=='person'}
        predictions = [{**d,'box':d['box2d'],'track_id':tracks[d['source']]} for d in r['valid'] if d['label']=='person']
        raw_frames.append({'frame':r['frame'],**match_frame(predictions,annotations[str(r['frame'])])})
    results['raw_2d_diagnostic'] = summarize(raw_frames)
    artifact = {'summaries':results,'per_frame':details,'raw_2d_per_frame':raw_frames,
                'bypass_identity_diagnostic':identity_diagnostic(raw_frames),
                'backend_stats':stats,'excluded_candidates':excluded,
                'valid_candidates':sum(len(r['valid']) for r in records),
                'valid_person_candidates':sum(d['label']=='person' for r in records for d in r['valid']),
                'backend_seconds':{k:sum(r['backend_seconds'] for r in v) for k,v in traces.items()},
                'frozen_input_hashes_unchanged':all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items()),
                'model_forward_runs':0, 'training_runs':0,
                'gate_pass':results['bypass']['fn']<=results['native']['fn'] and results['bypass']['fp']<=results['native']['fp'],
                'limitations':['Assistant visual 2D proxy annotations, not human-verified or full 3D GT.',
                               'Two uncertain body contours retained; identity unknowns excluded only from ID diagnostic.',
                               'Cold-start 11-keyframe cache test; person-category routing, not general motion classification.',
                               'No end-to-end AP, static regression or real-time claim.']}
    (OUT/'results.json').write_text(json.dumps(artifact,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'summaries':results,'identity':artifact['bypass_identity_diagnostic'],'gate_pass':artifact['gate_pass']},indent=2))


if __name__=='__main__':
    main()
