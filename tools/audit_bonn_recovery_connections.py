#!/usr/bin/env python3
"""Evaluate supplemental visual identity review; never rerun or change inference."""
import hashlib
import json
from pathlib import Path

from run_bonn_identity_recovery import evaluate
from run_bonn_fullbox_loop import identity_purity

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/bonn_recovery_review_20260914'
RUN=ROOT/'reports/bonn_identity_recovery_20260914'
BASE=ROOT/'reports/bonn_fullbox_loop_20260914'


def main():
    review=json.loads((OUT/'visual_review.json').read_text())
    paths=[OUT/'visual_review.json',Path(__file__),RUN/'results.json',RUN/'events.json',
           RUN/'motion_only_trace.json',RUN/'appearance_motion_trace.json',BASE/'flow_latest_trace.json',
           ROOT/'reports/bonn_crowd_native_dynamic_20260914/annotations.json',
           ROOT/'reports/bonn_detect_20260914/yoloe_candidates.json']
    paths += [ROOT/f'data_bonn/scene0002_01/frames/color/{f}.jpg' for f in review['viewed_frames']]
    paths += [ROOT/p/'manifest.json' for p in ['data_dyn','data_dyn_move','data_dyn_walk','data_dyn_walk2','data_ca1m_dyn']]
    paths += [ROOT/f'data_bonn_dl/{p}/groundtruth.txt' for p in ['rgbd_bonn_crowd','rgbd_bonn_person_tracking']]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    old=json.loads((RUN/'results.json').read_text());events=json.loads((RUN/'events.json').read_text())
    records={r['frame']:r for r in json.loads(paths[8].read_text()) if r['scene']=='scene0002_01' and 500<=r['frame']<=750}
    reference=json.loads(paths[7].read_text())['frames']
    traces={'existing_flow':json.loads(paths[6].read_text()),'motion_only':json.loads(paths[4].read_text()),
            'appearance_motion':json.loads(paths[5].read_text())}
    _,truth=evaluate(traces['existing_flow'],records,reference)
    truth={s:p for s,p in truth.items() if p is not None};assert len(truth)==16
    for source,person in review['source_identity_additions'].items():
        assert source not in truth
        truth[source]=person
    assert len(truth)==20
    results={}
    for arm,trace in traces.items():
        source_ids={r['source']:r['track_id'] for state in trace for r in state['outputs']
                    if r['label']=='person' and r['observed_now']}
        assert set(source_ids)==set(truth)
        purity=identity_purity([{'matches':[{'identity':truth[s],'track_id':k} for s,k in source_ids.items()]}])
        checked=[]
        for e in events.get(arm,[]):
            if e['event']!='recovered':continue
            prior={truth[s] for s in e['old_sources'] if s in truth};current=truth.get(e['new_source'])
            verdict='unknown' if current is None or not prior else 'correct' if prior=={current} else 'wrong'
            checked.append({**e,'old_visual_identities':sorted(prior),'new_visual_identity':current,'review_verdict':verdict})
        counts={k:sum(e['review_verdict']==k for e in checked) for k in ['correct','wrong','unknown']}
        results[arm]={'original_sparse_continuity':old['arms'][arm]['continuity'],
                      'supplemented_candidate_identity_purity':purity,'reviewed_recovery_counts':counts,'reviewed_recoveries':checked}
    for expected in review['connections']:
        actual=next(e for e in results['motion_only']['reviewed_recoveries'] if e['new_source']==expected['new_source'])
        assert actual['review_verdict']==expected['review_verdict']
    artifact={'review_type':review['reviewer'],'candidate_identity_labels':20,'added_candidate_identity_labels':4,
              'arms':results,'inference_reruns':0,'training_runs':0,'gpu_runs':0,
              'independent_human_gt':False,'formal_3d_ap_available':False,
              'decision':'motion_only has two visually identified cross-person recoveries and is not a validated safe recovery configuration; keep as a comparison. appearance_motion remains a conservative partial recovery (one correct). Neither fully solves dynamic detection.'}
    (OUT/'results.json').write_text(json.dumps(artifact,indent=2)+'\n')
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    (OUT/'audit.json').write_text(json.dumps({'input_sha256':hashes,'frozen_inputs_unchanged':True,
         'original_annotations_predictions_and_results_unchanged':True,'review_counts_match_visual_decisions':True},indent=2)+'\n')
    print(json.dumps({a:{'recovery':v['reviewed_recovery_counts'],
                        'mixed_tracks':v['supplemented_candidate_identity_purity']['tracks_containing_multiple_people'],
                        'purity':v['supplemented_candidate_identity_purity']['weighted_track_purity']} for a,v in results.items()},indent=2))


if __name__=='__main__':main()
