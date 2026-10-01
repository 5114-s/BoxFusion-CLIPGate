import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from run_bonn_memory_gate import GuardedRecovery


def loop():
    cfg=json.loads((ROOT/'reports/bonn_identity_recovery_20260914/protocol.json').read_text())
    rule=json.loads((ROOT/'reports/bonn_memory_gate_20260914/protocol.json').read_text())
    return GuardedRecovery('admission_reject',cfg,rule)


def row(i,source,points=16):
    return dict(label='person',track_id=i,source=source,center=[0,0,2],observed_now=True,support_points=points)


def test_single_weak_fragment_cannot_recruit_an_identity():
    l=loop();h=np.ones(128)/128
    l.step(0,0,[row(0,'0:0',3)],{'0:0':h})
    out=l.step(1,.1,[row(1,'1:0')],{'1:0':h})
    assert out[0]['track_id']==1 and not l.pairs[-1]['memory_mature']


def test_confirmed_track_can_recover_but_new_episode_must_remature():
    l=loop();h=np.ones(128)/128;past=None
    for f in range(5):
        r=row(0,'0:0');r['observed_now']=f==0
        past=l.step(f,f*.03,[r],{'0:0':h})
    out=l.step(6,.18,[row(1,'6:0')],{'6:0':h})
    assert out[0]['track_id']==0 and l.events[-1]['event']=='recovered'
    assert l.profiles[0]['consecutive']==1 and past[0]['track_id']==0


def test_same_frame_observation_cannot_calibrate_its_recovery():
    l=loop();h=np.ones(128)/128;different=np.zeros(128);different[0]=1
    for f in range(5):
        r=row(0,'0:0');r['observed_now']=f==0
        l.step(f,f*.03,[r],{'0:0':h})
    out=l.step(6,.18,[row(1,'6:0')],{'6:0':different})
    assert out[0]['track_id']==1 and l.pairs[-1]['past_calibration_pairs']==0
    assert not l.calibration


def test_ambiguous_mature_memories_are_not_forced_to_match():
    l=loop();h=np.ones(128)/128
    for f in range(5):
        rows=[row(0,'0:0'),row(1,'0:1')]
        for r in rows:r['observed_now']=f==0
        l.step(f,f*.03,rows,{'0:0':h,'0:1':h})
    out=l.step(6,.18,[row(2,'6:0')],{'6:0':h})
    assert out[0]['track_id']==2


def test_current_positive_pair_cannot_relax_another_current_recovery():
    l=loop();h=np.zeros(128);h[0]=1.
    for f in range(5):
        rows=[row(0,'0:0'),row(1,'0:1')]
        for r in rows:r['observed_now']=f==0
        l.step(f,f*.03,rows,{'0:0':h,'0:1':h})
    def at_distance(distance):
        value=np.zeros(128);value[0]=(1-distance**2)**2;value[1]=1-value[0]
        return value
    l.calibration=[.2,.2]
    out=l.step(5,.15,[row(0,'5:0'),row(2,'5:1')],{'5:0':at_distance(.6),'5:1':at_distance(.4)})
    assert out[1]['track_id']==2
    assert l.pairs[-1]['past_calibration_pairs']==2
    assert l.pairs[-1]['appearance_limit']==.35
    assert l.appearance_limit()>.4
