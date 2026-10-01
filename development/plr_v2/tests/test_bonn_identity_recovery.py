import copy
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from run_bonn_identity_recovery import Recovery


def cfg():return json.loads((ROOT/'reports/bonn_identity_recovery_20260914/protocol.json').read_text())


def row(track,source,x=0):
    return dict(label='person',track_id=track,source=source,center=[x,0,2],observed_now=True,score=.9,corners=[[x,0,2]]*8)


def test_recovery_preserves_geometry_and_does_not_relabel_past():
    loop=Recovery('appearance_motion',cfg());h=np.ones(128)/128
    past=loop.step(0,0,[row(3,'0:0')],{'0:0':h});saved=copy.deepcopy(past)
    loop.step(1,.03,[],{})
    current=row(5,'2:0',.2);out=loop.step(2,.07,[current],{'2:0':h})
    assert out[0]['track_id']==past[0]['track_id'];assert loop.events[-1]['event']=='recovered'
    assert out[0]['corners']==current['corners'];assert past==saved


def test_live_identity_cannot_be_stolen_by_a_new_tracklet():
    loop=Recovery('appearance_motion',cfg());h=np.ones(128)/128
    loop.step(0,0,[row(0,'0:0')],{'0:0':h})
    old={**row(0,'0:0'),'observed_now':False}
    out=loop.step(1,.03,[old,row(1,'1:0',.1)],{'1:0':h})
    assert len({r['track_id'] for r in out})==2


def test_equal_appearance_and_motion_is_rejected_as_ambiguous():
    loop=Recovery('appearance_motion',cfg());h=np.ones(128)/128
    loop.step(0,0,[row(0,'0:0',-.1),row(1,'0:1',.1)],{'0:0':h,'0:1':h})
    out=loop.step(1,.1,[row(2,'1:0')],{'1:0':h})
    assert out[0]['track_id']==2;assert loop.events[-1]['event']=='new_identity'


def test_missing_appearance_or_expired_memory_cannot_confirm_identity():
    for feature,age in [(None,.1),(np.ones(128)/128,3.1)]:
        loop=Recovery('appearance_motion',cfg());h=np.ones(128)/128
        loop.step(0,0,[row(0,'0:0')],{'0:0':h})
        out=loop.step(1,age,[row(1,'1:0')],{'1:0':feature})
        assert out[0]['track_id']==1


def test_identical_appearance_cannot_override_unreachable_world_motion():
    loop=Recovery('appearance_motion',cfg());h=np.ones(128)/128
    loop.step(0,0,[row(0,'0:0')],{'0:0':h})
    out=loop.step(1,.1,[row(1,'1:0',2.)],{'1:0':h})
    assert out[0]['track_id']==1
    assert not loop.pair_audit[-1]['reachable']
