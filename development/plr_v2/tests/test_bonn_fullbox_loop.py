import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from run_bonn_fullbox_loop import assign, box_row, translate, guarded_geometry, BoxLoop
import run_bonn_fullbox_loop as replay
import json


def test_full_box_translation_preserves_pairwise_geometry():
    # Applies equally to an oriented cuboid, not just a world-axis box.
    corners=np.arange(24,dtype=float).reshape(8,3)/10
    row={'center':corners.mean(0).tolist(),'corners':corners.tolist()}
    delta=np.array([.3,-.2,.8]);moved=translate(row,delta)
    np.testing.assert_allclose(np.asarray(moved['corners'])-moved['center'],corners-row['center'],atol=1e-12)
    np.testing.assert_allclose(row['corners'],corners)


def test_one_to_one_assignment_and_ambiguous_memory_rejection():
    assert len(assign([[.1,.2],[.2,np.inf]]))==2
    assert len(assign([[.1],[.2]]))==1
    assert assign([[.1,.12]],margin=.1)==[]
    assert assign([[.1,1.]],margin=.1)==[(0,0)]


def test_clipped_redetection_cannot_collapse_existing_shape():
    cfg=json.loads((ROOT/'reports/bonn_fullbox_loop_20260914/protocol.json').read_text())
    old=box_row(dict(world_aabb_lo=[0,0,0],world_aabb_hi=[1,2,1],score=.9,source='0:0'),0)
    d=dict(world_aabb_lo=[0,0,0],world_aabb_hi=[1,.5,1],score=.9,source='25:0',box2d=[0,10,200,200])
    out,audit=guarded_geometry(old,box_row(d,0),d,128,cfg)
    assert not audit['accepted'];np.testing.assert_allclose(out['corners'],old['corners'])


def test_memory_does_not_emit_after_flow_loss_but_current_observation_does():
    cfg=json.loads((ROOT/'reports/bonn_fullbox_loop_20260914/protocol.json').read_text())
    loop=BoxLoop('flow',cfg,{},np.eye(3))
    tr={'last_detection_frame':10,'flow':{'active':False}}
    assert loop.emit(tr,10,0)
    assert not loop.emit(tr,11,.03)


def test_revalidated_memory_resets_geometry_and_past_output_is_immutable(monkeypatch):
    import copy
    cfg=json.loads((ROOT/'reports/bonn_fullbox_loop_20260914/protocol.json').read_text())
    fc={'min_points':8}
    loop=BoxLoop('flow',cfg,fc,np.eye(3))
    flow={'uv':np.tile([10.,10.],(12,1)),'world':np.tile([0.,0.,2.],(12,1)),
          'anchor':np.array([0.,0.,2.]),'ids':np.arange(12),'active':True}
    monkeypatch.setattr(replay,'seed_detection',lambda *args:(copy.deepcopy(flow),np.ones(128)/128))
    d=dict(world_aabb_lo=[-.5,-1,1.5],world_aabb_hi=[.5,1,2.5],score=.9,source='0:0',box2d=[0,0,100,100])
    image=np.zeros((20,20),np.uint8);pose=np.eye(4)
    before=loop.step(0,0,None,image,None,None,pose,[d]);saved=copy.deepcopy(before)
    loop.tracks[0]['flow']['active']=False
    d2={**d,'world_aabb_lo':[-.2,-1,1.5],'world_aabb_hi':[.8,1,2.5],'source':'1:0'}
    after=loop.step(1,.03,image,image,None,None,pose,[d2])
    assert loop.events[-1]['event']=='memory_revalidated'
    assert after['latest'][0]['track_id']==after['guarded'][0]['track_id']==0
    np.testing.assert_allclose(after['guarded'][0]['center'],[.3,0,2])
    assert before==saved
