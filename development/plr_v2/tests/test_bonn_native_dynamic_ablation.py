"""Current-state semantics; integration evidence comes from the saved native run."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from run_bonn_native_dynamic_ablation import CurrentPeople, project


def observation(x, source):
    return {'world_aabb_lo':[x-.2,-.8,2.8], 'world_aabb_hi':[x+.2,.8,3.2],
            'score':.8,'label':'person','source':source}


def test_current_geometry_replaces_history_without_averaging():
    tracker=CurrentPeople()
    first=tracker.step([observation(0.,'0:0')],0)[0]
    current=tracker.step([observation(.9,'25:0')],25)[0]
    assert first['track_id']==current['track_id']
    np.testing.assert_allclose(current['center'],[.9,0.,3.])
    assert current['lineage']==['25:0']


def test_missing_observation_keeps_memory_but_emits_no_current_box():
    tracker=CurrentPeople()
    before=tracker.step([observation(0.,'0:0')],0)[0]
    assert tracker.step([],25)==[]
    assert tracker.memory[before['track_id']]['state']=='unobserved'
    after=tracker.step([observation(.1,'50:0')],50)[0]
    assert after['track_id']==before['track_id']


def test_two_detections_cannot_share_one_identity():
    tracker=CurrentPeople()
    tracker.step([observation(0.,'0:0')],0)
    current=tracker.step([observation(.1,'25:0'),observation(.2,'25:1')],25)
    assert len({x['track_id'] for x in current})==2


def test_projection_uses_current_camera_pose():
    tracker=CurrentPeople()
    row=tracker.step([observation(0.,'0:0')],0)[0]
    K=np.array([[500.,0.,320.],[0.,500.,240.],[0.,0.,1.]])
    pose=np.eye(4)
    assert project(row,K,pose) is not None
    pose[0,3]=100.
    assert project(row,K,pose) is None
