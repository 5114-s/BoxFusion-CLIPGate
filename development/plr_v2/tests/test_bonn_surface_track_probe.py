import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('probe',Path(__file__).resolve().parents[1]/'tools/run_bonn_surface_track_probe.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def test_camera_motion_does_not_become_object_motion():
    K=np.array([[400,0,320],[0,400,240],[0,0,1.]])
    points=np.array([[0,0,3],[.1,.2,3.3],[-.2,0,2.7]])
    pose=np.eye(4);angle=.2
    pose[:3,:3]=[[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]]
    pose[:3,3]=[.3,-.1,.2]
    uv,z=probe.project(points,K,pose)
    np.testing.assert_allclose(probe.lift(uv,z,K,pose),points,atol=1e-12)


def test_invalid_depth_and_offscreen_points_are_not_evidence():
    depth=np.ones((5,5));depth[2,2]=0
    _,valid=probe.sample_depth(depth,np.array([[2,2],[-2,1],[1,1],[np.nan,1]]))
    assert valid.tolist()==[False,False,True,False]


def test_motion_requires_support_and_rejects_minority_outliers():
    cfg={'min_points':8,'translation_residual_floor_m':.03,'robust_mad_multiplier':3.,'min_motion_inlier_fraction':.5}
    shift=np.tile([.02,0,0],(12,1));shift[-2:]=[1,1,1]
    delta,keep=probe.robust_translation(shift,cfg)
    np.testing.assert_allclose(delta,[.02,0,0]);assert keep.sum()==10
    assert probe.robust_translation(shift[:4],cfg)[0] is None
