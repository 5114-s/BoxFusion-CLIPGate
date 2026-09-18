import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_bonn_motion_pfo_probe import align_observation,box_corners,fit_box,project,transfer


def test_moving_both_box_and_virtual_camera_preserves_image_evidence():
    pose=np.eye(4);pose[:3,3]=[.2,-.1,.1]
    box=np.array([0.,0.,3.,.5,1.,.3]);R=np.eye(3);K=np.array([[500,0,320],[0,500,240],[0,0,1.]])
    shifted,virtual=align_observation(box,pose,np.array([.5,0,.1]))
    expected,_=project(box_corners(box,R),K,pose)
    actual,_=project(box_corners(shifted,R),K,virtual)
    wrong,_=project(box_corners(shifted,R),K,pose)
    np.testing.assert_allclose(actual,expected,atol=1e-10)
    assert np.max(abs(wrong-expected))>10


def test_point_transfer_recovers_translation_and_scale():
    source=np.array([0.,0.,3.,1.,2.,.5]);target=np.array([1.,0.,3.,2.,1.,.5])
    points=np.array([[.1,.4,3.1],[-.3,-.2,2.9]])
    expected=(points-source[:3])*np.array([2,.5,1])+target[:3]
    np.testing.assert_allclose(transfer(points,source,np.eye(3),target,np.eye(3)),expected)


def test_surface_box_preserves_given_axes_and_positive_extent():
    a=.4;R=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1.]])
    local=np.array([[-1,-2,0],[1,-2,0],[-1,2,0],[1,2,0]])
    box=fit_box(local@R.T+[0,0,3],R)
    np.testing.assert_allclose(box[:3],[0,0,3],atol=1e-12)
    assert box[-1]==.01 and np.all(box[3:]>0)
