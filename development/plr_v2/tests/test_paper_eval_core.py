import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from paper_eval_core import subgroup_ap, corners_from_center_size, best_view
from true_fusion_audit_core import class_agnostic_ap


def test_ignore_correct_outgroup_and_duplicates():
    g=corners_from_center_size([[0,0,2,1,1,1],[3,0,2,1,1,1]])
    p={'s':(g[[1,1,0]],np.array([.9,.8,.7]))}
    a=subgroup_ap(p,{'s':g},{'s':np.array([True,False])},.5)
    assert a['tp']==1 and a['fp']==0 and a['ignored']==2 and a['ap']>99.99


def test_allgroup_anchor_and_negative_scene():
    g=corners_from_center_size([[0,0,2,1,1,1]])
    p={'s':(g,np.array([.7])),'negative':(g,np.array([.9]))}
    gt={'s':g,'negative':np.empty((0,8,3))}
    a=subgroup_ap(p,gt,{s:np.ones(len(v),bool) for s,v in gt.items()},.5)
    assert a['fp']==1 and a['ap']==class_agnostic_ap(p,gt,.5)['ap']


def test_projection_abstention_preserves_rows():
    b=corners_from_center_size([[0,0,-3,1,1,1],[0,0,3,1,1,1]])
    k=np.array([[100,0,50],[0,100,50],[0,0,1]])
    ids,rect=best_view(b,[np.eye(4)],k,100,100)
    assert ids.tolist()==[-1,0] and (rect[1,2:]>rect[1,:2]).all()
