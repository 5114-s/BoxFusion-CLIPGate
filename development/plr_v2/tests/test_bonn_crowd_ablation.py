import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('crowd', Path(__file__).resolve().parents[1]/'tools/run_bonn_crowd_ablation.py')
crowd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(crowd)


def test_duplicate_is_not_any_unmatched_output():
    reference = {'people':[{'box':[0,0,10,10],'identity':'P1'}], 'ignore_regions':[]}
    predictions = [{'box':b} for b in ([0,0,10,10],[0,0,10,10],[30,30,40,40])]
    result = crowd.match_frame(predictions,reference)
    assert (result['tp'],result['fn'],result['duplicates'],result['other_fp']) == (1,0,1,1)


def test_assignment_maximizes_valid_coverage(monkeypatch):
    # A naive max-IoU assignment would prefer .99 + .49 to .51 + .51,
    # falsely turning two valid matches into one after thresholding.
    values = [[.99,.51],[.51,.49]]
    monkeypatch.setattr(crowd,'overlap',lambda a,b,*args: values[a][b])
    reference = {'people':[{'box':i,'identity':str(i)} for i in range(2)],'ignore_regions':[]}
    result = crowd.match_frame([{'box':i} for i in range(2)],reference)
    assert result['tp']==2 and result['fn']==0


def test_missing_frame_and_explicit_fragment_ignore():
    reference = {'people':[{'box':[0,0,10,10],'identity':'P1'}], 'ignore_regions':[[30,30,40,40]]}
    assert crowd.match_frame([],reference)['fn']==1
    result = crowd.match_frame([{'box':[30,30,40,40]}],reference)
    assert result['ignored']==1 and result['fn']==1 and result['fp']==0


def test_identity_gap_is_not_physical_absence_and_unknown_id_is_excluded():
    def frame(f,p,t):
        return {'frame':f,'matches':[{'identity':p,'track_id':t}]}
    result = crowd.identity_diagnostic([frame(0,'P1',0),{'frame':25,'matches':[]},frame(50,'P1',1),frame(75,None,4)])
    assert result['matched_identity_transitions']==1 and result['switches']==1
