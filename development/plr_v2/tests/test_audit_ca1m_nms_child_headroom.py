import importlib.util
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "tools/audit_ca1m_nms_child_headroom.py"
spec = importlib.util.spec_from_file_location("ca1m_child_audit", SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def cube(low=(0, 0, 0), high=(1, 1, 1)):
    return np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])], dtype=float)


def record(keyframe=1, child_frame=1):
    return {"type": "nms_merge", "scene_id": "1", "keyframe_id": keyframe, "parent_frame_id": 0, "child_frame_id": child_frame, "parent_init_id": 10, "child_init_id": 20, "parent_corners_world": cube((3, 3, 3), (4, 4, 4)).tolist(), "child_corners_world": cube().tolist()}


def scene_files(tmp_path, events, gt=None):
    diag, pred, root = (tmp_path / name for name in ("diag", "pred", "data"))
    for directory in (diag, pred, root / "1"):
        directory.mkdir(parents=True)
    (diag / "1_pvq_ar_summary.json").write_text(json.dumps({"scene_id": "1", "mode": "shadow", "nms_observer": True, "nms_records": len(events), "nms_record_cap": 100, "nms_record_cap_hit": False}))
    if events:
        (diag / "1_pvq_nms.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    with (pred / "1_boxes.pkl").open("wb") as handle:
        pickle.dump([[]], handle)
    np.save(root / "1/after_filter_boxes.npy", np.array([cube()] if gt is None else gt))
    return SimpleNamespace(diagnostics_root=diag, baseline_root=pred, data_root=root)


def test_repeated_child_does_not_become_multiview_support(tmp_path):
    args = scene_files(tmp_path, [record(k) for k in (1, 2, 3)])
    result = audit.audit_scene("1", args)
    total = audit.aggregate([result])["0.50"]
    assert total["additional_any_child_covered_gt"] == 1
    assert total["different_parent_uncovered_gt_two_child_frames"] == 0
    support = result["thresholds"]["0.50"]["gt"][0]["different_parent_missing_support"]
    assert support["events"] == 3
    assert support["distinct_emit_keyframes"] == 3
    assert support["distinct_child_frames"] == 1
    assert support["unique_child_observations"] == 1


def test_strict_threshold_and_coverage_is_not_one_to_one(tmp_path):
    # Identical GT duplicates are intentionally both covered by one child.
    args = scene_files(tmp_path, [record()], [cube(), cube(), cube(high=(2, 1, 1))])
    result = audit.audit_scene("1", args)
    totals = audit.aggregate([result])
    assert totals["0.50"]["additional_any_child_covered_gt"] == 2
    assert totals["0.25"]["additional_any_child_covered_gt"] == 3
    assert result["thresholds"]["0.50"]["gt"][2]["child_covered"] is False


def test_matching_parent_does_not_count_as_distinct_instance_loss(tmp_path):
    event = record()
    event["parent_corners_world"] = cube().tolist()
    result = audit.audit_scene("1", scene_files(tmp_path, [event]))
    total = audit.aggregate([result])["0.50"]
    assert total["additional_any_child_covered_gt"] == 1
    assert total["different_parent_uncovered_gt"] == 0
    assert result["thresholds"]["0.50"]["event_partition_best_gt"]["same_best_gt"] == 1


def test_zero_count_proves_missing_ledger_is_empty(tmp_path):
    result = audit.audit_scene("1", scene_files(tmp_path, []))
    assert result["nms_records"] == 0
    assert result["nms_ledger_sha256"] is None


@pytest.mark.parametrize("malformation", ["missing", "truncated", "count_mismatch", "wrong_scene", "future_frame"])
def test_invalid_ledgers_fail_closed(tmp_path, malformation):
    args = scene_files(tmp_path, [record()])
    path = args.diagnostics_root / "1_pvq_nms.jsonl"
    if malformation == "missing":
        path.unlink()
    elif malformation == "truncated":
        path.write_text(path.read_text().rstrip())
    elif malformation == "count_mismatch":
        path.write_text("")
    else:
        event = record()
        if malformation == "wrong_scene":
            event["scene_id"] = "2"
        else:
            event["child_frame_id"] = 2
        path.write_text(json.dumps(event) + "\n")
    with pytest.raises(ValueError):
        audit.audit_scene("1", args)


@pytest.mark.parametrize("change", [{"nms_record_cap_hit": True}, {"nms_records": 100}, {"nms_records": None}, {"nms_observer": False}])
def test_invalid_summary_fails_closed(tmp_path, change):
    args = scene_files(tmp_path, [])
    path = args.diagnostics_root / "1_pvq_ar_summary.json"
    summary = json.loads(path.read_text())
    summary.update(change)
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        audit.audit_scene("1", args)
