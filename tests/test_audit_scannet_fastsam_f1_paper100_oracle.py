from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import tools.audit_scannet_fastsam_f1_paper100_oracle as oracle
from tools.audit_scannet_boxer_unexplained_oracle import official_constant_evaluate
from tools.audit_scannet_fastsam_f1_paper100_oracle import (
    F1OracleError,
    FastSAMCandidate,
    _validate_output_path,
    candidate_q02_q98_aligned_minmax,
    canonical_ordered_hash_ledger,
    evaluate_f1_threshold,
)


def _candidate_row(
    q02=(0.0, 0.0, 0.0), q98=(2.0, 1.0, 1.0)
) -> dict[str, object]:
    lower = np.asarray(q02, dtype=np.float64)
    upper = np.asarray(q98, dtype=np.float64)
    return {
        "confidence": 0.75,
        "mask_sha256": "a" * 64,
        "pixel_count": 100,
        "points_and_voxel_keys_sha256": "b" * 64,
        "rank": 0,
        "raw_index": 3,
        "residual_pixel_count": 80,
        "residual_ratio": 0.8,
        "stored_point_count": 32,
        "support_pixel_count": 90,
        "tight_box_xyxy": [1, 2, 20, 30],
        "valid_pixel_count": 95,
        "valid_ratio": 0.95,
        "voxel_count": 32,
        "world_center": ((lower + upper) / 2.0).tolist(),
        "world_extent": (upper - lower).tolist(),
        "world_q02": lower.tolist(),
        "world_q98": upper.tolist(),
    }


def test_q02_q98_uses_all_eight_corners_before_axis_alignment():
    row = _candidate_row()
    angle = np.pi / 4.0
    alignment = np.eye(4)
    alignment[:2, :2] = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    alignment[:3, 3] = [4.0, -3.0, 2.0]
    actual = candidate_q02_q98_aligned_minmax(row, alignment)
    q02 = np.asarray(row["world_q02"])
    q98 = np.asarray(row["world_q98"])
    corners = np.asarray(
        [[x, y, z] for x in (q02[0], q98[0]) for y in (q02[1], q98[1]) for z in (q02[2], q98[2])]
    )
    transformed = corners @ alignment[:3, :3].T + alignment[:3, 3]
    expected = np.concatenate((transformed.min(0), transformed.max(0)))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.pop("world_q02"),
        lambda row: row.__setitem__("world_q98", [0.0, 1.0, 1.0]),
        lambda row: row.__setitem__("world_center", [99.0, 0.5, 0.5]),
        lambda row: row.__setitem__("world_extent", [99.0, 1.0, 1.0]),
        lambda row: row.__setitem__("world_q02", [float("nan"), 0.0, 0.0]),
    ],
)
def test_candidate_schema_and_geometry_fail_closed(mutation):
    row = _candidate_row()
    mutation(row)
    with pytest.raises(F1OracleError):
        candidate_q02_q98_aligned_minmax(row, np.eye(4))


def _candidate(index: int) -> FastSAMCandidate:
    return FastSAMCandidate(
        scene_id="scene0001_00",
        frame_id=0,
        frame_ordinal=0,
        candidate_index=index,
        rank=index,
        raw_index=index,
        aligned_minmax=np.zeros(6),
    )


def test_threshold_report_has_strict_edges_maximum_matching_and_suffix():
    native = [np.asarray([[0.9, 0.0, 0.0]])]
    # Candidate 0 can use GT1 or GT2, candidate 1 can only use GT1.  The
    # augmenting path is required for candidate maximum matching == 2.
    candidate = [np.asarray([[0.0, 0.8, 0.7], [0.0, 0.6, 0.50]])]
    baseline = official_constant_evaluate(native, [3], 0.50)
    report = evaluate_f1_threshold(
        scenes=["scene0001_00"],
        native_iou=native,
        candidate_iou=candidate,
        candidates=[[_candidate(0), _candidate(1)]],
        gt_counts=[3],
        baseline_evaluation=baseline,
        threshold=0.50,
    )
    assert report["native_maximum_matching_count"] == 1
    assert report["candidate_maximum_matching_count"] == 2
    assert report["union_maximum_matching_count"] == 3
    assert report["additional_union_matching_over_native"] == 2
    assert report["gt_selected_candidate_suffix"]["selected_candidate_count"] == 2
    # The exactly-0.50 edge was not needed and is never accepted.
    assert report["strict_iou_comparison"] == ">"


def test_ordered_hash_ledger_is_compact_scene_ordered_json(tmp_path):
    paths = [tmp_path / "b", tmp_path / "a"]
    paths[0].write_bytes(b"first")
    paths[1].write_bytes(b"second")
    scenes = ["scene0002_00", "scene0001_00"]
    report = canonical_ordered_hash_ledger(scenes, paths, "fixture")
    entries = [
        [scene, hashlib.sha256(path.read_bytes()).hexdigest()]
        for scene, path in zip(scenes, paths)
    ]
    expected = hashlib.sha256(
        json.dumps(entries, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    assert report == {"entries": entries, "sha256": expected}


def test_output_is_create_only_and_outside_inputs(tmp_path):
    protected = tmp_path / "inputs"
    protected.mkdir()
    with pytest.raises(F1OracleError, match="protected"):
        _validate_output_path(protected / "report.json", [protected])
    existing = tmp_path / "existing.json"
    existing.write_text("sealed", encoding="utf-8")
    with pytest.raises(F1OracleError, match="overwrite"):
        _validate_output_path(existing, [protected])
    with pytest.raises(F1OracleError, match="suffix"):
        _validate_output_path(tmp_path / "report.txt", [protected])


def test_main_writes_once_and_refuses_overwrite(tmp_path, monkeypatch):
    out = tmp_path / "reports" / "f1.json"
    fake_report = {
        "schema": oracle.SCHEMA,
        "totals": {"scene_count": 100},
        "decision": {"overall_pass": False},
    }
    monkeypatch.setattr(
        oracle, "audit_scannet_fastsam_f1_paper100_oracle", lambda **_: fake_report
    )
    assert oracle.main(["--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == fake_report
    before = out.read_bytes()
    with pytest.raises(F1OracleError, match="overwrite"):
        oracle.main(["--out", str(out)])
    assert out.read_bytes() == before
