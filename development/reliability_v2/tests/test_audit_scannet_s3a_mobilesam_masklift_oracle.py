from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.audit_scannet_s3a_mobilesam_masklift_oracle import (
    CONTINUATION_MIN_MATCHES,
    S3aMaskliftOracleError,
    _abstention_report,
    _frozen_topk_native_hashes,
    _geometry_report,
    _per_row_changes,
    _write_json_create_only,
    audit_scannet_s3a_mobilesam_masklift_oracle,
)


SCENE = "scene0000_00"


def test_synthetic_identical_membership_reports_mm_union_and_plus3_gate():
    # Native owns GT0.  The same four frozen rows are supplied to the candidate
    # geometry; three rows independently recover GT1..GT3.  This is a geometry
    # oracle only—there is no ranking, suffix, score, or AP construction here.
    native = np.asarray([[0.90, 0.0, 0.0, 0.0]], dtype=np.float64)
    candidate = np.asarray(
        [
            [0.80, 0.0, 0.0, 0.0],
            [0.0, 0.80, 0.0, 0.0],
            [0.0, 0.0, 0.80, 0.0],
            [0.0, 0.0, 0.0, 0.80],
        ],
        dtype=np.float64,
    )
    report = _geometry_report(
        scenes=(SCENE,),
        candidate_iou=(candidate,),
        candidate_global_rows=(np.arange(4, dtype=np.int64),),
        baseline_iou=(native,),
        total_gt=4,
    )
    assert report["candidate_count"] == 4
    for threshold in ("0.15", "0.25", "0.50"):
        row = report["per_threshold"][threshold]
        assert row["candidate_maximum_matching_count"] == 4
        assert row["native_maximum_matching_count"] == 1
        assert row["native_union_maximum_matching_count"] == 4
        assert row["additional_union_matching_over_native"] == 3
        assert row["continuation_required_additional_matches"] == 3
        assert row["passes_plus3_continuation_gate"] is True


def test_synthetic_strict_boundaries_crossings_and_abstentions():
    # Values equal to a threshold must remain below because the frozen protocol
    # is strict IoU > threshold.
    native = np.asarray([[0.9, 0.0, 0.0, 0.0]], dtype=np.float64)
    boundary = np.asarray(
        [
            [0.0, 0.15, 0.0, 0.0],
            [0.0, 0.0, 0.25, 0.0],
            [0.0, 0.0, 0.0, 0.50],
            [0.80, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    report = _geometry_report(
        scenes=(SCENE,),
        candidate_iou=(boundary,),
        candidate_global_rows=(np.arange(4, dtype=np.int64),),
        baseline_iou=(native,),
        total_gt=4,
    )
    assert report["per_threshold"]["0.15"]["candidate_maximum_matching_count"] == 3
    assert report["per_threshold"]["0.25"]["candidate_maximum_matching_count"] == 2
    assert report["per_threshold"]["0.50"]["candidate_maximum_matching_count"] == 1

    arrays = {
        "scene_index": np.zeros(4, dtype=np.int16),
        "frame_id": np.arange(4, dtype=np.int64),
        "sealed_npz_row": np.arange(4, dtype=np.int64),
        "boxer_source_row": np.arange(4, dtype=np.int32),
        "accepted": np.asarray([True, True, True, False]),
        "diagnostic_box_valid": np.ones(4, dtype=bool),
        "abstention_code": np.asarray([0, 0, 0, 4], dtype=np.int8),
    }
    raw = np.diag([0.10, 0.30, 0.60, 0.20]).astype(np.float64)
    primary = np.asarray(
        [[0.20, 0.0, 0.0, 0.0], [0.0, 0.20, 0.0, 0.0], [0.0, 0.0, 0.70, 0.0]],
        dtype=np.float64,
    )
    diagnostic = np.diag([0.20, 0.40, 0.40, 0.60]).astype(np.float64)
    changes = _per_row_changes(
        scenes=(SCENE,),
        arrays=arrays,
        raw_iou=(raw,),
        primary_iou=(primary,),
        diagnostic_iou=(diagnostic,),
        primary_rows=(np.asarray([0, 1, 2]),),
        diagnostic_rows=(np.arange(4),),
    )
    at_025 = changes["strict_threshold_crossings"]["0.25"]
    assert at_025["primary_q02_q98"]["loss_crossing"] == 1
    assert at_025["primary_q02_q98"]["invalid"] == 1
    assert at_025["diagnostic_q00_q100"]["gain_crossing"] == 1
    assert changes["rows"][3]["primary_best_iou"] is None
    assert changes["rows"][3]["diagnostic_best_iou"] == pytest.approx(0.6)

    abstentions = _abstention_report(
        arrays, (SCENE,), {0: "emitted_q02_q98", 4: "too_few_clean_voxels"}
    )
    assert abstentions["overall"]["abstained_count"] == 1
    assert abstentions["overall"]["abstention_rate"] == pytest.approx(0.25)


def test_topk_receipt_is_the_native_trust_anchor():
    hashes = {SCENE: "a" * 64}
    receipt = {
        "input_sha256_before": {"scenes": {SCENE: {"baseline": hashes[SCENE]}}},
        "input_sha256_after": {"scenes": {SCENE: {"baseline": hashes[SCENE]}}},
    }
    assert _frozen_topk_native_hashes(receipt, (SCENE,)) == hashes
    receipt["input_sha256_after"]["scenes"][SCENE]["baseline"] = "b" * 64
    with pytest.raises(S3aMaskliftOracleError, match="Top-K input hash identity"):
        _frozen_topk_native_hashes(receipt, (SCENE,))


def test_known_wrong_native_root_fails_before_any_gt_access_if_assets_exist(tmp_path):
    root = Path(__file__).resolve().parents[1]
    raw_root = root / "logs/scannet_boxer_unexplained_shadow_clean_in2_v5_score05/sealed"
    topk = root / "logs/scannet_boxer_per_view_topk_raw_ceiling_score05_dev3_v5.json"
    wrong_native = root / "results/scannet_graw_e2_replay1_score05"
    required = [
        raw_root / "boxer_shadow_candidates.json",
        raw_root / "boxer_shadow_candidates.npz",
        topk,
        wrong_native / "scene0568_00_boxes.pkl",
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("sealed production provenance assets are not present")

    # Deliberately nonexistent S3a/GT/scan paths prove the native cross-bind is
    # checked before the sidecar or any annotation can be opened.
    with pytest.raises(
        S3aMaskliftOracleError,
        match="baseline does not match frozen T05 receipt",
    ):
        audit_scannet_s3a_mobilesam_masklift_oracle(
            s3a_json=tmp_path / "not-opened.json",
            s3a_npz=tmp_path / "not-opened.npz",
            raw_boxer_json=raw_root / "boxer_shadow_candidates.json",
            raw_boxer_npz=raw_root / "boxer_shadow_candidates.npz",
            topk_receipt=topk,
            preregistration=root / "docs/S3_FROZEN_PROPOSAL_SOURCE_AUDIT.md",
            baseline_root=wrong_native,
            gt_root=tmp_path / "no-gt",
            scan_root=tmp_path / "no-scans",
        )


def test_report_writer_is_create_only(tmp_path):
    output = tmp_path / "report.json"
    payload = {"posthoc_dev_diagnostic": True, "not_deployable": True}
    _write_json_create_only(output, payload)
    before = output.read_bytes()
    with pytest.raises(S3aMaskliftOracleError, match="refusing to overwrite"):
        _write_json_create_only(output, payload)
    assert output.read_bytes() == before
    assert CONTINUATION_MIN_MATCHES == 3
