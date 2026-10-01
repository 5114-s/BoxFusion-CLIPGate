from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pytest

from tools.audit_tr3d_r3_shadow_active import (
    RawPrediction,
    _geometry_equal,
    _metric_exact,
    _write_create_only,
    audit_prediction_pair,
    load_raw_prediction,
    primary_expected_rows,
)


def _corners(center: float) -> np.ndarray:
    x0 = np.float32(center - 0.5)
    x1 = np.float32(center + 0.5)
    return np.asarray(
        [
            [x0, 0, 0],
            [x0, 0, 1],
            [x0, 1, 0],
            [x0, 1, 1],
            [x1, 0, 0],
            [x1, 0, 1],
            [x1, 1, 0],
            [x1, 1, 1],
        ],
        dtype=np.float32,
    )


def _prediction(*centers: float) -> RawPrediction:
    return RawPrediction(
        labels=tuple(range(len(centers))),
        corners=tuple(_corners(value) for value in centers),
        scores=tuple(float(0.9 - index * 0.1) for index in range(len(centers))),
    )


def test_load_raw_prediction_preserves_float32_geometry(tmp_path: Path) -> None:
    path = tmp_path / "scene0000_00_boxes.pkl"
    rows = [(0, _corners(0.0), 0.75), (1, _corners(2.0), 0.5)]
    with path.open("wb") as handle:
        pickle.dump([rows], handle)
    loaded = load_raw_prediction(path)
    assert loaded.count == 2
    assert loaded.labels == (0, 1)
    assert loaded.scores == (0.75, 0.5)
    assert loaded.corners[0].dtype == np.float32
    assert loaded.corners[0].tobytes() == rows[0][1].tobytes()


def test_primary_rule_has_stable_proposal_tie_and_strict_score_gate() -> None:
    result = primary_expected_rows(
        proposal_ids=np.asarray([12, 11, 20], dtype=np.int64),
        anchor_indices=np.asarray([0, 0, 1], dtype=np.int64),
        tr3d_scores=np.asarray([0.95, 0.95, 0.80], dtype=np.float32),
        anchor_scores=np.asarray(
            [0.90, float(np.float32(0.80))], dtype=np.float64
        ),
    )
    # Proposal 11 wins the exact score tie; equality at anchor1 is rejected.
    assert result == {0: 1}


def test_geometry_equality_checks_dtype_and_storage_layout() -> None:
    value = _corners(0.0)
    assert _geometry_equal(value, value.copy())
    assert not _geometry_equal(value, value.astype(np.float64))
    assert not _geometry_equal(value, np.asfortranarray(value))


def test_paired_audit_accepts_only_expected_geometry_change() -> None:
    baseline = _prediction(0.0, 2.0, 4.0)
    candidate = _corners(1.0)
    active = RawPrediction(
        labels=baseline.labels,
        corners=(baseline.corners[0].copy(), candidate.copy(), baseline.corners[2].copy()),
        scores=baseline.scores,
    )
    report = audit_prediction_pair(
        baseline, active, expected_candidates={1: candidate}
    )
    assert report["ok"]
    assert report["eligible_replacement_rows"] == 1
    assert report["actual_byte_changed_rows"] == 1
    assert report["selected_geometry_exact"] == 1
    assert report["unselected_geometry_exact"] == 2


def test_paired_audit_rejects_label_score_and_unselected_mutation() -> None:
    baseline = _prediction(0.0, 2.0)
    active = RawPrediction(
        labels=(99, baseline.labels[1]),
        corners=(_corners(7.0), baseline.corners[1].copy()),
        scores=(0.1, baseline.scores[1]),
    )
    report = audit_prediction_pair(baseline, active, expected_candidates={})
    assert not report["ok"]
    rendered = "\n".join(report["issues"])
    assert "label bytes changed" in rendered
    assert "score bytes changed" in rendered
    assert "expected frozen G0" in rendered


def test_selected_noop_is_distinguished_from_changed_row() -> None:
    baseline = _prediction(0.0)
    active = _prediction(0.0)
    report = audit_prediction_pair(
        baseline, active, expected_candidates={0: baseline.corners[0].copy()}
    )
    assert report["ok"]
    assert report["eligible_replacement_rows"] == 1
    assert report["expected_byte_changed_rows"] == 0
    assert report["actual_byte_changed_rows"] == 0


def test_metric_exact_is_bit_strict() -> None:
    metric = {
        "predictions": 3,
        "ground_truth": 2,
        "matched_tp": 1,
        "average_precision": 0.5,
        "final_precision": 1 / 3,
        "final_recall": 0.5,
    }
    assert _metric_exact(metric, dict(metric))
    changed = dict(metric)
    changed["average_precision"] = np.nextafter(0.5, 1.0)
    assert not _metric_exact(metric, changed)


def test_create_only_report_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "paired.json"
    _write_create_only(path, {"ok": True})
    assert path.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError, match="immutable shadow-active"):
        _write_create_only(path, {"ok": False})
    assert '"ok": true' in path.read_text(encoding="utf-8")
