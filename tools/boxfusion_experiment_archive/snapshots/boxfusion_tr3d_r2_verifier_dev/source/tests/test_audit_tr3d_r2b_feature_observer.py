from __future__ import annotations

import numpy as np
import pytest

from tools.audit_tr3d_r2b_feature_observer import (
    _rank,
    _route_pass,
    binary_auc_ap,
)


def test_binary_auc_ap_perfect_and_tied() -> None:
    labels = np.asarray([True, False, True, False], dtype=np.bool_)
    perfect = binary_auc_ap(labels, [0.9, 0.2, 0.8, 0.1])
    assert perfect["auc"] == 1.0
    assert perfect["average_precision"] == 1.0

    tied = binary_auc_ap(labels, [0.5, 0.5, 0.5, 0.5])
    assert tied["auc"] == 0.5
    assert tied["average_precision"] == pytest.approx((1.0 + 2 / 3) / 2)


def test_binary_auc_rejects_missing_class() -> None:
    with pytest.raises(ValueError, match="both positive and negative"):
        binary_auc_ap(np.asarray([True, True]), [1.0, 0.0])


def test_rank_has_deterministic_scene_and_proposal_ties() -> None:
    rows = [
        {"scene_id": "scene0002_00", "proposal_id": 1, "score": 0.5},
        {"scene_id": "scene0001_00", "proposal_id": 3, "score": 0.5},
        {"scene_id": "scene0001_00", "proposal_id": 2, "score": 0.5},
    ]
    assert _rank(rows, "score", 2) == {
        ("scene0001_00", 2),
        ("scene0001_00", 3),
    }


def _metric(*, precision: float, novel: int, coverage: int) -> dict:
    return {
        "independent_precision50_upper_bound": precision,
        "independent_tp50_scene_coverage": coverage,
        "thresholds": {"0.50": {"novel_oracle_tp": novel}},
    }


def test_route_gate_requires_material_increment() -> None:
    score = _metric(precision=0.30, novel=5, coverage=3)
    passed, _ = _route_pass(
        score, _metric(precision=0.35, novel=5, coverage=3)
    )
    assert passed
    passed, _ = _route_pass(
        score, _metric(precision=0.30, novel=7, coverage=3)
    )
    assert passed
    passed, reasons = _route_pass(
        score, _metric(precision=0.34, novel=6, coverage=4)
    )
    assert not passed and not reasons
