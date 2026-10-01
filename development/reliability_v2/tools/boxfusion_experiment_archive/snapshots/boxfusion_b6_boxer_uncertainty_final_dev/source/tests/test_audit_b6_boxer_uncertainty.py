import json

import numpy as np

from boxfusion.boxer_uncertainty import uncertainty_adjusted_selection
from tools.audit_b6_boxer_uncertainty import validate_diagnostic
from tools.report_b6_boxer_uncertainty import load_map


def test_diagnostic_accepts_json_null_for_fail_neutral_confidence(tmp_path):
    base = {
        "weights": np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
        "confidence": np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
        "selected_indices": np.asarray([0, 1, 2], dtype=np.int64),
        "selected_weights": np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
    }
    cfg = {"minimum_confidence": 0.05, "confidence_power": 1.0}
    confidence = np.asarray([np.nan, 0.8, 0.9, 0.95], dtype=np.float32)
    adjusted = uncertainty_adjusted_selection(
        base,
        confidence,
        np.ones(4, dtype=bool),
        cfg,
    )
    candidates = np.asarray([10, 11, 12, 13], dtype=np.int64)
    base_selected = candidates[base["selected_indices"]].tolist()
    active_selected = candidates[adjusted["selected_indices"]].tolist()
    selection_changed = bool(adjusted["selection_changed"])
    ranking_changed = bool(adjusted["ranking_changed"])
    candidate_changed = bool(adjusted["candidate_weights_changed"])
    effective_changed = bool(adjusted["effective_weights_changed"])
    record = {
        "candidate_indices": candidates.tolist(),
        "base_selected_indices": base_selected,
        "uncertainty_selected_indices": active_selected,
        "base_weights": adjusted["base_weights"].tolist(),
        "uncertainty_weights": adjusted["uncertainty_weights"].tolist(),
        "uncertainty_factors": adjusted["uncertainty_factors"].tolist(),
        "base_effective_weights": adjusted["base_effective_weights"].tolist(),
        "uncertainty_effective_weights": adjusted[
            "uncertainty_effective_weights"
        ].tolist(),
        "boxer_confidence": [None, 0.8, 0.9, 0.95],
        "boxer_geometry_applied": [True, True, True, True],
        "boxer_confidence_valid": [False, True, True, True],
        "selection_changed": selection_changed,
        "ranking_changed": ranking_changed,
        "candidate_weights_changed": candidate_changed,
        "weights_changed": effective_changed,
        "applied_to_fusion": True,
        "optimization_updated": True,
    }
    summary = {
        "fusion_groups": 1,
        "candidate_views": 4,
        "boxer_views": 4,
        "cutr_fallback_views": 0,
        "invalid_boxer_confidence": 1,
        "candidate_weight_changed_groups": int(candidate_changed),
        "weight_changed_groups": int(effective_changed),
        "selection_changed_groups": int(selection_changed),
        "ranking_changed_groups": int(ranking_changed),
        "active_groups": int(effective_changed),
        "optimization_updated_groups": 1,
        "active_updated_groups": int(effective_changed),
    }
    payload = {
        "schema": "boxfusion.boxer_uncertainty_fusion.scene.v1",
        "scene_id": "scene0000_00",
        "config": {
            "mode": "active",
            "confidence_power": 1.0,
            "minimum_confidence": 0.05,
        },
        "summary": summary,
        "records": [record],
    }
    path = tmp_path / "diagnostic.json"
    path.write_text(
        json.dumps(payload, allow_nan=False), encoding="utf-8"
    )

    _, issues = validate_diagnostic(path, "scene0000_00", "active")
    assert issues == []


def test_metric_loader_reads_three_thresholds_in_percent(tmp_path):
    path = tmp_path / "eval.log"
    path.write_text(
        "eval mAP: 0.40\neval mAP: 0.35\neval mAP: 0.15\n",
        encoding="utf-8",
    )
    assert load_map(path) == {"AP15": 40.0, "AP25": 35.0, "AP50": 15.0}
