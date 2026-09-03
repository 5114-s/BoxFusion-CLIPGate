import json

from tools.audit_final_boxer_uncertainty import (
    audit_directory,
    validate_payload,
)


def _payload(mode="observer"):
    corner_hash = "same" if mode == "observer" else "changed"
    applied = 0 if mode == "observer" else 1
    records = []
    if mode == "active":
        records.append(
            {
                "selection_changed": False,
                "ranking_changed": False,
                "applied": True,
                "baseline_corners": [[[0.0]]],
                "candidate_corners": [[[1.0]]],
            }
        )
    return {
        "schema": "boxfusion.final_boxer_uncertainty.scene.v1",
        "scene_id": "scene0000_00",
        "config": {"mode": mode},
        "summary": {
            "output_rows": 3,
            "matched_rows": 2,
            "weight_changed_rows": 1,
            "optimized_rows": applied,
            "applied_rows": applied,
            "selection_changed_rows": 0,
            "ranking_changed_rows": 0,
        },
        "contract": {
            "protected_fields_equal": True,
            "scene_fallback": False,
            "count_before": 3,
            "count_after": 3,
            "scores_sha256_before": "scores",
            "scores_sha256_after": "scores",
            "source_indices_sha256_before": "sources",
            "source_indices_sha256_after": "sources",
            "stable_ids_sha256_before": "ids",
            "stable_ids_sha256_after": "ids",
            "baseline_corners_sha256": "same",
            "output_corners_sha256": corner_hash,
        },
        "records": records,
    }


def test_valid_observer_and_active_payloads():
    assert validate_payload(_payload("observer"), expected_mode="observer") == []
    assert validate_payload(_payload("active"), expected_mode="active") == []


def test_audit_rejects_protected_score_mutation():
    payload = _payload("active")
    payload["contract"]["scores_sha256_after"] = "mutated"
    issues = validate_payload(payload, expected_mode="active")
    assert "scores_sha256 changed" in issues


def test_directory_audit_checks_scene_coverage(tmp_path):
    path = tmp_path / "scene0000_00_final_boxer_uncertainty.json"
    path.write_text(json.dumps(_payload("observer")), encoding="utf-8")
    report = audit_directory(
        tmp_path,
        expected_mode="observer",
        expected_scenes=["scene0000_00"],
    )
    assert report["ok"]
    assert report["aggregate"]["files"] == 1
    missing = audit_directory(
        tmp_path,
        expected_mode="observer",
        expected_scenes=["scene0000_00", "scene0001_00"],
    )
    assert not missing["ok"]
    assert missing["missing_scenes"] == ["scene0001_00"]
