from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from boxfusion.ca1m_tr3d_exploratory_gate_v5 import (
    AUTHORIZATIONS,
    DETECTOR_ROLES,
    FIT_FOLDS,
    GATE_ROLES,
    PendingProtocolError,
    sha256_file,
    static_report,
    validate_ready,
    validate_static_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/ca1m_tr3d_exploratory_gate_xfit_r2_v5_pending.json"
SCHEMA = ROOT / "config/ca1m_tr3d_exploratory_gate_xfit_r2_v5_pending.schema.json"
PREFLIGHT = ROOT / "tools/preflight_ca1m_tr3d_exploratory_gate_v5.py"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _write_mutation(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def test_pending_v5_contract_is_static_only_and_schema_bound():
    path, cfg = validate_static_config(CONFIG)
    assert path == CONFIG.resolve()
    assert cfg["authorizations"] == AUTHORIZATIONS
    assert not any(AUTHORIZATIONS.values())
    assert cfg["access"] == {
        "static_preflight_only": True,
        "ground_truth_access": False,
        "fold0_ground_truth_access": False,
        "fold1_metadata_or_ground_truth_access": False,
        "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "scannet_artifact_access": False,
    }
    assert cfg["schema_document"]["path"] == str(SCHEMA)
    assert cfg["schema_document"]["sha256"] == sha256_file(SCHEMA)
    report = static_report(CONFIG)
    assert report["ok"] is True
    assert report["candidate_or_gt_artifact_opened"] is False
    assert report["output_created"] is False
    assert report["run_authorized"] is False


def test_v5_freezes_detector_and_gate_double_oof_topology():
    _, cfg = validate_static_config(CONFIG)
    detector = cfg["protocol"]["detector_candidate_roles"]
    observed_detector = tuple((
        row["role"], tuple(row["detector_train_folds"]),
        tuple(row["candidate_output_folds"]), row["candidate_use"],
    ) for row in detector)
    assert observed_detector == DETECTOR_ROLES
    gate = cfg["protocol"]["gate_crossfit_roles"]
    observed_gate = tuple((
        row["role"], tuple(row["gate_train_folds"]),
        tuple(row["gate_oof_output_folds"]),
    ) for row in gate)
    assert observed_gate == GATE_ROLES
    assert tuple(cfg["protocol"]["fit_folds"]) == FIT_FOLDS
    assert cfg["protocol"]["threshold_selection_source"] == (
        "fold234_scene_grouped_gate_oof_only"
    )
    assert cfg["protocol"]["anchor_score_source"] == (
        "ca1m_native_b6_final_base_all_fold_oof_row_scores_v2"
    )
    assert cfg["protocol"][
        "deploy_anchor_scores_allowed_for_gate_fit_or_threshold_selection"
    ] is False
    assert cfg["protocol"][
        "fold0_used_for_threshold_or_hyperparameter_selection"
    ] is False
    assert cfg["protocol"]["fold1_or_validation_used_for_any_selection"] is False


def test_v5_outer_continuation_gate_is_fixed_and_fail_closed():
    _, cfg = validate_static_config(CONFIG)
    gate = cfg["protocol"]["outer_to_inner_continuation_gate"]
    assert gate == {
        "checkpoint_policy": "fixed_final_iter_only_no_checkpoint_selection",
        "fold0_role": "reused_dev_continuation_diagnostic_only",
        "raw_detector_ap_role": "diagnostic_only_not_a_continuation_criterion",
        "oracle": "same_gt_iou_gain_ge_0.05",
        "min_replacements": 10,
        "min_scenes": 5,
        "min_delta_ap15": 0.0,
        "min_delta_ap25": 0.0,
        "min_delta_ap50": 0.005,
        "all_checks_required": True,
        "inner_models_require_sealed_continuation_receipt": True,
        "failure_action": "stop_without_training_inner_xfit_models",
    }
    assert cfg["candidate_source"]["checkpoint_policy"] == (
        "fixed_final_iter_only_no_checkpoint_selection"
    )
    assert cfg["candidate_source"]["fold0_checkpoint_selection_allowed"] is False
    receipt = cfg["prerequisites"]["xfit_r2_outer_continuation_receipt"]
    assert receipt["state"] == "pending"
    assert receipt["path"] is None and receipt["sha256"] is None


def test_v5_requires_iou50_regression_groupwise_benefit_and_weak_raw_score():
    _, cfg = validate_static_config(CONFIG)
    heads = cfg["learning"]["heads"]
    assert set(heads) == {
        "candidate_iou_regression",
        "candidate_iou50_calibration",
        "pairwise_groupwise_benefit",
    }
    assert heads["candidate_iou_regression"]["target"] == (
        "candidate_max_gt_iou_continuous_0_1"
    )
    assert heads["candidate_iou50_calibration"]["target"] == (
        "candidate_max_gt_iou_strict_gt_0.50"
    )
    assert heads["pairwise_groupwise_benefit"]["group_keys"] == [
        "scene_id", "anchor_index",
    ]
    source = cfg["candidate_source"]
    assert source["raw_tr3d_score_direct_gate_allowed"] is False
    assert source["raw_score_only_model_allowed"] is False
    order = cfg["selection"]["within_group_order"]
    assert order[-2] == "raw_tr3d_score_desc_final_tie_break"
    assert all("raw_tr3d" not in field for field in order[:-2])


def test_v5_fold0_is_diagnostic_and_cannot_unlock_fold1_or_policy():
    _, cfg = validate_static_config(CONFIG)
    diagnostic = cfg["diagnostics"]
    assert diagnostic["fold0_role"] == "reused_dev_diagnostic_only"
    assert diagnostic["fold0_retuning_allowed"] is False
    assert diagnostic["fold0_model_selection_allowed"] is False
    assert diagnostic["fold0_result_can_authorize_policy"] is False
    assert diagnostic["fold0_result_can_authorize_fold1_or_validation"] is False
    assert cfg["authorizations"]["fold1_internal_check"] is False
    assert cfg["authorizations"]["official_validation"] is False
    assert cfg["authorizations"]["policy_activation"] is False


def test_v5_operational_preflight_stops_before_artifacts_or_outputs():
    with pytest.raises(PendingProtocolError, match="pending sealed asymmetric-xfit R2"):
        validate_ready(CONFIG)
    result = subprocess.run(
        [sys.executable, str(PREFLIGHT), "--preflight"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 3
    report = json.loads(result.stderr)
    assert report["ok"] is False
    assert report["failure_action"] == "stop_before_opening_candidate_gt_or_output"
    assert report["candidate_or_gt_artifact_opened"] is False
    assert report["output_created"] is False


def test_v5_static_cli_passes_without_write_or_run_surface():
    result = subprocess.run(
        [sys.executable, str(PREFLIGHT), "--static-contract"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["ok"] is True and report["output_created"] is False
    source = PREFLIGHT.read_text(encoding="utf-8")
    assert "--run" not in source
    assert "--train" not in source
    assert "--fold1" not in source
    assert "--official-val" not in source


@pytest.mark.parametrize("mutation", [
    "authorize_fit", "leak_detector_fold", "fold0_select", "strong_raw_score",
    "old_pool_path", "weaken_continuation", "bind_pending_input",
])
def test_v5_static_contract_rejects_unsafe_mutations(tmp_path: Path, mutation: str):
    value = copy.deepcopy(_config())
    if mutation == "authorize_fit":
        value["authorizations"]["gate_fit"] = True
    elif mutation == "leak_detector_fold":
        value["protocol"]["detector_candidate_roles"][0]["detector_train_folds"] = [2, 3, 4]
    elif mutation == "fold0_select":
        value["protocol"]["fold0_used_for_threshold_or_hyperparameter_selection"] = True
    elif mutation == "strong_raw_score":
        value["candidate_source"]["raw_tr3d_score_direct_gate_allowed"] = True
    elif mutation == "old_pool_path":
        value["outputs"]["dataset"] = (
            "/tmp/ca1m_tr3d_benefit_gate_final_base_v4/dataset.npz"
        )
    elif mutation == "weaken_continuation":
        value["protocol"]["outer_to_inner_continuation_gate"]["min_delta_ap50"] = 0.0
    elif mutation == "bind_pending_input":
        record = value["prerequisites"]["xfit_r2_candidate_collection"]
        record.update(state="ready", path="/tmp/candidates.json", sha256="0" * 64)
    with pytest.raises(ValueError):
        validate_static_config(_write_mutation(tmp_path, value))


def test_v5_has_no_formal_input_path_to_rejected_v4_pool_or_policy():
    _, cfg = validate_static_config(CONFIG)
    serialized_inputs = json.dumps({
        "candidate_source": cfg["candidate_source"],
        "prerequisites": cfg["prerequisites"],
        "outputs": cfg["outputs"],
    }).lower()
    assert "ca1m_tr3d_benefit_gate_final_base_v4" not in serialized_inputs
    assert "ca1m_tr3d_terminal_ca_native_train100_v4" not in serialized_inputs
    assert cfg["design_basis"]["source_is_formal_input"] is False
    assert cfg["design_basis"]["rejected_policy_is_input"] is False
    assert cfg["design_basis"]["old_candidate_pool_is_input"] is False
