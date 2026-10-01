from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from boxfusion.ca1m_incremental_l6 import (
    CA1MIncrementalL6Policy,
    CA_TR3D_BINDING_SCHEMA,
    CA_TR3D_BINDING_SHA256,
    CA_TR3D_CHECKPOINT_SHA256,
    FEATURE_NAMES,
    POLICY_SCHEMA,
    SCORE_POLICY,
    SOURCE_RANK_FORMULA,
    SPLIT_COUNTS,
    SPLIT_FOLDS,
    SPLIT_SHA256,
    UPSTREAM_ROUTE,
    assign_low_scores,
    source_aware_rank,
    validate_low_score_contract,
)
from tools.preflight_ca1m_incremental_l6_train100_v1 import validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/ca1m_incremental_l6_train100_v1.json"
LAUNCHER = ROOT / "scripts/run_ca1m_incremental_l6_train100_v1.sh"


def _policy_payload() -> dict:
    head = {
        "feature_mean": [0.0] * len(FEATURE_NAMES),
        "feature_scale": [1.0] * len(FEATURE_NAMES),
        "weights": [0.0] * len(FEATURE_NAMES),
        "bias": 0.0,
    }
    split = {
        role: {
            "folds": list(folds),
            "scene_count": SPLIT_COUNTS[role],
            "scene_list_sha256": SPLIT_SHA256[role],
        }
        for role, folds in SPLIT_FOLDS.items()
    }
    split.update({
        "train100_scene_list_sha256": SPLIT_SHA256["train100"],
        "official_validation_scene_list_sha256": SPLIT_SHA256[
            "official_validation_forbidden"
        ],
    })
    return {
        "schema": POLICY_SCHEMA,
        "complete": True,
        "activation_authorized": True,
        "dataset": "ca1m_train100",
        "train_only": True,
        "official_validation_access": False,
        "validation_predictions_used_for_training": False,
        "validation_overlap_count": 0,
        "upstream_route": UPSTREAM_ROUTE,
        "terminal_anchor_cross_fitted": True,
        "source_rank_formula": SOURCE_RANK_FORMULA,
        "score_policy": SCORE_POLICY,
        "feature_names": list(FEATURE_NAMES),
        "ca_native_tr3d_binding": {
            "schema": CA_TR3D_BINDING_SCHEMA,
            "sha256": CA_TR3D_BINDING_SHA256,
            "checkpoint_sha256": CA_TR3D_CHECKPOINT_SHA256,
            "initialization": "ca1m_random_scratch",
            "scannet_checkpoint_or_config_access": False,
        },
        "split": split,
        "upstream": {
            "final_base_manifest_sha256": "1" * 64,
            "native_b6_v2_collection_manifest_sha256": "2" * 64,
            "native_b6_v2_checkpoint_manifest_sha256": "3" * 64,
            "terminal_v4_manifest_sha256": "4" * 64,
            "terminal_benefit_v2_policy_sha256": "5" * 64,
            "post_terminal_anchor_manifest_sha256": "6" * 64,
        },
        "sample_gate": {
            "weights_train_candidates": 120,
            "weights_train_novel25_positive": 20,
            "weights_train_novel25_negative": 20,
            "weights_train_novel50_positive": 10,
            "threshold_dev_candidates": 20,
            "threshold_dev_positive_scenes": 4,
            "locked_internal_candidates": 20,
            "locked_internal_positive_scenes": 4,
        },
        "locked_internal_audit": {
            "consumed_once": True,
            "gate_passed": True,
            "official_validation_access": False,
        },
        "novelty25_head": head,
        "quality50_head": copy.deepcopy(head),
        "novelty_threshold": 0.7,
        "quality_threshold": 0.6,
        "hard_max_post_terminal_anchor_iou": 0.10,
        "max_candidates_per_scene": 6,
    }


def _immutable_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return path


def test_source_rank_and_global_low_score_contract_are_deterministic():
    raw = source_aware_rank(
        novelty_probability=0.7,
        visibility_quality_mean=0.0,
        support_ratio_mean=0.5,
        free_space_ratio_mean=0.25,
        selected_geometry="raw",
    )
    fused = source_aware_rank(
        novelty_probability=0.7,
        visibility_quality_mean=0.0,
        support_ratio_mean=0.5,
        free_space_ratio_mean=0.25,
        selected_geometry="fused",
    )
    assert fused == pytest.approx(raw + 0.04)
    scores = assign_low_scores([(1, 0, raw), (0, 0, fused)], 0.4)
    assert scores[(0, 0)] > scores[(1, 0)] > 0.0
    validate_low_score_contract(
        np.asarray([0.4, 0.7], np.float32),
        np.asarray(list(scores.values()), np.float32),
    )
    with pytest.raises(ValueError, match="below every anchor"):
        validate_low_score_contract([0.4], [0.4])


def test_ca_policy_schema_accepts_only_new_ca_provenance(tmp_path: Path):
    policy_path = _immutable_json(tmp_path / "ca_l6_policy.json", _policy_payload())
    policy = CA1MIncrementalL6Policy.load(policy_path)
    assert policy.sha256
    assert policy.probabilities(np.zeros(len(FEATURE_NAMES)))[0] == 0.5

    old = _policy_payload()
    old["schema"] = "boxfusion.tr3d_incremental_novelty_gate.v1"
    old_path = _immutable_json(tmp_path / "scannet_policy.json", old)
    with pytest.raises(ValueError, match="schema"):
        CA1MIncrementalL6Policy.load(old_path)

    wrong_model = _policy_payload()
    wrong_model["ca_native_tr3d_binding"]["checkpoint_sha256"] = "a" * 64
    wrong_path = _immutable_json(tmp_path / "wrong_model.json", wrong_model)
    with pytest.raises(ValueError, match="CA-scratch"):
        CA1MIncrementalL6Policy.load(wrong_path)


def test_real_static_preflight_binds_splits_and_ca_scratch_model_only():
    report = validate_config(CONFIG)
    assert report["static_contract_ready"] is True
    assert report["prerequisites_complete"] is False
    assert report["run_authorized"] is False
    assert report["scene_split"]["train100_scene_count"] == 100
    assert report["scene_split"]["validation_overlap_count"] == 0
    assert report["scene_split"]["ground_truth_files_opened"] is False
    assert report["ca_native_tr3d_binding"]["checkpoint_sha256"] == (
        CA_TR3D_CHECKPOINT_SHA256
    )
    assert report["ca_native_tr3d_binding"]["scannet_artifact_access"] is False
    assert report["required_upstream_chain"]["bound"] is False
    assert report["required_upstream_chain"]["artifacts_opened"] is False
    assert report["validation_ground_truth_files_opened"] is False
    assert report["gpu_started"] is False
    assert report["model_started"] is False


def test_partial_or_old_upstream_binding_is_fail_closed(tmp_path: Path):
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    value["required_upstream_chain"]["final_base"]["anchor_root"] = str(tmp_path)
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="partially bound"):
        validate_config(partial)

    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    value["required_upstream_chain"]["terminal_benefit_v2"][
        "policy_schema"
    ] = "boxfusion.ca1m_tr3d_terminal_gate_policy.v1"
    old = tmp_path / "old_terminal_policy_schema.json"
    old.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="terminal-benefit v2 schemas"):
        validate_config(old)


def test_launcher_run_is_fail_closed_before_gpu_gt_or_prediction_access():
    syntax = subprocess.run(
        ["bash", "-n", str(LAUNCHER)], capture_output=True, text=True, check=False
    )
    assert syntax.returncode == 0, syntax.stderr
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--run is fail-closed" in result.stderr
    assert '"gpu_started": false' in result.stdout
    assert '"model_started": false' in result.stdout
    assert '"validation_ground_truth_files_opened": false' in result.stdout


def test_launcher_rejects_raw_scannet_style_policy_override():
    environment = dict(os.environ)
    environment["BOXFUSION_LIGHTWEIGHT_POLICY"] = "/tmp/old_scannet_policy.json"
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--static-preflight"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Raw/legacy override is forbidden" in result.stderr
