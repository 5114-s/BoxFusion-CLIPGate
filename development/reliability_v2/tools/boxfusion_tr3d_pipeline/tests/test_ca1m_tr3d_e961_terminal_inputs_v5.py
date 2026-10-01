from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from boxfusion.ca1m_tr3d_e961_terminal_inputs_v5 import (
    AUTHORIZATIONS,
    DEFAULT_CONFIG,
    NAMESPACE,
    PendingE961InputsError,
    ROLE_ORDER,
    ROLE_SPECS,
    sha256_file,
    static_report,
    validate_operational_ready,
    validate_static_config,
)


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "tools/preflight_ca1m_tr3d_e961_terminal_inputs_v5.py"
RUNNER = ROOT / "tools/run_ca1m_tr3d_e961_terminal_inputs_v5.py"
CORE = ROOT / "boxfusion/ca1m_tr3d_e961_terminal_inputs_v5.py"
DOC = ROOT / "docs/CA1M_TR3D_E961_TERMINAL_INPUTS_V5.md"


def _config() -> dict:
    return json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def _write(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _rows(record: dict) -> tuple[str, ...]:
    path = Path(record["path"])
    assert sha256_file(path) == record["sha256"]
    return tuple(row for row in path.read_text().splitlines() if row)


def _path_values(value: object, key: str = "") -> list[str]:
    if isinstance(value, dict):
        rows: list[str] = []
        for child_key, child in value.items():
            if child_key in {"path", "root"} and isinstance(child, str):
                rows.append(child)
            rows.extend(_path_values(child, child_key))
        return rows
    if isinstance(value, list):
        rows = []
        for child in value:
            rows.extend(_path_values(child, key))
        return rows
    return []


def test_static_contract_passes_and_writes_nothing():
    output_root = Path("/extra/ZhaoX") / NAMESPACE
    existed = output_root.exists() or output_root.is_symlink()
    report = static_report(DEFAULT_CONFIG)
    assert report["ok"] is True
    assert report["runtime_ready"] is False
    assert report["four_success_receipts_opened"] is False
    assert report["checkpoint_opened"] is False
    assert report["candidate_or_ground_truth_artifact_opened"] is False
    assert report["fold1_or_official_validation_path_present"] is False
    assert report["gpu_started"] is False
    assert report["output_created"] is False
    assert (output_root.exists() or output_root.is_symlink()) is existed


def test_exact80_scene_topology_and_exact1001_e961_composition():
    _, config = validate_static_config(DEFAULT_CONFIG)
    roles = config["scene_contract"]["roles"]
    candidate = {role: _rows(roles[role]["candidate_scene_list"]) for role in ROLE_ORDER}
    flat = tuple(scene for role in ROLE_ORDER for scene in candidate[role])
    assert len(flat) == len(set(flat)) == 80
    assert sum(len(candidate[role]) for role in ROLE_ORDER[1:]) == 60
    for role, spec in ROLE_SPECS.items():
        train = _rows(roles[role]["train_scene_list"])
        assert len(train) == len(set(train)) == 1001
        assert not set(train) & set(candidate[role])
        assert roles[role]["candidate_output_fold"] not in spec["train_folds"]


def test_each_role_receipt_is_unbound_and_authorizations_are_false():
    _, config = validate_static_config(DEFAULT_CONFIG)
    assert config["authorizations"] == AUTHORIZATIONS
    for role in ROLE_ORDER:
        receipt = config["scene_contract"]["roles"][role]["source_success_receipt"]
        assert receipt["state"] == "pending"
        assert receipt["path"] is None and receipt["sha256"] is None
        assert receipt["schema"] == ROLE_SPECS[role]["receipt_schema"]
    continuation = config["continuation_receipt"]
    assert continuation["state"] == "pending"
    assert continuation["path"] is None and continuation["sha256"] is None


def test_anchor_rows_are_final_base_and_scores_are_b6_v2_oof_only():
    _, config = validate_static_config(DEFAULT_CONFIG)
    anchors = config["anchor_inputs"]
    assert anchors["final_base_collection"]["geometry_and_row_authority"] is True
    assert anchors["native_b6_collection"]["anchor_native_evidence_only"] is True
    sidecar = anchors["native_b6_oof_sidecar"]
    assert sidecar["score_member"] == "deployment_blend_oof_scores"
    assert sidecar["each_row_model_excludes_scene"] is True
    assert sidecar["deploy_or_in_sample_scores_allowed"] is False
    overlay = config["pipeline"]["O_cpu_overlay"]
    assert overlay["geometry_source"] == "sealed_final_base_prediction"
    assert overlay["deploy_scores_used"] is False


def test_no_fold1_or_validation_path_and_no_legacy_formal_input():
    _, config = validate_static_config(DEFAULT_CONFIG)
    paths = _path_values({
        "scene_contract": config["scene_contract"],
        "anchor_inputs": config["anchor_inputs"],
        "candidate_inputs": config["candidate_inputs"],
    })
    lowered = "\n".join(paths).lower()
    assert "fold1" not in lowered
    assert "/val" not in lowered and "validation" not in lowered
    for token in config["forbidden_reuse"]["forbidden_formal_input_path_tokens"]:
        assert token.lower() not in lowered


def test_static_core_cannot_load_npz_checkpoint_or_ground_truth():
    source = CORE.read_text(encoding="utf-8")
    assert "np.load" not in source
    assert "torch.load" not in source
    assert "ground_truth_loader" not in source
    assert "subprocess" not in source
    assert "cuda" not in source.lower()


def test_operational_preflight_fails_before_receipt_or_output_access(tmp_path: Path):
    sentinel = tmp_path / "receipt_must_not_be_opened.json"
    output = tmp_path / "output_must_not_exist"
    with pytest.raises(PendingE961InputsError, match="four authoritative E961 R2"):
        validate_operational_ready(DEFAULT_CONFIG)
    assert not sentinel.exists()
    assert not output.exists()
    result = subprocess.run(
        [sys.executable, str(PREFLIGHT), "--operational-preflight"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 3
    report = json.loads(result.stderr)
    assert report["receipt_opened"] is False
    assert report["checkpoint_opened"] is False
    assert report["candidate_or_ground_truth_artifact_opened"] is False
    assert report["fold1_or_official_validation_path_resolved"] is False
    assert report["gpu_started"] is False
    assert report["output_created"] is False


def test_runner_skeleton_is_static_pass_and_operationally_blocked():
    static = subprocess.run(
        [sys.executable, str(RUNNER), "--static-contract"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert static.returncode == 0, static.stderr
    report = json.loads(static.stdout)
    assert report["runner_surface"]["operational_actions_reachable"] is False
    for role in ROLE_ORDER:
        blocked = subprocess.run(
            [sys.executable, str(RUNNER), "--run-role", role],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        assert blocked.returncode == 3
        payload = json.loads(blocked.stderr)
        assert payload["requested_role"] == role
        assert payload["device_argument_consumed"] is False
        assert payload["output_created"] is False


def test_runner_exposes_no_gt_fold1_or_validation_argument():
    help_result = subprocess.run(
        [sys.executable, str(RUNNER), "--help"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert help_result.returncode == 0
    help_text = help_result.stdout.lower()
    assert "--gt" not in help_text
    assert "--fold1" not in help_text
    assert "--official" not in help_text
    assert "--validation" not in help_text
    assert "--checkpoint" not in help_text
    assert "--anchor" not in help_text


@pytest.mark.parametrize("mutation", [
    "authorize_gpu",
    "bind_one_receipt",
    "in_sample_detector",
    "deploy_score",
    "legacy_candidate_input",
    "fold1_path",
    "weaken_create_only",
    "skip_candidate_recompute",
    "wrong_train_list",
])
def test_static_contract_rejects_unsafe_mutations(tmp_path: Path, mutation: str):
    value = copy.deepcopy(_config())
    if mutation == "authorize_gpu":
        value["authorizations"]["gpu_proposal_collection"] = True
    elif mutation == "bind_one_receipt":
        receipt = value["scene_contract"]["roles"]["outer_dev"]["source_success_receipt"]
        receipt.update(state="ready", path="/tmp/outer.json", sha256="0" * 64)
    elif mutation == "in_sample_detector":
        value["scene_contract"]["roles"]["inner_holdout2"]["detector_train_folds"] = [2, 3, 4]
    elif mutation == "deploy_score":
        value["anchor_inputs"]["native_b6_oof_sidecar"]["deploy_or_in_sample_scores_allowed"] = True
    elif mutation == "legacy_candidate_input":
        value["candidate_inputs"]["processed_rgbd_root"] = (
            "/tmp/ca1m_tr3d_terminal_ca_native_train100_v4"
        )
    elif mutation == "fold1_path":
        value["anchor_inputs"]["final_base_prediction_root"] = "/tmp/fold1/predictions"
    elif mutation == "weaken_create_only":
        value["integrity"]["create_only"] = False
    elif mutation == "skip_candidate_recompute":
        value["pipeline"]["E_candidate_native"]["candidate_native_evidence"] = "reuse_v4"
    elif mutation == "wrong_train_list":
        value["scene_contract"]["roles"]["inner_holdout3"]["train_scene_list"] = copy.deepcopy(
            value["scene_contract"]["roles"]["inner_holdout2"]["train_scene_list"]
        )
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_static_config(_write(tmp_path, value))


def test_documentation_names_all_four_roles_and_blocker():
    text = DOC.read_text(encoding="utf-8")
    for role in ROLE_ORDER:
        assert f"`{role}`" in text
    assert "static contract complete; operational execution blocked" in text
    assert "fit60" in text and "reused-dev20" in text
    assert "all-fold OOF" in text
