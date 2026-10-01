from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from boxfusion import ca1m_tr3d_e961_outer_eval_v1 as contract
from tools import run_ca1m_tr3d_e961_outer_eval_v1 as runner


ROOT = Path("/data/ZhaoX/BoxFusion/tools/boxfusion_tr3d_pipeline")
CONFIG = ROOT / "config/ca1m_tr3d_e961_outer_dev_eval_v2.json"
RUNNER = ROOT / "tools/run_ca1m_tr3d_e961_outer_eval_v1.py"
WRAPPER = ROOT / "scripts/run_ca1m_tr3d_e961_outer_eval_v2.sh"
V1_WRAPPER = ROOT / "scripts/run_ca1m_tr3d_e961_outer_eval_v1.sh"
V1_SEALER = ROOT / "tools/seal_ca1m_tr3d_e961_outer_eval_protocol_v1.py"


def _corners(bounds: list[float]) -> np.ndarray:
    low = np.asarray(bounds[:3], np.float32)
    high = np.asarray(bounds[3:], np.float32)
    return np.asarray(
        [[x, y, z] for x in (low[0], high[0])
         for y in (low[1], high[1]) for z in (low[2], high[2])],
        np.float32,
    )


def _function_calls(path: Path) -> dict[str, list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: dict[str, list[str]] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        result[node.name] = [
            child.func.id
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        ]
    return result


def test_static_preflight_is_exact20_and_does_not_probe_expanded_artifacts() -> None:
    result = runner.preflight(CONFIG, None)
    assert result["ok"] is True
    assert result["read_only"] is True
    assert result["scene_count"] == 20
    assert result["fold0_role"] == "reused_dev"
    assert result["expanded_training_receipt_access"] is False
    assert result["expanded_checkpoint_access"] is False
    assert result["anchor_array_access"] is False
    assert result["ground_truth_access"] is False
    assert result["gpu_started"] is False
    assert result["fold1_access"] is False
    assert result["official_validation_access"] is False


def test_runtime_prereg_freezes_science_before_dynamic_paths_are_touched() -> None:
    source, cfg = contract.load_config(CONFIG)
    value = contract._expected_preregistration(source, cfg, "outer_run_001")
    assert value["sealed_before_expanded_training_receipt_access"] is True
    assert value["sealed_before_expanded_checkpoint_access"] is True
    assert value["sealed_before_fold0_gt_access"] is True
    assert value["expanded_training_receipt_access_at_seal"] is False
    assert value["expanded_checkpoint_access_at_seal"] is False
    assert value["fold0_gt_access_at_seal"] is False
    assert value["partition"] == "official_train_fold0_reused_dev_exact20"
    assert value["checkpoint_policy"]["checkpoint_name"] == "iter_11268.pth"
    assert value["checkpoint_policy"]["optimizer_updates"] == 11268
    assert value["checkpoint_policy"]["checkpoint_selection"] is False
    assert value["metric"] == {
        "class_mode": "CA_class_agnostic",
        "coordinate_frame": "world",
        "box_geometry": "axis_aligned_AABB_from_8_corners",
        "ranking": "global_prediction_score",
        "duplicate_matching": "one_detection_per_scene_gt_per_threshold",
        "iou_comparison": "strict_greater_than_threshold",
        "iou_thresholds": [0.15, 0.25, 0.50],
    }
    assert value["continuation_gate"]["pass_authorizes_inner_roles"] == list(
        contract.INNER_ROLES
    )


def test_v1_protocol_is_immutably_invalid_and_v2_binds_reviewed_r2() -> None:
    invalid = contract.validate_invalidated_protocol_v1()
    assert invalid["formal_v1_authorized"] is False
    path, value = contract.validate_protocol_preregistration(CONFIG)
    assert path.name == "PREREGISTRATION_PROTOCOL_V2.json"
    assert value["schema"].endswith("protocol_preregistration.v2")
    assert value["invalidated_predecessor"]["formal_v1_authorized"] is False
    frozen = value["outer_train_r2_frozen_review"]
    assert frozen["independent_review_pass"] is True
    assert frozen["tool"]["sha256"] == (
        "36f2f02cbc1201aa55adb1104cb198f1cb4fda478e466101dcafc7660790a70f"
    )
    assert frozen["trainer"]["sha256"] == (
        "b2da00db79f586d0e7dd4fc2d663ed53aaff08dd2aeca67c7c1dfd477ef93557"
    )
    assert frozen["driver"]["sha256"] == (
        "79167d075142f5ce674a686e3015114508bf2b2453590c69be40d8997c4ef9f2"
    )
    assert frozen["tests"]["sha256"] == (
        "8d985af88b42c557f21fec2e3b0d0670b613798e1495dcf60a36f841fb82cbff"
    )
    assert frozen["run_receipt_schema"].endswith("outer_train_run.r2")
    assert frozen["authorization_consumption_schema"].endswith(
        "outer_auth_consumption.r2"
    )
    assert frozen["training_started_claim_schema"].endswith(
        "outer_training_started.r2"
    )


def test_runner_orders_prereg_then_receipt_checkpoint_then_collection_then_gt() -> None:
    calls = _function_calls(RUNNER)
    receipt = calls["_read_training_receipt_after_prereg"]
    assert receipt.index("validate_preregistration") < receipt.index("read_json")
    evaluation = calls["evaluate"]
    assert evaluation.index("_load_collection") < evaluation.index("scene_ids") + 100
    # Private sealed helpers are attribute calls, so prove ordering directly in source.
    source = RUNNER.read_text(encoding="utf-8")
    evaluation_source = source[source.index("def evaluate("):source.index("def preflight(")]
    assert evaluation_source.index("_load_collection(") < evaluation_source.index(
        "sealed_r2._load_anchor_shadow("
    ) < evaluation_source.index("sealed_r2._load_ground_truth(")
    main_source = source[source.index("def main("):]
    assert main_source.index("seal_preregistration(") < main_source.index(
        "seal_binding("
    ) < main_source.index("_run_proposals(") < main_source.index("evaluate(")
    assert source.count("sealed_r2._load_ground_truth(") == 1


def test_gate_exact_boundaries_nonfinite_and_raw_diagnostic_independence() -> None:
    boundary = contract.continuation_gate(
        proposal_integrity_pass=True,
        scene_count=20,
        replacement_count=10,
        replacement_scene_count=5,
        oracle_ap_delta={"iou_0.15": 0.0, "iou_0.25": 0.0, "iou_0.50": 0.005},
    )
    assert boundary["pass"] is True
    assert boundary["authorized_inner_roles"] == list(contract.INNER_ROLES)
    for kwargs in (
        {"replacement_count": 9},
        {"replacement_scene_count": 4},
        {"oracle_ap_delta": {"iou_0.15": -1e-12, "iou_0.25": 0.0, "iou_0.50": 0.005}},
        {"oracle_ap_delta": {"iou_0.15": 0.0, "iou_0.25": -1e-12, "iou_0.50": 0.005}},
        {"oracle_ap_delta": {"iou_0.15": 0.0, "iou_0.25": 0.0, "iou_0.50": 0.004999}},
        {"oracle_ap_delta": {"iou_0.15": 0.0, "iou_0.25": 0.0, "iou_0.50": np.nan}},
    ):
        values = {
            "proposal_integrity_pass": True,
            "scene_count": 20,
            "replacement_count": 10,
            "replacement_scene_count": 5,
            "oracle_ap_delta": {"iou_0.15": 0.0, "iou_0.25": 0.0, "iou_0.50": 0.005},
        }
        values.update(kwargs)
        failed = contract.continuation_gate(**values)
        assert failed["pass"] is False
        assert failed["authorized_inner_roles"] == []
    # Raw AP is not an argument to the gate and cannot change the decision.
    assert "raw" not in contract.continuation_gate.__annotations__


def test_same_gt_oracle_preserves_oof_scores_and_requires_gain_005() -> None:
    gt = np.stack([_corners([0, 0, 0, 1, 1, 1])])
    anchor = np.stack([_corners([0, 0, 0, 0.8, 1, 1])])
    candidate = np.stack([_corners([0, 0, 0, 1, 1, 1])])
    output, summary = contract.same_gt_oracle_scene(
        anchor_corners=anchor,
        anchor_scores=np.asarray([0.91], np.float32),
        candidate_corners=candidate,
        candidate_scores=np.asarray([0.01], np.float32),
        gt_corners=gt,
        near_iou=0.15,
        min_gain=0.05,
    )
    assert np.array_equal(output[0], candidate[0])
    assert summary["selected_replacement_count"] == 1
    assert summary["scores_preserved"] is True
    assert summary["row_order_preserved"] is True
    assert summary["oracle_deployable"] is False


def test_official_ca_ap_uses_global_score_order_strict_iou_and_duplicate_matching() -> None:
    result = contract.official_ca_ap(
        scene_ids=np.asarray(["00000001", "00000001", "00000001"]),
        scores=np.asarray([0.9, 0.8, 0.7]),
        best_iou=np.asarray([0.15, 0.90, 0.95]),
        best_gt=np.asarray([0, 0, 0]),
        ground_truth_count=1,
    )
    assert result["iou_0.15"]["tp"] == 1
    assert result["iou_0.15"]["fp"] == 2
    assert result["iou_0.25"]["tp"] == 1
    assert result["iou_0.50"]["tp"] == 1


def _readonly(path: Path, data: str) -> Path:
    path.write_text(data, encoding="utf-8")
    path.chmod(0o444)
    return path


def _fake_binding(tmp_path: Path) -> runner.E961Binding:
    receipt = _readonly(tmp_path / "receipt.json", "{}\n")
    checkpoint = _readonly(tmp_path / "iter_11268.pth", "checkpoint\n")
    effective = _readonly(tmp_path / "outer_dev.py", "effective\n")
    binding = _readonly(tmp_path / "binding.json", "{}\n")
    return runner.E961Binding(
        path=binding, sha256=contract.sha256_file(binding), run_tag="run_001",
        receipt_path=receipt, receipt_sha256=contract.sha256_file(receipt),
        checkpoint_path=checkpoint, checkpoint_sha256=contract.sha256_file(checkpoint),
        effective_config_path=effective,
        effective_config_sha256=contract.sha256_file(effective),
    )


def test_pass_creates_only_three_role_authorization_and_fail_creates_only_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prereg = _readonly(tmp_path / "PREREGISTRATION.json", "{}\n")
    collection = _readonly(tmp_path / "PROPOSAL_COLLECTION.json", "{}\n")
    report = _readonly(tmp_path / "EVALUATION_REPORT.json", "{}\n")
    monkeypatch.setattr(runner, "PREREGISTRATION_PATH", prereg)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "CONTINUATION_PATH", tmp_path / "CONTINUATION.json")
    monkeypatch.setattr(runner, "INNER_AUTHORIZATION_PATH", tmp_path / "AUTH.json")
    monkeypatch.setattr(runner, "STOP_PATH", tmp_path / "STOP.json")
    binding = _fake_binding(tmp_path)
    passed = contract.continuation_gate(
        proposal_integrity_pass=True, scene_count=20,
        replacement_count=10, replacement_scene_count=5,
        oracle_ap_delta={"iou_0.15": 0.0, "iou_0.25": 0.0, "iou_0.50": 0.005},
    )
    path, value = runner._decision_artifact(
        gate=passed, report_path=report, binding=binding,
        collection_path=collection,
    )
    assert path.name == "AUTH.json"
    assert value["authorized_roles"] == list(contract.INNER_ROLES)
    assert not (tmp_path / "STOP.json").exists()

    other = tmp_path / "failure"
    other.mkdir()
    prereg2 = _readonly(other / "PREREGISTRATION.json", "{}\n")
    collection2 = _readonly(other / "PROPOSAL_COLLECTION.json", "{}\n")
    report2 = _readonly(other / "EVALUATION_REPORT.json", "{}\n")
    monkeypatch.setattr(runner, "PREREGISTRATION_PATH", prereg2)
    monkeypatch.setattr(runner, "CONTINUATION_PATH", other / "CONTINUATION.json")
    monkeypatch.setattr(runner, "INNER_AUTHORIZATION_PATH", other / "AUTH.json")
    monkeypatch.setattr(runner, "STOP_PATH", other / "STOP.json")
    failed = contract.continuation_gate(
        proposal_integrity_pass=True, scene_count=20,
        replacement_count=9, replacement_scene_count=5,
        oracle_ap_delta={"iou_0.15": 0.0, "iou_0.25": 0.0, "iou_0.50": 0.005},
    )
    path2, value2 = runner._decision_artifact(
        gate=failed, report_path=report2, binding=binding,
        collection_path=collection2,
    )
    assert path2.name == "STOP.json"
    assert value2["authorized_roles"] == []
    assert not (other / "AUTH.json").exists()


def test_run_tag_and_cuda_device_are_fixed() -> None:
    assert contract.validate_run_tag("outer_run_001") == "outer_run_001"
    assert runner._require_cuda_device("cuda:0") == "cuda:0"
    for tag in ("x", "..", "bad/tag", "bad tag", "a..b"):
        with pytest.raises(ValueError):
            contract.validate_run_tag(tag)
    for device in ("cpu", "cuda", "cuda:1", "0"):
        with pytest.raises(ValueError):
            runner._require_cuda_device(device)


def test_unique_wrapper_never_launches_inner_training() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert "--run OUTER_R2_RUN_TAG GPU_ID" in source
    assert "run_ca1m_tr3d_e961_outer_eval_v2.py" in source
    assert "train_tr3d_ca1m_e961_xfit_v1.sh" not in source
    assert "inner_holdout2" not in source
    assert "inner_holdout3" not in source
    assert "inner_holdout4" not in source


def test_legacy_v1_entrypoints_are_tombstones_not_v2_aliases() -> None:
    wrapper = V1_WRAPPER.read_text(encoding="utf-8")
    sealer = V1_SEALER.read_text(encoding="utf-8")
    for source in (wrapper, sealer):
        assert "INVALID/SUPERSEDED" in source
        assert "exit 66" in source or "return 66" in source
    assert "run_ca1m_tr3d_e961_outer_eval_v2.py" not in wrapper
    assert "seal_protocol_preregistration" not in sealer
