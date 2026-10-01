from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from boxfusion.ca1m_tr3d_inference_contract import (
    sha256_file as inference_sha256,
    validate_ca1m_point_inference_config,
)
from boxfusion.ca1m_tr3d_xfit_r2_eval import (
    continuation_gate,
    load_config,
    official_ca_ap,
    same_gt_oracle_scene,
    validate_effective_config,
    validate_outer_wrapper_log,
)
from tools.run_ca1m_tr3d_xfit_r2_outer_dev_eval import (
    _require_cuda_device,
    preflight,
)


OVM_ROOT = Path("/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev")
WORK_ROOT = Path(
    "/extra/ZhaoX/tr3d_ca1m_work_dirs/ca1m_xfit_v2_formal_r2/"
    "ca1m_xfit_v2_formal_outer_dev_seed0_r2"
)
PIPELINE_ROOT = Path("/data/ZhaoX/BoxFusion/tools/boxfusion_tr3d_pipeline")
EVAL_CONFIG = PIPELINE_ROOT / "config/ca1m_tr3d_xfit_r2_outer_dev_eval_v2.json"


def _corners(bounds: list[float]) -> np.ndarray:
    low = np.asarray(bounds[:3], dtype=np.float32)
    high = np.asarray(bounds[3:], dtype=np.float32)
    return np.asarray(
        [[x, y, z] for x in (low[0], high[0])
         for y in (low[1], high[1]) for z in (low[2], high[2])],
        dtype=np.float32,
    )


def test_official_ca_ap_is_strict_and_duplicate_aware() -> None:
    result = official_ca_ap(
        scene_ids=np.asarray(["00000001", "00000001", "00000001"]),
        scores=np.asarray([0.9, 0.8, 0.7]),
        best_iou=np.asarray([0.15, 0.20, 0.90]),
        best_gt=np.asarray([0, 0, 0]),
        ground_truth_count=1,
    )
    # IoU exactly 0.15 is an FP; the next row detects GT0 and the final row is
    # a duplicate FP despite its higher IoU.
    assert result["iou_0.15"]["tp"] == 1
    assert result["iou_0.15"]["fp"] == 2
    assert result["iou_0.25"]["tp"] == 1
    assert result["iou_0.50"]["tp"] == 1


def test_same_gt_oracle_freezes_gain_005_and_preserves_anchor_scores() -> None:
    gt = np.stack([
        _corners([0, 0, 0, 1, 1, 1]),
        _corners([3, 0, 0, 4, 1, 1]),
    ])
    anchors = np.stack([
        _corners([0, 0, 0, 0.8, 1, 1]),
        _corners([3, 0, 0, 4, 1, 1]),
    ])
    candidates = np.stack([
        _corners([0, 0, 0, 1, 1, 1]),
        _corners([3, 0, 0, 3.98, 1, 1]),
    ])
    output, summary = same_gt_oracle_scene(
        anchor_corners=anchors,
        anchor_scores=np.asarray([0.8, 0.7], np.float32),
        candidate_corners=candidates,
        candidate_scores=np.asarray([0.6, 0.9], np.float32),
        gt_corners=gt,
    )
    assert np.array_equal(output[0], candidates[0])
    # Candidate 1 improves by less than 0.05, so its geometry is unchanged.
    assert np.array_equal(output[1], anchors[1])
    assert summary["selected_replacement_count"] == 1
    assert summary["min_same_gt_iou_gain"] == 0.05
    assert summary["scores_preserved"] is True
    assert summary["oracle_deployable"] is False


def test_continuation_gate_boundary_and_failure_are_fail_closed() -> None:
    passed = continuation_gate(
        proposal_integrity_pass=True,
        scene_count=20,
        replacement_count=10,
        replacement_scene_count=5,
        oracle_ap_delta={"iou_0.15": 0.0, "iou_0.25": 0.0, "iou_0.50": 0.005},
    )
    assert passed["pass"] is True
    assert passed["continue_inner_training_authorized"] is True
    assert passed["authorized_inner_roles"] == [
        "inner_holdout2", "inner_holdout3", "inner_holdout4"
    ]
    failed = continuation_gate(
        proposal_integrity_pass=True,
        scene_count=20,
        replacement_count=9,
        replacement_scene_count=5,
        oracle_ap_delta={"iou_0.15": 0.0, "iou_0.25": 0.0, "iou_0.50": 0.005},
    )
    assert failed["pass"] is False
    assert failed["continue_inner_training_authorized"] is False
    assert failed["authorized_inner_roles"] == []
    nonfinite = continuation_gate(
        proposal_integrity_pass=True,
        scene_count=20,
        replacement_count=10,
        replacement_scene_count=5,
        oracle_ap_delta={"iou_0.15": 0.0, "iou_0.25": 0.0, "iou_0.50": np.inf},
    )
    assert nonfinite["pass"] is False


def test_r2_point_inference_matches_effective_outer_config_without_data_source() -> None:
    inference = OVM_ROOT / "config/tr3d/tr3d_ca1m_foreground_point_inference_xfit_r2.py"
    effective = WORK_ROOT / "outer_dev.py"
    result = validate_ca1m_point_inference_config(
        inference_path=inference,
        inference_sha256=inference_sha256(inference),
        effective_training_path=effective,
        effective_training_sha256=inference_sha256(effective),
    )
    assert result["architecture_matches_ca_training"] is True
    assert result["point_input_only"] is True
    assert result["ground_truth_access"] is False
    assert result["validation_access"] is False


def test_partial_effective_outer_config_already_satisfies_static_r2_contract() -> None:
    result = validate_effective_config(
        WORK_ROOT / "outer_dev.py", {"training": {"work_root": str(WORK_ROOT)}}
    )
    assert result["train_folds"] == [2, 3, 4]
    assert result["heldout_fold"] == 0
    assert result["optimizer_updates"] == 11268
    assert result["global_batch"] == 16
    assert result["initialization"] == "random_scratch_ca_only"


def test_fixed_config_preflight_is_pending_read_only_and_gt_free() -> None:
    _, cfg = load_config(EVAL_CONFIG)
    assert cfg["namespace"] == "ca1m_tr3d_xfit_r2_outer_dev_eval_v1"
    assert cfg["evaluation_stage"]["continuation_receipt"] == str(
        PIPELINE_ROOT
        / "reports/ca1m_tr3d_xfit_r2_outer_dev_eval_v1/continuation_receipt.json"
    )
    result = preflight(EVAL_CONFIG)
    assert result["ok"] is True
    assert result["read_only"] is True
    assert result["scene_count"] == 20
    assert result["fold1_access"] is False
    assert result["official_validation_access"] is False
    assert result["ground_truth_access"] is False
    assert result["gpu_started"] is False
    assert result["outer_wrapper_log"]["present"] is True
    if result["checkpoint_present"] is False:
        assert result["outer_wrapper_log"]["complete"] is False
    # This assertion remains true until the fixed post-training checkpoint is
    # available; preflight itself must never create it or a binding.
    if result["checkpoint_present"] is False:
        assert result["checkpoint_binding_ready"] is False


def test_runner_orders_sealed_exact20_collection_before_only_gt_call() -> None:
    source = (
        PIPELINE_ROOT / "tools/run_ca1m_tr3d_xfit_r2_outer_dev_eval.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    callers: dict[str, list[str]] = {}
    for name, function in functions.items():
        callers[name] = [
            node.func.id
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
    evaluate_calls = callers["evaluate"]
    assert evaluate_calls.index("_load_collection") < evaluate_calls.index(
        "_load_ground_truth"
    )
    assert [name for name, calls in callers.items() if "_load_ground_truth" in calls] == [
        "evaluate"
    ]
    assert "_load_ground_truth" not in callers["_run_proposals"]
    _, cfg = load_config(EVAL_CONFIG)
    assert cfg["ground_truth_after_proposal_seal_only"] is True
    assert cfg["proposal_stage"]["scene_count"] == 20
    assert cfg["proposal_stage"]["create_only"] is True


def test_formal_proposal_device_is_fixed_to_isolated_cuda_zero() -> None:
    assert _require_cuda_device("cuda:0") == "cuda:0"
    for invalid in ("cpu", "cuda", "cuda:1", "0"):
        with pytest.raises(ValueError):
            _require_cuda_device(invalid)


def test_outer_wrapper_log_requires_unique_success_terminal_and_r2_preamble(
    tmp_path: Path,
) -> None:
    work = (
        "/extra/ZhaoX/tr3d_ca1m_work_dirs/ca1m_xfit_v2_formal_r2/"
        "ca1m_xfit_v2_formal_outer_dev_seed0_r2"
    )
    required = "\n".join([
        "Formal CA-only TR3D asymmetric xfit-v2 R2: outer_dev",
        "  exact clean train_cfg: IterBasedTrainLoop(max_iters=11268)",
        "  LR milestones 7512,10329; global batch16; FP32; random scratch",
        "  no val/test loader; fold1 and official validation unopened",
        f"Completed formal R2 outer_dev: {work}/iter_11268.pth",
    ])
    path = tmp_path / "outer.log"
    path.write_text(required + "\nTRAIN_EXIT=0\n", encoding="utf-8")
    result = validate_outer_wrapper_log(path, require_fixed_source_path=False)
    assert result["terminal_line"] == "TRAIN_EXIT=0"
    assert result["role"] == "outer_dev"
    for bad in (
        required + "\n",
        required + "\nTRAIN_EXIT=0\nTRAIN_EXIT=0\n",
        required + "\nTRAIN_EXIT=1\n",
        required + "\nTraceback (most recent call last)\nTRAIN_EXIT=0\n",
    ):
        path.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError):
            validate_outer_wrapper_log(path, require_fixed_source_path=False)
