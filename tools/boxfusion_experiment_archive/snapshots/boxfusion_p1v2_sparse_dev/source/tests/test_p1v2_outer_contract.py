"""Outer isolation contracts for P1R/P1S."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from boxfusion.online_ablation import ONLINE_ABLATION_PROFILES
from boxfusion.p_ablation import (
    P_STAGE_ADDED_MODULE,
    P_STAGE_MODULE_MATRIX,
    P_STAGE_TO_PROFILE,
    apply_p_ablation,
)
from tools.build_p_run_manifest import _p1_provenance
from tools.validate_p1v2_run_artifacts import validate
import tools.report_p1v2_recall as recall_report


def _base():
    return {
        "online_refinement": {
            "quality": {"enabled": True, "soft_nms": {"enabled": True}},
            "residual_proposal": {"mode": "infer"},
            "occupancy_topk": {"enabled": True},
            "p2_local_mask_geometry": {"enabled": True},
            "p2_reliability_fusion": {"enabled": True},
        }
    }


@pytest.mark.parametrize(
    "stage,profile,head,target,added,sparse",
    [
        (
            "P1R",
            "p1r_snapshot_target_residual_observer",
            "per_voxel_mlp",
            "snapshot_inside_only",
            "snapshot_target_assignment",
            False,
        ),
        (
            "P1S",
            "p1s_native_sparse_context_observer",
            "native_sparse_context_v1",
            "snapshot_inside_only",
            "native_sparse_context",
            True,
        ),
    ],
)
def test_profile_isolates_exact_p1v2_change(
    stage, profile, head, target, added, sparse
):
    configured = apply_p_ablation(_base(), stage)
    online = configured["online_refinement"]
    residual = online["residual_proposal"]
    assert P_STAGE_TO_PROFILE[stage] == profile
    assert profile in ONLINE_ABLATION_PROFILES
    assert P_STAGE_ADDED_MODULE[stage] == added
    assert online["p_ablation_stage"] == stage
    assert online["p_ablation_profile"] == profile
    assert online["p_added_module"] == added
    assert residual["enabled"] is True
    assert residual["observer_only"] is True
    assert residual["mutate"] is False
    assert residual["collect_diagnostics"] is True
    assert residual["device"] == "cpu"
    assert residual["head_architecture"] == head
    assert residual["target_assignment_scope"] == target
    assert online["occupancy_topk"]["enabled"] is False
    assert online["p2_local_mask_geometry"]["enabled"] is False
    assert online["p2_reliability_fusion"]["enabled"] is False
    assert P_STAGE_MODULE_MATRIX[stage]["native_sparse_context"] is sparse


def test_p0_p1_share_explicit_historical_residual_contract():
    p0 = apply_p_ablation(_base(), "P0")["online_refinement"][
        "residual_proposal"
    ]
    p1 = apply_p_ablation(_base(), "P1")["online_refinement"][
        "residual_proposal"
    ]
    assert p0["head_architecture"] == p1["head_architecture"] == (
        "per_voxel_mlp"
    )
    assert p0["target_assignment_scope"] == p1[
        "target_assignment_scope"
    ] == "scene_global"


def _checkpoint(
    path: Path,
    *,
    head: str,
    target: str,
    b6_sha: str = "a" * 64,
    train_scenes=("scene0001_00",),
    forbidden_sha: str = "b" * 64,
) -> Path:
    model_config = {
        "input_dim": 14,
        "hidden_dim": 8,
        "regression_dim": 6,
    }
    if head == "native_sparse_context_v1":
        # NativeSparseResidualProposalHead.from_model_config is intentionally
        # strict and accepts ``architecture``, not an extra compatibility key.
        model_config["architecture"] = head
        model_config.update(
            {
                "dilations": [1, 2],
                "neighborhood": "axis6_multidilation_mean",
                "coordinate_layout": "xyz_or_batch_xyz",
                "regression_encoding": "center_delta_m_log_size_m",
            }
        )
    else:
        model_config["head_architecture"] = head
    torch.save(
        {
            "schema": "boxfusion.p1v2.test",
            "model_config": model_config,
            "training_config": {
                "target_assignment_scope": target,
            },
            "state_dict": {},
            "provenance": {
                "train_scene_ids": list(train_scenes),
                "forbidden_overlap": [],
                "train_scene_list_sha256": "c" * 64,
                "forbidden_scene_list_sha256": forbidden_sha,
                "b6_checkpoint_sha256": b6_sha,
            },
        },
        path,
    )
    return path


def test_manifest_recomputes_actual_train_val_overlap(tmp_path):
    forbidden = tmp_path / "val.txt"
    forbidden.write_text("scene0001_00\n", encoding="utf-8")
    forbidden_sha = hashlib.sha256(forbidden.read_bytes()).hexdigest()
    checkpoint = _checkpoint(
        tmp_path / "p1r.pt",
        head="per_voxel_mlp",
        target="snapshot_inside_only",
        forbidden_sha=forbidden_sha,
    )
    with pytest.raises(ValueError, match="overlap canonical validation"):
        _p1_provenance(
            checkpoint,
            expected_b6_sha256="a" * 64,
            forbidden_scene_list=forbidden,
            expected_head_architecture="per_voxel_mlp",
            expected_target_assignment_scope="snapshot_inside_only",
        )


def test_manifest_accepts_strict_sparse_architecture_key(tmp_path):
    forbidden = tmp_path / "val.txt"
    forbidden.write_text("scene0999_00\n", encoding="utf-8")
    forbidden_sha = hashlib.sha256(forbidden.read_bytes()).hexdigest()
    checkpoint = _checkpoint(
        tmp_path / "p1s.pt",
        head="native_sparse_context_v1",
        target="snapshot_inside_only",
        forbidden_sha=forbidden_sha,
    )
    provenance = _p1_provenance(
        checkpoint,
        expected_b6_sha256="a" * 64,
        forbidden_scene_list=forbidden,
        expected_head_architecture="native_sparse_context_v1",
        expected_target_assignment_scope="snapshot_inside_only",
    )
    assert provenance["head_architecture"] == "native_sparse_context_v1"
    assert provenance["target_assignment_scope"] == "snapshot_inside_only"


def _diagnostic(
    path: Path, *, stage: str, scene: str, checkpoint_sha: str
) -> None:
    profile, head = {
        "P1R": (
            "p1r_snapshot_target_residual_observer",
            "per_voxel_mlp",
        ),
        "P1S": (
            "p1s_native_sparse_context_observer",
            "native_sparse_context_v1",
        ),
    }[stage]
    np.savez_compressed(
        path,
        scene_id=np.asarray(scene),
        p1_schema=np.asarray("boxfusion.p1.test"),
        p1_stage=np.asarray(stage),
        p1_profile=np.asarray(profile),
        p1_enabled=np.asarray(True, dtype=bool),
        p1_observer_only=np.asarray(True, dtype=bool),
        p1_uses_ground_truth=np.asarray(False, dtype=bool),
        p1_reads_semantic_labels=np.asarray(False, dtype=bool),
        p1_mutation_enabled=np.asarray(False, dtype=bool),
        p1_applied_count=np.asarray(0, dtype=np.int64),
        p1_complete=np.asarray(True, dtype=bool),
        p1_class_agnostic=np.asarray(True, dtype=bool),
        p1_regression_dim=np.asarray(6, dtype=np.int64),
        p1_checkpoint_sha256=np.asarray(checkpoint_sha),
        p1_head_architecture=np.asarray(head),
        p1_target_assignment_scope=np.asarray("snapshot_inside_only"),
        p1_step_frame_ids=np.asarray([0], dtype=np.int64),
        p1_step_provider_steps=np.asarray([0], dtype=np.int64),
        p1_step_voxel_counts=np.asarray([2], dtype=np.int64),
        p1_step_candidate_counts=np.asarray([0], dtype=np.int64),
        p1_step_voxelize_seconds=np.asarray([0.1], dtype=np.float64),
        p1_step_head_seconds=np.asarray([0.2], dtype=np.float64),
        p1_step_nms_seconds=np.asarray([0.0], dtype=np.float64),
        p1_step_failed=np.asarray([False], dtype=bool),
        p1_candidate_boxes=np.empty((0, 6), dtype=np.float32),
        p1_candidate_corners=np.empty((0, 8, 3), dtype=np.float32),
        p1_candidate_scores=np.empty((0,), dtype=np.float32),
        p1_candidate_ids=np.empty((0,), dtype=np.str_),
    )


@pytest.mark.parametrize(
    "stage,head,name",
    [
        ("P1R", "per_voxel_mlp", "p1r.pt"),
        ("P1S", "native_sparse_context_v1", "p1s.pt"),
    ],
)
def test_artifact_validator_binds_stage_checkpoint_and_safety(
    tmp_path, stage, head, name
):
    scene = "scene0001_00"
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n", encoding="utf-8")
    checkpoint = _checkpoint(
        tmp_path / name,
        head=head,
        target="snapshot_inside_only",
    )
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    predictions = tmp_path / "predictions"
    diagnostics = tmp_path / "diagnostics"
    predictions.mkdir()
    diagnostics.mkdir()
    (predictions / f"{scene}_boxes.pkl").write_bytes(b"trusted-local")
    _diagnostic(
        diagnostics / f"{scene}_tracks.npz",
        stage=stage,
        scene=scene,
        checkpoint_sha=checkpoint_sha,
    )
    report = validate(
        stage=stage,
        scene_list=scene_list,
        prediction_root=predictions,
        diagnostics_root=diagnostics,
        expected_checkpoint=checkpoint,
    )
    assert report["ok"] is True
    assert report["steps"] == 1
    assert report["runtime_seconds"] == pytest.approx(0.3)

    with np.load(
        diagnostics / f"{scene}_tracks.npz", allow_pickle=False
    ) as old:
        payload = {key: np.array(old[key], copy=True) for key in old.files}
    payload["p1_mutation_enabled"] = np.asarray(True, dtype=bool)
    np.savez_compressed(diagnostics / f"{scene}_tracks.npz", **payload)
    with pytest.raises(ValueError, match="unsafe p1_mutation_enabled"):
        validate(
            stage=stage,
            scene_list=scene_list,
            prediction_root=predictions,
            diagnostics_root=diagnostics,
            expected_checkpoint=checkpoint,
        )


def test_protocol_scripts_are_present_and_p2_is_forbidden():
    runner = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_scannet_p1v2_ablation.sh"
    ).read_text(encoding="utf-8")
    assert "P1R" in runner and "P1S" in runner
    assert "unset BOXFUSION_P2_OCCUPANCY_CHECKPOINT" in runner
    assert "BOXFUSION_P1V2_FULL100" in runner


def _recall_payload(*, tp25: int, tp50: int) -> dict:
    return {
        "observer_only": True,
        "unsafe_scenes": [],
        "scene_count": 10,
        "ground_truth_count": 149,
        "p1_runtime_seconds_per_scene": 0.7,
        "candidates_per_scene": 200.0,
        "thresholds": {
            "0.25": {
                "novel_true_positives": tp25,
                "novel_recall_gain": tp25 / 149.0,
            },
            "0.50": {
                "novel_true_positives": tp50,
                "novel_recall_gain": tp50 / 149.0,
            },
        },
    }


def test_p1r_gate_requires_p1_noninferiority_and_two_ap50_tps(
    monkeypatch, tmp_path
):
    candidate = _recall_payload(tp25=10, tp50=2)
    reference = {
        **_recall_payload(tp25=10, tp50=0),
        "stage": "P1",
    }
    monkeypatch.setattr(
        recall_report, "read_scene_ids", lambda _path: ("scene0001_00",)
    )
    monkeypatch.setattr(
        recall_report, "evaluate", lambda **_kwargs: dict(candidate)
    )
    monkeypatch.setattr(
        recall_report,
        "_load_reference",
        lambda _path, expected_stage: dict(reference),
    )
    report = recall_report.build_report(
        stage="P1R",
        scene_list=tmp_path / "scenes.txt",
        prediction_root=tmp_path,
        diagnostics_root=tmp_path,
        gt_root=tmp_path,
        scans_root=tmp_path,
        reference_report=tmp_path / "p1.json",
    )
    assert report["fixed10_go_no_go"]["passes"] is True
    assert report["fixed10_go_no_go"]["decision"] == "GO_FULL100"

    candidate["thresholds"]["0.50"]["novel_true_positives"] = 1
    candidate["thresholds"]["0.50"]["novel_recall_gain"] = 1 / 149.0
    failed = recall_report.build_report(
        stage="P1R",
        scene_list=tmp_path / "scenes.txt",
        prediction_root=tmp_path,
        diagnostics_root=tmp_path,
        gt_root=tmp_path,
        scans_root=tmp_path,
        reference_report=tmp_path / "p1.json",
    )
    assert failed["fixed10_go_no_go"]["passes"] is False
    assert failed["fixed10_go_no_go"]["decision"] == "STOP_P1R"
