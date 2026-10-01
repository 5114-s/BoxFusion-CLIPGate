from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import boxfusion.ca1m_tr3d_e961_terminal_inputs_v5_r2 as route
from boxfusion.ca1m_tr3d_terminal import associate_terminal_candidates


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "tools/preflight_ca1m_tr3d_e961_terminal_inputs_v5_r2.py"
RUNNER = ROOT / "tools/run_ca1m_tr3d_e961_terminal_inputs_v5_r2.py"


def _corners(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0)) -> np.ndarray:
    center = np.asarray(center, np.float32); half = np.asarray(size, np.float32) / 2
    low, high = center - half, center + half
    return np.asarray([
        [low[0], low[1], low[2]], [low[0], low[1], high[2]],
        [low[0], high[1], low[2]], [low[0], high[1], high[2]],
        [high[0], low[1], low[2]], [high[0], low[1], high[2]],
        [high[0], high[1], low[2]], [high[0], high[1], high[2]],
    ], np.float32)


def test_static_pending_passes_with_frozen_new_inner_schema():
    report = route.validate_static_config()
    assert report["status"] == "PASS_STATIC_PENDING"
    assert report["scene_count"] == 80
    assert report["inner_receipt_schema"] == route.INNER_SCHEMA
    assert report["legacy_inner_receipt_rejected"] == route.LEGACY_INNER_SCHEMA
    cfg = json.loads(route.DEFAULT_CONFIG.read_text())
    for role in route.ROLE_ORDER[1:]:
        assert cfg["scene_contract"]["roles"][role]["source_success_receipt"]["schema"] == route.INNER_SCHEMA
    assert cfg["producer_contracts"]["inner"]["verifier"]["sha256"] == "d6d7a6c30f15d6f11a8c7e84b9e27d08665f7d4af1ebe0ae7ca4aee68aec9f03"


def test_operational_pending_stops_before_any_output_or_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(route, "_load_module", lambda *_a, **_k: pytest.fail("verifier import reached"))
    monkeypatch.setattr(route, "ensure_directory", lambda *_a, **_k: pytest.fail("mkdir reached"))
    with pytest.raises(route.PendingOperationalInputs):
        route.validate_operational_ready(route.DEFAULT_CONFIG)
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("program", [PREFLIGHT, RUNNER])
def test_cli_operational_pending_exit3(program: Path):
    args = [sys.executable, str(program)]
    args += ["--operational"] if program == PREFLIGHT else ["--operational-preflight"]
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 3
    value = json.loads(result.stderr)
    assert value["status"] == "BLOCKED_PENDING"
    assert value["gpu_started"] is False and value["output_created"] is False


def _proposal(tmp_path: Path, scene: str = "10000000") -> dict:
    summary = route.ProposalSummaryR2(
        scene, "outer_dev", 0, 3, 2, 100, 1, 0.01, "1" * 64,
        "2" * 64, "3" * 64, "4" * 64, "genuine", "cuda:0",
    )
    payload = route.proposal_payload(
        summary=summary, used_frame_ids=np.asarray([0, 20], np.int64),
        world_to_local=np.eye(4, dtype=np.float64),
        candidate_corners_world=_corners()[None],
        candidate_scores=np.asarray([0.7], np.float32),
        candidate_point_count=np.asarray([30], np.int64),
        candidate_boxes_local=np.asarray([[0, 0, 0, 2, 2, 2, 0]], np.float32),
        candidate_labels=np.asarray([0], np.int64),
    )
    path = tmp_path / f"{scene}_anchor_free_v5_r2.npz"
    route.write_npz_exclusive(path, payload)
    return route.load_proposal(path, expected_scene=scene)


def test_proposal_roundtrip_and_create_only(tmp_path: Path):
    loaded = _proposal(tmp_path)
    assert loaded["summary"].candidate_count == 1
    assert loaded["summary"].adapter_mode == "genuine"
    with pytest.raises(FileExistsError):
        route.write_bytes_exclusive(loaded["path"], b"overwrite")
    assert not (loaded["path"].stat().st_mode & 0o222)


def test_proposal_rejects_nonfinite():
    summary = route.ProposalSummaryR2(
        "10000000", "outer_dev", 0, 1, 1, 1, 1, 0.0,
        "1" * 64, "2" * 64, "3" * 64, "4" * 64, "genuine", "cuda:0",
    )
    with pytest.raises(ValueError, match="malformed"):
        route.proposal_payload(
            summary=summary, used_frame_ids=np.asarray([0]), world_to_local=np.eye(4),
            candidate_corners_world=np.full((1, 8, 3), np.nan, np.float32),
            candidate_scores=np.asarray([0.5], np.float32),
            candidate_point_count=np.asarray([0]),
            candidate_boxes_local=np.ones((1, 7), np.float32), candidate_labels=np.asarray([0]),
        )


def test_overlay_roundtrip_uses_oof_and_recomputes_association(tmp_path: Path):
    proposal = _proposal(tmp_path)
    anchors = _corners()[None]
    detector = np.asarray([0.2], np.float32); oof = np.asarray([0.9], np.float32)
    native = np.full((1, 14), 0.25, np.float32); native[:, 0] = detector
    association = associate_terminal_candidates(
        anchor_corners=anchors, anchor_scores=oof,
        candidate_corners=proposal["candidate_corners_world"],
        candidate_scores=proposal["candidate_scores"], near_iou=0.15,
    )
    summary = route.OverlaySummaryR2(
        "10000000", "outer_dev", 0, 1, 1, 1, proposal["sha256"],
        "5" * 64, "6" * 64, "7" * 64, "4" * 64,
    )
    payload = route.overlay_payload(
        summary=summary, anchor_corners=anchors, anchor_detector_scores=detector,
        anchor_native_features=native, anchor_scores_oof=oof, proposal=proposal,
        best_anchor_indices=association.best_anchor_indices,
        best_anchor_iou=association.best_anchor_iou,
        best_anchor_center_distance_m=association.best_anchor_center_distance_m,
        near_mask=association.near_mask,
    )
    path = tmp_path / "10000000_oof_overlay_v5_r2.npz"
    route.write_npz_exclusive(path, payload)
    loaded = route.load_overlay(path, proposal=proposal, expected_scene="10000000")
    assert np.array_equal(loaded["anchor_scores_oof"], oof)
    assert not np.array_equal(loaded["anchor_scores_oof"], detector)


def test_stage_e_synthetic_builds_40d_generic_v5_evidence(tmp_path: Path):
    from boxfusion.ca1m_tr3d_terminal_gate_v5 import load_candidate_evidence_v5
    ctx = _fake_context(tmp_path)
    normalized = route.ensure_runtime_namespace(ctx)["outer_dev"]
    proposal = _proposal(tmp_path, scene=ctx.roles["outer_dev"].scenes[0])
    anchors = _corners()[None]
    detector = np.asarray([0.2], np.float32); oof = np.asarray([0.8], np.float32)
    anchor_native = np.full((1, 14), 0.25, np.float32); anchor_native[:, 0] = detector
    candidate_native = np.full((1, 14), 0.3, np.float32)
    candidate_native[:, 0] = proposal["candidate_scores"]
    association = associate_terminal_candidates(
        anchor_corners=anchors, anchor_scores=oof,
        candidate_corners=proposal["candidate_corners_world"],
        candidate_scores=proposal["candidate_scores"], near_iou=0.15,
    )
    overlay = {
        "summary": route.OverlaySummaryR2(
            proposal["summary"].scene_id, "outer_dev", 0, 1, 1, 1,
            proposal["sha256"], "5" * 64, "6" * 64,
            route.sha256_file(Path(ctx.config["anchor_inputs"]["native_b6_oof_sidecar"]["path"])),
            ctx.authorization_sha256,
        ),
        "anchor_corners": anchors, "anchor_detector_scores": detector,
        "anchor_native_features": anchor_native, "anchor_scores_oof": oof,
        "candidate_corners_world": proposal["candidate_corners_world"],
        "candidate_scores": proposal["candidate_scores"],
        "best_anchor_indices": association.best_anchor_indices,
        "best_anchor_iou": association.best_anchor_iou,
        "best_anchor_center_distance_m": association.best_anchor_center_distance_m,
        "near_mask": association.near_mask,
    }
    target = tmp_path / f"{proposal['summary'].scene_id}_ca1m_tr3d_candidate_evidence_xfit_r2_v5.npz"
    route._write_generic_evidence(
        target, scene=proposal["summary"].scene_id, role="outer_dev", ctx=ctx,
        normalized_receipt=normalized, proposal=proposal, overlay=overlay,
        candidate_native=candidate_native,
    )
    evidence = load_candidate_evidence_v5(target)
    assert evidence.features.shape == (1, 40)
    assert np.array_equal(evidence.anchor_scores, oof)


def _fake_context(tmp_path: Path) -> route.ReadyContext:
    root = tmp_path / route.NAMESPACE
    continuation = tmp_path / "continuation.json"
    continuation.write_text(json.dumps({
        "schema": route.CONTINUATION_SCHEMA, "complete": True, "pass": True,
        "continue_inner_training_authorized": True,
        "authorized_inner_roles": list(route.ROLE_ORDER[1:]),
        "fold1_access": False, "official_validation_access": False,
        "checkpoint_selection": False,
    }))
    continuation.chmod(0o444)
    authorization = tmp_path / "authorization.json"; authorization.write_text("{}\n"); authorization.chmod(0o444)
    checkpoint = tmp_path / "iter_11268.pth"; checkpoint.write_bytes(b"checkpoint"); checkpoint.chmod(0o444)
    b6 = tmp_path / "oof.npz"; b6.write_bytes(b"sealed-oof"); b6.chmod(0o444)
    roles = {}
    prefixes = {"outer_dev": 10, "inner_holdout2": 20, "inner_holdout3": 30, "inner_holdout4": 40}
    for role in route.ROLE_ORDER:
        receipt = tmp_path / f"{role}.json"; receipt.write_text("{}\n"); receipt.chmod(0o444)
        train, fold, _ = route.ROLE_SPECS[role]
        scenes = tuple(f"{prefixes[role] * 1000000 + index:08d}" for index in range(20))
        roles[role] = route.VerifiedRole(
            role, train, fold, scenes, receipt, route.sha256_file(receipt), {},
            checkpoint, route.sha256_file(checkpoint),
        )
    cfg = {
        "namespace": route.NAMESPACE,
        "run_authorization": {"state": "bound", "path": str(authorization), "sha256": route.sha256_file(authorization), "schema": route.AUTH_SCHEMA},
        "outputs": {
            "namespace_root": str(root), "receipt_root": str(root / "normalized_receipts"),
            "evidence_root": str(root / "evidence"), "manifest_root": str(root / "manifests"),
            "combined_manifest": str(root / "manifests/CANDIDATE_COLLECTION_EXACT80.json"),
        },
        "anchor_inputs": {"native_b6_oof_sidecar": {"path": str(b6)}},
    }
    return route.ReadyContext(
        tmp_path / "ready.json", cfg, authorization, route.sha256_file(authorization),
        continuation, route.sha256_file(continuation), roles,
    )


def test_stage_m_synthetic_exact80_and_generic_loader(tmp_path: Path):
    from boxfusion.ca1m_tr3d_terminal_gate_v5 import _evidence_payload, load_candidate_collection_v5
    ctx = _fake_context(tmp_path)
    normalized = route.ensure_runtime_namespace(ctx)
    b6_sha = route.sha256_file(Path(ctx.config["anchor_inputs"]["native_b6_oof_sidecar"]["path"]))
    corners = _corners()[None]
    for role in route.ROLE_ORDER:
        evidence_root = Path(ctx.config["outputs"]["evidence_root"]) / role
        route.ensure_directory(evidence_root)
        for scene in ctx.roles[role].scenes:
            payload = _evidence_payload(
                scene_id=scene, fold_id=ctx.roles[role].heldout_fold, producer_role=role,
                producer_checkpoint_sha256=ctx.roles[role].checkpoint_sha256,
                training_receipt_sha256=route.sha256_file(normalized[role]),
                outer_continuation_receipt_sha256=ctx.continuation_sha256,
                b6_oof_sidecar_sha256=b6_sha, candidate_corners=corners,
                candidate_rows=np.asarray([0], np.int64), candidate_scores=np.asarray([0.6], np.float32),
                anchor_indices=np.asarray([0], np.int64), features=np.zeros((1, 40), np.float32),
                anchor_corners=corners, anchor_scores=np.asarray([0.7], np.float32),
            )
            route.write_npz_exclusive(route._evidence_path(ctx.config, role, scene), payload)
    result = route.seal_stage_m(ctx)
    assert result["scene_count"] == 80 and result["fit_scene_count"] == 60
    loaded = load_candidate_collection_v5(Path(ctx.config["outputs"]["combined_manifest"]))
    assert len(loaded.scenes) == 80


def test_no_ground_truth_fold1_or_validation_code_surface():
    text = Path(route.__file__).read_text()
    assert "load_ground_truth" not in text
    cfg = json.loads(route.DEFAULT_CONFIG.read_text())
    assert cfg["access"]["ground_truth_access"] is False
    assert cfg["access"]["fold1_path_present"] is False
    assert cfg["access"]["official_validation_path_present"] is False
