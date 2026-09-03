from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from boxfusion.ca1m_tr3d_terminal import (
    aligned_boxes_to_world_corners,
    associate_terminal_candidates,
)
from boxfusion.ca1m_tr3d_terminal_v4 import (
    OverlaySummary,
    ProposalCacheSummary,
    derive_demo_gap20_early_finalize_frame_ids,
    frame_lineage_json,
    load_overlay_cache,
    load_proposal_cache,
    overlay_payload,
    proposal_cache_payload,
    sha256_bytes,
    write_npz_create_only,
)
from boxfusion.ca1m_tr3d_worker_client import (
    CA1MTR3DWorker,
    INFERENCE_TIMEOUT_S,
    MAX_STARTUP_TIMEOUT_S,
    STARTUP_ABORT_GRACE_S,
)
from tools import overlay_ca1m_tr3d_terminal_v4 as overlay_v4
from tools import run_ca1m_tr3d_proposal_cache_v4 as proposal_v4
from tools.preflight_ca1m_tr3d_terminal_train100_v4 import validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/ca1m_tr3d_terminal_train100_v4_p5.json"
BINDING_SHA = "19b8c3d12de8dd8d3ffff1413c6c6003a5ccb1a10cf213b972ebd43fa9db5043"


def _proposal(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    scene = "12345678"
    transform = np.eye(4, dtype=np.float64)
    local = np.asarray([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]], dtype=np.float32)
    corners = aligned_boxes_to_world_corners(local, transform)
    frames = derive_demo_gap20_early_finalize_frame_ids(42)
    lineage = frame_lineage_json(scene, 42)
    code = json.dumps(
        {"files": {}, "schema": "boxfusion.ca1m_tr3d_proposal_code_manifest.v4"},
        separators=(",", ":"),
        sort_keys=True,
    )
    summary = ProposalCacheSummary(
        scene_id=scene,
        frame_count=42,
        used_frame_count=len(frames),
        point_count=100,
        candidate_count=1,
        model_runtime_s=0.25,
        source_points_sha256="a" * 64,
        frame_lineage_sha256=sha256_bytes(lineage.encode()),
        checkpoint_binding_sha256=BINDING_SHA,
        checkpoint_sha256="b" * 64,
        config_sha256="c" * 64,
        code_manifest_sha256=sha256_bytes(code.encode()),
        adapter_mode="genuine",
        device="cuda:0",
    )
    payload = proposal_cache_payload(
        summary=summary,
        used_frame_ids=frames,
        world_to_local=transform,
        candidate_corners_world=corners,
        candidate_scores=np.asarray([0.8], dtype=np.float32),
        candidate_point_count=np.asarray([12], dtype=np.int64),
        candidate_boxes_local=local,
        candidate_labels=np.asarray([0], dtype=np.int64),
        frame_lineage=lineage,
        code_manifest=code,
    )
    target = tmp_path / f"{scene}_ca1m_tr3d_proposals_v4.npz"
    write_npz_create_only(target, payload)
    return target, payload


def test_demo_early_finalize_lineage_is_derived_without_anchor_or_b6():
    assert derive_demo_gap20_early_finalize_frame_ids(1).tolist() == [0]
    assert derive_demo_gap20_early_finalize_frame_ids(41).tolist() == [0, 20]
    assert derive_demo_gap20_early_finalize_frame_ids(42).tolist() == [0, 20]
    assert derive_demo_gap20_early_finalize_frame_ids(326).tolist() == list(range(0, 301, 20))
    with pytest.raises(ValueError, match="gap=20"):
        derive_demo_gap20_early_finalize_frame_ids(42, gap=10)


def test_proposal_cache_is_create_only_validated_and_anchor_free(tmp_path: Path):
    target, payload = _proposal(tmp_path)
    loaded = load_proposal_cache(
        target, expected_scene="12345678", expected_binding_sha256=BINDING_SHA
    )
    assert loaded["summary"].candidate_count == 1
    assert bool(np.asarray(loaded["ground_truth_access"]).item()) is False
    assert bool(np.asarray(loaded["anchor_access"]).item()) is False
    assert bool(np.asarray(loaded["b6_access"]).item()) is False
    assert target.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        write_npz_create_only(target, payload)


def test_cpu_overlay_consumes_proposal_cache_without_rerunning_tr3d(tmp_path: Path):
    proposal_path, _ = _proposal(tmp_path)
    proposal = load_proposal_cache(proposal_path, expected_scene="12345678")
    anchors = np.array(proposal["candidate_corners_world"], copy=True)
    scores = np.asarray([0.5], dtype=np.float32)
    association = associate_terminal_candidates(
        anchor_corners=anchors,
        anchor_scores=scores,
        candidate_corners=proposal["candidate_corners_world"],
        candidate_scores=proposal["candidate_scores"],
        near_iou=0.15,
    )
    summary = OverlaySummary(
        scene_id="12345678",
        anchor_count=1,
        candidate_count=1,
        near_candidate_count=1,
        represented_anchor_count=1,
        proposal_cache_sha256=proposal["sha256"],
        final_anchor_sha256="d" * 64,
        final_anchor_manifest_sha256="e" * 64,
        native_b6_diagnostic_sha256="f" * 64,
        native_b6_collection_manifest_sha256="1" * 64,
        native_b6_checkpoint_sha256="2" * 64,
        native_b6_checkpoint_manifest_sha256="3" * 64,
        active_anchor_scores_sha256="4" * 64,
    )
    target = tmp_path / "12345678_ca1m_tr3d_overlay_v4.npz"
    write_npz_create_only(
        target,
        overlay_payload(
            summary=summary,
            anchor_corners=anchors,
            anchor_scores=scores,
            proposal=proposal,
            association=association,
        ),
    )
    loaded = load_overlay_cache(
        target,
        expected_scene="12345678",
        expected_proposal_sha256=proposal["sha256"],
    )
    assert loaded["summary"].near_candidate_count == 1
    assert bool(np.asarray(loaded["cpu_only"]).item()) is True
    assert bool(np.asarray(loaded["ground_truth_access"]).item()) is False


def test_real_static_preflight_proves_stage_p_isolated_and_overlay_blocked():
    report = validate_config(CONFIG)
    assert report["scene_count"] == 100
    assert report["processed_rgbd_inventory"]["frame_count"] == 61189
    assert report["processed_rgbd_inventory"]["demo_gap20_early_finalize_frame_count"] == 3010
    assert report["processed_rgbd_inventory"]["annotation_files_opened"] is False
    assert report["distribution_parity"] == {
        "receipt_path": str(
            ROOT
            / "manifests/ca1m_tr3d_terminal_ca_native_train100_v4/lineage_training_point_parity_v4.json"
        ),
        "receipt_sha256": "35d9dfafc7272d92d98c97c6ef23f4323432e9bd0af5045bc5f78b1ae9afa00d",
        "lineage_scenes": 100,
        "point_array_scenes": 100,
        "point_byte_scenes": 100,
        "training_distribution_points": 24382287,
        "old_diagnostic_runtime_dependency": False,
        "training_point_runtime_dependency": False,
    }
    assert report["proposal_cache_depends_on_anchor_or_b6"] is False
    assert report["proposal_stage_contract_ready"] is True
    assert report["proposal_stage_runtime_authorized"] is True
    assert report["proposal_authorization"] == {
        "path": str(
            ROOT
            / "manifests/ca1m_tr3d_terminal_ca_native_train100_v4/proposal_stage_authorization_v5.json"
        ),
        "sha256": "42c1580b99a83e1f6c44ac27428596dfc5ae1f141635d63d07cc1c2e7f09ae25",
        "decision": "ALLOW_STAGE_P_ONLY",
        "scope": "stage_p_only",
        "point_hash_precheck_before_gpu": True,
    }
    assert report["overlay_inputs_bound"] is False
    assert report["overlay_stage_runtime_authorized"] is False
    assert report["full_run_authorized"] is False
    assert report["tr3d_runtime_policy"] == {
        "startup_timeout_s": 600,
        "startup_timeout_bounded": True,
        "startup_timeout_config_bound": True,
        "startup_failure_reaps_worker": True,
        "startup_stack_dump_interval_s": 120,
        "worker_pipe_mode": "binary_unbuffered",
        "worker_pipe_read": "nonblocking_os_read_with_persistent_byte_buffer",
        "inference_timeout_s": 120,
        "inference_timeout_force_reap": True,
        "inference_stack_dump_interval_s": 30,
        "inference_phase_markers": [
            "request", "pre_sync", "pipeline", "test_step", "post_sync",
            "response",
        ],
    }
    assert report["checkpoint_binding"]["checkpoint_sha256"] == (
        "d3ba6cc22f0a1a11ab47e55ccdd21c2ef4a84efaf3c6359b7e8231a6c8d3b4a7"
    )


def test_checked_in_overlay_fails_before_artifact_access(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("worker/artifact access must not happen")

    monkeypatch.setattr(overlay_v4, "load_proposal_cache", forbidden)
    with pytest.raises(PermissionError, match="CPU overlay is not authorized"):
        overlay_v4.run(argparse.Namespace(collection_config=CONFIG, scene=[]))


def test_proposal_parser_has_no_anchor_b6_or_raw_model_argument():
    destinations = {action.dest for action in proposal_v4.parser()._actions}
    assert destinations == {"help", "collection_config", "scene", "device"}
    overlay_destinations = {action.dest for action in overlay_v4.parser()._actions}
    assert overlay_destinations == {"help", "collection_config", "scene"}
    source = (ROOT / "tools/run_ca1m_tr3d_proposal_cache_v4.py").read_text()
    assert source.index("Fail before constructing the GPU worker") < source.index(
        "with CA1MTR3DWorker"
    )


def test_config_forbids_old_terminal_and_old_b6_and_has_pending_overlay():
    value = json.loads(CONFIG.read_text())
    assert value["proposal_stage"]["anchor_access"] is False
    assert value["proposal_stage"]["b6_access"] is False
    assert value["proposal_stage"]["run_authorized"] is True
    assert value["proposal_stage"]["authorization_scope"] == "stage_p_only"
    assert value["proposal_stage"]["authorization_receipt_sha256"] == (
        "42c1580b99a83e1f6c44ac27428596dfc5ae1f141635d63d07cc1c2e7f09ae25"
    )
    assert value["proposal_stage"]["resume_policy"] == (
        "validate_complete_then_skip_else_fail"
    )
    assert value["overlay_stage"]["cpu_only"] is True
    assert value["frame_lineage"]["include_last"] is False
    assert value["frame_lineage"]["schema"].endswith(
        "demo_gap20_early_finalize_lineage.v1"
    )
    assert value["distribution_parity"]["status"] == "exact100_pass"
    assert value["distribution_parity"][
        "old_native_b6_diagnostic_runtime_dependency"
    ] is False
    assert value["tr3d_runtime"]["startup_timeout_s"] == 600
    pending = {
        key: item
        for key, item in value["overlay_stage"].items()
        if key.startswith("final_anchor_") or key.startswith("native_b6_v2_")
    }
    assert pending and all(item is None for item in pending.values())
    assert value["ca_native_tr3d_binding"]["initialization"] == "ca1m_random_scratch"
    assert value["forbidden_reuse"]["old_terminal_artifact_access"] is False
    assert value["forbidden_reuse"]["old_native_b6_artifact_access"] is False


def test_protocol_revision5_is_bounded_and_supersedes_revision4():
    assert MAX_STARTUP_TIMEOUT_S == 600.0
    assert STARTUP_ABORT_GRACE_S == 5.0
    assert INFERENCE_TIMEOUT_S == 120.0
    for invalid in (False, 0, -1, float("inf"), 600.01):
        with pytest.raises(ValueError, match="startup timeout"):
            CA1MTR3DWorker(
                python="missing-python",
                worker_script="missing-worker",
                runtime_root="missing-runtime",
                config="missing-config",
                checkpoint="missing-checkpoint",
                project_root="missing-project",
                vendor_root="missing-vendor",
                startup_timeout_s=invalid,
            )
    authorization = json.loads(
        (
            ROOT
            / "manifests/ca1m_tr3d_terminal_ca_native_train100_v4/"
            "proposal_stage_authorization_v5.json"
        ).read_text()
    )
    assert authorization["authorization_revision"] == 5
    assert authorization["runtime_policy"]["startup_timeout_s"] == 600
    assert authorization["runtime_policy"]["startup_timeout_bounded"] is True
    assert authorization["runtime_policy"]["startup_failure_reaps_worker"] is True
    assert authorization["runtime_policy"]["worker_pipe_mode"] == "binary_unbuffered"
    assert authorization["runtime_policy"]["inference_timeout_force_reap"] is True
    assert authorization["supersedes"]["authorization_sha256"] == (
        "de3063748ea757ae041b5a22112df47a1da46d71c809bc508914fc32032f7309"
    )


def test_binary_response_parser_handles_read_ahead_across_nonprotocol_lines():
    class Process:
        def __init__(self, stdout):
            self.stdout = stdout

        @staticmethod
        def poll():
            return None

    read_fd, write_fd = os.pipe()
    stdout = os.fdopen(read_fd, "rb", buffering=0)
    try:
        os.set_blocking(read_fd, False)
        worker = object.__new__(CA1MTR3DWorker)
        worker._stdout_buffer = bytearray()
        worker.process = Process(stdout)
        os.write(
            write_fd,
            b"ordinary OpenMMLab log\n"
            b"BOXFUSION_TR3D_RESPONSE "
            b'{"status":"ready","synthetic":false,"startup_s":2.29}\n',
        )
        response = worker._response(timeout_s=0.5)
        assert response["status"] == "ready"
        assert response["startup_s"] == 2.29
    finally:
        os.close(write_fd)
        stdout.close()


def test_launcher_is_valid_bash_and_run_is_fail_closed():
    launcher = ROOT / "scripts/collect_ca1m_tr3d_terminal_train100_v4.sh"
    syntax = subprocess.run(
        ["bash", "-n", str(launcher)], capture_output=True, text=True, check=False
    )
    assert syntax.returncode == 0, syntax.stderr
    result = subprocess.run(
        ["bash", str(launcher), "--run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--run is fail-closed" in result.stderr
    assert '"gpu_started": false' in result.stdout
