#!/usr/bin/env python3
"""Create the immutable stage-P-only authorization receipt for terminal v4."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_checkpoint_binding import (  # noqa: E402
    load_checkpoint_binding,
    regular_directory,
    regular_file,
)
from boxfusion.ca1m_tr3d_terminal_v4 import sha256_file  # noqa: E402
from boxfusion.ca1m_tr3d_inference_contract import (  # noqa: E402
    SCHEMA as INFERENCE_CONFIG_SCHEMA,
    validate_ca1m_point_inference_config,
)
from boxfusion.ca1m_tr3d_worker_client import (  # noqa: E402
    INFERENCE_TIMEOUT_S,
    MAX_STARTUP_TIMEOUT_S,
    STARTUP_ABORT_GRACE_S,
)


SCHEMA = "boxfusion.ca1m_tr3d_v4_proposal_authorization.v1"
PARITY_SHA = "35d9dfafc7272d92d98c97c6ef23f4323432e9bd0af5045bc5f78b1ae9afa00d"
BINDING_SHA = "19b8c3d12de8dd8d3ffff1413c6c6003a5ccb1a10cf213b972ebd43fa9db5043"
SCENE_SHA = "35321e9942dc5d512db2952b9ca6228b1291127e0c13fd92aa458f2d7eb7f9fd"
INFERENCE_CONFIG_SHA = "60a0e626d671a8b0270006143a062de69ebdd3d9516d5d47c81a6cec2dcd5da4"
OLD_AUTHORIZATION_SHA = "de3063748ea757ae041b5a22112df47a1da46d71c809bc508914fc32032f7309"


def _create_only(path: Path, value: dict) -> Path:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite proposal authorization: {target}") from error
        target.chmod(0o444)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def payload() -> dict:
    parity = regular_file(
        ROOT
        / "manifests/ca1m_tr3d_terminal_ca_native_train100_v4/lineage_training_point_parity_v4.json",
        "exact100 parity receipt",
    )
    binding_path = regular_file(
        ROOT
        / "manifests/ca1m_tr3d_terminal_ca_native_train100_v3/checkpoint_binding.json",
        "CA scratch checkpoint binding",
    )
    scene_list = regular_file(
        Path("/extra/ZhaoX/tr3d_ca1m_train100_v1/splits/train100.txt"),
        "exact100 scene list",
    )
    if sha256_file(parity) != PARITY_SHA or sha256_file(binding_path) != BINDING_SHA:
        raise ValueError("authorization prerequisites changed")
    if sha256_file(scene_list) != SCENE_SHA:
        raise ValueError("authorization exact100 scene contract changed")
    parity_value = json.loads(parity.read_text())
    if (
        parity_value.get("lineage_parity_scene_count") != 100
        or parity_value.get("point_array_parity_scene_count") != 100
        or parity_value.get("point_byte_parity_scene_count") != 100
        or parity_value.get("ground_truth_access") is not False
        or parity_value.get("gpu_started") is not False
    ):
        raise ValueError("authorization parity prerequisite is incomplete")
    binding = load_checkpoint_binding(binding_path)
    inference_config = regular_file(
        Path(
            "/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/config/tr3d/"
            "tr3d_ca1m_foreground_point_inference_v1.py"
        ),
        "CA-only point-inference config",
    )
    inference_contract = validate_ca1m_point_inference_config(
        inference_path=inference_config,
        inference_sha256=INFERENCE_CONFIG_SHA,
        effective_training_path=binding.effective_config_path,
        effective_training_sha256=binding.effective_config_sha256,
    )
    processed = regular_directory(
        Path("/extra/ZhaoX/boxfusion_ca1m_native_b6_train100_v1"),
        "processed train100 RGB-D root",
    )
    runtime_config = regular_file(
        ROOT / "config/ca1m_tr3d_terminal_train100_v4_p5.json",
        "stage-P runtime revision-5 config",
    )
    runtime_config_value = json.loads(runtime_config.read_text())
    runtime = runtime_config_value.get("tr3d_runtime") or {}
    if (
        runtime.get("startup_timeout_s") != 600
        or MAX_STARTUP_TIMEOUT_S != 600.0
        or STARTUP_ABORT_GRACE_S != 5.0
        or INFERENCE_TIMEOUT_S != 120.0
    ):
        raise ValueError("bounded Stage-P runtime policy changed")
    sources = {
        "proposal_runner": ROOT / "tools/run_ca1m_tr3d_proposal_cache_v4.py",
        "proposal_contract": ROOT / "boxfusion/ca1m_tr3d_terminal_v4.py",
        "checkpoint_binding_loader": ROOT / "boxfusion/ca1m_tr3d_checkpoint_binding.py",
        "backprojection": ROOT / "boxfusion/tr3d_incremental_online.py",
        "terminal_geometry": ROOT / "boxfusion/ca1m_tr3d_terminal.py",
        "worker_client": ROOT / "boxfusion/ca1m_tr3d_worker_client.py",
        "worker": ROOT / "tools/ca1m_tr3d_terminal_worker.py",
        "official_adapter": Path(
            "/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev/boxfusion/tr3d_inference.py"
        ),
        "ca_point_inference_contract": ROOT / "boxfusion/ca1m_tr3d_inference_contract.py",
        "ca_point_inference_config": inference_config,
    }
    code = {
        name: {
            "path": str(regular_file(path, f"stage-P code {name}")),
            "sha256": sha256_file(path),
        }
        for name, path in sorted(sources.items())
    }
    return {
        "schema": SCHEMA,
        "authorization_revision": 5,
        "complete": True,
        "create_only": True,
        "authorization_decision": "ALLOW_STAGE_P_ONLY",
        "proposal_gpu_execution_authorized": True,
        "full_two_stage_run_authorized": False,
        "overlay_execution_authorized": False,
        "ground_truth_access": False,
        "validation_ground_truth_access": False,
        "final_anchor_access": False,
        "native_b6_access": False,
        "old_terminal_cache_access": False,
        "scannet_checkpoint_or_config_access": False,
        "supersedes": {
            "authorization_path": str(
                ROOT
                / "manifests/ca1m_tr3d_terminal_ca_native_train100_v4/"
                "proposal_stage_authorization_v4.json"
            ),
            "authorization_sha256": OLD_AUTHORIZATION_SHA,
            "reason": (
                "revision 4 reached worker-ready on stderr, but TextIOWrapper read-ahead "
                "stranded the newline-framed READY record outside selector visibility"
            ),
            "old_receipt_overwritten": False,
            "old_proposal_artifact_count": 0,
        },
        "required_runtime_config": {
            "path": str(runtime_config),
            "startup_timeout_s": 600,
            "sha256_omitted_to_avoid_authorization_hash_cycle": True,
        },
        "runtime_policy": {
            "startup_timeout_s": 600,
            "startup_timeout_config_field": "tr3d_runtime.startup_timeout_s",
            "startup_timeout_bounded": True,
            "client_max_startup_timeout_s": 600,
            "startup_abort_grace_s": 5,
            "inference_timeout_s": 120,
            "startup_failure_reaps_worker": True,
            "startup_stack_dump_interval_s": 120,
            "worker_ready_flush": True,
            "worker_pipe_mode": "binary_unbuffered",
            "worker_pipe_read": "nonblocking_os_read_with_persistent_byte_buffer",
            "protocol_framing": "newline_delimited_utf8",
            "inference_timeout_force_reap": True,
            "inference_stack_dump_interval_s": 30,
            "inference_phase_markers": [
                "request", "pre_sync", "pipeline", "test_step", "post_sync",
                "response",
            ],
        },
        "scene_contract": {
            "path": str(scene_list), "sha256": SCENE_SHA, "count": 100,
        },
        "processed_rgbd": {
            "root": str(processed),
            "runtime_source": True,
            "point_hash_must_match_parity_before_gpu": True,
        },
        "distribution_parity": {
            "path": str(parity), "sha256": PARITY_SHA,
            "lineage_scenes": 100, "point_array_scenes": 100,
            "point_byte_scenes": 100, "training_distribution_points": 24382287,
            "old_diagnostic_runtime_dependency": False,
            "training_point_runtime_dependency": False,
        },
        "ca_scratch_checkpoint_binding": {
            "path": str(binding.manifest_path),
            "sha256": binding.manifest_sha256,
            "checkpoint_sha256": binding.checkpoint_sha256,
            "effective_config_sha256": binding.effective_config_sha256,
            "scannet_trained_module_access": False,
        },
        "ca_point_inference_config": {
            **inference_contract,
            "checkpoint_binding_sha256": binding.manifest_sha256,
        },
        "protocol": {
            "frame_lineage_schema": "boxfusion.ca1m_demo_gap20_early_finalize_lineage.v1",
            "pixel_stride": 4,
            "voxel_size_m": 0.01,
            "min_depth_m": 0.1,
            "max_depth_m": 6.0,
            "score_threshold": 0.01,
            "max_proposals": 256,
        },
        "output": {
            "root": str(
                ROOT / "diagnostics/ca1m_tr3d_terminal_ca_native_train100_v4/proposals"
            ),
            "schema": "boxfusion.ca1m_tr3d_anchor_free_proposal_cache.v4",
            "create_only": True,
            "resume_policy": "validate_complete_then_skip_else_fail",
        },
        "code": code,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> int:
    target = _create_only(parser().parse_args().output, payload())
    print(json.dumps({"complete": True, "output": str(target), "sha256": sha256_file(target)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
