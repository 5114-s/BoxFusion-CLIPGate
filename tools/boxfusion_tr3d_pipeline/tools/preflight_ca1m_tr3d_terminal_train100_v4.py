#!/usr/bin/env python3
"""GT-free static preflight for the CA-1M terminal-TR3D v4 split route."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_checkpoint_binding import (  # noqa: E402
    FORBIDDEN_SCANNET_SHA256,
    SCHEMA as BINDING_SCHEMA,
    load_checkpoint_binding,
    regular_directory,
    regular_file,
)
from boxfusion.ca1m_tr3d_inference_contract import (  # noqa: E402
    SCHEMA as INFERENCE_CONFIG_SCHEMA,
    validate_ca1m_point_inference_config,
)
from boxfusion.ca1m_tr3d_overlay_binding_v4 import (  # noqa: E402
    STAGE_P_RUNTIME_CONFIG_SHA256,
    validate_overlay_authorization,
    validate_proposal_collection,
)
from boxfusion.ca1m_tr3d_terminal_v4 import (  # noqa: E402
    FRAME_LINEAGE_SCHEMA,
    OVERLAY_SCHEMA,
    PROPOSAL_SCHEMA,
    derive_demo_gap20_early_finalize_frame_ids,
    load_proposal_cache,
    sha256_file,
)


CONFIG_SCHEMA = "boxfusion.ca1m_tr3d_terminal_two_stage_config.v4"
REPORT_SCHEMA = "boxfusion.ca1m_tr3d_terminal_two_stage_preflight.v4"
NAMESPACE = "ca1m_tr3d_terminal_ca_native_train100_v4"
SCENE_SHA256 = "35321e9942dc5d512db2952b9ca6228b1291127e0c13fd92aa458f2d7eb7f9fd"
BINDING_SHA256 = "19b8c3d12de8dd8d3ffff1413c6c6003a5ccb1a10cf213b972ebd43fa9db5043"
CHECKPOINT_SHA256 = "d3ba6cc22f0a1a11ab47e55ccdd21c2ef4a84efaf3c6359b7e8231a6c8d3b4a7"
CONFIG_SHA256 = "38368fb5eb692ae2452d098bd4bb0814bbbb83feae780cf83446121bd9e7b88b"
INFERENCE_CONFIG_SHA256 = "60a0e626d671a8b0270006143a062de69ebdd3d9516d5d47c81a6cec2dcd5da4"
PARITY_RECEIPT_SHA256 = "35d9dfafc7272d92d98c97c6ef23f4323432e9bd0af5045bc5f78b1ae9afa00d"
AUTHORIZATION_RECEIPT_SHA256 = "42c1580b99a83e1f6c44ac27428596dfc5ae1f141635d63d07cc1c2e7f09ae25"
SUPERSEDED_AUTHORIZATION_V4_SHA256 = "de3063748ea757ae041b5a22112df47a1da46d71c809bc508914fc32032f7309"
SCENE = re.compile(r"^[0-9]{8}$")
SHA = re.compile(r"^[0-9a-f]{64}$")


def _keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{name} keys differ: missing={sorted(expected-set(value))}, "
            f"extra={sorted(set(value)-expected)}"
        )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _json(path: Path, name: str) -> tuple[Path, dict[str, Any]]:
    source = regular_file(path, name)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return source, value


def _scene_ids(contract: Mapping[str, Any]) -> tuple[str, ...]:
    _keys(contract, {"path", "sha256", "count", "exact"}, "scene_contract")
    path = regular_file(Path(str(contract["path"])), "exact train100 scene list")
    if (
        contract["sha256"] != SCENE_SHA256
        or sha256_file(path) != SCENE_SHA256
        or contract["count"] != 100
        or contract["exact"] is not True
    ):
        raise ValueError("train100 scene contract differs")
    scenes = tuple(row.strip() for row in path.read_text().splitlines() if row.strip())
    if len(scenes) != 100 or len(set(scenes)) != 100 or any(SCENE.fullmatch(x) is None for x in scenes):
        raise ValueError("scene list must contain exactly 100 unique numeric CA IDs")
    return scenes


def _numeric_names(path: Path, name: str) -> set[int]:
    root = regular_directory(path, name)
    result: set[int] = set()
    for item in root.iterdir():
        if item.is_symlink() or not item.is_file() or item.suffix.lower() != ".png":
            continue
        try:
            value = int(item.stem)
        except ValueError:
            continue
        if value < 0 or value in result:
            raise ValueError(f"invalid duplicate numeric frame: {root}")
        result.add(value)
    if result != set(range(len(result))) or not result:
        raise ValueError(f"{name} is not contiguous 0..N-1")
    return result


def _processed_inventory(root: Path, scenes: tuple[str, ...]) -> dict[str, Any]:
    data_root = regular_directory(root, "processed train100 RGB-D root")
    total_frames = 0
    total_keyframes = 0
    min_frames: int | None = None
    max_frames = 0
    for scene in scenes:
        scene_root = regular_directory(data_root / scene, f"processed scene {scene}")
        rgb = _numeric_names(scene_root / "rgb", f"RGB frames {scene}")
        depth = _numeric_names(scene_root / "depth", f"depth frames {scene}")
        if rgb != depth:
            raise ValueError(f"{scene}: RGB/depth numeric lineage differs")
        poses_path = regular_file(scene_root / "all_poses.npy", f"poses {scene}")
        poses = np.load(poses_path, mmap_mode="r", allow_pickle=False)
        if poses.shape != (len(rgb), 4, 4):
            raise ValueError(f"{scene}: pose/frame count differs")
        per_frame = scene_root / "K_depth_per_frame.npy"
        if per_frame.exists() or per_frame.is_symlink():
            intrinsics = np.load(
                regular_file(per_frame, f"per-frame intrinsics {scene}"),
                mmap_mode="r",
                allow_pickle=False,
            )
            if intrinsics.shape != (len(rgb), 3, 3):
                raise ValueError(f"{scene}: intrinsic/frame count differs")
        else:
            regular_file(scene_root / "K_depth.txt", f"fallback intrinsics {scene}")
        keyframes = derive_demo_gap20_early_finalize_frame_ids(len(rgb))
        if keyframes[0] != 0 or keyframes[-1] > max(0, len(rgb) - 21):
            raise ValueError(f"{scene}: reachable demo early-finalize lineage differs")
        total_frames += len(rgb)
        total_keyframes += len(keyframes)
        min_frames = len(rgb) if min_frames is None else min(min_frames, len(rgb))
        max_frames = max(max_frames, len(rgb))
    return {
        "root": str(data_root),
        "scene_count": len(scenes),
        "frame_count": total_frames,
        "demo_gap20_early_finalize_frame_count": total_keyframes,
        "min_scene_frames": int(min_frames or 0),
        "max_scene_frames": max_frames,
        "files_opened": ["all_poses.npy", "K_depth_per_frame.npy_or_K_depth.txt"],
        "rgb_depth_image_bytes_opened": False,
        "annotation_files_opened": False,
    }


def _proposal_inventory(
    root: Path, scenes: tuple[str, ...], binding_sha: str
) -> dict[str, Any]:
    if root.is_symlink():
        raise ValueError("proposal cache root must not be a symlink")
    if not root.exists():
        return {"exists": False, "valid_count": 0, "complete": False, "missing_count": len(scenes)}
    directory = regular_directory(root, "v4 proposal cache root")
    expected = {f"{scene}_ca1m_tr3d_proposals_v4.npz": scene for scene in scenes}
    actual = {
        path.name
        for path in directory.iterdir()
        if path.is_file() and not path.is_symlink() and path.suffix == ".npz"
    }
    extra = actual - set(expected)
    if extra:
        raise ValueError(f"unexpected proposal-cache files: {sorted(extra)[:5]}")
    valid = 0
    for name in sorted(actual):
        load_proposal_cache(
            directory / name,
            expected_scene=expected[name],
            expected_binding_sha256=binding_sha,
        )
        valid += 1
    return {
        "exists": True,
        "valid_count": valid,
        "complete": valid == len(scenes),
        "missing_count": len(scenes) - valid,
    }


def _overlay_inventory(root: Path, scenes: tuple[str, ...]) -> dict[str, Any]:
    if root.is_symlink():
        raise ValueError("overlay root must not be a symlink")
    if not root.exists():
        return {"exists": False, "count": 0, "complete": False, "missing_count": len(scenes)}
    directory = regular_directory(root, "v4 overlay root")
    expected = {f"{scene}_ca1m_tr3d_overlay_v4.npz" for scene in scenes}
    actual = {
        path.name
        for path in directory.iterdir()
        if path.is_file() and not path.is_symlink() and path.suffix == ".npz"
    }
    if actual - expected:
        raise ValueError(f"unexpected overlay files: {sorted(actual-expected)[:5]}")
    return {
        "exists": True,
        "count": len(actual),
        "complete": actual == expected,
        "missing_count": len(expected - actual),
    }


def validate_config(config_path: Path) -> dict[str, Any]:
    source, cfg = _json(config_path, "terminal-v4 two-stage config")
    base_keys = {
        "schema", "namespace", "observer_only", "mutation_enabled",
        "ground_truth_access", "validation_ground_truth_access",
        "full_run_authorized", "runner_state", "scene_contract",
        "processed_rgbd", "frame_lineage", "distribution_parity",
        "proposal_stage", "overlay_stage",
        "ca_native_tr3d_binding", "ca_native_tr3d_inference",
        "tr3d_runtime", "protocol", "forbidden_reuse",
    }
    if set(cfg) not in (base_keys, base_keys | {"stage_o_binding"}):
        raise ValueError("terminal-v4 config keys differ")
    if (
        cfg["schema"] != CONFIG_SCHEMA
        or cfg["namespace"] != NAMESPACE
        or cfg["observer_only"] is not True
        or cfg["mutation_enabled"] is not False
        or cfg["ground_truth_access"] is not False
        or cfg["validation_ground_truth_access"] is not False
        or not isinstance(cfg["full_run_authorized"], bool)
    ):
        raise ValueError("terminal-v4 top-level isolation contract differs")
    scenes = _scene_ids(_mapping(cfg["scene_contract"], "scene_contract"))
    processed = _mapping(cfg["processed_rgbd"], "processed_rgbd")
    _keys(
        processed,
        {
            "root", "rgb_directory", "depth_directory", "poses_file",
            "per_frame_intrinsics_file", "fallback_intrinsics_file", "depth_scale",
            "allowed_inputs_only",
        },
        "processed_rgbd",
    )
    if (
        processed["rgb_directory"] != "rgb"
        or processed["depth_directory"] != "depth"
        or processed["poses_file"] != "all_poses.npy"
        or processed["per_frame_intrinsics_file"] != "K_depth_per_frame.npy"
        or processed["fallback_intrinsics_file"] != "K_depth.txt"
        or float(processed["depth_scale"]) != 1000.0
        or processed["allowed_inputs_only"]
        != [
            "rgb/*.png", "depth/*.png", "all_poses.npy",
            "K_depth_per_frame.npy", "K_depth.txt",
        ]
    ):
        raise ValueError("processed RGB-D allowed-input contract differs")
    frame = _mapping(cfg["frame_lineage"], "frame_lineage")
    _keys(
        frame,
        {
            "schema", "source", "start", "gap", "early_finalize_condition",
            "include_last",
            "anchor_manifest_required", "b6_diagnostic_required",
        },
        "frame_lineage",
    )
    if frame != {
        "schema": FRAME_LINEAGE_SCHEMA,
        "source": "direct_processed_train100_rgb_depth_pose_demo_loop_simulation",
        "start": 0,
        "gap": 20,
        "early_finalize_condition": "count_eq_N_minus_1_or_count_plus_gap_gt_N_minus_1_after_increment",
        "include_last": False,
        "anchor_manifest_required": False,
        "b6_diagnostic_required": False,
    }:
        raise ValueError("frame lineage is not the reachable demo early-finalize sequence")
    parity_cfg = _mapping(cfg["distribution_parity"], "distribution_parity")
    _keys(
        parity_cfg,
        {
            "schema", "receipt", "receipt_sha256", "scene_count",
            "lineage_parity_scene_count", "point_array_parity_scene_count",
            "point_byte_parity_scene_count",
            "old_native_b6_diagnostic_runtime_dependency",
            "training_point_runtime_dependency", "ground_truth_access", "status",
        },
        "distribution_parity",
    )
    parity_path, parity = _json(
        Path(str(parity_cfg["receipt"])), "v4 lineage/training-point parity receipt"
    )
    if parity_path.stat().st_mode & 0o222:
        raise ValueError("v4 parity receipt must be read-only")
    if (
        parity_cfg["schema"]
        != "boxfusion.ca1m_tr3d_v4_lineage_training_point_parity.v1"
        or parity_cfg["receipt_sha256"] != sha256_file(parity_path)
        or parity_cfg["receipt_sha256"] != PARITY_RECEIPT_SHA256
        or parity_cfg["scene_count"] != 100
        or parity_cfg["lineage_parity_scene_count"] != 100
        or parity_cfg["point_array_parity_scene_count"] != 100
        or parity_cfg["point_byte_parity_scene_count"] != 100
        or parity_cfg["old_native_b6_diagnostic_runtime_dependency"] is not False
        or parity_cfg["training_point_runtime_dependency"] is not False
        or parity_cfg["ground_truth_access"] is not False
        or parity_cfg["status"] != "exact100_pass"
    ):
        raise ValueError("v4 distribution-parity binding differs")
    for name, expected in {
        "schema": "boxfusion.ca1m_tr3d_v4_lineage_training_point_parity.v1",
        "complete": True,
        "ground_truth_access": False,
        "validation_ground_truth_access": False,
        "prediction_access": False,
        "evaluator_access": False,
        "gpu_started": False,
        "model_started": False,
        "scene_count": 100,
        "lineage_parity_scene_count": 100,
        "point_array_parity_scene_count": 100,
        "point_byte_parity_scene_count": 100,
    }.items():
        if parity.get(name) != expected:
            raise ValueError(f"v4 parity receipt field {name} differs")
    if parity.get("counts") != {
        "processed_frames": 61189,
        "reachable_demo_keyframes": 3010,
        "training_distribution_points": 24382287,
    }:
        raise ValueError("v4 parity receipt aggregate counts differ")
    lineage_receipt = parity.get("lineage_contract") or {}
    point_receipt = parity.get("point_contract") or {}
    if (
        lineage_receipt.get("schema") != FRAME_LINEAGE_SCHEMA
        or lineage_receipt.get("include_last") is not False
        or lineage_receipt.get("proposal_runtime_dependency") is not False
        or lineage_receipt.get("sealed_native_b6_diagnostic_role")
        != "one_time_protocol_oracle_only"
        or point_receipt.get("resize_or_orientation_transform")
        != "none_in_training_converter_or_v4_builder"
        or "equal CA-native training .bin arrays exactly"
        not in str(point_receipt.get("proof", ""))
        or any((parity.get("forbidden_files_opened") or {}).values())
    ):
        raise ValueError("v4 lineage/point proof contract differs")
    implementation = parity.get("implementation") or {}
    for name in (
        "audit", "proposal_runner", "v4_contract", "backprojection",
        "terminal_geometry", "training_converter",
    ):
        record = implementation.get(name) or {}
        implementation_path = regular_file(
            Path(str(record.get("path", ""))), f"parity implementation {name}"
        )
        if record.get("sha256") != sha256_file(implementation_path):
            raise ValueError(f"parity implementation {name} changed after exact100 audit")
    binding_cfg = _mapping(cfg["ca_native_tr3d_binding"], "ca_native_tr3d_binding")
    _keys(
        binding_cfg,
        {
            "schema", "path", "sha256", "checkpoint_sha256",
            "effective_config_sha256", "initialization",
            "raw_checkpoint_argument_allowed", "raw_config_argument_allowed",
            "scannet_checkpoint_or_config_allowed",
        },
        "ca_native_tr3d_binding",
    )
    binding_path = regular_file(Path(str(binding_cfg["path"])), "sealed CA TR3D binding")
    if (
        binding_cfg["schema"] != BINDING_SCHEMA
        or binding_cfg["sha256"] != BINDING_SHA256
        or sha256_file(binding_path) != BINDING_SHA256
        or binding_cfg["checkpoint_sha256"] != CHECKPOINT_SHA256
        or binding_cfg["effective_config_sha256"] != CONFIG_SHA256
        or binding_cfg["initialization"] != "ca1m_random_scratch"
        or binding_cfg["raw_checkpoint_argument_allowed"] is not False
        or binding_cfg["raw_config_argument_allowed"] is not False
        or binding_cfg["scannet_checkpoint_or_config_allowed"] is not False
    ):
        raise ValueError("CA-only checkpoint binding contract differs")
    binding = load_checkpoint_binding(binding_path)
    if (
        binding.manifest_sha256 != BINDING_SHA256
        or binding.checkpoint_sha256 != CHECKPOINT_SHA256
        or binding.effective_config_sha256 != CONFIG_SHA256
        or binding.checkpoint_sha256 in FORBIDDEN_SCANNET_SHA256
        or binding.effective_config_sha256 in FORBIDDEN_SCANNET_SHA256
    ):
        raise ValueError("sealed binding resolves a forbidden/non-CA model")
    inference_cfg = _mapping(
        cfg["ca_native_tr3d_inference"], "ca_native_tr3d_inference"
    )
    _keys(
        inference_cfg,
        {
            "schema", "path", "sha256", "checkpoint_binding_sha256",
            "effective_training_config_sha256", "standalone",
            "point_input_only", "dataset_shell_lazy_only",
            "ground_truth_access", "validation_access", "evaluator_access",
            "scannet_config_access",
        },
        "ca_native_tr3d_inference",
    )
    if (
        inference_cfg["schema"] != INFERENCE_CONFIG_SCHEMA
        or inference_cfg["sha256"] != INFERENCE_CONFIG_SHA256
        or inference_cfg["checkpoint_binding_sha256"] != BINDING_SHA256
        or inference_cfg["effective_training_config_sha256"] != CONFIG_SHA256
        or inference_cfg["standalone"] is not True
        or inference_cfg["point_input_only"] is not True
        or inference_cfg["dataset_shell_lazy_only"] is not True
        or inference_cfg["ground_truth_access"] is not False
        or inference_cfg["validation_access"] is not False
        or inference_cfg["evaluator_access"] is not False
        or inference_cfg["scannet_config_access"] is not False
    ):
        raise ValueError("CA-only point-inference binding contract differs")
    inference_contract = validate_ca1m_point_inference_config(
        inference_path=Path(str(inference_cfg["path"])),
        inference_sha256=INFERENCE_CONFIG_SHA256,
        effective_training_path=binding.effective_config_path,
        effective_training_sha256=binding.effective_config_sha256,
    )
    proposal = _mapping(cfg["proposal_stage"], "proposal_stage")
    _keys(
        proposal,
        {
            "stage_id", "schema", "status", "run_authorized", "gpu_required",
            "authorization_schema", "authorization_receipt",
            "authorization_receipt_sha256", "authorization_scope",
            "anchor_access", "b6_access", "ground_truth_access", "create_only",
            "resume_policy", "output_root", "exact_output_count",
        },
        "proposal_stage",
    )
    if (
        proposal["stage_id"] != "P"
        or proposal["schema"] != PROPOSAL_SCHEMA
        or proposal["status"] != "authorized_by_sealed_stage_p_receipt"
        or proposal["run_authorized"] is not True
        or proposal["authorization_schema"]
        != "boxfusion.ca1m_tr3d_v4_proposal_authorization.v1"
        or proposal["authorization_scope"] != "stage_p_only"
        or proposal["gpu_required"] is not True
        or proposal["anchor_access"] is not False
        or proposal["b6_access"] is not False
        or proposal["ground_truth_access"] is not False
        or proposal["create_only"] is not True
        or proposal["resume_policy"] != "validate_complete_then_skip_else_fail"
        or proposal["exact_output_count"] != 100
    ):
        raise ValueError("proposal stage isolation/resume contract differs")
    authorization_path, authorization = _json(
        Path(str(proposal["authorization_receipt"])),
        "terminal-v4 proposal authorization receipt",
    )
    authorization_sha = sha256_file(authorization_path)
    if authorization_path.stat().st_mode & 0o222:
        raise ValueError("proposal authorization receipt must be read-only")
    if (
        proposal["authorization_receipt_sha256"] != authorization_sha
        or authorization_sha != AUTHORIZATION_RECEIPT_SHA256
    ):
        raise ValueError("proposal authorization receipt SHA256 differs")
    for name, expected in {
        "schema": "boxfusion.ca1m_tr3d_v4_proposal_authorization.v1",
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
    }.items():
        if authorization.get(name) != expected:
            raise ValueError(f"proposal authorization field {name} differs")
    if authorization.get("authorization_revision") != 5:
        raise ValueError("proposal authorization is not binary-protocol revision 5")
    supersedes = _mapping(authorization.get("supersedes"), "authorization supersedes")
    superseded_path = regular_file(
        Path(str(supersedes.get("authorization_path", ""))),
        "superseded proposal authorization revision 4",
    )
    if (
        supersedes.get("authorization_sha256")
        != SUPERSEDED_AUTHORIZATION_V4_SHA256
        or sha256_file(superseded_path) != SUPERSEDED_AUTHORIZATION_V4_SHA256
        or supersedes.get("old_receipt_overwritten") is not False
        or supersedes.get("old_proposal_artifact_count") != 0
    ):
        raise ValueError("proposal authorization revision-4 supersession differs")
    required_runtime = _mapping(
        authorization.get("required_runtime_config"),
        "authorized Stage-P runtime config",
    )
    authorized_runtime_source = source
    if "stage_o_binding" in cfg:
        stage_o_binding = _mapping(cfg["stage_o_binding"], "Stage-O binding")
        _, proposal_collection, _ = validate_proposal_collection(
            stage_o_binding.get("proposal_collection"), cfg
        )
        proposal_runtime = _mapping(
            proposal_collection.get("runtime_config"),
            "sealed proposal-collection runtime config",
        )
        authorized_runtime_source = regular_file(
            Path(str(proposal_runtime.get("path", ""))),
            "sealed Stage-P runtime config",
        )
        if (
            proposal_runtime.get("sha256") != STAGE_P_RUNTIME_CONFIG_SHA256
            or sha256_file(authorized_runtime_source)
            != STAGE_P_RUNTIME_CONFIG_SHA256
            or (proposal_collection.get("stage_p_authorization") or {}).get("sha256")
            != authorization_sha
        ):
            raise ValueError(
                "Stage-O config lost its sealed Stage-P runtime/authorization binding"
            )
    if (
        Path(str(required_runtime.get("path", ""))).resolve()
        != authorized_runtime_source.resolve()
        or required_runtime.get("startup_timeout_s") != 600
        or required_runtime.get("sha256_omitted_to_avoid_authorization_hash_cycle")
        is not True
    ):
        raise ValueError("authorized Stage-P runtime config differs")
    runtime_policy = _mapping(
        authorization.get("runtime_policy"), "authorized Stage-P runtime policy"
    )
    if runtime_policy != {
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
    }:
        raise ValueError("authorized Stage-P bounded startup policy differs")
    authorization_scene = authorization.get("scene_contract") or {}
    authorization_parity = authorization.get("distribution_parity") or {}
    authorization_binding = authorization.get("ca_scratch_checkpoint_binding") or {}
    authorization_inference = authorization.get("ca_point_inference_config") or {}
    authorization_processed = authorization.get("processed_rgbd") or {}
    authorization_output = authorization.get("output") or {}
    if (
        authorization_scene.get("sha256") != SCENE_SHA256
        or authorization_scene.get("count") != 100
        or authorization_parity.get("sha256") != sha256_file(parity_path)
        or authorization_parity.get("lineage_scenes") != 100
        or authorization_parity.get("point_array_scenes") != 100
        or authorization_parity.get("point_byte_scenes") != 100
        or authorization_parity.get("old_diagnostic_runtime_dependency") is not False
        or authorization_parity.get("training_point_runtime_dependency") is not False
        or authorization_binding.get("sha256") != BINDING_SHA256
        or authorization_binding.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or authorization_binding.get("effective_config_sha256") != CONFIG_SHA256
        or authorization_binding.get("scannet_trained_module_access") is not False
        or authorization_inference.get("sha256") != INFERENCE_CONFIG_SHA256
        or authorization_inference.get("checkpoint_binding_sha256") != BINDING_SHA256
        or authorization_inference.get("point_input_only") is not True
        or authorization_inference.get("dataset_shell_lazy_only") is not True
        or authorization_inference.get("ground_truth_access") is not False
        or authorization_inference.get("validation_access") is not False
        or authorization_inference.get("evaluator_access") is not False
        or authorization_inference.get("scannet_config_access") is not False
        or Path(str(authorization_processed.get("root", ""))).resolve()
        != Path(str(processed["root"])).resolve()
        or authorization_processed.get("point_hash_must_match_parity_before_gpu")
        is not True
        or Path(str(authorization_output.get("root", ""))).resolve()
        != Path(str(proposal["output_root"])).resolve()
        or authorization_output.get("schema") != PROPOSAL_SCHEMA
        or authorization_output.get("create_only") is not True
        or authorization_output.get("resume_policy")
        != "validate_complete_then_skip_else_fail"
    ):
        raise ValueError("proposal authorization prerequisite binding differs")
    authorization_code = authorization.get("code") or {}
    for name in (
        "proposal_runner", "proposal_contract", "checkpoint_binding_loader",
        "backprojection", "terminal_geometry", "worker_client", "worker",
        "official_adapter", "ca_point_inference_contract",
        "ca_point_inference_config",
    ):
        record = authorization_code.get(name) or {}
        code_path = regular_file(
            Path(str(record.get("path", ""))), f"authorized stage-P code {name}"
        )
        if record.get("sha256") != sha256_file(code_path):
            raise ValueError(f"authorized stage-P code {name} changed")
    overlay = _mapping(cfg["overlay_stage"], "overlay_stage")
    _keys(
        overlay,
        {
            "stage_id", "schema", "status", "run_authorized", "gpu_required",
            "cpu_only", "ground_truth_access", "proposal_cache_root",
            "final_anchor_root", "final_anchor_manifest",
            "final_anchor_manifest_sha256", "native_b6_v2_diagnostics_root",
            "native_b6_v2_collection_manifest",
            "native_b6_v2_collection_manifest_sha256",
            "native_b6_v2_completion_root", "native_b6_v2_checkpoint",
            "native_b6_v2_checkpoint_sha256", "native_b6_v2_checkpoint_manifest",
            "native_b6_v2_checkpoint_manifest_sha256", "output_root", "create_only",
            "resume_policy", "exact_output_count",
        },
        "overlay_stage",
    )
    pending_names = (
        "final_anchor_root", "final_anchor_manifest", "final_anchor_manifest_sha256",
        "native_b6_v2_diagnostics_root", "native_b6_v2_collection_manifest",
        "native_b6_v2_collection_manifest_sha256", "native_b6_v2_completion_root",
        "native_b6_v2_checkpoint", "native_b6_v2_checkpoint_sha256",
        "native_b6_v2_checkpoint_manifest",
        "native_b6_v2_checkpoint_manifest_sha256",
    )
    pending_values = [overlay[name] for name in pending_names]
    all_pending = all(value is None for value in pending_values)
    all_bound = all(value is not None for value in pending_values)
    if not (all_pending or all_bound):
        raise ValueError("overlay final-anchor/B6-v2 binding is partial")
    if (
        overlay["stage_id"] != "O"
        or overlay["schema"] != OVERLAY_SCHEMA
        or not isinstance(overlay["run_authorized"], bool)
        or overlay["gpu_required"] is not False
        or overlay["cpu_only"] is not True
        or overlay["ground_truth_access"] is not False
        or Path(str(overlay["proposal_cache_root"])).resolve()
        != Path(str(proposal["output_root"])).resolve()
        or overlay["create_only"] is not True
        or overlay["resume_policy"] != "validate_complete_then_skip_else_fail"
        or overlay["exact_output_count"] != 100
    ):
        raise ValueError("overlay stage isolation/resume contract differs")
    if overlay["run_authorized"] is True and not all_bound:
        raise ValueError("overlay cannot be authorized while final anchor/B6 v2 are pending")
    stage_o_authorization = None
    if overlay["run_authorized"] is True:
        if "stage_o_binding" not in cfg:
            raise ValueError("authorized overlay lacks independent Stage-O binding")
        stage_o_authorization = validate_overlay_authorization(source, cfg)
    elif "stage_o_binding" in cfg:
        raise ValueError("pending overlay must not carry an active Stage-O binding")
    runtime = _mapping(cfg["tr3d_runtime"], "tr3d_runtime")
    _keys(
        runtime,
        {
            "worker_python", "worker_script", "runtime_root", "project_root",
            "vendor_root", "startup_timeout_s",
        },
        "tr3d_runtime",
    )
    if runtime["startup_timeout_s"] != 600:
        raise ValueError("formal Stage-P startup timeout must equal 600 seconds")
    for name in ("worker_script",):
        regular_file(Path(str(runtime[name])), name)
    for name in ("runtime_root", "project_root", "vendor_root"):
        regular_directory(Path(str(runtime[name])), name)
    worker_python = Path(str(runtime["worker_python"])).resolve()
    if not worker_python.is_file() or not os.access(worker_python, os.X_OK):
        raise FileNotFoundError(f"missing worker Python: {worker_python}")
    protocol = _mapping(cfg["protocol"], "protocol")
    _keys(
        protocol,
        {
            "prefix_id", "pixel_stride", "voxel_size_m", "min_depth_m",
            "max_depth_m", "near_iou", "score_threshold", "max_proposals",
        },
        "protocol",
    )
    if dict(protocol) != {
        "prefix_id": "p100_gap20",
        "pixel_stride": 4,
        "voxel_size_m": 0.01,
        "min_depth_m": 0.1,
        "max_depth_m": 6.0,
        "near_iou": 0.15,
        "score_threshold": 0.01,
        "max_proposals": 256,
    }:
        raise ValueError("terminal-v4 protocol differs")
    forbidden = _mapping(cfg["forbidden_reuse"], "forbidden_reuse")
    _keys(
        forbidden,
        {
            "scannet_tr3d_checkpoint_sha256", "old_terminal_namespaces",
            "old_native_b6_namespaces", "old_terminal_artifact_access",
            "old_native_b6_artifact_access", "exception",
        },
        "forbidden_reuse",
    )
    if (
        set(forbidden["scannet_tr3d_checkpoint_sha256"])
        != {
            "09f2f650540716556719d2858d9a484dcbf682e2e94576887855b9b637b6492e",
            "a484fd79093aa3004f4d2984e7ad8763c5d5cbec7edd04172d775643a8436448",
        }
        or forbidden["old_terminal_artifact_access"] is not False
        or forbidden["old_native_b6_artifact_access"] is not False
        or "v3 CA-only checkpoint binding" not in str(forbidden["exception"])
    ):
        raise ValueError("forbidden-reuse contract differs")
    inventory = _processed_inventory(Path(str(processed["root"])), scenes)
    proposal_inventory = _proposal_inventory(
        Path(str(proposal["output_root"])), scenes, binding.manifest_sha256
    )
    overlay_inventory = _overlay_inventory(Path(str(overlay["output_root"])), scenes)
    proposal_authorized = (
        proposal["run_authorized"] is True
        and authorization.get("proposal_gpu_execution_authorized") is True
    )
    overlay_authorized = overlay["run_authorized"] is True and all_bound
    full_authorized = (
        cfg["full_run_authorized"] is True
        and proposal_authorized
        and overlay_authorized
    )
    blocked: list[str] = []
    if not proposal_authorized:
        blocked.append("explicit_proposal_gpu_authorization_pending")
    if not all_bound:
        blocked.extend(
            [
                "final_base_train100_anchor_manifest_pending",
                "final_base_native_b6_v2_artifacts_pending",
            ]
        )
    if not overlay_authorized:
        blocked.append("cpu_overlay_authorization_pending")
    if not full_authorized:
        blocked.append("full_two_stage_run_not_authorized")
    return {
        "schema": REPORT_SCHEMA,
        "complete": True,
        "config_path": str(source),
        "config_sha256": sha256_file(source),
        "namespace": NAMESPACE,
        "scene_count": len(scenes),
        "scene_list_sha256": SCENE_SHA256,
        "ground_truth_access": False,
        "validation_ground_truth_access": False,
        "gpu_started": False,
        "proposal_stage_contract_ready": True,
        "proposal_stage_runtime_authorized": proposal_authorized,
        "proposal_authorization": {
            "path": str(authorization_path),
            "sha256": authorization_sha,
            "decision": authorization["authorization_decision"],
            "scope": "stage_p_only",
            "point_hash_precheck_before_gpu": True,
        },
        "ca_point_inference_config": inference_contract,
        "proposal_cache_depends_on_anchor_or_b6": False,
        "proposal_stage_anchor_access": False,
        "proposal_stage_b6_access": False,
        "distribution_parity": {
            "receipt_path": str(parity_path),
            "receipt_sha256": sha256_file(parity_path),
            "lineage_scenes": parity["lineage_parity_scene_count"],
            "point_array_scenes": parity["point_array_parity_scene_count"],
            "point_byte_scenes": parity["point_byte_parity_scene_count"],
            "training_distribution_points": parity["counts"][
                "training_distribution_points"
            ],
            "old_diagnostic_runtime_dependency": False,
            "training_point_runtime_dependency": False,
        },
        "overlay_stage_contract_ready": True,
        "overlay_inputs_bound": all_bound,
        "overlay_stage_runtime_authorized": overlay_authorized,
        "stage_o_authorization": stage_o_authorization,
        "overlay_cpu_only": True,
        "full_run_authorized": full_authorized,
        "blocked_reasons": blocked,
        "processed_rgbd_inventory": inventory,
        "proposal_inventory": proposal_inventory,
        "overlay_inventory": overlay_inventory,
        "checkpoint_binding": {
            "path": str(binding.manifest_path),
            "sha256": binding.manifest_sha256,
            "checkpoint_sha256": binding.checkpoint_sha256,
            "effective_config_sha256": binding.effective_config_sha256,
            "ca1m_random_scratch": True,
            "scannet_module_access": False,
        },
        "tr3d_runtime_policy": {
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
        },
        "forbidden_reuse": {
            "old_terminal_v1_v2_v3_artifacts": True,
            "old_native_b6_or_scannet_b6": True,
            "sealed_v3_binding_only_exception": True,
        },
    }


def write_json_create_only(path: Path, value: Mapping[str, Any]) -> Path:
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
            raise FileExistsError(f"refusing to overwrite v4 preflight report: {target}") from error
        target.chmod(0o444)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/ca1m_tr3d_terminal_train100_v4_p5.json",
    )
    value.add_argument("--require-run", action="store_true")
    value.add_argument("--require-proposal-run", action="store_true")
    value.add_argument("--output", type=Path)
    return value


def main() -> int:
    args = parser().parse_args()
    report = validate_config(args.config)
    if args.output is not None:
        write_json_create_only(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_run and not report["full_run_authorized"]:
        print("terminal-v4 --run is fail-closed: " + ", ".join(report["blocked_reasons"]), file=sys.stderr)
        return 2
    if args.require_proposal_run and not report["proposal_stage_runtime_authorized"]:
        print("terminal-v4 --run-proposals is not authorized", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
