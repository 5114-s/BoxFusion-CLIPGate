#!/usr/bin/env python3
"""Static/runtime preflight for isolated CA-native TR3D exact100 collection.

Static preflight intentionally succeeds before training completes, but reports
``ready_for_gpu=false``.  Runtime preflight additionally requires the sealed
checkpoint binding and never accepts a raw checkpoint or TR3D config path.
No ground-truth file is opened by this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_checkpoint_binding import (  # noqa: E402
    EXPECTED_SOURCE_CONFIG,
    EXPECTED_SOURCE_CONFIG_SHA256,
    EXPECTED_WORK_ROOT,
    SCHEMA as BINDING_SCHEMA,
    load_checkpoint_binding,
    regular_directory,
    regular_file,
    sha256_file,
)


CONFIG_SCHEMA = "boxfusion.ca1m_tr3d_terminal_collection_config.v3"
REPORT_SCHEMA = "boxfusion.ca1m_tr3d_terminal_collection_preflight.v3"
EXPECTED_NAMESPACE = "ca1m_tr3d_terminal_ca_native_train100_v3"
EXPECTED_SCENE_SHA256 = (
    "35321e9942dc5d512db2952b9ca6228b1291127e0c13fd92aa458f2d7eb7f9fd"
)
SCENE_RE = re.compile(r"^[0-9]{8}$")


def _expect_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{name} keys differ: missing={sorted(expected-set(value))}, "
            f"extra={sorted(set(value)-expected)}"
        )


def _json(path: Path, name: str) -> tuple[Path, dict[str, Any]]:
    source = regular_file(path, name)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return source, payload


def _scene_ids(path: Path, expected_sha: str, expected_count: int) -> tuple[str, ...]:
    source = regular_file(path, "exact100 scene list")
    if sha256_file(source) != expected_sha or expected_sha != EXPECTED_SCENE_SHA256:
        raise ValueError("exact100 scene-list SHA256 differs")
    scenes = tuple(
        row.strip()
        for row in source.read_text(encoding="utf-8").splitlines()
        if row.strip()
    )
    if (
        len(scenes) != expected_count
        or expected_count != 100
        or len(set(scenes)) != len(scenes)
        or any(SCENE_RE.fullmatch(scene) is None for scene in scenes)
    ):
        raise ValueError("scene list must contain exactly 100 unique CA-1M IDs")
    return scenes


def _check_hash_record(value: Mapping[str, Any], name: str) -> Path:
    _expect_keys(value, {"path", "sha256"}, name)
    source = regular_file(Path(str(value["path"])), name)
    if sha256_file(source) != value["sha256"]:
        raise ValueError(f"{name} SHA256 mismatch")
    return source


def _executable(path: Path, name: str) -> Path:
    """Allow only the environment's conventional python -> pythonX link."""

    value = path.resolve()
    if not value.is_file() or not os.access(value, os.X_OK):
        raise FileNotFoundError(f"missing executable {name}: {value}")
    return value


def _artifact_inventory(root: Path, suffix: str, scenes: tuple[str, ...]) -> dict[str, Any]:
    if root.is_symlink():
        raise ValueError(f"output root must not be a symlink: {root}")
    if not root.exists():
        return {"exists": False, "count": 0, "complete": False}
    directory = regular_directory(root, "v3 output root")
    expected = {f"{scene}{suffix}" for scene in scenes}
    actual = {
        path.name
        for path in directory.iterdir()
        if path.is_file() and not path.is_symlink() and path.name.endswith(suffix)
    }
    unexpected = sorted(actual - expected)
    if unexpected:
        raise ValueError(f"v3 output root contains unexpected artifacts: {unexpected[:5]}")
    for name in actual:
        path = directory / name
        if path.stat().st_size <= 0:
            raise ValueError(f"v3 output artifact is empty: {path}")
    return {
        "exists": True,
        "count": len(actual),
        "complete": actual == expected,
        "missing_count": len(expected - actual),
    }


def validate_config(config_path: Path, binding_path: Path | None) -> dict[str, Any]:
    source, cfg = _json(config_path, "v3 collection config")
    _expect_keys(
        cfg,
        {
            "schema", "namespace", "observer_only", "mutation_enabled",
            "ground_truth_access", "validation_ground_truth_access",
            "run_authorized", "runner_state", "scene_contract", "inputs",
            "proposal_stage", "anchor_overlay_stage", "tr3d",
            "terminal_protocol", "outputs", "forbidden_reuse",
        },
        "v3 collection config",
    )
    if (
        cfg["schema"] != CONFIG_SCHEMA
        or cfg["namespace"] != EXPECTED_NAMESPACE
        or cfg["observer_only"] is not True
        or cfg["mutation_enabled"] is not False
        or cfg["ground_truth_access"] is not False
        or cfg["validation_ground_truth_access"] is not False
        or cfg["run_authorized"] is not False
        or cfg["runner_state"] != "static_contract_only_pending_final_route"
    ):
        raise ValueError("v3 config observer/GT isolation contract differs")
    scene_cfg = cfg["scene_contract"]
    inputs = cfg["inputs"]
    proposal_stage = cfg["proposal_stage"]
    overlay_stage = cfg["anchor_overlay_stage"]
    tr3d = cfg["tr3d"]
    protocol = cfg["terminal_protocol"]
    outputs = cfg["outputs"]
    forbidden = cfg["forbidden_reuse"]
    for value, name in (
        (scene_cfg, "scene_contract"), (inputs, "inputs"),
        (proposal_stage, "proposal_stage"),
        (overlay_stage, "anchor_overlay_stage"), (tr3d, "tr3d"),
        (protocol, "terminal_protocol"), (outputs, "outputs"),
        (forbidden, "forbidden_reuse"),
    ):
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a JSON object")
    _expect_keys(scene_cfg, {"path", "sha256", "count", "exact"}, "scene_contract")
    if scene_cfg["exact"] is not True:
        raise ValueError("v3 collection must use an exact scene contract")
    scenes = _scene_ids(
        Path(str(scene_cfg["path"])),
        str(scene_cfg["sha256"]),
        int(scene_cfg["count"]),
    )
    _expect_keys(
        inputs,
        {"processed_ca1m_root", "upstream_contract"},
        "inputs",
    )
    data_root = regular_directory(
        Path(str(inputs["processed_ca1m_root"])), "processed CA-1M root"
    )
    if (
        "no prior terminal candidate cache" not in str(inputs["upstream_contract"])
        or "old anchor/B6 artifact" not in str(inputs["upstream_contract"])
    ):
        raise ValueError("upstream contract does not forbid old route reuse")
    for scene in scenes:
        regular_directory(data_root / scene, f"processed scene {scene}")
    _expect_keys(
        proposal_stage,
        {
            "schema", "anchor_access", "b6_access", "frame_lineage_manifest",
            "frame_lineage_manifest_sha256", "status",
        },
        "proposal_stage",
    )
    if (
        proposal_stage["schema"] != "boxfusion.ca1m_tr3d_proposal_cache.v1"
        or proposal_stage["anchor_access"] is not False
        or proposal_stage["b6_access"] is not False
        or proposal_stage["frame_lineage_manifest"] is not None
        or proposal_stage["frame_lineage_manifest_sha256"] is not None
        or proposal_stage["status"] != "pending_final_route_frame_lineage"
    ):
        raise ValueError("proposal stage is not fail-closed on final frame lineage")
    _expect_keys(
        overlay_stage,
        {
            "output_schema", "gpu_required", "final_anchor_root",
            "final_anchor_manifest", "final_anchor_manifest_sha256",
            "retrained_native_b6_diagnostics_root",
            "retrained_native_b6_checkpoint", "retrained_native_b6_manifest",
            "status",
        },
        "anchor_overlay_stage",
    )
    if (
        overlay_stage["output_schema"]
        != "boxfusion.ca1m_tr3d_terminal_observer.v1"
        or overlay_stage["gpu_required"] is not False
        or any(
            overlay_stage[name] is not None
            for name in (
                "final_anchor_root", "final_anchor_manifest",
                "final_anchor_manifest_sha256",
                "retrained_native_b6_diagnostics_root",
                "retrained_native_b6_checkpoint", "retrained_native_b6_manifest",
            )
        )
        or overlay_stage["status"]
        != "pending_g0_clip_reliable_topk3_and_retrained_b6"
    ):
        raise ValueError("anchor overlay is not fail-closed on final route inputs")
    _expect_keys(
        tr3d,
        {
            "binding_schema", "binding_environment_variable", "fixed_work_root",
            "source_config_path", "source_config_sha256", "worker_python",
            "worker_script", "runtime_root", "project_root", "vendor_root",
            "raw_checkpoint_argument_allowed", "raw_config_argument_allowed",
            "scannet_checkpoint_or_config_allowed",
        },
        "tr3d",
    )
    if (
        tr3d["binding_schema"] != BINDING_SCHEMA
        or tr3d["binding_environment_variable"]
        != "BOXFUSION_CA1M_TR3D_V3_CHECKPOINT_BINDING"
        or Path(str(tr3d["fixed_work_root"])).resolve() != EXPECTED_WORK_ROOT.resolve()
        or Path(str(tr3d["source_config_path"])).resolve()
        != EXPECTED_SOURCE_CONFIG.resolve()
        or tr3d["source_config_sha256"] != EXPECTED_SOURCE_CONFIG_SHA256
        or tr3d["raw_checkpoint_argument_allowed"] is not False
        or tr3d["raw_config_argument_allowed"] is not False
        or tr3d["scannet_checkpoint_or_config_allowed"] is not False
    ):
        raise ValueError("v3 TR3D binding/isolation contract differs")
    source_config = regular_file(EXPECTED_SOURCE_CONFIG, "CA-1M source config")
    if sha256_file(source_config) != EXPECTED_SOURCE_CONFIG_SHA256:
        raise ValueError("CA-1M source config changed after v3 freeze")
    _executable(Path(str(tr3d["worker_python"])), "TR3D Python")
    regular_file(Path(str(tr3d["worker_script"])), "CA-1M terminal worker")
    regular_directory(Path(str(tr3d["runtime_root"])), "TR3D runtime root")
    regular_directory(Path(str(tr3d["project_root"])), "TR3D project root")
    regular_directory(Path(str(tr3d["vendor_root"])), "TR3D vendor root")
    expected_protocol = {
        "cache_schema": "boxfusion.ca1m_tr3d_terminal_observer.v1",
        "prefix_id": "p100_gap20",
        "pixel_stride": 4,
        "voxel_size_m": 0.01,
        "min_depth_m": 0.1,
        "max_depth_m": 6.0,
        "near_iou": 0.15,
        "score_threshold": 0.01,
        "max_proposals": 256,
        "adapter_mode": "genuine",
    }
    if dict(protocol) != expected_protocol:
        raise ValueError("v3 terminal protocol differs from the frozen CA schema")
    _expect_keys(
        outputs,
        {
            "terminal_cache_root", "candidate_evidence_root", "report_root",
            "log_root", "proposal_cache_root", "create_only",
            "exact_proposal_cache_count", "exact_terminal_cache_count",
            "exact_candidate_evidence_count",
        },
        "outputs",
    )
    if (
        outputs["create_only"] is not True
        or int(outputs["exact_proposal_cache_count"]) != 100
        or int(outputs["exact_terminal_cache_count"]) != 100
        or int(outputs["exact_candidate_evidence_count"]) != 100
    ):
        raise ValueError("v3 output count/create-only contract differs")
    output_paths = {
        name: Path(str(outputs[name])).resolve()
        for name in (
            "terminal_cache_root", "candidate_evidence_root", "report_root", "log_root"
            , "proposal_cache_root"
        )
    }
    if len(set(output_paths.values())) != len(output_paths):
        raise ValueError("v3 output roots must be distinct")
    if any(EXPECTED_NAMESPACE not in os.fspath(path) for path in output_paths.values()):
        raise ValueError("all v3 outputs must be isolated in the v3 namespace")
    _expect_keys(forbidden, {"terminal_cache_roots", "benefit_datasets", "reason"}, "forbidden_reuse")
    forbidden_paths = {
        Path(str(path)).resolve()
        for path in (*forbidden["terminal_cache_roots"], *forbidden["benefit_datasets"])
    }
    if any(path in forbidden_paths for path in output_paths.values()):
        raise ValueError("v3 output aliases a forbidden v1/v2 artifact")
    if "regenerate all exact100" not in str(forbidden["reason"]):
        raise ValueError("v3 config does not require exact100 regeneration")
    terminal_inventory = _artifact_inventory(
        output_paths["terminal_cache_root"], "_ca1m_tr3d_terminal.npz", scenes
    )
    proposal_inventory = _artifact_inventory(
        output_paths["proposal_cache_root"], "_ca1m_tr3d_proposals.npz", scenes
    )
    evidence_inventory = _artifact_inventory(
        output_paths["candidate_evidence_root"], "_ca1m_native_b6.npz", scenes
    )
    binding = None
    if binding_path is not None:
        binding = load_checkpoint_binding(binding_path)
        if (
            binding.work_root != EXPECTED_WORK_ROOT.resolve()
            or binding.source_config_path != source_config
            or binding.source_config_sha256 != EXPECTED_SOURCE_CONFIG_SHA256
        ):
            raise ValueError("bound checkpoint disagrees with v3 collection config")
    return {
        "schema": REPORT_SCHEMA,
        "complete": True,
        "static_preflight": binding is None,
        "checkpoint_binding_valid": binding is not None,
        "ready_for_gpu": False,
        "run_authorized": False,
        "blocked_reasons": [
            "final_route_frame_lineage_manifest_pending",
            "g0_clip_reliable_topk3_anchor_manifest_pending",
            "retrained_native_b6_artifacts_pending",
            "proposal_overlay_split_not_materialized",
        ],
        "ground_truth_access": False,
        "validation_ground_truth_access": False,
        "config_path": os.fspath(source),
        "config_sha256": sha256_file(source),
        "namespace": EXPECTED_NAMESPACE,
        "scene_count": len(scenes),
        "scene_list_path": os.fspath(Path(str(scene_cfg["path"])).resolve()),
        "scene_list_sha256": EXPECTED_SCENE_SHA256,
        "checkpoint_binding": None if binding is None else {
            "path": os.fspath(binding.manifest_path),
            "sha256": binding.manifest_sha256,
            "checkpoint_path": os.fspath(binding.checkpoint_path),
            "checkpoint_sha256": binding.checkpoint_sha256,
            "effective_config_path": os.fspath(binding.effective_config_path),
            "effective_config_sha256": binding.effective_config_sha256,
        },
        "terminal_inventory": terminal_inventory,
        "proposal_inventory": proposal_inventory,
        "candidate_evidence_inventory": evidence_inventory,
        "output_roots": {name: os.fspath(path) for name, path in output_paths.items()},
        "forbidden_v1_v2_reuse": True,
    }


def write_json_create_only(path: Path, payload: dict[str, Any]) -> None:
    if path.is_symlink():
        raise ValueError(f"preflight output must not be a symlink: {path}")
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{target.name}.", dir=target.parent, delete=False
        ) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        target.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite preflight report: {target}") from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/ca1m_tr3d_terminal_train100_v3.json",
    )
    value.add_argument("--checkpoint-binding", type=Path)
    value.add_argument("--output", type=Path)
    return value


def main() -> int:
    args = parser().parse_args()
    report = validate_config(args.config, args.checkpoint_binding)
    if args.output is not None:
        write_json_create_only(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
