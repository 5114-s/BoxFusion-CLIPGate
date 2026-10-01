#!/usr/bin/env python3
"""Run the preregistered exact-20 E961 CA-only outer-dev evaluation.

``all`` first creates/verifies the deterministic runtime preregistration.  No
expanded training receipt, checkpoint, anchor array, or GT is inspected until
that create-only seal exists.  Proposal inference is point-only and GT-free;
fold-0 GT becomes reachable only after an exact finite CA-only proposal
collection is sealed.  The command never launches an inner training job.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_checkpoint_binding import (  # noqa: E402
    FORBIDDEN_SCANNET_SHA256,
)
from boxfusion.ca1m_tr3d_e961_outer_eval_v1 import (  # noqa: E402
    BINDING_PATH,
    BINDING_SCHEMA,
    COLLECTION_PATH,
    COLLECTION_SCHEMA,
    CONFIG_PATH,
    CONTINUATION_PATH,
    CONTINUATION_SCHEMA,
    DIAGNOSTIC_ROOT,
    INNER_AUTHORIZATION_PATH,
    INNER_AUTHORIZATION_SCHEMA,
    INNER_ROLES,
    NAMESPACE,
    POINT_CONFIG_SHA256,
    POINT_PARITY_SHA256,
    PREREGISTRATION_PATH,
    REPORT_PATH,
    REPORT_SCHEMA,
    STOP_PATH,
    STOP_SCHEMA,
    TRAIN_WORK_ROOT,
    continuation_gate,
    create_or_verify_json,
    expected_training_receipt,
    guard_fixed_path,
    load_config,
    metric_delta,
    read_json,
    regular_directory,
    regular_file,
    scene_ids,
    seal_preregistration,
    sha256_bytes,
    sha256_file,
    validate_inner_authorization,
    validate_preregisterable_static,
    validate_preregistration,
    validate_protocol_preregistration,
    validate_run_tag,
)
from boxfusion.ca1m_tr3d_inference_contract import (  # noqa: E402
    validate_ca1m_point_inference_config,
)
from boxfusion.ca1m_tr3d_terminal import terminal_world_to_local  # noqa: E402
from boxfusion.ca1m_tr3d_terminal_v4 import (  # noqa: E402
    PREFIX_ID,
    ProposalCacheSummary,
    frame_lineage_json,
    load_proposal_cache,
    proposal_cache_payload,
    write_npz_create_only,
)
from boxfusion.ca1m_tr3d_worker_client import CA1MTR3DWorker  # noqa: E402
from tools.run_ca1m_tr3d_proposal_cache_v4 import _build_scene_points  # noqa: E402
from tools import run_ca1m_tr3d_xfit_r2_outer_dev_eval as sealed_r2  # noqa: E402


TRAIN_RECEIPT_SCHEMA = "boxfusion.tr3d.ca1m_e961_outer_train_run.r2"
ANCHOR_SCHEMA = "boxfusion.ca1m_tr3d_xfit_r2_fold0_final_base_b6_oof.v1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class E961Binding:
    path: Path
    sha256: str
    run_tag: str
    receipt_path: Path
    receipt_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    effective_config_path: Path
    effective_config_sha256: str


def _regular_bound_artifact(
    record: Mapping[str, Any], expected_path: Path | None, name: str,
    *, immutable: bool = False, require_bytes: bool = True,
) -> Path:
    source = regular_file(Path(str(record.get("path", ""))), name, immutable=immutable)
    if expected_path is not None and source != expected_path.resolve():
        raise ValueError(f"{name} path differs")
    expected_sha = str(record.get("sha256", ""))
    if SHA256.fullmatch(expected_sha) is None or sha256_file(source) != expected_sha:
        raise ValueError(f"{name} SHA256 differs")
    if require_bytes and source.stat().st_size != int(record.get("bytes", -1)):
        raise ValueError(f"{name} byte count differs")
    return source


def _read_training_receipt_after_prereg(
    config_path: Path, run_tag: str,
) -> tuple[Path, dict[str, Any]]:
    """First expanded-training reachability point in the implementation."""

    validate_preregistration(config_path, run_tag)
    tag = validate_run_tag(run_tag)
    expected = expected_training_receipt(tag)
    guard_fixed_path(
        expected, Path("/extra/ZhaoX/tr3d_ca1m_e961_outer_train_r2"),
        "E961 outer training receipt",
    )
    # The training-side verifier is the sole authority for its exact receipt
    # schema and re-hashes the checkpoint/effective config.  Import is delayed
    # until after preregistration so even its verification cannot run early.
    training_tools = Path("/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/tools")
    if str(training_tools) not in sys.path:
        sys.path.insert(0, str(training_tools))
    training_path = regular_file(
        training_tools / "tr3d_ca1m_e961_outer_train_r2.py",
        "E961 outer training verifier",
    )
    spec = importlib.util.spec_from_file_location(
        "_boxfusion_bound_e961_outer_train_r2", training_path
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the bound E961 outer training verifier")
    training_contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training_contract)

    verified_value = training_contract.verify_success_receipt(expected)
    source, value = read_json(expected, "E961 outer training receipt", immutable=True)
    if verified_value != value:
        raise ValueError("training-side E961 receipt verification differs")
    if (
        value.get("schema") != TRAIN_RECEIPT_SCHEMA
        or value.get("complete") is not True
        or value.get("create_only") is not True
        or value.get("run_tag") != tag
        or value.get("role") != "outer_dev"
        or value.get("status") != "success"
        or value.get("exit_code") != 0
    ):
        raise ValueError("E961 outer training receipt identity/completion differs")
    terminal = value.get("terminal") or {}
    audit = terminal.get("checkpoint_audit") or {}
    checkpoint = audit.get("checkpoint") or {}
    effective_meta = terminal.get("effective_config") or {}
    claim = value.get("training_claim") or {}
    if (
        audit.get("schema")
        != "boxfusion.tr3d.mmengine_terminal_checkpoint_audit.r2"
        or audit.get("trusted_local_pickle_cpu_loaded") is not True
        or audit.get("meta_iter") != 11268
        or (audit.get("optimizer") or {}).get("all_steps") != 11268
        or (audit.get("scheduler") or {}).get("milestones") != [7512, 10329]
        or (audit.get("message_hub") or {}).get("history_length_each") != 11268
        or (terminal.get("inventory") or {}).get("no_symlinks") is not True
        or (terminal.get("inventory") or {}).get("no_special_files") is not True
        or claim != {
            "ca1m_training_data_only": True,
            "scannet_training_weights_loaded": False,
            "scannet_training_data_configured_or_opened": False,
            "plugin_imports_scannet_adapter_class_definition": True,
        }
        or value.get("retry_policy") != {
            "authorization_consumption_released": False,
            "different_run_tag_retry_allowed": False,
        }
    ):
        raise ValueError("E961 outer training protocol differs")
    log = value.get("log") or {}
    log_path = _regular_bound_artifact(
        log, expected.parent / "outer_dev.log", "outer training wrapper log",
    )
    log_text = log_path.read_text(encoding="utf-8", errors="strict")
    if (
        log.get("exit_marker") != "TRAIN_EXIT=0"
        or not log_text.endswith("TRAIN_EXIT=0\n")
        or len(re.findall(r"(?m)^TRAIN_EXIT=.*$", log_text)) != 1
        or "Traceback (most recent call last)" in log_text
        or "Iter(val)" in log_text or "Iter(test)" in log_text
    ):
        raise ValueError("E961 outer training wrapper log is not a unique success")
    return source, value


def _expected_work_root(run_tag: str) -> Path:
    return TRAIN_WORK_ROOT / validate_run_tag(run_tag)


def seal_binding(config_path: Path, run_tag: str) -> E961Binding:
    prereg_path, _ = validate_preregistration(config_path, run_tag)
    source, cfg = load_config(config_path)
    receipt_path, receipt = _read_training_receipt_after_prereg(config_path, run_tag)
    work_root = _expected_work_root(run_tag)
    terminal = receipt.get("terminal") or {}
    audit = terminal.get("checkpoint_audit") or {}
    checkpoint = _regular_bound_artifact(
        audit.get("checkpoint") or {}, work_root / "iter_11268.pth",
        "E961 outer iter-11268 checkpoint",
    )
    effective_record = terminal.get("effective_config") or {}
    effective_inventory = (
        (terminal.get("inventory") or {}).get("files") or {}
    ).get("outer_dev.py") or {}
    effective_bound = {
        "path": effective_record.get("path"),
        "sha256": effective_record.get("sha256"),
        "bytes": effective_inventory.get("bytes"),
    }
    effective = _regular_bound_artifact(
        effective_bound, work_root / "outer_dev.py",
        "E961 outer effective config",
    )
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha in FORBIDDEN_SCANNET_SHA256:
        raise ValueError("E961 outer checkpoint matches a forbidden ScanNet artifact")
    point = cfg["point_inference"]
    point_path = regular_file(Path(point["path"]), "point-only inference config")
    inference = validate_ca1m_point_inference_config(
        inference_path=point_path,
        inference_sha256=POINT_CONFIG_SHA256,
        effective_training_path=effective,
        effective_training_sha256=sha256_file(effective),
    )
    payload = {
        "schema": BINDING_SCHEMA,
        "complete": True, "create_only": True, "namespace": NAMESPACE,
        "role": "outer_dev", "run_tag": validate_run_tag(run_tag),
        "checkpoint_selection": False,
        "raw_detector_ap_checkpoint_selection": False,
        "training_receipt": {
            "path": str(receipt_path), "sha256": sha256_file(receipt_path),
            "schema": TRAIN_RECEIPT_SCHEMA,
        },
        "checkpoint": {
            "path": str(checkpoint), "sha256": checkpoint_sha,
            "bytes": checkpoint.stat().st_size, "optimizer_updates": 11268,
            "unique_final_checkpoint": True,
        },
        "effective_config": {
            "path": str(effective), "sha256": sha256_file(effective),
            "bytes": effective.stat().st_size,
        },
        "point_inference": inference,
        "preregistration": {
            "path": str(prereg_path), "sha256": sha256_file(prereg_path),
            "sealed_before_expanded_checkpoint_access": True,
        },
        "evaluation_config": {
            "path": str(source), "sha256": sha256_file(source),
        },
        "access": {
            "fold0_gt_access_during_binding": False,
            "fold1_access": False,
            "official_validation_access": False,
            "scannet_training_weights_loaded": False,
            "scannet_training_data_configured_or_opened": False,
            "plugin_imports_scannet_adapter_class_definition": True,
            "scannet_adapter_instantiated": False,
        },
        "implementation": cfg["implementation"],
    }
    guard_fixed_path(BINDING_PATH, ROOT, "E961 outer checkpoint binding")
    create_or_verify_json(BINDING_PATH, payload, "E961 outer checkpoint binding")
    return load_binding(config_path, run_tag)


def load_binding(config_path: Path, run_tag: str) -> E961Binding:
    prereg_path, _ = validate_preregistration(config_path, run_tag)
    receipt_path, receipt = _read_training_receipt_after_prereg(config_path, run_tag)
    source, value = read_json(BINDING_PATH, "E961 outer checkpoint binding", immutable=True)
    if (
        value.get("schema") != BINDING_SCHEMA
        or value.get("complete") is not True
        or value.get("create_only") is not True
        or value.get("namespace") != NAMESPACE
        or value.get("role") != "outer_dev"
        or value.get("run_tag") != validate_run_tag(run_tag)
        or value.get("checkpoint_selection") is not False
        or value.get("raw_detector_ap_checkpoint_selection") is not False
        or (value.get("training_receipt") or {}).get("sha256") != sha256_file(receipt_path)
        or (value.get("preregistration") or {}).get("sha256") != sha256_file(prereg_path)
    ):
        raise ValueError("E961 outer checkpoint binding contract differs")
    work_root = _expected_work_root(run_tag)
    checkpoint = _regular_bound_artifact(
        value.get("checkpoint") or {}, work_root / "iter_11268.pth",
        "bound E961 outer checkpoint",
    )
    effective = _regular_bound_artifact(
        value.get("effective_config") or {}, work_root / "outer_dev.py",
        "bound E961 effective config",
    )
    terminal = receipt.get("terminal") or {}
    if (
        ((terminal.get("checkpoint_audit") or {}).get("checkpoint") or {}).get(
            "sha256"
        ) != sha256_file(checkpoint)
        or (terminal.get("effective_config") or {}).get("sha256")
        != sha256_file(effective)
    ):
        raise ValueError("E961 binding differs from training receipt")
    return E961Binding(
        path=source, sha256=sha256_file(source), run_tag=validate_run_tag(run_tag),
        receipt_path=receipt_path, receipt_sha256=sha256_file(receipt_path),
        checkpoint_path=checkpoint, checkpoint_sha256=sha256_file(checkpoint),
        effective_config_path=effective, effective_config_sha256=sha256_file(effective),
    )


def _point_provenance(
    config_path: Path, cfg: Mapping[str, Any], binding: E961Binding,
) -> dict[str, Any]:
    inference_path = regular_file(
        Path(cfg["point_inference"]["path"]), "point-only inference config"
    )
    validate_ca1m_point_inference_config(
        inference_path=inference_path,
        inference_sha256=POINT_CONFIG_SHA256,
        effective_training_path=binding.effective_config_path,
        effective_training_sha256=binding.effective_config_sha256,
    )
    parity_path, parity = read_json(
        Path(cfg["point_lineage"]["receipt_path"]), "same-path point parity",
        immutable=True,
    )
    if (
        sha256_file(parity_path) != POINT_PARITY_SHA256
        or parity.get("complete") is not True
        or parity.get("ground_truth_access") is not False
        or parity.get("point_array_parity_scene_count") != 100
    ):
        raise ValueError("same-path point parity receipt differs")
    scenes = scene_ids(cfg)
    parity_scenes = parity.get("scenes") or {}
    if not set(scenes).issubset(parity_scenes):
        raise ValueError("same-path parity does not cover exact fold0")
    rows = {
        name: {
            "path": str(regular_file(Path(record["path"]), f"implementation {name}")),
            "sha256": record["sha256"],
        }
        for name, record in sorted(cfg["implementation"].items())
    }
    rows["evaluation_config"] = {
        "path": str(config_path), "sha256": sha256_file(config_path),
    }
    rows["checkpoint_binding"] = {
        "path": str(binding.path), "sha256": binding.sha256,
    }
    code_json = json.dumps(
        {"schema": "boxfusion.ca1m_tr3d_e961_proposal_code.v1", "files": rows},
        separators=(",", ":"), sort_keys=True,
    )
    return {
        "inference_path": inference_path,
        "parity_path": parity_path,
        "parity_scenes": parity_scenes,
        "code_json": code_json,
        "code_sha256": sha256_bytes(code_json.encode()),
    }


def _require_cuda_device(device: str) -> str:
    if str(device) != "cuda:0":
        raise ValueError("formal E961 proposal stage requires isolated cuda:0")
    return str(device)


def _run_proposals(config_path: Path, run_tag: str, device: str) -> dict[str, Any]:
    device = _require_cuda_device(device)
    validate_preregistration(config_path, run_tag)
    config_source, cfg = load_config(config_path)
    scenes = scene_ids(cfg)
    binding = load_binding(config_path, run_tag)
    provenance = _point_provenance(config_source, cfg, binding)
    parity_scenes = provenance["parity_scenes"]
    protocol = cfg["proposal_stage"]["protocol"]
    processed = cfg["point_lineage"]["processed_rgbd"]
    data_root = regular_directory(Path(processed["root"]), "same-path processed RGB-D")
    output_root = Path(cfg["proposal_stage"]["output_root"])
    guard_fixed_path(output_root / ".path_guard", Path("/extra/ZhaoX"), "proposal root")
    if output_root.is_symlink():
        raise ValueError("E961 proposal output root must not be a symlink")
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    expected_names = {f"{scene}_ca1m_tr3d_proposals_v4.npz" for scene in scenes}
    actual_names = {
        path.name for path in output_root.iterdir()
        if path.is_file() and not path.is_symlink() and path.suffix == ".npz"
    }
    if actual_names - expected_names:
        raise ValueError("E961 proposal root contains non-fold0 NPZ artifacts")
    reports: dict[str, Any] = {}
    pending: list[str] = []
    for scene in scenes:
        target = output_root / f"{scene}_ca1m_tr3d_proposals_v4.npz"
        if target.exists() or target.is_symlink():
            loaded = load_proposal_cache(
                target, expected_scene=scene,
                expected_binding_sha256=binding.sha256,
            )
            summary = loaded["summary"]
            if (
                summary.source_points_sha256
                != (parity_scenes.get(scene) or {}).get("world_point_array_sha256")
                or summary.checkpoint_sha256 != binding.checkpoint_sha256
                or summary.config_sha256 != POINT_CONFIG_SHA256
                or summary.code_manifest_sha256 != provenance["code_sha256"]
            ):
                raise ValueError(f"{scene}: resumed E961 proposal provenance differs")
            reports[scene] = {**summary.as_dict(), "resumed": True}
        else:
            pending.append(scene)
    verified_points: dict[str, tuple[Any, Any, Any, np.ndarray]] = {}
    for scene in pending:
        built = _build_scene_points(
            data_root=data_root, scene=scene, processed=processed, protocol=protocol,
        )
        point_sha = hashlib.sha256(built[3].tobytes(order="C")).hexdigest()
        if point_sha != (parity_scenes.get(scene) or {}).get("world_point_array_sha256"):
            raise ValueError(f"{scene}: E961 proposal points differ from sealed same path")
        verified_points[scene] = built
        print(f"E961 fold0 CPU same-point parity | scene={scene}, PASS", flush=True)
    runtime = cfg["runtime"]
    worker_python = Path(runtime["worker_python"]).resolve()
    if not worker_python.is_file() or not os.access(worker_python, os.X_OK):
        raise FileNotFoundError(f"missing executable worker Python: {worker_python}")
    worker_script = regular_file(Path(runtime["worker_script"]), "CA TR3D worker")
    runtime_root = regular_directory(Path(runtime["runtime_root"]), "TR3D runtime")
    project_root = regular_directory(Path(runtime["project_root"]), "TR3D project")
    vendor_root = regular_directory(Path(runtime["vendor_root"]), "TR3D vendor")
    if pending:
        with CA1MTR3DWorker(
            python=str(worker_python), worker_script=str(worker_script),
            runtime_root=str(runtime_root), config=str(provenance["inference_path"]),
            checkpoint=str(binding.checkpoint_path), project_root=str(project_root),
            vendor_root=str(vendor_root),
            startup_timeout_s=float(runtime["startup_timeout_s"]),
            device=device,
            extra_args=(
                "--score-threshold", str(protocol["score_threshold"]),
                "--max-proposals", str(protocol["max_proposals"]),
            ),
        ) as worker:
            if worker.adapter_mode != "genuine":
                raise ValueError("formal E961 proposal stage forbids synthetic TR3D")
            for scene in pending:
                rgb, poses, frames, points = verified_points.pop(scene)
                point_sha = hashlib.sha256(points.tobytes(order="C")).hexdigest()
                world_to_local = terminal_world_to_local(poses[int(frames[0])])
                result = worker.infer(
                    scene_id=scene, prefix_id=PREFIX_ID,
                    points_world_xyzrgb=points, world_to_local=world_to_local,
                )
                if result.source_points_sha256 != point_sha:
                    raise ValueError(f"{scene}: worker/source point SHA256 differs")
                lineage_json = frame_lineage_json(scene, len(rgb))
                summary = ProposalCacheSummary(
                    scene_id=scene, frame_count=len(rgb), used_frame_count=len(frames),
                    point_count=len(points), candidate_count=len(result.scores),
                    model_runtime_s=float(result.model_runtime_s),
                    source_points_sha256=point_sha,
                    frame_lineage_sha256=sha256_bytes(lineage_json.encode()),
                    checkpoint_binding_sha256=binding.sha256,
                    checkpoint_sha256=binding.checkpoint_sha256,
                    config_sha256=POINT_CONFIG_SHA256,
                    code_manifest_sha256=provenance["code_sha256"],
                    adapter_mode=result.adapter_mode, device=device,
                )
                payload = proposal_cache_payload(
                    summary=summary, used_frame_ids=frames,
                    world_to_local=world_to_local,
                    candidate_corners_world=result.corners_world,
                    candidate_scores=result.scores,
                    candidate_point_count=result.point_counts,
                    candidate_boxes_local=result.boxes_local,
                    candidate_labels=result.labels,
                    frame_lineage=lineage_json,
                    code_manifest=provenance["code_json"],
                )
                write_npz_create_only(
                    output_root / f"{scene}_ca1m_tr3d_proposals_v4.npz", payload
                )
                reports[scene] = {**summary.as_dict(), "resumed": False}
    collection = _seal_collection(config_path, run_tag)
    return {
        "schema": "boxfusion.ca1m_tr3d_e961_outer_dev_proposal_run.v1",
        "complete": True, "namespace": NAMESPACE,
        "scene_count": 20, "fold0_only": True,
        "ground_truth_access": False, "anchor_access": False,
        "fold1_access": False, "official_validation_access": False,
        "resumed_count": sum(bool(row["resumed"]) for row in reports.values()),
        "collection": collection,
    }


def _seal_collection(config_path: Path, run_tag: str) -> dict[str, Any]:
    validate_preregistration(config_path, run_tag)
    config_source, cfg = load_config(config_path)
    scenes = scene_ids(cfg)
    binding = load_binding(config_path, run_tag)
    provenance = _point_provenance(config_source, cfg, binding)
    root = regular_directory(Path(cfg["proposal_stage"]["output_root"]), "E961 proposal root")
    expected = {f"{scene}_ca1m_tr3d_proposals_v4.npz" for scene in scenes}
    actual = {
        path.name for path in root.iterdir()
        if path.is_file() and not path.is_symlink() and path.suffix == ".npz"
    }
    if actual != expected:
        raise ValueError("E961 proposal collection is not exact fold0-20")
    rows: list[dict[str, Any]] = []
    total = 0
    for scene in scenes:
        path = regular_file(
            root / f"{scene}_ca1m_tr3d_proposals_v4.npz",
            f"E961 proposal {scene}", immutable=True,
        )
        loaded = load_proposal_cache(
            path, expected_scene=scene, expected_binding_sha256=binding.sha256
        )
        summary = loaded["summary"]
        if (
            summary.adapter_mode != "genuine"
            or summary.device != "cuda:0"
            or summary.checkpoint_sha256 != binding.checkpoint_sha256
            or summary.source_points_sha256
            != (provenance["parity_scenes"].get(scene) or {}).get(
                "world_point_array_sha256"
            )
            or summary.config_sha256 != POINT_CONFIG_SHA256
            or summary.code_manifest_sha256 != provenance["code_sha256"]
            or not np.isfinite(loaded["candidate_corners_world"]).all()
            or not np.isfinite(loaded["candidate_scores"]).all()
            or np.any(loaded["candidate_labels"] != 0)
        ):
            raise ValueError(f"{scene}: E961 proposal is not finite genuine CA-only")
        total += summary.candidate_count
        rows.append({
            "scene_id": scene, "path": str(path), "sha256": sha256_file(path),
            "candidate_count": summary.candidate_count,
            "point_count": summary.point_count,
            "source_points_sha256": summary.source_points_sha256,
            "config_sha256": summary.config_sha256,
            "code_manifest_sha256": summary.code_manifest_sha256,
        })
    payload = {
        "schema": COLLECTION_SCHEMA,
        "complete": True, "create_only": True, "namespace": NAMESPACE,
        "partition": "official_train_fold0_reused_dev_exact20",
        "scene_count": 20, "candidate_count": total,
        "finite": True, "class_agnostic_labels_zero": True,
        "genuine_ca_only_checkpoint": True,
        "same_point_path": True,
        "ground_truth_access": False, "anchor_access": False,
        "fold1_access": False, "official_validation_access": False,
        "checkpoint_binding": {
            "path": str(binding.path), "sha256": binding.sha256,
            "checkpoint_sha256": binding.checkpoint_sha256,
        },
        "preregistration": {
            "path": str(PREREGISTRATION_PATH),
            "sha256": sha256_file(PREREGISTRATION_PATH),
        },
        "scene_list": cfg["scene_contract"],
        "point_provenance": {
            "parity_path": str(provenance["parity_path"]),
            "parity_sha256": sha256_file(provenance["parity_path"]),
            "point_inference_config_sha256": POINT_CONFIG_SHA256,
            "code_manifest_sha256": provenance["code_sha256"],
            "protocol": cfg["proposal_stage"]["protocol"],
        },
        "scenes": rows,
    }
    guard_fixed_path(COLLECTION_PATH, ROOT, "E961 proposal collection")
    create_or_verify_json(COLLECTION_PATH, payload, "E961 exact20 proposal collection")
    return payload


def _load_collection(
    config_path: Path, run_tag: str,
) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    validate_preregistration(config_path, run_tag)
    config_source, cfg = load_config(config_path)
    binding = load_binding(config_path, run_tag)
    provenance = _point_provenance(config_source, cfg, binding)
    path, value = read_json(COLLECTION_PATH, "E961 exact20 proposal collection", immutable=True)
    if (
        value.get("schema") != COLLECTION_SCHEMA
        or value.get("complete") is not True
        or value.get("create_only") is not True
        or value.get("namespace") != NAMESPACE
        or value.get("partition") != "official_train_fold0_reused_dev_exact20"
        or value.get("scene_count") != 20
        or value.get("finite") is not True
        or value.get("class_agnostic_labels_zero") is not True
        or value.get("genuine_ca_only_checkpoint") is not True
        or value.get("same_point_path") is not True
        or value.get("ground_truth_access") is not False
        or value.get("anchor_access") is not False
        or value.get("fold1_access") is not False
        or value.get("official_validation_access") is not False
        or (value.get("checkpoint_binding") or {}).get("sha256") != binding.sha256
        or (value.get("checkpoint_binding") or {}).get("checkpoint_sha256")
        != binding.checkpoint_sha256
        or (value.get("preregistration") or {}).get("sha256")
        != sha256_file(PREREGISTRATION_PATH)
    ):
        raise ValueError("E961 proposal collection contract differs")
    scenes = scene_ids(cfg)
    raw_rows = value.get("scenes") or []
    rows = {str(row.get("scene_id")): row for row in raw_rows}
    if len(raw_rows) != 20 or len(rows) != 20 or set(rows) != set(scenes):
        raise ValueError("E961 proposal collection does not cover exact fold0")
    root = Path(cfg["proposal_stage"]["output_root"]).resolve()
    total = 0
    for scene in scenes:
        record = rows[scene]
        proposal = regular_file(Path(record["path"]), f"E961 proposal {scene}", immutable=True)
        if (
            proposal.parent != root
            or proposal.name != f"{scene}_ca1m_tr3d_proposals_v4.npz"
            or sha256_file(proposal) != record.get("sha256")
        ):
            raise ValueError(f"{scene}: E961 proposal path/hash differs")
        loaded = load_proposal_cache(
            proposal, expected_scene=scene, expected_binding_sha256=binding.sha256
        )
        summary = loaded["summary"]
        if (
            record.get("candidate_count") != summary.candidate_count
            or record.get("point_count") != summary.point_count
            or record.get("source_points_sha256") != summary.source_points_sha256
            or record.get("config_sha256") != POINT_CONFIG_SHA256
            or record.get("code_manifest_sha256") != provenance["code_sha256"]
            or summary.checkpoint_sha256 != binding.checkpoint_sha256
            or summary.source_points_sha256
            != (provenance["parity_scenes"].get(scene) or {}).get(
                "world_point_array_sha256"
            )
            or summary.adapter_mode != "genuine"
            or summary.device != "cuda:0"
            or not np.isfinite(loaded["candidate_corners_world"]).all()
            or not np.isfinite(loaded["candidate_scores"]).all()
            or np.any(loaded["candidate_labels"] != 0)
        ):
            raise ValueError(f"{scene}: sealed E961 proposal differs")
        total += summary.candidate_count
    if value.get("candidate_count") != total:
        raise ValueError("E961 proposal collection candidate count differs")
    return path, value, rows


def _proposal_arrays(
    rows: Mapping[str, Mapping[str, Any]], scenes: tuple[str, ...], binding_sha: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for scene in scenes:
        loaded = load_proposal_cache(
            Path(str(rows[scene]["path"])), expected_scene=scene,
            expected_binding_sha256=binding_sha,
        )
        result[scene] = (
            np.asarray(loaded["candidate_corners_world"], np.float32),
            np.asarray(loaded["candidate_scores"], np.float32),
        )
    return result


def _decision_artifact(
    *, gate: Mapping[str, Any], report_path: Path, binding: E961Binding,
    collection_path: Path,
) -> tuple[Path, dict[str, Any]]:
    common = {
        "complete": True, "create_only": True, "namespace": NAMESPACE,
        "partition": "official_train_fold0_reused_dev_exact20",
        "scene_count": 20, "fold0_role": "reused_dev",
        "fold1_access": False, "official_validation_access": False,
        "raw_detector_ap_role": "diagnostic_only_no_checkpoint_selection",
        "checkpoint_selection": False,
        "preregistration": {
            "path": str(PREREGISTRATION_PATH),
            "sha256": sha256_file(PREREGISTRATION_PATH),
        },
        "checkpoint_binding": {
            "path": str(binding.path), "sha256": binding.sha256,
            "checkpoint_sha256": binding.checkpoint_sha256,
        },
        "proposal_collection": {
            "path": str(collection_path), "sha256": sha256_file(collection_path),
            "exact20_finite_ca_only": True,
        },
        "evaluation_report": {
            "path": str(report_path), "sha256": sha256_file(report_path),
        },
        "continuation_gate": gate,
        "pass": gate["pass"],
        "authorized_roles": gate["authorized_inner_roles"],
    }
    continuation = {"schema": CONTINUATION_SCHEMA, **common}
    guard_fixed_path(CONTINUATION_PATH, ROOT, "E961 continuation receipt")
    continuation_path = create_or_verify_json(
        CONTINUATION_PATH, continuation, "E961 continuation receipt"
    )
    bound = {
        "path": str(continuation_path), "sha256": sha256_file(continuation_path),
        "schema": CONTINUATION_SCHEMA,
    }
    if gate["pass"] is True:
        if STOP_PATH.exists() or STOP_PATH.is_symlink():
            raise FileExistsError("a create-only E961 STOP receipt already exists")
        payload = {
            "schema": INNER_AUTHORIZATION_SCHEMA,
            "complete": True, "create_only": True, "namespace": NAMESPACE,
            "pass": True, "authorized_roles": list(INNER_ROLES),
            "authorization_source": "preregistered_outer_dev_gate_only",
            "continuation_receipt": bound,
            "fold0_gt_role": "reused_dev_gate_only_not_inner_training",
            "fold1_access": False, "official_validation_access": False,
            "checkpoint_selection": False,
        }
        guard_fixed_path(INNER_AUTHORIZATION_PATH, ROOT, "E961 inner authorization")
        path = create_or_verify_json(
            INNER_AUTHORIZATION_PATH, payload, "E961 inner training authorization"
        )
        return path, payload
    if INNER_AUTHORIZATION_PATH.exists() or INNER_AUTHORIZATION_PATH.is_symlink():
        raise FileExistsError("a create-only E961 PASS authorization already exists")
    payload = {
        "schema": STOP_SCHEMA,
        "complete": True, "create_only": True, "namespace": NAMESPACE,
        "pass": False, "authorized_roles": [],
        "action": "stop_without_inner_training",
        "continuation_receipt": bound,
        "fold1_access": False, "official_validation_access": False,
    }
    guard_fixed_path(STOP_PATH, ROOT, "E961 inner STOP receipt")
    path = create_or_verify_json(STOP_PATH, payload, "E961 inner training STOP receipt")
    return path, payload


def evaluate(config_path: Path, run_tag: str) -> dict[str, Any]:
    prereg_path, prereg = validate_preregistration(config_path, run_tag)
    config_source, cfg = load_config(config_path)
    scenes = scene_ids(cfg)
    binding = load_binding(config_path, run_tag)
    # Exact20 proposal validation is the final condition before any anchor or GT access.
    collection_path, collection, e961_rows = _load_collection(config_path, run_tag)

    # Anchor/comparison arrays become reachable only after collection seal.
    anchor_manifest, anchors = sealed_r2._load_anchor_shadow(cfg, scenes)
    v1_manifest, v1_rows, v1_binding_sha = sealed_r2._load_v1_comparison(cfg, scenes)

    # Sole fold0-GT reachability point.  The sealed helper checks every scene is fold0.
    gt_inventory_path, ground_truth, gt_hashes = sealed_r2._load_ground_truth(cfg, scenes)
    e961_candidates = _proposal_arrays(e961_rows, scenes, binding.sha256)
    v1_candidates = sealed_r2._proposal_arrays(v1_rows, scenes, v1_binding_sha)
    baseline_predictions = {
        scene: (anchors[scene][0], anchors[scene][1]) for scene in scenes
    }
    baseline_metrics = sealed_r2._metrics(baseline_predictions, ground_truth, scenes)
    raw_e961_metrics = sealed_r2._metrics(e961_candidates, ground_truth, scenes)
    raw_v1_same_path_metrics = sealed_r2._metrics(v1_candidates, ground_truth, scenes)
    oracle_predictions, oracle_summary = sealed_r2._same_gt_oracle(
        anchors=anchors, candidates=e961_candidates,
        ground_truth=ground_truth, scenes=scenes,
    )
    v1_oracle_predictions, v1_oracle_summary = sealed_r2._same_gt_oracle(
        anchors=anchors, candidates=v1_candidates,
        ground_truth=ground_truth, scenes=scenes,
    )
    oracle_metrics = sealed_r2._metrics(oracle_predictions, ground_truth, scenes)
    v1_oracle_metrics = sealed_r2._metrics(v1_oracle_predictions, ground_truth, scenes)
    oracle_delta = metric_delta(oracle_metrics, baseline_metrics)
    v1_oracle_delta = metric_delta(v1_oracle_metrics, baseline_metrics)
    gate = continuation_gate(
        proposal_integrity_pass=(
            collection.get("scene_count") == 20
            and collection.get("finite") is True
            and collection.get("class_agnostic_labels_zero") is True
            and collection.get("genuine_ca_only_checkpoint") is True
            and collection.get("same_point_path") is True
        ),
        scene_count=20,
        replacement_count=oracle_summary["replacement_count"],
        replacement_scene_count=oracle_summary["replacement_scene_count"],
        oracle_ap_delta=oracle_delta,
    )
    report = {
        "schema": REPORT_SCHEMA,
        "complete": True, "create_only": True, "namespace": NAMESPACE,
        "partition": "official_train_fold0_reused_dev_exact20",
        "fold0_role": "reused_dev",
        "fold0_prior_exposure": prereg["fold0_prior_exposure"],
        "official_validation_comparable": False,
        "fold1_access": False, "official_validation_access": False,
        "scene_count": 20, "checkpoint_selection": False,
        "metric": prereg["metric"],
        "raw_detector_ap_role": "diagnostic_only_no_checkpoint_selection",
        "raw_detector_diagnostic": {
            "e961_outer": raw_e961_metrics,
            "v1_same_point_path": raw_v1_same_path_metrics,
            "e961_minus_v1_same_point_path": metric_delta(
                raw_e961_metrics, raw_v1_same_path_metrics
            ),
            "used_by_continuation_gate": False,
            "checkpoint_selection_authorized": False,
        },
        "final_base_b6_oof_same_gt_oracle_headroom": {
            "anchor_score_source": (
                "CA-native final-base+B6-v2 fold0 OOF; every score model excludes scene"
            ),
            "baseline": baseline_metrics,
            "e961_oracle": oracle_metrics,
            "e961_oracle_ap_delta": oracle_delta,
            "e961_oracle_geometry": oracle_summary,
            "v1_same_path_oracle": v1_oracle_metrics,
            "v1_same_path_oracle_ap_delta": v1_oracle_delta,
            "v1_same_path_oracle_geometry": v1_oracle_summary,
            "oracle_deployable": False,
        },
        "continuation_gate": gate,
        "inner_training_authorization": {
            "authorized": gate["pass"],
            "roles": gate["authorized_inner_roles"],
            "source": "preregistered_outer_dev_gate_only",
        },
        "provenance": {
            "evaluation_config": {
                "path": str(config_source), "sha256": sha256_file(config_source),
            },
            "preregistration": {
                "path": str(prereg_path), "sha256": sha256_file(prereg_path),
            },
            "checkpoint_binding": {
                "path": str(binding.path), "sha256": binding.sha256,
            },
            "proposal_collection": {
                "path": str(collection_path), "sha256": sha256_file(collection_path),
            },
            "fold0_anchor_manifest": {
                "path": str(anchor_manifest), "sha256": sha256_file(anchor_manifest),
            },
            "v1_same_point_path_manifest": {
                "path": str(v1_manifest), "sha256": sha256_file(v1_manifest),
            },
            "opaque_gt_inventory": {
                "path": str(gt_inventory_path), "sha256": sha256_file(gt_inventory_path),
            },
            "fold0_gt_hashes": gt_hashes,
        },
    }
    guard_fixed_path(REPORT_PATH, ROOT, "E961 evaluation report")
    report_path = create_or_verify_json(REPORT_PATH, report, "E961 outer evaluation report")
    decision_path, decision = _decision_artifact(
        gate=gate, report_path=report_path, binding=binding,
        collection_path=collection_path,
    )
    return {
        **report,
        "decision_artifact": {
            "path": str(decision_path), "sha256": sha256_file(decision_path),
            "schema": decision["schema"],
        },
    }


def preflight(config_path: Path, run_tag: str | None) -> dict[str, Any]:
    source, cfg = load_config(config_path)
    static = validate_preregisterable_static(cfg)
    protocol_path, protocol = validate_protocol_preregistration(config_path)
    prereg: dict[str, Any] = {"present": False}
    if PREREGISTRATION_PATH.exists() or PREREGISTRATION_PATH.is_symlink():
        if run_tag is None:
            prereg = {
                "present": True,
                "validation_deferred_until_explicit_run_tag": True,
            }
        else:
            path, value = validate_preregistration(config_path, run_tag)
            prereg = {
                "present": True, "valid": True,
                "path": str(path), "sha256": sha256_file(path),
                "outer_training_run_tag": value["outer_training_run_tag"],
            }
    return {
        "schema": "boxfusion.ca1m_tr3d_e961_outer_dev_eval_preflight.v1",
        "ok": True, "read_only": True, "namespace": NAMESPACE,
        "config_path": str(source), "config_sha256": sha256_file(source),
        "scene_count": 20, "fold0_role": "reused_dev",
        "expanded_training_receipt_access": False,
        "expanded_checkpoint_access": False,
        "anchor_array_access": False, "ground_truth_access": False,
        "gpu_started": False, "fold1_access": False,
        "official_validation_access": False,
        "protocol_preregistration": {
            "path": str(protocol_path), "sha256": sha256_file(protocol_path),
            "sealed_before_any_expanded_outer_checkpoint_access": protocol[
                "sealed_before_any_expanded_outer_checkpoint_access"
            ],
            "sealed_before_any_formal_fold0_anchor_or_gt_access": protocol[
                "sealed_before_any_formal_fold0_anchor_or_gt_access"
            ],
        },
        "preregistration": prereg, "static": static,
        "unique_future_command": (
            "bash scripts/run_ca1m_tr3d_e961_outer_eval_v2.sh "
            "--run <outer_training_run_tag> <gpu_id>"
        ),
    }


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    commands = parser.add_subparsers(dest="command", required=True)
    before = commands.add_parser("preflight")
    before.add_argument("--outer-run-tag")
    seal = commands.add_parser("seal-preregistration")
    seal.add_argument("--outer-run-tag", required=True)
    all_parser = commands.add_parser("all")
    all_parser.add_argument("--outer-run-tag", required=True)
    all_parser.add_argument("--device", default="cuda:0")
    verify = commands.add_parser("verify-inner-authorization")
    verify.add_argument("--role", required=True, choices=INNER_ROLES)
    return parser


def main() -> int:
    args = parser().parse_args()
    if args.command == "preflight":
        result = preflight(args.config, args.outer_run_tag)
    elif args.command == "seal-preregistration":
        path, value = seal_preregistration(args.config, args.outer_run_tag)
        result = {
            "schema": "boxfusion.ca1m_tr3d_e961_preregistration_seal_run.v1",
            "complete": True, "path": str(path), "sha256": sha256_file(path),
            "outer_training_run_tag": value["outer_training_run_tag"],
            "expanded_training_receipt_access": False,
            "expanded_checkpoint_access": False, "ground_truth_access": False,
            "gpu_started": False,
        }
    elif args.command == "all":
        # Ordering is a scientific boundary: do not combine/reorder these calls.
        prereg_path, _ = seal_preregistration(args.config, args.outer_run_tag)
        binding = seal_binding(args.config, args.outer_run_tag)
        proposals = _run_proposals(
            args.config, args.outer_run_tag, args.device
        )
        report = evaluate(args.config, args.outer_run_tag)
        result = {
            "schema": "boxfusion.ca1m_tr3d_e961_outer_dev_eval_all.v1",
            "complete": True,
            "preregistration": {
                "path": str(prereg_path), "sha256": sha256_file(prereg_path),
            },
            "binding_sha256": binding.sha256,
            "proposal_collection": proposals["collection"],
            "continuation_gate": report["continuation_gate"],
            "decision_artifact": report["decision_artifact"],
        }
    elif args.command == "verify-inner-authorization":
        result = validate_inner_authorization(args.role)
    else:  # pragma: no cover
        raise RuntimeError("unreachable command")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
