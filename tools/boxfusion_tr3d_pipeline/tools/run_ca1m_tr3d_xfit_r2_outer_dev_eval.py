#!/usr/bin/env python3
"""Run the sealed CA-only xfit-R2 outer-dev evaluation pipeline.

``all`` is the single post-training command.  It seals the fixed iter-11268
checkpoint and completed MMEngine log, builds an exact fold0 point-only
proposal collection, then performs the CPU-only official-CA and same-GT oracle
diagnostics.  A failed/pending checkpoint stops before a GPU worker exists;
GT is unreachable until the exact 20-scene proposal collection is sealed.
"""

from __future__ import annotations

import argparse
import hashlib
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
    sha256_bytes,
    write_npz_create_only,
)
from boxfusion.ca1m_tr3d_worker_client import CA1MTR3DWorker  # noqa: E402
from boxfusion.ca1m_tr3d_xfit_r2_eval import (  # noqa: E402
    COLLECTION_SCHEMA,
    CONTINUATION_RECEIPT_SCHEMA,
    NAMESPACE,
    OUTER_WRAPPER_LOG,
    PREREGISTRATION_V2_SCHEMA,
    PREREGISTRATION_V3_SCHEMA,
    REPORT_SCHEMA,
    continuation_gate,
    create_or_verify_json,
    evaluation_config_contract_sha256,
    load_binding,
    load_config,
    match_targets,
    metric_delta,
    official_ca_ap,
    preregistration_input_contract,
    read_json,
    regular_directory,
    regular_file,
    same_gt_oracle_scene,
    scene_ids,
    seal_binding,
    sha256_file,
    validate_effective_config,
    validate_outer_wrapper_log,
    validate_static_inputs,
)
from tools.run_ca1m_tr3d_proposal_cache_v4 import _build_scene_points  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/ca1m_tr3d_xfit_r2_outer_dev_eval_v2.json"
ANCHOR_SCHEMA = "boxfusion.ca1m_tr3d_xfit_r2_fold0_final_base_b6_oof.v1"
V1_COMPARISON_SCHEMA = "boxfusion.ca1m_tr3d_xfit_r2_v1_fold0_proposal_comparison.v1"
GT_INVENTORY_SCHEMA = "boxfusion.ca1m_tr3d_benefit_gate_gt_shadow_inventory.v1"
V1_BINDING_SHA256 = "19b8c3d12de8dd8d3ffff1413c6c6003a5ccb1a10cf213b972ebd43fa9db5043"
V1_CHECKPOINT_SHA256 = "d3ba6cc22f0a1a11ab47e55ccdd21c2ef4a84efaf3c6359b7e8231a6c8d3b4a7"
V1_EFFECTIVE_CONFIG_SHA256 = "38368fb5eb692ae2452d098bd4bb0814bbbb83feae780cf83446121bd9e7b88b"
V1_POINT_CONFIG_SHA256 = "60a0e626d671a8b0270006143a062de69ebdd3d9516d5d47c81a6cec2dcd5da4"
V1_CODE_MANIFEST_SHA256 = "88ca894181db4161d54b3b55f58664cd6def01f290aca44ccea1b7fecdfffa9a"
V1_COLLECTION_SHA256 = "a8a9bcbccb8212e6a346b60e3657859f06751b1d2309204919b0de725babc349"
_SHA = re.compile(r"^[0-9a-f]{64}$")


def _scalar(archive: Mapping[str, Any], name: str) -> Any:
    value = np.asarray(archive[name])
    if value.shape != ():
        raise ValueError(f"{name} must be scalar")
    return value.item()


def _bound_record(
    record: Mapping[str, Any], name: str, *, immutable: bool = True,
    schema: str | None = None,
) -> tuple[Path, dict[str, Any] | None]:
    path = regular_file(Path(str(record.get("path", ""))), name, immutable=immutable)
    expected = str(record.get("sha256", ""))
    if _SHA.fullmatch(expected) is None or sha256_file(path) != expected:
        raise ValueError(f"{name} SHA256 differs")
    value: dict[str, Any] | None = None
    if schema is not None:
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"{name} is not JSON") from error
        if not isinstance(parsed, dict) or parsed.get("schema") != schema:
            raise ValueError(f"{name} schema differs")
        value = parsed
    return path, value


def _validate_preregistration(cfg: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    record = (cfg.get("evaluation_stage") or {}).get("preregistration") or {}
    path, value = _bound_record(
        record, "outer-dev preregistration-v3", schema=PREREGISTRATION_V3_SCHEMA
    )
    assert value is not None
    gate = value.get("continuation_gate") or {}
    if (
        value.get("complete") is not True
        or value.get("create_only") is not True
        or value.get("namespace") != NAMESPACE
        or value.get("sealed_before_r2_fold0_gt_access") is not True
        or value.get("r2_fold0_gt_access_at_seal") is not False
        or value.get("r2_proposal_access_at_seal") is not False
        or value.get("gt_array_content_access_at_seal") is not False
        or value.get("gt_inventory_binding_is_opaque_metadata_only") is not True
        or value.get("fold0_role") != "reused_dev"
        or value.get("fold1_access") is not False
        or value.get("official_validation_access") is not False
        or value.get("raw_detector_ap_role") != "diagnostic_only_no_checkpoint_selection"
        or value.get("outer_wrapper_log_contract") != {
            "path": str(OUTER_WRAPPER_LOG),
            "regular_non_symlink_required": True,
            "terminal_line": "TRAIN_EXIT=0",
            "terminal_line_must_be_last_and_unique": True,
            "r2_runner_preamble_required": True,
            "role": "outer_dev",
            "optimizer_updates": 11268,
            "error_markers_forbidden": True,
            "binding_snapshot_required": True,
        }
        or gate != {
            "proposal_exact20_finite_ca_only": True,
            "same_gt_min_iou_gain": 0.05,
            "min_replacements": 10,
            "min_replacement_scenes": 5,
            "min_delta_ap15": 0.0,
            "min_delta_ap25": 0.0,
            "min_delta_ap50": 0.005,
            "pass_authorizes_inner_roles": [
                "inner_holdout2", "inner_holdout3", "inner_holdout4"
            ],
        }
    ):
        raise ValueError("outer-dev preregistration-v3 science contract differs")
    predecessor = value.get("predecessor") or {}
    predecessor_path, predecessor_value = _bound_record(
        predecessor, "outer-dev preregistration-v2", schema=PREREGISTRATION_V2_SCHEMA
    )
    assert predecessor_value is not None
    if (
        sha256_file(predecessor_path)
        != "ac432705669efad65da7337c9f083eeb9e8ac93c7b2da279f77af929c358d347"
        or predecessor_value.get("sealed_before_r2_fold0_gt_access") is not True
        or predecessor_value.get("r2_fold0_gt_access_at_seal") is not False
        or predecessor_value.get("r2_proposal_access_at_seal") is not False
        or predecessor_value.get("continuation_gate") != gate
        or (predecessor_value.get("predecessor") or {}).get("sha256")
        != "f215ed1ef22c0e167911694a2416c949379febce682310b26d2a97208b46b244"
    ):
        raise ValueError("outer-dev preregistration-v2 predecessor differs")
    contract = value.get("evaluation_config_contract") or {}
    if (
        Path(str(contract.get("path", ""))).resolve() != DEFAULT_CONFIG
        or contract.get("normalizer")
        != (
            "canonical_sorted_json_with_evaluation_stage."
            "preregistration_replaced_by_schema_self_marker"
        )
        or contract.get("semantic_sha256")
        != evaluation_config_contract_sha256(cfg)
        or contract.get("binds_complete_implementation_inventory") is not True
        or value.get("implementation") != cfg.get("implementation")
        or value.get("inputs") != preregistration_input_contract(cfg)
    ):
        raise ValueError("outer-dev preregistration-v3 final contract differs")
    return path, value


def _proposal_code_manifest(
    *, config_path: Path, inference_path: Path, binding_path: Path,
    runtime_root: Path, worker_script: Path,
) -> str:
    sources = {
        "xfit_r2_outer_dev_runner": Path(__file__),
        "xfit_r2_eval_contract": ROOT / "boxfusion/ca1m_tr3d_xfit_r2_eval.py",
        "v4_point_builder": ROOT / "tools/run_ca1m_tr3d_proposal_cache_v4.py",
        "v4_proposal_contract": ROOT / "boxfusion/ca1m_tr3d_terminal_v4.py",
        "terminal_geometry": ROOT / "boxfusion/ca1m_tr3d_terminal.py",
        "rgbd_backprojection": ROOT / "boxfusion/tr3d_incremental_online.py",
        "worker_client": ROOT / "boxfusion/ca1m_tr3d_worker_client.py",
        "worker": worker_script,
        "official_adapter": runtime_root / "boxfusion/tr3d_inference.py",
        "point_inference_contract": ROOT / "boxfusion/ca1m_tr3d_inference_contract.py",
        "point_inference_config": inference_path,
        "evaluation_config": config_path,
        "checkpoint_binding": binding_path,
    }
    rows = {
        name: sha256_file(regular_file(path, f"proposal code {name}"))
        for name, path in sorted(sources.items())
    }
    return json.dumps(
        {"schema": "boxfusion.ca1m_tr3d_xfit_r2_proposal_code.v1", "files": rows},
        separators=(",", ":"), sort_keys=True,
    )


def _load_point_parity(cfg: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    lineage = cfg["point_lineage"]
    path, value = read_json(
        Path(lineage["receipt_path"]), "point parity receipt", immutable=True
    )
    if sha256_file(path) != lineage["receipt_sha256"]:
        raise ValueError("point parity receipt SHA256 differs")
    return path, value


def _proposal_provenance(
    config_source: Path, cfg: Mapping[str, Any], binding: Any,
) -> dict[str, Any]:
    inference_record = cfg["point_inference"]
    inference_path = regular_file(
        Path(inference_record["path"]), "xfit-R2 point inference config",
        immutable=True,
    )
    inference_sha = str(inference_record["sha256"])
    validate_ca1m_point_inference_config(
        inference_path=inference_path, inference_sha256=inference_sha,
        effective_training_path=binding.effective_config_snapshot_path,
        effective_training_sha256=binding.effective_config_sha256,
    )
    parity_path, parity = _load_point_parity(cfg)
    runtime = cfg["runtime"]
    worker_script = regular_file(Path(runtime["worker_script"]), "CA TR3D worker")
    runtime_root = regular_directory(Path(runtime["runtime_root"]), "TR3D runtime")
    code_json = _proposal_code_manifest(
        config_path=config_source, inference_path=inference_path,
        binding_path=binding.path, runtime_root=runtime_root,
        worker_script=worker_script,
    )
    return {
        "inference_path": inference_path,
        "inference_sha256": inference_sha,
        "parity_path": parity_path,
        "parity_sha256": sha256_file(parity_path),
        "parity_scenes": parity.get("scenes") or {},
        "worker_script": worker_script,
        "runtime_root": runtime_root,
        "code_json": code_json,
        "code_sha256": sha256_bytes(code_json.encode()),
    }


def preflight(config_path: Path) -> dict[str, Any]:
    source, cfg = load_config(config_path)
    static = validate_static_inputs(cfg)
    scenes = scene_ids(cfg)
    prereg_path, _ = _validate_preregistration(cfg)
    training = cfg["training"]
    root = Path(training["work_root"]).resolve()
    checkpoint = root / training["checkpoint_name"]
    effective = root / training["effective_config_name"]
    effective_status: dict[str, Any] | None = None
    if effective.is_file() and not effective.is_symlink():
        effective_status = validate_effective_config(effective, cfg)
    binding_path = Path(training["binding_path"])
    binding_ready = binding_path.is_file() and not binding_path.is_symlink()
    if binding_ready:
        load_binding(binding_path)
    outer_path = Path(training["outer_wrapper_log_path"])
    outer_status: dict[str, Any] = {
        "path": str(OUTER_WRAPPER_LOG),
        "present": outer_path.is_file() and not outer_path.is_symlink(),
        "complete": False,
        "required_terminal_line": "TRAIN_EXIT=0",
    }
    if outer_status["present"]:
        try:
            outer_status["contract"] = validate_outer_wrapper_log(outer_path)
            outer_status["complete"] = True
        except (FileNotFoundError, UnicodeDecodeError, ValueError) as error:
            outer_status["pending_or_invalid_reason"] = str(error)
    return {
        "schema": "boxfusion.ca1m_tr3d_xfit_r2_outer_dev_preflight.v1",
        "ok": True,
        "read_only": True,
        "namespace": NAMESPACE,
        "config_path": str(source), "config_sha256": sha256_file(source),
        "scene_count": len(scenes), "fold0_only": True,
        "fold0_role": "reused_dev",
        "fold0_prior_exposure": ["v1_checkpoint_diagnostic", "terminal_v4_gate"],
        "fold1_access": False, "official_validation_access": False,
        "ground_truth_access": False, "gpu_started": False,
        "checkpoint_present": checkpoint.is_file() and not checkpoint.is_symlink(),
        "checkpoint_binding_ready": binding_ready,
        "outer_wrapper_log": outer_status,
        "effective_config": effective_status,
        "static_inputs": static,
        "preregistration": {
            "path": str(prereg_path), "sha256": sha256_file(prereg_path),
            "sealed_before_r2_fold0_gt_access": True,
        },
    }


def _require_cuda_device(device: str) -> str:
    if str(device) != "cuda:0":
        raise ValueError(
            "formal R2 proposal stage requires cuda:0 inside the isolated worker"
        )
    return str(device)


def _run_proposals(config_path: Path, device: str) -> dict[str, Any]:
    device = _require_cuda_device(device)
    config_source, cfg = load_config(config_path)
    validate_static_inputs(cfg)
    _validate_preregistration(cfg)
    scenes = scene_ids(cfg)
    binding = load_binding(Path(cfg["training"]["binding_path"]))
    if (
        binding.evaluation_config_path != config_source
        or binding.evaluation_config_sha256 != sha256_file(config_source)
    ):
        raise ValueError("R2 binding/evaluation config differs")
    provenance = _proposal_provenance(config_source, cfg, binding)
    inference_path = provenance["inference_path"]
    inference_sha = provenance["inference_sha256"]
    lineage = cfg["point_lineage"]
    parity_scenes = provenance["parity_scenes"]
    stage = cfg["proposal_stage"]
    protocol = stage["protocol"]
    processed = lineage["processed_rgbd"]
    data_root = regular_directory(Path(processed["root"]), "processed train100 RGB-D")
    output_root = Path(stage["output_root"])
    if output_root.is_symlink():
        raise ValueError("R2 proposal output root must not be a symlink")
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    expected_names = {f"{scene}_ca1m_tr3d_proposals_v4.npz" for scene in scenes}
    actual_names = {
        path.name for path in output_root.iterdir()
        if path.is_file() and not path.is_symlink() and path.suffix == ".npz"
    }
    if actual_names - expected_names:
        raise ValueError("R2 proposal root contains non-fold0 NPZ artifacts")
    runtime = cfg["runtime"]
    worker_python = Path(runtime["worker_python"]).resolve()
    if not worker_python.is_file() or not os.access(worker_python, os.X_OK):
        raise FileNotFoundError(f"missing executable worker Python: {worker_python}")
    worker_script = provenance["worker_script"]
    runtime_root = provenance["runtime_root"]
    project_root = regular_directory(Path(runtime["project_root"]), "TR3D project")
    vendor_root = regular_directory(Path(runtime["vendor_root"]), "TR3D vendor")
    code_json = provenance["code_json"]
    code_sha = provenance["code_sha256"]
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
                or summary.config_sha256 != inference_sha
                or summary.code_manifest_sha256 != code_sha
            ):
                raise ValueError(f"{scene}: resumed R2 proposal provenance differs")
            reports[scene] = {**summary.as_dict(), "resumed": True}
        else:
            pending.append(scene)
    verified_points: dict[str, tuple[Any, Any, Any, np.ndarray]] = {}
    for scene in pending:
        built = _build_scene_points(
            data_root=data_root, scene=scene, processed=processed,
            protocol=protocol,
        )
        point_sha = hashlib.sha256(built[3].tobytes(order="C")).hexdigest()
        if point_sha != (parity_scenes.get(scene) or {}).get("world_point_array_sha256"):
            raise ValueError(f"{scene}: R2 proposal points differ from sealed parity")
        verified_points[scene] = built
        print(f"R2 fold0 CPU point parity | scene={scene}, PASS", flush=True)
    if pending:
        with CA1MTR3DWorker(
            python=str(worker_python), worker_script=str(worker_script),
            runtime_root=str(runtime_root), config=str(inference_path),
            checkpoint=str(binding.checkpoint_path), project_root=str(project_root),
            vendor_root=str(vendor_root),
            startup_timeout_s=float(runtime["startup_timeout_s"]),
            device=str(device),
            extra_args=(
                "--score-threshold", str(protocol["score_threshold"]),
                "--max-proposals", str(protocol["max_proposals"]),
            ),
        ) as worker:
            if worker.adapter_mode != "genuine":
                raise ValueError("formal R2 proposal stage forbids synthetic TR3D")
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
                    scene_id=scene, frame_count=len(rgb),
                    used_frame_count=len(frames), point_count=len(points),
                    candidate_count=len(result.scores),
                    model_runtime_s=float(result.model_runtime_s),
                    source_points_sha256=point_sha,
                    frame_lineage_sha256=sha256_bytes(lineage_json.encode()),
                    checkpoint_binding_sha256=binding.sha256,
                    checkpoint_sha256=binding.checkpoint_sha256,
                    config_sha256=inference_sha,
                    code_manifest_sha256=code_sha,
                    adapter_mode=result.adapter_mode, device=str(device),
                )
                payload = proposal_cache_payload(
                    summary=summary, used_frame_ids=frames,
                    world_to_local=world_to_local,
                    candidate_corners_world=result.corners_world,
                    candidate_scores=result.scores,
                    candidate_point_count=result.point_counts,
                    candidate_boxes_local=result.boxes_local,
                    candidate_labels=result.labels,
                    frame_lineage=lineage_json, code_manifest=code_json,
                )
                target = output_root / f"{scene}_ca1m_tr3d_proposals_v4.npz"
                write_npz_create_only(target, payload)
                reports[scene] = {**summary.as_dict(), "resumed": False}
    collection = _seal_collection(config_path)
    return {
        "schema": "boxfusion.ca1m_tr3d_xfit_r2_outer_dev_proposal_run.v1",
        "complete": True, "namespace": NAMESPACE, "stage": "P",
        "fold0_only": True, "scene_count": 20,
        "ground_truth_access": False, "anchor_access": False, "b6_access": False,
        "resumed_count": sum(bool(row["resumed"]) for row in reports.values()),
        "scenes": reports, "collection": collection,
    }


def _seal_collection(config_path: Path) -> dict[str, Any]:
    config_source, cfg = load_config(config_path)
    validate_static_inputs(cfg)
    _validate_preregistration(cfg)
    scenes = scene_ids(cfg)
    binding = load_binding(Path(cfg["training"]["binding_path"]))
    if (
        binding.evaluation_config_path != config_source
        or binding.evaluation_config_sha256 != sha256_file(config_source)
    ):
        raise ValueError("R2 binding/evaluation config differs")
    provenance = _proposal_provenance(config_source, cfg, binding)
    parity_scenes = provenance["parity_scenes"]
    stage = cfg["proposal_stage"]
    root = regular_directory(Path(stage["output_root"]), "R2 proposal root")
    expected = {f"{scene}_ca1m_tr3d_proposals_v4.npz" for scene in scenes}
    actual = {
        path.name for path in root.iterdir()
        if path.is_file() and not path.is_symlink() and path.suffix == ".npz"
    }
    if actual != expected:
        raise ValueError("R2 proposal collection is not exact fold0-20")
    rows: list[dict[str, Any]] = []
    total = 0
    for scene in scenes:
        path = regular_file(
            root / f"{scene}_ca1m_tr3d_proposals_v4.npz",
            f"R2 proposal {scene}", immutable=True,
        )
        loaded = load_proposal_cache(
            path, expected_scene=scene, expected_binding_sha256=binding.sha256
        )
        summary = loaded["summary"]
        if (
            summary.adapter_mode != "genuine"
            or summary.checkpoint_sha256 != binding.checkpoint_sha256
            or summary.source_points_sha256
            != (parity_scenes.get(scene) or {}).get("world_point_array_sha256")
            or summary.config_sha256 != provenance["inference_sha256"]
            or summary.code_manifest_sha256 != provenance["code_sha256"]
            or summary.device != "cuda:0"
            or not np.isfinite(loaded["candidate_corners_world"]).all()
            or not np.isfinite(loaded["candidate_scores"]).all()
            or np.any(loaded["candidate_labels"] != 0)
        ):
            raise ValueError(f"{scene}: R2 proposal is not finite genuine CA-only")
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
        "train_only": True, "fold0_only": True, "fold_id": 0,
        "scene_count": 20, "candidate_count": total,
        "finite": True, "class_agnostic_labels_zero": True,
        "genuine_ca_only_checkpoint": True,
        "ground_truth_access": False, "anchor_access": False, "b6_access": False,
        "fold1_access": False, "official_validation_access": False,
        "checkpoint_binding": {
            "path": str(binding.path), "sha256": binding.sha256,
            "checkpoint_sha256": binding.checkpoint_sha256,
        },
        "scene_list": cfg["scene_contract"],
        "runtime_config": {
            "path": str(config_source), "sha256": sha256_file(config_source),
        },
        "point_provenance": {
            "receipt_path": str(provenance["parity_path"]),
            "receipt_sha256": provenance["parity_sha256"],
            "point_inference_config_sha256": provenance["inference_sha256"],
            "code_manifest_sha256": provenance["code_sha256"],
            "protocol": cfg["proposal_stage"]["protocol"],
        },
        "scenes": rows,
    }
    output = Path(stage["collection_manifest"])
    create_or_verify_json(output, payload, "R2 fold0 proposal collection")
    return payload


def _load_collection(
    config_path: Path, cfg: Mapping[str, Any], binding: Any,
) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    provenance = _proposal_provenance(config_path, cfg, binding)
    binding_sha = binding.sha256
    checkpoint_sha = binding.checkpoint_sha256
    stage = cfg["proposal_stage"]
    path, value = read_json(
        Path(stage["collection_manifest"]), "R2 proposal collection", immutable=True
    )
    if (
        value.get("schema") != COLLECTION_SCHEMA
        or value.get("complete") is not True
        or value.get("create_only") is not True
        or value.get("namespace") != NAMESPACE
        or value.get("train_only") is not True
        or value.get("scene_count") != 20
        or value.get("fold0_only") is not True
        or value.get("fold_id") != 0
        or value.get("finite") is not True
        or value.get("class_agnostic_labels_zero") is not True
        or value.get("genuine_ca_only_checkpoint") is not True
        or value.get("ground_truth_access") is not False
        or value.get("anchor_access") is not False
        or value.get("b6_access") is not False
        or value.get("fold1_access") is not False
        or value.get("official_validation_access") is not False
        or (value.get("checkpoint_binding") or {}).get("sha256") != binding_sha
        or (value.get("checkpoint_binding") or {}).get("checkpoint_sha256")
        != checkpoint_sha
        or (value.get("runtime_config") or {}).get("sha256")
        != sha256_file(config_path)
        or value.get("point_provenance") != {
            "receipt_path": str(provenance["parity_path"]),
            "receipt_sha256": provenance["parity_sha256"],
            "point_inference_config_sha256": provenance["inference_sha256"],
            "code_manifest_sha256": provenance["code_sha256"],
            "protocol": cfg["proposal_stage"]["protocol"],
        }
    ):
        raise ValueError("R2 proposal collection contract differs")
    scenes = scene_ids(cfg)
    raw_rows = value.get("scenes", [])
    rows = {str(row.get("scene_id")): row for row in raw_rows}
    if len(raw_rows) != 20 or set(rows) != set(scenes) or len(rows) != 20:
        raise ValueError("R2 proposal collection does not cover exact fold0")
    root = Path(stage["output_root"]).resolve()
    candidate_count = 0
    for scene in scenes:
        record = rows[scene]
        proposal = regular_file(Path(record["path"]), f"R2 proposal {scene}", immutable=True)
        if (
            proposal.parent != root
            or proposal.name != f"{scene}_ca1m_tr3d_proposals_v4.npz"
            or sha256_file(proposal) != record.get("sha256")
        ):
            raise ValueError(f"{scene}: R2 proposal collection path/hash differs")
        loaded = load_proposal_cache(
            proposal, expected_scene=scene, expected_binding_sha256=binding_sha
        )
        summary = loaded["summary"]
        if (
            record.get("candidate_count") != summary.candidate_count
            or record.get("point_count") != summary.point_count
            or record.get("source_points_sha256") != summary.source_points_sha256
            or record.get("config_sha256") != summary.config_sha256
            or record.get("code_manifest_sha256") != summary.code_manifest_sha256
            or summary.checkpoint_sha256 != checkpoint_sha
            or summary.source_points_sha256
            != (provenance["parity_scenes"].get(scene) or {}).get(
                "world_point_array_sha256"
            )
            or summary.config_sha256 != provenance["inference_sha256"]
            or summary.code_manifest_sha256 != provenance["code_sha256"]
            or summary.adapter_mode != "genuine"
            or summary.device != "cuda:0"
            or not np.isfinite(loaded["candidate_corners_world"]).all()
            or not np.isfinite(loaded["candidate_scores"]).all()
            or np.any(loaded["candidate_labels"] != 0)
        ):
            raise ValueError(f"{scene}: sealed R2 proposal metadata differs")
        candidate_count += summary.candidate_count
    if value.get("candidate_count") != candidate_count:
        raise ValueError("R2 proposal collection candidate count differs")
    return path, value, rows


def _load_anchor_shadow(
    cfg: Mapping[str, Any], scenes: tuple[str, ...]
) -> tuple[Path, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    record = cfg["evaluation_stage"]["anchor_shadow"]
    path, _ = _bound_record(record, "fold0 anchor shadow")
    manifest_path, manifest = _bound_record(
        cfg["evaluation_stage"]["anchor_shadow_manifest"],
        "fold0 anchor shadow manifest", schema=f"{ANCHOR_SCHEMA}.manifest",
    )
    assert manifest is not None
    if (
        manifest.get("fold0_only") is not True
        or manifest.get("scene_count") != 20
        or manifest.get("row_count") != 1505
        or manifest.get("fold1_access") is not False
        or manifest.get("official_validation_access") is not False
        or manifest.get("ground_truth_access") is not False
        or manifest.get("candidate_access") is not False
        or manifest.get("each_score_model_excludes_scene") is not True
        or (manifest.get("artifact") or {}).get("sha256") != sha256_file(path)
    ):
        raise ValueError("fold0 anchor shadow manifest differs")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "schema", "complete", "train_only", "fold0_only", "fold_id",
            "fold1_access", "official_validation_access", "ground_truth_access",
            "candidate_access", "each_score_model_excludes_scene", "scene_ids",
            "row_indices", "anchor_corners", "b6_oof_scores",
            "source_dataset_sha256", "source_dataset_manifest_sha256",
            "oof_sidecar_sha256", "final_base_manifest_sha256",
            "fold0_scene_list_sha256",
        }
        if set(archive.files) != required:
            raise ValueError("fold0 anchor shadow keys differ")
        if (
            _scalar(archive, "schema") != ANCHOR_SCHEMA
            or bool(_scalar(archive, "complete")) is not True
            or bool(_scalar(archive, "fold0_only")) is not True
            or int(_scalar(archive, "fold_id")) != 0
            or bool(_scalar(archive, "fold1_access")) is not False
            or bool(_scalar(archive, "official_validation_access")) is not False
            or bool(_scalar(archive, "ground_truth_access")) is not False
            or bool(_scalar(archive, "candidate_access")) is not False
        ):
            raise ValueError("fold0 anchor shadow scalar contract differs")
        scene_rows = np.asarray(archive["scene_ids"]).astype(str)
        row_indices = np.asarray(archive["row_indices"], np.int64)
        corners = np.asarray(archive["anchor_corners"], np.float32)
        scores = np.asarray(archive["b6_oof_scores"], np.float32)
    if (
        len(scene_rows) != 1505 or set(scene_rows.tolist()) != set(scenes)
        or corners.shape != (1505, 8, 3) or scores.shape != (1505,)
        or not np.isfinite(corners).all() or not np.isfinite(scores).all()
    ):
        raise ValueError("fold0 anchor shadow arrays differ")
    result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for scene in scenes:
        keep = scene_rows == scene
        rows = row_indices[keep]
        if not np.array_equal(rows, np.arange(len(rows), dtype=np.int64)):
            raise ValueError(f"{scene}: anchor row identity differs")
        result[scene] = (corners[keep], scores[keep], rows)
    return manifest_path, result


def _load_v1_comparison(
    cfg: Mapping[str, Any], scenes: tuple[str, ...]
) -> tuple[Path, dict[str, dict[str, Any]], str]:
    path, value = _bound_record(
        cfg["evaluation_stage"]["v1_fold0_comparison_manifest"],
        "v1 fold0 comparison manifest", schema=V1_COMPARISON_SCHEMA,
    )
    assert value is not None
    if (
        value.get("complete") is not True
        or value.get("comparison_only") is not True
        or value.get("activation_authorized") is not False
        or value.get("fold0_only") is not True
        or value.get("scene_count") != 20
        or value.get("fold1_proposal_artifact_access") is not False
        or value.get("official_validation_access") is not False
        or value.get("ground_truth_access") is not False
        or (value.get("source_manifest") or {}).get("sha256")
        != V1_COLLECTION_SHA256
    ):
        raise ValueError("v1 fold0 comparison isolation differs")
    source_manifest = regular_file(
        Path(str((value.get("source_manifest") or {}).get("path", ""))),
        "sealed v1 proposal collection", immutable=True,
    )
    if (
        source_manifest
        != ROOT / (
            "reports/ca1m_tr3d_terminal_ca_native_train100_v4/"
            "proposal_collection_manifest_v5.json"
        )
        or sha256_file(source_manifest) != V1_COLLECTION_SHA256
    ):
        raise ValueError("v1 source proposal collection differs")
    inference = value.get("point_inference_config") or {}
    inference_path = regular_file(
        Path(str(inference.get("path", ""))), "v1 point inference config"
    )
    if (
        inference_path
        != Path(
            "/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/config/tr3d/"
            "tr3d_ca1m_foreground_point_inference_v1.py"
        )
        or inference.get("sha256") != V1_POINT_CONFIG_SHA256
        or sha256_file(inference_path) != V1_POINT_CONFIG_SHA256
        or inference.get("schema")
        != "boxfusion.tr3d.ca1m_point_inference_config.v1"
        or inference.get("point_input_only") is not True
        or inference.get("standalone") is not True
        or inference.get("ground_truth_access") is not False
        or inference.get("validation_access") is not False
        or inference.get("evaluator_access") is not False
        or inference.get("scannet_config_access") is not False
        or inference.get("checkpoint_binding_sha256") != V1_BINDING_SHA256
        or inference.get("effective_training_config_sha256")
        != V1_EFFECTIVE_CONFIG_SHA256
    ):
        raise ValueError("v1 point-only inference binding differs")
    binding_record = value.get("checkpoint_binding") or {}
    binding_path = regular_file(
        Path(str(binding_record.get("path", ""))),
        "v1 checkpoint binding", immutable=True,
    )
    if (
        binding_path
        != ROOT / (
            "manifests/ca1m_tr3d_terminal_ca_native_train100_v3/"
            "checkpoint_binding.json"
        )
        or binding_record.get("sha256") != V1_BINDING_SHA256
        or sha256_file(binding_path) != V1_BINDING_SHA256
        or binding_record.get("checkpoint_sha256") != V1_CHECKPOINT_SHA256
        or binding_record.get("effective_config_sha256")
        != V1_EFFECTIVE_CONFIG_SHA256
        or binding_record.get("initialization") != "ca1m_random_scratch"
        or binding_record.get("scannet_checkpoint_or_config_allowed") is not False
    ):
        raise ValueError("v1 CA-only checkpoint binding differs")
    _, parity = _load_point_parity(cfg)
    parity_scenes = parity.get("scenes") or {}
    rows = {str(row.get("scene_id")): row for row in value.get("scenes", [])}
    if set(rows) != set(scenes):
        raise ValueError("v1 comparison does not cover exact fold0")
    binding_sha = str(binding_record.get("sha256", ""))
    if _SHA.fullmatch(binding_sha) is None:
        raise ValueError("v1 comparison checkpoint binding SHA256 differs")
    for scene in scenes:
        row = rows[scene]
        proposal = regular_file(Path(row["path"]), f"v1 proposal {scene}", immutable=True)
        if sha256_file(proposal) != row.get("sha256"):
            raise ValueError(f"{scene}: v1 comparison proposal changed")
        loaded = load_proposal_cache(
            proposal, expected_scene=scene, expected_binding_sha256=binding_sha
        )
        summary = loaded["summary"]
        if (
            row.get("candidate_count") != summary.candidate_count
            or row.get("point_count") != summary.point_count
            or row.get("source_points_sha256") != summary.source_points_sha256
            or row.get("code_manifest_sha256") != summary.code_manifest_sha256
            or summary.source_points_sha256
            != (parity_scenes.get(scene) or {}).get("world_point_array_sha256")
            or summary.config_sha256 != V1_POINT_CONFIG_SHA256
            or summary.checkpoint_sha256 != V1_CHECKPOINT_SHA256
            or summary.code_manifest_sha256 != V1_CODE_MANIFEST_SHA256
            or summary.adapter_mode != "genuine"
        ):
            raise ValueError(f"{scene}: v1 same-point-path provenance differs")
    return path, rows, binding_sha


def _load_ground_truth(
    cfg: Mapping[str, Any], scenes: tuple[str, ...]
) -> tuple[Path, dict[str, np.ndarray], dict[str, str]]:
    # This function is called only after _load_collection has sealed/validated
    # every R2 proposal.  It is the sole GT-reachable function in this runner.
    record = cfg["evaluation_stage"]["gt_shadow_inventory"]
    path, inventory = _bound_record(
        record, "fit/dev GT shadow inventory", schema=GT_INVENTORY_SCHEMA,
    )
    assert inventory is not None
    if (
        inventory.get("complete") is not True
        or inventory.get("scene_count") != 80
        or inventory.get("locked_internal_scene_count_accessed") != 0
        or inventory.get("locked_internal_fold_ids") != [1]
        or inventory.get("official_validation_comparable") is not False
    ):
        raise ValueError("GT shadow inventory isolation differs")
    rows = inventory.get("scenes") or {}
    root = Path(cfg["evaluation_stage"]["gt_shadow_root"]).resolve()
    ground_truth: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    for scene in scenes:
        row = rows.get(scene) or {}
        box_record = row.get("box") or {}
        box = regular_file(Path(str(box_record.get("path", ""))), f"GT {scene}", immutable=True)
        if (
            row.get("fold_id") != 0
            or box.parent != root / scene
            or sha256_file(box) != box_record.get("sha256")
        ):
            raise ValueError(f"{scene}: fold0 GT shadow binding differs")
        value = np.asarray(np.load(box, allow_pickle=False), dtype=np.float64)
        if value.ndim != 3 or value.shape[1:] != (8, 3) or not np.isfinite(value).all():
            raise ValueError(f"{scene}: fold0 GT boxes differ")
        ground_truth[scene] = value
        hashes[scene] = sha256_file(box)
    return path, ground_truth, hashes


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


def _metrics(
    predictions: Mapping[str, tuple[np.ndarray, np.ndarray]],
    ground_truth: Mapping[str, np.ndarray], scenes: tuple[str, ...],
) -> dict[str, dict[str, float | int]]:
    scene_rows: list[np.ndarray] = []
    score_rows: list[np.ndarray] = []
    iou_rows: list[np.ndarray] = []
    gt_rows: list[np.ndarray] = []
    for scene in scenes:
        corners, scores = predictions[scene]
        best_iou, best_gt = match_targets(corners, ground_truth[scene])
        scene_rows.append(np.full(len(scores), scene, dtype=np.str_))
        score_rows.append(scores)
        iou_rows.append(best_iou)
        gt_rows.append(best_gt)
    return official_ca_ap(
        scene_ids=np.concatenate(scene_rows), scores=np.concatenate(score_rows),
        best_iou=np.concatenate(iou_rows), best_gt=np.concatenate(gt_rows),
        ground_truth_count=sum(len(ground_truth[scene]) for scene in scenes),
    )


def _same_gt_oracle(
    *, anchors: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    candidates: Mapping[str, tuple[np.ndarray, np.ndarray]],
    ground_truth: Mapping[str, np.ndarray], scenes: tuple[str, ...],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    per_scene: dict[str, Any] = {}
    for scene in scenes:
        anchor_corners, anchor_scores, _ = anchors[scene]
        candidate_corners, candidate_scores = candidates[scene]
        corners, summary = same_gt_oracle_scene(
            anchor_corners=anchor_corners, anchor_scores=anchor_scores,
            candidate_corners=candidate_corners, candidate_scores=candidate_scores,
            gt_corners=ground_truth[scene], near_iou=0.15, min_gain=0.05,
        )
        predictions[scene] = (corners, anchor_scores)
        per_scene[scene] = summary
    return predictions, {
        "min_same_gt_iou_gain": 0.05,
        "replacement_count": sum(
            row["selected_replacement_count"] for row in per_scene.values()
        ),
        "replacement_scene_count": sum(
            row["selected_replacement_count"] > 0 for row in per_scene.values()
        ),
        "positive_iou_gain_sum": sum(
            row["positive_iou_gain_sum"] for row in per_scene.values()
        ),
        "oracle_deployable": False,
        "per_scene": per_scene,
    }


def evaluate(config_path: Path) -> dict[str, Any]:
    config_source, cfg = load_config(config_path)
    validate_static_inputs(cfg)
    scenes = scene_ids(cfg)
    binding = load_binding(Path(cfg["training"]["binding_path"]))
    if (
        binding.evaluation_config_path != config_source
        or binding.evaluation_config_sha256 != sha256_file(config_source)
    ):
        raise ValueError("R2 binding/evaluation config differs")
    prereg_path, prereg = _validate_preregistration(cfg)
    # Exact R2 collection seal is validated before any GT inventory or array.
    collection_path, collection, r2_rows = _load_collection(
        config_source, cfg, binding
    )
    anchor_manifest, anchors = _load_anchor_shadow(cfg, scenes)
    v1_manifest, v1_rows, v1_binding_sha = _load_v1_comparison(cfg, scenes)
    old_receipt_path, old_receipt = _bound_record(
        cfg["evaluation_stage"]["v1_sealed_raw_diagnostic"],
        "sealed v1 raw diagnostic",
        schema="boxfusion.tr3d.ca1m_checkpoint_dev_diagnostic_receipt.v1",
    )
    assert old_receipt is not None
    if (
        old_receipt.get("partition") != "threshold_dev_fold0"
        or old_receipt.get("scene_count") != 20
        or (old_receipt.get("authorization") or {}).get("diagnostic_only") is not True
        or (old_receipt.get("authorization") or {}).get("checkpoint_selection_authorized") is not False
    ):
        raise ValueError("sealed v1 raw diagnostic role differs")

    # First/only R2 fold0 GT reachability point.
    gt_inventory_path, ground_truth, gt_hashes = _load_ground_truth(cfg, scenes)
    r2_candidates = _proposal_arrays(r2_rows, scenes, binding.sha256)
    v1_candidates = _proposal_arrays(v1_rows, scenes, v1_binding_sha)
    baseline_predictions = {
        scene: (anchors[scene][0], anchors[scene][1]) for scene in scenes
    }
    baseline_metrics = _metrics(baseline_predictions, ground_truth, scenes)
    raw_r2_metrics = _metrics(r2_candidates, ground_truth, scenes)
    raw_v1_same_path_metrics = _metrics(v1_candidates, ground_truth, scenes)
    r2_oracle_predictions, r2_oracle_summary = _same_gt_oracle(
        anchors=anchors, candidates=r2_candidates,
        ground_truth=ground_truth, scenes=scenes,
    )
    v1_oracle_predictions, v1_oracle_summary = _same_gt_oracle(
        anchors=anchors, candidates=v1_candidates,
        ground_truth=ground_truth, scenes=scenes,
    )
    r2_oracle_metrics = _metrics(r2_oracle_predictions, ground_truth, scenes)
    v1_oracle_metrics = _metrics(v1_oracle_predictions, ground_truth, scenes)
    r2_oracle_delta = metric_delta(r2_oracle_metrics, baseline_metrics)
    v1_oracle_delta = metric_delta(v1_oracle_metrics, baseline_metrics)
    gate = continuation_gate(
        proposal_integrity_pass=(
            collection.get("scene_count") == 20
            and collection.get("finite") is True
            and collection.get("class_agnostic_labels_zero") is True
            and collection.get("genuine_ca_only_checkpoint") is True
        ),
        scene_count=20,
        replacement_count=r2_oracle_summary["replacement_count"],
        replacement_scene_count=r2_oracle_summary["replacement_scene_count"],
        oracle_ap_delta=r2_oracle_delta,
    )
    sealed_v1_ap = old_receipt.get("ap") or {}
    report = {
        "schema": REPORT_SCHEMA,
        "complete": True, "create_only": True, "namespace": NAMESPACE,
        "train_only": True, "partition": "threshold_dev_fold0",
        "fold0_role": "reused_dev",
        "fold0_prior_exposure": ["v1_checkpoint_diagnostic", "terminal_v4_gate"],
        "official_validation_comparable": False,
        "fold1_access": False, "official_validation_access": False,
        "scene_count": 20,
        "checkpoint_selection": False,
        "raw_detector_ap_role": "diagnostic_only_no_checkpoint_selection",
        "raw_detector": {
            "r2": raw_r2_metrics,
            "v1_same_point_path": raw_v1_same_path_metrics,
            "r2_minus_v1_same_point_path": metric_delta(
                raw_r2_metrics, raw_v1_same_path_metrics
            ),
            "sealed_v1_diagnostic_reference": {
                "ap": sealed_v1_ap,
                "prediction_count": old_receipt.get("prediction_count"),
                "source_receipt_path": str(old_receipt_path),
                "source_receipt_sha256": sha256_file(old_receipt_path),
                "r2_minus_reference_ap": {
                    "iou_0.15": float(raw_r2_metrics["iou_0.15"]["ap"] - sealed_v1_ap["ap15"]),
                    "iou_0.25": float(raw_r2_metrics["iou_0.25"]["ap"] - sealed_v1_ap["ap25"]),
                    "iou_0.50": float(raw_r2_metrics["iou_0.50"]["ap"] - sealed_v1_ap["ap50"]),
                },
            },
            "checkpoint_selection_authorized": False,
        },
        "final_base_b6_v2_same_gt_headroom": {
            "anchor_score_source": "CA-native B6-v2 fold0 OOF; each model excludes scene",
            "baseline": baseline_metrics,
            "r2_oracle": r2_oracle_metrics,
            "r2_oracle_ap_delta": r2_oracle_delta,
            "r2_oracle_geometry": r2_oracle_summary,
            "v1_same_path_oracle": v1_oracle_metrics,
            "v1_same_path_oracle_ap_delta": v1_oracle_delta,
            "v1_same_path_oracle_geometry": v1_oracle_summary,
            "oracle_deployable": False,
        },
        "continuation_gate": gate,
        "inner_training_authorization": {
            "authorized": gate["continue_inner_training_authorized"],
            "roles": gate["authorized_inner_roles"],
            "source": "preregistered outer-dev continuation gate only",
        },
        "preregistration": {
            "path": str(prereg_path), "sha256": sha256_file(prereg_path),
            "sealed_before_r2_fold0_gt_access": prereg["sealed_before_r2_fold0_gt_access"],
        },
        "provenance": {
            "evaluation_config": {"path": str(config_source), "sha256": sha256_file(config_source)},
            "checkpoint_binding": {"path": str(binding.path), "sha256": binding.sha256},
            "r2_proposal_collection": {"path": str(collection_path), "sha256": sha256_file(collection_path)},
            "fold0_anchor_shadow_manifest": {"path": str(anchor_manifest), "sha256": sha256_file(anchor_manifest)},
            "v1_fold0_comparison_manifest": {"path": str(v1_manifest), "sha256": sha256_file(v1_manifest)},
            "gt_shadow_inventory": {"path": str(gt_inventory_path), "sha256": sha256_file(gt_inventory_path)},
            "fold0_gt_hashes": gt_hashes,
        },
    }
    output = Path(cfg["evaluation_stage"]["report"])
    report_path = create_or_verify_json(
        output, report, "xfit-R2 outer-dev evaluation report"
    )
    receipt = {
        "schema": CONTINUATION_RECEIPT_SCHEMA,
        "complete": True, "create_only": True, "train_only": True,
        "partition": "threshold_dev_fold0", "fold0_role": "reused_dev",
        "scene_count": 20, "fold1_access": False,
        "official_validation_access": False,
        "raw_detector_ap_role": "diagnostic_only_no_checkpoint_selection",
        "checkpoint_selection": False,
        "checkpoint_binding": {
            "path": str(binding.path), "sha256": binding.sha256,
            "checkpoint_sha256": binding.checkpoint_sha256,
        },
        "proposal_collection": {
            "path": str(collection_path), "sha256": sha256_file(collection_path),
            "exact20": collection["scene_count"] == 20,
            "scene_count": collection["scene_count"],
            "finite": collection["finite"],
            "class_agnostic_labels_zero": collection["class_agnostic_labels_zero"],
            "genuine_ca_only_checkpoint": collection["genuine_ca_only_checkpoint"],
        },
        "preregistration": {
            "path": str(prereg_path), "sha256": sha256_file(prereg_path),
            "sealed_before_r2_fold0_gt_access": True,
        },
        "evaluation_report": {
            "path": str(report_path), "sha256": sha256_file(report_path),
            "schema": REPORT_SCHEMA,
        },
        "continuation_gate": gate,
        "pass": gate["pass"],
        "continue_inner_training_authorized": gate["continue_inner_training_authorized"],
        "authorized_inner_roles": gate["authorized_inner_roles"],
        "failure_action": gate["failure_action"],
    }
    create_or_verify_json(
        Path(cfg["evaluation_stage"]["continuation_receipt"]), receipt,
        "xfit-R2 outer-dev continuation receipt",
    )
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight")
    commands.add_parser("seal")
    proposal = commands.add_parser("proposals")
    proposal.add_argument("--device", default="cuda:0")
    commands.add_parser("evaluate")
    all_parser = commands.add_parser("all")
    all_parser.add_argument("--device", default="cuda:0")
    return value


def main() -> int:
    args = parser().parse_args()
    if args.command == "preflight":
        result = preflight(args.config)
    elif args.command == "seal":
        binding = seal_binding(args.config)
        result = {
            "schema": "boxfusion.ca1m_tr3d_xfit_r2_outer_dev_seal_run.v1",
            "complete": True, "binding_path": str(binding.path),
            "binding_sha256": binding.sha256,
            "checkpoint_sha256": binding.checkpoint_sha256,
            "gpu_started": False, "ground_truth_access": False,
        }
    elif args.command == "proposals":
        result = _run_proposals(args.config, args.device)
    elif args.command == "evaluate":
        result = evaluate(args.config)
    elif args.command == "all":
        binding = seal_binding(args.config)
        proposal = _run_proposals(args.config, args.device)
        report = evaluate(args.config)
        result = {
            "schema": "boxfusion.ca1m_tr3d_xfit_r2_outer_dev_all.v1",
            "complete": True,
            "binding_sha256": binding.sha256,
            "proposal_collection": proposal["collection"],
            "report": report,
        }
    else:  # pragma: no cover
        raise RuntimeError("unreachable command")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
