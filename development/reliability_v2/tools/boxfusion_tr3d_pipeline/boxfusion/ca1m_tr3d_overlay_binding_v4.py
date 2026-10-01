"""Authorization contract for the independently bound terminal-v4 Stage O."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np


BINDING_SCHEMA = "boxfusion.ca1m_tr3d_terminal_overlay_binding.v4"
AUTHORIZATION_SCHEMA = "boxfusion.ca1m_tr3d_terminal_overlay_authorization.v4"
PROPOSAL_COLLECTION_SCHEMA = "boxfusion.ca1m_tr3d_proposal_collection.v4"
SCENE_LIST_SHA256 = "35321e9942dc5d512db2952b9ca6228b1291127e0c13fd92aa458f2d7eb7f9fd"
FAILED_O1_CONFIG_SHA256 = "09ddf29a80f535af8ee3db811c637f82ad83eb624d56a1361be47cc2b80944ae"
FAILED_O1_AUTHORIZATION_SHA256 = "7eedda92b4d42a87906a74d3e74fd3d76616dd63e7d98bcb177948821c807446"
STAGE_P_RUNTIME_CONFIG_SHA256 = (
    "51334fdd27a68ebb916a61bcb78286efe169a36f82144850d3a1e1238afc8c72"
)
_SCENE = re.compile(r"^[0-9]{8}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, name: str, *, immutable: bool = True) -> Path:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink: {path}")
    source = path.resolve()
    if not source.is_file() or source.is_symlink() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"missing {name}: {source}")
    if immutable and source.stat().st_mode & 0o222:
        raise ValueError(f"{name} must be read-only: {source}")
    return source


def _json(path: Path, name: str) -> tuple[Path, dict[str, Any]]:
    source = _regular_file(path, name)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return source, value


def _record(
    record: Any,
    name: str,
    *,
    schema: str | None = None,
    immutable: bool = True,
) -> Path:
    if not isinstance(record, Mapping) or set(record) != {
        "path", "sha256", *(set() if schema is None else {"schema"})
    }:
        raise ValueError(f"{name} binding fields differ")
    path = _regular_file(Path(str(record["path"])), name, immutable=immutable)
    if record["sha256"] != sha256_file(path):
        raise ValueError(f"{name} SHA256 differs")
    if schema is not None and record["schema"] != schema:
        raise ValueError(f"{name} schema binding differs")
    if schema is not None:
        if path.suffix == ".json":
            _, payload = _json(path, name)
            actual_schema = payload.get("schema")
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                if "schema" not in archive.files:
                    raise ValueError(f"{name} has no internal schema")
                value = np.asarray(archive["schema"])
                if value.shape != ():
                    raise ValueError(f"{name} internal schema is not scalar")
                actual_schema = value.item()
        else:
            raise ValueError(f"{name} schema cannot be verified from {path.suffix}")
        if actual_schema != schema:
            raise ValueError(f"{name} internal schema differs")
    return path


def validate_proposal_collection(
    record: Any, cfg: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate the sealed exact100 P manifest and every immutable cache row."""

    source = _record(
        record, "Stage-P proposal collection", schema=PROPOSAL_COLLECTION_SCHEMA
    )
    _, value = _json(source, "Stage-P proposal collection")
    for key, expected in {
        "schema": PROPOSAL_COLLECTION_SCHEMA,
        "complete": True,
        "create_only": True,
        "stage": "P",
        "scene_count": 100,
        "ground_truth_access": False,
        "validation_ground_truth_access": False,
        "anchor_access": False,
        "b6_access": False,
        "gpu_started_by_manifest_sealer": False,
    }.items():
        if value.get(key) != expected:
            raise ValueError(f"Stage-P proposal collection field {key} differs")
    scene_contract = cfg.get("scene_contract") or {}
    scene_path = _regular_file(
        Path(str(scene_contract.get("path", ""))),
        "exact train100 scene list",
        immutable=False,
    )
    scenes = tuple(row.strip() for row in scene_path.read_text().splitlines() if row.strip())
    if (
        len(scenes) != 100
        or len(set(scenes)) != 100
        or any(_SCENE.fullmatch(scene) is None for scene in scenes)
        or scene_contract.get("sha256") != SCENE_LIST_SHA256
        or sha256_file(scene_path) != SCENE_LIST_SHA256
        or (value.get("scene_list") or {}).get("sha256") != SCENE_LIST_SHA256
    ):
        raise ValueError("Stage-P proposal collection scene contract differs")
    rows = value.get("scenes")
    if not isinstance(rows, list) or len(rows) != 100:
        raise ValueError("Stage-P proposal collection is not exact100")
    by_scene: dict[str, dict[str, Any]] = {}
    proposal_root = Path(
        str((cfg.get("overlay_stage") or {}).get("proposal_cache_root", ""))
    ).resolve()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Stage-P proposal collection row is not an object")
        scene = str(row.get("scene_id", ""))
        if scene not in scenes or scene in by_scene:
            raise ValueError("Stage-P proposal collection scene rows differ")
        path = _regular_file(
            Path(str(row.get("path", ""))), f"Stage-P proposal cache {scene}"
        )
        if (
            path.parent != proposal_root
            or path.name != f"{scene}_ca1m_tr3d_proposals_v4.npz"
            or row.get("sha256") != sha256_file(path)
        ):
            raise ValueError(f"{scene}: Stage-P manifest/cache authority differs")
        by_scene[scene] = dict(row)
    if set(by_scene) != set(scenes):
        raise ValueError("Stage-P proposal collection does not cover exact train100")
    runtime = value.get("runtime_config") or {}
    runtime_path = _regular_file(
        Path(str(runtime.get("path", ""))),
        "sealed Stage-P runtime config",
        immutable=False,
    )
    stage_p_authorization = value.get("stage_p_authorization") or {}
    configured_authorization = cfg.get("proposal_stage") or {}
    if (
        runtime_path.name != "ca1m_tr3d_terminal_train100_v4_p5.json"
        or runtime.get("sha256") != STAGE_P_RUNTIME_CONFIG_SHA256
        or sha256_file(runtime_path) != STAGE_P_RUNTIME_CONFIG_SHA256
        or Path(str(stage_p_authorization.get("path", ""))).resolve()
        != Path(str(configured_authorization.get("authorization_receipt", ""))).resolve()
        or stage_p_authorization.get("sha256")
        != configured_authorization.get("authorization_receipt_sha256")
    ):
        raise ValueError("Stage-P manifest runtime-config/authorization binding differs")
    authorization_path = _regular_file(
        Path(str(stage_p_authorization.get("path", ""))),
        "sealed Stage-P authorization",
    )
    if sha256_file(authorization_path) != stage_p_authorization.get("sha256"):
        raise ValueError("Stage-P manifest authorization SHA256 differs")
    for manifest_name, config_name in (
        ("checkpoint_binding", "ca_native_tr3d_binding"),
        ("point_inference_config", "ca_native_tr3d_inference"),
    ):
        if value.get(manifest_name) != cfg.get(config_name):
            raise ValueError(f"Stage-P manifest {manifest_name} binding differs")
    return source, value, by_scene


def validate_overlay_authorization(
    config_path: Path, cfg: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate config-to-auth-to-upstream/code bindings before Stage O I/O."""

    config = _regular_file(config_path, "overlay-bound config")
    binding = cfg.get("stage_o_binding")
    if not isinstance(binding, Mapping):
        raise PermissionError("Stage O requires an independent sealed binding")
    expected_keys = {
        "schema", "revision", "authorization_path", "proposal_collection",
        "final_base_manifest", "native_b6_v2_collection_manifest",
        "native_b6_v2_deployment_checkpoint",
        "native_b6_v2_deployment_checkpoint_manifest",
        "native_b6_v2_oof_row_scores",
        "native_b6_v2_oof_row_scores_manifest", "score_usage",
        "cpu_only", "ground_truth_access", "validation_ground_truth_access",
    }
    if set(binding) != expected_keys:
        raise ValueError("Stage-O binding key set differs")
    if (
        binding["schema"] != BINDING_SCHEMA
        or binding["revision"] != 2
        or binding["cpu_only"] is not True
        or binding["ground_truth_access"] is not False
        or binding["validation_ground_truth_access"] is not False
    ):
        raise ValueError("Stage-O isolation binding differs")
    score_usage = binding.get("score_usage")
    if score_usage != {
        "overlay_anchor_scores": "deployable_ca1m_native_b6_v2_checkpoint",
        "deployment_scores_allowed_for_overlay": True,
        "deployment_scores_allowed_for_stacked_gate_training": False,
        "stacked_gate_training_score_source": "all_fold_oof_row_scores_v2",
        "oof_sidecar_loaded_by_overlay": False,
    }:
        raise ValueError("Stage-O deploy/OOF score separation differs")
    authorization_path, authorization = _json(
        Path(str(binding["authorization_path"])), "Stage-O authorization"
    )
    if authorization_path.stat().st_mode & 0o222:
        raise ValueError("Stage-O authorization must be read-only")
    for key, expected in {
        "schema": AUTHORIZATION_SCHEMA,
        "revision": 2,
        "complete": True,
        "create_only": True,
        "authorization_decision": "ALLOW_STAGE_O_ONLY",
        "overlay_cpu_execution_authorized": True,
        "proposal_gpu_execution_authorized": False,
        "full_two_stage_run_authorized": False,
        "ground_truth_access": False,
        "validation_ground_truth_access": False,
        "deployment_scores_used_for_overlay": True,
        "deployment_scores_allowed_for_stacked_gate_training": False,
        "oof_scores_required_for_stacked_gate_training": True,
        "oof_sidecar_loaded_by_overlay": False,
    }.items():
        if authorization.get(key) != expected:
            raise ValueError(f"Stage-O authorization field {key} differs")
    supersedes = authorization.get("supersedes")
    if not isinstance(supersedes, Mapping):
        raise ValueError("Stage-O revision 2 lacks failed-revision supersession")
    failed_config = _record(
        supersedes.get("failed_config"),
        "failed Stage-O revision-1 config",
        schema="boxfusion.ca1m_tr3d_terminal_two_stage_config.v4",
    )
    failed_authorization = _record(
        supersedes.get("failed_authorization"),
        "failed Stage-O revision-1 authorization",
        schema=AUTHORIZATION_SCHEMA,
    )
    if (
        sha256_file(failed_config) != FAILED_O1_CONFIG_SHA256
        or sha256_file(failed_authorization) != FAILED_O1_AUTHORIZATION_SHA256
        or supersedes.get("reason")
        != "revision1_postflight_rejected_order_sensitive_protocol_validation"
        or supersedes.get("stage_o_execution_started") is not False
        or supersedes.get("overlay_artifact_count") != 0
        or supersedes.get("old_artifacts_overwritten") is not False
    ):
        raise ValueError("Stage-O failed-revision supersession differs")
    bound_config = authorization.get("bound_config") or {}
    if (
        Path(str(bound_config.get("path", ""))).resolve() != config
        or bound_config.get("sha256") != sha256_file(config)
    ):
        raise ValueError("Stage-O authorization/config binding differs")
    record_schemas = {
        "proposal_collection": PROPOSAL_COLLECTION_SCHEMA,
        "final_base_manifest": "boxfusion.ca1m_final_base_identity_audit.v1",
        "native_b6_v2_collection_manifest":
            "boxfusion.ca1m_native_b6_final_base_train_collection.v2",
        "native_b6_v2_deployment_checkpoint":
            "boxfusion.ca1m_native_b6_iou_mlp.v1",
        "native_b6_v2_deployment_checkpoint_manifest":
            "boxfusion.ca1m_native_b6_checkpoint_manifest.v1",
        "native_b6_v2_oof_row_scores":
            "boxfusion.ca1m_native_b6_oof_row_scores.v2",
        "native_b6_v2_oof_row_scores_manifest":
            "boxfusion.ca1m_native_b6_oof_row_scores_manifest.v2",
    }
    proposal_rows: dict[str, dict[str, Any]] | None = None
    for name, schema in record_schemas.items():
        bound_record = binding.get(name)
        authorized_record = (authorization.get("upstream") or {}).get(name)
        if bound_record != authorized_record:
            raise ValueError(f"Stage-O config/auth upstream {name} differs")
        if name == "proposal_collection":
            _, _, proposal_rows = validate_proposal_collection(bound_record, cfg)
        else:
            _record(bound_record, name, schema=schema)
    code = authorization.get("code") or {}
    if not isinstance(code, Mapping) or not code:
        raise ValueError("Stage-O authorization has no code manifest")
    for name, record in code.items():
        _record(record, f"Stage-O code {name}", immutable=False)
    overlay = cfg.get("overlay_stage") or {}
    field_records = {
        "final_anchor_manifest": "final_base_manifest",
        "native_b6_v2_collection_manifest": "native_b6_v2_collection_manifest",
        "native_b6_v2_checkpoint": "native_b6_v2_deployment_checkpoint",
        "native_b6_v2_checkpoint_manifest":
            "native_b6_v2_deployment_checkpoint_manifest",
    }
    for field, record_name in field_records.items():
        record = binding[record_name]
        if (
            Path(str(overlay.get(field, ""))).resolve()
            != Path(str(record["path"])).resolve()
            or overlay.get(f"{field}_sha256") != record["sha256"]
        ):
            raise ValueError(f"Stage-O overlay field {field} differs from authorization")
    if proposal_rows is None:
        raise RuntimeError("Stage-P proposal authority was not validated")
    return {
        "authorization_path": str(authorization_path),
        "authorization_sha256": sha256_file(authorization_path),
        "bound_config_sha256": sha256_file(config),
        "deployment_scores_used_for_overlay": True,
        "deployment_scores_allowed_for_stacked_gate_training": False,
        "oof_scores_required_for_stacked_gate_training": True,
        "oof_sidecar_loaded_by_overlay": False,
        "oof_usage": "provenance_binding_only",
        "oof_scores_consumed_by_overlay": False,
        "proposal_collection": {
            "path": str(Path(str(binding["proposal_collection"]["path"])).resolve()),
            "sha256": binding["proposal_collection"]["sha256"],
            "scene_count": 100,
        },
        "proposal_rows": proposal_rows,
    }


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "BINDING_SCHEMA",
    "PROPOSAL_COLLECTION_SCHEMA",
    "STAGE_P_RUNTIME_CONFIG_SHA256",
    "sha256_file",
    "validate_proposal_collection",
    "validate_overlay_authorization",
]
