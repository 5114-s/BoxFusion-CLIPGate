"""Executable, fail-closed E961 terminal-input collection (v5 R2).

Importing this module is side-effect free.  The checked-in configuration is a
pending static contract.  Operational functions become reachable only from a
separately sealed ready revision whose four canonical producer receipts pass
their producer-owned deep verifiers and whose authorization binds those exact
receipts.  No function in this module opens ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import re
import stat
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/ca1m_tr3d_e961_terminal_inputs_v5_r2_pending.json"
PENDING_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_pending_config.v2"
READY_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_ready_config.v2"
AUTH_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_run_authorization.v2"
NAMESPACE = "ca1m_tr3d_e961_terminal_inputs_xfit_r2_v5_r2"
OUTER_SCHEMA = "boxfusion.tr3d.ca1m_e961_outer_train_run.r2"
INNER_SCHEMA = "boxfusion.tr3d.ca1m_e961_inner_train_run.r2"
LEGACY_INNER_SCHEMA = "boxfusion.tr3d.ca1m_xfit_r2_inner_run_receipt.v1"
CONTINUATION_SCHEMA = "boxfusion.ca1m_tr3d_e961_outer_dev_continuation_receipt.v1"
PROPOSAL_SCHEMA = "boxfusion.ca1m_tr3d_e961_anchor_free_proposal.v5.r2"
PROPOSAL_COLLECTION_SCHEMA = f"{PROPOSAL_SCHEMA}.collection"
OVERLAY_SCHEMA = "boxfusion.ca1m_tr3d_e961_oof_overlay.v5.r2"
OVERLAY_COLLECTION_SCHEMA = f"{OVERLAY_SCHEMA}.collection"
ADAPTER_SCHEMA = "boxfusion.ca1m_tr3d_e961_verified_receipt_adapter.v2"

ROLE_ORDER = ("outer_dev", "inner_holdout2", "inner_holdout3", "inner_holdout4")
ROLE_SPECS = {
    "outer_dev": ((2, 3, 4), 0, 0),
    "inner_holdout2": ((3, 4), 2, 1),
    "inner_holdout3": ((2, 4), 3, 2),
    "inner_holdout4": ((2, 3), 4, 3),
}
OPERATIONAL_AUTHORIZATIONS = {
    "gpu_proposal_collection": True,
    "cpu_oof_overlay": True,
    "candidate_native_evidence": True,
    "exact80_manifest": True,
    "ground_truth_join": False,
    "gate_fit": False,
    "fold0_diagnostic": False,
    "fold1_internal_check": False,
    "official_validation": False,
    "policy_activation": False,
}

_SHA = re.compile(r"^[0-9a-f]{64}$")
_SCENE = re.compile(r"^[0-9]{8}$")


class PendingOperationalInputs(PermissionError):
    """The checked-in contract intentionally has no operational authority."""


def _identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _no_symlink_chain(path: Path, name: str, *, missing_tail: bool = False) -> None:
    path = Path(path)
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            if missing_tail:
                return
            raise
        if stat.S_ISLNK(mode):
            raise ValueError(f"{name} contains symlink: {current}")


def stable_bytes(path: Path, name: str, *, nonempty: bool = True) -> bytes:
    path = Path(path)
    _no_symlink_chain(path, name)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or (nonempty and before.st_size < 1):
            raise ValueError(f"{name} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    current = os.stat(path, follow_symlinks=False)
    if _identity(before) != _identity(after) or _identity(after) != _identity(current):
        raise ValueError(f"{name} inode/content changed while read")
    result = b"".join(chunks)
    if len(result) != before.st_size:
        raise ValueError(f"{name} changed size while read")
    return result


def sha256_file(path: Path) -> str:
    return hashlib.sha256(stable_bytes(Path(path), "SHA256 input", nonempty=False)).hexdigest()


def _sha(value: Any, name: str) -> str:
    result = str(value)
    if _SHA.fullmatch(result) is None:
        raise ValueError(f"{name} is not lowercase SHA256")
    return result


def _json(path: Path, name: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).absolute()
    try:
        value = json.loads(stable_bytes(source, name).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return source, value


def _record(record: Any, name: str, *, schema: str | None = None) -> tuple[Path, dict[str, Any]]:
    if not isinstance(record, Mapping):
        raise ValueError(f"{name} record must be an object")
    source, value = _json(Path(str(record.get("path", ""))), name)
    if sha256_file(source) != _sha(record.get("sha256"), f"{name} SHA256"):
        raise ValueError(f"{name} SHA256 differs")
    if schema is not None and (record.get("schema") != schema or value.get("schema") != schema):
        raise ValueError(f"{name} schema differs")
    return source, value


def _file_record(record: Any, name: str) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError(f"{name} record must be an object")
    path = Path(str(record.get("path", ""))).absolute()
    if sha256_file(path) != _sha(record.get("sha256"), f"{name} SHA256"):
        raise ValueError(f"{name} SHA256 differs")
    return path


def _scene_list(record: Any, name: str, count: int) -> tuple[str, ...]:
    path = _file_record(record, name)
    rows = tuple(row.strip() for row in stable_bytes(path, name).decode().splitlines() if row.strip())
    if len(rows) != count or len(set(rows)) != count or any(_SCENE.fullmatch(x) is None for x in rows):
        raise ValueError(f"{name} is not exact{count} unique scenes")
    return rows


def _pending(record: Any, schema: str, name: str) -> None:
    if record != {"state": "pending", "path": None, "sha256": None, "schema": schema}:
        raise ValueError(f"{name} is not an unbound pending record")


def _guard_output(path: Path, namespace_root: Path, name: str) -> Path:
    path, namespace_root = Path(path), Path(namespace_root)
    if not path.is_absolute() or not namespace_root.is_absolute():
        raise ValueError(f"{name} must be absolute")
    try:
        path.relative_to(namespace_root)
    except ValueError as error:
        raise ValueError(f"{name} escapes namespace") from error
    _no_symlink_chain(path, name, missing_tail=True)
    return path


def _load_config(path: Path) -> tuple[Path, dict[str, Any]]:
    source, cfg = _json(path, "E961 terminal-input config")
    if cfg.get("namespace") != NAMESPACE:
        raise ValueError("config namespace differs")
    if cfg.get("schema") not in {PENDING_SCHEMA, READY_SCHEMA}:
        raise ValueError("config schema differs")
    return source, cfg


def _validate_static_common(cfg: Mapping[str, Any]) -> dict[str, Any]:
    producers = cfg.get("producer_contracts") or {}
    outer = producers.get("outer") or {}
    inner = producers.get("inner") or {}
    if (
        outer.get("receipt_schema") != OUTER_SCHEMA
        or inner.get("receipt_schema") != INNER_SCHEMA
        or inner.get("rejected_legacy_schema") != LEGACY_INNER_SCHEMA
    ):
        raise ValueError("producer schema contract differs")
    outer_verifier = _file_record(outer.get("verifier"), "outer producer verifier")
    inner_verifier = _file_record(inner.get("verifier"), "inner producer verifier")
    _file_record(inner.get("static_queue_manifest"), "inner static queue manifest")
    dependencies = cfg.get("implementation_dependencies") or {}
    if len(dependencies) != 12:
        raise ValueError("implementation dependency inventory differs")
    for dependency, record in dependencies.items():
        _file_record(record, f"implementation dependency {dependency}")
    runtime = cfg.get("runtime") or {}
    worker_python = Path(str(runtime.get("worker_python", "")))
    if not worker_python.resolve().is_file() or not os.access(worker_python, os.X_OK):
        raise ValueError("formal worker Python is missing/not executable")
    for key in ("runtime_root", "project_root", "vendor_root"):
        directory = Path(str(runtime.get(key, "")))
        _no_symlink_chain(directory, f"runtime {key}")
        if not directory.is_dir():
            raise ValueError(f"runtime {key} is not a directory")
    candidate_inputs = cfg.get("candidate_inputs") or {}
    _file_record(candidate_inputs.get("point_inference_config"), "CA point-only inference config")
    processed_root = Path(str(candidate_inputs.get("processed_rgbd_root", "")))
    _no_symlink_chain(processed_root, "processed CA train RGB-D root")
    if not processed_root.is_dir():
        raise ValueError("processed CA train RGB-D root is missing")
    if candidate_inputs.get("protocol") != {
        "pixel_stride": 4, "voxel_size_m": 0.01, "min_depth_m": 0.1,
        "max_depth_m": 6.0, "depth_scale": 1000.0,
        "score_threshold": 0.01, "max_proposals": 256,
        "near_iou": 0.15, "prefix_id": "p100_gap20",
    }:
        raise ValueError("point/candidate protocol differs")

    scene = cfg.get("scene_contract") or {}
    _, selection = _record(
        scene.get("selection_contract"), "E961 selection contract",
        schema="boxfusion.tr3d.ca1m_e961_selection.v1",
    )
    if (
        selection.get("complete") is not True
        or scene.get("fit_folds") != [2, 3, 4]
        or scene.get("reused_dev_folds") != [0]
        or (scene.get("fit_scene_count"), scene.get("reused_dev_scene_count"), scene.get("scene_count"))
        != (60, 20, 80)
    ):
        raise ValueError("scene contract differs")
    roles = scene.get("roles") or {}
    if tuple(roles) != ROLE_ORDER:
        raise ValueError("role order differs")
    heldout_sets: list[set[str]] = []
    role_scenes: dict[str, tuple[str, ...]] = {}
    for role in ROLE_ORDER:
        train_folds, heldout, order = ROLE_SPECS[role]
        row = roles.get(role) or {}
        if (
            row.get("order") != order
            or tuple(row.get("train_folds", ())) != train_folds
            or row.get("heldout_fold") != heldout
            or row.get("train_scenes") != 1001
            or row.get("heldout_scenes") != 20
            or heldout in train_folds
        ):
            raise ValueError(f"{role}: fold/count contract differs")
        _scene_list(row.get("train_scene_list"), f"{role} train list", 1001)
        predicted = _scene_list(row.get("predict_scene_list"), f"{role} predict list", 20)
        role_scenes[role] = predicted
        heldout_sets.append(set(predicted))
    if len(set().union(*heldout_sets)) != 80 or any(
        heldout_sets[i] & heldout_sets[j] for i in range(4) for j in range(i + 1, 4)
    ):
        raise ValueError("role prediction lists are not disjoint exact80")

    anchors = cfg.get("anchor_inputs") or {}
    _, final = _record(
        anchors.get("final_base_collection"), "sealed final-base collection",
        schema="boxfusion.ca1m_final_base_identity_audit.v1",
    )
    _, b6 = _record(
        anchors.get("native_b6_collection"), "sealed native-B6 collection",
        schema="boxfusion.ca1m_native_b6_final_base_train_collection.v2",
    )
    _, oof_manifest = _record(
        anchors.get("native_b6_oof_sidecar_manifest"), "B6 OOF manifest",
        schema="boxfusion.ca1m_native_b6_oof_row_scores_manifest.v2",
    )
    oof_path = _file_record(anchors.get("native_b6_oof_sidecar"), "B6 OOF sidecar")
    expected80 = set().union(*(set(values) for values in role_scenes.values()))
    final_rows = final.get("per_scene") or {}
    b6_scene_ids = {str(row.get("scene_id")) for row in b6.get("scenes") or []}
    if (
        final.get("ok") is not True or final.get("scene_count") != 100
        or final.get("ground_truth_access") is not False
        or b6.get("complete") is not True or b6.get("scene_count") != 100
        or b6.get("old_native_b6_diagnostics_reused") is not False
        or oof_manifest.get("complete") is not True
        or oof_manifest.get("each_row_model_excludes_scene") is not True
        or not expected80.issubset(final_rows)
        or not expected80.issubset(b6_scene_ids)
    ):
        raise ValueError("final-base/B6 OOF static lineage differs")
    with np.load(BytesIO(stable_bytes(oof_path, "B6 OOF sidecar")), allow_pickle=False) as archive:
        required = {"schema", "complete", "each_row_model_excludes_scene", "scene_ids", "fold_ids", "heldout_model_fold_ids", "source_row_indices", "detector_scores", "deployment_blend_oof_scores"}
        if not required.issubset(archive.files):
            raise ValueError("B6 OOF sidecar members differ")
        scores = np.asarray(archive["deployment_blend_oof_scores"])
        sidecar_scenes = np.asarray(archive["scene_ids"]).astype(str)
        sidecar_folds = np.asarray(archive["fold_ids"])
        heldout_models = np.asarray(archive["heldout_model_fold_ids"])
        source_rows = np.asarray(archive["source_row_indices"])
        if (
            str(np.asarray(archive["schema"]).item()) != "boxfusion.ca1m_native_b6_oof_row_scores.v2"
            or bool(np.asarray(archive["complete"]).item()) is not True
            or bool(np.asarray(archive["each_row_model_excludes_scene"]).item()) is not True
            or scores.dtype != np.float32 or not np.isfinite(scores).all()
            or np.any((scores < 0) | (scores > 1))
        ):
            raise ValueError("B6 OOF sidecar content differs")
        for role, scenes in role_scenes.items():
            fold = ROLE_SPECS[role][1]
            for scene_id in scenes:
                mask = sidecar_scenes == scene_id
                count = int((final_rows.get(scene_id) or {}).get("active_rows", -1))
                if (
                    int(mask.sum()) != count
                    or not np.array_equal(source_rows[mask], np.arange(count, dtype=np.int64))
                    or not np.all(sidecar_folds[mask] == fold)
                    or not np.all(heldout_models[mask] == fold)
                ):
                    raise ValueError(f"{scene_id}: anchor OOF scene/fold/row topology differs")

    access = cfg.get("access") or {}
    if access != {
        "official_train_only": True,
        "ground_truth_access": False,
        "fold1_path_present": False,
        "official_validation_path_present": False,
        "scannet_weight_or_artifact_access": False,
        "old_ca_terminal_v1_v4_artifact_access": False,
    }:
        raise ValueError("access contract differs")
    output = cfg.get("outputs") or {}
    namespace_root = Path(str(output.get("namespace_root", "")))
    if namespace_root.name != NAMESPACE:
        raise ValueError("output namespace root differs")
    for key, value in output.items():
        if key != "namespace_root":
            _guard_output(Path(str(value)), namespace_root, f"output {key}")
    return {
        "roles": role_scenes,
        "outer_verifier": outer_verifier,
        "inner_verifier": inner_verifier,
        "oof_sidecar": oof_path,
        "output_root": namespace_root,
    }


def validate_static_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    source, cfg = _load_config(path)
    if (
        cfg.get("schema") != PENDING_SCHEMA
        or cfg.get("state") != "pending_formal_e961_receipts_and_run_authorization"
        or cfg.get("static_contract_only") is not True
        or any(cfg.get("authorizations", {}).values())
    ):
        raise ValueError("checked-in revision must remain pending and unauthorized")
    _pending(cfg.get("run_authorization"), AUTH_SCHEMA, "run authorization")
    _pending(cfg.get("continuation_receipt"), CONTINUATION_SCHEMA, "continuation receipt")
    for role in ROLE_ORDER:
        schema = OUTER_SCHEMA if role == "outer_dev" else INNER_SCHEMA
        _pending(cfg["scene_contract"]["roles"][role].get("source_success_receipt"), schema, f"{role} receipt")
    common = _validate_static_common(cfg)
    return {
        "schema": "boxfusion.ca1m_tr3d_e961_terminal_inputs_static_preflight.v2",
        "status": "PASS_STATIC_PENDING",
        "config": {"path": str(source), "sha256": sha256_file(source)},
        "namespace": NAMESPACE,
        "scene_count": 80,
        "role_scene_counts": {role: len(common["roles"][role]) for role in ROLE_ORDER},
        "inner_receipt_schema": INNER_SCHEMA,
        "legacy_inner_receipt_rejected": LEGACY_INNER_SCHEMA,
        "ground_truth_access": False,
        "fold1_path_present": False,
        "official_validation_path_present": False,
        "gpu_started": False,
        "output_created": False,
        "operational_authorized": False,
    }


def _load_module(path: Path, digest: str, name: str) -> Any:
    if sha256_file(path) != digest:
        raise ValueError(f"{name} verifier SHA256 changed")
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(f"_bound_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "verify_success_receipt", None)):
        raise ValueError(f"{name} verifier API differs")
    return module


def _continuation(record: Any) -> tuple[Path, dict[str, Any]]:
    path, value = _record(record, "outer continuation receipt", schema=CONTINUATION_SCHEMA)
    gate = value.get("continuation_gate") or {}
    if (
        value.get("complete") is not True
        or not (value.get("pass") is True or gate.get("pass") is True)
        or not (
            value.get("continue_inner_training_authorized") is True
            or gate.get("continue_inner_training_authorized") is True
        )
        or list(value.get("authorized_inner_roles") or gate.get("authorized_inner_roles") or [])
        != list(ROLE_ORDER[1:])
        or value.get("fold1_access") is not False
        or value.get("official_validation_access") is not False
        or value.get("checkpoint_selection") is not False
    ):
        raise ValueError("outer continuation is not a passing fixed-checkpoint receipt")
    return path, value


@dataclass(frozen=True)
class VerifiedRole:
    role: str
    train_folds: tuple[int, ...]
    heldout_fold: int
    scenes: tuple[str, ...]
    receipt_path: Path
    receipt_sha256: str
    receipt: Mapping[str, Any]
    checkpoint_path: Path
    checkpoint_sha256: str


@dataclass(frozen=True)
class ReadyContext:
    config_path: Path
    config: Mapping[str, Any]
    authorization_path: Path
    authorization_sha256: str
    continuation_path: Path
    continuation_sha256: str
    roles: Mapping[str, VerifiedRole]


def _checkpoint(receipt: Mapping[str, Any], role: str) -> tuple[Path, str]:
    record = receipt.get("checkpoint") or {}
    if role == "outer_dev":
        record = (((receipt.get("terminal") or {}).get("checkpoint_audit") or {}).get("checkpoint") or {})
    path = Path(str(record.get("path", ""))).absolute()
    digest = _sha(record.get("sha256"), f"{role} checkpoint SHA256")
    if path.name != "iter_11268.pth" or sha256_file(path) != digest:
        raise ValueError(f"{role}: fixed terminal checkpoint differs")
    return path, digest


def _authorization_projection(cfg: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(cfg))
    value["run_authorization"] = {
        "state": "bound", "path": "<redacted>", "sha256": "<redacted>", "schema": AUTH_SCHEMA,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_operational_ready(path: Path = DEFAULT_CONFIG) -> ReadyContext:
    """Deep-verify all formal receipts before any output/GPU operation."""

    source, cfg = _load_config(path)
    if cfg.get("schema") == PENDING_SCHEMA:
        # Deliberately before static input traversal, output lookup, device
        # validation, verifier import, receipt/checkpoint opening, or mkdir.
        raise PendingOperationalInputs("formal E961 receipts/run authorization are pending")
    if (
        cfg.get("state") != "ready_four_verified_e961_receipts"
        or cfg.get("static_contract_only") is not False
        or cfg.get("authorizations") != OPERATIONAL_AUTHORIZATIONS
    ):
        raise PermissionError("ready config state/authorization differs")

    continuation_path, _ = _continuation(cfg.get("continuation_receipt"))
    continuation_sha = sha256_file(continuation_path)
    producers = cfg["producer_contracts"]
    outer_v = producers["outer"]["verifier"]
    inner_v = producers["inner"]["verifier"]
    outer_module = _load_module(
        Path(outer_v["path"]), _sha(outer_v["sha256"], "outer verifier SHA256"), "e961_outer_r2",
    )
    inner_module = _load_module(
        Path(inner_v["path"]), _sha(inner_v["sha256"], "inner verifier SHA256"), "e961_inner_r2",
    )
    roles_cfg = cfg["scene_contract"]["roles"]
    verified: dict[str, VerifiedRole] = {}
    for role in ROLE_ORDER:
        train_folds, heldout, _ = ROLE_SPECS[role]
        record = roles_cfg[role].get("source_success_receipt") or {}
        expected_schema = OUTER_SCHEMA if role == "outer_dev" else INNER_SCHEMA
        if record.get("schema") == LEGACY_INNER_SCHEMA:
            raise ValueError(f"{role}: legacy 60-scene receipt is forbidden")
        receipt_path, receipt = _record(record, f"{role} source success receipt", schema=expected_schema)
        verified_value = (
            outer_module.verify_success_receipt(receipt_path)
            if role == "outer_dev" else inner_module.verify_success_receipt(receipt_path)
        )
        if isinstance(verified_value, tuple):
            verified_value = verified_value[1]
        if verified_value != receipt:
            raise ValueError(f"{role}: producer verifier result differs from stable receipt")
        protocol = receipt.get("training_protocol") or {}
        if role == "outer_dev":
            protocol = {
                "train_folds": [2, 3, 4], "heldout_fold": 0,
                "train_scenes": 1001, "heldout_scenes": 20,
                "initialization": "random_scratch_ca_only",
            }
        if (
            receipt.get("role") != role
            or receipt.get("status") != "success"
            or receipt.get("exit_code") != 0
            or list(protocol.get("train_folds", ())) != list(train_folds)
            or int(protocol.get("heldout_fold", -1)) != heldout
            or int(protocol.get("train_scenes", receipt.get("train_scenes", -1))) != 1001
            or int(protocol.get("heldout_scenes", receipt.get("heldout_scenes", -1))) != 20
            or protocol.get("initialization") != "random_scratch_ca_only"
        ):
            raise ValueError(f"{role}: producer receipt role/fold/training semantics differ")
        if role != "outer_dev":
            upstream = (receipt.get("passing_upstream") or {}).get("eval_v2_continuation_receipt") or {}
            if upstream.get("sha256") != continuation_sha:
                raise ValueError(f"{role}: inner receipt does not bind this continuation")
        checkpoint_path, checkpoint_sha = _checkpoint(receipt, role)
        effective_record = receipt.get("effective_config") or {}
        if role == "outer_dev":
            effective_record = (receipt.get("terminal") or {}).get("effective_config") or {}
        effective_path = _file_record(effective_record, f"{role} effective training config")
        from .ca1m_tr3d_inference_contract import validate_ca1m_point_inference_config
        point_record = cfg["candidate_inputs"]["point_inference_config"]
        validate_ca1m_point_inference_config(
            inference_path=Path(point_record["path"]),
            inference_sha256=point_record["sha256"],
            effective_training_path=effective_path,
            effective_training_sha256=sha256_file(effective_path),
        )
        verified[role] = VerifiedRole(
            role, train_folds, heldout,
            _scene_list(roles_cfg[role]["predict_scene_list"], f"{role} predict list", 20),
            receipt_path, sha256_file(receipt_path), receipt, checkpoint_path, checkpoint_sha,
        )

    auth_path, auth = _record(cfg.get("run_authorization"), "terminal-input run authorization", schema=AUTH_SCHEMA)
    expected_roles = [
        {"role": role, "receipt_path": str(verified[role].receipt_path),
         "receipt_sha256": verified[role].receipt_sha256,
         "checkpoint_sha256": verified[role].checkpoint_sha256}
        for role in ROLE_ORDER
    ]
    if (
        auth.get("complete") is not True or auth.get("create_only") is not True
        or auth.get("namespace") != NAMESPACE
        or auth.get("ready_config_projection_sha256") != _authorization_projection(cfg)
        or auth.get("producer_deep_verifiers_passed") is not True
        or auth.get("roles") != expected_roles
        or (auth.get("continuation_receipt") or {}).get("sha256") != continuation_sha
        or auth.get("authorizations") != OPERATIONAL_AUTHORIZATIONS
        or auth.get("ground_truth_access") is not False
        or auth.get("fold1_access") is not False
        or auth.get("official_validation_access") is not False
        or auth.get("formal_gpu_run_started") is not False
    ):
        raise PermissionError("terminal-input run authorization binding differs")

    # Static sources and output-parent writability are intentionally checked
    # only after all four producer verifiers and the authorization have passed.
    _validate_static_common(cfg)
    namespace_root = Path(cfg["outputs"]["namespace_root"])
    _no_symlink_chain(namespace_root, "namespace output", missing_tail=True)
    parent = namespace_root.parent
    _no_symlink_chain(parent, "namespace parent")
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        raise PermissionError("namespace parent is not host-writable/searchable")
    return ReadyContext(
        source, cfg, auth_path, sha256_file(auth_path), continuation_path,
        continuation_sha, verified,
    )


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def ensure_directory(path: Path) -> Path:
    """Create directories component-wise without following a symlink."""

    path = Path(path)
    _no_symlink_chain(path, "output directory", missing_tail=True)
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    _no_symlink_chain(current, "existing output parent")
    if not current.is_dir() or not os.access(current, os.W_OK | os.X_OK):
        raise PermissionError(f"output parent is not writable: {current}")
    for directory in reversed(missing):
        try:
            os.mkdir(directory, 0o700)
        except FileExistsError:
            pass
        mode = os.lstat(directory).st_mode
        if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
            raise ValueError(f"output component is not a real directory: {directory}")
    return path


def write_bytes_exclusive(path: Path, data: bytes) -> Path:
    """FUSE-safe create-only publication; a crash residue fails closed."""

    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing existing output: {path}")
    ensure_directory(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    complete = False
    try:
        view = memoryview(data)
        while view:
            count = os.write(fd, view)
            if count < 1:
                raise OSError("short create-only write")
            view = view[count:]
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        complete = True
    finally:
        os.close(fd)
    if not complete:
        # Preserve the partial inode: silent cleanup could make a crashed run
        # look resumable.  A later stable validator will reject it.
        raise RuntimeError(f"partial create-only output retained: {path}")
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if stable_bytes(path, "published output", nonempty=False) != data:
        raise IOError(f"published bytes differ: {path}")
    return path


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> Path:
    return write_bytes_exclusive(path, canonical_json(value))


def write_npz_exclusive(path: Path, value: Mapping[str, Any]) -> Path:
    stream = BytesIO()
    np.savez_compressed(stream, **value)
    return write_bytes_exclusive(path, stream.getvalue())


def create_or_verify(path: Path, data: bytes, name: str) -> Path:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or stable_bytes(path, name, nonempty=False) != data:
            raise ValueError(f"{name} resume bytes differ")
        if os.stat(path, follow_symlinks=False).st_mode & 0o222:
            raise ValueError(f"{name} resume artifact is writable")
        return path
    return write_bytes_exclusive(path, data)


def _record_dict(path: Path, schema: str | None = None) -> dict[str, Any]:
    result = {"path": str(path), "sha256": sha256_file(path)}
    if schema is not None:
        result["schema"] = schema
    return result


def ensure_runtime_namespace(ctx: ReadyContext) -> Mapping[str, Path]:
    """Claim one namespace and materialize generic-v5 receipt adapters.

    This is the first function allowed to create output.  Its argument can be
    produced only by the all-receipt deep preflight above.
    """

    root = ensure_directory(Path(ctx.config["outputs"]["namespace_root"]))
    owner = {
        "schema": "boxfusion.ca1m_tr3d_e961_terminal_inputs_namespace_owner.v2",
        "complete": True, "create_only": True, "namespace": NAMESPACE,
        "authorization": _record_dict(ctx.authorization_path, AUTH_SCHEMA),
        "config_projection_sha256": _authorization_projection(ctx.config),
        "resume_policy": "same_authorization_exact_revalidation_only",
    }
    create_or_verify(root / "NAMESPACE_OWNER.json", canonical_json(owner), "namespace owner")

    receipt_root = ensure_directory(Path(ctx.config["outputs"]["receipt_root"]))
    normalized: dict[str, Path] = {}
    for role in ROLE_ORDER:
        verified = ctx.roles[role]
        adapter_path = receipt_root / f"{role}_verified_producer_adapter.json"
        adapter = {
            "schema": ADAPTER_SCHEMA, "complete": True, "create_only": True,
            "status": "success", "role": role, "checkpoint_selection": False,
            "training_protocol": {
                "train_folds": list(verified.train_folds),
                "heldout_fold": verified.heldout_fold,
                "initialization": "random_scratch_ca_only",
                "scannet_checkpoint_or_module_access": False,
            },
            "checkpoint": {
                "path": str(verified.checkpoint_path),
                "sha256": verified.checkpoint_sha256,
                "optimizer_updates": 11268,
                "checkpoint_selection": False,
            },
            "source_producer_receipt": {
                **_record_dict(verified.receipt_path),
                "schema": OUTER_SCHEMA if role == "outer_dev" else INNER_SCHEMA,
                "producer_verify_success_receipt_passed": True,
            },
            "access": {
                "fold1_metadata_or_ground_truth_access": False,
                "official_validation_access": False,
                "scannet_checkpoint_or_module_access": False,
            },
        }
        create_or_verify(adapter_path, canonical_json(adapter), f"{role} verified adapter")
        normalized_path = receipt_root / f"{role}_detector_role_receipt_v5.json"
        normalized_payload = {
            "schema": "boxfusion.ca1m_tr3d_xfit_r2_detector_role_receipt.v1",
            "complete": True, "create_only": True, "role": role,
            "detector_train_folds": list(verified.train_folds),
            "candidate_output_fold": verified.heldout_fold,
            "checkpoint_policy": "fixed_iter_11268_only_no_selection",
            "initialization": "random_scratch_ca_only",
            "checkpoint": {"path": str(verified.checkpoint_path), "sha256": verified.checkpoint_sha256},
            "source_training_receipt": _record_dict(adapter_path, ADAPTER_SCHEMA),
            "outer_continuation_receipt": _record_dict(ctx.continuation_path, CONTINUATION_SCHEMA),
            "access": {
                "fold1_metadata_or_ground_truth_access": False,
                "official_validation_access": False,
                "scannet_checkpoint_or_module_access": False,
            },
        }
        create_or_verify(
            normalized_path, canonical_json(normalized_payload), f"{role} normalized receipt",
        )
        from .ca1m_tr3d_terminal_gate_v5 import load_detector_role_receipt_v5
        load_detector_role_receipt_v5(_record_dict(normalized_path, normalized_payload["schema"]), role)
        normalized[role] = normalized_path
    return normalized


@dataclass(frozen=True)
class ProposalSummaryR2:
    scene_id: str
    role: str
    fold_id: int
    frame_count: int
    used_frame_count: int
    point_count: int
    candidate_count: int
    model_runtime_s: float
    source_points_sha256: str
    checkpoint_sha256: str
    receipt_sha256: str
    authorization_sha256: str
    adapter_mode: str
    device: str


def proposal_payload(
    *, summary: ProposalSummaryR2, used_frame_ids: Any, world_to_local: Any,
    candidate_corners_world: Any, candidate_scores: Any, candidate_point_count: Any,
    candidate_boxes_local: Any, candidate_labels: Any,
) -> dict[str, Any]:
    frames = np.ascontiguousarray(used_frame_ids, dtype=np.int64)
    transform = np.ascontiguousarray(world_to_local, dtype=np.float64)
    corners = np.ascontiguousarray(candidate_corners_world, dtype=np.float32)
    scores = np.ascontiguousarray(candidate_scores, dtype=np.float32)
    support = np.ascontiguousarray(candidate_point_count, dtype=np.int64)
    boxes = np.ascontiguousarray(candidate_boxes_local, dtype=np.float32)
    labels = np.ascontiguousarray(candidate_labels, dtype=np.int64)
    n = len(scores)
    if (
        frames.shape != (summary.used_frame_count,) or len(frames) < 1
        or np.any(np.diff(frames) <= 0) or np.any(frames < 0)
        or transform.shape != (4, 4)
        or corners.shape != (n, 8, 3) or boxes.shape != (n, 7)
        or support.shape != (n,) or labels.shape != (n,)
        or n != summary.candidate_count or summary.point_count < 1
        or not all(np.isfinite(x).all() for x in (transform, corners, scores, boxes))
        or np.any((scores < 0) | (scores > 1)) or np.any(support < 0)
        or np.any(labels != 0) or np.any(boxes[:, 3:6] <= 0)
        or summary.adapter_mode != "genuine" or summary.device != "cuda:0"
    ):
        raise ValueError(f"{summary.scene_id}: malformed anchor-free proposal payload")
    metadata = {
        "schema": PROPOSAL_SCHEMA, "complete": True, "create_only": True,
        "ground_truth_access": False, "anchor_access": False, "b6_access": False,
        **summary.__dict__,
    }
    return {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "used_frame_ids": frames, "world_to_local": transform,
        "candidate_corners_world": corners, "candidate_scores": scores,
        "candidate_point_count": support, "candidate_boxes_local": boxes,
        "candidate_labels": labels,
    }


def load_proposal(path: Path, *, expected_scene: str | None = None) -> dict[str, Any]:
    source = Path(path)
    if os.stat(source, follow_symlinks=False).st_mode & 0o222:
        raise ValueError("proposal resume artifact must be read-only")
    raw = stable_bytes(source, "E961 proposal")
    with np.load(BytesIO(raw), allow_pickle=False) as archive:
        required = {
            "metadata_json", "used_frame_ids", "world_to_local",
            "candidate_corners_world", "candidate_scores", "candidate_point_count",
            "candidate_boxes_local", "candidate_labels",
        }
        if set(archive.files) != required:
            raise ValueError("proposal key inventory differs")
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
        arrays = {name: np.array(archive[name], copy=True) for name in required - {"metadata_json"}}
    if (
        metadata.get("schema") != PROPOSAL_SCHEMA
        or metadata.get("complete") is not True or metadata.get("create_only") is not True
        or metadata.get("ground_truth_access") is not False
        or metadata.get("anchor_access") is not False or metadata.get("b6_access") is not False
    ):
        raise ValueError("proposal metadata differs")
    fields = ProposalSummaryR2.__dataclass_fields__
    summary = ProposalSummaryR2(**{name: metadata[name] for name in fields})
    if expected_scene is not None and summary.scene_id != expected_scene:
        raise ValueError("proposal scene differs")
    rebuilt = proposal_payload(summary=summary, **arrays)
    for name in arrays:
        if not np.array_equal(arrays[name], rebuilt[name]):
            raise ValueError(f"proposal {name} differs on rebuild")
    return {"path": source, "sha256": hashlib.sha256(raw).hexdigest(), "summary": summary, **arrays}


def _exact_inventory(root: Path, names: set[str], suffix: str, name: str) -> None:
    if not root.exists():
        return
    del suffix
    actual = {entry.name for entry in root.iterdir()}
    if actual - names:
        raise ValueError(f"{name} contains unexpected/partial artifacts: {sorted(actual - names)}")


def _proposal_path(cfg: Mapping[str, Any], role: str, scene: str) -> Path:
    return Path(cfg["outputs"]["proposal_root"]) / role / f"{scene}_anchor_free_v5_r2.npz"


def run_stage_p(
    ctx: ReadyContext, role: str, *, device: str = "cuda:0",
    worker_factory: Callable[..., Any] | None = None,
    point_builder: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run/resume one exact-20 point-only, anchor-free detector role."""

    if role not in ROLE_SPECS or device != "cuda:0":
        raise ValueError("formal stage P requires a frozen role and cuda:0")
    ensure_runtime_namespace(ctx)
    verified = ctx.roles[role]
    root = ensure_directory(Path(ctx.config["outputs"]["proposal_root"]) / role)
    expected = {f"{scene}_anchor_free_v5_r2.npz" for scene in verified.scenes}
    _exact_inventory(root, expected, ".npz", f"{role} proposal root")
    pending: list[str] = []
    reports: dict[str, Any] = {}
    for scene in verified.scenes:
        target = _proposal_path(ctx.config, role, scene)
        if target.exists() or target.is_symlink():
            loaded = load_proposal(target, expected_scene=scene)
            summary = loaded["summary"]
            if (
                summary.role != role or summary.fold_id != verified.heldout_fold
                or summary.checkpoint_sha256 != verified.checkpoint_sha256
                or summary.receipt_sha256 != verified.receipt_sha256
                or summary.authorization_sha256 != ctx.authorization_sha256
            ):
                raise ValueError(f"{scene}: resumed proposal provenance differs")
            reports[scene] = {"sha256": loaded["sha256"], "candidate_count": summary.candidate_count, "resumed": True}
        else:
            pending.append(scene)
    if pending:
        if worker_factory is None:
            from .ca1m_tr3d_worker_client import CA1MTR3DWorker
            worker_factory = CA1MTR3DWorker
        if point_builder is None:
            from tools.run_ca1m_tr3d_proposal_cache_v4 import _build_scene_points
            point_builder = _build_scene_points
        from .ca1m_tr3d_terminal import terminal_world_to_local
        runtime, protocol = ctx.config["runtime"], ctx.config["candidate_inputs"]["protocol"]
        worker_args = dict(
            python=runtime["worker_python"], worker_script=runtime["worker_script"],
            runtime_root=runtime["runtime_root"],
            config=ctx.config["candidate_inputs"]["point_inference_config"]["path"],
            checkpoint=str(verified.checkpoint_path), project_root=runtime["project_root"],
            vendor_root=runtime["vendor_root"], startup_timeout_s=runtime["startup_timeout_s"],
            device=device, extra_args=("--score-threshold", str(protocol["score_threshold"]), "--max-proposals", str(protocol["max_proposals"])),
        )
        with worker_factory(**worker_args) as worker:
            if getattr(worker, "adapter_mode", None) != "genuine":
                raise ValueError("formal stage P forbids a synthetic detector adapter")
            for scene in pending:
                rgb, poses, frames, points = point_builder(
                    data_root=Path(ctx.config["candidate_inputs"]["processed_rgbd_root"]),
                    scene=scene,
                    processed={"root": ctx.config["candidate_inputs"]["processed_rgbd_root"], "depth_scale": protocol["depth_scale"]},
                    protocol=protocol,
                )
                points = np.ascontiguousarray(points, dtype=np.float32)
                point_sha = hashlib.sha256(points.tobytes(order="C")).hexdigest()
                transform = terminal_world_to_local(poses[int(frames[0])])
                result = worker.infer(
                    scene_id=scene, prefix_id=protocol["prefix_id"],
                    points_world_xyzrgb=points, world_to_local=transform,
                )
                if result.source_points_sha256 != point_sha:
                    raise ValueError(f"{scene}: detector point lineage differs")
                summary = ProposalSummaryR2(
                    scene, role, verified.heldout_fold, len(rgb), len(frames), len(points),
                    len(result.scores), float(result.model_runtime_s), point_sha,
                    verified.checkpoint_sha256, verified.receipt_sha256,
                    ctx.authorization_sha256, result.adapter_mode, device,
                )
                target = _proposal_path(ctx.config, role, scene)
                write_npz_exclusive(target, proposal_payload(
                    summary=summary, used_frame_ids=frames, world_to_local=transform,
                    candidate_corners_world=result.corners_world,
                    candidate_scores=result.scores, candidate_point_count=result.point_counts,
                    candidate_boxes_local=result.boxes_local, candidate_labels=result.labels,
                ))
                loaded = load_proposal(target, expected_scene=scene)
                reports[scene] = {"sha256": loaded["sha256"], "candidate_count": summary.candidate_count, "resumed": False}
    return seal_proposal_collection(ctx, role, reports)


def seal_proposal_collection(
    ctx: ReadyContext, role: str, reports: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    verified = ctx.roles[role]
    root = Path(ctx.config["outputs"]["proposal_root"]) / role
    expected = {f"{scene}_anchor_free_v5_r2.npz" for scene in verified.scenes}
    _exact_inventory(root, expected, ".npz", f"{role} proposal root")
    rows, total = [], 0
    for scene in verified.scenes:
        loaded = load_proposal(_proposal_path(ctx.config, role, scene), expected_scene=scene)
        summary = loaded["summary"]
        if summary.role != role or summary.fold_id != verified.heldout_fold:
            raise ValueError(f"{scene}: proposal detector OOF role differs")
        total += summary.candidate_count
        rows.append({"scene_id": scene, "path": str(loaded["path"]), "sha256": loaded["sha256"], "candidate_count": summary.candidate_count})
    payload = {
        "schema": PROPOSAL_COLLECTION_SCHEMA, "complete": True, "create_only": True,
        "role": role, "train_folds": list(verified.train_folds),
        "heldout_fold": verified.heldout_fold, "scene_count": 20,
        "candidate_count": total, "anchor_free": True, "ground_truth_access": False,
        "producer_receipt": _record_dict(verified.receipt_path, OUTER_SCHEMA if role == "outer_dev" else INNER_SCHEMA),
        "checkpoint_sha256": verified.checkpoint_sha256,
        "authorization_sha256": ctx.authorization_sha256, "scenes": rows,
    }
    manifest = ensure_directory(Path(ctx.config["outputs"]["manifest_root"])) / f"P_{role}_exact20.json"
    create_or_verify(manifest, canonical_json(payload), f"{role} proposal manifest")
    return {**payload, "manifest": _record_dict(manifest, PROPOSAL_COLLECTION_SCHEMA), "reports": reports or {}}


@dataclass(frozen=True)
class OverlaySummaryR2:
    scene_id: str
    role: str
    fold_id: int
    anchor_count: int
    candidate_count: int
    near_candidate_count: int
    proposal_sha256: str
    final_anchor_sha256: str
    anchor_native_sha256: str
    b6_oof_sidecar_sha256: str
    authorization_sha256: str


def _prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = stable_bytes(path, "sealed final-base prediction")
    with BytesIO(raw) as handle:
        value = pickle.load(handle)  # noqa: S301 -- SHA-bound local sealed artifact
        if handle.read(1):
            raise ValueError("final-base prediction has trailing bytes")
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], list):
        raise ValueError("final-base prediction batch shape differs")
    corners, scores = [], []
    for row in value[0]:
        if not isinstance(row, tuple) or len(row) != 3 or type(row[0]) is not int or row[0] != 0:
            raise ValueError("final-base prediction row differs")
        corner, score = np.asarray(row[1]), float(row[2])
        if corner.dtype != np.float32 or corner.shape != (8, 3) or not np.isfinite(corner).all() or not 0 <= score <= 1:
            raise ValueError("final-base geometry/score differs")
        corners.append(np.array(corner, copy=True, order="C")); scores.append(score)
    return (
        np.stack(corners).astype(np.float32) if corners else np.empty((0, 8, 3), np.float32),
        np.asarray(scores, dtype=np.float32),
    )


def _oof_rows(path: Path, scene: str, fold: int, detector_scores: np.ndarray) -> np.ndarray:
    with np.load(BytesIO(stable_bytes(path, "B6 OOF sidecar")), allow_pickle=False) as archive:
        scene_ids = np.asarray(archive["scene_ids"]).astype(str)
        mask = scene_ids == scene
        rows = np.asarray(archive["source_row_indices"])[mask]
        fold_ids = np.asarray(archive["fold_ids"])[mask]
        heldout = np.asarray(archive["heldout_model_fold_ids"])[mask]
        source_scores = np.asarray(archive["detector_scores"])[mask]
        oof = np.asarray(archive["deployment_blend_oof_scores"])[mask]
    count = len(detector_scores)
    if (
        rows.dtype != np.int64 or not np.array_equal(rows, np.arange(count, dtype=np.int64))
        or not np.array_equal(fold_ids, np.full(count, fold, dtype=fold_ids.dtype))
        or not np.array_equal(heldout, np.full(count, fold, dtype=heldout.dtype))
        or source_scores.dtype != np.float32 or not np.array_equal(source_scores, detector_scores)
        or oof.dtype != np.float32 or oof.shape != (count,) or not np.isfinite(oof).all()
        or np.any((oof < 0) | (oof > 1))
    ):
        raise ValueError(f"{scene}: B6 all-fold OOF row identity differs")
    return np.ascontiguousarray(oof)


def overlay_payload(
    *, summary: OverlaySummaryR2, anchor_corners: Any, anchor_detector_scores: Any,
    anchor_native_features: Any, anchor_scores_oof: Any, proposal: Mapping[str, Any],
    best_anchor_indices: Any, best_anchor_iou: Any,
    best_anchor_center_distance_m: Any, near_mask: Any,
) -> dict[str, Any]:
    anchors = np.ascontiguousarray(anchor_corners, np.float32)
    detector = np.ascontiguousarray(anchor_detector_scores, np.float32)
    native = np.ascontiguousarray(anchor_native_features, np.float32)
    oof = np.ascontiguousarray(anchor_scores_oof, np.float32)
    candidates = np.ascontiguousarray(proposal["candidate_corners_world"], np.float32)
    candidate_scores = np.ascontiguousarray(proposal["candidate_scores"], np.float32)
    best = np.ascontiguousarray(best_anchor_indices, np.int64)
    iou = np.ascontiguousarray(best_anchor_iou, np.float32)
    distance = np.ascontiguousarray(best_anchor_center_distance_m, np.float32)
    near = np.ascontiguousarray(near_mask, np.bool_)
    a, n = len(anchors), len(candidates)
    if (
        anchors.shape != (a, 8, 3) or detector.shape != (a,) or native.shape != (a, 14)
        or oof.shape != (a,) or candidates.shape != (n, 8, 3) or candidate_scores.shape != (n,)
        or best.shape != (n,) or iou.shape != (n,) or distance.shape != (n,) or near.shape != (n,)
        or a != summary.anchor_count or n != summary.candidate_count
        or int(near.sum()) != summary.near_candidate_count
        or not all(np.isfinite(x).all() for x in (anchors, detector, native, oof, candidates, candidate_scores, iou, distance))
        or np.any((detector < 0) | (detector > 1)) or np.any((oof < 0) | (oof > 1))
        or not np.array_equal(native[:, 0], detector)
    ):
        raise ValueError(f"{summary.scene_id}: malformed OOF overlay")
    metadata = {
        "schema": OVERLAY_SCHEMA, "complete": True, "create_only": True,
        "cpu_only": True, "ground_truth_access": False,
        "geometry_authority": "sealed_final_base_prediction",
        "score_source": "ca1m_native_b6_final_base_all_fold_oof_row_scores_v2",
        "deployment_scores_used": False, **summary.__dict__,
    }
    return {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "anchor_corners": anchors, "anchor_detector_scores": detector,
        "anchor_native_features": native, "anchor_scores_oof": oof,
        "candidate_corners_world": candidates, "candidate_scores": candidate_scores,
        "best_anchor_indices": best, "best_anchor_iou": iou,
        "best_anchor_center_distance_m": distance, "near_mask": near,
    }


def load_overlay(path: Path, *, proposal: Mapping[str, Any], expected_scene: str | None = None) -> dict[str, Any]:
    if os.stat(path, follow_symlinks=False).st_mode & 0o222:
        raise ValueError("overlay resume artifact must be read-only")
    raw = stable_bytes(path, "E961 OOF overlay")
    with np.load(BytesIO(raw), allow_pickle=False) as archive:
        required = {
            "metadata_json", "anchor_corners", "anchor_detector_scores", "anchor_native_features",
            "anchor_scores_oof", "candidate_corners_world", "candidate_scores",
            "best_anchor_indices", "best_anchor_iou", "best_anchor_center_distance_m", "near_mask",
        }
        if set(archive.files) != required:
            raise ValueError("overlay key inventory differs")
        meta = json.loads(str(np.asarray(archive["metadata_json"]).item()))
        arrays = {name: np.array(archive[name], copy=True) for name in required - {"metadata_json"}}
    if (
        meta.get("schema") != OVERLAY_SCHEMA or meta.get("complete") is not True
        or meta.get("create_only") is not True or meta.get("cpu_only") is not True
        or meta.get("ground_truth_access") is not False
        or meta.get("geometry_authority") != "sealed_final_base_prediction"
        or meta.get("score_source") != "ca1m_native_b6_final_base_all_fold_oof_row_scores_v2"
        or meta.get("deployment_scores_used") is not False
    ):
        raise ValueError("overlay metadata differs")
    summary = OverlaySummaryR2(**{name: meta[name] for name in OverlaySummaryR2.__dataclass_fields__})
    if expected_scene is not None and summary.scene_id != expected_scene:
        raise ValueError("overlay scene differs")
    if summary.proposal_sha256 != proposal["sha256"]:
        raise ValueError("overlay proposal binding differs")
    if (
        not np.array_equal(arrays["candidate_corners_world"], proposal["candidate_corners_world"])
        or not np.array_equal(arrays["candidate_scores"], proposal["candidate_scores"])
    ):
        raise ValueError("overlay candidate rows differ from proposal")
    rebuilt = overlay_payload(
        summary=summary, proposal=proposal,
        anchor_corners=arrays["anchor_corners"],
        anchor_detector_scores=arrays["anchor_detector_scores"],
        anchor_native_features=arrays["anchor_native_features"],
        anchor_scores_oof=arrays["anchor_scores_oof"],
        best_anchor_indices=arrays["best_anchor_indices"],
        best_anchor_iou=arrays["best_anchor_iou"],
        best_anchor_center_distance_m=arrays["best_anchor_center_distance_m"],
        near_mask=arrays["near_mask"],
    )
    for name in arrays:
        if not np.array_equal(arrays[name], rebuilt[name]):
            raise ValueError(f"overlay {name} differs on rebuild")
    from .ca1m_tr3d_terminal import associate_terminal_candidates
    association = associate_terminal_candidates(
        anchor_corners=arrays["anchor_corners"], anchor_scores=arrays["anchor_scores_oof"],
        candidate_corners=arrays["candidate_corners_world"],
        candidate_scores=arrays["candidate_scores"], near_iou=0.15,
    )
    for name in ("best_anchor_indices", "best_anchor_iou", "best_anchor_center_distance_m", "near_mask"):
        if not np.array_equal(arrays[name], getattr(association, name)):
            raise ValueError(f"overlay {name} differs from association recomputation")
    return {"path": Path(path), "sha256": hashlib.sha256(raw).hexdigest(), "summary": summary, **arrays}


def _overlay_path(cfg: Mapping[str, Any], role: str, scene: str) -> Path:
    return Path(cfg["outputs"]["overlay_root"]) / role / f"{scene}_oof_overlay_v5_r2.npz"


def _anchor_scene_inputs(ctx: ReadyContext, role: str, scene: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    anchors_cfg = ctx.config["anchor_inputs"]
    final_path, final = _record(anchors_cfg["final_base_collection"], "final-base collection", schema="boxfusion.ca1m_final_base_identity_audit.v1")
    b6_path, b6 = _record(anchors_cfg["native_b6_collection"], "native-B6 collection", schema="boxfusion.ca1m_native_b6_final_base_train_collection.v2")
    final_row = (final.get("per_scene") or {}).get(scene) or {}
    b6_rows = {str(row.get("scene_id")): row for row in b6.get("scenes") or []}
    b6_row = b6_rows.get(scene) or {}
    anchor_path = Path(anchors_cfg["final_base_prediction_root"]) / f"{scene}_boxes.pkl"
    anchor_sha = sha256_file(anchor_path)
    if final_row.get("active_prediction_sha256") != anchor_sha or b6_row.get("final_base_prediction_sha256") != anchor_sha:
        raise ValueError(f"{scene}: sealed final-base geometry binding differs")
    corners, detector_scores = _prediction(anchor_path)
    completion_path = b6_path.parent / "completion/offline_native_b6" / f"{scene}.json"
    completion, value = _json(completion_path, f"{scene} B6 completion")
    if (
        sha256_file(completion) != b6_row.get("observer_completion_sha256")
        or value.get("schema") != "boxfusion.ca1m_native_b6_final_base_scene_completion.v2"
        or value.get("complete") is not True or value.get("ground_truth_access", False) is not False
        or value.get("validation_ground_truth_access") is not False
        or value.get("old_native_b6_diagnostics_reused") is not False
        or (value.get("source_final_base_manifest") or {}).get("sha256") != sha256_file(final_path)
    ):
        raise ValueError(f"{scene}: B6 completion lineage differs")
    diagnostic_record = (value.get("artifacts") or {}).get("native_b6_diagnostic") or {}
    diagnostic = Path(str(diagnostic_record.get("path", "")))
    diagnostic_sha = sha256_file(diagnostic)
    if diagnostic_sha != diagnostic_record.get("sha256"):
        raise ValueError(f"{scene}: anchor-native diagnostic SHA256 differs")
    from .ca1m_native_b6_score import load_native_observer_diagnostic
    native = load_native_observer_diagnostic(
        diagnostic, scene_id=scene, corners=corners, scores=detector_scores,
    )["features"].astype(np.float32)
    oof_path = Path(anchors_cfg["native_b6_oof_sidecar"]["path"])
    oof = _oof_rows(oof_path, scene, ctx.roles[role].heldout_fold, detector_scores)
    return corners, detector_scores, native, oof, {
        "anchor_sha": anchor_sha, "native_sha": diagnostic_sha,
        "final_manifest_sha": sha256_file(final_path), "b6_collection_sha": sha256_file(b6_path),
    }


def run_stage_o(ctx: ReadyContext, role: str) -> dict[str, Any]:
    """CPU-only final-base geometry + all-fold OOF B6 association overlay."""

    if role not in ROLE_SPECS:
        raise ValueError("unknown role")
    ensure_runtime_namespace(ctx)
    seal_proposal_collection(ctx, role)
    verified = ctx.roles[role]
    root = ensure_directory(Path(ctx.config["outputs"]["overlay_root"]) / role)
    expected = {f"{scene}_oof_overlay_v5_r2.npz" for scene in verified.scenes}
    _exact_inventory(root, expected, ".npz", f"{role} overlay root")
    from .ca1m_tr3d_terminal import associate_terminal_candidates
    reports: dict[str, Any] = {}
    for scene in verified.scenes:
        proposal = load_proposal(_proposal_path(ctx.config, role, scene), expected_scene=scene)
        target = _overlay_path(ctx.config, role, scene)
        if target.exists() or target.is_symlink():
            loaded = load_overlay(target, proposal=proposal, expected_scene=scene)
            if loaded["summary"].authorization_sha256 != ctx.authorization_sha256:
                raise ValueError(f"{scene}: resumed overlay authorization differs")
            reports[scene] = {"sha256": loaded["sha256"], "resumed": True}
            continue
        corners, detector, native, oof, lineage = _anchor_scene_inputs(ctx, role, scene)
        association = associate_terminal_candidates(
            anchor_corners=corners, anchor_scores=oof,
            candidate_corners=proposal["candidate_corners_world"],
            candidate_scores=proposal["candidate_scores"],
            near_iou=float(ctx.config["candidate_inputs"]["protocol"]["near_iou"]),
        )
        summary = OverlaySummaryR2(
            scene, role, verified.heldout_fold, len(corners),
            len(proposal["candidate_scores"]), int(association.near_mask.sum()),
            proposal["sha256"], lineage["anchor_sha"], lineage["native_sha"],
            sha256_file(Path(ctx.config["anchor_inputs"]["native_b6_oof_sidecar"]["path"])),
            ctx.authorization_sha256,
        )
        write_npz_exclusive(target, overlay_payload(
            summary=summary, anchor_corners=corners, anchor_detector_scores=detector,
            anchor_native_features=native, anchor_scores_oof=oof, proposal=proposal,
            best_anchor_indices=association.best_anchor_indices,
            best_anchor_iou=association.best_anchor_iou,
            best_anchor_center_distance_m=association.best_anchor_center_distance_m,
            near_mask=association.near_mask,
        ))
        loaded = load_overlay(target, proposal=proposal, expected_scene=scene)
        reports[scene] = {"sha256": loaded["sha256"], "resumed": False}
    return seal_overlay_collection(ctx, role, reports)


def seal_overlay_collection(ctx: ReadyContext, role: str, reports: Mapping[str, Any] | None = None) -> dict[str, Any]:
    verified = ctx.roles[role]
    rows, near = [], 0
    for scene in verified.scenes:
        proposal = load_proposal(_proposal_path(ctx.config, role, scene), expected_scene=scene)
        overlay = load_overlay(_overlay_path(ctx.config, role, scene), proposal=proposal, expected_scene=scene)
        near += overlay["summary"].near_candidate_count
        rows.append({"scene_id": scene, "path": str(overlay["path"]), "sha256": overlay["sha256"], "near_candidate_count": overlay["summary"].near_candidate_count})
    payload = {
        "schema": OVERLAY_COLLECTION_SCHEMA, "complete": True, "create_only": True,
        "role": role, "fold_id": verified.heldout_fold, "scene_count": 20,
        "near_candidate_count": near, "cpu_only": True, "ground_truth_access": False,
        "anchor_geometry_source": "sealed_final_base_prediction",
        "anchor_score_source": "ca1m_native_b6_final_base_all_fold_oof_row_scores_v2",
        "deployment_scores_used": False, "scenes": rows,
    }
    manifest = ensure_directory(Path(ctx.config["outputs"]["manifest_root"])) / f"O_{role}_exact20.json"
    create_or_verify(manifest, canonical_json(payload), f"{role} overlay manifest")
    return {**payload, "manifest": _record_dict(manifest, OVERLAY_COLLECTION_SCHEMA), "reports": reports or {}}


def _candidate_diagnostic_path(cfg: Mapping[str, Any], role: str, scene: str) -> Path:
    return Path(cfg["outputs"]["candidate_diagnostic_root"]) / role / f"{scene}_ca1m_native_b6.npz"


def _evidence_path(cfg: Mapping[str, Any], role: str, scene: str) -> Path:
    return Path(cfg["outputs"]["evidence_root"]) / role / f"{scene}_ca1m_tr3d_candidate_evidence_xfit_r2_v5.npz"


def _fresh_candidate_native(
    ctx: ReadyContext, role: str, scene: str, proposal: Mapping[str, Any],
) -> tuple[np.ndarray, Path]:
    target = _candidate_diagnostic_path(ctx.config, role, scene)
    from .ca1m_native_b6_score import load_native_observer_diagnostic
    if not (target.exists() or target.is_symlink()):
        from PIL import Image
        from .ca1m_native_b6_observer import CA1MNativeB6Config, CA1MNativeB6Observer
        from tools.build_ca1m_tr3d_candidate_evidence_v4 import _scene_metadata
        scene_root = Path(ctx.config["candidate_inputs"]["processed_rgbd_root"]) / scene
        depth, poses, intrinsics = _scene_metadata(scene_root)
        with tempfile.TemporaryDirectory(prefix="e961_candidate_native_") as temporary:
            observer = CA1MNativeB6Observer(CA1MNativeB6Config(
                enabled=True, diagnostics_root=temporary, top_k=5, pixel_stride=4,
                margin=0.05, min_depth=0.10, max_depth=8.0, near_clip=1e-3,
                max_cached_keyframes=256,
            ))
            for frame in np.asarray(proposal["used_frame_ids"], dtype=np.int64).tolist():
                if frame >= len(poses) or frame not in depth:
                    raise ValueError(f"{scene}: proposal frame lineage is unavailable")
                depth_m = np.asarray(Image.open(depth[frame]), dtype=np.float32) / 1000.0
                observer.record_keyframe(
                    scene_id=scene, frame_id=frame, source_frame_id=str(frame),
                    depth_meters=depth_m, intrinsics=intrinsics[frame],
                    camera_to_world=poses[frame],
                )
            summary = observer.finalize(
                scene_id=scene, corners=proposal["candidate_corners_world"],
                scores=proposal["candidate_scores"],
                stable_ids=np.arange(len(proposal["candidate_scores"]), dtype=np.int64),
            )
            produced = Path(summary.diagnostic_path)
            write_bytes_exclusive(target, stable_bytes(produced, "fresh candidate-native diagnostic"))
    if target.is_symlink() or os.stat(target, follow_symlinks=False).st_mode & 0o222:
        raise ValueError(f"{scene}: candidate-native resume artifact is not immutable")
    evidence = load_native_observer_diagnostic(
        target, scene_id=scene, corners=proposal["candidate_corners_world"],
        scores=proposal["candidate_scores"],
    )
    features = np.ascontiguousarray(evidence["features"], dtype=np.float32)
    if not np.isfinite(features).all() or np.any((features < 0) | (features > 1)):
        raise ValueError(f"{scene}: candidate-native evidence is non-finite")
    return features, target


def _write_generic_evidence(
    path: Path, *, scene: str, role: str, ctx: ReadyContext,
    normalized_receipt: Path, proposal: Mapping[str, Any], overlay: Mapping[str, Any],
    candidate_native: np.ndarray,
) -> Path:
    from .ca1m_tr3d_terminal_gate_v4 import build_terminal_gate_features_v4
    from .ca1m_tr3d_terminal_gate_v5 import _evidence_payload, load_candidate_evidence_v5
    batch = build_terminal_gate_features_v4(
        proposal=proposal, overlay=overlay,
        anchor_native_evidence=overlay["anchor_native_features"],
        anchor_native_detector_scores=overlay["anchor_detector_scores"],
        candidate_native_evidence=candidate_native,
        anchor_scores=overlay["anchor_scores_oof"],
        score_source="ca1m_native_b6_final_base_all_fold_oof_row_scores_v2",
    )
    payload = _evidence_payload(
        scene_id=scene, fold_id=ctx.roles[role].heldout_fold, producer_role=role,
        producer_checkpoint_sha256=ctx.roles[role].checkpoint_sha256,
        training_receipt_sha256=sha256_file(normalized_receipt),
        outer_continuation_receipt_sha256=ctx.continuation_sha256,
        b6_oof_sidecar_sha256=sha256_file(Path(ctx.config["anchor_inputs"]["native_b6_oof_sidecar"]["path"])),
        candidate_corners=proposal["candidate_corners_world"][batch.candidate_rows],
        candidate_rows=batch.candidate_rows, candidate_scores=batch.candidate_scores,
        anchor_indices=batch.anchor_indices, features=batch.features,
        anchor_corners=overlay["anchor_corners"], anchor_scores=overlay["anchor_scores_oof"],
    )
    if path.exists() or path.is_symlink():
        existing = load_candidate_evidence_v5(path, expected_scene=scene)
        checks = {
            "candidate_corners": payload["candidate_corners"],
            "candidate_rows": payload["candidate_rows"],
            "candidate_scores": payload["candidate_scores"],
            "anchor_indices": payload["anchor_indices"],
            "features": payload["features"],
            "anchor_corners": payload["anchor_corners"],
            "anchor_scores": payload["anchor_scores_oof"],
        }
        for name, expected in checks.items():
            if not np.array_equal(getattr(existing, name), expected):
                raise ValueError(f"{scene}: resumed evidence {name} differs")
    else:
        write_bytes_exclusive(path, _npz_bytes(payload))
        load_candidate_evidence_v5(path, expected_scene=scene)
    return path


def _npz_bytes(value: Mapping[str, Any]) -> bytes:
    stream = BytesIO(); np.savez_compressed(stream, **value); return stream.getvalue()


def run_stage_e(ctx: ReadyContext, role: str) -> dict[str, Any]:
    """Recompute candidate-native evidence and materialize 40-D v5 rows."""

    if role not in ROLE_SPECS:
        raise ValueError("unknown role")
    normalized = ensure_runtime_namespace(ctx)[role]
    seal_proposal_collection(ctx, role)
    seal_overlay_collection(ctx, role)
    verified = ctx.roles[role]
    diagnostic_root = ensure_directory(Path(ctx.config["outputs"]["candidate_diagnostic_root"]) / role)
    evidence_root = ensure_directory(Path(ctx.config["outputs"]["evidence_root"]) / role)
    expected_diagnostics = {f"{scene}_ca1m_native_b6.npz" for scene in verified.scenes}
    expected_evidence = {f"{scene}_ca1m_tr3d_candidate_evidence_xfit_r2_v5.npz" for scene in verified.scenes}
    _exact_inventory(diagnostic_root, expected_diagnostics, ".npz", f"{role} candidate-native root")
    _exact_inventory(evidence_root, expected_evidence, ".npz", f"{role} evidence root")
    from .ca1m_tr3d_terminal_gate_v5 import load_candidate_evidence_v5
    rows, resumed = [], 0
    for scene in verified.scenes:
        proposal = load_proposal(_proposal_path(ctx.config, role, scene), expected_scene=scene)
        overlay = load_overlay(_overlay_path(ctx.config, role, scene), proposal=proposal, expected_scene=scene)
        native, diagnostic = _fresh_candidate_native(ctx, role, scene, proposal)
        target = _evidence_path(ctx.config, role, scene)
        existed = target.exists() or target.is_symlink()
        _write_generic_evidence(
            target, scene=scene, role=role, ctx=ctx, normalized_receipt=normalized,
            proposal=proposal, overlay=overlay, candidate_native=native,
        )
        evidence = load_candidate_evidence_v5(target, expected_scene=scene)
        if os.stat(target, follow_symlinks=False).st_mode & 0o222:
            raise ValueError(f"{scene}: evidence resume artifact is writable")
        if (
            evidence.producer_role != role or evidence.fold_id != verified.heldout_fold
            or evidence.producer_checkpoint_sha256 != verified.checkpoint_sha256
            or evidence.training_receipt_sha256 != sha256_file(normalized)
            or evidence.b6_oof_sidecar_sha256 != sha256_file(Path(ctx.config["anchor_inputs"]["native_b6_oof_sidecar"]["path"]))
        ):
            raise ValueError(f"{scene}: candidate evidence provenance differs")
        resumed += int(existed)
        rows.append({
            "scene_id": scene, "path": str(target), "sha256": sha256_file(target),
            "candidate_count": len(evidence.candidate_rows),
            "candidate_native_diagnostic": _record_dict(diagnostic),
        })
    payload = {
        "schema": "boxfusion.ca1m_tr3d_e961_candidate_native_collection.v5.r2",
        "complete": True, "create_only": True, "role": role,
        "fold_id": verified.heldout_fold, "scene_count": 20,
        "fresh_candidate_native": True, "ground_truth_access": False,
        "anchor_score_source": "ca1m_native_b6_final_base_all_fold_oof_row_scores_v2",
        "normalized_receipt": _record_dict(normalized, "boxfusion.ca1m_tr3d_xfit_r2_detector_role_receipt.v1"),
        "scenes": rows,
    }
    manifest = ensure_directory(Path(ctx.config["outputs"]["manifest_root"])) / f"E_{role}_exact20.json"
    create_or_verify(manifest, canonical_json(payload), f"{role} candidate-native manifest")
    return {**payload, "resumed_count": resumed, "manifest": _record_dict(manifest, payload["schema"])}


def seal_stage_m(ctx: ReadyContext) -> dict[str, Any]:
    """Seal the terminal-gate-v5-compatible exact fit60 + reused-dev20 input."""

    normalized = ensure_runtime_namespace(ctx)
    from . import ca1m_tr3d_terminal_gate_v5 as gate_v5
    manifest_root = ensure_directory(Path(ctx.config["outputs"]["manifest_root"]))
    b6_path = Path(ctx.config["anchor_inputs"]["native_b6_oof_sidecar"]["path"])
    b6_record = {
        "path": str(b6_path), "sha256": sha256_file(b6_path),
        "schema": "boxfusion.ca1m_native_b6_oof_row_scores.v2",
        "score_source": "ca1m_native_b6_final_base_all_fold_oof_row_scores_v2",
        "each_row_model_excludes_scene": True, "deploy_scores": False,
    }
    generic_order = ("inner_holdout2", "inner_holdout3", "inner_holdout4", "outer_dev")
    role_records: dict[str, dict[str, Any]] = {}
    role_payloads: dict[str, dict[str, Any]] = {}
    for role in generic_order:
        target = manifest_root / f"M_{role}_candidate_collection_exact20.json"
        evidence_root = Path(ctx.config["outputs"]["evidence_root"]) / role
        expected_evidence = {
            f"{scene}_ca1m_tr3d_candidate_evidence_xfit_r2_v5.npz"
            for scene in ctx.roles[role].scenes
        }
        _exact_inventory(evidence_root, expected_evidence, ".npz", f"{role} M evidence root")
        receipt_record = {
            "path": str(normalized[role]), "sha256": sha256_file(normalized[role]),
            "schema": gate_v5.ROLE_RECEIPT_SCHEMA,
        }
        _, receipt = gate_v5.load_detector_role_receipt_v5(receipt_record, role)
        rows, candidate_count, anchor_count = [], 0, 0
        for scene in ctx.roles[role].scenes:
            evidence_path = _evidence_path(ctx.config, role, scene)
            evidence = gate_v5.load_candidate_evidence_v5(evidence_path, expected_scene=scene)
            if (
                evidence.producer_role != role
                or evidence.fold_id != ctx.roles[role].heldout_fold
                or evidence.producer_checkpoint_sha256 != ctx.roles[role].checkpoint_sha256
                or evidence.training_receipt_sha256 != sha256_file(normalized[role])
                or evidence.outer_continuation_receipt_sha256 != ctx.continuation_sha256
                or evidence.b6_oof_sidecar_sha256 != b6_record["sha256"]
            ):
                raise ValueError(f"{scene}: M evidence/receipt provenance differs")
            candidate_count += len(evidence.candidate_rows)
            anchor_count += len(evidence.anchor_rows)
            rows.append({
                "schema": gate_v5.SCENE_MANIFEST_SCHEMA, "scene_id": scene,
                "fold_id": ctx.roles[role].heldout_fold, "producer_role": role,
                "producer_checkpoint_sha256": evidence.producer_checkpoint_sha256,
                "producer_train_folds": list(ctx.roles[role].train_folds),
                "path": str(evidence_path), "sha256": sha256_file(evidence_path),
                "candidate_count": len(evidence.candidate_rows),
                "anchor_count": len(evidence.anchor_rows),
                "candidate_corners_sha256": evidence.candidate_corners_sha256,
                "candidate_feature_sha256": evidence.candidate_feature_sha256,
                "anchor_identity_sha256": evidence.anchor_identity_sha256,
            })
        role_payload = {
            "schema": gate_v5.ROLE_COLLECTION_SCHEMA, "complete": True,
            "create_only": True, "ground_truth_access": False,
            "fold1_access": False, "official_validation_access": False,
            "role": role, "detector_train_folds": list(ctx.roles[role].train_folds),
            "candidate_output_fold": ctx.roles[role].heldout_fold,
            "scene_count": 20, "candidate_count": candidate_count,
            "anchor_count": anchor_count, "candidate_geometry_oof": True,
            "checkpoint_selection": False, "role_receipt": receipt_record,
            "b6_oof_sidecar": b6_record, "scenes": rows,
        }
        # The generic loader below reopens the producer adapter, continuation,
        # checkpoint, every evidence NPZ, and all hashes.
        del receipt
        create_or_verify(target, canonical_json(role_payload), f"{role} role collection")
        role_payloads[role] = role_payload
        role_records[role] = {
            "path": str(target), "sha256": sha256_file(target),
            "schema": gate_v5.ROLE_COLLECTION_SCHEMA,
        }
    combined = Path(ctx.config["outputs"]["combined_manifest"])
    scene_rows = [row for role in generic_order for row in role_payloads[role]["scenes"]]
    fold_counts = {
        str(fold): sum(int(row["fold_id"]) == fold for row in scene_rows)
        for fold in (0, 2, 3, 4)
    }
    combined_payload = {
        "schema": gate_v5.COLLECTION_SCHEMA, "complete": True, "create_only": True,
        "namespace": gate_v5.NAMESPACE, "scene_grouped": True,
        "candidate_geometry_oof_for_every_fit_scene": True,
        "fit_scene_count": 60, "outer_scene_count": 20, "scene_count": 80,
        "fold_counts": fold_counts, "fit_folds": [2, 3, 4], "outer_folds": [0],
        "ground_truth_access": False, "fold1_access": False,
        "official_validation_access": False, "deploy_candidate_or_anchor_scores": False,
        "legacy_v4_candidate_or_policy_reused": False,
        "roles": [{
            "role": role, **role_records[role],
            "detector_train_folds": list(ctx.roles[role].train_folds),
            "candidate_output_fold": ctx.roles[role].heldout_fold, "scene_count": 20,
        } for role in generic_order],
        "b6_oof_sidecar": b6_record,
        "outer_continuation_receipt_sha256": ctx.continuation_sha256,
        "scenes": sorted(scene_rows, key=lambda row: (int(row["fold_id"]), str(row["scene_id"]))),
    }
    create_or_verify(combined, canonical_json(combined_payload), "exact80 candidate collection")
    loaded = gate_v5.load_candidate_collection_v5(combined)
    if set(loaded.scenes) != set().union(*(set(ctx.roles[role].scenes) for role in ROLE_ORDER)):
        raise ValueError("exact80 manifest scene identity differs")
    wrapper = {
        "schema": "boxfusion.ca1m_tr3d_e961_terminal_inputs_exact80_receipt.v5.r2",
        "complete": True, "create_only": True, "namespace": NAMESPACE,
        "fit_scene_count": 60, "fit_folds": [2, 3, 4],
        "reused_dev_scene_count": 20, "reused_dev_folds": [0],
        "scene_count": 80, "each_scene_detector_excludes_scene": True,
        "b6_score_source": "all_fold_oof_each_row_model_excludes_scene",
        "ground_truth_access": False, "fold1_access": False,
        "official_validation_access": False,
        "candidate_collection": _record_dict(combined, gate_v5.COLLECTION_SCHEMA),
        "authorization": _record_dict(ctx.authorization_path, AUTH_SCHEMA),
    }
    wrapper_path = manifest_root / "M_EXACT80_R2_RECEIPT.json"
    create_or_verify(wrapper_path, canonical_json(wrapper), "E961 exact80 wrapper receipt")
    return {**wrapper, "receipt": _record_dict(wrapper_path, wrapper["schema"])}


def run_all(ctx: ReadyContext, *, device: str = "cuda:0") -> dict[str, Any]:
    results: dict[str, Any] = {}
    for role in ROLE_ORDER:
        results[f"P:{role}"] = run_stage_p(ctx, role, device=device)
        results[f"O:{role}"] = run_stage_o(ctx, role)
        results[f"E:{role}"] = run_stage_e(ctx, role)
    results["M"] = seal_stage_m(ctx)
    return {
        "schema": "boxfusion.ca1m_tr3d_e961_terminal_inputs_run.v5.r2",
        "complete": True, "ground_truth_access": False, "results": results,
    }


__all__ = [
    "DEFAULT_CONFIG", "PENDING_SCHEMA", "READY_SCHEMA", "AUTH_SCHEMA", "NAMESPACE",
    "OUTER_SCHEMA", "INNER_SCHEMA", "LEGACY_INNER_SCHEMA", "ROLE_ORDER", "ROLE_SPECS",
    "PendingOperationalInputs", "VerifiedRole", "ReadyContext", "ProposalSummaryR2",
    "OverlaySummaryR2", "validate_static_config", "validate_operational_ready",
    "ensure_runtime_namespace", "proposal_payload", "load_proposal", "overlay_payload",
    "load_overlay", "run_stage_p", "run_stage_o", "run_stage_e", "seal_stage_m", "run_all",
    "write_bytes_exclusive", "write_json_exclusive", "write_npz_exclusive", "sha256_file",
]
