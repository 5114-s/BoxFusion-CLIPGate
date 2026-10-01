"""R5 canonical-authority boundary for CA-only E961 terminal inputs.

R5 supersedes the never-authorized R4 preregistration.  Every stage rebuilds
its authority and R2 snapshot from the canonical READY/AUTH/BUNDLE/PREREG
files; the context carries only a private registry token and canonical writer
and parent descriptors.  No context-provided path, digest, role, continuation,
or R2 module record is used as authority.  Import is side-effect free and no
code here opens annotations or ground truth.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import stat
import struct
import sys
from typing import Any, Iterator, Mapping

from . import ca1m_tr3d_e961_terminal_inputs_v5_r3 as r3
from . import ca1m_tr3d_e961_terminal_inputs_v5_r4 as r4


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/ca1m_tr3d_e961_terminal_inputs_v5_r5_pending.json"
MANIFEST_ROOT = ROOT / "manifests/ca1m_tr3d_e961_terminal_inputs_v5_r5"
PREREGISTRATION_PATH = MANIFEST_ROOT / "PREREGISTRATION.json"
READY_CONFIG_PATH = MANIFEST_ROOT / "READY_CONFIG.json"
RUN_AUTHORIZATION_PATH = MANIFEST_ROOT / "RUN_AUTHORIZATION.json"
AUTHORIZATION_BUNDLE_PATH = MANIFEST_ROOT / "AUTHORIZATION_BUNDLE.json"
R4_INVALID_PATH = (
    ROOT / "manifests/ca1m_tr3d_e961_terminal_inputs_v5_r4/PREREGISTRATION_INVALID.json"
)

CONFIG_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_config.v5.r5"
PREREGISTRATION_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_preregistration.v5.r5"
R4_INVALID_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_preregistration_invalid.v5.r4"
AUTH_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_run_authorization.v5.r5"
BUNDLE_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_authorization_bundle.v5.r5"
EXACT80_RECEIPT_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_exact80_receipt.v5.r5"
STATIC_REPORT_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_static_preflight.v5.r5"
OPERATIONAL_REPORT_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_operational_preflight.v5.r5"
NAMESPACE = "ca1m_tr3d_e961_terminal_inputs_xfit_r2_v5_r5"
ROLE_ORDER = r3.ROLE_ORDER
ROLE_SPECS = r3.ROLE_SPECS
OUTER_SCHEMA = r3.OUTER_SCHEMA
INNER_SCHEMA = r3.INNER_SCHEMA
CONTINUATION_SCHEMA = r3.CONTINUATION_SCHEMA
LEGACY_INNER_SCHEMA = r3.LEGACY_INNER_SCHEMA
AUTHORIZATIONS = copy.deepcopy(r3.AUTHORIZATIONS)

canonical_json = r3.canonical_json
stable_bytes = r3.stable_bytes
sha256_file = r3.sha256_file
sha256_bytes = r3.sha256_bytes


class PendingOperationalInputs(PermissionError):
    pass


def _pending(schema: str, *, bundle: bool = False) -> dict[str, Any]:
    if bundle:
        return {"state": "pending", "path": None, "commit_id": None, "schema": schema}
    return {"state": "pending", "path": None, "sha256": None, "schema": schema}


def _record(path: Path, schema: str) -> dict[str, Any]:
    return {"state": "bound", "path": os.fspath(path), "sha256": sha256_file(path), "schema": schema}


def _json(path: Path, name: str) -> tuple[Path, dict[str, Any]]:
    return r3.read_json(path, name)


def _file_record(record: Any, name: str, schema: str | None = None) -> tuple[Path, dict[str, Any] | None]:
    return r3._file_record(record, name, schema=schema)


def _identity(path: Path) -> tuple[int, int, int, int, int]:
    parent_fd, parent_ids = r3._open_dir_chain(path.parent, f"{path.name} parent")
    try:
        info = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
    finally:
        os.close(parent_fd)
    r3._verify_dir_chain(path.parent, parent_ids, f"{path.name} parent")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_nlink


def _load_pending() -> tuple[Path, dict[str, Any]]:
    path, cfg = _json(DEFAULT_CONFIG, "R5 frozen pending config")
    if path != DEFAULT_CONFIG or cfg.get("schema") != CONFIG_SCHEMA or cfg.get("namespace") != NAMESPACE:
        raise ValueError("R5 pending config identity differs")
    return path, cfg


def _load_base(cfg: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path, base = _file_record(cfg.get("base_static_contract"), "R3 base static contract", r3.CONFIG_SCHEMA)
    if path != r3.DEFAULT_CONFIG:
        raise ValueError("R5 base static contract is noncanonical")
    # The R3 JSON freezes all semantic paths, but downstream pipeline modules
    # can legitimately advance before R5 is sealed.  Validate those exact
    # paths against a live SHA inventory which R5 preregisters independently.
    current = copy.deepcopy(base)
    for record in current["implementation"].values():
        record["sha256"] = sha256_file(Path(record["path"]))
    r3._validate_config_shape(current)
    return path, current


def _expected_outputs() -> dict[str, str]:
    root = Path("/extra/ZhaoX") / NAMESPACE
    return {
        "namespace_root": os.fspath(root),
        "proposal_root": os.fspath(root / "P_anchor_free"),
        "overlay_root": os.fspath(root / "O_oof_overlay"),
        "candidate_diagnostic_root": os.fspath(root / "E_candidate_native"),
        "evidence_root": os.fspath(root / "evidence"),
        "receipt_root": os.fspath(root / "normalized_receipts"),
        "manifest_root": os.fspath(root / "manifests"),
        "combined_manifest": os.fspath(root / "manifests/CANDIDATE_COLLECTION_EXACT80.json"),
    }


def _validate_shape(cfg: Mapping[str, Any], *, require_invalidation: bool = True) -> dict[str, Any]:
    if cfg.get("schema") != CONFIG_SCHEMA or cfg.get("namespace") != NAMESPACE:
        raise ValueError("R5 schema/namespace differs")
    if cfg.get("preregistration") != {"path": os.fspath(PREREGISTRATION_PATH), "schema": PREREGISTRATION_SCHEMA}:
        raise ValueError("R5 preregistration target differs")
    if cfg.get("sealed_dynamic_outputs") != {
        "ready_config": os.fspath(READY_CONFIG_PATH),
        "run_authorization": os.fspath(RUN_AUTHORIZATION_PATH),
        "authorization_bundle": os.fspath(AUTHORIZATION_BUNDLE_PATH),
    }:
        raise ValueError("R5 dynamic targets differ")
    _, base = _load_base(cfg)
    predecessor = cfg.get("invalidated_predecessor") or {}
    predecessor_path, _ = _file_record(
        predecessor.get("preregistration"), "R4 preregistration", r4.PREREGISTRATION_SCHEMA,
    )
    if predecessor_path != r4.PREREGISTRATION_PATH:
        raise ValueError("R4 predecessor preregistration is noncanonical")
    invalid = predecessor.get("invalidation")
    if require_invalidation:
        invalid_path, invalid_value = _file_record(invalid, "R4 invalidation", R4_INVALID_SCHEMA)
        if invalid_path != R4_INVALID_PATH or invalid_value.get("invalid") is not True:
            raise ValueError("R4 invalidation differs")
    elif invalid != _pending(R4_INVALID_SCHEMA):
        raise ValueError("R4 invalidation must be pending before create-only invalidation")
    receipts = cfg.get("producer_success_receipts") or {}
    if tuple(receipts) != ROLE_ORDER:
        raise ValueError("R5 producer role order differs")
    if cfg.get("outputs") != _expected_outputs():
        raise ValueError("R5 output paths differ")
    implementation = cfg.get("implementation") or {}
    required = {
        "current_core", "current_sealer", "current_preflight", "current_runner",
        "current_tests", "r4_static_helper", "r3_static_helper", "r2_execution_core",
    }
    if set(implementation) != required:
        raise ValueError("R5 implementation inventory differs")
    for key, record in implementation.items():
        _file_record(record, f"R5 implementation {key}")
    if cfg.get("access") != base.get("access"):
        raise ValueError("R5 access contract differs")
    return base


def _ready_delta(pending: Mapping[str, Any], ready: Mapping[str, Any]) -> dict[str, Any]:
    expected = copy.deepcopy(pending)
    result: dict[str, Any] = {}
    for role in ROLE_ORDER:
        value = (ready.get("producer_success_receipts") or {}).get(role)
        if not isinstance(value, Mapping) or value.get("state") != "bound":
            raise PendingOperationalInputs(f"{role} receipt pending")
        expected["producer_success_receipts"][role] = copy.deepcopy(value)
        result[role] = value
    continuation = ready.get("continuation_receipt")
    if not isinstance(continuation, Mapping) or continuation.get("state") != "bound":
        raise PendingOperationalInputs("continuation receipt pending")
    expected["continuation_receipt"] = copy.deepcopy(continuation)
    result["continuation_receipt"] = continuation
    authorization = ready.get("run_authorization")
    if not isinstance(authorization, Mapping) or authorization.get("state") != "committed_by_bundle":
        raise PendingOperationalInputs("authorization bundle pending")
    expected["run_authorization"] = copy.deepcopy(authorization)
    result["run_authorization"] = authorization
    if expected != ready:
        raise PermissionError("R5 ready differs outside the exact six dynamic fields")
    return result


def r4_invalidation_payload() -> dict[str, Any]:
    predecessor = r4.PREREGISTRATION_PATH
    expected_sha = "ff80299556e684f120857a23ad708f419101ad47925327872bc5c41df5c7918d"
    if sha256_file(predecessor) != expected_sha:
        raise ValueError("R4 predecessor changed")
    for forbidden in (
        r4.READY_CONFIG_PATH, r4.RUN_AUTHORIZATION_PATH,
        r4.AUTHORIZATION_BUNDLE_PATH, Path("/extra/ZhaoX") / r4.NAMESPACE,
    ):
        if forbidden.exists() or forbidden.is_symlink():
            raise PermissionError("cannot invalidate R4 after operational output")
    return {
        "schema": R4_INVALID_SCHEMA, "complete": True, "create_only": True,
        "invalid": True, "operational_authority": False,
        "predecessor": {"path": os.fspath(predecessor), "sha256": expected_sha, "schema": r4.PREREGISTRATION_SCHEMA},
        "audit_result": "CODE_BLOCK",
        "reasons": [
            "r2_module_path_sha_identity_could_be_replaced_as_a_self_consistent_context_triple",
            "ready_config_path_payload_and_r2_context_could_be_replaced_as_a_self_consistent_group",
            "roles_binding_and_r2_context_roles_could_be_replaced_as_a_self_consistent_group",
            "continuation_path_and_sha_could_be_replaced_as_a_self_consistent_group",
            "parent_and_writer_fd_path_identity_groups_were_not_reanchored_to_fixed_namespace_paths",
        ],
        "superseded_by_namespace": NAMESPACE,
        "ready_config_created": False, "run_authorization_created": False,
        "authorization_bundle_created": False,
        "runtime_namespace_created": False, "gpu_started": False,
        "ground_truth_access": False,
    }


def seal_r4_invalidation() -> Path:
    payload = r4_invalidation_payload()
    r3._host_target_probe(R4_INVALID_PATH.parent)
    fd = r3._claim_writer(R4_INVALID_PATH.parent, ".R4_INVALID.writer.claim", sha256_bytes(canonical_json(payload)))
    try:
        return r3._exclusive_bytes(R4_INVALID_PATH, canonical_json(payload))
    finally:
        os.close(fd)


def build_preregistration_payload() -> dict[str, Any]:
    source, cfg = _load_pending()
    base = _validate_shape(cfg)
    _, predecessor_base = _file_record(
        cfg["base_static_contract"], "R3 predecessor base static contract", r3.CONFIG_SCHEMA,
    )
    assert predecessor_base is not None
    predecessor_implementation = predecessor_base["implementation"]
    dependency_changes = [{
        "name": name,
        "path": base["implementation"][name]["path"],
        "frozen_r3_base_sha256": predecessor_implementation[name]["sha256"],
        "r5_preregistered_sha256": base["implementation"][name]["sha256"],
    } for name in sorted(base["implementation"])
        if predecessor_implementation[name]["sha256"] != base["implementation"][name]["sha256"]]
    if [item["name"] for item in dependency_changes] != ["v5_manifest_runtime"]:
        raise ValueError("R3-to-R5 dependency drift inventory differs")
    for role in ROLE_ORDER:
        schema = OUTER_SCHEMA if role == "outer_dev" else INNER_SCHEMA
        if cfg["producer_success_receipts"][role] != _pending(schema):
            raise ValueError(f"{role} is not pending")
    if cfg.get("continuation_receipt") != _pending(CONTINUATION_SCHEMA):
        raise ValueError("R5 continuation is not pending")
    if cfg.get("run_authorization") != _pending(BUNDLE_SCHEMA, bundle=True):
        raise ValueError("R5 bundle authorization is not pending")
    inventory = r3.processed_point_inventory(base)
    invalid_path, _ = _file_record(
        cfg["invalidated_predecessor"]["invalidation"], "R4 invalidation", R4_INVALID_SCHEMA,
    )
    return {
        "schema": PREREGISTRATION_SCHEMA, "complete": True, "create_only": True,
        "static_only": True, "namespace": NAMESPACE,
        "pending_config": {"path": os.fspath(source), "sha256": sha256_file(source), "schema": CONFIG_SCHEMA},
        "base_static_contract": copy.deepcopy(cfg["base_static_contract"]),
        "base_execution_dependencies": copy.deepcopy(base["implementation"]),
        "inherited_base_dependency_relationship": {
            "r4_predecessor_preregistration_invalid": True,
            "frozen_r3_base_config_sha256": sha256_file(r3.DEFAULT_CONFIG),
            "r5_rehashes_every_canonical_dependency": True,
            "changed_dependencies": dependency_changes,
        },
        "r4_invalidation": {"path": os.fspath(invalid_path), "sha256": sha256_file(invalid_path), "schema": R4_INVALID_SCHEMA},
        "implementation": copy.deepcopy(cfg["implementation"]),
        "canonical_paths": {
            **copy.deepcopy(cfg["sealed_dynamic_outputs"]),
            "continuation_receipt": os.fspath(r3.CONTINUATION_CANONICAL_PATH),
            "outer_receipt_root": base["producer_contracts"]["outer"]["canonical_root"],
            "inner_receipt_root": base["producer_contracts"]["inner"]["canonical_root"],
            "namespace_root": cfg["outputs"]["namespace_root"],
        },
        "processed_point_inventory": inventory,
        "ready_delta": {
            "only_six_fields": [
                *[f"producer_success_receipts.{role}" for role in ROLE_ORDER],
                "continuation_receipt", "run_authorization",
            ],
        },
        "runtime_invariants": {
            "persistent_writer_fd_claim_inode_lock": True,
            "writer_fd_open_description_lock": True,
            "persistent_parent_dirfd_devino": True,
            "canonical_r2_config_bytes_per_stage": True,
            "fresh_sha_bound_r2_module_per_stage": True,
            "bundle_is_last_and_only_commit_gate": True,
            "ready_auth_leaf_replay_before_bundle": True,
            "fresh_canonical_authority_rederived_per_stage": True,
            "context_fields_are_not_operational_authority": True,
            "canonical_authority_registry_is_outside_context": True,
            "r2_record_is_reanchored_to_ready_and_preregistration": True,
            "stage_uses_fresh_derived_r2_snapshot": True,
            "fixed_namespace_parent_and_writer_claim_paths": True,
        },
        "access": {"gpu_started": False, "ground_truth_access": False, "fold1_access": False, "official_validation_access": False},
    }


def validate_preregistration() -> tuple[Path, dict[str, Any]]:
    path, value = _json(PREREGISTRATION_PATH, "R5 preregistration")
    expected = build_preregistration_payload()
    if value != expected:
        raise ValueError("R5 preregistration/static inputs drifted")
    return path, value


def seal_preregistration() -> Path:
    payload = build_preregistration_payload()
    r3._host_target_probe(MANIFEST_ROOT.parent)
    r3.ensure_directory(MANIFEST_ROOT)
    fd = r3._claim_writer(MANIFEST_ROOT, ".PREREGISTRATION.writer.claim", sha256_bytes(canonical_json(payload)))
    try:
        return r3._exclusive_bytes(PREREGISTRATION_PATH, canonical_json(payload))
    finally:
        os.close(fd)


def validate_static_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    if Path(path) != DEFAULT_CONFIG:
        raise ValueError("R5 static config path is noncanonical")
    source, cfg = _load_pending()
    _validate_shape(cfg)
    prereg_path, prereg = validate_preregistration()
    return {
        "schema": STATIC_REPORT_SCHEMA, "status": "PASS_STATIC_PENDING",
        "config": {"path": os.fspath(source), "sha256": sha256_file(source)},
        "preregistration": {"path": os.fspath(prereg_path), "sha256": sha256_file(prereg_path)},
        "processed_point_inventory_sha256": prereg["processed_point_inventory"]["inventory_sha256"],
        "scene_count": 80, "operational_authorized": False,
        "output_created": False, "gpu_started": False, "ground_truth_access": False,
    }


def _exec_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load bound module {name}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    existed = name in sys.modules
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if existed:
            sys.modules[name] = previous
        else:
            sys.modules.pop(name, None)
    if Path(module.__file__).resolve() != path.resolve():
        raise ImportError(f"{name} __file__ differs")
    return module


def load_bound_producer_verifiers(base: Mapping[str, Any]) -> tuple[Any, Any]:
    """Load inner with its bound outer dependency temporarily injected.

    No sys.path entry is added.  Both the expected short outer alias and the
    unique temporary module entries are restored before this function returns.
    """

    outer_record = base["producer_contracts"]["outer"]["verifier"]
    inner_record = base["producer_contracts"]["inner"]["verifier"]
    outer_path, _ = _file_record(outer_record, "official outer verifier")
    inner_path, _ = _file_record(inner_record, "official inner verifier")
    nonce = secrets.token_hex(12)
    outer_name = f"_r4_bound_outer_{nonce}"
    inner_name = f"_r4_bound_inner_{nonce}"
    outer = _exec_module(outer_path, outer_name)
    alias = "tr3d_ca1m_e961_outer_train_r2"
    alias_existed, alias_previous = alias in sys.modules, sys.modules.get(alias)
    unique_existed, unique_previous = inner_name in sys.modules, sys.modules.get(inner_name)
    spec = importlib.util.spec_from_file_location(inner_name, inner_path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load official inner verifier")
    inner = importlib.util.module_from_spec(spec)
    sys.modules[alias] = outer
    sys.modules[inner_name] = inner
    try:
        spec.loader.exec_module(inner)
    finally:
        if alias_existed:
            sys.modules[alias] = alias_previous
        else:
            sys.modules.pop(alias, None)
        if unique_existed:
            sys.modules[inner_name] = unique_previous
        else:
            sys.modules.pop(inner_name, None)
    if (
        Path(inner.__file__).resolve() != inner_path.resolve()
        or getattr(inner, "outer", None) is not outer
        or not callable(getattr(outer, "verify_success_receipt", None))
        or not callable(getattr(inner, "verify_success_receipt", None))
        or sha256_file(outer_path) != outer_record["sha256"]
        or sha256_file(inner_path) != inner_record["sha256"]
    ):
        raise ImportError("bound outer/inner verifier dependency identity differs")
    return outer, inner


def _bound_receipt(path: Path, schema: str, name: str) -> dict[str, Any]:
    path = Path(path).absolute()
    _, value = _json(path, name)
    if value.get("schema") != schema:
        raise ValueError(f"{name} schema differs")
    return {"state": "bound", "path": os.fspath(path), "sha256": sha256_file(path), "schema": schema}


def _verify_dynamic(ready: Mapping[str, Any]) -> tuple[Path, dict[str, r3.VerifiedRole]]:
    _, pending = _load_pending()
    dynamic = _ready_delta(pending, ready)
    base = _validate_shape(ready)
    validate_preregistration()
    continuation_path, _ = r3._verify_continuation(dynamic["continuation_receipt"])
    continuation_sha = sha256_file(continuation_path)
    outer, inner = load_bound_producer_verifiers(base)
    roles: dict[str, r3.VerifiedRole] = {}
    for role in ROLE_ORDER:
        record = dynamic[role]
        schema = OUTER_SCHEMA if role == "outer_dev" else INNER_SCHEMA
        producer = base["producer_contracts"]["outer" if role == "outer_dev" else "inner"]
        path = r3._receipt_path(record, Path(producer["canonical_root"]), role, schema)
        path2, value = _file_record(record, f"{role} receipt", schema)
        if path2 != path:
            raise ValueError(f"{role} canonical receipt differs")
        official = outer.verify_success_receipt(path) if role == "outer_dev" else inner.verify_success_receipt(path)
        if isinstance(official, tuple):
            official = official[1]
        if official != value:
            raise ValueError(f"{role} official verification payload differs")
        train_folds, heldout, _ = ROLE_SPECS[role]
        if role != "outer_dev":
            protocol = value.get("training_protocol") or {}
            if (
                list(protocol.get("train_folds", ())) != list(train_folds)
                or protocol.get("heldout_fold") != heldout
                or protocol.get("train_scenes") != 1001
                or protocol.get("heldout_scenes") != 20
                or protocol.get("optimizer_updates") != 11268
                or protocol.get("initialization") != "random_scratch_ca_only"
                or protocol.get("checkpoint_selection") is not False
                or ((value.get("passing_upstream") or {}).get("eval_v2_continuation_receipt") or {}).get("sha256") != continuation_sha
            ):
                raise ValueError(f"{role} CA-only protocol differs")
        checkpoint, checkpoint_sha = r3._extract_checkpoint(value, role)
        scenes = r3._scene_list(base["scene_contract"]["roles"][role]["predict_scene_list"], f"{role} predict scenes", 20)
        roles[role] = r3.VerifiedRole(
            role, train_folds, heldout, scenes, path, sha256_file(path), value,
            checkpoint, checkpoint_sha,
        )
    return continuation_path, roles


def _r2_config(ready: Mapping[str, Any]) -> dict[str, Any]:
    base = _load_base(ready)[1]
    value = r3._r2_config({
        **base,
        "outputs": copy.deepcopy(ready["outputs"]),
        "continuation_receipt": copy.deepcopy(ready["continuation_receipt"]),
        "run_authorization": {
            "state": "bound", "path": os.fspath(AUTHORIZATION_BUNDLE_PATH),
            "sha256": "0" * 64, "schema": BUNDLE_SCHEMA,
        },
    })
    for role in ROLE_ORDER:
        value["scene_contract"]["roles"][role]["source_success_receipt"] = copy.deepcopy(
            ready["producer_success_receipts"][role]
        )
    value["namespace"] = NAMESPACE
    value["outputs"] = copy.deepcopy(ready["outputs"])
    return value


def _projection(ready: Mapping[str, Any]) -> str:
    value = copy.deepcopy(ready)
    value["run_authorization"] = {
        "state": "committed_by_bundle", "path": os.fspath(AUTHORIZATION_BUNDLE_PATH),
        "commit_id": "<redacted>", "schema": BUNDLE_SCHEMA,
    }
    return sha256_bytes(canonical_json(value))


def _commit_id(ready: Mapping[str, Any], continuation: Path, roles: Mapping[str, r3.VerifiedRole]) -> str:
    value = {
        "schema": "boxfusion.ca1m_tr3d_e961_terminal_inputs_commit_material.v5.r5",
        "pending_config_sha256": sha256_file(DEFAULT_CONFIG),
        "preregistration_sha256": sha256_file(PREREGISTRATION_PATH),
        "continuation_sha256": sha256_file(continuation),
        "roles": [{"role": role, "receipt_sha256": roles[role].receipt_sha256, "checkpoint_sha256": roles[role].checkpoint_sha256} for role in ROLE_ORDER],
        "ready_projection_sha256": _projection(ready),
    }
    return sha256_bytes(canonical_json(value))


def _authorization_payload(
    ready: Mapping[str, Any], continuation: Path, roles: Mapping[str, r3.VerifiedRole], commit_id: str,
) -> dict[str, Any]:
    r2_bytes = canonical_json(_r2_config(ready))
    return {
        "schema": AUTH_SCHEMA, "complete": True, "create_only": True,
        "namespace": NAMESPACE, "commit_id": commit_id,
        "ready_projection_sha256": _projection(ready),
        "pending_config_sha256": sha256_file(DEFAULT_CONFIG),
        "preregistration": {"path": os.fspath(PREREGISTRATION_PATH), "sha256": sha256_file(PREREGISTRATION_PATH)},
        "continuation_receipt": {"path": os.fspath(continuation), "sha256": sha256_file(continuation), "schema": CONTINUATION_SCHEMA},
        "roles": [{
            "role": role, "receipt_path": os.fspath(roles[role].receipt_path),
            "receipt_sha256": roles[role].receipt_sha256,
            "checkpoint_sha256": roles[role].checkpoint_sha256,
        } for role in ROLE_ORDER],
        "canonical_r2_config_sha256": sha256_bytes(r2_bytes),
        "producer_deep_verifiers_passed": True, "eval_pass_verified": True,
        "authorizations": copy.deepcopy(AUTHORIZATIONS),
        "ground_truth_access": False, "fold1_access": False,
        "official_validation_access": False, "formal_gpu_run_started": False,
    }


def _bundle_payload(ready_bytes: bytes, auth_bytes: bytes, commit_id: str) -> dict[str, Any]:
    return {
        "schema": BUNDLE_SCHEMA, "complete": True, "create_only": True,
        "namespace": NAMESPACE, "commit_id": commit_id,
        "commit_role": "last_published_unique_operational_gate",
        "ready_config": {"path": os.fspath(READY_CONFIG_PATH), "sha256": sha256_bytes(ready_bytes), "schema": CONFIG_SCHEMA},
        "run_authorization": {"path": os.fspath(RUN_AUTHORIZATION_PATH), "sha256": sha256_bytes(auth_bytes), "schema": AUTH_SCHEMA},
        "safe_replay": "verify_or_fill_exact_ready_and_auth_then_create_or_verify_bundle",
        "ground_truth_access": False, "gpu_started": False,
    }


def seal_ready_authorization_bundle(
    *, outer_receipt: Path, inner_holdout2_receipt: Path,
    inner_holdout3_receipt: Path, inner_holdout4_receipt: Path,
    continuation_receipt: Path,
) -> tuple[Path, Path, Path]:
    """Replay-safe pair publication; bundle is always published last."""

    validate_static_config()
    _, pending = _load_pending()
    ready = copy.deepcopy(pending)
    supplied = {
        "outer_dev": outer_receipt, "inner_holdout2": inner_holdout2_receipt,
        "inner_holdout3": inner_holdout3_receipt, "inner_holdout4": inner_holdout4_receipt,
    }
    for role, path in supplied.items():
        ready["producer_success_receipts"][role] = _bound_receipt(
            path, OUTER_SCHEMA if role == "outer_dev" else INNER_SCHEMA, f"{role} receipt",
        )
    ready["continuation_receipt"] = _bound_receipt(
        continuation_receipt, CONTINUATION_SCHEMA, "E961 continuation",
    )
    ready["run_authorization"] = {
        "state": "committed_by_bundle", "path": os.fspath(AUTHORIZATION_BUNDLE_PATH),
        "commit_id": "0" * 64, "schema": BUNDLE_SCHEMA,
    }
    continuation, roles = _verify_dynamic(ready)
    commit_id = _commit_id(ready, continuation, roles)
    ready["run_authorization"]["commit_id"] = commit_id
    _ready_delta(pending, ready)
    auth = _authorization_payload(ready, continuation, roles, commit_id)
    ready_bytes, auth_bytes = canonical_json(ready), canonical_json(auth)
    bundle = _bundle_payload(ready_bytes, auth_bytes, commit_id)
    bundle_bytes = canonical_json(bundle)

    r3._host_target_probe(MANIFEST_ROOT)
    claim_fd = r3._claim_writer(MANIFEST_ROOT, ".READY_BUNDLE_SEAL.writer.claim", commit_id)
    try:
        r3.create_or_verify(READY_CONFIG_PATH, ready_bytes, "R5 ready leaf")
        r3.create_or_verify(RUN_AUTHORIZATION_PATH, auth_bytes, "R5 authorization leaf")
        # The sole commit gate is last.  A crash before this line is pending and
        # an exact rerun safely verifies/fills either leaf before retrying it.
        r3.create_or_verify(AUTHORIZATION_BUNDLE_PATH, bundle_bytes, "R5 authorization bundle")
    finally:
        os.close(claim_fd)
    return READY_CONFIG_PATH, RUN_AUTHORIZATION_PATH, AUTHORIZATION_BUNDLE_PATH


def _load_committed_ready(path: Path) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes, dict[str, Any], bytes]:
    if path != READY_CONFIG_PATH:
        raise PermissionError("R5 operational ready path is noncanonical")
    ready_bytes = stable_bytes(path, "R5 ready config")
    ready = json.loads(ready_bytes)
    try:
        auth_bytes = stable_bytes(RUN_AUTHORIZATION_PATH, "R5 authorization")
        bundle_bytes = stable_bytes(AUTHORIZATION_BUNDLE_PATH, "R5 authorization bundle")
    except FileNotFoundError as error:
        raise PendingOperationalInputs("R5 ready/auth leaves are not bundle-committed") from error
    auth, bundle = json.loads(auth_bytes), json.loads(bundle_bytes)
    record = ready.get("run_authorization") or {}
    if (
        record.get("state") != "committed_by_bundle"
        or record.get("path") != os.fspath(AUTHORIZATION_BUNDLE_PATH)
        or record.get("schema") != BUNDLE_SCHEMA
        or bundle.get("schema") != BUNDLE_SCHEMA or bundle.get("complete") is not True
        or bundle.get("commit_role") != "last_published_unique_operational_gate"
        or bundle.get("commit_id") != record.get("commit_id")
        or (bundle.get("ready_config") or {}).get("path") != os.fspath(READY_CONFIG_PATH)
        or (bundle.get("ready_config") or {}).get("sha256") != sha256_bytes(ready_bytes)
        or (bundle.get("run_authorization") or {}).get("path") != os.fspath(RUN_AUTHORIZATION_PATH)
        or (bundle.get("run_authorization") or {}).get("sha256") != sha256_bytes(auth_bytes)
        or auth.get("schema") != AUTH_SCHEMA or auth.get("commit_id") != record.get("commit_id")
    ):
        raise PermissionError("R5 authorization bundle/leaf binding differs")
    return ready, ready_bytes, auth, auth_bytes, bundle, bundle_bytes


@dataclass(frozen=True)
class R2ContextSnapshot:
    config_path: Path
    config: Mapping[str, Any]
    authorization_path: Path
    authorization_sha256: str
    continuation_path: Path
    continuation_sha256: str
    roles: Mapping[str, r3.VerifiedRole]


@dataclass(frozen=True)
class CanonicalAuthority:
    ready_bytes: bytes
    ready_sha256: str
    authorization_bytes: bytes
    authorization_sha256: str
    bundle_bytes: bytes
    bundle_sha256: str
    preregistration_bytes: bytes
    preregistration_sha256: str
    writer_claim_identity: tuple[int, int, int, int, int]
    parent_identity: tuple[int, int]
    parent_chain: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class DerivedAuthority:
    ready: Mapping[str, Any]
    ready_bytes: bytes
    ready_sha256: str
    authorization: Mapping[str, Any]
    authorization_bytes: bytes
    authorization_sha256: str
    bundle: Mapping[str, Any]
    bundle_bytes: bytes
    bundle_sha256: str
    preregistration: Mapping[str, Any]
    preregistration_bytes: bytes
    preregistration_sha256: str
    continuation_path: Path
    continuation_sha256: str
    roles: Mapping[str, r3.VerifiedRole]
    r2_config: Mapping[str, Any]
    r2_config_bytes: bytes
    r2_config_sha256: str
    r2_module_path: Path
    r2_module_sha256: str
    r2_module_identity: tuple[int, int, int, int, int]
    r2_context: R2ContextSnapshot


@dataclass(frozen=True)
class ReadyContext:
    """Opaque runtime capability; none of its values describe stage inputs."""

    __slots__ = ("authority_token", "writer_fd", "parent_fd")
    authority_token: str
    writer_fd: int
    parent_fd: int

    def close(self) -> None:
        _RUNTIME_AUTHORITIES.pop(self.authority_token, None)
        for field in ("writer_fd", "parent_fd"):
            fd = getattr(self, field)
            if fd >= 0:
                os.close(fd)
                object.__setattr__(self, field, -1)


OUTPUT_PARENT_PATH = Path("/extra/ZhaoX")
WRITER_CLAIM_PATH = OUTPUT_PARENT_PATH / f".{NAMESPACE}.writer.claim"
R2_CANONICAL_PATH = ROOT / "boxfusion/ca1m_tr3d_e961_terminal_inputs_v5_r2.py"
_RUNTIME_AUTHORITIES: dict[str, CanonicalAuthority] = {}


def _roles_binding(roles: Mapping[str, r3.VerifiedRole]) -> bytes:
    return canonical_json({
        "roles": [{
            "role": role, "train_folds": list(roles[role].train_folds),
            "heldout_fold": roles[role].heldout_fold, "scenes": list(roles[role].scenes),
            "receipt_path": os.fspath(roles[role].receipt_path),
            "receipt_sha256": roles[role].receipt_sha256,
            "checkpoint_path": os.fspath(roles[role].checkpoint_path),
            "checkpoint_sha256": roles[role].checkpoint_sha256,
        } for role in ROLE_ORDER],
    })


def _fresh_r2_module(authority: DerivedAuthority) -> Any:
    path = authority.r2_module_path
    if (
        path != R2_CANONICAL_PATH
        or _identity(path) != authority.r2_module_identity
        or sha256_file(path) != authority.r2_module_sha256
    ):
        raise PermissionError("canonical R2 module file identity/content changed")
    name = f"boxfusion._r5_frozen_r2_{authority.r2_module_sha256[:12]}_{secrets.token_hex(12)}"
    if name in sys.modules:
        raise RuntimeError("fresh R2 private module name collision")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load frozen R2 execution module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    if Path(module.__file__).resolve() != path.resolve() or sha256_file(Path(module.__file__)) != authority.r2_module_sha256:
        raise ImportError("fresh R2 module __file__/SHA differs")
    for function_name in ("run_stage_p", "run_stage_o", "run_stage_e", "seal_stage_m"):
        function = getattr(module, function_name, None)
        if (
            not callable(function) or function.__module__ != name
            or Path(function.__code__.co_filename).resolve() != path.resolve()
        ):
            raise ImportError(f"fresh R2 {function_name} source identity differs")
    return module


def _assert_lock_still_owned(writer_fd: int) -> None:
    probe = os.open(WRITER_CLAIM_PATH, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        else:
            fcntl.flock(probe, fcntl.LOCK_UN)
            raise PermissionError("R5 writer descriptor no longer holds the exclusive lock")
    finally:
        os.close(probe)


_FLOCK_FORMAT = "hhqqi"


def _runtime_claim_bytes(bundle_sha256: str) -> bytes:
    return canonical_json({
        "schema": "boxfusion.ca1m_tr3d_e961_single_writer_claim.v5.r5",
        "complete": True, "create_only": True, "namespace": NAMESPACE,
        "authorization_sha256": bundle_sha256,
    })


def _install_fd_owned_lock(writer_fd: int) -> None:
    """Add an OFD read lock whose ownership is tied to this open description."""

    if not hasattr(fcntl, "F_OFD_SETLK") or not hasattr(fcntl, "F_OFD_GETLK"):
        raise RuntimeError("R5 requires Linux open-file-description lock support")
    request = struct.pack(_FLOCK_FORMAT, fcntl.F_RDLCK, os.SEEK_SET, 0, 0, 0)
    fcntl.fcntl(writer_fd, fcntl.F_OFD_SETLK, request)


def _assert_fd_owned_lock(writer_fd: int) -> None:
    """Prove this exact open description, not merely some FD, owns the lock."""

    query = struct.pack(_FLOCK_FORMAT, fcntl.F_WRLCK, os.SEEK_SET, 0, 0, 0)
    own = fcntl.fcntl(writer_fd, fcntl.F_OFD_GETLK, query)
    own_type = struct.unpack(_FLOCK_FORMAT, own[:struct.calcsize(_FLOCK_FORMAT)])[0]
    if own_type != fcntl.F_UNLCK:
        raise PermissionError("R5 writer FD is not the registered lock-owning open description")
    probe = os.open(WRITER_CLAIM_PATH, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        other = fcntl.fcntl(probe, fcntl.F_OFD_GETLK, query)
    finally:
        os.close(probe)
    other_type = struct.unpack(_FLOCK_FORMAT, other[:struct.calcsize(_FLOCK_FORMAT)])[0]
    if other_type != fcntl.F_RDLCK:
        raise PermissionError("R5 writer FD no longer owns its OFD lock")


def _claim_runtime_writer(bundle_sha256: str) -> int:
    if WRITER_CLAIM_PATH.parent != OUTPUT_PARENT_PATH or WRITER_CLAIM_PATH.name != f".{NAMESPACE}.writer.claim":
        raise PermissionError("R5 runtime writer claim path differs from the fixed namespace path")
    payload = _runtime_claim_bytes(bundle_sha256)
    try:
        r3._exclusive_bytes(WRITER_CLAIM_PATH, payload)
    except FileExistsError:
        if stable_bytes(WRITER_CLAIM_PATH, "R5 runtime writer claim") != payload:
            raise PermissionError("R5 runtime writer claim belongs to another authorization")
    parent_fd, _ = r3._open_dir_chain(OUTPUT_PARENT_PATH, "R5 runtime writer parent")
    try:
        writer_fd = os.open(
            WRITER_CLAIM_PATH.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
    finally:
        os.close(parent_fd)
    try:
        fcntl.flock(writer_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _install_fd_owned_lock(writer_fd)
    except Exception:
        os.close(writer_fd)
        raise PermissionError("another R5 writer currently owns the namespace")
    return writer_fd


def _authority_record(ctx: ReadyContext) -> CanonicalAuthority:
    if type(ctx) is not ReadyContext:
        raise PermissionError("R5 context type differs")
    authority = _RUNTIME_AUTHORITIES.get(ctx.authority_token)
    if authority is None:
        raise PermissionError("R5 context has no registered canonical authority")
    return authority


def _guard_persistent_handles(ctx: ReadyContext, authority: CanonicalAuthority) -> None:
    if ctx.writer_fd < 0 or ctx.parent_fd < 0:
        raise PermissionError("R5 persistent authority descriptor is closed")
    try:
        writer = os.fstat(ctx.writer_fd)
        parent = os.fstat(ctx.parent_fd)
    except OSError as error:
        raise PermissionError("R5 persistent authority descriptor is invalid") from error
    writer_identity = (writer.st_dev, writer.st_ino, writer.st_size, writer.st_mtime_ns, writer.st_nlink)
    if writer_identity != authority.writer_claim_identity or writer.st_nlink != 1:
        raise PermissionError("R5 writer FD differs from registered canonical claim")
    try:
        canonical_claim = _identity(WRITER_CLAIM_PATH)
    except (FileNotFoundError, NotADirectoryError, OSError) as error:
        raise PermissionError("R5 fixed writer claim no longer names the registered inode") from error
    if canonical_claim != authority.writer_claim_identity:
        raise PermissionError("R5 fixed writer claim no longer names the registered inode")
    expected_claim_bytes = _runtime_claim_bytes(authority.bundle_sha256)
    if stable_bytes(WRITER_CLAIM_PATH, "R5 fixed writer claim") != expected_claim_bytes:
        raise PermissionError("R5 fixed writer claim payload is not bound to the canonical bundle")
    _assert_fd_owned_lock(ctx.writer_fd)
    _assert_lock_still_owned(ctx.writer_fd)
    if not stat.S_ISDIR(parent.st_mode) or (parent.st_dev, parent.st_ino) != authority.parent_identity:
        raise PermissionError("R5 held output-parent dirfd differs from registered canonical parent")
    reopened_fd, reopened_chain = r3._open_dir_chain(OUTPUT_PARENT_PATH, "R5 fixed output parent")
    try:
        reopened = os.fstat(reopened_fd)
    finally:
        os.close(reopened_fd)
    if (
        tuple(reopened_chain) != authority.parent_chain
        or (reopened.st_dev, reopened.st_ino) != authority.parent_identity
    ):
        raise PermissionError("R5 fixed output-parent chain changed")
    r3._host_target_probe(OUTPUT_PARENT_PATH)


def _derive_canonical_payloads(expected: Any = None) -> DerivedAuthority:
    ready, ready_bytes, auth, auth_bytes, bundle, bundle_bytes = _load_committed_ready(READY_CONFIG_PATH)
    preregistration_bytes = stable_bytes(PREREGISTRATION_PATH, "R5 canonical preregistration")
    preregistration = json.loads(preregistration_bytes)
    ready_sha = sha256_bytes(ready_bytes)
    auth_sha = sha256_bytes(auth_bytes)
    bundle_sha = sha256_bytes(bundle_bytes)
    preregistration_sha = sha256_bytes(preregistration_bytes)
    if expected is not None and (
        ready_bytes != expected.ready_bytes or ready_sha != expected.ready_sha256
        or auth_bytes != expected.authorization_bytes or auth_sha != expected.authorization_sha256
        or bundle_bytes != expected.bundle_bytes or bundle_sha != expected.bundle_sha256
        or preregistration_bytes != expected.preregistration_bytes
        or preregistration_sha != expected.preregistration_sha256
    ):
        raise PermissionError("R5 canonical sealed authority bytes changed")

    _, pending = _load_pending()
    _ready_delta(pending, ready)
    expected_prereg_record = {
        "path": os.fspath(PREREGISTRATION_PATH), "sha256": preregistration_sha,
    }
    if (
        preregistration.get("schema") != PREREGISTRATION_SCHEMA
        or preregistration.get("complete") is not True
        or preregistration.get("create_only") is not True
        or preregistration.get("static_only") is not True
        or preregistration.get("namespace") != NAMESPACE
        or preregistration.get("pending_config") != {
            "path": os.fspath(DEFAULT_CONFIG), "sha256": sha256_file(DEFAULT_CONFIG),
            "schema": CONFIG_SCHEMA,
        }
        or preregistration.get("implementation") != pending.get("implementation")
        or (auth.get("preregistration") or {}).get("path") != expected_prereg_record["path"]
        or (auth.get("preregistration") or {}).get("sha256") != expected_prereg_record["sha256"]
    ):
        raise PermissionError("R5 preregistration/ready/authorization binding differs")
    for section in ("implementation", "base_execution_dependencies"):
        records = preregistration.get(section)
        if not isinstance(records, Mapping) or not records:
            raise PermissionError(f"R5 preregistration {section} inventory is missing")
        for name, dependency_record in records.items():
            try:
                _file_record(dependency_record, f"R5 preregistered {section}.{name}")
            except (OSError, ValueError) as error:
                raise PermissionError(f"R5 preregistered {section}.{name} changed") from error
    try:
        invalid_path, invalid_value = _file_record(
            preregistration.get("r4_invalidation"), "R5 preregistered R4 invalidation", R4_INVALID_SCHEMA,
        )
    except (OSError, ValueError) as error:
        raise PermissionError("R5 preregistered R4 invalidation changed") from error
    if invalid_path != R4_INVALID_PATH or invalid_value.get("invalid") is not True:
        raise PermissionError("R5 preregistered R4 invalidation differs")

    continuation, roles = _verify_dynamic(ready)
    commit_id = (ready.get("run_authorization") or {}).get("commit_id")
    expected_auth = _authorization_payload(ready, continuation, roles, commit_id)
    expected_bundle = _bundle_payload(ready_bytes, auth_bytes, commit_id)
    if auth != expected_auth or bundle != expected_bundle:
        raise PermissionError("R5 committed authorization payload differs")
    r2_config = _r2_config(ready)
    r2_bytes = canonical_json(r2_config)
    r2_sha = sha256_bytes(r2_bytes)
    if auth.get("canonical_r2_config_sha256") != r2_sha or r2_config.get("outputs") != _expected_outputs():
        raise PermissionError("R5 authorization R2 config/output binding differs")

    r2_record = ready["implementation"]["r2_execution_core"]
    if preregistration["implementation"].get("r2_execution_core") != r2_record:
        raise PermissionError("R5 R2 module record differs between ready and preregistration")
    r2_path, _ = _file_record(r2_record, "R5 canonical R2 execution core")
    if r2_path != R2_CANONICAL_PATH:
        raise PermissionError("R5 R2 execution path is noncanonical")
    continuation_sha = sha256_file(continuation)
    snapshot = R2ContextSnapshot(
        READY_CONFIG_PATH, copy.deepcopy(r2_config), AUTHORIZATION_BUNDLE_PATH,
        bundle_sha, continuation, continuation_sha, roles,
    )
    return DerivedAuthority(
        ready, ready_bytes, ready_sha, auth, auth_bytes, auth_sha,
        bundle, bundle_bytes, bundle_sha, preregistration,
        preregistration_bytes, preregistration_sha, continuation,
        continuation_sha, roles, r2_config, r2_bytes, r2_sha,
        r2_path, sha256_file(r2_path), _identity(r2_path), snapshot,
    )


def _guard_context(ctx: ReadyContext) -> DerivedAuthority:
    """Freshly derive all stage inputs from canonical sealed files."""

    authority = _authority_record(ctx)
    _guard_persistent_handles(ctx, authority)
    return _derive_canonical_payloads(authority)


def validate_operational_ready(path: Path = None) -> ReadyContext:
    if path is None:
        path = READY_CONFIG_PATH if READY_CONFIG_PATH.exists() and not READY_CONFIG_PATH.is_symlink() else DEFAULT_CONFIG
    path = Path(path)
    if path == DEFAULT_CONFIG:
        raise PendingOperationalInputs("formal R5 receipt bundle is pending")
    if path != READY_CONFIG_PATH:
        raise PermissionError("R5 operational ready path is noncanonical")
    validate_static_config()
    try:
        derived = _derive_canonical_payloads()
    except FileNotFoundError as error:
        raise PendingOperationalInputs("formal R5 receipt bundle is pending") from error

    if Path(derived.ready["outputs"]["namespace_root"]).parent != OUTPUT_PARENT_PATH:
        raise PermissionError("R5 ready output parent differs from fixed canonical parent")
    r3._host_target_probe(OUTPUT_PARENT_PATH)
    parent_fd, parent_chain = r3._open_dir_chain(OUTPUT_PARENT_PATH, "R5 fixed output parent")
    parent_stat = os.fstat(parent_fd)
    writer_fd = _claim_runtime_writer(derived.bundle_sha256)
    token = secrets.token_hex(32)
    while token in _RUNTIME_AUTHORITIES:
        token = secrets.token_hex(32)
    authority = CanonicalAuthority(
        derived.ready_bytes, derived.ready_sha256,
        derived.authorization_bytes, derived.authorization_sha256,
        derived.bundle_bytes, derived.bundle_sha256,
        derived.preregistration_bytes, derived.preregistration_sha256,
        _identity(WRITER_CLAIM_PATH), (parent_stat.st_dev, parent_stat.st_ino),
        tuple(parent_chain),
    )
    _RUNTIME_AUTHORITIES[token] = authority
    ctx = ReadyContext(token, writer_fd, parent_fd)
    try:
        fresh = _guard_context(ctx)
        _fresh_r2_module(fresh)
        return ctx
    except Exception:
        ctx.close()
        raise


@contextmanager
def _fresh_secure_r2(ctx: ReadyContext) -> Iterator[tuple[Any, DerivedAuthority]]:
    derived = _guard_context(ctx)
    module = _fresh_r2_module(derived)
    module.write_bytes_exclusive = r3.write_bytes_exclusive
    module.ensure_directory = r3.ensure_directory
    yield module, derived


def run_stage_p(ctx: ReadyContext, role: str, *, device: str = "cuda:0", **kwargs: Any) -> dict[str, Any]:
    with _fresh_secure_r2(ctx) as (module, derived):
        function = module.run_stage_p
        if function.__module__ != module.__name__:
            raise PermissionError("R5 stage-P callable source changed")
        result = function(derived.r2_context, role, device=device, **kwargs)
    _guard_context(ctx)
    return result


def run_stage_o(ctx: ReadyContext, role: str, **kwargs: Any) -> dict[str, Any]:
    with _fresh_secure_r2(ctx) as (module, derived):
        function = module.run_stage_o
        if function.__module__ != module.__name__:
            raise PermissionError("R5 stage-O callable source changed")
        result = function(derived.r2_context, role, **kwargs)
    _guard_context(ctx)
    return result


def run_stage_e(ctx: ReadyContext, role: str, **kwargs: Any) -> dict[str, Any]:
    with _fresh_secure_r2(ctx) as (module, derived):
        function = module.run_stage_e
        if function.__module__ != module.__name__:
            raise PermissionError("R5 stage-E callable source changed")
        result = function(derived.r2_context, role, **kwargs)
    _guard_context(ctx)
    return result


def seal_stage_m(ctx: ReadyContext) -> dict[str, Any]:
    with _fresh_secure_r2(ctx) as (module, derived):
        function = module.seal_stage_m
        if function.__module__ != module.__name__:
            raise PermissionError("R5 stage-M callable source changed")
        r2_result = function(derived.r2_context)
    derived = _guard_context(ctx)
    combined = Path(derived.ready["outputs"]["combined_manifest"])
    r2_receipt = (r2_result.get("receipt") or {})
    r2_receipt_path, _ = _file_record(
        r2_receipt, "R2 exact80 execution receipt",
        "boxfusion.ca1m_tr3d_e961_terminal_inputs_exact80_receipt.v5.r2",
    )
    wrapper = {
        "schema": EXACT80_RECEIPT_SCHEMA, "complete": True, "create_only": True,
        "namespace": NAMESPACE, "scene_count": 80, "fit_scene_count": 60,
        "fit_folds": [2, 3, 4], "reused_dev_scene_count": 20,
        "reused_dev_folds": [0], "each_scene_detector_excludes_scene": True,
        "b6_score_source": "all_fold_oof_each_row_model_excludes_scene",
        "legacy_v1_v4_candidate_or_policy_reused": False,
        "authorization_commit_id": derived.bundle["commit_id"],
        "candidate_collection": {
            "path": os.fspath(combined), "sha256": sha256_file(combined),
            "schema": "boxfusion.ca1m_tr3d_xfit_r2_candidate_collection.v1",
        },
        "r2_execution_receipt": {
            "path": os.fspath(r2_receipt_path), "sha256": sha256_file(r2_receipt_path),
            "schema": "boxfusion.ca1m_tr3d_e961_terminal_inputs_exact80_receipt.v5.r2",
            "operational_authority": False,
        },
        "r5_preregistration": {"path": os.fspath(PREREGISTRATION_PATH), "sha256": derived.preregistration_sha256, "schema": PREREGISTRATION_SCHEMA},
        "r5_ready_config": {"path": os.fspath(READY_CONFIG_PATH), "sha256": derived.ready_sha256, "schema": CONFIG_SCHEMA},
        "r5_run_authorization": {"path": os.fspath(RUN_AUTHORIZATION_PATH), "sha256": derived.authorization_sha256, "schema": AUTH_SCHEMA},
        "r5_authorization_bundle": {"path": os.fspath(AUTHORIZATION_BUNDLE_PATH), "sha256": derived.bundle_sha256, "schema": BUNDLE_SCHEMA},
        "ground_truth_access": False, "fold1_access": False,
        "official_validation_access": False,
    }
    target = Path(derived.ready["outputs"]["manifest_root"]) / "M_EXACT80_R5_RECEIPT.json"
    r3.create_or_verify(target, canonical_json(wrapper), "R5 exact80 receipt")
    return {**wrapper, "receipt": {"path": os.fspath(target), "sha256": sha256_file(target), "schema": EXACT80_RECEIPT_SCHEMA}}


def run_all(ctx: ReadyContext, *, device: str = "cuda:0") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role in ROLE_ORDER:
        result[f"P:{role}"] = run_stage_p(ctx, role, device=device)
        result[f"O:{role}"] = run_stage_o(ctx, role)
        result[f"E:{role}"] = run_stage_e(ctx, role)
    result["M"] = seal_stage_m(ctx)
    return {
        "schema": "boxfusion.ca1m_tr3d_e961_terminal_inputs_run.v5.r5",
        "complete": True, "fresh_sha_bound_r2_per_stage": True,
        "bundle_commit_verified": True, "ground_truth_access": False,
        "results": result,
    }


__all__ = [
    "DEFAULT_CONFIG", "PREREGISTRATION_PATH", "READY_CONFIG_PATH",
    "RUN_AUTHORIZATION_PATH", "AUTHORIZATION_BUNDLE_PATH", "R4_INVALID_PATH",
    "CONFIG_SCHEMA", "PREREGISTRATION_SCHEMA", "AUTH_SCHEMA", "BUNDLE_SCHEMA", "EXACT80_RECEIPT_SCHEMA",
    "NAMESPACE", "ROLE_ORDER", "PendingOperationalInputs", "ReadyContext",
    "R2ContextSnapshot", "load_bound_producer_verifiers", "r4_invalidation_payload",
    "seal_r4_invalidation", "build_preregistration_payload", "seal_preregistration",
    "validate_static_config", "seal_ready_authorization_bundle",
    "validate_operational_ready", "run_stage_p", "run_stage_o", "run_stage_e",
    "seal_stage_m", "run_all", "sha256_file",
]
