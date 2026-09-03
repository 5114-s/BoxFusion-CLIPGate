"""R4 commit and execution boundary for CA-only E961 terminal inputs.

R4 supersedes the never-authorized R3 static preregistration.  It retains the
R3 static dataset/producer contract but replaces its runtime authority with
persistent descriptor bindings, an exact canonical R2-context snapshot, fresh
SHA-bound R2 module loading for every stage, and a last-published authorization
bundle as the sole commit gate.  Import is side-effect free and no code here
opens annotations or ground truth.
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
import sys
from typing import Any, Iterator, Mapping

from . import ca1m_tr3d_e961_terminal_inputs_v5_r3 as r3


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/ca1m_tr3d_e961_terminal_inputs_v5_r4_pending.json"
MANIFEST_ROOT = ROOT / "manifests/ca1m_tr3d_e961_terminal_inputs_v5_r4"
PREREGISTRATION_PATH = MANIFEST_ROOT / "PREREGISTRATION.json"
READY_CONFIG_PATH = MANIFEST_ROOT / "READY_CONFIG.json"
RUN_AUTHORIZATION_PATH = MANIFEST_ROOT / "RUN_AUTHORIZATION.json"
AUTHORIZATION_BUNDLE_PATH = MANIFEST_ROOT / "AUTHORIZATION_BUNDLE.json"
R3_INVALID_PATH = (
    ROOT / "manifests/ca1m_tr3d_e961_terminal_inputs_v5_r3/PREREGISTRATION_V2_INVALID.json"
)

CONFIG_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_config.v5.r4"
PREREGISTRATION_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_preregistration.v5.r4"
R3_INVALID_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_preregistration_invalid.v5.r3.v2"
AUTH_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_run_authorization.v5.r4"
BUNDLE_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_authorization_bundle.v5.r4"
EXACT80_RECEIPT_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_exact80_receipt.v5.r4"
STATIC_REPORT_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_static_preflight.v5.r4"
OPERATIONAL_REPORT_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_operational_preflight.v5.r4"
NAMESPACE = "ca1m_tr3d_e961_terminal_inputs_xfit_r2_v5_r4"
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
    path, cfg = _json(DEFAULT_CONFIG, "R4 frozen pending config")
    if path != DEFAULT_CONFIG or cfg.get("schema") != CONFIG_SCHEMA or cfg.get("namespace") != NAMESPACE:
        raise ValueError("R4 pending config identity differs")
    return path, cfg


def _load_base(cfg: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path, base = _file_record(cfg.get("base_static_contract"), "R3 base static contract", r3.CONFIG_SCHEMA)
    if path != r3.DEFAULT_CONFIG:
        raise ValueError("R4 base static contract is noncanonical")
    # The R3 JSON freezes all semantic paths, but downstream pipeline modules
    # can legitimately advance before R4 is sealed.  Validate those exact
    # paths against a live SHA inventory which R4 preregisters independently.
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
        raise ValueError("R4 schema/namespace differs")
    if cfg.get("preregistration") != {"path": os.fspath(PREREGISTRATION_PATH), "schema": PREREGISTRATION_SCHEMA}:
        raise ValueError("R4 preregistration target differs")
    if cfg.get("sealed_dynamic_outputs") != {
        "ready_config": os.fspath(READY_CONFIG_PATH),
        "run_authorization": os.fspath(RUN_AUTHORIZATION_PATH),
        "authorization_bundle": os.fspath(AUTHORIZATION_BUNDLE_PATH),
    }:
        raise ValueError("R4 dynamic targets differ")
    _, base = _load_base(cfg)
    predecessor = cfg.get("invalidated_predecessor") or {}
    _file_record(predecessor.get("preregistration"), "R3 V2 preregistration", r3.PREREGISTRATION_SCHEMA)
    invalid = predecessor.get("invalidation")
    if require_invalidation:
        invalid_path, invalid_value = _file_record(invalid, "R3 V2 invalidation", R3_INVALID_SCHEMA)
        if invalid_path != R3_INVALID_PATH or invalid_value.get("invalid") is not True:
            raise ValueError("R3 invalidation differs")
    elif invalid != _pending(R3_INVALID_SCHEMA):
        raise ValueError("R3 invalidation must be pending before create-only invalidation")
    receipts = cfg.get("producer_success_receipts") or {}
    if tuple(receipts) != ROLE_ORDER:
        raise ValueError("R4 producer role order differs")
    if cfg.get("outputs") != _expected_outputs():
        raise ValueError("R4 output paths differ")
    implementation = cfg.get("implementation") or {}
    required = {
        "current_core", "current_sealer", "current_preflight", "current_runner",
        "current_tests", "r3_static_helper", "r2_execution_core",
    }
    if set(implementation) != required:
        raise ValueError("R4 implementation inventory differs")
    for key, record in implementation.items():
        _file_record(record, f"R4 implementation {key}")
    if cfg.get("access") != base.get("access"):
        raise ValueError("R4 access contract differs")
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
        raise PermissionError("R4 ready differs outside the exact six dynamic fields")
    return result


def r3_invalidation_payload() -> dict[str, Any]:
    predecessor = ROOT / "manifests/ca1m_tr3d_e961_terminal_inputs_v5_r3/PREREGISTRATION_V2.json"
    expected_sha = "ff0a8ca0c3d807cf9f6ab8e7d54d194c08be2f9ba898fb1a0d5b414a502c188d"
    if sha256_file(predecessor) != expected_sha:
        raise ValueError("R3 V2 predecessor changed")
    for forbidden in (
        ROOT / "manifests/ca1m_tr3d_e961_terminal_inputs_v5_r3/READY_CONFIG.json",
        ROOT / "manifests/ca1m_tr3d_e961_terminal_inputs_v5_r3/RUN_AUTHORIZATION.json",
        Path("/extra/ZhaoX/ca1m_tr3d_e961_terminal_inputs_xfit_r2_v5_r3"),
    ):
        if forbidden.exists() or forbidden.is_symlink():
            raise PermissionError("cannot invalidate R3 after operational output")
    return {
        "schema": R3_INVALID_SCHEMA, "complete": True, "create_only": True,
        "invalid": True, "operational_authority": False,
        "predecessor": {"path": os.fspath(predecessor), "sha256": expected_sha, "schema": r3.PREREGISTRATION_SCHEMA},
        "audit_result": "CODE_BLOCK",
        "reasons": [
            "writer_and_parent_descriptors_not_persistently_bound_and_revalidated",
            "r2_context_not_canonical_byte_bound_per_stage",
            "r2_stage_loader_trusted_replaceable_sys_modules_entry",
            "ready_authorization_pair_had_a_crash_window_without_last_commit_bundle",
            "inner_producer_verifier_import_was_not_dependency_injected_from_bound_outer",
        ],
        "superseded_by_namespace": NAMESPACE,
        "ready_config_created": False, "run_authorization_created": False,
        "runtime_namespace_created": False, "gpu_started": False,
        "ground_truth_access": False,
    }


def seal_r3_invalidation() -> Path:
    payload = r3_invalidation_payload()
    r3._host_target_probe(R3_INVALID_PATH.parent)
    fd = r3._claim_writer(R3_INVALID_PATH.parent, ".R3_V2_INVALID.writer.claim", sha256_bytes(canonical_json(payload)))
    try:
        return r3._exclusive_bytes(R3_INVALID_PATH, canonical_json(payload))
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
        "invalid_r3_sha256": predecessor_implementation[name]["sha256"],
        "r4_preregistered_sha256": base["implementation"][name]["sha256"],
    } for name in sorted(base["implementation"])
        if predecessor_implementation[name]["sha256"] != base["implementation"][name]["sha256"]]
    if [item["name"] for item in dependency_changes] != ["v5_manifest_runtime"]:
        raise ValueError("R3-to-R4 dependency drift inventory differs")
    for role in ROLE_ORDER:
        schema = OUTER_SCHEMA if role == "outer_dev" else INNER_SCHEMA
        if cfg["producer_success_receipts"][role] != _pending(schema):
            raise ValueError(f"{role} is not pending")
    if cfg.get("continuation_receipt") != _pending(CONTINUATION_SCHEMA):
        raise ValueError("R4 continuation is not pending")
    if cfg.get("run_authorization") != _pending(BUNDLE_SCHEMA, bundle=True):
        raise ValueError("R4 bundle authorization is not pending")
    inventory = r3.processed_point_inventory(base)
    invalid_path, _ = _file_record(
        cfg["invalidated_predecessor"]["invalidation"], "R3 V2 invalidation", R3_INVALID_SCHEMA,
    )
    return {
        "schema": PREREGISTRATION_SCHEMA, "complete": True, "create_only": True,
        "static_only": True, "namespace": NAMESPACE,
        "pending_config": {"path": os.fspath(source), "sha256": sha256_file(source), "schema": CONFIG_SCHEMA},
        "base_static_contract": copy.deepcopy(cfg["base_static_contract"]),
        "base_execution_dependencies": copy.deepcopy(base["implementation"]),
        "invalid_r3_dependency_relationship": {
            "predecessor_preregistration_invalid": True,
            "predecessor_config_sha256": sha256_file(r3.DEFAULT_CONFIG),
            "r4_rehashes_every_canonical_dependency": True,
            "changed_dependencies": dependency_changes,
        },
        "r3_invalidation": {"path": os.fspath(invalid_path), "sha256": sha256_file(invalid_path), "schema": R3_INVALID_SCHEMA},
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
            "persistent_parent_dirfd_devino": True,
            "canonical_r2_config_bytes_per_stage": True,
            "fresh_sha_bound_r2_module_per_stage": True,
            "bundle_is_last_and_only_commit_gate": True,
            "ready_auth_leaf_replay_before_bundle": True,
        },
        "access": {"gpu_started": False, "ground_truth_access": False, "fold1_access": False, "official_validation_access": False},
    }


def validate_preregistration() -> tuple[Path, dict[str, Any]]:
    path, value = _json(PREREGISTRATION_PATH, "R4 preregistration")
    expected = build_preregistration_payload()
    if value != expected:
        raise ValueError("R4 preregistration/static inputs drifted")
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
        raise ValueError("R4 static config path is noncanonical")
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
        "schema": "boxfusion.ca1m_tr3d_e961_terminal_inputs_commit_material.v5.r4",
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
        r3.create_or_verify(READY_CONFIG_PATH, ready_bytes, "R4 ready leaf")
        r3.create_or_verify(RUN_AUTHORIZATION_PATH, auth_bytes, "R4 authorization leaf")
        # The sole commit gate is last.  A crash before this line is pending and
        # an exact rerun safely verifies/fills either leaf before retrying it.
        r3.create_or_verify(AUTHORIZATION_BUNDLE_PATH, bundle_bytes, "R4 authorization bundle")
    finally:
        os.close(claim_fd)
    return READY_CONFIG_PATH, RUN_AUTHORIZATION_PATH, AUTHORIZATION_BUNDLE_PATH


def _load_committed_ready(path: Path) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes, dict[str, Any], bytes]:
    if path != READY_CONFIG_PATH:
        raise PermissionError("R4 operational ready path is noncanonical")
    ready_bytes = stable_bytes(path, "R4 ready config")
    ready = json.loads(ready_bytes)
    try:
        auth_bytes = stable_bytes(RUN_AUTHORIZATION_PATH, "R4 authorization")
        bundle_bytes = stable_bytes(AUTHORIZATION_BUNDLE_PATH, "R4 authorization bundle")
    except FileNotFoundError as error:
        raise PendingOperationalInputs("R4 ready/auth leaves are not bundle-committed") from error
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
        raise PermissionError("R4 authorization bundle/leaf binding differs")
    return ready, ready_bytes, auth, auth_bytes, bundle, bundle_bytes


@dataclass
class R2ContextSnapshot:
    config_path: Path
    config: Mapping[str, Any]
    authorization_path: Path
    authorization_sha256: str
    continuation_path: Path
    continuation_sha256: str
    roles: Mapping[str, r3.VerifiedRole]


@dataclass
class ReadyContext:
    config_path: Path
    ready: Mapping[str, Any]
    ready_bytes: bytes
    ready_sha256: str
    authorization: Mapping[str, Any]
    authorization_bytes: bytes
    authorization_sha256: str
    bundle: Mapping[str, Any]
    bundle_bytes: bytes
    bundle_sha256: str
    continuation_path: Path
    continuation_sha256: str
    roles: Mapping[str, r3.VerifiedRole]
    roles_binding_bytes: bytes
    r2_context: R2ContextSnapshot
    r2_config_bytes: bytes
    r2_config_sha256: str
    r2_module_path: Path
    r2_module_sha256: str
    r2_module_identity: tuple[int, int, int, int, int]
    writer_fd: int
    writer_claim_path: Path
    writer_claim_identity: tuple[int, int, int, int, int]
    parent_fd: int
    parent_path: Path
    parent_identity: tuple[int, int]
    parent_chain: tuple[tuple[int, int], ...]

    def close(self) -> None:
        for field in ("writer_fd", "parent_fd"):
            fd = getattr(self, field)
            if fd >= 0:
                os.close(fd)
                setattr(self, field, -1)


def _fresh_r2_module(ctx: ReadyContext) -> Any:
    path = ctx.r2_module_path
    if _identity(path) != ctx.r2_module_identity or sha256_file(path) != ctx.r2_module_sha256:
        raise PermissionError("frozen R2 module file identity/content changed")
    name = f"boxfusion._r4_frozen_r2_{ctx.r2_module_sha256[:12]}_{secrets.token_hex(12)}"
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
    if Path(module.__file__).resolve() != path.resolve() or sha256_file(Path(module.__file__)) != ctx.r2_module_sha256:
        raise ImportError("fresh R2 module __file__/SHA differs")
    for function_name in ("run_stage_p", "run_stage_o", "run_stage_e", "seal_stage_m"):
        function = getattr(module, function_name, None)
        if (
            not callable(function) or function.__module__ != name
            or Path(function.__code__.co_filename).resolve() != path.resolve()
        ):
            raise ImportError(f"fresh R2 {function_name} source identity differs")
    return module


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


def _assert_lock_still_owned(ctx: ReadyContext) -> None:
    probe = os.open(ctx.writer_claim_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        else:
            fcntl.flock(probe, fcntl.LOCK_UN)
            raise PermissionError("R4 writer descriptor no longer holds the exclusive lock")
    finally:
        os.close(probe)


def _guard_context(ctx: ReadyContext) -> None:
    """Reject every mutable-context diversion before stage code/device/output."""

    if ctx.writer_fd < 0 or ctx.parent_fd < 0:
        raise PermissionError("R4 persistent authority descriptor is closed")
    try:
        writer = os.fstat(ctx.writer_fd)
        parent = os.fstat(ctx.parent_fd)
    except OSError as error:
        raise PermissionError("R4 persistent authority descriptor is invalid") from error
    writer_identity = (writer.st_dev, writer.st_ino, writer.st_size, writer.st_mtime_ns, writer.st_nlink)
    if writer_identity != ctx.writer_claim_identity or writer.st_nlink != 1:
        raise PermissionError("R4 writer FD inode identity/link count changed")
    try:
        canonical_claim = _identity(ctx.writer_claim_path)
    except (FileNotFoundError, NotADirectoryError, OSError) as error:
        raise PermissionError("R4 canonical writer claim no longer names the held inode") from error
    if canonical_claim != ctx.writer_claim_identity:
        raise PermissionError("R4 canonical writer claim no longer names the held inode")
    _assert_lock_still_owned(ctx)
    if (parent.st_dev, parent.st_ino) != ctx.parent_identity:
        raise PermissionError("R4 held output-parent dirfd identity changed")
    reopened_fd, reopened_chain = r3._open_dir_chain(ctx.parent_path, "R4 output parent")
    try:
        reopened = os.fstat(reopened_fd)
    finally:
        os.close(reopened_fd)
    if tuple(reopened_chain) != tuple(ctx.parent_chain) or (reopened.st_dev, reopened.st_ino) != ctx.parent_identity:
        raise PermissionError("R4 canonical output-parent chain changed")
    if _identity(ctx.r2_module_path) != ctx.r2_module_identity or sha256_file(ctx.r2_module_path) != ctx.r2_module_sha256:
        raise PermissionError("R4 bound R2 module changed")
    if stable_bytes(ctx.config_path, "R4 ready config") != ctx.ready_bytes:
        raise PermissionError("R4 ready config changed")
    if stable_bytes(RUN_AUTHORIZATION_PATH, "R4 authorization") != ctx.authorization_bytes:
        raise PermissionError("R4 authorization changed")
    if stable_bytes(AUTHORIZATION_BUNDLE_PATH, "R4 bundle") != ctx.bundle_bytes:
        raise PermissionError("R4 bundle changed")
    if (
        canonical_json(ctx.ready) != ctx.ready_bytes
        or canonical_json(ctx.authorization) != ctx.authorization_bytes
        or canonical_json(ctx.bundle) != ctx.bundle_bytes
        or _roles_binding(ctx.roles) != ctx.roles_binding_bytes
    ):
        raise PermissionError("R4 in-memory ready/authority/role snapshot was diverted")
    current_r2 = canonical_json(_r2_config(ctx.ready))
    if current_r2 != ctx.r2_config_bytes or sha256_bytes(current_r2) != ctx.r2_config_sha256:
        raise PermissionError("R4 canonical R2 config derivation changed")
    if canonical_json(ctx.r2_context.config) != ctx.r2_config_bytes:
        raise PermissionError("R4 ctx.r2_context config was diverted")
    if (
        ctx.r2_context.config_path != ctx.config_path
        or ctx.r2_context.authorization_path != AUTHORIZATION_BUNDLE_PATH
        or ctx.r2_context.authorization_sha256 != ctx.bundle_sha256
        or ctx.r2_context.continuation_path != ctx.continuation_path
        or ctx.r2_context.continuation_sha256 != ctx.continuation_sha256
        or ctx.r2_context.roles is not ctx.roles
    ):
        raise PermissionError("R4 ctx.r2_context authority fields were diverted")
    if ctx.ready.get("outputs") != _expected_outputs() or ctx.r2_context.config.get("outputs") != _expected_outputs():
        raise PermissionError("R4 output namespace was diverted")


def validate_operational_ready(path: Path | None = None) -> ReadyContext:
    if path is None:
        path = READY_CONFIG_PATH if READY_CONFIG_PATH.exists() and not READY_CONFIG_PATH.is_symlink() else DEFAULT_CONFIG
    path = Path(path)
    if path == DEFAULT_CONFIG:
        raise PendingOperationalInputs("formal R4 receipt bundle is pending")
    try:
        ready, ready_bytes, auth, auth_bytes, bundle, bundle_bytes = _load_committed_ready(path)
    except FileNotFoundError as error:
        raise PendingOperationalInputs("formal R4 receipt bundle is pending") from error
    continuation, roles = _verify_dynamic(ready)
    commit_id = (ready.get("run_authorization") or {}).get("commit_id")
    expected_auth = _authorization_payload(ready, continuation, roles, commit_id)
    expected_bundle = _bundle_payload(ready_bytes, auth_bytes, commit_id)
    if auth != expected_auth or bundle != expected_bundle:
        raise PermissionError("R4 committed authorization payload differs")
    r2_config = _r2_config(ready)
    r2_bytes = canonical_json(r2_config)
    if auth.get("canonical_r2_config_sha256") != sha256_bytes(r2_bytes):
        raise PermissionError("R4 authorization R2 config binding differs")

    parent_path = Path(ready["outputs"]["namespace_root"]).parent
    r3._host_target_probe(parent_path)
    parent_fd, parent_chain = r3._open_dir_chain(parent_path, "R4 output parent")
    parent_stat = os.fstat(parent_fd)
    writer_claim = parent_path / f".{NAMESPACE}.writer.claim"
    writer_fd = r3._claim_writer(parent_path, writer_claim.name, sha256_bytes(bundle_bytes))
    try:
        r2_path, _ = _file_record(ready["implementation"]["r2_execution_core"], "R4 R2 execution core")
        snapshot = R2ContextSnapshot(
            path, copy.deepcopy(r2_config), AUTHORIZATION_BUNDLE_PATH,
            sha256_bytes(bundle_bytes), continuation, sha256_file(continuation), roles,
        )
        ctx = ReadyContext(
            path, ready, ready_bytes, sha256_bytes(ready_bytes), auth, auth_bytes,
            sha256_bytes(auth_bytes), bundle, bundle_bytes, sha256_bytes(bundle_bytes),
            continuation, sha256_file(continuation), roles, _roles_binding(roles), snapshot, r2_bytes,
            sha256_bytes(r2_bytes), r2_path, sha256_file(r2_path), _identity(r2_path),
            writer_fd, writer_claim, _identity(writer_claim), parent_fd, parent_path,
            (parent_stat.st_dev, parent_stat.st_ino), tuple(parent_chain),
        )
        _guard_context(ctx)
        # Prove a genuine fresh load before returning authority.
        _fresh_r2_module(ctx)
        return ctx
    except Exception:
        os.close(writer_fd)
        os.close(parent_fd)
        raise


@contextmanager
def _fresh_secure_r2(ctx: ReadyContext) -> Iterator[Any]:
    _guard_context(ctx)
    module = _fresh_r2_module(ctx)
    module.write_bytes_exclusive = r3.write_bytes_exclusive
    module.ensure_directory = r3.ensure_directory
    yield module


def run_stage_p(ctx: ReadyContext, role: str, *, device: str = "cuda:0", **kwargs: Any) -> dict[str, Any]:
    with _fresh_secure_r2(ctx) as module:
        function = module.run_stage_p
        if function.__module__ != module.__name__:
            raise PermissionError("R4 stage-P callable source changed")
        result = function(ctx.r2_context, role, device=device, **kwargs)
    _guard_context(ctx)
    return result


def run_stage_o(ctx: ReadyContext, role: str, **kwargs: Any) -> dict[str, Any]:
    with _fresh_secure_r2(ctx) as module:
        function = module.run_stage_o
        if function.__module__ != module.__name__:
            raise PermissionError("R4 stage-O callable source changed")
        result = function(ctx.r2_context, role, **kwargs)
    _guard_context(ctx)
    return result


def run_stage_e(ctx: ReadyContext, role: str, **kwargs: Any) -> dict[str, Any]:
    with _fresh_secure_r2(ctx) as module:
        function = module.run_stage_e
        if function.__module__ != module.__name__:
            raise PermissionError("R4 stage-E callable source changed")
        result = function(ctx.r2_context, role, **kwargs)
    _guard_context(ctx)
    return result


def seal_stage_m(ctx: ReadyContext) -> dict[str, Any]:
    with _fresh_secure_r2(ctx) as module:
        function = module.seal_stage_m
        if function.__module__ != module.__name__:
            raise PermissionError("R4 stage-M callable source changed")
        r2_result = function(ctx.r2_context)
    _guard_context(ctx)
    combined = Path(ctx.ready["outputs"]["combined_manifest"])
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
        "authorization_commit_id": ctx.bundle["commit_id"],
        "candidate_collection": {
            "path": os.fspath(combined), "sha256": sha256_file(combined),
            "schema": "boxfusion.ca1m_tr3d_xfit_r2_candidate_collection.v1",
        },
        "r2_execution_receipt": {
            "path": os.fspath(r2_receipt_path), "sha256": sha256_file(r2_receipt_path),
            "schema": "boxfusion.ca1m_tr3d_e961_terminal_inputs_exact80_receipt.v5.r2",
            "operational_authority": False,
        },
        "r4_preregistration": {"path": os.fspath(PREREGISTRATION_PATH), "sha256": sha256_file(PREREGISTRATION_PATH), "schema": PREREGISTRATION_SCHEMA},
        "r4_ready_config": {"path": os.fspath(READY_CONFIG_PATH), "sha256": ctx.ready_sha256, "schema": CONFIG_SCHEMA},
        "r4_run_authorization": {"path": os.fspath(RUN_AUTHORIZATION_PATH), "sha256": ctx.authorization_sha256, "schema": AUTH_SCHEMA},
        "r4_authorization_bundle": {"path": os.fspath(AUTHORIZATION_BUNDLE_PATH), "sha256": ctx.bundle_sha256, "schema": BUNDLE_SCHEMA},
        "ground_truth_access": False, "fold1_access": False,
        "official_validation_access": False,
    }
    target = Path(ctx.ready["outputs"]["manifest_root"]) / "M_EXACT80_R4_RECEIPT.json"
    r3.create_or_verify(target, canonical_json(wrapper), "R4 exact80 receipt")
    return {**wrapper, "receipt": {"path": os.fspath(target), "sha256": sha256_file(target), "schema": EXACT80_RECEIPT_SCHEMA}}


def run_all(ctx: ReadyContext, *, device: str = "cuda:0") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role in ROLE_ORDER:
        result[f"P:{role}"] = run_stage_p(ctx, role, device=device)
        result[f"O:{role}"] = run_stage_o(ctx, role)
        result[f"E:{role}"] = run_stage_e(ctx, role)
    result["M"] = seal_stage_m(ctx)
    return {
        "schema": "boxfusion.ca1m_tr3d_e961_terminal_inputs_run.v5.r4",
        "complete": True, "fresh_sha_bound_r2_per_stage": True,
        "bundle_commit_verified": True, "ground_truth_access": False,
        "results": result,
    }


__all__ = [
    "DEFAULT_CONFIG", "PREREGISTRATION_PATH", "READY_CONFIG_PATH",
    "RUN_AUTHORIZATION_PATH", "AUTHORIZATION_BUNDLE_PATH", "R3_INVALID_PATH",
    "CONFIG_SCHEMA", "PREREGISTRATION_SCHEMA", "AUTH_SCHEMA", "BUNDLE_SCHEMA", "EXACT80_RECEIPT_SCHEMA",
    "NAMESPACE", "ROLE_ORDER", "PendingOperationalInputs", "ReadyContext",
    "R2ContextSnapshot", "load_bound_producer_verifiers", "r3_invalidation_payload",
    "seal_r3_invalidation", "build_preregistration_payload", "seal_preregistration",
    "validate_static_config", "seal_ready_authorization_bundle",
    "validate_operational_ready", "run_stage_p", "run_stage_o", "run_stage_e",
    "seal_stage_m", "run_all", "sha256_file",
]
