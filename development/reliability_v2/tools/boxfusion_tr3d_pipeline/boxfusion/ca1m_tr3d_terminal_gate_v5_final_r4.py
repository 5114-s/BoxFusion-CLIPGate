"""Final R4, R6-only execution boundary for the CA-1M terminal gate v5.

The generic :mod:`ca1m_tr3d_terminal_gate_v5` module contains the numerical
dataset, three-head learner, OOF selection and geometry materializer.  This
revision adds the missing production boundary without changing that frozen
generic dependency:

* the sole candidate input is the canonical E961 terminal-inputs R6 exact80
  wrapper and collection;
* the R6 preregistration/ready/authorization/bundle commit chain is reopened
  and hash checked before an annotation loader or output path is reachable;
* one scientific preregistration freezes the exact double-OOF topology,
  learning heads, threshold grid, AP protocol and implementation bytes;
* a last-published run authorization binds that preregistration and an opaque
  CA-train annotation inventory; and
* fold 0 remains a reused-dev continuation diagnostic.  Fold 1 and official
  validation are deliberately absent from the runnable API.

Importing this module is side-effect free.  In particular it does not import
the evolving R6 producer implementation, inspect R6 outputs, open annotations,
create directories, or initialize CUDA.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import errno
import fcntl
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from . import ca1m_tr3d_terminal_gate_v5 as gate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PENDING_CONFIG = ROOT / "config/ca1m_tr3d_terminal_gate_v5_final_r4_pending.json"
MANIFEST_ROOT = ROOT / "manifests/ca1m_tr3d_terminal_gate_v5_final_r4"
PREREGISTRATION_PATH = MANIFEST_ROOT / "PREREGISTRATION.json"
PROTOCOL_PATH = MANIFEST_ROOT / "PREREGISTRATION_PROTOCOL.json"
READY_CONFIG_PATH = MANIFEST_ROOT / "READY_CONFIG.json"
RUN_AUTHORIZATION_PATH = MANIFEST_ROOT / "RUN_AUTHORIZATION.json"

R6_NAMESPACE = "ca1m_tr3d_e961_terminal_inputs_xfit_r2_v5_r6"
R6_ROOT = Path("/extra/ZhaoX") / R6_NAMESPACE
R6_COLLECTION_PATH = R6_ROOT / "manifests/CANDIDATE_COLLECTION_EXACT80.json"
R6_RECEIPT_PATH = R6_ROOT / "manifests/M_EXACT80_R6_RECEIPT.json"
R6_R2_EXECUTION_RECEIPT_PATH = R6_ROOT / "manifests/M_EXACT80_R2_RECEIPT.json"
R6_MANIFEST_ROOT = ROOT / "manifests/ca1m_tr3d_e961_terminal_inputs_v5_r6"
R6_PREREGISTRATION_PATH = R6_MANIFEST_ROOT / "PREREGISTRATION.json"
R6_READY_CONFIG_PATH = R6_MANIFEST_ROOT / "READY_CONFIG.json"
R6_RUN_AUTHORIZATION_PATH = R6_MANIFEST_ROOT / "RUN_AUTHORIZATION.json"
R6_AUTHORIZATION_BUNDLE_PATH = R6_MANIFEST_ROOT / "AUTHORIZATION_BUNDLE.json"
R6_CONFIG_PATH = ROOT / "config/ca1m_tr3d_e961_terminal_inputs_v5_r6_pending.json"
R6_CORE_PATH = ROOT / "boxfusion/ca1m_tr3d_e961_terminal_inputs_v5_r6.py"
R6_PREREGISTRATION_SHA256 = (
    "60267f07a77773f3e83e05e5a223a70f4e031b30f90724b02f80fecbc0b9edaa"
)
R6_CONFIG_SHA256 = (
    "c3641b8c1388bafe9f62fa97a9e119d17844e9e2fab605f8bd19906cddb03f0c"
)
R6_CORE_SHA256 = (
    "8f7bed7ee102923861d34de276092bc4645e0caf0fc7cc1eb94c0e7aa50e1b32"
)

LEGACY_FINAL_INVALID_PATH = (
    ROOT
    / "manifests/ca1m_tr3d_terminal_gate_v5_final/PENDING_REVISION_INVALID.json"
)
LEGACY_FINAL_INVALID_SCHEMA = (
    "boxfusion.ca1m_tr3d_terminal_gate_pending_revision_invalid.v5.final"
)
LEGACY_FINAL_INVALID_SHA256 = (
    "4bc9467df292a07e1fa34e7597a7bf47a9190299ad47c1d32edb7d5756beb5b1"
)
R3_PROTOCOL_PATH = (
    ROOT
    / "manifests/ca1m_tr3d_terminal_gate_v5_final_r3/"
    "PREREGISTRATION_PROTOCOL.json"
)
R3_PROTOCOL_SHA256 = (
    "aea2edefcd9dff937663d3f336f64df3e5c3a658ab5c6ac1ed7d751dff045edb"
)
R3_PROTOCOL_SCHEMA = (
    "boxfusion.ca1m_tr3d_terminal_gate_preregistration_protocol.v5.final.r3"
)
R3_PROTOCOL_INVALID_PATH = (
    ROOT
    / "manifests/ca1m_tr3d_terminal_gate_v5_final_r3/"
    "PREREGISTRATION_PROTOCOL_INVALID.json"
)
R3_PROTOCOL_INVALID_SHA256 = (
    "9a945949feb53709e257d89b45304fc1799ac89dd01adf7027f8a13368c52fea"
)
R3_PROTOCOL_INVALID_SCHEMA = (
    "boxfusion.ca1m_tr3d_terminal_gate_preregistration_protocol_invalid.v5.final.r3"
)
R3_RUNNER_PATH = ROOT / "tools/run_ca1m_tr3d_terminal_gate_v5_final_r3.py"
R3_RUNNER_TOMBSTONE_SHA256 = (
    "d0c8c42ebbcddb297b5151a6ba50c509837a5ec44ef21d0fbf7006ec43b45d09"
)

PENDING_SCHEMA = "boxfusion.ca1m_tr3d_terminal_gate_pending_config.v5.final.r4"
PROTOCOL_SCHEMA = "boxfusion.ca1m_tr3d_terminal_gate_preregistration_protocol.v5.final.r4"
PREREGISTRATION_SCHEMA = "boxfusion.ca1m_tr3d_terminal_gate_preregistration.v5.final.r4"
READY_SCHEMA = "boxfusion.ca1m_tr3d_terminal_gate_ready_config.v5.final.r4"
AUTHORIZATION_SCHEMA = "boxfusion.ca1m_tr3d_terminal_gate_run_authorization.v5.final.r4"
RUN_SCHEMA = "boxfusion.ca1m_tr3d_terminal_gate_run.v5.final.r4"
STOP_SCHEMA = "boxfusion.ca1m_tr3d_terminal_gate_stop.v5.final.r4"
MATERIALIZATION_COLLECTION_SCHEMA = (
    "boxfusion.ca1m_tr3d_geometry_materialization_collection.v5.final.r4"
)
GT_INVENTORY_SCHEMA = "boxfusion.ca1m_tr3d_benefit_gate_gt_shadow_inventory.v1"

R6_RECEIPT_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_exact80_receipt.v5.r6"
R6_R2_EXECUTION_RECEIPT_SCHEMA = (
    "boxfusion.ca1m_tr3d_e961_terminal_inputs_exact80_receipt.v5.r2"
)
R6_CONFIG_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_config.v5.r6"
R6_PREREGISTRATION_SCHEMA = (
    "boxfusion.ca1m_tr3d_e961_terminal_inputs_preregistration.v5.r6"
)
R6_AUTHORIZATION_SCHEMA = (
    "boxfusion.ca1m_tr3d_e961_terminal_inputs_run_authorization.v5.r6"
)
R6_BUNDLE_SCHEMA = (
    "boxfusion.ca1m_tr3d_e961_terminal_inputs_authorization_bundle.v5.r6"
)
OUTER_RUN_SCHEMA = "boxfusion.tr3d.ca1m_e961_outer_train_run.r2"
INNER_RUN_SCHEMA = "boxfusion.tr3d.ca1m_e961_inner_train_run.r2"

NAMESPACE = "ca1m_tr3d_terminal_gate_v5_final_r4"
RUNTIME_ROOT = Path("/extra/ZhaoX") / NAMESPACE
CANONICAL_RUNTIME_ROOT = Path("/extra/ZhaoX") / NAMESPACE
OUTPUT_PARENT_PATH = Path("/extra/ZhaoX")
RUN_CLAIM_PATH = OUTPUT_PARENT_PATH / f".{NAMESPACE}.run.claim"

GT_INVENTORY_PATH = (
    ROOT
    / "manifests/ca1m_tr3d_benefit_gate_final_base_v4/"
    "derived_train_gt_shadow_inventory_v1.json"
)
GT_INVENTORY_SHA256 = (
    "6c3bdfd666ca49558ac390197abeec588949f05e33d9c8a18d1b5c8326d9e9a7"
)
GT_INVENTORY_CONTENT_SHA256 = (
    "133db05b3296be5ce56c4e67db89c84a344a5803996cb0960c9bb044944ba50d"
)
GT_SHADOW_ROOT = (
    ROOT
    / "inputs/ca1m_tr3d_benefit_gate_final_base_v4/derived_train_gt_fitdev80"
)
GT_SOURCE_ROOT = Path("/extra/ZhaoX/boxfusion_ca1m_native_b6_train100_v1")
GT_SOURCE_DATASET_MANIFEST_PATH = (
    ROOT / "datasets/ca1m_native_b6_final_base_train100_v2.manifest.json"
)
GT_SOURCE_DATASET_MANIFEST_SHA256 = (
    "2330dc422671aa6de7d0b6ee07b1e8bdf1d934cef24a54f12285d6882df22e68"
)
GT_OOF_SIDECAR_PATH = ROOT / "models/ca1m_native_b6_final_base_oof_row_scores_v2.npz"
GT_OOF_SIDECAR_SHA256 = (
    "82b5e70c635958398c04b0e3ba5dbf25203b61bcabd725330bd68812d156e5ed"
)

IMPLEMENTATION_PATHS = {
    "generic_gate_core": ROOT / "boxfusion/ca1m_tr3d_terminal_gate_v5.py",
    "final_gate_boundary": ROOT / "boxfusion/ca1m_tr3d_terminal_gate_v5_final_r4.py",
    "static_preflight": ROOT / "tools/preflight_ca1m_tr3d_terminal_gate_v5_final_r4.py",
    "scientific_sealer": ROOT / "tools/seal_ca1m_tr3d_terminal_gate_v5_final_r4.py",
    "runner": ROOT / "tools/run_ca1m_tr3d_terminal_gate_v5_final_r4.py",
    "regression_tests": ROOT / "tests/test_ca1m_tr3d_terminal_gate_v5_final_r4.py",
}
GENERIC_GATE_SHA256 = (
    "818b3aa60e1706f8dc03fde6bb872d20e41f31b18e6df8c6dd4ee45ddc1e812d"
)

OUTPUT_PATHS = {
    "fit_dataset": RUNTIME_ROOT / "datasets/fold234_fit60.npz",
    "fit_dataset_manifest": RUNTIME_ROOT / "datasets/fold234_fit60.manifest.json",
    "oof_predictions": RUNTIME_ROOT / "results/fold234_scene_grouped_oof.npz",
    "threshold_receipt": RUNTIME_ROOT / "results/fold234_threshold_receipt.json",
    "exploratory_policy": RUNTIME_ROOT / "models/gate_v5.non_deployable.json",
    "fold0_dataset": RUNTIME_ROOT / "datasets/fold0_reused_dev20.npz",
    "fold0_dataset_manifest": RUNTIME_ROOT / "datasets/fold0_reused_dev20.manifest.json",
    "fold0_report": RUNTIME_ROOT / "reports/fold0_reused_dev_diagnostic.json",
    "materialization_root": RUNTIME_ROOT / "materialized_fold0",
    "materialization_manifest": RUNTIME_ROOT / "reports/fold0_materialization_manifest.json",
    "stop_receipt": RUNTIME_ROOT / "reports/STOP.json",
    "run_receipt": RUNTIME_ROOT / "reports/RUN_RECEIPT.json",
}

_SHA = re.compile(r"^[0-9a-f]{64}$")
_SCENE = re.compile(r"^[0-9]{8}$")
_FORBIDDEN_CANDIDATE_PATH_TOKENS = (
    "scannet",
    "m_exact80_r4_receipt",
    "m_exact80_r5_receipt",
    "m_exact80_r2_receipt",
    "ca1m_tr3d_e961_terminal_inputs_xfit_r2_v5_r4",
    "ca1m_tr3d_e961_terminal_inputs_xfit_r2_v5_r5",
    "ca1m_tr3d_benefit_gate_final_base_v4",
    "ca1m_tr3d_terminal_ca_native_train100_v4",
    "ca1m_tr3d_benefit_final_base_v4",
    "ca1m_fg_scratch_seed0_fp32_gb16_v1",
)


class PendingR6Inputs(PermissionError):
    """Raised before GT/output when the canonical R6 commit is unavailable."""


class FinalGateProtocolError(RuntimeError):
    """A final-gate provenance or execution invariant was violated."""


def _json_plain(value: Any) -> Any:
    """Thaw immutable authority values into strictly JSON-compatible data."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if type(key) is not str:
                raise TypeError("canonical JSON object keys must be strings")
            result[key] = _json_plain(child)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_plain(child) for child in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError(f"canonical JSON value has unsupported type: {type(value).__name__}")


def canonical_json(value: Mapping[str, Any]) -> bytes:
    plain = _json_plain(value)
    return (json.dumps(plain, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _deep_immutable(value: Any) -> Any:
    """Detach and recursively freeze JSON-like authority state."""

    if isinstance(value, Mapping):
        return MappingProxyType({
            key: _deep_immutable(child) for key, child in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_deep_immutable(child) for child in value)
    if isinstance(value, set):
        return frozenset(_deep_immutable(child) for child in value)
    return value


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    _, payload, _ = stable_bytes(
        Path(path), "SHA256 input", immutable=False, allow_empty=True,
    )
    return sha256_bytes(payload)


def _sha(value: Any, name: str) -> str:
    result = str(value)
    if _SHA.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return result


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _canonical_absolute(path: Path, name: str) -> Path:
    value = Path(path)
    if not value.is_absolute() or ".." in value.parts:
        raise ValueError(f"{name} must be absolute and normalized")
    normalized = Path(os.path.normpath(os.fspath(value)))
    if normalized != value:
        raise ValueError(f"{name} is not lexically canonical")
    return value


_DIR_FLAGS = (
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
)


def _open_dir_chain(
    path: Path, name: str,
) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Open a complete absolute directory chain without following symlinks."""

    value = _canonical_absolute(Path(path), name)
    descriptor = os.open(value.anchor, _DIR_FLAGS)
    identities: list[tuple[int, int]] = []
    try:
        root_info = os.fstat(descriptor)
        identities.append((int(root_info.st_dev), int(root_info.st_ino)))
        for component in value.parts[1:]:
            next_descriptor = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
            info = os.fstat(descriptor)
            if not stat.S_ISDIR(info.st_mode):
                raise NotADirectoryError(f"{name} component is not a directory")
            identities.append((int(info.st_dev), int(info.st_ino)))
        return descriptor, tuple(identities)
    except BaseException:
        os.close(descriptor)
        raise


def _verify_dir_chain(
    path: Path, expected: Sequence[tuple[int, int]], name: str,
) -> None:
    descriptor, observed = _open_dir_chain(path, name)
    os.close(descriptor)
    if tuple(observed) != tuple(expected):
        raise RuntimeError(f"{name} directory identity changed")


def _assert_no_symlink_chain(path: Path, name: str) -> None:
    absolute = _absolute_lexical(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            raise FileNotFoundError(f"missing {name}: {absolute}") from None
        if stat.S_ISLNK(mode):
            raise ValueError(f"{name} path contains a symlink: {current}")


def _identity(info: os.stat_result) -> dict[str, int]:
    return {
        "st_dev": int(info.st_dev), "st_ino": int(info.st_ino),
        "st_size": int(info.st_size), "st_mtime_ns": int(info.st_mtime_ns),
        "st_nlink": int(info.st_nlink),
    }


def stable_bytes(
    path: Path, name: str, *, immutable: bool = True, allow_empty: bool = False,
) -> tuple[Path, bytes, dict[str, int]]:
    """Read bytes through a pinned parent/inode; mode is never provenance."""

    source = _canonical_absolute(_absolute_lexical(path), name)
    parent_fd, parent_chain = _open_dir_chain(source.parent, f"{name} parent")
    descriptor = -1
    try:
        descriptor = os.open(
            source.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        before_fd = os.fstat(descriptor)
        if not stat.S_ISREG(before_fd.st_mode):
            raise ValueError(f"{name} must be a regular file: {source}")
        if immutable and before_fd.st_mode & 0o222:
            raise ValueError(f"{name} must be read-only: {source}")
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
        after_fd = os.fstat(descriptor)
        after_path = os.stat(
            source.name, dir_fd=parent_fd, follow_symlinks=False,
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)
    _verify_dir_chain(source.parent, parent_chain, f"{name} parent")
    identities = (
        _identity(before_fd),
        _identity(after_fd), _identity(after_path),
    )
    if len({tuple(value.items()) for value in identities}) != 1:
        raise RuntimeError(f"{name} inode changed while reading: {source}")
    payload = b"".join(blocks)
    if len(payload) != before_fd.st_size or (not payload and not allow_empty):
        raise ValueError(f"{name} is empty or size-inconsistent: {source}")
    return source, payload, identities[0]


def stable_json(
    path: Path, name: str, *, schema: str | None = None,
    immutable: bool = True,
) -> tuple[Path, dict[str, Any], bytes, dict[str, int]]:
    source, data, identity = stable_bytes(path, name, immutable=immutable)
    try:
        value = json.loads(data)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    if schema is not None and value.get("schema") != schema:
        raise ValueError(f"{name} schema differs")
    return source, value, data, identity


def artifact_record(
    path: Path, name: str, *, schema: str | None = None,
    immutable: bool = True,
) -> dict[str, Any]:
    source, data, identity = stable_bytes(path, name, immutable=immutable)
    result: dict[str, Any] = {
        "path": os.fspath(source), "sha256": sha256_bytes(data),
        "identity": identity,
    }
    if schema is not None:
        result["schema"] = schema
    return result


def validate_artifact_record(
    value: Any, name: str, *, schema: str | None = None,
    canonical_path: Path | None = None, require_identity: bool = False,
    immutable: bool = True,
) -> tuple[Path, bytes, dict[str, int]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} record must be an object")
    path = _absolute_lexical(Path(str(value.get("path", ""))))
    if canonical_path is not None and path != _absolute_lexical(canonical_path):
        raise ValueError(f"{name} canonical path differs")
    source, data, identity = stable_bytes(path, name, immutable=immutable)
    if sha256_bytes(data) != _sha(value.get("sha256"), f"{name} SHA256"):
        raise ValueError(f"{name} SHA256 differs")
    if schema is not None and value.get("schema") != schema:
        raise ValueError(f"{name} record schema differs")
    if require_identity and value.get("identity") != identity:
        raise ValueError(f"{name} inode identity differs")
    return source, data, identity


def _record_json(
    value: Any, name: str, *, schema: str, canonical_path: Path | None = None,
    require_identity: bool = False, immutable: bool = True,
) -> tuple[Path, dict[str, Any], bytes, dict[str, int]]:
    path, data, identity = validate_artifact_record(
        value, name, schema=schema, canonical_path=canonical_path,
        require_identity=require_identity, immutable=immutable,
    )
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not JSON") from error
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError(f"{name} payload schema differs")
    return path, payload, data, identity


def _ensure_directory_below(anchor: Path, target: Path, name: str) -> None:
    """Create only descendants of an already existing, inode-bound anchor."""

    anchor = _canonical_absolute(_absolute_lexical(anchor), f"{name} anchor")
    target = _canonical_absolute(_absolute_lexical(target), name)
    try:
        relative = target.relative_to(anchor)
    except ValueError:
        raise PermissionError(f"{name} is outside its fixed parent") from None
    descriptor, anchor_chain = _open_dir_chain(anchor, f"{name} anchor")
    try:
        for component in relative.parts:
            try:
                next_descriptor = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
            except FileNotFoundError:
                os.mkdir(component, 0o700, dir_fd=descriptor)
                os.fsync(descriptor)
                next_descriptor = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
            info = os.fstat(descriptor)
            if not stat.S_ISDIR(info.st_mode):
                raise NotADirectoryError(f"{name} component is not a directory")
    finally:
        os.close(descriptor)
    _verify_dir_chain(anchor, anchor_chain, f"{name} anchor")
    check_fd, _ = _open_dir_chain(target, name)
    os.close(check_fd)


def _exclusive_bytes_fuse(path: Path, payload: bytes, name: str) -> Path:
    """Publish once using hardlink atomicity, without trusting POSIX mode bits."""

    target = _canonical_absolute(_absolute_lexical(path), name)
    parent_fd, parent_chain = _open_dir_chain(target.parent, f"{name} parent")
    temporary = f".{target.name}.tmp.{os.getpid()}.{secrets.token_hex(12)}"
    descriptor = -1
    linked = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o444,
            dir_fd=parent_fd,
        )
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            if count < 1:
                raise OSError(f"short create-only write for {name}")
            view = view[count:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(
            temporary, target.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        linked = True
        os.fsync(parent_fd)
        os.unlink(temporary, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(parent_fd)
    _verify_dir_chain(target.parent, parent_chain, f"{name} parent")
    source, observed, identity = stable_bytes(
        target, name, immutable=False, allow_empty=True,
    )
    if not linked or observed != payload or identity["st_nlink"] != 1:
        raise RuntimeError(f"published {name} inode/bytes differ")
    return source


def _exclusive_bytes_at_fd(
    parent_fd: int, leaf: str, payload: bytes, name: str,
) -> dict[str, int]:
    """Publish a leaf relative to one already authorized directory FD."""

    if leaf in {"", ".", ".."} or "/" in leaf:
        raise ValueError(f"invalid {name} output leaf")
    try:
        os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(f"refusing existing {name}: {leaf}")
    temporary = f".{leaf}.tmp.{os.getpid()}.{secrets.token_hex(12)}"
    descriptor = -1
    linked = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o444, dir_fd=parent_fd,
        )
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            if count < 1:
                raise OSError(f"short create-only write for {name}")
            view = view[count:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(
            temporary, leaf, src_dir_fd=parent_fd, dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        linked = True
        os.fsync(parent_fd)
        os.unlink(temporary, dir_fd=parent_fd)
        os.fsync(parent_fd)
        verify_fd = os.open(
            leaf, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd,
        )
        try:
            before = os.fstat(verify_fd)
            chunks: list[bytes] = []
            while True:
                block = os.read(verify_fd, 1024 * 1024)
                if not block:
                    break
                chunks.append(block)
            after = os.fstat(verify_fd)
        finally:
            os.close(verify_fd)
        canonical = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        identities = (_identity(before), _identity(after), _identity(canonical))
        if (
            len({tuple(value.items()) for value in identities}) != 1
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or b"".join(chunks) != payload
        ):
            raise RuntimeError(f"published {name} FD/inode/bytes differ")
        return identities[0]
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        # A linked target is intentionally retained.  The one-use run claim
        # makes any partial publication permanently non-resumable.
        raise
    finally:
        if not linked:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass


def _read_bytes_at_fd(
    parent_fd: int, leaf: str, name: str,
) -> tuple[bytes, dict[str, int]]:
    """Read one regular leaf without reopening its parent path."""

    if leaf in {"", ".", ".."} or "/" in leaf:
        raise ValueError(f"invalid {name} leaf")
    descriptor = os.open(
        leaf, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{name} is not regular")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        current = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    finally:
        os.close(descriptor)
    identities = (_identity(before), _identity(after), _identity(current))
    if len({tuple(value.items()) for value in identities}) != 1:
        raise RuntimeError(f"{name} inode changed during held-FD read")
    payload = b"".join(chunks)
    if len(payload) != before.st_size:
        raise RuntimeError(f"{name} size changed during held-FD read")
    return payload, identities[0]


def _open_runtime_output_parent(
    capability: Any, relative_parent: Path, name: str,
) -> int:
    """Open/create a runtime descendant only from the held root description."""

    authority = _guard_run_capability(capability)
    if relative_parent.is_absolute() or ".." in relative_parent.parts:
        raise PermissionError(f"{name} relative parent differs")
    descriptor = os.dup(capability.runtime_fd)
    traversed: list[str] = []
    try:
        for component in relative_parent.parts:
            if component in {"", ".", ".."}:
                raise PermissionError(f"{name} contains invalid component")
            traversed.append(component)
            key = "/".join(traversed)
            try:
                next_descriptor = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
            except FileNotFoundError:
                # mkdirat is the ownership decision.  If another writer wins,
                # FileExistsError is not converted into reuse.
                os.mkdir(component, 0o700, dir_fd=descriptor)
                os.fsync(descriptor)
                next_descriptor = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
                created = os.fstat(next_descriptor)
                authority.runtime_directories[key] = (
                    int(created.st_dev), int(created.st_ino)
                )
            current = os.fstat(next_descriptor)
            expected = authority.runtime_directories.get(key)
            if (
                expected is None
                or not stat.S_ISDIR(current.st_mode)
                or (int(current.st_dev), int(current.st_ino)) != expected
            ):
                os.close(next_descriptor)
                raise PermissionError(f"{name} directory is not claim-owned")
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _write_runtime_bytes_fuse(
    path: Path, payload: bytes, name: str, *, runtime_root: Path,
    capability: Any = None,
) -> Path:
    target = _canonical_absolute(_absolute_lexical(path), name)
    root = _canonical_absolute(_absolute_lexical(runtime_root), "runtime root")
    try:
        relative = target.relative_to(root)
    except ValueError:
        raise PermissionError(f"{name} is outside the authorized runtime root") from None
    if capability is not None:
        authority = _guard_run_capability(capability)
        if authority.context.outputs != OUTPUT_PATHS or root != _absolute_lexical(RUNTIME_ROOT):
            raise PermissionError("formal runtime writer authority differs")
        parent_fd = _open_runtime_output_parent(
            capability, relative.parent, f"{name} output parent",
        )
        try:
            published_identity = _exclusive_bytes_at_fd(
                parent_fd, relative.name, payload, name,
            )
        finally:
            os.close(parent_fd)
        _guard_run_capability(capability)
        source, observed, identity = stable_bytes(
            target, name, immutable=False, allow_empty=True,
        )
        if observed != payload or identity != published_identity:
            raise RuntimeError(f"canonical {name} differs after FD publication")
        return source
    _ensure_directory_below(root.parent, target.parent, f"{name} output parent")
    return _exclusive_bytes_fuse(target, payload, name)


def _gate_regular_fuse(path: Path, name: str, *, immutable: bool = True) -> Path:
    del immutable
    source, _, _ = stable_bytes(path, name, immutable=False)
    return source


def _gate_json_fuse(
    path: Path, name: str, *, immutable: bool = True,
) -> tuple[Path, dict[str, Any]]:
    del immutable
    source, value, _, _ = stable_json(path, name, immutable=False)
    return source, value


def _gate_record_fuse(
    value: Any, name: str, *, schema: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} record must be an object")
    expected_schema = schema if schema is not None else value.get("schema")
    if not expected_schema:
        raise ValueError(f"{name} schema is missing")
    path, payload, _, _ = _record_json(
        value, name, schema=str(expected_schema), immutable=False,
    )
    return path, payload


def _load_candidate_evidence_fuse(
    path: Path, *, expected_scene: str | None = None,
) -> gate.CandidateEvidenceV5:
    """Byte-for-byte generic evidence parser with one FD-bound FUSE read."""

    source, data, _ = stable_bytes(path, "v5 candidate evidence", immutable=False)
    gate._assert_safe_runtime_path(source, "candidate evidence")
    if not source.name.endswith("_ca1m_tr3d_candidate_evidence_xfit_r2_v5.npz"):
        raise ValueError("v5 evidence filename differs")
    with np.load(BytesIO(data), allow_pickle=False) as archive:
        required = {
            "schema", "complete", "create_only", "ground_truth_access", "fold1_access",
            "official_validation_access", "scene_id", "fold_id", "producer_role",
            "producer_checkpoint_sha256", "producer_train_folds", "training_receipt_sha256",
            "outer_continuation_receipt_sha256", "anchor_score_source", "b6_oof_sidecar_sha256",
            "feature_names", "feature_groups_json", "candidate_corners", "candidate_rows",
            "candidate_scores", "anchor_indices", "features", "anchor_corners", "anchor_rows",
            "anchor_scores_oof", "candidate_corners_sha256", "candidate_feature_sha256",
            "anchor_identity_sha256",
        }
        if set(archive.files) != required:
            raise ValueError("v5 candidate evidence key set differs")
        if (
            gate._scalar(archive, "schema") != gate.EVIDENCE_SCHEMA
            or bool(gate._scalar(archive, "complete")) is not True
            or bool(gate._scalar(archive, "create_only")) is not True
            or bool(gate._scalar(archive, "ground_truth_access")) is not False
            or bool(gate._scalar(archive, "fold1_access")) is not False
            or bool(gate._scalar(archive, "official_validation_access")) is not False
            or gate._scalar(archive, "anchor_score_source") != gate.ANCHOR_SCORE_SOURCE
            or tuple(np.asarray(archive["feature_names"]).astype(str).tolist())
            != gate.FEATURE_NAMES
            or json.loads(str(gate._scalar(archive, "feature_groups_json")))
            != {key: list(value) for key, value in gate.REQUIRED_FEATURE_GROUPS.items()}
        ):
            raise ValueError("v5 candidate evidence scalar contract differs")
        scene = gate._scene(gate._scalar(archive, "scene_id"))
        fold = int(gate._scalar(archive, "fold_id"))
        role = str(gate._scalar(archive, "producer_role"))
        checkpoint_sha = gate._sha(
            gate._scalar(archive, "producer_checkpoint_sha256"), "checkpoint SHA256",
        )
        receipt_sha = gate._sha(
            gate._scalar(archive, "training_receipt_sha256"), "receipt SHA256",
        )
        continuation_sha = gate._sha(
            gate._scalar(archive, "outer_continuation_receipt_sha256"),
            "continuation SHA256",
        )
        b6_sha = gate._sha(
            gate._scalar(archive, "b6_oof_sidecar_sha256"), "B6 OOF SHA256",
        )
        arrays = {
            key: np.array(archive[key], copy=True)
            for key in (
                "candidate_corners", "candidate_rows", "candidate_scores",
                "anchor_indices", "features", "anchor_corners", "anchor_rows",
                "anchor_scores_oof", "producer_train_folds",
            )
        }
        recorded = {
            key: gate._sha(gate._scalar(archive, key), key)
            for key in (
                "candidate_corners_sha256", "candidate_feature_sha256",
                "anchor_identity_sha256",
            )
        }
    if expected_scene is not None and scene != gate._scene(expected_scene):
        raise ValueError("candidate evidence scene differs")
    train_folds, output_fold = gate.ROLE_SPECS.get(role, ((), -1))
    producer_folds = tuple(arrays.pop("producer_train_folds").tolist())
    if fold != output_fold or fold in train_folds or producer_folds != train_folds:
        raise ValueError("v5 evidence detector OOF topology differs")
    rebuilt = gate._evidence_payload(
        scene_id=scene, fold_id=fold, producer_role=role,
        producer_checkpoint_sha256=checkpoint_sha,
        training_receipt_sha256=receipt_sha,
        outer_continuation_receipt_sha256=continuation_sha,
        b6_oof_sidecar_sha256=b6_sha,
        candidate_corners=arrays["candidate_corners"],
        candidate_rows=arrays["candidate_rows"],
        candidate_scores=arrays["candidate_scores"],
        anchor_indices=arrays["anchor_indices"], features=arrays["features"],
        anchor_corners=arrays["anchor_corners"],
        anchor_scores=arrays["anchor_scores_oof"],
    )
    for key, digest in recorded.items():
        if str(np.asarray(rebuilt[key]).item()) != digest:
            raise ValueError(f"v5 evidence {key} differs")
    if not np.array_equal(arrays["anchor_rows"], rebuilt["anchor_rows"]):
        raise ValueError("v5 anchor row identity differs")
    return gate.CandidateEvidenceV5(
        path=source, scene_id=scene, fold_id=fold, producer_role=role,
        producer_checkpoint_sha256=checkpoint_sha,
        training_receipt_sha256=receipt_sha,
        outer_continuation_receipt_sha256=continuation_sha,
        b6_oof_sidecar_sha256=b6_sha,
        candidate_corners=gate._readonly(arrays["candidate_corners"]),
        candidate_rows=gate._readonly(arrays["candidate_rows"]),
        candidate_scores=gate._readonly(arrays["candidate_scores"]),
        anchor_indices=gate._readonly(arrays["anchor_indices"]),
        features=gate._readonly(arrays["features"]),
        anchor_corners=gate._readonly(arrays["anchor_corners"]),
        anchor_rows=gate._readonly(arrays["anchor_rows"]),
        anchor_scores=gate._readonly(arrays["anchor_scores_oof"]),
        candidate_corners_sha256=recorded["candidate_corners_sha256"],
        candidate_feature_sha256=recorded["candidate_feature_sha256"],
        anchor_identity_sha256=recorded["anchor_identity_sha256"],
    )


@contextmanager
def _fuse_gate_io(
    *, runtime_root: Path | None = None, capability: Any = None,
) -> Iterator[None]:
    """Adapt frozen generic I/O to FUSE while leaving its science untouched."""

    names = (
        "_regular", "_json", "_record", "load_candidate_evidence_v5",
        "sha256_file", "write_bytes_create_only", "write_json_create_only",
        "write_npz_create_only",
    )
    original = {name: getattr(gate, name) for name in names}
    gate._regular = _gate_regular_fuse
    gate._json = _gate_json_fuse
    gate._record = _gate_record_fuse
    gate.load_candidate_evidence_v5 = _load_candidate_evidence_fuse
    gate.sha256_file = sha256_file
    if runtime_root is not None:
        def write_bytes(path: Path, payload: bytes, name: str) -> Path:
            return _write_runtime_bytes_fuse(
                path, payload, name, runtime_root=runtime_root,
                capability=capability,
            )

        def write_json(path: Path, payload: Mapping[str, Any], name: str) -> Path:
            return write_bytes(path, gate._canonical_json(payload), name)

        def write_npz(path: Path, payload: Mapping[str, Any], name: str) -> Path:
            stream = BytesIO()
            np.savez_compressed(stream, **payload)
            return write_bytes(path, stream.getvalue(), name)

        gate.write_bytes_create_only = write_bytes
        gate.write_json_create_only = write_json
        gate.write_npz_create_only = write_npz
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(gate, name, value)


def _legacy_invalidation_pending_record() -> dict[str, Any]:
    return {
        "state": "invalidated",
        "path": os.fspath(LEGACY_FINAL_INVALID_PATH),
        "sha256": LEGACY_FINAL_INVALID_SHA256,
        "schema": LEGACY_FINAL_INVALID_SCHEMA,
        "operational_authority": False,
    }


def validate_legacy_final_invalidation() -> dict[str, Any]:
    path, value, data, identity = stable_json(
        LEGACY_FINAL_INVALID_PATH,
        "legacy terminal-gate final R1 invalidation",
        schema=LEGACY_FINAL_INVALID_SCHEMA,
    )
    if (
        sha256_bytes(data) != LEGACY_FINAL_INVALID_SHA256
        or value.get("complete") is not True
        or value.get("create_only") is not True
        or value.get("invalid") is not True
        or value.get("operational_authority") is not False
        or value.get("namespace") != "ca1m_tr3d_terminal_gate_v5_final"
        or value.get("superseded_by_namespace") != NAMESPACE
        or value.get("never_scientifically_preregistered") is not True
        or value.get("never_ready_authorized") is not True
        or value.get("runtime_namespace_created") is not False
        or value.get("ground_truth_access") is not False
        or value.get("gpu_started") is not False
        or (value.get("former_future_input") or {}).get("ever_committed") is not False
    ):
        raise PermissionError("legacy terminal-gate final R1 invalidation differs")
    invalid_upstream = value.get("invalid_upstream") or {}
    upstream_path, upstream, _, _ = _record_json(
        invalid_upstream,
        "legacy final invalid R4 upstream",
        schema="boxfusion.ca1m_tr3d_e961_terminal_inputs_preregistration_invalid.v5.r4",
        canonical_path=(
            ROOT
            / "manifests/ca1m_tr3d_e961_terminal_inputs_v5_r4/PREREGISTRATION_INVALID.json"
        ),
    )
    if upstream.get("invalid") is not True or upstream_path == R6_PREREGISTRATION_PATH:
        raise PermissionError("legacy final R4 upstream invalidation differs")
    former = value.get("former_implementation") or {}
    expected_live = {
        "core": ROOT / "boxfusion/ca1m_tr3d_terminal_gate_v5_final.py",
        "pending_config": ROOT / "config/ca1m_tr3d_terminal_gate_v5_final_pending.json",
        "preflight": ROOT / "tools/preflight_ca1m_tr3d_terminal_gate_v5_final.py",
        "sealer": ROOT / "tools/seal_ca1m_tr3d_terminal_gate_v5_final.py",
        "tests": ROOT / "tests/test_ca1m_tr3d_terminal_gate_v5_final.py",
    }
    if set(former) != {*expected_live, "runner_before_tombstone"}:
        raise PermissionError("legacy final implementation invalidation inventory differs")
    for name, canonical_path in expected_live.items():
        validate_artifact_record(
            former[name], f"legacy final {name}",
            canonical_path=canonical_path, immutable=False,
        )
    runner = former["runner_before_tombstone"]
    if (
        runner.get("path")
        != os.fspath(ROOT / "tools/run_ca1m_tr3d_terminal_gate_v5_final.py")
        or runner.get("sha256")
        != "5e41f404e5e6c8a7f1277f34f0e4be0566d2399633eb1ba3851a4cf7d04471f8"
    ):
        raise PermissionError("legacy final pre-tombstone runner binding differs")
    return {
        "path": os.fspath(path), "sha256": sha256_bytes(data),
        "schema": LEGACY_FINAL_INVALID_SCHEMA, "identity": identity,
        "operational_authority": False,
    }


def _predecessor_invalidation_pending_record() -> dict[str, Any]:
    return {
        "state": "invalidated", "path": os.fspath(R3_PROTOCOL_INVALID_PATH),
        "sha256": R3_PROTOCOL_INVALID_SHA256,
        "schema": R3_PROTOCOL_INVALID_SCHEMA, "operational_authority": False,
    }


def validate_r3_protocol_invalidation() -> dict[str, Any]:
    path, value, data, identity = stable_json(
        R3_PROTOCOL_INVALID_PATH, "final-R3 protocol invalidation",
        schema=R3_PROTOCOL_INVALID_SCHEMA,
    )
    protocol = value.get("invalidated_protocol") or {}
    if (
        sha256_bytes(data) != R3_PROTOCOL_INVALID_SHA256
        or set(value) != {
            "schema", "complete", "create_only", "invalid",
            "operational_authority", "namespace", "superseded_by_namespace",
            "reason", "invalidated_protocol", "runner_before_tombstone",
            "never_instance_preregistered", "never_ready_authorized",
            "access_before_invalidation",
        }
        or value.get("complete") is not True
        or value.get("create_only") is not True
        or value.get("invalid") is not True
        or value.get("operational_authority") is not False
        or value.get("namespace") != "ca1m_tr3d_terminal_gate_v5_final_r3"
        or value.get("superseded_by_namespace") != NAMESPACE
        or value.get("never_instance_preregistered") is not True
        or value.get("never_ready_authorized") is not True
        or (value.get("reason") or {}).get("code")
        != "CALLER_CONTEXT_INJECTION_AND_OUTPUT_PARENT_REOPEN_GAP"
        or any((value.get("access_before_invalidation") or {}).values())
        or protocol != {
            "path": os.fspath(R3_PROTOCOL_PATH), "sha256": R3_PROTOCOL_SHA256,
            "schema": R3_PROTOCOL_SCHEMA,
        }
        or value.get("runner_before_tombstone") != {
            "path": os.fspath(R3_RUNNER_PATH),
            "sha256": "dc3321ebe81e48a89159b11109865f0ca0f814f55b68453a8cb6b0e6d2fe6641",
        }
    ):
        raise PermissionError("final-R3 protocol invalidation differs")
    validate_artifact_record(
        protocol, "invalidated final-R3 static protocol",
        schema=R3_PROTOCOL_SCHEMA, canonical_path=R3_PROTOCOL_PATH,
    )
    _, runner_data, _ = stable_bytes(
        R3_RUNNER_PATH, "final-R3 runner tombstone", immutable=False,
    )
    if sha256_bytes(runner_data) != R3_RUNNER_TOMBSTONE_SHA256:
        raise PermissionError("final-R3 runner tombstone SHA differs")
    return {
        "path": os.fspath(path), "sha256": sha256_bytes(data),
        "schema": R3_PROTOCOL_INVALID_SCHEMA, "identity": identity,
        "operational_authority": False,
    }


def _path_fields(value: Any, location: str = "root") -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if isinstance(child, str) and any(
                token in key.lower() for token in ("path", "root", "file")
            ):
                result.append((child_location, child))
            result.extend(_path_fields(child, child_location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(_path_fields(child, f"{location}[{index}]"))
    return result


def _contains_scalar(value: Any, key: str, expected: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get(key) == expected:
            return True
        return any(_contains_scalar(child, key, expected) for child in value.values())
    if isinstance(value, list):
        return any(_contains_scalar(child, key, expected) for child in value)
    return False


def science_contract() -> dict[str, Any]:
    """Return the complete frozen scientific contract used by the sealer."""

    return {
        "candidate_input": {
            "only_source": "canonical_e961_terminal_inputs_r6_exact80_wrapper",
            "scene_count": 80, "fit_scene_count": 60,
            "fold0_scene_count": 20, "legacy_candidate_or_policy_reuse": False,
            "each_fit_scene_detector_excludes_scene": True,
            "old60_detector_receipts_allowed": False,
        },
        "detector_double_oof": {
            "inner_holdout2": {"train_folds": [3, 4], "output_fold": 2},
            "inner_holdout3": {"train_folds": [2, 4], "output_fold": 3},
            "inner_holdout4": {"train_folds": [2, 3], "output_fold": 4},
            "outer_dev": {"train_folds": [2, 3, 4], "output_fold": 0},
        },
        "anchor_scores": {
            "source": gate.ANCHOR_SCORE_SOURCE,
            "all_fold_oof_each_row_model_excludes_scene": True,
            "deploy_scores_allowed": False,
        },
        "gate_double_oof": {
            "fit_folds": [2, 3, 4],
            "gate_holdout2_train_folds": [3, 4],
            "gate_holdout3_train_folds": [2, 4],
            "gate_holdout4_train_folds": [2, 3],
            "threshold_source": "fold234_scene_grouped_gate_oof_only",
            "fold0_used_for_fit_or_selection": False,
        },
        "model": {
            "family": "three_head_low_capacity_linear_v1",
            "heads": [
                "continuous_candidate_iou_huber",
                "strict_candidate_iou_gt_0.50_logistic_calibration",
                "within_anchor_pairwise_same_gt_benefit",
            ],
            "pairwise_group_keys": ["scene_id", "anchor_index"],
            "target_switch_is_harm": True,
            "raw_score_penalty_multiplier": 4.0,
            "feature_names": list(gate.FEATURE_NAMES),
        },
        "selection": {
            "candidate_iou_grid": list(gate.IOU_GRID),
            "same_gt_gain_grid": list(gate.GAIN_GRID),
            "strict_iou50_probability_grid": list(gate.PROB_GRID),
            "max_replacements_per_scene": gate.MAX_REPLACEMENTS_PER_SCENE,
            "oof_safety_gate": dict(gate.OOF_SAFETY_GATE),
            "fold0_retuning": False,
        },
        "metric": {
            "box_geometry": "world_enclosing_aabb_from_8_corners",
            "ranking": "global_score_descending_numpy_default_argsort",
            "matching": "scene_local_duplicate_aware",
            "iou_comparison": "strict_greater_than",
            "thresholds": list(gate.IOU_THRESHOLDS),
            "scores_preserved": True,
        },
        "materialization": {
            "geometry_only": True, "score_values_preserved": True,
            "row_order_preserved": True, "row_count_preserved": True,
            "exploratory_noncanonical": True,
        },
        "isolation": {
            "fold0_role": "reused_dev_continuation_diagnostic_only",
            "formal_fold1_path_or_loader_present": False,
            "formal_official_validation_path_or_loader_present": False,
            "scannet_weight_or_artifact_access": False,
            "policy_activation_authorized": False,
        },
        "runtime_authority": {
            "formal_entry_accepts_context_or_loader": False,
            "claim_accepts_context": False,
            "fresh_canonical_context_rederived_at_every_stage_and_gt_load": True,
            "canonical_gt_scene_box_path_bound_at_decode": True,
            "output_parent_fd_held_from_probe_through_claim_lock_and_mkdirat": True,
            "formal_writers_use_local_dirfd_create_only_primitives": True,
            "single_attempt_claim_before_gt": True,
            "runtime_root_mkdirat_exclusive_after_claim": True,
            "runtime_root_persistent_fd_and_inode_bound": True,
            "all_descendants_opened_relative_to_runtime_root_fd": True,
            "unregistered_existing_directory_reuse": False,
            "partial_failure_restart_or_resume": False,
        },
    }


def _independent_voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    padded_recall = np.concatenate(([0.0], recall, [1.0]))
    padded_precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(padded_precision) - 1, 0, -1):
        padded_precision[index - 1] = max(
            padded_precision[index - 1], padded_precision[index]
        )
    changes = np.flatnonzero(padded_recall[1:] != padded_recall[:-1])
    return float(np.sum(
        (padded_recall[changes + 1] - padded_recall[changes])
        * padded_precision[changes + 1]
    ))


def ap_parity_fixture() -> dict[str, Any]:
    """Cross-check the imported metric with an independent frozen reference."""

    scenes = np.asarray(("00000001", "00000002", "00000001", "00000002", "00000001"))
    scores = np.asarray((0.99, 0.95, 0.90, 0.80, 0.70), np.float64)
    best_iou = np.asarray((0.15, 0.90, 0.80, 0.50, 0.95), np.float64)
    best_gt = np.asarray((0, 0, 0, 1, 1), np.int64)
    positives = 4
    imported = gate.official_ca_ap(
        scene_ids=scenes, scores=scores, best_iou=best_iou,
        best_gt=best_gt, ground_truth_count=positives,
    )
    order = np.argsort(-scores)
    reference: dict[str, dict[str, float | int]] = {}
    for threshold in gate.IOU_THRESHOLDS:
        tp = np.zeros(len(order), np.float64)
        fp = np.zeros(len(order), np.float64)
        detected: set[tuple[str, int]] = set()
        for rank, row in enumerate(order.tolist()):
            key = (str(scenes[row]), int(best_gt[row]))
            if best_iou[row] > threshold and best_gt[row] >= 0 and key not in detected:
                tp[rank] = 1.0
                detected.add(key)
            else:
                fp[rank] = 1.0
        cumulative_tp = np.cumsum(tp)
        cumulative_fp = np.cumsum(fp)
        recall = cumulative_tp / (positives + 1.0e-6)
        precision = cumulative_tp / np.maximum(
            cumulative_tp + cumulative_fp, np.finfo(np.float64).eps
        )
        key = f"iou_{threshold:.2f}"
        final_tp = int(cumulative_tp[-1])
        reference[key] = {
            "ap": _independent_voc_ap(recall, precision),
            "precision": float(precision[-1]), "recall": float(recall[-1]),
            "tp": final_tp, "fp": int(cumulative_fp[-1]),
            "fn": positives - final_tp,
        }
    for key in reference:
        for field in ("ap", "precision", "recall"):
            if not math.isclose(
                float(imported[key][field]), float(reference[key][field]),
                rel_tol=0.0, abs_tol=1.0e-15,
            ):
                raise RuntimeError(f"official AP parity differs: {key}.{field}")
        for field in ("tp", "fp", "fn"):
            if imported[key][field] != reference[key][field]:
                raise RuntimeError(f"official AP parity differs: {key}.{field}")
    fixture = {
        "scene_ids": scenes.tolist(), "scores": scores.tolist(),
        "best_iou": best_iou.tolist(), "best_gt": best_gt.tolist(),
        "ground_truth_count": positives,
    }
    return {
        "schema": "boxfusion.ca1m_tr3d_terminal_gate_ap_parity.v5.final.r4",
        "pass": True, "fixture_sha256": sha256_bytes(canonical_json(fixture)),
        "reference": reference, "imported": imported,
        "protocol": science_contract()["metric"],
    }


@dataclass(frozen=True)
class R6Exact80Binding:
    wrapper_path: Path
    wrapper_sha256: str
    wrapper_identity: Mapping[str, int]
    collection_path: Path
    collection_sha256: str
    collection_identity: Mapping[str, int]
    authorization_commit_id: str
    scene_folds: Mapping[str, int]
    upstream_records: Mapping[str, Mapping[str, Any]]


def _require_under(path: Path, root: Path, name: str) -> None:
    try:
        _absolute_lexical(path).relative_to(_absolute_lexical(root))
    except ValueError:
        raise ValueError(f"{name} is outside canonical R6 root") from None


def _validate_e961_producer_lineage(
    collection: gate.CandidateCollectionV5,
    r6_authorization: Mapping[str, Any],
) -> None:
    auth_roles = {
        str(row.get("role")): row for row in r6_authorization.get("roles", ())
        if isinstance(row, Mapping)
    }
    if tuple(auth_roles) != tuple(gate.ROLE_SPECS):
        # R6 authorizations use outer-first order, but exact membership is the
        # security property.  Do not use order as a hidden compatibility path.
        if set(auth_roles) != set(gate.ROLE_SPECS):
            raise ValueError("R6 authorization producer role set differs")
    for role in gate.ROLE_SPECS:
        role_path = collection.role_manifests[role]
        _require_under(role_path, R6_ROOT / "manifests", f"{role} role manifest")
        _, role_payload, _, _ = stable_json(
            role_path, f"{role} R6 role manifest",
            schema=gate.ROLE_COLLECTION_SCHEMA, immutable=False,
        )
        normalized_record = role_payload.get("role_receipt") or {}
        normalized_path, normalized, _, _ = _record_json(
            normalized_record, f"{role} normalized receipt",
            schema=gate.ROLE_RECEIPT_SCHEMA, immutable=False,
        )
        _require_under(normalized_path, R6_ROOT / "normalized_receipts", f"{role} normalized receipt")
        adapter_path, adapter, _, _ = _record_json(
            normalized.get("source_training_receipt"), f"{role} producer adapter",
            schema="boxfusion.ca1m_tr3d_e961_verified_receipt_adapter.v2",
            immutable=False,
        )
        _require_under(adapter_path, R6_ROOT / "normalized_receipts", f"{role} producer adapter")
        source_record = adapter.get("source_producer_receipt") or {}
        expected_schema = OUTER_RUN_SCHEMA if role == "outer_dev" else INNER_RUN_SCHEMA
        source_path, source_payload, _, _ = _record_json(
            source_record, f"{role} raw E961 receipt", schema=expected_schema,
            immutable=False,
        )
        if "e961" not in os.fspath(source_path).lower():
            raise ValueError(f"{role} producer receipt is not from E961")
        auth_row = auth_roles.get(role) or {}
        if (
            auth_row.get("receipt_path") != os.fspath(source_path)
            or auth_row.get("receipt_sha256") != sha256_file(source_path)
            or source_payload.get("status") != "success"
            or source_payload.get("exit_code") != 0
            or not _contains_scalar(source_payload, "train_scenes", 1001)
            or not _contains_scalar(source_payload, "optimizer_updates", 11268)
            or not _contains_scalar(source_payload, "initialization", "random_scratch_ca_only")
            or (adapter.get("checkpoint") or {}).get("sha256")
            != auth_row.get("checkpoint_sha256")
        ):
            raise ValueError(f"{role} is not the authorized CA-only E961 exact1001 producer")
        for location, raw_path in _path_fields({
            "normalized": normalized, "adapter": adapter,
        }, f"{role}"):
            lowered = raw_path.lower()
            if any(token in lowered for token in _FORBIDDEN_CANDIDATE_PATH_TOKENS):
                raise ValueError(f"{location} references forbidden legacy candidate path")


def load_r6_exact80_binding(
    receipt_path: Path = R6_RECEIPT_PATH,
) -> R6Exact80Binding:
    """Open the canonical R6 commit chain and exact80 collection, GT-free."""

    requested = _absolute_lexical(receipt_path)
    if requested != _absolute_lexical(R6_RECEIPT_PATH):
        raise ValueError("terminal gate accepts only the canonical R6 exact80 receipt")
    wrapper_path, wrapper, wrapper_bytes, wrapper_identity = stable_json(
        requested, "R6 exact80 wrapper", schema=R6_RECEIPT_SCHEMA,
        immutable=False,
    )
    expected_keys = {
        "schema", "complete", "create_only", "namespace",
        "fit_scene_count", "fit_folds", "reused_dev_scene_count",
        "reused_dev_folds", "scene_count", "each_scene_detector_excludes_scene",
        "b6_score_source", "ground_truth_access", "fold1_access",
        "official_validation_access", "legacy_v1_v4_candidate_or_policy_reused",
        "r6_preregistration", "r6_ready_config", "r6_run_authorization",
        "r6_authorization_bundle", "authorization_commit_id",
        "candidate_collection", "r2_execution_receipt",
    }
    if set(wrapper) != expected_keys:
        raise ValueError("R6 exact80 wrapper key set differs")
    if (
        wrapper.get("complete") is not True
        or wrapper.get("create_only") is not True
        or wrapper.get("namespace") != R6_NAMESPACE
        or wrapper.get("fit_scene_count") != 60
        or wrapper.get("fit_folds") != [2, 3, 4]
        or wrapper.get("reused_dev_scene_count") != 20
        or wrapper.get("reused_dev_folds") != [0]
        or wrapper.get("scene_count") != 80
        or wrapper.get("each_scene_detector_excludes_scene") is not True
        or wrapper.get("b6_score_source")
        != "all_fold_oof_each_row_model_excludes_scene"
        or wrapper.get("ground_truth_access") is not False
        or wrapper.get("fold1_access") is not False
        or wrapper.get("official_validation_access") is not False
        or wrapper.get("legacy_v1_v4_candidate_or_policy_reused") is not False
    ):
        raise ValueError("R6 exact80 wrapper science/isolation contract differs")
    commit_id = _sha(wrapper.get("authorization_commit_id"), "R6 commit id")
    r2_record = wrapper.get("r2_execution_receipt") or {}
    if r2_record.get("operational_authority") is not False:
        raise ValueError("R2 execution receipt must be explicitly non-authoritative")
    r2_path, _, r2_data, r2_identity = _record_json(
        {key: r2_record.get(key) for key in ("path", "sha256", "schema")},
        "R2 internal execution receipt",
        schema=R6_R2_EXECUTION_RECEIPT_SCHEMA,
        canonical_path=R6_R2_EXECUTION_RECEIPT_PATH,
        immutable=False,
    )
    _require_under(r2_path, R6_ROOT / "manifests", "R2 internal execution receipt")
    upstream_specs = {
        "r6_preregistration": (R6_PREREGISTRATION_SCHEMA, R6_PREREGISTRATION_PATH),
        "r6_ready_config": (R6_CONFIG_SCHEMA, R6_READY_CONFIG_PATH),
        "r6_run_authorization": (R6_AUTHORIZATION_SCHEMA, R6_RUN_AUTHORIZATION_PATH),
        "r6_authorization_bundle": (R6_BUNDLE_SCHEMA, R6_AUTHORIZATION_BUNDLE_PATH),
    }
    upstream: dict[str, dict[str, Any]] = {}
    for key, (schema, canonical_path) in upstream_specs.items():
        path, payload, data, identity = _record_json(
            wrapper[key], key, schema=schema, canonical_path=canonical_path
        )
        upstream[key] = {
            "path": os.fspath(path), "sha256": sha256_bytes(data),
            "schema": schema, "identity": identity, "payload": payload,
        }
    prereg = upstream["r6_preregistration"]["payload"]
    ready = upstream["r6_ready_config"]["payload"]
    authorization = upstream["r6_run_authorization"]["payload"]
    bundle = upstream["r6_authorization_bundle"]["payload"]
    if upstream["r6_preregistration"]["sha256"] != R6_PREREGISTRATION_SHA256:
        raise PermissionError("R6 preregistration does not match the frozen CODE-PASS SHA")
    _, r6_config_data, _ = validate_artifact_record(
        prereg.get("pending_config"), "R6 frozen pending config",
        schema=R6_CONFIG_SCHEMA, canonical_path=R6_CONFIG_PATH,
        immutable=False,
    )
    current_core = (prereg.get("implementation") or {}).get("current_core")
    _, r6_core_data, _ = validate_artifact_record(
        current_core, "R6 frozen execution core",
        canonical_path=R6_CORE_PATH, immutable=False,
    )
    if (
        sha256_bytes(r6_config_data) != R6_CONFIG_SHA256
        or sha256_bytes(r6_core_data) != R6_CORE_SHA256
    ):
        raise PermissionError("R6 frozen config/core SHA differs")
    if (
        prereg.get("complete") is not True
        or prereg.get("create_only") is not True
        or prereg.get("static_only") is not True
        or prereg.get("namespace") != R6_NAMESPACE
        or ready.get("namespace") != R6_NAMESPACE
        or ready.get("preregistration") != {
            "path": os.fspath(R6_PREREGISTRATION_PATH),
            "schema": R6_PREREGISTRATION_SCHEMA,
        }
        or ((ready.get("implementation") or {}).get("current_core"))
        != current_core
        or ((ready.get("outputs") or {}).get("namespace_root"))
        != os.fspath(R6_ROOT)
        or authorization.get("complete") is not True
        or authorization.get("create_only") is not True
        or authorization.get("namespace") != R6_NAMESPACE
        or authorization.get("commit_id") != commit_id
        or authorization.get("pending_config_sha256") != R6_CONFIG_SHA256
        or authorization.get("preregistration") != {
            "path": os.fspath(R6_PREREGISTRATION_PATH),
            "sha256": R6_PREREGISTRATION_SHA256,
        }
        or authorization.get("ground_truth_access") is not False
        or authorization.get("fold1_access") is not False
        or authorization.get("official_validation_access") is not False
        or authorization.get("formal_gpu_run_started") is not False
        or bundle.get("complete") is not True
        or bundle.get("create_only") is not True
        or bundle.get("namespace") != R6_NAMESPACE
        or bundle.get("commit_id") != commit_id
        or bundle.get("commit_role") != "last_published_unique_operational_gate"
        or bundle.get("ready_config") != {
            "path": os.fspath(R6_READY_CONFIG_PATH),
            "sha256": upstream["r6_ready_config"]["sha256"],
            "schema": R6_CONFIG_SCHEMA,
        }
        or bundle.get("run_authorization") != {
            "path": os.fspath(R6_RUN_AUTHORIZATION_PATH),
            "sha256": upstream["r6_run_authorization"]["sha256"],
            "schema": R6_AUTHORIZATION_SCHEMA,
        }
        or bundle.get("ground_truth_access") is not False
        or bundle.get("gpu_started") is not False
        or (ready.get("run_authorization") or {}).get("state")
        != "committed_by_bundle"
        or (ready.get("run_authorization") or {}).get("commit_id") != commit_id
        or (ready.get("run_authorization") or {}).get("path")
        != os.fspath(R6_AUTHORIZATION_BUNDLE_PATH)
    ):
        raise ValueError("R6 ready/auth/bundle commit chain differs")
    collection_path, collection_data, collection_identity = validate_artifact_record(
        wrapper.get("candidate_collection"), "R6 candidate collection",
        schema=gate.COLLECTION_SCHEMA, canonical_path=R6_COLLECTION_PATH,
        immutable=False,
    )
    with _fuse_gate_io():
        collection = gate.load_candidate_collection_v5(collection_path)
    if collection.payload.get("namespace") != gate.NAMESPACE:
        raise ValueError("R6 collection generic-v5 namespace differs")
    for row in collection.scenes.values():
        evidence_path = Path(str(row.get("path", "")))
        _require_under(evidence_path, R6_ROOT / "evidence", "R6 candidate evidence")
        if int(row["fold_id"]) in tuple(row["producer_train_folds"]):
            raise ValueError("R6 detector candidate is in-sample")
    _validate_e961_producer_lineage(collection, authorization)
    return R6Exact80Binding(
        wrapper_path=wrapper_path, wrapper_sha256=sha256_bytes(wrapper_bytes),
        wrapper_identity=_deep_immutable(wrapper_identity), collection_path=collection_path,
        collection_sha256=sha256_bytes(collection_data),
        collection_identity=_deep_immutable(collection_identity),
        authorization_commit_id=commit_id,
        scene_folds=_deep_immutable({
            scene: int(row["fold_id"]) for scene, row in collection.scenes.items()
        }),
        upstream_records=_deep_immutable({
            **{
                key: {field: value for field, value in record.items() if field != "payload"}
                for key, record in upstream.items()
            },
            "r2_internal_execution_receipt": {
                "path": os.fspath(r2_path), "sha256": sha256_bytes(r2_data),
                "schema": R6_R2_EXECUTION_RECEIPT_SCHEMA,
                "identity": r2_identity, "operational_authority": False,
            },
        }),
    )


def validate_pending_config(
    path: Path = DEFAULT_PENDING_CONFIG,
) -> tuple[Path, dict[str, Any]]:
    source, cfg, _, _ = stable_json(path, "terminal gate final pending config", immutable=False)
    if set(cfg) != {
        "schema", "namespace", "state", "authorizations", "access",
        "future_r6_input", "annotation_inventory", "science_contract",
        "invalidated_predecessor", "implementation", "outputs", "formal_artifacts",
    }:
        raise ValueError("terminal gate final pending config keys differ")
    if (
        cfg.get("schema") != PENDING_SCHEMA
        or cfg.get("namespace") != NAMESPACE
        or cfg.get("state") != "pending_r6_exact80"
        or cfg.get("authorizations") != {
            "scientific_preregistration": False, "ground_truth_join": False,
            "gate_fit": False, "threshold_selection": False,
            "fold0_reused_dev_diagnostic": False,
            "geometry_materialization": False, "fold1": False,
            "official_validation": False, "policy_activation": False,
        }
        or cfg.get("access") != {
            "static_only": True, "candidate_opened": False,
            "ground_truth_access": False, "output_created": False,
            "gpu_started": False, "fold1_path_present": False,
            "official_validation_path_present": False,
        }
        or cfg.get("science_contract") != science_contract()
    ):
        raise ValueError("terminal gate final pending fail-close contract differs")
    expected_r6 = {
        "state": "pending", "path": os.fspath(R6_RECEIPT_PATH),
        "sha256": None, "schema": R6_RECEIPT_SCHEMA,
        "only_candidate_input": True,
    }
    if cfg.get("future_r6_input") != expected_r6:
        raise ValueError("pending R6 exact80 input differs")
    if cfg.get("invalidated_predecessor") != _predecessor_invalidation_pending_record():
        raise ValueError("pending final-R3 invalidation binding differs")
    validate_r3_protocol_invalidation()
    if cfg.get("annotation_inventory") != {
        "state": "frozen_canonical", "path": os.fspath(GT_INVENTORY_PATH),
        "sha256": GT_INVENTORY_SHA256,
        "schema": GT_INVENTORY_SCHEMA, "ca_train_annotations_only": True,
        "old_gate_dataset_or_policy_reuse": False,
    }:
        raise ValueError("pending annotation inventory differs")
    if cfg.get("implementation") != {
        key: os.fspath(value) for key, value in IMPLEMENTATION_PATHS.items()
    }:
        raise ValueError("pending final-gate implementation inventory differs")
    if cfg.get("outputs") != {key: os.fspath(value) for key, value in OUTPUT_PATHS.items()}:
        raise ValueError("pending final-gate output paths differ")
    if cfg.get("formal_artifacts") != {
        "static_protocol": os.fspath(PROTOCOL_PATH),
        "preregistration": os.fspath(PREREGISTRATION_PATH),
        "ready_config": os.fspath(READY_CONFIG_PATH),
        "run_authorization": os.fspath(RUN_AUTHORIZATION_PATH),
    }:
        raise ValueError("pending formal artifact paths differ")
    return source, cfg


def static_preflight(
    path: Path = DEFAULT_PENDING_CONFIG,
) -> dict[str, Any]:
    source, _ = validate_pending_config(path)
    return {
        "schema": "boxfusion.ca1m_tr3d_terminal_gate_static_preflight.v5.final.r4",
        "status": "PASS_STATIC_PENDING_R6", "config": os.fspath(source),
        "config_sha256": sha256_file(source), "runtime_ready": False,
        "r6_candidate_opened": False, "ground_truth_access": False,
        "legacy_final_invalidated": True,
        "output_created": False, "directory_created": False,
        "gpu_started": False, "fold1_access": False,
        "official_validation_access": False,
        "failure_action": "stop_before_gt_mkdir_or_output_until_r6_exact80_is_committed",
    }


def operational_preflight_pending(
    path: Path = DEFAULT_PENDING_CONFIG,
) -> None:
    """Fail before resolving R6 children, GT, output parents, or CUDA."""

    validate_pending_config(path)
    if not R6_RECEIPT_PATH.exists() and not R6_RECEIPT_PATH.is_symlink():
        raise PendingR6Inputs(
            "canonical R6 exact80 receipt is absent; no GT, output directory, "
            "formal preregistration, trainer, materializer, fold1, validation, or GPU was reached"
        )
    # Once the canonical commit exists, a caller may explicitly proceed to the
    # scientific sealer.  This function itself still creates no artifact.
    load_r6_exact80_binding(R6_RECEIPT_PATH)


def _publish_json_replay_safe(
    path: Path, payload: Mapping[str, Any], name: str,
) -> Path:
    """Publish one fixed formal JSON through local dirfd-only primitives."""

    data = canonical_json(payload)
    target = _canonical_absolute(_absolute_lexical(path), name)
    allowed = {
        PROTOCOL_PATH, PREREGISTRATION_PATH, READY_CONFIG_PATH,
        RUN_AUTHORIZATION_PATH,
    }
    if target not in {_absolute_lexical(value) for value in allowed}:
        raise PermissionError(f"{name} target is not a fixed R4 formal artifact")
    manifests_fd, _ = _open_dir_chain(
        MANIFEST_ROOT.parent, f"{name} manifest parent",
    )
    formal_fd = -1
    try:
        try:
            formal_fd = os.open(MANIFEST_ROOT.name, _DIR_FLAGS, dir_fd=manifests_fd)
        except FileNotFoundError:
            try:
                os.mkdir(MANIFEST_ROOT.name, 0o755, dir_fd=manifests_fd)
            except FileExistsError:
                # A racing creator is acceptable only if the resulting entry
                # opens as the fixed no-follow directory below the held parent.
                pass
            os.fsync(manifests_fd)
            formal_fd = os.open(MANIFEST_ROOT.name, _DIR_FLAGS, dir_fd=manifests_fd)
        try:
            current, identity = _read_bytes_at_fd(formal_fd, target.name, name)
        except FileNotFoundError:
            identity = _exclusive_bytes_at_fd(formal_fd, target.name, data, name)
            current, observed_identity = _read_bytes_at_fd(
                formal_fd, target.name, name,
            )
            if observed_identity != identity:
                raise RuntimeError(f"published {name} identity differs")
        if current != data:
            raise FileExistsError(f"refusing differing existing {name}: {target}")
    finally:
        if formal_fd >= 0:
            os.close(formal_fd)
        os.close(manifests_fd)
    source, observed, observed_identity = stable_bytes(
        target, name, immutable=False,
    )
    if observed != data or observed_identity != identity:
        raise RuntimeError(f"canonical {name} differs after local publication")
    return source


@dataclass(frozen=True)
class GTInventoryBinding:
    path: Path
    sha256: str
    identity: Mapping[str, int]
    scene_rows: Mapping[str, Mapping[str, Any]]


def validate_gt_inventory_metadata(
    path: Path, *, r6_binding: R6Exact80Binding,
) -> GTInventoryBinding:
    """Hash opaque GT bytes and lineage metadata without decoding an array."""

    if _absolute_lexical(path) != _absolute_lexical(GT_INVENTORY_PATH):
        raise ValueError("only the canonical frozen CA-train GT inventory is accepted")
    source, value, data, identity = stable_json(
        path, "CA-train annotation inventory", schema=GT_INVENTORY_SCHEMA
    )
    if sha256_bytes(data) != GT_INVENTORY_SHA256:
        raise PermissionError("canonical CA-train GT inventory SHA256 differs")
    rows = value.get("scenes")
    if not isinstance(rows, Mapping):
        raise ValueError("annotation inventory scenes must be an object")
    normalized = {str(scene): row for scene, row in rows.items() if isinstance(row, Mapping)}
    fold_counts = {
        fold: sum(int(row.get("fold_id", -1)) == fold for row in normalized.values())
        for fold in (0, 2, 3, 4)
    }
    if (
        value.get("complete") is not True
        or value.get("create_only") is not True
        or set(value) != {
            "schema", "complete", "create_only", "file_count", "fit_fold_ids",
            "fit_scene_count", "gt_array_content_loaded", "inventory_sha256",
            "locked_internal_fold_ids", "locked_internal_scene_count_accessed",
            "official_validation_comparable", "oof_sidecar",
            "opaque_source_bytes_hashed_and_copied", "output_root", "scene_count",
            "scenes", "shadow_files_read_only", "source_bytes_mutated",
            "source_dataset_manifest", "source_root", "threshold_dev_fold_ids",
            "threshold_dev_scene_count", "train_only",
            "validation_ground_truth_access", "validation_prediction_access",
        }
        or value.get("file_count") != 160
        or value.get("scene_count") != 80
        or value.get("fit_scene_count") != 60
        or value.get("threshold_dev_scene_count") != 20
        or value.get("fit_fold_ids") != [2, 3, 4]
        or value.get("threshold_dev_fold_ids") != [0]
        or value.get("locked_internal_fold_ids") != [1]
        or value.get("locked_internal_scene_count_accessed") != 0
        or value.get("official_validation_comparable") is not False
        or value.get("validation_ground_truth_access") is not False
        or value.get("validation_prediction_access") is not False
        or value.get("train_only") is not True
        or value.get("gt_array_content_loaded") is not False
        or value.get("inventory_sha256") != GT_INVENTORY_CONTENT_SHA256
        or value.get("opaque_source_bytes_hashed_and_copied") is not True
        or value.get("shadow_files_read_only") is not True
        or value.get("source_bytes_mutated") is not False
        or value.get("output_root") != os.fspath(GT_SHADOW_ROOT)
        or value.get("source_root") != os.fspath(GT_SOURCE_ROOT)
        or value.get("source_dataset_manifest") != {
            "path": os.fspath(GT_SOURCE_DATASET_MANIFEST_PATH),
            "sha256": GT_SOURCE_DATASET_MANIFEST_SHA256,
        }
        or value.get("oof_sidecar") != {
            "path": os.fspath(GT_OOF_SIDECAR_PATH),
            "sha256": GT_OOF_SIDECAR_SHA256,
        }
        or set(normalized) != set(r6_binding.scene_folds)
        or fold_counts != {0: 20, 2: 20, 3: 20, 4: 20}
    ):
        raise ValueError("CA-train annotation inventory partition/isolation differs")
    validate_artifact_record(
        value["source_dataset_manifest"], "GT source dataset manifest",
        canonical_path=GT_SOURCE_DATASET_MANIFEST_PATH,
    )
    validate_artifact_record(
        value["oof_sidecar"], "GT all-fold OOF sidecar",
        canonical_path=GT_OOF_SIDECAR_PATH,
    )
    for scene, expected_fold in r6_binding.scene_folds.items():
        row = normalized[scene]
        box = row.get("box") or {}
        manifest = row.get("manifest") or {}
        box_path = GT_SHADOW_ROOT / scene / "derived_train_gt_boxes.npy"
        manifest_path = GT_SHADOW_ROOT / scene / "derived_train_gt_manifest.json"
        source_box_path = GT_SOURCE_ROOT / scene / "derived_train_gt_boxes.npy"
        source_manifest_path = GT_SOURCE_ROOT / scene / "derived_train_gt_manifest.json"
        if (
            _SCENE.fullmatch(scene) is None
            or set(row) != {"fold_id", "box", "manifest"}
            or row.get("fold_id") != expected_fold
            or not isinstance(box, Mapping)
            or not isinstance(manifest, Mapping)
            or set(box) != {"mode", "path", "sha256", "source_mode", "source_path"}
            or set(manifest) != {"mode", "path", "sha256", "source_mode", "source_path"}
            or box.get("path") != os.fspath(box_path)
            or manifest.get("path") != os.fspath(manifest_path)
            or box.get("source_path") != os.fspath(source_box_path)
            or manifest.get("source_path") != os.fspath(source_manifest_path)
            or box.get("mode") != "0o444" or manifest.get("mode") != "0o444"
            or box.get("source_mode") != "0o777"
            or manifest.get("source_mode") != "0o777"
            or _SHA.fullmatch(str(box.get("sha256", ""))) is None
            or _SHA.fullmatch(str(manifest.get("sha256", ""))) is None
        ):
            raise ValueError(f"{scene}: annotation inventory row differs")
        validate_artifact_record(
            box, f"CA-train GT shadow boxes {scene}", canonical_path=box_path,
        )
        _, manifest_data, _ = validate_artifact_record(
            manifest, f"CA-train GT shadow manifest {scene}",
            canonical_path=manifest_path,
        )
        try:
            manifest_value = json.loads(manifest_data)
        except json.JSONDecodeError as error:
            raise ValueError(f"{scene}: GT shadow manifest is not JSON") from error
        if (
            not isinstance(manifest_value, dict)
            or manifest_value.get("schema")
            != "boxfusion.ca1m_native_b6_train_scene.v1"
        ):
            raise ValueError(f"{scene}: GT shadow manifest schema differs")
        after_filter = (manifest_value.get("artifacts") or {}).get(
            "after_filter_boxes.npy"
        ) or {}
        storage = manifest_value.get("storage_filesystem_policy") or {}
        if (
            manifest_value.get("scene_id") != scene
            or manifest_value.get("source_split") != "train"
            or manifest_value.get("train_only") is not True
            or manifest_value.get("validation_ground_truth_access") is not False
            or manifest_value.get("validation_scene_overlap") is not False
            or manifest_value.get("official_validation_comparable") is not False
            or manifest_value.get("paper_validation_claim_permitted") is not False
            or manifest_value.get("derived_train_gt") is not True
            or manifest_value.get("derived_train_gt_artifact")
            != "derived_train_gt_boxes.npy"
            or manifest_value.get("derived_train_gt_sha256") != box.get("sha256")
            or manifest_value.get("compat_after_filter_sha256") != box.get("sha256")
            or after_filter.get("sha256") != box.get("sha256")
            or manifest_value.get("output_scene")
            != os.fspath(GT_SOURCE_ROOT / scene)
            or storage.get("filesystem_type") != "fuseblk"
            or storage.get("posix_mode_enforceable") is not False
            or storage.get("artifact_integrity_contract")
            != "regular_no_symlink_sha256_create_only"
        ):
            raise ValueError(f"{scene}: GT shadow manifest lineage differs")
    return GTInventoryBinding(
        source, sha256_bytes(data), _deep_immutable(identity),
        _deep_immutable(normalized),
    )


def _implementation_records() -> dict[str, Any]:
    return {
        key: artifact_record(path, f"final-gate implementation {key}", immutable=False)
        for key, path in IMPLEMENTATION_PATHS.items()
    }


def _r6_binding_record(binding: R6Exact80Binding) -> dict[str, Any]:
    return {
        "wrapper": {
            "path": os.fspath(binding.wrapper_path),
            "sha256": binding.wrapper_sha256,
            "schema": R6_RECEIPT_SCHEMA,
            "identity": dict(binding.wrapper_identity),
        },
        "candidate_collection": {
            "path": os.fspath(binding.collection_path),
            "sha256": binding.collection_sha256,
            "schema": gate.COLLECTION_SCHEMA,
            "identity": dict(binding.collection_identity),
        },
        "authorization_commit_id": binding.authorization_commit_id,
        "upstream_commit_chain": dict(binding.upstream_records),
    }


def build_static_protocol_payload(
    pending_config_path: Path = DEFAULT_PENDING_CONFIG,
) -> dict[str, Any]:
    """Freeze science/code before the future R6 candidate commit exists."""

    pending_path, _ = validate_pending_config(pending_config_path)
    predecessor_invalidation = validate_r3_protocol_invalidation()
    _, inventory, inventory_data, inventory_identity = stable_json(
        GT_INVENTORY_PATH, "canonical GT inventory receipt",
        schema=GT_INVENTORY_SCHEMA,
    )
    if (
        sha256_bytes(inventory_data) != GT_INVENTORY_SHA256
        or inventory.get("scene_count") != 80
        or inventory.get("train_only") is not True
        or inventory.get("validation_ground_truth_access") is not False
    ):
        raise PermissionError("canonical GT inventory receipt differs")
    r6_prereg = artifact_record(
        R6_PREREGISTRATION_PATH, "frozen R6 static preregistration",
        schema=R6_PREREGISTRATION_SCHEMA,
    )
    if r6_prereg["sha256"] != R6_PREREGISTRATION_SHA256:
        raise PermissionError("frozen R6 static preregistration SHA differs")
    implementation = _implementation_records()
    if implementation["generic_gate_core"]["sha256"] != GENERIC_GATE_SHA256:
        raise PermissionError("frozen generic terminal-gate core SHA differs")
    return {
        "schema": PROTOCOL_SCHEMA, "complete": True, "create_only": True,
        "static_science_protocol": True, "operational_authority": False,
        "namespace": NAMESPACE,
        "sealed_before_r6_exact80_wrapper_exists": True,
        "pending_config": artifact_record(
            pending_path, "final-gate pending config", immutable=False,
        ),
        "invalidated_predecessor": predecessor_invalidation,
        "future_r6": {
            "static_preregistration": r6_prereg,
            "expected_wrapper": {
                "path": os.fspath(R6_RECEIPT_PATH),
                "schema": R6_RECEIPT_SCHEMA,
                "namespace": R6_NAMESPACE,
                "scene_count": 80, "fit_folds": [2, 3, 4],
                "reused_dev_folds": [0], "only_candidate_input": True,
            },
        },
        "annotation_inventory_receipt": {
            "path": os.fspath(GT_INVENTORY_PATH),
            "sha256": GT_INVENTORY_SHA256, "schema": GT_INVENTORY_SCHEMA,
            "identity": inventory_identity,
            "scene_box_or_manifest_opened": False,
        },
        "science_contract": science_contract(),
        "official_ap_parity": ap_parity_fixture(),
        "implementation": implementation,
        "outputs": {key: os.fspath(value) for key, value in OUTPUT_PATHS.items()},
        "access_at_seal": {
            "r6_wrapper": False, "candidate_collection": False,
            "candidate_evidence": False, "gt_arrays": False,
            "gt_scene_manifests": False, "runtime_namespace": False,
            "gpu": False, "fold1": False, "official_validation": False,
        },
    }


def seal_static_protocol(
    *, output_path: Path | None = None,
    pending_config_path: Path = DEFAULT_PENDING_CONFIG,
) -> Path:
    if output_path is None:
        output_path = PROTOCOL_PATH
    if _absolute_lexical(output_path) != _absolute_lexical(PROTOCOL_PATH):
        raise ValueError("final-gate static protocol target is noncanonical")
    if R6_RECEIPT_PATH.exists() or R6_RECEIPT_PATH.is_symlink():
        raise PermissionError("static protocol must be sealed before the R6 wrapper exists")
    payload = build_static_protocol_payload(pending_config_path)
    if any(path.exists() or path.is_symlink() for path in (
        PREREGISTRATION_PATH, READY_CONFIG_PATH, RUN_AUTHORIZATION_PATH,
    )):
        raise FileExistsError("cannot seal static protocol after instance artifacts")
    output = _publish_json_replay_safe(
        output_path, payload, "final-gate static science protocol",
    )
    expected = build_static_protocol_payload(pending_config_path)
    if stable_bytes(output, "published static science protocol")[1] != canonical_json(expected):
        raise RuntimeError("static science protocol inputs drifted during publication")
    return output


def validate_static_protocol(
    path: Path | None = None,
) -> tuple[Path, dict[str, Any], bytes, dict[str, int]]:
    if path is None:
        path = PROTOCOL_PATH
    if _absolute_lexical(path) != _absolute_lexical(PROTOCOL_PATH):
        raise ValueError("final-gate static protocol path is noncanonical")
    source, value, data, identity = stable_json(
        path, "final-gate static science protocol", schema=PROTOCOL_SCHEMA,
    )
    pending_path = Path(str((value.get("pending_config") or {}).get("path", "")))
    expected = build_static_protocol_payload(pending_path)
    if (
        data != canonical_json(expected)
        or value.get("operational_authority") is not False
        or (value.get("access_at_seal") or {}).get("gt_arrays") is not False
        or (value.get("access_at_seal") or {}).get("candidate_collection") is not False
    ):
        raise PermissionError("static science protocol/current code or science differs")
    return source, value, data, identity


def build_preregistration_payload(
    *, gt_inventory_path: Path = GT_INVENTORY_PATH,
    pending_config_path: Path = DEFAULT_PENDING_CONFIG,
    r6_receipt_path: Path = R6_RECEIPT_PATH,
) -> dict[str, Any]:
    """Build the formal payload after R6, still before the first GT open."""

    pending_path, _ = validate_pending_config(pending_config_path)
    protocol_path, protocol, protocol_data, protocol_identity = validate_static_protocol()
    predecessor_invalidation = validate_r3_protocol_invalidation()
    r6 = load_r6_exact80_binding(r6_receipt_path)
    inventory = validate_gt_inventory_metadata(gt_inventory_path, r6_binding=r6)
    parity = ap_parity_fixture()
    return {
        "schema": PREREGISTRATION_SCHEMA,
        "complete": True, "create_only": True, "static_science_only": True,
        "namespace": NAMESPACE,
        "sealed_after_r6_exact80_commit": True,
        "sealed_before_first_gt_array_decode": True,
        "opaque_gt_bytes_hashed_at_seal": True,
        "gt_arrays_decoded_at_seal": False,
        "fold0_gt_access_at_seal": False,
        "fold1_or_validation_access_at_seal": False,
        "gpu_started_at_seal": False,
        "pending_config": artifact_record(
            pending_path, "final-gate pending config", immutable=False
        ),
        "static_protocol": {
            "path": os.fspath(protocol_path), "sha256": sha256_bytes(protocol_data),
            "schema": PROTOCOL_SCHEMA, "identity": protocol_identity,
            "science_contract_sha256": sha256_bytes(
                canonical_json(protocol["science_contract"])
            ),
            "implementation_sha256": sha256_bytes(
                canonical_json(protocol["implementation"])
            ),
        },
        "invalidated_predecessor": predecessor_invalidation,
        "r6_exact80": _r6_binding_record(r6),
        "annotation_inventory": {
            "path": os.fspath(inventory.path), "sha256": inventory.sha256,
            "schema": GT_INVENTORY_SCHEMA, "identity": dict(inventory.identity),
            "opaque_box_bytes_hashed_at_seal": True,
            "scene_manifests_parsed_at_seal": True,
            "gt_arrays_decoded_at_seal": False,
            "only_scene_box_records_consumable": True,
            "old_gate_dataset_features_or_policy_consumable": False,
        },
        "science_contract": science_contract(),
        "official_ap_parity": parity,
        "implementation": _implementation_records(),
        "outputs": {key: os.fspath(value) for key, value in OUTPUT_PATHS.items()},
        "failure_actions": {
            "r6_missing_or_changed": "stop_before_gt_mkdir_output_or_gpu",
            "fold234_oof_gate_fail": "publish_stop_without_fold0_gt_or_materialization",
            "partial_output": "fail_closed_no_overwrite_or_resume",
        },
    }


def seal_scientific_preregistration(
    *, gt_inventory_path: Path = GT_INVENTORY_PATH,
    output_path: Path = PREREGISTRATION_PATH,
    pending_config_path: Path = DEFAULT_PENDING_CONFIG,
    r6_receipt_path: Path = R6_RECEIPT_PATH,
) -> Path:
    if _absolute_lexical(output_path) != _absolute_lexical(PREREGISTRATION_PATH):
        raise ValueError("final gate preregistration target is noncanonical")
    # All future inputs and science are validated before the writer is allowed
    # to create the manifest directory.
    payload = build_preregistration_payload(
        gt_inventory_path=gt_inventory_path,
        pending_config_path=pending_config_path,
        r6_receipt_path=r6_receipt_path,
    )
    if READY_CONFIG_PATH.exists() or RUN_AUTHORIZATION_PATH.exists():
        raise FileExistsError("cannot seal preregistration after ready/authorization output")
    output = _publish_json_replay_safe(output_path, payload, "final-gate scientific preregistration")
    # Recompute after publication so code or R6 drift during the write cannot
    # silently enter a formally sealed chain.
    expected = build_preregistration_payload(
        gt_inventory_path=gt_inventory_path,
        pending_config_path=pending_config_path,
        r6_receipt_path=r6_receipt_path,
    )
    if stable_bytes(output, "published final-gate preregistration")[1] != canonical_json(expected):
        raise RuntimeError("final-gate preregistration inputs drifted during publication")
    return output


def validate_scientific_preregistration(
    path: Path = PREREGISTRATION_PATH,
) -> tuple[Path, dict[str, Any], R6Exact80Binding, GTInventoryBinding]:
    if _absolute_lexical(path) != _absolute_lexical(PREREGISTRATION_PATH):
        raise ValueError("final gate preregistration path is noncanonical")
    source, value, data, _ = stable_json(
        path, "final-gate scientific preregistration", schema=PREREGISTRATION_SCHEMA
    )
    inventory_record = value.get("annotation_inventory") or {}
    expected = build_preregistration_payload(
        gt_inventory_path=Path(str(inventory_record.get("path", ""))),
        pending_config_path=Path(str((value.get("pending_config") or {}).get("path", ""))),
        r6_receipt_path=Path(str(((value.get("r6_exact80") or {}).get("wrapper") or {}).get("path", ""))),
    )
    if data != canonical_json(expected):
        raise ValueError("final-gate preregistration/current inputs or science differ")
    r6 = load_r6_exact80_binding(Path(expected["r6_exact80"]["wrapper"]["path"]))
    inventory = validate_gt_inventory_metadata(
        Path(expected["annotation_inventory"]["path"]), r6_binding=r6
    )
    return source, value, r6, inventory


def _ready_payload(prereg_path: Path, prereg: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": READY_SCHEMA, "complete": True, "create_only": True,
        "namespace": NAMESPACE, "state": "ready_r6_exact80_bound",
        "preregistration": artifact_record(
            prereg_path, "final-gate preregistration", schema=PREREGISTRATION_SCHEMA
        ),
        "r6_exact80": prereg["r6_exact80"],
        "annotation_inventory": prereg["annotation_inventory"],
        "science_contract_sha256": sha256_bytes(canonical_json(prereg["science_contract"])),
        "implementation": prereg["implementation"],
        "outputs": prereg["outputs"],
        "authorizations": {
            "ground_truth_join_fold234": True, "gate_fit_fold234": True,
            "threshold_selection_fold234_oof": True,
            "fold0_reused_dev_after_oof_pass": True,
            "geometry_materialization_after_oof_pass": True,
            "fold1": False, "official_validation": False,
            "policy_activation": False,
        },
        "access_before_run": {
            "opaque_gt_bytes_hashed_at_instance_seal": True,
            "gt_arrays_decoded": False, "output_directory_created": False,
            "gpu_started": False, "fold1": False, "official_validation": False,
        },
    }


def seal_ready_authorization(
    *, preregistration_path: Path = PREREGISTRATION_PATH,
    ready_path: Path = READY_CONFIG_PATH,
    authorization_path: Path = RUN_AUTHORIZATION_PATH,
) -> tuple[Path, Path]:
    if (
        _absolute_lexical(ready_path) != _absolute_lexical(READY_CONFIG_PATH)
        or _absolute_lexical(authorization_path) != _absolute_lexical(RUN_AUTHORIZATION_PATH)
    ):
        raise ValueError("final-gate ready/authorization target is noncanonical")
    prereg_path, prereg, _, _ = validate_scientific_preregistration(preregistration_path)
    ready = _ready_payload(prereg_path, prereg)
    ready_bytes = canonical_json(ready)
    commit_material = {
        "schema": "boxfusion.ca1m_tr3d_terminal_gate_commit_material.v5.final.r4",
        "preregistration_sha256": sha256_file(prereg_path),
        "ready_config_sha256": sha256_bytes(ready_bytes),
        "r6_wrapper_sha256": prereg["r6_exact80"]["wrapper"]["sha256"],
        "r6_authorization_commit_id": prereg["r6_exact80"]["authorization_commit_id"],
        "annotation_inventory_sha256": prereg["annotation_inventory"]["sha256"],
    }
    commit_id = sha256_bytes(canonical_json(commit_material))
    authorization = {
        "schema": AUTHORIZATION_SCHEMA, "complete": True, "create_only": True,
        "namespace": NAMESPACE, "commit_id": commit_id,
        "commit_role": "last_published_unique_run_gate",
        "preregistration": artifact_record(
            prereg_path, "final-gate preregistration", schema=PREREGISTRATION_SCHEMA
        ),
        "ready_config": {
            "path": os.fspath(_absolute_lexical(ready_path)),
            "sha256": sha256_bytes(ready_bytes), "schema": READY_SCHEMA,
        },
        "r6_wrapper_sha256": prereg["r6_exact80"]["wrapper"]["sha256"],
        "r6_authorization_commit_id": prereg["r6_exact80"]["authorization_commit_id"],
        "annotation_inventory_sha256": prereg["annotation_inventory"]["sha256"],
        "authorizations": ready["authorizations"],
        "ground_truth_access_at_seal": False,
        "output_runtime_root_created_at_seal": False,
        "gpu_started_at_seal": False, "fold1_access": False,
        "official_validation_access": False,
    }
    # Ready is a non-authorizing leaf.  The authorization is published last;
    # a crash between them remains fail-closed and exact replay is byte-safe.
    ready_output = _publish_json_replay_safe(ready_path, ready, "final-gate ready config")
    authorization_output = _publish_json_replay_safe(
        authorization_path, authorization, "final-gate run authorization"
    )
    return ready_output, authorization_output


@dataclass(frozen=True)
class ExecutionContext:
    authorization_path: Path
    authorization_sha256: str
    authorization_identity: Mapping[str, int]
    preregistration_path: Path
    preregistration: Mapping[str, Any]
    ready_path: Path
    ready: Mapping[str, Any]
    r6: R6Exact80Binding
    inventory: GTInventoryBinding
    outputs: Mapping[str, Path]


class RunClaimConsumed(PermissionError):
    """The one permitted production attempt has already been consumed."""


@dataclass(frozen=True)
class _RunCapability:
    token: str
    writer_fd: int
    parent_fd: int
    runtime_fd: int


@dataclass(frozen=True)
class _RunAuthority:
    creator_pid: int
    context: ExecutionContext
    claim_bytes: bytes
    claim_identity: Mapping[str, int]
    parent_identity: tuple[int, int]
    parent_chain: tuple[tuple[int, int], ...]
    runtime_identity: tuple[int, int]
    runtime_directories: dict[str, tuple[int, int]]


_RUN_AUTHORITIES: dict[str, _RunAuthority] = {}


def _host_target_probe_at_fd(parent_fd: int) -> None:
    """Verify host semantics relative to one already-held output-parent FD."""

    parent_info = os.fstat(parent_fd)
    if not stat.S_ISDIR(parent_info.st_mode):
        raise NotADirectoryError("held output host parent is not a directory")
    leaf = f".{NAMESPACE}.probe.{os.getpid()}.{secrets.token_hex(12)}"
    probe_fd = -1
    try:
        os.mkdir(leaf, 0o700, dir_fd=parent_fd)
        probe_fd = os.open(leaf, _DIR_FLAGS, dir_fd=parent_fd)
        source_fd = os.open(
            "source.tmp",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o444,
            dir_fd=probe_fd,
        )
        try:
            payload = secrets.token_bytes(32)
            view = memoryview(payload)
            while view:
                count = os.write(source_fd, view)
                if count < 1:
                    raise OSError("host atomic probe short write")
                view = view[count:]
            os.fsync(source_fd)
        finally:
            os.close(source_fd)
        os.link(
            "source.tmp", "published", src_dir_fd=probe_fd,
            dst_dir_fd=probe_fd, follow_symlinks=False,
        )
        os.fsync(probe_fd)
        left = os.stat("source.tmp", dir_fd=probe_fd, follow_symlinks=False)
        right = os.stat("published", dir_fd=probe_fd, follow_symlinks=False)
        if (
            (left.st_dev, left.st_ino, left.st_nlink)
            != (right.st_dev, right.st_ino, right.st_nlink)
            or left.st_nlink != 2
        ):
            raise OSError("host atomic probe hardlink identity differs")
        os.unlink("published", dir_fd=probe_fd)
        os.unlink("source.tmp", dir_fd=probe_fd)
        os.fsync(probe_fd)
        os.close(probe_fd)
        probe_fd = -1
        os.rmdir(leaf, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except BaseException:
        if probe_fd >= 0:
            for name in ("published", "source.tmp"):
                try:
                    os.unlink(name, dir_fd=probe_fd)
                except FileNotFoundError:
                    pass
            os.close(probe_fd)
        try:
            os.rmdir(leaf, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        raise
    # parent_fd ownership remains with _claim_runtime across the next steps.


def _claim_bytes(context: ExecutionContext, creator_pid: int) -> bytes:
    return canonical_json({
        "schema": "boxfusion.ca1m_tr3d_terminal_gate_run_claim.v5.final.r4",
        "complete": True, "create_only": True, "single_attempt": True,
        "namespace": NAMESPACE, "creator_pid": creator_pid,
        "authorization_sha256": context.authorization_sha256,
        "preregistration_sha256": sha256_file(context.preregistration_path),
        "r6_wrapper_sha256": context.r6.wrapper_sha256,
        "r6_authorization_commit_id": context.r6.authorization_commit_id,
    })


def _assert_lock_still_owned(writer_fd: int, parent_fd: int) -> None:
    probe = os.open(
        RUN_CLAIM_PATH.name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd,
    )
    blocked = False
    try:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            blocked = True
        else:
            fcntl.flock(probe, fcntl.LOCK_UN)
    finally:
        os.close(probe)
    if not blocked:
        raise PermissionError("final-gate run claim lock is no longer held")
    try:
        fcntl.flock(writer_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        if error.errno in (errno.EACCES, errno.EAGAIN):
            raise PermissionError(
                "final-gate writer FD is not the lock-owning open description"
            ) from error
        raise


def _claim_runtime() -> _RunCapability:
    """Freshly derive authority and consume one attempt on one held parent FD."""

    parent_fd, parent_chain = _open_dir_chain(
        OUTPUT_PARENT_PATH, "final-gate fixed output parent",
    )
    writer_fd = -1
    runtime_fd = -1
    token: str | None = None
    try:
        context = load_execution_context(
            RUN_AUTHORIZATION_PATH, require_outputs_absent=True,
        )
        if (
            _absolute_lexical(RUNTIME_ROOT)
            != _absolute_lexical(OUTPUT_PARENT_PATH / NAMESPACE)
            or _absolute_lexical(RUN_CLAIM_PATH)
            != _absolute_lexical(OUTPUT_PARENT_PATH / f".{NAMESPACE}.run.claim")
            or context.outputs != OUTPUT_PATHS
        ):
            raise PermissionError("final-gate canonical runtime/claim paths differ")
        _host_target_probe_at_fd(parent_fd)
        creator_pid = os.getpid()
        payload = _claim_bytes(context, creator_pid)
        try:
            writer_fd = os.open(
                RUN_CLAIM_PATH.name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o444, dir_fd=parent_fd,
            )
        except FileExistsError as error:
            raise RunClaimConsumed(
                "final-gate run claim already exists; restart/resume is forbidden"
            ) from error
        fcntl.flock(writer_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        view = memoryview(payload)
        while view:
            count = os.write(writer_fd, view)
            if count < 1:
                raise OSError("short write for permanent final-gate run claim")
            view = view[count:]
        os.fsync(writer_fd)
        os.fsync(parent_fd)
        claim_data, claim_identity = _read_bytes_at_fd(
            parent_fd, RUN_CLAIM_PATH.name, "final-gate held run claim",
        )
        if claim_data != payload:
            raise RuntimeError("final-gate held run claim bytes differ")
        try:
            os.mkdir(NAMESPACE, 0o700, dir_fd=parent_fd)
        except FileExistsError as error:
            raise FileExistsError(
                "runtime root appeared after context validation; claim is consumed"
            ) from error
        os.fsync(parent_fd)
        runtime_fd = os.open(NAMESPACE, _DIR_FLAGS, dir_fd=parent_fd)
        claim_info = os.fstat(writer_fd)
        parent_info = os.fstat(parent_fd)
        runtime_info = os.fstat(runtime_fd)
        if not stat.S_ISDIR(runtime_info.st_mode):
            raise NotADirectoryError("claimed runtime root is not a directory")
        if _identity(claim_info) != claim_identity:
            raise RuntimeError("final-gate writer/claim inode differs")
        token = secrets.token_hex(32)
        if token in _RUN_AUTHORITIES:
            raise RuntimeError("final-gate private capability collision")
        authority = _RunAuthority(
            creator_pid=creator_pid, context=context, claim_bytes=payload,
            claim_identity=claim_identity,
            parent_identity=(int(parent_info.st_dev), int(parent_info.st_ino)),
            parent_chain=parent_chain,
            runtime_identity=(int(runtime_info.st_dev), int(runtime_info.st_ino)),
            runtime_directories={},
        )
        _RUN_AUTHORITIES[token] = authority
        capability = _RunCapability(
            token=token, writer_fd=writer_fd, parent_fd=parent_fd,
            runtime_fd=runtime_fd,
        )
        _guard_run_capability(capability)
        return capability
    except BaseException:
        if token is not None:
            _RUN_AUTHORITIES.pop(token, None)
        if runtime_fd >= 0:
            os.close(runtime_fd)
        if writer_fd >= 0:
            os.close(writer_fd)
        os.close(parent_fd)
        raise


def _guard_run_capability(capability: _RunCapability) -> _RunAuthority:
    if type(capability) is not _RunCapability:
        raise PermissionError("final-gate run capability type differs")
    authority = _RUN_AUTHORITIES.get(capability.token)
    if authority is None:
        raise PermissionError("final-gate run capability is not registered")
    if os.getpid() != authority.creator_pid:
        raise PermissionError("final-gate capability used outside its creator process")
    if capability.writer_fd < 0 or capability.parent_fd < 0 or capability.runtime_fd < 0:
        raise PermissionError("final-gate persistent authority descriptor is closed")
    try:
        writer = os.fstat(capability.writer_fd)
        parent = os.fstat(capability.parent_fd)
        runtime = os.fstat(capability.runtime_fd)
        canonical = os.stat(
            RUN_CLAIM_PATH.name, dir_fd=capability.parent_fd,
            follow_symlinks=False,
        )
        canonical_runtime = os.stat(
            NAMESPACE, dir_fd=capability.parent_fd, follow_symlinks=False,
        )
    except OSError as error:
        raise PermissionError("final-gate authority descriptor/path is invalid") from error
    if (
        _identity(writer) != authority.claim_identity
        or _identity(canonical) != authority.claim_identity
        or writer.st_nlink != 1
    ):
        raise PermissionError("final-gate fixed run claim inode differs")
    claim_data, claim_identity = _read_bytes_at_fd(
        capability.parent_fd, RUN_CLAIM_PATH.name,
        "final-gate fixed held run claim",
    )
    if claim_data != authority.claim_bytes or claim_identity != authority.claim_identity:
        raise PermissionError("final-gate fixed run claim bytes/identity differ")
    _assert_lock_still_owned(capability.writer_fd, capability.parent_fd)
    if (
        not stat.S_ISDIR(parent.st_mode)
        or (int(parent.st_dev), int(parent.st_ino)) != authority.parent_identity
    ):
        raise PermissionError("final-gate held output-parent descriptor differs")
    if (
        not stat.S_ISDIR(runtime.st_mode)
        or not stat.S_ISDIR(canonical_runtime.st_mode)
        or (int(runtime.st_dev), int(runtime.st_ino)) != authority.runtime_identity
        or (int(canonical_runtime.st_dev), int(canonical_runtime.st_ino))
        != authority.runtime_identity
    ):
        raise PermissionError("final-gate fixed runtime-root inode differs")
    reopened_fd, reopened_chain = _open_dir_chain(
        OUTPUT_PARENT_PATH, "final-gate fixed output parent",
    )
    try:
        reopened = os.fstat(reopened_fd)
    finally:
        os.close(reopened_fd)
    if (
        reopened_chain != authority.parent_chain
        or (int(reopened.st_dev), int(reopened.st_ino)) != authority.parent_identity
    ):
        raise PermissionError("final-gate fixed output-parent chain changed")
    reopened_runtime_fd, runtime_chain = _open_dir_chain(
        RUNTIME_ROOT, "final-gate fixed runtime root",
    )
    try:
        reopened_runtime = os.fstat(reopened_runtime_fd)
    finally:
        os.close(reopened_runtime_fd)
    if (
        tuple(runtime_chain[:-1]) != authority.parent_chain
        or (int(reopened_runtime.st_dev), int(reopened_runtime.st_ino))
        != authority.runtime_identity
    ):
        raise PermissionError("final-gate canonical runtime-root chain changed")
    for relative, expected in sorted(
        authority.runtime_directories.items(), key=lambda item: item[0].count("/"),
    ):
        descriptor = os.dup(capability.runtime_fd)
        try:
            for component in Path(relative).parts:
                next_descriptor = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
            current = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(current.st_mode)
                or (int(current.st_dev), int(current.st_ino)) != expected
            ):
                raise PermissionError(
                    "final-gate registered runtime subdirectory changed"
                )
        except OSError as error:
            raise PermissionError(
                "final-gate registered runtime subdirectory is unavailable"
            ) from error
        finally:
            os.close(descriptor)
    return authority


def _release_run_capability(capability: _RunCapability) -> None:
    _RUN_AUTHORITIES.pop(capability.token, None)
    for descriptor in (
        capability.runtime_fd, capability.writer_fd, capability.parent_fd,
    ):
        try:
            os.close(descriptor)
        except OSError:
            pass


def _validate_implementation_records(records: Any) -> None:
    if not isinstance(records, Mapping) or set(records) != set(IMPLEMENTATION_PATHS):
        raise ValueError("final-gate implementation records differ")
    for key, path in IMPLEMENTATION_PATHS.items():
        validate_artifact_record(
            records[key], f"final-gate implementation {key}",
            canonical_path=path, require_identity=True, immutable=False,
        )


def load_execution_context(
    authorization_path: Path = RUN_AUTHORIZATION_PATH,
    *, require_outputs_absent: bool = True,
) -> ExecutionContext:
    if _absolute_lexical(authorization_path) != _absolute_lexical(RUN_AUTHORIZATION_PATH):
        raise ValueError("final-gate run authorization path is noncanonical")
    auth_path, authorization, auth_data, auth_identity = stable_json(
        authorization_path, "final-gate run authorization", schema=AUTHORIZATION_SCHEMA
    )
    if (
        authorization.get("complete") is not True
        or authorization.get("create_only") is not True
        or authorization.get("namespace") != NAMESPACE
        or authorization.get("commit_role") != "last_published_unique_run_gate"
        or authorization.get("ground_truth_access_at_seal") is not False
        or authorization.get("output_runtime_root_created_at_seal") is not False
        or authorization.get("gpu_started_at_seal") is not False
        or authorization.get("fold1_access") is not False
        or authorization.get("official_validation_access") is not False
    ):
        raise PermissionError("final-gate run authorization semantics differ")
    prereg_path, prereg, r6, inventory = validate_scientific_preregistration(
        Path(str((authorization.get("preregistration") or {}).get("path", "")))
    )
    validate_artifact_record(
        authorization.get("preregistration"), "authorization preregistration",
        schema=PREREGISTRATION_SCHEMA, canonical_path=PREREGISTRATION_PATH,
        require_identity=True,
    )
    ready_path, ready, ready_data, _ = _record_json(
        authorization.get("ready_config"), "authorization ready config",
        schema=READY_SCHEMA, canonical_path=READY_CONFIG_PATH,
    )
    expected_ready = _ready_payload(prereg_path, prereg)
    if ready_data != canonical_json(expected_ready):
        raise ValueError("final-gate ready/current preregistration differ")
    commit_material = {
        "schema": "boxfusion.ca1m_tr3d_terminal_gate_commit_material.v5.final.r4",
        "preregistration_sha256": sha256_file(prereg_path),
        "ready_config_sha256": sha256_bytes(ready_data),
        "r6_wrapper_sha256": r6.wrapper_sha256,
        "r6_authorization_commit_id": r6.authorization_commit_id,
        "annotation_inventory_sha256": inventory.sha256,
    }
    if (
        authorization.get("commit_id") != sha256_bytes(canonical_json(commit_material))
        or authorization.get("r6_wrapper_sha256") != r6.wrapper_sha256
        or authorization.get("r6_authorization_commit_id") != r6.authorization_commit_id
        or authorization.get("annotation_inventory_sha256") != inventory.sha256
        or authorization.get("authorizations") != expected_ready["authorizations"]
        or sha256_bytes(auth_data) != sha256_file(auth_path)
    ):
        raise PermissionError("final-gate authorization commit binding differs")
    _validate_implementation_records(prereg.get("implementation"))
    outputs = {key: Path(value) for key, value in prereg["outputs"].items()}
    if outputs != OUTPUT_PATHS:
        raise ValueError("final-gate runtime output paths differ")
    if require_outputs_absent:
        for key, path in outputs.items():
            if key == "materialization_root":
                exists = path.exists() or path.is_symlink()
            else:
                exists = path.exists() or path.is_symlink()
            if exists:
                raise FileExistsError(f"refusing pre-existing final-gate output {key}: {path}")
        if RUNTIME_ROOT.exists() or RUNTIME_ROOT.is_symlink():
            raise FileExistsError(f"refusing pre-existing final-gate runtime root: {RUNTIME_ROOT}")
    return ExecutionContext(
        auth_path, sha256_bytes(auth_data), _deep_immutable(auth_identity),
        prereg_path, _deep_immutable(prereg), ready_path,
        _deep_immutable(ready), r6, inventory, _deep_immutable(outputs),
    )


def revalidate_execution_inputs(context: ExecutionContext) -> None:
    if context.authorization_path != _absolute_lexical(RUN_AUTHORIZATION_PATH):
        raise PermissionError("formal context authorization is not canonical")
    fresh = load_execution_context(
        RUN_AUTHORIZATION_PATH, require_outputs_absent=False,
    )
    if fresh != context:
        raise RuntimeError(
            "fresh canonical authorization/R6/candidate/folds/GT bindings changed"
        )


def _fresh_authority_context(capability: _RunCapability) -> ExecutionContext:
    """Return only the registered context after a full fresh canonical rederive."""

    authority = _guard_run_capability(capability)
    fresh = load_execution_context(
        RUN_AUTHORIZATION_PATH, require_outputs_absent=False,
    )
    if fresh != authority.context:
        raise RuntimeError(
            "fresh canonical formal context differs from private run authority"
        )
    return authority.context


def _revalidate_inventory_master(context: ExecutionContext) -> None:
    source, _, data, identity = stable_json(
        GT_INVENTORY_PATH, "canonical CA-train GT inventory",
        schema=GT_INVENTORY_SCHEMA,
    )
    if (
        source != context.inventory.path
        or sha256_bytes(data) != context.inventory.sha256
        or identity != context.inventory.identity
        or context.inventory.sha256 != GT_INVENTORY_SHA256
    ):
        raise PermissionError("canonical CA-train GT inventory changed during execution")


def _inventory_ground_truth_loader(
    context: ExecutionContext, capability: _RunCapability,
) -> Callable[[str], np.ndarray]:
    """Build the private production loader only after the run claim is held."""

    allowed = set(context.r6.scene_folds)

    def load(scene_id: str) -> np.ndarray:
        fresh_context = _fresh_authority_context(capability)
        if fresh_context is not context:
            raise PermissionError("GT loader lost its unique authority context")
        scene = str(scene_id)
        if scene not in allowed or context.r6.scene_folds[scene] not in (0, 2, 3, 4):
            raise PermissionError("GT request is outside exact fit60+fold0-dev20")
        authority = _guard_run_capability(capability)
        if authority.context is not context:
            raise PermissionError("GT loader context is not the claimed authority")
        _revalidate_inventory_master(context)
        row = context.inventory.scene_rows[scene]
        box_record = row.get("box") or {}
        canonical_box_path = (
            GT_SHADOW_ROOT / scene / "derived_train_gt_boxes.npy"
        )
        path, data, _ = validate_artifact_record(
            box_record, f"CA-train GT boxes {scene}",
            canonical_path=canonical_box_path,
        )
        validate_artifact_record(
            row.get("manifest"), f"CA-train GT manifest {scene}",
            canonical_path=GT_SHADOW_ROOT / scene / "derived_train_gt_manifest.json",
        )
        # np.load receives bytes from the already inode-bound read, avoiding a
        # second path open after the provenance check.
        from io import BytesIO
        value = np.asarray(np.load(BytesIO(data), allow_pickle=False))
        if (
            value.dtype != np.dtype(np.float64)
            or value.ndim != 3 or value.shape[1:] != (8, 3)
            or not np.isfinite(value).all()
            or sha256_bytes(data) != box_record.get("sha256")
            or path != _absolute_lexical(canonical_box_path)
        ):
            raise ValueError(f"{scene}: CA-train GT array contract differs")
        return value

    return load


def _guarded_loader(
    context: ExecutionContext, loader: Callable[[str], Any], allowed_folds: Sequence[int],
    capability: _RunCapability | None = None,
) -> Callable[[str], Any]:
    allowed = {
        scene for scene, fold in context.r6.scene_folds.items()
        if fold in tuple(int(value) for value in allowed_folds)
    }

    def load(scene: str) -> Any:
        if str(scene) not in allowed:
            raise PermissionError("ground-truth request is outside preregistered partition")
        if capability is not None:
            fresh_context = _fresh_authority_context(capability)
            if fresh_context is not context:
                raise PermissionError("ground-truth context is not the claimed authority")
        return loader(str(scene))

    return load


def _materialize_fold0(
    dataset: gate.GateDatasetV5, *, policy_path: Path,
    output_root: Path, manifest_path: Path,
) -> Path:
    policy_source, policy, model = gate.load_gate_policy_v5(policy_path, require_oof_pass=True)
    iou, probability, gain = gate.predict_gate_model_v5(model, dataset.features)
    predictions = gate.GatePredictionsV5(
        iou, probability, gain,
        np.asarray(["[2,3,4]"] * len(iou)),
    )
    thresholds = policy.get("thresholds") or {}
    selection = gate.select_replacements_v5(
        dataset, predictions,
        iou_threshold=float(thresholds["candidate_iou"]),
        gain_threshold=float(thresholds["same_gt_gain"]),
        probability_threshold=float(thresholds["iou50_probability"]),
    )
    selected_by_scene: dict[str, list[int]] = {scene: [] for scene in dataset.scene_order.astype(str)}
    for candidate_position in selection.candidate_positions.tolist():
        selected_by_scene[str(dataset.candidate_scene_ids[candidate_position])].append(
            int(candidate_position)
        )
    rows: list[dict[str, Any]] = []
    for scene in dataset.scene_order.astype(str).tolist():
        anchor_global = np.flatnonzero(dataset.anchor_scene_ids.astype(str) == scene)
        candidate_global = np.flatnonzero(dataset.candidate_scene_ids.astype(str) == scene)
        anchor_lookup = {int(global_row): local for local, global_row in enumerate(anchor_global.tolist())}
        candidate_lookup = {int(global_row): local for local, global_row in enumerate(candidate_global.tolist())}
        chosen = np.asarray(selected_by_scene[scene], np.int64)
        replace = np.asarray([
            anchor_lookup[int(dataset.candidate_anchor_positions[row])] for row in chosen
        ], np.int64)
        candidate_rows = np.asarray([candidate_lookup[int(row)] for row in chosen], np.int64)
        result = gate.materialize_geometry_only_v5(
            anchor_corners=np.asarray(dataset.anchor_corners[anchor_global], np.float32),
            anchor_scores=np.asarray(dataset.anchor_scores_oof[anchor_global], np.float32),
            candidate_corners=np.asarray(dataset.candidate_corners[candidate_global], np.float32),
            anchor_indices=replace, candidate_rows=candidate_rows,
        )
        target = output_root / f"{scene}_geometry_only_v5_final.npz"
        output = gate.write_materialized_geometry_v5(
            target, scene_id=scene,
            source_anchor_sha256=gate.sha256_array(dataset.anchor_corners[anchor_global]),
            source_candidate_sha256=gate.sha256_array(dataset.candidate_corners[candidate_global]),
            policy_sha256=sha256_file(policy_source), result=result,
        )
        rows.append({
            "scene_id": scene, "fold_id": 0,
            "path": os.fspath(output), "sha256": sha256_file(output),
            "identity": artifact_record(
                output, f"materialized scene {scene}", immutable=False,
            )["identity"],
            "anchor_count": len(anchor_global), "replacement_count": len(chosen),
            "scores_preserved": True, "row_order_preserved": True,
            "row_count_preserved": True,
        })
    payload = {
        "schema": MATERIALIZATION_COLLECTION_SCHEMA,
        "complete": True, "create_only": True,
        "report_label": "noncanonical_reused_dev_exploratory_materialization",
        "scene_count": 20, "fold_id": 0,
        "geometry_only": True, "scores_preserved": True,
        "row_order_preserved": True, "row_count_preserved": True,
        "policy_activation_authorized": False,
        "fold1_or_validation_authorized": False,
        "policy": artifact_record(
            policy_source, "final-gate policy", schema=gate.POLICY_SCHEMA,
            immutable=False,
        ),
        "scenes": rows,
    }
    return gate.write_json_create_only(
        manifest_path, payload, "final-gate fold0 materialization manifest"
    )


def _stage_guard(
    context: ExecutionContext, capability: _RunCapability | None,
) -> None:
    if capability is not None:
        fresh_context = _fresh_authority_context(capability)
        if fresh_context is not context:
            raise PermissionError("stage context is not the claimed authority")
        return
    revalidate_execution_inputs(context)


def _run_final_gate_impl(
    *, context: ExecutionContext, ground_truth_loader: Callable[[str], Any],
    runtime_root: Path, capability: _RunCapability | None,
) -> Path:
    """Shared numerical chain; production authority is supplied only internally."""

    # This call is CPU-only.  There is no device argument and no import of a
    # detector, CUDA, fold1, validation or legacy v4 gate implementation.
    with _fuse_gate_io(runtime_root=runtime_root, capability=capability):
        _stage_guard(context, capability)
        fit_loader = _guarded_loader(
            context, ground_truth_loader, (2, 3, 4), capability,
        )
        fit = gate.build_labeled_dataset_v5(
            context.r6.collection_path, purpose="fold234_oof_fit",
            ground_truth_loader=fit_loader,
        )
        _stage_guard(context, capability)
        fit_artifact, fit_manifest = gate.seal_gate_dataset_v5(
            fit, artifact_path=context.outputs["fit_dataset"],
            manifest_path=context.outputs["fit_dataset_manifest"],
        )
        _stage_guard(context, capability)
        result = gate.train_gate_oof_v5(fit)
        _stage_guard(context, capability)
        oof, threshold, policy = gate.seal_gate_oof_result_v5(
            fit, result, oof_path=context.outputs["oof_predictions"],
            threshold_path=context.outputs["threshold_receipt"],
            policy_path=context.outputs["exploratory_policy"],
        )
        _stage_guard(context, capability)
        common = {
            "authorization": artifact_record(
                context.authorization_path, "final-gate authorization",
                schema=AUTHORIZATION_SCHEMA,
            ),
            "r6_wrapper_sha256": context.r6.wrapper_sha256,
            "r6_candidate_collection_sha256": context.r6.collection_sha256,
            "fit_dataset": artifact_record(
                fit_artifact, "fit60 dataset", schema=gate.DATASET_SCHEMA,
                immutable=False,
            ),
            "fit_dataset_manifest": artifact_record(
                fit_manifest, "fit60 dataset manifest", immutable=False,
            ),
            "oof_predictions": artifact_record(
                oof, "fold234 OOF", schema=gate.OOF_SCHEMA, immutable=False,
            ),
            "threshold_receipt": artifact_record(
                threshold, "fold234 threshold receipt",
                schema=gate.THRESHOLD_SCHEMA, immutable=False,
            ),
            "exploratory_policy": artifact_record(
                policy, "exploratory policy", schema=gate.POLICY_SCHEMA,
                immutable=False,
            ),
            "fold1_access": False, "official_validation_access": False,
            "gpu_started": False, "policy_activation_authorized": False,
        }
        if result.threshold_receipt.get("safety_gate_passed") is not True:
            payload = {
                "schema": STOP_SCHEMA, "complete": True, "create_only": True,
                "status": "STOP_FOLD234_OOF_GATE_FAIL",
                "fold0_ground_truth_access": False,
                "fold0_materialization_created": False,
                **common,
            }
            output = gate.write_json_create_only(
                context.outputs["stop_receipt"], payload, "final-gate STOP receipt"
            )
            _stage_guard(context, capability)
            return output
        fold0_loader = _guarded_loader(
            context, ground_truth_loader, (0,), capability,
        )
        fold0 = gate.build_labeled_dataset_v5(
            context.r6.collection_path, purpose="fold0_reused_dev",
            ground_truth_loader=fold0_loader,
        )
        _stage_guard(context, capability)
        fold0_artifact, fold0_manifest = gate.seal_gate_dataset_v5(
            fold0, artifact_path=context.outputs["fold0_dataset"],
            manifest_path=context.outputs["fold0_dataset_manifest"],
        )
        _stage_guard(context, capability)
        fold0_report = gate.evaluate_fold0_reused_dev_v5(
            fold0, policy_path=policy, output_path=context.outputs["fold0_report"]
        )
        _stage_guard(context, capability)
        materialization = _materialize_fold0(
            fold0, policy_path=policy,
            output_root=context.outputs["materialization_root"],
            manifest_path=context.outputs["materialization_manifest"],
        )
        _stage_guard(context, capability)
        payload = {
            "schema": RUN_SCHEMA, "complete": True, "create_only": True,
            "status": "PASS_EXPLORATORY_FOLD0_DIAGNOSTIC_COMPLETE",
            "thresholds_selected_only_from_fold234_oof": True,
            "fold0_retuning": False, "fold0_model_selection": False,
            "fold0_result_can_authorize_policy": False,
            **common,
            "fold0_dataset": artifact_record(
                fold0_artifact, "fold0 dataset", schema=gate.DATASET_SCHEMA,
                immutable=False,
            ),
            "fold0_dataset_manifest": artifact_record(
                fold0_manifest, "fold0 dataset manifest", immutable=False,
            ),
            "fold0_report": artifact_record(
                fold0_report, "fold0 report", schema=gate.FOLD0_REPORT_SCHEMA,
                immutable=False,
            ),
            "materialization_manifest": artifact_record(
                materialization, "fold0 materialization manifest",
                schema=MATERIALIZATION_COLLECTION_SCHEMA, immutable=False,
            ),
        }
        output = gate.write_json_create_only(
            context.outputs["run_receipt"], payload, "final-gate run receipt"
        )
        _stage_guard(context, capability)
        return output


def run_final_gate() -> Path:
    """Freshly derive and execute the sole canonical formal attempt."""

    capability = _claim_runtime()
    try:
        context = _fresh_authority_context(capability)
        _stage_guard(context, capability)
        loader = _inventory_ground_truth_loader(context, capability)
        return _run_final_gate_impl(
            context=context, ground_truth_loader=loader,
            runtime_root=RUNTIME_ROOT, capability=capability,
        )
    finally:
        _release_run_capability(capability)


def run_final_gate_nonproduction(
    *, context: ExecutionContext, ground_truth_loader: Callable[[str], Any],
    nonproduction_root: Path,
) -> Path:
    """Test-only pure route; it is structurally unable to write production paths."""

    root = _canonical_absolute(_absolute_lexical(nonproduction_root), "nonproduction root")
    try:
        root.relative_to(Path("/tmp"))
    except ValueError:
        raise PermissionError("nonproduction gate root must be below /tmp") from None
    if root == CANONICAL_RUNTIME_ROOT or CANONICAL_RUNTIME_ROOT in root.parents:
        raise PermissionError("nonproduction helper cannot address canonical runtime")
    for name, path in context.outputs.items():
        if name == "materialization_root":
            candidate = path
        else:
            candidate = path
        try:
            _absolute_lexical(candidate).relative_to(root)
        except ValueError:
            raise PermissionError(
                "nonproduction helper output escapes its /tmp root"
            ) from None
    return _run_final_gate_impl(
        context=context, ground_truth_loader=ground_truth_loader,
        runtime_root=root, capability=None,
    )


__all__ = [
    "AUTHORIZATION_SCHEMA", "DEFAULT_PENDING_CONFIG", "FinalGateProtocolError",
    "GT_INVENTORY_SCHEMA", "IMPLEMENTATION_PATHS", "MANIFEST_ROOT",
    "MATERIALIZATION_COLLECTION_SCHEMA", "NAMESPACE", "OUTPUT_PATHS",
    "PENDING_SCHEMA", "PREREGISTRATION_PATH", "PREREGISTRATION_SCHEMA",
    "PROTOCOL_PATH", "PROTOCOL_SCHEMA",
    "PendingR6Inputs", "R6Exact80Binding", "R6_RECEIPT_PATH",
    "R6_RECEIPT_SCHEMA", "READY_CONFIG_PATH", "READY_SCHEMA",
    "RUN_AUTHORIZATION_PATH", "RUN_SCHEMA", "STOP_SCHEMA", "ap_parity_fixture",
    "ExecutionContext", "GTInventoryBinding", "RunClaimConsumed", "artifact_record",
    "build_preregistration_payload", "build_static_protocol_payload", "canonical_json",
    "load_execution_context", "load_r6_exact80_binding",
    "operational_preflight_pending", "science_contract", "sha256_bytes",
    "sha256_file", "seal_ready_authorization", "seal_scientific_preregistration",
    "seal_static_protocol",
    "stable_bytes", "stable_json", "static_preflight", "run_final_gate",
    "run_final_gate_nonproduction",
    "validate_artifact_record", "validate_gt_inventory_metadata",
    "validate_pending_config", "validate_scientific_preregistration",
    "validate_static_protocol",
]
