"""Hardened ready boundary for the CA-1M E961 terminal-input route (v5 R3).

R3 deliberately does not reimplement the already tested P/O/E/M computation.
It freezes and verifies the inputs, establishes a host-safe output boundary, and
then delegates those four stages to the SHA-bound R2 execution core.  Import is
side-effect free and neither this module nor its preflight opens annotations or
ground truth.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
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
from typing import Any, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/ca1m_tr3d_e961_terminal_inputs_v5_r3_pending.json"
MANIFEST_ROOT = ROOT / "manifests/ca1m_tr3d_e961_terminal_inputs_v5_r3"
V1_PREREGISTRATION_PATH = MANIFEST_ROOT / "PREREGISTRATION.json"
V1_PREREGISTRATION_SHA256 = "e0de5c980dab4b04bcc0c5e9f6943a2da613368473cf5916d7a3a4bf149d52a3"
V1_INVALID_PATH = MANIFEST_ROOT / "PREREGISTRATION_V1_INVALID.json"
PREREGISTRATION_PATH = MANIFEST_ROOT / "PREREGISTRATION_V2.json"
READY_CONFIG_PATH = MANIFEST_ROOT / "READY_CONFIG.json"
RUN_AUTHORIZATION_PATH = MANIFEST_ROOT / "RUN_AUTHORIZATION.json"
CONTINUATION_CANONICAL_PATH = (
    ROOT / "reports/ca1m_tr3d_e961_outer_dev_eval_v1/CONTINUATION_RECEIPT.json"
)

CONFIG_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_config.v5.r3"
V1_PREREGISTRATION_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_preregistration.v5.r3"
V1_INVALID_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_preregistration_invalid.v5.r3"
PREREGISTRATION_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_preregistration.v5.r3.v2"
AUTH_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_run_authorization.v5.r3"
STATIC_REPORT_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_static_preflight.v5.r3"
OPERATIONAL_REPORT_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_operational_preflight.v5.r3"
NAMESPACE = "ca1m_tr3d_e961_terminal_inputs_xfit_r2_v5_r3"
OUTER_SCHEMA = "boxfusion.tr3d.ca1m_e961_outer_train_run.r2"
INNER_SCHEMA = "boxfusion.tr3d.ca1m_e961_inner_train_run.r2"
LEGACY_INNER_SCHEMA = "boxfusion.tr3d.ca1m_xfit_r2_inner_run_receipt.v1"
CONTINUATION_SCHEMA = "boxfusion.ca1m_tr3d_e961_outer_dev_continuation_receipt.v1"
ROLE_ORDER = ("outer_dev", "inner_holdout2", "inner_holdout3", "inner_holdout4")
ROLE_SPECS = {
    "outer_dev": ((2, 3, 4), 0, 0),
    "inner_holdout2": ((3, 4), 2, 1),
    "inner_holdout3": ((2, 4), 3, 2),
    "inner_holdout4": ((2, 3), 4, 3),
}
AUTHORIZATIONS = {
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

_SHA = __import__("re").compile(r"^[0-9a-f]{64}$")
_SCENE = __import__("re").compile(r"^[0-9]{8}$")
_DIR_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


class PendingOperationalInputs(PermissionError):
    """The frozen pending config has no operational authority."""


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _absolute(path: Path, name: str) -> Path:
    value = Path(path)
    if not value.is_absolute() or ".." in value.parts:
        raise ValueError(f"{name} must be absolute and normalized")
    normalized = Path(os.path.normpath(os.fspath(value)))
    if normalized != value:
        raise ValueError(f"{name} is not lexically canonical")
    return value


def _open_dir_chain(path: Path, name: str) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Open every parent by dirfd without following a symlink."""

    path = _absolute(path, name)
    fd = os.open(path.anchor, _DIR_FLAGS)
    identities: list[tuple[int, int]] = []
    try:
        root_stat = os.fstat(fd)
        identities.append((root_stat.st_dev, root_stat.st_ino))
        for part in path.parts[1:]:
            next_fd = os.open(part, _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = next_fd
            current = os.fstat(fd)
            if not stat.S_ISDIR(current.st_mode):
                raise NotADirectoryError(f"{name} component is not a directory: {part}")
            identities.append((current.st_dev, current.st_ino))
        return fd, tuple(identities)
    except Exception:
        os.close(fd)
        raise


def _verify_dir_chain(path: Path, expected: Sequence[tuple[int, int]], name: str) -> None:
    fd, actual = _open_dir_chain(path, name)
    os.close(fd)
    if tuple(actual) != tuple(expected):
        raise ValueError(f"{name} parent directory identity changed")


def stable_bytes(path: Path, name: str, *, nonempty: bool = True) -> bytes:
    """Read a regular file through a fully bound parent chain."""

    path = _absolute(Path(path), name)
    parent_fd, parent_ids = _open_dir_chain(path.parent, f"{name} parent")
    fd = -1
    try:
        fd = os.open(path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or (nonempty and before.st_size < 1):
            raise ValueError(f"{name} is not a regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(fd)
        current = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if _identity(before) != _identity(after) or _identity(after) != _identity(current):
            raise ValueError(f"{name} inode/content changed while read")
        result = b"".join(chunks)
        if len(result) != before.st_size:
            raise ValueError(f"{name} changed size while read")
    finally:
        if fd >= 0:
            os.close(fd)
        os.close(parent_fd)
    _verify_dir_chain(path.parent, parent_ids, f"{name} parent")
    return result


def sha256_file(path: Path) -> str:
    return sha256_bytes(stable_bytes(Path(path), "SHA256 input", nonempty=False))


def _sha(value: Any, name: str) -> str:
    result = str(value)
    if _SHA.fullmatch(result) is None:
        raise ValueError(f"{name} is not lowercase SHA256")
    return result


def read_json(path: Path, name: str) -> tuple[Path, dict[str, Any]]:
    source = _absolute(Path(path), name)
    try:
        value = json.loads(stable_bytes(source, name).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return source, value


def _file_record(record: Any, name: str, *, schema: str | None = None) -> tuple[Path, dict[str, Any] | None]:
    if not isinstance(record, Mapping):
        raise ValueError(f"{name} record must be an object")
    path = _absolute(Path(str(record.get("path", ""))), name)
    digest = _sha(record.get("sha256"), f"{name} SHA256")
    if sha256_file(path) != digest:
        raise ValueError(f"{name} SHA256 differs")
    value = None
    if schema is not None:
        if record.get("schema") != schema:
            raise ValueError(f"{name} record schema differs")
        _, value = read_json(path, name)
        if value.get("schema") != schema:
            raise ValueError(f"{name} payload schema differs")
    return path, value


def _pending_record(schema: str) -> dict[str, Any]:
    return {"state": "pending", "path": None, "sha256": None, "schema": schema}


def _record(path: Path, schema: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"state": "bound", "path": os.fspath(path), "sha256": sha256_file(path)}
    if schema is not None:
        result["schema"] = schema
    return result


def _scene_list(record: Any, name: str, count: int) -> tuple[str, ...]:
    path, _ = _file_record(record, name)
    rows = tuple(x.strip() for x in stable_bytes(path, name).decode().splitlines() if x.strip())
    if len(rows) != count or len(set(rows)) != count or any(_SCENE.fullmatch(x) is None for x in rows):
        raise ValueError(f"{name} is not exact{count} unique scene ids")
    return rows


def _under(path: Path, root: Path, name: str) -> Path:
    path, root = _absolute(path, name), _absolute(root, f"{name} root")
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes its canonical root") from error
    return path


def _load_frozen_pending() -> tuple[Path, dict[str, Any]]:
    source, cfg = read_json(DEFAULT_CONFIG, "frozen R3 pending config")
    if source != DEFAULT_CONFIG:
        raise ValueError("pending config path differs")
    return source, cfg


def _validate_config_shape(cfg: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    if cfg.get("schema") != CONFIG_SCHEMA or cfg.get("namespace") != NAMESPACE:
        raise ValueError("R3 config schema/namespace differs")
    if cfg.get("preregistration") != {
        "path": os.fspath(PREREGISTRATION_PATH), "schema": PREREGISTRATION_SCHEMA,
    }:
        raise ValueError("canonical preregistration target differs")
    if cfg.get("sealed_dynamic_outputs") != {
        "ready_config": os.fspath(READY_CONFIG_PATH),
        "run_authorization": os.fspath(RUN_AUTHORIZATION_PATH),
    }:
        raise ValueError("canonical ready/authorization targets differ")

    producers = cfg.get("producer_contracts") or {}
    outer, inner = producers.get("outer") or {}, producers.get("inner") or {}
    if outer.get("receipt_schema") != OUTER_SCHEMA or inner.get("receipt_schema") != INNER_SCHEMA:
        raise ValueError("producer receipt schemas differ")
    if inner.get("rejected_legacy_schema") != LEGACY_INNER_SCHEMA:
        raise ValueError("legacy 60-scene receipt rejection differs")
    outer_v, _ = _file_record(outer.get("verifier"), "official outer R2 verifier")
    inner_v, _ = _file_record(inner.get("verifier"), "official inner R2 verifier")
    if outer_v.name != "tr3d_ca1m_e961_outer_train_r2.py" or inner_v.name != "tr3d_ca1m_e961_inner_queue_r2.py":
        raise ValueError("official producer verifier identity differs")
    outer_root = _absolute(Path(str(outer.get("canonical_root", ""))), "outer canonical receipt root")
    inner_root = _absolute(Path(str(inner.get("canonical_root", ""))), "inner canonical receipt root")
    if outer_root != Path("/extra/ZhaoX/tr3d_ca1m_e961_outer_train_r2/runs"):
        raise ValueError("outer canonical root differs")
    if inner_root != Path("/extra/ZhaoX/tr3d_ca1m_e961_inner_queue_r2/runs"):
        raise ValueError("inner canonical root differs")
    _file_record(inner.get("static_queue_manifest"), "inner static queue manifest")

    scene = cfg.get("scene_contract") or {}
    _file_record(scene.get("selection_contract"), "E961 selection contract", schema="boxfusion.tr3d.ca1m_e961_selection.v1")
    if (scene.get("fit_folds"), scene.get("reused_dev_folds"), scene.get("fit_scene_count"),
        scene.get("reused_dev_scene_count"), scene.get("scene_count")) != ([2, 3, 4], [0], 60, 20, 80):
        raise ValueError("scene split contract differs")
    roles = scene.get("roles") or {}
    if tuple(roles) != ROLE_ORDER:
        raise ValueError("role order differs")
    role_scenes: dict[str, tuple[str, ...]] = {}
    heldout_sets: list[set[str]] = []
    for role in ROLE_ORDER:
        train_folds, heldout, order = ROLE_SPECS[role]
        row = roles.get(role) or {}
        if (row.get("order"), tuple(row.get("train_folds", ())), row.get("heldout_fold"),
            row.get("train_scenes"), row.get("heldout_scenes")) != (order, train_folds, heldout, 1001, 20):
            raise ValueError(f"{role}: role/fold/count contract differs")
        _scene_list(row.get("train_scene_list"), f"{role} exact1001 train list", 1001)
        predicted = _scene_list(row.get("predict_scene_list"), f"{role} exact20 prediction list", 20)
        role_scenes[role] = predicted
        heldout_sets.append(set(predicted))
    if len(set().union(*heldout_sets)) != 80 or any(
        heldout_sets[i] & heldout_sets[j] for i in range(4) for j in range(i + 1, 4)
    ):
        raise ValueError("role prediction sets are not disjoint exact80")

    static = cfg.get("static_inputs") or {}
    for key, schema in (
        ("final_base_collection", "boxfusion.ca1m_final_base_identity_audit.v1"),
        ("native_b6_collection", "boxfusion.ca1m_native_b6_final_base_train_collection.v2"),
        ("native_b6_oof_manifest", "boxfusion.ca1m_native_b6_oof_row_scores_manifest.v2"),
        ("native_b6_oof_sidecar", "boxfusion.ca1m_native_b6_oof_row_scores.v2"),
    ):
        _file_record(static.get(key), f"static input {key}", schema=schema if key != "native_b6_oof_sidecar" else None)
    points = static.get("processed_point_inputs") or {}
    if points.get("inventory_policy") != "exact100_gap20_point_inputs_content_sha256":
        raise ValueError("processed point inventory policy differs")
    for key in ("scene_list", "subset_manifest", "readiness", "downloaded_sha256"):
        _file_record(points.get(key), f"processed exact100 {key}")
    _file_record(
        points.get("point_parity_receipt"), "processed exact100 point parity receipt",
        schema="boxfusion.ca1m_tr3d_v4_lineage_training_point_parity.v1",
    )
    if _absolute(Path(str(points.get("root", ""))), "processed exact100 root") != Path("/extra/ZhaoX/boxfusion_ca1m_native_b6_train100_v1"):
        raise ValueError("processed exact100 root differs")

    runtime = cfg.get("runtime") or {}
    _file_record(runtime.get("point_inference_config"), "CA-only point inference config")
    if runtime.get("protocol") != {
        "pixel_stride": 4, "voxel_size_m": 0.01, "min_depth_m": 0.1,
        "max_depth_m": 6.0, "depth_scale": 1000.0, "score_threshold": 0.01,
        "max_proposals": 256, "near_iou": 0.15, "prefix_id": "p100_gap20",
    } or runtime.get("formal_device") != "cuda:0":
        raise ValueError("formal point/GPU protocol differs")
    for key in ("runtime_root", "project_root", "vendor_root"):
        directory = _absolute(Path(str(runtime.get(key, ""))), f"runtime {key}")
        fd, _ = _open_dir_chain(directory, f"runtime {key}")
        os.close(fd)
    worker = _absolute(Path(str(runtime.get("worker_python", ""))), "worker Python")
    if not os.access(worker, os.X_OK):
        raise ValueError("worker Python is not executable")

    implementation = cfg.get("implementation") or {}
    required_impl = {
        "current_core", "current_sealer", "current_preflight", "current_runner", "current_tests",
        "r2_execution_core", "point_builder", "point_cache_core", "checkpoint_binding_core",
        "worker_client", "worker_cli", "feature_builder", "candidate_reader", "native_observer",
        "native_loader", "association", "v5_manifest_runtime", "inference_contract",
        "rgbd_backprojection", "outer_eval_core",
    }
    if set(implementation) != required_impl:
        raise ValueError("complete implementation dependency inventory differs")
    for key, record in implementation.items():
        _file_record(record, f"implementation {key}")

    outputs = cfg.get("outputs") or {}
    namespace_root = _absolute(Path(str(outputs.get("namespace_root", ""))), "namespace root")
    if namespace_root != Path("/extra/ZhaoX") / NAMESPACE:
        raise ValueError("canonical output namespace root differs")
    expected_output_keys = {
        "namespace_root", "proposal_root", "overlay_root", "candidate_diagnostic_root",
        "evidence_root", "receipt_root", "manifest_root", "combined_manifest",
    }
    if set(outputs) != expected_output_keys:
        raise ValueError("output path inventory differs")
    for key, value in outputs.items():
        if key != "namespace_root":
            _under(Path(str(value)), namespace_root, f"output {key}")
    if cfg.get("access") != {
        "official_train_only": True, "ground_truth_access": False,
        "fold1_path_present": False, "official_validation_path_present": False,
        "scannet_weight_or_artifact_access": False,
        "old_terminal_v1_v4_artifact_access": False,
    }:
        raise ValueError("CA-only access contract differs")
    return role_scenes


def _numeric_png_names(path: Path, name: str) -> tuple[str, ...]:
    fd, identities = _open_dir_chain(path, name)
    try:
        result: dict[int, str] = {}
        with os.scandir(fd) as entries:
            for entry in entries:
                item = entry.name
                if not item.endswith(".png"):
                    continue
                try:
                    index = int(item[:-4])
                except ValueError:
                    continue
                if not entry.is_file(follow_symlinks=False) or index < 0 or index in result:
                    raise ValueError(f"{name}: invalid/duplicate numeric PNG")
                result[index] = item
    finally:
        os.close(fd)
    _verify_dir_chain(path, identities, name)
    if set(result) != set(range(len(result))) or not result:
        raise ValueError(f"{name}: frames must be contiguous 0..N-1")
    return tuple(result[index] for index in range(len(result)))


def _gap20(frame_count: int) -> tuple[int, ...]:
    if frame_count < 1:
        raise ValueError("frame count must be positive")
    rows: list[int] = []
    index = 0
    while index < frame_count:
        if index % 20 == 0:
            rows.append(index)
        index += 1
        if index == frame_count - 1 or index + 20 > frame_count - 1:
            break
    if not rows:
        raise ValueError("gap20 produced no point frames")
    return tuple(rows)


def processed_point_inventory(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Bind exact visible topology and the sealed exact100 point content proof.

    The existing parity receipt was produced by rebuilding every point array
    from reachable RGB/depth/pose/intrinsics and proving byte equality with the
    CA-only training point file.  R3 binds that receipt and independently
    rechecks the current non-GT frame topology; it does not repeatedly stream
    tens of gigabytes from the active FUSE volume.
    """

    points = cfg["static_inputs"]["processed_point_inputs"]
    scenes = _scene_list(points["scene_list"], "processed exact100 scene list", 100)
    root = _absolute(Path(points["root"]), "processed exact100 root")
    root_fd, root_ids = _open_dir_chain(root, "processed exact100 root")
    try:
        with os.scandir(root_fd) as entries:
            visible = {
                entry.name for entry in entries
                if not entry.name.startswith(".") and entry.is_dir(follow_symlinks=False)
            }
        if visible != set(scenes):
            raise ValueError("processed point root visible inventory is not exact100")
    finally:
        os.close(root_fd)
    _verify_dir_chain(root, root_ids, "processed exact100 root")

    parity_path, parity = _file_record(
        points["point_parity_receipt"], "processed exact100 point parity receipt",
        schema="boxfusion.ca1m_tr3d_v4_lineage_training_point_parity.v1",
    )
    if (
        parity.get("complete") is not True or parity.get("create_only") is not True
        or parity.get("scene_count") != 100 or parity.get("lineage_parity_scene_count") != 100
        or parity.get("point_array_parity_scene_count") != 100
        or parity.get("point_byte_parity_scene_count") != 100
        or parity.get("ground_truth_access") is not False
        or parity.get("data_root") != os.fspath(root)
        or (parity.get("scene_list") or {}).get("sha256") != points["scene_list"]["sha256"]
        or set(parity.get("scenes") or {}) != set(scenes)
    ):
        raise ValueError("processed exact100 parity receipt differs")
    scene_rows: list[dict[str, Any]] = []
    total_used = 0
    for scene in scenes:
        scene_root = root / scene
        rgb_names = _numeric_png_names(scene_root / "rgb", f"{scene} RGB frames")
        depth_names = _numeric_png_names(scene_root / "depth", f"{scene} depth frames")
        if rgb_names != depth_names:
            raise ValueError(f"{scene}: RGB/depth topology differs")
        used = _gap20(len(rgb_names))
        # Presence and regular-file identity only; content equality is sealed
        # by the point-array hashes in the parity receipt below.
        point_sources: list[str] = []
        for fixed in ("all_poses.npy",):
            path = scene_root / fixed
            if not stat.S_ISREG(os.stat(path, follow_symlinks=False).st_mode):
                raise ValueError(f"{scene}: point source {fixed} is not regular")
            stable_bytes(path, f"{scene} point source {fixed}")
            point_sources.append(fixed)
        per_frame = scene_root / "K_depth_per_frame.npy"
        intrinsic = per_frame if per_frame.exists() and not per_frame.is_symlink() else scene_root / "K_depth.txt"
        if not stat.S_ISREG(os.stat(intrinsic, follow_symlinks=False).st_mode):
            raise ValueError(f"{scene}: point intrinsics are not regular")
        stable_bytes(intrinsic, f"{scene} point intrinsics")
        point_sources.append(intrinsic.name)
        sealed = parity["scenes"][scene]
        if (
            sealed.get("frame_count") != len(rgb_names)
            or sealed.get("used_frame_count") != len(used)
            or sealed.get("last_reachable_keyframe") != used[-1]
            or sealed.get("lineage_equal") is not True or sealed.get("array_equal") is not True
            or sealed.get("byte_equal") is not True
            or any(_SHA.fullmatch(str(sealed.get(key, ""))) is None for key in (
                "used_frame_ids_sha256", "world_point_array_sha256",
                "local_point_array_sha256", "training_point_file_sha256",
            ))
            or sealed.get("local_point_array_sha256") != sealed.get("training_point_file_sha256")
        ):
            raise ValueError(f"{scene}: sealed point content/topology differs")
        topology = sha256_bytes(canonical_json({"rgb": list(rgb_names), "depth": list(depth_names)}))
        scene_rows.append({
            "scene_id": scene, "frame_count": len(rgb_names), "used_frame_ids": list(used),
            "numeric_frame_topology_sha256": topology, "point_sources": point_sources,
            "used_frame_ids_sha256": sealed["used_frame_ids_sha256"],
            "world_point_count": sealed["world_point_count"],
            "world_point_array_sha256": sealed["world_point_array_sha256"],
            "local_point_array_sha256": sealed["local_point_array_sha256"],
            "training_point_file_sha256": sealed["training_point_file_sha256"],
        })
        total_used += len(used)
    payload: dict[str, Any] = {
        "schema": "boxfusion.ca1m_tr3d_e961_processed_point_inventory.v5.r3",
        "complete": True, "exact_visible_scene_count": 100,
        "scene_order": list(scenes), "scene_count": 100,
        "used_rgbd_frame_count": total_used, "policy": points["inventory_policy"],
        "point_parity_receipt": {
            "path": os.fspath(parity_path), "sha256": sha256_file(parity_path),
            "schema": parity["schema"],
        },
        "ground_truth_access": False, "scenes": scene_rows,
    }
    payload["inventory_sha256"] = sha256_bytes(canonical_json(payload))
    return payload


def build_preregistration_payload(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    if Path(config_path) != DEFAULT_CONFIG:
        raise ValueError("static preregistration accepts only the canonical pending config")
    source, cfg = _load_frozen_pending()
    roles = _validate_config_shape(cfg)
    for role in ROLE_ORDER:
        schema = OUTER_SCHEMA if role == "outer_dev" else INNER_SCHEMA
        if cfg["scene_contract"]["roles"][role].get("source_success_receipt") != _pending_record(schema):
            raise ValueError(f"{role}: frozen pending producer receipt is not pending")
    if cfg.get("continuation_receipt") != _pending_record(CONTINUATION_SCHEMA):
        raise ValueError("frozen continuation receipt is not pending")
    if cfg.get("run_authorization") != _pending_record(AUTH_SCHEMA):
        raise ValueError("frozen run authorization is not pending")
    inventory = processed_point_inventory(cfg)
    invalid_path, invalid = _file_record(
        {"path": os.fspath(V1_INVALID_PATH), "sha256": sha256_file(V1_INVALID_PATH), "schema": V1_INVALID_SCHEMA},
        "R3 preregistration V1 invalidation", schema=V1_INVALID_SCHEMA,
    )
    if (
        invalid.get("complete") is not True or invalid.get("invalid") is not True
        or (invalid.get("predecessor") or {}).get("sha256") != V1_PREREGISTRATION_SHA256
        or invalid.get("corrected_continuation_path") != os.fspath(CONTINUATION_CANONICAL_PATH)
    ):
        raise ValueError("R3 preregistration V1 invalidation differs")
    return {
        "schema": PREREGISTRATION_SCHEMA, "complete": True, "create_only": True,
        "static_only": True, "namespace": NAMESPACE,
        "pending_config": {"path": os.fspath(source), "sha256": sha256_file(source), "schema": CONFIG_SCHEMA},
        "invalidated_predecessor": {
            "path": os.fspath(V1_PREREGISTRATION_PATH), "sha256": V1_PREREGISTRATION_SHA256,
            "schema": V1_PREREGISTRATION_SCHEMA,
        },
        "predecessor_invalidation": {
            "path": os.fspath(invalid_path), "sha256": sha256_file(invalid_path),
            "schema": V1_INVALID_SCHEMA,
        },
        "dynamic_ready_delta": {
            "only_replaced_fields": [
                "scene_contract.roles.outer_dev.source_success_receipt",
                "scene_contract.roles.inner_holdout2.source_success_receipt",
                "scene_contract.roles.inner_holdout3.source_success_receipt",
                "scene_contract.roles.inner_holdout4.source_success_receipt",
                "continuation_receipt", "run_authorization",
            ],
            "all_other_json_values_byte_semantics_frozen": True,
        },
        "canonical_dynamic_paths": {
            "ready_config": os.fspath(READY_CONFIG_PATH),
            "run_authorization": os.fspath(RUN_AUTHORIZATION_PATH),
            "continuation_receipt": os.fspath(CONTINUATION_CANONICAL_PATH),
            "outer_receipt_root": cfg["producer_contracts"]["outer"]["canonical_root"],
            "inner_receipt_root": cfg["producer_contracts"]["inner"]["canonical_root"],
        },
        "implementation": copy.deepcopy(cfg["implementation"]),
        "processed_point_inventory": inventory,
        "scene_contract": {
            "scene_count": 80, "fit_scene_count": 60, "outer_scene_count": 20,
            "roles": [{"role": role, "heldout_fold": ROLE_SPECS[role][1], "scenes": list(roles[role])} for role in ROLE_ORDER],
        },
        "producer_verification": {
            "outer_api": "verify_success_receipt", "inner_api": "verify_success_receipt",
            "all_four_required": True, "eval_pass_required": True,
            "legacy_60_or_v1_v4_accepted": False,
        },
        "authorization_boundary": {
            "gpu_or_namespace_before_ready": False, "fold1_access": False,
            "official_validation_access": False, "ground_truth_access": False,
            "scannet_weight_or_artifact_access": False,
        },
    }


def validate_preregistration(cfg: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path, value = read_json(PREREGISTRATION_PATH, "R3 static preregistration")
    expected = build_preregistration_payload(DEFAULT_CONFIG)
    if value != expected:
        raise ValueError("static preregistration or bound inputs drifted")
    if sha256_file(DEFAULT_CONFIG) != (value.get("pending_config") or {}).get("sha256"):
        raise ValueError("frozen pending config changed after preregistration")
    del cfg
    return path, value


def validate_static_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    if Path(path) != DEFAULT_CONFIG:
        raise ValueError("static preflight accepts only canonical pending config")
    source, cfg = _load_frozen_pending()
    roles = _validate_config_shape(cfg)
    for role in ROLE_ORDER:
        schema = OUTER_SCHEMA if role == "outer_dev" else INNER_SCHEMA
        if cfg["scene_contract"]["roles"][role].get("source_success_receipt") != _pending_record(schema):
            raise ValueError(f"{role}: pending receipt differs")
    if cfg.get("continuation_receipt") != _pending_record(CONTINUATION_SCHEMA) or cfg.get("run_authorization") != _pending_record(AUTH_SCHEMA):
        raise ValueError("dynamic records are not frozen pending records")
    prereg_path, prereg = validate_preregistration(cfg)
    return {
        "schema": STATIC_REPORT_SCHEMA, "status": "PASS_STATIC_PENDING",
        "config": {"path": os.fspath(source), "sha256": sha256_file(source)},
        "preregistration": {"path": os.fspath(prereg_path), "sha256": sha256_file(prereg_path)},
        "processed_point_inventory_sha256": prereg["processed_point_inventory"]["inventory_sha256"],
        "scene_count": 80, "role_scene_counts": {role: len(rows) for role, rows in roles.items()},
        "operational_authorized": False, "gpu_started": False, "output_created": False,
        "ground_truth_access": False, "fold1_access": False, "official_validation_access": False,
    }


def _load_bound_module(record: Mapping[str, Any], name: str) -> Any:
    path, _ = _file_record(record, name)
    if record.get("api") != "verify_success_receipt":
        raise ValueError(f"{name} API differs")
    module_name = f"_r3_bound_{name}_{record['sha256'][:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    if not callable(getattr(module, "verify_success_receipt", None)):
        raise ValueError(f"{name} lacks verify_success_receipt")
    return module


def _ready_delta(pending: Mapping[str, Any], ready: Mapping[str, Any]) -> dict[str, Any]:
    """Return six bound records iff nothing else changed from pending."""

    expected = copy.deepcopy(pending)
    bound: dict[str, Any] = {}
    for role in ROLE_ORDER:
        record = (ready.get("scene_contract") or {}).get("roles", {}).get(role, {}).get("source_success_receipt")
        if not isinstance(record, Mapping) or record.get("state") != "bound":
            raise PendingOperationalInputs(f"{role} success receipt is pending")
        bound[role] = copy.deepcopy(record)
        expected["scene_contract"]["roles"][role]["source_success_receipt"] = copy.deepcopy(record)
    for key in ("continuation_receipt", "run_authorization"):
        record = ready.get(key)
        if not isinstance(record, Mapping) or record.get("state") != "bound":
            raise PendingOperationalInputs(f"{key} is pending")
        bound[key] = copy.deepcopy(record)
        expected[key] = copy.deepcopy(record)
    if expected != ready:
        raise PermissionError("ready config differs from pending outside the six authorized replacements")
    return bound


def _projection(ready: Mapping[str, Any]) -> str:
    value = copy.deepcopy(ready)
    value["run_authorization"] = {
        "state": "bound", "path": "<redacted>", "sha256": "<redacted>", "schema": AUTH_SCHEMA,
    }
    return sha256_bytes(canonical_json(value))


def _verify_continuation(record: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path, value = _file_record(record, "official E961 eval continuation", schema=CONTINUATION_SCHEMA)
    if path != CONTINUATION_CANONICAL_PATH:
        raise ValueError("continuation path is not the canonical E961 V2 evaluation output")
    gate = value.get("continuation_gate") or {}
    checks = gate.get("checks") or {}
    required_checks = {
        "proposal_exact20_finite_ca_only",
        "same_gt_gain_ge_0_05_replacements_ge_10",
        "same_gt_gain_ge_0_05_scenes_ge_5",
        "oracle_delta_ap15_nonnegative", "oracle_delta_ap25_nonnegative",
        "oracle_delta_ap50_at_least_0_005",
    }
    if (
        value.get("complete") is not True or value.get("create_only") is not True
        or value.get("pass") is not True or gate.get("pass") is not True
        or set(checks) != required_checks or any(checks[key] is not True for key in required_checks)
        or value.get("authorized_roles") != list(ROLE_ORDER[1:])
        or gate.get("authorized_inner_roles") != list(ROLE_ORDER[1:])
        or value.get("scene_count") != 20 or value.get("checkpoint_selection") is not False
        or value.get("fold1_access") is not False or value.get("official_validation_access") is not False
    ):
        raise PermissionError("official E961 outer evaluation is not an exact PASS")
    for key in ("preregistration", "checkpoint_binding", "proposal_collection", "evaluation_report"):
        child = value.get(key) or {}
        _file_record(child, f"eval PASS {key}")
    return path, value


def _receipt_path(record: Mapping[str, Any], root: Path, role: str, schema: str) -> Path:
    if record.get("schema") == LEGACY_INNER_SCHEMA:
        raise ValueError(f"{role}: legacy 60-scene receipt is forbidden")
    path = _absolute(Path(str(record.get("path", ""))), f"{role} receipt")
    _under(path, root, f"{role} receipt")
    if role == "outer_dev":
        if path.name != "RUN_RECEIPT.json":
            raise ValueError("outer receipt canonical leaf differs")
    elif path.name != "RUN_RECEIPT.json" or path.parent.name != f"{ROLE_SPECS[role][2]}_{role}":
        raise ValueError(f"{role}: inner receipt canonical leaf differs")
    if record.get("schema") != schema:
        raise ValueError(f"{role}: receipt schema differs")
    return path


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


@dataclass
class ReadyContext:
    config_path: Path
    config: Mapping[str, Any]
    authorization_path: Path
    authorization_sha256: str
    continuation_path: Path
    continuation_sha256: str
    roles: Mapping[str, VerifiedRole]
    r2_context: Any
    writer_fd: int

    def close(self) -> None:
        if self.writer_fd >= 0:
            os.close(self.writer_fd)
            self.writer_fd = -1


def _extract_checkpoint(receipt: Mapping[str, Any], role: str) -> tuple[Path, str]:
    record = receipt.get("checkpoint") or {}
    if role == "outer_dev":
        record = (((receipt.get("terminal") or {}).get("checkpoint_audit") or {}).get("checkpoint") or {})
    path = _absolute(Path(str(record.get("path", ""))), f"{role} checkpoint")
    digest = _sha(record.get("sha256"), f"{role} checkpoint SHA256")
    if path.name != "iter_11268.pth" or sha256_file(path) != digest:
        raise ValueError(f"{role}: terminal iter11268 checkpoint differs")
    lowered = os.fspath(path).lower()
    if "scannet" in lowered or "xfit_r2_formal" in lowered:
        raise ValueError(f"{role}: old/ScanNet checkpoint path is forbidden")
    return path, digest


def verify_dynamic_inputs(
    ready: Mapping[str, Any], *, outer_module: Any | None = None, inner_module: Any | None = None,
) -> tuple[Path, dict[str, Any], dict[str, VerifiedRole]]:
    pending_source, pending = _load_frozen_pending()
    del pending_source
    bound = _ready_delta(pending, ready)
    _validate_config_shape(ready)
    validate_preregistration(pending)
    continuation_path, continuation = _verify_continuation(bound["continuation_receipt"])
    continuation_sha = sha256_file(continuation_path)
    producer = ready["producer_contracts"]
    outer_module = outer_module or _load_bound_module(producer["outer"]["verifier"], "official_outer_r2")
    inner_module = inner_module or _load_bound_module(producer["inner"]["verifier"], "official_inner_r2")
    verified: dict[str, VerifiedRole] = {}
    for role in ROLE_ORDER:
        schema = OUTER_SCHEMA if role == "outer_dev" else INNER_SCHEMA
        root = Path(producer["outer" if role == "outer_dev" else "inner"]["canonical_root"])
        record = bound[role]
        receipt_path = _receipt_path(record, root, role, schema)
        receipt_path2, receipt = _file_record(record, f"{role} producer success receipt", schema=schema)
        if receipt_path2 != receipt_path:
            raise ValueError(f"{role}: receipt path identity differs")
        official = (
            outer_module.verify_success_receipt(receipt_path)
            if role == "outer_dev" else inner_module.verify_success_receipt(receipt_path)
        )
        if isinstance(official, tuple):
            official = official[1]
        if official != receipt:
            raise ValueError(f"{role}: official verifier payload differs from stable receipt")
        train_folds, heldout, _ = ROLE_SPECS[role]
        protocol = receipt.get("training_protocol") or {}
        if role == "outer_dev":
            protocol = {
                "train_folds": [2, 3, 4], "heldout_fold": 0,
                "train_scenes": 1001, "heldout_scenes": 20,
                "initialization": "random_scratch_ca_only", "optimizer_updates": 11268,
                "checkpoint_selection": False,
            }
        if (
            receipt.get("role") != role or receipt.get("status") != "success"
            or receipt.get("exit_code") != 0
            or list(protocol.get("train_folds", ())) != list(train_folds)
            or int(protocol.get("heldout_fold", -1)) != heldout
            or int(protocol.get("train_scenes", -1)) != 1001
            or int(protocol.get("heldout_scenes", -1)) != 20
            or protocol.get("initialization") != "random_scratch_ca_only"
            or int(protocol.get("optimizer_updates", -1)) != 11268
            or protocol.get("checkpoint_selection") is not False
        ):
            raise ValueError(f"{role}: formal CA-only role semantics differ")
        if role == "outer_dev":
            claim = receipt.get("training_claim") or {}
            if claim != {
                "ca1m_training_data_only": True,
                "scannet_training_weights_loaded": False,
                "scannet_training_data_configured_or_opened": False,
                "plugin_imports_scannet_adapter_class_definition": True,
            }:
                raise ValueError("outer_dev: CA-only training claim differs")
        else:
            upstream = (receipt.get("passing_upstream") or {}).get("eval_v2_continuation_receipt") or {}
            if upstream.get("sha256") != continuation_sha:
                raise ValueError(f"{role}: inner receipt does not bind this eval PASS")
            access = receipt.get("access") or {}
            if access != {
                "ca1m_training_only": True, "fold0_gt_access": False,
                "fold1_access": False, "official_validation_access": False,
                "scannet_training_weights_loaded": False,
                "scannet_training_data_configured_or_opened": False,
            }:
                raise ValueError(f"{role}: CA-only access claim differs")
        checkpoint, checkpoint_sha = _extract_checkpoint(receipt, role)
        scenes = _scene_list(ready["scene_contract"]["roles"][role]["predict_scene_list"], f"{role} predict list", 20)
        verified[role] = VerifiedRole(
            role, train_folds, heldout, scenes, receipt_path, sha256_file(receipt_path),
            receipt, checkpoint, checkpoint_sha,
        )
    return continuation_path, continuation, verified


def build_authorization_payload(
    ready_without_auth: Mapping[str, Any], continuation_path: Path,
    verified: Mapping[str, VerifiedRole],
) -> dict[str, Any]:
    pending_sha = sha256_file(DEFAULT_CONFIG)
    prereg_sha = sha256_file(PREREGISTRATION_PATH)
    continuation_sha = sha256_file(continuation_path)
    value = copy.deepcopy(ready_without_auth)
    value["run_authorization"] = {
        "state": "bound", "path": os.fspath(RUN_AUTHORIZATION_PATH),
        "sha256": "<authorization-sha256-not-part-of-projection>", "schema": AUTH_SCHEMA,
    }
    projection_value = copy.deepcopy(value)
    projection_value["run_authorization"] = {
        "state": "bound", "path": "<redacted>", "sha256": "<redacted>", "schema": AUTH_SCHEMA,
    }
    return {
        "schema": AUTH_SCHEMA, "complete": True, "create_only": True,
        "namespace": NAMESPACE, "pending_config_sha256": pending_sha,
        "preregistration": {"path": os.fspath(PREREGISTRATION_PATH), "sha256": prereg_sha, "schema": PREREGISTRATION_SCHEMA},
        "ready_config_projection_sha256": sha256_bytes(canonical_json(projection_value)),
        "producer_deep_verifiers_passed": True,
        "eval_pass_verified": True,
        "roles": [{
            "role": role, "receipt_path": os.fspath(verified[role].receipt_path),
            "receipt_sha256": verified[role].receipt_sha256,
            "checkpoint_sha256": verified[role].checkpoint_sha256,
        } for role in ROLE_ORDER],
        "continuation_receipt": {
            "path": os.fspath(continuation_path), "sha256": continuation_sha,
            "schema": CONTINUATION_SCHEMA,
        },
        "authorizations": copy.deepcopy(AUTHORIZATIONS),
        "ground_truth_access": False, "fold1_access": False,
        "official_validation_access": False, "scannet_weight_or_artifact_access": False,
        "formal_gpu_run_started": False,
    }


def _validate_authorization(
    ready: Mapping[str, Any], auth_path: Path, auth: Mapping[str, Any],
    continuation_path: Path, verified: Mapping[str, VerifiedRole],
) -> None:
    expected = build_authorization_payload(ready, continuation_path, verified)
    # build_authorization_payload always redacts run_authorization for projection,
    # so it is valid for a fully bound ready config as well.
    if auth != expected:
        raise PermissionError("run authorization does not exactly bind ready/producers/eval PASS")
    if auth_path != RUN_AUTHORIZATION_PATH:
        raise PermissionError("run authorization is noncanonical")


def _mkdirat_chain(path: Path, name: str) -> tuple[int, tuple[tuple[int, int], ...]]:
    path = _absolute(path, name)
    fd = os.open(path.anchor, _DIR_FLAGS)
    identities: list[tuple[int, int]] = []
    try:
        value = os.fstat(fd); identities.append((value.st_dev, value.st_ino))
        for part in path.parts[1:]:
            try:
                next_fd = os.open(part, _DIR_FLAGS, dir_fd=fd)
            except FileNotFoundError:
                os.mkdir(part, 0o700, dir_fd=fd)
                next_fd = os.open(part, _DIR_FLAGS, dir_fd=fd)
                os.fsync(fd)
            os.close(fd); fd = next_fd
            value = os.fstat(fd); identities.append((value.st_dev, value.st_ino))
        return fd, tuple(identities)
    except Exception:
        os.close(fd)
        raise


def ensure_directory(path: Path) -> Path:
    fd, identities = _mkdirat_chain(Path(path), "R3 output directory")
    os.close(fd)
    _verify_dir_chain(Path(path), identities, "R3 output directory")
    return Path(path)


def _host_target_probe(parent: Path) -> None:
    """Prove create/fsync/hardlink/unlink semantics in a dedicated temp dir."""

    parent = _absolute(parent, "host target parent")
    parent_fd, parent_ids = _open_dir_chain(parent, "host target parent")
    probe = f".r3_probe_{os.getpid()}_{secrets.token_hex(12)}"
    probe_fd = -1
    try:
        os.mkdir(probe, 0o700, dir_fd=parent_fd)
        probe_fd = os.open(probe, _DIR_FLAGS, dir_fd=parent_fd)
        source_fd = os.open(
            "source.tmp", os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o444, dir_fd=probe_fd,
        )
        try:
            payload = secrets.token_bytes(32)
            if os.write(source_fd, payload) != len(payload):
                raise OSError("host probe short write")
            os.fsync(source_fd)
        finally:
            os.close(source_fd)
        os.link("source.tmp", "published", src_dir_fd=probe_fd, dst_dir_fd=probe_fd, follow_symlinks=False)
        os.fsync(probe_fd)
        left = os.stat("source.tmp", dir_fd=probe_fd, follow_symlinks=False)
        right = os.stat("published", dir_fd=probe_fd, follow_symlinks=False)
        if (left.st_dev, left.st_ino, left.st_nlink) != (right.st_dev, right.st_ino, right.st_nlink) or left.st_nlink != 2:
            raise OSError("host probe hardlink identity differs")
        os.unlink("published", dir_fd=probe_fd)
        os.unlink("source.tmp", dir_fd=probe_fd)
        os.fsync(probe_fd)
        os.close(probe_fd); probe_fd = -1
        os.rmdir(probe, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except Exception:
        # Safe cleanup is restricted to this unpredictable dedicated directory.
        if probe_fd >= 0:
            for name in ("published", "source.tmp"):
                try:
                    os.unlink(name, dir_fd=probe_fd)
                except FileNotFoundError:
                    pass
            os.close(probe_fd)
        try:
            os.rmdir(probe, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(parent_fd)
    _verify_dir_chain(parent, parent_ids, "host target parent")


def _exclusive_bytes(path: Path, data: bytes) -> Path:
    """Create-only FUSE-compatible publication without chmod/fchmod."""

    path = _absolute(path, "create-only output")
    parent_fd, parent_ids = _open_dir_chain(path.parent, "create-only output parent")
    temp = f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(12)}"
    fd = -1
    linked = False
    try:
        fd = os.open(
            temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o444, dir_fd=parent_fd,
        )
        view = memoryview(data)
        while view:
            count = os.write(fd, view)
            if count < 1:
                raise OSError("short create-only write")
            view = view[count:]
        os.fsync(fd); os.close(fd); fd = -1
        os.link(temp, path.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd, follow_symlinks=False)
        linked = True
        os.fsync(parent_fd)
        os.unlink(temp, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except Exception:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(temp, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(parent_fd)
    _verify_dir_chain(path.parent, parent_ids, "create-only output parent")
    if not linked or stable_bytes(path, "create-only published output", nonempty=False) != data:
        raise OSError("published output differs")
    return path


def write_bytes_exclusive(path: Path, data: bytes) -> Path:
    ensure_directory(Path(path).parent)
    return _exclusive_bytes(Path(path), data)


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> Path:
    return write_bytes_exclusive(path, canonical_json(value))


def create_or_verify(path: Path, data: bytes, name: str) -> Path:
    path = Path(path)
    try:
        current = stable_bytes(path, name, nonempty=False)
    except FileNotFoundError:
        return write_bytes_exclusive(path, data)
    if current != data:
        raise ValueError(f"{name} resume bytes differ")
    return path


def _claim_writer(parent: Path, leaf: str, authorization_sha: str) -> int:
    payload = canonical_json({
        "schema": "boxfusion.ca1m_tr3d_e961_single_writer_claim.v5.r3",
        "complete": True, "create_only": True, "namespace": NAMESPACE,
        "authorization_sha256": authorization_sha,
    })
    path = parent / leaf
    try:
        _exclusive_bytes(path, payload)
    except FileExistsError:
        if stable_bytes(path, "single-writer claim") != payload:
            raise PermissionError("single-writer claim belongs to another authorization")
    parent_fd, _ = _open_dir_chain(parent, "writer claim parent")
    try:
        fd = os.open(path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except Exception:
        os.close(fd)
        raise PermissionError("another R3 writer currently owns the namespace")
    return fd


def seal_preregistration() -> Path:
    payload = build_preregistration_payload(DEFAULT_CONFIG)
    _host_target_probe(MANIFEST_ROOT.parent)
    ensure_directory(MANIFEST_ROOT)
    claim_fd = _claim_writer(MANIFEST_ROOT, ".PREREGISTRATION_V2.writer.claim", sha256_bytes(canonical_json(payload)))
    try:
        return _exclusive_bytes(PREREGISTRATION_PATH, canonical_json(payload))
    finally:
        os.close(claim_fd)


def seal_preregistration_v1_invalidation() -> Path:
    """Create-only evidence invalidating the never-authorized V1 static seal."""

    if sha256_file(V1_PREREGISTRATION_PATH) != V1_PREREGISTRATION_SHA256:
        raise ValueError("R3 V1 preregistration predecessor changed")
    _, predecessor = read_json(V1_PREREGISTRATION_PATH, "R3 V1 preregistration predecessor")
    if predecessor.get("schema") != V1_PREREGISTRATION_SCHEMA:
        raise ValueError("R3 V1 preregistration predecessor schema differs")
    for forbidden in (READY_CONFIG_PATH, RUN_AUTHORIZATION_PATH, Path("/extra/ZhaoX") / NAMESPACE):
        if forbidden.exists() or forbidden.is_symlink():
            raise PermissionError("cannot invalidate V1 after operational output exists")
    payload = {
        "schema": V1_INVALID_SCHEMA, "complete": True, "create_only": True,
        "invalid": True, "operational_authority": False,
        "predecessor": {
            "path": os.fspath(V1_PREREGISTRATION_PATH),
            "sha256": V1_PREREGISTRATION_SHA256,
            "schema": V1_PREREGISTRATION_SCHEMA,
        },
        "reason": "canonical_eval_continuation_path_was_incorrect",
        "incorrect_continuation_path": os.fspath(
            ROOT / "manifests/ca1m_tr3d_e961_outer_dev_eval_v1/CONTINUATION.json"
        ),
        "corrected_continuation_path": os.fspath(CONTINUATION_CANONICAL_PATH),
        "ready_config_created": False, "run_authorization_created": False,
        "runtime_namespace_created": False, "gpu_started": False,
        "ground_truth_access": False,
    }
    _host_target_probe(MANIFEST_ROOT)
    claim_fd = _claim_writer(MANIFEST_ROOT, ".PREREGISTRATION_INVALID.writer.claim", V1_PREREGISTRATION_SHA256)
    try:
        return _exclusive_bytes(V1_INVALID_PATH, canonical_json(payload))
    finally:
        os.close(claim_fd)


def _bound_input(path: Path, schema: str, name: str) -> dict[str, Any]:
    path = _absolute(Path(path), name)
    _, value = read_json(path, name)
    if value.get("schema") != schema:
        raise ValueError(f"{name} schema differs")
    return {"state": "bound", "path": os.fspath(path), "sha256": sha256_file(path), "schema": schema}


def seal_ready_and_authorization(
    *, outer_receipt: Path, inner_holdout2_receipt: Path,
    inner_holdout3_receipt: Path, inner_holdout4_receipt: Path,
    continuation_receipt: Path,
) -> tuple[Path, Path]:
    """Deep-verify all producers/eval and create the canonical auth+ready pair."""

    validate_static_config(DEFAULT_CONFIG)
    _, pending = _load_frozen_pending()
    ready = copy.deepcopy(pending)
    supplied = {
        "outer_dev": outer_receipt, "inner_holdout2": inner_holdout2_receipt,
        "inner_holdout3": inner_holdout3_receipt, "inner_holdout4": inner_holdout4_receipt,
    }
    for role, path in supplied.items():
        schema = OUTER_SCHEMA if role == "outer_dev" else INNER_SCHEMA
        ready["scene_contract"]["roles"][role]["source_success_receipt"] = _bound_input(path, schema, f"{role} success receipt")
    ready["continuation_receipt"] = _bound_input(
        continuation_receipt, CONTINUATION_SCHEMA, "official E961 eval continuation",
    )
    # A provisional redacted bound record permits the exact-six-delta check;
    # it is never published or accepted operationally.
    ready["run_authorization"] = {
        "state": "bound", "path": os.fspath(RUN_AUTHORIZATION_PATH),
        "sha256": "0" * 64, "schema": AUTH_SCHEMA,
    }
    continuation_path, _, verified = verify_dynamic_inputs(ready)
    auth = build_authorization_payload(ready, continuation_path, verified)
    auth_bytes = canonical_json(auth)
    ready["run_authorization"] = {
        "state": "bound", "path": os.fspath(RUN_AUTHORIZATION_PATH),
        "sha256": sha256_bytes(auth_bytes), "schema": AUTH_SCHEMA,
    }
    # Prove that the final ready differs in exactly the same six fields and
    # that authorization projection is unchanged by inserting its own digest.
    _ready_delta(pending, ready)
    if auth != build_authorization_payload(ready, continuation_path, verified):
        raise RuntimeError("authorization projection is not self-consistent")

    _host_target_probe(MANIFEST_ROOT)
    claim_fd = _claim_writer(MANIFEST_ROOT, ".READY_SEAL.writer.claim", sha256_bytes(auth_bytes))
    try:
        _exclusive_bytes(RUN_AUTHORIZATION_PATH, auth_bytes)
        _exclusive_bytes(READY_CONFIG_PATH, canonical_json(ready))
    finally:
        os.close(claim_fd)
    return READY_CONFIG_PATH, RUN_AUTHORIZATION_PATH


def _import_r2(cfg: Mapping[str, Any]) -> Any:
    record = cfg["implementation"]["r2_execution_core"]
    path, _ = _file_record(record, "frozen R2 execution core")
    spec = importlib.util.spec_from_file_location("_r3_frozen_r2_execution", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load frozen R2 execution core")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _r2_config(ready: Mapping[str, Any]) -> dict[str, Any]:
    static, runtime = ready["static_inputs"], ready["runtime"]
    return {
        "namespace": NAMESPACE,
        "run_authorization": copy.deepcopy(ready["run_authorization"]),
        "continuation_receipt": copy.deepcopy(ready["continuation_receipt"]),
        "scene_contract": copy.deepcopy(ready["scene_contract"]),
        "candidate_inputs": {
            "processed_rgbd_root": static["processed_point_inputs"]["root"],
            "point_inference_config": copy.deepcopy(runtime["point_inference_config"]),
            "protocol": copy.deepcopy(runtime["protocol"]),
        },
        "anchor_inputs": {
            "final_base_collection": copy.deepcopy(static["final_base_collection"]),
            "final_base_prediction_root": static["final_base_prediction_root"],
            "native_b6_collection": copy.deepcopy(static["native_b6_collection"]),
            "native_b6_receipt_root": static["native_b6_receipt_root"],
            "native_b6_diagnostic_root": static["native_b6_diagnostic_root"],
            "native_b6_oof_sidecar_manifest": copy.deepcopy(static["native_b6_oof_manifest"]),
            "native_b6_oof_sidecar": copy.deepcopy(static["native_b6_oof_sidecar"]),
        },
        "runtime": {
            key: copy.deepcopy(runtime[key]) for key in (
                "worker_python", "worker_script", "runtime_root", "project_root",
                "vendor_root", "startup_timeout_s",
            )
        },
        "outputs": copy.deepcopy(ready["outputs"]),
        "access": copy.deepcopy(ready["access"]),
    }


def validate_operational_ready(path: Path | None = None) -> ReadyContext:
    """Return authority only after deep producer/eval/static/host verification."""

    if path is None:
        path = READY_CONFIG_PATH if READY_CONFIG_PATH.exists() and not READY_CONFIG_PATH.is_symlink() else DEFAULT_CONFIG
    path = Path(path)
    if path == DEFAULT_CONFIG:
        # This is intentionally the first branch: no prereg, receipt,
        # checkpoint, host target, device, or output namespace is touched.
        raise PendingOperationalInputs("formal E961 receipts/run authorization are pending")
    if path != READY_CONFIG_PATH:
        raise PermissionError("operational config path is not the canonical sealed ready config")
    source, ready = read_json(path, "canonical R3 ready config")
    continuation_path, _, verified = verify_dynamic_inputs(ready)
    auth_path, auth = _file_record(ready["run_authorization"], "R3 run authorization", schema=AUTH_SCHEMA)
    _validate_authorization(ready, auth_path, auth or {}, continuation_path, verified)

    output_root = Path(ready["outputs"]["namespace_root"])
    output_parent = output_root.parent
    # Required host capability probe precedes claim, mkdir, namespace output,
    # worker construction, and CUDA use.
    _host_target_probe(output_parent)
    writer_fd = _claim_writer(output_parent, f".{NAMESPACE}.writer.claim", sha256_file(auth_path))
    try:
        r2 = _import_r2(ready)
        r2_roles = {
            role: r2.VerifiedRole(
                role, verified[role].train_folds, verified[role].heldout_fold,
                verified[role].scenes, verified[role].receipt_path,
                verified[role].receipt_sha256, verified[role].receipt,
                verified[role].checkpoint_path, verified[role].checkpoint_sha256,
            ) for role in ROLE_ORDER
        }
        r2_ctx = r2.ReadyContext(
            source, _r2_config(ready), auth_path, sha256_file(auth_path),
            continuation_path, sha256_file(continuation_path), r2_roles,
        )
        return ReadyContext(
            source, ready, auth_path, sha256_file(auth_path), continuation_path,
            sha256_file(continuation_path), verified, r2_ctx, writer_fd,
        )
    except Exception:
        os.close(writer_fd)
        raise


def _pre_operation_guard(ctx: ReadyContext) -> None:
    if ctx.config_path != READY_CONFIG_PATH or sha256_file(ctx.authorization_path) != ctx.authorization_sha256:
        raise PermissionError("R3 operational context authorization drifted")
    output_root = Path(ctx.config["outputs"]["namespace_root"])
    _under(output_root, Path("/extra/ZhaoX"), "runtime namespace")
    parent_fd, identities = _open_dir_chain(output_root.parent, "runtime namespace parent")
    os.close(parent_fd)
    _verify_dir_chain(output_root.parent, identities, "runtime namespace parent")


@contextmanager
def _r2_secure_publication(ctx: ReadyContext) -> Iterator[Any]:
    r2 = sys.modules.get("_r3_frozen_r2_execution") or _import_r2(ctx.config)
    originals = {name: getattr(r2, name) for name in ("write_bytes_exclusive", "ensure_directory")}
    r2.write_bytes_exclusive = write_bytes_exclusive
    r2.ensure_directory = ensure_directory
    try:
        yield r2
    finally:
        for name, value in originals.items():
            setattr(r2, name, value)


def run_stage_p(ctx: ReadyContext, role: str, *, device: str = "cuda:0", **kwargs: Any) -> dict[str, Any]:
    _pre_operation_guard(ctx)
    with _r2_secure_publication(ctx) as r2:
        return r2.run_stage_p(ctx.r2_context, role, device=device, **kwargs)


def run_stage_o(ctx: ReadyContext, role: str, **kwargs: Any) -> dict[str, Any]:
    _pre_operation_guard(ctx)
    with _r2_secure_publication(ctx) as r2:
        return r2.run_stage_o(ctx.r2_context, role, **kwargs)


def run_stage_e(ctx: ReadyContext, role: str, **kwargs: Any) -> dict[str, Any]:
    _pre_operation_guard(ctx)
    with _r2_secure_publication(ctx) as r2:
        return r2.run_stage_e(ctx.r2_context, role, **kwargs)


def seal_stage_m(ctx: ReadyContext) -> dict[str, Any]:
    _pre_operation_guard(ctx)
    with _r2_secure_publication(ctx) as r2:
        return r2.seal_stage_m(ctx.r2_context)


def run_all(ctx: ReadyContext, *, device: str = "cuda:0") -> dict[str, Any]:
    results: dict[str, Any] = {}
    for role in ROLE_ORDER:
        results[f"P:{role}"] = run_stage_p(ctx, role, device=device)
        results[f"O:{role}"] = run_stage_o(ctx, role)
        results[f"E:{role}"] = run_stage_e(ctx, role)
    results["M"] = seal_stage_m(ctx)
    return {
        "schema": "boxfusion.ca1m_tr3d_e961_terminal_inputs_run.v5.r3",
        "complete": True, "r2_execution_core_reused": True,
        "ground_truth_access": False, "results": results,
    }


__all__ = [
    "DEFAULT_CONFIG", "PREREGISTRATION_PATH", "READY_CONFIG_PATH", "RUN_AUTHORIZATION_PATH",
    "CONFIG_SCHEMA", "PREREGISTRATION_SCHEMA", "AUTH_SCHEMA", "NAMESPACE", "OUTER_SCHEMA",
    "INNER_SCHEMA", "LEGACY_INNER_SCHEMA", "CONTINUATION_SCHEMA", "ROLE_ORDER", "ROLE_SPECS",
    "PendingOperationalInputs", "VerifiedRole", "ReadyContext", "stable_bytes", "sha256_file",
    "processed_point_inventory", "build_preregistration_payload", "validate_static_config",
    "validate_operational_ready", "seal_preregistration", "seal_ready_and_authorization",
    "ensure_directory", "write_bytes_exclusive", "write_json_exclusive", "create_or_verify",
    "run_stage_p", "run_stage_o", "run_stage_e", "seal_stage_m", "run_all",
]
