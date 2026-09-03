#!/usr/bin/env python3
"""Create-only F4 FastSAM/F2 -> frozen BoxerNet geometry shadow replay.

This program is deliberately an observer.  It authenticates the sealed F0/F2
receipts, opens only each scheduled current frame, and asks a frozen BoxerNet
provider for one ``HB`` geometry per existing FastSAM source.  It never opens
ground truth, native detector output, an evaluator, or any future frame.

The Boxer dependency is imported only inside :func:`run_f4`, which keeps plan
and unit-test paths independent of CUDA and permits a fail-closed fake provider
to be injected by tests.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))
PROTOCOL_ID = "F4-FASTSAM-BOXER-GEOMETRY-SHADOW-PAPER100"
SCENE_SCHEMA = "boxfusion.scannet_fastsam_f4_boxer_paper100.scene.v1"
SHARD_SCHEMA = "boxfusion.scannet_fastsam_f4_boxer_paper100.shard.v1"

EXPECTED_F2_SCHEMA = "boxfusion.scannet_fastsam_f2_paper100.merge.v1"
EXPECTED_F2_PROTOCOL = "F2-DFU-LGF-lite-shadow-paper100"
EXPECTED_F2_SCENE_SCHEMA = "boxfusion.scannet_fastsam_f2_paper100.scene.v1"
EXPECTED_F0_SCHEMA = "boxfusion.scannet_fastsam_f0_full200.merge.v1"
EXPECTED_F0_PROTOCOL = "F0-frozen-FastSAM-x-residual-automatic-mask-shadow-full200"
EXPECTED_F0_SCENE_SCHEMA = "boxfusion.scannet_fastsam_f0_full200.scene.v1"

EXPECTED_SCENES = 100
EXPECTED_KEYFRAMES = 6_817
EXPECTED_SUCCESSFUL_FRAMES = 6_726
EXPECTED_SOURCES = 52_299
EXPECTED_SHARDS = 2
EXPECTED_SHARD_COUNTS: Mapping[int, Mapping[str, int]] = {
    0: {"keyframe_count": 3_259, "successful_frame_count": 3_189, "source_count": 24_863},
    1: {"keyframe_count": 3_558, "successful_frame_count": 3_537, "source_count": 27_436},
}

EXPECTED_SCENE_LIST_SHA256 = (
    "4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5"
)
EXPECTED_F0_RECEIPT_SHA256 = (
    "07249ead31ad150cb43d7a35f4c922ac70a8a2f95bcf0fcd24f61f944c1e58a1"
)
EXPECTED_F2_RECEIPT_SHA256 = (
    "455c0e36e35a30c7ba5915384e4d159a730a47b3368bf4b3fb6a5f6064f25603"
)
EXPECTED_BOXER_CHECKPOINT_SHA256 = (
    "d5a30b348a8f5b0e5990ff3aa0e8f473ce77d860da22586322e7f47abc83ca6f"
)
EXPECTED_DINO_CHECKPOINT_SHA256 = (
    "4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea"
)
EXPECTED_BOXERNET_SOURCE_SHA256 = (
    "a8009c1c0932aaab98bb074a2a4c50e55a3fbdfc3c6cb1afc9e1aef0e5324130"
)
EXPECTED_ADAPTER_SOURCE_SHA256 = (
    "3e82d49512de4abe61d033c2cca903993a83587d2ea56080ff71e42c2c7372a4"
)
EXPECTED_BOXER_COMMIT = "1f86542dc342a4b1d474c87c97c5d1d6566d9148"

MAX_SOURCES_PER_FRAME = 16
WARMUP_FORWARD_COUNT = 3
SOURCE_FRAME_STRIDE = 25.0

DEFAULT_F2_RECEIPT = (
    REPOSITORY_ROOT
    / "logs/scannet_fastsam_f2_paper100_score05/final/F2_FASTSAM_PAPER100.json"
)
DEFAULT_F0_RECEIPT = (
    REPOSITORY_ROOT
    / "logs/scannet_fastsam_f0_full200_score05/final/F0_FASTSAM_FULL200.json"
)
DEFAULT_SCENE_LIST = REPOSITORY_ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt"
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT / "logs/scannet_fastsam_f4_boxer_paper100_score05"
)
PROTOCOL_PATH = REPOSITORY_ROOT / "docs/F4_FASTSAM_BOXER_GEOMETRY_PROTOCOL_FREEZE.md"

CONTRACTS: Mapping[str, bool] = {
    "shadow_only": True,
    "birth_enabled": False,
    "native_output_mutation": False,
    "gt_access": False,
    "prediction_access": False,
    "evaluator_access": False,
    "future_frame_access": False,
    "training": False,
    "online_learning": False,
}


class F4RunnerError(RuntimeError):
    """Raised when a frozen input, provider result, or protocol differs."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _regular_file(path: Path, label: str, suffix: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise F4RunnerError(f"{label} must be a regular non-symlink file: {path}")
    result = path.resolve()
    if suffix is not None and result.suffix.lower() != suffix:
        raise F4RunnerError(f"{label} must be a {suffix} file: {result}")
    if result.suffix.lower() in {".pkl", ".pickle"}:
        raise F4RunnerError(f"prediction pickle input is forbidden: {result}")
    return result


def _read_json(path: Path, label: str) -> tuple[Path, dict[str, Any]]:
    source = _regular_file(path, label, ".json")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise F4RunnerError(f"invalid {label}: {source}") from error
    if not isinstance(value, dict):
        raise F4RunnerError(f"{label} must contain one JSON object: {source}")
    return source, value


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise F4RunnerError(f"refusing to overwrite output: {path}") from error
        return _sha256(path)
    finally:
        temporary.unlink(missing_ok=True)


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise F4RunnerError("runtime sample must be finite and non-negative")
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _jsonable(value: object, label: str) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    elif not isinstance(value, (Mapping, list, tuple, str, int, float, bool, type(None))):
        if hasattr(value, "__dict__"):
            value = vars(value)
        else:
            raise F4RunnerError(f"{label} is not JSON serializable")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item, f"{label}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, f"{label}[]") for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise F4RunnerError(f"{label} contains a non-finite number")
    return value


def _mapping(value: object, label: str) -> dict[str, Any]:
    result = _jsonable(value, label)
    if not isinstance(result, dict):
        raise F4RunnerError(f"{label} must be a mapping-like object")
    return result


def _field(value: object, name: str, default: object = None) -> object:
    """Read one duck-typed provider field without serializing sibling fields."""

    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _diagnostic_jsonable(value: object) -> Any:
    """JSON-normalize diagnostics, replacing non-finite values with ``None``.

    Invalid Boxer rows are required to abstain.  Consequently an invalid NaN
    in an optional confidence/log-variance diagnostic must not make the entire
    receipt unwritable, nor may it become a usable geometry.
    """

    if is_dataclass(value):
        value = {field: getattr(value, field) for field in value.__dataclass_fields__}
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    elif not isinstance(value, (Mapping, list, tuple, str, int, float, bool, type(None))):
        value = vars(value) if hasattr(value, "__dict__") else str(value)
    if isinstance(value, Mapping):
        return {str(key): _diagnostic_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_diagnostic_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _optional_diagnostic(value: object) -> Any:
    normalized = _diagnostic_jsonable(value)

    def contains_none(item: object) -> bool:
        if item is None:
            return True
        if isinstance(item, Mapping):
            return any(contains_none(child) for child in item.values())
        if isinstance(item, list):
            return any(contains_none(child) for child in item)
        return False

    return None if contains_none(normalized) else normalized


def _number(value: object, label: str, *, default: float | None = None) -> float:
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise F4RunnerError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise F4RunnerError(f"{label} must be finite and non-negative")
    return result


def _rehash_reference(value: object, label: str, suffix: str | None = None) -> Path:
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("path"), str)
        or not isinstance(value.get("sha256"), str)
    ):
        raise F4RunnerError(f"{label} seal is absent")
    path = _regular_file(Path(value["path"]), label, suffix)
    if _sha256(path) != value["sha256"]:
        raise F4RunnerError(f"{label} rehash differs")
    return path


def _seal(path: Path, label: str, suffix: str | None = None) -> dict[str, str]:
    source = _regular_file(path, label, suffix)
    return {"path": os.fspath(source), "sha256": _sha256(source)}


def _load_intrinsic(reference: Mapping[str, Any]) -> tuple[Path, np.ndarray]:
    path = _rehash_reference(reference, "sealed depth intrinsic", ".txt")
    try:
        matrix = np.loadtxt(path, dtype=np.float64)
    except (OSError, ValueError) as error:
        raise F4RunnerError("sealed depth intrinsic cannot be decoded") from error
    if matrix.shape == (4, 4):
        matrix = matrix[:3, :3]
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or matrix[0, 0] <= 0.0
        or matrix[1, 1] <= 0.0
    ):
        raise F4RunnerError("sealed depth intrinsic is invalid")
    return path, np.ascontiguousarray(matrix, dtype=np.float32)


def _default_frame_loader(
    rgb_path: Path,
    depth_path: Path,
    pose_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Importing OpenCV here keeps plan-only and receipt-only paths lightweight.
    try:
        import cv2  # type: ignore
    except ImportError as error:  # pragma: no cover - production dependency
        raise F4RunnerError("OpenCV is required for F4 production replay") from error
    bgr = cv2.imread(os.fspath(rgb_path), cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(os.fspath(depth_path), cv2.IMREAD_UNCHANGED)
    if bgr is None or depth_raw is None:
        raise F4RunnerError("current-frame RGB/depth cannot be decoded")
    if bgr.shape[:2] != (480, 640):
        bgr = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if depth_raw.shape != (480, 640) or depth_raw.ndim != 2:
        raise F4RunnerError("sealed metric depth must be 640x480")
    if not np.issubdtype(depth_raw.dtype, np.integer):
        raise F4RunnerError("sealed ScanNet depth must use integer millimetres")
    depth_m = depth_raw.astype(np.float32) / 1000.0
    try:
        pose = np.loadtxt(pose_path, dtype=np.float64)
    except (OSError, ValueError) as error:
        raise F4RunnerError("sealed current pose cannot be decoded") from error
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise F4RunnerError("successful F4 frame requires a finite 4x4 pose")
    return (
        np.ascontiguousarray(rgb, dtype=np.uint8),
        np.ascontiguousarray(depth_m, dtype=np.float32),
        np.ascontiguousarray(pose, dtype=np.float64),
    )


def _source_receipts(core_module: object | None, injected_factory: object | None) -> dict[str, Any]:
    paths: dict[str, Path] = {
        "runner": Path(__file__).resolve(),
        "protocol": PROTOCOL_PATH.resolve(),
    }
    if core_module is not None and getattr(core_module, "__file__", None):
        paths["core"] = Path(getattr(core_module, "__file__")).resolve()
    result: dict[str, Any] = {
        key: _seal(path, f"F4 {key} source") for key, path in paths.items()
    }
    if injected_factory is not None:
        result["provider_factory"] = {
            "injected": True,
            "module": getattr(injected_factory, "__module__", type(injected_factory).__module__),
            "qualname": getattr(injected_factory, "__qualname__", type(injected_factory).__qualname__),
        }
    return result


def _provider_frozen_receipts(provider: object, *, production: bool) -> dict[str, Any]:
    value = getattr(provider, "frozen_receipts", None)
    if callable(value):
        value = value()
    if value is None:
        value = getattr(provider, "model_receipts", None)
        if callable(value):
            value = value()
    if value is None:
        if production:
            raise F4RunnerError("production provider lacks frozen model receipts")
        return {"injected_test_provider": True}
    receipts = _mapping(value, "provider frozen receipts")

    # Rehash every recursively exposed path/SHA pair.  This supports both the
    # public dataclass and small duck-typed test providers without weakening
    # the production hash pins.
    seen_hashes: set[str] = set()

    def walk(item: object, label: str) -> None:
        if isinstance(item, Mapping):
            path = item.get("path")
            sha = item.get("sha256")
            if isinstance(path, str) and isinstance(sha, str):
                source = _regular_file(Path(path), label)
                if _sha256(source) != sha:
                    raise F4RunnerError(f"{label} rehash differs")
                seen_hashes.add(sha)
            for key, child in item.items():
                walk(child, f"{label}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{label}[{index}]")

    walk(receipts, "provider frozen receipt")
    if production:
        serialized = json.dumps(receipts, sort_keys=True)
        required_hashes = {
            EXPECTED_BOXER_CHECKPOINT_SHA256,
            EXPECTED_DINO_CHECKPOINT_SHA256,
            EXPECTED_BOXERNET_SOURCE_SHA256,
            EXPECTED_ADAPTER_SOURCE_SHA256,
        }
        if not required_hashes.issubset(seen_hashes) or EXPECTED_BOXER_COMMIT not in serialized:
            raise F4RunnerError("production frozen Boxer/DINO/source pins differ")
        clean_values = [
            value
            for key, value in _flatten_items(receipts)
            if key.lower().endswith(("clean", "worktree_clean"))
        ]
        if clean_values and not all(value is True for value in clean_values):
            raise F4RunnerError("production Boxer worktree is not clean")
    return receipts


def _flatten_items(value: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    result: list[tuple[str, Any]] = []
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            result.extend(_flatten_items(item, name))
        else:
            result.append((name, item))
    return result


def _load_inputs(
    f2_receipt_path: Path,
    f0_receipt_path: Path,
    scene_list_path: Path,
    *,
    expected_scene_count: int,
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...], tuple[dict[str, Any], ...], dict[str, dict[str, Any]]]:
    production = expected_scene_count == EXPECTED_SCENES
    f2_path, f2 = _read_json(f2_receipt_path, "sealed F2 merged receipt")
    f0_path, f0 = _read_json(f0_receipt_path, "sealed F0 merged receipt")
    f2_sha = _sha256(f2_path)
    f0_sha = _sha256(f0_path)
    if production and f2_sha != EXPECTED_F2_RECEIPT_SHA256:
        raise F4RunnerError("sealed production F2 receipt SHA-256 differs")
    if production and f0_sha != EXPECTED_F0_RECEIPT_SHA256:
        raise F4RunnerError("sealed production F0 receipt SHA-256 differs")

    coverage = f2.get("coverage")
    f2_rows = f2.get("scenes")
    if (
        f2.get("schema") != EXPECTED_F2_SCHEMA
        or f2.get("protocol_id") != EXPECTED_F2_PROTOCOL
        or f2.get("complete") is not True
        or f2.get("overall_pass") is not True
        or not isinstance(coverage, Mapping)
        or not isinstance(f2_rows, list)
        or len(f2_rows) != expected_scene_count
    ):
        raise F4RunnerError("sealed F2 merged receipt contract differs")
    scenes = tuple(str(scene) for scene in coverage.get("scene_order", ()))
    if (
        len(scenes) != expected_scene_count
        or len(set(scenes)) != len(scenes)
        or [row.get("scene_id") for row in f2_rows] != list(scenes)
        or [row.get("scene_index") for row in f2_rows] != list(range(expected_scene_count))
    ):
        raise F4RunnerError("sealed F2 scene order differs")
    if production and (
        coverage.get("keyframe_count") != EXPECTED_KEYFRAMES
        or coverage.get("successful_frame_count") != EXPECTED_SUCCESSFUL_FRAMES
        or coverage.get("source_count") != EXPECTED_SOURCES
        or coverage.get("identity_verified_source_count") != EXPECTED_SOURCES
    ):
        raise F4RunnerError("sealed F2 paper100 census differs")

    f0_rows = f0.get("scenes")
    if (
        f0.get("schema") != EXPECTED_F0_SCHEMA
        or f0.get("protocol_id") != EXPECTED_F0_PROTOCOL
        or f0.get("complete") is not True
        or not isinstance(f0_rows, list)
    ):
        raise F4RunnerError("sealed F0 merged receipt contract differs")
    f0_by_scene: dict[str, dict[str, Any]] = {}
    for row in f0_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("scene_id"), str):
            raise F4RunnerError("sealed F0 scene row is invalid")
        if row["scene_id"] in f0_by_scene:
            raise F4RunnerError("sealed F0 scene rows are duplicated")
        f0_by_scene[row["scene_id"]] = dict(row)
    if any(scene not in f0_by_scene for scene in scenes):
        raise F4RunnerError("sealed F0 receipt misses a paper100 scene")

    scene_list = _regular_file(scene_list_path, "paper100 scene list", ".txt")
    if production and _sha256(scene_list) != EXPECTED_SCENE_LIST_SHA256:
        raise F4RunnerError("paper100 scene-list SHA-256 differs")
    listed = tuple(
        line.strip()
        for line in scene_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if listed != scenes:
        raise F4RunnerError("paper100 scene-list order differs from sealed F2")

    return (
        {"path": os.fspath(f2_path), "sha256": f2_sha, "run_signature_sha256": f2.get("run_signature_sha256")},
        {"path": os.fspath(f0_path), "sha256": f0_sha, "run_signature_sha256": f0.get("run_signature_sha256")},
        scenes,
        tuple(dict(row) for row in f2_rows),
        f0_by_scene,
    )


def _source_join(
    *,
    scene_id: str,
    scene_index: int,
    frame_id: int,
    frame_ordinal: int,
    source: Mapping[str, Any],
    candidate: Mapping[str, Any],
    mask: Mapping[str, Any],
    expected_rank: int,
) -> dict[str, Any]:
    rank = source.get("rank")
    raw_index = source.get("raw_index")
    mask_sha = source.get("mask_sha256")
    expected_id = f"{scene_id}/frame_{frame_id:06d}/raw_{int(raw_index):03d}" if isinstance(raw_index, int) else ""
    if (
        rank != expected_rank
        or source.get("candidate_index") != expected_rank
        or not isinstance(raw_index, int)
        or source.get("source_id") != expected_id
    ):
        raise F4RunnerError(f"F2 source identity/order differs: {source.get('source_id')}")
    key = (rank, raw_index, mask_sha)
    if tuple(candidate.get(name) for name in ("rank", "raw_index", "mask_sha256")) != key:
        raise F4RunnerError(f"F2/F0 candidate join differs: {expected_id}")
    if tuple(mask.get(name) for name in ("rank", "raw_index", "mask_sha256")) != key:
        raise F4RunnerError(f"F2/F0 mask diagnostic join differs: {expected_id}")
    if mask.get("selected") is not True or mask.get("decision") != "selected":
        raise F4RunnerError(f"F0 joined mask is not selected: {expected_id}")
    for source_key, candidate_key in (
        ("confidence", "confidence"),
        ("mask_sha256", "mask_sha256"),
        ("points_and_voxel_keys_sha256", "points_and_voxel_keys_sha256"),
        ("stored_point_count", "stored_point_count"),
        ("f0_world_q02", "world_q02"),
        ("f0_world_q98", "world_q98"),
    ):
        if source.get(source_key) != candidate.get(candidate_key):
            raise F4RunnerError(f"F2/F0 source field differs: {expected_id}/{source_key}")
    tight_box = candidate.get("tight_box_xyxy")
    if tight_box != mask.get("tight_box_xyxy"):
        raise F4RunnerError(f"F0 candidate/mask tight box differs: {expected_id}")
    try:
        box = np.asarray(tight_box, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise F4RunnerError(f"invalid tight box: {expected_id}") from error
    if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        raise F4RunnerError(f"invalid tight box: {expected_id}")
    hypotheses = source.get("hypotheses")
    if not isinstance(hypotheses, Mapping) or tuple(hypotheses) != ("H0", "HL", "HLG"):
        # JSON preserves insertion order in the sealed receipt; key set is what
        # matters, but spelling an exact tuple catches accidental additions.
        if not isinstance(hypotheses, Mapping) or set(hypotheses) != {"H0", "HL", "HLG"}:
            raise F4RunnerError(f"sealed F2 hypotheses differ: {expected_id}")
    identity = {
        "scene_index": scene_index,
        "frame_ordinal": frame_ordinal,
        "frame_id": frame_id,
        "rank": rank,
        "raw_index": raw_index,
        "mask_sha256": mask_sha,
        "points_and_voxel_keys_sha256": source.get("points_and_voxel_keys_sha256"),
        "source_id": expected_id,
    }
    f0_lineage = {
        "candidate_sha256": _canonical_json_sha256(candidate),
        "mask_diagnostic_sha256": _canonical_json_sha256(mask),
        "provider_box_ignored": True,
    }
    f2_lineage = {
        "source_sha256": _canonical_json_sha256(source),
        "f2_receipt_result_sha256": (
            source.get("f2_receipt", {}).get("result_sha256")
            if isinstance(source.get("f2_receipt"), Mapping)
            else None
        ),
    }
    return {
        **identity,
        "candidate_index": expected_rank,
        "tight_box_xyxy": [float(value) for value in box],
        "f0_source_lineage": f0_lineage,
        "f2_source_lineage": f2_lineage,
        "sealed_f2_hypotheses_sha256": _canonical_json_sha256(hypotheses),
        "sealed_hypotheses": copy.deepcopy(dict(hypotheses)),
        "join_sha256": _canonical_json_sha256({"identity": identity, "f0": f0_lineage, "f2": f2_lineage, "tight_box_xyxy": box.tolist()}),
    }


def _result_rows_and_diagnostics(result: object) -> tuple[list[object], dict[str, Any]]:
    if isinstance(result, Mapping):
        rows = result.get("rows")
        diagnostics = result.get("diagnostics", {})
    else:
        rows = getattr(result, "rows", None)
        diagnostics = getattr(result, "diagnostics", {})
    if not isinstance(rows, (list, tuple)):
        raise F4RunnerError("Boxer provider result lacks an ordered rows sequence")
    return list(rows), _mapping(diagnostics, "Boxer frame diagnostics")


def _validate_provider_diagnostics(diagnostics: Mapping[str, Any], source_count: int) -> None:
    valid_count = diagnostics.get("valid_count")
    invalid_count = diagnostics.get("invalid_count")
    if (
        diagnostics.get("source_count") != source_count
        or isinstance(valid_count, bool)
        or not isinstance(valid_count, int)
        or isinstance(invalid_count, bool)
        or not isinstance(invalid_count, int)
        or valid_count < 0
        or invalid_count < 0
        or valid_count + invalid_count != source_count
        or diagnostics.get("cuda_synchronized") is not True
        or diagnostics.get("model_eval") is not True
        or diagnostics.get("model_parameters_frozen") is not True
        or diagnostics.get("model_forward_calls") != 1
    ):
        raise F4RunnerError("Boxer frame diagnostics violate the frozen one-forward contract")


def _normalize_hb(
    row_value: object,
    *,
    source_id: str,
    row_index: int,
    tight_box_xyxy: Sequence[float],
) -> dict[str, Any]:
    row_source_id = _field(row_value, "source_id")
    bound_row_index = _field(row_value, "row_index")
    if row_source_id not in (None, source_id):
        raise F4RunnerError(f"Boxer diagnostic source ID differs at row {row_index}")
    if bound_row_index not in (None, row_index):
        raise F4RunnerError(f"Boxer row index differs at row {row_index}")
    provider_box = _field(row_value, "input_tight_box_xyxy", tight_box_xyxy)
    try:
        provider_box_array = np.asarray(provider_box, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise F4RunnerError(f"Boxer input-box diagnostic is invalid at row {row_index}") from error
    expected_box = np.asarray(tight_box_xyxy, dtype=np.float64)
    if provider_box_array.shape != (4,) or not np.array_equal(provider_box_array, expected_box):
        raise F4RunnerError(f"Boxer did not use the sealed tight box at row {row_index}")

    core_valid = _field(row_value, "valid") is True
    validity = _diagnostic_jsonable(_field(row_value, "validity", {}))
    reason: str | None = None
    arrays: dict[str, np.ndarray] = {}
    specs = {
        "world_corners": (8, 3),
        "world_center": (3,),
        "local_extent": (3,),
        "world_rotation": (3, 3),
    }
    for key, shape in specs.items():
        try:
            array = np.asarray(_field(row_value, key), dtype=np.float64)
        except (TypeError, ValueError):
            array = np.empty((0,), dtype=np.float64)
        arrays[key] = array
        if array.shape != shape or not np.isfinite(array).all():
            reason = reason or f"invalid_{key}"
    try:
        camera_depth = float(_field(row_value, "camera_depth"))
    except (TypeError, ValueError):
        camera_depth = float("nan")
    if not math.isfinite(camera_depth) or camera_depth <= 1.0e-4:
        reason = reason or "nonpositive_camera_depth"
    if arrays["local_extent"].shape == (3,) and np.any(arrays["local_extent"] <= 0.0):
        reason = reason or "nonpositive_extent"
    if arrays["world_rotation"].shape == (3, 3) and np.isfinite(arrays["world_rotation"]).all():
        rotation = arrays["world_rotation"]
        if (
            np.linalg.det(rotation) <= 0.0
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-3, rtol=0.0)
        ):
            reason = reason or "rotation_not_so3"
    if not core_valid:
        explicit_reason = _field(row_value, "abstention_reason")
        validity_reasons = validity.get("reasons") if isinstance(validity, Mapping) else None
        if explicit_reason:
            reason = str(explicit_reason)
        elif isinstance(validity_reasons, list) and validity_reasons:
            reason = "provider_invalid:" + ",".join(str(item) for item in validity_reasons)
        else:
            reason = "provider_invalid"
    valid = core_valid and reason is None

    confidence = _optional_diagnostic(_field(row_value, "confidence"))
    logvar = _optional_diagnostic(_field(row_value, "logvar"))
    raw_params = _optional_diagnostic(_field(row_value, "raw_params"))
    provider_result_sha256 = _field(row_value, "result_sha256")
    if provider_result_sha256 is not None and not _valid_sha256(provider_result_sha256):
        raise F4RunnerError(f"Boxer provider result hash is invalid at row {row_index}")

    hb: dict[str, Any] = {
        "valid": valid,
        "abstention_reason": None if valid else reason,
        "row_index": row_index,
        "source_id": source_id,
        "input_tight_box_xyxy": [float(value) for value in expected_box],
        "world_corners": arrays["world_corners"].tolist() if valid else None,
        "world_center": arrays["world_center"].tolist() if valid else None,
        "local_extent": arrays["local_extent"].tolist() if valid else None,
        "world_rotation": arrays["world_rotation"].tolist() if valid else None,
        "camera_depth": camera_depth if valid else None,
        "confidence": confidence,
        "logvar": logvar,
        "raw_params": raw_params,
        "validity": validity,
        "provider_result_sha256": provider_result_sha256,
    }
    hb["result_sha256"] = _canonical_json_sha256(hb)
    return hb


def _extract_cuda_peak(diagnostics: Mapping[str, Any]) -> int:
    candidates: list[int] = []
    for key, value in _flatten_items(diagnostics):
        lower = key.lower()
        if "cuda" in lower or "memory" in lower:
            if lower.endswith(("peak_memory_bytes", "max_memory_allocated_bytes", "max_memory_reserved_bytes", "peak_bytes")):
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    candidates.append(value)
                elif isinstance(value, float) and math.isfinite(value) and value >= 0:
                    candidates.append(int(value))
    return max(candidates, default=0)


def _process_scene(
    *,
    f2_row: Mapping[str, Any],
    f0_merged_row: Mapping[str, Any],
    scene_index: int,
    provider: object,
    frame_loader: Callable[[Path, Path, Path], tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_root: Path,
    nonempty_call_index: list[int],
    model_receipts_sha256: str,
    run_signature_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scene_id = f2_row.get("scene_id")
    if not isinstance(scene_id, str) or f2_row.get("scene_index") != scene_index:
        raise F4RunnerError("F2 merged scene identity differs")
    f2_sidecar_path = _rehash_reference(f2_row.get("sidecar"), f"{scene_id} F2 sidecar", ".json")
    _, f2_scene = _read_json(f2_sidecar_path, f"{scene_id} F2 sidecar")
    if (
        f2_scene.get("schema") != EXPECTED_F2_SCENE_SCHEMA
        or f2_scene.get("protocol_id") != EXPECTED_F2_PROTOCOL
        or f2_scene.get("scene_id") != scene_id
        or f2_scene.get("scene_index") != scene_index
        or f2_scene.get("complete") is not True
    ):
        raise F4RunnerError(f"{scene_id} F2 sidecar contract differs")
    f0_sidecar_path = _rehash_reference(f2_scene.get("f0_sidecar"), f"{scene_id} F0 sidecar", ".json")
    if (
        not isinstance(f0_merged_row.get("sidecar"), Mapping)
        or f0_merged_row["sidecar"].get("path") != os.fspath(f0_sidecar_path)
        or f0_merged_row["sidecar"].get("sha256") != _sha256(f0_sidecar_path)
    ):
        raise F4RunnerError(f"{scene_id} F0 merged/scene lineage differs")
    _, f0_scene = _read_json(f0_sidecar_path, f"{scene_id} F0 sidecar")
    if (
        f0_scene.get("schema") != EXPECTED_F0_SCENE_SCHEMA
        or f0_scene.get("protocol_id") != EXPECTED_F0_PROTOCOL
        or f0_scene.get("scene_id") != scene_id
        or f0_scene.get("scene_index") != scene_index
        or f0_scene.get("complete") is not True
    ):
        raise F4RunnerError(f"{scene_id} F0 sidecar contract differs")

    evidence_path = _rehash_reference(f2_scene.get("evidence_npz"), f"{scene_id} F2 evidence", ".npz")
    schedule_path = _rehash_reference(f2_scene.get("schedule"), f"{scene_id} schedule", ".json")
    intrinsic_path, intrinsic = _load_intrinsic(f2_scene.get("intrinsic", {}))
    input_seals = [
        {"kind": "f2_sidecar", "path": os.fspath(f2_sidecar_path), "sha256": _sha256(f2_sidecar_path)},
        {"kind": "f0_sidecar", "path": os.fspath(f0_sidecar_path), "sha256": _sha256(f0_sidecar_path)},
        {"kind": "f2_evidence", "path": os.fspath(evidence_path), "sha256": _sha256(evidence_path)},
        {"kind": "schedule", "path": os.fspath(schedule_path), "sha256": _sha256(schedule_path)},
        {"kind": "intrinsic", "path": os.fspath(intrinsic_path), "sha256": _sha256(intrinsic_path)},
    ]
    f2_frames = f2_scene.get("frames")
    f0_frames = f0_scene.get("frames")
    if not isinstance(f2_frames, list) or not isinstance(f0_frames, list) or len(f2_frames) != len(f0_frames):
        raise F4RunnerError(f"{scene_id} F2/F0 frame ledger differs")

    output_frames: list[dict[str, Any]] = []
    source_ids: list[str] = []
    source_lineage_hashes: list[str] = []
    incremental_warm: list[float] = []
    composed_warm: list[float] = []
    all_incremental: list[float] = []
    all_composed: list[float] = []
    successful_count = 0
    provider_forward_count = 0
    valid_hb_count = 0
    cuda_peak = 0

    for frame_ordinal, (f2_frame, f0_frame) in enumerate(zip(f2_frames, f0_frames, strict=True)):
        if not isinstance(f2_frame, Mapping) or not isinstance(f0_frame, Mapping):
            raise F4RunnerError(f"{scene_id} frame row is invalid")
        frame_id = f2_frame.get("frame_id")
        if (
            f2_frame.get("frame_ordinal") != frame_ordinal
            or f0_frame.get("frame_ordinal") != frame_ordinal
            or f0_frame.get("frame_id") != frame_id
            or not isinstance(frame_id, int)
            or f0_frame.get("successful") is not f2_frame.get("successful")
        ):
            raise F4RunnerError(f"{scene_id} frame ledger differs at ordinal {frame_ordinal}")
        if f2_frame.get("successful") is not True:
            if f2_frame.get("sources") not in ([], None):
                raise F4RunnerError(f"{scene_id}/{frame_id} abstained frame has F2 sources")
            output_frames.append({
                "frame_ordinal": frame_ordinal,
                "frame_id": frame_id,
                "successful": False,
                "abstention": copy.deepcopy(f2_frame.get("abstention")),
                "current_only": True,
                "provider_invoked": False,
                "sources": [],
                "runtime": None,
            })
            continue

        successful_count += 1
        sources = f2_frame.get("sources")
        funnel = f0_frame.get("funnel")
        candidates = funnel.get("candidates") if isinstance(funnel, Mapping) else None
        masks = funnel.get("masks") if isinstance(funnel, Mapping) else None
        inputs = f0_frame.get("inputs")
        if not isinstance(sources, list) or not isinstance(candidates, list) or not isinstance(masks, list) or not isinstance(inputs, Mapping):
            raise F4RunnerError(f"{scene_id}/{frame_id} source inputs are absent")
        if len(sources) != len(candidates) or len(sources) > MAX_SOURCES_PER_FRAME:
            raise F4RunnerError(f"{scene_id}/{frame_id} source census exceeds frozen bounds")
        if inputs.get("current_pose_valid") is not True or inputs.get("f0_pose_forward_filled") is not False:
            raise F4RunnerError(f"{scene_id}/{frame_id} does not use the exact current pose")
        if inputs.get("producer_orientation") != 0 or inputs.get("producer_rotation_k") != 0:
            raise F4RunnerError(f"{scene_id}/{frame_id} producer orientation differs")
        if inputs.get("producer_depth_shape") != [480, 640] or inputs.get("producer_image_shape") != [480, 640, 3]:
            raise F4RunnerError(f"{scene_id}/{frame_id} producer shape differs")

        selected_masks: dict[tuple[Any, Any, Any], Mapping[str, Any]] = {}
        for mask in masks:
            if isinstance(mask, Mapping) and mask.get("selected") is True:
                key = (mask.get("rank"), mask.get("raw_index"), mask.get("mask_sha256"))
                if key in selected_masks:
                    raise F4RunnerError(f"{scene_id}/{frame_id} selected mask join is ambiguous")
                selected_masks[key] = mask
        if len(selected_masks) != len(sources):
            raise F4RunnerError(f"{scene_id}/{frame_id} selected mask census differs")

        joined: list[dict[str, Any]] = []
        for rank, source in enumerate(sources):
            if not isinstance(source, Mapping) or not isinstance(candidates[rank], Mapping):
                raise F4RunnerError(f"{scene_id}/{frame_id} source row is invalid")
            key = (source.get("rank"), source.get("raw_index"), source.get("mask_sha256"))
            mask = selected_masks.get(key)
            if mask is None:
                raise F4RunnerError(f"{scene_id}/{frame_id} exact F0 mask join failed")
            joined.append(_source_join(
                scene_id=scene_id,
                scene_index=scene_index,
                frame_id=frame_id,
                frame_ordinal=frame_ordinal,
                source=source,
                candidate=candidates[rank],
                mask=mask,
                expected_rank=rank,
            ))

        rgb_path = _regular_file(Path(str(inputs.get("rgb_path", ""))), "sealed current RGB")
        depth_path = _regular_file(Path(str(inputs.get("depth_path", ""))), "sealed current depth", ".png")
        pose_path = _regular_file(Path(str(inputs.get("pose_path", ""))), "sealed current pose", ".txt")
        for kind, path, expected_sha in (
            ("rgb", rgb_path, inputs.get("rgb_sha256")),
            ("depth", depth_path, inputs.get("depth_sha256")),
            ("pose", pose_path, inputs.get("pose_sha256")),
        ):
            if not isinstance(expected_sha, str) or _sha256(path) != expected_sha:
                raise F4RunnerError(f"{scene_id}/{frame_id} sealed {kind} rehash differs")
            input_seals.append({"kind": kind, "frame_ordinal": frame_ordinal, "frame_id": frame_id, "path": os.fspath(path), "sha256": expected_sha})

        provider_invoked = bool(joined)
        output_sources: list[dict[str, Any]] = []
        frame_runtime: dict[str, Any] | None = None
        if provider_invoked:
            call_index = nonempty_call_index[0]
            warmup_excluded = call_index < WARMUP_FORWARD_COUNT
            started_ns = time.perf_counter_ns()
            rgb, depth_m, camera_to_world = frame_loader(rgb_path, depth_path, pose_path)
            if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
                raise F4RunnerError("frame loader must return uint8 RGB[480,640,3]")
            if depth_m.shape != (480, 640) or not np.issubdtype(depth_m.dtype, np.floating):
                raise F4RunnerError("frame loader must return metric depth[480,640]")
            if camera_to_world.shape != (4, 4) or not np.isfinite(camera_to_world).all():
                raise F4RunnerError("frame loader must return a finite 4x4 camera pose")
            boxes = np.asarray([source["tight_box_xyxy"] for source in joined], dtype=np.float32)
            ids = [source["source_id"] for source in joined]
            infer = getattr(provider, "infer_batch", None)
            if not callable(infer):
                raise F4RunnerError("frozen Boxer provider lacks infer_batch")
            result = infer(
                scene_id=scene_id,
                frame_id=frame_id,
                rgb=np.ascontiguousarray(rgb),
                depth_m=np.ascontiguousarray(depth_m, dtype=np.float32),
                K=np.ascontiguousarray(intrinsic, dtype=np.float32),
                camera_to_world=np.ascontiguousarray(camera_to_world, dtype=np.float64),
                boxes_xyxy=np.ascontiguousarray(boxes),
                source_ids=tuple(ids),
            )
            rows, diagnostics = _result_rows_and_diagnostics(result)
            incremental_ms = (time.perf_counter_ns() - started_ns) / 1.0e6
            if len(rows) != len(joined):
                raise F4RunnerError(f"{scene_id}/{frame_id} Boxer output row count differs")
            _validate_provider_diagnostics(diagnostics, len(joined))
            for row_index, (source, provider_row) in enumerate(zip(joined, rows, strict=True)):
                hb = _normalize_hb(
                    provider_row,
                    source_id=source["source_id"],
                    row_index=row_index,
                    tight_box_xyxy=source["tight_box_xyxy"],
                )
                hypotheses = copy.deepcopy(source.pop("sealed_hypotheses"))
                hypotheses["HB"] = hb
                source["hypotheses"] = hypotheses
                source["source_lineage_sha256"] = _canonical_json_sha256({
                    "identity": {key: source[key] for key in ("scene_index", "frame_ordinal", "frame_id", "rank", "raw_index", "mask_sha256", "points_and_voxel_keys_sha256", "source_id")},
                    "join_sha256": source["join_sha256"],
                    "sealed_f2_hypotheses_sha256": source["sealed_f2_hypotheses_sha256"],
                    "hb_result_sha256": hb["result_sha256"],
                })
                valid_hb_count += int(hb["valid"])
                source_ids.append(source["source_id"])
                source_lineage_hashes.append(source["source_lineage_sha256"])
                output_sources.append(source)
            f2_runtime = f2_frame.get("runtime")
            if not isinstance(f2_runtime, Mapping):
                raise F4RunnerError(f"{scene_id}/{frame_id} sealed F2 runtime is absent")
            inherited_complete_ms = _number(f2_runtime.get("complete_ms"), "F2 complete_ms")
            composed_ms = inherited_complete_ms + incremental_ms
            deadline_missed_all = composed_ms >= 833.33
            deadline_missed_warm = (not warmup_excluded) and deadline_missed_all
            frame_runtime = {
                "provider_call_index_in_shard": call_index,
                "f4_warmup_excluded": warmup_excluded,
                "f4_incremental_ms": incremental_ms,
                "sealed_f0_f2_complete_ms": inherited_complete_ms,
                "replay_composed_ms": composed_ms,
                "replay_composed_ms_per_source_frame": composed_ms / SOURCE_FRAME_STRIDE,
                # Keep the cold/all-frame diagnostic visible, but only the
                # explicitly warm ledger is eligible for the formal gate.
                "gap25_deadline_missed": deadline_missed_all,
                "gap25_deadline_missed_warm": deadline_missed_warm,
                "provider_diagnostics": diagnostics,
            }
            nonempty_call_index[0] += 1
            provider_forward_count += 1
            cuda_peak = max(cuda_peak, _extract_cuda_peak(diagnostics))
            all_incremental.append(incremental_ms)
            all_composed.append(composed_ms)
            if not warmup_excluded:
                incremental_warm.append(incremental_ms)
                composed_warm.append(composed_ms)
        else:
            # A successful frame can contain zero selected sources.  It is
            # still causal evidence, but invoking Boxer would violate the
            # frozen 0-box rule.
            output_sources = []

        output_frames.append({
            "frame_ordinal": frame_ordinal,
            "frame_id": frame_id,
            "successful": True,
            "abstention": None,
            "current_only": True,
            "max_accessed_frame_ordinal": frame_ordinal,
            "provider_invoked": provider_invoked,
            "input": {
                "rgb": {"path": os.fspath(rgb_path), "sha256": inputs["rgb_sha256"]},
                "depth": {"path": os.fspath(depth_path), "sha256": inputs["depth_sha256"]},
                "pose": {"path": os.fspath(pose_path), "sha256": inputs["pose_sha256"]},
                "intrinsic": {"path": os.fspath(intrinsic_path), "sha256": f2_scene["intrinsic"]["sha256"]},
                "rgb_color_order": "RGB_after_BGR_to_RGB",
                "image_shape": [480, 640, 3],
                "depth_shape": [480, 640],
                "depth_unit": "metre",
                "box_source": "F0_candidate.tight_box_xyxy",
            },
            "sources": output_sources,
            "runtime": frame_runtime,
        })

    before_hash = _canonical_json_sha256(input_seals)
    for seal in input_seals:
        path = _regular_file(Path(seal["path"]), f"{scene_id} frozen input after replay")
        if _sha256(path) != seal["sha256"]:
            raise F4RunnerError(f"{scene_id} frozen input changed during replay")
    after_hash = _canonical_json_sha256(input_seals)
    if before_hash != after_hash:
        raise F4RunnerError(f"{scene_id} frozen-input aggregate changed")

    counts = {
        "keyframe_count": len(output_frames),
        "successful_frame_count": successful_count,
        "source_count": len(source_ids),
        "provider_forward_count": provider_forward_count,
        "valid_hb_count": valid_hb_count,
        "invalid_hb_count": len(source_ids) - valid_hb_count,
    }
    f2_summary = f2_scene.get("summary")
    inherited_cuda_peak = 0
    if isinstance(f2_summary, Mapping):
        value = f2_summary.get("gpu_peak_memory_bytes", 0)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            inherited_cuda_peak = value
    if len(output_frames) and inherited_cuda_peak == 0 and len(output_frames) >= 50:
        # Production sidecars always seal this replay peak; a large fixture
        # without it is much more likely to be a malformed production input.
        raise F4RunnerError(f"{scene_id} sealed F0+F2 CUDA peak is absent")
    scene_runtime = {
        "f4_incremental_all_ms": _distribution(all_incremental),
        "f4_incremental_warm_ms": _distribution(incremental_warm),
        "replay_composed_all_ms": _distribution(all_composed),
        "replay_composed_warm_ms": _distribution(composed_warm),
        "gap25_all_deadline_miss_count": int(sum(value >= 833.33 for value in all_composed)),
        "gap25_warm_deadline_miss_count": int(sum(value >= 833.33 for value in composed_warm)),
        "f4_cuda_peak_memory_bytes": cuda_peak,
        "sealed_f0_f2_cuda_peak_memory_bytes": inherited_cuda_peak,
        "cuda_peak_memory_bytes": max(cuda_peak, inherited_cuda_peak),
    }
    scene_receipt: dict[str, Any] = {
        "schema": SCENE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "scene_id": scene_id,
        "scene_index": scene_index,
        "run_signature_sha256": run_signature_sha256,
        "contracts": dict(CONTRACTS),
        "inputs": {
            "f2_sidecar": input_seals[0],
            "f0_sidecar": input_seals[1],
            "f2_evidence": input_seals[2],
            "schedule": input_seals[3],
            "intrinsic": input_seals[4],
            "frozen_inputs_before_sha256": before_hash,
            "frozen_inputs_after_sha256": after_hash,
            "model_receipts_sha256": model_receipts_sha256,
        },
        "frames": output_frames,
        "counts": counts,
        "runtime": scene_runtime,
        "source_ids_sha256": _canonical_json_sha256(source_ids),
        "source_lineage_sha256": _canonical_json_sha256(source_lineage_hashes),
        "native_output_mutation_count": 0,
    }
    scene_receipt["content_sha256"] = _canonical_json_sha256(scene_receipt)
    output_path = output_root / "scenes" / f"{scene_id}.json"
    output_sha = _atomic_create_json(output_path, scene_receipt)
    manifest_row = {
        "scene_id": scene_id,
        "scene_index": scene_index,
        "sidecar": {"path": os.fspath(output_path.resolve()), "sha256": output_sha},
        "counts": counts,
        "runtime": scene_runtime,
        "source_ids_sha256": scene_receipt["source_ids_sha256"],
        "source_lineage_sha256": scene_receipt["source_lineage_sha256"],
    }
    return manifest_row, scene_receipt


def run_f4(
    *,
    f2_receipt_path: Path = DEFAULT_F2_RECEIPT,
    f0_receipt_path: Path = DEFAULT_F0_RECEIPT,
    scene_list_path: Path = DEFAULT_SCENE_LIST,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    shard_index: int,
    num_shards: int = EXPECTED_SHARDS,
    provider_factory: Callable[..., object] | None = None,
    frame_loader: Callable[[Path, Path, Path], tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    expected_scene_count: int = EXPECTED_SCENES,
    expected_keyframes: int | None = None,
    expected_successful_frames: int | None = None,
    expected_sources: int | None = None,
    plan_only: bool = False,
) -> dict[str, Any]:
    """Run one deterministic F4 shard and return its manifest/plan.

    ``provider_factory`` and ``frame_loader`` are explicit injection seams for
    unit tests.  Production callers must omit both.
    """
    if num_shards != EXPECTED_SHARDS or shard_index not in range(num_shards):
        raise F4RunnerError("F4 is frozen to exactly two deterministic shards")
    production = expected_scene_count == EXPECTED_SCENES
    if production and (provider_factory is not None or frame_loader is not None):
        raise F4RunnerError("production F4 forbids injected provider/frame loader")
    f2_seal, f0_seal, scenes, f2_rows, f0_by_scene = _load_inputs(
        Path(f2_receipt_path),
        Path(f0_receipt_path),
        Path(scene_list_path),
        expected_scene_count=expected_scene_count,
    )
    assigned_indices = tuple(index for index in range(len(scenes)) if index % num_shards == shard_index)
    plan = {
        "protocol_id": PROTOCOL_ID,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "scene_count": len(assigned_indices),
        "scene_indices": list(assigned_indices),
        "scene_ids": [scenes[index] for index in assigned_indices],
        "f2_receipt": f2_seal,
        "f0_receipt": f0_seal,
        "output_root": os.fspath(Path(output_root).resolve()),
        "contracts": dict(CONTRACTS),
    }
    if plan_only:
        return plan

    core_module: object | None = None
    injected_factory = provider_factory
    if provider_factory is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not visible or "," in visible:
            raise F4RunnerError("production F4 requires one CUDA_VISIBLE_DEVICES token")
        try:
            from boxfusion import fastsam_boxer_f4_shadow as core_module  # type: ignore
        except ImportError as error:  # pragma: no cover - production dependency
            raise F4RunnerError("F4 Boxer core is unavailable") from error
        provider_factory = getattr(core_module, "create_frozen_boxer_provider", None)
        if not callable(provider_factory):
            raise F4RunnerError("F4 core lacks create_frozen_boxer_provider")
    assert provider_factory is not None
    provider = provider_factory(device="cuda")
    model_before = _provider_frozen_receipts(provider, production=production)
    model_before_sha = _canonical_json_sha256(model_before)
    sources = _source_receipts(core_module, injected_factory)
    signature_payload = {
        "protocol_id": PROTOCOL_ID,
        "f2_receipt": f2_seal,
        "f0_receipt": f0_seal,
        "scene_order": list(scenes),
        "scene_list_sha256": _sha256(_regular_file(Path(scene_list_path), "paper100 scene list", ".txt")),
        "sources": sources,
        "model_receipts_sha256": model_before_sha,
        "contracts": dict(CONTRACTS),
        "num_shards": num_shards,
    }
    run_signature = _canonical_json_sha256(signature_payload)
    loader = frame_loader or _default_frame_loader
    scene_rows: list[dict[str, Any]] = []
    all_scene_receipts: list[dict[str, Any]] = []
    nonempty_call_index = [0]
    for scene_index in assigned_indices:
        row, receipt = _process_scene(
            f2_row=f2_rows[scene_index],
            f0_merged_row=f0_by_scene[scenes[scene_index]],
            scene_index=scene_index,
            provider=provider,
            frame_loader=loader,
            output_root=Path(output_root),
            nonempty_call_index=nonempty_call_index,
            model_receipts_sha256=model_before_sha,
            run_signature_sha256=run_signature,
        )
        scene_rows.append(row)
        all_scene_receipts.append(receipt)

    model_after = _provider_frozen_receipts(provider, production=production)
    model_after_sha = _canonical_json_sha256(model_after)
    if model_after_sha != model_before_sha:
        raise F4RunnerError("frozen Boxer model/source receipts changed during replay")
    totals = {
        key: int(sum(row["counts"][key] for row in scene_rows))
        for key in ("keyframe_count", "successful_frame_count", "source_count", "provider_forward_count", "valid_hb_count", "invalid_hb_count")
    }
    if production:
        expected = EXPECTED_SHARD_COUNTS[shard_index]
        for key, value in expected.items():
            if totals[key] != value:
                raise F4RunnerError(f"production shard {shard_index} {key} differs")

    warm_incremental = [
        float(frame["runtime"]["f4_incremental_ms"])
        for scene in all_scene_receipts
        for frame in scene["frames"]
        if frame.get("runtime") is not None and frame["runtime"].get("f4_warmup_excluded") is False
    ]
    warm_composed = [
        float(frame["runtime"]["replay_composed_ms"])
        for scene in all_scene_receipts
        for frame in scene["frames"]
        if frame.get("runtime") is not None and frame["runtime"].get("f4_warmup_excluded") is False
    ]
    manifest: dict[str, Any] = {
        "schema": SHARD_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "run_signature_sha256": run_signature,
        "signature_payload_sha256": _canonical_json_sha256(signature_payload),
        "contracts": dict(CONTRACTS),
        "inputs": {"f2_receipt": f2_seal, "f0_receipt": f0_seal, "scene_list": {"path": os.fspath(Path(scene_list_path).resolve()), "sha256": _sha256(Path(scene_list_path))}},
        "sources_receipt": sources,
        "model_receipts_before": model_before,
        "model_receipts_after": model_after,
        "model_receipts_sha256": model_before_sha,
        "environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "scenes": scene_rows,
        "totals": totals,
        "runtime": {
            "f4_incremental_warm_ms": _distribution(warm_incremental),
            "replay_composed_warm_ms": _distribution(warm_composed),
            "gap25_all_deadline_miss_count": int(sum(scene["runtime"]["gap25_all_deadline_miss_count"] for scene in all_scene_receipts)),
            "gap25_warm_deadline_miss_count": int(sum(scene["runtime"]["gap25_warm_deadline_miss_count"] for scene in all_scene_receipts)),
            "cuda_peak_memory_bytes": max((scene["runtime"]["cuda_peak_memory_bytes"] for scene in all_scene_receipts), default=0),
            "cold_model_load_ms": _number(getattr(provider, "model_load_ms", 0.0), "model_load_ms", default=0.0),
            "warmup_forward_count": min(WARMUP_FORWARD_COUNT, totals["provider_forward_count"]),
        },
        "native_output_mutation_count": 0,
    }
    manifest["content_sha256"] = _canonical_json_sha256(manifest)
    manifest_path = Path(output_root) / "shards" / f"shard-{shard_index:03d}-of-{num_shards:03d}.json"
    manifest["manifest_path"] = os.fspath(manifest_path.resolve())
    manifest_sha = _atomic_create_json(manifest_path, manifest)
    manifest["manifest_sha256"] = manifest_sha
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f2-receipt", type=Path, default=DEFAULT_F2_RECEIPT)
    parser.add_argument("--f0-receipt", type=Path, default=DEFAULT_F0_RECEIPT)
    parser.add_argument("--scene-list", type=Path, default=DEFAULT_SCENE_LIST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=EXPECTED_SHARDS)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_f4(
        f2_receipt_path=args.f2_receipt,
        f0_receipt_path=args.f0_receipt,
        scene_list_path=args.scene_list,
        output_root=args.output_root,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        plan_only=args.plan_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
