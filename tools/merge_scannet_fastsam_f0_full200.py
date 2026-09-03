#!/usr/bin/env python3
"""Validate and merge two frozen-FastSAM F0 full200 shadow shards.

This program is deliberately a receipt-only reducer.  Its only data inputs are
the pre-registered scene list, two shard JSON manifests, and the per-scene JSON
sidecars named by those manifests.  It neither accepts nor discovers ground
truth, evaluator state, terminal predictions, or prediction pickle files.

Structural/provenance failures abort without publishing anything.  Capacity,
latency, and memory thresholds are recorded as pass/fail gates in the final
receipt so that a failed experiment remains auditable.  Publication is
create-only and atomic: an existing ``F0_FASTSAM_FULL200.json`` is never
replaced.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "boxfusion.scannet_fastsam_f0_full200.merge.v1"
SHARD_SCHEMA = "boxfusion.scannet_fastsam_f0_full200.shard.v1"
SCENE_SCHEMA = "boxfusion.scannet_fastsam_f0_full200.scene.v1"
OUTPUT_NAME = "F0_FASTSAM_FULL200.json"
EXPECTED_PROTOCOL_ID = "F0-frozen-FastSAM-x-residual-automatic-mask-shadow-full200"
EXPECTED_SCENE_LIST_SHA256 = (
    "0e7e722d3e93ec4b721f12293a3f1e98ca62d475b42cc8b9d491878a897e9bd1"
)
EXPECTED_TORCH_VERSION = "2.6.0+cu124"
EXPECTED_TORCH_CUDA_VERSION = "12.4"
EXPECTED_ULTRALYTICS_VERSION = "8.4.105"
EXPECTED_OPENCV_VERSION = "4.6.0"
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 3090"
EXPECTED_COMPUTE_CAPABILITY = (8, 6)
EXPECTED_GPU_UUID_BY_LOGICAL_DEVICE = {
    "cuda:0": "GPU-97755ff7-98ad-196d-1250-21eb5c95149d",
    "cuda:1": "GPU-2715f5df-abd1-cb90-a32b-770881114397",
}
EXPECTED_CUDA_SYNCHRONIZATION_CONTRACT = (
    "provider synchronizes CUDA before and after every prediction; "
    "host masks bound the following CPU residual core"
)
EXPECTED_SHARD_CONTRACTS = {
    "shadow_only": True,
    "no_output_affecting": True,
    "birth_enabled": False,
    "ground_truth_access": False,
    "annotation_access": False,
    "evaluator_access": False,
    "terminal_native_prediction_access": False,
    "terminal_native_prediction_mutation": False,
    "terminal_prediction_pickle_write": False,
    "cutr_current_pred_boxes_access": True,
    "cutr_nonbox_field_use": False,
    "cutr_payload_deserialization_scope": "full_safe_payload",
    "clip_or_semantic_use": False,
    "tracking_or_history": False,
    "training": False,
    "online_learning": False,
    "external_pretraining_frozen": True,
    "current_pose_required_no_forward_fill": True,
}
EXPECTED_CHECKPOINT_PATH = Path(
    "/data/ZhaoX/RoboFusion/RoboFusion-master/focalconvsamfusion/OpenPCDet/"
    "checkpoints/FastSAM.pt"
)
EXPECTED_CHECKPOINT_BYTES = 144_943_063
EXPECTED_CHECKPOINT_SHA256 = (
    "c0be4e7ddbe4c15333d15a859c676d053c486d0a746a3be6a7a9790d52a9b6d7"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE_PATHS = {
    "runner": REPOSITORY_ROOT / "tools/run_scannet_fastsam_f0_full200.py",
    "core": REPOSITORY_ROOT / "boxfusion/fastsam_residual_shadow.py",
    "provider": REPOSITORY_ROOT / "boxfusion/fastsam_automatic_provider.py",
}
EXPECTED_NON_UPRIGHT_KEYFRAMES = {
    ("scene0246_00", 1900): 1,
    ("scene0426_00", 2200): 3,
}
EXPECTED_EXECUTION_CENSUS_SHA256 = (
    "c306d37296b3dcbea7266202eb0ca86482cf32175f7909c0b1d97ea696e46b53"
)
EXPECTED_SHARD_EXECUTION_COUNTS = {
    0: {
        "keyframes": 6_460,
        "invalid_pose_frames": 121,
        "non_upright_producer_frames": 0,
        "successful_frames": 6_339,
    },
    1: {
        "keyframes": 6_481,
        "invalid_pose_frames": 108,
        "non_upright_producer_frames": 2,
        "successful_frames": 6_371,
    },
}
EXPECTED_INVALID_CURRENT_POSE_FRAMES = 229
EXPECTED_NON_UPRIGHT_FRAME_COUNT = 2
EXPECTED_SUCCESSFUL_PROVIDER_FRAMES = 12_710

EXPECTED_SCENES = 200
EXPECTED_SHARDS = 2
EXPECTED_KEYFRAMES = 12_941

MIN_ACCEPTED_LIFTS = 1_500
MIN_ACCEPTED_SCENES = 160
MAX_CAP_SATURATED_SUCCESS_FRACTION = 0.25
MAX_PROVIDER_P95_MS = 200.0
MAX_COMPLETE_P95_MS = 250.0
MAX_COMPLETE_MS_EXCLUSIVE = 833.33
MAX_AMORTIZED_MS_PER_SOURCE_FRAME = 10.0
SOURCE_FRAME_STRIDE = 25.0
MAX_GPU_PEAK_BYTES = 4 * 1024**3

RUNTIME_HISTOGRAM_UPPER_BOUNDS_MS = (
    10.0,
    20.0,
    40.0,
    60.0,
    80.0,
    100.0,
    150.0,
    200.0,
    250.0,
    400.0,
    600.0,
    833.33,
)
LIFTS_PER_SCENE_HISTOGRAM_UPPER_BOUNDS = (0, 1, 5, 10, 25, 50, 100, 250, 500)
CONFIDENCE_HISTOGRAM_UPPER_BOUNDS = (
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
)
RATIO_HISTOGRAM_UPPER_BOUNDS = (
    0.0,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
)
MASK_PIXEL_AREA_HISTOGRAM_UPPER_BOUNDS = (
    255,
    511,
    1_023,
    2_047,
    4_095,
    8_191,
    16_383,
    32_767,
    65_535,
    131_071,
    307_200,
)
RAW_MASKS_PER_FRAME_HISTOGRAM_UPPER_BOUNDS = (0, 1, 5, 10, 20, 40, 60, 80, 99, 100)
SELECTED_LIFTS_PER_FRAME_HISTOGRAM_UPPER_BOUNDS = (0, 1, 2, 4, 8, 12, 15, 16)
MAX_MASK_PIXELS = 480 * 640
PROVIDER_MAX_DET = 100
TOP_K_LIFTS_PER_FRAME = 16
SHARD_WARMUP_SUCCESSFUL_CALLS = 3


class F0MergeError(RuntimeError):
    """Raised when an F0 shard violates a sealed structural contract."""


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _nonnegative_int(value: object, label: str) -> int:
    if not _is_int(value) or int(value) < 0:
        raise F0MergeError(f"{label} must be a non-negative integer")
    return int(value)


def _nonnegative_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise F0MergeError(f"{label} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise F0MergeError(f"{label} must be a finite non-negative number")
    return result


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


def _regular_file(path: Path, label: str, *, suffix: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise F0MergeError(f"{label} must be a regular non-symlink file: {path}")
    resolved = path.resolve()
    if suffix is not None and resolved.suffix.lower() != suffix:
        raise F0MergeError(f"{label} must be a {suffix} file: {resolved}")
    if resolved.suffix.lower() in {".pkl", ".pickle"}:
        raise F0MergeError(f"prediction pickle inputs are forbidden: {resolved}")
    return resolved


def _read_json(path: Path, label: str) -> tuple[Path, dict[str, Any]]:
    source = _regular_file(path, label, suffix=".json")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise F0MergeError(f"invalid {label}: {source}") from error
    if not isinstance(value, dict):
        raise F0MergeError(f"{label} must contain one JSON object: {source}")
    return source, value


def _read_scene_list(path: Path) -> tuple[Path, list[str], str]:
    source = _regular_file(path, "exact full200 scene list")
    try:
        rows = [
            line.strip()
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    except (OSError, UnicodeDecodeError) as error:
        raise F0MergeError(f"invalid exact full200 scene list: {source}") from error
    if len(rows) != EXPECTED_SCENES or len(set(rows)) != EXPECTED_SCENES:
        raise F0MergeError(
            f"exact scene list must contain {EXPECTED_SCENES} unique rows, found {len(rows)}"
        )
    if any("/" in row or "\\" in row or not row for row in rows):
        raise F0MergeError("scene identifiers must be non-empty path-free strings")
    digest = _sha256(source)
    if digest != EXPECTED_SCENE_LIST_SHA256:
        raise F0MergeError(
            "full200 scene-list SHA-256 differs from the frozen F0 protocol"
        )
    return source, rows, digest


def _hex_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise F0MergeError(f"{label} must be a SHA-256 hex digest")
    return value.lower()


def _resolve_sidecar(reference: object, manifest_path: Path, label: str) -> Path:
    if not isinstance(reference, str) or not reference:
        raise F0MergeError(f"{label} path must be a non-empty string")
    raw = Path(reference)
    if raw.suffix.lower() != ".json":
        raise F0MergeError(f"{label} must reference JSON, not {raw.suffix or 'no suffix'}")
    if raw.is_absolute():
        return _regular_file(raw, label, suffix=".json")
    candidates = []
    for base in (manifest_path.parent, manifest_path.parent.parent):
        candidate = base / raw
        if candidate.exists() or candidate.is_symlink():
            resolved = _regular_file(candidate, label, suffix=".json")
            if resolved not in candidates:
                candidates.append(resolved)
    if len(candidates) != 1:
        raise F0MergeError(
            f"{label} relative path must resolve uniquely from shard directories: {raw}"
        )
    return candidates[0]


def _validate_schedule_ledger(
    *,
    manifest_scene: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    manifest_path: Path,
    scene: str,
    frame_ids: Sequence[int],
) -> dict[str, str]:
    schedule = _mapping(sidecar.get("schedule"), f"{scene} sidecar schedule")
    row_root = manifest_scene.get("schedule_root")
    sidecar_root = schedule.get("root")
    if not isinstance(row_root, str) or row_root != sidecar_root:
        raise F0MergeError(f"{scene} schedule root differs between receipts")
    raw_schedule_root = Path(row_root)
    if raw_schedule_root.is_symlink() or not raw_schedule_root.is_dir():
        raise F0MergeError(f"{scene} schedule root is not a regular directory")
    schedule_root = raw_schedule_root.resolve()
    row_path = manifest_scene.get("schedule_path")
    sidecar_path_value = schedule.get("manifest_path")
    if row_path != sidecar_path_value:
        raise F0MergeError(f"{scene} schedule path differs between receipts")
    schedule_path = _resolve_sidecar(
        row_path, manifest_path, f"CuTR schedule manifest {scene}"
    )
    if schedule_path.parent.parent != schedule_root or schedule_path.parent.name != scene:
        raise F0MergeError(f"{scene} schedule path is outside its sealed root")
    row_hash = _hex_digest(
        manifest_scene.get("schedule_sha256"), f"{scene} row schedule hash"
    )
    sidecar_hash = _hex_digest(
        schedule.get("manifest_sha256"), f"{scene} sidecar schedule hash"
    )
    actual_hash = _sha256(schedule_path)
    if row_hash != sidecar_hash or actual_hash != row_hash:
        raise F0MergeError(f"{scene} schedule manifest rehash differs")
    _, schedule_manifest = _read_json(
        schedule_path, f"CuTR schedule manifest {scene}"
    )
    recorded = schedule_manifest.get("recorded_frame_ids")
    records = schedule_manifest.get("records")
    if (
        not isinstance(recorded, list)
        or any(not _is_int(value) for value in recorded)
        or recorded != list(frame_ids)
        or not isinstance(records, list)
        or len(records) != len(recorded)
        or schedule_manifest.get("record_count") != len(recorded)
        or any(
            not isinstance(record, dict) or record.get("frame_id") != frame_id
            for frame_id, record in zip(recorded, records)
        )
    ):
        raise F0MergeError(f"{scene} frame IDs differ from sealed schedule records")
    ledger_hash = _canonical_json_sha256(list(frame_ids))
    if _hex_digest(
        sidecar.get("frame_id_ledger_sha256"), f"{scene} sidecar frame ledger"
    ) != ledger_hash or _hex_digest(
        manifest_scene.get("frame_id_ledger_sha256"),
        f"{scene} manifest frame ledger",
    ) != ledger_hash:
        raise F0MergeError(f"{scene} frame-id ledger hash differs")
    producer_fingerprint = _hex_digest(
        schedule_manifest.get("producer_fingerprint"),
        f"{scene} schedule producer fingerprint",
    )
    return {
        "path": os.fspath(schedule_path),
        "root": os.fspath(schedule_root),
        "sha256": actual_hash,
        "producer_fingerprint": producer_fingerprint,
    }


def _contract_value(
    contracts: Mapping[str, Any],
    *,
    positive_keys: Sequence[str] = (),
    negative_keys: Sequence[str] = (),
    label: str,
) -> bool:
    present: list[bool] = []
    for key in positive_keys:
        if key in contracts:
            if not isinstance(contracts[key], bool):
                raise F0MergeError(f"contract {key} must be boolean")
            present.append(bool(contracts[key]))
    for key in negative_keys:
        if key in contracts:
            if not isinstance(contracts[key], bool):
                raise F0MergeError(f"contract {key} must be boolean")
            present.append(not bool(contracts[key]))
    if not present:
        raise F0MergeError(f"missing sealed {label} contract")
    if not all(present):
        raise F0MergeError(f"sealed {label} contract is violated")
    return True


def _validate_contracts(value: object, label: str) -> dict[str, bool]:
    if not isinstance(value, dict):
        raise F0MergeError(f"{label} contracts must be an object")
    contracts: Mapping[str, Any] = value
    canonical = {
        "shadow_only": _contract_value(
            contracts, positive_keys=("shadow_only",), label="shadow-only"
        ),
        "no_output_affecting": _contract_value(
            contracts,
            positive_keys=("no_output_affecting", "output_unchanged"),
            negative_keys=("output_affecting",),
            label="no-output-affecting",
        ),
        "birth_disabled": _contract_value(
            contracts,
            positive_keys=("no_birth", "birth_disabled"),
            negative_keys=("birth_enabled",),
            label="birth-disabled",
        ),
        "ground_truth_access_disabled": _contract_value(
            contracts,
            positive_keys=("no_ground_truth_access", "ground_truth_access_disabled"),
            negative_keys=("ground_truth_access",),
            label="no-ground-truth-access",
        ),
        "native_prediction_access_disabled": _contract_value(
            contracts,
            positive_keys=(
                "no_native_prediction_access",
                "native_prediction_access_disabled",
            ),
            negative_keys=(
                "terminal_native_prediction_access",
                "native_prediction_access",
            ),
            label="no-terminal-native-prediction-access",
        ),
        "cutr_pred_boxes_access_only": _contract_value(
            contracts,
            positive_keys=(
                "cutr_current_pred_boxes_access",
                "current_frame_cutr_pred_boxes_only",
            ),
            label="current-CuTR-pred-box access",
        ),
        "cutr_nonbox_field_use_disabled": _contract_value(
            contracts,
            negative_keys=("cutr_nonbox_field_use",),
            label="no-CuTR-nonbox-field-use",
        ),
        "clip_or_semantic_use_disabled": _contract_value(
            contracts,
            negative_keys=("clip_or_semantic_use", "clip_or_semantic_access"),
            label="no-CLIP-or-semantic-use",
        ),
        "current_pose_no_forward_fill": _contract_value(
            contracts,
            positive_keys=("current_pose_required_no_forward_fill",),
            label="current-pose-no-forward-fill",
        ),
    }
    if contracts.get("cutr_payload_deserialization_scope") != "full_safe_payload":
        raise F0MergeError(
            "contract must disclose full_safe_payload CuTR deserialization scope"
        )
    return canonical


def _validate_environment(
    manifest: Mapping[str, Any], *, shard_index: int
) -> tuple[dict[str, Any], str]:
    environment = _mapping(
        manifest.get("environment"), f"shard {shard_index} runtime environment"
    )
    expected_keys = {
        "production_cuda_required",
        "dependency_injected_provider",
        "conda_environment",
        "python_version",
        "torch_version",
        "torch_cuda_version",
        "opencv_version",
        "ultralytics_version",
        "device",
        "cuda_available",
        "gpu_name",
        "gpu_uuid",
        "compute_capability",
        "cuda_visible_devices",
        "cuda_synchronization_contract",
    }
    if set(environment) != expected_keys:
        raise F0MergeError(
            f"shard {shard_index} environment receipt keys differ from frozen protocol"
        )
    expected_device = f"cuda:{shard_index}"
    expected_values: Mapping[str, object] = {
        "production_cuda_required": True,
        "dependency_injected_provider": False,
        "conda_environment": "boxfusion-online",
        "torch_version": EXPECTED_TORCH_VERSION,
        "torch_cuda_version": EXPECTED_TORCH_CUDA_VERSION,
        "opencv_version": EXPECTED_OPENCV_VERSION,
        "ultralytics_version": EXPECTED_ULTRALYTICS_VERSION,
        "device": expected_device,
        "cuda_available": True,
        "gpu_name": EXPECTED_GPU_NAME,
        "compute_capability": list(EXPECTED_COMPUTE_CAPABILITY),
        "cuda_visible_devices": None,
    }
    for key, expected in expected_values.items():
        if environment.get(key) != expected:
            raise F0MergeError(
                f"shard {shard_index} environment {key} differs from frozen RTX3090 protocol"
            )
    gpu_uuid = environment.get("gpu_uuid")
    if not isinstance(gpu_uuid, str) or not gpu_uuid:
        raise F0MergeError(f"shard {shard_index} GPU UUID receipt is missing")
    normalized_uuid = gpu_uuid if gpu_uuid.startswith("GPU-") else f"GPU-{gpu_uuid}"
    if normalized_uuid != EXPECTED_GPU_UUID_BY_LOGICAL_DEVICE[expected_device]:
        raise F0MergeError(f"shard {shard_index} GPU UUID differs from logical device")
    python_version = environment.get("python_version")
    sync_contract = environment.get("cuda_synchronization_contract")
    if not isinstance(python_version, str) or not python_version:
        raise F0MergeError(f"shard {shard_index} Python version receipt is missing")
    if sync_contract != EXPECTED_CUDA_SYNCHRONIZATION_CONTRACT:
        raise F0MergeError(f"shard {shard_index} CUDA sync contract differs")
    canonical = dict(environment)
    return canonical, _canonical_json_sha256(canonical)


def _validate_execution_identity(
    manifest: Mapping[str, Any], *, shard_index: int
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    checkpoint = _mapping(
        manifest.get("checkpoint"), f"shard {shard_index} checkpoint identity"
    )
    checkpoint_path = _regular_file(
        Path(str(checkpoint.get("path", ""))), "frozen FastSAM checkpoint"
    )
    normalized_checkpoint = {
        "path": os.fspath(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
        "sha256": _sha256(checkpoint_path),
    }
    expected_checkpoint = {
        "path": os.fspath(EXPECTED_CHECKPOINT_PATH.resolve()),
        "bytes": EXPECTED_CHECKPOINT_BYTES,
        "sha256": EXPECTED_CHECKPOINT_SHA256,
    }
    if dict(checkpoint) != normalized_checkpoint or normalized_checkpoint != expected_checkpoint:
        raise F0MergeError(f"shard {shard_index} frozen FastSAM checkpoint differs")

    sources = _mapping(manifest.get("sources"), f"shard {shard_index} sources")
    if set(sources) != set(EXPECTED_SOURCE_PATHS):
        raise F0MergeError(f"shard {shard_index} source identity keys differ")
    normalized_sources: dict[str, dict[str, str]] = {}
    for name, expected_path_raw in EXPECTED_SOURCE_PATHS.items():
        row = _mapping(sources.get(name), f"shard {shard_index} {name} source")
        path = _regular_file(Path(str(row.get("path", ""))), f"frozen {name} source")
        expected_path = expected_path_raw.resolve()
        normalized = {"path": os.fspath(path), "sha256": _sha256(path)}
        if path != expected_path or dict(row) != normalized:
            raise F0MergeError(f"shard {shard_index} frozen {name} source differs")
        normalized_sources[name] = normalized
    return normalized_checkpoint, normalized_sources


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise F0MergeError(f"{label} must be an object")
    return value


def _first_path(value: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> object:
    for path in paths:
        cursor: object = value
        for key in path:
            if not isinstance(cursor, dict) or key not in cursor:
                break
            cursor = cursor[key]
        else:
            return cursor
    return None


def _extract_runtime_from_frames(
    frames: Sequence[Mapping[str, Any]], scene: str, *, expected_device: str | None = None
) -> tuple[list[float], list[float], list[float], list[bool], list[int]]:
    provider: list[float] = []
    complete: list[float] = []
    receipt_total: list[float] = []
    successful_warmup_flags: list[bool] = []
    successful_call_indices: list[int] = []
    provider_paths = (
        ("runtime_ms", "provider"),
        ("runtime_ms", "fastsam"),
        ("runtime", "provider_ms"),
        ("runtime", "fastsam_ms"),
        ("provider_runtime_ms",),
    )
    complete_paths = (
        ("runtime_ms", "complete"),
        ("runtime_ms", "total"),
        ("runtime", "complete_ms"),
        ("runtime", "total_ms"),
        ("complete_runtime_ms",),
    )
    core_paths = (("runtime_ms", "core"), ("runtime", "core_ms"), ("core_runtime_ms",))
    receipt_paths = (
        ("runtime_ms", "receipt_total"),
        ("runtime", "receipt_total_ms"),
        ("receipt_total_runtime_ms",),
    )
    for index, frame in enumerate(frames):
        warmup = _first_path(
            frame,
            (("runtime_ms", "warmup_excluded"), ("runtime", "warmup_excluded")),
        )
        if warmup is not None and not isinstance(warmup, bool):
            raise F0MergeError(f"{scene} frame {index} warmup flag must be boolean")
        if warmup is None:
            raise F0MergeError(f"{scene} frame {index} is missing warmup flag")
        successful = frame.get("successful")
        if not isinstance(successful, bool):
            raise F0MergeError(f"{scene} frame {index} successful flag must be boolean")
        provider_value = _first_path(frame, provider_paths)
        complete_value = _first_path(frame, complete_paths)
        core_value = _first_path(frame, core_paths)
        receipt_value = _first_path(frame, receipt_paths)
        if any(value is None for value in (provider_value, core_value, complete_value, receipt_value)):
            raise F0MergeError(f"{scene} frame {index} has incomplete runtime receipt")
        provider_ms = _nonnegative_float(provider_value, f"{scene} provider runtime")
        core_ms = _nonnegative_float(core_value, f"{scene} core runtime")
        complete_ms = _nonnegative_float(complete_value, f"{scene} complete runtime")
        receipt_ms = _nonnegative_float(receipt_value, f"{scene} receipt total runtime")
        if complete_ms + 1e-3 < provider_ms + core_ms:
            raise F0MergeError(
                f"{scene} complete runtime is below provider plus core runtime"
            )
        if receipt_ms + 1e-6 < complete_ms:
            raise F0MergeError(f"{scene} receipt total runtime is below complete runtime")
        receipt_total.append(receipt_ms)
        call_index = _first_path(
            frame,
            (("runtime", "provider_call_index_in_shard"), ("provider_call_index_in_shard",)),
        )
        if successful:
            if not _is_int(call_index) or int(call_index) < 0:
                raise F0MergeError(f"{scene} successful frame lacks provider call index")
            successful_warmup_flags.append(bool(warmup))
            successful_call_indices.append(int(call_index))
            if expected_device is not None:
                _validate_provider_timing(
                    frame.get("provider_timing"),
                    expected_device=expected_device,
                    external_provider_ms=provider_ms,
                    label=f"{scene} frame {index}",
                )
            if not warmup:
                provider.append(provider_ms)
                complete.append(complete_ms)
        else:
            if call_index is not None:
                raise F0MergeError(f"{scene} abstained frame has a provider call index")
            if warmup or provider_ms != 0.0 or core_ms != 0.0 or complete_ms != 0.0:
                raise F0MergeError(f"{scene} abstained frame has provider/core runtime")
            if frame.get("provider_timing") not in ({}, None):
                raise F0MergeError(f"{scene} abstained frame has provider timing")
    return (
        provider,
        complete,
        receipt_total,
        successful_warmup_flags,
        successful_call_indices,
    )


def _validate_provider_timing(
    value: object,
    *,
    expected_device: str,
    external_provider_ms: float,
    label: str,
) -> dict[str, Any]:
    timing = _mapping(value, f"{label} provider timing")
    if timing.get("device") != expected_device or timing.get("cuda_synchronized") is not True:
        raise F0MergeError(f"{label} provider timing is not synchronized on {expected_device}")
    timestamps = []
    for key in ("started_ns", "prediction_finished_ns", "finished_ns"):
        number = timing.get(key)
        if not _is_int(number) or int(number) < 0:
            raise F0MergeError(f"{label} provider timestamp {key} is invalid")
        timestamps.append(int(number))
    if timestamps != sorted(timestamps) or timestamps[0] == timestamps[-1]:
        raise F0MergeError(f"{label} provider timestamps are not monotonic")
    seconds: dict[str, float] = {}
    milliseconds: dict[str, float] = {}
    for phase in ("prediction", "extraction", "total"):
        seconds[phase] = _nonnegative_float(
            timing.get(f"{phase}_seconds"), f"{label} {phase} seconds"
        )
        milliseconds[phase] = _nonnegative_float(
            timing.get(f"{phase}_ms"), f"{label} {phase} milliseconds"
        )
        if not math.isclose(
            milliseconds[phase], seconds[phase] * 1000.0, abs_tol=1e-6, rel_tol=1e-9
        ):
            raise F0MergeError(f"{label} provider {phase} seconds/ms differ")
    if not math.isclose(
        seconds["total"],
        seconds["prediction"] + seconds["extraction"],
        abs_tol=1e-6,
        rel_tol=1e-6,
    ):
        raise F0MergeError(f"{label} provider phase timing does not sum to total")
    if external_provider_ms + 1e-3 < milliseconds["total"]:
        raise F0MergeError(f"{label} external provider runtime is below synchronized timing")
    memory = {}
    for key in (
        "memory_allocated_before_bytes",
        "memory_allocated_after_bytes",
        "memory_reserved_before_bytes",
        "memory_reserved_after_bytes",
        "max_memory_allocated_bytes",
        "max_memory_reserved_bytes",
    ):
        memory[key] = _nonnegative_int(timing.get(key), f"{label} provider memory {key}")
    if memory["max_memory_allocated_bytes"] <= 0:
        raise F0MergeError(f"{label} CUDA allocated-memory peak is zero")
    if (
        memory["max_memory_allocated_bytes"] < memory["memory_allocated_before_bytes"]
        or memory["max_memory_allocated_bytes"] < memory["memory_allocated_after_bytes"]
        or memory["max_memory_reserved_bytes"] < memory["memory_reserved_before_bytes"]
        or memory["max_memory_reserved_bytes"] < memory["memory_reserved_after_bytes"]
    ):
        raise F0MergeError(f"{label} provider peak memory receipt is inconsistent")
    return dict(timing)


def _extract_summary_samples(
    sidecar: Mapping[str, Any], scene: str
) -> tuple[list[float], list[float], list[float]]:
    runtime = _first_path(sidecar, (("summary", "runtime"), ("runtime",)))
    if not isinstance(runtime, dict):
        return [], [], []
    provider_value = _first_path(
        runtime,
        (
            ("provider", "samples_ms"),
            ("provider_ms", "samples"),
            ("provider_samples_ms",),
            ("fastsam_samples_ms",),
        ),
    )
    complete_value = _first_path(
        runtime,
        (
            ("complete", "samples_ms"),
            ("complete_ms", "samples"),
            ("complete_samples_ms",),
        ),
    )
    receipt_value = _first_path(
        runtime,
        (
            ("receipt_total", "samples_ms"),
            ("receipt_total_ms", "samples"),
            ("receipt_total_samples_ms",),
        ),
    )
    if provider_value is None and complete_value is None and receipt_value is None:
        return [], [], []
    if not isinstance(provider_value, list) or not isinstance(complete_value, list):
        raise F0MergeError(f"{scene} summary provider/complete samples must be lists")
    if receipt_value is not None and not isinstance(receipt_value, list):
        raise F0MergeError(f"{scene} summary receipt runtime samples must be a list")
    provider = [
        _nonnegative_float(value, f"{scene} provider runtime sample")
        for value in provider_value
    ]
    complete = [
        _nonnegative_float(value, f"{scene} complete runtime sample")
        for value in complete_value
    ]
    receipt = [
        _nonnegative_float(value, f"{scene} receipt runtime sample")
        for value in (receipt_value or [])
    ]
    return provider, complete, receipt


def _extract_memory(
    manifest_scene: Mapping[str, Any], sidecar: Mapping[str, Any], scene: str
) -> tuple[int, int]:
    def one(names: Sequence[Sequence[str]], label: str) -> int:
        values = []
        for source in (manifest_scene, sidecar):
            value = _first_path(source, names)
            if value is not None:
                values.append(_nonnegative_int(value, f"{scene} {label}"))
        if not values:
            raise F0MergeError(f"missing {scene} {label}")
        if len(set(values)) != 1:
            raise F0MergeError(f"{scene} {label} differs between manifest and sidecar")
        return values[0]

    cpu = one(
        (
            ("cpu_peak_rss_bytes",),
            ("summary", "cpu_peak_rss_bytes"),
            ("summary", "memory", "cpu_peak_rss_bytes"),
        ),
        "CPU peak RSS",
    )
    gpu = one(
        (
            ("gpu_peak_memory_bytes",),
            ("summary", "gpu_peak_memory_bytes"),
            ("summary", "memory", "gpu_peak_memory_bytes"),
        ),
        "GPU peak memory",
    )
    return cpu, gpu


def _validate_counts(
    manifest_scene: Mapping[str, Any], sidecar: Mapping[str, Any], scene: str
) -> dict[str, int]:
    manifest_counts = _mapping(manifest_scene.get("counts"), f"{scene} manifest counts")
    sidecar_counts = _first_path(sidecar, (("summary", "counts"), ("counts",)))
    sidecar_counts = _mapping(sidecar_counts, f"{scene} sidecar counts")
    result: dict[str, int] = {}
    for key, value in sidecar_counts.items():
        if not isinstance(key, str) or not key:
            raise F0MergeError(f"{scene} count names must be non-empty strings")
        result[key] = _nonnegative_int(value, f"{scene} count {key}")
    for core in (
        "keyframes",
        "accepted_lifts",
        "successful_frames",
        "cap_saturated_frames",
        "provider_max_det_saturated_frames",
        "warmup_excluded_successful_frames",
        "non_upright_producer_frames",
    ):
        if core not in result:
            raise F0MergeError(f"missing {scene} core count {core}")
    if result["cap_saturated_frames"] > result["successful_frames"]:
        raise F0MergeError(f"{scene} cap-saturated frames exceed successful frames")
    for key, value in manifest_counts.items():
        checked = _nonnegative_int(value, f"{scene} manifest count {key}")
        if key in result and result[key] != checked:
            raise F0MergeError(f"{scene} count {key} differs between manifest and sidecar")
    for core in (
        "keyframes",
        "accepted_lifts",
        "successful_frames",
        "cap_saturated_frames",
        "provider_max_det_saturated_frames",
        "warmup_excluded_successful_frames",
        "non_upright_producer_frames",
    ):
        if core not in manifest_counts:
            raise F0MergeError(f"missing {scene} manifest core count {core}")
    return result


def _empty_histogram(bounds: Sequence[float]) -> dict[str, Any]:
    return {
        "upper_bounds": list(bounds),
        "bin_counts": [0 for _ in bounds],
        "overflow_count": 0,
        "sample_count": 0,
    }


def _histogram_add(histogram: dict[str, Any], value: float) -> None:
    histogram["sample_count"] += 1
    for index, upper in enumerate(histogram["upper_bounds"]):
        if value <= float(upper):
            histogram["bin_counts"][index] += 1
            return
    histogram["overflow_count"] += 1


def _merge_histograms(histograms: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not histograms:
        raise F0MergeError("cannot merge an empty histogram collection")
    bounds = histograms[0].get("upper_bounds")
    if not isinstance(bounds, list):
        raise F0MergeError("histogram upper bounds are invalid")
    output = _empty_histogram(bounds)
    for histogram in histograms:
        if histogram.get("upper_bounds") != bounds:
            raise F0MergeError("fixed histogram bounds differ across scenes")
        counts = histogram.get("bin_counts")
        if not isinstance(counts, list) or len(counts) != len(bounds):
            raise F0MergeError("fixed histogram bin count shape differs")
        for index, count in enumerate(counts):
            output["bin_counts"][index] += _nonnegative_int(
                count, "histogram bin count"
            )
        output["overflow_count"] += _nonnegative_int(
            histogram.get("overflow_count"), "histogram overflow count"
        )
        output["sample_count"] += _nonnegative_int(
            histogram.get("sample_count"), "histogram sample count"
        )
    if (
        sum(output["bin_counts"]) + output["overflow_count"]
        != output["sample_count"]
    ):
        raise F0MergeError("fixed histogram totals are inconsistent")
    return output


def _ratio(value: object, label: str) -> float:
    result = _nonnegative_float(value, label)
    if result > 1.0:
        raise F0MergeError(f"{label} must be in [0,1]")
    return result


def _validate_frame_funnels(
    frames: Sequence[Mapping[str, Any]], scene: str, counts: Mapping[str, int]
) -> dict[str, Any]:
    histograms = {
        "raw_confidence": _empty_histogram(CONFIDENCE_HISTOGRAM_UPPER_BOUNDS),
        "mask_pixel_area": _empty_histogram(MASK_PIXEL_AREA_HISTOGRAM_UPPER_BOUNDS),
        "valid_depth_ratio": _empty_histogram(RATIO_HISTOGRAM_UPPER_BOUNDS),
        "residual_ratio": _empty_histogram(RATIO_HISTOGRAM_UPPER_BOUNDS),
        "raw_masks_per_successful_frame": _empty_histogram(
            RAW_MASKS_PER_FRAME_HISTOGRAM_UPPER_BOUNDS
        ),
        "selected_lifts_per_successful_frame": _empty_histogram(
            SELECTED_LIFTS_PER_FRAME_HISTOGRAM_UPPER_BOUNDS
        ),
    }
    computed: Counter[str] = Counter()
    seen_non_upright: set[tuple[str, int]] = set()
    for frame_position, frame in enumerate(frames):
        computed["keyframes"] += 1
        inputs = _mapping(frame.get("inputs"), f"{scene} frame {frame_position} inputs")
        cutr_count = _nonnegative_int(
            inputs.get("cutr_box_count"), f"{scene} frame {frame_position} CuTR count"
        )
        frame_id = _nonnegative_int(
            frame.get("frame_id"), f"{scene} frame {frame_position} id"
        )
        orientation = _nonnegative_int(
            inputs.get("producer_orientation"),
            f"{scene} frame {frame_position} producer orientation",
        )
        expected_orientation = EXPECTED_NON_UPRIGHT_KEYFRAMES.get((scene, frame_id), 0)
        if orientation != expected_orientation:
            raise F0MergeError(f"{scene}/{frame_id} orientation differs from census")
        cache_image_size = inputs.get("cutr_cache_image_size")
        expected_image_size = [480, 640] if orientation == 0 else [640, 480]
        if cache_image_size != expected_image_size:
            raise F0MergeError(f"{scene}/{frame_id} cache-coordinate image size differs")
        current_pose_valid = inputs.get("current_pose_valid")
        if not isinstance(current_pose_valid, bool):
            raise F0MergeError(f"{scene}/{frame_id} current-pose validity is not boolean")
        if inputs.get("f0_pose_forward_filled") is not False:
            raise F0MergeError(f"{scene}/{frame_id} illegally forward-fills pose")
        computed["cutr_boxes"] += cutr_count
        successful = frame.get("successful")
        if not isinstance(successful, bool):
            raise F0MergeError(f"{scene} frame {frame_position} successful must be boolean")
        funnel_value = frame.get("funnel")
        if not successful:
            if funnel_value is not None:
                raise F0MergeError(f"{scene} abstained frame carries a mask funnel")
            reason = frame.get("abstention")
            if (scene, frame_id) in EXPECTED_NON_UPRIGHT_KEYFRAMES:
                if reason != "non_upright_cache_coordinate_frame" or not current_pose_valid:
                    raise F0MergeError(
                        f"{scene}/{frame_id} must be the sealed non-upright abstention"
                    )
                computed["non_upright_producer_frames"] += 1
                seen_non_upright.add((scene, frame_id))
            elif reason == "invalid_current_pose" and not current_pose_valid:
                computed["invalid_pose_frames"] += 1
            else:
                raise F0MergeError(f"{scene}/{frame_id} has an unsealed abstention")
            continue
        if not current_pose_valid or orientation != 0:
            raise F0MergeError(f"{scene}/{frame_id} provider ran on invalid pose/orientation")
        computed["successful_frames"] += 1
        if frame.get("abstention") is not None:
            raise F0MergeError(f"{scene} successful frame also carries an abstention")
        funnel = _mapping(funnel_value, f"{scene} frame {frame_position} funnel")
        funnel_counts: dict[str, int] = {}
        for key in (
            "input_mask_count",
            "input_explained_box_count",
            "explained_union_pixels",
            "pre_dedup_eligible_count",
            "deduplicated_count",
            "post_dedup_count",
            "lifting_eligible_count",
            "selected_count",
            "cap_rejected_count",
        ):
            funnel_counts[key] = _nonnegative_int(
                funnel.get(key), f"{scene} frame {frame_position} funnel {key}"
            )
        raw_count = funnel_counts["input_mask_count"]
        selected_count = funnel_counts["selected_count"]
        if raw_count > PROVIDER_MAX_DET:
            raise F0MergeError(f"{scene} frame exceeds frozen provider max_det=100")
        if selected_count > TOP_K_LIFTS_PER_FRAME:
            raise F0MergeError(f"{scene} frame exceeds frozen selected-lift cap=16")
        if funnel_counts["input_explained_box_count"] != cutr_count:
            raise F0MergeError(f"{scene} frame explained-box funnel count differs")
        if funnel_counts["explained_union_pixels"] > MAX_MASK_PIXELS:
            raise F0MergeError(f"{scene} frame explained union exceeds image area")

        masks_value = funnel.get("masks")
        candidates_value = funnel.get("candidates")
        if not isinstance(masks_value, list) or not isinstance(candidates_value, list):
            raise F0MergeError(f"{scene} frame funnel masks/candidates must be lists")
        if len(masks_value) != raw_count or len(candidates_value) != selected_count:
            raise F0MergeError(f"{scene} frame funnel list counts differ")
        mask_by_index: dict[int, Mapping[str, Any]] = {}
        pre_count = dedup_count = lifted_count = selected_flag_count = cap_count = 0
        for mask_position, raw_value in enumerate(masks_value):
            mask = _mapping(
                raw_value, f"{scene} frame {frame_position} mask {mask_position}"
            )
            raw_index = mask.get("raw_index")
            if not _is_int(raw_index) or not 0 <= int(raw_index) < raw_count:
                raise F0MergeError(f"{scene} frame has invalid raw mask index")
            raw_index = int(raw_index)
            if raw_index in mask_by_index:
                raise F0MergeError(f"{scene} frame has duplicate raw mask index")
            mask_by_index[raw_index] = mask
            confidence = _ratio(mask.get("confidence"), f"{scene} raw confidence")
            pixel_count = _nonnegative_int(
                mask.get("pixel_count"), f"{scene} mask pixel count"
            )
            valid_count = _nonnegative_int(
                mask.get("valid_pixel_count"), f"{scene} valid-depth pixel count"
            )
            residual_count = _nonnegative_int(
                mask.get("residual_pixel_count"), f"{scene} residual pixel count"
            )
            if pixel_count > MAX_MASK_PIXELS or valid_count > pixel_count or residual_count > valid_count:
                raise F0MergeError(f"{scene} mask support counts are inconsistent")
            valid_ratio = _ratio(mask.get("valid_ratio"), f"{scene} valid-depth ratio")
            residual_ratio = _ratio(mask.get("residual_ratio"), f"{scene} residual ratio")
            expected_valid = valid_count / pixel_count if pixel_count else 0.0
            expected_residual = residual_count / valid_count if valid_count else 0.0
            if not math.isclose(valid_ratio, expected_valid, abs_tol=1e-9, rel_tol=0.0):
                raise F0MergeError(f"{scene} valid-depth ratio differs from pixel counts")
            if not math.isclose(
                residual_ratio, expected_residual, abs_tol=1e-9, rel_tol=0.0
            ):
                raise F0MergeError(f"{scene} residual ratio differs from pixel counts")
            flags = {}
            for key in ("pre_dedup_eligible", "deduplicated", "lifted", "selected"):
                if not isinstance(mask.get(key), bool):
                    raise F0MergeError(f"{scene} mask flag {key} must be boolean")
                flags[key] = bool(mask[key])
            pre_count += flags["pre_dedup_eligible"]
            dedup_count += flags["deduplicated"]
            lifted_count += flags["lifted"]
            selected_flag_count += flags["selected"]
            cap_count += mask.get("decision") == "top_k_cap"
            _histogram_add(histograms["raw_confidence"], confidence)
            _histogram_add(histograms["mask_pixel_area"], float(pixel_count))
            _histogram_add(histograms["valid_depth_ratio"], valid_ratio)
            _histogram_add(histograms["residual_ratio"], residual_ratio)
        if sorted(mask_by_index) != list(range(raw_count)):
            raise F0MergeError(f"{scene} frame raw mask ledger is incomplete")
        expected_post = pre_count - dedup_count
        cross_checks = {
            "pre_dedup_eligible_count": pre_count,
            "deduplicated_count": dedup_count,
            "post_dedup_count": expected_post,
            "lifting_eligible_count": lifted_count,
            "selected_count": selected_flag_count,
            "cap_rejected_count": cap_count,
        }
        for key, actual in cross_checks.items():
            if funnel_counts[key] != actual:
                raise F0MergeError(f"{scene} frame funnel {key} differs from masks")
        if lifted_count != selected_flag_count:
            raise F0MergeError(f"{scene} lifted and selected mask counts differ")
        candidate_indices: list[int] = []
        for rank, raw_value in enumerate(candidates_value):
            candidate = _mapping(raw_value, f"{scene} candidate {rank}")
            raw_index = candidate.get("raw_index")
            if not _is_int(raw_index) or int(raw_index) not in mask_by_index:
                raise F0MergeError(f"{scene} candidate raw index is invalid")
            raw_index = int(raw_index)
            mask = mask_by_index[raw_index]
            if not mask["selected"] or candidate.get("rank") != rank or mask.get("rank") != rank:
                raise F0MergeError(f"{scene} candidate rank/selected receipt differs")
            candidate_indices.append(raw_index)
        if len(set(candidate_indices)) != len(candidate_indices):
            raise F0MergeError(f"{scene} candidates contain duplicate raw indices")
        rejection_counts = _mapping(
            funnel.get("rejection_counts"), f"{scene} frame rejection counts"
        )
        checked_rejections = {
            str(key): _nonnegative_int(value, f"{scene} rejection count {key}")
            for key, value in rejection_counts.items()
        }
        if sum(checked_rejections.values()) != raw_count:
            raise F0MergeError(f"{scene} rejection/decision counts do not cover raw masks")

        computed["raw_masks"] += raw_count
        computed["pre_dedup_eligible_masks"] += pre_count
        computed["deduplicated_masks"] += dedup_count
        computed["lifting_eligible_masks"] += lifted_count
        computed["accepted_lifts"] += selected_count
        computed["cap_rejected_masks"] += cap_count
        computed["cap_saturated_frames"] += cap_count > 0
        computed["provider_max_det_saturated_frames"] += raw_count == PROVIDER_MAX_DET
        _histogram_add(histograms["raw_masks_per_successful_frame"], float(raw_count))
        _histogram_add(
            histograms["selected_lifts_per_successful_frame"], float(selected_count)
        )

    for key in (
        "keyframes",
        "successful_frames",
        "invalid_pose_frames",
        "non_upright_producer_frames",
        "cutr_boxes",
        "raw_masks",
        "pre_dedup_eligible_masks",
        "deduplicated_masks",
        "lifting_eligible_masks",
        "accepted_lifts",
        "cap_rejected_masks",
        "cap_saturated_frames",
    ):
        if counts.get(key) != computed[key]:
            raise F0MergeError(f"{scene} aggregate count {key} differs from frame funnels")
    if "provider_max_det_saturated_frames" in counts and (
        counts["provider_max_det_saturated_frames"]
        != computed["provider_max_det_saturated_frames"]
    ):
        raise F0MergeError(f"{scene} provider max_det saturation count differs")
    return {
        "histograms": histograms,
        "provider_max_det_saturated_frames": computed[
            "provider_max_det_saturated_frames"
        ],
        "seen_non_upright_keyframes": sorted(seen_non_upright),
    }


def _validate_scene_sidecar(
    manifest_scene: Mapping[str, Any],
    manifest_path: Path,
    *,
    expected_scene: str,
    expected_index: int,
    expected_protocol_id: str,
    expected_run_signature: str,
    expected_environment_sha256: str | None = None,
    expected_device: str | None = None,
    expected_checkpoint: Mapping[str, Any] | None = None,
    expected_sources: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    reference = manifest_scene.get("sidecar_path")
    sidecar_path = _resolve_sidecar(
        reference, manifest_path, f"F0 sidecar {expected_scene}"
    )
    expected_hash = _hex_digest(
        manifest_scene.get("sidecar_sha256"), f"{expected_scene} sidecar hash"
    )
    actual_hash = _sha256(sidecar_path)
    if actual_hash != expected_hash:
        raise F0MergeError(f"rehash failed for {expected_scene} sidecar")
    _, sidecar = _read_json(sidecar_path, f"F0 sidecar {expected_scene}")
    if sidecar.get("schema") != SCENE_SCHEMA:
        raise F0MergeError(f"unexpected sidecar schema for {expected_scene}")
    if sidecar.get("complete") is not True:
        raise F0MergeError(f"incomplete sidecar for {expected_scene}")
    if sidecar.get("protocol_id") != expected_protocol_id:
        raise F0MergeError(f"sidecar protocol id differs for {expected_scene}")
    if sidecar.get("scene_id") != expected_scene:
        raise F0MergeError(f"sidecar scene identity differs for {expected_scene}")
    if sidecar.get("scene_index") != expected_index:
        raise F0MergeError(f"sidecar scene index differs for {expected_scene}")
    if sidecar.get("run_signature_sha256") != expected_run_signature:
        raise F0MergeError(f"sidecar run signature differs for {expected_scene}")
    if expected_environment_sha256 is not None and sidecar.get(
        "environment_sha256"
    ) != expected_environment_sha256:
        raise F0MergeError(f"sidecar environment hash differs for {expected_scene}")
    if expected_checkpoint is not None and sidecar.get("checkpoint") != dict(
        expected_checkpoint
    ):
        raise F0MergeError(f"sidecar checkpoint identity differs for {expected_scene}")
    if expected_sources is not None and sidecar.get("sources") != {
        key: dict(value) for key, value in expected_sources.items()
    }:
        raise F0MergeError(f"sidecar source identity differs for {expected_scene}")
    if "contracts" in sidecar:
        _validate_contracts(sidecar["contracts"], f"sidecar {expected_scene}")
    frames_value = sidecar.get("frames")
    if not isinstance(frames_value, list):
        raise F0MergeError(f"{expected_scene} frames must be a list")
    frames: list[Mapping[str, Any]] = []
    frame_ids: list[int] = []
    for position, row in enumerate(frames_value):
        frame = _mapping(row, f"{expected_scene} frame row {position}")
        frame_id = frame.get("frame_id")
        if not _is_int(frame_id) or int(frame_id) < 0:
            raise F0MergeError(f"invalid frame id in {expected_scene} row {position}")
        frames.append(frame)
        frame_ids.append(int(frame_id))
    if frame_ids != sorted(set(frame_ids)):
        raise F0MergeError(f"duplicate or non-increasing frames in {expected_scene}")
    schedule_receipt = _validate_schedule_ledger(
        manifest_scene=manifest_scene,
        sidecar=sidecar,
        manifest_path=manifest_path,
        scene=expected_scene,
        frame_ids=frame_ids,
    )
    keyframe_count = _nonnegative_int(
        manifest_scene.get("keyframe_count"), f"{expected_scene} keyframe count"
    )
    if keyframe_count != len(frames):
        raise F0MergeError(f"keyframe count differs for {expected_scene}")
    sidecar_keyframes = _first_path(
        sidecar, (("summary", "keyframe_count"), ("keyframe_count",))
    )
    if sidecar_keyframes is not None and _nonnegative_int(
        sidecar_keyframes, f"{expected_scene} sidecar keyframe count"
    ) != keyframe_count:
        raise F0MergeError(f"sidecar keyframe count differs for {expected_scene}")

    counts = _validate_counts(manifest_scene, sidecar, expected_scene)
    if counts["keyframes"] != keyframe_count:
        raise F0MergeError(f"counted keyframes differ for {expected_scene}")
    if counts["successful_frames"] > keyframe_count:
        raise F0MergeError(f"successful frames exceed keyframes for {expected_scene}")

    funnel_stats = _validate_frame_funnels(frames, expected_scene, counts)
    (
        provider,
        complete,
        receipt_total,
        warmup_flags,
        call_indices,
    ) = _extract_runtime_from_frames(
        frames, expected_scene, expected_device=expected_device
    )
    summary_provider, summary_complete, summary_receipt = _extract_summary_samples(
        sidecar, expected_scene
    )
    if provider and summary_provider:
        if (
            provider != summary_provider
            or complete != summary_complete
            or (summary_receipt and receipt_total != summary_receipt)
        ):
            raise F0MergeError(f"runtime samples differ for {expected_scene}")
    elif summary_provider:
        provider, complete = summary_provider, summary_complete
        if summary_receipt:
            receipt_total = summary_receipt
    if counts["warmup_excluded_successful_frames"] != sum(warmup_flags):
        raise F0MergeError(f"{expected_scene} warmup-excluded count differs from frames")
    cpu_peak, gpu_peak = _extract_memory(manifest_scene, sidecar, expected_scene)
    if counts["successful_frames"] and gpu_peak <= 0:
        raise F0MergeError(f"{expected_scene} successful CUDA scene has zero GPU peak")
    if expected_device is not None:
        provider_allocated_peak = max(
            (
                int(frame["provider_timing"]["max_memory_allocated_bytes"])
                for frame in frames
                if frame["successful"]
            ),
            default=0,
        )
        if gpu_peak < provider_allocated_peak:
            raise F0MergeError(
                f"{expected_scene} GPU peak is below provider allocated-memory peak"
            )

    return {
        "scene_id": expected_scene,
        "scene_index": expected_index,
        "keyframe_count": keyframe_count,
        "frame_ids": frame_ids,
        "counts": counts,
        "provider_samples_ms": provider,
        "complete_samples_ms": complete,
        "receipt_total_samples_ms": receipt_total,
        "successful_warmup_flags": warmup_flags,
        "successful_call_indices": call_indices,
        "quality_histograms": funnel_stats["histograms"],
        "provider_max_det_saturated_frames": funnel_stats[
            "provider_max_det_saturated_frames"
        ],
        "seen_non_upright_keyframes": funnel_stats["seen_non_upright_keyframes"],
        "execution_census_records": [
            {
                "scene_index": expected_index,
                "scene_id": expected_scene,
                "frame_id": int(frame["frame_id"]),
                "current_pose_valid": bool(frame["inputs"]["current_pose_valid"]),
                "sealed_non_upright_orientation": int(
                    frame["inputs"]["producer_orientation"]
                ),
                "provider_success": bool(frame["successful"]),
            }
            for frame in frames
        ],
        "cpu_peak_rss_bytes": cpu_peak,
        "gpu_peak_memory_bytes": gpu_peak,
        "sidecar": {
            "path": os.fspath(sidecar_path),
            "sha256": actual_hash,
        },
        "schedule": schedule_receipt,
        "first_frame": {
            "frame_id": frame_ids[0],
            "rgb_path": frames[0]["inputs"].get("rgb_path"),
            "rgb_sha256": frames[0]["inputs"].get("rgb_sha256"),
        },
    }


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        raise F0MergeError("cannot compute an empty runtime distribution")
    position = (len(sorted_values) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise F0MergeError("runtime distribution cannot be empty")
    ordered = sorted(float(value) for value in values)
    mean = math.fsum(ordered) / len(ordered)
    variance = math.fsum((value - mean) ** 2 for value in ordered) / len(ordered)
    return {
        "count": len(ordered),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": ordered[0],
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
    }


def _fixed_histogram(values: Iterable[float], bounds: Sequence[float]) -> dict[str, Any]:
    counts = [0 for _ in bounds]
    overflow = 0
    sample_count = 0
    for raw in values:
        value = float(raw)
        sample_count += 1
        for index, upper in enumerate(bounds):
            if value <= float(upper):
                counts[index] += 1
                break
        else:
            overflow += 1
    return {
        "upper_bounds": list(bounds),
        "bin_counts": counts,
        "overflow_count": overflow,
        "sample_count": sample_count,
    }


def _gate(actual: float | int, comparator: str, threshold: float | int) -> dict[str, Any]:
    if comparator == ">=":
        passed = actual >= threshold
    elif comparator == "<=":
        passed = actual <= threshold
    elif comparator == "<":
        passed = actual < threshold
    else:  # pragma: no cover - programmer error, not receipt input
        raise AssertionError(comparator)
    return {
        "actual": actual,
        "comparator": comparator,
        "threshold": threshold,
        "passed": bool(passed),
    }


def _validate_shard_header(
    manifest: Mapping[str, Any],
    path: Path,
    *,
    scene_list_path: Path,
    scene_list_hash: str,
) -> tuple[int, int, str, str, list[int], list[str], Sequence[Mapping[str, Any]]]:
    if manifest.get("schema") != SHARD_SCHEMA:
        raise F0MergeError(f"unexpected shard schema: {path}")
    if manifest.get("mode") != "shadow" or manifest.get("complete") is not True:
        raise F0MergeError(f"shard must be a completed shadow receipt: {path}")
    protocol_id = manifest.get("protocol_id")
    if protocol_id != EXPECTED_PROTOCOL_ID:
        raise F0MergeError(f"protocol id differs from frozen F0: {path}")
    signature = _hex_digest(manifest.get("run_signature_sha256"), "run signature")
    if manifest.get("contracts") != EXPECTED_SHARD_CONTRACTS:
        raise F0MergeError(f"shard contracts differ from frozen F0 protocol: {path}")
    contracts = _validate_contracts(manifest.get("contracts"), f"shard {path}")
    if not all(contracts.values()):  # defensive; _validate_contracts already raises
        raise F0MergeError(f"shard contracts failed: {path}")

    scene_list = _mapping(manifest.get("scene_list"), f"scene-list receipt {path}")
    if scene_list.get("sha256") != scene_list_hash:
        raise F0MergeError(f"scene-list hash differs in {path}")
    if scene_list.get("exact_scene_count") != EXPECTED_SCENES:
        raise F0MergeError(f"scene-list count differs in {path}")
    if manifest.get("full200_keyframe_count") != EXPECTED_KEYFRAMES:
        raise F0MergeError(f"full200 keyframe declaration differs in {path}")
    recorded_path = scene_list.get("path")
    if not isinstance(recorded_path, str) or Path(recorded_path).resolve() != scene_list_path:
        raise F0MergeError(f"scene-list path differs in {path}")

    shard = _mapping(manifest.get("shard"), f"shard descriptor {path}")
    shard_index = shard.get("index")
    shard_count = shard.get("count")
    if not _is_int(shard_index) or not _is_int(shard_count):
        raise F0MergeError(f"invalid shard index/count in {path}")
    if shard_count != EXPECTED_SHARDS or not 0 <= shard_index < shard_count:
        raise F0MergeError(f"expected a two-way shard index in {path}")
    indices = shard.get("scene_indices")
    order = shard.get("scene_order")
    scenes = manifest.get("scenes")
    if (
        not isinstance(indices, list)
        or not isinstance(order, list)
        or not isinstance(scenes, list)
        or any(not _is_int(value) for value in indices)
        or any(not isinstance(value, str) for value in order)
        or any(not isinstance(value, dict) for value in scenes)
        or len(indices) != len(order)
        or len(order) != len(scenes)
    ):
        raise F0MergeError(f"invalid shard scene ledger in {path}")
    if indices != sorted(set(indices)):
        raise F0MergeError(f"duplicate or unordered scene indices in {path}")
    expected_indices = list(range(int(shard_index), EXPECTED_SCENES, EXPECTED_SHARDS))
    if [int(value) for value in indices] != expected_indices:
        raise F0MergeError(f"shard {shard_index} is not the frozen parity partition")
    return (
        int(shard_index),
        int(shard_count),
        protocol_id,
        signature,
        [int(value) for value in indices],
        list(order),
        scenes,
    )


def _validate_resume_rewarm(
    *,
    manifest: Mapping[str, Any],
    manifest_scene_rows: Sequence[Mapping[str, Any]],
    validated_scenes: Sequence[Mapping[str, Any]],
    expected_device: str,
    shard_index: int,
) -> dict[str, Any]:
    resumed_flags = []
    for position, row in enumerate(manifest_scene_rows):
        resumed = row.get("resumed")
        if not isinstance(resumed, bool):
            raise F0MergeError(f"shard {shard_index} scene {position} resumed flag missing")
        resumed_flags.append(resumed)
    completed_prefix = 0
    while completed_prefix < len(resumed_flags) and resumed_flags[completed_prefix]:
        completed_prefix += 1
    if any(resumed_flags[completed_prefix:]):
        raise F0MergeError(f"shard {shard_index} resumed scenes are not an exact prefix")
    pending = len(resumed_flags) - completed_prefix
    required = completed_prefix > 0 and pending > 0
    receipt = _mapping(
        manifest.get("resume_rewarm"), f"shard {shard_index} resume rewarm"
    )
    if receipt.get("required") is not required:
        raise F0MergeError(f"shard {shard_index} resume rewarm requirement differs")
    if receipt.get("completed_scene_count") != completed_prefix or receipt.get(
        "pending_scene_count"
    ) != pending:
        raise F0MergeError(f"shard {shard_index} resume rewarm scene counts differ")
    for key in (
        "all_successful",
        "excluded_from_scene_counts",
        "excluded_from_capacity",
        "excluded_from_runtime_distributions",
    ):
        if receipt.get(key) is not True:
            raise F0MergeError(f"shard {shard_index} resume rewarm {key} must be true")
    call_count = _nonnegative_int(
        receipt.get("call_count"), f"shard {shard_index} resume rewarm call count"
    )
    if manifest.get("resume_rewarm_calls") != call_count:
        raise F0MergeError(f"shard {shard_index} resume rewarm top-level count differs")
    calls = receipt.get("calls")
    if not isinstance(calls, list) or len(calls) != call_count:
        raise F0MergeError(f"shard {shard_index} resume rewarm calls differ")
    if not required:
        expected_reason = (
            "no_pending_scene" if pending == 0 else "fresh_or_resume_without_completed_scene"
        )
        if call_count != 0 or calls or receipt.get("reason") != expected_reason:
            raise F0MergeError(f"shard {shard_index} unexpected resume rewarm evidence")
        return {
            "required": False,
            "call_count": 0,
            "completed_scene_count": completed_prefix,
            "pending_scene_count": pending,
        }

    if call_count != SHARD_WARMUP_SUCCESSFUL_CALLS or receipt.get("reason") != (
        "cold_resume_with_completed_prefix_and_pending_suffix"
    ):
        raise F0MergeError(f"shard {shard_index} resume rewarm is not exactly three calls")
    first_pending = validated_scenes[completed_prefix]
    first_frame = first_pending["first_frame"]
    if (
        receipt.get("scene_id") != first_pending["scene_id"]
        or receipt.get("frame_id") != first_frame["frame_id"]
        or receipt.get("rgb_path") != first_frame["rgb_path"]
        or receipt.get("rgb_sha256") != first_frame["rgb_sha256"]
    ):
        raise F0MergeError(f"shard {shard_index} resume rewarm source differs")
    rgb_path = _regular_file(Path(str(receipt["rgb_path"])), "resume rewarm RGB")
    if _sha256(rgb_path) != _hex_digest(
        receipt.get("rgb_sha256"), "resume rewarm RGB hash"
    ):
        raise F0MergeError(f"shard {shard_index} resume rewarm RGB rehash differs")
    for ordinal, raw_call in enumerate(calls):
        call = _mapping(raw_call, f"shard {shard_index} rewarm call {ordinal}")
        if call.get("ordinal") != ordinal or call.get("success") is not True:
            raise F0MergeError(f"shard {shard_index} resume rewarm call order differs")
        wall_ms = _nonnegative_float(
            call.get("wall_ms"), f"shard {shard_index} resume rewarm wall"
        )
        raw_masks = _nonnegative_int(
            call.get("raw_mask_count"), f"shard {shard_index} rewarm raw masks"
        )
        if raw_masks > PROVIDER_MAX_DET:
            raise F0MergeError(f"shard {shard_index} resume rewarm exceeds max_det")
        for key in ("masks_sha256", "confidences_sha256", "boxes_xyxy_sha256"):
            _hex_digest(call.get(key), f"shard {shard_index} rewarm {key}")
        _validate_provider_timing(
            call.get("provider_timing"),
            expected_device=expected_device,
            external_provider_ms=wall_ms,
            label=f"shard {shard_index} resume rewarm call {ordinal}",
        )
    return {
        "required": True,
        "call_count": call_count,
        "completed_scene_count": completed_prefix,
        "pending_scene_count": pending,
        "scene_id": first_pending["scene_id"],
        "frame_id": first_frame["frame_id"],
    }


def _validate_shared_run_signature(
    *,
    manifests: Sequence[Mapping[str, Any]],
    ordered_scenes: Sequence[Mapping[str, Any]],
    exact_scene_order: Sequence[str],
    scene_list_sha256: str,
    checkpoint: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, str]],
    expected_signature: str,
) -> dict[str, Any]:
    scene_root_values = [manifest.get("scene_root") for manifest in manifests]
    if (
        any(not isinstance(value, str) for value in scene_root_values)
        or len(set(scene_root_values)) != 1
    ):
        raise F0MergeError("shards disagree on sealed scene root")
    scene_root = Path(scene_root_values[0])
    if scene_root.is_symlink() or not scene_root.is_dir():
        raise F0MergeError(f"sealed scene root is not a regular directory: {scene_root}")
    scene_root = scene_root.resolve()
    core_receipts = []
    environment_protocols = []
    for shard_index, manifest in enumerate(manifests):
        policy = _mapping(manifest.get("policy"), f"shard {shard_index} policy")
        core_receipts.append(
            {
                "core_schema": policy.get("core_schema"),
                "core_policy": policy.get("core"),
            }
        )
        environment = _mapping(
            manifest.get("environment"), f"shard {shard_index} environment"
        )
        environment_protocols.append(
            {
                key: value
                for key, value in environment.items()
                if key not in {"device", "gpu_uuid"}
            }
        )
    if any(receipt != core_receipts[0] for receipt in core_receipts[1:]):
        raise F0MergeError("shards disagree on frozen F0 core policy")
    core_schema = core_receipts[0]["core_schema"]
    core_policy = core_receipts[0]["core_policy"]
    if not isinstance(core_schema, str) or not core_schema or not isinstance(core_policy, dict):
        raise F0MergeError("frozen F0 core policy receipt is invalid")
    if any(value != environment_protocols[0] for value in environment_protocols[1:]):
        raise F0MergeError("shards disagree on invariant environment protocol")
    schedule_manifests = [
        {
            "scene_id": scene["scene_id"],
            "root": scene["schedule"]["root"],
            "sha256": scene["schedule"]["sha256"],
            "producer_fingerprint": scene["schedule"]["producer_fingerprint"],
        }
        for scene in ordered_scenes
    ]
    payload = {
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "scene_list_sha256": scene_list_sha256,
        "scene_order": list(exact_scene_order),
        "scene_root": os.fspath(scene_root),
        "schedule_manifests": schedule_manifests,
        "checkpoint": dict(checkpoint),
        "sources": {key: dict(value) for key, value in sources.items()},
        "environment_protocol": environment_protocols[0],
        "core_schema": core_schema,
        "core_policy": core_policy,
    }
    actual = _canonical_json_sha256(payload)
    if actual != expected_signature:
        raise F0MergeError("shared run signature does not reconstruct from sealed inputs")
    return {"sha256": actual, "payload_sha256": actual}


def build_full200_receipt(
    *, scene_list_path: Path, shard_manifest_paths: Sequence[Path]
) -> dict[str, Any]:
    """Build (but do not publish) the strict F0 full200 aggregate receipt."""

    if len(shard_manifest_paths) != EXPECTED_SHARDS:
        raise F0MergeError(f"exactly {EXPECTED_SHARDS} shard manifests are required")
    exact_list_path, exact_scenes, scene_list_hash = _read_scene_list(scene_list_path)

    loaded = []
    seen_manifest_paths: set[Path] = set()
    for supplied in shard_manifest_paths:
        path, manifest = _read_json(supplied, "F0 shard manifest")
        if path in seen_manifest_paths:
            raise F0MergeError(f"duplicate shard manifest: {path}")
        seen_manifest_paths.add(path)
        header = _validate_shard_header(
            manifest,
            path,
            scene_list_path=exact_list_path,
            scene_list_hash=scene_list_hash,
        )
        environment, environment_hash = _validate_environment(
            manifest, shard_index=header[0]
        )
        if manifest.get("environment_sha256") != environment_hash:
            raise F0MergeError(f"shard {header[0]} environment hash differs")
        checkpoint, sources = _validate_execution_identity(
            manifest, shard_index=header[0]
        )
        loaded.append(
            (
                header[0],
                path,
                manifest,
                header,
                environment,
                environment_hash,
                checkpoint,
                sources,
            )
        )
    loaded.sort(key=lambda row: row[0])
    if [row[0] for row in loaded] != list(range(EXPECTED_SHARDS)):
        raise F0MergeError("shard indices must be exactly [0, 1]")
    protocol_ids = {row[3][2] for row in loaded}
    if len(protocol_ids) != 1:
        raise F0MergeError("shards disagree on protocol id")
    protocol_id = next(iter(protocol_ids))
    shard_signatures = [row[3][3] for row in loaded]
    if len(set(shard_signatures)) != 1:
        raise F0MergeError("shards disagree on shared run signature")
    merged_signature = shard_signatures[0]

    per_index: dict[int, dict[str, Any]] = {}
    input_shards: list[dict[str, Any]] = []
    identities = {
        _canonical_json_sha256({"checkpoint": row[6], "sources": row[7]})
        for row in loaded
    }
    if len(identities) != 1:
        raise F0MergeError("shards disagree on checkpoint/source identity")
    for (
        shard_index,
        path,
        _manifest,
        header,
        environment,
        environment_hash,
        checkpoint,
        sources,
    ) in loaded:
        _, _, _, shard_signature, indices, order, scene_rows = header
        if order != [exact_scenes[index] for index in indices if 0 <= index < len(exact_scenes)]:
            raise F0MergeError(f"shard {shard_index} does not follow exact scene-list order")
        if any(index < 0 or index >= EXPECTED_SCENES for index in indices):
            raise F0MergeError(f"out-of-range scene index in shard {shard_index}")
        validated_shard_scenes: list[dict[str, Any]] = []
        for index, scene, row in zip(indices, order, scene_rows):
            if index in per_index:
                raise F0MergeError(f"duplicate scene coverage at index {index}")
            if row.get("scene_id") != scene or row.get("scene_index") != index:
                raise F0MergeError(f"manifest scene identity differs at index {index}")
            validated = _validate_scene_sidecar(
                row,
                path,
                expected_scene=scene,
                expected_index=index,
                expected_protocol_id=protocol_id,
                expected_run_signature=shard_signature,
                expected_environment_sha256=environment_hash,
                expected_device=str(environment["device"]),
                expected_checkpoint=checkpoint,
                expected_sources=sources,
            )
            per_index[index] = validated
            validated_shard_scenes.append(validated)
        shard_call_indices = [
            value
            for scene in validated_shard_scenes
            for value in scene["successful_call_indices"]
        ]
        shard_warmup_flags = [
            value
            for scene in validated_shard_scenes
            for value in scene["successful_warmup_flags"]
        ]
        if shard_call_indices != list(range(len(shard_call_indices))):
            raise F0MergeError(f"shard {shard_index} provider call indices are not contiguous")
        expected_warmup_flags = [
            index < SHARD_WARMUP_SUCCESSFUL_CALLS
            for index in range(len(shard_call_indices))
        ]
        if shard_warmup_flags != expected_warmup_flags:
            raise F0MergeError(f"shard {shard_index} warmup exclusion is not first-three global")
        rewarm_receipt = _validate_resume_rewarm(
            manifest=_manifest,
            manifest_scene_rows=scene_rows,
            validated_scenes=validated_shard_scenes,
            expected_device=str(environment["device"]),
            shard_index=shard_index,
        )
        declared_shard_keyframes = sum(
            _nonnegative_int(row.get("keyframe_count"), "shard scene keyframes")
            for row in scene_rows
        )
        if _manifest.get("shard_keyframe_count") != declared_shard_keyframes:
            raise F0MergeError(f"shard {shard_index} keyframe declaration differs")
        manifest_totals = _mapping(
            _manifest.get("totals"), f"shard {shard_index} aggregate totals"
        )
        recomputed_totals: dict[str, int] = {}
        for key in sorted({name for row in scene_rows for name in row["counts"]}):
            recomputed_totals[key] = sum(
                _nonnegative_int(row["counts"].get(key, 0), f"shard count {key}")
                for row in scene_rows
            )
        count_totals = {
            key: value
            for key, value in manifest_totals.items()
            if key
            not in {
                "candidate_scene_count",
                "cap_saturation_ratio",
                "provider_max_det_saturation_ratio",
            }
        }
        checked_manifest_totals = {
            key: _nonnegative_int(value, f"shard total {key}")
            for key, value in count_totals.items()
        }
        if checked_manifest_totals != recomputed_totals:
            raise F0MergeError(f"shard {shard_index} aggregate totals differ")
        expected_census_receipt = _mapping(
            _manifest.get("expected_execution_census"),
            f"shard {shard_index} expected execution census",
        )
        expected_shard_census = EXPECTED_SHARD_EXECUTION_COUNTS[shard_index]
        if (
            expected_census_receipt.get("sha256")
            != EXPECTED_EXECUTION_CENSUS_SHA256
            or expected_census_receipt.get("counts") != expected_shard_census
        ):
            raise F0MergeError(
                f"shard {shard_index} sealed execution census receipt differs"
            )
        observed_shard_census = {
            key: recomputed_totals.get(key) for key in expected_shard_census
        }
        if observed_shard_census != expected_shard_census:
            raise F0MergeError(
                f"shard {shard_index} observed execution census differs"
            )
        candidate_scenes = sum(
            _nonnegative_int(row["counts"]["accepted_lifts"], "accepted lifts") > 0
            for row in scene_rows
        )
        if _nonnegative_int(
            manifest_totals.get("candidate_scene_count"),
            f"shard {shard_index} candidate scene count",
        ) != candidate_scenes:
            raise F0MergeError(f"shard {shard_index} candidate scene total differs")
        successful = recomputed_totals["successful_frames"]
        expected_cap_ratio = (
            recomputed_totals["cap_saturated_frames"] / successful
            if successful
            else 0.0
        )
        actual_cap_ratio = _nonnegative_float(
            manifest_totals.get("cap_saturation_ratio"),
            f"shard {shard_index} cap saturation ratio",
        )
        if not math.isclose(actual_cap_ratio, expected_cap_ratio, abs_tol=1e-12, rel_tol=0.0):
            raise F0MergeError(f"shard {shard_index} cap saturation total differs")
        expected_provider_cap_ratio = (
            recomputed_totals["provider_max_det_saturated_frames"] / successful
            if successful
            else 0.0
        )
        actual_provider_cap_ratio = _nonnegative_float(
            manifest_totals.get("provider_max_det_saturation_ratio"),
            f"shard {shard_index} provider max_det saturation ratio",
        )
        if not math.isclose(
            actual_provider_cap_ratio,
            expected_provider_cap_ratio,
            abs_tol=1e-12,
            rel_tol=0.0,
        ):
            raise F0MergeError(
                f"shard {shard_index} provider max_det saturation total differs"
            )
        input_shards.append(
            {
                "index": shard_index,
                "path": os.fspath(path),
                "sha256": _sha256(path),
                "scene_count": len(indices),
                "environment_sha256": environment_hash,
                "device": environment["device"],
                "gpu_uuid": environment["gpu_uuid"],
                "run_signature_sha256": shard_signature,
                "resume_rewarm": rewarm_receipt,
                "checkpoint_sha256": checkpoint["sha256"],
                "sources_sha256": _canonical_json_sha256(sources),
                "execution_census": observed_shard_census,
            }
        )
    if sorted(per_index) != list(range(EXPECTED_SCENES)):
        missing = sorted(set(range(EXPECTED_SCENES)) - set(per_index))
        raise F0MergeError(f"incomplete full200 scene coverage; missing indices {missing}")

    ordered_scenes = [per_index[index] for index in range(EXPECTED_SCENES)]
    if [row["scene_id"] for row in ordered_scenes] != exact_scenes:
        raise F0MergeError("merged scene order differs from exact full200 list")
    signature_receipt = _validate_shared_run_signature(
        manifests=[row[2] for row in loaded],
        ordered_scenes=ordered_scenes,
        exact_scene_order=exact_scenes,
        scene_list_sha256=scene_list_hash,
        checkpoint=loaded[0][6],
        sources=loaded[0][7],
        expected_signature=merged_signature,
    )
    frame_keys: set[tuple[str, int]] = set()
    for row in ordered_scenes:
        for frame_id in row["frame_ids"]:
            key = (row["scene_id"], frame_id)
            if key in frame_keys:
                raise F0MergeError(f"duplicate merged frame key: {key}")
            frame_keys.add(key)
    total_keyframes = sum(row["keyframe_count"] for row in ordered_scenes)
    if total_keyframes != EXPECTED_KEYFRAMES or len(frame_keys) != EXPECTED_KEYFRAMES:
        raise F0MergeError(
            f"expected {EXPECTED_KEYFRAMES} unique keyframes, found {len(frame_keys)}"
        )
    execution_census_records = [
        record
        for scene in ordered_scenes
        for record in scene["execution_census_records"]
    ]
    execution_census_sha256 = _canonical_json_sha256(execution_census_records)
    if execution_census_sha256 != EXPECTED_EXECUTION_CENSUS_SHA256:
        raise F0MergeError("full200 execution census ledger SHA-256 differs")
    seen_non_upright = {
        tuple(key)
        for scene in ordered_scenes
        for key in scene["seen_non_upright_keyframes"]
    }
    if seen_non_upright != set(EXPECTED_NON_UPRIGHT_KEYFRAMES):
        raise F0MergeError("full200 non-upright abstention census differs")

    all_count_keys = sorted(
        {key for scene in ordered_scenes for key in scene["counts"]}
    )
    total_counts = {
        key: sum(scene["counts"].get(key, 0) for scene in ordered_scenes)
        for key in all_count_keys
    }
    sealed_execution_counts = {
        "keyframes": EXPECTED_KEYFRAMES,
        "invalid_pose_frames": EXPECTED_INVALID_CURRENT_POSE_FRAMES,
        "non_upright_producer_frames": EXPECTED_NON_UPRIGHT_FRAME_COUNT,
        "successful_frames": EXPECTED_SUCCESSFUL_PROVIDER_FRAMES,
    }
    for key, expected in sealed_execution_counts.items():
        if total_counts.get(key) != expected:
            raise F0MergeError(
                f"full200 sealed execution count {key} differs: "
                f"{total_counts.get(key)} != {expected}"
            )
    accepted_lifts = total_counts["accepted_lifts"]
    accepted_scene_count = sum(
        scene["counts"]["accepted_lifts"] > 0 for scene in ordered_scenes
    )
    successful_frames = total_counts["successful_frames"]
    cap_saturated_frames = total_counts["cap_saturated_frames"]
    cap_fraction = (
        cap_saturated_frames / successful_frames if successful_frames else 1.0
    )
    provider_max_det_saturated_frames = sum(
        scene["provider_max_det_saturated_frames"] for scene in ordered_scenes
    )
    if (
        total_counts.get("provider_max_det_saturated_frames")
        != provider_max_det_saturated_frames
    ):
        raise F0MergeError("full200 provider max_det saturation count differs")
    provider_max_det_saturated_fraction = (
        provider_max_det_saturated_frames / successful_frames
        if successful_frames
        else 0.0
    )
    quality_histogram_names = tuple(ordered_scenes[0]["quality_histograms"])
    quality_histograms = {
        name: _merge_histograms(
            [scene["quality_histograms"][name] for scene in ordered_scenes]
        )
        for name in quality_histogram_names
    }
    if quality_histograms["raw_confidence"]["sample_count"] != total_counts["raw_masks"]:
        raise F0MergeError("raw-mask quality histogram count differs from raw masks")
    for name in ("mask_pixel_area", "valid_depth_ratio", "residual_ratio"):
        if quality_histograms[name]["sample_count"] != total_counts["raw_masks"]:
            raise F0MergeError(f"{name} histogram count differs from raw masks")
    for name in (
        "raw_masks_per_successful_frame",
        "selected_lifts_per_successful_frame",
    ):
        if quality_histograms[name]["sample_count"] != successful_frames:
            raise F0MergeError(f"{name} histogram count differs from successful frames")
    if any(histogram["overflow_count"] for histogram in quality_histograms.values()):
        raise F0MergeError("bounded no-GT quality histogram unexpectedly overflowed")

    provider_samples = [
        value for scene in ordered_scenes for value in scene["provider_samples_ms"]
    ]
    complete_samples = [
        value for scene in ordered_scenes for value in scene["complete_samples_ms"]
    ]
    receipt_total_samples = [
        value
        for scene in ordered_scenes
        for value in scene["receipt_total_samples_ms"]
    ]
    if not provider_samples or not complete_samples:
        raise F0MergeError("merged provider or complete runtime samples are empty")
    expected_measured = successful_frames - total_counts["warmup_excluded_successful_frames"]
    if len(provider_samples) != expected_measured or len(complete_samples) != expected_measured:
        raise F0MergeError("measured runtime sample count differs from warmup contract")
    if len(receipt_total_samples) != total_keyframes:
        raise F0MergeError("receipt-total runtime sample count differs from keyframes")
    provider_distribution = _distribution(provider_samples)
    complete_distribution = _distribution(complete_samples)
    receipt_total_distribution = _distribution(receipt_total_samples)
    amortized_ms = float(complete_distribution["mean"]) / SOURCE_FRAME_STRIDE
    cpu_peak = max(scene["cpu_peak_rss_bytes"] for scene in ordered_scenes)
    gpu_peak = max(scene["gpu_peak_memory_bytes"] for scene in ordered_scenes)
    if gpu_peak <= 0:
        raise F0MergeError("CUDA peak memory is zero; CPU-only receipts cannot pass F0")

    gates = {
        "accepted_lifts": _gate(accepted_lifts, ">=", MIN_ACCEPTED_LIFTS),
        "accepted_scene_coverage": _gate(
            accepted_scene_count, ">=", MIN_ACCEPTED_SCENES
        ),
        "cap_saturated_successful_frame_fraction": _gate(
            cap_fraction, "<=", MAX_CAP_SATURATED_SUCCESS_FRACTION
        ),
        "provider_runtime_p95_ms": _gate(
            float(provider_distribution["p95"]), "<=", MAX_PROVIDER_P95_MS
        ),
        "complete_runtime_p95_ms": _gate(
            float(complete_distribution["p95"]), "<=", MAX_COMPLETE_P95_MS
        ),
        "complete_runtime_max_ms": _gate(
            float(complete_distribution["max"]), "<", MAX_COMPLETE_MS_EXCLUSIVE
        ),
        "amortized_complete_ms_per_source_frame": _gate(
            amortized_ms, "<=", MAX_AMORTIZED_MS_PER_SOURCE_FRAME
        ),
        "gpu_peak_memory_bytes": _gate(gpu_peak, "<=", MAX_GPU_PEAK_BYTES),
    }

    scene_receipts = [
        {
            "scene_id": scene["scene_id"],
            "scene_index": scene["scene_index"],
            "keyframe_count": scene["keyframe_count"],
            "counts": scene["counts"],
            "runtime_sample_count": len(scene["provider_samples_ms"]),
            "cpu_peak_rss_bytes": scene["cpu_peak_rss_bytes"],
            "gpu_peak_memory_bytes": scene["gpu_peak_memory_bytes"],
            "sidecar": scene["sidecar"],
        }
        for scene in ordered_scenes
    ]
    return {
        "schema": SCHEMA,
        "complete": True,
        "protocol_id": protocol_id,
        "run_signature_sha256": merged_signature,
        "shard_run_signatures_sha256": shard_signatures,
        "contracts": {
            "shadow_only": True,
            "no_output_affecting": True,
            "birth_enabled": False,
            "ground_truth_access": False,
            "annotation_access": False,
            "evaluator_access": False,
            "terminal_native_prediction_access": False,
            "terminal_native_prediction_mutation": False,
            "terminal_prediction_pickle_write": False,
            "cutr_current_pred_boxes_access": True,
            "cutr_nonbox_field_use": False,
            "cutr_payload_deserialization_scope": "full_safe_payload",
            "clip_or_semantic_use": False,
            "tracking_or_history": False,
            "training": False,
            "online_learning": False,
            "external_pretraining_frozen": True,
            "current_pose_required_no_forward_fill": True,
            "prediction_pickle_inputs": False,
        },
        "scene_list": {
            "path": os.fspath(exact_list_path),
            "sha256": scene_list_hash,
            "exact_scene_count": EXPECTED_SCENES,
        },
        "inputs": {
            "shards": input_shards,
            "shared_run_signature": signature_receipt,
        },
        "coverage": {
            "scene_count": len(ordered_scenes),
            "scene_order": exact_scenes,
            "unique_keyframe_count": len(frame_keys),
            "expected_keyframe_count": EXPECTED_KEYFRAMES,
            "duplicate_frame_count": 0,
            "sealed_execution_census": sealed_execution_counts,
            "sealed_execution_census_sha256": execution_census_sha256,
        },
        "counts": total_counts,
        "capacity": {
            "accepted_lifts": accepted_lifts,
            "accepted_scene_count": accepted_scene_count,
            "successful_frames": successful_frames,
            "cap_saturated_frames": cap_saturated_frames,
            "cap_saturated_successful_frame_fraction": cap_fraction,
            "provider_max_det": PROVIDER_MAX_DET,
            "provider_max_det_saturated_frames": provider_max_det_saturated_frames,
            "provider_max_det_saturated_successful_frame_fraction": (
                provider_max_det_saturated_fraction
            ),
            "accepted_lifts_per_scene_histogram": _fixed_histogram(
                (scene["counts"]["accepted_lifts"] for scene in ordered_scenes),
                LIFTS_PER_SCENE_HISTOGRAM_UPPER_BOUNDS,
            ),
        },
        "no_gt_mask_quality_histograms": {
            "raw_fastsam_confidence": quality_histograms["raw_confidence"],
            "mask_pixel_area": quality_histograms["mask_pixel_area"],
            "valid_depth_ratio": quality_histograms["valid_depth_ratio"],
            "residual_ratio": quality_histograms["residual_ratio"],
            "raw_mask_count_per_successful_frame": quality_histograms[
                "raw_masks_per_successful_frame"
            ],
            "selected_lifts_per_successful_frame": quality_histograms[
                "selected_lifts_per_successful_frame"
            ],
            "raw_samples_included": False,
            "ground_truth_used": False,
        },
        "runtime": {
            "samples_ms": {
                "provider": provider_samples,
                "complete": complete_samples,
                "receipt_total": receipt_total_samples,
            },
            "provider_ms": {
                "distribution": provider_distribution,
                "histogram": _fixed_histogram(
                    provider_samples, RUNTIME_HISTOGRAM_UPPER_BOUNDS_MS
                ),
            },
            "complete_ms": {
                "distribution": complete_distribution,
                "histogram": _fixed_histogram(
                    complete_samples, RUNTIME_HISTOGRAM_UPPER_BOUNDS_MS
                ),
            },
            "receipt_total_ms": {
                "distribution": receipt_total_distribution,
                "histogram": _fixed_histogram(
                    receipt_total_samples, RUNTIME_HISTOGRAM_UPPER_BOUNDS_MS
                ),
                "gate_input": False,
            },
            "source_frame_stride": int(SOURCE_FRAME_STRIDE),
            "amortized_complete_ms_per_source_frame": amortized_ms,
        },
        "memory": {
            "cpu_peak_rss_bytes": cpu_peak,
            "gpu_peak_memory_bytes": gpu_peak,
            "gpu_peak_limit_bytes": MAX_GPU_PEAK_BYTES,
        },
        "gates": gates,
        "overall_pass": all(gate["passed"] for gate in gates.values()),
        "scenes": scene_receipts,
    }


def _json_payload(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def publish_create_only(output_dir: Path, receipt: Mapping[str, Any]) -> Path:
    """Atomically create the final named receipt without replacing anything."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise F0MergeError(f"output directory must be a non-symlink directory: {output_dir}")
    output_dir = output_dir.resolve()
    target = output_dir / OUTPUT_NAME
    if target.exists() or target.is_symlink():
        raise F0MergeError(f"create-only output already exists: {target}")
    payload = _json_payload(receipt)
    temporary: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=f".{OUTPUT_NAME}.", suffix=".tmp", dir=output_dir
        )
        temporary = Path(raw_path)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise F0MergeError(f"create-only output raced with an existing file: {target}") from error
        directory_fd = os.open(output_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument(
        "--shard-manifest", type=Path, action="append", required=True,
        help="repeat exactly twice, once for each F0 shard",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = build_full200_receipt(
        scene_list_path=args.scene_list,
        shard_manifest_paths=args.shard_manifest,
    )
    target = publish_create_only(args.output_dir, receipt)
    print(
        json.dumps(
            {
                "output": os.fspath(target),
                "overall_pass": receipt["overall_pass"],
                "accepted_lifts": receipt["capacity"]["accepted_lifts"],
                "accepted_scene_count": receipt["capacity"]["accepted_scene_count"],
                "provider_p95_ms": receipt["runtime"]["provider_ms"]["distribution"]["p95"],
                "complete_p95_ms": receipt["runtime"]["complete_ms"]["distribution"]["p95"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
