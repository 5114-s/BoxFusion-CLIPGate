#!/usr/bin/env python3
"""Independent create-only merge and replay auditor for frozen N0a extra100.

This program is deliberately separate from the N0a producer.  It authenticates
every shard, scene receipt, evidence array, and frozen-input ledger before it
is allowed to issue the protocol's final decision.  Production replay is run
in a fresh Python process on the same physical GPU: all sources in scene
indices 100--149 are replayed, and later sources are selected by the frozen
SHA-256 predicate.  A selected source always causes its complete original
frame batch to be executed; only the selected subset is compared.

The program has no ground-truth, evaluator, native-prediction, class, CLIP,
training, tracking, or output-mutation input.  Its only persistent output is a
create-only JSON audit receipt.  Its authorizing future-only gate runs actual
SAM2/core replay while explicitly adding, altering, and deleting future-named
files in a private temporary mirror; real ScanNet files remain read-only.  A
smaller in-memory causal fixture is separately labelled non-authorizing.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence
import warnings

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

PROTOCOL_ID = "N0A-FROZEN-SAM2-IMAGE-BOXPROMPT-MASKLIFT-EXTRA100-SHADOW"
SCENE_SCHEMA = "boxfusion.scannet_sam2_image_masklift_n0a_extra100.scene.v2"
SHARD_SCHEMA = "boxfusion.scannet_sam2_image_masklift_n0a_extra100.shard.v2"
EVIDENCE_SCHEMA = "boxfusion.scannet_sam2_image_masklift_n0a_extra100.evidence.v1"
MERGE_SCHEMA = "boxfusion.scannet_sam2_image_masklift_n0a_extra100.merge.v2"
WORKER_SCHEMA = "boxfusion.scannet_sam2_image_masklift_n0a_extra100.replay_worker.v2"
FUTURE_CASE_SCHEMA = (
    "boxfusion.scannet_sam2_image_masklift_n0a_extra100.future_case.v1"
)
CORE_SCHEMA = "boxfusion.sam2_image_masklift_n0a.v1"
EXPECTED_F0_SCENE_SCHEMA = "boxfusion.scannet_fastsam_f0_full200.scene.v1"
EXPECTED_F0_MERGE_SCHEMA = "boxfusion.scannet_fastsam_f0_full200.merge.v1"
EXPECTED_F0_PROTOCOL = "F0-frozen-FastSAM-x-residual-automatic-mask-shadow-full200"

EXPECTED_PROTOCOL_SHA256 = (
    "a0cf925eac638993f7458d6f4debd79b0113553bd5174eb544d4eea9334f307b"
)
EXPECTED_CORE_SHA256 = (
    "80897a977d5694fec6322dadfd94dd6f6fb1bdf9af87b6055b4939df5bf4dced"
)
EXPECTED_PROVIDER_SHA256 = (
    "cbe1ba2cdb0853f49b2ab780c9feb0cea72a9f926700e82882857cc361f6f32e"
)
EXPECTED_RUNNER_SHA256 = (
    "6b1587d0db19b0154e6e6c91aaadb46046b556735870309f2eac14c611e24317"
)
EXPECTED_F0_RECEIPT_SHA256 = (
    "07249ead31ad150cb43d7a35f4c922ac70a8a2f95bcf0fcd24f61f944c1e58a1"
)
EXPECTED_FULL200_SCENE_LIST_SHA256 = (
    "0e7e722d3e93ec4b721f12293a3f1e98ca62d475b42cc8b9d491878a897e9bd1"
)
EXPECTED_EXTRA100_SCENE_LIST_SHA256 = (
    "f28e6997b2f50799020cf827edfe6a1520b4afe8e17de7c5564004208b8a2287"
)
EXPECTED_FRAME_LEDGER_SHA256 = (
    "f4fa82ce8a1513262fe10278eed54a33874df00c1cea0964c8afb3945b137818"
)
EXPECTED_SOURCE_LEDGER_SHA256 = (
    "1f03cc600de29930d3b314588326f35a7f0fcd995ab2700341a2469d8bbbcb00"
)
EXPECTED_SIDECAR_LEDGER_SHA256 = (
    "0471aa066706ed6ccd17da58bf986fb3d7434d65833c5d01d23dcac976957834"
)

EXPECTED_COHORT_START = 100
EXPECTED_SCENES = 100
EXPECTED_KEYFRAMES = 6_124
EXPECTED_SUCCESSFUL_FRAMES = 5_984
EXPECTED_SOURCES = 46_090
EXPECTED_PROVIDER_FORWARDS = 5_739
EXPECTED_SUCCESSFUL_EMPTY_FRAMES = 245
EXPECTED_AUTHENTICATED_WARNINGS = 11_478
FIRST_FULL_REPLAY_SCENES = 50
REPLAY_SELECTOR_LIMIT = 0x0290
MASK_PACKED_BYTES = 480 * 640 // 8
MAX_STORED_POINTS = 2_048
SOURCE_FRAME_STRIDE = 25.0
MAX_CUDA_BYTES = 4 * 1024**3
FUTURE_CASES = (
    "baseline",
    "referenced_future_changed",
    "unreferenced_future_added",
    "unreferenced_future_altered",
    "unreferenced_future_deleted",
)

SCENE_COUNT_KEYS = (
    "keyframe_count",
    "successful_frame_count",
    "source_count",
    "provider_forward_count",
    "valid_hs_count",
    "invalid_hs_count",
    "nontrivial_hs_count",
    "authenticated_warning_count",
)
SEALED_CENSUS_KEYS = (
    "keyframe_count",
    "successful_frame_count",
    "source_count",
    "provider_forward_count",
)
SOURCE_ROW_KEYS = {
    "scene_index", "scene_id", "frame_ordinal", "frame_id", "rank",
    "raw_index", "mask_sha256", "points_and_voxel_keys_sha256",
    "source_id", "prompt_tight_box_xyxy", "sam2", "n0a_receipt",
    "nontrivial_vs_h0", "h0_hs_iou3d", "maximum_face_displacement_m",
    "evidence_index", "point_offset", "source_lineage_sha256",
}
EVIDENCE_ARRAY_NAMES = {
    "schema_utf8",
    "mask_packbits",
    "points_world",
    "voxel_keys",
    "point_offsets",
    "frame_ordinals",
    "frame_ids",
    "ranks",
    "raw_indices",
    "selected_hypothesis_indices",
    "predicted_ious",
    "all_predicted_ious",
    "result_sha256_ascii",
}
RUNNER_PENDING_DECISION = "awaiting_complete_extra100_merge_and_mandatory_replay"

WARNING_POLICY_ID = "N0A-WARN-V2-EXACT-2XCUMSUM-POSENC-143-144"
EXPECTED_WARNING_SOURCE = Path(
    "/data/ZhaoX/OVM3D-Dett/third_party/Grounded-SAM-2/"
    "sam2/modeling/position_encoding.py"
)
EXPECTED_WARNING_SOURCE_RELATIVE = "sam2/modeling/position_encoding.py"
EXPECTED_WARNING_SOURCE_SHA256 = (
    "14ae89d7ae68f61e2ffcba09eb171d8df9a7298332d4da99036d703294f89ec1"
)
EXPECTED_WARNING_LINES = (143, 144)
EXPECTED_WARNING_MESSAGE = (
    "cumsum_cuda_kernel does not have a deterministic implementation, but you "
    "set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can "
    "file an issue at https://github.com/pytorch/pytorch/issues to help us "
    "prioritize adding deterministic support for this operation. (Triggered "
    "internally at ../aten/src/ATen/Context.cpp:91.)"
)
EXPECTED_WARNING_MESSAGE_SHA256 = (
    "ed71c50715686ffdf28200dc9deb5f46c8d1f641a112c5050777b9401be90fd8"
)

EXPECTED_SAM2_SOURCE_ROOT = Path(
    "/data/ZhaoX/OVM3D-Dett/third_party/Grounded-SAM-2"
)
EXPECTED_SAM2_CHECKPOINT = (
    EXPECTED_SAM2_SOURCE_ROOT / "checkpoints/sam2.1_hiera_large.pt"
)
EXPECTED_SAM2_CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_l.yaml"
EXPECTED_SAM2_SOURCE_TREE_SHA256 = (
    "cc5a594bab1508ab69cbedfbb83ba8e226f848dd142a3deba8c195ee1e2469cf"
)
EXPECTED_SAM2_CONFIG_SHA256 = (
    "545e4325aa5c19a1615d43c946b07276ed4c57214eacf1437e38fa3d9374f636"
)
EXPECTED_SAM2_CHECKPOINT_BYTES = 898_083_611
EXPECTED_SAM2_CHECKPOINT_SHA256 = (
    "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
)
EXPECTED_SAM2_CRITICAL_SOURCE_SHA256: Mapping[str, str] = {
    "sam2/sam2_image_predictor.py":
        "f13e5f9d94e5c8d9d2c3622dab20c8f334c089ef2ee5ea8e199da7d332b029ba",
    "sam2/build_sam.py":
        "bc49ac8e9ebf871790fa2e5f0e70bd5734e010966eff66af0c350ceaf14f3f1e",
    "sam2/modeling/sam2_base.py":
        "6d81450e897d0735f9be369771f2b3fb6eadb90dd3f3ac16b3f7a8c8eb1a052a",
    "sam2/utils/transforms.py":
        "ba3a64f4600c62f209206a6df3b40e3fcf133edae32fad658831bb0c2a6d1146",
    EXPECTED_WARNING_SOURCE_RELATIVE: EXPECTED_WARNING_SOURCE_SHA256,
}

EXPECTED_PROVIDER_CONFIG: Mapping[str, Any] = {
    "source_root": os.fspath(EXPECTED_SAM2_SOURCE_ROOT),
    "config_name": EXPECTED_SAM2_CONFIG_NAME,
    "checkpoint_path": os.fspath(EXPECTED_SAM2_CHECKPOINT),
    "source_file_glob": "sam2/**/*.py",
    "source_file_count": 23,
    "source_tree_sha256": EXPECTED_SAM2_SOURCE_TREE_SHA256,
    "config_sha256": EXPECTED_SAM2_CONFIG_SHA256,
    "checkpoint_bytes": EXPECTED_SAM2_CHECKPOINT_BYTES,
    "checkpoint_sha256": EXPECTED_SAM2_CHECKPOINT_SHA256,
    "device": "cuda",
    "apply_postprocessing": True,
    "autocast_dtype": "bfloat16",
    "multimask_output": True,
    "return_logits": False,
    "normalize_coords": True,
    "mask_threshold": 0.0,
    "max_boxes_per_frame": 16,
    "multimask_hypotheses": 3,
}

EXPECTED_ENVIRONMENT_VERSIONS: Mapping[str, str] = {
    "python": "3.10.19",
    "torch": "2.5.1+cu121",
    "torchvision": "0.20.1+cu121",
    "numpy": "2.2.6",
    "opencv": "4.13.0",
    "hydra": "1.3.2",
    "omegaconf": "2.3.0",
    "pillow": "12.0.0",
}
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 3090"
EXPECTED_GPU_COMPUTE_CAPABILITY = [8, 6]
EXPECTED_GPU_TOTAL_MEMORY_BYTES = 25_429_606_400
EXPECTED_DETERMINISM_RECEIPT: Mapping[str, Any] = {
    "seed": 0,
    "pythonhashseed": "0",
    "cublas_workspace_config": ":4096:8",
    "deterministic_algorithms": True,
    "deterministic_algorithms_warn_only": True,
    "registered_nondeterministic_warning": "aten::cumsum_cuda",
    "warning_policy_id": WARNING_POLICY_ID,
    "expected_warning_count_per_nonempty_forward": 2,
    "bitwise_replay_required": True,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cuda_matmul_tf32": False,
    "cudnn_tf32": False,
}

MIN_VALID_SOURCE_COUNT = 36_872
MIN_VALID_SCENE_COUNT = 90
MIN_NONTRIVIAL_SOURCE_COUNT = 1_440
MIN_NONTRIVIAL_SCENE_COUNT = 50

DISCARD_DECISION = "discard_n0a_contract_or_determinism_failure"
CAPACITY_STOP_DECISION = "stop_n0a_insufficient_valid_or_distinct_geometry"
RUNTIME_STOP_DECISION = "stop_n0a_realtime_gate_failed"
RETAIN_DECISION = "retain_n0a_for_n0b_and_one_sealed_capacity_evaluation_only"

PROTOCOL_PATH = (
    REPOSITORY_ROOT / "docs/N0A_SAM2_IMAGE_MASKLIFT_EXTRA100_PROTOCOL_FREEZE.md"
)

CONTRACTS: Mapping[str, bool] = {
    "shadow_only": True,
    "birth_enabled": False,
    "active_authorized": False,
    "native_prediction_access": False,
    "native_output_mutation": False,
    "ground_truth_access": False,
    "gt_access": False,
    "annotation_access": False,
    "evaluator_access": False,
    "future_frame_access": False,
    "history_or_tracking": False,
    "class_clip_or_semantic_use": False,
    "training": False,
    "online_learning": False,
}


class N0AAuditError(RuntimeError):
    """A receipt, ledger, replay, frozen asset, or audit contract differed."""


@dataclass(frozen=True)
class AuditExpectations:
    """Census/ledger expectations; production defaults are immutable."""

    cohort_start: int = EXPECTED_COHORT_START
    scene_count: int = EXPECTED_SCENES
    keyframe_count: int = EXPECTED_KEYFRAMES
    successful_frame_count: int = EXPECTED_SUCCESSFUL_FRAMES
    source_count: int = EXPECTED_SOURCES
    scene_list_sha256: str = EXPECTED_EXTRA100_SCENE_LIST_SHA256
    frame_ledger_sha256: str = EXPECTED_FRAME_LEDGER_SHA256
    source_ledger_sha256: str = EXPECTED_SOURCE_LEDGER_SHA256
    sidecar_ledger_sha256: str = EXPECTED_SIDECAR_LEDGER_SHA256
    first_full_replay_scenes: int = FIRST_FULL_REPLAY_SCENES


@dataclass
class SourceRecord:
    scene_position: int
    scene_index: int
    scene_id: str
    frame_ordinal: int
    frame_id: int
    rank: int
    raw_index: int
    source_id: str
    identity: dict[str, Any]
    prompt_box: list[float]
    h0: dict[str, Any]
    expected_selected_index: int
    expected_selected_iou_bytes: bytes
    expected_all_iou_bytes: bytes
    expected_mask_sha256: str
    expected_result_sha256: str
    expected_valid: bool
    expected_abstention_reason: str | None
    expected_nontrivial: bool


@dataclass
class FrameRecord:
    scene_position: int
    scene_index: int
    scene_id: str
    frame_ordinal: int
    frame_id: int
    rgb: dict[str, str]
    depth: dict[str, str]
    pose: dict[str, str]
    intrinsic: dict[str, str]
    sources: list[SourceRecord]


@dataclass
class AuditBundle:
    manifest_paths: list[Path]
    manifests: list[dict[str, Any]]
    frames: list[FrameRecord]
    input_seals: dict[str, str]
    provider_config: dict[str, Any]
    environment_receipt: dict[str, Any]
    gpu_uuid: str
    counts: dict[str, int]
    valid_scene_count: int
    nontrivial_scene_count: int
    warm_incremental_ms: list[float]
    warm_composed_ms: list[float]
    deadline_misses: int
    cuda_peak_memory_bytes: int
    runner_decisions: list[str]
    ledger_hashes: dict[str, str]
    source_receipt_hashes: dict[str, str]
    future_fixture_source: dict[str, Any]
    audit_notes: list[str] = field(default_factory=list)


def _canonical_json_bytes(value: object, *, sort_keys: bool = True) -> bytes:
    return json.dumps(
        value,
        sort_keys=sort_keys,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_json_sha256(value: object, *, sort_keys: bool = True) -> str:
    return hashlib.sha256(_canonical_json_bytes(value, sort_keys=sort_keys)).hexdigest()


def _created_json_sha256(value: Mapping[str, Any]) -> str:
    """Hash the exact pretty JSON bytes emitted by ``_atomic_create_json``."""

    rendered = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    return hashlib.sha256(rendered.encode("ascii")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _regular_file(path: Path, label: str, suffix: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise N0AAuditError(f"{label} must be a regular non-symlink file: {path}")
    result = path.resolve()
    if suffix is not None and result.suffix.lower() != suffix:
        raise N0AAuditError(f"{label} must be a {suffix} file: {result}")
    if result.suffix.lower() in {".pkl", ".pickle"}:
        raise N0AAuditError(f"prediction pickle input is forbidden: {result}")
    return result


def _read_json(path: Path, label: str) -> tuple[Path, dict[str, Any]]:
    source = _regular_file(path, label, ".json")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise N0AAuditError(f"invalid {label}: {source}") from error
    if not isinstance(value, dict):
        raise N0AAuditError(f"{label} must contain one JSON object")
    return source, value


def _content_hash_valid(value: Mapping[str, Any]) -> bool:
    payload = dict(value)
    claimed = payload.pop("content_sha256", None)
    return _valid_sha256(claimed) and claimed == _canonical_json_sha256(payload)


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
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
            raise N0AAuditError(f"refusing to overwrite audit output: {path}") from error
        return _sha256(path)
    finally:
        temporary.unlink(missing_ok=True)


def _seal_reference(
    value: object,
    label: str,
    seals: dict[str, str],
    suffix: str | None = None,
) -> Path:
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("path"), str)
        or not _valid_sha256(value.get("sha256"))
    ):
        raise N0AAuditError(f"{label} seal is absent")
    path = _regular_file(Path(str(value["path"])), label, suffix)
    actual = _sha256(path)
    if actual != value["sha256"]:
        raise N0AAuditError(f"{label} rehash differs")
    key = os.fspath(path)
    previous = seals.get(key)
    if previous is not None and previous != actual:
        raise N0AAuditError(f"conflicting hashes for {label}: {path}")
    seals[key] = actual
    return path


def _snapshot_hash(seals: Mapping[str, str]) -> str:
    return _canonical_json_sha256([[path, seals[path]] for path in sorted(seals)])


def _rehash_snapshot(seals: Mapping[str, str]) -> tuple[bool, list[str], str]:
    changed: list[str] = []
    rows: list[list[str]] = []
    for name in sorted(seals):
        try:
            path = _regular_file(Path(name), "global audit rehash")
            actual = _sha256(path)
        except (N0AAuditError, OSError):
            actual = "unavailable"
        rows.append([name, actual])
        if actual != seals[name]:
            changed.append(name)
    return not changed, changed, _canonical_json_sha256(rows)


def _exact_contracts(value: object, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(CONTRACTS):
        raise N0AAuditError(f"{label} contracts are absent")
    for key, expected in CONTRACTS.items():
        if value.get(key) is not expected:
            raise N0AAuditError(f"{label} contract differs: {key}")
    forbidden_true = {
        "birth_enabled",
        "active_authorized",
        "native_prediction_access",
        "native_output_mutation",
        "ground_truth_access",
        "gt_access",
        "annotation_access",
        "evaluator_access",
        "future_frame_access",
        "history_or_tracking",
        "class_clip_or_semantic_use",
        "training",
        "online_learning",
    }
    if any(value.get(key) is True for key in forbidden_true):
        raise N0AAuditError(f"{label} enables a forbidden contract")


def _expected_warning_policy() -> dict[str, Any]:
    return {
        "policy_id": WARNING_POLICY_ID,
        "expected_count_per_nonempty_forward": 2,
        "category": "builtins.UserWarning",
        "message_type": "builtins.UserWarning",
        "source_path": os.fspath(EXPECTED_WARNING_SOURCE),
        "source_relative_path": EXPECTED_WARNING_SOURCE_RELATIVE,
        "source_sha256": EXPECTED_WARNING_SOURCE_SHA256,
        "ordered_lines": list(EXPECTED_WARNING_LINES),
        "message_sha256": EXPECTED_WARNING_MESSAGE_SHA256,
    }


def _expected_warning_evidence() -> dict[str, Any]:
    return {
        "policy_id": WARNING_POLICY_ID,
        "count": len(EXPECTED_WARNING_LINES),
        "ordered_lines": list(EXPECTED_WARNING_LINES),
        "source_sha256": EXPECTED_WARNING_SOURCE_SHA256,
        "message_sha256": EXPECTED_WARNING_MESSAGE_SHA256,
    }


def _validate_exact_warning_rows(
    caught: Sequence[object], *, label: str
) -> dict[str, Any]:
    """Authenticate the complete frozen two-warning tuple for one forward."""

    if len(caught) != len(EXPECTED_WARNING_LINES):
        raise N0AAuditError(f"{label} must emit exactly two warnings")
    for row, expected_line in zip(caught, EXPECTED_WARNING_LINES):
        category = getattr(row, "category", None)
        message = getattr(row, "message", None)
        filename = getattr(row, "filename", None)
        lineno = getattr(row, "lineno", None)
        if category is not UserWarning or type(message) is not UserWarning:
            raise N0AAuditError(f"{label} warning category/message type differs")
        if str(message) != EXPECTED_WARNING_MESSAGE:
            raise N0AAuditError(f"{label} warning message differs")
        if filename != os.fspath(EXPECTED_WARNING_SOURCE):
            raise N0AAuditError(f"{label} warning source path differs")
        source = _regular_file(
            Path(str(filename)), f"{label} warning source", ".py"
        )
        if (
            source != EXPECTED_WARNING_SOURCE
            or _sha256(source) != EXPECTED_WARNING_SOURCE_SHA256
            or lineno != expected_line
        ):
            raise N0AAuditError(f"{label} warning source identity/line differs")
    return _expected_warning_evidence()


def _validate_provider_config(value: object, label: str) -> dict[str, Any]:
    """Validate the receipt against frozen constants before opening its paths."""

    if not isinstance(value, Mapping) or dict(value) != dict(EXPECTED_PROVIDER_CONFIG):
        raise N0AAuditError(f"{label} provider configuration differs")
    return dict(value)


def _validate_environment_receipt(
    value: object, *, provider_config: Mapping[str, Any], label: str
) -> str:
    """Authenticate one producer shard's complete frozen environment receipt."""

    if not isinstance(value, Mapping) or set(value) != {
        "preflight", "platform", "cuda_visible_devices"
    }:
        raise N0AAuditError(f"{label} environment receipt differs")
    if not isinstance(value.get("platform"), str) or not value["platform"]:
        raise N0AAuditError(f"{label} platform receipt differs")
    visible = value.get("cuda_visible_devices")
    if visible is not None and not isinstance(visible, str):
        raise N0AAuditError(f"{label} CUDA visibility receipt differs")
    preflight = value.get("preflight")
    if not isinstance(preflight, Mapping) or set(preflight) != {
        "conda_environment", "versions", "gpu", "determinism", "provider_config"
    }:
        raise N0AAuditError(f"{label} production preflight receipt differs")
    if (
        preflight.get("conda_environment") != "gsam2_env"
        or preflight.get("versions") != dict(EXPECTED_ENVIRONMENT_VERSIONS)
        or preflight.get("determinism") != dict(EXPECTED_DETERMINISM_RECEIPT)
        or preflight.get("provider_config") != dict(provider_config)
    ):
        raise N0AAuditError(f"{label} frozen environment policy differs")
    gpu = preflight.get("gpu")
    if not isinstance(gpu, Mapping) or set(gpu) != {
        "logical_index", "physical_index", "uuid", "name",
        "compute_capability", "total_memory_bytes",
    }:
        raise N0AAuditError(f"{label} GPU receipt differs")
    if (
        isinstance(gpu.get("logical_index"), bool)
        or not isinstance(gpu.get("logical_index"), int)
        or gpu["logical_index"] < 0
        or isinstance(gpu.get("physical_index"), bool)
        or not isinstance(gpu.get("physical_index"), int)
        or gpu["physical_index"] < 0
        or not isinstance(gpu.get("uuid"), str)
        or not gpu["uuid"].startswith("GPU-")
        or gpu.get("name") != EXPECTED_GPU_NAME
        or gpu.get("compute_capability") != EXPECTED_GPU_COMPUTE_CAPABILITY
        or isinstance(gpu.get("total_memory_bytes"), bool)
        or not isinstance(gpu.get("total_memory_bytes"), int)
        or gpu["total_memory_bytes"] != EXPECTED_GPU_TOTAL_MEMORY_BYTES
    ):
        raise N0AAuditError(f"{label} frozen GPU identity differs")
    if visible is None or not visible.strip():
        visible_token = str(gpu["logical_index"])
    else:
        tokens = [token.strip() for token in visible.split(",")]
        if gpu["logical_index"] >= len(tokens):
            raise N0AAuditError(f"{label} CUDA visibility mapping differs")
        visible_token = tokens[gpu["logical_index"]]
    if (
        visible_token.isdigit()
        and int(visible_token) != gpu["physical_index"]
    ) or (
        visible_token.startswith("GPU-") and visible_token != gpu["uuid"]
    ) or not (visible_token.isdigit() or visible_token.startswith("GPU-")):
        raise N0AAuditError(f"{label} CUDA visibility mapping differs")
    return str(gpu["uuid"])


def replay_sample_selected(source_id: str) -> bool:
    """Return the exact frozen post-prefix 1% sample predicate."""

    try:
        encoded = source_id.encode("ascii")
    except UnicodeEncodeError as error:
        raise N0AAuditError("source_id is not ASCII") from error
    prefix = int.from_bytes(hashlib.sha256(encoded).digest()[:2], "big")
    return prefix < REPLAY_SELECTOR_LIMIT


def _source_id(scene_id: str, frame_id: int, raw_index: int) -> str:
    return f"{scene_id}/frame_{frame_id:06d}/raw_{raw_index:03d}"


def _little_bytes(value: object, dtype: str, shape: tuple[int, ...], label: str) -> bytes:
    try:
        array = np.asarray(value, dtype=np.dtype(dtype))
    except (TypeError, ValueError, OverflowError) as error:
        raise N0AAuditError(f"{label} cannot be converted") from error
    if array.shape != shape or not np.isfinite(array).all():
        raise N0AAuditError(f"{label} shape or values differ")
    return np.ascontiguousarray(array, dtype=np.dtype(dtype)).tobytes(order="C")


def _result_payload_from_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    mask = receipt.get("mask")
    points = receipt.get("points")
    if not isinstance(mask, Mapping) or not isinstance(points, Mapping):
        raise N0AAuditError("core mask/points receipt is absent")
    return {
        "schema": receipt.get("schema"),
        "protocol_id": receipt.get("protocol_id"),
        "f0_source_identity": receipt.get("f0_source_identity"),
        "f0_source_identity_sha256": receipt.get("f0_source_identity_sha256"),
        "h0_input": receipt.get("h0_input"),
        "h0_input_sha256": receipt.get("h0_input_sha256"),
        "hypotheses": receipt.get("hypotheses"),
        "mask": dict(mask),
        "points": {
            "voxel_count": points.get("voxel_count"),
            "quantile_point_count": points.get("quantile_point_count"),
            "stored_point_count": points.get("stored_point_count"),
            "points_and_voxel_keys_sha256": points.get(
                "points_and_voxel_keys_sha256"
            ),
        },
        "valid": receipt.get("valid"),
        "abstention_reason": receipt.get("abstention_reason"),
        "input_sha256": receipt.get("input_sha256"),
    }


def _validate_core_receipt(
    receipt: object,
    identity: Mapping[str, Any],
    evidence_mask_sha: str,
    evidence_points_sha: str,
    evidence_result_sha: str,
    label: str,
) -> tuple[dict[str, Any], bool, str | None]:
    if not isinstance(receipt, Mapping):
        raise N0AAuditError(f"{label} core receipt is absent")
    expected_receipt_keys = {
        "schema", "protocol_id", "mode", "contracts",
        "f0_source_identity", "f0_source_identity_sha256",
        "h0_input", "h0_input_sha256", "hypotheses", "mask", "points",
        "valid", "abstention_reason", "input_sha256", "result_sha256",
    }
    if set(receipt) != expected_receipt_keys:
        raise N0AAuditError(f"{label} core receipt key set differs")
    if (
        receipt.get("schema") != CORE_SCHEMA
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("mode") != "shadow"
        or receipt.get("f0_source_identity") != identity
    ):
        raise N0AAuditError(f"{label} core identity/schema differs")
    if receipt.get("f0_source_identity_sha256") != _canonical_json_sha256(identity):
        raise N0AAuditError(f"{label} core identity hash differs")
    h0_input = receipt.get("h0_input")
    if (
        not isinstance(h0_input, Mapping)
        or set(h0_input) != {
            "valid", "world_q02", "world_q98", "world_center", "world_extent"
        }
    ):
        raise N0AAuditError(f"{label} H0 input is absent")
    if receipt.get("h0_input_sha256") != _canonical_json_sha256(h0_input):
        raise N0AAuditError(f"{label} H0 input hash differs")
    mask = receipt.get("mask")
    points = receipt.get("points")
    if not isinstance(mask, Mapping) or not isinstance(points, Mapping):
        raise N0AAuditError(f"{label} mask/point receipt differs")
    if set(mask) != {
        "shape", "bitorder", "packed_byte_count", "sha256",
        "tight_box_xyxy", "pixel_count", "valid_depth_ratio",
        "interior_pixel_count", "metric_depth_pixel_count",
        "depth_jump_pixel_count", "support_pixel_count",
    } or set(points) != {
        "voxel_size_m", "voxel_representative", "voxel_count",
        "quantile_point_count", "stored_point_count",
        "maximum_stored_point_count", "points_and_voxel_keys_sha256",
    }:
        raise N0AAuditError(f"{label} mask/point receipt key set differs")
    stored_count = points.get("stored_point_count")
    if (
        mask.get("shape") != [480, 640]
        or mask.get("bitorder") != "little"
        or mask.get("packed_byte_count") != MASK_PACKED_BYTES
        or mask.get("sha256") != evidence_mask_sha
        or isinstance(stored_count, bool)
        or not isinstance(stored_count, int)
        or stored_count not in range(MAX_STORED_POINTS + 1)
        or points.get("points_and_voxel_keys_sha256") != evidence_points_sha
        or receipt.get("result_sha256") != evidence_result_sha
    ):
        raise N0AAuditError(f"{label} core evidence hashes/counts differ")
    core_contracts = receipt.get("contracts")
    required_core_contracts = {
        "f0_source_identity_preserved": True,
        "ground_truth_access": False,
        "semantic_or_clip_access": False,
        "native_prediction_access": False,
        "history_or_state": False,
        "training": False,
        "online_learning": False,
        "birth_enabled": False,
        "native_output_mutation": False,
    }
    if (
        not isinstance(core_contracts, Mapping)
        or set(core_contracts) != set(required_core_contracts)
        or any(
        core_contracts.get(key) is not expected
        for key, expected in required_core_contracts.items()
        )
    ):
        raise N0AAuditError(f"{label} core contracts differ")
    expected_result = _canonical_json_sha256(_result_payload_from_receipt(receipt))
    if receipt.get("result_sha256") != expected_result:
        raise N0AAuditError(f"{label} core result hash is not self-consistent")
    valid = receipt.get("valid")
    reason = receipt.get("abstention_reason")
    hypotheses = receipt.get("hypotheses")
    if (
        not isinstance(valid, bool)
        or (valid and reason is not None)
        or (not valid and (not isinstance(reason, str) or not reason))
        or not isinstance(hypotheses, Mapping)
        or not isinstance(hypotheses.get("H0"), Mapping)
        or not isinstance(hypotheses.get("HS"), Mapping)
        or hypotheses["HS"].get("valid") is not valid
        or hypotheses["HS"].get("abstention_reason") != reason
    ):
        raise N0AAuditError(f"{label} HS validity/abstention differs")
    expected_valid_aabb_keys = {
        "name", "valid", "abstention_reason", "q02", "q98", "center", "extent"
    }
    expected_invalid_aabb_keys = {"name", "valid", "abstention_reason"}
    if (
        set(hypotheses) != {"H0", "HS"}
        or set(hypotheses["H0"]) != expected_valid_aabb_keys
        or set(hypotheses["HS"])
        != (expected_valid_aabb_keys if valid else expected_invalid_aabb_keys)
        or hypotheses["H0"].get("name") != "H0"
        or hypotheses["H0"].get("valid") is not True
        or hypotheses["H0"].get("abstention_reason") is not None
        or hypotheses["HS"].get("name") != "HS"
    ):
        raise N0AAuditError(f"{label} H0/HS receipt key set differs")
    return dict(h0_input), valid, reason


def _receipt_nontrivial(
    receipt: Mapping[str, Any], label: str
) -> tuple[bool, float | None, float | None]:
    """Recompute the frozen H0/HS distinction from receipt geometry."""

    if receipt.get("valid") is not True:
        return False, None, None
    hypotheses = receipt.get("hypotheses")
    if not isinstance(hypotheses, Mapping):
        raise N0AAuditError(f"{label} hypotheses are absent")
    h0, hs = hypotheses.get("H0"), hypotheses.get("HS")
    if not isinstance(h0, Mapping) or not isinstance(hs, Mapping):
        raise N0AAuditError(f"{label} H0/HS is absent")
    try:
        h0_lo = np.asarray(h0["q02"], dtype=np.float64)
        h0_hi = np.asarray(h0["q98"], dtype=np.float64)
        hs_lo = np.asarray(hs["q02"], dtype=np.float64)
        hs_hi = np.asarray(hs["q98"], dtype=np.float64)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise N0AAuditError(f"{label} H0/HS bounds differ") from error
    if any(array.shape != (3,) or not np.isfinite(array).all() for array in (h0_lo, h0_hi, hs_lo, hs_hi)):
        raise N0AAuditError(f"{label} H0/HS bounds differ")
    iou = _aabb_iou(h0_lo, h0_hi, hs_lo, hs_hi)
    displacement = float(
        np.max(np.abs(np.concatenate((h0_lo - hs_lo, h0_hi - hs_hi))))
    )
    return iou < 0.90 or displacement >= 0.05, iou, displacement


def _selected_f0_mask(frame: Mapping[str, Any], rank: int, raw_index: int, mask_sha: str) -> Mapping[str, Any]:
    funnel = frame.get("funnel")
    if not isinstance(funnel, Mapping) or not isinstance(funnel.get("masks"), list):
        raise N0AAuditError("F0 selected-mask diagnostics are absent")
    rows = [
        row
        for row in funnel["masks"]
        if isinstance(row, Mapping)
        and row.get("selected") is True
        and row.get("rank") == rank
        and row.get("raw_index") == raw_index
        and row.get("mask_sha256") == mask_sha
    ]
    if len(rows) != 1 or rows[0].get("decision") != "selected":
        raise N0AAuditError("F0 selected-mask join differs")
    return rows[0]


def _load_evidence(path: Path, label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != EVIDENCE_ARRAY_NAMES:
                raise N0AAuditError(f"{label} evidence key set differs")
            arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    except (OSError, ValueError, EOFError) as error:
        if isinstance(error, N0AAuditError):
            raise
        raise N0AAuditError(f"{label} evidence cannot be decoded") from error
    schema_raw = arrays["schema_utf8"]
    if schema_raw.dtype != np.uint8 or bytes(schema_raw).decode("utf-8") != EVIDENCE_SCHEMA:
        raise N0AAuditError(f"{label} evidence schema differs")
    count = int(arrays["mask_packbits"].shape[0])
    if arrays["mask_packbits"].dtype != np.uint8 or arrays["mask_packbits"].shape != (count, MASK_PACKED_BYTES):
        raise N0AAuditError(f"{label} packed-mask array differs")
    if (
        arrays["points_world"].dtype != np.float64
        or arrays["points_world"].ndim != 2
        or arrays["points_world"].shape[1:] != (3,)
        or not np.isfinite(arrays["points_world"]).all()
    ):
        raise N0AAuditError(f"{label} point array differs")
    if arrays["voxel_keys"].dtype != np.int64 or arrays["voxel_keys"].shape != arrays["points_world"].shape:
        raise N0AAuditError(f"{label} voxel-key array differs")
    for key in ("point_offsets", "frame_ordinals", "frame_ids", "ranks", "raw_indices", "selected_hypothesis_indices"):
        if arrays[key].dtype != np.int64:
            raise N0AAuditError(f"{label} {key} dtype differs")
    if arrays["point_offsets"].shape != (count + 1,) or arrays["point_offsets"][0] != 0 or arrays["point_offsets"][-1] != len(arrays["points_world"]) or np.any(np.diff(arrays["point_offsets"]) < 0):
        raise N0AAuditError(f"{label} point offsets differ")
    for key in ("frame_ordinals", "frame_ids", "ranks", "raw_indices", "selected_hypothesis_indices", "predicted_ious"):
        if arrays[key].shape != (count,):
            raise N0AAuditError(f"{label} {key} shape differs")
    if arrays["predicted_ious"].dtype != np.float32 or arrays["all_predicted_ious"].dtype != np.float32 or arrays["all_predicted_ious"].shape != (count, 3):
        raise N0AAuditError(f"{label} predicted-IoU evidence differs")
    if arrays["result_sha256_ascii"].dtype.kind != "S" or arrays["result_sha256_ascii"].shape != (count,):
        raise N0AAuditError(f"{label} result hash array differs")
    return arrays


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise N0AAuditError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise N0AAuditError(f"{label} must be finite and non-negative")
    return result


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise N0AAuditError("runtime sample differs")
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _runner_runtime_gate_receipt(
    incremental: Sequence[float],
    composed: Sequence[float],
    *,
    deadline_misses: int,
    cuda_peak: int,
) -> dict[str, Any]:
    """Recompute the runner's preliminary runtime gate without trusting it."""

    inc = _distribution(incremental)
    replay = _distribution(composed)
    mean_per_raw = (
        float(replay["mean"]) / SOURCE_FRAME_STRIDE if replay["count"] else 0.0
    )
    gates = {
        "n0a_incremental_warm_p95_ms": {
            "actual": inc["p95"], "threshold": 250.0,
            "comparator": "<=", "passed": inc["p95"] <= 250.0,
        },
        "replay_composed_warm_p95_ms": {
            "actual": replay["p95"], "threshold": 500.0,
            "comparator": "<=", "passed": replay["p95"] <= 500.0,
        },
        "replay_composed_warm_max_ms": {
            "actual": replay["max"], "threshold": 833.33,
            "comparator": "<", "passed": replay["max"] < 833.33,
        },
        "replay_composed_mean_per_raw_frame_ms": {
            "actual": mean_per_raw, "threshold": 20.0,
            "comparator": "<=", "passed": mean_per_raw <= 20.0,
        },
        "gap25_warm_deadline_miss_count": {
            "actual": deadline_misses, "threshold": 0,
            "comparator": "==", "passed": deadline_misses == 0,
        },
        "cuda_peak_memory_bytes": {
            "actual": cuda_peak, "threshold": MAX_CUDA_BYTES,
            "comparator": "<=", "passed": cuda_peak <= MAX_CUDA_BYTES,
        },
    }
    return {
        "gates": gates,
        "overall_pass": all(row["passed"] for row in gates.values()),
    }


def _runner_capacity_gate_receipt(
    *,
    totals: Mapping[str, int],
    valid_scene_count: int,
    nontrivial_scene_count: int,
    merge_only: bool,
) -> tuple[dict[str, Any], bool | None]:
    gates: dict[str, Any] = {
        "merge_only": merge_only,
        "valid_hs_count": {
            "actual": totals["valid_hs_count"], "threshold": MIN_VALID_SOURCE_COUNT,
            "comparator": ">=",
            "passed": totals["valid_hs_count"] >= MIN_VALID_SOURCE_COUNT,
        },
        "valid_scene_count": {
            "actual": valid_scene_count, "threshold": MIN_VALID_SCENE_COUNT,
            "comparator": ">=", "passed": valid_scene_count >= MIN_VALID_SCENE_COUNT,
        },
        "nontrivial_hs_count": {
            "actual": totals["nontrivial_hs_count"],
            "threshold": MIN_NONTRIVIAL_SOURCE_COUNT, "comparator": ">=",
            "passed": totals["nontrivial_hs_count"] >= MIN_NONTRIVIAL_SOURCE_COUNT,
        },
        "nontrivial_scene_count": {
            "actual": nontrivial_scene_count,
            "threshold": MIN_NONTRIVIAL_SCENE_COUNT, "comparator": ">=",
            "passed": nontrivial_scene_count >= MIN_NONTRIVIAL_SCENE_COUNT,
        },
    }
    overall = (
        all(row["passed"] for key, row in gates.items() if key != "merge_only")
        if not merge_only
        else None
    )
    return gates, overall


def _runner_determinism_gate_receipt(
    *,
    warning_policy: Mapping[str, Any],
    warning_count: int,
    forward_count: int,
    source_ids: Sequence[str],
) -> dict[str, Any]:
    selected = [source_id for source_id in source_ids if replay_sample_selected(source_id)]
    return {
        "status": "pending_create_only_merge_replay_receipt",
        "overall_pass": None,
        "registered_warn_only_operation": "aten::cumsum_cuda",
        "warning_policy": dict(warning_policy),
        "authenticated_warning_count": warning_count,
        "expected_warning_count": 2 * forward_count,
        "warning_count_formula": "2 * provider_forward_count",
        "per_forward_exact_warning_pair_passed": True,
        "half_prefix_exact_source_result_replay": None,
        "fresh_same_gpu_one_percent_exact_replay": None,
        "one_percent_selector": (
            "big_endian_first_two_bytes_of_sha256(source_id_ASCII)_lt_0x0290"
        ),
        "shard_selected_source_count": len(selected),
        "shard_selected_source_ids_sha256": _canonical_json_sha256(selected),
        "shard_selected_source_ids": selected,
        "future_frame_perturbation_invariance": None,
        "n0b_or_gt_stage_authorized": False,
    }


def _validate_source_row(
    *,
    source: Mapping[str, Any],
    f0_candidate: Mapping[str, Any],
    f0_frame: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    evidence_index: int,
    scene_position: int,
    scene_index: int,
    scene_id: str,
    frame_ordinal: int,
    frame_id: int,
    rank: int,
) -> tuple[SourceRecord, list[Any], str]:
    if set(source) != SOURCE_ROW_KEYS:
        raise N0AAuditError("N0a source receipt key set differs")
    raw_index = f0_candidate.get("raw_index")
    if isinstance(raw_index, bool) or not isinstance(raw_index, int):
        raise N0AAuditError("F0 raw index differs")
    expected_id = _source_id(scene_id, frame_id, raw_index)
    identity_keys = (
        "scene_index", "scene_id", "frame_ordinal", "frame_id", "rank",
        "raw_index", "mask_sha256", "points_and_voxel_keys_sha256", "source_id",
    )
    identity = {key: source.get(key) for key in identity_keys}
    expected_identity = {
        "scene_index": scene_index,
        "scene_id": scene_id,
        "frame_ordinal": frame_ordinal,
        "frame_id": frame_id,
        "rank": rank,
        "raw_index": raw_index,
        "mask_sha256": f0_candidate.get("mask_sha256"),
        "points_and_voxel_keys_sha256": f0_candidate.get("points_and_voxel_keys_sha256"),
        "source_id": expected_id,
    }
    if identity != expected_identity or source.get("prompt_tight_box_xyxy") != f0_candidate.get("tight_box_xyxy"):
        raise N0AAuditError(f"{expected_id} F0 identity/prompt differs")
    if source.get("evidence_index") != evidence_index:
        raise N0AAuditError(f"{expected_id} evidence index differs")
    for key, expected in (
        ("frame_ordinals", frame_ordinal), ("frame_ids", frame_id),
        ("ranks", rank), ("raw_indices", raw_index),
    ):
        if int(arrays[key][evidence_index]) != expected:
            raise N0AAuditError(f"{expected_id} evidence identity differs: {key}")
    offsets = arrays["point_offsets"]
    start, stop = int(offsets[evidence_index]), int(offsets[evidence_index + 1])
    if source.get("point_offset") != [start, stop] or stop - start > MAX_STORED_POINTS:
        raise N0AAuditError(f"{expected_id} point offset differs")
    packed = np.ascontiguousarray(arrays["mask_packbits"][evidence_index], dtype=np.uint8)
    mask_sha = hashlib.sha256(packed.tobytes()).hexdigest()
    point_digest = hashlib.sha256()
    point_digest.update(np.ascontiguousarray(arrays["points_world"][start:stop], dtype="<f8").tobytes())
    point_digest.update(np.ascontiguousarray(arrays["voxel_keys"][start:stop], dtype="<i8").tobytes())
    points_sha = point_digest.hexdigest()
    try:
        result_sha = bytes(arrays["result_sha256_ascii"][evidence_index]).decode("ascii")
    except UnicodeDecodeError as error:
        raise N0AAuditError(f"{expected_id} result hash is not ASCII") from error
    core_receipt = source.get("n0a_receipt")
    core, valid, reason = _validate_core_receipt(
        core_receipt, identity, mask_sha, points_sha, result_sha, expected_id
    )
    assert isinstance(core_receipt, Mapping)
    points_receipt = core_receipt.get("points")
    if (
        not isinstance(points_receipt, Mapping)
        or points_receipt.get("stored_point_count") != stop - start
        or isinstance(points_receipt.get("voxel_count"), bool)
        or not isinstance(points_receipt.get("voxel_count"), int)
        or points_receipt.get("voxel_count", -1) < stop - start
        or points_receipt.get("quantile_point_count")
        != points_receipt.get("voxel_count")
    ):
        raise N0AAuditError(f"{expected_id} stored/quantile point census differs")
    if core.get("world_q02", core.get("q02")) != f0_candidate.get("world_q02") or core.get("world_q98", core.get("q98")) != f0_candidate.get("world_q98"):
        raise N0AAuditError(f"{expected_id} H0 geometry differs from F0")
    sam2 = source.get("sam2")
    if (
        not isinstance(sam2, Mapping)
        or set(sam2) != {
            "selected_hypothesis_index", "predicted_iou",
            "all_predicted_ious", "used_for_detection_score",
        }
        or sam2.get("used_for_detection_score") is not False
    ):
        raise N0AAuditError(f"{expected_id} SAM2 diagnostic differs")
    selected_index = int(arrays["selected_hypothesis_indices"][evidence_index])
    if selected_index not in range(3) or sam2.get("selected_hypothesis_index") != selected_index:
        raise N0AAuditError(f"{expected_id} selected hypothesis differs")
    evidence_selected = np.asarray(arrays["predicted_ious"][evidence_index], dtype="<f4")
    evidence_all = np.ascontiguousarray(arrays["all_predicted_ious"][evidence_index], dtype="<f4")
    json_selected_bytes = _little_bytes(sam2.get("predicted_iou"), "<f4", (), f"{expected_id} selected IoU")
    json_all_bytes = _little_bytes(sam2.get("all_predicted_ious"), "<f4", (3,), f"{expected_id} all IoUs")
    if json_selected_bytes != evidence_selected.tobytes() or json_all_bytes != evidence_all.tobytes() or evidence_selected.tobytes() != evidence_all[selected_index].tobytes():
        raise N0AAuditError(f"{expected_id} predicted-IoU bytes differ")
    if int(np.argmax(evidence_all)) != selected_index:
        raise N0AAuditError(f"{expected_id} selected hypothesis violates argmax")
    selected_mask = _selected_f0_mask(
        f0_frame, rank, raw_index, str(identity["mask_sha256"])
    )
    lineage = _canonical_json_sha256(
        {
            "identity": identity,
            "f0_candidate_sha256": _canonical_json_sha256(f0_candidate),
            "f0_mask_diagnostic_sha256": _canonical_json_sha256(selected_mask),
            "n0a_result_sha256": result_sha,
            "sam2_selected_hypothesis_index": selected_index,
            "sam2_all_predicted_ious": [float(value) for value in evidence_all],
        }
    )
    if source.get("source_lineage_sha256") != lineage:
        raise N0AAuditError(f"{expected_id} source lineage differs")
    nontrivial, h0_hs_iou, face_displacement = _receipt_nontrivial(
        core_receipt, expected_id
    )
    recorded_nontrivial = source.get("nontrivial_vs_h0")
    recorded_iou = source.get("h0_hs_iou3d")
    recorded_displacement = source.get("maximum_face_displacement_m")
    if recorded_nontrivial is not nontrivial:
        raise N0AAuditError(f"{expected_id} nontrivial flag differs")
    if h0_hs_iou is None:
        if recorded_iou is not None or recorded_displacement is not None:
            raise N0AAuditError(f"{expected_id} invalid-HS geometry diagnostics differ")
    elif (
        not isinstance(recorded_iou, (int, float))
        or not isinstance(recorded_displacement, (int, float))
        or float(recorded_iou) != h0_hs_iou
        or float(recorded_displacement) != face_displacement
    ):
        raise N0AAuditError(f"{expected_id} geometry diagnostics differ")
    record = SourceRecord(
        scene_position=scene_position,
        scene_index=scene_index,
        scene_id=scene_id,
        frame_ordinal=frame_ordinal,
        frame_id=frame_id,
        rank=rank,
        raw_index=raw_index,
        source_id=expected_id,
        identity=identity,
        prompt_box=[float(value) for value in source["prompt_tight_box_xyxy"]],
        h0=core,
        expected_selected_index=selected_index,
        expected_selected_iou_bytes=evidence_selected.tobytes(),
        expected_all_iou_bytes=evidence_all.tobytes(),
        expected_mask_sha256=mask_sha,
        expected_result_sha256=result_sha,
        expected_valid=valid,
        expected_abstention_reason=reason,
        expected_nontrivial=nontrivial,
    )
    ledger = [
        scene_index, scene_id, frame_ordinal, frame_id, rank, raw_index,
        identity["mask_sha256"], identity["points_and_voxel_keys_sha256"],
        [int(value) for value in source["prompt_tight_box_xyxy"]],
    ]
    return record, ledger, lineage


def _same_reference(left: object, right: object) -> bool:
    return (
        isinstance(left, Mapping)
        and isinstance(right, Mapping)
        and left.get("path") == right.get("path")
        and left.get("sha256") == right.get("sha256")
    )


def _validate_manifest_scene_row_schema(
    row: object, *, label: str
) -> Mapping[str, Any]:
    """Reject every unrecognized nested producer field before opening it."""

    expected_row_keys = {
        "scene_id", "scene_index", "sidecar", "evidence_npz", "counts",
        "runtime", "source_ids_sha256", "source_lineage_sha256",
        "excluded_runtime_reporting", "resumed",
    }
    if not isinstance(row, Mapping) or set(row) != expected_row_keys:
        raise N0AAuditError(f"{label} scene row differs")
    sidecar = row.get("sidecar")
    evidence = row.get("evidence_npz")
    counts = row.get("counts")
    if (
        not isinstance(sidecar, Mapping)
        or set(sidecar) != {"path", "sha256"}
        or not isinstance(evidence, Mapping)
        or set(evidence) != {"path", "sha256", "byte_count", "schema"}
        or evidence.get("schema") != EVIDENCE_SCHEMA
        or isinstance(evidence.get("byte_count"), bool)
        or not isinstance(evidence.get("byte_count"), int)
        or evidence.get("byte_count") <= 0
        or not isinstance(counts, Mapping)
        or set(counts) != set(SCENE_COUNT_KEYS)
        or any(
            isinstance(counts.get(key), bool)
            or not isinstance(counts.get(key), int)
            or counts.get(key) < 0
            for key in SCENE_COUNT_KEYS
        )
        or row.get("resumed") is not False
    ):
        raise N0AAuditError(f"{label} nested scene-row schema differs")
    return row


def load_and_validate_bundle(
    manifest_paths: Sequence[Path],
    *,
    expectations: AuditExpectations = AuditExpectations(),
) -> AuditBundle:
    """Fully authenticate all manifests, receipts, evidence and ledgers."""

    if not manifest_paths:
        raise N0AAuditError("at least one explicit shard manifest is required")
    seals: dict[str, str] = {}
    manifests: list[dict[str, Any]] = []
    manifest_sources: list[tuple[Path, dict[str, Any]]] = []
    for ordinal, raw_path in enumerate(manifest_paths):
        path, manifest = _read_json(Path(raw_path), f"N0a shard manifest {ordinal}")
        seals[os.fspath(path)] = _sha256(path)
        expected_manifest_keys = {
            "schema", "protocol_id", "complete", "shard_index", "num_shards",
            "run_signature_sha256", "signature_payload_sha256", "contracts",
            "inputs", "global_input_integrity", "source_receipts",
            "provider_config", "warning_policy", "environment", "scenes",
            "totals", "expected_shard_census", "runtime",
            "excluded_runtime_reporting", "runtime_gates",
            "runtime_gates_preliminary", "capacity_gates",
            "capacity_gates_overall_pass", "capacity_gates_preliminary",
            "determinism_gates", "decision", "resumed_scene_count",
            "native_output_mutation_count", "content_sha256",
        }
        if (
            manifest.get("schema") != SHARD_SCHEMA
            or manifest.get("protocol_id") != PROTOCOL_ID
            or manifest.get("complete") is not True
            or not _content_hash_valid(manifest)
            or not _valid_sha256(manifest.get("run_signature_sha256"))
            or not _valid_sha256(manifest.get("signature_payload_sha256"))
            or set(manifest) != expected_manifest_keys
        ):
            raise N0AAuditError(f"shard manifest contract/content differs: {path}")
        _exact_contracts(manifest.get("contracts"), f"shard {path.name}")
        manifests.append(manifest)
        manifest_sources.append((path, manifest))
    num_shards_values = {manifest.get("num_shards") for manifest in manifests}
    if len(num_shards_values) != 1:
        raise N0AAuditError("shard num_shards values differ")
    num_shards = next(iter(num_shards_values))
    if isinstance(num_shards, bool) or not isinstance(num_shards, int) or num_shards < 1:
        raise N0AAuditError("invalid num_shards")
    indices = [manifest.get("shard_index") for manifest in manifests]
    if sorted(indices) != list(range(num_shards)) or len(manifests) != num_shards:
        raise N0AAuditError("explicit manifests do not form one complete shard set")
    manifest_sources.sort(key=lambda pair: int(pair[1]["shard_index"]))
    manifests = [pair[1] for pair in manifest_sources]

    canonical_provider: dict[str, Any] | None = None
    canonical_warning_policy = _expected_warning_policy()
    source_receipt_hashes: dict[str, str] | None = None
    canonical_environment: dict[str, Any] | None = None
    canonical_global_seals: dict[str, dict[str, str]] | None = None
    frozen_scene_ids: tuple[str, ...] | None = None
    frozen_f0_sidecars: dict[str, dict[str, str]] | None = None
    gpu_uuids: set[str] = set()
    scene_entries: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    runner_decisions: list[str] = []
    universe_rows: list[dict[str, Any]] = []
    for manifest in manifests:
        provider_dict = _validate_provider_config(
            manifest.get("provider_config"), "shard"
        )
        if canonical_provider is None:
            canonical_provider = provider_dict
        elif canonical_provider != provider_dict:
            raise N0AAuditError("provider configurations differ across shards")
        if manifest.get("warning_policy") != canonical_warning_policy:
            raise N0AAuditError("shard exact warning policy differs")
        receipts = manifest.get("source_receipts")
        if (
            not isinstance(receipts, Mapping)
            or set(receipts) != {"runner", "protocol", "core", "provider"}
        ):
            raise N0AAuditError("source receipts are absent")
        current_hashes: dict[str, str] = {}
        for role in ("runner", "protocol", "core", "provider"):
            if (
                not isinstance(receipts.get(role), Mapping)
                or set(receipts[role]) != {"path", "sha256"}
            ):
                raise N0AAuditError(f"sealed {role} reference schema differs")
            path = _seal_reference(receipts.get(role), f"sealed {role}", seals)
            current_hashes[role] = _sha256(path)
        if (
            current_hashes["runner"] != EXPECTED_RUNNER_SHA256
            or current_hashes["protocol"] != EXPECTED_PROTOCOL_SHA256
            or current_hashes["core"] != EXPECTED_CORE_SHA256
            or current_hashes["provider"] != EXPECTED_PROVIDER_SHA256
        ):
            raise N0AAuditError("frozen runner/protocol/core/provider SHA differs")
        if source_receipt_hashes is None:
            source_receipt_hashes = current_hashes
        elif source_receipt_hashes != current_hashes:
            raise N0AAuditError("source receipt hashes differ across shards")
        inputs = manifest.get("inputs")
        universe = inputs.get("universe") if isinstance(inputs, Mapping) else None
        if (
            not isinstance(inputs, Mapping)
            or set(inputs) != {"f0_receipt", "universe"}
            or not isinstance(universe, Mapping)
            or set(universe) != {
                "full200_scene_list", "derived_cohort_scene_list_sha256",
                "frame_ledger_sha256", "source_ledger_sha256",
                "sidecar_ledger_sha256", "census", "scene_census",
            }
        ):
            raise N0AAuditError("authenticated universe is absent")
        if (
            not isinstance(inputs.get("f0_receipt"), Mapping)
            or set(inputs["f0_receipt"])
            != {"path", "sha256", "run_signature_sha256"}
            or not isinstance(universe.get("full200_scene_list"), Mapping)
            or set(universe["full200_scene_list"]) != {"path", "sha256"}
        ):
            raise N0AAuditError("authenticated universe reference schema differs")
        f0_path = _seal_reference(inputs.get("f0_receipt"), "sealed F0 merge", seals, ".json")
        if _sha256(f0_path) != EXPECTED_F0_RECEIPT_SHA256:
            raise N0AAuditError("sealed F0 merge hash differs")
        list_path = _seal_reference(universe.get("full200_scene_list"), "sealed full200 list", seals, ".txt")
        if _sha256(list_path) != EXPECTED_FULL200_SCENE_LIST_SHA256:
            raise N0AAuditError("sealed full200 scene-list hash differs")
        if frozen_scene_ids is None or frozen_f0_sidecars is None:
            listed_all = tuple(
                line.strip()
                for line in list_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            _, frozen_f0_merge = _read_json(f0_path, "frozen F0 merge")
            frozen_rows = frozen_f0_merge.get("scenes")
            if (
                frozen_f0_merge.get("schema") != EXPECTED_F0_MERGE_SCHEMA
                or frozen_f0_merge.get("protocol_id") != EXPECTED_F0_PROTOCOL
                or frozen_f0_merge.get("complete") is not True
                or frozen_f0_merge.get("overall_pass") is not True
                or not isinstance(frozen_rows, list)
                or len(listed_all) != len(frozen_rows)
            ):
                raise N0AAuditError("frozen F0 merge/list contract differs")
            selected_ids = listed_all[
                expectations.cohort_start :
                expectations.cohort_start + expectations.scene_count
            ]
            selected_rows = frozen_rows[
                expectations.cohort_start :
                expectations.cohort_start + expectations.scene_count
            ]
            if (
                len(selected_ids) != expectations.scene_count
                or len(set(selected_ids)) != expectations.scene_count
                or any(
                    not isinstance(row, Mapping)
                    or row.get("scene_id") != scene_id
                    or row.get("scene_index") != scene_index
                    or not isinstance(row.get("sidecar"), Mapping)
                    or set(row["sidecar"]) != {"path", "sha256"}
                    for scene_index, (scene_id, row) in enumerate(
                        zip(selected_ids, selected_rows),
                        start=expectations.cohort_start,
                    )
                )
            ):
                raise N0AAuditError("frozen F0 extra cohort identity differs")
            frozen_scene_ids = tuple(selected_ids)
            frozen_f0_sidecars = {
                scene_id: {
                    "path": str(row["sidecar"]["path"]),
                    "sha256": str(row["sidecar"]["sha256"]),
                }
                for scene_id, row in zip(selected_ids, selected_rows)
            }
        if (
            universe.get("derived_cohort_scene_list_sha256") != expectations.scene_list_sha256
            or universe.get("frame_ledger_sha256") != expectations.frame_ledger_sha256
            or universe.get("source_ledger_sha256") != expectations.source_ledger_sha256
            or universe.get("sidecar_ledger_sha256") != expectations.sidecar_ledger_sha256
        ):
            raise N0AAuditError("manifest universe ledger seals differ")
        universe_rows.append(dict(universe))
        global_integrity = manifest.get("global_input_integrity")
        global_rows = (
            global_integrity.get("seals")
            if isinstance(global_integrity, Mapping)
            else None
        )
        scene_census = universe.get("scene_census")
        census = universe.get("census")
        if (
            not isinstance(scene_census, Mapping)
            or set(scene_census) != set(frozen_scene_ids)
            or not isinstance(census, Mapping)
            or set(census) != {
                "scene_count", "keyframe_count", "successful_frame_count",
                "source_count", "provider_forward_count",
                "successful_empty_frame_count",
            }
            or census != {
                "scene_count": expectations.scene_count,
                "keyframe_count": expectations.keyframe_count,
                "successful_frame_count": expectations.successful_frame_count,
                "source_count": expectations.source_count,
                "provider_forward_count": EXPECTED_PROVIDER_FORWARDS,
                "successful_empty_frame_count": EXPECTED_SUCCESSFUL_EMPTY_FRAMES,
            }
        ):
            raise N0AAuditError("authenticated universe census schema differs")
        for expected_index, scene_id in enumerate(
            frozen_scene_ids, start=expectations.cohort_start
        ):
            census_row = scene_census[scene_id]
            if (
                not isinstance(census_row, Mapping)
                or set(census_row) != {
                    "scene_index", "keyframe_count", "successful_frame_count",
                    "source_count", "provider_forward_count",
                }
                or census_row.get("scene_index") != expected_index
                or any(
                    isinstance(census_row.get(key), bool)
                    or not isinstance(census_row.get(key), int)
                    or census_row.get(key) < 0
                    for key in (
                        "keyframe_count", "successful_frame_count", "source_count",
                        "provider_forward_count",
                    )
                )
            ):
                raise N0AAuditError("authenticated scene census schema differs")
        if (
            not isinstance(global_integrity, Mapping)
            or set(global_integrity) != {
                "seal_count", "before_sha256", "after_sha256", "passed",
                "seals", "sam2_asset_end_rehash",
            }
            or global_integrity.get("passed") is not True
            or global_integrity.get("sam2_asset_end_rehash")
            != "pending_mandatory_merge_replay_audit_provider_authenticated_before_masks"
            or not isinstance(global_rows, list)
            or global_integrity.get("seal_count") != len(global_rows)
            or global_integrity.get("before_sha256")
            != _canonical_json_sha256(global_rows)
            or global_integrity.get("after_sha256")
            != global_integrity.get("before_sha256")
        ):
            raise N0AAuditError("shard global input-integrity receipt differs")
        expected_global_references: dict[str, Mapping[str, Any]] = {
            "f0_full200_receipt": inputs["f0_receipt"],
            "f0_full200_scene_list": universe["full200_scene_list"],
            "runner": receipts["runner"],
            "protocol": receipts["protocol"],
            "core": receipts["core"],
            "provider": receipts["provider"],
            "warning_source": {
                "path": os.fspath(EXPECTED_WARNING_SOURCE),
                "sha256": EXPECTED_WARNING_SOURCE_SHA256,
            },
            **{
                f"f0_sidecar:{scene_id}": reference
                for scene_id, reference in frozen_f0_sidecars.items()
            },
        }
        role_map: dict[str, dict[str, str]] = {}
        for seal in global_rows:
            if (
                not isinstance(seal, Mapping)
                or set(seal) != {"role", "path", "sha256"}
                or not isinstance(seal.get("role"), str)
                or seal["role"] in role_map
            ):
                raise N0AAuditError("shard global input seal schema differs")
            role = str(seal["role"])
            expected_reference = expected_global_references.get(role)
            if (
                expected_reference is None
                or not _same_reference(seal, expected_reference)
            ):
                raise N0AAuditError("shard global input allowlist differs")
            path = _seal_reference(seal, f"global input {role}", seals)
            role_map[role] = {
                "path": os.fspath(path), "sha256": str(seal["sha256"])
            }
        if set(role_map) != set(expected_global_references):
            raise N0AAuditError("shard global input role census differs")
        if canonical_global_seals is None:
            canonical_global_seals = role_map
        elif role_map != canonical_global_seals:
            raise N0AAuditError("global input seals differ across shards")
        environment = manifest.get("environment")
        gpu_uuids.add(
            _validate_environment_receipt(
                environment,
                provider_config=provider_dict,
                label=f"shard {manifest.get('shard_index')}",
            )
        )
        assert isinstance(environment, Mapping)
        if canonical_environment is None:
            canonical_environment = copy.deepcopy(dict(environment))
        elif environment != canonical_environment:
            raise N0AAuditError("exact environment receipt differs across shards")
        if manifest.get("native_output_mutation_count") != 0:
            raise N0AAuditError("runner reports native output mutation")
        runner_decisions.append(str(manifest.get("decision")))
        rows = manifest.get("scenes")
        if not isinstance(rows, list):
            raise N0AAuditError("manifest scene rows are absent")
        for row in rows:
            row = _validate_manifest_scene_row_schema(
                row, label=f"shard {manifest.get('shard_index')}"
            )
            if (
                isinstance(row.get("scene_index"), bool)
                or not isinstance(row.get("scene_index"), int)
                or int(row["scene_index"]) % num_shards
                != int(manifest["shard_index"])
            ):
                raise N0AAuditError("scene is assigned to the wrong shard")
            scene_entries.append((manifest, row))
    if len(gpu_uuids) != 1:
        raise N0AAuditError("shards were not produced on one authenticated GPU")
    if any(row != universe_rows[0] for row in universe_rows[1:]):
        raise N0AAuditError("authenticated universe differs across shards")
    assert (
        canonical_provider is not None
        and canonical_environment is not None
        and canonical_global_seals is not None
        and source_receipt_hashes is not None
    )

    # The receipt was compared to hard-coded constants above.  Only now is it
    # safe for the auditor to open the fixed asset paths.
    if canonical_provider != dict(EXPECTED_PROVIDER_CONFIG):
        raise N0AAuditError("frozen provider configuration differs")
    checkpoint_path = _regular_file(EXPECTED_SAM2_CHECKPOINT, "SAM2 checkpoint")
    checkpoint_sha = _sha256(checkpoint_path)
    if (
        checkpoint_sha != EXPECTED_SAM2_CHECKPOINT_SHA256
        or checkpoint_path.stat().st_size != EXPECTED_SAM2_CHECKPOINT_BYTES
    ):
        raise N0AAuditError("SAM2 checkpoint seal differs")
    seals[os.fspath(checkpoint_path)] = checkpoint_sha
    source_root = EXPECTED_SAM2_SOURCE_ROOT
    config_path = _regular_file(
        source_root / "sam2" / EXPECTED_SAM2_CONFIG_NAME, "SAM2 config"
    )
    config_sha = _sha256(config_path)
    if config_sha != EXPECTED_SAM2_CONFIG_SHA256:
        raise N0AAuditError("SAM2 config seal differs")
    seals[os.fspath(config_path)] = config_sha
    source_files = sorted(
        (source_root / "sam2").rglob("*.py"),
        key=lambda path: path.relative_to(source_root).as_posix(),
    )
    source_digest = hashlib.sha256()
    if len(source_files) != 23:
        raise N0AAuditError("SAM2 source-file census differs")
    for path in source_files:
        regular = _regular_file(path, "SAM2 source")
        digest = _sha256(regular)
        relative = regular.relative_to(source_root).as_posix()
        expected_critical = EXPECTED_SAM2_CRITICAL_SOURCE_SHA256.get(relative)
        if expected_critical is not None and digest != expected_critical:
            raise N0AAuditError(f"critical SAM2 source SHA differs: {relative}")
        source_digest.update(f"{digest}  {relative}\n".encode("ascii"))
        seals[os.fspath(regular)] = digest
    if source_digest.hexdigest() != EXPECTED_SAM2_SOURCE_TREE_SHA256:
        raise N0AAuditError("SAM2 source-tree manifest hash differs")
    if any(
        not (source_root / relative).is_file()
        for relative in EXPECTED_SAM2_CRITICAL_SOURCE_SHA256
    ):
        raise N0AAuditError("critical SAM2 source census differs")
    warning_source = _regular_file(
        EXPECTED_WARNING_SOURCE, "frozen warning source", ".py"
    )
    if _sha256(warning_source) != EXPECTED_WARNING_SOURCE_SHA256:
        raise N0AAuditError("frozen warning source SHA differs")
    if (
        hashlib.sha256(EXPECTED_WARNING_MESSAGE.encode("utf-8")).hexdigest()
        != EXPECTED_WARNING_MESSAGE_SHA256
    ):
        raise N0AAuditError("frozen warning message SHA differs")
    seals[os.fspath(warning_source)] = EXPECTED_WARNING_SOURCE_SHA256

    if len(scene_entries) != expectations.scene_count:
        raise N0AAuditError("complete shard scene census differs")
    scene_entries.sort(key=lambda pair: int(pair[1].get("scene_index", -1)))
    expected_indices = list(range(expectations.cohort_start, expectations.cohort_start + expectations.scene_count))
    if [row.get("scene_index") for _, row in scene_entries] != expected_indices:
        raise N0AAuditError("complete scene index/order differs")

    frames_out: list[FrameRecord] = []
    frame_ledger: list[list[Any]] = []
    source_ledger: list[list[Any]] = []
    sidecar_ledger: list[list[Any]] = []
    source_ids: set[str] = set()
    totals = {
        "scene_count": len(scene_entries), "keyframe_count": 0,
        "successful_frame_count": 0, "source_count": 0,
        "provider_forward_count": 0, "valid_hs_count": 0,
        "invalid_hs_count": 0, "nontrivial_hs_count": 0,
        "authenticated_warning_count": 0,
    }
    valid_scene_count = 0
    nontrivial_scene_count = 0
    warm_incremental: list[float] = []
    warm_composed: list[float] = []
    deadline_misses = 0
    cuda_peak = 0
    scene_ids_order: list[str] = []
    shard_forward_cursors = {
        int(manifest["shard_index"]): 0 for manifest in manifests
    }
    shard_warm_incremental: dict[int, list[float]] = {
        index: [] for index in shard_forward_cursors
    }
    shard_warm_composed: dict[int, list[float]] = {
        index: [] for index in shard_forward_cursors
    }
    shard_deadline_misses: dict[int, int] = {
        index: 0 for index in shard_forward_cursors
    }
    shard_cuda_peak: dict[int, int] = {
        index: 0 for index in shard_forward_cursors
    }
    shard_source_ids: dict[int, list[str]] = {
        index: [] for index in shard_forward_cursors
    }
    future_fixture_source: dict[str, Any] | None = None
    for scene_position, (manifest, row) in enumerate(scene_entries):
        scene_index = int(row["scene_index"])
        scene_id = row.get("scene_id")
        if not isinstance(scene_id, str):
            raise N0AAuditError("scene ID differs")
        scene_ids_order.append(scene_id)
        scene_path = _seal_reference(row.get("sidecar"), f"{scene_id} N0a scene", seals, ".json")
        evidence_path = _seal_reference(row.get("evidence_npz"), f"{scene_id} N0a evidence", seals, ".npz")
        _, scene = _read_json(scene_path, f"{scene_id} N0a scene")
        expected_scene_keys = {
            "schema", "protocol_id", "complete", "scene_id", "scene_index",
            "run_signature_sha256", "contracts", "warning_policy", "inputs",
            "evidence_npz", "frames", "counts", "runtime",
            "source_ids_sha256", "source_lineage_sha256",
            "native_output_mutation_count", "bounded_state", "content_sha256",
        }
        if (
            scene.get("schema") != SCENE_SCHEMA or scene.get("protocol_id") != PROTOCOL_ID
            or scene.get("complete") is not True or scene.get("scene_id") != scene_id
            or scene.get("scene_index") != scene_index or not _content_hash_valid(scene)
            or scene.get("run_signature_sha256") != manifest.get("run_signature_sha256")
            or scene.get("native_output_mutation_count") != 0
            or set(scene) != expected_scene_keys
        ):
            raise N0AAuditError(f"{scene_id} scene receipt contract/content differs")
        _exact_contracts(scene.get("contracts"), f"scene {scene_id}")
        if scene.get("warning_policy") != canonical_warning_policy:
            raise N0AAuditError(f"{scene_id} exact warning policy differs")
        if scene.get("evidence_npz") != row.get("evidence_npz"):
            raise N0AAuditError(f"{scene_id} evidence references differ")
        evidence_reference = scene.get("evidence_npz")
        if (
            not isinstance(evidence_reference, Mapping)
            or set(evidence_reference)
            != {"path", "sha256", "byte_count", "schema"}
            or evidence_reference.get("schema") != EVIDENCE_SCHEMA
            or isinstance(evidence_reference.get("byte_count"), bool)
            or not isinstance(evidence_reference.get("byte_count"), int)
            or evidence_reference.get("byte_count") <= 0
            or evidence_reference.get("byte_count") != evidence_path.stat().st_size
        ):
            raise N0AAuditError(f"{scene_id} evidence reference schema differs")
        scene_inputs = scene.get("inputs")
        if (
            not isinstance(scene_inputs, Mapping)
            or set(scene_inputs) != {
                "f0_sidecar", "intrinsic", "frozen_inputs_before_sha256",
                "frozen_inputs_after_sha256",
            }
        ):
            raise N0AAuditError(f"{scene_id} scene inputs are absent")
        f0_sidecar_path = _seal_reference(scene_inputs.get("f0_sidecar"), f"{scene_id} F0 sidecar", seals, ".json")
        intrinsic_path = _seal_reference(scene_inputs.get("intrinsic"), f"{scene_id} intrinsic", seals, ".txt")
        if (
            set(scene_inputs["f0_sidecar"]) != {"kind", "path", "sha256"}
            or set(scene_inputs["intrinsic"]) != {"kind", "path", "sha256"}
            or scene_inputs["f0_sidecar"].get("kind") != "f0_sidecar"
            or scene_inputs["intrinsic"].get("kind") != "intrinsic"
        ):
            raise N0AAuditError(f"{scene_id} scene input roles differ")
        if not _same_reference(
            canonical_global_seals.get(f"f0_sidecar:{scene_id}"),
            scene_inputs["f0_sidecar"],
        ):
            raise N0AAuditError(f"{scene_id} global/scene F0 sidecar seals differ")
        scene_input_seals: list[dict[str, Any]] = [
            dict(scene_inputs["f0_sidecar"]), dict(scene_inputs["intrinsic"])
        ]
        _, f0_scene = _read_json(f0_sidecar_path, f"{scene_id} F0 sidecar")
        if (
            f0_scene.get("schema") != EXPECTED_F0_SCENE_SCHEMA
            or f0_scene.get("protocol_id") != EXPECTED_F0_PROTOCOL
            or f0_scene.get("scene_id") != scene_id
            or f0_scene.get("scene_index") != scene_index
            or f0_scene.get("complete") is not True
        ):
            raise N0AAuditError(f"{scene_id} F0 sidecar identity differs")
        sidecar_ledger.append([scene_index, scene_id, f0_sidecar_path.name, _sha256(f0_sidecar_path)])
        arrays = _load_evidence(evidence_path, scene_id)
        output_frames = scene.get("frames")
        f0_frames = f0_scene.get("frames")
        if not isinstance(output_frames, list) or not isinstance(f0_frames, list) or len(output_frames) != len(f0_frames):
            raise N0AAuditError(f"{scene_id} frame ledger differs")
        if future_fixture_source is None:
            for current_ordinal, current_frame in enumerate(f0_frames):
                current_funnel = (
                    current_frame.get("funnel")
                    if isinstance(current_frame, Mapping)
                    else None
                )
                current_candidates = (
                    current_funnel.get("candidates")
                    if isinstance(current_funnel, Mapping)
                    else None
                )
                if (
                    not isinstance(current_frame, Mapping)
                    or current_frame.get("successful") is not True
                    or not isinstance(current_candidates, list)
                    or not current_candidates
                ):
                    continue
                for future_frame in f0_frames[current_ordinal + 1 :]:
                    future_funnel = (
                        future_frame.get("funnel")
                        if isinstance(future_frame, Mapping)
                        else None
                    )
                    future_candidates = (
                        future_funnel.get("candidates")
                        if isinstance(future_funnel, Mapping)
                        else None
                    )
                    if (
                        not isinstance(future_frame, Mapping)
                        or future_frame.get("successful") is not True
                        or future_candidates != []
                    ):
                        continue
                    fixture_input_seals: dict[str, dict[str, str]] = {}
                    for prefix, fixture_frame in (
                        ("current", current_frame),
                        ("future", future_frame),
                    ):
                        frame_inputs = fixture_frame.get("inputs")
                        if not isinstance(frame_inputs, Mapping):
                            raise N0AAuditError(
                                f"{scene_id} future-gate {prefix} inputs differ"
                            )
                        for kind, suffix in (
                            ("rgb", None), ("depth", ".png"), ("pose", ".txt")
                        ):
                            raw_path = frame_inputs.get(f"{kind}_path")
                            expected_sha = frame_inputs.get(f"{kind}_sha256")
                            if not isinstance(raw_path, str) or not _valid_sha256(
                                expected_sha
                            ):
                                raise N0AAuditError(
                                    f"{scene_id} future-gate {prefix} {kind} seal differs"
                                )
                            path = _regular_file(
                                Path(raw_path),
                                f"{scene_id} future-gate {prefix} {kind}",
                                suffix,
                            )
                            actual = _sha256(path)
                            if actual != expected_sha:
                                raise N0AAuditError(
                                    f"{scene_id} future-gate {prefix} {kind} rehash differs"
                                )
                            seals[os.fspath(path)] = actual
                            fixture_input_seals[f"{prefix}_{kind}"] = {
                                "path": os.fspath(path), "sha256": actual
                            }
                    if (
                        fixture_input_seals["current_rgb"]["sha256"]
                        == fixture_input_seals["future_rgb"]["sha256"]
                    ):
                        raise N0AAuditError(
                            f"{scene_id} future-gate RGB perturbation would be inert"
                        )
                    future_fixture_source = {
                        "scene_id": scene_id,
                        "original_scene_index": scene_index,
                        "intrinsic": {
                            "path": os.fspath(intrinsic_path),
                            "sha256": _sha256(intrinsic_path),
                        },
                        "current_frame": copy.deepcopy(dict(current_frame)),
                        "future_frame": copy.deepcopy(dict(future_frame)),
                        "input_seals": fixture_input_seals,
                    }
                    break
                if future_fixture_source is not None:
                    break
        evidence_cursor = 0
        scene_source_ids: list[str] = []
        scene_lineages: list[str] = []
        scene_valid = 0
        scene_nontrivial = 0
        scene_successful = 0
        scene_forwards = 0
        scene_authenticated_warning_count = 0
        scene_incremental_all: list[float] = []
        scene_incremental_warm: list[float] = []
        scene_composed_all: list[float] = []
        scene_composed_warm: list[float] = []
        scene_deadline_all = 0
        scene_deadline_warm = 0
        scene_frame_provider_peak = 0
        for ordinal, (frame, f0_frame) in enumerate(zip(output_frames, f0_frames)):
            if not isinstance(frame, Mapping) or not isinstance(f0_frame, Mapping):
                raise N0AAuditError(f"{scene_id} frame row differs")
            frame_id = f0_frame.get("frame_id")
            if frame.get("frame_ordinal") != ordinal or f0_frame.get("frame_ordinal") != ordinal or frame.get("frame_id") != frame_id or not isinstance(frame_id, int):
                raise N0AAuditError(f"{scene_id} frame identity/order differs")
            if frame.get("current_only") is not True or frame.get("successful") != f0_frame.get("successful"):
                raise N0AAuditError(f"{scene_id}/{frame_id} current-only/status differs")
            expected_abstention = (
                None
                if f0_frame.get("successful") is True
                else f0_frame.get("abstention")
            )
            if frame.get("abstention") != expected_abstention:
                raise N0AAuditError(f"{scene_id}/{frame_id} abstention differs")
            if "max_accessed_frame_ordinal" in frame and frame.get("max_accessed_frame_ordinal") != ordinal:
                raise N0AAuditError(f"{scene_id}/{frame_id} future-frame ordinal was accessed")
            frame_ledger.append([scene_index, scene_id, ordinal, frame_id])
            scene_successful += int(frame.get("successful") is True)
            funnel = f0_frame.get("funnel")
            candidates = funnel.get("candidates", []) if isinstance(funnel, Mapping) else []
            if not isinstance(candidates, list):
                raise N0AAuditError(f"{scene_id}/{frame_id} F0 candidates differ")
            output_sources = frame.get("sources")
            if not isinstance(output_sources, list) or len(output_sources) != len(candidates):
                raise N0AAuditError(f"{scene_id}/{frame_id} source census differs")
            invoked = bool(candidates)
            expected_frame_keys = {
                "frame_ordinal", "frame_id", "successful", "abstention",
                "current_only", "provider_invoked", "authenticated_warning_count",
                "sources", "runtime",
            }
            if frame.get("successful") is True:
                expected_frame_keys.add("max_accessed_frame_ordinal")
            if invoked:
                expected_frame_keys.update({"provider_forward_count", "input"})
            if set(frame) != expected_frame_keys:
                raise N0AAuditError(f"{scene_id}/{frame_id} frame key set differs")
            if frame.get("provider_invoked") is not invoked:
                raise N0AAuditError(f"{scene_id}/{frame_id} provider invocation differs")
            if not invoked:
                if (
                    frame.get("runtime") is not None
                    or frame.get("authenticated_warning_count") != 0
                ):
                    raise N0AAuditError(
                        f"{scene_id}/{frame_id} empty frame warning/runtime differs"
                    )
                continue
            scene_forwards += 1
            frame_input = frame.get("input")
            f0_input = f0_frame.get("inputs")
            if (
                not isinstance(frame_input, Mapping)
                or set(frame_input) != {
                    "rgb", "depth", "pose", "intrinsic", "rgb_color_order",
                    "box_source",
                }
                or frame_input.get("rgb_color_order")
                != "RGB_after_exact_BGR_to_RGB"
                or frame_input.get("box_source")
                != "sealed_F0_candidate.tight_box_xyxy"
                or not isinstance(f0_input, Mapping)
            ):
                raise N0AAuditError(f"{scene_id}/{frame_id} input seals differ")
            input_refs: dict[str, dict[str, str]] = {}
            for kind, suffix in (("rgb", None), ("depth", ".png"), ("pose", ".txt")):
                if (
                    not isinstance(frame_input.get(kind), Mapping)
                    or set(frame_input[kind]) != {"path", "sha256"}
                ):
                    raise N0AAuditError(
                        f"{scene_id}/{frame_id} {kind} reference schema differs"
                    )
                path = _seal_reference(frame_input.get(kind), f"{scene_id}/{frame_id} {kind}", seals, suffix)
                expected_path = f0_input.get(f"{kind}_path")
                expected_sha = f0_input.get(f"{kind}_sha256")
                if os.fspath(path) != os.fspath(Path(str(expected_path)).resolve()) or _sha256(path) != expected_sha:
                    raise N0AAuditError(f"{scene_id}/{frame_id} {kind} differs from F0")
                input_refs[kind] = {"path": os.fspath(path), "sha256": _sha256(path)}
                scene_input_seals.append(
                    {
                        "kind": kind,
                        "frame_ordinal": ordinal,
                        "frame_id": frame_id,
                        "path": os.fspath(path),
                        "sha256": _sha256(path),
                    }
                )
            if (
                not isinstance(frame_input.get("intrinsic"), Mapping)
                or set(frame_input["intrinsic"]) != {"path", "sha256"}
                or not _same_reference(
                    frame_input.get("intrinsic"), scene_inputs.get("intrinsic")
                )
            ):
                raise N0AAuditError(f"{scene_id}/{frame_id} intrinsic reference differs")
            frame_sources: list[SourceRecord] = []
            for rank, (source_raw, candidate_raw) in enumerate(zip(output_sources, candidates)):
                if not isinstance(source_raw, Mapping) or not isinstance(candidate_raw, Mapping) or candidate_raw.get("rank") != rank:
                    raise N0AAuditError(f"{scene_id}/{frame_id}/{rank} source row differs")
                record, ledger, lineage = _validate_source_row(
                    source=source_raw, f0_candidate=candidate_raw, f0_frame=f0_frame,
                    arrays=arrays, evidence_index=evidence_cursor,
                    scene_position=scene_position, scene_index=scene_index,
                    scene_id=scene_id, frame_ordinal=ordinal, frame_id=frame_id,
                    rank=rank,
                )
                evidence_cursor += 1
                if record.source_id in source_ids:
                    raise N0AAuditError("duplicate source ID")
                source_ids.add(record.source_id)
                scene_source_ids.append(record.source_id)
                shard_source_ids[int(manifest["shard_index"])].append(
                    record.source_id
                )
                scene_lineages.append(lineage)
                source_ledger.append(ledger)
                frame_sources.append(record)
                scene_valid += int(record.expected_valid)
                scene_nontrivial += int(record.expected_nontrivial)
            frames_out.append(
                FrameRecord(
                    scene_position=scene_position, scene_index=scene_index,
                    scene_id=scene_id, frame_ordinal=ordinal, frame_id=frame_id,
                    rgb=input_refs["rgb"], depth=input_refs["depth"], pose=input_refs["pose"],
                    intrinsic={"path": os.fspath(intrinsic_path), "sha256": _sha256(intrinsic_path)},
                    sources=frame_sources,
                )
            )
            runtime = frame.get("runtime")
            expected_frame_runtime_keys = {
                "provider_call_index_in_shard",
                "n0a_warmup_excluded",
                "decode_ms",
                "sam2_provider_ms",
                "cold_provider_load_and_first_forward_ms",
                "cold_provider_metric_includes_first_forward",
                "sam2_provider_timing",
                "deterministic_warning_evidence",
                "masklift_ms",
                "n0a_incremental_ms",
                "offline_evidence_buffer_ms_excluded",
                "sealed_f0_complete_ms",
                "replay_composed_ms",
                "replay_composed_ms_per_source_frame",
                "gap25_deadline_missed",
                "gap25_deadline_missed_warm",
            }
            if (
                not isinstance(runtime, Mapping)
                or set(runtime) != expected_frame_runtime_keys
            ):
                raise N0AAuditError(f"{scene_id}/{frame_id} runtime is absent")
            shard_index = int(manifest["shard_index"])
            expected_call = shard_forward_cursors[shard_index]
            expected_warmup = expected_call < 3
            shard_forward_cursors[shard_index] += 1
            decode_ms = _number(runtime.get("decode_ms"), "decode runtime")
            provider_ms = _number(runtime.get("sam2_provider_ms"), "provider runtime")
            cold_ms = _number(
                runtime.get("cold_provider_load_and_first_forward_ms"),
                "cold provider load/first-forward runtime",
            )
            lift_ms = _number(runtime.get("masklift_ms"), "masklift runtime")
            incremental_ms = _number(runtime.get("n0a_incremental_ms"), "incremental runtime")
            f0_ms = _number(runtime.get("sealed_f0_complete_ms"), "sealed F0 runtime")
            composed_ms = _number(runtime.get("replay_composed_ms"), "composed runtime")
            f0_runtime = f0_frame.get("runtime")
            if not isinstance(f0_runtime, Mapping):
                raise N0AAuditError(f"{scene_id}/{frame_id} F0 runtime is absent")
            authenticated_f0_ms = _number(
                f0_runtime.get("complete_ms"), "authenticated F0 complete runtime"
            )
            tolerance = max(1.0e-9, composed_ms * 1.0e-12)
            warning_count = frame.get("authenticated_warning_count")
            if (
                runtime.get("provider_call_index_in_shard") != expected_call
                or runtime.get("n0a_warmup_excluded") is not expected_warmup
                or cold_ms != (provider_ms if expected_call == 0 else 0.0)
                or runtime.get("cold_provider_metric_includes_first_forward")
                is not (expected_call == 0)
                or runtime.get("sealed_f0_complete_ms")
                != f0_runtime.get("complete_ms")
                or f0_ms != authenticated_f0_ms
                or abs(decode_ms + provider_ms + lift_ms - incremental_ms) > tolerance
                or abs(f0_ms + incremental_ms - composed_ms) > tolerance
                or runtime.get("replay_composed_ms_per_source_frame")
                != composed_ms / SOURCE_FRAME_STRIDE
                or runtime.get("gap25_deadline_missed")
                is not (composed_ms >= 833.33)
                or runtime.get("gap25_deadline_missed_warm")
                is not ((not expected_warmup) and composed_ms >= 833.33)
                or frame.get("provider_forward_count") != 1
                or warning_count != len(EXPECTED_WARNING_LINES)
                or runtime.get("deterministic_warning_evidence")
                != _expected_warning_evidence()
            ):
                raise N0AAuditError(f"{scene_id}/{frame_id} runtime accounting differs")
            _number(
                runtime.get("offline_evidence_buffer_ms_excluded"),
                "offline evidence runtime",
            )
            provider_timing = runtime.get("sam2_provider_timing")
            if not isinstance(provider_timing, Mapping) or set(provider_timing) != {
                "encoder_ms", "decoder_and_host_mask_ms", "complete_ms",
                "cuda_synchronized", "peak_allocated_memory_bytes",
            }:
                raise N0AAuditError(f"{scene_id}/{frame_id} provider timing differs")
            encoder_ms = _number(provider_timing["encoder_ms"], "encoder runtime")
            decoder_ms = _number(provider_timing["decoder_and_host_mask_ms"], "decoder runtime")
            complete_ms = _number(provider_timing["complete_ms"], "provider complete runtime")
            if (
                abs(encoder_ms + decoder_ms - complete_ms)
                > max(1.0e-6, complete_ms * 1.0e-9)
                or provider_timing["cuda_synchronized"] is not True
                or isinstance(provider_timing["peak_allocated_memory_bytes"], bool)
                or not isinstance(provider_timing["peak_allocated_memory_bytes"], int)
                or provider_timing["peak_allocated_memory_bytes"] < 0
            ):
                raise N0AAuditError(f"{scene_id}/{frame_id} provider timing values differ")
            scene_frame_provider_peak = max(
                scene_frame_provider_peak,
                int(provider_timing["peak_allocated_memory_bytes"]),
            )
            scene_authenticated_warning_count += int(warning_count)
            scene_incremental_all.append(incremental_ms)
            scene_composed_all.append(composed_ms)
            scene_deadline_all += int(composed_ms >= 833.33)
            if not expected_warmup:
                warm_incremental.append(incremental_ms)
                warm_composed.append(composed_ms)
                deadline_misses += int(composed_ms >= 833.33)
                shard_warm_incremental[shard_index].append(incremental_ms)
                shard_warm_composed[shard_index].append(composed_ms)
                shard_deadline_misses[shard_index] += int(composed_ms >= 833.33)
                scene_incremental_warm.append(incremental_ms)
                scene_composed_warm.append(composed_ms)
                scene_deadline_warm += int(composed_ms >= 833.33)
        if evidence_cursor != arrays["mask_packbits"].shape[0]:
            raise N0AAuditError(f"{scene_id} evidence source census differs")
        counts = scene.get("counts")
        row_counts = row.get("counts")
        if (
            not isinstance(counts, Mapping)
            or set(counts) != set(SCENE_COUNT_KEYS)
            or not isinstance(row_counts, Mapping)
            or set(row_counts) != set(SCENE_COUNT_KEYS)
            or counts != row_counts
        ):
            raise N0AAuditError(f"{scene_id} counts are absent")
        expected_counts = {
            "keyframe_count": len(output_frames), "successful_frame_count": scene_successful,
            "source_count": len(scene_source_ids), "provider_forward_count": scene_forwards,
            "valid_hs_count": scene_valid, "invalid_hs_count": len(scene_source_ids) - scene_valid,
            "nontrivial_hs_count": scene_nontrivial,
        }
        for key, expected in expected_counts.items():
            if counts.get(key) != expected or row.get("counts", {}).get(key) != expected:
                raise N0AAuditError(f"{scene_id} {key} differs")
            totals[key] += expected
        allowed_count = counts.get("authenticated_warning_count")
        if (
            isinstance(allowed_count, bool)
            or not isinstance(allowed_count, int)
            or allowed_count < 0
            or allowed_count != scene_authenticated_warning_count
            or allowed_count != 2 * scene_forwards
            or row.get("counts", {}).get("authenticated_warning_count")
            != allowed_count
        ):
            raise N0AAuditError(f"{scene_id} authenticated warning count differs")
        totals["authenticated_warning_count"] += allowed_count
        if scene.get("source_ids_sha256") != _canonical_json_sha256(scene_source_ids) or row.get("source_ids_sha256") != scene.get("source_ids_sha256"):
            raise N0AAuditError(f"{scene_id} source-ID aggregate differs")
        if scene.get("source_lineage_sha256") != _canonical_json_sha256(scene_lineages) or row.get("source_lineage_sha256") != scene.get("source_lineage_sha256"):
            raise N0AAuditError(f"{scene_id} source-lineage aggregate differs")
        scene_input_aggregate = _canonical_json_sha256(scene_input_seals)
        if (
            scene_inputs.get("frozen_inputs_before_sha256") != scene_input_aggregate
            or scene_inputs.get("frozen_inputs_after_sha256") != scene_input_aggregate
        ):
            raise N0AAuditError(f"{scene_id} frozen-input aggregate differs")
        bounded = scene.get("bounded_state")
        if (
            not isinstance(bounded, Mapping)
            or set(bounded) != {
                "cross_frame_model_or_object_state",
                "current_frame_arrays_released_before_next_frame",
                "evidence_spool_is_output_only_offline_state",
                "maximum_sources_per_current_frame",
                "maximum_stored_points_per_source",
            }
            or any(
            (
                bounded.get("cross_frame_model_or_object_state") is not False,
                bounded.get("current_frame_arrays_released_before_next_frame") is not True,
                bounded.get("evidence_spool_is_output_only_offline_state") is not True,
                bounded.get("maximum_sources_per_current_frame") != 16,
                bounded.get("maximum_stored_points_per_source") != MAX_STORED_POINTS,
            )
            )
        ):
            raise N0AAuditError(f"{scene_id} bounded-state contract differs")
        valid_scene_count += int(scene_valid > 0)
        nontrivial_scene_count += int(scene_nontrivial > 0)
        runtime_scene = scene.get("runtime")
        expected_scene_runtime = {
            "n0a_incremental_all_ms": _distribution(scene_incremental_all),
            "n0a_incremental_warm_ms": _distribution(scene_incremental_warm),
            "replay_composed_all_ms": _distribution(scene_composed_all),
            "replay_composed_warm_ms": _distribution(scene_composed_warm),
            "gap25_all_deadline_miss_count": scene_deadline_all,
            "gap25_warm_deadline_miss_count": scene_deadline_warm,
        }
        excluded = (
            runtime_scene.get("excluded_offline_reporting")
            if isinstance(runtime_scene, Mapping)
            else None
        )
        expected_excluded_keys = {
            "input_pre_rehash_ms",
            "intrinsic_decode_ms",
            "input_end_rehash_ms",
            "evidence_npz_compression_write_ms",
            "included_in_online_or_warm_distributions",
        }
        if (
            not isinstance(runtime_scene, Mapping)
            or set(runtime_scene) != set(expected_scene_runtime) | {
                "cuda_peak_memory_bytes", "excluded_offline_reporting"
            }
            or any(
                runtime_scene.get(key) != expected
                for key, expected in expected_scene_runtime.items()
            )
            or row.get("runtime") != runtime_scene
            or not isinstance(excluded, Mapping)
            or set(excluded) != expected_excluded_keys
            or excluded.get("included_in_online_or_warm_distributions") is not False
        ):
            raise N0AAuditError(f"{scene_id} runtime aggregate differs")
        for key in expected_excluded_keys - {
            "included_in_online_or_warm_distributions"
        }:
            _number(excluded.get(key), f"{scene_id} excluded offline {key}")
        row_excluded = row.get("excluded_runtime_reporting")
        if (
            not isinstance(row_excluded, Mapping)
            or set(row_excluded) != expected_excluded_keys
            | {"scene_json_serialization_write_ms"}
            or any(row_excluded.get(key) != excluded.get(key) for key in excluded)
        ):
            raise N0AAuditError(f"{scene_id} excluded runtime reporting differs")
        _number(
            row_excluded.get("scene_json_serialization_write_ms"),
            f"{scene_id} scene JSON serialization runtime",
        )
        peak_value = runtime_scene.get("cuda_peak_memory_bytes")
        if (
            isinstance(peak_value, bool)
            or not isinstance(peak_value, int)
            or peak_value < scene_frame_provider_peak
        ):
            raise N0AAuditError(f"{scene_id} CUDA peak value differs")
        cuda_peak = max(cuda_peak, peak_value, scene_frame_provider_peak)
        shard_cuda_peak[int(manifest["shard_index"])] = max(
            shard_cuda_peak[int(manifest["shard_index"])],
            peak_value,
            scene_frame_provider_peak,
        )

    scene_list_sha = hashlib.sha256(("\n".join(scene_ids_order) + "\n").encode("utf-8")).hexdigest()
    ledgers = {
        "scene_list_sha256": scene_list_sha,
        "frame_ledger_sha256": _canonical_json_sha256(frame_ledger, sort_keys=False),
        "source_ledger_sha256": _canonical_json_sha256(source_ledger, sort_keys=False),
        "sidecar_ledger_sha256": _canonical_json_sha256(sidecar_ledger, sort_keys=False),
    }
    expected_ledgers = {
        "scene_list_sha256": expectations.scene_list_sha256,
        "frame_ledger_sha256": expectations.frame_ledger_sha256,
        "source_ledger_sha256": expectations.source_ledger_sha256,
        "sidecar_ledger_sha256": expectations.sidecar_ledger_sha256,
    }
    if ledgers != expected_ledgers:
        raise N0AAuditError("reconstructed complete ledger hash differs")
    expected_census = {
        "scene_count": expectations.scene_count,
        "keyframe_count": expectations.keyframe_count,
        "successful_frame_count": expectations.successful_frame_count,
        "source_count": expectations.source_count,
        "provider_forward_count": EXPECTED_PROVIDER_FORWARDS,
        "authenticated_warning_count": EXPECTED_AUTHENTICATED_WARNINGS,
    }
    if {key: totals[key] for key in expected_census} != expected_census:
        raise N0AAuditError("reconstructed complete census differs")
    if len(source_ids) != expectations.source_count:
        raise N0AAuditError("unique source-ID census differs")
    if (
        totals["successful_frame_count"] - totals["provider_forward_count"]
        != EXPECTED_SUCCESSFUL_EMPTY_FRAMES
        or totals["authenticated_warning_count"]
        != 2 * totals["provider_forward_count"]
    ):
        raise N0AAuditError("provider/empty-frame/warning census differs")
    for manifest in manifests:
        shard_totals = manifest.get("totals")
        if (
            not isinstance(shard_totals, Mapping)
            or set(shard_totals) != set(SCENE_COUNT_KEYS)
            or any(
                isinstance(shard_totals.get(key), bool)
                or not isinstance(shard_totals.get(key), int)
                or shard_totals.get(key) < 0
                for key in SCENE_COUNT_KEYS
            )
        ):
            raise N0AAuditError("shard totals are absent")
        shard_rows = manifest.get("scenes", [])
        for key in ("keyframe_count", "successful_frame_count", "source_count", "provider_forward_count", "valid_hs_count", "invalid_hs_count", "nontrivial_hs_count", "authenticated_warning_count"):
            recomputed = sum(int(row["counts"][key]) for row in shard_rows)
            if shard_totals.get(key) != recomputed:
                raise N0AAuditError(f"shard total differs: {key}")
        if shard_totals.get("authenticated_warning_count") != 2 * shard_totals.get(
            "provider_forward_count", -1
        ):
            raise N0AAuditError("shard warning distribution differs")
        expected_shard = manifest.get("expected_shard_census")
        if (
            not isinstance(expected_shard, Mapping)
            or set(expected_shard) != set(SEALED_CENSUS_KEYS)
            or any(
                isinstance(expected_shard.get(key), bool)
                or not isinstance(expected_shard.get(key), int)
                or expected_shard.get(key) < 0
                for key in SEALED_CENSUS_KEYS
            )
        ):
            raise N0AAuditError("expected shard census is absent")
        for key in (
            "keyframe_count", "successful_frame_count", "source_count",
            "provider_forward_count",
        ):
            recomputed = sum(int(row["counts"][key]) for row in shard_rows)
            if expected_shard.get(key) != recomputed:
                raise N0AAuditError(f"authenticated shard census differs: {key}")
        shard_index = int(manifest["shard_index"])
        expected_runtime = {
            "n0a_incremental_warm_ms": _distribution(
                shard_warm_incremental[shard_index]
            ),
            "replay_composed_warm_ms": _distribution(
                shard_warm_composed[shard_index]
            ),
            "gap25_warm_deadline_miss_count": shard_deadline_misses[shard_index],
            "cuda_peak_memory_bytes": shard_cuda_peak[shard_index],
        }
        if manifest.get("runtime") != expected_runtime:
            raise N0AAuditError("shard runtime aggregate differs")
        expected_runtime_gates = _runner_runtime_gate_receipt(
            shard_warm_incremental[shard_index],
            shard_warm_composed[shard_index],
            deadline_misses=shard_deadline_misses[shard_index],
            cuda_peak=shard_cuda_peak[shard_index],
        )
        if (
            manifest.get("runtime_gates") != expected_runtime_gates
            or manifest.get("runtime_gates_preliminary") is not True
        ):
            raise N0AAuditError("shard preliminary runtime gates differ")
        shard_valid_scene_count = sum(
            int(row["counts"]["valid_hs_count"] > 0) for row in shard_rows
        )
        shard_nontrivial_scene_count = sum(
            int(row["counts"]["nontrivial_hs_count"] > 0) for row in shard_rows
        )
        expected_capacity_gates, expected_capacity_pass = (
            _runner_capacity_gate_receipt(
                totals=shard_totals,
                valid_scene_count=shard_valid_scene_count,
                nontrivial_scene_count=shard_nontrivial_scene_count,
                merge_only=len(shard_rows) != expectations.scene_count,
            )
        )
        expected_determinism_gates = _runner_determinism_gate_receipt(
            warning_policy=canonical_warning_policy,
            warning_count=int(shard_totals["authenticated_warning_count"]),
            forward_count=int(shard_totals["provider_forward_count"]),
            source_ids=shard_source_ids[shard_index],
        )
        if (
            manifest.get("capacity_gates") != expected_capacity_gates
            or manifest.get("capacity_gates_overall_pass") is not expected_capacity_pass
            or manifest.get("capacity_gates_preliminary") is not True
            or manifest.get("determinism_gates") != expected_determinism_gates
            or manifest.get("decision") != RUNNER_PENDING_DECISION
            or manifest.get("resumed_scene_count") != 0
            or any(row.get("resumed") is not False for row in shard_rows)
        ):
            raise N0AAuditError("shard closed gate/decision receipt differs")
        excluded = manifest.get("excluded_runtime_reporting")
        expected_excluded_keys = {
            "included_in_online_or_warm_distributions",
            "cold_provider_initialization_ms",
            "cold_model_provider_load_and_first_forward_ms",
            "cold_model_load_is_combined_with_first_forward",
            "sealed_universe_pre_authentication_ms",
            "global_input_end_rehash_ms",
            "scene_aggregate_ms",
            "input_pre_rehash_total_ms",
            "input_end_rehash_total_ms",
            "shard_manifest_json_write_reporting",
        }
        scene_excluded_keys = {
            "input_pre_rehash_ms",
            "intrinsic_decode_ms",
            "input_end_rehash_ms",
            "evidence_npz_compression_write_ms",
            "scene_json_serialization_write_ms",
        }
        expected_scene_excluded = {
            key: float(
                sum(
                    _number(
                        row.get("excluded_runtime_reporting", {}).get(key),
                        f"shard scene excluded runtime {key}",
                    )
                    for row in shard_rows
                )
            )
            for key in scene_excluded_keys
        }
        if (
            not isinstance(excluded, Mapping)
            or set(excluded) != expected_excluded_keys
            or excluded.get("included_in_online_or_warm_distributions") is not False
            or excluded.get("cold_model_load_is_combined_with_first_forward")
            is not True
            or excluded.get("shard_manifest_json_write_reporting")
            != "measured_after_seal_and_returned_out_of_band_to_avoid_self_reference"
            or excluded.get("scene_aggregate_ms") != expected_scene_excluded
        ):
            raise N0AAuditError("shard excluded runtime reporting differs")
        cold_rows = [
            frame["runtime"]
            for row in shard_rows
            for _, scene_receipt in [
                _read_json(
                    Path(str(row.get("sidecar", {}).get("path", ""))),
                    "shard excluded-runtime scene",
                )
            ]
            for frame in scene_receipt.get("frames", [])
            if isinstance(frame, Mapping)
            and isinstance(frame.get("runtime"), Mapping)
            and frame["runtime"].get(
                "cold_provider_metric_includes_first_forward"
            )
            is True
        ]
        expected_cold_rows = int(shard_totals.get("provider_forward_count", 0) > 0)
        expected_cold_ms = float(
            sum(
                _number(
                    row.get("cold_provider_load_and_first_forward_ms"),
                    "shard cold provider load/first-forward runtime",
                )
                for row in cold_rows
            )
        )
        sealed_universe_ms = _number(
            excluded.get("sealed_universe_pre_authentication_ms"),
            "shard sealed-universe authentication runtime",
        )
        global_end_ms = _number(
            excluded.get("global_input_end_rehash_ms"),
            "shard global input end-rehash runtime",
        )
        _number(
            excluded.get("cold_provider_initialization_ms"),
            "shard cold provider initialization runtime",
        )
        if (
            len(cold_rows) != expected_cold_rows
            or _number(
                excluded.get("cold_model_provider_load_and_first_forward_ms"),
                "shard cold provider load/first-forward aggregate",
            )
            != expected_cold_ms
            or _number(
                excluded.get("input_pre_rehash_total_ms"),
                "shard input pre-rehash total",
            )
            != sealed_universe_ms
            + expected_scene_excluded["input_pre_rehash_ms"]
            or _number(
                excluded.get("input_end_rehash_total_ms"),
                "shard input end-rehash total",
            )
            != global_end_ms
            + expected_scene_excluded["input_end_rehash_ms"]
        ):
            raise N0AAuditError("shard excluded runtime totals differ")

    if _sha256(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise N0AAuditError("auditor-local protocol SHA differs")
    seals[os.fspath(PROTOCOL_PATH.resolve())] = EXPECTED_PROTOCOL_SHA256
    auditor_path = _regular_file(Path(__file__).resolve(), "N0a replay auditor source")
    seals[os.fspath(auditor_path)] = _sha256(auditor_path)
    if future_fixture_source is None:
        raise N0AAuditError(
            "no authenticated nonempty-current/later-successful-empty future fixture exists"
        )
    return AuditBundle(
        manifest_paths=[pair[0] for pair in manifest_sources], manifests=manifests,
        frames=frames_out, input_seals=seals, provider_config=canonical_provider,
        environment_receipt=canonical_environment,
        gpu_uuid=next(iter(gpu_uuids)), counts=totals,
        valid_scene_count=valid_scene_count, nontrivial_scene_count=nontrivial_scene_count,
        warm_incremental_ms=warm_incremental, warm_composed_ms=warm_composed,
        deadline_misses=deadline_misses, cuda_peak_memory_bytes=cuda_peak,
        runner_decisions=runner_decisions, ledger_hashes=ledgers,
        source_receipt_hashes=source_receipt_hashes,
        future_fixture_source=future_fixture_source,
    )


def replay_schedule(
    frames: Sequence[FrameRecord], *, first_full_scenes: int = FIRST_FULL_REPLAY_SCENES
) -> list[tuple[FrameRecord, tuple[int, ...]]]:
    """Group the frozen source sample by complete original frame batch."""

    schedule: list[tuple[FrameRecord, tuple[int, ...]]] = []
    for frame in frames:
        compare = tuple(
            index
            for index, source in enumerate(frame.sources)
            if source.scene_position < first_full_scenes
            or replay_sample_selected(source.source_id)
        )
        if compare:
            schedule.append((frame, compare))
    return schedule


def _jsonable_provider_config(value: object) -> dict[str, Any]:
    if not hasattr(value, "__dict__"):
        raise N0AAuditError("runtime provider configuration is absent")
    result: dict[str, Any] = {}
    for key, item in vars(value).items():
        result[str(key)] = os.fspath(item) if isinstance(item, Path) else item
    return result


def _authenticate_runtime_environment(bundle: AuditBundle) -> object:
    """Re-establish the exact frozen software/device policy in the worker."""

    if os.environ.get("PYTHONHASHSEED") != "0":
        raise N0AAuditError("fresh production replay requires PYTHONHASHSEED=0")
    if os.environ.get("CONDA_DEFAULT_ENV") != "gsam2_env":
        raise N0AAuditError("fresh replay is outside the frozen gsam2_env")
    producer_environment = bundle.environment_receipt
    if (
        not isinstance(producer_environment, Mapping)
        or producer_environment.get("platform") != platform.platform()
        or producer_environment.get("cuda_visible_devices")
        != os.environ.get("CUDA_VISIBLE_DEVICES")
    ):
        raise N0AAuditError("fresh replay process/platform receipt differs")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        import cv2  # type: ignore
        import hydra  # type: ignore
        import omegaconf  # type: ignore
        import PIL  # type: ignore
        import torch
        import torchvision  # type: ignore
    except ImportError as error:
        raise N0AAuditError("fresh replay environment dependency is absent") from error
    versions = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "numpy": np.__version__,
        "opencv": str(cv2.__version__),
        "hydra": str(hydra.__version__),
        "omegaconf": str(omegaconf.__version__),
        "pillow": str(PIL.__version__),
    }
    if versions != dict(EXPECTED_ENVIRONMENT_VERSIONS) or not torch.cuda.is_available():
        raise N0AAuditError("fresh replay software/CUDA environment differs")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except (AttributeError, RuntimeError) as error:
        raise N0AAuditError("fresh replay deterministic policy could not be set") from error
    if (
        not torch.are_deterministic_algorithms_enabled()
        or not torch.is_deterministic_algorithms_warn_only_enabled()
        or torch.backends.cudnn.benchmark
        or not torch.backends.cudnn.deterministic
        or torch.backends.cuda.matmul.allow_tf32
        or torch.backends.cudnn.allow_tf32
    ):
        raise N0AAuditError("fresh replay deterministic policy differs")
    logical_index = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(logical_index)
    if (
        str(properties.name) != EXPECTED_GPU_NAME
        or [int(properties.major), int(properties.minor)]
        != EXPECTED_GPU_COMPUTE_CAPABILITY
        or int(properties.total_memory) != EXPECTED_GPU_TOTAL_MEMORY_BYTES
    ):
        raise N0AAuditError("fresh replay GPU model/capability differs")
    try:
        completed = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=index,uuid,name",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise N0AAuditError("could not authenticate replay GPU UUID") from error
    rows: list[tuple[int, str, str]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3 or not fields[0].isdigit() or not fields[1].startswith("GPU-"):
            raise N0AAuditError("nvidia-smi replay GPU identity differs")
        rows.append((int(fields[0]), fields[1], fields[2]))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip():
        physical_token = str(logical_index)
    else:
        tokens = [token.strip() for token in visible.split(",")]
        if logical_index >= len(tokens):
            raise N0AAuditError("CUDA visibility cannot identify replay GPU")
        physical_token = tokens[logical_index]
    if physical_token.isdigit():
        matches = [row for row in rows if row[0] == int(physical_token)]
    elif physical_token.startswith("GPU-"):
        matches = [row for row in rows if row[1] == physical_token]
    else:
        raise N0AAuditError("MIG/ambiguous replay GPU selection is not frozen")
    if (
        len(matches) != 1
        or matches[0][1] != bundle.gpu_uuid
        or matches[0][2] != EXPECTED_GPU_NAME
    ):
        raise N0AAuditError("fresh replay GPU UUID differs from producer GPU")
    producer_gpu = producer_environment.get("preflight", {}).get("gpu", {})
    if (
        not isinstance(producer_gpu, Mapping)
        or producer_gpu.get("logical_index") != logical_index
        or producer_gpu.get("physical_index") != matches[0][0]
        or producer_gpu.get("uuid") != matches[0][1]
        or producer_gpu.get("name") != str(properties.name)
        or producer_gpu.get("compute_capability")
        != [int(properties.major), int(properties.minor)]
        or producer_gpu.get("total_memory_bytes") != int(properties.total_memory)
    ):
        raise N0AAuditError("fresh replay exact GPU receipt differs")
    from boxfusion import sam2_boxprompt_provider as provider_module
    if _jsonable_provider_config(provider_module.PRODUCTION_CONFIG) != dict(
        EXPECTED_PROVIDER_CONFIG
    ):
        raise N0AAuditError("runtime provider configuration differs")
    return torch


def _provider_arrays(result: object, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def field(name: str, alias: str | None = None) -> object:
        if isinstance(result, Mapping):
            value = result.get(name)
            return result.get(alias) if value is None and alias else value
        value = getattr(result, name, None)
        return getattr(result, alias, None) if value is None and alias else value

    masks = np.asarray(field("masks"))
    selected = np.asarray(field("selected_hypothesis_indices", "selected_indices"))
    selected_ious = np.asarray(field("predicted_ious"), dtype=np.float32)
    all_ious = np.asarray(field("all_predicted_ious"), dtype=np.float32)
    if masks.dtype != np.bool_ or masks.shape != (count, 480, 640):
        raise N0AAuditError("replay provider mask shape/dtype differs")
    if selected.shape != (count,) or selected.dtype.kind not in "iu":
        raise N0AAuditError("replay selected-index shape/dtype differs")
    selected = np.asarray(selected, dtype=np.int64)
    if selected_ious.shape != (count,) or all_ious.shape != (count, 3) or not np.isfinite(all_ious).all():
        raise N0AAuditError("replay predicted-IoU shape/value differs")
    if np.any((selected < 0) | (selected >= 3)) or not np.array_equal(selected, np.argmax(all_ious, axis=1)):
        raise N0AAuditError("replay selected hypothesis violates frozen rule")
    return np.ascontiguousarray(masks), selected, np.ascontiguousarray(selected_ious), np.ascontiguousarray(all_ious)


def _load_frame(frame: FrameRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        import cv2  # type: ignore
    except ImportError as error:
        raise N0AAuditError("OpenCV is required for production replay") from error
    for name, seal in (("rgb", frame.rgb), ("depth", frame.depth), ("pose", frame.pose), ("intrinsic", frame.intrinsic)):
        path = _regular_file(Path(seal["path"]), f"replay {name}")
        if _sha256(path) != seal["sha256"]:
            raise N0AAuditError(f"replay {name} rehash differs")
    bgr = cv2.imread(frame.rgb["path"], cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(frame.depth["path"], cv2.IMREAD_UNCHANGED)
    if bgr is None or depth_raw is None or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3 or depth_raw.shape != (480, 640) or not np.issubdtype(depth_raw.dtype, np.integer):
        raise N0AAuditError("replay RGB/depth decode differs")
    rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (640, 480), interpolation=cv2.INTER_LINEAR)
    try:
        pose = np.loadtxt(frame.pose["path"], dtype=np.float64)
        intrinsic = np.loadtxt(frame.intrinsic["path"], dtype=np.float64)
    except (OSError, ValueError) as error:
        raise N0AAuditError("replay pose/intrinsic decode differs") from error
    if intrinsic.shape == (4, 4):
        intrinsic = intrinsic[:3, :3]
    return (
        np.ascontiguousarray(rgb, dtype=np.uint8),
        np.ascontiguousarray(depth_raw.astype(np.float64) / 1000.0),
        np.ascontiguousarray(pose, dtype=np.float64),
        np.ascontiguousarray(intrinsic, dtype=np.float64),
    )


def _aabb_iou(lower_a: np.ndarray, upper_a: np.ndarray, lower_b: np.ndarray, upper_b: np.ndarray) -> float:
    intersection = np.maximum(np.minimum(upper_a, upper_b) - np.maximum(lower_a, lower_b), 0.0)
    intersection_volume = float(np.prod(intersection))
    volume_a = float(np.prod(upper_a - lower_a))
    volume_b = float(np.prod(upper_b - lower_b))
    union = volume_a + volume_b - intersection_volume
    return intersection_volume / union if union > 0.0 else 0.0


def _result_nontrivial(result: object) -> bool:
    if getattr(result, "valid", False) is not True:
        return False
    h0, hs = getattr(result, "h0"), getattr(result, "hs")
    h0_lo, h0_hi = np.asarray(h0.q02), np.asarray(h0.q98)
    hs_lo, hs_hi = np.asarray(hs.q02), np.asarray(hs.q98)
    iou = _aabb_iou(h0_lo, h0_hi, hs_lo, hs_hi)
    face = float(np.max(np.abs(np.concatenate((h0_lo - hs_lo, h0_hi - hs_hi)))))
    return iou < 0.90 or face >= 0.05


def compare_replay_source(
    source: SourceRecord,
    *,
    selected_index: int,
    selected_iou: np.float32,
    all_ious: np.ndarray,
    selected_mask: np.ndarray,
    core_result: object,
) -> list[str]:
    """Return exact replay mismatches for one source (empty means pass)."""

    failures: list[str] = []
    if int(selected_index) != source.expected_selected_index:
        failures.append("selected_hypothesis_index")
    if np.asarray(selected_iou, dtype="<f4").tobytes() != source.expected_selected_iou_bytes:
        failures.append("selected_predicted_iou_bytes")
    if np.ascontiguousarray(all_ious, dtype="<f4").tobytes() != source.expected_all_iou_bytes:
        failures.append("all_predicted_iou_bytes")
    packed = np.packbits(np.asarray(selected_mask, dtype=np.bool_).reshape(-1), bitorder="little")
    if hashlib.sha256(packed.tobytes()).hexdigest() != source.expected_mask_sha256:
        failures.append("mask_sha256")
    if getattr(core_result, "source_id", None) != source.source_id:
        failures.append("source_identity")
    if getattr(core_result, "result_sha256", None) != source.expected_result_sha256:
        failures.append("core_result_sha256")
    if getattr(core_result, "valid", None) is not source.expected_valid:
        failures.append("valid")
    if getattr(core_result, "abstention_reason", None) != source.expected_abstention_reason:
        failures.append("abstention_reason")
    if _result_nontrivial(core_result) is not source.expected_nontrivial:
        failures.append("nontrivial_vs_h0")
    return failures


def perform_replay(
    bundle: AuditBundle,
    *,
    provider_factory: Callable[[], object] | None = None,
    frame_loader: Callable[[FrameRecord], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = _load_frame,
    first_full_scenes: int = FIRST_FULL_REPLAY_SCENES,
    production_preflight: bool = True,
) -> dict[str, Any]:
    """Execute the prespecified replay schedule in the current process."""

    if production_preflight:
        _authenticate_runtime_environment(bundle)
    from boxfusion.sam2_masklift_n0a import lift_sam2_mask
    if provider_factory is None:
        from boxfusion.sam2_boxprompt_provider import FrozenSAM2BoxPromptProvider
        provider_factory = FrozenSAM2BoxPromptProvider
    provider = provider_factory()
    infer = getattr(provider, "predict", None)
    if not callable(infer):
        infer = provider if callable(provider) else None
    if not callable(infer):
        raise N0AAuditError("replay provider lacks predict")
    schedule = replay_schedule(bundle.frames, first_full_scenes=first_full_scenes)
    target_count = sum(len(indices) for _, indices in schedule)
    compared = 0
    authenticated_warning_count = 0
    mismatches: list[dict[str, Any]] = []
    for frame, compare_indices in schedule:
        rgb, depth, pose, intrinsic = frame_loader(frame)
        boxes = np.asarray([source.prompt_box for source in frame.sources], dtype=np.float32)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            provider_result = infer(np.ascontiguousarray(rgb), np.ascontiguousarray(boxes))
        warning_evidence = _validate_exact_warning_rows(
            caught,
            label=f"replay {frame.scene_id}/{frame.frame_id} provider forward",
        )
        authenticated_warning_count += int(warning_evidence["count"])
        masks, selected, selected_ious, all_ious = _provider_arrays(provider_result, len(frame.sources))
        for index in compare_indices:
            source = frame.sources[index]
            result = lift_sam2_mask(
                f0_source_identity=source.identity,
                selected_mask=masks[index],
                depth_m=depth,
                intrinsics=intrinsic,
                camera_to_world=pose,
                h0=source.h0,
            )
            failures = compare_replay_source(
                source, selected_index=int(selected[index]), selected_iou=selected_ious[index],
                all_ious=all_ious[index], selected_mask=masks[index], core_result=result,
            )
            compared += 1
            if failures and len(mismatches) < 100:
                mismatches.append({"source_id": source.source_id, "fields": failures})
        del provider_result, masks, selected, selected_ious, all_ious, rgb, depth, pose, intrinsic
    return {
        "schema": WORKER_SCHEMA,
        "complete": True,
        "fresh_process": False,
        "same_gpu_uuid": bundle.gpu_uuid,
        "selector": "sha256(source_id.encode('ascii'))[:2]_big_endian_lt_0x0290",
        "first_full_scene_count": first_full_scenes,
        "full_batch_for_sampled_frame": True,
        "scheduled_frame_batch_count": len(schedule),
        "provider_forward_count": len(schedule),
        "scheduled_source_comparison_count": target_count,
        "compared_source_count": compared,
        "warning_policy": _expected_warning_policy(),
        "authenticated_warning_count": authenticated_warning_count,
        "expected_warning_count": 2 * len(schedule),
        "warning_count_formula": "2 * provider_forward_count",
        "per_forward_exact_warning_pair_passed": True,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "overall_pass": bool(
            compared == target_count
            and not mismatches
            and authenticated_warning_count == 2 * len(schedule)
        ),
    }


def perform_mirrored_future_perturbation(
    bundle: AuditBundle,
    *,
    provider_factory: Callable[[], object] | None = None,
) -> dict[str, Any]:
    """Run a real-model causal check while mutating only explicit /tmp mirrors.

    One authenticated non-empty current frame is copied to a private temporary
    directory.  The exact current files are then reloaded and the complete
    prompt batch is inferred before and after adding, altering, and deleting
    explicitly named *future* RGB/depth/pose files.  No future path is passed
    to either provider or core and no directory enumeration is performed.
    """

    if not bundle.frames:
        raise N0AAuditError("mirrored future perturbation has no non-empty frame")
    source_frame = bundle.frames[0]
    from boxfusion.sam2_masklift_n0a import lift_sam2_mask
    if provider_factory is None:
        from boxfusion.sam2_boxprompt_provider import FrozenSAM2BoxPromptProvider
        provider_factory = FrozenSAM2BoxPromptProvider
    provider = provider_factory()
    infer = getattr(provider, "predict", None)
    if not callable(infer):
        infer = provider if callable(provider) else None
    if not callable(infer):
        raise N0AAuditError("mirrored future perturbation provider lacks predict")

    def run_current(frame: FrameRecord) -> tuple[list[dict[str, Any]], int]:
        rgb, depth, pose, intrinsic = _load_frame(frame)
        boxes = np.asarray(
            [source.prompt_box for source in frame.sources], dtype=np.float32
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            provider_result = infer(
                np.ascontiguousarray(rgb), np.ascontiguousarray(boxes)
            )
        warning_evidence = _validate_exact_warning_rows(
            caught, label="non-authorizing provider/core mirror forward"
        )
        masks, selected, selected_ious, all_ious = _provider_arrays(
            provider_result, len(frame.sources)
        )
        rows: list[dict[str, Any]] = []
        for index, source in enumerate(frame.sources):
            result = lift_sam2_mask(
                f0_source_identity=source.identity,
                selected_mask=masks[index],
                depth_m=depth,
                intrinsics=intrinsic,
                camera_to_world=pose,
                h0=source.h0,
            )
            packed = np.packbits(
                masks[index].reshape(-1), bitorder="little"
            ).astype(np.uint8, copy=False)
            rows.append(
                {
                    "source_id": source.source_id,
                    "selected_hypothesis_index": int(selected[index]),
                    "selected_predicted_iou_f32_le_hex": np.asarray(
                        selected_ious[index], dtype="<f4"
                    ).tobytes().hex(),
                    "all_predicted_ious_f32_le_hex": np.ascontiguousarray(
                        all_ious[index], dtype="<f4"
                    ).tobytes().hex(),
                    "mask_sha256": hashlib.sha256(packed.tobytes()).hexdigest(),
                    "core_result_sha256": str(result.result_sha256),
                    "valid": bool(result.valid),
                    "abstention_reason": result.abstention_reason,
                    "nontrivial_vs_h0": _result_nontrivial(result),
                }
            )
        return rows, int(warning_evidence["count"])

    with tempfile.TemporaryDirectory(prefix="n0a-future-mirror-") as directory:
        root = Path(directory)
        current_paths: dict[str, Path] = {}
        original = {
            "rgb": source_frame.rgb,
            "depth": source_frame.depth,
            "pose": source_frame.pose,
            "intrinsic": source_frame.intrinsic,
        }
        for kind, seal in original.items():
            suffix = Path(seal["path"]).suffix
            destination = root / f"current_{kind}{suffix}"
            shutil.copyfile(seal["path"], destination)
            if _sha256(destination) != seal["sha256"]:
                raise N0AAuditError("mirrored current input copy differs")
            current_paths[kind] = destination
        mirrored = FrameRecord(
            scene_position=source_frame.scene_position,
            scene_index=source_frame.scene_index,
            scene_id=source_frame.scene_id,
            frame_ordinal=source_frame.frame_ordinal,
            frame_id=source_frame.frame_id,
            rgb={"path": os.fspath(current_paths["rgb"]), "sha256": source_frame.rgb["sha256"]},
            depth={"path": os.fspath(current_paths["depth"]), "sha256": source_frame.depth["sha256"]},
            pose={"path": os.fspath(current_paths["pose"]), "sha256": source_frame.pose["sha256"]},
            intrinsic={"path": os.fspath(current_paths["intrinsic"]), "sha256": source_frame.intrinsic["sha256"]},
            sources=source_frame.sources,
        )
        authenticated_warning_count = 0
        baseline_rows, warning_count = run_current(mirrored)
        authenticated_warning_count += warning_count
        baseline_sha = _canonical_json_sha256(baseline_rows)

        future_frame_id = source_frame.frame_id + 25
        future_paths = {
            "rgb": root / f"frame_{future_frame_id:06d}.color{current_paths['rgb'].suffix}",
            "depth": root / f"frame_{future_frame_id:06d}.depth{current_paths['depth'].suffix}",
            "pose": root / f"frame_{future_frame_id:06d}.pose{current_paths['pose'].suffix}",
        }
        case_hashes: dict[str, str] = {}

        for kind in ("rgb", "depth", "pose"):
            shutil.copyfile(current_paths[kind], future_paths[kind])
        added_rows, warning_count = run_current(mirrored)
        authenticated_warning_count += warning_count
        case_hashes["future_files_added"] = _canonical_json_sha256(added_rows)

        for ordinal, kind in enumerate(("rgb", "depth", "pose"), start=1):
            future_paths[kind].write_bytes(
                f"N0a explicit future-only altered fixture {ordinal}\n".encode("ascii")
            )
        altered_rows, warning_count = run_current(mirrored)
        authenticated_warning_count += warning_count
        case_hashes["future_files_altered"] = _canonical_json_sha256(altered_rows)

        for kind in ("rgb", "depth", "pose"):
            future_paths[kind].unlink()
        deleted_rows, warning_count = run_current(mirrored)
        authenticated_warning_count += warning_count
        case_hashes["future_files_deleted"] = _canonical_json_sha256(deleted_rows)
        future_paths_absent = all(not path.exists() for path in future_paths.values())
        passed = (
            all(value == baseline_sha for value in case_hashes.values())
            and future_paths_absent
            and authenticated_warning_count == 8
        )
        return {
            "authorizing": False,
            "evidence_strength": "non_authorizing_provider_core_mirror",
            "fresh_worker_execution": True,
            "actual_sam2_provider_and_core_executed": True,
            "complete_current_frame_batch_replayed": True,
            "source_count": len(source_frame.sources),
            "source_scene_id": source_frame.scene_id,
            "source_frame_ordinal": source_frame.frame_ordinal,
            "source_frame_id": source_frame.frame_id,
            "explicit_future_frame_id": future_frame_id,
            "real_dataset_file_mutation_performed": False,
            "mirrored_future_files_add_alter_delete_performed": True,
            "directory_search_performed": False,
            "future_path_passed_to_provider_or_core": False,
            "baseline_batch_result_sha256": baseline_sha,
            "case_batch_result_sha256": case_hashes,
            "future_paths_absent_after_fixture": future_paths_absent,
            "warning_policy": _expected_warning_policy(),
            "provider_forward_count": 4,
            "authenticated_warning_count": authenticated_warning_count,
            "expected_warning_count": 8,
            "warning_count_formula": "2 * provider_forward_count",
            "per_forward_exact_warning_pair_passed": True,
            "overall_pass": passed,
        }


def _fixture_input_mapping(
    frame: Mapping[str, Any], copied: Mapping[str, Path]
) -> dict[str, Any]:
    """Return only the current-frame fields consumed by the frozen runner."""

    original = frame.get("inputs")
    if not isinstance(original, Mapping):
        raise N0AAuditError("future fixture F0 frame inputs are absent")
    required = {
        "current_pose_valid",
        "f0_pose_forward_filled",
        "producer_orientation",
        "producer_rotation_k",
        "producer_depth_shape",
        "producer_image_shape",
    }
    if not required.issubset(original):
        raise N0AAuditError("future fixture F0 frame contract differs")
    result = {key: copy.deepcopy(original[key]) for key in sorted(required)}
    for kind in ("rgb", "depth", "pose"):
        path = copied[kind]
        result[f"{kind}_path"] = os.fspath(path)
        result[f"{kind}_sha256"] = _sha256(path)
    result["producer_pose_path"] = result["pose_path"]
    result["producer_pose_sha256"] = result["pose_sha256"]
    return result


def _fixture_frame_payload(
    frame: Mapping[str, Any],
    *,
    ordinal: int,
    copied: Mapping[str, Path],
    include_sources: bool,
) -> dict[str, Any]:
    """Whitelist the only F0 fields needed by the real two-frame runner gate."""

    frame_id = frame.get("frame_id")
    runtime = frame.get("runtime")
    funnel = frame.get("funnel")
    if (
        frame.get("successful") is not True
        or isinstance(frame_id, bool)
        or not isinstance(frame_id, int)
        or not isinstance(runtime, Mapping)
        or not isinstance(runtime.get("complete_ms"), (int, float))
        or isinstance(runtime.get("complete_ms"), bool)
        or not isinstance(funnel, Mapping)
    ):
        raise N0AAuditError("future fixture source frame contract differs")
    candidates_raw = funnel.get("candidates")
    masks_raw = funnel.get("masks")
    if not isinstance(candidates_raw, list) or not isinstance(masks_raw, list):
        raise N0AAuditError("future fixture source funnel differs")
    if not include_sources:
        if candidates_raw != []:
            raise N0AAuditError("future fixture later frame is not successful-empty")
        candidates: list[dict[str, Any]] = []
        selected_masks: list[dict[str, Any]] = []
    else:
        if not candidates_raw:
            raise N0AAuditError("future fixture current source batch is empty")
        candidate_keys = (
            "rank", "raw_index", "mask_sha256",
            "points_and_voxel_keys_sha256", "tight_box_xyxy",
            "world_q02", "world_q98", "world_center", "world_extent",
        )
        mask_keys = (
            "decision", "tight_box_xyxy", "selected", "rank", "raw_index",
            "mask_sha256",
        )
        candidates = []
        selected_masks = []
        for expected_rank, raw_candidate in enumerate(candidates_raw):
            if not isinstance(raw_candidate, Mapping):
                raise N0AAuditError("future fixture candidate differs")
            candidate = {
                key: copy.deepcopy(raw_candidate.get(key)) for key in candidate_keys
            }
            if candidate["rank"] != expected_rank:
                raise N0AAuditError("future fixture candidate rank differs")
            matches = [
                raw_mask
                for raw_mask in masks_raw
                if isinstance(raw_mask, Mapping)
                and raw_mask.get("selected") is True
                and raw_mask.get("rank") == candidate["rank"]
                and raw_mask.get("raw_index") == candidate["raw_index"]
                and raw_mask.get("mask_sha256") == candidate["mask_sha256"]
            ]
            if len(matches) != 1:
                raise N0AAuditError("future fixture selected-mask join differs")
            candidates.append(candidate)
            selected_masks.append(
                {key: copy.deepcopy(matches[0].get(key)) for key in mask_keys}
            )
    return {
        "abstention": None,
        "frame_id": frame_id,
        "frame_ordinal": ordinal,
        "funnel": {"candidates": candidates, "masks": selected_masks},
        "inputs": _fixture_input_mapping(frame, copied),
        "runtime": {"complete_ms": float(runtime["complete_ms"])},
        "successful": True,
    }


def _prepare_runner_future_case(
    *, fixture: Mapping[str, Any], case_name: str, case_root: Path
) -> dict[str, Any]:
    """Materialize one sealed, private, two-frame F0 universe."""

    if case_name not in FUTURE_CASES:
        raise N0AAuditError("unknown future perturbation case")
    scene_id = fixture.get("scene_id")
    current = fixture.get("current_frame")
    future = fixture.get("future_frame")
    input_seals = fixture.get("input_seals")
    intrinsic = fixture.get("intrinsic")
    if (
        not isinstance(scene_id, str)
        or not isinstance(current, Mapping)
        or not isinstance(future, Mapping)
        or not isinstance(input_seals, Mapping)
        or not isinstance(intrinsic, Mapping)
    ):
        raise N0AAuditError("future fixture source differs")
    data_root = case_root / "data"
    data_root.mkdir(parents=True, exist_ok=False)

    copied: dict[str, dict[str, Path]] = {"current": {}, "future": {}}
    for prefix in ("current", "future"):
        for kind in ("rgb", "depth", "pose"):
            seal = input_seals.get(f"{prefix}_{kind}")
            if (
                not isinstance(seal, Mapping)
                or not isinstance(seal.get("path"), str)
                or not _valid_sha256(seal.get("sha256"))
            ):
                raise N0AAuditError(f"future fixture {prefix} {kind} seal differs")
            source = _regular_file(
                Path(str(seal["path"])), f"future fixture source {prefix} {kind}"
            )
            if _sha256(source) != seal["sha256"]:
                raise N0AAuditError(
                    f"future fixture source {prefix} {kind} rehash differs"
                )
            destination = data_root / f"{prefix}_{kind}{source.suffix}"
            shutil.copyfile(source, destination)
            if _sha256(destination) != seal["sha256"]:
                raise N0AAuditError(
                    f"future fixture copied {prefix} {kind} differs"
                )
            copied[prefix][kind] = destination.resolve()

    intrinsic_source = _regular_file(
        Path(str(intrinsic.get("path", ""))), "future fixture intrinsic", ".txt"
    )
    if (
        not _valid_sha256(intrinsic.get("sha256"))
        or _sha256(intrinsic_source) != intrinsic["sha256"]
    ):
        raise N0AAuditError("future fixture intrinsic seal differs")
    intrinsic_copy = data_root / "intrinsic.txt"
    shutil.copyfile(intrinsic_source, intrinsic_copy)

    original_future_rgb_sha = _sha256(copied["future"]["rgb"])
    unreferenced = data_root / (
        f"future_{int(future.get('frame_id', -1)):06d}_unreferenced.bin"
    )
    operations: list[str] = []
    unreferenced_sha_after_add: str | None = None
    unreferenced_sha_after_alter: str | None = None
    unreferenced_added_payload = b"N0A unreferenced future fixture: added\n"
    if case_name == "referenced_future_changed":
        shutil.copyfile(copied["current"]["rgb"], copied["future"]["rgb"])
        operations.append("referenced_future_rgb_changed")
        if _sha256(copied["future"]["rgb"]) == original_future_rgb_sha:
            raise N0AAuditError("referenced future RGB perturbation is inert")
    elif case_name == "unreferenced_future_added":
        unreferenced.write_bytes(unreferenced_added_payload)
        unreferenced_sha_after_add = _sha256(unreferenced)
        operations.append("unreferenced_future_file_added")
    elif case_name == "unreferenced_future_altered":
        unreferenced.write_bytes(unreferenced_added_payload)
        unreferenced_sha_after_add = _sha256(unreferenced)
        unreferenced.write_bytes(b"N0A unreferenced future fixture: altered\n")
        unreferenced_sha_after_alter = _sha256(unreferenced)
        operations.extend(
            ["unreferenced_future_file_added", "unreferenced_future_file_altered"]
        )
    elif case_name == "unreferenced_future_deleted":
        unreferenced.write_bytes(unreferenced_added_payload)
        unreferenced_sha_after_add = _sha256(unreferenced)
        unreferenced.unlink()
        operations.extend(
            ["unreferenced_future_file_added", "unreferenced_future_file_deleted"]
        )

    current_copy = _fixture_frame_payload(
        current, ordinal=0, copied=copied["current"], include_sources=True
    )
    future_copy = _fixture_frame_payload(
        future, ordinal=1, copied=copied["future"], include_sources=False
    )
    current_funnel = current_copy.get("funnel")
    future_funnel = future_copy.get("funnel")
    if (
        current_copy.get("successful") is not True
        or not isinstance(current_funnel, Mapping)
        or not isinstance(current_funnel.get("candidates"), list)
        or not current_funnel["candidates"]
        or future_copy.get("successful") is not True
        or not isinstance(future_funnel, Mapping)
        or future_funnel.get("candidates") != []
        or not isinstance(current_copy.get("frame_id"), int)
        or not isinstance(future_copy.get("frame_id"), int)
        or current_copy["frame_id"] >= future_copy["frame_id"]
    ):
        raise N0AAuditError("future fixture two-frame order/content differs")

    sidecar = {
        "schema": EXPECTED_F0_SCENE_SCHEMA,
        "protocol_id": EXPECTED_F0_PROTOCOL,
        "complete": True,
        "scene_id": scene_id,
        "scene_index": 0,
        "intrinsic": {
            "path": os.fspath(intrinsic_copy.resolve()),
            "sha256": _sha256(intrinsic_copy),
        },
        "frames": [current_copy, future_copy],
    }
    sidecar_path = case_root / "f0_scene.json"
    _atomic_create_json(sidecar_path, sidecar)
    scene_list_path = case_root / "scene_list.txt"
    scene_list_path.write_text(f"{scene_id}\n", encoding="utf-8")
    merge = {
        "schema": EXPECTED_F0_MERGE_SCHEMA,
        "protocol_id": EXPECTED_F0_PROTOCOL,
        "complete": True,
        "overall_pass": True,
        "coverage": {},
        "scenes": [
            {
                "scene_id": scene_id,
                "scene_index": 0,
                "sidecar": {
                    "path": os.fspath(sidecar_path.resolve()),
                    "sha256": _sha256(sidecar_path),
                },
            }
        ],
    }
    merge_path = case_root / "f0_merge.json"
    _atomic_create_json(merge_path, merge)
    return {
        "scene_id": scene_id,
        "source_count": len(current_funnel["candidates"]),
        "merge_path": merge_path,
        "scene_list_path": scene_list_path,
        "output_root": case_root / "runner_output",
        "data_root": data_root.resolve(),
        "current_paths": [
            os.fspath(copied["current"][kind])
            for kind in ("rgb", "depth", "pose")
        ],
        "future_paths": [
            os.fspath(copied["future"][kind])
            for kind in ("rgb", "depth", "pose")
        ],
        "unreferenced_path": os.fspath(unreferenced.resolve()),
        "mutation": {
            "operations": operations,
            "referenced_future_rgb_original_sha256": original_future_rgb_sha,
            "referenced_future_rgb_sealed_sha256": _sha256(
                copied["future"]["rgb"]
            ),
            "referenced_future_rgb_sha256_in_sidecar": future_copy["inputs"][
                "rgb_sha256"
            ],
            "sidecar_payload_after_seal_update": sidecar,
            "sidecar_sha256_after_seal_update": _sha256(sidecar_path),
            "unreferenced_sha256_after_add": unreferenced_sha_after_add,
            "unreferenced_sha256_after_alter": unreferenced_sha_after_alter,
            "unreferenced_exists_at_runner_start": unreferenced.exists(),
            "unreferenced_sha256_at_runner_start": (
                _sha256(unreferenced) if unreferenced.exists() else None
            ),
        },
    }


def _current_batch_fingerprint(
    *, scene_receipt: Mapping[str, Any], evidence_path: Path
) -> dict[str, Any]:
    frames = scene_receipt.get("frames")
    if not isinstance(frames, list) or len(frames) != 2:
        raise N0AAuditError("future-gate runner frame receipt differs")
    current, future = frames
    if (
        not isinstance(current, Mapping)
        or not isinstance(future, Mapping)
        or current.get("frame_ordinal") != 0
        or future.get("frame_ordinal") != 1
        or current.get("provider_invoked") is not True
        or future.get("provider_invoked") is not False
        or future.get("sources") != []
        or future.get("runtime") is not None
    ):
        raise N0AAuditError("future-gate current/future runner boundary differs")
    sources = current.get("sources")
    current_input = current.get("input")
    if (
        not isinstance(sources, list)
        or not sources
        or not isinstance(current_input, Mapping)
        or any(
            not isinstance(current_input.get(kind), Mapping)
            or not _valid_sha256(current_input[kind].get("sha256"))
            for kind in ("rgb", "depth", "pose", "intrinsic")
        )
    ):
        raise N0AAuditError("future-gate current complete batch is absent")
    decoded = _load_evidence(evidence_path, "future-gate")
    arrays: list[dict[str, Any]] = []
    for name in sorted(decoded):
        array = np.ascontiguousarray(decoded[name])
        arrays.append(
            {
                "name": name,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
        )
    payload = {
        "current_frame_id": current.get("frame_id"),
        "current_input_sha256": {
            kind: current_input[kind]["sha256"]
            for kind in ("rgb", "depth", "pose", "intrinsic")
        },
        "source_count": len(sources),
        "source_ids": [source.get("source_id") for source in sources],
        "complete_source_rows": sources,
        "evidence_arrays": arrays,
    }
    return {
        **payload,
        "complete_current_batch_sha256": _canonical_json_sha256(payload),
    }


def _current_batch_fingerprint_valid(value: object) -> bool:
    """Independently bind every current source row and NPZ-array fingerprint."""

    if not isinstance(value, Mapping) or set(value) != {
        "current_frame_id", "current_input_sha256", "source_count", "source_ids",
        "complete_source_rows", "evidence_arrays",
        "complete_current_batch_sha256",
    }:
        return False
    frame_id = value.get("current_frame_id")
    current_input_sha256 = value.get("current_input_sha256")
    source_count = value.get("source_count")
    source_ids = value.get("source_ids")
    source_rows = value.get("complete_source_rows")
    metadata = value.get("evidence_arrays")
    if (
        isinstance(frame_id, bool)
        or not isinstance(frame_id, int)
        or isinstance(source_count, bool)
        or not isinstance(source_count, int)
        or source_count <= 0
        or not isinstance(current_input_sha256, Mapping)
        or set(current_input_sha256) != {"rgb", "depth", "pose", "intrinsic"}
        or any(not _valid_sha256(current_input_sha256.get(kind)) for kind in current_input_sha256)
        or not isinstance(source_ids, list)
        or len(source_ids) != source_count
        or any(not isinstance(source_id, str) or not source_id for source_id in source_ids)
        or len(set(source_ids)) != source_count
        or not isinstance(source_rows, list)
        or len(source_rows) != source_count
        or any(
            not isinstance(row, Mapping)
            or set(row) != SOURCE_ROW_KEYS
            or row.get("source_id") != source_ids[index]
            or row.get("evidence_index") != index
            for index, row in enumerate(source_rows)
        )
        or not isinstance(metadata, list)
        or len(metadata) != len(EVIDENCE_ARRAY_NAMES)
    ):
        return False
    if any(
        not isinstance(item, Mapping)
        or set(item) != {"name", "dtype", "shape", "sha256"}
        or not isinstance(item.get("name"), str)
        or not isinstance(item.get("dtype"), str)
        or not isinstance(item.get("shape"), list)
        or any(isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in item.get("shape", []))
        or not _valid_sha256(item.get("sha256"))
        for item in metadata
    ):
        return False
    if [item["name"] for item in metadata] != sorted(EVIDENCE_ARRAY_NAMES):
        return False
    by_name = {item["name"]: item for item in metadata}
    if set(by_name) != EVIDENCE_ARRAY_NAMES:
        return False
    points_shape = by_name["points_world"]["shape"]
    if len(points_shape) != 2 or points_shape[1:] != [3]:
        return False
    point_count = points_shape[0]
    previous_stop = 0
    for row in source_rows:
        point_offset = row.get("point_offset")
        if (
            not isinstance(point_offset, list)
            or len(point_offset) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in point_offset)
            or point_offset[0] != previous_stop
            or point_offset[1] < point_offset[0]
            or point_offset[1] - point_offset[0] > MAX_STORED_POINTS
        ):
            return False
        previous_stop = point_offset[1]
    if previous_stop != point_count or point_count > MAX_STORED_POINTS * source_count:
        return False
    expected_specs = {
        "schema_utf8": (
            "|u1", [len(EVIDENCE_SCHEMA.encode("utf-8"))],
        ),
        "mask_packbits": ("|u1", [source_count, MASK_PACKED_BYTES]),
        "points_world": ("<f8", [point_count, 3]),
        "voxel_keys": ("<i8", [point_count, 3]),
        "point_offsets": ("<i8", [source_count + 1]),
        "frame_ordinals": ("<i8", [source_count]),
        "frame_ids": ("<i8", [source_count]),
        "ranks": ("<i8", [source_count]),
        "raw_indices": ("<i8", [source_count]),
        "selected_hypothesis_indices": ("<i8", [source_count]),
        "predicted_ious": ("<f4", [source_count]),
        "all_predicted_ious": ("<f4", [source_count, 3]),
        "result_sha256_ascii": ("|S64", [source_count]),
    }
    if any(
        by_name[name]["dtype"] != dtype or by_name[name]["shape"] != shape
        for name, (dtype, shape) in expected_specs.items()
    ):
        return False
    if by_name["schema_utf8"]["sha256"] != hashlib.sha256(
        EVIDENCE_SCHEMA.encode("utf-8")
    ).hexdigest():
        return False
    payload = {
        "current_frame_id": frame_id,
        "current_input_sha256": dict(current_input_sha256),
        "source_count": source_count,
        "source_ids": source_ids,
        "complete_source_rows": source_rows,
        "evidence_arrays": metadata,
    }
    return bool(
        _valid_sha256(value.get("complete_current_batch_sha256"))
        and value.get("complete_current_batch_sha256")
        == _canonical_json_sha256(payload)
    )


def _future_case_entry(request_path: Path, output_path: Path) -> None:
    """Execute exactly one runner-level perturbation in this fresh process."""

    _, request = _read_json(request_path, "future case request")
    if (
        request.get("schema") != FUTURE_CASE_SCHEMA
        or request.get("mode") != "runner_level_future_case"
        or not _content_hash_valid(request)
        or set(request) != {
            "schema", "mode", "case_name", "fixture", "gpu_uuid",
            "environment_receipt", "request_nonce", "content_sha256",
        }
        or request.get("case_name") not in FUTURE_CASES
        or not isinstance(request.get("fixture"), Mapping)
        or not isinstance(request.get("gpu_uuid"), str)
        or not isinstance(request.get("environment_receipt"), Mapping)
        or not _valid_sha256(request.get("request_nonce"))
    ):
        raise N0AAuditError("future case request differs")
    from types import SimpleNamespace

    environment_bundle = SimpleNamespace(
        gpu_uuid=request["gpu_uuid"],
        environment_receipt=request["environment_receipt"],
    )
    _authenticate_runtime_environment(environment_bundle)
    case_name = str(request["case_name"])
    case_root = request_path.parent / "case"
    case_root.mkdir(parents=False, exist_ok=False)
    prepared = _prepare_runner_future_case(
        fixture=request["fixture"], case_name=case_name, case_root=case_root
    )

    data_root = os.fspath(prepared["data_root"])
    access_events: list[dict[str, str]] = []

    def record_path(event: str, raw_path: object) -> None:
        if not isinstance(raw_path, (str, bytes, os.PathLike)):
            return
        try:
            path = os.path.abspath(os.fsdecode(os.fspath(raw_path)))
            if os.path.commonpath([data_root, path]) != data_root:
                return
        except (OSError, TypeError, ValueError):
            return
        access_events.append({"event": event, "path": path})

    def audit_hook(event: str, args: tuple[object, ...]) -> None:
        if event in {"open", "os.listdir", "os.scandir", "glob.glob", "glob.glob/2"}:
            if args:
                record_path(event, args[0])

    sys.addaudithook(audit_hook)
    try:
        import cv2  # type: ignore
        from tools import run_scannet_sam2_n0a_extra100 as runner
    except ImportError as error:
        raise N0AAuditError("future-gate runner dependency is absent") from error
    original_imread = cv2.imread

    def traced_imread(path: object, *args: object, **kwargs: object) -> object:
        record_path("cv2.imread", path)
        return original_imread(path, *args, **kwargs)

    cv2.imread = traced_imread
    try:
        manifest = runner.run_n0a(
            f0_receipt_path=Path(prepared["merge_path"]),
            full200_scene_list_path=Path(prepared["scene_list_path"]),
            output_root=Path(prepared["output_root"]),
            shard_index=0,
            num_shards=1,
            cohort_start=0,
            expected_scene_count=1,
            expected_keyframes=2,
            expected_successful_frames=2,
            expected_sources=int(prepared["source_count"]),
            provider_factory=None,
            frame_loader=None,
            plan_only=False,
            resume=False,
        )
    finally:
        cv2.imread = original_imread

    if (
        manifest.get("schema") != SHARD_SCHEMA
        or manifest.get("complete") is not True
        or manifest.get("warning_policy") != _expected_warning_policy()
        or manifest.get("provider_config") != dict(EXPECTED_PROVIDER_CONFIG)
        or set(manifest.get("source_receipts", {}))
        != {"runner", "protocol", "core", "provider"}
    ):
        raise N0AAuditError("future-gate actual default runner receipt differs")
    expected_source_hashes = {
        "runner": EXPECTED_RUNNER_SHA256,
        "protocol": EXPECTED_PROTOCOL_SHA256,
        "core": EXPECTED_CORE_SHA256,
        "provider": EXPECTED_PROVIDER_SHA256,
    }
    for role, expected_sha in expected_source_hashes.items():
        if manifest["source_receipts"][role].get("sha256") != expected_sha:
            raise N0AAuditError(f"future-gate {role} source SHA differs")
    rows = manifest.get("scenes")
    totals = manifest.get("totals")
    if (
        not isinstance(rows, list)
        or len(rows) != 1
        or not isinstance(rows[0], Mapping)
        or not isinstance(totals, Mapping)
        or totals.get("provider_forward_count") != 1
        or totals.get("authenticated_warning_count") != 2
    ):
        raise N0AAuditError("future-gate runner warning/forward census differs")
    scene_path, scene_receipt = _read_json(
        Path(str(rows[0].get("sidecar", {}).get("path", ""))),
        "future-gate N0a scene",
    )
    if _sha256(scene_path) != rows[0].get("sidecar", {}).get("sha256"):
        raise N0AAuditError("future-gate scene seal differs")
    scene_frames = scene_receipt.get("frames")
    if (
        scene_receipt.get("warning_policy") != _expected_warning_policy()
        or scene_receipt.get("counts", {}).get("provider_forward_count") != 1
        or scene_receipt.get("counts", {}).get("authenticated_warning_count") != 2
        or not isinstance(scene_frames, list)
        or scene_frames[0].get("authenticated_warning_count") != 2
        or scene_frames[0].get("runtime", {}).get(
            "deterministic_warning_evidence"
        )
        != _expected_warning_evidence()
        or scene_frames[1].get("authenticated_warning_count") != 0
    ):
        raise N0AAuditError("future-gate exact per-forward warning evidence differs")
    evidence_path = _regular_file(
        Path(str(rows[0].get("evidence_npz", {}).get("path", ""))),
        "future-gate evidence",
        ".npz",
    )
    if _sha256(evidence_path) != rows[0].get("evidence_npz", {}).get("sha256"):
        raise N0AAuditError("future-gate evidence seal differs")
    fingerprint = _current_batch_fingerprint(
        scene_receipt=scene_receipt, evidence_path=evidence_path
    )
    if fingerprint["source_count"] != prepared["source_count"]:
        raise N0AAuditError("future-gate complete current batch census differs")

    normalized_future = {os.path.abspath(path) for path in prepared["future_paths"]}
    normalized_current = {os.path.abspath(path) for path in prepared["current_paths"]}
    normalized_unreferenced = os.path.abspath(str(prepared["unreferenced_path"]))
    forbidden_events = [
        row
        for row in access_events
        if row["path"] in normalized_future
        or row["path"] == normalized_unreferenced
    ]
    enumeration_events = [
        row
        for row in access_events
        if row["event"] in {"os.listdir", "os.scandir", "glob.glob", "glob.glob/2"}
    ]
    current_accessed = normalized_current.issubset(
        {row["path"] for row in access_events}
    )
    result: dict[str, Any] = {
        "schema": FUTURE_CASE_SCHEMA,
        "complete": True,
        "case_name": case_name,
        "fresh_process": True,
        "worker_pid": os.getpid(),
        "request_nonce": request["request_nonce"],
        "request_file_sha256": _sha256(request_path),
        "actual_default_runner_sam2_core": True,
        "provider_or_frame_loader_injected": False,
        "environment_gpu_determinism_authenticated": True,
        "warning_policy": _expected_warning_policy(),
        "provider_forward_count": 1,
        "authenticated_warning_count": 2,
        "expected_warning_count": 2,
        "per_forward_exact_warning_pair_passed": True,
        "per_forward_warning_evidence": [_expected_warning_evidence()],
        "complete_current_frame_batch_replayed": True,
        "current_batch": fingerprint,
        "mutation": prepared["mutation"],
        "path_access_instrumentation": {
            "python_audit_hook_events": [
                "open", "os.listdir", "os.scandir", "glob.glob", "glob.glob/2"
            ],
            "native_cv2_imread_wrapped_without_loader_injection": True,
            "events": access_events,
            "fixture_data_root": data_root,
            "expected_current_rgb_depth_pose_paths": sorted(normalized_current),
            "expected_future_rgb_depth_pose_paths": sorted(normalized_future),
            "expected_unreferenced_future_path": normalized_unreferenced,
            "current_rgb_depth_pose_accessed": current_accessed,
            "future_rgb_depth_pose_or_unreferenced_access_events": forbidden_events,
            "fixture_data_directory_enumeration_events": enumeration_events,
            "stronger_no_future_open_at_any_time_passed": not forbidden_events,
        },
        "overall_pass": bool(
            current_accessed and not forbidden_events and not enumeration_events
        ),
    }
    result["content_sha256"] = _canonical_json_sha256(result)
    _atomic_create_json(output_path, result)


def perform_runner_future_perturbation(bundle: AuditBundle) -> dict[str, Any]:
    """Run every future perturbation through a fresh real runner process."""

    case_results: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="n0a-runner-future-gate-") as directory:
        root = Path(directory)
        for case_name in FUTURE_CASES:
            case_dir = root / case_name
            case_dir.mkdir(parents=True, exist_ok=False)
            request_path = case_dir / "request.json"
            output_path = case_dir / "result.json"
            request: dict[str, Any] = {
                "schema": FUTURE_CASE_SCHEMA,
                "mode": "runner_level_future_case",
                "case_name": case_name,
                "fixture": bundle.future_fixture_source,
                "gpu_uuid": bundle.gpu_uuid,
                "environment_receipt": bundle.environment_receipt,
                "request_nonce": os.urandom(32).hex(),
            }
            request["content_sha256"] = _canonical_json_sha256(request)
            request_path.write_bytes(_canonical_json_bytes(request))
            environment = dict(os.environ)
            environment["PYTHONHASHSEED"] = "0"
            environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        os.fspath(Path(__file__).resolve()),
                        "--future-case-request",
                        os.fspath(request_path),
                        "--future-case-output",
                        os.fspath(output_path),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=None,
                    env=environment,
                )
            except OSError as error:
                raise N0AAuditError(
                    f"future-gate case {case_name} could not start"
                ) from error
            if completed.returncode != 0 or not output_path.is_file():
                stderr = completed.stderr.strip().splitlines()
                detail = stderr[-1][:500] if stderr else "no case output"
                raise N0AAuditError(
                    f"future-gate case {case_name} failed: {detail}"
                )
            _, result = _read_json(output_path, f"future-gate {case_name} result")
            if (
                result.get("schema") != FUTURE_CASE_SCHEMA
                or result.get("case_name") != case_name
                or not _content_hash_valid(result)
                or result.get("fresh_process") is not True
                or result.get("worker_pid") == os.getpid()
                or result.get("request_nonce") != request["request_nonce"]
                or result.get("request_file_sha256") != _sha256(request_path)
                or result.get("overall_pass") is not True
            ):
                raise N0AAuditError(f"future-gate case {case_name} receipt differs")
            case_results[case_name] = result

    batch_hashes = {
        name: result["current_batch"]["complete_current_batch_sha256"]
        for name, result in case_results.items()
    }
    baseline_hash = batch_hashes["baseline"]
    unique_pids = {int(result["worker_pid"]) for result in case_results.values()}
    total_forwards = sum(
        int(result["provider_forward_count"]) for result in case_results.values()
    )
    total_warnings = sum(
        int(result["authenticated_warning_count"])
        for result in case_results.values()
    )
    referenced = case_results["referenced_future_changed"]["mutation"]
    overall_pass = bool(
        set(case_results) == set(FUTURE_CASES)
        and all(value == baseline_hash for value in batch_hashes.values())
        and len(unique_pids) == len(FUTURE_CASES)
        and total_forwards == len(FUTURE_CASES)
        and total_warnings == 2 * total_forwards
        and referenced["referenced_future_rgb_original_sha256"]
        != referenced["referenced_future_rgb_sealed_sha256"]
        and all(
            result["path_access_instrumentation"][
                "stronger_no_future_open_at_any_time_passed"
            ]
            is True
            for result in case_results.values()
        )
    )
    receipt: dict[str, Any] = {
        "authorizing": True,
        "evidence_strength": "fresh_real_runner_two_frame_file_perturbation",
        "case_names": list(FUTURE_CASES),
        "fresh_distinct_process_count": len(unique_pids),
        "actual_default_runner_sam2_core_every_case": True,
        "complete_current_frame_batch_replayed_every_case": True,
        "warning_policy": _expected_warning_policy(),
        "provider_forward_count": total_forwards,
        "authenticated_warning_count": total_warnings,
        "expected_warning_count": 2 * total_forwards,
        "warning_count_formula": "2 * provider_forward_count",
        "per_forward_exact_warning_pair_passed": True,
        "baseline_current_batch_sha256": baseline_hash,
        "case_current_batch_sha256": batch_hashes,
        "referenced_future_content_changed_and_sidecar_resealed": (
            referenced["referenced_future_rgb_original_sha256"]
            != referenced["referenced_future_rgb_sealed_sha256"]
        ),
        "unreferenced_future_add_alter_delete_executed": True,
        "future_rgb_depth_pose_opened_at_any_time": False,
        "fixture_data_directory_enumerated": False,
        "cases": case_results,
        "overall_pass": overall_pass,
    }
    receipt["content_sha256"] = _canonical_json_sha256(receipt)
    return receipt


def _worker_entry(request_path: Path, output_path: Path) -> None:
    _, request = _read_json(request_path, "replay worker request")
    if request.get("schema") != WORKER_SCHEMA or request.get("mode") != "fresh_replay":
        raise N0AAuditError("replay worker request differs")
    paths = request.get("manifest_paths")
    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
        raise N0AAuditError("replay worker manifest paths differ")
    bundle = load_and_validate_bundle([Path(path) for path in paths])
    before = _snapshot_hash(bundle.input_seals)
    runner_future = perform_runner_future_perturbation(bundle)
    result = perform_replay(bundle, production_preflight=True)
    try:
        mirrored_future = perform_mirrored_future_perturbation(bundle)
    except (N0AAuditError, OSError, RuntimeError) as error:
        mirrored_future = {
            "authorizing": False,
            "evidence_strength": "non_authorizing_provider_core_mirror",
            "complete": False,
            "overall_pass": False,
            "diagnostic_error_type": type(error).__name__,
            "diagnostic_error": str(error)[:1000],
        }
    result["runner_level_future_perturbation"] = runner_future
    result["mirrored_future_only_file_perturbation"] = mirrored_future
    result["fresh_process"] = True
    result["worker_pid"] = os.getpid()
    passed, changed, after = _rehash_snapshot(bundle.input_seals)
    result["global_inputs_assets_before_sha256"] = before
    result["global_inputs_assets_after_sha256"] = after
    result["global_inputs_assets_unchanged"] = passed and before == after
    result["changed_paths"] = changed
    result["overall_pass"] = bool(
        result["overall_pass"]
        and runner_future["overall_pass"]
        and passed
        and before == after
    )
    result["content_sha256"] = _canonical_json_sha256(result)
    _atomic_create_json(output_path, result)


def spawn_fresh_replay(bundle: AuditBundle) -> dict[str, Any]:
    """Run replay via this auditor in a new interpreter and return its receipt."""

    with tempfile.TemporaryDirectory(prefix="n0a-audit-worker-") as directory:
        root = Path(directory)
        request_path = root / "request.json"
        output_path = root / "result.json"
        request = {
            "schema": WORKER_SCHEMA,
            "mode": "fresh_replay",
            "manifest_paths": [os.fspath(path) for path in bundle.manifest_paths],
        }
        request_path.write_bytes(_canonical_json_bytes(request))
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = "0"
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            completed = subprocess.run(
                [sys.executable, os.fspath(Path(__file__).resolve()), "--worker-request", os.fspath(request_path), "--worker-output", os.fspath(output_path)],
                check=False, capture_output=True, text=True, timeout=None, env=environment,
            )
        except OSError as error:
            raise N0AAuditError("fresh replay process could not start") from error
        if completed.returncode != 0 or not output_path.is_file():
            stderr = completed.stderr.strip().splitlines()
            detail = stderr[-1][:500] if stderr else "no worker output"
            raise N0AAuditError(f"fresh replay process failed: {detail}")
        _, result = _read_json(output_path, "fresh replay result")
        if result.get("schema") != WORKER_SCHEMA or not _content_hash_valid(result):
            raise N0AAuditError("fresh replay result receipt differs")
        if result.get("fresh_process") is not True or result.get("worker_pid") == os.getpid():
            raise N0AAuditError("replay did not run in a fresh process")
        return result


def future_only_isolation_fixture() -> dict[str, Any]:
    """Exercise causal hashing without touching or discovering real data files."""

    def earlier_result(current: Mapping[str, Any]) -> str:
        return _canonical_json_sha256(
            {"frame_ordinal": 3, "source_id": "fixture/frame_000075/raw_000", "current": current}
        )

    current = {"rgb_sha256": "1" * 64, "depth_sha256": "2" * 64, "pose_sha256": "3" * 64}
    baseline = earlier_result(current)
    future_sets = {
        "future_added": {4: {"rgb_sha256": "4" * 64}},
        "future_removed": {},
        "future_altered": {4: {"rgb_sha256": "5" * 64}},
    }
    results = {name: earlier_result(current) for name in future_sets}
    passed = all(value == baseline for value in results.values())
    return {
        "evidence_strength": "isolated_contract_fixture_only",
        "real_dataset_future_file_mutation_performed": False,
        "real_dataset_future_invariance_claimed": False,
        "directory_search_performed": False,
        "fixture_uses_explicit_in_memory_future_rows_only": True,
        "earlier_hash_before_sha256": baseline,
        "earlier_hash_after_by_case": results,
        "cases": sorted(future_sets),
        "overall_pass": passed,
    }


def _capacity_gates(bundle: AuditBundle) -> dict[str, Any]:
    gates = {
        "valid_hs_count": {"actual": bundle.counts["valid_hs_count"], "threshold": MIN_VALID_SOURCE_COUNT, "comparator": ">=", "passed": bundle.counts["valid_hs_count"] >= MIN_VALID_SOURCE_COUNT},
        "valid_scene_count": {"actual": bundle.valid_scene_count, "threshold": MIN_VALID_SCENE_COUNT, "comparator": ">=", "passed": bundle.valid_scene_count >= MIN_VALID_SCENE_COUNT},
        "nontrivial_hs_count": {"actual": bundle.counts["nontrivial_hs_count"], "threshold": MIN_NONTRIVIAL_SOURCE_COUNT, "comparator": ">=", "passed": bundle.counts["nontrivial_hs_count"] >= MIN_NONTRIVIAL_SOURCE_COUNT},
        "nontrivial_scene_count": {"actual": bundle.nontrivial_scene_count, "threshold": MIN_NONTRIVIAL_SCENE_COUNT, "comparator": ">=", "passed": bundle.nontrivial_scene_count >= MIN_NONTRIVIAL_SCENE_COUNT},
    }
    return {"gates": gates, "overall_pass": all(row["passed"] for row in gates.values())}


def _runtime_gates(bundle: AuditBundle) -> dict[str, Any]:
    incremental = _distribution(bundle.warm_incremental_ms)
    composed = _distribution(bundle.warm_composed_ms)
    mean_per_raw = float(composed["mean"]) / SOURCE_FRAME_STRIDE if composed["count"] else math.inf
    gates = {
        "n0a_incremental_warm_p95_ms": {"actual": incremental["p95"], "threshold": 250.0, "comparator": "<=", "passed": incremental["count"] > 0 and incremental["p95"] <= 250.0},
        "replay_composed_warm_p95_ms": {"actual": composed["p95"], "threshold": 500.0, "comparator": "<=", "passed": composed["count"] > 0 and composed["p95"] <= 500.0},
        "replay_composed_warm_max_ms": {"actual": composed["max"], "threshold": 833.33, "comparator": "<", "passed": composed["count"] > 0 and composed["max"] < 833.33},
        "replay_composed_mean_per_raw_frame_ms": {"actual": mean_per_raw, "threshold": 20.0, "comparator": "<=", "passed": mean_per_raw <= 20.0},
        "gap25_warm_deadline_miss_count": {"actual": bundle.deadline_misses, "threshold": 0, "comparator": "==", "passed": bundle.deadline_misses == 0},
        "cuda_peak_memory_bytes": {"actual": bundle.cuda_peak_memory_bytes, "threshold": MAX_CUDA_BYTES, "comparator": "<=", "passed": bundle.cuda_peak_memory_bytes <= MAX_CUDA_BYTES},
    }
    return {"warm_incremental_ms": incremental, "warm_composed_ms": composed, "gates": gates, "overall_pass": all(row["passed"] for row in gates.values())}


def final_decision(*, integrity_pass: bool, capacity_pass: bool, runtime_pass: bool) -> str:
    if not integrity_pass:
        return DISCARD_DECISION
    if not capacity_pass:
        return CAPACITY_STOP_DECISION
    if not runtime_pass:
        return RUNTIME_STOP_DECISION
    return RETAIN_DECISION


def _future_mutation_receipts_valid(cases: Mapping[str, Any]) -> bool:
    expected_operations = {
        "baseline": [],
        "referenced_future_changed": ["referenced_future_rgb_changed"],
        "unreferenced_future_added": ["unreferenced_future_file_added"],
        "unreferenced_future_altered": [
            "unreferenced_future_file_added",
            "unreferenced_future_file_altered",
        ],
        "unreferenced_future_deleted": [
            "unreferenced_future_file_added",
            "unreferenced_future_file_deleted",
        ],
    }
    mutations: dict[str, Mapping[str, Any]] = {}
    expected_keys = {
        "operations", "referenced_future_rgb_original_sha256",
        "referenced_future_rgb_sealed_sha256",
        "referenced_future_rgb_sha256_in_sidecar",
        "sidecar_payload_after_seal_update",
        "sidecar_sha256_after_seal_update",
        "unreferenced_sha256_after_add",
        "unreferenced_sha256_after_alter",
        "unreferenced_exists_at_runner_start",
        "unreferenced_sha256_at_runner_start",
    }
    for name in FUTURE_CASES:
        row = cases.get(name)
        mutation = row.get("mutation") if isinstance(row, Mapping) else None
        current_batch = row.get("current_batch") if isinstance(row, Mapping) else None
        sidecar = (
            mutation.get("sidecar_payload_after_seal_update")
            if isinstance(mutation, Mapping)
            else None
        )
        frames = sidecar.get("frames") if isinstance(sidecar, Mapping) else None
        current_frame = (
            frames[0] if isinstance(frames, list) and len(frames) == 2 else None
        )
        future_frame = (
            frames[1] if isinstance(frames, list) and len(frames) == 2 else None
        )
        current_inputs = (
            current_frame.get("inputs")
            if isinstance(current_frame, Mapping)
            else None
        )
        future_inputs = (
            future_frame.get("inputs")
            if isinstance(future_frame, Mapping)
            else None
        )
        if (
            not isinstance(mutation, Mapping)
            or set(mutation) != expected_keys
            or mutation.get("operations") != expected_operations[name]
            or not _valid_sha256(
                mutation.get("referenced_future_rgb_original_sha256")
            )
            or not _valid_sha256(
                mutation.get("referenced_future_rgb_sealed_sha256")
            )
            or mutation.get("referenced_future_rgb_sha256_in_sidecar")
            != mutation.get("referenced_future_rgb_sealed_sha256")
            or not _valid_sha256(mutation.get("sidecar_sha256_after_seal_update"))
            or not isinstance(sidecar, Mapping)
            or set(sidecar) != {
                "schema", "protocol_id", "complete", "scene_id", "scene_index",
                "intrinsic", "frames",
            }
            or sidecar.get("schema") != EXPECTED_F0_SCENE_SCHEMA
            or sidecar.get("protocol_id") != EXPECTED_F0_PROTOCOL
            or sidecar.get("complete") is not True
            or sidecar.get("scene_index") != 0
            or _created_json_sha256(sidecar)
            != mutation.get("sidecar_sha256_after_seal_update")
            or not isinstance(current_inputs, Mapping)
            or not isinstance(future_inputs, Mapping)
            or future_inputs.get("rgb_sha256")
            != mutation.get("referenced_future_rgb_sealed_sha256")
            or not isinstance(current_batch, Mapping)
            or not isinstance(current_batch.get("current_input_sha256"), Mapping)
            or any(
                current_inputs.get(f"{kind}_sha256")
                != current_batch["current_input_sha256"].get(kind)
                for kind in ("rgb", "depth", "pose")
            )
            or type(mutation.get("unreferenced_exists_at_runner_start")) is not bool
        ):
            return False
        mutations[name] = mutation
    originals = {
        mutation["referenced_future_rgb_original_sha256"]
        for mutation in mutations.values()
    }
    if len(originals) != 1:
        return False
    original = next(iter(originals))
    for name in (
        "baseline", "unreferenced_future_added", "unreferenced_future_altered",
        "unreferenced_future_deleted",
    ):
        if mutations[name]["referenced_future_rgb_sealed_sha256"] != original:
            return False
    if mutations["referenced_future_changed"][
        "referenced_future_rgb_sealed_sha256"
    ] == original:
        return False
    referenced_current = cases["referenced_future_changed"]["current_batch"][
        "current_input_sha256"
    ]["rgb"]
    if mutations["referenced_future_changed"][
        "referenced_future_rgb_sealed_sha256"
    ] != referenced_current:
        return False
    added = mutations["unreferenced_future_added"][
        "unreferenced_sha256_after_add"
    ]
    altered = mutations["unreferenced_future_altered"][
        "unreferenced_sha256_after_alter"
    ]
    if not _valid_sha256(added) or not _valid_sha256(altered) or added == altered:
        return False
    if any(
        mutations[name]["unreferenced_sha256_after_add"] != added
        for name in (
            "unreferenced_future_added", "unreferenced_future_altered",
            "unreferenced_future_deleted",
        )
    ):
        return False
    if any(
        mutations[name]["unreferenced_sha256_after_add"] is not None
        or mutations[name]["unreferenced_sha256_after_alter"] is not None
        or mutations[name]["unreferenced_exists_at_runner_start"] is not False
        or mutations[name]["unreferenced_sha256_at_runner_start"] is not None
        for name in ("baseline", "referenced_future_changed")
    ):
        return False
    added_case = mutations["unreferenced_future_added"]
    altered_case = mutations["unreferenced_future_altered"]
    deleted_case = mutations["unreferenced_future_deleted"]
    return bool(
        added_case["unreferenced_sha256_after_alter"] is None
        and added_case["unreferenced_exists_at_runner_start"] is True
        and added_case["unreferenced_sha256_at_runner_start"] == added
        and altered_case["unreferenced_sha256_at_runner_start"] == altered
        and altered_case["unreferenced_exists_at_runner_start"] is True
        and deleted_case["unreferenced_sha256_after_alter"] is None
        and deleted_case["unreferenced_exists_at_runner_start"] is False
        and deleted_case["unreferenced_sha256_at_runner_start"] is None
    )


def _runner_future_gate_authorizes(value: Mapping[str, Any]) -> bool:
    """Authenticate the causal gate independently of any summary booleans."""

    if (
        set(value) != {
            "authorizing", "evidence_strength", "case_names",
            "fresh_distinct_process_count",
            "actual_default_runner_sam2_core_every_case",
            "complete_current_frame_batch_replayed_every_case",
            "warning_policy", "provider_forward_count",
            "authenticated_warning_count", "expected_warning_count",
            "warning_count_formula", "per_forward_exact_warning_pair_passed",
            "baseline_current_batch_sha256", "case_current_batch_sha256",
            "referenced_future_content_changed_and_sidecar_resealed",
            "unreferenced_future_add_alter_delete_executed",
            "future_rgb_depth_pose_opened_at_any_time",
            "fixture_data_directory_enumerated", "cases", "overall_pass",
            "content_sha256",
        }
        or not _content_hash_valid(value)
    ):
        return False
    cases = value.get("cases")
    case_hashes = value.get("case_current_batch_sha256")
    if (
        not isinstance(cases, Mapping)
        or set(cases) != set(FUTURE_CASES)
        or not isinstance(case_hashes, Mapping)
        or set(case_hashes) != set(FUTURE_CASES)
        or value.get("case_names") != list(FUTURE_CASES)
    ):
        return False
    baseline_hash = value.get("baseline_current_batch_sha256")
    if (
        not _valid_sha256(baseline_hash)
        or any(case_hashes[name] != baseline_hash for name in FUTURE_CASES)
    ):
        return False
    expected_operations = {
        "baseline": [],
        "referenced_future_changed": ["referenced_future_rgb_changed"],
        "unreferenced_future_added": ["unreferenced_future_file_added"],
        "unreferenced_future_altered": [
            "unreferenced_future_file_added",
            "unreferenced_future_file_altered",
        ],
        "unreferenced_future_deleted": [
            "unreferenced_future_file_added",
            "unreferenced_future_file_deleted",
        ],
    }
    expected_unreferenced_exists = {
        "baseline": False,
        "referenced_future_changed": False,
        "unreferenced_future_added": True,
        "unreferenced_future_altered": True,
        "unreferenced_future_deleted": False,
    }
    pids: set[int] = set()
    nonces: set[str] = set()
    for name in FUTURE_CASES:
        row = cases[name]
        if not isinstance(row, Mapping):
            return False
        current = row.get("current_batch")
        access = row.get("path_access_instrumentation")
        mutation = row.get("mutation")
        pid = row.get("worker_pid")
        nonce = row.get("request_nonce")
        expected_case_keys = {
            "schema", "complete", "case_name", "fresh_process", "worker_pid",
            "request_nonce", "request_file_sha256",
            "actual_default_runner_sam2_core",
            "provider_or_frame_loader_injected",
            "environment_gpu_determinism_authenticated", "warning_policy",
            "provider_forward_count", "authenticated_warning_count",
            "expected_warning_count", "per_forward_exact_warning_pair_passed",
            "complete_current_frame_batch_replayed", "current_batch", "mutation",
            "path_access_instrumentation", "overall_pass", "content_sha256",
        }
        if (
            set(row) != expected_case_keys
            or not _content_hash_valid(row)
            or row.get("schema") != FUTURE_CASE_SCHEMA
            or row.get("complete") is not True
            or row.get("case_name") != name
            or row.get("fresh_process") is not True
            or isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid <= 0
            or not _valid_sha256(nonce)
            or not _valid_sha256(row.get("request_file_sha256"))
            or row.get("actual_default_runner_sam2_core") is not True
            or row.get("provider_or_frame_loader_injected") is not False
            or row.get("environment_gpu_determinism_authenticated") is not True
            or row.get("warning_policy") != _expected_warning_policy()
            or type(row.get("provider_forward_count")) is not int
            or row.get("provider_forward_count") != 1
            or type(row.get("authenticated_warning_count")) is not int
            or row.get("authenticated_warning_count") != 2
            or type(row.get("expected_warning_count")) is not int
            or row.get("expected_warning_count") != 2
            or row.get("per_forward_exact_warning_pair_passed") is not True
            or row.get("complete_current_frame_batch_replayed") is not True
            or row.get("overall_pass") is not True
            or not isinstance(current, Mapping)
            or set(current) != {
                "current_frame_id", "source_count", "source_ids",
                "complete_source_rows", "evidence_arrays",
                "complete_current_batch_sha256",
            }
            or current.get("complete_current_batch_sha256") != case_hashes[name]
            or isinstance(current.get("source_count"), bool)
            or not isinstance(current.get("source_count"), int)
            or current.get("source_count") <= 0
            or not isinstance(current.get("source_ids"), list)
            or len(current["source_ids"]) != current["source_count"]
            or len(set(current["source_ids"])) != current["source_count"]
            or not isinstance(current.get("complete_source_rows"), list)
            or len(current["complete_source_rows"]) != current["source_count"]
            or not isinstance(current.get("evidence_arrays"), list)
            or not current["evidence_arrays"]
            or not isinstance(access, Mapping)
            or set(access) != {
                "python_audit_hook_events",
                "native_cv2_imread_wrapped_without_loader_injection", "events",
                "fixture_data_root", "expected_current_rgb_depth_pose_paths",
                "expected_future_rgb_depth_pose_paths",
                "expected_unreferenced_future_path",
                "current_rgb_depth_pose_accessed",
                "future_rgb_depth_pose_or_unreferenced_access_events",
                "fixture_data_directory_enumeration_events",
                "stronger_no_future_open_at_any_time_passed",
            }
            or access.get("native_cv2_imread_wrapped_without_loader_injection")
            is not True
            or access.get("current_rgb_depth_pose_accessed") is not True
            or access.get("future_rgb_depth_pose_or_unreferenced_access_events")
            != []
            or access.get("fixture_data_directory_enumeration_events") != []
            or access.get("stronger_no_future_open_at_any_time_passed") is not True
            or not isinstance(mutation, Mapping)
            or set(mutation) != {
                "operations", "referenced_future_rgb_original_sha256",
                "referenced_future_rgb_sealed_sha256",
                "sidecar_sha256_after_seal_update",
                "unreferenced_exists_at_runner_start",
                "unreferenced_sha256_at_runner_start",
            }
            or mutation.get("operations") != expected_operations[name]
            or mutation.get("unreferenced_exists_at_runner_start")
            is not expected_unreferenced_exists[name]
        ):
            return False
        evidence_arrays = current["evidence_arrays"]
        if any(
            not isinstance(item, Mapping)
            or set(item) != {"name", "dtype", "shape", "sha256"}
            or not isinstance(item.get("name"), str)
            or not isinstance(item.get("dtype"), str)
            or not isinstance(item.get("shape"), list)
            or not _valid_sha256(item.get("sha256"))
            for item in evidence_arrays
        ) or len({item["name"] for item in evidence_arrays}) != len(evidence_arrays):
            return False
        expected_current_paths = access.get(
            "expected_current_rgb_depth_pose_paths"
        )
        expected_future_paths = access.get("expected_future_rgb_depth_pose_paths")
        unreferenced_path = access.get("expected_unreferenced_future_path")
        data_root = access.get("fixture_data_root")
        events = access.get("events")
        if (
            access.get("python_audit_hook_events")
            != ["open", "os.listdir", "os.scandir", "glob.glob", "glob.glob/2"]
            or not isinstance(data_root, str)
            or not isinstance(expected_current_paths, list)
            or len(expected_current_paths) != 3
            or len(set(expected_current_paths)) != 3
            or not isinstance(expected_future_paths, list)
            or len(expected_future_paths) != 3
            or len(set(expected_future_paths)) != 3
            or not isinstance(unreferenced_path, str)
            or not isinstance(events, list)
            or any(
                not isinstance(event, Mapping)
                or set(event) != {"event", "path"}
                or not isinstance(event.get("event"), str)
                or not isinstance(event.get("path"), str)
                for event in events
            )
        ):
            return False
        event_paths = {event["path"] for event in events}
        forbidden_paths = set(expected_future_paths) | {unreferenced_path}
        enumeration = [
            event
            for event in events
            if event["event"]
            in {"os.listdir", "os.scandir", "glob.glob", "glob.glob/2"}
        ]
        try:
            all_paths_scoped = all(
                os.path.commonpath([data_root, path]) == data_root
                for path in (
                    list(expected_current_paths)
                    + list(expected_future_paths)
                    + [unreferenced_path]
                    + [event["path"] for event in events]
                )
            )
        except ValueError:
            return False
        if (
            not all_paths_scoped
            or not set(expected_current_paths).issubset(event_paths)
            or event_paths.intersection(forbidden_paths)
            or enumeration
        ):
            return False
        pids.add(pid)
        nonces.add(str(nonce))
    referenced = cases["referenced_future_changed"]["mutation"]
    return bool(
        value.get("authorizing") is True
        and value.get("evidence_strength")
        == "fresh_real_runner_two_frame_file_perturbation"
        and value.get("overall_pass") is True
        and type(value.get("fresh_distinct_process_count")) is int
        and value.get("fresh_distinct_process_count") == len(FUTURE_CASES)
        and len(pids) == len(FUTURE_CASES)
        and len(nonces) == len(FUTURE_CASES)
        and value.get("actual_default_runner_sam2_core_every_case") is True
        and value.get("complete_current_frame_batch_replayed_every_case") is True
        and value.get("warning_policy") == _expected_warning_policy()
        and type(value.get("provider_forward_count")) is int
        and value.get("provider_forward_count") == len(FUTURE_CASES)
        and type(value.get("authenticated_warning_count")) is int
        and value.get("authenticated_warning_count") == 2 * len(FUTURE_CASES)
        and type(value.get("expected_warning_count")) is int
        and value.get("expected_warning_count") == 2 * len(FUTURE_CASES)
        and value.get("warning_count_formula") == "2 * provider_forward_count"
        and value.get("per_forward_exact_warning_pair_passed") is True
        and value.get("referenced_future_content_changed_and_sidecar_resealed")
        is True
        and referenced.get("referenced_future_rgb_original_sha256")
        != referenced.get("referenced_future_rgb_sealed_sha256")
        and value.get("unreferenced_future_add_alter_delete_executed") is True
        and value.get("future_rgb_depth_pose_opened_at_any_time") is False
        and value.get("fixture_data_directory_enumerated") is False
    )


def _replay_authorizes(
    replay: Mapping[str, Any],
    *,
    gpu_uuid: str,
    expected_frame_batches: int,
    expected_sources: int,
) -> bool:
    future = replay.get("runner_level_future_perturbation")
    if not isinstance(future, Mapping) or not _runner_future_gate_authorizes(future):
        return False
    mirrored = replay.get("mirrored_future_only_file_perturbation")
    if isinstance(mirrored, Mapping) and mirrored.get("authorizing") is not False:
        return False
    return bool(
        replay.get("schema") == WORKER_SCHEMA
        and replay.get("overall_pass") is True
        and replay.get("complete") is True
        and replay.get("fresh_process") is True
        and replay.get("same_gpu_uuid") == gpu_uuid
        and replay.get("mismatch_count") == 0
        and replay.get("mismatches") == []
        and replay.get("selector")
        == "sha256(source_id.encode('ascii'))[:2]_big_endian_lt_0x0290"
        and replay.get("first_full_scene_count") == FIRST_FULL_REPLAY_SCENES
        and replay.get("full_batch_for_sampled_frame") is True
        and replay.get("scheduled_frame_batch_count") == expected_frame_batches
        and replay.get("provider_forward_count") == expected_frame_batches
        and replay.get("scheduled_source_comparison_count") == expected_sources
        and replay.get("compared_source_count") == expected_sources
        and expected_sources > 0
        and replay.get("warning_policy") == _expected_warning_policy()
        and replay.get("authenticated_warning_count")
        == 2 * expected_frame_batches
        and replay.get("expected_warning_count") == 2 * expected_frame_batches
        and replay.get("warning_count_formula") == "2 * provider_forward_count"
        and replay.get("per_forward_exact_warning_pair_passed") is True
        and replay.get("global_inputs_assets_unchanged") is True
        and replay.get("global_inputs_assets_before_sha256")
        == replay.get("global_inputs_assets_after_sha256")
    )


def audit_n0a(
    *,
    manifest_paths: Sequence[Path],
    output_path: Path,
    replay_executor: Callable[[AuditBundle], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the complete production audit and create its sole final receipt."""

    bundle = load_and_validate_bundle(manifest_paths)
    before = _snapshot_hash(bundle.input_seals)
    future_contract_fixture = future_only_isolation_fixture()
    try:
        replay = dict(
            replay_executor(bundle)
            if replay_executor is not None
            else spawn_fresh_replay(bundle)
        )
    except (N0AAuditError, OSError, subprocess.SubprocessError) as error:
        replay = {
            "schema": WORKER_SCHEMA,
            "complete": False,
            "fresh_process": False,
            "fresh_process_requested": replay_executor is None,
            "same_gpu_uuid": bundle.gpu_uuid,
            "mismatch_count": None,
            "overall_pass": False,
            "fatal_error_type": type(error).__name__,
            "fatal_error": str(error)[:1000],
            "mirrored_future_only_file_perturbation": None,
        }
    if replay_executor is not None:
        # Dependency injection is useful for unit tests but cannot authorize a
        # production retain decision.
        replay["fresh_process"] = False
        replay["overall_pass"] = False
        replay["injected_executor_non_authorizing"] = True
    unchanged, changed, after = _rehash_snapshot(bundle.input_seals)
    expected_schedule = replay_schedule(bundle.frames)
    replay_pass = bool(
        replay_executor is None
        and _replay_authorizes(
            replay,
            gpu_uuid=bundle.gpu_uuid,
            expected_frame_batches=len(expected_schedule),
            expected_sources=sum(len(indices) for _, indices in expected_schedule),
        )
    )
    integrity_pass = bool(
        replay_pass and unchanged and before == after
    )
    capacity = _capacity_gates(bundle)
    runtime = _runtime_gates(bundle)
    decision = final_decision(
        integrity_pass=integrity_pass,
        capacity_pass=bool(capacity["overall_pass"]),
        runtime_pass=bool(runtime["overall_pass"]),
    )
    receipt: dict[str, Any] = {
        "schema": MERGE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "contracts": dict(CONTRACTS),
        "auditor_is_only_final_decision_authority": True,
        "upstream_runner_decisions_authoritative": False,
        "upstream_runner_decisions": bundle.runner_decisions,
        "inputs": {
            "shard_manifests": [
                {"path": os.fspath(path), "sha256": _sha256(path)}
                for path in bundle.manifest_paths
            ],
            "auditor_source": {
                "path": os.fspath(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "source_receipt_sha256": bundle.source_receipt_hashes,
            "global_inputs_assets_before_sha256": before,
            "global_inputs_assets_after_sha256": after,
            "global_inputs_assets_unchanged": unchanged and before == after,
            "changed_paths": changed,
            "seal_count": len(bundle.input_seals),
        },
        "coverage": dict(bundle.counts),
        "ledger_hashes": bundle.ledger_hashes,
        "determinism_audit": {
            "fresh_same_gpu_replay": replay,
            "future_only_contract_fixture_non_authorizing": future_contract_fixture,
            "future_only_authorizing_evidence": replay.get(
                "runner_level_future_perturbation"
            ),
            "provider_core_mirror_non_authorizing": replay.get(
                "mirrored_future_only_file_perturbation"
            ),
            "overall_pass": integrity_pass,
        },
        "capacity_gates": capacity,
        "runtime_gates": runtime,
        "integrity_and_determinism_pass": integrity_pass,
        "capacity_gates_overall_pass": capacity["overall_pass"],
        "runtime_gates_overall_pass": runtime["overall_pass"],
        "decision": decision,
        "native_output_mutation_count": 0,
        "n0b_or_gt_stage_authorized": decision == RETAIN_DECISION,
        "actual_prediction_or_ap_claim": False,
    }
    receipt["content_sha256"] = _canonical_json_sha256(receipt)
    receipt_sha = _atomic_create_json(Path(output_path), receipt)
    receipt["receipt_path"] = os.fspath(Path(output_path).resolve())
    receipt["receipt_sha256"] = receipt_sha
    return receipt


def _create_fail_closed_receipt(
    *, manifest_paths: Sequence[Path], output_path: Path, error: Exception
) -> dict[str, Any]:
    """Seal a final discard decision when authentication cannot complete."""

    manifest_rows: list[dict[str, Any]] = []
    for raw_path in manifest_paths:
        path = Path(raw_path)
        row: dict[str, Any] = {"requested_path": os.fspath(path)}
        try:
            regular = _regular_file(path, "failed-audit shard manifest", ".json")
            row.update({"path": os.fspath(regular), "sha256": _sha256(regular)})
        except (N0AAuditError, OSError):
            row["available_regular_file"] = False
        manifest_rows.append(row)
    receipt: dict[str, Any] = {
        "schema": MERGE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "contracts": dict(CONTRACTS),
        "auditor_is_only_final_decision_authority": True,
        "upstream_runner_decisions_authoritative": False,
        "coverage_authenticated": False,
        "inputs": {"requested_shard_manifests": manifest_rows},
        "integrity_and_determinism_pass": False,
        "capacity_gates_overall_pass": None,
        "runtime_gates_overall_pass": None,
        "decision": DISCARD_DECISION,
        "fatal_error_type": type(error).__name__,
        "fatal_error": str(error)[:1000],
        "native_output_mutation_count": 0,
        "n0b_or_gt_stage_authorized": False,
        "actual_prediction_or_ap_claim": False,
    }
    receipt["content_sha256"] = _canonical_json_sha256(receipt)
    receipt_sha = _atomic_create_json(output_path, receipt)
    receipt["receipt_path"] = os.fspath(output_path.resolve())
    receipt["receipt_sha256"] = receipt_sha
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-manifest", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--future-case-request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--future-case-output", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.future_case_request is not None or args.future_case_output is not None:
        if (
            args.future_case_request is None
            or args.future_case_output is None
            or args.worker_request is not None
            or args.worker_output is not None
            or args.shard_manifest
            or args.output is not None
        ):
            raise N0AAuditError("future case worker mode arguments differ")
        _future_case_entry(args.future_case_request, args.future_case_output)
        return
    if args.worker_request is not None or args.worker_output is not None:
        if (
            args.worker_request is None
            or args.worker_output is None
            or args.shard_manifest
            or args.output is not None
        ):
            raise N0AAuditError("worker mode arguments differ")
        _worker_entry(args.worker_request, args.worker_output)
        return
    if args.output is None or not args.shard_manifest:
        raise N0AAuditError("production audit requires explicit --shard-manifest entries and --output")
    if args.output.exists():
        raise N0AAuditError(f"refusing to overwrite audit output: {args.output}")
    try:
        receipt = audit_n0a(
            manifest_paths=args.shard_manifest, output_path=args.output
        )
    except (N0AAuditError, OSError, subprocess.SubprocessError) as error:
        receipt = _create_fail_closed_receipt(
            manifest_paths=args.shard_manifest,
            output_path=args.output,
            error=error,
        )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
