#!/usr/bin/env python3
"""Run the frozen dev3 Boxer-Top4 MobileSAM per-view lifting shadow.

The executable has deliberately no annotation/evaluation input surface.  It
selects the already sealed Boxer rows by frozen score, pairs each row to its
exact numeric OWL source index, and exports every per-view MobileSAM mask and
bounded RGB-D point fragment.  S3a performs no tracking, suppression, fusion,
or birth; those stages are conditional on a separate read-only audit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "boxfusion.boxer_mobilesam_masklift_shadow.v1"
DEV3_SCENES = ("scene0568_00", "scene0606_01", "scene0377_02")

SEALED_ROOT = (
    REPOSITORY_ROOT
    / "logs"
    / "scannet_boxer_unexplained_shadow_clean_in2_v5_score05"
)
SEALED_JSON = SEALED_ROOT / "sealed" / "boxer_shadow_candidates.json"
SEALED_NPZ = SEALED_ROOT / "sealed" / "boxer_shadow_candidates.npz"
BOXER_RAW_ROOT = SEALED_ROOT / "boxer_raw"
SCHEDULE_ROOT = (
    REPOSITORY_ROOT
    / "cache"
    / "cutr_postfilter_v3"
    / "scannet-graw-e2-score05-preflight3-v3-r1"
)
SCENE_ROOT = REPOSITORY_ROOT / "upstream_clean" / "scannet_readme_frames"
SCENE_LIST = (
    REPOSITORY_ROOT
    / "evaluation"
    / "data_util"
    / "meta_data"
    / "scannetv2_graw_e2_preflight3.txt"
)
FORMAL_T05_ROOT = REPOSITORY_ROOT / "results" / "scannet_topk_fusion_score05"
EXPECTED_FORMAL_T05_SHA256: Mapping[str, str] = {
    "scene0568_00": "b55ce48fb6eb4dad9ee5bfe7007c3dbc9898b3f72ddbc5ad428b8be6414bcd2d",
    "scene0606_01": "d4e8d6dc85c917ac1634b81a45adb3866279d3e02f470c43b23bd71f5bb3ef1c",
    "scene0377_02": "ed7f849a33d45eebe846559a90aeb7de1a97f2eb169c3a7c0cb5de61d3dab35b",
}

PREREGISTRATION = REPOSITORY_ROOT / "docs" / "S3_FROZEN_PROPOSAL_SOURCE_AUDIT.md"
TOPK_RECEIPT = (
    REPOSITORY_ROOT
    / "logs"
    / "scannet_boxer_per_view_topk_raw_ceiling_score05_dev3_v5.json"
)
TOPK_TOOL = REPOSITORY_ROOT / "tools" / "audit_scannet_boxer_per_view_topk_ceiling.py"
RUNTIME_RECEIPT = (
    REPOSITORY_ROOT
    / "logs"
    / "scannet_s3_mobilesam_runtime"
    / "mobilesam_top4_rtx3090_receipt.json"
)
RUNTIME_TOOL = REPOSITORY_ROOT / "tools" / "benchmark_mobilesam_boxprompt.py"
OBJECT_MEMORY_SOURCE = (
    REPOSITORY_ROOT
    / "tools"
    / "boxfusion_tr3d_pipeline"
    / "boxfusion"
    / "object_memory.py"
)

MOBILESAM_ROOT = Path(
    "/data/ZhaoX/RoboFusion/RoboFusion-master/focalconvsamfusion/OpenPCDet/"
    "pcdet/models/backbones_3d/focal_sparse_conv/MobileSAM"
)
MOBILESAM_CHECKPOINT = MOBILESAM_ROOT / "weights" / "mobile_sam.pt"
BOXERNET_SOURCE = Path(
    "/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer/"
    "boxernet/boxernet.py"
)
BOXER_FILE_IO_SOURCE = Path(
    "/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer/"
    "utils/file_io.py"
)

OUTPUT_JSON_NAME = "s3a_boxer_mobilesam_masklift_shadow.json"
OUTPUT_NPZ_NAME = "s3a_boxer_mobilesam_masklift_shadow.npz"

TOP_K = 4
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
MASK_PACKED_BYTES = (IMAGE_HEIGHT * IMAGE_WIDTH + 7) // 8
MIN_CLEAN_VOXELS = 16

EXPECTED_HASHES: Mapping[str, tuple[Path, str]] = {
    "sealed_json": (
        SEALED_JSON,
        "84eb4f2c62d1573d9e9f1ec4c3df5a6cac16ad10c8cece0989d37dd97b734e9e",
    ),
    "sealed_npz": (
        SEALED_NPZ,
        "c1a921d70de447bf528711a71deb34cf93a9bf671d3514baafa42b7b1b8b4a6c",
    ),
    "scene_list": (
        SCENE_LIST,
        "117b5bea04c557f52d4c2a9435c3961bbaae66e420fb5bb849a278f89fe454fc",
    ),
    "preregistration": (
        PREREGISTRATION,
        "ee742d4b0b9d3e26208ed8b59e587ed6de046ed850a22b80314fd8f939cad191",
    ),
    "topk_receipt": (
        TOPK_RECEIPT,
        "d4ba67b37d362842333ac525abe32f6807c4fba90af83b699bbfc1494aa5ea1f",
    ),
    "topk_tool": (
        TOPK_TOOL,
        "9a756f474e40e7b991453b09cb006b1147432aab124a55d33e4613d2adad1b44",
    ),
    "runtime_receipt": (
        RUNTIME_RECEIPT,
        "a1769f0186d2cabcfcfea9330a508cc8701d1b07b6ba4a66c35e9922ba489c14",
    ),
    "runtime_tool": (
        RUNTIME_TOOL,
        "062d29814df5123fe207d3be7a0862d42327a605e4f2e8b5e0c530914c993eb0",
    ),
    "object_memory_source": (
        OBJECT_MEMORY_SOURCE,
        "c2f3f0e0753a34430f0d9d03c65039aa6eee80114a1337676ec4b5f1eaa60938",
    ),
    "mobilesam_checkpoint": (
        MOBILESAM_CHECKPOINT,
        "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f",
    ),
    "boxernet_source": (
        BOXERNET_SOURCE,
        "a8009c1c0932aaab98bb074a2a4c50e55a3fbdfc3c6cb1afc9e1aef0e5324130",
    ),
    "boxer_file_io_source": (
        BOXER_FILE_IO_SOURCE,
        "72b140e7e235571e734e70c4f8c682de133cf1a16615f8c250a046df93ae1ee9",
    ),
}

EXPECTED_MOBILESAM_SOURCE_HASHES: Mapping[str, tuple[Path, str]] = {
    "build_sam": (
        MOBILESAM_ROOT / "mobile_sam" / "build_sam.py",
        "6d24c42834216cde03a7b7a441242f8a947ddb968f1013c8fa988dde4400bcf9",
    ),
    "predictor": (
        MOBILESAM_ROOT / "mobile_sam" / "predictor.py",
        "6dafa1d20430f6c4e0d37c53bfa80a8241619abb4264d26ca30d843991b760ff",
    ),
    "tiny_vit": (
        MOBILESAM_ROOT / "mobile_sam" / "modeling" / "tiny_vit_sam.py",
        "30beab10f151e3ffb49a4a63043ff48c59ebe46c15ada113eafe1ff58b3c102d",
    ),
}

EXPECTED_SELECTION_SHA256 = (
    "68049b78dba86441a6b691d1687b9fd2c90fc22f9f6e4c7c78548cc64384b306"
)
EXPECTED_CHECKPOINT_BYTES = 40_728_226
EXPECTED_INPUT_SCHEMA = "boxfusion.owl_boxer_shadow_candidates.v1"

EXPECTED_NPZ_INPUT_ARRAYS = {
    "per_view_center_world",
    "per_view_extent_xyz",
    "per_view_frame_id",
    "per_view_quaternion_wxyz",
    "per_view_scene_index",
    "per_view_source_instance_id",
    "per_view_source_row",
    "per_view_source_score",
    "scene_ids",
    "tracked_center_world",
    "tracked_extent_xyz",
    "tracked_instance_id",
    "tracked_quaternion_wxyz",
    "tracked_scene_index",
    "tracked_source_row",
    "tracked_source_score",
}

OWL_HEADER = (
    b"time_ns,frame_id,sensor,device,img_width,img_height,x1,y1,x2,y2,"
    b"name,instance,sem_id,prob"
)
BOXER_HEADER = (
    b"time_ns,tx_world_object,ty_world_object,tz_world_object,qw_world_object,"
    b"qx_world_object,qy_world_object,qz_world_object,scale_x,scale_y,scale_z,"
    b"name,instance,sem_id,prob"
)

ABSTENTION_CODES: Mapping[int, str] = {
    0: "emitted_q02_q98",
    1: "invalid_or_degenerate_prompt_box",
    2: "empty_mobilesam_mask",
    3: "no_valid_depth_after_fixed_cleaning",
    4: "fewer_than_16_unique_clean_voxels",
}

OBJECT_MEMORY_CONFIG: Mapping[str, object] = {
    "enabled": True,
    "min_depth": 0.10,
    "max_depth": 6.00,
    "depth_scale": 1000.0,
    "mask_threshold": 0.50,
    "mask_edge_margin": 1,
    "depth_edge_threshold": 0.15,
    "voxel_size": 0.02,
    "max_points_per_observation": 2048,
    "max_points_per_object": 8192,
    "aabb_lower_quantile": 0.02,
    "aabb_upper_quantile": 0.98,
    "min_points_for_aabb": MIN_CLEAN_VOXELS,
    "minimum_aabb_dimension": 0.02,
    "min_confirmations": 3,
    "track_ttl": 10,
    "association_iou_threshold": 0.05,
    "association_center_distance": 0.75,
    "association_inside_fraction": 0.25,
}


class S3aShadowError(ValueError):
    """Raised when a frozen input or output-inert contract check fails."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    raw = path
    resolved = raw.resolve()
    if raw.is_symlink() or not resolved.is_file():
        raise S3aShadowError(f"{label} must be a regular non-symlink file: {raw}")
    return resolved


def _read_json(path: Path, label: str) -> dict[str, Any]:
    path = _regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise S3aShadowError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise S3aShadowError(f"{label} must contain a JSON object: {path}")
    return value


def _hash_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _array_content_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("xb") as raw:
        with zipfile.ZipFile(
            raw, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name in sorted(arrays):
                payload = io.BytesIO()
                np.lib.format.write_array(
                    payload, np.ascontiguousarray(arrays[name]), allow_pickle=False
                )
                info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, payload.getvalue(), compresslevel=9)
        raw.flush()
        os.fsync(raw.fileno())


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_fixed_assets() -> dict[str, dict[str, object]]:
    ledger: dict[str, dict[str, object]] = {}
    for name, (raw_path, expected) in EXPECTED_HASHES.items():
        path = _regular_file(raw_path, f"frozen {name}")
        actual = _sha256(path)
        if actual != expected:
            raise S3aShadowError(
                f"frozen {name} SHA-256 mismatch: expected={expected}, actual={actual}"
            )
        ledger[name] = {
            "path": os.fspath(path),
            "sha256": actual,
            "bytes": path.stat().st_size,
        }
    if ledger["mobilesam_checkpoint"]["bytes"] != EXPECTED_CHECKPOINT_BYTES:
        raise S3aShadowError("MobileSAM checkpoint byte size changed")
    for name, (raw_path, expected) in EXPECTED_MOBILESAM_SOURCE_HASHES.items():
        path = _regular_file(raw_path, f"MobileSAM {name} source")
        actual = _sha256(path)
        if actual != expected:
            raise S3aShadowError(f"MobileSAM {name} source SHA-256 mismatch")
        ledger[f"mobilesam_source_{name}"] = {
            "path": os.fspath(path),
            "sha256": actual,
            "bytes": path.stat().st_size,
        }
    scene_lines = tuple(
        line.strip()
        for line in SCENE_LIST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if scene_lines != DEV3_SCENES:
        raise S3aShadowError("frozen dev3 scene order changed")
    topk = _read_json(TOPK_RECEIPT, "Top-K ceiling receipt")
    budget = topk.get("budgets", {}).get(str(TOP_K), {})
    if (
        budget.get("selection_sha256") != EXPECTED_SELECTION_SHA256
        or budget.get("top_k_per_frame") != TOP_K
        or budget.get("candidate_count") != 814
    ):
        raise S3aShadowError("Top-K ceiling receipt no longer binds frozen K=4")
    runtime = _read_json(RUNTIME_RECEIPT, "MobileSAM runtime receipt")
    if (
        runtime.get("schema") != "boxfusion.mobilesam_boxprompt_runtime_receipt.v1"
        or runtime.get("gt_access") is not False
        or runtime.get("input", {}).get("resized_hw") != [IMAGE_HEIGHT, IMAGE_WIDTH]
        or runtime.get("input", {}).get("box_count") != TOP_K
        or runtime.get("input", {}).get("multimask_output") is not True
    ):
        raise S3aShadowError("MobileSAM runtime receipt contract mismatch")
    return ledger


def _load_object_memory_module() -> ModuleType:
    name = "boxfusion_s3a_frozen_object_memory"
    spec = importlib.util.spec_from_file_location(name, OBJECT_MEMORY_SOURCE)
    if spec is None or spec.loader is None:
        raise S3aShadowError("could not load frozen object-memory source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        sys.modules.pop(name, None)
        raise S3aShadowError("could not import frozen object-memory source") from error
    resolved = module.resolve_object_memory_config(OBJECT_MEMORY_CONFIG)
    if resolved != dict(OBJECT_MEMORY_CONFIG):
        raise S3aShadowError("fixed object-memory profile did not resolve exactly")
    return module


def _load_sealed_candidates() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = _read_json(SEALED_JSON, "sealed Boxer JSON")
    required = {
        "schema": EXPECTED_INPUT_SCHEMA,
        "profile": "clean_in2",
        "mode": "shadow",
        "output_inert": True,
        "birth": False,
        "gt_access": False,
        "gt_access_guard_verified": True,
        "semantic_source_exported": False,
        "coordinate_frame": "scannet_world",
        "per_view_candidate_count": 3085,
        "scene_count": len(DEV3_SCENES),
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise S3aShadowError(
                f"sealed Boxer contract mismatch for {key}: {manifest.get(key)!r}"
            )
    if (
        manifest.get("npz_file") != SEALED_NPZ.name
        or manifest.get("npz_sha256") != EXPECTED_HASHES["sealed_npz"][1]
    ):
        raise S3aShadowError("sealed Boxer JSON does not bind the expected NPZ")
    assets = manifest.get("assets_and_protocol")
    if not isinstance(assets, Mapping):
        raise S3aShadowError("sealed Boxer asset ledger is absent")
    expected_assets = {
        "profile": "clean_in2",
        "detector": "owl",
        "detector_hw": 960,
        "threshold_2d": 0.25,
        "threshold_3d": 0.5,
        "nms_iou_2d": 0.5,
        "start_n": 1,
        "skip_n": 25,
        "boxernet_source_sha256": EXPECTED_HASHES["boxernet_source"][1],
    }
    for key, expected in expected_assets.items():
        if assets.get(key) != expected:
            raise S3aShadowError(f"sealed Boxer asset mismatch for {key}")
    try:
        with np.load(SEALED_NPZ, allow_pickle=False) as source:
            if set(source.files) != EXPECTED_NPZ_INPUT_ARRAYS:
                raise S3aShadowError("unexpected sealed Boxer NPZ array schema")
            arrays = {name: np.array(source[name], copy=True) for name in source.files}
    except (OSError, ValueError) as error:
        if isinstance(error, S3aShadowError):
            raise
        raise S3aShadowError("could not load sealed Boxer NPZ") from error
    if manifest.get("candidate_content_sha256") != _array_content_sha256(arrays):
        raise S3aShadowError("sealed Boxer candidate content SHA-256 mismatch")
    if tuple(str(value) for value in arrays["scene_ids"].tolist()) != DEV3_SCENES:
        raise S3aShadowError("sealed Boxer scene order changed")
    count = len(arrays["per_view_scene_index"])
    expected_shapes = {
        "per_view_scene_index": (count,),
        "per_view_frame_id": (count,),
        "per_view_source_row": (count,),
        "per_view_source_instance_id": (count,),
        "per_view_source_score": (count,),
        "per_view_center_world": (count, 3),
        "per_view_extent_xyz": (count, 3),
        "per_view_quaternion_wxyz": (count, 4),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise S3aShadowError(f"sealed array shape mismatch for {name}")
    integer_names = (
        "per_view_scene_index",
        "per_view_frame_id",
        "per_view_source_row",
        "per_view_source_instance_id",
    )
    if any(arrays[name].dtype.kind not in "iu" for name in integer_names):
        raise S3aShadowError("sealed integer provenance array has wrong dtype")
    numeric = np.concatenate(
        [
            arrays["per_view_source_score"].reshape(-1),
            arrays["per_view_center_world"].reshape(-1),
            arrays["per_view_extent_xyz"].reshape(-1),
            arrays["per_view_quaternion_wxyz"].reshape(-1),
        ]
    )
    if not np.isfinite(numeric).all():
        raise S3aShadowError("sealed Boxer geometry contains non-finite values")
    if (
        np.any(arrays["per_view_extent_xyz"] <= 0.0)
        or np.any(arrays["per_view_source_score"] < 0.0)
        or np.any(arrays["per_view_source_score"] >= 1.0)
    ):
        raise S3aShadowError("sealed Boxer geometry or score is out of range")
    scene_index = arrays["per_view_scene_index"]
    if np.any((scene_index < 0) | (scene_index >= len(DEV3_SCENES))):
        raise S3aShadowError("sealed Boxer scene index is out of range")
    ledgers = manifest.get("scenes")
    if not isinstance(ledgers, list) or len(ledgers) != len(DEV3_SCENES):
        raise S3aShadowError("sealed Boxer per-scene ledger is invalid")
    for index, (scene, ledger) in enumerate(zip(DEV3_SCENES, ledgers)):
        if (
            not isinstance(ledger, Mapping)
            or ledger.get("scene_id") != scene
            or ledger.get("scene_index") != index
            or ledger.get("gt_access_guard_verified") is not True
            or ledger.get("per_view_extra_schedule_rows_excluded") != 0
        ):
            raise S3aShadowError(f"sealed scene ledger mismatch for {scene}")
        positions = np.flatnonzero(scene_index == index)
        if ledger.get("per_view_kept_rows") != len(positions):
            raise S3aShadowError(f"sealed row count mismatch for {scene}")
        if len(np.unique(arrays["per_view_source_row"][positions])) != len(positions):
            raise S3aShadowError(f"duplicate Boxer source row in {scene}")
    return manifest, arrays


def _selection_sha256(indices_by_scene: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for scene_index, values in enumerate(indices_by_scene):
        array = np.ascontiguousarray(values, dtype=np.int64)
        digest.update(np.asarray([scene_index, len(array)], dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _select_top4(
    arrays: Mapping[str, np.ndarray],
) -> tuple[tuple[np.ndarray, ...], str]:
    selections: list[np.ndarray] = []
    scene_array = arrays["per_view_scene_index"]
    frames = arrays["per_view_frame_id"]
    scores = arrays["per_view_source_score"]
    source_rows = arrays["per_view_source_row"]
    for scene_index in range(len(DEV3_SCENES)):
        scene_positions = np.flatnonzero(scene_array == scene_index)
        selected: list[int] = []
        for frame_id in sorted(np.unique(frames[scene_positions]).tolist()):
            positions = scene_positions[frames[scene_positions] == frame_id]
            order = sorted(
                positions.tolist(),
                key=lambda row: (
                    -float(scores[row]),
                    int(source_rows[row]),
                    int(row),
                ),
            )
            selected.extend(order[:TOP_K])
        selections.append(np.asarray(selected, dtype=np.int64))
    result = tuple(selections)
    digest = _selection_sha256(result)
    if digest != EXPECTED_SELECTION_SHA256:
        raise S3aShadowError(
            f"Top4 selection drifted: expected={EXPECTED_SELECTION_SHA256}, actual={digest}"
        )
    counts = tuple(len(values) for values in result)
    if counts != (262, 436, 116):
        raise S3aShadowError(f"unexpected Top4 scene counts: {counts}")
    return result, digest


def _parse_ascii_int(value: bytes, label: str) -> int:
    try:
        decoded = value.decode("ascii")
        result = int(decoded)
    except (UnicodeDecodeError, ValueError) as error:
        raise S3aShadowError(f"invalid integer {label}") from error
    return result


def _parse_ascii_float(value: bytes, label: str) -> float:
    try:
        decoded = value.decode("ascii")
        result = float(decoded)
    except (UnicodeDecodeError, ValueError) as error:
        raise S3aShadowError(f"invalid floating value {label}") from error
    if not math.isfinite(result):
        raise S3aShadowError(f"non-finite floating value {label}")
    return result


def _read_owl_numeric_rows(path: Path) -> tuple[list[dict[str, Any]], dict[int, list[int]]]:
    """Parse numeric OWL fields without decoding or retaining semantic columns."""

    path = _regular_file(path, "frozen OWL CSV")
    lines = path.read_bytes().splitlines()
    if not lines or lines[0].rstrip(b"\r") != OWL_HEADER:
        raise S3aShadowError(f"unexpected OWL CSV header: {path}")
    rows: list[dict[str, Any]] = []
    grouped: dict[int, list[int]] = {}
    for data_row, raw_line in enumerate(lines[1:]):
        fields = raw_line.rstrip(b"\r").split(b",")
        if len(fields) != 14:
            raise S3aShadowError(
                f"OWL CSV row {data_row + 2} is not simple 14-column CSV"
            )
        time_ns = _parse_ascii_int(fields[0], "OWL time_ns")
        frame_ordinal = _parse_ascii_int(fields[1], "OWL frame_id")
        width = _parse_ascii_int(fields[4], "OWL img_width")
        height = _parse_ascii_int(fields[5], "OWL img_height")
        if (width, height) != (960, 960):
            raise S3aShadowError("OWL detector image dimensions changed")
        box = np.asarray(
            [_parse_ascii_float(fields[index], f"OWL box column {index}") for index in range(6, 10)],
            dtype=np.float32,
        )
        probability = _parse_ascii_float(fields[13], "OWL source probability")
        if not 0.0 <= probability <= 1.0:
            raise S3aShadowError("OWL source probability is outside [0,1]")
        rows.append(
            {
                "data_row": data_row,
                "line_number": data_row + 2,
                "time_ns": time_ns,
                "frame_ordinal": frame_ordinal,
                "box_xyxy_960": box,
            }
        )
        grouped.setdefault(time_ns, []).append(data_row)
    return rows, grouped


def _read_boxer_numeric_rows(path: Path) -> list[dict[str, Any]]:
    """Parse numeric Boxer fields without decoding or retaining semantic columns."""

    path = _regular_file(path, "frozen Boxer CSV")
    lines = path.read_bytes().splitlines()
    if not lines or lines[0].rstrip(b"\r") != BOXER_HEADER:
        raise S3aShadowError(f"unexpected Boxer CSV header: {path}")
    rows: list[dict[str, Any]] = []
    for data_row, raw_line in enumerate(lines[1:]):
        fields = raw_line.rstrip(b"\r").split(b",")
        if len(fields) != 15:
            raise S3aShadowError(
                f"Boxer CSV row {data_row + 2} is not simple 15-column CSV"
            )
        rows.append(
            {
                "data_row": data_row,
                "line_number": data_row + 2,
                "time_ns": _parse_ascii_int(fields[0], "Boxer time_ns"),
                "center_recentered": np.asarray(
                    [
                        _parse_ascii_float(fields[index], f"Boxer center {index}")
                        for index in range(1, 4)
                    ],
                    dtype=np.float32,
                ),
                "quaternion_wxyz": np.asarray(
                    [
                        _parse_ascii_float(fields[index], f"Boxer quaternion {index}")
                        for index in range(4, 8)
                    ],
                    dtype=np.float32,
                ),
                "extent_xyz": np.asarray(
                    [
                        _parse_ascii_float(fields[index], f"Boxer extent {index}")
                        for index in range(8, 11)
                    ],
                    dtype=np.float32,
                ),
                "instance_id": _parse_ascii_int(fields[12], "Boxer instance"),
                "probability": _parse_ascii_float(fields[14], "Boxer probability"),
            }
        )
    return rows


def _map_owl_box_to_depth(box_xyxy_960: np.ndarray) -> np.ndarray:
    box = np.asarray(box_xyxy_960, dtype=np.float64).reshape(-1)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise S3aShadowError("OWL box must contain four finite coordinates")
    mapped = box * np.asarray([2.0 / 3.0, 0.5, 2.0 / 3.0, 0.5])
    mapped[[0, 2]] = np.clip(mapped[[0, 2]], 0.0, float(IMAGE_WIDTH))
    mapped[[1, 3]] = np.clip(mapped[[1, 3]], 0.0, float(IMAGE_HEIGHT))
    return mapped.astype(np.float32)


def _hash_formal_t05_predictions() -> dict[str, dict[str, str]]:
    """Hash the three fixed formal-T05 files without trusting Boxer ledgers."""

    root = FORMAL_T05_ROOT.resolve()
    if FORMAL_T05_ROOT.is_symlink() or not root.is_dir():
        raise S3aShadowError(
            f"formal T05 root must be a regular directory: {FORMAL_T05_ROOT}"
        )
    expected_root = (
        REPOSITORY_ROOT / "results" / "scannet_topk_fusion_score05"
    ).resolve()
    if root != expected_root:
        raise S3aShadowError(
            f"formal T05 root mismatch: expected={expected_root}, actual={root}"
        )
    if set(EXPECTED_FORMAL_T05_SHA256) != set(DEV3_SCENES):
        raise S3aShadowError("formal T05 expected-hash scene set is invalid")
    output: dict[str, dict[str, str]] = {}
    for scene in DEV3_SCENES:
        path = _regular_file(
            root / f"{scene}_boxes.pkl", f"formal T05 prediction for {scene}"
        )
        actual = _sha256(path)
        expected = EXPECTED_FORMAL_T05_SHA256[scene]
        if actual != expected:
            raise S3aShadowError(
                f"formal T05 SHA-256 mismatch for {scene}: "
                f"expected={expected}, actual={actual}"
            )
        output[scene] = {
            "path": os.fspath(path),
            "sha256": actual,
            "expected_sha256": expected,
        }
    return output


def _frame_paths(scene: str, frame_id: int) -> dict[str, Path]:
    frames = SCENE_ROOT / scene / "frames"
    return {
        "rgb": frames / "color" / f"{frame_id}.jpg",
        "depth": frames / "depth" / f"{frame_id}.png",
        "pose": frames / "pose" / f"{frame_id}.txt",
    }


def _load_schedule_contract(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    scene_ledgers = manifest.get("scenes")
    if not isinstance(scene_ledgers, list):
        raise S3aShadowError("sealed Boxer scene ledgers are missing")
    sealed_by_scene = {row.get("scene_id"): row for row in scene_ledgers}
    output: dict[str, dict[str, Any]] = {}
    for scene in DEV3_SCENES:
        sealed = sealed_by_scene.get(scene)
        if not isinstance(sealed, Mapping):
            raise S3aShadowError(f"sealed schedule ledger missing for {scene}")
        schedule_path = _regular_file(
            SCHEDULE_ROOT / scene / "manifest.json", f"sealed schedule for {scene}"
        )
        schedule_hash = _sha256(schedule_path)
        if schedule_hash != sealed.get("sealed_schedule_manifest_sha256"):
            raise S3aShadowError(f"sealed gap25 schedule hash changed for {scene}")
        schedule = _read_json(schedule_path, f"sealed schedule for {scene}")
        frames = schedule.get("recorded_frame_ids")
        if (
            not isinstance(frames, list)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in frames)
            or frames != sorted(set(frames))
            or schedule.get("record_count") != len(frames)
            or len(frames) != sealed.get("sealed_manifest_record_count")
        ):
            raise S3aShadowError(f"invalid fixed frame schedule for {scene}")
        invalid = sealed.get("sealed_schedule_invalid_pose_frame_ids_excluded")
        if (
            not isinstance(invalid, list)
            or any(not isinstance(value, int) for value in invalid)
            or not set(invalid).issubset(frames)
        ):
            raise S3aShadowError(f"invalid pose-abstention ledger for {scene}")
        valid = tuple(value for value in frames if value not in set(invalid))
        if len(valid) != sealed.get("sealed_schedule_frame_count"):
            raise S3aShadowError(f"valid sealed schedule count changed for {scene}")
        intrinsic_path = _regular_file(
            SCENE_ROOT
            / scene
            / "frames"
            / "intrinsic"
            / "intrinsic_depth.txt",
            f"depth intrinsics for {scene}",
        )
        try:
            intrinsic = np.loadtxt(intrinsic_path, dtype=np.float64)
        except (OSError, ValueError) as error:
            raise S3aShadowError(f"invalid depth intrinsics for {scene}") from error
        if (
            intrinsic.shape != (4, 4)
            or not np.isfinite(intrinsic).all()
            or intrinsic[0, 0] <= 0.0
            or intrinsic[1, 1] <= 0.0
        ):
            raise S3aShadowError(f"invalid depth intrinsics matrix for {scene}")
        output[scene] = {
            "manifest_path": schedule_path,
            "manifest_sha256": schedule_hash,
            "recorded_frame_ids": tuple(frames),
            "valid_frame_ids": valid,
            "invalid_pose_frame_ids": tuple(invalid),
            # Boxer/OWL frame_id is the ordinal in the valid-pose provider
            # stream.  The raw manifest ordinal is retained separately so the
            # one invalid scene0606 frame can never shift provenance silently.
            "schedule_ordinal": {frame_id: index for index, frame_id in enumerate(valid)},
            "manifest_ordinal": {frame_id: index for index, frame_id in enumerate(frames)},
            "intrinsic_path": intrinsic_path,
            "intrinsic": intrinsic[:3, :3].copy(),
        }
    return output


def _schedule_raw_input_ledger(
    schedules: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for scene in DEV3_SCENES:
        schedule = schedules[scene]
        digest = hashlib.sha256()
        individual_count = 0
        observed_invalid: list[int] = []
        intrinsic_path = _regular_file(
            Path(schedule["intrinsic_path"]), f"depth intrinsics for {scene}"
        )
        intrinsic_hash = _sha256(intrinsic_path)
        digest.update(b"intrinsic\0")
        digest.update(intrinsic_hash.encode("ascii"))
        for frame_id in schedule["recorded_frame_ids"]:
            paths = _frame_paths(scene, int(frame_id))
            for kind in ("rgb", "depth", "pose"):
                path = _regular_file(paths[kind], f"{scene} frame {frame_id} {kind}")
                file_hash = _sha256(path)
                digest.update(f"{frame_id}:{kind}".encode("ascii"))
                digest.update(b"\0")
                digest.update(file_hash.encode("ascii"))
                individual_count += 1
            try:
                pose = np.loadtxt(paths["pose"], dtype=np.float64)
            except (OSError, ValueError):
                pose = np.empty((0, 0), dtype=np.float64)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                observed_invalid.append(int(frame_id))
        if tuple(observed_invalid) != tuple(schedule["invalid_pose_frame_ids"]):
            raise S3aShadowError(
                f"observed invalid poses differ from frozen schedule for {scene}"
            )
        output[scene] = {
            "schedule_manifest_sha256": schedule["manifest_sha256"],
            "raw_rgb_depth_pose_intrinsic_ledger_sha256": digest.hexdigest(),
            "raw_file_count": individual_count + 1,
            "recorded_frame_count": len(schedule["recorded_frame_ids"]),
            "valid_frame_count": len(schedule["valid_frame_ids"]),
            "invalid_pose_frame_ids": list(schedule["invalid_pose_frame_ids"]),
        }
    return output


def _validate_pair_and_build_records(
    *,
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    selections: Sequence[np.ndarray],
    schedules: Mapping[str, Mapping[str, Any]],
    requested_scenes: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    selected_scenes = set(requested_scenes)
    records: list[dict[str, Any]] = []
    source_summary: dict[str, dict[str, Any]] = {}
    sealed_by_scene = {row["scene_id"]: row for row in manifest["scenes"]}
    for scene_index, scene in enumerate(DEV3_SCENES):
        owl_path = _regular_file(
            BOXER_RAW_ROOT / scene / "owl_2dbbs.csv", f"OWL CSV for {scene}"
        )
        boxer_path = _regular_file(
            BOXER_RAW_ROOT / scene / "boxer_3dbbs.csv", f"Boxer CSV for {scene}"
        )
        sealed = sealed_by_scene[scene]
        expected_inputs = sealed.get("inputs")
        if not isinstance(expected_inputs, Mapping):
            raise S3aShadowError(f"sealed CSV ledger missing for {scene}")
        owl_hash = _sha256(owl_path)
        boxer_hash = _sha256(boxer_path)
        if owl_hash != expected_inputs.get("owl_2dbbs_csv_sha256"):
            raise S3aShadowError(f"OWL CSV changed for {scene}")
        if boxer_hash != expected_inputs.get("boxer_3dbbs_csv_sha256"):
            raise S3aShadowError(f"Boxer CSV changed for {scene}")
        owl_rows, owl_groups = _read_owl_numeric_rows(owl_path)
        boxer_rows = _read_boxer_numeric_rows(boxer_path)
        positions = selections[scene_index]
        schedule = schedules[scene]
        valid_frames = set(schedule["valid_frame_ids"])
        world_offset = np.asarray(sealed["world_offset_xyz"], dtype=np.float32)
        for npz_row in positions.tolist():
            frame_id = int(arrays["per_view_frame_id"][npz_row])
            source_row = int(arrays["per_view_source_row"][npz_row])
            instance_id = int(arrays["per_view_source_instance_id"][npz_row])
            if frame_id not in valid_frames:
                raise S3aShadowError(f"selected Boxer row is outside valid schedule: {scene}")
            if not 0 <= source_row < len(boxer_rows):
                raise S3aShadowError(f"Boxer source row is out of range for {scene}")
            boxer = boxer_rows[source_row]
            if (
                boxer["time_ns"] != frame_id
                or boxer["instance_id"] != instance_id
                or not math.isclose(
                    boxer["probability"],
                    float(arrays["per_view_source_score"][npz_row]),
                    rel_tol=0.0,
                    abs_tol=5e-7,
                )
                or not np.allclose(
                    boxer["center_recentered"] + world_offset,
                    arrays["per_view_center_world"][npz_row],
                    rtol=0.0,
                    atol=5e-6,
                )
                or not np.allclose(
                    boxer["quaternion_wxyz"],
                    arrays["per_view_quaternion_wxyz"][npz_row],
                    rtol=0.0,
                    atol=5e-6,
                )
                or not np.allclose(
                    boxer["extent_xyz"],
                    arrays["per_view_extent_xyz"][npz_row],
                    rtol=0.0,
                    atol=5e-6,
                )
            ):
                raise S3aShadowError(f"sealed Boxer numeric provenance mismatch for {scene}")
            group = owl_groups.get(frame_id)
            if group is None or not 0 <= instance_id < len(group):
                raise S3aShadowError(f"OWL source instance is out of range for {scene}")
            owl_global_row = group[instance_id]
            owl = owl_rows[owl_global_row]
            schedule_ordinal = int(schedule["schedule_ordinal"][frame_id])
            if (
                owl["time_ns"] != frame_id
                or owl["frame_ordinal"] != schedule_ordinal
            ):
                raise S3aShadowError(f"OWL timestamp/frame ordinal mismatch for {scene}")
            if scene in selected_scenes:
                records.append(
                    {
                        "scene": scene,
                        "scene_index_full": scene_index,
                        "frame_id": frame_id,
                        "schedule_ordinal": schedule_ordinal,
                        "manifest_schedule_ordinal": int(
                            schedule["manifest_ordinal"][frame_id]
                        ),
                        "sealed_npz_row": npz_row,
                        "boxer_source_row": source_row,
                        "boxer_csv_line_number": int(boxer["line_number"]),
                        "source_instance_id": instance_id,
                        "owl_csv_source_row": int(owl["data_row"]),
                        "owl_csv_line_number": int(owl["line_number"]),
                        "source_score": float(arrays["per_view_source_score"][npz_row]),
                        "owl_box_xyxy_960": owl["box_xyxy_960"].copy(),
                        "prompt_box_xyxy_640x480": _map_owl_box_to_depth(
                            owl["box_xyxy_960"]
                        ),
                        "raw_boxer_center_world": arrays[
                            "per_view_center_world"
                        ][npz_row].astype(np.float32, copy=True),
                        "raw_boxer_quaternion_wxyz": arrays[
                            "per_view_quaternion_wxyz"
                        ][npz_row].astype(np.float32, copy=True),
                        "raw_boxer_extent_xyz": arrays[
                            "per_view_extent_xyz"
                        ][npz_row].astype(np.float32, copy=True),
                    }
                )
        source_summary[scene] = {
            "owl_csv_path": os.fspath(owl_path),
            "owl_csv_sha256": owl_hash,
            "owl_numeric_row_count": len(owl_rows),
            "boxer_csv_path": os.fspath(boxer_path),
            "boxer_csv_sha256": boxer_hash,
            "boxer_numeric_row_count": len(boxer_rows),
            "selected_top4_row_count": len(positions),
            "semantic_columns_decoded": False,
            "semantic_columns_consumed": False,
        }
    expected_requested = sum(
        len(selections[DEV3_SCENES.index(scene)]) for scene in requested_scenes
    )
    if len(records) != expected_requested:
        raise S3aShadowError("paired record count differs from exact Top4 membership")
    return records, source_summary


class MobileSAMBoxPromptEngine:
    """Thin frozen box-only MobileSAM adapter with synchronized timings."""

    def __init__(self, device: str) -> None:
        if not isinstance(device, str) or not device.startswith("cuda"):
            raise S3aShadowError("formal S3a shadow requires a CUDA device")
        try:
            import torch
        except ImportError as error:
            raise S3aShadowError("PyTorch is unavailable") from error
        if not torch.cuda.is_available():
            raise S3aShadowError("CUDA is unavailable for MobileSAM S3a")
        try:
            torch.device(device)
        except (RuntimeError, ValueError) as error:
            raise S3aShadowError(f"invalid CUDA device: {device}") from error

        source_root = os.fspath(MOBILESAM_ROOT.resolve())
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
        try:
            from mobile_sam import SamPredictor, sam_model_registry
        except ImportError as error:
            raise S3aShadowError("could not import frozen MobileSAM source") from error
        imported = Path(sys.modules["mobile_sam"].__file__).resolve()
        try:
            imported.relative_to(MOBILESAM_ROOT.resolve())
        except ValueError as error:
            raise S3aShadowError("MobileSAM imported from an unexpected source root") from error

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.set_device(torch.device(device))
        torch.cuda.reset_peak_memory_stats(torch.device(device))
        started = time.perf_counter()
        model = sam_model_registry["vit_t"](checkpoint=os.fspath(MOBILESAM_CHECKPOINT))
        model.to(device=device)
        model.eval()
        torch.cuda.synchronize(torch.device(device))
        self.cold_start_seconds = time.perf_counter() - started
        self.parameter_count = int(sum(value.numel() for value in model.parameters()))
        if self.parameter_count != 10_130_092:
            raise S3aShadowError("MobileSAM parameter count changed")
        self.torch = torch
        self.device = device
        self.model = model
        self.predictor = SamPredictor(model)

    def predict(
        self, image_rgb: np.ndarray, boxes_xyxy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
        image = np.asarray(image_rgb)
        boxes = np.asarray(boxes_xyxy, dtype=np.float32)
        if image.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3) or image.dtype != np.uint8:
            raise S3aShadowError("MobileSAM image must be uint8 [480,640,3]")
        if (
            boxes.ndim != 2
            or boxes.shape[1:] != (4,)
            or not 1 <= len(boxes) <= TOP_K
            or not np.isfinite(boxes).all()
        ):
            raise S3aShadowError("MobileSAM prompt batch must contain 1..4 finite boxes")
        torch = self.torch
        torch.cuda.synchronize(torch.device(self.device))
        started = time.perf_counter()
        with torch.inference_mode():
            self.predictor.set_image(image)
            torch.cuda.synchronize(torch.device(self.device))
            encoded = time.perf_counter()
            prompt_boxes = torch.as_tensor(
                boxes, dtype=torch.float32, device=self.device
            )
            transformed = self.predictor.transform.apply_boxes_torch(
                prompt_boxes, image.shape[:2]
            )
            masks, predicted_iou, _ = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed,
                multimask_output=True,
            )
            best = torch.argmax(predicted_iou, dim=1)
            indices = torch.arange(len(masks), device=self.device)
            selected_masks = masks[indices, best]
            selected_iou = predicted_iou[indices, best]
            host_masks = selected_masks.detach().cpu().numpy()
            host_iou = selected_iou.detach().cpu().numpy()
            host_best = best.detach().cpu().numpy()
            torch.cuda.synchronize(torch.device(self.device))
            finished = time.perf_counter()
        host_masks = np.asarray(host_masks, dtype=bool)
        if host_masks.shape != (len(boxes), IMAGE_HEIGHT, IMAGE_WIDTH):
            raise S3aShadowError(
                f"MobileSAM mask shape changed: {host_masks.shape!r}"
            )
        host_iou = np.asarray(host_iou, dtype=np.float32).reshape(-1)
        host_best = np.asarray(host_best, dtype=np.int8).reshape(-1)
        if (
            host_iou.shape != (len(boxes),)
            or host_best.shape != (len(boxes),)
            or not np.isfinite(host_iou).all()
            or np.any((host_best < 0) | (host_best >= 3))
        ):
            raise S3aShadowError("MobileSAM hypothesis output is invalid")
        return (
            host_masks,
            host_iou,
            host_best,
            {
                "encoder_ms": (encoded - started) * 1000.0,
                "decoder_and_host_mask_ms": (finished - encoded) * 1000.0,
                "provider_ms": (finished - started) * 1000.0,
            },
        )

    def runtime_metadata(self) -> dict[str, Any]:
        torch = self.torch
        props = torch.cuda.get_device_properties(torch.device(self.device))
        return {
            "device": self.device,
            "gpu_name": props.name,
            "gpu_total_memory_bytes": int(props.total_memory),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cold_start_seconds": self.cold_start_seconds,
            "parameter_count": self.parameter_count,
            "peak_allocated_memory_bytes": int(
                torch.cuda.max_memory_allocated(torch.device(self.device))
            ),
        }


def _decode_frame(scene: str, frame_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import cv2
    except ImportError as error:
        raise S3aShadowError("OpenCV is unavailable") from error
    paths = _frame_paths(scene, frame_id)
    bgr = cv2.imread(os.fspath(paths["rgb"]), cv2.IMREAD_COLOR)
    depth = cv2.imread(os.fspath(paths["depth"]), cv2.IMREAD_UNCHANGED)
    if bgr is None or depth is None:
        raise S3aShadowError(f"could not decode frozen frame {scene}/{frame_id}")
    if depth.shape != (IMAGE_HEIGHT, IMAGE_WIDTH) or depth.dtype != np.uint16:
        raise S3aShadowError(
            f"sensor depth must be uint16 [480,640]: {scene}/{frame_id}"
        )
    bgr = cv2.resize(
        bgr, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR
    )
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    try:
        pose = np.loadtxt(paths["pose"], dtype=np.float64)
    except (OSError, ValueError) as error:
        raise S3aShadowError(f"could not load frozen pose {scene}/{frame_id}") from error
    if (
        pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or not np.allclose(
            pose[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-6, rtol=0.0
        )
    ):
        raise S3aShadowError(f"invalid current pose {scene}/{frame_id}")
    return np.ascontiguousarray(rgb), np.ascontiguousarray(depth), pose


def _pack_mask(mask: np.ndarray) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    if value.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise S3aShadowError("mask pack input must have shape [480,640]")
    packed = np.packbits(value.reshape(-1), bitorder="little")
    if packed.shape != (MASK_PACKED_BYTES,):
        raise S3aShadowError("packed mask size changed")
    return packed


def _empty_row_result() -> dict[str, Any]:
    empty_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=bool)
    return {
        "selected_hypothesis_index": -1,
        "predicted_iou": 0.0,
        "sam_mask_packed": _pack_mask(empty_mask),
        "cleaned_depth_mask_packed": _pack_mask(empty_mask),
        "sam_mask_pixel_count": 0,
        "valid_depth_pixel_count": 0,
        "raw_point_count": 0,
        "unique_voxel_count": 0,
        "retained_point_count": 0,
        "median_depth_m": 0.0,
        "median_depth_valid": False,
        "points_world": np.empty((0, 3), dtype=np.float32),
        "accepted": False,
        "abstention_code": 1,
        "reported_q02_q98_center_world": np.zeros(3, dtype=np.float32),
        "reported_q02_q98_extent_xyz": np.zeros(3, dtype=np.float32),
        "diagnostic_q00_q100_center_world": np.zeros(3, dtype=np.float32),
        "diagnostic_q00_q100_extent_xyz": np.zeros(3, dtype=np.float32),
        "diagnostic_box_valid": False,
        "lifting_ms": 0.0,
    }


def _lift_mask_row(
    *,
    mask: np.ndarray,
    predicted_iou: float,
    hypothesis_index: int,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    pose: np.ndarray,
    object_memory: ModuleType,
) -> dict[str, Any]:
    started = time.perf_counter()
    depth_observation = object_memory.extract_masked_world_points(
        depth=depth,
        mask=mask,
        intrinsics=intrinsic,
        camera_to_world=pose,
        config=OBJECT_MEMORY_CONFIG,
    )
    points = np.asarray(depth_observation.points_world, dtype=np.float32)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise S3aShadowError("fixed mask lifting returned invalid world points")
    mask_pixels = int(np.count_nonzero(mask))
    if mask_pixels != int(depth_observation.mask_pixels):
        raise S3aShadowError("MobileSAM mask pixel accounting mismatch")
    voxel_count = int(depth_observation.voxel_point_count)
    retained = int(depth_observation.retained_point_count)
    if retained != len(points) or retained > 2048 or voxel_count < retained:
        raise S3aShadowError("fixed point-cap accounting mismatch")

    result = _empty_row_result()
    result.update(
        {
            "selected_hypothesis_index": int(hypothesis_index),
            "predicted_iou": float(predicted_iou),
            "sam_mask_packed": _pack_mask(mask),
            "cleaned_depth_mask_packed": _pack_mask(
                depth_observation.valid_pixel_mask
            ),
            "sam_mask_pixel_count": mask_pixels,
            "valid_depth_pixel_count": int(depth_observation.valid_depth_pixels),
            "raw_point_count": int(depth_observation.raw_point_count),
            "unique_voxel_count": voxel_count,
            "retained_point_count": retained,
            "median_depth_m": (
                float(depth_observation.median_depth)
                if depth_observation.median_depth is not None
                else 0.0
            ),
            "median_depth_valid": depth_observation.median_depth is not None,
            "points_world": points,
        }
    )
    if retained > 0:
        minimum = np.min(points, axis=0)
        maximum = np.max(points, axis=0)
        result["diagnostic_q00_q100_center_world"] = (
            (minimum + maximum) * 0.5
        ).astype(np.float32)
        result["diagnostic_q00_q100_extent_xyz"] = (maximum - minimum).astype(
            np.float32
        )
        result["diagnostic_box_valid"] = True

    if mask_pixels == 0:
        code = 2
    elif int(depth_observation.valid_depth_pixels) == 0:
        code = 3
    elif voxel_count < MIN_CLEAN_VOXELS:
        code = 4
    else:
        center, extent = object_memory.robust_quantile_aabb(
            points,
            lower_quantile=0.02,
            upper_quantile=0.98,
            min_points=MIN_CLEAN_VOXELS,
            minimum_dimension=0.02,
        )
        result["reported_q02_q98_center_world"] = np.asarray(
            center, dtype=np.float32
        )
        result["reported_q02_q98_extent_xyz"] = np.asarray(
            extent, dtype=np.float32
        )
        result["accepted"] = True
        code = 0
    result["abstention_code"] = code
    result["lifting_ms"] = (time.perf_counter() - started) * 1000.0
    return result


def _group_record_indices_by_frame(
    records: Sequence[Mapping[str, Any]],
) -> list[tuple[str, int, list[int]]]:
    groups: list[tuple[str, int, list[int]]] = []
    for index, record in enumerate(records):
        key = (str(record["scene"]), int(record["frame_id"]))
        if not groups or groups[-1][:2] != key:
            groups.append((key[0], key[1], [index]))
        else:
            groups[-1][2].append(index)
    return groups


def _process_records(
    *,
    records: Sequence[Mapping[str, Any]],
    requested_scenes: Sequence[str],
    schedules: Mapping[str, Mapping[str, Any]],
    engine: Any,
    object_memory: ModuleType,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    row_results: list[dict[str, Any] | None] = [None] * len(records)
    frame_runtime: list[dict[str, Any]] = []
    groups = _group_record_indices_by_frame(records)
    for group_ordinal, (scene, frame_id, indices) in enumerate(groups):
        if len(indices) > TOP_K:
            raise S3aShadowError("per-frame Top4 cap was exceeded")
        loaded = time.perf_counter()
        rgb, depth, pose = _decode_frame(scene, frame_id)
        decode_ms = (time.perf_counter() - loaded) * 1000.0
        boxes = np.stack(
            [records[index]["prompt_box_xyxy_640x480"] for index in indices], axis=0
        ).astype(np.float32)
        valid_prompt = (
            np.isfinite(boxes).all(axis=1)
            & (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )
        valid_positions = np.flatnonzero(valid_prompt)
        provider = {
            "encoder_ms": 0.0,
            "decoder_and_host_mask_ms": 0.0,
            "provider_ms": 0.0,
        }
        if len(valid_positions):
            masks, predicted_iou, hypotheses, provider = engine.predict(
                rgb, boxes[valid_positions]
            )
            if len(masks) != len(valid_positions):
                raise S3aShadowError("MobileSAM batch cardinality changed")
            for batch_index, local_position in enumerate(valid_positions.tolist()):
                record_index = indices[local_position]
                row_results[record_index] = _lift_mask_row(
                    mask=masks[batch_index],
                    predicted_iou=float(predicted_iou[batch_index]),
                    hypothesis_index=int(hypotheses[batch_index]),
                    depth=depth,
                    intrinsic=np.asarray(schedules[scene]["intrinsic"]),
                    pose=pose,
                    object_memory=object_memory,
                )
        for local_position, record_index in enumerate(indices):
            if valid_prompt[local_position]:
                if row_results[record_index] is None:
                    raise S3aShadowError("valid prompt did not receive a row result")
            else:
                row_results[record_index] = _empty_row_result()
            result = row_results[record_index]
            assert result is not None
            result["decode_ms"] = decode_ms
            result["encoder_ms"] = float(provider["encoder_ms"])
            result["decoder_ms"] = float(provider["decoder_and_host_mask_ms"])
            result["frame_provider_ms"] = float(provider["provider_ms"])
        frame_runtime.append(
            {
                "group_ordinal": group_ordinal,
                "scene_id": scene,
                "frame_id": frame_id,
                "prompt_count": len(valid_positions),
                "selected_row_count": len(indices),
                "decode_ms": decode_ms,
                "encoder_ms": float(provider["encoder_ms"]),
                "decoder_and_host_mask_ms": float(
                    provider["decoder_and_host_mask_ms"]
                ),
                "provider_ms": float(provider["provider_ms"]),
                "lifting_ms": float(
                    sum(
                        row_results[index]["lifting_ms"]  # type: ignore[index]
                        for index in indices
                    )
                ),
            }
        )
    if any(value is None for value in row_results):
        raise S3aShadowError("one or more exact Top4 rows were not exported")
    completed = [value for value in row_results if value is not None]

    point_offsets = [0]
    point_blocks: list[np.ndarray] = []
    for result in completed:
        points = np.asarray(result["points_world"], dtype=np.float32)
        point_blocks.append(points)
        point_offsets.append(point_offsets[-1] + len(points))
    points_world = (
        np.concatenate(point_blocks, axis=0).astype(np.float32, copy=False)
        if point_blocks and point_offsets[-1]
        else np.empty((0, 3), dtype=np.float32)
    )
    local_scene_index = {scene: index for index, scene in enumerate(requested_scenes)}

    def row_array(key: str, dtype: Any) -> np.ndarray:
        return np.asarray([record[key] for record in records], dtype=dtype)

    def result_array(key: str, dtype: Any) -> np.ndarray:
        return np.asarray([result[key] for result in completed], dtype=dtype)

    sam_packed = np.stack(
        [np.asarray(result["sam_mask_packed"], dtype=np.uint8) for result in completed]
    )
    clean_packed = np.stack(
        [
            np.asarray(result["cleaned_depth_mask_packed"], dtype=np.uint8)
            for result in completed
        ]
    )
    point_hashes = np.asarray(
        [_hash_array(np.asarray(result["points_world"], dtype=np.float32)) for result in completed],
        dtype="<U64",
    )
    arrays: dict[str, np.ndarray] = {
        "scene_ids": np.asarray(requested_scenes, dtype="<U12"),
        "scene_index": np.asarray(
            [local_scene_index[str(record["scene"])] for record in records],
            dtype=np.int16,
        ),
        "schedule_ordinal": row_array("schedule_ordinal", np.int32),
        "manifest_schedule_ordinal": row_array(
            "manifest_schedule_ordinal", np.int32
        ),
        "frame_id": row_array("frame_id", np.int64),
        "sealed_npz_row": row_array("sealed_npz_row", np.int64),
        "boxer_source_row": row_array("boxer_source_row", np.int32),
        "boxer_csv_line_number": row_array("boxer_csv_line_number", np.int32),
        "source_instance_id": row_array("source_instance_id", np.int32),
        "owl_csv_source_row": row_array("owl_csv_source_row", np.int32),
        "owl_csv_line_number": row_array("owl_csv_line_number", np.int32),
        "source_score": row_array("source_score", np.float32),
        "owl_box_xyxy_960": row_array("owl_box_xyxy_960", np.float32),
        "prompt_box_xyxy_640x480": row_array(
            "prompt_box_xyxy_640x480", np.float32
        ),
        "raw_boxer_center_world": row_array("raw_boxer_center_world", np.float32),
        "raw_boxer_quaternion_wxyz": row_array(
            "raw_boxer_quaternion_wxyz", np.float32
        ),
        "raw_boxer_extent_xyz": row_array("raw_boxer_extent_xyz", np.float32),
        "selected_hypothesis_index": result_array(
            "selected_hypothesis_index", np.int8
        ),
        "predicted_iou": result_array("predicted_iou", np.float32),
        "sam_mask_packed": sam_packed,
        "cleaned_depth_mask_packed": clean_packed,
        "sam_mask_sha256": np.asarray(
            [_hash_array(value) for value in sam_packed], dtype="<U64"
        ),
        "cleaned_depth_mask_sha256": np.asarray(
            [_hash_array(value) for value in clean_packed], dtype="<U64"
        ),
        "sam_mask_pixel_count": result_array("sam_mask_pixel_count", np.int32),
        "valid_depth_pixel_count": result_array(
            "valid_depth_pixel_count", np.int32
        ),
        "raw_point_count": result_array("raw_point_count", np.int32),
        "unique_voxel_count": result_array("unique_voxel_count", np.int32),
        "retained_point_count": result_array("retained_point_count", np.int32),
        "median_depth_m": result_array("median_depth_m", np.float32),
        "median_depth_valid": result_array("median_depth_valid", bool),
        "point_offsets": np.asarray(point_offsets, dtype=np.int64),
        "points_world": points_world,
        "points_sha256": point_hashes,
        "accepted": result_array("accepted", bool),
        "abstention_code": result_array("abstention_code", np.int8),
        "reported_q02_q98_center_world": result_array(
            "reported_q02_q98_center_world", np.float32
        ),
        "reported_q02_q98_extent_xyz": result_array(
            "reported_q02_q98_extent_xyz", np.float32
        ),
        "diagnostic_q00_q100_center_world": result_array(
            "diagnostic_q00_q100_center_world", np.float32
        ),
        "diagnostic_q00_q100_extent_xyz": result_array(
            "diagnostic_q00_q100_extent_xyz", np.float32
        ),
        "diagnostic_box_valid": result_array("diagnostic_box_valid", bool),
        "decode_ms": result_array("decode_ms", np.float32),
        "encoder_ms": result_array("encoder_ms", np.float32),
        "decoder_ms": result_array("decoder_ms", np.float32),
        "frame_provider_ms": result_array("frame_provider_ms", np.float32),
        "lifting_ms": result_array("lifting_ms", np.float32),
    }
    if sam_packed.shape != (len(records), MASK_PACKED_BYTES):
        raise S3aShadowError("sealed MobileSAM mask matrix has wrong shape")
    if clean_packed.shape != (len(records), MASK_PACKED_BYTES):
        raise S3aShadowError("sealed cleaned-depth mask matrix has wrong shape")
    if not np.array_equal(
        np.diff(arrays["point_offsets"]), arrays["retained_point_count"]
    ):
        raise S3aShadowError("point offsets do not match retained row counts")
    if int(arrays["point_offsets"][-1]) != len(points_world):
        raise S3aShadowError("concatenated point count mismatch")
    if not set(np.unique(arrays["abstention_code"]).tolist()).issubset(
        ABSTENTION_CODES
    ):
        raise S3aShadowError("unknown abstention code was generated")
    if not np.array_equal(arrays["accepted"], arrays["abstention_code"] == 0):
        raise S3aShadowError("accepted rows and abstention codes disagree")
    accepted_count = int(np.count_nonzero(arrays["accepted"]))
    per_code = {
        str(code): int(np.count_nonzero(arrays["abstention_code"] == code))
        for code in ABSTENTION_CODES
    }
    runtime_values = np.asarray(
        [row["provider_ms"] for row in frame_runtime], dtype=np.float64
    )
    summary = {
        "row_count": len(records),
        "accepted_row_count": accepted_count,
        "abstained_row_count": len(records) - accepted_count,
        "abstention_count_by_code": per_code,
        "frame_count": len(groups),
        "retained_point_count": int(len(points_world)),
        "frame_runtime": frame_runtime,
        "runtime_aggregate": {
            "provider_mean_ms": (
                float(np.mean(runtime_values)) if len(runtime_values) else 0.0
            ),
            "provider_p50_ms": (
                float(np.quantile(runtime_values, 0.50)) if len(runtime_values) else 0.0
            ),
            "provider_p95_ms": (
                float(np.quantile(runtime_values, 0.95)) if len(runtime_values) else 0.0
            ),
            "provider_max_ms": (
                float(np.max(runtime_values)) if len(runtime_values) else 0.0
            ),
            "provider_sum_seconds": float(np.sum(runtime_values) / 1000.0),
            "lifting_sum_seconds": float(
                np.sum(arrays["lifting_ms"], dtype=np.float64) / 1000.0
            ),
        },
    }
    return arrays, summary


def _input_snapshot(
    *,
    fixed_assets: Mapping[str, Mapping[str, object]],
    schedules: Mapping[str, Mapping[str, Any]],
    source_summary: Mapping[str, Mapping[str, Any]],
    native: Mapping[str, Mapping[str, str]],
    runner_source_sha256: str,
) -> dict[str, Any]:
    return {
        "fixed_assets": fixed_assets,
        "schedule_raw_inputs": _schedule_raw_input_ledger(schedules),
        "numeric_source_csv": source_summary,
        "native_t05": native,
        "runner_source": {
            "path": os.fspath(Path(__file__).resolve()),
            "sha256": runner_source_sha256,
        },
    }


def _scene_summaries(
    arrays: Mapping[str, np.ndarray],
    requested_scenes: Sequence[str],
    source_summary: Mapping[str, Mapping[str, Any]],
    schedules: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for scene_index, scene in enumerate(requested_scenes):
        positions = np.flatnonzero(arrays["scene_index"] == scene_index)
        codes = arrays["abstention_code"][positions]
        summaries[scene] = {
            "row_count": int(len(positions)),
            "frame_count": int(len(np.unique(arrays["frame_id"][positions]))),
            "accepted_row_count": int(np.count_nonzero(arrays["accepted"][positions])),
            "abstained_row_count": int(
                len(positions) - np.count_nonzero(arrays["accepted"][positions])
            ),
            "abstention_count_by_code": {
                str(code): int(np.count_nonzero(codes == code))
                for code in ABSTENTION_CODES
            },
            "retained_point_count": int(
                np.sum(arrays["retained_point_count"][positions], dtype=np.int64)
            ),
            "source": dict(source_summary[scene]),
            "schedule": {
                "manifest_path": os.fspath(schedules[scene]["manifest_path"]),
                "manifest_sha256": schedules[scene]["manifest_sha256"],
                "recorded_frame_count": len(schedules[scene]["recorded_frame_ids"]),
                "valid_frame_count": len(schedules[scene]["valid_frame_ids"]),
                "invalid_pose_frame_ids": list(
                    schedules[scene]["invalid_pose_frame_ids"]
                ),
            },
        }
    return summaries


def _native_hash_mapping(
    native: Mapping[str, Mapping[str, str]],
) -> dict[str, str]:
    return {scene: native[scene]["sha256"] for scene in DEV3_SCENES}


def _build_manifest(
    *,
    arrays: Mapping[str, np.ndarray],
    summary: Mapping[str, Any],
    requested_scenes: Sequence[str],
    selection_sha256: str,
    source_summary: Mapping[str, Mapping[str, Any]],
    schedules: Mapping[str, Mapping[str, Any]],
    fixed_assets: Mapping[str, Mapping[str, object]],
    native_before: Mapping[str, Mapping[str, str]],
    native_after: Mapping[str, Mapping[str, str]],
    input_before: Mapping[str, Any],
    input_after: Mapping[str, Any],
    runtime_metadata: Mapping[str, Any],
    engineering_smoke: bool,
    npz_sha256: str,
) -> dict[str, Any]:
    if input_before != input_after:
        raise S3aShadowError("one or more frozen inputs changed during S3a inference")
    native_hash_before = _native_hash_mapping(native_before)
    native_hash_after = _native_hash_mapping(native_after)
    if native_hash_before != native_hash_after:
        raise S3aShadowError("native T05 bytes changed during S3a inference")
    content_sha256 = _array_content_sha256(arrays)
    baseline_roots = {
        os.fspath(Path(native_before[scene]["path"]).parent)
        for scene in DEV3_SCENES
    }
    if len(baseline_roots) != 1:
        raise S3aShadowError("native T05 predictions do not share one baseline root")
    baseline_root = next(iter(baseline_roots))
    if Path(baseline_root).resolve() != FORMAL_T05_ROOT.resolve():
        raise S3aShadowError("native predictions are not rooted at formal T05")
    return {
        "schema": SCHEMA,
        "mode": "shadow",
        "output_inert": True,
        "per_view_only": True,
        "birth": False,
        "active_authorized": False,
        "gt_access": False,
        "oracle_access": False,
        "labels_loaded": False,
        "labels_exported": False,
        "semantic_columns_decoded": False,
        "semantic_columns_consumed": False,
        "clip_access": False,
        "tracking": False,
        "suppression": False,
        "terminal_fusion": False,
        "native_overlap_rejection": False,
        "unexplained_depth_gate": False,
        "future_frame_access": False,
        "native_mutation_applied": False,
        "native_identity_source": "formal_t05_direct_file_hashes",
        "sealed_boxer_native_identity_ledger_trusted": False,
        "training": False,
        "optimizer": False,
        "threshold_tuning_performed": False,
        "posthoc_selection_performed": False,
        "H10_not_authorized": True,
        "full100_not_authorized": True,
        "not_deployable": True,
        "engineering_smoke": engineering_smoke,
        "dev3_complete": not engineering_smoke,
        "dev3_scene_order": list(DEV3_SCENES),
        "scene_order": list(requested_scenes),
        "scene_count": len(requested_scenes),
        "row_count": int(summary["row_count"]),
        "accepted_row_count": int(summary["accepted_row_count"]),
        "abstained_row_count": int(summary["abstained_row_count"]),
        "abstention_codes": {str(key): value for key, value in ABSTENTION_CODES.items()},
        "abstention_count_by_code": summary["abstention_count_by_code"],
        "complete_exact_top4_membership_for_requested_scenes": True,
        "complete_exact_top4_membership_for_dev3": not engineering_smoke,
        "preregistration_sha256": EXPECTED_HASHES["preregistration"][1],
        "selection": {
            "top_k_per_frame": TOP_K,
            "selection_sha256": selection_sha256,
            "selection_rule": "descending_source_score_then_ascending_source_row",
            "full_tie_break": (
                "descending_source_score_then_ascending_source_row_then_"
                "ascending_sealed_npz_row"
            ),
            "selection_used_semantics": False,
            "selection_used_only_frozen_source_score": True,
            "selected_membership_npz_rows_sha256": _hash_array(
                arrays["sealed_npz_row"]
            ),
        },
        "input": {
            "sealed_boxer_json_sha256": EXPECTED_HASHES["sealed_json"][1],
            "sealed_boxer_npz_sha256": EXPECTED_HASHES["sealed_npz"][1],
            "topk_receipt_sha256": EXPECTED_HASHES["topk_receipt"][1],
            "runtime_receipt_sha256": EXPECTED_HASHES["runtime_receipt"][1],
            "baseline_root": baseline_root,
            "formal_t05_root": os.fspath(FORMAL_T05_ROOT.resolve()),
            "formal_t05_expected_sha256": dict(EXPECTED_FORMAL_T05_SHA256),
        },
        "fixed_assets": fixed_assets,
        "native_prediction_sha256_before": native_hash_before,
        "native_prediction_sha256_after": native_hash_after,
        "native_prediction_hash_identity": native_hash_before == native_hash_after,
        "input_sha256_before": input_before,
        "input_sha256_after": input_after,
        "input_hash_identity": input_before == input_after,
        "geometry_profile": {
            "image_height": IMAGE_HEIGHT,
            "image_width": IMAGE_WIDTH,
            "owl_box_mapping": "x*2/3,y*1/2_then_clip",
            "prompt": "box_only",
            "shared_image_embedding_once_per_frame": True,
            "multimask_output": True,
            "mask_choice": "maximum_frozen_predicted_iou_lowest_index_tie",
            "mask_probability_threshold_equivalent": 0.50,
            "min_depth_m": 0.10,
            "max_depth_m": 6.00,
            "mask_edge_margin_pixels": 1,
            "depth_discontinuity_m": 0.15,
            "voxel_size_m": 0.02,
            "voxel_indexing": "signed_floor",
            "minimum_unique_clean_voxels": MIN_CLEAN_VOXELS,
            "max_points_per_observation": 2048,
            "primary_aabb_quantiles": [0.02, 0.98],
            "primary_minimum_dimension_m": 0.02,
            "diagnostic_aabb_quantiles": [0.0, 1.0],
            "coordinate_frame": "scannet_world",
        },
        "mask_encoding": {
            "shape_hw": [IMAGE_HEIGHT, IMAGE_WIDTH],
            "packed_axis": "flattened_row_major",
            "packbits_bitorder": "little",
            "packed_bytes_per_row": MASK_PACKED_BYTES,
            "sam_mask_array": "sam_mask_packed",
            "cleaned_depth_mask_array": "cleaned_depth_mask_packed",
        },
        "point_encoding": {
            "points_array": "points_world",
            "offsets_array": "point_offsets",
            "row_count_array": "retained_point_count",
            "deterministic_cap": 2048,
        },
        "runtime": {
            **dict(runtime_metadata),
            **dict(summary["runtime_aggregate"]),
            "frame_measurements": summary["frame_runtime"],
            "authoritative_isolated_receipt": os.fspath(RUNTIME_RECEIPT.resolve()),
            "authoritative_isolated_receipt_sha256": EXPECTED_HASHES[
                "runtime_receipt"
            ][1],
            "same_gpu_copipeline_qualification": False,
        },
        "scenes": _scene_summaries(
            arrays, requested_scenes, source_summary, schedules
        ),
        "npz_file": OUTPUT_NPZ_NAME,
        "npz_sha256": npz_sha256,
        "candidate_content_sha256": content_sha256,
        "conclusion_guardrail": (
            "No accuracy claim is made by this inference sidecar. A separate "
            "read-only dev3 oracle must test the fixed q02/q98 and prespecified "
            "q00/q100 geometries before S3b tracking is considered."
        ),
    }


def _check_output_location(output_root: Path) -> Path:
    raw = output_root
    output = raw.resolve()
    if raw.is_symlink() or output.exists():
        raise S3aShadowError(f"refusing to overwrite output root: {output}")
    protected_roots = (
        SEALED_ROOT.resolve(),
        SCHEDULE_ROOT.resolve(),
        SCENE_ROOT.resolve(),
        FORMAL_T05_ROOT.resolve(),
    )
    for protected in protected_roots:
        try:
            output.relative_to(protected)
        except ValueError:
            continue
        raise S3aShadowError(f"output root is inside protected input: {protected}")
    return output


def _publish_create_only(
    *, output_root: Path, arrays: Mapping[str, np.ndarray], manifest: Mapping[str, Any]
) -> None:
    output = output_root.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.stage.", dir=output.parent)
    )
    claimed = False
    try:
        _write_deterministic_npz(staging / OUTPUT_NPZ_NAME, arrays)
        _write_json_exclusive(staging / OUTPUT_JSON_NAME, manifest)
        try:
            output.mkdir()
            claimed = True
        except FileExistsError as error:
            raise S3aShadowError(f"refusing to overwrite output root: {output}") from error
        os.link(staging / OUTPUT_NPZ_NAME, output / OUTPUT_NPZ_NAME)
        os.link(staging / OUTPUT_JSON_NAME, output / OUTPUT_JSON_NAME)
        descriptor = os.open(output, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except Exception:
        if claimed:
            shutil.rmtree(output, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def run_s3a_shadow(
    *,
    output_root: Path,
    device: str,
    engineering_smoke_scene: str | None = None,
    engine_factory: Any = MobileSAMBoxPromptEngine,
) -> dict[str, Any]:
    output = _check_output_location(output_root)
    if engineering_smoke_scene is not None and engineering_smoke_scene not in DEV3_SCENES:
        raise S3aShadowError("engineering smoke scene is outside frozen dev3")
    requested_scenes = (
        (engineering_smoke_scene,)
        if engineering_smoke_scene is not None
        else DEV3_SCENES
    )
    runner_source = _regular_file(Path(__file__), "S3a runner source")
    runner_hash_before = _sha256(runner_source)
    fixed_before = _validate_fixed_assets()
    manifest, sealed_arrays = _load_sealed_candidates()
    selections, selection_hash = _select_top4(sealed_arrays)
    schedules_before = _load_schedule_contract(manifest)
    native_before = _hash_formal_t05_predictions()
    records, source_before = _validate_pair_and_build_records(
        manifest=manifest,
        arrays=sealed_arrays,
        selections=selections,
        schedules=schedules_before,
        requested_scenes=requested_scenes,
    )
    input_before = _input_snapshot(
        fixed_assets=fixed_before,
        schedules=schedules_before,
        source_summary=source_before,
        native=native_before,
        runner_source_sha256=runner_hash_before,
    )

    object_memory = _load_object_memory_module()
    engine = engine_factory(device)
    arrays, summary = _process_records(
        records=records,
        requested_scenes=requested_scenes,
        schedules=schedules_before,
        engine=engine,
        object_memory=object_memory,
    )
    runtime_metadata = engine.runtime_metadata()

    fixed_after = _validate_fixed_assets()
    schedules_after = _load_schedule_contract(manifest)
    native_after = _hash_formal_t05_predictions()
    _records_after, source_after = _validate_pair_and_build_records(
        manifest=manifest,
        arrays=sealed_arrays,
        selections=selections,
        schedules=schedules_after,
        requested_scenes=requested_scenes,
    )
    runner_hash_after = _sha256(runner_source)
    input_after = _input_snapshot(
        fixed_assets=fixed_after,
        schedules=schedules_after,
        source_summary=source_after,
        native=native_after,
        runner_source_sha256=runner_hash_after,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.npz.", dir=output.parent
    )
    os.close(descriptor)
    temporary_npz = Path(temporary_name)
    temporary_npz.unlink()
    try:
        _write_deterministic_npz(temporary_npz, arrays)
        npz_sha256 = _sha256(temporary_npz)
    finally:
        temporary_npz.unlink(missing_ok=True)
    output_manifest = _build_manifest(
        arrays=arrays,
        summary=summary,
        requested_scenes=requested_scenes,
        selection_sha256=selection_hash,
        source_summary=source_before,
        schedules=schedules_before,
        fixed_assets=fixed_before,
        native_before=native_before,
        native_after=native_after,
        input_before=input_before,
        input_after=input_after,
        runtime_metadata=runtime_metadata,
        engineering_smoke=engineering_smoke_scene is not None,
        npz_sha256=npz_sha256,
    )
    _publish_create_only(output_root=output, arrays=arrays, manifest=output_manifest)
    if _sha256(output / OUTPUT_NPZ_NAME) != npz_sha256:
        raise S3aShadowError("published S3a NPZ hash differs from staged content")
    return output_manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--engineering-smoke-scene",
        choices=DEV3_SCENES,
        default=None,
        help="process one frozen dev3 scene while still validating full Top4 membership",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_s3a_shadow(
        output_root=args.output_root,
        device=args.device,
        engineering_smoke_scene=args.engineering_smoke_scene,
    )
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "scene_order": result["scene_order"],
                "row_count": result["row_count"],
                "accepted_row_count": result["accepted_row_count"],
                "output_root": os.fspath(args.output_root.resolve()),
                "gt_access": False,
                "birth": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
