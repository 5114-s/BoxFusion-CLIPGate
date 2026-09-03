#!/usr/bin/env python3
"""Validate two F4 shadow shards and publish one create-only paper100 seal."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "F4-FASTSAM-BOXER-GEOMETRY-SHADOW-PAPER100"
SCENE_SCHEMA = "boxfusion.scannet_fastsam_f4_boxer_paper100.scene.v1"
SHARD_SCHEMA = "boxfusion.scannet_fastsam_f4_boxer_paper100.shard.v1"
MERGE_SCHEMA = "boxfusion.scannet_fastsam_f4_boxer_paper100.merge.v1"
EXPECTED_F2_SCENE_SCHEMA = "boxfusion.scannet_fastsam_f2_paper100.scene.v1"
EXPECTED_SCENES = 100
EXPECTED_SHARDS = 2
EXPECTED_KEYFRAMES = 6_817
EXPECTED_SUCCESSFUL_FRAMES = 6_726
EXPECTED_SOURCES = 52_299
EXPECTED_SHARD_COUNTS: Mapping[int, Mapping[str, int]] = {
    0: {"keyframe_count": 3_259, "successful_frame_count": 3_189, "source_count": 24_863},
    1: {"keyframe_count": 3_558, "successful_frame_count": 3_537, "source_count": 27_436},
}
WARMUP_FORWARD_COUNT = 3
MAX_F4_INCREMENTAL_P95_MS = 100.0
MAX_COMPOSED_P95_MS = 350.0
MAX_COMPOSED_MS_EXCLUSIVE = 833.33
MAX_COMPOSED_MEAN_PER_SOURCE_FRAME_MS = 14.0
SOURCE_FRAME_STRIDE = 25.0
MAX_CUDA_PEAK_BYTES = 4 * 1024**3
OUTPUT_NAME = "F4_FASTSAM_BOXER_PAPER100.json"

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

DEFAULT_SHARDS = (
    REPOSITORY_ROOT / "logs/scannet_fastsam_f4_boxer_paper100_score05/shards/shard-000-of-002.json",
    REPOSITORY_ROOT / "logs/scannet_fastsam_f4_boxer_paper100_score05/shards/shard-001-of-002.json",
)
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "logs/scannet_fastsam_f4_boxer_paper100_score05/final"


class F4MergeError(RuntimeError):
    """Raised when an F4 shard, source lineage, or runtime seal differs."""


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
        raise F4MergeError(f"{label} must be a regular non-symlink file: {path}")
    result = path.resolve()
    if suffix is not None and result.suffix.lower() != suffix:
        raise F4MergeError(f"{label} must be a {suffix} file: {result}")
    if result.suffix.lower() in {".pkl", ".pickle"}:
        raise F4MergeError(f"prediction pickle input is forbidden: {result}")
    return result


def _read_json(path: Path, label: str) -> tuple[Path, dict[str, Any]]:
    source = _regular_file(path, label, ".json")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise F4MergeError(f"invalid {label}: {source}") from error
    if not isinstance(value, dict):
        raise F4MergeError(f"{label} must contain one JSON object")
    return source, value


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise F4MergeError(f"refusing to overwrite output: {path}") from error
        return _sha256(path)
    finally:
        temporary.unlink(missing_ok=True)


def _rehash_reference(value: object, label: str, suffix: str | None = None) -> Path:
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("path"), str)
        or not isinstance(value.get("sha256"), str)
    ):
        raise F4MergeError(f"{label} seal is absent")
    path = _regular_file(Path(value["path"]), label, suffix)
    if _sha256(path) != value["sha256"]:
        raise F4MergeError(f"{label} rehash differs")
    return path


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise F4MergeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise F4MergeError(f"{label} must be finite and non-negative")
    return result


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise F4MergeError("runtime samples must be finite and non-negative")
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _gate(actual: float | int, comparator: str, threshold: float | int) -> dict[str, Any]:
    if comparator == "<=":
        passed = actual <= threshold
    elif comparator == "<":
        passed = actual < threshold
    elif comparator == "==":
        passed = actual == threshold
    else:  # pragma: no cover - internal invariant
        raise AssertionError(comparator)
    return {
        "actual": actual,
        "comparator": comparator,
        "threshold": threshold,
        "pass": bool(passed),
        "passed": bool(passed),
    }


def _content_hash_without(value: Mapping[str, Any], *keys: str) -> str:
    payload = dict(value)
    for key in keys:
        payload.pop(key, None)
    return _canonical_json_sha256(payload)


def _validate_hb(value: object, source_id: str, tight_box: Sequence[float], row_index: int) -> None:
    if not isinstance(value, Mapping):
        raise F4MergeError(f"{source_id} HB is absent")
    if (
        value.get("source_id") != source_id
        or value.get("row_index") != row_index
        or value.get("input_tight_box_xyxy") != list(tight_box)
        or not isinstance(value.get("result_sha256"), str)
    ):
        raise F4MergeError(f"{source_id} HB identity differs")
    if _content_hash_without(value, "result_sha256") != value["result_sha256"]:
        raise F4MergeError(f"{source_id} HB result hash differs")
    provider_result = value.get("provider_result_sha256")
    if provider_result is not None and not _valid_sha256(provider_result):
        raise F4MergeError(f"{source_id} provider HB result hash differs")
    if value.get("valid") is not True:
        if not isinstance(value.get("abstention_reason"), str):
            raise F4MergeError(f"{source_id} invalid HB lacks abstention reason")
        for key in ("world_corners", "world_center", "local_extent", "world_rotation", "camera_depth"):
            if value.get(key) is not None:
                raise F4MergeError(f"{source_id} invalid HB retains usable geometry")
        return
    try:
        corners = np.asarray(value.get("world_corners"), dtype=np.float64)
        center = np.asarray(value.get("world_center"), dtype=np.float64)
        extent = np.asarray(value.get("local_extent"), dtype=np.float64)
        rotation = np.asarray(value.get("world_rotation"), dtype=np.float64)
        camera_depth = float(value.get("camera_depth"))
    except (TypeError, ValueError) as error:
        raise F4MergeError(f"{source_id} valid HB is malformed") from error
    if (
        corners.shape != (8, 3)
        or center.shape != (3,)
        or extent.shape != (3,)
        or rotation.shape != (3, 3)
        or not np.isfinite(corners).all()
        or not np.isfinite(center).all()
        or not np.isfinite(extent).all()
        or not np.isfinite(rotation).all()
        or np.any(extent <= 0.0)
        or not math.isfinite(camera_depth)
        or camera_depth <= 1.0e-4
        or np.linalg.det(rotation) <= 0.0
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-3, rtol=0.0)
    ):
        raise F4MergeError(f"{source_id} valid HB violates frozen geometry rules")


def _validate_source(
    source: Mapping[str, Any],
    f2_source: Mapping[str, Any],
    *,
    scene_id: str,
    scene_index: int,
    frame_id: int,
    frame_ordinal: int,
    row_index: int,
) -> tuple[str, str]:
    identity_keys = (
        "scene_index", "frame_ordinal", "frame_id", "rank", "raw_index",
        "mask_sha256", "points_and_voxel_keys_sha256", "source_id",
    )
    identity = {key: source.get(key) for key in identity_keys}
    expected_id = f"{scene_id}/frame_{frame_id:06d}/raw_{int(f2_source.get('raw_index', -1)):03d}"
    if (
        identity["scene_index"] != scene_index
        or identity["frame_ordinal"] != frame_ordinal
        or identity["frame_id"] != frame_id
        or identity["rank"] != row_index
        or source.get("candidate_index") != row_index
        or identity["raw_index"] != f2_source.get("raw_index")
        or identity["mask_sha256"] != f2_source.get("mask_sha256")
        or identity["points_and_voxel_keys_sha256"] != f2_source.get("points_and_voxel_keys_sha256")
        or identity["source_id"] != expected_id
        or f2_source.get("source_id") != expected_id
    ):
        raise F4MergeError(f"{scene_id}/{frame_id} source identity/order differs")
    hypotheses = source.get("hypotheses")
    if not isinstance(hypotheses, Mapping) or set(hypotheses) != {"H0", "HL", "HLG", "HB"}:
        raise F4MergeError(f"{expected_id} geometry hypotheses differ")
    sealed_hypotheses = {key: hypotheses[key] for key in ("H0", "HL", "HLG")}
    if sealed_hypotheses != f2_source.get("hypotheses"):
        raise F4MergeError(f"{expected_id} H0/HL/HLG are not exact F2 copies")
    if _canonical_json_sha256(sealed_hypotheses) != source.get("sealed_f2_hypotheses_sha256"):
        raise F4MergeError(f"{expected_id} sealed F2 hypothesis hash differs")
    tight_box = source.get("tight_box_xyxy")
    if not isinstance(tight_box, list) or len(tight_box) != 4:
        raise F4MergeError(f"{expected_id} tight box is absent")
    _validate_hb(hypotheses["HB"], expected_id, tight_box, row_index)

    f0_lineage = source.get("f0_source_lineage")
    f2_lineage = source.get("f2_source_lineage")
    if not isinstance(f0_lineage, Mapping) or not isinstance(f2_lineage, Mapping):
        raise F4MergeError(f"{expected_id} source lineage is absent")
    if f2_lineage.get("source_sha256") != _canonical_json_sha256(f2_source):
        raise F4MergeError(f"{expected_id} F2 source lineage differs")
    join_payload = {
        "identity": identity,
        "f0": dict(f0_lineage),
        "f2": dict(f2_lineage),
        "tight_box_xyxy": tight_box,
    }
    if _canonical_json_sha256(join_payload) != source.get("join_sha256"):
        raise F4MergeError(f"{expected_id} source join hash differs")
    source_lineage_payload = {
        "identity": identity,
        "join_sha256": source["join_sha256"],
        "sealed_f2_hypotheses_sha256": source["sealed_f2_hypotheses_sha256"],
        "hb_result_sha256": hypotheses["HB"]["result_sha256"],
    }
    if _canonical_json_sha256(source_lineage_payload) != source.get("source_lineage_sha256"):
        raise F4MergeError(f"{expected_id} source-lineage hash differs")
    return expected_id, str(source["source_lineage_sha256"])


def _validate_scene(
    scene_row: Mapping[str, Any],
    *,
    expected_scene_index: int,
    expected_call_index: list[int],
    expected_run_signature: str,
) -> tuple[dict[str, int], list[str], list[str], list[float], list[float], list[float], int, int, int]:
    scene_id = scene_row.get("scene_id")
    if not isinstance(scene_id, str) or scene_row.get("scene_index") != expected_scene_index:
        raise F4MergeError("manifest scene identity differs")
    sidecar_path = _rehash_reference(scene_row.get("sidecar"), f"{scene_id} F4 sidecar", ".json")
    _, scene = _read_json(sidecar_path, f"{scene_id} F4 sidecar")
    if (
        scene.get("schema") != SCENE_SCHEMA
        or scene.get("protocol_id") != PROTOCOL_ID
        or scene.get("complete") is not True
        or scene.get("scene_id") != scene_id
        or scene.get("scene_index") != expected_scene_index
        or scene.get("run_signature_sha256") != expected_run_signature
        or scene.get("contracts") != CONTRACTS
        or scene.get("native_output_mutation_count") != 0
    ):
        raise F4MergeError(f"{scene_id} F4 sidecar contract differs")
    if _content_hash_without(scene, "content_sha256") != scene.get("content_sha256"):
        raise F4MergeError(f"{scene_id} F4 content hash differs")
    inputs = scene.get("inputs")
    if not isinstance(inputs, Mapping):
        raise F4MergeError(f"{scene_id} frozen inputs are absent")
    f2_path = _rehash_reference(inputs.get("f2_sidecar"), f"{scene_id} F2 sidecar", ".json")
    _, f2_scene = _read_json(f2_path, f"{scene_id} F2 sidecar")
    if f2_scene.get("schema") != EXPECTED_F2_SCENE_SCHEMA:
        raise F4MergeError(f"{scene_id} upstream F2 schema differs")
    for key, suffix in (("f0_sidecar", ".json"), ("f2_evidence", ".npz"), ("schedule", ".json"), ("intrinsic", ".txt")):
        _rehash_reference(inputs.get(key), f"{scene_id} {key}", suffix)
    before_seals = [dict(inputs[key]) for key in ("f2_sidecar", "f0_sidecar", "f2_evidence", "schedule", "intrinsic")]
    for seal, kind in zip(before_seals, ("f2_sidecar", "f0_sidecar", "f2_evidence", "schedule", "intrinsic"), strict=True):
        seal["kind"] = kind
    frames = scene.get("frames")
    f2_frames = f2_scene.get("frames")
    if not isinstance(frames, list) or not isinstance(f2_frames, list) or len(frames) != len(f2_frames):
        raise F4MergeError(f"{scene_id} frame ledger differs")

    source_ids: list[str] = []
    lineage_hashes: list[str] = []
    incremental_warm: list[float] = []
    composed_warm: list[float] = []
    composed_all: list[float] = []
    successful = 0
    provider_forwards = 0
    valid_hb = 0
    deadline_misses_all = 0
    deadline_misses_warm = 0
    for frame_ordinal, (frame, f2_frame) in enumerate(zip(frames, f2_frames, strict=True)):
        if not isinstance(frame, Mapping) or not isinstance(f2_frame, Mapping):
            raise F4MergeError(f"{scene_id} frame row is malformed")
        frame_id = f2_frame.get("frame_id")
        if (
            frame.get("frame_ordinal") != frame_ordinal
            or frame.get("frame_id") != frame_id
            or frame.get("successful") is not f2_frame.get("successful")
            or frame.get("current_only") is not True
        ):
            raise F4MergeError(f"{scene_id} current-frame ledger differs at {frame_ordinal}")
        sources = frame.get("sources")
        f2_sources = f2_frame.get("sources")
        if not isinstance(sources, list) or not isinstance(f2_sources, list):
            raise F4MergeError(f"{scene_id}/{frame_id} source ledger is absent")
        if frame.get("successful") is not True:
            if sources or f2_sources or frame.get("provider_invoked") is not False or frame.get("runtime") is not None:
                raise F4MergeError(f"{scene_id}/{frame_id} abstained-frame contract differs")
            continue
        successful += 1
        if frame.get("max_accessed_frame_ordinal") != frame_ordinal:
            raise F4MergeError(f"{scene_id}/{frame_id} future-frame access detected")
        frame_input = frame.get("input")
        if not isinstance(frame_input, Mapping) or frame_input.get("box_source") != "F0_candidate.tight_box_xyxy":
            raise F4MergeError(f"{scene_id}/{frame_id} Boxer input lineage differs")
        for key, kind in (("rgb", "rgb"), ("depth", "depth"), ("pose", "pose")):
            path = _rehash_reference(frame_input.get(key), f"{scene_id}/{frame_id} {key}")
            before_seals.append({"kind": kind, "frame_ordinal": frame_ordinal, "frame_id": frame_id, "path": os.fspath(path), "sha256": frame_input[key]["sha256"]})
        if len(sources) != len(f2_sources):
            raise F4MergeError(f"{scene_id}/{frame_id} F4/F2 source count differs")
        invoked = bool(sources)
        if frame.get("provider_invoked") is not invoked:
            raise F4MergeError(f"{scene_id}/{frame_id} 0-box invocation rule differs")
        for row_index, (source, f2_source) in enumerate(zip(sources, f2_sources, strict=True)):
            if not isinstance(source, Mapping) or not isinstance(f2_source, Mapping):
                raise F4MergeError(f"{scene_id}/{frame_id} source row is malformed")
            source_id, lineage = _validate_source(
                source,
                f2_source,
                scene_id=scene_id,
                scene_index=expected_scene_index,
                frame_id=int(frame_id),
                frame_ordinal=frame_ordinal,
                row_index=row_index,
            )
            source_ids.append(source_id)
            lineage_hashes.append(lineage)
            valid_hb += int(source["hypotheses"]["HB"]["valid"])
        runtime = frame.get("runtime")
        if invoked:
            if not isinstance(runtime, Mapping) or runtime.get("provider_call_index_in_shard") != expected_call_index[0]:
                raise F4MergeError(f"{scene_id}/{frame_id} provider call ledger differs")
            diagnostics = runtime.get("provider_diagnostics")
            frame_valid_hb = sum(
                int(source["hypotheses"]["HB"]["valid"])
                for source in sources
                if isinstance(source, Mapping)
                and isinstance(source.get("hypotheses"), Mapping)
                and isinstance(source["hypotheses"].get("HB"), Mapping)
            )
            if (
                not isinstance(diagnostics, Mapping)
                or diagnostics.get("source_count") != len(sources)
                or diagnostics.get("valid_count") != frame_valid_hb
                or diagnostics.get("invalid_count") != len(sources) - frame_valid_hb
                or diagnostics.get("cuda_synchronized") is not True
                or diagnostics.get("model_eval") is not True
                or diagnostics.get("model_parameters_frozen") is not True
                or diagnostics.get("model_forward_calls") != 1
            ):
                raise F4MergeError(f"{scene_id}/{frame_id} frozen provider diagnostics differ")
            warmup = expected_call_index[0] < WARMUP_FORWARD_COUNT
            if runtime.get("f4_warmup_excluded") is not warmup:
                raise F4MergeError(f"{scene_id}/{frame_id} warm-up ledger differs")
            incremental = _number(runtime.get("f4_incremental_ms"), "F4 incremental runtime")
            inherited = _number(runtime.get("sealed_f0_f2_complete_ms"), "sealed F0+F2 runtime")
            composed = _number(runtime.get("replay_composed_ms"), "composed runtime")
            if not math.isclose(composed, inherited + incremental, abs_tol=1.0e-9, rel_tol=0.0):
                raise F4MergeError(f"{scene_id}/{frame_id} composed runtime arithmetic differs")
            if not math.isclose(_number(runtime.get("replay_composed_ms_per_source_frame"), "amortized runtime"), composed / SOURCE_FRAME_STRIDE, abs_tol=1.0e-9, rel_tol=0.0):
                raise F4MergeError(f"{scene_id}/{frame_id} amortized runtime arithmetic differs")
            missed_all = composed >= MAX_COMPOSED_MS_EXCLUSIVE
            missed_warm = (not warmup) and missed_all
            if (
                runtime.get("gap25_deadline_missed") is not missed_all
                or runtime.get("gap25_deadline_missed_warm") is not missed_warm
            ):
                raise F4MergeError(f"{scene_id}/{frame_id} deadline ledger differs")
            deadline_misses_all += int(missed_all)
            deadline_misses_warm += int(missed_warm)
            composed_all.append(composed)
            if not warmup:
                incremental_warm.append(incremental)
                composed_warm.append(composed)
            expected_call_index[0] += 1
            provider_forwards += 1
        elif runtime is not None:
            raise F4MergeError(f"{scene_id}/{frame_id} zero-source frame has runtime/provider evidence")

    if _canonical_json_sha256(before_seals) != inputs.get("frozen_inputs_before_sha256") or inputs.get("frozen_inputs_after_sha256") != inputs.get("frozen_inputs_before_sha256"):
        raise F4MergeError(f"{scene_id} frozen-input aggregate seal differs")
    if len(source_ids) != len(set(source_ids)):
        raise F4MergeError(f"{scene_id} source identities are duplicated")
    counts = {
        "keyframe_count": len(frames),
        "successful_frame_count": successful,
        "source_count": len(source_ids),
        "provider_forward_count": provider_forwards,
        "valid_hb_count": valid_hb,
        "invalid_hb_count": len(source_ids) - valid_hb,
    }
    if counts != scene.get("counts") or counts != scene_row.get("counts"):
        raise F4MergeError(f"{scene_id} scene census differs")
    if _canonical_json_sha256(source_ids) != scene.get("source_ids_sha256") or scene_row.get("source_ids_sha256") != scene.get("source_ids_sha256"):
        raise F4MergeError(f"{scene_id} source-order hash differs")
    if _canonical_json_sha256(lineage_hashes) != scene.get("source_lineage_sha256") or scene_row.get("source_lineage_sha256") != scene.get("source_lineage_sha256"):
        raise F4MergeError(f"{scene_id} source-lineage aggregate differs")
    runtime = scene.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("gap25_all_deadline_miss_count") != deadline_misses_all
        or runtime.get("gap25_warm_deadline_miss_count") != deadline_misses_warm
    ):
        raise F4MergeError(f"{scene_id} runtime summary differs")
    f4_peak = int(_number(runtime.get("f4_cuda_peak_memory_bytes"), "scene F4 CUDA peak"))
    inherited_peak = int(_number(runtime.get("sealed_f0_f2_cuda_peak_memory_bytes"), "scene sealed F0+F2 CUDA peak"))
    cuda_peak = int(_number(runtime.get("cuda_peak_memory_bytes"), "scene CUDA peak"))
    if cuda_peak != max(f4_peak, inherited_peak):
        raise F4MergeError(f"{scene_id} total CUDA peak arithmetic differs")
    f2_summary = f2_scene.get("summary")
    if isinstance(f2_summary, Mapping) and isinstance(f2_summary.get("gpu_peak_memory_bytes"), int):
        if inherited_peak != f2_summary["gpu_peak_memory_bytes"]:
            raise F4MergeError(f"{scene_id} sealed F0+F2 CUDA peak differs")
    return (
        counts,
        source_ids,
        lineage_hashes,
        incremental_warm,
        composed_warm,
        composed_all,
        deadline_misses_all,
        deadline_misses_warm,
        cuda_peak,
    )


def merge_f4(
    *,
    shard_paths: Sequence[Path] = DEFAULT_SHARDS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    expected_scene_count: int = EXPECTED_SCENES,
    expected_keyframes: int | None = None,
    expected_successful_frames: int | None = None,
    expected_sources: int | None = None,
) -> dict[str, Any]:
    if len(shard_paths) != EXPECTED_SHARDS:
        raise F4MergeError("F4 merge requires exactly two shard manifests")
    production = expected_scene_count == EXPECTED_SCENES
    expected_keyframes = EXPECTED_KEYFRAMES if expected_keyframes is None and production else expected_keyframes
    expected_successful_frames = EXPECTED_SUCCESSFUL_FRAMES if expected_successful_frames is None and production else expected_successful_frames
    expected_sources = EXPECTED_SOURCES if expected_sources is None and production else expected_sources

    loaded: list[tuple[Path, dict[str, Any]]] = []
    for index, input_path in enumerate(shard_paths):
        path, shard = _read_json(Path(input_path), f"F4 shard {index}")
        if (
            shard.get("schema") != SHARD_SCHEMA
            or shard.get("protocol_id") != PROTOCOL_ID
            or shard.get("complete") is not True
            or shard.get("shard_index") != index
            or shard.get("num_shards") != EXPECTED_SHARDS
            or shard.get("contracts") != CONTRACTS
            or shard.get("native_output_mutation_count") != 0
        ):
            raise F4MergeError(f"F4 shard {index} contract differs")
        if _content_hash_without(shard, "content_sha256", "manifest_path") != shard.get("content_sha256"):
            raise F4MergeError(f"F4 shard {index} content hash differs")
        loaded.append((path, shard))
    left, right = loaded[0][1], loaded[1][1]
    for key in ("run_signature_sha256", "signature_payload_sha256", "model_receipts_sha256"):
        if left.get(key) != right.get(key) or not isinstance(left.get(key), str):
            raise F4MergeError(f"shard shared {key} differs")
    if left.get("inputs") != right.get("inputs") or left.get("sources_receipt") != right.get("sources_receipt"):
        raise F4MergeError("shard frozen input/source receipts differ")
    for shard in (left, right):
        if shard.get("model_receipts_before") != shard.get("model_receipts_after"):
            raise F4MergeError("frozen model receipts changed within a shard")
        if _canonical_json_sha256(shard.get("model_receipts_before")) != shard.get("model_receipts_sha256"):
            raise F4MergeError("frozen model receipt hash differs")
    if left.get("model_receipts_before") != right.get("model_receipts_before"):
        raise F4MergeError("frozen model receipts differ across shards")

    rows_by_index: dict[int, tuple[int, Mapping[str, Any]]] = {}
    for shard_index, (_, shard) in enumerate(loaded):
        rows = shard.get("scenes")
        if not isinstance(rows, list):
            raise F4MergeError(f"shard {shard_index} scene rows are absent")
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("scene_index"), int):
                raise F4MergeError(f"shard {shard_index} scene row is malformed")
            scene_index = row["scene_index"]
            if scene_index % EXPECTED_SHARDS != shard_index or scene_index in rows_by_index:
                raise F4MergeError("scene shard partition differs")
            rows_by_index[scene_index] = (shard_index, row)
    if sorted(rows_by_index) != list(range(expected_scene_count)):
        raise F4MergeError("merged scene partition is incomplete")

    totals = {key: 0 for key in ("keyframe_count", "successful_frame_count", "source_count", "provider_forward_count", "valid_hb_count", "invalid_hb_count")}
    source_ids: list[str] = []
    lineage_hashes: list[str] = []
    incremental_warm: list[float] = []
    composed_warm: list[float] = []
    composed_all: list[float] = []
    deadline_misses_all = 0
    deadline_misses_warm = 0
    cuda_peak = 0
    scene_rows: list[dict[str, Any]] = []
    call_indices = [[0], [0]]
    for scene_index in range(expected_scene_count):
        shard_index, row = rows_by_index[scene_index]
        (
            counts,
            ids,
            lineages,
            incremental,
            composed,
            all_composed,
            all_misses,
            warm_misses,
            peak,
        ) = _validate_scene(
            row,
            expected_scene_index=scene_index,
            expected_call_index=call_indices[shard_index],
            expected_run_signature=str(left["run_signature_sha256"]),
        )
        for key in totals:
            totals[key] += counts[key]
        source_ids.extend(ids)
        lineage_hashes.extend(lineages)
        incremental_warm.extend(incremental)
        composed_warm.extend(composed)
        composed_all.extend(all_composed)
        deadline_misses_all += all_misses
        deadline_misses_warm += warm_misses
        cuda_peak = max(cuda_peak, peak)
        scene_rows.append(dict(row))

    if len(source_ids) != len(set(source_ids)):
        raise F4MergeError("global F4 source identities are duplicated")
    if expected_keyframes is not None and totals["keyframe_count"] != expected_keyframes:
        raise F4MergeError("merged keyframe census differs")
    if expected_successful_frames is not None and totals["successful_frame_count"] != expected_successful_frames:
        raise F4MergeError("merged successful-frame census differs")
    if expected_sources is not None and totals["source_count"] != expected_sources:
        raise F4MergeError("merged source census differs")
    if production:
        for shard_index, (_, shard) in enumerate(loaded):
            for key, expected in EXPECTED_SHARD_COUNTS[shard_index].items():
                if shard.get("totals", {}).get(key) != expected:
                    raise F4MergeError(f"production shard {shard_index} {key} differs")

    incremental_distribution = _distribution(incremental_warm)
    composed_distribution = _distribution(composed_warm)
    composed_all_distribution = _distribution(composed_all)
    composed_mean_per_source_frame = float(composed_distribution["mean"]) / SOURCE_FRAME_STRIDE
    gates = {
        "integrity_complete": _gate(len(scene_rows), "==", expected_scene_count),
        "exact_keyframes": _gate(totals["keyframe_count"], "==", expected_keyframes if expected_keyframes is not None else totals["keyframe_count"]),
        "exact_successful_frames": _gate(totals["successful_frame_count"], "==", expected_successful_frames if expected_successful_frames is not None else totals["successful_frame_count"]),
        "exact_sources": _gate(totals["source_count"], "==", expected_sources if expected_sources is not None else totals["source_count"]),
        "f4_incremental_warm_p95_ms": _gate(float(incremental_distribution["p95"]), "<=", MAX_F4_INCREMENTAL_P95_MS),
        "replay_composed_warm_p95_ms": _gate(float(composed_distribution["p95"]), "<=", MAX_COMPOSED_P95_MS),
        "replay_composed_warm_max_ms": _gate(float(composed_distribution["max"]), "<", MAX_COMPOSED_MS_EXCLUSIVE),
        "replay_composed_mean_per_source_frame_ms": _gate(composed_mean_per_source_frame, "<=", MAX_COMPOSED_MEAN_PER_SOURCE_FRAME_MS),
        "gap25_warm_deadline_miss_count": _gate(deadline_misses_warm, "==", 0),
        "cuda_peak_memory_bytes": _gate(cuda_peak, "<=", MAX_CUDA_PEAK_BYTES),
        "native_output_mutation_count": _gate(0, "==", 0),
    }
    overall_pass = all(gate["pass"] for gate in gates.values())
    runtime_gate_names = (
        "f4_incremental_warm_p95_ms",
        "replay_composed_warm_p95_ms",
        "replay_composed_warm_max_ms",
        "replay_composed_mean_per_source_frame_ms",
        "gap25_warm_deadline_miss_count",
        "cuda_peak_memory_bytes",
    )
    runtime_pass = all(gates[name]["pass"] for name in runtime_gate_names)
    totals["identity_verified_source_count"] = totals["source_count"]
    receipt: dict[str, Any] = {
        "schema": MERGE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "overall_pass": overall_pass,
        "run_signature_sha256": left["run_signature_sha256"],
        "contracts": dict(CONTRACTS),
        "inputs": {
            "shards": [{"path": os.fspath(path), "sha256": _sha256(path), "shard_index": index} for index, (path, _) in enumerate(loaded)],
            "f2_receipt": left["inputs"]["f2_receipt"],
            "f0_receipt": left["inputs"]["f0_receipt"],
            "scene_list": left["inputs"]["scene_list"],
            "sources_receipt": left["sources_receipt"],
            "model_receipts": left["model_receipts_before"],
            "model_receipts_sha256": left["model_receipts_sha256"],
        },
        "coverage": {
            "scene_count": len(scene_rows),
            "scene_order": [row["scene_id"] for row in scene_rows],
            "keyframe_count": totals["keyframe_count"],
            "successful_frame_count": totals["successful_frame_count"],
            "source_count": totals["source_count"],
            "exact_source_partition": True,
            "exact_source_order": True,
            "source_ids_sha256": _canonical_json_sha256(source_ids),
            "source_lineage_sha256": _canonical_json_sha256(lineage_hashes),
        },
        "causality": {
            "overall_pass": True,
            "current_frame_only": True,
            "maximum_lookahead_frames": 0,
            "maximum_logical_accessed_ordinal": True,
            "future_frame_access": False,
            "source_order_identity": True,
            "provider_called_only_for_nonempty_successful_frames": True,
            "first_three_nonempty_forwards_per_shard_excluded_only_from_warm_distributions": True,
        },
        "totals": totals,
        "runtime": {
            "overall_pass": runtime_pass,
            "gates": {name: gates[name] for name in runtime_gate_names},
            "f4_incremental_warm_ms": incremental_distribution,
            "replay_composed_warm_ms": composed_distribution,
            "replay_composed_all_ms": composed_all_distribution,
            "replay_composed_mean_per_source_frame_ms": composed_mean_per_source_frame,
            "gap25_all_deadline_miss_count": deadline_misses_all,
            "gap25_warm_deadline_miss_count": deadline_misses_warm,
            "cuda_peak_memory_bytes": cuda_peak,
            "cold_model_load_excluded": True,
            "warmup_forward_count_per_shard": [min(WARMUP_FORWARD_COUNT, call[0]) for call in call_indices],
        },
        "gates": gates,
        "scenes": scene_rows,
        "native_output_mutation_count": 0,
        "oracle_authorization": {
            "allowed": overall_pass,
            "scope": "separate_post_seal_f4_geometry_capacity_oracle_only",
            "active_birth_authorized": False,
        },
    }
    receipt["content_sha256"] = _canonical_json_sha256(receipt)
    output_path = Path(output_dir) / OUTPUT_NAME
    output_sha = _atomic_create_json(output_path, receipt)
    receipt["receipt_path"] = os.fspath(output_path.resolve())
    receipt["receipt_sha256"] = output_sha
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, action="append", dest="shards")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = _parser().parse_args()
    shards = tuple(args.shards) if args.shards is not None else DEFAULT_SHARDS
    result = merge_f4(shard_paths=shards, output_dir=args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
