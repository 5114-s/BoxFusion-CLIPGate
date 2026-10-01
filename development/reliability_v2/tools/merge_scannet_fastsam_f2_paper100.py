#!/usr/bin/env python3
"""Validate two F2 paper100 shards and publish one create-only receipt.

Only F2 JSON/NPZ receipts and the sealed scene list are accepted.  Prediction
pickles, ground truth, annotations, and evaluator inputs have no interface in
this reducer.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "boxfusion.scannet_fastsam_f2_paper100.merge.v1"
SHARD_SCHEMA = "boxfusion.scannet_fastsam_f2_paper100.shard.v1"
SCENE_SCHEMA = "boxfusion.scannet_fastsam_f2_paper100.scene.v1"
EVIDENCE_SCHEMA = "boxfusion.scannet_fastsam_f2_paper100.evidence.v1"
PROTOCOL_ID = "F2-DFU-LGF-lite-shadow-paper100"
OUTPUT_NAME = "F2_FASTSAM_PAPER100.json"
EXPECTED_FULL200_SCENE_LIST_SHA256 = (
    "0e7e722d3e93ec4b721f12293a3f1e98ca62d475b42cc8b9d491878a897e9bd1"
)
EXPECTED_SCENES = 100
EXPECTED_SHARDS = 2
EXPECTED_KEYFRAMES = 6_817
EXPECTED_SUCCESSFUL_FRAMES = 6_726
EXPECTED_SOURCES = 52_299
EXPECTED_INVALID_POSE_FRAMES = 89
EXPECTED_NON_UPRIGHT_FRAMES = 2
EXPECTED_SHARD_COUNTS: Mapping[int, Mapping[str, int]] = {
    0: {"keyframes": 3_259, "successful_frames": 3_189, "sources": 24_863},
    1: {"keyframes": 3_558, "successful_frames": 3_537, "sources": 27_436},
}
MASK_SHAPE = (480, 640)
MASK_PACKED_BYTES = MASK_SHAPE[0] * MASK_SHAPE[1] // 8
MAX_COMPLETE_P95_MS = 250.0
MAX_PROVIDER_P95_MS = 200.0
MAX_COMPLETE_MS_EXCLUSIVE = 833.33
MAX_AMORTIZED_MS_PER_SOURCE_FRAME = 10.0
MAX_F2_AMORTIZED_MS_PER_SOURCE_FRAME = 2.0
SOURCE_FRAME_STRIDE = 25.0
MAX_GPU_PEAK_BYTES = 4 * 1024**3
DEFAULT_SCENE_LIST = (
    REPOSITORY_ROOT / "evaluation/data_util/meta_data/scannetv2_val_f0_full200.txt"
)
DEFAULT_SHARDS = (
    REPOSITORY_ROOT
    / "logs/scannet_fastsam_f2_paper100_score05/shards/shard-000-of-002.json",
    REPOSITORY_ROOT
    / "logs/scannet_fastsam_f2_paper100_score05/shards/shard-001-of-002.json",
)
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT / "logs/scannet_fastsam_f2_paper100_score05/final"
)
SOURCE_ID_RE = re.compile(
    r"^(scene\d{4}_\d{2})/frame_(\d{6})/raw_(\d{3})$"
)


class F2MergeError(RuntimeError):
    """Raised when F2 receipt structure or provenance differs."""


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


def _regular_file(path: Path, label: str, suffix: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise F2MergeError(f"{label} must be a regular non-symlink file: {path}")
    result = path.resolve()
    if suffix is not None and result.suffix.lower() != suffix:
        raise F2MergeError(f"{label} must be a {suffix} file: {result}")
    if result.suffix.lower() in {".pkl", ".pickle"}:
        raise F2MergeError(f"prediction pickle input is forbidden: {result}")
    return result


def _read_json(path: Path, label: str) -> tuple[Path, dict[str, Any]]:
    source = _regular_file(path, label, ".json")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise F2MergeError(f"invalid {label}: {source}") from error
    if not isinstance(value, dict):
        raise F2MergeError(f"{label} must contain one JSON object: {source}")
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
            raise F2MergeError(f"refusing to overwrite output: {path}") from error
        return _sha256(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_scene_list(
    path: Path, expected_scene_count: int
) -> tuple[Path, list[str], dict[str, Any]]:
    source = _regular_file(path, "F2 scene ledger")
    rows = [
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if expected_scene_count == EXPECTED_SCENES:
        if len(rows) != 200 or _sha256(source) != EXPECTED_FULL200_SCENE_LIST_SHA256:
            raise F2MergeError("production F2 requires the sealed full200 scene ledger")
        selected = rows[:EXPECTED_SCENES]
    else:
        if len(rows) != expected_scene_count:
            raise F2MergeError("test F2 scene count differs")
        selected = rows
    if len(set(selected)) != len(selected):
        raise F2MergeError("F2 paper100 prefix has duplicate scenes")
    return source, selected, {
        "path": os.fspath(source),
        "sha256": _sha256(source),
        "source_scene_count": len(rows),
        "selected_prefix_count": len(selected),
        "selected_order_sha256": _canonical_json_sha256(selected),
    }


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise F2MergeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise F2MergeError(f"{label} must be finite and non-negative")
    return result


def _xyz(value: object, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise F2MergeError(f"{label} must be a finite xyz vector")
    return array


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _validate_hypotheses(
    source: Mapping[str, Any],
    label: str,
    candidate_runtime_samples: defaultdict[str, list[float]],
) -> None:
    hypotheses = source.get("hypotheses")
    if not isinstance(hypotheses, dict) or set(hypotheses) != {"H0", "HL", "HLG"}:
        raise F2MergeError(f"{label} must contain exactly H0/HL/HLG")
    f0_q02 = _xyz(source.get("f0_world_q02"), f"{label}.f0_world_q02")
    f0_q98 = _xyz(source.get("f0_world_q98"), f"{label}.f0_world_q98")
    if np.any(f0_q98 <= f0_q02):
        raise F2MergeError(f"{label} F0 geometry has a non-positive extent")
    for name in ("H0", "HL", "HLG"):
        row = hypotheses[name]
        if not isinstance(row, dict) or row.get("valid") is not True:
            raise F2MergeError(f"{label}.{name} is not a valid fail-open geometry")
        q02 = _xyz(row.get("q02"), f"{label}.{name}.q02")
        q98 = _xyz(row.get("q98"), f"{label}.{name}.q98")
        center = _xyz(row.get("center"), f"{label}.{name}.center")
        extent = _xyz(row.get("extent"), f"{label}.{name}.extent")
        if (
            np.any(q98 <= q02)
            or not np.allclose(center, (q02 + q98) * 0.5, rtol=0.0, atol=1e-12)
            or not np.allclose(extent, q98 - q02, rtol=0.0, atol=1e-12)
            or not isinstance(row.get("stored_point_count"), int)
            or row["stored_point_count"] < 1
            or not isinstance(row.get("points_and_voxel_keys_sha256"), str)
            or len(row["points_and_voxel_keys_sha256"]) != 64
            or not isinstance(row.get("diagnostics"), dict)
        ):
            raise F2MergeError(f"{label}.{name} geometry/receipt is invalid")
    if not np.array_equal(_xyz(hypotheses["H0"]["q02"], "H0.q02"), f0_q02) or not np.array_equal(
        _xyz(hypotheses["H0"]["q98"], "H0.q98"), f0_q98
    ):
        raise F2MergeError(f"{label} H0 differs bitwise from F0 geometry")
    receipt = source.get("f2_receipt")
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema")
        != "boxfusion.fastsam_dfu_lgf_shadow.f2.v1"
        or receipt.get("mode") != "shadow"
        or not isinstance(receipt.get("input_sha256"), str)
        or len(receipt["input_sha256"]) != 64
        or not isinstance(receipt.get("result_sha256"), str)
        or len(receipt["result_sha256"]) != 64
        or not isinstance(receipt.get("diagnostics"), dict)
    ):
        raise F2MergeError(f"{label} F2 core receipt is invalid")
    diagnostics = receipt["diagnostics"]
    for source_key, output_key in (
        ("validation_elapsed_ms", "f2_candidate_validation_ms"),
        ("local_elapsed_ms", "f2_candidate_local_ms"),
        ("global_elapsed_ms", "f2_candidate_global_ms"),
        ("total_elapsed_ms", "f2_candidate_total_ms"),
    ):
        candidate_runtime_samples[output_key].append(
            _number(diagnostics.get(source_key), f"{label}.{source_key}")
        )


def _validate_evidence(
    *,
    evidence_path: Path,
    evidence_sha: str,
    scene: str,
    json_sources: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    source = _regular_file(evidence_path, f"F2 evidence {scene}", ".npz")
    if _sha256(source) != evidence_sha:
        raise F2MergeError(f"F2 evidence rehash differs: {scene}")
    try:
        archive = np.load(source, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise F2MergeError(f"could not load F2 evidence: {scene}") from error
    with archive:
        required = {
            "schema",
            "scene_id",
            "mask_shape",
            "mask_bitorder",
            "source_ids",
            "frame_ids",
            "raw_indices",
            "ranks",
            "candidate_indices",
            "masks_packbits",
            "point_offsets",
            "points_world",
            "voxel_keys",
            "hl_index_offsets",
            "hl_retained_indices",
            "hlg_index_offsets",
            "hlg_retained_indices",
        }
        if set(archive.files) != required:
            raise F2MergeError(f"F2 evidence array schema differs: {scene}")
        if (
            str(archive["schema"].item()) != EVIDENCE_SCHEMA
            or str(archive["scene_id"].item()) != scene
            or archive["mask_shape"].tolist() != list(MASK_SHAPE)
            or str(archive["mask_bitorder"].item()) != "little"
        ):
            raise F2MergeError(f"F2 evidence metadata differs: {scene}")
        source_ids = [str(value) for value in archive["source_ids"].tolist()]
        expected_ids = [str(row["source_id"]) for row in json_sources]
        count = len(expected_ids)
        if source_ids != expected_ids:
            raise F2MergeError(f"F2 JSON/NPZ source order differs: {scene}")
        frame_ids = archive["frame_ids"]
        raw_indices = archive["raw_indices"]
        ranks = archive["ranks"]
        candidate_indices = archive["candidate_indices"]
        masks = archive["masks_packbits"]
        point_offsets = archive["point_offsets"]
        points = archive["points_world"]
        keys = archive["voxel_keys"]
        hl_offsets = archive["hl_index_offsets"]
        hl_indices = archive["hl_retained_indices"]
        hlg_offsets = archive["hlg_index_offsets"]
        hlg_indices = archive["hlg_retained_indices"]
        if (
            frame_ids.shape != (count,)
            or raw_indices.shape != (count,)
            or ranks.shape != (count,)
            or candidate_indices.shape != (count,)
            or masks.shape != (count, MASK_PACKED_BYTES)
            or masks.dtype != np.uint8
            or point_offsets.shape != (count + 1,)
            or point_offsets[0] != 0
            or np.any(point_offsets[1:] <= point_offsets[:-1])
            or points.shape != keys.shape
            or points.shape != (int(point_offsets[-1]), 3)
            or points.dtype != np.dtype("<f8")
            or keys.dtype != np.dtype("<i8")
            or hl_offsets.shape != (count + 1,)
            or hlg_offsets.shape != (count + 1,)
            or hl_offsets[0] != 0
            or hlg_offsets[0] != 0
            or np.any(hl_offsets[1:] <= hl_offsets[:-1])
            or np.any(hlg_offsets[1:] <= hlg_offsets[:-1])
            or hl_indices.shape != (int(hl_offsets[-1]),)
            or hlg_indices.shape != (int(hlg_offsets[-1]),)
            or hl_indices.dtype != np.dtype("<i8")
            or hlg_indices.dtype != np.dtype("<i8")
        ):
            raise F2MergeError(f"F2 evidence shapes/dtypes differ: {scene}")
        for index, row in enumerate(json_sources):
            if (
                int(frame_ids[index]) != row["_frame_id"]
                or int(raw_indices[index]) != row["raw_index"]
                or int(ranks[index]) != row["rank"]
                or int(candidate_indices[index]) != row["candidate_index"]
                or hashlib.sha256(masks[index].tobytes()).hexdigest()
                != row["mask_sha256"]
            ):
                raise F2MergeError(f"F2 evidence identity differs: {row['source_id']}")
            point_start = int(point_offsets[index])
            point_stop = int(point_offsets[index + 1])
            raw_points = points[point_start:point_stop]
            raw_keys = keys[point_start:point_stop]
            raw_digest = hashlib.sha256()
            raw_digest.update(np.asarray(raw_points, dtype="<f8").tobytes())
            raw_digest.update(np.asarray(raw_keys, dtype="<i8").tobytes())
            if (
                point_stop - point_start != row["stored_point_count"]
                or raw_digest.hexdigest()
                != row["points_and_voxel_keys_sha256"]
                or raw_digest.hexdigest()
                != row["hypotheses"]["H0"]["points_and_voxel_keys_sha256"]
            ):
                raise F2MergeError(f"F2 H0 point evidence differs: {row['source_id']}")
            for name, index_values, index_offsets in (
                ("HL", hl_indices, hl_offsets),
                ("HLG", hlg_indices, hlg_offsets),
            ):
                start = int(index_offsets[index])
                stop = int(index_offsets[index + 1])
                selected = np.asarray(index_values[start:stop], dtype=np.int64)
                if (
                    len(selected) != row["hypotheses"][name]["stored_point_count"]
                    or np.any(selected < 0)
                    or np.any(selected >= len(raw_points))
                    or (len(selected) > 1 and np.any(selected[1:] <= selected[:-1]))
                ):
                    raise F2MergeError(
                        f"F2 {name} retained-index evidence differs: {row['source_id']}"
                    )
                digest = hashlib.sha256()
                digest.update(np.asarray(raw_points[selected], dtype="<f8").tobytes())
                digest.update(np.asarray(raw_keys[selected], dtype="<i8").tobytes())
                if (
                    digest.hexdigest()
                    != row["hypotheses"][name]["points_and_voxel_keys_sha256"]
                ):
                    raise F2MergeError(
                        f"F2 {name} point evidence differs: {row['source_id']}"
                    )
        return {
            "source_count": count,
            "raw_point_count": int(point_offsets[-1]),
            "hl_retained_index_count": int(hl_offsets[-1]),
            "hlg_retained_index_count": int(hlg_offsets[-1]),
        }


def merge_f2(
    *,
    shard_paths: Sequence[Path],
    scene_list_path: Path,
    output_dir: Path,
    _expected_scene_count: int = EXPECTED_SCENES,
) -> dict[str, Any]:
    """Validate shard/scene/evidence receipts and publish the F2 final receipt."""

    production = _expected_scene_count == EXPECTED_SCENES
    expected_shards = EXPECTED_SHARDS if production else len(shard_paths)
    if len(shard_paths) != expected_shards or not shard_paths:
        raise F2MergeError(f"expected {expected_shards} F2 shard manifests")
    scene_list_source, scenes, scene_list = _read_scene_list(
        scene_list_path, _expected_scene_count
    )
    shards: dict[int, tuple[Path, dict[str, Any]]] = {}
    for raw_path in shard_paths:
        path, value = _read_json(raw_path, "F2 shard manifest")
        shard = value.get("shard")
        if (
            value.get("schema") != SHARD_SCHEMA
            or value.get("protocol_id") != PROTOCOL_ID
            or value.get("complete") is not True
            or value.get("mode") != "shadow"
            or not isinstance(shard, dict)
            or not isinstance(shard.get("index"), int)
            or shard.get("count") != expected_shards
            or shard["index"] in shards
        ):
            raise F2MergeError(f"invalid F2 shard contract: {path}")
        shards[shard["index"]] = (path, value)
    if set(shards) != set(range(expected_shards)):
        raise F2MergeError("F2 shard indices are incomplete")
    signatures = {value["run_signature_sha256"] for _path, value in shards.values()}
    if len(signatures) != 1:
        raise F2MergeError("F2 shard run signatures differ")
    run_signature = next(iter(signatures))
    checkpoint_receipts = {
        _canonical_json_sha256(value.get("checkpoint"))
        for _path, value in shards.values()
    }
    source_receipts = {
        _canonical_json_sha256(value.get("sources_receipt"))
        for _path, value in shards.values()
    }
    if len(checkpoint_receipts) != 1 or len(source_receipts) != 1:
        raise F2MergeError("F2 shard execution identities differ")

    scene_rows: dict[int, tuple[Path, Mapping[str, Any], Mapping[str, Any]]] = {}
    for shard_index, (manifest_path, manifest) in shards.items():
        expected_indices = [
            index for index in range(len(scenes)) if index % expected_shards == shard_index
        ]
        expected_order = [scenes[index] for index in expected_indices]
        shard = manifest["shard"]
        rows = manifest.get("scenes")
        if (
            shard.get("scene_indices") != expected_indices
            or shard.get("scene_order") != expected_order
            or manifest.get("paper100_scene_order") != scenes
            or manifest.get("scene_list") != scene_list
            or not isinstance(rows, list)
            or len(rows) != len(expected_indices)
        ):
            raise F2MergeError(f"F2 shard scene coverage differs: {manifest_path}")
        if production and any(
            int(manifest.get("totals", {}).get(key, -1)) != value
            for key, value in EXPECTED_SHARD_COUNTS[shard_index].items()
        ):
            raise F2MergeError(f"F2 shard census differs: {manifest_path}")
        for expected_index, expected_scene, row in zip(
            expected_indices, expected_order, rows
        ):
            if (
                not isinstance(row, dict)
                or row.get("scene_index") != expected_index
                or row.get("scene_id") != expected_scene
                or not isinstance(row.get("sidecar_path"), str)
                or not isinstance(row.get("evidence_npz_path"), str)
            ):
                raise F2MergeError(f"F2 shard scene row differs: {expected_scene}")
            sidecar_path, sidecar = _read_json(
                Path(row["sidecar_path"]), f"F2 scene sidecar {expected_scene}"
            )
            if _sha256(sidecar_path) != row.get("sidecar_sha256"):
                raise F2MergeError(f"F2 scene sidecar rehash differs: {expected_scene}")
            scene_rows[expected_index] = (sidecar_path, row, sidecar)

    if set(scene_rows) != set(range(len(scenes))):
        raise F2MergeError("F2 scene union is incomplete")
    totals: Counter[str] = Counter()
    runtime_samples: defaultdict[str, list[float]] = defaultdict(list)
    candidate_runtime_samples: defaultdict[str, list[float]] = defaultdict(list)
    all_source_ids: set[str] = set()
    output_scene_rows: list[dict[str, Any]] = []
    gpu_peak = 0
    cpu_peak = 0
    raw_point_total = 0
    hl_retained_index_total = 0
    hlg_retained_index_total = 0
    for scene_index, scene in enumerate(scenes):
        sidecar_path, manifest_row, sidecar = scene_rows[scene_index]
        if (
            sidecar.get("schema") != SCENE_SCHEMA
            or sidecar.get("protocol_id") != PROTOCOL_ID
            or sidecar.get("complete") is not True
            or sidecar.get("run_signature_sha256") != run_signature
            or sidecar.get("scene_id") != scene
            or sidecar.get("scene_index") != scene_index
        ):
            raise F2MergeError(f"F2 scene contract differs: {scene}")
        frames = sidecar.get("frames")
        summary = sidecar.get("summary")
        if not isinstance(frames, list) or not isinstance(summary, dict):
            raise F2MergeError(f"F2 scene frames/summary missing: {scene}")
        json_sources: list[dict[str, Any]] = []
        scene_counts: Counter[str] = Counter()
        frame_ids_seen: set[int] = set()
        for frame in frames:
            frame_id = frame.get("frame_id")
            if not isinstance(frame_id, int) or frame_id in frame_ids_seen:
                raise F2MergeError(f"F2 frame ledger invalid: {scene}")
            frame_ids_seen.add(frame_id)
            scene_counts["keyframes"] += 1
            sources = frame.get("sources")
            if not isinstance(sources, list):
                raise F2MergeError(f"F2 frame sources invalid: {scene}/{frame_id}")
            if frame.get("successful") is True:
                identity = frame.get("identity")
                runtime = frame.get("runtime")
                if (
                    not isinstance(identity, dict)
                    or identity.get("exact_equal") is not True
                    or identity.get("selected_source_count") != len(sources)
                    or not isinstance(runtime, dict)
                ):
                    raise F2MergeError(f"F2 frame identity differs: {scene}/{frame_id}")
                scene_counts["successful_frames"] += 1
                scene_counts["identity_verified_frames"] += 1
                scene_counts["replayed_raw_masks"] += int(identity["mask_count"])
                if not runtime.get("warmup_excluded"):
                    for name in ("provider_ms", "f0_core_ms", "f2_core_ms", "complete_ms"):
                        runtime_samples[name].append(
                            _number(runtime.get(name), f"{scene}/{frame_id}.{name}")
                        )
            elif frame.get("abstention") == "invalid_current_pose":
                scene_counts["invalid_pose_frames"] += 1
                if sources:
                    raise F2MergeError("abstained F2 frame contains sources")
            elif frame.get("abstention") == "non_upright_cache_coordinate_frame":
                scene_counts["non_upright_producer_frames"] += 1
                if sources:
                    raise F2MergeError("abstained F2 frame contains sources")
            else:
                raise F2MergeError(f"unknown F2 frame state: {scene}/{frame_id}")
            for candidate_index, source in enumerate(sources):
                if not isinstance(source, dict):
                    raise F2MergeError(f"invalid F2 source: {scene}/{frame_id}")
                source_id = source.get("source_id")
                match = SOURCE_ID_RE.fullmatch(str(source_id))
                if (
                    match is None
                    or match.group(1) != scene
                    or int(match.group(2)) != frame_id
                    or int(match.group(3)) != source.get("raw_index")
                    or source.get("candidate_index") != candidate_index
                    or source.get("rank") != candidate_index
                    or source_id in all_source_ids
                    or not isinstance(source.get("mask_sha256"), str)
                    or len(source["mask_sha256"]) != 64
                    or not isinstance(source.get("points_and_voxel_keys_sha256"), str)
                    or len(source["points_and_voxel_keys_sha256"]) != 64
                ):
                    raise F2MergeError(f"F2 source identity invalid: {source_id}")
                _validate_hypotheses(
                    source, str(source_id), candidate_runtime_samples
                )
                all_source_ids.add(str(source_id))
                copied = dict(source)
                copied["_frame_id"] = frame_id
                json_sources.append(copied)
                scene_counts["sources"] += 1
                scene_counts["identity_verified_sources"] += 1
        recorded_counts = summary.get("counts")
        if not isinstance(recorded_counts, dict) or any(
            int(recorded_counts.get(key, -1)) != value
            for key, value in scene_counts.items()
        ):
            raise F2MergeError(f"F2 scene count summary differs: {scene}")
        evidence = sidecar.get("evidence_npz")
        if (
            not isinstance(evidence, dict)
            or evidence.get("path") != manifest_row.get("evidence_npz_path")
            or evidence.get("sha256") != manifest_row.get("evidence_npz_sha256")
        ):
            raise F2MergeError(f"F2 evidence reference differs: {scene}")
        evidence_counts = _validate_evidence(
            evidence_path=Path(evidence["path"]),
            evidence_sha=evidence["sha256"],
            scene=scene,
            json_sources=json_sources,
        )
        if evidence_counts["source_count"] != scene_counts["sources"]:
            raise F2MergeError(f"F2 evidence source count differs: {scene}")
        raw_point_total += evidence_counts["raw_point_count"]
        hl_retained_index_total += evidence_counts["hl_retained_index_count"]
        hlg_retained_index_total += evidence_counts["hlg_retained_index_count"]
        totals.update(scene_counts)
        cpu_peak = max(cpu_peak, int(summary["cpu_peak_rss_bytes"]))
        gpu_peak = max(gpu_peak, int(summary["gpu_peak_memory_bytes"]))
        output_scene_rows.append(
            {
                "scene_id": scene,
                "scene_index": scene_index,
                "sidecar": {
                    "path": os.fspath(sidecar_path),
                    "sha256": _sha256(sidecar_path),
                },
                "evidence_npz": {
                    "path": evidence["path"],
                    "sha256": evidence["sha256"],
                },
                "counts": dict(sorted(scene_counts.items())),
            }
        )
    if production and (
        totals["keyframes"] != EXPECTED_KEYFRAMES
        or totals["successful_frames"] != EXPECTED_SUCCESSFUL_FRAMES
        or totals["sources"] != EXPECTED_SOURCES
        or totals["identity_verified_sources"] != EXPECTED_SOURCES
        or totals["invalid_pose_frames"] != EXPECTED_INVALID_POSE_FRAMES
        or totals["non_upright_producer_frames"] != EXPECTED_NON_UPRIGHT_FRAMES
    ):
        raise F2MergeError("F2 final paper100 census differs")
    complete_distribution = _distribution(runtime_samples["complete_ms"])
    runtime = {
        key: _distribution(values) for key, values in sorted(runtime_samples.items())
    }
    for name in ("provider_ms", "f0_core_ms", "f2_core_ms", "complete_ms"):
        runtime.setdefault(name, _distribution(runtime_samples[name]))
    runtime["f2_candidate_diagnostics"] = {
        key: _distribution(values)
        for key, values in sorted(candidate_runtime_samples.items())
    }
    runtime["amortized_complete_ms_per_source_frame"] = (
        float(complete_distribution["mean"]) / SOURCE_FRAME_STRIDE
    )
    runtime["amortized_f2_core_ms_per_source_frame"] = (
        float(runtime["f2_core_ms"]["mean"]) / SOURCE_FRAME_STRIDE
    )
    identity_ratio = (
        totals["identity_verified_sources"] / totals["sources"]
        if totals["sources"]
        else 1.0
    )
    coverage_pass = (
            not production
            or (
                totals["keyframes"] == EXPECTED_KEYFRAMES
                and totals["successful_frames"] == EXPECTED_SUCCESSFUL_FRAMES
                and totals["sources"] == EXPECTED_SOURCES
            )
        )
    identity_pass = identity_ratio == 1.0
    gate_items = {
        "coverage": {
            "actual": int(totals["sources"]),
            "comparator": "==",
            "threshold": EXPECTED_SOURCES if production else int(totals["sources"]),
            "passed": coverage_pass,
        },
        "f0_identity_ratio": {
            "actual": identity_ratio,
            "comparator": "==",
            "threshold": 1.0,
            "passed": identity_pass,
        },
        "provider_runtime_p95_ms": {
            "actual": runtime["provider_ms"]["p95"],
            "comparator": "<=",
            "threshold": MAX_PROVIDER_P95_MS,
            "passed": runtime["provider_ms"]["p95"] <= MAX_PROVIDER_P95_MS,
        },
        "complete_runtime_p95_ms": {
            "actual": complete_distribution["p95"],
            "comparator": "<=",
            "threshold": MAX_COMPLETE_P95_MS,
            "passed": complete_distribution["p95"] <= MAX_COMPLETE_P95_MS,
        },
        "complete_runtime_max_ms": {
            "actual": complete_distribution["max"],
            "comparator": "<",
            "threshold": MAX_COMPLETE_MS_EXCLUSIVE,
            "passed": complete_distribution["max"] < MAX_COMPLETE_MS_EXCLUSIVE,
        },
        "amortized_complete_ms_per_source_frame": {
            "actual": runtime["amortized_complete_ms_per_source_frame"],
            "comparator": "<=",
            "threshold": MAX_AMORTIZED_MS_PER_SOURCE_FRAME,
            "passed": runtime["amortized_complete_ms_per_source_frame"]
            <= MAX_AMORTIZED_MS_PER_SOURCE_FRAME,
        },
        "amortized_f2_core_ms_per_source_frame": {
            "actual": runtime["amortized_f2_core_ms_per_source_frame"],
            "comparator": "<=",
            "threshold": MAX_F2_AMORTIZED_MS_PER_SOURCE_FRAME,
            "passed": runtime["amortized_f2_core_ms_per_source_frame"]
            <= MAX_F2_AMORTIZED_MS_PER_SOURCE_FRAME,
        },
        "gpu_peak_memory_bytes": {
            "actual": gpu_peak,
            "comparator": "<=",
            "threshold": MAX_GPU_PEAK_BYTES,
            "passed": gpu_peak <= MAX_GPU_PEAK_BYTES,
        },
    }
    runtime_names = (
        "provider_runtime_p95_ms",
        "complete_runtime_p95_ms",
        "complete_runtime_max_ms",
        "amortized_complete_ms_per_source_frame",
        "amortized_f2_core_ms_per_source_frame",
        "gpu_peak_memory_bytes",
    )
    runtime_pass = all(gate_items[name]["passed"] for name in runtime_names)
    overall_pass = all(item["passed"] for item in gate_items.values())
    gates = {
        **gate_items,
        "runtime": {"overall_pass": runtime_pass, "gate_names": list(runtime_names)},
        "overall_pass": overall_pass,
    }
    receipt = {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "overall_pass": overall_pass,
        "run_signature_sha256": run_signature,
        "scene_list": scene_list,
        "coverage": {
            "scene_count": len(scenes),
            "scene_order": scenes,
            "expected_keyframe_count": (
                EXPECTED_KEYFRAMES if production else totals["keyframes"]
            ),
            "keyframe_count": totals["keyframes"],
            "successful_frame_count": totals["successful_frames"],
            "source_count": totals["sources"],
            "identity_verified_source_count": totals[
                "identity_verified_sources"
            ],
            "identity_ratio": identity_ratio,
        },
        "contracts": {
            "shadow_only": True,
            "birth_enabled": False,
            "ground_truth_access": False,
            "prediction_access": False,
            "evaluator_access": False,
            "native_output_mutation": False,
            "training": False,
            "f0_exact_replay_required": True,
        },
        "inputs": {
            "scene_list": os.fspath(scene_list_source),
            "shards": [
                {"path": os.fspath(path), "sha256": _sha256(path)}
                for path, _value in (shards[index] for index in sorted(shards))
            ],
        },
        "scenes": output_scene_rows,
        "totals": {
            "scene_count": len(scenes),
            "keyframe_count": totals["keyframes"],
            "successful_frame_count": totals["successful_frames"],
            "source_count": totals["sources"],
            "identity_verified_source_count": totals["identity_verified_sources"],
            "invalid_pose_frame_count": totals["invalid_pose_frames"],
            "non_upright_producer_frame_count": totals[
                "non_upright_producer_frames"
            ],
            "replayed_raw_mask_count": totals["replayed_raw_masks"],
            "raw_point_count": raw_point_total,
            "hl_retained_index_count": hl_retained_index_total,
            "hlg_retained_index_count": hlg_retained_index_total,
        },
        "runtime": runtime,
        "memory": {
            "cpu_peak_rss_bytes": cpu_peak,
            "gpu_peak_memory_bytes": gpu_peak,
        },
        "gates": gates,
        "conclusion_guardrail": (
            "F2 is a no-GT geometry shadow and reports no AP. H0/HL/HLG "
            "capacity is evaluated only by the separately frozen F2 oracle."
        ),
    }
    output = output_dir.resolve() / OUTPUT_NAME
    digest = _atomic_create_json(output, receipt)
    print(f"Saved: {output} (sha256={digest}, pass={overall_pass})", flush=True)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge F2 paper100 shadow receipts")
    parser.add_argument("--shard", action="append", type=Path, dest="shards")
    parser.add_argument("--scene-list", type=Path, default=DEFAULT_SCENE_LIST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = _parser().parse_args()
    merge_f2(
        shard_paths=tuple(args.shards) if args.shards else DEFAULT_SHARDS,
        scene_list_path=args.scene_list,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
