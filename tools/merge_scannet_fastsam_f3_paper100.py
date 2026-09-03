#!/usr/bin/env python3
"""Validate F3 paper100 shards and publish one create-only shadow receipt."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "boxfusion.scannet_fastsam_f3_openbox.merge.v1"
SCENE_SCHEMA = "boxfusion.scannet_fastsam_f3_openbox.scene.v1"
SHARD_SCHEMA = "boxfusion.scannet_fastsam_f3_openbox.shard.v1"
PROTOCOL_ID = "F3-FASTSAM-OPENBOX-PROJECTION-SHADOW-PAPER100"
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
MAX_F3_MEAN_MS = 25.0
MAX_F3_P95_MS = 40.0
MAX_F3_AMORTIZED_MS_PER_SOURCE_FRAME = 1.0
MAX_COMPOSED_P95_MS = 250.0
MAX_COMPOSED_MS_EXCLUSIVE = 833.33
MAX_COMPOSED_AMORTIZED_MS_PER_SOURCE_FRAME = 10.0
SOURCE_FRAME_STRIDE = 25.0
MAX_GPU_PEAK_BYTES = 4 * 1024**3
OUTPUT_NAME = "F3_FASTSAM_OPENBOX_PAPER100.json"
DEFAULT_SHARDS = (
    REPOSITORY_ROOT
    / "logs/scannet_fastsam_f3_openbox_paper100_score05/shards/shard-000-of-002.json",
    REPOSITORY_ROOT
    / "logs/scannet_fastsam_f3_openbox_paper100_score05/shards/shard-001-of-002.json",
)
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT / "logs/scannet_fastsam_f3_openbox_paper100_score05/final"
)


class F3MergeError(RuntimeError):
    """Raised when F3 shard, scene, provenance, or causality differs."""


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
        raise F3MergeError(f"{label} must be a regular non-symlink file: {path}")
    result = path.resolve()
    if suffix is not None and result.suffix.lower() != suffix:
        raise F3MergeError(f"{label} must be a {suffix} file: {result}")
    if result.suffix.lower() in {".pkl", ".pickle"}:
        raise F3MergeError(f"prediction pickle input is forbidden: {result}")
    return result


def _read_json(path: Path, label: str) -> tuple[Path, dict[str, Any]]:
    source = _regular_file(path, label, ".json")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise F3MergeError(f"invalid {label}: {source}") from error
    if not isinstance(value, dict):
        raise F3MergeError(f"{label} must contain one JSON object: {source}")
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
            raise F3MergeError(f"refusing to overwrite output: {path}") from error
        return _sha256(path)
    finally:
        temporary.unlink(missing_ok=True)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise F3MergeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise F3MergeError(f"{label} must be finite and non-negative")
    return result


def _xyz(value: object, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise F3MergeError(f"{label} must be a finite xyz vector") from error
    if result.shape != (3,) or not np.isfinite(result).all():
        raise F3MergeError(f"{label} must be a finite xyz vector")
    return result


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _rehash_reference(value: object, label: str) -> Path:
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("path"), str)
        or not isinstance(value.get("sha256"), str)
    ):
        raise F3MergeError(f"{label} seal is absent")
    path = _regular_file(Path(value["path"]), label)
    if _sha256(path) != value["sha256"]:
        raise F3MergeError(f"{label} rehash differs")
    return path


def _validate_hypothesis(value: object, label: str) -> None:
    if not isinstance(value, Mapping):
        raise F3MergeError(f"{label} hypothesis is absent")
    valid = value.get("valid") is True
    folds = value.get("fold_ious")
    count = value.get("valid_fold_count")
    if (
        not isinstance(folds, list)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count != len(folds)
    ):
        raise F3MergeError(f"{label} fold ledger differs")
    for index, overlap in enumerate(folds):
        value_number = _number(overlap, f"{label}.fold_ious[{index}]")
        if value_number > 1.0:
            raise F3MergeError(f"{label} fold IoU exceeds one")
    score = value.get("score")
    if score is not None and _number(score, f"{label}.score") > 1.0:
        raise F3MergeError(f"{label} score exceeds one")
    if valid:
        q02 = _xyz(value.get("q02"), f"{label}.q02")
        q98 = _xyz(value.get("q98"), f"{label}.q98")
        center = _xyz(value.get("center"), f"{label}.center")
        extent = _xyz(value.get("extent"), f"{label}.extent")
        if (
            np.any(q98 <= q02)
            or score is None
            or not np.allclose(center, (q02 + q98) * 0.5, rtol=0.0, atol=1e-12)
            or not np.allclose(extent, q98 - q02, rtol=0.0, atol=1e-12)
        ):
            raise F3MergeError(f"{label} valid geometry differs")
    elif any(value.get(key) is not None for key in ("q02", "q98", "center", "extent")):
        raise F3MergeError(f"{label} invalid hypothesis carries geometry")


def _validate_track(
    track: Mapping[str, Any],
    scene: str,
    source_to_track: Mapping[str, int],
    source_to_frame: Mapping[str, int],
) -> tuple[int, list[str]]:
    track_id = track.get("track_id")
    sources = track.get("source_ids")
    frames = track.get("frame_ids")
    if (
        isinstance(track_id, bool)
        or not isinstance(track_id, int)
        or track_id < 0
        or not isinstance(sources, list)
        or not isinstance(frames, list)
        or len(sources) != len(frames)
        or track.get("observation_count") != len(sources)
        or len(set(sources)) != len(sources)
        or any(source_to_track.get(source) != track_id for source in sources)
        or any(source_to_frame.get(source) != frame for source, frame in zip(sources, frames))
    ):
        raise F3MergeError(f"F3 track lineage differs: {scene}/{track_id}")
    retained_sources = track.get("retained_source_ids")
    retained_frames = track.get("retained_frame_ids")
    if (
        not isinstance(retained_sources, list)
        or not isinstance(retained_frames, list)
        or len(retained_sources) != len(retained_frames)
        or len(retained_sources) > 5
        or track.get("retained_observation_count") != len(retained_sources)
        or retained_sources != sources[-5:]
        or retained_frames != frames[-5:]
        or type(track.get("confirmed")) is not bool
        or track.get("confirmed") is not (len(retained_sources) >= 3)
    ):
        raise F3MergeError(f"F3 bounded track evidence differs: {scene}/{track_id}")
    hypotheses = track.get("hypotheses")
    if not isinstance(hypotheses, Mapping) or set(hypotheses) != {"B", "C"}:
        raise F3MergeError(f"F3 B/C ledger differs: {scene}/{track_id}")
    _validate_hypothesis(hypotheses["B"], f"{scene}/{track_id}.B")
    _validate_hypothesis(hypotheses["C"], f"{scene}/{track_id}.C")
    selector = track.get("selector")
    if not isinstance(selector, Mapping):
        raise F3MergeError(f"F3 selector is absent: {scene}/{track_id}")
    chosen = selector.get("chosen")
    if chosen not in (None, "B", "C"):
        raise F3MergeError(f"F3 selector choice differs: {scene}/{track_id}")
    if chosen is None:
        if any(
            selector.get(key) is not None
            for key in ("q02", "q98", "center", "extent", "score")
        ):
            raise F3MergeError(f"F3 abstaining selector carries geometry: {scene}/{track_id}")
    else:
        selected = hypotheses[chosen]
        if (
            selected.get("valid") is not True
            or selector.get("q02") != selected.get("q02")
            or selector.get("q98") != selected.get("q98")
            or selector.get("center") != selected.get("center")
            or selector.get("extent") != selected.get("extent")
            or selector.get("score") != selected.get("score")
        ):
            raise F3MergeError(f"F3 selector does not copy chosen geometry: {scene}/{track_id}")
    return track_id, [str(source) for source in sources]


def _gate(actual: float | int, comparator: str, threshold: float | int) -> dict[str, Any]:
    if comparator == "<=":
        passed = actual <= threshold
    elif comparator == "<":
        passed = actual < threshold
    elif comparator == "==":
        passed = actual == threshold
    else:  # pragma: no cover - private constructor
        raise AssertionError(comparator)
    return {
        "actual": actual,
        "threshold": threshold,
        "comparator": comparator,
        "passed": bool(passed),
    }


def merge_f3(
    *,
    shard_paths: Sequence[Path],
    output_dir: Path,
    _expected_scene_count: int = EXPECTED_SCENES,
) -> dict[str, Any]:
    """Validate all F3 shards/scenes and seal the oracle-facing receipt."""

    production = _expected_scene_count == EXPECTED_SCENES
    expected_shards = EXPECTED_SHARDS if production else len(shard_paths)
    if not shard_paths or len(shard_paths) != expected_shards:
        raise F3MergeError(f"expected {expected_shards} F3 shard manifests")
    shards: dict[int, tuple[Path, dict[str, Any]]] = {}
    for raw in shard_paths:
        path, value = _read_json(raw, "F3 shard manifest")
        shard = value.get("shard")
        if (
            value.get("schema") != SHARD_SCHEMA
            or value.get("protocol_id") != PROTOCOL_ID
            or value.get("mode") != "shadow"
            or value.get("complete") is not True
            or not isinstance(shard, Mapping)
            or isinstance(shard.get("index"), bool)
            or not isinstance(shard.get("index"), int)
            or shard.get("count") != expected_shards
            or shard["index"] in shards
        ):
            raise F3MergeError(f"invalid F3 shard contract: {path}")
        shards[shard["index"]] = (path, value)
    if set(shards) != set(range(expected_shards)):
        raise F3MergeError("F3 shard indices are incomplete")
    signatures = {value.get("run_signature_sha256") for _path, value in shards.values()}
    scene_orders = {
        tuple(value.get("paper100_scene_order", ())) for _path, value in shards.values()
    }
    f2_receipts = {
        _canonical_json_sha256(value.get("f2_receipt")) for _path, value in shards.values()
    }
    f2_oracles = {
        _canonical_json_sha256(value.get("f2_oracle")) for _path, value in shards.values()
    }
    source_receipts = {
        _canonical_json_sha256(value.get("sources_receipt"))
        for _path, value in shards.values()
    }
    if (
        len(signatures) != 1
        or len(scene_orders) != 1
        or len(f2_receipts) != 1
        or len(f2_oracles) != 1
        or len(source_receipts) != 1
    ):
        raise F3MergeError("F3 shard execution/provenance identities differ")
    run_signature = next(iter(signatures))
    scenes = next(iter(scene_orders))
    if len(scenes) != _expected_scene_count or len(set(scenes)) != len(scenes):
        raise F3MergeError("F3 scene universe differs")
    first_manifest = shards[0][1]
    f2_receipt = first_manifest.get("f2_receipt")
    f2_oracle = first_manifest.get("f2_oracle")
    if not isinstance(f2_receipt, Mapping) or not isinstance(f2_oracle, Mapping):
        raise F3MergeError("F3 upstream receipt seals are absent")
    _rehash_reference(f2_receipt, "sealed F2 merged receipt")
    _rehash_reference(f2_oracle, "sealed F2 oracle receipt")

    scene_rows: dict[int, tuple[Path, Mapping[str, Any], dict[str, Any]]] = {}
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
            or not isinstance(rows, list)
            or len(rows) != len(expected_order)
        ):
            raise F3MergeError(f"F3 shard coverage differs: {manifest_path}")
        if production and any(
            int(manifest.get("totals", {}).get(key, -1)) != value
            for key, value in EXPECTED_SHARD_COUNTS[shard_index].items()
        ):
            raise F3MergeError(f"F3 shard census differs: {manifest_path}")
        for scene_index, scene, row in zip(expected_indices, expected_order, rows):
            if (
                not isinstance(row, Mapping)
                or row.get("scene_id") != scene
                or row.get("scene_index") != scene_index
                or not isinstance(row.get("sidecar_path"), str)
                or not isinstance(row.get("sidecar_sha256"), str)
            ):
                raise F3MergeError(f"F3 shard scene row differs: {scene}")
            sidecar_path, sidecar = _read_json(
                Path(row["sidecar_path"]), f"F3 scene sidecar {scene}"
            )
            if _sha256(sidecar_path) != row["sidecar_sha256"]:
                raise F3MergeError(f"F3 scene sidecar rehash differs: {scene}")
            scene_rows[scene_index] = (sidecar_path, row, sidecar)
    if set(scene_rows) != set(range(len(scenes))):
        raise F3MergeError("F3 scene union is incomplete")

    totals: Counter[str] = Counter()
    all_sources: set[str] = set()
    f3_samples: list[float] = []
    composed_samples: list[float] = []
    output_scenes: list[dict[str, Any]] = []
    max_inherited_gpu = 0
    max_cpu_rss = 0
    integrity_pass = True
    prefix_pass = True
    query_pass = True
    partition_pass = True
    logical_pass = True
    for scene_index, scene in enumerate(scenes):
        sidecar_path, manifest_row, receipt = scene_rows[scene_index]
        if (
            receipt.get("schema") != SCENE_SCHEMA
            or receipt.get("protocol_id") != PROTOCOL_ID
            or receipt.get("complete") is not True
            or receipt.get("run_signature_sha256") != run_signature
            or receipt.get("scene_id") != scene
            or receipt.get("scene_index") != scene_index
            or receipt.get("contracts", {}).get("birth_enabled") is not False
            or receipt.get("contracts", {}).get("fastsam_rerun") is not False
            or receipt.get("contracts", {}).get("h0_only") is not True
        ):
            raise F3MergeError(f"F3 scene contract differs: {scene}")
        inputs = receipt.get("inputs")
        if not isinstance(inputs, Mapping):
            raise F3MergeError(f"F3 scene input seals are absent: {scene}")
        for name in ("f2_sidecar", "f2_evidence_npz", "f0_sidecar", "schedule", "intrinsic"):
            _rehash_reference(inputs.get(name), f"{scene} {name}")
        sealed_sources = inputs.get("sources")
        if sealed_sources != first_manifest.get("sources_receipt") or not isinstance(
            sealed_sources, Mapping
        ):
            raise F3MergeError(f"F3 source-code receipt differs: {scene}")
        for name, source_receipt in sealed_sources.items():
            _rehash_reference(source_receipt, f"{scene} frozen source {name}")
        if inputs.get("f2_receipt") != f2_receipt or inputs.get("f2_oracle") != f2_oracle:
            raise F3MergeError(f"F3 upstream receipt identity differs: {scene}")
        f2_scene_path, f2_scene = _read_json(
            Path(inputs["f2_sidecar"]["path"]), f"F2 scene replay source {scene}"
        )
        if (
            _sha256(f2_scene_path) != inputs["f2_sidecar"]["sha256"]
            or f2_scene.get("schema") != EXPECTED_F2_SCENE_SCHEMA
            or f2_scene.get("scene_id") != scene
            or f2_scene.get("scene_index") != scene_index
        ):
            raise F3MergeError(f"F3/F2 scene identity differs: {scene}")
        frames = receipt.get("frames")
        f2_frames = f2_scene.get("frames")
        if not isinstance(frames, list) or not isinstance(f2_frames, list) or len(frames) != len(f2_frames):
            raise F3MergeError(f"F3/F2 frame ledger differs: {scene}")
        source_to_track: dict[str, int] = {}
        source_to_frame: dict[str, int] = {}
        seen_track_ids: set[int] = set()
        created_track_ids: list[int] = []
        historical_max_created = -1
        scene_source_ids: list[str] = []
        successful_count = 0
        for ordinal, (frame, f2_frame) in enumerate(zip(frames, f2_frames)):
            expected_source_ids = [str(row["source_id"]) for row in f2_frame.get("sources", ())]
            if (
                not isinstance(frame, Mapping)
                or frame.get("ordinal") != ordinal
                or frame.get("frame_id") != f2_frame.get("frame_id")
                or bool(frame.get("successful")) != bool(f2_frame.get("successful"))
                or frame.get("source_ids") != expected_source_ids
            ):
                raise F3MergeError(f"F3 frame/source prefix differs: {scene}/{ordinal}")
            assignments = frame.get("assignments")
            if not isinstance(assignments, list) or [row.get("source_id") for row in assignments] != expected_source_ids:
                raise F3MergeError(f"F3 assignment order differs: {scene}/{ordinal}")
            for source_id, assignment in zip(expected_source_ids, assignments):
                track_id = assignment.get("track_id")
                if (
                    source_id in all_sources
                    or source_id in source_to_track
                    or isinstance(track_id, bool)
                    or not isinstance(track_id, int)
                    or track_id < 0
                    or assignment.get("action") not in {"matched", "created"}
                ):
                    raise F3MergeError(f"F3 source assignment differs: {source_id}")
                expected_create = track_id not in seen_track_ids
                if expected_create != (assignment.get("action") == "created"):
                    raise F3MergeError(f"F3 assignment action/history differs: {source_id}")
                if expected_create:
                    created_track_ids.append(track_id)
                    seen_track_ids.add(track_id)
                source_to_track[source_id] = track_id
                source_to_frame[source_id] = int(frame["frame_id"])
                all_sources.add(source_id)
                scene_source_ids.append(source_id)
            frame_created = sorted(
                assignment["track_id"]
                for assignment in assignments
                if assignment["action"] == "created"
            )
            if frame_created and (
                frame_created != list(range(frame_created[0], frame_created[-1] + 1))
                or frame_created[0] <= historical_max_created
            ):
                raise F3MergeError(f"F3 created track IDs differ: {scene}/{ordinal}")
            if frame_created:
                historical_max_created = frame_created[-1]
            max_access = frame.get("max_logical_accessed_ordinal")
            if (
                isinstance(max_access, bool)
                or not isinstance(max_access, int)
                or max_access > ordinal
                or frame.get("query_before_commit") is not True
                or frame.get("prefix_invariance") is not True
            ):
                raise F3MergeError(f"F3 causal frame receipt differs: {scene}/{ordinal}")
            f3_samples.append(_number(frame.get("f3_core_ms"), f"{scene}/{ordinal}.f3_core_ms"))
            composed = _number(
                frame.get("composed_complete_ms"), f"{scene}/{ordinal}.composed_complete_ms"
            )
            if frame.get("successful") is True and frame.get("inherited_warmup_excluded") is not True:
                composed_samples.append(composed)
            successful_count += int(frame.get("successful") is True)
        tracks = receipt.get("tracks")
        if not isinstance(tracks, list):
            raise F3MergeError(f"F3 track ledger is absent: {scene}")
        track_ids: list[int] = []
        flattened: list[str] = []
        for track in tracks:
            if not isinstance(track, Mapping):
                raise F3MergeError(f"F3 track row is invalid: {scene}")
            track_id, sources = _validate_track(track, scene, source_to_track, source_to_frame)
            track_ids.append(track_id)
            flattened.extend(sources)
        if (
            track_ids != sorted(track_ids)
            or len(track_ids) != len(set(track_ids))
            or len(flattened) != len(set(flattened))
            or set(flattened) != set(scene_source_ids)
            or len(flattened) != len(scene_source_ids)
            or set(track_ids) != set(range(len(track_ids)))
            or sorted(created_track_ids) != list(range(len(track_ids)))
        ):
            raise F3MergeError(f"F3 tracks do not partition sources: {scene}")
        counts = receipt.get("counts")
        expected_counts = {
            "keyframe_count": len(frames),
            "successful_frame_count": successful_count,
            "source_count": len(scene_source_ids),
            "identity_verified_source_count": len(scene_source_ids),
            "track_count": len(tracks),
            "confirmed_track_count": sum(bool(track.get("confirmed")) for track in tracks),
            "selected_track_count": sum(track.get("selector", {}).get("chosen") is not None for track in tracks),
        }
        if not isinstance(counts, Mapping) or any(counts.get(key) != value for key, value in expected_counts.items()):
            raise F3MergeError(f"F3 scene count summary differs: {scene}")
        causality = receipt.get("causality")
        if not isinstance(causality, Mapping) or causality.get("overall_pass") is not True:
            raise F3MergeError(f"F3 scene causality failed: {scene}")
        prefix_pass &= causality.get("prefix_invariance", {}).get("passed") is True
        query_pass &= causality.get("query_before_commit", {}).get("passed") is True
        partition_pass &= causality.get("one_source_one_track", {}).get("passed") is True
        logical_pass &= causality.get("maximum_logical_accessed_ordinal", {}).get("passed") is True
        runtime = receipt.get("runtime")
        if not isinstance(runtime, Mapping) or runtime.get("new_gpu_allocation_bytes") != 0:
            raise F3MergeError(f"F3 scene runtime/GPU contract differs: {scene}")
        max_inherited_gpu = max(
            max_inherited_gpu, int(runtime.get("inherited_gpu_peak_memory_bytes", 0))
        )
        max_cpu_rss = max(max_cpu_rss, int(runtime.get("cpu_peak_rss_bytes", 0)))
        totals.update(expected_counts)
        output_scenes.append(
            {
                "scene_id": scene,
                "scene_index": scene_index,
                "sidecar": {"path": os.fspath(sidecar_path), "sha256": _sha256(sidecar_path)},
                "counts": expected_counts,
            }
        )

    if production and (
        totals["keyframe_count"] != EXPECTED_KEYFRAMES
        or totals["successful_frame_count"] != EXPECTED_SUCCESSFUL_FRAMES
        or totals["source_count"] != EXPECTED_SOURCES
        or totals["identity_verified_source_count"] != EXPECTED_SOURCES
    ):
        raise F3MergeError("F3 final paper100 census differs")
    f3_distribution = _distribution(f3_samples)
    composed_distribution = _distribution(composed_samples)
    amortized_f3 = float(f3_distribution["mean"]) / SOURCE_FRAME_STRIDE
    amortized_composed = float(composed_distribution["mean"]) / SOURCE_FRAME_STRIDE
    runtime_gates = {
        "f3_incremental_mean_ms": _gate(f3_distribution["mean"], "<=", MAX_F3_MEAN_MS),
        "f3_incremental_p95_ms": _gate(f3_distribution["p95"], "<=", MAX_F3_P95_MS),
        "amortized_f3_ms_per_source_frame": _gate(
            amortized_f3, "<=", MAX_F3_AMORTIZED_MS_PER_SOURCE_FRAME
        ),
        "composed_complete_p95_ms": _gate(
            composed_distribution["p95"], "<=", MAX_COMPOSED_P95_MS
        ),
        "composed_complete_max_ms": _gate(
            composed_distribution["max"], "<", MAX_COMPOSED_MS_EXCLUSIVE
        ),
        "amortized_composed_complete_ms_per_source_frame": _gate(
            amortized_composed, "<=", MAX_COMPOSED_AMORTIZED_MS_PER_SOURCE_FRAME
        ),
        "new_gpu_allocation_bytes": _gate(0, "==", 0),
        "total_gpu_peak_memory_bytes": _gate(
            max_inherited_gpu, "<=", MAX_GPU_PEAK_BYTES
        ),
    }
    runtime_pass = all(value["passed"] for value in runtime_gates.values())
    expected_source_count = EXPECTED_SOURCES if production else totals["source_count"]
    expected_keyframes = EXPECTED_KEYFRAMES if production else totals["keyframe_count"]
    integrity = {
        "scene_coverage": _gate(len(scenes), "==", _expected_scene_count),
        "keyframe_coverage": _gate(totals["keyframe_count"], "==", expected_keyframes),
        "source_identity": _gate(totals["identity_verified_source_count"], "==", expected_source_count),
        "source_track_partition": _gate(len(all_sources), "==", expected_source_count),
    }
    integrity["overall_pass"] = integrity_pass and all(
        value["passed"] for value in integrity.values() if isinstance(value, Mapping)
    )
    causality = {
        "prefix_invariance": {"passed": bool(prefix_pass)},
        "query_before_commit": {"passed": bool(query_pass)},
        "one_source_one_track": {"passed": bool(partition_pass)},
        "maximum_logical_accessed_ordinal": {"passed": bool(logical_pass)},
    }
    causality["overall_pass"] = all(value["passed"] for value in causality.values())
    runtime = {
        "f3_core_ms": f3_distribution,
        "composed_complete_ms": composed_distribution,
        "amortized_f3_ms_per_source_frame": amortized_f3,
        "amortized_composed_ms_per_source_frame": amortized_composed,
        "gates": runtime_gates,
        "overall_pass": runtime_pass,
    }
    overall_pass = bool(
        integrity["overall_pass"] and causality["overall_pass"] and runtime_pass
    )
    receipt = {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "overall_pass": overall_pass,
        "run_signature_sha256": run_signature,
        "coverage": {
            "scene_order": list(scenes),
            "scene_count": len(scenes),
            "keyframe_count": totals["keyframe_count"],
            "successful_frame_count": totals["successful_frame_count"],
            "source_count": totals["source_count"],
            "identity_verified_source_count": totals["identity_verified_source_count"],
        },
        "contracts": {
            "shadow_only": True,
            "observer_only": True,
            "birth_enabled": False,
            "fastsam_rerun": False,
            "h0_only": True,
            "ground_truth_access": False,
            "prediction_access": False,
            "evaluator_access": False,
            "rgb_access": False,
            "depth_pixel_access": False,
            "native_output_mutation": False,
            "future_frame_logical_access": False,
            "hl_hlg_access": False,
            "training": False,
            "online_learning": False,
        },
        "inputs": {
            "f2_receipt": dict(f2_receipt),
            "f2_oracle": dict(f2_oracle),
            "shards": [
                {"path": os.fspath(path), "sha256": _sha256(path)}
                for path, _manifest in (shards[index] for index in sorted(shards))
            ],
        },
        "integrity": integrity,
        "causality": causality,
        "runtime": runtime,
        "memory": {
            "cpu_peak_rss_bytes": max_cpu_rss,
            "new_gpu_allocation_bytes": 0,
            "inherited_gpu_peak_memory_bytes": max_inherited_gpu,
        },
        "totals": {
            "scene_count": len(scenes),
            "keyframe_count": totals["keyframe_count"],
            "successful_frame_count": totals["successful_frame_count"],
            "source_count": totals["source_count"],
            "identity_verified_source_count": totals["identity_verified_source_count"],
            "track_count": totals["track_count"],
            "confirmed_track_count": totals["confirmed_track_count"],
            "selected_track_count": totals["selected_track_count"],
        },
        "scenes": output_scenes,
        "conclusion_guardrail": (
            "F3 is a no-GT observer-only projection shadow. B/C/fixed-selector "
            "capacity is measured only by the separately sealed oracle."
        ),
    }
    output = output_dir.resolve() / OUTPUT_NAME
    digest = _atomic_create_json(output, receipt)
    print(f"Saved: {output} (sha256={digest}, pass={overall_pass})", flush=True)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge F3 paper100 shadow receipts")
    parser.add_argument("--shard", action="append", type=Path, dest="shards")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = _parser().parse_args()
    merge_f3(
        shard_paths=tuple(args.shards) if args.shards else DEFAULT_SHARDS,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
