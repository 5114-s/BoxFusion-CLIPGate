#!/usr/bin/env python3
"""Deterministically merge two create-only target-first SAM2 shadow roots.

The merge is no-GT and output-inert.  It authenticates both JSON/NPZ pairs,
requires disjoint scene subsets whose union is the official paper100 order,
then rebases shard-local scene and proposal-row indices into one global
ledger.  It refuses to overwrite its output root.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "boxfusion.scannet_target_first_sam2_masklift_paper100.v1"
OUTPUT_JSON = "TARGET_FIRST_SAM2_MASKLIFT_PAPER100.json"
OUTPUT_NPZ = "TARGET_FIRST_SAM2_MASKLIFT_PAPER100.npz"
DEFAULT_SCENE_LIST = REPOSITORY_ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "logs/scannet_target_first_sam2_masklift_paper100_score05"

EXPECTED_ARRAYS = {
    "target_group_names",
    "proposal_scene_index", "proposal_frame_id", "proposal_source_row",
    "proposal_source_instance_id", "proposal_semantic_id",
    "proposal_target_group_index", "proposal_score", "proposal_prompt_box_xyxy",
    "proposal_raw_center_world", "proposal_raw_quaternion_wxyz",
    "proposal_raw_extent_xyz", "proposal_lift_accepted",
    "proposal_abstention_code", "proposal_predicted_iou",
    "proposal_retained_point_count", "proposal_points_sha256",
    "proposal_lift_center_world", "proposal_lift_extent_xyz",
    "track_scene_index", "track_target_group_index", "track_semantic_id",
    "track_group_track_id", "track_confirmation_frame_id",
    "track_evidence_global_rows", "track_fused_point_offsets",
    "track_fused_points_world", "track_fused_aabb_corners",
    "track_fused_obb_corners", "track_pre_novelty_pass",
    "track_native_novelty_pass", "track_accepted_shadow",
}


class SAM2MaskLiftMergeError(RuntimeError):
    """A shard, ledger, index, or output contract differed."""


@dataclass(frozen=True)
class Shard:
    root: Path
    json_path: Path
    npz_path: Path
    json_sha256: str
    npz_sha256: str
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]
    scene_order: tuple[str, ...]
    scene_nodes: dict[str, dict[str, Any]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise SAM2MaskLiftMergeError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve()


def _scene_list(path: Path, expected: int) -> tuple[str, ...]:
    source = _regular(path, "official scene list")
    scenes = tuple(
        line.strip() for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(scenes) != expected or len(set(scenes)) != len(scenes):
        raise SAM2MaskLiftMergeError(f"expected {expected} unique official scenes")
    return scenes


def _load_shard(root: Path) -> Shard:
    if root.is_symlink() or not root.is_dir():
        raise SAM2MaskLiftMergeError(f"shard root must be a non-symlink directory: {root}")
    root = root.resolve()
    json_path = _regular(root / OUTPUT_JSON, "SAM2 shard JSON")
    npz_path = _regular(root / OUTPUT_NPZ, "SAM2 shard NPZ")
    json_sha, npz_sha = _sha256(json_path), _sha256(npz_path)
    try:
        manifest = json.loads(
            json_path.read_text(encoding="ascii"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise SAM2MaskLiftMergeError(f"invalid shard JSON: {json_path}") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise SAM2MaskLiftMergeError("shard schema differs")
    required_false = {
        "birth", "native_mutation_applied", "gt_access", "evaluator_access",
        "annotation_input_surface", "annotation_path_argument", "training",
        "target_dataset_training", "online_learning",
    }
    if manifest.get("mode") != "shadow" or manifest.get("output_inert") is not True:
        raise SAM2MaskLiftMergeError("shard is not an output-inert shadow")
    if any(manifest.get(key) is not False for key in required_false):
        raise SAM2MaskLiftMergeError("shard declares a forbidden capability")
    if manifest.get("npz_file") != OUTPUT_NPZ:
        raise SAM2MaskLiftMergeError("shard NPZ basename differs")
    declarations = manifest.get("npz_arrays")
    if not isinstance(declarations, dict) or set(declarations) != EXPECTED_ARRAYS:
        raise SAM2MaskLiftMergeError("shard NPZ declaration set differs")
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            if set(archive.files) != EXPECTED_ARRAYS:
                raise SAM2MaskLiftMergeError("shard NPZ array set differs")
            arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    except SAM2MaskLiftMergeError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise SAM2MaskLiftMergeError(f"invalid shard NPZ: {npz_path}") from error
    for name, array in arrays.items():
        declaration = declarations[name]
        if (
            not isinstance(declaration, dict)
            or array.dtype.hasobject
            or declaration.get("dtype") != array.dtype.str
            or declaration.get("shape") != list(array.shape)
            or declaration.get("sha256") != _hash_array(array)
        ):
            raise SAM2MaskLiftMergeError(f"shard NPZ metadata differs: {name}")
    scene_order = manifest.get("scene_order")
    if (
        not isinstance(scene_order, list)
        or any(not isinstance(scene, str) for scene in scene_order)
        or len(set(scene_order)) != len(scene_order)
        or manifest.get("scene_count") != len(scene_order)
    ):
        raise SAM2MaskLiftMergeError("shard scene order/count differs")
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != len(scene_order):
        raise SAM2MaskLiftMergeError("shard canonical scenes differ")
    scene_nodes: dict[str, dict[str, Any]] = {}
    for expected_scene, node in zip(scene_order, scenes):
        if not isinstance(node, dict) or node.get("scene_id") != expected_scene:
            raise SAM2MaskLiftMergeError("shard canonical scene identity differs")
        if not isinstance(node.get("receipts"), list):
            raise SAM2MaskLiftMergeError("shard scene receipts differ")
        scene_nodes[expected_scene] = node
    if set(manifest.get("inputs", {})) != set(scene_order) or set(manifest.get("scene_summaries", {})) != set(scene_order):
        raise SAM2MaskLiftMergeError("shard per-scene ledgers differ")
    # Sharded runs preserve official paper100 scene indices (for example
    # 50..99), so exact range validation is deferred until the official list
    # is available in the merge entry point.
    shard_indices = set(int(value) for value in arrays["proposal_scene_index"])
    shard_indices.update(int(value) for value in arrays["track_scene_index"])
    if len(shard_indices) > len(scene_order) or any(value < 0 for value in shard_indices):
        raise SAM2MaskLiftMergeError("shard scene-index census differs")
    _validate_array_shapes(
        arrays,
        len(scene_order),
        allowed_scene_indices=shard_indices,
    )
    if _sha256(json_path) != json_sha or _sha256(npz_path) != npz_sha:
        raise SAM2MaskLiftMergeError("shard changed while being read")
    return Shard(root, json_path, npz_path, json_sha, npz_sha, manifest, arrays, tuple(scene_order), scene_nodes)


def _validate_array_shapes(
    arrays: Mapping[str, np.ndarray],
    scene_count: int,
    *,
    allowed_scene_indices: set[int] | None = None,
) -> None:
    proposal_count = len(arrays["proposal_scene_index"])
    track_count = len(arrays["track_scene_index"])
    for name, array in arrays.items():
        if name.startswith("proposal_") and len(array) != proposal_count:
            raise SAM2MaskLiftMergeError(f"proposal array length differs: {name}")
        if name.startswith("track_") and name not in {
            "track_fused_point_offsets", "track_fused_points_world"
        } and len(array) != track_count:
            raise SAM2MaskLiftMergeError(f"track array length differs: {name}")
    offsets = arrays["track_fused_point_offsets"]
    if (
        offsets.shape != (track_count + 1,)
        or offsets.dtype.kind not in "iu"
        or int(offsets[0]) != 0
        or np.any(np.diff(offsets) < 0)
        or int(offsets[-1]) != len(arrays["track_fused_points_world"])
    ):
        raise SAM2MaskLiftMergeError("track fused-point offsets differ")
    proposal_scenes = arrays["proposal_scene_index"]
    track_scenes = arrays["track_scene_index"]
    if allowed_scene_indices is None:
        allowed_scene_indices = set(range(scene_count))
    if (
        proposal_scenes.dtype.kind not in "iu"
        or track_scenes.dtype.kind not in "iu"
        or any(int(value) not in allowed_scene_indices for value in proposal_scenes)
        or any(int(value) not in allowed_scene_indices for value in track_scenes)
    ):
        raise SAM2MaskLiftMergeError("shard-local scene indices differ")
    evidence = arrays["track_evidence_global_rows"]
    if evidence.shape != (track_count, 3) or evidence.dtype.kind not in "iu":
        raise SAM2MaskLiftMergeError("track evidence row shape differs")
    if track_count and np.any((evidence < 0) | (evidence >= proposal_count)):
        raise SAM2MaskLiftMergeError("track evidence row is outside shard proposal ledger")
    for track_index, rows in enumerate(evidence):
        if not np.all(proposal_scenes[rows] == track_scenes[track_index]):
            raise SAM2MaskLiftMergeError("track evidence crosses scenes")


def _same_metadata(shards: Sequence[Shard], key: str) -> Any:
    values = [shard.manifest.get(key) for shard in shards]
    if any(value != values[0] for value in values[1:]):
        raise SAM2MaskLiftMergeError(f"shard frozen metadata differs: {key}")
    return values[0]


def _deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("xb") as raw:
        with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name in sorted(arrays):
                payload = io.BytesIO()
                np.lib.format.write_array(payload, np.ascontiguousarray(arrays[name]), allow_pickle=False)
                info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, payload.getvalue(), compresslevel=9)
        raw.flush()
        os.fsync(raw.fileno())


def _publish(output_root: Path, arrays: Mapping[str, np.ndarray], manifest: Mapping[str, Any]) -> None:
    output = output_root.resolve()
    if output_root.is_symlink() or output.exists():
        raise SAM2MaskLiftMergeError(f"refusing to overwrite output root: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.merge-stage-", dir=output.parent))
    try:
        _deterministic_npz(stage / OUTPUT_NPZ, arrays)
        payload = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
        with (stage / OUTPUT_JSON).open("x", encoding="ascii") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(stage, output)
        stage = Path()
    finally:
        if stage != Path() and stage.exists():
            shutil.rmtree(stage)


def merge_target_first_sam2_masklift_paper100(
    *,
    shard_roots: Sequence[Path],
    scene_list: Path,
    output_root: Path,
    expected_scene_count: int = 100,
) -> dict[str, Any]:
    if len(shard_roots) != 2:
        raise SAM2MaskLiftMergeError("exactly two shard roots are required")
    if Path(shard_roots[0]).resolve() == Path(shard_roots[1]).resolve():
        raise SAM2MaskLiftMergeError("shard roots must be distinct")
    official = _scene_list(scene_list, expected_scene_count)
    shards = tuple(_load_shard(Path(root)) for root in shard_roots)
    owner: dict[str, tuple[Shard, int]] = {}
    for shard in shards:
        for local_index, scene in enumerate(shard.scene_order):
            if scene not in official:
                raise SAM2MaskLiftMergeError(f"shard scene is outside paper100: {scene}")
            if scene in owner:
                raise SAM2MaskLiftMergeError(f"shard scene overlap: {scene}")
            owner[scene] = (shard, local_index)
    if set(owner) != set(official):
        missing = sorted(set(official) - set(owner))
        raise SAM2MaskLiftMergeError(f"shards do not cover paper100; missing={missing[:3]}")
    official_index = {scene: index for index, scene in enumerate(official)}
    shard_scene_indices: dict[Path, dict[str, int]] = {}
    for shard in shards:
        observed = set(int(value) for value in shard.arrays["proposal_scene_index"])
        observed.update(int(value) for value in shard.arrays["track_scene_index"])
        official_mapping = {
            scene: official_index[scene] for scene in shard.scene_order
        }
        local_mapping = {
            scene: index for index, scene in enumerate(shard.scene_order)
        }
        official_allowed = set(official_mapping.values())
        local_allowed = set(local_mapping.values())
        if observed <= official_allowed:
            mapping = official_mapping
        elif observed <= local_allowed:
            # Compatibility with independently produced create-only shards
            # that use a local 0..N-1 scene ledger.
            mapping = local_mapping
        else:
            raise SAM2MaskLiftMergeError("shard scene indices match neither official nor local order")
        shard_scene_indices[shard.root] = mapping
        _validate_array_shapes(
            shard.arrays,
            len(official),
            allowed_scene_indices=set(mapping.values()),
        )
    groups = shards[0].arrays["target_group_names"]
    if not np.array_equal(groups, shards[1].arrays["target_group_names"]):
        raise SAM2MaskLiftMergeError("target group names differ across shards")

    proposal_names = sorted(name for name in EXPECTED_ARRAYS if name.startswith("proposal_") and name != "proposal_scene_index")
    track_names = sorted(name for name in EXPECTED_ARRAYS if name.startswith("track_") and name not in {
        "track_scene_index", "track_evidence_global_rows", "track_fused_point_offsets", "track_fused_points_world"
    })
    proposal_parts: dict[str, list[np.ndarray]] = {name: [] for name in proposal_names}
    proposal_scene_parts: list[np.ndarray] = []
    track_parts: dict[str, list[np.ndarray]] = {name: [] for name in track_names}
    track_scene_parts: list[np.ndarray] = []
    evidence_parts: list[np.ndarray] = []
    fused_blocks: list[np.ndarray] = []
    old_to_new: dict[tuple[Path, int], int] = {}
    scene_track_indices: dict[str, np.ndarray] = {}
    scene_proposal_indices: dict[str, np.ndarray] = {}
    new_proposal_count = 0

    for global_scene_index, scene in enumerate(official):
        shard, _local_scene_ordinal = owner[scene]
        source_scene_index = shard_scene_indices[shard.root][scene]
        pidx = np.flatnonzero(shard.arrays["proposal_scene_index"] == source_scene_index)
        tidx = np.flatnonzero(shard.arrays["track_scene_index"] == source_scene_index)
        scene_proposal_indices[scene], scene_track_indices[scene] = pidx, tidx
        for old_index in pidx.tolist():
            old_to_new[(shard.root, old_index)] = new_proposal_count
            new_proposal_count += 1
        proposal_scene_parts.append(np.full(len(pidx), global_scene_index, dtype=np.int16))
        for name in proposal_names:
            proposal_parts[name].append(shard.arrays[name][pidx])
        track_scene_parts.append(np.full(len(tidx), global_scene_index, dtype=np.int16))
        for name in track_names:
            track_parts[name].append(shard.arrays[name][tidx])
        old_evidence = shard.arrays["track_evidence_global_rows"][tidx]
        mapped = np.asarray(
            [[old_to_new[(shard.root, int(row))] for row in triple] for triple in old_evidence],
            dtype=np.int64,
        ).reshape(-1, 3)
        evidence_parts.append(mapped)
        offsets = shard.arrays["track_fused_point_offsets"]
        points = shard.arrays["track_fused_points_world"]
        for track_index in tidx.tolist():
            fused_blocks.append(points[int(offsets[track_index]) : int(offsets[track_index + 1])])

    def concatenate(parts: Sequence[np.ndarray], template: np.ndarray) -> np.ndarray:
        return np.concatenate(parts, axis=0) if parts else np.empty((0, *template.shape[1:]), dtype=template.dtype)

    arrays: dict[str, np.ndarray] = {
        "target_group_names": np.ascontiguousarray(groups),
        "proposal_scene_index": concatenate(proposal_scene_parts, shards[0].arrays["proposal_scene_index"]),
        "track_scene_index": concatenate(track_scene_parts, shards[0].arrays["track_scene_index"]),
        "track_evidence_global_rows": concatenate(evidence_parts, shards[0].arrays["track_evidence_global_rows"]),
    }
    arrays.update({name: concatenate(parts, shards[0].arrays[name]) for name, parts in proposal_parts.items()})
    arrays.update({name: concatenate(parts, shards[0].arrays[name]) for name, parts in track_parts.items()})
    fused_offsets = [0]
    for block in fused_blocks:
        fused_offsets.append(fused_offsets[-1] + len(block))
    arrays["track_fused_point_offsets"] = np.asarray(fused_offsets, dtype=np.int64)
    arrays["track_fused_points_world"] = (
        np.concatenate(fused_blocks, axis=0).astype(np.float32, copy=False)
        if fused_offsets[-1] else np.empty((0, 3), dtype=np.float32)
    )
    _validate_array_shapes(arrays, len(official))

    scenes_json: list[dict[str, Any]] = []
    tracks_json: list[dict[str, Any]] = []
    inputs: dict[str, Any] = {}
    summaries: dict[str, Any] = {}
    track_cursor = 0
    for global_scene_index, scene in enumerate(official):
        shard, _local_scene_ordinal = owner[scene]
        source_scene_index = shard_scene_indices[shard.root][scene]
        tidx = scene_track_indices[scene]
        receipts = []
        source_receipts = shard.scene_nodes[scene]["receipts"]
        if len(source_receipts) != len(tidx):
            raise SAM2MaskLiftMergeError(f"JSON/NPZ receipt count differs: {scene}")
        for local_ordinal, (receipt, old_track_index) in enumerate(zip(source_receipts, tidx.tolist())):
            if not isinstance(receipt, dict):
                raise SAM2MaskLiftMergeError(f"invalid receipt object: {scene}")
            row = dict(receipt)
            mapped = arrays["track_evidence_global_rows"][track_cursor + local_ordinal].tolist()
            old_rows = shard.arrays["track_evidence_global_rows"][old_track_index]
            expected_sources = shard.arrays["proposal_source_row"][old_rows].astype(np.int64).tolist()
            if row.get("evidence_source_rows") != expected_sources:
                raise SAM2MaskLiftMergeError(f"receipt evidence source rows differ: {scene}")
            if row.get("track_id") != int(shard.arrays["track_group_track_id"][old_track_index]):
                raise SAM2MaskLiftMergeError(f"receipt track ID differs: {scene}")
            row["evidence_global_rows"] = mapped
            receipts.append(row)
        scenes_json.append({"scene_id": scene, "receipts": receipts})
        source_tracks = shard.manifest.get("tracks")
        if not isinstance(source_tracks, list):
            raise SAM2MaskLiftMergeError("shard track ledger is absent")
        for local_ordinal, old_track_index in enumerate(tidx.tolist()):
            if old_track_index >= len(source_tracks) or not isinstance(source_tracks[old_track_index], dict):
                raise SAM2MaskLiftMergeError("shard track ledger order differs")
            row = dict(source_tracks[old_track_index])
            if row.get("scene_id") != scene or row.get("scene_index") != source_scene_index:
                raise SAM2MaskLiftMergeError("track scene identity differs")
            row["scene_index"] = global_scene_index
            row["evidence_global_rows"] = arrays["track_evidence_global_rows"][track_cursor + local_ordinal].tolist()
            tracks_json.append(row)
        track_cursor += len(tidx)
        inputs[scene] = shard.manifest["inputs"][scene]
        summaries[scene] = shard.manifest["scene_summaries"][scene]

    runtime_numeric = {
        "measured_frame_count", "provider_sum_seconds", "provider_mean_ms",
        "provider_p50_ms", "provider_p95_ms", "incremental_total_mean_ms",
        "incremental_total_p50_ms", "incremental_total_p95_ms",
        "incremental_runtime_gate_ms", "incremental_runtime_gate_pass",
    }
    runtime_metadata = [
        {
            key: value for key, value in shard.manifest["runtime"].items()
            if key not in runtime_numeric and key != "sam2_device"
        }
        for shard in shards
    ]
    if runtime_metadata[0] != runtime_metadata[1]:
        raise SAM2MaskLiftMergeError("shard runtime/model metadata differs")
    frame_counts = [int(shard.manifest["runtime"]["measured_frame_count"]) for shard in shards]
    provider_seconds = [float(shard.manifest["runtime"]["provider_sum_seconds"]) for shard in shards]
    incremental_means = [float(shard.manifest["runtime"]["incremental_total_mean_ms"]) for shard in shards]
    if any(count < 0 for count in frame_counts) or any(not math.isfinite(value) for value in provider_seconds + incremental_means):
        raise SAM2MaskLiftMergeError("shard runtime aggregate differs")
    runtime_gate_values = [
        float(shard.manifest["runtime"]["incremental_runtime_gate_ms"])
        for shard in shards
    ]
    if (
        any(not math.isfinite(value) for value in runtime_gate_values)
        or any(value != runtime_gate_values[0] for value in runtime_gate_values[1:])
    ):
        raise SAM2MaskLiftMergeError("shard runtime gate differs")
    total_frames = sum(frame_counts)
    runtime = {
        **runtime_metadata[0],
        "aggregation": "exact_counts_sums_and_weighted_means;quantiles_retained_per_shard_only",
        "measured_frame_count": total_frames,
        "provider_sum_seconds": sum(provider_seconds),
        "provider_mean_ms": (1000.0 * sum(provider_seconds) / total_frames) if total_frames else 0.0,
        "incremental_total_mean_ms": (
            sum(count * mean for count, mean in zip(frame_counts, incremental_means)) / total_frames
            if total_frames else 0.0
        ),
        "incremental_runtime_gate_ms": runtime_gate_values[0],
        "incremental_runtime_gate_pass": all(bool(shard.manifest["runtime"].get("incremental_runtime_gate_pass")) for shard in shards),
        "sam2_devices": [shard.manifest["runtime"].get("sam2_device") for shard in shards],
        "shards": [shard.manifest["runtime"] for shard in shards],
    }

    common_keys = (
        "top_k_per_frame", "raw_min_score", "selection_source", "routing_policy",
        "coordinate_frame", "checkpoint", "receipt_scene_ledger", "scene_list",
        "runner_source",
        "past_only_tracking", "past_only_confirmation", "native_clip_unchanged",
        "external_pretraining_frozen", "exact_raw_to_owl_key", "target_alias_matching",
        "old_receipt_membership_consumed", "old_receipt_decisions_consumed",
    )
    common = {key: _same_metadata(shards, key) for key in common_keys}
    decisions = [str(row.get("decision")) for row in tracks_json]
    manifest: dict[str, Any] = {
        "schema": SCHEMA, "mode": "shadow", "scene_count": len(official),
        "scene_order": list(official), "output_inert": True, "birth": False,
        "native_mutation_applied": False, "gt_access": False,
        "evaluator_access": False, "annotation_input_surface": False,
        "annotation_path_argument": False, "training": False,
        "target_dataset_training": False, "online_learning": False,
        **common,
        "target_prompt_count": len(arrays["proposal_scene_index"]),
        "target_prompt_frame_count": len(set(zip(arrays["proposal_scene_index"].tolist(), arrays["proposal_frame_id"].tolist()))),
        "inputs": inputs, "scenes": scenes_json, "scene_summaries": summaries,
        "lifted_row_count": len(arrays["proposal_scene_index"]),
        "accepted_lifted_row_count": int(np.count_nonzero(arrays["proposal_lift_accepted"])),
        "receipt_count": len(arrays["track_scene_index"]),
        "pre_novelty_pass_count": int(np.count_nonzero(arrays["track_pre_novelty_pass"])),
        "accepted_shadow_count": int(np.count_nonzero(arrays["track_accepted_shadow"])),
        "decision_counts": {reason: decisions.count(reason) for reason in sorted(set(decisions))},
        "tracks": tracks_json, "runtime": runtime, "npz_file": OUTPUT_NPZ,
        "npz_arrays": {name: {"dtype": array.dtype.str, "shape": list(array.shape), "sha256": _hash_array(array)} for name, array in arrays.items()},
        "merge": {
            "create_only": True, "gt_access": False, "evaluator_access": False,
            "scene_index_rebased_to_official_order": True,
            "evidence_global_rows_rebased": True, "fused_point_offsets_rebuilt": True,
            "source": {"path": os.fspath(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
            "shards": [
                {"root": os.fspath(shard.root), "json_sha256": shard.json_sha256, "npz_sha256": shard.npz_sha256, "scene_order": list(shard.scene_order)}
                for shard in shards
            ],
        },
        "conclusion_guardrail": "Merged no-GT shadow only; AP requires the separately frozen active materializer.",
    }
    _publish(output_root, arrays, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", action="append", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, default=DEFAULT_SCENE_LIST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expected-scene-count", type=int, default=100)
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = merge_target_first_sam2_masklift_paper100(
        shard_roots=args.shard_root, scene_list=args.scene_list,
        output_root=args.output_root, expected_scene_count=args.expected_scene_count,
    )
    print(json.dumps({
        "schema": manifest["schema"], "scene_count": manifest["scene_count"],
        "target_prompt_count": manifest["target_prompt_count"],
        "receipt_count": manifest["receipt_count"],
        "output_root": os.fspath(args.output_root.resolve()),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
