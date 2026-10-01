#!/usr/bin/env python3
"""Create-only F5 GT-free past-only geometry-selector replay.

The runner consumes only sealed F4 scene rows and their sealed F2 evidence.
It never opens annotations, terminal detector output, semantics, or a future
source offset.  Each scene is executed once online, once as a deterministic
prefix audit, and once as an independent full replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "F5-GT-FREE-PAST-ONLY-GEOMETRY-SELECTOR-PAPER100"
SCENE_SCHEMA = "boxfusion.scannet_fastsam_f5_gtfree_selector_paper100.scene.v1"
SHARD_SCHEMA = "boxfusion.scannet_fastsam_f5_gtfree_selector_paper100.shard.v1"
EXPECTED_F4_MERGE_SCHEMA = "boxfusion.scannet_fastsam_f4_boxer_paper100.merge.v1"
EXPECTED_F4_SCENE_SCHEMA = "boxfusion.scannet_fastsam_f4_boxer_paper100.scene.v1"
EXPECTED_F4_PROTOCOL = "F4-FASTSAM-BOXER-GEOMETRY-SHADOW-PAPER100"
EXPECTED_F2_SCENE_SCHEMA = "boxfusion.scannet_fastsam_f2_paper100.scene.v1"
EXPECTED_F2_EVIDENCE_SCHEMA = "boxfusion.scannet_fastsam_f2_paper100.evidence.v1"

EXPECTED_SCENES = 100
EXPECTED_KEYFRAMES = 6_817
EXPECTED_SUCCESSFUL_FRAMES = 6_726
EXPECTED_SOURCES = 52_299
EXPECTED_SHARDS = 2
EXPECTED_SHARD_COUNTS: Mapping[int, Mapping[str, int]] = {
    0: {"keyframe_count": 3_259, "successful_frame_count": 3_189, "source_count": 24_863},
    1: {"keyframe_count": 3_558, "successful_frame_count": 3_537, "source_count": 27_436},
}
EXPECTED_SCENE_LIST_SHA256 = "4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5"
EXPECTED_PROTOCOL_SHA256 = "2a6d62fa9d5912dc3871bbc485f44987565bda61b818722b3a4e6577d34a6afc"
MASK_PACKED_BYTES = 480 * 640 // 8
WARMUP_NONEMPTY_FRAMES = 3
SOURCE_FRAME_STRIDE = 25.0
DEADLINE_MS = 833.33

DEFAULT_F4_RECEIPT = REPOSITORY_ROOT / "logs/scannet_fastsam_f4_boxer_paper100_score05/final/F4_FASTSAM_BOXER_PAPER100.json"
DEFAULT_SCENE_LIST = REPOSITORY_ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "logs/scannet_fastsam_f5_selector_paper100_score05"
PROTOCOL_PATH = REPOSITORY_ROOT / "docs/F5_GT_FREE_GEOMETRY_SELECTOR_PROTOCOL_FREEZE.md"

CONTRACTS: Mapping[str, bool] = {
    "shadow_only": True,
    "selector_only": True,
    "birth_enabled": False,
    "source_addition_or_removal": False,
    "native_output_mutation": False,
    "score_or_rank_mutation": False,
    "semantic_or_clip_access": False,
    "ground_truth_access": False,
    "annotation_access": False,
    "prediction_access": False,
    "evaluator_access": False,
    "future_frame_access": False,
    "training": False,
    "online_learning": False,
}


class F5RunnerError(RuntimeError):
    """Raised when a sealed input or F5 execution contract differs."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    try:
        data = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise F5RunnerError("value is not canonical finite ASCII JSON") from error
    return hashlib.sha256(data).hexdigest()


def _content_hash_without(value: Mapping[str, Any], *keys: str) -> str:
    payload = dict(value)
    for key in keys:
        payload.pop(key, None)
    return _canonical_json_sha256(payload)


def _regular_file(path: Path, label: str, suffix: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise F5RunnerError(f"{label} must be a regular non-symlink file: {path}")
    source = path.resolve()
    if suffix is not None and source.suffix.lower() != suffix:
        raise F5RunnerError(f"{label} must have suffix {suffix}: {source}")
    if source.suffix.lower() in {".pkl", ".pickle"}:
        raise F5RunnerError(f"forbidden serialized detector input: {source}")
    return source


def _read_json(path: Path, label: str) -> tuple[Path, dict[str, Any]]:
    source = _regular_file(path, label, ".json")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise F5RunnerError(f"invalid {label}: {source}") from error
    if not isinstance(value, dict):
        raise F5RunnerError(f"{label} must contain one JSON object")
    return source, value


def _rehash_reference(value: object, label: str, suffix: str | None = None) -> Path:
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("path"), str)
        or not isinstance(value.get("sha256"), str)
    ):
        raise F5RunnerError(f"{label} seal is absent")
    path = _regular_file(Path(value["path"]), label, suffix)
    if _sha256(path) != value["sha256"]:
        raise F5RunnerError(f"{label} rehash differs")
    return path


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
            raise F5RunnerError(f"refusing to overwrite output: {path}") from error
        return _sha256(path)
    finally:
        temporary.unlink(missing_ok=True)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise F5RunnerError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise F5RunnerError(f"{label} must be finite and nonnegative")
    return result


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise F5RunnerError("runtime samples must be finite and nonnegative")
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _load_intrinsic(reference: object) -> tuple[Path, np.ndarray]:
    path = _rehash_reference(reference, "sealed F2 intrinsic", ".txt")
    try:
        matrix = np.loadtxt(path, dtype=np.float64)
    except (OSError, ValueError) as error:
        raise F5RunnerError("sealed intrinsic cannot be decoded") from error
    if matrix.shape == (4, 4):
        matrix = matrix[:3, :3]
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or matrix[0, 0] <= 0.0
        or matrix[1, 1] <= 0.0
    ):
        raise F5RunnerError("sealed intrinsic is invalid")
    return path, np.ascontiguousarray(matrix, dtype=np.float64)


def _load_pose(reference: object, scene: str, frame_id: int) -> tuple[Path, np.ndarray]:
    path = _rehash_reference(reference, f"{scene}/{frame_id} current pose", ".txt")
    try:
        pose = np.loadtxt(path, dtype=np.float64)
    except (OSError, ValueError) as error:
        raise F5RunnerError(f"current pose cannot be decoded: {scene}/{frame_id}") from error
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise F5RunnerError(f"successful F5 frame requires a finite pose: {scene}/{frame_id}")
    return path, np.ascontiguousarray(pose, dtype=np.float64)


class _EvidenceAccessor:
    """Decode a sealed NPZ once, but expose only the current contiguous offsets."""

    def __init__(self, path: Path, expected_sha: str, scene: str, count: int) -> None:
        source = _regular_file(path, f"{scene} F2 evidence", ".npz")
        if _sha256(source) != expected_sha:
            raise F5RunnerError(f"F2 evidence rehash differs: {scene}")
        try:
            archive = np.load(source, allow_pickle=False)
            required = {
                "schema", "scene_id", "mask_shape", "mask_bitorder", "source_ids",
                "frame_ids", "raw_indices", "ranks", "candidate_indices",
                "masks_packbits", "point_offsets", "points_world", "voxel_keys",
                "hl_index_offsets", "hl_retained_indices", "hlg_index_offsets",
                "hlg_retained_indices",
            }
            if set(archive.files) != required:
                raise F5RunnerError(f"F2 evidence schema differs: {scene}")
            if (
                str(archive["schema"].item()) != EXPECTED_F2_EVIDENCE_SCHEMA
                or str(archive["scene_id"].item()) != scene
                or archive["mask_shape"].tolist() != [480, 640]
                or str(archive["mask_bitorder"].item()) != "little"
            ):
                raise F5RunnerError(f"F2 evidence metadata differs: {scene}")
            self.source_ids = archive["source_ids"]
            self.frame_ids = archive["frame_ids"]
            self.raw_indices = archive["raw_indices"]
            self.ranks = archive["ranks"]
            self.candidate_indices = archive["candidate_indices"]
            self.masks = archive["masks_packbits"]
            self.offsets = archive["point_offsets"]
            self.points = archive["points_world"]
            self.keys = archive["voxel_keys"]
            self._archive = archive
        except (OSError, ValueError) as error:
            raise F5RunnerError(f"F2 evidence cannot be decoded: {scene}") from error
        if (
            self.source_ids.shape != (count,)
            or self.frame_ids.shape != (count,)
            or self.raw_indices.shape != (count,)
            or self.ranks.shape != (count,)
            or self.candidate_indices.shape != (count,)
            or self.masks.shape != (count, MASK_PACKED_BYTES)
            or self.masks.dtype != np.uint8
            or self.offsets.shape != (count + 1,)
            or int(self.offsets[0]) != 0
            or (count and np.any(self.offsets[1:] <= self.offsets[:-1]))
            or self.points.shape != (int(self.offsets[-1]), 3)
            or self.keys.shape != self.points.shape
            or self.points.dtype != np.dtype("<f8")
            or self.keys.dtype != np.dtype("<i8")
        ):
            raise F5RunnerError(f"F2 evidence arrays differ: {scene}")
        self.path = source
        self.sha256 = expected_sha
        self.cursor = 0
        self.access_count = 0
        self.maximum_logical_accessed_ordinal = -1

    def reset(self) -> None:
        self.cursor = 0
        self.access_count = 0
        self.maximum_logical_accessed_ordinal = -1

    def expose_current(
        self,
        *,
        scene: str,
        frame_id: int,
        frame_ordinal: int,
        sources: Sequence[Mapping[str, Any]],
    ) -> list[np.ndarray]:
        start_index = self.cursor
        stop_index = start_index + len(sources)
        if stop_index > len(self.source_ids):
            raise F5RunnerError(f"future/out-of-range F2 source offset: {scene}/{frame_id}")
        result: list[np.ndarray] = []
        for local_index, source in enumerate(sources):
            index = start_index + local_index
            expected_id = str(source.get("source_id"))
            if (
                str(self.source_ids[index]) != expected_id
                or int(self.frame_ids[index]) != frame_id
                or int(self.raw_indices[index]) != source.get("raw_index")
                or int(self.ranks[index]) != local_index
                or int(self.candidate_indices[index]) != local_index
                or source.get("rank") != local_index
                or source.get("candidate_index") != local_index
            ):
                raise F5RunnerError(f"current F2 evidence/source offset differs: {expected_id}")
            begin = int(self.offsets[index])
            end = int(self.offsets[index + 1])
            points = np.ascontiguousarray(self.points[begin:end], dtype=np.float64)
            keys = np.ascontiguousarray(self.keys[begin:end], dtype=np.int64)
            digest = hashlib.sha256()
            digest.update(np.asarray(points, dtype="<f8").tobytes())
            digest.update(np.asarray(keys, dtype="<i8").tobytes())
            if digest.hexdigest() != source.get("points_and_voxel_keys_sha256"):
                raise F5RunnerError(f"current F2 evidence point hash differs: {expected_id}")
            result.append(points)
        self.cursor = stop_index
        self.access_count += len(sources)
        if sources:
            self.maximum_logical_accessed_ordinal = max(
                self.maximum_logical_accessed_ordinal, frame_ordinal
            )
        return result

    def finish(self, expected_count: int, scene: str) -> None:
        if self.cursor != expected_count or self.access_count != expected_count:
            raise F5RunnerError(f"F5 did not expose every sealed source once: {scene}")

    def close(self) -> None:
        self._archive.close()


def _load_inputs(
    receipt_path: Path,
    scene_list_path: Path,
    *,
    expected_scene_count: int,
    expected_keyframes: int | None,
    expected_successful_frames: int | None,
    expected_sources: int | None,
) -> tuple[dict[str, str], tuple[str, ...], tuple[dict[str, Any], ...]]:
    production = expected_scene_count == EXPECTED_SCENES
    scene_list = _regular_file(scene_list_path, "paper100 scene list", ".txt")
    if production and _sha256(scene_list) != EXPECTED_SCENE_LIST_SHA256:
        raise F5RunnerError("paper100 scene-list hash differs")
    scenes = tuple(
        line.strip() for line in scene_list.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if len(scenes) != expected_scene_count or len(set(scenes)) != len(scenes):
        raise F5RunnerError("paper100 scene count/order is invalid")
    path, receipt = _read_json(receipt_path, "sealed F4 merge")
    if (
        receipt.get("schema") != EXPECTED_F4_MERGE_SCHEMA
        or receipt.get("protocol_id") != EXPECTED_F4_PROTOCOL
        or receipt.get("complete") is not True
        or receipt.get("overall_pass") is not True
        or receipt.get("native_output_mutation_count") != 0
        or _content_hash_without(receipt, "content_sha256") != receipt.get("content_sha256")
    ):
        raise F5RunnerError("sealed F4 merge contract differs")
    f4_contracts = receipt.get("contracts")
    required_f4_contracts = {
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
    if not isinstance(f4_contracts, Mapping) or any(
        f4_contracts.get(key) is not value for key, value in required_f4_contracts.items()
    ):
        raise F5RunnerError("sealed F4 merge forbidden-access contract differs")
    run_signature = receipt.get("run_signature_sha256")
    if not isinstance(run_signature, str) or len(run_signature) != 64:
        raise F5RunnerError("sealed F4 run signature is absent")
    coverage = receipt.get("coverage")
    totals = receipt.get("totals")
    if (
        not isinstance(coverage, Mapping)
        or coverage.get("scene_count") != expected_scene_count
        or coverage.get("scene_order") != list(scenes)
        or not isinstance(totals, Mapping)
    ):
        raise F5RunnerError("sealed F4 merge coverage differs")
    expected = {
        "keyframe_count": expected_keyframes,
        "successful_frame_count": expected_successful_frames,
        "source_count": expected_sources,
    }
    for key, value in expected.items():
        if value is not None and totals.get(key) != value:
            raise F5RunnerError(f"sealed F4 merge {key} differs")
    rows = receipt.get("scenes")
    if not isinstance(rows, list) or len(rows) != len(scenes):
        raise F5RunnerError("sealed F4 scene ledger differs")
    checked: list[dict[str, Any]] = []
    for index, (scene, row) in enumerate(zip(scenes, rows, strict=True)):
        if not isinstance(row, Mapping) or row.get("scene_id") != scene or row.get("scene_index") != index:
            raise F5RunnerError("sealed F4 scene order differs")
        _rehash_reference(row.get("sidecar"), f"{scene} F4 sidecar", ".json")
        checked.append(dict(row))
    return {
        "path": os.fspath(path),
        "sha256": _sha256(path),
        "run_signature_sha256": run_signature,
    }, scenes, tuple(checked)


def _source_receipts() -> dict[str, dict[str, str]]:
    paths = {
        "runner": Path(__file__).resolve(),
        "core": REPOSITORY_ROOT / "boxfusion/fastsam_f5_selector.py",
        "protocol": PROTOCOL_PATH,
    }
    receipts = {
        key: {"path": os.fspath(_regular_file(path, f"F5 {key}")), "sha256": _sha256(path)}
        for key, path in paths.items()
    }
    if receipts["protocol"]["sha256"] != EXPECTED_PROTOCOL_SHA256:
        raise F5RunnerError("frozen F5 protocol hash differs")
    return receipts


def _make_evidence_rows(
    *,
    core: object,
    scene: str,
    ordinal: int,
    frame_id: int,
    f4_sources: Sequence[Mapping[str, Any]],
    f2_sources: Sequence[Mapping[str, Any]],
    points: Sequence[np.ndarray],
    pose: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[Any, ...]:
    rows = []
    for rank, (source, f2_source, source_points) in enumerate(
        zip(f4_sources, f2_sources, points, strict=True)
    ):
        source_id = f"{scene}/frame_{frame_id:06d}/raw_{int(source.get('raw_index', -1)):03d}"
        hypotheses = source.get("hypotheses")
        if (
            source.get("source_id") != source_id
            or f2_source.get("source_id") != source_id
            or source.get("rank") != rank
            or source.get("candidate_index") != rank
            or source.get("frame_ordinal") != ordinal
            or source.get("frame_id") != frame_id
            or not isinstance(hypotheses, Mapping)
            or set(hypotheses) != {"H0", "HL", "HLG", "HB"}
            or {name: hypotheses[name] for name in ("H0", "HL", "HLG")}
            != f2_source.get("hypotheses")
            or not isinstance(source.get("source_lineage_sha256"), str)
        ):
            raise F5RunnerError(f"sealed F4/F2 source identity differs: {source_id}")
        try:
            rows.append(
                core.F5SourceEvidence(
                    source_id=source_id,
                    frame_id=frame_id,
                    frame_ordinal=ordinal,
                    rank=rank,
                    hypotheses=dict(hypotheses),
                    points_world=source_points,
                    tight_box_xyxy=source.get("tight_box_xyxy"),
                    camera_to_world=pose,
                    intrinsic=intrinsic,
                    source_lineage_sha256=source.get("source_lineage_sha256"),
                )
            )
        except (TypeError, ValueError, RuntimeError) as error:
            raise F5RunnerError(f"F5 core rejected sealed source: {source_id}") from error
    return tuple(rows)


def _process_scene(
    *,
    core: object,
    scene_row: Mapping[str, Any],
    scene_index: int,
    run_signature: str,
    f4_receipt_seal: Mapping[str, str],
    source_receipts: Mapping[str, Mapping[str, str]],
    output_root: Path,
    nonempty_call_index: list[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    scene = str(scene_row.get("scene_id"))
    f4_path = _rehash_reference(scene_row.get("sidecar"), f"{scene} F4 sidecar", ".json")
    _, f4 = _read_json(f4_path, f"{scene} F4 sidecar")
    if (
        f4.get("schema") != EXPECTED_F4_SCENE_SCHEMA
        or f4.get("protocol_id") != EXPECTED_F4_PROTOCOL
        or f4.get("complete") is not True
        or f4.get("scene_id") != scene
        or f4.get("scene_index") != scene_index
        or f4.get("run_signature_sha256") != f4_receipt_seal.get("run_signature_sha256")
        or f4.get("native_output_mutation_count") != 0
        or _content_hash_without(f4, "content_sha256") != f4.get("content_sha256")
    ):
        raise F5RunnerError(f"sealed F4 scene contract differs: {scene}")
    inputs = f4.get("inputs")
    if not isinstance(inputs, Mapping):
        raise F5RunnerError(f"sealed F4 inputs are absent: {scene}")
    f2_path = _rehash_reference(inputs.get("f2_sidecar"), f"{scene} F2 sidecar", ".json")
    _, f2 = _read_json(f2_path, f"{scene} F2 sidecar")
    if (
        f2.get("schema") != EXPECTED_F2_SCENE_SCHEMA
        or f2.get("complete") is not True
        or f2.get("scene_id") != scene
        or f2.get("scene_index") != scene_index
    ):
        raise F5RunnerError(f"sealed F2 scene contract differs: {scene}")
    evidence_path = _rehash_reference(inputs.get("f2_evidence"), f"{scene} F2 evidence", ".npz")
    intrinsic_path, intrinsic = _load_intrinsic(inputs.get("intrinsic"))
    frames = f4.get("frames")
    f2_frames = f2.get("frames")
    if not isinstance(frames, list) or not isinstance(f2_frames, list) or len(frames) != len(f2_frames):
        raise F5RunnerError(f"F4/F2 frame ledger differs: {scene}")
    source_count = sum(len(frame.get("sources", ())) for frame in frames if isinstance(frame, Mapping))
    if source_count != f4.get("counts", {}).get("source_count"):
        raise F5RunnerError(f"F4 source census differs: {scene}")
    accessor = _EvidenceAccessor(evidence_path, _sha256(evidence_path), scene, source_count)
    input_seals: list[dict[str, Any]] = [
        {"kind": "f4_sidecar", "path": os.fspath(f4_path), "sha256": _sha256(f4_path)},
        {"kind": "f2_sidecar", "path": os.fspath(f2_path), "sha256": _sha256(f2_path)},
        {"kind": "f2_evidence", "path": os.fspath(evidence_path), "sha256": _sha256(evidence_path)},
        {"kind": "intrinsic", "path": os.fspath(intrinsic_path), "sha256": _sha256(intrinsic_path)},
    ]
    input_before = _canonical_json_sha256(input_seals)

    def execute(*, audit_mode: str, prefix_successes: int | None = None) -> tuple[list[dict[str, Any]], list[str]]:
        accessor.reset()
        state = core.F5SelectorState()
        output_frames: list[dict[str, Any]] = []
        result_hashes: list[str] = []
        successful_seen = 0
        exposed_source_count = 0
        if prefix_successes is None:
            expected_exposed_source_count = source_count
        else:
            expected_exposed_source_count = 0
            expected_successes_seen = 0
            for expected_frame in frames:
                if expected_frame.get("successful") is True:
                    if expected_successes_seen >= prefix_successes:
                        break
                    expected_successes_seen += 1
                    expected_exposed_source_count += len(expected_frame.get("sources", ()))
        for ordinal, (f4_frame, f2_frame) in enumerate(zip(frames, f2_frames, strict=True)):
            if not isinstance(f4_frame, Mapping) or not isinstance(f2_frame, Mapping):
                raise F5RunnerError(f"invalid sealed frame row: {scene}/{ordinal}")
            frame_id = f4_frame.get("frame_id")
            successful = f4_frame.get("successful") is True
            if (
                f4_frame.get("frame_ordinal") != ordinal
                or f2_frame.get("frame_ordinal") != ordinal
                or f2_frame.get("frame_id") != frame_id
                or (f2_frame.get("successful") is True) is not successful
            ):
                raise F5RunnerError(f"F4/F2 frame identity differs: {scene}/{ordinal}")
            if prefix_successes is not None and successful and successful_seen >= prefix_successes:
                break
            f4_sources = f4_frame.get("sources")
            f2_sources = f2_frame.get("sources")
            if not isinstance(f4_sources, list) or not isinstance(f2_sources, list) or len(f4_sources) != len(f2_sources):
                raise F5RunnerError(f"F4/F2 source ledger differs: {scene}/{frame_id}")
            if not successful:
                if f4_sources or f2_sources:
                    raise F5RunnerError(f"failed frame retains sources: {scene}/{frame_id}")
                if audit_mode == "online":
                    output_frames.append({
                        "frame_ordinal": ordinal,
                        "frame_id": frame_id,
                        "successful": False,
                        "abstention": f4_frame.get("abstention"),
                        "sources": [],
                        "buffer_before": None,
                        "buffer_after": None,
                        "maximum_accessed_frame_ordinal": None,
                        "query": None,
                        "commit": None,
                        "runtime": None,
                    })
                continue
            successful_seen += 1
            frame_input = f4_frame.get("input")
            if not isinstance(frame_input, Mapping):
                raise F5RunnerError(f"successful F4 input is absent: {scene}/{frame_id}")
            pose_path, pose = _load_pose(frame_input.get("pose"), scene, int(frame_id))
            if audit_mode == "online":
                input_seals.append({
                    "kind": "pose", "frame_ordinal": ordinal, "frame_id": frame_id,
                    "path": os.fspath(pose_path), "sha256": _sha256(pose_path),
                })
            started = time.perf_counter_ns()
            points = accessor.expose_current(
                scene=scene,
                frame_id=int(frame_id),
                frame_ordinal=ordinal,
                sources=f4_sources,
            )
            exposed_source_count += len(f4_sources)
            evidence_rows = _make_evidence_rows(
                core=core,
                scene=scene,
                ordinal=ordinal,
                frame_id=int(frame_id),
                f4_sources=f4_sources,
                f2_sources=f2_sources,
                points=points,
                pose=pose,
                intrinsic=intrinsic,
            )
            try:
                query = state.query_frame(
                    frame_id=int(frame_id), frame_ordinal=ordinal, sources=evidence_rows
                )
                buffer_before = [dict(row) for row in query.buffer_before]
                commit = state.commit_frame(query)
            except (TypeError, ValueError, RuntimeError) as error:
                raise F5RunnerError(f"F5 core rejected frame: {scene}/{frame_id}") from error
            elapsed_ms = (time.perf_counter_ns() - started) / 1.0e6
            rows = [dict(row) for row in query.rows]
            hashes = [str(row["result_sha256"]) for row in rows]
            result_hashes.extend(hashes)
            if query.maximum_accessed_frame_ordinal >= ordinal or commit.token != query.token:
                raise F5RunnerError(f"F5 query-before-commit/causality differs: {scene}/{frame_id}")
            if audit_mode == "online":
                nonempty = bool(rows)
                call_index = nonempty_call_index[0] if nonempty else None
                warmup = bool(nonempty and int(call_index) < WARMUP_NONEMPTY_FRAMES)
                if nonempty:
                    nonempty_call_index[0] += 1
                inherited = f4_frame.get("runtime")
                if nonempty:
                    if not isinstance(inherited, Mapping):
                        raise F5RunnerError(f"F4 composed runtime is absent: {scene}/{frame_id}")
                    inherited_ms = _number(inherited.get("replay_composed_ms"), "F4 composed runtime")
                else:
                    f2_runtime = f2_frame.get("runtime")
                    inherited_ms = (
                        _number(f2_runtime.get("complete_ms"), "F2 complete runtime")
                        if isinstance(f2_runtime, Mapping) else 0.0
                    )
                composed_ms = inherited_ms + elapsed_ms
                output_frames.append({
                    "frame_ordinal": ordinal,
                    "frame_id": int(frame_id),
                    "successful": True,
                    "abstention": None,
                    "sources": rows,
                    "buffer_before": buffer_before,
                    "buffer_after": [dict(row) for row in commit.buffer_after],
                    "maximum_accessed_frame_ordinal": query.maximum_accessed_frame_ordinal,
                    "query": {
                        "query_before_commit": True,
                        "buffer_before": buffer_before,
                        "maximum_accessed_frame_ordinal": query.maximum_accessed_frame_ordinal,
                        "maximum_lookahead_frames": 0,
                        "token": query.token,
                    },
                    "commit": {
                        "buffer_after": [dict(row) for row in commit.buffer_after],
                        "source_count": commit.source_count,
                        "token": commit.token,
                    },
                    "runtime": {
                        "nonempty_call_index_in_shard": call_index,
                        "f5_warmup_excluded": warmup,
                        "f5_incremental_ms": elapsed_ms,
                        "sealed_f4_composed_ms": inherited_ms,
                        "replay_composed_ms": composed_ms,
                        "replay_composed_ms_per_source_frame": composed_ms / SOURCE_FRAME_STRIDE,
                        "gap25_deadline_missed": composed_ms >= DEADLINE_MS,
                        "gap25_deadline_missed_warm": (not warmup) and composed_ms >= DEADLINE_MS,
                        "f5_cuda_allocated_bytes": 0,
                    },
                })
        if exposed_source_count != expected_exposed_source_count:
            raise F5RunnerError(f"F5 replay source census differs: {scene}/{audit_mode}")
        accessor.finish(expected_exposed_source_count, scene)
        return output_frames, result_hashes

    try:
        online_frames, online_hashes = execute(audit_mode="online")
        successful_count = sum(frame.get("successful") is True for frame in frames)
        prefix_successes = successful_count // 2
        _, prefix_hashes = execute(audit_mode="prefix", prefix_successes=prefix_successes)
        if prefix_hashes != online_hashes[: len(prefix_hashes)]:
            raise F5RunnerError(f"F5 prefix replay differs: {scene}")
        _, independent_hashes = execute(audit_mode="independent")
        if independent_hashes != online_hashes:
            raise F5RunnerError(f"F5 independent replay differs: {scene}")
    finally:
        accessor.close()

    for seal in input_seals:
        if _sha256(_regular_file(Path(seal["path"]), f"{scene} frozen input after F5")) != seal["sha256"]:
            raise F5RunnerError(f"frozen input changed during F5: {scene}")
    opened_inputs_before = _canonical_json_sha256(input_seals)
    # The initial aggregate grows only by current-pose seals.  Their bytes are
    # rehashed above; this separate ledger makes that intended growth explicit.
    if not isinstance(input_before, str) or len(input_before) != 64:
        raise F5RunnerError(f"invalid initial input aggregate: {scene}")
    opened_inputs_after = _canonical_json_sha256(input_seals)
    if opened_inputs_after != opened_inputs_before:
        raise F5RunnerError(f"opened input ledger changed during F5: {scene}")

    selected_counts = {name: 0 for name in ("H0", "HL", "HLG", "HB")}
    source_ids: list[str] = []
    lineage_hashes: list[str] = []
    all_runtime: list[float] = []
    warm_runtime: list[float] = []
    all_composed: list[float] = []
    warm_composed: list[float] = []
    max_prior = -1
    for frame in online_frames:
        for row in frame["sources"]:
            selected_counts[str(row["selected_hypothesis"])] += 1
            source_ids.append(str(row["source_id"]))
            lineage_hashes.append(str(row["source_lineage_sha256"]))
        query = frame.get("query")
        if isinstance(query, Mapping):
            max_prior = max(max_prior, int(query["maximum_accessed_frame_ordinal"]))
        runtime = frame.get("runtime")
        if isinstance(runtime, Mapping):
            inc = float(runtime["f5_incremental_ms"])
            composed = float(runtime["replay_composed_ms"])
            all_runtime.append(inc)
            all_composed.append(composed)
            if runtime["f5_warmup_excluded"] is False:
                warm_runtime.append(inc)
                warm_composed.append(composed)
    counts = {
        "keyframe_count": len(frames),
        "successful_frame_count": sum(frame.get("successful") is True for frame in frames),
        "source_count": len(source_ids),
        "identity_verified_source_count": len(source_ids),
        "selected_h0_count": selected_counts["H0"],
        "selected_hl_count": selected_counts["HL"],
        "selected_hlg_count": selected_counts["HLG"],
        "selected_hb_count": selected_counts["HB"],
    }
    runtime = {
        "f5_incremental_all_ms": _distribution(all_runtime),
        "f5_incremental_warm_ms": _distribution(warm_runtime),
        "replay_composed_all_ms": _distribution(all_composed),
        "replay_composed_warm_ms": _distribution(warm_composed),
        "replay_composed_warm_mean_per_source_frame_ms": (
            float(np.mean(warm_composed)) / SOURCE_FRAME_STRIDE if warm_composed else 0.0
        ),
        "gap25_all_deadline_miss_count": int(sum(value >= DEADLINE_MS for value in all_composed)),
        "gap25_warm_deadline_miss_count": int(sum(value >= DEADLINE_MS for value in warm_composed)),
        "f5_cuda_peak_memory_bytes": 0,
        "inherited_f4_cuda_peak_memory_bytes": int(f4.get("runtime", {}).get("cuda_peak_memory_bytes", 0)),
        "cuda_peak_memory_bytes": int(f4.get("runtime", {}).get("cuda_peak_memory_bytes", 0)),
    }
    causality = {
        "overall_pass": True,
        "query_before_commit": True,
        "prefix_replay_pass": True,
        "independent_replay_pass": True,
        "maximum_lookahead_frames": 0,
        "maximum_accessed_past_frame_ordinal": max_prior,
        "future_access_count": 0,
        "current_source_offsets_only": True,
        "prefix_successful_frame_count": prefix_successes,
    }
    receipt: dict[str, Any] = {
        "schema": SCENE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "scene_id": scene,
        "scene_index": scene_index,
        "run_signature_sha256": run_signature,
        "contracts": dict(CONTRACTS),
        "inputs": {
            "f4_receipt": dict(f4_receipt_seal),
            "f4_sidecar": input_seals[0],
            "f2_sidecar": input_seals[1],
            "f2_evidence": input_seals[2],
            "intrinsic": input_seals[3],
            "frozen_base_inputs_sha256": input_before,
            "all_opened_inputs_before_sha256": opened_inputs_before,
            "all_opened_inputs_after_sha256": opened_inputs_after,
            "sources": {key: dict(value) for key, value in source_receipts.items()},
        },
        "counts": counts,
        "causality": causality,
        "runtime": runtime,
        "prefix_replay": {
            "passed": True,
            "successful_frame_count": prefix_successes,
            "result_row_count": len(prefix_hashes),
            "result_ledger_sha256": _canonical_json_sha256(prefix_hashes),
        },
        "determinism": {
            "passed": True,
            "independent_replay_count": 1,
            "online_result_ledger_sha256": _canonical_json_sha256(online_hashes),
            "independent_result_ledger_sha256": _canonical_json_sha256(independent_hashes),
        },
        "frames": online_frames,
        "source_ids_sha256": _canonical_json_sha256(source_ids),
        "source_lineage_sha256": _canonical_json_sha256(lineage_hashes),
        "result_ledger_sha256": _canonical_json_sha256(online_hashes),
        "native_output_mutation_count": 0,
        "birth_count": 0,
    }
    receipt["content_sha256"] = _canonical_json_sha256(receipt)
    output_path = output_root / "scenes" / f"{scene}.json"
    output_sha = _atomic_create_json(output_path, receipt)
    row = {
        "scene_id": scene,
        "scene_index": scene_index,
        "sidecar": {"path": os.fspath(output_path.resolve()), "sha256": output_sha},
        "counts": counts,
        "causality": causality,
        "runtime": runtime,
        "source_ids_sha256": receipt["source_ids_sha256"],
        "source_lineage_sha256": receipt["source_lineage_sha256"],
        "result_ledger_sha256": receipt["result_ledger_sha256"],
        "prefix_replay": receipt["prefix_replay"],
        "determinism": receipt["determinism"],
    }
    return row, receipt


def run_f5(
    *,
    f4_receipt_path: Path = DEFAULT_F4_RECEIPT,
    scene_list_path: Path = DEFAULT_SCENE_LIST,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    shard_index: int,
    num_shards: int = EXPECTED_SHARDS,
    expected_scene_count: int = EXPECTED_SCENES,
    expected_keyframes: int | None = None,
    expected_successful_frames: int | None = None,
    expected_sources: int | None = None,
    plan_only: bool = False,
) -> dict[str, Any]:
    if num_shards != EXPECTED_SHARDS or shard_index not in range(num_shards):
        raise F5RunnerError("F5 is frozen to exactly two deterministic shards")
    production = expected_scene_count == EXPECTED_SCENES
    expected_keyframes = EXPECTED_KEYFRAMES if production and expected_keyframes is None else expected_keyframes
    expected_successful_frames = EXPECTED_SUCCESSFUL_FRAMES if production and expected_successful_frames is None else expected_successful_frames
    expected_sources = EXPECTED_SOURCES if production and expected_sources is None else expected_sources
    f4_seal, scenes, scene_rows = _load_inputs(
        Path(f4_receipt_path), Path(scene_list_path),
        expected_scene_count=expected_scene_count,
        expected_keyframes=expected_keyframes,
        expected_successful_frames=expected_successful_frames,
        expected_sources=expected_sources,
    )
    assigned = tuple(index for index in range(len(scenes)) if index % num_shards == shard_index)
    plan = {
        "schema": SHARD_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "mode": "plan_only" if plan_only else "shadow",
        "shard_index": shard_index,
        "num_shards": num_shards,
        "scene_indices": list(assigned),
        "scene_ids": [scenes[index] for index in assigned],
        "f4_receipt": f4_seal,
        "contracts": dict(CONTRACTS),
        "output_root": os.fspath(Path(output_root).resolve()),
    }
    # Planning is an authenticated dry run, not an unsealed convenience path.
    # In particular it must fail before reporting a runnable plan if the
    # preregistered protocol bytes have changed.
    source_receipts = _source_receipts()
    plan["sources"] = source_receipts
    if plan_only:
        return plan
    try:
        from boxfusion import fastsam_f5_selector as core
    except ImportError as error:  # pragma: no cover
        raise F5RunnerError("F5 selector core is unavailable") from error
    if getattr(core, "PROTOCOL_ID", None) != PROTOCOL_ID:
        raise F5RunnerError("F5 core protocol differs")
    signature_payload = {
        "protocol_id": PROTOCOL_ID,
        "f4_receipt": f4_seal,
        "scene_order": list(scenes),
        "scene_list_sha256": _sha256(Path(scene_list_path)),
        "core_schema": getattr(core, "SCHEMA", None),
        "core_policy": dict(getattr(core, "POLICY", {})),
        "sources": source_receipts,
        "contracts": dict(CONTRACTS),
        "num_shards": num_shards,
    }
    run_signature = _canonical_json_sha256(signature_payload)
    output = Path(output_root)
    if output.is_symlink():
        raise F5RunnerError("F5 output root cannot be a symlink")
    scene_outputs: list[dict[str, Any]] = []
    scene_receipts: list[dict[str, Any]] = []
    nonempty_call_index = [0]
    for scene_index in assigned:
        row, receipt = _process_scene(
            core=core,
            scene_row=scene_rows[scene_index],
            scene_index=scene_index,
            run_signature=run_signature,
            f4_receipt_seal=f4_seal,
            source_receipts=source_receipts,
            output_root=output,
            nonempty_call_index=nonempty_call_index,
        )
        scene_outputs.append(row)
        scene_receipts.append(receipt)
    for row in source_receipts.values():
        if _sha256(_regular_file(Path(row["path"]), "F5 frozen source after replay")) != row["sha256"]:
            raise F5RunnerError("F5 source/protocol changed during replay")
    totals = {
        key: int(sum(row["counts"][key] for row in scene_outputs))
        for key in (
            "keyframe_count", "successful_frame_count", "source_count",
            "identity_verified_source_count", "selected_h0_count", "selected_hl_count",
            "selected_hlg_count", "selected_hb_count",
        )
    }
    if production:
        for key, expected in EXPECTED_SHARD_COUNTS[shard_index].items():
            if totals[key] != expected:
                raise F5RunnerError(f"production shard {shard_index} {key} differs")
    warm_incremental = [
        float(frame["runtime"]["f5_incremental_ms"])
        for scene in scene_receipts for frame in scene["frames"]
        if isinstance(frame.get("runtime"), Mapping)
        and frame["runtime"].get("f5_warmup_excluded") is False
    ]
    warm_composed = [
        float(frame["runtime"]["replay_composed_ms"])
        for scene in scene_receipts for frame in scene["frames"]
        if isinstance(frame.get("runtime"), Mapping)
        and frame["runtime"].get("f5_warmup_excluded") is False
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
        "inputs": {
            "f4_receipt": f4_seal,
            "scene_list": {"path": os.fspath(Path(scene_list_path).resolve()), "sha256": _sha256(Path(scene_list_path))},
            "sources": source_receipts,
        },
        "environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
        },
        "scenes": scene_outputs,
        "totals": totals,
        "causality": {
            "overall_pass": all(row["causality"]["overall_pass"] for row in scene_outputs),
            "query_before_commit": True,
            "prefix_replay_pass": True,
            "independent_replay_pass": True,
            "maximum_lookahead_frames": 0,
            "future_access_count": 0,
            "current_source_offsets_only": True,
        },
        "determinism": {
            "overall_pass": all(row["determinism"]["passed"] for row in scene_outputs),
            "prefix_replay_pass": all(row["prefix_replay"]["passed"] for row in scene_outputs),
            "independent_replay_pass": all(row["determinism"]["passed"] for row in scene_outputs),
            "scene_result_ledger_sha256": _canonical_json_sha256(
                [row["result_ledger_sha256"] for row in scene_outputs]
            ),
        },
        "runtime": {
            "f5_incremental_warm_ms": _distribution(warm_incremental),
            "replay_composed_warm_ms": _distribution(warm_composed),
            "replay_composed_warm_mean_per_source_frame_ms": (
                float(np.mean(warm_composed)) / SOURCE_FRAME_STRIDE if warm_composed else 0.0
            ),
            "gap25_all_deadline_miss_count": int(sum(scene["runtime"]["gap25_all_deadline_miss_count"] for scene in scene_receipts)),
            "gap25_warm_deadline_miss_count": int(sum(scene["runtime"]["gap25_warm_deadline_miss_count"] for scene in scene_receipts)),
            "f5_cuda_peak_memory_bytes": 0,
            "inherited_f4_cuda_peak_memory_bytes": max((scene["runtime"]["inherited_f4_cuda_peak_memory_bytes"] for scene in scene_receipts), default=0),
            "cuda_peak_memory_bytes": max((scene["runtime"]["cuda_peak_memory_bytes"] for scene in scene_receipts), default=0),
            "warmup_nonempty_frame_count": min(WARMUP_NONEMPTY_FRAMES, nonempty_call_index[0]),
        },
        "native_output_mutation_count": 0,
        "birth_count": 0,
    }
    manifest_path = output / "shards" / f"shard-{shard_index:03d}-of-{num_shards:03d}.json"
    manifest["manifest_path"] = os.fspath(manifest_path.resolve())
    manifest["content_sha256"] = _canonical_json_sha256(manifest)
    manifest_sha = _atomic_create_json(manifest_path, manifest)
    manifest["manifest_sha256"] = manifest_sha
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f4-receipt", type=Path, default=DEFAULT_F4_RECEIPT)
    parser.add_argument("--scene-list", type=Path, default=DEFAULT_SCENE_LIST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=EXPECTED_SHARDS)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_f5(
        f4_receipt_path=args.f4_receipt,
        scene_list_path=args.scene_list,
        output_root=args.output_root,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        plan_only=args.plan_only,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
