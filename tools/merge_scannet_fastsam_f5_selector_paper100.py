#!/usr/bin/env python3
"""Fail-closed merge for the frozen F5 paper100 selector shadow.

The merge performs a third, untimed replay of the sealed F4/F2 evidence.  It
uses no annotations, detector predictions, evaluator output, oracle report or
semantic feature.  The replay is audit overhead: only runtimes sealed by the
two F5 shards enter the formal online gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))
PROTOCOL_ID = "F5-GT-FREE-PAST-ONLY-GEOMETRY-SELECTOR-PAPER100"
PROTOCOL_SHA256 = "2a6d62fa9d5912dc3871bbc485f44987565bda61b818722b3a4e6577d34a6afc"
SCENE_SCHEMA = "boxfusion.scannet_fastsam_f5_gtfree_selector_paper100.scene.v1"
SHARD_SCHEMA = "boxfusion.scannet_fastsam_f5_gtfree_selector_paper100.shard.v1"
MERGE_SCHEMA = "boxfusion.scannet_fastsam_f5_gtfree_selector_paper100.merge.v1"
EXPECTED_F4_MERGE_SCHEMA = "boxfusion.scannet_fastsam_f4_boxer_paper100.merge.v1"
EXPECTED_F4_SCENE_SCHEMA = "boxfusion.scannet_fastsam_f4_boxer_paper100.scene.v1"
EXPECTED_F4_PROTOCOL = "F4-FASTSAM-BOXER-GEOMETRY-SHADOW-PAPER100"
EXPECTED_F2_SCENE_SCHEMA = "boxfusion.scannet_fastsam_f2_paper100.scene.v1"
EXPECTED_F2_EVIDENCE_SCHEMA = "boxfusion.scannet_fastsam_f2_paper100.evidence.v1"

EXPECTED_SCENES = 100
EXPECTED_SHARDS = 2
EXPECTED_KEYFRAMES = 6_817
EXPECTED_SUCCESSFUL_FRAMES = 6_726
EXPECTED_SOURCES = 52_299
EXPECTED_SHARD_COUNTS: Mapping[int, Mapping[str, int]] = {
    0: {"keyframe_count": 3_259, "successful_frame_count": 3_189, "source_count": 24_863},
    1: {"keyframe_count": 3_558, "successful_frame_count": 3_537, "source_count": 27_436},
}

WARMUP_NONEMPTY_FRAMES = 3
SOURCE_FRAME_STRIDE = 25.0
MAX_F5_INCREMENTAL_P95_MS = 25.0
MAX_COMPOSED_P95_MS = 375.0
MAX_COMPOSED_MS_EXCLUSIVE = 833.33
MAX_COMPOSED_MEAN_PER_SOURCE_FRAME_MS = 15.0
MAX_CUDA_PEAK_BYTES = 4 * 1024**3
MAX_BUFFERED_FRAMES = 3
MAX_SOURCES_PER_FRAME = 16
MIN_SELECTED_HB_SOURCES = 128
MIN_SELECTED_HB_SCENES = 20
MAX_SELECTED_HB_FRACTION = 0.20
MASK_PACKED_BYTES = 480 * 640 // 8

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

DEFAULT_SHARDS = (
    REPOSITORY_ROOT / "logs/scannet_fastsam_f5_selector_paper100_score05/shards/shard-000-of-002.json",
    REPOSITORY_ROOT / "logs/scannet_fastsam_f5_selector_paper100_score05/shards/shard-001-of-002.json",
)
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "logs/scannet_fastsam_f5_selector_paper100_score05/final"
OUTPUT_NAME = "F5_GT_FREE_SELECTOR_PAPER100.json"
PROTOCOL_PATH = REPOSITORY_ROOT / "docs/F5_GT_FREE_GEOMETRY_SELECTOR_PROTOCOL_FREEZE.md"


class F5MergeError(RuntimeError):
    """Raised when a shard, lineage, decision or causal proof differs."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise F5MergeError("value is not canonical finite ASCII JSON") from error
    return hashlib.sha256(payload).hexdigest()


def _content_hash_without(value: Mapping[str, Any], *keys: str) -> str:
    payload = dict(value)
    for key in keys:
        payload.pop(key, None)
    return _canonical_json_sha256(payload)


def _regular_file(path: Path, label: str, suffix: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise F5MergeError(f"{label} must be a regular non-symlink file: {path}")
    result = path.resolve()
    if suffix is not None and result.suffix.lower() != suffix:
        raise F5MergeError(f"{label} must have suffix {suffix}: {result}")
    if result.suffix.lower() in {".pkl", ".pickle"}:
        raise F5MergeError(f"forbidden serialized detector input: {result}")
    return result


def _read_json(path: Path, label: str) -> tuple[Path, dict[str, Any]]:
    source = _regular_file(path, label, ".json")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise F5MergeError(f"invalid {label}: {source}") from error
    if not isinstance(value, dict):
        raise F5MergeError(f"{label} must contain one JSON object")
    return source, value


def _rehash_reference(value: object, label: str, suffix: str | None = None) -> Path:
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("path"), str)
        or not isinstance(value.get("sha256"), str)
    ):
        raise F5MergeError(f"{label} seal is absent")
    path = _regular_file(Path(value["path"]), label, suffix)
    if _sha256(path) != value["sha256"]:
        raise F5MergeError(f"{label} rehash differs")
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
            raise F5MergeError(f"refusing to overwrite output: {path}") from error
        return _sha256(path)
    finally:
        temporary.unlink(missing_ok=True)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise F5MergeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise F5MergeError(f"{label} must be finite and nonnegative")
    return result


def _strict_int(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise F5MergeError(f"{label} must be an integer >= {minimum}")
    return value


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise F5MergeError("runtime samples must be finite and nonnegative")
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
    elif comparator == ">=":
        passed = actual >= threshold
    else:  # pragma: no cover
        raise AssertionError(comparator)
    return {
        "actual": actual,
        "comparator": comparator,
        "threshold": threshold,
        "pass": bool(passed),
        "passed": bool(passed),
    }


class _Evidence:
    """Sequential access to the F2 evidence that F5 is allowed to consume."""

    def __init__(self, path: Path, *, scene_id: str, source_count: int) -> None:
        try:
            archive = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise F5MergeError(f"cannot decode sealed F2 evidence: {scene_id}") from error
        required = {
            "schema", "scene_id", "mask_shape", "mask_bitorder", "source_ids",
            "frame_ids", "raw_indices", "ranks", "candidate_indices", "masks_packbits",
            "point_offsets", "points_world", "voxel_keys", "hl_index_offsets",
            "hl_retained_indices", "hlg_index_offsets", "hlg_retained_indices",
        }
        if set(archive.files) != required:
            archive.close()
            raise F5MergeError(f"F2 evidence schema differs: {scene_id}")
        try:
            if (
                str(archive["schema"].item()) != EXPECTED_F2_EVIDENCE_SCHEMA
                or str(archive["scene_id"].item()) != scene_id
                or archive["mask_shape"].tolist() != [480, 640]
                or str(archive["mask_bitorder"].item()) != "little"
            ):
                raise F5MergeError(f"F2 evidence metadata differs: {scene_id}")
            self.source_ids = archive["source_ids"]
            self.frame_ids = archive["frame_ids"]
            self.raw_indices = archive["raw_indices"]
            self.ranks = archive["ranks"]
            self.candidate_indices = archive["candidate_indices"]
            self.masks = archive["masks_packbits"]
            self.offsets = archive["point_offsets"]
            self.points = archive["points_world"]
            self.keys = archive["voxel_keys"]
            if (
                self.source_ids.shape != (source_count,)
                or self.frame_ids.shape != (source_count,)
                or self.raw_indices.shape != (source_count,)
                or self.ranks.shape != (source_count,)
                or self.candidate_indices.shape != (source_count,)
                or self.masks.shape != (source_count, MASK_PACKED_BYTES)
                or self.masks.dtype != np.uint8
                or self.offsets.shape != (source_count + 1,)
                or int(self.offsets[0]) != 0
                or (source_count and np.any(self.offsets[1:] <= self.offsets[:-1]))
                or self.points.shape != (int(self.offsets[-1]), 3)
                or self.keys.shape != self.points.shape
                or self.points.dtype != np.dtype("<f8")
                or self.keys.dtype != np.dtype("<i8")
            ):
                raise F5MergeError(f"F2 evidence arrays differ: {scene_id}")
        except Exception:
            archive.close()
            raise
        self._archive = archive
        self.cursor = 0

    def take(
        self,
        *,
        scene_id: str,
        frame_id: int,
        sources: Sequence[Mapping[str, Any]],
    ) -> list[np.ndarray]:
        rows: list[np.ndarray] = []
        for rank, source in enumerate(sources):
            index = self.cursor
            if index >= len(self.source_ids):
                raise F5MergeError(f"F2 evidence offset exceeds source count: {scene_id}/{frame_id}")
            source_id = source.get("source_id")
            if (
                str(self.source_ids[index]) != source_id
                or int(self.frame_ids[index]) != frame_id
                or int(self.raw_indices[index]) != source.get("raw_index")
                or int(self.ranks[index]) != rank
                or int(self.candidate_indices[index]) != rank
            ):
                raise F5MergeError(f"F2 evidence/source identity differs: {source_id}")
            begin, end = int(self.offsets[index]), int(self.offsets[index + 1])
            points = np.ascontiguousarray(self.points[begin:end], dtype=np.float64)
            keys = np.ascontiguousarray(self.keys[begin:end], dtype=np.int64)
            digest = hashlib.sha256()
            digest.update(np.asarray(points, dtype="<f8").tobytes())
            digest.update(np.asarray(keys, dtype="<i8").tobytes())
            if digest.hexdigest() != source.get("points_and_voxel_keys_sha256"):
                raise F5MergeError(f"F2 evidence point hash differs: {source_id}")
            rows.append(points)
            self.cursor += 1
        return rows

    def finish(self, source_count: int, scene_id: str) -> None:
        if self.cursor != source_count:
            raise F5MergeError(f"not every F2 source was consumed exactly once: {scene_id}")

    def close(self) -> None:
        self._archive.close()


def _load_matrix(reference: object, *, label: str, shape: tuple[int, int]) -> np.ndarray:
    path = _rehash_reference(reference, label, ".txt")
    try:
        value = np.loadtxt(path, dtype=np.float64)
    except (OSError, ValueError) as error:
        raise F5MergeError(f"cannot decode {label}") from error
    if shape == (3, 3) and value.shape == (4, 4):
        value = value[:3, :3]
    if value.shape != shape or not np.isfinite(value).all():
        raise F5MergeError(f"{label} shape/value differs")
    if shape == (3, 3) and (value[0, 0] <= 0.0 or value[1, 1] <= 0.0):
        raise F5MergeError(f"{label} focal lengths differ")
    return np.ascontiguousarray(value, dtype=np.float64)


def _expected_geometry(source: Mapping[str, Any], name: str) -> dict[str, Any]:
    hypotheses = source.get("hypotheses")
    if not isinstance(hypotheses, Mapping) or name not in hypotheses:
        raise F5MergeError(f"{source.get('source_id')} selected missing hypothesis {name}")
    row = hypotheses[name]
    if not isinstance(row, Mapping):
        raise F5MergeError(f"{source.get('source_id')} selected malformed hypothesis {name}")
    if name != "HB":
        lower = np.asarray(row.get("q02"), dtype=np.float64)
        upper = np.asarray(row.get("q98"), dtype=np.float64)
        if lower.shape != (3,) or upper.shape != (3,) or not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(upper <= lower):
            raise F5MergeError(f"{source.get('source_id')} selected AABB is invalid")
        return {
            "kind": "world_aabb",
            "hypothesis": name,
            "q02": lower.tolist(),
            "q98": upper.tolist(),
            "center": ((lower + upper) * 0.5).tolist(),
            "extent": (upper - lower).tolist(),
        }
    center = np.asarray(row.get("world_center"), dtype=np.float64)
    extent = np.asarray(row.get("local_extent"), dtype=np.float64)
    rotation = np.asarray(row.get("world_rotation"), dtype=np.float64)
    corners = np.asarray(row.get("world_corners"), dtype=np.float64)
    if (
        row.get("valid") is not True
        or center.shape != (3,)
        or extent.shape != (3,)
        or rotation.shape != (3, 3)
        or corners.shape != (8, 3)
        or not all(np.isfinite(value).all() for value in (center, extent, rotation, corners))
        or np.any(extent <= 0.0)
    ):
        raise F5MergeError(f"{source.get('source_id')} selected HB is invalid")
    lower, upper = corners.min(axis=0), corners.max(axis=0)
    return {
        "kind": "world_obb",
        "hypothesis": "HB",
        "world_center": center.tolist(),
        "local_extent": extent.tolist(),
        "world_rotation": rotation.tolist(),
        "world_corners": corners.tolist(),
        "envelope_q02": lower.tolist(),
        "envelope_q98": upper.tolist(),
        "envelope_center": ((lower + upper) * 0.5).tolist(),
        "envelope_extent": (upper - lower).tolist(),
    }


def _verify_selected_row(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    f4_source: Mapping[str, Any],
    *,
    buffer_before: Sequence[Mapping[str, Any]],
) -> tuple[str, bool]:
    if dict(actual) != dict(expected):
        raise F5MergeError(f"{f4_source.get('source_id')} selector row differs from audit replay")
    if type(actual.get("formal_score")) is not float or actual.get("formal_score") != 1.0:
        raise F5MergeError(f"{f4_source.get('source_id')} formal score differs from 1.0")
    if actual.get("source_id") != f4_source.get("source_id") or actual.get("source_lineage_sha256") != f4_source.get("source_lineage_sha256"):
        raise F5MergeError(f"{f4_source.get('source_id')} F4 lineage differs")
    hypotheses = f4_source.get("hypotheses")
    if not isinstance(hypotheses, Mapping) or set(hypotheses) != {"H0", "HL", "HLG", "HB"}:
        raise F5MergeError(f"{f4_source.get('source_id')} F4 hypothesis set differs")
    expected_hashes = {name: _canonical_json_sha256(hypotheses[name]) for name in ("H0", "HL", "HLG", "HB")}
    if actual.get("input_hypothesis_sha256") != expected_hashes:
        raise F5MergeError(f"{f4_source.get('source_id')} input hypothesis hashes differ")
    selected = actual.get("selected_hypothesis")
    if selected not in {"H0", "HL", "HLG", "HB"}:
        raise F5MergeError(f"{f4_source.get('source_id')} selected hypothesis differs")
    geometry = _expected_geometry(f4_source, str(selected))
    if actual.get("selected_geometry") != geometry:
        raise F5MergeError(f"{f4_source.get('source_id')} selected geometry is not an exact F4 copy")
    if _canonical_json_sha256(geometry) != actual.get("selected_geometry_sha256"):
        raise F5MergeError(f"{f4_source.get('source_id')} selected geometry hash differs")
    if _content_hash_without(actual, "result_sha256") != actual.get("result_sha256"):
        raise F5MergeError(f"{f4_source.get('source_id')} result hash differs")
    if selected == "H0" and actual.get("base_hypothesis") != "H0":
        raise F5MergeError(f"{f4_source.get('source_id')} H0 fallback proof differs")

    hb_selected = selected == "HB"
    if hb_selected:
        matched = actual.get("matched_past")
        if (
            actual.get("hb_abstention_reason") is not None
            or not isinstance(matched, list)
            or actual.get("matched_past_frame_count", 0) < 2
            or actual.get("hb_consistent_past_frame_count", 0) < 2
        ):
            raise F5MergeError(f"{f4_source.get('source_id')} selected HB lacks two-past proof")
        available: set[tuple[int, str, str]] = set()
        for frame in buffer_before:
            ordinal = _strict_int(frame.get("frame_ordinal"), "buffer frame ordinal")
            ids = frame.get("source_ids")
            hashes = frame.get("result_sha256")
            if not isinstance(ids, list) or not isinstance(hashes, list) or len(ids) != len(hashes):
                raise F5MergeError("buffer source/result ledger differs")
            available.update((ordinal, str(source_id), str(result_hash)) for source_id, result_hash in zip(ids, hashes))
        confirmed_ordinals: set[int] = set()
        for proof in matched:
            if not isinstance(proof, Mapping):
                raise F5MergeError(f"{f4_source.get('source_id')} HB past proof is malformed")
            key = (int(proof.get("frame_ordinal", -1)), str(proof.get("source_id")), str(proof.get("result_sha256")))
            if key not in available:
                raise F5MergeError(f"{f4_source.get('source_id')} HB cites unavailable past evidence")
            if proof.get("passed_hb_consistency") is True:
                confirmed_ordinals.add(key[0])
            if proof.get("hb_consistency_evaluated") is not True:
                raise F5MergeError(f"{f4_source.get('source_id')} selected HB contains an unevaluated past proof")
            for metric in (
                "base_iou3d", "base_symmetric_containment", "base_normalized_center_distance",
                "hb_iou3d", "hb_symmetric_containment", "hb_normalized_center_distance",
            ):
                _number(proof.get(metric), f"selected HB past {metric}")
        if len(matched) != actual.get("matched_past_frame_count") or len(confirmed_ordinals) < 2:
            raise F5MergeError(f"{f4_source.get('source_id')} HB has fewer than two distinct confirming frames")
        diagnostics = actual.get("hb_diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise F5MergeError(f"{f4_source.get('source_id')} HB current proof is absent")
        required = {
            "boxer_confidence": (0.55, None),
            "point_count": (16.0, None),
            "exact_point_support": (0.60, None),
            "expanded_point_support": (0.80, None),
            "projection_iou": (0.50, None),
        }
        for key, (minimum, _) in required.items():
            if _number(diagnostics.get(key), f"HB {key}") < minimum:
                raise F5MergeError(f"{f4_source.get('source_id')} HB current gate {key} differs")
        if _number(diagnostics.get("hb_base_normalized_center_distance"), "HB base ND") > 0.50:
            raise F5MergeError(f"{f4_source.get('source_id')} HB center proof differs")
        overlap = _number(diagnostics.get("hb_base_iou3d"), "HB base IoU")
        containment = _number(diagnostics.get("hb_base_symmetric_containment"), "HB base containment")
        if overlap < 0.20 and containment < 0.70:
            raise F5MergeError(f"{f4_source.get('source_id')} HB overlap proof differs")
    return str(selected), hb_selected


def _validate_buffer(
    value: object,
    *,
    current_ordinal: int,
    label: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > MAX_BUFFERED_FRAMES:
        raise F5MergeError(f"{label} exceeds the three-frame bound")
    rows: list[dict[str, Any]] = []
    last = -1
    for frame in value:
        if not isinstance(frame, Mapping):
            raise F5MergeError(f"{label} row is malformed")
        ordinal = _strict_int(frame.get("frame_ordinal"), f"{label} ordinal")
        source_ids = frame.get("source_ids")
        result_hashes = frame.get("result_sha256")
        if (
            ordinal <= last
            or ordinal > current_ordinal
            or current_ordinal - ordinal > MAX_BUFFERED_FRAMES
            or not isinstance(source_ids, list)
            or not isinstance(result_hashes, list)
            or len(source_ids) != len(result_hashes)
            or len(source_ids) > MAX_SOURCES_PER_FRAME
        ):
            raise F5MergeError(f"{label} causal/source bound differs")
        rows.append(dict(frame))
        last = ordinal
    return rows


def _validate_scene(
    scene_row: Mapping[str, Any],
    *,
    expected_scene_index: int,
    expected_run_signature: str,
    expected_f4_receipt: Mapping[str, str],
    expected_source_receipts: Mapping[str, Mapping[str, str]],
    expected_nonempty_call_index: list[int],
    core: object,
) -> dict[str, Any]:
    scene_id = scene_row.get("scene_id")
    if not isinstance(scene_id, str) or scene_row.get("scene_index") != expected_scene_index:
        raise F5MergeError("F5 manifest scene identity/order differs")
    sidecar_path = _rehash_reference(scene_row.get("sidecar"), f"{scene_id} F5 sidecar", ".json")
    _, scene = _read_json(sidecar_path, f"{scene_id} F5 sidecar")
    if (
        scene.get("schema") != SCENE_SCHEMA
        or scene.get("protocol_id") != PROTOCOL_ID
        or scene.get("complete") is not True
        or scene.get("scene_id") != scene_id
        or scene.get("scene_index") != expected_scene_index
        or scene.get("run_signature_sha256") != expected_run_signature
        or scene.get("contracts") != CONTRACTS
        or scene.get("native_output_mutation_count") != 0
        or scene.get("birth_count") != 0
        or _content_hash_without(scene, "content_sha256") != scene.get("content_sha256")
    ):
        raise F5MergeError(f"{scene_id} F5 scene contract differs")
    inputs = scene.get("inputs")
    if not isinstance(inputs, Mapping) or inputs.get("f4_receipt") != expected_f4_receipt or inputs.get("sources") != expected_source_receipts:
        raise F5MergeError(f"{scene_id} F5 frozen input/source receipts differ")
    f4_path = _rehash_reference(inputs.get("f4_sidecar"), f"{scene_id} F4 sidecar", ".json")
    _, f4 = _read_json(f4_path, f"{scene_id} F4 sidecar")
    if (
        f4.get("schema") != EXPECTED_F4_SCENE_SCHEMA
        or f4.get("protocol_id") != EXPECTED_F4_PROTOCOL
        or f4.get("complete") is not True
        or f4.get("scene_id") != scene_id
        or f4.get("scene_index") != expected_scene_index
        or f4.get("run_signature_sha256") != expected_f4_receipt.get("run_signature_sha256")
        or f4.get("native_output_mutation_count") != 0
        or _content_hash_without(f4, "content_sha256") != f4.get("content_sha256")
    ):
        raise F5MergeError(f"{scene_id} sealed F4 scene differs")
    f4_inputs = f4.get("inputs")
    if not isinstance(f4_inputs, Mapping):
        raise F5MergeError(f"{scene_id} F4 inputs are absent")
    for key in ("f2_sidecar", "f2_evidence", "intrinsic"):
        f5_seal = inputs.get(key)
        f4_seal = f4_inputs.get(key)
        if (
            not isinstance(f5_seal, Mapping)
            or not isinstance(f4_seal, Mapping)
            or {name: f5_seal.get(name) for name in ("path", "sha256")}
            != {name: f4_seal.get(name) for name in ("path", "sha256")}
        ):
            raise F5MergeError(f"{scene_id} F5 {key} lineage differs from F4")
    f2_path = _rehash_reference(inputs.get("f2_sidecar"), f"{scene_id} F2 sidecar", ".json")
    _, f2 = _read_json(f2_path, f"{scene_id} F2 sidecar")
    if f2.get("schema") != EXPECTED_F2_SCENE_SCHEMA or f2.get("complete") is not True or f2.get("scene_id") != scene_id or f2.get("scene_index") != expected_scene_index:
        raise F5MergeError(f"{scene_id} F2 sidecar differs")
    evidence_path = _rehash_reference(inputs.get("f2_evidence"), f"{scene_id} F2 evidence", ".npz")
    intrinsic = _load_matrix(inputs.get("intrinsic"), label=f"{scene_id} intrinsic", shape=(3, 3))
    frames = scene.get("frames")
    f4_frames = f4.get("frames")
    f2_frames = f2.get("frames")
    if not isinstance(frames, list) or not isinstance(f4_frames, list) or not isinstance(f2_frames, list) or len(frames) != len(f4_frames) or len(frames) != len(f2_frames):
        raise F5MergeError(f"{scene_id} frame ledger differs")
    source_count = sum(len(frame.get("sources", ())) for frame in f4_frames if isinstance(frame, Mapping))
    evidence = _Evidence(evidence_path, scene_id=scene_id, source_count=source_count)
    state = core.F5SelectorState()
    source_ids: list[str] = []
    lineage_hashes: list[str] = []
    result_hashes: list[str] = []
    selected_counts = {name: 0 for name in ("H0", "HL", "HLG", "HB")}
    successful_count = 0
    hb_proof_count = 0
    incremental_all: list[float] = []
    incremental_warm: list[float] = []
    composed_all: list[float] = []
    composed_warm: list[float] = []
    deadline_all = 0
    deadline_warm = 0
    max_buffer_frames = 0
    max_buffer_sources = 0
    opened: list[dict[str, Any]] = [
        {"kind": "f4_sidecar", "path": os.fspath(f4_path), "sha256": _sha256(f4_path)},
        {"kind": "f2_sidecar", "path": os.fspath(f2_path), "sha256": _sha256(f2_path)},
        {"kind": "f2_evidence", "path": os.fspath(evidence_path), "sha256": _sha256(evidence_path)},
        {"kind": "intrinsic", "path": os.fspath(_rehash_reference(inputs.get("intrinsic"), f"{scene_id} intrinsic", ".txt")), "sha256": inputs["intrinsic"]["sha256"]},
    ]
    if _canonical_json_sha256(opened) != inputs.get("frozen_base_inputs_sha256"):
        evidence.close()
        raise F5MergeError(f"{scene_id} frozen base-input aggregate differs")
    try:
        for ordinal, (frame, f4_frame, f2_frame) in enumerate(zip(frames, f4_frames, f2_frames)):
            if not isinstance(frame, Mapping) or not isinstance(f4_frame, Mapping) or not isinstance(f2_frame, Mapping):
                raise F5MergeError(f"{scene_id}/{ordinal} frame row is malformed")
            frame_id = f4_frame.get("frame_id")
            successful = f4_frame.get("successful") is True
            if (
                frame.get("frame_ordinal") != ordinal
                or frame.get("frame_id") != frame_id
                or (frame.get("successful") is True) is not successful
                or f2_frame.get("frame_ordinal") != ordinal
                or f2_frame.get("frame_id") != frame_id
                or (f2_frame.get("successful") is True) is not successful
            ):
                raise F5MergeError(f"{scene_id}/{ordinal} frame identity differs")
            f4_sources = f4_frame.get("sources")
            f2_sources = f2_frame.get("sources")
            actual_sources = frame.get("sources")
            if not isinstance(f4_sources, list) or not isinstance(f2_sources, list) or not isinstance(actual_sources, list) or len(f4_sources) != len(f2_sources) or len(f4_sources) != len(actual_sources):
                raise F5MergeError(f"{scene_id}/{frame_id} source partition differs")
            if len(actual_sources) > MAX_SOURCES_PER_FRAME:
                raise F5MergeError(f"{scene_id}/{frame_id} exceeds the 16-source frame cap")
            if not successful:
                if actual_sources or f4_sources or frame.get("query") is not None or frame.get("commit") is not None or frame.get("runtime") is not None or frame.get("buffer_before") is not None or frame.get("buffer_after") is not None or frame.get("maximum_accessed_frame_ordinal") is not None:
                    raise F5MergeError(f"{scene_id}/{frame_id} failed-frame contract differs")
                continue
            successful_count += 1
            pose_reference = f4_frame.get("input", {}).get("pose") if isinstance(f4_frame.get("input"), Mapping) else None
            pose_path = _rehash_reference(pose_reference, f"{scene_id}/{frame_id} pose", ".txt")
            pose = _load_matrix(pose_reference, label=f"{scene_id}/{frame_id} pose", shape=(4, 4))
            opened.append({"kind": "pose", "frame_ordinal": ordinal, "frame_id": frame_id, "path": os.fspath(pose_path), "sha256": pose_reference["sha256"]})
            points = evidence.take(scene_id=scene_id, frame_id=int(frame_id), sources=f4_sources)
            evidence_rows = []
            for rank, (f4_source, f2_source, point_rows) in enumerate(zip(f4_sources, f2_sources, points)):
                if not isinstance(f4_source, Mapping) or not isinstance(f2_source, Mapping):
                    raise F5MergeError(f"{scene_id}/{frame_id}/{rank} source row is malformed")
                source_id = f4_source.get("source_id")
                if (
                    source_id != f2_source.get("source_id")
                    or f4_source.get("rank") != rank
                    or f4_source.get("candidate_index") != rank
                    or f4_source.get("frame_ordinal") != ordinal
                    or f4_source.get("frame_id") != frame_id
                    or {name: f4_source.get("hypotheses", {}).get(name) for name in ("H0", "HL", "HLG")} != f2_source.get("hypotheses")
                ):
                    raise F5MergeError(f"{source_id} F4/F2 source lineage differs")
                try:
                    evidence_rows.append(core.F5SourceEvidence(
                        source_id=source_id,
                        frame_id=int(frame_id),
                        frame_ordinal=ordinal,
                        rank=rank,
                        hypotheses=dict(f4_source["hypotheses"]),
                        points_world=point_rows,
                        tight_box_xyxy=f4_source.get("tight_box_xyxy"),
                        camera_to_world=pose,
                        intrinsic=intrinsic,
                        source_lineage_sha256=f4_source.get("source_lineage_sha256"),
                    ))
                except (TypeError, ValueError, RuntimeError) as error:
                    raise F5MergeError(f"{source_id} audit replay rejected sealed evidence") from error
            try:
                query = state.query_frame(frame_id=int(frame_id), frame_ordinal=ordinal, sources=tuple(evidence_rows))
            except (TypeError, ValueError, RuntimeError) as error:
                raise F5MergeError(f"{scene_id}/{frame_id} audit query failed") from error
            query_row = frame.get("query")
            commit_row = frame.get("commit")
            if not isinstance(query_row, Mapping) or not isinstance(commit_row, Mapping):
                raise F5MergeError(f"{scene_id}/{frame_id} query/commit receipt is absent")
            expected_before = [dict(row) for row in query.buffer_before]
            actual_before = _validate_buffer(frame.get("buffer_before"), current_ordinal=ordinal, label=f"{scene_id}/{frame_id} buffer_before")
            if actual_before != expected_before or query_row.get("buffer_before") != expected_before:
                raise F5MergeError(f"{scene_id}/{frame_id} causal buffer-before differs")
            if frame.get("maximum_accessed_frame_ordinal") != query.maximum_accessed_frame_ordinal or query_row.get("maximum_accessed_frame_ordinal") != query.maximum_accessed_frame_ordinal or query_row.get("maximum_lookahead_frames") != 0 or query_row.get("query_before_commit") is not True:
                raise F5MergeError(f"{scene_id}/{frame_id} query causality differs")
            if query.maximum_accessed_frame_ordinal >= ordinal:
                raise F5MergeError(f"{scene_id}/{frame_id} future-frame access detected")
            if len(query.rows) != len(actual_sources):
                raise F5MergeError(f"{scene_id}/{frame_id} audit replay source count differs")
            max_buffer_frames = max(max_buffer_frames, len(actual_before))
            max_buffer_sources = max(max_buffer_sources, *(len(row["source_ids"]) for row in actual_before), 0)
            for actual, expected, f4_source in zip(actual_sources, query.rows, f4_sources):
                if not isinstance(actual, Mapping):
                    raise F5MergeError(f"{scene_id}/{frame_id} selector result is malformed")
                selected, hb_selected = _verify_selected_row(actual, expected, f4_source, buffer_before=actual_before)
                selected_counts[selected] += 1
                hb_proof_count += int(hb_selected)
                source_ids.append(str(actual["source_id"]))
                lineage_hashes.append(str(actual["source_lineage_sha256"]))
                result_hashes.append(str(actual["result_sha256"]))
            try:
                commit = state.commit_frame(query)
            except (TypeError, ValueError, RuntimeError) as error:
                raise F5MergeError(f"{scene_id}/{frame_id} audit commit failed") from error
            expected_after = [dict(row) for row in commit.buffer_after]
            actual_after = _validate_buffer(frame.get("buffer_after"), current_ordinal=ordinal, label=f"{scene_id}/{frame_id} buffer_after")
            if (
                actual_after != expected_after
                or commit_row.get("buffer_after") != expected_after
                or commit_row.get("source_count") != len(actual_sources)
                or query_row.get("token") != query.token
                or commit_row.get("token") != commit.token
            ):
                raise F5MergeError(f"{scene_id}/{frame_id} query/commit ledger differs")
            max_buffer_frames = max(max_buffer_frames, len(actual_after))
            max_buffer_sources = max(max_buffer_sources, *(len(row["source_ids"]) for row in actual_after), 0)

            runtime = frame.get("runtime")
            if not isinstance(runtime, Mapping):
                raise F5MergeError(f"{scene_id}/{frame_id} runtime is absent")
            nonempty = bool(actual_sources)
            expected_call = expected_nonempty_call_index[0] if nonempty else None
            warmup = bool(nonempty and expected_call < WARMUP_NONEMPTY_FRAMES)
            if nonempty:
                expected_nonempty_call_index[0] += 1
            if runtime.get("nonempty_call_index_in_shard") != expected_call or runtime.get("f5_warmup_excluded") is not warmup:
                raise F5MergeError(f"{scene_id}/{frame_id} warmup/call ledger differs")
            incremental = _number(runtime.get("f5_incremental_ms"), "F5 incremental runtime")
            if nonempty:
                inherited = _number(f4_frame.get("runtime", {}).get("replay_composed_ms"), "sealed F4 composed runtime")
            else:
                inherited = _number(f2_frame.get("runtime", {}).get("complete_ms", 0.0), "sealed F2 complete runtime")
            composed = _number(runtime.get("replay_composed_ms"), "F5 composed runtime")
            if not math.isclose(_number(runtime.get("sealed_f4_composed_ms"), "sealed inherited runtime"), inherited, abs_tol=1.0e-9, rel_tol=0.0) or not math.isclose(composed, inherited + incremental, abs_tol=1.0e-9, rel_tol=0.0) or not math.isclose(_number(runtime.get("replay_composed_ms_per_source_frame"), "amortized runtime"), composed / SOURCE_FRAME_STRIDE, abs_tol=1.0e-9, rel_tol=0.0):
                raise F5MergeError(f"{scene_id}/{frame_id} runtime arithmetic differs")
            missed = composed >= MAX_COMPOSED_MS_EXCLUSIVE
            missed_warm = (not warmup) and missed
            if runtime.get("gap25_deadline_missed") is not missed or runtime.get("gap25_deadline_missed_warm") is not missed_warm:
                raise F5MergeError(f"{scene_id}/{frame_id} deadline ledger differs")
            if runtime.get("f5_cuda_allocated_bytes") != 0:
                raise F5MergeError(f"{scene_id}/{frame_id} F5 allocated CUDA memory")
            incremental_all.append(incremental)
            composed_all.append(composed)
            deadline_all += int(missed)
            deadline_warm += int(missed_warm)
            if not warmup:
                incremental_warm.append(incremental)
                composed_warm.append(composed)
        evidence.finish(source_count, scene_id)
    finally:
        evidence.close()

    if len(source_ids) != len(set(source_ids)):
        raise F5MergeError(f"{scene_id} source identities are duplicated")
    opened_hash = _canonical_json_sha256(opened)
    if inputs.get("all_opened_inputs_before_sha256") != opened_hash or inputs.get("all_opened_inputs_after_sha256") != opened_hash:
        raise F5MergeError(f"{scene_id} opened-input before/after seal differs")
    expected_counts = {
        "keyframe_count": len(frames),
        "successful_frame_count": successful_count,
        "source_count": len(source_ids),
        "identity_verified_source_count": len(source_ids),
        "selected_h0_count": selected_counts["H0"],
        "selected_hl_count": selected_counts["HL"],
        "selected_hlg_count": selected_counts["HLG"],
        "selected_hb_count": selected_counts["HB"],
    }
    if scene.get("counts") != expected_counts or scene_row.get("counts") != expected_counts:
        raise F5MergeError(f"{scene_id} count/selection census differs")
    source_ids_hash = _canonical_json_sha256(source_ids)
    lineage_hash = _canonical_json_sha256(lineage_hashes)
    result_hash = _canonical_json_sha256(result_hashes)
    if scene.get("source_ids_sha256") != source_ids_hash or scene_row.get("source_ids_sha256") != source_ids_hash or scene.get("source_lineage_sha256") != lineage_hash or scene_row.get("source_lineage_sha256") != lineage_hash or scene.get("result_ledger_sha256") != result_hash or scene_row.get("result_ledger_sha256") != result_hash:
        raise F5MergeError(f"{scene_id} ordered source/result ledger differs")
    prefix_successes = successful_count // 2
    prefix_hashes: list[str] = []
    successes_seen = 0
    for frame in frames:
        if frame.get("successful") is True:
            if successes_seen >= prefix_successes:
                break
            successes_seen += 1
            prefix_hashes.extend(str(row["result_sha256"]) for row in frame["sources"])
    expected_prefix = {
        "passed": True,
        "successful_frame_count": prefix_successes,
        "result_row_count": len(prefix_hashes),
        "result_ledger_sha256": _canonical_json_sha256(prefix_hashes),
    }
    expected_determinism = {
        "passed": True,
        "independent_replay_count": 1,
        "online_result_ledger_sha256": result_hash,
        "independent_result_ledger_sha256": result_hash,
    }
    if scene.get("prefix_replay") != expected_prefix or scene_row.get("prefix_replay") != expected_prefix or scene.get("determinism") != expected_determinism or scene_row.get("determinism") != expected_determinism:
        raise F5MergeError(f"{scene_id} prefix/determinism proof differs")
    expected_causality = {
        "overall_pass": True,
        "query_before_commit": True,
        "prefix_replay_pass": True,
        "independent_replay_pass": True,
        "maximum_lookahead_frames": 0,
        "maximum_accessed_past_frame_ordinal": max((frame.get("maximum_accessed_frame_ordinal", -1) for frame in frames if frame.get("successful") is True), default=-1),
        "future_access_count": 0,
        "current_source_offsets_only": True,
        "prefix_successful_frame_count": prefix_successes,
    }
    if scene.get("causality") != expected_causality or scene_row.get("causality") != expected_causality:
        raise F5MergeError(f"{scene_id} causality summary differs")
    inherited_cuda_peak = int(f4.get("runtime", {}).get("cuda_peak_memory_bytes", 0))
    expected_runtime = {
        "f5_incremental_all_ms": _distribution(incremental_all),
        "f5_incremental_warm_ms": _distribution(incremental_warm),
        "replay_composed_all_ms": _distribution(composed_all),
        "replay_composed_warm_ms": _distribution(composed_warm),
        "replay_composed_warm_mean_per_source_frame_ms": (float(np.mean(composed_warm)) / SOURCE_FRAME_STRIDE if composed_warm else 0.0),
        "gap25_all_deadline_miss_count": deadline_all,
        "gap25_warm_deadline_miss_count": deadline_warm,
        "f5_cuda_peak_memory_bytes": 0,
        "inherited_f4_cuda_peak_memory_bytes": inherited_cuda_peak,
        "cuda_peak_memory_bytes": inherited_cuda_peak,
    }
    if scene.get("runtime") != expected_runtime or scene_row.get("runtime") != expected_runtime:
        raise F5MergeError(f"{scene_id} runtime summary differs")
    return {
        "scene_row": dict(scene_row),
        "counts": expected_counts,
        "source_ids": source_ids,
        "lineages": lineage_hashes,
        "results": result_hashes,
        "selected_hb": selected_counts["HB"],
        "hb_proof_count": hb_proof_count,
        "incremental_all": incremental_all,
        "incremental_warm": incremental_warm,
        "composed_all": composed_all,
        "composed_warm": composed_warm,
        "deadline_all": deadline_all,
        "deadline_warm": deadline_warm,
        "cuda_peak": expected_runtime["cuda_peak_memory_bytes"],
        "max_buffer_frames": max_buffer_frames,
        "max_buffer_sources": max_buffer_sources,
    }


def merge_f5(
    *,
    shard_paths: Sequence[Path] = DEFAULT_SHARDS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    expected_scene_count: int = EXPECTED_SCENES,
    expected_keyframes: int | None = None,
    expected_successful_frames: int | None = None,
    expected_sources: int | None = None,
    min_selected_hb_sources: int | None = None,
    min_selected_hb_scenes: int | None = None,
    max_selected_hb_fraction: float = MAX_SELECTED_HB_FRACTION,
) -> dict[str, Any]:
    """Validate exactly two F5 shards and publish one create-only receipt."""

    if len(shard_paths) != EXPECTED_SHARDS:
        raise F5MergeError("F5 merge requires exactly two shard manifests")
    production = expected_scene_count == EXPECTED_SCENES
    expected_keyframes = EXPECTED_KEYFRAMES if production and expected_keyframes is None else expected_keyframes
    expected_successful_frames = EXPECTED_SUCCESSFUL_FRAMES if production and expected_successful_frames is None else expected_successful_frames
    expected_sources = EXPECTED_SOURCES if production and expected_sources is None else expected_sources
    min_selected_hb_sources = MIN_SELECTED_HB_SOURCES if min_selected_hb_sources is None else min_selected_hb_sources
    min_selected_hb_scenes = MIN_SELECTED_HB_SCENES if min_selected_hb_scenes is None else min_selected_hb_scenes
    if production and (min_selected_hb_sources != MIN_SELECTED_HB_SOURCES or min_selected_hb_scenes != MIN_SELECTED_HB_SCENES or max_selected_hb_fraction != MAX_SELECTED_HB_FRACTION):
        raise F5MergeError("production F5 coverage gates are frozen")
    if _sha256(_regular_file(PROTOCOL_PATH, "F5 frozen protocol", ".md")) != PROTOCOL_SHA256:
        raise F5MergeError("F5 protocol hash differs")
    merge_source_path = _regular_file(Path(__file__).resolve(), "F5 merge source", ".py")
    merge_source_seal = {
        "path": os.fspath(merge_source_path),
        "sha256": _sha256(merge_source_path),
    }

    loaded: list[tuple[Path, dict[str, Any]]] = []
    for shard_index, shard_path in enumerate(shard_paths):
        path, shard = _read_json(Path(shard_path), f"F5 shard {shard_index}")
        if (
            shard.get("schema") != SHARD_SCHEMA
            or shard.get("protocol_id") != PROTOCOL_ID
            or shard.get("complete") is not True
            or shard.get("shard_index") != shard_index
            or shard.get("num_shards") != EXPECTED_SHARDS
            or shard.get("contracts") != CONTRACTS
            or shard.get("native_output_mutation_count") != 0
            or shard.get("birth_count") != 0
            or _content_hash_without(shard, "content_sha256") != shard.get("content_sha256")
        ):
            raise F5MergeError(f"F5 shard {shard_index} contract/content differs")
        loaded.append((path, shard))
    left, right = loaded[0][1], loaded[1][1]
    for key in ("run_signature_sha256", "signature_payload_sha256"):
        if not isinstance(left.get(key), str) or left.get(key) != right.get(key):
            raise F5MergeError(f"F5 shard shared {key} differs")
    if left.get("inputs") != right.get("inputs"):
        raise F5MergeError("F5 shard frozen inputs differ")
    inputs = left.get("inputs")
    if not isinstance(inputs, Mapping):
        raise F5MergeError("F5 shard inputs are absent")
    f4_receipt_path = _rehash_reference(inputs.get("f4_receipt"), "sealed F4 merge", ".json")
    _, f4_receipt = _read_json(f4_receipt_path, "sealed F4 merge")
    if (
        f4_receipt.get("schema") != EXPECTED_F4_MERGE_SCHEMA
        or f4_receipt.get("protocol_id") != EXPECTED_F4_PROTOCOL
        or f4_receipt.get("complete") is not True
        or f4_receipt.get("overall_pass") is not True
        or f4_receipt.get("native_output_mutation_count") != 0
        or _content_hash_without(f4_receipt, "content_sha256") != f4_receipt.get("content_sha256")
    ):
        raise F5MergeError("sealed F4 merge contract differs")
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
    f4_contracts = f4_receipt.get("contracts")
    f4_run_signature = f4_receipt.get("run_signature_sha256")
    if (
        not isinstance(f4_contracts, Mapping)
        or any(f4_contracts.get(key) is not value for key, value in required_f4_contracts.items())
        or not isinstance(f4_run_signature, str)
        or len(f4_run_signature) != 64
        or inputs["f4_receipt"].get("run_signature_sha256") != f4_run_signature
    ):
        raise F5MergeError("sealed F4 provenance/forbidden-access contract differs")
    scene_list_path = _rehash_reference(inputs.get("scene_list"), "sealed paper100 scene list", ".txt")
    scene_order = [line.strip() for line in scene_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(scene_order) != expected_scene_count or len(set(scene_order)) != len(scene_order):
        raise F5MergeError("scene-list count/order differs")
    f4_scene_rows = f4_receipt.get("scenes")
    f4_coverage = f4_receipt.get("coverage")
    f4_totals = f4_receipt.get("totals")
    if (
        not isinstance(f4_scene_rows, list)
        or len(f4_scene_rows) != expected_scene_count
        or not isinstance(f4_coverage, Mapping)
        or f4_coverage.get("scene_count") != expected_scene_count
        or f4_coverage.get("scene_order") != scene_order
        or not isinstance(f4_totals, Mapping)
    ):
        raise F5MergeError("F4 scene ledger differs")
    for key, expected in (
        ("keyframe_count", expected_keyframes),
        ("successful_frame_count", expected_successful_frames),
        ("source_count", expected_sources),
    ):
        if expected is not None and f4_totals.get(key) != expected:
            raise F5MergeError(f"sealed F4 {key} differs")
    source_receipts = inputs.get("sources")
    if not isinstance(source_receipts, Mapping) or set(source_receipts) != {"runner", "core", "protocol"}:
        raise F5MergeError("F5 source receipt set differs")
    for name, seal in source_receipts.items():
        _rehash_reference(seal, f"F5 frozen {name}")
    expected_source_paths = {
        "runner": REPOSITORY_ROOT / "tools/run_scannet_fastsam_f5_selector_paper100.py",
        "core": REPOSITORY_ROOT / "boxfusion/fastsam_f5_selector.py",
        "protocol": PROTOCOL_PATH,
    }
    for name, expected_path in expected_source_paths.items():
        if Path(source_receipts[name]["path"]).resolve() != expected_path.resolve():
            raise F5MergeError(f"F5 frozen {name} path differs")
    if source_receipts["protocol"].get("sha256") != PROTOCOL_SHA256:
        raise F5MergeError("shard protocol receipt differs from frozen protocol")
    try:
        from boxfusion import fastsam_f5_selector as core
    except ImportError as error:  # pragma: no cover
        raise F5MergeError("F5 selector audit core is unavailable") from error
    if getattr(core, "PROTOCOL_ID", None) != PROTOCOL_ID:
        raise F5MergeError("F5 selector core protocol differs")
    signature_payload = {
        "protocol_id": PROTOCOL_ID,
        "f4_receipt": dict(inputs["f4_receipt"]),
        "scene_order": scene_order,
        "scene_list_sha256": inputs["scene_list"]["sha256"],
        "core_schema": getattr(core, "SCHEMA", None),
        "core_policy": dict(getattr(core, "POLICY", {})),
        "sources": {key: dict(value) for key, value in source_receipts.items()},
        "contracts": dict(CONTRACTS),
        "num_shards": EXPECTED_SHARDS,
    }
    if _canonical_json_sha256(signature_payload) != left.get("run_signature_sha256") or _canonical_json_sha256(signature_payload) != left.get("signature_payload_sha256"):
        raise F5MergeError("F5 run signature differs")

    rows_by_index: dict[int, tuple[int, Mapping[str, Any]]] = {}
    for shard_index, (_, shard) in enumerate(loaded):
        rows = shard.get("scenes")
        if not isinstance(rows, list):
            raise F5MergeError(f"F5 shard {shard_index} scene rows are absent")
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("scene_index"), int):
                raise F5MergeError(f"F5 shard {shard_index} scene row is malformed")
            scene_index = row["scene_index"]
            if scene_index % EXPECTED_SHARDS != shard_index or scene_index in rows_by_index:
                raise F5MergeError("F5 deterministic scene partition differs")
            rows_by_index[scene_index] = (shard_index, row)
    if sorted(rows_by_index) != list(range(expected_scene_count)):
        raise F5MergeError("merged F5 scene partition is incomplete")

    totals = {key: 0 for key in (
        "keyframe_count", "successful_frame_count", "source_count", "identity_verified_source_count",
        "selected_h0_count", "selected_hl_count", "selected_hlg_count", "selected_hb_count",
    )}
    source_ids: list[str] = []
    lineages: list[str] = []
    results: list[str] = []
    incremental_all: list[float] = []
    incremental_warm: list[float] = []
    composed_all: list[float] = []
    composed_warm: list[float] = []
    deadline_all = 0
    deadline_warm = 0
    cuda_peak = 0
    max_buffer_frames = 0
    max_buffer_sources = 0
    hb_scenes = 0
    hb_proof_count = 0
    scene_rows: list[dict[str, Any]] = []
    validated_scenes: list[dict[str, Any]] = []
    call_indices = [[0], [0]]
    for scene_index in range(expected_scene_count):
        shard_index, row = rows_by_index[scene_index]
        f4_row = f4_scene_rows[scene_index]
        if not isinstance(f4_row, Mapping) or f4_row.get("scene_id") != scene_order[scene_index] or row.get("scene_id") != scene_order[scene_index] or f4_row.get("scene_index") != scene_index:
            raise F5MergeError("F4/F5 scene identity differs")
        if row.get("sidecar") is None or _rehash_reference(row.get("sidecar"), "F5 scene sidecar", ".json") == _rehash_reference(f4_row.get("sidecar"), "F4 scene sidecar", ".json"):
            raise F5MergeError("F5 sidecar cannot alias its F4 input")
        validated = _validate_scene(
            row,
            expected_scene_index=scene_index,
            expected_run_signature=str(left["run_signature_sha256"]),
            expected_f4_receipt=inputs["f4_receipt"],
            expected_source_receipts=source_receipts,
            expected_nonempty_call_index=call_indices[shard_index],
            core=core,
        )
        if (
            f4_row.get("source_ids_sha256") != _canonical_json_sha256(validated["source_ids"])
            or f4_row.get("source_lineage_sha256") != _canonical_json_sha256(validated["lineages"])
            or f4_row.get("counts", {}).get("source_count") != validated["counts"]["source_count"]
        ):
            raise F5MergeError(f"{row.get('scene_id')} F5 source ledger differs from sealed F4 merge")
        for key in totals:
            totals[key] += validated["counts"][key]
        source_ids.extend(validated["source_ids"])
        lineages.extend(validated["lineages"])
        results.extend(validated["results"])
        incremental_all.extend(validated["incremental_all"])
        incremental_warm.extend(validated["incremental_warm"])
        composed_all.extend(validated["composed_all"])
        composed_warm.extend(validated["composed_warm"])
        deadline_all += validated["deadline_all"]
        deadline_warm += validated["deadline_warm"]
        cuda_peak = max(cuda_peak, validated["cuda_peak"])
        max_buffer_frames = max(max_buffer_frames, validated["max_buffer_frames"])
        max_buffer_sources = max(max_buffer_sources, validated["max_buffer_sources"])
        hb_scenes += int(validated["selected_hb"] > 0)
        hb_proof_count += validated["hb_proof_count"]
        scene_rows.append(validated["scene_row"])
        validated_scenes.append(validated)

    if len(source_ids) != len(set(source_ids)):
        raise F5MergeError("global F5 source identities are duplicated")
    expected_values = {
        "keyframe_count": expected_keyframes,
        "successful_frame_count": expected_successful_frames,
        "source_count": expected_sources,
    }
    for key, expected in expected_values.items():
        if expected is not None and totals[key] != expected:
            raise F5MergeError(f"merged F5 {key} differs")
    if totals["identity_verified_source_count"] != totals["source_count"] or sum(totals[key] for key in ("selected_h0_count", "selected_hl_count", "selected_hlg_count", "selected_hb_count")) != totals["source_count"] or hb_proof_count != totals["selected_hb_count"]:
        raise F5MergeError("one-source/one-selection or HB proof census differs")
    if production:
        for shard_index, (_, shard) in enumerate(loaded):
            for key, expected in EXPECTED_SHARD_COUNTS[shard_index].items():
                if shard.get("totals", {}).get(key) != expected:
                    raise F5MergeError(f"production shard {shard_index} {key} differs")
    for shard_index, (_, shard) in enumerate(loaded):
        expected_shard_totals = {
            key: sum(scene_rows[index]["counts"][key] for index in range(shard_index, expected_scene_count, EXPECTED_SHARDS))
            for key in totals
        }
        if shard.get("totals") != expected_shard_totals:
            raise F5MergeError(f"F5 shard {shard_index} totals differ")
        expected_scene_result_hash = _canonical_json_sha256([scene_rows[index]["result_ledger_sha256"] for index in range(shard_index, expected_scene_count, EXPECTED_SHARDS)])
        expected_determinism = {
            "overall_pass": True,
            "prefix_replay_pass": True,
            "independent_replay_pass": True,
            "scene_result_ledger_sha256": expected_scene_result_hash,
        }
        if shard.get("determinism") != expected_determinism:
            raise F5MergeError(f"F5 shard {shard_index} determinism summary differs")
        expected_causality = {
            "overall_pass": True,
            "query_before_commit": True,
            "prefix_replay_pass": True,
            "independent_replay_pass": True,
            "maximum_lookahead_frames": 0,
            "future_access_count": 0,
            "current_source_offsets_only": True,
        }
        if shard.get("causality") != expected_causality:
            raise F5MergeError(f"F5 shard {shard_index} causality summary differs")
        assigned = validated_scenes[shard_index::EXPECTED_SHARDS]
        shard_incremental_warm = [value for scene in assigned for value in scene["incremental_warm"]]
        shard_composed_warm = [value for scene in assigned for value in scene["composed_warm"]]
        shard_inherited_peak = max((scene["cuda_peak"] for scene in assigned), default=0)
        expected_runtime = {
            "f5_incremental_warm_ms": _distribution(shard_incremental_warm),
            "replay_composed_warm_ms": _distribution(shard_composed_warm),
            "replay_composed_warm_mean_per_source_frame_ms": (
                float(np.mean(shard_composed_warm)) / SOURCE_FRAME_STRIDE
                if shard_composed_warm else 0.0
            ),
            "gap25_all_deadline_miss_count": sum(scene["deadline_all"] for scene in assigned),
            "gap25_warm_deadline_miss_count": sum(scene["deadline_warm"] for scene in assigned),
            "f5_cuda_peak_memory_bytes": 0,
            "inherited_f4_cuda_peak_memory_bytes": shard_inherited_peak,
            "cuda_peak_memory_bytes": shard_inherited_peak,
            "warmup_nonempty_frame_count": min(WARMUP_NONEMPTY_FRAMES, call_indices[shard_index][0]),
        }
        if shard.get("runtime") != expected_runtime:
            raise F5MergeError(f"F5 shard {shard_index} runtime summary differs")

    incremental_distribution = _distribution(incremental_warm)
    composed_distribution = _distribution(composed_warm)
    all_incremental_distribution = _distribution(incremental_all)
    all_composed_distribution = _distribution(composed_all)
    composed_mean_per_source_frame = float(composed_distribution["mean"]) / SOURCE_FRAME_STRIDE
    hb_ratio = totals["selected_hb_count"] / totals["source_count"] if totals["source_count"] else 0.0
    gates = {
        "integrity_complete": _gate(len(scene_rows), "==", expected_scene_count),
        "exact_keyframes": _gate(totals["keyframe_count"], "==", expected_keyframes if expected_keyframes is not None else totals["keyframe_count"]),
        "exact_successful_frames": _gate(totals["successful_frame_count"], "==", expected_successful_frames if expected_successful_frames is not None else totals["successful_frame_count"]),
        "exact_unique_sources": _gate(totals["source_count"], "==", expected_sources if expected_sources is not None else totals["source_count"]),
        "identity_verified_sources": _gate(totals["identity_verified_source_count"], "==", totals["source_count"]),
        "one_selection_per_source": _gate(sum(totals[key] for key in ("selected_h0_count", "selected_hl_count", "selected_hlg_count", "selected_hb_count")), "==", totals["source_count"]),
        "selected_hb_proof_count": _gate(hb_proof_count, "==", totals["selected_hb_count"]),
        "selected_hb_min_sources": _gate(totals["selected_hb_count"], ">=", min_selected_hb_sources),
        "selected_hb_min_scenes": _gate(hb_scenes, ">=", min_selected_hb_scenes),
        "selected_hb_max_fraction": _gate(hb_ratio, "<=", max_selected_hb_fraction),
        "prefix_replay": _gate(0, "==", 0),
        "independent_cpu_replay": _gate(0, "==", 0),
        "maximum_lookahead_frames": _gate(0, "==", 0),
        "maximum_buffered_frames": _gate(max_buffer_frames, "<=", MAX_BUFFERED_FRAMES),
        "maximum_sources_per_buffered_frame": _gate(max_buffer_sources, "<=", MAX_SOURCES_PER_FRAME),
        "native_output_mutation_count": _gate(0, "==", 0),
        "source_addition_or_removal_count": _gate(0, "==", 0),
        "score_rank_semantic_mutation_count": _gate(0, "==", 0),
        "forbidden_access_count": _gate(0, "==", 0),
        "training_or_online_learning_count": _gate(0, "==", 0),
        "birth_count": _gate(0, "==", 0),
        "f5_incremental_warm_p95_ms": _gate(float(incremental_distribution["p95"]), "<=", MAX_F5_INCREMENTAL_P95_MS),
        "replay_composed_warm_p95_ms": _gate(float(composed_distribution["p95"]), "<=", MAX_COMPOSED_P95_MS),
        "replay_composed_warm_max_ms": _gate(float(composed_distribution["max"]), "<", MAX_COMPOSED_MS_EXCLUSIVE),
        "replay_composed_warm_mean_per_source_frame_ms": _gate(composed_mean_per_source_frame, "<=", MAX_COMPOSED_MEAN_PER_SOURCE_FRAME_MS),
        "gap25_warm_deadline_miss_count": _gate(deadline_warm, "==", 0),
        "cuda_peak_memory_bytes": _gate(cuda_peak, "<=", MAX_CUDA_PEAK_BYTES),
        "f5_cuda_allocated_bytes": _gate(0, "==", 0),
    }
    runtime_gate_names = (
        "f5_incremental_warm_p95_ms", "replay_composed_warm_p95_ms",
        "replay_composed_warm_max_ms", "replay_composed_warm_mean_per_source_frame_ms",
        "gap25_warm_deadline_miss_count", "cuda_peak_memory_bytes", "f5_cuda_allocated_bytes",
    )
    runtime_pass = all(gates[name]["pass"] for name in runtime_gate_names)
    noncoverage_gate_names = tuple(name for name in gates if name not in {"selected_hb_min_sources", "selected_hb_min_scenes", "selected_hb_max_fraction"})
    noncoverage_pass = all(gates[name]["pass"] for name in noncoverage_gate_names)
    overall_pass = all(gate["pass"] for gate in gates.values())
    if not noncoverage_pass:
        decision = "discard_f5_selector"
    elif not gates["selected_hb_max_fraction"]["pass"]:
        decision = "stop_f5_overbroad_hb"
    elif not gates["selected_hb_min_sources"]["pass"] or not gates["selected_hb_min_scenes"]["pass"]:
        decision = "stop_f5_insufficient_confirmed_hb"
    else:
        decision = "retain_f5_for_one_separately_sealed_evaluation_only"

    receipt: dict[str, Any] = {
        "schema": MERGE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "complete": True,
        "overall_pass": overall_pass,
        "decision": decision,
        "run_signature_sha256": left["run_signature_sha256"],
        "contracts": dict(CONTRACTS),
        "inputs": {
            "shards": [{"path": os.fspath(path), "sha256": _sha256(path), "shard_index": index} for index, (path, _) in enumerate(loaded)],
            "f4_receipt": dict(inputs["f4_receipt"]),
            "scene_list": dict(inputs["scene_list"]),
            "sources": {key: dict(value) for key, value in source_receipts.items()},
            "merge_source": dict(merge_source_seal),
        },
        "coverage": {
            "scene_count": len(scene_rows),
            "scene_order": scene_order,
            "keyframe_count": totals["keyframe_count"],
            "successful_frame_count": totals["successful_frame_count"],
            "source_count": totals["source_count"],
            "exact_source_partition": True,
            "exact_source_order": True,
            "source_ids_sha256": _canonical_json_sha256(source_ids),
            "source_lineage_sha256": _canonical_json_sha256(lineages),
            "result_ledger_sha256": _canonical_json_sha256(results),
        },
        "selection": {
            "selected_h0_count": totals["selected_h0_count"],
            "selected_hl_count": totals["selected_hl_count"],
            "selected_hlg_count": totals["selected_hlg_count"],
            "selected_hb_count": totals["selected_hb_count"],
            "selected_hb_scene_count": hb_scenes,
            "selected_hb_fraction": hb_ratio,
            "selected_hb_complete_proof_count": hb_proof_count,
            "formal_score": 1.0,
        },
        "causality": {
            "overall_pass": True,
            "query_before_commit": True,
            "prefix_replay_pass": True,
            "maximum_lookahead_frames": 0,
            "future_access_count": 0,
            "maximum_buffered_frames": max_buffer_frames,
            "maximum_sources_per_buffered_frame": max_buffer_sources,
        },
        "determinism": {
            "overall_pass": True,
            "prefix_replay_pass": True,
            "independent_cpu_replay_pass": True,
            "audit_replay_pass": True,
            "result_ledger_sha256": _canonical_json_sha256(results),
        },
        "totals": totals,
        "runtime": {
            "overall_pass": runtime_pass,
            "gates": {name: gates[name] for name in runtime_gate_names},
            "f5_incremental_all_ms": all_incremental_distribution,
            "f5_incremental_warm_ms": incremental_distribution,
            "replay_composed_all_ms": all_composed_distribution,
            "replay_composed_warm_ms": composed_distribution,
            "replay_composed_warm_mean_per_source_frame_ms": composed_mean_per_source_frame,
            "gap25_all_deadline_miss_count": deadline_all,
            "gap25_warm_deadline_miss_count": deadline_warm,
            "cuda_peak_memory_bytes": cuda_peak,
            "f5_cuda_allocated_bytes": 0,
            "warmup_nonempty_frame_count_per_shard": [min(WARMUP_NONEMPTY_FRAMES, value[0]) for value in call_indices],
            "all_frame_deadline_misses_are_diagnostic_only": True,
        },
        "gates": gates,
        "scenes": scene_rows,
        "native_output_mutation_count": 0,
        "source_addition_or_removal_count": 0,
        "score_rank_semantic_mutation_count": 0,
        "forbidden_access_count": 0,
        "training_or_online_learning_count": 0,
        "birth_count": 0,
        "evaluation_authorization": {
            "allowed": overall_pass,
            "scope": "one_separately_sealed_constant_score_geometry_evaluation_only",
            "birth_authorized": False,
            "deployment_authorized": False,
        },
    }
    if _sha256(merge_source_path) != merge_source_seal["sha256"]:
        raise F5MergeError("F5 merge source changed during merge")
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    shards = tuple(args.shards) if args.shards is not None else DEFAULT_SHARDS
    result = merge_f5(shard_paths=shards, output_dir=args.output_dir)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
