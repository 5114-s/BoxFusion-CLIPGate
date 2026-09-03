#!/usr/bin/env python3
"""Replay sealed F2/H0 evidence through the causal F3 projection shadow.

The runner never invokes FastSAM and never reads RGB, ground truth, native
predictions, an evaluator, labels, or CLIP features.  It authenticates the
create-only F2 paper100 sidecars/NPZ evidence and the corresponding F0 camera
receipts, then exposes each source to the F3 core only at its sealed frame
ordinal.  F3 remains an observer: internal tracks and B/C hypotheses cannot
create or mutate a detector output.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPOSITORY_ROOT / "tools"
for _root in (REPOSITORY_ROOT, TOOLS_ROOT):
    if os.fspath(_root) not in sys.path:
        sys.path.insert(0, os.fspath(_root))

from boxfusion import fastsam_openbox_f3_shadow as f3_core  # noqa: E402
import run_scannet_fastsam_f0_full200 as f0_runner  # noqa: E402
import run_scannet_fastsam_f2_paper100 as f2_runner  # noqa: E402


PROTOCOL_ID = "F3-FASTSAM-OPENBOX-PROJECTION-SHADOW-PAPER100"
SCENE_SCHEMA = "boxfusion.scannet_fastsam_f3_openbox.scene.v1"
SHARD_SCHEMA = "boxfusion.scannet_fastsam_f3_openbox.shard.v1"
EXPECTED_F2_MERGE_SCHEMA = "boxfusion.scannet_fastsam_f2_paper100.merge.v1"
EXPECTED_F2_PROTOCOL_ID = "F2-DFU-LGF-lite-shadow-paper100"
EXPECTED_F2_EVIDENCE_SCHEMA = (
    "boxfusion.scannet_fastsam_f2_paper100.evidence.v1"
)
EXPECTED_F2_ORACLE_SCHEMA = "boxfusion.scannet_fastsam_f2_paper100_oracle.v1"
EXPECTED_F2_RECEIPT_SHA256 = (
    "455c0e36e35a30c7ba5915384e4d159a730a47b3368bf4b3fb6a5f6064f25603"
)
EXPECTED_F2_ORACLE_SHA256 = (
    "2c3d73f777331617c798aca5e6fdcf819a0267b7d698bdab88f70f7b72dbaff5"
)
EXPECTED_SCENES = 100
EXPECTED_KEYFRAMES = 6_817
EXPECTED_SUCCESSFUL_FRAMES = 6_726
EXPECTED_SOURCES = 52_299
EXPECTED_SHARD_COUNTS: Mapping[int, Mapping[str, int]] = {
    # These names intentionally match the public per-scene ``counts`` schema.
    # The upstream F2 receipt uses the shorter names; F3 normalizes them when
    # sealing a scene, so comparing those F2 names here would read missing
    # Counter entries as zero and reject an otherwise exact census.
    0: {
        "keyframe_count": 3_259,
        "successful_frame_count": 3_189,
        "source_count": 24_863,
    },
    1: {
        "keyframe_count": 3_558,
        "successful_frame_count": 3_537,
        "source_count": 27_436,
    },
}
MASK_SHAPE = (480, 640)
MASK_PACKED_BYTES = MASK_SHAPE[0] * MASK_SHAPE[1] // 8
F3_VOXEL_SIZE_M = 0.05
F3_MAX_VOXELS_PER_OBSERVATION = 512
SOURCE_FRAME_STRIDE = 25.0

DEFAULT_F2_RECEIPT = (
    REPOSITORY_ROOT
    / "logs/scannet_fastsam_f2_paper100_score05/final/F2_FASTSAM_PAPER100.json"
)
DEFAULT_F2_ORACLE = (
    REPOSITORY_ROOT
    / "reports/fastsam_f2_paper100_oracle/F2_FASTSAM_PAPER100_ORACLE.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT / "logs/scannet_fastsam_f3_openbox_paper100_score05"
)
PROTOCOL_PATH = (
    REPOSITORY_ROOT / "docs/F3_FASTSAM_OPENBOX_PROJECTION_PROTOCOL_FREEZE.md"
)


class F3RunnerError(RuntimeError):
    """Raised when sealed replay identity or the F3 protocol differs."""


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
        raise F3RunnerError(f"{label} must be a regular non-symlink file: {path}")
    result = path.resolve()
    if suffix is not None and result.suffix.lower() != suffix:
        raise F3RunnerError(f"{label} must be a {suffix} file: {result}")
    if result.suffix.lower() in {".pkl", ".pickle"}:
        raise F3RunnerError(f"prediction pickle input is forbidden: {result}")
    return result


def _read_json(path: Path, label: str) -> tuple[Path, dict[str, Any]]:
    source = _regular_file(path, label, ".json")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise F3RunnerError(f"invalid {label}: {source}") from error
    if not isinstance(value, dict):
        raise F3RunnerError(f"{label} must contain one JSON object: {source}")
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
            raise F3RunnerError(f"refusing to overwrite output: {path}") from error
        return _sha256(path)
    finally:
        temporary.unlink(missing_ok=True)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise F3RunnerError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise F3RunnerError(f"{label} must be finite and non-negative")
    return result


def _xyz(value: object, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise F3RunnerError(f"{label} must be a finite xyz vector") from error
    if result.shape != (3,) or not np.isfinite(result).all():
        raise F3RunnerError(f"{label} must be a finite xyz vector")
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


def _environment() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "f3_new_gpu_allocation_bytes": 0,
    }


def _source_receipts() -> dict[str, dict[str, str]]:
    paths = {
        "runner": Path(__file__).resolve(),
        "core": Path(f3_core.__file__).resolve(),
        "protocol": PROTOCOL_PATH.resolve(),
        "f2_runner": Path(f2_runner.__file__).resolve(),
        "f0_runner": Path(f0_runner.__file__).resolve(),
    }
    return {
        key: {
            "path": os.fspath(_regular_file(path, f"F3 {key} source")),
            "sha256": _sha256(path),
        }
        for key, path in paths.items()
    }


def _load_f2_inputs(
    f2_receipt_path: Path,
    f2_oracle_path: Path,
    *,
    expected_scene_count: int,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    tuple[str, ...],
    tuple[dict[str, Any], ...],
]:
    production = expected_scene_count == EXPECTED_SCENES
    f2_path, f2 = _read_json(f2_receipt_path, "sealed F2 merged receipt")
    f2_sha = _sha256(f2_path)
    if production and f2_sha != EXPECTED_F2_RECEIPT_SHA256:
        raise F3RunnerError("sealed production F2 receipt SHA-256 differs")
    coverage = f2.get("coverage")
    rows = f2.get("scenes")
    if (
        f2.get("schema") != EXPECTED_F2_MERGE_SCHEMA
        or f2.get("protocol_id") != EXPECTED_F2_PROTOCOL_ID
        or f2.get("complete") is not True
        or f2.get("overall_pass") is not True
        or not isinstance(coverage, dict)
        or not isinstance(rows, list)
        or len(rows) != expected_scene_count
        or coverage.get("scene_count") != expected_scene_count
        or coverage.get("source_count")
        != (EXPECTED_SOURCES if production else coverage.get("source_count"))
    ):
        raise F3RunnerError("sealed F2 merged receipt contract differs")
    scenes = tuple(str(value) for value in coverage.get("scene_order", ()))
    if (
        len(scenes) != expected_scene_count
        or len(set(scenes)) != len(scenes)
        or [row.get("scene_id") for row in rows] != list(scenes)
        or [row.get("scene_index") for row in rows]
        != list(range(expected_scene_count))
    ):
        raise F3RunnerError("sealed F2 paper100 scene order differs")
    if production and (
        coverage.get("keyframe_count") != EXPECTED_KEYFRAMES
        or coverage.get("successful_frame_count") != EXPECTED_SUCCESSFUL_FRAMES
        or coverage.get("source_count") != EXPECTED_SOURCES
        or coverage.get("identity_verified_source_count") != EXPECTED_SOURCES
    ):
        raise F3RunnerError("sealed F2 paper100 census differs")

    scene_list = f2.get("scene_list")
    if not isinstance(scene_list, dict):
        raise F3RunnerError("sealed F2 scene-list receipt is absent")
    scene_list_source = _regular_file(Path(scene_list.get("path", "")), "F2 scene list")
    if _sha256(scene_list_source) != scene_list.get("sha256"):
        raise F3RunnerError("sealed F2 scene-list rehash differs")

    oracle_path, oracle = _read_json(f2_oracle_path, "sealed F2 oracle receipt")
    oracle_sha = _sha256(oracle_path)
    if production and oracle_sha != EXPECTED_F2_ORACLE_SHA256:
        raise F3RunnerError("sealed production F2 oracle SHA-256 differs")
    decision = oracle.get("decision")
    if (
        oracle.get("schema") != EXPECTED_F2_ORACLE_SCHEMA
        or not isinstance(decision, dict)
        or decision.get("authorize_f3_projection_self_validation_shadow") is not True
        or decision.get("f3_shadow_geometry_input") != "F1_H0_only"
        or decision.get("retain_f2_geometry_for_f3") is not False
        or decision.get("authorize_active_birth") is not False
    ):
        raise F3RunnerError("F2 oracle did not authorize H0-only F3 shadow")
    return (
        {
            "path": os.fspath(f2_path),
            "sha256": f2_sha,
            "run_signature_sha256": f2.get("run_signature_sha256"),
            "scene_list": dict(scene_list),
        },
        {
            "path": os.fspath(oracle_path),
            "sha256": oracle_sha,
            "schema": oracle.get("schema"),
            "decision_sha256": _canonical_json_sha256(decision),
        },
        scenes,
        tuple(dict(row) for row in rows),
    )


def _load_intrinsic(path: Path, expected_sha: str) -> np.ndarray:
    source = _regular_file(path, "sealed ScanNet depth intrinsic", ".txt")
    if _sha256(source) != expected_sha:
        raise F3RunnerError("sealed depth-intrinsic rehash differs")
    try:
        matrix = np.loadtxt(source, dtype=np.float64)
    except (OSError, ValueError) as error:
        raise F3RunnerError("sealed depth intrinsic cannot be decoded") from error
    if matrix.shape == (4, 4):
        matrix = matrix[:3, :3]
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or matrix[0, 0] <= 0.0
        or matrix[1, 1] <= 0.0
    ):
        raise F3RunnerError("sealed depth intrinsic is invalid")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _load_pose(path: Path, expected_sha: str, valid: bool) -> np.ndarray | None:
    source = _regular_file(path, "sealed ScanNet pose", ".txt")
    if _sha256(source) != expected_sha:
        raise F3RunnerError(f"sealed pose rehash differs: {source}")
    pose = f0_runner._read_pose(source)
    if valid and pose is None:
        raise F3RunnerError(f"successful F2 frame has an invalid pose: {source}")
    if not valid and pose is not None:
        # A non-upright producer frame may still have a valid raw pose.  Only
        # current-pose-invalid abstentions require a non-finite pose.
        return np.ascontiguousarray(pose, dtype=np.float64)
    return None if pose is None else np.ascontiguousarray(pose, dtype=np.float64)


class _EvidenceAccessor:
    """Physically decoded NPZ with a strict logical frame-ordinal guard."""

    def __init__(
        self,
        path: Path,
        expected_sha: str,
        scene: str,
        expected_source_count: int,
    ) -> None:
        source = _regular_file(path, f"F2 evidence {scene}", ".npz")
        if _sha256(source) != expected_sha:
            raise F3RunnerError(f"F2 evidence rehash differs: {scene}")
        try:
            archive = np.load(source, allow_pickle=False)
            self._archive = archive
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
                raise F3RunnerError(f"F2 evidence array schema differs: {scene}")
            if (
                str(archive["schema"].item()) != EXPECTED_F2_EVIDENCE_SCHEMA
                or str(archive["scene_id"].item()) != scene
                or archive["mask_shape"].tolist() != list(MASK_SHAPE)
                or str(archive["mask_bitorder"].item()) != "little"
            ):
                raise F3RunnerError(f"F2 evidence metadata differs: {scene}")
            self.source_ids = archive["source_ids"]
            self.frame_ids = archive["frame_ids"]
            self.raw_indices = archive["raw_indices"]
            self.ranks = archive["ranks"]
            self.candidate_indices = archive["candidate_indices"]
            self.masks = archive["masks_packbits"]
            self.offsets = archive["point_offsets"]
            self.points = archive["points_world"]
            self.keys = archive["voxel_keys"]
        except (OSError, ValueError) as error:
            raise F3RunnerError(f"F2 evidence cannot be decoded: {scene}") from error
        count = expected_source_count
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
            or self.points.shape != self.keys.shape
            or self.points.shape != (int(self.offsets[-1]), 3)
            or self.points.dtype != np.dtype("<f8")
            or self.keys.dtype != np.dtype("<i8")
        ):
            raise F3RunnerError(f"F2 H0 evidence shapes/dtypes differ: {scene}")
        self.path = source
        self.sha256 = expected_sha
        self.max_logical_accessed_ordinal = -1
        self.access_count = 0

    def close(self) -> None:
        self._archive.close()

    def expose(
        self,
        index: int,
        *,
        current_ordinal: int,
        source_ordinal: int,
    ) -> dict[str, Any]:
        if source_ordinal > current_ordinal:
            raise F3RunnerError("future F2 source logical access was attempted")
        if index < 0 or index >= len(self.source_ids):
            raise F3RunnerError("F2 evidence source index is out of range")
        start = int(self.offsets[index])
        stop = int(self.offsets[index + 1])
        self.max_logical_accessed_ordinal = max(
            self.max_logical_accessed_ordinal, source_ordinal
        )
        self.access_count += 1
        return {
            "source_id": str(self.source_ids[index]),
            "frame_id": int(self.frame_ids[index]),
            "raw_index": int(self.raw_indices[index]),
            "rank": int(self.ranks[index]),
            "candidate_index": int(self.candidate_indices[index]),
            "mask_packbits": np.ascontiguousarray(self.masks[index], dtype=np.uint8),
            "points_world": np.ascontiguousarray(
                self.points[start:stop], dtype=np.float64
            ),
            "sealed_voxel_keys": np.ascontiguousarray(
                self.keys[start:stop], dtype=np.int64
            ),
        }


def _validate_source(
    *,
    source: Mapping[str, Any],
    f0_candidate: Mapping[str, Any],
    evidence: Mapping[str, Any],
    scene: str,
    frame_id: int,
    candidate_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_id = source.get("source_id")
    expected_id = f"{scene}/frame_{frame_id:06d}/raw_{int(source.get('raw_index', -1)):03d}"
    if (
        source_id != expected_id
        or source.get("candidate_index") != candidate_index
        or source.get("rank") != candidate_index
        or evidence["source_id"] != source_id
        or evidence["frame_id"] != frame_id
        or evidence["raw_index"] != source.get("raw_index")
        or evidence["rank"] != source.get("rank")
        or evidence["candidate_index"] != candidate_index
    ):
        raise F3RunnerError(f"F2 source identity differs: {source_id}")
    for key in (
        "raw_index",
        "rank",
        "confidence",
        "mask_sha256",
        "points_and_voxel_keys_sha256",
        "stored_point_count",
        "world_q02",
        "world_q98",
    ):
        source_key = {"world_q02": "f0_world_q02", "world_q98": "f0_world_q98"}.get(
            key, key
        )
        if source.get(source_key) != f0_candidate.get(key):
            raise F3RunnerError(f"F2/F0 H0 source differs: {source_id}/{key}")
    hypotheses = source.get("hypotheses")
    h0 = hypotheses.get("H0") if isinstance(hypotheses, dict) else None
    if (
        not isinstance(h0, dict)
        or h0.get("valid") is not True
        or h0.get("q02") != source.get("f0_world_q02")
        or h0.get("q98") != source.get("f0_world_q98")
    ):
        raise F3RunnerError(f"F3 input is not exact F1/H0: {source_id}")
    packed = evidence["mask_packbits"]
    if hashlib.sha256(packed.tobytes()).hexdigest() != source.get("mask_sha256"):
        raise F3RunnerError(f"F2 mask evidence differs: {source_id}")
    points = evidence["points_world"]
    sealed_keys = evidence["sealed_voxel_keys"]
    digest = hashlib.sha256()
    digest.update(np.asarray(points, dtype="<f8").tobytes())
    digest.update(np.asarray(sealed_keys, dtype="<i8").tobytes())
    if (
        len(points) != source.get("stored_point_count")
        or digest.hexdigest() != source.get("points_and_voxel_keys_sha256")
        or digest.hexdigest() != h0.get("points_and_voxel_keys_sha256")
    ):
        raise F3RunnerError(f"F2 H0 point evidence differs: {source_id}")
    q02 = _xyz(source.get("f0_world_q02"), f"{source_id}.q02")
    q98 = _xyz(source.get("f0_world_q98"), f"{source_id}.q98")
    if np.any(q98 <= q02):
        raise F3RunnerError(f"F2 H0 box has non-positive extent: {source_id}")
    return points, q02, q98


def _bounded_f3_voxel_keys(points_world: np.ndarray) -> np.ndarray:
    """Return the exact frozen 5 cm unique/capped ledger with a faster sort.

    ``np.unique(..., axis=0)`` internally builds and sorts a structured copy.
    The lexsort-plus-adjacent predicate below produces the same lexicographic
    rows, then applies the protocol's identical inclusive linspace cap.  The
    public core still validates the resulting at-most-512 rows, so this is an
    execution-only optimization and not a trusted-input shortcut.
    """

    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise F3RunnerError("authenticated H0 points must be finite [N,3]")
    scaled = points / F3_VOXEL_SIZE_M
    if np.max(np.abs(scaled), initial=0.0) > np.iinfo(np.int64).max / 4:
        raise F3RunnerError("H0 points exceed safe 5 cm voxel coordinates")
    keys = np.floor(scaled).astype(np.int64)
    if not len(keys):
        return np.empty((0, 3), dtype=np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    ordered = keys[order]
    keep = np.empty(len(ordered), dtype=np.bool_)
    keep[0] = True
    keep[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    unique = ordered[keep]
    if len(unique) > F3_MAX_VOXELS_PER_OBSERVATION:
        indices = np.linspace(
            0,
            len(unique) - 1,
            num=F3_MAX_VOXELS_PER_OBSERVATION,
            endpoint=True,
            dtype=np.int64,
        )
        unique = unique[indices]
    return np.ascontiguousarray(unique, dtype=np.int64)


def _as_dict(value: object, serializer: Callable[[object], Mapping[str, Any]]) -> dict[str, Any]:
    try:
        row = serializer(value)
    except (TypeError, ValueError) as error:
        raise F3RunnerError("F3 core serializer rejected its own result") from error
    if not isinstance(row, Mapping):
        raise F3RunnerError("F3 core serializer did not return a mapping")
    return dict(row)


def _hypothesis_json(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise F3RunnerError(f"{label} hypothesis is absent")
    row = dict(value)
    valid = row.get("valid") is True
    q02_raw = row.get("q02", row.get("world_q02"))
    q98_raw = row.get("q98", row.get("world_q98"))
    if valid:
        q02 = _xyz(q02_raw, f"{label}.q02")
        q98 = _xyz(q98_raw, f"{label}.q98")
        if np.any(q98 <= q02):
            raise F3RunnerError(f"{label} valid geometry has non-positive extent")
        q02_json: list[float] | None = q02.tolist()
        q98_json: list[float] | None = q98.tolist()
        center_json: list[float] | None = ((q02 + q98) * 0.5).tolist()
        extent_json: list[float] | None = (q98 - q02).tolist()
    else:
        q02_json = None
        q98_json = None
        center_json = None
        extent_json = None
    fold_ious_raw = row.get("fold_ious", ())
    if not isinstance(fold_ious_raw, (list, tuple)):
        raise F3RunnerError(f"{label}.fold_ious must be a sequence")
    fold_ious = [_number(value, f"{label}.fold_iou") for value in fold_ious_raw]
    if any(value > 1.0 for value in fold_ious):
        raise F3RunnerError(f"{label}.fold_ious must be in [0,1]")
    valid_fold_count = int(row.get("valid_fold_count", len(fold_ious)))
    if valid_fold_count != len(fold_ious):
        raise F3RunnerError(f"{label} valid-fold ledger differs")
    score_raw = row.get("score")
    score = None if score_raw is None else _number(score_raw, f"{label}.score")
    if score is not None and score > 1.0:
        raise F3RunnerError(f"{label}.score must be in [0,1]")
    return {
        "valid": valid,
        "reason": str(row.get("reason", "valid" if valid else "invalid")),
        "q02": q02_json,
        "q98": q98_json,
        "center": center_json,
        "extent": extent_json,
        "score": score,
        "fold_ious": fold_ious,
        "valid_fold_count": valid_fold_count,
    }


def _track_json(
    value: Mapping[str, Any],
    *,
    all_sources: Sequence[str],
    all_frames: Sequence[int],
) -> dict[str, Any]:
    track_id = int(value.get("track_id", -1))
    if track_id < 0:
        raise F3RunnerError("F3 terminal track ID is invalid")
    core_sources = value.get("source_ids")
    core_frames = value.get("frame_ids")
    if (
        core_sources != list(all_sources)
        or core_frames != [int(item) for item in all_frames]
        or value.get("observation_count") != len(all_sources)
        or value.get("total_observation_count") != len(all_sources)
    ):
        raise F3RunnerError(f"track {track_id} terminal lineage differs from commits")
    hypotheses = value.get("hypotheses")
    if not isinstance(hypotheses, Mapping):
        hypotheses = {"B": value.get("B"), "C": value.get("C")}
    b = _hypothesis_json(hypotheses.get("B"), f"track {track_id}.B")
    c = _hypothesis_json(hypotheses.get("C"), f"track {track_id}.C")
    c_raw = hypotheses.get("C")
    b_raw = hypotheses.get("B")
    stability = c_raw.get("stability", {}) if isinstance(c_raw, Mapping) else {}
    loo_full_ious = (
        stability.get("pairwise_aabb_ious", ())
        if isinstance(stability, Mapping)
        else ()
    )
    c["loo_full_aabb_ious"] = [
        _number(item, f"track {track_id}.C.loo_full_aabb_iou")
        for item in loo_full_ious
    ]
    b_lower = b["q02"]
    b_upper = b["q98"]
    if (b_lower is None or b_upper is None) and isinstance(b_raw, Mapping):
        # B can be available but invalid solely because score<0.10.  Its
        # selected candidate geometry remains sealed in candidate_evaluations.
        source_id = b_raw.get("source_id")
        candidates = b_raw.get("candidate_evaluations", ())
        if isinstance(candidates, list):
            for candidate in candidates:
                if isinstance(candidate, Mapping) and candidate.get("source_id") == source_id:
                    b_lower = candidate.get("q02")
                    b_upper = candidate.get("q98")
                    break
    if c["valid"] and b_lower is not None and b_upper is not None:
        b_q02 = _xyz(b_lower, f"track {track_id}.B.available_q02")
        b_q98 = _xyz(b_upper, f"track {track_id}.B.available_q98")
        c_q02 = _xyz(c["q02"], f"track {track_id}.C.q02")
        c_q98 = _xyz(c["q98"], f"track {track_id}.C.q98")
        b_extent = b_q98 - b_q02
        c_extent = c_q98 - c_q02
        c["center_shift_from_b_m"] = float(
            np.linalg.norm((c_q02 + c_q98) * 0.5 - (b_q02 + b_q98) * 0.5)
        )
        c["extent_ratios"] = (c_extent / b_extent).tolist()
        c["volume_ratio"] = float(np.prod(c_extent) / np.prod(b_extent))
    else:
        c["center_shift_from_b_m"] = None
        c["extent_ratios"] = None
        c["volume_ratio"] = None
    selector_raw = value.get("selector")
    if not isinstance(selector_raw, Mapping):
        raise F3RunnerError(f"track {track_id} selector is absent")
    chosen_raw = selector_raw.get("chosen")
    chosen = None if chosen_raw in (None, "none", "NONE") else str(chosen_raw).upper()
    if chosen not in (None, "B", "C"):
        raise F3RunnerError(f"track {track_id} selector choice is invalid")
    selected = None if chosen is None else {"B": b, "C": c}[chosen]
    if selected is not None and not selected["valid"]:
        raise F3RunnerError(f"track {track_id} selected an invalid hypothesis")
    selector = {
        "chosen": chosen,
        "reason": str(selector_raw.get("reason", "abstain" if chosen is None else "selected")),
        "q02": None if selected is None else selected["q02"],
        "q98": None if selected is None else selected["q98"],
        "center": None if selected is None else selected["center"],
        "extent": None if selected is None else selected["extent"],
        "score": None if selected is None else selected["score"],
    }
    retained_sources = value.get("retained_source_ids", ())
    retained_frames = value.get("retained_frame_ids", ())
    if not isinstance(retained_sources, (list, tuple)) or not isinstance(
        retained_frames, (list, tuple)
    ):
        raise F3RunnerError(f"track {track_id} retained evidence is invalid")
    if (
        list(retained_sources) != list(all_sources)[-5:]
        or [int(item) for item in retained_frames] != [int(item) for item in all_frames][-5:]
    ):
        raise F3RunnerError(f"track {track_id} bounded evidence differs from lineage")
    return {
        "track_id": track_id,
        # Full assignment provenance is maintained by the runner.  It is
        # intentionally separate from the bounded five-observation evidence.
        "source_ids": list(all_sources),
        "frame_ids": [int(value) for value in all_frames],
        "observation_count": len(all_sources),
        "retained_source_ids": [str(item) for item in retained_sources],
        "retained_frame_ids": [int(item) for item in retained_frames],
        "retained_observation_count": len(retained_sources),
        "confirmed": bool(value.get("confirmed", len(retained_sources) >= 3)),
        "hypotheses": {"B": b, "C": c},
        "selector": selector,
    }


def _signature_payload(
    *,
    scenes: Sequence[str],
    f2_receipt: Mapping[str, Any],
    f2_oracle: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, str]],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "scene_order": list(scenes),
        "f2_receipt": dict(f2_receipt),
        "f2_oracle": dict(f2_oracle),
        "sources": {key: dict(value) for key, value in sources.items()},
        "environment": dict(environment),
        "f3_core_schema": getattr(f3_core, "SCHEMA", None),
        "f3_core_policy": dict(getattr(f3_core, "POLICY", {})),
    }


def _process_scene(
    *,
    scene: str,
    scene_index: int,
    f2_row: Mapping[str, Any],
    run_signature: str,
    f2_receipt: Mapping[str, Any],
    f2_oracle: Mapping[str, Any],
    source_receipts: Mapping[str, Mapping[str, str]],
    tracker_factory: Callable[[], Any] | None,
) -> dict[str, Any]:
    audit_started = time.perf_counter()
    sidecar_ref = f2_row.get("sidecar")
    evidence_ref = f2_row.get("evidence_npz")
    if not isinstance(sidecar_ref, Mapping) or not isinstance(evidence_ref, Mapping):
        raise F3RunnerError(f"F2 scene references are absent: {scene}")
    f2_sidecar_path, f2_scene = _read_json(
        Path(str(sidecar_ref.get("path"))), f"sealed F2 scene {scene}"
    )
    if _sha256(f2_sidecar_path) != sidecar_ref.get("sha256"):
        raise F3RunnerError(f"F2 scene sidecar rehash differs: {scene}")
    if (
        f2_scene.get("schema") != f2_runner.SCENE_SCHEMA
        or f2_scene.get("protocol_id") != f2_runner.PROTOCOL_ID
        or f2_scene.get("complete") is not True
        or f2_scene.get("scene_id") != scene
        or f2_scene.get("scene_index") != scene_index
    ):
        raise F3RunnerError(f"F2 scene contract differs: {scene}")
    if (
        f2_scene.get("evidence_npz", {}).get("path") != evidence_ref.get("path")
        or f2_scene.get("evidence_npz", {}).get("sha256")
        != evidence_ref.get("sha256")
    ):
        raise F3RunnerError(f"F2 scene evidence reference differs: {scene}")
    f0_ref = f2_scene.get("f0_sidecar")
    if not isinstance(f0_ref, Mapping):
        raise F3RunnerError(f"F0 scene reference is absent: {scene}")
    f0_path, f0_scene = _read_json(
        Path(str(f0_ref.get("path"))), f"sealed F0 scene {scene}"
    )
    if _sha256(f0_path) != f0_ref.get("sha256"):
        raise F3RunnerError(f"F0 scene sidecar rehash differs: {scene}")
    if (
        f0_scene.get("schema") != f0_runner.SCENE_SCHEMA
        or f0_scene.get("protocol_id") != f0_runner.PROTOCOL_ID
        or f0_scene.get("complete") is not True
        or f0_scene.get("scene_id") != scene
        or f0_scene.get("scene_index") != scene_index
    ):
        raise F3RunnerError(f"F0 scene contract differs: {scene}")
    schedule_ref = f2_scene.get("schedule")
    intrinsic_ref = f2_scene.get("intrinsic")
    if not isinstance(schedule_ref, Mapping) or not isinstance(intrinsic_ref, Mapping):
        raise F3RunnerError(f"F2 schedule/intrinsic reference is absent: {scene}")
    schedule_path, schedule = _read_json(
        Path(str(schedule_ref.get("path"))), f"sealed schedule {scene}"
    )
    if _sha256(schedule_path) != schedule_ref.get("sha256"):
        raise F3RunnerError(f"sealed schedule rehash differs: {scene}")
    schedule_frames = schedule.get("recorded_frame_ids")
    if not isinstance(schedule_frames, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in schedule_frames
    ):
        raise F3RunnerError(f"sealed schedule frame ledger is invalid: {scene}")
    intrinsic_path = Path(str(intrinsic_ref.get("path")))
    intrinsics = _load_intrinsic(intrinsic_path, str(intrinsic_ref.get("sha256")))
    f2_frames = f2_scene.get("frames")
    f0_frames = f0_scene.get("frames")
    if (
        not isinstance(f2_frames, list)
        or not isinstance(f0_frames, list)
        or [row.get("frame_id") for row in f2_frames] != schedule_frames
        or [row.get("frame_id") for row in f0_frames] != schedule_frames
        or [row.get("frame_ordinal") for row in f2_frames]
        != list(range(len(schedule_frames)))
    ):
        raise F3RunnerError(f"F2/F0/schedule frame ledger differs: {scene}")
    source_count = sum(len(frame.get("sources", ())) for frame in f2_frames)
    if source_count != f2_scene.get("summary", {}).get("counts", {}).get("sources"):
        raise F3RunnerError(f"F2 scene source census differs: {scene}")
    accessor = _EvidenceAccessor(
        Path(str(evidence_ref.get("path"))),
        str(evidence_ref.get("sha256")),
        scene,
        source_count,
    )
    input_audit_ms = (time.perf_counter() - audit_started) * 1000.0

    tracker = (
        tracker_factory()
        if tracker_factory is not None
        else f3_core.FastSAMOpenBoxF3ShadowTracker()
    )
    frames: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    track_sources: defaultdict[int, list[str]] = defaultdict(list)
    track_frames: defaultdict[int, list[int]] = defaultdict(list)
    source_index = 0
    f3_samples: list[float] = []
    composed_samples: list[float] = []
    frame_hash_ms_total = 0.0
    source_identity_ms_total = 0.0
    max_logical_accessed_ordinal = -1
    prefix_invariance = True
    query_before_commit = True
    all_assignments_unique = True
    try:
        for ordinal, (frame_id, f2_frame, f0_frame) in enumerate(
            zip(schedule_frames, f2_frames, f0_frames)
        ):
            frame_audit_started = time.perf_counter()
            if bool(f2_frame.get("successful")) != bool(f0_frame.get("successful")):
                raise F3RunnerError(f"F2/F0 frame state differs: {scene}/{frame_id}")
            inputs = f0_frame.get("inputs")
            if not isinstance(inputs, Mapping):
                raise F3RunnerError(f"F0 frame inputs are absent: {scene}/{frame_id}")
            depth_path = _regular_file(
                Path(str(inputs.get("depth_path"))), "sealed ScanNet depth", ".png"
            )
            if _sha256(depth_path) != inputs.get("depth_sha256"):
                raise F3RunnerError(f"sealed depth rehash differs: {scene}/{frame_id}")
            pose = _load_pose(
                Path(str(inputs.get("pose_path"))),
                str(inputs.get("pose_sha256")),
                bool(inputs.get("current_pose_valid")),
            )
            successful = bool(f2_frame.get("successful"))
            if successful and pose is None:
                raise F3RunnerError(f"successful F3 source frame lacks pose: {scene}/{frame_id}")
            frame_hash_ms = (time.perf_counter() - frame_audit_started) * 1000.0
            frame_hash_ms_total += frame_hash_ms

            source_rows = f2_frame.get("sources")
            funnel = f0_frame.get("funnel")
            if successful:
                f0_candidates = funnel.get("candidates") if isinstance(funnel, Mapping) else None
            else:
                if funnel is not None:
                    raise F3RunnerError(f"failed F0 frame carries a funnel: {scene}/{frame_id}")
                f0_candidates = []
            if not isinstance(source_rows, list) or not isinstance(f0_candidates, list):
                raise F3RunnerError(f"F2/F0 source rows are absent: {scene}/{frame_id}")
            if len(source_rows) != len(f0_candidates):
                raise F3RunnerError(f"F2/F0 source count differs: {scene}/{frame_id}")
            current: list[tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
            identity_started = time.perf_counter()
            for candidate_index, (source, f0_candidate) in enumerate(
                zip(source_rows, f0_candidates)
            ):
                if not isinstance(source, Mapping) or not isinstance(f0_candidate, Mapping):
                    raise F3RunnerError(f"invalid F2/F0 source: {scene}/{frame_id}")
                exposed = accessor.expose(
                    source_index,
                    current_ordinal=ordinal,
                    source_ordinal=ordinal,
                )
                points, q02, q98 = _validate_source(
                    source=source,
                    f0_candidate=f0_candidate,
                    evidence=exposed,
                    scene=scene,
                    frame_id=frame_id,
                    candidate_index=candidate_index,
                )
                current.append((source, points, q02, q98, exposed["mask_packbits"]))
                source_index += 1
            source_identity_ms = (time.perf_counter() - identity_started) * 1000.0
            source_identity_ms_total += source_identity_ms

            f3_started = time.perf_counter()
            observations = []
            for source, points, q02, q98, packed in current:
                # F0's sealed keys are 2 cm and are used only for identity.
                # F3's preregistered bounded evidence uses signed-floor 5 cm.
                keys_5cm = _bounded_f3_voxel_keys(points)
                observations.append(
                    f3_core.make_observation(
                        source_id=str(source["source_id"]),
                        frame_id=int(frame_id),
                        frame_ordinal=ordinal,
                        confidence=float(source["confidence"]),
                        world_q02=q02,
                        world_q98=q98,
                        voxel_keys=keys_5cm,
                        camera_to_world=pose,
                        intrinsics=intrinsics,
                        mask_packbits=packed,
                    )
                )
            try:
                query = tracker.query(
                    frame_id=int(frame_id),
                    frame_ordinal=ordinal,
                    observations=tuple(observations),
                    # Camera provenance for the current scheduled update has
                    # already been authenticated even when it has no source.
                    # Thus the frame itself is the maximum logical access.
                    max_logical_accessed_ordinal=ordinal,
                )
                commit = tracker.commit(query)
            except (TypeError, ValueError, RuntimeError) as error:
                raise F3RunnerError(
                    f"F3 core rejected sealed frame {scene}/{frame_id}"
                ) from error
            f3_core_ms = (time.perf_counter() - f3_started) * 1000.0
            f3_samples.append(f3_core_ms)
            commit_row = _as_dict(commit, f3_core.frame_commit_to_dict)
            assignment_rows = commit_row.get("assignments")
            if not isinstance(assignment_rows, list):
                raise F3RunnerError(f"F3 assignment ledger is absent: {scene}/{frame_id}")
            assignment_by_source: dict[str, dict[str, Any]] = {}
            assigned_ids: set[str] = set()
            for row in assignment_rows:
                if not isinstance(row, Mapping):
                    raise F3RunnerError("F3 assignment row is invalid")
                source_id = str(row.get("source_id"))
                track_id = row.get("track_id")
                action = str(row.get("action"))
                if (
                    source_id in assigned_ids
                    or source_id not in {str(item[0]["source_id"]) for item in current}
                    or isinstance(track_id, bool)
                    or not isinstance(track_id, int)
                    or track_id < 0
                    or action not in {"matched", "created"}
                ):
                    raise F3RunnerError(f"F3 assignment differs: {scene}/{frame_id}")
                assigned_ids.add(source_id)
                assignment_by_source[source_id] = {
                    "source_id": source_id,
                    "track_id": track_id,
                    "action": action,
                }
            expected_ids = [str(row[0]["source_id"]) for row in current]
            if assigned_ids != set(expected_ids) or len(assignment_by_source) != len(expected_ids):
                all_assignments_unique = False
                raise F3RunnerError(f"F3 did not assign every source: {scene}/{frame_id}")
            # The core's deterministic capacity order is lexical source_id;
            # the public scene ledger remains in the original F0/F2 rank order.
            assignments = [assignment_by_source[source_id] for source_id in expected_ids]
            for assignment in assignments:
                track_id = int(assignment["track_id"])
                track_sources[track_id].append(str(assignment["source_id"]))
                track_frames[track_id].append(int(frame_id))
            retired = commit_row.get(
                "retired_track_ids", commit_row.get("retired_ids", ())
            )
            if not isinstance(retired, (list, tuple)) or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in retired
            ):
                raise F3RunnerError(f"F3 retired-track ledger differs: {scene}/{frame_id}")
            reported_max = int(
                commit_row.get(
                    "max_logical_accessed_ordinal",
                    ordinal,
                )
            )
            max_logical_accessed_ordinal = max(
                max_logical_accessed_ordinal, reported_max
            )
            if reported_max > ordinal:
                raise F3RunnerError(f"F3 future access detected: {scene}/{frame_id}")
            audit_complete = commit_row.get("audit_complete", True) is True
            query_before_commit &= bool(
                commit_row.get("query_before_commit", audit_complete)
            )
            prefix_invariance &= bool(
                commit_row.get("prefix_invariance", audit_complete)
            )
            inherited = f2_frame.get("runtime")
            if not isinstance(inherited, Mapping):
                raise F3RunnerError(f"F2 runtime is absent: {scene}/{frame_id}")
            composed_ms = (
                _number(inherited.get("provider_ms", 0.0), "F2 provider_ms")
                + _number(inherited.get("f0_core_ms", 0.0), "F2 f0_core_ms")
                + f3_core_ms
            )
            if successful and not bool(inherited.get("warmup_excluded")):
                composed_samples.append(composed_ms)
            source_ids = [str(source["source_id"]) for source in source_rows]
            frame_row = {
                "ordinal": ordinal,
                "frame_id": int(frame_id),
                "successful": successful,
                "source_ids": source_ids,
                "assignments": assignments,
                "retired_ids": [int(value) for value in retired],
                "max_logical_accessed_ordinal": reported_max,
                "query_before_commit": bool(
                    commit_row.get("query_before_commit", audit_complete)
                ),
                "prefix_invariance": bool(
                    commit_row.get("prefix_invariance", audit_complete)
                ),
                "f3_core_ms": f3_core_ms,
                "core_reported_ms": _number(
                    commit_row.get("elapsed_ms", 0.0), "F3 core elapsed_ms"
                ),
                "composed_complete_ms": composed_ms,
                "inherited_warmup_excluded": bool(
                    inherited.get("warmup_excluded", False)
                ),
                "input_hash_ms": frame_hash_ms,
                "identity_audit_ms": source_identity_ms,
            }
            frames.append(frame_row)
            counts["keyframes"] += 1
            counts["successful_frames"] += int(successful)
            counts["sources"] += len(source_ids)
            counts["identity_verified_sources"] += len(source_ids)
    finally:
        accessor.close()
    if source_index != source_count or accessor.access_count != source_count:
        raise F3RunnerError(f"F3 did not expose every sealed source exactly once: {scene}")

    terminal = tracker.finalize()
    terminal_row = _as_dict(terminal, f3_core.terminal_seal_to_dict)
    core_tracks = terminal_row.get("tracks")
    if not isinstance(core_tracks, list):
        raise F3RunnerError(f"F3 terminal track ledger is absent: {scene}")
    core_by_id: dict[int, Mapping[str, Any]] = {}
    for value in core_tracks:
        if not isinstance(value, Mapping):
            raise F3RunnerError(f"F3 terminal track row is invalid: {scene}")
        track_id = int(value.get("track_id", -1))
        if track_id < 0 or track_id in core_by_id:
            raise F3RunnerError(f"F3 terminal track IDs differ: {scene}")
        core_by_id[track_id] = value
    if set(core_by_id) != set(track_sources):
        raise F3RunnerError(f"F3 terminal tracks do not cover assignments: {scene}")
    tracks = [
        _track_json(
            core_by_id[track_id],
            all_sources=track_sources[track_id],
            all_frames=track_frames[track_id],
        )
        for track_id in sorted(core_by_id)
    ]
    flattened = [source for track in tracks for source in track["source_ids"]]
    frame_sources = [source for frame in frames for source in frame["source_ids"]]
    one_source_one_track = bool(
        all_assignments_unique
        and len(flattened) == len(set(flattened))
        and set(flattened) == set(frame_sources)
        and len(flattened) == len(frame_sources)
    )
    counts["tracks"] = len(tracks)
    counts["confirmed_tracks"] = sum(track["confirmed"] for track in tracks)
    counts["selected_tracks"] = sum(
        track["selector"]["chosen"] is not None for track in tracks
    )
    inherited_gpu = int(
        f2_scene.get("summary", {}).get("gpu_peak_memory_bytes", 0)
    )
    runtime = {
        "f3_core_ms": _distribution(f3_samples),
        "composed_complete_ms": _distribution(composed_samples),
        "amortized_f3_ms_per_source_frame": (
            float(np.mean(f3_samples)) / SOURCE_FRAME_STRIDE if f3_samples else 0.0
        ),
        "amortized_composed_ms_per_source_frame": (
            float(np.mean(composed_samples)) / SOURCE_FRAME_STRIDE
            if composed_samples
            else 0.0
        ),
        "input_audit_ms": input_audit_ms,
        "frame_input_hash_ms_total": frame_hash_ms_total,
        "source_identity_audit_ms_total": source_identity_ms_total,
        "new_gpu_allocation_bytes": 0,
        "inherited_gpu_peak_memory_bytes": inherited_gpu,
        "cpu_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
    }
    causality = {
        "prefix_invariance": {"passed": bool(prefix_invariance)},
        "query_before_commit": {"passed": bool(query_before_commit)},
        "one_source_one_track": {"passed": one_source_one_track},
        "maximum_logical_accessed_ordinal": {
            "actual": max_logical_accessed_ordinal,
            "threshold": len(frames) - 1,
            "comparator": "<=",
            "passed": max_logical_accessed_ordinal <= len(frames) - 1,
        },
        "future_access_count": 0,
    }
    causality["overall_pass"] = all(
        causality[name]["passed"]
        for name in (
            "prefix_invariance",
            "query_before_commit",
            "one_source_one_track",
            "maximum_logical_accessed_ordinal",
        )
    )
    contracts = {
        "shadow_only": True,
        "observer_only": True,
        "birth_enabled": False,
        "native_output_mutation": False,
        "ground_truth_access": False,
        "prediction_access": False,
        "evaluator_access": False,
        "rgb_access": False,
        "depth_pixel_access": False,
        "semantic_or_clip_access": False,
        "training": False,
        "online_learning": False,
        "fastsam_rerun": False,
        "h0_only": True,
        "future_frame_logical_access": False,
        "hl_hlg_access": False,
    }
    return {
        "schema": SCENE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "run_signature_sha256": run_signature,
        "scene_id": scene,
        "scene_index": scene_index,
        "contracts": contracts,
        "inputs": {
            "f2_receipt": dict(f2_receipt),
            "f2_oracle": dict(f2_oracle),
            "f2_sidecar": {
                "path": os.fspath(f2_sidecar_path),
                "sha256": _sha256(f2_sidecar_path),
            },
            "f2_evidence_npz": {
                "path": os.fspath(accessor.path),
                "sha256": accessor.sha256,
            },
            "f0_sidecar": {"path": os.fspath(f0_path), "sha256": _sha256(f0_path)},
            "schedule": {
                "path": os.fspath(schedule_path),
                "sha256": _sha256(schedule_path),
            },
            "intrinsic": {
                "path": os.fspath(intrinsic_path.resolve()),
                "sha256": _sha256(intrinsic_path),
            },
            "sources": {key: dict(value) for key, value in source_receipts.items()},
        },
        "counts": {
            "keyframe_count": counts["keyframes"],
            "successful_frame_count": counts["successful_frames"],
            "source_count": counts["sources"],
            "identity_verified_source_count": counts["identity_verified_sources"],
            "track_count": counts["tracks"],
            "confirmed_track_count": counts["confirmed_tracks"],
            "selected_track_count": counts["selected_tracks"],
        },
        "causality": causality,
        "runtime": runtime,
        "frames": frames,
        "tracks": tracks,
        "conclusion_guardrail": (
            "F3 is an output-inert no-GT projection shadow. Its B/C boxes have "
            "no AP and cannot authorize a prediction birth before the sealed oracle."
        ),
    }


def _manifest_scene_row(
    receipt: Mapping[str, Any], path: Path, digest: str, *, resumed: bool
) -> dict[str, Any]:
    return {
        "scene_id": receipt["scene_id"],
        "scene_index": receipt["scene_index"],
        "sidecar_path": os.fspath(path.resolve()),
        "sidecar_sha256": digest,
        "counts": dict(receipt["counts"]),
        "causality": dict(receipt["causality"]),
        "runtime": dict(receipt["runtime"]),
        "resumed": resumed,
    }


def _resume_scene(
    path: Path,
    *,
    scene: str,
    scene_index: int,
    run_signature: str,
) -> tuple[dict[str, Any], str]:
    source, receipt = _read_json(path, f"resumed F3 scene {scene}")
    if (
        receipt.get("schema") != SCENE_SCHEMA
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("complete") is not True
        or receipt.get("scene_id") != scene
        or receipt.get("scene_index") != scene_index
        or receipt.get("run_signature_sha256") != run_signature
        or receipt.get("contracts", {}).get("birth_enabled") is not False
        or receipt.get("contracts", {}).get("fastsam_rerun") is not False
        or receipt.get("causality", {}).get("overall_pass") is not True
    ):
        raise F3RunnerError(f"resumed F3 scene contract differs: {scene}")
    inputs = receipt.get("inputs")
    if not isinstance(inputs, Mapping):
        raise F3RunnerError(f"resumed F3 input seals are absent: {scene}")
    for key in ("f2_sidecar", "f2_evidence_npz", "f0_sidecar", "schedule", "intrinsic"):
        row = inputs.get(key)
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("path"), str)
            or not isinstance(row.get("sha256"), str)
            or _sha256(_regular_file(Path(row["path"]), f"resumed {key}"))
            != row["sha256"]
        ):
            raise F3RunnerError(f"resumed F3 input changed: {scene}/{key}")
    return receipt, _sha256(source)


def run_f3(
    *,
    f2_receipt_path: Path,
    f2_oracle_path: Path,
    output_root: Path,
    shard_index: int = 0,
    num_shards: int = 2,
    resume: bool = False,
    plan_only: bool = False,
    tracker_factory: Callable[[], Any] | None = None,
    _expected_scene_count: int = EXPECTED_SCENES,
) -> dict[str, Any]:
    """Run one deterministic shard from sealed F2 evidence, without FastSAM."""

    if (
        isinstance(shard_index, bool)
        or isinstance(num_shards, bool)
        or not isinstance(shard_index, int)
        or not isinstance(num_shards, int)
        or num_shards < 1
        or shard_index < 0
        or shard_index >= num_shards
    ):
        raise F3RunnerError("invalid deterministic shard index/count")
    production = _expected_scene_count == EXPECTED_SCENES
    if production and num_shards != 2:
        raise F3RunnerError("production F3 requires exactly two deterministic shards")
    f2_receipt, f2_oracle, scenes, f2_rows = _load_f2_inputs(
        f2_receipt_path,
        f2_oracle_path,
        expected_scene_count=_expected_scene_count,
    )
    selected_indices = tuple(
        index for index in range(len(scenes)) if index % num_shards == shard_index
    )
    selected_scenes = tuple(scenes[index] for index in selected_indices)
    selected_rows = tuple(f2_rows[index] for index in selected_indices)
    plan = {
        "schema": SHARD_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "mode": "plan_only" if plan_only else "shadow",
        "paper100_scene_order": list(scenes),
        "f2_receipt": dict(f2_receipt),
        "f2_oracle": dict(f2_oracle),
        "shard": {
            "index": shard_index,
            "count": num_shards,
            "scene_indices": list(selected_indices),
            "scene_order": list(selected_scenes),
        },
    }
    if plan_only:
        print(json.dumps(plan, sort_keys=True), flush=True)
        return plan

    environment = _environment()
    source_receipts = _source_receipts()
    run_signature = _canonical_json_sha256(
        _signature_payload(
            scenes=scenes,
            f2_receipt=f2_receipt,
            f2_oracle=f2_oracle,
            sources=source_receipts,
            environment=environment,
        )
    )
    output = output_root.resolve()
    if output_root.is_symlink():
        raise F3RunnerError("F3 output root cannot be a symlink")
    scene_dir = output / "scenes"
    shard_dir = output / "shards"
    manifest_path = shard_dir / f"shard-{shard_index:03d}-of-{num_shards:03d}.json"
    sidecars = tuple(scene_dir / f"{scene}.json" for scene in selected_scenes)
    if manifest_path.exists() or manifest_path.is_symlink():
        if not resume:
            raise F3RunnerError(f"refusing to overwrite output: {manifest_path}")
        _path, manifest = _read_json(manifest_path, "resumed F3 shard")
        if (
            manifest.get("schema") != SHARD_SCHEMA
            or manifest.get("protocol_id") != PROTOCOL_ID
            or manifest.get("complete") is not True
            or manifest.get("run_signature_sha256") != run_signature
            or manifest.get("shard") != plan["shard"]
        ):
            raise F3RunnerError("resumed F3 shard contract differs")
        for ordinal, (scene_index, scene) in enumerate(
            zip(selected_indices, selected_scenes)
        ):
            receipt, digest = _resume_scene(
                sidecars[ordinal],
                scene=scene,
                scene_index=scene_index,
                run_signature=run_signature,
            )
            expected = _manifest_scene_row(
                receipt, sidecars[ordinal], digest, resumed=manifest["scenes"][ordinal].get("resumed", False)
            )
            if any(
                manifest["scenes"][ordinal].get(key) != value
                for key, value in expected.items()
                if key != "resumed"
            ):
                raise F3RunnerError(f"resumed F3 scene manifest differs: {scene}")
        return manifest

    present = tuple(path.exists() or path.is_symlink() for path in sidecars)
    if any(present) and not resume:
        raise F3RunnerError("refusing to overwrite existing F3 scene receipt")
    if any(path.is_symlink() for path in sidecars):
        raise F3RunnerError("F3 resume refuses symlink scene receipts")
    completed_prefix = 0
    while completed_prefix < len(present) and present[completed_prefix]:
        completed_prefix += 1
    if any(present[completed_prefix:]):
        raise F3RunnerError("F3 resume scenes must form an exact completed prefix")

    rows: list[dict[str, Any]] = []
    run_started = time.perf_counter()
    for ordinal in range(completed_prefix):
        scene = selected_scenes[ordinal]
        receipt, digest = _resume_scene(
            sidecars[ordinal],
            scene=scene,
            scene_index=selected_indices[ordinal],
            run_signature=run_signature,
        )
        rows.append(_manifest_scene_row(receipt, sidecars[ordinal], digest, resumed=True))
    for ordinal in range(completed_prefix, len(selected_scenes)):
        scene = selected_scenes[ordinal]
        receipt = _process_scene(
            scene=scene,
            scene_index=selected_indices[ordinal],
            f2_row=selected_rows[ordinal],
            run_signature=run_signature,
            f2_receipt=f2_receipt,
            f2_oracle=f2_oracle,
            source_receipts=source_receipts,
            tracker_factory=tracker_factory,
        )
        digest = _atomic_create_json(sidecars[ordinal], receipt)
        rows.append(_manifest_scene_row(receipt, sidecars[ordinal], digest, resumed=False))
        print(
            f"[{ordinal + 1}/{len(selected_scenes)}] {scene}: "
            f"frames={receipt['counts']['keyframe_count']} "
            f"sources={receipt['counts']['source_count']} "
            f"tracks={receipt['counts']['track_count']} written",
            flush=True,
        )

    for row in source_receipts.values():
        if _sha256(_regular_file(Path(row["path"]), "F3 frozen source")) != row["sha256"]:
            raise F3RunnerError("F3 source changed during execution")
    if _sha256(_regular_file(f2_receipt_path, "F2 receipt")) != f2_receipt["sha256"]:
        raise F3RunnerError("F2 receipt changed during F3 execution")
    if _sha256(_regular_file(f2_oracle_path, "F2 oracle")) != f2_oracle["sha256"]:
        raise F3RunnerError("F2 oracle changed during F3 execution")
    totals: Counter[str] = Counter()
    for row in rows:
        totals.update({key: int(value) for key, value in row["counts"].items()})
    if production and any(
        totals[key] != value for key, value in EXPECTED_SHARD_COUNTS[shard_index].items()
    ):
        raise F3RunnerError("F3 shard census differs from sealed paper100")
    manifest = {
        **plan,
        "mode": "shadow",
        "complete": True,
        "run_signature_sha256": run_signature,
        "environment": environment,
        "sources_receipt": source_receipts,
        "contracts": {
            "shadow_only": True,
            "observer_only": True,
            "birth_enabled": False,
            "fastsam_rerun": False,
            "h0_only": True,
            "ground_truth_access": False,
            "prediction_access": False,
            "rgb_access": False,
            "depth_pixel_access": False,
            "evaluator_access": False,
            "native_output_mutation": False,
            "training": False,
            "online_learning": False,
            "future_frame_logical_access": False,
            "hl_hlg_access": False,
        },
        "scenes": rows,
        "totals": dict(sorted(totals.items())),
        "runtime": {
            "wall_seconds": float(time.perf_counter() - run_started),
            "cpu_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "new_gpu_allocation_bytes": 0,
        },
        "conclusion_guardrail": (
            "F3 seals observer-only B/C projection hypotheses; AP is reserved "
            "for the separately preregistered GT oracle."
        ),
    }
    _atomic_create_json(manifest_path, manifest)
    print(f"Saved: {manifest_path}", flush=True)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sealed paper100 F3 projection shadow")
    parser.add_argument("--f2-receipt", type=Path, default=DEFAULT_F2_RECEIPT)
    parser.add_argument("--f2-oracle", type=Path, default=DEFAULT_F2_ORACLE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    run_f3(
        f2_receipt_path=args.f2_receipt,
        f2_oracle_path=args.f2_oracle,
        output_root=args.output_root,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        resume=args.resume,
        plan_only=args.plan_only,
    )


if __name__ == "__main__":
    main()
