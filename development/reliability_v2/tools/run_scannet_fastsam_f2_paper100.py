#!/usr/bin/env python3
"""Deterministically replay F0 paper100 evidence and run F2 DFU-LGF shadow.

F2 is a geometry-only counterfactual.  It reruns the frozen FastSAM provider
and the unchanged F0 residual core, requires exact equality with every sealed
F0 mask/candidate receipt, and only then computes the H0/HL/HLG hypotheses.
It never reads ground truth, predictions, an evaluator, labels, or semantics,
and it cannot create a birth or mutate detector output.

The first 100 rows of the sealed F0 full200 scene ledger are the paper100
universe.  Two shards are selected by original paper100 index modulo two.
Scene JSON and compressed evidence NPZ files use atomic create-only
publication; ``--resume`` accepts only an exact completed prefix.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
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

from boxfusion import fastsam_dfu_lgf_shadow as f2_core  # noqa: E402
from boxfusion import fastsam_residual_shadow as f0_core  # noqa: E402
import run_scannet_fastsam_f0_full200 as f0_runner  # noqa: E402


PROTOCOL_ID = "F2-DFU-LGF-lite-shadow-paper100"
SCENE_SCHEMA = "boxfusion.scannet_fastsam_f2_paper100.scene.v1"
EVIDENCE_SCHEMA = "boxfusion.scannet_fastsam_f2_paper100.evidence.v1"
SHARD_SCHEMA = "boxfusion.scannet_fastsam_f2_paper100.shard.v1"
EXPECTED_F0_MERGE_SCHEMA = "boxfusion.scannet_fastsam_f0_full200.merge.v1"
EXPECTED_F0_PROTOCOL_ID = (
    "F0-frozen-FastSAM-x-residual-automatic-mask-shadow-full200"
)
EXPECTED_F0_RECEIPT_SHA256 = (
    "07249ead31ad150cb43d7a35f4c922ac70a8a2f95bcf0fcd24f61f944c1e58a1"
)
EXPECTED_F0_RUNNER_SHA256 = (
    "638eb8670513aa03e3d20dbf47604fb0777a5487c9600143c7d3317ae6d5bf83"
)
EXPECTED_F0_CORE_SHA256 = (
    "a7cf6e3ae4777ee62ca5a1aa9dbc9a38e91cacc8ce77ab15cea940c838686e48"
)
EXPECTED_FASTSAM_PROVIDER_SHA256 = (
    "1e48f6676300dead2e77fad2a95be377d7d650980ba2377ab17cbb10c1f69f05"
)
EXPECTED_SCIPY_VERSION = "1.15.3"
EXPECTED_FULL200_SCENE_LIST_SHA256 = f0_runner.EXPECTED_SCENE_LIST_SHA256
EXPECTED_PAPER100_SCENES = 100
EXPECTED_PAPER100_KEYFRAMES = 6_817
EXPECTED_PAPER100_SUCCESSFUL_FRAMES = 6_726
EXPECTED_PAPER100_SOURCES = 52_299
EXPECTED_PAPER100_INVALID_POSE_FRAMES = 89
EXPECTED_PAPER100_NON_UPRIGHT_FRAMES = 2
EXPECTED_SHARD_COUNTS: Mapping[int, Mapping[str, int]] = {
    0: {"keyframes": 3_259, "successful_frames": 3_189, "sources": 24_863},
    1: {"keyframes": 3_558, "successful_frames": 3_537, "sources": 27_436},
}
MASK_PACKED_BYTES = f0_core.IMAGE_HEIGHT * f0_core.IMAGE_WIDTH // 8
SHARD_WARMUP_SUCCESSFUL_CALLS = f0_runner.SHARD_WARMUP_SUCCESSFUL_CALLS

DEFAULT_SCENE_LIST = f0_runner.DEFAULT_SCENE_LIST
DEFAULT_SCENE_ROOT = f0_runner.DEFAULT_SCENE_ROOT
DEFAULT_SCHEDULE_ROOTS = f0_runner.DEFAULT_SCHEDULE_ROOTS
DEFAULT_CHECKPOINT = f0_runner.DEFAULT_CHECKPOINT
DEFAULT_F0_RECEIPT = (
    REPOSITORY_ROOT
    / "logs/scannet_fastsam_f0_full200_score05/final/F0_FASTSAM_FULL200.json"
)
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "logs/scannet_fastsam_f2_paper100_score05"


class F2RunnerError(RuntimeError):
    """Raised when replay identity or a sealed F2 contract differs."""


def _validate_production_frozen_sources() -> None:
    from boxfusion import fastsam_automatic_provider as provider_module

    expected = {
        Path(f0_runner.__file__).resolve(): EXPECTED_F0_RUNNER_SHA256,
        Path(f0_core.__file__).resolve(): EXPECTED_F0_CORE_SHA256,
        Path(provider_module.__file__).resolve(): EXPECTED_FASTSAM_PROVIDER_SHA256,
    }
    for path, digest in expected.items():
        if _sha256(_regular_file(path, "frozen F0 source")) != digest:
            raise F2RunnerError(f"frozen F0 source SHA-256 differs: {path}")


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
        raise F2RunnerError(f"{label} must be a regular non-symlink file: {path}")
    result = path.resolve()
    if suffix is not None and result.suffix.lower() != suffix:
        raise F2RunnerError(f"{label} must be a {suffix} file: {result}")
    if result.suffix.lower() in {".pkl", ".pickle"}:
        raise F2RunnerError(f"prediction pickle input is forbidden: {result}")
    return result


def _read_json(path: Path, label: str) -> dict[str, Any]:
    source = _regular_file(path, label, ".json")
    try:
        result = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise F2RunnerError(f"invalid {label}: {source}") from error
    if not isinstance(result, dict):
        raise F2RunnerError(f"{label} must contain one JSON object: {source}")
    return result


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
            raise F2RunnerError(f"refusing to overwrite output: {path}") from error
        return _sha256(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_create_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise F2RunnerError(f"refusing to overwrite output: {path}") from error
        return _sha256(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_paper100_scene_list(
    path: Path, expected_scene_count: int
) -> tuple[tuple[str, ...], dict[str, Any]]:
    source = _regular_file(path, "F2 scene list")
    rows = tuple(
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    production = expected_scene_count == EXPECTED_PAPER100_SCENES
    if production:
        if len(rows) != f0_runner.EXPECTED_SCENE_COUNT:
            raise F2RunnerError("production F2 requires the sealed full200 ledger")
        if _sha256(source) != EXPECTED_FULL200_SCENE_LIST_SHA256:
            raise F2RunnerError("sealed full200 scene-list SHA-256 differs")
        selected = rows[:EXPECTED_PAPER100_SCENES]
    else:
        if len(rows) != expected_scene_count:
            raise F2RunnerError(
                f"expected {expected_scene_count} test scenes, found {len(rows)}"
            )
        selected = rows
    if len(set(selected)) != len(selected) or any(
        not row or "/" in row or "\\" in row or row in {".", ".."}
        for row in selected
    ):
        raise F2RunnerError("paper100 scene ledger is unsafe or contains duplicates")
    return selected, {
        "path": os.fspath(source),
        "sha256": _sha256(source),
        "source_scene_count": len(rows),
        "selected_prefix_count": len(selected),
        "selected_order_sha256": _canonical_json_sha256(list(selected)),
    }


def _load_f0_references(
    path: Path,
    scenes: Sequence[str],
    *,
    production: bool,
) -> tuple[dict[str, Any], dict[str, tuple[Path, str, dict[str, Any]]]]:
    source = _regular_file(path, "sealed F0 merged receipt", ".json")
    if production and _sha256(source) != EXPECTED_F0_RECEIPT_SHA256:
        raise F2RunnerError("sealed F0 merged receipt SHA-256 differs")
    receipt = _read_json(source, "sealed F0 merged receipt")
    coverage = receipt.get("coverage")
    rows = receipt.get("scenes")
    if (
        receipt.get("schema") != EXPECTED_F0_MERGE_SCHEMA
        or receipt.get("protocol_id") != EXPECTED_F0_PROTOCOL_ID
        or receipt.get("complete") is not True
        or receipt.get("overall_pass") is not True
        or not isinstance(coverage, dict)
        or not isinstance(rows, list)
        or coverage.get("scene_order", [])[: len(scenes)] != list(scenes)
        or [row.get("scene_id") for row in rows[: len(scenes)]] != list(scenes)
    ):
        raise F2RunnerError("F0 merged receipt does not seal the paper100 prefix")
    references: dict[str, tuple[Path, str, dict[str, Any]]] = {}
    for expected_index, (scene, row) in enumerate(zip(scenes, rows)):
        nested = row.get("sidecar")
        if (
            not isinstance(nested, dict)
            or row.get("scene_index") != expected_index
            or not isinstance(nested.get("path"), str)
            or not isinstance(nested.get("sha256"), str)
        ):
            raise F2RunnerError(f"invalid F0 merged scene row: {scene}")
        sidecar_path = _regular_file(
            Path(nested["path"]), f"F0 scene sidecar {scene}", ".json"
        )
        digest = _sha256(sidecar_path)
        if digest != nested["sha256"]:
            raise F2RunnerError(f"F0 scene sidecar rehash differs: {scene}")
        sidecar = _read_json(sidecar_path, f"F0 scene sidecar {scene}")
        if (
            sidecar.get("schema") != f0_runner.SCENE_SCHEMA
            or sidecar.get("protocol_id") != f0_runner.PROTOCOL_ID
            or sidecar.get("complete") is not True
            or sidecar.get("scene_id") != scene
            or sidecar.get("scene_index") != expected_index
        ):
            raise F2RunnerError(f"F0 scene sidecar contract differs: {scene}")
        references[scene] = (sidecar_path, digest, sidecar)
    return {
        "path": os.fspath(source),
        "sha256": _sha256(source),
        "schema": receipt["schema"],
        "protocol_id": receipt["protocol_id"],
        "run_signature_sha256": receipt.get("run_signature_sha256"),
    }, references


def _float_vector(value: object, label: str) -> list[float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise F2RunnerError(f"{label} must be a finite xyz vector")
    return array.tolist()


def _as_mapping(value: object, label: str) -> dict[str, Any]:
    if is_dataclass(value):
        result = asdict(value)
    elif isinstance(value, Mapping):
        result = dict(value)
    elif hasattr(value, "to_dict"):
        result = dict(value.to_dict())
    else:
        raise F2RunnerError(f"{label} is not serializable")
    return result


def _hypothesis_json(
    hypothesis: object,
    *,
    name: str,
    fallback_from: str | None,
) -> dict[str, Any]:
    row = _as_mapping(hypothesis, f"F2 {name} hypothesis")
    q02 = _float_vector(
        row.get("world_q02", getattr(hypothesis, "world_q02", None)),
        f"{name}.q02",
    )
    q98 = _float_vector(
        row.get("world_q98", getattr(hypothesis, "world_q98", None)),
        f"{name}.q98",
    )
    center = _float_vector(
        row.get("world_center", getattr(hypothesis, "world_center", None)),
        f"{name}.center",
    )
    extent = _float_vector(
        row.get("world_extent", getattr(hypothesis, "world_extent", None)),
        f"{name}.extent",
    )
    if np.any(np.asarray(q98) <= np.asarray(q02)):
        raise F2RunnerError(f"{name} geometry has a non-positive extent")
    retained = row.get("retained_indices", getattr(hypothesis, "retained_indices", None))
    if retained is None:
        raise F2RunnerError(f"{name} hypothesis has no retained-index ledger")
    indices = np.asarray(retained, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0):
        raise F2RunnerError(f"{name} retained indices are invalid")
    recorded_fallback = row.get(
        "fallback_from", getattr(hypothesis, "fallback_from", None)
    )
    applied = name == "H0" or recorded_fallback is None
    fallback = recorded_fallback is not None
    reason = str(row.get("reason", getattr(hypothesis, "reason", "ok")))
    if fallback and recorded_fallback is None:
        recorded_fallback = fallback_from
    diagnostics = {
        "applied": applied,
        "fallback": fallback,
        "fallback_from": recorded_fallback,
        "reason": reason,
        "retained_point_count": int(len(indices)),
        "source_point_count": int(
            row.get("source_point_count", getattr(hypothesis, "source_point_count", len(indices)))
        ),
    }
    return {
        "valid": True,
        "q02": q02,
        "q98": q98,
        "center": center,
        "extent": extent,
        "stored_point_count": int(len(indices)),
        "diagnostics": diagnostics,
        "_retained_indices": indices,
    }


def _refine_candidate(candidate: Any) -> Any:
    try:
        return f2_core.refine_fastsam_candidate(
            points_world=candidate.points_world,
            voxel_keys=candidate.voxel_keys,
            world_q02=candidate.world_q02,
            world_q98=candidate.world_q98,
        )
    except (TypeError, ValueError) as error:
        raise F2RunnerError("F2 core rejected an authenticated F0 candidate") from error


def _serialize_refined_candidate(
    candidate: Any, result: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build receipt/hash evidence outside the online ``complete_ms`` scope."""

    serialized = f2_core.dfu_lgf_result_to_dict(result)
    if (
        serialized.get("schema") != f2_core.SCHEMA
        or serialized.get("mode") != "shadow"
        or not isinstance(serialized.get("diagnostics"), dict)
    ):
        raise F2RunnerError("F2 core serializer contract differs")
    hypotheses = {
        "H0": _hypothesis_json(result.h0, name="H0", fallback_from=None),
        "HL": _hypothesis_json(result.hl, name="HL", fallback_from="H0"),
        "HLG": _hypothesis_json(result.hlg, name="HLG", fallback_from="HL"),
    }
    if not np.array_equal(
        np.asarray(hypotheses["H0"]["q02"], dtype=np.float64),
        np.asarray(candidate.world_q02, dtype=np.float64),
    ) or not np.array_equal(
        np.asarray(hypotheses["H0"]["q98"], dtype=np.float64),
        np.asarray(candidate.world_q98, dtype=np.float64),
    ):
        raise F2RunnerError("F2 H0 is not bit-identical to sealed F0 geometry")
    points = np.asarray(candidate.points_world, dtype=np.float64)
    keys = np.asarray(candidate.voxel_keys, dtype=np.int64)
    for name, row in hypotheses.items():
        indices = row["_retained_indices"]
        if (
            np.any(indices >= len(points))
            or len(np.unique(indices)) != len(indices)
            or (name == "H0" and not np.array_equal(indices, np.arange(len(points))))
        ):
            raise F2RunnerError(f"F2 {name} retained-index ledger differs")
        digest = hashlib.sha256()
        digest.update(np.asarray(points[indices], dtype="<f8").tobytes())
        digest.update(np.asarray(keys[indices], dtype="<i8").tobytes())
        row["points_and_voxel_keys_sha256"] = digest.hexdigest()
        row.pop("_retained_indices")
    result_receipt = {
        "schema": serialized["schema"],
        "mode": serialized["mode"],
        "input_sha256": serialized["input_sha256"],
        "result_sha256": serialized["result_sha256"],
        "diagnostics": serialized["diagnostics"],
    }
    return hypotheses, result_receipt


def _result_hlg_indices(result: Any, point_count: int) -> np.ndarray:
    value = getattr(result.hlg, "retained_indices", None)
    if value is None:
        row = _as_mapping(result.hlg, "F2 HLG hypothesis")
        value = row.get("retained_indices")
    indices = np.asarray(value, dtype=np.int64)
    if (
        indices.ndim != 1
        or not len(indices)
        or np.any(indices < 0)
        or np.any(indices >= point_count)
        or len(np.unique(indices)) != len(indices)
    ):
        raise F2RunnerError("F2 HLG retained-index ledger is invalid")
    return indices


def _source_id(scene: str, frame_id: int, raw_index: int) -> str:
    return f"{scene}/frame_{frame_id:06d}/raw_{raw_index:03d}"


def _funnel_identity(
    generated: Mapping[str, Any], sealed: object, scene: str, frame_id: int
) -> dict[str, Any]:
    if not isinstance(sealed, dict):
        raise F2RunnerError(f"F0 funnel is absent for {scene}/{frame_id}")
    generated_sha = _canonical_json_sha256(generated)
    sealed_sha = _canonical_json_sha256(sealed)
    if generated != sealed or generated_sha != sealed_sha:
        raise F2RunnerError(
            f"F0 deterministic replay differs for {scene}/{frame_id}: "
            f"{generated_sha} != {sealed_sha}"
        )
    return {
        "exact_equal": True,
        "generated_funnel_sha256": generated_sha,
        "sealed_funnel_sha256": sealed_sha,
        "mask_count": int(generated["input_mask_count"]),
        "selected_source_count": int(generated["selected_count"]),
    }


def _percentiles(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {
            "sample_count": 0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "sample_count": int(len(array)),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.quantile(array, 0.50)),
        "p95_ms": float(np.quantile(array, 0.95)),
        "max_ms": float(array.max()),
    }


def _environment(device: str, production: bool) -> dict[str, Any]:
    try:
        import scipy
    except ImportError as error:  # pragma: no cover - production dependency
        raise F2RunnerError("F2 exact-radius SciPy dependency is unavailable") from error
    if production and scipy.__version__ != EXPECTED_SCIPY_VERSION:
        raise F2RunnerError(
            "production F2 SciPy version differs from the pre-GT runtime amendment"
        )
    receipt = f0_runner._environment_receipt(device, production=production)
    receipt["scipy_version"] = scipy.__version__
    receipt["f2_local_index_backend"] = (
        "scipy.spatial.cKDTree.query_pairs(eps=0)+exact_squared_predicate"
    )
    return receipt


def _source_receipts(provider: Any) -> dict[str, dict[str, str]]:
    paths = {
        "runner": Path(__file__).resolve(),
        "f0_runner": Path(f0_runner.__file__).resolve(),
        "f0_core": Path(f0_core.__file__).resolve(),
        "f2_core": Path(f2_core.__file__).resolve(),
        "provider": Path(f0_runner._provider_source(provider)["path"]),
    }
    return {
        key: {"path": os.fspath(path), "sha256": _sha256(path)}
        for key, path in paths.items()
    }


def _signature_payload(
    *,
    scenes: Sequence[str],
    scene_list: Mapping[str, Any],
    schedules: Mapping[str, Mapping[str, Any]],
    scene_root: Path,
    f0_receipt: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, str]],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    environment_protocol = {
        key: value
        for key, value in environment.items()
        if key not in {"device", "gpu_uuid"}
    }
    return {
        "protocol_id": PROTOCOL_ID,
        "scene_order": list(scenes),
        "scene_list": dict(scene_list),
        "scene_root": os.fspath(scene_root),
        "schedule_manifests": [
            {
                "scene_id": scene,
                "path": os.fspath(schedules[scene]["path"]),
                "sha256": schedules[scene]["sha256"],
            }
            for scene in scenes
        ],
        "f0_receipt": dict(f0_receipt),
        "checkpoint": dict(checkpoint),
        "sources": {key: dict(value) for key, value in sources.items()},
        "environment_protocol": environment_protocol,
        "f0_schema": f0_core.SCHEMA,
        "f0_policy": dict(f0_core.POLICY),
        "f2_schema": f2_core.SCHEMA,
        "f2_policy": dict(f2_core.POLICY),
    }


def _evidence_arrays(
    *,
    scene: str,
    source_ids: Sequence[str],
    frame_ids: Sequence[int],
    raw_indices: Sequence[int],
    ranks: Sequence[int],
    candidate_indices: Sequence[int],
    packed_masks: Sequence[np.ndarray],
    points_world: Sequence[np.ndarray],
    voxel_keys: Sequence[np.ndarray],
    hl_retained_indices: Sequence[np.ndarray],
    hlg_retained_indices: Sequence[np.ndarray],
) -> dict[str, np.ndarray]:
    count = len(source_ids)
    if any(
        len(values) != count
        for values in (
            frame_ids,
            raw_indices,
            ranks,
            candidate_indices,
            packed_masks,
            points_world,
            voxel_keys,
            hl_retained_indices,
            hlg_retained_indices,
        )
    ):
        raise F2RunnerError(f"evidence ledger lengths differ for {scene}")
    masks = (
        np.stack(packed_masks).astype(np.uint8, copy=False)
        if count
        else np.empty((0, MASK_PACKED_BYTES), dtype=np.uint8)
    )
    if masks.shape != (count, MASK_PACKED_BYTES):
        raise F2RunnerError(f"packed mask shape differs for {scene}")
    offsets = np.zeros(count + 1, dtype=np.int64)
    if count:
        offsets[1:] = np.cumsum(
            np.asarray([len(points) for points in points_world], dtype=np.int64)
        )
    points_flat = (
        np.concatenate(points_world, axis=0).astype("<f8", copy=False)
        if offsets[-1]
        else np.empty((0, 3), dtype="<f8")
    )
    keys_flat = (
        np.concatenate(voxel_keys, axis=0).astype("<i8", copy=False)
        if offsets[-1]
        else np.empty((0, 3), dtype="<i8")
    )
    if points_flat.shape != keys_flat.shape or points_flat.shape != (offsets[-1], 3):
        raise F2RunnerError(f"cleaned point evidence shape differs for {scene}")
    hl_offsets = np.zeros(count + 1, dtype=np.int64)
    hlg_offsets = np.zeros(count + 1, dtype=np.int64)
    if count:
        hl_offsets[1:] = np.cumsum(
            np.asarray([len(value) for value in hl_retained_indices], dtype=np.int64)
        )
        hlg_offsets[1:] = np.cumsum(
            np.asarray([len(value) for value in hlg_retained_indices], dtype=np.int64)
        )
    hl_flat = (
        np.concatenate(hl_retained_indices).astype("<i8", copy=False)
        if hl_offsets[-1]
        else np.empty(0, dtype="<i8")
    )
    hlg_flat = (
        np.concatenate(hlg_retained_indices).astype("<i8", copy=False)
        if hlg_offsets[-1]
        else np.empty(0, dtype="<i8")
    )
    width = max((len(value) for value in source_ids), default=1)
    return {
        "schema": np.asarray(EVIDENCE_SCHEMA),
        "scene_id": np.asarray(scene),
        "mask_shape": np.asarray(
            [f0_core.IMAGE_HEIGHT, f0_core.IMAGE_WIDTH], dtype=np.int16
        ),
        "mask_bitorder": np.asarray("little"),
        "source_ids": np.asarray(source_ids, dtype=f"U{width}"),
        "frame_ids": np.asarray(frame_ids, dtype=np.int32),
        "raw_indices": np.asarray(raw_indices, dtype=np.int16),
        "ranks": np.asarray(ranks, dtype=np.int16),
        "candidate_indices": np.asarray(candidate_indices, dtype=np.int16),
        "masks_packbits": masks,
        "point_offsets": offsets,
        "points_world": points_flat,
        "voxel_keys": keys_flat,
        "hl_index_offsets": hl_offsets,
        "hl_retained_indices": hl_flat,
        "hlg_index_offsets": hlg_offsets,
        "hlg_retained_indices": hlg_flat,
    }


def _process_scene(
    *,
    scene: str,
    scene_index: int,
    schedule: Mapping[str, Any],
    f0_reference: tuple[Path, str, Mapping[str, Any]],
    scene_root: Path,
    provider: Any,
    device: str,
    run_signature: str,
    warmup_state: dict[str, int],
    environment: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    sources_receipt: Mapping[str, Mapping[str, str]],
    evidence_path: Path,
) -> tuple[dict[str, Any], str]:
    f0_path, f0_sha, f0_scene = f0_reference
    f0_frames = f0_scene.get("frames")
    if not isinstance(f0_frames, list) or [row.get("frame_id") for row in f0_frames] != list(
        schedule["frames"]
    ):
        raise F2RunnerError(f"F0/schedule frame ledger differs for {scene}")
    intrinsic_path, intrinsic = f0_runner._load_intrinsic(scene_root, scene)
    frames: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    runtimes: defaultdict[str, list[float]] = defaultdict(list)
    all_source_ids: list[str] = []
    all_frame_ids: list[int] = []
    all_raw_indices: list[int] = []
    all_ranks: list[int] = []
    all_candidate_indices: list[int] = []
    all_masks: list[np.ndarray] = []
    all_points: list[np.ndarray] = []
    all_keys: list[np.ndarray] = []
    all_hl_indices: list[np.ndarray] = []
    all_hlg_indices: list[np.ndarray] = []
    seen_source_ids: set[str] = set()
    f2_fallbacks: Counter[str] = Counter()
    f2_applied: Counter[str] = Counter()
    f2_changed: Counter[str] = Counter()
    f2_core_ms_total = 0.0
    evidence_prepare_ms_total = 0.0
    f0_replay_mask_count = 0

    f0_source_by_frame = {row["frame_id"]: row for row in f0_frames}
    f0_runner._reset_gpu_peak(device)
    for ordinal, frame_id in enumerate(schedule["frames"]):
        receipt_started = time.perf_counter()
        decode_started = time.perf_counter()
        bgr, depth_mm, paths = f0_runner._decode_frame(scene_root, scene, frame_id)
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        cache_started = time.perf_counter()
        (
            cache_path,
            cutr_boxes,
            cache_sha,
            cache_image_size,
            sealed_input_signature,
        ) = f0_runner._load_cutr_boxes(schedule, scene, frame_id)
        cache_ms = (time.perf_counter() - cache_started) * 1000.0
        current_pose = f0_runner._read_pose(paths["pose"])
        reconstructed_signature, metadata = f0_runner._reconstruct_cutr_input_signature(
            bgr=bgr,
            depth_mm=depth_mm,
            intrinsic=intrinsic,
            scene_root=scene_root,
            scene=scene,
            frame_id=frame_id,
        )
        f0_frame = f0_source_by_frame[frame_id]
        f0_inputs = f0_frame.get("inputs")
        if (
            reconstructed_signature != sealed_input_signature
            or not isinstance(f0_inputs, dict)
            or f0_inputs.get("rgb_sha256") != _sha256(paths["rgb"])
            or f0_inputs.get("depth_sha256") != _sha256(paths["depth"])
            or f0_inputs.get("pose_sha256") != _sha256(paths["pose"])
            or f0_inputs.get("cutr_cache_sha256") != cache_sha
            or f0_inputs.get("cutr_input_signature") != reconstructed_signature
        ):
            raise F2RunnerError(f"sealed input identity differs for {scene}/{frame_id}")
        orientation = int(metadata["producer_orientation"])
        successful = current_pose is not None and orientation == 0
        if bool(f0_frame.get("successful")) != successful:
            raise F2RunnerError(f"F0 abstention replay differs for {scene}/{frame_id}")
        counts["keyframes"] += 1
        frame: dict[str, Any] = {
            "frame_id": int(frame_id),
            "frame_ordinal": ordinal,
            "successful": successful,
            "abstention": None if successful else f0_frame.get("abstention"),
            "sources": [],
        }
        if not successful:
            counts[str(f0_frame.get("abstention"))] += 1
            receipt_total_ms = (time.perf_counter() - receipt_started) * 1000.0
            frame["identity"] = {"exact_equal": True, "selected_source_count": 0}
            frame["runtime"] = {
                "decode_ms": decode_ms,
                "cache_ms": cache_ms,
                "provider_ms": 0.0,
                "f0_core_ms": 0.0,
                "f2_core_ms": 0.0,
                "complete_ms": 0.0,
                "replay_identity_ms": 0.0,
                "evidence_prepare_ms": 0.0,
                "receipt_total_ms": receipt_total_ms,
                "provider_call_index_in_shard": None,
                "warmup_excluded": False,
            }
            frames.append(frame)
            continue

        provider_started = time.perf_counter()
        masks, confidences, provider_boxes, provider_timing = f0_runner._provider_predict(
            provider, bgr
        )
        f0_runner._validate_provider_timing_environment(provider_timing, environment)
        provider_ms = (time.perf_counter() - provider_started) * 1000.0
        f0_started = time.perf_counter()
        try:
            f0_result = f0_core.select_and_lift_residual_masks(
                masks=masks,
                confidences=confidences,
                depth_m=depth_mm.astype(np.float32) / 1000.0,
                explained_boxes_xyxy=cutr_boxes,
                intrinsics=intrinsic,
                camera_to_world=current_pose,
            )
        except ValueError as error:
            raise F2RunnerError(
                f"F0 core rejected replay input {scene}/{frame_id}"
            ) from error
        f0_core_ms = (time.perf_counter() - f0_started) * 1000.0
        # Authenticate the entire F0 funnel before any F2 geometry is applied.
        # This audit wall time is recorded separately and excluded from the
        # online gate.
        replay_identity_started = time.perf_counter()
        generated_funnel = f0_runner._frame_funnel(f0_result, provider_boxes)
        identity = _funnel_identity(
            generated_funnel, f0_frame.get("funnel"), scene, frame_id
        )
        f0_replay_mask_count += identity["mask_count"]
        counts["successful_frames"] += 1
        counts["identity_verified_frames"] += 1
        sealed_candidates = f0_frame["funnel"]["candidates"]
        if len(sealed_candidates) != len(f0_result.candidates):
            raise F2RunnerError(f"F0 candidate count differs for {scene}/{frame_id}")
        for candidate_index, (candidate, sealed_candidate) in enumerate(
            zip(f0_result.candidates, sealed_candidates)
        ):
            if f0_runner._candidate_json(candidate) != sealed_candidate:
                raise F2RunnerError(
                    f"F0 selected candidate differs for {scene}/{frame_id}/{candidate_index}"
                )
        replay_identity_ms = (time.perf_counter() - replay_identity_started) * 1000.0

        f2_started = time.perf_counter()
        refined: list[tuple[Any, Any]] = []
        for candidate in f0_result.candidates:
            refined.append((candidate, _refine_candidate(candidate)))
        f2_core_ms = (time.perf_counter() - f2_started) * 1000.0
        f2_core_ms_total += f2_core_ms
        # All three operational components are individually timed after the
        # provider's CUDA synchronization contract.  Replay authentication and
        # receipt construction are intentionally outside this gate.
        complete_ms = provider_ms + f0_core_ms + f2_core_ms

        evidence_started = time.perf_counter()
        source_rows: list[dict[str, Any]] = []
        for candidate_index, (candidate, f2_result) in enumerate(refined):
            hypotheses, f2_receipt = _serialize_refined_candidate(
                candidate, f2_result
            )
            source_id = _source_id(scene, frame_id, int(candidate.raw_index))
            if source_id in seen_source_ids:
                raise F2RunnerError(f"duplicate F2 source_id: {source_id}")
            seen_source_ids.add(source_id)
            hlg_indices = _result_hlg_indices(f2_result, candidate.stored_point_count)
            packed = np.packbits(
                masks[candidate.raw_index].reshape(-1), bitorder="little"
            )
            if packed.shape != (MASK_PACKED_BYTES,) or hashlib.sha256(
                packed.tobytes()
            ).hexdigest() != candidate.mask_sha256:
                raise F2RunnerError(f"packed mask identity differs: {source_id}")
            for hypothesis_name in ("HL", "HLG"):
                diagnostics = hypotheses[hypothesis_name]["diagnostics"]
                if diagnostics["applied"]:
                    f2_applied[hypothesis_name] += 1
                if diagnostics["fallback"]:
                    f2_fallbacks[hypothesis_name] += 1
                if not np.array_equal(
                    np.asarray(hypotheses[hypothesis_name]["q02"]),
                    np.asarray(hypotheses["H0"]["q02"]),
                ) or not np.array_equal(
                    np.asarray(hypotheses[hypothesis_name]["q98"]),
                    np.asarray(hypotheses["H0"]["q98"]),
                ):
                    f2_changed[hypothesis_name] += 1
            source_rows.append(
                {
                    "source_id": source_id,
                    "candidate_index": candidate_index,
                    "raw_index": int(candidate.raw_index),
                    "rank": int(candidate.rank),
                    "confidence": float(candidate.confidence),
                    "mask_sha256": candidate.mask_sha256,
                    "points_and_voxel_keys_sha256": candidate.points_sha256,
                    "stored_point_count": int(candidate.stored_point_count),
                    "f0_world_q02": candidate.world_q02.tolist(),
                    "f0_world_q98": candidate.world_q98.tolist(),
                    "f2_receipt": f2_receipt,
                    "hypotheses": hypotheses,
                }
            )
            all_source_ids.append(source_id)
            all_frame_ids.append(frame_id)
            all_raw_indices.append(int(candidate.raw_index))
            all_ranks.append(int(candidate.rank))
            all_candidate_indices.append(candidate_index)
            all_masks.append(np.asarray(packed, dtype=np.uint8))
            all_points.append(
                np.ascontiguousarray(candidate.points_world, dtype=np.float64)
            )
            all_keys.append(
                np.ascontiguousarray(candidate.voxel_keys, dtype=np.int64)
            )
            all_hl_indices.append(
                np.ascontiguousarray(f2_result.hl.retained_indices, dtype=np.int64)
            )
            all_hlg_indices.append(np.ascontiguousarray(hlg_indices, dtype=np.int64))
        evidence_prepare_ms = (time.perf_counter() - evidence_started) * 1000.0
        evidence_prepare_ms_total += evidence_prepare_ms
        counts["sources"] += len(source_rows)
        counts["identity_verified_sources"] += len(source_rows)
        provider_call_index = warmup_state["successful_provider_calls"]
        warmup_excluded = provider_call_index < SHARD_WARMUP_SUCCESSFUL_CALLS
        warmup_state["successful_provider_calls"] += 1
        frame.update(
            {
                "identity": identity,
                "sources": source_rows,
                "runtime": {
                    "decode_ms": decode_ms,
                    "cache_ms": cache_ms,
                    "provider_ms": provider_ms,
                    "f0_core_ms": f0_core_ms,
                    "f2_core_ms": f2_core_ms,
                    "complete_ms": complete_ms,
                    "replay_identity_ms": replay_identity_ms,
                    "evidence_prepare_ms": evidence_prepare_ms,
                    "receipt_total_ms": (
                        time.perf_counter() - receipt_started
                    )
                    * 1000.0,
                    "provider_call_index_in_shard": provider_call_index,
                    "warmup_excluded": warmup_excluded,
                },
            }
        )
        frames.append(frame)
        if not warmup_excluded:
            for key, value in frame["runtime"].items():
                if key.endswith("_ms") and key not in {
                    "evidence_prepare_ms",
                    "receipt_total_ms",
                    "decode_ms",
                    "cache_ms",
                }:
                    runtimes[key].append(float(value))

    arrays = _evidence_arrays(
        scene=scene,
        source_ids=all_source_ids,
        frame_ids=all_frame_ids,
        raw_indices=all_raw_indices,
        ranks=all_ranks,
        candidate_indices=all_candidate_indices,
        packed_masks=all_masks,
        points_world=all_points,
        voxel_keys=all_keys,
        hl_retained_indices=all_hl_indices,
        hlg_retained_indices=all_hlg_indices,
    )
    serialization_started = time.perf_counter()
    evidence_sha = _atomic_create_npz(evidence_path, arrays)
    evidence_serialization_ms = (time.perf_counter() - serialization_started) * 1000.0
    counts["replayed_raw_masks"] = f0_replay_mask_count
    counts["invalid_pose_frames"] = counts.pop("invalid_current_pose", 0)
    counts["non_upright_producer_frames"] = counts.pop(
        "non_upright_cache_coordinate_frame", 0
    )
    summary = {
        "counts": dict(sorted(counts.items())),
        "identity_ratio": (
            counts["identity_verified_sources"] / counts["sources"]
            if counts["sources"]
            else 1.0
        ),
        "f2_applied_counts": dict(sorted(f2_applied.items())),
        "f2_fallback_counts": dict(sorted(f2_fallbacks.items())),
        "f2_geometry_changed_counts": dict(sorted(f2_changed.items())),
        "runtime": {
            key: _percentiles(values) for key, values in sorted(runtimes.items())
        },
        "f2_core_ms_total": float(f2_core_ms_total),
        "evidence_prepare_ms_total": float(evidence_prepare_ms_total),
        "evidence_serialization_ms": float(evidence_serialization_ms),
        "cpu_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "gpu_peak_memory_bytes": int(f0_runner._gpu_peak_bytes(device)),
    }
    return {
        "schema": SCENE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "run_signature_sha256": run_signature,
        "scene_id": scene,
        "scene_index": scene_index,
        "frame_id_ledger_sha256": f0_runner._frame_id_ledger_sha256(schedule["frames"]),
        "f0_sidecar": {"path": os.fspath(f0_path), "sha256": f0_sha},
        "evidence_npz": {
            "schema": EVIDENCE_SCHEMA,
            "path": os.fspath(evidence_path.resolve()),
            "sha256": evidence_sha,
            "source_count": len(all_source_ids),
            "mask_shape": [f0_core.IMAGE_HEIGHT, f0_core.IMAGE_WIDTH],
            "mask_bitorder": "little",
            "mask_packed_bytes_per_source": MASK_PACKED_BYTES,
            "raw_point_count": int(arrays["point_offsets"][-1]),
            "hl_retained_index_count": int(arrays["hl_index_offsets"][-1]),
            "hlg_retained_index_count": int(arrays["hlg_index_offsets"][-1]),
        },
        "schedule": {
            "path": os.fspath(schedule["path"]),
            "sha256": schedule["sha256"],
            "keyframe_count": len(schedule["frames"]),
        },
        "intrinsic": {
            "path": os.fspath(intrinsic_path),
            "sha256": _sha256(intrinsic_path),
        },
        "environment_sha256": _canonical_json_sha256(environment),
        "checkpoint": dict(checkpoint),
        "sources_receipt": {
            key: dict(value) for key, value in sources_receipt.items()
        },
        "contracts": {
            "shadow_only": True,
            "birth_enabled": False,
            "ground_truth_access": False,
            "prediction_access": False,
            "evaluator_access": False,
            "semantic_or_clip_access": False,
            "training": False,
            "online_learning": False,
            "current_and_past_history_access": False,
            "f0_exact_replay_required": True,
            "native_output_mutation": False,
        },
        "frames": frames,
        "summary": summary,
    }, evidence_sha


def _scene_manifest_row(
    receipt: Mapping[str, Any], sidecar_path: Path, sidecar_sha: str
) -> dict[str, Any]:
    return {
        "scene_id": receipt["scene_id"],
        "scene_index": receipt["scene_index"],
        "sidecar_path": os.fspath(sidecar_path.resolve()),
        "sidecar_sha256": sidecar_sha,
        "evidence_npz_path": receipt["evidence_npz"]["path"],
        "evidence_npz_sha256": receipt["evidence_npz"]["sha256"],
        "frame_id_ledger_sha256": receipt["frame_id_ledger_sha256"],
        "counts": receipt["summary"]["counts"],
        "runtime": receipt["summary"]["runtime"],
        "cpu_peak_rss_bytes": receipt["summary"]["cpu_peak_rss_bytes"],
        "gpu_peak_memory_bytes": receipt["summary"]["gpu_peak_memory_bytes"],
    }


def _resume_scene(
    *,
    path: Path,
    evidence_path: Path,
    scene: str,
    scene_index: int,
    run_signature: str,
    schedule: Mapping[str, Any],
    provider_call_start: int,
    environment: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    sources_receipt: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, Any], str, int]:
    receipt = _read_json(path, f"resumed F2 scene {scene}")
    evidence = _regular_file(evidence_path, f"resumed F2 evidence {scene}", ".npz")
    if (
        receipt.get("schema") != SCENE_SCHEMA
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("complete") is not True
        or receipt.get("scene_id") != scene
        or receipt.get("scene_index") != scene_index
        or receipt.get("run_signature_sha256") != run_signature
        or receipt.get("environment_sha256") != _canonical_json_sha256(environment)
        or receipt.get("checkpoint") != dict(checkpoint)
        or receipt.get("sources_receipt")
        != {key: dict(value) for key, value in sources_receipt.items()}
        or receipt.get("schedule", {}).get("sha256") != schedule["sha256"]
        or receipt.get("evidence_npz", {}).get("path") != os.fspath(evidence)
        or receipt.get("evidence_npz", {}).get("sha256") != _sha256(evidence)
    ):
        raise F2RunnerError(f"resumed F2 scene contract differs: {scene}")
    frames = receipt.get("frames")
    if not isinstance(frames, list) or [row.get("frame_id") for row in frames] != list(
        schedule["frames"]
    ):
        raise F2RunnerError(f"resumed F2 frame ledger differs: {scene}")
    successful = [row for row in frames if row.get("successful") is True]
    expected_calls = list(range(provider_call_start, provider_call_start + len(successful)))
    if [row["runtime"]["provider_call_index_in_shard"] for row in successful] != expected_calls:
        raise F2RunnerError(f"resumed F2 provider ledger differs: {scene}")
    if any(
        row["runtime"]["warmup_excluded"]
        != (index < SHARD_WARMUP_SUCCESSFUL_CALLS)
        for row, index in zip(successful, expected_calls)
    ):
        raise F2RunnerError(f"resumed F2 warmup ledger differs: {scene}")
    source_count = sum(len(row.get("sources", [])) for row in frames)
    if (
        source_count != receipt["summary"]["counts"]["sources"]
        or receipt["summary"]["counts"]["identity_verified_sources"] != source_count
    ):
        raise F2RunnerError(f"resumed F2 source ledger differs: {scene}")
    return receipt, _sha256(path), len(successful)


def run_f2(
    *,
    schedule_roots: Sequence[Path],
    scene_root: Path,
    scene_list_path: Path,
    f0_receipt_path: Path,
    output_root: Path,
    device: str,
    shard_index: int = 0,
    num_shards: int = 2,
    resume: bool = False,
    plan_only: bool = False,
    provider_factory: Callable[[Path, str], Any] | None = None,
    _expected_scene_count: int = EXPECTED_PAPER100_SCENES,
) -> dict[str, Any]:
    """Run one deterministic paper100 F2 replay shard."""

    f0_runner._validate_shard(shard_index, num_shards)
    production = _expected_scene_count == EXPECTED_PAPER100_SCENES
    if production:
        _validate_production_frozen_sources()
    if production and num_shards != 2:
        raise F2RunnerError("production F2 requires exactly two deterministic shards")
    scenes, scene_list = _read_paper100_scene_list(
        scene_list_path, _expected_scene_count
    )
    f0_receipt, f0_references = _load_f0_references(
        f0_receipt_path, scenes, production=production
    )
    schedules = f0_runner._resolve_schedules(schedule_roots, scenes)
    total_keyframes = sum(len(schedules[scene]["frames"]) for scene in scenes)
    if production and total_keyframes != EXPECTED_PAPER100_KEYFRAMES:
        raise F2RunnerError("paper100 schedule keyframe census differs")
    selected_indices = tuple(
        index for index in range(len(scenes)) if index % num_shards == shard_index
    )
    selected_scenes = tuple(scenes[index] for index in selected_indices)
    plan = {
        "schema": SHARD_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "mode": "plan_only" if plan_only else "shadow",
        "scene_list": scene_list,
        "paper100_scene_order": list(scenes),
        "paper100_keyframe_count": total_keyframes,
        "f0_receipt": f0_receipt,
        "shard": {
            "index": shard_index,
            "count": num_shards,
            "scene_indices": list(selected_indices),
            "scene_order": list(selected_scenes),
        },
        "shard_keyframe_count": sum(
            len(schedules[scene]["frames"]) for scene in selected_scenes
        ),
        "schedule_roots": [os.fspath(Path(root).resolve()) for root in schedule_roots],
        "scene_root": os.fspath(scene_root.resolve()),
    }
    if plan_only:
        print(json.dumps(plan, sort_keys=True), flush=True)
        return plan
    if production and provider_factory is None and device != f"cuda:{shard_index}":
        raise F2RunnerError("production F2 binds shard index to cuda:0/cuda:1")

    output = output_root.resolve()
    if output_root.is_symlink():
        raise F2RunnerError("F2 output root cannot be a symlink")
    scene_dir = output / "scenes"
    arrays_dir = output / "arrays"
    shard_dir = output / "shards"
    manifest_path = shard_dir / f"shard-{shard_index:03d}-of-{num_shards:03d}.json"
    sidecars = tuple(scene_dir / f"{scene}.json" for scene in selected_scenes)
    evidence_paths = tuple(arrays_dir / f"{scene}.npz" for scene in selected_scenes)
    environment = _environment(device, production=provider_factory is None)
    if manifest_path.exists() or manifest_path.is_symlink():
        if not resume:
            raise F2RunnerError(f"refusing to overwrite output: {manifest_path}")
        manifest = _read_json(manifest_path, "resumed F2 shard manifest")
        checkpoint = manifest.get("checkpoint")
        sources_receipt = manifest.get("sources_receipt")
        if not isinstance(checkpoint, dict) or not isinstance(sources_receipt, dict):
            raise F2RunnerError("resumed F2 shard identity is absent")
        for row in list(sources_receipt.values()) + [checkpoint]:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("path"), str)
                or not isinstance(row.get("sha256"), str)
                or _sha256(_regular_file(Path(row["path"]), "resumed frozen input"))
                != row["sha256"]
            ):
                raise F2RunnerError("resumed F2 execution identity changed")
        expected_signature = _canonical_json_sha256(
            _signature_payload(
                scenes=scenes,
                scene_list=scene_list,
                schedules=schedules,
                scene_root=scene_root.resolve(),
                f0_receipt=f0_receipt,
                checkpoint=checkpoint,
                sources=sources_receipt,
                environment=environment,
            )
        )
        manifest_rows = manifest.get("scenes")
        if (
            manifest.get("schema") != SHARD_SCHEMA
            or manifest.get("protocol_id") != PROTOCOL_ID
            or manifest.get("mode") != "shadow"
            or manifest.get("complete") is not True
            or manifest.get("run_signature_sha256") != expected_signature
            or manifest.get("environment") != environment
            or manifest.get("scene_list") != scene_list
            or manifest.get("paper100_scene_order") != list(scenes)
            or manifest.get("shard") != plan["shard"]
            or not isinstance(manifest_rows, list)
            or len(manifest_rows) != len(selected_scenes)
        ):
            raise F2RunnerError("resumed F2 shard manifest contract differs")
        provider_call_start = 0
        recomputed_totals: Counter[str] = Counter()
        for ordinal, (scene_index, scene, row) in enumerate(
            zip(selected_indices, selected_scenes, manifest_rows)
        ):
            if not isinstance(row, dict) or row.get("scene_id") != scene:
                raise F2RunnerError("resumed F2 shard scene order differs")
            receipt, sidecar_sha, successful_calls = _resume_scene(
                path=sidecars[ordinal],
                evidence_path=evidence_paths[ordinal],
                scene=scene,
                scene_index=scene_index,
                run_signature=expected_signature,
                schedule=schedules[scene],
                provider_call_start=provider_call_start,
                environment=environment,
                checkpoint=checkpoint,
                sources_receipt=sources_receipt,
            )
            expected_row = _scene_manifest_row(receipt, sidecars[ordinal], sidecar_sha)
            if any(row.get(key) != value for key, value in expected_row.items()):
                raise F2RunnerError(f"resumed F2 shard scene row differs: {scene}")
            recomputed_totals.update(
                {key: int(value) for key, value in expected_row["counts"].items()}
            )
            provider_call_start += successful_calls
        if manifest.get("totals") != dict(sorted(recomputed_totals.items())):
            raise F2RunnerError("resumed F2 shard totals differ")
        return manifest
    exists = tuple(
        sidecar.exists() or sidecar.is_symlink() or evidence.exists() or evidence.is_symlink()
        for sidecar, evidence in zip(sidecars, evidence_paths)
    )
    pairs_complete = tuple(
        (sidecar.exists() and not sidecar.is_symlink())
        and (evidence.exists() and not evidence.is_symlink())
        for sidecar, evidence in zip(sidecars, evidence_paths)
    )
    if any(exists) and not resume:
        raise F2RunnerError("refusing to overwrite existing F2 scene evidence")
    if any(present and not complete for present, complete in zip(exists, pairs_complete)):
        raise F2RunnerError("resume requires complete JSON+NPZ scene pairs")
    completed_prefix = 0
    while completed_prefix < len(pairs_complete) and pairs_complete[completed_prefix]:
        completed_prefix += 1
    if any(pairs_complete[completed_prefix:]):
        raise F2RunnerError("resume scene pairs must form an exact completed prefix")

    provider: Any | None = None
    pending_count = len(selected_scenes) - completed_prefix
    if pending_count:
        factory = provider_factory or f0_runner._default_provider_factory
        provider = factory(DEFAULT_CHECKPOINT, device)
        checkpoint = f0_runner._checkpoint_metadata(provider)
        sources_receipt = _source_receipts(provider)
    elif completed_prefix:
        identity = _read_json(sidecars[0], "resumed F2 execution identity")
        checkpoint = identity["checkpoint"]
        sources_receipt = identity["sources_receipt"]
        for row in list(sources_receipt.values()) + [checkpoint]:
            if _sha256(_regular_file(Path(row["path"]), "resumed frozen input")) != row["sha256"]:
                raise F2RunnerError("resumed F2 execution identity changed")
    else:
        raise F2RunnerError("empty F2 shard has no execution identity")
    run_signature = _canonical_json_sha256(
        _signature_payload(
            scenes=scenes,
            scene_list=scene_list,
            schedules=schedules,
            scene_root=scene_root.resolve(),
            f0_receipt=f0_receipt,
            checkpoint=checkpoint,
            sources=sources_receipt,
            environment=environment,
        )
    )

    rows: list[dict[str, Any]] = []
    warmup_state = {"successful_provider_calls": 0}
    run_started = time.perf_counter()
    for ordinal in range(completed_prefix):
        scene = selected_scenes[ordinal]
        receipt, sidecar_sha, successful_calls = _resume_scene(
            path=sidecars[ordinal],
            evidence_path=evidence_paths[ordinal],
            scene=scene,
            scene_index=selected_indices[ordinal],
            run_signature=run_signature,
            schedule=schedules[scene],
            provider_call_start=warmup_state["successful_provider_calls"],
            environment=environment,
            checkpoint=checkpoint,
            sources_receipt=sources_receipt,
        )
        warmup_state["successful_provider_calls"] += successful_calls
        row = _scene_manifest_row(receipt, sidecars[ordinal], sidecar_sha)
        row["resumed"] = True
        rows.append(row)

    resume_rewarm: dict[str, Any]
    if completed_prefix and pending_count:
        first_scene = selected_scenes[completed_prefix]
        assert provider is not None
        resume_rewarm = f0_runner._resume_rewarm(
            provider=provider,
            environment=environment,
            scene_root=scene_root.resolve(),
            scene=first_scene,
            frame_id=schedules[first_scene]["frames"][0],
            completed_scene_count=completed_prefix,
            pending_scene_count=pending_count,
        )
    else:
        resume_rewarm = {
            "required": False,
            "completed_scene_count": completed_prefix,
            "pending_scene_count": pending_count,
            "call_count": 0,
            "excluded_from_runtime": True,
        }

    for ordinal in range(completed_prefix, len(selected_scenes)):
        scene = selected_scenes[ordinal]
        assert provider is not None
        receipt, _evidence_sha = _process_scene(
            scene=scene,
            scene_index=selected_indices[ordinal],
            schedule=schedules[scene],
            f0_reference=f0_references[scene],
            scene_root=scene_root.resolve(),
            provider=provider,
            device=device,
            run_signature=run_signature,
            warmup_state=warmup_state,
            environment=environment,
            checkpoint=checkpoint,
            sources_receipt=sources_receipt,
            evidence_path=evidence_paths[ordinal],
        )
        sidecar_sha = _atomic_create_json(sidecars[ordinal], receipt)
        row = _scene_manifest_row(receipt, sidecars[ordinal], sidecar_sha)
        row["resumed"] = False
        rows.append(row)
        counts = row["counts"]
        print(
            f"[{ordinal + 1}/{len(selected_scenes)}] {scene}: "
            f"frames={counts['keyframes']} sources={counts['sources']} identity=100% written",
            flush=True,
        )

    for row in sources_receipt.values():
        if _sha256(_regular_file(Path(row["path"]), "frozen source")) != row["sha256"]:
            raise F2RunnerError(f"frozen source changed during F2 run: {row['path']}")
    if _sha256(_regular_file(Path(checkpoint["path"]), "FastSAM checkpoint")) != checkpoint["sha256"]:
        raise F2RunnerError("FastSAM checkpoint changed during F2 run")
    if _sha256(_regular_file(f0_receipt_path, "sealed F0 receipt")) != f0_receipt["sha256"]:
        raise F2RunnerError("F0 receipt changed during F2 run")
    totals: Counter[str] = Counter()
    for row in rows:
        totals.update({key: int(value) for key, value in row["counts"].items()})
    if production:
        expected = EXPECTED_SHARD_COUNTS[shard_index]
        if any(totals[key] != value for key, value in expected.items()):
            raise F2RunnerError("F2 shard census differs from sealed paper100")
    contracts = {
        "shadow_only": True,
        "birth_enabled": False,
        "ground_truth_access": False,
        "annotation_access": False,
        "prediction_access": False,
        "evaluator_access": False,
        "native_output_mutation": False,
        "semantic_or_clip_access": False,
        "training": False,
        "online_learning": False,
        "f0_exact_replay_required": True,
        "mask_and_cleaned_points_sealed_for_f3": True,
    }
    manifest = {
        **plan,
        "mode": "shadow",
        "complete": True,
        "run_signature_sha256": run_signature,
        "environment": environment,
        "environment_sha256": _canonical_json_sha256(environment),
        "checkpoint": checkpoint,
        "sources_receipt": sources_receipt,
        "contracts": contracts,
        "policy": {
            "f0_schema": f0_core.SCHEMA,
            "f0": dict(f0_core.POLICY),
            "f2_schema": f2_core.SCHEMA,
            "f2": dict(f2_core.POLICY),
            "hypothesis_chain": ["H0", "HL", "HLG"],
            "failure": "fail-open H0 -> HL -> HLG; no source deletion",
            "runtime_gate_scope": "provider + unchanged F0 core + F2 core",
            "evidence_serialization_gate_input": False,
        },
        "resume_rewarm": resume_rewarm,
        "scenes": rows,
        "totals": dict(sorted(totals.items())),
        "runtime": {
            "wall_seconds": float(time.perf_counter() - run_started),
            "cpu_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "gpu_peak_memory_bytes": max(
                (int(row["gpu_peak_memory_bytes"]) for row in rows), default=0
            ),
        },
        "conclusion_guardrail": (
            "F2 is a GT-free shadow geometry replay. It has no AP and cannot "
            "establish an active gain until the separately sealed oracle runs."
        ),
    }
    _atomic_create_json(manifest_path, manifest)
    print(f"Saved: {manifest_path}", flush=True)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paper100 F2 DFU-LGF shadow")
    parser.add_argument("--schedule-root", action="append", type=Path, dest="schedule_roots")
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--scene-list", type=Path, default=DEFAULT_SCENE_LIST)
    parser.add_argument("--f0-receipt", type=Path, default=DEFAULT_F0_RECEIPT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    run_f2(
        schedule_roots=(
            tuple(args.schedule_roots) if args.schedule_roots else DEFAULT_SCHEDULE_ROOTS
        ),
        scene_root=args.scene_root,
        scene_list_path=args.scene_list,
        f0_receipt_path=args.f0_receipt,
        output_root=args.output_root,
        device=args.device,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        resume=args.resume,
        plan_only=args.plan_only,
    )


if __name__ == "__main__":
    main()
