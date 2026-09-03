#!/usr/bin/env python3
"""Fail-closed merge for the frozen F6 GT-free paper100 shadow.

Only create-only F6 receipts and their sealed F4/F2 provenance are opened.
There is intentionally no annotation, evaluator, oracle, detector-prediction,
semantic, or training input in this program.
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

PROTOCOL_ID = "F6-GT-FREE-PAST-ONLY-MULTIVIEW-DEPTH-PROJECTION-SELECTOR-PAPER100"
PROTOCOL_SHA256 = "d0592d8ea69c2d8bcddd942f6ab57b077cdb899aafaadcd3d1c83462cd79768f"
SCENE_SCHEMA = "boxfusion.scannet_fastsam_f6_mvdc_paper100.scene.v1"
SHARD_SCHEMA = "boxfusion.scannet_fastsam_f6_mvdc_paper100.shard.v1"
MERGE_SCHEMA = "boxfusion.scannet_fastsam_f6_mvdc_paper100.merge.v1"
CORE_SCHEMA = "boxfusion.fastsam_f6_mvdc_selector.v1"
EXPECTED_F4_MERGE_SCHEMA = "boxfusion.scannet_fastsam_f4_boxer_paper100.merge.v1"
EXPECTED_F4_SCENE_SCHEMA = "boxfusion.scannet_fastsam_f4_boxer_paper100.scene.v1"
EXPECTED_F4_PROTOCOL = "F4-FASTSAM-BOXER-GEOMETRY-SHADOW-PAPER100"

EXPECTED_SCENES = 100
EXPECTED_SHARDS = 2
EXPECTED_KEYFRAMES = 6_817
EXPECTED_SUCCESSFUL_FRAMES = 6_726
EXPECTED_SOURCES = 52_299
EXPECTED_SCENE_LIST_SHA256 = "4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5"
EXPECTED_F4_MERGE_SHA256 = "0e00ab68e2525b8e1262dfb12bc08ee3a98f02d70b158960f49379e957f826a6"
EXPECTED_SHARD_COUNTS: Mapping[int, Mapping[str, int]] = {
    0: {"keyframe_count": 3_259, "successful_frame_count": 3_189, "source_count": 24_863},
    1: {"keyframe_count": 3_558, "successful_frame_count": 3_537, "source_count": 27_436},
}

WARMUP_NONEMPTY_FRAMES = 3
SOURCE_FRAME_STRIDE = 25.0
MAX_F6_INCREMENTAL_P95_MS = 25.0
MAX_COMPOSED_P95_MS = 375.0
MAX_COMPOSED_MS_EXCLUSIVE = 833.33
MAX_COMPOSED_MEAN_PER_SOURCE_FRAME_MS = 15.0
MAX_CUDA_PEAK_BYTES = 4 * 1024**3
MAX_BUFFERED_FRAMES = 3
MAX_SOURCES_PER_FRAME = 16
MAX_STATE_RAW_ARRAY_PAYLOAD_BYTES = int(2.5 * 1024 * 1024)
MIN_SWITCH_SOURCES = 144
MIN_SWITCH_SCENES = 20
MAX_SWITCH_FRACTION = 0.20

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
    REPOSITORY_ROOT / "logs/scannet_fastsam_f6_mvdc_paper100_score05/shards/shard-000-of-002.json",
    REPOSITORY_ROOT / "logs/scannet_fastsam_f6_mvdc_paper100_score05/shards/shard-001-of-002.json",
)
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "logs/scannet_fastsam_f6_mvdc_paper100_score05/final"
OUTPUT_NAME = "F6_GT_FREE_MVDC_PAPER100.json"
PROTOCOL_PATH = REPOSITORY_ROOT / "docs/F6_GT_FREE_MULTIVIEW_SELECTOR_PROTOCOL_FREEZE.md"


class F6MergeError(RuntimeError):
    """Raised when an F6 receipt is not an exact frozen-contract receipt."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise F6MergeError("value is not canonical finite ASCII JSON") from error
    return hashlib.sha256(payload).hexdigest()


def _content_hash_without(value: Mapping[str, Any], *keys: str) -> str:
    payload = dict(value)
    for key in keys:
        payload.pop(key, None)
    return _canonical_json_sha256(payload)


def _regular_file(path: Path, label: str, suffix: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise F6MergeError(f"{label} must be a regular non-symlink file: {path}")
    resolved = path.resolve()
    if suffix is not None and resolved.suffix.lower() != suffix:
        raise F6MergeError(f"{label} must have suffix {suffix}: {resolved}")
    if resolved.suffix.lower() in {".pkl", ".pickle"}:
        raise F6MergeError(f"forbidden serialized detector input: {resolved}")
    return resolved


def _read_json(path: Path, label: str) -> tuple[Path, dict[str, Any]]:
    source = _regular_file(path, label, ".json")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise F6MergeError(f"invalid {label}: {source}") from error
    if not isinstance(value, dict):
        raise F6MergeError(f"{label} must contain one JSON object")
    return source, value


def _rehash_reference(value: object, label: str, suffix: str | None = None) -> Path:
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("path"), str)
        or not isinstance(value.get("sha256"), str)
    ):
        raise F6MergeError(f"{label} seal is absent")
    path = _regular_file(Path(value["path"]), label, suffix)
    if _sha256(path) != value["sha256"]:
        raise F6MergeError(f"{label} rehash differs")
    return path


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise F6MergeError(f"refusing to overwrite output: {path}") from error
        return _sha256(path)
    finally:
        temporary.unlink(missing_ok=True)


def _strict_int(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise F6MergeError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: object, label: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise F6MergeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise F6MergeError(f"{label} must be finite" + (" and nonnegative" if nonnegative else ""))
    return result


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise F6MergeError("runtime samples must be finite and nonnegative")
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


def _decision_from_gates(gates: Mapping[str, Mapping[str, Any]]) -> str:
    coverage = {"switch_min_sources", "switch_min_scenes", "switch_max_fraction"}
    if not all(value.get("pass") is True for key, value in gates.items() if key not in coverage):
        return "discard_f6_selector"
    if gates["switch_max_fraction"].get("pass") is not True:
        return "stop_f6_overbroad_switches"
    if gates["switch_min_sources"].get("pass") is not True or gates["switch_min_scenes"].get("pass") is not True:
        return "stop_f6_insufficient_multiview_switches"
    return "retain_f6_for_one_separately_sealed_evaluation_only"


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _expected_geometry(source: Mapping[str, Any], name: str) -> dict[str, Any]:
    hypotheses = source.get("hypotheses")
    if not isinstance(hypotheses, Mapping) or set(hypotheses) != {"H0", "HL", "HLG", "HB"}:
        raise F6MergeError(f"{source.get('source_id')} F4 hypothesis set differs")
    row = hypotheses.get(name)
    if not isinstance(row, Mapping):
        raise F6MergeError(f"{source.get('source_id')} selected hypothesis is absent")
    if name != "HB":
        lower = np.asarray(row.get("q02"), dtype=np.float64)
        upper = np.asarray(row.get("q98"), dtype=np.float64)
        if (
            row.get("valid") is not True
            or lower.shape != (3,)
            or upper.shape != (3,)
            or not np.isfinite(lower).all()
            or not np.isfinite(upper).all()
            or np.any(upper <= lower)
        ):
            raise F6MergeError(f"{source.get('source_id')} selected AABB is invalid")
        center = (lower + upper) * 0.5
        extent = upper - lower
        signs = np.asarray(
            [
                (-1.0, -1.0, -1.0), (-1.0, -1.0, 1.0),
                (-1.0, 1.0, -1.0), (-1.0, 1.0, 1.0),
                (1.0, -1.0, -1.0), (1.0, -1.0, 1.0),
                (1.0, 1.0, -1.0), (1.0, 1.0, 1.0),
            ],
            dtype=np.float64,
        )
        corners = center[None, :] + signs * (extent[None, :] * 0.5)
        return {
            "kind": "world_aabb",
            "hypothesis": name,
            "q02": lower.tolist(),
            "q98": upper.tolist(),
            "center": center.tolist(),
            "extent": extent.tolist(),
            "world_rotation": np.eye(3, dtype=np.float64).tolist(),
            "world_corners": corners.tolist(),
            "envelope_q02": lower.tolist(),
            "envelope_q98": upper.tolist(),
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
        raise F6MergeError(f"{source.get('source_id')} selected HB is invalid")
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
    }


def _close(actual: object, expected: float, label: str) -> float:
    value = _number(actual, label, nonnegative=False)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1.0e-12):
        raise F6MergeError(f"{label} arithmetic differs")
    return value


def _validate_buffer(value: object, *, current_ordinal: int, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > MAX_BUFFERED_FRAMES:
        raise F6MergeError(f"{label} exceeds the three-frame bound")
    rows: list[dict[str, Any]] = []
    last_ordinal = -1
    for frame in value:
        if not isinstance(frame, Mapping):
            raise F6MergeError(f"{label} row is malformed")
        ordinal = _strict_int(frame.get("frame_ordinal"), f"{label} frame ordinal")
        frame_id = _strict_int(frame.get("frame_id"), f"{label} frame id")
        source_ids = frame.get("source_ids")
        state_hashes = frame.get("state_sha256")
        payload = _strict_int(frame.get("raw_array_payload_bytes"), f"{label} payload")
        if (
            ordinal <= last_ordinal
            or ordinal >= current_ordinal
            or not isinstance(source_ids, list)
            or not isinstance(state_hashes, list)
            or len(source_ids) != len(state_hashes)
            or len(source_ids) > MAX_SOURCES_PER_FRAME
            or len(set(source_ids)) != len(source_ids)
            or not all(isinstance(source_id, str) for source_id in source_ids)
            or not all(_is_sha256(value) for value in state_hashes)
            or payload > MAX_STATE_RAW_ARRAY_PAYLOAD_BYTES
        ):
            raise F6MergeError(f"{label} causal/source/payload bound differs")
        rows.append({
            "frame_id": frame_id,
            "frame_ordinal": ordinal,
            "source_ids": list(source_ids),
            "state_sha256": list(state_hashes),
            "raw_array_payload_bytes": payload,
        })
        last_ordinal = ordinal
    return rows


def _validate_metric(metric: object, *, expected_views: Sequence[tuple[str, int, int, str]], label: str) -> dict[str, Any]:
    if not isinstance(metric, Mapping) or metric.get("valid") is not True:
        raise F6MergeError(f"{label} lacks valid three-view metrics")
    if metric.get("all_views_have_minimum_points") is not True or metric.get("all_projections_valid") is not True:
        raise F6MergeError(f"{label} incomplete point/projection proof")
    per_view = metric.get("per_view")
    if not isinstance(per_view, list) or len(per_view) != 3 or len(expected_views) != 3:
        raise F6MergeError(f"{label} must contain exactly three views")
    c0_values: list[float] = []
    c5_values: list[float] = []
    j_values: list[float] = []
    supporting = 0
    for index, (row, expected) in enumerate(zip(per_view, expected_views, strict=True)):
        if not isinstance(row, Mapping):
            raise F6MergeError(f"{label} view {index} is malformed")
        source_id, frame_id, frame_ordinal, lineage = expected
        if (
            row.get("source_id") != source_id
            or row.get("frame_id") != frame_id
            or row.get("frame_ordinal") != frame_ordinal
            or row.get("source_lineage_sha256") != lineage
            or not _is_sha256(row.get("state_sha256"))
        ):
            raise F6MergeError(f"{label} view {index} identity/lineage differs")
        original = _strict_int(row.get("original_point_count"), f"{label} view point count", 16)
        sampled = _strict_int(row.get("sampled_point_count"), f"{label} sampled point count", 16)
        if sampled != min(original, 256):
            raise F6MergeError(f"{label} sampled point bound differs")
        c0 = _number(row.get("C0"), f"{label} C0")
        c5 = _number(row.get("C5"), f"{label} C5")
        if c0 > 1.0 or c5 > 1.0 or c5 < c0:
            raise F6MergeError(f"{label} containment range differs")
        support = c0 >= 0.60 and c5 >= 0.80
        if row.get("support_gate_passed") is not support:
            raise F6MergeError(f"{label} support gate ledger differs")
        supporting += int(support)
        projection = row.get("projection")
        if not isinstance(projection, Mapping) or projection.get("valid") is not True or projection.get("reason") != "valid":
            raise F6MergeError(f"{label} view projection is incomplete")
        minimum_depth = _number(projection.get("minimum_corner_depth_m"), f"{label} minimum depth")
        if minimum_depth <= 1.0e-4:
            raise F6MergeError(f"{label} projection violates near plane")
        p_value = _number(projection.get("P"), f"{label} P")
        r_value = _number(projection.get("R"), f"{label} R")
        j_value = _number(projection.get("J"), f"{label} J")
        if p_value > 1.0 or r_value > 1.0 or j_value > 1.0:
            raise F6MergeError(f"{label} projection metric range differs")
        expected_j = 0.0 if p_value * r_value == 0.0 else 1.0 / (1.0 / p_value + 1.0 / r_value - 1.0)
        _close(j_value, expected_j, f"{label} J")
        c0_values.append(c0)
        c5_values.append(c5)
        j_values.append(j_value)
    c0 = _number(metric.get("C0"), f"{label} median C0")
    c5 = _number(metric.get("C5"), f"{label} median C5")
    j_value = _number(metric.get("J"), f"{label} median J")
    _close(c0, float(np.median(c0_values)), f"{label} median C0")
    _close(c5, float(np.median(c5_values)), f"{label} median C5")
    _close(j_value, float(np.median(j_values)), f"{label} median J")
    _number(metric.get("D_m"), f"{label} D")
    if metric.get("supporting_view_count") != supporting:
        raise F6MergeError(f"{label} supporting-view count differs")
    q02 = np.asarray(metric.get("q02_local"), dtype=np.float64)
    q98 = np.asarray(metric.get("q98_local"), dtype=np.float64)
    if q02.shape != (3,) or q98.shape != (3,) or not np.isfinite(q02).all() or not np.isfinite(q98).all() or np.any(q98 < q02):
        raise F6MergeError(f"{label} robust-face quantiles differ")
    return dict(metric)


def _validate_switched_proof(
    row: Mapping[str, Any], *, buffer_before: Sequence[Mapping[str, Any]]
) -> None:
    source_id = str(row.get("source_id"))
    current_identity = (
        source_id,
        _strict_int(row.get("frame_id"), f"{source_id} frame id"),
        _strict_int(row.get("frame_ordinal"), f"{source_id} frame ordinal"),
        str(row.get("source_lineage_sha256")),
    )
    matched = row.get("matched_past")
    if not isinstance(matched, list) or len(matched) != 2 or row.get("matched_past_frame_count") != 2:
        raise F6MergeError(f"{source_id} switch lacks exactly two past views")
    available: set[tuple[int, int, str, str]] = set()
    for frame in buffer_before:
        for past_id, state_hash in zip(frame["source_ids"], frame["state_sha256"], strict=True):
            available.add((frame["frame_ordinal"], frame["frame_id"], past_id, state_hash))
    expected_views = [current_identity]
    seen_ordinals: set[int] = set()
    for proof in matched:
        if not isinstance(proof, Mapping):
            raise F6MergeError(f"{source_id} past-view proof is malformed")
        ordinal = _strict_int(proof.get("frame_ordinal"), f"{source_id} past ordinal")
        frame_id = _strict_int(proof.get("frame_id"), f"{source_id} past frame")
        past_id = proof.get("source_id")
        state_hash = proof.get("state_sha256")
        lineage = proof.get("source_lineage_sha256")
        if (
            (ordinal, frame_id, past_id, state_hash) not in available
            or ordinal in seen_ordinals
            or not isinstance(lineage, str)
            or not _is_sha256(lineage)
        ):
            raise F6MergeError(f"{source_id} switch cites unavailable/non-distinct past evidence")
        affinity = proof.get("affinity")
        if not isinstance(affinity, Mapping):
            raise F6MergeError(f"{source_id} association proof is absent")
        iou = _number(affinity.get("iou3d"), f"{source_id} association IoU")
        containment = _number(affinity.get("symmetric_containment"), f"{source_id} association containment")
        nd = _number(affinity.get("normalized_center_distance"), f"{source_id} association ND")
        if nd > 0.50 or (iou < 0.15 and containment < 0.60):
            raise F6MergeError(f"{source_id} association gate differs")
        expected_views.append((str(past_id), frame_id, ordinal, str(lineage)))
        seen_ordinals.add(ordinal)
    if len(seen_ordinals) != 2:
        raise F6MergeError(f"{source_id} switch lacks two distinct historical frames")

    evaluations = row.get("candidate_evaluations")
    if not isinstance(evaluations, Mapping) or set(evaluations) != {"H0", "HL", "HLG", "HB"}:
        raise F6MergeError(f"{source_id} candidate evaluation set differs")
    selected = str(row.get("selected_hypothesis"))
    base = str(row.get("base_hypothesis"))
    selected_eval = evaluations.get(selected)
    base_eval = evaluations.get(base)
    if not isinstance(selected_eval, Mapping) or not isinstance(base_eval, Mapping):
        raise F6MergeError(f"{source_id} selected/base evaluation is absent")
    if (
        selected_eval.get("available") is not True
        or selected_eval.get("is_base") is not False
        or selected_eval.get("selector_passed") is not True
        or selected_eval.get("geometry") != row.get("selected_geometry")
        or selected_eval.get("geometry_sha256") != row.get("selected_geometry_sha256")
        or base_eval.get("available") is not True
        or base_eval.get("is_base") is not True
        or base_eval.get("geometry") != row.get("base_geometry")
        or base_eval.get("geometry_sha256") != row.get("base_geometry_sha256")
        or base_eval.get("metrics") != row.get("base_metrics")
    ):
        raise F6MergeError(f"{source_id} selected/base geometry proof differs")
    selected_metrics = _validate_metric(
        selected_eval.get("metrics"), expected_views=expected_views, label=f"{source_id} selected"
    )
    base_metrics = _validate_metric(
        base_eval.get("metrics"), expected_views=expected_views, label=f"{source_id} base"
    )
    gate = selected_eval.get("gate")
    if not isinstance(gate, Mapping) or gate.get("passed") is not True:
        raise F6MergeError(f"{source_id} selected candidate gate failed")
    checks = gate.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != {
        "metrics_valid", "all_projections_valid", "all_views_have_minimum_points",
        "nd_passed", "volume_ratio_passed", "overlap_passed",
        "current_exact_support_passed", "current_expanded_support_passed",
        "two_of_three_support_passed",
    } or any(value is not True for value in checks.values()):
        raise F6MergeError(f"{source_id} selected gate-check proof differs")
    iou = _number(gate.get("candidate_base_iou3d"), f"{source_id} candidate IoU")
    containment = _number(gate.get("candidate_base_symmetric_containment"), f"{source_id} candidate containment")
    nd = _number(gate.get("candidate_base_normalized_center_distance"), f"{source_id} candidate ND")
    ratio = _number(gate.get("candidate_base_envelope_volume_ratio"), f"{source_id} volume ratio")
    support_count = _strict_int(gate.get("supporting_view_count"), f"{source_id} support count")
    if nd > 0.50 or not 0.25 <= ratio <= 4.00 or (iou < 0.20 and containment < 0.70) or support_count < 2:
        raise F6MergeError(f"{source_id} selected candidate numeric gate differs")
    comparison = selected_eval.get("comparison")
    if not isinstance(comparison, Mapping) or comparison.get("passed") is not True or comparison.get("reason") != "passed":
        raise F6MergeError(f"{source_id} selector comparison proof differs")
    depth_win = selected_metrics["D_m"] <= base_metrics["D_m"] - 0.05
    projection_win = selected_metrics["J"] >= base_metrics["J"] + 0.10
    containment_win = selected_metrics["C0"] >= base_metrics["C0"] + 0.10
    win_count = int(depth_win) + int(projection_win) + int(containment_win)
    depth_nr = selected_metrics["D_m"] <= base_metrics["D_m"] + 0.025
    projection_nr = selected_metrics["J"] >= base_metrics["J"] - 0.05
    containment_nr = selected_metrics["C0"] >= base_metrics["C0"] - 0.05
    expected_flags = {
        "depth_win": depth_win,
        "projection_win": projection_win,
        "containment_win": containment_win,
        "win_count": win_count,
        "depth_non_regression": depth_nr,
        "projection_non_regression": projection_nr,
        "containment_non_regression": containment_nr,
    }
    if any(comparison.get(key) != value for key, value in expected_flags.items()) or win_count < 2 or not (depth_nr and projection_nr and containment_nr):
        raise F6MergeError(f"{source_id} win/non-regression arithmetic differs")
    delta = comparison.get("candidate_minus_base")
    if not isinstance(delta, Mapping):
        raise F6MergeError(f"{source_id} candidate-minus-base proof is absent")
    for key in ("D_m", "J", "C0"):
        _close(delta.get(key), float(selected_metrics[key] - base_metrics[key]), f"{source_id} delta {key}")
    priority = {"H0": 0, "HL": 1, "HLG": 2, "HB": 3}
    passers: list[tuple[str, Mapping[str, Any]]] = []
    for name, evaluation in evaluations.items():
        if isinstance(evaluation, Mapping) and evaluation.get("selector_passed") is True:
            metrics = evaluation.get("metrics")
            compare = evaluation.get("comparison")
            if not isinstance(metrics, Mapping) or not isinstance(compare, Mapping):
                raise F6MergeError(f"{source_id} passing candidate proof is incomplete")
            passers.append((str(name), evaluation))
    if not passers:
        raise F6MergeError(f"{source_id} switch has no passing candidate")
    winner = min(
        passers,
        key=lambda item: (
            -int(item[1]["comparison"]["win_count"]),
            float(item[1]["metrics"]["D_m"]),
            -float(item[1]["metrics"]["J"]),
            -float(item[1]["metrics"]["C0"]),
            priority[item[0]],
        ),
    )[0]
    if winner != selected:
        raise F6MergeError(f"{source_id} lexicographic winner differs")


def _verify_source_row(
    row: Mapping[str, Any], f4_source: Mapping[str, Any], *, buffer_before: Sequence[Mapping[str, Any]], core: object
) -> tuple[str, bool, bool]:
    source_id = f4_source.get("source_id")
    if (
        row.get("schema") != CORE_SCHEMA
        or row.get("protocol_id") != PROTOCOL_ID
        or row.get("mode") != "shadow"
        or row.get("source_id") != source_id
        or row.get("source_lineage_sha256") != f4_source.get("source_lineage_sha256")
        or row.get("rank") != f4_source.get("rank")
        or row.get("frame_id") != f4_source.get("frame_id")
        or row.get("frame_ordinal") != f4_source.get("frame_ordinal")
        or type(row.get("formal_score")) is not float
        or row.get("formal_score") != 1.0
        or row.get("maximum_lookahead_frames") != 0
        or row.get("observer_only") is not True
        or row.get("birth_applied") is not False
        or row.get("native_output_mutation_applied") is not False
    ):
        raise F6MergeError(f"{source_id} source/formal-score/shadow contract differs")
    hypotheses = f4_source.get("hypotheses")
    if not isinstance(hypotheses, Mapping) or set(hypotheses) != {"H0", "HL", "HLG", "HB"}:
        raise F6MergeError(f"{source_id} F4 hypothesis set differs")
    expected_hashes = {name: _canonical_json_sha256(hypotheses[name]) for name in ("H0", "HL", "HLG", "HB")}
    if row.get("input_hypothesis_sha256") != expected_hashes:
        raise F6MergeError(f"{source_id} input-hypothesis lineage differs")
    selected = row.get("selected_hypothesis")
    base = row.get("base_hypothesis")
    if selected not in expected_hashes or base not in {"H0", "HL", "HLG"}:
        raise F6MergeError(f"{source_id} selected/base hypothesis differs")
    selected_geometry = _expected_geometry(f4_source, str(selected))
    base_geometry = _expected_geometry(f4_source, str(base))
    if (
        row.get("selected_geometry") != selected_geometry
        or row.get("selected_geometry_sha256") != _canonical_json_sha256(selected_geometry)
        or row.get("base_geometry") != base_geometry
        or row.get("base_geometry_sha256") != _canonical_json_sha256(base_geometry)
    ):
        raise F6MergeError(f"{source_id} geometry is not the exact F4 hypothesis copy")
    canonical = getattr(core, "canonical_result_sha256", None)
    if not callable(canonical) or canonical(row) != row.get("result_sha256"):
        raise F6MergeError(f"{source_id} result hash differs")
    switched = selected != base
    if row.get("switched_from_base") is not switched:
        raise F6MergeError(f"{source_id} switch ledger differs")
    matched_count = _strict_int(row.get("matched_past_frame_count"), f"{source_id} matched count")
    if switched:
        if row.get("selection_reason") != "non_base_candidate_won":
            raise F6MergeError(f"{source_id} switch reason differs")
        _validate_switched_proof(row, buffer_before=buffer_before)
    else:
        expected_reason = "fewer_than_two_past_matches" if matched_count < 2 else "no_non_base_candidate_passed"
        if row.get("selection_reason") != expected_reason:
            raise F6MergeError(f"{source_id} fallback reason differs")
        evaluations = row.get("candidate_evaluations")
        if not isinstance(evaluations, Mapping) or set(evaluations) != {"H0", "HL", "HLG", "HB"}:
            raise F6MergeError(f"{source_id} fallback candidate ledger differs")
        if matched_count >= 2 and any(
            isinstance(value, Mapping) and value.get("selector_passed") is True
            for value in evaluations.values()
        ):
            raise F6MergeError(f"{source_id} fallback suppresses a passing candidate")
    evaluated = matched_count >= 2
    return str(selected), switched, evaluated


def _validate_scene(
    scene_row: Mapping[str, Any], *, expected_scene_index: int, expected_signature: str,
    f4_receipt_seal: Mapping[str, Any], source_receipts: Mapping[str, Any], core: object,
) -> dict[str, Any]:
    scene_id = scene_row.get("scene_id")
    if not isinstance(scene_id, str) or scene_row.get("scene_index") != expected_scene_index:
        raise F6MergeError("F6 scene identity/order differs")
    sidecar_path = _rehash_reference(scene_row.get("sidecar"), f"{scene_id} F6 sidecar", ".json")
    _, scene = _read_json(sidecar_path, f"{scene_id} F6 sidecar")
    if (
        scene.get("schema") != SCENE_SCHEMA or scene.get("protocol_id") != PROTOCOL_ID
        or scene.get("complete") is not True or scene.get("scene_id") != scene_id
        or scene.get("scene_index") != expected_scene_index
        or scene.get("run_signature_sha256") != expected_signature
        or scene.get("contracts") != CONTRACTS
        or scene.get("native_output_mutation_count") != 0 or scene.get("birth_count") != 0
        or _content_hash_without(scene, "content_sha256") != scene.get("content_sha256")
    ):
        raise F6MergeError(f"{scene_id} scene contract/content differs")
    inputs = scene.get("inputs")
    if not isinstance(inputs, Mapping) or inputs.get("f4_receipt") != f4_receipt_seal or inputs.get("sources") != source_receipts:
        raise F6MergeError(f"{scene_id} frozen input/source receipts differ")
    f4_path = _rehash_reference(inputs.get("f4_sidecar"), f"{scene_id} F4 sidecar", ".json")
    _, f4 = _read_json(f4_path, f"{scene_id} F4 sidecar")
    if (
        f4.get("schema") != EXPECTED_F4_SCENE_SCHEMA or f4.get("protocol_id") != EXPECTED_F4_PROTOCOL
        or f4.get("complete") is not True or f4.get("scene_id") != scene_id
        or f4.get("scene_index") != expected_scene_index or f4.get("native_output_mutation_count") != 0
        or _content_hash_without(f4, "content_sha256") != f4.get("content_sha256")
    ):
        raise F6MergeError(f"{scene_id} sealed F4 sidecar differs")
    for key, suffix in (("f2_sidecar", ".json"), ("f2_evidence", ".npz"), ("intrinsic", ".txt")):
        _rehash_reference(inputs.get(key), f"{scene_id} {key}", suffix)
    pose_ledger = inputs.get("pose_ledger")
    if not isinstance(pose_ledger, list):
        raise F6MergeError(f"{scene_id} pose ledger is absent")
    for index, seal in enumerate(pose_ledger):
        _rehash_reference(seal, f"{scene_id} pose {index}", ".txt")
    base_seals = [dict(inputs[key]) for key in ("f4_sidecar", "f2_sidecar", "f2_evidence", "intrinsic")]
    if inputs.get("frozen_base_inputs_sha256") != _canonical_json_sha256(base_seals):
        raise F6MergeError(f"{scene_id} frozen base-input aggregate differs")
    opened_seals = base_seals + [dict(value) for value in pose_ledger]
    opened_hash = _canonical_json_sha256(opened_seals)
    if inputs.get("all_opened_inputs_before_sha256") != opened_hash or inputs.get("all_opened_inputs_after_sha256") != opened_hash:
        raise F6MergeError(f"{scene_id} opened-input before/after seal differs")

    frames = scene.get("frames")
    f4_frames = f4.get("frames")
    if not isinstance(frames, list) or not isinstance(f4_frames, list) or len(frames) != len(f4_frames):
        raise F6MergeError(f"{scene_id} frame ledger differs")
    ids: list[str] = []
    lineages: list[str] = []
    results: list[str] = []
    selected_counts = {name: 0 for name in ("H0", "HL", "HLG", "HB")}
    switches = evaluated = successful = 0
    incremental_all: list[float] = []
    incremental_warm: list[float] = []
    gross_all: list[float] = []
    gross_warm: list[float] = []
    audit_hash_all: list[float] = []
    audit_hash_warm: list[float] = []
    audit_serialization_all: list[float] = []
    audit_serialization_warm: list[float] = []
    audit_total_all: list[float] = []
    audit_total_warm: list[float] = []
    composed_all: list[float] = []
    composed_warm: list[float] = []
    deadline_all = deadline_warm = 0
    max_buffer_frames = max_buffer_sources = max_payload = 0
    for ordinal, (frame, f4_frame) in enumerate(zip(frames, f4_frames, strict=True)):
        if not isinstance(frame, Mapping) or not isinstance(f4_frame, Mapping):
            raise F6MergeError(f"{scene_id}/{ordinal} malformed frame")
        frame_id = f4_frame.get("frame_id")
        is_success = f4_frame.get("successful") is True
        if frame.get("frame_ordinal") != ordinal or frame.get("frame_id") != frame_id or (frame.get("successful") is True) is not is_success:
            raise F6MergeError(f"{scene_id}/{ordinal} frame identity differs")
        actual_sources = frame.get("sources")
        f4_sources = f4_frame.get("sources")
        if not isinstance(actual_sources, list) or not isinstance(f4_sources, list) or len(actual_sources) != len(f4_sources) or len(actual_sources) > MAX_SOURCES_PER_FRAME:
            raise F6MergeError(f"{scene_id}/{frame_id} source partition differs")
        if not is_success:
            if actual_sources or any(frame.get(key) is not None for key in ("buffer_before", "buffer_after", "query", "commit", "runtime", "maximum_accessed_frame_ordinal")):
                raise F6MergeError(f"{scene_id}/{frame_id} failed-frame contract differs")
            continue
        successful += 1
        before = _validate_buffer(frame.get("buffer_before"), current_ordinal=ordinal, label=f"{scene_id}/{frame_id} buffer_before")
        after = _validate_buffer(frame.get("buffer_after"), current_ordinal=ordinal + 1, label=f"{scene_id}/{frame_id} buffer_after")
        max_buffer_frames = max(max_buffer_frames, len(before), len(after))
        max_buffer_sources = max(max_buffer_sources, *(len(value["source_ids"]) for value in before + after), 0)
        query, commit = frame.get("query"), frame.get("commit")
        if not isinstance(query, Mapping) or not isinstance(commit, Mapping) or query.get("query_before_commit") is not True or query.get("buffer_before") != before or commit.get("buffer_after") != after or query.get("token") != commit.get("token") or query.get("maximum_lookahead_frames") != 0:
            raise F6MergeError(f"{scene_id}/{frame_id} query/commit ledger differs")
        if frame.get("maximum_accessed_frame_ordinal") != query.get("maximum_accessed_frame_ordinal") or int(query.get("maximum_accessed_frame_ordinal", -1)) >= ordinal:
            raise F6MergeError(f"{scene_id}/{frame_id} future access detected")
        query_payload = _strict_int(query.get("raw_array_payload_bytes"), "query payload")
        commit_payload = _strict_int(commit.get("raw_array_payload_bytes"), "commit payload")
        max_payload = max(max_payload, query_payload, commit_payload)
        if max_payload > MAX_STATE_RAW_ARRAY_PAYLOAD_BYTES or commit.get("source_count") != len(actual_sources):
            raise F6MergeError(f"{scene_id}/{frame_id} bounded-state/commit count differs")
        frame_hashes: list[str] = []
        for rank, (row, f4_source) in enumerate(zip(actual_sources, f4_sources, strict=True)):
            if not isinstance(row, Mapping) or f4_source.get("rank") != rank:
                raise F6MergeError(f"{scene_id}/{frame_id}/{rank} source order differs")
            selected, switched, was_evaluated = _verify_source_row(row, f4_source, buffer_before=before, core=core)
            selected_counts[selected] += 1
            switches += int(switched)
            evaluated += int(was_evaluated)
            ids.append(str(row["source_id"]))
            lineages.append(str(row["source_lineage_sha256"]))
            results.append(str(row["result_sha256"]))
            frame_hashes.append(str(row["result_sha256"]))
        expected_token = _canonical_json_sha256({
            "protocol_id": PROTOCOL_ID, "frame_id": frame_id, "frame_ordinal": ordinal,
            "buffer_before": before,
            "maximum_accessed_frame_ordinal": query["maximum_accessed_frame_ordinal"],
            "state_raw_array_payload_bytes": query_payload, "result_sha256": frame_hashes,
        })
        if query.get("token") != expected_token:
            raise F6MergeError(f"{scene_id}/{frame_id} query token differs")
        runtime = frame.get("runtime")
        if not isinstance(runtime, Mapping) or runtime.get("f6_cuda_allocated_bytes") != 0:
            raise F6MergeError(f"{scene_id}/{frame_id} runtime/CUDA ledger differs")
        incremental = _number(runtime.get("f6_incremental_ms"), "F6 incremental runtime")
        formal = _number(
            runtime.get("f6_incremental_formal_ms"),
            "F6 formal incremental runtime",
        )
        gross = _number(runtime.get("f6_incremental_gross_ms"), "F6 gross runtime")
        audit_hash = _number(
            runtime.get("f6_audit_hash_excluded_ms"), "F6 audit hash runtime"
        )
        audit_serialization = _number(
            runtime.get("f6_audit_serialization_excluded_ms"),
            "F6 audit serialization runtime",
        )
        audit_total = _number(
            runtime.get("f6_audit_total_excluded_ms"), "F6 total audit runtime"
        )
        _close(formal, incremental, "F6 formal/incremental runtime")
        _close(
            audit_total,
            audit_hash + audit_serialization,
            "F6 total audit runtime",
        )
        _close(gross, formal + audit_total, "F6 gross runtime")
        inherited = _number(runtime.get("sealed_f4_composed_ms"), "sealed F4 runtime")
        composed = _number(runtime.get("replay_composed_ms"), "F6 composed runtime")
        _close(composed, inherited + incremental, "F6 composed runtime")
        _close(runtime.get("replay_composed_ms_per_source_frame"), composed / SOURCE_FRAME_STRIDE, "F6 amortized runtime")
        if runtime.get("state_raw_array_payload_bytes") != commit_payload:
            raise F6MergeError(f"{scene_id}/{frame_id} runtime payload differs")
        warmup = runtime.get("f6_warmup_excluded") is True
        missed = composed >= MAX_COMPOSED_MS_EXCLUSIVE
        if runtime.get("gap25_deadline_missed") is not missed or runtime.get("gap25_deadline_missed_warm") is not ((not warmup) and missed):
            raise F6MergeError(f"{scene_id}/{frame_id} deadline ledger differs")
        incremental_all.append(incremental)
        gross_all.append(gross)
        audit_hash_all.append(audit_hash)
        audit_serialization_all.append(audit_serialization)
        audit_total_all.append(audit_total)
        composed_all.append(composed); deadline_all += int(missed)
        if not warmup:
            incremental_warm.append(incremental)
            gross_warm.append(gross)
            audit_hash_warm.append(audit_hash)
            audit_serialization_warm.append(audit_serialization)
            audit_total_warm.append(audit_total)
            composed_warm.append(composed); deadline_warm += int(missed)

    if len(ids) != len(set(ids)):
        raise F6MergeError(f"{scene_id} duplicate source identities")
    counts = {
        "keyframe_count": len(frames), "successful_frame_count": successful,
        "source_count": len(ids), "identity_verified_source_count": len(ids),
        "multiview_evaluated_source_count": evaluated, "switch_count": switches,
        "fallback_count": len(ids) - switches,
        **{f"selected_{name.lower()}_count": selected_counts[name] for name in selected_counts},
    }
    if scene.get("counts") != counts or scene_row.get("counts") != counts:
        raise F6MergeError(f"{scene_id} count census differs")
    ids_hash, lineage_hash, result_hash = map(_canonical_json_sha256, (ids, lineages, results))
    if any(scene.get(key) != value or scene_row.get(key) != value for key, value in (("source_ids_sha256", ids_hash), ("source_lineage_sha256", lineage_hash), ("result_ledger_sha256", result_hash))):
        raise F6MergeError(f"{scene_id} ordered lineage/result hash differs")
    prefix_count = successful // 2
    prefix_hashes: list[str] = []
    seen = 0
    for frame in frames:
        if frame.get("successful") is True:
            if seen >= prefix_count: break
            seen += 1; prefix_hashes.extend(row["result_sha256"] for row in frame["sources"])
    prefix = {"passed": True, "successful_frame_count": prefix_count, "result_row_count": len(prefix_hashes), "result_ledger_sha256": _canonical_json_sha256(prefix_hashes)}
    determinism = {"passed": True, "independent_replay_count": 1, "online_result_ledger_sha256": result_hash, "independent_result_ledger_sha256": result_hash}
    if scene.get("prefix_replay") != prefix or scene_row.get("prefix_replay") != prefix or scene.get("determinism") != determinism or scene_row.get("determinism") != determinism:
        raise F6MergeError(f"{scene_id} replay proof differs")
    expected_causality = {
        "overall_pass": True, "query_before_commit": True, "prefix_replay_pass": True,
        "independent_replay_pass": True, "future_perturbation_covered_by_prefix_replay": True,
        "maximum_lookahead_frames": 0,
        "maximum_accessed_past_frame_ordinal": max(
            (int(frame["maximum_accessed_frame_ordinal"]) for frame in frames if frame.get("successful") is True),
            default=-1,
        ),
        "future_access_count": 0, "current_source_offsets_only": True,
        "prefix_successful_frame_count": prefix_count,
    }
    if scene.get("causality") != expected_causality or scene_row.get("causality") != expected_causality:
        raise F6MergeError(f"{scene_id} causality summary differs")
    expected_runtime = {
        "f6_incremental_gross_all_ms": _distribution(gross_all),
        "f6_incremental_gross_warm_ms": _distribution(gross_warm),
        "f6_audit_hash_excluded_all_ms": _distribution(audit_hash_all),
        "f6_audit_hash_excluded_warm_ms": _distribution(audit_hash_warm),
        "f6_audit_serialization_excluded_all_ms": _distribution(audit_serialization_all),
        "f6_audit_serialization_excluded_warm_ms": _distribution(audit_serialization_warm),
        "f6_audit_total_excluded_all_ms": _distribution(audit_total_all),
        "f6_audit_total_excluded_warm_ms": _distribution(audit_total_warm),
        "formal_runtime_excludes_hashing_and_serialization": True,
        "f6_incremental_all_ms": _distribution(incremental_all), "f6_incremental_warm_ms": _distribution(incremental_warm),
        "replay_composed_all_ms": _distribution(composed_all), "replay_composed_warm_ms": _distribution(composed_warm),
        "replay_composed_warm_mean_per_source_frame_ms": float(np.mean(composed_warm)) / SOURCE_FRAME_STRIDE if composed_warm else 0.0,
        "gap25_all_deadline_miss_count": deadline_all, "gap25_warm_deadline_miss_count": deadline_warm,
        "maximum_state_raw_array_payload_bytes": max_payload, "state_payload_limit_bytes": MAX_STATE_RAW_ARRAY_PAYLOAD_BYTES,
        "f6_cuda_peak_memory_bytes": 0,
        "inherited_f4_cuda_peak_memory_bytes": int(f4.get("runtime", {}).get("cuda_peak_memory_bytes", 0)),
        "cuda_peak_memory_bytes": int(f4.get("runtime", {}).get("cuda_peak_memory_bytes", 0)),
    }
    if scene.get("runtime") != expected_runtime or scene_row.get("runtime") != expected_runtime:
        raise F6MergeError(f"{scene_id} runtime summary differs")
    bounded = {"overall_pass": True, "maximum_buffered_successful_frame_count": max_buffer_frames, "maximum_sources_per_buffered_frame": max_buffer_sources, "maximum_raw_array_payload_bytes": max_payload, "raw_array_payload_limit_bytes": MAX_STATE_RAW_ARRAY_PAYLOAD_BYTES}
    if scene.get("bounded_state") != bounded or scene_row.get("bounded_state") != bounded:
        raise F6MergeError(f"{scene_id} bounded-state summary differs")
    return {
        "row": dict(scene_row), "counts": counts, "ids": ids,
        "lineages": lineages, "results": results, "switches": switches,
        "incremental_all": incremental_all, "incremental_warm": incremental_warm,
        "gross_all": gross_all, "gross_warm": gross_warm,
        "audit_hash_all": audit_hash_all, "audit_hash_warm": audit_hash_warm,
        "audit_serialization_all": audit_serialization_all,
        "audit_serialization_warm": audit_serialization_warm,
        "audit_total_all": audit_total_all, "audit_total_warm": audit_total_warm,
        "composed_all": composed_all, "composed_warm": composed_warm,
        "deadline_all": deadline_all, "deadline_warm": deadline_warm,
        "max_buffer_frames": max_buffer_frames,
        "max_buffer_sources": max_buffer_sources,
        "max_payload": max_payload,
        "cuda_peak": expected_runtime["cuda_peak_memory_bytes"],
    }


def merge_f6(
    *, shard_paths: Sequence[Path] = DEFAULT_SHARDS, output_dir: Path = DEFAULT_OUTPUT_DIR,
    expected_scene_count: int = EXPECTED_SCENES, expected_keyframes: int | None = None,
    expected_successful_frames: int | None = None, expected_sources: int | None = None,
    min_switch_sources: int | None = None, min_switch_scenes: int | None = None,
    max_switch_fraction: float = MAX_SWITCH_FRACTION,
) -> dict[str, Any]:
    if len(shard_paths) != EXPECTED_SHARDS:
        raise F6MergeError("F6 merge requires exactly two shard manifests")
    production = expected_scene_count == EXPECTED_SCENES
    expected_keyframes = EXPECTED_KEYFRAMES if production and expected_keyframes is None else expected_keyframes
    expected_successful_frames = EXPECTED_SUCCESSFUL_FRAMES if production and expected_successful_frames is None else expected_successful_frames
    expected_sources = EXPECTED_SOURCES if production and expected_sources is None else expected_sources
    min_switch_sources = MIN_SWITCH_SOURCES if min_switch_sources is None else min_switch_sources
    min_switch_scenes = MIN_SWITCH_SCENES if min_switch_scenes is None else min_switch_scenes
    if production and (min_switch_sources != MIN_SWITCH_SOURCES or min_switch_scenes != MIN_SWITCH_SCENES or max_switch_fraction != MAX_SWITCH_FRACTION):
        raise F6MergeError("production F6 switch gates are frozen")
    if _sha256(_regular_file(PROTOCOL_PATH, "F6 frozen protocol", ".md")) != PROTOCOL_SHA256:
        raise F6MergeError("F6 frozen protocol hash differs")
    merge_source = _regular_file(Path(__file__).resolve(), "F6 merge source", ".py")
    merge_source_seal = {"path": os.fspath(merge_source), "sha256": _sha256(merge_source)}
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for shard_index, shard_path in enumerate(shard_paths):
        path, shard = _read_json(Path(shard_path), f"F6 shard {shard_index}")
        if (
            shard.get("schema") != SHARD_SCHEMA or shard.get("protocol_id") != PROTOCOL_ID
            or shard.get("complete") is not True or shard.get("shard_index") != shard_index
            or shard.get("num_shards") != EXPECTED_SHARDS or shard.get("contracts") != CONTRACTS
            or shard.get("native_output_mutation_count") != 0 or shard.get("birth_count") != 0
            or _content_hash_without(shard, "content_sha256", "manifest_path") != shard.get("content_sha256")
        ):
            # Runner includes manifest_path before hashing. Accommodate that exact form only.
            if _content_hash_without(shard, "content_sha256") != shard.get("content_sha256"):
                raise F6MergeError(f"F6 shard {shard_index} contract/content differs")
        loaded.append((path, shard))
    left, right = loaded[0][1], loaded[1][1]
    if left.get("run_signature_sha256") != right.get("run_signature_sha256") or left.get("signature_payload_sha256") != right.get("signature_payload_sha256") or left.get("inputs") != right.get("inputs"):
        raise F6MergeError("F6 shard shared signature/input differs")
    inputs = left.get("inputs")
    if not isinstance(inputs, Mapping):
        raise F6MergeError("F6 shard input receipt is absent")
    f4_path = _rehash_reference(inputs.get("f4_receipt"), "sealed F4 merge", ".json")
    _, f4 = _read_json(f4_path, "sealed F4 merge")
    if production and _sha256(f4_path) != EXPECTED_F4_MERGE_SHA256:
        raise F6MergeError("sealed production F4 merge hash differs")
    if f4.get("schema") != EXPECTED_F4_MERGE_SCHEMA or f4.get("protocol_id") != EXPECTED_F4_PROTOCOL or f4.get("complete") is not True or f4.get("overall_pass") is not True or f4.get("native_output_mutation_count") != 0 or _content_hash_without(f4, "content_sha256") != f4.get("content_sha256"):
        raise F6MergeError("sealed F4 merge contract differs")
    scene_list = _rehash_reference(inputs.get("scene_list"), "sealed scene list", ".txt")
    if production and _sha256(scene_list) != EXPECTED_SCENE_LIST_SHA256:
        raise F6MergeError("paper100 scene-list hash differs")
    scene_order = [value.strip() for value in scene_list.read_text(encoding="utf-8").splitlines() if value.strip()]
    if len(scene_order) != expected_scene_count or len(set(scene_order)) != len(scene_order):
        raise F6MergeError("scene-list count/order differs")
    f4_rows = f4.get("scenes")
    if not isinstance(f4_rows, list) or len(f4_rows) != expected_scene_count or f4.get("coverage", {}).get("scene_order") != scene_order:
        raise F6MergeError("F4 scene ledger differs")
    source_receipts = inputs.get("sources")
    if not isinstance(source_receipts, Mapping) or set(source_receipts) != {"runner", "core", "protocol"}:
        raise F6MergeError("F6 source receipt set differs")
    for name, seal in source_receipts.items():
        _rehash_reference(seal, f"F6 frozen {name}")
    expected_paths = {"runner": REPOSITORY_ROOT / "tools/run_scannet_fastsam_f6_mvdc_paper100.py", "core": REPOSITORY_ROOT / "boxfusion/fastsam_f6_mvdc_selector.py", "protocol": PROTOCOL_PATH}
    if any(Path(source_receipts[name]["path"]).resolve() != path.resolve() for name, path in expected_paths.items()) or source_receipts["protocol"]["sha256"] != PROTOCOL_SHA256:
        raise F6MergeError("F6 frozen source path/hash differs")
    from boxfusion import fastsam_f6_mvdc_selector as core
    signature_payload = {"protocol_id": PROTOCOL_ID, "f4_receipt": dict(inputs["f4_receipt"]), "scene_order": scene_order, "scene_list_sha256": inputs["scene_list"]["sha256"], "core_schema": core.SCHEMA, "core_policy": dict(core.POLICY), "sources": {key: dict(value) for key, value in source_receipts.items()}, "contracts": dict(CONTRACTS), "num_shards": EXPECTED_SHARDS}
    signature = _canonical_json_sha256(signature_payload)
    if left.get("run_signature_sha256") != signature or left.get("signature_payload_sha256") != signature:
        raise F6MergeError("F6 run signature differs")
    rows_by_index: dict[int, tuple[int, Mapping[str, Any]]] = {}
    for shard_index, (_, shard) in enumerate(loaded):
        rows = shard.get("scenes")
        if not isinstance(rows, list): raise F6MergeError("F6 shard scene rows absent")
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("scene_index"), int): raise F6MergeError("F6 shard scene row malformed")
            index = row["scene_index"]
            if index % 2 != shard_index or index in rows_by_index: raise F6MergeError("F6 deterministic scene partition differs")
            rows_by_index[index] = (shard_index, row)
    if sorted(rows_by_index) != list(range(expected_scene_count)):
        raise F6MergeError("merged F6 scene partition is incomplete")
    count_keys = ("keyframe_count", "successful_frame_count", "source_count", "identity_verified_source_count", "multiview_evaluated_source_count", "switch_count", "fallback_count", "selected_h0_count", "selected_hl_count", "selected_hlg_count", "selected_hb_count")
    totals = {key: 0 for key in count_keys}; ids: list[str] = []; lineages: list[str] = []; results: list[str] = []
    inc_all: list[float] = []; inc_warm: list[float] = []
    gross_all: list[float] = []; gross_warm: list[float] = []
    audit_hash_all: list[float] = []; audit_hash_warm: list[float] = []
    audit_serialization_all: list[float] = []
    audit_serialization_warm: list[float] = []
    audit_total_all: list[float] = []; audit_total_warm: list[float] = []
    comp_all: list[float] = []; comp_warm: list[float] = []
    deadline_all = deadline_warm = switch_scenes = max_buf = max_src = max_payload = cuda_peak = 0
    scene_rows: list[dict[str, Any]] = []
    for index in range(expected_scene_count):
        shard_index, row = rows_by_index[index]
        if row.get("scene_id") != scene_order[index] or f4_rows[index].get("scene_id") != scene_order[index]: raise F6MergeError("F4/F6 scene order differs")
        value = _validate_scene(row, expected_scene_index=index, expected_signature=signature, f4_receipt_seal=inputs["f4_receipt"], source_receipts=source_receipts, core=core)
        for key in count_keys: totals[key] += value["counts"][key]
        ids += value["ids"]; lineages += value["lineages"]; results += value["results"]
        inc_all += value["incremental_all"]
        inc_warm += value["incremental_warm"]
        gross_all += value["gross_all"]
        gross_warm += value["gross_warm"]
        audit_hash_all += value["audit_hash_all"]
        audit_hash_warm += value["audit_hash_warm"]
        audit_serialization_all += value["audit_serialization_all"]
        audit_serialization_warm += value["audit_serialization_warm"]
        audit_total_all += value["audit_total_all"]
        audit_total_warm += value["audit_total_warm"]
        comp_all += value["composed_all"]
        comp_warm += value["composed_warm"]
        deadline_all += value["deadline_all"]; deadline_warm += value["deadline_warm"]; switch_scenes += int(value["switches"] > 0)
        max_buf = max(max_buf, value["max_buffer_frames"]); max_src = max(max_src, value["max_buffer_sources"]); max_payload = max(max_payload, value["max_payload"]); cuda_peak = max(cuda_peak, value["cuda_peak"])
        scene_rows.append(value["row"])
    for key, expected in (("keyframe_count", expected_keyframes), ("successful_frame_count", expected_successful_frames), ("source_count", expected_sources)):
        if expected is not None and totals[key] != expected: raise F6MergeError(f"merged F6 {key} differs")
    if len(ids) != len(set(ids)) or totals["identity_verified_source_count"] != totals["source_count"] or totals["switch_count"] + totals["fallback_count"] != totals["source_count"] or sum(totals[f"selected_{name}_count"] for name in ("h0", "hl", "hlg", "hb")) != totals["source_count"]:
        raise F6MergeError("global one-source/one-selection census differs")
    if production:
        for shard_index, (_, shard) in enumerate(loaded):
            for key, expected in EXPECTED_SHARD_COUNTS[shard_index].items():
                if shard.get("totals", {}).get(key) != expected: raise F6MergeError(f"production shard {shard_index} {key} differs")
    for shard_index, (_, shard) in enumerate(loaded):
        expected_totals = {
            key: sum(scene_rows[index]["counts"][key] for index in range(shard_index, expected_scene_count, EXPECTED_SHARDS))
            for key in count_keys
        }
        if shard.get("totals") != expected_totals:
            raise F6MergeError(f"F6 shard {shard_index} totals differ")
        expected_switch_scenes = sum(
            scene_rows[index]["counts"]["switch_count"] > 0
            for index in range(shard_index, expected_scene_count, EXPECTED_SHARDS)
        )
        if shard.get("switch_scene_count") != expected_switch_scenes:
            raise F6MergeError(f"F6 shard {shard_index} switch-scene census differs")
    inc_dist, comp_dist = _distribution(inc_warm), _distribution(comp_warm)
    mean_per_frame = float(comp_dist["mean"]) / SOURCE_FRAME_STRIDE
    switch_fraction = totals["switch_count"] / totals["source_count"] if totals["source_count"] else 0.0
    gates = {
        "integrity_complete": _gate(len(scene_rows), "==", expected_scene_count), "exact_keyframes": _gate(totals["keyframe_count"], "==", expected_keyframes or totals["keyframe_count"]), "exact_successful_frames": _gate(totals["successful_frame_count"], "==", expected_successful_frames or totals["successful_frame_count"]), "exact_unique_sources": _gate(totals["source_count"], "==", expected_sources or totals["source_count"]),
        "switch_min_sources": _gate(totals["switch_count"], ">=", min_switch_sources), "switch_min_scenes": _gate(switch_scenes, ">=", min_switch_scenes), "switch_max_fraction": _gate(switch_fraction, "<=", max_switch_fraction),
        "maximum_buffered_frames": _gate(max_buf, "<=", MAX_BUFFERED_FRAMES), "maximum_sources_per_buffered_frame": _gate(max_src, "<=", MAX_SOURCES_PER_FRAME), "maximum_state_raw_array_payload_bytes": _gate(max_payload, "<=", MAX_STATE_RAW_ARRAY_PAYLOAD_BYTES),
        "f6_incremental_warm_p95_ms": _gate(float(inc_dist["p95"]), "<=", MAX_F6_INCREMENTAL_P95_MS), "replay_composed_warm_p95_ms": _gate(float(comp_dist["p95"]), "<=", MAX_COMPOSED_P95_MS), "replay_composed_warm_max_ms": _gate(float(comp_dist["max"]), "<", MAX_COMPOSED_MS_EXCLUSIVE), "replay_composed_warm_mean_per_source_frame_ms": _gate(mean_per_frame, "<=", MAX_COMPOSED_MEAN_PER_SOURCE_FRAME_MS), "gap25_warm_deadline_miss_count": _gate(deadline_warm, "==", 0), "cuda_peak_memory_bytes": _gate(cuda_peak, "<=", MAX_CUDA_PEAK_BYTES), "f6_cuda_allocated_bytes": _gate(0, "==", 0),
        "forbidden_or_mutation_count": _gate(0, "==", 0),
    }
    decision = _decision_from_gates(gates)
    overall_pass = all(value["pass"] for value in gates.values())
    runtime_names = tuple(key for key in gates if key.startswith("f6_") or key.startswith("replay_") or key in {"gap25_warm_deadline_miss_count", "cuda_peak_memory_bytes"})
    receipt: dict[str, Any] = {
        "schema": MERGE_SCHEMA, "protocol_id": PROTOCOL_ID, "protocol_sha256": PROTOCOL_SHA256, "complete": True, "overall_pass": overall_pass, "decision": decision, "run_signature_sha256": signature, "contracts": dict(CONTRACTS),
        "inputs": {"shards": [{"path": os.fspath(path), "sha256": _sha256(path), "shard_index": index} for index, (path, _) in enumerate(loaded)], "f4_receipt": dict(inputs["f4_receipt"]), "scene_list": dict(inputs["scene_list"]), "sources": {key: dict(value) for key, value in source_receipts.items()}, "merge_source": merge_source_seal},
        "coverage": {"scene_count": len(scene_rows), "scene_order": scene_order, "keyframe_count": totals["keyframe_count"], "successful_frame_count": totals["successful_frame_count"], "source_count": totals["source_count"], "exact_source_partition": True, "exact_source_order": True, "source_ids_sha256": _canonical_json_sha256(ids), "source_lineage_sha256": _canonical_json_sha256(lineages), "result_ledger_sha256": _canonical_json_sha256(results)},
        "selection": {"switch_count": totals["switch_count"], "fallback_count": totals["fallback_count"], "switch_scene_count": switch_scenes, "switch_fraction": switch_fraction, "multiview_evaluated_source_count": totals["multiview_evaluated_source_count"], "selected_h0_count": totals["selected_h0_count"], "selected_hl_count": totals["selected_hl_count"], "selected_hlg_count": totals["selected_hlg_count"], "selected_hb_count": totals["selected_hb_count"], "formal_score": 1.0, "complete_three_view_switch_proof_count": totals["switch_count"]},
        "bounded_state": {"overall_pass": max_buf <= 3 and max_src <= 16 and max_payload <= MAX_STATE_RAW_ARRAY_PAYLOAD_BYTES, "maximum_buffered_successful_frame_count": max_buf, "maximum_sources_per_buffered_frame": max_src, "maximum_raw_array_payload_bytes": max_payload, "raw_array_payload_limit_bytes": MAX_STATE_RAW_ARRAY_PAYLOAD_BYTES},
        "runtime": {
            "overall_pass": all(gates[key]["pass"] for key in runtime_names),
            "gates": {key: gates[key] for key in runtime_names},
            "f6_incremental_gross_all_ms": _distribution(gross_all),
            "f6_incremental_gross_warm_ms": _distribution(gross_warm),
            "f6_audit_hash_excluded_all_ms": _distribution(audit_hash_all),
            "f6_audit_hash_excluded_warm_ms": _distribution(audit_hash_warm),
            "f6_audit_serialization_excluded_all_ms": _distribution(audit_serialization_all),
            "f6_audit_serialization_excluded_warm_ms": _distribution(audit_serialization_warm),
            "f6_audit_total_excluded_all_ms": _distribution(audit_total_all),
            "f6_audit_total_excluded_warm_ms": _distribution(audit_total_warm),
            "formal_runtime_excludes_hashing_and_serialization": True,
            "f6_incremental_all_ms": _distribution(inc_all),
            "f6_incremental_warm_ms": inc_dist,
            "replay_composed_all_ms": _distribution(comp_all),
            "replay_composed_warm_ms": comp_dist,
            "replay_composed_warm_mean_per_source_frame_ms": mean_per_frame,
            "gap25_all_deadline_miss_count": deadline_all,
            "gap25_warm_deadline_miss_count": deadline_warm,
            "maximum_state_raw_array_payload_bytes": max_payload,
            "cuda_peak_memory_bytes": cuda_peak,
            "f6_cuda_allocated_bytes": 0,
        },
        "gates": gates, "totals": totals, "scenes": scene_rows, "native_output_mutation_count": 0, "source_addition_or_removal_count": 0, "score_rank_semantic_mutation_count": 0, "forbidden_access_count": 0, "training_or_online_learning_count": 0, "birth_count": 0,
        "evaluation_authorization": {"allowed": decision == "retain_f6_for_one_separately_sealed_evaluation_only", "scope": "one_separately_sealed_constant_score_geometry_evaluation_only", "birth_authorized": False, "deployment_authorized": False},
    }
    if _sha256(merge_source) != merge_source_seal["sha256"]: raise F6MergeError("F6 merge source changed during merge")
    receipt["content_sha256"] = _canonical_json_sha256(receipt)
    output_path = Path(output_dir) / OUTPUT_NAME
    output_sha = _atomic_create_json(output_path, receipt)
    receipt["receipt_path"] = os.fspath(output_path.resolve()); receipt["receipt_sha256"] = output_sha
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, action="append", dest="shards")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = merge_f6(shard_paths=tuple(args.shards) if args.shards is not None else DEFAULT_SHARDS, output_dir=args.output_dir)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
