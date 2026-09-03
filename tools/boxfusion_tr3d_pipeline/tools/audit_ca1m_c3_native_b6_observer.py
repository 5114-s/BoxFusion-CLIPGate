#!/usr/bin/env python3
"""Fail-closed audit for the CA-1M-native final-OBB B6 observer.

The authoritative identity pair is produced in one process: ``anchor-root``
is written immediately before final-box observation and ``observer-root`` is
written afterwards.  The observer is valid only when those prediction files
are byte-identical and every final row has exactly one diagnostic mapping.
The older P1 tree is informational (independent GPU fusion is not bitwise
stable), but labels, scores, row count/order, and Selective-Boxer decisions
must still agree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.tr3d_r4_smov_observer import corners_to_yaw_boxes


DIAGNOSTIC_SUFFIX = "_ca1m_native_b6.npz"
DIAGNOSTIC_SCHEMA = "boxfusion.ca1m_native_b6_observer.v1"
EXPECTED_FEATURE_NAMES = (
    "detector_score",
    "support_given_depth",
    "occluded_given_depth",
    "free_given_depth",
    "invalid_ratio",
    "view_coverage",
    "sample_support",
    "area_quality",
    "area_stability",
    "support_view_mean",
    "support_view_min",
    "free_view_max",
    "aspect_balance",
    "height_balance",
)
DEPTH_CLASS_NAMES = ("support", "occluded", "free_space", "invalid")
REQUIRED_DIAGNOSTIC_KEYS = {
    "schema",
    "scene_id",
    "result_indices",
    "corners",
    "scores",
    "feature_names",
    "features",
    "valid_evidence",
    "summary_json",
    "stable_ids",
    "yaw_boxes",
    "used_frame_ids",
    "topk_frame_ids",
    "topk_view_valid",
    "topk_projected_area_fraction",
    "per_view_depth_counts",
    "per_view_depth_evidence",
    "aggregate_depth_counts",
    "aggregate_depth_evidence",
    "aggregate_view_count",
    "aggregate_sample_count",
    "projectable",
}
SAFETY_SCALAR_CONTRACTS = {
    "complete": True,
    "observer_only": True,
    "mutation_enabled": False,
    "applied_count": 0,
    "ground_truth_access": False,
    "clip_access": False,
}
BOXER_STABLE_KEYS = (
    "schema",
    "scene_id",
    "frame_id",
    "attempt_id",
    "apply_stage",
    "mode",
    "mutation_enabled",
    "selective_gate_enabled",
    "selective_gate",
    "count",
    "eligible_count",
    "applied_count",
    "fallback_count",
    "gate_accepted",
    "gate_reasons",
    "gate_rejection_counts",
    "cutr_geometry_sha256",
    "actual_geometry_sha256",
    "scores_sha256",
    "input_pred_proj_xy_sha256",
    "boxer_commit",
    "boxer_checkpoint_sha256",
)
RUNTIME_PATTERN = re.compile(
    r"^Cost:\s*([0-9]+(?:\.[0-9]+)?)\s*s\s+Average FPS:\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*$",
    re.MULTILINE,
)


def _fractions(counts: np.ndarray) -> np.ndarray:
    total = counts.sum(axis=-1, keepdims=True, dtype=np.int64)
    return np.divide(
        counts,
        total,
        out=np.zeros(counts.shape, dtype=np.float32),
        where=total > 0,
    ).astype(np.float32)


def recompute_features(
    scores: np.ndarray,
    yaw_boxes: np.ndarray,
    topk_valid: np.ndarray,
    topk_area: np.ndarray,
    per_view_counts: np.ndarray,
    aggregate_counts: np.ndarray,
) -> np.ndarray:
    """Independently reconstruct all 14 native-B6 diagnostic features."""

    count = len(scores)
    output = np.zeros((count, len(EXPECTED_FEATURE_NAMES)), dtype=np.float32)
    if count == 0:
        return output
    aggregate_total = aggregate_counts.sum(axis=1, dtype=np.int64)
    classified = aggregate_counts[:, :3].sum(axis=1, dtype=np.int64)
    evidence = np.divide(
        aggregate_counts,
        aggregate_total[:, None],
        out=np.zeros_like(aggregate_counts, dtype=np.float32),
        where=aggregate_total[:, None] > 0,
    )
    conditional = np.divide(
        aggregate_counts[:, :3],
        classified[:, None],
        out=np.zeros((count, 3), dtype=np.float32),
        where=classified[:, None] > 0,
    )
    view_total = per_view_counts.sum(axis=2, dtype=np.int64)
    view_classified = per_view_counts[:, :, :3].sum(axis=2, dtype=np.int64)
    support_by_view = np.divide(
        per_view_counts[:, :, 0],
        view_classified,
        out=np.zeros(view_classified.shape, dtype=np.float32),
        where=view_classified > 0,
    )
    free_by_view = np.divide(
        per_view_counts[:, :, 2],
        view_classified,
        out=np.zeros(view_classified.shape, dtype=np.float32),
        where=view_classified > 0,
    )
    for row in range(count):
        valid_slots = topk_valid[row] & (view_total[row] > 0)
        areas = topk_area[row, topk_valid[row]]
        support_values = support_by_view[row, valid_slots]
        free_values = free_by_view[row, valid_slots]
        if len(areas):
            area_mean = float(areas.mean())
            logs = np.log(np.maximum(areas, 1e-6))
            area_stability = float(np.exp(-logs.std()))
        else:
            area_mean = 0.0
            area_stability = 0.0
        dx, dy, dz = (float(value) for value in yaw_boxes[row, 3:6])
        planar = float(np.sqrt(dx * dy))
        output[row] = (
            scores[row],
            conditional[row, 0],
            conditional[row, 1],
            conditional[row, 2],
            evidence[row, 3],
            float(topk_valid[row].sum() / topk_valid.shape[1]),
            float(np.clip(np.log1p(aggregate_total[row]) / np.log1p(65536), 0, 1)),
            float(np.clip(area_mean / 0.10, 0, 1)),
            float(np.clip(area_stability, 0, 1)),
            float(support_values.mean()) if len(support_values) else 0.0,
            float(support_values.min()) if len(support_values) else 0.0,
            float(free_values.max()) if len(free_values) else 0.0,
            min(dx, dy) / max(dx, dy),
            min(dz, planar) / max(dz, planar),
        )
    return output


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(data) == 0:
        return {"count": 0, "min": 0.0, "q25": 0.0, "q50": 0.0, "q75": 0.0, "max": 0.0}
    return {
        "count": int(len(data)),
        "min": float(data.min()),
        "q25": float(np.quantile(data, 0.25)),
        "q50": float(np.quantile(data, 0.50)),
        "q75": float(np.quantile(data, 0.75)),
        "max": float(data.max()),
    }


def _histogram(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(unique, counts)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(array: np.ndarray, name: str) -> Any:
    value = np.asarray(array)
    if value.shape != ():
        raise ValueError(f"diagnostic {name} must be scalar")
    return value.item()


def read_scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list must be non-empty and unique")
    if any(not scene.isdigit() for scene in scenes):
        raise ValueError("CA-1M scene identifiers must be numeric")
    return scenes


def exact_scene_files(
    root: Path,
    scenes: Iterable[str],
    suffix: str,
) -> dict[str, Path]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"artifact root must be a regular directory: {root}")
    expected = {f"{scene}{suffix}" for scene in scenes}
    actual = {path.name for path in root.glob(f"*{suffix}") if path.is_file()}
    if actual != expected:
        raise ValueError(
            f"scene files disagree in {root}: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    result = {scene: root / f"{scene}{suffix}" for scene in scenes}
    for path in result.values():
        mode = path.lstat().st_mode
        if path.is_symlink() or not stat.S_ISREG(mode) or path.stat().st_size <= 0:
            raise ValueError(f"artifact must be a non-empty regular file: {path}")
    return result


def load_prediction(path: Path) -> list[tuple[int, np.ndarray, float]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
        if handle.read(1):
            raise ValueError(f"trailing bytes in prediction: {path}")
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], list):
        raise ValueError(f"prediction must contain exactly one list batch: {path}")
    rows: list[tuple[int, np.ndarray, float]] = []
    for row_index, row in enumerate(payload[0]):
        if not isinstance(row, tuple) or len(row) != 3:
            raise ValueError(f"invalid prediction row {row_index}: {path}")
        label, corners, score = row
        corners = np.asarray(corners)
        score = float(score)
        if int(label) != 0:
            raise ValueError(f"non-class-agnostic row {row_index}: {path}")
        if corners.shape != (8, 3) or not np.isfinite(corners).all():
            raise ValueError(f"invalid OBB corners in row {row_index}: {path}")
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"invalid score in row {row_index}: {path}")
        rows.append((int(label), np.array(corners, copy=True), score))
    return rows


def compare_same_run(
    scene: str,
    anchor_path: Path,
    observer_path: Path,
) -> tuple[list[tuple[int, np.ndarray, float]], dict[str, Any]]:
    anchor_hash = sha256(anchor_path)
    observer_hash = sha256(observer_path)
    if anchor_hash != observer_hash:
        raise ValueError(f"{scene}: same-run prediction bytes differ")
    anchor = load_prediction(anchor_path)
    observer = load_prediction(observer_path)
    if len(anchor) != len(observer):
        raise ValueError(f"{scene}: same-run prediction count differs")
    for row_index, (left, right) in enumerate(zip(anchor, observer)):
        if left[0] != right[0] or left[2] != right[2] or not np.array_equal(left[1], right[1]):
            raise ValueError(f"{scene}: same-run semantic row differs at {row_index}")
    return observer, {
        "rows": len(observer),
        "byte_identity": True,
        "semantic_identity": True,
        "prediction_sha256": observer_hash,
    }


def audit_diagnostic(
    scene: str,
    path: Path,
    prediction: list[tuple[int, np.ndarray, float]],
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        keys = set(payload.files)
        missing = REQUIRED_DIAGNOSTIC_KEYS - keys
        if missing:
            raise ValueError(f"{scene}: diagnostic keys missing: {sorted(missing)}")
        if str(scalar(payload["schema"], "schema")) != DIAGNOSTIC_SCHEMA:
            raise ValueError(f"{scene}: diagnostic schema disagrees")
        if str(scalar(payload["scene_id"], "scene_id")) != scene:
            raise ValueError(f"{scene}: diagnostic scene_id disagrees")
        for key, expected in SAFETY_SCALAR_CONTRACTS.items():
            if key not in keys or scalar(payload[key], key) != expected:
                raise ValueError(f"{scene}: diagnostic safety scalar {key} disagrees")

        count = len(prediction)
        result_indices = np.asarray(payload["result_indices"])
        if result_indices.ndim != 1 or result_indices.dtype.kind not in "iu":
            raise ValueError(f"{scene}: result_indices must be an integer vector")
        expected_indices = np.arange(count, dtype=np.int64)
        if not np.array_equal(result_indices.astype(np.int64, copy=False), expected_indices):
            raise ValueError(f"{scene}: result_indices must equal arange(prediction_rows)")

        corners = np.asarray(payload["corners"])
        scores = np.asarray(payload["scores"])
        yaw_boxes = np.asarray(payload["yaw_boxes"])
        features = np.asarray(payload["features"])
        names_array = np.asarray(payload["feature_names"])
        valid_evidence = np.asarray(payload["valid_evidence"])
        if corners.shape != (count, 8, 3) or corners.dtype.kind != "f":
            raise ValueError(f"{scene}: corners must be a floating [N,8,3] tensor")
        if not np.isfinite(corners).all():
            raise ValueError(f"{scene}: diagnostic corners contain non-finite values")
        if scores.shape != (count,) or scores.dtype.kind != "f" or not np.isfinite(scores).all():
            raise ValueError(f"{scene}: scores must be a finite floating [N] vector")
        if names_array.ndim != 1 or names_array.dtype.kind not in "US":
            raise ValueError(f"{scene}: feature_names must be a string vector")
        feature_names = tuple(str(value) for value in names_array.tolist())
        if feature_names != EXPECTED_FEATURE_NAMES:
            raise ValueError(f"{scene}: exact 14-column feature schema/order disagrees")
        if features.shape != (count, len(feature_names)) or features.dtype.kind != "f":
            raise ValueError(f"{scene}: features must have shape [N,F]")
        if (
            not np.isfinite(features).all()
            or np.any(features < 0.0)
            or np.any(features > 1.0)
        ):
            raise ValueError(f"{scene}: features must be finite in [0,1]")
        if yaw_boxes.shape != (count, 7) or yaw_boxes.dtype.kind != "f":
            raise ValueError(f"{scene}: yaw_boxes must be a floating [N,7] tensor")
        if not np.isfinite(yaw_boxes).all() or np.any(yaw_boxes[:, 3:6] <= 0.0):
            raise ValueError(f"{scene}: yaw_boxes must be finite with positive extents")
        if valid_evidence.shape != (count,) or valid_evidence.dtype != np.bool_:
            raise ValueError(f"{scene}: valid_evidence must be a bool [N] vector")
        stable_ids = np.asarray(payload.get("stable_ids"))
        if stable_ids.shape != (count,) or stable_ids.dtype.kind not in "iu":
            raise ValueError(f"{scene}: stable_ids must be an integer [N] vector")
        if count and (np.any(stable_ids < 0) or len(np.unique(stable_ids)) != count):
            raise ValueError(f"{scene}: stable_ids must be unique and non-negative")

        topk_valid = np.asarray(payload.get("topk_view_valid"))
        topk_ids = np.asarray(payload.get("topk_frame_ids"))
        topk_area = np.asarray(payload.get("topk_projected_area_fraction"))
        used_frame_ids = np.asarray(payload.get("used_frame_ids"))
        per_view_counts = np.asarray(payload.get("per_view_depth_counts"))
        per_view_evidence = np.asarray(payload.get("per_view_depth_evidence"))
        aggregate_counts = np.asarray(payload.get("aggregate_depth_counts"))
        aggregate_evidence = np.asarray(payload.get("aggregate_depth_evidence"))
        aggregate_view_count = np.asarray(payload.get("aggregate_view_count"))
        aggregate_sample_count = np.asarray(payload.get("aggregate_sample_count"))
        projectable = np.asarray(payload.get("projectable"))
        if (
            topk_valid.ndim != 2
            or topk_valid.shape[0] != count
            or topk_valid.shape[1] < 1
            or topk_valid.dtype != np.bool_
        ):
            raise ValueError(f"{scene}: invalid topk_view_valid")
        if topk_ids.shape != topk_valid.shape or topk_ids.dtype.kind not in "iu":
            raise ValueError(f"{scene}: invalid topk_frame_ids")
        if topk_area.shape != topk_valid.shape or topk_area.dtype.kind != "f":
            raise ValueError(f"{scene}: invalid Top-K projected area")
        if (
            not np.isfinite(topk_area).all()
            or np.any(topk_area < 0.0)
            or np.any(topk_area > 1.0)
            or np.any(topk_area[~topk_valid] != 0.0)
        ):
            raise ValueError(f"{scene}: invalid Top-K projected area values")
        if used_frame_ids.ndim != 1 or used_frame_ids.dtype.kind not in "iu":
            raise ValueError(f"{scene}: used_frame_ids must be an integer vector")
        if (
            np.any(used_frame_ids < 0)
            or len(np.unique(used_frame_ids)) != len(used_frame_ids)
            or (len(used_frame_ids) > 1 and np.any(np.diff(used_frame_ids) <= 0))
        ):
            raise ValueError(f"{scene}: used_frame_ids must be unique and increasing")
        if np.any(topk_ids[topk_valid] < 0) or np.any(topk_ids[~topk_valid] != -1):
            raise ValueError(f"{scene}: invalid Top-K frame sentinels")
        if np.any(~np.isin(topk_ids[topk_valid], used_frame_ids)):
            raise ValueError(f"{scene}: Top-K references an unused frame")
        if topk_valid.shape[1] > 1 and np.any(topk_valid[:, 1:] & ~topk_valid[:, :-1]):
            raise ValueError(f"{scene}: Top-K valid slots must be left packed")
        if per_view_counts.shape != (*topk_valid.shape, 4) or per_view_counts.dtype.kind not in "iu":
            raise ValueError(f"{scene}: invalid per-view depth counts")
        if np.any(per_view_counts < 0) or np.any(per_view_counts[~topk_valid] != 0):
            raise ValueError(f"{scene}: invalid or nonzero padded per-view depth counts")
        if per_view_evidence.shape != per_view_counts.shape or not np.isfinite(per_view_evidence).all():
            raise ValueError(f"{scene}: invalid per-view evidence")
        if aggregate_counts.shape != (count, 4) or aggregate_counts.dtype.kind not in "iu":
            raise ValueError(f"{scene}: invalid aggregate depth counts")
        if np.any(aggregate_counts < 0):
            raise ValueError(f"{scene}: negative aggregate depth counts")
        if aggregate_evidence.shape != aggregate_counts.shape or not np.isfinite(aggregate_evidence).all():
            raise ValueError(f"{scene}: invalid aggregate evidence")
        expected_counts = per_view_counts.sum(axis=1, dtype=np.int64)
        if not np.array_equal(aggregate_counts, expected_counts):
            raise ValueError(f"{scene}: aggregate counts != sum(per-view counts)")
        expected_per_view = _fractions(per_view_counts)
        expected_aggregate = _fractions(aggregate_counts)
        if not np.array_equal(per_view_evidence, expected_per_view) or not np.array_equal(
            aggregate_evidence, expected_aggregate
        ):
            raise ValueError(f"{scene}: redundant depth evidence disagrees with counts")
        expected_view_count = topk_valid.sum(axis=1, dtype=np.int32)
        expected_sample_count = aggregate_counts.sum(axis=1, dtype=np.int64)
        if (
            aggregate_view_count.shape != (count,)
            or aggregate_view_count.dtype.kind not in "iu"
            or not np.array_equal(aggregate_view_count, expected_view_count)
        ):
            raise ValueError(f"{scene}: aggregate_view_count disagrees with Top-K")
        if (
            aggregate_sample_count.shape != (count,)
            or aggregate_sample_count.dtype.kind not in "iu"
            or not np.array_equal(aggregate_sample_count, expected_sample_count)
        ):
            raise ValueError(f"{scene}: aggregate_sample_count disagrees with counts")
        expected_projectable = topk_valid.any(axis=1)
        if (
            projectable.shape != (count,)
            or projectable.dtype != np.bool_
            or not np.array_equal(projectable, expected_projectable)
        ):
            raise ValueError(f"{scene}: projectable disagrees with Top-K views")
        classified_per_view = per_view_counts[:, :, :3].sum(axis=2, dtype=np.int64)
        classified_samples = aggregate_counts[:, :3].sum(axis=1, dtype=np.int64)
        expected_valid = classified_samples > 0
        if not np.array_equal(valid_evidence, expected_valid):
            raise ValueError(f"{scene}: valid_evidence disagrees with depth counts")

        expected_corners = (
            np.stack([row[1] for row in prediction], axis=0).astype(corners.dtype, copy=False)
            if count
            else np.empty((0, 8, 3), dtype=corners.dtype)
        )
        expected_scores = np.asarray([row[2] for row in prediction], dtype=scores.dtype)
        if not np.array_equal(corners, expected_corners):
            maximum = float(np.max(np.abs(corners.astype(np.float64) - expected_corners)))
            raise ValueError(f"{scene}: diagnostic corners do not map exactly, max_abs={maximum}")
        if not np.array_equal(scores, expected_scores):
            raise ValueError(f"{scene}: diagnostic scores do not map exactly")
        expected_yaw_boxes = corners_to_yaw_boxes(corners).astype(np.float32)
        if not np.array_equal(yaw_boxes, expected_yaw_boxes):
            maximum = (
                float(np.max(np.abs(yaw_boxes.astype(np.float64) - expected_yaw_boxes)))
                if count
                else 0.0
            )
            raise ValueError(f"{scene}: yaw_boxes do not recompute from corners, max_abs={maximum}")
        expected_features = recompute_features(
            scores.astype(np.float32, copy=False),
            expected_yaw_boxes,
            topk_valid,
            topk_area,
            per_view_counts,
            aggregate_counts,
        )
        if not np.array_equal(features, expected_features):
            maximum = (
                float(np.max(np.abs(features.astype(np.float64) - expected_features)))
                if count
                else 0.0
            )
            raise ValueError(f"{scene}: 14-column features do not recompute, max_abs={maximum}")

        summary = json.loads(str(scalar(payload["summary_json"], "summary_json")))
        if not isinstance(summary, dict):
            raise ValueError(f"{scene}: summary_json must decode to an object")
        required_summary = {
            "enabled": True,
            "observer_only": True,
            "mutation_enabled": False,
        }
        for key, expected in required_summary.items():
            if summary.get(key) is not expected:
                raise ValueError(f"{scene}: summary {key} safety contract disagrees")
        if int(summary.get("applied_count", -1)) != 0:
            raise ValueError(f"{scene}: observer applied_count is not zero")
        if int(summary.get("prediction_rows", -1)) != count:
            raise ValueError(f"{scene}: summary prediction_rows disagrees")
        if int(summary.get("mapping_rows", -1)) != count:
            raise ValueError(f"{scene}: summary mapping_rows disagrees")
        if int(summary.get("projectable_rows", -1)) != int(projectable.sum()):
            raise ValueError(f"{scene}: summary projectable_rows disagrees")
        if int(summary.get("frame_count", -1)) != len(used_frame_ids):
            raise ValueError(f"{scene}: summary frame_count disagrees")
        if summary.get("orientation_contract") != "processed_upright_per_frame_intrinsics_v1":
            raise ValueError(f"{scene}: summary orientation contract disagrees")
        observer_seconds = float(summary.get("observer_seconds", -1.0))
        if not np.isfinite(observer_seconds) or observer_seconds < 0.0:
            raise ValueError(f"{scene}: invalid observer_seconds")
        valid_count = int(np.count_nonzero(valid_evidence))
        if "valid_evidence_rows" in summary and int(summary["valid_evidence_rows"]) != valid_count:
            raise ValueError(f"{scene}: summary valid_evidence_rows disagrees")

        sampled_per_view = per_view_counts.sum(axis=2, dtype=np.int64) > 0
        sampled_views = (sampled_per_view & topk_valid).sum(axis=1, dtype=np.int32)
        classified_views = ((classified_per_view > 0) & topk_valid).sum(
            axis=1, dtype=np.int32
        )
        row_evidence = [
            {
                "result_index": int(row),
                "stable_id": int(stable_ids[row]),
                "valid_views": int(aggregate_view_count[row]),
                "sampled_views": int(sampled_views[row]),
                "classified_views": int(classified_views[row]),
                "total_samples": int(aggregate_sample_count[row]),
                "classified_samples": int(classified_samples[row]),
                "valid_evidence": bool(valid_evidence[row]),
            }
            for row in range(count)
        ]

        return {
            "prediction_rows": count,
            "mapping_rows": count,
            "mapping_coverage": 1.0,
            "valid_evidence_rows": valid_count,
            "valid_evidence_coverage": float(valid_count / count) if count else 1.0,
            "feature_count": len(feature_names),
            "feature_names": list(feature_names),
            "yaw_and_features_recomputed": True,
            "depth_redundancy_recomputed": True,
            "top_k": int(topk_valid.shape[1]),
            "used_frames": int(len(used_frame_ids)),
            "projectable_rows": int(projectable.sum()),
            "aggregate_depth_counts": {
                name: int(aggregate_counts[:, column].sum())
                for column, name in enumerate(DEPTH_CLASS_NAMES)
            },
            "valid_views_per_row": _distribution(aggregate_view_count),
            "sampled_views_per_row": _distribution(sampled_views),
            "classified_views_per_row": _distribution(classified_views),
            "total_samples_per_row": _distribution(aggregate_sample_count),
            "classified_samples_per_row": _distribution(classified_samples),
            "rows_by_valid_view_count": _histogram(aggregate_view_count),
            "rows_by_classified_view_count": _histogram(classified_views),
            "row_evidence": row_evidence,
            "observer_seconds": observer_seconds,
            "diagnostic_sha256": sha256(path),
        }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row {line_number}: {path}")
            rows.append(value)
    if not rows:
        raise ValueError(f"empty Boxer diagnostic: {path}")
    return rows


def audit_boxer(scene: str, historical_path: Path, observer_path: Path) -> dict[str, Any]:
    historical = read_jsonl(historical_path)
    observer = read_jsonl(observer_path)
    if len(historical) != len(observer):
        raise ValueError(f"{scene}: Boxer call count differs")
    for row_index, (left, right) in enumerate(zip(historical, observer)):
        for key in BOXER_STABLE_KEYS:
            if left.get(key) != right.get(key):
                raise ValueError(
                    f"{scene}: Boxer deterministic field {key!r} differs at call {row_index}"
                )
    return {
        "calls": len(observer),
        "proposals": sum(int(row.get("count", 0)) for row in observer),
        "applied": sum(int(row.get("applied_count", 0)) for row in observer),
        "deterministic_fields_identity": True,
        "diagnostic_sha256": sha256(observer_path),
    }


def audit_historical_prediction(
    scene: str,
    historical_path: Path,
    current: list[tuple[int, np.ndarray, float]],
) -> dict[str, Any]:
    historical = load_prediction(historical_path)
    if len(historical) != len(current):
        raise ValueError(f"{scene}: historical P1 row count differs")
    if any(left[0] != right[0] or left[2] != right[2] for left, right in zip(historical, current)):
        raise ValueError(f"{scene}: historical P1 label/score/order differs")
    deltas = np.asarray(
        [
            np.max(np.abs(left[1].astype(np.float64) - right[1].astype(np.float64)))
            for left, right in zip(historical, current)
        ],
        dtype=np.float64,
    )
    return {
        "rows": len(current),
        "historical_prediction_sha256": sha256(historical_path),
        "changed_corner_rows": int(np.count_nonzero(deltas)),
        "corner_max_abs": float(deltas.max()) if len(deltas) else 0.0,
        "label_score_order_identity": True,
        "informational_corner_drift_only": True,
    }


def audit_runtime_log(path: Path) -> dict[str, float | str]:
    matches = RUNTIME_PATTERN.findall(path.read_text(errors="strict"))
    if len(matches) != 1:
        raise ValueError(f"expected one Cost/Average FPS summary: {path}")
    cost, fps = (float(value) for value in matches[0])
    if not np.isfinite(cost) or not np.isfinite(fps) or cost <= 0.0 or fps <= 0.0:
        raise ValueError(f"invalid runtime summary: {path}")
    return {
        "cost_seconds": cost,
        "average_fps": fps,
        "frame_equivalent": cost * fps,
        "sha256": sha256(path),
    }


def write_json_atomic(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing existing audit report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing existing audit report: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def build_train_readiness(
    *,
    engineering_identity_ok: bool,
    mapping_coverage: float,
    valid_evidence_coverage: float,
    feature_integrity_ok: bool,
) -> dict[str, Any]:
    """Report prerequisites without authorizing training from validation data."""

    evidence_warning_threshold = 0.80
    prerequisites = {
        "engineering_identity_ok": bool(engineering_identity_ok),
        "mapping_coverage_is_100_percent": bool(mapping_coverage == 1.0),
        "feature_integrity_ok": bool(feature_integrity_ok),
        "valid_evidence_warning_threshold_met": bool(
            valid_evidence_coverage >= evidence_warning_threshold
        ),
    }
    return {
        "authorized": False,
        "status": "NOT_AUTHORIZED_FIXED10_VALIDATION_ONLY",
        "fixed10_validation_only": True,
        "ca1m_train_only_data_used": False,
        "evidence_warning_threshold": evidence_warning_threshold,
        "prerequisites": prerequisites,
        "prerequisites_passed": bool(all(prerequisites.values())),
        "reason": (
            "The fixed-ten CA-1M validation observer can validate engineering "
            "identity and feature integrity, but it cannot authorize model "
            "training, calibration, threshold selection, or active validation."
        ),
        "required_next_gate": (
            "Collect and audit an independent CA-1M train-only subset, then "
            "freeze scene-level train/development/calibration splits."
        ),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--scene-list", type=Path, required=True)
    result.add_argument("--anchor-root", type=Path, required=True)
    result.add_argument("--observer-root", type=Path, required=True)
    result.add_argument("--diagnostics-root", type=Path, required=True)
    result.add_argument("--historical-prediction-root", type=Path, required=True)
    result.add_argument("--historical-boxer-root", type=Path, required=True)
    result.add_argument("--observer-boxer-root", type=Path, required=True)
    result.add_argument("--historical-log-root", type=Path, required=True)
    result.add_argument("--observer-log-root", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    scenes = read_scenes(args.scene_list)
    anchor_files = exact_scene_files(args.anchor_root, scenes, "_boxes.pkl")
    observer_files = exact_scene_files(args.observer_root, scenes, "_boxes.pkl")
    diagnostic_files = exact_scene_files(args.diagnostics_root, scenes, DIAGNOSTIC_SUFFIX)
    historical_predictions = exact_scene_files(
        args.historical_prediction_root, scenes, "_boxes.pkl"
    )
    historical_boxer = exact_scene_files(
        args.historical_boxer_root, scenes, "_boxer_lifting.jsonl"
    )
    observer_boxer = exact_scene_files(
        args.observer_boxer_root, scenes, "_boxer_lifting.jsonl"
    )
    historical_logs = exact_scene_files(args.historical_log_root, scenes, ".log")
    observer_logs = exact_scene_files(args.observer_log_root, scenes, ".log")

    report: dict[str, Any] = {
        "schema": "boxfusion.ca1m_c3_native_b6_identity_audit.v2",
        "ok": True,
        "ok_scope": "engineering_identity_and_feature_integrity_only",
        "dataset": "CA1M",
        "observer_only": True,
        "mutation_enabled": False,
        "identity_anchor_contract": "same_run_pre_native_b6_finalize",
        "historical_p1_role": "replay_drift_and_runtime_reference_only",
        "scene_list": str(args.scene_list.resolve()),
        "scene_list_sha256": sha256(args.scene_list),
        "scenes": len(scenes),
        "per_scene": {},
    }
    total_rows = 0
    total_valid = 0
    total_projectable = 0
    total_depth_counts = np.zeros(4, dtype=np.int64)
    all_valid_views: list[int] = []
    all_sampled_views: list[int] = []
    all_classified_views: list[int] = []
    all_total_samples: list[int] = []
    all_classified_samples: list[int] = []
    total_observer_seconds = 0.0
    historical_seconds = 0.0
    historical_frames = 0.0
    observer_seconds = 0.0
    observer_frames = 0.0
    historical_scene_fps: dict[str, float] = {}
    observer_scene_fps: dict[str, float] = {}
    feature_schema: tuple[str, ...] | None = None
    for scene in scenes:
        prediction, identity = compare_same_run(
            scene, anchor_files[scene], observer_files[scene]
        )
        diagnostic = audit_diagnostic(scene, diagnostic_files[scene], prediction)
        current_schema = tuple(diagnostic["feature_names"])
        if feature_schema is None:
            feature_schema = current_schema
        elif current_schema != feature_schema:
            raise ValueError(f"{scene}: feature schema/order differs across scenes")
        boxer = audit_boxer(scene, historical_boxer[scene], observer_boxer[scene])
        historical = audit_historical_prediction(
            scene, historical_predictions[scene], prediction
        )
        old_runtime = audit_runtime_log(historical_logs[scene])
        new_runtime = audit_runtime_log(observer_logs[scene])
        report["per_scene"][scene] = {
            **identity,
            **diagnostic,
            "historical_p1": historical,
            "boxer": boxer,
            "runtime": {
                "historical_p1": old_runtime,
                "native_observer": new_runtime,
                "observer_over_historical_cost_ratio": (
                    float(new_runtime["cost_seconds"] / old_runtime["cost_seconds"])
                ),
            },
        }
        total_rows += len(prediction)
        total_valid += int(diagnostic["valid_evidence_rows"])
        total_projectable += int(diagnostic["projectable_rows"])
        total_depth_counts += np.asarray(
            [diagnostic["aggregate_depth_counts"][name] for name in DEPTH_CLASS_NAMES],
            dtype=np.int64,
        )
        for row in diagnostic["row_evidence"]:
            all_valid_views.append(int(row["valid_views"]))
            all_sampled_views.append(int(row["sampled_views"]))
            all_classified_views.append(int(row["classified_views"]))
            all_total_samples.append(int(row["total_samples"]))
            all_classified_samples.append(int(row["classified_samples"]))
        total_observer_seconds += float(diagnostic["observer_seconds"])
        historical_seconds += float(old_runtime["cost_seconds"])
        historical_frames += float(old_runtime["frame_equivalent"])
        observer_seconds += float(new_runtime["cost_seconds"])
        observer_frames += float(new_runtime["frame_equivalent"])
        historical_scene_fps[scene] = float(old_runtime["average_fps"])
        observer_scene_fps[scene] = float(new_runtime["average_fps"])

    mapping_coverage = 1.0
    valid_evidence_coverage = float(total_valid / total_rows) if total_rows else 1.0
    minimum_historical_scene = min(historical_scene_fps, key=historical_scene_fps.get)
    minimum_observer_scene = min(observer_scene_fps, key=observer_scene_fps.get)
    report["summary"] = {
        "prediction_rows": total_rows,
        "mapping_rows": total_rows,
        "mapping_coverage": mapping_coverage,
        "projectable_rows": total_projectable,
        "projectable_coverage": float(total_projectable / total_rows) if total_rows else 1.0,
        "valid_evidence_rows": total_valid,
        "valid_evidence_coverage": valid_evidence_coverage,
        "feature_count": len(feature_schema or ()),
        "feature_names": list(feature_schema or ()),
        "observer_kernel_seconds": total_observer_seconds,
        "identity_scenes": len(scenes),
        "yaw_and_features_recomputed_scenes": len(scenes),
        "depth_redundancy_recomputed_scenes": len(scenes),
        "evidence_strata": {
            "aggregate_depth_counts": {
                name: int(total_depth_counts[column])
                for column, name in enumerate(DEPTH_CLASS_NAMES)
            },
            "valid_views_per_row": _distribution(np.asarray(all_valid_views)),
            "sampled_views_per_row": _distribution(np.asarray(all_sampled_views)),
            "classified_views_per_row": _distribution(np.asarray(all_classified_views)),
            "total_samples_per_row": _distribution(np.asarray(all_total_samples)),
            "classified_samples_per_row": _distribution(
                np.asarray(all_classified_samples)
            ),
            "rows_by_valid_view_count": _histogram(np.asarray(all_valid_views)),
            "rows_by_classified_view_count": _histogram(
                np.asarray(all_classified_views)
            ),
        },
        "runtime": {
            "historical_p1_total_seconds": historical_seconds,
            "native_observer_total_seconds": observer_seconds,
            "historical_p1_frame_weighted_fps": (
                historical_frames / historical_seconds if historical_seconds else 0.0
            ),
            "native_observer_frame_weighted_fps": (
                observer_frames / observer_seconds if observer_seconds else 0.0
            ),
            "observer_over_historical_p1_cost_ratio": (
                observer_seconds / historical_seconds if historical_seconds else 0.0
            ),
            "historical_p1_min_scene_fps": {
                "scene_id": minimum_historical_scene,
                "fps": historical_scene_fps[minimum_historical_scene],
            },
            "native_observer_min_scene_fps": {
                "scene_id": minimum_observer_scene,
                "fps": observer_scene_fps[minimum_observer_scene],
            },
        },
    }
    report["engineering_identity"] = {
        "ok": True,
        "same_run_byte_identity_scenes": len(scenes),
        "same_run_semantic_identity_scenes": len(scenes),
        "mapping_coverage": mapping_coverage,
        "feature_integrity_scenes": len(scenes),
        "scope": "observer_identity_and_diagnostic_integrity",
    }
    report["train_readiness"] = build_train_readiness(
        engineering_identity_ok=True,
        mapping_coverage=mapping_coverage,
        valid_evidence_coverage=valid_evidence_coverage,
        feature_integrity_ok=True,
    )
    write_json_atomic(args.output, report)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"CA-1M C3 native B6 identity audit passed: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
