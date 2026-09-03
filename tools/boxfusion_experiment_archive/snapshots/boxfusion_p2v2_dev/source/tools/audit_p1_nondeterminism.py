#!/usr/bin/env python3
"""Audit P1 against a frozen P0 run and observed P0 repeat drift.

Prediction files are trusted-local pickle artifacts.  Loading them is disabled
unless ``--trusted-local-pickles`` is supplied.  The audit separates:

* structural drift (missing/extra detections, labels, order and score rank);
* numerical drift (box corners, centers/extents, scores and matched IoU);
* metric drift parsed from the three ScanNet evaluation thresholds;
* manifest comparability and the P1 observer-only safety contract.

One or more P0 repeats form a deterministic *observed envelope*.  This is a
descriptive envelope, not a confidence interval; in particular, one repeat
cannot establish a stochastic distribution.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


SCHEMA = "boxfusion.p1.nondeterminism_audit.v1"
MANIFEST_SCHEMA = "boxfusion.p_ablation.run_manifest.v1"
P1_DIAGNOSTIC_SCHEMA = "boxfusion.p1.residual_proposal_observer.v1"
_SCENE_RE = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_EVAL_RE = re.compile(
    r"^\s*eval\s+(mAP|APrec|ARecall):\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)
_OUTPUT_KEYS = {
    "prediction_root",
    "diagnostics_root",
    "log_root",
    "evaluation_root",
}
_P1_ONLY_KEYS = {
    "stage",
    "profile",
    "p1_checkpoint",
    "p1_checkpoint_sha256",
    "p1_training_provenance",
}
_REQUIRED_DIAGNOSTIC_SCALARS: dict[str, Any] = {
    "p1_stage": "P1",
    "p1_profile": "p1_residual_proposal_observer",
    "p1_enabled": True,
    "p1_observer_only": True,
    "p1_uses_ground_truth": False,
    "p1_mutation_enabled": False,
    "p1_applied_count": 0,
    "p1_complete": True,
    "p1_class_agnostic": True,
    "p1_regression_dim": 6,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_scene_ids(path: Path) -> list[str]:
    rows = [
        row.strip()
        for row in path.read_text(encoding="utf-8").splitlines()
        if row.strip()
    ]
    if not rows:
        raise ValueError(f"{path}: scene list is empty")
    if len(rows) != len(set(rows)):
        raise ValueError(f"{path}: scene list contains duplicates")
    invalid = [row for row in rows if _SCENE_RE.fullmatch(row) is None]
    if invalid:
        raise ValueError(f"{path}: invalid scene id {invalid[0]!r}")
    return rows


def _finite_array(value: Any, *, role: str, shape_tail: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{role}: expected a non-object numeric array")
    expected_ndim = 1 + len(shape_tail)
    if array.ndim != expected_ndim or tuple(array.shape[1:]) != shape_tail:
        raise ValueError(
            f"{role}: expected shape (N,{','.join(map(str, shape_tail))}), "
            f"got {array.shape}"
        )
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{role}: contains NaN or infinity")
    return result


def _label(value: Any, role: str) -> str | int:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, bool):
        raise ValueError(f"{role}: Boolean labels are not accepted")
    if isinstance(value, (str, int)):
        return value
    raise ValueError(f"{role}: unsupported label type {type(value).__name__}")


def load_prediction(path: Path) -> dict[str, Any]:
    """Load one trusted-local BoxFusion ``*_boxes.pkl`` artifact strictly."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - explicit CLI acknowledgement
    if not isinstance(payload, (list, tuple)) or len(payload) != 1:
        raise ValueError(f"{path}: expected a one-item outer sequence")
    detections = payload[0]
    if not isinstance(detections, (list, tuple)):
        raise ValueError(f"{path}: detections must be a sequence")
    labels: list[str | int] = []
    corners_raw: list[np.ndarray] = []
    scores_raw: list[Any] = []
    for index, detection in enumerate(detections):
        if not isinstance(detection, (list, tuple)) or len(detection) != 3:
            raise ValueError(
                f"{path}: detection {index} must be "
                "[label, corners, score]"
            )
        labels.append(_label(detection[0], f"{path}: label[{index}]"))
        corner = np.asarray(detection[1])
        if (
            corner.dtype.hasobject
            or corner.shape != (8, 3)
            or not np.issubdtype(corner.dtype, np.number)
            or not np.isfinite(corner).all()
        ):
            raise ValueError(f"{path}: invalid corners at detection {index}")
        corners_raw.append(corner)
        scores_raw.append(detection[2])
    corners_array = (
        np.stack(corners_raw)
        if corners_raw
        else np.empty((0, 8, 3), dtype=np.float32)
    )
    corners = _finite_array(
        corners_array, role=f"{path}: corners", shape_tail=(8, 3)
    )
    scores_array = np.asarray(scores_raw)
    if (
        scores_array.dtype.hasobject
        or scores_array.ndim != 1
        or not np.issubdtype(scores_array.dtype, np.number)
    ):
        raise ValueError(f"{path}: scores must be a non-object numeric vector")
    scores = np.asarray(scores_array, dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"{path}: scores contain NaN or infinity")
    return {
        "labels": labels,
        "corners": corners,
        "scores": scores,
        "corner_dtype": str(corners_array.dtype),
        "score_dtype": str(scores_array.dtype),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _aabbs(corners: np.ndarray) -> np.ndarray:
    if corners.shape[0] == 0:
        return np.empty((0, 6), dtype=np.float64)
    return np.concatenate((corners.min(axis=1), corners.max(axis=1)), axis=1)


def pairwise_aabb_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or left.shape[1:] != (6,):
        raise ValueError("left AABBs must have shape (N,6)")
    if right.ndim != 2 or right.shape[1:] != (6,):
        raise ValueError("right AABBs must have shape (M,6)")
    if left.shape[0] == 0 or right.shape[0] == 0:
        return np.zeros((left.shape[0], right.shape[0]), dtype=np.float64)
    intersection_min = np.maximum(left[:, None, :3], right[None, :, :3])
    intersection_max = np.minimum(left[:, None, 3:], right[None, :, 3:])
    intersection = np.prod(
        np.maximum(intersection_max - intersection_min, 0.0), axis=2
    )
    left_volume = np.prod(np.maximum(left[:, 3:] - left[:, :3], 0.0), axis=1)
    right_volume = np.prod(
        np.maximum(right[:, 3:] - right[:, :3], 0.0), axis=1
    )
    union = left_volume[:, None] + right_volume[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


def _inversions(values: Sequence[int]) -> int:
    # Scene-level detection counts are small; this transparent implementation
    # is preferable to a harder-to-audit Fenwick tree.
    return sum(
        int(values[left] > values[right])
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )


def _score_rank_inversions(
    baseline_scores: np.ndarray,
    candidate_scores: np.ndarray,
) -> int:
    if baseline_scores.size < 2:
        return 0
    baseline_order = np.lexsort(
        (np.arange(baseline_scores.size), -baseline_scores)
    )
    candidate_order = np.lexsort(
        (np.arange(candidate_scores.size), -candidate_scores)
    )
    candidate_position = np.empty(candidate_order.size, dtype=np.int64)
    candidate_position[candidate_order] = np.arange(candidate_order.size)
    return _inversions(candidate_position[baseline_order].tolist())


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if values.size else 0.0


def _numeric_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()) if values.size else 0.0,
        "p50": _quantile(values, 0.50),
        "p90": _quantile(values, 0.90),
        "p99": _quantile(values, 0.99),
        "max": float(values.max()) if values.size else 0.0,
    }


def _canonical_corner_distances(
    baseline: np.ndarray, candidate: np.ndarray
) -> tuple[float, float]:
    distances = np.linalg.norm(
        baseline[:, None, :] - candidate[None, :, :], axis=2
    )
    rows, columns = linear_sum_assignment(distances)
    matched = distances[rows, columns]
    return float(matched.mean()), float(matched.max())


def compare_predictions(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    match_iou: float,
) -> dict[str, Any]:
    """Compare one scene, matching detections geometrically with Hungarian."""

    if not 0.0 <= match_iou <= 1.0:
        raise ValueError("match_iou must be in [0,1]")
    baseline_boxes = _aabbs(np.asarray(baseline["corners"]))
    candidate_boxes = _aabbs(np.asarray(candidate["corners"]))
    ious = pairwise_aabb_iou(baseline_boxes, candidate_boxes)
    if ious.size:
        row_ids = np.arange(ious.shape[0])[:, None]
        column_ids = np.arange(ious.shape[1])[None, :]
        tie = np.abs(row_ids - column_ids) * (
            np.finfo(np.float64).eps / max(1, ious.shape[0] + ious.shape[1])
        )
        assigned_rows, assigned_columns = linear_sum_assignment(1.0 - ious + tie)
        accepted = ious[assigned_rows, assigned_columns] > match_iou
        assigned_rows = assigned_rows[accepted]
        assigned_columns = assigned_columns[accepted]
    else:
        assigned_rows = np.empty(0, dtype=np.int64)
        assigned_columns = np.empty(0, dtype=np.int64)
    order = np.argsort(assigned_rows)
    rows = assigned_rows[order]
    columns = assigned_columns[order]
    baseline_unmatched = sorted(set(range(len(baseline_boxes))) - set(rows.tolist()))
    candidate_unmatched = sorted(
        set(range(len(candidate_boxes))) - set(columns.tolist())
    )
    label_mismatches = sum(
        baseline["labels"][left] != candidate["labels"][right]
        for left, right in zip(rows, columns)
    )
    matched_iou = (
        ious[rows, columns] if rows.size else np.empty(0, dtype=np.float64)
    )
    corner_abs = (
        np.abs(
            np.asarray(baseline["corners"])[rows]
            - np.asarray(candidate["corners"])[columns]
        ).reshape(-1)
        if rows.size
        else np.empty(0, dtype=np.float64)
    )
    centers_left = (baseline_boxes[rows, :3] + baseline_boxes[rows, 3:]) / 2.0
    centers_right = (
        candidate_boxes[columns, :3] + candidate_boxes[columns, 3:]
    ) / 2.0
    center_shift = (
        np.linalg.norm(centers_left - centers_right, axis=1)
        if rows.size
        else np.empty(0, dtype=np.float64)
    )
    extents_left = baseline_boxes[rows, 3:] - baseline_boxes[rows, :3]
    extents_right = candidate_boxes[columns, 3:] - candidate_boxes[columns, :3]
    extent_delta = (
        np.abs(extents_left - extents_right).reshape(-1)
        if rows.size
        else np.empty(0, dtype=np.float64)
    )
    score_abs = (
        np.abs(
            np.asarray(baseline["scores"])[rows]
            - np.asarray(candidate["scores"])[columns]
        )
        if rows.size
        else np.empty(0, dtype=np.float64)
    )
    canonical_means: list[float] = []
    canonical_maxima: list[float] = []
    changed_boxes = 0
    changed_scores = 0
    for left, right in zip(rows.tolist(), columns.tolist()):
        left_corners = np.asarray(baseline["corners"])[left]
        right_corners = np.asarray(candidate["corners"])[right]
        if not np.array_equal(left_corners, right_corners):
            changed_boxes += 1
        if baseline["scores"][left] != candidate["scores"][right]:
            changed_scores += 1
        mean_distance, max_distance = _canonical_corner_distances(
            left_corners, right_corners
        )
        canonical_means.append(mean_distance)
        canonical_maxima.append(max_distance)
    matched_baseline_scores = np.asarray(baseline["scores"])[rows]
    matched_candidate_scores = np.asarray(candidate["scores"])[columns]
    structure = {
        "baseline_count": len(baseline_boxes),
        "candidate_count": len(candidate_boxes),
        "count_delta": len(candidate_boxes) - len(baseline_boxes),
        "matched_count": int(rows.size),
        "baseline_missing_count": len(baseline_unmatched),
        "candidate_extra_count": len(candidate_unmatched),
        "baseline_missing_indices": baseline_unmatched,
        "candidate_extra_indices": candidate_unmatched,
        "label_mismatch_count": int(label_mismatches),
        "index_identity": bool(
            rows.size == len(baseline_boxes) == len(candidate_boxes)
            and np.array_equal(rows, columns)
        ),
        "order_inversions": _inversions(columns.tolist()),
        "score_rank_inversions": _score_rank_inversions(
            matched_baseline_scores, matched_candidate_scores
        ),
    }
    numeric = {
        "changed_box_count": changed_boxes,
        "changed_score_count": changed_scores,
        "corner_abs": _numeric_summary(corner_abs),
        "center_shift": _numeric_summary(center_shift),
        "extent_abs": _numeric_summary(extent_delta),
        "score_abs": _numeric_summary(score_abs),
        "matched_iou": {
            "mean": float(matched_iou.mean()) if matched_iou.size else 0.0,
            "p01": _quantile(matched_iou, 0.01),
            "min": float(matched_iou.min()) if matched_iou.size else 0.0,
            "loss_max": (
                float(1.0 - matched_iou.min()) if matched_iou.size else 1.0
            ),
        },
        "canonical_corner": {
            "mean": (
                float(np.mean(canonical_means)) if canonical_means else 0.0
            ),
            "max": max(canonical_maxima, default=0.0),
        },
    }
    return {
        "bit_exact_file": baseline["sha256"] == candidate["sha256"],
        "baseline_sha256": baseline["sha256"],
        "candidate_sha256": candidate["sha256"],
        "structure": structure,
        "numeric": numeric,
        "_arrays": {
            "corner_abs": corner_abs,
            "center_shift": center_shift,
            "extent_abs": extent_delta,
            "score_abs": score_abs,
            "matched_iou": matched_iou,
        },
    }


def _public_scene_report(report: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "_arrays"}


def compare_prediction_roots(
    *,
    scenes: Sequence[str],
    baseline_root: Path,
    candidate_root: Path,
    match_iou: float,
) -> dict[str, Any]:
    expected = {f"{scene}_boxes.pkl" for scene in scenes}
    for role, root in (
        ("baseline", baseline_root),
        ("candidate", candidate_root),
    ):
        if not root.is_dir():
            raise FileNotFoundError(f"{role} prediction root: {root}")
        actual = {
            path.name
            for path in root.glob("scene*_boxes.pkl")
            if path.is_file()
        }
        if actual != expected:
            raise ValueError(
                f"{role} prediction set mismatch: "
                f"missing={sorted(expected - actual)[:8]}, "
                f"extra={sorted(actual - expected)[:8]}"
            )
    scene_reports: dict[str, dict[str, Any]] = {}
    arrays: dict[str, list[np.ndarray]] = {
        "corner_abs": [],
        "center_shift": [],
        "extent_abs": [],
        "score_abs": [],
        "matched_iou": [],
    }
    totals = {
        "baseline_count": 0,
        "candidate_count": 0,
        "matched_count": 0,
        "baseline_missing_count": 0,
        "candidate_extra_count": 0,
        "label_mismatch_count": 0,
        "order_inversions": 0,
        "score_rank_inversions": 0,
        "changed_box_count": 0,
        "changed_score_count": 0,
    }
    bit_exact_files = 0
    for scene in scenes:
        baseline = load_prediction(baseline_root / f"{scene}_boxes.pkl")
        candidate = load_prediction(candidate_root / f"{scene}_boxes.pkl")
        comparison = compare_predictions(
            baseline, candidate, match_iou=match_iou
        )
        scene_reports[scene] = _public_scene_report(comparison)
        bit_exact_files += int(comparison["bit_exact_file"])
        for key in arrays:
            arrays[key].append(comparison["_arrays"][key])
        structure = comparison["structure"]
        numeric = comparison["numeric"]
        for key in (
            "baseline_count",
            "candidate_count",
            "matched_count",
            "baseline_missing_count",
            "candidate_extra_count",
            "label_mismatch_count",
            "order_inversions",
            "score_rank_inversions",
        ):
            totals[key] += int(structure[key])
        totals["changed_box_count"] += int(numeric["changed_box_count"])
        totals["changed_score_count"] += int(numeric["changed_score_count"])
    joined = {
        key: (
            np.concatenate(values)
            if values and any(value.size for value in values)
            else np.empty(0, dtype=np.float64)
        )
        for key, values in arrays.items()
    }
    aggregate = {
        **totals,
        "scene_count": len(scenes),
        "bit_exact_file_count": bit_exact_files,
        "bit_exact_all_files": bit_exact_files == len(scenes),
        "corner_abs": _numeric_summary(joined["corner_abs"]),
        "center_shift": _numeric_summary(joined["center_shift"]),
        "extent_abs": _numeric_summary(joined["extent_abs"]),
        "score_abs": _numeric_summary(joined["score_abs"]),
        "matched_iou": {
            "mean": (
                float(joined["matched_iou"].mean())
                if joined["matched_iou"].size
                else 0.0
            ),
            "p01": _quantile(joined["matched_iou"], 0.01),
            "min": (
                float(joined["matched_iou"].min())
                if joined["matched_iou"].size
                else 0.0
            ),
            "loss_max": (
                float(1.0 - joined["matched_iou"].min())
                if joined["matched_iou"].size
                else 1.0
            ),
        },
    }
    aggregate["structure_identical"] = all(
        aggregate[key] == 0
        for key in (
            "baseline_missing_count",
            "candidate_extra_count",
            "label_mismatch_count",
            "order_inversions",
            "score_rank_inversions",
        )
    )
    return {
        "baseline_root": str(baseline_root.resolve()),
        "candidate_root": str(candidate_root.resolve()),
        "match_iou_strictly_greater_than": match_iou,
        "aggregate": aggregate,
        "scenes": scene_reports,
    }


def parse_eval_log(
    path: Path, iou_thresholds: Sequence[float]
) -> dict[str, dict[str, float]]:
    entries: list[tuple[str, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _EVAL_RE.match(line)
        if match:
            value = float(match.group(2))
            if not math.isfinite(value):
                raise ValueError(f"{path}: non-finite evaluation metric")
            entries.append((match.group(1), value))
    expected_names = ["mAP", "APrec", "ARecall"] * len(iou_thresholds)
    if [name for name, _ in entries] != expected_names:
        raise ValueError(
            f"{path}: expected {len(expected_names)} ordered evaluation "
            f"lines, got {[name for name, _ in entries]!r}"
        )
    report: dict[str, dict[str, float]] = {}
    for index, threshold in enumerate(iou_thresholds):
        rows = entries[index * 3 : index * 3 + 3]
        report[f"{threshold:.2f}"] = {
            name: value for name, value in rows
        }
    return report


def _metric_delta(
    baseline: Mapping[str, Mapping[str, float]],
    candidate: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    if baseline.keys() != candidate.keys():
        raise ValueError("evaluation thresholds differ")
    return {
        threshold: {
            name: float(candidate[threshold][name] - baseline[threshold][name])
            for name in ("mAP", "APrec", "ARecall")
        }
        for threshold in baseline
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def _without(mapping: Mapping[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    result = copy.deepcopy(dict(mapping))
    for key in keys:
        result.pop(key, None)
    return result


def _first_difference(
    left: Any, right: Any, path: str = "$"
) -> str | None:
    if type(left) is not type(right):
        return f"{path}: type {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return f"{path}: keys differ"
        for key in left:
            result = _first_difference(left[key], right[key], f"{path}.{key}")
            if result is not None:
                return result
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: lengths differ"
        for index, (a, b) in enumerate(zip(left, right)):
            result = _first_difference(a, b, f"{path}[{index}]")
            if result is not None:
                return result
        return None
    return None if left == right else f"{path}: {left!r} != {right!r}"


def _path_matches(value: Any, expected: Path) -> bool:
    return isinstance(value, str) and Path(value).resolve() == expected.resolve()


def compare_manifests(
    *,
    p0_manifest_path: Path,
    repeat_manifest_paths: Sequence[Path],
    p1_manifest_path: Path,
    p0_root: Path,
    repeat_roots: Sequence[Path],
    p1_root: Path,
    p1_diagnostics_root: Path,
    scene_list: Path | None = None,
    p0_eval_log: Path | None = None,
    repeat_eval_logs: Sequence[Path] | None = None,
    p1_eval_log: Path | None = None,
) -> dict[str, Any]:
    p0 = _load_json(p0_manifest_path)
    repeats = [_load_json(path) for path in repeat_manifest_paths]
    p1 = _load_json(p1_manifest_path)
    issues: list[str] = []
    for role, manifest in [
        ("P0", p0),
        *[(f"P0 repeat {index + 1}", value) for index, value in enumerate(repeats)],
        ("P1", p1),
    ]:
        if manifest.get("schema") != MANIFEST_SCHEMA:
            issues.append(f"{role}: unexpected manifest schema")
        if scene_list is not None:
            if manifest.get("scene_count") != len(read_scene_ids(scene_list)):
                issues.append(f"{role}: scene_count differs from scene list")
            if manifest.get("scene_list_sha256") != _sha256(scene_list):
                issues.append(f"{role}: scene_list_sha256 differs from input")
    if p0.get("stage") != "P0" or p0.get("profile") != "p0_frozen_b6":
        issues.append("P0: non-canonical stage/profile")
    if not _path_matches(p0.get("prediction_root"), p0_root):
        issues.append("P0: prediction_root does not match CLI input")
    if p0_eval_log is not None and not _path_matches(
        str(p0_eval_log.resolve()),
        Path(str(p0.get("log_root", ""))) / "eval_stdout.log",
    ):
        issues.append("P0: evaluation log is not the manifest-bound log")
    p0_effective = _without(p0, _OUTPUT_KEYS)
    for index, (manifest, root) in enumerate(zip(repeats, repeat_roots)):
        if (
            manifest.get("stage") != "P0"
            or manifest.get("profile") != "p0_frozen_b6"
        ):
            issues.append(f"P0 repeat {index + 1}: non-canonical stage/profile")
        if not _path_matches(manifest.get("prediction_root"), root):
            issues.append(
                f"P0 repeat {index + 1}: prediction_root does not match CLI input"
            )
        if repeat_eval_logs is not None and not _path_matches(
            str(repeat_eval_logs[index].resolve()),
            Path(str(manifest.get("log_root", ""))) / "eval_stdout.log",
        ):
            issues.append(
                f"P0 repeat {index + 1}: evaluation log is not "
                "the manifest-bound log"
            )
        difference = _first_difference(
            p0_effective, _without(manifest, _OUTPUT_KEYS)
        )
        if difference is not None:
            issues.append(f"P0 repeat {index + 1}: effective manifest {difference}")
    if (
        p1.get("stage") != "P1"
        or p1.get("profile") != "p1_residual_proposal_observer"
    ):
        issues.append("P1: non-canonical stage/profile")
    if not _path_matches(p1.get("prediction_root"), p1_root):
        issues.append("P1: prediction_root does not match CLI input")
    if not _path_matches(p1.get("diagnostics_root"), p1_diagnostics_root):
        issues.append("P1: diagnostics_root does not match CLI input")
    if p1_eval_log is not None and not _path_matches(
        str(p1_eval_log.resolve()),
        Path(str(p1.get("log_root", ""))) / "eval_stdout.log",
    ):
        issues.append("P1: evaluation log is not the manifest-bound log")
    difference = _first_difference(
        _without(p0, _OUTPUT_KEYS | _P1_ONLY_KEYS),
        _without(p1, _OUTPUT_KEYS | _P1_ONLY_KEYS),
    )
    if difference is not None:
        issues.append(f"P1: frozen B6 manifest fields differ: {difference}")
    checkpoint_sha = p1.get("p1_checkpoint_sha256")
    if not isinstance(checkpoint_sha, str) or _SHA_RE.fullmatch(
        checkpoint_sha.lower()
    ) is None:
        issues.append("P1: invalid checkpoint SHA256")
    provenance = p1.get("p1_training_provenance")
    if not isinstance(provenance, dict):
        issues.append("P1: missing train-only provenance")
    else:
        if provenance.get("b6_checkpoint_sha256") != p0.get(
            "b6_checkpoint_sha256"
        ):
            issues.append("P1: provenance B6 checkpoint differs from P0")
        if provenance.get("forbidden_overlap", []) != []:
            issues.append("P1: training provenance reports forbidden overlap")
    return {
        "ok": not issues,
        "issues": issues,
        "p0_manifest_sha256": _sha256(p0_manifest_path),
        "p0_repeat_manifest_sha256": [
            _sha256(path) for path in repeat_manifest_paths
        ],
        "p1_manifest_sha256": _sha256(p1_manifest_path),
        "p1_checkpoint_sha256": checkpoint_sha,
        "_p1_manifest": p1,
    }


def _np_scalar(archive: Any, key: str, path: Path) -> Any:
    if key not in archive.files:
        raise ValueError(f"{path}: missing {key}")
    value = np.asarray(archive[key])
    if value.shape != () or value.dtype.hasobject:
        raise ValueError(f"{path}: {key} must be a non-object scalar")
    result = value.item()
    if isinstance(result, bytes):
        result = result.decode("utf-8")
    return result


def audit_p1_diagnostics(
    *,
    scenes: Sequence[str],
    diagnostics_root: Path,
    expected_checkpoint_sha256: str | None,
) -> dict[str, Any]:
    expected = {f"{scene}_tracks.npz" for scene in scenes}
    if not diagnostics_root.is_dir():
        raise FileNotFoundError(diagnostics_root)
    actual = {
        path.name
        for path in diagnostics_root.glob("scene*_tracks.npz")
        if path.is_file()
    }
    issues: list[str] = []
    if actual != expected:
        issues.append(
            "diagnostic set mismatch: "
            f"missing={sorted(expected - actual)[:8]}, "
            f"extra={sorted(actual - expected)[:8]}"
        )
    checkpoint_hashes: set[str] = set()
    voxel_count = 0
    candidate_count = 0
    for scene in scenes:
        path = diagnostics_root / f"{scene}_tracks.npz"
        if not path.is_file():
            continue
        try:
            with np.load(path, allow_pickle=False) as archive:
                for name in archive.files:
                    if np.asarray(archive[name]).dtype.hasobject:
                        raise ValueError(f"{path}: object array {name} is forbidden")
                if _np_scalar(archive, "scene_id", path) != scene:
                    raise ValueError(f"{path}: scene_id mismatch")
                if (
                    _np_scalar(archive, "p1_schema", path)
                    != P1_DIAGNOSTIC_SCHEMA
                ):
                    raise ValueError(f"{path}: unexpected p1_schema")
                for key, expected_value in _REQUIRED_DIAGNOSTIC_SCALARS.items():
                    value = _np_scalar(archive, key, path)
                    if isinstance(expected_value, bool):
                        valid = isinstance(value, (bool, np.bool_)) and bool(
                            value
                        ) is expected_value
                    elif isinstance(expected_value, int):
                        valid = (
                            isinstance(value, (int, np.integer))
                            and not isinstance(value, (bool, np.bool_))
                            and int(value) == expected_value
                        )
                    else:
                        valid = value == expected_value
                    if not valid:
                        raise ValueError(
                            f"{path}: {key}={value!r}, expected "
                            f"{expected_value!r}"
                        )
                checkpoint = _np_scalar(
                    archive, "p1_checkpoint_sha256", path
                )
                if not isinstance(checkpoint, str) or _SHA_RE.fullmatch(
                    checkpoint.lower()
                ) is None:
                    raise ValueError(f"{path}: invalid checkpoint SHA256")
                checkpoint_hashes.add(checkpoint.lower())
                features = np.asarray(archive["p1_voxel_features"])
                centers = np.asarray(archive["p1_voxel_centers"])
                offsets = np.asarray(archive["p1_voxel_offsets"])
                feature_names = np.asarray(archive["p1_feature_names"])
                if (
                    features.ndim != 2
                    or feature_names.ndim != 1
                    or feature_names.dtype.hasobject
                    or feature_names.shape[0] != features.shape[1]
                    or centers.shape != (features.shape[0], 3)
                    or offsets.ndim != 1
                    or offsets.size == 0
                    or not np.issubdtype(offsets.dtype, np.integer)
                    or offsets[0] != 0
                    or offsets[-1] != features.shape[0]
                    or np.any(np.diff(offsets) < 0)
                ):
                    raise ValueError(f"{path}: invalid voxel array contract")
                if not (
                    np.all(np.isfinite(features))
                    and np.all(np.isfinite(centers))
                ):
                    raise ValueError(f"{path}: non-finite voxel diagnostics")
                voxel_count += int(features.shape[0])
                for applied_name in ("p1_applied", "p1_candidate_applied"):
                    if applied_name in archive.files:
                        applied = np.asarray(archive[applied_name])
                        if applied.dtype != np.bool_ or np.any(applied):
                            raise ValueError(
                                f"{path}: unsafe {applied_name} values"
                            )
                if "p1_candidate_boxes" in archive.files:
                    boxes = np.asarray(archive["p1_candidate_boxes"])
                    if boxes.ndim != 2 or boxes.shape[1] != 6:
                        raise ValueError(
                            f"{path}: p1_candidate_boxes must be (N,6)"
                        )
                    if not np.all(np.isfinite(boxes)):
                        raise ValueError(
                            f"{path}: non-finite candidate boxes"
                        )
                    candidate_count += int(boxes.shape[0])
                    if "p1_candidate_scores" in archive.files:
                        scores = np.asarray(archive["p1_candidate_scores"])
                        if scores.shape != (boxes.shape[0],) or not np.all(
                            np.isfinite(scores)
                        ):
                            raise ValueError(
                                f"{path}: invalid candidate scores"
                            )
        except (KeyError, OSError, ValueError) as error:
            issues.append(str(error))
    if len(checkpoint_hashes) != 1:
        issues.append(
            f"P1 diagnostics reference {len(checkpoint_hashes)} checkpoints"
        )
    actual_checkpoint = (
        next(iter(checkpoint_hashes)) if len(checkpoint_hashes) == 1 else None
    )
    if (
        expected_checkpoint_sha256 is not None
        and actual_checkpoint != expected_checkpoint_sha256.lower()
    ):
        issues.append("P1 diagnostic checkpoint differs from run manifest")
    return {
        "ok": not issues,
        "issues": issues,
        "scene_count": len(scenes),
        "voxel_count": voxel_count,
        "candidate_count": candidate_count,
        "checkpoint_sha256": actual_checkpoint,
    }


def _envelope_metrics(
    comparison: Mapping[str, Any],
    ap_delta: Mapping[str, Mapping[str, float]] | None,
) -> dict[str, float]:
    aggregate = comparison["aggregate"]
    metrics = {
        "baseline_missing_count": float(aggregate["baseline_missing_count"]),
        "candidate_extra_count": float(aggregate["candidate_extra_count"]),
        "label_mismatch_count": float(aggregate["label_mismatch_count"]),
        "order_inversions": float(aggregate["order_inversions"]),
        "score_rank_inversions": float(aggregate["score_rank_inversions"]),
        "corner_abs_p99": float(aggregate["corner_abs"]["p99"]),
        "corner_abs_max": float(aggregate["corner_abs"]["max"]),
        "center_shift_max": float(aggregate["center_shift"]["max"]),
        "extent_abs_max": float(aggregate["extent_abs"]["max"]),
        "score_abs_p99": float(aggregate["score_abs"]["p99"]),
        "score_abs_max": float(aggregate["score_abs"]["max"]),
        "matched_iou_loss_max": float(aggregate["matched_iou"]["loss_max"]),
    }
    if ap_delta is not None:
        for threshold, rows in ap_delta.items():
            for metric, value in rows.items():
                metrics[
                    f"abs_{metric}_delta_at_{threshold}"
                ] = abs(float(value))
    return metrics


def _scene_envelope_metrics(scene: Mapping[str, Any]) -> dict[str, float]:
    structure = scene["structure"]
    numeric = scene["numeric"]
    return {
        "baseline_missing_count": float(structure["baseline_missing_count"]),
        "candidate_extra_count": float(structure["candidate_extra_count"]),
        "label_mismatch_count": float(structure["label_mismatch_count"]),
        "order_inversions": float(structure["order_inversions"]),
        "score_rank_inversions": float(structure["score_rank_inversions"]),
        "corner_abs_p99": float(numeric["corner_abs"]["p99"]),
        "corner_abs_max": float(numeric["corner_abs"]["max"]),
        "center_shift_max": float(numeric["center_shift"]["max"]),
        "extent_abs_max": float(numeric["extent_abs"]["max"]),
        "score_abs_p99": float(numeric["score_abs"]["p99"]),
        "score_abs_max": float(numeric["score_abs"]["max"]),
        "matched_iou_loss_max": float(numeric["matched_iou"]["loss_max"]),
    }


def _compare_to_envelope(
    observed: Mapping[str, float],
    repeats: Sequence[Mapping[str, float]],
    *,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    envelope = {
        key: max(float(repeat[key]) for repeat in repeats)
        for key in observed
    }
    rows: dict[str, Any] = {}
    for key, value in observed.items():
        limit = envelope[key]
        rows[key] = {
            "p1": float(value),
            "p0_repeat_envelope": limit,
            "within": float(value) <= limit + tolerance,
            "ratio": (
                float(value) / limit
                if limit > 0.0
                else (0.0 if float(value) == 0.0 else None)
            ),
        }
    return {
        "within_all": all(row["within"] for row in rows.values()),
        "metrics": rows,
    }


def audit_nondeterminism(
    *,
    scene_list: Path,
    p0_root: Path,
    p0_manifest: Path,
    p0_eval_log: Path,
    p0_repeats: Sequence[tuple[Path, Path, Path]],
    p1_root: Path,
    p1_diagnostics_root: Path,
    p1_manifest: Path,
    p1_eval_log: Path,
    match_iou: float = 0.25,
    iou_thresholds: Sequence[float] = (0.15, 0.25, 0.50),
) -> dict[str, Any]:
    if not p0_repeats:
        raise ValueError("at least one P0 repeat is required")
    scenes = read_scene_ids(scene_list)
    repeat_roots = [row[0] for row in p0_repeats]
    manifest_report = compare_manifests(
        p0_manifest_path=p0_manifest,
        repeat_manifest_paths=[row[1] for row in p0_repeats],
        p1_manifest_path=p1_manifest,
        p0_root=p0_root,
        repeat_roots=repeat_roots,
        p1_root=p1_root,
        p1_diagnostics_root=p1_diagnostics_root,
        scene_list=scene_list,
        p0_eval_log=p0_eval_log,
        repeat_eval_logs=[row[2] for row in p0_repeats],
        p1_eval_log=p1_eval_log,
    )
    p1_manifest_value = manifest_report.pop("_p1_manifest")
    diagnostics_report = audit_p1_diagnostics(
        scenes=scenes,
        diagnostics_root=p1_diagnostics_root,
        expected_checkpoint_sha256=p1_manifest_value.get(
            "p1_checkpoint_sha256"
        ),
    )
    baseline_eval = parse_eval_log(p0_eval_log, iou_thresholds)
    p1_eval = parse_eval_log(p1_eval_log, iou_thresholds)
    p1_eval_delta = _metric_delta(baseline_eval, p1_eval)
    p1_comparison = compare_prediction_roots(
        scenes=scenes,
        baseline_root=p0_root,
        candidate_root=p1_root,
        match_iou=match_iou,
    )
    repeat_reports: list[dict[str, Any]] = []
    repeat_metric_rows: list[dict[str, float]] = []
    repeat_scene_metric_rows: dict[str, list[dict[str, float]]] = {
        scene: [] for scene in scenes
    }
    for index, (root, manifest_path, eval_log) in enumerate(p0_repeats):
        comparison = compare_prediction_roots(
            scenes=scenes,
            baseline_root=p0_root,
            candidate_root=root,
            match_iou=match_iou,
        )
        evaluation = parse_eval_log(eval_log, iou_thresholds)
        evaluation_delta = _metric_delta(baseline_eval, evaluation)
        repeat_reports.append(
            {
                "repeat_index": index + 1,
                "manifest": str(manifest_path.resolve()),
                "evaluation": evaluation,
                "evaluation_delta_from_p0": evaluation_delta,
                "prediction_comparison": comparison,
            }
        )
        repeat_metric_rows.append(
            _envelope_metrics(comparison, evaluation_delta)
        )
        for scene in scenes:
            repeat_scene_metric_rows[scene].append(
                _scene_envelope_metrics(comparison["scenes"][scene])
            )
    p1_metrics = _envelope_metrics(p1_comparison, p1_eval_delta)
    aggregate_envelope = _compare_to_envelope(
        p1_metrics, repeat_metric_rows
    )
    per_scene_envelope = {
        scene: _compare_to_envelope(
            _scene_envelope_metrics(p1_comparison["scenes"][scene]),
            repeat_scene_metric_rows[scene],
        )
        for scene in scenes
    }
    metric_identical = all(
        value == 0.0
        for threshold in p1_eval_delta.values()
        for value in threshold.values()
    )
    bit_exact = bool(p1_comparison["aggregate"]["bit_exact_all_files"])
    structure_identical = bool(
        p1_comparison["aggregate"]["structure_identical"]
    )
    comparability_ok = bool(manifest_report["ok"])
    safety_ok = bool(diagnostics_report["ok"])
    if not comparability_ok or not safety_ok:
        verdict = "unsafe_or_incomparable"
    elif bit_exact and metric_identical:
        verdict = "bit_exact"
    elif (
        structure_identical
        and metric_identical
        and aggregate_envelope["within_all"]
    ):
        verdict = "metric_identical_consistent_with_observed_p0_drift"
    elif aggregate_envelope["within_all"]:
        verdict = "within_observed_p0_drift_but_metrics_or_structure_changed"
    else:
        verdict = "exceeds_observed_p0_drift"
    return {
        "schema": SCHEMA,
        "ok": comparability_ok and safety_ok,
        "verdict": verdict,
        "scene_list": str(scene_list.resolve()),
        "scene_list_sha256": _sha256(scene_list),
        "scene_count": len(scenes),
        "matching": {
            "method": "Hungarian on axis-aligned 3D IoU",
            "acceptance": f"IoU > {match_iou}",
            "match_iou": match_iou,
        },
        "evidence_limits": {
            "p0_repeat_count": len(p0_repeats),
            "is_statistical_confidence_interval": False,
            "single_repeat_warning": (
                "One P0 repeat gives only an observed drift envelope; it "
                "cannot estimate a nondeterminism distribution."
                if len(p0_repeats) == 1
                else None
            ),
        },
        "comparability": manifest_report,
        "p1_observer_safety": diagnostics_report,
        "evaluation": {
            "p0": baseline_eval,
            "p1": p1_eval,
            "p1_delta_from_p0": p1_eval_delta,
            "metric_identical": metric_identical,
        },
        "p1_prediction_comparison": p1_comparison,
        "p0_repeats": repeat_reports,
        "observed_p0_drift_envelope": {
            "aggregate": aggregate_envelope,
            "per_scene": per_scene_envelope,
            "per_scene_within_count": sum(
                report["within_all"] for report in per_scene_envelope.values()
            ),
        },
        "conclusions": {
            "bit_exact": bit_exact,
            "structure_identical": structure_identical,
            "metric_identical": metric_identical,
            "within_observed_aggregate_p0_drift": aggregate_envelope[
                "within_all"
            ],
            "comparability_ok": comparability_ok,
            "observer_safety_ok": safety_ok,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", required=True, type=Path)
    parser.add_argument("--p0-root", required=True, type=Path)
    parser.add_argument("--p0-manifest", required=True, type=Path)
    parser.add_argument("--p0-eval-log", required=True, type=Path)
    parser.add_argument(
        "--p0-repeat",
        required=True,
        action="append",
        nargs=3,
        metavar=("PREDICTION_ROOT", "MANIFEST", "EVAL_LOG"),
        help="repeatable P0 repeat triplet",
    )
    parser.add_argument("--p1-root", required=True, type=Path)
    parser.add_argument("--p1-diagnostics-root", required=True, type=Path)
    parser.add_argument("--p1-manifest", required=True, type=Path)
    parser.add_argument("--p1-eval-log", required=True, type=Path)
    parser.add_argument("--match-iou", type=float, default=0.25)
    parser.add_argument(
        "--iou-thresholds",
        type=float,
        nargs="+",
        default=(0.15, 0.25, 0.50),
    )
    parser.add_argument(
        "--trusted-local-pickles",
        action="store_true",
        help="acknowledge that prediction pickle files can execute code",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write JSON here; without this flag the audit is read-only",
    )
    parser.add_argument("--require-within-envelope", action="store_true")
    parser.add_argument("--require-bit-exact", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.trusted_local_pickles:
        parser.error(
            "refusing executable pickle input without "
            "--trusted-local-pickles"
        )
    report = audit_nondeterminism(
        scene_list=args.scene_list,
        p0_root=args.p0_root,
        p0_manifest=args.p0_manifest,
        p0_eval_log=args.p0_eval_log,
        p0_repeats=[
            (Path(root), Path(manifest), Path(log))
            for root, manifest, log in args.p0_repeat
        ],
        p1_root=args.p1_root,
        p1_diagnostics_root=args.p1_diagnostics_root,
        p1_manifest=args.p1_manifest,
        p1_eval_log=args.p1_eval_log,
        match_iou=args.match_iou,
        iou_thresholds=args.iou_thresholds,
    )
    rendered = json.dumps(
        report, indent=2, sort_keys=True, allow_nan=False
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["ok"]:
        return 2
    if (
        args.require_within_envelope
        and not report["conclusions"]["within_observed_aggregate_p0_drift"]
    ):
        return 3
    if args.require_bit_exact and not report["conclusions"]["bit_exact"]:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
