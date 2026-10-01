#!/usr/bin/env python3
"""Report the observer-only P2-v2 local Mask-RGBD geometry recall.

The formal BoxFusion predictions are never modified.  For every scene this
tool first freezes the class-agnostic matches obtained by ``B6 ∪ P1 ∪ P2``.
P2-v2 candidates are then matched, in their deterministic score order, only
against ground-truth boxes that remain uncovered.  This makes the reported
incremental true positives and recall gain monotonic and isolates the value
of the new geometry stream from score-ordering side effects.

Ground truth is read only by this offline reporting tool.  The diagnostic
contract is validated fail-closed before any metric is produced.

Prediction pickle files are trusted local experiment artifacts and must not
come from an untrusted source.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boxfusion.p2_local_mask_geometry import (  # noqa: E402
    P2V2_DIAGNOSTIC_SCHEMA,
    P2V2_SOURCE,
)
from tools.report_p1_residual_recall import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    center_size_to_corners,
    load_axis_alignment,
    load_gt_boxes,
    read_scene_ids,
    score_ordered_match,
    validate_thresholds,
)
from tools.report_p2_occupancy_recall import (  # noqa: E402
    CandidateStream,
    _concatenate,
    _merge_p1_p2_unique,
    _stream,
    _validate_exact_scene_set,
    corners_to_minmax,
    load_p2_diagnostic,
    load_predictions,
    pairwise_aabb_iou,
    transform_corners,
)


REPORT_SCHEMA = "boxfusion.p2v2.local_geometry_recall_report.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROFILE = "p2v2_local_component_mask_rgbd_observer"


@dataclass(frozen=True)
class P2V2Candidates:
    corners_world: np.ndarray
    scores: np.ndarray
    candidate_ids: np.ndarray
    runtime_seconds: float
    step_count: int
    mask_observation_count: int
    mask_component_count: int
    eligible_pair_count: int
    parent_p2_checkpoint_sha256: str
    mask_provider: str


def _scalar(
    archive: Mapping[str, np.ndarray], key: str, path: Path
) -> np.ndarray:
    if key not in archive:
        raise ValueError(f"{path}: missing {key}")
    value = np.asarray(archive[key])
    if value.shape != () or value.dtype.hasobject:
        raise ValueError(f"{path}: {key} must be a non-object scalar")
    return value


def _text(
    archive: Mapping[str, np.ndarray], key: str, path: Path
) -> str:
    value = _scalar(archive, key, path).item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise ValueError(f"{path}: {key} must be text")
    return value


def _boolean(
    archive: Mapping[str, np.ndarray], key: str, path: Path
) -> bool:
    value = _scalar(archive, key, path)
    if value.dtype != np.dtype(bool):
        raise ValueError(f"{path}: {key} must be Boolean")
    return bool(value.item())


def _integer(
    archive: Mapping[str, np.ndarray], key: str, path: Path
) -> int:
    value = _scalar(archive, key, path)
    if not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{path}: {key} must be integer")
    return int(value.item())


def _config(
    archive: Mapping[str, np.ndarray], path: Path
) -> Mapping[str, Any]:
    try:
        value = json.loads(_text(archive, "p2v2_config_json", path))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: malformed p2v2_config_json") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: p2v2_config_json must encode a mapping")
    for key, expected in (
        ("enabled", True),
        ("observer_only", True),
        ("mutate", False),
        ("collect_diagnostics", True),
    ):
        if value.get(key) is not expected:
            raise ValueError(
                f"{path}: unsafe p2v2_config_json.{key}"
            )
    return value


def _one_dimensional(
    archive: Mapping[str, np.ndarray],
    key: str,
    path: Path,
    *,
    kind: str,
) -> np.ndarray:
    if key not in archive:
        raise ValueError(f"{path}: missing {key}")
    value = np.asarray(archive[key])
    if value.ndim != 1 or value.dtype.hasobject:
        raise ValueError(f"{path}: {key} must be a non-object vector")
    if kind == "integer" and not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{path}: {key} must be integer")
    if kind == "floating" and not np.issubdtype(value.dtype, np.floating):
        raise ValueError(f"{path}: {key} must be floating")
    if kind == "boolean" and value.dtype != np.dtype(bool):
        raise ValueError(f"{path}: {key} must be Boolean")
    if kind == "text" and value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{path}: {key} must be text")
    return value


def _load_step_contract(
    archive: Mapping[str, np.ndarray], path: Path
) -> tuple[float, int, int, int, int]:
    integer_keys = (
        "p2v2_step_frame_ids",
        "p2v2_step_provider_steps",
        "p2v2_step_selected_voxel_counts",
        "p2v2_step_occupancy_component_counts",
        "p2v2_step_mask_observation_counts",
        "p2v2_step_mask_component_counts",
        "p2v2_step_eligible_pair_counts",
        "p2v2_step_candidate_counts",
    )
    rows = {
        key: _one_dimensional(
            archive, key, path, kind="integer"
        ).astype(np.int64, copy=False)
        for key in integer_keys
    }
    seconds = _one_dimensional(
        archive, "p2v2_step_seconds", path, kind="floating"
    ).astype(np.float64, copy=False)
    failed = _one_dimensional(
        archive, "p2v2_step_failed", path, kind="boolean"
    )
    errors = _one_dimensional(
        archive, "p2v2_step_errors", path, kind="text"
    )
    lengths = {len(value) for value in (*rows.values(), seconds, failed, errors)}
    if len(lengths) != 1:
        raise ValueError(f"{path}: P2-v2 step arrays disagree in length")
    step_count = next(iter(lengths), 0)
    if step_count < 1:
        raise ValueError(f"{path}: P2-v2 observer never executed")
    if (
        not np.isfinite(seconds).all()
        or np.any(seconds < 0.0)
        or np.any(failed)
        or any(str(value) for value in errors.tolist())
    ):
        raise ValueError(f"{path}: incomplete or failed P2-v2 step")
    if any(np.any(value < 0) for value in rows.values()):
        raise ValueError(f"{path}: negative P2-v2 step count")

    for parent_key, child_key in (
        ("p2_step_frame_ids", "p2v2_step_frame_ids"),
        ("p2_step_provider_steps", "p2v2_step_provider_steps"),
        (
            "p2_step_selected_voxel_counts",
            "p2v2_step_selected_voxel_counts",
        ),
    ):
        if parent_key not in archive:
            raise ValueError(f"{path}: missing {parent_key}")
        parent = np.asarray(archive[parent_key])
        if (
            parent.ndim != 1
            or not np.issubdtype(parent.dtype, np.integer)
            or not np.array_equal(parent, rows[child_key])
        ):
            raise ValueError(
                f"{path}: P2/P2-v2 scheduling is not aligned"
            )
    if np.any(
        rows["p2v2_step_candidate_counts"]
        > rows["p2v2_step_eligible_pair_counts"]
    ):
        raise ValueError(f"{path}: impossible P2-v2 candidate count")
    return (
        float(np.sum(seconds, dtype=np.float64)),
        int(step_count),
        int(np.sum(rows["p2v2_step_mask_observation_counts"])),
        int(np.sum(rows["p2v2_step_mask_component_counts"])),
        int(np.sum(rows["p2v2_step_eligible_pair_counts"])),
    )


def _load_candidate_contract(
    archive: Mapping[str, np.ndarray], path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = (
        "p2v2_candidate_ids",
        "p2v2_parent_p2_candidate_ids",
        "p2v2_mask_source_ids",
        "p2v2_candidate_boxes",
        "p2v2_candidate_corners",
        "p2v2_candidate_parent_boxes",
        "p2v2_candidate_scores",
        "p2v2_candidate_parent_objectness",
        "p2v2_candidate_occupancy_scores",
        "p2v2_candidate_mask_scores",
        "p2v2_candidate_valid_depth_ratios",
        "p2v2_candidate_component_point_counts",
        "p2v2_candidate_component_voxel_counts",
        "p2v2_candidate_selected_voxels_inside",
        "p2v2_candidate_anchor_inside",
        "p2v2_candidate_parent_iou",
        "p2v2_candidate_normalized_center_distance",
        "p2v2_candidate_extent_ratios",
        "p2v2_candidate_center_shift_ratios",
        "p2v2_candidate_applied",
    )
    missing = [key for key in required if key not in archive]
    if missing:
        raise ValueError(f"{path}: missing {missing[0]}")
    ids = np.asarray(archive["p2v2_candidate_ids"])
    parents = np.asarray(archive["p2v2_parent_p2_candidate_ids"])
    masks = np.asarray(archive["p2v2_mask_source_ids"])
    boxes = np.asarray(archive["p2v2_candidate_boxes"])
    corners = np.asarray(archive["p2v2_candidate_corners"])
    parent_boxes = np.asarray(archive["p2v2_candidate_parent_boxes"])
    count = len(ids)
    if (
        ids.ndim != 1
        or ids.dtype.hasobject
        or ids.dtype.kind not in {"U", "S"}
        or len(np.unique(ids)) != count
        or parents.shape != (count,)
        or parents.dtype.hasobject
        or parents.dtype.kind not in {"U", "S"}
        or masks.shape != (count,)
        or masks.dtype.hasobject
        or masks.dtype.kind not in {"U", "S"}
        or boxes.shape != (count, 6)
        or corners.shape != (count, 8, 3)
        or parent_boxes.shape != (count, 6)
    ):
        raise ValueError(f"{path}: invalid P2-v2 candidate identity/geometry")
    vector_specs = {
        "p2v2_candidate_scores": ("floating", 0.0, 1.0),
        "p2v2_candidate_parent_objectness": ("floating", 0.0, 1.0),
        "p2v2_candidate_occupancy_scores": ("floating", 0.0, 1.0),
        "p2v2_candidate_mask_scores": ("floating", 0.0, 1.0),
        "p2v2_candidate_valid_depth_ratios": ("floating", 0.0, 1.0),
        "p2v2_candidate_parent_iou": ("floating", 0.0, 1.0),
        "p2v2_candidate_normalized_center_distance": (
            "floating",
            0.0,
            None,
        ),
        "p2v2_candidate_component_point_counts": ("integer", 1.0, None),
        "p2v2_candidate_component_voxel_counts": ("integer", 1.0, None),
        "p2v2_candidate_selected_voxels_inside": ("integer", 1.0, None),
    }
    for key, (kind, lower, upper) in vector_specs.items():
        values = _one_dimensional(archive, key, path, kind=kind)
        if (
            values.shape != (count,)
            or not np.isfinite(values).all()
            or np.any(values < lower)
            or (upper is not None and np.any(values > upper))
        ):
            raise ValueError(f"{path}: invalid {key}")
    anchors = _one_dimensional(
        archive, "p2v2_candidate_anchor_inside", path, kind="boolean"
    )
    applied = _one_dimensional(
        archive, "p2v2_candidate_applied", path, kind="boolean"
    )
    extent_ratios = np.asarray(archive["p2v2_candidate_extent_ratios"])
    center_shifts = np.asarray(
        archive["p2v2_candidate_center_shift_ratios"]
    )
    if (
        anchors.shape != (count,)
        or not np.all(anchors)
        or applied.shape != (count,)
        or np.any(applied)
        or extent_ratios.shape != (count, 3)
        or center_shifts.shape != (count, 3)
        or not np.isfinite(extent_ratios).all()
        or not np.isfinite(center_shifts).all()
        or np.any(extent_ratios <= 0.0)
        or np.any(center_shifts < 0.0)
        or not np.isfinite(boxes).all()
        or not np.isfinite(corners).all()
        or not np.isfinite(parent_boxes).all()
        or np.any(boxes[:, 3:] <= 0.0)
        or np.any(parent_boxes[:, 3:] <= 0.0)
    ):
        raise ValueError(f"{path}: invalid or unsafe P2-v2 candidates")
    expected = corners_to_minmax(center_size_to_corners(boxes))
    observed = corners_to_minmax(corners)
    if not np.allclose(expected, observed, rtol=1e-5, atol=1e-5):
        raise ValueError(f"{path}: P2-v2 box and corner aliases disagree")
    return (
        np.asarray(corners, dtype=np.float64),
        np.asarray(archive["p2v2_candidate_scores"], dtype=np.float64),
        np.asarray(ids),
    )


def load_p2v2_candidates(
    path: str | os.PathLike[str], *, expected_scene_id: str
) -> P2V2Candidates:
    diagnostic_path = Path(path)
    if not diagnostic_path.is_file():
        raise FileNotFoundError(diagnostic_path)
    with np.load(diagnostic_path, allow_pickle=False) as source:
        archive = {
            key: np.array(source[key], copy=True) for key in source.files
        }
    if any(value.dtype.hasobject for value in archive.values()):
        raise ValueError(f"{diagnostic_path}: object dtype is forbidden")
    expected_text = {
        "scene_id": expected_scene_id,
        "p2v2_schema": P2V2_DIAGNOSTIC_SCHEMA,
        "p2v2_stage": "P2V2",
        "p2v2_profile": _PROFILE,
        "p2v2_source": P2V2_SOURCE,
    }
    for key, expected in expected_text.items():
        if _text(archive, key, diagnostic_path) != expected:
            raise ValueError(f"{diagnostic_path}: invalid {key}")
    for key, expected in (
        ("p2v2_enabled", True),
        ("p2v2_observer_only", True),
        ("p2v2_uses_ground_truth", False),
        ("p2v2_reads_semantic_labels", False),
        ("p2v2_mutation_enabled", False),
        ("p2v2_complete", True),
    ):
        if _boolean(archive, key, diagnostic_path) is not expected:
            raise ValueError(f"{diagnostic_path}: unsafe {key}")
    if _integer(archive, "p2v2_applied_count", diagnostic_path) != 0:
        raise ValueError(f"{diagnostic_path}: P2-v2 mutated formal output")
    _config(archive, diagnostic_path)
    provider = _text(archive, "p2v2_mask_provider", diagnostic_path)
    if not provider:
        raise ValueError(f"{diagnostic_path}: empty mask provider")
    parent_sha = _text(
        archive, "p2v2_parent_p2_checkpoint_sha256", diagnostic_path
    ).lower()
    p2_sha = _text(archive, "p2_checkpoint_sha256", diagnostic_path).lower()
    if (
        _SHA256.fullmatch(parent_sha) is None
        or parent_sha != p2_sha
    ):
        raise ValueError(
            f"{diagnostic_path}: P2-v2 parent checkpoint mismatch"
        )
    runtime, step_count, masks, components, pairs = _load_step_contract(
        archive, diagnostic_path
    )
    corners, scores, ids = _load_candidate_contract(
        archive, diagnostic_path
    )
    return P2V2Candidates(
        corners_world=corners,
        scores=scores,
        candidate_ids=ids,
        runtime_seconds=runtime,
        step_count=step_count,
        mask_observation_count=masks,
        mask_component_count=components,
        eligible_pair_count=pairs,
        parent_p2_checkpoint_sha256=parent_sha,
        mask_provider=provider,
    )


def _baseline_stream(
    *,
    prediction_path: Path,
    diagnostic_path: Path,
    scene_id: str,
    alignment: np.ndarray,
) -> tuple[CandidateStream, P2V2Candidates, float, float]:
    predictions = load_predictions(prediction_path)
    diagnostic = load_p2_diagnostic(
        diagnostic_path, expected_scene_id=scene_id
    )
    b6 = _stream(
        corners_to_minmax(
            transform_corners(predictions.corners_world, alignment)
        ),
        predictions.scores,
        np.asarray(
            [f"b6:{index:06d}" for index in range(len(predictions.scores))],
            dtype=np.str_,
        ),
    )
    p1 = _stream(
        corners_to_minmax(
            transform_corners(diagnostic.p1.corners_world, alignment)
        ),
        diagnostic.p1.scores,
        diagnostic.p1.candidate_ids,
    )
    p2 = _stream(
        corners_to_minmax(
            transform_corners(diagnostic.p2.corners_world, alignment)
        ),
        diagnostic.p2.objectness_scores,
        diagnostic.p2.candidate_ids,
    )
    p1_p2, _ = _merge_p1_p2_unique(p1, p2)
    return (
        _concatenate((("b6", b6), ("p1p2", p1_p2))),
        load_p2v2_candidates(
            diagnostic_path, expected_scene_id=scene_id
        ),
        float(diagnostic.p1.runtime_seconds),
        float(diagnostic.p2.incremental_runtime_seconds),
    )


def _incremental(
    baseline: CandidateStream,
    candidates: CandidateStream,
    gt_boxes: np.ndarray,
    threshold: float,
) -> tuple[int, int]:
    baseline_match = score_ordered_match(
        pairwise_aabb_iou(baseline.boxes, gt_boxes),
        baseline.scores,
        threshold,
        tie_break_ids=baseline.ids,
    )
    covered = np.zeros(len(gt_boxes), dtype=bool)
    covered[baseline_match.matched_gt] = True
    novel_match = score_ordered_match(
        pairwise_aabb_iou(candidates.boxes, gt_boxes),
        candidates.scores,
        threshold,
        allowed_gt=~covered,
        tie_break_ids=candidates.ids,
    )
    return baseline_match.true_positive_count, novel_match.true_positive_count


def evaluate(
    *,
    scenes: Sequence[str],
    prediction_root: str | os.PathLike[str],
    diagnostics_root: str | os.PathLike[str],
    gt_root: str | os.PathLike[str],
    scans_root: str | os.PathLike[str],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    minimum_delta_r25_pp: float = 3.0,
    minimum_delta_r50_pp: float = 1.0,
) -> dict[str, Any]:
    thresholds = validate_thresholds(thresholds)
    threshold_keys = {f"{value:.2f}" for value in thresholds}
    if not {"0.25", "0.50"}.issubset(threshold_keys):
        raise ValueError("go/no-go requires IoU thresholds 0.25 and 0.50")
    for name, value in (
        ("minimum_delta_r25_pp", minimum_delta_r25_pp),
        ("minimum_delta_r50_pp", minimum_delta_r50_pp),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    scene_ids = tuple(str(scene) for scene in scenes)
    if not scene_ids or len(scene_ids) != len(set(scene_ids)):
        raise ValueError("scenes must be non-empty and unique")
    prediction_directory = Path(prediction_root)
    diagnostic_directory = Path(diagnostics_root)
    gt_directory = Path(gt_root)
    scans_directory = Path(scans_root)
    for role, root in (
        ("prediction", prediction_directory),
        ("diagnostics", diagnostic_directory),
        ("ground-truth", gt_directory),
        ("scans", scans_directory),
    ):
        if not root.is_dir():
            raise FileNotFoundError(f"{role} root not found: {root}")
    _validate_exact_scene_set(
        prediction_directory,
        scene_ids,
        suffix="_boxes.pkl",
        role="prediction",
    )
    _validate_exact_scene_set(
        diagnostic_directory,
        scene_ids,
        suffix="_tracks.npz",
        role="diagnostic",
    )

    totals = {
        f"{threshold:.2f}": {"baseline": 0, "incremental": 0}
        for threshold in thresholds
    }
    total_gt = 0
    candidate_count = 0
    runtime = 0.0
    p1_runtime = 0.0
    p2_runtime = 0.0
    step_count = 0
    mask_observations = 0
    mask_components = 0
    eligible_pairs = 0
    checkpoint_shas: set[str] = set()
    providers: set[str] = set()
    per_scene: dict[str, Any] = {}
    for scene_id in scene_ids:
        diagnostic_path = diagnostic_directory / f"{scene_id}_tracks.npz"
        alignment = load_axis_alignment(scans_directory, scene_id)
        baseline, p2v2, scene_p1_runtime, scene_p2_runtime = (
            _baseline_stream(
                prediction_path=(
                    prediction_directory / f"{scene_id}_boxes.pkl"
                ),
                diagnostic_path=diagnostic_path,
                scene_id=scene_id,
                alignment=alignment,
            )
        )
        candidates = _stream(
            corners_to_minmax(
                transform_corners(p2v2.corners_world, alignment)
            ),
            p2v2.scores,
            p2v2.candidate_ids,
        )
        gt_boxes = load_gt_boxes(gt_directory / f"{scene_id}_bbox.npy")
        scene_thresholds: dict[str, Any] = {}
        for threshold in thresholds:
            key = f"{threshold:.2f}"
            baseline_tp, incremental_tp = _incremental(
                baseline, candidates, gt_boxes, threshold
            )
            totals[key]["baseline"] += int(baseline_tp)
            totals[key]["incremental"] += int(incremental_tp)
            scene_thresholds[key] = {
                "baseline_true_positives": int(baseline_tp),
                "baseline_recall": float(
                    baseline_tp / max(len(gt_boxes), 1)
                ),
                "incremental_true_positives": int(incremental_tp),
                "p2v2_candidate_precision": float(
                    incremental_tp / max(len(candidates), 1)
                ),
                "recall_gain": float(
                    incremental_tp / max(len(gt_boxes), 1)
                ),
                "combined_true_positives": int(
                    baseline_tp + incremental_tp
                ),
                "combined_recall": float(
                    (baseline_tp + incremental_tp)
                    / max(len(gt_boxes), 1)
                ),
            }
        per_scene[scene_id] = {
            "ground_truth_count": int(len(gt_boxes)),
            "baseline_candidate_count": int(len(baseline)),
            "p2v2_candidate_count": int(len(candidates)),
            "p1_runtime_seconds": float(scene_p1_runtime),
            "p2_incremental_runtime_seconds": float(scene_p2_runtime),
            "p2v2_runtime_seconds": float(p2v2.runtime_seconds),
            "thresholds": scene_thresholds,
        }
        total_gt += len(gt_boxes)
        candidate_count += len(candidates)
        runtime += p2v2.runtime_seconds
        p1_runtime += scene_p1_runtime
        p2_runtime += scene_p2_runtime
        step_count += p2v2.step_count
        mask_observations += p2v2.mask_observation_count
        mask_components += p2v2.mask_component_count
        eligible_pairs += p2v2.eligible_pair_count
        checkpoint_shas.add(p2v2.parent_p2_checkpoint_sha256)
        providers.add(p2v2.mask_provider)
    if len(checkpoint_shas) != 1 or len(providers) != 1:
        raise ValueError("P2-v2 provenance changed across scenes")

    threshold_report: dict[str, Any] = {}
    for key, row in totals.items():
        baseline_tp = int(row["baseline"])
        incremental_tp = int(row["incremental"])
        gain = float(incremental_tp / max(total_gt, 1))
        threshold_report[key] = {
            "ground_truth_count": int(total_gt),
            "baseline": {
                "source": "b6_p1_p2_union",
                "true_positives": baseline_tp,
                "recall": float(baseline_tp / max(total_gt, 1)),
            },
            "p2v2_incremental": {
                "candidate_count": int(candidate_count),
                "true_positives": incremental_tp,
                "precision": float(
                    incremental_tp / max(candidate_count, 1)
                ),
                "recall_gain": gain,
                "recall_gain_percentage_points": float(100.0 * gain),
            },
            "combined": {
                "source": "b6_p1_p2_union_plus_p2v2",
                "true_positives": baseline_tp + incremental_tp,
                "recall": float(
                    (baseline_tp + incremental_tp) / max(total_gt, 1)
                ),
            },
        }
    delta_r25 = threshold_report["0.25"]["p2v2_incremental"][
        "recall_gain_percentage_points"
    ]
    delta_r50 = threshold_report["0.50"]["p2v2_incremental"][
        "recall_gain_percentage_points"
    ]
    passed = bool(
        delta_r25 >= minimum_delta_r25_pp
        and delta_r50 >= minimum_delta_r50_pp
    )
    scene_count = len(scene_ids)
    return {
        "schema": REPORT_SCHEMA,
        "matching_contract": (
            "class-agnostic; freeze stable score-ordered B6/P1/P2 matches, "
            "then stable P2-v2 score order against only uncovered GT; "
            "strict IoU > threshold; one-to-one per scene"
        ),
        "observer_only": True,
        "ground_truth_usage": "offline_evaluation_only",
        "safety": {
            "validated": True,
            "uses_ground_truth_online": False,
            "reads_semantic_labels": False,
            "mutation_enabled": False,
            "applied_count": 0,
            "parent_p2_checkpoint_sha256": next(iter(checkpoint_shas)),
            "mask_provider": next(iter(providers)),
        },
        "scene_count": int(scene_count),
        "ground_truth_count": int(total_gt),
        "candidate_count": int(candidate_count),
        "candidate_count_per_scene": float(
            candidate_count / max(scene_count, 1)
        ),
        "observer_work": {
            "step_count": int(step_count),
            "mask_observation_count": int(mask_observations),
            "mask_component_count": int(mask_components),
            "eligible_pair_count": int(eligible_pairs),
        },
        "runtime_seconds": {
            "p1": float(p1_runtime),
            "p2_incremental": float(p2_runtime),
            "baseline_p1_p2_total": float(p1_runtime + p2_runtime),
            "p2v2_incremental": float(runtime),
            "combined_p1_p2_p2v2_total": float(
                p1_runtime + p2_runtime + runtime
            ),
            "p2v2_incremental_per_scene": float(
                runtime / max(scene_count, 1)
            ),
            "p2v2_incremental_per_step": float(
                runtime / max(step_count, 1)
            ),
            "p2v2_overhead_fraction_of_p1_p2": (
                float(runtime / (p1_runtime + p2_runtime))
                if p1_runtime + p2_runtime > 0.0
                else None
            ),
        },
        "thresholds": threshold_report,
        "go_no_go": {
            "scope": (
                "fixed10" if scene_count == 10 else f"{scene_count}_scenes"
            ),
            "minimum_delta_recall_at_025_percentage_points": float(
                minimum_delta_r25_pp
            ),
            "minimum_delta_recall_at_050_percentage_points": float(
                minimum_delta_r50_pp
            ),
            "observed_delta_recall_at_025_percentage_points": float(
                delta_r25
            ),
            "observed_delta_recall_at_050_percentage_points": float(
                delta_r50
            ),
            "passed": passed,
            "decision": "GO_TO_P3" if passed else "STOP_P2V2",
        },
        "per_scene": per_scene,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--diagnostics-root", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--scans-root", required=True, type=Path)
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS
    )
    parser.add_argument(
        "--minimum-delta-r25-pp", type=float, default=3.0
    )
    parser.add_argument(
        "--minimum-delta-r50-pp", type=float, default=1.0
    )
    parser.add_argument("--output", "--output-json", dest="output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate(
        scenes=read_scene_ids(args.scene_list),
        prediction_root=args.prediction_root,
        diagnostics_root=args.diagnostics_root,
        gt_root=args.gt_root,
        scans_root=args.scans_root,
        thresholds=args.thresholds,
        minimum_delta_r25_pp=args.minimum_delta_r25_pp,
        minimum_delta_r50_pp=args.minimum_delta_r50_pp,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        os.replace(temporary, args.output)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
