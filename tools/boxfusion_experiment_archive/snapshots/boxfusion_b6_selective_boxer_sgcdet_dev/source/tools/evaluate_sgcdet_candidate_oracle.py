#!/usr/bin/env python3
"""Offline GT oracle for one frozen G0 + SGCDet active run.

The online sparse-refiner serializes its local box, local-frame transform, and
bounded residual for every valid prediction.  Rejected rows keep the original
geometry in the exported pickle, so this tool reconstructs their *proposed*
geometry with the exact runtime equations before measuring it against ScanNet
ground truth.

This is a diagnostic only.  Ground truth is never imported by inference and
the oracle selections produced here must not be reported as deployable model
results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.analyze_fused_oracle import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    center_size_to_minmax,
    corners_to_minmax,
    load_axis_alignment,
    maximum_matches,
    pairwise_aabb_iou,
    read_scene_ids,
    transform_corners,
)


SCHEMA = "boxfusion.sgcdet_candidate_oracle.v1"
GEOMETRY_SCHEMA = "boxfusion.full_output_geometry_prepost.v1"
PAIR_SCHEMA = "sgcdet_observer_active_pair_v1"
SPARSE_SCHEMA = "boxfusion.sgcdet_local_sparse_refiner"
FROZEN_CHECKPOINT_SHA256 = (
    "beda774fc3b8f384b408a14388d6b115704e5039b7a110a187760ac9cfd6d182"
)
SIGNS = np.asarray(
    (
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ),
    dtype=np.float64,
)


@dataclass(frozen=True)
class SceneData:
    scene_id: str
    labels: tuple[int, ...]
    scores: np.ndarray
    original_corners: np.ndarray
    active_corners: np.ndarray
    candidate_corners: np.ndarray
    candidate_valid: np.ndarray
    candidate_accepted: np.ndarray
    candidate_reason: tuple[str, ...]
    candidate_iou_prediction: np.ndarray
    improvement_prediction: np.ndarray
    uncertainty_prediction: np.ndarray
    gt_minmax: np.ndarray
    axis_alignment: np.ndarray


@dataclass(frozen=True)
class CandidateRef:
    scene_id: str
    row_index: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint_decoder(
    path: Path, expected_sha256: str | None
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha256 = _sha256(path)
    expected_sha256 = expected_sha256.lower() if expected_sha256 else None
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(
            f"{path}: checkpoint SHA256 {actual_sha256} != {expected_sha256}"
        )
    try:
        import torch
    except ImportError as error:  # pragma: no cover - production environment.
        raise ImportError("reading the frozen checkpoint requires PyTorch") from error
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - old PyTorch compatibility.
        payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema") != SPARSE_SCHEMA:
        raise ValueError(f"{path}: unexpected sparse-refiner checkpoint schema")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError(f"{path}: checkpoint config is missing")
    required = (
        "max_center_fraction",
        "max_log_dimension_residual",
        "minimum_dimension",
    )
    decoder: dict[str, float] = {}
    for key in required:
        value = config.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{path}: invalid checkpoint config.{key}")
        decoder[key] = float(value)
        if decoder[key] <= 0.0:
            raise ValueError(f"{path}: checkpoint config.{key} must be positive")
    metadata = payload.get("metadata")
    return {
        "path": str(path.resolve()),
        "sha256": actual_sha256,
        "schema": payload["schema"],
        "coordinate_frame": payload.get("coordinate_frame"),
        "reference": payload.get("reference"),
        "decoder": decoder,
        "training_metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
    }


def _scalar_text(value: Any, *, name: str, path: Path) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.hasobject:
        raise ValueError(f"{path}: {name} must be a scalar string")
    result = array.item()
    if isinstance(result, bytes):
        result = result.decode("utf-8")
    if not isinstance(result, str):
        raise ValueError(f"{path}: {name} must be a scalar string")
    return result


def _load_prediction(path: Path) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(f"{path}: expected a one-scene outer list")
    rows = payload[0]
    if not isinstance(rows, list):
        raise ValueError(f"{path}: prediction rows must be a list")
    labels: list[int] = []
    corners: list[np.ndarray] = []
    scores: list[float] = []
    for index, row in enumerate(rows):
        if not isinstance(row, tuple) or len(row) != 3:
            raise ValueError(f"{path}: invalid prediction row {index}")
        label, box_corners, score = row
        corner_array = np.asarray(box_corners)
        if corner_array.shape != (8, 3) or corner_array.dtype != np.float32:
            raise ValueError(
                f"{path}: row {index} corners must be float32 [8,3]"
            )
        if not np.isfinite(corner_array).all() or not np.isfinite(score):
            raise ValueError(f"{path}: row {index} contains non-finite values")
        labels.append(int(label))
        corners.append(corner_array.copy())
        scores.append(float(score))
    corner_result = (
        np.stack(corners)
        if corners
        else np.empty((0, 8, 3), dtype=np.float32)
    )
    return (
        tuple(labels),
        corner_result,
        np.asarray(scores, dtype=np.float64),
    )


def reconstruct_candidates(
    local_boxes: np.ndarray,
    center_residual: np.ndarray,
    log_dimension_residual: np.ndarray,
    frame_center: np.ndarray,
    frame_basis: np.ndarray,
    valid: np.ndarray,
    *,
    max_center_fraction: float,
    max_log_dimension_residual: float,
    minimum_dimension: float,
) -> np.ndarray:
    """Reproduce ``apply_sgcdet_sparse_residual_numpy`` + world decoding."""

    local = np.asarray(local_boxes, dtype=np.float64)
    center_delta = np.asarray(center_residual, dtype=np.float64)
    log_delta = np.asarray(log_dimension_residual, dtype=np.float64)
    origin = np.asarray(frame_center, dtype=np.float64)
    basis = np.asarray(frame_basis, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    rows = len(valid)
    expected = {
        "local_boxes": (rows, 6),
        "center_residual": (rows, 3),
        "log_dimension_residual": (rows, 3),
        "frame_center": (rows, 3),
        "frame_basis": (rows, 3, 3),
    }
    for name, value in (
        ("local_boxes", local),
        ("center_residual", center_delta),
        ("log_dimension_residual", log_delta),
        ("frame_center", origin),
        ("frame_basis", basis),
    ):
        if value.shape != expected[name]:
            raise ValueError(f"{name}: shape {value.shape} != {expected[name]}")
    output = np.full((rows, 8, 3), np.nan, dtype=np.float32)
    if not valid.any():
        return output
    selected = np.flatnonzero(valid)
    selected_local = local[selected]
    if (
        not np.isfinite(selected_local).all()
        or np.any(selected_local[:, 3:6] <= 0.0)
        or not np.isfinite(center_delta[selected]).all()
        or not np.isfinite(log_delta[selected]).all()
        or not np.isfinite(origin[selected]).all()
        or not np.isfinite(basis[selected]).all()
    ):
        raise ValueError("valid sparse rows contain invalid reconstruction inputs")
    center_limit = max_center_fraction * selected_local[:, 3:6]
    candidate_center = selected_local[:, :3] + np.clip(
        center_delta[selected], -center_limit, center_limit
    )
    candidate_dims = selected_local[:, 3:6] * np.exp(
        np.clip(
            log_delta[selected],
            -max_log_dimension_residual,
            max_log_dimension_residual,
        )
    )
    candidate_dims = np.maximum(candidate_dims, minimum_dimension)
    local_corners = (
        candidate_center[:, None, :]
        + SIGNS[None, :, :] * candidate_dims[:, None, :] * 0.5
    )
    world_corners = (
        local_corners @ np.transpose(basis[selected], (0, 2, 1))
        + origin[selected, None, :]
    )
    output[selected] = world_corners.astype(np.float32)
    return output


def _load_scene(
    scene_id: str,
    *,
    prediction_root: Path,
    diagnostics_root: Path,
    gt_root: Path,
    scan_root: Path,
    max_center_fraction: float,
    max_log_dimension_residual: float,
    minimum_dimension: float,
    reconstruction_atol: float,
) -> SceneData:
    prediction_path = prediction_root / f"{scene_id}_boxes.pkl"
    diagnostic_path = diagnostics_root / f"{scene_id}_tracks.npz"
    labels, exported_corners, scores = _load_prediction(prediction_path)
    if not diagnostic_path.is_file():
        raise FileNotFoundError(diagnostic_path)
    with np.load(diagnostic_path, allow_pickle=False) as data:
        required = {
            "output_geometry_schema",
            "sparse_pair_schema",
            "output_pre_geometry_corners",
            "output_post_geometry_corners",
            "output_refit_applied",
            "result_indices",
            "sparse_local_boxes",
            "sparse_frame_center",
            "sparse_frame_basis",
            "sparse_output_valid",
            "sparse_accepted",
            "sparse_runtime_reason",
            "sparse_center_residual",
            "sparse_log_dimension_residual",
            "sparse_candidate_iou",
            "sparse_improvement_probability",
            "sparse_uncertainty",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{diagnostic_path}: missing arrays {missing}")
        if _scalar_text(
            data["output_geometry_schema"],
            name="output_geometry_schema",
            path=diagnostic_path,
        ) != GEOMETRY_SCHEMA:
            raise ValueError(f"{diagnostic_path}: unexpected geometry schema")
        if _scalar_text(
            data["sparse_pair_schema"],
            name="sparse_pair_schema",
            path=diagnostic_path,
        ) != PAIR_SCHEMA:
            raise ValueError(f"{diagnostic_path}: unexpected sparse pair schema")
        original = np.asarray(data["output_pre_geometry_corners"])
        active = np.asarray(data["output_post_geometry_corners"])
        applied = np.asarray(data["output_refit_applied"], dtype=bool)
        result_indices = np.asarray(data["result_indices"], dtype=np.int64)
        sparse_valid = np.asarray(data["sparse_output_valid"], dtype=bool)
        sparse_accepted = np.asarray(data["sparse_accepted"], dtype=bool)
        sparse_reason = np.asarray(data["sparse_runtime_reason"])
        sparse_candidate_iou = np.asarray(
            data["sparse_candidate_iou"], dtype=np.float64
        )
        sparse_improvement = np.asarray(
            data["sparse_improvement_probability"], dtype=np.float64
        )
        sparse_uncertainty = np.asarray(
            data["sparse_uncertainty"], dtype=np.float64
        )
        sparse_candidates = reconstruct_candidates(
            data["sparse_local_boxes"],
            data["sparse_center_residual"],
            data["sparse_log_dimension_residual"],
            data["sparse_frame_center"],
            data["sparse_frame_basis"],
            sparse_valid,
            max_center_fraction=max_center_fraction,
            max_log_dimension_residual=max_log_dimension_residual,
            minimum_dimension=minimum_dimension,
        )

    row_count = len(labels)
    if original.shape != (row_count, 8, 3) or active.shape != (
        row_count,
        8,
        3,
    ):
        raise ValueError(f"{diagnostic_path}: full-output geometry shape mismatch")
    if original.dtype != np.float32 or active.dtype != np.float32:
        raise ValueError(f"{diagnostic_path}: full-output geometry must be float32")
    if not np.array_equal(exported_corners, active):
        raise ValueError(f"{diagnostic_path}: exported geometry != diagnostic post")
    sparse_rows = len(result_indices)
    one_d_arrays = {
        "sparse_output_valid": sparse_valid,
        "sparse_accepted": sparse_accepted,
        "sparse_runtime_reason": sparse_reason,
        "sparse_candidate_iou": sparse_candidate_iou,
        "sparse_improvement_probability": sparse_improvement,
        "sparse_uncertainty": sparse_uncertainty,
    }
    for name, value in one_d_arrays.items():
        if value.shape != (sparse_rows,):
            raise ValueError(f"{diagnostic_path}: {name} shape mismatch")
    if result_indices.size and (
        int(result_indices.min()) < 0
        or int(result_indices.max()) >= row_count
        or np.unique(result_indices).size != result_indices.size
        or (result_indices.size > 1 and np.any(np.diff(result_indices) <= 0))
    ):
        raise ValueError(f"{diagnostic_path}: invalid result_indices")
    if np.any(sparse_accepted & ~sparse_valid):
        raise ValueError(f"{diagnostic_path}: accepted row lacks a valid candidate")

    candidate_corners = np.full_like(original, np.nan)
    candidate_valid = np.zeros(row_count, dtype=bool)
    candidate_accepted = np.zeros(row_count, dtype=bool)
    candidate_reason = np.full(row_count, "sparse_unobserved", dtype="<U96")
    candidate_iou_prediction = np.full(row_count, np.nan, dtype=np.float64)
    improvement_prediction = np.full(row_count, np.nan, dtype=np.float64)
    uncertainty_prediction = np.full(row_count, np.nan, dtype=np.float64)
    candidate_corners[result_indices] = sparse_candidates
    candidate_valid[result_indices] = sparse_valid
    candidate_accepted[result_indices] = sparse_accepted
    candidate_reason[result_indices] = sparse_reason
    candidate_iou_prediction[result_indices] = sparse_candidate_iou
    improvement_prediction[result_indices] = sparse_improvement
    uncertainty_prediction[result_indices] = sparse_uncertainty
    if not np.array_equal(applied, candidate_accepted):
        raise ValueError(f"{diagnostic_path}: applied rows != accepted rows")
    if candidate_accepted.any():
        reconstruction_error = float(
            np.max(
                np.abs(
                    candidate_corners[candidate_accepted]
                    - active[candidate_accepted]
                )
            )
        )
        if reconstruction_error > reconstruction_atol:
            raise ValueError(
                f"{diagnostic_path}: candidate reconstruction max error "
                f"{reconstruction_error:.9g} > {reconstruction_atol:.9g}"
            )
    rejected = ~candidate_accepted
    if not np.array_equal(active[rejected], original[rejected]):
        raise ValueError(
            f"{diagnostic_path}: non-accepted rows contain another geometry mutation"
        )

    gt_payload = np.load(gt_root / f"{scene_id}_bbox.npy")
    gt_minmax = center_size_to_minmax(gt_payload)
    return SceneData(
        scene_id=scene_id,
        labels=labels,
        scores=scores,
        original_corners=original.copy(),
        active_corners=active.copy(),
        candidate_corners=candidate_corners,
        candidate_valid=candidate_valid,
        candidate_accepted=candidate_accepted,
        candidate_reason=tuple(str(value) for value in candidate_reason.tolist()),
        candidate_iou_prediction=candidate_iou_prediction,
        improvement_prediction=improvement_prediction,
        uncertainty_prediction=uncertainty_prediction,
        gt_minmax=gt_minmax,
        axis_alignment=load_axis_alignment(scan_root, scene_id),
    )


def _prediction_minmax(scene: SceneData, corners: np.ndarray) -> np.ndarray:
    return corners_to_minmax(
        transform_corners(corners, scene.axis_alignment)
    )


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """The continuous VOC AP implementation used by ``eval_det.py``."""

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    changing = np.where(mrec[1:] != mrec[:-1])[0]
    return float(
        np.sum(
            (mrec[changing + 1] - mrec[changing])
            * mpre[changing + 1]
        )
    )


def _official_threshold_metrics(
    scene_ious: Mapping[str, np.ndarray],
    scene_scores: Mapping[str, np.ndarray],
    scene_order: tuple[str, ...],
    ground_truth_count: int,
    threshold: float,
) -> dict[str, Any]:
    """Match the class-agnostic ``eval_det_cls`` score/duplicate semantics."""

    prediction_scene: list[str] = []
    prediction_row: list[int] = []
    confidence: list[float] = []
    for scene_id in scene_order:
        scores = scene_scores[scene_id]
        prediction_scene.extend([scene_id] * len(scores))
        prediction_row.extend(range(len(scores)))
        confidence.extend(float(value) for value in scores)
    confidence_array = np.asarray(confidence, dtype=np.float64)
    # Deliberately retain NumPy's default argsort, as does eval_det.py.
    order = np.argsort(-confidence_array)
    detected = {
        scene_id: np.zeros(scene_ious[scene_id].shape[1], dtype=bool)
        for scene_id in scene_order
    }
    true_positive = np.zeros(len(order), dtype=np.float64)
    false_positive = np.zeros(len(order), dtype=np.float64)
    for rank, flat_index in enumerate(order):
        scene_id = prediction_scene[int(flat_index)]
        row = prediction_row[int(flat_index)]
        iou = scene_ious[scene_id][row]
        if not len(iou):
            false_positive[rank] = 1.0
            continue
        gt_index = int(np.argmax(iou))
        # The vendored evaluator uses strict >, not >=.
        if iou[gt_index] > threshold and not detected[scene_id][gt_index]:
            true_positive[rank] = 1.0
            detected[scene_id][gt_index] = True
        else:
            false_positive[rank] = 1.0
    cumulative_tp = np.cumsum(true_positive)
    cumulative_fp = np.cumsum(false_positive)
    recall = cumulative_tp / float(ground_truth_count + 1e-6)
    precision = cumulative_tp / np.maximum(
        cumulative_tp + cumulative_fp, np.finfo(np.float64).eps
    )
    ap = _voc_ap(recall, precision)
    return {
        "ap": ap,
        "ap_percent": 100.0 * ap,
        "recall": float(recall[-1]) if len(recall) else 0.0,
        "precision": float(precision[-1]) if len(precision) else 0.0,
        "true_positives": int(true_positive.sum()),
        "false_positives": int(false_positive.sum()),
    }


def _geometry_metrics(
    scenes: Iterable[SceneData],
    geometry: Mapping[str, np.ndarray],
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    scene_ious: dict[str, np.ndarray] = {}
    scene_scores: dict[str, np.ndarray] = {}
    scene_order: list[str] = []
    total_gt = total_predictions = 0
    for scene in scenes:
        corners = geometry[scene.scene_id]
        iou = pairwise_aabb_iou(
            _prediction_minmax(scene, corners), scene.gt_minmax
        )
        scene_ious[scene.scene_id] = iou
        scene_scores[scene.scene_id] = scene.scores
        scene_order.append(scene.scene_id)
        total_gt += len(scene.gt_minmax)
        total_predictions += len(scene.scores)
    output: dict[str, Any] = {}
    for threshold in thresholds:
        metrics = _official_threshold_metrics(
            scene_ious,
            scene_scores,
            tuple(scene_order),
            total_gt,
            threshold,
        )
        maximum_matches_count = 0
        strict_threshold = float(np.nextafter(threshold, np.inf))
        for scene_id in scene_order:
            matched_predictions, _ = maximum_matches(
                scene_ious[scene_id], strict_threshold
            )
            maximum_matches_count += len(matched_predictions)
        metrics["maximum_cardinality_matches"] = int(
            maximum_matches_count
        )
        metrics["maximum_cardinality_recall"] = float(
            maximum_matches_count / (total_gt + 1e-6)
        )
        output[f"{threshold:.2f}"] = metrics
    return {
        "predictions": total_predictions,
        "ground_truth": total_gt,
        "thresholds": output,
    }


def _replace_valid_candidates(scene: SceneData) -> np.ndarray:
    result = scene.original_corners.copy()
    result[scene.candidate_valid] = scene.candidate_corners[
        scene.candidate_valid
    ]
    return result


def _rowwise_best_geometry(scene: SceneData) -> tuple[np.ndarray, np.ndarray]:
    result = scene.original_corners.copy()
    selected = np.zeros(len(scene.scores), dtype=bool)
    if not scene.candidate_valid.any():
        return result, selected
    original_iou = pairwise_aabb_iou(
        _prediction_minmax(scene, scene.original_corners), scene.gt_minmax
    )
    candidate_rows = np.flatnonzero(scene.candidate_valid)
    candidate_iou = pairwise_aabb_iou(
        _prediction_minmax(scene, scene.candidate_corners[candidate_rows]),
        scene.gt_minmax,
    )
    original_best = original_iou[candidate_rows].max(axis=1, initial=0.0)
    candidate_best = candidate_iou.max(axis=1, initial=0.0)
    use_candidate = candidate_best > original_best + 1e-12
    selected[candidate_rows[use_candidate]] = True
    result[selected] = scene.candidate_corners[selected]
    return result, selected


def _ap_at_threshold(
    scenes: tuple[SceneData, ...],
    geometry: Mapping[str, np.ndarray],
    threshold: float,
) -> float:
    return float(
        _geometry_metrics(scenes, geometry, (threshold,))["thresholds"][
            f"{threshold:.2f}"
        ]["ap"]
    )


def _forward_ap_oracle(
    scenes: tuple[SceneData, ...],
    threshold: float,
) -> tuple[dict[str, np.ndarray], tuple[CandidateRef, ...], float]:
    """Greedily add only candidate replacements that increase exact AP."""

    geometry = {
        scene.scene_id: scene.original_corners.copy() for scene in scenes
    }
    remaining = [
        CandidateRef(scene.scene_id, int(index))
        for scene in scenes
        for index in np.flatnonzero(scene.candidate_valid)
    ]
    scene_by_id = {scene.scene_id: scene for scene in scenes}
    selected: list[CandidateRef] = []
    current_ap = _ap_at_threshold(scenes, geometry, threshold)
    while remaining:
        best_ref: CandidateRef | None = None
        best_ap = current_ap
        for candidate in remaining:
            scene = scene_by_id[candidate.scene_id]
            row = candidate.row_index
            original = geometry[candidate.scene_id][row].copy()
            geometry[candidate.scene_id][row] = scene.candidate_corners[row]
            candidate_ap = _ap_at_threshold(scenes, geometry, threshold)
            geometry[candidate.scene_id][row] = original
            if candidate_ap > best_ap + 1e-12:
                best_ap = candidate_ap
                best_ref = candidate
        if best_ref is None:
            break
        scene = scene_by_id[best_ref.scene_id]
        geometry[best_ref.scene_id][best_ref.row_index] = (
            scene.candidate_corners[best_ref.row_index]
        )
        selected.append(best_ref)
        remaining.remove(best_ref)
        current_ap = best_ap
    return geometry, tuple(selected), current_ap


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"min": None, "q10": None, "q50": None, "q90": None, "max": None}
    return {
        "min": float(np.min(array)),
        "q10": float(np.quantile(array, 0.10)),
        "q50": float(np.quantile(array, 0.50)),
        "q90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
    }


def _candidate_diagnostics(
    scenes: tuple[SceneData, ...], thresholds: tuple[float, ...]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        candidate_rows = np.flatnonzero(scene.candidate_valid)
        if not len(candidate_rows):
            continue
        original_iou = pairwise_aabb_iou(
            _prediction_minmax(scene, scene.original_corners), scene.gt_minmax
        )[candidate_rows]
        candidate_iou = pairwise_aabb_iou(
            _prediction_minmax(scene, scene.candidate_corners[candidate_rows]),
            scene.gt_minmax,
        )
        original_best = original_iou.max(axis=1, initial=0.0)
        candidate_best = candidate_iou.max(axis=1, initial=0.0)
        if scene.gt_minmax.shape[0]:
            original_target = np.argmax(original_iou, axis=1)
            candidate_target = np.argmax(candidate_iou, axis=1)
            candidate_fixed_target = candidate_iou[
                np.arange(len(candidate_rows)), original_target
            ]
        else:
            original_target = np.full(len(candidate_rows), -1, dtype=np.int64)
            candidate_target = np.full(len(candidate_rows), -1, dtype=np.int64)
            candidate_fixed_target = np.zeros(len(candidate_rows))
        for local_index, row_index in enumerate(candidate_rows):
            rows.append(
                {
                    "scene_id": scene.scene_id,
                    "row_index": int(row_index),
                    "score": float(scene.scores[row_index]),
                    "accepted": bool(scene.candidate_accepted[row_index]),
                    "reason": scene.candidate_reason[row_index],
                    "original_best_iou": float(original_best[local_index]),
                    "candidate_best_iou": float(candidate_best[local_index]),
                    "delta_best_iou": float(
                        candidate_best[local_index] - original_best[local_index]
                    ),
                    "original_best_gt_index": int(
                        original_target[local_index]
                    ),
                    "candidate_best_gt_index": int(
                        candidate_target[local_index]
                    ),
                    "best_gt_switched": bool(
                        original_target[local_index]
                        != candidate_target[local_index]
                    ),
                    "candidate_fixed_original_gt_iou": float(
                        candidate_fixed_target[local_index]
                    ),
                    "delta_fixed_original_gt_iou": float(
                        candidate_fixed_target[local_index]
                        - original_best[local_index]
                    ),
                    "predicted_candidate_iou": float(
                        scene.candidate_iou_prediction[row_index]
                    ),
                    "predicted_improvement_probability": float(
                        scene.improvement_prediction[row_index]
                    ),
                    "predicted_uncertainty": float(
                        scene.uncertainty_prediction[row_index]
                    ),
                }
            )
    delta = np.asarray([row["delta_best_iou"] for row in rows])
    fixed_delta = np.asarray(
        [row["delta_fixed_original_gt_iou"] for row in rows]
    )
    accepted = np.asarray([row["accepted"] for row in rows], dtype=bool)
    output: dict[str, Any] = {
        "valid_candidates": len(rows),
        "accepted_candidates": int(accepted.sum()),
        "rejected_valid_candidates": int((~accepted).sum()),
        "improved_best_iou": int((delta > 1e-12).sum()),
        "degraded_best_iou": int((delta < -1e-12).sum()),
        "unchanged_best_iou": int((np.abs(delta) <= 1e-12).sum()),
        "best_gt_switched": int(
            sum(bool(row["best_gt_switched"]) for row in rows)
        ),
        "delta_best_iou": _quantiles(delta),
        "delta_fixed_original_gt_iou": _quantiles(fixed_delta),
        "gate_confusion": {
            "accepted_improved": int((accepted & (delta > 1e-12)).sum()),
            "accepted_degraded": int((accepted & (delta < -1e-12)).sum()),
            "accepted_unchanged": int(
                (accepted & (np.abs(delta) <= 1e-12)).sum()
            ),
            "rejected_improved": int(((~accepted) & (delta > 1e-12)).sum()),
            "rejected_degraded": int(((~accepted) & (delta < -1e-12)).sum()),
            "rejected_unchanged": int(
                ((~accepted) & (np.abs(delta) <= 1e-12)).sum()
            ),
        },
        "crossings": {},
        "rows": rows,
    }
    original_best = np.asarray([row["original_best_iou"] for row in rows])
    candidate_best = np.asarray([row["candidate_best_iou"] for row in rows])
    for threshold in thresholds:
        upward = (original_best <= threshold) & (candidate_best > threshold)
        downward = (original_best > threshold) & (candidate_best <= threshold)
        output["crossings"][f"{threshold:.2f}"] = {
            "up": int(upward.sum()),
            "down": int(downward.sum()),
            "accepted_up": int((accepted & upward).sum()),
            "accepted_down": int((accepted & downward).sum()),
            "rejected_up": int(((~accepted) & upward).sum()),
            "rejected_down": int(((~accepted) & downward).sum()),
        }
    return output


def _write_predictions(
    root: Path,
    scenes: tuple[SceneData, ...],
    geometry: Mapping[str, np.ndarray],
) -> None:
    if root.exists():
        raise FileExistsError(f"refusing to overwrite counterfactual root: {root}")
    root.mkdir(parents=True)
    for scene in scenes:
        corners = np.asarray(geometry[scene.scene_id], dtype=np.float32)
        rows = [
            (int(label), corners[index].copy(), float(scene.scores[index]))
            for index, label in enumerate(scene.labels)
        ]
        with (root / f"{scene.scene_id}_boxes.pkl").open("wb") as handle:
            pickle.dump([rows], handle, protocol=pickle.HIGHEST_PROTOCOL)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = _load_checkpoint_decoder(
        args.checkpoint, args.expected_checkpoint_sha256
    )
    decoder = checkpoint["decoder"]
    scene_ids = read_scene_ids(args.scene_list)
    scenes = tuple(
        _load_scene(
            scene_id,
            prediction_root=args.prediction_root,
            diagnostics_root=args.diagnostics_root,
            gt_root=args.gt_root,
            scan_root=args.scan_root,
            max_center_fraction=decoder["max_center_fraction"],
            max_log_dimension_residual=decoder[
                "max_log_dimension_residual"
            ],
            minimum_dimension=decoder["minimum_dimension"],
            reconstruction_atol=args.reconstruction_atol,
        )
        for scene_id in scene_ids
    )
    thresholds = tuple(args.thresholds)
    identity = {scene.scene_id: scene.original_corners for scene in scenes}
    active = {scene.scene_id: scene.active_corners for scene in scenes}
    all_valid = {
        scene.scene_id: _replace_valid_candidates(scene) for scene in scenes
    }
    rowwise: dict[str, np.ndarray] = {}
    rowwise_selected: dict[str, list[int]] = {}
    for scene in scenes:
        corners, selected = _rowwise_best_geometry(scene)
        rowwise[scene.scene_id] = corners
        rowwise_selected[scene.scene_id] = np.flatnonzero(selected).tolist()

    methods: dict[str, Any] = {
        "identity": _geometry_metrics(scenes, identity, thresholds),
        "active": _geometry_metrics(scenes, active, thresholds),
        "all_valid_candidates": _geometry_metrics(
            scenes, all_valid, thresholds
        ),
        "rowwise_best_iou_oracle": _geometry_metrics(
            scenes, rowwise, thresholds
        ),
    }
    forward_geometries: dict[str, dict[str, np.ndarray]] = {}
    forward_selections: dict[str, list[dict[str, Any]]] = {}
    for threshold in thresholds:
        key = f"forward_ap_oracle_{threshold:.2f}"
        geometry, selected, _ = _forward_ap_oracle(scenes, threshold)
        forward_geometries[key] = geometry
        forward_selections[key] = [
            {"scene_id": item.scene_id, "row_index": item.row_index}
            for item in selected
        ]
        methods[key] = _geometry_metrics(scenes, geometry, thresholds)

    if args.counterfactual_root is not None:
        output_geometries = {
            "identity": identity,
            "active_replay": active,
            "all_valid_candidates": all_valid,
            "rowwise_best_iou_oracle": rowwise,
            **forward_geometries,
        }
        for name, geometry in output_geometries.items():
            _write_predictions(
                args.counterfactual_root / name, scenes, geometry
            )

    identity_metrics = methods["identity"]["thresholds"]
    delta_from_identity: dict[str, dict[str, float]] = {}
    for method_name, metrics in methods.items():
        delta_from_identity[method_name] = {
            key: float(value["ap"] - identity_metrics[key]["ap"])
            for key, value in metrics["thresholds"].items()
        }
    report = {
        "schema": SCHEMA,
        "warning": (
            "GT-only diagnostic. Oracle geometry selections are not a "
            "deployable or validation-time model result."
        ),
        "inputs": {
            "prediction_root": str(args.prediction_root.resolve()),
            "diagnostics_root": str(args.diagnostics_root.resolve()),
            "gt_root": str(args.gt_root.resolve()),
            "scan_root": str(args.scan_root.resolve()),
            "scene_list": str(args.scene_list.resolve()),
            "scenes": len(scenes),
        },
        "decoder": {
            **decoder,
            "accepted_reconstruction_atol": args.reconstruction_atol,
        },
        "checkpoint": checkpoint,
        "candidate_diagnostics": _candidate_diagnostics(scenes, thresholds),
        "methods": methods,
        "delta_ap_from_identity": delta_from_identity,
        "rowwise_best_selected": rowwise_selected,
        "forward_ap_oracle_selected": forward_selections,
        "counterfactual_root": (
            str(args.counterfactual_root.resolve())
            if args.counterfactual_root is not None
            else None
        ),
    }
    return report


def _thresholds(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(","))
    if not result or any(item <= 0.0 or item > 1.0 for item in result):
        raise argparse.ArgumentTypeError("thresholds must lie in (0,1]")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("thresholds must be unique")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default=FROZEN_CHECKPOINT_SHA256,
    )
    parser.add_argument(
        "--thresholds", type=_thresholds, default=DEFAULT_THRESHOLDS
    )
    parser.add_argument("--reconstruction-atol", type=float, default=5e-6)
    parser.add_argument("--counterfactual-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_checkpoint_sha256 and (
        len(args.expected_checkpoint_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in args.expected_checkpoint_sha256.lower()
        )
    ):
        parser.error("--expected-checkpoint-sha256 must be a SHA256 hex digest")
    for name in ("reconstruction_atol",):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    return args


def main() -> int:
    args = parse_args()
    report = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
