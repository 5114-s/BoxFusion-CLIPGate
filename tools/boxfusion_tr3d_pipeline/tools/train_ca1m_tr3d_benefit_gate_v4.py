#!/usr/bin/env python3
"""Fit CA-1M terminal-benefit v4 on folds 2/3/4 and tune on fold 0.

The input schema excludes fold 1.  Threshold gates use the official CA
world-enclosing-AABB, global-score, duplicate-aware AP reconstruction.  The
result remains non-canonical until a later separately authorized protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_terminal_gate_v4 import (  # noqa: E402
    BENEFIT_TARGET,
    DATASET_SCHEMA,
    DEV_GATE,
    FAILURE_ACTION,
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    FIT_DECAY_STEPS,
    FIT_ITERATIONS,
    FIT_L2,
    FIT_LEARNING_RATE,
    GATE_TRAIN_FOLDS,
    GT_SHADOW_INVENTORY_SCHEMA,
    IOU_THRESHOLDS,
    LOCKED_INTERNAL_FOLDS,
    MAX_REPLACEMENTS,
    OBJECTIVE_FIELDS,
    POLICY_SCHEMA,
    PREREGISTRATION_SCHEMA,
    QUALITY_TARGET,
    SELECTION_RULE,
    THRESHOLD_DEV_FOLDS,
    THRESHOLD_GRID,
    preregistration_science_contract,
    validate_preregistration_record,
    write_binding_create_only,
)
from boxfusion.ca1m_tr3d_terminal_v4 import sha256_file  # noqa: E402
from tools.build_ca1m_tr3d_benefit_dataset_v4 import (  # noqa: E402
    MANIFEST_SCHEMA as DATASET_MANIFEST_SCHEMA,
    SCORE_SOURCE,
)


REPORT_SCHEMA = "boxfusion.ca1m_tr3d_benefit_gate_training_report.v4"


def _regular(path: Path, name: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink: {path}")
    result = path.resolve()
    if (
        not result.is_file()
        or result.is_symlink()
        or result.stat().st_size <= 0
        or result.stat().st_mode & 0o222
    ):
        raise ValueError(f"{name} must be a sealed regular file: {result}")
    return result


def _json(path: Path, name: str) -> tuple[Path, dict[str, Any]]:
    source = _regular(path, name)
    try:
        payload = json.loads(source.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain an object")
    return source, payload


def _scalar(archive: Any, name: str) -> Any:
    if name not in archive.files:
        raise ValueError(f"dataset lacks scalar {name}")
    value = np.asarray(archive[name])
    if value.shape != ():
        raise ValueError(f"dataset {name} must be scalar")
    return value.item()


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    result = np.empty_like(source)
    positive = source >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-np.minimum(source[positive], 700.0)))
    exponential = np.exp(np.maximum(source[~positive], -700.0))
    result[~positive] = exponential / (1.0 + exponential)
    return result


def fit_logistic(
    standardized: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, float]:
    x = np.asarray(standardized, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    positives = int(np.count_nonzero(y > 0.5))
    negatives = len(y) - positives
    if positives < 5 or negatives < 5:
        raise ValueError("terminal gate v4 target lacks five positive/negative fit rows")
    sample_weight = np.where(
        y > 0.5, len(y) / (2.0 * positives), len(y) / (2.0 * negatives)
    )
    weights = np.zeros(x.shape[1], np.float64)
    bias = 0.0
    denominator = float(sample_weight.sum())
    for step in range(FIT_ITERATIONS):
        probability = stable_sigmoid(x @ weights + bias)
        error = (probability - y) * sample_weight
        rate = FIT_LEARNING_RATE / math.sqrt(1.0 + step / FIT_DECAY_STEPS)
        weights -= rate * ((x.T @ error) / denominator + FIT_L2 * weights)
        bias -= rate * float(error.sum() / denominator)
    if not np.isfinite(weights).all() or not math.isfinite(bias):
        raise RuntimeError("terminal gate v4 logistic fit became non-finite")
    return weights, bias


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    changed = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]))


def official_ca_ap(
    *,
    scene_ids: np.ndarray,
    scores: np.ndarray,
    best_iou: np.ndarray,
    best_gt: np.ndarray,
    gt_counts: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Global-score, duplicate-aware AP using strict world-AABB IoU gates."""

    scenes = np.asarray(scene_ids).astype(str)
    rank_score = np.asarray(scores, dtype=np.float64)
    iou = np.asarray(best_iou, dtype=np.float64)
    gt = np.asarray(best_gt, dtype=np.int64)
    count = len(rank_score)
    if (
        scenes.shape != rank_score.shape
        or iou.shape != rank_score.shape
        or gt.shape != rank_score.shape
        or not np.isfinite(rank_score).all()
        or not np.isfinite(iou).all()
    ):
        raise ValueError("official CA AP rows are invalid")
    # Match evaluation/utils/eval_det.py exactly.  The bound fold-0 OOF score
    # vector is audited below to contain no ties, so NumPy's official default
    # quicksort and a stable sort are row-for-row identical for this run.
    order = np.argsort(-rank_score)
    positives = int(np.asarray(gt_counts, dtype=np.int64).sum())
    result: dict[str, dict[str, float | int]] = {}
    for threshold in IOU_THRESHOLDS:
        tp = np.zeros(count, np.float64)
        fp = np.zeros(count, np.float64)
        detected: set[tuple[str, int]] = set()
        for rank, row in enumerate(order.tolist()):
            gt_index = int(gt[row])
            key = (str(scenes[row]), gt_index)
            if iou[row] > threshold and gt_index >= 0 and key not in detected:
                tp[rank] = 1.0
                detected.add(key)
            else:
                fp[rank] = 1.0
        cumulative_tp = np.cumsum(tp)
        cumulative_fp = np.cumsum(fp)
        recall = cumulative_tp / float(positives + 1.0e-6)
        precision = cumulative_tp / np.maximum(
            cumulative_tp + cumulative_fp, np.finfo(np.float64).eps
        )
        final_tp = int(cumulative_tp[-1]) if count else 0
        final_fp = int(cumulative_fp[-1]) if count else 0
        result[f"iou_{threshold:.2f}"] = {
            "ap": _voc_ap(recall, precision),
            "precision": float(precision[-1]) if count else 0.0,
            "recall": float(recall[-1]) if count else 0.0,
            "tp": final_tp,
            "fp": final_fp,
            "fn": positives - final_tp,
        }
    return result


def score_tie_audit(scores: np.ndarray) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("official CA AP score tie audit input is invalid")
    _, counts = np.unique(values, return_counts=True)
    tied = counts[counts > 1]
    official_order = np.argsort(-values)
    stable_order = np.argsort(-values, kind="mergesort")
    return {
        "row_count": len(values),
        "unique_score_count": len(counts),
        "tied_score_value_count": len(tied),
        "rows_in_ties": int(tied.sum()) if len(tied) else 0,
        "max_tie_multiplicity": int(tied.max()) if len(tied) else 1,
        "official_numpy_default_argsort": True,
        "default_quicksort_equals_stable_order": bool(np.array_equal(
            official_order, stable_order
        )),
    }


def _auc(target: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(target, np.bool_)
    values = np.asarray(score, np.float64)
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) * 0.5
        start = stop
    return float((ranks[y].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def select_rows(
    *,
    scene_ids: np.ndarray,
    candidate_rows: np.ndarray,
    anchor_indices: np.ndarray,
    candidate_scores: np.ndarray,
    quality_probability: np.ndarray,
    benefit_probability: np.ndarray,
    quality_threshold: float,
    benefit_threshold: float,
) -> np.ndarray:
    selected: list[int] = []
    for scene in sorted(set(np.asarray(scene_ids).astype(str).tolist())):
        scene_rows = np.flatnonzero(np.asarray(scene_ids).astype(str) == scene)
        eligible = scene_rows[
            (quality_probability[scene_rows] >= quality_threshold)
            & (benefit_probability[scene_rows] >= benefit_threshold)
        ]
        chosen: list[int] = []
        for anchor in np.unique(anchor_indices[eligible]).tolist():
            rows = eligible[anchor_indices[eligible] == anchor]
            chosen.append(max(
                rows.tolist(),
                key=lambda row: (
                    float(benefit_probability[row]), float(quality_probability[row]),
                    float(candidate_scores[row]), -int(candidate_rows[row]),
                ),
            ))
        chosen.sort(key=lambda row: (
            -float(benefit_probability[row]), -float(quality_probability[row]),
            -float(candidate_scores[row]), int(anchor_indices[row]), int(candidate_rows[row]),
        ))
        chosen = chosen[:MAX_REPLACEMENTS]
        chosen.sort(key=lambda row: (int(anchor_indices[row]), int(candidate_rows[row])))
        selected.extend(chosen)
    return np.asarray(selected, np.int64)


def evaluate_point(
    *,
    quality_threshold: float,
    benefit_threshold: float,
    quality_probability: np.ndarray,
    benefit_probability: np.ndarray,
    candidate: Mapping[str, np.ndarray],
    baseline: Mapping[str, np.ndarray],
    baseline_metrics: Mapping[str, Mapping[str, float | int]],
) -> dict[str, Any]:
    selected = select_rows(
        scene_ids=candidate["scene_ids"],
        candidate_rows=candidate["candidate_rows"],
        anchor_indices=candidate["anchor_indices"],
        candidate_scores=candidate["candidate_scores"],
        quality_probability=quality_probability,
        benefit_probability=benefit_probability,
        quality_threshold=quality_threshold,
        benefit_threshold=benefit_threshold,
    )
    output_iou = baseline["best_iou"].copy()
    output_gt = baseline["best_gt"].copy()
    lookup = {
        (str(scene), int(row)): index
        for index, (scene, row) in enumerate(zip(
            baseline["scene_ids"].tolist(), baseline["row_indices"].tolist()
        ))
    }
    for row in selected.tolist():
        target = lookup[(str(candidate["scene_ids"][row]), int(candidate["anchor_indices"][row]))]
        output_iou[target] = candidate["best_iou"][row]
        output_gt[target] = candidate["best_gt"][row]
    active = official_ca_ap(
        scene_ids=baseline["scene_ids"], scores=baseline["scores"],
        best_iou=output_iou, best_gt=output_gt, gt_counts=baseline["gt_counts"],
    )
    delta = {key: float(active[key]["ap"] - baseline_metrics[key]["ap"]) for key in active}
    count = len(selected)
    positive_fraction = float(candidate["benefit_target"][selected].mean()) if count else 0.0
    severe = (
        (candidate["same_gt_gain"][selected] <= -0.05)
        & (~candidate["target_switch"][selected])
    )
    severe_fraction = float(severe.mean()) if count else 0.0
    switch_fraction = float(candidate["target_switch"][selected].mean()) if count else 0.0
    scene_count = len(set(candidate["scene_ids"][selected].tolist()))
    checks = {
        "delta_ap15": delta["iou_0.15"] >= DEV_GATE["min_delta_ap15"],
        "delta_ap25": delta["iou_0.25"] >= DEV_GATE["min_delta_ap25"],
        "delta_ap50": delta["iou_0.50"] >= DEV_GATE["min_delta_ap50"],
        "replacement_count": count >= DEV_GATE["min_replacements"],
        "replacement_scene_count": scene_count >= DEV_GATE["min_scenes"],
        "positive_gain_fraction": positive_fraction >= DEV_GATE["min_positive_gain_fraction"],
        "severe_harm_fraction": severe_fraction <= DEV_GATE["max_severe_harm_fraction"],
        "target_switch_fraction": switch_fraction <= DEV_GATE["max_target_switch_fraction"],
    }
    checks["pass"] = all(checks.values())
    return {
        "quality_threshold": quality_threshold,
        "benefit_threshold": benefit_threshold,
        "replacement_count": count,
        "replacement_scene_count": scene_count,
        "positive_gain_fraction": positive_fraction,
        "severe_harm_fraction": severe_fraction,
        "target_switch_fraction": switch_fraction,
        "metrics": active,
        "ap_delta": delta,
        "gate": checks,
        "selected_dataset_rows": candidate["dataset_rows"][selected].tolist(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = _regular(args.dataset, "terminal gate v4 dataset")
    manifest_path, manifest = _json(args.dataset_manifest, "terminal gate v4 dataset manifest")
    if (
        manifest.get("schema") != DATASET_MANIFEST_SCHEMA
        or manifest.get("complete") is not True
        or manifest.get("train_only") is not True
        or manifest.get("scene_count") != 80
        or manifest.get("fit_scene_count") != 60
        or manifest.get("threshold_dev_scene_count") != 20
        or manifest.get("locked_internal_scene_count_accessed") != 0
        or manifest.get("locked_internal_fold1_gt_access") is not False
        or manifest.get("validation_ground_truth_access") is not False
        or manifest.get("validation_prediction_access") is not False
        or manifest.get("official_validation_comparable") is not False
        or manifest.get("anchor_score_source") != SCORE_SOURCE
        or manifest.get("deploy_b6_scores_used_for_stacked_training") is not False
        or (manifest.get("dataset") or {}).get("path") != str(dataset_path)
        or (manifest.get("dataset") or {}).get("sha256") != sha256_file(dataset_path)
        or manifest.get("legacy_artifact_reuse") is not False
    ):
        raise ValueError("terminal gate v4 dataset manifest contract differs")
    preregistration_record = manifest.get("preregistration_manifest") or {}
    preregistration_path, preregistration = validate_preregistration_record(
        preregistration_record
    )
    preregistration_sha256 = sha256_file(preregistration_path)
    gt_inventory_record = manifest.get("derived_train_gt_inventory_receipt") or {}
    preregistered_gt_inventory = (
        (preregistration.get("upstream") or {}).get(
            "derived_train_gt_inventory_receipt"
        ) or {}
    )
    if (
        preregistration_record.get("schema") != PREREGISTRATION_SCHEMA
        or preregistration_record.get("sha256") != preregistration_sha256
        or preregistration_record.get("sealed_before_first_gt_join") is not True
        or manifest.get("source_code_sha256")
        != (preregistration.get("code") or {}).get("dataset_builder", {}).get("sha256")
        or preregistration.get("science") != preregistration_science_contract()
        or gt_inventory_record.get("schema") != GT_SHADOW_INVENTORY_SCHEMA
        or gt_inventory_record.get("path") != preregistered_gt_inventory.get("path")
        or gt_inventory_record.get("sha256")
        != preregistered_gt_inventory.get("sha256")
    ):
        raise ValueError("dataset manifest does not reverse-bind preregistered code/science")
    with np.load(dataset_path, allow_pickle=False) as archive:
        fixed = {
            "schema": DATASET_SCHEMA,
            "complete": True,
            "train_only": True,
            "validation_ground_truth_access": False,
            "validation_prediction_access": False,
            "official_validation_comparable": False,
            "locked_internal_fold1_gt_access": False,
            "feature_schema": FEATURE_SCHEMA,
            "quality_target_schema": QUALITY_TARGET,
            "benefit_target_schema": BENEFIT_TARGET,
            "anchor_score_source": SCORE_SOURCE,
            "deploy_b6_scores_used_for_stacked_training": False,
            "preregistration_manifest_sha256": preregistration_sha256,
            "derived_train_gt_inventory_receipt_sha256": str(
                gt_inventory_record["sha256"]
            ),
        }
        for name, expected in fixed.items():
            if _scalar(archive, name) != expected:
                raise ValueError(f"terminal gate v4 dataset field {name} differs")
        if tuple(str(value) for value in np.asarray(archive["feature_names"]).tolist()) != FEATURE_NAMES:
            raise ValueError("terminal gate v4 feature names differ")
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    features = np.asarray(arrays["features"], np.float64)
    folds = np.asarray(arrays["fold_ids"], np.int64)
    scene_folds = np.asarray(arrays["scene_fold_ids"], np.int64)
    if (
        features.ndim != 2
        or features.shape[1] != len(FEATURE_NAMES)
        or not np.isfinite(features).all()
        or set(np.unique(folds).tolist()) != {0, 2, 3, 4}
        or 1 in set(scene_folds.tolist())
    ):
        raise ValueError("terminal gate v4 dataset includes invalid/fold1 rows")
    fit = np.isin(folds, GATE_TRAIN_FOLDS)
    dev = np.isin(folds, THRESHOLD_DEV_FOLDS)
    mean = features[fit].mean(axis=0)
    scale = features[fit].std(axis=0)
    scale[scale < 1.0e-6] = 1.0
    standardized = (features - mean) / scale
    quality_weights, quality_bias = fit_logistic(
        standardized[fit], np.asarray(arrays["quality25_target"], np.float64)[fit]
    )
    benefit_weights, benefit_bias = fit_logistic(
        standardized[fit], np.asarray(arrays["benefit05_target"], np.float64)[fit]
    )
    quality_probability = stable_sigmoid(standardized @ quality_weights + quality_bias)
    benefit_probability = stable_sigmoid(standardized @ benefit_weights + benefit_bias)
    candidate = {
        "dataset_rows": np.flatnonzero(dev).astype(np.int64),
        "scene_ids": np.asarray(arrays["scene_ids"]).astype(str)[dev],
        "candidate_rows": np.asarray(arrays["candidate_rows"], np.int64)[dev],
        "anchor_indices": np.asarray(arrays["anchor_indices"], np.int64)[dev],
        "candidate_scores": np.asarray(arrays["candidate_scores"], np.float32)[dev],
        "best_iou": np.asarray(arrays["candidate_best_iou"], np.float64)[dev],
        "best_gt": np.asarray(arrays["candidate_best_gt_indices"], np.int64)[dev],
        "benefit_target": np.asarray(arrays["benefit05_target"], np.bool_)[dev],
        "target_switch": np.asarray(arrays["target_switch"], np.bool_)[dev],
        "same_gt_gain": np.asarray(arrays["same_gt_iou_gain"], np.float64)[dev],
    }
    baseline_dev = np.asarray(arrays["baseline_fold_ids"], np.int64) == 0
    scene_dev = scene_folds == 0
    baseline = {
        "scene_ids": np.asarray(arrays["baseline_scene_ids"]).astype(str)[baseline_dev],
        "row_indices": np.asarray(arrays["baseline_row_indices"], np.int64)[baseline_dev],
        "scores": np.asarray(arrays["baseline_scores"], np.float32)[baseline_dev],
        "best_iou": np.asarray(arrays["baseline_best_iou"], np.float64)[baseline_dev],
        "best_gt": np.asarray(arrays["baseline_best_gt_indices"], np.int64)[baseline_dev],
        "gt_counts": np.asarray(arrays["scene_gt_counts"], np.int64)[scene_dev],
    }
    tie_audit = score_tie_audit(baseline["scores"])
    if (
        tie_audit["tied_score_value_count"] != 0
        or tie_audit["rows_in_ties"] != 0
        or tie_audit["default_quicksort_equals_stable_order"] is not True
    ):
        raise ValueError(
            "fold-0 OOF anchor scores contain unresolved official-AP tie semantics"
        )
    baseline_metrics = official_ca_ap(
        scene_ids=baseline["scene_ids"], scores=baseline["scores"],
        best_iou=baseline["best_iou"], best_gt=baseline["best_gt"],
        gt_counts=baseline["gt_counts"],
    )
    points = [
        evaluate_point(
            quality_threshold=quality_threshold,
            benefit_threshold=benefit_threshold,
            quality_probability=quality_probability[dev],
            benefit_probability=benefit_probability[dev],
            candidate=candidate,
            baseline=baseline,
            baseline_metrics=baseline_metrics,
        )
        for quality_threshold in THRESHOLD_GRID
        for benefit_threshold in THRESHOLD_GRID
    ]
    eligible = [point for point in points if point["gate"]["pass"]]

    def objective(point: Mapping[str, Any]) -> tuple[float, ...]:
        return (
            float(point["ap_delta"]["iou_0.50"]),
            float(point["ap_delta"]["iou_0.25"]),
            float(point["ap_delta"]["iou_0.15"]),
            float(point["positive_gain_fraction"]),
            -float(point["replacement_count"]),
            float(point["quality_threshold"]),
            float(point["benefit_threshold"]),
        )

    chosen = max(eligible, key=objective) if eligible else max(points, key=objective)
    dev_pass = bool(eligible)
    binding_sha = str(np.asarray(arrays["training_binding_sha256"]).item())
    policy = {
        "schema": POLICY_SCHEMA,
        "complete": True,
        "train_only": True,
        "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "official_validation_comparable": False,
        "formal_canonical103_authorized": False,
        "threshold_dev_gate_passed": dev_pass,
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "quality_target": QUALITY_TARGET,
        "benefit_target": BENEFIT_TARGET,
        "selection_rule": SELECTION_RULE,
        "fit_fold_ids": list(GATE_TRAIN_FOLDS),
        "threshold_dev_fold_ids": list(THRESHOLD_DEV_FOLDS),
        "locked_internal_fold_ids": list(LOCKED_INTERNAL_FOLDS),
        "anchor_score_source": SCORE_SOURCE,
        "deploy_b6_scores_used_for_stacked_training": False,
        "training_binding_sha256": binding_sha,
        "preregistration_manifest_sha256": preregistration_sha256,
        "dataset_sha256": sha256_file(dataset_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "quality25": {
            "mean": mean.tolist(), "scale": scale.tolist(),
            "weights": quality_weights.tolist(), "bias": quality_bias,
            "threshold": chosen["quality_threshold"],
        },
        "benefit05": {
            "mean": mean.tolist(), "scale": scale.tolist(),
            "weights": benefit_weights.tolist(), "bias": benefit_bias,
            "threshold": chosen["benefit_threshold"],
        },
        "max_replacements_per_scene": MAX_REPLACEMENTS,
        "source_code_sha256": sha256_file(Path(__file__).resolve()),
    }
    current_preregistration_path, current_preregistration = validate_preregistration_record(
        preregistration_record
    )
    if (
        current_preregistration_path != preregistration_path
        or sha256_file(current_preregistration_path) != preregistration_sha256
        or current_preregistration.get("science") != preregistration_science_contract()
    ):
        raise ValueError("preregistration/code/science changed during gate training")
    policy_output = args.output_policy.resolve()
    report_output = args.output_report.resolve()
    if policy_output == report_output or policy_output.exists() or report_output.exists():
        raise FileExistsError("refusing existing/aliased terminal gate v4 training outputs")
    write_binding_create_only(policy_output, policy)
    compact_points: list[dict[str, Any]] = []
    for point in points:
        copy = dict(point)
        selected = np.asarray(copy.pop("selected_dataset_rows"), np.int64)
        copy["selected_dataset_rows_sha256"] = hashlib.sha256(selected.tobytes()).hexdigest()
        compact_points.append(copy)
    report = {
        "schema": REPORT_SCHEMA,
        "complete": True,
        "train_only": True,
        "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "official_validation_comparable": False,
        "locked_internal_fold1_accessed": False,
        "formal_canonical103_authorized": False,
        "threshold_dev_gate_passed": dev_pass,
        "failure_action": None if dev_pass else FAILURE_ACTION,
        "policy": {"path": str(policy_output), "sha256": sha256_file(policy_output)},
        "dataset": {"path": str(dataset_path), "sha256": sha256_file(dataset_path)},
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "training_binding_sha256": binding_sha,
        "preregistration_manifest": {
            "path": str(preregistration_path),
            "sha256": preregistration_sha256,
            "schema": PREREGISTRATION_SCHEMA,
            "sealed_before_first_gt_join": True,
        },
        "derived_train_gt_inventory_receipt": dict(gt_inventory_record),
        "anchor_score_source": SCORE_SOURCE,
        "deploy_b6_scores_used_for_stacked_training": False,
        "fit_fold_ids": list(GATE_TRAIN_FOLDS),
        "threshold_dev_fold_ids": list(THRESHOLD_DEV_FOLDS),
        "locked_internal_fold_ids": list(LOCKED_INTERNAL_FOLDS),
        "fit_rows": int(fit.sum()),
        "dev_rows": int(dev.sum()),
        "fit_scene_count": len(set(np.asarray(arrays["scene_ids"]).astype(str)[fit].tolist())),
        "dev_scene_count": len(set(candidate["scene_ids"].tolist())),
        "fit_class_counts": {
            "quality25_positive": int(np.count_nonzero(np.asarray(arrays["quality25_target"])[fit])),
            "benefit05_positive": int(np.count_nonzero(np.asarray(arrays["benefit05_target"])[fit])),
        },
        "dev_auc": {
            "quality25": _auc(np.asarray(arrays["quality25_target"])[dev], quality_probability[dev]),
            "benefit05": _auc(np.asarray(arrays["benefit05_target"])[dev], benefit_probability[dev]),
        },
        "official_ca_ap_protocol": (
            "world_enclosing_aabb_global_score_duplicate_aware_strict_gt_"
            "numpy_default_argsort"
        ),
        "score_tie_audit": tie_audit,
        "baseline_metrics": baseline_metrics,
        "dev_gate_contract": DEV_GATE,
        "threshold_grid": list(THRESHOLD_GRID),
        "objective_fields": list(OBJECTIVE_FIELDS),
        "preregistered_science": preregistration_science_contract(),
        "eligible_operating_point_count": len(eligible),
        "chosen_operating_point": chosen,
        "operating_points": compact_points,
        "source_code_sha256": sha256_file(Path(__file__).resolve()),
    }
    try:
        write_binding_create_only(report_output, report)
    except BaseException:
        policy_output.unlink(missing_ok=True)
        raise
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--dataset", type=Path, required=True)
    value.add_argument("--dataset-manifest", type=Path, required=True)
    value.add_argument("--output-policy", type=Path, required=True)
    value.add_argument("--output-report", type=Path, required=True)
    return value


def main() -> int:
    report = run(parser().parse_args())
    print(json.dumps({
        "complete": True,
        "threshold_dev_gate_passed": report["threshold_dev_gate_passed"],
        "eligible_operating_point_count": report["eligible_operating_point_count"],
        "locked_internal_fold1_accessed": False,
        "chosen_operating_point": report["chosen_operating_point"],
    }, sort_keys=True))
    return 0 if report["threshold_dev_gate_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
