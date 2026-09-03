#!/usr/bin/env python3
"""Fit and calibrate a train-only CA-1M terminal-TR3D benefit gate.

Weights are fit only on frozen folds 2/3/4.  The pair of probability
thresholds is chosen only on fold 0, using the official CA-1M global-score,
duplicate-aware AP protocol reconstructed from precomputed max-IoU targets.
Fold 1 is not present in this program's input dataset and cannot be read here.

This stage never emits an active runtime policy.  It emits an immutable
calibration model and report; a separate one-time fold-1 audit is required for
activation authorization.
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
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_terminal import sha256_file  # noqa: E402
from boxfusion.ca1m_tr3d_terminal_gate import (  # noqa: E402
    BENEFIT_TARGET,
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    QUALITY_TARGET,
    SELECTION_RULE,
)
from tools.build_ca1m_tr3d_benefit_dataset import (  # noqa: E402
    MANIFEST_SCHEMA as DATASET_MANIFEST_SCHEMA,
    SCHEMA as DATASET_SCHEMA,
    SPLIT_SCHEMA,
)


SCHEMA = "boxfusion.ca1m_tr3d_benefit_calibration.v1"
REPORT_SCHEMA = "boxfusion.ca1m_tr3d_benefit_calibration_report.v1"
THRESHOLDS = (0.15, 0.25, 0.50)
TRAIN_FOLDS = (2, 3, 4)
DEV_FOLDS = (0,)
LOCKED_FOLDS = (1,)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--dataset", type=Path, required=True)
    value.add_argument("--dataset-manifest", type=Path, required=True)
    value.add_argument("--split-manifest", type=Path, required=True)
    value.add_argument("--output-model", type=Path, required=True)
    value.add_argument("--output-report", type=Path, required=True)
    return value


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    result = path.resolve()
    if not result.is_file() or result.is_symlink() or result.stat().st_size <= 0:
        raise FileNotFoundError(f"missing regular {label}: {result}")
    return result


def _json(path: Path, label: str) -> tuple[Path, Mapping[str, Any]]:
    source = _regular(path, label)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON {label}: {source}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return source, value


def _scalar(archive: Any, key: str) -> Any:
    if key not in archive.files:
        raise ValueError(f"dataset lacks scalar {key}")
    value = np.asarray(archive[key])
    if value.shape != ():
        raise ValueError(f"dataset field {key} must be scalar")
    return value.item()


def _sigmoid(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    output = np.empty_like(source)
    positive = source >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-np.minimum(source[positive], 700.0)))
    exponential = np.exp(np.maximum(source[~positive], -700.0))
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _fit_logistic(
    standardized: np.ndarray,
    target: np.ndarray,
    *,
    iterations: int,
    learning_rate: float,
    decay_steps: float,
    l2: float,
) -> tuple[np.ndarray, float]:
    x = np.asarray(standardized, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    positives = int(np.count_nonzero(y > 0.5))
    negatives = len(y) - positives
    if positives < 5 or negatives < 5:
        raise ValueError("logistic target lacks at least five positive/negative rows")
    sample_weight = np.where(
        y > 0.5,
        len(y) / (2.0 * positives),
        len(y) / (2.0 * negatives),
    )
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    denominator = float(sample_weight.sum())
    for step in range(iterations):
        probability = _sigmoid(x @ weights + bias)
        error = (probability - y) * sample_weight
        rate = learning_rate / math.sqrt(1.0 + step / decay_steps)
        weights -= rate * ((x.T @ error) / denominator + l2 * weights)
        bias -= rate * float(error.sum() / denominator)
    if not np.isfinite(weights).all() or not math.isfinite(bias):
        raise RuntimeError("logistic optimization produced non-finite parameters")
    return weights, bias


def _auc(target: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(target, dtype=np.bool_)
    values = np.asarray(score, dtype=np.float64)
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) * 0.5
        start = stop
    return float((ranks[y].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    changed = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]))


def _metrics(
    *,
    scene_ids: np.ndarray,
    scores: np.ndarray,
    best_iou: np.ndarray,
    best_gt: np.ndarray,
    scene_table: np.ndarray,
    gt_counts: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    count = len(scores)
    if not (
        scene_ids.shape == scores.shape == best_iou.shape == best_gt.shape == (count,)
        and np.isfinite(scores).all()
        and np.isfinite(best_iou).all()
    ):
        raise ValueError("invalid rows for CA-1M AP computation")
    order = np.argsort(-scores.astype(np.float64))
    positives = int(np.asarray(gt_counts, dtype=np.int64).sum())
    result: dict[str, dict[str, float | int]] = {}
    for threshold in THRESHOLDS:
        tp = np.zeros(count, dtype=np.float64)
        fp = np.zeros(count, dtype=np.float64)
        detected: set[tuple[str, int]] = set()
        for rank, row in enumerate(order.tolist()):
            gt_index = int(best_gt[row])
            key = (str(scene_ids[row]), gt_index)
            if best_iou[row] > threshold and gt_index >= 0 and key not in detected:
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


def _gate_conditions(
    *,
    delta: Mapping[str, float],
    replacements: int,
    scenes: int,
    positive_fraction: float,
    severe_harm_fraction: float,
    target_switch_fraction: float,
    contract: Mapping[str, Any],
) -> dict[str, bool]:
    checks = {
        "delta_ap15": delta["iou_0.15"] >= float(contract["min_delta_ap15"]),
        "delta_ap25": delta["iou_0.25"] >= float(contract["min_delta_ap25"]),
        "delta_ap50": delta["iou_0.50"] >= float(contract["min_delta_ap50"]),
        "replacement_count": replacements >= int(contract["min_replacements"]),
        "replacement_scene_count": scenes >= int(contract["min_scenes"]),
        "positive_gain_fraction": positive_fraction
        >= float(contract["min_positive_gain_fraction"]),
        "severe_harm_fraction": severe_harm_fraction
        <= float(contract["max_severe_harm_fraction"]),
        "target_switch_fraction": target_switch_fraction
        <= float(contract["max_target_switch_fraction"]),
    }
    checks["pass"] = all(checks.values())
    return checks


def _selection(
    *,
    candidate_scene: np.ndarray,
    candidate_rows: np.ndarray,
    anchor_indices: np.ndarray,
    candidate_scores: np.ndarray,
    quality_probability: np.ndarray,
    benefit_probability: np.ndarray,
    quality_threshold: float,
    benefit_threshold: float,
    maximum: int,
) -> np.ndarray:
    selected: list[int] = []
    for scene in sorted(set(candidate_scene.tolist())):
        scene_rows = np.flatnonzero(candidate_scene == scene)
        eligible = scene_rows[
            (quality_probability[scene_rows] >= quality_threshold)
            & (benefit_probability[scene_rows] >= benefit_threshold)
        ]
        chosen: list[int] = []
        for anchor in np.unique(anchor_indices[eligible]).tolist():
            local = eligible[anchor_indices[eligible] == anchor]
            best = max(
                local.tolist(),
                key=lambda row: (
                    float(benefit_probability[row]),
                    float(quality_probability[row]),
                    float(candidate_scores[row]),
                    -int(candidate_rows[row]),
                ),
            )
            chosen.append(best)
        chosen.sort(
            key=lambda row: (
                -float(benefit_probability[row]),
                -float(quality_probability[row]),
                -float(candidate_scores[row]),
                int(anchor_indices[row]),
                int(candidate_rows[row]),
            )
        )
        chosen = chosen[:maximum]
        chosen.sort(key=lambda row: (int(anchor_indices[row]), int(candidate_rows[row])))
        selected.extend(chosen)
    return np.asarray(selected, dtype=np.int64)


def _evaluate_operating_point(
    *,
    quality_threshold: float,
    benefit_threshold: float,
    quality_probability: np.ndarray,
    benefit_probability: np.ndarray,
    candidate: Mapping[str, np.ndarray],
    baseline: Mapping[str, np.ndarray],
    baseline_metrics: Mapping[str, Mapping[str, float | int]],
    max_replacements: int,
    gate_contract: Mapping[str, Any],
) -> dict[str, Any]:
    selected = _selection(
        candidate_scene=candidate["scene_ids"],
        candidate_rows=candidate["candidate_rows"],
        anchor_indices=candidate["anchor_indices"],
        candidate_scores=candidate["candidate_scores"],
        quality_probability=quality_probability,
        benefit_probability=benefit_probability,
        quality_threshold=quality_threshold,
        benefit_threshold=benefit_threshold,
        maximum=max_replacements,
    )
    output_iou = baseline["best_iou"].copy()
    output_gt = baseline["best_gt"].copy()
    baseline_lookup = {
        (str(scene), int(row)): index
        for index, (scene, row) in enumerate(
            zip(baseline["scene_ids"].tolist(), baseline["row_indices"].tolist())
        )
    }
    for row in selected.tolist():
        key = (str(candidate["scene_ids"][row]), int(candidate["anchor_indices"][row]))
        target = baseline_lookup[key]
        output_iou[target] = candidate["best_iou"][row]
        output_gt[target] = candidate["best_gt"][row]
    active_metrics = _metrics(
        scene_ids=baseline["scene_ids"],
        scores=baseline["scores"],
        best_iou=output_iou,
        best_gt=output_gt,
        scene_table=baseline["scene_table"],
        gt_counts=baseline["gt_counts"],
    )
    delta = {
        key: float(active_metrics[key]["ap"] - baseline_metrics[key]["ap"])
        for key in baseline_metrics
    }
    count = len(selected)
    positive_fraction = (
        float(candidate["benefit_target"][selected].mean()) if count else 0.0
    )
    severe_harm = (
        candidate["same_gt_gain"][selected] <= -0.05
    ) & (~candidate["target_switch"][selected])
    severe_harm_fraction = float(severe_harm.mean()) if count else 0.0
    target_switch_fraction = (
        float(candidate["target_switch"][selected].mean()) if count else 0.0
    )
    scene_count = len(set(candidate["scene_ids"][selected].tolist()))
    checks = _gate_conditions(
        delta=delta,
        replacements=count,
        scenes=scene_count,
        positive_fraction=positive_fraction,
        severe_harm_fraction=severe_harm_fraction,
        target_switch_fraction=target_switch_fraction,
        contract=gate_contract,
    )
    return {
        "quality_threshold": quality_threshold,
        "benefit_threshold": benefit_threshold,
        "replacement_count": count,
        "replacement_scene_count": scene_count,
        "positive_gain_count": int(np.count_nonzero(candidate["benefit_target"][selected])),
        "positive_gain_fraction": positive_fraction,
        "severe_harm_count": int(np.count_nonzero(severe_harm)),
        "severe_harm_fraction": severe_harm_fraction,
        "target_switch_count": int(np.count_nonzero(candidate["target_switch"][selected])),
        "target_switch_fraction": target_switch_fraction,
        "metrics": active_metrics,
        "ap_delta": delta,
        "gate": checks,
        "selected_dataset_rows": candidate["dataset_rows"][selected].tolist(),
    }


def _write_npz_create_only(path: Path, arrays: Mapping[str, Any]) -> Path:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, target)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite calibration model: {target}") from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return target


def _write_json_create_only(path: Path, payload: Mapping[str, Any]) -> Path:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{target.name}.", suffix=".tmp",
            dir=target.parent, delete=False
        ) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, target)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite calibration report: {target}") from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return target


def _code_manifest() -> dict[str, str]:
    sources = (
        Path(__file__).resolve(),
        ROOT / "tools/build_ca1m_tr3d_benefit_dataset.py",
        ROOT / "boxfusion/ca1m_tr3d_terminal_gate.py",
        ROOT / "boxfusion/ca1m_tr3d_terminal.py",
    )
    return {
        str(source.relative_to(ROOT)): sha256_file(_regular(source, "training code source"))
        for source in sources
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    dataset_path = _regular(args.dataset, "benefit fit/dev dataset")
    dataset_manifest_path, dataset_manifest = _json(
        args.dataset_manifest, "benefit fit/dev dataset manifest"
    )
    split_path, split = _json(args.split_manifest, "frozen benefit split manifest")
    if (
        dataset_manifest.get("schema") != DATASET_MANIFEST_SCHEMA
        or dataset_manifest.get("complete") is not True
        or dataset_manifest.get("train_only") is not True
        or dataset_manifest.get("official_validation_access") is not False
        or dataset_manifest.get("partition") != "fit_dev"
        or dataset_manifest.get("dataset_sha256") != sha256_file(dataset_path)
        or dataset_manifest.get("split_manifest_sha256") != sha256_file(split_path)
        or split.get("schema") != SPLIT_SCHEMA
        or split.get("complete") is not True
        or split.get("train_only") is not True
        or split.get("official_validation_access") is not False
    ):
        raise ValueError("benefit fit/dev provenance mismatch")
    dataset_code = dataset_manifest.get("code_manifest")
    if not isinstance(dataset_code, Mapping):
        raise ValueError("benefit dataset lacks a frozen code manifest")
    for relative in (
        "tools/build_ca1m_tr3d_benefit_dataset.py",
        "boxfusion/ca1m_tr3d_terminal_gate.py",
        "boxfusion/ca1m_tr3d_terminal.py",
    ):
        if dataset_code.get(relative) != sha256_file(_regular(ROOT / relative, relative)):
            raise ValueError(f"benefit dataset code changed after construction: {relative}")
    model_contract = split.get("model")
    dev_gate = split.get("threshold_dev_gate")
    if not isinstance(model_contract, Mapping) or not isinstance(dev_gate, Mapping):
        raise ValueError("frozen benefit split lacks model/dev gate")
    expected_model = {
        "kind": "dual_class_balanced_logistic",
        "heads": ["quality25", "benefit05"],
        "normalization": "weights_train_only_mean_std",
        "selection": SELECTION_RULE,
    }
    for key, expected in expected_model.items():
        if model_contract.get(key) != expected:
            raise ValueError(f"frozen model contract changed: {key}")
    threshold_choice = model_contract.get("threshold_selection")
    expected_objective = [
        "delta_ap50", "delta_ap25", "delta_ap15", "positive_gain_fraction",
        "negative_replacement_count", "quality_threshold", "benefit_threshold",
    ]
    if (
        not isinstance(threshold_choice, Mapping)
        or threshold_choice.get("eligible")
        != "must_pass_all_threshold_dev_gate_conditions"
        or threshold_choice.get("maximize_lexicographically") != expected_objective
    ):
        raise ValueError("threshold selection objective is not frozen")
    with np.load(dataset_path, allow_pickle=False) as archive:
        if (
            str(_scalar(archive, "schema")) != DATASET_SCHEMA
            or bool(_scalar(archive, "complete")) is not True
            or bool(_scalar(archive, "train_only")) is not True
            or bool(_scalar(archive, "official_validation_access")) is not False
            or str(_scalar(archive, "partition")) != "fit_dev"
            or str(_scalar(archive, "feature_schema")) != FEATURE_SCHEMA
            or tuple(np.asarray(archive["feature_names"]).astype(str).tolist())
            != FEATURE_NAMES
            or str(_scalar(archive, "quality_target_schema")) != QUALITY_TARGET
            or str(_scalar(archive, "benefit_target_schema")) != BENEFIT_TARGET
        ):
            raise ValueError("unsupported benefit fit/dev dataset schema")
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}

    features = np.asarray(arrays["features"], dtype=np.float64)
    folds = np.asarray(arrays["fold_ids"], dtype=np.int64)
    if (
        features.ndim != 2
        or features.shape[1] != len(FEATURE_NAMES)
        or not np.isfinite(features).all()
        or set(np.unique(folds).tolist()) != {0, 2, 3, 4}
    ):
        raise ValueError("fit/dev feature matrix or folds are invalid")
    train = np.isin(folds, TRAIN_FOLDS)
    dev = folds == 0
    mean = features[train].mean(axis=0)
    scale = features[train].std(axis=0)
    scale[scale < 1.0e-6] = 1.0
    standardized_train = (features[train] - mean) / scale
    standardized_all = (features - mean) / scale
    iterations = int(model_contract["iterations"])
    learning_rate = float(model_contract["learning_rate"])
    decay_steps = float(model_contract["learning_rate_decay_steps"])
    l2 = float(model_contract["l2"])
    quality_weights, quality_bias = _fit_logistic(
        standardized_train,
        np.asarray(arrays["quality25_target"], dtype=np.float64)[train],
        iterations=iterations,
        learning_rate=learning_rate,
        decay_steps=decay_steps,
        l2=l2,
    )
    benefit_weights, benefit_bias = _fit_logistic(
        standardized_train,
        np.asarray(arrays["benefit05_target"], dtype=np.float64)[train],
        iterations=iterations,
        learning_rate=learning_rate,
        decay_steps=decay_steps,
        l2=l2,
    )
    quality_probability = _sigmoid(standardized_all @ quality_weights + quality_bias)
    benefit_probability = _sigmoid(standardized_all @ benefit_weights + benefit_bias)

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
    scene_dev = np.asarray(arrays["scene_fold_ids"], np.int64) == 0
    baseline = {
        "scene_ids": np.asarray(arrays["baseline_scene_ids"]).astype(str)[baseline_dev],
        "row_indices": np.asarray(arrays["baseline_row_indices"], np.int64)[baseline_dev],
        "scores": np.asarray(arrays["baseline_scores"], np.float32)[baseline_dev],
        "best_iou": np.asarray(arrays["baseline_best_iou"], np.float64)[baseline_dev],
        "best_gt": np.asarray(arrays["baseline_best_gt_indices"], np.int64)[baseline_dev],
        "scene_table": np.asarray(arrays["scene_table"]).astype(str)[scene_dev],
        "gt_counts": np.asarray(arrays["scene_gt_counts"], np.int64)[scene_dev],
    }
    baseline_metrics = _metrics(
        scene_ids=baseline["scene_ids"], scores=baseline["scores"],
        best_iou=baseline["best_iou"], best_gt=baseline["best_gt"],
        scene_table=baseline["scene_table"], gt_counts=baseline["gt_counts"],
    )
    grid_contract = model_contract.get("threshold_grid")
    if not isinstance(grid_contract, Mapping):
        raise ValueError("threshold grid contract is missing")
    start = float(grid_contract["start"])
    stop = float(grid_contract["stop"])
    step = float(grid_contract["step"])
    grid = [round(float(value), 12) for value in np.arange(start, stop + step * 0.5, step)]
    grid.extend(float(value) for value in grid_contract.get("include", ()))
    grid = sorted(set(grid))
    if not grid or any(not 0.0 < value < 1.0 for value in grid):
        raise ValueError("invalid frozen threshold grid")
    max_replacements = int(model_contract["max_replacements_per_scene"])
    operating_points: list[dict[str, Any]] = []
    for quality_threshold in grid:
        for benefit_threshold in grid:
            point = _evaluate_operating_point(
                quality_threshold=quality_threshold,
                benefit_threshold=benefit_threshold,
                quality_probability=quality_probability[dev],
                benefit_probability=benefit_probability[dev],
                candidate=candidate,
                baseline=baseline,
                baseline_metrics=baseline_metrics,
                max_replacements=max_replacements,
                gate_contract=dev_gate,
            )
            operating_points.append(point)
    eligible = [point for point in operating_points if point["gate"]["pass"]]
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
    chosen = max(eligible, key=objective) if eligible else max(operating_points, key=objective)
    dev_pass = bool(eligible)
    model_arrays: dict[str, Any] = {
        "schema": np.asarray(SCHEMA),
        "complete": np.asarray(True, np.bool_),
        "train_only": np.asarray(True, np.bool_),
        "activation_authorized": np.asarray(False, np.bool_),
        "threshold_dev_gate_passed": np.asarray(dev_pass, np.bool_),
        "one_time_internal_check_pending": np.asarray(dev_pass, np.bool_),
        "feature_schema": np.asarray(FEATURE_SCHEMA),
        "feature_names": np.asarray(FEATURE_NAMES),
        "feature_mean": mean.astype(np.float64),
        "feature_scale": scale.astype(np.float64),
        "quality25_weights": quality_weights.astype(np.float64),
        "quality25_bias": np.asarray(quality_bias, np.float64),
        "benefit05_weights": benefit_weights.astype(np.float64),
        "benefit05_bias": np.asarray(benefit_bias, np.float64),
        "quality25_threshold": np.asarray(chosen["quality_threshold"], np.float64),
        "benefit05_threshold": np.asarray(chosen["benefit_threshold"], np.float64),
        "max_replacements_per_scene": np.asarray(max_replacements, np.int64),
        "fit_fold_ids": np.asarray(TRAIN_FOLDS, np.int8),
        "calibration_fold_ids": np.asarray(DEV_FOLDS, np.int8),
        "locked_internal_fold_ids": np.asarray(LOCKED_FOLDS, np.int8),
        "dataset_sha256": np.asarray(sha256_file(dataset_path)),
        "dataset_manifest_sha256": np.asarray(sha256_file(dataset_manifest_path)),
        "split_manifest_sha256": np.asarray(sha256_file(split_path)),
        "code_manifest_json": np.asarray(
            json.dumps(_code_manifest(), sort_keys=True, separators=(",", ":"))
        ),
    }
    output_model_target = args.output_model.resolve()
    output_report_target = args.output_report.resolve()
    if output_model_target == output_report_target:
        raise ValueError("calibration model and report outputs must be distinct")
    if output_model_target.exists() or output_report_target.exists():
        raise FileExistsError("refusing a partial/overwriting calibration transaction")
    model_path = _write_npz_create_only(output_model_target, model_arrays)
    # Selected row identities are useful for auditing but needlessly large in
    # all 400 operating points.  Seal their hashes and keep the chosen list.
    chosen_selected_rows = list(chosen["selected_dataset_rows"])
    compact_points: list[dict[str, Any]] = []
    for point in operating_points:
        selected = np.asarray(point.pop("selected_dataset_rows"), dtype=np.int64)
        point["selected_dataset_rows_sha256"] = hashlib.sha256(
            selected.tobytes(order="C")
        ).hexdigest()
        compact_points.append(point)
    chosen["selected_dataset_rows"] = chosen_selected_rows
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "complete": True,
        "train_only": True,
        "official_validation_access": False,
        "activation_authorized": False,
        "threshold_dev_gate_passed": dev_pass,
        "locked_internal_check_accessed": False,
        "locked_internal_check_authorized": dev_pass,
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "dataset_sha256": sha256_file(dataset_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "split_manifest_sha256": sha256_file(split_path),
        "code_manifest": _code_manifest(),
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "quality_target": QUALITY_TARGET,
        "benefit_target": BENEFIT_TARGET,
        "selection_rule": SELECTION_RULE,
        "fit_fold_ids": list(TRAIN_FOLDS),
        "threshold_dev_fold_ids": list(DEV_FOLDS),
        "locked_internal_fold_ids": list(LOCKED_FOLDS),
        "fit_rows": int(train.sum()),
        "dev_rows": int(dev.sum()),
        "fit_scene_count": len(set(np.asarray(arrays["scene_ids"]).astype(str)[train].tolist())),
        "dev_scene_count": len(set(candidate["scene_ids"].tolist())),
        "fit_class_counts": {
            "quality25_positive": int(np.count_nonzero(np.asarray(arrays["quality25_target"])[train])),
            "benefit05_positive": int(np.count_nonzero(np.asarray(arrays["benefit05_target"])[train])),
        },
        "dev_auc": {
            "quality25": _auc(np.asarray(arrays["quality25_target"])[dev], quality_probability[dev]),
            "benefit05": _auc(np.asarray(arrays["benefit05_target"])[dev], benefit_probability[dev]),
        },
        "evidence_validity": {
            "fit_candidate_valid_rows": int(
                np.count_nonzero(np.asarray(arrays["candidate_valid_evidence"])[train])
            ),
            "fit_anchor_valid_rows": int(
                np.count_nonzero(np.asarray(arrays["anchor_valid_evidence"])[train])
            ),
            "dev_candidate_valid_rows": int(
                np.count_nonzero(np.asarray(arrays["candidate_valid_evidence"])[dev])
            ),
            "dev_anchor_valid_rows": int(
                np.count_nonzero(np.asarray(arrays["anchor_valid_evidence"])[dev])
            ),
            "selection_policy": "validity_encoded_in_frozen_features_no_hidden_row_filter",
        },
        "baseline_metrics": baseline_metrics,
        "threshold_grid": grid,
        "threshold_selection_objective": expected_objective,
        "eligible_operating_point_count": len(eligible),
        "chosen_operating_point": chosen,
        "operating_points": compact_points,
        "failure_action": None if dev_pass else "stop_without_opening_locked_fold1",
    }
    # Replace the compact crossing placeholder with explicit fit/dev gain/loss
    # counts.  Keeping all four values avoids a hidden target-crossing policy.
    report["crossing_counts"] = {
        split_name: {
            suffix: {
                key: int(np.count_nonzero(np.asarray(arrays[f"{key}"])[mask]))
                for key in (
                    f"cross{suffix}_gain",
                    f"cross{suffix}_loss",
                    f"identity_cross{suffix}_gain",
                    f"identity_cross{suffix}_loss",
                )
            }
            for suffix in ("15", "25", "50")
        }
        for split_name, mask in (("fit", train), ("dev", dev))
    }
    try:
        _write_json_create_only(output_report_target, report)
    except BaseException:
        try:
            model_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return report


def main() -> int:
    args = parser().parse_args()
    result = run(args)
    print(json.dumps({
        "complete": result["complete"],
        "threshold_dev_gate_passed": result["threshold_dev_gate_passed"],
        "eligible_operating_point_count": result["eligible_operating_point_count"],
        "dev_auc": result["dev_auc"],
        "baseline_metrics": result["baseline_metrics"],
        "chosen_operating_point": result["chosen_operating_point"],
        "failure_action": result["failure_action"],
    }, indent=2, sort_keys=True))
    return 0 if result["threshold_dev_gate_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
