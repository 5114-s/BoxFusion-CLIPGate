#!/usr/bin/env python3
"""Train and freeze the scene-grouped R3 veto-only risk calibrator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.tr3d_r2_provenance import code_artifact_tree_sha256, sha256_file  # noqa: E402
from boxfusion.tr3d_r3_calibration_dataset import (  # noqa: E402
    HARM_CLASS,
    IOU_THRESHOLDS,
    load_dataset,
    prediction_rows,
)
from boxfusion.tr3d_r3_calibrator import (  # noqa: E402
    CLASS_NAMES,
    FEATURE_NAMES,
    R3VetoCalibrator,
)
from tools.audit_tr3d_r3_near_correction import scored_detection_metrics  # noqa: E402


REPORT_SCHEMA = "boxfusion.tr3d_r3_veto_training_report.v1"
FOLD_COUNT = 5
MIN_GAIN = 100
MIN_HARM = 100
MIN_CLASS_SCENES = 30
MIN_CROSSING_PRECISION = 0.80
MIN_BASELINE_AP50_GAIN = 0.03
MAX_BASELINE_AP15_AP25_LOSS = 0.005
MIN_POSITIVE_FOLDS = 4
MIN_HARM_REDUCTION = 0.25
MIN_RAW_AP50_EXTRA_GAIN = 0.005


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def scene_fold(scene_id: str) -> int:
    return int(hashlib.sha256(scene_id.encode("utf-8")).hexdigest(), 16) % FOLD_COUNT


def _fit(features: np.ndarray, labels: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    if set(np.unique(labels).tolist()) != set(range(len(CLASS_NAMES))):
        raise ValueError("every training split must contain gain/neutral/harm")
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        fit_intercept=True,
        max_iter=2000,
        random_state=0,
        solver="lbfgs",
        tol=1e-8,
    ).fit(scaler.transform(features), labels)
    if model.n_iter_.max() >= model.max_iter:
        raise RuntimeError("R3 veto logistic regression did not converge")
    if model.classes_.tolist() != list(range(len(CLASS_NAMES))):
        raise ValueError("R3 veto class order changed")
    return scaler, model


def _metrics(rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for threshold in IOU_THRESHOLDS:
        values = scored_detection_metrics(rows, threshold)
        result[f"{threshold:.2f}"] = values
    return result


def _ap(metrics: Mapping[str, Any], threshold: float) -> float:
    return float(metrics[f"{threshold:.2f}"]["average_precision"])


def _write_create_only(path: Path, payload: Mapping[str, Any], label: str) -> None:
    encoded = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"immutable R3 {label} exists: {path}") from error
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def train(dataset_path: Path) -> tuple[R3VetoCalibrator, dict[str, Any]]:
    dataset = load_dataset(dataset_path)
    overlap = dataset.provenance.get("validation_scene_overlap")
    if type(overlap) is not int or overlap != 0:
        raise ValueError("R3 calibration dataset is not train-only")
    scene_folds = np.asarray(
        [scene_fold(str(value)) for value in dataset.scene_ids], dtype=np.int64
    )
    sample_folds = scene_folds[dataset.sample_scene_index]
    observed_folds = set(scene_folds.tolist())
    if observed_folds != set(range(FOLD_COUNT)):
        raise ValueError("train scenes do not cover all five deterministic folds")

    probabilities = np.full(
        (dataset.sample_count, len(CLASS_NAMES)), np.nan, dtype=np.float64
    )
    fold_models: list[dict[str, Any]] = []
    for fold in range(FOLD_COUNT):
        train_mask = sample_folds != fold
        test_mask = sample_folds == fold
        if not np.any(train_mask) or not np.any(test_mask):
            raise ValueError(f"fold {fold} has an empty train/test sample set")
        scaler, model = _fit(dataset.features[train_mask], dataset.labels[train_mask])
        probabilities[test_mask] = model.predict_proba(
            scaler.transform(dataset.features[test_mask])
        )
        fold_models.append(
            {
                "fold": fold,
                "train_scenes": int(np.count_nonzero(scene_folds != fold)),
                "test_scenes": int(np.count_nonzero(scene_folds == fold)),
                "train_samples": int(np.count_nonzero(train_mask)),
                "test_samples": int(np.count_nonzero(test_mask)),
                "iterations": int(model.n_iter_.max()),
            }
        )
    if not np.isfinite(probabilities).all():
        raise RuntimeError("OOF probability matrix is incomplete")
    accepted = probabilities[:, HARM_CLASS] < np.maximum(
        probabilities[:, 0], probabilities[:, 1]
    )
    raw = np.ones(dataset.sample_count, dtype=np.bool_)
    baseline_metrics = _metrics(prediction_rows(dataset, None))
    raw_metrics = _metrics(prediction_rows(dataset, raw))
    veto_metrics = _metrics(prediction_rows(dataset, accepted))

    fold_evaluation: list[dict[str, Any]] = []
    positive_folds = 0
    raw_non_regressing_ap50_folds = 0
    worst_fold_veto_minus_raw_ap15 = float("inf")
    worst_fold_veto_minus_raw_ap25 = float("inf")
    all_rows = prediction_rows(dataset, None)
    raw_rows = prediction_rows(dataset, raw)
    veto_rows = prediction_rows(dataset, accepted)
    for fold in range(FOLD_COUNT):
        keep = {
            str(scene)
            for scene, scene_fold_value in zip(dataset.scene_ids.tolist(), scene_folds.tolist())
            if scene_fold_value == fold
        }
        baseline_fold = _metrics([row for row in all_rows if row[0] in keep])
        raw_fold = _metrics([row for row in raw_rows if row[0] in keep])
        veto_fold = _metrics([row for row in veto_rows if row[0] in keep])
        delta = _ap(veto_fold, 0.50) - _ap(baseline_fold, 0.50)
        positive_folds += int(delta > 0)
        veto_minus_raw = {
            f"AP{int(threshold * 100)}": (
                _ap(veto_fold, threshold) - _ap(raw_fold, threshold)
            )
            for threshold in IOU_THRESHOLDS
        }
        raw_non_regressing_ap50_folds += int(veto_minus_raw["AP50"] >= -1e-12)
        worst_fold_veto_minus_raw_ap15 = min(
            worst_fold_veto_minus_raw_ap15, veto_minus_raw["AP15"]
        )
        worst_fold_veto_minus_raw_ap25 = min(
            worst_fold_veto_minus_raw_ap25, veto_minus_raw["AP25"]
        )
        fold_evaluation.append(
            {
                "fold": fold,
                "baseline": baseline_fold,
                "raw_primary": raw_fold,
                "veto": veto_fold,
                "veto_minus_baseline_ap50": delta,
                "veto_minus_raw": veto_minus_raw,
            }
        )

    labels = dataset.labels.astype(np.int64)
    gains = (dataset.tp_deltas[:, 2] > 0) & accepted
    losses = (dataset.tp_deltas[:, 2] < 0) & accepted
    crossing_total = int(np.count_nonzero(gains) + np.count_nonzero(losses))
    crossing_precision = (
        float(np.count_nonzero(gains) / crossing_total) if crossing_total else 0.0
    )
    raw_harm = int(np.count_nonzero(labels == HARM_CLASS))
    accepted_harm = int(np.count_nonzero((labels == HARM_CLASS) & accepted))
    harm_reduction = (
        float((raw_harm - accepted_harm) / raw_harm) if raw_harm else 0.0
    )
    gain_scene_count = len(
        set(dataset.sample_scene_index[labels == 0].astype(int).tolist())
    )
    harm_scene_count = len(
        set(dataset.sample_scene_index[labels == HARM_CLASS].astype(int).tolist())
    )
    checks = {
        "minimum_gain_samples": int(np.count_nonzero(labels == 0)) >= MIN_GAIN,
        "minimum_harm_samples": raw_harm >= MIN_HARM,
        "minimum_gain_scenes": gain_scene_count >= MIN_CLASS_SCENES,
        "minimum_harm_scenes": harm_scene_count >= MIN_CLASS_SCENES,
        "crossing_precision": crossing_precision >= MIN_CROSSING_PRECISION,
        "baseline_ap50_gain": (
            _ap(veto_metrics, 0.50) - _ap(baseline_metrics, 0.50)
            >= MIN_BASELINE_AP50_GAIN
        ),
        "baseline_ap15_non_regression": (
            _ap(veto_metrics, 0.15) - _ap(baseline_metrics, 0.15)
            >= -MAX_BASELINE_AP15_AP25_LOSS
        ),
        "baseline_ap25_non_regression": (
            _ap(veto_metrics, 0.25) - _ap(baseline_metrics, 0.25)
            >= -MAX_BASELINE_AP15_AP25_LOSS
        ),
        "positive_folds": positive_folds >= MIN_POSITIVE_FOLDS,
        "raw_primary_ap50_non_regression": (
            _ap(veto_metrics, 0.50) >= _ap(raw_metrics, 0.50)
        ),
        "raw_primary_safety_or_extra_gain": (
            harm_reduction >= MIN_HARM_REDUCTION
            or _ap(veto_metrics, 0.50) - _ap(raw_metrics, 0.50)
            >= MIN_RAW_AP50_EXTRA_GAIN
        ),
        "fold_raw_ap50_non_regression": (
            raw_non_regressing_ap50_folds >= MIN_POSITIVE_FOLDS
        ),
        "worst_fold_raw_ap15_loss": (
            worst_fold_veto_minus_raw_ap15 >= -MAX_BASELINE_AP15_AP25_LOSS
        ),
        "worst_fold_raw_ap25_loss": (
            worst_fold_veto_minus_raw_ap25 >= -MAX_BASELINE_AP15_AP25_LOSS
        ),
    }
    gate_pass = all(checks.values())

    scaler, final_model = _fit(dataset.features, dataset.labels)
    dataset_sha = sha256_file(dataset_path)
    report_core = {
        "schema": REPORT_SCHEMA,
        "complete": True,
        "train_only": True,
        "veto_only": True,
        "scene_grouped_oof": True,
        "fold_function": "int(sha256(scene_id),16)%5",
        "model_type": "balanced_multinomial_logistic_regression_C1_lbfgs",
        "model_semantics": "three_class_multinomial_softmax",
        "software_versions": {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "gate": "accept_if_max_gain_or_neutral_probability_strictly_gt_harm",
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_sha,
        "scene_list_sha256": str(dataset.provenance["scene_list_sha256"]),
        "counts": {
            "scenes": dataset.scene_count,
            "samples": dataset.sample_count,
            "gain": int(np.count_nonzero(labels == 0)),
            "safe_neutral": int(np.count_nonzero(labels == 1)),
            "harm": raw_harm,
            "gain_scenes": gain_scene_count,
            "harm_scenes": harm_scene_count,
            "accepted": int(np.count_nonzero(accepted)),
            "vetoed": int(np.count_nonzero(~accepted)),
            "accepted_harm": accepted_harm,
        },
        "baseline": baseline_metrics,
        "raw_primary": raw_metrics,
        "oof_veto": veto_metrics,
        "oof_deltas": {
            f"AP{int(threshold * 100)}": (
                _ap(veto_metrics, threshold) - _ap(baseline_metrics, threshold)
            )
            for threshold in IOU_THRESHOLDS
        },
        "oof_veto_minus_raw": {
            f"AP{int(threshold * 100)}": (
                _ap(veto_metrics, threshold) - _ap(raw_metrics, threshold)
            )
            for threshold in IOU_THRESHOLDS
        },
        "crossing_precision": crossing_precision,
        "harm_reduction": harm_reduction,
        "positive_folds": positive_folds,
        "raw_non_regressing_ap50_folds": raw_non_regressing_ap50_folds,
        "worst_fold_veto_minus_raw": {
            "AP15": worst_fold_veto_minus_raw_ap15,
            "AP25": worst_fold_veto_minus_raw_ap25,
        },
        "checks": checks,
        "gate_pass": gate_pass,
        "fold_training": fold_models,
        "fold_evaluation": fold_evaluation,
        "training_overlap_disclosure": {
            "tr3d_checkpoint_training_overlap": bool(
                dataset.provenance.get("tr3d_checkpoint_training_overlap")
            ),
            "independent_calibration_proof": bool(
                dataset.provenance.get("independent_calibration_proof")
            ),
            "formal_training_free_claim_permitted": False,
            "formal_independent_activation_authorized": False,
        },
        "thresholds": {
            "min_gain": MIN_GAIN,
            "min_harm": MIN_HARM,
            "min_class_scenes": MIN_CLASS_SCENES,
            "min_crossing_precision": MIN_CROSSING_PRECISION,
            "min_baseline_ap50_gain": MIN_BASELINE_AP50_GAIN,
            "max_baseline_ap15_ap25_loss": MAX_BASELINE_AP15_AP25_LOSS,
            "min_positive_folds": MIN_POSITIVE_FOLDS,
            "min_harm_reduction": MIN_HARM_REDUCTION,
            "min_raw_ap50_extra_gain": MIN_RAW_AP50_EXTRA_GAIN,
        },
    }
    model = R3VetoCalibrator(
        feature_mean=np.asarray(scaler.mean_, dtype=np.float64),
        feature_scale=np.asarray(scaler.scale_, dtype=np.float64),
        coefficients=np.asarray(final_model.coef_, dtype=np.float64),
        intercept=np.asarray(final_model.intercept_, dtype=np.float64),
        activation_authorized=gate_pass,
        dataset_sha256=dataset_sha,
        scene_list_sha256=str(dataset.provenance["scene_list_sha256"]),
        metadata={
            "train_gate_pass": gate_pass,
            "independent_calibration_proof": bool(
                dataset.provenance.get("independent_calibration_proof")
            ),
            "tr3d_checkpoint_training_overlap": bool(
                dataset.provenance.get("tr3d_checkpoint_training_overlap")
            ),
            "training_code_sha256": code_artifact_tree_sha256(
                (
                    Path(__file__),
                    _ROOT / "boxfusion" / "tr3d_r3_calibrator.py",
                    _ROOT / "boxfusion" / "tr3d_r3_calibration_dataset.py",
                    _ROOT / "tools" / "audit_tr3d_r3_near_correction.py",
                )
            ),
            "formal_independent_activation_authorized": False,
            "software_versions": {
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "inference_lineage_contract": dict(
                dataset.provenance.get("inference_lineage_contract", {})
            ),
        },
    ).validate()
    return model, report_core


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_path = args.dataset.resolve()
    model_path = args.model.expanduser().absolute()
    report_path = args.report.expanduser().absolute()
    if model_path.exists() or report_path.exists():
        raise FileExistsError("R3 calibrator model/report namespace already exists")
    model, report = train(dataset_path)
    _write_create_only(model_path, model.as_dict(), "calibrator")
    report["model_path"] = str(model_path)
    report["model_sha256"] = sha256_file(model_path)
    _write_create_only(report_path, report, "training report")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
