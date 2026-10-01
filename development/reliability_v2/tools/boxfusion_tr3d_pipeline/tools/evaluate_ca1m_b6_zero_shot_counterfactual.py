#!/usr/bin/env python3
"""Evaluate one frozen ScanNet-B6 score counterfactual on CA-1M.

The tool never mutates observer or P1 predictions.  It creates an explicitly
non-active score-only prediction tree, evaluates P1 and the counterfactual with
the same CA-1M OBB evaluator, and records the cross-domain feature statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.quality_score import (
    QUALITY_FEATURE_NAMES,
    load_quality_scorer,
)
from boxfusion.online_refinement import corners_to_center_size


METRIC_PATTERN = re.compile(r"^eval (mAP|APrec|ARecall): ([0-9]+(?:\.[0-9]+)?)$", re.M)
THRESHOLDS = (0.15, 0.25, 0.50)
EXPECTED_DIAGNOSTIC_KEYS = {
    "scene_id",
    "boxes",
    "scores",
    "quality_features",
    "points",
    "point_mask",
    "source_indices",
    "track_ids",
    "result_indices",
    "labels",
    "quality_feature_names",
    "summary_json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list must be non-empty and unique")
    return scenes


def load_prediction(path: Path) -> list[tuple[int, np.ndarray, float]]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
        if handle.read(1):
            raise ValueError(f"trailing bytes in prediction: {path}")
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], list):
        raise ValueError(f"prediction must contain exactly one list batch: {path}")
    rows: list[tuple[int, np.ndarray, float]] = []
    for row_index, row in enumerate(value[0]):
        if not isinstance(row, tuple) or len(row) != 3:
            raise ValueError(f"invalid prediction row {row_index}: {path}")
        label, corners, score = row
        corners = np.asarray(corners)
        score = float(score)
        if int(label) != 0 or corners.shape != (8, 3):
            raise ValueError(f"invalid class-agnostic OBB row {row_index}: {path}")
        if not np.isfinite(corners).all() or not np.isfinite(score):
            raise ValueError(f"non-finite prediction row {row_index}: {path}")
        rows.append((0, corners.copy(), score))
    return rows


def save_prediction_atomic(path: Path, rows: list[tuple[int, np.ndarray, float]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing existing counterfactual prediction: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        pickle.dump([rows], handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_diagnostic(
    path: Path,
    scene: str,
    prediction: list[tuple[int, np.ndarray, float]],
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != EXPECTED_DIAGNOSTIC_KEYS:
            raise ValueError(f"{scene}: diagnostic schema disagrees")
        if str(np.asarray(payload["scene_id"]).item()) != scene:
            raise ValueError(f"{scene}: diagnostic scene_id disagrees")
        names = tuple(str(value) for value in payload["quality_feature_names"].tolist())
        if names != QUALITY_FEATURE_NAMES:
            raise ValueError(f"{scene}: B6 feature names/order disagree")
        result_indices = np.asarray(payload["result_indices"])
        source_indices = np.asarray(payload["source_indices"])
        boxes = np.asarray(payload["boxes"])
        features = np.asarray(payload["quality_features"], dtype=np.float64)
        scores = np.asarray(payload["scores"], dtype=np.float64)
        if result_indices.dtype.kind not in "iu" or result_indices.ndim != 1:
            raise ValueError(f"{scene}: invalid result_indices")
        result_indices = result_indices.astype(np.int64, copy=False)
        if len(result_indices) and (
            np.any(np.diff(result_indices) <= 0)
            or result_indices[0] < 0
            or result_indices[-1] >= len(prediction)
        ):
            raise ValueError(f"{scene}: result_indices are not strict in-range indices")
        if source_indices.dtype.kind not in "iu" or not np.array_equal(
            source_indices.astype(np.int64, copy=False), result_indices
        ):
            raise ValueError(f"{scene}: no-op source/result indices disagree")
        if boxes.dtype != np.float32 or boxes.shape != (len(result_indices), 6):
            raise ValueError(f"{scene}: invalid diagnostic box tensor")
        if len(result_indices):
            corners = np.stack(
                [prediction[int(index)][1] for index in result_indices], axis=0
            ).astype(np.float32, copy=False)
        else:
            corners = np.empty((0, 8, 3), dtype=np.float32)
        expected_boxes = corners_to_center_size(corners)
        if not np.array_equal(boxes, expected_boxes):
            raise ValueError(f"{scene}: diagnostic boxes do not map to anchor OBB rows")
        if features.shape != (len(result_indices), len(QUALITY_FEATURE_NAMES)):
            raise ValueError(f"{scene}: invalid B6 feature tensor")
        if scores.shape != (len(result_indices),):
            raise ValueError(f"{scene}: invalid diagnostic scores")
        if not np.isfinite(features).all() or np.any(features < 0) or np.any(features > 1):
            raise ValueError(f"{scene}: B6 features must be finite in [0,1]")
        expected_scores = np.asarray(
            [prediction[int(index)][2] for index in result_indices], dtype=scores.dtype
        )
        if not np.array_equal(scores, expected_scores):
            raise ValueError(f"{scene}: diagnostic scores do not map to anchor rows")
        if len(result_indices) and not np.array_equal(
            features[:, 0], scores.astype(features.dtype)
        ):
            raise ValueError(f"{scene}: detector-score feature disagrees")
        summary = json.loads(str(np.asarray(payload["summary_json"]).item()))
        if not isinstance(summary, dict) or not summary.get("enabled", False):
            raise ValueError(f"{scene}: invalid diagnostic summary")
        for key in ("supplemental_output", "refits_accepted", "neural_refits_accepted"):
            if int(summary.get(key, -1)) != 0:
                raise ValueError(f"{scene}: diagnostic mutation counter {key} is nonzero")
        return {
            "result_indices": result_indices.copy(),
            "features": features.copy(),
            "scores": scores.copy(),
        }


def rank_displacement(before: np.ndarray, after: np.ndarray) -> dict[str, float | int]:
    if before.shape != after.shape:
        raise ValueError("rank score vectors disagree")
    if len(before) == 0:
        return {"changed_positions": 0, "mean_abs_displacement": 0.0, "max_abs_displacement": 0}
    old_order = np.argsort(-before, kind="mergesort")
    new_order = np.argsort(-after, kind="mergesort")
    old_position = np.empty(len(before), dtype=np.int64)
    new_position = np.empty(len(after), dtype=np.int64)
    old_position[old_order] = np.arange(len(before))
    new_position[new_order] = np.arange(len(after))
    displacement = np.abs(new_position - old_position)
    return {
        "changed_positions": int(np.count_nonzero(displacement)),
        "mean_abs_displacement": float(displacement.mean()),
        "max_abs_displacement": int(displacement.max()),
    }


def parse_metrics(text: str) -> dict[str, dict[str, float]]:
    rows = METRIC_PATTERN.findall(text)
    if len(rows) != len(THRESHOLDS) * 3:
        raise ValueError(f"expected 9 evaluator metrics, found {len(rows)}")
    output: dict[str, dict[str, float]] = {}
    for threshold_index, threshold in enumerate(THRESHOLDS):
        chunk = rows[threshold_index * 3 : (threshold_index + 1) * 3]
        values = {name: float(value) for name, value in chunk}
        if set(values) != {"mAP", "APrec", "ARecall"}:
            raise ValueError("evaluator metric order/schema disagrees")
        output[f"{threshold:.2f}"] = values
    return output


def run_evaluator(
    *,
    python: Path,
    evaluation_dir: Path,
    data_root: Path,
    prediction_root: Path,
    log_path: Path,
    tmp_root: Path,
    gpu: int,
) -> dict[str, dict[str, float]]:
    if log_path.exists():
        raise FileExistsError(f"refusing existing evaluation log: {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        "eval_ca1m.py",
        "--dataset",
        "ca1m",
        "--data_path",
        str(data_root),
        "--pred_root",
        str(prediction_root),
        "--ap_iou_thresholds",
        "0.15,0.25,0.5",
        "--num_workers",
        "0",
        "--cluster_sampling",
        "seed_fps",
        "--use_3d_nms",
        "--use_cls_nms",
        "--per_class_proposal",
        "--gpu",
        str(gpu),
    ]
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "TMPDIR": str(tmp_root),
            "TMP": str(tmp_root),
            "TEMP": str(tmp_root),
        }
    )
    process = subprocess.run(
        command,
        cwd=evaluation_dir,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.write_text(process.stdout)
    if process.returncode != 0:
        raise RuntimeError(
            f"CA-1M evaluator failed ({process.returncode}); inspect {log_path}"
        )
    return parse_metrics(process.stdout)


def write_json_atomic(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def feature_shift_report(features: np.ndarray, scorer: Any) -> dict[str, Any]:
    if len(features) == 0:
        return {"rows": 0, "per_feature": {}}
    mean = np.asarray(scorer.feature_mean, dtype=np.float64)
    scale = np.asarray(scorer.feature_scale, dtype=np.float64)
    z = (features - mean) / scale
    result = {}
    for index, name in enumerate(QUALITY_FEATURE_NAMES):
        column = features[:, index]
        z_column = np.abs(z[:, index])
        result[name] = {
            "mean": float(column.mean()),
            "std": float(column.std()),
            "z_abs_q50": float(np.quantile(z_column, 0.50)),
            "z_abs_q90": float(np.quantile(z_column, 0.90)),
            "outside_3sigma_fraction": float(np.mean(z_column > 3.0)),
        }
    return {"rows": len(features), "per_feature": result}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--scene-list", type=Path, required=True)
    result.add_argument("--anchor-root", type=Path, required=True)
    result.add_argument("--historical-anchor-root", type=Path, required=True)
    result.add_argument("--observer-root", type=Path, required=True)
    result.add_argument("--diagnostics-root", type=Path, required=True)
    result.add_argument("--identity-audit", type=Path, required=True)
    result.add_argument("--quality-checkpoint", type=Path, required=True)
    result.add_argument("--expected-quality-sha256", required=True)
    result.add_argument("--detector-blend", type=float, default=0.40)
    result.add_argument("--counterfactual-root", type=Path, required=True)
    result.add_argument("--eval-data-root", type=Path, required=True)
    result.add_argument("--evaluation-dir", type=Path, required=True)
    result.add_argument("--python", type=Path, required=True)
    result.add_argument("--log-root", type=Path, required=True)
    result.add_argument("--tmp-root", type=Path, required=True)
    result.add_argument("--gpu", type=int, default=0)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    if not 0.0 <= args.detector_blend <= 1.0 or not np.isfinite(args.detector_blend):
        raise ValueError("detector blend must be finite in [0,1]")
    if args.output.exists() or args.log_root.exists() or args.tmp_root.exists():
        raise FileExistsError(
            "refusing existing counterfactual report/log/tmp namespace"
        )
    if args.counterfactual_root.exists():
        raise FileExistsError(
            f"refusing existing counterfactual root: {args.counterfactual_root}"
        )
    for path in (
        args.anchor_root,
        args.historical_anchor_root,
        args.observer_root,
        args.diagnostics_root,
        args.eval_data_root,
        args.evaluation_dir,
    ):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if not args.python.is_file() or not args.quality_checkpoint.is_file():
        raise FileNotFoundError("python or B6 checkpoint is missing")
    checkpoint_sha = sha256(args.quality_checkpoint)
    if checkpoint_sha != args.expected_quality_sha256:
        raise ValueError(
            f"B6 checkpoint SHA disagrees: {checkpoint_sha} != "
            f"{args.expected_quality_sha256}"
        )
    scorer = load_quality_scorer(args.quality_checkpoint, method="iou_mlp")
    scenes = read_scenes(args.scene_list)
    if not args.identity_audit.is_file():
        raise FileNotFoundError(args.identity_audit)
    identity = json.loads(args.identity_audit.read_text())
    if not isinstance(identity, dict):
        raise ValueError("identity audit must be a JSON object")
    identity_per_scene = identity.get("per_scene")
    identity_contract = {
        "schema": identity.get("schema")
        == "boxfusion.ca1m_c2_b6_zero_shot_identity.v1",
        "ok": identity.get("ok") is True,
        "output_mutation_authorized": identity.get("output_mutation_authorized")
        is False,
        "dataset": identity.get("dataset") == "CA1M",
        "scenes": identity.get("scenes") == len(scenes),
        "scene_list_sha256": identity.get("scene_list_sha256") == sha256(args.scene_list),
        "quality_checkpoint_sha256": identity.get("quality_checkpoint_sha256")
        == checkpoint_sha,
        "identity_anchor_contract": identity.get("identity_anchor_contract")
        == "same_run_pre_online_finalize",
        "historical_p1_role": identity.get("historical_p1_role")
        == "replay_drift_and_metric_reference_only",
        "per_scene": isinstance(identity_per_scene, dict)
        and set(identity_per_scene) == set(scenes),
    }
    failed_identity = [key for key, value in identity_contract.items() if not value]
    if failed_identity:
        raise ValueError(
            "identity audit contract disagrees: " + ", ".join(failed_identity)
        )
    eval_entries = {path.name for path in args.eval_data_root.iterdir()}
    if eval_entries != set(scenes):
        raise ValueError("evaluation view is not the exact scene list")
    historical_names = {
        path.name
        for path in args.historical_anchor_root.glob("*_boxes.pkl")
        if path.is_file()
    }
    expected_historical_names = {f"{scene}_boxes.pkl" for scene in scenes}
    if historical_names != expected_historical_names:
        raise ValueError("historical P1 prediction set is not the exact scene list")

    # The parent is created here, then every prediction is still written
    # atomically and create-only by ``save_prediction_atomic``.
    args.counterfactual_root.mkdir(parents=True, exist_ok=False)
    try:
        scene_reports: dict[str, Any] = {}
        feature_rows: list[np.ndarray] = []
        quality_rows: list[np.ndarray] = []
        total_predictions = 0
        total_observed = 0
        for scene in scenes:
            anchor_path = args.anchor_root / f"{scene}_boxes.pkl"
            observer_path = args.observer_root / f"{scene}_boxes.pkl"
            diagnostic_path = args.diagnostics_root / f"{scene}_tracks.npz"
            if not anchor_path.is_file() or not observer_path.is_file() or not diagnostic_path.is_file():
                raise FileNotFoundError(f"missing anchor/observer/diagnostic for {scene}")
            rows = load_prediction(anchor_path)
            observer_rows = load_prediction(observer_path)
            if len(rows) != len(observer_rows) or any(
                left[0] != right[0]
                or not np.array_equal(left[1], right[1])
                or left[2] != right[2]
                for left, right in zip(rows, observer_rows)
            ):
                raise ValueError(f"{scene}: current observer is not anchor-identical")
            identity_scene = identity_per_scene[scene]
            identity_hashes = {
                "anchor_prediction_sha256": sha256(anchor_path),
                "observer_prediction_sha256": sha256(observer_path),
                "diagnostic_sha256": sha256(diagnostic_path),
            }
            for key, value in identity_hashes.items():
                if identity_scene.get(key) != value:
                    raise ValueError(f"{scene}: identity audit {key} is stale")
            historical_path = args.historical_anchor_root / f"{scene}_boxes.pkl"
            historical_drift = identity_scene.get("historical_anchor_drift")
            if (
                not isinstance(historical_drift, dict)
                or historical_drift.get("informational_only") is not True
                or historical_drift.get("historical_prediction_sha256")
                != sha256(historical_path)
            ):
                raise ValueError(
                    f"{scene}: historical P1 identity binding is stale"
                )
            if identity_scene.get("semantic_identity") is not True or identity_scene.get(
                "boxer", {}
            ).get("deterministic_fields_identity") is not True:
                raise ValueError(f"{scene}: identity/Boxer audit did not pass")
            diagnostic = load_diagnostic(diagnostic_path, scene, rows)
            original_scores = np.asarray([row[2] for row in rows], dtype=np.float64)
            indices = diagnostic["result_indices"]
            predictions = scorer.predict(diagnostic["features"])
            ranking = np.asarray(predictions["ranking_score"], dtype=np.float64)
            if ranking.shape != (len(indices),) or not np.isfinite(ranking).all():
                raise ValueError(f"{scene}: invalid ScanNet-B6 ranking output")
            if np.any(ranking < 0.0) or np.any(ranking > 1.0):
                raise ValueError(f"{scene}: ScanNet-B6 ranking output lies outside [0,1]")
            counterfactual_scores = original_scores.copy()
            counterfactual_scores[indices] = (
                args.detector_blend * original_scores[indices]
                + (1.0 - args.detector_blend) * ranking
            )
            counterfactual_rows = [
                (label, corners.copy(), float(counterfactual_scores[index]))
                for index, (label, corners, _) in enumerate(rows)
            ]
            output_path = args.counterfactual_root / f"{scene}_boxes.pkl"
            save_prediction_atomic(output_path, counterfactual_rows)
            # Re-read the artifact before accepting it.
            reloaded = load_prediction(output_path)
            for row_index, (source, target) in enumerate(zip(rows, reloaded)):
                if source[0] != target[0] or not np.array_equal(source[1], target[1]):
                    raise ValueError(f"{scene}: geometry changed at row {row_index}")
            components = {
                key: np.asarray(value, dtype=np.float64)
                for key, value in predictions.items()
            }
            scene_reports[scene] = {
                "prediction_rows": len(rows),
                "observed_rows": len(indices),
                "coverage": float(len(indices) / len(rows)) if rows else 0.0,
                "rank": rank_displacement(original_scores, counterfactual_scores),
                "score_delta_min": float(np.min(counterfactual_scores - original_scores)) if rows else 0.0,
                "score_delta_max": float(np.max(counterfactual_scores - original_scores)) if rows else 0.0,
                "score_delta_mean": float(np.mean(counterfactual_scores - original_scores)) if rows else 0.0,
                "b6_components_mean": {
                    key: float(value.mean()) if len(value) else 0.0
                    for key, value in components.items()
                },
                "prediction_sha256": sha256(output_path),
            }
            total_predictions += len(rows)
            total_observed += len(indices)
            feature_rows.append(diagnostic["features"])
            quality_rows.append(ranking)
    except Exception:
        # Fail closed without deleting evidence: move partial materialization
        # out of the formal name so a retry cannot silently mix artifacts.
        quarantined = args.counterfactual_root.with_name(
            args.counterfactual_root.name + f".failed.{os.getpid()}"
        )
        os.replace(args.counterfactual_root, quarantined)
        raise

    anchor_log = args.log_root / "eval_anchor_p1.log"
    historical_log = args.log_root / "eval_historical_p1.log"
    counterfactual_log = args.log_root / "eval_b6_zero_shot_counterfactual.log"
    historical_metrics = run_evaluator(
        python=args.python,
        evaluation_dir=args.evaluation_dir,
        data_root=args.eval_data_root,
        prediction_root=args.historical_anchor_root,
        log_path=historical_log,
        tmp_root=args.tmp_root / "historical",
        gpu=args.gpu,
    )
    anchor_metrics = run_evaluator(
        python=args.python,
        evaluation_dir=args.evaluation_dir,
        data_root=args.eval_data_root,
        prediction_root=args.anchor_root,
        log_path=anchor_log,
        tmp_root=args.tmp_root / "anchor",
        gpu=args.gpu,
    )
    counterfactual_metrics = run_evaluator(
        python=args.python,
        evaluation_dir=args.evaluation_dir,
        data_root=args.eval_data_root,
        prediction_root=args.counterfactual_root,
        log_path=counterfactual_log,
        tmp_root=args.tmp_root / "counterfactual",
        gpu=args.gpu,
    )
    delta = {
        threshold: {
            key: counterfactual_metrics[threshold][key] - anchor_metrics[threshold][key]
            for key in ("mAP", "APrec", "ARecall")
        }
        for threshold in anchor_metrics
    }
    historical_to_same_run_delta = {
        threshold: {
            key: anchor_metrics[threshold][key] - historical_metrics[threshold][key]
            for key in ("mAP", "APrec", "ARecall")
        }
        for threshold in anchor_metrics
    }
    all_features = (
        np.concatenate(feature_rows, axis=0)
        if feature_rows
        else np.empty((0, len(QUALITY_FEATURE_NAMES)), dtype=np.float64)
    )
    all_quality = (
        np.concatenate(quality_rows, axis=0)
        if quality_rows
        else np.empty((0,), dtype=np.float64)
    )
    coverage = float(total_observed / total_predictions) if total_predictions else 0.0
    gate = {
        "identity_required_before_use": all(identity_contract.values()),
        "coverage_at_least_0_60": coverage >= 0.60,
        "delta_ap15_at_least_minus_0_003": delta["0.15"]["mAP"] >= -0.003,
        "delta_ap25_at_least_0_005": delta["0.25"]["mAP"] >= 0.005,
        "delta_ap50_at_least_0_005": delta["0.50"]["mAP"] >= 0.005,
    }
    gate["counterfactual_pass"] = all(gate.values())
    report = {
        "schema": "boxfusion.ca1m_b6_zero_shot_counterfactual.v1",
        "dataset": "CA1M",
        "status": "diagnostic_non_authoritative",
        "active_materialization_authorized": False,
        "target_dataset_training_used": False,
        "source_supervision": "ScanNet train-only B6 checkpoint",
        "obb_geometry_unchanged": True,
        "feature_geometry_contract": "world_obb_to_enclosing_world_aabb_proxy",
        "scene_list": str(args.scene_list.resolve()),
        "scene_list_sha256": sha256(args.scene_list),
        "identity_audit": str(args.identity_audit.resolve()),
        "identity_audit_sha256": sha256(args.identity_audit),
        "scenes": len(scenes),
        "quality_checkpoint": str(args.quality_checkpoint.resolve()),
        "quality_checkpoint_sha256": checkpoint_sha,
        "detector_blend": args.detector_blend,
        "quality_blend": 1.0 - args.detector_blend,
        "prediction_rows": total_predictions,
        "observed_rows": total_observed,
        "coverage": coverage,
        "b6_ranking_mean": float(all_quality.mean()) if len(all_quality) else 0.0,
        "b6_ranking_std": float(all_quality.std()) if len(all_quality) else 0.0,
        "feature_domain_shift": feature_shift_report(all_features, scorer),
        "historical_p1_metrics": historical_metrics,
        "same_run_anchor_metrics": anchor_metrics,
        "historical_to_same_run_delta": historical_to_same_run_delta,
        "counterfactual_metrics": counterfactual_metrics,
        "delta": delta,
        "pre_registered_gate": gate,
        "per_scene": scene_reports,
        "logs": {
            "historical_p1": str(historical_log.resolve()),
            "anchor": str(anchor_log.resolve()),
            "counterfactual": str(counterfactual_log.resolve()),
        },
    }
    write_json_atomic(args.output, report)
    print(json.dumps({"coverage": coverage, "delta": delta, "gate": gate}, indent=2, sort_keys=True))
    print(f"CA-1M B6 zero-shot counterfactual completed: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
