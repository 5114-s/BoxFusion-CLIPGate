#!/usr/bin/env python3
"""Train and authorize a scene-grouped, train-only C3 source gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.tr3d_c3_online_active import FEATURE_NAMES, POLICY_SCHEMA
from boxfusion.tr3d_c3_online_identity import PARENT_SCORE_ROUTE, ROUTE
from boxfusion.tr3d_c2_maskrgbd_cache import sha256_file
from tools.build_tr3d_c3_source_gate_dataset import DATASET_SCHEMA, read_scenes


def fold_for_scene(scene_id: str, folds: int) -> int:
    digest = hashlib.sha256(scene_id.encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    target_iou: np.ndarray,
    *,
    iterations: int,
    learning_rate: float,
    l2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    values = (features - mean) / scale
    positives = max(int(np.count_nonzero(labels)), 1)
    negatives = max(len(labels) - positives, 1)
    weights = np.where(labels > 0.5, len(labels) / (2.0 * positives), len(labels) / (2.0 * negatives))
    weights *= np.where(target_iou >= 0.50, 2.0, 1.0)
    coefficients = np.zeros(features.shape[1], dtype=np.float64)
    bias = 0.0
    normalizer = float(np.sum(weights))
    for step in range(iterations):
        probability = sigmoid(values @ coefficients + bias)
        error = (probability - labels) * weights
        rate = learning_rate / math.sqrt(1.0 + step / 200.0)
        coefficients -= rate * ((values.T @ error) / normalizer + l2 * coefficients)
        bias -= rate * float(np.sum(error) / normalizer)
    return coefficients, mean, scale, bias


def precision(mask: np.ndarray, targets: np.ndarray, threshold: float) -> float:
    count = int(np.count_nonzero(mask))
    return float(np.count_nonzero(mask & (targets >= threshold)) / count) if count else 0.0


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"refusing existing C3 policy: {path}") from error
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def train(args: argparse.Namespace) -> dict:
    dataset_path = args.dataset.resolve()
    if not dataset_path.is_file() or dataset_path.is_symlink() or dataset_path.stat().st_mode & 0o022:
        raise ValueError("C3 dataset must be an immutable regular file")
    with np.load(dataset_path, allow_pickle=False) as archive:
        expected = {
            "features", "target_iou", "scene_ids", "proposal_ids", "feature_names",
            "schema", "route", "train_scene_list_sha256",
            "forbidden_validation_scene_list_sha256",
        }
        if set(archive.files) != expected:
            raise ValueError(f"unexpected C3 dataset keys: {sorted(set(archive.files) ^ expected)}")
        if str(np.asarray(archive["schema"]).item()) != DATASET_SCHEMA:
            raise ValueError("unsupported C3 dataset schema")
        dataset_route = str(np.asarray(archive["route"]).item())
        if dataset_route not in {ROUTE, PARENT_SCORE_ROUTE}:
            raise ValueError("C3 dataset route mismatch")
        features = np.asarray(archive["features"], dtype=np.float64)
        target_iou = np.asarray(archive["target_iou"], dtype=np.float64)
        scene_ids = np.asarray(archive["scene_ids"]).astype(str)
        names = tuple(str(item) for item in np.asarray(archive["feature_names"]).tolist())
        train_list_sha = str(np.asarray(archive["train_scene_list_sha256"]).item())
        forbidden_list_sha = str(np.asarray(archive["forbidden_validation_scene_list_sha256"]).item())
    if names != FEATURE_NAMES or features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError("C3 dataset feature schema mismatch")
    if target_iou.shape != (len(features),) or scene_ids.shape != (len(features),):
        raise ValueError("C3 dataset row alignment mismatch")
    if not np.isfinite(features).all() or not np.isfinite(target_iou).all() or np.any((target_iou < 0) | (target_iou > 1)):
        raise ValueError("C3 dataset has invalid numeric values")

    train_scenes = read_scenes(args.train_scene_list.resolve())
    forbidden = read_scenes(args.forbidden_validation_scene_list.resolve())
    if sha256_file(args.train_scene_list) != train_list_sha or sha256_file(args.forbidden_validation_scene_list) != forbidden_list_sha:
        raise ValueError("C3 dataset scene-list provenance mismatch")
    if set(scene_ids.tolist()) - set(train_scenes) or set(train_scenes) & set(forbidden) or len(forbidden) < 100:
        raise ValueError("C3 dataset contains forbidden or unknown scenes")

    labels = (target_iou >= 0.25).astype(np.float64)
    if np.count_nonzero(labels) < 10 or np.count_nonzero(labels == 0) < 10:
        raise ValueError("C3 dataset needs at least ten positive and ten negative rows")
    fold_ids = np.asarray([fold_for_scene(scene, args.folds) for scene in scene_ids], dtype=np.int32)
    oof = np.full(len(features), np.nan, dtype=np.float64)
    per_fold: list[dict] = []
    for fold in range(args.folds):
        validation = fold_ids == fold
        training = ~validation
        if not np.any(validation) or len(np.unique(labels[training])) != 2:
            raise ValueError(f"scene-group fold {fold} is empty or single-class")
        coefficients, mean, scale, bias = fit_logistic(
            features[training], labels[training], target_iou[training],
            iterations=args.iterations, learning_rate=args.learning_rate, l2=args.l2,
        )
        oof[validation] = sigmoid(((features[validation] - mean) / scale) @ coefficients + bias)
        per_fold.append({
            "fold": fold,
            "scenes": len(set(scene_ids[validation].tolist())),
            "samples": int(np.count_nonzero(validation)),
            "positive_iou25": int(np.count_nonzero(target_iou[validation] >= 0.25)),
            "positive_iou50": int(np.count_nonzero(target_iou[validation] >= 0.50)),
        })
    if not np.isfinite(oof).all():
        raise RuntimeError("C3 OOF prediction coverage is incomplete")

    choices: list[dict] = []
    for threshold in np.linspace(0.05, 0.95, 181):
        selected = oof >= threshold
        count = int(np.count_nonzero(selected))
        positive_folds = sum(bool(np.any(selected & (fold_ids == fold))) for fold in range(args.folds))
        row = {
            "threshold": float(threshold),
            "selected": count,
            "precision_iou25": precision(selected, target_iou, 0.25),
            "precision_iou50": precision(selected, target_iou, 0.50),
            "positive_folds": positive_folds,
        }
        if (
            count >= args.min_selected
            and row["precision_iou25"] >= args.min_precision_iou25
            and row["precision_iou50"] >= args.min_precision_iou50
            and positive_folds >= args.min_positive_folds
        ):
            choices.append(row)
    authorized = bool(choices)
    chosen = max(choices, key=lambda row: (row["selected"], row["precision_iou50"], -row["threshold"])) if choices else {
        "threshold": 0.95,
        "selected": int(np.count_nonzero(oof >= 0.95)),
        "precision_iou25": precision(oof >= 0.95, target_iou, 0.25),
        "precision_iou50": precision(oof >= 0.95, target_iou, 0.50),
        "positive_folds": sum(bool(np.any((oof >= 0.95) & (fold_ids == fold))) for fold in range(args.folds)),
    }
    coefficients, mean, scale, bias = fit_logistic(
        features, labels, target_iou,
        iterations=args.iterations, learning_rate=args.learning_rate, l2=args.l2,
    )
    payload = {
        "schema": POLICY_SCHEMA,
        "complete": True,
        "activation_authorized": authorized,
        "train_only": True,
        "scene_group_oof": True,
        "ground_truth_used_only_for_training": True,
        "validation_predictions_used_for_training": False,
        "dataset": "scannet",
        "route": dataset_route,
        "feature_names": list(FEATURE_NAMES),
        "weights": coefficients.tolist(),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "bias": float(bias),
        "probability_threshold": float(chosen["threshold"]),
        "output_score_min": args.output_score_min,
        "output_score_max": args.output_score_max,
        "max_candidates_per_scene": args.max_candidates_per_scene,
        "max_anchor_iou": args.max_anchor_iou,
        "training_scene_ids": list(train_scenes),
        "forbidden_validation_scene_ids": list(forbidden),
        "validation_overlap_count": 0,
        "training_scene_list_sha256": train_list_sha,
        "training_data_sha256": sha256_file(dataset_path),
        "training_data": str(dataset_path),
        "forbidden_validation_scene_list_sha256": forbidden_list_sha,
        "oof": {
            "folds": args.folds,
            "selection": chosen,
            "requirements": {
                "min_selected": args.min_selected,
                "min_precision_iou25": args.min_precision_iou25,
                "min_precision_iou50": args.min_precision_iou50,
                "min_positive_folds": args.min_positive_folds,
            },
            "per_fold": per_fold,
        },
    }
    atomic_json(args.output.resolve(), payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--train-scene-list", type=Path, required=True)
    parser.add_argument("--forbidden-validation-scene-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=1600)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.002)
    parser.add_argument("--min-selected", type=int, default=20)
    parser.add_argument("--min-precision-iou25", type=float, default=0.60)
    parser.add_argument("--min-precision-iou50", type=float, default=0.25)
    parser.add_argument("--min-positive-folds", type=int, default=4)
    parser.add_argument("--output-score-min", type=float, default=0.02)
    parser.add_argument("--output-score-max", type=float, default=0.39)
    parser.add_argument("--max-candidates-per-scene", type=int, default=8)
    parser.add_argument("--max-anchor-iou", type=float, default=0.15)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = train(arguments)
    print(json.dumps({
        "activation_authorized": result["activation_authorized"],
        "oof": result["oof"],
        "output": str(arguments.output.resolve()),
    }, indent=2, sort_keys=True))
