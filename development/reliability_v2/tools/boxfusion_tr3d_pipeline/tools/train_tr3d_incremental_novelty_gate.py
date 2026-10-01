#!/usr/bin/env python3
"""Train a scene-grouped novelty gate from train-only incremental tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.tr3d_incremental_gate import (  # noqa: E402
    DATASET_SCHEMA, FEATURE_NAMES, POLICY_SCHEMA,
)
from tools.build_tr3d_c3_source_gate_dataset import read_scenes  # noqa: E402
from tools.build_tr3d_incremental_novelty_dataset import sha256  # noqa: E402
from tools.train_tr3d_c3_source_gate import (  # noqa: E402
    atomic_json, fit_logistic, fold_for_scene, sigmoid,
)


def _precision(selected: np.ndarray, labels: np.ndarray) -> float:
    count = int(selected.sum())
    return float(np.count_nonzero(selected & labels) / count) if count else 0.0


def _recall(selected: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    return float(np.count_nonzero(selected & labels) / positives) if positives else 0.0


def train(args: argparse.Namespace) -> dict:
    dataset = args.dataset.resolve()
    if not dataset.is_file() or dataset.is_symlink() or dataset.stat().st_mode & 0o022:
        raise ValueError("dataset must be an immutable regular file")
    with np.load(dataset, allow_pickle=False) as archive:
        expected = {"features", "target_iou", "novel_iou15", "novel_iou25",
                    "novel_iou50", "scene_ids", "track_ids", "feature_names",
                    "schema", "train_scene_list_sha256",
                    "forbidden_validation_scene_list_sha256"}
        if set(archive.files) != expected:
            raise ValueError(f"unexpected dataset keys: {sorted(set(archive.files) ^ expected)}")
        if str(np.asarray(archive["schema"]).item()) != DATASET_SCHEMA:
            raise ValueError("unsupported incremental novelty dataset")
        x = np.asarray(archive["features"], dtype=np.float64)
        iou = np.asarray(archive["target_iou"], dtype=np.float64)
        n15 = np.asarray(archive["novel_iou15"], dtype=np.bool_)
        n25 = np.asarray(archive["novel_iou25"], dtype=np.bool_)
        n50 = np.asarray(archive["novel_iou50"], dtype=np.bool_)
        scene_ids = np.asarray(archive["scene_ids"]).astype(str)
        names = tuple(np.asarray(archive["feature_names"]).astype(str).tolist())
        train_sha = str(np.asarray(archive["train_scene_list_sha256"]).item())
        forbidden_sha = str(np.asarray(archive["forbidden_validation_scene_list_sha256"]).item())
    if names != FEATURE_NAMES or x.shape != (len(iou), len(FEATURE_NAMES)):
        raise ValueError("feature contract mismatch")
    if not np.isfinite(x).all() or not np.isfinite(iou).all():
        raise ValueError("dataset contains non-finite values")
    train_scenes = read_scenes(args.train_scene_list.resolve())
    forbidden = read_scenes(args.forbidden_validation_scene_list.resolve())
    if (sha256(args.train_scene_list.resolve()) != train_sha
            or sha256(args.forbidden_validation_scene_list.resolve()) != forbidden_sha
            or set(train_scenes) & set(forbidden)
            or set(scene_ids.tolist()) - set(train_scenes)):
        raise ValueError("scene-list provenance or leakage check failed")
    labels = n25.astype(np.float64)
    if n25.sum() < 20 or (~n25).sum() < 20 or n50.sum() < 10:
        raise ValueError("insufficient novelty positives/negatives")
    folds = np.asarray([fold_for_scene(scene, args.folds) for scene in scene_ids])
    oof = np.full(len(x), np.nan)
    fold_report = []
    for fold in range(args.folds):
        validation = folds == fold; fitting = ~validation
        if not validation.any() or len(np.unique(labels[fitting])) != 2:
            raise ValueError(f"fold {fold} is empty or single-class")
        coef, mean, scale, bias = fit_logistic(
            x[fitting], labels[fitting], iou[fitting], iterations=args.iterations,
            learning_rate=args.learning_rate, l2=args.l2,
        )
        oof[validation] = sigmoid(((x[validation] - mean) / scale) @ coef + bias)
        fold_report.append({"fold": fold, "scenes": len(set(scene_ids[validation])),
                            "samples": int(validation.sum()),
                            "novel25": int(n25[validation].sum()),
                            "novel50": int(n50[validation].sum())})
    choices = []
    for threshold in np.linspace(0.05, 0.95, 181):
        selected = oof >= threshold
        positive_folds = sum(bool(np.any(selected & (folds == fold))) for fold in range(args.folds))
        row = {"threshold": float(threshold), "selected": int(selected.sum()),
               "precision_novel15": _precision(selected, n15),
               "precision_novel25": _precision(selected, n25),
               "precision_novel50": _precision(selected, n50),
               "recall_novel25": _recall(selected, n25),
               "recall_novel50": _recall(selected, n50),
               "positive_folds": positive_folds}
        if (row["selected"] >= args.min_selected
                and row["precision_novel25"] >= args.min_precision_novel25
                and row["precision_novel50"] >= args.min_precision_novel50
                and row["recall_novel50"] >= args.min_recall_novel50
                and positive_folds >= args.min_positive_folds):
            choices.append(row)
    authorized = bool(choices)
    chosen = max(choices, key=lambda row: (row["recall_novel50"], row["precision_novel50"], row["selected"])) if choices else {
        "threshold": 0.95, "selected": int((oof >= 0.95).sum()),
        "precision_novel15": _precision(oof >= 0.95, n15),
        "precision_novel25": _precision(oof >= 0.95, n25),
        "precision_novel50": _precision(oof >= 0.95, n50),
        "recall_novel25": _recall(oof >= 0.95, n25),
        "recall_novel50": _recall(oof >= 0.95, n50),
        "positive_folds": sum(bool(np.any((oof >= 0.95) & (folds == fold))) for fold in range(args.folds)),
    }
    coef, mean, scale, bias = fit_logistic(
        x, labels, iou, iterations=args.iterations,
        learning_rate=args.learning_rate, l2=args.l2,
    )
    policy = {"schema": POLICY_SCHEMA, "complete": True,
              "activation_authorized": authorized, "train_only": True,
              "scene_group_oof": True, "ground_truth_used_only_for_training": True,
              "validation_predictions_used_for_training": False,
              "feature_names": list(FEATURE_NAMES), "weights": coef.tolist(),
              "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
              "bias": float(bias), "probability_threshold": chosen["threshold"],
              "max_candidates_per_scene": args.max_candidates_per_scene,
              "hard_max_anchor_iou": args.hard_max_anchor_iou,
              "training_scene_ids": list(train_scenes),
              "forbidden_validation_scene_ids": list(forbidden),
              "validation_overlap_count": 0, "training_data": str(dataset),
              "training_data_sha256": sha256(dataset),
              "oof": {"selection": chosen, "folds": args.folds,
                      "per_fold": fold_report,
                      "requirements": {"min_selected": args.min_selected,
                          "min_precision_novel25": args.min_precision_novel25,
                          "min_precision_novel50": args.min_precision_novel50,
                          "min_recall_novel50": args.min_recall_novel50,
                          "min_positive_folds": args.min_positive_folds}}}
    atomic_json(args.output.resolve(), policy)
    return policy


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--dataset", type=Path, required=True)
    value.add_argument("--train-scene-list", type=Path, required=True)
    value.add_argument("--forbidden-validation-scene-list", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--folds", type=int, default=5)
    value.add_argument("--iterations", type=int, default=1800)
    value.add_argument("--learning-rate", type=float, default=0.06)
    value.add_argument("--l2", type=float, default=0.003)
    value.add_argument("--min-selected", type=int, default=20)
    value.add_argument("--min-precision-novel25", type=float, default=0.70)
    value.add_argument("--min-precision-novel50", type=float, default=0.45)
    value.add_argument("--min-recall-novel50", type=float, default=0.15)
    value.add_argument("--min-positive-folds", type=int, default=4)
    value.add_argument("--max-candidates-per-scene", type=int, default=6)
    value.add_argument("--hard-max-anchor-iou", type=float, default=0.10)
    return value


if __name__ == "__main__":
    result = train(parser().parse_args())
    print(json.dumps({"activation_authorized": result["activation_authorized"],
                      "oof": result["oof"]}, indent=2, sort_keys=True))
