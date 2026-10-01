#!/usr/bin/env python3
"""Build and train the train-only online keep/replace/append policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.tr3d_c2_maskrgbd_cache import sha256_file
from boxfusion.tr3d_c3_online_identity import PARENT_SCORE_ROUTE, SCHEMA as IDENTITY_SCHEMA
from boxfusion.tr3d_online_evidence_fusion import (
    FEATURE_NAMES, POLICY_SCHEMA, aabb_iou, fusion_features,
)
from boxfusion.tr3d_residual_cache import load_tr3d_residual_cache
from tools.build_tr3d_c3_source_gate_dataset import (
    box_iou, gt_minmax, read_scenes, transformed_minmax, unaligned_to_aligned,
)


def _prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    rows = payload[0]
    corners = np.asarray([row[1] for row in rows], dtype=np.float32)
    scores = np.asarray([row[2] for row in rows], dtype=np.float32)
    if not len(rows):
        corners = np.empty((0, 8, 3), dtype=np.float32)
    if corners.shape != (len(rows), 8, 3) or scores.shape != (len(rows),):
        raise ValueError(f"invalid prediction: {path}")
    return corners, scores


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path); path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"refusing policy overwrite: {path}") from error
    finally:
        try: os.unlink(temporary)
        except FileNotFoundError: pass


def _fold(scene: str, count: int) -> int:
    return int.from_bytes(hashlib.sha256(scene.encode("ascii")).digest()[:8], "big") % count


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _fit(x: np.ndarray, y: np.ndarray, *, iterations: int = 1600) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    z = (x - mean) / scale
    positives = max(int(y.sum()), 1)
    negatives = max(len(y) - positives, 1)
    sample = np.where(y > 0.5, len(y) / (2 * positives), len(y) / (2 * negatives))
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    for step in range(iterations):
        probability = _sigmoid(z @ weights + bias)
        error = (probability - y) * sample
        rate = 0.08 / math.sqrt(1.0 + step / 200.0)
        weights -= rate * ((z.T @ error) / sample.sum() + 0.002 * weights)
        bias -= rate * float(error.sum() / sample.sum())
    return weights, mean, scale, bias


def _choose_threshold(
    probability: np.ndarray,
    eligible: np.ndarray,
    positive: np.ndarray,
    cross50_gain: np.ndarray,
    cross50_loss: np.ndarray,
    folds: np.ndarray,
    *,
    min_selected: int,
    min_precision: float,
) -> dict:
    choices = []
    operating_points = []
    for threshold in np.linspace(0.05, 0.95, 181):
        selected = eligible & (probability >= threshold)
        count = int(selected.sum())
        precision = float(positive[selected].mean()) if count else 0.0
        positive_folds = sum(bool(np.any(selected & (folds == fold))) for fold in np.unique(folds))
        gain, loss = int(cross50_gain[selected].sum()), int(cross50_loss[selected].sum())
        row = {
            "threshold": float(threshold), "selected": count, "precision": precision,
            "positive_folds": positive_folds, "cross50_gain": gain,
            "cross50_loss": loss, "cross50_net": gain - loss,
        }
        operating_points.append(row)
        if count >= min_selected and precision >= min_precision and positive_folds >= 4 and gain >= loss:
            choices.append(row)
    if choices:
        result = max(
            choices,
            key=lambda row: (row["selected"], row["cross50_net"], -row["threshold"]),
        )
        result["authorized"] = True
        return result
    diagnostic = max(
        (
            row for row in operating_points
            if row["selected"] >= min_selected and row["positive_folds"] >= 4
        ),
        key=lambda row: (row["precision"], row["cross50_net"], row["selected"]),
        default=None,
    )
    return {
        "threshold": 0.999, "selected": 0, "precision": 0.0,
        "positive_folds": 0, "cross50_gain": 0, "cross50_loss": 0,
        "cross50_net": 0, "authorized": False,
        "best_unautorized_operating_point": diagnostic,
    }


def train(args: argparse.Namespace) -> dict:
    train_scenes = read_scenes(args.train_scene_list.resolve())
    forbidden = read_scenes(args.forbidden_validation_scene_list.resolve())
    if set(train_scenes) & set(forbidden) or len(forbidden) < 100:
        raise ValueError("train/validation scene partition is invalid")

    features, scenes = [], []
    append_eligible, append_positive = [], []
    replace_eligible, replace_positive = [], []
    append_gain50, append_loss50, replace_gain50, replace_loss50 = [], [], [], []
    statistics = {"rows": 0, "online_gate": 0}
    for scene in train_scenes:
        diagnostic_path = args.diagnostics_root / f"{scene}_c3_online_identity.json"
        payload = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        if (
            payload.get("schema") != IDENTITY_SCHEMA or payload.get("scene_id") != scene
            or payload.get("route") != PARENT_SCORE_ROUTE or not payload.get("observer_only")
            or payload.get("ground_truth_access") or payload.get("clip_access")
        ):
            raise ValueError(f"invalid train-only diagnostic: {diagnostic_path}")
        anchors, anchor_scores = _prediction(args.anchor_prediction_root / f"{scene}_boxes.pkl")
        parent_path = Path(payload["parent_cache"]).resolve()
        if sha256_file(parent_path) != payload["parent_cache_sha256"]:
            raise ValueError(f"{scene}: parent cache hash mismatch")
        with np.load(parent_path, allow_pickle=False) as archive:
            checkpoint = str(archive["checkpoint_sha256"].item())
            config = str(archive["config_sha256"].item())
        parent = load_tr3d_residual_cache(
            parent_path, expected_scene_id=scene, expected_prefix_id="p100",
            expected_checkpoint_sha256=checkpoint, expected_config_sha256=config,
        )
        transform = unaligned_to_aligned(parent.aligned_to_unaligned)
        aligned_anchors = transformed_minmax(anchors.astype(np.float64), transform)
        gt = gt_minmax(args.ground_truth_root / f"{scene}_bbox.npy")
        anchor_gt = box_iou(aligned_anchors, gt)
        for candidate in payload.get("candidates", ()):
            if not candidate.get("online_yoloe_mask2_depth"):
                continue
            statistics["online_gate"] += 1
            row = int(candidate["parent_row"])
            corners = np.asarray(parent.corners_world[row], dtype=np.float32)
            overlaps = aabb_iou(corners, anchors)
            nearest = int(np.argmax(overlaps)) if len(overlaps) else -1
            overlap = float(overlaps[nearest]) if nearest >= 0 else 0.0
            nearest_corners = anchors[nearest] if nearest >= 0 else corners
            nearest_score = float(anchor_scores[nearest]) if nearest >= 0 else 0.0
            values = fusion_features(
                candidate, corners, float(parent.scores_3d[row]), nearest_corners,
                nearest_score, overlap,
            )
            aligned_candidate = transformed_minmax(corners[None].astype(np.float64), transform)
            candidate_gt = box_iou(aligned_candidate, gt)[0]
            candidate_best_index = int(np.argmax(candidate_gt)) if len(gt) else -1
            candidate_best = float(candidate_gt[candidate_best_index]) if len(gt) else 0.0
            if nearest >= 0 and len(gt):
                anchor_best_index = int(np.argmax(anchor_gt[nearest]))
                anchor_best = float(anchor_gt[nearest, anchor_best_index])
            else:
                anchor_best_index, anchor_best = -1, 0.0
            same_object = candidate_best_index >= 0 and candidate_best_index == anchor_best_index
            features.append(values); scenes.append(scene)
            append_eligible.append(overlap <= args.max_append_anchor_iou)
            append_positive.append(candidate_best >= 0.25)
            append_gain50.append(candidate_best >= 0.50)
            append_loss50.append(False)
            replace_eligible.append(args.min_replace_anchor_iou <= overlap <= args.max_replace_anchor_iou)
            improves = same_object and candidate_best >= anchor_best + args.min_replace_iou_gain
            replace_positive.append(improves)
            replace_gain50.append(same_object and anchor_best < 0.50 <= candidate_best)
            replace_loss50.append(same_object and candidate_best < 0.50 <= anchor_best)
            statistics["rows"] += 1

    x = np.asarray(features, dtype=np.float64)
    scene_values = np.asarray(scenes)
    if len(x) < 50 or x.shape[1] != len(FEATURE_NAMES):
        raise ValueError("evidence-fusion training set is too small")
    folds = np.asarray([_fold(scene, args.folds) for scene in scenes], dtype=np.int32)
    statistics.update({
        "append_eligible": int(np.count_nonzero(append_eligible)),
        "append_positive": int(np.count_nonzero(np.asarray(append_eligible) & np.asarray(append_positive))),
        "replace_eligible": int(np.count_nonzero(replace_eligible)),
        "replace_positive": int(np.count_nonzero(np.asarray(replace_eligible) & np.asarray(replace_positive))),
        "replace_cross50_gain": int(np.count_nonzero(np.asarray(replace_eligible) & np.asarray(replace_gain50))),
        "replace_cross50_loss": int(np.count_nonzero(np.asarray(replace_eligible) & np.asarray(replace_loss50))),
    })
    gates = {}
    for name, eligible_values, positive_values, gain_values, loss_values, minimum, precision in (
        ("append", append_eligible, append_positive, append_gain50, append_loss50, args.min_append_selected, args.min_append_precision),
        ("replace", replace_eligible, replace_positive, replace_gain50, replace_loss50, args.min_replace_selected, args.min_replace_precision),
    ):
        eligible = np.asarray(eligible_values, dtype=np.bool_)
        positive = np.asarray(positive_values, dtype=np.bool_)
        gain = np.asarray(gain_values, dtype=np.bool_)
        loss = np.asarray(loss_values, dtype=np.bool_)
        y = positive.astype(np.float64)
        if positive.sum() < 5 or (~positive).sum() < 5:
            raise ValueError(f"{name} gate lacks positive/negative examples")
        oof = np.full(len(x), np.nan, dtype=np.float64)
        for fold in range(args.folds):
            validation = folds == fold
            training = ~validation
            weights, mean, scale, bias = _fit(x[training], y[training])
            oof[validation] = _sigmoid(((x[validation] - mean) / scale) @ weights + bias)
        choice = _choose_threshold(oof, eligible, positive, gain, loss, folds, min_selected=minimum, min_precision=precision)
        weights, mean, scale, bias = _fit(x, y)
        gates[name] = {
            "weights": weights.tolist(), "feature_mean": mean.tolist(),
            "feature_scale": scale.tolist(), "bias": float(bias),
            "probability_threshold": float(choice["threshold"]), "oof": choice,
        }

    authorized = gates["append"]["oof"]["selected"] >= args.min_append_selected and gates["replace"]["oof"]["selected"] >= args.min_replace_selected
    payload = {
        "schema": POLICY_SCHEMA, "complete": True,
        "activation_authorized": authorized, "train_only": True,
        "scene_group_oof": True, "ground_truth_used_only_for_training": True,
        "validation_predictions_used_for_training": False,
        "route": PARENT_SCORE_ROUTE, "feature_names": list(FEATURE_NAMES),
        "append_gate": gates["append"], "replace_gate": gates["replace"],
        "max_append_anchor_iou": args.max_append_anchor_iou,
        "min_replace_anchor_iou": args.min_replace_anchor_iou,
        "max_replace_anchor_iou": args.max_replace_anchor_iou,
        "output_score_min": 0.02, "output_score_max": 0.39,
        "max_appends_per_scene": 8,
        "training_scene_ids": list(train_scenes),
        "forbidden_validation_scene_ids": list(forbidden),
        "training_scene_list_sha256": sha256_file(args.train_scene_list),
        "forbidden_validation_scene_list_sha256": sha256_file(args.forbidden_validation_scene_list),
        "statistics": statistics,
    }
    _atomic_json(args.output.resolve(), payload)
    return payload


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--train-scene-list", type=Path, required=True)
    value.add_argument("--forbidden-validation-scene-list", type=Path, required=True)
    value.add_argument("--diagnostics-root", type=Path, required=True)
    value.add_argument("--anchor-prediction-root", type=Path, required=True)
    value.add_argument("--ground-truth-root", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--folds", type=int, default=5)
    value.add_argument("--max-append-anchor-iou", type=float, default=0.15)
    value.add_argument("--min-replace-anchor-iou", type=float, default=0.15)
    value.add_argument("--max-replace-anchor-iou", type=float, default=0.85)
    value.add_argument("--min-replace-iou-gain", type=float, default=0.05)
    value.add_argument("--min-append-selected", type=int, default=20)
    value.add_argument("--min-append-precision", type=float, default=0.80)
    value.add_argument("--min-replace-selected", type=int, default=10)
    value.add_argument("--min-replace-precision", type=float, default=0.80)
    return value


if __name__ == "__main__":
    args = parser().parse_args()
    report = train(args)
    print(json.dumps({
        "activation_authorized": report["activation_authorized"],
        "append_oof": report["append_gate"]["oof"],
        "replace_oof": report["replace_gate"]["oof"],
        "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))
