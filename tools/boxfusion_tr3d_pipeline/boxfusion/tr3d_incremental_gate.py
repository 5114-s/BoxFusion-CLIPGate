"""GT-free feature contract for incremental TR3D novelty selection."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


POLICY_SCHEMA = "boxfusion.tr3d_incremental_novelty_gate.v1"
DATASET_SCHEMA = "boxfusion.tr3d_incremental_novelty_dataset.v1"
FEATURE_NAMES = (
    "best_score", "score_mean", "score_std", "log_hit_count",
    "lifespan_fraction", "hit_rate", "center_jitter_m",
    "extent_jitter_relative", "mean_match_iou", "log_point_support",
    "log_point_density", "anchor_iou_max", "anchor_center_distance_m",
    "matched_anchor_score", "log_volume", "log_aspect_ratio",
    "first_call_fraction", "last_call_fraction",
)


def candidate_features(row: Mapping[str, Any], provider_calls: int) -> np.ndarray:
    corners = np.asarray(row["best_corners_world"], dtype=np.float64)
    if corners.shape != (8, 3) or not np.isfinite(corners).all():
        raise ValueError("incremental candidate corners must be finite [8,3]")
    extent = np.maximum(np.ptp(corners, axis=0), 1e-6)
    volume = float(np.prod(extent))
    aspect = float(extent.max() / extent.min())
    calls = max(int(provider_calls), 1)
    values = np.asarray([
        row["best_score"], row["score_mean"], row["score_std"],
        np.log1p(row["hit_count"]), row["lifespan_calls"] / calls,
        row["hit_rate"], row["center_jitter_m"],
        row["extent_jitter_relative"], row["mean_match_iou"],
        np.log1p(row["point_support"]), np.log1p(row["point_density"]),
        row["anchor_iou_max"], min(float(row["anchor_center_distance_m"]), 5.0),
        row["matched_anchor_score"], np.log(max(volume, 1e-6)),
        np.log(max(aspect, 1.0)), row["first_call"] / calls,
        row["last_call"] / calls,
    ], dtype=np.float64)
    if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
        raise ValueError("incremental candidate features are invalid")
    return values.astype(np.float32)


@dataclass(frozen=True)
class IncrementalNoveltyPolicy:
    """Immutable train-only policy used by the validation active branch."""

    path: Path
    sha256: str
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    probability_threshold: float
    max_candidates_per_scene: int
    hard_max_anchor_iou: float

    @classmethod
    def load(cls, path: str | Path) -> "IncrementalNoveltyPolicy":
        resolved = Path(path).resolve()
        if (
            not resolved.is_file()
            or resolved.is_symlink()
            or resolved.stat().st_mode & 0o022
        ):
            raise ValueError("incremental novelty policy must be immutable")
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if (
            payload.get("schema") != POLICY_SCHEMA
            or not payload.get("complete")
            or not payload.get("activation_authorized")
            or not payload.get("train_only")
            or not payload.get("scene_group_oof")
            or not payload.get("ground_truth_used_only_for_training")
            or payload.get("validation_predictions_used_for_training")
            or int(payload.get("validation_overlap_count", -1)) != 0
            or tuple(payload.get("feature_names", ())) != FEATURE_NAMES
        ):
            raise ValueError("incremental novelty policy is not activation-authorized")
        mean = np.asarray(payload["feature_mean"], dtype=np.float64)
        scale = np.asarray(payload["feature_scale"], dtype=np.float64)
        weights = np.asarray(payload["weights"], dtype=np.float64)
        if (
            mean.shape != (len(FEATURE_NAMES),)
            or scale.shape != mean.shape
            or weights.shape != mean.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or not np.isfinite(weights).all()
            or np.any(scale <= 0.0)
        ):
            raise ValueError("incremental novelty policy feature tensors are invalid")
        threshold = float(payload["probability_threshold"])
        max_candidates = int(payload["max_candidates_per_scene"])
        max_anchor_iou = float(payload["hard_max_anchor_iou"])
        bias = float(payload["bias"])
        if (
            not all(math.isfinite(value) for value in (threshold, max_anchor_iou, bias))
            or not 0.0 < threshold < 1.0
            or not 0.0 <= max_anchor_iou < 1.0
            or max_candidates < 1
        ):
            raise ValueError("incremental novelty policy thresholds are invalid")
        return cls(
            path=resolved,
            sha256=hashlib.sha256(resolved.read_bytes()).hexdigest(),
            mean=mean,
            scale=scale,
            weights=weights,
            bias=bias,
            probability_threshold=threshold,
            max_candidates_per_scene=max_candidates,
            hard_max_anchor_iou=max_anchor_iou,
        )

    def probability(self, row: Mapping[str, Any], provider_calls: int) -> float:
        features = candidate_features(row, provider_calls).astype(np.float64)
        logit = float(((features - self.mean) / self.scale) @ self.weights + self.bias)
        # Stable scalar sigmoid.
        if logit >= 0.0:
            value = 1.0 / (1.0 + math.exp(-min(logit, 700.0)))
        else:
            exp_value = math.exp(max(logit, -700.0))
            value = exp_value / (1.0 + exp_value)
        return float(value)


__all__ = [
    "DATASET_SCHEMA", "FEATURE_NAMES", "POLICY_SCHEMA",
    "IncrementalNoveltyPolicy", "candidate_features",
]
