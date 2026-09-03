"""Train-only linear risk calibration for frozen R3 geometry replacement.

The calibrator is intentionally veto-only.  It receives only candidates that
already pass the frozen R3 primary rule and may reject some of them as risky.
It cannot introduce a candidate, change the selected proposal for an anchor,
or change labels, scores, output order, or output count.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from .tr3d_r3_active import (
    R3NearCacheLike,
    materialize_shadow_active_prediction,
    primary_candidate_rows,
)


CALIBRATOR_SCHEMA = "boxfusion.tr3d_r3_veto_calibrator.v1"
CALIBRATED_SUMMARY_SCHEMA = "boxfusion.tr3d_r3_calibrated_summary.v1"
FEATURE_NAMES = (
    "logit_tr3d_score",
    "logit_anchor_score",
    "anchor_iou",
    "center_distance_over_anchor_diagonal",
    "abs_log_volume_ratio",
    "log1p_point_density_m3",
)
CLASS_NAMES = ("gain", "safe_neutral", "harm")
HARM_CLASS_INDEX = 2
_PROBABILITY_EPSILON = 1e-6


def _finite_vector(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def _logit(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON)
    return np.log(clipped) - np.log1p(-clipped)


def candidate_features(
    source_payload: object,
    cache: Any,
    proposal_rows: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Extract the pre-registered six geometry/score risk features."""

    outer = source_payload
    if type(outer) not in {list, tuple} or len(outer) != 1:
        raise ValueError("prediction payload must contain exactly one batch row")
    detections = outer[0]
    if type(detections) not in {list, tuple}:
        raise ValueError("prediction batch row must be a built-in list or tuple")
    rows = np.asarray(proposal_rows, dtype=np.int64)
    if rows.ndim != 1:
        raise ValueError("proposal_rows must be one-dimensional")
    proposal_count = len(np.asarray(cache.proposal_ids))
    if len(rows) and (np.any(rows < 0) or np.any(rows >= proposal_count)):
        raise ValueError("proposal_rows are out of range")
    anchor_index = np.asarray(cache.anchor_index, dtype=np.int64)
    if anchor_index.shape != (proposal_count,):
        raise ValueError("cache anchor_index must be [proposal_count]")
    if proposal_count and (
        np.any(anchor_index < 0) or np.any(anchor_index >= len(detections))
    ):
        raise ValueError("cache anchor_index is out of prediction range")
    tr3d_score = _finite_vector(cache.tr3d_score, (proposal_count,), "tr3d_score")
    anchor_iou = _finite_vector(cache.anchor_iou, (proposal_count,), "anchor_iou")
    center = _finite_vector(
        cache.center_distance_over_anchor_diagonal,
        (proposal_count,),
        "center_distance_over_anchor_diagonal",
    )
    volume = _finite_vector(cache.volume_ratio, (proposal_count,), "volume_ratio")
    density = _finite_vector(
        cache.point_density_m3, (proposal_count,), "point_density_m3"
    )
    if (
        np.any((tr3d_score < 0) | (tr3d_score > 1))
        or np.any((anchor_iou < 0) | (anchor_iou > 1))
        or np.any(center < 0)
        or np.any(volume <= 0)
        or np.any(density < 0)
    ):
        raise ValueError("R3 calibration inputs violate feature ranges")
    anchor_scores = np.asarray(
        [float(detections[int(anchor_index[row])][2]) for row in rows],
        dtype=np.float64,
    )
    if not np.isfinite(anchor_scores).all() or np.any(
        (anchor_scores < 0) | (anchor_scores > 1)
    ):
        raise ValueError("prediction anchor scores must be finite in [0,1]")
    if not len(rows):
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    return np.column_stack(
        (
            _logit(tr3d_score[rows]),
            _logit(anchor_scores),
            anchor_iou[rows],
            center[rows],
            np.abs(np.log(volume[rows])),
            np.log1p(density[rows]),
        )
    ).astype(np.float64, copy=False)


@dataclass(frozen=True)
class R3VetoCalibrator:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    activation_authorized: bool
    dataset_sha256: str
    scene_list_sha256: str
    metadata: Mapping[str, Any]

    def validate(self) -> "R3VetoCalibrator":
        if type(self.activation_authorized) is not bool:
            raise ValueError("activation_authorized must be a strict JSON boolean")
        width = len(FEATURE_NAMES)
        mean = _finite_vector(self.feature_mean, (width,), "feature_mean")
        scale = _finite_vector(self.feature_scale, (width,), "feature_scale")
        coefficients = _finite_vector(
            self.coefficients, (len(CLASS_NAMES), width), "coefficients"
        )
        intercept = _finite_vector(
            self.intercept, (len(CLASS_NAMES),), "intercept"
        )
        if np.any(scale <= 0):
            raise ValueError("feature_scale must be positive")
        for name, value in (
            ("dataset_sha256", self.dataset_sha256),
            ("scene_list_sha256", self.scene_list_sha256),
        ):
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a SHA256")
            int(value, 16)
        encoded = json.dumps(dict(self.metadata), sort_keys=True, allow_nan=False)
        normalized_metadata = json.loads(encoded)
        if not isinstance(normalized_metadata, dict):
            raise ValueError("metadata must be a JSON mapping")
        train_gate = normalized_metadata.get("train_gate_pass")
        if train_gate is not None and (
            type(train_gate) is not bool or train_gate is not self.activation_authorized
        ):
            raise ValueError(
                "metadata train_gate_pass must strictly match activation_authorized"
            )
        formal = normalized_metadata.get("formal_independent_activation_authorized")
        if formal is not None and type(formal) is not bool:
            raise ValueError(
                "formal_independent_activation_authorized must be boolean"
            )
        object.__setattr__(self, "feature_mean", mean)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "intercept", intercept)
        return self

    def probabilities(self, features: object) -> np.ndarray:
        self.validate()
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
            raise ValueError("features must be [N,6]")
        if not np.isfinite(values).all():
            raise ValueError("features contain non-finite values")
        logits = (
            (values - self.feature_mean) / self.feature_scale
        ) @ self.coefficients.T + self.intercept
        logits -= logits.max(axis=1, keepdims=True) if len(logits) else 0.0
        exponential = np.exp(logits)
        return exponential / exponential.sum(axis=1, keepdims=True)

    def accept(self, features: object) -> np.ndarray:
        """Accept only when non-harm probability strictly exceeds harm.

        Exact ties fail closed.  ``argmax != harm`` would accept a tie because
        NumPy returns the first class, which is unsafe for a veto gate.
        """

        probabilities = self.probabilities(features)
        return probabilities[:, HARM_CLASS_INDEX] < np.maximum(
            probabilities[:, 0], probabilities[:, 1]
        )

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": CALIBRATOR_SCHEMA,
            "feature_names": list(FEATURE_NAMES),
            "class_names": list(CLASS_NAMES),
            "gate": "accept_if_max_gain_or_neutral_probability_strictly_gt_harm",
            "veto_only": True,
            "may_add_primary_replacements": False,
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept.tolist(),
            "activation_authorized": bool(self.activation_authorized),
            "dataset_sha256": self.dataset_sha256,
            "scene_list_sha256": self.scene_list_sha256,
            "metadata": dict(self.metadata),
        }


def calibrator_sha256(model: R3VetoCalibrator) -> str:
    encoded = json.dumps(
        model.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_calibrator(path: str | Path) -> R3VetoCalibrator:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != CALIBRATOR_SCHEMA:
        raise ValueError("unsupported R3 calibrator schema")
    if payload.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("R3 calibrator feature order changed")
    if payload.get("class_names") != list(CLASS_NAMES):
        raise ValueError("R3 calibrator class order changed")
    if (
        payload.get("gate")
        != "accept_if_max_gain_or_neutral_probability_strictly_gt_harm"
        or payload.get("veto_only") is not True
        or payload.get("may_add_primary_replacements") is not False
    ):
        raise ValueError("R3 calibrator violates the veto-only contract")
    if type(payload.get("activation_authorized")) is not bool:
        raise ValueError("activation_authorized must be a strict JSON boolean")
    return R3VetoCalibrator(
        feature_mean=np.asarray(payload["feature_mean"], dtype=np.float64),
        feature_scale=np.asarray(payload["feature_scale"], dtype=np.float64),
        coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
        intercept=np.asarray(payload["intercept"], dtype=np.float64),
        activation_authorized=payload["activation_authorized"],
        dataset_sha256=str(payload["dataset_sha256"]),
        scene_list_sha256=str(payload["scene_list_sha256"]),
        metadata=dict(payload.get("metadata", {})),
    ).validate()


def materialize_calibrated_prediction(
    source_payload: object,
    cache: R3NearCacheLike,
    calibrator: R3VetoCalibrator,
    *,
    require_authorized: bool = True,
) -> tuple[object, dict[str, Any]]:
    """Apply a calibrated subset of the frozen primary replacements."""

    model = calibrator.validate()
    if require_authorized and not model.activation_authorized:
        raise PermissionError("R3 calibrator did not pass its train-only gate")
    primary = np.asarray(
        primary_candidate_rows(source_payload, cache), dtype=np.int64
    )
    features = candidate_features(source_payload, cache, primary)
    probabilities = model.probabilities(features)
    accepted_mask = model.accept(features)
    accepted = primary[accepted_mask]
    restricted = SimpleNamespace(
        anchor_count=int(cache.anchor_count),
        proposal_ids=np.asarray(cache.proposal_ids)[accepted],
        proposal_corners_world=np.asarray(cache.proposal_corners_world)[accepted],
        anchor_index=np.asarray(cache.anchor_index)[accepted],
        tr3d_score=np.asarray(cache.tr3d_score)[accepted],
        anchor_score=np.asarray(cache.anchor_score)[accepted],
    )
    output, active = materialize_shadow_active_prediction(source_payload, restricted)
    decisions = []
    for position, proposal_row in enumerate(primary):
        decisions.append(
            {
                "proposal_row": int(proposal_row),
                "proposal_id": int(np.asarray(cache.proposal_ids)[proposal_row]),
                "anchor_index": int(np.asarray(cache.anchor_index)[proposal_row]),
                "accepted": bool(accepted_mask[position]),
                "probabilities": probabilities[position].tolist(),
                "predicted_class": CLASS_NAMES[int(np.argmax(probabilities[position]))],
            }
        )
    return output, {
        "schema": CALIBRATED_SUMMARY_SCHEMA,
        "veto_only": True,
        "calibrator_sha256": calibrator_sha256(model),
        "primary_count": int(len(primary)),
        "accepted_count": int(np.count_nonzero(accepted_mask)),
        "vetoed_count": int(np.count_nonzero(~accepted_mask)),
        "active_summary": active.as_dict(),
        "decisions": decisions,
    }
