"""Strict NumPy runtime for a pairwise AP50-aware geometry safety gate.

The gate never proposes geometry.  It compares one frozen B6 box with one
already verified local-geometry candidate and predicts:

* the candidate's IoU gain and heteroscedastic uncertainty;
* probabilities of improvement and harm;
* original/candidate IoU; and
* probabilities of an upward IoU-0.25/0.50 threshold crossing.

Training is deliberately separate from runtime.  Checkpoints contain a fixed
feature-name contract and small MLP weights that are evaluated with NumPy, so
online inference does not require PyTorch and cannot silently reorder inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple, Union

import numpy as np


AP50_SAFETY_GATE_SCHEMA = "boxfusion.ap50_delta_uncertainty_gate"
AP50_SAFETY_GATE_FORMAT_VERSION = 1
AP50_SAFETY_GATE_OUTPUT_NAMES = (
    "delta_mean",
    "delta_log_variance",
    "improvement_probability",
    "harm_probability",
    "original_iou",
    "candidate_iou",
    "cross_iou25_probability",
    "cross_iou50_probability",
)
AP50_SAFETY_GATE_OUTPUT_DIM = len(AP50_SAFETY_GATE_OUTPUT_NAMES)


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative_exp = np.exp(values[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return result


def _string_scalar(name: str, value: Any) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar string")
    result = str(array.item())
    if not result:
        raise ValueError(f"{name} cannot be empty")
    return result


def _integer_scalar(name: str, value: Any) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer scalar")
    return int(array.item())


@dataclass(frozen=True)
class AP50SafetyPrediction:
    """One immutable gate prediction."""

    delta_mean: float
    delta_std: float
    improvement_probability: float
    harm_probability: float
    original_iou: float
    candidate_iou: float
    cross_iou25_probability: float
    cross_iou50_probability: float

    @property
    def predicted_iou_margin(self) -> float:
        return float(self.candidate_iou - self.original_iou)


@dataclass(frozen=True)
class AP50SafetyDecision:
    """Auditable decision returned after hard geometry verification."""

    accepted: bool
    reason: str
    lower_confidence_delta: float
    prediction: AP50SafetyPrediction


@dataclass(frozen=True)
class AP50SafetyGateConfig:
    """Conservative activation thresholds.

    ``require_iou50_crossing`` is intended for an AP50-focused ablation.  The
    default accepts high-confidence improvements at any IoU while still
    exposing crossing probabilities for reporting.
    """

    minimum_improvement_probability: float = 0.75
    maximum_harm_probability: float = 0.15
    uncertainty_multiplier: float = 1.64
    maximum_delta_std: float = 0.15
    minimum_delta_lower_bound: float = 0.005
    minimum_predicted_iou_margin: float = 0.005
    require_iou50_crossing: bool = False
    minimum_iou50_crossing_probability: float = 0.60

    def validated(self) -> "AP50SafetyGateConfig":
        probability_names = (
            "minimum_improvement_probability",
            "maximum_harm_probability",
            "minimum_iou50_crossing_probability",
        )
        for name in probability_names:
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not np.isscalar(value)
                or not np.isfinite(value)
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must lie in [0,1]")
        for name in (
            "uncertainty_multiplier",
            "maximum_delta_std",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not np.isscalar(value)
                or not np.isfinite(value)
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "minimum_delta_lower_bound",
            "minimum_predicted_iou_margin",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not np.isscalar(value)
                or not np.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if not isinstance(self.require_iou50_crossing, (bool, np.bool_)):
            raise TypeError("require_iou50_crossing must be Boolean")
        return self


class AP50SafetyGate:
    """Small strict-schema NumPy MLP and conservative activation policy."""

    def __init__(
        self,
        *,
        feature_names: Sequence[str],
        weights: Sequence[np.ndarray],
        biases: Sequence[np.ndarray],
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        maximum_absolute_delta: float = 1.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        names = tuple(str(name) for name in feature_names)
        if not names or any(not name for name in names):
            raise ValueError("feature_names must contain non-empty names")
        if len(set(names)) != len(names):
            raise ValueError("feature_names must be unique")
        if not isinstance(weights, Sequence) or isinstance(
            weights, (str, bytes)
        ):
            raise TypeError("weights must be a sequence")
        if not isinstance(biases, Sequence) or isinstance(
            biases, (str, bytes)
        ):
            raise TypeError("biases must be a sequence")
        if len(weights) == 0 or len(weights) != len(biases):
            raise ValueError("weights and biases must have equal non-zero length")

        validated_weights = []
        validated_biases = []
        input_dim = len(names)
        for index, (weight, bias) in enumerate(zip(weights, biases)):
            weight_array = np.asarray(weight, dtype=np.float64)
            bias_array = np.asarray(bias, dtype=np.float64)
            if (
                weight_array.ndim != 2
                or weight_array.shape[0] != input_dim
                or weight_array.shape[1] < 1
            ):
                raise ValueError(
                    f"weight_{index} must have shape [{input_dim}, output_dim]"
                )
            if bias_array.shape != (weight_array.shape[1],):
                raise ValueError(
                    f"bias_{index} must have shape [{weight_array.shape[1]}]"
                )
            if (
                not np.isfinite(weight_array).all()
                or not np.isfinite(bias_array).all()
            ):
                raise ValueError("gate weights and biases must be finite")
            validated_weights.append(weight_array.copy())
            validated_biases.append(bias_array.copy())
            input_dim = int(weight_array.shape[1])
        if input_dim != AP50_SAFETY_GATE_OUTPUT_DIM:
            raise ValueError(
                "final gate layer must have "
                f"{AP50_SAFETY_GATE_OUTPUT_DIM} outputs"
            )

        mean = np.asarray(feature_mean, dtype=np.float64)
        scale = np.asarray(feature_scale, dtype=np.float64)
        expected = (len(names),)
        if mean.shape != expected or scale.shape != expected:
            raise ValueError(f"feature normalization must have shape {expected}")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("feature normalization must be finite")
        if np.any(scale <= 0.0):
            raise ValueError("feature_scale must be strictly positive")
        if (
            isinstance(maximum_absolute_delta, (bool, np.bool_))
            or not np.isscalar(maximum_absolute_delta)
            or not np.isfinite(maximum_absolute_delta)
            or float(maximum_absolute_delta) <= 0.0
            or float(maximum_absolute_delta) > 1.0
        ):
            raise ValueError("maximum_absolute_delta must lie in (0,1]")

        for array in (
            *validated_weights,
            *validated_biases,
            mean,
            scale,
        ):
            array.setflags(write=False)
        self.feature_names = names
        self.weights = tuple(validated_weights)
        self.biases = tuple(validated_biases)
        self.feature_mean = mean
        self.feature_scale = scale
        self.maximum_absolute_delta = float(maximum_absolute_delta)
        self.metadata = dict(metadata or {})

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def _feature_matrix(
        self,
        features: Union[
            np.ndarray,
            Mapping[str, Any],
            Sequence[Mapping[str, Any]],
        ],
    ) -> Tuple[np.ndarray, bool]:
        squeeze = False
        if isinstance(features, Mapping):
            received = set(features)
            expected = set(self.feature_names)
            missing = sorted(expected - received)
            extra = sorted(received - expected)
            if missing or extra:
                raise ValueError(
                    "gate feature schema mismatch: "
                    f"missing={missing}, extra={extra}"
                )
            values = np.asarray(
                [features[name] for name in self.feature_names],
                dtype=np.float64,
            )[None, :]
            squeeze = True
        elif isinstance(features, np.ndarray):
            values = np.asarray(features, dtype=np.float64)
            if values.ndim == 1:
                values = values[None, :]
                squeeze = True
        elif isinstance(features, Sequence) and not isinstance(
            features, (str, bytes)
        ):
            if len(features) == 0:
                values = np.empty((0, self.feature_dim), dtype=np.float64)
            elif all(isinstance(item, Mapping) for item in features):
                rows = []
                for item in features:
                    received = set(item)
                    expected = set(self.feature_names)
                    if received != expected:
                        raise ValueError("gate feature schema mismatch")
                    rows.append([item[name] for name in self.feature_names])
                values = np.asarray(rows, dtype=np.float64)
            else:
                values = np.asarray(features, dtype=np.float64)
                if values.ndim == 1:
                    values = values[None, :]
                    squeeze = True
        else:
            raise TypeError("unsupported gate feature container")
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError(
                f"gate features must have shape [N,{self.feature_dim}]"
            )
        if not np.isfinite(values).all():
            raise ValueError("gate features must be finite")
        return values, squeeze

    def predict_raw(self, features: Any) -> Mapping[str, np.ndarray]:
        values, _ = self._feature_matrix(features)
        activations = (values - self.feature_mean) / self.feature_scale
        for index, (weight, bias) in enumerate(
            zip(self.weights, self.biases)
        ):
            activations = activations @ weight + bias
            if index + 1 < len(self.weights):
                activations = np.maximum(activations, 0.0)
        delta_mean = (
            np.tanh(activations[:, 0]) * self.maximum_absolute_delta
        )
        delta_log_variance = np.clip(activations[:, 1], -12.0, 0.0)
        probabilities = _stable_sigmoid(activations[:, 2:])
        return {
            "delta_mean": delta_mean.astype(np.float32),
            "delta_std": np.exp(0.5 * delta_log_variance).astype(np.float32),
            "improvement_probability": probabilities[:, 0].astype(np.float32),
            "harm_probability": probabilities[:, 1].astype(np.float32),
            "original_iou": probabilities[:, 2].astype(np.float32),
            "candidate_iou": probabilities[:, 3].astype(np.float32),
            "cross_iou25_probability": probabilities[:, 4].astype(np.float32),
            "cross_iou50_probability": probabilities[:, 5].astype(np.float32),
        }

    def predict(self, features: Any) -> Union[
        AP50SafetyPrediction, Tuple[AP50SafetyPrediction, ...]
    ]:
        _, squeeze = self._feature_matrix(features)
        raw = self.predict_raw(features)
        count = len(raw["delta_mean"])
        predictions = tuple(
            AP50SafetyPrediction(
                delta_mean=float(raw["delta_mean"][index]),
                delta_std=float(raw["delta_std"][index]),
                improvement_probability=float(
                    raw["improvement_probability"][index]
                ),
                harm_probability=float(raw["harm_probability"][index]),
                original_iou=float(raw["original_iou"][index]),
                candidate_iou=float(raw["candidate_iou"][index]),
                cross_iou25_probability=float(
                    raw["cross_iou25_probability"][index]
                ),
                cross_iou50_probability=float(
                    raw["cross_iou50_probability"][index]
                ),
            )
            for index in range(count)
        )
        if squeeze:
            return predictions[0]
        return predictions

    def decide(
        self,
        features: Any,
        *,
        geometry_verified: bool,
        config: AP50SafetyGateConfig | None = None,
    ) -> AP50SafetyDecision:
        if not isinstance(geometry_verified, (bool, np.bool_)):
            raise TypeError("geometry_verified must be Boolean")
        prediction = self.predict(features)
        if not isinstance(prediction, AP50SafetyPrediction):
            raise ValueError("decide expects exactly one feature row")
        cfg = (config or AP50SafetyGateConfig()).validated()
        lower = float(
            prediction.delta_mean
            - float(cfg.uncertainty_multiplier) * prediction.delta_std
        )
        reason = "accepted"
        accepted = True
        if not bool(geometry_verified):
            accepted, reason = False, "geometry_not_verified"
        elif prediction.improvement_probability < float(
            cfg.minimum_improvement_probability
        ):
            accepted, reason = False, "improvement_probability"
        elif prediction.harm_probability > float(
            cfg.maximum_harm_probability
        ):
            accepted, reason = False, "harm_probability"
        elif prediction.delta_std > float(cfg.maximum_delta_std):
            accepted, reason = False, "uncertainty"
        elif lower < float(cfg.minimum_delta_lower_bound):
            accepted, reason = False, "delta_lower_bound"
        elif prediction.predicted_iou_margin < float(
            cfg.minimum_predicted_iou_margin
        ):
            accepted, reason = False, "predicted_iou_margin"
        elif bool(cfg.require_iou50_crossing) and (
            prediction.cross_iou50_probability
            < float(cfg.minimum_iou50_crossing_probability)
        ):
            accepted, reason = False, "iou50_crossing_probability"
        return AP50SafetyDecision(
            accepted=accepted,
            reason=reason,
            lower_confidence_delta=lower,
            prediction=prediction,
        )


def load_ap50_safety_gate(
    checkpoint_path: Union[str, Path],
) -> AP50SafetyGate:
    """Load a strict, non-pickled NPZ gate checkpoint."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"AP50 safety checkpoint not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        files = set(archive.files)
        if "num_layers" not in files:
            raise ValueError("AP50 safety checkpoint is missing num_layers")
        layer_count = _integer_scalar("num_layers", archive["num_layers"])
        expected = {
            "schema",
            "format_version",
            "feature_names",
            "output_names",
            "feature_mean",
            "feature_scale",
            "maximum_absolute_delta",
            "num_layers",
            "metadata_json",
        }
        for index in range(layer_count):
            expected.add(f"weight_{index}")
            expected.add(f"bias_{index}")
        if files != expected:
            raise ValueError(
                "AP50 safety checkpoint keys do not match strict schema: "
                f"missing={sorted(expected - files)}, "
                f"extra={sorted(files - expected)}"
            )
        if _string_scalar("schema", archive["schema"]) != (
            AP50_SAFETY_GATE_SCHEMA
        ):
            raise ValueError("unsupported AP50 safety checkpoint schema")
        if _integer_scalar(
            "format_version", archive["format_version"]
        ) != AP50_SAFETY_GATE_FORMAT_VERSION:
            raise ValueError("unsupported AP50 safety checkpoint version")
        feature_names = tuple(
            str(value) for value in np.asarray(archive["feature_names"]).tolist()
        )
        output_names = tuple(
            str(value) for value in np.asarray(archive["output_names"]).tolist()
        )
        if output_names != AP50_SAFETY_GATE_OUTPUT_NAMES:
            raise ValueError("AP50 safety checkpoint output schema mismatch")
        metadata_text = _string_scalar(
            "metadata_json", archive["metadata_json"]
        )
        try:
            metadata = json.loads(metadata_text)
        except json.JSONDecodeError as error:
            raise ValueError("metadata_json is invalid JSON") from error
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata_json must contain an object")
        maximum_absolute_delta = float(
            np.asarray(archive["maximum_absolute_delta"]).item()
        )
        return AP50SafetyGate(
            feature_names=feature_names,
            weights=tuple(
                np.asarray(archive[f"weight_{index}"], dtype=np.float32)
                for index in range(layer_count)
            ),
            biases=tuple(
                np.asarray(archive[f"bias_{index}"], dtype=np.float32)
                for index in range(layer_count)
            ),
            feature_mean=np.asarray(archive["feature_mean"], dtype=np.float32),
            feature_scale=np.asarray(
                archive["feature_scale"], dtype=np.float32
            ),
            maximum_absolute_delta=maximum_absolute_delta,
            metadata=metadata,
        )


__all__ = [
    "AP50_SAFETY_GATE_SCHEMA",
    "AP50_SAFETY_GATE_FORMAT_VERSION",
    "AP50_SAFETY_GATE_OUTPUT_NAMES",
    "AP50_SAFETY_GATE_OUTPUT_DIM",
    "AP50SafetyPrediction",
    "AP50SafetyDecision",
    "AP50SafetyGateConfig",
    "AP50SafetyGate",
    "load_ap50_safety_gate",
]
