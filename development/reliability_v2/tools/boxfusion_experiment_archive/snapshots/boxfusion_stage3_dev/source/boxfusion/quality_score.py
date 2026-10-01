"""Fixed-schema detection-quality scoring and stable 3D AABB Soft-NMS."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


# All features are normalized probabilities/qualities in [0, 1].  Keeping one
# immutable schema prevents accidentally training and deploying different
# column orders.
QUALITY_FEATURE_NAMES = (
    "detector_score",
    "mask_confidence",
    "valid_depth_ratio",
    "depth_support",
    "projection_iou",
    "geometry_consistency",
    "appearance_consistency",
    "view_count_quality",
    "box_stability",
    "source_agreement",
    "area_quality",
    "refiner_quality",
)
QUALITY_FEATURE_DIM = len(QUALITY_FEATURE_NAMES)

DEFAULT_HEURISTIC_WEIGHTS = np.asarray(
    [
        0.20,
        0.06,
        0.06,
        0.12,
        0.10,
        0.12,
        0.07,
        0.08,
        0.08,
        0.04,
        0.02,
        0.05,
    ],
    dtype=np.float64,
)


def _validate_feature_array(
    features: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    values = np.asarray(features, dtype=np.float64)
    squeeze = values.ndim == 1
    if squeeze:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != QUALITY_FEATURE_DIM:
        raise ValueError(
            "quality features must have shape "
            f"[N, {QUALITY_FEATURE_DIM}] or [{QUALITY_FEATURE_DIM}]"
        )
    if not np.isfinite(values).all():
        raise ValueError("quality features must be finite")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("quality features must lie in [0, 1]")
    return values, squeeze


def quality_feature_vector(features: Mapping[str, Any]) -> np.ndarray:
    """Convert one exact-schema mapping into a read-only feature vector."""

    if not isinstance(features, Mapping):
        raise TypeError("features must be a mapping")
    received = set(features)
    expected = set(QUALITY_FEATURE_NAMES)
    missing = sorted(expected - received)
    extra = sorted(received - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError(
            "quality feature schema mismatch: " + ", ".join(details)
        )

    ordered = []
    for name in QUALITY_FEATURE_NAMES:
        value = features[name]
        if not np.isscalar(value):
            raise TypeError(f"quality feature {name!r} must be scalar")
        ordered.append(float(value))
    vector, _ = _validate_feature_array(np.asarray(ordered))
    result = vector[0].astype(np.float32)
    result.setflags(write=False)
    return result


def quality_feature_matrix(
    features: Union[
        np.ndarray,
        Mapping[str, Any],
        Sequence[Mapping[str, Any]],
    ],
) -> np.ndarray:
    """Return a validated ``[N, F]`` matrix in the fixed schema order."""

    if isinstance(features, Mapping):
        return quality_feature_vector(features)[None, :]
    if isinstance(features, np.ndarray):
        matrix, _ = _validate_feature_array(features)
        return matrix.astype(np.float32)
    if isinstance(features, Sequence) and not isinstance(
        features, (str, bytes)
    ):
        if len(features) == 0:
            return np.empty((0, QUALITY_FEATURE_DIM), dtype=np.float32)
        if all(isinstance(item, Mapping) for item in features):
            return np.stack(
                [quality_feature_vector(item) for item in features]
            )
        matrix, _ = _validate_feature_array(np.asarray(features))
        return matrix.astype(np.float32)
    raise TypeError(
        "features must be an array, mapping, or sequence of mappings"
    )


def _prepare_scorer_features(
    features: Union[
        np.ndarray,
        Mapping[str, Any],
        Sequence[Mapping[str, Any]],
    ],
) -> Tuple[np.ndarray, bool]:
    if isinstance(features, Mapping):
        return quality_feature_vector(features)[None, :].astype(
            np.float64
        ), True
    values = quality_feature_matrix(features).astype(np.float64)
    raw = np.asarray(features)
    is_record_sequence = (
        isinstance(features, Sequence)
        and len(features) > 0
        and isinstance(features[0], Mapping)
    )
    squeeze = (
        not is_record_sequence
        and raw.ndim == 1
        and raw.shape == (QUALITY_FEATURE_DIM,)
    )
    return values, squeeze


def _restore_score_shape(scores: np.ndarray, squeeze: bool) -> Any:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    return float(scores[0]) if squeeze else scores


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


class HeuristicQualityScorer:
    """Monotonic weighted-average scorer used without learned weights."""

    def __init__(
        self, weights: Optional[np.ndarray] = None
    ) -> None:
        weights = (
            DEFAULT_HEURISTIC_WEIGHTS
            if weights is None
            else np.asarray(weights, dtype=np.float64)
        )
        if weights.shape != (QUALITY_FEATURE_DIM,):
            raise ValueError(
                f"weights must have shape [{QUALITY_FEATURE_DIM}]"
            )
        if not np.isfinite(weights).all():
            raise ValueError("heuristic weights must be finite")
        if (weights < 0.0).any():
            raise ValueError("heuristic weights must be non-negative")
        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError("at least one heuristic weight must be positive")
        self.weights = (weights / total).copy()
        self.weights.setflags(write=False)

    def score(self, features: Any) -> Any:
        matrix, squeeze = _prepare_scorer_features(features)
        scores = matrix @ self.weights
        return _restore_score_shape(np.clip(scores, 0.0, 1.0), squeeze)

    __call__ = score


class LinearQualityScorer:
    """Sigmoid linear calibration over the fixed quality schema."""

    def __init__(
        self, weights: np.ndarray, bias: float = 0.0
    ) -> None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (QUALITY_FEATURE_DIM,):
            raise ValueError(
                f"weights must have shape [{QUALITY_FEATURE_DIM}]"
            )
        if not np.isfinite(weights).all():
            raise ValueError("linear weights must be finite")
        if not np.isscalar(bias) or not np.isfinite(bias):
            raise ValueError("linear bias must be a finite scalar")
        self.weights = weights.copy()
        self.weights.setflags(write=False)
        self.bias = float(bias)

    def score(self, features: Any) -> Any:
        matrix, squeeze = _prepare_scorer_features(features)
        scores = _stable_sigmoid(matrix @ self.weights + self.bias)
        return _restore_score_shape(scores, squeeze)

    __call__ = score


class MLPQualityScorer:
    """Small NumPy MLP calibrator with ReLU hidden layers and sigmoid output."""

    def __init__(
        self,
        weights: Sequence[np.ndarray],
        biases: Sequence[np.ndarray],
    ) -> None:
        if not isinstance(weights, Sequence) or isinstance(
            weights, (str, bytes)
        ):
            raise TypeError("weights must be a sequence of matrices")
        if not isinstance(biases, Sequence) or isinstance(
            biases, (str, bytes)
        ):
            raise TypeError("biases must be a sequence of vectors")
        if len(weights) == 0 or len(weights) != len(biases):
            raise ValueError(
                "MLP requires the same non-zero number of weights and biases"
            )

        validated_weights = []
        validated_biases = []
        input_dim = QUALITY_FEATURE_DIM
        for layer_index, (weight, bias) in enumerate(
            zip(weights, biases)
        ):
            weight = np.asarray(weight, dtype=np.float64)
            bias = np.asarray(bias, dtype=np.float64)
            if weight.ndim != 2 or weight.shape[0] != input_dim:
                raise ValueError(
                    f"weight_{layer_index} must have shape "
                    f"[{input_dim}, output_dim]"
                )
            if weight.shape[1] < 1:
                raise ValueError("MLP layers must have positive width")
            if bias.shape != (weight.shape[1],):
                raise ValueError(
                    f"bias_{layer_index} must have shape "
                    f"[{weight.shape[1]}]"
                )
            if not np.isfinite(weight).all() or not np.isfinite(bias).all():
                raise ValueError("MLP weights and biases must be finite")
            validated_weights.append(weight.copy())
            validated_biases.append(bias.copy())
            input_dim = weight.shape[1]
        if input_dim != 1:
            raise ValueError("the final MLP layer must have one output")

        for array in (*validated_weights, *validated_biases):
            array.setflags(write=False)
        self.weights = tuple(validated_weights)
        self.biases = tuple(validated_biases)

    def score(self, features: Any) -> Any:
        activations, squeeze = _prepare_scorer_features(features)
        for layer_index, (weight, bias) in enumerate(
            zip(self.weights, self.biases)
        ):
            activations = activations @ weight + bias
            if layer_index + 1 < len(self.weights):
                activations = np.maximum(activations, 0.0)
        scores = _stable_sigmoid(activations[:, 0])
        return _restore_score_shape(scores, squeeze)

    __call__ = score


def make_quality_scorer(
    method: str = "heuristic",
    *,
    weights: Optional[Any] = None,
    biases: Optional[Sequence[np.ndarray]] = None,
    bias: float = 0.0,
) -> Union[
    HeuristicQualityScorer,
    LinearQualityScorer,
    MLPQualityScorer,
]:
    """Construct a strictly validated heuristic, linear, or MLP scorer."""

    if not isinstance(method, str):
        raise TypeError("method must be a string")
    method = method.lower()
    if method == "heuristic":
        if biases is not None or bias != 0.0:
            raise ValueError(
                "heuristic scoring does not accept biases or bias"
            )
        return HeuristicQualityScorer(weights)
    if method == "linear":
        if weights is None:
            raise ValueError("linear scoring requires weights")
        if biases is not None:
            raise ValueError(
                "linear scoring accepts bias, not a biases sequence"
            )
        return LinearQualityScorer(weights, bias)
    if method == "mlp":
        if weights is None or biases is None:
            raise ValueError("MLP scoring requires weights and biases")
        if bias != 0.0:
            raise ValueError("MLP scoring uses its biases sequence")
        return MLPQualityScorer(weights, biases)
    raise ValueError("method must be 'heuristic', 'linear', or 'mlp'")


def load_quality_scorer(
    checkpoint_path: Union[str, Path],
    *,
    method: str,
) -> Union[LinearQualityScorer, MLPQualityScorer]:
    """Load a strict, pickle-free ``.npz`` linear or MLP quality scorer.

    Linear schema:
        ``feature_names``, ``weight``, ``bias``.

    MLP schema:
        ``feature_names``, ``num_layers``, followed by
        ``weight_0``, ``bias_0``, ..., ``weight_L``, ``bias_L``.
    """

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"quality checkpoint not found: {path}")
    if not isinstance(method, str):
        raise TypeError("method must be a string")
    method = method.lower()
    if method not in {"linear", "mlp"}:
        raise ValueError("only learned linear or MLP scorers use checkpoints")

    with np.load(path, allow_pickle=False) as checkpoint:
        keys = set(checkpoint.files)
        if "feature_names" not in keys:
            raise ValueError("quality checkpoint is missing feature_names")
        names = tuple(
            str(value)
            for value in np.asarray(checkpoint["feature_names"]).tolist()
        )
        if names != QUALITY_FEATURE_NAMES:
            raise ValueError(
                "quality checkpoint feature schema/order does not match"
            )

        if method == "linear":
            expected = {"feature_names", "weight", "bias"}
            if keys != expected:
                raise ValueError(
                    "linear checkpoint keys must be exactly "
                    f"{sorted(expected)}"
                )
            bias_values = np.asarray(
                checkpoint["bias"], dtype=np.float64
            )
            if bias_values.size != 1:
                raise ValueError("linear checkpoint bias must be scalar")
            return LinearQualityScorer(
                checkpoint["weight"], float(bias_values.reshape(-1)[0])
            )

        if "num_layers" not in keys:
            raise ValueError("MLP checkpoint is missing num_layers")
        layer_values = np.asarray(checkpoint["num_layers"])
        if layer_values.size != 1:
            raise ValueError("num_layers must be scalar")
        raw_layer_count = layer_values.reshape(-1)[0]
        if (
            isinstance(raw_layer_count, (bool, np.bool_))
            or not np.isfinite(raw_layer_count)
            or float(raw_layer_count) != int(raw_layer_count)
        ):
            raise ValueError("num_layers must be an integer")
        layer_count = int(raw_layer_count)
        if layer_count < 1:
            raise ValueError("num_layers must be positive")
        expected = {"feature_names", "num_layers"}
        expected.update(f"weight_{index}" for index in range(layer_count))
        expected.update(f"bias_{index}" for index in range(layer_count))
        if keys != expected:
            raise ValueError(
                "MLP checkpoint layer keys do not exactly match num_layers"
            )
        weights = [
            checkpoint[f"weight_{index}"] for index in range(layer_count)
        ]
        biases = [
            checkpoint[f"bias_{index}"] for index in range(layer_count)
        ]
        return MLPQualityScorer(weights, biases)


def _validate_aabb_batch(
    boxes: np.ndarray, *, name: str = "boxes"
) -> np.ndarray:
    values = np.asarray(boxes, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"{name} must have shape [N, 6] or [6]")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    if values.shape[0] and not (values[:, 3:6] > 0.0).all():
        raise ValueError(f"{name} dimensions must be positive")
    return values


def pairwise_aabb_iou_3d(
    boxes_a: np.ndarray, boxes_b: np.ndarray
) -> np.ndarray:
    """Pairwise IoU for center-size AABBs in three dimensions."""

    a = _validate_aabb_batch(boxes_a, name="boxes_a")
    b = _validate_aabb_batch(boxes_b, name="boxes_b")
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    a_min = a[:, None, :3] - 0.5 * a[:, None, 3:6]
    a_max = a[:, None, :3] + 0.5 * a[:, None, 3:6]
    b_min = b[None, :, :3] - 0.5 * b[None, :, 3:6]
    b_max = b[None, :, :3] + 0.5 * b[None, :, 3:6]
    intersection_size = np.maximum(
        np.minimum(a_max, b_max) - np.maximum(a_min, b_min), 0.0
    )
    intersection = np.prod(intersection_size, axis=-1)
    volume_a = np.prod(a[:, 3:6], axis=1)[:, None]
    volume_b = np.prod(b[:, 3:6], axis=1)[None, :]
    union = volume_a + volume_b - intersection
    iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )
    return np.clip(iou, 0.0, 1.0).astype(np.float32)


def aabb_iou_3d(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU between one center-size AABB and a batch of AABBs."""

    query = np.asarray(box)
    if query.shape != (6,):
        raise ValueError("box must have shape [6]")
    return pairwise_aabb_iou_3d(query, boxes)[0]


def soft_nms_aabb_3d(
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    method: str = "linear",
    iou_threshold: float = 0.25,
    sigma: float = 0.5,
    score_threshold: float = 1e-3,
    max_detections: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stable greedy Soft-NMS for center-size 3D AABBs.

    Returns:
        ``(indices, decayed_scores)`` in final descending selection order.
        Equal scores are always resolved by the lower original index.
    """

    box_values = _validate_aabb_batch(boxes)
    score_values = np.asarray(scores, dtype=np.float64)
    if score_values.ndim != 1 or score_values.shape[0] != box_values.shape[0]:
        raise ValueError("scores must have shape [N]")
    if not np.isfinite(score_values).all():
        raise ValueError("scores must be finite")
    if ((score_values < 0.0) | (score_values > 1.0)).any():
        raise ValueError("scores must lie in [0, 1]")
    if not isinstance(method, str):
        raise TypeError("method must be a string")
    method = method.lower()
    if method not in {"linear", "gaussian", "hard"}:
        raise ValueError("method must be 'linear', 'gaussian', or 'hard'")

    scalar_bounds = {
        "iou_threshold": (iou_threshold, 0.0, 1.0),
        "score_threshold": (score_threshold, 0.0, 1.0),
    }
    for name, (value, lower, upper) in scalar_bounds.items():
        if (
            not np.isscalar(value)
            or not np.isfinite(value)
            or float(value) < lower
            or float(value) > upper
        ):
            raise ValueError(f"{name} must lie in [{lower}, {upper}]")
    if not np.isscalar(sigma) or not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be positive and finite")
    if max_detections is not None:
        if (
            isinstance(max_detections, bool)
            or not isinstance(max_detections, (int, np.integer))
            or int(max_detections) < 1
        ):
            raise ValueError("max_detections must be a positive integer")
        max_detections = int(max_detections)

    if box_values.shape[0] == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )

    updated_scores = score_values.copy()
    active = np.flatnonzero(
        updated_scores >= float(score_threshold)
    ).astype(np.int64)
    selected_indices = []
    selected_scores = []

    while active.size:
        # lexsort uses the final key as primary: descending score first, then
        # the original index for deterministic tie-breaking.
        ranking = np.lexsort((active, -updated_scores[active]))
        best = int(active[ranking[0]])
        selected_indices.append(best)
        selected_scores.append(float(updated_scores[best]))
        if (
            max_detections is not None
            and len(selected_indices) >= max_detections
        ):
            break

        active = active[active != best]
        if active.size == 0:
            break
        overlaps = aabb_iou_3d(box_values[best], box_values[active]).astype(
            np.float64
        )
        if method == "linear":
            decay = np.ones_like(overlaps)
            over_threshold = overlaps > float(iou_threshold)
            decay[over_threshold] = 1.0 - overlaps[over_threshold]
        elif method == "gaussian":
            decay = np.exp(-(overlaps * overlaps) / float(sigma))
        else:
            decay = (overlaps <= float(iou_threshold)).astype(np.float64)
        updated_scores[active] *= decay
        active = active[
            updated_scores[active] >= float(score_threshold)
        ]

    return (
        np.asarray(selected_indices, dtype=np.int64),
        np.asarray(selected_scores, dtype=np.float32),
    )


__all__ = [
    "QUALITY_FEATURE_NAMES",
    "QUALITY_FEATURE_DIM",
    "DEFAULT_HEURISTIC_WEIGHTS",
    "quality_feature_vector",
    "quality_feature_matrix",
    "HeuristicQualityScorer",
    "LinearQualityScorer",
    "MLPQualityScorer",
    "make_quality_scorer",
    "load_quality_scorer",
    "aabb_iou_3d",
    "pairwise_aabb_iou_3d",
    "soft_nms_aabb_3d",
]
