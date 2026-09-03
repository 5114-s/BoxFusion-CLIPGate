#!/usr/bin/env python3
"""Train pickle-free linear or multi-task IoU-aware quality scoring.

The input ``.npz`` must contain ``quality_features`` with shape ``[N, 12]``
and exactly one target array: ``target_iou`` (continuous in ``[0, 1]``),
``target_binary`` (only 0/1), or ``target`` (select semantics with
``--target-kind``).  Optional ``scene_ids`` enable a leakage-safe scene-level
train/validation split.  ``feature_names`` is validated against the immutable
runtime schema.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from boxfusion.quality_score import (
    DEFAULT_IOU_AWARE_RANKING_WEIGHTS,
    IOU_AWARE_OUTPUT_NAMES,
    IOU_AWARE_THRESHOLDS,
    QUALITY_FEATURE_DIM,
    QUALITY_FEATURE_NAMES,
    IoUAwareMLPQualityScorer,
    LinearQualityScorer,
    load_quality_scorer,
)


_TARGET_KEYS = {"target_iou", "target_binary", "target"}
_REFINER_SIDECAR_KEYS = {
    "points",
    "point_mask",
    "boxes",
    "target_boxes",
    "scene_ids",
}


@dataclass(frozen=True)
class QualityTrainingData:
    features: np.ndarray
    targets: np.ndarray
    target_kind: str
    scene_ids: Optional[np.ndarray] = None

    @property
    def sample_count(self) -> int:
        return int(self.features.shape[0])


def _validate_target_kind(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("target_kind must be a string")
    value = value.lower()
    if value not in {"auto", "iou", "binary"}:
        raise ValueError("target_kind must be 'auto', 'iou', or 'binary'")
    return value


def load_quality_dataset(
    path: Union[str, os.PathLike],
    *,
    target_kind: str = "auto",
    require_two_samples: bool = True,
) -> QualityTrainingData:
    """Load and validate fixed-order features and IoU/binary targets."""

    target_kind = _validate_target_kind(target_kind)
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"quality dataset not found: {dataset_path}")
    if dataset_path.suffix.lower() != ".npz":
        raise ValueError("quality dataset must be a .npz archive")

    try:
        with np.load(dataset_path, allow_pickle=False) as archive:
            keys = set(archive.files)
            selected_targets = sorted(keys & _TARGET_KEYS)
            if len(selected_targets) != 1:
                raise ValueError(
                    "quality dataset must contain exactly one of "
                    "target_iou, target_binary, or target"
                )
            expected = {
                "quality_features",
                selected_targets[0],
            }
            # The offline refiner dataset builder deliberately emits
            # ``quality_features`` and ``target_iou`` alongside geometry.
            # Accept those known sidecars so one pickle-free artifact can
            # train both heads, while still rejecting unknown fields.
            allowed = expected | {"feature_names"} | _REFINER_SIDECAR_KEYS
            missing = sorted(expected - keys)
            extra = sorted(keys - allowed)
            if missing or extra:
                details = []
                if missing:
                    details.append(f"missing={missing}")
                if extra:
                    details.append(f"unexpected={extra}")
                raise ValueError(
                    "quality dataset keys are invalid: "
                    + ", ".join(details)
                )
            if "feature_names" in keys:
                names_array = np.asarray(archive["feature_names"])
                if names_array.ndim != 1:
                    raise ValueError(
                        "feature_names must be one-dimensional"
                    )
                names = tuple(str(item) for item in names_array.tolist())
                if names != QUALITY_FEATURE_NAMES:
                    raise ValueError(
                        "feature_names schema/order does not match "
                        "QUALITY_FEATURE_NAMES"
                    )
            features_raw = np.asarray(archive["quality_features"])
            targets_raw = np.asarray(archive[selected_targets[0]])
            scene_ids_raw = (
                np.asarray(archive["scene_ids"])
                if "scene_ids" in keys
                else None
            )
            selected_target_key = selected_targets[0]
    except ValueError:
        raise
    except (OSError, TypeError) as error:
        raise ValueError(
            f"could not read pickle-free dataset {dataset_path}: {error}"
        ) from error

    if (
        features_raw.ndim != 2
        or features_raw.shape[1] != QUALITY_FEATURE_DIM
    ):
        raise ValueError(
            "quality_features must have shape "
            f"[N, {QUALITY_FEATURE_DIM}]"
        )
    if features_raw.shape[0] == 0:
        raise ValueError("quality_features must contain at least one sample")
    if require_two_samples and features_raw.shape[0] < 2:
        raise ValueError("quality training requires at least two samples")
    if not np.issubdtype(features_raw.dtype, np.floating):
        raise TypeError("quality_features must use a floating-point dtype")
    features = np.asarray(features_raw, dtype=np.float64)
    if not np.isfinite(features).all():
        raise ValueError("quality_features must be finite")
    if ((features < 0.0) | (features > 1.0)).any():
        raise ValueError("quality_features must lie in [0, 1]")

    if targets_raw.ndim == 2 and targets_raw.shape[1] == 1:
        targets_raw = targets_raw[:, 0]
    if targets_raw.ndim != 1 or targets_raw.shape[0] != features.shape[0]:
        raise ValueError("quality target must have shape [N] or [N, 1]")
    if not np.issubdtype(targets_raw.dtype, np.number):
        raise TypeError("quality target must be numeric")
    targets = np.asarray(targets_raw, dtype=np.float64)
    if not np.isfinite(targets).all():
        raise ValueError("quality target must be finite")
    if ((targets < 0.0) | (targets > 1.0)).any():
        raise ValueError("quality target must lie in [0, 1]")

    inferred_kind = (
        "iou" if selected_target_key == "target_iou" else "binary"
    )
    if selected_target_key == "target":
        inferred_kind = (
            "binary"
            if np.isin(targets, np.asarray([0.0, 1.0])).all()
            else "iou"
        )
    if target_kind == "auto":
        resolved_kind = inferred_kind
    else:
        resolved_kind = target_kind
        key_kind = {
            "target_iou": "iou",
            "target_binary": "binary",
        }.get(selected_target_key)
        if key_kind is not None and key_kind != target_kind:
            raise ValueError(
                f"{selected_target_key} is incompatible with "
                f"target_kind={target_kind!r}"
            )
    if resolved_kind == "binary" and not np.isin(
        targets, np.asarray([0.0, 1.0])
    ).all():
        raise ValueError("binary quality targets must contain only 0 and 1")

    scene_ids = None
    if scene_ids_raw is not None:
        if scene_ids_raw.ndim != 1 or scene_ids_raw.shape[0] != features.shape[0]:
            raise ValueError("scene_ids must have shape [N]")
        if scene_ids_raw.dtype.hasobject:
            raise TypeError("scene_ids must not use object dtype")
        scene_ids = np.asarray(scene_ids_raw, dtype=np.str_)
        if np.any(np.char.str_len(scene_ids) == 0):
            raise ValueError("scene_ids must be non-empty strings")

    return QualityTrainingData(
        features=np.ascontiguousarray(features),
        targets=np.ascontiguousarray(targets),
        target_kind=resolved_kind,
        scene_ids=scene_ids,
    )


def deterministic_split(
    sample_count: int, validation_fraction: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(sample_count, bool) or not isinstance(
        sample_count, (int, np.integer)
    ):
        raise TypeError("sample_count must be an integer")
    if int(sample_count) < 2:
        raise ValueError("sample_count must be at least two")
    if (
        not np.isscalar(validation_fraction)
        or not np.isfinite(validation_fraction)
        or not 0.0 < float(validation_fraction) < 1.0
    ):
        raise ValueError("validation_fraction must lie strictly in (0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    validation_count = int(round(int(sample_count) * validation_fraction))
    validation_count = min(max(validation_count, 1), int(sample_count) - 1)
    permutation = np.random.default_rng(int(seed)).permutation(
        int(sample_count)
    )
    return (
        np.sort(permutation[validation_count:]).astype(np.int64),
        np.sort(permutation[:validation_count]).astype(np.int64),
    )


def deterministic_group_split(
    group_ids: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split complete scenes so one scene never leaks into both partitions."""

    groups = np.asarray(group_ids)
    if groups.ndim != 1 or groups.size < 2:
        raise ValueError("group_ids must be a one-dimensional array")
    if groups.dtype.hasobject:
        raise TypeError("group_ids must not use object dtype")
    unique = np.unique(groups)
    if unique.size < 2:
        raise ValueError(
            "scene-level validation requires at least two unique scene_ids"
        )
    if (
        not np.isscalar(validation_fraction)
        or not np.isfinite(validation_fraction)
        or not 0.0 < float(validation_fraction) < 1.0
    ):
        raise ValueError("validation_fraction must lie strictly in (0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    validation_group_count = int(round(unique.size * validation_fraction))
    validation_group_count = min(
        max(validation_group_count, 1), unique.size - 1
    )
    shuffled = np.random.default_rng(int(seed)).permutation(unique)
    validation_groups = shuffled[:validation_group_count]
    validation_mask = np.isin(groups, validation_groups)
    training = np.flatnonzero(~validation_mask).astype(np.int64)
    validation = np.flatnonzero(validation_mask).astype(np.int64)
    if training.size == 0 or validation.size == 0:
        raise RuntimeError("scene-level split produced an empty partition")
    return training, validation


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def _binary_cross_entropy(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    bias: float,
) -> float:
    logits = features @ weights + bias
    # max(z, 0) - z*y + log(1 + exp(-abs(z))) is stable BCE.
    losses = (
        np.maximum(logits, 0.0)
        - logits * targets
        + np.log1p(np.exp(-np.abs(logits)))
    )
    return float(losses.mean())


def fit_linear_quality_scorer(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    epochs: int = 1000,
    learning_rate: float = 0.05,
    l2_weight: float = 1e-4,
) -> Tuple[np.ndarray, float]:
    """Fit deterministic full-batch logistic regression with Adam."""

    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if (
        features.ndim != 2
        or features.shape[1] != QUALITY_FEATURE_DIM
        or features.shape[0] == 0
    ):
        raise ValueError(
            "features must have shape "
            f"[N, {QUALITY_FEATURE_DIM}] with N > 0"
        )
    if targets.shape != (features.shape[0],):
        raise ValueError("targets must have shape [N]")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("features and targets must be finite")
    if ((features < 0.0) | (features > 1.0)).any():
        raise ValueError("features must lie in [0, 1]")
    if ((targets < 0.0) | (targets > 1.0)).any():
        raise ValueError("targets must lie in [0, 1]")
    if isinstance(epochs, bool) or not isinstance(
        epochs, (int, np.integer)
    ):
        raise TypeError("epochs must be an integer")
    if int(epochs) < 1:
        raise ValueError("epochs must be positive")
    for name, value, allow_zero in (
        ("learning_rate", learning_rate, False),
        ("l2_weight", l2_weight, True),
    ):
        if (
            not np.isscalar(value)
            or not np.isfinite(value)
            or float(value) < 0.0
            or (not allow_zero and float(value) == 0.0)
        ):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be {qualifier} and finite")

    weights = np.zeros(QUALITY_FEATURE_DIM, dtype=np.float64)
    bias = 0.0
    first_moment = np.zeros_like(weights)
    second_moment = np.zeros_like(weights)
    first_bias = 0.0
    second_bias = 0.0
    beta_one, beta_two, epsilon = 0.9, 0.999, 1e-8
    for step in range(1, int(epochs) + 1):
        probabilities = _stable_sigmoid(features @ weights + bias)
        residual = probabilities - targets
        weight_gradient = (
            features.T @ residual / features.shape[0]
            + float(l2_weight) * weights
        )
        bias_gradient = float(residual.mean())
        first_moment = (
            beta_one * first_moment
            + (1.0 - beta_one) * weight_gradient
        )
        second_moment = (
            beta_two * second_moment
            + (1.0 - beta_two) * weight_gradient * weight_gradient
        )
        first_bias = beta_one * first_bias + (1.0 - beta_one) * bias_gradient
        second_bias = (
            beta_two * second_bias
            + (1.0 - beta_two) * bias_gradient * bias_gradient
        )
        corrected_first = first_moment / (1.0 - beta_one**step)
        corrected_second = second_moment / (1.0 - beta_two**step)
        corrected_first_bias = first_bias / (1.0 - beta_one**step)
        corrected_second_bias = second_bias / (1.0 - beta_two**step)
        weights -= float(learning_rate) * corrected_first / (
            np.sqrt(corrected_second) + epsilon
        )
        bias -= float(learning_rate) * corrected_first_bias / (
            np.sqrt(corrected_second_bias) + epsilon
        )
    return weights.astype(np.float32), float(bias)


def _average_precision_binary(
    scores: np.ndarray, targets: np.ndarray
) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if scores.shape != targets.shape or scores.ndim != 1:
        raise ValueError("scores and targets must be matching vectors")
    positives = targets >= 0.5
    positive_count = int(np.count_nonzero(positives))
    if positive_count == 0:
        return float("nan")
    order = np.lexsort((np.arange(scores.size), -scores))
    ranked = positives[order].astype(np.float64)
    precision = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
    return float(np.sum(precision * ranked) / positive_count)


def _parse_positive_ints(
    values: Sequence[int], *, name: str
) -> Tuple[int, ...]:
    parsed = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) < 1
        ):
            raise ValueError(f"{name} must contain positive integers")
        parsed.append(int(value))
    if not parsed:
        raise ValueError(f"{name} must not be empty")
    return tuple(parsed)


def fit_iou_aware_mlp(
    features: np.ndarray,
    target_iou: np.ndarray,
    *,
    feature_dim: Optional[int] = None,
    strict_threshold_targets: bool = False,
    hidden_dims: Sequence[int] = (64, 32),
    epochs: int = 400,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    iou_loss_weight: float = 1.0,
    threshold_loss_weight: float = 1.0,
    monotonic_loss_weight: float = 0.10,
    seed: int = 1337,
) -> Tuple[
    Tuple[np.ndarray, ...],
    Tuple[np.ndarray, ...],
    np.ndarray,
    np.ndarray,
]:
    """Train a small CPU MLP with continuous and three ordinal IoU targets."""

    features = np.asarray(features, dtype=np.float32)
    exact_target_iou = np.asarray(target_iou, dtype=np.float64)
    target_iou = exact_target_iou.astype(np.float32)
    resolved_feature_dim = (
        QUALITY_FEATURE_DIM if feature_dim is None else int(feature_dim)
    )
    if resolved_feature_dim < 1:
        raise ValueError("feature_dim must be a positive integer")
    if (
        features.ndim != 2
        or features.shape[1] != resolved_feature_dim
        or features.shape[0] < 2
    ):
        raise ValueError(
            "features must have shape "
            f"[N, {resolved_feature_dim}] with N >= 2"
        )
    if target_iou.shape != (features.shape[0],):
        raise ValueError("target_iou must have shape [N]")
    if not np.isfinite(features).all() or not np.isfinite(target_iou).all():
        raise ValueError("training arrays must be finite")
    if ((features < 0.0) | (features > 1.0)).any():
        raise ValueError("features must lie in [0, 1]")
    if ((target_iou < 0.0) | (target_iou > 1.0)).any():
        raise ValueError("target_iou must lie in [0, 1]")
    hidden_dims = _parse_positive_ints(hidden_dims, name="hidden_dims")
    if (
        isinstance(epochs, bool)
        or not isinstance(epochs, (int, np.integer))
        or int(epochs) < 1
    ):
        raise ValueError("epochs must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if not isinstance(strict_threshold_targets, (bool, np.bool_)):
        raise TypeError("strict_threshold_targets must be boolean")
    for name, value, allow_zero in (
        ("learning_rate", learning_rate, False),
        ("weight_decay", weight_decay, True),
        ("iou_loss_weight", iou_loss_weight, True),
        ("threshold_loss_weight", threshold_loss_weight, True),
        ("monotonic_loss_weight", monotonic_loss_weight, True),
    ):
        if (
            not np.isscalar(value)
            or not np.isfinite(value)
            or float(value) < 0.0
            or (not allow_zero and float(value) == 0.0)
        ):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be {qualifier} and finite")
    if float(iou_loss_weight) + float(threshold_loss_weight) <= 0.0:
        raise ValueError("at least one supervised loss weight must be positive")

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required to train iou_mlp") from error

    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    feature_mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = np.maximum(feature_scale, 1e-4).astype(np.float32)
    normalized = (features - feature_mean) / feature_scale

    layers = []
    input_dim = resolved_feature_dim
    for width in hidden_dims:
        layers.extend((torch.nn.Linear(input_dim, width), torch.nn.ReLU()))
        input_dim = width
    layers.append(torch.nn.Linear(input_dim, len(IOU_AWARE_OUTPUT_NAMES)))
    model = torch.nn.Sequential(*layers).cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    feature_tensor = torch.from_numpy(
        np.ascontiguousarray(normalized)
    )
    target_tensor = torch.from_numpy(np.ascontiguousarray(target_iou))
    thresholds = torch.tensor(IOU_AWARE_THRESHOLDS, dtype=torch.float32)
    if strict_threshold_targets:
        # Preserve the evaluator-grade float64 crossing decision before the
        # continuous regression target is reduced to the model's float32.
        binary_targets = torch.from_numpy(
            (exact_target_iou[:, None] > np.asarray(IOU_AWARE_THRESHOLDS)[None, :])
            .astype(np.float32)
        )
    else:
        binary_targets = (
            target_tensor[:, None] >= thresholds[None, :]
        ).to(torch.float32)
    positive_counts = binary_targets.sum(dim=0)
    negative_counts = binary_targets.shape[0] - positive_counts
    positive_weights = negative_counts / positive_counts.clamp_min(1.0)
    positive_weights = positive_weights.clamp(0.25, 8.0)

    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(feature_tensor)
        predicted_iou = torch.sigmoid(logits[:, 0])
        iou_loss = functional.smooth_l1_loss(
            predicted_iou, target_tensor, beta=0.10
        )
        threshold_loss = functional.binary_cross_entropy_with_logits(
            logits[:, 1:],
            binary_targets,
            pos_weight=positive_weights,
        )
        probabilities = torch.sigmoid(logits[:, 1:])
        monotonic_loss = torch.relu(
            probabilities[:, 1:] - probabilities[:, :-1]
        ).mean()
        loss = (
            float(iou_loss_weight) * iou_loss
            + float(threshold_loss_weight) * threshold_loss
            + float(monotonic_loss_weight) * monotonic_loss
        )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite IoU-aware training loss")
        loss.backward()
        optimizer.step()

    weights = []
    biases = []
    for layer in model:
        if isinstance(layer, torch.nn.Linear):
            weights.append(
                layer.weight.detach().cpu().numpy().T.astype(np.float32)
            )
            biases.append(
                layer.bias.detach().cpu().numpy().astype(np.float32)
            )
    return (
        tuple(weights),
        tuple(biases),
        feature_mean,
        feature_scale,
    )


def _iou_aware_metrics(
    scorer: IoUAwareMLPQualityScorer,
    features: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    prediction = scorer.predict(features)
    predicted_iou = np.asarray(
        prediction["predicted_iou"], dtype=np.float64
    )
    metrics: Dict[str, float] = {
        "mae_iou": float(np.mean(np.abs(predicted_iou - targets))),
    }
    for threshold, name in zip(
        IOU_AWARE_THRESHOLDS, IOU_AWARE_OUTPUT_NAMES[1:]
    ):
        probabilities = np.asarray(prediction[name], dtype=np.float64)
        binary = (targets >= threshold).astype(np.float64)
        clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
        metrics[f"bce_{threshold:.2f}"] = float(
            -np.mean(
                binary * np.log(clipped)
                + (1.0 - binary) * np.log(1.0 - clipped)
            )
        )
        metrics[f"ap_{threshold:.2f}"] = _average_precision_binary(
            probabilities, binary
        )
    return metrics


def train_iou_aware_quality_scorer(
    data: QualityTrainingData,
    training: np.ndarray,
    validation: np.ndarray,
    output_path: Union[str, os.PathLike],
    *,
    hidden_dims: Sequence[int] = (64, 32),
    ranking_weights: np.ndarray = DEFAULT_IOU_AWARE_RANKING_WEIGHTS,
    epochs: int = 400,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    iou_loss_weight: float = 1.0,
    threshold_loss_weight: float = 1.0,
    monotonic_loss_weight: float = 0.10,
    seed: int = 1337,
) -> Dict[str, object]:
    if data.target_kind != "iou":
        raise ValueError("iou_mlp requires continuous target_iou supervision")
    rank = np.asarray(ranking_weights, dtype=np.float64)
    if rank.shape != (len(IOU_AWARE_OUTPUT_NAMES),):
        raise ValueError(
            f"ranking_weights must have {len(IOU_AWARE_OUTPUT_NAMES)} values"
        )
    if not np.isfinite(rank).all() or np.any(rank < 0.0) or rank.sum() <= 0.0:
        raise ValueError("ranking_weights must be finite and non-negative")
    rank = rank / rank.sum()
    weights, biases, feature_mean, feature_scale = fit_iou_aware_mlp(
        data.features[training],
        data.targets[training],
        hidden_dims=hidden_dims,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        iou_loss_weight=iou_loss_weight,
        threshold_loss_weight=threshold_loss_weight,
        monotonic_loss_weight=monotonic_loss_weight,
        seed=seed,
    )

    output = Path(output_path)
    if output.suffix.lower() != ".npz":
        raise ValueError("quality checkpoint must end in .npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {
        "format_version": np.asarray(1, dtype=np.int64),
        "feature_names": np.asarray(QUALITY_FEATURE_NAMES),
        "output_names": np.asarray(IOU_AWARE_OUTPUT_NAMES),
        "iou_thresholds": np.asarray(IOU_AWARE_THRESHOLDS, dtype=np.float32),
        "ranking_weights": rank.astype(np.float32),
        "feature_mean": feature_mean.astype(np.float32),
        "feature_scale": feature_scale.astype(np.float32),
        "num_layers": np.asarray(len(weights), dtype=np.int64),
    }
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        arrays[f"weight_{index}"] = weight
        arrays[f"bias_{index}"] = bias
    temporary = output.with_name(output.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, output)
    scorer = load_quality_scorer(output, method="iou_mlp")
    if not isinstance(scorer, IoUAwareMLPQualityScorer):
        raise RuntimeError("runtime loaded the wrong quality scorer type")
    return {
        "output": str(output),
        "target_kind": data.target_kind,
        "model": "iou_mlp",
        "samples": data.sample_count,
        "train_samples": int(training.size),
        "validation_samples": int(validation.size),
        "epochs": int(epochs),
        "seed": int(seed),
        "hidden_dims": list(_parse_positive_ints(hidden_dims, name="hidden_dims")),
        "ranking_weights": rank.tolist(),
        "train": _iou_aware_metrics(
            scorer, data.features[training], data.targets[training]
        ),
        "validation": _iou_aware_metrics(
            scorer, data.features[validation], data.targets[validation]
        ),
    }


def train_quality_calibrator(
    dataset_path: Union[str, os.PathLike],
    output_path: Union[str, os.PathLike],
    *,
    model: str = "linear",
    target_kind: str = "auto",
    epochs: int = 1000,
    learning_rate: float = 0.05,
    l2_weight: float = 1e-4,
    validation_fraction: float = 0.2,
    seed: int = 1337,
    hidden_dims: Sequence[int] = (64, 32),
    ranking_weights: np.ndarray = DEFAULT_IOU_AWARE_RANKING_WEIGHTS,
    iou_loss_weight: float = 1.0,
    threshold_loss_weight: float = 1.0,
    monotonic_loss_weight: float = 0.10,
) -> Dict[str, object]:
    """Fit and save a checkpoint compatible with ``load_quality_scorer``."""

    if not isinstance(model, str):
        raise TypeError("model must be a string")
    model = model.strip().lower()
    if model not in {"linear", "iou_mlp"}:
        raise ValueError("model must be linear or iou_mlp")
    data = load_quality_dataset(dataset_path, target_kind=target_kind)
    if data.scene_ids is None:
        training, validation = deterministic_split(
            data.sample_count, validation_fraction, seed
        )
        split_kind = "sample"
    else:
        training, validation = deterministic_group_split(
            data.scene_ids, validation_fraction, seed
        )
        split_kind = "scene"

    if model == "iou_mlp":
        result = train_iou_aware_quality_scorer(
            data,
            training,
            validation,
            output_path,
            hidden_dims=hidden_dims,
            ranking_weights=ranking_weights,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=l2_weight,
            iou_loss_weight=iou_loss_weight,
            threshold_loss_weight=threshold_loss_weight,
            monotonic_loss_weight=monotonic_loss_weight,
            seed=seed,
        )
        result["split_kind"] = split_kind
        if data.scene_ids is not None:
            result["train_scenes"] = int(
                np.unique(data.scene_ids[training]).size
            )
            result["validation_scenes"] = int(
                np.unique(data.scene_ids[validation]).size
            )
        return result

    weights, bias = fit_linear_quality_scorer(
        data.features[training],
        data.targets[training],
        epochs=epochs,
        learning_rate=learning_rate,
        l2_weight=l2_weight,
    )
    output = Path(output_path)
    if output.suffix.lower() != ".npz":
        raise ValueError("quality checkpoint must end in .npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.npz")
    np.savez(
        temporary,
        feature_names=np.asarray(QUALITY_FEATURE_NAMES),
        weight=weights,
        bias=np.asarray(bias, dtype=np.float32),
    )
    os.replace(temporary, output)
    # Verify that the file is not only syntactically valid, but accepted by
    # the production strict loader.
    scorer = load_quality_scorer(output, method="linear")

    def metrics(indices: np.ndarray) -> Dict[str, float]:
        scores = np.asarray(scorer(data.features[indices]), dtype=np.float64)
        targets = data.targets[indices]
        return {
            "bce": _binary_cross_entropy(
                data.features[indices], targets, weights, bias
            ),
            "mae": float(np.mean(np.abs(scores - targets))),
        }

    result: Dict[str, object] = {
        "output": str(output),
        "target_kind": data.target_kind,
        "model": "linear",
        "split_kind": split_kind,
        "samples": data.sample_count,
        "train_samples": int(training.size),
        "validation_samples": int(validation.size),
        "epochs": int(epochs),
        "seed": int(seed),
        "train": metrics(training),
        "validation": metrics(validation),
    }
    if data.scene_ids is not None:
        result["train_scenes"] = int(
            np.unique(data.scene_ids[training]).size
        )
        result["validation_scenes"] = int(
            np.unique(data.scene_ids[validation]).size
        )
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="training .npz archive")
    parser.add_argument("--output", required=True, help="output .npz weights")
    parser.add_argument(
        "--model",
        choices=("linear", "iou_mlp"),
        default="linear",
    )
    parser.add_argument(
        "--target-kind",
        choices=("auto", "iou", "binary"),
        default="auto",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--l2-weight", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--hidden-dims",
        default="64,32",
        help="comma-separated hidden widths for iou_mlp",
    )
    parser.add_argument(
        "--ranking-weights",
        default="0.10,0.20,0.30,0.40",
        help="predicted-IoU,P15,P25,P50 ranking weights",
    )
    parser.add_argument("--iou-loss-weight", type=float, default=1.0)
    parser.add_argument("--threshold-loss-weight", type=float, default=1.0)
    parser.add_argument("--monotonic-loss-weight", type=float, default=0.10)
    return parser


def _parse_csv_numbers(
    value: str,
    *,
    cast: type,
    expected_count: Optional[int] = None,
) -> Tuple[Any, ...]:
    try:
        parsed = tuple(
            cast(item.strip())
            for item in str(value).split(",")
            if item.strip()
        )
    except ValueError as error:
        raise ValueError(f"invalid comma-separated value: {value!r}") from error
    if not parsed:
        raise ValueError("comma-separated value must not be empty")
    if expected_count is not None and len(parsed) != expected_count:
        raise ValueError(
            f"expected {expected_count} comma-separated values, "
            f"received {len(parsed)}"
        )
    return parsed


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    try:
        hidden_dims = _parse_csv_numbers(
            arguments.hidden_dims, cast=int
        )
        ranking_weights = np.asarray(
            _parse_csv_numbers(
                arguments.ranking_weights,
                cast=float,
                expected_count=len(IOU_AWARE_OUTPUT_NAMES),
            ),
            dtype=np.float64,
        )
        epochs = (
            arguments.epochs
            if arguments.epochs is not None
            else (400 if arguments.model == "iou_mlp" else 1000)
        )
        learning_rate = (
            arguments.learning_rate
            if arguments.learning_rate is not None
            else (1e-3 if arguments.model == "iou_mlp" else 0.05)
        )
        result = train_quality_calibrator(
            arguments.input,
            arguments.output,
            model=arguments.model,
            target_kind=arguments.target_kind,
            epochs=epochs,
            learning_rate=learning_rate,
            l2_weight=arguments.l2_weight,
            validation_fraction=arguments.validation_fraction,
            seed=arguments.seed,
            hidden_dims=hidden_dims,
            ranking_weights=ranking_weights,
            iou_loss_weight=arguments.iou_loss_weight,
            threshold_loss_weight=arguments.threshold_loss_weight,
            monotonic_loss_weight=arguments.monotonic_loss_weight,
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
