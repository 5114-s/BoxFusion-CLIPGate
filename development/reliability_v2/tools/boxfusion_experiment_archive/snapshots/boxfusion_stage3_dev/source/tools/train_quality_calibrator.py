#!/usr/bin/env python3
"""Train a fixed-schema linear quality calibrator without pickle.

The input ``.npz`` must contain ``quality_features`` with shape ``[N, 12]``
and exactly one target array: ``target_iou`` (continuous in ``[0, 1]``),
``target_binary`` (only 0/1), or ``target`` (select semantics with
``--target-kind``).  An optional ``feature_names`` array is validated against
the immutable runtime schema.
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
    QUALITY_FEATURE_DIM,
    QUALITY_FEATURE_NAMES,
    LinearQualityScorer,
    load_quality_scorer,
)


_TARGET_KEYS = {"target_iou", "target_binary", "target"}
_REFINER_SIDECAR_KEYS = {
    "points",
    "point_mask",
    "boxes",
    "target_boxes",
}


@dataclass(frozen=True)
class QualityTrainingData:
    features: np.ndarray
    targets: np.ndarray
    target_kind: str

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

    return QualityTrainingData(
        features=np.ascontiguousarray(features),
        targets=np.ascontiguousarray(targets),
        target_kind=resolved_kind,
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


def train_quality_calibrator(
    dataset_path: Union[str, os.PathLike],
    output_path: Union[str, os.PathLike],
    *,
    target_kind: str = "auto",
    epochs: int = 1000,
    learning_rate: float = 0.05,
    l2_weight: float = 1e-4,
    validation_fraction: float = 0.2,
    seed: int = 1337,
) -> Dict[str, object]:
    """Fit and save a checkpoint compatible with ``load_quality_scorer``."""

    data = load_quality_dataset(dataset_path, target_kind=target_kind)
    training, validation = deterministic_split(
        data.sample_count, validation_fraction, seed
    )
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

    return {
        "output": str(output),
        "target_kind": data.target_kind,
        "samples": data.sample_count,
        "train_samples": int(training.size),
        "validation_samples": int(validation.size),
        "epochs": int(epochs),
        "seed": int(seed),
        "train": metrics(training),
        "validation": metrics(validation),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="training .npz archive")
    parser.add_argument("--output", required=True, help="output .npz weights")
    parser.add_argument(
        "--target-kind",
        choices=("auto", "iou", "binary"),
        default="auto",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-weight", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    try:
        result = train_quality_calibrator(
            arguments.input,
            arguments.output,
            target_kind=arguments.target_kind,
            epochs=arguments.epochs,
            learning_rate=arguments.learning_rate,
            l2_weight=arguments.l2_weight,
            validation_fraction=arguments.validation_fraction,
            seed=arguments.seed,
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        parser.error(str(error))
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
