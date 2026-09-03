#!/usr/bin/env python3
"""Train the lightweight BoxFusion object-local box refiner on CPU.

The input is a pickle-free ``.npz`` archive with the following arrays:

``points``
    Float ``[N, P, 3]`` object-local observations in world coordinates.
``point_mask``
    Boolean ``[N, P]`` valid-point mask.
``boxes``
    Float ``[N, 6]`` input AABBs as ``cx, cy, cz, dx, dy, dz``.
``quality_features``
    Float ``[N, 12]`` matrix in ``QUALITY_FEATURE_NAMES`` order.
``target_boxes``
    Float ``[N, 6]`` target AABBs in the same convention.
``target_iou`` (optional)
    Float ``[N]`` matched-target IoU used by the quality head.  When omitted,
    it is deterministically derived from ``boxes`` and ``target_boxes``.

An optional ``feature_names`` array is accepted and, when present, must
exactly match the runtime quality schema.  The output is the strict wrapped
checkpoint consumed by :func:`boxfusion.box_refiner.load_box_refiner`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as error:  # pragma: no cover - actionable CLI failure.
    raise ImportError("training the box refiner requires PyTorch") from error

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from boxfusion.box_refiner import BoxRefinerConfig, PointNetBoxRefiner
from boxfusion.quality_score import QUALITY_FEATURE_DIM, QUALITY_FEATURE_NAMES


_REQUIRED_KEYS = {
    "points",
    "point_mask",
    "boxes",
    "quality_features",
    "target_boxes",
}
_OPTIONAL_KEYS = {"feature_names", "target_iou", "scene_ids"}


@dataclass(frozen=True)
class BoxRefinerTrainingData:
    """Validated NumPy arrays used by the refiner trainer."""

    points: np.ndarray
    point_mask: np.ndarray
    boxes: np.ndarray
    quality_features: np.ndarray
    target_boxes: np.ndarray
    target_iou: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.points.shape[0])


def _require_finite_float_array(
    value: np.ndarray,
    *,
    name: str,
    shape_tail: Tuple[int, ...],
    ndim: int,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim or tuple(array.shape[-len(shape_tail) :]) != shape_tail:
        expected = ", ".join(str(item) for item in shape_tail)
        raise ValueError(
            f"{name} must have {ndim} dimensions ending in [{expected}]"
        )
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must use a floating-point dtype")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _validate_feature_names(value: np.ndarray) -> None:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError("feature_names must be one-dimensional")
    names = tuple(str(item) for item in array.tolist())
    if names != QUALITY_FEATURE_NAMES:
        raise ValueError(
            "feature_names schema/order does not match QUALITY_FEATURE_NAMES"
        )


def load_box_refiner_dataset(
    path: Union[str, os.PathLike],
    *,
    require_two_samples: bool = True,
) -> BoxRefinerTrainingData:
    """Load and strictly validate a pickle-free box-refiner dataset."""

    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"box-refiner dataset not found: {dataset_path}")
    if dataset_path.suffix.lower() != ".npz":
        raise ValueError("box-refiner dataset must be a .npz archive")

    try:
        with np.load(dataset_path, allow_pickle=False) as archive:
            keys = set(archive.files)
            missing = sorted(_REQUIRED_KEYS - keys)
            extra = sorted(keys - _REQUIRED_KEYS - _OPTIONAL_KEYS)
            if missing or extra:
                details = []
                if missing:
                    details.append(f"missing={missing}")
                if extra:
                    details.append(f"unexpected={extra}")
                raise ValueError(
                    "box-refiner dataset keys are invalid: "
                    + ", ".join(details)
                )
            if "feature_names" in keys:
                _validate_feature_names(archive["feature_names"])

            points = _require_finite_float_array(
                archive["points"],
                name="points",
                shape_tail=(3,),
                ndim=3,
            )
            point_mask_raw = np.asarray(archive["point_mask"])
            boxes = _require_finite_float_array(
                archive["boxes"],
                name="boxes",
                shape_tail=(6,),
                ndim=2,
            )
            quality_features = _require_finite_float_array(
                archive["quality_features"],
                name="quality_features",
                shape_tail=(QUALITY_FEATURE_DIM,),
                ndim=2,
            )
            target_boxes = _require_finite_float_array(
                archive["target_boxes"],
                name="target_boxes",
                shape_tail=(6,),
                ndim=2,
            )
            target_iou_raw = (
                np.asarray(archive["target_iou"])
                if "target_iou" in keys
                else None
            )
    except ValueError:
        raise
    except (OSError, TypeError) as error:
        raise ValueError(
            f"could not read pickle-free dataset {dataset_path}: {error}"
        ) from error

    sample_count, point_count, _ = points.shape
    if point_count == 0:
        raise ValueError("points must contain at least one point per sample")
    if require_two_samples and sample_count < 2:
        raise ValueError(
            "box-refiner training requires at least two samples"
        )
    expected_first_dimensions = {
        "point_mask": point_mask_raw.shape[:1],
        "boxes": boxes.shape[:1],
        "quality_features": quality_features.shape[:1],
        "target_boxes": target_boxes.shape[:1],
    }
    for name, first_dimension in expected_first_dimensions.items():
        if first_dimension != (sample_count,):
            raise ValueError(
                f"{name} sample count must match points ({sample_count})"
            )
    if point_mask_raw.shape != (sample_count, point_count):
        raise ValueError(
            "point_mask must have shape "
            f"[{sample_count}, {point_count}]"
        )
    if point_mask_raw.dtype != np.bool_:
        raise TypeError("point_mask must have Boolean dtype")
    point_mask = np.asarray(point_mask_raw, dtype=np.bool_)
    if not point_mask.any(axis=1).all():
        raise ValueError("every sample must contain at least one valid point")
    if not (boxes[:, 3:6] > 0.0).all():
        raise ValueError("boxes dimensions must be positive")
    if not (target_boxes[:, 3:6] > 0.0).all():
        raise ValueError("target_boxes dimensions must be positive")
    if (
        (quality_features < 0.0).any()
        or (quality_features > 1.0).any()
    ):
        raise ValueError("quality_features must lie in [0, 1]")
    if target_iou_raw is None:
        boxes_min = boxes[:, :3] - 0.5 * boxes[:, 3:6]
        boxes_max = boxes[:, :3] + 0.5 * boxes[:, 3:6]
        targets_min = target_boxes[:, :3] - 0.5 * target_boxes[:, 3:6]
        targets_max = target_boxes[:, :3] + 0.5 * target_boxes[:, 3:6]
        intersection = np.maximum(
            np.minimum(boxes_max, targets_max)
            - np.maximum(boxes_min, targets_min),
            0.0,
        ).prod(axis=1)
        union = (
            boxes[:, 3:6].prod(axis=1)
            + target_boxes[:, 3:6].prod(axis=1)
            - intersection
        )
        target_iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0.0,
        ).astype(np.float32)
    else:
        if (
            target_iou_raw.ndim == 2
            and target_iou_raw.shape == (sample_count, 1)
        ):
            target_iou_raw = target_iou_raw[:, 0]
        if target_iou_raw.shape != (sample_count,):
            raise ValueError("target_iou must have shape [N] or [N, 1]")
        if not np.issubdtype(target_iou_raw.dtype, np.number):
            raise TypeError("target_iou must be numeric")
        target_iou = np.asarray(target_iou_raw, dtype=np.float32)
        if not np.isfinite(target_iou).all():
            raise ValueError("target_iou must be finite")
        if ((target_iou < 0.0) | (target_iou > 1.0)).any():
            raise ValueError("target_iou must lie in [0, 1]")

    return BoxRefinerTrainingData(
        points=np.ascontiguousarray(points),
        point_mask=np.ascontiguousarray(point_mask),
        boxes=np.ascontiguousarray(boxes),
        quality_features=np.ascontiguousarray(quality_features),
        target_boxes=np.ascontiguousarray(target_boxes),
        target_iou=np.ascontiguousarray(target_iou),
    )


def deterministic_split(
    sample_count: int, validation_fraction: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return deterministic non-empty train and validation index arrays."""

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
    validation = np.sort(permutation[:validation_count]).astype(np.int64)
    training = np.sort(permutation[validation_count:]).astype(np.int64)
    return training, validation


class _ArrayDataset(Dataset):
    def __init__(
        self, data: BoxRefinerTrainingData, indices: np.ndarray
    ) -> None:
        self.points = torch.from_numpy(data.points[indices])
        self.point_mask = torch.from_numpy(data.point_mask[indices])
        self.boxes = torch.from_numpy(data.boxes[indices])
        self.quality_features = torch.from_numpy(
            data.quality_features[indices]
        )
        self.target_boxes = torch.from_numpy(data.target_boxes[indices])
        self.target_iou = torch.from_numpy(data.target_iou[indices])

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.points[index],
            self.point_mask[index],
            self.boxes[index],
            self.quality_features[index],
            self.target_boxes[index],
            self.target_iou[index],
        )


def _aligned_aabb_iou(
    predicted_boxes: torch.Tensor, target_boxes: torch.Tensor
) -> torch.Tensor:
    predicted_min = (
        predicted_boxes[:, :3] - 0.5 * predicted_boxes[:, 3:6]
    )
    predicted_max = (
        predicted_boxes[:, :3] + 0.5 * predicted_boxes[:, 3:6]
    )
    target_min = target_boxes[:, :3] - 0.5 * target_boxes[:, 3:6]
    target_max = target_boxes[:, :3] + 0.5 * target_boxes[:, 3:6]
    intersection_dims = (
        torch.minimum(predicted_max, target_max)
        - torch.maximum(predicted_min, target_min)
    ).clamp_min(0.0)
    intersection = intersection_dims.prod(dim=1)
    predicted_volume = predicted_boxes[:, 3:6].prod(dim=1)
    target_volume = target_boxes[:, 3:6].prod(dim=1)
    union = predicted_volume + target_volume - intersection
    return intersection / union.clamp_min(torch.finfo(union.dtype).eps)


def box_refiner_loss(
    output: Mapping[str, torch.Tensor],
    boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    quality_target: Optional[torch.Tensor] = None,
    *,
    center_weight: float = 1.0,
    dimension_weight: float = 1.0,
    iou_weight: float = 1.0,
    quality_weight: float = 0.25,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Smooth-L1 residual loss plus differentiable IoU and quality targets."""

    target_center_fraction = (
        target_boxes[:, :3] - boxes[:, :3]
    ) / boxes[:, 3:6].clamp_min(1e-6)
    target_log_dimensions = torch.log(
        target_boxes[:, 3:6] / boxes[:, 3:6].clamp_min(1e-6)
    )
    center_loss = F.smooth_l1_loss(
        output["center_residual_fraction"], target_center_fraction
    )
    dimension_loss = F.smooth_l1_loss(
        output["log_dimension_residual"], target_log_dimensions
    )

    predicted_boxes = torch.cat(
        (
            boxes[:, :3] + output["center_residual"],
            boxes[:, 3:6] * torch.exp(output["log_dimension_residual"]),
        ),
        dim=1,
    )
    iou = _aligned_aabb_iou(predicted_boxes, target_boxes)
    iou_loss = 1.0 - iou.mean()
    if quality_target is None:
        quality_target = iou.detach()
    elif quality_target.shape != iou.shape:
        raise ValueError("quality_target must have shape [B]")
    if not torch.isfinite(quality_target).all():
        raise ValueError("quality_target must be finite")
    if not torch.all((quality_target >= 0.0) & (quality_target <= 1.0)):
        raise ValueError("quality_target must lie in [0, 1]")
    quality_target = quality_target.to(
        device=iou.device, dtype=iou.dtype
    ).detach().clamp(0.0, 1.0)
    quality_loss = F.binary_cross_entropy(
        output["quality"].clamp(1e-6, 1.0 - 1e-6),
        quality_target,
    )
    total = (
        float(center_weight) * center_loss
        + float(dimension_weight) * dimension_loss
        + float(iou_weight) * iou_loss
        + float(quality_weight) * quality_loss
    )
    return total, {
        "loss": total.detach(),
        "center_loss": center_loss.detach(),
        "dimension_loss": dimension_loss.detach(),
        "iou_loss": iou_loss.detach(),
        "quality_loss": quality_loss.detach(),
        "mean_iou": iou.detach().mean(),
    }


def _validate_positive_scalar(name: str, value: float) -> float:
    if not np.isscalar(value) or not np.isfinite(value) or float(value) <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch, "set_num_threads"):
        torch.set_num_threads(1)


def _run_epoch(
    model: PointNetBoxRefiner,
    loader: DataLoader,
    *,
    optimizer: Optional[torch.optim.Optimizer],
    loss_weights: Mapping[str, float],
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: Dict[str, float] = {}
    total_samples = 0
    for (
        points,
        point_mask,
        boxes,
        quality_features,
        targets,
        target_iou,
    ) in loader:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(points, boxes, quality_features, point_mask)
            loss, metrics = box_refiner_loss(
                output,
                boxes,
                targets,
                target_iou,
                **loss_weights,
            )
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        batch_size = int(points.shape[0])
        total_samples += batch_size
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value) * batch_size
    if total_samples == 0:
        raise ValueError("data loader produced no samples")
    return {name: value / total_samples for name, value in totals.items()}


def train_box_refiner(
    dataset_path: Union[str, os.PathLike],
    output_path: Union[str, os.PathLike],
    *,
    config: Optional[BoxRefinerConfig] = None,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    validation_fraction: float = 0.2,
    seed: int = 1337,
    center_weight: float = 1.0,
    dimension_weight: float = 1.0,
    iou_weight: float = 1.0,
    quality_weight: float = 0.25,
) -> Dict[str, object]:
    """Train on CPU and write a runtime-compatible strict checkpoint."""

    for name, value in (("epochs", epochs), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if int(value) < 1:
            raise ValueError(f"{name} must be positive")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    learning_rate = _validate_positive_scalar(
        "learning_rate", learning_rate
    )
    if (
        not np.isscalar(weight_decay)
        or not np.isfinite(weight_decay)
        or float(weight_decay) < 0.0
    ):
        raise ValueError("weight_decay must be non-negative and finite")
    loss_weights = {}
    for name, value in (
        ("center_weight", center_weight),
        ("dimension_weight", dimension_weight),
        ("iou_weight", iou_weight),
        ("quality_weight", quality_weight),
    ):
        loss_weights[name] = _validate_positive_scalar(name, value)

    data = load_box_refiner_dataset(dataset_path)
    model_config = (config or BoxRefinerConfig()).validated()
    if model_config.point_feature_dim != 3:
        raise ValueError(
            "training data contains xyz only; point_feature_dim must equal 3"
        )
    if model_config.quality_feature_dim != QUALITY_FEATURE_DIM:
        raise ValueError(
            "BoxRefiner quality_feature_dim must match the fixed "
            f"{QUALITY_FEATURE_DIM}-feature schema"
        )
    training_indices, validation_indices = deterministic_split(
        data.sample_count, validation_fraction, seed
    )
    _set_determinism(int(seed))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    training_loader = DataLoader(
        _ArrayDataset(data, training_indices),
        batch_size=min(int(batch_size), len(training_indices)),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        _ArrayDataset(data, validation_indices),
        batch_size=min(int(batch_size), len(validation_indices)),
        shuffle=False,
        num_workers=0,
    )
    model = PointNetBoxRefiner(model_config).cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(weight_decay),
    )

    train_metrics: Dict[str, float] = {}
    validation_metrics: Dict[str, float] = {}
    for _ in range(int(epochs)):
        train_metrics = _run_epoch(
            model,
            training_loader,
            optimizer=optimizer,
            loss_weights=loss_weights,
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            optimizer=None,
            loss_weights=loss_weights,
        )

    output = Path(output_path)
    if output.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("box-refiner checkpoint must end in .pt or .pth")
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": model_config.architecture_dict(),
    }
    temporary = output.with_name(output.name + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, output)

    return {
        "output": str(output),
        "samples": data.sample_count,
        "train_samples": int(training_indices.size),
        "validation_samples": int(validation_indices.size),
        "epochs": int(epochs),
        "seed": int(seed),
        "train": train_metrics,
        "validation": validation_metrics,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="training .npz archive")
    parser.add_argument("--output", required=True, help="output .pt checkpoint")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--point-hidden-dim", type=int, default=64)
    parser.add_argument("--point-embedding-dim", type=int, default=128)
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--max-center-fraction", type=float, default=0.25)
    parser.add_argument(
        "--max-log-dimension-residual",
        type=float,
        default=float(math.log(1.5)),
    )
    parser.add_argument("--center-weight", type=float, default=1.0)
    parser.add_argument("--dimension-weight", type=float, default=1.0)
    parser.add_argument("--iou-weight", type=float, default=1.0)
    parser.add_argument("--quality-weight", type=float, default=0.25)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    try:
        config = BoxRefinerConfig(
            point_feature_dim=3,
            quality_feature_dim=QUALITY_FEATURE_DIM,
            point_hidden_dim=arguments.point_hidden_dim,
            point_embedding_dim=arguments.point_embedding_dim,
            head_hidden_dim=arguments.head_hidden_dim,
            max_center_fraction=arguments.max_center_fraction,
            max_log_dimension_residual=(
                arguments.max_log_dimension_residual
            ),
        ).validated()
        result = train_box_refiner(
            arguments.input,
            arguments.output,
            config=config,
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            weight_decay=arguments.weight_decay,
            validation_fraction=arguments.validation_fraction,
            seed=arguments.seed,
            center_weight=arguments.center_weight,
            dimension_weight=arguments.dimension_weight,
            iou_weight=arguments.iou_weight,
            quality_weight=arguments.quality_weight,
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        parser.error(str(error))
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
