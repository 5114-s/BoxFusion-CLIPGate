#!/usr/bin/env python3
"""Train the orientation-aware B5-v2 local box refiner on CPU.

Unlike the legacy trainer, this tool requires ``scene_ids`` and splits whole
scenes, never individual samples.  Training batches are deterministically
balanced between reachable geometry improvements and rejection examples.
Geometry losses are evaluated only on the former; negative samples supervise
only the binary quality/acceptance head.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as error:  # pragma: no cover
    raise ImportError("training B5-v2 requires PyTorch") from error

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from boxfusion.oriented_box_refiner import (
    OrientedBoxRefinerConfig,
    PointNetOrientedBoxRefiner,
    make_oriented_box_refiner_checkpoint,
)
from boxfusion.quality_score import QUALITY_FEATURE_NAMES
from tools.build_oriented_refiner_dataset import (
    DATASET_FORMAT_VERSION,
    DATASET_SCHEMA,
)


COORDINATE_FRAME = "box_local"
QUALITY_FEATURE_DIM = len(QUALITY_FEATURE_NAMES)
SCENE_PATTERN = re.compile(r"scene\d{4}_\d{2}")
_SAMPLE_KEYS = frozenset(
    {
        "points_local",
        "point_mask",
        "local_boxes",
        "quality_features",
        "target_residual",
        "quality_target",
        "geometry_mask",
        "scene_ids",
        "original_iou",
        "refined_iou",
        "matched_gt_index",
        "target_center_local_unclipped",
        "target_dimensions_local_unclipped",
        "basis_world",
        "result_indices",
        "track_ids",
    }
)
_METADATA_KEYS = frozenset(
    {
        "schema",
        "format_version",
        "coordinate_frame",
        "quality_feature_names",
        "max_center_fraction",
        "max_log_dimension_residual",
    }
)


@dataclass(frozen=True)
class OrientedRefinerTrainingData:
    points_local: np.ndarray
    point_mask: np.ndarray
    local_boxes: np.ndarray
    quality_features: np.ndarray
    target_residual: np.ndarray
    quality_target: np.ndarray
    geometry_mask: np.ndarray
    scene_ids: np.ndarray
    max_center_fraction: float
    max_log_dimension_residual: float

    @property
    def sample_count(self) -> int:
        return int(self.points_local.shape[0])


def _scalar_string(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError(f"{name} must not use object dtype")
    if array.ndim != 0:
        raise ValueError(f"{name} must be scalar")
    scalar = array.item()
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    if not isinstance(scalar, str):
        raise TypeError(f"{name} must be a string")
    return scalar


def _scalar_float(value: np.ndarray, name: str) -> float:
    array = np.asarray(value)
    if array.ndim != 0 or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be a numeric scalar")
    scalar = float(array)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def load_oriented_refiner_dataset(
    path: str | os.PathLike,
) -> OrientedRefinerTrainingData:
    """Load and strictly validate a pickle-free B5-v2 archive."""

    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if dataset_path.suffix.lower() != ".npz":
        raise ValueError("B5-v2 dataset must end in .npz")
    with np.load(dataset_path, allow_pickle=False) as archive:
        keys = set(archive.files)
        expected = _SAMPLE_KEYS | _METADATA_KEYS
        if keys != expected:
            missing = sorted(expected - keys)
            extra = sorted(keys - expected)
            raise ValueError(
                f"B5-v2 dataset keys are invalid: missing={missing}, "
                f"unexpected={extra}"
            )
        if _scalar_string(archive["schema"], "schema") != DATASET_SCHEMA:
            raise ValueError("B5-v2 dataset schema mismatch")
        version = np.asarray(archive["format_version"])
        if (
            version.ndim != 0
            or not np.issubdtype(version.dtype, np.integer)
            or int(version) != DATASET_FORMAT_VERSION
        ):
            raise ValueError("B5-v2 dataset format_version mismatch")
        if (
            _scalar_string(archive["coordinate_frame"], "coordinate_frame")
            != COORDINATE_FRAME
        ):
            raise ValueError("B5-v2 dataset must use box_local coordinates")
        feature_names = tuple(
            str(item)
            for item in np.asarray(archive["quality_feature_names"]).tolist()
        )
        if feature_names != QUALITY_FEATURE_NAMES:
            raise ValueError("quality feature schema/order mismatch")
        arrays = {
            name: np.asarray(archive[name]).copy() for name in _SAMPLE_KEYS
        }
        max_center_fraction = _scalar_float(
            archive["max_center_fraction"], "max_center_fraction"
        )
        max_log_dimension_residual = _scalar_float(
            archive["max_log_dimension_residual"],
            "max_log_dimension_residual",
        )

    if max_center_fraction <= 0.0 or max_log_dimension_residual <= 0.0:
        raise ValueError("stored model residual bounds must be positive")
    points = arrays["points_local"]
    if (
        points.ndim != 3
        or points.shape[2] != 3
        or points.shape[0] < 2
        or points.shape[1] < 1
    ):
        raise ValueError("points_local must have shape [N>=2, P>=1, 3]")
    if not np.issubdtype(points.dtype, np.floating):
        raise TypeError("points_local must use floating-point dtype")
    points = np.asarray(points, dtype=np.float32)
    if not np.isfinite(points).all():
        raise ValueError("points_local must be finite")
    sample_count, point_count, _ = points.shape

    point_mask = arrays["point_mask"]
    if point_mask.shape != (sample_count, point_count):
        raise ValueError("point_mask must have shape [N, P]")
    if point_mask.dtype != np.bool_:
        raise TypeError("point_mask must have Boolean dtype")
    if not point_mask.any(axis=1).all():
        raise ValueError("every sample must contain at least one valid point")
    # Padding must be canonical so it can never encode scene-specific garbage.
    if not np.all(points[~point_mask] == 0.0):
        raise ValueError("masked points_local padding must be exactly zero")

    local_boxes = arrays["local_boxes"]
    quality_features = arrays["quality_features"]
    target_residual = arrays["target_residual"]
    expected_shapes = {
        "local_boxes": (sample_count, 6),
        "quality_features": (sample_count, QUALITY_FEATURE_DIM),
        "target_residual": (sample_count, 6),
    }
    for name, expected_shape in expected_shapes.items():
        value = locals()[name]
        if value.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
        if not np.issubdtype(value.dtype, np.floating):
            raise TypeError(f"{name} must use floating-point dtype")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    local_boxes = np.asarray(local_boxes, dtype=np.float32)
    quality_features = np.asarray(quality_features, dtype=np.float32)
    target_residual = np.asarray(target_residual, dtype=np.float32)
    if not np.allclose(local_boxes[:, :3], 0.0, atol=1e-7):
        raise ValueError("local_boxes centres must be the local origin")
    if (local_boxes[:, 3:6] <= 0.0).any():
        raise ValueError("local_boxes dimensions must be positive")
    if (
        (quality_features < 0.0).any()
        or (quality_features > 1.0).any()
    ):
        raise ValueError("quality_features must lie in [0, 1]")
    if (
        np.abs(target_residual[:, :3]) > max_center_fraction + 1e-5
    ).any():
        raise ValueError("target centre residual exceeds stored model bound")
    if (
        np.abs(target_residual[:, 3:])
        > max_log_dimension_residual + 1e-5
    ).any():
        raise ValueError("target dimension residual exceeds stored model bound")

    geometry_mask = arrays["geometry_mask"]
    if geometry_mask.shape != (sample_count,) or geometry_mask.dtype != np.bool_:
        raise TypeError("geometry_mask must be Boolean with shape [N]")
    quality_target = arrays["quality_target"]
    if quality_target.shape != (sample_count,) or not np.issubdtype(
        quality_target.dtype, np.floating
    ):
        raise ValueError("quality_target must be floating with shape [N]")
    quality_target = np.asarray(quality_target, dtype=np.float32)
    if (
        not np.isfinite(quality_target).all()
        or not np.isin(quality_target, (0.0, 1.0)).all()
    ):
        raise ValueError("quality_target must contain only 0 and 1")
    if not np.array_equal(quality_target.astype(bool), geometry_mask):
        raise ValueError("quality_target must exactly encode geometry_mask")
    if not geometry_mask.any() or geometry_mask.all():
        raise ValueError("dataset must contain geometry positives and negatives")

    scene_ids = arrays["scene_ids"]
    if (
        scene_ids.shape != (sample_count,)
        or scene_ids.dtype.hasobject
        or scene_ids.dtype.kind not in {"U", "S"}
    ):
        raise TypeError("scene_ids must be a non-object string array [N]")
    scene_ids = scene_ids.astype(np.str_)
    if any(SCENE_PATTERN.fullmatch(scene) is None for scene in scene_ids):
        raise ValueError("scene_ids contains an invalid ScanNet scene id")
    if len(np.unique(scene_ids)) < 2:
        raise ValueError("scene-level splitting requires at least two scenes")

    return OrientedRefinerTrainingData(
        points_local=np.ascontiguousarray(points),
        point_mask=np.ascontiguousarray(point_mask),
        local_boxes=np.ascontiguousarray(local_boxes),
        quality_features=np.ascontiguousarray(quality_features),
        target_residual=np.ascontiguousarray(target_residual),
        quality_target=np.ascontiguousarray(quality_target),
        geometry_mask=np.ascontiguousarray(geometry_mask),
        scene_ids=np.ascontiguousarray(scene_ids),
        max_center_fraction=max_center_fraction,
        max_log_dimension_residual=max_log_dimension_residual,
    )


def deterministic_scene_split(
    scene_ids: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split complete scenes deterministically and assert zero leakage."""

    scene_ids = np.asarray(scene_ids)
    if scene_ids.ndim != 1 or scene_ids.dtype.hasobject:
        raise TypeError("scene_ids must be a one-dimensional safe string array")
    unique_scenes = np.unique(scene_ids)
    if len(unique_scenes) < 2:
        raise ValueError("scene split requires at least two unique scenes")
    if (
        not np.isscalar(validation_fraction)
        or not np.isfinite(validation_fraction)
        or not 0.0 < float(validation_fraction) < 1.0
    ):
        raise ValueError("validation_fraction must lie strictly in (0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    validation_scene_count = int(
        round(len(unique_scenes) * float(validation_fraction))
    )
    validation_scene_count = min(
        max(validation_scene_count, 1), len(unique_scenes) - 1
    )
    permutation = np.random.default_rng(int(seed)).permutation(unique_scenes)
    validation_scenes = set(permutation[:validation_scene_count].tolist())
    validation = np.flatnonzero(
        np.isin(scene_ids, list(validation_scenes))
    ).astype(np.int64)
    training = np.flatnonzero(
        ~np.isin(scene_ids, list(validation_scenes))
    ).astype(np.int64)
    if set(scene_ids[training].tolist()) & set(scene_ids[validation].tolist()):
        raise RuntimeError("scene-level split leaked scenes")
    return training, validation


def balanced_epoch_indices(
    training_indices: np.ndarray,
    geometry_mask: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Return a deterministic 50/50 positive-negative epoch sample."""

    indices = np.asarray(training_indices, dtype=np.int64)
    mask = np.asarray(geometry_mask, dtype=np.bool_)
    positives = indices[mask[indices]]
    negatives = indices[~mask[indices]]
    if not len(positives) or not len(negatives):
        raise ValueError(
            "training scenes must contain geometry positives and negatives"
        )
    per_class = max(len(positives), len(negatives))
    rng = np.random.default_rng(int(seed))
    positive_sample = rng.choice(
        positives, per_class, replace=len(positives) < per_class
    )
    negative_sample = rng.choice(
        negatives, per_class, replace=len(negatives) < per_class
    )
    combined = np.concatenate((positive_sample, negative_sample))
    rng.shuffle(combined)
    return combined.astype(np.int64, copy=False)


class _ArrayDataset(Dataset):
    def __init__(
        self, data: OrientedRefinerTrainingData, indices: np.ndarray
    ) -> None:
        self.points = torch.from_numpy(data.points_local[indices])
        self.point_mask = torch.from_numpy(data.point_mask[indices])
        self.boxes = torch.from_numpy(data.local_boxes[indices])
        self.quality_features = torch.from_numpy(
            data.quality_features[indices]
        )
        self.target_residual = torch.from_numpy(
            data.target_residual[indices]
        )
        self.quality_target = torch.from_numpy(data.quality_target[indices])
        self.geometry_mask = torch.from_numpy(data.geometry_mask[indices])

    def __len__(self) -> int:
        return int(len(self.points))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            self.points[index],
            self.point_mask[index],
            self.boxes[index],
            self.quality_features[index],
            self.target_residual[index],
            self.quality_target[index],
            self.geometry_mask[index],
        )


def oriented_refiner_loss(
    output: Mapping[str, torch.Tensor],
    target_residual: torch.Tensor,
    quality_target: torch.Tensor,
    geometry_mask: torch.Tensor,
    *,
    center_weight: float = 1.0,
    dimension_weight: float = 1.0,
    quality_weight: float = 1.0,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Masked geometry regression plus all-sample acceptance BCE."""

    required = {
        "center_residual_fraction",
        "log_dimension_residual",
        "quality",
    }
    missing = required - set(output)
    if missing:
        raise ValueError(f"model output is missing {sorted(missing)}")
    if target_residual.ndim != 2 or target_residual.shape[1] != 6:
        raise ValueError("target_residual must have shape [B, 6]")
    batch_size = target_residual.shape[0]
    if quality_target.shape != (batch_size,):
        raise ValueError("quality_target must have shape [B]")
    if geometry_mask.shape != (batch_size,) or geometry_mask.dtype is not torch.bool:
        raise TypeError("geometry_mask must be Boolean with shape [B]")
    if not torch.isfinite(target_residual).all():
        raise ValueError("target_residual must be finite")
    if not torch.all((quality_target == 0.0) | (quality_target == 1.0)):
        raise ValueError("quality_target must be binary")
    if torch.any(geometry_mask):
        center_loss = F.smooth_l1_loss(
            output["center_residual_fraction"][geometry_mask],
            target_residual[geometry_mask, :3],
        )
        dimension_loss = F.smooth_l1_loss(
            output["log_dimension_residual"][geometry_mask],
            target_residual[geometry_mask, 3:],
        )
    else:
        # Validation may legitimately contain no positive scene.  Keep a
        # differentiable zero on the correct device without using negatives.
        center_loss = output["center_residual_fraction"].sum() * 0.0
        dimension_loss = output["log_dimension_residual"].sum() * 0.0
    quality = output["quality"]
    if quality.shape == (batch_size, 1):
        quality = quality[:, 0]
    if quality.shape != (batch_size,):
        raise ValueError("model quality output must have shape [B]")
    quality_loss = F.binary_cross_entropy(
        quality.clamp(1e-6, 1.0 - 1e-6), quality_target
    )
    total = (
        float(center_weight) * center_loss
        + float(dimension_weight) * dimension_loss
        + float(quality_weight) * quality_loss
    )
    prediction = quality >= 0.5
    accuracy = (
        prediction == quality_target.to(dtype=torch.bool)
    ).to(dtype=quality.dtype).mean()
    return total, {
        "loss": total.detach(),
        "center_loss": center_loss.detach(),
        "dimension_loss": dimension_loss.detach(),
        "quality_loss": quality_loss.detach(),
        "quality_accuracy": accuracy.detach(),
        "geometry_positive_fraction": geometry_mask.to(
            dtype=quality.dtype
        ).mean().detach(),
    }


def _set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _run_epoch(
    model: PointNetOrientedBoxRefiner,
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
        target_residual,
        quality_target,
        geometry_mask,
    ) in loader:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(
                points, boxes, quality_features, point_mask=point_mask
            )
            loss, metrics = oriented_refiner_loss(
                output,
                target_residual,
                quality_target,
                geometry_mask,
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


def _positive_finite(name: str, value: float) -> float:
    if (
        not np.isscalar(value)
        or not np.isfinite(value)
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def train_oriented_box_refiner(
    dataset_path: str | os.PathLike,
    output_path: str | os.PathLike,
    *,
    config: Optional[OrientedBoxRefinerConfig] = None,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    validation_fraction: float = 0.2,
    seed: int = 1337,
    center_weight: float = 1.0,
    dimension_weight: float = 1.0,
    quality_weight: float = 1.0,
) -> Dict[str, object]:
    """Train B5-v2 and write a strict runtime-compatible checkpoint."""

    for name, value in (("epochs", epochs), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if int(value) < 1:
            raise ValueError(f"{name} must be positive")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    learning_rate = _positive_finite("learning_rate", learning_rate)
    if (
        not np.isscalar(weight_decay)
        or not np.isfinite(weight_decay)
        or float(weight_decay) < 0.0
    ):
        raise ValueError("weight_decay must be non-negative and finite")
    loss_weights = {
        name: _positive_finite(name, value)
        for name, value in (
            ("center_weight", center_weight),
            ("dimension_weight", dimension_weight),
            ("quality_weight", quality_weight),
        )
    }
    data = load_oriented_refiner_dataset(dataset_path)
    model_config = (config or OrientedBoxRefinerConfig()).validated()
    if model_config.point_feature_dim != 3:
        raise ValueError("B5-v2 training data contains xyz only")
    if model_config.quality_feature_dim != QUALITY_FEATURE_DIM:
        raise ValueError("quality_feature_dim does not match runtime schema")
    if not np.isclose(
        model_config.max_center_fraction,
        data.max_center_fraction,
        atol=1e-7,
    ):
        raise ValueError("model and dataset max_center_fraction differ")
    if not np.isclose(
        model_config.max_log_dimension_residual,
        data.max_log_dimension_residual,
        atol=1e-7,
    ):
        raise ValueError(
            "model and dataset max_log_dimension_residual differ"
        )
    training_indices, validation_indices = deterministic_scene_split(
        data.scene_ids, validation_fraction, seed
    )
    training_scenes = sorted(set(data.scene_ids[training_indices].tolist()))
    validation_scenes = sorted(
        set(data.scene_ids[validation_indices].tolist())
    )
    if set(training_scenes) & set(validation_scenes):
        raise RuntimeError("train/validation scene leakage")
    # Fail before constructing a model: balancing cannot invent a missing
    # class, and silently falling back would violate the B5-v2 protocol.
    balanced_epoch_indices(training_indices, data.geometry_mask, seed)
    _set_determinism(int(seed))

    validation_loader = DataLoader(
        _ArrayDataset(data, validation_indices),
        batch_size=min(int(batch_size), len(validation_indices)),
        shuffle=False,
        num_workers=0,
    )
    model = PointNetOrientedBoxRefiner(model_config).cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(weight_decay),
    )
    train_metrics: Dict[str, float] = {}
    validation_metrics: Dict[str, float] = {}
    best_validation_loss = float("inf")
    best_state: Dict[str, torch.Tensor] | None = None
    for epoch in range(int(epochs)):
        epoch_indices = balanced_epoch_indices(
            training_indices, data.geometry_mask, int(seed) + epoch
        )
        training_loader = DataLoader(
            _ArrayDataset(data, epoch_indices),
            batch_size=min(int(batch_size), len(epoch_indices)),
            shuffle=False,
            num_workers=0,
        )
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
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint state")
    model.load_state_dict(best_state, strict=True)

    output = Path(output_path)
    if output.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("B5-v2 checkpoint must end in .pt or .pth")
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = make_oriented_box_refiner_checkpoint(model)
    temporary = output.with_name(output.name + ".tmp")
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "output": str(output),
        "samples": data.sample_count,
        "train_samples": int(len(training_indices)),
        "validation_samples": int(len(validation_indices)),
        "train_scenes": training_scenes,
        "validation_scenes": validation_scenes,
        "scene_leakage": False,
        "epochs": int(epochs),
        "seed": int(seed),
        "balanced_epoch_samples": int(
            len(
                balanced_epoch_indices(
                    training_indices, data.geometry_mask, seed
                )
            )
        ),
        "best_validation_loss": float(best_validation_loss),
        "train": train_metrics,
        "validation": validation_metrics,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="B5-v2 training NPZ")
    parser.add_argument("--output", required=True, help="output .pt checkpoint")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--point-hidden-dim", type=int, default=64)
    parser.add_argument("--point-embedding-dim", type=int, default=128)
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--max-center-fraction", type=float, default=0.15)
    parser.add_argument(
        "--max-log-dimension-residual",
        type=float,
        default=float(math.log(1.25)),
    )
    parser.add_argument("--center-weight", type=float, default=1.0)
    parser.add_argument("--dimension-weight", type=float, default=1.0)
    parser.add_argument("--quality-weight", type=float, default=1.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    try:
        config = OrientedBoxRefinerConfig(
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
        result = train_oriented_box_refiner(
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
            quality_weight=arguments.quality_weight,
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        parser.error(str(error))
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
