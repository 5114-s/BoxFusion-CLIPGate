"""Lightweight object-local point-cloud box refinement.

The online integration is deliberately split into two layers:

* :class:`PointNetBoxRefiner` is a small PyTorch model that predicts bounded
  center and log-dimension residuals plus a quality probability.
* :func:`apply_box_residual_numpy` is a dependency-light, defensive NumPy
  implementation used by the BoxFusion runtime.

Importing this module remains possible when PyTorch is not installed.  NumPy
utilities continue to work, while constructing/loading the neural refiner
raises an actionable :class:`ImportError`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np

try:  # Keep geometry-only users independent from the PyTorch installation.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised in torch-free deployments.
    torch = None
    nn = None


DEFAULT_QUALITY_FEATURE_DIM = 12


@dataclass(frozen=True)
class BoxRefinerConfig:
    """Architecture and safety bounds for :class:`PointNetBoxRefiner`."""

    point_feature_dim: int = 3
    quality_feature_dim: int = DEFAULT_QUALITY_FEATURE_DIM
    point_hidden_dim: int = 64
    point_embedding_dim: int = 128
    head_hidden_dim: int = 128
    max_center_fraction: float = 0.25
    max_log_dimension_residual: float = float(np.log(1.5))
    minimum_dimension: float = 1e-3
    normalized_point_limit: float = 4.0

    def validated(self) -> "BoxRefinerConfig":
        """Validate all fields and return ``self`` for fluent construction."""

        integer_fields = (
            "point_feature_dim",
            "quality_feature_dim",
            "point_hidden_dim",
            "point_embedding_dim",
            "head_hidden_dim",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(f"{name} must be an integer")
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.point_feature_dim < 3:
            raise ValueError("point_feature_dim must include at least xyz")

        positive_fields = (
            "max_center_fraction",
            "max_log_dimension_residual",
            "minimum_dimension",
            "normalized_point_limit",
        )
        for name in positive_fields:
            value = getattr(self, name)
            if not np.isscalar(value) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite scalar")
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive")
        return self

    def architecture_dict(self) -> Dict[str, Union[int, float]]:
        """Return the exact architecture signature stored in checkpoints."""

        return asdict(self)


_ModuleBase = nn.Module if nn is not None else object


class PointNetBoxRefiner(_ModuleBase):
    """A compact PointNet-style residual and quality head.

    Args:
        config: Model architecture and output bounds.

    Inputs to :meth:`forward`:
        points: ``[B, N, C]`` object-local observations in world coordinates.
        boxes: ``[B, 6]`` boxes as ``[cx, cy, cz, dx, dy, dz]``.
        quality_features: ``[B, F]`` finite features in ``[0, 1]``.
        point_mask: Optional Boolean ``[B, N]`` valid-point mask.

    The center residual is returned in metric units and is bounded per axis by
    ``max_center_fraction * original_dimension``.  Dimension residuals are in
    log space and bounded by ``max_log_dimension_residual``.
    """

    def __init__(self, config: Optional[BoxRefinerConfig] = None) -> None:
        if nn is None or torch is None:
            raise ImportError(
                "PointNetBoxRefiner requires PyTorch; NumPy box utilities "
                "remain available without it"
            )
        super().__init__()
        self.config = (config or BoxRefinerConfig()).validated()

        cfg = self.config
        self.point_mlp = nn.Sequential(
            nn.Linear(cfg.point_feature_dim, cfg.point_hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(cfg.point_hidden_dim, cfg.point_embedding_dim),
            nn.ReLU(inplace=False),
        )
        head_input_dim = (
            2 * cfg.point_embedding_dim + cfg.quality_feature_dim
        )
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, cfg.head_hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(cfg.head_hidden_dim, cfg.head_hidden_dim),
            nn.ReLU(inplace=False),
        )
        self.output_layer = nn.Linear(cfg.head_hidden_dim, 7)

        # A checkpoint-free model is a safe identity refiner with quality 0.5.
        # Production enabling still requires a checkpoint via
        # ``build_box_refiner``.
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    @staticmethod
    def _require_tensor(name: str, value: Any) -> Any:
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a torch.Tensor")
        return value

    def _validate_inputs(
        self,
        points: Any,
        boxes: Any,
        quality_features: Any,
        point_mask: Optional[Any],
    ) -> Tuple[Any, Any, Any, Any]:
        points = self._require_tensor("points", points)
        boxes = self._require_tensor("boxes", boxes)
        quality_features = self._require_tensor(
            "quality_features", quality_features
        )

        if points.ndim != 3:
            raise ValueError("points must have shape [B, N, C]")
        batch_size, point_count, feature_dim = points.shape
        if point_count < 1:
            raise ValueError("points must contain at least one point")
        if feature_dim != self.config.point_feature_dim:
            raise ValueError(
                "points feature dimension must equal "
                f"{self.config.point_feature_dim}, received {feature_dim}"
            )
        if boxes.ndim != 2 or boxes.shape != (batch_size, 6):
            raise ValueError("boxes must have shape [B, 6]")
        expected_quality_shape = (
            batch_size,
            self.config.quality_feature_dim,
        )
        if (
            quality_features.ndim != 2
            or quality_features.shape != expected_quality_shape
        ):
            raise ValueError(
                "quality_features must have shape "
                f"{expected_quality_shape}"
            )
        if not points.is_floating_point():
            raise TypeError("points must use a floating-point dtype")
        if not boxes.is_floating_point():
            raise TypeError("boxes must use a floating-point dtype")
        if not quality_features.is_floating_point():
            raise TypeError(
                "quality_features must use a floating-point dtype"
            )
        if (
            points.device != boxes.device
            or points.device != quality_features.device
        ):
            raise ValueError("all refiner inputs must be on the same device")
        if points.dtype != boxes.dtype or points.dtype != quality_features.dtype:
            raise ValueError("all refiner inputs must have the same dtype")
        if not torch.isfinite(points).all():
            raise ValueError("points must be finite")
        if not torch.isfinite(boxes).all():
            raise ValueError("boxes must be finite")
        if not torch.isfinite(quality_features).all():
            raise ValueError("quality_features must be finite")
        if not torch.all(boxes[:, 3:6] > 0.0):
            raise ValueError("box dimensions must be positive")
        if not torch.all(
            (quality_features >= 0.0) & (quality_features <= 1.0)
        ):
            raise ValueError("quality_features must lie in [0, 1]")

        if point_mask is None:
            point_mask = torch.ones(
                (batch_size, point_count),
                dtype=torch.bool,
                device=points.device,
            )
        else:
            point_mask = self._require_tensor("point_mask", point_mask)
            if point_mask.ndim != 2 or point_mask.shape != (
                batch_size,
                point_count,
            ):
                raise ValueError("point_mask must have shape [B, N]")
            if point_mask.dtype is not torch.bool:
                raise TypeError("point_mask must have Boolean dtype")
            if point_mask.device != points.device:
                raise ValueError("point_mask must be on the points device")
        if not torch.all(point_mask.any(dim=1)):
            raise ValueError("every sample must contain a valid point")
        return points, boxes, quality_features, point_mask

    def forward(
        self,
        points: Any,
        boxes: Any,
        quality_features: Any,
        point_mask: Optional[Any] = None,
    ) -> Dict[str, Any]:
        points, boxes, quality_features, point_mask = self._validate_inputs(
            points, boxes, quality_features, point_mask
        )
        cfg = self.config

        normalized_xyz = (
            points[..., :3] - boxes[:, None, :3]
        ) / boxes[:, None, 3:6].clamp_min(cfg.minimum_dimension)
        normalized_xyz = normalized_xyz.clamp(
            -cfg.normalized_point_limit,
            cfg.normalized_point_limit,
        )
        if cfg.point_feature_dim == 3:
            encoded_input = normalized_xyz
        else:
            encoded_input = torch.cat(
                (normalized_xyz, points[..., 3:]), dim=-1
            )

        point_embedding = self.point_mlp(encoded_input)
        mask = point_mask.unsqueeze(-1)
        maximum = point_embedding.masked_fill(
            ~mask, torch.finfo(point_embedding.dtype).min
        ).amax(dim=1)
        mean = (point_embedding * mask).sum(dim=1) / mask.sum(
            dim=1
        ).clamp_min(1)
        fused = torch.cat((maximum, mean, quality_features), dim=-1)
        raw = self.output_layer(self.head(fused))

        center_fraction = torch.tanh(raw[:, :3]) * cfg.max_center_fraction
        center_residual = center_fraction * boxes[:, 3:6]
        log_dimension_residual = (
            torch.tanh(raw[:, 3:6])
            * cfg.max_log_dimension_residual
        )
        quality = torch.sigmoid(raw[:, 6])
        return {
            "center_residual": center_residual,
            "center_residual_fraction": center_fraction,
            "log_dimension_residual": log_dimension_residual,
            "quality": quality,
        }


def _as_box_batch(
    boxes: np.ndarray,
) -> Tuple[np.ndarray, bool, np.dtype]:
    original = np.asarray(boxes)
    original_dtype = original.dtype
    squeeze = original.ndim == 1
    if squeeze:
        original = original[None, :]
    if original.ndim != 2 or original.shape[1] < 6:
        raise ValueError("boxes must have shape [N, D>=6] or [D>=6]")
    if not np.issubdtype(original.dtype, np.number):
        raise TypeError("boxes must be numeric")
    values = np.asarray(original, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("boxes must be finite")
    if not (values[:, 3:6] > 0.0).all():
        raise ValueError("box dimensions must be positive")
    return values, squeeze, original_dtype


def _as_residual_batch(
    residual: np.ndarray,
    name: str,
    batch_size: int,
) -> np.ndarray:
    value = np.asarray(residual, dtype=np.float64)
    if value.ndim == 1:
        value = value[None, :]
    if value.shape != (batch_size, 3):
        raise ValueError(f"{name} must have shape [N, 3] or [3]")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def apply_box_residual_numpy(
    boxes: np.ndarray,
    center_residual: np.ndarray,
    log_dimension_residual: np.ndarray,
    *,
    max_center_fraction: float = 0.25,
    max_abs_log_dimension_residual: float = float(np.log(1.5)),
    minimum_dimension: float = 1e-3,
    maximum_dimension: Optional[float] = None,
) -> np.ndarray:
    """Apply bounded residuals to AABBs without mutating the inputs.

    ``boxes`` use ``[cx, cy, cz, dx, dy, dz, ...]``.  Extra columns such as
    yaw are copied unchanged.  Center residuals are clipped to a fraction of
    each original dimension; log-dimension residuals are symmetrically
    clipped before exponentiation.
    """

    box_batch, squeeze, original_dtype = _as_box_batch(boxes)
    center = _as_residual_batch(
        center_residual, "center_residual", box_batch.shape[0]
    )
    log_dimension = _as_residual_batch(
        log_dimension_residual,
        "log_dimension_residual",
        box_batch.shape[0],
    )

    scalar_parameters = {
        "max_center_fraction": max_center_fraction,
        "max_abs_log_dimension_residual": (
            max_abs_log_dimension_residual
        ),
        "minimum_dimension": minimum_dimension,
    }
    for name, value in scalar_parameters.items():
        if not np.isscalar(value) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite scalar")
        if float(value) <= 0.0:
            raise ValueError(f"{name} must be positive")
    if maximum_dimension is not None:
        if (
            not np.isscalar(maximum_dimension)
            or not np.isfinite(maximum_dimension)
            or float(maximum_dimension) <= float(minimum_dimension)
        ):
            raise ValueError(
                "maximum_dimension must be finite and greater than "
                "minimum_dimension"
            )

    result = box_batch.copy()
    original_dimensions = box_batch[:, 3:6]
    center_limit = float(max_center_fraction) * original_dimensions
    result[:, :3] += np.clip(center, -center_limit, center_limit)

    log_limit = float(max_abs_log_dimension_residual)
    dimensions = original_dimensions * np.exp(
        np.clip(log_dimension, -log_limit, log_limit)
    )
    dimensions = np.maximum(dimensions, float(minimum_dimension))
    if maximum_dimension is not None:
        dimensions = np.minimum(dimensions, float(maximum_dimension))
    result[:, 3:6] = dimensions

    output_dtype = (
        original_dtype
        if np.issubdtype(original_dtype, np.floating)
        else np.dtype(np.float32)
    )
    result = result.astype(output_dtype, copy=False)
    return result[0] if squeeze else result


def _torch_load_weights(path: Path, map_location: Any) -> Any:
    if torch is None:
        raise ImportError("loading a box-refiner checkpoint requires PyTorch")
    try:
        return torch.load(
            str(path), map_location=map_location, weights_only=True
        )
    except TypeError:  # PyTorch before ``weights_only`` was introduced.
        return torch.load(str(path), map_location=map_location)


def load_box_refiner_checkpoint(
    model: PointNetBoxRefiner,
    checkpoint_path: Union[str, Path],
    *,
    map_location: Any = "cpu",
) -> PointNetBoxRefiner:
    """Strictly load a raw or wrapped state dict into ``model``.

    Wrapped checkpoints may contain ``{"state_dict": ..., "config": ...}``.
    When configuration metadata is present it must exactly match the model.
    Missing/unexpected parameters are rejected by ``strict=True``.
    """

    if torch is None or nn is None:
        raise ImportError("loading a box-refiner checkpoint requires PyTorch")
    if not isinstance(model, PointNetBoxRefiner):
        raise TypeError("model must be a PointNetBoxRefiner")
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"box-refiner checkpoint not found: {path}")

    checkpoint = _torch_load_weights(path, map_location)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must contain a mapping")

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        metadata_config = checkpoint.get("config")
    else:
        state_dict = checkpoint
        metadata_config = None
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("checkpoint state_dict must be a non-empty mapping")
    if not all(isinstance(key, str) for key in state_dict):
        raise ValueError("checkpoint state_dict keys must be strings")
    if not all(torch.is_tensor(value) for value in state_dict.values()):
        raise ValueError("checkpoint state_dict values must be tensors")

    if metadata_config is not None:
        if not isinstance(metadata_config, Mapping):
            raise ValueError("checkpoint config must be a mapping")
        expected = model.config.architecture_dict()
        received = dict(metadata_config)
        if received != expected:
            raise ValueError(
                "checkpoint config does not match the model configuration"
            )

    try:
        model.load_state_dict(dict(state_dict), strict=True)
    except RuntimeError as error:
        raise ValueError(
            f"incompatible box-refiner checkpoint: {error}"
        ) from error
    return model


def build_box_refiner(
    *,
    enabled: bool,
    checkpoint_path: Optional[Union[str, Path]] = None,
    config: Optional[BoxRefinerConfig] = None,
    device: Any = "cpu",
) -> Optional[PointNetBoxRefiner]:
    """Build the production refiner, or return ``None`` when disabled.

    The disabled path intentionally performs no PyTorch or filesystem work and
    does not require a checkpoint.  Enabling always requires a strict
    checkpoint so random neural residuals cannot silently alter detections.
    """

    if not isinstance(enabled, (bool, np.bool_)):
        raise TypeError("enabled must be Boolean")
    if not bool(enabled):
        return None
    if torch is None or nn is None:
        raise ImportError("enabled box refinement requires PyTorch")
    if checkpoint_path is None:
        raise ValueError(
            "checkpoint_path is required when box refinement is enabled"
        )

    model = PointNetBoxRefiner(config)
    load_box_refiner_checkpoint(
        model, checkpoint_path, map_location=device
    )
    model.to(device)
    model.eval()
    return model


# Clear alias for configuration loaders that use "load" terminology.
load_box_refiner = build_box_refiner


__all__ = [
    "DEFAULT_QUALITY_FEATURE_DIM",
    "BoxRefinerConfig",
    "PointNetBoxRefiner",
    "apply_box_residual_numpy",
    "build_box_refiner",
    "load_box_refiner",
    "load_box_refiner_checkpoint",
]
