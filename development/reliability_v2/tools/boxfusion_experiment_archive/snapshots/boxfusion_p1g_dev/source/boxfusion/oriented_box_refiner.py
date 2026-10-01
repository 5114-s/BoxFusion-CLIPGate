"""Orientation-preserving, object-local neural box refinement.

This module implements the model layer for B5-v2.  It deliberately predicts
geometry only in the coordinate frame of the original oriented box:

* points and boxes are expressed in ``box_local`` coordinates;
* center residuals are bounded fractions of the original local dimensions;
* dimension residuals are bounded in log space; and
* an improvement probability lets the runtime reject unsafe refinements.

The output layer is initialized to an identity geometry transform and a low
improvement probability.  Consequently, an untrained model is safe by
construction, although production construction still requires a strictly
versioned checkpoint.

NumPy-only geometry users can import and use
:func:`apply_local_box_residual_numpy` without PyTorch being installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np

try:  # Keep the pure NumPy residual path independent from PyTorch.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised in torch-free deployments.
    torch = None
    nn = None


ORIENTED_BOX_REFINER_SCHEMA = "boxfusion.oriented_box_refiner"
ORIENTED_BOX_REFINER_FORMAT_VERSION = 1
ORIENTED_BOX_REFINER_COORDINATE_FRAME = "box_local"
ORIENTED_BOX_REFINER_QUALITY_FEATURE_DIM = 12

_CHECKPOINT_KEYS = frozenset(
    {
        "schema",
        "format_version",
        "coordinate_frame",
        "config",
        "state_dict",
    }
)


@dataclass(frozen=True)
class OrientedBoxRefinerConfig:
    """Architecture and hard output bounds for B5-v2.

    ``point_feature_dim`` and ``quality_feature_dim`` are fixed by the B5-v2
    data schema.  They remain explicit checkpoint fields so an incompatible
    training artifact fails loudly instead of being silently interpreted.
    """

    point_feature_dim: int = 3
    quality_feature_dim: int = ORIENTED_BOX_REFINER_QUALITY_FEATURE_DIM
    point_hidden_dim: int = 64
    point_embedding_dim: int = 128
    head_hidden_dim: int = 128
    max_center_fraction: float = 0.15
    max_log_dimension_residual: float = float(np.log(1.25))
    minimum_dimension: float = 1e-3
    normalized_point_limit: float = 4.0
    default_quality_probability: float = 0.01

    def validated(self) -> "OrientedBoxRefinerConfig":
        """Validate the complete architecture signature and return ``self``."""

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
        if int(self.point_feature_dim) != 3:
            raise ValueError("point_feature_dim must equal 3 for local xyz")
        if (
            int(self.quality_feature_dim)
            != ORIENTED_BOX_REFINER_QUALITY_FEATURE_DIM
        ):
            raise ValueError(
                "quality_feature_dim must equal "
                f"{ORIENTED_BOX_REFINER_QUALITY_FEATURE_DIM}"
            )

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

        probability = self.default_quality_probability
        if not np.isscalar(probability) or not np.isfinite(probability):
            raise ValueError(
                "default_quality_probability must be a finite scalar"
            )
        if not 0.0 < float(probability) < 0.5:
            raise ValueError(
                "default_quality_probability must lie strictly in (0, 0.5)"
            )
        return self

    def architecture_dict(self) -> Dict[str, Union[int, float]]:
        """Return the exact configuration persisted in B5-v2 checkpoints."""

        return asdict(self)


_ModuleBase = nn.Module if nn is not None else object


class PointNetOrientedBoxRefiner(_ModuleBase):
    """A lightweight max+mean PointNet for local oriented-box refinement.

    Args:
        config: Architecture, normalization limits, and residual bounds.

    Inputs to :meth:`forward`:
        points_local: Float tensor ``[B, N, 3]`` in the original box frame.
        local_boxes: Float tensor ``[B, 6]`` as
            ``[cx, cy, cz, dx, dy, dz]`` in that same frame.
        quality_features: Float tensor ``[B, 12]`` with finite values in
            ``[0, 1]``.
        point_mask: Optional Boolean tensor ``[B, N]``.  Every sample must
            contain at least one valid point.

    Returns:
        A mapping with bounded metric ``center_residual``, bounded
        ``center_residual_fraction``, bounded ``log_dimension_residual``, and
        the predicted refinement-improvement probability ``quality``.
    """

    def __init__(
        self, config: Optional[OrientedBoxRefinerConfig] = None
    ) -> None:
        if nn is None or torch is None:
            raise ImportError(
                "PointNetOrientedBoxRefiner requires PyTorch; the NumPy "
                "local-box residual utility remains available without it"
            )
        super().__init__()
        self.config = (config or OrientedBoxRefinerConfig()).validated()
        cfg = self.config

        self.point_mlp = nn.Sequential(
            nn.Linear(cfg.point_feature_dim, cfg.point_hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(cfg.point_hidden_dim, cfg.point_embedding_dim),
            nn.ReLU(inplace=False),
        )
        self.head = nn.Sequential(
            nn.Linear(
                2 * cfg.point_embedding_dim + cfg.quality_feature_dim,
                cfg.head_hidden_dim,
            ),
            nn.ReLU(inplace=False),
            nn.Linear(cfg.head_hidden_dim, cfg.head_hidden_dim),
            nn.ReLU(inplace=False),
        )
        self.output_layer = nn.Linear(cfg.head_hidden_dim, 7)
        self._initialize_safe_identity()

    def _initialize_safe_identity(self) -> None:
        """Initialize zero geometry residuals and a default reject decision."""

        cfg = self.config
        probability = float(cfg.default_quality_probability)
        quality_logit = float(np.log(probability / (1.0 - probability)))
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        with torch.no_grad():
            self.output_layer.bias[6] = quality_logit

    @staticmethod
    def _require_tensor(name: str, value: Any) -> Any:
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a torch.Tensor")
        return value

    def _validate_inputs(
        self,
        points_local: Any,
        local_boxes: Any,
        quality_features: Any,
        point_mask: Optional[Any],
    ) -> Tuple[Any, Any, Any, Any]:
        points_local = self._require_tensor(
            "points_local", points_local
        )
        local_boxes = self._require_tensor("local_boxes", local_boxes)
        quality_features = self._require_tensor(
            "quality_features", quality_features
        )

        if points_local.ndim != 3:
            raise ValueError("points_local must have shape [B, N, 3]")
        batch_size, point_count, feature_dim = points_local.shape
        if point_count < 1:
            raise ValueError(
                "points_local must contain at least one point"
            )
        if feature_dim != self.config.point_feature_dim:
            raise ValueError(
                "points_local feature dimension must equal "
                f"{self.config.point_feature_dim}, received {feature_dim}"
            )
        if local_boxes.ndim != 2 or local_boxes.shape != (batch_size, 6):
            raise ValueError("local_boxes must have shape [B, 6]")
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

        float_inputs = (
            ("points_local", points_local),
            ("local_boxes", local_boxes),
            ("quality_features", quality_features),
        )
        for name, value in float_inputs:
            if not value.is_floating_point():
                raise TypeError(f"{name} must use a floating-point dtype")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
        if (
            points_local.device != local_boxes.device
            or points_local.device != quality_features.device
        ):
            raise ValueError("all refiner inputs must be on the same device")
        if (
            points_local.dtype != local_boxes.dtype
            or points_local.dtype != quality_features.dtype
        ):
            raise ValueError("all refiner inputs must have the same dtype")
        if not torch.all(local_boxes[:, 3:6] > 0.0):
            raise ValueError("local box dimensions must be positive")
        if not torch.all(
            (quality_features >= 0.0) & (quality_features <= 1.0)
        ):
            raise ValueError("quality_features must lie in [0, 1]")

        if point_mask is None:
            point_mask = torch.ones(
                (batch_size, point_count),
                dtype=torch.bool,
                device=points_local.device,
            )
        else:
            point_mask = self._require_tensor("point_mask", point_mask)
            if point_mask.ndim != 2 or point_mask.shape != (
                batch_size,
                point_count,
            ):
                raise ValueError("point_mask must have shape [B, N]")
            if point_mask.dtype != torch.bool:
                raise TypeError("point_mask must have Boolean dtype")
            if point_mask.device != points_local.device:
                raise ValueError(
                    "point_mask must be on the points_local device"
                )
        if not torch.all(point_mask.any(dim=1)):
            raise ValueError("every sample must contain a valid point")
        return points_local, local_boxes, quality_features, point_mask

    def forward(
        self,
        points_local: Any,
        local_boxes: Any,
        quality_features: Any,
        point_mask: Optional[Any] = None,
    ) -> Dict[str, Any]:
        (
            points_local,
            local_boxes,
            quality_features,
            point_mask,
        ) = self._validate_inputs(
            points_local, local_boxes, quality_features, point_mask
        )
        cfg = self.config

        dimensions = local_boxes[:, None, 3:6].clamp_min(
            cfg.minimum_dimension
        )
        normalized_points = (
            points_local - local_boxes[:, None, :3]
        ) / dimensions
        normalized_points = normalized_points.clamp(
            -cfg.normalized_point_limit,
            cfg.normalized_point_limit,
        )

        point_embedding = self.point_mlp(normalized_points)
        mask = point_mask.unsqueeze(-1)
        maximum = point_embedding.masked_fill(
            ~mask, torch.finfo(point_embedding.dtype).min
        ).amax(dim=1)
        mean = (point_embedding * mask).sum(dim=1) / mask.sum(
            dim=1
        ).clamp_min(1)
        fused = torch.cat((maximum, mean, quality_features), dim=-1)
        raw = self.output_layer(self.head(fused))

        center_fraction = (
            torch.tanh(raw[:, :3]) * cfg.max_center_fraction
        )
        center_residual = center_fraction * local_boxes[:, 3:6]
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


def _as_local_box_batch(
    local_boxes: np.ndarray,
) -> Tuple[np.ndarray, bool, np.dtype]:
    original = np.asarray(local_boxes)
    original_dtype = original.dtype
    squeeze = original.ndim == 1
    if squeeze:
        original = original[None, :]
    if original.ndim != 2 or original.shape[1] < 6:
        raise ValueError(
            "local_boxes must have shape [B, D>=6] or [D>=6]"
        )
    if not np.issubdtype(original.dtype, np.number):
        raise TypeError("local_boxes must be numeric")
    values = np.asarray(original, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("local_boxes must be finite")
    if not (values[:, 3:6] > 0.0).all():
        raise ValueError("local box dimensions must be positive")
    return values, squeeze, original_dtype


def _as_local_residual_batch(
    residual: np.ndarray,
    name: str,
    batch_size: int,
) -> np.ndarray:
    values = np.asarray(residual)
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape != (batch_size, 3):
        raise ValueError(f"{name} must have shape [B, 3] or [3]")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    return values


def _validated_positive_scalar(name: str, value: Any) -> float:
    if not np.isscalar(value) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def apply_local_box_residual_numpy(
    local_boxes: np.ndarray,
    center_residual: np.ndarray,
    log_dimension_residual: np.ndarray,
    *,
    max_center_fraction: float = 0.15,
    max_abs_log_dimension_residual: float = float(np.log(1.25)),
    minimum_dimension: float = 1e-3,
    maximum_dimension: Optional[float] = None,
) -> np.ndarray:
    """Apply bounded residuals to boxes in the original box-local frame.

    Args:
        local_boxes: ``[B, D>=6]`` or ``[D>=6]`` values ordered as
            ``[cx, cy, cz, dx, dy, dz, ...]``.  Extra metadata columns are
            copied unchanged.
        center_residual: Metric local-coordinate offsets ``[B, 3]``.
        log_dimension_residual: Local log-scale offsets ``[B, 3]``.
        max_center_fraction: Per-axis absolute center bound relative to the
            corresponding original local dimension.
        max_abs_log_dimension_residual: Symmetric log-scale bound.
        minimum_dimension: Hard positive output extent floor.
        maximum_dimension: Optional hard output extent ceiling.

    The function is side-effect free and contains no PyTorch dependency.
    """

    box_batch, squeeze, original_dtype = _as_local_box_batch(local_boxes)
    center = _as_local_residual_batch(
        center_residual, "center_residual", box_batch.shape[0]
    )
    log_dimension = _as_local_residual_batch(
        log_dimension_residual,
        "log_dimension_residual",
        box_batch.shape[0],
    )

    center_fraction = _validated_positive_scalar(
        "max_center_fraction", max_center_fraction
    )
    log_limit = _validated_positive_scalar(
        "max_abs_log_dimension_residual",
        max_abs_log_dimension_residual,
    )
    minimum = _validated_positive_scalar(
        "minimum_dimension", minimum_dimension
    )
    maximum = None
    if maximum_dimension is not None:
        maximum = _validated_positive_scalar(
            "maximum_dimension", maximum_dimension
        )
        if maximum <= minimum:
            raise ValueError(
                "maximum_dimension must be greater than minimum_dimension"
            )

    result = box_batch.copy()
    original_dimensions = box_batch[:, 3:6]
    center_limit = center_fraction * original_dimensions
    result[:, :3] += np.clip(center, -center_limit, center_limit)
    dimensions = original_dimensions * np.exp(
        np.clip(log_dimension, -log_limit, log_limit)
    )
    dimensions = np.maximum(dimensions, minimum)
    if maximum is not None:
        dimensions = np.minimum(dimensions, maximum)
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
        raise ImportError(
            "loading an oriented box-refiner checkpoint requires PyTorch"
        )
    try:
        return torch.load(
            str(path), map_location=map_location, weights_only=True
        )
    except TypeError:  # PyTorch before ``weights_only`` was introduced.
        return torch.load(str(path), map_location=map_location)


def make_oriented_box_refiner_checkpoint(
    model: PointNetOrientedBoxRefiner,
) -> Dict[str, Any]:
    """Build the exact, strictly versioned B5-v2 checkpoint payload."""

    if torch is None or nn is None:
        raise ImportError(
            "creating an oriented box-refiner checkpoint requires PyTorch"
        )
    if not isinstance(model, PointNetOrientedBoxRefiner):
        raise TypeError("model must be a PointNetOrientedBoxRefiner")
    return {
        "schema": ORIENTED_BOX_REFINER_SCHEMA,
        "format_version": ORIENTED_BOX_REFINER_FORMAT_VERSION,
        "coordinate_frame": ORIENTED_BOX_REFINER_COORDINATE_FRAME,
        "config": model.config.architecture_dict(),
        "state_dict": model.state_dict(),
    }


def load_oriented_box_refiner_checkpoint(
    model: PointNetOrientedBoxRefiner,
    checkpoint_path: Union[str, Path],
    *,
    map_location: Any = "cpu",
) -> PointNetOrientedBoxRefiner:
    """Strictly validate and load one B5-v2 ``box_local`` checkpoint."""

    if torch is None or nn is None:
        raise ImportError(
            "loading an oriented box-refiner checkpoint requires PyTorch"
        )
    if not isinstance(model, PointNetOrientedBoxRefiner):
        raise TypeError("model must be a PointNetOrientedBoxRefiner")
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"oriented box-refiner checkpoint not found: {path}"
        )

    checkpoint = _torch_load_weights(path, map_location)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    received_keys = set(checkpoint.keys())
    if received_keys != _CHECKPOINT_KEYS:
        missing = sorted(_CHECKPOINT_KEYS - received_keys)
        extra = sorted(received_keys - _CHECKPOINT_KEYS)
        raise ValueError(
            "oriented box-refiner checkpoint keys do not match the strict "
            f"schema (missing={missing}, extra={extra})"
        )
    if checkpoint["schema"] != ORIENTED_BOX_REFINER_SCHEMA:
        raise ValueError("unsupported oriented box-refiner checkpoint schema")
    version = checkpoint["format_version"]
    if (
        isinstance(version, bool)
        or not isinstance(version, (int, np.integer))
        or int(version) != ORIENTED_BOX_REFINER_FORMAT_VERSION
    ):
        raise ValueError(
            "unsupported oriented box-refiner checkpoint format_version"
        )
    if (
        checkpoint["coordinate_frame"]
        != ORIENTED_BOX_REFINER_COORDINATE_FRAME
    ):
        raise ValueError(
            "oriented box-refiner checkpoint coordinate_frame must be "
            f"'{ORIENTED_BOX_REFINER_COORDINATE_FRAME}'"
        )

    metadata_config = checkpoint["config"]
    if not isinstance(metadata_config, Mapping):
        raise ValueError("checkpoint config must be a mapping")
    if dict(metadata_config) != model.config.architecture_dict():
        raise ValueError(
            "checkpoint config does not match the model configuration"
        )

    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("checkpoint state_dict must be a non-empty mapping")
    if not all(isinstance(key, str) for key in state_dict):
        raise ValueError("checkpoint state_dict keys must be strings")
    if not all(torch.is_tensor(value) for value in state_dict.values()):
        raise ValueError("checkpoint state_dict values must be tensors")
    try:
        model.load_state_dict(dict(state_dict), strict=True)
    except RuntimeError as error:
        raise ValueError(
            f"incompatible oriented box-refiner checkpoint: {error}"
        ) from error
    return model


def build_oriented_box_refiner(
    *,
    enabled: bool,
    checkpoint_path: Optional[Union[str, Path]] = None,
    config: Optional[OrientedBoxRefinerConfig] = None,
    device: Any = "cpu",
) -> Optional[PointNetOrientedBoxRefiner]:
    """Build a production B5-v2 model or return ``None`` when disabled.

    The disabled route deliberately performs no filesystem or PyTorch work.
    Enabling always requires a strict checkpoint, preventing an untrained
    identity/default-reject model from being mistaken for a trained refiner.
    """

    if not isinstance(enabled, (bool, np.bool_)):
        raise TypeError("enabled must be Boolean")
    if not bool(enabled):
        return None
    if torch is None or nn is None:
        raise ImportError("enabled oriented box refinement requires PyTorch")
    if checkpoint_path is None:
        raise ValueError(
            "checkpoint_path is required when oriented box refinement is "
            "enabled"
        )

    model = PointNetOrientedBoxRefiner(config)
    load_oriented_box_refiner_checkpoint(
        model, checkpoint_path, map_location=device
    )
    model.to(device)
    model.eval()
    return model


load_oriented_box_refiner = build_oriented_box_refiner


__all__ = [
    "ORIENTED_BOX_REFINER_SCHEMA",
    "ORIENTED_BOX_REFINER_FORMAT_VERSION",
    "ORIENTED_BOX_REFINER_COORDINATE_FRAME",
    "ORIENTED_BOX_REFINER_QUALITY_FEATURE_DIM",
    "OrientedBoxRefinerConfig",
    "PointNetOrientedBoxRefiner",
    "apply_local_box_residual_numpy",
    "make_oriented_box_refiner_checkpoint",
    "load_oriented_box_refiner_checkpoint",
    "build_oriented_box_refiner",
    "load_oriented_box_refiner",
]
