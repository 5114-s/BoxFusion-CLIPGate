"""Reliability-aware multi-view local geometry and quality prediction.

This module is the shared B3 -> B5 + B6-v2 head.  It keeps the selected
Mask-RGBD views separate instead of flattening them into one point cloud:

* a small PointNet encodes each view in the original BoxFusion OBB frame;
* learned view attention fuses the Top-K evidence with fixed reliability
  attributes;
* one geometry branch predicts bounded centre/size residuals and the
  probability that applying them improves evaluator IoU; and
* two quality branches predict IoU/Q15/Q25/Q50 for the original box and the
  proposed candidate respectively.

The runtime chooses the quality branch matching the geometry it actually
exports.  A rejected candidate can therefore never receive the candidate's
quality score.  The geometry output is initialized to identity and the
improvement probability to a conservative reject value.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from boxfusion.object_memory import deterministic_bounded_sample
from boxfusion.quality_score import (
    DEFAULT_IOU_AWARE_RANKING_WEIGHTS,
    IOU_AWARE_OUTPUT_NAMES,
    IOU_AWARE_THRESHOLDS,
    QUALITY_FEATURE_DIM,
)

try:  # Pure NumPy input preparation remains usable without PyTorch.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised in torch-free deployments.
    torch = None
    nn = None


JOINT_LOCAL_HEAD_SCHEMA = "boxfusion.joint_local_head"
JOINT_LOCAL_HEAD_FORMAT_VERSION = 1
JOINT_LOCAL_HEAD_COORDINATE_FRAME = "box_local"
JOINT_LOCAL_HEAD_INPUT_SCHEMA = "topk_mask_rgbd_local_v1"
JOINT_LOCAL_HEAD_OUTPUT_SCHEMA = "dual_aligned_iou_geometry_v1"

JOINT_VIEW_FEATURE_NAMES = (
    "view_quality",
    "confidence",
    "valid_depth_ratio",
    "projection_iou",
    "point_count_ratio",
    "camera_valid",
    "direction_local_x",
    "direction_local_y",
    "direction_local_z",
)
JOINT_VIEW_FEATURE_DIM = len(JOINT_VIEW_FEATURE_NAMES)
JOINT_QUALITY_BRANCH_NAMES = ("original", "candidate")
JOINT_QUALITY_COMPONENT_NAMES = IOU_AWARE_OUTPUT_NAMES

_CHECKPOINT_KEYS = frozenset(
    {
        "schema",
        "format_version",
        "coordinate_frame",
        "input_schema",
        "output_schema",
        "config",
        "state_dict",
        "metadata",
    }
)


def _logit(probability: float) -> float:
    value = float(probability)
    return float(np.log(value / (1.0 - value)))


def _finite_positive(name: str, value: Any) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not np.isscalar(value)
        or not np.isfinite(value)
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


@dataclass(frozen=True)
class JointLocalHeadConfig:
    """Architecture and hard output bounds for the joint local head."""

    point_feature_dim: int = 3
    view_feature_dim: int = JOINT_VIEW_FEATURE_DIM
    quality_feature_dim: int = QUALITY_FEATURE_DIM
    point_hidden_dim: int = 48
    point_embedding_dim: int = 96
    view_embedding_dim: int = 96
    head_hidden_dim: int = 128
    max_center_fraction: float = 0.15
    max_log_dimension_residual: float = float(np.log(1.25))
    minimum_dimension: float = 1e-3
    normalized_point_limit: float = 4.0
    default_improvement_probability: float = 0.01
    default_iou_probability: float = 0.10
    ranking_weights: Tuple[float, float, float, float] = tuple(
        float(value) for value in DEFAULT_IOU_AWARE_RANKING_WEIGHTS
    )
    minimum_log_variance: float = -6.0
    maximum_log_variance: float = 2.0

    def validated(self) -> "JointLocalHeadConfig":
        integer_fields = (
            "point_feature_dim",
            "view_feature_dim",
            "quality_feature_dim",
            "point_hidden_dim",
            "point_embedding_dim",
            "view_embedding_dim",
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
            raise ValueError("point_feature_dim must equal 3")
        if int(self.view_feature_dim) != JOINT_VIEW_FEATURE_DIM:
            raise ValueError(
                f"view_feature_dim must equal {JOINT_VIEW_FEATURE_DIM}"
            )
        if int(self.quality_feature_dim) != QUALITY_FEATURE_DIM:
            raise ValueError(
                f"quality_feature_dim must equal {QUALITY_FEATURE_DIM}"
            )
        for name in (
            "max_center_fraction",
            "max_log_dimension_residual",
            "minimum_dimension",
            "normalized_point_limit",
        ):
            _finite_positive(name, getattr(self, name))
        for name in (
            "default_improvement_probability",
            "default_iou_probability",
        ):
            value = getattr(self, name)
            if (
                not np.isscalar(value)
                or not np.isfinite(value)
                or not 0.0 < float(value) < 0.5
            ):
                raise ValueError(f"{name} must lie strictly in (0, 0.5)")
        weights = np.asarray(self.ranking_weights, dtype=np.float64)
        if (
            weights.shape != (4,)
            or not np.isfinite(weights).all()
            or (weights < 0.0).any()
            or not np.isclose(weights.sum(), 1.0, atol=1e-8, rtol=0.0)
        ):
            raise ValueError(
                "ranking_weights must be four non-negative values summing to 1"
            )
        for name in ("minimum_log_variance", "maximum_log_variance"):
            value = getattr(self, name)
            if not np.isscalar(value) or not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if float(self.minimum_log_variance) >= float(
            self.maximum_log_variance
        ):
            raise ValueError(
                "minimum_log_variance must be below maximum_log_variance"
            )
        return self

    def architecture_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["ranking_weights"] = tuple(
            float(value) for value in self.ranking_weights
        )
        return result


@dataclass(frozen=True)
class JointViewInputs:
    """Exact per-instance NumPy view tensors shared by runtime/diagnostics."""

    points_local: np.ndarray
    point_mask: np.ndarray
    view_features: np.ndarray
    view_mask: np.ndarray


def _validated_frame(
    frame_center: Any, frame_basis: Any
) -> Tuple[np.ndarray, np.ndarray]:
    center = np.asarray(frame_center, dtype=np.float64)
    basis = np.asarray(frame_basis, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("frame_center must have finite shape [3]")
    if basis.shape != (3, 3) or not np.isfinite(basis).all():
        raise ValueError("frame_basis must have finite shape [3,3]")
    if not np.allclose(
        basis.T @ basis, np.eye(3), atol=2e-3, rtol=0.0
    ):
        raise ValueError("frame_basis must be orthonormal")
    if float(np.linalg.det(basis)) <= 0.0:
        raise ValueError("frame_basis must be right-handed")
    return center, basis


def prepare_joint_view_inputs(
    records: Sequence[Any],
    *,
    frame_center: Any,
    frame_basis: Any,
    max_views: int,
    points_per_view: int,
) -> JointViewInputs:
    """Convert selected ``MemoryViewRecord``-like objects to joint tensors.

    The function intentionally relies only on the records' public attributes,
    allowing offline diagnostic builders to reuse this exact implementation.
    Invalid/padded slots are zero and masked.  Direction values are mapped
    from ``[-1, 1]`` to ``[0, 1]`` and use a separate camera-valid bit.
    """

    if isinstance(max_views, bool) or not isinstance(
        max_views, (int, np.integer)
    ):
        raise TypeError("max_views must be an integer")
    if isinstance(points_per_view, bool) or not isinstance(
        points_per_view, (int, np.integer)
    ):
        raise TypeError("points_per_view must be an integer")
    max_views = int(max_views)
    points_per_view = int(points_per_view)
    if max_views <= 0 or points_per_view <= 0:
        raise ValueError("max_views and points_per_view must be positive")
    center, basis = _validated_frame(frame_center, frame_basis)

    points_local = np.zeros(
        (max_views, points_per_view, 3), dtype=np.float32
    )
    point_mask = np.zeros((max_views, points_per_view), dtype=bool)
    view_features = np.zeros(
        (max_views, JOINT_VIEW_FEATURE_DIM), dtype=np.float32
    )
    view_mask = np.zeros(max_views, dtype=bool)

    for view_index, record in enumerate(tuple(records)[:max_views]):
        points_world = np.asarray(record.points_world, dtype=np.float64)
        if (
            points_world.ndim != 2
            or points_world.shape[1] != 3
            or not np.isfinite(points_world).all()
            or len(points_world) == 0
        ):
            continue
        sampled_world = deterministic_bounded_sample(
            points_world, points_per_view
        ).astype(np.float64, copy=False)
        sampled_local = (sampled_world - center[None, :]) @ basis
        count = len(sampled_local)
        points_local[view_index, :count] = sampled_local.astype(np.float32)
        point_mask[view_index, :count] = True
        view_mask[view_index] = True

        camera_position = getattr(record, "camera_position", None)
        camera_valid = camera_position is not None
        direction_features = np.full(3, 0.5, dtype=np.float64)
        if camera_valid:
            camera = np.asarray(camera_position, dtype=np.float64)
            if camera.shape != (3,) or not np.isfinite(camera).all():
                raise ValueError(
                    "record.camera_position must have finite shape [3]"
                )
            direction_world = center - camera
            norm = float(np.linalg.norm(direction_world))
            if norm > 1e-8:
                direction_local = (direction_world / norm) @ basis
                direction_features = 0.5 * (
                    np.clip(direction_local, -1.0, 1.0) + 1.0
                )
            else:
                camera_valid = False

        scalar_features = (
            float(record.quality),
            float(record.confidence),
            float(record.valid_depth_ratio),
            float(record.projection_mask_iou),
            float(count) / float(points_per_view),
            float(camera_valid),
        )
        features = np.concatenate(
            (
                np.asarray(scalar_features, dtype=np.float64),
                direction_features,
            )
        )
        if (
            features.shape != (JOINT_VIEW_FEATURE_DIM,)
            or not np.isfinite(features).all()
            or (features < 0.0).any()
            or (features > 1.0).any()
        ):
            raise ValueError("joint view features must lie in [0,1]")
        view_features[view_index] = features.astype(np.float32)

    return JointViewInputs(
        points_local=points_local,
        point_mask=point_mask,
        view_features=view_features,
        view_mask=view_mask,
    )


_ModuleBase = nn.Module if nn is not None else object


class MultiViewJointLocalHead(_ModuleBase):
    """Small batched per-view PointNet with geometry and dual quality heads."""

    def __init__(
        self, config: Optional[JointLocalHeadConfig] = None
    ) -> None:
        if nn is None or torch is None:
            raise ImportError("MultiViewJointLocalHead requires PyTorch")
        super().__init__()
        self.config = (config or JointLocalHeadConfig()).validated()
        cfg = self.config
        self.point_mlp = nn.Sequential(
            nn.Linear(cfg.point_feature_dim, cfg.point_hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(cfg.point_hidden_dim, cfg.point_embedding_dim),
            nn.ReLU(inplace=False),
        )
        self.view_mlp = nn.Sequential(
            nn.Linear(
                2 * cfg.point_embedding_dim + cfg.view_feature_dim,
                cfg.view_embedding_dim,
            ),
            nn.ReLU(inplace=False),
            nn.Linear(cfg.view_embedding_dim, cfg.view_embedding_dim),
            nn.ReLU(inplace=False),
        )
        self.view_attention = nn.Sequential(
            nn.Linear(
                cfg.view_embedding_dim + cfg.view_feature_dim,
                max(cfg.view_embedding_dim // 2, 16),
            ),
            nn.ReLU(inplace=False),
            nn.Linear(max(cfg.view_embedding_dim // 2, 16), 1),
        )
        self.shared_head = nn.Sequential(
            nn.Linear(
                2 * cfg.view_embedding_dim + cfg.quality_feature_dim,
                cfg.head_hidden_dim,
            ),
            nn.ReLU(inplace=False),
            nn.Linear(cfg.head_hidden_dim, cfg.head_hidden_dim),
            nn.ReLU(inplace=False),
        )
        self.geometry_layer = nn.Linear(cfg.head_hidden_dim, 6)
        self.improvement_layer = nn.Linear(cfg.head_hidden_dim, 1)
        self.quality_layer = nn.Linear(cfg.head_hidden_dim, 8)
        self.log_variance_layer = nn.Linear(cfg.head_hidden_dim, 2)
        self._initialize_safe_identity()

    def _initialize_safe_identity(self) -> None:
        cfg = self.config
        for layer in (
            self.geometry_layer,
            self.improvement_layer,
            self.quality_layer,
            self.log_variance_layer,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        with torch.no_grad():
            self.improvement_layer.bias[0] = _logit(
                cfg.default_improvement_probability
            )
            # [IoU, Q15, P(Q25|Q15), P(Q50|Q25)] for each branch.
            quality_bias = torch.tensor(
                [
                    _logit(cfg.default_iou_probability),
                    _logit(min(2.0 * cfg.default_iou_probability, 0.49)),
                    0.0,
                    0.0,
                ]
                * 2,
                dtype=self.quality_layer.bias.dtype,
                device=self.quality_layer.bias.device,
            )
            self.quality_layer.bias.copy_(quality_bias)

    @staticmethod
    def _require_tensor(name: str, value: Any) -> Any:
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a torch.Tensor")
        return value

    def _validate_inputs(
        self,
        points_local: Any,
        point_mask: Any,
        view_features: Any,
        view_mask: Any,
        local_boxes: Any,
        quality_features: Any,
    ) -> Tuple[Any, Any, Any, Any, Any, Any]:
        points_local = self._require_tensor("points_local", points_local)
        point_mask = self._require_tensor("point_mask", point_mask)
        view_features = self._require_tensor(
            "view_features", view_features
        )
        view_mask = self._require_tensor("view_mask", view_mask)
        local_boxes = self._require_tensor("local_boxes", local_boxes)
        quality_features = self._require_tensor(
            "quality_features", quality_features
        )
        if points_local.ndim != 4:
            raise ValueError(
                "points_local must have shape [B,V,P,3]"
            )
        batch, views, points, features = points_local.shape
        if batch < 1 or views < 1 or points < 1:
            raise ValueError("joint input batch/view/point sizes must be positive")
        if features != self.config.point_feature_dim:
            raise ValueError("points_local feature dimension is invalid")
        expected_shapes = {
            "point_mask": (point_mask, (batch, views, points)),
            "view_features": (
                view_features,
                (batch, views, self.config.view_feature_dim),
            ),
            "view_mask": (view_mask, (batch, views)),
            "local_boxes": (local_boxes, (batch, 6)),
            "quality_features": (
                quality_features,
                (batch, self.config.quality_feature_dim),
            ),
        }
        for name, (value, shape) in expected_shapes.items():
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if point_mask.dtype != torch.bool or view_mask.dtype != torch.bool:
            raise TypeError("point_mask and view_mask must be Boolean")
        float_values = (
            ("points_local", points_local),
            ("view_features", view_features),
            ("local_boxes", local_boxes),
            ("quality_features", quality_features),
        )
        for name, value in float_values:
            if not value.is_floating_point():
                raise TypeError(f"{name} must use floating-point dtype")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
            if value.device != points_local.device:
                raise ValueError("all joint inputs must use one device")
            if value.dtype != points_local.dtype:
                raise ValueError("all floating joint inputs must use one dtype")
        if point_mask.device != points_local.device or view_mask.device != (
            points_local.device
        ):
            raise ValueError("all joint masks must use the input device")
        if not torch.all(local_boxes[:, 3:6] > 0.0):
            raise ValueError("local box dimensions must be positive")
        if not torch.all(
            (view_features >= 0.0) & (view_features <= 1.0)
        ):
            raise ValueError("view_features must lie in [0,1]")
        if not torch.all(
            (quality_features >= 0.0) & (quality_features <= 1.0)
        ):
            raise ValueError("quality_features must lie in [0,1]")
        if not torch.all(view_mask.any(dim=1)):
            raise ValueError("every sample must contain a valid view")
        point_views = point_mask.any(dim=2)
        if not torch.equal(point_views, view_mask):
            raise ValueError(
                "view_mask must exactly equal point_mask.any(dim=2)"
            )
        return (
            points_local,
            point_mask,
            view_features,
            view_mask,
            local_boxes,
            quality_features,
        )

    def forward(
        self,
        points_local: Any,
        point_mask: Any,
        view_features: Any,
        view_mask: Any,
        local_boxes: Any,
        quality_features: Any,
    ) -> Dict[str, Any]:
        (
            points_local,
            point_mask,
            view_features,
            view_mask,
            local_boxes,
            quality_features,
        ) = self._validate_inputs(
            points_local,
            point_mask,
            view_features,
            view_mask,
            local_boxes,
            quality_features,
        )
        cfg = self.config
        dimensions = local_boxes[:, None, None, 3:6].clamp_min(
            cfg.minimum_dimension
        )
        normalized = (
            points_local - local_boxes[:, None, None, :3]
        ) / dimensions
        normalized = normalized.clamp(
            -cfg.normalized_point_limit, cfg.normalized_point_limit
        )
        point_embeddings = self.point_mlp(normalized)
        mask = point_mask.unsqueeze(-1)
        maximum = point_embeddings.masked_fill(
            ~mask, torch.finfo(point_embeddings.dtype).min
        ).amax(dim=2)
        maximum = torch.where(
            view_mask.unsqueeze(-1), maximum, torch.zeros_like(maximum)
        )
        mean = (point_embeddings * mask).sum(dim=2) / mask.sum(
            dim=2
        ).clamp_min(1)
        view_embeddings = self.view_mlp(
            torch.cat((maximum, mean, view_features), dim=-1)
        )
        attention_logits = self.view_attention(
            torch.cat((view_embeddings, view_features), dim=-1)
        ).squeeze(-1)
        attention_logits = attention_logits.masked_fill(
            ~view_mask, torch.finfo(attention_logits.dtype).min
        )
        attention = torch.softmax(attention_logits, dim=1)
        weighted = (view_embeddings * attention.unsqueeze(-1)).sum(dim=1)
        view_maximum = view_embeddings.masked_fill(
            ~view_mask.unsqueeze(-1),
            torch.finfo(view_embeddings.dtype).min,
        ).amax(dim=1)
        shared = self.shared_head(
            torch.cat((weighted, view_maximum, quality_features), dim=-1)
        )

        geometry_raw = self.geometry_layer(shared)
        center_fraction = (
            torch.tanh(geometry_raw[:, :3]) * cfg.max_center_fraction
        )
        center_residual = center_fraction * local_boxes[:, 3:6]
        log_dimension_residual = (
            torch.tanh(geometry_raw[:, 3:6])
            * cfg.max_log_dimension_residual
        )
        improvement = torch.sigmoid(
            self.improvement_layer(shared).squeeze(-1)
        )

        quality_raw = self.quality_layer(shared).reshape(-1, 2, 4)
        predicted_iou = torch.sigmoid(quality_raw[..., 0])
        q15 = torch.sigmoid(quality_raw[..., 1])
        q25 = q15 * torch.sigmoid(quality_raw[..., 2])
        q50 = q25 * torch.sigmoid(quality_raw[..., 3])
        quality_components = torch.stack(
            (predicted_iou, q15, q25, q50), dim=-1
        )
        ranking_weights = torch.as_tensor(
            cfg.ranking_weights,
            device=quality_components.device,
            dtype=quality_components.dtype,
        )
        ranking_scores = (
            quality_components * ranking_weights[None, None, :]
        ).sum(dim=-1)
        log_variance = self.log_variance_layer(shared).clamp(
            cfg.minimum_log_variance, cfg.maximum_log_variance
        )
        uncertainty = torch.exp(0.5 * log_variance)
        return {
            "center_residual": center_residual,
            "center_residual_fraction": center_fraction,
            "log_dimension_residual": log_dimension_residual,
            "improvement_probability": improvement,
            "quality_components": quality_components,
            "ranking_scores": ranking_scores,
            "quality_log_variance": log_variance,
            "quality_uncertainty": uncertainty,
            "view_attention": attention,
        }


def make_joint_local_head_checkpoint(
    model: MultiViewJointLocalHead,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if torch is None or nn is None:
        raise ImportError("creating a joint checkpoint requires PyTorch")
    if not isinstance(model, MultiViewJointLocalHead):
        raise TypeError("model must be MultiViewJointLocalHead")
    return {
        "schema": JOINT_LOCAL_HEAD_SCHEMA,
        "format_version": JOINT_LOCAL_HEAD_FORMAT_VERSION,
        "coordinate_frame": JOINT_LOCAL_HEAD_COORDINATE_FRAME,
        "input_schema": JOINT_LOCAL_HEAD_INPUT_SCHEMA,
        "output_schema": JOINT_LOCAL_HEAD_OUTPUT_SCHEMA,
        "config": model.config.architecture_dict(),
        "state_dict": model.state_dict(),
        "metadata": dict(metadata or {}),
    }


def _torch_load(path: Path, map_location: Any) -> Any:
    if torch is None:
        raise ImportError("loading a joint checkpoint requires PyTorch")
    try:
        return torch.load(
            path, map_location=map_location, weights_only=True
        )
    except TypeError:  # PyTorch before weights_only support.
        return torch.load(path, map_location=map_location)


def load_joint_local_head_checkpoint(
    model: MultiViewJointLocalHead,
    checkpoint_path: Union[str, Path],
    *,
    map_location: Any = "cpu",
) -> Mapping[str, Any]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"joint checkpoint not found: {path}")
    checkpoint = _torch_load(path, map_location)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("joint checkpoint must contain a mapping")
    if set(checkpoint) != set(_CHECKPOINT_KEYS):
        raise ValueError("joint checkpoint keys do not match strict schema")
    expected_scalars = {
        "schema": JOINT_LOCAL_HEAD_SCHEMA,
        "format_version": JOINT_LOCAL_HEAD_FORMAT_VERSION,
        "coordinate_frame": JOINT_LOCAL_HEAD_COORDINATE_FRAME,
        "input_schema": JOINT_LOCAL_HEAD_INPUT_SCHEMA,
        "output_schema": JOINT_LOCAL_HEAD_OUTPUT_SCHEMA,
    }
    for name, expected in expected_scalars.items():
        if checkpoint[name] != expected:
            raise ValueError(f"unsupported joint checkpoint {name}")
    if not isinstance(checkpoint["config"], Mapping):
        raise TypeError("joint checkpoint config must be a mapping")
    normalized_config = JointLocalHeadConfig(
        **dict(checkpoint["config"])
    ).validated()
    if (
        normalized_config.architecture_dict()
        != model.config.architecture_dict()
    ):
        raise ValueError("joint checkpoint architecture does not match model")
    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("joint checkpoint state_dict must be non-empty")
    try:
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"incompatible joint checkpoint: {error}") from error
    metadata = checkpoint["metadata"]
    if not isinstance(metadata, Mapping):
        raise TypeError("joint checkpoint metadata must be a mapping")
    return dict(metadata)


def build_joint_local_head(
    *,
    enabled: bool,
    checkpoint_path: Optional[Union[str, Path]],
    config: Optional[JointLocalHeadConfig] = None,
    device: Any = "cpu",
) -> Optional[MultiViewJointLocalHead]:
    if not isinstance(enabled, (bool, np.bool_)):
        raise TypeError("enabled must be Boolean")
    if not enabled:
        return None
    if checkpoint_path is None:
        raise ValueError(
            "checkpoint_path is required when joint local head is enabled"
        )
    model = MultiViewJointLocalHead(config)
    load_joint_local_head_checkpoint(
        model, checkpoint_path, map_location=device
    )
    model.to(device)
    model.eval()
    return model


__all__ = [
    "JOINT_LOCAL_HEAD_SCHEMA",
    "JOINT_LOCAL_HEAD_FORMAT_VERSION",
    "JOINT_LOCAL_HEAD_COORDINATE_FRAME",
    "JOINT_LOCAL_HEAD_INPUT_SCHEMA",
    "JOINT_LOCAL_HEAD_OUTPUT_SCHEMA",
    "JOINT_VIEW_FEATURE_NAMES",
    "JOINT_VIEW_FEATURE_DIM",
    "JOINT_QUALITY_BRANCH_NAMES",
    "JOINT_QUALITY_COMPONENT_NAMES",
    "JointLocalHeadConfig",
    "JointViewInputs",
    "prepare_joint_view_inputs",
    "MultiViewJointLocalHead",
    "make_joint_local_head_checkpoint",
    "load_joint_local_head_checkpoint",
    "build_joint_local_head",
]
