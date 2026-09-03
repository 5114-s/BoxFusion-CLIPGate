"""SGCDet-inspired sparse, object-local box refinement.

This is a clean-room, dependency-light adaptation of the *sparse volume
construction* idea in SGCDet.  It is intentionally not a copy of the SGCDet
detector: BoxFusion already supplies object hypotheses and metric RGB-D
points, so this module builds a small canonical volume around each hypothesis
and predicts only a bounded local geometry residual.

The coarse-to-fine path is fixed and checkpointed:

``8 x 8 x 4 coarse volume -> learned occupancy -> stable hard Top-25% of a
16 x 16 x 8 fine volume -> selected-token aggregation -> local residual``.

Only the selected fine tokens enter the refinement MLP.  Selection is made
from detached occupancy logits and uses a stable sort, making ties
deterministic and preventing gradients from pretending that the discrete
Top-K operation is differentiable.  Occupancy logits remain differentiable
and are returned together with point-derived occupancy targets for an
explicit auxiliary loss.

The final layer is initialized to an identity geometry transform and
conservative probabilities.  Importantly, runtime activation must still use
a strictly versioned trained checkpoint; the safe initialization is not a
substitute for training or validation.

The implementation uses only NumPy and PyTorch.  It has no dependency on
MMCV, MMDetection3D, spconv, or the vendored SGCDet environment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np

from boxfusion.oriented_box_refiner import apply_local_box_residual_numpy

try:  # The NumPy residual utility remains importable without PyTorch.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - torch is present in test/runtime env.
    torch = None
    nn = None


SGCDET_SPARSE_REFINER_SCHEMA = "boxfusion.sgcdet_local_sparse_refiner"
SGCDET_SPARSE_REFINER_FORMAT_VERSION = 1
SGCDET_SPARSE_REFINER_COORDINATE_FRAME = "box_local"
SGCDET_SPARSE_REFINER_REFERENCE = "RM-Zhang/SGCDet@eb4ba52"
SGCDET_SPARSE_REFINER_QUALITY_DIM = 12

_CHECKPOINT_KEYS = frozenset(
    {
        "schema",
        "format_version",
        "coordinate_frame",
        "reference",
        "config",
        "state_dict",
        "metadata",
    }
)


def _probability_logit(value: float) -> float:
    probability = float(value)
    return float(np.log(probability / (1.0 - probability)))


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_positive(name: str, value: Any) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not np.isscalar(value)
        or not np.isfinite(value)
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _grid_tuple(name: str, value: Any) -> Tuple[int, int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise TypeError(f"{name} must be a length-three tuple")
    return tuple(
        _positive_integer(f"{name}[{index}]", item)
        for index, item in enumerate(value)
    )


@dataclass(frozen=True)
class SGCDetLocalSparseRefinerConfig:
    """Complete architecture signature and hard safety bounds.

    ``view_feature_dim`` is fixed at construction.  When no view features are
    available, :meth:`SGCDetInspiredLocalSparseRefiner.forward` supplies a
    zero tensor of this width, so the same checkpoint supports both the
    single-cloud and multi-view call sites.
    """

    point_feature_dim: int = 3
    quality_feature_dim: int = SGCDET_SPARSE_REFINER_QUALITY_DIM
    view_feature_dim: int = 9
    coarse_grid_size: Tuple[int, int, int] = (8, 8, 4)
    fine_grid_size: Tuple[int, int, int] = (16, 16, 8)
    topk_fraction: float = 0.25
    coarse_hidden_dim: int = 48
    coarse_embedding_dim: int = 64
    occupancy_hidden_dim: int = 48
    selected_hidden_dim: int = 64
    selected_embedding_dim: int = 96
    head_hidden_dim: int = 128
    grid_padding_fraction: float = 0.25
    max_center_fraction: float = 0.15
    max_log_dimension_residual: float = float(np.log(1.25))
    minimum_dimension: float = 1e-3
    default_candidate_iou: float = 0.10
    default_improvement_probability: float = 0.01
    default_uncertainty: float = 0.50
    minimum_uncertainty: float = 0.01
    maximum_uncertainty: float = 0.99

    def validated(self) -> "SGCDetLocalSparseRefinerConfig":
        for name in (
            "point_feature_dim",
            "quality_feature_dim",
            "view_feature_dim",
            "coarse_hidden_dim",
            "coarse_embedding_dim",
            "occupancy_hidden_dim",
            "selected_hidden_dim",
            "selected_embedding_dim",
            "head_hidden_dim",
        ):
            _positive_integer(name, getattr(self, name))
        if int(self.point_feature_dim) != 3:
            raise ValueError("point_feature_dim must equal 3 for local xyz")
        if int(self.quality_feature_dim) != SGCDET_SPARSE_REFINER_QUALITY_DIM:
            raise ValueError(
                "quality_feature_dim must equal "
                f"{SGCDET_SPARSE_REFINER_QUALITY_DIM}"
            )

        coarse = _grid_tuple("coarse_grid_size", self.coarse_grid_size)
        fine = _grid_tuple("fine_grid_size", self.fine_grid_size)
        if any(fine[index] != 2 * coarse[index] for index in range(3)):
            raise ValueError(
                "fine_grid_size must be exactly twice coarse_grid_size on "
                "every axis"
            )
        if coarse != (8, 8, 4) or fine != (16, 16, 8):
            raise ValueError(
                "the v1 schema requires an 8x8x4 -> 16x16x8 volume"
            )

        fraction = self.topk_fraction
        if (
            isinstance(fraction, (bool, np.bool_))
            or not np.isscalar(fraction)
            or not np.isfinite(fraction)
            or not 0.0 < float(fraction) <= 1.0
        ):
            raise ValueError("topk_fraction must lie in (0, 1]")
        if not np.isclose(float(fraction), 0.25, atol=0.0, rtol=0.0):
            raise ValueError("the v1 schema requires hard Top-25% selection")

        padding = self.grid_padding_fraction
        if (
            isinstance(padding, (bool, np.bool_))
            or not np.isscalar(padding)
            or not np.isfinite(padding)
            or float(padding) < 0.0
            or float(padding) > 1.0
        ):
            raise ValueError("grid_padding_fraction must lie in [0, 1]")
        for name in (
            "max_center_fraction",
            "max_log_dimension_residual",
            "minimum_dimension",
        ):
            _finite_positive(name, getattr(self, name))

        for name in (
            "default_candidate_iou",
            "default_improvement_probability",
            "default_uncertainty",
            "minimum_uncertainty",
            "maximum_uncertainty",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not np.isscalar(value)
                or not np.isfinite(value)
                or not 0.0 < float(value) < 1.0
            ):
                raise ValueError(f"{name} must lie strictly in (0, 1)")
        if float(self.default_improvement_probability) >= 0.5:
            raise ValueError(
                "default_improvement_probability must be below 0.5"
            )
        if float(self.minimum_uncertainty) >= float(self.maximum_uncertainty):
            raise ValueError(
                "minimum_uncertainty must be below maximum_uncertainty"
            )
        if not (
            float(self.minimum_uncertainty)
            <= float(self.default_uncertainty)
            <= float(self.maximum_uncertainty)
        ):
            raise ValueError(
                "default_uncertainty must lie inside the uncertainty bounds"
            )
        return self

    @property
    def fine_voxel_count(self) -> int:
        return int(np.prod(_grid_tuple("fine_grid_size", self.fine_grid_size)))

    @property
    def selected_token_count(self) -> int:
        # The v1 grid/fraction are exact, but ceil also defines future-safe
        # behavior for any schema extension.
        return int(np.ceil(self.fine_voxel_count * float(self.topk_fraction)))

    def architecture_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["coarse_grid_size"] = tuple(int(v) for v in self.coarse_grid_size)
        result["fine_grid_size"] = tuple(int(v) for v in self.fine_grid_size)
        return result


def stable_hard_topk(
    scores: Any, k: int
) -> Tuple[Any, Any]:
    """Return deterministic detached Top-K indices and a Boolean hard mask.

    Equal scores retain their original flat-voxel order, so ties select the
    lowest voxel indices.  Selection never carries a gradient; values gathered
    from the original logits/probabilities can still be used by downstream
    differentiable computation.
    """

    if torch is None:
        raise ImportError("stable_hard_topk requires PyTorch")
    if not torch.is_tensor(scores):
        raise TypeError("scores must be a torch.Tensor")
    if scores.ndim != 2 or scores.shape[1] < 1:
        raise ValueError("scores must have shape [B, N] with N > 0")
    if not scores.is_floating_point():
        raise TypeError("scores must use a floating-point dtype")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite")
    selected_count = _positive_integer("k", k)
    if selected_count > scores.shape[1]:
        raise ValueError("k cannot exceed the score dimension")

    detached = scores.detach()
    # stable=True is essential: torch.topk does not promise deterministic tie
    # ordering.  Supported PyTorch releases for BoxFusion provide this flag.
    order = torch.argsort(
        detached, dim=1, descending=True, stable=True
    )
    indices = order[:, :selected_count].detach()
    hard_mask = torch.zeros_like(detached, dtype=torch.bool)
    hard_mask.scatter_(1, indices, True)
    return indices, hard_mask.detach()


_ModuleBase = nn.Module if nn is not None else object


class SGCDetInspiredLocalSparseRefiner(_ModuleBase):
    """Coarse-to-fine sparse residual head for BoxFusion object crops."""

    # density, occupied, cell offset mean/std (3+3), view support, cell xyz (3)
    _GEOMETRY_VOXEL_FEATURE_DIM = 12
    _SELECTED_STATS_DIM = 4

    def __init__(
        self, config: Optional[SGCDetLocalSparseRefinerConfig] = None
    ) -> None:
        if torch is None or nn is None:
            raise ImportError(
                "SGCDetInspiredLocalSparseRefiner requires PyTorch"
            )
        super().__init__()
        self.config = (
            config or SGCDetLocalSparseRefinerConfig()
        ).validated()
        cfg = self.config
        voxel_feature_dim = (
            self._GEOMETRY_VOXEL_FEATURE_DIM + cfg.view_feature_dim
        )

        self.coarse_encoder = nn.Sequential(
            nn.Linear(voxel_feature_dim, cfg.coarse_hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(cfg.coarse_hidden_dim, cfg.coarse_embedding_dim),
            nn.ReLU(inplace=False),
        )
        self.coarse_occupancy_head = nn.Linear(
            cfg.coarse_embedding_dim, 1
        )
        fine_context_dim = voxel_feature_dim + cfg.coarse_embedding_dim
        self.fine_occupancy_head = nn.Sequential(
            nn.Linear(fine_context_dim, cfg.occupancy_hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(cfg.occupancy_hidden_dim, 1),
        )
        # This encoder is invoked only after hard Top-K gathering.
        self.selected_token_encoder = nn.Sequential(
            nn.Linear(fine_context_dim, cfg.selected_hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(cfg.selected_hidden_dim, cfg.selected_embedding_dim),
            nn.ReLU(inplace=False),
        )

        head_input_dim = (
            2 * cfg.selected_embedding_dim
            + cfg.quality_feature_dim
            + cfg.view_feature_dim
            + self._SELECTED_STATS_DIM
        )
        self.refinement_head = nn.Sequential(
            nn.Linear(head_input_dim, cfg.head_hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(cfg.head_hidden_dim, cfg.head_hidden_dim),
            nn.ReLU(inplace=False),
        )
        # center xyz, log-dimension xyz, IoU, improvement, uncertainty
        self.output_layer = nn.Linear(cfg.head_hidden_dim, 9)
        self._initialize_safe_identity()

        parent_indices = self._make_fine_parent_indices(
            cfg.coarse_grid_size, cfg.fine_grid_size
        )
        self.register_buffer(
            "_fine_parent_indices", parent_indices, persistent=False
        )

    def _initialize_safe_identity(self) -> None:
        cfg = self.config
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        uncertainty_span = (
            float(cfg.maximum_uncertainty)
            - float(cfg.minimum_uncertainty)
        )
        normalized_uncertainty = (
            float(cfg.default_uncertainty)
            - float(cfg.minimum_uncertainty)
        ) / uncertainty_span
        with torch.no_grad():
            self.output_layer.bias[6] = _probability_logit(
                cfg.default_candidate_iou
            )
            self.output_layer.bias[7] = _probability_logit(
                cfg.default_improvement_probability
            )
            self.output_layer.bias[8] = _probability_logit(
                normalized_uncertainty
            )

    @staticmethod
    def _make_fine_parent_indices(
        coarse_grid: Tuple[int, int, int],
        fine_grid: Tuple[int, int, int],
    ) -> Any:
        fine_coordinates = torch.stack(
            torch.meshgrid(
                *(torch.arange(size, dtype=torch.long) for size in fine_grid),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 3)
        parents = fine_coordinates // 2
        return (
            (parents[:, 0] * coarse_grid[1] + parents[:, 1])
            * coarse_grid[2]
            + parents[:, 2]
        )

    @staticmethod
    def _require_tensor(name: str, value: Any) -> Any:
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a torch.Tensor")
        return value

    def _validate_inputs(
        self,
        points_local: Any,
        point_mask: Optional[Any],
        local_boxes: Any,
        quality_features: Any,
        view_features: Optional[Any],
        view_mask: Optional[Any],
    ) -> Tuple[Any, Any, Any, Any, Any, Any, bool]:
        cfg = self.config
        points_local = self._require_tensor("points_local", points_local)
        if points_local.ndim not in (3, 4) or points_local.shape[-1] != 3:
            raise ValueError(
                "points_local must have shape [B, P, 3] or [B, V, P, 3]"
            )
        single_view_input = points_local.ndim == 3
        if single_view_input:
            points_local = points_local.unsqueeze(1)
        batch_size, view_count, point_count, _ = points_local.shape
        if batch_size < 1 or view_count < 1 or point_count < 1:
            raise ValueError(
                "points_local dimensions B, V, and P must be positive"
            )
        if not points_local.is_floating_point():
            raise TypeError("points_local must use a floating-point dtype")
        if not torch.isfinite(points_local).all():
            raise ValueError("points_local must be finite")

        if view_mask is None:
            view_mask = torch.ones(
                (batch_size, view_count),
                dtype=torch.bool,
                device=points_local.device,
            )
        else:
            view_mask = self._require_tensor("view_mask", view_mask)
            if single_view_input and view_mask.ndim == 1:
                view_mask = view_mask.unsqueeze(1)
            if view_mask.shape != (batch_size, view_count):
                raise ValueError("view_mask must have shape [B, V]")
            if view_mask.dtype != torch.bool:
                raise TypeError("view_mask must have Boolean dtype")
            if view_mask.device != points_local.device:
                raise ValueError("view_mask must be on the points_local device")
        if not torch.all(view_mask.any(dim=1)):
            raise ValueError("every sample must contain a valid view")

        if point_mask is None:
            point_mask = torch.ones(
                (batch_size, view_count, point_count),
                dtype=torch.bool,
                device=points_local.device,
            )
        else:
            point_mask = self._require_tensor("point_mask", point_mask)
            if single_view_input and point_mask.ndim == 2:
                point_mask = point_mask.unsqueeze(1)
            if point_mask.shape != (batch_size, view_count, point_count):
                raise ValueError(
                    "point_mask must match points without the xyz dimension"
                )
            if point_mask.dtype != torch.bool:
                raise TypeError("point_mask must have Boolean dtype")
            if point_mask.device != points_local.device:
                raise ValueError(
                    "point_mask must be on the points_local device"
                )
        point_mask = point_mask & view_mask.unsqueeze(-1)
        if not torch.all(point_mask.reshape(batch_size, -1).any(dim=1)):
            raise ValueError("every sample must contain at least one valid point")

        local_boxes = self._require_tensor("local_boxes", local_boxes)
        quality_features = self._require_tensor(
            "quality_features", quality_features
        )
        if local_boxes.shape != (batch_size, 6):
            raise ValueError("local_boxes must have shape [B, 6]")
        if quality_features.shape != (batch_size, cfg.quality_feature_dim):
            raise ValueError(
                "quality_features must have shape "
                f"[B, {cfg.quality_feature_dim}]"
            )
        for name, value in (
            ("local_boxes", local_boxes),
            ("quality_features", quality_features),
        ):
            if not value.is_floating_point():
                raise TypeError(f"{name} must use a floating-point dtype")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
            if value.device != points_local.device:
                raise ValueError(f"{name} must be on the points_local device")
            if value.dtype != points_local.dtype:
                raise ValueError(f"{name} must have the points_local dtype")
        if not torch.all(local_boxes[:, 3:6] > 0.0):
            raise ValueError("local box dimensions must be positive")
        if not torch.all(
            (quality_features >= 0.0) & (quality_features <= 1.0)
        ):
            raise ValueError("quality_features must lie in [0, 1]")

        if view_features is None:
            view_features = torch.zeros(
                (batch_size, view_count, cfg.view_feature_dim),
                dtype=points_local.dtype,
                device=points_local.device,
            )
        else:
            view_features = self._require_tensor(
                "view_features", view_features
            )
            if single_view_input and view_features.ndim == 2:
                view_features = view_features.unsqueeze(1)
            expected = (batch_size, view_count, cfg.view_feature_dim)
            if view_features.shape != expected:
                raise ValueError(f"view_features must have shape {expected}")
            if not view_features.is_floating_point():
                raise TypeError("view_features must use a floating-point dtype")
            if not torch.isfinite(view_features).all():
                raise ValueError("view_features must be finite")
            if view_features.device != points_local.device:
                raise ValueError(
                    "view_features must be on the points_local device"
                )
            if view_features.dtype != points_local.dtype:
                raise ValueError(
                    "view_features must have the points_local dtype"
                )

        parameter = next(self.parameters())
        if parameter.device != points_local.device:
            raise ValueError("model and inputs must be on the same device")
        if parameter.dtype != points_local.dtype:
            raise ValueError("model and floating inputs must have the same dtype")
        return (
            points_local,
            point_mask,
            local_boxes,
            quality_features,
            view_features,
            view_mask,
            single_view_input,
        )

    @staticmethod
    def _voxel_centers(
        grid: Tuple[int, int, int], *, device: Any, dtype: Any
    ) -> Any:
        axes = [
            (torch.arange(size, device=device, dtype=dtype) + 0.5)
            / float(size)
            * 2.0
            - 1.0
            for size in grid
        ]
        return torch.stack(
            torch.meshgrid(*axes, indexing="ij"), dim=-1
        ).reshape(-1, 3)

    def _voxelize(
        self,
        points: Any,
        point_mask: Any,
        local_boxes: Any,
        view_features: Any,
        grid: Tuple[int, int, int],
    ) -> Tuple[Any, Any, Any, Any]:
        """Build differentiable statistics with discrete point assignment."""

        batch_size, view_count, point_count, _ = points.shape
        voxel_count = int(np.prod(grid))
        dimensions = local_boxes[:, None, None, 3:6]
        centers = local_boxes[:, None, None, :3]
        span = dimensions * (
            1.0 + 2.0 * float(self.config.grid_padding_fraction)
        )
        normalized = (points - centers) / span + 0.5
        in_grid = ((normalized >= 0.0) & (normalized < 1.0)).all(dim=-1)
        valid = point_mask & in_grid

        grid_tensor = points.new_tensor(grid).view(1, 1, 1, 3)
        coordinates = torch.floor(normalized * grid_tensor).to(torch.long)
        coordinate_min = torch.zeros_like(coordinates)
        coordinate_max = torch.tensor(
            grid, device=points.device, dtype=torch.long
        ).view(1, 1, 1, 3) - 1
        coordinates = torch.maximum(
            coordinate_min, torch.minimum(coordinates, coordinate_max)
        )
        linear = (
            (coordinates[..., 0] * grid[1] + coordinates[..., 1])
            * grid[2]
            + coordinates[..., 2]
        )

        batch_offsets = (
            torch.arange(batch_size, device=points.device)
            .view(batch_size, 1, 1)
            * voxel_count
        )
        global_linear = (linear + batch_offsets).reshape(-1)
        valid_flat = valid.reshape(-1)
        chosen = global_linear[valid_flat]

        count_flat = points.new_zeros(batch_size * voxel_count)
        count_flat.scatter_add_(
            0, chosen, points.new_ones(chosen.shape[0])
        )
        counts = count_flat.view(batch_size, voxel_count)

        cell_position = normalized * grid_tensor
        cell_offset = cell_position - torch.floor(cell_position) - 0.5
        offset_flat = cell_offset.reshape(-1, 3)[valid_flat]
        offset_sum = points.new_zeros(batch_size * voxel_count, 3)
        offset_square_sum = points.new_zeros(batch_size * voxel_count, 3)
        scatter_index = chosen[:, None].expand(-1, 3)
        offset_sum.scatter_add_(0, scatter_index, offset_flat)
        offset_square_sum.scatter_add_(
            0, scatter_index, offset_flat.square()
        )
        divisor = count_flat.clamp_min(1.0).unsqueeze(-1)
        offset_mean = (offset_sum / divisor).view(batch_size, voxel_count, 3)
        variance = (offset_square_sum / divisor) - (
            offset_sum / divisor
        ).square()
        offset_std = variance.clamp_min(0.0).sqrt().view(
            batch_size, voxel_count, 3
        )

        # Unique view support and view-feature averaging per voxel.
        view_offsets = (
            torch.arange(batch_size * view_count, device=points.device)
            .view(batch_size, view_count, 1)
            * voxel_count
        )
        view_linear = (linear + view_offsets).reshape(-1)
        chosen_view = view_linear[valid_flat]
        per_view_counts = points.new_zeros(
            batch_size * view_count * voxel_count
        )
        per_view_counts.scatter_add_(
            0, chosen_view, points.new_ones(chosen_view.shape[0])
        )
        view_hits = (
            per_view_counts.view(batch_size, view_count, voxel_count) > 0.0
        )
        view_hit_float = view_hits.to(points.dtype)
        view_support_count = view_hit_float.sum(dim=1)
        view_support = view_support_count / float(view_count)
        voxel_view_features = torch.einsum(
            "bvn,bvf->bnf", view_hit_float, view_features
        ) / view_support_count.clamp_min(1.0).unsqueeze(-1)

        valid_point_count = valid.reshape(batch_size, -1).sum(dim=1)
        density_denominator = torch.log1p(
            valid_point_count.to(points.dtype).clamp_min(1.0)
        ).unsqueeze(-1).clamp_min(1.0)
        density = torch.log1p(counts) / density_denominator
        occupied = counts > 0.0
        centers_feature = self._voxel_centers(
            grid, device=points.device, dtype=points.dtype
        ).unsqueeze(0).expand(batch_size, -1, -1)
        features = torch.cat(
            (
                density.unsqueeze(-1),
                occupied.to(points.dtype).unsqueeze(-1),
                offset_mean,
                offset_std,
                view_support.unsqueeze(-1),
                centers_feature,
                voxel_view_features,
            ),
            dim=-1,
        )
        return features, occupied.to(points.dtype), counts, valid_point_count

    @staticmethod
    def _gather_tokens(tokens: Any, indices: Any) -> Any:
        return torch.gather(
            tokens,
            1,
            indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
        )

    def forward(
        self,
        points_local: Any,
        point_mask: Optional[Any],
        local_boxes: Any,
        quality_features: Any,
        view_features: Optional[Any] = None,
        view_mask: Optional[Any] = None,
    ) -> Dict[str, Any]:
        (
            points_local,
            point_mask,
            local_boxes,
            quality_features,
            view_features,
            view_mask,
            _,
        ) = self._validate_inputs(
            points_local,
            point_mask,
            local_boxes,
            quality_features,
            view_features,
            view_mask,
        )
        cfg = self.config
        coarse_features, coarse_targets, _, _ = self._voxelize(
            points_local,
            point_mask,
            local_boxes,
            view_features,
            cfg.coarse_grid_size,
        )
        fine_features, occupancy_targets, _, valid_point_count = self._voxelize(
            points_local,
            point_mask,
            local_boxes,
            view_features,
            cfg.fine_grid_size,
        )

        coarse_embedding = self.coarse_encoder(coarse_features)
        coarse_occupancy_logits = self.coarse_occupancy_head(
            coarse_embedding
        ).squeeze(-1)
        parent_embedding = coarse_embedding[:, self._fine_parent_indices, :]
        fine_context = torch.cat((fine_features, parent_embedding), dim=-1)
        occupancy_logits = self.fine_occupancy_head(fine_context).squeeze(-1)
        occupancy_probability = torch.sigmoid(occupancy_logits)

        selected_indices, selected_mask = stable_hard_topk(
            occupancy_logits, cfg.selected_token_count
        )
        selected_context = self._gather_tokens(
            fine_context, selected_indices
        )
        # The costly token encoder processes precisely K selected voxels.
        selected_embedding = self.selected_token_encoder(selected_context)
        selected_probability = torch.gather(
            occupancy_probability, 1, selected_indices
        )
        selected_targets = torch.gather(
            occupancy_targets, 1, selected_indices
        )
        selected_maximum = selected_embedding.amax(dim=1)
        probability_weight = selected_probability.unsqueeze(-1)
        selected_mean = (
            selected_embedding * probability_weight
        ).sum(dim=1) / probability_weight.sum(dim=1).clamp_min(1e-6)

        valid_views = (
            point_mask.any(dim=2) & view_mask
        ).to(points_local.dtype)
        global_view_features = (
            view_features * valid_views.unsqueeze(-1)
        ).sum(dim=1) / valid_views.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        selected_fraction = points_local.new_full(
            (points_local.shape[0],), float(cfg.topk_fraction)
        )
        selected_stats_tensor = torch.stack(
            (
                selected_probability.mean(dim=1),
                selected_probability.amax(dim=1),
                selected_targets.mean(dim=1),
                selected_fraction,
            ),
            dim=-1,
        )
        fused = torch.cat(
            (
                selected_maximum,
                selected_mean,
                quality_features,
                global_view_features,
                selected_stats_tensor,
            ),
            dim=-1,
        )
        raw = self.output_layer(self.refinement_head(fused))

        center_residual_fraction = (
            torch.tanh(raw[:, :3]) * cfg.max_center_fraction
        )
        center_residual = center_residual_fraction * local_boxes[:, 3:6]
        log_dimension_residual = (
            torch.tanh(raw[:, 3:6])
            * cfg.max_log_dimension_residual
        )
        candidate_iou = torch.sigmoid(raw[:, 6])
        improvement_probability = torch.sigmoid(raw[:, 7])
        uncertainty_unit = torch.sigmoid(raw[:, 8])
        uncertainty = cfg.minimum_uncertainty + uncertainty_unit * (
            cfg.maximum_uncertainty - cfg.minimum_uncertainty
        )

        selected_count = torch.full(
            (points_local.shape[0],),
            cfg.selected_token_count,
            dtype=torch.long,
            device=points_local.device,
        )
        selected_stats = {
            "count": selected_count,
            "fraction": selected_fraction,
            "occupancy_mean": selected_probability.mean(dim=1),
            "occupancy_maximum": selected_probability.amax(dim=1),
            "target_fraction": selected_targets.mean(dim=1),
            "valid_point_count": valid_point_count,
        }
        return {
            "center_residual": center_residual,
            "center_residual_fraction": center_residual_fraction,
            "log_dimension_residual": log_dimension_residual,
            "candidate_iou": candidate_iou,
            "improvement_probability": improvement_probability,
            "uncertainty": uncertainty,
            "coarse_occupancy_logits": coarse_occupancy_logits,
            "coarse_occupancy_targets": coarse_targets,
            "occupancy_logits": occupancy_logits,
            "occupancy_targets": occupancy_targets,
            "selected_indices": selected_indices,
            "selected_mask": selected_mask,
            "selected_stats": selected_stats,
        }


def apply_sgcdet_sparse_residual_numpy(
    local_boxes: np.ndarray,
    center_residual: np.ndarray,
    log_dimension_residual: np.ndarray,
    *,
    config: Optional[SGCDetLocalSparseRefinerConfig] = None,
    maximum_dimension: Optional[float] = None,
) -> np.ndarray:
    """Apply one bounded sparse-refiner residual without a torch dependency."""

    cfg = (config or SGCDetLocalSparseRefinerConfig()).validated()
    return apply_local_box_residual_numpy(
        local_boxes,
        center_residual,
        log_dimension_residual,
        max_center_fraction=cfg.max_center_fraction,
        max_abs_log_dimension_residual=cfg.max_log_dimension_residual,
        minimum_dimension=cfg.minimum_dimension,
        maximum_dimension=maximum_dimension,
    )


def _torch_load_weights(path: Path, map_location: Any) -> Any:
    if torch is None:
        raise ImportError("loading a sparse-refiner checkpoint requires PyTorch")
    try:
        return torch.load(
            str(path), map_location=map_location, weights_only=True
        )
    except TypeError:  # pragma: no cover - compatibility with older PyTorch.
        return torch.load(str(path), map_location=map_location)


def _validated_metadata(metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    if not all(isinstance(key, str) for key in metadata):
        raise ValueError("metadata keys must be strings")
    return dict(metadata)


def make_sgcdet_sparse_refiner_checkpoint(
    model: SGCDetInspiredLocalSparseRefiner,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the exact v1 checkpoint payload."""

    if torch is None or nn is None:
        raise ImportError("creating a sparse-refiner checkpoint requires PyTorch")
    if not isinstance(model, SGCDetInspiredLocalSparseRefiner):
        raise TypeError("model must be an SGCDetInspiredLocalSparseRefiner")
    return {
        "schema": SGCDET_SPARSE_REFINER_SCHEMA,
        "format_version": SGCDET_SPARSE_REFINER_FORMAT_VERSION,
        "coordinate_frame": SGCDET_SPARSE_REFINER_COORDINATE_FRAME,
        "reference": SGCDET_SPARSE_REFINER_REFERENCE,
        "config": model.config.architecture_dict(),
        "state_dict": model.state_dict(),
        "metadata": _validated_metadata(metadata),
    }


def save_sgcdet_sparse_refiner_checkpoint(
    model: SGCDetInspiredLocalSparseRefiner,
    checkpoint_path: Union[str, Path],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Atomically persist one strictly versioned checkpoint."""

    if torch is None:
        raise ImportError("saving a sparse-refiner checkpoint requires PyTorch")
    path = Path(checkpoint_path)
    if path.exists() and path.is_dir():
        raise IsADirectoryError(f"checkpoint path is a directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(
        make_sgcdet_sparse_refiner_checkpoint(model, metadata=metadata),
        str(temporary),
    )
    temporary.replace(path)
    return path


def _validate_checkpoint_payload(
    checkpoint: Any,
    model: SGCDetInspiredLocalSparseRefiner,
) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    received = set(checkpoint.keys())
    if received != _CHECKPOINT_KEYS:
        missing = sorted(_CHECKPOINT_KEYS - received)
        extra = sorted(received - _CHECKPOINT_KEYS)
        raise ValueError(
            "sparse-refiner checkpoint keys do not match the strict schema "
            f"(missing={missing}, extra={extra})"
        )
    if checkpoint["schema"] != SGCDET_SPARSE_REFINER_SCHEMA:
        raise ValueError("unsupported sparse-refiner checkpoint schema")
    version = checkpoint["format_version"]
    if (
        isinstance(version, bool)
        or not isinstance(version, (int, np.integer))
        or int(version) != SGCDET_SPARSE_REFINER_FORMAT_VERSION
    ):
        raise ValueError("unsupported sparse-refiner checkpoint format_version")
    if (
        checkpoint["coordinate_frame"]
        != SGCDET_SPARSE_REFINER_COORDINATE_FRAME
    ):
        raise ValueError(
            "sparse-refiner checkpoint coordinate_frame must be box_local"
        )
    if checkpoint["reference"] != SGCDET_SPARSE_REFINER_REFERENCE:
        raise ValueError("sparse-refiner checkpoint reference commit differs")
    if not isinstance(checkpoint["config"], Mapping):
        raise ValueError("checkpoint config must be a mapping")
    if dict(checkpoint["config"]) != model.config.architecture_dict():
        raise ValueError("checkpoint config does not match the model config")
    _validated_metadata(checkpoint["metadata"])
    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("checkpoint state_dict must be a non-empty mapping")
    if not all(isinstance(key, str) for key in state_dict):
        raise ValueError("checkpoint state_dict keys must be strings")
    if not all(torch.is_tensor(value) for value in state_dict.values()):
        raise ValueError("checkpoint state_dict values must be tensors")
    return checkpoint


def load_sgcdet_sparse_refiner_checkpoint(
    model: SGCDetInspiredLocalSparseRefiner,
    checkpoint_path: Union[str, Path],
    *,
    map_location: Any = "cpu",
) -> SGCDetInspiredLocalSparseRefiner:
    """Strictly validate and load one sparse-refiner checkpoint."""

    if torch is None or nn is None:
        raise ImportError("loading a sparse-refiner checkpoint requires PyTorch")
    if not isinstance(model, SGCDetInspiredLocalSparseRefiner):
        raise TypeError("model must be an SGCDetInspiredLocalSparseRefiner")
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"sparse-refiner checkpoint not found: {path}")
    checkpoint = _validate_checkpoint_payload(
        _torch_load_weights(path, map_location), model
    )
    try:
        model.load_state_dict(dict(checkpoint["state_dict"]), strict=True)
    except RuntimeError as error:
        raise ValueError(f"incompatible sparse-refiner checkpoint: {error}") from error
    return model


def build_sgcdet_sparse_refiner(
    *,
    enabled: bool,
    checkpoint_path: Optional[Union[str, Path]] = None,
    config: Optional[SGCDetLocalSparseRefinerConfig] = None,
    device: Any = "cpu",
) -> Optional[SGCDetInspiredLocalSparseRefiner]:
    """Build an evaluated model; enabling always requires a checkpoint."""

    if not isinstance(enabled, (bool, np.bool_)):
        raise TypeError("enabled must be Boolean")
    if not bool(enabled):
        return None
    if torch is None or nn is None:
        raise ImportError("enabled sparse refinement requires PyTorch")
    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required when refinement is enabled")
    path = Path(checkpoint_path)
    if config is None:
        if not path.is_file():
            raise FileNotFoundError(f"sparse-refiner checkpoint not found: {path}")
        payload = _torch_load_weights(path, "cpu")
        if not isinstance(payload, Mapping) or not isinstance(
            payload.get("config"), Mapping
        ):
            raise ValueError("checkpoint has no valid config mapping")
        try:
            config = SGCDetLocalSparseRefinerConfig(
                **dict(payload["config"])
            ).validated()
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid sparse-refiner checkpoint config: {error}") from error
    model = SGCDetInspiredLocalSparseRefiner(config)
    load_sgcdet_sparse_refiner_checkpoint(
        model, path, map_location=device
    )
    model.to(device)
    model.eval()
    return model


__all__ = [
    "SGCDET_SPARSE_REFINER_SCHEMA",
    "SGCDET_SPARSE_REFINER_FORMAT_VERSION",
    "SGCDET_SPARSE_REFINER_COORDINATE_FRAME",
    "SGCDET_SPARSE_REFINER_REFERENCE",
    "SGCDET_SPARSE_REFINER_QUALITY_DIM",
    "SGCDetLocalSparseRefinerConfig",
    "SGCDetInspiredLocalSparseRefiner",
    "stable_hard_topk",
    "apply_sgcdet_sparse_residual_numpy",
    "make_sgcdet_sparse_refiner_checkpoint",
    "save_sgcdet_sparse_refiner_checkpoint",
    "load_sgcdet_sparse_refiner_checkpoint",
    "build_sgcdet_sparse_refiner",
]
