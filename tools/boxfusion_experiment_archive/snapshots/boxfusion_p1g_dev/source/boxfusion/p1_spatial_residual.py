"""Native sparse spatial context for the P1-v2 residual proposal observer.

The original P1 head applies an MLP independently to every residual voxel.
This module supplies a small, dependency-free submanifold backbone which
propagates information over the six axis-aligned occupied neighbours while
preserving the original voxel row/anchor set.

No dense volume is constructed.  Neighbours are resolved with a collision-free
mixed-radix integer key and vectorised ``torch.searchsorted``.  The topology is
derived from integer coordinates only, is detached from autograd, and is never
modified by the network.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn


P1_SPATIAL_ARCHITECTURE = "native_sparse_context_v1"
P1_SPATIAL_FEATURE_DIM = 14
P1_SPATIAL_REGRESSION_DIM = 6
P1_SPATIAL_NEIGHBORHOOD = "axis6_submanifold"

_AXIS6 = (
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
)
_INT_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}
_INT64_MAX = torch.iinfo(torch.int64).max


def _validate_dilations(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("dilations must be a sequence of positive integers")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("dilations must contain positive integers")
        if value <= 0:
            raise ValueError("dilations must contain positive integers")
        result.append(int(value))
    if not result:
        raise ValueError("at least one sparse dilation is required")
    if len(set(result)) != len(result):
        raise ValueError("sparse dilations must be unique")
    return tuple(result)


def _canonical_coordinates(
    coordinates: Any,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return detached ``[V,batch,x,y,z]``-style coordinates as ``[V,4]``.

    A three-column input represents one sparse sample and receives a zero
    batch column.  Four-column input is interpreted as ``batch,x,y,z``.
    """

    values = (
        coordinates
        if isinstance(coordinates, torch.Tensor)
        else torch.as_tensor(coordinates)
    )
    if values.ndim != 2 or values.shape[1] not in (3, 4):
        raise ValueError("coordinates must have shape [V,3] or [V,4]")
    if values.dtype not in _INT_DTYPES:
        raise TypeError("coordinates must use an integer dtype")
    values = values.detach().to(device=device, dtype=torch.int64)
    if values.shape[1] == 3:
        batch = torch.zeros(
            (len(values), 1), dtype=torch.int64, device=values.device
        )
        values = torch.cat((batch, values), dim=1)
    return values.contiguous()


def _mixed_radix_keys(
    coordinates: torch.Tensor,
    lower: torch.Tensor,
    spans: tuple[int, int, int, int],
) -> torch.Tensor:
    shifted = coordinates - lower
    if bool(torch.any(shifted < 0)):
        raise RuntimeError("sparse coordinates lie below the key domain")
    keys = shifted[:, 0]
    for axis in range(1, 4):
        keys = keys * spans[axis] + shifted[:, axis]
    return keys


def build_axis6_neighbor_indices(
    coordinates: Any,
    dilations: Sequence[int] = (1, 2, 4),
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build deterministic occupied-neighbour indices.

    Args:
        coordinates: Integer ``[V,3]`` xyz or ``[V,4]`` batch-xyz rows.
        dilations: Positive voxel offsets.  Direction order for every
            dilation is ``-x,+x,-y,+y,-z,+z``.
        device: Optional output/computation device.

    Returns:
        Detached ``int64 [V,L,6]`` indices into the original row order.
        Missing neighbours are encoded as ``-1``.
    """

    resolved_dilations = _validate_dilations(dilations)
    target_device = None if device is None else torch.device(device)
    coords = _canonical_coordinates(coordinates, device=target_device)
    voxel_count = int(coords.shape[0])
    if voxel_count == 0:
        return torch.empty(
            (0, len(resolved_dilations), 6),
            dtype=torch.int64,
            device=coords.device,
        )

    maximum_dilation = max(resolved_dilations)
    spatial_padding = torch.tensor(
        (0, maximum_dilation, maximum_dilation, maximum_dilation),
        dtype=torch.int64,
        device=coords.device,
    )
    lower = torch.amin(coords, dim=0) - spatial_padding
    upper = torch.amax(coords, dim=0) + spatial_padding
    span_tensor = upper - lower + 1
    spans = tuple(int(value) for value in span_tensor.detach().cpu().tolist())
    domain_size = 1
    for span in spans:
        if span <= 0 or domain_size > _INT64_MAX // span:
            raise OverflowError(
                "coordinate range is too large for collision-free int64 keys"
            )
        domain_size *= span

    keys = _mixed_radix_keys(coords, lower, spans)
    sorted_keys, sorted_rows = torch.sort(keys)
    if voxel_count > 1 and bool(
        torch.any(sorted_keys[1:] == sorted_keys[:-1])
    ):
        raise ValueError(
            "coordinates must be unique within each sparse batch"
        )

    offsets = torch.tensor(
        [
            [
                (0, dilation * dx, dilation * dy, dilation * dz)
                for dx, dy, dz in _AXIS6
            ]
            for dilation in resolved_dilations
        ],
        dtype=torch.int64,
        device=coords.device,
    )
    queries = coords[:, None, None, :] + offsets[None, :, :, :]
    query_keys = _mixed_radix_keys(
        queries.reshape(-1, 4), lower, spans
    ).reshape(voxel_count, len(resolved_dilations), 6)
    flat_queries = query_keys.reshape(-1).contiguous()
    positions = torch.searchsorted(sorted_keys, flat_queries)
    safe_positions = positions.clamp(max=voxel_count - 1)
    matches = (positions < voxel_count) & (
        sorted_keys[safe_positions] == flat_queries
    )
    neighbours = torch.where(
        matches,
        sorted_rows[safe_positions],
        torch.full_like(safe_positions, -1),
    )
    return neighbours.reshape(
        voxel_count, len(resolved_dilations), 6
    ).detach()


class _SparseAxisResidualBlock(nn.Module):
    """One fixed-resolution sparse six-neighbour residual block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.input_norm = nn.LayerNorm(self.channels)
        self.center = nn.Linear(self.channels, self.channels)
        self.directional_weight = nn.Parameter(
            torch.empty(6, self.channels, self.channels)
        )
        self.activation = nn.SiLU()
        self.ffn_norm = nn.LayerNorm(self.channels)
        self.ffn = nn.Sequential(
            nn.Linear(self.channels, 2 * self.channels),
            nn.SiLU(),
            nn.Linear(2 * self.channels, self.channels),
        )
        nn.init.kaiming_uniform_(
            self.directional_weight, a=math.sqrt(5)
        )

    def forward(
        self, features: torch.Tensor, neighbour_indices: torch.Tensor
    ) -> torch.Tensor:
        voxel_count = int(features.shape[0])
        if neighbour_indices.shape != (voxel_count, 6):
            raise ValueError("neighbour_indices must have shape [V,6]")
        if neighbour_indices.dtype != torch.int64:
            raise TypeError("neighbour_indices must use int64")
        if voxel_count == 0:
            return features

        normalized = self.input_norm(features)
        valid = neighbour_indices >= 0
        safe = neighbour_indices.clamp(min=0)
        gathered = normalized.index_select(0, safe.reshape(-1)).reshape(
            voxel_count, 6, self.channels
        )
        gathered = gathered * valid.unsqueeze(-1).to(gathered.dtype)
        directional = torch.einsum(
            "vkc,kco->vko", gathered, self.directional_weight
        )
        spatial = self.center(normalized) + directional.sum(dim=1) / 6.0
        output = features + self.activation(spatial)
        return output + self.ffn(self.ffn_norm(output))


class NativeSparseResidualProposalHead(nn.Module):
    """Class-agnostic native sparse P1 proposal head.

    The head preserves one output row per input voxel.  It predicts no semantic
    classes and has the same logits/regression shapes as the legacy P1 MLP.
    """

    architecture = P1_SPATIAL_ARCHITECTURE

    def __init__(
        self,
        input_dim: int = P1_SPATIAL_FEATURE_DIM,
        hidden_dim: int = 48,
        regression_dim: int = P1_SPATIAL_REGRESSION_DIM,
        dilations: Sequence[int] = (1, 2, 4),
    ) -> None:
        super().__init__()
        if isinstance(input_dim, bool) or int(input_dim) != (
            P1_SPATIAL_FEATURE_DIM
        ):
            raise ValueError(
                f"input_dim must equal the fixed P1 dimension "
                f"{P1_SPATIAL_FEATURE_DIM}"
            )
        if isinstance(hidden_dim, bool) or int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if isinstance(regression_dim, bool) or int(regression_dim) != (
            P1_SPATIAL_REGRESSION_DIM
        ):
            raise ValueError(
                f"regression_dim must equal {P1_SPATIAL_REGRESSION_DIM}"
            )
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.regression_dim = int(regression_dim)
        self.dilations = _validate_dilations(dilations)
        self.stem = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )
        self.spatial_blocks = nn.ModuleList(
            _SparseAxisResidualBlock(self.hidden_dim)
            for _ in self.dilations
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.objectness = nn.Linear(self.hidden_dim, 1)
        self.regression = nn.Linear(
            self.hidden_dim, self.regression_dim
        )

    def encode(
        self, features: torch.Tensor, coordinates: Any
    ) -> torch.Tensor:
        """Return the frozen sparse feature rows before prediction heads.

        P1G uses this public method to freeze the P1S proposal/objectness
        path while fitting a separate geometry-only regression head.  It
        preserves one encoded row per original sparse anchor.
        """

        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a torch.Tensor")
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                f"features must have shape [V,{self.input_dim}]"
            )
        if not features.is_floating_point():
            raise TypeError("features must use a floating dtype")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("features must be finite")
        coords = _canonical_coordinates(
            coordinates, device=features.device
        )
        if len(coords) != len(features):
            raise ValueError("features and coordinates must have equal V")
        if len(features) == 0:
            return features.new_empty((0, self.hidden_dim))

        topology = build_axis6_neighbor_indices(
            coords, self.dilations, device=features.device
        )
        encoded = self.stem(features)
        for block_index, block in enumerate(self.spatial_blocks):
            encoded = block(encoded, topology[:, block_index, :])
        return self.output_norm(encoded)

    def forward(
        self, features: torch.Tensor, coordinates: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(features, coordinates)
        return self.objectness(encoded), self.regression(encoded)

    def model_config(self) -> dict[str, Any]:
        return {
            "architecture": P1_SPATIAL_ARCHITECTURE,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "regression_dim": self.regression_dim,
            "dilations": list(self.dilations),
            "neighborhood": P1_SPATIAL_NEIGHBORHOOD,
            "coordinate_layout": "xyz_or_batch_xyz",
            "regression_encoding": "center_delta_m_log_size_m",
        }

    @classmethod
    def from_model_config(
        cls, config: Mapping[str, Any]
    ) -> "NativeSparseResidualProposalHead":
        """Reconstruct a head from its strict checkpoint model mapping."""

        if not isinstance(config, Mapping):
            raise TypeError("model_config must be a mapping")
        required = {
            "architecture",
            "input_dim",
            "hidden_dim",
            "regression_dim",
            "dilations",
            "neighborhood",
            "coordinate_layout",
            "regression_encoding",
        }
        missing = sorted(required - set(config))
        unknown = sorted(set(config) - required)
        if missing or unknown:
            detail = []
            if missing:
                detail.append("missing=" + ",".join(missing))
            if unknown:
                detail.append("unknown=" + ",".join(unknown))
            raise ValueError("invalid sparse model_config: " + "; ".join(detail))
        if config["architecture"] != P1_SPATIAL_ARCHITECTURE:
            raise ValueError("sparse model_config architecture mismatch")
        if config["neighborhood"] != P1_SPATIAL_NEIGHBORHOOD:
            raise ValueError("sparse model_config neighborhood mismatch")
        if config["coordinate_layout"] != "xyz_or_batch_xyz":
            raise ValueError("sparse model_config coordinate layout mismatch")
        if config["regression_encoding"] != (
            "center_delta_m_log_size_m"
        ):
            raise ValueError("sparse model_config regression encoding mismatch")
        return cls(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            regression_dim=config["regression_dim"],
            dilations=config["dilations"],
        )


__all__ = [
    "NativeSparseResidualProposalHead",
    "P1_SPATIAL_ARCHITECTURE",
    "P1_SPATIAL_FEATURE_DIM",
    "P1_SPATIAL_NEIGHBORHOOD",
    "P1_SPATIAL_REGRESSION_DIM",
    "build_axis6_neighbor_indices",
]
