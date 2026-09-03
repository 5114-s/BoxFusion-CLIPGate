"""Deterministic raw/fused proposal-query observer.

This module provides a small NumPy-only adapter for comparing proposals from
four deliberately distinct streams:

``original``
    The frozen upstream BoxFusion proposal.
``raw_mask``
    A proposal fitted directly from mask and measured-depth points.
``superpoint``
    A proposal fitted after local connected-component/superpoint separation.
``occupancy``
    A proposal produced by a sparse occupancy or MSR refiner.

The adapter canonicalizes 6D centre-size boxes and explicit 8-corner boxes,
computes pairwise geometric consensus, creates one fixed-schema feature row
per candidate, and deterministically selects an *observer candidate*.  It has
no final-box input and no mutation path: selection is diagnostic evidence,
not permission to replace an upstream detection.

Without a checkpoint, selection is explicitly labelled
``deterministic_heuristic`` and learned scores are NaN.  Optional learned
selection accepts only strict, pickle-free ``.npz`` linear or MLP checkpoints
whose feature names exactly match :data:`RAW_FUSED_QUERY_FEATURE_NAMES`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


RAW_FUSED_QUERY_SCHEMA = "boxfusion.raw_fused_query.observer.v1"
RAW_FUSED_CANDIDATE_TABLE_SCHEMA = (
    "boxfusion.raw_fused_query.candidate_table.v1"
)
RAW_FUSED_PAIRWISE_SCHEMA = "boxfusion.raw_fused_query.pairwise.v1"
RAW_FUSED_QUERY_SCORER_SCHEMA = "boxfusion.raw_fused_query.scorer.v1"
RAW_FUSED_QUERY_SCORER_FORMAT_VERSION = 1

RAW_FUSED_QUERY_SOURCES = (
    "original",
    "raw_mask",
    "superpoint",
    "occupancy",
)

RAW_FUSED_INPUT_QUALITY_NAMES = (
    "proposal_score",
    "mask_quality",
    "depth_quality",
    "geometry_quality",
    "view_quality",
)
RAW_FUSED_INPUT_QUALITY_DIM = len(RAW_FUSED_INPUT_QUALITY_NAMES)

RAW_FUSED_QUERY_FEATURE_NAMES = (
    "source_original",
    "source_raw_mask",
    "source_superpoint",
    "source_occupancy",
    *RAW_FUSED_INPUT_QUALITY_NAMES,
    "quality_mean",
    "quality_min",
    "mean_pairwise_iou",
    "max_pairwise_iou",
    "mean_center_similarity",
    "max_center_similarity",
    "mean_extent_similarity",
    "max_extent_similarity",
    "mean_pairwise_consensus",
    "max_pairwise_consensus",
    "mean_cross_source_consensus",
    "max_cross_source_consensus",
    "cross_source_support_025",
    "cross_source_support_050",
    "source_support_fraction",
    "original_max_consensus",
    "raw_mask_max_consensus",
    "superpoint_max_consensus",
    "occupancy_max_consensus",
    "volume_similarity_to_original",
    "consensus_margin",
)
RAW_FUSED_QUERY_FEATURE_DIM = len(RAW_FUSED_QUERY_FEATURE_NAMES)
assert RAW_FUSED_QUERY_FEATURE_DIM == 30

_SOURCE_TO_INDEX = {
    source: index for index, source in enumerate(RAW_FUSED_QUERY_SOURCES)
}
_CORNER_SIGNS = np.asarray(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)


def _read_only(
    value: Any,
    *,
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative_exp = np.exp(values[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return result


def _string_scalar(name: str, value: Any) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar string")
    result = str(array.item())
    if not result:
        raise ValueError(f"{name} cannot be empty")
    return result


def _integer_scalar(name: str, value: Any) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer scalar")
    return int(array.item())


def _float_scalar(name: str, value: Any) -> float:
    array = np.asarray(value, dtype=np.float64)
    if array.size != 1:
        raise ValueError(f"{name} must be a scalar")
    result = float(array.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def raw_fused_input_quality_vector(features: Mapping[str, Any]) -> np.ndarray:
    """Return one read-only quality vector in the exact public schema."""

    if not isinstance(features, Mapping):
        raise TypeError("input quality features must be a mapping")
    received = set(features)
    expected = set(RAW_FUSED_INPUT_QUALITY_NAMES)
    missing = sorted(expected - received)
    extra = sorted(received - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError(
            "input quality feature schema mismatch: " + ", ".join(details)
        )
    values = []
    for name in RAW_FUSED_INPUT_QUALITY_NAMES:
        value = features[name]
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
            raise TypeError(f"input quality feature {name!r} must be scalar")
        values.append(float(value))
    vector = np.asarray(values, dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("input quality features must be finite")
    if np.any((vector < 0.0) | (vector > 1.0)):
        raise ValueError("input quality features must lie in [0, 1]")
    return _read_only(vector, dtype=np.float32)


def _quality_matrix(features: Any, count: int, source: str) -> np.ndarray:
    if count == 0:
        if features is None:
            return np.empty(
                (0, RAW_FUSED_INPUT_QUALITY_DIM), dtype=np.float64
            )
        values = np.asarray(features)
        if values.size != 0:
            raise ValueError(
                f"{source} has quality rows but no candidate boxes"
            )
        return np.empty(
            (0, RAW_FUSED_INPUT_QUALITY_DIM), dtype=np.float64
        )
    if features is None:
        raise ValueError(f"missing quality features for {source}")
    if isinstance(features, Mapping):
        if count != 1:
            raise ValueError(
                f"{source} mapping quality is valid only for one candidate"
            )
        matrix = raw_fused_input_quality_vector(features)[None, :]
    elif (
        isinstance(features, Sequence)
        and not isinstance(features, (str, bytes, np.ndarray))
        and all(isinstance(item, Mapping) for item in features)
    ):
        if len(features) != count:
            raise ValueError(
                f"{source} quality row count must match candidate count"
            )
        matrix = np.stack(
            [raw_fused_input_quality_vector(item) for item in features]
        )
    else:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim == 1:
            if count != 1:
                raise ValueError(
                    f"{source} one-dimensional quality requires one candidate"
                )
            matrix = matrix[None, :]
    if matrix.shape != (count, RAW_FUSED_INPUT_QUALITY_DIM):
        raise ValueError(
            f"{source} quality features must have shape "
            f"[{count}, {RAW_FUSED_INPUT_QUALITY_DIM}]"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"{source} quality features must be finite")
    if np.any((matrix < 0.0) | (matrix > 1.0)):
        raise ValueError(
            f"{source} quality features must lie in [0, 1]"
        )
    return np.asarray(matrix, dtype=np.float64)


def _resolve_six_d_formats(
    value: Union[str, Mapping[str, str]],
) -> Mapping[str, str]:
    if isinstance(value, str):
        formats = {source: value for source in RAW_FUSED_QUERY_SOURCES}
    elif isinstance(value, Mapping):
        unknown = sorted(set(value) - set(RAW_FUSED_QUERY_SOURCES))
        if unknown:
            raise ValueError(
                "unknown six_d_format source(s): " + ", ".join(unknown)
            )
        formats = {
            source: value.get(source, "center_size")
            for source in RAW_FUSED_QUERY_SOURCES
        }
    else:
        raise TypeError("six_d_format must be a string or source mapping")
    normalized = {}
    for source, box_format in formats.items():
        if not isinstance(box_format, str):
            raise TypeError(f"six_d_format for {source} must be a string")
        box_format = box_format.strip().lower()
        if box_format not in {"center_size", "min_max"}:
            raise ValueError(
                "six_d_format must be 'center_size' or 'min_max'"
            )
        normalized[source] = box_format
    return normalized


def _six_d_to_corners(boxes: np.ndarray, box_format: str) -> np.ndarray:
    if box_format == "center_size":
        centers = boxes[:, :3]
        sizes = boxes[:, 3:6]
        if np.any(sizes <= 0.0):
            raise ValueError("center_size boxes require positive dimensions")
    else:
        lower = boxes[:, :3]
        upper = boxes[:, 3:6]
        sizes = upper - lower
        if np.any(sizes <= 0.0):
            raise ValueError("min_max boxes require upper > lower")
        centers = 0.5 * (lower + upper)
    return (
        centers[:, None, :]
        + 0.5 * sizes[:, None, :] * _CORNER_SIGNS[None, :, :]
    )


def _canonicalize_boxes(
    boxes: Any,
    *,
    source: str,
    six_d_format: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if boxes is None:
        corners = np.empty((0, 8, 3), dtype=np.float64)
    else:
        values = np.asarray(boxes)
        if values.dtype.kind not in "fiu":
            raise TypeError(f"{source} candidate boxes must be numeric")
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            if values.ndim not in {1, 2, 3}:
                raise ValueError(f"malformed empty {source} candidate boxes")
            corners = np.empty((0, 8, 3), dtype=np.float64)
        elif values.shape == (6,):
            corners = _six_d_to_corners(
                values[None, :], six_d_format
            )
        elif values.ndim == 2 and values.shape[1] == 6:
            corners = _six_d_to_corners(values, six_d_format)
        elif values.shape == (8, 3):
            corners = values[None, :, :].copy()
        elif values.ndim == 3 and values.shape[1:] == (8, 3):
            corners = values.copy()
        else:
            raise ValueError(
                f"{source} boxes must have shape [6], [N,6], [8,3], "
                "or [N,8,3]"
            )
    if not np.isfinite(corners).all():
        raise ValueError(f"{source} candidate boxes must be finite")
    if len(corners) == 0:
        return (
            corners,
            np.empty((0, 6), dtype=np.float64),
            np.empty((0, 6), dtype=np.float64),
        )
    lower = corners.min(axis=1)
    upper = corners.max(axis=1)
    sizes = upper - lower
    if np.any(sizes <= 0.0):
        raise ValueError(
            f"{source} corner boxes require positive AABB extent"
        )
    centers = 0.5 * (lower + upper)
    center_sizes = np.concatenate((centers, sizes), axis=1)
    aabbs = np.concatenate((lower, upper), axis=1)
    return corners, center_sizes, aabbs


@dataclass(frozen=True)
class RawFusedCandidateTable:
    """Immutable, source-preserving canonical candidate table."""

    schema: str
    candidate_ids: Tuple[str, ...]
    sources: Tuple[str, ...]
    source_indices: np.ndarray
    corners: np.ndarray
    center_sizes: np.ndarray
    aabbs: np.ndarray
    quality_feature_names: Tuple[str, ...]
    quality_features: np.ndarray

    def __len__(self) -> int:
        return len(self.candidate_ids)


@dataclass(frozen=True)
class RawFusedPairwiseConsensus:
    """Read-only pairwise geometry matrices."""

    schema: str
    iou_3d: np.ndarray
    center_similarity: np.ndarray
    extent_similarity: np.ndarray
    consensus: np.ndarray


@dataclass(frozen=True)
class SelectedObserverCandidate:
    """One diagnostic selection; ``applied`` is permanently false."""

    index: int
    candidate_id: str
    source: str
    source_index: int
    corners: np.ndarray
    center_size: np.ndarray
    feature_vector: np.ndarray
    heuristic_score: float
    learned_score: Optional[float]
    selection_score: float
    selection_mode: str
    observer_only: bool = True
    applied: bool = False


@dataclass(frozen=True)
class RawFusedQueryObservation:
    """Complete immutable observer output."""

    schema: str
    candidate_table: RawFusedCandidateTable
    pairwise_consensus: RawFusedPairwiseConsensus
    feature_names: Tuple[str, ...]
    features: np.ndarray
    heuristic_scores: np.ndarray
    learned_scores: np.ndarray
    selection_scores: np.ndarray
    selected: SelectedObserverCandidate
    selection_mode: str
    learned_scorer_used: bool
    scorer_model_type: str
    scorer_checkpoint: Optional[str]
    observer_only: bool = True
    mutation_enabled: bool = False


def _pairwise_geometry(
    center_sizes: np.ndarray,
    aabbs: np.ndarray,
) -> RawFusedPairwiseConsensus:
    count = len(center_sizes)
    lower = aabbs[:, :3]
    upper = aabbs[:, 3:6]
    sizes = center_sizes[:, 3:6]
    centers = center_sizes[:, :3]

    intersection_lower = np.maximum(lower[:, None, :], lower[None, :, :])
    intersection_upper = np.minimum(upper[:, None, :], upper[None, :, :])
    intersection_size = np.maximum(
        intersection_upper - intersection_lower, 0.0
    )
    intersection = np.prod(intersection_size, axis=2)
    volumes = np.prod(sizes, axis=1)
    union = volumes[:, None] + volumes[None, :] - intersection
    iou = np.divide(
        intersection,
        union,
        out=np.zeros((count, count), dtype=np.float64),
        where=union > 0.0,
    )

    center_distance = np.linalg.norm(
        centers[:, None, :] - centers[None, :, :], axis=2
    )
    diagonals = np.linalg.norm(sizes, axis=1)
    center_scale = 0.5 * (
        diagonals[:, None] + diagonals[None, :]
    )
    center_similarity = np.clip(
        1.0
        - np.divide(
            center_distance,
            center_scale,
            out=np.zeros_like(center_distance),
            where=center_scale > 0.0,
        ),
        0.0,
        1.0,
    )

    log_extent_ratio = np.abs(
        np.log(sizes[:, None, :] / sizes[None, :, :])
    )
    extent_similarity = np.exp(-np.mean(log_extent_ratio, axis=2))
    consensus = (
        0.60 * iou
        + 0.25 * center_similarity
        + 0.15 * center_similarity * extent_similarity
    )
    for matrix in (iou, center_similarity, extent_similarity, consensus):
        np.fill_diagonal(matrix, 1.0)
    return RawFusedPairwiseConsensus(
        schema=RAW_FUSED_PAIRWISE_SCHEMA,
        iou_3d=_read_only(iou, dtype=np.float32),
        center_similarity=_read_only(
            center_similarity, dtype=np.float32
        ),
        extent_similarity=_read_only(
            extent_similarity, dtype=np.float32
        ),
        consensus=_read_only(consensus, dtype=np.float32),
    )


def _masked_mean_max(
    values: np.ndarray,
    mask: np.ndarray,
) -> Tuple[float, float]:
    selected = values[mask]
    if selected.size == 0:
        return 0.0, 0.0
    return float(selected.mean()), float(selected.max())


def _candidate_features(
    table: RawFusedCandidateTable,
    pairwise: RawFusedPairwiseConsensus,
) -> np.ndarray:
    count = len(table)
    output = np.zeros(
        (count, RAW_FUSED_QUERY_FEATURE_DIM), dtype=np.float64
    )
    source_array = np.asarray(table.sources, dtype=np.str_)
    volumes = np.prod(table.center_sizes[:, 3:6], axis=1)
    original_mask = source_array == "original"

    for index, source in enumerate(table.sources):
        other = np.ones(count, dtype=bool)
        other[index] = False
        cross = other & (source_array != source)
        one_hot = np.zeros(len(RAW_FUSED_QUERY_SOURCES))
        one_hot[_SOURCE_TO_INDEX[source]] = 1.0
        quality = np.asarray(table.quality_features[index], dtype=np.float64)

        mean_iou, max_iou = _masked_mean_max(
            pairwise.iou_3d[index], other
        )
        mean_center, max_center = _masked_mean_max(
            pairwise.center_similarity[index], other
        )
        mean_extent, max_extent = _masked_mean_max(
            pairwise.extent_similarity[index], other
        )
        mean_consensus, max_consensus = _masked_mean_max(
            pairwise.consensus[index], other
        )
        mean_cross, max_cross = _masked_mean_max(
            pairwise.consensus[index], cross
        )
        cross_values = pairwise.consensus[index, cross]
        support_025 = (
            float(np.mean(cross_values >= 0.25))
            if cross_values.size
            else 0.0
        )
        support_050 = (
            float(np.mean(cross_values >= 0.50))
            if cross_values.size
            else 0.0
        )

        supported_sources = 0
        source_maxima = []
        for candidate_source in RAW_FUSED_QUERY_SOURCES:
            source_mask = other & (source_array == candidate_source)
            _, source_max = _masked_mean_max(
                pairwise.consensus[index], source_mask
            )
            source_maxima.append(source_max)
            if candidate_source != source and source_max >= 0.25:
                supported_sources += 1
        source_support_fraction = supported_sources / float(
            len(RAW_FUSED_QUERY_SOURCES) - 1
        )

        if source == "original":
            volume_similarity = 1.0
        elif np.any(original_mask):
            ratios = volumes[index] / volumes[original_mask]
            volume_similarity = float(
                np.max(np.exp(-np.abs(np.log(ratios))))
            )
        else:
            volume_similarity = 0.0
        consensus_margin = max(0.0, max_cross - mean_cross)

        output[index] = np.asarray(
            [
                *one_hot,
                *quality,
                float(quality.mean()),
                float(quality.min()),
                mean_iou,
                max_iou,
                mean_center,
                max_center,
                mean_extent,
                max_extent,
                mean_consensus,
                max_consensus,
                mean_cross,
                max_cross,
                support_025,
                support_050,
                source_support_fraction,
                *source_maxima,
                volume_similarity,
                consensus_margin,
            ],
            dtype=np.float64,
        )
    if not np.isfinite(output).all():
        raise RuntimeError("raw/fused query features must be finite")
    if np.any((output < 0.0) | (output > 1.0)):
        raise RuntimeError("raw/fused query features must lie in [0, 1]")
    return output


_HEURISTIC_WEIGHTS = np.zeros(
    RAW_FUSED_QUERY_FEATURE_DIM, dtype=np.float64
)
for _name, _weight in {
    "proposal_score": 0.12,
    "mask_quality": 0.05,
    "depth_quality": 0.10,
    "geometry_quality": 0.12,
    "view_quality": 0.06,
    "quality_mean": 0.08,
    "quality_min": 0.03,
    "mean_pairwise_consensus": 0.05,
    "max_pairwise_consensus": 0.05,
    "mean_cross_source_consensus": 0.10,
    "max_cross_source_consensus": 0.10,
    "cross_source_support_025": 0.03,
    "cross_source_support_050": 0.03,
    "source_support_fraction": 0.04,
    "original_max_consensus": 0.02,
    "volume_similarity_to_original": 0.02,
}.items():
    _HEURISTIC_WEIGHTS[RAW_FUSED_QUERY_FEATURE_NAMES.index(_name)] = _weight
assert np.isclose(_HEURISTIC_WEIGHTS.sum(), 1.0)
_HEURISTIC_WEIGHTS.setflags(write=False)


class RawFusedQueryScorer:
    """Strict fixed-schema scorer base class."""

    model_type: str

    def _feature_matrix(self, features: Any) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != (
            RAW_FUSED_QUERY_FEATURE_DIM
        ):
            raise ValueError(
                "raw/fused query scorer features must have shape "
                f"[N, {RAW_FUSED_QUERY_FEATURE_DIM}]"
            )
        if not np.isfinite(values).all():
            raise ValueError("raw/fused query scorer features must be finite")
        return values

    def score(self, features: Any) -> np.ndarray:
        raise NotImplementedError

    __call__ = score


class LinearRawFusedQueryScorer(RawFusedQueryScorer):
    """Sigmoid linear scorer loaded from a strict NPZ checkpoint."""

    model_type = "linear"

    def __init__(self, weight: np.ndarray, bias: float) -> None:
        weight = np.asarray(weight, dtype=np.float64)
        if weight.shape != (RAW_FUSED_QUERY_FEATURE_DIM,):
            raise ValueError(
                "linear raw/fused query weight has the wrong shape"
            )
        if not np.isfinite(weight).all():
            raise ValueError("linear raw/fused query weight must be finite")
        self.weight = _read_only(weight, dtype=np.float64)
        self.bias = _float_scalar("linear scorer bias", bias)

    def score(self, features: Any) -> np.ndarray:
        matrix = self._feature_matrix(features)
        return _read_only(
            _stable_sigmoid(matrix @ self.weight + self.bias),
            dtype=np.float32,
        )

    __call__ = score


class MLPRawFusedQueryScorer(RawFusedQueryScorer):
    """ReLU MLP with a sigmoid scalar output."""

    model_type = "mlp"

    def __init__(
        self,
        weights: Sequence[np.ndarray],
        biases: Sequence[np.ndarray],
    ) -> None:
        if (
            not isinstance(weights, Sequence)
            or isinstance(weights, (str, bytes))
            or not isinstance(biases, Sequence)
            or isinstance(biases, (str, bytes))
        ):
            raise TypeError("MLP weights and biases must be sequences")
        if len(weights) == 0 or len(weights) != len(biases):
            raise ValueError(
                "MLP requires matching non-empty weights and biases"
            )
        validated_weights = []
        validated_biases = []
        input_dim = RAW_FUSED_QUERY_FEATURE_DIM
        for index, (weight, bias) in enumerate(zip(weights, biases)):
            weight = np.asarray(weight, dtype=np.float64)
            bias = np.asarray(bias, dtype=np.float64)
            if (
                weight.ndim != 2
                or weight.shape[0] != input_dim
                or weight.shape[1] < 1
            ):
                raise ValueError(
                    f"weight_{index} must have shape "
                    f"[{input_dim}, output_dim]"
                )
            if bias.shape != (weight.shape[1],):
                raise ValueError(
                    f"bias_{index} must have shape [{weight.shape[1]}]"
                )
            if not np.isfinite(weight).all() or not np.isfinite(bias).all():
                raise ValueError("MLP weights and biases must be finite")
            validated_weights.append(
                _read_only(weight, dtype=np.float64)
            )
            validated_biases.append(_read_only(bias, dtype=np.float64))
            input_dim = int(weight.shape[1])
        if input_dim != 1:
            raise ValueError("final MLP layer must have one output")
        self.weights = tuple(validated_weights)
        self.biases = tuple(validated_biases)

    def score(self, features: Any) -> np.ndarray:
        activations = self._feature_matrix(features)
        for index, (weight, bias) in enumerate(
            zip(self.weights, self.biases)
        ):
            activations = activations @ weight + bias
            if index + 1 < len(self.weights):
                activations = np.maximum(activations, 0.0)
        return _read_only(
            _stable_sigmoid(activations[:, 0]), dtype=np.float32
        )

    __call__ = score


def load_raw_fused_query_scorer(
    checkpoint_path: Union[str, Path],
) -> RawFusedQueryScorer:
    """Load a strict pickle-free linear or MLP scorer checkpoint."""

    path = Path(checkpoint_path)
    if path.suffix.lower() != ".npz":
        raise ValueError("raw/fused query checkpoint must end in .npz")
    if not path.is_file():
        raise FileNotFoundError(
            f"raw/fused query checkpoint not found: {path}"
        )
    with np.load(path, allow_pickle=False) as checkpoint:
        keys = set(checkpoint.files)
        common = {
            "schema",
            "format_version",
            "model_type",
            "feature_names",
        }
        if not common <= keys:
            raise ValueError(
                "raw/fused query checkpoint is missing required metadata"
            )
        schema = _string_scalar("schema", checkpoint["schema"])
        if schema != RAW_FUSED_QUERY_SCORER_SCHEMA:
            raise ValueError("raw/fused query checkpoint schema mismatch")
        version = _integer_scalar(
            "format_version", checkpoint["format_version"]
        )
        if version != RAW_FUSED_QUERY_SCORER_FORMAT_VERSION:
            raise ValueError(
                "unsupported raw/fused query checkpoint format version"
            )
        model_type = _string_scalar(
            "model_type", checkpoint["model_type"]
        ).strip().lower()
        feature_names = tuple(
            str(value)
            for value in np.asarray(checkpoint["feature_names"]).tolist()
        )
        if feature_names != RAW_FUSED_QUERY_FEATURE_NAMES:
            raise ValueError(
                "raw/fused query checkpoint feature schema/order mismatch"
            )

        if model_type == "linear":
            expected = common | {"weight", "bias"}
            if keys != expected:
                raise ValueError(
                    "linear raw/fused query checkpoint keys must be exact"
                )
            return LinearRawFusedQueryScorer(
                checkpoint["weight"], checkpoint["bias"]
            )
        if model_type != "mlp":
            raise ValueError(
                "raw/fused query model_type must be linear or mlp"
            )
        if "num_layers" not in keys:
            raise ValueError("MLP checkpoint is missing num_layers")
        layer_count = _integer_scalar(
            "num_layers", checkpoint["num_layers"]
        )
        if layer_count < 1:
            raise ValueError("num_layers must be positive")
        expected = common | {"num_layers"}
        expected.update(
            f"weight_{index}" for index in range(layer_count)
        )
        expected.update(
            f"bias_{index}" for index in range(layer_count)
        )
        if keys != expected:
            raise ValueError("MLP checkpoint layer keys must be exact")
        return MLPRawFusedQueryScorer(
            [
                checkpoint[f"weight_{index}"]
                for index in range(layer_count)
            ],
            [
                checkpoint[f"bias_{index}"]
                for index in range(layer_count)
            ],
        )


def _candidate_table(
    *,
    boxes_by_source: Mapping[str, Any],
    quality_by_source: Mapping[str, Any],
    six_d_formats: Mapping[str, str],
) -> RawFusedCandidateTable:
    corners_rows = []
    center_size_rows = []
    aabb_rows = []
    quality_rows = []
    sources = []
    source_indices = []
    candidate_ids = []
    for source in RAW_FUSED_QUERY_SOURCES:
        corners, center_sizes, aabbs = _canonicalize_boxes(
            boxes_by_source[source],
            source=source,
            six_d_format=six_d_formats[source],
        )
        quality = _quality_matrix(
            quality_by_source.get(source), len(corners), source
        )
        for source_index in range(len(corners)):
            corners_rows.append(corners[source_index])
            center_size_rows.append(center_sizes[source_index])
            aabb_rows.append(aabbs[source_index])
            quality_rows.append(quality[source_index])
            sources.append(source)
            source_indices.append(source_index)
            candidate_ids.append(f"{source}:{source_index}")
    if not any(source == "original" for source in sources):
        raise ValueError("at least one original candidate is required")
    count = len(sources)
    return RawFusedCandidateTable(
        schema=RAW_FUSED_CANDIDATE_TABLE_SCHEMA,
        candidate_ids=tuple(candidate_ids),
        sources=tuple(sources),
        source_indices=_read_only(source_indices, dtype=np.int64),
        corners=_read_only(
            np.stack(corners_rows), dtype=np.float32
        ),
        center_sizes=_read_only(
            np.stack(center_size_rows), dtype=np.float32
        ),
        aabbs=_read_only(np.stack(aabb_rows), dtype=np.float32),
        quality_feature_names=RAW_FUSED_INPUT_QUALITY_NAMES,
        quality_features=_read_only(
            np.stack(quality_rows).reshape(
                count, RAW_FUSED_INPUT_QUALITY_DIM
            ),
            dtype=np.float32,
        ),
    )


def _select_index(
    table: RawFusedCandidateTable,
    features: np.ndarray,
    selection_scores: np.ndarray,
    heuristic_scores: np.ndarray,
) -> int:
    max_cross_index = RAW_FUSED_QUERY_FEATURE_NAMES.index(
        "max_cross_source_consensus"
    )
    quality_index = RAW_FUSED_QUERY_FEATURE_NAMES.index("quality_mean")
    order = sorted(
        range(len(table)),
        key=lambda index: (
            -float(selection_scores[index]),
            -float(heuristic_scores[index]),
            -float(features[index, max_cross_index]),
            -float(features[index, quality_index]),
            _SOURCE_TO_INDEX[table.sources[index]],
            tuple(
                np.round(table.center_sizes[index], decimals=9).tolist()
            ),
            int(table.source_indices[index]),
        ),
    )
    return int(order[0])


def observe_raw_fused_query(
    *,
    original: Any,
    raw_mask: Any = None,
    superpoint: Any = None,
    occupancy: Any = None,
    quality_features: Mapping[str, Any],
    six_d_format: Union[
        str, Mapping[str, str]
    ] = "center_size",
    scorer_checkpoint: Optional[Union[str, Path]] = None,
) -> RawFusedQueryObservation:
    """Build a deterministic, observer-only raw/fused query table.

    ``quality_features`` maps each non-empty source to either a ``[N, 5]``
    array, one exact-schema mapping for a single candidate, or a sequence of
    exact-schema mappings.  Six-dimensional boxes default to
    ``[center_x, center_y, center_z, size_x, size_y, size_z]``.  Explicit
    corner inputs have shape ``[N, 8, 3]``.

    The returned ``selected`` candidate is diagnostic only.  This function
    neither accepts nor returns mutable final detector boxes.
    """

    if not isinstance(quality_features, Mapping):
        raise TypeError("quality_features must be a source mapping")
    unknown_quality_sources = sorted(
        set(quality_features) - set(RAW_FUSED_QUERY_SOURCES)
    )
    if unknown_quality_sources:
        raise ValueError(
            "unknown quality source(s): "
            + ", ".join(unknown_quality_sources)
        )
    formats = _resolve_six_d_formats(six_d_format)
    table = _candidate_table(
        boxes_by_source={
            "original": original,
            "raw_mask": raw_mask,
            "superpoint": superpoint,
            "occupancy": occupancy,
        },
        quality_by_source=quality_features,
        six_d_formats=formats,
    )
    pairwise = _pairwise_geometry(table.center_sizes, table.aabbs)
    features = _candidate_features(table, pairwise)
    heuristic_scores = np.clip(
        features @ _HEURISTIC_WEIGHTS, 0.0, 1.0
    )

    if scorer_checkpoint is None:
        learned_scores = np.full(len(table), np.nan, dtype=np.float64)
        selection_scores = heuristic_scores
        selection_mode = "deterministic_heuristic"
        learned_scorer_used = False
        scorer_model_type = "none"
        checkpoint_value = None
    else:
        scorer = load_raw_fused_query_scorer(scorer_checkpoint)
        learned_scores = np.asarray(
            scorer.score(features), dtype=np.float64
        )
        if learned_scores.shape != (len(table),):
            raise RuntimeError(
                "raw/fused query scorer returned the wrong shape"
            )
        if (
            not np.isfinite(learned_scores).all()
            or np.any((learned_scores < 0.0) | (learned_scores > 1.0))
        ):
            raise RuntimeError(
                "raw/fused query scorer returned invalid scores"
            )
        selection_scores = learned_scores
        selection_mode = f"learned_{scorer.model_type}_npz"
        learned_scorer_used = True
        scorer_model_type = scorer.model_type
        checkpoint_value = str(Path(scorer_checkpoint).resolve())

    selected_index = _select_index(
        table,
        features,
        selection_scores,
        heuristic_scores,
    )
    selected_learned_score = (
        float(learned_scores[selected_index])
        if learned_scorer_used
        else None
    )
    selected = SelectedObserverCandidate(
        index=selected_index,
        candidate_id=table.candidate_ids[selected_index],
        source=table.sources[selected_index],
        source_index=int(table.source_indices[selected_index]),
        corners=_read_only(
            table.corners[selected_index], dtype=np.float32
        ),
        center_size=_read_only(
            table.center_sizes[selected_index], dtype=np.float32
        ),
        feature_vector=_read_only(
            features[selected_index], dtype=np.float32
        ),
        heuristic_score=float(heuristic_scores[selected_index]),
        learned_score=selected_learned_score,
        selection_score=float(selection_scores[selected_index]),
        selection_mode=selection_mode,
    )
    return RawFusedQueryObservation(
        schema=RAW_FUSED_QUERY_SCHEMA,
        candidate_table=table,
        pairwise_consensus=pairwise,
        feature_names=RAW_FUSED_QUERY_FEATURE_NAMES,
        features=_read_only(features, dtype=np.float32),
        heuristic_scores=_read_only(
            heuristic_scores, dtype=np.float32
        ),
        learned_scores=_read_only(learned_scores, dtype=np.float32),
        selection_scores=_read_only(
            selection_scores, dtype=np.float32
        ),
        selected=selected,
        selection_mode=selection_mode,
        learned_scorer_used=learned_scorer_used,
        scorer_model_type=scorer_model_type,
        scorer_checkpoint=checkpoint_value,
    )


__all__ = [
    "LinearRawFusedQueryScorer",
    "MLPRawFusedQueryScorer",
    "RAW_FUSED_CANDIDATE_TABLE_SCHEMA",
    "RAW_FUSED_INPUT_QUALITY_DIM",
    "RAW_FUSED_INPUT_QUALITY_NAMES",
    "RAW_FUSED_PAIRWISE_SCHEMA",
    "RAW_FUSED_QUERY_FEATURE_DIM",
    "RAW_FUSED_QUERY_FEATURE_NAMES",
    "RAW_FUSED_QUERY_SCHEMA",
    "RAW_FUSED_QUERY_SCORER_FORMAT_VERSION",
    "RAW_FUSED_QUERY_SCORER_SCHEMA",
    "RAW_FUSED_QUERY_SOURCES",
    "RawFusedCandidateTable",
    "RawFusedPairwiseConsensus",
    "RawFusedQueryObservation",
    "RawFusedQueryScorer",
    "SelectedObserverCandidate",
    "load_raw_fused_query_scorer",
    "observe_raw_fused_query",
    "raw_fused_input_quality_vector",
]
