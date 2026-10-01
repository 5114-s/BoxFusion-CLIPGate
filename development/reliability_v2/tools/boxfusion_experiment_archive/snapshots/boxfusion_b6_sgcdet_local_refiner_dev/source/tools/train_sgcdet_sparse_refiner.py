#!/usr/bin/env python3
"""Train the SGCDet-inspired local sparse refiner on train-only K=5 data.

The split unit is an entire ScanNet scene.  Occupancy is learned on the fixed
coarse/fine local volumes, geometry regression is enabled only for strict B5
geometry positives, and quality/uncertainty targets describe the candidate
actually produced by the current residual.  AP50 crossing and preservation
terms are therefore evaluator-aligned without leaking validation scenes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as error:  # pragma: no cover
    raise ImportError("sparse-refiner training requires PyTorch") from error

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from boxfusion.quality_score import QUALITY_FEATURE_NAMES
from boxfusion.sgcdet_local_sparse_refiner import (
    SGCDET_SPARSE_REFINER_REFERENCE,
    SGCDetInspiredLocalSparseRefiner,
    SGCDetLocalSparseRefinerConfig,
    load_sgcdet_sparse_refiner_checkpoint,
    make_sgcdet_sparse_refiner_checkpoint,
)
from tools.build_sgcdet_sparse_refiner_dataset import (
    DATASET_FORMAT_VERSION,
    DATASET_SCHEMA,
    INPUT_SCHEMA,
    METADATA_KEYS,
    OBJECTIVE,
    SAMPLE_KEYS,
)
from tools.train_oriented_box_refiner import (
    balanced_epoch_indices,
    deterministic_scene_split,
    differentiable_aligned_aabb_iou,
)


SCENE_PATTERN = re.compile(r"scene\d{4}_\d{2}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
AP50_THRESHOLD = 0.50
AP50_LOSS_TARGET = 0.5001


@dataclass(frozen=True)
class SparseRefinerTrainingData:
    points_local: np.ndarray
    point_mask: np.ndarray
    view_features: np.ndarray
    view_mask: np.ndarray
    local_boxes: np.ndarray
    quality_features: np.ndarray
    target_residual: np.ndarray
    geometry_mask: np.ndarray
    scene_ids: np.ndarray
    result_indices: np.ndarray
    matched_gt_index: np.ndarray
    baseline_iou: np.ndarray
    target_iou: np.ndarray
    iou_gain: np.ndarray
    cross_iou50: np.ndarray
    ap50_weight: np.ndarray
    runtime_eligible: np.ndarray
    identity_tp50: np.ndarray
    aligned_basis: np.ndarray
    original_aligned_center: np.ndarray
    matched_gt_box: np.ndarray
    max_center_fraction: float
    max_log_dimension_residual: float
    source_joint_dataset_sha256: str
    source_b5_dataset_sha256: str
    forbidden_scene_count: int
    forbidden_scene_sha256: str
    training_scene_sha256: str
    points_per_view: int

    @property
    def sample_count(self) -> int:
        return int(self.points_local.shape[0])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scene_sha256(scene_ids: Sequence[str]) -> str:
    payload = "\n".join(sorted(set(scene_ids))) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _scalar_string(value: Any, name: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.hasobject:
        raise TypeError(f"{name} must be a safe scalar string")
    item = array.item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    if not isinstance(item, str):
        raise TypeError(f"{name} must be a string")
    return item


def _scalar_integer(value: Any, name: str) -> int:
    array = np.asarray(value)
    if (
        array.ndim != 0
        or array.dtype == np.bool_
        or not np.issubdtype(array.dtype, np.integer)
    ):
        raise TypeError(f"{name} must be an integer scalar")
    return int(array)


def _scalar_float(value: Any, name: str) -> float:
    array = np.asarray(value)
    if (
        array.ndim != 0
        or array.dtype == np.bool_
        or not np.issubdtype(array.dtype, np.number)
    ):
        raise TypeError(f"{name} must be a numeric scalar")
    result = float(array)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _safe_load_archive(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file() or path.suffix.lower() != ".npz":
        raise FileNotFoundError(f"sparse-refiner dataset is absent: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            expected = SAMPLE_KEYS | METADATA_KEYS
            keys = set(archive.files)
            if keys != expected:
                raise ValueError(
                    "sparse-refiner dataset keys are invalid: "
                    f"missing={sorted(expected - keys)}, "
                    f"unexpected={sorted(keys - expected)}"
                )
            arrays = {
                name: np.asarray(archive[name]).copy()
                for name in archive.files
            }
    except ValueError as error:
        if "Object arrays cannot be loaded" in str(error):
            raise ValueError("dataset contains forbidden object arrays") from error
        raise
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("dataset contains forbidden object dtype")
    return arrays


def load_sgcdet_sparse_refiner_dataset(
    path: str | os.PathLike[str],
) -> SparseRefinerTrainingData:
    arrays = _safe_load_archive(Path(path))
    if _scalar_string(arrays["schema"], "schema") != DATASET_SCHEMA:
        raise ValueError("sparse-refiner dataset schema mismatch")
    if _scalar_integer(arrays["format_version"], "format_version") != DATASET_FORMAT_VERSION:
        raise ValueError("sparse-refiner dataset format version mismatch")
    if _scalar_string(arrays["input_schema"], "input_schema") != INPUT_SCHEMA:
        raise ValueError("sparse-refiner input schema mismatch")
    if _scalar_string(arrays["objective"], "objective") != OBJECTIVE:
        raise ValueError("sparse-refiner objective mismatch")
    if _scalar_integer(arrays["top_k_views"], "top_k_views") != 5:
        raise ValueError("sparse-refiner dataset requires K=5")
    points_per_view = _scalar_integer(
        arrays["points_per_view"], "points_per_view"
    )
    if points_per_view != 128:
        raise ValueError("sparse-refiner dataset requires P=128")
    strict = np.asarray(arrays["strict_k5_diagnostics"])
    if strict.ndim != 0 or strict.dtype != np.bool_ or not bool(strict):
        raise ValueError("dataset is not from strict K=5 diagnostics")

    points = arrays["points_local"]
    if (
        points.dtype != np.float32
        or points.ndim != 4
        or points.shape[1:] != (5, 128, 3)
        or points.shape[0] < 2
    ):
        raise TypeError("points_local must be float32 [N,5,128,3]")
    n = int(points.shape[0])
    point_mask = arrays["point_mask"]
    view_mask = arrays["view_mask"]
    if point_mask.shape != (n, 5, 128) or point_mask.dtype != np.bool_:
        raise TypeError("point_mask must be Boolean [N,5,128]")
    if view_mask.shape != (n, 5) or view_mask.dtype != np.bool_:
        raise TypeError("view_mask must be Boolean [N,5]")
    if not np.array_equal(view_mask, point_mask.any(axis=2)):
        raise ValueError("point/view masks disagree")
    if not view_mask.any(axis=1).all():
        raise ValueError("every sample must have a valid view")
    if not np.isfinite(points).all() or not np.all(points[~point_mask] == 0.0):
        raise ValueError("points or masked padding are invalid")

    float_shapes = {
        "view_features": (n, 5, 9),
        "local_boxes": (n, 6),
        "quality_features": (n, len(QUALITY_FEATURE_NAMES)),
        "target_residual": (n, 6),
        "aligned_basis": (n, 3, 3),
        "original_aligned_center": (n, 3),
        "matched_gt_box": (n, 6),
    }
    for name, shape in float_shapes.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != np.float32:
            raise TypeError(f"{name} must be float32 with shape {shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    if not np.allclose(arrays["local_boxes"][:, :3], 0.0, atol=1e-7, rtol=0.0):
        raise ValueError("local box centers must be canonical zero")
    if (arrays["local_boxes"][:, 3:6] <= 0.0).any():
        raise ValueError("local box dimensions must be positive")
    if (arrays["quality_features"] < 0.0).any() or (
        arrays["quality_features"] > 1.0
    ).any():
        raise ValueError("quality features must lie in [0,1]")
    if (
        arrays["view_features"] < 0.0
    ).any() or (arrays["view_features"] > 1.0).any():
        raise ValueError("view features must lie in [0,1]")
    if not np.all(arrays["view_features"][~view_mask] == 0.0):
        raise ValueError("masked view features must be zero")

    bool_names = (
        "geometry_mask",
        "cross_iou50",
        "runtime_eligible",
        "identity_tp50",
        "candidate_oracle_tp50",
    )
    for name in bool_names:
        value = arrays[name]
        if value.shape != (n,) or value.dtype != np.bool_:
            raise TypeError(f"{name} must be Boolean [N]")
    if not arrays["geometry_mask"].any() or arrays["geometry_mask"].all():
        raise ValueError("dataset must contain geometry positives and negatives")
    expected_cross = (
        arrays["runtime_eligible"]
        & ~arrays["identity_tp50"]
        & arrays["candidate_oracle_tp50"]
    )
    if not np.array_equal(arrays["cross_iou50"], expected_cross):
        raise ValueError("cross_iou50 provenance is invalid")

    for name in ("baseline_iou", "target_iou", "iou_gain", "ap50_weight"):
        value = arrays[name]
        if value.shape != (n,) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise TypeError(f"{name} must be finite float32 [N]")
    for name in ("baseline_iou", "target_iou"):
        if (arrays[name] < 0.0).any() or (arrays[name] > 1.0).any():
            raise ValueError(f"{name} must lie in [0,1]")
    expected_gain = np.maximum(
        arrays["target_iou"] - arrays["baseline_iou"], 0.0
    )
    if not np.allclose(arrays["iou_gain"], expected_gain, atol=2e-5, rtol=0.0):
        raise ValueError("iou_gain disagrees with baseline/target IoU")
    if (arrays["ap50_weight"] < 1.0).any():
        raise ValueError("ap50 weights must be at least one")

    for name in ("result_indices", "track_ids", "matched_gt_index"):
        value = arrays[name]
        if value.shape != (n,) or value.dtype != np.int64:
            raise TypeError(f"{name} must be int64 [N]")
    if (arrays["matched_gt_index"] < -1).any():
        raise ValueError("matched_gt_index must be at least -1")

    scene_ids = arrays["scene_ids"]
    if scene_ids.shape != (n,) or scene_ids.dtype.hasobject or scene_ids.dtype.kind not in {"U", "S"}:
        raise TypeError("scene_ids must be a safe string array [N]")
    scene_ids = scene_ids.astype(np.str_)
    scenes = sorted(np.unique(scene_ids).tolist())
    if len(scenes) < 2 or any(SCENE_PATTERN.fullmatch(s) is None for s in scenes):
        raise ValueError("training scene ids are invalid")
    if (
        _scalar_integer(arrays["training_scene_count"], "training_scene_count")
        != len(scenes)
        or _scalar_integer(arrays["diagnostic_scene_count"], "diagnostic_scene_count")
        != len(scenes)
        or _scalar_string(arrays["training_scene_sha256"], "training_scene_sha256")
        != _scene_sha256(scenes)
        or _scalar_string(arrays["diagnostic_scene_sha256"], "diagnostic_scene_sha256")
        != _scene_sha256(scenes)
    ):
        raise ValueError("training/diagnostic scene provenance is inconsistent")
    for scene in scenes:
        rows = scene_ids == scene
        if len(np.unique(arrays["result_indices"][rows])) != int(rows.sum()):
            raise ValueError(f"{scene}: duplicate result_indices")

    for name in ("source_joint_dataset_sha256", "source_b5_dataset_sha256", "forbidden_scene_sha256"):
        value = _scalar_string(arrays[name], name)
        if SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"{name} is not SHA256")
    max_center = _scalar_float(arrays["max_center_fraction"], "max_center_fraction")
    max_dimension = _scalar_float(
        arrays["max_log_dimension_residual"], "max_log_dimension_residual"
    )
    if (np.abs(arrays["target_residual"][:, :3]) > max_center + 1e-5).any():
        raise ValueError("center target exceeds architecture bound")
    if (np.abs(arrays["target_residual"][:, 3:]) > max_dimension + 1e-5).any():
        raise ValueError("dimension target exceeds architecture bound")

    return SparseRefinerTrainingData(
        points_local=np.ascontiguousarray(points),
        point_mask=np.ascontiguousarray(point_mask),
        view_features=np.ascontiguousarray(arrays["view_features"]),
        view_mask=np.ascontiguousarray(view_mask),
        local_boxes=np.ascontiguousarray(arrays["local_boxes"]),
        quality_features=np.ascontiguousarray(arrays["quality_features"]),
        target_residual=np.ascontiguousarray(arrays["target_residual"]),
        geometry_mask=np.ascontiguousarray(arrays["geometry_mask"]),
        scene_ids=np.ascontiguousarray(scene_ids),
        result_indices=np.ascontiguousarray(arrays["result_indices"]),
        matched_gt_index=np.ascontiguousarray(arrays["matched_gt_index"]),
        baseline_iou=np.ascontiguousarray(arrays["baseline_iou"]),
        target_iou=np.ascontiguousarray(arrays["target_iou"]),
        iou_gain=np.ascontiguousarray(arrays["iou_gain"]),
        cross_iou50=np.ascontiguousarray(arrays["cross_iou50"]),
        ap50_weight=np.ascontiguousarray(arrays["ap50_weight"]),
        runtime_eligible=np.ascontiguousarray(arrays["runtime_eligible"]),
        identity_tp50=np.ascontiguousarray(arrays["identity_tp50"]),
        aligned_basis=np.ascontiguousarray(arrays["aligned_basis"]),
        original_aligned_center=np.ascontiguousarray(arrays["original_aligned_center"]),
        matched_gt_box=np.ascontiguousarray(arrays["matched_gt_box"]),
        max_center_fraction=max_center,
        max_log_dimension_residual=max_dimension,
        source_joint_dataset_sha256=_scalar_string(
            arrays["source_joint_dataset_sha256"], "source_joint_dataset_sha256"
        ),
        source_b5_dataset_sha256=_scalar_string(
            arrays["source_b5_dataset_sha256"], "source_b5_dataset_sha256"
        ),
        forbidden_scene_count=_scalar_integer(
            arrays["forbidden_scene_count"], "forbidden_scene_count"
        ),
        forbidden_scene_sha256=_scalar_string(
            arrays["forbidden_scene_sha256"], "forbidden_scene_sha256"
        ),
        training_scene_sha256=_scalar_string(
            arrays["training_scene_sha256"], "training_scene_sha256"
        ),
        points_per_view=points_per_view,
    )


def deterministic_scene_holdout(
    scene_ids: np.ndarray, validation_fraction: float = 0.2, seed: int = 1337
) -> tuple[np.ndarray, np.ndarray]:
    train, validation = deterministic_scene_split(
        np.asarray(scene_ids), float(validation_fraction), int(seed)
    )
    train_scenes = set(np.asarray(scene_ids)[train].tolist())
    validation_scenes = set(np.asarray(scene_ids)[validation].tolist())
    if not train_scenes or not validation_scenes or train_scenes & validation_scenes:
        raise RuntimeError("scene-level split leaked or produced an empty side")
    return train, validation


class _SparseArrayDataset(Dataset):
    def __init__(self, data: SparseRefinerTrainingData, indices: np.ndarray) -> None:
        indices = np.asarray(indices, dtype=np.int64)
        names = (
            "points_local", "point_mask", "view_features", "view_mask",
            "local_boxes", "quality_features", "target_residual",
            "geometry_mask", "matched_gt_index", "baseline_iou", "target_iou",
            "iou_gain", "cross_iou50", "ap50_weight", "runtime_eligible",
            "identity_tp50", "aligned_basis", "original_aligned_center",
            "matched_gt_box",
        )
        self.values = {
            name: torch.from_numpy(np.ascontiguousarray(getattr(data, name)[indices]))
            for name in names
        }

    def __len__(self) -> int:
        return int(len(self.values["points_local"]))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {name: value[index] for name, value in self.values.items()}


def _weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if mask is not None:
        values = values[mask]
        weights = weights[mask]
    if values.numel() == 0:
        return values.sum() * 0.0
    normalized = weights / weights.mean().clamp_min(torch.finfo(weights.dtype).eps)
    return (values * normalized).mean()


def _balanced_occupancy_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape != targets.shape or targets.dtype != logits.dtype:
        raise TypeError("occupancy logits/targets must be same-shape floating tensors")
    if not torch.all((targets == 0.0) | (targets == 1.0)):
        raise ValueError("occupancy targets must be binary")
    positive = targets.sum(dim=1)
    negative = (1.0 - targets).sum(dim=1)
    positive_loss = -(targets * F.logsigmoid(logits)).sum(dim=1) / positive.clamp_min(1.0)
    negative_loss = -((1.0 - targets) * F.logsigmoid(-logits)).sum(dim=1) / negative.clamp_min(1.0)
    both = (positive > 0.0) & (negative > 0.0)
    values = torch.where(both, 0.5 * (positive_loss + negative_loss), positive_loss + negative_loss)
    return values.mean()


def sgcdet_sparse_refiner_loss(
    output: Mapping[str, torch.Tensor],
    *,
    target_residual: torch.Tensor,
    geometry_mask: torch.Tensor,
    local_boxes: torch.Tensor,
    matched_gt_index: torch.Tensor,
    baseline_iou: torch.Tensor,
    target_iou: torch.Tensor,
    aligned_basis: torch.Tensor,
    original_aligned_center: torch.Tensor,
    matched_gt_box: torch.Tensor,
    iou_gain: torch.Tensor,
    cross_iou50: torch.Tensor,
    ap50_weight: torch.Tensor,
    runtime_eligible: torch.Tensor,
    identity_tp50: torch.Tensor,
    occupancy_weight: float = 1.0,
    residual_weight: float = 1.0,
    candidate_iou_weight: float = 1.0,
    improvement_weight: float = 1.0,
    uncertainty_weight: float = 0.10,
    iou_gain_weight: float = 2.0,
    cross_iou50_weight: float = 4.0,
    preserve_iou50_weight: float = 2.0,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    batch = int(target_residual.shape[0])
    required_shapes = {
        "center_residual_fraction": (batch, 3),
        "log_dimension_residual": (batch, 3),
        "candidate_iou": (batch,),
        "improvement_probability": (batch,),
        "uncertainty": (batch,),
    }
    for name, shape in required_shapes.items():
        if name not in output or output[name].shape != shape:
            raise ValueError(f"model output {name} must have shape {shape}")
    for logits_name, targets_name in (
        ("coarse_occupancy_logits", "coarse_occupancy_targets"),
        ("occupancy_logits", "occupancy_targets"),
    ):
        if logits_name not in output or targets_name not in output:
            raise ValueError(f"model output is missing {logits_name}/{targets_name}")
    if geometry_mask.shape != (batch,) or geometry_mask.dtype != torch.bool:
        raise TypeError("geometry_mask must be Boolean [B]")
    if ap50_weight.shape != (batch,) or not torch.all(ap50_weight >= 1.0):
        raise ValueError("ap50_weight must be [B] and at least one")

    fine_occupancy = _balanced_occupancy_bce(
        output["occupancy_logits"], output["occupancy_targets"]
    )
    coarse_occupancy = _balanced_occupancy_bce(
        output["coarse_occupancy_logits"], output["coarse_occupancy_targets"]
    )
    occupancy_loss = 0.5 * (fine_occupancy + coarse_occupancy)

    center_values = F.smooth_l1_loss(
        output["center_residual_fraction"], target_residual[:, :3],
        beta=0.05, reduction="none",
    ).mean(dim=1)
    dimension_values = F.smooth_l1_loss(
        output["log_dimension_residual"], target_residual[:, 3:],
        beta=0.05, reduction="none",
    ).mean(dim=1)
    residual_loss = _weighted_mean(
        center_values + dimension_values, ap50_weight, geometry_mask
    )

    realized_iou = differentiable_aligned_aabb_iou(
        output, local_boxes, aligned_basis, original_aligned_center, matched_gt_box
    )
    matched = matched_gt_index >= 0
    quality_target = realized_iou.detach().clamp(0.0, 1.0)
    candidate_iou_values = F.binary_cross_entropy(
        output["candidate_iou"].clamp(1e-6, 1.0 - 1e-6),
        quality_target,
        reduction="none",
    )
    candidate_iou_loss = _weighted_mean(candidate_iou_values, ap50_weight, matched)

    improve_target = (
        quality_target > baseline_iou + 1e-4
    ).to(output["improvement_probability"].dtype)
    improvement_values = F.binary_cross_entropy(
        output["improvement_probability"].clamp(1e-6, 1.0 - 1e-6),
        improve_target,
        reduction="none",
    )
    improvement_loss = _weighted_mean(
        improvement_values, ap50_weight, runtime_eligible & matched
    )

    error = output["candidate_iou"] - quality_target
    variance = output["uncertainty"].square().clamp_min(1e-6)
    uncertainty_values = 0.5 * (error.square() / variance + torch.log(variance))
    uncertainty_loss = _weighted_mean(uncertainty_values, ap50_weight, matched)

    realized_gain = realized_iou - baseline_iou
    gain_values = F.smooth_l1_loss(
        realized_gain, iou_gain, beta=0.05, reduction="none"
    )
    iou_gain_loss = _weighted_mean(gain_values, ap50_weight, geometry_mask)
    cross_values = F.relu(AP50_LOSS_TARGET - realized_iou).square()
    cross_loss = _weighted_mean(cross_values, ap50_weight, cross_iou50)
    preserve_mask = runtime_eligible & matched & identity_tp50
    preserve_loss = _weighted_mean(cross_values, ap50_weight, preserve_mask)

    losses = {
        "occupancy": occupancy_loss,
        "residual": residual_loss,
        "candidate_iou": candidate_iou_loss,
        "improvement": improvement_loss,
        "uncertainty": uncertainty_loss,
        "iou_gain": iou_gain_loss,
        "cross_iou50": cross_loss,
        "preserve_iou50": preserve_loss,
    }
    weights = {
        "occupancy": float(occupancy_weight),
        "residual": float(residual_weight),
        "candidate_iou": float(candidate_iou_weight),
        "improvement": float(improvement_weight),
        "uncertainty": float(uncertainty_weight),
        "iou_gain": float(iou_gain_weight),
        "cross_iou50": float(cross_iou50_weight),
        "preserve_iou50": float(preserve_iou50_weight),
    }
    total = sum(weights[name] * losses[name] for name in losses)
    if not torch.isfinite(total):
        raise RuntimeError("sparse-refiner loss became non-finite")

    candidate_tp50 = realized_iou >= AP50_THRESHOLD
    cross_success = cross_iou50 & candidate_tp50
    drop50 = preserve_mask & ~candidate_tp50
    metrics: Dict[str, torch.Tensor] = {
        "loss": total.detach(),
        **{f"{name}_loss": value.detach() for name, value in losses.items()},
        "realized_candidate_iou": realized_iou.mean().detach(),
        "candidate_iou_mae": error.abs().mean().detach(),
        "mean_uncertainty": output["uncertainty"].mean().detach(),
        "cross50_success_count": cross_success.sum().detach(),
        "drop50_count": drop50.sum().detach(),
        "eligible_matched_count": (runtime_eligible & matched).sum().detach(),
        "geometry_positive_count": geometry_mask.sum().detach(),
    }
    # Target IoU is diagnostic only; candidate-quality supervision always
    # uses the current residual's realized IoU above.
    metrics["oracle_realized_iou_gap"] = (
        target_iou - realized_iou.detach()
    ).abs().mean()
    return total, metrics


def _set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _run_epoch(
    model: SGCDetInspiredLocalSparseRefiner,
    loader: DataLoader,
    *,
    optimizer: Optional[torch.optim.Optimizer],
    loss_weights: Mapping[str, float],
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: Dict[str, float] = {}
    samples = 0
    for batch in loader:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(
                batch["points_local"], batch["point_mask"],
                batch["local_boxes"], batch["quality_features"],
                batch["view_features"], batch["view_mask"],
            )
            loss, metrics = sgcdet_sparse_refiner_loss(
                output,
                target_residual=batch["target_residual"],
                geometry_mask=batch["geometry_mask"],
                local_boxes=batch["local_boxes"],
                matched_gt_index=batch["matched_gt_index"],
                baseline_iou=batch["baseline_iou"],
                target_iou=batch["target_iou"],
                aligned_basis=batch["aligned_basis"],
                original_aligned_center=batch["original_aligned_center"],
                matched_gt_box=batch["matched_gt_box"],
                iou_gain=batch["iou_gain"],
                cross_iou50=batch["cross_iou50"],
                ap50_weight=batch["ap50_weight"],
                runtime_eligible=batch["runtime_eligible"],
                identity_tp50=batch["identity_tp50"],
                **loss_weights,
            )
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        batch_size = int(batch["points_local"].shape[0])
        samples += batch_size
        for name, value in metrics.items():
            number = float(value)
            if name.endswith("_count"):
                totals[name] = totals.get(name, 0.0) + number
            else:
                totals[name] = totals.get(name, 0.0) + number * batch_size
    if samples == 0:
        raise ValueError("data loader produced no samples")
    result = {
        name: value if name.endswith("_count") else value / samples
        for name, value in totals.items()
    }
    denominator = result.get("eligible_matched_count", 0.0)
    result["local_net_tp50_proxy"] = (
        (result.get("cross50_success_count", 0.0) - result.get("drop50_count", 0.0))
        / denominator if denominator > 0.0 else 0.0
    )
    return result


def _nonnegative(name: str, value: float) -> float:
    if isinstance(value, bool) or not np.isfinite(value) or float(value) < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return float(value)


def train_sgcdet_sparse_refiner(
    dataset_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    config: Optional[SGCDetLocalSparseRefinerConfig] = None,
    epochs: int = 40,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    validation_fraction: float = 0.2,
    seed: int = 1337,
    identity_only: bool = False,
    **loss_weights: float,
) -> Dict[str, Any]:
    dataset = Path(dataset_path)
    dataset_sha256 = _sha256_file(dataset)
    data = load_sgcdet_sparse_refiner_dataset(dataset)
    cfg = (config or SGCDetLocalSparseRefinerConfig()).validated()
    if not np.isclose(cfg.max_center_fraction, data.max_center_fraction, atol=1e-7, rtol=0.0):
        raise ValueError("model/dataset max_center_fraction differs")
    if not np.isclose(
        cfg.max_log_dimension_residual,
        data.max_log_dimension_residual,
        atol=1e-7,
        rtol=0.0,
    ):
        raise ValueError("model/dataset dimension residual bound differs")
    if isinstance(identity_only, (np.bool_, bool)) is False:
        raise TypeError("identity_only must be Boolean")
    for name, value in (("epochs", epochs), ("batch_size", batch_size), ("seed", seed)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must lie in (0,1)")
    learning_rate = _nonnegative("learning_rate", learning_rate)
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    weight_decay = _nonnegative("weight_decay", weight_decay)
    default_weights = {
        "occupancy_weight": 1.0,
        "residual_weight": 1.0,
        "candidate_iou_weight": 1.0,
        "improvement_weight": 1.0,
        "uncertainty_weight": 0.10,
        "iou_gain_weight": 2.0,
        "cross_iou50_weight": 4.0,
        "preserve_iou50_weight": 2.0,
    }
    unknown = set(loss_weights) - set(default_weights)
    if unknown:
        raise TypeError(f"unknown loss weights: {sorted(unknown)}")
    default_weights.update(
        {name: _nonnegative(name, value) for name, value in loss_weights.items()}
    )
    if sum(default_weights.values()) <= 0.0:
        raise ValueError("at least one loss weight must be positive")

    train_indices, validation_indices = deterministic_scene_holdout(
        data.scene_ids, validation_fraction, seed
    )
    train_scenes = sorted(set(data.scene_ids[train_indices].tolist()))
    validation_scenes = sorted(set(data.scene_ids[validation_indices].tolist()))
    _set_determinism(seed)
    model = SGCDetInspiredLocalSparseRefiner(cfg).cpu()

    best_epoch = -1
    best_validation_loss = float("nan")
    best_validation_proxy = float("nan")
    train_metrics: Dict[str, float] = {}
    validation_metrics: Dict[str, float] = {}
    if not identity_only:
        balanced_epoch_indices(train_indices, data.geometry_mask, seed)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        validation_loader = DataLoader(
            _SparseArrayDataset(data, validation_indices),
            batch_size=min(batch_size, len(validation_indices)),
            shuffle=False, num_workers=0,
        )
        best_state: Optional[Dict[str, torch.Tensor]] = None
        selection: Optional[tuple[float, float]] = None
        for epoch in range(epochs):
            epoch_indices = balanced_epoch_indices(
                train_indices, data.geometry_mask, seed + epoch
            )
            training_loader = DataLoader(
                _SparseArrayDataset(data, epoch_indices),
                batch_size=min(batch_size, len(epoch_indices)),
                shuffle=False, num_workers=0,
            )
            current_train = _run_epoch(
                model, training_loader, optimizer=optimizer,
                loss_weights=default_weights,
            )
            current_validation = _run_epoch(
                model, validation_loader, optimizer=None,
                loss_weights=default_weights,
            )
            candidate = (
                float(current_validation["loss"]),
                -float(current_validation["local_net_tp50_proxy"]),
            )
            if selection is None or candidate < selection:
                selection = candidate
                best_epoch = epoch
                best_validation_loss = candidate[0]
                best_validation_proxy = -candidate[1]
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                train_metrics = dict(current_train)
                validation_metrics = dict(current_validation)
        if best_state is None:
            raise RuntimeError("training produced no checkpoint")
        model.load_state_dict(best_state, strict=True)

    if dataset_sha256 != _sha256_file(dataset):
        raise RuntimeError("training dataset changed during training")
    metadata: Dict[str, Any] = {
        "training_dataset_schema": DATASET_SCHEMA,
        "training_dataset_format_version": DATASET_FORMAT_VERSION,
        "training_dataset_sha256": dataset_sha256,
        "source_joint_dataset_sha256": data.source_joint_dataset_sha256,
        "source_b5_dataset_sha256": data.source_b5_dataset_sha256,
        "objective": "identity" if identity_only else OBJECTIVE,
        "identity_only": bool(identity_only),
        "reference": SGCDET_SPARSE_REFINER_REFERENCE,
        "samples": data.sample_count,
        "training_samples": int(len(train_indices)),
        "validation_samples": int(len(validation_indices)),
        "training_scenes": train_scenes,
        "validation_scenes": validation_scenes,
        "training_scene_sha256": _scene_sha256(train_scenes),
        "validation_scene_sha256": _scene_sha256(validation_scenes),
        "forbidden_scene_count": data.forbidden_scene_count,
        "forbidden_scene_sha256": data.forbidden_scene_sha256,
        "scene_leakage": False,
        "seed": int(seed),
        "epochs": 0 if identity_only else int(epochs),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "best_validation_local_net_tp50_proxy": best_validation_proxy,
        "loss_weights": dict(default_weights),
        "device": "cpu",
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
    }
    output = Path(output_path)
    if output.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("checkpoint must end in .pt or .pth")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        torch.save(
            make_sgcdet_sparse_refiner_checkpoint(model, metadata=metadata),
            temporary,
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    verification = SGCDetInspiredLocalSparseRefiner(cfg).cpu()
    load_sgcdet_sparse_refiner_checkpoint(
        verification, output, map_location="cpu"
    )
    return {
        "output": str(output),
        "identity_only": bool(identity_only),
        "samples": data.sample_count,
        "train_samples": int(len(train_indices)),
        "validation_samples": int(len(validation_indices)),
        "train_scenes": train_scenes,
        "validation_scenes": validation_scenes,
        "scene_leakage": False,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "best_validation_local_net_tp50_proxy": best_validation_proxy,
        "train": train_metrics,
        "validation": validation_metrics,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--identity-only", action="store_true")
    parser.add_argument("--coarse-hidden-dim", type=int, default=48)
    parser.add_argument("--coarse-embedding-dim", type=int, default=64)
    parser.add_argument("--occupancy-hidden-dim", type=int, default=48)
    parser.add_argument("--selected-hidden-dim", type=int, default=64)
    parser.add_argument("--selected-embedding-dim", type=int, default=96)
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--max-center-fraction", type=float, default=0.15)
    parser.add_argument(
        "--max-log-dimension-residual", type=float,
        default=float(math.log(1.25)),
    )
    for name, default in (
        ("occupancy", 1.0), ("residual", 1.0),
        ("candidate-iou", 1.0), ("improvement", 1.0),
        ("uncertainty", 0.10), ("iou-gain", 2.0),
        ("cross-iou50", 4.0), ("preserve-iou50", 2.0),
    ):
        parser.add_argument(f"--{name}-weight", type=float, default=default)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config = SGCDetLocalSparseRefinerConfig(
        coarse_hidden_dim=args.coarse_hidden_dim,
        coarse_embedding_dim=args.coarse_embedding_dim,
        occupancy_hidden_dim=args.occupancy_hidden_dim,
        selected_hidden_dim=args.selected_hidden_dim,
        selected_embedding_dim=args.selected_embedding_dim,
        head_hidden_dim=args.head_hidden_dim,
        max_center_fraction=args.max_center_fraction,
        max_log_dimension_residual=args.max_log_dimension_residual,
    ).validated()
    result = train_sgcdet_sparse_refiner(
        args.input, args.output, config=config, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction, seed=args.seed,
        identity_only=args.identity_only,
        occupancy_weight=args.occupancy_weight,
        residual_weight=args.residual_weight,
        candidate_iou_weight=args.candidate_iou_weight,
        improvement_weight=args.improvement_weight,
        uncertainty_weight=args.uncertainty_weight,
        iou_gain_weight=args.iou_gain_weight,
        cross_iou50_weight=args.cross_iou50_weight,
        preserve_iou50_weight=args.preserve_iou50_weight,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
