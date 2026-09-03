#!/usr/bin/env python3
"""Train the P2 foreground-occupancy head from train-only P1 diagnostics.

P1 diagnostics contain inputs but no labels.  This utility constructs labels
offline from ScanNet *training* ground truth and frozen B6 predictions:

1. Ground-truth boxes already covered by frozen B6 are removed.
2. Every observed P1 residual voxel whose centre lies inside a remaining box
   receives foreground-occupancy target one; every other voxel receives zero.
3. A small class-agnostic MLP is trained with occupancy BCE only.

The online P2 observer never receives ground truth.  The saved checkpoint is
bound to the exact P1 and B6 checkpoints and records the forbidden validation
split, so runtime provenance checks can fail closed.

Only trusted, locally produced BoxFusion prediction pickle files should be
supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boxfusion.occupancy_topk import (  # noqa: E402
    ForegroundOccupancyHead,
    P2_HEAD_SCHEMA,
    assign_foreground_occupancy_targets,
)
from boxfusion.residual_proposal import (  # noqa: E402
    P1_FEATURE_DIM,
    P1_FEATURE_NAMES,
    P1_HEAD_SCHEMA,
)
from tools.train_p1_residual_head import (  # noqa: E402
    deterministic_scene_split,
    load_axis_alignment,
    load_gt_boxes,
    load_prediction_corners,
    load_scene_voxels,
    read_scene_ids,
    residual_gt_world_boxes,
    validate_train_split,
)


TRAINING_SCHEMA = "boxfusion.p2_occupancy_training.v1"


@dataclass(frozen=True)
class OccupancyTrainingData:
    """Validated, scene-labelled P2 training rows."""

    features: np.ndarray
    occupancy: np.ndarray
    scene_ids: np.ndarray
    feature_names: tuple[str, ...]
    scene_summaries: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        features = np.asarray(self.features)
        occupancy = np.asarray(self.occupancy)
        scene_ids = np.asarray(self.scene_ids)
        if features.ndim != 2 or features.shape[1] != P1_FEATURE_DIM:
            raise ValueError(
                f"P2 features must have shape [N,{P1_FEATURE_DIM}]"
            )
        if occupancy.shape != (len(features),):
            raise ValueError("P2 occupancy targets must have shape [N]")
        if scene_ids.shape != (len(features),) or scene_ids.dtype.hasobject:
            raise ValueError("P2 scene_ids must be a non-object [N] array")
        if (
            not np.isfinite(features).all()
            or not np.isfinite(occupancy).all()
            or np.any((occupancy != 0.0) & (occupancy != 1.0))
        ):
            raise ValueError("P2 features/occupancy targets are invalid")
        if tuple(self.feature_names) != P1_FEATURE_NAMES:
            raise ValueError("P2 feature schema must exactly match P1")


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checkpoint_mapping(path: Path, *, role: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {role} checkpoint: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - PyTorch < 2.0
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"{role} checkpoint must contain a mapping")
    return payload


def validate_parent_provenance(
    *,
    p1_checkpoint: str | os.PathLike[str],
    b6_checkpoint: str | os.PathLike[str],
    train_scene_list: str | os.PathLike[str],
    forbidden_scene_list: str | os.PathLike[str],
    train_scenes: Sequence[str],
) -> dict[str, Any]:
    """Validate that P2 inputs match the exact train-only P1/B6 parents."""

    p1_path = Path(p1_checkpoint)
    b6_path = Path(b6_checkpoint)
    train_path = Path(train_scene_list)
    forbidden_path = Path(forbidden_scene_list)
    for role, path in (
        ("B6", b6_path),
        ("training scene list", train_path),
        ("forbidden scene list", forbidden_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing {role}: {path}")
    payload = _load_checkpoint_mapping(p1_path, role="P1")
    if payload.get("schema") != P1_HEAD_SCHEMA:
        raise ValueError("P1 checkpoint schema mismatch")
    if tuple(payload.get("feature_names", ())) != P1_FEATURE_NAMES:
        raise ValueError("P1 checkpoint feature schema mismatch")
    if not isinstance(payload.get("model_config"), Mapping) or not isinstance(
        payload.get("state_dict"), Mapping
    ):
        raise ValueError("P1 checkpoint lacks model_config/state_dict")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("P1 checkpoint lacks train-only provenance")

    recorded_scenes = provenance.get("train_scene_ids")
    recorded_overlap = provenance.get("forbidden_overlap")
    recorded_train_sha = str(
        provenance.get("train_scene_list_sha256", "")
    ).lower()
    recorded_forbidden_sha = str(
        provenance.get("forbidden_scene_list_sha256", "")
    ).lower()
    recorded_b6_sha = str(
        provenance.get("b6_checkpoint_sha256", "")
    ).lower()
    expected_train_sha = file_sha256(train_path)
    expected_forbidden_sha = file_sha256(forbidden_path)
    expected_b6_sha = file_sha256(b6_path)
    canonical_scenes = tuple(str(scene) for scene in train_scenes)
    scene_summaries = provenance.get("scene_summaries")
    summary_by_scene: dict[str, dict[str, str]] = {}
    if (
        not isinstance(scene_summaries, Sequence)
        or isinstance(scene_summaries, (str, bytes))
    ):
        raise ValueError("P1 checkpoint lacks artifact scene summaries")
    for row in scene_summaries:
        if not isinstance(row, Mapping):
            raise ValueError("P1 artifact scene summary must be a mapping")
        scene_id = row.get("scene_id")
        hashes = {
            key: str(row.get(key, "")).lower()
            for key in (
                "diagnostic_sha256",
                "prediction_sha256",
                "ground_truth_sha256",
            )
        }
        if (
            not isinstance(scene_id, str)
            or scene_id in summary_by_scene
            or any(
                len(value) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in value
                )
                for value in hashes.values()
            )
        ):
            raise ValueError("P1 artifact scene summary is invalid")
        summary_by_scene[scene_id] = hashes
    if (
        not isinstance(recorded_scenes, Sequence)
        or isinstance(recorded_scenes, (str, bytes))
        or tuple(recorded_scenes) != canonical_scenes
        or recorded_overlap != []
        or recorded_train_sha != expected_train_sha
        or recorded_forbidden_sha != expected_forbidden_sha
        or recorded_b6_sha != expected_b6_sha
        or tuple(summary_by_scene) != canonical_scenes
    ):
        raise ValueError(
            "P1 checkpoint provenance disagrees with the requested "
            "train-only P2 inputs"
        )
    return {
        "p1_checkpoint_sha256": file_sha256(p1_path),
        "b6_checkpoint_sha256": expected_b6_sha,
        "train_scene_list_sha256": expected_train_sha,
        "forbidden_scene_list_sha256": expected_forbidden_sha,
        "scene_artifact_hashes": summary_by_scene,
    }


def validate_parent_artifact_chain(
    scene_summaries: Sequence[Mapping[str, Any]],
    expected_hashes: Mapping[str, Mapping[str, str]],
) -> None:
    """Fail closed if P2 inputs differ from exact P1 training artifacts."""

    actual_by_scene: dict[str, Mapping[str, Any]] = {}
    for row in scene_summaries:
        scene_id = row.get("scene_id")
        if not isinstance(scene_id, str) or scene_id in actual_by_scene:
            raise ValueError("P2 scene artifact summary is invalid")
        actual_by_scene[scene_id] = row
    if tuple(actual_by_scene) != tuple(expected_hashes):
        raise ValueError("P2 artifact scene set/order disagrees with P1")
    for scene_id, expected in expected_hashes.items():
        actual = actual_by_scene[scene_id]
        for key in (
            "diagnostic_sha256",
            "prediction_sha256",
            "ground_truth_sha256",
        ):
            if str(actual.get(key, "")).lower() != str(
                expected.get(key, "")
            ).lower():
                raise ValueError(
                    f"{scene_id}: P2 {key} disagrees with exact P1 artifact"
                )


def _deterministic_subsample(
    scene_id: str,
    occupancy: np.ndarray,
    *,
    maximum_voxels: int,
    negative_ratio: float,
    seed: int,
) -> np.ndarray:
    labels = np.asarray(occupancy, dtype=np.float32).reshape(-1)
    positive = np.flatnonzero(labels > 0.5)
    negative = np.flatnonzero(labels <= 0.5)
    if not math.isfinite(float(negative_ratio)) or float(negative_ratio) < 0.0:
        raise ValueError("negative_ratio must be finite and non-negative")
    if isinstance(maximum_voxels, bool) or int(maximum_voxels) < 0:
        raise ValueError("maximum_voxels must be a non-negative integer")
    desired_negative = min(
        len(negative),
        int(round(max(len(positive), 1) * float(negative_ratio))),
    )
    digest = hashlib.sha256(f"{int(seed)}:{scene_id}".encode()).digest()
    rng = np.random.default_rng(
        int.from_bytes(digest[:8], "little") % (2**32)
    )
    chosen_negative = (
        np.sort(
            rng.choice(negative, size=desired_negative, replace=False)
        )
        if desired_negative < len(negative)
        else negative
    )
    selected = np.sort(
        np.concatenate((positive, chosen_negative))
    ).astype(np.int64)
    limit = int(maximum_voxels)
    if limit > 0 and len(selected) > limit:
        if len(positive) >= limit:
            selected = np.sort(
                rng.choice(positive, size=limit, replace=False)
            ).astype(np.int64)
        else:
            remaining = limit - len(positive)
            sampled_negative = (
                np.sort(
                    rng.choice(
                        chosen_negative, size=remaining, replace=False
                    )
                )
                if remaining < len(chosen_negative)
                else chosen_negative
            )
            selected = np.sort(
                np.concatenate((positive, sampled_negative))
            ).astype(np.int64)
    return selected


def build_training_data(
    *,
    scenes: Sequence[str],
    diagnostics_root: str | os.PathLike[str],
    prediction_root: str | os.PathLike[str],
    gt_root: str | os.PathLike[str],
    scans_root: str | os.PathLike[str],
    covered_iou: float = 0.15,
    occupancy_margin: float = 0.0,
    maximum_voxels_per_scene: int = 60000,
    negative_ratio: float = 8.0,
    seed: int = 1337,
) -> OccupancyTrainingData:
    """Construct class-agnostic occupancy labels from train-only inputs."""

    diagnostics = Path(diagnostics_root)
    predictions = Path(prediction_root)
    ground_truth = Path(gt_root)
    scans = Path(scans_root)
    for role, root in (
        ("diagnostics", diagnostics),
        ("prediction", predictions),
        ("ground-truth", ground_truth),
        ("scans", scans),
    ):
        if not root.is_dir():
            raise FileNotFoundError(f"{role} root not found: {root}")
    if not 0.0 <= float(covered_iou) <= 1.0:
        raise ValueError("covered_iou must lie in [0,1]")
    if not math.isfinite(float(occupancy_margin)) or float(
        occupancy_margin
    ) < 0.0:
        raise ValueError("occupancy_margin must be finite and non-negative")

    feature_parts: list[np.ndarray] = []
    occupancy_parts: list[np.ndarray] = []
    scene_parts: list[np.ndarray] = []
    summaries: list[Mapping[str, Any]] = []
    canonical_names: tuple[str, ...] | None = None
    for scene_id in scenes:
        diagnostic_path = diagnostics / f"{scene_id}_tracks.npz"
        prediction_path = predictions / f"{scene_id}_boxes.pkl"
        gt_path = ground_truth / f"{scene_id}_bbox.npy"
        inputs = load_scene_voxels(
            diagnostic_path, expected_scene_id=scene_id
        )
        if canonical_names is None:
            canonical_names = inputs.feature_names
        elif inputs.feature_names != canonical_names:
            raise ValueError(
                f"{scene_id}: P1 feature schema differs across scenes"
            )
        gt_boxes = load_gt_boxes(gt_path)
        baseline_corners = load_prediction_corners(prediction_path)
        alignment = load_axis_alignment(scans, scene_id)
        residual_boxes, residual_indices = residual_gt_world_boxes(
            gt_boxes,
            baseline_corners,
            alignment,
            covered_iou=float(covered_iou),
        )
        occupancy = assign_foreground_occupancy_targets(
            inputs.centers_world,
            residual_boxes,
            margin=float(occupancy_margin),
        )
        selected = _deterministic_subsample(
            scene_id,
            occupancy,
            maximum_voxels=int(maximum_voxels_per_scene),
            negative_ratio=float(negative_ratio),
            seed=int(seed),
        )
        if len(selected):
            feature_parts.append(inputs.features[selected])
            occupancy_parts.append(occupancy[selected])
            scene_parts.append(
                np.full(
                    len(selected),
                    scene_id,
                    dtype=f"<U{max(12, len(scene_id))}",
                )
            )
        summaries.append(
            {
                "scene_id": scene_id,
                "diagnostic_sha256": file_sha256(diagnostic_path),
                "prediction_sha256": file_sha256(prediction_path),
                "ground_truth_sha256": file_sha256(gt_path),
                "snapshots": int(len(inputs.offsets) - 1),
                "voxel_count": int(len(inputs.features)),
                "selected_voxels": int(len(selected)),
                "positive_voxels": int(
                    np.sum(occupancy[selected] > 0.5)
                ),
                "negative_voxels": int(
                    np.sum(occupancy[selected] <= 0.5)
                ),
                "ground_truth_count": int(len(gt_boxes)),
                "residual_ground_truth_count": int(len(residual_boxes)),
                "residual_ground_truth_indices": residual_indices.tolist(),
            }
        )
    if canonical_names is None or not feature_parts:
        raise ValueError("no P2 training samples were loaded")
    features = np.concatenate(feature_parts, axis=0).astype(
        np.float32, copy=False
    )
    occupancy = np.concatenate(occupancy_parts, axis=0).astype(
        np.float32, copy=False
    )
    scene_ids = np.concatenate(scene_parts, axis=0)
    if not np.any(occupancy > 0.5):
        raise ValueError(
            "P2 training data has no positive occupancy voxels"
        )
    if not np.any(occupancy <= 0.5):
        raise ValueError(
            "P2 training data has no negative occupancy voxels"
        )
    return OccupancyTrainingData(
        features=np.ascontiguousarray(features),
        occupancy=np.ascontiguousarray(occupancy),
        scene_ids=np.asarray(scene_ids, dtype=np.str_),
        feature_names=canonical_names,
        scene_summaries=tuple(summaries),
    )


def _bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    positive_weight: torch.Tensor,
) -> torch.Tensor:
    values = logits.reshape(-1)
    labels = targets.reshape(-1)
    if values.shape != labels.shape:
        raise ValueError("P2 logits and occupancy targets must align")
    return F.binary_cross_entropy_with_logits(
        values, labels, pos_weight=positive_weight
    )


def _occupancy_metrics(
    logits: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    probability = torch.sigmoid(logits.reshape(-1))
    target = targets.reshape(-1)
    predicted = probability >= 0.5
    positive = target > 0.5
    true_positive = torch.sum(predicted & positive).item()
    false_positive = torch.sum(predicted & ~positive).item()
    false_negative = torch.sum(~predicted & positive).item()
    correct = torch.sum(predicted == positive).item()
    return {
        "precision_at_0p5": float(
            true_positive / max(true_positive + false_positive, 1)
        ),
        "recall_at_0p5": float(
            true_positive / max(true_positive + false_negative, 1)
        ),
        "accuracy_at_0p5": float(correct / max(len(target), 1)),
        "positive_count": float(torch.sum(positive).item()),
        "sample_count": float(len(target)),
    }


def train_occupancy_head(
    data: OccupancyTrainingData,
    *,
    hidden_dim: int = 32,
    validation_fraction: float = 0.20,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 8192,
    seed: int = 1337,
    device: str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Train with scene-disjoint validation and occupancy BCE only."""

    if int(hidden_dim) < 1 or int(epochs) < 1 or int(batch_size) < 1:
        raise ValueError("hidden_dim, epochs, and batch_size must be positive")
    if float(learning_rate) <= 0.0 or float(weight_decay) < 0.0:
        raise ValueError("learning_rate must be >0 and weight_decay >=0")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch_device = torch.device(device)
    (
        train_indices,
        validation_indices,
        training_scenes,
        validation_scenes,
    ) = deterministic_scene_split(
        data.scene_ids,
        validation_fraction=float(validation_fraction),
        seed=int(seed),
    )
    if set(training_scenes) & set(validation_scenes):
        raise RuntimeError("P2 scene-disjoint split contains overlap")
    model = ForegroundOccupancyHead(
        input_dim=int(data.features.shape[1]),
        hidden_dim=int(hidden_dim),
    ).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    training_targets = data.occupancy[train_indices]
    positive_count = float(np.sum(training_targets > 0.5))
    negative_count = float(len(training_targets) - positive_count)
    if positive_count <= 0.0 or negative_count <= 0.0:
        raise ValueError(
            "P2 training partition must contain both occupancy classes"
        )
    positive_weight_value = min(
        max(negative_count / positive_count, 1.0), 50.0
    )
    positive_weight = torch.tensor(
        positive_weight_value, dtype=torch.float32, device=torch_device
    )
    features = torch.from_numpy(data.features)
    targets = torch.from_numpy(data.occupancy)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    best_loss = math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    last_record: dict[str, float] | None = None
    for epoch in range(int(epochs)):
        model.train()
        permutation = train_indices[
            torch.randperm(
                len(train_indices), generator=generator
            ).numpy()
        ]
        total_loss = 0.0
        total_samples = 0
        for start in range(0, len(permutation), int(batch_size)):
            batch = permutation[start : start + int(batch_size)]
            batch_features = features[batch].to(torch_device)
            batch_targets = targets[batch].to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = _bce_loss(
                logits,
                batch_targets,
                positive_weight=positive_weight,
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item()) * len(batch)
            total_samples += len(batch)
        model.eval()
        with torch.inference_mode():
            validation_logits = model(
                features[validation_indices].to(torch_device)
            )
            validation_loss = _bce_loss(
                validation_logits,
                targets[validation_indices].to(torch_device),
                positive_weight=positive_weight,
            )
        last_record = {
            "epoch": float(epoch),
            "training_occupancy_bce": float(
                total_loss / max(total_samples, 1)
            ),
            "validation_occupancy_bce": float(
                validation_loss.item()
            ),
        }
        if last_record["validation_occupancy_bce"] < best_loss:
            best_loss = last_record["validation_occupancy_bce"]
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is None or last_record is None:
        raise RuntimeError("P2 training did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.inference_mode():
        validation_logits = model(
            features[validation_indices].to(torch_device)
        )
        validation_metrics = _occupancy_metrics(
            validation_logits,
            targets[validation_indices].to(torch_device),
        )
    return model.cpu(), {
        "objective": "occupancy_bce_only",
        "loss_terms": {"occupancy_bce": 1.0},
        "best_epoch": int(best_epoch),
        "best_validation_occupancy_bce": float(best_loss),
        "training_scenes": list(training_scenes),
        "validation_scenes": list(validation_scenes),
        "training_samples": int(len(train_indices)),
        "validation_samples": int(len(validation_indices)),
        "positive_weight": float(positive_weight_value),
        "validation_metrics": validation_metrics,
        "last_epoch": last_record,
    }


def save_checkpoint(
    output_path: str | os.PathLike[str],
    *,
    model: nn.Module,
    feature_names: Sequence[str],
    hidden_dim: int,
    training_config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Path:
    """Atomically save a plain state-dict P2 checkpoint."""

    output = Path(output_path)
    if output.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("P2 checkpoint must end in .pt or .pth")
    names = tuple(str(name) for name in feature_names)
    if names != P1_FEATURE_NAMES:
        raise ValueError("P2 checkpoint feature schema mismatch")
    model_config = {
        "input_dim": len(names),
        "hidden_dim": int(hidden_dim),
        "output_dim": 1,
        "target": "voxel_center_inside_residual_gt_aabb",
    }
    checkpoint = {
        "schema": P2_HEAD_SCHEMA,
        "model_config": model_config,
        "feature_names": list(names),
        "state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
        "config": {
            "model": dict(model_config),
            "training": dict(training_config),
        },
        "training_config": dict(training_config),
        "metrics": dict(metrics),
        "provenance": dict(provenance),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, output)
    loaded = _load_checkpoint_mapping(output, role="saved P2")
    if (
        loaded.get("schema") != P2_HEAD_SCHEMA
        or tuple(loaded.get("feature_names", ())) != names
    ):
        raise RuntimeError("saved P2 checkpoint failed schema verification")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-root", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--scans-root", required=True, type=Path)
    parser.add_argument("--train-scene-list", required=True, type=Path)
    parser.add_argument("--forbidden-scene-list", required=True, type=Path)
    parser.add_argument("--p1-checkpoint", required=True, type=Path)
    parser.add_argument("--b6-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--covered-iou", type=float, default=0.15)
    parser.add_argument("--occupancy-margin", type=float, default=0.0)
    parser.add_argument("--max-voxels-per-scene", type=int, default=60000)
    parser.add_argument("--negative-ratio", type=float, default=8.0)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--summary-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    train_scenes = read_scene_ids(args.train_scene_list, role="P2 training")
    forbidden_scenes = read_scene_ids(
        args.forbidden_scene_list, role="forbidden validation"
    )
    train_scenes = validate_train_split(train_scenes, forbidden_scenes)
    parent = validate_parent_provenance(
        p1_checkpoint=args.p1_checkpoint,
        b6_checkpoint=args.b6_checkpoint,
        train_scene_list=args.train_scene_list,
        forbidden_scene_list=args.forbidden_scene_list,
        train_scenes=train_scenes,
    )
    data = build_training_data(
        scenes=train_scenes,
        diagnostics_root=args.diagnostics_root,
        prediction_root=args.prediction_root,
        gt_root=args.gt_root,
        scans_root=args.scans_root,
        covered_iou=args.covered_iou,
        occupancy_margin=args.occupancy_margin,
        maximum_voxels_per_scene=args.max_voxels_per_scene,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
    )
    validate_parent_artifact_chain(
        data.scene_summaries,
        parent["scene_artifact_hashes"],
    )
    training_config = {
        "schema": TRAINING_SCHEMA,
        "objective": "occupancy_bce_only",
        "target": "voxel_center_inside_residual_gt_aabb",
        "covered_iou": float(args.covered_iou),
        "occupancy_margin": float(args.occupancy_margin),
        "maximum_voxels_per_scene": int(args.max_voxels_per_scene),
        "negative_ratio": float(args.negative_ratio),
        "hidden_dim": int(args.hidden_dim),
        "validation_fraction": float(args.validation_fraction),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "device": str(args.device),
        "loss_terms": {"occupancy_bce": 1.0},
    }
    model, metrics = train_occupancy_head(
        data,
        hidden_dim=args.hidden_dim,
        validation_fraction=args.validation_fraction,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    provenance = {
        "train_scene_list": str(args.train_scene_list.resolve()),
        "train_scene_list_sha256": parent[
            "train_scene_list_sha256"
        ],
        "forbidden_scene_list": str(
            args.forbidden_scene_list.resolve()
        ),
        "forbidden_scene_list_sha256": parent[
            "forbidden_scene_list_sha256"
        ],
        "train_scene_ids": list(train_scenes),
        "forbidden_scene_count": int(len(forbidden_scenes)),
        "forbidden_overlap": [],
        "p1_checkpoint": str(args.p1_checkpoint.resolve()),
        "p1_checkpoint_sha256": parent[
            "p1_checkpoint_sha256"
        ],
        "b6_checkpoint": str(args.b6_checkpoint.resolve()),
        "b6_checkpoint_sha256": parent[
            "b6_checkpoint_sha256"
        ],
        "diagnostics_root": str(args.diagnostics_root.resolve()),
        "prediction_root": str(args.prediction_root.resolve()),
        "gt_root": str(args.gt_root.resolve()),
        "scans_root": str(args.scans_root.resolve()),
        "scene_summaries": list(data.scene_summaries),
    }
    output = save_checkpoint(
        args.output,
        model=model,
        feature_names=data.feature_names,
        hidden_dim=args.hidden_dim,
        training_config=training_config,
        metrics=metrics,
        provenance=provenance,
    )
    summary = {
        "schema": P2_HEAD_SCHEMA,
        "training_schema": TRAINING_SCHEMA,
        "objective": "occupancy_bce_only",
        "output": str(output.resolve()),
        "scene_count": int(len(train_scenes)),
        "sample_count": int(len(data.features)),
        "positive_count": int(np.sum(data.occupancy > 0.5)),
        "negative_count": int(np.sum(data.occupancy <= 0.5)),
        "feature_names": list(data.feature_names),
        "p1_checkpoint_sha256": parent["p1_checkpoint_sha256"],
        "b6_checkpoint_sha256": parent["b6_checkpoint_sha256"],
        "metrics": metrics,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.summary_json.with_name(
            args.summary_json.name + ".tmp"
        )
        temporary.write_text(rendered + "\n", encoding="utf-8")
        os.replace(temporary, args.summary_json)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
