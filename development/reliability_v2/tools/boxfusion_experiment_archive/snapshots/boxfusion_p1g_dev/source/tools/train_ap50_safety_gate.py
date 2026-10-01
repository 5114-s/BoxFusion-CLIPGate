#!/usr/bin/env python3
"""Train a train-only pairwise ΔIoU/uncertainty safety gate.

Input archives use a non-pickled strict schema:

```
schema          scalar "boxfusion.ap50_gate_training"
format_version  scalar int64 1
feature_names   [F] string
gate_features   [N,F] float
original_iou    [N] float in [0,1]
candidate_iou   [N] float in [0,1]
scene_ids       [N] string
```

The split is scene-disjoint.  A ScanNet validation list can be supplied as a
hard forbidden set; any overlap aborts before training.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ap50_safety_gate import (  # noqa: E402
    AP50_SAFETY_GATE_FORMAT_VERSION,
    AP50_SAFETY_GATE_OUTPUT_NAMES,
    AP50_SAFETY_GATE_SCHEMA,
    AP50_SAFETY_GATE_OUTPUT_DIM,
    load_ap50_safety_gate,
)


TRAINING_SCHEMA = "boxfusion.ap50_gate_training"
TRAINING_FORMAT_VERSION = 1
TRAINING_KEYS = frozenset(
    {
        "schema",
        "format_version",
        "feature_names",
        "gate_features",
        "original_iou",
        "candidate_iou",
        "scene_ids",
    }
)


@dataclass(frozen=True)
class GateTrainingData:
    feature_names: Tuple[str, ...]
    features: np.ndarray
    original_iou: np.ndarray
    candidate_iou: np.ndarray
    scene_ids: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(len(self.features))


def _scalar_string(name: str, value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar")
    result = str(array.item())
    if not result:
        raise ValueError(f"{name} cannot be empty")
    return result


def _load_archive(path: Path) -> GateTrainingData:
    if not path.is_file():
        raise FileNotFoundError(f"training archive not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        files = set(archive.files)
        if files != set(TRAINING_KEYS):
            raise ValueError(
                f"{path}: training schema keys mismatch: "
                f"missing={sorted(TRAINING_KEYS - files)}, "
                f"extra={sorted(files - TRAINING_KEYS)}"
            )
        if _scalar_string("schema", archive["schema"]) != TRAINING_SCHEMA:
            raise ValueError(f"{path}: unsupported training schema")
        version = np.asarray(archive["format_version"])
        if (
            version.shape != ()
            or version.dtype.kind not in "iu"
            or int(version.item()) != TRAINING_FORMAT_VERSION
        ):
            raise ValueError(f"{path}: unsupported training format version")
        feature_names = tuple(
            str(item)
            for item in np.asarray(archive["feature_names"]).tolist()
        )
        if (
            not feature_names
            or any(not name for name in feature_names)
            or len(set(feature_names)) != len(feature_names)
        ):
            raise ValueError(f"{path}: invalid feature_names")
        features = np.asarray(archive["gate_features"], dtype=np.float32)
        original_iou = np.asarray(
            archive["original_iou"], dtype=np.float32
        )
        candidate_iou = np.asarray(
            archive["candidate_iou"], dtype=np.float32
        )
        scene_ids = np.asarray(archive["scene_ids"])
    if features.ndim != 2 or features.shape[1] != len(feature_names):
        raise ValueError(f"{path}: gate_features shape mismatch")
    count = len(features)
    if count < 1:
        raise ValueError(f"{path}: training archive is empty")
    if original_iou.shape != (count,) or candidate_iou.shape != (count,):
        raise ValueError(f"{path}: IoU arrays must have shape [{count}]")
    if scene_ids.shape != (count,) or scene_ids.dtype.kind not in "US":
        raise ValueError(f"{path}: scene_ids must be a string vector")
    if (
        not np.isfinite(features).all()
        or not np.isfinite(original_iou).all()
        or not np.isfinite(candidate_iou).all()
    ):
        raise ValueError(f"{path}: training arrays must be finite")
    if (
        ((original_iou < 0.0) | (original_iou > 1.0)).any()
        or ((candidate_iou < 0.0) | (candidate_iou > 1.0)).any()
    ):
        raise ValueError(f"{path}: IoU supervision must lie in [0,1]")
    if any(not str(scene_id) for scene_id in scene_ids):
        raise ValueError(f"{path}: scene_ids cannot contain empty values")
    return GateTrainingData(
        feature_names=feature_names,
        features=np.ascontiguousarray(features),
        original_iou=np.ascontiguousarray(original_iou),
        candidate_iou=np.ascontiguousarray(candidate_iou),
        scene_ids=np.asarray(scene_ids, dtype=str),
    )


def load_training_data(paths: Sequence[Path]) -> GateTrainingData:
    if not paths:
        raise ValueError("at least one training archive is required")
    archives = tuple(_load_archive(Path(path)) for path in paths)
    names = archives[0].feature_names
    if any(archive.feature_names != names for archive in archives[1:]):
        raise ValueError("training archives use different feature schemas")
    return GateTrainingData(
        feature_names=names,
        features=np.concatenate(
            [archive.features for archive in archives], axis=0
        ),
        original_iou=np.concatenate(
            [archive.original_iou for archive in archives], axis=0
        ),
        candidate_iou=np.concatenate(
            [archive.candidate_iou for archive in archives], axis=0
        ),
        scene_ids=np.concatenate(
            [archive.scene_ids for archive in archives], axis=0
        ),
    )


def _read_scene_list(path: Path | None) -> frozenset[str]:
    if path is None:
        return frozenset()
    if not path.is_file():
        raise FileNotFoundError(f"scene list not found: {path}")
    scenes = frozenset(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not scenes:
        raise ValueError(f"scene list is empty: {path}")
    return scenes


def validate_forbidden_scenes(
    data: GateTrainingData, forbidden: Iterable[str]
) -> None:
    overlap = sorted(set(data.scene_ids.tolist()) & set(forbidden))
    if overlap:
        preview = ", ".join(overlap[:8])
        raise ValueError(
            "training data overlaps forbidden validation scenes: "
            f"{preview}"
        )


def scene_disjoint_split(
    scene_ids: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...], Tuple[str, ...]]:
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must lie in (0,1)")
    unique = np.asarray(sorted(set(np.asarray(scene_ids, dtype=str))))
    if len(unique) < 2:
        raise ValueError("scene-disjoint training requires at least two scenes")
    rng = np.random.default_rng(int(seed))
    shuffled = unique[rng.permutation(len(unique))]
    validation_count = int(
        np.clip(
            round(len(unique) * float(validation_fraction)),
            1,
            len(unique) - 1,
        )
    )
    validation_scenes = tuple(sorted(shuffled[:validation_count].tolist()))
    training_scenes = tuple(sorted(shuffled[validation_count:].tolist()))
    validation = np.isin(scene_ids, validation_scenes)
    training = np.isin(scene_ids, training_scenes)
    if not training.any() or not validation.any() or (training & validation).any():
        raise RuntimeError("failed to create a scene-disjoint split")
    return training, validation, training_scenes, validation_scenes


def _positive_weight(target, torch):
    positives = target.sum()
    negatives = target.numel() - positives
    return (negatives / positives.clamp_min(1.0)).clamp(0.25, 20.0)


def _targets(data: GateTrainingData, torch, improvement_margin: float):
    original = torch.from_numpy(data.original_iou)
    candidate = torch.from_numpy(data.candidate_iou)
    delta = candidate - original
    return {
        "original": original,
        "candidate": candidate,
        "delta": delta,
        "improve": (delta > float(improvement_margin)).float(),
        "harm": (delta < -float(improvement_margin)).float(),
        "cross25": ((original < 0.25) & (candidate >= 0.25)).float(),
        "cross50": ((original < 0.50) & (candidate >= 0.50)).float(),
    }


def _loss(
    raw,
    target: Mapping[str, object],
    indices,
    *,
    torch,
    functional,
    maximum_absolute_delta: float,
    crossing_weight: float,
    harm_weight: float,
):
    selected = {name: values[indices] for name, values in target.items()}
    delta_mean = (
        torch.tanh(raw[:, 0]) * float(maximum_absolute_delta)
    )
    log_variance = raw[:, 1].clamp(-12.0, 0.0)
    delta_error = delta_mean - selected["delta"]
    delta_nll = 0.5 * (
        torch.exp(-log_variance) * delta_error.square() + log_variance
    ).mean()
    improvement_loss = functional.binary_cross_entropy_with_logits(
        raw[:, 2],
        selected["improve"],
        pos_weight=_positive_weight(selected["improve"], torch),
    )
    harm_loss = functional.binary_cross_entropy_with_logits(
        raw[:, 3],
        selected["harm"],
        pos_weight=_positive_weight(selected["harm"], torch),
    )
    original_loss = functional.smooth_l1_loss(
        torch.sigmoid(raw[:, 4]), selected["original"], beta=0.10
    )
    candidate_loss = functional.smooth_l1_loss(
        torch.sigmoid(raw[:, 5]), selected["candidate"], beta=0.10
    )
    cross25_loss = functional.binary_cross_entropy_with_logits(
        raw[:, 6],
        selected["cross25"],
        pos_weight=_positive_weight(selected["cross25"], torch),
    )
    cross50_loss = functional.binary_cross_entropy_with_logits(
        raw[:, 7],
        selected["cross50"],
        pos_weight=_positive_weight(selected["cross50"], torch),
    )
    total = (
        delta_nll
        + improvement_loss
        + float(harm_weight) * harm_loss
        + original_loss
        + candidate_loss
        + float(crossing_weight) * (cross25_loss + 2.0 * cross50_loss)
    )
    return total


def train_gate(
    data: GateTrainingData,
    output_path: Path,
    *,
    validation_fraction: float = 0.20,
    hidden_dims: Sequence[int] = (64, 32),
    epochs: int = 400,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    improvement_margin: float = 0.005,
    maximum_absolute_delta: float = 1.0,
    crossing_weight: float = 2.0,
    harm_weight: float = 2.0,
    seed: int = 1337,
    training_archives: Sequence[Mapping[str, str]] = (),
) -> Dict[str, object]:
    try:
        import torch
        from torch.nn import functional
    except ImportError as error:
        raise ImportError("training the AP50 gate requires PyTorch") from error
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if (
        not hidden_dims
        or any(isinstance(width, bool) or int(width) <= 0 for width in hidden_dims)
    ):
        raise ValueError("hidden_dims must contain positive integers")
    training, validation, train_scenes, validation_scenes = (
        scene_disjoint_split(
            data.scene_ids,
            validation_fraction=validation_fraction,
            seed=seed,
        )
    )
    train_features = data.features[training].astype(np.float64)
    feature_mean = train_features.mean(axis=0)
    feature_scale = train_features.std(axis=0)
    feature_scale = np.where(feature_scale < 1e-6, 1.0, feature_scale)
    normalized = (
        (data.features.astype(np.float64) - feature_mean) / feature_scale
    ).astype(np.float32)

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    layer_dims = (
        data.features.shape[1],
        *(int(width) for width in hidden_dims),
        AP50_SAFETY_GATE_OUTPUT_DIM,
    )
    modules = []
    for index, (input_dim, output_dim) in enumerate(
        zip(layer_dims[:-1], layer_dims[1:])
    ):
        modules.append(torch.nn.Linear(input_dim, output_dim))
        if index + 2 < len(layer_dims):
            modules.append(torch.nn.ReLU(inplace=False))
    model = torch.nn.Sequential(*modules).cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    feature_tensor = torch.from_numpy(np.ascontiguousarray(normalized))
    targets = _targets(data, torch, improvement_margin)
    training_indices = torch.from_numpy(np.flatnonzero(training))
    validation_indices = torch.from_numpy(np.flatnonzero(validation))

    best_loss = float("inf")
    best_state = None
    for _ in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        raw = model(feature_tensor[training_indices])
        loss = _loss(
            raw,
            targets,
            training_indices,
            torch=torch,
            functional=functional,
            maximum_absolute_delta=maximum_absolute_delta,
            crossing_weight=crossing_weight,
            harm_weight=harm_weight,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite AP50 gate training loss")
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_raw = model(feature_tensor[validation_indices])
            validation_loss = _loss(
                validation_raw,
                targets,
                validation_indices,
                torch=torch,
                functional=functional,
                maximum_absolute_delta=maximum_absolute_delta,
                crossing_weight=crossing_weight,
                harm_weight=harm_weight,
            )
        value = float(validation_loss)
        if value < best_loss:
            best_loss = value
            best_state = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("AP50 gate training did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)

    weights = []
    biases = []
    for layer in model:
        if isinstance(layer, torch.nn.Linear):
            weights.append(
                layer.weight.detach().numpy().T.astype(np.float32)
            )
            biases.append(layer.bias.detach().numpy().astype(np.float32))
    delta = data.candidate_iou - data.original_iou
    metadata = {
        "training_schema": TRAINING_SCHEMA,
        "training_samples": int(training.sum()),
        "validation_samples": int(validation.sum()),
        "training_scene_count": len(train_scenes),
        "validation_scene_count": len(validation_scenes),
        "training_scenes": list(train_scenes),
        "validation_scenes": list(validation_scenes),
        "improvement_margin": float(improvement_margin),
        "improvement_positives": int(np.count_nonzero(delta > improvement_margin)),
        "harm_positives": int(np.count_nonzero(delta < -improvement_margin)),
        "cross_iou25_positives": int(
            np.count_nonzero(
                (data.original_iou < 0.25) & (data.candidate_iou >= 0.25)
            )
        ),
        "cross_iou50_positives": int(
            np.count_nonzero(
                (data.original_iou < 0.50) & (data.candidate_iou >= 0.50)
            )
        ),
        "best_validation_loss": best_loss,
        "seed": int(seed),
    }
    if training_archives:
        metadata["training_archives"] = [
            {
                "path": str(item["path"]),
                "sha256": str(item["sha256"]),
            }
            for item in training_archives
        ]
    arrays = {
        "schema": np.asarray(AP50_SAFETY_GATE_SCHEMA),
        "format_version": np.asarray(
            AP50_SAFETY_GATE_FORMAT_VERSION, dtype=np.int64
        ),
        "feature_names": np.asarray(data.feature_names),
        "output_names": np.asarray(AP50_SAFETY_GATE_OUTPUT_NAMES),
        "feature_mean": feature_mean.astype(np.float32),
        "feature_scale": feature_scale.astype(np.float32),
        "maximum_absolute_delta": np.asarray(
            maximum_absolute_delta, dtype=np.float32
        ),
        "num_layers": np.asarray(len(weights), dtype=np.int64),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        arrays[f"weight_{index}"] = weight
        arrays[f"bias_{index}"] = bias
    output = Path(output_path)
    if output.suffix.lower() != ".npz":
        raise ValueError("AP50 safety checkpoint must end in .npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, output)
    loaded = load_ap50_safety_gate(output)
    if loaded.feature_names != data.feature_names:
        raise RuntimeError("exported AP50 gate feature schema changed")
    return {"output": str(output), **metadata}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--forbidden-scene-list", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--improvement-margin", type=float, default=0.005)
    parser.add_argument("--crossing-weight", type=float, default=2.0)
    parser.add_argument("--harm-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    data = load_training_data(tuple(args.inputs))
    forbidden = _read_scene_list(args.forbidden_scene_list)
    validate_forbidden_scenes(data, forbidden)
    archive_provenance = []
    for path in args.inputs:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        archive_provenance.append(
            {
                "path": str(Path(path).resolve()),
                "sha256": digest.hexdigest(),
            }
        )
    summary = train_gate(
        data,
        args.output,
        validation_fraction=args.validation_fraction,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        improvement_margin=args.improvement_margin,
        crossing_weight=args.crossing_weight,
        harm_weight=args.harm_weight,
        seed=args.seed,
        training_archives=archive_provenance,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
