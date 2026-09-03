#!/usr/bin/env python3
"""Train the isolated P1G geometry-only regression head.

P1G freezes the complete P1S sparse encoder, objectness path and raw geometry
head.  Complete snapshots are encoded once, only positive anchor rows are
retained, and a zero-initialized linear correction head is optimized with
decoder-consistent GIoU and Smooth-L1 losses.  At initialization the shared
adapter/decoder reproduces the old P1S clip/exp geometry.

The split protocol is intentionally explicit:

* ``fit`` is the only split used by the optimizer;
* ``cal`` selects the epoch lexicographically by decoded IoU>0.5 fraction and
  then mean decoded IoU;
* ``audit`` is evaluated exactly once, after the selected state is frozen;
* none of the three splits may overlap the complete ScanNet validation list.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boxfusion.p1_geometry_loss import (  # noqa: E402
    P1G_DEFAULT_ADAPTER_EPSILON,
    decode_p1g_residual_aabb,
    p1g_residual_geometry_aligned_loss,
    paired_aabb_iou,
    world_aabb_to_aligned_aabb,
)
from boxfusion.p1_geometry_refiner import (  # noqa: E402
    P1G_ARCHITECTURE,
    P1G_CHECKPOINT_SCHEMA,
    P1G_REGRESSION_ENCODING,
    P1GeometryRegressionHead,
    load_p1g_checkpoint as load_canonical_p1g_checkpoint,
    sha256_file,
)
from boxfusion.p1_spatial_residual import (  # noqa: E402
    NativeSparseResidualProposalHead,
    P1_SPATIAL_ARCHITECTURE,
)
from boxfusion.residual_proposal import (  # noqa: E402
    P1S_HEAD_SCHEMA,
    P1_FEATURE_NAMES,
)
from tools.train_p1v2_residual_head import (  # noqa: E402
    P1V2TrainingData,
    build_training_data,
    read_scene_ids,
    validate_source_collection_provenance,
)
from tools.train_p1_residual_head import (  # noqa: E402
    load_axis_alignment,
    load_gt_boxes,
)


P1G_TRAINING_SCHEMA = "boxfusion.p1g_geometry_training.v2"
P1G_SELECTION_PRIMARY = "cal_decoded_aligned_fraction_iou_gt_0p5"
P1G_SELECTION_SECONDARY = "cal_decoded_aligned_mean_iou"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def sha256_json(payload: Mapping[str, Any]) -> str:
    rendered = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _load_torch_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with old PyTorch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: checkpoint must contain a mapping")
    return dict(payload)


@dataclass(frozen=True)
class FrozenP1S:
    model: NativeSparseResidualProposalHead
    checkpoint_path: Path
    checkpoint_sha256: str
    model_config: Mapping[str, Any]
    provenance: Mapping[str, Any]


def load_frozen_p1s(
    checkpoint_path: str | os.PathLike[str],
) -> FrozenP1S:
    """Load and strictly freeze the source P1S encoder/regression head."""

    path = Path(checkpoint_path)
    payload = _load_torch_mapping(path)
    required = {
        "schema",
        "variant",
        "head_architecture",
        "target_assignment_scope",
        "model_config",
        "feature_names",
        "state_dict",
        "training_config",
        "metrics",
        "provenance",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError("P1S checkpoint missing: " + ", ".join(missing))
    if payload["schema"] != P1S_HEAD_SCHEMA:
        raise ValueError("source checkpoint is not strict P1S schema")
    if (
        payload["variant"] != "P1S"
        or payload["head_architecture"] != P1_SPATIAL_ARCHITECTURE
        or payload["target_assignment_scope"] != "snapshot_inside_only"
    ):
        raise ValueError("source checkpoint is not the P1S controlled variant")
    if tuple(payload["feature_names"]) != tuple(P1_FEATURE_NAMES):
        raise ValueError("source P1S feature schema mismatch")
    model_config = payload["model_config"]
    state_dict = payload["state_dict"]
    training_config = payload["training_config"]
    provenance = payload["provenance"]
    if not isinstance(model_config, Mapping) or not isinstance(
        state_dict, Mapping
    ):
        raise ValueError("source P1S lacks model_config/state_dict")
    if not isinstance(training_config, Mapping) or (
        training_config.get("target_assignment_scope")
        != "snapshot_inside_only"
    ):
        raise ValueError("source P1S training contract mismatch")
    if not isinstance(provenance, Mapping):
        raise ValueError("source P1S lacks provenance")
    if provenance.get("forbidden_overlap") != []:
        raise ValueError("source P1S provenance contains validation leakage")
    dataset_hash = provenance.get("dataset_fingerprint_sha256")
    if (
        not isinstance(dataset_hash, str)
        or _SHA256_PATTERN.fullmatch(dataset_hash) is None
    ):
        raise ValueError("source P1S dataset fingerprint is invalid")

    model = NativeSparseResidualProposalHead.from_model_config(model_config)
    model.load_state_dict(dict(state_dict), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if not callable(getattr(model, "encode", None)):
        raise ValueError("source P1S model lacks the frozen encode contract")
    return FrozenP1S(
        model=model,
        checkpoint_path=path.resolve(),
        checkpoint_sha256=sha256_file(path),
        model_config=dict(model_config),
        provenance=dict(provenance),
    )


@dataclass(frozen=True)
class EncodedPositiveScene:
    scene_id: str
    hidden: torch.Tensor
    frozen_p1s_raw_regression: torch.Tensor
    anchor_centers: torch.Tensor
    target_boxes_aligned: torch.Tensor
    axis_alignment: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str) or not self.scene_id:
            raise ValueError("encoded scene_id must be non-empty")
        if (
            self.hidden.ndim != 2
            or self.frozen_p1s_raw_regression.shape
            != (len(self.hidden), 6)
            or self.anchor_centers.shape != (len(self.hidden), 3)
            or self.target_boxes_aligned.shape != (len(self.hidden), 6)
            or self.axis_alignment.shape != (len(self.hidden), 4, 4)
        ):
            raise ValueError("encoded positive tensors have invalid shapes")
        tensors = (
            self.hidden,
            self.frozen_p1s_raw_regression,
            self.anchor_centers,
            self.target_boxes_aligned,
            self.axis_alignment,
        )
        if any(
            not tensor.is_floating_point()
            or tensor.device.type != "cpu"
            or not bool(torch.isfinite(tensor).all())
            for tensor in tensors
        ):
            raise ValueError("encoded positive tensors must be finite CPU floats")
        if bool(torch.any(self.target_boxes_aligned[:, 3:] <= 0.0)):
            raise ValueError("encoded targets must have positive extents")
        if len(self.axis_alignment) and not bool(
            torch.allclose(
                self.axis_alignment[:, 3, :],
                torch.tensor(
                    (0.0, 0.0, 0.0, 1.0),
                    dtype=self.axis_alignment.dtype,
                )
                .reshape(1, 4)
                .expand(len(self.axis_alignment), -1),
                rtol=1e-5,
                atol=1e-5,
            )
        ):
            raise ValueError("axis_alignment must be homogeneous")


@dataclass(frozen=True)
class EncodedPositiveDataset:
    scenes: tuple[EncodedPositiveScene, ...]
    hidden_dim: int

    def __post_init__(self) -> None:
        if int(self.hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        ids = [scene.scene_id for scene in self.scenes]
        if len(set(ids)) != len(ids):
            raise ValueError("encoded dataset contains duplicate scenes")
        if any(scene.hidden.shape[1] != self.hidden_dim for scene in self.scenes):
            raise ValueError("encoded scene hidden dimensions disagree")

    @property
    def scene_ids(self) -> tuple[str, ...]:
        return tuple(scene.scene_id for scene in self.scenes)

    @property
    def positive_count(self) -> int:
        return int(sum(len(scene.hidden) for scene in self.scenes))

    def concatenate(
        self, scene_ids: Sequence[str]
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        requested = tuple(str(scene) for scene in scene_ids)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("split scene_ids must be non-empty and unique")
        by_id = {scene.scene_id: scene for scene in self.scenes}
        missing = sorted(set(requested) - set(by_id))
        if missing:
            raise ValueError("encoded split scene is absent: " + missing[0])
        rows = [by_id[scene] for scene in requested]
        hidden = torch.cat([row.hidden for row in rows], dim=0)
        frozen_raw = torch.cat(
            [row.frozen_p1s_raw_regression for row in rows], dim=0
        )
        anchors = torch.cat([row.anchor_centers for row in rows], dim=0)
        targets = torch.cat(
            [row.target_boxes_aligned for row in rows], dim=0
        )
        alignments = torch.cat(
            [row.axis_alignment for row in rows], dim=0
        )
        if not len(hidden):
            raise ValueError("encoded split has zero positive anchors")
        return hidden, frozen_raw, anchors, targets, alignments


def precompute_positive_hidden(
    data: P1V2TrainingData,
    source_model: NativeSparseResidualProposalHead,
    *,
    gt_root: str | os.PathLike[str],
    scans_root: str | os.PathLike[str],
    device: str | torch.device = "cpu",
) -> EncodedPositiveDataset:
    """Encode positives and bind them to original aligned ScanNet GT.

    ``scene.assigned_gt`` indexes the residual-target array used by P1S.
    ``residual_ground_truth_indices`` maps that local residual index back to
    the original aligned GT row.  This avoids supervising P1G toward the
    inverse-transform-then-enclose world AABB used by the legacy P1 target.
    """

    if tuple(data.feature_names) != tuple(P1_FEATURE_NAMES):
        raise ValueError("training data feature schema differs from P1S")
    if not callable(getattr(source_model, "encode", None)):
        raise TypeError("source_model must expose encode(features, coordinates)")
    source_regression = getattr(source_model, "regression", None)
    if not isinstance(source_regression, nn.Linear):
        raise TypeError("source_model must expose its frozen nn.Linear regression")
    target_device = torch.device(device)
    source_model = source_model.to(target_device).eval()
    for parameter in source_model.parameters():
        if parameter.requires_grad:
            raise ValueError("source P1S must be fully frozen")
    hidden_dim = int(getattr(source_model, "hidden_dim", 0))
    if hidden_dim <= 0:
        raise ValueError("source P1S hidden_dim is invalid")
    gt_directory = Path(gt_root)
    scans_directory = Path(scans_root)
    if not gt_directory.is_dir() or not scans_directory.is_dir():
        raise FileNotFoundError("P1G GT/scans roots must be directories")
    summaries: dict[str, Mapping[str, Any]] = {}
    for summary in data.scene_summaries:
        if not isinstance(summary, Mapping):
            raise ValueError("P1-v2 scene summary must be a mapping")
        scene_id = summary.get("scene_id")
        if not isinstance(scene_id, str) or scene_id in summaries:
            raise ValueError("P1-v2 scene summaries have invalid scene IDs")
        summaries[scene_id] = summary
    if set(summaries) != set(data.scene_ids):
        raise ValueError("P1-v2 scene summaries do not match training scenes")

    encoded_scenes: list[EncodedPositiveScene] = []
    with torch.inference_mode():
        for scene in data.scenes:
            summary = summaries[scene.scene_id]
            residual_indices_raw = summary.get(
                "residual_ground_truth_indices"
            )
            if (
                not isinstance(residual_indices_raw, Sequence)
                or isinstance(residual_indices_raw, (str, bytes))
            ):
                raise ValueError(
                    f"{scene.scene_id}: residual GT index map is absent"
                )
            residual_indices = np.asarray(
                residual_indices_raw, dtype=np.int64
            )
            if (
                residual_indices.ndim != 1
                or len(np.unique(residual_indices)) != len(residual_indices)
                or np.any(residual_indices < 0)
            ):
                raise ValueError(
                    f"{scene.scene_id}: residual GT index map is invalid"
                )
            aligned_gt = load_gt_boxes(
                gt_directory / f"{scene.scene_id}_bbox.npy"
            )
            if len(residual_indices) and np.any(
                residual_indices >= len(aligned_gt)
            ):
                raise ValueError(
                    f"{scene.scene_id}: residual GT index is out of bounds"
                )
            alignment_numpy = load_axis_alignment(
                scans_directory, scene.scene_id
            ).astype(np.float32)
            hidden_rows: list[torch.Tensor] = []
            frozen_raw_rows: list[torch.Tensor] = []
            anchor_rows: list[torch.Tensor] = []
            target_rows: list[torch.Tensor] = []
            alignment_rows: list[torch.Tensor] = []
            for snapshot_index in range(scene.snapshot_count):
                selected = scene.snapshot_slice(snapshot_index)
                features = torch.from_numpy(scene.features[selected]).to(
                    target_device
                )
                coordinates = torch.from_numpy(
                    scene.coordinates[selected]
                ).to(target_device)
                # Encoding the complete snapshot is essential: selecting
                # positive voxels before this call would destroy sparse
                # neighbourhood context.
                encoded = source_model.encode(features, coordinates)
                frozen_raw = source_regression(encoded)
                positive_numpy = scene.objectness[selected] > 0.5
                if not np.any(positive_numpy):
                    continue
                if not np.all(scene.loss_mask[selected][positive_numpy]):
                    raise ValueError("positive P1-v2 anchors are absent from loss")
                if np.any(scene.assigned_gt[selected][positive_numpy] < 0):
                    raise ValueError("positive P1-v2 anchor lacks GT assignment")
                assigned_residual = scene.assigned_gt[selected][
                    positive_numpy
                ].astype(np.int64)
                if np.any(assigned_residual >= len(residual_indices)):
                    raise ValueError(
                        f"{scene.scene_id}: assigned residual GT is out of bounds"
                    )
                original_gt_indices = residual_indices[assigned_residual]
                positive = torch.from_numpy(positive_numpy).to(
                    target_device
                )
                anchors = torch.from_numpy(
                    scene.centers_world[selected][positive_numpy]
                ).to(target_device)
                targets = torch.from_numpy(
                    np.asarray(
                        aligned_gt[original_gt_indices, :6],
                        dtype=np.float32,
                    )
                ).to(target_device)
                alignment_batch = torch.from_numpy(
                    np.broadcast_to(
                        alignment_numpy,
                        (len(original_gt_indices), 4, 4),
                    ).copy()
                ).to(
                    target_device
                )
                hidden_rows.append(encoded[positive].detach().cpu())
                frozen_raw_rows.append(
                    frozen_raw[positive].detach().cpu()
                )
                anchor_rows.append(anchors.detach().cpu())
                target_rows.append(targets.detach().cpu())
                alignment_rows.append(alignment_batch.detach().cpu())
            encoded_scenes.append(
                EncodedPositiveScene(
                    scene_id=scene.scene_id,
                    hidden=(
                        torch.cat(hidden_rows, dim=0)
                        if hidden_rows
                        else torch.empty((0, hidden_dim))
                    ).to(dtype=torch.float32).contiguous(),
                    frozen_p1s_raw_regression=(
                        torch.cat(frozen_raw_rows, dim=0)
                        if frozen_raw_rows
                        else torch.empty((0, 6))
                    ).to(dtype=torch.float32).contiguous(),
                    anchor_centers=(
                        torch.cat(anchor_rows, dim=0)
                        if anchor_rows
                        else torch.empty((0, 3))
                    ).to(dtype=torch.float32).contiguous(),
                    target_boxes_aligned=(
                        torch.cat(target_rows, dim=0)
                        if target_rows
                        else torch.empty((0, 6))
                    ).to(dtype=torch.float32).contiguous(),
                    axis_alignment=(
                        torch.cat(alignment_rows, dim=0)
                        if alignment_rows
                        else torch.empty((0, 4, 4))
                    ).to(dtype=torch.float32).contiguous(),
                )
            )
    return EncodedPositiveDataset(
        scenes=tuple(encoded_scenes), hidden_dim=hidden_dim
    )


def validate_split_protocol(
    *,
    dataset_scene_ids: Sequence[str],
    fit_scene_ids: Sequence[str],
    cal_scene_ids: Sequence[str],
    audit_scene_ids: Sequence[str],
    full_val_scene_ids: Sequence[str],
) -> dict[str, list[str]]:
    """Validate explicit, mutually exclusive fit/cal/audit scene roles."""

    dataset = tuple(str(value) for value in dataset_scene_ids)
    fit = tuple(str(value) for value in fit_scene_ids)
    cal = tuple(str(value) for value in cal_scene_ids)
    audit = tuple(str(value) for value in audit_scene_ids)
    full_val = tuple(str(value) for value in full_val_scene_ids)
    for role, rows in (
        ("dataset", dataset),
        ("fit", fit),
        ("cal", cal),
        ("audit", audit),
        ("full_val", full_val),
    ):
        if not rows:
            raise ValueError(f"{role} scene list must be non-empty")
        if len(set(rows)) != len(rows):
            raise ValueError(f"{role} scene list contains duplicates")
    dataset_set = set(dataset)
    missing = sorted((set(fit) | set(cal) | set(audit)) - dataset_set)
    if missing:
        raise ValueError("split scene is absent from train100: " + missing[0])
    pairs = {
        "fit_cal": sorted(set(fit) & set(cal)),
        "fit_audit": sorted(set(fit) & set(audit)),
        "cal_audit": sorted(set(cal) & set(audit)),
        "fit_full_val": sorted(set(fit) & set(full_val)),
        "cal_full_val": sorted(set(cal) & set(full_val)),
        "audit_full_val": sorted(set(audit) & set(full_val)),
    }
    leaking = {name: values for name, values in pairs.items() if values}
    if leaking:
        name = sorted(leaking)[0]
        raise ValueError(f"P1G split leakage in {name}: {leaking[name][0]}")
    return pairs


def _decoder_config(
    *,
    max_center_offset: float,
    min_box_extent: float,
    max_box_extent: float,
    adapter_epsilon: float = P1G_DEFAULT_ADAPTER_EPSILON,
) -> dict[str, Any]:
    values = {
        "encoding": P1G_REGRESSION_ENCODING,
        "adapter_epsilon": float(adapter_epsilon),
        "max_center_offset": float(max_center_offset),
        "min_box_extent": float(min_box_extent),
        "max_box_extent": float(max_box_extent),
    }
    if (
        not all(
            math.isfinite(float(value))
            for key, value in values.items()
            if key != "encoding"
        )
        or values["adapter_epsilon"] <= 0.0
        or values["adapter_epsilon"] >= 0.5
        or values["max_center_offset"] <= 0.0
        or values["min_box_extent"] <= 0.0
        or values["max_box_extent"] <= values["min_box_extent"]
    ):
        raise ValueError("invalid P1G decoder bounds")
    return values


def evaluate_geometry(
    head: P1GeometryRegressionHead,
    tensors: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    *,
    decoder_config: Mapping[str, Any],
    smooth_l1_weight: float,
    smooth_l1_beta: float,
    batch_size: int,
    device: str | torch.device,
    role: str,
) -> dict[str, Any]:
    """Evaluate one encoded split without changing the head."""

    if role not in {"cal", "audit"}:
        raise ValueError("evaluation role must be cal or audit")
    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    hidden, frozen_raw, anchors, targets_aligned, alignments = tensors
    if not len(hidden):
        raise ValueError(f"{role} split has zero positive anchors")
    target_device = torch.device(device)
    head.eval()
    iou_rows: list[torch.Tensor] = []
    loss_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(hidden), int(batch_size)):
            stop = min(start + int(batch_size), len(hidden))
            batch_hidden = hidden[start:stop].to(target_device)
            batch_frozen_raw = frozen_raw[start:stop].to(target_device)
            batch_anchors = anchors[start:stop].to(target_device)
            batch_targets = targets_aligned[start:stop].to(target_device)
            batch_alignments = alignments[start:stop].to(target_device)
            correction = head(batch_hidden)
            decoded_world = decode_p1g_residual_aabb(
                batch_frozen_raw,
                correction,
                batch_anchors,
                max_center_offset=decoder_config["max_center_offset"],
                min_box_extent=decoder_config["min_box_extent"],
                max_box_extent=decoder_config["max_box_extent"],
                adapter_epsilon=decoder_config["adapter_epsilon"],
            )
            decoded_aligned = world_aabb_to_aligned_aabb(
                decoded_world, batch_alignments
            )
            iou_rows.append(
                paired_aabb_iou(
                    decoded_aligned, batch_targets
                ).detach().cpu()
            )
            loss = p1g_residual_geometry_aligned_loss(
                batch_frozen_raw,
                correction,
                batch_anchors,
                batch_targets,
                batch_alignments,
                max_center_offset=decoder_config["max_center_offset"],
                min_box_extent=decoder_config["min_box_extent"],
                max_box_extent=decoder_config["max_box_extent"],
                adapter_epsilon=decoder_config["adapter_epsilon"],
                smooth_l1_weight=smooth_l1_weight,
                smooth_l1_beta=smooth_l1_beta,
                reduction="sum",
            )
            loss_sum += float(loss.item())
    iou = torch.cat(iou_rows)
    return {
        "role": role,
        "positive_anchor_count": int(len(iou)),
        "decoded_aligned_fraction_iou_gt_0p5": float(
            torch.mean((iou > 0.5).to(torch.float64)).item()
        ),
        "decoded_aligned_mean_iou": float(
            torch.mean(iou.to(torch.float64)).item()
        ),
        "decoded_aligned_iou_q10": float(torch.quantile(iou, 0.10).item()),
        "decoded_aligned_iou_q50": float(torch.quantile(iou, 0.50).item()),
        "decoded_aligned_iou_q90": float(torch.quantile(iou, 0.90).item()),
        "aligned_geometry_loss": float(loss_sum / len(iou)),
    }


def train_geometry_refiner(
    encoded: EncodedPositiveDataset,
    *,
    fit_scene_ids: Sequence[str],
    cal_scene_ids: Sequence[str],
    audit_scene_ids: Sequence[str],
    epochs: int = 80,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_center_offset: float = 1.0,
    min_box_extent: float = 0.08,
    max_box_extent: float = 4.0,
    adapter_epsilon: float = P1G_DEFAULT_ADAPTER_EPSILON,
    smooth_l1_weight: float = 0.1,
    smooth_l1_beta: float = 0.1,
    seed: int = 1337,
    device: str | torch.device = "cpu",
) -> tuple[P1GeometryRegressionHead, dict[str, Any]]:
    """Fit on ``fit``, select only on ``cal``, then audit exactly once."""

    if isinstance(epochs, bool) or int(epochs) <= 0:
        raise ValueError("epochs must be positive")
    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    for name, value, lower_open in (
        ("learning_rate", learning_rate, True),
        ("weight_decay", weight_decay, False),
        ("smooth_l1_weight", smooth_l1_weight, False),
        ("smooth_l1_beta", smooth_l1_beta, True),
    ):
        number = float(value)
        if not math.isfinite(number) or (
            number <= 0.0 if lower_open else number < 0.0
        ):
            raise ValueError(f"{name} is invalid")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or int(seed) != seed
    ):
        raise ValueError("seed must be an integer")

    fit_tensors = encoded.concatenate(fit_scene_ids)
    cal_tensors = encoded.concatenate(cal_scene_ids)
    audit_tensors = encoded.concatenate(audit_scene_ids)
    decoder = _decoder_config(
        max_center_offset=max_center_offset,
        min_box_extent=min_box_extent,
        max_box_extent=max_box_extent,
        adapter_epsilon=adapter_epsilon,
    )
    target_device = torch.device(device)
    torch.manual_seed(int(seed))
    if target_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True)
    head = P1GeometryRegressionHead(hidden_dim=encoded.hidden_dim)
    head.to(target_device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    (
        fit_hidden,
        fit_frozen_raw,
        fit_anchors,
        fit_targets,
        fit_alignments,
    ) = fit_tensors
    generator = torch.Generator(device="cpu")
    history: list[dict[str, Any]] = []
    best_key = (-math.inf, -math.inf)
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    best_cal: dict[str, Any] | None = None

    for epoch in range(int(epochs)):
        head.train()
        generator.manual_seed(int(seed) + 104729 * epoch)
        order = torch.randperm(len(fit_hidden), generator=generator)
        epoch_loss_sum = 0.0
        for start in range(0, len(order), int(batch_size)):
            selected = order[start : start + int(batch_size)]
            hidden = fit_hidden[selected].to(target_device)
            frozen_raw = fit_frozen_raw[selected].to(target_device)
            anchors = fit_anchors[selected].to(target_device)
            targets = fit_targets[selected].to(target_device)
            alignments = fit_alignments[selected].to(target_device)
            optimizer.zero_grad(set_to_none=True)
            correction = head(hidden)
            loss = p1g_residual_geometry_aligned_loss(
                frozen_raw,
                correction,
                anchors,
                targets,
                alignments,
                max_center_offset=decoder["max_center_offset"],
                min_box_extent=decoder["min_box_extent"],
                max_box_extent=decoder["max_box_extent"],
                adapter_epsilon=decoder["adapter_epsilon"],
                smooth_l1_weight=smooth_l1_weight,
                smooth_l1_beta=smooth_l1_beta,
                reduction="mean",
            )
            loss.backward()
            optimizer.step()
            epoch_loss_sum += float(loss.detach().item()) * len(selected)
        cal = evaluate_geometry(
            head,
            cal_tensors,
            decoder_config=decoder,
            smooth_l1_weight=smooth_l1_weight,
            smooth_l1_beta=smooth_l1_beta,
            batch_size=batch_size,
            device=target_device,
            role="cal",
        )
        key = (
            float(cal["decoded_aligned_fraction_iou_gt_0p5"]),
            float(cal["decoded_aligned_mean_iou"]),
        )
        history.append(
            {
                "epoch": int(epoch),
                "fit_geometry_loss": float(
                    epoch_loss_sum / len(fit_hidden)
                ),
                "calibration": cal,
                "selection_key": [key[0], key[1]],
            }
        )
        if key > best_key:
            best_key = key
            best_epoch = int(epoch)
            best_cal = dict(cal)
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in head.state_dict().items()
            }
    if best_state is None or best_cal is None:
        raise RuntimeError("P1G training produced no selected state")
    head.load_state_dict(best_state, strict=True)
    head.eval()
    audit = evaluate_geometry(
        head,
        audit_tensors,
        decoder_config=decoder,
        smooth_l1_weight=smooth_l1_weight,
        smooth_l1_beta=smooth_l1_beta,
        batch_size=batch_size,
        device=target_device,
        role="audit",
    )
    metrics = {
        "selection": {
            "primary": P1G_SELECTION_PRIMARY,
            "secondary": P1G_SELECTION_SECONDARY,
            "comparison": "lexicographic_max",
            "best_epoch": int(best_epoch),
            "best_key": [float(best_key[0]), float(best_key[1])],
            "audit_used_for_selection": False,
        },
        "fit_positive_anchor_count": int(len(fit_hidden)),
        "cal_positive_anchor_count": int(len(cal_tensors[0])),
        "audit_positive_anchor_count": int(len(audit_tensors[0])),
        "best_calibration": best_cal,
        "audit": audit,
        "audit_evaluation_count": 1,
        "history": history,
    }
    return head.cpu(), metrics


def save_p1g_checkpoint(
    output_path: str | os.PathLike[str],
    *,
    head: P1GeometryRegressionHead,
    decoder_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    source_p1s: Mapping[str, Any],
    provenance: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Path:
    output = Path(output_path)
    if output.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("P1G checkpoint must end in .pt or .pth")
    payload = {
        "schema": P1G_CHECKPOINT_SCHEMA,
        "architecture": P1G_ARCHITECTURE,
        "regression_encoding": P1G_REGRESSION_ENCODING,
        "initialization": (
            "zero_residual_correction_function_preserving_v2"
        ),
        "observer_only": True,
        "uses_ground_truth": False,
        "class_agnostic": True,
        "semantic_features": False,
        "feature_names": list(P1_FEATURE_NAMES),
        "model_config": head.model_config(
            max_center_offset=decoder_config["max_center_offset"],
            min_box_extent=decoder_config["min_box_extent"],
            max_box_extent=decoder_config["max_box_extent"],
            adapter_epsilon=decoder_config["adapter_epsilon"],
        ),
        "decoder_config": dict(decoder_config),
        "state_dict": {
            name: value.detach().cpu().clone()
            for name, value in head.state_dict().items()
        },
        "training_config": dict(training_config),
        "source_p1s": dict(source_p1s),
        "provenance": dict(provenance),
        "metrics": dict(metrics),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    # Round-trip the exact persisted artifact through the canonical runtime
    # loader; the trainer does not maintain a second checkpoint contract.
    load_canonical_p1g_checkpoint(
        output,
        expected_p1s_checkpoint_sha256=source_p1s[
            "checkpoint_sha256"
        ],
    )
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-root", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--scans-root", required=True, type=Path)
    parser.add_argument("--train-scene-list", required=True, type=Path)
    parser.add_argument("--fit-scene-list", required=True, type=Path)
    parser.add_argument("--cal-scene-list", required=True, type=Path)
    parser.add_argument("--audit-scene-list", required=True, type=Path)
    parser.add_argument("--full-val-scene-list", required=True, type=Path)
    parser.add_argument("--b6-checkpoint", required=True, type=Path)
    parser.add_argument("--source-p1-checkpoint", required=True, type=Path)
    parser.add_argument("--p1s-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--covered-iou", type=float, default=0.15)
    parser.add_argument("--assignment-topk", type=int, default=6)
    parser.add_argument("--negative-ratio", type=float, default=8.0)
    parser.add_argument(
        "--maximum-loss-voxels-per-snapshot", type=int, default=4096
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-center-offset", type=float, default=1.0)
    parser.add_argument("--min-box-extent", type=float, default=0.08)
    parser.add_argument("--max-box-extent", type=float, default=4.0)
    parser.add_argument(
        "--adapter-epsilon",
        type=float,
        default=P1G_DEFAULT_ADAPTER_EPSILON,
    )
    parser.add_argument("--smooth-l1-weight", type=float, default=0.1)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    train_scenes = read_scene_ids(args.train_scene_list, role="train100")
    fit_scenes = read_scene_ids(args.fit_scene_list, role="fit")
    cal_scenes = read_scene_ids(args.cal_scene_list, role="calibration")
    audit_scenes = read_scene_ids(args.audit_scene_list, role="audit")
    full_val_scenes = read_scene_ids(
        args.full_val_scene_list, role="complete ScanNet validation"
    )
    overlaps = validate_split_protocol(
        dataset_scene_ids=train_scenes,
        fit_scene_ids=fit_scenes,
        cal_scene_ids=cal_scenes,
        audit_scene_ids=audit_scenes,
        full_val_scene_ids=full_val_scenes,
    )
    source_binding = validate_source_collection_provenance(
        args.source_p1_checkpoint,
        scenes=train_scenes,
        diagnostics_root=args.diagnostics_root,
        prediction_root=args.prediction_root,
        gt_root=args.gt_root,
        train_scene_list=args.train_scene_list,
        forbidden_scene_list=args.full_val_scene_list,
        b6_checkpoint=args.b6_checkpoint,
    )
    data = build_training_data(
        scenes=train_scenes,
        diagnostics_root=args.diagnostics_root,
        prediction_root=args.prediction_root,
        gt_root=args.gt_root,
        scans_root=args.scans_root,
        covered_iou=args.covered_iou,
        assignment_topk=args.assignment_topk,
        negative_ratio=args.negative_ratio,
        maximum_loss_voxels_per_snapshot=(
            args.maximum_loss_voxels_per_snapshot
        ),
        seed=args.seed,
    )
    frozen = load_frozen_p1s(args.p1s_checkpoint)
    source_dataset_hash = frozen.provenance[
        "dataset_fingerprint_sha256"
    ]
    if source_dataset_hash != data.dataset_fingerprint_sha256:
        raise ValueError(
            "P1G data fingerprint differs from the frozen P1S source"
        )
    source_b6_hash = frozen.provenance.get("b6_checkpoint_sha256")
    observed_b6_hash = sha256_file(args.b6_checkpoint)
    if source_b6_hash != observed_b6_hash:
        raise ValueError("P1S and P1G use different frozen B6 checkpoints")
    source_train_scenes = frozen.provenance.get("source_train_scene_ids")
    if (
        not isinstance(source_train_scenes, Sequence)
        or isinstance(source_train_scenes, (str, bytes))
        or set(source_train_scenes) != set(train_scenes)
    ):
        raise ValueError("P1S source train100 scenes differ from P1G")

    encoded = precompute_positive_hidden(
        data,
        frozen.model,
        gt_root=args.gt_root,
        scans_root=args.scans_root,
        device=args.device,
    )
    head, metrics = train_geometry_refiner(
        encoded,
        fit_scene_ids=fit_scenes,
        cal_scene_ids=cal_scenes,
        audit_scene_ids=audit_scenes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_center_offset=args.max_center_offset,
        min_box_extent=args.min_box_extent,
        max_box_extent=args.max_box_extent,
        adapter_epsilon=args.adapter_epsilon,
        smooth_l1_weight=args.smooth_l1_weight,
        smooth_l1_beta=args.smooth_l1_beta,
        seed=args.seed,
        device=args.device,
    )
    decoder = _decoder_config(
        max_center_offset=args.max_center_offset,
        min_box_extent=args.min_box_extent,
        max_box_extent=args.max_box_extent,
        adapter_epsilon=args.adapter_epsilon,
    )
    training_config = {
        "schema": P1G_TRAINING_SCHEMA,
        "frozen_encoder": True,
        "frozen_objectness": True,
        "frozen_p1s_raw_regression": True,
        "function_preserving_initialization": True,
        "trainable_parameters": [
            "correction.weight",
            "correction.bias",
        ],
        "loss": (
            "aligned_one_minus_giou_plus_weighted_decoded_smooth_l1_"
            "on_frozen_p1s_plus_residual_correction"
        ),
        "regression_encoding": P1G_REGRESSION_ENCODING,
        "adapter_epsilon": float(args.adapter_epsilon),
        "target_frame": "original_scannet_aligned_aabb",
        "axis_alignment_runtime_input": False,
        "smooth_l1_weight": float(args.smooth_l1_weight),
        "smooth_l1_beta": float(args.smooth_l1_beta),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "deterministic_algorithms": True,
        "fit_role": "optimizer_only",
        "cal_role": "epoch_selection_only",
        "audit_role": "one_evaluation_after_best_state_frozen",
    }
    source_p1s = {
        "checkpoint": str(frozen.checkpoint_path),
        "checkpoint_sha256": frozen.checkpoint_sha256,
        "schema": P1S_HEAD_SCHEMA,
        "model_config_sha256": sha256_json(frozen.model_config),
        "dataset_fingerprint_sha256": source_dataset_hash,
    }
    selected = set(fit_scenes) | set(cal_scenes) | set(audit_scenes)
    provenance = {
        # Canonical runtime-loader fields.
        "p1s_checkpoint_sha256": frozen.checkpoint_sha256,
        "forbidden_overlap": [],
        "forbidden_scene_list_sha256": sha256_file(
            args.full_val_scene_list
        ),
        "dataset_fingerprint_sha256": data.dataset_fingerprint_sha256,
        "train_scene_list": str(args.train_scene_list.resolve()),
        "train_scene_list_sha256": sha256_file(args.train_scene_list),
        "train_scene_ids": list(train_scenes),
        "fit_scene_list": str(args.fit_scene_list.resolve()),
        "fit_scene_list_sha256": sha256_file(args.fit_scene_list),
        "fit_scene_ids": list(fit_scenes),
        "cal_scene_list": str(args.cal_scene_list.resolve()),
        "cal_scene_list_sha256": sha256_file(args.cal_scene_list),
        "cal_scene_ids": list(cal_scenes),
        "audit_scene_list": str(args.audit_scene_list.resolve()),
        "audit_scene_list_sha256": sha256_file(args.audit_scene_list),
        "audit_scene_ids": list(audit_scenes),
        "full_val_scene_list": str(args.full_val_scene_list.resolve()),
        "full_val_scene_list_sha256": sha256_file(
            args.full_val_scene_list
        ),
        "full_val_scene_ids": list(full_val_scenes),
        "full_val_scene_count": int(len(full_val_scenes)),
        "split_overlaps": overlaps,
        "unused_train_scene_ids": sorted(set(train_scenes) - selected),
        "b6_checkpoint": str(args.b6_checkpoint.resolve()),
        "b6_checkpoint_sha256": observed_b6_hash,
        "source_collection_binding": source_binding,
        "diagnostics_root": str(args.diagnostics_root.resolve()),
        "prediction_root": str(args.prediction_root.resolve()),
        "gt_root": str(args.gt_root.resolve()),
        "scans_root": str(args.scans_root.resolve()),
    }
    output = save_p1g_checkpoint(
        args.output,
        head=head,
        decoder_config=decoder,
        training_config=training_config,
        source_p1s=source_p1s,
        provenance=provenance,
        metrics=metrics,
    )
    summary = {
        "schema": P1G_CHECKPOINT_SCHEMA,
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "source_p1s": source_p1s,
        "dataset_fingerprint_sha256": data.dataset_fingerprint_sha256,
        "train_scene_count": int(len(train_scenes)),
        "fit_scene_count": int(len(fit_scenes)),
        "cal_scene_count": int(len(cal_scenes)),
        "audit_scene_count": int(len(audit_scenes)),
        "encoded_positive_count": int(encoded.positive_count),
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
