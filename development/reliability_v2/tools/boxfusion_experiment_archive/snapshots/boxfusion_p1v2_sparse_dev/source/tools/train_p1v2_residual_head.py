#!/usr/bin/env python3
"""Train the controlled P1R/P1S residual-proposal ablation.

This trainer deliberately consumes the already frozen, read-only P1
``collect`` diagnostics produced in ``boxfusion_p1_dev``.  It does not run
BoxFusion again and it never changes the frozen predictions.

The two variants differ in exactly one model component:

``P1R``
    Per-snapshot, inside-only targets with the legacy per-voxel MLP.

``P1S``
    The identical targets and loss masks with the native sparse-context head.

Every snapshot is forwarded with its complete voxel context.  Negative
subsampling only creates a loss mask; it never removes voxels before the
model, which is essential for a fair spatial-context ablation.

Only trusted, locally produced BoxFusion pickle prediction files should be
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

from boxfusion.residual_proposal import (  # noqa: E402
    P1_DIAGNOSTIC_SCHEMA,
    P1_FEATURE_DIM,
    P1_FEATURE_NAMES,
    P1_HEAD_SCHEMA,
    ResidualVoxelProposalHead,
)
from boxfusion.p1_spatial_residual import (  # noqa: E402
    P1_SPATIAL_ARCHITECTURE,
    NativeSparseResidualProposalHead,
)
from tools.train_p1_residual_head import (  # noqa: E402
    _file_sha256,
    load_axis_alignment,
    load_gt_boxes,
    load_prediction_corners,
    residual_gt_world_boxes,
)


P1R_CHECKPOINT_SCHEMA = "boxfusion.p1r_snapshot_residual_head.v1"
P1S_CHECKPOINT_SCHEMA = "boxfusion.p1s_native_sparse_residual_head.v1"
TRAINING_SCHEMA = "boxfusion.p1v2_snapshot_inside_training.v1"
TARGET_ASSIGNMENT_SCOPE = "snapshot_inside_only"
P1R_ARCHITECTURE = "per_voxel_mlp"
VARIANT_SCHEMAS = {
    "P1R": P1R_CHECKPOINT_SCHEMA,
    "P1S": P1S_CHECKPOINT_SCHEMA,
}
VARIANT_ARCHITECTURES = {
    "P1R": P1R_ARCHITECTURE,
    "P1S": P1_SPATIAL_ARCHITECTURE,
}


@dataclass(frozen=True)
class SnapshotTargets:
    """Targets for one complete snapshot."""

    objectness: np.ndarray
    regression: np.ndarray
    assigned_gt: np.ndarray


@dataclass(frozen=True)
class SceneTrainingContext:
    """All frozen observer inputs and offline labels for one scene."""

    scene_id: str
    features: np.ndarray
    coordinates: np.ndarray
    centers_world: np.ndarray
    offsets: np.ndarray
    frame_ids: np.ndarray
    provider_steps: np.ndarray
    objectness: np.ndarray
    regression: np.ndarray
    assigned_gt: np.ndarray
    loss_mask: np.ndarray
    feature_names: tuple[str, ...]
    voxel_size: float
    diagnostic_path: Path

    @property
    def snapshot_count(self) -> int:
        return int(len(self.offsets) - 1)

    def snapshot_slice(self, snapshot_index: int) -> slice:
        index = int(snapshot_index)
        if index < 0 or index >= self.snapshot_count:
            raise IndexError(index)
        return slice(int(self.offsets[index]), int(self.offsets[index + 1]))


@dataclass(frozen=True)
class P1V2TrainingData:
    """Scene-preserving dataset used by both controlled variants."""

    scenes: tuple[SceneTrainingContext, ...]
    feature_names: tuple[str, ...]
    scene_summaries: tuple[Mapping[str, Any], ...]
    dataset_fingerprint_sha256: str

    @property
    def scene_ids(self) -> tuple[str, ...]:
        return tuple(scene.scene_id for scene in self.scenes)

    @property
    def voxel_count(self) -> int:
        return int(sum(len(scene.features) for scene in self.scenes))

    @property
    def loss_sample_count(self) -> int:
        return int(sum(np.count_nonzero(scene.loss_mask) for scene in self.scenes))

    @property
    def positive_count(self) -> int:
        return int(
            sum(
                np.count_nonzero(
                    scene.loss_mask & (scene.objectness > 0.5)
                )
                for scene in self.scenes
            )
        )


def normalize_variant(value: str) -> str:
    variant = str(value).strip().upper()
    if variant not in VARIANT_SCHEMAS:
        raise ValueError("variant must be P1R or P1S")
    return variant


def read_scene_ids(
    path: str | os.PathLike[str], *, role: str
) -> tuple[str, ...]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"{role} scene list not found: {source}")
    scenes = tuple(
        row.strip()
        for row in source.read_text(encoding="utf-8").splitlines()
        if row.strip()
    )
    if not scenes:
        raise ValueError(f"{role} scene list is empty: {source}")
    if len(set(scenes)) != len(scenes):
        raise ValueError(f"{role} scene list contains duplicates: {source}")
    for scene_id in scenes:
        if not (
            len(scene_id) == 12
            and scene_id.startswith("scene")
            and scene_id[5:9].isdigit()
            and scene_id[9] == "_"
            and scene_id[10:].isdigit()
        ):
            raise ValueError(
                f"invalid ScanNet scene id in {role} list: {scene_id!r}"
            )
    return scenes


def validate_external_split(
    train_scenes: Iterable[str], forbidden_scenes: Iterable[str]
) -> tuple[str, ...]:
    train = tuple(str(scene) for scene in train_scenes)
    forbidden = frozenset(str(scene) for scene in forbidden_scenes)
    overlap = sorted(set(train) & forbidden)
    if overlap:
        raise ValueError(
            "P1-v2 train/forbidden scene leakage: "
            + ", ".join(overlap[:16])
        )
    return train


def validate_source_collection_provenance(
    source_checkpoint: str | os.PathLike[str],
    *,
    scenes: Sequence[str],
    diagnostics_root: str | os.PathLike[str],
    prediction_root: str | os.PathLike[str],
    gt_root: str | os.PathLike[str],
    train_scene_list: str | os.PathLike[str],
    forbidden_scene_list: str | os.PathLike[str],
    b6_checkpoint: str | os.PathLike[str],
) -> dict[str, Any]:
    """Bind frozen collect inputs to the legacy P1/B6 provenance witness.

    The historical collect driver did not write a run manifest.  Its trained
    P1 checkpoint does contain the B6 hash plus the exact diagnostic,
    prediction, and GT hashes for every training scene.  Requiring those
    hashes prevents a caller from pairing arbitrary collect artifacts with a
    different B6 checkpoint and then self-reporting that checkpoint's hash.
    """

    checkpoint_path = Path(source_checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"source P1 provenance checkpoint not found: {checkpoint_path}"
        )
    try:
        payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema") != (
        P1_HEAD_SCHEMA
    ):
        raise ValueError("source P1 provenance checkpoint schema mismatch")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("source P1 checkpoint lacks provenance")

    requested_scenes = tuple(str(scene) for scene in scenes)
    source_scenes = provenance.get("train_scene_ids")
    if (
        not isinstance(source_scenes, Sequence)
        or isinstance(source_scenes, (str, bytes))
        or tuple(str(scene) for scene in source_scenes) != requested_scenes
    ):
        raise ValueError(
            "source P1 checkpoint scene order disagrees with training list"
        )
    expected_scalars = {
        "b6_checkpoint_sha256": _file_sha256(Path(b6_checkpoint)),
        "train_scene_list_sha256": _file_sha256(Path(train_scene_list)),
        "forbidden_scene_list_sha256": _file_sha256(
            Path(forbidden_scene_list)
        ),
    }
    for name, expected in expected_scalars.items():
        observed = provenance.get(name)
        if not isinstance(observed, str) or observed.lower() != expected:
            raise ValueError(
                f"source P1 provenance {name} disagrees with current input"
            )
    if provenance.get("forbidden_overlap") != []:
        raise ValueError("source P1 provenance contains forbidden overlap")

    summaries = provenance.get("scene_summaries")
    if (
        not isinstance(summaries, Sequence)
        or isinstance(summaries, (str, bytes))
    ):
        raise ValueError("source P1 provenance lacks scene summaries")
    by_scene: dict[str, Mapping[str, Any]] = {}
    for row in summaries:
        if not isinstance(row, Mapping):
            raise ValueError("source P1 scene summary must be a mapping")
        scene_id = row.get("scene_id")
        if (
            not isinstance(scene_id, str)
            or scene_id in by_scene
            or scene_id not in requested_scenes
        ):
            raise ValueError("source P1 scene summaries are invalid")
        by_scene[scene_id] = row
    if set(by_scene) != set(requested_scenes):
        raise ValueError("source P1 scene summary set is incomplete")

    diagnostics = Path(diagnostics_root)
    predictions = Path(prediction_root)
    ground_truth = Path(gt_root)
    for scene_id in requested_scenes:
        row = by_scene[scene_id]
        actual = {
            "diagnostic_sha256": _file_sha256(
                diagnostics / f"{scene_id}_tracks.npz"
            ),
            "prediction_sha256": _file_sha256(
                predictions / f"{scene_id}_boxes.pkl"
            ),
            "ground_truth_sha256": _file_sha256(
                ground_truth / f"{scene_id}_bbox.npy"
            ),
        }
        for name, digest in actual.items():
            observed = row.get(name)
            if not isinstance(observed, str) or observed.lower() != digest:
                raise ValueError(
                    f"{scene_id}: source P1 {name} binding mismatch"
                )
    return {
        "verified": True,
        "schema": str(payload["schema"]),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "b6_checkpoint_sha256": expected_scalars[
            "b6_checkpoint_sha256"
        ],
        "scene_count": len(requested_scenes),
    }


def deterministic_scene_partition(
    scene_ids: Sequence[str],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    scenes = tuple(str(scene) for scene in scene_ids)
    if len(scenes) < 2 or len(set(scenes)) != len(scenes):
        raise ValueError(
            "P1-v2 optimization split needs at least two unique scenes"
        )
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must lie strictly in (0,1)")
    ordered = np.asarray(sorted(scenes), dtype=np.str_)
    rng = np.random.default_rng(int(seed))
    shuffled = ordered[rng.permutation(len(ordered))]
    validation_count = min(
        max(1, int(round(len(ordered) * float(validation_fraction)))),
        len(ordered) - 1,
    )
    validation = tuple(sorted(str(x) for x in shuffled[:validation_count]))
    training = tuple(sorted(str(x) for x in shuffled[validation_count:]))
    if set(training) & set(validation):
        raise RuntimeError("internal scene split is not disjoint")
    return training, validation


def _scalar_text(
    archive: Mapping[str, np.ndarray], name: str, path: Path
) -> str:
    if name not in archive:
        raise ValueError(f"{path}: missing {name}")
    value = np.asarray(archive[name])
    if value.shape != () or value.dtype.hasobject:
        raise ValueError(f"{path}: {name} must be a scalar string")
    item = value.item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    if not isinstance(item, str) or not item:
        raise ValueError(f"{path}: {name} must be a non-empty string")
    return item


def _scalar_bool(
    archive: Mapping[str, np.ndarray], name: str, path: Path
) -> bool:
    if name not in archive:
        raise ValueError(f"{path}: missing {name}")
    value = np.asarray(archive[name])
    if value.shape != () or value.dtype != np.dtype(bool):
        raise ValueError(f"{path}: {name} must be a Boolean scalar")
    return bool(value.item())


def _scalar_integer(
    archive: Mapping[str, np.ndarray], name: str, path: Path
) -> int:
    if name not in archive:
        raise ValueError(f"{path}: missing {name}")
    value = np.asarray(archive[name])
    if (
        value.shape != ()
        or not np.issubdtype(value.dtype, np.integer)
        or np.issubdtype(value.dtype, np.bool_)
    ):
        raise ValueError(f"{path}: {name} must be an integer scalar")
    return int(value.item())


def _feature_names(
    archive: Mapping[str, np.ndarray], path: Path
) -> tuple[str, ...]:
    if "p1_feature_names" not in archive:
        raise ValueError(f"{path}: missing p1_feature_names")
    raw = np.asarray(archive["p1_feature_names"])
    if raw.dtype.hasobject:
        raise TypeError(f"{path}: p1_feature_names cannot use object dtype")
    if raw.shape == ():
        parsed = json.loads(str(raw.item()))
        names = tuple(str(value) for value in parsed)
    elif raw.ndim == 1:
        names = tuple(str(value) for value in raw.tolist())
    else:
        raise ValueError(f"{path}: p1_feature_names must be scalar JSON or [F]")
    if names != tuple(P1_FEATURE_NAMES):
        raise ValueError(f"{path}: P1 feature schema disagrees with runtime")
    return names


def load_scene_context(
    path: str | os.PathLike[str], *, expected_scene_id: str | None = None
) -> SceneTrainingContext:
    """Strictly load legacy P1 collect diagnostics without pickle."""

    diagnostic_path = Path(path)
    if not diagnostic_path.is_file():
        raise FileNotFoundError(diagnostic_path)
    with np.load(diagnostic_path, allow_pickle=False) as archive_obj:
        archive = {
            key: np.array(archive_obj[key], copy=True)
            for key in archive_obj.files
        }

    required = {
        "scene_id",
        "p1_schema",
        "p1_stage",
        "p1_profile",
        "p1_enabled",
        "p1_observer_only",
        "p1_uses_ground_truth",
        "p1_mutation_enabled",
        "p1_applied_count",
        "p1_complete",
        "p1_class_agnostic",
        "p1_regression_dim",
        "p1_config_json",
        "p1_feature_names",
        "p1_step_frame_ids",
        "p1_step_provider_steps",
        "p1_step_voxel_counts",
        "p1_voxel_offsets",
        "p1_voxel_coords",
        "p1_voxel_centers",
        "p1_voxel_features",
    }
    missing = sorted(required - set(archive))
    if missing:
        raise ValueError(f"{diagnostic_path}: missing fields {missing}")
    expected_text = {
        "p1_schema": P1_DIAGNOSTIC_SCHEMA,
        "p1_stage": "P1",
        "p1_profile": "p1_residual_proposal_observer",
    }
    for name, expected in expected_text.items():
        observed = _scalar_text(archive, name, diagnostic_path)
        if observed != expected:
            raise ValueError(
                f"{diagnostic_path}: {name}={observed!r}, expected {expected!r}"
            )
    expected_bools = {
        "p1_enabled": True,
        "p1_observer_only": True,
        "p1_uses_ground_truth": False,
        "p1_mutation_enabled": False,
        "p1_complete": True,
        "p1_class_agnostic": True,
    }
    for name, expected in expected_bools.items():
        observed = _scalar_bool(archive, name, diagnostic_path)
        if observed != expected:
            raise ValueError(
                f"{diagnostic_path}: unsafe {name}={observed}, "
                f"expected {expected}"
            )
    if _scalar_integer(archive, "p1_applied_count", diagnostic_path) != 0:
        raise ValueError(f"{diagnostic_path}: observer changed output")
    if _scalar_integer(archive, "p1_regression_dim", diagnostic_path) != 6:
        raise ValueError(f"{diagnostic_path}: P1 regression is not 6-D")
    # The frozen legacy diagnostics predate these optional audit fields.
    if "p1_reads_semantic_labels" in archive and _scalar_bool(
        archive, "p1_reads_semantic_labels", diagnostic_path
    ):
        raise ValueError(f"{diagnostic_path}: diagnostic reads semantic labels")

    scene_id = _scalar_text(archive, "scene_id", diagnostic_path)
    if expected_scene_id is not None and scene_id != expected_scene_id:
        raise ValueError(
            f"{diagnostic_path}: scene_id {scene_id!r} != "
            f"{expected_scene_id!r}"
        )
    try:
        config = json.loads(
            _scalar_text(archive, "p1_config_json", diagnostic_path)
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{diagnostic_path}: invalid p1_config_json"
        ) from error
    if not isinstance(config, Mapping):
        raise ValueError(f"{diagnostic_path}: p1_config_json must be a mapping")
    if (
        config.get("mode") != "collect"
        or config.get("collect_voxel_inputs") is not True
        or config.get("observer_only") is not True
        or config.get("mutate") is not False
    ):
        raise ValueError(f"{diagnostic_path}: unsafe P1 collect configuration")
    voxel_size = float(config.get("voxel_size", math.nan))
    if not math.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError(f"{diagnostic_path}: invalid P1 voxel_size")

    features = np.asarray(archive["p1_voxel_features"])
    coordinates = np.asarray(archive["p1_voxel_coords"])
    centers = np.asarray(archive["p1_voxel_centers"])
    offsets = np.asarray(archive["p1_voxel_offsets"])
    frame_ids = np.asarray(archive["p1_step_frame_ids"])
    provider_steps = np.asarray(archive["p1_step_provider_steps"])
    step_counts = np.asarray(archive["p1_step_voxel_counts"])
    if (
        features.ndim != 2
        or features.shape[1] != P1_FEATURE_DIM
        or not np.issubdtype(features.dtype, np.floating)
        or not np.isfinite(features).all()
    ):
        raise ValueError(
            f"{diagnostic_path}: p1_voxel_features must be finite [V,14]"
        )
    voxel_count = len(features)
    if (
        coordinates.shape != (voxel_count, 3)
        or not np.issubdtype(coordinates.dtype, np.integer)
    ):
        raise ValueError(
            f"{diagnostic_path}: p1_voxel_coords must be integer [V,3]"
        )
    if (
        centers.shape != (voxel_count, 3)
        or not np.issubdtype(centers.dtype, np.floating)
        or not np.isfinite(centers).all()
    ):
        raise ValueError(
            f"{diagnostic_path}: p1_voxel_centers must be finite [V,3]"
        )
    if offsets.ndim != 1 or not np.issubdtype(offsets.dtype, np.integer):
        raise ValueError(f"{diagnostic_path}: offsets must be integer [S+1]")
    offsets = np.asarray(offsets, dtype=np.int64)
    if (
        len(offsets) < 1
        or int(offsets[0]) != 0
        or int(offsets[-1]) != voxel_count
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError(f"{diagnostic_path}: invalid ragged offsets")
    snapshot_count = len(offsets) - 1
    for name, values in (
        ("p1_step_frame_ids", frame_ids),
        ("p1_step_provider_steps", provider_steps),
        ("p1_step_voxel_counts", step_counts),
    ):
        if (
            values.shape != (snapshot_count,)
            or not np.issubdtype(values.dtype, np.integer)
        ):
            raise ValueError(
                f"{diagnostic_path}: {name} must be integer [S]"
            )
    if not np.array_equal(
        np.diff(offsets), np.asarray(step_counts, dtype=np.int64)
    ):
        raise ValueError(
            f"{diagnostic_path}: step voxel counts disagree with offsets"
        )
    expected_centers = (
        np.asarray(coordinates, dtype=np.float64) + 0.5
    ) * voxel_size
    if not np.allclose(centers, expected_centers, atol=1e-5, rtol=1e-6):
        raise ValueError(
            f"{diagnostic_path}: coordinates/centers/voxel_size disagree"
        )
    for snapshot_index in range(snapshot_count):
        start, stop = int(offsets[snapshot_index]), int(
            offsets[snapshot_index + 1]
        )
        rows = np.asarray(coordinates[start:stop], dtype=np.int64)
        if len(rows) and len(np.unique(rows, axis=0)) != len(rows):
            raise ValueError(
                f"{diagnostic_path}: duplicate voxel coordinate in "
                f"snapshot {snapshot_index}"
            )

    names = _feature_names(archive, diagnostic_path)
    zeros = np.zeros(voxel_count, dtype=np.float32)
    return SceneTrainingContext(
        scene_id=scene_id,
        features=np.ascontiguousarray(features, dtype=np.float32),
        coordinates=np.ascontiguousarray(coordinates, dtype=np.int32),
        centers_world=np.ascontiguousarray(centers, dtype=np.float32),
        offsets=offsets,
        frame_ids=np.ascontiguousarray(frame_ids, dtype=np.int64),
        provider_steps=np.ascontiguousarray(provider_steps, dtype=np.int64),
        objectness=zeros.copy(),
        regression=np.zeros((voxel_count, 6), dtype=np.float32),
        assigned_gt=np.full(voxel_count, -1, dtype=np.int64),
        loss_mask=np.zeros(voxel_count, dtype=bool),
        feature_names=names,
        voxel_size=voxel_size,
        diagnostic_path=diagnostic_path,
    )


def assign_snapshot_inside_targets(
    voxel_centers: np.ndarray,
    target_boxes: np.ndarray,
    *,
    topk: int,
) -> SnapshotTargets:
    """Assign only voxels geometrically inside a target in this snapshot."""

    centers = np.asarray(voxel_centers, dtype=np.float64)
    boxes = np.asarray(target_boxes, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("voxel_centers must have shape [V,3]")
    if boxes.size == 0:
        boxes = np.empty((0, 6), dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 6:
        raise ValueError("target_boxes must have shape [G,6]")
    if (
        not np.isfinite(centers).all()
        or not np.isfinite(boxes).all()
        or (len(boxes) and np.any(boxes[:, 3:] <= 0.0))
    ):
        raise ValueError("target assignment inputs are invalid")
    if isinstance(topk, bool) or int(topk) <= 0:
        raise ValueError("topk must be a positive integer")
    topk = int(topk)
    objectness = np.zeros(len(centers), dtype=np.float32)
    regression = np.zeros((len(centers), 6), dtype=np.float32)
    assigned = np.full(len(centers), -1, dtype=np.int64)
    if not len(centers) or not len(boxes):
        return SnapshotTargets(objectness, regression, assigned)

    proposals: list[tuple[float, int, int]] = []
    for target_index, box in enumerate(boxes):
        lower = box[:3] - 0.5 * box[3:]
        upper = box[:3] + 0.5 * box[3:]
        inside = np.flatnonzero(
            np.all(
                (centers >= lower[None]) & (centers <= upper[None]),
                axis=1,
            )
        )
        # Unlike legacy P1, there is intentionally no nearest-outside
        # fallback.  A snapshot that does not observe a target supplies no
        # positive anchor for that target.
        if not len(inside):
            continue
        distance = np.linalg.norm(centers[inside] - box[:3][None], axis=1)
        order = np.lexsort((inside, distance))[:topk]
        for local_index in order.tolist():
            proposals.append(
                (
                    float(distance[local_index]),
                    int(target_index),
                    int(inside[local_index]),
                )
            )
    for _, target_index, voxel_index in sorted(
        proposals, key=lambda row: (row[0], row[1], row[2])
    ):
        if assigned[voxel_index] >= 0:
            continue
        assigned[voxel_index] = target_index
        objectness[voxel_index] = 1.0
        regression[voxel_index, :3] = (
            boxes[target_index, :3] - centers[voxel_index]
        ).astype(np.float32)
        regression[voxel_index, 3:] = np.log(
            boxes[target_index, 3:]
        ).astype(np.float32)
    return SnapshotTargets(objectness, regression, assigned)


def deterministic_snapshot_loss_mask(
    scene_id: str,
    snapshot_index: int,
    objectness: np.ndarray,
    *,
    negative_ratio: float,
    maximum_loss_voxels: int,
    seed: int,
) -> np.ndarray:
    """Select loss rows while retaining the complete forward context."""

    labels = np.asarray(objectness, dtype=np.float32).reshape(-1)
    if not math.isfinite(float(negative_ratio)) or float(negative_ratio) < 0.0:
        raise ValueError("negative_ratio must be finite and non-negative")
    if isinstance(maximum_loss_voxels, bool) or int(maximum_loss_voxels) < 0:
        raise ValueError("maximum_loss_voxels must be a non-negative integer")
    positive = np.flatnonzero(labels > 0.5)
    negative = np.flatnonzero(labels <= 0.5)
    desired_negative = min(
        len(negative),
        int(round(max(len(positive), 1) * float(negative_ratio))),
    )
    if int(maximum_loss_voxels) > 0:
        desired_negative = min(
            desired_negative,
            max(int(maximum_loss_voxels) - len(positive), 0),
        )
    digest = hashlib.sha256(
        f"{int(seed)}:{scene_id}:{int(snapshot_index)}".encode("utf-8")
    ).digest()
    rng = np.random.default_rng(
        int.from_bytes(digest[:8], "little") % (2**32)
    )
    chosen_negative = (
        np.sort(
            rng.choice(negative, size=desired_negative, replace=False)
        ).astype(np.int64)
        if desired_negative < len(negative)
        else negative
    )
    mask = np.zeros(len(labels), dtype=bool)
    mask[positive] = True
    mask[chosen_negative] = True
    return mask


def _dataset_fingerprint(
    summaries: Sequence[Mapping[str, Any]],
    *,
    covered_iou: float,
    assignment_topk: int,
    negative_ratio: float,
    maximum_loss_voxels_per_snapshot: int,
    seed: int,
) -> str:
    payload = {
        "target_assignment_scope": TARGET_ASSIGNMENT_SCOPE,
        "covered_iou": float(covered_iou),
        "assignment_topk": int(assignment_topk),
        "negative_ratio": float(negative_ratio),
        "maximum_loss_voxels_per_snapshot": int(
            maximum_loss_voxels_per_snapshot
        ),
        "seed": int(seed),
        "scenes": list(summaries),
    }
    rendered = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def build_training_data(
    *,
    scenes: Sequence[str],
    diagnostics_root: str | os.PathLike[str],
    prediction_root: str | os.PathLike[str],
    gt_root: str | os.PathLike[str],
    scans_root: str | os.PathLike[str],
    covered_iou: float = 0.15,
    assignment_topk: int = 6,
    negative_ratio: float = 8.0,
    maximum_loss_voxels_per_snapshot: int = 4096,
    seed: int = 1337,
) -> P1V2TrainingData:
    """Build one immutable, context-preserving dataset for P1R and P1S."""

    diagnostics = Path(diagnostics_root)
    predictions = Path(prediction_root)
    gt_directory = Path(gt_root)
    scans = Path(scans_root)
    for role, root in (
        ("diagnostics", diagnostics),
        ("predictions", predictions),
        ("ground truth", gt_directory),
        ("scans", scans),
    ):
        if not root.is_dir():
            raise FileNotFoundError(f"{role} root not found: {root}")
    if not 0.0 <= float(covered_iou) <= 1.0:
        raise ValueError("covered_iou must lie in [0,1]")

    contexts: list[SceneTrainingContext] = []
    summaries: list[Mapping[str, Any]] = []
    canonical_names: tuple[str, ...] | None = None
    for scene_id in scenes:
        diagnostic_path = diagnostics / f"{scene_id}_tracks.npz"
        prediction_path = predictions / f"{scene_id}_boxes.pkl"
        gt_path = gt_directory / f"{scene_id}_bbox.npy"
        alignment_path = scans / scene_id / f"{scene_id}.txt"
        inputs = load_scene_context(
            diagnostic_path, expected_scene_id=scene_id
        )
        if canonical_names is None:
            canonical_names = inputs.feature_names
        elif canonical_names != inputs.feature_names:
            raise ValueError(
                f"{scene_id}: feature schema differs across scenes"
            )
        ground_truth = load_gt_boxes(gt_path)
        baseline = load_prediction_corners(prediction_path)
        alignment = load_axis_alignment(scans, scene_id)
        residual_boxes, residual_indices = residual_gt_world_boxes(
            ground_truth,
            baseline,
            alignment,
            covered_iou=covered_iou,
        )
        objectness = np.zeros(len(inputs.features), dtype=np.float32)
        regression = np.zeros((len(inputs.features), 6), dtype=np.float32)
        assigned_gt = np.full(len(inputs.features), -1, dtype=np.int64)
        loss_mask = np.zeros(len(inputs.features), dtype=bool)
        snapshot_summaries: list[dict[str, Any]] = []
        for snapshot_index in range(inputs.snapshot_count):
            selected = inputs.snapshot_slice(snapshot_index)
            targets = assign_snapshot_inside_targets(
                inputs.centers_world[selected],
                residual_boxes,
                topk=assignment_topk,
            )
            local_loss_mask = deterministic_snapshot_loss_mask(
                scene_id,
                snapshot_index,
                targets.objectness,
                negative_ratio=negative_ratio,
                maximum_loss_voxels=maximum_loss_voxels_per_snapshot,
                seed=seed,
            )
            objectness[selected] = targets.objectness
            regression[selected] = targets.regression
            assigned_gt[selected] = targets.assigned_gt
            loss_mask[selected] = local_loss_mask
            snapshot_summaries.append(
                {
                    "snapshot_index": int(snapshot_index),
                    "frame_id": int(inputs.frame_ids[snapshot_index]),
                    "provider_step": int(
                        inputs.provider_steps[snapshot_index]
                    ),
                    "voxel_count": int(
                        inputs.offsets[snapshot_index + 1]
                        - inputs.offsets[snapshot_index]
                    ),
                    "loss_voxel_count": int(
                        np.count_nonzero(local_loss_mask)
                    ),
                    "positive_count": int(
                        np.count_nonzero(targets.objectness > 0.5)
                    ),
                    "observed_residual_gt_count": int(
                        len(np.unique(targets.assigned_gt[
                            targets.assigned_gt >= 0
                        ]))
                    ),
                }
            )
        context = SceneTrainingContext(
            scene_id=inputs.scene_id,
            features=inputs.features,
            coordinates=inputs.coordinates,
            centers_world=inputs.centers_world,
            offsets=inputs.offsets,
            frame_ids=inputs.frame_ids,
            provider_steps=inputs.provider_steps,
            objectness=objectness,
            regression=regression,
            assigned_gt=assigned_gt,
            loss_mask=loss_mask,
            feature_names=inputs.feature_names,
            voxel_size=inputs.voxel_size,
            diagnostic_path=inputs.diagnostic_path,
        )
        contexts.append(context)
        summaries.append(
            {
                "scene_id": scene_id,
                "diagnostic_sha256": _file_sha256(diagnostic_path),
                "prediction_sha256": _file_sha256(prediction_path),
                "ground_truth_sha256": _file_sha256(gt_path),
                "axis_alignment_sha256": _file_sha256(alignment_path),
                "snapshot_count": int(context.snapshot_count),
                "voxel_count": int(len(context.features)),
                "loss_voxel_count": int(np.count_nonzero(loss_mask)),
                "positive_count": int(
                    np.count_nonzero(loss_mask & (objectness > 0.5))
                ),
                "ground_truth_count": int(len(ground_truth)),
                "residual_ground_truth_count": int(len(residual_boxes)),
                "residual_ground_truth_indices": residual_indices.tolist(),
                "snapshots": snapshot_summaries,
            }
        )
    if canonical_names is None or not contexts:
        raise ValueError("no P1-v2 scenes were loaded")
    fingerprint = _dataset_fingerprint(
        summaries,
        covered_iou=covered_iou,
        assignment_topk=assignment_topk,
        negative_ratio=negative_ratio,
        maximum_loss_voxels_per_snapshot=(
            maximum_loss_voxels_per_snapshot
        ),
        seed=seed,
    )
    data = P1V2TrainingData(
        scenes=tuple(contexts),
        feature_names=canonical_names,
        scene_summaries=tuple(summaries),
        dataset_fingerprint_sha256=fingerprint,
    )
    if data.loss_sample_count == 0:
        raise ValueError("P1-v2 dataset contains zero loss rows")
    if data.positive_count == 0:
        raise ValueError(
            "P1-v2 dataset contains zero inside-snapshot positive rows"
        )
    return data


def _make_head(
    variant: str, *, input_dim: int, hidden_dim: int
) -> nn.Module:
    variant = normalize_variant(variant)
    if int(input_dim) != P1_FEATURE_DIM:
        raise ValueError("P1-v2 input feature dimension must remain 14")
    if int(hidden_dim) <= 0:
        raise ValueError("hidden_dim must be positive")
    if variant == "P1R":
        return ResidualVoxelProposalHead(
            input_dim=int(input_dim),
            hidden_dim=int(hidden_dim),
            regression_dim=6,
        )
    return NativeSparseResidualProposalHead(
        input_dim=int(input_dim),
        hidden_dim=int(hidden_dim),
        regression_dim=6,
    )


def _forward_snapshot(
    model: nn.Module,
    *,
    variant: str,
    features: torch.Tensor,
    coordinates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if normalize_variant(variant) == "P1R":
        logits, regression = model(features)
    else:
        logits, regression = model(features, coordinates)
    if logits.shape == (len(features), 1):
        logits = logits[:, 0]
    if logits.shape != (len(features),) or regression.shape != (
        len(features),
        6,
    ):
        raise ValueError("P1-v2 head returned an invalid output shape")
    return logits, regression


def _snapshot_loss(
    logits: torch.Tensor,
    predicted_regression: torch.Tensor,
    objectness: torch.Tensor,
    target_regression: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    positive_weight: torch.Tensor,
    regression_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = loss_mask.to(dtype=torch.bool)
    if selected.ndim != 1 or selected.shape != objectness.shape:
        raise ValueError("loss_mask must align with snapshot labels")
    if not bool(torch.any(selected)):
        zero = logits.sum() * 0.0
        return zero, zero, zero
    classification = F.binary_cross_entropy_with_logits(
        logits[selected],
        objectness[selected],
        pos_weight=positive_weight,
    )
    positive = selected & (objectness > 0.5)
    if bool(torch.any(positive)):
        regression = F.smooth_l1_loss(
            predicted_regression[positive],
            target_regression[positive],
            beta=0.10,
        )
    else:
        regression = predicted_regression.sum() * 0.0
    total = classification + float(regression_weight) * regression
    return total, classification, regression


def _ordered_snapshots(
    data: P1V2TrainingData,
    selected_scene_ids: Sequence[str],
    *,
    seed: int,
    epoch: int,
    shuffle: bool,
) -> list[tuple[SceneTrainingContext, int]]:
    selected = frozenset(str(scene) for scene in selected_scene_ids)
    rows = [
        (scene, snapshot_index)
        for scene in data.scenes
        if scene.scene_id in selected
        for snapshot_index in range(scene.snapshot_count)
        if bool(np.any(scene.loss_mask[scene.snapshot_slice(snapshot_index)]))
    ]
    rows.sort(key=lambda row: (row[0].scene_id, row[1]))
    if shuffle and rows:
        rng = np.random.default_rng(int(seed) + 104729 * int(epoch))
        order = rng.permutation(len(rows))
        rows = [rows[int(index)] for index in order]
    return rows


def _evaluate(
    model: nn.Module,
    data: P1V2TrainingData,
    scene_ids: Sequence[str],
    *,
    variant: str,
    device: torch.device,
    positive_weight: torch.Tensor,
    regression_weight: float,
) -> dict[str, Any]:
    model.eval()
    total_weighted_loss = 0.0
    total_classification = 0.0
    total_regression = 0.0
    total_rows = 0
    total_snapshots = 0
    true_positive = false_positive = false_negative = 0
    positive_rows = 0
    regression_absolute_sum = 0.0
    regression_element_count = 0
    with torch.no_grad():
        for scene, snapshot_index in _ordered_snapshots(
            data, scene_ids, seed=0, epoch=0, shuffle=False
        ):
            selected = scene.snapshot_slice(snapshot_index)
            features = torch.from_numpy(scene.features[selected]).to(device)
            coordinates = torch.from_numpy(
                scene.coordinates[selected]
            ).to(device)
            objectness = torch.from_numpy(
                scene.objectness[selected]
            ).to(device)
            regression = torch.from_numpy(
                scene.regression[selected]
            ).to(device)
            loss_mask = torch.from_numpy(
                scene.loss_mask[selected]
            ).to(device)
            logits, predicted = _forward_snapshot(
                model,
                variant=variant,
                features=features,
                coordinates=coordinates,
            )
            loss, classification, regression_loss = _snapshot_loss(
                logits,
                predicted,
                objectness,
                regression,
                loss_mask,
                positive_weight=positive_weight,
                regression_weight=regression_weight,
            )
            count = int(torch.count_nonzero(loss_mask).item())
            # Training optimizes the mean of complete-snapshot losses.  Keep
            # checkpoint selection on that same macro objective; micro row
            # counts remain available separately for precision/recall.
            total_weighted_loss += float(loss.item())
            total_classification += float(classification.item())
            total_regression += float(regression_loss.item())
            total_rows += count
            total_snapshots += 1
            mask = loss_mask.to(dtype=torch.bool)
            target = objectness > 0.5
            prediction = torch.sigmoid(logits) >= 0.5
            true_positive += int(
                torch.count_nonzero(mask & prediction & target).item()
            )
            false_positive += int(
                torch.count_nonzero(mask & prediction & ~target).item()
            )
            false_negative += int(
                torch.count_nonzero(mask & ~prediction & target).item()
            )
            positive = mask & target
            positive_count = int(torch.count_nonzero(positive).item())
            positive_rows += positive_count
            if positive_count:
                regression_absolute_sum += float(
                    torch.sum(
                        torch.abs(predicted[positive] - regression[positive])
                    ).item()
                )
                regression_element_count += positive_count * 6
    if total_rows == 0 or total_snapshots == 0:
        raise RuntimeError("evaluation split contains zero selected loss rows")
    return {
        "loss": float(total_weighted_loss / total_snapshots),
        "classification_loss": float(
            total_classification / total_snapshots
        ),
        "regression_loss": float(total_regression / total_snapshots),
        "loss_aggregation": "snapshot_macro",
        "snapshot_count": float(total_snapshots),
        "precision_at_0p5": float(
            true_positive / max(true_positive + false_positive, 1)
        ),
        "recall_at_0p5": float(
            true_positive / max(true_positive + false_negative, 1)
        ),
        "regression_mae_positive": float(
            regression_absolute_sum / max(regression_element_count, 1)
        ),
        "positive_count": float(positive_rows),
        "sample_count": float(total_rows),
    }


def train_variant(
    data: P1V2TrainingData,
    *,
    variant: str,
    hidden_dim: int = 64,
    validation_fraction: float = 0.20,
    epochs: int = 80,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    regression_weight: float = 1.0,
    snapshots_per_optimizer_step: int = 4,
    seed: int = 1337,
    device: str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Train one variant with a scene-disjoint optimization split."""

    variant = normalize_variant(variant)
    if int(hidden_dim) <= 0 or int(epochs) <= 0:
        raise ValueError("hidden_dim and epochs must be positive")
    if int(snapshots_per_optimizer_step) <= 0:
        raise ValueError("snapshots_per_optimizer_step must be positive")
    if float(learning_rate) <= 0.0 or float(weight_decay) < 0.0:
        raise ValueError("invalid optimizer configuration")
    if float(regression_weight) < 0.0:
        raise ValueError("regression_weight must be non-negative")
    torch_device = torch.device(str(device))
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested unavailable CUDA device: {device}")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    training_scenes, validation_scenes = deterministic_scene_partition(
        data.scene_ids,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    if set(training_scenes) & set(validation_scenes):
        raise RuntimeError("optimization train/validation leakage")

    model = _make_head(
        variant,
        input_dim=len(data.feature_names),
        hidden_dim=int(hidden_dim),
    ).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    training_contexts = [
        scene for scene in data.scenes if scene.scene_id in training_scenes
    ]
    train_positive = sum(
        np.count_nonzero(scene.loss_mask & (scene.objectness > 0.5))
        for scene in training_contexts
    )
    train_rows = sum(
        np.count_nonzero(scene.loss_mask) for scene in training_contexts
    )
    train_negative = train_rows - train_positive
    if train_rows <= 0 or train_positive <= 0:
        raise ValueError("optimization training split has no positive rows")
    positive_weight_value = min(
        max(float(train_negative) / float(train_positive), 1.0), 50.0
    )
    positive_weight = torch.tensor(
        positive_weight_value,
        device=torch_device,
        dtype=torch.float32,
    )
    best_loss = math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    accumulation = int(snapshots_per_optimizer_step)
    for epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_snapshots = 0
        pending = 0
        snapshots = _ordered_snapshots(
            data,
            training_scenes,
            seed=seed,
            epoch=epoch,
            shuffle=True,
        )
        for row_index, (scene, snapshot_index) in enumerate(snapshots):
            selected = scene.snapshot_slice(snapshot_index)
            features = torch.from_numpy(scene.features[selected]).to(
                torch_device
            )
            coordinates = torch.from_numpy(
                scene.coordinates[selected]
            ).to(torch_device)
            objectness = torch.from_numpy(
                scene.objectness[selected]
            ).to(torch_device)
            regression = torch.from_numpy(
                scene.regression[selected]
            ).to(torch_device)
            loss_mask = torch.from_numpy(scene.loss_mask[selected]).to(
                torch_device
            )
            logits, predicted = _forward_snapshot(
                model,
                variant=variant,
                features=features,
                coordinates=coordinates,
            )
            loss, _, _ = _snapshot_loss(
                logits,
                predicted,
                objectness,
                regression,
                loss_mask,
                positive_weight=positive_weight,
                regression_weight=regression_weight,
            )
            (loss / accumulation).backward()
            pending += 1
            running_loss += float(loss.detach().item())
            running_snapshots += 1
            if pending == accumulation or row_index == len(snapshots) - 1:
                # Correct the final partial accumulation without changing the
                # gradients of complete groups.
                if pending < accumulation:
                    scale = float(accumulation) / float(pending)
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(scale)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
        validation = _evaluate(
            model,
            data,
            validation_scenes,
            variant=variant,
            device=torch_device,
            positive_weight=positive_weight,
            regression_weight=regression_weight,
        )
        epoch_record = {
            "epoch": int(epoch),
            "training_loss": float(
                running_loss / max(running_snapshots, 1)
            ),
            "validation_loss": float(validation["loss"]),
        }
        history.append(epoch_record)
        if float(validation["loss"]) < best_loss:
            best_loss = float(validation["loss"])
            best_epoch = int(epoch)
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("P1-v2 training did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    validation = _evaluate(
        model,
        data,
        validation_scenes,
        variant=variant,
        device=torch_device,
        positive_weight=positive_weight,
        regression_weight=regression_weight,
    )
    metrics: dict[str, Any] = {
        "variant": variant,
        "head_architecture": VARIANT_ARCHITECTURES[variant],
        "target_assignment_scope": TARGET_ASSIGNMENT_SCOPE,
        "deterministic_algorithms": True,
        "loss_aggregation": "snapshot_macro",
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "training_scenes": list(training_scenes),
        "validation_scenes": list(validation_scenes),
        "optimization_split_overlap": [],
        "training_loss_samples": int(train_rows),
        "validation_metrics": validation,
        "positive_weight": float(positive_weight_value),
        "last_epoch": history[-1],
    }
    return model.cpu(), metrics


def checkpoint_model_config(
    model: nn.Module, *, variant: str, hidden_dim: int
) -> dict[str, Any]:
    """Return the exact runtime reconstruction contract."""

    variant = normalize_variant(variant)
    if variant == "P1S":
        if not hasattr(model, "model_config"):
            raise TypeError("P1S head lacks model_config()")
        config = dict(model.model_config())
        if config.get("architecture") != P1_SPATIAL_ARCHITECTURE:
            raise ValueError("P1S model_config architecture mismatch")
        # NativeSparseResidualProposalHead.from_model_config is intentionally
        # strict, so no trainer-only keys are inserted here.
        return config
    return {
        "input_dim": int(P1_FEATURE_DIM),
        "hidden_dim": int(hidden_dim),
        "regression_dim": 6,
        "regression_encoding": "center_delta_m_log_size_m",
        "head_architecture": P1R_ARCHITECTURE,
        "target_assignment_scope": TARGET_ASSIGNMENT_SCOPE,
    }


def save_checkpoint(
    output_path: str | os.PathLike[str],
    *,
    model: nn.Module,
    variant: str,
    hidden_dim: int,
    feature_names: Sequence[str],
    training_config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Path:
    variant = normalize_variant(variant)
    output = Path(output_path)
    if output.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("P1-v2 checkpoint must end in .pt or .pth")
    model_config = checkpoint_model_config(
        model, variant=variant, hidden_dim=hidden_dim
    )
    checkpoint = {
        "schema": VARIANT_SCHEMAS[variant],
        "variant": variant,
        "head_architecture": VARIANT_ARCHITECTURES[variant],
        "target_assignment_scope": TARGET_ASSIGNMENT_SCOPE,
        "model_config": model_config,
        "feature_names": [str(name) for name in feature_names],
        "state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
        "training_config": dict(training_config),
        "metrics": dict(metrics),
        "provenance": dict(provenance),
    }
    if (
        checkpoint["training_config"].get("target_assignment_scope")
        != TARGET_ASSIGNMENT_SCOPE
    ):
        raise ValueError("training_config target assignment scope mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, output)
    try:
        loaded = torch.load(output, map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(output, map_location="cpu")
    if (
        not isinstance(loaded, Mapping)
        or loaded.get("schema") != VARIANT_SCHEMAS[variant]
        or loaded.get("head_architecture")
        != VARIANT_ARCHITECTURES[variant]
        or loaded.get("target_assignment_scope")
        != TARGET_ASSIGNMENT_SCOPE
        or loaded.get("training_config", {}).get(
            "target_assignment_scope"
        )
        != TARGET_ASSIGNMENT_SCOPE
        or tuple(loaded.get("feature_names", ()))
        != tuple(P1_FEATURE_NAMES)
        or not isinstance(loaded.get("provenance"), Mapping)
    ):
        raise RuntimeError("saved P1-v2 checkpoint failed contract audit")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=("P1R", "P1S"))
    parser.add_argument("--diagnostics-root", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--scans-root", required=True, type=Path)
    parser.add_argument("--train-scene-list", required=True, type=Path)
    parser.add_argument("--forbidden-scene-list", required=True, type=Path)
    parser.add_argument("--b6-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--source-p1-checkpoint",
        required=True,
        type=Path,
        help=(
            "legacy P1 checkpoint used as the immutable collect/B6 "
            "provenance witness"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--covered-iou", type=float, default=0.15)
    parser.add_argument("--assignment-topk", type=int, default=6)
    parser.add_argument("--negative-ratio", type=float, default=8.0)
    parser.add_argument(
        "--maximum-loss-voxels-per-snapshot", type=int, default=4096
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--regression-weight", type=float, default=1.0)
    parser.add_argument(
        "--snapshots-per-optimizer-step", type=int, default=4
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--summary-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    variant = normalize_variant(args.variant)
    train_scenes = read_scene_ids(args.train_scene_list, role="training")
    forbidden_scenes = read_scene_ids(
        args.forbidden_scene_list, role="forbidden validation"
    )
    train_scenes = validate_external_split(train_scenes, forbidden_scenes)
    source_binding = validate_source_collection_provenance(
        args.source_p1_checkpoint,
        scenes=train_scenes,
        diagnostics_root=args.diagnostics_root,
        prediction_root=args.prediction_root,
        gt_root=args.gt_root,
        train_scene_list=args.train_scene_list,
        forbidden_scene_list=args.forbidden_scene_list,
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
    training_config = {
        "schema": TRAINING_SCHEMA,
        "variant": variant,
        "head_architecture": VARIANT_ARCHITECTURES[variant],
        "target_assignment_scope": TARGET_ASSIGNMENT_SCOPE,
        "deterministic_algorithms": True,
        "loss_aggregation": "snapshot_macro",
        "covered_iou": float(args.covered_iou),
        "assignment_topk": int(args.assignment_topk),
        "negative_ratio": float(args.negative_ratio),
        "maximum_loss_voxels_per_snapshot": int(
            args.maximum_loss_voxels_per_snapshot
        ),
        "hidden_dim": int(args.hidden_dim),
        "validation_fraction": float(args.validation_fraction),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "regression_weight": float(args.regression_weight),
        "snapshots_per_optimizer_step": int(
            args.snapshots_per_optimizer_step
        ),
        "seed": int(args.seed),
        "device": str(args.device),
    }
    model, metrics = train_variant(
        data,
        variant=variant,
        hidden_dim=args.hidden_dim,
        validation_fraction=args.validation_fraction,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        regression_weight=args.regression_weight,
        snapshots_per_optimizer_step=args.snapshots_per_optimizer_step,
        seed=args.seed,
        device=args.device,
    )
    optimization_train = tuple(metrics["training_scenes"])
    optimization_validation = tuple(metrics["validation_scenes"])
    provenance = {
        "source": "frozen_p1_collect_train100",
        "source_diagnostic_schema": P1_DIAGNOSTIC_SCHEMA,
        "source_collection_binding": source_binding,
        "source_train_scene_ids": list(train_scenes),
        # Kept for compatibility with the runtime's train-only audit.
        "train_scene_ids": list(train_scenes),
        "optimization_train_scene_ids": list(optimization_train),
        "optimization_validation_scene_ids": list(
            optimization_validation
        ),
        "optimization_split_overlap": [],
        "train_scene_list": str(args.train_scene_list.resolve()),
        "train_scene_list_sha256": _file_sha256(args.train_scene_list),
        "forbidden_scene_list": str(
            args.forbidden_scene_list.resolve()
        ),
        "forbidden_scene_list_sha256": _file_sha256(
            args.forbidden_scene_list
        ),
        "forbidden_scene_count": int(len(forbidden_scenes)),
        "forbidden_overlap": [],
        "b6_checkpoint": str(args.b6_checkpoint.resolve()),
        "b6_checkpoint_sha256": _file_sha256(args.b6_checkpoint),
        "diagnostics_root": str(args.diagnostics_root.resolve()),
        "prediction_root": str(args.prediction_root.resolve()),
        "gt_root": str(args.gt_root.resolve()),
        "scans_root": str(args.scans_root.resolve()),
        "dataset_fingerprint_sha256": data.dataset_fingerprint_sha256,
        "scene_summaries": list(data.scene_summaries),
    }
    output = save_checkpoint(
        args.output,
        model=model,
        variant=variant,
        hidden_dim=args.hidden_dim,
        feature_names=data.feature_names,
        training_config=training_config,
        metrics=metrics,
        provenance=provenance,
    )
    summary = {
        "schema": VARIANT_SCHEMAS[variant],
        "variant": variant,
        "head_architecture": VARIANT_ARCHITECTURES[variant],
        "target_assignment_scope": TARGET_ASSIGNMENT_SCOPE,
        "output": str(output.resolve()),
        "scene_count": int(len(data.scenes)),
        "snapshot_count": int(
            sum(scene.snapshot_count for scene in data.scenes)
        ),
        "context_voxel_count": int(data.voxel_count),
        "loss_sample_count": int(data.loss_sample_count),
        "positive_count": int(data.positive_count),
        "dataset_fingerprint_sha256": data.dataset_fingerprint_sha256,
        "source_collection_binding": source_binding,
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
