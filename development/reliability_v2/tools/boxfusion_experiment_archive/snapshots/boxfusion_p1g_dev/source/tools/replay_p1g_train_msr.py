#!/usr/bin/env python3
"""Offline, train-only replay and geometry audit for frozen P1S.

This utility deliberately does not run Cubify Anything, YOLOE, B6, or the
ScanNet evaluator.  It replays the frozen P1S head from the immutable legacy
P1 collect snapshots, reconstructs real-depth residual points for the exact
scheduled frames, and feeds those observations to the observer-only P1G
multi-view occupancy/MSR module.

Ground truth is loaded only after all proposals and refinements have been
constructed.  It is used for two offline diagnostics:

* an exact six-face reachability sweep at fixed normalized face limits; and
* original-versus-refined IoU statistics for the actual MSR proposal.

Every per-scene output is an ``allow_pickle=False`` compatible NPZ.  The
source P1S checkpoint, B6 checkpoint, legacy diagnostic, B6 prediction, GT,
axis alignment, requested scene list, and forbidden validation list are
bound by SHA256 provenance.  Formal BoxFusion predictions are never read back
for mutation and are never written by this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boxfusion.p1_multiview_geometry import (  # noqa: E402
    P1MultiViewGeometryObserver,
)
from boxfusion.residual_proposal import (  # noqa: E402
    P1_FEATURE_NAMES,
    P1S_HEAD_SCHEMA,
    P1ResidualProposalObserver,
    ResidualProposalConfig,
    ResidualVoxelBatch,
    center_size_to_corners,
    corners_to_center_size,
    load_residual_proposal_head,
    pairwise_aabb_iou,
    points_explained_by_boxes,
    stable_nms_aabb,
)
from tools.train_p1_residual_head import (  # noqa: E402
    load_axis_alignment,
    load_gt_boxes,
    load_prediction_corners,
    transform_points,
)
from tools.train_p1v2_residual_head import (  # noqa: E402
    load_scene_context,
)


OUTPUT_SCHEMA = "boxfusion.p1g.train_msr_replay.v1"
SUMMARY_SCHEMA = "boxfusion.p1g.train_msr_replay_summary.v1"
DEFAULT_FACE_LIMITS = (0.18, 0.25, 0.50, 0.75)
_SCENE_LENGTH = 12


def file_sha256(path: str | os.PathLike[str]) -> str:
    """Return the SHA256 of one required regular file."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_scene_ids(
    path: str | os.PathLike[str], *, role: str
) -> tuple[str, ...]:
    """Read a non-empty, unique list of canonical ScanNet scene IDs."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"{role} scene list not found: {source}")
    rows = tuple(
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not rows:
        raise ValueError(f"{role} scene list is empty")
    if len(set(rows)) != len(rows):
        raise ValueError(f"{role} scene list contains duplicates")
    for scene_id in rows:
        if not (
            len(scene_id) == _SCENE_LENGTH
            and scene_id.startswith("scene")
            and scene_id[5:9].isdigit()
            and scene_id[9] == "_"
            and scene_id[10:].isdigit()
        ):
            raise ValueError(f"invalid {role} scene id: {scene_id!r}")
    return rows


def validate_scene_partition(
    requested: Sequence[str], forbidden: Sequence[str]
) -> None:
    """Fail closed on any target validation scene leakage."""

    overlap = sorted(set(requested) & set(forbidden))
    if overlap:
        raise ValueError(
            "P1G replay scene list overlaps forbidden validation scenes: "
            + ", ".join(overlap[:16])
        )


@dataclass(frozen=True)
class ReplaySceneInputs:
    """Strict legacy P1 snapshot tensors required to replay P1S."""

    scene_id: str
    coordinates: np.ndarray
    centers: np.ndarray
    features: np.ndarray
    point_counts: np.ndarray
    offsets: np.ndarray
    frame_ids: np.ndarray
    provider_steps: np.ndarray
    voxel_size: float
    diagnostic_path: Path


def load_replay_scene_inputs(
    path: str | os.PathLike[str], *, expected_scene_id: str
) -> ReplaySceneInputs:
    """Load and cross-check one immutable legacy P1 collect archive."""

    source = Path(path)
    # Reuse the complete legacy safety/schema validator rather than accepting
    # a merely shape-compatible archive.
    context = load_scene_context(source, expected_scene_id=expected_scene_id)
    with np.load(source, allow_pickle=False) as archive:
        if "p1_voxel_point_counts" not in archive.files:
            raise ValueError(f"{source}: missing p1_voxel_point_counts")
        counts = np.array(archive["p1_voxel_point_counts"], copy=True)
    if (
        counts.shape != (len(context.features),)
        or not np.issubdtype(counts.dtype, np.integer)
        or np.any(counts <= 0)
    ):
        raise ValueError(
            f"{source}: p1_voxel_point_counts must be positive integer [V]"
        )
    return ReplaySceneInputs(
        scene_id=context.scene_id,
        coordinates=context.coordinates,
        centers=context.centers_world,
        features=context.features,
        point_counts=np.ascontiguousarray(counts, dtype=np.int32),
        offsets=context.offsets,
        frame_ids=context.frame_ids,
        provider_steps=context.provider_steps,
        voxel_size=float(context.voxel_size),
        diagnostic_path=source,
    )


def _load_checkpoint_mapping(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - PyTorch < 2.0
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("P1S checkpoint must contain a mapping")
    return payload


@dataclass(frozen=True)
class FrozenParents:
    """Validated frozen P1S/B6 lineage and CPU proposal head."""

    proposal_observer: P1ResidualProposalObserver
    p1s_payload: Mapping[str, Any]
    p1s_sha256: str
    b6_sha256: str
    source_scenes: tuple[str, ...]
    scene_summaries: Mapping[str, Mapping[str, Any]]


def load_frozen_parents(
    p1s_checkpoint: str | os.PathLike[str],
    b6_checkpoint: str | os.PathLike[str],
    *,
    requested_scenes: Sequence[str],
    forbidden_scene_list: str | os.PathLike[str],
    score_threshold: float = 0.05,
    max_scene_candidates: int = 256,
) -> FrozenParents:
    """Load P1S on CPU and validate its complete train-only parent binding."""

    p1s_path = Path(p1s_checkpoint)
    b6_path = Path(b6_checkpoint)
    forbidden_path = Path(forbidden_scene_list)
    for role, source in (
        ("P1S checkpoint", p1s_path),
        ("B6 checkpoint", b6_path),
        ("forbidden scene list", forbidden_path),
    ):
        if not source.is_file():
            raise FileNotFoundError(f"{role} not found: {source}")
    payload = _load_checkpoint_mapping(p1s_path)
    if payload.get("schema") != P1S_HEAD_SCHEMA:
        raise ValueError("P1S checkpoint schema mismatch")
    if tuple(payload.get("feature_names", ())) != tuple(P1_FEATURE_NAMES):
        raise ValueError("P1S feature schema mismatch")
    model_config = payload.get("model_config")
    training_config = payload.get("training_config")
    provenance = payload.get("provenance")
    if not all(
        isinstance(value, Mapping)
        for value in (model_config, training_config, provenance)
    ):
        raise ValueError("P1S checkpoint lacks strict config/provenance")
    if (
        model_config.get("architecture") != "native_sparse_context_v1"
        or training_config.get("target_assignment_scope")
        != "snapshot_inside_only"
    ):
        raise ValueError("checkpoint is not the frozen P1S variant")
    b6_sha = file_sha256(b6_path)
    if str(provenance.get("b6_checkpoint_sha256", "")).lower() != b6_sha:
        raise ValueError("P1S checkpoint binds a different B6 checkpoint")
    if (
        str(provenance.get("forbidden_scene_list_sha256", "")).lower()
        != file_sha256(forbidden_path)
        or provenance.get("forbidden_overlap") != []
    ):
        raise ValueError("P1S forbidden-split provenance mismatch")
    source_scenes_value = provenance.get("train_scene_ids")
    if (
        not isinstance(source_scenes_value, Sequence)
        or isinstance(source_scenes_value, (str, bytes))
    ):
        raise ValueError("P1S provenance lacks train_scene_ids")
    source_scenes = tuple(str(value) for value in source_scenes_value)
    if not set(requested_scenes).issubset(source_scenes):
        raise ValueError("requested scenes are not a subset of P1S train data")
    summaries_value = provenance.get("scene_summaries")
    if (
        not isinstance(summaries_value, Sequence)
        or isinstance(summaries_value, (str, bytes))
    ):
        raise ValueError("P1S provenance lacks scene artifact summaries")
    summaries: dict[str, Mapping[str, Any]] = {}
    for row in summaries_value:
        if not isinstance(row, Mapping):
            raise ValueError("P1S scene summary must be a mapping")
        scene_id = row.get("scene_id")
        if not isinstance(scene_id, str) or scene_id in summaries:
            raise ValueError("P1S scene artifact summaries are invalid")
        summaries[scene_id] = row
    if set(source_scenes) != set(summaries):
        raise ValueError("P1S scene artifact provenance is incomplete")

    hidden_dim = int(model_config.get("hidden_dim", -1))
    config = ResidualProposalConfig(
        enabled=True,
        observer_only=True,
        mutate=False,
        collect_diagnostics=True,
        mode="infer",
        checkpoint=str(p1s_path),
        device="cpu",
        voxel_size=0.08,
        hidden_dim=hidden_dim,
        head_architecture="native_sparse_context_v1",
        target_assignment_scope="snapshot_inside_only",
        score_threshold=float(score_threshold),
        max_scene_candidates=int(max_scene_candidates),
    ).validated()
    head, p1s_sha, metadata = load_residual_proposal_head(
        p1s_path,
        expected_config=config,
        device="cpu",
        expected_b6_checkpoint_sha256=b6_sha,
    )
    if metadata is not payload and metadata.get("schema") != payload.get(
        "schema"
    ):
        raise RuntimeError("P1S metadata changed during strict load")
    observer = P1ResidualProposalObserver(
        config,
        head=head,
        device="cpu",
    )
    return FrozenParents(
        proposal_observer=observer,
        p1s_payload=payload,
        p1s_sha256=p1s_sha,
        b6_sha256=b6_sha,
        source_scenes=source_scenes,
        scene_summaries=summaries,
    )


def validate_scene_artifact_binding(
    *,
    scene_id: str,
    inputs_path: Path,
    prediction_path: Path,
    gt_path: Path,
    alignment_path: Path,
    expected: Mapping[str, Any],
) -> dict[str, str]:
    """Validate exact artifacts against the frozen P1S training manifest."""

    actual = {
        "diagnostic_sha256": file_sha256(inputs_path),
        "prediction_sha256": file_sha256(prediction_path),
        "ground_truth_sha256": file_sha256(gt_path),
        "axis_alignment_sha256": file_sha256(alignment_path),
    }
    for name, digest in actual.items():
        observed = str(expected.get(name, "")).lower()
        if observed != digest:
            raise ValueError(
                f"{scene_id}: {name} differs from frozen P1S provenance"
            )
    return actual


def replay_p1s_candidates(
    inputs: ReplaySceneInputs,
    proposal_observer: P1ResidualProposalObserver,
    *,
    baseline_corners: np.ndarray,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Replay frozen P1S per snapshot and its deterministic scene NMS."""

    proposal_observer.reset(inputs.scene_id)
    step_observations: list[Any] = []
    all_rows: list[Any] = []
    stable_ids = np.arange(len(baseline_corners), dtype=np.int64)
    with torch.inference_mode():
        for snapshot_index in range(len(inputs.offsets) - 1):
            start = int(inputs.offsets[snapshot_index])
            stop = int(inputs.offsets[snapshot_index + 1])
            if stop <= start:
                continue
            batch = ResidualVoxelBatch(
                coordinates=inputs.coordinates[start:stop],
                centers=inputs.centers[start:stop],
                features=inputs.features[start:stop],
                point_counts=inputs.point_counts[start:stop],
                input_point_count=int(inputs.point_counts[start:stop].sum()),
                explained_point_count=0,
                residual_point_count=int(
                    inputs.point_counts[start:stop].sum()
                ),
            )
            logits, regression = proposal_observer.head(
                torch.from_numpy(inputs.features[start:stop]),
                torch.from_numpy(inputs.coordinates[start:stop]),
            )
            proposals = proposal_observer.decode(
                batch,
                logits,
                regression,
                scene_id=inputs.scene_id,
                frame_index=int(inputs.frame_ids[snapshot_index]),
                provider_step=int(inputs.provider_steps[snapshot_index]),
                global_corners=baseline_corners,
                global_stable_ids=stable_ids,
            )
            all_rows.extend(proposals)
            step_observations.append(
                SimpleNamespace(
                    frame_index=int(inputs.frame_ids[snapshot_index]),
                    provider_step=int(
                        inputs.provider_steps[snapshot_index]
                    ),
                    proposals=proposals,
                )
            )
    if not all_rows:
        return (), tuple(step_observations)
    boxes = np.stack([row.box for row in all_rows], axis=0)
    scores = np.asarray([row.objectness for row in all_rows])
    ids = [row.candidate_id for row in all_rows]
    keep = stable_nms_aabb(
        boxes,
        scores,
        proposal_observer.config.scene_nms_iou,
        tie_breakers=ids,
        max_output=proposal_observer.config.max_scene_candidates,
    )
    return (
        tuple(all_rows[int(index)] for index in keep),
        tuple(step_observations),
    )


def backproject_depth(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
    *,
    depth_scale: float = 1000.0,
    stride: int = 4,
    min_depth: float = 0.15,
    max_depth: float = 8.0,
) -> tuple[np.ndarray, float]:
    """Backproject one metric ScanNet depth frame into world coordinates."""

    values = np.asarray(depth)
    matrix = np.asarray(intrinsic, dtype=np.float64)
    pose = np.asarray(camera_to_world, dtype=np.float64)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.number):
        raise ValueError("depth must be a numeric [H,W] array")
    if matrix.shape not in {(3, 3), (4, 4)}:
        raise ValueError("intrinsic must have shape [3,3] or [4,4]")
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("camera_to_world must be finite [4,4]")
    if (
        not math.isfinite(float(depth_scale))
        or float(depth_scale) <= 0.0
        or isinstance(stride, bool)
        or int(stride) <= 0
        or not 0.0 < float(min_depth) < float(max_depth)
    ):
        raise ValueError("invalid depth backprojection configuration")
    sampled = values[:: int(stride), :: int(stride)].astype(np.float64)
    z = sampled / float(depth_scale)
    valid = (
        np.isfinite(z)
        & (z >= float(min_depth))
        & (z <= float(max_depth))
    )
    valid_ratio = float(valid.mean()) if valid.size else 0.0
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), valid_ratio
    y, x = np.mgrid[
        0 : values.shape[0] : int(stride),
        0 : values.shape[1] : int(stride),
    ]
    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    if (
        not np.isfinite([fx, fy, cx, cy]).all()
        or abs(fx) <= 1e-12
        or abs(fy) <= 1e-12
    ):
        raise ValueError("intrinsic focal lengths are invalid")
    camera = np.stack(
        (
            (x[valid] - cx) * z[valid] / fx,
            (y[valid] - cy) * z[valid] / fy,
            z[valid],
        ),
        axis=1,
    )
    world = transform_points(camera, pose)
    return np.ascontiguousarray(world, dtype=np.float32), valid_ratio


def load_scheduled_geometry(
    *,
    scene_id: str,
    frame_ids: Sequence[int],
    frames_root: str | os.PathLike[str],
    baseline_corners: np.ndarray,
    stride: int,
    depth_scale: float,
    min_depth: float,
    max_depth: float,
    explained_margin: float,
) -> tuple[Mapping[int, Mapping[str, Any]], Mapping[str, str]]:
    """Load exact scheduled depth/pose files and reconstruct residual points."""

    scene_root = Path(frames_root) / scene_id / "frames"
    intrinsic_path = scene_root / "intrinsic" / "intrinsic_depth.txt"
    if not intrinsic_path.is_file():
        raise FileNotFoundError(intrinsic_path)
    intrinsic = np.loadtxt(intrinsic_path)
    records: dict[int, Mapping[str, Any]] = {}
    hashes: dict[str, str] = {
        str(intrinsic_path.resolve()): file_sha256(intrinsic_path)
    }
    for raw_frame_id in frame_ids:
        frame_id = int(raw_frame_id)
        depth_path = scene_root / "depth" / f"{frame_id}.png"
        pose_path = scene_root / "pose" / f"{frame_id}.txt"
        if not depth_path.is_file() or not pose_path.is_file():
            raise FileNotFoundError(
                f"{scene_id}: missing depth/pose for frame {frame_id}"
            )
        depth = np.asarray(Image.open(depth_path))
        pose = np.loadtxt(pose_path)
        points, valid_ratio = backproject_depth(
            depth,
            intrinsic,
            pose,
            depth_scale=depth_scale,
            stride=stride,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        if len(points) and len(baseline_corners):
            explained = points_explained_by_boxes(
                points,
                baseline_corners,
                margin=float(explained_margin),
            )
            points = np.ascontiguousarray(points[~explained])
        records[frame_id] = {
            "geometry_points_world": points,
            "camera_position": np.asarray(pose[:3, 3], dtype=np.float32),
            "valid_depth_ratio": float(valid_ratio),
        }
        hashes[str(depth_path.resolve())] = file_sha256(depth_path)
        hashes[str(pose_path.resolve())] = file_sha256(pose_path)
    return records, hashes


def attach_geometry_to_observations(
    step_observations: Sequence[Any],
    geometry: Mapping[int, Mapping[str, Any]],
) -> tuple[Any, ...]:
    """Create structural P1G observations without mutating P1S dataclasses."""

    result = []
    for row in step_observations:
        frame_id = int(row.frame_index)
        if frame_id not in geometry:
            raise ValueError(f"missing scheduled geometry for frame {frame_id}")
        view = geometry[frame_id]
        result.append(
            SimpleNamespace(
                frame_index=frame_id,
                provider_step=int(row.provider_step),
                proposals=tuple(row.proposals),
                geometry_points_world=view["geometry_points_world"],
                camera_position=view["camera_position"],
            )
        )
    return tuple(result)


def gt_boxes_in_world(
    gt_aligned: np.ndarray, axis_alignment: np.ndarray
) -> np.ndarray:
    """Convert aligned ScanNet AABBs into world-frame AABB enclosures."""

    if len(gt_aligned) == 0:
        return np.empty((0, 6), dtype=np.float64)
    corners = center_size_to_corners(gt_aligned)
    world = transform_points(corners, np.linalg.inv(axis_alignment))
    lower = world.min(axis=1)
    upper = world.max(axis=1)
    return np.concatenate(((lower + upper) * 0.5, upper - lower), axis=1)


def bounded_face_oracle(
    candidate_box: np.ndarray,
    target_box: np.ndarray,
    maximum_face_shift_ratio: float,
) -> np.ndarray:
    """Move each AABB face toward GT under one normalized shift limit."""

    candidate = np.asarray(candidate_box, dtype=np.float64)
    target = np.asarray(target_box, dtype=np.float64)
    ratio = float(maximum_face_shift_ratio)
    if (
        candidate.shape != (6,)
        or target.shape != (6,)
        or not np.isfinite(candidate).all()
        or not np.isfinite(target).all()
        or np.any(candidate[3:] <= 0.0)
        or np.any(target[3:] <= 0.0)
        or not math.isfinite(ratio)
        or ratio < 0.0
    ):
        raise ValueError("invalid bounded face-oracle inputs")
    candidate_lower = candidate[:3] - 0.5 * candidate[3:]
    candidate_upper = candidate[:3] + 0.5 * candidate[3:]
    target_lower = target[:3] - 0.5 * target[3:]
    target_upper = target[:3] + 0.5 * target[3:]
    bound = ratio * candidate[3:]
    refined_lower = candidate_lower + np.clip(
        target_lower - candidate_lower, -bound, bound
    )
    refined_upper = candidate_upper + np.clip(
        target_upper - candidate_upper, -bound, bound
    )
    extent = refined_upper - refined_lower
    if np.any(extent <= 1e-8):
        raise RuntimeError("bounded face oracle produced invalid extents")
    return np.concatenate(
        ((refined_lower + refined_upper) * 0.5, extent)
    )


def _max_iou(boxes: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if not len(targets):
        return np.empty((0,), dtype=np.float64)
    if not len(boxes):
        return np.zeros(len(targets), dtype=np.float64)
    return pairwise_aabb_iou(boxes, targets).max(axis=0)


def feasibility_sweep(
    *,
    candidate_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    baseline_boxes: np.ndarray,
    face_limits: Sequence[float] = DEFAULT_FACE_LIMITS,
    covered_iou: float = 0.15,
    initial_min_iou: float = 0.15,
    initial_max_iou: float = 0.50,
) -> dict[str, np.ndarray]:
    """Evaluate the exact face-bound oracle once per residual GT."""

    candidates = np.asarray(candidate_boxes, dtype=np.float64).reshape(-1, 6)
    targets = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 6)
    baseline = np.asarray(baseline_boxes, dtype=np.float64).reshape(-1, 6)
    limits = np.asarray(tuple(face_limits), dtype=np.float64)
    if (
        limits.ndim != 1
        or not len(limits)
        or not np.isfinite(limits).all()
        or np.any(limits < 0.0)
    ):
        raise ValueError("face_limits must be finite and non-negative")
    baseline_best = _max_iou(baseline, targets)
    candidate_iou = pairwise_aabb_iou(candidates, targets)
    target_indices = []
    candidate_indices = []
    initial_ious = []
    refined_ious = []
    for target_index in np.flatnonzero(
        baseline_best <= float(covered_iou)
    ):
        if not len(candidates):
            continue
        candidate_index = int(np.argmax(candidate_iou[:, target_index]))
        initial = float(candidate_iou[candidate_index, target_index])
        if not (
            initial > float(initial_min_iou)
            and initial <= float(initial_max_iou)
        ):
            continue
        values = []
        for limit in limits:
            refined = bounded_face_oracle(
                candidates[candidate_index],
                targets[target_index],
                float(limit),
            )
            values.append(
                float(
                    pairwise_aabb_iou(
                        refined[None], targets[target_index][None]
                    )[0, 0]
                )
            )
        target_indices.append(int(target_index))
        candidate_indices.append(candidate_index)
        initial_ious.append(initial)
        refined_ious.append(values)
    return {
        "face_limits": limits.astype(np.float32),
        "target_indices": np.asarray(target_indices, dtype=np.int64),
        "candidate_indices": np.asarray(candidate_indices, dtype=np.int64),
        "initial_iou": np.asarray(initial_ious, dtype=np.float32),
        "refined_iou": np.asarray(refined_ious, dtype=np.float32).reshape(
            -1, len(limits)
        ),
        "baseline_best_iou": baseline_best.astype(np.float32),
    }


def actual_iou_diagnostics(
    *,
    parent_boxes: np.ndarray,
    refined_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    baseline_boxes: np.ndarray,
    covered_iou: float,
) -> dict[str, np.ndarray]:
    """Compare MSR boxes with the same original-best GT per candidate."""

    parent = np.asarray(parent_boxes, dtype=np.float64).reshape(-1, 6)
    refined = np.asarray(refined_boxes, dtype=np.float64).reshape(-1, 6)
    targets = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 6)
    if len(parent) != len(refined):
        raise ValueError("parent/refined candidates must align")
    baseline_best = _max_iou(
        np.asarray(baseline_boxes, dtype=np.float64).reshape(-1, 6),
        targets,
    )
    if not len(parent) or not len(targets):
        count = len(parent)
        return {
            "matched_gt_index": np.full(count, -1, dtype=np.int64),
            "matched_gt_is_residual": np.zeros(count, dtype=bool),
            "original_iou": np.zeros(count, dtype=np.float32),
            "refined_same_gt_iou": np.zeros(count, dtype=np.float32),
            "refined_best_iou": np.zeros(count, dtype=np.float32),
        }
    original_matrix = pairwise_aabb_iou(parent, targets)
    refined_matrix = pairwise_aabb_iou(refined, targets)
    matched = np.argmax(original_matrix, axis=1).astype(np.int64)
    rows = np.arange(len(parent), dtype=np.int64)
    return {
        "matched_gt_index": matched,
        "matched_gt_is_residual": (
            baseline_best[matched] <= float(covered_iou)
        ),
        "original_iou": original_matrix[rows, matched].astype(np.float32),
        "refined_same_gt_iou": refined_matrix[rows, matched].astype(
            np.float32
        ),
        "refined_best_iou": refined_matrix.max(axis=1).astype(np.float32),
    }


def _json_scalar(value: Any) -> np.ndarray:
    return np.asarray(
        json.dumps(value, sort_keys=True, separators=(",", ":"))
    )


def _scene_summary(
    scene_id: str,
    *,
    candidates: Sequence[Any],
    p1g_rows: Sequence[Any],
    actual: Mapping[str, np.ndarray],
    feasibility: Mapping[str, np.ndarray],
    runtime_seconds: float,
) -> dict[str, Any]:
    original = np.asarray(actual["original_iou"])
    refined = np.asarray(actual["refined_same_gt_iou"])
    residual = np.asarray(actual["matched_gt_is_residual"], dtype=bool)
    valid = residual & (original > 0.0)
    delta = refined - original
    delta_epsilon = 1e-6
    ratios = np.asarray(feasibility["face_limits"])
    oracle = np.asarray(feasibility["refined_iou"])
    return {
        "scene_id": scene_id,
        "candidate_count": len(candidates),
        "p1g_row_count": len(p1g_rows),
        "p1g_changed_count": int(
            sum(bool(row.is_candidate) for row in p1g_rows)
        ),
        "p1g_failure_count": int(
            sum(str(row.reason).startswith("identity_exception:") for row in p1g_rows)
        ),
        "p1g_runtime_seconds": float(runtime_seconds),
        "actual": {
            "matched_residual_count": int(np.count_nonzero(valid)),
            "improved_count": int(
                np.count_nonzero(valid & (delta > delta_epsilon))
            ),
            "degraded_count": int(
                np.count_nonzero(valid & (delta < -delta_epsilon))
            ),
            "cross_iou50": int(
                np.count_nonzero(
                    valid & (original <= 0.50) & (refined > 0.50)
                )
            ),
            "fall_iou50": int(
                np.count_nonzero(
                    valid & (original > 0.50) & (refined <= 0.50)
                )
            ),
            "median_delta_iou": (
                float(np.median(delta[valid]))
                if np.any(valid)
                else None
            ),
        },
        "feasibility": {
            f"{float(limit):.2f}": {
                "sample_count": int(len(oracle)),
                "cross_iou50": (
                    int(np.count_nonzero(oracle[:, index] > 0.50))
                    if len(oracle)
                    else 0
                ),
            }
            for index, limit in enumerate(ratios)
        },
    }


def replay_scene(
    *,
    scene_id: str,
    diagnostics_root: Path,
    prediction_root: Path,
    gt_root: Path,
    scans_root: Path,
    frames_root: Path,
    parents: FrozenParents,
    output_root: Path,
    face_limits: Sequence[float],
    covered_iou: float,
    depth_stride: int,
    depth_scale: float,
    min_depth: float,
    max_depth: float,
    explained_margin: float,
    p1g_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay and write one train-only scene."""

    diagnostic_path = diagnostics_root / f"{scene_id}_tracks.npz"
    prediction_path = prediction_root / f"{scene_id}_boxes.pkl"
    gt_path = gt_root / f"{scene_id}_bbox.npy"
    alignment_path = scans_root / scene_id / f"{scene_id}.txt"
    artifact_hashes = validate_scene_artifact_binding(
        scene_id=scene_id,
        inputs_path=diagnostic_path,
        prediction_path=prediction_path,
        gt_path=gt_path,
        alignment_path=alignment_path,
        expected=parents.scene_summaries[scene_id],
    )
    inputs = load_replay_scene_inputs(
        diagnostic_path, expected_scene_id=scene_id
    )
    baseline_corners = load_prediction_corners(prediction_path)
    baseline_boxes = corners_to_center_size(baseline_corners)
    anchors, step_rows = replay_p1s_candidates(
        inputs,
        parents.proposal_observer,
        baseline_corners=baseline_corners,
    )
    geometry, frame_hashes = load_scheduled_geometry(
        scene_id=scene_id,
        frame_ids=inputs.frame_ids,
        frames_root=frames_root,
        baseline_corners=baseline_corners,
        stride=depth_stride,
        depth_scale=depth_scale,
        min_depth=min_depth,
        max_depth=max_depth,
        explained_margin=explained_margin,
    )
    observations = attach_geometry_to_observations(step_rows, geometry)
    geometry_observer = P1MultiViewGeometryObserver(
        p1g_config,
        parent_checkpoint_sha256=parents.p1s_sha256,
    )
    p1g_rows = geometry_observer.observe_scene(
        scene_id=scene_id,
        anchors=anchors,
        observations=observations,
    )
    if len(p1g_rows) != len(anchors):
        raise RuntimeError("P1G observer violated one-to-one candidate contract")
    parent_boxes = (
        np.stack([row.box for row in anchors]).astype(np.float32)
        if anchors
        else np.empty((0, 6), dtype=np.float32)
    )
    refined_boxes = (
        np.stack([row.refined_box for row in p1g_rows]).astype(np.float32)
        if p1g_rows
        else np.empty((0, 6), dtype=np.float32)
    )
    gt_aligned = load_gt_boxes(gt_path)
    alignment = load_axis_alignment(scans_root, scene_id)
    gt_world = gt_boxes_in_world(gt_aligned, alignment)
    feasibility = feasibility_sweep(
        candidate_boxes=parent_boxes,
        gt_boxes=gt_world,
        baseline_boxes=baseline_boxes,
        face_limits=face_limits,
        covered_iou=covered_iou,
    )
    actual = actual_iou_diagnostics(
        parent_boxes=parent_boxes,
        refined_boxes=refined_boxes,
        gt_boxes=gt_world,
        baseline_boxes=baseline_boxes,
        covered_iou=covered_iou,
    )
    summary = _scene_summary(
        scene_id,
        candidates=anchors,
        p1g_rows=p1g_rows,
        actual=actual,
        feasibility=feasibility,
        runtime_seconds=geometry_observer.runtime_seconds,
    )
    p1g_payload = geometry_observer.diagnostic_payload()
    candidate_ids = np.asarray(
        [row.candidate_id for row in anchors], dtype=np.str_
    )
    candidate_scores = np.asarray(
        [row.objectness for row in anchors], dtype=np.float32
    )
    candidate_frame_ids = np.asarray(
        [row.frame_index for row in anchors], dtype=np.int64
    )
    candidate_provider_steps = np.asarray(
        [row.provider_step for row in anchors], dtype=np.int64
    )
    provenance = {
        "schema": OUTPUT_SCHEMA,
        "scene_id": scene_id,
        "p1s_checkpoint_sha256": parents.p1s_sha256,
        "b6_checkpoint_sha256": parents.b6_sha256,
        **artifact_hashes,
        "scheduled_frame_files": frame_hashes,
        "geometry_source": "scheduled_depth_minus_frozen_final_b6_boxes",
    }
    payload: dict[str, np.ndarray] = {
        "schema": np.asarray(OUTPUT_SCHEMA),
        "scene_id": np.asarray(scene_id),
        "offline_uses_ground_truth": np.asarray(True, dtype=bool),
        "observer_only": np.asarray(True, dtype=bool),
        "mutation_enabled": np.asarray(False, dtype=bool),
        "applied_count": np.asarray(0, dtype=np.int64),
        "p1s_checkpoint_sha256": np.asarray(parents.p1s_sha256),
        "b6_checkpoint_sha256": np.asarray(parents.b6_sha256),
        "candidate_ids": candidate_ids,
        "candidate_scores": candidate_scores,
        "candidate_frame_ids": candidate_frame_ids,
        "candidate_provider_steps": candidate_provider_steps,
        "candidate_boxes": parent_boxes,
        "refined_boxes": refined_boxes,
        "gt_boxes_world": gt_world.astype(np.float32),
        "baseline_boxes": baseline_boxes.astype(np.float32),
        "actual_matched_gt_indices": actual["matched_gt_index"],
        "actual_matched_gt_is_residual": actual[
            "matched_gt_is_residual"
        ],
        "actual_original_iou": actual["original_iou"],
        "actual_refined_same_gt_iou": actual["refined_same_gt_iou"],
        "actual_refined_best_iou": actual["refined_best_iou"],
        "feasibility_face_limits": feasibility["face_limits"],
        "feasibility_target_indices": feasibility["target_indices"],
        "feasibility_candidate_indices": feasibility["candidate_indices"],
        "feasibility_initial_iou": feasibility["initial_iou"],
        "feasibility_refined_iou": feasibility["refined_iou"],
        "baseline_best_gt_iou": feasibility["baseline_best_iou"],
        "provenance_json": _json_scalar(provenance),
        "summary_json": _json_scalar(summary),
    }
    for name, value in p1g_payload.items():
        if name in payload:
            raise RuntimeError(f"P1G diagnostic key collision: {name}")
        payload[name] = value
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{scene_id}_p1g_train_msr.npz"
    np.savez_compressed(output_path, **payload)
    # Prove immediately that no object array was accidentally persisted.
    with np.load(output_path, allow_pickle=False) as archive:
        if any(archive[name].dtype.hasobject for name in archive.files):
            raise RuntimeError("P1G replay wrote an object-dtype array")
    summary["output"] = str(output_path.resolve())
    summary["output_sha256"] = file_sha256(output_path)
    return summary


def aggregate_summaries(
    summaries: Sequence[Mapping[str, Any]],
    *,
    face_limits: Sequence[float],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate integer diagnostics without re-reading GT or predictions."""

    ratios = tuple(float(value) for value in face_limits)
    actual_keys = (
        "matched_residual_count",
        "improved_count",
        "degraded_count",
        "cross_iou50",
        "fall_iou50",
    )
    return {
        "schema": SUMMARY_SCHEMA,
        "scene_count": len(summaries),
        "scene_ids": [str(row["scene_id"]) for row in summaries],
        "candidate_count": int(
            sum(int(row["candidate_count"]) for row in summaries)
        ),
        "p1g_changed_count": int(
            sum(int(row["p1g_changed_count"]) for row in summaries)
        ),
        "p1g_failure_count": int(
            sum(int(row["p1g_failure_count"]) for row in summaries)
        ),
        "p1g_runtime_seconds": float(
            sum(float(row["p1g_runtime_seconds"]) for row in summaries)
        ),
        "actual": {
            key: int(sum(int(row["actual"][key]) for row in summaries))
            for key in actual_keys
        },
        "feasibility": {
            f"{ratio:.2f}": {
                "sample_count": int(
                    sum(
                        int(row["feasibility"][f"{ratio:.2f}"][
                            "sample_count"
                        ])
                        for row in summaries
                    )
                ),
                "cross_iou50": int(
                    sum(
                        int(row["feasibility"][f"{ratio:.2f}"][
                            "cross_iou50"
                        ])
                        for row in summaries
                    )
                ),
            }
            for ratio in ratios
        },
        "provenance": dict(provenance),
        "scenes": list(summaries),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--forbidden-scene-list", type=Path, required=True)
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--scans-root", type=Path, required=True)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--p1s-checkpoint", type=Path, required=True)
    parser.add_argument("--b6-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument(
        "--face-limits",
        type=float,
        nargs="+",
        default=list(DEFAULT_FACE_LIMITS),
    )
    parser.add_argument("--covered-iou", type=float, default=0.15)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--max-scene-candidates", type=int, default=256)
    parser.add_argument("--depth-stride", type=int, default=4)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--min-depth", type=float, default=0.15)
    parser.add_argument("--max-depth", type=float, default=8.0)
    parser.add_argument("--explained-margin", type=float, default=0.05)
    parser.add_argument("--association-iou", type=float, default=0.10)
    parser.add_argument("--crop-scale", type=float, default=1.35)
    parser.add_argument("--top-k-views", type=int, default=5)
    parser.add_argument("--max-points-per-view", type=int, default=768)
    parser.add_argument("--view-diversity-weight", type=float, default=0.25)
    parser.add_argument("--msr-max-face-shift-ratio", type=float, default=0.18)
    parser.add_argument("--msr-min-extent-ratio", type=float, default=0.70)
    parser.add_argument("--msr-max-extent-ratio", type=float, default=1.25)
    parser.add_argument("--msr-max-center-shift-ratio", type=float, default=0.15)
    parser.add_argument(
        "--msr-evidence-profile",
        choices=("conservative", "permissive"),
        default="conservative",
        help=(
            "Controlled train-only diagnostic. permissive relaxes only "
            "occupancy/face evidence gates; it never enables mutation."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    scenes = read_scene_ids(args.scene_list, role="P1G replay")
    forbidden = read_scene_ids(
        args.forbidden_scene_list, role="forbidden"
    )
    validate_scene_partition(scenes, forbidden)
    for role, root in (
        ("diagnostics", args.diagnostics_root),
        ("predictions", args.prediction_root),
        ("ground truth", args.gt_root),
        ("scans", args.scans_root),
        ("frames", args.frames_root),
    ):
        if not root.is_dir():
            raise FileNotFoundError(f"{role} root not found: {root}")
    parents = load_frozen_parents(
        args.p1s_checkpoint,
        args.b6_checkpoint,
        requested_scenes=scenes,
        forbidden_scene_list=args.forbidden_scene_list,
        score_threshold=args.score_threshold,
        max_scene_candidates=args.max_scene_candidates,
    )
    proposal_config = {
        "maximum_face_shift_ratio": float(
            args.msr_max_face_shift_ratio
        ),
        "minimum_extent_ratio": float(args.msr_min_extent_ratio),
        "maximum_extent_ratio": float(args.msr_max_extent_ratio),
        "maximum_center_shift_ratio": float(
            args.msr_max_center_shift_ratio
        ),
    }
    if args.msr_evidence_profile == "permissive":
        # Keep the two-view requirement so this remains a genuine
        # multi-view test.  Only evidence thresholds are relaxed.  Comparing
        # this profile with the conservative profile separates an internal
        # gate/parameter bottleneck from wrong point association or boundary
        # estimation.
        proposal_config.update(
            {
                "min_total_points": 64,
                "min_component_points": 32,
                "min_component_inside_fraction": 0.20,
                "face_min_points_per_view": 4,
                "face_max_uncertainty_ratio": 0.35,
                "face_min_support": 0.10,
                "face_min_empty_evidence": 0.0,
                "maximum_support_drop": 0.20,
            }
        )
    p1g_config = {
        "enabled": True,
        "observer_only": True,
        "mutate": False,
        "collect_diagnostics": True,
        "association_iou": float(args.association_iou),
        "crop_scale": float(args.crop_scale),
        "top_k_views": int(args.top_k_views),
        "view_diversity_weight": float(args.view_diversity_weight),
        "max_points_per_view": int(args.max_points_per_view),
        "max_candidates": int(args.max_scene_candidates),
        "proposal": proposal_config,
    }
    run_provenance = {
        "scene_list": str(args.scene_list.resolve()),
        "scene_list_sha256": file_sha256(args.scene_list),
        "forbidden_scene_list": str(
            args.forbidden_scene_list.resolve()
        ),
        "forbidden_scene_list_sha256": file_sha256(
            args.forbidden_scene_list
        ),
        "forbidden_overlap": [],
        "p1s_checkpoint": str(args.p1s_checkpoint.resolve()),
        "p1s_checkpoint_sha256": parents.p1s_sha256,
        "b6_checkpoint": str(args.b6_checkpoint.resolve()),
        "b6_checkpoint_sha256": parents.b6_sha256,
        "face_limits": [float(value) for value in args.face_limits],
        "covered_iou": float(args.covered_iou),
        "msr_evidence_profile": str(args.msr_evidence_profile),
        "p1g_config": p1g_config,
    }
    summaries = []
    for index, scene_id in enumerate(scenes, start=1):
        print(f"[{index}/{len(scenes)}] replaying {scene_id}", flush=True)
        summary = replay_scene(
            scene_id=scene_id,
            diagnostics_root=args.diagnostics_root,
            prediction_root=args.prediction_root,
            gt_root=args.gt_root,
            scans_root=args.scans_root,
            frames_root=args.frames_root,
            parents=parents,
            output_root=args.output_root,
            face_limits=args.face_limits,
            covered_iou=args.covered_iou,
            depth_stride=args.depth_stride,
            depth_scale=args.depth_scale,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            explained_margin=args.explained_margin,
            p1g_config=p1g_config,
        )
        summaries.append(summary)
        print(
            f"[{index}/{len(scenes)}] {scene_id}: "
            f"candidates={summary['candidate_count']}, "
            f"changed={summary['p1g_changed_count']}, "
            f"cross50={summary['actual']['cross_iou50']}",
            flush=True,
        )
    report = aggregate_summaries(
        summaries,
        face_limits=args.face_limits,
        provenance=run_provenance,
    )
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
