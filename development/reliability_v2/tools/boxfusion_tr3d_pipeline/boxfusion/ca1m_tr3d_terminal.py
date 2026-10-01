"""GT-free terminal TR3D candidate observation for processed CA-1M scenes.

The ScanNet R3 implementation cannot be reused directly: it freezes ScanNet
scene identifiers, trajectory clocks, axis-alignment provenance, and cache
schemas.  This module keeps the *method* (terminal class-agnostic TR3D
proposals associated to the current BoxFusion rows) while defining a small,
CA-1M-specific geometry contract.

This first stage is deliberately observer-only.  It never exposes a function
that writes active predictions.  A train-only replacement policy must be
learned and audited before any candidate geometry can replace an anchor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

import numpy as np


SCHEMA = "boxfusion.ca1m_tr3d_terminal_observer.v1"
COORDINATE_FRAME = "ca1m_gravity_aligned_world"
BOX_MODE = "depth_center_size_yaw_z"
CORNER_SEMANTICS = "unordered_8_corners_world_enclosing_aabb_eval"
DEFAULT_NEAR_IOU = 0.15

_SCENE_RE = re.compile(r"^[0-9]{8}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CORNER_SIGNS = np.asarray(
    [
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1],
    ],
    dtype=np.float64,
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_scene_id(scene_id: str) -> str:
    value = str(scene_id)
    if _SCENE_RE.fullmatch(value) is None:
        raise ValueError(f"invalid CA-1M scene id: {value!r}")
    return value


def validate_sha256(value: str, name: str) -> str:
    normalized = str(value).lower()
    if _SHA256_RE.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return normalized


def validate_homogeneous(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8)
    ):
        raise ValueError(f"{name} must be a finite homogeneous [4,4] matrix")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
    ):
        raise ValueError(f"{name} must contain a proper rigid rotation")
    return np.ascontiguousarray(matrix)


def terminal_world_to_local(first_camera_to_world: Any) -> np.ndarray:
    """Return a deterministic translation-only numerical normalization.

    Processed CA-1M is already gravity aligned.  We intentionally do not infer
    a Manhattan yaw from validation data.  Removing only the first observed
    camera translation keeps the TR3D sparse coordinates compact while the
    exact inverse maps every proposal back to the published world frame.
    """

    camera_to_world = validate_homogeneous(
        first_camera_to_world, "first_camera_to_world"
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = -camera_to_world[:3, 3]
    return result


def depth_box_corners(boxes: Any) -> np.ndarray:
    values = np.asarray(boxes, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] not in {6, 7}:
        raise ValueError("boxes must have shape [N,6] or [N,7]")
    if values.shape[1] == 6:
        values = np.concatenate(
            (values, np.zeros((len(values), 1), dtype=np.float64)), axis=1
        )
    if (
        not np.isfinite(values).all()
        or (len(values) and np.any(values[:, 3:6] <= 0.0))
    ):
        raise ValueError("boxes must be finite with positive extents")
    local = _CORNER_SIGNS[None] * (0.5 * values[:, None, 3:6])
    cosine = np.cos(values[:, 6])
    sine = np.sin(values[:, 6])
    result = np.empty_like(local)
    result[:, :, 0] = local[:, :, 0] * cosine[:, None] - local[:, :, 1] * sine[:, None]
    result[:, :, 1] = local[:, :, 0] * sine[:, None] + local[:, :, 1] * cosine[:, None]
    result[:, :, 2] = local[:, :, 2]
    return result + values[:, None, :3]


def aligned_boxes_to_world_corners(
    boxes_local: Any, world_to_local: Any
) -> np.ndarray:
    transform = validate_homogeneous(world_to_local, "world_to_local")
    local = depth_box_corners(boxes_local)
    local_to_world = np.linalg.inv(transform)
    world = (
        local @ local_to_world[:3, :3].T
        + local_to_world[None, None, :3, 3]
    )
    return np.ascontiguousarray(world, dtype=np.float32)


def voxel_downsample_first(points: Any, voxel_size: float) -> np.ndarray:
    values = np.asarray(points)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("points must have shape [N,6]")
    if not np.isfinite(values).all():
        raise ValueError("points must be finite")
    if not math.isfinite(float(voxel_size)) or float(voxel_size) <= 0.0:
        raise ValueError("voxel_size must be finite and positive")
    if not len(values):
        return np.empty((0, 6), dtype=np.float32)
    keys = np.floor(values[:, :3] / float(voxel_size)).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return np.ascontiguousarray(values[np.sort(first)], dtype=np.float32)


def world_aabb(corners: Any) -> np.ndarray:
    values = np.asarray(corners, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (8, 3):
        raise ValueError("corners must have shape [N,8,3]")
    if not np.isfinite(values).all():
        raise ValueError("corners must be finite")
    if not len(values):
        return np.empty((0, 6), dtype=np.float64)
    result = np.concatenate((values.min(axis=1), values.max(axis=1)), axis=1)
    if np.any(result[:, 3:] <= result[:, :3]):
        raise ValueError("corners must define positive-volume boxes")
    return result


def pairwise_world_aabb_iou(left_corners: Any, right_corners: Any) -> np.ndarray:
    left = world_aabb(left_corners)
    right = world_aabb(right_corners)
    if not len(left) or not len(right):
        return np.zeros((len(left), len(right)), dtype=np.float64)
    overlap = np.maximum(
        np.minimum(left[:, None, 3:], right[None, :, 3:])
        - np.maximum(left[:, None, :3], right[None, :, :3]),
        0.0,
    )
    intersection = np.prod(overlap, axis=2)
    left_volume = np.prod(left[:, 3:] - left[:, :3], axis=1)
    right_volume = np.prod(right[:, 3:] - right[:, :3], axis=1)
    union = left_volume[:, None] + right_volume[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


@dataclass(frozen=True)
class TerminalAssociation:
    best_anchor_indices: np.ndarray
    best_anchor_iou: np.ndarray
    best_anchor_center_distance_m: np.ndarray
    near_mask: np.ndarray
    represented_anchor_indices: np.ndarray
    legacy_rule_selected_candidate_rows: np.ndarray
    legacy_rule_selected_anchor_indices: np.ndarray


def associate_terminal_candidates(
    *,
    anchor_corners: Any,
    anchor_scores: Any,
    candidate_corners: Any,
    candidate_scores: Any,
    near_iou: float = DEFAULT_NEAR_IOU,
) -> TerminalAssociation:
    """Associate candidates without authorizing any geometry mutation.

    ``legacy_rule_*`` records what the frozen ScanNet heuristic would have
    selected.  It is diagnostic only; CA-1M activation requires a train-only
    policy and is intentionally absent from this module.
    """

    anchors = np.asarray(anchor_corners)
    scores = np.asarray(anchor_scores, dtype=np.float64)
    candidates = np.asarray(candidate_corners)
    candidate_confidence = np.asarray(candidate_scores, dtype=np.float64)
    world_aabb(anchors)
    world_aabb(candidates)
    if scores.shape != (len(anchors),) or candidate_confidence.shape != (len(candidates),):
        raise ValueError("scores must align with their corner rows")
    if not np.isfinite(scores).all() or not np.isfinite(candidate_confidence).all():
        raise ValueError("scores must be finite")
    threshold = float(near_iou)
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("near_iou must be finite and in (0,1)")

    if not len(candidates):
        empty_i64 = np.empty((0,), dtype=np.int64)
        return TerminalAssociation(
            best_anchor_indices=empty_i64,
            best_anchor_iou=np.empty((0,), dtype=np.float32),
            best_anchor_center_distance_m=np.empty((0,), dtype=np.float32),
            near_mask=np.empty((0,), dtype=np.bool_),
            represented_anchor_indices=empty_i64,
            legacy_rule_selected_candidate_rows=empty_i64,
            legacy_rule_selected_anchor_indices=empty_i64,
        )
    if not len(anchors):
        return TerminalAssociation(
            best_anchor_indices=np.full(len(candidates), -1, dtype=np.int64),
            best_anchor_iou=np.zeros(len(candidates), dtype=np.float32),
            best_anchor_center_distance_m=np.full(len(candidates), 1.0e6, dtype=np.float32),
            near_mask=np.zeros(len(candidates), dtype=np.bool_),
            represented_anchor_indices=np.empty((0,), dtype=np.int64),
            legacy_rule_selected_candidate_rows=np.empty((0,), dtype=np.int64),
            legacy_rule_selected_anchor_indices=np.empty((0,), dtype=np.int64),
        )

    iou = pairwise_world_aabb_iou(candidates, anchors)
    candidate_centers = candidates.astype(np.float64).mean(axis=1)
    anchor_centers = anchors.astype(np.float64).mean(axis=1)
    distance = np.linalg.norm(
        candidate_centers[:, None] - anchor_centers[None], axis=2
    )
    best_iou = iou.max(axis=1)
    best_anchor = np.empty(len(candidates), dtype=np.int64)
    best_distance = np.empty(len(candidates), dtype=np.float64)
    for row in range(len(candidates)):
        tied = np.flatnonzero(iou[row] == best_iou[row])
        nearest = int(tied[int(np.argmin(distance[row, tied]))])
        best_anchor[row] = nearest
        best_distance[row] = distance[row, nearest]
    near_mask = best_iou > threshold
    represented = np.unique(best_anchor[near_mask])

    selected_rows: list[int] = []
    selected_anchors: list[int] = []
    for anchor in represented.tolist():
        rows = np.flatnonzero(near_mask & (best_anchor == anchor))
        order = np.lexsort((rows, -candidate_confidence[rows]))
        candidate_row = int(rows[int(order[0])])
        if candidate_confidence[candidate_row] > scores[anchor]:
            selected_rows.append(candidate_row)
            selected_anchors.append(anchor)

    return TerminalAssociation(
        best_anchor_indices=np.ascontiguousarray(best_anchor, dtype=np.int64),
        best_anchor_iou=np.ascontiguousarray(best_iou, dtype=np.float32),
        best_anchor_center_distance_m=np.ascontiguousarray(best_distance, dtype=np.float32),
        near_mask=np.ascontiguousarray(near_mask, dtype=np.bool_),
        represented_anchor_indices=np.ascontiguousarray(represented, dtype=np.int64),
        legacy_rule_selected_candidate_rows=np.asarray(selected_rows, dtype=np.int64),
        legacy_rule_selected_anchor_indices=np.asarray(selected_anchors, dtype=np.int64),
    )


@dataclass(frozen=True)
class TerminalObserverSummary:
    scene_id: str
    anchor_count: int
    candidate_count: int
    near_candidate_count: int
    represented_anchor_count: int
    legacy_rule_selected_count: int
    used_frame_count: int
    point_count: int
    model_runtime_s: float
    source_anchor_prediction_sha256: str
    active_anchor_scores_sha256: str
    native_b6_diagnostic_sha256: str
    native_b6_checkpoint_sha256: str
    native_b6_manifest_sha256: str
    source_points_sha256: str
    checkpoint_sha256: str
    config_sha256: str
    code_manifest_sha256: str
    adapter_mode: str
    prefix_id: str
    device: str
    pixel_stride: int
    voxel_size_m: float
    min_depth_m: float
    max_depth_m: float
    near_iou: float
    score_threshold: float
    max_proposals: int
    materialized_active_verified: bool

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "schema": SCHEMA,
                "complete": True,
                "observer_only": True,
                "mutation_enabled": False,
                "applied_count": 0,
                "ground_truth_access": False,
                "validation_policy_selection_authorized": False,
                "coordinate_frame": COORDINATE_FRAME,
                "box_mode": BOX_MODE,
                "corner_semantics": CORNER_SEMANTICS,
                "legacy_rule_is_diagnostic_only": True,
            }
        )
        return result


def observation_payload(
    *,
    summary: TerminalObserverSummary,
    used_frame_ids: Any,
    world_to_local: Any,
    anchor_corners: Any,
    anchor_scores: Any,
    candidate_corners: Any,
    candidate_scores: Any,
    candidate_point_count: Any,
    candidate_boxes_local: Any,
    candidate_labels: Any,
    association: TerminalAssociation,
    code_manifest_json: str,
) -> dict[str, np.ndarray]:
    scene_id = validate_scene_id(summary.scene_id)
    for name, value in (
        ("source_anchor_prediction_sha256", summary.source_anchor_prediction_sha256),
        ("active_anchor_scores_sha256", summary.active_anchor_scores_sha256),
        ("native_b6_diagnostic_sha256", summary.native_b6_diagnostic_sha256),
        ("native_b6_checkpoint_sha256", summary.native_b6_checkpoint_sha256),
        ("native_b6_manifest_sha256", summary.native_b6_manifest_sha256),
        ("source_points_sha256", summary.source_points_sha256),
        ("checkpoint_sha256", summary.checkpoint_sha256),
        ("config_sha256", summary.config_sha256),
        ("code_manifest_sha256", summary.code_manifest_sha256),
    ):
        validate_sha256(value, name)
    frames = np.asarray(used_frame_ids)
    anchors = np.asarray(anchor_corners)
    scores = np.asarray(anchor_scores)
    candidates = np.asarray(candidate_corners)
    candidate_confidence = np.asarray(candidate_scores)
    point_count = np.asarray(candidate_point_count)
    local_boxes = np.asarray(candidate_boxes_local)
    labels = np.asarray(candidate_labels)
    if frames.dtype != np.dtype(np.int64) or frames.ndim != 1:
        raise ValueError("used_frame_ids must be int64 [V]")
    if len(frames) != len(np.unique(frames)) or np.any(frames < 0):
        raise ValueError("used_frame_ids must be unique and non-negative")
    if anchors.dtype != np.dtype(np.float32) or anchors.shape != (summary.anchor_count, 8, 3):
        raise ValueError("anchor_corners must be float32 [A,8,3]")
    if scores.dtype != np.dtype(np.float32) or scores.shape != (summary.anchor_count,):
        raise ValueError("anchor_scores must be float32 [A]")
    if candidates.dtype != np.dtype(np.float32) or candidates.shape != (summary.candidate_count, 8, 3):
        raise ValueError("candidate_corners must be float32 [C,8,3]")
    if candidate_confidence.dtype != np.dtype(np.float32) or candidate_confidence.shape != (summary.candidate_count,):
        raise ValueError("candidate_scores must be float32 [C]")
    if point_count.dtype != np.dtype(np.int64) or point_count.shape != (summary.candidate_count,):
        raise ValueError("candidate_point_count must be int64 [C]")
    if local_boxes.dtype != np.dtype(np.float32) or local_boxes.shape != (
        summary.candidate_count,
        7,
    ):
        raise ValueError("candidate_boxes_local must be float32 [C,7]")
    if labels.dtype != np.dtype(np.int64) or labels.shape != (
        summary.candidate_count,
    ):
        raise ValueError("candidate_labels must be int64 [C]")
    if np.any(labels != 0):
        raise ValueError("candidate_labels must be class-agnostic zero")
    if np.any(point_count < 0):
        raise ValueError("candidate_point_count must be non-negative")
    if not np.isfinite(anchors).all() or not np.isfinite(scores).all():
        raise ValueError("anchors must be finite")
    if not np.isfinite(candidates).all() or not np.isfinite(candidate_confidence).all():
        raise ValueError("candidates must be finite")
    if (
        np.any(scores < 0.0)
        or np.any(scores > 1.0)
        or np.any(candidate_confidence < 0.0)
        or np.any(candidate_confidence > 1.0)
    ):
        raise ValueError("anchor/candidate scores must be in [0,1]")
    candidate_count = summary.candidate_count
    if (
        association.best_anchor_indices.shape != (candidate_count,)
        or association.best_anchor_iou.shape != (candidate_count,)
        or association.best_anchor_center_distance_m.shape != (candidate_count,)
        or association.near_mask.shape != (candidate_count,)
        or not np.isfinite(association.best_anchor_iou).all()
        or not np.isfinite(association.best_anchor_center_distance_m).all()
        or np.any(association.best_anchor_iou < 0.0)
        or np.any(association.best_anchor_iou > 1.0)
    ):
        raise ValueError("terminal association arrays are malformed")
    if summary.used_frame_count != len(frames):
        raise ValueError("summary used_frame_count disagrees")
    if (
        summary.point_count < 1
        or not math.isfinite(summary.model_runtime_s)
        or summary.model_runtime_s < 0.0
    ):
        raise ValueError("summary point/runtime contract disagrees")
    if summary.near_candidate_count != int(association.near_mask.sum()):
        raise ValueError("summary near candidate count disagrees")
    if summary.represented_anchor_count != len(association.represented_anchor_indices):
        raise ValueError("summary represented anchor count disagrees")
    if summary.legacy_rule_selected_count != len(
        association.legacy_rule_selected_candidate_rows
    ):
        raise ValueError("summary legacy selection count disagrees")
    transform = validate_homogeneous(world_to_local, "world_to_local")
    reconstructed = aligned_boxes_to_world_corners(local_boxes, transform)
    if not np.array_equal(reconstructed, candidates):
        raise ValueError("candidate local boxes do not reconstruct world corners exactly")
    if summary.adapter_mode not in {"genuine", "synthetic"}:
        raise ValueError("adapter_mode must be genuine or synthetic")
    if summary.prefix_id != "p100_gap20" or not summary.device:
        raise ValueError("invalid terminal prefix/device contract")
    if (
        summary.pixel_stride < 1
        or not math.isfinite(summary.voxel_size_m)
        or summary.voxel_size_m <= 0.0
        or not 0.0 < summary.min_depth_m < summary.max_depth_m
        or not 0.0 < summary.near_iou < 1.0
        or not 0.0 <= summary.score_threshold <= 1.0
        or summary.max_proposals < 1
    ):
        raise ValueError("invalid terminal protocol parameters")
    try:
        parsed_code_manifest = json.loads(code_manifest_json)
    except json.JSONDecodeError as error:
        raise ValueError("code manifest must be valid JSON") from error
    canonical_code_manifest = json.dumps(parsed_code_manifest, sort_keys=True)
    if canonical_code_manifest != code_manifest_json:
        raise ValueError("code manifest JSON must be canonical")
    if sha256_bytes(code_manifest_json.encode("utf-8")) != summary.code_manifest_sha256:
        raise ValueError("code manifest SHA256 mismatch")
    payload = {
        "schema": np.asarray(SCHEMA),
        "complete": np.asarray(True, dtype=np.bool_),
        "observer_only": np.asarray(True, dtype=np.bool_),
        "mutation_enabled": np.asarray(False, dtype=np.bool_),
        "applied_count": np.asarray(0, dtype=np.int64),
        "ground_truth_access": np.asarray(False, dtype=np.bool_),
        "scene_id": np.asarray(scene_id),
        "coordinate_frame": np.asarray(COORDINATE_FRAME),
        "box_mode": np.asarray(BOX_MODE),
        "corner_semantics": np.asarray(CORNER_SEMANTICS),
        "source_anchor_prediction_sha256": np.asarray(
            summary.source_anchor_prediction_sha256
        ),
        "active_anchor_scores_sha256": np.asarray(
            summary.active_anchor_scores_sha256
        ),
        "native_b6_diagnostic_sha256": np.asarray(
            summary.native_b6_diagnostic_sha256
        ),
        "native_b6_checkpoint_sha256": np.asarray(
            summary.native_b6_checkpoint_sha256
        ),
        "native_b6_manifest_sha256": np.asarray(
            summary.native_b6_manifest_sha256
        ),
        "source_points_sha256": np.asarray(summary.source_points_sha256),
        "checkpoint_sha256": np.asarray(summary.checkpoint_sha256),
        "config_sha256": np.asarray(summary.config_sha256),
        "code_manifest_sha256": np.asarray(summary.code_manifest_sha256),
        "code_manifest_json": np.asarray(code_manifest_json),
        "adapter_mode": np.asarray(summary.adapter_mode),
        "prefix_id": np.asarray(summary.prefix_id),
        "device": np.asarray(summary.device),
        "pixel_stride": np.asarray(summary.pixel_stride, dtype=np.int64),
        "voxel_size_m": np.asarray(summary.voxel_size_m, dtype=np.float64),
        "min_depth_m": np.asarray(summary.min_depth_m, dtype=np.float64),
        "max_depth_m": np.asarray(summary.max_depth_m, dtype=np.float64),
        "near_iou": np.asarray(summary.near_iou, dtype=np.float64),
        "score_threshold": np.asarray(summary.score_threshold, dtype=np.float64),
        "max_proposals": np.asarray(summary.max_proposals, dtype=np.int64),
        "materialized_active_verified": np.asarray(
            summary.materialized_active_verified, dtype=np.bool_
        ),
        "point_count": np.asarray(summary.point_count, dtype=np.int64),
        "model_runtime_s": np.asarray(summary.model_runtime_s, dtype=np.float64),
        "used_frame_ids": np.ascontiguousarray(frames),
        "world_to_local": transform,
        "anchor_corners": np.ascontiguousarray(anchors),
        "anchor_scores": np.ascontiguousarray(scores),
        "candidate_corners": np.ascontiguousarray(candidates),
        "candidate_scores": np.ascontiguousarray(candidate_confidence),
        "candidate_point_count": np.ascontiguousarray(point_count),
        "candidate_boxes_local": np.ascontiguousarray(local_boxes),
        "candidate_labels": np.ascontiguousarray(labels),
        "best_anchor_indices": association.best_anchor_indices,
        "best_anchor_iou": association.best_anchor_iou,
        "best_anchor_center_distance_m": association.best_anchor_center_distance_m,
        "near_mask": association.near_mask,
        "represented_anchor_indices": association.represented_anchor_indices,
        "legacy_rule_selected_candidate_rows": association.legacy_rule_selected_candidate_rows,
        "legacy_rule_selected_anchor_indices": association.legacy_rule_selected_anchor_indices,
        "summary_json": np.asarray(json.dumps(summary.as_dict(), sort_keys=True)),
    }
    return payload


def write_npz_create_only(
    path: str | os.PathLike[str], payload: Mapping[str, np.ndarray]
) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    np.savez_compressed(buffer, **payload)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(buffer.getvalue())
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite terminal observer artifact: {target}"
            ) from error
        target.chmod(0o444)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return target


__all__ = [
    "BOX_MODE",
    "COORDINATE_FRAME",
    "CORNER_SEMANTICS",
    "DEFAULT_NEAR_IOU",
    "SCHEMA",
    "TerminalAssociation",
    "TerminalObserverSummary",
    "aligned_boxes_to_world_corners",
    "associate_terminal_candidates",
    "depth_box_corners",
    "observation_payload",
    "pairwise_world_aabb_iou",
    "sha256_array",
    "sha256_bytes",
    "sha256_file",
    "terminal_world_to_local",
    "validate_homogeneous",
    "validate_scene_id",
    "voxel_downsample_first",
    "world_aabb",
    "write_npz_create_only",
]
