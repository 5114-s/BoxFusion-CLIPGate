#!/usr/bin/env python3
"""Read-only ScanNet oracle audit for frozen Boxer geometry.

The frozen Boxer process produces raw per-view and terminal tracked OBB CSVs
without access to ScanNet ground truth.  This program is the *only* stage that
reads GT.  It restores Boxer's ScanNet ``world_offset``, applies the frozen
unexplained-depth test from
``docs/UNEXPLAINED_DEPTH_BOXER_ORACLE_PREREGISTRATION.md``, and reports
class-agnostic oracle headroom.  It never writes a prediction pickle.

The constant-score evaluator below intentionally reproduces the official
BoxFusion ScanNet behavior, including NumPy's default ``argsort`` tie order and
the strict ``IoU > threshold`` comparison.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image


SCHEMA = "boxfusion.scannet_boxer_unexplained_oracle.v1"
THRESHOLDS = (0.15, 0.25, 0.50)
POOL_NAMES = ("raw", "depth_gated", "tracked")
SCENE_PATTERN = re.compile(r"scene\d{4}_\d{2}")

PIXEL_STRIDE = 4
MIN_DEPTH_M = 0.10
MAX_DEPTH_M = 8.00
NATIVE_AABB_EXPANSION_M = 0.05
VOXEL_SIZE_M = 0.05
MIN_CANDIDATE_VOXELS = 16
MIN_UNEXPLAINED_VOXELS = 16
MIN_UNEXPLAINED_RATIO = 0.50
BOXER_START_N = 1
BOXER_SKIP_N = 25

CSV_FIELDS = (
    "time_ns",
    "tx_world_object",
    "ty_world_object",
    "tz_world_object",
    "qw_world_object",
    "qx_world_object",
    "qy_world_object",
    "qz_world_object",
    "scale_x",
    "scale_y",
    "scale_z",
    "name",
    "instance",
    "sem_id",
    "prob",
)

_SIGNS = np.asarray(
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


class OracleAuditError(ValueError):
    """Raised when an input violates the frozen audit contract."""


@dataclass(frozen=True)
class Candidate:
    """One Boxer OBB restored to the unrecentered ScanNet world frame."""

    scene_id: str
    source: str
    row_index: int
    frame_id: int
    instance_id: int
    name: str
    probability: float
    center_world: np.ndarray
    rotation_world_object: np.ndarray
    size: np.ndarray
    corners_world: np.ndarray
    aligned_minmax: np.ndarray

    @property
    def candidate_id(self) -> str:
        return f"{self.scene_id}:{self.source}:{self.row_index:06d}"


@dataclass(frozen=True)
class GateRow:
    candidate_id: str
    frame_id: int
    candidate_voxels: int
    unexplained_voxels: int
    unexplained_ratio: float
    accepted: bool
    reason: str


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise OracleAuditError(f"{label} must be a regular non-symlink file: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _threshold_key(threshold: float) -> str:
    return f"{threshold:.2f}"


def load_scene_list(path: Path) -> list[str]:
    _regular_file(path, "scene list")
    scenes: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        scene = line.strip()
        if not scene or scene.startswith("#"):
            continue
        if SCENE_PATTERN.fullmatch(scene) is None:
            raise OracleAuditError(
                f"invalid scene ID at {path}:{line_number}: {scene!r}"
            )
        if scene in scenes:
            raise OracleAuditError(f"duplicate scene in {path}: {scene}")
        scenes.append(scene)
    if not scenes:
        raise OracleAuditError(f"scene list is empty: {path}")
    return scenes


def load_sealed_schedule(path: Path, scene_id: str) -> tuple[list[int], str]:
    """Load the one authoritative keyframe schedule from a sealed cache manifest."""

    _regular_file(path, "sealed proposal-cache manifest")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OracleAuditError(f"invalid sealed schedule manifest: {path}") from error
    if not isinstance(payload, dict):
        raise OracleAuditError(f"sealed schedule manifest must be an object: {path}")
    if payload.get("schema") != "boxfusion.cutr_postfilter_cache.v3":
        raise OracleAuditError(f"unexpected sealed schedule schema: {path}")
    if payload.get("scene_id") != scene_id:
        raise OracleAuditError(
            f"sealed schedule scene mismatch: expected={scene_id}, actual={payload.get('scene_id')}"
        )
    namespace = payload.get("namespace")
    if not isinstance(namespace, str) or not namespace:
        raise OracleAuditError(f"invalid sealed schedule namespace: {path}")
    raw_ids = payload.get("recorded_frame_ids")
    if (
        not isinstance(raw_ids, list)
        or not raw_ids
        or any(type(value) is not int or value < 0 for value in raw_ids)
    ):
        raise OracleAuditError(f"invalid recorded_frame_ids in {path}")
    frame_ids = list(raw_ids)
    if frame_ids != sorted(set(frame_ids)):
        raise OracleAuditError(f"recorded_frame_ids must be unique and increasing: {path}")
    if payload.get("record_count") != len(frame_ids):
        raise OracleAuditError(f"record_count does not match recorded_frame_ids: {path}")
    schedule = payload.get("schedule")
    if not isinstance(schedule, dict) or schedule.get("gap") != BOXER_SKIP_N:
        raise OracleAuditError(f"sealed schedule gap is not {BOXER_SKIP_N}: {path}")
    if len(frame_ids) > 1 and any(
        right - left != BOXER_SKIP_N for left, right in zip(frame_ids, frame_ids[1:])
    ):
        raise OracleAuditError(f"sealed schedule is not a fixed gap-{BOXER_SKIP_N} prefix: {path}")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != len(frame_ids):
        raise OracleAuditError(f"sealed schedule records do not match record_count: {path}")
    record_ids: list[int] = []
    for record in records:
        if not isinstance(record, dict) or type(record.get("frame_id")) is not int:
            raise OracleAuditError(f"invalid sealed schedule record: {path}")
        record_ids.append(record["frame_id"])
    if record_ids != frame_ids:
        raise OracleAuditError(f"sealed record order differs from recorded_frame_ids: {path}")
    return frame_ids, namespace


def load_axis_alignment(path: Path) -> np.ndarray:
    _regular_file(path, "axis-alignment metadata")
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("axisAlignment"):
            try:
                values = [float(value) for value in line.split("=", 1)[1].split()]
            except (IndexError, ValueError) as error:
                raise OracleAuditError(f"malformed axisAlignment in {path}") from error
            if len(values) != 16:
                raise OracleAuditError(
                    f"axisAlignment in {path} has {len(values)} values"
                )
            matrix = np.asarray(values, dtype=np.float64).reshape(4, 4)
            if not np.isfinite(matrix).all():
                raise OracleAuditError(f"non-finite axisAlignment in {path}")
            return matrix
    raise OracleAuditError(f"axisAlignment missing in {path}")


def load_gt_minmax(path: Path) -> np.ndarray:
    _regular_file(path, "ScanNet GT")
    gt = np.load(path, allow_pickle=False).astype(np.float64)
    if gt.ndim != 2 or gt.shape[1] < 6 or not np.isfinite(gt[:, :6]).all():
        raise OracleAuditError(f"invalid ScanNet GT array: {path}")
    if np.any(gt[:, 3:6] < 0.0):
        raise OracleAuditError(f"negative ScanNet GT extent: {path}")
    return np.concatenate(
        (gt[:, :3] - gt[:, 3:6] / 2.0, gt[:, :3] + gt[:, 3:6] / 2.0),
        axis=1,
    )


def load_baseline_boxes(
    path: Path, alignment: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return native world AABBs and official aligned-frame AABBs."""

    _regular_file(path, "T05 prediction")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(
        payload[0], list
    ):
        raise OracleAuditError(f"unexpected prediction container: {path}")
    rows = payload[0]
    if not rows:
        empty = np.empty((0, 6), dtype=np.float64)
        return empty, empty.copy()
    try:
        corners = np.asarray([row[1] for row in rows], dtype=np.float64)
        scores = np.asarray([float(row[2]) for row in rows], dtype=np.float64)
    except (IndexError, TypeError, ValueError) as error:
        raise OracleAuditError(f"invalid prediction rows: {path}") from error
    if corners.shape != (len(rows), 8, 3) or not np.isfinite(corners).all():
        raise OracleAuditError(f"invalid prediction corners: {path}")
    if scores.shape != (len(rows),) or not np.isfinite(scores).all():
        raise OracleAuditError(f"invalid prediction scores: {path}")
    world = np.concatenate((corners.min(axis=1), corners.max(axis=1)), axis=1)
    aligned_corners = corners @ alignment[:3, :3].T + alignment[:3, 3]
    aligned = np.concatenate(
        (aligned_corners.min(axis=1), aligned_corners.max(axis=1)), axis=1
    )
    return world, aligned


def aligned_iou_matrix(boxes: np.ndarray, gt: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1:] != (6,):
        raise OracleAuditError(f"boxes must have shape Nx6, got {boxes.shape}")
    if gt.ndim != 2 or gt.shape[1:] != (6,):
        raise OracleAuditError(f"GT must have shape Gx6, got {gt.shape}")
    if len(boxes) == 0 or len(gt) == 0:
        return np.zeros((len(boxes), len(gt)), dtype=np.float64)
    lower = np.maximum(boxes[:, None, :3], gt[None, :, :3])
    upper = np.minimum(boxes[:, None, 3:], gt[None, :, 3:])
    intersection = np.prod(np.maximum(upper - lower, 0.0), axis=2)
    box_volume = np.prod(np.maximum(boxes[:, 3:] - boxes[:, :3], 0.0), axis=1)
    gt_volume = np.prod(np.maximum(gt[:, 3:] - gt[:, :3], 0.0), axis=1)
    union = box_volume[:, None] + gt_volume[None, :] - intersection
    return intersection / np.maximum(union, np.finfo(np.float64).eps)


def _load_pose(path: Path, label: str) -> np.ndarray:
    _regular_file(path, label)
    try:
        pose = np.loadtxt(path, dtype=np.float64)
    except (OSError, ValueError) as error:
        raise OracleAuditError(f"could not load {label}: {path}") from error
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise OracleAuditError(f"invalid {label}: {path}")
    if np.max(np.abs(pose[3] - np.asarray([0.0, 0.0, 0.0, 1.0]))) > 1e-5:
        raise OracleAuditError(f"invalid homogeneous row in {label}: {path}")
    return pose


def boxer_frame_schedule(scene_dir: Path) -> tuple[list[int], np.ndarray, Path]:
    """Mirror ScanNetLoader's start=1, skip=25 and invalid-pose filtering."""

    color_dir = scene_dir / "frames" / "color"
    pose_dir = scene_dir / "frames" / "pose"
    if not color_dir.is_dir() or not pose_dir.is_dir():
        raise OracleAuditError(f"missing ScanNet RGB/pose directories: {scene_dir}")
    ids: dict[int, Path] = {}
    for path in color_dir.iterdir():
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        try:
            frame_id = int(path.stem)
        except ValueError as error:
            raise OracleAuditError(f"non-numeric ScanNet color frame: {path}") from error
        if frame_id in ids:
            raise OracleAuditError(
                f"duplicate color frame ID {frame_id}: {ids[frame_id]} and {path}"
            )
        ids[frame_id] = path
    if not ids:
        raise OracleAuditError(f"no ScanNet color frames: {color_dir}")
    scheduled = sorted(ids)[BOXER_START_N - 1 :: BOXER_SKIP_N]
    valid: list[int] = []
    first_pose: np.ndarray | None = None
    first_pose_path: Path | None = None
    for frame_id in scheduled:
        path = pose_dir / f"{frame_id}.txt"
        if not path.is_file():
            continue
        try:
            pose = np.loadtxt(path, dtype=np.float64)
        except (OSError, ValueError):
            continue
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            continue
        valid.append(frame_id)
        if first_pose is None:
            first_pose = pose
            first_pose_path = path
    if first_pose is None or first_pose_path is None:
        raise OracleAuditError(f"Boxer schedule has no valid pose: {scene_dir}")
    return valid, first_pose[:3, 3].copy(), first_pose_path


def _quaternion_rotation(q: np.ndarray, path: Path, row_number: int) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 1e-12 or abs(norm - 1.0) > 5e-3:
        raise OracleAuditError(
            f"invalid quaternion norm at {path}:{row_number}: {norm}"
        )
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_boxer_csv(
    path: Path,
    *,
    scene_id: str,
    source: str,
    world_offset: np.ndarray,
    alignment: np.ndarray,
    allowed_raw_frames: set[int] | None = None,
) -> list[Candidate]:
    _regular_file(path, f"Boxer {source} CSV")
    candidates: list[Candidate] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise OracleAuditError(
                f"unexpected Boxer CSV schema in {path}: {reader.fieldnames}"
            )
        for row_index, row in enumerate(reader):
            row_number = row_index + 2
            try:
                frame_id = int(row["time_ns"])
                instance_id = int(row["instance"])
                int(row["sem_id"])
                probability = float(row["prob"])
                center_recentred = np.asarray(
                    [
                        float(row["tx_world_object"]),
                        float(row["ty_world_object"]),
                        float(row["tz_world_object"]),
                    ],
                    dtype=np.float64,
                )
                quaternion = np.asarray(
                    [
                        float(row["qw_world_object"]),
                        float(row["qx_world_object"]),
                        float(row["qy_world_object"]),
                        float(row["qz_world_object"]),
                    ],
                    dtype=np.float64,
                )
                size = np.asarray(
                    [float(row["scale_x"]), float(row["scale_y"]), float(row["scale_z"])],
                    dtype=np.float64,
                )
            except (TypeError, ValueError) as error:
                raise OracleAuditError(f"malformed Boxer row at {path}:{row_number}") from error
            values = np.concatenate((center_recentred, quaternion, size, [probability]))
            if not np.isfinite(values).all() or np.any(size <= 0.0):
                raise OracleAuditError(f"non-finite/non-positive Boxer row at {path}:{row_number}")
            if not 0.0 <= probability <= 1.0:
                raise OracleAuditError(f"Boxer probability outside [0,1] at {path}:{row_number}")
            if source == "raw":
                if allowed_raw_frames is not None and frame_id not in allowed_raw_frames:
                    raise OracleAuditError(
                        f"raw Boxer frame {frame_id} violates frozen schedule at {path}:{row_number}"
                    )
            elif source == "tracked":
                if frame_id != 0:
                    raise OracleAuditError(
                        f"terminal tracked Boxer row must have time_ns=0 at {path}:{row_number}"
                    )
            else:
                raise OracleAuditError(f"unknown Boxer source: {source}")
            rotation = _quaternion_rotation(quaternion, path, row_number)
            center_world = center_recentred + world_offset
            local_corners = _SIGNS * (size / 2.0)
            corners_world = local_corners @ rotation.T + center_world
            aligned_corners = corners_world @ alignment[:3, :3].T + alignment[:3, 3]
            aligned_minmax = np.concatenate(
                (aligned_corners.min(axis=0), aligned_corners.max(axis=0))
            )
            candidates.append(
                Candidate(
                    scene_id=scene_id,
                    source=source,
                    row_index=row_index,
                    frame_id=frame_id,
                    instance_id=instance_id,
                    name=row["name"],
                    probability=probability,
                    center_world=center_world,
                    rotation_world_object=rotation,
                    size=size,
                    corners_world=corners_world,
                    aligned_minmax=aligned_minmax,
                )
            )
    return candidates


def _load_intrinsics(path: Path) -> np.ndarray:
    _regular_file(path, "depth intrinsics")
    try:
        matrix = np.loadtxt(path, dtype=np.float64)
    except (OSError, ValueError) as error:
        raise OracleAuditError(f"could not load depth intrinsics: {path}") from error
    if matrix.shape not in {(3, 3), (4, 4)} or not np.isfinite(matrix).all():
        raise OracleAuditError(f"invalid depth intrinsics: {path}")
    intrinsics = matrix[:3, :3]
    height_guard = float(intrinsics[2, 2])
    if (
        intrinsics[0, 0] <= 0.0
        or intrinsics[1, 1] <= 0.0
        or abs(height_guard - 1.0) > 1e-5
    ):
        raise OracleAuditError(f"invalid pinhole intrinsics: {path}")
    return intrinsics


def load_stride4_world_points(
    depth_path: Path, pose_path: Path, intrinsics: np.ndarray
) -> np.ndarray:
    _regular_file(depth_path, "ScanNet depth")
    pose = _load_pose(pose_path, "ScanNet camera pose")
    try:
        with Image.open(depth_path) as image:
            depth_raw = np.asarray(image)
    except (OSError, ValueError) as error:
        raise OracleAuditError(f"could not load ScanNet depth: {depth_path}") from error
    if depth_raw.ndim != 2 or not np.issubdtype(depth_raw.dtype, np.integer):
        raise OracleAuditError(f"ScanNet depth must be a 2D integer millimetre image: {depth_path}")
    depth_m = depth_raw.astype(np.float64) / 1000.0
    rows = np.arange(0, depth_m.shape[0], PIXEL_STRIDE, dtype=np.int64)
    cols = np.arange(0, depth_m.shape[1], PIXEL_STRIDE, dtype=np.int64)
    grid_rows, grid_cols = np.meshgrid(rows, cols, indexing="ij")
    sampled = depth_m[grid_rows, grid_cols]
    valid = (
        np.isfinite(sampled)
        & (sampled >= MIN_DEPTH_M)
        & (sampled <= MAX_DEPTH_M)
    )
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64)
    v = grid_rows[valid].astype(np.float64)
    u = grid_cols[valid].astype(np.float64)
    z = sampled[valid]
    camera = np.column_stack(
        (
            (u - intrinsics[0, 2]) / intrinsics[0, 0] * z,
            (v - intrinsics[1, 2]) / intrinsics[1, 1] * z,
            z,
        )
    )
    world = camera @ pose[:3, :3].T + pose[:3, 3]
    if not np.isfinite(world).all():
        raise OracleAuditError(f"non-finite world points from depth: {depth_path}")
    return world


def signed_voxel_centroids(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return lexicographic signed-floor keys and per-key point centroids."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise OracleAuditError(f"voxel points must be finite Nx3, got {points.shape}")
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.int64), np.empty((0, 3), dtype=np.float64)
    scaled = points / VOXEL_SIZE_M
    if np.max(np.abs(scaled), initial=0.0) > np.iinfo(np.int64).max / 4:
        raise OracleAuditError("voxel coordinate overflow")
    raw_keys = np.floor(scaled).astype(np.int64)
    keys, inverse, counts = np.unique(
        raw_keys, axis=0, return_inverse=True, return_counts=True
    )
    centroids = np.empty((len(keys), 3), dtype=np.float64)
    for axis in range(3):
        centroids[:, axis] = np.bincount(
            inverse, weights=points[:, axis], minlength=len(keys)
        ) / counts
    return keys, centroids


def gate_candidates_for_frame(
    candidates: Sequence[Candidate],
    world_points: np.ndarray,
    native_world_aabbs: np.ndarray,
) -> tuple[list[Candidate], list[GateRow]]:
    """Apply the frozen OBB-support and unexplained-voxel gate."""

    world_points = np.asarray(world_points, dtype=np.float64)
    native = np.asarray(native_world_aabbs, dtype=np.float64)
    if world_points.ndim != 2 or world_points.shape[1:] != (3,):
        raise OracleAuditError("world_points must have shape Nx3")
    if native.ndim != 2 or native.shape[1:] != (6,):
        raise OracleAuditError("native_world_aabbs must have shape Nx6")
    expanded = native.copy()
    if len(expanded):
        expanded[:, :3] -= NATIVE_AABB_EXPANSION_M
        expanded[:, 3:] += NATIVE_AABB_EXPANSION_M
    accepted: list[Candidate] = []
    rows: list[GateRow] = []
    for candidate in candidates:
        local = (world_points - candidate.center_world) @ candidate.rotation_world_object
        support_mask = np.all(
            np.abs(local) <= candidate.size[None, :] / 2.0 + 1e-9, axis=1
        )
        _, centroids = signed_voxel_centroids(world_points[support_mask])
        support_count = len(centroids)
        if len(expanded) and support_count:
            explained = np.any(
                np.all(
                    (centroids[:, None, :] >= expanded[None, :, :3])
                    & (centroids[:, None, :] <= expanded[None, :, 3:]),
                    axis=2,
                ),
                axis=1,
            )
            unexplained_count = int(np.count_nonzero(~explained))
        else:
            unexplained_count = support_count
        ratio = float(unexplained_count / support_count) if support_count else 0.0
        if support_count < MIN_CANDIDATE_VOXELS:
            reason = "insufficient_candidate_voxels"
            keep = False
        elif unexplained_count < MIN_UNEXPLAINED_VOXELS:
            reason = "insufficient_unexplained_voxels"
            keep = False
        elif ratio < MIN_UNEXPLAINED_RATIO:
            reason = "unexplained_ratio_below_0.50"
            keep = False
        else:
            reason = "accepted"
            keep = True
            accepted.append(candidate)
        rows.append(
            GateRow(
                candidate_id=candidate.candidate_id,
                frame_id=candidate.frame_id,
                candidate_voxels=support_count,
                unexplained_voxels=unexplained_count,
                unexplained_ratio=ratio,
                accepted=keep,
                reason=reason,
            )
        )
    return accepted, rows


def _voc_ap(tp: np.ndarray, fp: np.ndarray, npos: int) -> tuple[float, float, float]:
    tp_cumulative = np.cumsum(tp, dtype=np.float64)
    fp_cumulative = np.cumsum(fp, dtype=np.float64)
    recall = tp_cumulative / float(npos + 1e-6)
    precision = tp_cumulative / np.maximum(
        tp_cumulative + fp_cumulative, np.finfo(np.float64).eps
    )
    padded_recall = np.concatenate(([0.0], recall, [1.0]))
    padded_precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(padded_precision.size - 1, 0, -1):
        padded_precision[index - 1] = max(
            padded_precision[index - 1], padded_precision[index]
        )
    changes = np.where(padded_recall[1:] != padded_recall[:-1])[0]
    ap = float(
        np.sum(
            (padded_recall[changes + 1] - padded_recall[changes])
            * padded_precision[changes + 1]
        )
    )
    final_recall = float(recall[-1]) if len(recall) else 0.0
    final_precision = float(precision[-1]) if len(precision) else 0.0
    return ap, final_recall, final_precision


def official_constant_evaluate(
    iou_by_scene: Sequence[np.ndarray], gt_counts: Sequence[int], threshold: float
) -> dict[str, object]:
    """Exact class-agnostic constant-score evaluator, including tie order."""

    if len(iou_by_scene) != len(gt_counts):
        raise OracleAuditError("IoU scene count does not match GT counts")
    lengths = np.asarray([len(matrix) for matrix in iou_by_scene], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    total_predictions = int(offsets[-1])
    confidence = np.ones(total_predictions, dtype=np.float64)
    # Deliberately omit ``kind=``: this mirrors evaluation/utils/eval_det.py.
    order = np.argsort(-confidence)
    scene_indices = np.searchsorted(offsets[1:], order, side="right")
    local_indices = order - offsets[scene_indices]
    matched = [np.zeros(int(count), dtype=bool) for count in gt_counts]
    tp = np.zeros(total_predictions, dtype=np.float64)
    for detection_index, (scene_index, local_index) in enumerate(
        zip(scene_indices, local_indices)
    ):
        overlaps = iou_by_scene[int(scene_index)][int(local_index)]
        if overlaps.size:
            gt_index = int(np.argmax(overlaps))
            if overlaps[gt_index] > threshold and not matched[int(scene_index)][gt_index]:
                matched[int(scene_index)][gt_index] = True
                tp[detection_index] = 1.0
    fp = 1.0 - tp
    npos = int(sum(gt_counts))
    ap, recall, precision = _voc_ap(tp, fp, npos)
    return {
        "ap": ap,
        "ap_points": 100.0 * ap,
        "recall": recall,
        "precision": precision,
        "greedy_tp": int(tp.sum()),
        "false_positive": int(fp.sum()),
        "unmatched_gt_count": int(npos - tp.sum()),
        "matched_gt_masks": matched,
        "evaluation_order": order,
    }


def strict_maximum_matching(
    iou: np.ndarray, threshold: float, gt_mask: np.ndarray | None = None
) -> list[tuple[int, int]]:
    """Deterministic maximum-cardinality matching on the strict IoU graph."""

    matrix = np.asarray(iou, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise OracleAuditError("IoU matrix must be finite and two-dimensional")
    if gt_mask is None:
        enabled = np.ones(matrix.shape[1], dtype=bool)
    else:
        enabled = np.asarray(gt_mask, dtype=bool)
        if enabled.shape != (matrix.shape[1],):
            raise OracleAuditError("GT mask does not match IoU matrix")
    adjacency: list[list[int]] = []
    for candidate_index in range(matrix.shape[0]):
        gt_indices = np.flatnonzero(
            enabled & (matrix[candidate_index] > threshold)
        ).tolist()
        gt_indices.sort(key=lambda index: (-matrix[candidate_index, index], index))
        adjacency.append(gt_indices)
    matched_candidate = np.full(matrix.shape[1], -1, dtype=np.int64)

    def augment(candidate_index: int, seen: np.ndarray) -> bool:
        for gt_index in adjacency[candidate_index]:
            if seen[gt_index]:
                continue
            seen[gt_index] = True
            previous = int(matched_candidate[gt_index])
            if previous < 0 or augment(previous, seen):
                matched_candidate[gt_index] = candidate_index
                return True
        return False

    for candidate_index in range(matrix.shape[0]):
        augment(candidate_index, np.zeros(matrix.shape[1], dtype=bool))
    pairs = [
        (int(candidate_index), int(gt_index))
        for gt_index, candidate_index in enumerate(matched_candidate)
        if candidate_index >= 0
    ]
    return sorted(pairs)


def _pool_threshold_report(
    matrices: Sequence[np.ndarray],
    baseline_matrices: Sequence[np.ndarray],
    scenes: Sequence[str],
    baseline_matched: Sequence[np.ndarray],
    threshold: float,
    baseline_ap: float,
) -> dict[str, object]:
    coverage_total = 0
    matching_total = 0
    unmatched_total = 0
    recovered_coverage_total = 0
    recovered_matching_total = 0
    native_matching_total = 0
    union_matching_total = 0
    per_scene: dict[str, object] = {}
    for scene, matrix, baseline_matrix, matched in zip(
        scenes, matrices, baseline_matrices, baseline_matched
    ):
        coverage_mask = (
            np.any(matrix > threshold, axis=0)
            if len(matrix)
            else np.zeros(matrix.shape[1], dtype=bool)
        )
        pairs = strict_maximum_matching(matrix, threshold)
        unmatched = ~np.asarray(matched, dtype=bool)
        recovered_mask = coverage_mask & unmatched
        recovery_pairs = strict_maximum_matching(matrix, threshold, unmatched)
        native_pairs = strict_maximum_matching(baseline_matrix, threshold)
        union_matrix = np.concatenate((baseline_matrix, matrix), axis=0)
        union_pairs = strict_maximum_matching(union_matrix, threshold)
        coverage_total += int(coverage_mask.sum())
        matching_total += len(pairs)
        unmatched_total += int(unmatched.sum())
        recovered_coverage_total += int(recovered_mask.sum())
        recovered_matching_total += len(recovery_pairs)
        native_matching_total += len(native_pairs)
        union_matching_total += len(union_pairs)
        per_scene[scene] = {
            "candidate_count": int(len(matrix)),
            "gt_count": int(matrix.shape[1]),
            "coverage_gt_count": int(coverage_mask.sum()),
            "covered_gt_indices": np.flatnonzero(coverage_mask).tolist(),
            "maximum_matching_count": len(pairs),
            "maximum_matching_pairs": [list(pair) for pair in pairs],
            "baseline_unmatched_gt_count": int(unmatched.sum()),
            "baseline_unmatched_recovered_count": int(recovered_mask.sum()),
            "baseline_unmatched_recovered_indices": np.flatnonzero(recovered_mask).tolist(),
            "baseline_unmatched_maximum_matching_count": len(recovery_pairs),
            "native_maximum_matching_count": len(native_pairs),
            "native_union_pool_maximum_matching_count": len(union_pairs),
            "union_recoverable_gt_count_over_native_maximum_matching": (
                len(union_pairs) - len(native_pairs)
            ),
        }
    gt_total = int(sum(matrix.shape[1] for matrix in matrices))
    official_recall_denominator = float(gt_total + 1e-6)
    exact_recall_denominator = float(gt_total) if gt_total else 1.0
    union_recall_ceiling = union_matching_total / exact_recall_denominator
    native_matching_recall = native_matching_total / exact_recall_denominator
    ideal_tp_first_ap_upper_bound = (
        union_matching_total / official_recall_denominator if gt_total else 0.0
    )
    ideal_delta_points = 100.0 * (ideal_tp_first_ap_upper_bound - baseline_ap)
    incremental_recall_headroom_points = 100.0 * (
        union_matching_total - native_matching_total
    ) / exact_recall_denominator
    return {
        "coverage_gt_count": coverage_total,
        "coverage_ratio": float(coverage_total / gt_total) if gt_total else 0.0,
        "maximum_matching_count": matching_total,
        "maximum_matching_ratio": float(matching_total / gt_total) if gt_total else 0.0,
        "baseline_unmatched_gt_count": unmatched_total,
        "baseline_unmatched_recovered_count": recovered_coverage_total,
        "baseline_unmatched_recovery_ratio": (
            float(recovered_coverage_total / unmatched_total) if unmatched_total else 0.0
        ),
        "baseline_unmatched_maximum_matching_count": recovered_matching_total,
        "native_maximum_matching_count": native_matching_total,
        "native_maximum_matching_recall": native_matching_recall,
        "native_union_pool_maximum_matching_count": union_matching_total,
        "union_recoverable_gt_count_over_native_maximum_matching": (
            union_matching_total - native_matching_total
        ),
        "incremental_recall_headroom_points": incremental_recall_headroom_points,
        "passes_plus10_via_recovered_misses_only": (
            incremental_recall_headroom_points >= 10.0
        ),
        "union_recall_ceiling": union_recall_ceiling,
        "union_recall_ceiling_points": 100.0 * union_recall_ceiling,
        "ideal_tp_first_ap_upper_bound": ideal_tp_first_ap_upper_bound,
        "ideal_tp_first_ap_upper_bound_points": 100.0 * ideal_tp_first_ap_upper_bound,
        "ideal_tp_first_max_delta_ap_points_over_official_baseline": ideal_delta_points,
        "necessary_headroom_for_plus10": {
            "target_delta_ap_points": 10.0,
            "baseline_official_ap_points": 100.0 * baseline_ap,
            "ideal_tp_first_ap_upper_bound_points": 100.0 * ideal_tp_first_ap_upper_bound,
            "maximum_possible_delta_ap_points": ideal_delta_points,
            "incremental_recall_headroom_points": incremental_recall_headroom_points,
            "passes_ideal_tp_first_ap_upper_bound": ideal_delta_points >= 10.0,
            "passes_plus10_via_recovered_misses_only": (
                incremental_recall_headroom_points >= 10.0
            ),
            "passes": incremental_recall_headroom_points >= 10.0,
            "necessary_not_sufficient": True,
        },
        "per_scene": per_scene,
    }


def _json_baseline_evaluation(
    evaluation: Mapping[str, object], scenes: Sequence[str]
) -> dict[str, object]:
    masks = evaluation["matched_gt_masks"]
    assert isinstance(masks, list)
    return {
        key: value
        for key, value in evaluation.items()
        if key not in {"matched_gt_masks", "evaluation_order"}
    } | {
        "per_scene": {
            scene: {
                "greedy_tp": int(np.count_nonzero(mask)),
                "unmatched_gt_count": int(len(mask) - np.count_nonzero(mask)),
                "matched_gt_indices": np.flatnonzero(mask).tolist(),
                "unmatched_gt_indices": np.flatnonzero(~mask).tolist(),
            }
            for scene, mask in zip(scenes, masks)
        }
    }


def _gt_selected_suffix_report(
    *,
    scenes: Sequence[str],
    baseline_matrices: Sequence[np.ndarray],
    pool_matrices: Sequence[np.ndarray],
    pool_candidates: Sequence[Sequence[Candidate]],
    gt_counts: Sequence[int],
    baseline_evaluation: Mapping[str, object],
    threshold: float,
) -> dict[str, object]:
    baseline_masks = baseline_evaluation["matched_gt_masks"]
    assert isinstance(baseline_masks, list)
    selected_matrices: list[np.ndarray] = []
    selected_rows: dict[str, list[dict[str, object]]] = {}
    intended_count = 0
    for scene, matrix, candidates, matched in zip(
        scenes, pool_matrices, pool_candidates, baseline_masks
    ):
        pairs = strict_maximum_matching(matrix, threshold, ~matched)
        candidate_indices = sorted(candidate_index for candidate_index, _ in pairs)
        target_by_candidate = {candidate_index: gt_index for candidate_index, gt_index in pairs}
        selected_matrices.append(
            matrix[candidate_indices]
            if candidate_indices
            else np.empty((0, matrix.shape[1]), dtype=np.float64)
        )
        selected_rows[scene] = [
            {
                "candidate_id": candidates[index].candidate_id,
                "candidate_index": index,
                "target_gt_index": target_by_candidate[index],
                "target_iou": float(matrix[index, target_by_candidate[index]]),
            }
            for index in candidate_indices
        ]
        intended_count += len(pairs)
    combined = [
        np.concatenate((baseline, suffix), axis=0)
        for baseline, suffix in zip(baseline_matrices, selected_matrices)
    ]
    evaluated = official_constant_evaluate(combined, gt_counts, threshold)
    baseline_ap = float(baseline_evaluation["ap"])
    baseline_tp = int(baseline_evaluation["greedy_tp"])
    return {
        "oracle_only": True,
        "deployable": False,
        "gt_used": True,
        "constructive_counterfactual": True,
        "mathematical_upper_bound": False,
        "selection": "strict_maximum_matching_against_official_baseline_unmatched_gt",
        "candidate_suffix_order": "source_csv_row_order_within_each_scene",
        "native_rows_are_on_disk_prefix": True,
        "official_tie_order": "numpy.argsort_default_all_scores_1.0",
        "selected_candidate_count": int(sum(len(rows) for rows in selected_rows.values())),
        "intended_recovery_matching_count": intended_count,
        "ap": float(evaluated["ap"]),
        "ap_points": float(evaluated["ap_points"]),
        "delta_ap_points": 100.0 * (float(evaluated["ap"]) - baseline_ap),
        "greedy_tp": int(evaluated["greedy_tp"]),
        "delta_greedy_tp": int(evaluated["greedy_tp"]) - baseline_tp,
        "false_positive": int(evaluated["false_positive"]),
        "recall": float(evaluated["recall"]),
        "precision": float(evaluated["precision"]),
        "per_scene_selection": selected_rows,
    }


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _validate_output_path(output: Path, protected_roots: Iterable[Path]) -> None:
    if output.suffix.lower() != ".json":
        raise OracleAuditError("oracle output must have a .json suffix")
    if output.exists() or output.is_symlink():
        raise OracleAuditError(f"refusing to overwrite oracle output: {output}")
    for root in protected_roots:
        if _path_is_within(output, root):
            raise OracleAuditError(
                f"oracle output must not be inside protected input root {root}: {output}"
            )


def validate_shadow_seal(
    *,
    json_path: Path,
    npz_path: Path,
    scenes: Sequence[str],
    raw_candidates: Sequence[Sequence[Candidate]],
    tracked_candidates: Sequence[Sequence[Candidate]],
    boxer_root: Path,
    schedule_root: Path,
) -> dict[str, object]:
    """Verify that direct CSV geometry is identical to the sealed sidecar."""

    _regular_file(json_path, "Boxer shadow seal JSON")
    _regular_file(npz_path, "Boxer shadow seal NPZ")
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OracleAuditError(f"invalid Boxer shadow seal JSON: {json_path}") from error
    if not isinstance(payload, dict):
        raise OracleAuditError("Boxer shadow seal JSON must be an object")
    required_contract = {
        "schema": "boxfusion.owl_boxer_shadow_candidates.v1",
        "mode": "shadow",
        "coordinate_frame": "scannet_world",
        "gt_access": False,
        "gt_access_guard": (
            "BOXFUSION_SHADOW_GT_ACCESS=forbidden annotation_path=None"
        ),
        "gt_access_guard_verified": True,
        "output_inert": True,
        "birth": False,
        "native_before_after_identity": True,
        "native_clip_unchanged": True,
        "semantic_source_exported": False,
    }
    for key, expected in required_contract.items():
        if payload.get(key) != expected:
            raise OracleAuditError(
                f"Boxer shadow seal contract mismatch for {key}: {payload.get(key)!r}"
            )
    if payload.get("npz_file") != npz_path.name:
        raise OracleAuditError("Boxer shadow seal NPZ filename mismatch")
    npz_sha = _sha256(npz_path)
    if payload.get("npz_sha256") != npz_sha:
        raise OracleAuditError("Boxer shadow seal NPZ SHA-256 mismatch")
    if payload.get("scene_count") != len(scenes):
        raise OracleAuditError("Boxer shadow seal scene count mismatch")
    if payload.get("per_view_candidate_count") != sum(map(len, raw_candidates)):
        raise OracleAuditError("Boxer shadow seal per-view count mismatch")
    if payload.get("tracked_candidate_count") != sum(map(len, tracked_candidates)):
        raise OracleAuditError("Boxer shadow seal tracked count mismatch")
    assets = payload.get("assets_and_protocol")
    if not isinstance(assets, dict):
        raise OracleAuditError("Boxer shadow seal asset/protocol ledger missing")
    required_assets = {
        "profile": "clean_in2",
        "detector": "owl",
        "taxonomy": "lvisplus",
        "taxonomy_count": 1220,
        "start_n": BOXER_START_N,
        "skip_n": BOXER_SKIP_N,
        "threshold_2d": 0.25,
        "threshold_3d": 0.5,
        "nms_iou_2d": 0.5,
    }
    for key, expected in required_assets.items():
        if assets.get(key) != expected:
            raise OracleAuditError(
                f"Boxer shadow seal asset/protocol mismatch for {key}: "
                f"{assets.get(key)!r}"
            )
    asset_hash_keys = (
        "boxer_checkpoint_sha256",
        "boxernet_source_sha256",
        "dinov3_checkpoint_sha256",
        "owl_checkpoint_sha256",
        "owl_text_cache_sha256",
        "owl_wrapper_sha256",
        "run_boxer_sha256",
        "taxonomy_sha256",
    )
    for key in asset_hash_keys:
        value = assets.get(key)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise OracleAuditError(f"invalid Boxer shadow seal asset hash: {key}")
    scene_rows = payload.get("scenes")
    if not isinstance(scene_rows, list) or len(scene_rows) != len(scenes):
        raise OracleAuditError("Boxer shadow seal scene ledger mismatch")
    for scene_index, (scene, row) in enumerate(zip(scenes, scene_rows)):
        if not isinstance(row, dict) or row.get("scene_id") != scene:
            raise OracleAuditError("Boxer shadow seal scene order mismatch")
        if row.get("scene_index") != scene_index:
            raise OracleAuditError("Boxer shadow seal scene index mismatch")
        if row.get("tracked_schedule_clean") is not True:
            raise OracleAuditError(f"sealed tracked pool is schedule-contaminated: {scene}")
        if row.get("gt_access_guard_verified") is not True:
            raise OracleAuditError(f"sealed no-GT guard is unverified: {scene}")
        if row.get("per_view_extra_schedule_rows_excluded") != 0:
            raise OracleAuditError(f"sealed per-view pool contains extra schedule rows: {scene}")
        inputs = row.get("inputs")
        if not isinstance(inputs, dict):
            raise OracleAuditError(f"sealed scene input ledger missing: {scene}")
        raw_path = boxer_root / scene / "boxer_3dbbs.csv"
        tracked_path = boxer_root / scene / "boxer_3dbbs_tracked.csv"
        manifest_path = schedule_root / scene / "manifest.json"
        if inputs.get("boxer_3dbbs_csv_sha256") != _sha256(raw_path):
            raise OracleAuditError(f"sealed raw CSV hash mismatch: {scene}")
        if inputs.get("boxer_3dbbs_tracked_csv_sha256") != _sha256(tracked_path):
            raise OracleAuditError(f"sealed tracked CSV hash mismatch: {scene}")
        if row.get("sealed_schedule_manifest_sha256") != _sha256(manifest_path):
            raise OracleAuditError(f"sealed schedule manifest hash mismatch: {scene}")
        if row.get("per_view_kept_rows") != len(raw_candidates[scene_index]):
            raise OracleAuditError(f"sealed per-view scene count mismatch: {scene}")
        if row.get("tracked_kept_rows") != len(tracked_candidates[scene_index]):
            raise OracleAuditError(f"sealed tracked scene count mismatch: {scene}")
        offset = np.asarray(row.get("world_offset_xyz"), dtype=np.float64)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise OracleAuditError(f"invalid sealed world offset: {scene}")

    expected_arrays = {
        "per_view_center_world",
        "per_view_extent_xyz",
        "per_view_frame_id",
        "per_view_quaternion_wxyz",
        "per_view_scene_index",
        "per_view_source_instance_id",
        "per_view_source_row",
        "per_view_source_score",
        "scene_ids",
        "tracked_center_world",
        "tracked_extent_xyz",
        "tracked_instance_id",
        "tracked_quaternion_wxyz",
        "tracked_scene_index",
        "tracked_source_row",
        "tracked_source_score",
    }
    try:
        with np.load(npz_path, allow_pickle=False) as arrays:
            if set(arrays.files) != expected_arrays:
                raise OracleAuditError("unexpected Boxer shadow seal NPZ schema")
            if arrays["scene_ids"].tolist() != list(scenes):
                raise OracleAuditError("Boxer shadow seal NPZ scene order mismatch")
            for scene_index, candidates in enumerate(raw_candidates):
                positions = np.flatnonzero(
                    arrays["per_view_scene_index"] == scene_index
                )
                if len(positions) != len(candidates):
                    raise OracleAuditError("sealed NPZ per-view scene count mismatch")
                candidates_by_row = {candidate.row_index: candidate for candidate in candidates}
                observed_rows = arrays["per_view_source_row"][positions]
                if (
                    len(candidates_by_row) != len(candidates)
                    or len(np.unique(observed_rows)) != len(observed_rows)
                    or set(observed_rows.tolist()) != set(candidates_by_row)
                ):
                    raise OracleAuditError("sealed NPZ per-view source row mismatch")
                ordered_candidates = [
                    candidates_by_row[int(row_index)] for row_index in observed_rows
                ]
                expected_frames = np.asarray(
                    [candidate.frame_id for candidate in ordered_candidates], dtype=np.int64
                )
                if not np.array_equal(arrays["per_view_frame_id"][positions], expected_frames):
                    raise OracleAuditError("sealed NPZ per-view frame mismatch")
                expected_instances = np.asarray(
                    [candidate.instance_id for candidate in ordered_candidates], dtype=np.int64
                )
                if not np.array_equal(
                    arrays["per_view_source_instance_id"][positions], expected_instances
                ):
                    raise OracleAuditError("sealed NPZ per-view instance mismatch")
                _validate_sealed_candidate_arrays(
                    npz_path=npz_path,
                    positions=positions,
                    candidates=ordered_candidates,
                    center=arrays["per_view_center_world"],
                    extent=arrays["per_view_extent_xyz"],
                    quaternion=arrays["per_view_quaternion_wxyz"],
                    score=arrays["per_view_source_score"],
                    label="per-view",
                )
            for scene_index, candidates in enumerate(tracked_candidates):
                positions = np.flatnonzero(arrays["tracked_scene_index"] == scene_index)
                if len(positions) != len(candidates):
                    raise OracleAuditError("sealed NPZ tracked scene count mismatch")
                candidates_by_row = {candidate.row_index: candidate for candidate in candidates}
                observed_rows = arrays["tracked_source_row"][positions]
                if (
                    len(candidates_by_row) != len(candidates)
                    or len(np.unique(observed_rows)) != len(observed_rows)
                    or set(observed_rows.tolist()) != set(candidates_by_row)
                ):
                    raise OracleAuditError("sealed NPZ tracked source row mismatch")
                ordered_candidates = [
                    candidates_by_row[int(row_index)] for row_index in observed_rows
                ]
                expected_instances = np.asarray(
                    [candidate.instance_id for candidate in ordered_candidates], dtype=np.int64
                )
                if not np.array_equal(arrays["tracked_instance_id"][positions], expected_instances):
                    raise OracleAuditError("sealed NPZ tracked instance mismatch")
                _validate_sealed_candidate_arrays(
                    npz_path=npz_path,
                    positions=positions,
                    candidates=ordered_candidates,
                    center=arrays["tracked_center_world"],
                    extent=arrays["tracked_extent_xyz"],
                    quaternion=arrays["tracked_quaternion_wxyz"],
                    score=arrays["tracked_source_score"],
                    label="tracked",
                )
    except (OSError, ValueError) as error:
        if isinstance(error, OracleAuditError):
            raise
        raise OracleAuditError(f"invalid Boxer shadow seal NPZ: {npz_path}") from error
    return {
        "verified": True,
        "json_path": os.fspath(json_path),
        "json_sha256": _sha256(json_path),
        "npz_path": os.fspath(npz_path),
        "npz_sha256": npz_sha,
        "schema": payload["schema"],
        "gt_access_guard": payload["gt_access_guard"],
        "gt_access_guard_verified": payload["gt_access_guard_verified"],
        "asset_sha256": {key: assets[key] for key in asset_hash_keys},
        "candidate_content_sha256": payload.get("candidate_content_sha256"),
        "native_identity_ledger_sha256": payload.get(
            "native_identity_ledger_sha256"
        ),
    }


def _validate_sealed_candidate_arrays(
    *,
    npz_path: Path,
    positions: np.ndarray,
    candidates: Sequence[Candidate],
    center: np.ndarray,
    extent: np.ndarray,
    quaternion: np.ndarray,
    score: np.ndarray,
    label: str,
) -> None:
    expected_center = np.asarray(
        [candidate.center_world for candidate in candidates], dtype=np.float64
    ).reshape((-1, 3))
    expected_extent = np.asarray(
        [candidate.size for candidate in candidates], dtype=np.float64
    ).reshape((-1, 3))
    expected_score = np.asarray(
        [candidate.probability for candidate in candidates], dtype=np.float64
    )
    if not np.allclose(center[positions], expected_center, rtol=0.0, atol=2e-5):
        raise OracleAuditError(f"sealed NPZ {label} center mismatch")
    if not np.allclose(extent[positions], expected_extent, rtol=0.0, atol=2e-5):
        raise OracleAuditError(f"sealed NPZ {label} extent mismatch")
    if not np.allclose(score[positions], expected_score, rtol=0.0, atol=2e-6):
        raise OracleAuditError(f"sealed NPZ {label} score mismatch")
    observed_q = np.asarray(quaternion[positions], dtype=np.float64)
    if observed_q.shape != (len(candidates), 4) or not np.isfinite(observed_q).all():
        raise OracleAuditError(f"sealed NPZ {label} quaternion schema mismatch")
    for index, (candidate, q) in enumerate(zip(candidates, observed_q), 1):
        rotation = _quaternion_rotation(q, npz_path, index)
        if not np.allclose(
            rotation, candidate.rotation_world_object, rtol=0.0, atol=2e-5
        ):
            raise OracleAuditError(f"sealed NPZ {label} rotation mismatch")


def audit_scannet_boxer_unexplained_oracle(
    *,
    boxer_root: Path,
    schedule_root: Path,
    baseline_root: Path,
    scene_list: Path,
    gt_root: Path,
    scan_root: Path,
    scene_rgbd_root: Path,
    shadow_seal_json: Path | None = None,
    shadow_seal_npz: Path | None = None,
) -> dict[str, object]:
    scenes = load_scene_list(scene_list)
    baseline_before = {
        scene: _sha256(
            _regular_file(baseline_root / f"{scene}_boxes.pkl", "T05 prediction")
        )
        for scene in scenes
    }

    gt_by_scene: list[np.ndarray] = []
    gt_counts: list[int] = []
    baseline_world: list[np.ndarray] = []
    baseline_aligned: list[np.ndarray] = []
    baseline_iou: list[np.ndarray] = []
    pool_candidates: dict[str, list[list[Candidate]]] = {
        name: [] for name in POOL_NAMES
    }
    pool_iou: dict[str, list[np.ndarray]] = {name: [] for name in POOL_NAMES}
    scene_reports: dict[str, object] = {}
    input_hashes: dict[str, object] = {
        "scene_list": _sha256(scene_list),
        "baseline_predictions": baseline_before,
        "scenes": {},
    }
    schedule_namespace: str | None = None
    tracked_schedule_clean_all = True

    for scene in scenes:
        metadata_path = scan_root / scene / f"{scene}.txt"
        gt_path = gt_root / f"{scene}_bbox.npy"
        baseline_path = baseline_root / f"{scene}_boxes.pkl"
        raw_csv = boxer_root / scene / "boxer_3dbbs.csv"
        tracked_csv = boxer_root / scene / "boxer_3dbbs_tracked.csv"
        schedule_manifest = schedule_root / scene / "manifest.json"
        scene_dir = scene_rgbd_root / scene
        alignment = load_axis_alignment(metadata_path)
        gt = load_gt_minmax(gt_path)
        world_boxes, aligned_boxes = load_baseline_boxes(baseline_path, alignment)
        expected_schedule, namespace = load_sealed_schedule(schedule_manifest, scene)
        if schedule_namespace is None:
            schedule_namespace = namespace
        elif schedule_namespace != namespace:
            raise OracleAuditError(
                f"sealed schedule namespace changed across scenes: {schedule_namespace} != {namespace}"
            )
        loader_schedule, world_offset, first_pose_path = boxer_frame_schedule(scene_dir)
        raw_unfiltered = load_boxer_csv(
            raw_csv,
            scene_id=scene,
            source="raw",
            world_offset=world_offset,
            alignment=alignment,
        )
        expected_set = set(expected_schedule)
        observed_set = {candidate.frame_id for candidate in raw_unfiltered}
        extra_frame_ids = sorted(observed_set - expected_set)
        zero_candidate_frame_ids = sorted(expected_set - observed_set)
        extra_candidate_count = sum(
            candidate.frame_id not in expected_set for candidate in raw_unfiltered
        )
        tracked_schedule_clean_all = tracked_schedule_clean_all and not extra_frame_ids
        raw = [
            candidate
            for candidate in raw_unfiltered
            if candidate.frame_id in expected_set
        ]
        tracked = load_boxer_csv(
            tracked_csv,
            scene_id=scene,
            source="tracked",
            world_offset=world_offset,
            alignment=alignment,
        )

        intrinsics_path = scene_dir / "frames" / "intrinsic" / "intrinsic_depth.txt"
        intrinsics = _load_intrinsics(intrinsics_path)
        by_frame: dict[int, list[Candidate]] = {}
        for candidate in raw:
            by_frame.setdefault(candidate.frame_id, []).append(candidate)
        gated: list[Candidate] = []
        gate_rows: list[GateRow] = []
        depth_hashes: dict[str, str] = {}
        pose_hashes: dict[str, str] = {}
        sampled_point_counts: dict[str, int] = {}
        for frame_id in sorted(by_frame):
            depth_path = scene_dir / "frames" / "depth" / f"{frame_id}.png"
            pose_path = scene_dir / "frames" / "pose" / f"{frame_id}.txt"
            points = load_stride4_world_points(depth_path, pose_path, intrinsics)
            accepted, rows = gate_candidates_for_frame(
                by_frame[frame_id], points, world_boxes
            )
            gated.extend(accepted)
            gate_rows.extend(rows)
            depth_hashes[str(frame_id)] = _sha256(depth_path)
            pose_hashes[str(frame_id)] = _sha256(pose_path)
            sampled_point_counts[str(frame_id)] = int(len(points))

        candidates_for_scene = {
            "raw": raw,
            "depth_gated": gated,
            "tracked": tracked,
        }
        gt_by_scene.append(gt)
        gt_counts.append(len(gt))
        baseline_world.append(world_boxes)
        baseline_aligned.append(aligned_boxes)
        baseline_matrix = aligned_iou_matrix(aligned_boxes, gt)
        baseline_iou.append(baseline_matrix)
        for pool_name in POOL_NAMES:
            candidates = candidates_for_scene[pool_name]
            boxes = (
                np.stack([candidate.aligned_minmax for candidate in candidates])
                if candidates
                else np.empty((0, 6), dtype=np.float64)
            )
            pool_candidates[pool_name].append(candidates)
            pool_iou[pool_name].append(aligned_iou_matrix(boxes, gt))

        reason_counts: dict[str, int] = {}
        for row in gate_rows:
            reason_counts[row.reason] = reason_counts.get(row.reason, 0) + 1
        scene_reports[scene] = {
            "gt_count": len(gt),
            "baseline_prediction_count": len(aligned_boxes),
            "world_offset_restored": world_offset.tolist(),
            "sealed_schedule_namespace": namespace,
            "sealed_schedule_frame_count": len(expected_schedule),
            "sealed_schedule_frame_ids": expected_schedule,
            "boxer_loader_unbounded_valid_frame_count": len(loader_schedule),
            "raw_observed_frame_count_before_schedule_filter": len(observed_set),
            "raw_nonempty_scheduled_frame_count": len(observed_set & expected_set),
            "raw_missing_frame_ids_treated_as_zero_candidates": zero_candidate_frame_ids,
            "raw_zero_candidate_frame_ids": zero_candidate_frame_ids,
            "raw_extra_frame_ids_excluded": extra_frame_ids,
            "raw_extra_candidate_count_excluded": extra_candidate_count,
            "raw_candidate_count_before_schedule_filter": len(raw_unfiltered),
            "raw_candidate_count": len(raw),
            "depth_gated_candidate_count": len(gated),
            "tracked_candidate_count": len(tracked),
            "tracked_pool_schedule_clean": not extra_frame_ids,
            "depth_processed_frame_count": len(by_frame),
            "stride4_valid_point_counts": sampled_point_counts,
            "depth_gate_reason_counts": reason_counts,
            "depth_gate_rows": [
                {
                    "candidate_id": row.candidate_id,
                    "frame_id": row.frame_id,
                    "candidate_voxels": row.candidate_voxels,
                    "unexplained_voxels": row.unexplained_voxels,
                    "unexplained_ratio": row.unexplained_ratio,
                    "accepted": row.accepted,
                    "reason": row.reason,
                }
                for row in gate_rows
            ],
        }
        input_hashes["scenes"][scene] = {
            "gt": _sha256(gt_path),
            "axis_alignment": _sha256(metadata_path),
            "boxer_raw_csv": _sha256(raw_csv),
            "boxer_tracked_csv": _sha256(tracked_csv),
            "sealed_schedule_manifest": _sha256(schedule_manifest),
            "depth_intrinsics": _sha256(intrinsics_path),
            "boxer_world_offset_pose": _sha256(first_pose_path),
            "processed_depth": depth_hashes,
            "processed_pose": pose_hashes,
        }

    baseline_evaluations = {
        threshold: official_constant_evaluate(baseline_iou, gt_counts, threshold)
        for threshold in THRESHOLDS
    }
    pools_report: dict[str, object] = {}
    suffix_report: dict[str, object] = {}
    for pool_name in POOL_NAMES:
        pools_report[pool_name] = {
            "candidate_count": int(
                sum(len(candidates) for candidates in pool_candidates[pool_name])
            ),
            "per_threshold": {
                _threshold_key(threshold): _pool_threshold_report(
                    pool_iou[pool_name],
                    baseline_iou,
                    scenes,
                    baseline_evaluations[threshold]["matched_gt_masks"],
                    threshold,
                    float(baseline_evaluations[threshold]["ap"]),
                )
                for threshold in THRESHOLDS
            },
        }
        if pool_name == "tracked":
            pools_report[pool_name]["schedule_clean"] = tracked_schedule_clean_all
            pools_report[pool_name]["schedule_warning"] = (
                None
                if tracked_schedule_clean_all
                else "terminal tracker CSV cannot be decontaminated after extra raw frames"
            )
        suffix_report[pool_name] = {
            "per_threshold": {
                _threshold_key(threshold): _gt_selected_suffix_report(
                    scenes=scenes,
                    baseline_matrices=baseline_iou,
                    pool_matrices=pool_iou[pool_name],
                    pool_candidates=pool_candidates[pool_name],
                    gt_counts=gt_counts,
                    baseline_evaluation=baseline_evaluations[threshold],
                    threshold=threshold,
                )
                for threshold in THRESHOLDS
            }
        }

    depth_constructive_deltas = {
        _threshold_key(threshold): float(
            suffix_report["depth_gated"]["per_threshold"][
                _threshold_key(threshold)
            ]["delta_ap_points"]
        )
        for threshold in THRESHOLDS
    }
    depth_necessary = {
        _threshold_key(threshold): bool(
            pools_report["depth_gated"]["per_threshold"][
                _threshold_key(threshold)
            ]["necessary_headroom_for_plus10"]["passes"]
        )
        for threshold in THRESHOLDS
    }
    depth_recovered_miss_headroom = {
        _threshold_key(threshold): float(
            pools_report["depth_gated"]["per_threshold"][
                _threshold_key(threshold)
            ]["incremental_recall_headroom_points"]
        )
        for threshold in THRESHOLDS
    }
    constructive_passes = all(value >= 10.0 for value in depth_constructive_deltas.values())
    necessary_passes = all(depth_necessary.values())
    promotion = {
        "preregistered_pool": "depth_gated",
        "target_delta_ap_points_each_threshold": 10.0,
        "constructive_counterfactual_delta_ap_points": depth_constructive_deltas,
        "constructive_counterfactual_passes_all_thresholds": constructive_passes,
        "necessary_headroom_passes": depth_necessary,
        "necessary_headroom_passes_all_thresholds": necessary_passes,
        "incremental_recall_headroom_points": depth_recovered_miss_headroom,
        "decision": (
            "preflight_pass_repeat_on_larger_sealed_set"
            if constructive_passes and necessary_passes
            else (
                "reject_frozen_depth_gate_insufficient_recovered_miss_headroom"
                if not necessary_passes
                else "reject_frozen_depth_gate_preflight"
            )
        ),
        "birth_may_be_enabled": False,
        "interpretation": (
            "union matching is a loose necessary ceiling; the fixed-suffix result is a "
            "constructive GT-selected counterfactual, not a mathematical AP upper bound"
        ),
    }

    baseline_after = {
        scene: _sha256(baseline_root / f"{scene}_boxes.pkl") for scene in scenes
    }
    if baseline_after != baseline_before:
        raise OracleAuditError("native T05 predictions changed during read-only audit")
    if (shadow_seal_json is None) != (shadow_seal_npz is None):
        raise OracleAuditError("shadow seal JSON and NPZ must be supplied together")
    shadow_seal = None
    if shadow_seal_json is not None and shadow_seal_npz is not None:
        shadow_seal = validate_shadow_seal(
            json_path=shadow_seal_json,
            npz_path=shadow_seal_npz,
            scenes=scenes,
            raw_candidates=pool_candidates["raw"],
            tracked_candidates=pool_candidates["tracked"],
            boxer_root=boxer_root,
            schedule_root=schedule_root,
        )

    return {
        "schema": SCHEMA,
        "oracle_only": True,
        "deployable": False,
        "gt_used": True,
        "birth_enabled": False,
        "native_predictions_modified": False,
        "score_mode": "constant_1.0",
        "class_mode": "class_agnostic",
        "strict_iou_comparison": ">",
        "scene_order": scenes,
        "sealed_schedule_namespace": schedule_namespace,
        "thresholds": list(THRESHOLDS),
        "frozen_geometry": {
            "boxer_world_offset_restored": True,
            "pixel_stride": PIXEL_STRIDE,
            "valid_depth_m": [MIN_DEPTH_M, MAX_DEPTH_M],
            "candidate_support": "depth_points_inside_boxer_obb",
            "voxel_size_m": VOXEL_SIZE_M,
            "voxel_quantization": "signed_floor",
            "voxel_representative": "mean_of_supported_points_per_voxel",
            "minimum_candidate_voxels": MIN_CANDIDATE_VOXELS,
            "minimum_unexplained_voxels": MIN_UNEXPLAINED_VOXELS,
            "minimum_unexplained_ratio": MIN_UNEXPLAINED_RATIO,
            "native_explanation": "final_t05_world_aabb",
            "native_aabb_expansion_m": NATIVE_AABB_EXPANSION_M,
            "future_aware_offline_filter": True,
            "boxer_start_n": BOXER_START_N,
            "boxer_skip_n": BOXER_SKIP_N,
        },
        "totals": {
            "scene_count": len(scenes),
            "gt_count": int(sum(gt_counts)),
            "baseline_prediction_count": int(sum(len(rows) for rows in baseline_aligned)),
        },
        "baseline": {
            "per_threshold": {
                _threshold_key(threshold): _json_baseline_evaluation(
                    baseline_evaluations[threshold], scenes
                )
                for threshold in THRESHOLDS
            }
        },
        "pools": pools_report,
        "gt_selected_fixed_native_prefix_suffix": suffix_report,
        "promotion": promotion,
        "scenes": scene_reports,
        "input_sha256": input_hashes,
        "native_prediction_sha256_before": baseline_before,
        "native_prediction_sha256_after": baseline_after,
        "shadow_seal": shadow_seal,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit frozen Boxer unexplained-depth proposal oracle on ScanNet"
    )
    parser.add_argument(
        "--boxer-root",
        required=True,
        type=Path,
        help="Root containing <scene>/boxer_3dbbs{,_tracked}.csv",
    )
    parser.add_argument(
        "--schedule-root",
        required=True,
        type=Path,
        help="Sealed CuTR cache root containing <scene>/manifest.json",
    )
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--scene-list", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--scan-root", required=True, type=Path)
    parser.add_argument("--scene-rgbd-root", required=True, type=Path)
    parser.add_argument("--shadow-seal-json", type=Path)
    parser.add_argument("--shadow-seal-npz", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_output_path(
        args.out,
        (
            args.boxer_root,
            args.schedule_root,
            args.baseline_root,
            args.gt_root,
            args.scan_root,
            args.scene_rgbd_root,
            *(path for path in (args.shadow_seal_json, args.shadow_seal_npz) if path),
        ),
    )
    report = audit_scannet_boxer_unexplained_oracle(
        boxer_root=args.boxer_root,
        schedule_root=args.schedule_root,
        baseline_root=args.baseline_root,
        scene_list=args.scene_list,
        gt_root=args.gt_root,
        scan_root=args.scan_root,
        scene_rgbd_root=args.scene_rgbd_root,
        shadow_seal_json=args.shadow_seal_json,
        shadow_seal_npz=args.shadow_seal_npz,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"schema": SCHEMA, "out": os.fspath(args.out), "totals": report["totals"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
