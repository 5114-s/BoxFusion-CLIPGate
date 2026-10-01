#!/usr/bin/env python3
"""Target-first, frozen MobileSAM mask-lift observer for ScanNet full100.

The runner deliberately has no annotation or evaluator argument.  It starts
from every exact target-vocabulary OWLv2/Raw-Boxer row, applies a deterministic
Top-4 cap per causal keyframe, lifts frozen MobileSAM box masks through RGB-D
and pose, and reconstructs first-three receipts from the lifted AABBs.  It is
an output-inert shadow: native predictions are read only to report novelty
metrics and no prediction pickle is written or changed.

The old Raw-Boxer receipt manifest is used only as an immutable full100 scene
ledger (not as proposal membership or tracking input).  In particular, rows
rejected by the old birth policy remain eligible here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import pickle
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
import zipfile

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))
TOOLS_ROOT = REPOSITORY_ROOT / "tools"
if os.fspath(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(TOOLS_ROOT))

from boxfusion import target_masklift as target_core  # noqa: E402
import run_scannet_raw_boxer_clip_vocab_shadow_full100 as clip_shadow  # noqa: E402
import run_scannet_s3a_boxer_mobilesam_masklift_shadow as s3a  # noqa: E402


SCHEMA = "boxfusion.scannet_target_first_mobilesam_masklift_full100.v1"
EXPECTED_RECEIPT_SCHEMA = "boxfusion.scannet_raw_boxer_past3_birth_full100.v2_m50"
OUTPUT_JSON = "TARGET_FIRST_MOBILESAM_MASKLIFT_FULL100.json"
OUTPUT_NPZ = "TARGET_FIRST_MOBILESAM_MASKLIFT_FULL100.npz"

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
TOP_K_PER_FRAME = 4
RAW_MIN_SCORE = 0.40

# Frozen R15 routing policy.  All values are no-GT geometric/source tests.
ROUTING_POLICY: Mapping[str, float | int] = {
    "min_evidence_score": RAW_MIN_SCORE,
    "min_mean_score": 0.50,
    "min_median_pairwise_mask_aabb_iou": 0.15,
    "max_pairwise_mask_center_distance_m": 0.50,
    "min_first_last_frame_span": 50,
    "min_camera_baseline_m": 0.10,
    "min_view_ray_span_deg": 5.0,
    "fusion_voxel_size_m": 0.05,
    "min_supported_voxels": 24,
    "min_supported_voxels_per_view": 8,
    "min_fused_extent_m": 0.05,
    "max_fused_to_raw_medoid_center_m": 0.75,
    "native_max_aabb_iou": 0.10,
    "native_max_bidirectional_containment": 0.50,
}

SIGNS = np.asarray(
    [
        (-1.0, -1.0, -1.0),
        (-1.0, -1.0, +1.0),
        (-1.0, +1.0, -1.0),
        (-1.0, +1.0, +1.0),
        (+1.0, -1.0, -1.0),
        (+1.0, -1.0, +1.0),
        (+1.0, +1.0, -1.0),
        (+1.0, +1.0, +1.0),
    ],
    dtype=np.float64,
)

RAW_COLUMNS = {
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
}


class TargetFirstMaskLiftError(RuntimeError):
    """Raised when a no-GT input or causal shadow contract is violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise TargetFirstMaskLiftError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    path = _regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TargetFirstMaskLiftError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise TargetFirstMaskLiftError(f"{label} must contain a JSON object")
    return value


def _finite(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TargetFirstMaskLiftError(f"{label} must be finite") from error
    if not math.isfinite(result):
        raise TargetFirstMaskLiftError(f"{label} must be finite")
    return result


def _integer(value: object, label: str, minimum: int = 0) -> int:
    number = _finite(value, label)
    if not number.is_integer() or number < minimum:
        raise TargetFirstMaskLiftError(f"{label} must be an integer >= {minimum}")
    return int(number)


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    path = _regular_file(path, "source CSV")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise TargetFirstMaskLiftError(f"CSV missing columns {sorted(missing)}: {path}")
            return [dict(row) for row in reader]
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise TargetFirstMaskLiftError(f"invalid CSV: {path}") from error


def _quaternion_rotation(quaternion_wxyz: object) -> np.ndarray:
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    norm2 = float(q @ q) if q.shape == (4,) else 0.0
    if q.shape != (4,) or not np.isfinite(q).all() or norm2 <= 1e-12:
        raise TargetFirstMaskLiftError("invalid Raw Boxer quaternion")
    w, x, y, z = q
    scale = 2.0 / norm2
    return np.asarray(
        [
            [1 - scale * (y * y + z * z), scale * (x * y - z * w), scale * (x * z + y * w)],
            [scale * (x * y + z * w), 1 - scale * (x * x + z * z), scale * (y * z - x * w)],
            [scale * (x * z - y * w), scale * (y * z + x * w), 1 - scale * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _corners(center: object, extent: object, rotation: np.ndarray | None = None) -> np.ndarray:
    center_array = np.asarray(center, dtype=np.float64)
    extent_array = np.asarray(extent, dtype=np.float64)
    if center_array.shape != (3,) or extent_array.shape != (3,) or np.any(extent_array <= 0):
        raise TargetFirstMaskLiftError("box center/extent must be finite positive [3]")
    if not np.isfinite(center_array).all() or not np.isfinite(extent_array).all():
        raise TargetFirstMaskLiftError("box center/extent must be finite positive [3]")
    local = SIGNS * (extent_array / 2.0)
    return local if rotation is None else local @ np.asarray(rotation).T


def _aabb_corners(center: object, extent: object) -> np.ndarray:
    return _corners(center, extent) + np.asarray(center, dtype=np.float64)


def _obb_corners(center: object, extent: object, rotation: np.ndarray) -> np.ndarray:
    return _corners(center, extent, rotation) + np.asarray(center, dtype=np.float64)


def _bounds_and_volume(corners: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = np.asarray(corners, dtype=np.float64)
    if boxes.size == 0:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0,))
    if boxes.ndim == 2:
        boxes = boxes[None]
    if boxes.ndim != 3 or boxes.shape[1:] != (8, 3) or not np.isfinite(boxes).all():
        raise TargetFirstMaskLiftError("corners must have shape [N,8,3]")
    lower = boxes.min(axis=1)
    upper = boxes.max(axis=1)
    return lower, upper, np.prod(upper - lower, axis=1)


def _aabb_overlap(
    left_corners: object, right_corners: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ll, lu, lv = _bounds_and_volume(left_corners)
    rl, ru, rv = _bounds_and_volume(right_corners)
    if not len(ll) or not len(rl):
        shape = (len(ll), len(rl))
        empty = np.zeros(shape, dtype=np.float64)
        return empty, empty.copy(), empty.copy()
    extent = np.maximum(np.minimum(lu[:, None], ru[None]) - np.maximum(ll[:, None], rl[None]), 0)
    intersection = np.prod(extent, axis=2)
    union = lv[:, None] + rv[None] - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    left_in_right = np.divide(intersection, lv[:, None], out=np.zeros_like(intersection), where=lv[:, None] > 0)
    right_in_left = np.divide(intersection, rv[None], out=np.zeros_like(intersection), where=rv[None] > 0)
    return iou, left_in_right, right_in_left


def _circular_pi_yaw(quaternions_wxyz: object) -> float:
    values = np.asarray(quaternions_wxyz, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or not len(values):
        raise TargetFirstMaskLiftError("yaw input must have shape [N,4]")
    yaws = np.asarray(
        [math.atan2(rotation[1, 0], rotation[0, 0]) for rotation in map(_quaternion_rotation, values)]
    )
    # A box orientation is pi-periodic, hence the doubled-angle mean.
    return 0.5 * math.atan2(float(np.sin(2 * yaws).mean()), float(np.cos(2 * yaws).mean()))


def _quantile_yaw_obb(points_world: object, yaw: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 1 or not np.isfinite(points).all():
        raise TargetFirstMaskLiftError("fused OBB points must be finite [N,3]")
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]], dtype=np.float64)
    local = points @ rotation
    lower, upper = np.quantile(local, [0.02, 0.98], axis=0)
    extent = np.maximum(upper - lower, 0.02)
    center_world = ((lower + upper) * 0.5) @ rotation.T
    return center_world.astype(np.float32), extent.astype(np.float32), _obb_corners(center_world, extent, rotation).astype(np.float32)


def _fuse_three_view_points(
    points_by_view: Sequence[np.ndarray],
    *,
    object_memory: Any,
    observation_voxel_m: float = 0.02,
    fusion_voxel_m: float = 0.05,
    cap: int = 8192,
) -> dict[str, Any]:
    """Call the tested core consensus after fixed 2-cm per-view bounding."""

    if fusion_voxel_m != target_core.VOXEL_SIZE_METERS:
        raise TargetFirstMaskLiftError("fusion voxel size differs from tested target core")
    downsampled = []
    for value in points_by_view:
        points = object_memory.voxel_downsample(value, observation_voxel_m)
        points = object_memory.deterministic_bounded_sample(points, cap)
        downsampled.append(np.asarray(points, dtype=np.float32))
    consensus = target_core.fuse_three_view_points(downsampled)
    return {
        "points": np.asarray(consensus.supported_points, dtype=np.float32),
        "supported_voxel_count": int(consensus.voxel_count),
        "per_view_supported_voxels": [
            int(value) for value in consensus.exact_supported_voxel_counts
        ],
        "per_view_neighborhood_supported_voxels": [
            int(value) for value in consensus.neighborhood_supported_voxel_counts
        ],
        "supported_point_counts": [int(value) for value in consensus.supported_point_counts],
    }


def _pairwise_metrics(corners: object) -> dict[str, Any]:
    boxes = np.asarray(corners, dtype=np.float64)
    iou, _, _ = _aabb_overlap(boxes, boxes)
    lower, upper, _ = _bounds_and_volume(boxes)
    centers = 0.5 * (lower + upper)
    distances = np.linalg.norm(centers[:, None] - centers[None], axis=2)
    upper_values = iou[np.triu_indices(len(boxes), 1)]
    distance_values = distances[np.triu_indices(len(boxes), 1)]
    costs = np.sum(1.0 - iou, axis=1)
    medoid = min(range(len(boxes)), key=lambda index: (float(costs[index]), index))
    return {
        "min_pairwise_iou": float(np.min(upper_values)),
        "median_pairwise_iou": float(np.median(upper_values)),
        "max_pairwise_center_distance_m": float(np.max(distance_values)),
        "center_rms_m": float(np.sqrt(np.mean(np.sum((centers - centers.mean(axis=0)) ** 2, axis=1)))),
        "medoid_index": int(medoid),
        "centers": centers,
    }


def _view_diversity(camera_centers: np.ndarray, object_center: np.ndarray) -> tuple[float, float]:
    centers = np.asarray(camera_centers, dtype=np.float64)
    center = np.asarray(object_center, dtype=np.float64)
    baselines = np.linalg.norm(centers[:, None] - centers[None], axis=2)
    rays = center[None] - centers
    norms = np.linalg.norm(rays, axis=1)
    if np.any(norms <= 1e-9):
        return float(baselines.max()), 0.0
    unit = rays / norms[:, None]
    angles = np.degrees(np.arccos(np.clip(unit @ unit.T, -1, 1)))
    return float(baselines.max()), float(angles.max())


def _read_schedule(schedule_root: Path, scene_root: Path, scene: str) -> dict[str, Any]:
    manifest_path = _regular_file(schedule_root / scene / "manifest.json", f"schedule for {scene}")
    manifest = _read_json(manifest_path, f"schedule for {scene}")
    raw = manifest.get("recorded_frame_ids")
    if not isinstance(raw, list) or any(isinstance(x, bool) or not isinstance(x, int) for x in raw) or raw != sorted(set(raw)):
        raise TargetFirstMaskLiftError(f"invalid schedule frame IDs for {scene}")
    valid: list[int] = []
    poses: dict[int, np.ndarray] = {}
    for frame_id in raw:
        path = scene_root / scene / "frames" / "pose" / f"{frame_id}.txt"
        try:
            pose = np.loadtxt(_regular_file(path, f"pose {scene}/{frame_id}"), dtype=np.float64)
        except (OSError, ValueError, TargetFirstMaskLiftError):
            continue
        if pose.shape == (4, 4) and np.isfinite(pose).all():
            valid.append(frame_id)
            poses[frame_id] = pose
    if not valid:
        raise TargetFirstMaskLiftError(f"schedule has no valid pose for {scene}")
    intrinsic_path = _regular_file(scene_root / scene / "frames/intrinsic/intrinsic_depth.txt", f"intrinsic for {scene}")
    intrinsic = np.loadtxt(intrinsic_path, dtype=np.float64)
    if intrinsic.shape != (4, 4) or not np.isfinite(intrinsic).all():
        raise TargetFirstMaskLiftError(f"invalid depth intrinsic for {scene}")
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "valid_frames": tuple(valid),
        "valid_ordinal": {frame: index for index, frame in enumerate(valid)},
        "poses": poses,
        "intrinsic_path": intrinsic_path,
        "intrinsic": intrinsic[:3, :3].copy(),
    }


def _raw_geometry(row: Mapping[str, str], world_offset: np.ndarray) -> dict[str, np.ndarray]:
    center = np.asarray([_finite(row[key], key) for key in ("tx_world_object", "ty_world_object", "tz_world_object")]) + world_offset
    quaternion = np.asarray([_finite(row[key], key) for key in ("qw_world_object", "qx_world_object", "qy_world_object", "qz_world_object")])
    extent = np.asarray([_finite(row[key], key) for key in ("scale_x", "scale_y", "scale_z")])
    return {
        "center": center.astype(np.float32),
        "quaternion": quaternion.astype(np.float32),
        "extent": extent.astype(np.float32),
        "corners": _obb_corners(center, extent, _quaternion_rotation(quaternion)).astype(np.float32),
    }


def _select_target_candidates(
    raw_rows: Sequence[Mapping[str, str]],
    owl_rows: Sequence[Mapping[str, str]],
    *,
    valid_ordinal: Mapping[int, int],
    world_offset: np.ndarray,
    min_score: float = RAW_MIN_SCORE,
    top_k: int = TOP_K_PER_FRAME,
) -> list[dict[str, Any]]:
    owl_by_frame = clip_shadow._index_owl_rows(owl_rows)
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for source_row, raw in enumerate(raw_rows):
        frame_id = _integer(raw.get("time_ns"), "Raw time_ns")
        if frame_id not in valid_ordinal:
            raise TargetFirstMaskLiftError(f"Raw row references off-schedule frame {frame_id}")
        score = _finite(raw.get("prob"), "Raw probability")
        if score < min_score:
            continue
        instance = _integer(raw.get("instance"), "Raw instance")
        frame_owl = owl_by_frame.get(frame_id)
        if frame_owl is None or instance >= len(frame_owl):
            raise TargetFirstMaskLiftError(f"missing exact OWL key {(frame_id, instance)}")
        owl = frame_owl[instance]
        raw_name, owl_name = str(raw.get("name", "")), str(owl.get("name", ""))
        raw_sem = _integer(raw.get("sem_id"), "Raw semantic ID")
        owl_sem = _integer(owl.get("sem_id"), "OWL semantic ID")
        if (
            clip_shadow._normalize_owl_name(raw_name)
            != clip_shadow._normalize_owl_name(owl_name)
            or raw_sem != owl_sem
        ):
            raise TargetFirstMaskLiftError(f"Raw/OWL identity differs at source row {source_row}")
        group = clip_shadow._resolve_owl_target_group(owl_name)
        if group is None:
            continue
        owl_ordinal = _integer(owl.get("frame_id"), "OWL frame ordinal")
        if owl_ordinal != valid_ordinal[frame_id]:
            raise TargetFirstMaskLiftError(f"OWL provider ordinal differs at {(frame_id, instance)}")
        width = _integer(owl.get("img_width"), "OWL width", 1)
        height = _integer(owl.get("img_height"), "OWL height", 1)
        bbox = np.asarray([_finite(owl.get(key), f"OWL {key}") for key in ("x1", "y1", "x2", "y2")], dtype=np.float32)
        mapped = bbox * np.asarray([IMAGE_WIDTH / width, IMAGE_HEIGHT / height, IMAGE_WIDTH / width, IMAGE_HEIGHT / height], dtype=np.float32)
        mapped[[0, 2]] = np.clip(mapped[[0, 2]], 0, IMAGE_WIDTH)
        mapped[[1, 3]] = np.clip(mapped[[1, 3]], 0, IMAGE_HEIGHT)
        geometry = _raw_geometry(raw, world_offset)
        by_frame.setdefault(frame_id, []).append(
            {
                "frame_id": frame_id,
                "source_row": source_row,
                "source_instance_id": instance,
                "semantic_id": raw_sem,
                "target_group": group,
                "raw_name": raw_name,
                "score": score,
                "owl_box_xyxy": bbox,
                "prompt_box_xyxy": mapped,
                **{f"raw_{key}": value for key, value in geometry.items()},
            }
        )
    output: list[dict[str, Any]] = []
    for frame_id in sorted(by_frame):
        ranked = sorted(by_frame[frame_id], key=lambda row: (-row["score"], row["source_row"]))
        output.extend(ranked[:top_k])
    return output


def _load_native(path: Path) -> np.ndarray:
    path = _regular_file(path, "native prediction")
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as error:
        raise TargetFirstMaskLiftError(f"cannot load native prediction: {path}") from error
    if not isinstance(payload, (list, tuple)) or len(payload) != 1 or not isinstance(payload[0], (list, tuple)):
        raise TargetFirstMaskLiftError(f"invalid native prediction schema: {path}")
    corners: list[np.ndarray] = []
    for index, row in enumerate(payload[0]):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise TargetFirstMaskLiftError(f"invalid native row {index}: {path}")
        box = np.asarray(row[1], dtype=np.float64)
        if box.shape != (8, 3) or not np.isfinite(box).all():
            raise TargetFirstMaskLiftError(f"invalid native corners {index}: {path}")
        corners.append(box)
    return np.stack(corners) if corners else np.empty((0, 8, 3), dtype=np.float64)


def _decode_frame(scene_root: Path, scene: str, frame_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import cv2
    except ImportError as error:
        raise TargetFirstMaskLiftError("OpenCV is unavailable") from error
    base = scene_root / scene / "frames"
    rgb_path = _regular_file(base / "color" / f"{frame_id}.jpg", "RGB frame")
    depth_path = _regular_file(base / "depth" / f"{frame_id}.png", "depth frame")
    pose_path = _regular_file(base / "pose" / f"{frame_id}.txt", "pose frame")
    bgr = cv2.imread(os.fspath(rgb_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(os.fspath(depth_path), cv2.IMREAD_UNCHANGED)
    pose = np.loadtxt(pose_path, dtype=np.float64)
    if bgr is None or depth is None or depth.shape != (IMAGE_HEIGHT, IMAGE_WIDTH) or depth.dtype != np.uint16:
        raise TargetFirstMaskLiftError(f"invalid RGB-D frame {scene}/{frame_id}")
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise TargetFirstMaskLiftError(f"invalid pose frame {scene}/{frame_id}")
    bgr = cv2.resize(bgr, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), np.ascontiguousarray(depth), pose


def _process_scene(
    *,
    scene: str,
    scene_index: int,
    candidates: list[dict[str, Any]],
    schedule: Mapping[str, Any],
    scene_root: Path,
    native_corners: np.ndarray,
    engine: Any,
    object_memory: Any,
    global_row_start: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[np.ndarray], list[dict[str, Any]]]:
    by_frame: dict[int, list[int]] = {}
    for local_index, row in enumerate(candidates):
        row["scene_index"] = scene_index
        row["global_row"] = global_row_start + local_index
        by_frame.setdefault(int(row["frame_id"]), []).append(local_index)
    runtime_rows: list[dict[str, Any]] = []
    for frame_id in sorted(by_frame):
        indices = by_frame[frame_id]
        if len(indices) > TOP_K_PER_FRAME:
            raise TargetFirstMaskLiftError("target-first Top4 cap was exceeded")
        started = time.perf_counter()
        rgb, depth, pose = _decode_frame(scene_root, scene, frame_id)
        decode_ms = (time.perf_counter() - started) * 1000.0
        boxes = np.stack([candidates[index]["prompt_box_xyxy"] for index in indices]).astype(np.float32)
        valid = np.isfinite(boxes).all(axis=1) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        provider_total = 0.0
        encoder_total = 0.0
        decoder_total = 0.0
        lifting_total = 0.0
        valid_positions = np.flatnonzero(valid)
        if len(valid_positions):
            # The explicit chunk keeps the contract valid if a future caller
            # lowers the selection cap while preserving shared frame ordering.
            for chunk_start in range(0, len(valid_positions), TOP_K_PER_FRAME):
                chunk = valid_positions[chunk_start : chunk_start + TOP_K_PER_FRAME]
                masks, predicted_iou, hypotheses, timing = engine.predict(rgb, boxes[chunk])
                provider_total += float(timing["provider_ms"])
                encoder_total += float(timing["encoder_ms"])
                decoder_total += float(timing["decoder_and_host_mask_ms"])
                for batch_index, position in enumerate(chunk.tolist()):
                    result = s3a._lift_mask_row(
                        mask=masks[batch_index],
                        predicted_iou=float(predicted_iou[batch_index]),
                        hypothesis_index=int(hypotheses[batch_index]),
                        depth=depth,
                        intrinsic=np.asarray(schedule["intrinsic"]),
                        pose=pose,
                        object_memory=object_memory,
                    )
                    result.pop("sam_mask_packed", None)
                    result.pop("cleaned_depth_mask_packed", None)
                    result["points_sha256"] = _hash_array(np.asarray(result["points_world"], dtype=np.float32))
                    lifting_total += float(result["lifting_ms"])
                    candidates[indices[position]]["lift"] = result
        for position, local_index in enumerate(indices):
            if not valid[position]:
                result = s3a._empty_row_result()
                result.pop("sam_mask_packed", None)
                result.pop("cleaned_depth_mask_packed", None)
                result["points_sha256"] = _hash_array(np.empty((0, 3), dtype=np.float32))
                candidates[local_index]["lift"] = result
        runtime_rows.append(
            {
                "scene_id": scene,
                "frame_id": frame_id,
                "prompt_count": len(indices),
                "valid_prompt_count": int(len(valid_positions)),
                "decode_ms": decode_ms,
                "encoder_ms": encoder_total,
                "decoder_and_host_mask_ms": decoder_total,
                "provider_ms": provider_total,
                "lifting_ms": lifting_total,
                "total_incremental_ms": decode_ms + provider_total + lifting_total,
            }
        )

    # The tested core tracker associates only within the collapsed exact target
    # group.  Raw OWL semantic IDs remain diagnostics and do not split aliases.
    tracker = target_core.PastOnlyTargetTracker()
    observations: dict[int, list[target_core.TargetObservation]] = {}
    for local_index, row in enumerate(candidates):
        lift = row["lift"]
        if not lift["accepted"]:
            continue
        center = np.asarray(lift["reported_q02_q98_center_world"], dtype=np.float64)
        half = np.asarray(lift["reported_q02_q98_extent_xyz"], dtype=np.float64) * 0.5
        observations.setdefault(int(row["frame_id"]), []).append(
            target_core.TargetObservation(
                observation_id=local_index,
                frame_id=int(row["frame_id"]),
                target_group=str(row["target_group"]),
                aabb_lower=center - half,
                aabb_upper=center + half,
            )
        )
    for frame_id in schedule["valid_frames"]:
        tracker.update(int(frame_id), observations.get(int(frame_id), ()))

    track_rows: list[dict[str, Any]] = []
    fused_points: list[np.ndarray] = []
    for receipt in tracker.confirmed_tracks:
            group = receipt.target_group
            evidence_indices = list(receipt.evidence_observation_ids)
            evidence = [candidates[index] for index in evidence_indices]
            points_by_view = [np.asarray(row["lift"]["points_world"], dtype=np.float32) for row in evidence]
            fusion = _fuse_three_view_points(points_by_view, object_memory=object_memory)
            points = np.asarray(fusion["points"], dtype=np.float32)
            raw_corners = np.stack([row["raw_corners"] for row in evidence])
            mask_corners = np.stack(
                [_aabb_corners(row["lift"]["reported_q02_q98_center_world"], row["lift"]["reported_q02_q98_extent_xyz"]) for row in evidence]
            )
            raw_metrics = _pairwise_metrics(raw_corners)
            mask_metrics = _pairwise_metrics(mask_corners)
            raw_medoid = int(raw_metrics["medoid_index"])
            raw_medoid_center = np.asarray(evidence[raw_medoid]["raw_center"], dtype=np.float64)
            scores = [float(row["score"]) for row in evidence]
            frames = [int(row["frame_id"]) for row in evidence]
            camera_centers = np.stack([schedule["poses"][frame][:3, 3] for frame in frames])
            if len(points) >= 1:
                fused_center, fused_extent = object_memory.robust_quantile_aabb(points, 0.02, 0.98, 1, 0.02)
                fused_aabb = _aabb_corners(fused_center, fused_extent).astype(np.float32)
                obb = target_core.robust_yaw_obb(
                    points,
                    quaternions_wxyz=np.stack(
                        [row["raw_quaternion"] for row in evidence]
                    ),
                )
                yaw = float(obb.yaw_rad)
                obb_center = np.asarray(obb.center, dtype=np.float32)
                obb_extent = np.asarray(obb.extent, dtype=np.float32)
                fused_obb = np.asarray(obb.corners, dtype=np.float32)
                baseline, ray_span = _view_diversity(camera_centers, obb_center)
                center_to_raw = float(np.linalg.norm(obb_center - raw_medoid_center))
                native_iou, candidate_in_native, native_in_candidate = _aabb_overlap(fused_obb[None], native_corners)
                max_iou = float(native_iou.max(initial=0.0))
                max_candidate_in_native = float(candidate_in_native.max(initial=0.0))
                max_native_in_candidate = float(native_in_candidate.max(initial=0.0))
            else:
                fused_center = np.zeros(3, dtype=np.float32)
                fused_extent = np.zeros(3, dtype=np.float32)
                fused_aabb = np.zeros((8, 3), dtype=np.float32)
                obb_center = np.zeros(3, dtype=np.float32)
                obb_extent = np.zeros(3, dtype=np.float32)
                fused_obb = np.zeros((8, 3), dtype=np.float32)
                yaw = 0.0
                baseline, ray_span, center_to_raw = 0.0, 0.0, 1.0e30
                max_iou = max_candidate_in_native = max_native_in_candidate = 0.0

            gates = {
                "same_exact_target_group": len({row["target_group"] for row in evidence}) == 1,
                "min_score": min(scores) >= float(ROUTING_POLICY["min_evidence_score"]),
                "mean_score": float(np.mean(scores)) >= float(ROUTING_POLICY["min_mean_score"]),
                "mask_median_iou": mask_metrics["median_pairwise_iou"] >= float(ROUTING_POLICY["min_median_pairwise_mask_aabb_iou"]),
                "mask_center_distance": mask_metrics["max_pairwise_center_distance_m"] <= float(ROUTING_POLICY["max_pairwise_mask_center_distance_m"]),
                "frame_span": frames[-1] - frames[0] >= int(ROUTING_POLICY["min_first_last_frame_span"]),
                "camera_baseline": baseline >= float(ROUTING_POLICY["min_camera_baseline_m"]),
                "view_ray_span": ray_span >= float(ROUTING_POLICY["min_view_ray_span_deg"]),
                "supported_voxels": int(fusion["supported_voxel_count"]) >= int(ROUTING_POLICY["min_supported_voxels"]),
                "per_view_supported_voxels": min(fusion["per_view_supported_voxels"]) >= int(ROUTING_POLICY["min_supported_voxels_per_view"]),
                "fused_extent": bool(len(points)) and float(np.min(obb_extent)) >= float(ROUTING_POLICY["min_fused_extent_m"]),
                "fused_to_raw_center": center_to_raw <= float(ROUTING_POLICY["max_fused_to_raw_medoid_center_m"]),
            }
            pre_novelty_pass = all(gates.values())
            novelty = {
                "max_native_aabb_iou": max_iou,
                "max_candidate_in_native_containment": max_candidate_in_native,
                "max_native_in_candidate_containment": max_native_in_candidate,
                "pass": (
                    max_iou < float(ROUTING_POLICY["native_max_aabb_iou"])
                    and max_candidate_in_native < float(ROUTING_POLICY["native_max_bidirectional_containment"])
                    and max_native_in_candidate < float(ROUTING_POLICY["native_max_bidirectional_containment"])
                ),
            }
            decision = "accepted_shadow" if pre_novelty_pass and novelty["pass"] else (
                "native_overlap" if pre_novelty_pass else next(name for name, passed in gates.items() if not passed)
            )
            track_rows.append(
                {
                    "scene_index": scene_index,
                    "scene_id": scene,
                    "target_group": group,
                    "semantic_id": int(evidence[0]["semantic_id"]),
                    "evidence_semantic_ids": [int(row["semantic_id"]) for row in evidence],
                    "semantic_id_consistent": len({int(row["semantic_id"]) for row in evidence}) == 1,
                    "group_track_id": int(receipt.track_id),
                    "confirmation_frame_id": int(frames[-1]),
                    "evidence_local_rows": evidence_indices,
                    "evidence_global_rows": [int(row["global_row"]) for row in evidence],
                    "evidence_frame_ids": frames,
                    "evidence_source_rows": [int(row["source_row"]) for row in evidence],
                    "evidence_scores": scores,
                    "raw_metrics": {key: value for key, value in raw_metrics.items() if key not in ("centers",)},
                    "mask_metrics": {key: value for key, value in mask_metrics.items() if key not in ("centers",)},
                    "min_raw_medoid_extent_m": float(np.min(np.ptp(raw_corners[raw_medoid], axis=0))),
                    "camera_baseline_m": baseline,
                    "view_ray_span_deg": ray_span,
                    "supported_voxel_count": int(fusion["supported_voxel_count"]),
                    "per_view_supported_voxels": fusion["per_view_supported_voxels"],
                    "per_view_neighborhood_supported_voxels": fusion[
                        "per_view_neighborhood_supported_voxels"
                    ],
                    "fused_to_raw_medoid_center_m": center_to_raw,
                    "fused_aabb_center_world": np.asarray(fused_center).tolist(),
                    "fused_aabb_extent_xyz": np.asarray(fused_extent).tolist(),
                    "fused_obb_center_world": np.asarray(obb_center).tolist(),
                    "fused_obb_extent_xyz": np.asarray(obb_extent).tolist(),
                    "fused_obb_yaw_rad": yaw,
                    "fused_point_count": len(points),
                    "fused_points_sha256": _hash_array(points),
                    "gates": gates,
                    "pre_novelty_pass": pre_novelty_pass,
                    "native_novelty": novelty,
                    "decision": decision,
                    "fused_aabb_corners": fused_aabb,
                    "fused_obb_corners": fused_obb,
                }
            )
            fused_points.append(points)

    return candidates, track_rows, fused_points, runtime_rows


def _deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("xb") as raw:
        with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name in sorted(arrays):
                payload = io.BytesIO()
                np.lib.format.write_array(payload, np.ascontiguousarray(arrays[name]), allow_pickle=False)
                info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, payload.getvalue(), compresslevel=9)
        raw.flush()
        os.fsync(raw.fileno())


def _publish(output_root: Path, arrays: Mapping[str, np.ndarray], manifest: Mapping[str, Any]) -> None:
    output = output_root.resolve()
    if output_root.is_symlink() or output.exists():
        raise TargetFirstMaskLiftError(f"refusing to overwrite output root: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage.", dir=output.parent))
    try:
        _deterministic_npz(staging / OUTPUT_NPZ, arrays)
        payload = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
        with (staging / OUTPUT_JSON).open("x", encoding="ascii") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        output.mkdir()
        os.link(staging / OUTPUT_NPZ, output / OUTPUT_NPZ)
        os.link(staging / OUTPUT_JSON, output / OUTPUT_JSON)
    except Exception:
        if output.exists():
            shutil.rmtree(output, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _scene_order(scene_list_path: Path, expected_scene_count: int, scene: str | None, max_scenes: int | None) -> tuple[list[str], list[str]]:
    full = [line.strip() for line in _regular_file(scene_list_path, "scene list").read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if len(full) != expected_scene_count or len(set(full)) != len(full):
        raise TargetFirstMaskLiftError(f"expected {expected_scene_count} unique scenes, found {len(full)}")
    if scene is not None:
        if scene not in full:
            raise TargetFirstMaskLiftError(f"requested scene is outside scene list: {scene}")
        selected = [scene]
    else:
        selected = list(full)
    if max_scenes is not None:
        if max_scenes < 1:
            raise TargetFirstMaskLiftError("max-scenes must be positive")
        selected = selected[:max_scenes]
    return full, selected


def _load_scene_candidates(
    *,
    scene: str,
    receipt_scene: Mapping[str, Any],
    raw_log_root: Path,
    schedule_root: Path,
    scene_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    schedule = _read_schedule(schedule_root, scene_root, scene)
    offset = np.asarray(receipt_scene.get("world_offset_xyz"), dtype=np.float64)
    expected_offset = np.asarray(schedule["poses"][schedule["valid_frames"][0]][:3, 3])
    if offset.shape != (3,) or not np.isfinite(offset).all() or not np.allclose(offset, expected_offset, atol=1e-5, rtol=0):
        raise TargetFirstMaskLiftError(f"receipt world offset differs from schedule for {scene}")
    raw_path = _regular_file(raw_log_root / "boxer_raw" / scene / "boxer_3dbbs.csv", f"Raw Boxer CSV {scene}")
    owl_path = _regular_file(raw_log_root / "boxer_raw" / scene / "owl_2dbbs.csv", f"OWL CSV {scene}")
    raw_rows = _read_csv(raw_path, RAW_COLUMNS)
    owl_rows = _read_csv(owl_path, clip_shadow.OWL_REQUIRED_COLUMNS)
    candidates = _select_target_candidates(raw_rows, owl_rows, valid_ordinal=schedule["valid_ordinal"], world_offset=offset)
    ledger = {
        "raw_boxer_csv": os.fspath(raw_path),
        "raw_boxer_csv_sha256": _sha256(raw_path),
        "owl_csv": os.fspath(owl_path),
        "owl_csv_sha256": _sha256(owl_path),
        "schedule_manifest": os.fspath(schedule["manifest_path"]),
        "schedule_manifest_sha256": schedule["manifest_sha256"],
        "intrinsic": os.fspath(schedule["intrinsic_path"]),
        "intrinsic_sha256": _sha256(schedule["intrinsic_path"]),
        "valid_keyframe_count": len(schedule["valid_frames"]),
        "target_prompt_count": len(candidates),
        "target_prompt_frame_count": len({row["frame_id"] for row in candidates}),
    }
    return candidates, schedule, ledger


def _arrays(candidates: Sequence[Mapping[str, Any]], tracks: Sequence[Mapping[str, Any]], fused_blocks: Sequence[np.ndarray], groups: Sequence[str]) -> dict[str, np.ndarray]:
    group_index = {group: index for index, group in enumerate(groups)}
    proposal = {
        "proposal_scene_index": np.asarray([row["scene_index"] for row in candidates], dtype=np.int16),
        "proposal_frame_id": np.asarray([row["frame_id"] for row in candidates], dtype=np.int64),
        "proposal_source_row": np.asarray([row["source_row"] for row in candidates], dtype=np.int32),
        "proposal_source_instance_id": np.asarray([row["source_instance_id"] for row in candidates], dtype=np.int32),
        "proposal_semantic_id": np.asarray([row["semantic_id"] for row in candidates], dtype=np.int32),
        "proposal_target_group_index": np.asarray([group_index[row["target_group"]] for row in candidates], dtype=np.int8),
        "proposal_score": np.asarray([row["score"] for row in candidates], dtype=np.float32),
        "proposal_prompt_box_xyxy": np.asarray([row["prompt_box_xyxy"] for row in candidates], dtype=np.float32),
        "proposal_raw_center_world": np.asarray([row["raw_center"] for row in candidates], dtype=np.float32),
        "proposal_raw_quaternion_wxyz": np.asarray([row["raw_quaternion"] for row in candidates], dtype=np.float32),
        "proposal_raw_extent_xyz": np.asarray([row["raw_extent"] for row in candidates], dtype=np.float32),
        "proposal_lift_accepted": np.asarray([row["lift"]["accepted"] for row in candidates], dtype=bool),
        "proposal_abstention_code": np.asarray([row["lift"]["abstention_code"] for row in candidates], dtype=np.int8),
        "proposal_predicted_iou": np.asarray([row["lift"]["predicted_iou"] for row in candidates], dtype=np.float32),
        "proposal_retained_point_count": np.asarray([row["lift"]["retained_point_count"] for row in candidates], dtype=np.int32),
        "proposal_points_sha256": np.asarray([row["lift"]["points_sha256"] for row in candidates], dtype="<U64"),
        "proposal_lift_center_world": np.asarray([row["lift"]["reported_q02_q98_center_world"] for row in candidates], dtype=np.float32),
        "proposal_lift_extent_xyz": np.asarray([row["lift"]["reported_q02_q98_extent_xyz"] for row in candidates], dtype=np.float32),
    }
    offsets = [0]
    for block in fused_blocks:
        offsets.append(offsets[-1] + len(block))
    fused = np.concatenate(fused_blocks, axis=0).astype(np.float32, copy=False) if offsets[-1] else np.empty((0, 3), dtype=np.float32)
    track = {
        "track_scene_index": np.asarray([row["scene_index"] for row in tracks], dtype=np.int16),
        "track_target_group_index": np.asarray([group_index[row["target_group"]] for row in tracks], dtype=np.int8),
        "track_semantic_id": np.asarray([row["semantic_id"] for row in tracks], dtype=np.int32),
        "track_group_track_id": np.asarray([row["group_track_id"] for row in tracks], dtype=np.int32),
        "track_confirmation_frame_id": np.asarray([row["confirmation_frame_id"] for row in tracks], dtype=np.int64),
        "track_evidence_global_rows": np.asarray([row["evidence_global_rows"] for row in tracks], dtype=np.int64).reshape(-1, 3),
        "track_fused_point_offsets": np.asarray(offsets, dtype=np.int64),
        "track_fused_points_world": fused,
        "track_fused_aabb_corners": np.asarray([row["fused_aabb_corners"] for row in tracks], dtype=np.float32).reshape(-1, 8, 3),
        "track_fused_obb_corners": np.asarray([row["fused_obb_corners"] for row in tracks], dtype=np.float32).reshape(-1, 8, 3),
        "track_pre_novelty_pass": np.asarray([row["pre_novelty_pass"] for row in tracks], dtype=bool),
        "track_native_novelty_pass": np.asarray([row["native_novelty"]["pass"] for row in tracks], dtype=bool),
        "track_accepted_shadow": np.asarray([row["decision"] == "accepted_shadow" for row in tracks], dtype=bool),
    }
    return {"target_group_names": np.asarray(groups, dtype="<U32"), **proposal, **track}


def _canonical_scene_receipts(
    scene_order: Sequence[str], tracks: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Expose a compact stable JSON interface for the active materializer."""

    output: list[dict[str, Any]] = []
    for scene_index, scene_id in enumerate(scene_order):
        receipts: list[dict[str, Any]] = []
        for row in tracks:
            if int(row["scene_index"]) != scene_index:
                continue
            scores = [float(value) for value in row["evidence_scores"]]
            mask = row["mask_metrics"]
            frames = [int(value) for value in row["evidence_frame_ids"]]
            receipts.append(
                {
                    "track_id": int(row["group_track_id"]),
                    "group_track_id": int(row["group_track_id"]),
                    "target_group": str(row["target_group"]),
                    "confirmation_frame_id": int(row["confirmation_frame_id"]),
                    "evidence_frame_ids": frames,
                    "evidence_source_rows": [
                        int(value) for value in row["evidence_source_rows"]
                    ],
                    "evidence_global_rows": [
                        int(value) for value in row["evidence_global_rows"]
                    ],
                    "evidence_scores": scores,
                    "evidence_semantic_ids": [
                        int(value) for value in row["evidence_semantic_ids"]
                    ],
                    "semantic_id_consistent": bool(row["semantic_id_consistent"]),
                    "raw_mean_score": float(np.mean(scores)),
                    "min_evidence_score": float(min(scores)),
                    "median_pairwise_mask_aabb_iou": float(
                        mask["median_pairwise_iou"]
                    ),
                    "max_pairwise_mask_center_distance_m": float(
                        mask["max_pairwise_center_distance_m"]
                    ),
                    "first_last_frame_span": int(frames[-1] - frames[0]),
                    "max_camera_baseline_m": float(row["camera_baseline_m"]),
                    "max_view_ray_span_deg": float(row["view_ray_span_deg"]),
                    "supported_voxel_count": int(row["supported_voxel_count"]),
                    "view_supported_voxel_counts": [
                        int(value) for value in row["per_view_supported_voxels"]
                    ],
                    "fused_center_to_raw_medoid_m": float(
                        row["fused_to_raw_medoid_center_m"]
                    ),
                    "fused_min_obb_extent_m": float(
                        np.min(np.asarray(row["fused_obb_extent_xyz"]))
                    ),
                    "fused_obb": {
                        "center_world": list(row["fused_obb_center_world"]),
                        "extent_xyz": list(row["fused_obb_extent_xyz"]),
                        "yaw_rad": float(row["fused_obb_yaw_rad"]),
                        "corners_world": np.asarray(
                            row["fused_obb_corners"], dtype=np.float64
                        ).tolist(),
                    },
                    "fused_corners_world": np.asarray(
                        row["fused_obb_corners"], dtype=np.float64
                    ).tolist(),
                    "fused_aabb": {
                        "center_world": list(row["fused_aabb_center_world"]),
                        "extent_xyz": list(row["fused_aabb_extent_xyz"]),
                        "corners_world": np.asarray(
                            row["fused_aabb_corners"], dtype=np.float64
                        ).tolist(),
                    },
                    "gates": dict(row["gates"]),
                    "native_novelty": dict(row["native_novelty"]),
                    "accepted": row["decision"] == "accepted_shadow",
                    "decision": str(row["decision"]),
                }
            )
        output.append({"scene_id": scene_id, "receipts": receipts})
    return output


def run_shadow(
    *,
    receipt_manifest_path: Path,
    raw_log_root: Path,
    schedule_root: Path,
    scene_root: Path,
    scene_list_path: Path,
    baseline_root: Path,
    checkpoint: Path,
    output_root: Path,
    device: str,
    expected_scene_count: int = 100,
    scene: str | None = None,
    max_scenes: int | None = None,
    plan_only: bool = False,
    engine_factory: Callable[[str], Any] = s3a.MobileSAMBoxPromptEngine,
) -> dict[str, Any]:
    full_scenes, selected_scenes = _scene_order(scene_list_path, expected_scene_count, scene, max_scenes)
    receipt = _read_json(receipt_manifest_path, "old receipt scene ledger")
    if receipt.get("schema") != EXPECTED_RECEIPT_SCHEMA or receipt.get("gt_access") is not False or receipt.get("evaluator_access") is not False:
        raise TargetFirstMaskLiftError("old receipt manifest contract mismatch")
    receipt_scenes = receipt.get("scenes")
    if not isinstance(receipt_scenes, dict) or set(receipt_scenes) != set(full_scenes):
        raise TargetFirstMaskLiftError("receipt scene set differs from official scene list")

    scene_inputs: dict[str, tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]] = {}
    for scene_id in selected_scenes:
        scene_inputs[scene_id] = _load_scene_candidates(
            scene=scene_id,
            receipt_scene=receipt_scenes[scene_id],
            raw_log_root=raw_log_root,
            schedule_root=schedule_root,
            scene_root=scene_root,
        )
    plan = {
        "schema": SCHEMA,
        "mode": "plan_only" if plan_only else "shadow",
        "scene_count": len(selected_scenes),
        "scene_order": selected_scenes,
        "target_prompt_count": sum(len(value[0]) for value in scene_inputs.values()),
        "target_prompt_frame_count": sum(len({row["frame_id"] for row in value[0]}) for value in scene_inputs.values()),
        "top_k_per_frame": TOP_K_PER_FRAME,
        "raw_min_score": RAW_MIN_SCORE,
        "selection_source": "all_raw_rows_not_old_receipt_membership",
    }
    if plan_only:
        print(json.dumps(plan, sort_keys=True), flush=True)
        return plan

    if output_root.exists() or output_root.is_symlink():
        raise TargetFirstMaskLiftError(f"refusing to overwrite output: {output_root}")
    checkpoint_path = _regular_file(checkpoint, "MobileSAM checkpoint")
    # The reused engine is intentionally pinned to the same checkpoint path.
    if checkpoint_path != s3a.MOBILESAM_CHECKPOINT.resolve():
        raise TargetFirstMaskLiftError("engine/checkpoint path mismatch")
    object_memory = s3a._load_object_memory_module()
    engine = engine_factory(device)

    candidates_all: list[dict[str, Any]] = []
    tracks_all: list[dict[str, Any]] = []
    fused_all: list[np.ndarray] = []
    runtime_all: list[dict[str, Any]] = []
    input_ledgers: dict[str, Any] = {}
    scene_summaries: dict[str, Any] = {}
    for scene_index, scene_id in enumerate(selected_scenes):
        candidates, schedule, ledger = scene_inputs[scene_id]
        native_path = _regular_file(baseline_root / f"{scene_id}_boxes.pkl", f"baseline prediction {scene_id}")
        native = _load_native(native_path)
        processed, tracks, fused, runtime = _process_scene(
            scene=scene_id,
            scene_index=scene_index,
            candidates=candidates,
            schedule=schedule,
            scene_root=scene_root,
            native_corners=native,
            engine=engine,
            object_memory=object_memory,
            global_row_start=len(candidates_all),
        )
        candidates_all.extend(processed)
        tracks_all.extend(tracks)
        fused_all.extend(fused)
        runtime_all.extend(runtime)
        ledger["native_prediction"] = os.fspath(native_path)
        ledger["native_prediction_sha256"] = _sha256(native_path)
        input_ledgers[scene_id] = ledger
        scene_summaries[scene_id] = {
            "prompt_count": len(processed),
            "accepted_lift_count": sum(bool(row["lift"]["accepted"]) for row in processed),
            "receipt_count": len(tracks),
            "pre_novelty_pass_count": sum(bool(row["pre_novelty_pass"]) for row in tracks),
            "accepted_shadow_count": sum(row["decision"] == "accepted_shadow" for row in tracks),
        }
        print(
            f"[{scene_index + 1}/{len(selected_scenes)}] {scene_id}: "
            f"prompts={len(processed)} receipts={len(tracks)} "
            f"accepted_shadow={scene_summaries[scene_id]['accepted_shadow_count']}",
            flush=True,
        )

    groups = sorted(clip_shadow.OWL_TARGET_GROUP_ALIASES)
    arrays = _arrays(candidates_all, tracks_all, fused_all, groups)
    track_manifest: list[dict[str, Any]] = []
    for row in tracks_all:
        track_manifest.append({key: value for key, value in row.items() if key not in ("fused_aabb_corners", "fused_obb_corners")})
    provider = np.asarray([row["provider_ms"] for row in runtime_all], dtype=np.float64)
    incremental = np.asarray(
        [row["total_incremental_ms"] for row in runtime_all], dtype=np.float64
    )
    manifest = {
        **plan,
        "mode": "shadow",
        "output_inert": True,
        "birth": False,
        "native_mutation_applied": False,
        "gt_access": False,
        "evaluator_access": False,
        "annotation_input_surface": False,
        "annotation_path_argument": False,
        "training": False,
        "target_dataset_training": False,
        "online_learning": False,
        "external_pretraining_frozen": True,
        "past_only_tracking": True,
        "past_only_confirmation": True,
        "native_clip_unchanged": True,
        "old_receipt_membership_consumed": False,
        "old_receipt_decisions_consumed": False,
        "exact_raw_to_owl_key": "(time_ns,instance)",
        "target_alias_matching": "normalized_exact_lookup_only",
        "routing_policy": dict(ROUTING_POLICY),
        "coordinate_frame": "scannet_world",
        "checkpoint": {"path": os.fspath(checkpoint_path), "sha256": _sha256(checkpoint_path), "bytes": checkpoint_path.stat().st_size},
        "runner_source": {"path": os.fspath(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
        "receipt_scene_ledger": {"path": os.fspath(receipt_manifest_path.resolve()), "sha256": _sha256(receipt_manifest_path)},
        "scene_list": {"path": os.fspath(scene_list_path.resolve()), "sha256": _sha256(scene_list_path)},
        "inputs": input_ledgers,
        "scenes": _canonical_scene_receipts(selected_scenes, tracks_all),
        "scene_summaries": scene_summaries,
        "lifted_row_count": len(candidates_all),
        "accepted_lifted_row_count": sum(bool(row["lift"]["accepted"]) for row in candidates_all),
        "receipt_count": len(tracks_all),
        "pre_novelty_pass_count": sum(bool(row["pre_novelty_pass"]) for row in tracks_all),
        "accepted_shadow_count": sum(row["decision"] == "accepted_shadow" for row in tracks_all),
        "decision_counts": {
            reason: sum(row["decision"] == reason for row in tracks_all)
            for reason in sorted({str(row["decision"]) for row in tracks_all})
        },
        "tracks": track_manifest,
        "runtime": {
            **engine.runtime_metadata(),
            "measured_frame_count": len(runtime_all),
            "provider_sum_seconds": float(provider.sum() / 1000.0),
            "provider_mean_ms": float(provider.mean()) if len(provider) else 0.0,
            "provider_p50_ms": float(np.quantile(provider, 0.50)) if len(provider) else 0.0,
            "provider_p95_ms": float(np.quantile(provider, 0.95)) if len(provider) else 0.0,
            "incremental_total_mean_ms": float(incremental.mean())
            if len(incremental)
            else 0.0,
            "incremental_total_p50_ms": float(np.quantile(incremental, 0.50))
            if len(incremental)
            else 0.0,
            "incremental_total_p95_ms": float(np.quantile(incremental, 0.95))
            if len(incremental)
            else 0.0,
            "incremental_runtime_gate_ms": 200.0,
            "incremental_runtime_gate_pass": bool(len(incremental))
            and float(np.quantile(incremental, 0.95)) < 200.0,
        },
        "npz_file": OUTPUT_NPZ,
        "npz_arrays": {name: {"dtype": array.dtype.str, "shape": list(array.shape), "sha256": _hash_array(array)} for name, array in arrays.items()},
        "conclusion_guardrail": "This no-GT shadow exports capacity and geometry only; AP requires a separately frozen active materializer.",
    }
    _publish(output_root, arrays, manifest)
    print(f"Saved: {output_root.resolve()}", flush=True)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run target-first frozen MobileSAM mask-lift full100 shadow")
    parser.add_argument("--receipt-manifest", type=Path, default=REPOSITORY_ROOT / "results/scannet_cbest_raw_boxer_past3_birth_v2_m50_score05/RAW_BOXER_PAST3_BIRTH_FULL100.json")
    parser.add_argument("--raw-log-root", type=Path, default=REPOSITORY_ROOT / "logs/scannet_raw_boxer_full100_score05_v1")
    parser.add_argument("--schedule-root", type=Path, default=Path("/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/cache/cutr_proposals/scannet-score05-gap25-postfilter-v2"))
    parser.add_argument("--scene-root", type=Path, default=REPOSITORY_ROOT / "upstream_clean/scannet_readme_frames")
    parser.add_argument("--scene-list", type=Path, default=REPOSITORY_ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt")
    parser.add_argument("--baseline-root", type=Path, default=REPOSITORY_ROOT / "results/scannet_t05_boxer_replay_active_score05")
    parser.add_argument("--checkpoint", type=Path, default=s3a.MOBILESAM_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=REPOSITORY_ROOT / "logs/scannet_target_first_mobilesam_masklift_full100_score05")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-scene-count", type=int, default=100)
    parser.add_argument("--scene")
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    run_shadow(
        receipt_manifest_path=args.receipt_manifest,
        raw_log_root=args.raw_log_root,
        schedule_root=args.schedule_root,
        scene_root=args.scene_root,
        scene_list_path=args.scene_list,
        baseline_root=args.baseline_root,
        checkpoint=args.checkpoint,
        output_root=args.output_root,
        device=args.device,
        expected_scene_count=args.expected_scene_count,
        scene=args.scene,
        max_scenes=args.max_scenes,
        plan_only=args.plan_only,
    )


if __name__ == "__main__":
    main()
