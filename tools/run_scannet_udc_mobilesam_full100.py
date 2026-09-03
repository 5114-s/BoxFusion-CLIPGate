#!/usr/bin/env python3
"""Run the no-GT UDC + frozen MobileSAM causal shadow on ScanNet full100.

This executable intentionally has no annotation, evaluator, or terminal-native
prediction input.  At each sealed gap-25 keyframe it reads the current CuTR-v2
2D boxes, constructs deterministic unexplained-depth prompts, applies frozen
box-prompted MobileSAM, lifts accepted masks through the current RGB-D/pose,
and confirms the first three observations of a past-only geometric track.

The output is a shadow receipt.  It never writes prediction pickles and it does
not perform the future-aware terminal novelty step used by the separate active
materializer.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))
TOOLS_ROOT = REPOSITORY_ROOT / "tools"
if os.fspath(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(TOOLS_ROOT))

from boxfusion import target_masklift as target_core  # noqa: E402
from boxfusion import udc_mobilesam as udc_core  # noqa: E402
import run_scannet_s3a_boxer_mobilesam_masklift_shadow as s3a  # noqa: E402


SCHEMA = "boxfusion.scannet_udc_mobilesam_full100.v1"
OUTPUT_JSON = "UDC_MOBILESAM_FULL100.json"
EXPECTED_CACHE_SCHEMA = "boxfusion.cutr_postfilter_cache.v2"
EXPECTED_CHECKPOINT_SHA256 = (
    "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f"
)

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
TOP_K_PER_FRAME = 2
VOXEL_SIZE_M = 0.05

MASK_MIN_PREDICTED_IOU = 0.80
MASK_MIN_PIXELS = 200
MASK_MAX_PIXELS = int(0.40 * IMAGE_HEIGHT * IMAGE_WIDTH)
MASK_MIN_SEED_COVERAGE = 0.50
PROMPT_MIN_SIDE_PX = 16.0
PROMPT_MIN_AREA_PX = 400.0
PROMPT_MAX_AREA_PX = float(MASK_MAX_PIXELS)
PROMPT_MAX_ASPECT = 6.0
RAW_NEARBY_CHEBYSHEV_RADIUS = 2
MIN_ACCEPTED_VIEW_VOXELS = 16

RAW_MATCH_CENTER_M = 0.50
RAW_MATCH_IOU = 0.05
RAW_MATCH_CONTAINMENT = 0.30
RAW_TRACK_TTL_SOURCE_FRAMES = 250
RAW_TRACK_CAP = 128
RAW_TRACK_VOXEL_CAP = 4096

CONFIRM_POLICY: Mapping[str, float | int | str | bool] = {
    "association": "raw_track_identity_first3",
    "secondary_target_masklift_tracker": False,
    "views": 3,
    "min_frame_span": 50,
    "min_camera_baseline_m": 0.15,
    "min_view_ray_span_deg": 8.0,
    "min_predicted_iou": 0.80,
    "min_median_predicted_iou": 0.85,
    "min_median_pairwise_lifted_iou": 0.15,
    "max_pairwise_lifted_center_distance_m": 0.40,
    "min_fused_voxels": 32,
    "min_fused_extent_m": 0.05,
    "max_fused_extent_m": 2.50,
    "max_fused_diagonal_m": 3.00,
    "min_fused_volume_m3": 0.001,
    "max_fused_volume_m3": 4.00,
}


class UDCRunnerError(RuntimeError):
    """Raised when a sealed input or no-GT execution contract is violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_array(value: object) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise UDCRunnerError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    source = _regular_file(path, label)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UDCRunnerError(f"invalid {label}: {source}") from error
    if not isinstance(value, dict):
        raise UDCRunnerError(f"{label} must contain a JSON object: {source}")
    return value


def _scene_order(
    scene_list_path: Path,
    expected_scene_count: int,
    scene: str | None,
    max_scenes: int | None,
) -> tuple[list[str], list[str]]:
    rows = [
        line.strip()
        for line in _regular_file(scene_list_path, "official scene list")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(rows) != expected_scene_count or len(set(rows)) != len(rows):
        raise UDCRunnerError(
            f"expected {expected_scene_count} unique scenes, found {len(rows)}"
        )
    if scene is not None:
        if scene not in rows:
            raise UDCRunnerError(f"requested scene is outside official list: {scene}")
        selected = [scene]
    else:
        selected = list(rows)
    if max_scenes is not None:
        if max_scenes < 1:
            raise UDCRunnerError("max-scenes must be positive")
        selected = selected[:max_scenes]
    return rows, selected


def _read_schedule(schedule_root: Path, scene: str) -> dict[str, Any]:
    path = _regular_file(schedule_root / scene / "manifest.json", f"CuTR schedule {scene}")
    manifest = _read_json(path, f"CuTR schedule {scene}")
    frames = manifest.get("recorded_frame_ids")
    records = manifest.get("records")
    if (
        not isinstance(frames, list)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in frames)
        or frames != sorted(set(frames))
        or not isinstance(records, list)
        or len(records) != len(frames)
        or manifest.get("record_count") != len(frames)
    ):
        raise UDCRunnerError(f"invalid sealed CuTR schedule for {scene}")
    by_frame: dict[int, dict[str, Any]] = {}
    for expected, record in zip(frames, records):
        if not isinstance(record, dict) or record.get("frame_id") != expected:
            raise UDCRunnerError(f"CuTR record order differs for {scene}")
        count = record.get("count")
        digest = record.get("sha256")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise UDCRunnerError(f"invalid CuTR record receipt for {scene}/{expected}")
        by_frame[expected] = record
    if manifest.get("proposal_count") != sum(int(row["count"]) for row in records):
        raise UDCRunnerError(f"CuTR proposal total differs for {scene}")
    return {
        "path": path,
        "sha256": _sha256(path),
        "manifest": manifest,
        "frames": tuple(frames),
        "records": by_frame,
    }


def _load_intrinsic(scene_root: Path, scene: str) -> tuple[Path, np.ndarray]:
    path = _regular_file(
        scene_root / scene / "frames/intrinsic/intrinsic_depth.txt",
        f"depth intrinsic {scene}",
    )
    try:
        matrix = np.loadtxt(path, dtype=np.float64)
    except (OSError, ValueError) as error:
        raise UDCRunnerError(f"invalid depth intrinsic for {scene}") from error
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or matrix[0, 0] <= 0.0
        or matrix[1, 1] <= 0.0
        or abs(float(np.linalg.det(matrix[:3, :3]))) <= 1e-12
    ):
        raise UDCRunnerError(f"invalid depth intrinsic for {scene}")
    return path, matrix[:3, :3].copy()


def _valid_pose(value: np.ndarray) -> bool:
    return bool(
        value.shape == (4, 4)
        and np.isfinite(value).all()
        and np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6, rtol=0.0)
        and np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-3, rtol=0.0)
        and math.isclose(float(np.linalg.det(value[:3, :3])), 1.0, abs_tol=1e-3)
    )


def _read_pose(path: Path) -> np.ndarray | None:
    try:
        value = np.loadtxt(_regular_file(path, "pose"), dtype=np.float64)
    except (OSError, ValueError, UDCRunnerError):
        return None
    return value if _valid_pose(value) else None


def _decode_rgb_depth(
    scene_root: Path, scene: str, frame_id: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Path]]:
    try:
        import cv2
    except ImportError as error:
        raise UDCRunnerError("OpenCV is unavailable") from error
    base = scene_root / scene / "frames"
    paths = {
        "rgb": _regular_file(base / "color" / f"{frame_id}.jpg", "RGB frame"),
        "depth": _regular_file(base / "depth" / f"{frame_id}.png", "depth frame"),
        "pose": _regular_file(base / "pose" / f"{frame_id}.txt", "pose frame"),
    }
    bgr = cv2.imread(os.fspath(paths["rgb"]), cv2.IMREAD_COLOR)
    depth = cv2.imread(os.fspath(paths["depth"]), cv2.IMREAD_UNCHANGED)
    if bgr is None or depth is None:
        raise UDCRunnerError(f"could not decode RGB-D {scene}/{frame_id}")
    if depth.shape != (IMAGE_HEIGHT, IMAGE_WIDTH) or depth.dtype != np.uint16:
        raise UDCRunnerError(f"depth must be uint16 [480,640]: {scene}/{frame_id}")
    bgr = cv2.resize(bgr, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb), np.ascontiguousarray(depth), paths


def _load_cutr_boxes(
    schedule_root: Path,
    scene: str,
    frame_id: int,
    record: Mapping[str, Any],
) -> tuple[Path, np.ndarray]:
    path = _regular_file(
        schedule_root / scene / f"frame_{frame_id:06d}.pt",
        f"CuTR cache {scene}/{frame_id}",
    )
    actual_hash = _sha256(path)
    if actual_hash != record.get("sha256"):
        raise UDCRunnerError(f"CuTR cache hash changed for {scene}/{frame_id}")
    try:
        import torch

        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except Exception as error:
        raise UDCRunnerError(f"could not load CuTR cache {scene}/{frame_id}") from error
    if not isinstance(payload, dict) or payload.get("schema") != EXPECTED_CACHE_SCHEMA:
        raise UDCRunnerError(f"CuTR cache schema differs for {scene}/{frame_id}")
    fields = payload.get("fields")
    if not isinstance(fields, dict) or "pred_boxes" not in fields:
        raise UDCRunnerError(f"CuTR pred_boxes missing for {scene}/{frame_id}")
    raw = fields["pred_boxes"]
    try:
        boxes = np.asarray(raw.detach().cpu().numpy(), dtype=np.float32)
    except AttributeError:
        boxes = np.asarray(raw, dtype=np.float32)
    if (
        boxes.ndim != 2
        or boxes.shape[1:] != (4,)
        or not np.isfinite(boxes).all()
        or len(boxes) != int(record["count"])
        or payload.get("count") != len(boxes)
    ):
        raise UDCRunnerError(f"invalid CuTR boxes for {scene}/{frame_id}")
    # CuTR should have emitted proper half-open boxes; malformed rows fail
    # closed instead of being silently repaired or thresholded a second time.
    if len(boxes) and np.any((boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])):
        raise UDCRunnerError(f"degenerate CuTR boxes for {scene}/{frame_id}")
    return path, boxes


def _expanded_prompt_box(box_xyxy: object) -> tuple[np.ndarray, str | None]:
    box = np.asarray(box_xyxy, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        return np.zeros(4, dtype=np.float32), "invalid_prompt"
    width = float(box[2] - box[0])
    height = float(box[3] - box[1])
    if width <= 0.0 or height <= 0.0:
        return np.zeros(4, dtype=np.float32), "invalid_prompt"
    expand_x = max(8.0, 0.10 * width)
    expand_y = max(8.0, 0.10 * height)
    value = np.asarray(
        [
            max(0.0, box[0] - expand_x),
            max(0.0, box[1] - expand_y),
            min(float(IMAGE_WIDTH), box[2] + expand_x + 1.0),
            min(float(IMAGE_HEIGHT), box[3] + expand_y + 1.0),
        ],
        dtype=np.float32,
    )
    side_x, side_y = float(value[2] - value[0]), float(value[3] - value[1])
    area = side_x * side_y
    aspect = max(side_x / max(side_y, 1e-12), side_y / max(side_x, 1e-12))
    if min(side_x, side_y) < PROMPT_MIN_SIDE_PX:
        return value, "prompt_side"
    if area < PROMPT_MIN_AREA_PX or area > PROMPT_MAX_AREA_PX:
        return value, "prompt_area"
    if aspect > PROMPT_MAX_ASPECT:
        return value, "prompt_aspect"
    return value, None


def _source_pixels(prompt: Any) -> np.ndarray:
    for name in ("source_pixels_yx", "pixel_indices_yx", "pixels_yx"):
        if hasattr(prompt, name):
            value = np.asarray(getattr(prompt, name), dtype=np.int64)
            break
    else:
        raise UDCRunnerError("UDC core prompt does not expose source_pixels_yx")
    if value.ndim != 2 or value.shape[1:] != (2,) or not len(value):
        raise UDCRunnerError("UDC source pixels must be nonempty [N,2]")
    if (
        np.any(value[:, 0] < 0)
        or np.any(value[:, 0] >= IMAGE_HEIGHT)
        or np.any(value[:, 1] < 0)
        or np.any(value[:, 1] >= IMAGE_WIDTH)
    ):
        raise UDCRunnerError("UDC source pixels fall outside the image")
    return value


def _source_voxels(
    prompt: Any,
    source_pixels_yx: np.ndarray,
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    pose: np.ndarray,
) -> np.ndarray:
    if hasattr(prompt, "voxel_keys"):
        value = np.asarray(prompt.voxel_keys, dtype=np.int64)
        if value.ndim == 2 and value.shape[1:] == (3,) and len(value):
            return np.unique(value, axis=0)
    y = source_pixels_yx[:, 0]
    x = source_pixels_yx[:, 1]
    z = depth_m[y, x]
    pixels = np.column_stack((x, y, np.ones(len(x), dtype=np.float64)))
    rays = pixels @ np.linalg.inv(intrinsic).T
    rays /= rays[:, 2:3]
    camera = rays * z[:, None]
    world = camera @ pose[:3, :3].T + pose[:3, 3]
    return np.unique(np.floor(world / VOXEL_SIZE_M).astype(np.int64), axis=0)


def _rows_present(queries: np.ndarray, rows: np.ndarray) -> np.ndarray:
    dtype = np.dtype([("x", "<i8"), ("y", "<i8"), ("z", "<i8")])
    needles = np.ascontiguousarray(queries, dtype=np.int64).view(dtype).reshape(-1)
    haystack = np.ascontiguousarray(rows, dtype=np.int64).view(dtype).reshape(-1)
    positions = np.searchsorted(haystack, needles)
    output = positions < len(haystack)
    valid = np.flatnonzero(output)
    if len(valid):
        output[valid] = haystack[positions[valid]] == needles[valid]
    return output


def _near_component_points(
    points_world: object,
    component_voxels: object,
    radius: int = RAW_NEARBY_CHEBYSHEV_RADIUS,
) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float32)
    voxels = np.unique(np.asarray(component_voxels, dtype=np.int64), axis=0)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise UDCRunnerError("lifted points must be finite [N,3]")
    if voxels.ndim != 2 or voxels.shape[1:] != (3,):
        raise UDCRunnerError("component voxels must have shape [N,3]")
    if not len(points) or not len(voxels):
        return np.empty((0, 3), dtype=np.float32)
    keys = np.floor(points.astype(np.float64) / VOXEL_SIZE_M).astype(np.int64)
    keep = np.zeros(len(points), dtype=bool)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                pending = np.flatnonzero(~keep)
                if not len(pending):
                    break
                shifted = keys[pending] + np.asarray([dx, dy, dz], dtype=np.int64)
                keep[pending] = _rows_present(shifted, voxels)
    return np.ascontiguousarray(points[keep])


def _mask_gate(
    mask: object, predicted_iou: float, source_pixels_yx: np.ndarray
) -> tuple[dict[str, Any], str | None]:
    value = np.asarray(mask, dtype=bool)
    if value.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
        raise UDCRunnerError("MobileSAM mask shape differs")
    pixels = int(np.count_nonzero(value))
    seed_count = int(np.count_nonzero(value[source_pixels_yx[:, 0], source_pixels_yx[:, 1]]))
    coverage = seed_count / len(source_pixels_yx)
    metrics = {
        "predicted_iou": float(predicted_iou),
        "mask_pixel_count": pixels,
        "seed_pixel_count": len(source_pixels_yx),
        "covered_seed_pixel_count": seed_count,
        "seed_coverage": float(coverage),
    }
    if (
        not math.isfinite(float(predicted_iou))
        or predicted_iou < MASK_MIN_PREDICTED_IOU
        or predicted_iou > 1.0
    ):
        return metrics, "mobilesam_predicted_iou"
    if pixels < MASK_MIN_PIXELS or pixels > MASK_MAX_PIXELS:
        return metrics, "mobilesam_mask_area"
    if coverage < MASK_MIN_SEED_COVERAGE:
        return metrics, "mobilesam_seed_coverage"
    return metrics, None


def _clip_mask_to_prompt(mask: object, box_xyxy: object) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    box = np.asarray(box_xyxy, dtype=np.float64)
    if value.shape != (IMAGE_HEIGHT, IMAGE_WIDTH) or box.shape != (4,):
        raise UDCRunnerError("mask/prompt clipping input differs")
    x1 = max(0, min(IMAGE_WIDTH, int(math.floor(float(box[0])))))
    y1 = max(0, min(IMAGE_HEIGHT, int(math.floor(float(box[1])))))
    x2 = max(0, min(IMAGE_WIDTH, int(math.ceil(float(box[2])))))
    y2 = max(0, min(IMAGE_HEIGHT, int(math.ceil(float(box[3])))))
    output = np.zeros_like(value)
    if x2 > x1 and y2 > y1:
        output[y1:y2, x1:x2] = value[y1:y2, x1:x2]
    return output


def _aabb_overlap(
    left_lower: object, left_upper: object, right_lower: object, right_upper: object
) -> tuple[float, float, float]:
    overlap = target_core.aabb_overlap(left_lower, left_upper, right_lower, right_upper)
    return float(overlap.iou), float(overlap.left_containment), float(overlap.right_containment)


@dataclass
class _RawTrack:
    track_id: int
    first_frame: int
    last_frame: int
    observations: int
    lower: np.ndarray
    upper: np.ndarray
    voxels: np.ndarray
    confirmed: bool = False


class _RawTrackManager:
    """Bounded deterministic class-agnostic component tracker."""

    def __init__(self) -> None:
        self.tracks: dict[int, _RawTrack] = {}
        self.next_track_id = 0
        self.last_frame: int | None = None
        self.expired_count = 0
        self.capacity_eviction_count = 0

    def update(
        self, frame_id: int, prompts: Sequence[Any], voxel_rows: Sequence[np.ndarray]
    ) -> tuple[list[int], dict[str, Any]]:
        if self.last_frame is not None and frame_id <= self.last_frame:
            raise UDCRunnerError("raw tracker frames must be strictly increasing")
        expired = sorted(
            key
            for key, track in self.tracks.items()
            if frame_id - track.last_frame > RAW_TRACK_TTL_SOURCE_FRAMES
        )
        for key in expired:
            del self.tracks[key]
        self.expired_count += len(expired)
        before = len(self.tracks)
        edges: list[tuple[float, float, float, int, int]] = []
        for prompt_index, prompt in enumerate(prompts):
            lower = np.asarray(prompt.world_q02, dtype=np.float64)
            upper = np.asarray(prompt.world_q98, dtype=np.float64)
            center = 0.5 * (lower + upper)
            for track_id, track in sorted(self.tracks.items()):
                iou, left_in_right, right_in_left = _aabb_overlap(
                    lower, upper, track.lower, track.upper
                )
                containment = max(left_in_right, right_in_left)
                distance = float(np.linalg.norm(center - 0.5 * (track.lower + track.upper)))
                if distance <= RAW_MATCH_CENTER_M and (
                    iou >= RAW_MATCH_IOU or containment >= RAW_MATCH_CONTAINMENT
                ):
                    edges.append((-iou, -containment, distance, track_id, prompt_index))
        edges.sort()
        assigned: dict[int, int] = {}
        used_tracks: set[int] = set()
        for _niou, _ncontain, _distance, track_id, prompt_index in edges:
            if track_id in used_tracks or prompt_index in assigned:
                continue
            used_tracks.add(track_id)
            assigned[prompt_index] = track_id
        output: list[int] = []
        created = 0
        for prompt_index, (prompt, new_voxels) in enumerate(zip(prompts, voxel_rows)):
            bounded_new_voxels = _bounded_track_voxels(new_voxels)
            track_id = assigned.get(prompt_index)
            if track_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1
                created += 1
                track = _RawTrack(
                    track_id=track_id,
                    first_frame=frame_id,
                    last_frame=frame_id,
                    observations=1,
                    lower=np.asarray(prompt.world_q02, dtype=np.float64).copy(),
                    upper=np.asarray(prompt.world_q98, dtype=np.float64).copy(),
                    voxels=bounded_new_voxels,
                )
                self.tracks[track_id] = track
            else:
                track = self.tracks[track_id]
                combined = _bounded_track_voxels(
                    np.concatenate((track.voxels, bounded_new_voxels), axis=0)
                )
                track.last_frame = frame_id
                track.observations += 1
                track.lower = np.asarray(prompt.world_q02, dtype=np.float64).copy()
                track.upper = np.asarray(prompt.world_q98, dtype=np.float64).copy()
                track.voxels = combined
            output.append(track_id)
        while len(self.tracks) > RAW_TRACK_CAP:
            evict = min(
                self.tracks,
                key=lambda key: (self.tracks[key].last_frame, key),
            )
            del self.tracks[evict]
            self.capacity_eviction_count += 1
        self.last_frame = frame_id
        return output, {
            "state_before_count": before,
            "state_after_count": len(self.tracks),
            "expired_track_ids": expired,
            "created_count": created,
            "assigned_track_ids": output,
        }

    def mark_confirmed(self, track_id: int) -> None:
        if track_id in self.tracks:
            self.tracks[track_id].confirmed = True


def _bounded_track_voxels(value: object) -> np.ndarray:
    """Return lexicographically sorted unique raw-track voxels under the cap."""

    voxels = np.asarray(value, dtype=np.int64)
    if voxels.ndim != 2 or voxels.shape[1:] != (3,):
        raise UDCRunnerError("raw track voxels must have shape [N,3]")
    unique = np.unique(voxels, axis=0)
    return np.ascontiguousarray(unique[:RAW_TRACK_VOXEL_CAP])


def _append_raw_track_observation(
    histories: dict[int, list[dict[str, Any]]], observation: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """Accumulate exactly the first three accepted lifts for one raw identity."""

    raw_track_id = int(observation["raw_track_id"])
    frame_id = int(observation["frame_id"])
    rows = histories.setdefault(raw_track_id, [])
    if rows and frame_id <= int(rows[-1]["frame_id"]):
        raise UDCRunnerError(
            "raw-track accepted-lift frames must be strictly increasing"
        )
    if len(rows) >= 3:
        raise UDCRunnerError("confirmed raw track received an extra observation")
    rows.append(observation)
    if len(rows) == 3:
        return rows[0], rows[1], rows[2]
    return None


def _pairwise_lift_metrics(observations: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    values_iou: list[float] = []
    values_distance: list[float] = []
    for left in range(len(observations)):
        for right in range(left + 1, len(observations)):
            one, two = observations[left], observations[right]
            iou, _, _ = _aabb_overlap(one["lower"], one["upper"], two["lower"], two["upper"])
            values_iou.append(iou)
            values_distance.append(
                float(
                    np.linalg.norm(
                        0.5 * (one["lower"] + one["upper"])
                        - 0.5 * (two["lower"] + two["upper"])
                    )
                )
            )
    return {
        "median_pairwise_iou": float(np.median(values_iou)),
        "max_pairwise_center_distance_m": float(max(values_distance)),
    }


def _view_diversity(camera_centers: np.ndarray, object_center: np.ndarray) -> tuple[float, float]:
    baselines = np.linalg.norm(camera_centers[:, None] - camera_centers[None, :], axis=2)
    rays = object_center[None] - camera_centers
    norms = np.linalg.norm(rays, axis=1)
    if np.any(norms <= 1e-12):
        return float(baselines.max()), 0.0
    unit = rays / norms[:, None]
    angles = np.degrees(np.arccos(np.clip(unit @ unit.T, -1.0, 1.0)))
    return float(baselines.max()), float(angles.max())


def _pca_yaw(points_world: np.ndarray) -> float:
    xy = np.asarray(points_world, dtype=np.float64)[:, :2]
    centered = xy - np.median(xy, axis=0)
    covariance = centered.T @ centered / max(len(centered), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if float(eigenvalues[-1]) <= 1e-12 or (
        float(eigenvalues[-1] - eigenvalues[0]) <= 1e-6 * float(eigenvalues[-1])
    ):
        return 0.0
    vector = eigenvectors[:, -1]
    yaw = math.atan2(float(vector[1]), float(vector[0]))
    # Yaw-only boxes are pi-periodic.  Canonicalize to [-pi/2, pi/2).
    return float((yaw + math.pi / 2.0) % math.pi - math.pi / 2.0)


def _confirm_receipt(
    *,
    scene: str,
    raw_track_id: int,
    tracker_track_id: int,
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(observations) != 3:
        raise UDCRunnerError("confirmation requires exactly first three observations")
    if tracker_track_id != raw_track_id:
        raise UDCRunnerError(
            "confirmation identity must equal the raw UDC track identity"
        )
    points_by_view = [np.asarray(row["points_world"], dtype=np.float32) for row in observations]
    consensus = target_core.fuse_three_view_points(points_by_view)
    points = np.asarray(consensus.supported_points, dtype=np.float64)
    pairwise = _pairwise_lift_metrics(observations)
    frames = [int(row["frame_id"]) for row in observations]
    predicted = np.asarray([row["predicted_iou"] for row in observations], dtype=np.float64)
    supported_voxels = int(consensus.voxel_count)
    if len(points):
        yaw = _pca_yaw(points)
        obb = target_core.robust_yaw_obb(points, yaw_rad=yaw)
        center = np.asarray(obb.center, dtype=np.float64)
        extent = np.asarray(obb.extent, dtype=np.float64)
        corners = np.asarray(obb.corners, dtype=np.float64)
        diagonal = float(np.linalg.norm(extent))
        volume = float(np.prod(extent))
        camera_centers = np.stack([row["camera_center"] for row in observations])
        baseline, ray_span = _view_diversity(camera_centers, center)
    else:
        yaw = diagonal = volume = baseline = ray_span = 0.0
        center = extent = np.zeros(3, dtype=np.float64)
        corners = np.zeros((8, 3), dtype=np.float64)
    gates = {
        "three_distinct_increasing_frames": len(set(frames)) == 3 and frames == sorted(frames),
        "frame_span": frames[-1] - frames[0] >= int(CONFIRM_POLICY["min_frame_span"]),
        "camera_baseline": baseline >= float(CONFIRM_POLICY["min_camera_baseline_m"]),
        "view_ray_span": ray_span >= float(CONFIRM_POLICY["min_view_ray_span_deg"]),
        "minimum_predicted_iou": float(predicted.min()) >= float(CONFIRM_POLICY["min_predicted_iou"]),
        "median_predicted_iou": float(np.median(predicted)) >= float(CONFIRM_POLICY["min_median_predicted_iou"]),
        "lifted_median_pairwise_iou": pairwise["median_pairwise_iou"] >= float(CONFIRM_POLICY["min_median_pairwise_lifted_iou"]),
        "lifted_center_distance": pairwise["max_pairwise_center_distance_m"] <= float(CONFIRM_POLICY["max_pairwise_lifted_center_distance_m"]),
        "fused_supported_voxels": supported_voxels >= int(CONFIRM_POLICY["min_fused_voxels"]),
        "fused_extent_min": bool(len(points)) and float(extent.min()) >= float(CONFIRM_POLICY["min_fused_extent_m"]),
        "fused_extent_max": bool(len(points)) and float(extent.max()) <= float(CONFIRM_POLICY["max_fused_extent_m"]),
        "fused_diagonal": diagonal <= float(CONFIRM_POLICY["max_fused_diagonal_m"]),
        "fused_volume": float(CONFIRM_POLICY["min_fused_volume_m3"]) <= volume <= float(CONFIRM_POLICY["max_fused_volume_m3"]),
    }
    passed = all(gates.values())
    return {
        "scene_id": scene,
        # There is deliberately no second association layer: the confirmation
        # identity is the raw component-track identity.
        "track_id": int(tracker_track_id),
        "raw_track_id": int(raw_track_id),
        "confirmation_tracker_id": int(tracker_track_id),
        "confirmation_frame_id": frames[-1],
        "evidence_frame_ids": frames,
        "evidence_observation_ids": [int(row["observation_id"]) for row in observations],
        "evidence_predicted_iou": predicted.tolist(),
        "mean_predicted_iou": float(predicted.mean()),
        "median_predicted_iou": float(np.median(predicted)),
        "min_predicted_iou": float(predicted.min()),
        "median_pairwise_lifted_aabb_iou": pairwise["median_pairwise_iou"],
        "max_pairwise_lifted_center_distance_m": pairwise["max_pairwise_center_distance_m"],
        "max_camera_baseline_m": baseline,
        "max_view_ray_span_deg": ray_span,
        "supported_voxel_count": supported_voxels,
        "view_supported_voxel_counts": [int(value) for value in consensus.exact_supported_voxel_counts],
        "fused_point_count": len(points),
        "fused_points_sha256": _hash_array(points.astype(np.float32)),
        "fused_obb": {
            "center_world": center.tolist(),
            "extent_xyz": extent.tolist(),
            "yaw_rad": yaw,
            "corners_world": corners.tolist(),
        },
        "gates": gates,
        "pre_novelty_pass": passed,
        "accepted": passed,
        "decision": "pre_novelty_pass" if passed else next(name for name, value in gates.items() if not value),
    }


def _accepted_lift(
    *,
    mask: np.ndarray,
    predicted_iou: float,
    hypothesis_index: int,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    pose: np.ndarray,
    component_voxels: np.ndarray,
    object_memory: Any,
) -> tuple[dict[str, Any], str | None]:
    result = s3a._lift_mask_row(
        mask=mask,
        predicted_iou=predicted_iou,
        hypothesis_index=hypothesis_index,
        depth=depth,
        intrinsic=intrinsic,
        pose=pose,
        object_memory=object_memory,
    )
    if not bool(result["accepted"]):
        return result, f"mask_lift_code_{int(result['abstention_code'])}"
    nearby = _near_component_points(result["points_world"], component_voxels)
    nearby = np.asarray(object_memory.voxel_downsample(nearby, 0.02), dtype=np.float32)
    nearby = np.asarray(object_memory.deterministic_bounded_sample(nearby, 2048), dtype=np.float32)
    voxel_count = len(np.unique(np.floor(nearby / VOXEL_SIZE_M).astype(np.int64), axis=0)) if len(nearby) else 0
    result["near_component_point_count"] = len(nearby)
    result["near_component_voxel_count"] = voxel_count
    result["points_world"] = nearby
    if voxel_count < MIN_ACCEPTED_VIEW_VOXELS:
        result["accepted"] = False
        return result, "fewer_than_16_near_component_voxels"
    center, extent = object_memory.robust_quantile_aabb(
        nearby, lower_quantile=0.02, upper_quantile=0.98, min_points=1, minimum_dimension=0.02
    )
    result["reported_q02_q98_center_world"] = np.asarray(center, dtype=np.float32)
    result["reported_q02_q98_extent_xyz"] = np.asarray(extent, dtype=np.float32)
    return result, None


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    return {
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.quantile(array, 0.50)),
        "p95_ms": float(np.quantile(array, 0.95)),
    }


def _append_runtime_sample(
    *,
    preprocessing: list[float],
    provider_and_lifting: list[float],
    complete: list[float],
    preprocess_ms: float,
    provider_and_lifting_ms: float,
    total_ms: float,
    mobilesam_prompted: bool,
    warmup_excluded: bool,
) -> None:
    """Record measured runtime without zero-valued no-prompt provider samples."""

    if warmup_excluded:
        return
    preprocessing.append(float(preprocess_ms))
    complete.append(float(total_ms))
    if mobilesam_prompted:
        provider_and_lifting.append(float(provider_and_lifting_ms))


def _publish(output_root: Path, manifest: Mapping[str, Any]) -> None:
    output = output_root.resolve()
    if output_root.is_symlink() or output.exists():
        raise UDCRunnerError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage.", dir=output.parent))
    try:
        payload = json.dumps(
            manifest, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
        ) + "\n"
        target = staging / OUTPUT_JSON
        with target.open("x", encoding="ascii") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        output.mkdir()
        os.link(target, output / OUTPUT_JSON)
    except Exception:
        if output.exists():
            shutil.rmtree(output, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _process_scene(
    *,
    scene: str,
    schedule: Mapping[str, Any],
    schedule_root: Path,
    scene_root: Path,
    intrinsic: np.ndarray,
    engine: Any,
    object_memory: Any,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, list[float]],
]:
    raw_tracker = _RawTrackManager()
    raw_observation_histories: dict[int, list[dict[str, Any]]] = {}
    receipts: list[dict[str, Any]] = []
    frames_output: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    previous_pose: np.ndarray | None = None
    next_observation_id = 0
    confirmed_raw: set[int] = set()
    runtime_preprocess: list[float] = []
    runtime_provider_lift: list[float] = []
    runtime_total: list[float] = []
    warmup_discarded = False

    for frame_ordinal, frame_id in enumerate(schedule["frames"]):
        frame_started = time.perf_counter()
        rgb, depth, paths = _decode_rgb_depth(scene_root, scene, frame_id)
        current_pose = _read_pose(paths["pose"])
        pose_forward_filled = current_pose is None and previous_pose is not None
        pose = current_pose if current_pose is not None else previous_pose
        if current_pose is not None:
            previous_pose = current_pose.copy()
        cache_path, boxes = _load_cutr_boxes(
            schedule_root, scene, frame_id, schedule["records"][frame_id]
        )
        frame_ledger: dict[str, Any] = {
            "frame_id": frame_id,
            "frame_ordinal": frame_ordinal,
            "rgb_sha256": _sha256(paths["rgb"]),
            "depth_sha256": _sha256(paths["depth"]),
            "pose_sha256": _sha256(paths["pose"]),
            "cutr_cache_sha256": _sha256(cache_path),
            "cutr_cache_input_signature": schedule["records"][frame_id].get("input_signature"),
            "cutr_box_count": len(boxes),
            "pose_forward_filled_from_past": pose_forward_filled,
            "pose_valid_or_forward_filled": pose is not None,
            "prompts": [],
        }
        if pose is None:
            rejection_counts["no_current_or_past_valid_pose"] += 1
            preprocess_ms = (time.perf_counter() - frame_started) * 1000.0
            total_ms = (time.perf_counter() - frame_started) * 1000.0
            frame_ledger.update(
                {
                    "abstention": "no_current_or_past_valid_pose",
                    "raw_tracker": {"state_before_count": len(raw_tracker.tracks), "state_after_count": len(raw_tracker.tracks)},
                    "preprocess_ms": preprocess_ms,
                    "provider_and_lifting_ms": 0.0,
                    "total_incremental_ms": total_ms,
                }
            )
            frames_output.append(frame_ledger)
            _append_runtime_sample(
                preprocessing=runtime_preprocess,
                provider_and_lifting=runtime_provider_lift,
                complete=runtime_total,
                preprocess_ms=preprocess_ms,
                provider_and_lifting_ms=0.0,
                total_ms=total_ms,
                mobilesam_prompted=False,
                warmup_excluded=False,
            )
            continue

        preprocess_started = time.perf_counter()
        depth_m = depth.astype(np.float32) / 1000.0
        try:
            udc = udc_core.generate_residual_box_prompts(
                depth_m=depth_m,
                explained_boxes_xyxy=boxes,
                intrinsics=intrinsic,
                camera_to_world=pose,
            )
        except ValueError as error:
            rejection_counts["udc_invalid_input"] += 1
            preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0
            total_ms = (time.perf_counter() - frame_started) * 1000.0
            frame_ledger.update(
                {
                    "abstention": "udc_invalid_input",
                    "error": str(error),
                    "raw_tracker": {"state_before_count": len(raw_tracker.tracks), "state_after_count": len(raw_tracker.tracks)},
                    "preprocess_ms": preprocess_ms,
                    "provider_and_lifting_ms": 0.0,
                    "total_incremental_ms": total_ms,
                }
            )
            frames_output.append(frame_ledger)
            _append_runtime_sample(
                preprocessing=runtime_preprocess,
                provider_and_lifting=runtime_provider_lift,
                complete=runtime_total,
                preprocess_ms=preprocess_ms,
                provider_and_lifting_ms=0.0,
                total_ms=total_ms,
                mobilesam_prompted=False,
                warmup_excluded=False,
            )
            continue
        prompts = list(udc.prompts)
        source_pixels = [_source_pixels(prompt) for prompt in prompts]
        source_voxels = [
            _source_voxels(prompt, pixels, depth_m, intrinsic, pose)
            for prompt, pixels in zip(prompts, source_pixels)
        ]
        raw_track_ids, raw_state = raw_tracker.update(frame_id, prompts, source_voxels)
        expanded = [_expanded_prompt_box(prompt.box_xyxy) for prompt in prompts]
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0
        provider_started = time.perf_counter()
        valid_positions = [
            index
            for index, (_box, reason) in enumerate(expanded)
            if reason is None and raw_track_ids[index] not in confirmed_raw
        ]
        results: dict[int, tuple[np.ndarray, float, int, Mapping[str, float]]] = {}
        provider_timing = {"provider_ms": 0.0, "encoder_ms": 0.0, "decoder_and_host_mask_ms": 0.0}
        if valid_positions:
            prompt_boxes = np.stack([expanded[index][0] for index in valid_positions]).astype(np.float32)
            masks, predicted_iou, hypotheses, timing = engine.predict(rgb, prompt_boxes)
            provider_timing = {key: float(value) for key, value in timing.items()}
            for batch, position in enumerate(valid_positions):
                results[position] = (
                    np.asarray(masks[batch], dtype=bool),
                    float(predicted_iou[batch]),
                    int(hypotheses[batch]),
                    provider_timing,
                )

        confirmation_assignment_count = 0
        new_confirmation_count = 0
        for index, (prompt, pixels, voxels, raw_track_id) in enumerate(
            zip(prompts, source_pixels, source_voxels, raw_track_ids)
        ):
            prompt_box, prompt_reason = expanded[index]
            row: dict[str, Any] = {
                "rank": int(prompt.rank),
                "component_id": int(prompt.component_id),
                "raw_track_id": int(raw_track_id),
                "component_grid_pixel_count": int(prompt.grid_pixel_count),
                "component_box_xyxy": np.asarray(prompt.box_xyxy).tolist(),
                "prompt_box_xyxy": prompt_box.tolist(),
                "component_voxel_count": len(voxels),
                "source_pixel_count": len(pixels),
            }
            if prompt_reason is not None:
                rejection_counts[prompt_reason] += 1
                row.update({"accepted_lift": False, "decision": prompt_reason})
                frame_ledger["prompts"].append(row)
                continue
            if raw_track_id in confirmed_raw:
                rejection_counts["already_confirmed_raw_track"] += 1
                row.update(
                    {
                        "accepted_lift": False,
                        "decision": "already_confirmed_raw_track",
                    }
                )
                frame_ledger["prompts"].append(row)
                continue
            mask, predicted_iou, hypothesis, _timing = results[index]
            mask = _clip_mask_to_prompt(mask, prompt_box)
            mask_metrics, mask_reason = _mask_gate(mask, predicted_iou, pixels)
            row.update(mask_metrics)
            row["selected_hypothesis_index"] = hypothesis
            if mask_reason is not None:
                rejection_counts[mask_reason] += 1
                row.update({"accepted_lift": False, "decision": mask_reason})
                frame_ledger["prompts"].append(row)
                continue
            lift, lift_reason = _accepted_lift(
                mask=mask,
                predicted_iou=predicted_iou,
                hypothesis_index=hypothesis,
                depth=depth,
                intrinsic=intrinsic,
                pose=pose,
                component_voxels=voxels,
                object_memory=object_memory,
            )
            row.update(
                {
                    "mask_lift_abstention_code": int(lift["abstention_code"]),
                    "near_component_point_count": int(lift.get("near_component_point_count", 0)),
                    "near_component_voxel_count": int(lift.get("near_component_voxel_count", 0)),
                }
            )
            if lift_reason is not None:
                rejection_counts[lift_reason] += 1
                row.update({"accepted_lift": False, "decision": lift_reason})
                frame_ledger["prompts"].append(row)
                continue
            center = np.asarray(lift["reported_q02_q98_center_world"], dtype=np.float64)
            extent = np.asarray(lift["reported_q02_q98_extent_xyz"], dtype=np.float64)
            lower, upper = center - 0.5 * extent, center + 0.5 * extent
            observation_id = next_observation_id
            next_observation_id += 1
            observation = {
                "observation_id": observation_id,
                "raw_track_id": raw_track_id,
                "frame_id": frame_id,
                "predicted_iou": predicted_iou,
                "lower": lower,
                "upper": upper,
                "points_world": np.asarray(lift["points_world"], dtype=np.float32),
                "camera_center": np.asarray(pose[:3, 3], dtype=np.float64),
            }
            first_three = _append_raw_track_observation(
                raw_observation_histories, observation
            )
            confirmation_assignment_count += 1
            row.update(
                {
                    "observation_id": observation_id,
                    "accepted_lift": True,
                    "decision": "accepted_lift",
                    "lift_center_world": center.tolist(),
                    "lift_extent_xyz": extent.tolist(),
                    "lift_points_sha256": _hash_array(lift["points_world"]),
                }
            )
            frame_ledger["prompts"].append(row)
            if first_three is not None:
                receipt = _confirm_receipt(
                    scene=scene,
                    raw_track_id=raw_track_id,
                    tracker_track_id=raw_track_id,
                    observations=first_three,
                )
                receipts.append(receipt)
                confirmed_raw.add(raw_track_id)
                raw_tracker.mark_confirmed(raw_track_id)
                new_confirmation_count += 1

        provider_lift_ms = (time.perf_counter() - provider_started) * 1000.0
        total_ms = (time.perf_counter() - frame_started) * 1000.0
        diagnostics = udc.diagnostics
        frame_ledger.update(
            {
                "abstention": None,
                "udc": {
                    "valid_depth_grid_pixels": int(diagnostics.valid_depth_grid_pixels),
                    "edge_rejected_grid_pixels": int(diagnostics.edge_rejected_grid_pixels),
                    "explained_valid_grid_pixels": int(diagnostics.explained_valid_grid_pixels),
                    "residual_grid_pixels": int(diagnostics.residual_grid_pixels),
                    "component_count": int(diagnostics.component_count),
                    "eligible_component_count": int(diagnostics.eligible_component_count),
                    "selected_component_count": int(diagnostics.selected_component_count),
                    "rejection_counts": dict(diagnostics.rejection_counts),
                },
                "raw_tracker": raw_state,
                "confirmation_assignment_count": confirmation_assignment_count,
                "new_confirmation_count": new_confirmation_count,
                "preprocess_ms": preprocess_ms,
                "provider_ms": provider_timing["provider_ms"],
                "provider_and_lifting_ms": provider_lift_ms,
                "total_incremental_ms": total_ms,
                "runtime_warmup_excluded": bool(valid_positions and not warmup_discarded),
            }
        )
        frames_output.append(frame_ledger)
        # Discard the first prompted frame from the warm runtime distribution;
        # its inference still contributes proposals and remains in the ledger.
        warmup_excluded = bool(valid_positions and not warmup_discarded)
        if warmup_excluded:
            warmup_discarded = True
        _append_runtime_sample(
            preprocessing=runtime_preprocess,
            provider_and_lifting=runtime_provider_lift,
            complete=runtime_total,
            preprocess_ms=preprocess_ms,
            provider_and_lifting_ms=provider_lift_ms,
            total_ms=total_ms,
            mobilesam_prompted=bool(valid_positions),
            warmup_excluded=warmup_excluded,
        )

    scene_summary = {
        "keyframe_count": len(schedule["frames"]),
        "raw_component_count": sum(row.get("udc", {}).get("component_count", 0) for row in frames_output),
        "eligible_component_count": sum(row.get("udc", {}).get("eligible_component_count", 0) for row in frames_output),
        "prompt_count": sum(len(row["prompts"]) for row in frames_output),
        "accepted_mask_lift_count": sum(
            bool(prompt.get("accepted_lift")) for row in frames_output for prompt in row["prompts"]
        ),
        "confirmed_track_count": len(receipts),
        "pre_novelty_pass_count": sum(bool(row["pre_novelty_pass"]) for row in receipts),
        "candidate_count": sum(bool(row["pre_novelty_pass"]) for row in receipts),
        "candidate_track_ids": [int(row["track_id"]) for row in receipts if row["pre_novelty_pass"]],
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "raw_tracker_expired_count": raw_tracker.expired_count,
        "raw_tracker_capacity_eviction_count": raw_tracker.capacity_eviction_count,
        "pose_forward_fill_count": sum(bool(row["pose_forward_filled_from_past"]) for row in frames_output),
        "invalid_input_abstention_count": sum(row["abstention"] is not None for row in frames_output),
        "runtime": {
            "warmup_prompted_frame_excluded": warmup_discarded,
            "measured_keyframe_count": len(runtime_total),
            "udc_preprocessing": _percentiles(runtime_preprocess),
            "mobilesam_and_mask_lifting": _percentiles(runtime_provider_lift),
            "complete_incremental_keyframe": _percentiles(runtime_total),
        },
    }
    return scene_summary, frames_output, receipts, {
        "preprocessing": runtime_preprocess,
        "provider_and_lifting": runtime_provider_lift,
        "complete": runtime_total,
    }


def run_shadow(
    *,
    schedule_root: Path,
    scene_root: Path,
    scene_list_path: Path,
    checkpoint: Path,
    output_root: Path,
    device: str,
    expected_scene_count: int = 100,
    scene: str | None = None,
    max_scenes: int | None = None,
    plan_only: bool = False,
    engine_factory: Callable[[str], Any] = s3a.MobileSAMBoxPromptEngine,
) -> dict[str, Any]:
    full_scenes, selected_scenes = _scene_order(
        scene_list_path, expected_scene_count, scene, max_scenes
    )
    schedules = {scene_id: _read_schedule(schedule_root, scene_id) for scene_id in selected_scenes}
    plan = {
        "schema": SCHEMA,
        "mode": "plan_only" if plan_only else "shadow",
        "scene_count": len(selected_scenes),
        "scene_order": selected_scenes,
        "keyframe_count": sum(len(row["frames"]) for row in schedules.values()),
        "cutr_box_count": sum(
            int(row["manifest"]["proposal_count"]) for row in schedules.values()
        ),
        "top_k_per_frame": TOP_K_PER_FRAME,
        "output_json": OUTPUT_JSON,
    }
    if plan_only:
        print(json.dumps(plan, sort_keys=True), flush=True)
        return plan

    if output_root.exists() or output_root.is_symlink():
        raise UDCRunnerError(f"refusing to overwrite output: {output_root}")
    checkpoint_path = _regular_file(checkpoint, "MobileSAM checkpoint")
    if checkpoint_path != s3a.MOBILESAM_CHECKPOINT.resolve():
        raise UDCRunnerError("MobileSAM checkpoint path differs from frozen engine")
    checkpoint_hash = _sha256(checkpoint_path)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise UDCRunnerError("MobileSAM checkpoint SHA-256 differs")
    object_memory = s3a._load_object_memory_module()
    engine = engine_factory(device)

    scenes_output: list[dict[str, Any]] = []
    input_ledgers: dict[str, Any] = {}
    all_runtime_pre: list[float] = []
    all_runtime_provider: list[float] = []
    all_runtime_total: list[float] = []
    for scene_index, scene_id in enumerate(selected_scenes):
        intrinsic_path, intrinsic = _load_intrinsic(scene_root, scene_id)
        summary, frames, receipts, runtime_samples = _process_scene(
            scene=scene_id,
            schedule=schedules[scene_id],
            schedule_root=schedule_root,
            scene_root=scene_root,
            intrinsic=intrinsic,
            engine=engine,
            object_memory=object_memory,
        )
        scenes_output.append(
            {
                "scene_id": scene_id,
                "summary": summary,
                "frames": frames,
                "confirmations": receipts,
                # The active materializer consumes only fully gated,
                # positive-volume pre-novelty candidates.
                "receipts": [row for row in receipts if row["pre_novelty_pass"]],
            }
        )
        input_ledgers[scene_id] = {
            "schedule_manifest": os.fspath(schedules[scene_id]["path"]),
            "schedule_manifest_sha256": schedules[scene_id]["sha256"],
            "intrinsic": os.fspath(intrinsic_path),
            "intrinsic_sha256": _sha256(intrinsic_path),
        }
        all_runtime_pre.extend(runtime_samples["preprocessing"])
        all_runtime_provider.extend(runtime_samples["provider_and_lifting"])
        all_runtime_total.extend(runtime_samples["complete"])
        print(
            f"[{scene_index + 1}/{len(selected_scenes)}] {scene_id}: "
            f"prompts={summary['prompt_count']} lifts={summary['accepted_mask_lift_count']} "
            f"confirmed={summary['confirmed_track_count']} pre_novel={summary['pre_novelty_pass_count']}",
            flush=True,
        )

    totals = {
        "keyframe_count": sum(row["summary"]["keyframe_count"] for row in scenes_output),
        "raw_component_count": sum(row["summary"]["raw_component_count"] for row in scenes_output),
        "eligible_component_count": sum(row["summary"]["eligible_component_count"] for row in scenes_output),
        "prompt_count": sum(row["summary"]["prompt_count"] for row in scenes_output),
        "accepted_mask_lift_count": sum(row["summary"]["accepted_mask_lift_count"] for row in scenes_output),
        "confirmed_pre_novelty_track_count": sum(row["summary"]["pre_novelty_pass_count"] for row in scenes_output),
        "candidate_scene_count": sum(row["summary"]["pre_novelty_pass_count"] > 0 for row in scenes_output),
        "raw_tracker_capacity_eviction_count": sum(row["summary"]["raw_tracker_capacity_eviction_count"] for row in scenes_output),
        "invalid_input_abstention_count": sum(row["summary"]["invalid_input_abstention_count"] for row in scenes_output),
    }
    manifest = {
        **plan,
        "mode": "shadow",
        "contracts": {
            "causal_shadow_generation": True,
            "strict_online_native_novelty": False,
            "terminal_replay_materialization": False,
            "output_inert": True,
            "native_prediction_access": False,
            "native_prediction_mutation": False,
            "gt_access": False,
            "annotation_access": False,
            "evaluator_access": False,
            "training": False,
            "target_dataset_training": False,
            "online_learning": False,
            "external_pretraining_frozen": True,
            "current_frame_cutr_boxes_only": True,
            "past_only_tracking_and_confirmation": True,
            "native_clip_unchanged": True,
        },
        "policy": {
            "udc_core_schema": udc_core.SCHEMA,
            "udc_core_policy": dict(udc_core.POLICY),
            "prompt": {
                "expand_px_min": 8.0,
                "expand_fraction": 0.10,
                "min_side_px": PROMPT_MIN_SIDE_PX,
                "min_area_px": PROMPT_MIN_AREA_PX,
                "max_area_px": PROMPT_MAX_AREA_PX,
                "max_aspect": PROMPT_MAX_ASPECT,
            },
            "mask": {
                "min_predicted_iou": MASK_MIN_PREDICTED_IOU,
                "min_pixels": MASK_MIN_PIXELS,
                "max_pixels": MASK_MAX_PIXELS,
                "min_source_pixel_coverage": MASK_MIN_SEED_COVERAGE,
                "clip_mask_to_expanded_prompt": True,
                "near_component_chebyshev_voxel_radius": RAW_NEARBY_CHEBYSHEV_RADIUS,
                "min_near_component_voxels": MIN_ACCEPTED_VIEW_VOXELS,
            },
            "raw_tracker": {
                "match_center_m": RAW_MATCH_CENTER_M,
                "match_iou": RAW_MATCH_IOU,
                "match_max_directional_containment": RAW_MATCH_CONTAINMENT,
                "ttl_source_frames": RAW_TRACK_TTL_SOURCE_FRAMES,
                "live_track_cap": RAW_TRACK_CAP,
                "voxel_cap_per_track": RAW_TRACK_VOXEL_CAP,
            },
            "confirmation": dict(CONFIRM_POLICY),
        },
        "checkpoint": {
            "path": os.fspath(checkpoint_path),
            "sha256": checkpoint_hash,
            "bytes": checkpoint_path.stat().st_size,
        },
        "sources": {
            "runner": {"path": os.fspath(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
            "udc_core": {"path": os.fspath(Path(udc_core.__file__).resolve()), "sha256": _sha256(Path(udc_core.__file__).resolve())},
            "target_masklift": {"path": os.fspath(Path(target_core.__file__).resolve()), "sha256": _sha256(Path(target_core.__file__).resolve())},
        },
        "scene_list": {"path": os.fspath(scene_list_path.resolve()), "sha256": _sha256(scene_list_path)},
        "inputs": input_ledgers,
        "scenes": scenes_output,
        "totals": totals,
        "capacity_reference": {
            "minimum_confirmed_pre_novelty_tracks": 300,
            "minimum_candidates": 250,
            "minimum_candidate_scenes": 80,
            "pass": totals["confirmed_pre_novelty_track_count"] >= 300
            and totals["confirmed_pre_novelty_track_count"] >= 250
            and totals["candidate_scene_count"] >= 80,
        },
        "runtime": {
            **engine.runtime_metadata(),
            "udc_preprocessing": _percentiles(all_runtime_pre),
            "mobilesam_and_mask_lifting": _percentiles(all_runtime_provider),
            "complete_incremental_keyframe": _percentiles(all_runtime_total),
            "amortized_mean_overhead_per_stream_frame_ms": (
                _percentiles(all_runtime_total)["mean_ms"] / 25.0
            ),
            "reference_limits": {
                "udc_preprocessing_p95_ms": 60.0,
                "complete_mean_ms": 150.0,
                "complete_p95_ms": 250.0,
                "amortized_mean_ms": 6.0,
            },
        },
        "cpu_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "conclusion_guardrail": (
            "This no-GT artifact measures causal proposal capacity and geometry only. "
            "Terminal native novelty and AP belong to the separately frozen materializer."
        ),
    }
    _publish(output_root, manifest)
    print(f"Saved: {(output_root / OUTPUT_JSON).resolve()}", flush=True)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run no-GT UDC + frozen MobileSAM full100 causal shadow"
    )
    parser.add_argument(
        "--schedule-root",
        type=Path,
        default=Path(
            "/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/cache/cutr_proposals/"
            "scannet-score05-gap25-postfilter-v2"
        ),
    )
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=REPOSITORY_ROOT / "upstream_clean/scannet_readme_frames",
    )
    parser.add_argument(
        "--scene-list",
        type=Path,
        default=REPOSITORY_ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt",
    )
    parser.add_argument("--checkpoint", type=Path, default=s3a.MOBILESAM_CHECKPOINT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "logs/scannet_udc_mobilesam_full100_score05",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-scene-count", type=int, default=100)
    parser.add_argument("--scene")
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    run_shadow(
        schedule_root=args.schedule_root,
        scene_root=args.scene_root,
        scene_list_path=args.scene_list,
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
