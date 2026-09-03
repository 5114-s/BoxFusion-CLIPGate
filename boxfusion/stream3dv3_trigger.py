"""Bounded, training-free discovery trigger and FastSAM mask preselector.

This module is intentionally independent from the live BoxFusion entrypoint so
it can be audited without changing an active experiment.  The depth trigger
queries an append-only *past* world-voxel memory, makes the current decision,
and only then admits current evidence through an exact-token commit.  The mask
preselector uses FastSAM's native boxes/confidences for a cheap first pass and
touches full-resolution masks only for a small bounded shortlist.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
import hashlib
import hmac
import math
from typing import Mapping, Sequence

import numpy as np


IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640


@dataclass(frozen=True)
class DepthTriggerConfig:
    sample_stride: int = 8
    voxel_size_m: float = 0.15
    min_depth_m: float = 0.10
    max_depth_m: float = 6.0
    native_expand_px: float = 4.0
    confirmations: int = 2
    tentative_ttl_keyframes: int = 2
    min_persistent_voxels: int = 48
    min_persistent_fraction: float = 0.08
    cooldown_keyframes: int = 4
    burst_keyframes: int = 3
    max_confirmed_voxels: int = 50_000
    max_tentative_voxels: int = 20_000

    def __post_init__(self) -> None:
        integer_positive = (
            "sample_stride",
            "confirmations",
            "tentative_ttl_keyframes",
            "min_persistent_voxels",
            "cooldown_keyframes",
            "max_confirmed_voxels",
            "max_tentative_voxels",
        )
        for name in integer_positive:
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.burst_keyframes, bool) or int(self.burst_keyframes) < 0:
            raise ValueError("burst_keyframes must be a non-negative integer")
        for name in ("voxel_size_m", "min_depth_m", "max_depth_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_depth_m <= self.min_depth_m:
            raise ValueError("depth interval is empty")
        if not math.isfinite(float(self.native_expand_px)) or self.native_expand_px < 0:
            raise ValueError("native_expand_px must be finite and non-negative")
        if not 0.0 <= float(self.min_persistent_fraction) <= 1.0:
            raise ValueError("min_persistent_fraction must lie in [0,1]")


@dataclass(frozen=True)
class DepthTriggerQuery:
    frame_id: int
    frame_ordinal: int
    run_discovery: bool
    reason: str
    sampled_voxels: int
    unknown_voxels: int
    persistent_voxels: int
    persistent_fraction: float
    memory_version_before: int
    token: str


@dataclass(frozen=True)
class _TriggerPending:
    public: DepthTriggerQuery
    confirmed: tuple[tuple[int, int, int], ...]
    tentative: tuple[tuple[tuple[int, int, int], int, int], ...]
    last_run_ordinal: int | None
    burst_remaining: int


def _as_depth(value: object) -> np.ndarray:
    depth = np.asarray(value)
    if depth.shape != (IMAGE_HEIGHT, IMAGE_WIDTH) or depth.dtype.kind not in "iuf":
        raise ValueError("depth_m must be numeric [480,640]")
    result = np.asarray(depth, dtype=np.float64)
    return np.ascontiguousarray(result)


def _as_intrinsics(value: object) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (4, 4):
        matrix = matrix[:3, :3]
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or matrix[0, 0] <= 0.0
        or matrix[1, 1] <= 0.0
    ):
        raise ValueError("intrinsics must be finite [3,3] or [4,4]")
    return np.ascontiguousarray(matrix)


def _as_pose(value: object) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("camera_to_world must be finite [4,4]")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-6):
        raise ValueError("camera_to_world has an invalid homogeneous row")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=5e-3):
        raise ValueError("camera_to_world rotation is not orthonormal")
    return np.ascontiguousarray(pose)


def _as_boxes(value: object) -> np.ndarray:
    boxes = np.asarray(value, dtype=np.float64)
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1:] != (4,) or not np.isfinite(boxes).all():
        raise ValueError("boxes must be finite [N,4]")
    if np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1]):
        raise ValueError("boxes must satisfy x2>x1 and y2>y1")
    return np.ascontiguousarray(boxes)


def _sampled_world_voxels(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    native_boxes: np.ndarray,
    config: DepthTriggerConfig,
) -> tuple[tuple[int, int, int], ...]:
    stride = config.sample_stride
    rows = np.arange(stride // 2, IMAGE_HEIGHT, stride, dtype=np.int64)
    cols = np.arange(stride // 2, IMAGE_WIDTH, stride, dtype=np.int64)
    grid_y, grid_x = np.meshgrid(rows, cols, indexing="ij")
    sampled_depth = depth[grid_y, grid_x]
    valid = (
        np.isfinite(sampled_depth)
        & (sampled_depth >= config.min_depth_m)
        & (sampled_depth <= config.max_depth_m)
    )
    if len(native_boxes):
        explained = np.zeros(valid.shape, dtype=np.bool_)
        margin = float(config.native_expand_px)
        for x1, y1, x2, y2 in native_boxes:
            explained |= (
                (grid_x >= x1 - margin)
                & (grid_x <= x2 + margin)
                & (grid_y >= y1 - margin)
                & (grid_y <= y2 + margin)
            )
        valid &= ~explained
    if not np.any(valid):
        return ()
    x = grid_x[valid].astype(np.float64)
    y = grid_y[valid].astype(np.float64)
    z = sampled_depth[valid]
    camera = np.column_stack(
        (
            (x - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (y - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        )
    )
    world = camera @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    scaled = world / config.voxel_size_m
    if not np.isfinite(scaled).all():
        raise ValueError("sampled world points are non-finite")
    keys = np.unique(np.floor(scaled).astype(np.int64), axis=0)
    return tuple((int(row[0]), int(row[1]), int(row[2])) for row in keys)


def _trigger_token(
    *,
    frame_id: int,
    frame_ordinal: int,
    run: bool,
    reason: str,
    sampled: int,
    unknown: int,
    persistent: int,
    fraction: float,
    version: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray([frame_id, frame_ordinal], dtype="<i8").tobytes())
    digest.update(np.asarray([run], dtype=np.uint8).tobytes())
    digest.update(reason.encode("ascii"))
    digest.update(np.asarray([sampled, unknown, persistent, version], dtype="<i8").tobytes())
    digest.update(np.asarray([fraction], dtype="<f8").tobytes())
    return digest.hexdigest()


class DepthResidualEventGate:
    """Past-only persistent world-voxel discovery trigger."""

    def __init__(self, config: DepthTriggerConfig | None = None) -> None:
        self.config = config or DepthTriggerConfig()
        self._confirmed: OrderedDict[tuple[int, int, int], None] = OrderedDict()
        self._tentative: dict[tuple[int, int, int], tuple[int, int]] = {}
        self._pending: _TriggerPending | None = None
        self._last_ordinal = -1
        self._last_run_ordinal: int | None = None
        self._burst_remaining = 0
        self._memory_version = 0
        self._stats: Counter[str] = Counter()

    def query(
        self,
        *,
        frame_id: int,
        frame_ordinal: int,
        depth_m: object,
        intrinsics: object,
        camera_to_world: object,
        native_boxes_xyxy: object,
    ) -> DepthTriggerQuery:
        if self._pending is not None:
            raise RuntimeError("previous depth-trigger query was not committed")
        frame = int(frame_id)
        ordinal = int(frame_ordinal)
        if frame < 0 or ordinal <= self._last_ordinal:
            raise ValueError("depth-trigger frame ordinal must increase")
        depth = _as_depth(depth_m)
        K = _as_intrinsics(intrinsics)
        pose = _as_pose(camera_to_world)
        boxes = _as_boxes(native_boxes_xyxy)
        sampled_keys = _sampled_world_voxels(depth, K, pose, boxes, self.config)
        sampled_set = set(sampled_keys)
        confirmed_set = set(self._confirmed)
        unknown_set = sampled_set.difference(confirmed_set)

        tentative = {
            key: value
            for key, value in self._tentative.items()
            if ordinal - value[1] <= self.config.tentative_ttl_keyframes
        }
        for key in sorted(unknown_set):
            previous = tentative.get(key)
            hits = previous[0] + 1 if previous is not None and previous[1] == ordinal - 1 else 1
            tentative[key] = (hits, ordinal)
        persistent_keys = {
            key for key in unknown_set if tentative[key][0] >= self.config.confirmations
        }
        persistent_fraction = len(persistent_keys) / max(len(unknown_set), 1)

        confirmed = OrderedDict(self._confirmed)
        for key in sorted(persistent_keys):
            confirmed[key] = None
            tentative.pop(key, None)
        while len(confirmed) > self.config.max_confirmed_voxels:
            confirmed.popitem(last=False)
        if len(tentative) > self.config.max_tentative_voxels:
            ranked = sorted(
                tentative.items(), key=lambda item: (-item[1][1], -item[1][0], item[0])
            )[: self.config.max_tentative_voxels]
            tentative = dict(ranked)

        if self._last_run_ordinal is None:
            run, reason = True, "bootstrap"
        elif self._burst_remaining > 0:
            run, reason = True, "bounded_burst"
        else:
            cooldown_ready = (
                ordinal - self._last_run_ordinal >= self.config.cooldown_keyframes
            )
            persistent_ready = (
                len(persistent_keys) >= self.config.min_persistent_voxels
                and persistent_fraction >= self.config.min_persistent_fraction
            )
            run = bool(cooldown_ready and persistent_ready)
            reason = "persistent_novel_depth" if run else (
                "cooldown" if not cooldown_ready else "insufficient_persistence"
            )

        if run and reason in {"bootstrap", "persistent_novel_depth"}:
            burst_remaining = self.config.burst_keyframes
        elif run and reason == "bounded_burst":
            burst_remaining = max(self._burst_remaining - 1, 0)
        else:
            burst_remaining = self._burst_remaining
        last_run = ordinal if run else self._last_run_ordinal
        token = _trigger_token(
            frame_id=frame,
            frame_ordinal=ordinal,
            run=run,
            reason=reason,
            sampled=len(sampled_keys),
            unknown=len(unknown_set),
            persistent=len(persistent_keys),
            fraction=persistent_fraction,
            version=self._memory_version,
        )
        public = DepthTriggerQuery(
            frame_id=frame,
            frame_ordinal=ordinal,
            run_discovery=run,
            reason=reason,
            sampled_voxels=len(sampled_keys),
            unknown_voxels=len(unknown_set),
            persistent_voxels=len(persistent_keys),
            persistent_fraction=float(persistent_fraction),
            memory_version_before=self._memory_version,
            token=token,
        )
        self._pending = _TriggerPending(
            public=public,
            confirmed=tuple(confirmed),
            tentative=tuple(
                (key, value[0], value[1]) for key, value in sorted(tentative.items())
            ),
            last_run_ordinal=last_run,
            burst_remaining=burst_remaining,
        )
        return public

    def commit(self, query: DepthTriggerQuery, *, token: str | None = None) -> None:
        pending = self._pending
        if pending is None or query is not pending.public:
            raise ValueError("commit requires the exact pending trigger query")
        supplied = query.token if token is None else token
        if not hmac.compare_digest(str(supplied), query.token):
            raise ValueError("trigger token mismatch")
        if query.memory_version_before != self._memory_version:
            raise RuntimeError("depth-trigger memory changed after query")
        self._confirmed = OrderedDict((key, None) for key in pending.confirmed)
        self._tentative = {
            key: (hits, ordinal) for key, hits, ordinal in pending.tentative
        }
        self._last_ordinal = query.frame_ordinal
        self._last_run_ordinal = pending.last_run_ordinal
        self._burst_remaining = pending.burst_remaining
        self._memory_version += 1
        self._stats["queries"] += 1
        self._stats["runs"] += int(query.run_discovery)
        self._stats["skips"] += int(not query.run_discovery)
        self._stats[f"reason_{query.reason}"] += 1
        self._pending = None

    def summary(self) -> Mapping[str, object]:
        return {
            **dict(sorted(self._stats.items())),
            "confirmed_voxels": len(self._confirmed),
            "tentative_voxels": len(self._tentative),
            "memory_version": self._memory_version,
            "burst_remaining": self._burst_remaining,
            "last_run_ordinal": self._last_run_ordinal,
        }


@dataclass(frozen=True)
class MaskPreselection:
    original_indices: np.ndarray
    box_shortlist_count: int
    precise_eligible_count: int
    input_count: int

    def __post_init__(self) -> None:
        indices = np.asarray(self.original_indices, dtype=np.int64)
        if indices.ndim != 1 or len(np.unique(indices)) != len(indices):
            raise ValueError("preselected indices must be unique [K]")
        frozen = np.frombuffer(np.ascontiguousarray(indices).tobytes(), dtype=np.int64)
        object.__setattr__(self, "original_indices", frozen)


def _explained_union(boxes: np.ndarray, expand_px: float = 4.0) -> np.ndarray:
    result = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.bool_)
    for x1, y1, x2, y2 in boxes:
        left = max(0, int(np.floor(x1 - expand_px)))
        top = max(0, int(np.floor(y1 - expand_px)))
        right = min(IMAGE_WIDTH, int(np.ceil(x2 + expand_px)) + 1)
        bottom = min(IMAGE_HEIGHT, int(np.ceil(y2 + expand_px)) + 1)
        if right > left and bottom > top:
            result[top:bottom, left:right] = True
    return result


def preselect_fastsam_masks(
    *,
    masks: object,
    confidences: object,
    boxes_xyxy: object,
    depth_m: object,
    native_boxes_xyxy: object,
    box_shortlist: int = 12,
    mask_cap: int = 6,
) -> MaskPreselection:
    """Return original FastSAM indices through a deterministic two-level cap."""

    mask_array = np.asarray(masks)
    confidence = np.asarray(confidences, dtype=np.float64)
    boxes = _as_boxes(boxes_xyxy)
    depth = _as_depth(depth_m)
    native = _as_boxes(native_boxes_xyxy)
    count = len(mask_array) if mask_array.ndim == 3 else -1
    if mask_array.shape != (count, IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError("masks must have shape [N,480,640]")
    if confidence.shape != (count,) or boxes.shape != (count, 4):
        raise ValueError("FastSAM masks/confidences/boxes are misaligned")
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("FastSAM confidences must lie in [0,1]")
    if box_shortlist < 1 or mask_cap < 1 or mask_cap > box_shortlist:
        raise ValueError("mask caps must satisfy 1 <= mask_cap <= box_shortlist")
    if count == 0:
        return MaskPreselection(np.empty((0,), dtype=np.int64), 0, 0, 0)

    explained = _explained_union(native)
    integral = np.pad(
        np.cumsum(np.cumsum(explained.astype(np.int64), axis=0), axis=1),
        ((1, 0), (1, 0)),
    )
    cheap_rows: list[tuple[tuple[float, ...], int]] = []
    for index, (box, score) in enumerate(zip(boxes, confidence)):
        x1 = max(0, min(IMAGE_WIDTH - 1, int(np.floor(box[0]))))
        y1 = max(0, min(IMAGE_HEIGHT - 1, int(np.floor(box[1]))))
        x2 = max(x1 + 1, min(IMAGE_WIDTH, int(np.ceil(box[2])) + 1))
        y2 = max(y1 + 1, min(IMAGE_HEIGHT, int(np.ceil(box[3])) + 1))
        width, height = x2 - x1, y2 - y1
        area = width * height
        if area < 200 or area > 122_880 or min(width, height) < 16:
            continue
        if max(width, height) / max(min(width, height), 1) > 6.0:
            continue
        covered = (
            integral[y2, x2]
            - integral[y1, x2]
            - integral[y2, x1]
            + integral[y1, x1]
        )
        rectangle_residual = max(1.0 - float(covered) / max(area, 1), 0.0)
        cheap_rows.append(
            ((float(score) * math.sqrt(max(rectangle_residual, 1.0e-6)), float(score), rectangle_residual, -float(area), -float(index)), index)
        )
    shortlist = [
        index
        for _, index in sorted(cheap_rows, key=lambda row: row[0], reverse=True)[
            :box_shortlist
        ]
    ]

    valid_depth = np.isfinite(depth) & (depth >= 0.10) & (depth <= 6.0)
    precise: list[tuple[tuple[float, ...], int]] = []
    for index in shortlist:
        mask = np.asarray(mask_array[index], dtype=np.bool_)
        pixels = int(np.count_nonzero(mask))
        if pixels < 200 or pixels > 122_880:
            continue
        valid = mask & valid_depth
        valid_count = int(np.count_nonzero(valid))
        residual_count = int(np.count_nonzero(valid & ~explained))
        valid_ratio = valid_count / max(pixels, 1)
        residual_ratio = residual_count / max(valid_count, 1)
        if valid_ratio < 0.50 or residual_count < 200 or residual_ratio < 0.20:
            continue
        score = float(confidence[index])
        precise.append(
            ((score, residual_ratio, valid_ratio, float(residual_count), -float(index)), index)
        )
    selected = np.asarray(
        [index for _, index in sorted(precise, key=lambda row: row[0], reverse=True)[:mask_cap]],
        dtype=np.int64,
    )
    return MaskPreselection(
        original_indices=selected,
        box_shortlist_count=len(shortlist),
        precise_eligible_count=len(precise),
        input_count=count,
    )


__all__ = [
    "DepthResidualEventGate",
    "DepthTriggerConfig",
    "DepthTriggerQuery",
    "MaskPreselection",
    "preselect_fastsam_masks",
]
