"""GT-free, final-box-centric quality evidence for processed CA-1M scenes.

This module deliberately has no prediction mutation API.  It caches the
causal depth/K/pose triples already consumed by BoxFusion and, at scene end,
queries every final world-frame yaw OBB on a deterministic projected-area
Top-K.  The resulting create-only NPZ is training data *input*, not an active
quality score.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from numbers import Integral, Real
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping

import numpy as np

from .tr3d_r2_geometry import (
    classify_depth_rays,
    project_yaw_obb_to_depth,
    stable_top_k_view_indices,
)
from .tr3d_r4_smov_observer import corners_to_yaw_boxes


SCHEMA = "boxfusion.ca1m_native_b6_observer.v1"
DEPTH_CLASS_NAMES = ("support", "occluded", "free_space", "invalid")
FEATURE_NAMES = (
    "detector_score",
    "support_given_depth",
    "occluded_given_depth",
    "free_given_depth",
    "invalid_ratio",
    "view_coverage",
    "sample_support",
    "area_quality",
    "area_stability",
    "support_view_mean",
    "support_view_min",
    "free_view_max",
    "aspect_balance",
    "height_balance",
)


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _finite(name: str, value: Any, *, minimum: float, strict: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    valid = result > minimum if strict else result >= minimum
    if not np.isfinite(result) or not valid:
        relation = "greater than" if strict else "at least"
        raise ValueError(f"{name} must be finite and {relation} {minimum}")
    return result


def _regular_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _rigid(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite [4,4] matrix")
    if not np.allclose(result[3], [0, 0, 0, 1], rtol=0, atol=1e-8):
        raise ValueError(f"{name} must be homogeneous")
    rotation = result[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-4):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, rtol=0, atol=1e-4):
        raise ValueError(f"{name} rotation must be proper")
    return np.ascontiguousarray(result)


def _intrinsics(value: Any, image_shape: tuple[int, int]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape == (4, 4):
        result = result[:3, :3]
    if result.shape != (3, 3) or not np.isfinite(result).all():
        raise ValueError("intrinsics must be finite [3,3] or [4,4]")
    height, width = image_shape
    if result[0, 0] <= 0 or result[1, 1] <= 0:
        raise ValueError("intrinsics focal lengths must be positive")
    if not (0 <= result[0, 2] < width and 0 <= result[1, 2] < height):
        raise ValueError("intrinsics principal point must lie in the depth image")
    return np.ascontiguousarray(result)


@dataclass(frozen=True)
class CA1MNativeB6Config:
    enabled: bool = False
    diagnostics_root: str = ""
    top_k: int = 5
    pixel_stride: int = 4
    margin: float = 0.05
    min_depth: float = 0.10
    max_depth: float = 8.0
    near_clip: float = 1e-3
    max_cached_keyframes: int = 256

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "CA1MNativeB6Config":
        raw = dict(value or {})
        diagnostics = dict(raw.get("diagnostics") or {})
        config = cls(
            enabled=bool(raw.get("enabled", False)),
            diagnostics_root=str(diagnostics.get("root", "")),
            top_k=_positive_int("top_k_views", raw.get("top_k_views", 5)),
            pixel_stride=_positive_int("pixel_stride", raw.get("pixel_stride", 4)),
            margin=_finite("depth_margin_m", raw.get("depth_margin_m", 0.05), minimum=0),
            min_depth=_finite("min_depth_m", raw.get("min_depth_m", 0.10), minimum=0),
            max_depth=_finite("max_depth_m", raw.get("max_depth_m", 8.0), minimum=0),
            near_clip=_finite("near_clip_m", raw.get("near_clip_m", 1e-3), minimum=0, strict=True),
            max_cached_keyframes=_positive_int(
                "max_cached_keyframes", raw.get("max_cached_keyframes", 256)
            ),
        )
        if config.max_depth <= config.min_depth:
            raise ValueError("max_depth_m must exceed min_depth_m")
        if config.enabled:
            if raw.get("observer_only") is not True:
                raise ValueError("enabled native B6 must set observer_only=true")
            if diagnostics.get("enabled") is not True:
                raise ValueError("enabled native B6 requires diagnostics.enabled=true")
            _regular_text("diagnostics.root", config.diagnostics_root)
        return config


@dataclass(frozen=True)
class CA1MNativeB6Summary:
    prediction_rows: int
    mapping_rows: int
    projectable_rows: int
    valid_evidence_rows: int
    frame_count: int
    observer_seconds: float
    diagnostic_path: str


@dataclass(frozen=True)
class _Frame:
    scene_id: str
    frame_id: int
    source_frame_id: str
    depth: np.ndarray
    intrinsics: np.ndarray
    camera_to_world: np.ndarray


def _fractions(counts: np.ndarray) -> np.ndarray:
    total = counts.sum(axis=-1, keepdims=True, dtype=np.int64)
    return np.divide(
        counts,
        total,
        out=np.zeros(counts.shape, dtype=np.float32),
        where=total > 0,
    ).astype(np.float32)


def _features(
    scores: np.ndarray,
    yaw_boxes: np.ndarray,
    topk_valid: np.ndarray,
    topk_area: np.ndarray,
    per_view_counts: np.ndarray,
    aggregate_counts: np.ndarray,
) -> np.ndarray:
    count = len(scores)
    output = np.zeros((count, len(FEATURE_NAMES)), dtype=np.float32)
    if count == 0:
        return output
    aggregate_total = aggregate_counts.sum(axis=1, dtype=np.int64)
    classified = aggregate_counts[:, :3].sum(axis=1, dtype=np.int64)
    evidence = np.divide(
        aggregate_counts,
        aggregate_total[:, None],
        out=np.zeros_like(aggregate_counts, dtype=np.float32),
        where=aggregate_total[:, None] > 0,
    )
    conditional = np.divide(
        aggregate_counts[:, :3],
        classified[:, None],
        out=np.zeros((count, 3), dtype=np.float32),
        where=classified[:, None] > 0,
    )
    view_total = per_view_counts.sum(axis=2, dtype=np.int64)
    view_classified = per_view_counts[:, :, :3].sum(axis=2, dtype=np.int64)
    support_by_view = np.divide(
        per_view_counts[:, :, 0],
        view_classified,
        out=np.zeros(view_classified.shape, dtype=np.float32),
        where=view_classified > 0,
    )
    free_by_view = np.divide(
        per_view_counts[:, :, 2],
        view_classified,
        out=np.zeros(view_classified.shape, dtype=np.float32),
        where=view_classified > 0,
    )
    for row in range(count):
        valid_slots = topk_valid[row] & (view_total[row] > 0)
        areas = topk_area[row, topk_valid[row]]
        support_values = support_by_view[row, valid_slots]
        free_values = free_by_view[row, valid_slots]
        if len(areas):
            area_mean = float(areas.mean())
            logs = np.log(np.maximum(areas, 1e-6))
            area_stability = float(np.exp(-logs.std()))
        else:
            area_mean = 0.0
            area_stability = 0.0
        dx, dy, dz = (float(value) for value in yaw_boxes[row, 3:6])
        planar = float(np.sqrt(dx * dy))
        output[row] = (
            scores[row],
            conditional[row, 0],
            conditional[row, 1],
            conditional[row, 2],
            evidence[row, 3],
            float(topk_valid[row].sum() / topk_valid.shape[1]),
            float(np.clip(np.log1p(aggregate_total[row]) / np.log1p(65536), 0, 1)),
            float(np.clip(area_mean / 0.10, 0, 1)),
            float(np.clip(area_stability, 0, 1)),
            float(support_values.mean()) if len(support_values) else 0.0,
            float(support_values.min()) if len(support_values) else 0.0,
            float(free_values.max()) if len(free_values) else 0.0,
            min(dx, dy) / max(dx, dy),
            min(dz, planar) / max(dz, planar),
        )
    if not np.isfinite(output).all() or np.any(output < 0) or np.any(output > 1):
        raise ValueError("native B6 features must be finite in [0,1]")
    return output


def _write_npz_create_only(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    np.savez_compressed(buffer, **payload)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(buffer.getvalue())
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"immutable native B6 diagnostic exists: {path}") from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


class CA1MNativeB6Observer:
    def __init__(self, config: CA1MNativeB6Config):
        self.config = config
        self.enabled = config.enabled
        self._frames: list[_Frame] = []
        self._scene_id: str | None = None

    def record_keyframe(
        self,
        *,
        scene_id: str,
        frame_id: int,
        source_frame_id: str,
        depth_meters: Any,
        intrinsics: Any,
        camera_to_world: Any,
    ) -> None:
        if not self.enabled:
            return
        scene = _regular_text("scene_id", scene_id)
        if self._scene_id is None:
            self._scene_id = scene
        elif self._scene_id != scene:
            raise ValueError("native B6 observer cannot mix scenes")
        if isinstance(frame_id, (bool, np.bool_)) or not isinstance(frame_id, Integral):
            raise ValueError("frame_id must be a non-negative integer")
        frame = int(frame_id)
        if frame < 0 or any(value.frame_id == frame for value in self._frames):
            raise ValueError("frame_id must be unique and non-negative")
        if len(self._frames) >= self.config.max_cached_keyframes:
            raise ValueError("native B6 keyframe cache limit exceeded")
        depth = np.asarray(depth_meters)
        if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.number):
            raise ValueError("depth_meters must be a numeric [H,W] image")
        depth = np.array(depth, dtype=np.float32, order="C", copy=True)
        image_shape = (int(depth.shape[0]), int(depth.shape[1]))
        intrinsic = _intrinsics(intrinsics, image_shape)
        transform = _rigid(camera_to_world, "camera_to_world")
        self._frames.append(
            _Frame(
                scene_id=scene,
                frame_id=frame,
                source_frame_id=str(source_frame_id),
                depth=depth,
                intrinsics=intrinsic,
                camera_to_world=transform,
            )
        )

    def finalize(
        self,
        *,
        scene_id: str,
        corners: Any,
        scores: Any,
        stable_ids: Any,
    ) -> CA1MNativeB6Summary:
        if not self.enabled:
            raise RuntimeError("disabled native B6 observer cannot finalize")
        started = time.perf_counter()
        scene = _regular_text("scene_id", scene_id)
        if self._scene_id is not None and self._scene_id != scene:
            raise ValueError("native B6 scene mismatch")
        input_corners = np.asarray(corners)
        input_scores = np.asarray(scores)
        input_ids = np.asarray(stable_ids)
        if input_corners.ndim != 3 or input_corners.shape[1:] != (8, 3):
            raise ValueError("corners must be finite [N,8,3]")
        count = len(input_corners)
        if input_scores.shape != (count,) or input_ids.shape != (count,):
            raise ValueError("scores/stable_ids must align with corners")
        if input_ids.dtype.kind not in "iu" or np.any(input_ids < 0):
            raise ValueError("stable_ids must be non-negative integers")
        if len(np.unique(input_ids)) != count:
            raise ValueError("stable_ids must be unique")
        if not np.isfinite(input_corners).all() or not np.isfinite(input_scores).all():
            raise ValueError("prediction input must be finite")
        original_corners = np.array(input_corners, order="C", copy=True)
        original_scores = np.array(input_scores, order="C", copy=True)
        original_ids = np.array(input_ids, order="C", copy=True)
        frozen_corners = np.array(input_corners, dtype=np.float32, order="C", copy=True)
        frozen_scores = np.array(input_scores, dtype=np.float32, order="C", copy=True)
        frozen_ids = np.array(input_ids, dtype=np.int64, order="C", copy=True)
        yaw_boxes = corners_to_yaw_boxes(frozen_corners).astype(np.float32)
        frame_ids = np.asarray([value.frame_id for value in self._frames], dtype=np.int64)
        frame_count = len(self._frames)
        projected_area = np.zeros((count, frame_count), dtype=np.float64)
        projected_valid = np.zeros((count, frame_count), dtype=np.bool_)
        for row, box in enumerate(yaw_boxes):
            for frame_index, frame in enumerate(self._frames):
                projection = project_yaw_obb_to_depth(
                    box,
                    frame.intrinsics,
                    frame.camera_to_world,
                    frame.depth.shape,
                    near_clip=self.config.near_clip,
                )
                if projection is not None:
                    projected_valid[row, frame_index] = True
                    projected_area[row, frame_index] = projection.area_ratio
        topk_ids = np.full((count, self.config.top_k), -1, dtype=np.int64)
        topk_valid = np.zeros((count, self.config.top_k), dtype=np.bool_)
        topk_area = np.zeros((count, self.config.top_k), dtype=np.float32)
        selected_indices: list[np.ndarray] = []
        for row in range(count):
            selected = stable_top_k_view_indices(
                projected_area[row],
                self.config.top_k,
                frame_ids=frame_ids,
                valid_mask=projected_valid[row],
            ) if frame_count else np.empty((0,), dtype=np.int64)
            selected_indices.append(selected)
            valid_count = len(selected)
            if valid_count:
                topk_ids[row, :valid_count] = frame_ids[selected]
                topk_valid[row, :valid_count] = True
                topk_area[row, :valid_count] = projected_area[row, selected]
        per_view_counts = np.zeros(
            (count, self.config.top_k, len(DEPTH_CLASS_NAMES)), dtype=np.int32
        )
        for row, selected in enumerate(selected_indices):
            for slot, frame_index in enumerate(selected.tolist()):
                frame = self._frames[frame_index]
                classification = classify_depth_rays(
                    frame.depth,
                    yaw_boxes[row],
                    frame.intrinsics,
                    frame.camera_to_world,
                    pixel_stride=self.config.pixel_stride,
                    margin=self.config.margin,
                    min_depth=self.config.min_depth,
                    max_depth=self.config.max_depth,
                    near_clip=self.config.near_clip,
                )
                if classification is None:
                    continue
                values = np.asarray(
                    (
                        classification.support_count,
                        classification.occluded_count,
                        classification.free_space_count,
                        classification.invalid_count,
                    ),
                    dtype=np.int64,
                )
                if int(values.sum()) != classification.sample_count:
                    raise AssertionError("depth classes do not partition samples")
                if np.any(values > np.iinfo(np.int32).max):
                    raise OverflowError("native B6 view count exceeds int32")
                per_view_counts[row, slot] = values.astype(np.int32)
        per_view_evidence = _fractions(per_view_counts)
        aggregate_counts = per_view_counts.sum(axis=1, dtype=np.int64)
        aggregate_evidence = _fractions(aggregate_counts)
        aggregate_views = topk_valid.sum(axis=1, dtype=np.int32)
        aggregate_samples = aggregate_counts.sum(axis=1, dtype=np.int64)
        classified_samples = aggregate_counts[:, :3].sum(axis=1, dtype=np.int64)
        valid_evidence = classified_samples > 0
        projectable = projected_valid.any(axis=1) if frame_count else np.zeros(count, bool)
        features = _features(
            frozen_scores, yaw_boxes, topk_valid, topk_area,
            per_view_counts, aggregate_counts,
        )
        runtime = time.perf_counter() - started
        summary_payload = {
            "enabled": True,
            "observer_only": True,
            "mutation_enabled": False,
            "applied_count": 0,
            "ground_truth_access": False,
            "clip_access": False,
            "prediction_rows": count,
            "mapping_rows": count,
            "projectable_rows": int(projectable.sum()),
            "valid_evidence_rows": int(valid_evidence.sum()),
            "frame_count": frame_count,
            "observer_seconds": runtime,
            "orientation_contract": "processed_upright_per_frame_intrinsics_v1",
        }
        target = Path(self.config.diagnostics_root) / f"{scene}_ca1m_native_b6.npz"
        payload = {
            "schema": np.asarray(SCHEMA),
            "complete": np.asarray(True, dtype=np.bool_),
            "observer_only": np.asarray(True, dtype=np.bool_),
            "mutation_enabled": np.asarray(False, dtype=np.bool_),
            "applied_count": np.asarray(0, dtype=np.int64),
            "ground_truth_access": np.asarray(False, dtype=np.bool_),
            "clip_access": np.asarray(False, dtype=np.bool_),
            "scene_id": np.asarray(scene),
            "result_indices": np.arange(count, dtype=np.int64),
            "stable_ids": frozen_ids,
            "corners": frozen_corners,
            "scores": frozen_scores,
            "yaw_boxes": yaw_boxes,
            "used_frame_ids": frame_ids,
            "topk_frame_ids": topk_ids,
            "topk_view_valid": topk_valid,
            "topk_projected_area_fraction": topk_area,
            "per_view_depth_counts": per_view_counts,
            "per_view_depth_evidence": per_view_evidence,
            "aggregate_depth_counts": aggregate_counts,
            "aggregate_depth_evidence": aggregate_evidence,
            "aggregate_view_count": aggregate_views,
            "aggregate_sample_count": aggregate_samples,
            "projectable": projectable.astype(np.bool_),
            "valid_evidence": valid_evidence.astype(np.bool_),
            "feature_names": np.asarray(FEATURE_NAMES, dtype=np.str_),
            "features": features,
            "summary_json": np.asarray(json.dumps(summary_payload, sort_keys=True)),
        }
        _write_npz_create_only(target, payload)
        if (
            not np.array_equal(input_corners, original_corners)
            or not np.array_equal(input_scores, original_scores)
            or not np.array_equal(input_ids, original_ids)
        ):
            raise RuntimeError("native B6 observer mutated its input arrays")
        return CA1MNativeB6Summary(
            prediction_rows=count,
            mapping_rows=count,
            projectable_rows=int(projectable.sum()),
            valid_evidence_rows=int(valid_evidence.sum()),
            frame_count=frame_count,
            observer_seconds=runtime,
            diagnostic_path=str(target),
        )

    @staticmethod
    def summary_text(summary: CA1MNativeB6Summary) -> str:
        return (
            "CA-1M native B6 observer summary | "
            f"rows={summary.mapping_rows}/{summary.prediction_rows}, "
            f"projectable={summary.projectable_rows}, "
            f"valid_evidence={summary.valid_evidence_rows}, "
            f"frames={summary.frame_count}, "
            f"observer_ms={summary.observer_seconds * 1000.0:.3f}"
        )


def build_ca1m_native_b6_observer(
    cfg: Mapping[str, Any],
    diagnostics_root_override: str | os.PathLike[str] | None = None,
) -> CA1MNativeB6Observer:
    raw = dict(cfg.get("ca1m_native_b6_observer") or {})
    if diagnostics_root_override is not None:
        raw.setdefault("diagnostics", {})["root"] = str(diagnostics_root_override)
    return CA1MNativeB6Observer(CA1MNativeB6Config.from_mapping(raw))


__all__ = [
    "CA1MNativeB6Config",
    "CA1MNativeB6Observer",
    "CA1MNativeB6Summary",
    "DEPTH_CLASS_NAMES",
    "FEATURE_NAMES",
    "SCHEMA",
    "build_ca1m_native_b6_observer",
]
