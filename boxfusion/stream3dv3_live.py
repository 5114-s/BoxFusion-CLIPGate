"""Strict online Stream3Dv3-lite residual discovery route.

Data flow:

    past-only depth trigger -> bounded FastSAM preselection -> F0/F2
    -> causal F3 association -> delayed two-view F4
    -> 7D robust fusion -> later selection view -> independent acceptance view
    -> absolute gate -> zero-to-two low-score births

The module is self-contained and is not imported by the public demo entrypoint
until an experiment explicitly enables it.  It owns no trainable parameters,
ground-truth/evaluator API, proposal replay, teacher cache, or terminal oracle.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

from boxfusion.fastsam_automatic_provider import FrozenFastSAMAutomaticMaskProvider
from boxfusion.fastsam_boxer_f4_shadow import FrozenFastSAMBoxerF4Provider
from boxfusion.fastsam_dfu_lgf_shadow import refine_fastsam_candidate
from boxfusion.fastsam_openbox_f3_shadow import FastSAMOpenBoxF3ShadowTracker, make_observation
from boxfusion.fastsam_residual_shadow import select_and_lift_residual_masks
from boxfusion.stream3dv2_lite import aabb_overlap, normalized_center_distance
from boxfusion.stream3dv3_track_fusion import (
    AcceptanceConfig,
    FrozenTrackGeometry,
    TrackEvidenceView,
    TrackFusionResult,
    accept_frozen_geometry,
    attach_boxer_observation,
    build_and_select_geometry,
    pack_mask,
)
from boxfusion.stream3dv3_trigger import (
    DepthResidualEventGate,
    DepthTriggerConfig,
    preselect_fastsam_masks,
)


SCHEMA = "boxfusion.stream3dv3_live.v1"
MAX_F3_VOXELS_PER_OBSERVATION = 512
MAX_TRACK_VIEWS = 6
MAX_ACCEPTED_TRACKS = 128
EVALUATOR_CONFIDENCE_THRESHOLD = 0.05
SCORE_EPSILON = 1.0e-6
SCANNET_MIN_AABB_EXTENT_M = 0.30
NATIVE_DUPLICATE_CENTER_DISTANCE = 0.25


class Stream3Dv3LiveError(RuntimeError):
    pass


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise Stream3Dv3LiveError(f"{label} must be a mapping")
    return value


def _positive(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise Stream3Dv3LiveError(f"{label} must be finite and positive")
    return result


@dataclass(frozen=True)
class Stream3Dv3LiveConfig:
    enabled: bool
    strict_fresh: bool
    fastsam_checkpoint: str
    native_score_lower_bound: float
    target_end_to_end_fps: float
    addon_deadline_ms: float
    diagnostics_root: str | None
    box_shortlist: int
    prelift_top_k: int
    f4_max_views_per_track: int
    f4_max_sources_per_batch: int
    f4_min_baseline_m: float
    f4_min_view_angle_deg: float
    max_births_per_scene: int
    trigger: DepthTriggerConfig
    acceptance: AcceptanceConfig

    @classmethod
    def from_mapping(cls, value: object) -> "Stream3Dv3LiveConfig":
        section = _mapping(value, "online_stream3dv3")
        enabled = bool(section.get("enabled", False))
        f0 = _mapping(section.get("f0", {}), "online_stream3dv3.f0")
        f4 = _mapping(section.get("f4", {}), "online_stream3dv3.f4")
        output = _mapping(section.get("output", {}), "online_stream3dv3.output")
        trigger_row = _mapping(section.get("trigger", {}), "online_stream3dv3.trigger")
        acceptance_row = _mapping(section.get("acceptance", {}), "online_stream3dv3.acceptance")
        diagnostics_root = section.get("diagnostics_root")
        trigger = DepthTriggerConfig(
            sample_stride=int(trigger_row.get("sample_stride", 8)),
            voxel_size_m=float(trigger_row.get("voxel_size_m", 0.15)),
            min_depth_m=float(trigger_row.get("min_depth_m", 0.10)),
            max_depth_m=float(trigger_row.get("max_depth_m", 6.0)),
            native_expand_px=float(trigger_row.get("native_expand_px", 4.0)),
            confirmations=int(trigger_row.get("confirmations", 2)),
            tentative_ttl_keyframes=int(trigger_row.get("tentative_ttl_keyframes", 2)),
            min_persistent_voxels=int(trigger_row.get("min_persistent_voxels", 48)),
            min_persistent_fraction=float(trigger_row.get("min_persistent_fraction", 0.08)),
            cooldown_keyframes=int(trigger_row.get("cooldown_keyframes", 4)),
            burst_keyframes=int(trigger_row.get("burst_keyframes", 4)),
            max_confirmed_voxels=int(trigger_row.get("max_confirmed_voxels", 50_000)),
            max_tentative_voxels=int(trigger_row.get("max_tentative_voxels", 20_000)),
        )
        acceptance_defaults = AcceptanceConfig()
        acceptance_kwargs = {
            name: type(getattr(acceptance_defaults, name))(
                acceptance_row.get(name, getattr(acceptance_defaults, name))
            )
            for name in asdict(acceptance_defaults)
        }
        acceptance = AcceptanceConfig(**acceptance_kwargs)
        box_shortlist = int(f0.get("box_shortlist", 12))
        prelift_top_k = int(f0.get("prelift_top_k", 6))
        f4_views = int(f4.get("max_views_per_track", 2))
        f4_batch = int(f4.get("max_sources_per_batch", 6))
        max_births = int(output.get("max_births_per_scene", 2))
        if not 1 <= prelift_top_k <= box_shortlist <= 100:
            raise Stream3Dv3LiveError("F0 caps must satisfy 1<=prelift_top_k<=box_shortlist<=100")
        if f4_views != 2:
            raise Stream3Dv3LiveError("V3 first version requires exactly two F4 views per track")
        if not 1 <= f4_batch <= 16:
            raise Stream3Dv3LiveError("f4.max_sources_per_batch must lie in [1,16]")
        if not 0 <= max_births <= 2:
            raise Stream3Dv3LiveError("output.max_births_per_scene must lie in [0,2]")
        native_floor = _positive(
            section.get("native_score_lower_bound", 0.125),
            "native_score_lower_bound",
        )
        if native_floor <= EVALUATOR_CONFIDENCE_THRESHOLD + SCORE_EPSILON:
            raise Stream3Dv3LiveError("native score floor leaves no append-score interval")
        return cls(
            enabled=enabled,
            strict_fresh=bool(section.get("strict_fresh", True)),
            fastsam_checkpoint=str(section.get("fastsam_checkpoint", "")),
            native_score_lower_bound=native_floor,
            target_end_to_end_fps=_positive(
                section.get("target_end_to_end_fps", 20.0),
                "target_end_to_end_fps",
            ),
            addon_deadline_ms=_positive(
                section.get("addon_deadline_ms", 285.0),
                "addon_deadline_ms",
            ),
            diagnostics_root=None if diagnostics_root in (None, "") else str(diagnostics_root),
            box_shortlist=box_shortlist,
            prelift_top_k=prelift_top_k,
            f4_max_views_per_track=f4_views,
            f4_max_sources_per_batch=f4_batch,
            f4_min_baseline_m=_positive(f4.get("min_baseline_m", 0.10), "f4.min_baseline_m"),
            f4_min_view_angle_deg=_positive(f4.get("min_view_angle_deg", 8.0), "f4.min_view_angle_deg"),
            max_births_per_scene=max_births,
            trigger=trigger,
            acceptance=acceptance,
        )


@dataclass(frozen=True)
class LiveTerminalResult:
    boxes_3d: np.ndarray
    scores: np.ndarray
    birth_count: int
    overlay_count: int
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class _OutputCandidate:
    track_id: int
    fusion: TrackFusionResult
    native_overlap: tuple[float, float, float]
    native_nd: float

    @property
    def rank(self) -> tuple[float, ...]:
        return (
            float(self.fusion.evidence_score),
            float(self.fusion.acceptance_receipt.quality),
            float(self.fusion.selection_receipt.quality),
            -float(self.fusion.geometry.hb_center_rms_m),
            -float(self.track_id),
        )


def _rgb(value: object) -> np.ndarray:
    image = np.asarray(value)
    if image.shape != (480, 640, 3):
        raise Stream3Dv3LiveError(f"RGB must be [480,640,3], got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all():
            raise Stream3Dv3LiveError("RGB contains NaN/Inf")
        if float(image.max(initial=0.0)) <= 1.0 + 1.0e-5:
            image = image * 255.0
    return np.ascontiguousarray(np.rint(np.clip(image, 0, 255)).astype(np.uint8))


def _depth(value: object) -> np.ndarray:
    depth = np.asarray(value)
    if depth.shape != (480, 640):
        raise Stream3Dv3LiveError("depth must be [480,640]")
    result = depth.astype(np.float32)
    if np.issubdtype(depth.dtype, np.integer):
        result /= 1000.0
    result[~np.isfinite(result)] = 0.0
    result[result < 0.0] = 0.0
    return np.ascontiguousarray(result)


def _matrix(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape == (4, 4) and shape == (3, 3):
        result = result[:3, :3]
    if result.shape != shape or not np.isfinite(result).all():
        raise Stream3Dv3LiveError(f"{label} must be finite {shape}")
    return np.ascontiguousarray(result)


def _boxes(value: object) -> np.ndarray:
    boxes = np.asarray(value, dtype=np.float64)
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1:] != (4,) or not np.isfinite(boxes).all():
        raise Stream3Dv3LiveError("native boxes must be finite [N,4]")
    return np.ascontiguousarray(boxes)


def _bounded_keys(points_world: np.ndarray) -> np.ndarray:
    keys = np.floor(np.asarray(points_world, dtype=np.float64) / 0.05).astype(np.int64)
    if not len(keys):
        return np.empty((0, 3), dtype=np.int64)
    keys = np.unique(keys, axis=0)
    if len(keys) > MAX_F3_VOXELS_PER_OBSERVATION:
        indices = np.linspace(0, len(keys) - 1, MAX_F3_VOXELS_PER_OBSERVATION, dtype=np.int64)
        keys = keys[indices]
    return np.ascontiguousarray(keys)


def _quantiles(values: Sequence[float]) -> dict[str, float | int | None]:
    rows = np.asarray(values, dtype=np.float64)
    if not len(rows):
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(len(rows)),
        "mean": float(np.mean(rows)),
        "p50": float(np.quantile(rows, 0.50)),
        "p95": float(np.quantile(rows, 0.95)),
        "max": float(np.max(rows)),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=os.fspath(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _view_diversity(current: TrackEvidenceView, references: Sequence[TrackEvidenceView]) -> tuple[float, float]:
    target = (current.world_q02 + current.world_q98) * 0.5
    current_ray = target - current.camera_position
    current_ray /= max(float(np.linalg.norm(current_ray)), 1.0e-8)
    baseline = 0.0
    angle = 0.0
    for prior in references:
        baseline = max(baseline, float(np.linalg.norm(current.camera_position - prior.camera_position)))
        prior_ray = target - prior.camera_position
        prior_ray /= max(float(np.linalg.norm(prior_ray)), 1.0e-8)
        angle = max(angle, math.degrees(math.acos(float(np.clip(current_ray @ prior_ray, -1.0, 1.0)))))
    return baseline, angle


def _native_relation(corners: np.ndarray, native: np.ndarray) -> tuple[tuple[float, float, float], float]:
    if not len(native):
        return (0.0, 0.0, 0.0), float("inf")
    relations = [aabb_overlap(corners, row) for row in native]
    distances = [normalized_center_distance(corners, row) for row in native]
    index = max(
        range(len(relations)),
        key=lambda row: (
            relations[row][0],
            max(relations[row][1:]),
            -distances[row],
            -row,
        ),
    )
    return tuple(float(value) for value in relations[index]), float(distances[index])


def _native_novel(overlap: tuple[float, float, float], center_distance: float) -> bool:
    return (
        overlap[0] < 0.10
        and overlap[1] < 0.50
        and overlap[2] < 0.50
        and center_distance >= NATIVE_DUPLICATE_CENTER_DISTANCE
    )


def _scannet_size_valid(corners: np.ndarray) -> bool:
    spans = np.ptp(np.asarray(corners, dtype=np.float64), axis=0)
    return bool(np.all(spans >= SCANNET_MIN_AABB_EXTENT_M))


def _self_duplicate(left: np.ndarray, right: np.ndarray) -> bool:
    iou, left_in_right, right_in_left = aabb_overlap(left, right)
    return iou >= 0.15 or left_in_right >= 0.25 or right_in_left >= 0.25


class Stream3Dv3LiveRoute:
    def __init__(
        self,
        config: Stream3Dv3LiveConfig,
        *,
        lifting_adapter: Any,
        device: str,
        fastsam_provider: Any = None,
        f4_provider: Any = None,
    ) -> None:
        if not config.enabled or not config.strict_fresh:
            raise Stream3Dv3LiveError("V3 route requires enabled strict-fresh mode")
        if lifting_adapter is None and f4_provider is None:
            raise Stream3Dv3LiveError("V3 delayed F4 requires a Boxer adapter")
        self.config = config
        self._lifting_adapter = lifting_adapter
        self._device = str(device)
        self._fastsam = fastsam_provider
        self._f4 = f4_provider
        self._f3 = FastSAMOpenBoxF3ShadowTracker()
        self._trigger = DepthResidualEventGate(config.trigger)
        self._track_views: dict[int, list[TrackEvidenceView]] = {}
        self._frozen: dict[int, FrozenTrackGeometry] = {}
        self._accepted: dict[int, TrackFusionResult] = {}
        self._rejected: set[int] = set()
        self._sealed_tracks: set[int] = set()
        self._scene_id: str | None = None
        self._last_raw_frame = -1
        self._last_keyframe = -1
        self._last_ordinal = -1
        self._finalized = False
        self._counts: Counter[str] = Counter()
        self._gate_rejections: Counter[str] = Counter()
        self._stage_ms: dict[str, list[float]] = defaultdict(list)
        self._keyframe_ms: list[float] = []
        self._f4_per_track: Counter[int] = Counter()
        self._max_f4_attempts_observed = 0
        self._peak_cuda_allocated = 0
        self._peak_cuda_reserved = 0
        self._raw_frame_count = 0
        self._pipeline_started_at: float | None = None
        self._dataset_frame_count: int | None = None
        self._expected_raw_frame_count: int | None = None
        self._keyframe_gap: int | None = None

    def bind_scene(self, scene_id: str) -> None:
        if self._scene_id is None:
            self._scene_id = str(scene_id)
        elif self._scene_id != str(scene_id):
            raise Stream3Dv3LiveError("one V3 route cannot mix scenes")

    def start_pipeline_clock(self, *, dataset_frame_count: int, keyframe_gap: int) -> None:
        """Start the end-to-end scene clock immediately before iteration."""

        if self._pipeline_started_at is not None:
            raise Stream3Dv3LiveError("pipeline clock may start only once")
        frames = int(dataset_frame_count)
        gap = int(keyframe_gap)
        if frames <= 0 or gap <= 0:
            raise Stream3Dv3LiveError("dataset_frame_count and keyframe_gap must be positive")
        self._dataset_frame_count = frames
        self._keyframe_gap = gap
        # Match demo.py's released terminal condition: it stops once no later
        # gap-spaced keyframe remains, after processing max(1, N-gap) frames.
        self._expected_raw_frame_count = max(1, frames - gap)
        self._pipeline_started_at = time.perf_counter()

    def poll(self, raw_frame_id: int) -> None:
        frame = int(raw_frame_id)
        if frame != self._raw_frame_count:
            raise Stream3Dv3LiveError(
                f"raw frames must be consecutive from zero; expected {self._raw_frame_count}, got {frame}"
            )
        if self._pipeline_started_at is None:
            self._pipeline_started_at = time.perf_counter()
        self._raw_frame_count += 1
        self._last_raw_frame = frame

    def _ensure_fastsam(self) -> Any:
        if self._fastsam is None:
            self._fastsam = FrozenFastSAMAutomaticMaskProvider(
                self.config.fastsam_checkpoint, device=self._device
            )
        return self._fastsam

    def _ensure_f4(self) -> Any:
        if self._f4 is None:
            if getattr(self._lifting_adapter, "model", None) is None:
                self._lifting_adapter._load_model()
            self._f4 = FrozenFastSAMBoxerF4Provider(
                self._lifting_adapter,
                device=self._device,
                precision=str(self._lifting_adapter.config.precision),
            )
        return self._f4

    def _record_cuda(self) -> None:
        try:
            import torch

            if self._device.startswith("cuda") and torch.cuda.is_available():
                self._peak_cuda_allocated = max(self._peak_cuda_allocated, int(torch.cuda.max_memory_allocated()))
                self._peak_cuda_reserved = max(self._peak_cuda_reserved, int(torch.cuda.max_memory_reserved()))
        except Exception:
            pass

    def _retire(self, track_ids: Sequence[int]) -> None:
        for track_id in track_ids:
            self._track_views.pop(int(track_id), None)
            self._frozen.pop(int(track_id), None)
            self._rejected.discard(int(track_id))
            self._sealed_tracks.discard(int(track_id))
            self._f4_per_track.pop(int(track_id), None)
            self._counts["tracks_retired"] += 1

    def _commit_empty(self, frame: int, ordinal: int) -> None:
        query = self._f3.query(frame, ordinal, (), max_logical_accessed_ordinal=ordinal)
        commit = self._f3.commit(query)
        self._retire(commit.retired_track_ids)

    def _eligible_for_f4(self, track_id: int, current: TrackEvidenceView) -> tuple[bool, float, float]:
        if track_id in self._frozen or track_id in self._sealed_tracks:
            return False, 0.0, 0.0
        prior = self._track_views.get(track_id, [])
        if len(prior) < 1 or self._f4_per_track[track_id] >= self.config.f4_max_views_per_track:
            return False, 0.0, 0.0
        references = [view for view in prior if view.has_hb] or prior[-1:]
        baseline, angle = _view_diversity(current, references)
        eligible = baseline >= self.config.f4_min_baseline_m or angle >= self.config.f4_min_view_angle_deg
        return eligible, baseline, angle

    @staticmethod
    def _fitting_views(rows: Sequence[TrackEvidenceView]) -> list[TrackEvidenceView] | None:
        hb = [view for view in rows if view.has_hb]
        if len(hb) < 2 or len(rows) < 3:
            return None
        selected = hb[:2]
        remaining = [view for view in rows if view.source_id not in {item.source_id for item in selected}]
        if not remaining:
            return None
        # Prefer the temporally earliest independent coarse view.  All rows are
        # already associated causally by F3.
        selected.append(remaining[0])
        return sorted(selected, key=lambda row: (row.frame_ordinal, row.source_id))

    def _advance_track_state(self, track_id: int, current: TrackEvidenceView) -> None:
        was_frozen = track_id in self._frozen
        rows = (self._track_views.get(track_id, []) + [current])[-MAX_TRACK_VIEWS:]
        self._track_views[track_id] = rows
        if track_id in self._sealed_tracks:
            return
        if was_frozen:
            frozen = self._frozen.pop(track_id)
            result = accept_frozen_geometry(
                frozen,
                current,
                total_distinct_views=len({view.frame_id for view in rows}),
                config=self.config.acceptance,
            )
            if result.absolute_pass:
                self._accepted[track_id] = result
                self._sealed_tracks.add(track_id)
                if len(self._accepted) > MAX_ACCEPTED_TRACKS:
                    worst = min(
                        self._accepted,
                        key=lambda key: (
                            self._accepted[key].evidence_score,
                            -int(key),
                        ),
                    )
                    self._accepted.pop(worst)
                    self._counts["accepted_pool_evictions"] += 1
                self._counts["tracks_accepted"] += 1
            else:
                self._rejected.add(track_id)
                self._sealed_tracks.add(track_id)
                self._counts["tracks_rejected"] += 1
                self._gate_rejections.update(result.reasons)
            return
        fitting = self._fitting_views(rows[:-1])
        if fitting is None:
            return
        try:
            self._frozen[track_id] = build_and_select_geometry(fitting, current)
            self._counts["tracks_frozen"] += 1
        except (ValueError, np.linalg.LinAlgError):
            self._counts["freeze_failures"] += 1

    def process_keyframe(
        self,
        *,
        scene_id: str,
        frame_id: int,
        rgb: object,
        depth_m: object,
        intrinsics: object,
        camera_to_world: object,
        native_boxes_xyxy: object,
    ) -> None:
        if self._finalized:
            raise Stream3Dv3LiveError("route is finalized")
        self.bind_scene(scene_id)
        frame = int(frame_id)
        ordinal = self._last_ordinal + 1
        if frame <= self._last_keyframe:
            raise Stream3Dv3LiveError("keyframe IDs must increase")
        started = time.perf_counter()
        raw_image = np.asarray(rgb)
        if raw_image.shape == (640, 480, 3):
            # Match the existing strict-live contract for the two known
            # non-upright ScanNet producer frames.  Rotating RGB/depth here
            # would invent a new coordinate policy, so advance only the
            # causal F3 clock and leave native BoxFusion untouched.
            self._commit_empty(frame, ordinal)
            self._counts["abstain_non_upright_producer_frame"] += 1
            total = (time.perf_counter() - started) * 1000.0
            self._finish_keyframe(frame, ordinal, total)
            return
        image = _rgb(rgb)
        depth = _depth(depth_m)
        K = _matrix(intrinsics, (3, 3), "intrinsics")
        pose = _matrix(camera_to_world, (4, 4), "camera_to_world")
        native_boxes = _boxes(native_boxes_xyxy)

        stage = time.perf_counter()
        trigger = self._trigger.query(
            frame_id=frame,
            frame_ordinal=ordinal,
            depth_m=depth,
            intrinsics=K,
            camera_to_world=pose,
            native_boxes_xyxy=native_boxes,
        )
        self._stage_ms["depth_trigger"].append((time.perf_counter() - stage) * 1000.0)
        if not trigger.run_discovery:
            stage = time.perf_counter()
            self._commit_empty(frame, ordinal)
            self._stage_ms["f3_and_state"].append((time.perf_counter() - stage) * 1000.0)
            self._trigger.commit(trigger)
            self._counts["keyframes_skipped"] += 1
            total = (time.perf_counter() - started) * 1000.0
            self._finish_keyframe(frame, ordinal, total)
            return

        self._counts["discovery_keyframes"] += 1
        stage = time.perf_counter()
        fastsam = self._ensure_fastsam().infer_bgr(image[..., ::-1].copy())
        self._stage_ms["fastsam"].append((time.perf_counter() - stage) * 1000.0)
        self._counts["fastsam_masks"] += fastsam.count

        stage = time.perf_counter()
        preselection = preselect_fastsam_masks(
            masks=fastsam.masks,
            confidences=fastsam.confidences,
            boxes_xyxy=fastsam.boxes_xyxy,
            depth_m=depth,
            native_boxes_xyxy=native_boxes,
            box_shortlist=self.config.box_shortlist,
            mask_cap=self.config.prelift_top_k,
        )
        selected_indices = preselection.original_indices
        self._stage_ms["mask_preselection"].append((time.perf_counter() - stage) * 1000.0)
        self._counts["box_shortlisted"] += preselection.box_shortlist_count
        self._counts["masks_prelift"] += len(selected_indices)

        stage = time.perf_counter()
        f0 = select_and_lift_residual_masks(
            masks=np.asarray(fastsam.masks[selected_indices]),
            confidences=np.asarray(fastsam.confidences[selected_indices]),
            depth_m=depth,
            explained_boxes_xyxy=native_boxes,
            intrinsics=K,
            camera_to_world=pose,
        )
        self._stage_ms["f0_residual_lift"].append((time.perf_counter() - stage) * 1000.0)
        self._counts["f0_candidates"] += len(f0.candidates)

        stage = time.perf_counter()
        f2_rows = [
            refine_fastsam_candidate(
                points_world=row.points_world,
                world_q02=row.world_q02,
                world_q98=row.world_q98,
                voxel_keys=row.voxel_keys,
            )
            for row in f0.candidates
        ]
        self._stage_ms["f2_refine"].append((time.perf_counter() - stage) * 1000.0)

        f3_observations = []
        drafts: dict[str, TrackEvidenceView] = {}
        source_boxes: dict[str, np.ndarray] = {}
        for candidate, f2 in zip(f0.candidates, f2_rows):
            original_index = int(selected_indices[candidate.raw_index])
            source_id = f"{scene_id}/frame_{frame:06d}/raw_{original_index:03d}"
            retained = candidate.points_world[f2.hlg.retained_indices]
            points = retained if len(retained) else candidate.points_world
            keys = _bounded_keys(points)
            mask = np.asarray(fastsam.masks[original_index], dtype=np.bool_)
            f3_observations.append(
                make_observation(
                    source_id=source_id,
                    frame_id=frame,
                    frame_ordinal=ordinal,
                    confidence=candidate.confidence,
                    world_q02=f2.hlg.world_q02,
                    world_q98=f2.hlg.world_q98,
                    voxel_keys=keys,
                    camera_to_world=pose,
                    intrinsics=K,
                    mask=mask,
                )
            )
            drafts[source_id] = TrackEvidenceView(
                source_id=source_id,
                frame_id=frame,
                frame_ordinal=ordinal,
                mask_confidence=candidate.confidence,
                residual_ratio=candidate.residual_ratio,
                valid_ratio=candidate.valid_ratio,
                tight_box_xyxy=candidate.tight_box_xyxy,
                mask_packbits=pack_mask(mask),
                points_world=points,
                world_q02=f2.hlg.world_q02,
                world_q98=f2.hlg.world_q98,
                intrinsics=K,
                camera_to_world=pose,
            )
            source_boxes[source_id] = np.asarray(candidate.tight_box_xyxy, dtype=np.float32)

        stage = time.perf_counter()
        f3_query = self._f3.query(
            frame,
            ordinal,
            f3_observations,
            max_logical_accessed_ordinal=ordinal,
        )
        f3_commit = self._f3.commit(f3_query)
        self._retire(f3_commit.retired_track_ids)
        assignments = {
            row.source_id: int(row.track_id)
            for row in f3_commit.assignments
            if row.track_id is not None
        }
        self._stage_ms["f3_and_state"].append((time.perf_counter() - stage) * 1000.0)

        eligible = []
        for source_id, track_id in assignments.items():
            allowed, baseline, angle = self._eligible_for_f4(track_id, drafts[source_id])
            if allowed:
                quality = drafts[source_id].mask_confidence * math.sqrt(
                    max(drafts[source_id].residual_ratio * drafts[source_id].valid_ratio, 0.0)
                )
                eligible.append((quality, baseline, angle, source_id, track_id))
        eligible.sort(key=lambda row: (row[0], row[1], row[2], row[3]), reverse=True)
        eligible = eligible[: self.config.f4_max_sources_per_batch]
        selected_sources = [row[3] for row in eligible]
        if selected_sources:
            stage = time.perf_counter()
            for source_id in selected_sources:
                self._f4_per_track[assignments[source_id]] += 1
                self._max_f4_attempts_observed = max(
                    self._max_f4_attempts_observed,
                    int(self._f4_per_track[assignments[source_id]]),
                )
            self._counts["f4_attempts"] += len(selected_sources)
            f4 = self._ensure_f4().infer_batch(
                scene_id,
                frame,
                image,
                depth,
                K,
                pose,
                np.stack([source_boxes[source] for source in selected_sources]),
                selected_sources,
            )
            self._stage_ms["f4_boxer"].append((time.perf_counter() - stage) * 1000.0)
            self._counts["f4_batches"] += 1
            for row in f4.rows:
                track_id = assignments[row.source_id]
                if (
                    row.valid
                    and row.world_center is not None
                    and row.local_extent is not None
                    and row.world_rotation is not None
                    and row.confidence is not None
                ):
                    drafts[row.source_id] = attach_boxer_observation(
                        drafts[row.source_id],
                        center=row.world_center,
                        extent=row.local_extent,
                        rotation=row.world_rotation,
                        confidence=row.confidence,
                    )
                    self._counts["f4_valid"] += 1
                else:
                    self._counts["f4_invalid"] += 1
        else:
            self._stage_ms["f4_boxer"].append(0.0)

        stage = time.perf_counter()
        for source_id, track_id in sorted(assignments.items(), key=lambda row: (row[1], row[0])):
            self._advance_track_state(track_id, drafts[source_id])
        self._stage_ms["track_fusion_and_gate"].append((time.perf_counter() - stage) * 1000.0)
        self._trigger.commit(trigger)
        total = (time.perf_counter() - started) * 1000.0
        self._finish_keyframe(frame, ordinal, total)

    def _finish_keyframe(self, frame: int, ordinal: int, total_ms: float) -> None:
        self._keyframe_ms.append(float(total_ms))
        self._counts["keyframes"] += 1
        self._counts["addon_deadline_misses"] += int(total_ms > self.config.addon_deadline_ms)
        self._last_keyframe = frame
        self._last_ordinal = ordinal
        self._record_cuda()

    def _diagnostics(
        self,
        *,
        native_boxes: np.ndarray,
        output_boxes: np.ndarray,
        native_scores: np.ndarray,
        output_scores: np.ndarray,
        births: int,
    ) -> dict[str, Any]:
        f3_terminal = self._f3.finalize()
        assert self._pipeline_started_at is not None
        pipeline_seconds = time.perf_counter() - self._pipeline_started_at
        if not math.isfinite(pipeline_seconds) or pipeline_seconds <= 0.0:
            raise Stream3Dv3LiveError("invalid pipeline runtime")
        native = int(len(native_scores))
        output = int(len(output_scores))
        native_prefix = np.ascontiguousarray(native_scores, dtype=np.float32)
        output_prefix = np.ascontiguousarray(output_scores[:native], dtype=np.float32)
        append_scores = np.ascontiguousarray(output_scores[native:], dtype=np.float32)
        native_geometry = np.ascontiguousarray(native_boxes, dtype=np.float32)
        output_geometry_prefix = np.ascontiguousarray(output_boxes[:native], dtype=np.float32)
        return {
            "schema": SCHEMA,
            "complete": True,
            "scene_id": self._scene_id,
            "run_fingerprint": os.environ.get("BOXFUSION_RUN_FINGERPRINT"),
            "fresh_inference": True,
            "training_free": True,
            "pose_source": "scannet_provided_pose",
            "past_only": True,
            "query_before_commit": True,
            "selection_and_acceptance_held_out": True,
            "future_access_count": 0,
            "ground_truth_access": False,
            "annotation_access": False,
            "evaluator_access": False,
            "proposal_cache_access": False,
            "teacher_cache_access": False,
            "terminal_cache_access": False,
            "native_scores_preserved": True,
            "score_audit": {
                "dtype": "float32",
                "native_prefix_sha256": hashlib.sha256(native_prefix.tobytes()).hexdigest(),
                "output_prefix_sha256": hashlib.sha256(output_prefix.tobytes()).hexdigest(),
                "append_scores": [float(value) for value in append_scores],
                "native_min": None if not native else float(np.min(native_prefix)),
            },
            "geometry_audit": {
                "dtype": "float32",
                "native_prefix_sha256": hashlib.sha256(native_geometry.tobytes()).hexdigest(),
                "output_prefix_sha256": hashlib.sha256(output_geometry_prefix.tobytes()).hexdigest(),
            },
            "target_end_to_end_fps": self.config.target_end_to_end_fps,
            "addon_deadline_ms": self.config.addon_deadline_ms,
            "raw_frame_count": int(self._raw_frame_count),
            "pipeline_seconds": float(pipeline_seconds),
            "runtime": {
                "raw_frame_count": int(self._raw_frame_count),
                "pipeline_seconds": float(pipeline_seconds),
                "end_to_end_fps": float(self._raw_frame_count / pipeline_seconds),
                "clock_scope": (
                    "pre-iteration V3 clock (first-poll fallback) through terminal V3 "
                    "decision and F3 finalization; includes frame loading, native BoxFusion, "
                    "and synchronous V3 work; excludes process/model construction and "
                    "diagnostic serialization"
                ),
            },
            "schedule": {
                "dataset_frame_count": self._dataset_frame_count,
                "keyframe_gap": self._keyframe_gap,
                "expected_raw_frame_count": self._expected_raw_frame_count,
                "expected_keyframe_count": (
                    None
                    if self._expected_raw_frame_count is None or self._keyframe_gap is None
                    else (self._expected_raw_frame_count + self._keyframe_gap - 1)
                    // self._keyframe_gap
                ),
            },
            "counts": {
                **{key: int(value) for key, value in sorted(self._counts.items())},
                "native": native,
                "births": int(births),
                "overlays": 0,
                "output": output,
                "accepted_track_pool": len(self._accepted),
            },
            "trigger": dict(self._trigger.summary()),
            "gate_rejections": dict(sorted(self._gate_rejections.items())),
            "f4_per_track": {str(key): int(value) for key, value in sorted(self._f4_per_track.items())},
            "timing_ms": {
                "keyframe_total": _quantiles(self._keyframe_ms),
                **{key: _quantiles(value) for key, value in sorted(self._stage_ms.items())},
            },
            "bounded": {
                "max_track_views": MAX_TRACK_VIEWS,
                "max_accepted_tracks": MAX_ACCEPTED_TRACKS,
                "max_f4_views_per_track": self.config.f4_max_views_per_track,
                "max_f4_attempts_observed": self._max_f4_attempts_observed,
                "max_f4_sources_per_batch": self.config.f4_max_sources_per_batch,
                "prelift_top_k": self.config.prelift_top_k,
                "max_births_per_scene": self.config.max_births_per_scene,
            },
            "f3": {
                "keyframes": f3_terminal.keyframe_count,
                "audit_complete": f3_terminal.audit_complete,
                "max_logical_accessed_ordinal": f3_terminal.max_logical_accessed_ordinal,
            },
            "sam3": {"enabled": False},
            "peak_cuda_allocated_bytes": self._peak_cuda_allocated,
            "peak_cuda_reserved_bytes": self._peak_cuda_reserved,
        }

    def finalize(
        self,
        *,
        native_boxes_3d: object,
        native_scores: object,
        final_frame_id: int,
    ) -> LiveTerminalResult:
        if self._finalized:
            raise Stream3Dv3LiveError("finalize may run only once")
        self._finalized = True
        if self._pipeline_started_at is None or self._raw_frame_count <= 0:
            raise Stream3Dv3LiveError("finalize requires at least one raw-frame poll")
        if os.environ.get("BOXFUSION_STRICT_LIVE") == "1" and self._expected_raw_frame_count is None:
            raise Stream3Dv3LiveError("strict live run did not bind the dataset schedule")
        if (
            self._expected_raw_frame_count is not None
            and self._raw_frame_count != self._expected_raw_frame_count
        ):
            raise Stream3Dv3LiveError(
                "raw-frame count differs from the bound demo terminal schedule"
            )
        if int(final_frame_id) != self._last_raw_frame:
            raise Stream3Dv3LiveError("final_frame_id does not match the last raw-frame poll")
        boxes = np.asarray(native_boxes_3d, dtype=np.float64)
        scores = np.asarray(native_scores, dtype=np.float64)
        if boxes.shape != (len(scores), 8, 3) or not np.isfinite(boxes).all():
            raise Stream3Dv3LiveError("native boxes/scores are misaligned")
        if not np.isfinite(scores).all() or np.any(scores <= 0.0):
            raise Stream3Dv3LiveError("native scores must be finite and positive")
        candidates = []
        for track_id, fusion in self._accepted.items():
            overlap, nd = _native_relation(fusion.geometry.corners, boxes)
            if not _scannet_size_valid(fusion.geometry.corners):
                self._gate_rejections["scannet_size_filter"] += 1
                continue
            if not _native_novel(overlap, nd):
                self._gate_rejections["native_duplication"] += 1
                continue
            candidates.append(_OutputCandidate(track_id, fusion, overlap, nd))
        selected: list[_OutputCandidate] = []
        for candidate in sorted(candidates, key=lambda row: row.rank, reverse=True):
            if any(_self_duplicate(candidate.fusion.geometry.corners, prior.fusion.geometry.corners) for prior in selected):
                self._gate_rejections["self_nms"] += 1
                continue
            if len(selected) >= self.config.max_births_per_scene:
                self._gate_rejections["scene_cap"] += 1
                continue
            selected.append(candidate)
        output_boxes = np.array(boxes, copy=True)
        output_scores = np.array(scores, copy=True)
        if selected:
            floor = (
                min(float(np.min(scores)), self.config.native_score_lower_bound)
                if len(scores)
                else self.config.native_score_lower_bound
            )
            append_scores = np.linspace(
                floor - SCORE_EPSILON,
                EVALUATOR_CONFIDENCE_THRESHOLD + SCORE_EPSILON,
                len(selected),
                dtype=np.float64,
            )
            ordered = sorted(selected, key=lambda row: row.rank, reverse=True)
            output_boxes = np.concatenate(
                (output_boxes, np.stack([row.fusion.geometry.corners for row in ordered])), axis=0
            )
            output_scores = np.concatenate((output_scores, append_scores))
        if len(scores) and len(output_scores) > len(scores):
            if not np.all(output_scores[len(scores) :] < np.min(scores)):
                raise Stream3Dv3LiveError("birth scores are not a strict low-score suffix")
        if not np.array_equal(output_scores[: len(scores)], scores):
            raise Stream3Dv3LiveError("native score prefix changed")
        native_scores_f32 = np.ascontiguousarray(scores, dtype=np.float32)
        output_scores_f32 = np.ascontiguousarray(output_scores, dtype=np.float32)
        native_boxes_f32 = np.ascontiguousarray(boxes, dtype=np.float32)
        output_boxes_f32 = np.ascontiguousarray(output_boxes, dtype=np.float32)
        if not np.array_equal(output_scores_f32[: len(scores)], native_scores_f32):
            raise Stream3Dv3LiveError("float32 native score prefix changed")
        if not np.array_equal(output_boxes_f32[: len(scores)], native_boxes_f32):
            raise Stream3Dv3LiveError("float32 native geometry prefix changed")
        diagnostics = self._diagnostics(
            native_boxes=native_boxes_f32,
            output_boxes=output_boxes_f32,
            native_scores=native_scores_f32,
            output_scores=output_scores_f32,
            births=len(selected),
        )
        if self.config.diagnostics_root is not None:
            assert self._scene_id is not None
            _atomic_json(Path(self.config.diagnostics_root) / f"{self._scene_id}.json", diagnostics)
        return LiveTerminalResult(
            boxes_3d=output_boxes_f32,
            scores=output_scores_f32,
            birth_count=len(selected),
            overlay_count=0,
            diagnostics=diagnostics,
        )

    def close(self) -> None:
        return None


def build_stream3dv3_live_route(
    cfg: Mapping[str, Any],
    *,
    lifting_adapter: Any,
    device: str,
) -> Stream3Dv3LiveRoute | None:
    config = Stream3Dv3LiveConfig.from_mapping(cfg.get("online_stream3dv3", {}))
    if not config.enabled:
        return None
    if bool(_mapping(cfg.get("online_stream3dv2", {}), "online_stream3dv2").get("enabled", False)):
        raise Stream3Dv3LiveError("Stream3Dv2 and Stream3Dv3 are mutually exclusive")
    proposal_mode = str(
        _mapping(_mapping(cfg.get("lifting", {}), "lifting").get("proposal_cache", {}), "proposal_cache").get("mode", "disabled")
    ).lower()
    if proposal_mode not in {"disabled", "none", "off", ""}:
        raise Stream3Dv3LiveError("strict V3 route forbids proposal replay/record")
    return Stream3Dv3LiveRoute(
        config,
        lifting_adapter=lifting_adapter,
        device=device,
    )


__all__ = [
    "LiveTerminalResult",
    "SCHEMA",
    "Stream3Dv3LiveConfig",
    "Stream3Dv3LiveError",
    "Stream3Dv3LiveRoute",
    "build_stream3dv3_live_route",
]
