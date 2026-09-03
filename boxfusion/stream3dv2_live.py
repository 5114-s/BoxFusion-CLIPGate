"""Strict live execution path for the F4 Stream3Dv2-lite experiment.

The route consumes fresh current-frame RGB-D, pose and native 2D boxes.  It
does not accept proposal, teacher, track or terminal caches.  FastSAM/F2/F4
run on true keyframes, F3 and :mod:`stream3dv2_online_state` enforce
query-before-commit causal association, and a one-slot SAM3 subprocess is
polled on every raw frame.  Only results that were available no later than a
track's decision frame may influence that track.

The terminal map update is intentionally conservative: native rows and their
real scores are retained, at most one safe geometry overlay is applied, and
at most six genuinely unmatched rows receive a strictly lower score suffix.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
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
from boxfusion.fastsam_openbox_f3_shadow import (
    FastSAMOpenBoxF3ShadowTracker,
    make_observation,
)
from boxfusion.fastsam_residual_shadow import select_and_lift_residual_masks
from boxfusion.live_sam3_client import LiveSAM3Client, LiveSAM3Config, LiveSAM3Result
from boxfusion.sam3_diverse_maskdepth_birth import (
    SAM3BirthConfig,
    SAM3MemoryTeacherView,
)
from boxfusion.stream3dv2_lite import TrackView
from boxfusion.stream3dv2_online_state import (
    FinalizedTrack,
    Stream3Dv2OnlineState,
    TrackUpdate,
)
from boxfusion.stream3dv3_trigger import (
    DepthResidualEventGate,
    DepthTriggerConfig,
    preselect_fastsam_masks,
)
from boxfusion.tm_fpf_c1 import TMFPFC1ContractError
from tools.materialize_scannet_f4_stream3dv2_lite_full100 import (
    MAX_OVERLAYS_PER_SCENE,
    PRESELECT_BIRTHS_PER_SCENE,
    PRESELECT_OVERLAYS_PER_SCENE,
    Candidate,
    _is_native_novel,
    _native_relation,
    _overlay_safe,
    _score_births,
    _select_births,
    _semantic_enrich,
)
from tools.materialize_scannet_raw_boxer_past3_birth_full100 import NativePrediction


SCHEMA = "boxfusion.stream3dv2_live.v1"
MAX_SEMANTIC_VIEWS_HARD = 32
MAX_F3_VOXELS_PER_OBSERVATION = 512
_TM_FPF_C1_VIEW_ABSTENTIONS = {
    "target mask has too few pixels": "too_few_mask_pixels",
    "target mask has too few valid depth pixels": "too_few_valid_depth_pixels",
}


class Stream3Dv2LiveError(RuntimeError):
    """A strict live route contract was violated."""


def tm_fpf_c1_view_abstention_reason(error: BaseException) -> str | None:
    """Classify only the two expected per-view evidence insufficiencies."""

    if not isinstance(error, TMFPFC1ContractError):
        return None
    return _TM_FPF_C1_VIEW_ABSTENTIONS.get(str(error))


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise Stream3Dv2LiveError(f"{label} must be a mapping")
    return value


def _finite_positive(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise Stream3Dv2LiveError(f"{label} must be finite and positive")
    return result


@dataclass(frozen=True)
class Stream3Dv2LiveConfig:
    enabled: bool
    fastsam_checkpoint: str
    native_score_lower_bound: float
    keyframe_deadline_ms: float
    max_semantic_views: int
    diagnostics_root: str | None
    sam3_enabled: bool
    sam3_interval_keyframes: int
    sam3_drain_timeout_seconds: float
    sam3_config: LiveSAM3Config
    lightweight_enabled: bool = False
    depth_trigger_enabled: bool = False
    depth_trigger_config: DepthTriggerConfig = field(
        default_factory=DepthTriggerConfig
    )
    fastsam_box_shortlist: int = 16
    fastsam_top_k: int = 16
    conditional_f2: bool = False
    f4_top_m_tracks: int = 16
    terminal_clip_enabled: bool = False
    terminal_clip_batch_size: int = 32

    @classmethod
    def from_mapping(cls, value: object) -> "Stream3Dv2LiveConfig":
        section = _mapping(value, "online_stream3dv2")
        enabled = bool(section.get("enabled", False))
        sam3 = _mapping(section.get("sam3", {}), "online_stream3dv2.sam3")
        lightweight = _mapping(
            section.get("lightweight", {}),
            "online_stream3dv2.lightweight",
        )
        lightweight_enabled = bool(lightweight.get("enabled", False))
        trigger_row = _mapping(
            lightweight.get("depth_trigger", {}),
            "online_stream3dv2.lightweight.depth_trigger",
        )
        preselect_row = _mapping(
            lightweight.get("fastsam_top_k", {}),
            "online_stream3dv2.lightweight.fastsam_top_k",
        )
        terminal_clip_row = _mapping(
            lightweight.get("terminal_clip", {}),
            "online_stream3dv2.lightweight.terminal_clip",
        )
        max_views = int(section.get("max_semantic_views", 8))
        interval = int(sam3.get("proposal_interval_keyframes", 5))
        if not 1 <= max_views <= MAX_SEMANTIC_VIEWS_HARD:
            raise Stream3Dv2LiveError(
                f"max_semantic_views must be in [1,{MAX_SEMANTIC_VIEWS_HARD}]"
            )
        if interval < 1:
            raise Stream3Dv2LiveError("SAM3 proposal interval must be positive")
        if int(sam3.get("max_pending", 1)) != 1:
            raise Stream3Dv2LiveError("strict live SAM3 queue capacity is exactly one")
        box_shortlist = int(preselect_row.get("box_shortlist", 16))
        fastsam_top_k = int(preselect_row.get("mask_cap", 16))
        f4_top_m_tracks = int(lightweight.get("f4_top_m_tracks", 16))
        terminal_clip_batch_size = int(terminal_clip_row.get("batch_size", 32))
        if not 1 <= fastsam_top_k <= box_shortlist <= 100:
            raise Stream3Dv2LiveError(
                "lightweight FastSAM caps must satisfy "
                "1<=mask_cap<=box_shortlist<=100"
            )
        if not 1 <= f4_top_m_tracks <= 16:
            raise Stream3Dv2LiveError(
                "lightweight.f4_top_m_tracks must lie in [1,16]"
            )
        if not 1 <= terminal_clip_batch_size <= 256:
            raise Stream3Dv2LiveError(
                "lightweight.terminal_clip.batch_size must lie in [1,256]"
            )
        trigger = DepthTriggerConfig(
            sample_stride=int(trigger_row.get("sample_stride", 8)),
            voxel_size_m=float(trigger_row.get("voxel_size_m", 0.15)),
            min_depth_m=float(trigger_row.get("min_depth_m", 0.10)),
            max_depth_m=float(trigger_row.get("max_depth_m", 6.0)),
            native_expand_px=float(trigger_row.get("native_expand_px", 4.0)),
            confirmations=int(trigger_row.get("confirmations", 2)),
            tentative_ttl_keyframes=int(
                trigger_row.get("tentative_ttl_keyframes", 2)
            ),
            min_persistent_voxels=int(
                trigger_row.get("min_persistent_voxels", 48)
            ),
            min_persistent_fraction=float(
                trigger_row.get("min_persistent_fraction", 0.08)
            ),
            cooldown_keyframes=int(trigger_row.get("cooldown_keyframes", 4)),
            burst_keyframes=int(trigger_row.get("burst_keyframes", 4)),
            max_confirmed_voxels=int(
                trigger_row.get("max_confirmed_voxels", 50_000)
            ),
            max_tentative_voxels=int(
                trigger_row.get("max_tentative_voxels", 20_000)
            ),
        )
        precision = str(sam3.get("precision", "bfloat16")).lower()
        if precision == "bfloat16":
            precision = "bf16"
        elif precision == "float32":
            precision = "fp32"
        diagnostics_root = section.get("diagnostics_root")
        if diagnostics_root in (None, ""):
            diagnostics_root = None
        config = LiveSAM3Config(
            enabled=bool(sam3.get("enabled", True)) and enabled,
            python_executable=str(
                sam3.get("python", "/home/admin1/miniconda3/envs/sam3/bin/python")
            ),
            sam3_root=str(sam3.get("source_root", "")),
            checkpoint=str(sam3.get("checkpoint", "")),
            bpe_path=(
                None if sam3.get("bpe_path") in (None, "") else str(sam3["bpe_path"])
            ),
            device=str(sam3.get("device", "cuda:0")),
            precision=precision,
            resolution=int(sam3.get("resolution", 1008)),
            max_proposals=int(sam3.get("max_proposals", 64)),
            late_after_s=float(sam3.get("late_after_seconds", 2.0)),
            drop_late_results=True,
            raise_worker_errors=bool(sam3.get("raise_worker_errors", False)),
        )
        return cls(
            enabled=enabled,
            fastsam_checkpoint=str(section.get("fastsam_checkpoint", "")),
            native_score_lower_bound=_finite_positive(
                section.get("native_score_lower_bound", 0.125),
                "native_score_lower_bound",
            ),
            keyframe_deadline_ms=_finite_positive(
                section.get("keyframe_deadline_ms", 833.333333),
                "keyframe_deadline_ms",
            ),
            max_semantic_views=max_views,
            diagnostics_root=None if diagnostics_root is None else str(diagnostics_root),
            sam3_enabled=config.enabled,
            sam3_interval_keyframes=interval,
            sam3_drain_timeout_seconds=_finite_positive(
                sam3.get("drain_timeout_seconds", 0.833333),
                "sam3.drain_timeout_seconds",
            ),
            sam3_config=config,
            lightweight_enabled=lightweight_enabled,
            depth_trigger_enabled=(
                lightweight_enabled and bool(trigger_row.get("enabled", True))
            ),
            depth_trigger_config=trigger,
            fastsam_box_shortlist=box_shortlist,
            fastsam_top_k=fastsam_top_k,
            conditional_f2=(
                lightweight_enabled and bool(lightweight.get("conditional_f2", True))
            ),
            f4_top_m_tracks=f4_top_m_tracks,
            terminal_clip_enabled=(
                lightweight_enabled
                and bool(terminal_clip_row.get("enabled", False))
            ),
            terminal_clip_batch_size=terminal_clip_batch_size,
        )


@dataclass(frozen=True)
class LiveTerminalResult:
    boxes_3d: np.ndarray
    scores: np.ndarray
    birth_count: int
    overlay_count: int
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class NativeTargetMaskFrame:
    """The exact automatic-mask frame already consumed by live discovery.

    This is a borrowed, immediate-use view of the one FastSAM inference for a
    keyframe.  It deliberately contains no residual/F0 masks: TM-FPF must
    match native rows against the original class-agnostic automatic masks.
    ``process_keyframe`` returns ``None`` whenever discovery abstains before
    FastSAM (for example a depth-trigger miss or non-upright producer frame).
    """

    scene_id: str
    frame_id: int
    native_boxes_xyxy: np.ndarray
    masks: np.ndarray
    automatic_boxes_xyxy: np.ndarray
    automatic_confidences: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    camera_to_world: np.ndarray


@dataclass(frozen=True)
class _SemanticAvailability:
    view: SAM3MemoryTeacherView
    ready_frame_id: int


def _rgb_uint8(value: object) -> np.ndarray:
    image = np.asarray(value)
    if image.shape != (480, 640, 3):
        raise Stream3Dv2LiveError(f"live RGB must be [480,640,3], got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all():
            raise Stream3Dv2LiveError("live RGB contains NaN or Inf")
        if float(image.max(initial=0.0)) <= 1.0 + 1.0e-5:
            image = image * 255.0
        image = np.rint(np.clip(image, 0.0, 255.0)).astype(np.uint8)
    else:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _depth_float(value: object) -> np.ndarray:
    depth = np.asarray(value)
    if depth.shape != (480, 640):
        raise Stream3Dv2LiveError(f"live depth must be [480,640], got {depth.shape}")
    if np.issubdtype(depth.dtype, np.integer):
        result = depth.astype(np.float32) / 1000.0
    else:
        result = depth.astype(np.float32)
    result[~np.isfinite(result)] = 0.0
    result[result < 0.0] = 0.0
    return np.ascontiguousarray(result)


def _matrix(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    if result.shape != shape or not np.isfinite(result).all():
        raise Stream3Dv2LiveError(f"{label} must be finite with shape {shape}")
    return result


def _boxes_xyxy(value: object) -> np.ndarray:
    boxes = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1:] != (4,) or not np.isfinite(boxes).all():
        raise Stream3Dv2LiveError("native 2D boxes must be finite [N,4]")
    return boxes


def _bounded_f3_keys(points_world: np.ndarray) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float64)
    keys = np.floor(points / 0.05).astype(np.int64)
    if not len(keys):
        return np.empty((0, 3), dtype=np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    ordered = keys[order]
    keep = np.empty(len(ordered), dtype=np.bool_)
    keep[0] = True
    keep[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    unique = ordered[keep]
    if len(unique) > MAX_F3_VOXELS_PER_OBSERVATION:
        indices = np.linspace(
            0,
            len(unique) - 1,
            MAX_F3_VOXELS_PER_OBSERVATION,
            endpoint=True,
            dtype=np.int64,
        )
        unique = unique[indices]
    return np.ascontiguousarray(unique, dtype=np.int64)


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
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


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=os.fspath(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
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


class Stream3Dv2LiveRoute:
    """One-scene strict live route with bounded model and evidence state."""

    def __init__(
        self,
        config: Stream3Dv2LiveConfig,
        *,
        lifting_adapter: Any,
        device: str,
        fastsam_provider: Any = None,
        sam3_client: LiveSAM3Client | None = None,
        f4_provider: Any = None,
    ) -> None:
        if not config.enabled:
            raise Stream3Dv2LiveError("cannot construct a disabled live route")
        if lifting_adapter is None and f4_provider is None:
            raise Stream3Dv2LiveError("live F4 requires the active Boxer adapter")
        self.config = config
        self._lifting_adapter = lifting_adapter
        self._device = str(device)
        self._fastsam = fastsam_provider
        self._sam3 = sam3_client or LiveSAM3Client(config.sam3_config)
        self._f4 = f4_provider
        self._f3 = FastSAMOpenBoxF3ShadowTracker()
        self._state = Stream3Dv2OnlineState()
        self._depth_trigger = (
            DepthResidualEventGate(config.depth_trigger_config)
            if config.depth_trigger_enabled
            else None
        )
        self._scene_id: str | None = None
        self._last_raw_frame_id = -1
        self._last_keyframe_id = -1
        self._last_keyframe_ordinal = -1
        self._finalized = False
        self._finalized_tracks: list[FinalizedTrack] = []
        self._semantic_views: list[_SemanticAvailability] = []
        self._sam_pending_inputs: dict[int, dict[str, Any]] = {}
        self._stage_ms: dict[str, list[float]] = defaultdict(list)
        self._frame_total_ms: list[float] = []
        self._counts: Counter[str] = Counter()
        self._hypotheses: Counter[str] = Counter()
        self._peak_cuda_allocated_bytes = 0
        self._peak_cuda_reserved_bytes = 0

    @property
    def enabled(self) -> bool:
        return True

    def bind_scene(self, scene_id: str) -> None:
        if self._scene_id is None:
            self._scene_id = str(scene_id)
        elif self._scene_id != str(scene_id):
            raise Stream3Dv2LiveError(
                f"one live route cannot mix scenes: {self._scene_id} != {scene_id}"
            )

    def _ensure_fastsam(self) -> Any:
        if self._fastsam is None:
            self._fastsam = FrozenFastSAMAutomaticMaskProvider(
                self.config.fastsam_checkpoint,
                device=self._device,
            )
        return self._fastsam

    def _ensure_f4(self) -> Any:
        if self._f4 is None:
            if getattr(self._lifting_adapter, "model", None) is None:
                self._lifting_adapter._load_model()
            precision = str(self._lifting_adapter.config.precision)
            self._f4 = FrozenFastSAMBoxerF4Provider(
                self._lifting_adapter,
                device=self._device,
                precision=precision,
            )
        return self._f4

    def _record_cuda_peak(self) -> None:
        try:
            import torch

            if str(self._device).startswith("cuda") and torch.cuda.is_available():
                self._peak_cuda_allocated_bytes = max(
                    self._peak_cuda_allocated_bytes,
                    int(torch.cuda.max_memory_allocated()),
                )
                self._peak_cuda_reserved_bytes = max(
                    self._peak_cuda_reserved_bytes,
                    int(torch.cuda.max_memory_reserved()),
                )
        except Exception:
            return

    def _accept_sam3_result(self, result: LiveSAM3Result, ready_frame_id: int) -> None:
        pending = self._sam_pending_inputs.pop(result.request_id, None)
        if pending is None:
            raise Stream3Dv2LiveError("SAM3 result has no bound live input")
        source_frame_id = int(pending["frame_id"])
        if source_frame_id > ready_frame_id:
            raise Stream3Dv2LiveError("SAM3 result became ready before its source frame")
        view = SAM3MemoryTeacherView(
            frame_id=source_frame_id,
            intrinsics=pending["intrinsics"],
            camera_to_world=pending["camera_to_world"],
            depth_m=pending["depth_m"],
            masks_packbits=result.masks_packbits,
            scores=result.scores,
            labels=np.asarray(result.labels, dtype=str),
            image_shape=result.image_shape,
        )
        self._semantic_views.append(
            _SemanticAvailability(view=view, ready_frame_id=int(ready_frame_id))
        )
        self._semantic_views.sort(key=lambda row: (row.view.frame_id, row.ready_frame_id))
        if len(self._semantic_views) > self.config.max_semantic_views:
            self._semantic_views = self._semantic_views[-self.config.max_semantic_views :]
        self._counts["sam3_results_accepted"] += 1
        self._counts["sam3_proposals"] += result.count

    def poll(self, raw_frame_id: int) -> None:
        """Poll the single async slot without waiting on one raw frame."""

        if self._finalized:
            raise Stream3Dv2LiveError("live route is already finalized")
        current = int(raw_frame_id)
        if current < self._last_raw_frame_id:
            raise Stream3Dv2LiveError("raw frame IDs must be monotonic")
        self._last_raw_frame_id = current
        started = time.perf_counter()
        result = self._sam3.poll(0.0)
        self._stage_ms["sam3_poll"].append((time.perf_counter() - started) * 1000.0)
        if result is not None:
            self._accept_sam3_result(result, current)
        elif not self._sam3.pending and self._sam_pending_inputs:
            # The client consumed and deliberately dropped a late/error result.
            self._counts["sam3_result_drops"] += len(self._sam_pending_inputs)
            self._sam_pending_inputs.clear()

    def _commit_abstained_keyframe(
        self,
        *,
        frame: int,
        ordinal: int,
        reason: str,
        frame_started: float,
    ) -> None:
        """Advance causal clocks without adapting an unsupported image frame."""

        stage = time.perf_counter()
        f3_query = self._f3.query(
            frame,
            ordinal,
            (),
            max_logical_accessed_ordinal=ordinal,
        )
        f3_commit = self._f3.commit(f3_query)
        live_ids = set(self._state.live_track_ids)
        retire = tuple(
            str(value) for value in f3_commit.retired_track_ids if str(value) in live_ids
        )
        state_query, _ = self._state.process_frame(
            frame_id=frame,
            frame_ordinal=ordinal,
            updates=(),
            retire_track_ids=retire,
        )
        self._finalized_tracks.extend(state_query.retired)
        self._stage_ms["f3_and_memory"].append(
            (time.perf_counter() - stage) * 1000.0
        )
        total_ms = (time.perf_counter() - frame_started) * 1000.0
        self._frame_total_ms.append(total_ms)
        self._counts["keyframes"] += 1
        self._counts[f"abstain_{reason}"] += 1
        self._counts["deadline_misses"] += int(
            total_ms > self.config.keyframe_deadline_ms
        )
        self._last_keyframe_id = frame
        self._last_keyframe_ordinal = ordinal
        self._record_cuda_peak()

    def _process_lightweight_discovery(
        self,
        *,
        scene_id: str,
        frame: int,
        ordinal: int,
        image: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        camera_to_world: np.ndarray,
        native_boxes: np.ndarray,
        fastsam: Any,
    ) -> None:
        """Run the bounded V2 path while preserving its terminal selector.

        F3 sees every preselected F0 observation before current evidence is
        committed.  Only the highest-ranked *distinct tracks* proceed to F2
        and F4, so the expensive Boxer pass is bounded without introducing
        Stream3Dv3's five-view acceptance gate.
        """

        stage = time.perf_counter()
        preselection = preselect_fastsam_masks(
            masks=fastsam.masks,
            confidences=fastsam.confidences,
            boxes_xyxy=fastsam.boxes_xyxy,
            depth_m=depth,
            native_boxes_xyxy=native_boxes,
            box_shortlist=self.config.fastsam_box_shortlist,
            mask_cap=self.config.fastsam_top_k,
        )
        selected_indices = preselection.original_indices
        self._stage_ms["mask_preselection"].append(
            (time.perf_counter() - stage) * 1000.0
        )
        self._counts["box_shortlisted"] += preselection.box_shortlist_count
        self._counts["masks_prelift"] += len(selected_indices)

        stage = time.perf_counter()
        f0 = select_and_lift_residual_masks(
            masks=np.asarray(fastsam.masks[selected_indices]),
            confidences=np.asarray(fastsam.confidences[selected_indices]),
            depth_m=depth,
            explained_boxes_xyxy=native_boxes,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
        )
        self._stage_ms["f0_residual_lift"].append(
            (time.perf_counter() - stage) * 1000.0
        )
        self._counts["f0_candidates"] += len(f0.candidates)

        candidates_by_source: dict[str, Any] = {}
        f3_observations = []
        for candidate in f0.candidates:
            original_index = int(selected_indices[candidate.raw_index])
            source_id = (
                f"{scene_id}/frame_{frame:06d}/raw_{original_index:03d}"
            )
            candidates_by_source[source_id] = candidate
            f3_observations.append(
                make_observation(
                    source_id=source_id,
                    frame_id=frame,
                    frame_ordinal=ordinal,
                    confidence=candidate.confidence,
                    world_q02=candidate.world_q02,
                    world_q98=candidate.world_q98,
                    voxel_keys=_bounded_f3_keys(candidate.points_world),
                    camera_to_world=camera_to_world,
                    intrinsics=intrinsics,
                    mask=fastsam.masks[original_index],
                )
            )

        f3_stage = time.perf_counter()
        f3_query = self._f3.query(
            frame,
            ordinal,
            f3_observations,
            max_logical_accessed_ordinal=ordinal,
        )
        f3_commit = self._f3.commit(f3_query)
        assignments = {
            row.source_id: str(row.track_id)
            for row in f3_commit.assignments
            if row.track_id is not None
        }
        self._stage_ms["f3_association"].append(
            (time.perf_counter() - f3_stage) * 1000.0
        )

        ranked_tracks: list[tuple[tuple[float, ...], str, str]] = []
        for source_id, track_id in assignments.items():
            candidate = candidates_by_source[source_id]
            quality = float(candidate.confidence) * math.sqrt(
                max(
                    float(candidate.residual_ratio)
                    * float(candidate.valid_ratio),
                    0.0,
                )
            )
            ranked_tracks.append(
                (
                    (
                        quality,
                        float(candidate.confidence),
                        float(candidate.residual_ratio),
                        float(candidate.valid_ratio),
                        -float(candidate.raw_index),
                    ),
                    source_id,
                    track_id,
                )
            )
        ranked_tracks.sort(key=lambda row: (row[0], row[1]), reverse=True)
        selected_tracks: set[str] = set()
        selected_sources: list[str] = []
        for _, source_id, track_id in ranked_tracks:
            if track_id in selected_tracks:
                continue
            selected_tracks.add(track_id)
            selected_sources.append(source_id)
            if len(selected_sources) >= self.config.f4_top_m_tracks:
                break
        self._counts["f4_track_shortlist"] += len(selected_sources)
        self._counts["f4_track_dropped"] += max(
            len(assignments) - len(selected_sources), 0
        )

        stage_f2 = time.perf_counter()
        sources_to_refine = (
            selected_sources
            if self.config.conditional_f2
            else list(candidates_by_source)
        )
        refined_by_source = {
            source_id: refine_fastsam_candidate(
                points_world=candidates_by_source[source_id].points_world,
                world_q02=candidates_by_source[source_id].world_q02,
                world_q98=candidates_by_source[source_id].world_q98,
                voxel_keys=candidates_by_source[source_id].voxel_keys,
            )
            for source_id in sources_to_refine
        }
        self._stage_ms["f2_refine"].append(
            (time.perf_counter() - stage_f2) * 1000.0
        )
        self._counts["f2_candidates"] += len(refined_by_source)

        rows_by_source: dict[str, Any] = {}
        if selected_sources:
            stage_f4 = time.perf_counter()
            selected_boxes = np.stack(
                [
                    candidates_by_source[source_id].tight_box_xyxy
                    for source_id in selected_sources
                ]
            ).astype(np.float32)
            f4 = self._ensure_f4().infer_batch(
                scene_id,
                frame,
                image,
                depth,
                intrinsics,
                camera_to_world,
                selected_boxes,
                tuple(selected_sources),
            )
            self._stage_ms["f4_boxer"].append(
                (time.perf_counter() - stage_f4) * 1000.0
            )
            self._counts["f4_batches"] += 1
            self._counts["f4_valid"] += f4.diagnostics.valid_count
            self._counts["f4_invalid"] += f4.diagnostics.invalid_count
            rows_by_source = {row.source_id: row for row in f4.rows}
        else:
            self._stage_ms["f4_boxer"].append(0.0)

        updates: list[TrackUpdate] = []
        for source_id in selected_sources:
            hb = rows_by_source[source_id]
            if not hb.valid or hb.world_corners is None:
                continue
            candidate = candidates_by_source[source_id]
            f2 = refined_by_source[source_id]
            local_indices = f2.hlg.retained_indices
            points = (
                candidate.points_world[local_indices]
                if len(local_indices)
                else candidate.points_world
            )
            updates.append(
                TrackUpdate(
                    track_id=assignments[source_id],
                    view=TrackView(
                        source_id=source_id,
                        frame_id=frame,
                        frame_ordinal=ordinal,
                        mask_confidence=float(candidate.confidence),
                        hb_confidence=float(
                            0.0 if hb.confidence is None else hb.confidence
                        ),
                        points_world=np.ascontiguousarray(points, dtype=np.float64),
                        hb_corners=hb.world_corners,
                    ),
                )
            )

        state_stage = time.perf_counter()
        live_ids = set(self._state.live_track_ids)
        retire = tuple(
            str(value)
            for value in f3_commit.retired_track_ids
            if str(value) in live_ids
        )
        state_query, _ = self._state.process_frame(
            frame_id=frame,
            frame_ordinal=ordinal,
            updates=updates,
            retire_track_ids=retire,
        )
        self._finalized_tracks.extend(state_query.retired)
        self._stage_ms["f3_and_memory"].append(
            (time.perf_counter() - state_stage) * 1000.0
        )

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
    ) -> NativeTargetMaskFrame | None:
        """Commit one causal update and expose its already-run FastSAM frame.

        The return value is evidence-only and cannot affect online native
        association.  It is consumed immediately by the terminal-only
        TM-FPF-C1 collector in :mod:`demo`; callers that do not enable that
        refiner may continue to ignore it.
        """

        if self._finalized:
            raise Stream3Dv2LiveError("live route is already finalized")
        self.bind_scene(scene_id)
        frame = int(frame_id)
        if frame <= self._last_keyframe_id:
            raise Stream3Dv2LiveError("keyframe IDs must be strictly increasing")
        self.poll(frame)
        ordinal = self._last_keyframe_ordinal + 1
        frame_started = time.perf_counter()
        raw_image = np.asarray(rgb)
        if raw_image.shape == (640, 480, 3):
            # Match the sealed F0 production protocol: two known ScanNet
            # producer keyframes are non-upright and use a different image
            # coordinate frame.  Rotating RGB/depth/boxes here would invent a
            # new proposal policy, so the live branch makes an explicit causal
            # empty commit while native BoxFusion remains untouched.
            self._commit_abstained_keyframe(
                frame=frame,
                ordinal=ordinal,
                reason="non_upright_producer_frame",
                frame_started=frame_started,
            )
            return None
        image = _rgb_uint8(rgb)
        depth = _depth_float(depth_m)
        K = _matrix(intrinsics, (3, 3), "intrinsics")
        pose = _matrix(camera_to_world, (4, 4), "camera_to_world")
        native_boxes = _boxes_xyxy(native_boxes_xyxy)
        trigger = None
        if self._depth_trigger is not None:
            stage = time.perf_counter()
            trigger = self._depth_trigger.query(
                frame_id=frame,
                frame_ordinal=ordinal,
                depth_m=depth,
                intrinsics=K,
                camera_to_world=pose,
                native_boxes_xyxy=native_boxes,
            )
            self._stage_ms["depth_trigger"].append(
                (time.perf_counter() - stage) * 1000.0
            )
            if not trigger.run_discovery:
                self._depth_trigger.commit(trigger)
                self._commit_abstained_keyframe(
                    frame=frame,
                    ordinal=ordinal,
                    reason=f"depth_trigger_{trigger.reason}",
                    frame_started=frame_started,
                )
                return None
            self._counts["depth_trigger_runs"] += 1
        stage = time.perf_counter()
        fastsam = self._ensure_fastsam().infer_bgr(image[..., ::-1].copy())
        self._stage_ms["fastsam"].append((time.perf_counter() - stage) * 1000.0)
        self._counts["fastsam_masks"] += fastsam.count

        if self.config.lightweight_enabled:
            self._process_lightweight_discovery(
                scene_id=scene_id,
                frame=frame,
                ordinal=ordinal,
                image=image,
                depth=depth,
                intrinsics=K,
                camera_to_world=pose,
                native_boxes=native_boxes,
                fastsam=fastsam,
            )
            if trigger is not None:
                assert self._depth_trigger is not None
                self._depth_trigger.commit(trigger)

            stage = time.perf_counter()
            if ordinal % self.config.sam3_interval_keyframes == 0:
                request_id = self._sam3.submit(
                    image,
                    context={
                        "scene_id": scene_id,
                        "frame_id": frame,
                        "frame_ordinal": ordinal,
                    },
                )
                if request_id is not None:
                    self._sam_pending_inputs[request_id] = {
                        "frame_id": frame,
                        "depth_m": depth.copy(),
                        "intrinsics": K.copy(),
                        "camera_to_world": pose.copy(),
                    }
                    self._counts["sam3_submitted"] += 1
                else:
                    self._counts["sam3_submit_drops"] += 1
            self._stage_ms["sam3_submit"].append(
                (time.perf_counter() - stage) * 1000.0
            )

            total_ms = (time.perf_counter() - frame_started) * 1000.0
            self._frame_total_ms.append(total_ms)
            self._counts["keyframes"] += 1
            self._counts["deadline_misses"] += int(
                total_ms > self.config.keyframe_deadline_ms
            )
            self._last_keyframe_id = frame
            self._last_keyframe_ordinal = ordinal
            self._record_cuda_peak()
            return NativeTargetMaskFrame(
                scene_id=str(scene_id),
                frame_id=frame,
                native_boxes_xyxy=native_boxes,
                masks=np.asarray(fastsam.masks),
                automatic_boxes_xyxy=np.asarray(fastsam.boxes_xyxy),
                automatic_confidences=np.asarray(fastsam.confidences),
                depth_m=depth,
                intrinsics=K,
                camera_to_world=pose,
            )

        stage = time.perf_counter()
        f0 = select_and_lift_residual_masks(
            masks=fastsam.masks,
            confidences=fastsam.confidences,
            depth_m=depth,
            explained_boxes_xyxy=native_boxes,
            intrinsics=K,
            camera_to_world=pose,
        )
        self._stage_ms["f0_residual_lift"].append(
            (time.perf_counter() - stage) * 1000.0
        )
        self._counts["f0_candidates"] += len(f0.candidates)

        stage = time.perf_counter()
        refined = [
            refine_fastsam_candidate(
                points_world=row.points_world,
                world_q02=row.world_q02,
                world_q98=row.world_q98,
                voxel_keys=row.voxel_keys,
            )
            for row in f0.candidates
        ]
        self._stage_ms["f2_refine"].append((time.perf_counter() - stage) * 1000.0)

        source_ids = tuple(
            f"{scene_id}/frame_{frame:06d}/raw_{row.raw_index:03d}"
            for row in f0.candidates
        )
        boxes = (
            np.stack([row.tight_box_xyxy for row in f0.candidates]).astype(np.float32)
            if f0.candidates
            else np.empty((0, 4), dtype=np.float32)
        )
        stage = time.perf_counter()
        f4 = self._ensure_f4().infer_batch(
            scene_id,
            frame,
            image,
            depth,
            K,
            pose,
            boxes,
            source_ids,
        )
        self._stage_ms["f4_boxer"].append((time.perf_counter() - stage) * 1000.0)
        self._counts["f4_valid"] += f4.diagnostics.valid_count
        self._counts["f4_invalid"] += f4.diagnostics.invalid_count

        stage = time.perf_counter()
        observations = []
        retained_points: dict[str, np.ndarray] = {}
        for source_id, candidate, f2 in zip(source_ids, f0.candidates, refined):
            local_indices = f2.hlg.retained_indices
            points = (
                candidate.points_world[local_indices]
                if len(local_indices)
                else candidate.points_world
            )
            retained_points[source_id] = np.ascontiguousarray(points, dtype=np.float64)
            observations.append(
                make_observation(
                    source_id=source_id,
                    frame_id=frame,
                    frame_ordinal=ordinal,
                    confidence=candidate.confidence,
                    world_q02=candidate.world_q02,
                    world_q98=candidate.world_q98,
                    voxel_keys=_bounded_f3_keys(candidate.points_world),
                    camera_to_world=pose,
                    intrinsics=K,
                    mask=fastsam.masks[candidate.raw_index],
                )
            )
        f3_query = self._f3.query(
            frame,
            ordinal,
            observations,
            max_logical_accessed_ordinal=ordinal,
        )
        f3_commit = self._f3.commit(f3_query)
        assignments = {
            row.source_id: row.track_id
            for row in f3_commit.assignments
            if row.track_id is not None
        }
        rows_by_source = {row.source_id: row for row in f4.rows}
        confidence_by_source = {
            source_id: candidate.confidence
            for source_id, candidate in zip(source_ids, f0.candidates)
        }
        updates: list[TrackUpdate] = []
        for source_id in source_ids:
            track_id = assignments.get(source_id)
            hb = rows_by_source[source_id]
            if track_id is None or not hb.valid or hb.world_corners is None:
                continue
            updates.append(
                TrackUpdate(
                    track_id=str(track_id),
                    view=TrackView(
                        source_id=source_id,
                        frame_id=frame,
                        frame_ordinal=ordinal,
                        mask_confidence=float(confidence_by_source[source_id]),
                        hb_confidence=float(
                            0.0 if hb.confidence is None else hb.confidence
                        ),
                        points_world=retained_points[source_id],
                        hb_corners=hb.world_corners,
                    ),
                )
            )
        live_ids = set(self._state.live_track_ids)
        retire = tuple(
            str(value) for value in f3_commit.retired_track_ids if str(value) in live_ids
        )
        state_query, _ = self._state.process_frame(
            frame_id=frame,
            frame_ordinal=ordinal,
            updates=updates,
            retire_track_ids=retire,
        )
        self._finalized_tracks.extend(state_query.retired)
        self._stage_ms["f3_and_memory"].append(
            (time.perf_counter() - stage) * 1000.0
        )

        stage = time.perf_counter()
        if ordinal % self.config.sam3_interval_keyframes == 0:
            request_id = self._sam3.submit(
                image,
                context={
                    "scene_id": scene_id,
                    "frame_id": frame,
                    "frame_ordinal": ordinal,
                },
            )
            if request_id is not None:
                self._sam_pending_inputs[request_id] = {
                    "frame_id": frame,
                    "depth_m": depth.copy(),
                    "intrinsics": K.copy(),
                    "camera_to_world": pose.copy(),
                }
                self._counts["sam3_submitted"] += 1
            else:
                self._counts["sam3_submit_drops"] += 1
        self._stage_ms["sam3_submit"].append((time.perf_counter() - stage) * 1000.0)

        total_ms = (time.perf_counter() - frame_started) * 1000.0
        self._frame_total_ms.append(total_ms)
        self._counts["keyframes"] += 1
        self._counts["deadline_misses"] += int(
            total_ms > self.config.keyframe_deadline_ms
        )
        self._last_keyframe_id = frame
        self._last_keyframe_ordinal = ordinal
        self._record_cuda_peak()
        return NativeTargetMaskFrame(
            scene_id=str(scene_id),
            frame_id=frame,
            native_boxes_xyxy=native_boxes,
            masks=np.asarray(fastsam.masks),
            automatic_boxes_xyxy=np.asarray(fastsam.boxes_xyxy),
            automatic_confidences=np.asarray(fastsam.confidences),
            depth_m=depth,
            intrinsics=K,
            camera_to_world=pose,
        )

    def _drain_sam3(self, final_frame_id: int) -> None:
        started = time.perf_counter()
        result = self._sam3.drain(self.config.sam3_drain_timeout_seconds)
        self._stage_ms["sam3_terminal_drain"].append(
            (time.perf_counter() - started) * 1000.0
        )
        if result is not None:
            self._accept_sam3_result(result, final_frame_id)
        elif not self._sam3.pending and self._sam_pending_inputs:
            self._counts["sam3_result_drops"] += len(self._sam_pending_inputs)
            self._sam_pending_inputs.clear()
        if self._sam3.pending:
            self._counts["sam3_drain_timeouts"] += 1
            self._sam_pending_inputs.clear()
        self._sam3.close(drain=False)

    @staticmethod
    def _native(boxes: np.ndarray, scores: np.ndarray) -> NativePrediction:
        rows = [
            (0, np.ascontiguousarray(boxes[index], dtype=np.float32), float(scores[index]))
            for index in range(len(boxes))
        ]
        return NativePrediction(payload=[rows], rows=rows, corners=boxes)

    def _eligible_semantic_views(self, decision_frame_id: int) -> list[SAM3MemoryTeacherView]:
        return [
            row.view
            for row in self._semantic_views
            if row.view.frame_id <= decision_frame_id
            and row.ready_frame_id <= decision_frame_id
        ]

    def _candidate_pools(
        self,
        tracks: Sequence[FinalizedTrack],
        native: NativePrediction,
    ) -> tuple[list[Candidate], list[Candidate]]:
        births: list[Candidate] = []
        overlays: list[Candidate] = []
        assert self._scene_id is not None
        for row in tracks:
            geometry = row.geometry
            self._hypotheses[geometry.chosen_hypothesis] += 1
            native_index, overlap, nd, volume_ratio = _native_relation(
                geometry.corners, native
            )
            candidate = Candidate(
                scene=self._scene_id,
                track_id=int(row.track_id),
                geometry=geometry,
                native_index=native_index,
                native_overlap=overlap,
                native_nd=nd,
                native_volume_ratio=volume_ratio,
            )
            (births if _is_native_novel(overlap) else overlays).append(candidate)
        return births, overlays

    def record_terminal_clip(
        self,
        *,
        proposal_count: int,
        batch_count: int,
        elapsed_ms: float,
    ) -> None:
        """Attach the one-shot native semantic batch to route diagnostics."""

        if not self.config.terminal_clip_enabled:
            raise Stream3Dv2LiveError("terminal CLIP is disabled")
        if self._finalized:
            raise Stream3Dv2LiveError("terminal CLIP must run before finalization")
        proposals = int(proposal_count)
        batches = int(batch_count)
        elapsed = float(elapsed_ms)
        if proposals < 0 or batches < 0 or not math.isfinite(elapsed) or elapsed < 0.0:
            raise Stream3Dv2LiveError("invalid terminal CLIP diagnostics")
        if self._counts.get("terminal_clip_calls", 0):
            raise Stream3Dv2LiveError("terminal CLIP may run only once")
        self._counts["terminal_clip_calls"] = 1
        self._counts["terminal_clip_proposals"] = proposals
        self._counts["terminal_clip_batches"] = batches
        self._stage_ms["terminal_clip"].append(elapsed)

    def _diagnostics(
        self,
        *,
        native_count: int,
        output_count: int,
        birth_count: int,
        overlay_count: int,
        candidate_count: int,
    ) -> dict[str, Any]:
        state_stats = asdict(self._state.statistics)
        sam3_stats = self._sam3.diagnostics()
        return {
            "schema": SCHEMA,
            "complete": True,
            "scene_id": self._scene_id,
            "training_free": True,
            "gt_access": False,
            "annotation_access": False,
            "evaluator_access": False,
            "proposal_cache_access": False,
            "teacher_cache_access": False,
            "terminal_cache_access": False,
            "past_only": True,
            "query_before_commit": True,
            "terminal_output_decision": True,
            "native_scores_preserved": True,
            "native_score_lower_bound": self.config.native_score_lower_bound,
            "lightweight": {
                "enabled": self.config.lightweight_enabled,
                "depth_trigger_enabled": self.config.depth_trigger_enabled,
                "fastsam_box_shortlist": self.config.fastsam_box_shortlist,
                "fastsam_top_k": self.config.fastsam_top_k,
                "conditional_f2": self.config.conditional_f2,
                "f4_top_m_tracks": self.config.f4_top_m_tracks,
                "terminal_clip_enabled": self.config.terminal_clip_enabled,
                "terminal_clip_batch_size": self.config.terminal_clip_batch_size,
            },
            "bounded": {
                "sam3_queue_capacity": 1,
                "state_max_tracks": 1024,
                "state_max_views_per_track": 5,
                "semantic_views": self.config.max_semantic_views,
                "fastsam_candidates_per_keyframe": (
                    self.config.fastsam_top_k
                    if self.config.lightweight_enabled
                    else 16
                ),
                "f4_tracks_per_keyframe": (
                    self.config.f4_top_m_tracks
                    if self.config.lightweight_enabled
                    else 16
                ),
            },
            "depth_trigger": (
                {"enabled": False}
                if self._depth_trigger is None
                else {
                    "enabled": True,
                    **dict(self._depth_trigger.summary()),
                }
            ),
            "counts": {
                **{key: int(value) for key, value in sorted(self._counts.items())},
                "native": int(native_count),
                "candidates": int(candidate_count),
                "births": int(birth_count),
                "overlays": int(overlay_count),
                "output": int(output_count),
                "semantic_views": len(self._semantic_views),
            },
            "timing_ms": {
                "keyframe_total": _quantiles(self._frame_total_ms),
                **{
                    key: _quantiles(values)
                    for key, values in sorted(self._stage_ms.items())
                },
            },
            "deadline_ms": self.config.keyframe_deadline_ms,
            "state": state_stats,
            "sam3": sam3_stats,
            "chosen_hypotheses": dict(sorted(self._hypotheses.items())),
            "peak_cuda_allocated_bytes": self._peak_cuda_allocated_bytes,
            "peak_cuda_reserved_bytes": self._peak_cuda_reserved_bytes,
            "future_access_count": int(state_stats["future_access_count"]),
            "late_result_count": int(sam3_stats["late_count"]),
        }

    def finalize(
        self,
        *,
        native_boxes_3d: object,
        native_scores: object,
        final_frame_id: int,
    ) -> LiveTerminalResult:
        """Drain once, freeze past state and form the real-score terminal map."""

        if self._finalized:
            raise Stream3Dv2LiveError("live route finalize may run only once")
        self._finalized = True
        final_frame = int(final_frame_id)
        boxes = np.ascontiguousarray(np.asarray(native_boxes_3d, dtype=np.float64))
        scores = np.ascontiguousarray(np.asarray(native_scores, dtype=np.float64))
        if boxes.shape != (len(scores), 8, 3) or not np.isfinite(boxes).all():
            raise Stream3Dv2LiveError("terminal native boxes/scores are misaligned")
        if not np.isfinite(scores).all() or np.any(scores <= 0.0):
            raise Stream3Dv2LiveError("terminal native scores must be finite and positive")

        self._drain_sam3(final_frame)
        f3_terminal = self._f3.finalize()
        if f3_terminal.max_logical_accessed_ordinal is not None and (
            f3_terminal.max_logical_accessed_ordinal > self._last_keyframe_ordinal
        ):
            raise Stream3Dv2LiveError("F3 terminal seal accessed a future ordinal")
        state_terminal = self._state.finalize()
        tracks = tuple(self._finalized_tracks) + state_terminal.tracks
        native = self._native(boxes, scores)
        birth_pool, overlay_pool = self._candidate_pools(tracks, native)
        birth_preselected = sorted(
            birth_pool, key=lambda row: row.pre_rank, reverse=True
        )[:PRESELECT_BIRTHS_PER_SCENE]
        overlay_preselected = sorted(
            overlay_pool, key=lambda row: row.pre_rank, reverse=True
        )[:PRESELECT_OVERLAYS_PER_SCENE]
        for candidate in birth_preselected + overlay_preselected:
            views = self._eligible_semantic_views(
                candidate.geometry.decision_frame_id
            )
            _semantic_enrich(candidate, views, SAM3BirthConfig())
        births, _ = _select_births(birth_preselected)

        overlays: list[Candidate] = []
        used_native: set[int] = set()
        for candidate in sorted(
            overlay_preselected, key=lambda row: row.final_rank, reverse=True
        ):
            safe, _ = _overlay_safe(candidate, native)
            if (
                safe
                and candidate.native_index is not None
                and candidate.native_index not in used_native
                and len(overlays) < MAX_OVERLAYS_PER_SCENE
            ):
                overlays.append(candidate)
                used_native.add(candidate.native_index)

        if births:
            if not len(scores):
                births = []
            else:
                score_floor = min(
                    float(scores.min()), self.config.native_score_lower_bound
                )
                _score_births(births, score_floor)
        output_boxes = boxes.copy()
        output_scores = scores.copy()
        for candidate in overlays:
            assert candidate.native_index is not None
            output_boxes[candidate.native_index] = candidate.geometry.corners
        if births:
            ordered = sorted(
                births, key=lambda row: float(row.append_score), reverse=True
            )
            output_boxes = np.concatenate(
                (
                    output_boxes,
                    np.stack([row.geometry.corners for row in ordered]),
                ),
                axis=0,
            )
            output_scores = np.concatenate(
                (
                    output_scores,
                    np.asarray([row.append_score for row in ordered], dtype=np.float64),
                )
            )
        if len(scores) and births and not np.all(output_scores[len(scores) :] < scores.min()):
            raise Stream3Dv2LiveError("appended score suffix is not below every native score")
        if not np.array_equal(output_scores[: len(scores)], scores):
            raise Stream3Dv2LiveError("live route changed a native score or order")

        self._record_cuda_peak()
        diagnostics = self._diagnostics(
            native_count=len(scores),
            output_count=len(output_scores),
            birth_count=len(births),
            overlay_count=len(overlays),
            candidate_count=len(tracks),
        )
        if self.config.diagnostics_root is not None:
            assert self._scene_id is not None
            _atomic_json(
                Path(self.config.diagnostics_root) / f"{self._scene_id}.json",
                diagnostics,
            )
        return LiveTerminalResult(
            boxes_3d=np.ascontiguousarray(output_boxes, dtype=np.float32),
            scores=np.ascontiguousarray(output_scores, dtype=np.float32),
            birth_count=len(births),
            overlay_count=len(overlays),
            diagnostics=diagnostics,
        )

    def close(self) -> None:
        self._sam3.close(drain=False)


def build_stream3dv2_live_route(
    cfg: Mapping[str, Any],
    *,
    lifting_adapter: Any,
    device: str,
) -> Stream3Dv2LiveRoute | None:
    config = Stream3Dv2LiveConfig.from_mapping(cfg.get("online_stream3dv2", {}))
    if not config.enabled:
        return None
    return Stream3Dv2LiveRoute(
        config,
        lifting_adapter=lifting_adapter,
        device=device,
    )


__all__ = [
    "LiveTerminalResult",
    "NativeTargetMaskFrame",
    "SCHEMA",
    "Stream3Dv2LiveConfig",
    "Stream3Dv2LiveError",
    "Stream3Dv2LiveRoute",
    "build_stream3dv2_live_route",
    "tm_fpf_c1_view_abstention_reason",
]
