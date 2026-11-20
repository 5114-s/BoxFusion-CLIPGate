"""Causal dynamic-object branch for the released BoxFusion pipeline.

The branch is intentionally downstream of native 3D NMS and 2D matching.  It
does not replace their association decisions or append proposal rows.  Instead
it observes one representative proposal per native row, maintains a bounded
dynamic object state, freezes confirmed moving rows before all-history PFO, and
can materialize a separate current-state output at any processed prefix.

No learned component is introduced.  Optional appearance descriptors reuse
features already computed by BoxFusion's frozen CLIP model.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
import json
import math
from pathlib import Path
import re
import time
from typing import Mapping, Sequence

import numpy as np

from boxfusion.causal_dynamic_objects import (
    BoxObservation,
    CausalDynamicObjectMap,
    DynamicFrameCommit,
    DynamicObjectConfig,
    Lifecycle,
    MotionState,
    ObjectSnapshot,
    TrackVisibility,
    Visibility,
    resolve_causal_dynamic_object_config,
)


SCHEMA = "boxfusion.causal_dynamic_branch.v1"
_SAFE_SCENE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class DynamicBranchContractError(ValueError):
    """Raised before native BoxFusion state is mutated."""


class DynamicBranchMode(str, Enum):
    DISABLED = "disabled"
    SHADOW = "shadow"
    ACTIVE = "active"


@dataclass(frozen=True)
class CausalDynamicBranchConfig:
    mode: DynamicBranchMode = DynamicBranchMode.DISABLED
    events_root: str | None = None
    current_output_root: str | None = None
    use_appearance: bool = True
    max_aliases_per_track: int = 16
    max_recent_events: int = 64
    max_event_records: int = 4096
    max_visibility_tracks: int = 256
    max_visibility_rays: int = 25
    min_depth_m: float = 0.05
    max_depth_m: float = 12.0
    projection_margin_pixels: int = 4
    occlusion_margin_m: float = 0.12
    free_space_margin_m: float = 0.15
    min_visibility_depth_samples: int = 3
    min_occlusion_ratio: float = 0.60
    min_free_space_ratio: float = 0.60

    def __post_init__(self) -> None:
        try:
            mode = DynamicBranchMode(self.mode)
        except ValueError as error:
            raise DynamicBranchContractError(
                "causal_dynamic_branch.mode must be disabled, shadow, or active"
            ) from error
        object.__setattr__(self, "mode", mode)
        if not isinstance(self.use_appearance, (bool, np.bool_)):
            raise DynamicBranchContractError("use_appearance must be boolean")
        for name in (
            "max_aliases_per_track",
            "max_recent_events",
            "max_event_records",
            "max_visibility_tracks",
            "max_visibility_rays",
            "min_visibility_depth_samples",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) < 1
            ):
                raise DynamicBranchContractError(f"{name} must be a positive integer")
        if self.max_aliases_per_track > 64:
            raise DynamicBranchContractError("max_aliases_per_track must be <= 64")
        if self.max_recent_events > 256:
            raise DynamicBranchContractError("max_recent_events must be <= 256")
        if self.max_visibility_rays > 81:
            raise DynamicBranchContractError("max_visibility_rays must be <= 81")
        if self.min_visibility_depth_samples > self.max_visibility_rays:
            raise DynamicBranchContractError(
                "min_visibility_depth_samples exceeds visibility ray capacity"
            )
        for name in (
            "min_depth_m",
            "max_depth_m",
            "occlusion_margin_m",
            "free_space_margin_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise DynamicBranchContractError(f"{name} must be finite and positive")
        if self.max_depth_m <= self.min_depth_m:
            raise DynamicBranchContractError("max_depth_m must exceed min_depth_m")
        if (
            isinstance(self.projection_margin_pixels, (bool, np.bool_))
            or not isinstance(self.projection_margin_pixels, (int, np.integer))
            or self.projection_margin_pixels < 0
        ):
            raise DynamicBranchContractError(
                "projection_margin_pixels must be a non-negative integer"
            )
        for name in ("min_occlusion_ratio", "min_free_space_ratio"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise DynamicBranchContractError(f"{name} must be in [0, 1]")
        for name in ("events_root", "current_output_root"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise DynamicBranchContractError(f"{name} must be null or non-empty")
        if mode != DynamicBranchMode.DISABLED and self.events_root is None:
            raise DynamicBranchContractError(
                "enabled dynamic branch requires an explicit events_root"
            )
        if mode == DynamicBranchMode.ACTIVE and self.current_output_root is None:
            raise DynamicBranchContractError(
                "active dynamic branch requires current_output_root"
            )


def resolve_causal_dynamic_branch_config(
    cfg: Mapping | None,
) -> CausalDynamicBranchConfig:
    if cfg is None:
        return CausalDynamicBranchConfig()
    if not isinstance(cfg, Mapping):
        raise DynamicBranchContractError("config must be a mapping")
    section = cfg.get("causal_dynamic_branch", {})
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        raise DynamicBranchContractError("causal_dynamic_branch must be a mapping")
    known = set(CausalDynamicBranchConfig.__dataclass_fields__)
    unknown = sorted(set(section) - known)
    if unknown:
        raise DynamicBranchContractError(
            "unknown causal_dynamic_branch option(s): " + ", ".join(unknown)
        )
    return CausalDynamicBranchConfig(**dict(section))


def _numpy(value, *, dtype=np.float64) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


def _readonly(value, *, dtype=np.float64) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _rotation_yaw(rotation: np.ndarray) -> float:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise DynamicBranchContractError("box rotation must be a finite 3x3 matrix")
    return float(math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))


def _wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _corners_from_params(
    parameters: np.ndarray, rotations: np.ndarray
) -> np.ndarray:
    parameters = np.asarray(parameters, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    if parameters.ndim != 2 or parameters.shape[1] != 6:
        raise DynamicBranchContractError("box parameters must have shape (N, 6)")
    if rotations.shape != (parameters.shape[0], 3, 3):
        raise DynamicBranchContractError("rotations must have shape (N, 3, 3)")
    count = parameters.shape[0]
    local = np.zeros((count, 3, 8), dtype=np.float64)
    length = parameters[:, 3:4]
    height = parameters[:, 4:5]
    width = parameters[:, 5:6]
    local[:, 0, [0, 3, 4, 7]] = -length / 2.0
    local[:, 0, [1, 2, 5, 6]] = length / 2.0
    local[:, 1, [0, 1, 4, 5]] = -height / 2.0
    local[:, 1, [2, 3, 6, 7]] = height / 2.0
    local[:, 2, [0, 1, 2, 3]] = -width / 2.0
    local[:, 2, [4, 5, 6, 7]] = width / 2.0
    corners = np.einsum("nij,njk->nik", rotations, local)
    corners += parameters[:, :3, None]
    return np.transpose(corners, (0, 2, 1))


@dataclass(frozen=True)
class DynamicBranchFrameResult:
    source_frame_id: int
    frame_ordinal: int
    commit: DynamicFrameCommit | None
    observation_to_track: tuple[tuple[str, str], ...]
    dynamic_track_ids: tuple[str, ...]
    reactivated_track_ids: tuple[str, ...]
    relocated_track_ids: tuple[str, ...]
    evicted_track_ids: tuple[str, ...]
    visibility_counts: tuple[tuple[str, int], ...]
    elapsed_ms: float


@dataclass(frozen=True)
class CurrentMaterialization:
    corners: np.ndarray
    scores: np.ndarray
    mask: np.ndarray
    row_track_ids: tuple[str | None, ...]
    modified_rows: tuple[int, ...]
    dropped_rows: tuple[int, ...]


class CausalDynamicBranch:
    """Pipeline adapter around :class:`CausalDynamicObjectMap`."""

    def __init__(
        self,
        cfg: Mapping | None,
        *,
        scene_id: str,
    ) -> None:
        self.config = resolve_causal_dynamic_branch_config(cfg)
        if not isinstance(scene_id, str) or not _SAFE_SCENE_ID.fullmatch(scene_id):
            raise DynamicBranchContractError("scene_id is empty or unsafe")
        self.scene_id = scene_id
        state_config = resolve_causal_dynamic_object_config(cfg)
        state_config = replace(state_config, enabled=self.enabled)
        self.state = CausalDynamicObjectMap(state_config)
        self._next_ordinal = 0
        self._last_source_frame_id = -1
        self._aliases: dict[str, deque[int]] = {}
        self._init_to_track: dict[int, str] = {}
        self._latest_init: dict[str, int] = {}
        self._latest_rotation: dict[str, np.ndarray] = {}
        self._latest_observation_yaw: dict[str, float] = {}
        self._persistent_rotation: dict[str, np.ndarray] = {}
        self._persistent_observation_yaw: dict[str, float] = {}
        self._persistent_box7: dict[str, np.ndarray] = {}
        self._latest_source_frame: dict[str, int] = {}
        self._dynamic_track_ids: set[str] = set()
        self._recent_events: deque[dict] = deque(
            maxlen=self.config.max_recent_events
        )
        self._event_records = 0
        self._event_records_dropped = 0
        self._closed = False
        self._timings_ms: deque[float] = deque(
            maxlen=self.config.max_recent_events
        )
        self._stats = {
            "keyframes": 0,
            "observations": 0,
            "births": 0,
            "reactivations": 0,
            "relocations": 0,
            "evictions": 0,
            "capacity_drops": 0,
            "static_fusion_bypasses": 0,
            "persistent_modified_rows": 0,
            "persistent_dropped_rows": 0,
            "current_modified_rows": 0,
            "current_dropped_rows": 0,
        }
        self._event_path: Path | None = None
        if self.enabled:
            root = Path(self.config.events_root).expanduser()
            root.mkdir(parents=True, exist_ok=True)
            self._event_path = root / f"{self.scene_id}.jsonl"
            # Each scene invocation owns its audit artifact.  Opening once in
            # write mode prevents stale suffixes from a shorter rerun.
            self._event_handle = self._event_path.open("w", encoding="utf-8")
        else:
            self._event_handle = None

    @property
    def enabled(self) -> bool:
        return self.config.mode != DynamicBranchMode.DISABLED

    @property
    def active(self) -> bool:
        return self.config.mode == DynamicBranchMode.ACTIVE

    @property
    def needs_appearance(self) -> bool:
        return self.enabled and self.config.use_appearance

    @property
    def event_path(self) -> Path | None:
        return self._event_path

    @property
    def recent_events(self) -> tuple[dict, ...]:
        return tuple(dict(row) for row in self._recent_events)

    @property
    def dynamic_fusion_ids(self) -> frozenset[int]:
        result = set()
        for track_id in self._dynamic_track_ids:
            result.update(self._aliases.get(track_id, ()))
        return frozenset(result)

    def _write_event(self, event: Mapping) -> None:
        row = dict(event)
        self._recent_events.append(row)
        if self._event_handle is None:
            return
        is_summary = row.get("type") == "summary"
        limit = self.config.max_event_records if is_summary else max(
            self.config.max_event_records - 1, 0
        )
        if self._event_records >= limit:
            self._event_records_dropped += 1
            return
        self._event_handle.write(json.dumps(row, sort_keys=True) + "\n")
        self._event_handle.flush()
        self._event_records += 1

    def _register_alias(self, track_id: str, init_id: int) -> None:
        aliases = self._aliases.setdefault(
            track_id, deque(maxlen=self.config.max_aliases_per_track)
        )
        if init_id in aliases:
            return
        if len(aliases) == aliases.maxlen:
            removed = aliases[0]
            if self._init_to_track.get(removed) == track_id:
                del self._init_to_track[removed]
        aliases.append(init_id)
        self._init_to_track[init_id] = track_id

    def _remove_track_cache(self, track_id: str) -> None:
        for init_id in self._aliases.pop(track_id, ()):
            if self._init_to_track.get(init_id) == track_id:
                del self._init_to_track[init_id]
        self._latest_init.pop(track_id, None)
        self._latest_rotation.pop(track_id, None)
        self._latest_observation_yaw.pop(track_id, None)
        self._persistent_rotation.pop(track_id, None)
        self._persistent_observation_yaw.pop(track_id, None)
        self._persistent_box7.pop(track_id, None)
        self._latest_source_frame.pop(track_id, None)
        self._dynamic_track_ids.discard(track_id)

    def _all_snapshots(self) -> dict[str, ObjectSnapshot]:
        rows = {
            row.track_id: row
            for row in self.state.current_snapshots(include_tentative=True)
        }
        for row in self.state.persistent_snapshots():
            # Active/occluded objects must expose current geometry. Persistent
            # geometry only fills dormant/retired tracks absent from the
            # current view.
            rows.setdefault(row.track_id, row)
        return rows

    def _visibility_for_snapshot(
        self,
        snapshot: ObjectSnapshot,
        *,
        depth_m,
        intrinsics,
        camera_to_world,
    ) -> Visibility:
        if depth_m is None or intrinsics is None or camera_to_world is None:
            return Visibility.UNKNOWN
        depth = _numpy(depth_m)
        calibration = _numpy(intrinsics)
        transform = _numpy(camera_to_world)
        if depth.ndim != 2 or calibration.shape[0] < 3 or calibration.shape[1] < 3:
            return Visibility.UNKNOWN
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            return Visibility.UNKNOWN
        center_world = np.concatenate((snapshot.box7[:3], [1.0]))
        try:
            center_camera = np.linalg.solve(transform, center_world)[:3]
        except np.linalg.LinAlgError:
            return Visibility.UNKNOWN
        z_value = float(center_camera[2])
        if not np.isfinite(z_value) or z_value <= self.config.min_depth_m:
            return Visibility.OUT_OF_VIEW
        fx, fy = float(calibration[0, 0]), float(calibration[1, 1])
        cx, cy = float(calibration[0, 2]), float(calibration[1, 2])
        if min(fx, fy) <= 0.0 or not np.all(np.isfinite([fx, fy, cx, cy])):
            return Visibility.UNKNOWN
        u_value = fx * float(center_camera[0]) / z_value + cx
        v_value = fy * float(center_camera[1]) / z_value + cy
        margin = self.config.projection_margin_pixels
        height, width = depth.shape
        if (
            u_value < -margin
            or u_value >= width + margin
            or v_value < -margin
            or v_value >= height + margin
        ):
            return Visibility.OUT_OF_VIEW

        # A compact ray bundle around the projected center.  The half diagonal
        # deliberately over-approximates depth extent, so only decisive free
        # space can retire a track.
        radius = max(1, int(round(min(4.0, fx * float(max(snapshot.box7[3:6])) / max(z_value, 1e-6) * 0.12))))
        side = max(1, int(math.floor(math.sqrt(self.config.max_visibility_rays))))
        offsets = np.linspace(-radius, radius, side).round().astype(np.int64)
        samples = []
        center_u, center_v = int(round(u_value)), int(round(v_value))
        for y_offset in offsets:
            for x_offset in offsets:
                if len(samples) >= self.config.max_visibility_rays:
                    break
                x_coord = center_u + int(x_offset)
                y_coord = center_v + int(y_offset)
                if 0 <= x_coord < width and 0 <= y_coord < height:
                    value = float(depth[y_coord, x_coord])
                    if (
                        np.isfinite(value)
                        and self.config.min_depth_m <= value <= self.config.max_depth_m
                    ):
                        samples.append(value)
        if len(samples) < self.config.min_visibility_depth_samples:
            return Visibility.UNKNOWN
        values = np.asarray(samples, dtype=np.float64)
        half_depth = 0.5 * float(np.linalg.norm(snapshot.box7[3:6]))
        near_depth = max(self.config.min_depth_m, z_value - half_depth)
        far_depth = z_value + half_depth
        occlusion_ratio = float(
            np.mean(values < near_depth - self.config.occlusion_margin_m)
        )
        free_space_ratio = float(
            np.mean(values > far_depth + self.config.free_space_margin_m)
        )
        if occlusion_ratio >= self.config.min_occlusion_ratio:
            return Visibility.OCCLUDED
        if free_space_ratio >= self.config.min_free_space_ratio:
            return Visibility.EXPECTED_VISIBLE
        return Visibility.UNKNOWN

    def _visibility_rows(
        self, *, depth_m, intrinsics, camera_to_world
    ) -> tuple[TrackVisibility, ...]:
        snapshots = self._all_snapshots()
        priority = sorted(
            snapshots.values(),
            key=lambda row: (
                row.lifecycle in (Lifecycle.RETIRED, Lifecycle.DORMANT),
                -row.last_observed_frame,
                row.track_id,
            ),
        )[: self.config.max_visibility_tracks]
        return tuple(
            TrackVisibility(
                row.track_id,
                self._visibility_for_snapshot(
                    row,
                    depth_m=depth_m,
                    intrinsics=intrinsics,
                    camera_to_world=camera_to_world,
                ),
            )
            for row in priority
        )

    def _extract_observations(
        self,
        current_instances,
        fusion_list: Sequence[Sequence[int]],
        camera_to_world,
        appearance_by_init_id: Mapping[int, object] | None = None,
    ) -> tuple[
        tuple[BoxObservation, ...],
        dict[str, tuple[int, tuple[int, ...], int, np.ndarray, float]],
    ]:
        if current_instances is None or len(current_instances) == 0:
            return (), {}
        required = ("init_id", "scores", "pred_boxes_3d")
        missing = [name for name in required if not current_instances.has(name)]
        if missing:
            raise DynamicBranchContractError(
                "current instances lack dynamic fields: " + ", ".join(missing)
            )
        init_ids = _numpy(current_instances.init_id, dtype=np.int64).reshape(-1)
        scores = _numpy(current_instances.scores).reshape(-1)
        boxes = _numpy(current_instances.pred_boxes_3d.tensor)
        rotations = _numpy(current_instances.pred_boxes_3d.R)
        count = len(init_ids)
        if (
            scores.shape != (count,)
            or boxes.shape != (count, 6)
            or rotations.shape != (count, 3, 3)
        ):
            raise DynamicBranchContractError("current dynamic arrays are misaligned")
        if len(set(int(value) for value in init_ids)) != count:
            raise DynamicBranchContractError("current init_id values are not unique")
        if not np.all(np.isfinite(boxes)) or np.any(boxes[:, 3:6] <= 0.0):
            raise DynamicBranchContractError("current 3D boxes are invalid")

        owner: dict[int, int] = {}
        current_set = {int(value) for value in init_ids}
        for row_index, raw_ids in enumerate(fusion_list):
            for raw_id in raw_ids:
                init_id = int(raw_id)
                if init_id not in current_set:
                    continue
                if init_id in owner:
                    raise DynamicBranchContractError(
                        "one current init_id belongs to multiple native rows"
                    )
                owner[init_id] = row_index
        # Fail open for a proposal absent from native bookkeeping by giving it
        # an isolated synthetic row; it cannot suppress an existing native row.
        next_synthetic = len(fusion_list)
        groups: dict[int, list[int]] = {}
        for index, raw_id in enumerate(init_ids):
            init_id = int(raw_id)
            row = owner.get(init_id)
            if row is None:
                row = next_synthetic
                next_synthetic += 1
            groups.setdefault(row, []).append(index)

        appearances = None
        if self.config.use_appearance and current_instances.has("appearance_features"):
            appearances = _numpy(current_instances.appearance_features)
            if appearances.ndim != 2 or appearances.shape[0] != count:
                raise DynamicBranchContractError("appearance features are misaligned")
        appearance_overrides: dict[int, np.ndarray] = {}
        if appearance_by_init_id is not None:
            if not isinstance(appearance_by_init_id, Mapping):
                raise DynamicBranchContractError(
                    "appearance_by_init_id must be a mapping"
                )
            current_ids = {int(value) for value in init_ids}
            for raw_init_id, feature in appearance_by_init_id.items():
                init_id = int(raw_init_id)
                if init_id not in current_ids:
                    raise DynamicBranchContractError(
                        "appearance override references a non-current init_id"
                    )
                array = _numpy(feature).reshape(-1)
                if array.size == 0 or not np.all(np.isfinite(array)):
                    raise DynamicBranchContractError(
                        "appearance override must be a non-empty finite vector"
                    )
                appearance_overrides[init_id] = array
        camera_position = None
        if camera_to_world is not None:
            transform = _numpy(camera_to_world)
            if transform.shape == (4, 4) and np.all(np.isfinite(transform)):
                camera_position = transform[:3, 3]

        candidates = []
        metadata = {}
        for row_index in sorted(groups):
            members = groups[row_index]
            representative = max(
                members,
                key=lambda index: (float(scores[index]), -int(init_ids[index])),
            )
            init_id = int(init_ids[representative])
            observation_id = (
                f"{self.scene_id}/kf_{self._next_ordinal:06d}/init_{init_id:09d}"
            )
            box7 = np.concatenate(
                (boxes[representative], [_rotation_yaw(rotations[representative])])
            )
            view_direction = None
            if camera_position is not None:
                direction = boxes[representative, :3] - camera_position
                if np.all(np.isfinite(direction)) and np.linalg.norm(direction) > 1e-9:
                    view_direction = direction
            score = float(np.clip(scores[representative], 0.0, 1.0))
            appearance = appearance_overrides.get(init_id)
            if appearance is None and appearances is not None:
                appearance = appearances[representative]
            candidates.append(
                (
                    score,
                    init_id,
                    BoxObservation(
                        observation_id=observation_id,
                        box7=box7,
                        score=score,
                        appearance=appearance,
                        view_direction=view_direction,
                    ),
                )
            )
            metadata[observation_id] = (
                row_index,
                tuple(sorted(int(init_ids[index]) for index in members)),
                init_id,
                np.array(rotations[representative], copy=True),
                float(box7[6]),
            )

        candidates.sort(key=lambda row: (-row[0], row[1]))
        capacity = self.state.config.max_observations_per_frame
        retained = candidates[:capacity]
        retained_ids = {row[2].observation_id for row in retained}
        metadata = {key: value for key, value in metadata.items() if key in retained_ids}
        return tuple(row[2] for row in retained), metadata

    def process_keyframe(
        self,
        *,
        source_frame_id: int,
        current_instances,
        fusion_list: Sequence[Sequence[int]],
        depth_m=None,
        intrinsics=None,
        camera_to_world=None,
        appearance_by_init_id: Mapping[int, object] | None = None,
    ) -> DynamicBranchFrameResult:
        if self._closed:
            raise DynamicBranchContractError("dynamic branch is already closed")
        if isinstance(source_frame_id, (bool, np.bool_)):
            raise DynamicBranchContractError("source_frame_id must be an integer")
        source_frame = int(source_frame_id)
        if source_frame <= self._last_source_frame_id:
            raise DynamicBranchContractError("source_frame_id must be strictly increasing")
        if not self.enabled:
            self._last_source_frame_id = source_frame
            result = DynamicBranchFrameResult(
                source_frame_id=source_frame,
                frame_ordinal=self._next_ordinal,
                commit=None,
                observation_to_track=(),
                dynamic_track_ids=(),
                reactivated_track_ids=(),
                relocated_track_ids=(),
                evicted_track_ids=(),
                visibility_counts=(),
                elapsed_ms=0.0,
            )
            self._next_ordinal += 1
            return result

        started = time.perf_counter()
        observations, metadata = self._extract_observations(
            current_instances,
            fusion_list,
            camera_to_world,
            appearance_by_init_id=appearance_by_init_id,
        )
        visibility_rows = self._visibility_rows(
            depth_m=depth_m,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
        )
        query = self.state.query_frame(
            frame_ordinal=self._next_ordinal,
            observations=observations,
            visibilities=visibility_rows,
        )
        commit = self.state.commit_frame(query, token=query.token)
        for track_id in commit.evicted_track_ids:
            self._remove_track_cache(track_id)
        observed_geometry: dict[str, tuple[np.ndarray, float]] = {}
        for observation_id, track_id in commit.observation_to_track:
            (
                row_index,
                member_ids,
                representative_init_id,
                rotation,
                observation_yaw,
            ) = metadata[observation_id]
            # Keep one stable anchor per native row.  Continuing observations
            # on an already represented row do not consume alias capacity.
            row_ids = set(int(value) for value in fusion_list[row_index]) if row_index < len(fusion_list) else set(member_ids)
            known_on_row = any(
                self._init_to_track.get(init_id) == track_id for init_id in row_ids
            )
            if not known_on_row:
                self._register_alias(track_id, representative_init_id)
            self._latest_init[track_id] = representative_init_id
            self._latest_rotation[track_id] = _readonly(rotation)
            self._latest_observation_yaw[track_id] = observation_yaw
            self._latest_source_frame[track_id] = source_frame
            observed_geometry[track_id] = (rotation, observation_yaw)

        snapshots = self._all_snapshots()
        for track_id, (rotation, observation_yaw) in observed_geometry.items():
            snapshot = snapshots[track_id]
            if (
                track_id not in self._persistent_rotation
                or (
                    track_id not in self._dynamic_track_ids
                    and snapshot.motion_state != MotionState.DYNAMIC
                )
            ):
                self._persistent_rotation[track_id] = _readonly(rotation)
                self._persistent_observation_yaw[track_id] = observation_yaw
                self._persistent_box7[track_id] = self.state.track_snapshot(
                    track_id, persistent=True
                ).box7
        # Dynamic identity is a track property, not a one-frame velocity label.
        # Latch it after confirmation so a moved object that later stops cannot
        # fall back into all-history PFO or resurrect its older native aliases.
        self._dynamic_track_ids.update(
            track_id
            for track_id, row in snapshots.items()
            if row.motion_state == MotionState.DYNAMIC
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._timings_ms.append(elapsed_ms)
        self._stats["keyframes"] += 1
        self._stats["observations"] += len(observations)
        self._stats["births"] += len(commit.birth_observation_ids)
        self._stats["reactivations"] += len(commit.reactivated_track_ids)
        self._stats["relocations"] += len(commit.relocated_track_ids)
        self._stats["evictions"] += len(commit.evicted_track_ids)
        self._stats["capacity_drops"] += len(query.capacity_dropped_observation_ids)
        visibility_counts = {
            value.value: sum(row.visibility == value for row in visibility_rows)
            for value in Visibility
        }
        event = {
            "schema": SCHEMA,
            "type": "frame",
            "scene_id": self.scene_id,
            "mode": self.config.mode.value,
            "source_frame_id": source_frame,
            "frame_ordinal": self._next_ordinal,
            "query_before_commit": commit.query_before_commit,
            "maximum_accessed_frame_ordinal": query.maximum_accessed_frame_ordinal,
            "observation_to_track": [list(row) for row in commit.observation_to_track],
            "birth_observation_ids": list(commit.birth_observation_ids),
            "reactivated_track_ids": list(commit.reactivated_track_ids),
            "relocated_track_ids": list(commit.relocated_track_ids),
            "evicted_track_ids": list(commit.evicted_track_ids),
            "capacity_dropped_observation_ids": list(
                query.capacity_dropped_observation_ids
            ),
            "visibility_counts": visibility_counts,
            "tracks": [
                {
                    "track_id": row.track_id,
                    "box7": row.box7.tolist(),
                    "score": row.score,
                    "lifecycle": row.lifecycle.value,
                    "motion_state": row.motion_state.value,
                    "dynamic_probability": row.dynamic_probability,
                    "velocity4": row.velocity4.tolist(),
                    "last_observed_frame": row.last_observed_frame,
                }
                for row in sorted(snapshots.values(), key=lambda item: item.track_id)
            ],
            "elapsed_ms": elapsed_ms,
        }
        self._write_event(event)
        result = DynamicBranchFrameResult(
            source_frame_id=source_frame,
            frame_ordinal=self._next_ordinal,
            commit=commit,
            observation_to_track=commit.observation_to_track,
            dynamic_track_ids=tuple(sorted(self._dynamic_track_ids)),
            reactivated_track_ids=commit.reactivated_track_ids,
            relocated_track_ids=commit.relocated_track_ids,
            evicted_track_ids=commit.evicted_track_ids,
            visibility_counts=tuple(sorted(visibility_counts.items())),
            elapsed_ms=elapsed_ms,
        )
        self._last_source_frame_id = source_frame
        self._next_ordinal += 1
        return result

    def is_dynamic_fusion_ids(self, fusion_ids: Sequence[int]) -> bool:
        if not self.active:
            return False
        return any(int(value) in self.dynamic_fusion_ids for value in fusion_ids)

    def record_static_fusion_bypass(
        self, *, native_row_index: int, fusion_ids: Sequence[int]
    ) -> None:
        if self.active and self.is_dynamic_fusion_ids(fusion_ids):
            self._stats["static_fusion_bypasses"] += 1

    def _row_track_ids(
        self, fusion_list: Sequence[Sequence[int]]
    ) -> tuple[str | None, ...]:
        rows = []
        for fusion_ids in fusion_list:
            candidates = {
                self._init_to_track[int(value)]
                for value in fusion_ids
                if int(value) in self._init_to_track
            }
            if not candidates:
                rows.append(None)
            else:
                rows.append(
                    max(
                        candidates,
                        key=lambda track_id: (
                            self._latest_source_frame.get(track_id, -1),
                            track_id,
                        ),
                    )
                )
        return tuple(rows)

    def materialize_persistent(
        self,
        instances,
        fusion_list: Sequence[Sequence[int]],
        *,
        base_geometry=None,
    ) -> CurrentMaterialization:
        """Materialize one persistent anchor row for each latched dynamic ID."""

        geometry = instances.pred_boxes_3d if base_geometry is None else base_geometry
        parameters = _numpy(geometry.tensor)
        rotations = _numpy(geometry.R)
        scores = _numpy(instances.scores).reshape(-1)
        if (
            parameters.ndim != 2
            or parameters.shape[1] != 6
            or rotations.shape != (parameters.shape[0], 3, 3)
            or scores.shape != (parameters.shape[0],)
            or len(fusion_list) != parameters.shape[0]
        ):
            raise DynamicBranchContractError("terminal native rows are misaligned")
        row_track_ids = self._row_track_ids(fusion_list)
        output_parameters = np.array(parameters, copy=True)
        output_rotations = np.array(rotations, copy=True)
        output_scores = np.array(scores, copy=True)
        mask = np.ones(len(scores), dtype=bool)
        modified = []
        dropped = []
        if self.active:
            rows_by_track: dict[str, list[int]] = {}
            for row_index, track_id in enumerate(row_track_ids):
                if track_id in self._dynamic_track_ids:
                    rows_by_track.setdefault(track_id, []).append(row_index)
            for track_id, row_indices in sorted(rows_by_track.items()):
                snapshot = self.state.track_snapshot(track_id, persistent=True)
                anchor_box7 = self._persistent_box7.get(track_id, snapshot.box7)
                aliases = self._aliases.get(track_id, ())
                first_alias = int(aliases[0]) if aliases else None
                containing_first = [
                    row_index
                    for row_index in row_indices
                    if first_alias is not None
                    and first_alias
                    in {int(value) for value in fusion_list[row_index]}
                ]
                candidates = containing_first or row_indices
                chosen = max(
                    candidates,
                    key=lambda row_index: (float(scores[row_index]), -row_index),
                )
                for row_index in row_indices:
                    if row_index != chosen:
                        mask[row_index] = False
                        dropped.append(row_index)
                output_parameters[chosen, :3] = anchor_box7[:3]
                output_parameters[chosen, 3:6] = anchor_box7[3:6]
                if track_id in self._persistent_rotation:
                    anchor_rotation = self._persistent_rotation[track_id]
                    anchor_yaw = self._persistent_observation_yaw[track_id]
                    yaw_delta = _wrap_angle(float(anchor_box7[6]) - anchor_yaw)
                    output_rotations[chosen] = _rotation_z(yaw_delta) @ anchor_rotation
                output_scores[chosen] = float(snapshot.persistent_score)
                modified.append(chosen)
        corners = _corners_from_params(output_parameters, output_rotations)
        self._stats["persistent_modified_rows"] += len(modified)
        self._stats["persistent_dropped_rows"] += len(dropped)
        return CurrentMaterialization(
            corners=_readonly(corners),
            scores=_readonly(output_scores),
            mask=_readonly(mask, dtype=bool),
            row_track_ids=row_track_ids,
            modified_rows=tuple(sorted(modified)),
            dropped_rows=tuple(sorted(dropped)),
        )

    def materialize_current(
        self,
        instances,
        fusion_list: Sequence[Sequence[int]],
        *,
        base_geometry=None,
    ) -> CurrentMaterialization:
        geometry = instances.pred_boxes_3d if base_geometry is None else base_geometry
        parameters = _numpy(geometry.tensor)
        rotations = _numpy(geometry.R)
        scores = _numpy(instances.scores).reshape(-1)
        if (
            parameters.ndim != 2
            or parameters.shape[1] != 6
            or rotations.shape != (parameters.shape[0], 3, 3)
            or scores.shape != (parameters.shape[0],)
            or len(fusion_list) != parameters.shape[0]
        ):
            raise DynamicBranchContractError("terminal native rows are misaligned")
        row_track_ids = self._row_track_ids(fusion_list)
        output_parameters = np.array(parameters, copy=True)
        output_rotations = np.array(rotations, copy=True)
        output_scores = np.array(scores, copy=True)
        mask = np.ones(len(scores), dtype=bool)
        modified = []
        dropped = []
        if self.active:
            current = {
                row.track_id: row for row in self.state.current_snapshots()
            }
            tracked_snapshots = {
                track_id: self.state.track_snapshot(track_id)
                for track_id in sorted(
                    {value for value in row_track_ids if value is not None}
                )
            }
            rows_by_track: dict[str, list[int]] = {}
            for row_index, track_id in enumerate(row_track_ids):
                snapshot = tracked_snapshots.get(track_id)
                inactive = snapshot is not None and snapshot.lifecycle in (
                    Lifecycle.DORMANT,
                    Lifecycle.RETIRED,
                )
                # A static object merely outside the camera remains a valid
                # global-map object. Suppress only identities whose location
                # is dynamic/uncertain or whose absence was contradicted by
                # visible free space (which increments visible misses).
                if inactive and (
                    track_id in self._dynamic_track_ids
                    or snapshot.consecutive_visible_misses > 0
                ):
                    mask[row_index] = False
                    dropped.append(row_index)
                elif track_id in self._dynamic_track_ids:
                    rows_by_track.setdefault(track_id, []).append(row_index)
            for track_id, row_indices in sorted(rows_by_track.items()):
                snapshot = current.get(track_id)
                if snapshot is None:
                    for row_index in row_indices:
                        mask[row_index] = False
                        dropped.append(row_index)
                    continue
                latest_init = self._latest_init.get(track_id)
                containing_latest = [
                    row_index
                    for row_index in row_indices
                    if latest_init is not None
                    and latest_init in {int(value) for value in fusion_list[row_index]}
                ]
                candidates = containing_latest or row_indices
                chosen = max(
                    candidates,
                    key=lambda row_index: (float(scores[row_index]), -row_index),
                )
                for row_index in row_indices:
                    if row_index != chosen:
                        mask[row_index] = False
                        dropped.append(row_index)
                output_parameters[chosen, :3] = snapshot.box7[:3]
                output_parameters[chosen, 3:6] = snapshot.box7[3:6]
                if track_id in self._latest_rotation:
                    latest_rotation = self._latest_rotation[track_id]
                    latest_yaw = self._latest_observation_yaw[track_id]
                    yaw_delta = _wrap_angle(float(snapshot.box7[6]) - latest_yaw)
                    output_rotations[chosen] = _rotation_z(yaw_delta) @ latest_rotation
                output_scores[chosen] = float(snapshot.current_score)
                modified.append(chosen)
        corners = _corners_from_params(output_parameters, output_rotations)
        self._stats["current_modified_rows"] += len(modified)
        self._stats["current_dropped_rows"] += len(dropped)
        return CurrentMaterialization(
            corners=_readonly(corners),
            scores=_readonly(output_scores),
            mask=_readonly(mask, dtype=bool),
            row_track_ids=row_track_ids,
            modified_rows=tuple(sorted(modified)),
            dropped_rows=tuple(sorted(dropped)),
        )

    def summary(self) -> dict:
        timings = np.asarray(self._timings_ms, dtype=np.float64)
        return {
            "schema": SCHEMA,
            "scene_id": self.scene_id,
            "mode": self.config.mode.value,
            **dict(self._stats),
            "state_tracks": self.state.track_count,
            "dynamic_tracks": len(self._dynamic_track_ids),
            "event_records": self._event_records,
            "event_records_dropped": self._event_records_dropped,
            "timing_ms": {
                "count": int(timings.size),
                "p50": float(np.percentile(timings, 50)) if timings.size else 0.0,
                "p95": float(np.percentile(timings, 95)) if timings.size else 0.0,
                "max": float(np.max(timings)) if timings.size else 0.0,
                "sum": float(np.sum(timings)) if timings.size else 0.0,
            },
        }

    def close(self) -> dict:
        if self._closed:
            return self.summary()
        summary = self.summary()
        if self.enabled:
            self._write_event({"schema": SCHEMA, "type": "summary", **summary})
        if self._event_handle is not None:
            self._event_handle.close()
            self._event_handle = None
        self._closed = True
        return summary


def build_causal_dynamic_branch(
    cfg: Mapping | None, *, scene_id: str
) -> CausalDynamicBranch:
    return CausalDynamicBranch(cfg, scene_id=scene_id)


def dynamic_event_ledger_complete(path: str | Path, *, scene_id: str) -> bool:
    """Return whether a scene ledger ended with its durable summary record.

    Prediction files are written before :meth:`CausalDynamicBranch.close`.  A
    process can therefore leave apparently complete prediction views while its
    identity ledger is truncated.  Resume callers use this side-effect-free
    check to rerun that scene instead of permanently skipping it.
    """

    if not isinstance(scene_id, str) or not _SAFE_SCENE_ID.fullmatch(scene_id):
        return False
    event_path = Path(path)
    if not event_path.is_file() or event_path.stat().st_size <= 0:
        return False
    final_record = None
    try:
        with event_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if line:
                    final_record = json.loads(line)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(final_record, Mapping)
        and final_record.get("schema") == SCHEMA
        and final_record.get("type") == "summary"
        and final_record.get("scene_id") == scene_id
    )


__all__ = [
    "CausalDynamicBranch",
    "CausalDynamicBranchConfig",
    "CurrentMaterialization",
    "DynamicBranchContractError",
    "DynamicBranchFrameResult",
    "DynamicBranchMode",
    "SCHEMA",
    "build_causal_dynamic_branch",
    "dynamic_event_ledger_complete",
    "resolve_causal_dynamic_branch_config",
]
