"""Training-free third-view confirmation for online side-candidate births.

This module is an observer, not a native prediction filter.  It tracks only
post-native BoxFusion identities and the frame provenance of the init ids in
their fusion groups.  A side candidate becomes confirmed once one of its
observed fusion groups contains sources from at least three distinct frames.

The transaction is deliberately strict: every :meth:`observe` must be closed
by exactly one :meth:`finalize` before the next frame can be observed.  The
native keep mask returned by ``finalize`` is always identity; a separate
frozen mask records counterfactual evidence for a future side-candidate gate.
Neither mask is applied here.  Callers must observe only real CuTR proposal
commit frames, never terminal frames that merely reuse stale predictions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_THIRD_VIEW_BIRTH_LITE_CONFIG = {
    "enabled": False,
    "observer_only": True,
    "min_distinct_source_frames": 3,
    "max_tracks": 1024,
    "max_sources_per_group": 5,
    "max_diagnostic_examples": 64,
}


@dataclass(frozen=True)
class ThirdViewTrackObservation:
    """Immutable state of one stable track after a post-native observation."""

    stable_id: int
    status: str
    source_init_ids: Tuple[int, ...]
    source_frame_ids: Tuple[int, ...]
    distinct_source_frames: int
    birth_frame_id: int
    confirmed_frame_id: Optional[int]
    birth_to_confirm_latency_frames: Optional[int]
    observations: int
    predecessor_stable_ids: Tuple[int, ...]
    merge_observed: bool


@dataclass(frozen=True)
class ThirdViewBirthBatch:
    """One pending post-native observation transaction."""

    scene_id: str
    frame_id: int
    tracks: Tuple[ThirdViewTrackObservation, ...]
    confirmed_tracks: int
    probationary_tracks: int
    new_tracks: int
    merged_predecessors: int
    retired_tracks: int


@dataclass(frozen=True)
class ThirdViewFinalDiagnostic:
    """Counterfactual side-candidate decision aligned to one final stable id."""

    stable_id: int
    status: str
    distinct_source_frames: int
    source_frame_ids: Tuple[int, ...]
    birth_frame_id: int
    confirmed_frame_id: Optional[int]
    birth_to_confirm_latency_frames: Optional[int]
    would_admit_side_candidate: bool
    reason: str


@dataclass(frozen=True)
class ThirdViewFinalizeResult:
    """Frozen masks that explicitly preserve every native CuTR anchor."""

    scene_id: str
    frame_id: int
    stable_ids: Tuple[int, ...]
    keep_mask: Tuple[bool, ...]
    would_admit_side_candidate_mask: Tuple[bool, ...]
    diagnostics: Tuple[ThirdViewFinalDiagnostic, ...]
    counterfactual_only: bool = True
    side_candidate_gate_only: bool = True
    native_filter_applied: bool = False


@dataclass(frozen=True)
class ThirdViewBirthSnapshot:
    scene_id: Optional[str]
    last_observed_frame_id: Optional[int]
    pending_frame_id: Optional[int]
    active_track_ids: Tuple[int, ...]
    confirmed_track_ids: Tuple[int, ...]
    probationary_track_ids: Tuple[int, ...]


@dataclass(frozen=True)
class _TrackState:
    stable_id: int
    birth_frame_id: int
    last_seen_frame_id: int
    confirmed_frame_id: Optional[int]
    observations: int
    source_init_ids: Tuple[int, ...]
    source_frame_ids: Tuple[int, ...]
    abstain_reason: Optional[str]


@dataclass(frozen=True)
class _PendingTransaction:
    batch: ThirdViewBirthBatch
    current_stable_ids: Tuple[int, ...]


def _strict_bool(name: str, value: object) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _strict_int(name: str, value: object, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def resolve_third_view_birth_lite_config(
    config: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Validate the bounded observer contract and frozen S0 threshold."""

    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("third_view_birth_lite config must be a mapping")
    unknown = sorted(set(config) - set(DEFAULT_THIRD_VIEW_BIRTH_LITE_CONFIG))
    if unknown:
        raise ValueError(
            "Unknown third_view_birth_lite config key(s): "
            + ", ".join(unknown)
        )
    resolved = dict(DEFAULT_THIRD_VIEW_BIRTH_LITE_CONFIG)
    resolved.update(config)
    resolved["enabled"] = _strict_bool(
        "third_view_birth_lite.enabled", resolved["enabled"]
    )
    resolved["observer_only"] = _strict_bool(
        "third_view_birth_lite.observer_only", resolved["observer_only"]
    )
    if resolved["enabled"] and not resolved["observer_only"]:
        raise ValueError(
            "third_view_birth_lite is not authorized to mutate predictions; "
            "observer_only must remain true"
        )

    limits = {
        "min_distinct_source_frames": (1, 5),
        "max_tracks": (1, 1024),
        "max_sources_per_group": (1, 5),
        "max_diagnostic_examples": (0, 1024),
    }
    for key, (minimum, maximum) in limits.items():
        resolved[key] = _strict_int(
            f"third_view_birth_lite.{key}", resolved[key], minimum
        )
        if resolved[key] > maximum:
            raise ValueError(
                f"third_view_birth_lite.{key} must not exceed {maximum}"
            )
    if (
        resolved["min_distinct_source_frames"]
        > resolved["max_sources_per_group"]
    ):
        raise ValueError(
            "third_view_birth_lite.min_distinct_source_frames must not exceed "
            "max_sources_per_group"
        )
    if resolved["enabled"] and resolved["min_distinct_source_frames"] != 3:
        raise ValueError(
            "enabled third_view_birth_lite must keep the frozen three-view "
            "threshold: min_distinct_source_frames=3"
        )
    return resolved


def _scene_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("scene_id must be a non-empty string")
    return value


def _integer_vector(
    values: object,
    name: str,
    *,
    unique: bool = False,
) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a one-dimensional integer sequence")
    if isinstance(values, np.ndarray):
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(
                f"{name} must be a one-dimensional integer sequence"
            )
        raw = tuple(values.tolist())
    else:
        if not isinstance(values, Sequence):
            raise ValueError(
                f"{name} must be a one-dimensional integer sequence"
            )
        raw = tuple(values)
    result = tuple(
        _strict_int(f"{name}[{index}]", item)
        for index, item in enumerate(raw)
    )
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique ids")
    return result


def _source_frame_lookup(values: object) -> Dict[int, int]:
    """Copy a dense sequence or sparse init-id mapping into plain integers."""

    if isinstance(values, Mapping):
        result: Dict[int, int] = {}
        for raw_init_id, raw_frame_id in values.items():
            init_id = _strict_int("source_frame_ids key", raw_init_id)
            frame_id = _strict_int(
                f"source_frame_ids[{init_id}]", raw_frame_id
            )
            result[init_id] = frame_id
        return result
    vector = _integer_vector(values, "source_frame_ids")
    return {init_id: frame_id for init_id, frame_id in enumerate(vector)}


def _fusion_groups(
    groups: object,
    *,
    max_sources_per_group: int,
) -> Tuple[Tuple[int, ...], ...]:
    if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence):
        raise ValueError("current_fusion_groups must be a sequence")
    result = []
    globally_seen = set()
    for row, group in enumerate(groups):
        if isinstance(group, (str, bytes)):
            raise ValueError(
                f"current_fusion_groups[{row}] must be an integer iterable"
            )
        try:
            raw = tuple(group)
        except TypeError as error:
            raise ValueError(
                f"current_fusion_groups[{row}] must be an integer iterable"
            ) from error
        if not raw:
            raise ValueError(f"current_fusion_groups[{row}] must not be empty")
        normalized = tuple(
            _strict_int(f"current_fusion_groups[{row}]", item) for item in raw
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError(
                f"current_fusion_groups[{row}] must contain unique init ids"
            )
        overlap = globally_seen.intersection(normalized)
        if overlap:
            raise ValueError(
                "current_fusion_groups must not share init ids across rows"
            )
        globally_seen.update(normalized)
        result.append(tuple(sorted(normalized)))
    return tuple(result)


def _percentile(values: Sequence[int], percentile: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


class ThirdViewBirthLiteObserver:
    """Causal, bounded confirmation observer for non-native side candidates."""

    _LATENCY_WINDOW = 2048

    def __init__(self, config: Optional[Mapping[str, object]] = None):
        self.config = resolve_third_view_birth_lite_config(config)
        self.enabled = bool(self.config["enabled"])
        self.observer_only = bool(self.config["observer_only"])
        self._scene_id: Optional[str] = None
        self._last_observed_frame_id: Optional[int] = None
        self._pending: Optional[_PendingTransaction] = None
        self._tracks: Dict[int, _TrackState] = {}
        self._latency_samples = deque(maxlen=self._LATENCY_WINDOW)
        self._pipeline_samples = deque(maxlen=self._LATENCY_WINDOW)
        self._diagnostic_examples: list[Dict[str, object]] = []
        self._stats = self._new_stats()

    @staticmethod
    def _new_stats() -> Dict[str, int]:
        return {
            "observations": 0,
            "finalizations": 0,
            "track_rows_observed": 0,
            "births": 0,
            "confirmation_events": 0,
            "retired_tracks_total": 0,
            "retired_confirmed": 0,
            "retired_probationary": 0,
            "merge_events": 0,
            "merged_predecessors": 0,
            "source_cap_overflow_rows": 0,
            "peak_active_tracks": 0,
            "final_rows": 0,
            "final_confirmed": 0,
            "final_probationary": 0,
            "would_suppress": 0,
            "birth_to_confirm_latency_count": 0,
            "birth_to_confirm_latency_total_frames": 0,
            "birth_to_confirm_latency_max_frames": 0,
            "pipeline_observe_calls": 0,
            "pipeline_observe_ms_total": 0.0,
            "pipeline_observe_ms_max": 0.0,
        }

    @property
    def scene_id(self) -> Optional[str]:
        return self._scene_id

    def reset_scene(self, scene_id: str) -> None:
        """Clear bounded state and bind the observer to one scene."""

        scene = _scene_id(scene_id)
        if self._pending is not None:
            raise ValueError("pending observation must be finalized before reset")
        self._scene_id = scene
        self._last_observed_frame_id = None
        self._tracks.clear()
        self._latency_samples.clear()
        self._pipeline_samples.clear()
        self._diagnostic_examples.clear()
        self._stats = self._new_stats()

    def _append_diagnostic(self, row: Dict[str, object]) -> None:
        limit = int(self.config["max_diagnostic_examples"])
        if len(self._diagnostic_examples) < limit:
            self._diagnostic_examples.append(dict(row))

    @staticmethod
    def _track_row(
        state: _TrackState,
        predecessor_ids: Tuple[int, ...],
    ) -> ThirdViewTrackObservation:
        confirmed = (
            state.confirmed_frame_id is not None
            and state.abstain_reason is None
        )
        latency = (
            state.confirmed_frame_id - state.birth_frame_id
            if state.confirmed_frame_id is not None
            else None
        )
        return ThirdViewTrackObservation(
            stable_id=state.stable_id,
            status=(
                "confirmed"
                if confirmed
                else (
                    "abstained_source_overflow"
                    if state.abstain_reason is not None
                    else "probationary"
                )
            ),
            source_init_ids=state.source_init_ids,
            source_frame_ids=state.source_frame_ids,
            distinct_source_frames=len(state.source_frame_ids),
            birth_frame_id=state.birth_frame_id,
            confirmed_frame_id=state.confirmed_frame_id,
            birth_to_confirm_latency_frames=latency,
            observations=state.observations,
            predecessor_stable_ids=predecessor_ids,
            merge_observed=len(predecessor_ids) > 1,
        )

    def observe(
        self,
        *,
        scene_id: str,
        frame_id: int,
        current_fusion_groups: Sequence[Iterable[int]],
        current_stable_ids: Sequence[int],
        source_frame_ids: object,
    ) -> ThirdViewBirthBatch:
        """Atomically observe one post-native frame without reading predictions.

        ``source_frame_ids`` may be a dense sequence indexed by init id or a
        sparse ``init_id -> frame_id`` mapping containing the sources used by
        the current groups. Repeated frame ids inside one fusion group count as
        one view. First confirmation requires three distinct frame ids in the
        *current* group and is sticky thereafter.
        """

        if not self.enabled:
            raise RuntimeError("third_view_birth_lite observer is disabled")
        scene = _scene_id(scene_id)
        frame = _strict_int("frame_id", frame_id)
        groups = _fusion_groups(
            current_fusion_groups,
            max_sources_per_group=int(self.config["max_sources_per_group"]),
        )
        stable_ids = _integer_vector(
            current_stable_ids, "current_stable_ids", unique=True
        )
        sources = _source_frame_lookup(source_frame_ids)
        if len(groups) != len(stable_ids):
            raise ValueError(
                "current_stable_ids must be row-aligned with fusion groups"
            )
        if len(stable_ids) > int(self.config["max_tracks"]):
            raise ValueError("current tracks exceed the configured bounded capacity")
        for row, group in enumerate(groups):
            for init_id in group:
                if init_id not in sources:
                    raise ValueError(
                        f"current_fusion_groups[{row}] init id {init_id} is "
                        "outside source_frame_ids"
                    )
                if sources[init_id] > frame:
                    raise ValueError(
                        "source_frame_ids must not contain future evidence"
                    )

        # Transaction/order checks happen after pure input validation and
        # before any observer state changes.
        if self._pending is not None:
            raise ValueError("previous observation must be closed by finalize")
        if self._scene_id is not None and self._scene_id != scene:
            raise ValueError(
                f"third_view_birth_lite is bound to {self._scene_id}, not {scene}"
            )
        if (
            self._last_observed_frame_id is not None
            and frame <= self._last_observed_frame_id
        ):
            raise ValueError("observation frame ids must be strictly increasing")

        old_tracks = self._tracks
        current_row_by_source = {
            source_id: row
            for row, group in enumerate(groups)
            for source_id in group
        }

        # Assign every prior track to at most one current row.  A preserved
        # stable id wins over incidental init-id overlap; otherwise ambiguous
        # splits fail before mutation instead of cloning historical evidence.
        predecessors_by_row: list[list[int]] = [[] for _ in groups]
        current_index_by_stable = {
            stable_id: index for index, stable_id in enumerate(stable_ids)
        }
        for old_id, old_state in old_tracks.items():
            same_row = current_index_by_stable.get(old_id)
            overlap_rows = sorted(
                {
                    current_row_by_source[source_id]
                    for source_id in old_state.source_init_ids
                    if source_id in current_row_by_source
                }
            )
            if same_row is not None:
                assigned_row = same_row
            elif len(overlap_rows) == 1:
                assigned_row = overlap_rows[0]
            elif len(overlap_rows) > 1:
                raise ValueError(
                    f"prior stable id {old_id} ambiguously split across groups"
                )
            else:
                continue
            predecessors_by_row[assigned_row].append(old_id)

        new_tracks: Dict[int, _TrackState] = {}
        predecessor_sets = []
        new_births = 0
        confirmation_latencies = []
        merge_events = 0
        merged_predecessors = 0
        rows = []
        threshold = int(self.config["min_distinct_source_frames"])
        source_cap = int(self.config["max_sources_per_group"])
        overflow_rows = 0
        for index, (group, stable_id) in enumerate(zip(groups, stable_ids)):
            predecessor_ids = tuple(sorted(predecessors_by_row[index]))
            predecessor_sets.extend(predecessor_ids)
            predecessors = tuple(old_tracks[value] for value in predecessor_ids)
            source_overflow = len(group) > source_cap
            overflow_rows += int(source_overflow)
            retained_group = tuple(group[:source_cap])
            current_source_frames = tuple(
                sorted({sources[init_id] for init_id in retained_group})
            )
            if predecessors:
                birth_frame = min(item.birth_frame_id for item in predecessors)
                prior_confirmed = [
                    item.confirmed_frame_id
                    for item in predecessors
                    if item.confirmed_frame_id is not None
                ]
                confirmed_frame = min(prior_confirmed) if prior_confirmed else None
                observations = max(item.observations for item in predecessors) + 1
                prior_abstentions = tuple(
                    item.abstain_reason
                    for item in predecessors
                    if item.abstain_reason is not None
                )
            else:
                birth_frame = frame
                confirmed_frame = None
                observations = 1
                new_births += 1
                prior_abstentions = ()
            abstain_reason = (
                "source_cap_overflow"
                if source_overflow or prior_abstentions
                else None
            )
            newly_confirmed = (
                abstain_reason is None
                and
                confirmed_frame is None
                and len(current_source_frames) >= threshold
            )
            if newly_confirmed:
                confirmed_frame = frame
                confirmation_latencies.append(frame - birth_frame)
            if len(predecessor_ids) > 1:
                merge_events += 1
                merged_predecessors += len(predecessor_ids) - 1
            state = _TrackState(
                stable_id=stable_id,
                birth_frame_id=birth_frame,
                last_seen_frame_id=frame,
                confirmed_frame_id=confirmed_frame,
                observations=observations,
                source_init_ids=retained_group,
                source_frame_ids=current_source_frames,
                abstain_reason=abstain_reason,
            )
            new_tracks[stable_id] = state
            rows.append(self._track_row(state, predecessor_ids))

        inherited = set(predecessor_sets)
        retired_ids = tuple(sorted(set(old_tracks) - set(stable_ids)))
        # A predecessor whose id changed or was absorbed is retired as an id,
        # even though its earliest causal evidence is inherited by the winner.
        # A disappeared state is retired without inheritance.
        retired_count = len(retired_ids)
        retired_confirmed = sum(
            old_tracks[value].confirmed_frame_id is not None
            for value in retired_ids
        )
        retired_probationary = retired_count - retired_confirmed

        batch = ThirdViewBirthBatch(
            scene_id=scene,
            frame_id=frame,
            tracks=tuple(rows),
            confirmed_tracks=sum(row.status == "confirmed" for row in rows),
            probationary_tracks=sum(row.status == "probationary" for row in rows),
            new_tracks=new_births,
            merged_predecessors=merged_predecessors,
            retired_tracks=retired_count,
        )

        # Commit all state and counters only after the complete batch exists.
        self._scene_id = scene
        self._last_observed_frame_id = frame
        self._tracks = new_tracks
        self._pending = _PendingTransaction(batch, stable_ids)
        self._stats["observations"] += 1
        self._stats["track_rows_observed"] += len(rows)
        self._stats["births"] += new_births
        self._stats["confirmation_events"] += len(confirmation_latencies)
        self._stats["retired_tracks_total"] += retired_count
        self._stats["retired_confirmed"] += retired_confirmed
        self._stats["retired_probationary"] += retired_probationary
        self._stats["merge_events"] += merge_events
        self._stats["merged_predecessors"] += merged_predecessors
        self._stats["source_cap_overflow_rows"] += overflow_rows
        self._stats["peak_active_tracks"] = max(
            self._stats["peak_active_tracks"], len(new_tracks)
        )
        for latency in confirmation_latencies:
            self._latency_samples.append(latency)
            self._stats["birth_to_confirm_latency_count"] += 1
            self._stats["birth_to_confirm_latency_total_frames"] += latency
            self._stats["birth_to_confirm_latency_max_frames"] = max(
                self._stats["birth_to_confirm_latency_max_frames"], latency
            )
        for retired_id in retired_ids:
            old = old_tracks[retired_id]
            if retired_id in inherited:
                reason = "merged_or_remapped"
            else:
                reason = "track_disappeared"
            self._append_diagnostic(
                {
                    "kind": "retired",
                    "scene_id": scene,
                    "frame_id": frame,
                    "stable_id": retired_id,
                    "status": (
                        "confirmed"
                        if old.confirmed_frame_id is not None
                        else "probationary"
                    ),
                    "reason": reason,
                }
            )
        return batch

    def finalize(
        self,
        *,
        final_stable_ids: Sequence[int],
    ) -> ThirdViewFinalizeResult:
        """Close the pending frame and return a side-candidate-only mask.

        The input may be an ordered subset of the latest active stable ids,
        but every id must be known and unique.  A bad finalize leaves the
        transaction pending so the caller can retry with corrected alignment.
        """

        if not self.enabled:
            raise RuntimeError("third_view_birth_lite observer is disabled")
        stable_ids = _integer_vector(
            final_stable_ids, "final_stable_ids", unique=True
        )
        if self._pending is None:
            raise ValueError("finalize requires one pending observation")
        active = set(self._pending.current_stable_ids)
        unknown = tuple(value for value in stable_ids if value not in active)
        if unknown:
            raise ValueError(
                "final_stable_ids contain ids absent from the latest observation: "
                + ", ".join(str(value) for value in unknown)
            )

        diagnostics = []
        mask = []
        for stable_id in stable_ids:
            state = self._tracks[stable_id]
            confirmed = (
                state.confirmed_frame_id is not None
                and state.abstain_reason is None
            )
            latency = (
                state.confirmed_frame_id - state.birth_frame_id
                if state.confirmed_frame_id is not None
                else None
            )
            diagnostics.append(
                ThirdViewFinalDiagnostic(
                    stable_id=stable_id,
                    status=(
                        "confirmed"
                        if confirmed
                        else (
                            "abstained_source_overflow"
                            if state.abstain_reason is not None
                            else "probationary"
                        )
                    ),
                    distinct_source_frames=len(state.source_frame_ids),
                    source_frame_ids=state.source_frame_ids,
                    birth_frame_id=state.birth_frame_id,
                    confirmed_frame_id=state.confirmed_frame_id,
                    birth_to_confirm_latency_frames=latency,
                    would_admit_side_candidate=confirmed,
                    reason=(
                        "three_distinct_source_frames_confirmed"
                        if confirmed
                        else (
                            state.abstain_reason
                            if state.abstain_reason is not None
                            else "fewer_than_three_distinct_source_frames"
                        )
                    ),
                )
            )
            mask.append(bool(confirmed))
        result = ThirdViewFinalizeResult(
            scene_id=self._pending.batch.scene_id,
            frame_id=self._pending.batch.frame_id,
            stable_ids=stable_ids,
            # The native mask is deliberately identity.  Confirmation is a
            # separate diagnostic for future PUF/async *side* candidates.
            keep_mask=(True,) * len(stable_ids),
            would_admit_side_candidate_mask=tuple(mask),
            diagnostics=tuple(diagnostics),
        )

        # Finalization changes observer accounting only.  No caller-owned
        # array/list and no native BoxFusion prediction is ever mutated.
        self._pending = None
        confirmed_count = sum(mask)
        probationary_count = len(mask) - confirmed_count
        self._stats["finalizations"] += 1
        self._stats["final_rows"] += len(mask)
        self._stats["final_confirmed"] += confirmed_count
        self._stats["final_probationary"] += probationary_count
        self._stats["would_suppress"] += probationary_count
        for row in diagnostics:
            self._append_diagnostic(
                {
                    "kind": "final",
                    "scene_id": result.scene_id,
                    "frame_id": result.frame_id,
                    "stable_id": row.stable_id,
                    "status": row.status,
                    "would_admit_side_candidate": (
                        row.would_admit_side_candidate
                    ),
                    "reason": row.reason,
                }
            )
        return result

    def snapshot(self) -> ThirdViewBirthSnapshot:
        confirmed = tuple(
            sorted(
                stable_id
                for stable_id, state in self._tracks.items()
                if state.confirmed_frame_id is not None
                and state.abstain_reason is None
            )
        )
        probationary = tuple(
            sorted(set(self._tracks) - set(confirmed))
        )
        return ThirdViewBirthSnapshot(
            scene_id=self._scene_id,
            last_observed_frame_id=self._last_observed_frame_id,
            pending_frame_id=(
                self._pending.batch.frame_id if self._pending is not None else None
            ),
            active_track_ids=tuple(sorted(self._tracks)),
            confirmed_track_ids=confirmed,
            probationary_track_ids=probationary,
        )

    def record_pipeline_timing(self, observe_ms: object) -> None:
        """Record wrapper cost without granting the observer runtime authority."""

        if isinstance(observe_ms, (bool, np.bool_)) or not isinstance(
            observe_ms, Real
        ):
            raise ValueError("observe_ms must be a finite non-negative number")
        value = float(observe_ms)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("observe_ms must be a finite non-negative number")
        self._stats["pipeline_observe_calls"] += 1
        self._stats["pipeline_observe_ms_total"] += value
        self._stats["pipeline_observe_ms_max"] = max(
            float(self._stats["pipeline_observe_ms_max"]), value
        )
        self._pipeline_samples.append(value)

    def summary(self) -> Dict[str, object]:
        """Return JSON-safe safety, boundedness, and outcome evidence."""

        active_confirmed = sum(
            state.confirmed_frame_id is not None
            and state.abstain_reason is None
            for state in self._tracks.values()
        )
        active_probationary = len(self._tracks) - active_confirmed
        latency_count = self._stats["birth_to_confirm_latency_count"]
        pipeline_count = self._stats["pipeline_observe_calls"]
        latency_mean = (
            self._stats["birth_to_confirm_latency_total_frames"] / latency_count
            if latency_count
            else None
        )
        return {
            "schema": "boxfusion.third_view_birth_lite_shadow.v1",
            "enabled": self.enabled,
            "observer_only": self.observer_only,
            "training_free": True,
            "causal": True,
            "bounded_memory": True,
            "counterfactual_only": True,
            "side_candidate_gate_only": True,
            "active_authorized": False,
            "native_filter_applied": False,
            "native_keep_mask_identity": True,
            "native_would_suppress": 0,
            "would_suppress_scope": "side_candidates_only",
            "current_cutr_commit_only_contract": True,
            "terminal_stale_frames_observed": False,
            "native_outputs_mutated": False,
            "ground_truth_access": False,
            "detector_score_access": False,
            "clip_access": False,
            "puf_access": False,
            "thresholds_frozen": bool(self.enabled),
            "effective_config": dict(self.config),
            "min_distinct_source_frames": int(
                self.config["min_distinct_source_frames"]
            ),
            "max_tracks": int(self.config["max_tracks"]),
            "max_sources_per_group": int(
                self.config["max_sources_per_group"]
            ),
            "scene_id": self._scene_id,
            "pending_frame_id": (
                self._pending.batch.frame_id if self._pending is not None else None
            ),
            "active_tracks": len(self._tracks),
            "active_confirmed_tracks": active_confirmed,
            "active_probationary_tracks": active_probationary,
            **dict(self._stats),
            "birth_to_confirm_latency_mean_frames": latency_mean,
            "birth_to_confirm_latency_p50_frames": _percentile(
                self._latency_samples, 50.0
            ),
            "birth_to_confirm_latency_p95_frames": _percentile(
                self._latency_samples, 95.0
            ),
            "latency_window_size": len(self._latency_samples),
            "pipeline_observe_ms_mean": (
                self._stats["pipeline_observe_ms_total"] / pipeline_count
                if pipeline_count
                else 0.0
            ),
            "pipeline_observe_ms_p95": (
                float(np.percentile(np.asarray(self._pipeline_samples), 95.0))
                if self._pipeline_samples
                else 0.0
            ),
            "diagnostic_examples": [
                dict(value) for value in self._diagnostic_examples
            ],
        }

    def summary_line(self) -> str:
        summary = self.summary()
        return (
            "Third-view-birth-lite shadow summary | "
            f"observations={summary['observations']}, "
            f"active confirmed/probationary="
            f"{summary['active_confirmed_tracks']}/"
            f"{summary['active_probationary_tracks']}, "
            f"confirmation_events={summary['confirmation_events']}, "
            f"pipeline_mean/p95/max_ms="
            f"{summary['pipeline_observe_ms_mean']:.4f}/"
            f"{summary['pipeline_observe_ms_p95']:.4f}/"
            f"{summary['pipeline_observe_ms_max']:.4f}"
        )


def build_third_view_birth_lite(
    root_config: Optional[Mapping[str, object]] = None,
) -> ThirdViewBirthLiteObserver:
    """Build from either a root config or a direct section mapping."""

    if root_config is None:
        section: Mapping[str, object] = {}
    elif not isinstance(root_config, Mapping):
        raise ValueError("root config must be a mapping")
    elif "third_view_birth_lite" in root_config:
        candidate = root_config["third_view_birth_lite"]
        if candidate is None:
            section = {}
        elif not isinstance(candidate, Mapping):
            raise ValueError("third_view_birth_lite section must be a mapping")
        else:
            section = candidate
    elif set(root_config).issubset(DEFAULT_THIRD_VIEW_BIRTH_LITE_CONFIG):
        section = root_config
    else:
        # A normal application config that predates this optional observer has
        # no section and must preserve the original disabled path.
        section = {}
    return ThirdViewBirthLiteObserver(section)


__all__ = [
    "DEFAULT_THIRD_VIEW_BIRTH_LITE_CONFIG",
    "ThirdViewBirthBatch",
    "ThirdViewBirthLiteObserver",
    "ThirdViewBirthSnapshot",
    "ThirdViewFinalDiagnostic",
    "ThirdViewFinalizeResult",
    "ThirdViewTrackObservation",
    "build_third_view_birth_lite",
    "resolve_third_view_birth_lite_config",
]
