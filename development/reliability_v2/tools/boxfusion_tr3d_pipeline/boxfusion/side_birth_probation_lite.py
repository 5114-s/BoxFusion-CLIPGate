"""Bounded, training-free probation ledger for caller-selected side births.

The ledger is deliberately an observer.  A caller enrolls only proposals that
its already-frozen pipeline classified as *birth* events, then calls
``observe_true_cutr_keyframe`` once for every real CuTR commit (including
commits with no new births).  An enrolled committed stable id is confirmed on
its third distinct observed keyframe.  Confirmation is sticky and never
changes a native prediction.

``native_target_kind`` and ``native_target_ids`` are frozen diagnostic labels.
They are validated and stored with each event, but are not consulted by any
state transition.  They are read for the first time by ``close_scene`` when it
computes precision and retention.  Likewise, probabilities and margins are
receipt-only diagnostics.  This module imports no detector, ground-truth,
score model, CLIP, PUF, Torch, or other learned component.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA = "boxfusion.side_birth_probation_lite_shadow.v1"

DEFAULT_SIDE_BIRTH_PROBATION_LITE_CONFIG = {
    "enabled": False,
    "observer_only": True,
    "min_distinct_keyframes": 3,
    "max_missed_keyframes": 10,
    "max_pending_tracks": 256,
    "max_birth_events": 8192,
}

_FROZEN_MIN_DISTINCT_KEYFRAMES = 3
_FROZEN_MAX_MISSED_KEYFRAMES = 10
_HARD_MAX_PENDING_TRACKS = 256
_HARD_MAX_BIRTH_EVENTS = 8192
_NATIVE_TARGET_KINDS = frozenset(
    {"birth", "unique_history", "ambiguous_history", "unresolved"}
)


def _strict_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _strict_int(name: str, value: object, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_probability(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    result = float(value)
    if not isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return result


def _scene_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("scene_id must be a non-empty string")
    return value


def _optional_stable_id(name: str, value: object) -> Optional[int]:
    if value is None:
        return None
    return _strict_int(name, value)


def _id_tuple(name: str, values: object) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise ValueError(f"{name} must be an iterable of non-negative ids")
    try:
        raw = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(
            f"{name} must be an iterable of non-negative ids"
        ) from error
    result = tuple(
        _strict_int(f"{name}[{index}]", value)
        for index, value in enumerate(raw)
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique ids")
    return tuple(sorted(result))


def _active_ids(name: str, values: object) -> Tuple[int, ...]:
    """Copy and collapse an active-id iterable without mutating caller data."""

    if isinstance(values, (str, bytes, Mapping)):
        raise ValueError(f"{name} must be an iterable of non-negative ids")
    try:
        raw = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(
            f"{name} must be an iterable of non-negative ids"
        ) from error
    return tuple(
        sorted(
            {
                _strict_int(f"{name}[{index}]", value)
                for index, value in enumerate(raw)
            }
        )
    )


def resolve_side_birth_probation_lite_config(
    config: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Strictly validate the frozen observer contract and hard bounds."""

    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("side_birth_probation_lite config must be a mapping")
    unknown = sorted(set(config) - set(DEFAULT_SIDE_BIRTH_PROBATION_LITE_CONFIG))
    if unknown:
        raise ValueError(
            "Unknown side_birth_probation_lite config key(s): "
            + ", ".join(unknown)
        )
    resolved = dict(DEFAULT_SIDE_BIRTH_PROBATION_LITE_CONFIG)
    resolved.update(config)
    resolved["enabled"] = _strict_bool(
        "side_birth_probation_lite.enabled", resolved["enabled"]
    )
    resolved["observer_only"] = _strict_bool(
        "side_birth_probation_lite.observer_only", resolved["observer_only"]
    )
    if not resolved["observer_only"]:
        raise ValueError(
            "side_birth_probation_lite has no active authority; "
            "observer_only must remain true"
        )

    limits = {
        "min_distinct_keyframes": (1, _FROZEN_MIN_DISTINCT_KEYFRAMES),
        "max_missed_keyframes": (0, _FROZEN_MAX_MISSED_KEYFRAMES),
        "max_pending_tracks": (1, _HARD_MAX_PENDING_TRACKS),
        "max_birth_events": (1, _HARD_MAX_BIRTH_EVENTS),
    }
    for key, (minimum, maximum) in limits.items():
        resolved[key] = _strict_int(
            f"side_birth_probation_lite.{key}", resolved[key], minimum
        )
        if resolved[key] > maximum:
            raise ValueError(
                f"side_birth_probation_lite.{key} must not exceed {maximum}"
            )
    if resolved["enabled"]:
        if resolved["min_distinct_keyframes"] != _FROZEN_MIN_DISTINCT_KEYFRAMES:
            raise ValueError(
                "enabled side_birth_probation_lite must keep the frozen "
                "three-keyframe threshold: min_distinct_keyframes=3"
            )
        if resolved["max_missed_keyframes"] != _FROZEN_MAX_MISSED_KEYFRAMES:
            raise ValueError(
                "enabled side_birth_probation_lite must keep the frozen TTL: "
                "max_missed_keyframes=10"
            )
    return resolved


def _native_target(
    kind: object, ids: object
) -> Tuple[str, Tuple[int, ...]]:
    if not isinstance(kind, str) or kind not in _NATIVE_TARGET_KINDS:
        allowed = ", ".join(sorted(_NATIVE_TARGET_KINDS))
        raise ValueError(f"native_target_kind must be one of: {allowed}")
    normalized_ids = _id_tuple("native_target_ids", ids)
    expected = {
        "birth": (0, 0),
        "unique_history": (1, 1),
        "ambiguous_history": (2, None),
        "unresolved": (0, 0),
    }[kind]
    minimum, maximum = expected
    if len(normalized_ids) < minimum or (
        maximum is not None and len(normalized_ids) > maximum
    ):
        cardinality = {
            "birth": "empty",
            "unique_history": "exactly one id",
            "ambiguous_history": "at least two ids",
            "unresolved": "empty",
        }[kind]
        raise ValueError(
            f"native_target_ids must be {cardinality} for {kind}"
        )
    return kind, normalized_ids


@dataclass(frozen=True)
class SideBirthSeedEvent:
    """Neutral, frozen diagnostic row supplied for one screened birth."""

    proposal_id: int
    committed_stable_id: Optional[int]
    top_probability: float
    margin: float
    native_target_kind: str
    native_target_ids: Tuple[int, ...]

    def __post_init__(self) -> None:
        proposal_id = _strict_int("proposal_id", self.proposal_id)
        stable_id = _optional_stable_id(
            "committed_stable_id", self.committed_stable_id
        )
        probability = _finite_probability(
            "top_probability", self.top_probability
        )
        margin = _finite_probability("margin", self.margin)
        kind, ids = _native_target(
            self.native_target_kind, self.native_target_ids
        )
        object.__setattr__(self, "proposal_id", proposal_id)
        object.__setattr__(self, "committed_stable_id", stable_id)
        object.__setattr__(self, "top_probability", probability)
        object.__setattr__(self, "margin", margin)
        object.__setattr__(self, "native_target_kind", kind)
        object.__setattr__(self, "native_target_ids", ids)


@dataclass(frozen=True)
class SideBirthEvent:
    """A seed bound to its real CuTR scene, frame, and keyframe step."""

    event_index: int
    scene_id: str
    frame_id: int
    keyframe_step: int
    proposal_id: int
    committed_stable_id: Optional[int]
    top_probability: float
    margin: float
    native_target_kind: str
    native_target_ids: Tuple[int, ...]


@dataclass(frozen=True)
class SideBirthKeyframeResult:
    """Immutable operational result of one true-keyframe transaction."""

    scene_id: str
    frame_id: int
    keyframe_step: int
    enrolled_events: Tuple[SideBirthEvent, ...]
    observed_track_ids: Tuple[int, ...]
    pending_track_ids: Tuple[int, ...]
    confirmed_track_ids: Tuple[int, ...]
    retired_track_ids: Tuple[int, ...]
    newly_confirmed_track_ids: Tuple[int, ...]
    newly_retired_track_ids: Tuple[int, ...]
    capacity_rejected_event_indices: Tuple[int, ...]
    birth_event_capacity_rejected_events: int
    audit_complete: bool

    @property
    def pending_tracks(self) -> int:
        return len(self.pending_track_ids)

    @property
    def confirmed_tracks(self) -> int:
        return len(self.confirmed_track_ids)


@dataclass(frozen=True)
class SideBirthProbationSnapshot:
    scene_id: Optional[str]
    last_frame_id: Optional[int]
    last_keyframe_step: Optional[int]
    birth_events: int
    pending_track_ids: Tuple[int, ...]
    confirmed_track_ids: Tuple[int, ...]
    retired_track_ids: Tuple[int, ...]
    pending_track_cap_hits: int
    birth_event_cap_hits: int
    birth_event_capacity_rejected_events: int
    audit_complete: bool
    closed: bool


@dataclass(frozen=True)
class SideBirthEventReceipt:
    """Terminal result for one event; all collection fields are tuples."""

    event_index: int
    scene_id: str
    frame_id: int
    keyframe_step: int
    proposal_id: int
    committed_stable_id: Optional[int]
    top_probability: float
    margin: float
    native_target_kind: str
    native_target_ids: Tuple[int, ...]
    observed_frame_ids: Tuple[int, ...]
    observed_keyframe_steps: Tuple[int, ...]
    confirmed: bool
    confirmation_frame_id: Optional[int]
    confirmation_keyframe_step: Optional[int]
    retired: bool
    retirement_frame_id: Optional[int]
    retirement_keyframe_step: Optional[int]
    active_at_terminal: bool
    status: str
    latency_frames: Optional[int]
    latency_keyframes: Optional[int]
    reason: str

    # Compact aliases keep the receipt pleasant to consume while the stored
    # names remain explicit about frame ids versus keyframe sequence numbers.
    @property
    def observed_frames(self) -> Tuple[int, ...]:
        return self.observed_frame_ids

    @property
    def observed_steps(self) -> Tuple[int, ...]:
        return self.observed_keyframe_steps

    @property
    def confirmed_frame_id(self) -> Optional[int]:
        return self.confirmation_frame_id

    @property
    def confirmed_keyframe_step(self) -> Optional[int]:
        return self.confirmation_keyframe_step

    @property
    def retired_frame_id(self) -> Optional[int]:
        return self.retirement_frame_id

    @property
    def retired_keyframe_step(self) -> Optional[int]:
        return self.retirement_keyframe_step

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "event_index": self.event_index,
            "scene_id": self.scene_id,
            "frame_id": self.frame_id,
            "keyframe_step": self.keyframe_step,
            "proposal_id": self.proposal_id,
            "committed_stable_id": self.committed_stable_id,
            "top_probability": self.top_probability,
            "margin": self.margin,
            "native_target_kind": self.native_target_kind,
            "native_target_ids": list(self.native_target_ids),
            "observed_frame_ids": list(self.observed_frame_ids),
            "observed_keyframe_steps": list(self.observed_keyframe_steps),
            "confirmed": self.confirmed,
            "confirmation_frame_id": self.confirmation_frame_id,
            "confirmation_keyframe_step": self.confirmation_keyframe_step,
            "retired": self.retired,
            "retirement_frame_id": self.retirement_frame_id,
            "retirement_keyframe_step": self.retirement_keyframe_step,
            "active_at_terminal": self.active_at_terminal,
            "status": self.status,
            "latency_frames": self.latency_frames,
            "latency_keyframes": self.latency_keyframes,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SideBirthProbationMetrics:
    event_total: int
    event_evaluable: int
    event_unresolved: int
    event_native_birth: int
    event_native_unique_history: int
    event_native_ambiguous_history: int
    event_confirmed: int
    event_confirmed_evaluable: int
    event_confirmed_native_birth: int
    event_confirmed_precision: Optional[float]
    event_confirmed_retention: Optional[float]
    unique_track_total: int
    unique_track_evaluable: int
    unique_track_unresolved: int
    unique_track_native_birth: int
    unique_track_native_history: int
    unique_track_mixed_labels: int
    unique_track_confirmed: int
    unique_track_confirmed_evaluable: int
    unique_track_confirmed_native_birth: int
    unique_track_confirmed_precision: Optional[float]
    unique_track_confirmed_retention: Optional[float]

    @property
    def event_precision(self) -> Optional[float]:
        return self.event_confirmed_precision

    @property
    def event_retention(self) -> Optional[float]:
        return self.event_confirmed_retention

    @property
    def unique_track_precision(self) -> Optional[float]:
        return self.unique_track_confirmed_precision

    @property
    def unique_track_retention(self) -> Optional[float]:
        return self.unique_track_confirmed_retention

    def to_json_dict(self) -> Dict[str, object]:
        result = dict(vars(self))
        result.update(
            {
                "event_precision": self.event_precision,
                "event_retention": self.event_retention,
                "unique_track_precision": self.unique_track_precision,
                "unique_track_retention": self.unique_track_retention,
            }
        )
        return result


@dataclass(frozen=True)
class SideBirthSceneReceipt:
    """Complete immutable terminal receipt for a scene."""

    scene_id: str
    terminal_frame_id: int
    last_keyframe_step: Optional[int]
    active_terminal_stable_ids: Tuple[int, ...]
    events: Tuple[SideBirthEventReceipt, ...]
    metrics: SideBirthProbationMetrics
    keyframes_observed: int
    pending_track_cap_hits: int
    pending_track_capacity_rejected_events: int
    birth_event_cap_hits: int
    birth_event_overflow_attempts: int
    birth_event_capacity_rejected_events: int
    audit_complete: bool
    enabled: bool
    min_distinct_keyframes: int
    max_missed_keyframes: int
    max_pending_tracks: int
    max_birth_events: int
    observer_only: bool = True
    training_free: bool = True
    causal: bool = True
    active_authorized: bool = False
    native_outputs_mutated: bool = False
    ground_truth_access: bool = False
    detector_score_access: bool = False
    clip_access: bool = False
    puf_access: bool = False
    puf_event_source: bool = True
    puf_state_access: bool = False
    native_labels_used_for_state_transitions: bool = False
    native_labels_used_at_close_only: bool = True
    schema: str = SCHEMA

    def to_json_dict(self) -> Dict[str, object]:
        """Return a fresh, strictly JSON-safe dictionary."""

        return {
            "schema": self.schema,
            "scene_id": self.scene_id,
            "terminal_frame_id": self.terminal_frame_id,
            "last_keyframe_step": self.last_keyframe_step,
            "active_terminal_stable_ids": list(
                self.active_terminal_stable_ids
            ),
            "events": [event.to_json_dict() for event in self.events],
            "metrics": self.metrics.to_json_dict(),
            "keyframes_observed": self.keyframes_observed,
            "event_count": len(self.events),
            "pending_track_cap_hits": self.pending_track_cap_hits,
            "pending_track_capacity_rejected_events": (
                self.pending_track_capacity_rejected_events
            ),
            "birth_event_cap_hits": self.birth_event_cap_hits,
            "birth_event_overflow_attempts": self.birth_event_overflow_attempts,
            "birth_event_capacity_rejected_events": (
                self.birth_event_capacity_rejected_events
            ),
            "audit_complete": self.audit_complete,
            "enabled": self.enabled,
            "thresholds_frozen": self.enabled,
            "bounded_memory": True,
            "true_cutr_keyframes_only": True,
            "caller_screened_birth_events_only": True,
            "effective_config": {
                "enabled": self.enabled,
                "observer_only": self.observer_only,
                "min_distinct_keyframes": self.min_distinct_keyframes,
                "max_missed_keyframes": self.max_missed_keyframes,
                "max_pending_tracks": self.max_pending_tracks,
                "max_birth_events": self.max_birth_events,
            },
            "observer_only": self.observer_only,
            "training_free": self.training_free,
            "module_training_free": self.training_free,
            "no_additional_training": self.training_free,
            "causal": self.causal,
            "active_authorized": self.active_authorized,
            "native_outputs_mutated": self.native_outputs_mutated,
            "ground_truth_access": self.ground_truth_access,
            "detector_score_access": self.detector_score_access,
            "clip_access": self.clip_access,
            "puf_access": self.puf_access,
            "puf_event_source": self.puf_event_source,
            "puf_state_access": self.puf_state_access,
            "event_source": "puf_arbitration_lite.action_birth",
            "puf_birth_event_input": True,
            "puf_internal_state_access": self.puf_state_access,
            "puf_access_semantics": "direct_module_access",
            "native_labels_used_for_state_transitions": (
                self.native_labels_used_for_state_transitions
            ),
            "native_labels_used_at_close_only": (
                self.native_labels_used_at_close_only
            ),
        }


class SideBirthCapacityError(RuntimeError):
    """Legacy exception kept for import compatibility.

    Event-cap overflow is now handled in-band so an indefinitely long online
    stream cannot be stopped by this observer.  No current ledger path raises
    this exception.
    """


@dataclass
class _TrackState:
    stable_id: int
    birth_frame_id: int
    birth_keyframe_step: int
    observed_frame_ids: list[int]
    observed_keyframe_steps: list[int]
    missed_keyframes: int = 0
    confirmation_frame_id: Optional[int] = None
    confirmation_keyframe_step: Optional[int] = None
    retirement_frame_id: Optional[int] = None
    retirement_keyframe_step: Optional[int] = None
    retirement_reason: Optional[str] = None

    @property
    def confirmed(self) -> bool:
        return self.confirmation_keyframe_step is not None

    @property
    def retired(self) -> bool:
        return self.retirement_frame_id is not None


@dataclass(frozen=True)
class _EventLink:
    event: SideBirthEvent
    disposition: str
    reason: Optional[str]


_SEED_KEYS = frozenset(
    {
        "proposal_id",
        "committed_stable_id",
        "top_probability",
        "margin",
        "native_target_kind",
        "native_target_ids",
    }
)
_CONTEXT_KEYS = frozenset({"scene_id", "frame_id", "keyframe_step"})


def _seed_from_value(
    value: object,
    *,
    index: int,
    scene_id: str,
    frame_id: int,
    keyframe_step: int,
) -> SideBirthSeedEvent:
    if isinstance(value, SideBirthSeedEvent):
        return value
    if not isinstance(value, Mapping):
        raise ValueError(
            f"birth_events[{index}] must be a SideBirthSeedEvent or mapping"
        )
    keys = set(value)
    unknown = sorted(keys - _SEED_KEYS - _CONTEXT_KEYS)
    if unknown:
        raise ValueError(
            f"birth_events[{index}] has unknown key(s): "
            + ", ".join(unknown)
        )
    missing = sorted(_SEED_KEYS - keys)
    if missing:
        raise ValueError(
            f"birth_events[{index}] is missing key(s): " + ", ".join(missing)
        )
    context_present = keys.intersection(_CONTEXT_KEYS)
    if context_present and context_present != _CONTEXT_KEYS:
        raise ValueError(
            f"birth_events[{index}] must contain all scene/frame/step fields"
        )
    if context_present:
        if _scene_id(value["scene_id"]) != scene_id:
            raise ValueError(f"birth_events[{index}] belongs to another scene")
        if _strict_int("event.frame_id", value["frame_id"]) != frame_id:
            raise ValueError(f"birth_events[{index}] has a mismatched frame_id")
        if (
            _strict_int("event.keyframe_step", value["keyframe_step"])
            != keyframe_step
        ):
            raise ValueError(
                f"birth_events[{index}] has a mismatched keyframe_step"
            )
    return SideBirthSeedEvent(
        proposal_id=value["proposal_id"],
        committed_stable_id=value["committed_stable_id"],
        top_probability=value["top_probability"],
        margin=value["margin"],
        native_target_kind=value["native_target_kind"],
        native_target_ids=value["native_target_ids"],
    )


class SideBirthProbationLiteLedger:
    """Causal, observer-only ledger for screened side-birth events."""

    def __init__(self, config: Optional[Mapping[str, object]] = None):
        resolved = resolve_side_birth_probation_lite_config(config)
        self.config = MappingProxyType(resolved)
        self.enabled = bool(resolved["enabled"])
        self.observer_only = True
        self._scene_id: Optional[str] = None
        self._last_frame_id: Optional[int] = None
        self._last_keyframe_step: Optional[int] = None
        self._keyframes_observed = 0
        self._tracks: Dict[int, _TrackState] = {}
        self._events: list[_EventLink] = []
        self._event_keys: set[Tuple[int, int]] = set()
        self._pending_track_cap_hits = 0
        self._pending_track_capacity_rejected_events = 0
        self._birth_event_cap_hits = 0
        self._birth_event_overflow_attempts = 0
        self._late_reappearances = 0
        self._late_seed_rejected_events = 0
        self._audit_complete = True
        self._closed_receipt: Optional[SideBirthSceneReceipt] = None
        self._pipeline_samples = deque(maxlen=2048)
        self._pipeline_ms_total = 0.0
        self._pipeline_ms_max = 0.0

    def _require_open_enabled(self) -> None:
        if not self.enabled:
            raise RuntimeError("side_birth_probation_lite is disabled")
        if self._closed_receipt is not None:
            raise RuntimeError("side-birth scene is already closed")

    def _normalize_events(
        self,
        birth_events: object,
        *,
        scene_id: str,
        frame_id: int,
        keyframe_step: int,
    ) -> Tuple[SideBirthSeedEvent, ...]:
        if isinstance(birth_events, (str, bytes, Mapping)):
            raise ValueError("birth_events must be a sequence of event rows")
        try:
            raw = tuple(birth_events)  # type: ignore[arg-type]
        except TypeError as error:
            raise ValueError(
                "birth_events must be a sequence of event rows"
            ) from error
        seeds = tuple(
            _seed_from_value(
                value,
                index=index,
                scene_id=scene_id,
                frame_id=frame_id,
                keyframe_step=keyframe_step,
            )
            for index, value in enumerate(raw)
        )
        proposal_ids = [seed.proposal_id for seed in seeds]
        if len(set(proposal_ids)) != len(proposal_ids):
            raise ValueError("birth_events must have unique proposal ids per frame")
        keys = {(frame_id, proposal_id) for proposal_id in proposal_ids}
        if self._event_keys.intersection(keys):
            raise ValueError("birth event was already enrolled")
        return seeds

    def observe_true_cutr_keyframe(
        self,
        *,
        scene_id: str,
        frame_id: int,
        keyframe_step: int,
        birth_events: Sequence[object],
        observed_stable_ids: Iterable[int],
    ) -> SideBirthKeyframeResult:
        """Atomically observe one real CuTR commit.

        ``birth_events`` must already contain only caller-screened births.
        Empty batches are required on later keyframes so that TTL advances.
        Duplicate stable ids in ``observed_stable_ids`` are intentionally
        collapsed: one committed track contributes at most one view per frame.

        If the bounded event receipt capacity cannot fit the complete incoming
        batch, no seed in that batch is enrolled.  The true-keyframe
        transaction itself still commits: existing tracks observe this frame,
        confirmation/TTL advances, and the next keyframe remains contiguous.
        This is fail-closed for supplemental births and non-blocking for the
        native online pipeline.
        """

        self._require_open_enabled()
        scene = _scene_id(scene_id)
        frame = _strict_int("frame_id", frame_id)
        step = _strict_int("keyframe_step", keyframe_step)
        active_ids = _active_ids("observed_stable_ids", observed_stable_ids)
        seeds = self._normalize_events(
            birth_events,
            scene_id=scene,
            frame_id=frame,
            keyframe_step=step,
        )

        # All validation precedes state mutation.  In particular, diagnostics
        # have been validated above but are not carried into transition logic.
        if self._scene_id is not None and scene != self._scene_id:
            raise ValueError(
                f"ledger is bound to scene {self._scene_id!r}, not {scene!r}"
            )
        if self._last_frame_id is not None and frame <= self._last_frame_id:
            raise ValueError("frame_id must be strictly increasing")
        if self._last_keyframe_step is not None and step != self._last_keyframe_step + 1:
            raise ValueError("keyframe_step must be contiguous and increasing by one")
        active_set = set(active_ids)
        for seed in seeds:
            stable_id = seed.committed_stable_id
            if stable_id is not None and stable_id not in active_set:
                raise ValueError(
                    "every resolved birth committed_stable_id must be present "
                    "in observed_stable_ids on its enrollment keyframe"
                )
        retired_reappeared = sorted(
            stable_id
            for stable_id in active_set
            if stable_id in self._tracks and self._tracks[stable_id].retired
        )

        event_limit = int(self.config["max_birth_events"])
        event_capacity_rejected = 0
        if len(self._events) + len(seeds) > event_limit:
            # Enrollment remains batch-atomic and fail-closed, while the
            # keyframe observation remains live.  Discarding the normalized
            # local tuple cannot mutate caller input and prevents rejected
            # identities from entering either the track or receipt ledgers.
            self._birth_event_cap_hits += 1
            self._birth_event_overflow_attempts += len(seeds)
            self._audit_complete = False
            event_capacity_rejected = len(seeds)
            seeds = ()

        if self._scene_id is None:
            self._scene_id = scene
        self._late_reappearances += len(retired_reappeared)

        newly_confirmed: list[int] = []
        newly_retired: list[int] = []
        threshold = int(self.config["min_distinct_keyframes"])
        ttl = int(self.config["max_missed_keyframes"])

        # First advance pre-existing tracks.  A confirmation or TTL retirement
        # on this keyframe can safely release a pending slot before new births
        # are enrolled.
        for stable_id in sorted(self._tracks):
            state = self._tracks[stable_id]
            if state.retired:
                continue
            if stable_id in active_set:
                state.observed_frame_ids.append(frame)
                state.observed_keyframe_steps.append(step)
                state.missed_keyframes = 0
                if not state.confirmed and len(state.observed_frame_ids) >= threshold:
                    state.confirmation_frame_id = frame
                    state.confirmation_keyframe_step = step
                    newly_confirmed.append(stable_id)
            else:
                state.missed_keyframes += 1
                if state.missed_keyframes > ttl:
                    state.retirement_frame_id = frame
                    state.retirement_keyframe_step = step
                    state.retirement_reason = "ttl_expired"
                    newly_retired.append(stable_id)

        pending_count = sum(
            not state.confirmed and not state.retired
            for state in self._tracks.values()
        )
        existing_ids = set(self._tracks)
        new_ids = sorted(
            {
                seed.committed_stable_id
                for seed in seeds
                if seed.committed_stable_id is not None
                and seed.committed_stable_id not in existing_ids
            }
        )
        available = max(
            int(self.config["max_pending_tracks"]) - pending_count, 0
        )
        accepted_new_ids = set(new_ids[:available])
        rejected_new_ids = set(new_ids[available:])
        if rejected_new_ids:
            self._pending_track_cap_hits += 1
            self._audit_complete = False

        for stable_id in sorted(accepted_new_ids):
            self._tracks[stable_id] = _TrackState(
                stable_id=stable_id,
                birth_frame_id=frame,
                birth_keyframe_step=step,
                observed_frame_ids=[frame],
                observed_keyframe_steps=[step],
            )

        first_index = len(self._events)
        bound_events: list[SideBirthEvent] = []
        rejected_event_indices: list[int] = []
        for offset, seed in enumerate(seeds):
            event_index = first_index + offset
            event = SideBirthEvent(
                event_index=event_index,
                scene_id=scene,
                frame_id=frame,
                keyframe_step=step,
                proposal_id=seed.proposal_id,
                committed_stable_id=seed.committed_stable_id,
                top_probability=seed.top_probability,
                margin=seed.margin,
                native_target_kind=seed.native_target_kind,
                native_target_ids=seed.native_target_ids,
            )
            if seed.committed_stable_id is None:
                disposition = "unresolved"
                reason = "unresolved_committed_stable_id"
            elif seed.committed_stable_id in rejected_new_ids:
                disposition = "capacity_rejected"
                reason = "max_pending_tracks_exceeded"
                rejected_event_indices.append(event_index)
            elif (
                seed.committed_stable_id in self._tracks
                and self._tracks[seed.committed_stable_id].retired
            ):
                disposition = "retired_identity_reused"
                reason = "stable_id_reappeared_after_probation_ttl"
                self._late_seed_rejected_events += 1
            else:
                disposition = "tracked"
                reason = None
            self._events.append(
                _EventLink(event=event, disposition=disposition, reason=reason)
            )
            self._event_keys.add((frame, seed.proposal_id))
            bound_events.append(event)

        self._pending_track_capacity_rejected_events += len(
            rejected_event_indices
        )
        self._last_frame_id = frame
        self._last_keyframe_step = step
        self._keyframes_observed += 1
        return self._keyframe_result(
            frame=frame,
            step=step,
            events=tuple(bound_events),
            observed_track_ids=tuple(
                stable_id
                for stable_id in active_ids
                if stable_id in self._tracks and not self._tracks[stable_id].retired
            ),
            newly_confirmed=tuple(newly_confirmed),
            newly_retired=tuple(newly_retired),
            rejected_event_indices=tuple(rejected_event_indices),
            birth_event_capacity_rejected_events=event_capacity_rejected,
        )

    def _keyframe_result(
        self,
        *,
        frame: int,
        step: int,
        events: Tuple[SideBirthEvent, ...],
        observed_track_ids: Tuple[int, ...],
        newly_confirmed: Tuple[int, ...],
        newly_retired: Tuple[int, ...],
        rejected_event_indices: Tuple[int, ...],
        birth_event_capacity_rejected_events: int,
    ) -> SideBirthKeyframeResult:
        pending = tuple(
            sorted(
                stable_id
                for stable_id, state in self._tracks.items()
                if not state.confirmed and not state.retired
            )
        )
        confirmed = tuple(
            sorted(
                stable_id
                for stable_id, state in self._tracks.items()
                if state.confirmed
            )
        )
        retired = tuple(
            sorted(
                stable_id
                for stable_id, state in self._tracks.items()
                if state.retired
            )
        )
        return SideBirthKeyframeResult(
            scene_id=self._scene_id or "",
            frame_id=frame,
            keyframe_step=step,
            enrolled_events=events,
            observed_track_ids=observed_track_ids,
            pending_track_ids=pending,
            confirmed_track_ids=confirmed,
            retired_track_ids=retired,
            newly_confirmed_track_ids=newly_confirmed,
            newly_retired_track_ids=newly_retired,
            capacity_rejected_event_indices=rejected_event_indices,
            birth_event_capacity_rejected_events=(
                birth_event_capacity_rejected_events
            ),
            audit_complete=self._audit_complete,
        )

    def snapshot(self) -> SideBirthProbationSnapshot:
        pending = tuple(
            sorted(
                stable_id
                for stable_id, state in self._tracks.items()
                if not state.confirmed and not state.retired
            )
        )
        confirmed = tuple(
            sorted(
                stable_id
                for stable_id, state in self._tracks.items()
                if state.confirmed
            )
        )
        retired = tuple(
            sorted(
                stable_id
                for stable_id, state in self._tracks.items()
                if state.retired
            )
        )
        return SideBirthProbationSnapshot(
            scene_id=self._scene_id,
            last_frame_id=self._last_frame_id,
            last_keyframe_step=self._last_keyframe_step,
            birth_events=len(self._events),
            pending_track_ids=pending,
            confirmed_track_ids=confirmed,
            retired_track_ids=retired,
            pending_track_cap_hits=self._pending_track_cap_hits,
            birth_event_cap_hits=self._birth_event_cap_hits,
            birth_event_capacity_rejected_events=(
                self._birth_event_overflow_attempts
            ),
            audit_complete=self._audit_complete,
            closed=self._closed_receipt is not None,
        )

    @staticmethod
    def _event_receipt(
        link: _EventLink,
        tracks: Mapping[int, _TrackState],
        terminal_active_ids: frozenset[int],
        confirmation_threshold: int,
    ) -> SideBirthEventReceipt:
        event = link.event
        if link.disposition == "unresolved":
            return SideBirthEventReceipt(
                event_index=event.event_index,
                scene_id=event.scene_id,
                frame_id=event.frame_id,
                keyframe_step=event.keyframe_step,
                proposal_id=event.proposal_id,
                committed_stable_id=None,
                top_probability=event.top_probability,
                margin=event.margin,
                native_target_kind=event.native_target_kind,
                native_target_ids=event.native_target_ids,
                observed_frame_ids=(),
                observed_keyframe_steps=(),
                confirmed=False,
                confirmation_frame_id=None,
                confirmation_keyframe_step=None,
                retired=True,
                retirement_frame_id=event.frame_id,
                retirement_keyframe_step=event.keyframe_step,
                active_at_terminal=False,
                status="retired_unresolved",
                latency_frames=None,
                latency_keyframes=None,
                reason=link.reason or "unresolved_committed_stable_id",
            )
        if link.disposition in {
            "capacity_rejected",
            "retired_identity_reused",
        }:
            identity_reused = link.disposition == "retired_identity_reused"
            return SideBirthEventReceipt(
                event_index=event.event_index,
                scene_id=event.scene_id,
                frame_id=event.frame_id,
                keyframe_step=event.keyframe_step,
                proposal_id=event.proposal_id,
                committed_stable_id=event.committed_stable_id,
                top_probability=event.top_probability,
                margin=event.margin,
                native_target_kind=event.native_target_kind,
                native_target_ids=event.native_target_ids,
                observed_frame_ids=(),
                observed_keyframe_steps=(),
                confirmed=False,
                confirmation_frame_id=None,
                confirmation_keyframe_step=None,
                retired=True,
                retirement_frame_id=event.frame_id,
                retirement_keyframe_step=event.keyframe_step,
                active_at_terminal=(
                    event.committed_stable_id in terminal_active_ids
                ),
                status=(
                    "retired_identity_reused"
                    if identity_reused
                    else "capacity_rejected"
                ),
                latency_frames=None,
                latency_keyframes=None,
                reason=(
                    link.reason
                    or (
                        "stable_id_reappeared_after_probation_ttl"
                        if identity_reused
                        else "max_pending_tracks_exceeded"
                    )
                ),
            )

        assert event.committed_stable_id is not None
        state = tracks[event.committed_stable_id]
        event_observations = tuple(
            (frame_id, step)
            for frame_id, step in zip(
                state.observed_frame_ids, state.observed_keyframe_steps
            )
            if step >= event.keyframe_step
        )
        confirmed = len(event_observations) >= confirmation_threshold
        if confirmed:
            confirmation_frame_id, confirmation_keyframe_step = (
                event_observations[confirmation_threshold - 1]
            )
        else:
            confirmation_frame_id = None
            confirmation_keyframe_step = None
        retired = state.retired
        if confirmed and retired:
            status = "confirmed_retired"
            reason = "third_distinct_true_keyframe"
        elif confirmed:
            status = "confirmed"
            reason = "third_distinct_true_keyframe"
        elif retired:
            status = "retired_probationary"
            reason = state.retirement_reason or "retired_before_confirmation"
        else:
            status = "probationary_unconfirmed"
            reason = "scene_closed_before_third_view"
        latency_frames = (
            max(confirmation_frame_id - event.frame_id, 0)
            if confirmation_frame_id is not None
            else None
        )
        latency_keyframes = (
            max(confirmation_keyframe_step - event.keyframe_step, 0)
            if confirmation_keyframe_step is not None
            else None
        )
        return SideBirthEventReceipt(
            event_index=event.event_index,
            scene_id=event.scene_id,
            frame_id=event.frame_id,
            keyframe_step=event.keyframe_step,
            proposal_id=event.proposal_id,
            committed_stable_id=event.committed_stable_id,
            top_probability=event.top_probability,
            margin=event.margin,
            native_target_kind=event.native_target_kind,
            native_target_ids=event.native_target_ids,
            observed_frame_ids=tuple(value[0] for value in event_observations),
            observed_keyframe_steps=tuple(value[1] for value in event_observations),
            confirmed=confirmed,
            confirmation_frame_id=confirmation_frame_id,
            confirmation_keyframe_step=confirmation_keyframe_step,
            retired=retired,
            retirement_frame_id=state.retirement_frame_id,
            retirement_keyframe_step=state.retirement_keyframe_step,
            active_at_terminal=(event.committed_stable_id in terminal_active_ids),
            status=status,
            latency_frames=latency_frames,
            latency_keyframes=latency_keyframes,
            reason=reason,
        )

    @staticmethod
    def _metrics(
        events: Tuple[SideBirthEventReceipt, ...]
    ) -> SideBirthProbationMetrics:
        """Read diagnostic native labels only after all state is terminal."""

        event_evaluable = sum(
            event.native_target_kind != "unresolved" for event in events
        )
        event_birth = sum(
            event.native_target_kind == "birth" for event in events
        )
        event_unique = sum(
            event.native_target_kind == "unique_history" for event in events
        )
        event_ambiguous = sum(
            event.native_target_kind == "ambiguous_history"
            for event in events
        )
        confirmed = tuple(event for event in events if event.confirmed)
        confirmed_evaluable = sum(
            event.native_target_kind != "unresolved" for event in confirmed
        )
        confirmed_birth = sum(
            event.native_target_kind == "birth" for event in confirmed
        )

        by_track: Dict[int, list[SideBirthEventReceipt]] = {}
        for event in events:
            if event.committed_stable_id is not None:
                by_track.setdefault(event.committed_stable_id, []).append(event)
        track_evaluable = 0
        track_unresolved = 0
        track_birth = 0
        track_history = 0
        track_mixed = 0
        track_confirmed = 0
        track_confirmed_evaluable = 0
        track_confirmed_birth = 0
        for rows in by_track.values():
            kinds = {
                row.native_target_kind
                for row in rows
                if row.native_target_kind != "unresolved"
            }
            is_confirmed = any(row.confirmed for row in rows)
            track_confirmed += int(is_confirmed)
            if not kinds:
                track_unresolved += 1
                continue
            track_evaluable += 1
            is_birth = kinds == {"birth"}
            if is_birth:
                track_birth += 1
            else:
                track_history += 1
            if len(kinds) > 1:
                track_mixed += 1
            if is_confirmed:
                track_confirmed_evaluable += 1
                track_confirmed_birth += int(is_birth)

        return SideBirthProbationMetrics(
            event_total=len(events),
            event_evaluable=event_evaluable,
            event_unresolved=len(events) - event_evaluable,
            event_native_birth=event_birth,
            event_native_unique_history=event_unique,
            event_native_ambiguous_history=event_ambiguous,
            event_confirmed=len(confirmed),
            event_confirmed_evaluable=confirmed_evaluable,
            event_confirmed_native_birth=confirmed_birth,
            event_confirmed_precision=(
                confirmed_birth / confirmed_evaluable
                if confirmed_evaluable
                else None
            ),
            event_confirmed_retention=(
                confirmed_birth / event_birth if event_birth else None
            ),
            unique_track_total=len(by_track),
            unique_track_evaluable=track_evaluable,
            unique_track_unresolved=track_unresolved,
            unique_track_native_birth=track_birth,
            unique_track_native_history=track_history,
            unique_track_mixed_labels=track_mixed,
            unique_track_confirmed=track_confirmed,
            unique_track_confirmed_evaluable=track_confirmed_evaluable,
            unique_track_confirmed_native_birth=track_confirmed_birth,
            unique_track_confirmed_precision=(
                track_confirmed_birth / track_confirmed_evaluable
                if track_confirmed_evaluable
                else None
            ),
            unique_track_confirmed_retention=(
                track_confirmed_birth / track_birth if track_birth else None
            ),
        )

    def close_scene(
        self,
        *,
        scene_id: str,
        terminal_frame_id: int,
        active_stable_ids: Iterable[int],
    ) -> SideBirthSceneReceipt:
        """Close the scene and compute label-based metrics exactly once."""

        self._require_open_enabled()
        scene = _scene_id(scene_id)
        terminal_frame = _strict_int("terminal_frame_id", terminal_frame_id)
        active_ids = _active_ids("active_stable_ids", active_stable_ids)
        if self._scene_id is not None and scene != self._scene_id:
            raise ValueError(
                f"ledger is bound to scene {self._scene_id!r}, not {scene!r}"
            )
        if self._last_frame_id is not None and terminal_frame < self._last_frame_id:
            raise ValueError("terminal_frame_id cannot precede the last keyframe")
        if self._scene_id is None:
            self._scene_id = scene
        active_set = set(active_ids)
        for stable_id, state in self._tracks.items():
            if not state.retired and stable_id not in active_set:
                state.retirement_frame_id = terminal_frame
                state.retirement_keyframe_step = self._last_keyframe_step
                state.retirement_reason = "terminal_inactive"

        event_receipts = tuple(
            self._event_receipt(
                link,
                self._tracks,
                frozenset(active_set),
                int(self.config["min_distinct_keyframes"]),
            )
            for link in self._events
        )
        receipt = SideBirthSceneReceipt(
            scene_id=scene,
            terminal_frame_id=terminal_frame,
            last_keyframe_step=self._last_keyframe_step,
            active_terminal_stable_ids=active_ids,
            events=event_receipts,
            metrics=self._metrics(event_receipts),
            keyframes_observed=self._keyframes_observed,
            pending_track_cap_hits=self._pending_track_cap_hits,
            pending_track_capacity_rejected_events=(
                self._pending_track_capacity_rejected_events
            ),
            birth_event_cap_hits=self._birth_event_cap_hits,
            birth_event_overflow_attempts=self._birth_event_overflow_attempts,
            birth_event_capacity_rejected_events=(
                self._birth_event_overflow_attempts
            ),
            audit_complete=self._audit_complete,
            enabled=self.enabled,
            min_distinct_keyframes=int(self.config["min_distinct_keyframes"]),
            max_missed_keyframes=int(self.config["max_missed_keyframes"]),
            max_pending_tracks=int(self.config["max_pending_tracks"]),
            max_birth_events=int(self.config["max_birth_events"]),
        )
        self._closed_receipt = receipt
        return receipt

    def record_pipeline_timing(self, observe_ms: object) -> None:
        """Record full wrapper time for one true-CuTR commit."""

        if isinstance(observe_ms, bool) or not isinstance(observe_ms, Real):
            raise ValueError("observe_ms must be finite and non-negative")
        value = float(observe_ms)
        if not isfinite(value) or value < 0.0:
            raise ValueError("observe_ms must be finite and non-negative")
        self._pipeline_samples.append(value)
        self._pipeline_ms_total += value
        self._pipeline_ms_max = max(self._pipeline_ms_max, value)

    def _timing_summary(self) -> Dict[str, object]:
        calls = len(self._pipeline_samples)
        return {
            "pipeline_observe_calls": calls,
            "pipeline_observe_ms_total": self._pipeline_ms_total,
            "pipeline_observe_ms_mean": (
                self._pipeline_ms_total / calls if calls else 0.0
            ),
            "pipeline_observe_ms_p95": (
                float(np.percentile(np.asarray(self._pipeline_samples), 95.0))
                if self._pipeline_samples
                else 0.0
            ),
            "pipeline_observe_ms_max": self._pipeline_ms_max,
        }

    def summary(self) -> Dict[str, object]:
        """Return operational safety evidence, or the terminal receipt."""

        if self._closed_receipt is not None:
            result = self._closed_receipt.to_json_dict()
            result["late_reappearances"] = self._late_reappearances
            result["late_seed_rejected_events"] = (
                self._late_seed_rejected_events
            )
            result.update(self._timing_summary())
            return result
        snapshot = self.snapshot()
        result = {
            "schema": SCHEMA,
            "enabled": self.enabled,
            "observer_only": True,
            "training_free": True,
            "module_training_free": True,
            "no_additional_training": True,
            "causal": True,
            "bounded_memory": True,
            "active_authorized": False,
            "native_outputs_mutated": False,
            "ground_truth_access": False,
            "detector_score_access": False,
            "clip_access": False,
            "puf_access": False,
            "puf_event_source": True,
            "puf_state_access": False,
            "event_source": "puf_arbitration_lite.action_birth",
            "puf_birth_event_input": True,
            "puf_internal_state_access": False,
            "puf_access_semantics": "direct_module_access",
            "native_labels_used_for_state_transitions": False,
            "native_labels_used_at_close_only": True,
            "thresholds_frozen": bool(self.enabled),
            "true_cutr_keyframes_only": True,
            "caller_screened_birth_events_only": True,
            "effective_config": dict(self.config),
            "scene_id": snapshot.scene_id,
            "last_frame_id": snapshot.last_frame_id,
            "last_keyframe_step": snapshot.last_keyframe_step,
            "keyframes_observed": self._keyframes_observed,
            "event_count": snapshot.birth_events,
            "pending_track_ids": list(snapshot.pending_track_ids),
            "confirmed_track_ids": list(snapshot.confirmed_track_ids),
            "retired_track_ids": list(snapshot.retired_track_ids),
            "pending_track_cap_hits": snapshot.pending_track_cap_hits,
            "pending_track_capacity_rejected_events": (
                self._pending_track_capacity_rejected_events
            ),
            "birth_event_cap_hits": snapshot.birth_event_cap_hits,
            "birth_event_overflow_attempts": self._birth_event_overflow_attempts,
            "birth_event_capacity_rejected_events": (
                snapshot.birth_event_capacity_rejected_events
            ),
            "late_reappearances": self._late_reappearances,
            "late_seed_rejected_events": self._late_seed_rejected_events,
            "audit_complete": snapshot.audit_complete,
            "closed": False,
        }
        result.update(self._timing_summary())
        return result


def build_side_birth_probation_lite(
    root_config: Optional[Mapping[str, object]] = None,
) -> SideBirthProbationLiteLedger:
    """Build from a root config or a direct side-birth section."""

    if root_config is None:
        section: Mapping[str, object] = {}
    elif not isinstance(root_config, Mapping):
        raise ValueError("root config must be a mapping")
    elif "side_birth_probation_lite" in root_config:
        candidate = root_config["side_birth_probation_lite"]
        if candidate is None:
            section = {}
        elif not isinstance(candidate, Mapping):
            raise ValueError(
                "side_birth_probation_lite section must be a mapping"
            )
        else:
            section = candidate
    elif set(root_config).issubset(DEFAULT_SIDE_BIRTH_PROBATION_LITE_CONFIG):
        section = root_config
    else:
        # Backward-compatible root configs leave the optional ledger disabled.
        section = {}
    return SideBirthProbationLiteLedger(section)


__all__ = [
    "DEFAULT_SIDE_BIRTH_PROBATION_LITE_CONFIG",
    "SCHEMA",
    "SideBirthCapacityError",
    "SideBirthEvent",
    "SideBirthEventReceipt",
    "SideBirthKeyframeResult",
    "SideBirthProbationLiteLedger",
    "SideBirthProbationMetrics",
    "SideBirthProbationSnapshot",
    "SideBirthSceneReceipt",
    "SideBirthSeedEvent",
    "build_side_birth_probation_lite",
    "resolve_side_birth_probation_lite_config",
]
