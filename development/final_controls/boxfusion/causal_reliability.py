"""Bounded causal multi-view reliability accumulation.

The accumulator is deliberately independent of a detector or mapper.  A
hypothesis is identified by a stable integer, and each update contributes one
matched or visible-unmatched observation from the current keyframe.  Evidence
is compressed into a bounded set of directionally diverse slots; no future
frame or scene-end replay is required.

The returned ``lower`` value is a beta-inspired soft-evidence reliability
bound.  It is not presented as a calibrated posterior probability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Hashable

import numpy as np


def _unit_direction(value: object) -> np.ndarray:
    direction = np.asarray(value, dtype=np.float64).reshape(3)
    if not np.isfinite(direction).all():
        raise ValueError("view direction must be finite")
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        raise ValueError("view direction must be nonzero")
    return direction / norm


@dataclass(frozen=True)
class ReliabilityObservation:
    frame_id: int
    ordinal: int
    direction: np.ndarray
    strength: float
    visibility: float
    matched: bool

    def __post_init__(self) -> None:
        direction = _unit_direction(self.direction)
        strength = float(self.strength)
        visibility = float(self.visibility)
        if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError("strength must lie in [0,1]")
        if not math.isfinite(visibility) or not 0.0 <= visibility <= 1.0:
            raise ValueError("visibility must lie in [0,1]")
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "ordinal", int(self.ordinal))
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "strength", strength)
        object.__setattr__(self, "visibility", visibility)
        object.__setattr__(self, "matched", bool(self.matched))

    @property
    def priority(self) -> tuple[float, float, int, int]:
        # A matched observation always dominates an unmatched observation in
        # the same direction.  Stronger and more visible evidence wins; the
        # newest frame resolves exact ties deterministically.
        return (
            1.0 if self.matched else 0.0,
            self.strength if self.matched else self.visibility,
            self.ordinal,
            self.frame_id,
        )


@dataclass
class _HypothesisState:
    slots: list[ReliabilityObservation] = field(default_factory=list)
    first_support: float | None = None
    maximum_support: float = 0.0
    support_sum: float = 0.0
    support_count: int = 0
    ema_support: float = 0.0
    last_seen_ordinal: int = -1
    geometry_resets: int = 0

    def clear_evidence(self) -> None:
        self.slots.clear()
        self.first_support = None
        self.maximum_support = 0.0
        self.support_sum = 0.0
        self.support_count = 0
        self.ema_support = 0.0


class CausalReliabilityState:
    """Bounded multi-view evidence state shared by recovery and reranking."""

    def __init__(
        self,
        *,
        max_view_slots: int = 8,
        min_angular_separation_deg: float = 30.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
        residual_weight: float = 0.25,
        negative_weight: float = 0.25,
        lower_bound_kappa: float = 1.0,
        ema_decay: float = 0.8,
        state_ttl_keyframes: int = 10,
        max_states: int = 4096,
    ) -> None:
        if max_view_slots < 1 or max_states < 1 or state_ttl_keyframes < 0:
            raise ValueError("invalid reliability-state bounds")
        if not 0.0 < min_angular_separation_deg <= 180.0:
            raise ValueError("min_angular_separation_deg must lie in (0,180]")
        for name, value in (
            ("alpha0", alpha0),
            ("beta0", beta0),
            ("residual_weight", residual_weight),
            ("negative_weight", negative_weight),
            ("lower_bound_kappa", lower_bound_kappa),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if alpha0 <= 0.0 or beta0 <= 0.0:
            raise ValueError("alpha0 and beta0 must be positive")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must lie in [0,1)")

        self.max_view_slots = int(max_view_slots)
        self.minimum_direction_cosine = math.cos(
            math.radians(float(min_angular_separation_deg))
        )
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        self.residual_weight = float(residual_weight)
        self.negative_weight = float(negative_weight)
        self.lower_bound_kappa = float(lower_bound_kappa)
        self.ema_decay = float(ema_decay)
        self.state_ttl_keyframes = int(state_ttl_keyframes)
        self.max_states = int(max_states)
        self.states: dict[Hashable, _HypothesisState] = {}
        self.capacity_drops = 0
        self.slot_replacements = 0
        self.geometry_resets = 0

    def _state(self, identity: Hashable, ordinal: int) -> _HypothesisState:
        state = self.states.get(identity)
        if state is not None:
            state.last_seen_ordinal = max(state.last_seen_ordinal, int(ordinal))
            return state
        if len(self.states) >= self.max_states:
            oldest = min(
                self.states,
                key=lambda key: (self.states[key].last_seen_ordinal, repr(key)),
            )
            del self.states[oldest]
            self.capacity_drops += 1
        state = _HypothesisState(last_seen_ordinal=int(ordinal))
        self.states[identity] = state
        return state

    def prune(self, ordinal: int, current_ids: set[Hashable] | None = None) -> None:
        protected = set() if current_ids is None else set(current_ids)
        expired = [
            identity
            for identity, state in self.states.items()
            if identity not in protected
            and int(ordinal) - state.last_seen_ordinal > self.state_ttl_keyframes
        ]
        for identity in expired:
            del self.states[identity]

    def reset_evidence(self, identity: Hashable, *, ordinal: int) -> None:
        state = self._state(identity, ordinal)
        state.clear_evidence()
        state.geometry_resets += 1
        self.geometry_resets += 1

    def discard(self, identity: Hashable) -> None:
        """Remove one hypothesis and all of its accumulated evidence."""
        self.states.pop(identity, None)

    def identities(self) -> tuple[Hashable, ...]:
        """Return a deterministic snapshot of the active identities."""
        return tuple(sorted(self.states, key=repr))

    def update(
        self,
        identity: Hashable,
        *,
        frame_id: int,
        ordinal: int,
        view_direction: object,
        strength: float,
        visibility: float,
        matched: bool,
    ) -> None:
        observation = ReliabilityObservation(
            frame_id=frame_id,
            ordinal=ordinal,
            direction=view_direction,
            strength=strength,
            visibility=visibility,
            matched=matched,
        )
        state = self._state(identity, ordinal)
        if observation.matched:
            value = observation.strength
            if state.first_support is None:
                state.first_support = value
                state.ema_support = value
            else:
                state.ema_support = (
                    self.ema_decay * state.ema_support
                    + (1.0 - self.ema_decay) * value
                )
            state.maximum_support = max(state.maximum_support, value)
            state.support_sum += value
            state.support_count += 1

        if not state.slots:
            state.slots.append(observation)
            return
        comparable = [
            index
            for index, slot in enumerate(state.slots)
            if slot.matched == observation.matched
        ]
        similarities = np.asarray(
            [
                float(np.dot(observation.direction, state.slots[index].direction))
                for index in comparable
            ]
        )
        nearest = comparable[int(np.argmax(similarities))] if comparable else None
        if (
            nearest is not None
            and float(np.max(similarities)) >= self.minimum_direction_cosine
        ):
            if observation.priority > state.slots[nearest].priority:
                state.slots[nearest] = observation
                self.slot_replacements += 1
            return
        if len(state.slots) < self.max_view_slots:
            state.slots.append(observation)
            return
        weakest = min(
            range(len(state.slots)), key=lambda index: state.slots[index].priority
        )
        if observation.priority > state.slots[weakest].priority:
            state.slots[weakest] = observation
            self.slot_replacements += 1

    def summary(self, identity: Hashable) -> dict[str, float | int]:
        state = self.states.get(identity)
        if state is None:
            return {
                "alpha": self.alpha0,
                "beta": self.beta0,
                "mean": self.alpha0 / (self.alpha0 + self.beta0),
                "std": 0.0,
                "lower": 0.0,
                "effective_views": 0,
                "positive_views": 0,
                "negative_views": 0,
                "first": 0.0,
                "mean_support": 0.0,
                "max": 0.0,
                "ema": 0.0,
                "diverse_max": 0.0,
            }
        positive = [slot for slot in state.slots if slot.matched]
        negative = [slot for slot in state.slots if not slot.matched]
        alpha = self.alpha0 + sum(
            slot.strength * slot.visibility for slot in positive
        )
        beta = self.beta0 + self.residual_weight * sum(
            (1.0 - slot.strength) * slot.visibility for slot in positive
        ) + self.negative_weight * sum(slot.visibility for slot in negative)
        total = alpha + beta
        mean = alpha / total
        variance = alpha * beta / (total * total * (total + 1.0))
        std = math.sqrt(max(0.0, variance))
        lower = float(np.clip(mean - self.lower_bound_kappa * std, 0.0, 1.0))
        return {
            "alpha": float(alpha),
            "beta": float(beta),
            "mean": float(mean),
            "std": float(std),
            "lower": lower,
            "effective_views": len(state.slots),
            "positive_views": len(positive),
            "negative_views": len(negative),
            "first": 0.0 if state.first_support is None else state.first_support,
            "mean_support": (
                state.support_sum / state.support_count
                if state.support_count
                else 0.0
            ),
            "max": state.maximum_support,
            "ema": state.ema_support,
            "diverse_max": max((slot.strength for slot in positive), default=0.0),
        }

    def diagnostics(self) -> dict[str, int | float]:
        return {
            "states": len(self.states),
            "max_states": self.max_states,
            "max_view_slots": self.max_view_slots,
            "capacity_drops": self.capacity_drops,
            "slot_replacements": self.slot_replacements,
            "geometry_resets": self.geometry_resets,
        }
