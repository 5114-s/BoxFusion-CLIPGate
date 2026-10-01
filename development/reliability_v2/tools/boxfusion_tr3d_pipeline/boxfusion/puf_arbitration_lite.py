"""Conservative training-free arbitration for PUF-lite shadow decisions.

PUF-lite scores proposals independently, so duplicate CuTR proposals can pick
the same historical track.  This module never reassigns a loser to a weaker
track and never suppresses it.  It selects at most one high-confidence PUF
recommendation per historical track and marks every other proposal as an
explicit native-BoxFusion fallback.

The module is observer-only.  Native targets are accepted after association
solely for diagnostics and cannot affect later arbitration.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from numbers import Integral, Real
from time import perf_counter_ns
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .puf_lite import PUFProposalDecision, PUFQueryBatch


DEFAULT_PUF_ARBITRATION_LITE_CONFIG = {
    "enabled": False,
    "observer_only": True,
    "track_min_probability": 0.70,
    "track_min_margin": 0.20,
    "birth_min_probability": 0.70,
    "birth_min_margin": 0.20,
    "conflict_min_owner_gap": 0.10,
    "max_proposals": 256,
    "probability_tolerance": 1e-12,
    "max_diagnostic_examples": 64,
}


def _strict_bool(name: str, value: object) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _strict_int(name: str, value: object, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_float(name: str, value: object, minimum: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return result


def resolve_puf_arbitration_lite_config(
    config: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("puf_arbitration_lite config must be a mapping")
    unknown = sorted(set(config) - set(DEFAULT_PUF_ARBITRATION_LITE_CONFIG))
    if unknown:
        raise ValueError(
            "Unknown puf_arbitration_lite config key(s): "
            + ", ".join(unknown)
        )
    resolved = dict(DEFAULT_PUF_ARBITRATION_LITE_CONFIG)
    resolved.update(config)
    resolved["enabled"] = _strict_bool(
        "puf_arbitration_lite.enabled", resolved["enabled"]
    )
    resolved["observer_only"] = _strict_bool(
        "puf_arbitration_lite.observer_only", resolved["observer_only"]
    )
    if resolved["enabled"] and not resolved["observer_only"]:
        raise ValueError(
            "puf_arbitration_lite active association is not authorized; "
            "observer_only must remain true"
        )
    for key in (
        "track_min_probability",
        "track_min_margin",
        "birth_min_probability",
        "birth_min_margin",
        "conflict_min_owner_gap",
        "probability_tolerance",
    ):
        resolved[key] = _finite_float(
            f"puf_arbitration_lite.{key}", resolved[key], 0.0
        )
    for key in (
        "track_min_probability",
        "track_min_margin",
        "birth_min_probability",
        "birth_min_margin",
        "conflict_min_owner_gap",
    ):
        if resolved[key] > 1.0:
            raise ValueError(f"puf_arbitration_lite.{key} must not exceed 1")
    if resolved["probability_tolerance"] <= 0.0:
        raise ValueError(
            "puf_arbitration_lite.probability_tolerance must be positive"
        )
    if resolved["probability_tolerance"] > 1e-6:
        raise ValueError(
            "puf_arbitration_lite.probability_tolerance must not exceed 1e-6"
        )
    resolved["max_proposals"] = _strict_int(
        "puf_arbitration_lite.max_proposals",
        resolved["max_proposals"],
        1,
    )
    if resolved["max_proposals"] > 1024:
        raise ValueError(
            "puf_arbitration_lite.max_proposals must not exceed 1024"
        )
    resolved["max_diagnostic_examples"] = _strict_int(
        "puf_arbitration_lite.max_diagnostic_examples",
        resolved["max_diagnostic_examples"],
        0,
    )
    if resolved["enabled"]:
        frozen = {
            "track_min_probability": 0.70,
            "track_min_margin": 0.20,
            "birth_min_probability": 0.70,
            "birth_min_margin": 0.20,
            "conflict_min_owner_gap": 0.10,
        }
        changed = [
            key for key, value in frozen.items() if resolved[key] != value
        ]
        if changed:
            raise ValueError(
                "enabled puf_arbitration_lite must keep frozen thresholds: "
                + ", ".join(changed)
            )
    return resolved


@dataclass(frozen=True)
class ArbitrationDecision:
    proposal_id: int
    source_valid: bool
    source_conflict: bool
    raw_predicted_birth: Optional[bool]
    raw_predicted_track_id: Optional[int]
    top_probability: Optional[float]
    competitor_probability: Optional[float]
    margin: Optional[float]
    confidence_eligible: bool
    conflict_group_size: int
    conflict_winner: bool
    action: str
    selected_track_id: Optional[int]
    selected_global_row: Optional[int]
    selected_source: Optional[str]
    reason: str


@dataclass(frozen=True)
class PUFArbitrationBatch:
    scene_id: str
    frame_id: int
    history_max_frame_id: Optional[int]
    proposal_ids: Tuple[int, ...]
    rows: Tuple[ArbitrationDecision, ...]
    query_ms: float


@dataclass
class _WorkingDecision:
    source: PUFProposalDecision
    top_probability: Optional[float]
    competitor_probability: Optional[float]
    margin: Optional[float]
    confidence_eligible: bool
    selected_global_row: Optional[int]
    selected_source: Optional[str]
    conflict_group_size: int = 0
    conflict_winner: bool = False
    action: str = "native_fallback"
    reason: str = "uninitialized"


class PUFArbitrationLiteObserver:
    """Select at most one high-confidence override per historical track."""

    _LATENCY_WINDOW = 2048

    def __init__(self, config: Optional[Mapping[str, object]] = None):
        self.config = resolve_puf_arbitration_lite_config(config)
        self.enabled = bool(self.config["enabled"])
        self.observer_only = bool(self.config["observer_only"])
        self._scene_id: Optional[str] = None
        self._last_query_frame_id: Optional[int] = None
        self._pending_batch: Optional[PUFArbitrationBatch] = None
        self._last_observed_batch: Optional[PUFArbitrationBatch] = None
        self._query_samples = deque(maxlen=self._LATENCY_WINDOW)
        self._examples: list[dict[str, object]] = []
        self._stats = self._new_stats()

    @staticmethod
    def _new_stats() -> Dict[str, object]:
        return {
            "queries": 0,
            "proposals": 0,
            "source_valid_rows": 0,
            "source_invalid_rows": 0,
            "source_conflict_groups": 0,
            "source_conflict_rows": 0,
            "proposal_cap_batches": 0,
            "high_confidence_track_rows": 0,
            "high_confidence_birth_rows": 0,
            "selected_track_overrides": 0,
            "selected_birth_overrides": 0,
            "native_fallback_rows": 0,
            "conflict_groups_with_winner": 0,
            "conflict_groups_without_winner": 0,
            "conflict_tie_abstentions": 0,
            "conflict_low_confidence_abstentions": 0,
            "conflict_winners": 0,
            "conflict_losers_deferred": 0,
            "duplicate_selected_tracks": 0,
            "native_history_unique": 0,
            "native_history_ambiguous": 0,
            "native_births": 0,
            "native_unresolved": 0,
            "selected_evaluable": 0,
            "selected_correct": 0,
            "selected_wrong": 0,
            "selected_track_evaluable": 0,
            "selected_track_correct": 0,
            "selected_birth_evaluable": 0,
            "selected_birth_correct": 0,
            "false_track_overrides": 0,
            "false_birth_overrides": 0,
            "conflict_winner_evaluable": 0,
            "conflict_winner_correct": 0,
            "conflict_native_supported_groups": 0,
            "conflict_native_unique_positive_groups": 0,
            "conflict_native_multi_positive_groups": 0,
            "conflict_native_unsupported_groups": 0,
            "conflict_native_unresolved_groups": 0,
            "conflict_owner_group_evaluable": 0,
            "conflict_owner_group_correct": 0,
            "conflict_loser_rows": 0,
            "conflict_loser_native_positive": 0,
            "fallback_on_unique_history": 0,
            "fallback_on_ambiguous_history": 0,
            "fallback_on_birth": 0,
            "query_ms_total": 0.0,
            "query_ms_max": 0.0,
            "pipeline_query_calls": 0,
            "pipeline_query_ms_total": 0.0,
            "pipeline_query_ms_max": 0.0,
            "pipeline_observe_calls": 0,
            "pipeline_observe_ms_total": 0.0,
            "pipeline_observe_ms_max": 0.0,
        }

    def reset_scene(self, scene_id: str) -> None:
        scene_id = str(scene_id)
        if not scene_id:
            raise ValueError("scene_id must not be empty")
        self._scene_id = scene_id
        self._last_query_frame_id = None
        self._pending_batch = None
        self._last_observed_batch = None
        self._query_samples.clear()
        self._examples.clear()
        self._stats = self._new_stats()

    def _bind_scene(self, scene_id: str) -> str:
        scene_id = str(scene_id)
        if not scene_id:
            raise ValueError("scene_id must not be empty")
        if self._scene_id is None:
            self.reset_scene(scene_id)
        elif scene_id != self._scene_id:
            raise ValueError(
                f"puf_arbitration_lite is bound to {self._scene_id}, "
                f"not {scene_id}"
            )
        return scene_id

    def _prepare(self, row: PUFProposalDecision) -> _WorkingDecision:
        if not isinstance(row, PUFProposalDecision):
            raise ValueError("PUF arbitration rows must be PUFProposalDecision")
        if not row.valid:
            return _WorkingDecision(
                source=row,
                top_probability=None,
                competitor_probability=None,
                margin=None,
                confidence_eligible=False,
                selected_global_row=None,
                selected_source=None,
                action="native_fallback",
                reason="invalid_puf_native_fallback",
            )
        if row.birth_probability is None or row.predicted_birth is None:
            raise ValueError("valid PUF row is missing a birth posterior")
        probabilities = [float(item.probability) for item in row.candidates]
        probabilities.append(float(row.birth_probability))
        if (
            not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities)
            or abs(sum(probabilities) - 1.0)
            > float(self.config["probability_tolerance"])
        ):
            raise ValueError("valid PUF row contains an invalid probability distribution")
        candidate_ids = [int(item.track_id) for item in row.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("valid PUF row contains duplicate candidate track IDs")
        if len(candidate_ids) > 3:
            raise ValueError("valid PUF row exceeds the Top-3 candidate bound")

        if row.predicted_birth:
            if row.predicted_track_id is not None or row.predicted_global_row is not None:
                raise ValueError("birth PUF row must not select a track")
            top = float(row.birth_probability)
            competitor = max(
                (float(item.probability) for item in row.candidates),
                default=0.0,
            )
            margin = top - competitor
            eligible = (
                top >= float(self.config["birth_min_probability"])
                and margin >= float(self.config["birth_min_margin"])
            )
            return _WorkingDecision(
                source=row,
                top_probability=top,
                competitor_probability=competitor,
                margin=margin,
                confidence_eligible=eligible,
                selected_global_row=None,
                selected_source=None,
                action="birth" if eligible else "native_fallback",
                reason=(
                    "high_confidence_birth"
                    if eligible
                    else "low_confidence_birth_native_fallback"
                ),
            )

        if row.predicted_track_id is None or row.predicted_global_row is None:
            raise ValueError("track PUF row is missing its selected track")
        selected = [
            item
            for item in row.candidates
            if int(item.track_id) == int(row.predicted_track_id)
        ]
        if len(selected) != 1:
            raise ValueError("PUF selected track is absent or duplicated")
        if int(selected[0].global_row) != int(row.predicted_global_row):
            raise ValueError(
                "PUF selected track/global-row snapshot is inconsistent"
            )
        top = float(selected[0].probability)
        competitor = max(
            [float(row.birth_probability)]
            + [
                float(item.probability)
                for item in row.candidates
                if int(item.track_id) != int(row.predicted_track_id)
            ]
        )
        margin = top - competitor
        eligible = (
            top >= float(self.config["track_min_probability"])
            and margin >= float(self.config["track_min_margin"])
            and selected[0].source == "qim"
        )
        return _WorkingDecision(
            source=row,
            top_probability=top,
            competitor_probability=competitor,
            margin=margin,
            confidence_eligible=eligible,
            selected_global_row=int(row.predicted_global_row),
            selected_source=str(selected[0].source),
            action="track" if eligible else "native_fallback",
            reason=(
                "high_confidence_track"
                if eligible
                else "low_confidence_track_native_fallback"
            ),
        )

    @staticmethod
    def _winner_key(
        item: _WorkingDecision,
    ) -> Tuple[float, float, float, float, float, int, int, int]:
        selected = next(
            candidate
            for candidate in item.source.candidates
            if candidate.track_id == item.source.predicted_track_id
        )
        return (
            -float(item.top_probability),
            -float(item.margin),
            -float(selected.likelihood),
            -float(selected.overlap_support),
            -float(selected.shared_key_fraction),
            0 if selected.source == "qim" else 1,
            int(selected.qim_rank) if selected.qim_rank is not None else 1 << 30,
            int(item.source.proposal_id),
        )

    def query(self, *, puf_batch: PUFQueryBatch) -> PUFArbitrationBatch:
        """Freeze conservative arbitration before native association."""

        start = perf_counter_ns()
        if not self.enabled:
            raise RuntimeError("puf_arbitration_lite observer is disabled")
        if self._pending_batch is not None:
            raise ValueError("previous arbitration query must be observed first")
        if not isinstance(puf_batch, PUFQueryBatch):
            raise ValueError("puf_batch must be a PUFQueryBatch")
        scene_id = self._bind_scene(puf_batch.scene_id)
        frame_id = int(puf_batch.frame_id)
        if self._last_query_frame_id is not None and frame_id <= self._last_query_frame_id:
            raise ValueError("arbitration frame ids must be strictly increasing")
        if puf_batch.history_max_frame_id is not None and puf_batch.history_max_frame_id >= frame_id:
            raise ValueError("arbitration history must precede the current frame")
        if len(puf_batch.rows) != len(puf_batch.proposal_ids):
            raise ValueError("PUF rows must align with proposal IDs")
        if tuple(row.proposal_id for row in puf_batch.rows) != tuple(puf_batch.proposal_ids):
            raise ValueError("PUF row proposal IDs are not aligned")
        if len(set(puf_batch.proposal_ids)) != len(puf_batch.proposal_ids):
            raise ValueError("PUF proposal IDs must be unique")

        cap_exceeded = len(puf_batch.rows) > int(self.config["max_proposals"])
        if cap_exceeded:
            working = [
                _WorkingDecision(
                    source=row,
                    top_probability=None,
                    competitor_probability=None,
                    margin=None,
                    confidence_eligible=False,
                    selected_global_row=None,
                    selected_source=None,
                    action="native_fallback",
                    reason="proposal_cap_native_fallback",
                )
                for row in puf_batch.rows
            ]
        else:
            working = [self._prepare(row) for row in puf_batch.rows]

        track_groups: Dict[int, list[_WorkingDecision]] = {}
        for item in working:
            if (
                item.source.valid
                and item.source.predicted_track_id is not None
                and item.top_probability is not None
            ):
                track_groups.setdefault(
                    int(item.source.predicted_track_id), []
                ).append(item)

        conflict_groups = 0
        conflict_rows = 0
        conflict_groups_with_winner = 0
        conflict_tie_abstentions = 0
        conflict_low_confidence_abstentions = 0
        for track_id, group in sorted(track_groups.items()):
            group_size = len(group)
            for item in group:
                item.conflict_group_size = group_size
            if group_size < 2:
                continue
            conflict_groups += 1
            conflict_rows += group_size
            ordered = sorted(group, key=self._winner_key)
            best = ordered[0]
            if not best.confidence_eligible:
                for item in group:
                    item.action = "native_fallback"
                    item.reason = "conflict_low_confidence_native_fallback"
                conflict_low_confidence_abstentions += 1
                continue
            owner_gap = float(best.top_probability) - float(
                ordered[1].top_probability
            )
            if owner_gap < float(self.config["conflict_min_owner_gap"]):
                for item in group:
                    item.action = "native_fallback"
                    item.reason = "conflict_owner_gap_native_fallback"
                conflict_tie_abstentions += 1
                continue
            winner = best
            winner.conflict_winner = True
            winner.action = "track"
            winner.reason = "conflict_winner"
            conflict_groups_with_winner += 1
            for item in group:
                if item is winner:
                    continue
                item.action = "native_fallback"
                item.reason = "conflict_loser_native_fallback"

        selected_track_ids = [
            int(item.source.predicted_track_id)
            for item in working
            if item.action == "track"
        ]
        duplicate_selected = len(selected_track_ids) - len(set(selected_track_ids))
        if duplicate_selected:
            raise RuntimeError("arbitration produced duplicate selected tracks")

        rows = tuple(
            ArbitrationDecision(
                proposal_id=int(item.source.proposal_id),
                source_valid=bool(item.source.valid),
                source_conflict=bool(item.source.conflict),
                raw_predicted_birth=item.source.predicted_birth,
                raw_predicted_track_id=item.source.predicted_track_id,
                top_probability=item.top_probability,
                competitor_probability=item.competitor_probability,
                margin=item.margin,
                confidence_eligible=bool(item.confidence_eligible),
                conflict_group_size=int(item.conflict_group_size),
                conflict_winner=bool(item.conflict_winner),
                action=item.action,
                selected_track_id=(
                    int(item.source.predicted_track_id)
                    if item.action == "track"
                    else None
                ),
                selected_global_row=(
                    int(item.selected_global_row)
                    if item.action == "track"
                    else None
                ),
                selected_source=(
                    item.selected_source if item.action == "track" else None
                ),
                reason=item.reason,
            )
            for item in working
        )
        elapsed_ms = (perf_counter_ns() - start) / 1e6
        batch = PUFArbitrationBatch(
            scene_id=scene_id,
            frame_id=frame_id,
            history_max_frame_id=puf_batch.history_max_frame_id,
            proposal_ids=tuple(int(value) for value in puf_batch.proposal_ids),
            rows=rows,
            query_ms=float(elapsed_ms),
        )
        self._last_query_frame_id = frame_id
        self._pending_batch = batch
        self._stats["queries"] += 1
        self._stats["proposals"] += len(rows)
        self._stats["source_valid_rows"] += sum(row.source_valid for row in rows)
        self._stats["source_invalid_rows"] += sum(not row.source_valid for row in rows)
        self._stats["source_conflict_groups"] += conflict_groups
        self._stats["source_conflict_rows"] += conflict_rows
        self._stats["proposal_cap_batches"] += int(cap_exceeded)
        self._stats["high_confidence_track_rows"] += sum(
            row.confidence_eligible and row.raw_predicted_track_id is not None
            for row in rows
        )
        self._stats["high_confidence_birth_rows"] += sum(
            row.confidence_eligible and row.raw_predicted_birth is True
            for row in rows
        )
        self._stats["selected_track_overrides"] += sum(
            row.action == "track" for row in rows
        )
        self._stats["selected_birth_overrides"] += sum(
            row.action == "birth" for row in rows
        )
        self._stats["native_fallback_rows"] += sum(
            row.action == "native_fallback" for row in rows
        )
        self._stats["conflict_groups_with_winner"] += conflict_groups_with_winner
        self._stats["conflict_groups_without_winner"] += (
            conflict_groups - conflict_groups_with_winner
        )
        self._stats["conflict_tie_abstentions"] += conflict_tie_abstentions
        self._stats["conflict_low_confidence_abstentions"] += (
            conflict_low_confidence_abstentions
        )
        self._stats["conflict_winners"] += sum(row.conflict_winner for row in rows)
        self._stats["conflict_losers_deferred"] += sum(
            row.conflict_group_size > 1
            and not row.conflict_winner
            and row.action == "native_fallback"
            for row in rows
        )
        self._stats["duplicate_selected_tracks"] += duplicate_selected
        self._stats["query_ms_total"] += elapsed_ms
        self._stats["query_ms_max"] = max(
            float(self._stats["query_ms_max"]), elapsed_ms
        )
        self._query_samples.append(float(elapsed_ms))
        return batch

    def _add_example(
        self,
        row: ArbitrationDecision,
        native_kind: str,
        native_ids: Sequence[int],
    ) -> None:
        if len(self._examples) >= int(self.config["max_diagnostic_examples"]):
            return
        self._examples.append(
            {
                "frame_id": self._pending_batch.frame_id if self._pending_batch else None,
                "proposal_id": row.proposal_id,
                "native_kind": native_kind,
                "native_track_ids": tuple(int(value) for value in native_ids),
                "action": row.action,
                "selected_track_id": row.selected_track_id,
                "top_probability": row.top_probability,
                "margin": row.margin,
                "reason": row.reason,
                "conflict_group_size": row.conflict_group_size,
            }
        )

    def observe_native_targets(
        self,
        batch: PUFArbitrationBatch,
        native_target_track_ids: Sequence[Optional[Iterable[int]]],
    ) -> None:
        """Record selective precision without changing future decisions."""

        if batch is self._last_observed_batch:
            raise ValueError("arbitration batch was already observed")
        if batch is not self._pending_batch:
            raise ValueError("native targets require the pending arbitration batch")
        if len(native_target_track_ids) != len(batch.rows):
            raise ValueError("native targets must align with arbitration rows")
        normalized = []
        for raw_targets in native_target_track_ids:
            if raw_targets is None:
                normalized.append(None)
                continue
            targets = set()
            for value in raw_targets:
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                    raise ValueError("native target IDs must contain integers")
                if int(value) < 0:
                    raise ValueError("native target IDs must be non-negative")
                targets.add(int(value))
            normalized.append(frozenset(targets))

        conflict_groups: Dict[int, list[int]] = {}
        for index, row in enumerate(batch.rows):
            if (
                row.raw_predicted_track_id is not None
                and row.conflict_group_size > 1
            ):
                conflict_groups.setdefault(
                    int(row.raw_predicted_track_id), []
                ).append(index)
        for track_id, indices in sorted(conflict_groups.items()):
            positives = [
                index
                for index in indices
                if normalized[index] is not None
                and track_id in normalized[index]
            ]
            has_unresolved = any(normalized[index] is None for index in indices)
            if has_unresolved:
                self._stats["conflict_native_unresolved_groups"] += 1
            elif positives:
                self._stats["conflict_native_supported_groups"] += 1
                if len(positives) == 1:
                    self._stats["conflict_native_unique_positive_groups"] += 1
                else:
                    self._stats["conflict_native_multi_positive_groups"] += 1
            else:
                self._stats["conflict_native_unsupported_groups"] += 1

            owners = [
                index for index in indices if batch.rows[index].conflict_winner
            ]
            if len(owners) > 1:
                raise RuntimeError("conflict group contains multiple owners")
            if owners and not has_unresolved:
                self._stats["conflict_owner_group_evaluable"] += 1
                self._stats["conflict_owner_group_correct"] += int(
                    owners[0] in positives
                )
            for index in indices:
                if not owners or normalized[index] is None:
                    continue
                if index == owners[0]:
                    continue
                self._stats["conflict_loser_rows"] += 1
                self._stats["conflict_loser_native_positive"] += int(
                    index in positives
                )

        for row, targets in zip(batch.rows, normalized):
            if targets is None:
                self._stats["native_unresolved"] += 1
                continue
            if not targets:
                native_kind = "birth"
                self._stats["native_births"] += 1
            elif len(targets) == 1:
                native_kind = "unique_history"
                self._stats["native_history_unique"] += 1
            else:
                native_kind = "ambiguous_history"
                self._stats["native_history_ambiguous"] += 1

            if row.action == "native_fallback":
                self._stats[f"fallback_on_{native_kind}"] += 1
                continue

            self._stats["selected_evaluable"] += 1
            if row.action == "track":
                self._stats["selected_track_evaluable"] += 1
                correct = bool(targets) and row.selected_track_id in targets
                if correct:
                    self._stats["selected_track_correct"] += 1
                elif not targets:
                    self._stats["false_track_overrides"] += 1
            elif row.action == "birth":
                self._stats["selected_birth_evaluable"] += 1
                correct = not targets
                if correct:
                    self._stats["selected_birth_correct"] += 1
                else:
                    self._stats["false_birth_overrides"] += 1
            else:
                raise RuntimeError(f"unknown arbitration action: {row.action}")
            if correct:
                self._stats["selected_correct"] += 1
            else:
                self._stats["selected_wrong"] += 1
                self._add_example(row, native_kind, sorted(targets))

            if row.conflict_winner:
                self._stats["conflict_winner_evaluable"] += 1
                self._stats["conflict_winner_correct"] += int(correct)

        self._last_observed_batch = batch
        self._pending_batch = None

    def record_pipeline_timing(
        self,
        *,
        query_ms: Optional[float] = None,
        observe_ms: Optional[float] = None,
    ) -> None:
        if query_ms is None and observe_ms is None:
            raise ValueError("at least one pipeline timing value is required")
        for stage, value in (("query", query_ms), ("observe", observe_ms)):
            if value is None:
                continue
            timing = _finite_float(f"pipeline_{stage}_ms", value, 0.0)
            self._stats[f"pipeline_{stage}_calls"] += 1
            self._stats[f"pipeline_{stage}_ms_total"] += timing
            self._stats[f"pipeline_{stage}_ms_max"] = max(
                float(self._stats[f"pipeline_{stage}_ms_max"]), timing
            )

    @staticmethod
    def _rate(numerator: int, denominator: int) -> Optional[float]:
        return numerator / denominator if denominator else None

    def summary(self) -> Dict[str, object]:
        result = dict(self._stats)
        proposals = int(result["proposals"])
        selected = int(result["selected_evaluable"])
        selected_track = int(result["selected_track_evaluable"])
        selected_birth = int(result["selected_birth_evaluable"])
        conflict_evaluable = int(result["conflict_winner_evaluable"])
        conflict_group_evaluable = int(
            result["conflict_owner_group_evaluable"]
        )
        queries = int(result["queries"])
        pipeline_queries = int(result["pipeline_query_calls"])
        pipeline_observes = int(result["pipeline_observe_calls"])
        result.update(
            {
                "schema": "boxfusion.puf_arbitration_lite_shadow.v1",
                "enabled": self.enabled,
                "observer_only": self.observer_only,
                "active_authorized": False,
                "training_free": True,
                "causal": True,
                "online_update": False,
                "semantic_access": False,
                "semantic_mutation": False,
                "ground_truth_access": False,
                "detector_score_access": False,
                "reassigns_losers": False,
                "suppresses_proposals": False,
                "scene_id": self._scene_id,
                "effective_config": dict(self.config),
                "selected_override_rate": self._rate(
                    int(result["selected_track_overrides"])
                    + int(result["selected_birth_overrides"]),
                    proposals,
                ),
                "native_fallback_rate": self._rate(
                    int(result["native_fallback_rows"]), proposals
                ),
                "selective_precision": self._rate(
                    int(result["selected_correct"]), selected
                ),
                "selected_track_precision": self._rate(
                    int(result["selected_track_correct"]), selected_track
                ),
                "selected_birth_precision": self._rate(
                    int(result["selected_birth_correct"]), selected_birth
                ),
                "conflict_winner_precision": self._rate(
                    int(result["conflict_winner_correct"]), conflict_evaluable
                ),
                "conflict_owner_group_precision": self._rate(
                    int(result["conflict_owner_group_correct"]),
                    conflict_group_evaluable,
                ),
                "conflict_loser_native_positive_rate": self._rate(
                    int(result["conflict_loser_native_positive"]),
                    int(result["conflict_loser_rows"]),
                ),
                "conflict_group_resolution_rate": self._rate(
                    int(result["conflict_groups_with_winner"]),
                    int(result["source_conflict_groups"]),
                ),
                "query_ms_mean": (
                    float(result["query_ms_total"]) / queries if queries else 0.0
                ),
                "query_ms_p95": (
                    float(np.percentile(np.asarray(self._query_samples), 95))
                    if self._query_samples
                    else 0.0
                ),
                "pipeline_query_ms_mean": (
                    float(result["pipeline_query_ms_total"]) / pipeline_queries
                    if pipeline_queries
                    else 0.0
                ),
                "pipeline_observe_ms_mean": (
                    float(result["pipeline_observe_ms_total"]) / pipeline_observes
                    if pipeline_observes
                    else 0.0
                ),
                "diagnostic_examples": tuple(self._examples),
            }
        )
        return result

    def summary_line(self) -> str:
        summary = self.summary()

        def rate(value: object) -> str:
            return "nan" if value is None else f"{float(value):.4f}"

        return (
            "PUF-arbitration-lite shadow summary | "
            f"queries/proposals={summary['queries']}/{summary['proposals']}, "
            f"source_conflict_groups/rows="
            f"{summary['source_conflict_groups']}/"
            f"{summary['source_conflict_rows']}, "
            f"selected_track/birth/fallback="
            f"{summary['selected_track_overrides']}/"
            f"{summary['selected_birth_overrides']}/"
            f"{summary['native_fallback_rows']}, "
            f"duplicate_selected={summary['duplicate_selected_tracks']}, "
            f"selective_precision={rate(summary['selective_precision'])}, "
            f"conflict_winner_precision="
            f"{rate(summary['conflict_winner_precision'])}, "
            f"query_mean/p95/max_ms={summary['query_ms_mean']:.3f}/"
            f"{summary['query_ms_p95']:.3f}/"
            f"{summary['query_ms_max']:.3f}"
        )


def build_puf_arbitration_lite(
    config: Mapping[str, object],
) -> PUFArbitrationLiteObserver:
    if not isinstance(config, Mapping):
        raise ValueError("application config must be a mapping")
    return PUFArbitrationLiteObserver(
        config.get("puf_arbitration_lite", {})
    )


__all__ = [
    "ArbitrationDecision",
    "DEFAULT_PUF_ARBITRATION_LITE_CONFIG",
    "PUFArbitrationBatch",
    "PUFArbitrationLiteObserver",
    "build_puf_arbitration_lite",
    "resolve_puf_arbitration_lite_config",
]
