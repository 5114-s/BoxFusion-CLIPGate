"""Output-inert PUF probability observer over SMOV-clean Gclean evidence.

``PufGcleanShadow`` delegates fragment extraction, causal begin-frame-past
memory, native reservation, and native-unmatched filtering to
``GcleanShadow``.  The exact sealed proposal batch and historical snapshot
used by Group3D-lite additionally expose every positive-overlap Top-8 pair.
Those pairs are jointly normalized per proposal by :mod:`boxfusion.puf_lite`.

The fixed PUF mass 0.4 is a null/unmatched state in this training-free arm.
``beta_null <= 0.5`` emits a *shadow directive* to the stable-ID argmax track;
otherwise the proposal abstains.  Birth is permanently disabled.  Directives
are diagnostics only: this module never mutates native association, rows,
geometry, score, class, CLIP state, fusion history, or random-number state.
The separate ``associations`` output is a deterministic active-safe subset:
it requires a strictly positive track-vs-alternative margin and removes every
directive in a same-frame, same-past-track conflict group.  It too remains
output-inert here.

Distance units are explicit.  Pair evidence stores centroid distance in 5 cm
voxel units and every serialized candidate also contains the conversion to
metres (``distance_m = distance_voxels * 0.05``).
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
import json
import os
import tempfile
import time
from types import MappingProxyType
from typing import Mapping, Optional, Sequence

import numpy as np

from boxfusion.gclean_shadow import (
    GcleanFrameToken,
    GcleanShadow,
    GcleanShadowResult,
)
from boxfusion.group3d_lite import PairEvidenceResult, VoxelPairEvidence
from boxfusion.observer_track_registry import IdentityResolution
from boxfusion.puf_lite import PUFCandidateInput, PUFLiteResult, compute_puf_lite
from boxfusion.smov_fragments import PreparedKeyframe


SCHEMA = "boxfusion.puf_gclean_shadow.v1"
MODE = "shadow"
FRAGMENT_SOURCE = "smov_clean"
CANDIDATE_SOURCE = "gclean_positive_overlap_top8"

_F_VOXEL_SIZE_METERS = 0.05
_F_MAX_PROPOSALS = 64
_F_MAX_CANDIDATES_PER_PROPOSAL = 8
_F_MAX_TOTAL_CANDIDATES = 512
_F_TIMING_WINDOW = 4096
_F_MAX_DIAGNOSTIC_FRAMES = 16_384
_F_DIAGNOSTIC_BYTES = 32 * 1024 * 1024
_F_JSON_DEPTH = 16
_F_JSON_CONTAINER_ITEMS = 65_536


@dataclass(frozen=True)
class PufGcleanFrameToken:
    """Opaque wrapper token; only its creating observer may consume it."""

    serial: int
    frame_id: int
    _inner: GcleanFrameToken


@dataclass(frozen=True)
class PufGcleanCandidateProbability:
    proposal_id: int
    past_track_id: int
    intersection: int
    proposal_voxel_count: int
    past_track_voxel_count: int
    proposal_containment: float
    past_track_containment: float
    spatial_likelihood: float
    beta: float
    centroid_distance_voxels: float
    centroid_distance_m: float


@dataclass(frozen=True)
class PufGcleanProposalProbability:
    proposal_id: int
    native_track_id: int
    beta_null: float
    candidates: tuple[PufGcleanCandidateProbability, ...]
    argmax_past_track_id: Optional[int]
    argmax_beta: float
    paper_gate: bool
    margin: Optional[float]
    normalized_entropy: float
    normalization_error: float
    gclean_accepted_past_track_id: Optional[int]
    agrees_with_gclean: Optional[bool]
    birth_enabled: bool = False


@dataclass(frozen=True)
class PufGcleanDirective:
    """Counterfactual paper decision; never an executed native mutation."""

    proposal_id: int
    native_track_id: int
    past_track_id: int
    beta_track: float
    beta_null: float
    margin: float
    gclean_accepted_past_track_id: Optional[int]
    agrees_with_gclean: Optional[bool]
    birth_enabled: bool = False


@dataclass(frozen=True)
class PufGcleanAssociation:
    """Deterministic active-safe subset of the paper shadow directives."""

    proposal_id: int
    native_track_id: int
    past_track_id: int
    beta_track: float
    beta_null: float
    margin: float
    gclean_accepted_past_track_id: Optional[int]
    agrees_with_gclean: Optional[bool]
    birth_enabled: bool = False


@dataclass(frozen=True)
class PufGcleanShadowResult:
    frame_id: int
    rows: tuple[PufGcleanProposalProbability, ...]
    directives: tuple[PufGcleanDirective, ...]
    candidate_proposal_ids: tuple[int, ...]
    candidate_native_track_ids: tuple[int, ...]
    associations: tuple[PufGcleanAssociation, ...]
    gclean_associations: tuple[object, ...]
    evidence_diagnostics: Mapping[str, object]
    puf_diagnostics: Mapping[str, object]
    fail_open: bool
    fail_open_code: str
    same_track_conflict_groups: int
    same_track_conflict_directives: int
    evidence_elapsed_ms: float
    probability_elapsed_ms: float
    # Full incremental PUF cost: pair evidence plus probability/action audit.
    puf_elapsed_ms: float
    gclean_total_observer_elapsed_ms: float
    total_observer_elapsed_ms: float
    gclean_result: GcleanShadowResult
    mode: str = MODE
    fragment_source: str = FRAGMENT_SOURCE
    candidate_source: str = CANDIDATE_SOURCE
    birth_enabled: bool = False


def _timing(values: deque[float]) -> Mapping[str, object]:
    samples = tuple(float(value) for value in values)
    array = np.asarray(samples, dtype=np.float64)
    return MappingProxyType(
        {
            "count": len(samples),
            "samples_ms": samples,
            "mean_ms": float(np.mean(array)) if len(array) else 0.0,
            "p50_ms": float(np.percentile(array, 50)) if len(array) else 0.0,
            "p95_ms": float(np.percentile(array, 95)) if len(array) else 0.0,
            "max_ms": float(np.max(array)) if len(array) else 0.0,
        }
    )


def _evidence_diagnostics(evidence: Optional[PairEvidenceResult]) -> Mapping[str, object]:
    if evidence is None:
        return MappingProxyType(
            {"fail_open": True, "code": "missing_pair_evidence"}
        )
    return MappingProxyType(asdict(evidence.diagnostics))


def _candidate_inputs(
    evidence: PairEvidenceResult,
) -> tuple[tuple[PUFCandidateInput, ...], tuple[int, ...]]:
    proposal_ids = tuple(int(row.proposal_id) for row in evidence.proposals)
    flattened: list[PUFCandidateInput] = []
    for row in evidence.proposals:
        for item in row.candidates:
            flattened.append(
                PUFCandidateInput(
                    proposal_id=int(item.proposal_id),
                    track_id=int(item.track_id),
                    intersection=int(item.intersection),
                    proposal_voxel_count=int(item.proposal_voxel_count),
                    track_voxel_count=int(item.track_voxel_count),
                    # puf_lite treats this as copied audit metadata only.  The
                    # wrapper names and converts its voxel unit explicitly.
                    centroid_distance=float(item.centroid_distance_voxels),
                )
            )
    return tuple(flattened), proposal_ids


def _fail_open_puf_diagnostics(code: str) -> Mapping[str, object]:
    return MappingProxyType(
        {
            "fail_open": True,
            "code": str(code),
            "input_candidate_count": 0,
            "proposal_count": 0,
            "lambda_null": 0.4,
            "max_proposals": _F_MAX_PROPOSALS,
            "max_candidates_per_proposal": _F_MAX_CANDIDATES_PER_PROPOSAL,
            "max_total_candidates": _F_MAX_TOTAL_CANDIDATES,
            "max_normalization_error": 0.0,
            "semantic_factor": "identity",
            "birth_enabled": False,
        }
    )


def _join_probabilities(
    *,
    evidence: PairEvidenceResult,
    puf: PUFLiteResult,
    gclean: GcleanShadowResult,
) -> tuple[
    tuple[PufGcleanProposalProbability, ...],
    tuple[PufGcleanDirective, ...],
]:
    evidence_by_proposal = {
        int(row.proposal_id): {
            int(item.track_id): item for item in row.candidates
        }
        for row in evidence.proposals
    }
    if len(evidence_by_proposal) != len(evidence.proposals):
        raise ValueError("duplicate_evidence_proposal_id")
    native_by_proposal = dict(
        zip(gclean.candidate_proposal_ids, gclean.candidate_native_track_ids)
    )
    if len(native_by_proposal) != len(gclean.candidate_proposal_ids):
        raise ValueError("duplicate_candidate_proposal_id")
    accepted_by_proposal = {
        int(item.proposal_id): int(item.past_track_id)
        for item in gclean.associations
    }
    if len(accepted_by_proposal) != len(gclean.associations):
        raise ValueError("duplicate_gclean_association")

    rows: list[PufGcleanProposalProbability] = []
    directives: list[PufGcleanDirective] = []
    for probability in puf.proposals:
        proposal_id = int(probability.proposal_id)
        if proposal_id not in evidence_by_proposal:
            raise ValueError("puf_proposal_outside_evidence")
        native_track_id = native_by_proposal.get(proposal_id)
        if native_track_id is None:
            raise ValueError("puf_proposal_missing_native_track")
        evidence_candidates = evidence_by_proposal[proposal_id]
        joined: list[PufGcleanCandidateProbability] = []
        for candidate in probability.candidates:
            pair: Optional[VoxelPairEvidence] = evidence_candidates.get(
                int(candidate.track_id)
            )
            if pair is None:
                raise ValueError("puf_candidate_outside_evidence")
            distance_voxels = float(pair.centroid_distance_voxels)
            joined.append(
                PufGcleanCandidateProbability(
                    proposal_id=proposal_id,
                    past_track_id=int(candidate.track_id),
                    intersection=int(pair.intersection),
                    proposal_voxel_count=int(pair.proposal_voxel_count),
                    past_track_voxel_count=int(pair.track_voxel_count),
                    proposal_containment=float(pair.proposal_containment),
                    past_track_containment=float(pair.track_containment),
                    spatial_likelihood=float(candidate.spatial_likelihood),
                    beta=float(candidate.beta),
                    centroid_distance_voxels=distance_voxels,
                    centroid_distance_m=distance_voxels * _F_VOXEL_SIZE_METERS,
                )
            )
        if len(joined) != len(evidence_candidates):
            raise ValueError("puf_candidate_set_mismatch")

        accepted = accepted_by_proposal.get(proposal_id)
        agrees = (
            None
            if accepted is None or probability.argmax_track_id is None
            else int(probability.argmax_track_id) == accepted
        )
        row = PufGcleanProposalProbability(
            proposal_id=proposal_id,
            native_track_id=int(native_track_id),
            beta_null=float(probability.beta_null),
            candidates=tuple(joined),
            argmax_past_track_id=(
                None
                if probability.argmax_track_id is None
                else int(probability.argmax_track_id)
            ),
            argmax_beta=float(probability.argmax_beta),
            paper_gate=bool(probability.paper_gate),
            margin=(
                None if probability.margin is None else float(probability.margin)
            ),
            normalized_entropy=float(probability.normalized_entropy),
            normalization_error=float(probability.normalization_error),
            gclean_accepted_past_track_id=accepted,
            agrees_with_gclean=agrees,
        )
        rows.append(row)
        if row.paper_gate:
            if row.argmax_past_track_id is None or row.margin is None:
                raise ValueError("paper_gate_missing_argmax")
            directives.append(
                PufGcleanDirective(
                    proposal_id=proposal_id,
                    native_track_id=row.native_track_id,
                    past_track_id=row.argmax_past_track_id,
                    beta_track=row.argmax_beta,
                    beta_null=row.beta_null,
                    margin=row.margin,
                    gclean_accepted_past_track_id=accepted,
                    agrees_with_gclean=agrees,
                )
            )
    return tuple(rows), tuple(directives)


def _conflict_counts(
    directives: Sequence[PufGcleanDirective],
) -> tuple[int, int]:
    counts: Counter[int] = Counter(item.past_track_id for item in directives)
    groups = sum(int(value > 1) for value in counts.values())
    rows = sum(value for value in counts.values() if value > 1)
    return groups, rows


def _active_safe_associations(
    directives: Sequence[PufGcleanDirective],
) -> tuple[PufGcleanAssociation, ...]:
    """Remove whole same-track conflict groups, then require positive margin.

    Conflict membership is computed across *all* paper directives before the
    margin gate.  Thus one ambiguous directive cannot be hidden to allow a
    competing proposal to claim the same historical track.
    """

    counts: Counter[int] = Counter(item.past_track_id for item in directives)
    output = [
        PufGcleanAssociation(
            proposal_id=item.proposal_id,
            native_track_id=item.native_track_id,
            past_track_id=item.past_track_id,
            beta_track=item.beta_track,
            beta_null=item.beta_null,
            margin=item.margin,
            gclean_accepted_past_track_id=(
                item.gclean_accepted_past_track_id
            ),
            agrees_with_gclean=item.agrees_with_gclean,
        )
        for item in directives
        if item.margin > 0.0 and counts[item.past_track_id] == 1
    ]
    output.sort(key=lambda item: (item.proposal_id, item.past_track_id))
    return tuple(output)


class PufGcleanShadow:
    """Causal PUF-lite observer wrapping one output-inert Gclean sidecar."""

    def __init__(self) -> None:
        self._inner = GcleanShadow()
        self._pending: Optional[PufGcleanFrameToken] = None
        self._serial = 0
        self._stats: Counter[str] = Counter()
        self._failure_reasons: Counter[str] = Counter()
        self._evidence_timings: deque[float] = deque(maxlen=_F_TIMING_WINDOW)
        self._probability_timings: deque[float] = deque(maxlen=_F_TIMING_WINDOW)
        self._puf_timings: deque[float] = deque(maxlen=_F_TIMING_WINDOW)
        self._total_timings: deque[float] = deque(maxlen=_F_TIMING_WINDOW)

    @property
    def pending(self) -> bool:
        return self._pending is not None

    @property
    def memory_track_ids(self) -> tuple[int, ...]:
        if self._pending is not None:
            raise RuntimeError("memory is not externally stable during a keyframe")
        return self._inner.memory_track_ids

    def begin_keyframe(
        self,
        frame_id: int,
        *,
        active_track_ids: Optional[Sequence[int]] = None,
    ) -> PufGcleanFrameToken:
        if self._pending is not None:
            raise RuntimeError("a PUF-Gclean keyframe is already pending")
        inner = self._inner.begin_keyframe(
            frame_id, active_track_ids=active_track_ids
        )
        self._serial += 1
        token = PufGcleanFrameToken(self._serial, int(frame_id), inner)
        self._pending = token
        return token

    def abort_keyframe(self, token: PufGcleanFrameToken) -> None:
        if token is not self._pending:
            raise RuntimeError("abort must use the exact pending PUF-Gclean token")
        self._inner.abort_keyframe(token._inner)
        self._pending = None
        self._stats["aborted_keyframes"] += 1

    def finish_keyframe(
        self,
        token: PufGcleanFrameToken,
        *,
        batch: PreparedKeyframe,
        resolution: IdentityResolution,
        reserved_past_track_ids: Optional[Sequence[int]] = None,
        unmatched_retained_proposal_ids: Optional[Sequence[int]] = None,
    ) -> PufGcleanShadowResult:
        if token is not self._pending:
            raise RuntimeError("finish must use the exact pending PUF-Gclean token")
        try:
            gclean = self._inner.finish_keyframe(
                token._inner,
                batch=batch,
                resolution=resolution,
                reserved_past_track_ids=reserved_past_track_ids,
                unmatched_retained_proposal_ids=unmatched_retained_proposal_ids,
                collect_pair_evidence=True,
            )
        except Exception:
            if not self._inner.pending:
                self._pending = None
            raise
        self._pending = None

        probability_started = time.perf_counter_ns()
        evidence = gclean.pair_evidence
        evidence_diag = _evidence_diagnostics(evidence)
        evidence_elapsed_ms = float(evidence_diag.get("elapsed_ms", 0.0))
        invalid_evidence_timing = (
            not np.isfinite(evidence_elapsed_ms) or evidence_elapsed_ms < 0.0
        )
        if invalid_evidence_timing:
            evidence_elapsed_ms = 0.0
        rows: tuple[PufGcleanProposalProbability, ...] = ()
        directives: tuple[PufGcleanDirective, ...] = ()
        associations: tuple[PufGcleanAssociation, ...] = ()
        evidence_fail_open = bool(evidence_diag.get("fail_open", True))
        evidence_code = str(
            evidence_diag.get("code", "missing_pair_evidence")
        )
        matcher_fail_open = bool(
            gclean.matcher_diagnostics.get("fail_open", True)
        )
        matcher_code = str(gclean.matcher_diagnostics.get("code", "unknown"))
        fail_open = False
        code = "ok"
        if matcher_fail_open:
            fail_open = True
            code = "gclean_matcher:" + matcher_code
            puf_diag = _fail_open_puf_diagnostics(code)
        elif evidence is None or evidence_fail_open:
            fail_open = True
            code = "evidence:" + evidence_code
            puf_diag = _fail_open_puf_diagnostics(code)
        elif invalid_evidence_timing:
            fail_open = True
            code = "evidence:invalid_elapsed_ms"
            puf_diag = _fail_open_puf_diagnostics(code)
        else:
            try:
                candidate_inputs, proposal_ids = _candidate_inputs(evidence)
                puf = compute_puf_lite(
                    candidate_inputs, proposal_ids=proposal_ids
                )
                puf_diag = MappingProxyType(asdict(puf.diagnostics))
                fail_open = bool(puf.diagnostics.fail_open)
                code = str(puf.diagnostics.code)
                if not fail_open:
                    rows, directives = _join_probabilities(
                        evidence=evidence, puf=puf, gclean=gclean
                    )
                    associations = _active_safe_associations(directives)
            except Exception as error:
                fail_open = True
                code = "wrapper_exception:" + type(error).__name__
                puf_diag = _fail_open_puf_diagnostics(code)
                rows, directives, associations = (), (), ()

        probability_elapsed_ms = (
            time.perf_counter_ns() - probability_started
        ) / 1e6
        puf_elapsed_ms = evidence_elapsed_ms + probability_elapsed_ms
        conflict_groups, conflict_rows = _conflict_counts(directives)
        self._stats["keyframes"] += 1
        self._stats["proposals"] += len(rows)
        self._stats["positive_candidate_pairs"] += sum(
            len(row.candidates) for row in rows
        )
        self._stats["paper_directives"] += len(directives)
        self._stats["null_decisions"] += sum(
            int(not row.paper_gate) for row in rows
        )
        self._stats["null_only_proposals"] += sum(
            int(not row.candidates) for row in rows
        )
        self._stats["gclean_accepted"] += len(gclean.associations)
        self._stats["directive_agrees_with_gclean"] += sum(
            int(item.agrees_with_gclean is True) for item in directives
        )
        self._stats["puf_only_directives"] += sum(
            int(item.gclean_accepted_past_track_id is None)
            for item in directives
        )
        self._stats["same_track_conflict_groups"] += conflict_groups
        self._stats["same_track_conflict_directives"] += conflict_rows
        self._stats["active_safe_associations"] += len(associations)
        if fail_open:
            self._stats["fail_open_keyframes"] += 1
            self._failure_reasons[code] += 1

        # Gclean's declared total already contains upstream SMOV batch time,
        # voxel adaptation, matcher, pair evidence, and commit.  Only the
        # post-Gclean probability/action pass is additive here; adding evidence
        # again would double-count it, while a local wall timer would omit the
        # SMOV work completed before this method received ``batch``.
        total_ms = (
            float(gclean.total_observer_elapsed_ms)
            + probability_elapsed_ms
        )
        self._evidence_timings.append(evidence_elapsed_ms)
        self._probability_timings.append(probability_elapsed_ms)
        self._puf_timings.append(puf_elapsed_ms)
        self._total_timings.append(total_ms)

        return PufGcleanShadowResult(
            frame_id=gclean.frame_id,
            rows=rows,
            directives=directives,
            candidate_proposal_ids=tuple(gclean.candidate_proposal_ids),
            candidate_native_track_ids=tuple(
                gclean.candidate_native_track_ids
            ),
            associations=associations,
            gclean_associations=tuple(gclean.associations),
            evidence_diagnostics=evidence_diag,
            puf_diagnostics=puf_diag,
            fail_open=fail_open,
            fail_open_code=code,
            same_track_conflict_groups=conflict_groups,
            same_track_conflict_directives=conflict_rows,
            evidence_elapsed_ms=evidence_elapsed_ms,
            probability_elapsed_ms=probability_elapsed_ms,
            puf_elapsed_ms=puf_elapsed_ms,
            gclean_total_observer_elapsed_ms=float(
                gclean.total_observer_elapsed_ms
            ),
            total_observer_elapsed_ms=total_ms,
            gclean_result=gclean,
        )

    def diagnostics(self) -> Mapping[str, object]:
        inner = self._inner.diagnostics()
        return MappingProxyType(
            {
                "schema": SCHEMA,
                "mode": MODE,
                "fragment_source": FRAGMENT_SOURCE,
                "candidate_source": CANDIDATE_SOURCE,
                "birth_enabled": False,
                "fail_open": self._stats["fail_open_keyframes"] > 0,
                "lambda_null": 0.4,
                "paper_gate": "beta_null<=0.5_then_stable_track_argmax",
                "active_safe_gate": (
                    "paper_directive_and_margin>0_and_unique_past_track"
                ),
                "pending": self._pending is not None,
                "last_frame": inner["last_frame"],
                "memory_track_ids": tuple(inner["memory_track_ids"]),
                "caps": MappingProxyType(
                    {
                        "max_proposals": _F_MAX_PROPOSALS,
                        "max_candidates_per_proposal": (
                            _F_MAX_CANDIDATES_PER_PROPOSAL
                        ),
                        "max_total_candidates": _F_MAX_TOTAL_CANDIDATES,
                    }
                ),
                "stats": MappingProxyType(dict(self._stats)),
                "failure_reasons": MappingProxyType(
                    dict(sorted(self._failure_reasons.items()))
                ),
                "timing": MappingProxyType(
                    {
                        "evidence": _timing(self._evidence_timings),
                        "probability": _timing(self._probability_timings),
                        "puf_incremental": _timing(self._puf_timings),
                        "total_observer": _timing(self._total_timings),
                    }
                ),
                "gclean": inner,
            }
        )


def puf_gclean_result_to_dict(
    result: PufGcleanShadowResult,
) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "mode": MODE,
        "fragment_source": FRAGMENT_SOURCE,
        "candidate_source": CANDIDATE_SOURCE,
        "birth_enabled": False,
        "frame_id": result.frame_id,
        "candidate_proposal_ids": list(result.candidate_proposal_ids),
        "candidate_native_track_ids": list(
            result.candidate_native_track_ids
        ),
        "rows": [asdict(item) for item in result.rows],
        "directives": [asdict(item) for item in result.directives],
        "associations": [asdict(item) for item in result.associations],
        "gclean_accepted_associations": [
            asdict(item) for item in result.gclean_associations
        ],
        "evidence_diagnostics": dict(result.evidence_diagnostics),
        "puf_diagnostics": dict(result.puf_diagnostics),
        "fail_open": result.fail_open,
        "fail_open_code": result.fail_open_code,
        "same_track_conflict_groups": result.same_track_conflict_groups,
        "same_track_conflict_directives": (
            result.same_track_conflict_directives
        ),
        "evidence_elapsed_ms": result.evidence_elapsed_ms,
        "probability_elapsed_ms": result.probability_elapsed_ms,
        "puf_elapsed_ms": result.puf_elapsed_ms,
        "gclean_total_observer_elapsed_ms": (
            result.gclean_total_observer_elapsed_ms
        ),
        "total_observer_elapsed_ms": result.total_observer_elapsed_ms,
    }


def _plain_json_value(value: object, *, _depth: int = 0) -> object:
    if _depth > _F_JSON_DEPTH:
        raise ValueError("PUF-Gclean diagnostic nesting exceeds the depth cap")
    if isinstance(value, Mapping):
        if len(value) > _F_JSON_CONTAINER_ITEMS:
            raise ValueError("PUF-Gclean mapping exceeds the item cap")
        return {
            str(key): _plain_json_value(item, _depth=_depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        if len(value) > _F_JSON_CONTAINER_ITEMS:
            raise ValueError("PUF-Gclean sequence exceeds the item cap")
        return [
            _plain_json_value(item, _depth=_depth + 1) for item in value
        ]
    if isinstance(value, np.generic):
        return _plain_json_value(value.item(), _depth=_depth + 1)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(
        "PUF-Gclean diagnostics contain a non-JSON value: "
        + type(value).__name__
    )


def write_puf_gclean_shadow_diagnostics(
    path: os.PathLike[str] | str,
    *,
    scene_id: str,
    results: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    trace_valid: bool,
) -> str:
    """Atomically write one bounded PUF-Gclean scene diagnostic."""

    if isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
        raise ValueError("PUF-Gclean results must be a bounded sequence")
    if len(results) > _F_MAX_DIAGNOSTIC_FRAMES:
        raise ValueError("PUF-Gclean frame count exceeds the hard cap")
    if not isinstance(summary, Mapping):
        raise ValueError("PUF-Gclean summary must be a mapping")
    frame_fail_open = any(
        not isinstance(item, Mapping) or bool(item.get("fail_open", True))
        for item in results
    )
    summary_fail_open = bool(summary.get("fail_open", True))
    destination = os.path.abspath(os.fspath(path))
    payload = {
        "schema": SCHEMA,
        "mode": MODE,
        "fragment_source": FRAGMENT_SOURCE,
        "candidate_source": CANDIDATE_SOURCE,
        "birth_enabled": False,
        "fail_open": summary_fail_open or frame_fail_open,
        "scene_id": str(scene_id),
        "trace_valid": bool(trace_valid),
        "frame_count": len(results),
        "frames": _plain_json_value(results),
        "summary": _plain_json_value(summary),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(encoded) > _F_DIAGNOSTIC_BYTES:
        raise ValueError("PUF-Gclean diagnostic exceeds the 32 MiB cap")
    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + os.path.basename(destination) + ".",
        suffix=".tmp",
        dir=parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


__all__ = [
    "CANDIDATE_SOURCE",
    "FRAGMENT_SOURCE",
    "MODE",
    "PufGcleanAssociation",
    "PufGcleanCandidateProbability",
    "PufGcleanDirective",
    "PufGcleanFrameToken",
    "PufGcleanProposalProbability",
    "PufGcleanShadow",
    "PufGcleanShadowResult",
    "SCHEMA",
    "puf_gclean_result_to_dict",
    "write_puf_gclean_shadow_diagnostics",
]
