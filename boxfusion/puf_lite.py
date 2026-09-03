"""Training-free probabilistic scoring for secondary voxel associations.

This module implements the geometry-only part of PUF's per-observation
normalization as a small, output-inert computation.  Each proposal must be
supplied with *all* of its candidate tracks.  The spatial likelihood is the
asymmetric proposal containment used by PUF's voxel backend::

    L(proposal, track) = intersection / proposal_voxel_count

The fixed mass ``0.4`` is retained as a null/unmatched hypothesis.  It is not
a birth action: this module never creates a node, mutates a track, or changes
semantic evidence.  ``paper_gate`` merely reports the counterfactual PUF rule
``beta_null <= 0.5``; callers must not interpret it as an executed merge.

The public entry point is batch fail-open.  Any malformed or over-limit input
returns an empty result with ``diagnostics.fail_open`` set, so invalid evidence
cannot accidentally authorize an association.  Valid results contain only
frozen dataclasses, tuples, and scalar values and therefore do not alias the
caller-owned records.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Optional, Sequence

import numpy as np


SCHEMA = "boxfusion.puf_lite.v1"

# Executable limits are deliberately private and fixed.  In particular, the
# null mass is a literature-default preregistration, not a caller parameter or
# a value that may be tuned against evaluation ground truth.
_F_MAX_PROPOSALS = 64
_F_MAX_CANDIDATES_PER_PROPOSAL = 8
_F_MAX_TOTAL_CANDIDATES = 512
_F_MAX_PROPOSAL_VOXELS = 512
_F_MAX_TRACK_VOXELS = 1024
_F_MAX_ID = (1 << 63) - 1
_F_MAX_NORMALIZATION_ERROR = 1e-12


@dataclass(frozen=True)
class PUFCandidateInput:
    """Convenience input record; mappings and equivalent objects also work."""

    proposal_id: int
    track_id: int
    intersection: int
    proposal_voxel_count: int
    track_voxel_count: Optional[int] = None
    centroid_distance: Optional[float] = None


@dataclass(frozen=True)
class PUFCandidateProbability:
    """One candidate likelihood and its jointly normalized probability."""

    proposal_id: int
    track_id: int
    intersection: int
    proposal_voxel_count: int
    spatial_likelihood: float
    beta: float
    track_voxel_count: Optional[int] = None
    centroid_distance: Optional[float] = None


@dataclass(frozen=True)
class PUFProposalProbability:
    """Read-only probabilistic audit for one proposal.

    ``margin`` is the best track probability minus its strongest alternative,
    where the alternatives include both the null state and the runner-up
    track.  It may therefore be negative even though an argmax track exists.
    """

    proposal_id: int
    beta_null: float
    candidates: tuple[PUFCandidateProbability, ...]
    argmax_track_id: Optional[int]
    argmax_beta: float
    paper_gate: bool
    margin: Optional[float]
    normalized_entropy: float
    normalization_error: float
    birth_enabled: bool = False


@dataclass(frozen=True)
class PUFLiteDiagnostics:
    """Stable status and bounds metadata for a batch invocation."""

    fail_open: bool
    code: str
    input_candidate_count: int
    proposal_count: int
    lambda_null: float
    max_proposals: int
    max_candidates_per_proposal: int
    max_total_candidates: int
    max_normalization_error: float
    semantic_factor: str = "identity"
    birth_enabled: bool = False


@dataclass(frozen=True)
class PUFLiteResult:
    """Immutable batch result sorted by proposal ID."""

    proposals: tuple[PUFProposalProbability, ...]
    diagnostics: PUFLiteDiagnostics
    schema: str = SCHEMA


@dataclass(frozen=True)
class _ValidatedCandidate:
    proposal_id: int
    track_id: int
    intersection: int
    proposal_voxel_count: int
    track_voxel_count: Optional[int]
    centroid_distance: Optional[float]


class _InvalidInput(ValueError):
    pass


_MISSING = object()


def _value(item: object, name: str, *, optional: bool = False) -> object:
    if isinstance(item, Mapping):
        value = item.get(name, _MISSING)
    else:
        value = getattr(item, name, _MISSING)
    if value is _MISSING:
        if optional:
            return None
        raise _InvalidInput("missing_%s" % name)
    return value


def _integer(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise _InvalidInput("invalid_%s" % name)
    result = int(value)
    if result < minimum or result > maximum:
        raise _InvalidInput("invalid_%s" % name)
    return result


def _optional_distance(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise _InvalidInput("invalid_centroid_distance")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise _InvalidInput("invalid_centroid_distance")
    return result


def _validate_candidate(item: object) -> _ValidatedCandidate:
    proposal_id = _integer(
        _value(item, "proposal_id"),
        "proposal_id",
        minimum=0,
        maximum=_F_MAX_ID,
    )
    track_id = _integer(
        _value(item, "track_id"),
        "track_id",
        minimum=0,
        maximum=_F_MAX_ID,
    )
    proposal_voxel_count = _integer(
        _value(item, "proposal_voxel_count"),
        "proposal_voxel_count",
        minimum=1,
        maximum=_F_MAX_PROPOSAL_VOXELS,
    )
    intersection = _integer(
        _value(item, "intersection"),
        "intersection",
        minimum=1,
        maximum=proposal_voxel_count,
    )
    raw_track_count = _value(item, "track_voxel_count", optional=True)
    track_voxel_count = None
    if raw_track_count is not None:
        track_voxel_count = _integer(
            raw_track_count,
            "track_voxel_count",
            minimum=1,
            maximum=_F_MAX_TRACK_VOXELS,
        )
        if intersection > track_voxel_count:
            raise _InvalidInput("intersection_exceeds_track")
    centroid_distance = _optional_distance(
        _value(item, "centroid_distance", optional=True)
    )
    return _ValidatedCandidate(
        proposal_id=proposal_id,
        track_id=track_id,
        intersection=intersection,
        proposal_voxel_count=proposal_voxel_count,
        track_voxel_count=track_voxel_count,
        centroid_distance=centroid_distance,
    )


def _diagnostics(
    *,
    fail_open: bool,
    code: str,
    input_count: int,
    proposal_count: int,
    max_normalization_error: float = 0.0,
) -> PUFLiteDiagnostics:
    # Keep the executable null mass a local literal as well as private state;
    # no public alias can be rebound to alter the scoring rule.
    lambda_null = 0.4
    return PUFLiteDiagnostics(
        fail_open=fail_open,
        code=code,
        input_candidate_count=input_count,
        proposal_count=proposal_count,
        lambda_null=lambda_null,
        max_proposals=_F_MAX_PROPOSALS,
        max_candidates_per_proposal=_F_MAX_CANDIDATES_PER_PROPOSAL,
        max_total_candidates=_F_MAX_TOTAL_CANDIDATES,
        max_normalization_error=float(max_normalization_error),
    )


def _normalized_entropy(probabilities: Sequence[float]) -> float:
    if len(probabilities) <= 1:
        return 0.0
    entropy = -sum(value * math.log(value) for value in probabilities if value > 0.0)
    result = entropy / math.log(len(probabilities))
    # Suppress harmless floating excursions outside the documented [0, 1].
    return min(1.0, max(0.0, float(result)))


def _score_proposal(
    proposal_id: int, candidates: Sequence[_ValidatedCandidate]
) -> PUFProposalProbability:
    # This literal is intentionally neither configurable nor inferred from GT.
    lambda_null = 0.4
    ordered = tuple(sorted(candidates, key=lambda item: item.track_id))
    if not ordered:
        return PUFProposalProbability(
            proposal_id=proposal_id,
            beta_null=1.0,
            candidates=(),
            argmax_track_id=None,
            argmax_beta=0.0,
            paper_gate=False,
            margin=None,
            normalized_entropy=0.0,
            normalization_error=0.0,
        )
    likelihoods = tuple(
        item.intersection / item.proposal_voxel_count for item in ordered
    )
    normalizer = lambda_null + math.fsum(likelihoods)
    beta_null = lambda_null / normalizer
    betas = tuple(value / normalizer for value in likelihoods)
    normalization_error = abs(math.fsum((beta_null,) + betas) - 1.0)

    # Highest probability wins; stable track ID resolves an exact tie.
    best_index = min(
        range(len(ordered)), key=lambda index: (-betas[index], ordered[index].track_id)
    )
    best_beta = betas[best_index]
    runner_up_beta = max(
        (value for index, value in enumerate(betas) if index != best_index),
        default=0.0,
    )
    margin = best_beta - max(beta_null, runner_up_beta)
    output_candidates = tuple(
        PUFCandidateProbability(
            proposal_id=item.proposal_id,
            track_id=item.track_id,
            intersection=item.intersection,
            proposal_voxel_count=item.proposal_voxel_count,
            spatial_likelihood=likelihood,
            beta=beta,
            track_voxel_count=item.track_voxel_count,
            centroid_distance=item.centroid_distance,
        )
        for item, likelihood, beta in zip(ordered, likelihoods, betas)
    )
    return PUFProposalProbability(
        proposal_id=proposal_id,
        beta_null=float(beta_null),
        candidates=output_candidates,
        argmax_track_id=ordered[best_index].track_id,
        argmax_beta=float(best_beta),
        paper_gate=bool(beta_null <= 0.5),
        margin=float(margin),
        normalized_entropy=_normalized_entropy((beta_null,) + betas),
        normalization_error=float(normalization_error),
    )


def compute_puf_lite(
    candidates: Sequence[object],
    proposal_ids: Optional[Sequence[int]] = None,
) -> PUFLiteResult:
    """Jointly normalize geometry likelihoods for each proposal.

    The input must be a finite ``Sequence`` rather than a lazy iterator so its
    hard cap can be checked before any record is consumed.  Proposal IDs may
    repeat to describe multiple tracks, but each ``(proposal_id, track_id)``
    pair must be unique and every record for a proposal must agree on its voxel
    count.  ``proposal_ids`` optionally declares the complete proposal
    universe.  A declared proposal with no positive-intersection candidates is
    retained as a null-only row, while every candidate is required to belong to
    that universe.  Omitting it preserves the original inferred-universe API.

    Invalid input fails open at batch granularity: ``proposals`` is empty and
    ``diagnostics.code`` explains the first deterministic validation failure.
    No exception from ordinary malformed data escapes this function.
    """

    input_count = 0
    max_normalization_error = 0.0
    try:
        if not isinstance(candidates, Sequence) or isinstance(
            candidates, (str, bytes, bytearray)
        ):
            raise _InvalidInput("invalid_candidate_sequence")
        input_count = len(candidates)
        if input_count > _F_MAX_TOTAL_CANDIDATES:
            raise _InvalidInput("total_candidate_cap")

        explicit_universe: Optional[set[int]] = None
        if proposal_ids is not None:
            if not isinstance(proposal_ids, Sequence) or isinstance(
                proposal_ids, (str, bytes, bytearray)
            ):
                raise _InvalidInput("invalid_proposal_universe")
            if len(proposal_ids) > _F_MAX_PROPOSALS:
                raise _InvalidInput("proposal_cap")
            validated_ids = tuple(
                _integer(
                    value,
                    "proposal_id",
                    minimum=0,
                    maximum=_F_MAX_ID,
                )
                for value in proposal_ids
            )
            if len(set(validated_ids)) != len(validated_ids):
                raise _InvalidInput("duplicate_proposal_id")
            explicit_universe = set(validated_ids)
            grouped: dict[int, list[_ValidatedCandidate]] = {
                proposal_id: [] for proposal_id in validated_ids
            }
        else:
            grouped = {}

        seen_pairs: set[tuple[int, int]] = set()
        proposal_counts: dict[int, int] = {}
        for raw in candidates:
            item = _validate_candidate(raw)
            if (
                explicit_universe is not None
                and item.proposal_id not in explicit_universe
            ):
                raise _InvalidInput("candidate_outside_proposal_universe")
            pair = (item.proposal_id, item.track_id)
            if pair in seen_pairs:
                raise _InvalidInput("duplicate_candidate_pair")
            seen_pairs.add(pair)
            previous_count = proposal_counts.setdefault(
                item.proposal_id, item.proposal_voxel_count
            )
            if previous_count != item.proposal_voxel_count:
                raise _InvalidInput("inconsistent_proposal_voxel_count")
            group = grouped.setdefault(item.proposal_id, [])
            group.append(item)
            if len(group) > _F_MAX_CANDIDATES_PER_PROPOSAL:
                raise _InvalidInput("candidate_per_proposal_cap")
            if explicit_universe is None and len(grouped) > _F_MAX_PROPOSALS:
                raise _InvalidInput("proposal_cap")

        proposals = tuple(
            _score_proposal(proposal_id, grouped[proposal_id])
            for proposal_id in sorted(grouped)
        )
        if proposals:
            raw_max = max(item.normalization_error for item in proposals)
            if not math.isfinite(raw_max):
                # Keep fail-open diagnostics JSON-safe while still reporting a
                # value that is unambiguously outside the frozen tolerance.
                max_normalization_error = 1.0
            else:
                max_normalization_error = float(raw_max)
        if max_normalization_error > _F_MAX_NORMALIZATION_ERROR:
            raise _InvalidInput("normalization_error")
        return PUFLiteResult(
            proposals=proposals,
            diagnostics=_diagnostics(
                fail_open=False,
                code="ok",
                input_count=input_count,
                proposal_count=len(proposals),
                max_normalization_error=max_normalization_error,
            ),
        )
    except _InvalidInput as error:
        return PUFLiteResult(
            proposals=(),
            diagnostics=_diagnostics(
                fail_open=True,
                code=str(error),
                input_count=input_count,
                proposal_count=0,
                max_normalization_error=max_normalization_error,
            ),
        )
    except Exception as error:  # defensive fail-open for hostile record objects
        return PUFLiteResult(
            proposals=(),
            diagnostics=_diagnostics(
                fail_open=True,
                code="exception:%s" % type(error).__name__,
                input_count=input_count,
                proposal_count=0,
                max_normalization_error=max_normalization_error,
            ),
        )
