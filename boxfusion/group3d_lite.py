"""Optimized API-compatible implementation of frozen Group3D-lite.

``group3d_lite_oracle`` preserves the prior reference implementation.  This
module changes only internal representation work: NumPy broad-phase packing and
small-array lexicographic set construction.  Thresholds and decision ordering
are intentionally delegated to the same frozen helpers as the oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import time
from typing import Any, Mapping, Sequence

import numpy as np

from . import group3d_lite_oracle as _o

Association = _o.Association
Diagnostics = _o.Diagnostics
MatchResult = _o.MatchResult
VOXEL_SIZE_METERS = _o.VOXEL_SIZE_METERS
MAX_PROPOSALS = _o.MAX_PROPOSALS
MAX_TRACKS = _o.MAX_TRACKS
MAX_CANDIDATES = _o.MAX_CANDIDATES
MAX_VIEWS_PER_TRACK = _o.MAX_VIEWS_PER_TRACK
MAX_VOXELS_PER_VIEW = _o.MAX_VOXELS_PER_VIEW
MAX_UNION_VOXELS_PER_TRACK = _o.MAX_UNION_VOXELS_PER_TRACK
MIN_VOXELS = _o.MIN_VOXELS
MIN_INTERSECTION = _o.MIN_INTERSECTION
MIN_JACCARD = _o.MIN_JACCARD
MIN_CONTAINMENT = _o.MIN_CONTAINMENT
MAX_CONTAINMENT = _o.MAX_CONTAINMENT
MIN_RUNNER_UP_MARGIN = _o.MIN_RUNNER_UP_MARGIN
_SNAPSHOT_SEAL = object()  # capability minted only by prepare_track_snapshot
# Public aliases above are compatibility/audit metadata only. All executable
# decisions use these module-private literal backstops, so rebinding an exported
# name cannot weaken the frozen preregistration policy at runtime.
_F_MAX_PROPOSALS = 64
_F_MAX_TRACKS = 1024
_F_MAX_CANDIDATES = 8
_F_MAX_VIEWS = 5
_F_MAX_VIEW_VOXELS = 512
_F_MAX_UNION_VOXELS = 1024
_F_MIN_VOXELS = 16
_F_MIN_INTERSECTION = 8
_F_MIN_JACCARD = 0.10
_F_MIN_CONTAINMENT = 0.15
_F_MAX_CONTAINMENT = 0.40
_F_MIN_MARGIN = 0.05
_F_COORDINATE_LIMIT = 1 << 52


def _normalize_voxels_fast(value: Any, *, cap: int, label: str) -> np.ndarray:
    """Same validation/set semantics as ``np.unique(axis=0)``, with less setup.

    Normal fragments are at most 512 rows.  Direct lexicographic sorting avoids
    the structured-array machinery used by ``np.unique(..., axis=0)`` while
    retaining its sorted, duplicate-free int64 output exactly.
    """
    if not isinstance(value, np.ndarray) or value.ndim != 2 or value.shape[1:] != (3,):
        raise ValueError("invalid_voxels_%s" % label)
    if not np.issubdtype(value.dtype, np.signedinteger):
        raise ValueError("non_integer_voxels_%s" % label)
    if value.shape[0] > cap:
        raise ValueError("voxel_cap_%s" % label)
    v = value.astype(np.int64, copy=False)
    if v.size and (np.any(v > _F_COORDINATE_LIMIT) or np.any(v < -_F_COORDINATE_LIMIT)):
        raise ValueError("coordinate_overflow_%s" % label)
    if len(v) < 2:
        return v.copy()
    ordered = v[np.lexsort((v[:, 2], v[:, 1], v[:, 0]))]
    keep = np.empty(len(ordered), dtype=bool)
    keep[0] = True
    keep[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    return ordered[keep]


def _track_union(views: Any) -> np.ndarray:
    if not isinstance(views, Sequence) or isinstance(views, (str, bytes)) or not (1 <= len(views) <= _F_MAX_VIEWS):
        raise ValueError("invalid_track_views")
    normalized = [_normalize_voxels_fast(view, cap=_F_MAX_VIEW_VOXELS, label="track_view") for view in views]
    # A common cached snapshot has exactly one retained view.  It is already a
    # lexicographic set, so re-uniquing it is redundant and was the former hot
    # path.  Multi-view union uses the same lexicographic set operation.
    union = normalized[0] if len(normalized) == 1 else _normalize_voxels_fast(
        np.concatenate(normalized, axis=0), cap=_F_MAX_VIEWS * _F_MAX_VIEW_VOXELS,
        label="track_union",
    )
    if len(union) > _F_MAX_UNION_VOXELS:
        raise ValueError("track_union_cap")
    return union


def match_voxels(
    proposals: Sequence[Mapping[str, Any]],
    tracks: Sequence[Mapping[str, Any]],
    eligible_track_mask: Any,
) -> MatchResult:
    """API-compatible, causal/stateless frozen matcher; see oracle for schema."""
    diagnostic = Diagnostics()
    try:
        chosen = _o._validated_proposals(proposals)
        diagnostic.selected_proposal_ids = tuple(row[0] for row in chosen)
        if not isinstance(tracks, Sequence) or isinstance(tracks, (str, bytes)):
            return _o._fail("invalid_tracks", diagnostic)
        if len(tracks) > _F_MAX_TRACKS:
            return _o._fail("track_cap", diagnostic)
        mask = np.asarray(eligible_track_mask)
        if mask.dtype != np.bool_ or mask.ndim != 1 or len(mask) != len(tracks):
            return _o._fail("invalid_eligibility_mask", diagnostic)

        proposal_fragments = []
        skipped_p: list[int] = []
        fragment_errors: list[tuple[int, str]] = []
        for proposal_id, _score, item in chosen:
            try:
                voxels = _normalize_voxels_fast(_o._item_value(item, "voxels"), cap=_F_MAX_VIEW_VOXELS,
                                                 label="proposal")
            except ValueError as exc:
                skipped_p.append(proposal_id)
                fragment_errors.append((proposal_id, str(exc)))
                continue
            if len(voxels) < _F_MIN_VOXELS:
                skipped_p.append(proposal_id)
                continue
            proposal_fragments.append(_o._fragment(proposal_id, voxels))
        diagnostic.skipped_proposal_ids = tuple(sorted(skipped_p))
        diagnostic.fragment_errors = tuple(sorted(fragment_errors))

        seen_tracks: set[int] = set()
        track_fragments = []
        skipped_t: list[int] = []
        for index, item in enumerate(tracks):
            track_id = _o._as_id(_o._item_value(item, "id"), "track")
            if track_id in seen_tracks:
                return _o._fail("duplicate_track_id", diagnostic)
            seen_tracks.add(track_id)
            if not bool(mask[index]):
                skipped_t.append(track_id)
                continue
            union = _track_union(_o._item_value(item, "views"))
            if len(union) < _F_MIN_VOXELS:
                skipped_t.append(track_id)
                continue
            track_fragments.append(_o._fragment(track_id, union))
        diagnostic.skipped_track_ids = tuple(sorted(skipped_t))

        edges = []
        if track_fragments:
            # Compact contiguous arrays make every proposal's 1024-track AABB
            # test one vectorized operation rather than a Python pair loop.
            track_lo = np.stack([track.lo for track in track_fragments])
            track_hi = np.stack([track.hi for track in track_fragments])
            track_centroids = np.stack([track.centroid for track in track_fragments])
            track_ids = np.asarray([track.item_id for track in track_fragments], dtype=np.int64)
            for proposal in proposal_fragments:
                overlaps = (np.all(proposal.lo - 2 <= track_hi + 2, axis=1)
                            & np.all(track_lo - 2 <= proposal.hi + 2, axis=1))
                indices = np.flatnonzero(overlaps)
                diagnostic.aabb_pairs += len(indices)
                distances = np.linalg.norm(track_centroids[indices] - proposal.centroid, axis=1)
                # Candidate semantics: ascending centroid distance, then stable
                # ID; only the first eight proceed to exact set intersection.
                chosen_indices = indices[np.lexsort((track_ids[indices], distances))[:_F_MAX_CANDIDATES]]
                diagnostic.candidate_pairs += len(chosen_indices)
                for track_index in chosen_indices:
                    possible = _o._edge(proposal, track_fragments[int(track_index)])
                    if possible is not None:
                        edges.append(possible)
        diagnostic.threshold_pairs = len(edges)
        if not edges:
            return MatchResult((), diagnostic)

        by_proposal: dict[int, list[Any]] = {}
        by_track: dict[int, list[Any]] = {}
        for edge in edges:
            by_proposal.setdefault(edge.proposal_id, []).append(edge)
            by_track.setdefault(edge.track_id, []).append(edge)
        for values in by_proposal.values():
            values.sort(key=_o._rank_for_proposal)
        for values in by_track.values():
            values.sort(key=_o._rank_for_track)

        mutual = []
        for values in by_proposal.values():
            best = values[0]
            if by_track[best.track_id][0] is best:
                mutual.append(best)
        diagnostic.mutual_pairs = len(mutual)
        accepted = []
        for edge in mutual:
            p_values, t_values = by_proposal[edge.proposal_id], by_track[edge.track_id]
            p_margin = len(p_values) == 1 or edge.dice - p_values[1].dice >= _F_MIN_MARGIN
            t_margin = len(t_values) == 1 or edge.dice - t_values[1].dice >= _F_MIN_MARGIN
            if p_margin and t_margin:
                accepted.append(Association(edge.proposal_id, edge.track_id, edge.dice,
                                            edge.jaccard, edge.intersection, edge.centroid_distance))
        accepted.sort(key=lambda a: a.proposal_id)
        diagnostic.accepted_pairs = len(accepted)
        return MatchResult(tuple(accepted), diagnostic)
    except ValueError as exc:
        return _o._fail(str(exc), diagnostic)
    except Exception as exc:
        return _o._fail("exception:%s" % type(exc).__name__, diagnostic)


def _readonly(array: np.ndarray) -> np.ndarray:
    """Own and freeze an array so a prepared snapshot cannot alias caller data."""
    frozen = np.ascontiguousarray(array).copy()
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True)
class PreparedFragment:
    """Immutable, precomputed past-track union used by ``match_prepared``."""

    item_id: int
    voxels: np.ndarray
    centroid: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    voxel_count: int
    keys: frozenset[bytes]
    digest: str
    valid: bool

    @property
    def track_id(self) -> int:
        return self.item_id


@dataclass(frozen=True)
class _PreparedPayload:
    """Single sealed identity binding all match-visible cached fields."""

    tracks: tuple[PreparedFragment, ...]
    track_ids: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    centroids: np.ndarray
    valid: np.ndarray
    digest: str
    shape: tuple[int, int, int]
    valid_count: int
    seal: object


@dataclass(frozen=True)
class PreparedTrackSnapshot:
    """Atomic begin-frame-past view onto one sealed internal payload."""

    tracks: tuple[PreparedFragment, ...]
    track_ids: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    centroids: np.ndarray
    valid: np.ndarray
    digest: str
    shape: tuple[int, int, int]
    valid_count: int
    _seal: object
    _payload: _PreparedPayload


@dataclass(frozen=True)
class PrepareResult:
    snapshot: PreparedTrackSnapshot | None
    diagnostics: Diagnostics


@dataclass(frozen=True)
class PreparedProposal:
    """Immutable current proposal fragment, prepared once per keyframe."""

    item_id: int
    voxels: np.ndarray
    centroid: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    voxel_count: int
    keys: frozenset[bytes]


@dataclass(frozen=True)
class PreparedProposalBatch:
    proposals: tuple[PreparedProposal, ...]
    lo: np.ndarray
    hi: np.ndarray
    centroids: np.ndarray
    selected_proposal_ids: tuple[int, ...]
    skipped_proposal_ids: tuple[int, ...]
    fragment_errors: tuple[tuple[int, str], ...]
    digest: str
    _seal: object


@dataclass(frozen=True)
class PrepareProposalsResult:
    batch: PreparedProposalBatch | None
    diagnostics: Diagnostics


@dataclass(frozen=True)
class VoxelPairEvidence:
    """Read-only geometry evidence for one proposal/history candidate pair.

    Distances deliberately remain in signed 5 cm voxel coordinates.  A caller
    that needs metric units must multiply ``centroid_distance_voxels`` by the
    frozen 0.05 m voxel size; keeping the unit in the field name prevents the
    ambiguity present in the legacy association diagnostic.
    """

    proposal_id: int
    track_id: int
    intersection: int
    proposal_voxel_count: int
    track_voxel_count: int
    proposal_containment: float
    track_containment: float
    centroid_distance_voxels: float


@dataclass(frozen=True)
class ProposalPairEvidence:
    """All positive-overlap candidates retained for one valid proposal."""

    proposal_id: int
    candidates: tuple[VoxelPairEvidence, ...]


@dataclass(frozen=True)
class EvidenceDiagnostics:
    """Bounded audit trail; ``fail_open`` always means no evidence rows."""

    fail_open: bool = False
    code: str = "ok"
    selected_proposal_ids: tuple[int, ...] = ()
    skipped_proposal_ids: tuple[int, ...] = ()
    skipped_track_ids: tuple[int, ...] = ()
    fragment_errors: tuple[tuple[int, str], ...] = ()
    aabb_pairs: int = 0
    candidate_pairs: int = 0
    positive_intersection_pairs: int = 0
    # Wall-clock cost of evidence extraction only.  Proposal preparation,
    # Group3D matching, and native-memory commit are deliberately outside this
    # interval so downstream observers can report an honest incremental cost.
    elapsed_ms: float = 0.0


@dataclass(frozen=True)
class PairEvidenceResult:
    """Immutable per-proposal evidence for a downstream probability observer."""

    proposals: tuple[ProposalPairEvidence, ...]
    diagnostics: EvidenceDiagnostics


def _voxel_keys(voxels: np.ndarray) -> frozenset[bytes]:
    """Collision-free 24-byte row identities, built only at preparation time."""
    contiguous = np.ascontiguousarray(voxels, dtype=np.int64)
    return frozenset(row.tobytes() for row in contiguous)


def _fragment_digest(track_id: int, voxels: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(np.asarray([track_id, len(voxels)], dtype=np.int64).tobytes())
    hasher.update(voxels.tobytes(order="C"))
    return hasher.hexdigest()


def _snapshot_digest(fragments: Sequence[PreparedFragment]) -> str:
    hasher = hashlib.sha256()
    for fragment in fragments:
        hasher.update(bytes.fromhex(fragment.digest))
    return hasher.hexdigest()


def _prepare_failure(code: str, diagnostic: Diagnostics) -> PrepareResult:
    diagnostic.fail_open = True
    diagnostic.code = code
    return PrepareResult(None, diagnostic)


def _prepare_track_fragment(item: Mapping[str, Any]) -> PreparedFragment:
    track_id = _o._as_id(_o._item_value(item, "id"), "track")
    union = _track_union(_o._item_value(item, "views"))
    valid = len(union) >= _F_MIN_VOXELS
    if valid:
        fragment = _o._fragment(track_id, union)
        centroid, lo, hi = fragment.centroid, fragment.lo, fragment.hi
    else:
        centroid = np.zeros(3, dtype=np.float64)
        lo = np.zeros(3, dtype=np.int64)
        hi = np.zeros(3, dtype=np.int64)
    return PreparedFragment(
        item_id=track_id, voxels=_readonly(union), centroid=_readonly(centroid),
        lo=_readonly(lo), hi=_readonly(hi), voxel_count=len(union), keys=_voxel_keys(union),
        digest=_fragment_digest(track_id, union), valid=valid,
    )


def _build_snapshot(fragments: Sequence[PreparedFragment]) -> PreparedTrackSnapshot:
    n_tracks = len(fragments)
    track_ids = _readonly(np.asarray([fragment.item_id for fragment in fragments], dtype=np.int64))
    lo = _readonly(np.stack([fragment.lo for fragment in fragments]) if fragments else np.empty((0, 3), dtype=np.int64))
    hi = _readonly(np.stack([fragment.hi for fragment in fragments]) if fragments else np.empty((0, 3), dtype=np.int64))
    centroids = _readonly(np.stack([fragment.centroid for fragment in fragments]) if fragments else np.empty((0, 3), dtype=np.float64))
    valid = _readonly(np.asarray([fragment.valid for fragment in fragments], dtype=bool))
    valid_count = int(valid.sum())
    digest = _snapshot_digest(fragments)
    shape = (n_tracks, 3, valid_count)
    payload = _PreparedPayload(
        tracks=tuple(fragments), track_ids=track_ids, lo=lo, hi=hi, centroids=centroids,
        valid=valid, digest=digest, shape=shape, valid_count=valid_count, seal=_SNAPSHOT_SEAL,
    )
    return PreparedTrackSnapshot(
        tracks=payload.tracks, track_ids=payload.track_ids, lo=payload.lo, hi=payload.hi,
        centroids=payload.centroids, valid=payload.valid, digest=payload.digest, shape=payload.shape,
        valid_count=payload.valid_count, _seal=_SNAPSHOT_SEAL, _payload=payload,
    )


def prepare_track_snapshot(tracks: Sequence[Mapping[str, Any]]) -> PrepareResult:
    """Prepare a bounded, immutable begin-frame-past cache at commit time.

    This function performs all historical 5-view union/AABB/centroid work once.
    It never mutates a caller track or an existing snapshot. Structural/cap
    failures return ``snapshot=None`` and a fail-open diagnostic.
    """
    diagnostic = Diagnostics()
    try:
        if not isinstance(tracks, Sequence) or isinstance(tracks, (str, bytes)):
            return _prepare_failure("invalid_tracks", diagnostic)
        if len(tracks) > _F_MAX_TRACKS:
            return _prepare_failure("track_cap", diagnostic)
        seen: set[int] = set()
        fragments: list[PreparedFragment] = []
        skipped: list[int] = []
        for item in tracks:
            track_id = _o._as_id(_o._item_value(item, "id"), "track")
            if track_id in seen:
                return _prepare_failure("duplicate_track_id", diagnostic)
            seen.add(track_id)
            prepared_fragment = _prepare_track_fragment(item)
            if not prepared_fragment.valid:
                skipped.append(track_id)
            fragments.append(prepared_fragment)
        snapshot = _build_snapshot(fragments)
        diagnostic.skipped_track_ids = tuple(sorted(skipped))
        return PrepareResult(snapshot, diagnostic)
    except ValueError as exc:
        return _prepare_failure(str(exc), diagnostic)
    except Exception as exc:
        return _prepare_failure("exception:%s" % type(exc).__name__, diagnostic)


def update_prepared_track_snapshot(
    previous: PreparedTrackSnapshot,
    committed_tracks: Sequence[Mapping[str, Any]],
) -> PrepareResult:
    """Atomically prepare a post-commit snapshot without mutating ``previous``.

    ``committed_tracks`` is the complete new past snapshot, supplied after all
    current-frame decisions commit. This keeps cache lifecycle outside the
    matcher and therefore preserves causality and stateless matching.
    """
    diagnostic = Diagnostics()
    if _validate_prepared_snapshot(previous) is not None:
        return _prepare_failure("invalid_previous_snapshot", diagnostic)
    return prepare_track_snapshot(committed_tracks)


def update_touched_prepared_tracks(
    previous: PreparedTrackSnapshot,
    touched_tracks: Sequence[Mapping[str, Any]],
) -> PrepareResult:
    """Commit an immutable replacement snapshot after updating existing tracks.

    This bounded fast update accepts at most the existing 1,024 IDs; untouched
    prepared fragments are structurally shared. Adding/removing tracks remains
    an explicit full ``update_prepared_track_snapshot`` operation.
    """
    diagnostic = Diagnostics()
    try:
        invalid = _validate_prepared_snapshot(previous)
        if invalid is not None:
            return _prepare_failure(invalid, diagnostic)
        if not isinstance(touched_tracks, Sequence) or isinstance(touched_tracks, (str, bytes)):
            return _prepare_failure("invalid_touched_tracks", diagnostic)
        payload = previous._payload
        index_by_id = {fragment.item_id: index for index, fragment in enumerate(payload.tracks)}
        replacements: dict[int, PreparedFragment] = {}
        for item in touched_tracks:
            fragment = _prepare_track_fragment(item)
            if fragment.item_id not in index_by_id:
                return _prepare_failure("unknown_touched_track_id", diagnostic)
            if fragment.item_id in replacements:
                return _prepare_failure("duplicate_touched_track_id", diagnostic)
            replacements[fragment.item_id] = fragment
        fragments = list(payload.tracks)
        for track_id, fragment in replacements.items():
            fragments[index_by_id[track_id]] = fragment
        snapshot = _build_snapshot(fragments)
        diagnostic.skipped_track_ids = tuple(sorted(fragment.item_id for fragment in replacements.values()
                                                    if not fragment.valid))
        return PrepareResult(snapshot, diagnostic)
    except ValueError as exc:
        return _prepare_failure(str(exc), diagnostic)
    except Exception as exc:
        return _prepare_failure("exception:%s" % type(exc).__name__, diagnostic)


def _validate_prepared_snapshot(snapshot: Any) -> str | None:
    """Constant-size summary/shape validation; cached voxel data is not rebuilt."""
    if not isinstance(snapshot, PreparedTrackSnapshot):
        return "invalid_prepared_snapshot"
    payload = snapshot._payload
    if not isinstance(payload, _PreparedPayload) or snapshot._seal is not _SNAPSHOT_SEAL or payload.seal is not _SNAPSHOT_SEAL:
        return "invalid_prepared_snapshot"
    # All match-visible data is read from this private payload below. Therefore
    # an object.__setattr__ replacement of public mirror fields (notably
    # ``snapshot.valid``) cannot influence a decision; no per-frame digest walk
    # is needed to defend this non-hostile Python object model.
    n_tracks = len(payload.tracks)
    if (n_tracks > _F_MAX_TRACKS or payload.shape != (n_tracks, 3, payload.valid_count)
            or payload.valid_count < 0 or payload.valid_count > n_tracks):
        return "invalid_prepared_shape"
    arrays = (payload.track_ids, payload.lo, payload.hi, payload.centroids, payload.valid)
    if (any(not isinstance(array, np.ndarray) or array.flags.writeable for array in arrays)
            or payload.track_ids.shape != (n_tracks,)
            or payload.lo.shape != (n_tracks, 3)
            or payload.hi.shape != (n_tracks, 3)
            or payload.centroids.shape != (n_tracks, 3)
            or payload.valid.shape != (n_tracks,)
            or not isinstance(payload.digest, str) or len(payload.digest) != 64):
        return "invalid_prepared_shape"
    return None


def prepare_proposals(proposals: Sequence[Mapping[str, Any]]) -> PrepareProposalsResult:
    """Prepare the frozen top-64 current fragments once for a keyframe."""
    diagnostic = Diagnostics()
    try:
        chosen = _o._validated_proposals(proposals)
        selected_ids = tuple(row[0] for row in chosen)
        prepared: list[PreparedProposal] = []
        skipped: list[int] = []
        errors: list[tuple[int, str]] = []
        digest_source = hashlib.sha256()
        for proposal_id, _score, item in chosen:
            try:
                voxels = _normalize_voxels_fast(_o._item_value(item, "voxels"), cap=_F_MAX_VIEW_VOXELS,
                                                 label="proposal")
            except ValueError as exc:
                skipped.append(proposal_id)
                errors.append((proposal_id, str(exc)))
                continue
            if len(voxels) < _F_MIN_VOXELS:
                skipped.append(proposal_id)
                continue
            fragment = _o._fragment(proposal_id, voxels)
            keys = _voxel_keys(voxels)
            digest_source.update(bytes.fromhex(_fragment_digest(proposal_id, voxels)))
            prepared.append(PreparedProposal(
                item_id=proposal_id, voxels=_readonly(voxels), centroid=_readonly(fragment.centroid),
                lo=_readonly(fragment.lo), hi=_readonly(fragment.hi), voxel_count=len(voxels), keys=keys,
            ))
        diagnostic.selected_proposal_ids = selected_ids
        diagnostic.skipped_proposal_ids = tuple(sorted(skipped))
        diagnostic.fragment_errors = tuple(sorted(errors))
        lo = _readonly(np.stack([proposal.lo for proposal in prepared]) if prepared else np.empty((0, 3), dtype=np.int64))
        hi = _readonly(np.stack([proposal.hi for proposal in prepared]) if prepared else np.empty((0, 3), dtype=np.int64))
        centroids = _readonly(np.stack([proposal.centroid for proposal in prepared]) if prepared else np.empty((0, 3), dtype=np.float64))
        batch = PreparedProposalBatch(
            proposals=tuple(prepared), lo=lo, hi=hi, centroids=centroids, selected_proposal_ids=selected_ids,
            skipped_proposal_ids=diagnostic.skipped_proposal_ids,
            fragment_errors=diagnostic.fragment_errors, digest=digest_source.hexdigest(), _seal=_SNAPSHOT_SEAL,
        )
        return PrepareProposalsResult(batch, diagnostic)
    except ValueError as exc:
        diagnostic.fail_open = True
        diagnostic.code = str(exc)
        return PrepareProposalsResult(None, diagnostic)
    except Exception as exc:
        diagnostic.fail_open = True
        diagnostic.code = "exception:%s" % type(exc).__name__
        return PrepareProposalsResult(None, diagnostic)


def _validate_prepared_proposals(batch: Any) -> str | None:
    n_proposals = len(batch.proposals) if isinstance(batch, PreparedProposalBatch) else -1
    if (not isinstance(batch, PreparedProposalBatch) or n_proposals > _F_MAX_PROPOSALS
            or batch._seal is not _SNAPSHOT_SEAL or not isinstance(batch.digest, str)
            or len(batch.digest) != 64
            or any(not isinstance(array, np.ndarray) or array.flags.writeable for array in (batch.lo, batch.hi, batch.centroids))
            or batch.lo.shape != (n_proposals, 3) or batch.hi.shape != (n_proposals, 3)
            or batch.centroids.shape != (n_proposals, 3)):
        return "invalid_prepared_proposals"
    return None


def _extract_prepared_pair_evidence(
    batch: PreparedProposalBatch,
    snapshot: PreparedTrackSnapshot,
    eligible_track_mask: Any,
) -> PairEvidenceResult:
    """Collect bounded positive voxel intersections without making decisions.

    This is a read-only sidecar for probability calibration.  It deliberately
    shares the prepared matcher's broad phase (two-voxel AABB expansion, then
    the nearest eight tracks by centroid distance and stable ID), but it does
    *not* apply Group3D's intersection, IoU, containment, mutual-best, or margin
    gates.  Every retained candidate with at least one shared voxel is exposed.

    Structural failures fail open to an empty evidence result.  The native
    matcher is neither called nor mutated, and the sealed prepared inputs are
    only read.
    """

    empty = PairEvidenceResult((), EvidenceDiagnostics(fail_open=True))
    try:
        invalid = _validate_prepared_snapshot(snapshot)
        if invalid is None:
            invalid = _validate_prepared_proposals(batch)
        if invalid is not None:
            return PairEvidenceResult(
                (), EvidenceDiagnostics(fail_open=True, code=invalid)
            )

        payload = snapshot._payload
        base = {
            "selected_proposal_ids": batch.selected_proposal_ids,
            "skipped_proposal_ids": batch.skipped_proposal_ids,
            "fragment_errors": batch.fragment_errors,
        }
        mask = np.asarray(eligible_track_mask)
        if mask.dtype != np.bool_ or mask.ndim != 1 or len(mask) != len(payload.tracks):
            return PairEvidenceResult(
                (),
                EvidenceDiagnostics(
                    fail_open=True,
                    code="invalid_eligibility_mask",
                    **base,
                ),
            )

        active = mask & payload.valid
        skipped = payload.track_ids[~mask | ~payload.valid]
        skipped_track_ids = tuple(sorted(int(value) for value in skipped))
        active_indices = np.flatnonzero(active)
        if not len(active_indices):
            rows = tuple(
                ProposalPairEvidence(int(proposal.item_id), ())
                for proposal in batch.proposals
            )
            return PairEvidenceResult(
                rows,
                EvidenceDiagnostics(
                    skipped_track_ids=skipped_track_ids,
                    **base,
                ),
            )

        if len(active_indices) == len(payload.tracks):
            track_ids, track_lo, track_hi, track_centroids = (
                payload.track_ids,
                payload.lo,
                payload.hi,
                payload.centroids,
            )
        else:
            track_ids = payload.track_ids[active_indices]
            track_lo = payload.lo[active_indices]
            track_hi = payload.hi[active_indices]
            track_centroids = payload.centroids[active_indices]

        aabb_pairs = 0
        candidate_pairs = 0
        positive_pairs = 0
        rows: list[ProposalPairEvidence] = []
        candidate_cache: dict[
            tuple[bytes, bytes, bytes],
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ] = {}
        for proposal in batch.proposals:
            geometry_key = (
                proposal.lo.tobytes(),
                proposal.hi.tobytes(),
                proposal.centroid.tobytes(),
            )
            cached_candidates = candidate_cache.get(geometry_key)
            if cached_candidates is None:
                overlaps = (
                    np.all(proposal.lo - 2 <= track_hi + 2, axis=1)
                    & np.all(track_lo - 2 <= proposal.hi + 2, axis=1)
                )
                indices = np.flatnonzero(overlaps)
                distances = np.linalg.norm(
                    track_centroids[indices] - proposal.centroid, axis=1
                )
                candidate_order = np.lexsort(
                    (track_ids[indices], distances)
                )[:_F_MAX_CANDIDATES]
                chosen_indices = indices[candidate_order]
                candidate_cache[geometry_key] = (
                    indices,
                    candidate_order,
                    distances,
                )
            else:
                indices, candidate_order, distances = cached_candidates
                chosen_indices = indices[candidate_order]
            aabb_pairs += len(indices)
            candidate_pairs += len(chosen_indices)

            candidates: list[VoxelPairEvidence] = []
            for relative_index, local_index in zip(
                candidate_order, chosen_indices
            ):
                track = payload.tracks[int(active_indices[int(local_index)])]
                intersection = len(proposal.keys.intersection(track.keys))
                if intersection <= 0:
                    continue
                distance = float(distances[int(relative_index)])
                if not np.isfinite(distance):
                    raise ValueError("nonfinite_centroid_distance")
                candidates.append(
                    VoxelPairEvidence(
                        proposal_id=int(proposal.item_id),
                        track_id=int(track.item_id),
                        intersection=int(intersection),
                        proposal_voxel_count=int(proposal.voxel_count),
                        track_voxel_count=int(track.voxel_count),
                        proposal_containment=float(
                            intersection / proposal.voxel_count
                        ),
                        track_containment=float(
                            intersection / track.voxel_count
                        ),
                        centroid_distance_voxels=distance,
                    )
                )
            candidates.sort(
                key=lambda item: (
                    item.centroid_distance_voxels,
                    item.track_id,
                )
            )
            positive_pairs += len(candidates)
            rows.append(
                ProposalPairEvidence(
                    proposal_id=int(proposal.item_id),
                    candidates=tuple(candidates),
                )
            )

        return PairEvidenceResult(
            proposals=tuple(rows),
            diagnostics=EvidenceDiagnostics(
                skipped_track_ids=skipped_track_ids,
                aabb_pairs=aabb_pairs,
                candidate_pairs=candidate_pairs,
                positive_intersection_pairs=positive_pairs,
                **base,
            ),
        )
    except Exception as exc:
        return PairEvidenceResult(
            empty.proposals,
            EvidenceDiagnostics(
                fail_open=True,
                code="exception:%s" % type(exc).__name__,
            ),
        )


def extract_prepared_pair_evidence(
    batch: PreparedProposalBatch,
    snapshot: PreparedTrackSnapshot,
    eligible_track_mask: Any,
) -> PairEvidenceResult:
    """Timed public wrapper around the read-only pair-evidence extractor.

    The timer spans every success and fail-open path in the evidence pass and
    is attached to the immutable diagnostics.  It intentionally excludes
    proposal/snapshot preparation and the native Group3D matcher.
    """

    started = time.perf_counter_ns()
    result = _extract_prepared_pair_evidence(
        batch, snapshot, eligible_track_mask
    )
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    return PairEvidenceResult(
        proposals=result.proposals,
        diagnostics=replace(result.diagnostics, elapsed_ms=elapsed_ms),
    )


def _edge_cached(
    proposal: PreparedProposal,
    track: PreparedFragment,
    centroid_distance: float,
    metric_cache: dict[Any, Any],
) -> Any | None:
    """Exact frozen metrics, with set identities prebuilt during preparation."""
    cache_key = (proposal.keys, track.keys)
    metric = metric_cache.get(cache_key)
    if metric is None and cache_key not in metric_cache:
        inter = len(proposal.keys.intersection(track.keys))
        if inter < _F_MIN_INTERSECTION:
            metric_cache[cache_key] = False
            return None
        union = proposal.voxel_count + track.voxel_count - inter
        jaccard = inter / union
        cp, ct = inter / proposal.voxel_count, inter / track.voxel_count
        if (jaccard < _F_MIN_JACCARD or min(cp, ct) < _F_MIN_CONTAINMENT
                or max(cp, ct) < _F_MAX_CONTAINMENT):
            metric_cache[cache_key] = False
            return None
        metric = (inter, jaccard, 2.0 * inter / (proposal.voxel_count + track.voxel_count))
        metric_cache[cache_key] = metric
    if metric is False:
        return None
    if not np.isfinite(centroid_distance):
        return None
    inter, jaccard, dice = metric
    return _o._Edge(proposal.item_id, track.item_id, dice, jaccard, inter, float(centroid_distance))


def match_prepared_proposals(
    batch: PreparedProposalBatch,
    snapshot: PreparedTrackSnapshot,
    eligible_track_mask: Any,
) -> MatchResult:
    """Fast matching path: both current and historical fragments are prepared."""
    diagnostic = Diagnostics()
    try:
        invalid = _validate_prepared_snapshot(snapshot) or _validate_prepared_proposals(batch)
        if invalid is not None:
            return _o._fail(invalid, diagnostic)
        payload = snapshot._payload
        diagnostic.selected_proposal_ids = batch.selected_proposal_ids
        diagnostic.skipped_proposal_ids = batch.skipped_proposal_ids
        diagnostic.fragment_errors = batch.fragment_errors
        mask = np.asarray(eligible_track_mask)
        if mask.dtype != np.bool_ or mask.ndim != 1 or len(mask) != len(payload.tracks):
            return _o._fail("invalid_eligibility_mask", diagnostic)
        active = mask & payload.valid
        skipped = payload.track_ids[~mask | ~payload.valid]
        diagnostic.skipped_track_ids = tuple(sorted(int(value) for value in skipped))
        active_indices = np.flatnonzero(active)
        if not len(active_indices):
            return MatchResult((), diagnostic)
        if len(active_indices) == len(payload.tracks):
            track_ids, track_lo, track_hi, track_centroids = (payload.track_ids, payload.lo,
                                                               payload.hi, payload.centroids)
        else:
            track_ids = payload.track_ids[active_indices]
            track_lo = payload.lo[active_indices]
            track_hi = payload.hi[active_indices]
            track_centroids = payload.centroids[active_indices]
        edges = []
        metric_cache: dict[Any, Any] = {}
        candidate_cache: dict[tuple[bytes, bytes, bytes], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for proposal in batch.proposals:
            geometry_key = (proposal.lo.tobytes(), proposal.hi.tobytes(), proposal.centroid.tobytes())
            cached_candidates = candidate_cache.get(geometry_key)
            if cached_candidates is None:
                overlaps = (np.all(proposal.lo - 2 <= track_hi + 2, axis=1)
                            & np.all(track_lo - 2 <= proposal.hi + 2, axis=1))
                indices = np.flatnonzero(overlaps)
                distances = np.linalg.norm(track_centroids[indices] - proposal.centroid, axis=1)
                candidate_order = np.lexsort((track_ids[indices], distances))[:_F_MAX_CANDIDATES]
                chosen_indices = indices[candidate_order]
                candidate_cache[geometry_key] = (indices, candidate_order, distances)
            else:
                indices, candidate_order, distances = cached_candidates
                chosen_indices = indices[candidate_order]
            diagnostic.aabb_pairs += len(indices)
            diagnostic.candidate_pairs += len(chosen_indices)
            for relative_index, local_index in zip(candidate_order, chosen_indices):
                track = payload.tracks[int(active_indices[int(local_index)])]
                possible = _edge_cached(proposal, track, float(distances[int(relative_index)]), metric_cache)
                if possible is not None:
                    edges.append(possible)
        diagnostic.threshold_pairs = len(edges)
        if not edges:
            return MatchResult((), diagnostic)
        by_proposal: dict[int, list[Any]] = {}
        by_track: dict[int, list[Any]] = {}
        for edge in edges:
            by_proposal.setdefault(edge.proposal_id, []).append(edge)
            by_track.setdefault(edge.track_id, []).append(edge)
        for values in by_proposal.values():
            values.sort(key=_o._rank_for_proposal)
        for values in by_track.values():
            values.sort(key=_o._rank_for_track)
        mutual = []
        for values in by_proposal.values():
            best = values[0]
            if by_track[best.track_id][0] is best:
                mutual.append(best)
        diagnostic.mutual_pairs = len(mutual)
        accepted = []
        for edge in mutual:
            p_values, t_values = by_proposal[edge.proposal_id], by_track[edge.track_id]
            p_margin = len(p_values) == 1 or edge.dice - p_values[1].dice >= _F_MIN_MARGIN
            t_margin = len(t_values) == 1 or edge.dice - t_values[1].dice >= _F_MIN_MARGIN
            if p_margin and t_margin:
                accepted.append(Association(edge.proposal_id, edge.track_id, edge.dice,
                                            edge.jaccard, edge.intersection, edge.centroid_distance))
        accepted.sort(key=lambda association: association.proposal_id)
        diagnostic.accepted_pairs = len(accepted)
        return MatchResult(tuple(accepted), diagnostic)
    except Exception as exc:
        return _o._fail("exception:%s" % type(exc).__name__, diagnostic)


def match_prepared(
    proposals: Sequence[Mapping[str, Any]],
    snapshot: PreparedTrackSnapshot,
    eligible_track_mask: Any,
) -> MatchResult:
    """Raw-proposal compatibility wrapper; prepare once then use the fast path."""
    invalid = _validate_prepared_snapshot(snapshot)
    if invalid is not None:
        return _o._fail(invalid, Diagnostics())
    prepared = prepare_proposals(proposals)
    if prepared.batch is None:
        return MatchResult((), prepared.diagnostics)
    return match_prepared_proposals(prepared.batch, snapshot, eligible_track_mask)
