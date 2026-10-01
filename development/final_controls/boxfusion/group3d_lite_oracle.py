"""Frozen NumPy-only Group3D-lite reference matcher.

This module deliberately returns only duplicate-to-old-track associations.
Callers retain ownership of native BoxFusion predictions and apply no mutation
until the complete decision list has been produced.

The uppercase names are public audit metadata.  Executable policy uses private
literal backstops so rebinding an exported name cannot alter the preregistered
association policy in a long-running Python process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


# Public, inspectable policy metadata.
VOXEL_SIZE_METERS = 0.05
MAX_PROPOSALS = 64
MAX_TRACKS = 1024
MAX_CANDIDATES = 8
MAX_VIEWS_PER_TRACK = 5
MAX_VOXELS_PER_VIEW = 512
MAX_UNION_VOXELS_PER_TRACK = 1024
MIN_VOXELS = 16
MIN_INTERSECTION = 8
MIN_JACCARD = 0.10
MIN_CONTAINMENT = 0.15
MAX_CONTAINMENT = 0.40
MIN_RUNNER_UP_MARGIN = 0.05

# Private executable-policy backstops.  Do not replace these with references to
# the public names above: public names are intentionally safe to rebind.
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
_F_COORDINATE_LIMIT = 1 << 52  # exact in float64, with AABB-expansion headroom


@dataclass(frozen=True)
class Association:
    """An accepted proposal -> begin-frame-past track association."""

    proposal_id: int
    track_id: int
    dice: float
    jaccard: float
    intersection: int
    centroid_distance: float


@dataclass
class Diagnostics:
    """Non-throwing audit trail. ``fail_open`` always means no association."""

    fail_open: bool = False
    code: str = "ok"
    selected_proposal_ids: tuple[int, ...] = ()
    skipped_proposal_ids: tuple[int, ...] = ()
    skipped_track_ids: tuple[int, ...] = ()
    fragment_errors: tuple[tuple[int, str], ...] = ()
    aabb_pairs: int = 0
    candidate_pairs: int = 0
    threshold_pairs: int = 0
    mutual_pairs: int = 0
    accepted_pairs: int = 0


@dataclass(frozen=True)
class MatchResult:
    associations: tuple[Association, ...]
    diagnostics: Diagnostics


@dataclass(frozen=True)
class _Fragment:
    item_id: int
    voxels: np.ndarray
    centroid: np.ndarray
    lo: np.ndarray
    hi: np.ndarray


@dataclass(frozen=True)
class _Edge:
    proposal_id: int
    track_id: int
    dice: float
    jaccard: float
    intersection: int
    centroid_distance: float


def _fail(code: str, diagnostics: Diagnostics) -> MatchResult:
    diagnostics.fail_open = True
    diagnostics.code = code
    return MatchResult((), diagnostics)


def _as_id(value: Any, kind: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("invalid_%s_id" % kind)
    return int(value)


def _item_value(item: Any, name: str) -> Any:
    if not isinstance(item, Mapping) or name not in item:
        raise ValueError("missing_%s" % name)
    return item[name]


def _normalize_voxels(value: Any, *, cap: int, label: str) -> np.ndarray:
    """Validate an Nx3 signed-int array and return its lexicographic set."""
    if not isinstance(value, np.ndarray) or value.ndim != 2 or value.shape[1:] != (3,):
        raise ValueError("invalid_voxels_%s" % label)
    if not np.issubdtype(value.dtype, np.signedinteger):
        raise ValueError("non_integer_voxels_%s" % label)
    if value.shape[0] > cap:
        raise ValueError("voxel_cap_%s" % label)
    voxels = value.astype(np.int64, copy=False)
    if voxels.size and (
        np.any(voxels > _F_COORDINATE_LIMIT)
        or np.any(voxels < -_F_COORDINATE_LIMIT)
    ):
        raise ValueError("coordinate_overflow_%s" % label)
    return np.unique(voxels, axis=0)


def _fragment(item_id: int, voxels: np.ndarray) -> _Fragment:
    return _Fragment(
        item_id=item_id,
        voxels=voxels,
        centroid=voxels.astype(np.float64).mean(axis=0),
        lo=voxels.min(axis=0),
        hi=voxels.max(axis=0),
    )


def _aabb_overlap(a: _Fragment, b: _Fragment) -> bool:
    # Both fragments expand by exactly two 5-cm voxels.
    return bool(np.all(a.lo - 2 <= b.hi + 2) and np.all(b.lo - 2 <= a.hi + 2))


def _intersection_size(a: np.ndarray, b: np.ndarray) -> int:
    return len(
        {(int(x), int(y), int(z)) for x, y, z in a}.intersection(
            (int(x), int(y), int(z)) for x, y, z in b
        )
    )


def _edge(proposal: _Fragment, track: _Fragment) -> _Edge | None:
    p_count, t_count = len(proposal.voxels), len(track.voxels)
    if p_count < _F_MIN_VOXELS or t_count < _F_MIN_VOXELS:
        return None
    intersection = _intersection_size(proposal.voxels, track.voxels)
    if intersection < _F_MIN_INTERSECTION:
        return None
    union = p_count + t_count - intersection
    jaccard = intersection / union
    proposal_containment = intersection / p_count
    track_containment = intersection / t_count
    if (
        jaccard < _F_MIN_JACCARD
        or min(proposal_containment, track_containment) < _F_MIN_CONTAINMENT
        or max(proposal_containment, track_containment) < _F_MAX_CONTAINMENT
    ):
        return None
    dice = 2.0 * intersection / (p_count + t_count)
    distance = float(np.linalg.norm(proposal.centroid - track.centroid))
    if not np.isfinite(distance):
        return None
    return _Edge(
        proposal.item_id,
        track.item_id,
        dice,
        jaccard,
        intersection,
        distance,
    )


def _rank_for_proposal(edge: _Edge) -> tuple[float, float, int, float, int, int]:
    return (
        -edge.dice,
        -edge.jaccard,
        -edge.intersection,
        edge.centroid_distance,
        edge.track_id,
        edge.proposal_id,
    )


def _rank_for_track(edge: _Edge) -> tuple[float, float, int, float, int, int]:
    return _rank_for_proposal(edge)


def _validated_proposals(
    proposals: Sequence[Mapping[str, Any]],
) -> list[tuple[int, float, Any]]:
    if not isinstance(proposals, Sequence) or isinstance(proposals, (str, bytes)):
        raise ValueError("invalid_proposals")
    seen: set[int] = set()
    rows: list[tuple[int, float, Any]] = []
    for item in proposals:
        proposal_id = _as_id(_item_value(item, "id"), "proposal")
        if proposal_id in seen:
            raise ValueError("duplicate_proposal_id")
        seen.add(proposal_id)
        score = _item_value(item, "score")
        if isinstance(score, (bool, np.bool_)) or not isinstance(
            score, (int, float, np.number)
        ):
            raise ValueError("invalid_proposal_score")
        score = float(score)
        if not np.isfinite(score):
            raise ValueError("nonfinite_proposal_score")
        rows.append((proposal_id, score, item))
    # Detector score is frozen; proposal ID deterministically breaks score ties.
    return sorted(rows, key=lambda row: (-row[1], row[0]))[:_F_MAX_PROPOSALS]


def match_voxels(
    proposals: Sequence[Mapping[str, Any]],
    tracks: Sequence[Mapping[str, Any]],
    eligible_track_mask: Any,
) -> MatchResult:
    """Match current fragments only to explicitly eligible past tracks.

    Proposal schema: ``id``, ``score``, ``voxels`` (Nx3 signed-int ndarray).
    Track schema: ``id``, ``views`` (one to five Nx3 signed-int ndarrays).
    Structural errors and resource-cap violations fail open with no association.
    """
    diagnostic = Diagnostics()
    try:
        chosen = _validated_proposals(proposals)
        diagnostic.selected_proposal_ids = tuple(row[0] for row in chosen)
        if not isinstance(tracks, Sequence) or isinstance(tracks, (str, bytes)):
            return _fail("invalid_tracks", diagnostic)
        if len(tracks) > _F_MAX_TRACKS:
            return _fail("track_cap", diagnostic)
        mask = np.asarray(eligible_track_mask)
        if mask.dtype != np.bool_ or mask.ndim != 1 or len(mask) != len(tracks):
            return _fail("invalid_eligibility_mask", diagnostic)

        proposal_fragments: list[_Fragment] = []
        skipped_proposals: list[int] = []
        fragment_errors: list[tuple[int, str]] = []
        for proposal_id, _score, item in chosen:
            try:
                voxels = _normalize_voxels(
                    _item_value(item, "voxels"),
                    cap=_F_MAX_VIEW_VOXELS,
                    label="proposal",
                )
            except ValueError as exc:
                skipped_proposals.append(proposal_id)
                fragment_errors.append((proposal_id, str(exc)))
                continue
            if len(voxels) < _F_MIN_VOXELS:
                skipped_proposals.append(proposal_id)
                continue
            proposal_fragments.append(_fragment(proposal_id, voxels))
        diagnostic.skipped_proposal_ids = tuple(sorted(skipped_proposals))
        diagnostic.fragment_errors = tuple(sorted(fragment_errors))

        seen_tracks: set[int] = set()
        track_fragments: list[_Fragment] = []
        skipped_tracks: list[int] = []
        for index, item in enumerate(tracks):
            track_id = _as_id(_item_value(item, "id"), "track")
            if track_id in seen_tracks:
                return _fail("duplicate_track_id", diagnostic)
            seen_tracks.add(track_id)
            if not bool(mask[index]):
                skipped_tracks.append(track_id)
                continue
            views = _item_value(item, "views")
            if (
                not isinstance(views, Sequence)
                or isinstance(views, (str, bytes))
                or not (1 <= len(views) <= _F_MAX_VIEWS)
            ):
                return _fail("invalid_track_views", diagnostic)
            normalized = [
                _normalize_voxels(
                    view,
                    cap=_F_MAX_VIEW_VOXELS,
                    label="track_view",
                )
                for view in views
            ]
            union = np.unique(np.concatenate(normalized, axis=0), axis=0)
            if len(union) > _F_MAX_UNION_VOXELS:
                return _fail("track_union_cap", diagnostic)
            if len(union) < _F_MIN_VOXELS:
                skipped_tracks.append(track_id)
                continue
            track_fragments.append(_fragment(track_id, union))
        diagnostic.skipped_track_ids = tuple(sorted(skipped_tracks))

        edges: list[_Edge] = []
        if track_fragments:
            track_lo = np.stack([track.lo for track in track_fragments])
            track_hi = np.stack([track.hi for track in track_fragments])
            track_centroids = np.stack(
                [track.centroid for track in track_fragments]
            )
            track_ids = np.asarray(
                [track.item_id for track in track_fragments], dtype=np.int64
            )
        for proposal in proposal_fragments:
            if not track_fragments:
                break
            overlaps = np.all(proposal.lo - 2 <= track_hi + 2, axis=1) & np.all(
                track_lo - 2 <= proposal.hi + 2, axis=1
            )
            indices = np.flatnonzero(overlaps)
            diagnostic.aabb_pairs += len(indices)
            distances = np.linalg.norm(
                track_centroids[indices] - proposal.centroid, axis=1
            )
            order = np.lexsort((track_ids[indices], distances))[:_F_MAX_CANDIDATES]
            nearby_indices = indices[order]
            diagnostic.candidate_pairs += len(nearby_indices)
            for index in nearby_indices:
                possible = _edge(proposal, track_fragments[int(index)])
                if possible is not None:
                    edges.append(possible)
        diagnostic.threshold_pairs = len(edges)
        if not edges:
            return MatchResult((), diagnostic)

        by_proposal: dict[int, list[_Edge]] = {}
        by_track: dict[int, list[_Edge]] = {}
        for edge in edges:
            by_proposal.setdefault(edge.proposal_id, []).append(edge)
            by_track.setdefault(edge.track_id, []).append(edge)
        for values in by_proposal.values():
            values.sort(key=_rank_for_proposal)
        for values in by_track.values():
            values.sort(key=_rank_for_track)

        mutual: list[_Edge] = []
        for values in by_proposal.values():
            best = values[0]
            if by_track[best.track_id][0] is best:
                mutual.append(best)
        diagnostic.mutual_pairs = len(mutual)

        accepted: list[Association] = []
        for edge in mutual:
            proposal_edges = by_proposal[edge.proposal_id]
            track_edges = by_track[edge.track_id]
            proposal_margin = (
                len(proposal_edges) == 1
                or edge.dice - proposal_edges[1].dice >= _F_MIN_MARGIN
            )
            track_margin = (
                len(track_edges) == 1
                or edge.dice - track_edges[1].dice >= _F_MIN_MARGIN
            )
            if proposal_margin and track_margin:
                accepted.append(
                    Association(
                        edge.proposal_id,
                        edge.track_id,
                        edge.dice,
                        edge.jaccard,
                        edge.intersection,
                        edge.centroid_distance,
                    )
                )
        accepted.sort(key=lambda association: association.proposal_id)
        diagnostic.accepted_pairs = len(accepted)
        return MatchResult(tuple(accepted), diagnostic)
    except ValueError as exc:
        return _fail(str(exc), diagnostic)
    except Exception as exc:
        return _fail("exception:%s" % type(exc).__name__, diagnostic)
