"""Bounded live state for the Stream3Dv2-lite geometry builder.

The state deliberately separates a frame into ``query_frame`` and
``commit_frame``.  A query may inspect only views committed by earlier frame
ordinals.  Tracks requested for retirement are therefore materialized before
any current-frame view is admitted to memory.  The exact query object and its
content token are both required for commit.

This module owns no association policy: the caller supplies a stable track ID
for each current :class:`~boxfusion.stream3dv2_lite.TrackView` and explicitly
identifies tracks retired by its online associator.  It only enforces causal,
bounded storage and calls ``build_track_geometry`` at retirement/finalize.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
from typing import Sequence

import numpy as np

from boxfusion.stream3dv2_lite import (
    MAX_RETAINED_VIEWS,
    TrackGeometry,
    TrackView,
    build_track_geometry,
)


SCHEMA = "boxfusion.stream3dv2_online_state.v1"
MAX_LIVE_TRACKS = 1024
MAX_VIEWS_PER_TRACK = 5

if MAX_VIEWS_PER_TRACK != MAX_RETAINED_VIEWS:  # pragma: no cover - import guard
    raise RuntimeError("online and geometry-builder view caps must agree")


class OnlineStateContractError(ValueError):
    """Raised before state mutation when the online protocol is violated."""


def _strict_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise OnlineStateContractError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise OnlineStateContractError(f"{name} must be non-negative")
    return result


def _track_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise OnlineStateContractError("track_id must be a non-empty string")
    return value


def _array_sha256(value: np.ndarray) -> str:
    row = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(row.dtype).encode("ascii"))
    digest.update(json.dumps(list(row.shape), separators=(",", ":")).encode("ascii"))
    digest.update(row.tobytes(order="C"))
    return digest.hexdigest()


def _view_sha256(view: TrackView) -> str:
    payload = {
        "source_id": view.source_id,
        "frame_id": view.frame_id,
        "frame_ordinal": view.frame_ordinal,
        "mask_confidence": float(view.mask_confidence).hex(),
        "hb_confidence": float(view.hb_confidence).hex(),
        "points_world_sha256": _array_sha256(view.points_world),
        "hb_corners_sha256": _array_sha256(view.hb_corners),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _geometry_sha256(geometry: TrackGeometry) -> str:
    payload = {
        "source_ids": list(geometry.source_ids),
        "frame_ids": list(geometry.frame_ids),
        "decision_frame_id": geometry.decision_frame_id,
        "decision_frame_ordinal": geometry.decision_frame_ordinal,
        "selected_source_ids": list(geometry.selected_source_ids),
        "hb_source_id": geometry.hb_source_id,
        "hypotheses": {
            key: _array_sha256(value)
            for key, value in sorted(geometry.hypotheses.items())
        },
        "hypothesis_quality": {
            key: float(value).hex()
            for key, value in sorted(geometry.hypothesis_quality.items())
        },
        "chosen_hypothesis": geometry.chosen_hypothesis,
        "corners_sha256": _array_sha256(geometry.corners),
        "refined_points_sha256": _array_sha256(geometry.refined_points),
        "distinct_view_count": geometry.distinct_view_count,
        "set_cover_fraction": float(geometry.set_cover_fraction).hex(),
        "median_pairwise_hb_iou": float(geometry.median_pairwise_hb_iou).hex(),
        "median_pairwise_hb_containment": float(
            geometry.median_pairwise_hb_containment
        ).hex(),
        "hb_center_rms_m": float(geometry.hb_center_rms_m).hex(),
        "point_inside_hb_fraction": float(geometry.point_inside_hb_fraction).hex(),
        "pmr_seed_fraction": float(geometry.pmr_seed_fraction).hex(),
        "pmr_retained_fraction": float(geometry.pmr_retained_fraction).hex(),
        "mask_confidence_mean": float(geometry.mask_confidence_mean).hex(),
        "hb_confidence_mean": float(geometry.hb_confidence_mean).hex(),
        "preliminary_score": float(geometry.preliminary_score).hex(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class TrackUpdate:
    """One current-frame view assigned to one live track."""

    track_id: str
    view: TrackView

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", _track_id(self.track_id))
        if not isinstance(self.view, TrackView):
            raise OnlineStateContractError("view must be a TrackView")


@dataclass(frozen=True)
class FinalizedTrack:
    """Geometry emitted from a frozen, past-only track prefix."""

    track_id: str
    geometry: TrackGeometry
    input_view_count: int
    last_view_frame_ordinal: int


@dataclass(frozen=True)
class OnlineFrameQuery:
    frame_id: int
    frame_ordinal: int
    updates: tuple[TrackUpdate, ...]
    retire_track_ids: tuple[str, ...]
    retired: tuple[FinalizedTrack, ...]
    memory_version_before: int
    live_track_count_before: int
    live_view_count_before: int
    maximum_accessed_frame_ordinal: int
    token: str


@dataclass(frozen=True)
class OnlineFrameCommit:
    frame_id: int
    frame_ordinal: int
    retired: tuple[FinalizedTrack, ...]
    committed_view_count: int
    live_track_count_after: int
    live_view_count_after: int
    retired_before_current_views: bool
    memory_version_after: int
    token: str


@dataclass(frozen=True)
class OnlineStatistics:
    committed_frame_count: int
    committed_view_count: int
    retired_track_count: int
    terminal_finalized_track_count: int
    peak_live_track_count: int
    peak_live_view_count: int
    live_track_count: int
    live_view_count: int
    last_committed_frame_ordinal: int
    future_access_count: int
    query_before_commit: bool


@dataclass(frozen=True)
class OnlineTerminalSeal:
    tracks: tuple[FinalizedTrack, ...]
    statistics: OnlineStatistics
    maximum_accessed_frame_ordinal: int


def _query_token(
    *,
    frame_id: int,
    frame_ordinal: int,
    updates: Sequence[TrackUpdate],
    retire_track_ids: Sequence[str],
    retired: Sequence[FinalizedTrack],
    memory_version_before: int,
    live_track_count_before: int,
    live_view_count_before: int,
    maximum_accessed_frame_ordinal: int,
) -> str:
    payload = {
        "schema": SCHEMA,
        "frame_id": frame_id,
        "frame_ordinal": frame_ordinal,
        "updates": [
            {"track_id": row.track_id, "view_sha256": _view_sha256(row.view)}
            for row in updates
        ],
        "retire_track_ids": list(retire_track_ids),
        "retired": [
            {
                "track_id": row.track_id,
                "input_view_count": row.input_view_count,
                "last_view_frame_ordinal": row.last_view_frame_ordinal,
                "geometry_sha256": _geometry_sha256(row.geometry),
            }
            for row in retired
        ],
        "memory_version_before": memory_version_before,
        "live_track_count_before": live_track_count_before,
        "live_view_count_before": live_view_count_before,
        "maximum_accessed_frame_ordinal": maximum_accessed_frame_ordinal,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class Stream3Dv2OnlineState:
    """Exact-token, query-before-commit bounded TrackView memory."""

    def __init__(self) -> None:
        self._tracks: dict[str, tuple[TrackView, ...]] = {}
        self._seen_source_ids: set[str] = set()
        self._seen_frame_ids: set[int] = set()
        self._pending: OnlineFrameQuery | None = None
        self._terminal: OnlineTerminalSeal | None = None
        self._last_committed_ordinal = -1
        self._memory_version = 0
        self._committed_frame_count = 0
        self._committed_view_count = 0
        self._retired_track_count = 0
        self._terminal_finalized_track_count = 0
        self._peak_live_track_count = 0
        self._peak_live_view_count = 0

    @property
    def live_track_count(self) -> int:
        return len(self._tracks)

    @property
    def live_view_count(self) -> int:
        return sum(len(rows) for rows in self._tracks.values())

    @property
    def live_track_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._tracks))

    @property
    def statistics(self) -> OnlineStatistics:
        return OnlineStatistics(
            committed_frame_count=self._committed_frame_count,
            committed_view_count=self._committed_view_count,
            retired_track_count=self._retired_track_count,
            terminal_finalized_track_count=self._terminal_finalized_track_count,
            peak_live_track_count=self._peak_live_track_count,
            peak_live_view_count=self._peak_live_view_count,
            live_track_count=self.live_track_count,
            live_view_count=self.live_view_count,
            last_committed_frame_ordinal=self._last_committed_ordinal,
            future_access_count=0,
            query_before_commit=True,
        )

    def views_for_track(self, track_id: str) -> tuple[TrackView, ...]:
        """Return the immutable retained prefix for diagnostics/tests."""

        key = _track_id(track_id)
        if key not in self._tracks:
            raise OnlineStateContractError("track is not live")
        return self._tracks[key]

    def _ensure_open(self) -> None:
        if self._terminal is not None:
            raise OnlineStateContractError("online state has already been finalized")

    def query_frame(
        self,
        *,
        frame_id: int,
        frame_ordinal: int,
        updates: Sequence[TrackUpdate] = (),
        retire_track_ids: Sequence[str] = (),
    ) -> OnlineFrameQuery:
        """Inspect the committed prefix and prepare one atomic frame commit."""

        self._ensure_open()
        if self._pending is not None:
            raise OnlineStateContractError("the previous query has not been committed")
        current_frame_id = _strict_nonnegative_int(frame_id, "frame_id")
        current_ordinal = _strict_nonnegative_int(frame_ordinal, "frame_ordinal")
        if current_ordinal <= self._last_committed_ordinal:
            raise OnlineStateContractError("frame ordinals must be strictly increasing")
        if current_frame_id in self._seen_frame_ids:
            raise OnlineStateContractError("frame_id was already committed")

        update_rows = tuple(updates)
        if any(not isinstance(row, TrackUpdate) for row in update_rows):
            raise OnlineStateContractError("updates must contain only TrackUpdate rows")
        update_track_ids = [row.track_id for row in update_rows]
        update_source_ids = [row.view.source_id for row in update_rows]
        if len(set(update_track_ids)) != len(update_track_ids):
            raise OnlineStateContractError("a track may receive at most one view per frame")
        if len(set(update_source_ids)) != len(update_source_ids):
            raise OnlineStateContractError("duplicate source_id in current frame")
        if any(source_id in self._seen_source_ids for source_id in update_source_ids):
            raise OnlineStateContractError("source_id was already committed")
        for row in update_rows:
            if (
                row.view.frame_id != current_frame_id
                or row.view.frame_ordinal != current_ordinal
            ):
                raise OnlineStateContractError(
                    "current view identity must exactly match the queried frame"
                )
        ordered_updates = tuple(
            sorted(update_rows, key=lambda row: (row.track_id, row.view.source_id))
        )

        retirement_rows = tuple(_track_id(value) for value in retire_track_ids)
        if len(set(retirement_rows)) != len(retirement_rows):
            raise OnlineStateContractError("duplicate retirement track_id")
        ordered_retirements = tuple(sorted(retirement_rows))
        unknown = [track_id for track_id in ordered_retirements if track_id not in self._tracks]
        if unknown:
            raise OnlineStateContractError("cannot retire a track absent from past memory")
        if set(update_track_ids).intersection(ordered_retirements):
            raise OnlineStateContractError(
                "a track cannot retire and receive a current view in the same frame"
            )

        # All rows in self._tracks were committed at earlier ordinals.  Check
        # this invariant before geometry construction so corruption fails
        # closed without publishing a pending token.
        prior_views = [view for rows in self._tracks.values() for view in rows]
        if any(view.frame_ordinal >= current_ordinal for view in prior_views):
            raise OnlineStateContractError("past memory contains current/future evidence")
        maximum_accessed = max(
            (view.frame_ordinal for view in prior_views), default=-1
        )

        retired: list[FinalizedTrack] = []
        for track_id in ordered_retirements:
            views = self._tracks[track_id]
            geometry = build_track_geometry(views)
            if geometry.decision_frame_ordinal >= current_ordinal:
                raise OnlineStateContractError("retirement geometry accessed a future frame")
            retired.append(
                FinalizedTrack(
                    track_id=track_id,
                    geometry=geometry,
                    input_view_count=len(views),
                    last_view_frame_ordinal=max(row.frame_ordinal for row in views),
                )
            )

        remaining_ids = set(self._tracks).difference(ordered_retirements)
        new_ids = {row.track_id for row in ordered_updates}.difference(remaining_ids)
        if len(remaining_ids) + len(new_ids) > MAX_LIVE_TRACKS:
            raise OnlineStateContractError("live track capacity would be exceeded")

        token = _query_token(
            frame_id=current_frame_id,
            frame_ordinal=current_ordinal,
            updates=ordered_updates,
            retire_track_ids=ordered_retirements,
            retired=retired,
            memory_version_before=self._memory_version,
            live_track_count_before=self.live_track_count,
            live_view_count_before=self.live_view_count,
            maximum_accessed_frame_ordinal=maximum_accessed,
        )
        query = OnlineFrameQuery(
            frame_id=current_frame_id,
            frame_ordinal=current_ordinal,
            updates=ordered_updates,
            retire_track_ids=ordered_retirements,
            retired=tuple(retired),
            memory_version_before=self._memory_version,
            live_track_count_before=self.live_track_count,
            live_view_count_before=self.live_view_count,
            maximum_accessed_frame_ordinal=maximum_accessed,
            token=token,
        )
        self._pending = query
        return query

    def commit_frame(self, query: OnlineFrameQuery, *, token: str) -> OnlineFrameCommit:
        """Atomically retire past tracks, then admit the current-frame views."""

        self._ensure_open()
        if self._pending is None or query is not self._pending:
            raise OnlineStateContractError("commit requires the exact pending query object")
        if not isinstance(token, str) or not hmac.compare_digest(token, query.token):
            raise OnlineStateContractError("commit token differs from the pending query")
        if (
            query.memory_version_before != self._memory_version
            or query.live_track_count_before != self.live_track_count
            or query.live_view_count_before != self.live_view_count
        ):
            raise OnlineStateContractError("past memory changed after query")
        expected_token = _query_token(
            frame_id=query.frame_id,
            frame_ordinal=query.frame_ordinal,
            updates=query.updates,
            retire_track_ids=query.retire_track_ids,
            retired=query.retired,
            memory_version_before=query.memory_version_before,
            live_track_count_before=query.live_track_count_before,
            live_view_count_before=query.live_view_count_before,
            maximum_accessed_frame_ordinal=query.maximum_accessed_frame_ordinal,
        )
        if not hmac.compare_digest(expected_token, query.token):
            raise OnlineStateContractError("pending query content/token changed")

        # Construct the full next state locally.  Retirement is intentionally
        # applied before any update, permitting a full 1024-track memory to
        # retire one track and admit one replacement atomically.
        next_tracks = dict(self._tracks)
        for track_id in query.retire_track_ids:
            if track_id not in next_tracks:
                raise OnlineStateContractError("retirement track disappeared after query")
            del next_tracks[track_id]
        for update in query.updates:
            retained = next_tracks.get(update.track_id, ())
            next_tracks[update.track_id] = (retained + (update.view,))[-MAX_VIEWS_PER_TRACK:]
        if len(next_tracks) > MAX_LIVE_TRACKS or any(
            len(rows) > MAX_VIEWS_PER_TRACK for rows in next_tracks.values()
        ):
            raise OnlineStateContractError("bounded-memory invariant failed")

        self._tracks = next_tracks
        self._seen_source_ids.update(row.view.source_id for row in query.updates)
        self._seen_frame_ids.add(query.frame_id)
        self._last_committed_ordinal = query.frame_ordinal
        self._memory_version += 1
        self._committed_frame_count += 1
        self._committed_view_count += len(query.updates)
        self._retired_track_count += len(query.retired)
        self._peak_live_track_count = max(
            self._peak_live_track_count, self.live_track_count
        )
        self._peak_live_view_count = max(
            self._peak_live_view_count, self.live_view_count
        )
        self._pending = None
        return OnlineFrameCommit(
            frame_id=query.frame_id,
            frame_ordinal=query.frame_ordinal,
            retired=query.retired,
            committed_view_count=len(query.updates),
            live_track_count_after=self.live_track_count,
            live_view_count_after=self.live_view_count,
            retired_before_current_views=True,
            memory_version_after=self._memory_version,
            token=query.token,
        )

    def process_frame(
        self,
        *,
        frame_id: int,
        frame_ordinal: int,
        updates: Sequence[TrackUpdate] = (),
        retire_track_ids: Sequence[str] = (),
    ) -> tuple[OnlineFrameQuery, OnlineFrameCommit]:
        """Convenience wrapper preserving the same exact-token protocol."""

        query = self.query_frame(
            frame_id=frame_id,
            frame_ordinal=frame_ordinal,
            updates=updates,
            retire_track_ids=retire_track_ids,
        )
        return query, self.commit_frame(query, token=query.token)

    def finalize(self) -> OnlineTerminalSeal:
        """Materialize every still-live track and permanently seal the state."""

        if self._pending is not None:
            raise OnlineStateContractError("pending query must be committed before finalize")
        if self._terminal is not None:
            return self._terminal
        tracks = tuple(
            FinalizedTrack(
                track_id=track_id,
                geometry=build_track_geometry(self._tracks[track_id]),
                input_view_count=len(self._tracks[track_id]),
                last_view_frame_ordinal=max(
                    row.frame_ordinal for row in self._tracks[track_id]
                ),
            )
            for track_id in sorted(self._tracks)
        )
        maximum_accessed = max(
            (
                row.frame_ordinal
                for rows in self._tracks.values()
                for row in rows
            ),
            default=-1,
        )
        self._terminal_finalized_track_count += len(tracks)
        self._tracks = {}
        terminal = OnlineTerminalSeal(
            tracks=tracks,
            statistics=self.statistics,
            maximum_accessed_frame_ordinal=maximum_accessed,
        )
        self._terminal = terminal
        return terminal


__all__ = [
    "FinalizedTrack",
    "MAX_LIVE_TRACKS",
    "MAX_VIEWS_PER_TRACK",
    "OnlineFrameCommit",
    "OnlineFrameQuery",
    "OnlineStateContractError",
    "OnlineStatistics",
    "OnlineTerminalSeal",
    "SCHEMA",
    "Stream3Dv2OnlineState",
    "TrackUpdate",
]
