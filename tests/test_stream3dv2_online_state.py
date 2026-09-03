from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from boxfusion.stream3dv2_lite import TrackView
from boxfusion.stream3dv2_online_state import (
    MAX_LIVE_TRACKS,
    MAX_VIEWS_PER_TRACK,
    OnlineStateContractError,
    Stream3Dv2OnlineState,
    TrackUpdate,
)


_SIGNS = np.asarray(
    [
        (-1.0, -1.0, -1.0),
        (-1.0, -1.0, +1.0),
        (-1.0, +1.0, -1.0),
        (-1.0, +1.0, +1.0),
        (+1.0, -1.0, -1.0),
        (+1.0, -1.0, +1.0),
        (+1.0, +1.0, -1.0),
        (+1.0, +1.0, +1.0),
    ],
    dtype=np.float64,
)


def _view(source_id: str, ordinal: int, *, frame_id: int | None = None) -> TrackView:
    base = np.stack(
        np.meshgrid(
            np.arange(3, dtype=np.float64),
            np.arange(3, dtype=np.float64),
            np.arange(2, dtype=np.float64),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    points = (base + np.asarray([0.25 + ordinal * 0.02, 0.25, 20.25])) * 0.05
    center = np.asarray([0.1 + ordinal * 0.001, 0.1, 1.05])
    corners = center[None] + _SIGNS * np.asarray([0.8, 0.8, 0.5])[None] * 0.5
    return TrackView(
        source_id=source_id,
        frame_id=ordinal * 25 if frame_id is None else frame_id,
        frame_ordinal=ordinal,
        mask_confidence=0.8,
        hb_confidence=0.7,
        points_world=points,
        hb_corners=corners,
    )


def _update(track_id: str, ordinal: int, *, source_id: str | None = None) -> TrackUpdate:
    return TrackUpdate(
        track_id=track_id,
        view=_view(source_id or f"{track_id}:{ordinal}", ordinal),
    )


def _commit(
    state: Stream3Dv2OnlineState,
    ordinal: int,
    updates: tuple[TrackUpdate, ...] = (),
    retire: tuple[str, ...] = (),
):
    query = state.query_frame(
        frame_id=ordinal * 25,
        frame_ordinal=ordinal,
        updates=updates,
        retire_track_ids=retire,
    )
    return query, state.commit_frame(query, token=query.token)


def test_query_before_commit_requires_exact_object_and_token_and_is_atomic():
    state = Stream3Dv2OnlineState()
    query = state.query_frame(
        frame_id=0,
        frame_ordinal=0,
        updates=(_update("track", 0),),
    )

    assert state.live_track_count == 0
    assert query.maximum_accessed_frame_ordinal == -1
    forged = replace(query)
    with pytest.raises(OnlineStateContractError, match="exact pending"):
        state.commit_frame(forged, token=forged.token)
    with pytest.raises(OnlineStateContractError, match="token differs"):
        state.commit_frame(query, token="0" * 64)
    assert state.live_track_count == 0

    commit = state.commit_frame(query, token=query.token)
    assert commit.live_track_count_after == 1
    assert commit.retired_before_current_views is True
    with pytest.raises(OnlineStateContractError, match="exact pending"):
        state.commit_frame(query, token=query.token)


def test_retirement_geometry_contains_only_past_views_and_precedes_current_write():
    state = Stream3Dv2OnlineState()
    _commit(state, 0, (_update("old", 0),))
    _commit(state, 1, (_update("old", 1),))

    query = state.query_frame(
        frame_id=50,
        frame_ordinal=2,
        updates=(_update("new", 2),),
        retire_track_ids=("old",),
    )
    assert state.live_track_ids == ("old",)
    assert query.maximum_accessed_frame_ordinal == 1
    assert len(query.retired) == 1
    retired = query.retired[0]
    assert retired.geometry.source_ids == ("old:0", "old:1")
    assert retired.geometry.decision_frame_ordinal == 1
    assert "new:2" not in retired.geometry.source_ids

    commit = state.commit_frame(query, token=query.token)
    assert commit.retired_before_current_views is True
    assert state.live_track_ids == ("new",)
    assert state.views_for_track("new")[0].frame_ordinal == 2


def test_frame_order_pending_duplicate_and_future_inputs_fail_closed():
    state = Stream3Dv2OnlineState()
    _commit(state, 2, (_update("track", 2),))

    for ordinal in (1, 2):
        with pytest.raises(OnlineStateContractError, match="strictly increasing"):
            state.query_frame(frame_id=100 + ordinal, frame_ordinal=ordinal)
    with pytest.raises(OnlineStateContractError, match="already committed"):
        state.query_frame(frame_id=50, frame_ordinal=3)
    future = TrackUpdate("future", _view("future:4", 4, frame_id=75))
    with pytest.raises(OnlineStateContractError, match="exactly match"):
        state.query_frame(frame_id=75, frame_ordinal=3, updates=(future,))
    assert state.statistics.future_access_count == 0
    assert state.live_track_ids == ("track",)

    pending = state.query_frame(frame_id=75, frame_ordinal=3)
    with pytest.raises(OnlineStateContractError, match="previous query"):
        state.query_frame(frame_id=100, frame_ordinal=4)
    state.commit_frame(pending, token=pending.token)


def test_duplicate_updates_sources_retirements_and_reuse_are_rejected():
    state = Stream3Dv2OnlineState()
    _commit(state, 0, (_update("a", 0, source_id="source"),))

    with pytest.raises(OnlineStateContractError, match="at most one"):
        state.query_frame(
            frame_id=25,
            frame_ordinal=1,
            updates=(_update("a", 1), _update("a", 1, source_id="other")),
        )
    with pytest.raises(OnlineStateContractError, match="duplicate source_id"):
        state.query_frame(
            frame_id=25,
            frame_ordinal=1,
            updates=(
                _update("b", 1, source_id="same"),
                _update("c", 1, source_id="same"),
            ),
        )
    with pytest.raises(OnlineStateContractError, match="already committed"):
        state.query_frame(
            frame_id=25,
            frame_ordinal=1,
            updates=(_update("b", 1, source_id="source"),),
        )
    with pytest.raises(OnlineStateContractError, match="duplicate retirement"):
        state.query_frame(
            frame_id=25,
            frame_ordinal=1,
            retire_track_ids=("a", "a"),
        )
    with pytest.raises(OnlineStateContractError, match="absent from past"):
        state.query_frame(
            frame_id=25,
            frame_ordinal=1,
            retire_track_ids=("not-yet-created",),
        )
    with pytest.raises(OnlineStateContractError, match="cannot retire"):
        state.query_frame(
            frame_id=25,
            frame_ordinal=1,
            updates=(_update("a", 1),),
            retire_track_ids=("a",),
        )
    assert state.live_view_count == 1


def test_each_track_retains_exactly_the_latest_five_views():
    state = Stream3Dv2OnlineState()
    for ordinal in range(8):
        _commit(state, ordinal, (_update("track", ordinal),))

    views = state.views_for_track("track")
    assert MAX_VIEWS_PER_TRACK == 5
    assert tuple(row.frame_ordinal for row in views) == (3, 4, 5, 6, 7)
    assert state.statistics.peak_live_track_count == 1
    assert state.statistics.peak_live_view_count == 5
    assert state.statistics.future_access_count == 0

    seal = state.finalize()
    assert seal.tracks[0].geometry.source_ids == tuple(f"track:{i}" for i in range(3, 8))


def test_track_capacity_allows_retire_then_replace_in_one_commit():
    state = Stream3Dv2OnlineState()
    first = tuple(_update(f"track-{index:04d}", 0) for index in range(MAX_LIVE_TRACKS))
    _commit(state, 0, first)
    assert state.live_track_count == MAX_LIVE_TRACKS

    with pytest.raises(OnlineStateContractError, match="capacity"):
        state.query_frame(
            frame_id=25,
            frame_ordinal=1,
            updates=(_update("overflow", 1),),
        )
    query = state.query_frame(
        frame_id=25,
        frame_ordinal=1,
        updates=(_update("replacement", 1),),
        retire_track_ids=("track-0000",),
    )
    commit = state.commit_frame(query, token=query.token)
    assert commit.live_track_count_after == MAX_LIVE_TRACKS
    assert "track-0000" not in state.live_track_ids
    assert "replacement" in state.live_track_ids
    assert state.statistics.peak_live_track_count == MAX_LIVE_TRACKS
    assert state.statistics.peak_live_view_count == MAX_LIVE_TRACKS


def test_finalize_materializes_all_active_tracks_is_idempotent_and_seals_state():
    state = Stream3Dv2OnlineState()
    _commit(state, 0, (_update("b", 0), _update("a", 0)))
    _commit(state, 1, (_update("a", 1),))

    seal = state.finalize()
    assert tuple(row.track_id for row in seal.tracks) == ("a", "b")
    assert seal.maximum_accessed_frame_ordinal == 1
    assert seal.statistics.terminal_finalized_track_count == 2
    assert seal.statistics.live_track_count == 0
    assert seal.statistics.live_view_count == 0
    assert seal.statistics.peak_live_track_count == 2
    assert seal.statistics.peak_live_view_count == 3
    assert seal.statistics.future_access_count == 0
    assert state.finalize() is seal
    with pytest.raises(OnlineStateContractError, match="already been finalized"):
        state.query_frame(frame_id=50, frame_ordinal=2)


def test_finalize_rejects_a_pending_query_without_changing_memory():
    state = Stream3Dv2OnlineState()
    _commit(state, 0, (_update("track", 0),))
    query = state.query_frame(frame_id=25, frame_ordinal=1)
    with pytest.raises(OnlineStateContractError, match="pending query"):
        state.finalize()
    assert state.live_track_count == 1
    state.commit_frame(query, token=query.token)
    assert len(state.finalize().tracks) == 1
