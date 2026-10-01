from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import boxfusion.causal_dynamic_objects as dynamic_objects
from boxfusion.causal_dynamic_objects import (
    BoxObservation,
    CausalDynamicObjectMap,
    DynamicObjectConfig,
    DynamicObjectContractError,
    Lifecycle,
    MotionState,
    TrackVisibility,
    Visibility,
    resolve_causal_dynamic_object_config,
)


def _box(x: float, *, yaw: float = 0.0) -> np.ndarray:
    return np.asarray([x, 0.0, 1.0, 0.8, 0.7, 1.0, yaw], dtype=np.float64)


def _appearance(index: int, dimension: int = 8) -> np.ndarray:
    row = np.zeros(dimension, dtype=np.float64)
    row[index % dimension] = 1.0
    return row


def _observation(
    observation_id: str,
    x: float,
    *,
    score: float = 0.8,
    appearance: np.ndarray | None = None,
    view_direction: np.ndarray | None = None,
) -> BoxObservation:
    return BoxObservation(
        observation_id=observation_id,
        box7=_box(x),
        score=score,
        appearance=appearance,
        view_direction=view_direction,
    )


def _step(
    state: CausalDynamicObjectMap,
    frame: int,
    observations=(),
    visibilities=(),
):
    query = state.query_frame(
        frame_ordinal=frame,
        observations=tuple(observations),
        visibilities=tuple(visibilities),
    )
    return query, state.commit_frame(query, token=query.token)


def _config(**kwargs) -> DynamicObjectConfig:
    defaults = dict(
        enabled=True,
        min_confirmed_hits=1,
        max_center_distance_m=0.75,
        max_relocation_distance_m=5.0,
        null_track_cost=0.40,
        null_observation_cost=0.40,
    )
    defaults.update(kwargs)
    return DynamicObjectConfig(**defaults)


def _exhaustive_null_assignment(tracks, observations, frame, cfg):
    """Reference the original all-pairs implementation used before prefiltering."""

    track_count = len(tracks)
    observation_count = len(observations)
    pair_cost = np.full((track_count, observation_count), 1.0e6)
    metrics = {}
    for track_index, track in enumerate(tracks):
        for observation_index, observation in enumerate(observations):
            metric = dynamic_objects._pair_metric(track, observation, frame, cfg)
            if metric.valid:
                pair_cost[track_index, observation_index] = metric.cost
                metrics[(track_index, observation_index)] = metric
    dimension = track_count + observation_count
    augmented = np.full((dimension, dimension), 1.0e6)
    augmented[:track_count, :observation_count] = pair_cost
    for track_index in range(track_count):
        augmented[track_index, observation_count + track_index] = cfg.null_track_cost
    for observation_index in range(observation_count):
        augmented[track_count + observation_index, observation_index] = (
            cfg.null_observation_cost
        )
    augmented[track_count:, observation_count:] = 0.0
    assignment = dynamic_objects._hungarian_min_cost(augmented)
    matched = []
    for track_index in range(track_count):
        observation_index = int(assignment[track_index])
        if (track_index, observation_index) in metrics:
            matched.append(
                (track_index, observation_index, metrics[(track_index, observation_index)])
            )
    used_tracks = {row[0] for row in matched}
    used_observations = {row[1] for row in matched}
    return (
        matched,
        set(range(track_count)) - used_tracks,
        set(range(observation_count)) - used_observations,
    )


def test_config_is_disabled_by_default_resolves_section_and_enforces_bounds():
    assert resolve_causal_dynamic_object_config({}).enabled is False
    cfg = resolve_causal_dynamic_object_config(
        {"dynamic_objects": {"enabled": True, "history_size": 3}}
    )
    assert cfg.enabled is True
    assert cfg.history_size == 3
    with pytest.raises(DynamicObjectContractError, match="history_size"):
        DynamicObjectConfig(history_size=6)
    with pytest.raises(DynamicObjectContractError, match="max_appearance"):
        DynamicObjectConfig(max_appearance_prototypes=5)
    with pytest.raises(DynamicObjectContractError, match="max_unobserved"):
        DynamicObjectConfig(
            max_unobserved_coast_frames=31,
            max_reactivation_age=30,
        )
    with pytest.raises(DynamicObjectContractError, match="unknown"):
        resolve_causal_dynamic_object_config(
            {"dynamic_objects": {"not_a_real_option": 1}}
        )


def test_query_before_commit_is_read_only_exact_token_and_exposes_mapping():
    state = CausalDynamicObjectMap(_config())
    observation = _observation("frame0:init7", 0.0)
    query = state.query_frame(frame_ordinal=0, observations=(observation,))

    assert state.track_count == 0
    assert state.current_snapshots() == ()
    assert query.maximum_accessed_frame_ordinal == -1
    assert query.birth_observation_ids == ("frame0:init7",)
    assert query.claimed_observation_ids == ("frame0:init7",)
    assert query.observation_to_track == (("frame0:init7", "dyn-000000"),)

    with pytest.raises(DynamicObjectContractError, match="exact pending"):
        state.commit_frame(replace(query), token=query.token)
    with pytest.raises(DynamicObjectContractError, match="token differs"):
        state.commit_frame(query, token="0" * 64)
    assert state.track_count == 0

    commit = state.commit_frame(query, token=query.token)
    assert commit.query_before_commit is True
    assert commit.observation_to_track == query.observation_to_track
    assert state.observation_track_map() == {"frame0:init7": "dyn-000000"}
    assert state.recent_observation_ids("dyn-000000") == ("frame0:init7",)
    with pytest.raises(DynamicObjectContractError, match="exact pending"):
        state.commit_frame(query, token=query.token)


def test_null_aware_assignment_is_one_to_one_and_bad_match_becomes_birth():
    state = CausalDynamicObjectMap(_config(max_center_distance_m=0.5))
    _step(
        state,
        0,
        (
            _observation("a0", 0.0, appearance=_appearance(0)),
            _observation("b0", 3.0, appearance=_appearance(1)),
        ),
    )
    query, _ = _step(
        state,
        1,
        (
            _observation("near-a", 0.05, appearance=_appearance(0)),
            _observation("far-new", 8.0, appearance=_appearance(2)),
        ),
    )

    assert len(query.associations) == 1
    assert query.associations[0].observation_id == "near-a"
    assert query.associations[0].track_id == "dyn-000000"
    assert query.birth_observation_ids == ("far-new",)
    assert dict(query.observation_to_track)["far-new"] == "dyn-000002"
    assert len(set(dict(query.observation_to_track).values())) == 2


def test_constant_velocity_prediction_and_dynamic_short_window_are_bounded():
    cfg = _config(
        history_size=5,
        dynamic_speed_low_m_per_frame=0.01,
        dynamic_speed_high_m_per_frame=0.04,
        dynamic_probability_update_rate=0.70,
        velocity_update_rate=1.0,
    )
    state = CausalDynamicObjectMap(cfg)
    for frame in range(7):
        query, _ = _step(
            state,
            frame,
            (_observation(f"moving:{frame}", 0.10 * frame),),
        )
        if frame:
            assert query.birth_observation_ids == ()

    snapshot = state.track_snapshot("dyn-000000")
    assert snapshot.motion_state == MotionState.DYNAMIC
    assert snapshot.dynamic_probability >= cfg.dynamic_probability_threshold
    assert snapshot.velocity4[0] > 0.07
    assert snapshot.recent_observation_ids == tuple(
        f"moving:{frame}" for frame in range(2, 7)
    )
    assert len(snapshot.recent_observation_ids) == 5

    # The next query is matched against the constant-velocity prediction even
    # though the measurement lies beyond the unpredicted center gate.
    query = state.query_frame(
        frame_ordinal=8,
        observations=(_observation("moving:8", 0.80),),
    )
    assert query.birth_observation_ids == ()
    assert query.associations[0].track_id == "dyn-000000"
    state.commit_frame(query, token=query.token)

    observed_x = state.current_snapshots()[0].box7[0]
    _step(
        state,
        9,
        visibilities=(
            TrackVisibility("dyn-000000", Visibility.OCCLUDED),
        ),
    )
    predicted = state.current_snapshots()[0]
    assert predicted.current_score == state.track_snapshot("dyn-000000").current_score
    assert predicted.box7[0] > observed_x


def test_appearance_memory_is_bounded_to_four_view_diverse_prototypes():
    state = CausalDynamicObjectMap(
        _config(max_appearance_prototypes=4, prototype_merge_cosine=0.999)
    )
    directions = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    )
    for frame, direction in enumerate(directions):
        _step(
            state,
            frame,
            (
                _observation(
                    f"view:{frame}",
                    0.0,
                    appearance=_appearance(frame),
                    view_direction=np.asarray(direction),
                ),
            ),
        )
    snapshot = state.track_snapshot("dyn-000000")
    assert snapshot.appearance_prototype_count == 4
    assert len(snapshot.recent_observation_ids) == 5


def test_occlusion_does_not_decay_but_visible_absence_dormants_and_retires():
    state = CausalDynamicObjectMap(
        _config(
            dormant_after_visible_misses=2,
            retire_after_visible_misses=3,
            visible_miss_score_decay=0.5,
        )
    )
    _step(state, 0, (_observation("object:0", 0.0, score=0.8),))
    before = state.track_snapshot("dyn-000000")
    _step(
        state,
        1,
        visibilities=(
            TrackVisibility("dyn-000000", Visibility.OCCLUDED),
        ),
    )
    occluded = state.track_snapshot("dyn-000000")
    assert occluded.lifecycle == Lifecycle.OCCLUDED
    assert occluded.current_score == before.current_score
    assert occluded.persistent_score == before.persistent_score
    assert len(state.current_snapshots()) == 1

    _step(
        state,
        2,
        visibilities=(
            TrackVisibility("dyn-000000", Visibility.EXPECTED_VISIBLE),
        ),
    )
    _step(
        state,
        3,
        visibilities=(
            TrackVisibility("dyn-000000", Visibility.EXPECTED_VISIBLE),
        ),
    )
    dormant = state.track_snapshot("dyn-000000")
    assert dormant.lifecycle == Lifecycle.DORMANT
    assert state.current_snapshots() == ()
    assert len(state.persistent_snapshots()) == 1
    assert state.persistent_snapshots()[0].score == before.persistent_score

    _step(
        state,
        4,
        visibilities=(
            TrackVisibility("dyn-000000", Visibility.EXPECTED_VISIBLE),
        ),
    )
    assert state.track_snapshot("dyn-000000").lifecycle == Lifecycle.RETIRED
    assert state.current_snapshots() == ()
    assert len(state.persistent_snapshots()) == 1


@pytest.mark.parametrize("visibility", list(Visibility))
def test_miss_ablation_preserves_confirmed_state_beyond_retirement_age(visibility):
    state = CausalDynamicObjectMap(_config(miss_lifecycle_updates=False))
    _step(state, 0, (_observation("object:0", 0.0, score=0.8),))
    before = state.track_snapshot("dyn-000000")
    for frame in range(1, 35):
        _step(state, frame, visibilities=(TrackVisibility("dyn-000000", visibility),))
    after = state.track_snapshot("dyn-000000")
    assert after.lifecycle == Lifecycle.CONFIRMED
    assert after.current_score == before.current_score
    assert after.consecutive_visible_misses == 0
    assert after.recent_observation_ids == before.recent_observation_ids
    assert len(state.current_snapshots()) == 1
    assert state.track_count == 1


def test_miss_ablation_switch_defaults_on_and_rejects_non_boolean():
    assert DynamicObjectConfig().miss_lifecycle_updates is True
    with pytest.raises(DynamicObjectContractError, match="miss_lifecycle_updates"):
        DynamicObjectConfig(miss_lifecycle_updates="false")


def test_strong_appearance_relocates_and_reactivates_same_track_id():
    cfg = _config(
        max_center_distance_m=0.4,
        max_relocation_distance_m=5.0,
        min_reactivation_appearance=0.70,
        dormant_after_visible_misses=1,
        retire_after_visible_misses=2,
    )
    state = CausalDynamicObjectMap(cfg)
    feature = np.asarray([0.7, 0.2, 0.1, 0.0])
    _step(state, 0, (_observation("old:0", 0.0, appearance=feature),))
    for frame in (1, 2):
        _step(
            state,
            frame,
            visibilities=(
                TrackVisibility("dyn-000000", Visibility.EXPECTED_VISIBLE),
            ),
        )
    assert state.track_snapshot("dyn-000000").lifecycle == Lifecycle.RETIRED

    query, commit = _step(
        state,
        3,
        (_observation("new:3", 2.0, appearance=feature),),
    )
    assert query.birth_observation_ids == ()
    assert query.reactivated_track_ids == ("dyn-000000",)
    assert query.relocated_track_ids == ("dyn-000000",)
    assert query.observation_to_track == (("new:3", "dyn-000000"),)
    assert commit.reactivated_track_ids == ("dyn-000000",)
    assert state.track_count == 1
    assert state.track_snapshot("dyn-000000").lifecycle == Lifecycle.CONFIRMED
    assert len(state.current_snapshots()) == 1
    # The historical persistent geometry stays at the old location for a
    # dynamic relocation while the current geometry follows the new location.
    persistent = state.persistent_snapshots()[0]
    current = state.current_snapshots()[0]
    assert persistent.box7[0] < 0.5
    assert current.box7[0] > 1.0


def test_two_runs_are_deterministic_and_output_arrays_are_immutable():
    def run_once():
        state = CausalDynamicObjectMap(_config())
        records = []
        for frame in range(3):
            query, commit = _step(
                state,
                frame,
                (
                    _observation(f"a:{frame}", 0.05 * frame),
                    _observation(f"b:{frame}", 2.0 + 0.03 * frame),
                ),
            )
            records.append(
                (
                    query.token,
                    query.observation_to_track,
                    commit.current_count,
                )
            )
        return records, state

    first, first_state = run_once()
    second, _ = run_once()
    assert first == second
    snapshot = first_state.current_snapshots()[0]
    with pytest.raises(ValueError):
        snapshot.box7[0] = 100.0
    with pytest.raises(ValueError):
        snapshot.velocity4[0] = 100.0


def test_full_capacity_reclaims_tentative_track_before_dropping_birth():
    state = CausalDynamicObjectMap(
        _config(max_tracks=2, min_confirmed_hits=2, max_center_distance_m=0.25)
    )
    _step(
        state,
        0,
        (
            _observation("weak-old", 0.0, score=0.20),
            _observation("strong-old", 2.0, score=0.80),
        ),
    )
    assert state.track_count == 2
    assert all(
        row.lifecycle == Lifecycle.TENTATIVE
        for row in state.current_snapshots(include_tentative=True)
    )

    query = state.query_frame(
        frame_ordinal=1,
        observations=(_observation("new-birth", 10.0, score=0.70),),
    )
    # Capacity is reclaimed using the stable lowest-score/oldest/ID order;
    # the observation is accepted instead of being rejected forever.
    assert query.evicted_track_ids == ("dyn-000000",)
    assert query.capacity_dropped_observation_ids == ()
    assert query.birth_observation_ids == ("new-birth",)
    assert query.observation_to_track == (("new-birth", "dyn-000002"),)
    commit = state.commit_frame(query, token=query.token)
    assert commit.evicted_track_ids == ("dyn-000000",)
    assert state.track_count == 2
    assert state.observation_track_map() == {
        "strong-old": "dyn-000001",
        "new-birth": "dyn-000002",
    }


def test_capacity_never_evicts_a_track_matched_in_the_current_frame():
    state = CausalDynamicObjectMap(
        _config(max_tracks=2, min_confirmed_hits=3, max_center_distance_m=0.25)
    )
    _step(
        state,
        0,
        (
            _observation("match-me", 0.0, score=0.10),
            _observation("spare", 2.0, score=0.90),
        ),
    )
    query, commit = _step(
        state,
        1,
        (
            _observation("matched-now", 0.02, score=0.10),
            _observation("new-now", 10.0, score=0.70),
        ),
    )
    assert query.associations[0].track_id == "dyn-000000"
    assert query.evicted_track_ids == ("dyn-000001",)
    assert dict(commit.observation_to_track) == {
        "matched-now": "dyn-000000",
        "new-now": "dyn-000002",
    }
    assert set(row.track_id for row in state.current_snapshots(include_tentative=True)) == {
        "dyn-000000",
        "dyn-000002",
    }


def test_vectorized_prefilter_is_equivalent_to_exhaustive_all_pair_semantics():
    rng = np.random.default_rng(7319)
    cfg = _config(
        max_center_distance_m=0.65,
        max_relocation_distance_m=3.0,
        min_reactivation_appearance=0.55,
        max_reactivation_age=8,
    )
    for trial in range(100):
        track_count = int(rng.integers(1, 5))
        observation_count = int(rng.integers(1, 5))
        tracks = []
        for index in range(track_count):
            dimension = 3 if (trial + index) % 3 else 4
            appearance = rng.normal(size=dimension)
            source = BoxObservation(
                observation_id=f"source:{trial}:{index}",
                box7=np.concatenate(
                    (
                        rng.uniform(-1.5, 1.5, size=3),
                        rng.uniform(0.25, 1.4, size=3),
                        rng.uniform(-np.pi, np.pi, size=1),
                    )
                ),
                score=float(rng.uniform(0.1, 0.95)),
                appearance=appearance,
            )
            track = dynamic_objects._new_track(
                f"track:{index}", source, 0, cfg
            )
            if (trial + index) % 2 == 0:
                track = replace(
                    track,
                    lifecycle=Lifecycle.RETIRED,
                    ever_confirmed=True,
                )
            tracks.append(track)

        observations = []
        for index in range(observation_count):
            appearance = None
            if (trial + index) % 4:
                dimension = 3 if (trial + index) % 3 else 4
                appearance = rng.normal(size=dimension)
            observations.append(
                BoxObservation(
                    observation_id=f"query:{trial}:{index}",
                    box7=np.concatenate(
                        (
                            rng.uniform(-2.5, 2.5, size=3),
                            rng.uniform(0.25, 1.4, size=3),
                            rng.uniform(-np.pi, np.pi, size=1),
                        )
                    ),
                    score=float(rng.uniform(0.1, 0.95)),
                    appearance=appearance,
                )
            )

        expected = _exhaustive_null_assignment(tracks, observations, 3, cfg)
        actual = dynamic_objects._null_aware_assignment(
            tracks, observations, 3, cfg
        )
        expected_pairs = {(row[0], row[1]): row[2] for row in expected[0]}
        actual_pairs = {(row[0], row[1]): row[2] for row in actual[0]}
        assert actual_pairs.keys() == expected_pairs.keys()
        for pair in actual_pairs:
            assert actual_pairs[pair].valid is True
            assert actual_pairs[pair].cost == pytest.approx(expected_pairs[pair].cost)
            assert actual_pairs[pair].center_distance_m == pytest.approx(
                expected_pairs[pair].center_distance_m
            )
        assert actual[1:] == expected[1:]


def test_crowded_top_k_keeps_appearance_identity_edges_and_bounds_exact_pairs(
    monkeypatch,
):
    """Exercise the K=4 pruning path on a dense crossing configuration."""

    count = 32
    shift = 8
    top_k = 4
    radius = 0.20
    cfg = _config(
        max_tracks=count,
        max_observations_per_frame=count,
        association_candidate_top_k=top_k,
        max_center_distance_m=1.20,
    )

    def circle_box(index: int) -> np.ndarray:
        angle = 2.0 * np.pi * index / count
        return np.asarray(
            [
                radius * np.cos(angle),
                radius * np.sin(angle),
                0.8,
                0.55,
                0.55,
                0.75,
                0.0,
            ],
            dtype=np.float64,
        )

    tracks = []
    observations = []
    for index in range(count):
        identity = np.eye(count, dtype=np.float64)[index]
        source = BoxObservation(
            observation_id=f"source:{index}",
            box7=circle_box(index),
            score=0.8,
            appearance=identity,
        )
        tracks.append(dynamic_objects._new_track(f"track:{index}", source, 0, cfg))
        observations.append(
            BoxObservation(
                observation_id=f"query:{index}",
                box7=circle_box((index + shift) % count),
                score=0.8,
                appearance=identity,
            )
        )

    # The exhaustive optimum follows appearance identity rather than the
    # geometrically nearest (but wrong) object after the cyclic displacement.
    expected = _exhaustive_null_assignment(tracks, observations, 1, cfg)
    assert {(row[0], row[1]) for row in expected[0]} == {
        (index, index) for index in range(count)
    }

    original_pair_metric = dynamic_objects._pair_metric
    exact_pair_calls = 0

    def counted_pair_metric(*args, **kwargs):
        nonlocal exact_pair_calls
        exact_pair_calls += 1
        return original_pair_metric(*args, **kwargs)

    monkeypatch.setattr(dynamic_objects, "_pair_metric", counted_pair_metric)
    actual = dynamic_objects._null_aware_assignment(
        tracks, observations, 1, cfg
    )
    assert {(row[0], row[1]) for row in actual[0]} == {
        (index, index) for index in range(count)
    }
    assert actual[1:] == expected[1:]

    raw_pair_count = count * count
    directed_union_bound = 2 * top_k * (count + count)
    assert exact_pair_calls < raw_pair_count
    assert exact_pair_calls <= directed_union_bound


def test_vectorized_appearance_preselection_matches_scalar_similarity():
    cfg = _config(prototype_merge_cosine=0.99)
    first = BoxObservation(
        observation_id="source:3d",
        box7=_box(0.0),
        score=0.8,
        appearance=np.asarray([1.0, 0.0, 0.0]),
        view_direction=np.asarray([1.0, 0.0, 0.0]),
    )
    track_3d = dynamic_objects._new_track("track:3d", first, 0, cfg)
    second = BoxObservation(
        observation_id="source:3d:second",
        box7=_box(0.0),
        score=0.8,
        appearance=np.asarray([0.0, 1.0, 0.0]),
        view_direction=np.asarray([0.0, 1.0, 0.0]),
    )
    track_3d = replace(
        track_3d,
        prototypes=dynamic_objects._add_prototype(
            track_3d.prototypes, second, 1, cfg
        ),
    )
    source_4d = BoxObservation(
        observation_id="source:4d",
        box7=_box(1.0),
        score=0.8,
        appearance=np.asarray([0.0, 0.0, 1.0, 0.0]),
    )
    track_4d = dynamic_objects._new_track("track:4d", source_4d, 0, cfg)
    observations = (
        BoxObservation(
            observation_id="query:3d",
            box7=_box(0.0),
            score=0.8,
            appearance=np.asarray([0.6, 0.8, 0.0]),
            view_direction=np.asarray([-1.0, 0.0, 0.0]),
        ),
        BoxObservation(
            observation_id="query:4d",
            box7=_box(1.0),
            score=0.8,
            appearance=np.asarray([0.0, 0.0, 1.0, 0.0]),
        ),
        _observation("query:none", 2.0),
    )
    tracks = (track_3d, track_4d)
    actual = dynamic_objects._coarse_appearance_similarity(tracks, observations)
    for track_index, track in enumerate(tracks):
        for observation_index, observation in enumerate(observations):
            expected = dynamic_objects._prototype_similarity(track, observation)
            if expected is None:
                assert np.isnan(actual[track_index, observation_index])
            else:
                assert actual[track_index, observation_index] == pytest.approx(expected)


def test_long_unobserved_track_stops_coasting_then_dormants_and_retires():
    cfg = _config(
        max_unobserved_coast_frames=2,
        max_reactivation_age=4,
        dynamic_probability_threshold=0.40,
        static_probability_threshold=0.20,
    )
    state = CausalDynamicObjectMap(cfg)
    _step(state, 0, (_observation("moving:0", 0.0, score=0.8),))
    _step(state, 1, (_observation("moving:1", 0.20, score=0.8),))
    observed = state.track_snapshot("dyn-000000")
    assert observed.motion_state == MotionState.DYNAMIC

    internal = state._tracks["dyn-000000"]
    far_prediction = dynamic_objects._predict_box(internal, 100, cfg)
    expected_x = (
        internal.current_box7[0]
        + internal.velocity4[0]
        * internal.dynamic_probability
        * cfg.max_unobserved_coast_frames
    )
    assert far_prediction[0] == pytest.approx(expected_x)

    _step(
        state,
        2,
        visibilities=(
            TrackVisibility("dyn-000000", Visibility.OUT_OF_VIEW),
        ),
    )
    assert state.track_snapshot("dyn-000000").lifecycle == Lifecycle.CONFIRMED

    _step(
        state,
        3,
        visibilities=(
            TrackVisibility("dyn-000000", Visibility.UNKNOWN),
        ),
    )
    dormant = state.track_snapshot("dyn-000000")
    assert dormant.lifecycle == Lifecycle.DORMANT
    assert dormant.current_score == observed.current_score
    assert state.current_snapshots() == ()

    for frame, visibility in (
        (4, Visibility.OUT_OF_VIEW),
        (5, Visibility.UNKNOWN),
        (6, Visibility.OUT_OF_VIEW),
    ):
        _step(
            state,
            frame,
            visibilities=(TrackVisibility("dyn-000000", visibility),),
        )
    retired = state.track_snapshot("dyn-000000")
    assert retired.lifecycle == Lifecycle.RETIRED
    assert retired.current_score == observed.current_score
    assert len(state.persistent_snapshots()) == 1
