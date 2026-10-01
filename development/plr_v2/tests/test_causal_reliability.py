import numpy as np

from boxfusion.causal_reliability import CausalReliabilityState


def add(state, identity, frame, direction, strength=0.9, matched=True):
    state.update(
        identity,
        frame_id=frame,
        ordinal=frame,
        view_direction=direction,
        strength=strength if matched else 0.0,
        visibility=1.0,
        matched=matched,
    )


def test_directional_slots_do_not_count_repeated_nearby_views_as_independent():
    state = CausalReliabilityState(min_angular_separation_deg=30.0)
    add(state, 1, 0, [0, 0, 1], 0.7)
    add(state, 1, 1, [0.05, 0, 1], 0.9)
    summary = state.summary(1)
    assert summary["positive_views"] == 1
    assert summary["effective_views"] == 1
    assert summary["max"] == 0.9
    assert summary["first"] == 0.7
    assert summary["mean_support"] == 0.8


def test_diverse_positive_views_raise_bound_and_negative_view_lowers_it():
    positive = CausalReliabilityState(min_angular_separation_deg=20.0)
    mixed = CausalReliabilityState(min_angular_separation_deg=20.0)
    directions = ([1, 0, 1], [0, 0, 1], [-1, 0, 1])
    for frame, direction in enumerate(directions):
        add(positive, "box", frame, direction, 0.95)
        add(mixed, "box", frame, direction, 0.95)
    add(mixed, "box", 3, [0, 1, 1], matched=False)
    assert positive.summary("box")["positive_views"] == 3
    assert mixed.summary("box")["negative_views"] == 1
    assert mixed.summary("box")["lower"] < positive.summary("box")["lower"]


def test_reset_discard_ttl_and_caps_are_bounded():
    state = CausalReliabilityState(max_states=2, max_view_slots=2, state_ttl_keyframes=1)
    add(state, "a", 0, [1, 0, 1])
    add(state, "b", 1, [0, 0, 1])
    add(state, "c", 2, [-1, 0, 1])
    assert state.diagnostics()["states"] == 2
    assert state.diagnostics()["capacity_drops"] == 1

    state.reset_evidence("c", ordinal=2)
    assert state.summary("c")["positive_views"] == 0
    assert state.diagnostics()["geometry_resets"] == 1
    state.discard("c")
    assert "c" not in state.identities()
    state.prune(4)
    assert state.diagnostics()["states"] == 0

