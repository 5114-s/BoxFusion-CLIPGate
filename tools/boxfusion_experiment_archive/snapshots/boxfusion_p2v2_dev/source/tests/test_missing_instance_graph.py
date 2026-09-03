"""CPU-only contracts for the standalone missing-instance observer."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "boxfusion"
    / "missing_instance_graph.py"
)
SPEC = importlib.util.spec_from_file_location(
    "boxfusion_missing_instance_graph", SOURCE
)
missing_graph = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = missing_graph
SPEC.loader.exec_module(missing_graph)


def observer_config(**overrides):
    config = {
        "enabled": True,
        "minimum_component_pixels": 4,
        "minimum_component_points": 4,
        "maximum_components_per_proposal": 4,
        "component_max_depth_jump": 0.25,
        "component_max_world_distance": 0.20,
        "aabb_lower_quantile": 0.0,
        "aabb_upper_quantile": 1.0,
        "minimum_dimension": 0.02,
        "minimum_iou_3d": 0.01,
        "minimum_containment": 0.10,
        "minimum_projection_support": 0.01,
        "minimum_geometry_matches": 2,
        "maximum_center_distance": 0.75,
        "same_view_duplicate_iou": 0.85,
        "same_view_duplicate_containment": 0.95,
        "global_reject_iou": 0.30,
        "global_reject_containment": 0.70,
        "candidate_duplicate_iou": 0.35,
        "candidate_duplicate_containment": 0.70,
        "min_unique_frames": 2,
        "track_ttl_provider_calls": 3,
        "archive_confirmed": True,
        "max_nodes_per_track": 16,
        "max_edges_per_track": 32,
        "max_points_per_component": 256,
        "max_points_per_track": 512,
    }
    config.update(overrides)
    return missing_graph.resolve_missing_instance_graph_config(config)


def camera_inputs():
    intrinsics = np.asarray(
        [[100.0, 0.0, 20.0], [0.0, 100.0, 20.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return intrinsics, np.eye(4, dtype=np.float64)


def observation(
    frame_id,
    proposal_id,
    rectangles,
    *,
    depths=3.0,
    label="chair",
    score=0.90,
    provider="sam3",
    feature=None,
):
    mask = np.zeros((40, 40), dtype=np.float32)
    depth = np.full((40, 40), np.nan, dtype=np.float32)
    if np.isscalar(depths):
        depth_values = [float(depths)] * len(rectangles)
    else:
        depth_values = list(depths)
    for (row_start, row_stop, col_start, col_stop), value in zip(
        rectangles, depth_values
    ):
        mask[row_start:row_stop, col_start:col_stop] = 1.0
        depth[row_start:row_stop, col_start:col_stop] = value
    intrinsics, pose = camera_inputs()
    return missing_graph.MaskDepthProposalObservation(
        frame_id=frame_id,
        proposal_id=proposal_id,
        mask=mask,
        depth=depth,
        intrinsics=intrinsics,
        camera_to_world=pose,
        score=score,
        label=label,
        feature=feature,
        provider=provider,
    )


LEFT = ((17, 23, 8, 14),)
RIGHT = ((17, 23, 26, 32),)
OVERLAP_LEFT = ((17, 23, 10, 18),)
OVERLAP_RIGHT = ((17, 23, 13, 21),)


def candidate_signature(candidate):
    return (
        candidate.track_id,
        candidate.label,
        candidate.frame_ids,
        candidate.node_count,
        candidate.edge_count,
        tuple(np.round(candidate.oriented_box, 6)),
        tuple(np.round(candidate.feature_vector, 6)),
    )


def test_defaults_are_disabled_and_configuration_is_strict():
    config = missing_graph.resolve_missing_instance_graph_config()
    assert config["enabled"] is False
    assert config["fail_open"] is True
    assert config["min_unique_frames"] == 2
    assert config["archive_confirmed"] is True
    with pytest.raises(ValueError, match="Unknown"):
        missing_graph.resolve_missing_instance_graph_config(
            {"minimum_component_pointz": 4}
        )


@pytest.mark.parametrize(
    "override",
    [
        {"enabled": 1},
        {"fail_open": "yes"},
        {"component_connectivity": 6},
        {"minimum_component_points": 0},
        {"minimum_geometry_matches": 4},
        {"min_unique_frames": 1},
        {"max_nodes_per_track": 1},
        {"depth_scale": 0.0},
        {"min_depth": 2.0, "max_depth": 1.0},
        {"aabb_lower_quantile": 0.8, "aabb_upper_quantile": 0.2},
        {"minimum_iou_3d": np.nan},
        {"semantic_compatibility_groups": [["chair"]]},
        {
            "semantic_compatibility_groups": [
                ["chair", "seat"],
                ["seat", "stool"],
            ]
        },
    ],
)
def test_invalid_configuration_fails_fast(override):
    with pytest.raises(ValueError):
        missing_graph.resolve_missing_instance_graph_config(override)


def test_observation_copies_and_freezes_all_caller_arrays():
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[2:6, 2:6] = 1.0
    depth = np.full((8, 8), np.nan, dtype=np.float32)
    depth[2:6, 2:6] = 2.0
    intrinsics = np.asarray(
        [[50.0, 0.0, 4.0], [0.0, 50.0, 4.0], [0.0, 0.0, 1.0]]
    )
    pose = np.eye(4)
    feature = np.asarray([3.0, 4.0])
    item = missing_graph.MaskDepthProposalObservation(
        frame_id=0,
        proposal_id="p",
        mask=mask,
        depth=depth,
        intrinsics=intrinsics,
        camera_to_world=pose,
        score=0.8,
        label="Office_Chair",
        feature=feature,
        provider="YOLOE",
    )
    mask[:] = 0.0
    depth[:] = 9.0
    intrinsics[:] = 1.0
    pose[:] = 2.0
    feature[:] = 0.0
    assert np.count_nonzero(item.mask) == 16
    assert np.all(item.depth[2:6, 2:6] == 2.0)
    assert item.intrinsics[0, 0] == 50.0
    np.testing.assert_array_equal(item.camera_to_world, np.eye(4))
    np.testing.assert_allclose(item.feature, [0.6, 0.8])
    assert item.provider == "yoloe"
    for array in (
        item.mask,
        item.depth,
        item.intrinsics,
        item.camera_to_world,
        item.feature,
    ):
        assert array.flags.writeable is False


def test_depth_aware_lifting_separates_touching_depth_surfaces():
    # The 2D mask is connected, but the two halves are one metre apart in
    # camera Z.  The 3D depth jump must split them deterministically.
    item = observation(
        0,
        "split",
        ((16, 24, 12, 16), (16, 24, 16, 20)),
        depths=(2.0, 3.0),
    )
    components = missing_graph.lift_mask_depth_components(
        item, observer_config()
    )
    assert len(components) == 2
    assert [component.component_index for component in components] == [0, 1]
    assert [component.pixel_count for component in components] == [32, 32]
    assert components[0].component_id.endswith("component=0")
    assert components[1].component_id.endswith("component=1")
    assert components[0].points_world.flags.writeable is False
    assert components[0].component_mask.flags.writeable is False
    assert abs(
        float(components[0].aabb[2] - components[1].aabb[2])
    ) == pytest.approx(1.0)


def test_small_fragment_is_rejected_with_an_auditable_reason():
    main = (16, 22, 12, 18)
    speck = (2, 3, 2, 3)
    item = observation(0, "fragmented", (main, speck))
    observer = missing_graph.MissingInstanceGraph(observer_config())
    result = observer.update([item])
    assert observer.active_track_ids == (0,)
    assert len(result.candidates) == 0
    assert any(
        audit.reason == "component_too_small"
        and audit.pixel_count == 1
        and audit.accepted is False
        for audit in result.observations
    )
    assert any(
        audit.reason == "seeded" and audit.track_id == 0
        for audit in result.observations
    )


def test_single_view_never_emits_and_repeated_same_frame_cannot_confirm():
    observer = missing_graph.MissingInstanceGraph(observer_config())
    first = observer.update([observation(7, "a", LEFT)])
    assert first.candidates == ()
    assert first.decisions[0].reason == "insufficient_unique_views"

    # A repeated provider call for the same real frame creates no graph edge
    # to the existing track and therefore cannot create a two-view output.
    second = observer.update([observation(7, "b", LEFT)])
    assert second.candidates == ()
    assert all(
        decision.reason == "insufficient_unique_views"
        for decision in second.decisions
    )
    assert not second.associations


def test_two_views_emit_oriented_observer_record_and_fixed_features():
    observer = missing_graph.MissingInstanceGraph(observer_config())
    observer.update(
        [
            observation(
                0, "a", LEFT, feature=np.asarray([1.0, 0.0])
            )
        ]
    )
    result = observer.update(
        [
            observation(
                1, "b", LEFT, feature=np.asarray([2.0, 0.0])
            )
        ]
    )
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.schema == missing_graph.MISSING_INSTANCE_GRAPH_SCHEMA
    assert candidate.candidate_id == candidate.track_id
    assert candidate.observer_only is True
    assert candidate.valid is True
    assert candidate.verified is True
    assert candidate.confirmed is True
    assert candidate.reason == "confirmed_multiview"
    assert candidate.frame_ids == (0, 1)
    assert candidate.oriented_box.shape == (7,)
    assert candidate.corners.shape == (8, 3)
    assert candidate.aabb.shape == (6,)
    assert candidate.feature_vector.shape == (
        len(missing_graph.MISSING_GRAPH_FEATURE_NAMES),
    )
    assert len(missing_graph.MISSING_GRAPH_FEATURE_NAMES) == 20
    assert len(set(missing_graph.MISSING_GRAPH_FEATURE_NAMES)) == 20
    assert np.isfinite(candidate.feature_vector).all()
    assert np.all(candidate.feature_vector >= 0.0)
    assert np.all(candidate.feature_vector <= 1.0)
    assert candidate.feature_vector[0] == pytest.approx(1.0)
    assert candidate.feature_vector.flags.writeable is False
    assert candidate.oriented_box.flags.writeable is False
    assert candidate.corners.flags.writeable is False
    assert candidate.aabb.flags.writeable is False
    np.testing.assert_allclose(candidate.appearance_feature, [1.0, 0.0])
    payload = candidate.as_dict()
    assert payload["feature_names"] == (
        missing_graph.MISSING_GRAPH_FEATURE_NAMES
    )
    assert payload["candidate_id"] == candidate.track_id
    assert payload["valid"] is True
    assert payload["verified"] is True
    assert payload["confirmed"] is True
    payload["oriented_box"][:] = 99.0
    assert not np.all(candidate.oriented_box == 99.0)
    supplemental = candidate.as_supplemental_candidate()
    assert set(supplemental) == {
        "candidate_id",
        "source",
        "corners",
        "score",
        "label",
        "valid",
        "verified",
    }
    assert supplemental["candidate_id"] == candidate.track_id
    assert supplemental["source"] == "missing_graph"
    assert supplemental["valid"] is True
    assert supplemental["verified"] is True
    supplemental["corners"][:] = -99.0
    assert not np.all(candidate.corners == -99.0)


def test_geometry_is_hard_evidence_despite_matching_semantics_and_features():
    observer = missing_graph.MissingInstanceGraph(observer_config())
    observer.update(
        [
            observation(
                0, "left", LEFT, feature=np.asarray([1.0, 0.0])
            )
        ]
    )
    result = observer.update(
        [
            observation(
                1, "far", RIGHT, feature=np.asarray([1.0, 0.0])
            )
        ]
    )
    assert result.candidates == ()
    assert len(observer.active_track_ids) == 2
    edge = result.associations[0]
    assert edge.semantic_compatibility == pytest.approx(1.0)
    assert edge.appearance_cosine == pytest.approx(1.0)
    assert edge.accepted is False
    assert edge.reason == "geometry"


def test_adjacent_same_class_objects_remain_two_deterministic_tracks():
    observer = missing_graph.MissingInstanceGraph(observer_config())
    first = [
        observation(0, "right-0", RIGHT, score=0.88),
        observation(0, "left-0", LEFT, score=0.92),
    ]
    second = [
        observation(1, "left-1", LEFT, score=0.91),
        observation(1, "right-1", RIGHT, score=0.87),
    ]
    observer.update(first)
    result = observer.update(second)
    assert len(result.candidates) == 2
    assert [candidate.track_id for candidate in result.candidates] == [0, 1]
    centers = sorted(float(candidate.aabb[0]) for candidate in result.candidates)
    assert centers[1] - centers[0] > 0.3
    assert all(candidate.label == "chair" for candidate in result.candidates)
    assert sum(edge.accepted for edge in result.associations) == 2
    assert any(
        not edge.accepted and edge.reason == "geometry"
        for edge in result.associations
    )


def test_semantic_mismatch_cannot_associate_even_with_identical_geometry():
    observer = missing_graph.MissingInstanceGraph(observer_config())
    observer.update([observation(0, "chair", LEFT, label="chair")])
    result = observer.update(
        [observation(1, "table", LEFT, label="table")]
    )
    assert result.candidates == ()
    assert observer.active_track_ids == (0, 1)
    assert len(result.associations) == 1
    assert result.associations[0].reason == "semantic"
    assert result.associations[0].geometry_matches == 3


def test_configured_semantic_synonyms_can_associate_but_not_rescue_geometry():
    config = observer_config(
        semantic_compatibility_groups=(("sofa", "couch"),)
    )
    observer = missing_graph.MissingInstanceGraph(config)
    observer.update([observation(0, "sofa", LEFT, label="sofa")])
    result = observer.update(
        [observation(1, "couch", LEFT, label="couch")]
    )
    assert len(result.candidates) == 1
    assert result.associations[0].semantic_compatibility == 1.0

    far_observer = missing_graph.MissingInstanceGraph(config)
    far_observer.update([observation(0, "sofa", LEFT, label="sofa")])
    far = far_observer.update(
        [observation(1, "couch", RIGHT, label="couch")]
    )
    assert far.candidates == ()
    assert far.associations[0].reason == "geometry"


def test_fragmented_first_view_is_bridged_without_duplicate_output():
    left_fragment = (17, 23, 10, 15)
    right_fragment = (17, 23, 18, 23)
    bridge = (17, 23, 10, 23)
    observer = missing_graph.MissingInstanceGraph(
        observer_config(
            same_view_duplicate_iou=0.99,
            same_view_duplicate_containment=0.99,
        )
    )
    first = observer.update(
        [observation(0, "fragmented", (left_fragment, right_fragment))]
    )
    assert first.candidates == ()
    assert observer.active_track_ids == (0, 1)

    second = observer.update([observation(1, "bridge", (bridge,))])
    # The bridge can update only one track in this view.  It confirms that
    # track and leaves the other fragment pending; no single-view fragment is
    # emitted as a second object.
    assert len(second.candidates) == 1
    assert sum(
        decision.reason == "insufficient_unique_views"
        for decision in second.decisions
    ) == 1
    candidate = second.candidates[0]
    assert candidate.aabb[3] > 0.30
    assert candidate.features.unique_views == 2


def test_global_overlap_rejection_happens_before_track_creation_and_at_readout():
    item = observation(0, "global-duplicate", LEFT)
    component = missing_graph.lift_mask_depth_components(
        item, observer_config()
    )[0]
    global_box = component.aabb.copy()
    global_before = global_box.copy()
    observer = missing_graph.MissingInstanceGraph(observer_config())
    result = observer.update([item], global_boxes=global_box[None, :])
    assert result.candidates == ()
    assert observer.active_track_ids == ()
    assert any(
        audit.reason == "global_overlap" for audit in result.observations
    )
    np.testing.assert_array_equal(global_box, global_before)

    observer.update([observation(1, "a", RIGHT)])
    confirmed = observer.update([observation(2, "b", RIGHT)])
    assert len(confirmed.candidates) == 1
    candidate_box = confirmed.candidates[0].aabb.copy()
    assert observer.candidates(candidate_box[None, :]) == ()


def test_confirmed_candidate_duplicate_rejection_is_ranked_and_auditable():
    config = observer_config(
        same_view_duplicate_iou=0.99,
        same_view_duplicate_containment=0.99,
        candidate_duplicate_iou=0.20,
        candidate_duplicate_containment=0.60,
    )
    observer = missing_graph.MissingInstanceGraph(config)
    observer.update(
        [
            observation(0, "a0", OVERLAP_LEFT, score=0.95),
            observation(0, "b0", OVERLAP_RIGHT, score=0.80),
        ]
    )
    result = observer.update(
        [
            observation(1, "b1", OVERLAP_RIGHT, score=0.79),
            observation(1, "a1", OVERLAP_LEFT, score=0.94),
        ]
    )
    assert len(result.candidates) == 1
    assert result.candidates[0].score > 0.8
    duplicate_decisions = [
        decision
        for decision in result.decisions
        if decision.reason == "duplicate_candidate"
    ]
    assert len(duplicate_decisions) == 1
    assert duplicate_decisions[0].duplicate_of_track_id == (
        result.candidates[0].track_id
    )
    assert (
        duplicate_decisions[0].maximum_candidate_iou
        >= config["candidate_duplicate_iou"]
    )


def test_provider_call_ttl_discards_unconfirmed_and_archives_confirmed():
    config = observer_config(track_ttl_provider_calls=1)
    unconfirmed = missing_graph.MissingInstanceGraph(config)
    unconfirmed.update([observation(10, "seed", LEFT)])
    still_live = unconfirmed.update([])
    assert still_live.expired_track_ids == ()
    expired = unconfirmed.update([])
    assert expired.expired_track_ids == (0,)
    assert expired.discarded_track_ids == (0,)
    assert expired.archived_track_ids == ()
    assert unconfirmed.active_track_ids == ()

    confirmed = missing_graph.MissingInstanceGraph(config)
    confirmed.update([observation(100, "a", LEFT)])
    confirmed.update([observation(200, "b", LEFT)])
    confirmed.update([])
    archived = confirmed.update([])
    assert archived.expired_track_ids == (0,)
    assert archived.archived_track_ids == (0,)
    assert archived.discarded_track_ids == ()
    assert confirmed.active_track_ids == ()
    assert confirmed.archived_track_ids == (0,)
    assert len(archived.candidates) == 1
    assert archived.candidates[0].lifecycle_state == "archived"
    names = missing_graph.MISSING_GRAPH_FEATURE_NAMES
    active_index = names.index("lifecycle_active")
    assert archived.candidates[0].feature_vector[active_index] == 0.0


def test_provider_call_clock_is_explicit_and_strictly_monotonic():
    observer = missing_graph.MissingInstanceGraph(
        observer_config(track_ttl_provider_calls=2)
    )
    first = observer.update(
        [observation(0, "seed", LEFT)], provider_call_index=4
    )
    assert first.provider_call_index == 4
    second = observer.update([], provider_call_index=6)
    assert second.expired_track_ids == ()
    third = observer.update([], provider_call_index=7)
    assert third.expired_track_ids == (0,)
    with pytest.raises(ValueError, match="increase strictly"):
        observer.update([], provider_call_index=7)


def test_already_lifted_adapter_update_and_empty_call_api():
    raw_first = observation(0, "raw-0", LEFT)
    raw_second = observation(1, "raw-1", LEFT)
    first_component = missing_graph.lift_mask_depth_components(
        raw_first, observer_config()
    )[0]

    # Match the existing online LiftedProposal field layout and deliberately
    # omit raw depth.  AABB/mask projection remains hard geometry.
    second_points = missing_graph.lift_mask_depth_components(
        raw_second, observer_config()
    )[0].points_world
    lifted_like = types.SimpleNamespace(
        box=first_component.aabb.copy(),
        observation=types.SimpleNamespace(points_world=second_points),
        proposal=types.SimpleNamespace(
            mask=raw_second.mask,
            score=0.9,
            label="chair",
            feature=np.asarray([1.0, 0.0], dtype=np.float32),
        ),
        view=types.SimpleNamespace(
            frame_index=1,
            intrinsics=raw_second.intrinsics,
            camera_to_world=raw_second.camera_to_world,
        ),
    )
    adapted = missing_graph.coerce_lifted_mask_component(
        lifted_like,
        proposal_id="adapted-1",
        config=observer_config(),
    )
    assert adapted.frame_id == 1
    assert adapted.source_valid_depth_pixels == 0
    assert np.isnan(adapted.depth).all()

    observer = missing_graph.MissingInstanceGraph(observer_config())
    first = observer.update_lifted(
        [first_component], provider_call_index=3
    )
    assert first.candidates == ()
    second = observer.update_lifted(
        [adapted], provider_call_index=4
    )
    assert len(second.candidates) == 1
    empty = observer.advance_provider_call(provider_call_index=5)
    assert empty.provider_call_index == 5
    assert len(empty.candidates) == 1


def test_fail_open_skips_bad_observation_and_preserves_good_evidence():
    observer = missing_graph.MissingInstanceGraph(observer_config())
    result = observer.update(
        [
            {"frame_id": 0, "proposal_id": "bad", "score": 0.9},
            observation(0, "good", LEFT),
        ]
    )
    assert result.failed_open is True
    assert len(result.errors) == 1
    assert observer.active_track_ids == (0,)
    assert any(
        audit.reason == "invalid_observation"
        for audit in result.observations
    )
    assert any(
        audit.reason == "seeded" for audit in result.observations
    )

    confirmed = observer.update([observation(1, "good-2", LEFT)])
    assert len(confirmed.candidates) == 1


def test_unexpected_association_failure_is_transactional_and_fail_open(
    monkeypatch,
):
    observer = missing_graph.MissingInstanceGraph(observer_config())
    observer.update([observation(0, "seed", LEFT)])
    before = observer.summary()

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic association failure")

    monkeypatch.setattr(missing_graph, "_evaluate_association", fail)
    result = observer.update([observation(1, "current", LEFT)])
    assert result.failed_open is True
    assert result.candidates == ()
    assert "synthetic association failure" in result.errors[0]
    # The graph transaction rolled back, while the provider-call clock still
    # advanced as an empty observation step.
    assert observer.active_track_ids == (0,)
    assert observer.summary()["next_track_id"] == before["next_track_id"]
    assert observer.last_provider_call_index == 1


def test_fail_open_false_propagates_observation_errors():
    observer = missing_graph.MissingInstanceGraph(
        observer_config(fail_open=False)
    )
    with pytest.raises(ValueError):
        observer.update(
            [{"frame_id": 0, "proposal_id": "bad", "score": 0.9}]
        )
    assert observer.active_track_ids == ()
    assert observer.last_provider_call_index is None


def test_order_is_exactly_deterministic_under_proposal_permutations():
    calls = [
        [
            observation(0, "right-0", RIGHT, score=0.85),
            observation(0, "left-0", LEFT, score=0.95),
        ],
        [
            observation(1, "right-1", RIGHT, score=0.84),
            observation(1, "left-1", LEFT, score=0.94),
        ],
    ]
    forward = missing_graph.MissingInstanceGraph(observer_config())
    reverse = missing_graph.MissingInstanceGraph(observer_config())
    forward_result = None
    reverse_result = None
    for batch in calls:
        forward_result = forward.update(batch)
        reverse_result = reverse.update(list(reversed(batch)))
    assert forward_result is not None
    assert reverse_result is not None
    assert [
        candidate_signature(candidate)
        for candidate in forward_result.candidates
    ] == [
        candidate_signature(candidate)
        for candidate in reverse_result.candidates
    ]
    assert [
        (decision.track_id, decision.reason)
        for decision in forward_result.decisions
    ] == [
        (decision.track_id, decision.reason)
        for decision in reverse_result.decisions
    ]


def test_update_never_mutates_raw_observation_or_global_arrays():
    intrinsics, pose = camera_inputs()
    mask = np.zeros((40, 40), dtype=np.float32)
    mask[17:23, 8:14] = 1.0
    depth = np.full((40, 40), np.nan, dtype=np.float32)
    depth[17:23, 8:14] = 3.0
    feature = np.asarray([1.0, 2.0], dtype=np.float32)
    raw = {
        "frame_id": 0,
        "proposal_id": "raw",
        "mask": mask,
        "depth": depth,
        "intrinsics": intrinsics,
        "camera_to_world": pose,
        "score": 0.8,
        "label": "chair",
        "feature": feature,
        "provider": "sam3",
    }
    global_boxes = np.asarray(
        [[5.0, 5.0, 5.0, 1.0, 1.0, 1.0]], dtype=np.float32
    )
    snapshots = {
        name: np.array(value, copy=True)
        for name, value in raw.items()
        if isinstance(value, np.ndarray)
    }
    global_snapshot = global_boxes.copy()
    observer = missing_graph.MissingInstanceGraph(observer_config())
    observer.update([raw], global_boxes=global_boxes)
    for name, expected in snapshots.items():
        np.testing.assert_equal(raw[name], expected)
    np.testing.assert_equal(global_boxes, global_snapshot)


def test_oriented_box_fit_is_permutation_deterministic_and_canonical():
    angle = np.deg2rad(30.0)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    basis = np.asarray([[cosine, -sine], [sine, cosine]])
    local_xy = np.asarray(
        [
            [-1.0, -0.25],
            [-1.0, 0.25],
            [1.0, -0.25],
            [1.0, 0.25],
            [0.0, -0.25],
            [0.0, 0.25],
        ]
    )
    xy = local_xy @ basis.T
    points = np.column_stack((xy, np.linspace(0.0, 1.0, len(xy))))
    config = observer_config(
        aabb_lower_quantile=0.0,
        aabb_upper_quantile=1.0,
        minimum_orientation_anisotropy=0.01,
    )
    first = missing_graph.oriented_box_from_points(points, config)
    second = missing_graph.oriented_box_from_points(points[::-1], config)
    np.testing.assert_allclose(first[0], second[0], atol=1e-6)
    np.testing.assert_allclose(first[1], second[1], atol=1e-6)
    assert first[2] == pytest.approx(second[2])
    assert -np.pi / 2 <= float(first[0][6]) < np.pi / 2
    assert abs(float(first[0][6])) == pytest.approx(angle, abs=0.05)
    assert first[0].flags.writeable is False
    assert first[1].flags.writeable is False


def test_disabled_observer_is_a_true_noop():
    observer = missing_graph.MissingInstanceGraph()
    result = observer.update([object()])
    assert result.disabled is True
    assert result.candidates == ()
    assert result.observations == ()
    assert observer.active_track_ids == ()
    assert observer.last_provider_call_index is None
