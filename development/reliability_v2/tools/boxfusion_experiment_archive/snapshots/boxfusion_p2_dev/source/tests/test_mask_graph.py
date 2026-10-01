import importlib.util
import itertools
import sys
import types
from pathlib import Path

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "boxfusion" / "mask_graph.py"
SPEC = importlib.util.spec_from_file_location("boxfusion_mask_graph", SOURCE)
mask_graph = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mask_graph
SPEC.loader.exec_module(mask_graph)


def graph_config(**overrides):
    config = {
        "enabled": True,
        "max_nodes": 8,
        "max_edges": 16,
        "min_unique_frames": 2,
    }
    config.update(overrides)
    return mask_graph.resolve_mask_graph_config(config)


def cube_points(center=(0.0, 0.0, 3.0), dims=(1.0, 1.0, 1.0)):
    center = np.asarray(center, dtype=np.float32)
    half = np.asarray(dims, dtype=np.float32) * 0.5
    signs = np.asarray(
        list(itertools.product((-1.0, 1.0), repeat=3)),
        dtype=np.float32,
    )
    return center[None, :] + signs * half[None, :]


def projection_inputs(mask=None):
    if mask is None:
        mask = np.zeros((40, 40), dtype=np.bool_)
        mask[16:24, 16:24] = True
    intrinsics = np.asarray(
        [[20.0, 0.0, 20.0], [0.0, 20.0, 20.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return mask, intrinsics, np.eye(4, dtype=np.float32)


def node(
    node_id,
    frame_id,
    *,
    center=(0.0, 0.0, 3.0),
    dims=(1.0, 1.0, 1.0),
    feature=(1.0, 0.0),
    label="chair",
    mask=None,
    with_projection=True,
):
    kwargs = {}
    if with_projection:
        mask_value, intrinsics, pose = projection_inputs(mask)
        kwargs.update(
            mask=mask_value,
            intrinsics=intrinsics,
            camera_to_world=pose,
        )
    return mask_graph.MaskGraphNode(
        node_id=node_id,
        frame_id=frame_id,
        center=np.asarray(center),
        dims=np.asarray(dims),
        points_world=cube_points(center, dims),
        appearance_feature=(
            None if feature is None else np.asarray(feature, dtype=np.float32)
        ),
        label=label,
        confidence=0.8,
        **kwargs,
    )


def track(
    *,
    center=(0.0, 0.0, 3.0),
    dims=(1.0, 1.0, 1.0),
    feature=(1.0, 0.0),
    label="chair",
):
    return mask_graph.MaskGraphTrackEvidence(
        center=np.asarray(center),
        dims=np.asarray(dims),
        points_world=cube_points(center, dims),
        appearance_feature=(
            None if feature is None else np.asarray(feature, dtype=np.float32)
        ),
        label=label,
    )


def edge(source, target, *, score=0.8, sample_count=1):
    return mask_graph.MaskGraphEdge(
        source_id=source.node_id,
        target_id=target.node_id,
        source_frame_id=source.frame_id,
        target_frame_id=target.frame_id,
        accepted=True,
        reason="accepted",
        score=score,
        geometry_score=score,
        iou_3d=score,
        observation_inside_track=score,
        track_inside_observation=score,
        mutual_inside=score,
        projection_iou=score,
        appearance_cosine=score,
        appearance_compatibility=score,
        label_compatibility=score,
        geometry_matches=3,
        sample_count=sample_count,
    )


def test_default_is_disabled_and_config_rejects_unknown_keys():
    config = mask_graph.resolve_mask_graph_config()
    assert config["enabled"] is False
    assert config["require_projection"] is True
    assert config["min_unique_frames"] == 2
    with pytest.raises(ValueError, match="Unknown"):
        mask_graph.resolve_mask_graph_config({"max_nodez": 3})


@pytest.mark.parametrize(
    "override",
    [
        {"enabled": 1},
        {"require_projection": "yes"},
        {"max_nodes": 1},
        {"max_edges": 0},
        {"min_unique_frames": 1},
        {"minimum_geometry_matches": 4},
        {"minimum_edge_score": -0.1},
        {"minimum_iou_3d": 1.1},
        {"appearance_neutral_cosine": -1.1},
        {"near_clip": 0.0},
        {
            "iou_3d_weight": 0.0,
            "mutual_inside_weight": 0.0,
            "projection_iou_weight": 0.0,
        },
        {"label_compatibility_groups": ["chair", "seat"]},
        {"label_compatibility_groups": [["chair"]]},
        {
            "label_compatibility_groups": [
                ["chair", "seat"],
                ["seat", "stool"],
            ]
        },
        {"minimum_edge_score": np.nan},
    ],
)
def test_invalid_config_fails_fast(override):
    with pytest.raises(ValueError):
        mask_graph.resolve_mask_graph_config(override)


def test_label_groups_are_normalized_and_immutable_tuples():
    config = mask_graph.resolve_mask_graph_config(
        {"label_compatibility_groups": [["Office_Chair", "desk-chair"]]}
    )
    assert config["label_compatibility_groups"] == (
        ("office chair", "desk chair"),
    )


def test_node_normalizes_features_and_freezes_copied_arrays():
    center = np.asarray([0.0, 0.0, 3.0], dtype=np.float64)
    item = node("a", 1, feature=(3.0, 4.0))
    center[:] = 10.0
    np.testing.assert_allclose(item.center, [0.0, 0.0, 3.0])
    np.testing.assert_allclose(item.appearance_feature, [0.6, 0.8])
    assert not item.center.flags.writeable
    assert not item.points_world.flags.writeable
    assert not item.mask.flags.writeable
    with pytest.raises(ValueError, match="all be present"):
        mask_graph.MaskGraphNode(
            node_id=0,
            frame_id=0,
            center=[0.0, 0.0, 3.0],
            dims=[1.0, 1.0, 1.0],
            mask=np.ones((2, 2)),
        )
    with pytest.raises(ValueError, match="non-zero"):
        node("zero", 0, feature=(0.0, 0.0))


def test_state_counts_unique_frames_and_latches_confirmation():
    graph = mask_graph.MaskGraphState(
        track_id=7,
        config=graph_config(max_nodes=2),
    )
    assert graph.add_node(node("a", 3, with_projection=False)) is False
    assert graph.add_node(node("b", 3, with_projection=False)) is False
    assert graph.unique_frame_count == 1
    assert graph.confirmed is False
    assert graph.add_node(node("c", 4, with_projection=False)) is True
    assert graph.unique_frame_ids == (3, 4)
    assert graph.confirmed is True
    assert graph.confirmation_frame_id == 4
    # Both confirming frames may later leave bounded memory; confirmation is
    # intentionally irreversible.
    graph.add_node(node("d", 5, with_projection=False))
    graph.add_node(node("e", 5, with_projection=False))
    assert graph.unique_frame_ids == (5,)
    assert graph.confirmed is True


def test_state_persists_only_compact_node_metadata():
    graph = mask_graph.MaskGraphState(config=graph_config())
    incoming = node("large", 7)
    assert incoming.points_world is not None
    assert incoming.mask is not None
    assert incoming.intrinsics is not None
    assert incoming.camera_to_world is not None

    graph.add_node(incoming)
    persisted = graph.nodes["large"]
    assert persisted is not incoming
    assert persisted.points_world is None
    assert persisted.mask is None
    assert persisted.intrinsics is None
    assert persisted.camera_to_world is None
    np.testing.assert_array_equal(persisted.box, incoming.box)
    np.testing.assert_array_equal(
        persisted.appearance_feature,
        incoming.appearance_feature,
    )
    assert persisted.frame_id == incoming.frame_id
    assert persisted.label == incoming.label
    assert persisted.confidence == incoming.confidence


def test_state_bounds_nodes_edges_and_accumulates_repeated_edge_samples():
    graph = mask_graph.MaskGraphState(
        config=graph_config(max_nodes=3, max_edges=2),
    )
    a = node("a", 1, with_projection=False)
    b = node("b", 2, with_projection=False)
    c = node("c", 3, with_projection=False)
    d = node("d", 4, with_projection=False)
    for item in (a, b, c):
        graph.add_node(item)

    graph.add_edge(edge(a, b, score=0.4))
    graph.add_edge(edge(a, b, score=0.8, sample_count=2))
    accumulated = graph.edges[("a", "b")]
    assert accumulated.sample_count == 3
    assert accumulated.score == pytest.approx((0.4 + 1.6) / 3.0)
    assert accumulated.unique_frame_ids == (1, 2)

    graph.add_edge(edge(b, c, score=0.7))
    graph.add_edge(edge(a, c, score=0.6))
    assert graph.edge_count == 2
    assert ("a", "b") not in graph.edges
    graph.add_node(d)
    assert tuple(graph.nodes) == ("b", "c", "d")
    assert all("a" not in key for key in graph.edges)


def test_identical_cross_view_observation_has_all_required_metrics():
    graph = mask_graph.MaskGraphState(config=graph_config())
    graph.add_node(node("seed", 1))
    result = mask_graph.evaluate_edge(
        track(),
        graph,
        node("current", 2),
    )
    assert result.accepted is True
    assert result.reason == "accepted"
    assert result.iou_3d == pytest.approx(1.0)
    assert result.mutual_inside == pytest.approx(1.0)
    assert result.projection_iou == pytest.approx(1.0)
    assert result.appearance_cosine == pytest.approx(1.0)
    assert result.label_compatibility == pytest.approx(1.0)
    assert result.geometry_matches == 3
    assert result.score == pytest.approx(1.0)


def test_track_projection_to_current_mask_is_a_required_geometry_gate():
    graph = mask_graph.MaskGraphState(config=graph_config())
    graph.add_node(node("seed", 1))
    disjoint = np.zeros((40, 40), dtype=np.bool_)
    disjoint[0:4, 0:4] = True
    result = mask_graph.evaluate_edge(
        track(),
        graph,
        node("current", 2, mask=disjoint),
    )
    assert result.iou_3d == pytest.approx(1.0)
    assert result.mutual_inside == pytest.approx(1.0)
    assert result.projection_iou == pytest.approx(0.0)
    assert result.accepted is False
    assert result.reason == "projection"


def test_projection_context_matches_uncached_result_and_is_reusable(
    monkeypatch,
):
    graph = mask_graph.MaskGraphState(config=graph_config())
    graph.add_node(node("seed", 1))
    current = node("current", 2)
    aggregate = track()

    uncached = mask_graph.evaluate_edge(aggregate, graph, current)

    inverse_calls = 0
    original_inverse = mask_graph.np.linalg.inv

    def counted_inverse(value):
        nonlocal inverse_calls
        inverse_calls += 1
        return original_inverse(value)

    monkeypatch.setattr(mask_graph.np.linalg, "inv", counted_inverse)
    context = mask_graph.build_projection_context(current, graph.config)
    assert context is not None
    assert inverse_calls == 1
    assert context.mask_area == int(np.count_nonzero(current.mask))
    assert not context.binary_mask.flags.writeable
    assert not context.world_to_camera.flags.writeable

    first = mask_graph.evaluate_edge(
        aggregate,
        graph,
        current,
        projection_context=context,
    )
    second = mask_graph.evaluate_edge(
        aggregate,
        graph,
        current,
        projection_context=context,
    )
    assert inverse_calls == 1
    assert first == uncached
    assert second == uncached


def test_projection_absence_is_explicit_not_silently_neutral():
    graph = mask_graph.MaskGraphState(config=graph_config())
    graph.add_node(node("seed", 1, with_projection=False))
    result = mask_graph.evaluate_edge(
        track(),
        graph,
        node("current", 2, with_projection=False),
    )
    assert result.projection_iou is None
    assert result.reason == "missing_projection"
    assert result.accepted is False


def test_disjoint_aabb_short_circuits_points_and_projection(monkeypatch):
    graph = mask_graph.MaskGraphState(config=graph_config())
    graph.add_node(node("seed", 1))
    far = node(
        "far",
        2,
        center=(4.0, 0.0, 3.0),
        # The mask still agrees with the current track projection.
        mask=projection_inputs()[0],
    )

    def unexpected_call(*args, **kwargs):
        raise AssertionError("expensive geometry path must be skipped")

    monkeypatch.setattr(
        mask_graph, "_point_inside_fraction", unexpected_call
    )
    monkeypatch.setattr(
        mask_graph, "_project_track_mask_iou", unexpected_call
    )
    result = mask_graph.evaluate_edge(track(), graph, far)
    # A disjoint metric AABB is a conclusive hard-geometry rejection. The
    # online path deliberately short-circuits before image projection rather
    # than spending rasterization work on a pair that cannot be associated.
    assert result.projection_iou is None
    assert result.iou_3d == 0.0
    assert result.mutual_inside == 0.0
    assert result.geometry_matches == 0
    assert result.reason == "geometry"


def test_unreachable_geometry_bound_skips_points_and_projection(monkeypatch):
    graph = mask_graph.MaskGraphState(
        config=graph_config(
            minimum_iou_3d=0.90,
            minimum_geometry_matches=3,
        )
    )
    graph.add_node(node("seed", 1))
    partially_overlapping = node(
        "partial",
        2,
        center=(0.4, 0.0, 3.0),
    )

    def unexpected_call(*args, **kwargs):
        raise AssertionError("unreachable match count must short-circuit")

    monkeypatch.setattr(
        mask_graph, "_point_inside_fraction", unexpected_call
    )
    monkeypatch.setattr(
        mask_graph, "_project_track_mask_iou", unexpected_call
    )
    result = mask_graph.evaluate_edge(
        track(), graph, partially_overlapping
    )
    assert 0.0 < result.iou_3d < 0.90
    assert result.projection_iou is None
    assert result.geometry_matches == 0
    assert result.reason == "geometry"


def test_appearance_cosine_is_a_soft_penalty_not_a_hard_veto():
    graph = mask_graph.MaskGraphState(config=graph_config())
    graph.add_node(node("seed", 1))
    positive = mask_graph.evaluate_edge(
        track(feature=(1.0, 0.0)),
        graph,
        node("same", 2, feature=(1.0, 0.0)),
    )
    negative = mask_graph.evaluate_edge(
        track(feature=(1.0, 0.0)),
        graph,
        node("opposite", 2, feature=(-1.0, 0.0)),
    )
    assert positive.accepted is True
    assert negative.accepted is True
    assert negative.appearance_cosine == pytest.approx(-1.0)
    assert negative.appearance_compatibility == pytest.approx(0.0)
    assert negative.score < positive.score


def test_labels_use_compatible_groups_and_remain_soft_on_mismatch():
    graph = mask_graph.MaskGraphState(
        config=graph_config(
            label_weight=0.5,
            label_compatibility_groups=[["sofa", "couch"]],
        )
    )
    graph.add_node(node("seed", 1, label="sofa"))
    compatible = mask_graph.evaluate_edge(
        track(label="sofa"),
        graph,
        node("compatible", 2, label="couch"),
    )
    mismatch = mask_graph.evaluate_edge(
        track(label="sofa"),
        graph,
        node("mismatch", 2, label="table"),
    )
    assert compatible.label_compatibility == pytest.approx(1.0)
    assert mismatch.label_compatibility == pytest.approx(0.25)
    assert compatible.accepted is True
    assert mismatch.accepted is True
    assert mismatch.score < compatible.score


def test_label_compatibility_callback_is_bounded_and_used():
    graph = mask_graph.MaskGraphState(config=graph_config())
    graph.add_node(node("seed", 1))
    result = mask_graph.evaluate_edge(
        track(),
        graph,
        node("target", 2),
        label_compatibility=lambda left, right: 0.75,
    )
    assert result.label_compatibility == pytest.approx(0.75)
    with pytest.raises(ValueError, match="override"):
        mask_graph.evaluate_edge(
            track(),
            graph,
            node("bad", 2),
            label_compatibility=lambda left, right: 1.5,
        )


def test_update_seeds_then_confirms_only_on_a_new_frame():
    graph = mask_graph.MaskGraphState(track_id=12, config=graph_config())
    first = mask_graph.update_mask_graph(track(), graph, node("a", 1))
    assert first.seeded is True
    assert first.accepted is True
    assert first.confirmed is False

    duplicate_view = mask_graph.update_mask_graph(
        track(), graph, node("b", 1)
    )
    assert duplicate_view.accepted is True
    assert duplicate_view.confirmed is False
    assert graph.unique_frame_count == 1

    new_view = mask_graph.update_mask_graph(track(), graph, node("c", 2))
    assert new_view.accepted is True
    assert new_view.became_confirmed is True
    assert new_view.confirmed is True
    assert graph.node_count == 3
    assert graph.edge_count == 2


def test_disabled_update_is_a_true_noop():
    graph = mask_graph.MaskGraphState()
    result = mask_graph.update_mask_graph(object(), graph, object())
    assert result.accepted is False
    assert result.reason == "disabled"
    assert graph.node_count == 0
    assert graph.edge_count == 0


def test_lifted_like_adapter_uses_existing_boxfusion_field_layout():
    mask, intrinsics, pose = projection_inputs()
    proposal = types.SimpleNamespace(
        mask=mask,
        feature=np.asarray([3.0, 4.0], dtype=np.float32),
        label="chair",
        score=0.75,
    )
    observation = types.SimpleNamespace(
        points_world=cube_points(),
    )
    view = types.SimpleNamespace(
        frame_index=9,
        intrinsics=intrinsics,
        camera_to_world=pose,
    )
    lifted = types.SimpleNamespace(
        box=np.asarray([0.0, 0.0, 3.0, 1.0, 1.0, 1.0]),
        proposal=proposal,
        observation=observation,
        view=view,
    )
    converted = mask_graph.coerce_mask_graph_node(lifted, node_id="lifted")
    assert converted.frame_id == 9
    assert converted.confidence == pytest.approx(0.75)
    assert converted.label == "chair"
    np.testing.assert_allclose(converted.appearance_feature, [0.6, 0.8])
    np.testing.assert_array_equal(converted.mask, mask)


def test_candidate_track_like_adapter_reads_memory_and_stats():
    memory = types.SimpleNamespace(
        aabb=(
            np.asarray([0.0, 0.0, 3.0]),
            np.asarray([1.0, 1.0, 1.0]),
        ),
        points=cube_points(),
    )
    stats = types.SimpleNamespace(
        feature_sum=np.asarray([2.0, 0.0]),
        feature_count=2,
        label="chair",
    )
    adapted = mask_graph.coerce_track_evidence(
        types.SimpleNamespace(memory=memory, stats=stats)
    )
    np.testing.assert_allclose(adapted.center, [0.0, 0.0, 3.0])
    np.testing.assert_allclose(adapted.appearance_feature, [1.0, 0.0])
    assert adapted.label == "chair"


def test_summary_reports_bounds_confirmation_and_edge_samples():
    graph = mask_graph.MaskGraphState(track_id="candidate-1", config=graph_config())
    graph.add_node(node("a", 1, with_projection=False))
    graph.add_node(node("b", 2, with_projection=False))
    graph.add_edge(edge(graph.nodes["a"], graph.nodes["b"], sample_count=3))
    assert graph.summary() == {
        "enabled": True,
        "track_id": "candidate-1",
        "nodes": 2,
        "edges": 1,
        "unique_frames": 2,
        "confirmed": True,
        "confirmation_frame_id": 2,
        "edge_samples": 3,
    }
