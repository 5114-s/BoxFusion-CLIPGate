"""CPU-only integration contracts for the missing-track Mask Graph route."""

from __future__ import annotations

import math
from copy import deepcopy
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from boxfusion.object_memory import (
    CandidateTrack,
    ObjectGeometryMemory,
    aabb_corners,
)
from boxfusion.online_ablation import apply_online_ablation_profile
from boxfusion.online_refinement import (
    DEFAULT_ONLINE_REFINEMENT_CONFIG,
    EvidenceStats,
    GlobalProposalMatch,
    OnlineRefinementController,
    SupplementalEvidence,
    bev_iou_and_containment,
    supplemental_extent_is_valid,
)
from boxfusion.supplemental_proposals import SupplementalProposal


class _SequenceProvider:
    def __init__(self, batches):
        self.batches = list(batches)
        self.calls = 0

    def predict(self, images, *, frame_ids=None):
        assert len(images) == 1
        assert frame_ids is not None and len(frame_ids) == 1
        batch = self.batches[min(self.calls, len(self.batches) - 1)]
        self.calls += 1
        return [batch]


class _OrderingRefiner(torch.nn.Module):
    def __init__(self, events) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.events = events
        self.config = SimpleNamespace(
            max_center_fraction=0.15,
            max_log_dimension_residual=math.log(1.25),
            minimum_dimension=1e-3,
        )

    def forward(self, points, boxes, quality_features, point_mask):
        self.events.append("b5")
        batch = points.shape[0]
        zeros = torch.zeros(
            (batch, 3), dtype=points.dtype, device=points.device
        )
        quality = torch.full(
            (batch,), 0.99, dtype=points.dtype, device=points.device
        )
        return {
            "center_residual": zeros,
            "log_dimension_residual": zeros,
            "quality": quality,
        }


def _inputs():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    depth = np.full((16, 16), 2.0, dtype=np.float32)
    intrinsics = np.asarray(
        [[80.0, 0.0, 7.5], [0.0, 80.0, 7.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    pose = np.eye(4, dtype=np.float32)
    mask = np.zeros((16, 16), dtype=bool)
    mask[3:13, 3:13] = True
    proposal = SupplementalProposal(
        bbox=np.asarray([3.0, 3.0, 13.0, 13.0], dtype=np.float32),
        score=0.8,
        mask=mask,
        label="chair",
        feature=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    )
    return image, depth, intrinsics, pose, proposal


def _runtime_config(tmp_path, profile):
    online = deepcopy(DEFAULT_ONLINE_REFINEMENT_CONFIG)
    online["enabled"] = False
    online["inference_every_keyframes"] = 1
    online["supplemental_proposals"] = {"enabled": False}
    # DEFAULT_ONLINE_REFINEMENT_CONFIG intentionally leaves this subsection
    # empty for the runtime resolver; ablation profiles require every output
    # section to carry an explicit Boolean toggle.
    online["object_memory"] = {"enabled": True}
    online["diagnostics"].update(
        {
            "enabled": False,
            "dump_track_memory": False,
            "root": str(tmp_path),
        }
    )
    config = apply_online_ablation_profile(
        {"dataset": "scannet", "online_refinement": online},
        profile,
    )
    runtime = config["online_refinement"]
    runtime["object_memory"].update(
        {
            "min_depth": 0.1,
            "max_depth": 5.0,
            "depth_scale": 1.0,
            "mask_edge_margin": 0,
            "depth_edge_threshold": None,
            "voxel_size": 0.0,
            "max_points_per_observation": 512,
            "max_points_per_object": 1024,
            "aabb_lower_quantile": 0.0,
            "aabb_upper_quantile": 1.0,
            "min_points_for_aabb": 4,
            "minimum_aabb_dimension": 0.01,
            "association_iou_threshold": 0.0,
            "association_center_distance": 1.0,
            "association_inside_fraction": 0.0,
        }
    )
    # The integration tests isolate temporal graph state. Identical masks
    # should deterministically associate without depending on rasterization
    # boundary conventions.
    runtime["mask_graph"].update(
        {
            "minimum_edge_score": 0.0,
            "minimum_iou_3d": 0.0,
            "minimum_mutual_inside": 0.0,
            "minimum_projection_iou": 0.0,
            "minimum_geometry_matches": 1,
            "require_projection": False,
        }
    )
    return config


def _process(
    controller,
    frame_id,
    *,
    corners=None,
    scores=None,
    stable_ids=None,
):
    image, depth, intrinsics, pose, _ = _inputs()
    if corners is None:
        corners = np.empty((0, 8, 3), dtype=np.float32)
    if scores is None:
        scores = np.empty(0, dtype=np.float32)
    if stable_ids is None:
        stable_ids = np.empty(0, dtype=np.int64)
    controller.process_keyframe(
        image=image,
        depth=depth,
        intrinsics=intrinsics,
        camera_to_world=pose,
        frame_id=frame_id,
        scene_id="scene0000_00",
        global_corners=corners,
        global_scores=scores,
        stable_ids=stable_ids,
    )


def _install_confirmed_track(
    controller,
    track_id,
    box,
    *,
    detector_score,
    label=None,
):
    box = np.asarray(box, dtype=np.float32)
    points = aabb_corners(box[:3], box[3:]).astype(np.float32)
    memory = ObjectGeometryMemory(track_id, controller.object_config)
    memory._points = points.copy()
    memory._geometry_points = points.copy()
    memory.observation_count = 3
    memory.unique_view_count = 3
    track = CandidateTrack(
        track_id=track_id,
        memory=memory,
        created_frame=0,
        last_frame=2,
        created_lifecycle_step=0,
        last_lifecycle_step=2,
        hit_count=3,
        view_count=3,
        confirmed=True,
    )
    stats = EvidenceStats()
    stats.scores = [float(detector_score)]
    if label is not None:
        stats.label_votes[str(label)] = float(detector_score)
    controller.track_manager.tracks[track_id] = track
    controller.track_manager.next_track_id = max(
        controller.track_manager.next_track_id, track_id + 1
    )
    controller.supplemental_metadata[track_id] = SupplementalEvidence(
        track_id=track_id,
        stats=stats,
    )
    return track


def _manual_supplemental_config(tmp_path, profile):
    config = _runtime_config(tmp_path, profile)
    online = config["online_refinement"]
    online["supplemental_output"].update(
        {
            "require_mask_graph_confirmation": False,
            "min_score": 0.0,
            "min_projection_iou": 0.0,
        }
    )
    return config


def _enable_diagnostics(config, root):
    config["online_refinement"]["diagnostics"].update(
        {
            "enabled": True,
            "dump_track_memory": True,
            "root": str(root),
            "point_count": 32,
        }
    )
    return config


def _supplemental_config(tmp_path, profile):
    return _runtime_config(tmp_path, profile)["online_refinement"][
        "supplemental_output"
    ]


def test_c1_extent_policy_is_class_aware_and_profile_isolated(tmp_path):
    legacy = _supplemental_config(
        tmp_path,
        "missing_mask_graph_supplemental"
    )
    c1 = _supplemental_config(
        tmp_path, "missing_mask_graph_c1_recovery"
    )

    for key in (
        "recover_absorbed_confirmed",
        "class_aware_extent",
        "bev_duplicate_enabled",
        "planar_duplicate_enabled",
        "rank_after_globals",
    ):
        assert legacy[key] is False
        assert c1[key] is True

    sink = np.asarray([0.3145803, 0.2913228, 0.16876557])
    real_door = np.asarray([0.5968561, 0.05, 1.17758])
    false_door = np.asarray([0.4285, 0.0701, 0.5568])

    assert not supplemental_extent_is_valid(
        sink,
        "sink",
        legacy,
        default_minimum_extent=0.30,
    )
    assert supplemental_extent_is_valid(
        sink,
        "sink",
        c1,
        default_minimum_extent=0.30,
    )
    assert supplemental_extent_is_valid(
        real_door,
        "door",
        c1,
        default_minimum_extent=0.30,
    )
    assert not supplemental_extent_is_valid(
        false_door,
        "door",
        c1,
        default_minimum_extent=0.30,
    )

    unknown = np.asarray([0.35, 0.35, 0.35])
    assert supplemental_extent_is_valid(
        unknown,
        "unknown",
        c1,
        default_minimum_extent=0.30,
    )
    assert not supplemental_extent_is_valid(
        unknown,
        "unknown",
        c1,
        default_minimum_extent=0.40,
    )


def test_c1_bev_duplicate_gate_rejects_bed_but_not_sink(tmp_path):
    # Fixed10 SAM3 diagnostics: the bed footprint is contained by an
    # existing global detection, while the absorbed sink remains distinct.
    bed = np.asarray(
        [2.4012995, 2.6610436, 0.5090493, 1.5012228, 1.5615145, 0.46252525]
    )
    global_bed = np.asarray(
        [2.4246194, 2.8759356, 0.69887924, 1.7287321, 2.5414772, 1.670018]
    )
    sink = np.asarray(
        [1.4348634, 2.605733, 2.0791547, 0.3145803, 0.2913228, 0.16876557]
    )
    nearby_global = np.asarray(
        [1.4267473, 2.6152325, 2.0392272, 0.61255765, 0.45444393, 0.19321382]
    )
    cfg = _supplemental_config(
        tmp_path, "missing_mask_graph_c1_recovery"
    )

    bed_iou, bed_containment = bev_iou_and_containment(
        bed, global_bed
    )
    sink_iou, sink_containment = bev_iou_and_containment(
        sink, nearby_global
    )

    assert bed_iou >= cfg["bev_duplicate_iou"]
    assert bed_containment >= cfg["bev_duplicate_containment"]
    assert not (
        sink_iou >= cfg["bev_duplicate_iou"]
        and sink_containment >= cfg["bev_duplicate_containment"]
    )


def test_mask_graph_observer_is_exact_output_identity(tmp_path):
    *_, proposal = _inputs()
    config = _runtime_config(
        tmp_path, "missing_mask_graph_observer"
    )
    controller = OnlineRefinementController(
        config,
        provider=_SequenceProvider([[proposal], [proposal], [proposal]]),
    )
    box = np.asarray([1.0, 0.0, 2.0, 0.21, 0.19, 0.17])
    corners = aabb_corners(box[:3], box[3:])[None].astype(np.float32)
    scores = np.asarray([0.8125], dtype=np.float32)
    stable_ids = np.asarray([17], dtype=np.int64)

    for frame_id in range(3):
        _process(
            controller,
            frame_id,
            corners=corners,
            scores=scores,
            stable_ids=stable_ids,
        )
    result = controller.finalize(
        global_corners=corners,
        global_scores=scores,
        stable_ids=stable_ids,
    )

    np.testing.assert_array_equal(result.corners, corners)
    np.testing.assert_array_equal(result.scores, scores)
    assert result.source_indices.tolist() == [0]
    assert result.stable_ids.tolist() == [17]
    assert result.summary["supplemental_output"] == 0
    assert result.summary["mask_graph_confirmed_components"] == 1


def test_supplemental_requires_graph_confirmation_not_only_track_confirmation(
    tmp_path,
):
    *_, proposal = _inputs()
    config = _runtime_config(
        tmp_path, "missing_mask_graph_supplemental"
    )
    online = config["online_refinement"]
    # Deliberately make the legacy candidate track confirm one view before
    # the graph, proving the output gate consumes the graph latch.
    online["object_memory"]["min_confirmations"] = 2
    online["supplemental_output"]["min_confirmations"] = 2
    online["supplemental_output"]["minimum_extent"] = 0.0
    online["supplemental_output"]["min_score"] = 0.0
    online["supplemental_output"]["min_projection_iou"] = 0.0
    controller = OnlineRefinementController(
        config,
        provider=_SequenceProvider([[proposal], [proposal], [proposal]]),
    )
    empty_corners = np.empty((0, 8, 3), dtype=np.float32)
    empty_scores = np.empty(0, dtype=np.float32)
    empty_ids = np.empty(0, dtype=np.int64)

    for frame_id in range(2):
        _process(controller, frame_id)
    track = controller.track_manager.tracks[0]
    graph = controller.supplemental_metadata[0].graph
    assert track.confirmed is True
    assert graph is not None and graph.confirmed is False

    before = controller.finalize(
        global_corners=empty_corners,
        global_scores=empty_scores,
        stable_ids=empty_ids,
    )
    assert before.boxes.shape == (0, 6)
    assert before.summary["supplemental_rejected_graph"] == 1

    _process(controller, 2)
    assert graph.confirmed is True
    after = controller.finalize(
        global_corners=empty_corners,
        global_scores=empty_scores,
        stable_ids=empty_ids,
    )
    assert after.boxes.shape == (1, 6)
    assert after.source_indices.tolist() == [-1]
    assert after.stable_ids.tolist() == [-1]
    assert after.summary["supplemental_rejected_graph"] == 0
    assert after.summary["supplemental_output"] == 1


def test_weak_global_match_is_shadowed_into_missing_track_graph(tmp_path):
    *_, proposal = _inputs()
    config = _runtime_config(
        tmp_path, "missing_mask_graph_observer"
    )
    controller = OnlineRefinementController(
        config, provider=_SequenceProvider([[proposal]])
    )

    def force_weak_match(self, lifted, boxes, intrinsics, pose):
        assert len(lifted) == 1 and len(boxes) == 1
        match = GlobalProposalMatch(
            global_index=0,
            overlap_3d=0.10,
            projection_iou=0.20,
            point_support=0.50,
            center_distance=0.01,
            score=0.70,
            strong=False,
        )
        return {0: 0}, {0: match}

    controller._match_to_globals_detailed = MethodType(
        force_weak_match, controller
    )
    box = np.asarray([0.0, 0.0, 2.0, 0.40, 0.40, 0.40])
    corners = aabb_corners(box[:3], box[3:])[None].astype(np.float32)
    scores = np.asarray([0.8], dtype=np.float32)
    stable_ids = np.asarray([3], dtype=np.int64)
    _process(
        controller,
        0,
        corners=corners,
        scores=scores,
        stable_ids=stable_ids,
    )

    assert controller.stats["matched_global"] == 1
    assert controller.stats["strong_global_matches"] == 0
    assert controller.stats["weak_global_matches"] == 1
    assert controller.stats["weak_shadow_candidates"] == 1
    assert controller.stats["unmatched_candidates"] == 0
    assert controller.stats["candidate_updates"] == 1
    assert sorted(controller.track_manager.tracks) == [0]
    graph = controller.supplemental_metadata[0].graph
    assert graph is not None
    assert len(graph.nodes) == 1


def test_supplemental_b6_scores_once_before_b5_runs(tmp_path):
    events = []
    mappings = []
    *_, proposal = _inputs()
    config = _runtime_config(
        tmp_path, "missing_mask_graph_b5_b6"
    )
    online = config["online_refinement"]
    online["supplemental_output"]["minimum_extent"] = 0.0
    online["supplemental_output"]["min_score"] = 0.0
    online["supplemental_output"]["min_projection_iou"] = 0.0
    online["quality"]["blend_with_detector"] = 0.0
    online["quality"]["preserve_original_floor"] = False
    online["box_refiner"]["point_count"] = 32
    online["refit"].update(
        {
            "min_views": 2,
            "min_points": 4,
            "min_original_point_support": 0.0,
            "min_candidate_point_support": 0.0,
            "max_candidate_support_drop": 1.0,
            "min_reprojection_iou": 0.0,
            "min_reprojection_improvement": -1.0,
        }
    )

    def score(mapping):
        events.append("b6")
        mappings.append(dict(mapping))
        return 0.123

    controller = OnlineRefinementController(
        config,
        provider=_SequenceProvider([[proposal], [proposal], [proposal]]),
        box_refiner=_OrderingRefiner(events),
        quality_scorer=score,
    )
    for frame_id in range(3):
        _process(controller, frame_id)
    result = controller.finalize(
        global_corners=np.empty((0, 8, 3), dtype=np.float32),
        global_scores=np.empty(0, dtype=np.float32),
        stable_ids=np.empty(0, dtype=np.int64),
    )

    assert events == ["b6", "b5"]
    assert len(mappings) == 1
    assert mappings[0]["refiner_quality"] == pytest.approx(0.5)
    assert result.scores.tolist() == pytest.approx([0.123])
    assert result.summary["supplemental_b5_attempted"] == 1
    assert result.summary["supplemental_output"] == 1


def test_post_b5_extent_gate_rejects_box_that_shrinks_below_threshold(
    tmp_path,
):
    config = _manual_supplemental_config(
        tmp_path, "missing_mask_graph_b5_b6"
    )
    controller = OnlineRefinementController(
        config,
        provider=_SequenceProvider([[]]),
        box_refiner=object(),
        quality_scorer=lambda mapping: 0.8,
    )
    original = np.asarray([0.0, 0.0, 2.0, 0.31, 0.31, 0.31])
    _install_confirmed_track(
        controller, 0, original, detector_score=0.8
    )
    shrunken = original.copy()
    shrunken[3:6] = 0.25

    def shrink(self, original_corners, evidence, mapping):
        return (
            shrunken.copy(),
            aabb_corners(shrunken[:3], shrunken[3:]),
            0.99,
            True,
            "neural_accepted",
        )

    controller._run_oriented_neural_refiner = MethodType(
        shrink, controller
    )
    result = controller.finalize(
        global_corners=np.empty((0, 8, 3), dtype=np.float32),
        global_scores=np.empty(0, dtype=np.float32),
        stable_ids=np.empty(0, dtype=np.int64),
    )

    assert result.boxes.shape == (0, 6)
    assert result.summary["supplemental_rejected_extent"] == 0
    assert result.summary["supplemental_rejected_refined_extent"] == 1
    assert result.summary["supplemental_b5_attempted"] == 1
    assert result.summary["supplemental_b5_accepted"] == 1


def test_post_b5_global_gate_rejects_candidate_moved_into_global_box(
    tmp_path,
):
    config = _manual_supplemental_config(
        tmp_path, "missing_mask_graph_b5_b6"
    )
    controller = OnlineRefinementController(
        config,
        provider=_SequenceProvider([[]]),
        box_refiner=object(),
        quality_scorer=lambda mapping: 0.8,
    )
    original = np.asarray([0.0, 0.0, 2.0, 0.31, 0.31, 0.31])
    global_box = np.asarray([1.0, 0.0, 2.0, 0.40, 0.40, 0.40])
    _install_confirmed_track(
        controller, 0, original, detector_score=0.8
    )

    def move_to_global(self, original_corners, evidence, mapping):
        return (
            global_box.copy(),
            aabb_corners(global_box[:3], global_box[3:]),
            0.99,
            True,
            "neural_accepted",
        )

    controller._run_oriented_neural_refiner = MethodType(
        move_to_global, controller
    )
    global_corners = aabb_corners(
        global_box[:3], global_box[3:]
    )[None].astype(np.float32)
    result = controller.finalize(
        global_corners=global_corners,
        global_scores=np.asarray([0.75], dtype=np.float32),
        stable_ids=np.asarray([9], dtype=np.int64),
    )

    assert result.source_indices.tolist() == [0]
    assert result.stable_ids.tolist() == [9]
    assert result.summary["supplemental_rejected_global"] == 0
    assert result.summary["supplemental_rejected_refined_global"] == 1
    assert result.summary["supplemental_output"] == 0


def test_post_b5_supplemental_gate_rejects_newly_colliding_candidate(
    tmp_path,
):
    config = _manual_supplemental_config(
        tmp_path, "missing_mask_graph_b5_b6"
    )
    controller = OnlineRefinementController(
        config,
        provider=_SequenceProvider([[]]),
        box_refiner=object(),
        quality_scorer=lambda mapping: mapping["detector_score"],
    )
    first = np.asarray([0.0, 0.0, 2.0, 0.31, 0.31, 0.31])
    second = np.asarray([1.0, 0.0, 2.0, 0.31, 0.31, 0.31])
    _install_confirmed_track(
        controller, 0, first, detector_score=0.9
    )
    _install_confirmed_track(
        controller, 1, second, detector_score=0.8
    )

    def collide(self, original_corners, evidence, mapping):
        return (
            first.copy(),
            aabb_corners(first[:3], first[3:]),
            0.99,
            True,
            "neural_accepted",
        )

    controller._run_oriented_neural_refiner = MethodType(
        collide, controller
    )
    result = controller.finalize(
        global_corners=np.empty((0, 8, 3), dtype=np.float32),
        global_scores=np.empty(0, dtype=np.float32),
        stable_ids=np.empty(0, dtype=np.int64),
    )

    assert result.stable_ids.tolist() == [-1]
    assert result.summary["supplemental_deduplicated"] == 0
    assert result.summary["supplemental_refined_deduplicated"] == 1
    assert result.summary["supplemental_output"] == 1


def test_duplicate_supplemental_tracks_are_ranked_by_frozen_b6_score(
    tmp_path,
):
    config = _manual_supplemental_config(
        tmp_path, "missing_mask_graph_b6"
    )
    online = config["online_refinement"]
    online["quality"]["blend_with_detector"] = 0.0
    online["quality"]["preserve_original_floor"] = False
    controller = OnlineRefinementController(
        config,
        provider=_SequenceProvider([[]]),
        quality_scorer=lambda mapping: 1.0
        - float(mapping["detector_score"]),
    )
    duplicate = np.asarray([0.0, 0.0, 2.0, 0.31, 0.31, 0.31])
    _install_confirmed_track(
        controller, 0, duplicate, detector_score=0.9
    )
    _install_confirmed_track(
        controller, 1, duplicate, detector_score=0.6
    )

    result = controller.finalize(
        global_corners=np.empty((0, 8, 3), dtype=np.float32),
        global_scores=np.empty(0, dtype=np.float32),
        stable_ids=np.empty(0, dtype=np.int64),
    )

    # Track 0 has the higher detector score (0.9), while track 1 has the
    # higher frozen B6 score (0.4). Final suppression must retain track 1.
    assert result.stable_ids.tolist() == [-2]
    assert result.scores.tolist() == pytest.approx([0.4])
    assert result.summary["supplemental_deduplicated"] == 1
    assert result.summary["supplemental_output"] == 1


def test_observer_diagnostics_include_live_archived_and_retired_graphs(
    tmp_path,
):
    *_, proposal = _inputs()
    config = _enable_diagnostics(
        _runtime_config(
            tmp_path, "missing_mask_graph_observer"
        ),
        tmp_path,
    )
    controller = OnlineRefinementController(
        config,
        provider=_SequenceProvider([[proposal], [proposal], [proposal]]),
    )
    for frame_id in range(3):
        _process(controller, frame_id)

    active = controller.track_manager.tracks[0]
    active_metadata = controller.supplemental_metadata[0]
    archived = deepcopy(active)
    archived.track_id = 1
    archived.memory.track_id = 1
    archived_metadata = deepcopy(active_metadata)
    archived_metadata.track_id = 1
    controller.track_manager.archived_tracks[1] = archived
    controller.supplemental_metadata[1] = archived_metadata

    retired = deepcopy(active)
    retired.track_id = 2
    retired.memory.track_id = 2
    retired_metadata = deepcopy(active_metadata)
    retired_metadata.track_id = 2
    retired_snapshot = controller._mask_graph_snapshot(
        retired,
        retired_metadata,
        lifecycle_state="discarded",
        event_frame=3,
    )
    assert retired_snapshot is not None
    controller.retired_mask_graph_snapshots.append(retired_snapshot)

    result = controller.finalize(
        global_corners=np.empty((0, 8, 3), dtype=np.float32),
        global_scores=np.empty(0, dtype=np.float32),
        stable_ids=np.empty(0, dtype=np.int64),
    )
    assert result.boxes.shape == (0, 6)
    assert result.summary["supplemental_output"] == 0
    assert result.summary["mask_graph_components"] == 3
    assert result.summary["mask_graph_retired_components"] == 1
    assert result.summary["supplemental_minimum_extent"] == pytest.approx(
        0.30
    )
    live_memories = (active.memory, archived.memory)
    assert result.summary["supplemental_top_k_candidate_views"] == sum(
        memory.view_candidate_count for memory in live_memories
    )
    assert result.summary["supplemental_top_k_selected_views"] == sum(
        memory.selected_view_count for memory in live_memories
    )
    assert result.summary["supplemental_top_k_geometry_points"] == sum(
        memory.geometry_num_points for memory in live_memories
    )

    with np.load(
        tmp_path / "scene0000_00_tracks.npz", allow_pickle=False
    ) as payload:
        assert payload["boxes"].shape == (0, 6)
        states = {
            int(track_id): str(state)
            for track_id, state in zip(
                payload["graph_component_track_ids"],
                payload["graph_component_states"],
            )
        }
        assert states == {0: "active", 1: "archived", 2: "discarded"}
        assert payload["graph_component_boxes"].shape == (3, 6)
        assert payload["graph_component_node_counts"].shape == (3,)
        assert payload["graph_component_edge_counts"].shape == (3,)
        assert payload["supplemental_minimum_extent"].item() == pytest.approx(
            0.30
        )


def test_absorbed_graph_snapshot_survives_metadata_removal(tmp_path):
    *_, proposal = _inputs()
    config = _enable_diagnostics(
        _runtime_config(
            tmp_path, "missing_mask_graph_observer"
        ),
        tmp_path,
    )
    controller = OnlineRefinementController(
        config,
        provider=_SequenceProvider([[proposal], [proposal]]),
    )
    _process(controller, 0)
    center, dims = controller.track_manager.tracks[0].memory.aabb
    global_box = np.concatenate((center, dims))
    global_corners = aabb_corners(
        global_box[:3], global_box[3:]
    )[None].astype(np.float32)
    _process(
        controller,
        1,
        corners=global_corners,
        scores=np.asarray([0.75], dtype=np.float32),
        stable_ids=np.asarray([7], dtype=np.int64),
    )

    assert controller.track_manager.tracks == {}
    assert controller.supplemental_metadata == {}
    assert [
        row["lifecycle_state"]
        for row in controller.retired_mask_graph_snapshots
    ] == ["absorbed"]
    result = controller.finalize(
        global_corners=global_corners,
        global_scores=np.asarray([0.75], dtype=np.float32),
        stable_ids=np.asarray([7], dtype=np.int64),
    )
    assert result.source_indices.tolist() == [0]
    assert result.summary["mask_graph_retired_components"] == 1
    with np.load(
        tmp_path / "scene0000_00_tracks.npz", allow_pickle=False
    ) as payload:
        assert payload["graph_component_states"].tolist() == ["absorbed"]
        assert payload["graph_component_node_counts"].tolist() == [1]


@pytest.mark.parametrize(
    "profile,expected_sources,expected_recovery",
    [
        ("missing_mask_graph_supplemental", [0], 0),
        ("missing_mask_graph_c1_recovery", [0, -1], 1),
    ],
)
def test_only_c1_recovers_graph_confirmed_absorbed_track(
    tmp_path,
    profile,
    expected_sources,
    expected_recovery,
):
    *_, proposal = _inputs()
    config = _runtime_config(tmp_path, profile)
    supplemental = config["online_refinement"]["supplemental_output"]
    supplemental.update(
        {
            "minimum_extent": 0.0,
            "min_score": 0.0,
            "min_projection_iou": 0.0,
        }
    )
    controller = OnlineRefinementController(
        config,
        provider=_SequenceProvider(
            [[proposal], [proposal], [proposal], [proposal]]
        ),
    )
    for frame_id in range(3):
        _process(controller, frame_id)
    track = controller.track_manager.tracks[0]
    graph = controller.supplemental_metadata[0].graph
    assert track.confirmed is True
    assert graph is not None and graph.confirmed is True

    def force_strong_match(self, lifted, boxes, intrinsics, pose):
        assert len(lifted) == 1 and len(boxes) == 1
        match = GlobalProposalMatch(
            global_index=0,
            overlap_3d=0.80,
            projection_iou=0.90,
            point_support=0.90,
            center_distance=0.0,
            score=3.0,
            strong=True,
        )
        return {0: 0}, {0: match}

    controller._match_to_globals_detailed = MethodType(
        force_strong_match, controller
    )
    # The forced association models a coarse global box swallowing the
    # candidate at one keyframe.  Its final geometry is deliberately far
    # away, reproducing the lifecycle failure found for scene0583's sink.
    global_box = np.asarray(
        [4.0, 4.0, 2.0, 0.50, 0.50, 0.50], dtype=np.float32
    )
    global_corners = aabb_corners(
        global_box[:3], global_box[3:]
    )[None].astype(np.float32)
    global_scores = np.asarray([0.75], dtype=np.float32)
    global_ids = np.asarray([7], dtype=np.int64)
    _process(
        controller,
        3,
        corners=global_corners,
        scores=global_scores,
        stable_ids=global_ids,
    )

    assert controller.track_manager.tracks == {}
    assert controller.supplemental_metadata == {}
    assert len(controller.absorbed_supplemental_records) == expected_recovery
    result = controller.finalize(
        global_corners=global_corners,
        global_scores=global_scores,
        stable_ids=global_ids,
    )

    assert result.source_indices.tolist() == expected_sources
    assert result.summary["absorbed_recovery_stored"] == expected_recovery
    assert result.summary["absorbed_recovery_output"] == expected_recovery
    if expected_recovery:
        assert result.stable_ids.tolist() == [7, -1]
        rank_quality = (
            (1.0 - supplemental["rank_projection_weight"]) * 0.8
            + supplemental["rank_projection_weight"]
            * float(result.quality_features[1, 4])
            + supplemental["rank_recovered_bonus"]
        )
        assert result.scores[1] == pytest.approx(
            supplemental["rank_score_floor"]
            + (
                supplemental["rank_score_ceiling"]
                - supplemental["rank_score_floor"]
            )
            * rank_quality
        )
        assert result.scores[1] < result.scores[0]
        assert result.summary["absorbed_recovery_considered"] == 1
        assert result.summary["absorbed_recovery_eligible"] == 1
        assert result.summary["supplemental_scores_rank_mapped"] == 1


def test_c1_fixed_rank_band_preserves_eligibility_and_strict_order(
    tmp_path,
):
    config = _manual_supplemental_config(
        tmp_path, "missing_mask_graph_c1_recovery"
    )
    controller = OnlineRefinementController(
        config,
        provider=_SequenceProvider([[]]),
    )
    _install_confirmed_track(
        controller,
        0,
        np.asarray([2.0, 0.0, 2.0, 0.50, 0.50, 0.50]),
        detector_score=0.80,
    )
    _install_confirmed_track(
        controller,
        1,
        np.asarray([3.0, 0.0, 2.0, 0.50, 0.50, 0.50]),
        detector_score=0.60,
    )
    global_box = np.asarray(
        [0.0, 0.0, 2.0, 0.50, 0.50, 0.50], dtype=np.float32
    )
    result = controller.finalize(
        global_corners=aabb_corners(
            global_box[:3], global_box[3:]
        )[None],
        # A low global score must not turn an already eligible supplemental
        # row into a min-score rejection.
        global_scores=np.asarray([0.20], dtype=np.float32),
        stable_ids=np.asarray([9], dtype=np.int64),
    )

    assert result.source_indices.tolist() == [0, -1, -1]
    assert result.stable_ids.tolist() == [9, -1, -2]
    assert result.summary["supplemental_rejected_score"] == 0
    assert result.summary["supplemental_scores_rank_mapped"] == 2
    assert (
        result.scores[1]
        > result.scores[2]
        >= config["online_refinement"]["supplemental_output"][
            "rank_score_floor"
        ]
    )
    assert result.scores[1] <= config["online_refinement"][
        "supplemental_output"
    ]["rank_score_ceiling"]


def test_c1_duplicate_reference_excludes_global_removed_at_export(
    tmp_path,
):
    config = _manual_supplemental_config(
        tmp_path, "missing_mask_graph_c1_recovery"
    )
    config["online_refinement"]["output_filter"][
        "final_minimum_extent"
    ] = 0.40
    controller = OnlineRefinementController(
        config,
        provider=_SequenceProvider([[]]),
    )
    candidate = np.asarray(
        [0.0, 0.0, 2.0, 0.50, 0.50, 0.50], dtype=np.float32
    )
    _install_confirmed_track(
        controller, 0, candidate, detector_score=0.80
    )
    # This row has a containing BEV footprint, but its 0.20-m Z extent means
    # it cannot survive the exact final 0.40-m ScanNet export gate.
    filtered_global = np.asarray(
        [0.0, 0.0, 2.0, 1.00, 1.00, 0.20], dtype=np.float32
    )
    result = controller.finalize(
        global_corners=aabb_corners(
            filtered_global[:3], filtered_global[3:]
        )[None],
        global_scores=np.asarray([0.80], dtype=np.float32),
        stable_ids=np.asarray([9], dtype=np.int64),
    )

    assert result.source_indices.tolist() == [-1]
    assert result.stable_ids.tolist() == [-1]
    assert result.summary["supplemental_rejected_bev_global"] == 0
    assert result.summary["supplemental_output"] == 1


def test_c1_bev_gate_keeps_vertically_separated_instance(tmp_path):
    config = _manual_supplemental_config(
        tmp_path, "missing_mask_graph_c1_recovery"
    )
    config["online_refinement"]["output_filter"][
        "final_minimum_extent"
    ] = 0.40
    controller = OnlineRefinementController(
        config,
        provider=_SequenceProvider([[]]),
    )
    candidate = np.asarray(
        [0.0, 0.0, 2.0, 0.50, 0.50, 0.50], dtype=np.float32
    )
    _install_confirmed_track(
        controller, 0, candidate, detector_score=0.80
    )
    stacked_global = np.asarray(
        [0.0, 0.0, 3.0, 0.80, 0.80, 0.50], dtype=np.float32
    )
    result = controller.finalize(
        global_corners=aabb_corners(
            stacked_global[:3], stacked_global[3:]
        )[None],
        global_scores=np.asarray([0.80], dtype=np.float32),
        stable_ids=np.asarray([9], dtype=np.int64),
    )

    assert result.source_indices.tolist() == [0, -1]
    assert result.summary["supplemental_rejected_bev_global"] == 0


def test_c1_unknown_extent_uses_exact_final_export_threshold(tmp_path):
    config = _manual_supplemental_config(
        tmp_path, "missing_mask_graph_c1_recovery"
    )
    config["online_refinement"]["output_filter"][
        "final_minimum_extent"
    ] = 0.40
    controller = OnlineRefinementController(
        config,
        provider=_SequenceProvider([[]]),
    )
    _install_confirmed_track(
        controller,
        0,
        np.asarray([0.0, 0.0, 2.0, 0.35, 0.35, 0.35]),
        detector_score=0.80,
    )
    result = controller.finalize(
        global_corners=np.empty((0, 8, 3), dtype=np.float32),
        global_scores=np.empty(0, dtype=np.float32),
        stable_ids=np.empty(0, dtype=np.int64),
    )

    assert result.boxes.shape == (0, 6)
    assert result.summary["supplemental_minimum_extent"] == pytest.approx(
        0.40
    )
    assert result.summary["supplemental_rejected_extent"] == 0
    assert result.summary["supplemental_rejected_class_extent"] == 1


def test_c1_planar_dedup_keeps_better_projected_real_door(tmp_path):
    config = _manual_supplemental_config(
        tmp_path, "missing_mask_graph_c1_recovery"
    )
    config["online_refinement"]["output_filter"][
        "final_minimum_extent"
    ] = 0.40
    controller = OnlineRefinementController(
        config,
        provider=_SequenceProvider([[]]),
    )

    # These are the two fragmented door tracks observed in the fixed10
    # scene0435_00 C1-v2 run.  Full 3D geometry is noisy, but their BEV
    # footprints and vertical spans describe the same thin planar instance.
    lower_projection_door = np.asarray(
        [
            6.663379,
            0.3521572,
            0.63462186,
            0.6920061,
            0.06998497,
            0.99396014,
        ],
        dtype=np.float32,
    )
    higher_projection_door = np.asarray(
        [
            6.728367,
            0.33210748,
            0.72257775,
            0.5968561,
            0.05000001,
            1.17758,
        ],
        dtype=np.float32,
    )
    _install_confirmed_track(
        controller,
        0,
        lower_projection_door,
        detector_score=0.9231771,
        label="door",
    )
    _install_confirmed_track(
        controller,
        1,
        higher_projection_door,
        detector_score=0.8828125,
        label="door",
    )

    original_quality_mapping = controller._quality_mapping

    def fixed_real_projection(self, **kwargs):
        mapping = dict(original_quality_mapping(**kwargs))
        mapping["projection_iou"] = (
            0.7103579
            if float(kwargs["detector_score"]) > 0.90
            else 0.77842546
        )
        return mapping

    controller._quality_mapping = MethodType(
        fixed_real_projection, controller
    )
    result = controller.finalize(
        global_corners=np.empty((0, 8, 3), dtype=np.float32),
        global_scores=np.empty(0, dtype=np.float32),
        stable_ids=np.empty(0, dtype=np.int64),
    )

    assert result.source_indices.tolist() == [-1]
    assert result.stable_ids.tolist() == [-2]
    np.testing.assert_allclose(
        result.boxes[0], higher_projection_door, rtol=0.0, atol=1e-6
    )
    assert result.labels == ("door",)
    supplemental = config["online_refinement"]["supplemental_output"]
    assert (
        supplemental["rank_score_floor"]
        <= result.scores[0]
        <= supplemental["rank_score_ceiling"]
    )
    assert result.summary["supplemental_planar_deduplicated"] == 1
    assert result.summary["supplemental_deduplicated"] == 0
    assert result.summary["supplemental_output"] == 1


def test_discarded_graph_snapshot_survives_track_and_metadata_removal(
    tmp_path,
):
    *_, proposal = _inputs()
    config = _enable_diagnostics(
        _runtime_config(
            tmp_path, "missing_mask_graph_observer"
        ),
        tmp_path,
    )
    controller = OnlineRefinementController(
        config,
        provider=_SequenceProvider(
            [[proposal], [], [], [], [], []]
        ),
    )
    for frame_id in range(6):
        _process(controller, frame_id)

    assert controller.track_manager.tracks == {}
    assert controller.supplemental_metadata == {}
    assert controller.stats["candidate_discarded"] == 1
    assert [
        row["lifecycle_state"]
        for row in controller.retired_mask_graph_snapshots
    ] == ["discarded"]
    result = controller.finalize(
        global_corners=np.empty((0, 8, 3), dtype=np.float32),
        global_scores=np.empty(0, dtype=np.float32),
        stable_ids=np.empty(0, dtype=np.int64),
    )
    assert result.summary["mask_graph_retired_components"] == 1
    assert result.summary["candidate_discarded_total"] == 1
    with np.load(
        tmp_path / "scene0000_00_tracks.npz", allow_pickle=False
    ) as payload:
        assert payload["graph_component_states"].tolist() == ["discarded"]
        assert payload["graph_component_view_counts"].tolist() == [1]


def test_supplemental_evidence_view_arrays_align_with_output_row(tmp_path):
    *_, proposal = _inputs()
    config = _runtime_config(
        tmp_path, "missing_mask_graph_supplemental"
    )
    online = config["online_refinement"]
    online["supplemental_output"].update(
        {
            "minimum_extent": 0.0,
            "min_score": 0.0,
            "min_projection_iou": 0.0,
        }
    )
    _enable_diagnostics(config, tmp_path)
    controller = OnlineRefinementController(
        config,
        provider=_SequenceProvider([[proposal], [proposal], [proposal]]),
    )
    for frame_id in range(3):
        _process(controller, frame_id)
    evidence = controller.supplemental_metadata[0].stats
    result = controller.finalize(
        global_corners=np.empty((0, 8, 3), dtype=np.float32),
        global_scores=np.empty(0, dtype=np.float32),
        stable_ids=np.empty(0, dtype=np.int64),
    )

    assert result.source_indices.tolist() == [-1]
    assert len(evidence.view_records) == 3
    assert result.summary["supplemental_top_k_candidate_views"] == 3
    assert result.summary["supplemental_top_k_selected_views"] == 3
    assert result.summary["supplemental_top_k_geometry_points"] > 0
    assert result.summary["supplemental_minimum_extent"] == pytest.approx(
        0.0
    )
    with np.load(
        tmp_path / "scene0000_00_tracks.npz", allow_pickle=False
    ) as payload:
        assert payload["result_indices"].tolist() == [0]
        assert payload["source_indices"].tolist() == [-1]
        assert payload["output_is_supplemental"].tolist() == [True]
        assert payload["output_mask_graph_confirmed"].tolist() == [True]
        assert payload["box_refiner_view_valid"][0].tolist() == [
            True,
            True,
            True,
            False,
            False,
        ]
        assert payload["box_refiner_view_frame_ids"][0, :3].tolist() == [
            0,
            1,
            2,
        ]
        np.testing.assert_allclose(
            payload["box_refiner_view_scores"][0, :3],
            [0.8, 0.8, 0.8],
        )
        np.testing.assert_allclose(
            payload["box_refiner_view_bboxes"][0, :3],
            np.tile(proposal.bbox, (3, 1)),
        )
        assert payload["view_candidate_counts"].tolist() == [3]
        assert payload["selected_view_counts"].tolist() == [3]
