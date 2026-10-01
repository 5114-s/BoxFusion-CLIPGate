"""CPU-only integration contracts for C2 supplemental geometry.

C2 is a geometry-only child of C1.  These tests deliberately construct the
same retained C1 rows without running a proposal provider so changes in
provider weights or rasterization cannot weaken the output contracts.
"""

from __future__ import annotations

from copy import deepcopy
import json

import numpy as np

from boxfusion.depth_occupancy_refiner import DepthOccupancyProposal
from boxfusion.object_memory import (
    CandidateTrack,
    MemoryViewRecord,
    ObjectGeometryMemory,
    aabb_corners,
    project_aabb_to_image,
)
from boxfusion.online_ablation import apply_online_ablation_profile
from boxfusion.online_refinement import (
    DEFAULT_ONLINE_REFINEMENT_CONFIG,
    EvidenceStats,
    OnlineRefinementController,
    SupplementalEvidence,
    ViewEvidence,
)


class _NoopProvider:
    def predict(self, images, *, frame_ids=None):
        return [[] for _ in images]


def _config(tmp_path, profile):
    online = deepcopy(DEFAULT_ONLINE_REFINEMENT_CONFIG)
    online["enabled"] = False
    online["supplemental_proposals"] = {"enabled": False}
    online["object_memory"] = {"enabled": True}
    online["diagnostics"].update(
        {
            "enabled": False,
            "dump_track_memory": False,
            "root": str(tmp_path),
        }
    )
    config = apply_online_ablation_profile(
        {
            "dataset": "scannet",
            "detection": {"score_thresh": 0.40},
            "online_refinement": online,
        },
        profile,
    )
    runtime = config["online_refinement"]
    runtime["object_memory"].update(
        {
            "voxel_size": 0.0,
            "max_points_per_observation": 512,
            "max_points_per_object": 2048,
            "aabb_lower_quantile": 0.0,
            "aabb_upper_quantile": 1.0,
            "min_points_for_aabb": 4,
            "minimum_aabb_dimension": 0.01,
        }
    )
    runtime["supplemental_output"].update(
        {
            "require_mask_graph_confirmation": False,
            "minimum_extent": 0.0,
            "min_score": 0.0,
            "min_projection_iou": 0.0,
        }
    )
    runtime["output_filter"]["minimum_extent"] = 0.0
    runtime["output_filter"]["final_minimum_extent"] = 0.0
    # Observer identity does not depend on expansion proposals; active tests
    # override this bound explicitly in their permissive verification fixture.
    runtime["supplemental_geometry_refiner"][
        "maximum_extent_ratio"
    ] = 1.0
    return config


def _install_confirmed_track(
    controller,
    track_id,
    box,
    *,
    detector_score=0.80,
    label="sink",
    geometry_points=None,
    projection_box=None,
):
    box = np.asarray(box, dtype=np.float32)
    points = aabb_corners(box[:3], box[3:]).astype(np.float32)
    if geometry_points is None:
        geometry_points = points
    geometry_points = np.asarray(geometry_points, dtype=np.float32)
    memory = ObjectGeometryMemory(track_id, controller.object_config)
    memory._points = points.copy()
    intrinsics = np.asarray(
        [
            [100.0, 0.0, 320.0],
            [0.0, 100.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    camera_to_world = np.eye(4, dtype=np.float32)
    view_records = [
        MemoryViewRecord(
            frame_id=frame_id,
            points_world=geometry_points,
            quality=0.90,
            confidence=float(detector_score),
            valid_depth_ratio=1.0,
            projection_mask_iou=1.0,
            camera_position=np.zeros(3, dtype=np.float32),
        )
        for frame_id in range(3)
    ]
    memory._set_view_candidates(view_records)
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
    stats.label_votes[str(label)] = float(detector_score)
    if projection_box is None:
        projection_box = box
    projection_box = np.asarray(projection_box, dtype=np.float32)
    bbox = project_aabb_to_image(
        projection_box[:3],
        projection_box[3:],
        intrinsics,
        camera_to_world,
        (480, 640),
    )
    assert bbox is not None
    stats.view_records = [
        ViewEvidence(
            frame_index=frame_id,
            score=float(detector_score),
            bbox=bbox,
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            image_shape=(480, 640),
            area_ratio=0.01,
        )
        for frame_id in range(3)
    ]
    controller.track_manager.tracks[track_id] = track
    controller.track_manager.next_track_id = max(
        controller.track_manager.next_track_id,
        track_id + 1,
    )
    controller.supplemental_metadata[track_id] = SupplementalEvidence(
        track_id=track_id,
        stats=stats,
    )
    return track


def _inner_grid(box, scale=0.80):
    box = np.asarray(box, dtype=np.float32)
    coordinates = np.linspace(-0.5, 0.5, 5, dtype=np.float32)
    return np.asarray(
        [
            box[:3] + scale * box[3:] * np.asarray([x, y, z])
            for x in coordinates
            for y in coordinates
            for z in coordinates
        ],
        dtype=np.float32,
    )


def _permissive_c2(config):
    cfg = config["online_refinement"]["supplemental_geometry_refiner"]
    cfg.update(
        {
            "minimum_points": 8,
            "minimum_views": 3,
            "maximum_center_shift_ratio": 1.0,
            "minimum_extent_ratio": 0.10,
            "maximum_extent_ratio": 2.0,
            "minimum_absolute_projection_iou": 0.0,
            "minimum_projection_view_iou": 0.0,
            "minimum_projection_views": 1,
            "small_minimum_candidate_support": 0.0,
            "small_maximum_projection_drop": 1.0,
            "small_minimum_projection_views": 1,
            "planar_minimum_component_fraction": 0.0,
            "planar_minimum_density_ratio": 1.0,
            "planar_minimum_candidate_support": 0.0,
            "planar_minimum_view_point_support": 0.0,
            "planar_minimum_point_support_views": 1,
            "planar_maximum_projection_drop": 1.0,
        }
    )
    cfg["proposal"].update({"min_views": 3, "min_points": 8})
    return config


def _candidate_proposal(candidate, points, *, branch="solid"):
    return DepthOccupancyProposal(
        candidate=np.asarray(candidate, dtype=np.float32),
        component_fraction=1.0,
        points=np.asarray(points, dtype=np.float32),
        planar=branch == "planar",
        reason="candidate",
        component_density=1.0,
        second_component_density=0.0,
        density_ratio=1.0,
        branch=branch,
    )


def _finalize_fixture(tmp_path, profile):
    controller = OnlineRefinementController(
        _config(tmp_path, profile),
        provider=_NoopProvider(),
    )
    supplemental_box = np.asarray(
        [2.0, 0.0, 2.0, 0.50, 0.50, 0.50],
        dtype=np.float32,
    )
    _install_confirmed_track(controller, 0, supplemental_box)
    global_box = np.asarray(
        [0.0, 0.0, 2.0, 0.60, 0.60, 0.60],
        dtype=np.float32,
    )
    result = controller.finalize(
        global_corners=aabb_corners(
            global_box[:3], global_box[3:]
        )[None].astype(np.float32),
        global_scores=np.asarray([0.75], dtype=np.float32),
        stable_ids=np.asarray([9], dtype=np.int64),
    )
    return result


def test_c2_observer_is_bit_exact_c1_output(tmp_path):
    c1 = _finalize_fixture(
        tmp_path / "c1",
        "missing_mask_graph_c1_recovery",
    )
    observer = _finalize_fixture(
        tmp_path / "observer",
        "missing_mask_graph_c2_geometry_observer",
    )

    np.testing.assert_array_equal(observer.corners, c1.corners)
    np.testing.assert_array_equal(observer.boxes, c1.boxes)
    np.testing.assert_array_equal(observer.scores, c1.scores)
    np.testing.assert_array_equal(
        observer.source_indices, c1.source_indices
    )
    np.testing.assert_array_equal(observer.stable_ids, c1.stable_ids)
    np.testing.assert_array_equal(
        observer.quality_features, c1.quality_features
    )
    np.testing.assert_array_equal(
        observer.refit_original_boxes, c1.refit_original_boxes
    )
    np.testing.assert_array_equal(
        observer.refit_local_candidate_boxes,
        c1.refit_local_candidate_boxes,
    )
    np.testing.assert_array_equal(
        observer.refit_applied, c1.refit_applied
    )
    assert observer.labels == c1.labels
    assert observer.source_indices.tolist() == [0, -1]
    assert observer.stable_ids.tolist() == [9, -1]


def _two_track_result(
    tmp_path,
    profile,
    *,
    diagnostics=False,
):
    config = _config(tmp_path, profile)
    if profile == "missing_mask_graph_c2_geometry":
        _permissive_c2(config)
    if diagnostics:
        config["online_refinement"]["diagnostics"].update(
            {
                "enabled": True,
                "dump_track_memory": True,
                "root": str(tmp_path),
                "point_count": 32,
            }
        )
    controller = OnlineRefinementController(
        config,
        provider=_NoopProvider(),
    )
    first_box = np.asarray(
        [0.0, 0.0, 2.0, 0.50, 0.50, 0.50],
        dtype=np.float32,
    )
    first_candidate = np.asarray(
        [0.0, 0.0, 2.0, 0.40, 0.40, 0.40],
        dtype=np.float32,
    )
    second_box = np.asarray(
        [2.0, 0.0, 2.0, 0.50, 0.50, 0.50],
        dtype=np.float32,
    )
    second_candidate = np.asarray(
        [2.0, 0.0, 2.0, 0.40, 0.40, 0.40],
        dtype=np.float32,
    )
    _install_confirmed_track(
        controller,
        0,
        first_box,
        detector_score=0.80,
        geometry_points=_inner_grid(first_candidate),
        projection_box=first_candidate,
    )
    _install_confirmed_track(
        controller,
        1,
        second_box,
        detector_score=0.70,
        geometry_points=_inner_grid(second_candidate),
        projection_box=second_candidate,
    )
    global_box = np.asarray(
        [4.0, 0.0, 2.0, 0.60, 0.60, 0.60],
        dtype=np.float32,
    )
    result = controller.finalize(
        global_corners=aabb_corners(
            global_box[:3], global_box[3:]
        )[None].astype(np.float32),
        global_scores=np.asarray([0.75], dtype=np.float32),
        stable_ids=np.asarray([9], dtype=np.int64),
        scene_id="scene0000_00" if diagnostics else None,
    )
    return (
        controller,
        result,
        first_box,
        first_candidate,
        second_box,
        second_candidate,
        global_box,
    )


def test_c2_active_changes_only_verified_supplemental_and_falls_back_per_row(
    tmp_path,
    monkeypatch,
):
    _, c1, *_ = _two_track_result(
        tmp_path / "c1",
        "missing_mask_graph_c1_recovery",
    )

    calls = []
    full_memory_calls = []

    def deterministic_proposal(
        original_box,
        geometry_points,
            view_count,
            *,
            full_memory_points=None,
            branch_hint=None,
            config=None,
        ):
        original = np.asarray(original_box, dtype=np.float32)
        calls.append(original.copy())
        full_memory_calls.append(
            np.asarray(full_memory_points, dtype=np.float32).copy()
        )
        if float(original[0]) > 1.0:
            raise ValueError("deterministic per-row failure")
        candidate = original.copy()
        candidate[3:] = 0.40
        return _candidate_proposal(candidate, geometry_points)

    monkeypatch.setattr(
        "boxfusion.online_refinement."
        "propose_depth_occupancy_refinement",
        deterministic_proposal,
    )
    (
        controller,
        active,
        first_box,
        first_candidate,
        second_box,
        _,
        global_box,
    ) = _two_track_result(
        tmp_path / "active",
        "missing_mask_graph_c2_geometry",
    )

    # C2 is a geometry-only child of the frozen C1 row set.
    np.testing.assert_array_equal(
        active.source_indices, c1.source_indices
    )
    np.testing.assert_array_equal(active.stable_ids, c1.stable_ids)
    np.testing.assert_array_equal(active.scores, c1.scores)
    np.testing.assert_array_equal(
        active.quality_features, c1.quality_features
    )
    assert active.labels == c1.labels
    assert active.source_indices.tolist() == [0, -1, -1]
    assert active.stable_ids.tolist() == [9, -1, -2]

    # The global row is never proposed to C2 and remains byte-identical.
    np.testing.assert_allclose(
        active.boxes[0], global_box, rtol=0.0, atol=2e-7
    )
    np.testing.assert_array_equal(active.boxes[0], c1.boxes[0])
    np.testing.assert_array_equal(active.corners[0], c1.corners[0])
    assert len(calls) == 2
    assert len(full_memory_calls) == 2
    # C2 classification uses selected Top-K geometry, while its envelope and
    # occupancy proposal receive the independent bounded full memory.
    assert all(points.shape == (8, 3) for points in full_memory_calls)
    assert all(not np.array_equal(row, global_box) for row in calls)

    # The first supplemental row is verified and changed.  The second fake
    # proposal raises, so only that row falls back to its exact C1 geometry.
    np.testing.assert_array_equal(active.boxes[1], first_candidate)
    assert not np.array_equal(active.boxes[1], first_box)
    np.testing.assert_array_equal(active.boxes[2], second_box)
    np.testing.assert_array_equal(active.boxes[2], c1.boxes[2])
    np.testing.assert_array_equal(active.corners[2], c1.corners[2])
    assert active.refit_reasons[1:] == (
        "c2_solid_accepted",
        "supplemental_identity",
    )

    accepted = controller._last_c2_runtime[-1]
    rejected = controller._last_c2_runtime[-2]
    assert (
        accepted["attempted"],
        accepted["proposed"],
        accepted["verified"],
        accepted["applied"],
        accepted["reason"],
    ) == (True, True, True, True, "accepted")
    assert rejected["attempted"] is True
    assert rejected["proposed"] is False
    assert rejected["verified"] is False
    assert rejected["applied"] is False
    assert rejected["reason"] == "proposal_error"
    assert active.summary["c2_attempted"] == 2
    assert active.summary["c2_proposed"] == 1
    assert active.summary["c2_verified"] == 1
    assert active.summary["c2_applied"] == 1
    assert active.summary["c2_rejections"] == {"proposal_error": 1}


def test_c2_diagnostic_arrays_align_with_retained_output_rows(
    tmp_path,
    monkeypatch,
):
    def deterministic_proposal(
        original_box,
        geometry_points,
            view_count,
            *,
            full_memory_points=None,
            branch_hint=None,
            config=None,
        ):
        original = np.asarray(original_box, dtype=np.float32)
        if float(original[0]) > 1.0:
            raise ValueError("deterministic per-row failure")
        candidate = original.copy()
        candidate[3:] = 0.40
        return _candidate_proposal(candidate, geometry_points)

    monkeypatch.setattr(
        "boxfusion.online_refinement."
        "propose_depth_occupancy_refinement",
        deterministic_proposal,
    )
    _, result, *_ = _two_track_result(
        tmp_path,
        "missing_mask_graph_c2_geometry",
        diagnostics=True,
    )
    path = tmp_path / "scene0000_00_tracks.npz"
    assert path.is_file()

    with np.load(path, allow_pickle=False) as payload:
        # This fixture has no online memory for the directly supplied global
        # row, so the diagnostic result-index table contains exactly the two
        # retained supplemental rows.
        assert payload["result_indices"].tolist() == [1, 2]
        assert payload["source_indices"].tolist() == [-1, -1]
        assert payload["track_ids"].tolist() == [-1, -2]
        assert payload[
            "supplemental_geometry_diagnostics_schema"
        ].item() == "c2_depth_occupancy_v1"

        aligned_keys = (
            "c2_attempted",
            "c2_proposed",
            "c2_verified",
            "c2_applied",
            "c2_reason",
            "c2_branch",
            "c2_original_boxes",
            "c2_candidate_boxes",
            "c2_component_fraction",
            "c2_component_density",
            "c2_density_ratio",
            "c2_point_count",
            "c2_view_count",
            "c2_original_support",
            "c2_candidate_support",
            "c2_original_projection",
            "c2_candidate_projection",
            "c2_projection_delta",
            "c2_projection_views",
            "c2_point_support_views",
            "c2_center_shift_ratio",
            "c2_extent_ratios",
        )
        for key in aligned_keys:
            assert payload[key].shape[0] == 2, key

        assert payload["c2_attempted"].tolist() == [True, True]
        assert payload["c2_proposed"].tolist() == [True, False]
        assert payload["c2_verified"].tolist() == [True, False]
        assert payload["c2_applied"].tolist() == [True, False]
        assert payload["c2_reason"].tolist() == [
            "accepted",
            "proposal_error",
        ]
        assert payload["c2_branch"].tolist() == ["solid", "identity"]
        np.testing.assert_array_equal(
            payload["c2_original_boxes"], result.boxes[[1, 2]]
            * np.asarray(
                [
                    [1.0, 1.0, 1.0, 1.25, 1.25, 1.25],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                dtype=np.float32,
            )
        )
        np.testing.assert_array_equal(
            payload["c2_candidate_boxes"][0],
            result.boxes[1],
        )
        np.testing.assert_array_equal(
            payload["c2_candidate_boxes"][1],
            result.boxes[2],
        )
        assert payload["c2_reason"].dtype.kind == "U"
        assert payload["c2_branch"].dtype.kind == "U"


def test_c2_planar_exact_minimum_survives_corner_round_trip(
    tmp_path,
    monkeypatch,
):
    config = _permissive_c2(
        _config(
            tmp_path,
            "missing_mask_graph_c2_geometry",
        )
    )
    controller = OnlineRefinementController(
        config,
        provider=_NoopProvider(),
    )
    original = np.asarray(
        [2.0, 0.3423491, 2.0, 0.60, 0.05, 0.80],
        dtype=np.float32,
    )
    candidate = np.asarray(
        [2.0, 0.3423491, 2.0, 0.60, 0.034, 0.80],
        dtype=np.float32,
    )
    _install_confirmed_track(
        controller,
        0,
        original,
        label="door",
        geometry_points=_inner_grid(candidate),
        projection_box=candidate,
    )

    def planar_proposal(
        original_box,
        geometry_points,
        view_count,
        *,
        full_memory_points=None,
        branch_hint=None,
        config=None,
    ):
        assert branch_hint == "planar"
        return _candidate_proposal(
            candidate, geometry_points, branch="planar"
        )

    monkeypatch.setattr(
        "boxfusion.online_refinement."
        "propose_depth_occupancy_refinement",
        planar_proposal,
    )
    result = controller.finalize(
        global_corners=np.empty((0, 8, 3), dtype=np.float32),
        global_scores=np.empty(0, dtype=np.float32),
        stable_ids=np.empty(0, dtype=np.int64),
    )

    assert result.stable_ids.tolist() == [-1]
    assert result.refit_reasons == ("c2_planar_accepted",)
    np.testing.assert_array_equal(result.boxes[0], candidate)
    assert (
        np.ptp(result.corners[0], axis=0).min()
        < config["online_refinement"][
            "supplemental_geometry_refiner"
        ]["refined_planar_minimum_extent"]
    )


def test_c2_sink_exact_threshold_survives_corner_round_trip(
    tmp_path,
    monkeypatch,
):
    config = _permissive_c2(
        _config(
            tmp_path,
            "missing_mask_graph_c2_geometry",
        )
    )
    runtime = config["online_refinement"]
    runtime["output_filter"]["final_minimum_extent"] = 0.40
    controller = OnlineRefinementController(
        config,
        provider=_NoopProvider(),
    )
    original = np.asarray(
        [2.0, 0.1, 2.0, 0.20, 0.30, 0.40],
        dtype=np.float32,
    )
    boundary_dims = np.nextafter(
        np.asarray([0.12, 0.20, 0.30], dtype=np.float32),
        np.asarray(np.inf, dtype=np.float32),
    )
    candidate = np.concatenate(
        (
            np.asarray([2.0, 0.1, 2.0], dtype=np.float32),
            boundary_dims,
        )
    )
    _install_confirmed_track(
        controller,
        0,
        original,
        label="sink",
        geometry_points=_inner_grid(candidate),
        projection_box=candidate,
    )

    def solid_proposal(
        original_box,
        geometry_points,
        view_count,
        *,
        full_memory_points=None,
        branch_hint=None,
        config=None,
    ):
        assert branch_hint == "solid"
        return _candidate_proposal(
            candidate, geometry_points, branch="solid"
        )

    monkeypatch.setattr(
        "boxfusion.online_refinement."
        "propose_depth_occupancy_refinement",
        solid_proposal,
    )
    result = controller.finalize(
        global_corners=np.empty((0, 8, 3), dtype=np.float32),
        global_scores=np.empty(0, dtype=np.float32),
        stable_ids=np.empty(0, dtype=np.int64),
    )

    assert result.stable_ids.tolist() == [-1]
    assert result.refit_reasons == ("c2_solid_accepted",)
    np.testing.assert_array_equal(result.boxes[0], candidate)
    assert (
        np.ptp(result.corners[0], axis=0).min()
        < runtime["supplemental_output"]["small_min_extent"]
    )
    assert result.summary["c2_applied"] == 1


def test_c2_rejects_a_duplicate_created_by_an_earlier_refinement(
    tmp_path,
    monkeypatch,
):
    config = _permissive_c2(
        _config(
            tmp_path,
            "missing_mask_graph_c2_geometry",
        )
    )
    controller = OnlineRefinementController(
        config,
        provider=_NoopProvider(),
    )
    first = np.asarray(
        [0.0, 0.0, 2.0, 0.40, 0.40, 0.40],
        dtype=np.float32,
    )
    second = np.asarray(
        [0.8, 0.0, 2.0, 0.40, 0.40, 0.40],
        dtype=np.float32,
    )
    converged = np.asarray(
        [0.4, 0.0, 2.0, 0.40, 0.40, 0.40],
        dtype=np.float32,
    )
    _install_confirmed_track(
        controller,
        0,
        first,
        detector_score=0.80,
        geometry_points=_inner_grid(converged),
        projection_box=converged,
    )
    _install_confirmed_track(
        controller,
        1,
        second,
        detector_score=0.70,
        geometry_points=_inner_grid(converged),
        projection_box=converged,
    )

    def converging_proposal(
        original_box,
        geometry_points,
        view_count,
        *,
        full_memory_points=None,
        branch_hint=None,
        config=None,
    ):
        return _candidate_proposal(
            converged, geometry_points, branch="solid"
        )

    monkeypatch.setattr(
        "boxfusion.online_refinement."
        "propose_depth_occupancy_refinement",
        converging_proposal,
    )
    result = controller.finalize(
        global_corners=np.empty((0, 8, 3), dtype=np.float32),
        global_scores=np.empty(0, dtype=np.float32),
        stable_ids=np.empty(0, dtype=np.int64),
    )

    assert result.stable_ids.tolist() == [-1, -2]
    np.testing.assert_array_equal(result.boxes[0], converged)
    np.testing.assert_array_equal(
        result.boxes[1], result.refit_original_boxes[1]
    )
    assert not np.array_equal(result.boxes[1], converged)
    assert result.refit_reasons == (
        "c2_solid_accepted",
        "supplemental_identity",
    )
    assert controller._last_c2_runtime[-1]["reason"] == "accepted"
    assert (
        controller._last_c2_runtime[-2]["reason"]
        == "structural_supplemental"
    )


def _fragment_snapshot(
    track_id,
    *,
    state,
    frame,
    box,
    views,
    score,
    label="sink",
):
    return {
        "track_id": int(track_id),
        "lifecycle_state": str(state),
        "event_frame": int(frame),
        "box": np.asarray(box, dtype=np.float32),
        "hit_count": int(views),
        "view_count": int(views),
        "track_confirmed": False,
        "node_count": int(views),
        "edge_count": max(int(views) - 1, 0),
        "unique_frame_count": int(views),
        "graph_confirmed": False,
        "confirmation_frame_id": "",
        "mean_edge_score": 0.50,
        "mean_geometry_score": 0.50,
        "mean_iou_3d": 0.50,
        "mean_mutual_inside": 0.50,
        "mean_projection_iou": 0.50,
        "mean_appearance_cosine": 0.80,
        "mean_detector_score": float(score),
        "label": str(label),
        "rejections": {},
        "memory_view_candidates": int(views),
        "memory_selected_views": int(views),
        "memory_geometry_points": 256 * int(views),
    }


def test_c3_stitch_observer_is_c2_output_identity_and_dumps_candidates(
    tmp_path,
):
    global_box = np.asarray(
        [3.0, 0.0, 2.0, 0.60, 0.60, 0.60],
        dtype=np.float32,
    )
    global_corners = aabb_corners(
        global_box[:3], global_box[3:]
    )[None].astype(np.float32)
    global_scores = np.asarray([0.75], dtype=np.float32)
    stable_ids = np.asarray([9], dtype=np.int64)

    c2 = OnlineRefinementController(
        _config(
            tmp_path / "c2",
            "missing_mask_graph_c2_geometry",
        ),
        provider=_NoopProvider(),
    ).finalize(
        global_corners=global_corners,
        global_scores=global_scores,
        stable_ids=stable_ids,
    )

    config = _config(
        tmp_path / "diagnostics",
        "missing_mask_graph_c3_stitch_observer",
    )
    config["online_refinement"]["diagnostics"].update(
        {
            "enabled": True,
            "dump_track_memory": True,
            "root": str(tmp_path / "diagnostics"),
            "point_count": 32,
        }
    )
    controller = OnlineRefinementController(
        config,
        provider=_NoopProvider(),
    )
    invalid_snapshot = _fragment_snapshot(
        2,
        state="discarded",
        frame=20,
        box=[1.0, 0.0, 2.0, 0.50, 0.50, 0.50],
        views=1,
        score=0.90,
    )
    # A sparse ObjectGeometryMemory can legitimately have no AABB after
    # voxel downsampling.  The snapshot then contains NaNs; C3 diagnostics
    # must skip it rather than aborting or changing the frozen C2 result.
    invalid_snapshot["box"] = np.full(6, np.nan, dtype=np.float32)
    malformed_snapshot = _fragment_snapshot(
        3,
        state="discarded",
        frame=30,
        box=[1.5, 0.0, 2.0, 0.50, 0.50, 0.50],
        views=1,
        score=0.90,
    )
    malformed_snapshot["view_count"] = "malformed"
    controller.retired_mask_graph_snapshots.extend(
        [
            _fragment_snapshot(
                0,
                state="discarded",
                frame=0,
                box=[0.0, 0.0, 2.0, 0.50, 0.50, 0.50],
                views=1,
                score=0.90,
            ),
            _fragment_snapshot(
                1,
                state="active",
                frame=10,
                box=[0.05, 0.0, 2.0, 0.50, 0.50, 0.50],
                views=2,
                score=0.80,
            ),
            invalid_snapshot,
            malformed_snapshot,
            _fragment_snapshot(
                4,
                state="discarded",
                frame=0,
                box=[2.00, 0.0, 2.0, 0.50, 0.50, 0.50],
                views=1,
                score=0.90,
                label="table",
            ),
            _fragment_snapshot(
                5,
                state="active",
                frame=10,
                box=[2.05, 0.0, 2.0, 0.50, 0.50, 0.50],
                views=2,
                score=0.82,
                label="table",
            ),
            _fragment_snapshot(
                6,
                state="archived",
                frame=20,
                box=[2.10, 0.0, 2.0, 0.50, 0.50, 0.50],
                views=3,
                score=0.80,
                label="table",
            ),
        ]
    )
    original_snapshot_builder = controller._live_mask_graph_snapshots
    snapshot_calls = 0

    def one_snapshot_pass():
        nonlocal snapshot_calls
        snapshot_calls += 1
        if snapshot_calls > 1:
            raise RuntimeError("C3 rebuilt lifecycle diagnostics")
        return original_snapshot_builder()

    controller._live_mask_graph_snapshots = one_snapshot_pass
    observer = controller.finalize(
        global_corners=global_corners,
        global_scores=global_scores,
        stable_ids=stable_ids,
        scene_id="scene0000_00",
    )

    for name in (
        "corners",
        "boxes",
        "scores",
        "source_indices",
        "stable_ids",
        "quality_features",
        "refit_original_boxes",
        "refit_original_corners",
        "refit_applied",
    ):
        np.testing.assert_array_equal(
            getattr(observer, name), getattr(c2, name)
        )
    assert observer.labels == c2.labels
    assert observer.refit_reasons == c2.refit_reasons
    assert observer.summary["fragment_stitch_enabled"] is True
    assert observer.summary["fragment_stitch_candidates"] == 2
    assert observer.summary["fragment_stitch_invalid_snapshots"] == 2
    assert observer.summary["fragment_stitch_fail_open"] is False
    assert observer.summary["fragment_stitch_error"] == ""
    assert snapshot_calls == 1
    assert (
        observer.summary["mutation_fragment_stitch_enabled"]
        is False
    )

    path = tmp_path / "diagnostics" / "scene0000_00_tracks.npz"
    with np.load(path, allow_pickle=False) as payload:
        assert payload["fragment_stitch_enabled"].item() is True
        assert (
            payload["mutation_fragment_stitch_enabled"].item()
            is False
        )
        assert (
            payload["fragment_stitch_invalid_snapshots"].item() == 2
        )
        assert payload["graph_component_track_ids"].tolist() == [
            0,
            1,
            4,
            5,
            6,
        ]
        assert payload["fragment_stitch_fail_open"].item() is False
        assert payload["fragment_stitch_error"].item() == ""
        assert json.loads(
            payload["fragment_stitch_config_json"].item()
        ) == config["online_refinement"]["fragment_stitch"]
        assert payload["fragment_stitch_diagnostics_schema"].item() == (
            "mask_graph_fragment_stitch_v2"
        )
        assert payload[
            "fragment_stitch_candidate_track_ids"
        ].tolist() == [[0, 1, -1], [4, 5, 6]]
        assert payload[
            "fragment_stitch_candidate_track_mask"
        ].tolist() == [[True, True, False], [True, True, True]]
        assert payload[
            "fragment_stitch_candidate_representative_track_ids"
        ].tolist() == [1, 5]
        assert payload[
            "fragment_stitch_candidate_labels"
        ].tolist() == ["sink", "table"]
        assert payload[
            "fragment_stitch_candidate_event_frames"
        ].tolist() == [[0, 10, -1], [0, 10, 20]]
        assert payload[
            "fragment_stitch_candidate_states_json"
        ].tolist() == [
            "[\"discarded\",\"active\"]",
            "[\"discarded\",\"active\",\"archived\"]",
        ]
        assert payload[
            "fragment_stitch_candidate_boxes"
        ].shape == (2, 6)
        for name in (
            "fragment_stitch_candidate_min_pair_iou",
            "fragment_stitch_candidate_min_pair_containment",
            "fragment_stitch_candidate_max_pair_center_distance",
            "fragment_stitch_candidate_max_detector_score",
            "fragment_stitch_candidate_mean_detector_score",
        ):
            assert payload[name].shape == (2,)


def test_c3_stitch_empty_diagnostics_have_stable_zero_shapes(tmp_path):
    config = _config(
        tmp_path / "empty_diagnostics",
        "missing_mask_graph_c3_stitch_observer",
    )
    config["online_refinement"]["diagnostics"].update(
        {
            "enabled": True,
            "dump_track_memory": True,
            "root": str(tmp_path / "empty_diagnostics"),
            "point_count": 32,
        }
    )
    box = np.asarray(
        [3.0, 0.0, 2.0, 0.60, 0.60, 0.60],
        dtype=np.float32,
    )
    controller = OnlineRefinementController(
        config,
        provider=_NoopProvider(),
    )
    result = controller.finalize(
        global_corners=aabb_corners(
            box[:3], box[3:]
        )[None].astype(np.float32),
        global_scores=np.asarray([0.75], dtype=np.float32),
        stable_ids=np.asarray([9], dtype=np.int64),
        scene_id="scene0000_01",
    )

    assert result.summary["fragment_stitch_candidates"] == 0
    path = (
        tmp_path
        / "empty_diagnostics"
        / "scene0000_01_tracks.npz"
    )
    with np.load(path, allow_pickle=False) as payload:
        assert payload[
            "fragment_stitch_candidate_track_ids"
        ].shape == (0, 0)
        assert payload[
            "fragment_stitch_candidate_track_mask"
        ].shape == (0, 0)
        assert payload[
            "fragment_stitch_candidate_event_frames"
        ].shape == (0, 0)
        assert payload[
            "fragment_stitch_candidate_boxes"
        ].shape == (0, 6)
        assert payload[
            "fragment_stitch_candidate_labels"
        ].shape == (0,)
        assert payload["graph_component_boxes"].shape == (0, 6)
