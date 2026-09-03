import json
import pickle

import numpy as np
import pytest

from boxfusion.oriented_box_refiner import OrientedBoxRefinerConfig
from boxfusion.online_refinement import (
    OnlineRefinementController,
    ViewEvidence,
)
from boxfusion.quality_score import QUALITY_FEATURE_NAMES
from tools.build_oriented_refiner_dataset import (
    BuildConfig,
    STRICT_PROVENANCE_EXPECTED,
    TARGET_LINE_SEARCH_ALPHAS,
    V2_METADATA_KEYS,
    V2_SAMPLE_KEYS,
    _local_box_to_world_corners,
    _projection_iou_for_corners,
    _scene_training_arrays,
    build_oriented_refiner_dataset,
    greedy_scene_tp50_flags,
    load_scene_diagnostics,
    runtime_refit_gate,
)
from tools.train_oriented_box_refiner import (
    differentiable_aligned_aabb_iou,
    load_oriented_refiner_dataset,
    oriented_refiner_loss,
    train_oriented_box_refiner,
)


def _corners(center, dimensions):
    signs = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]
    )
    return np.asarray(center) + signs * (0.5 * np.asarray(dimensions))


def _provenance_payload():
    payload = {}
    for name, value in STRICT_PROVENANCE_EXPECTED.items():
        if isinstance(value, bool):
            payload[name] = np.asarray(value, dtype=np.bool_)
        elif isinstance(value, int):
            payload[name] = np.asarray(value, dtype=np.int64)
        elif isinstance(value, float):
            payload[name] = np.asarray(value, dtype=np.float64)
        else:
            payload[name] = np.asarray(value)
    payload["summary_json"] = np.asarray(
        json.dumps(STRICT_PROVENANCE_EXPECTED, sort_keys=True)
    )
    return payload


def _diagnostic_payload(scene="scene0000_00", observations=1):
    points = np.zeros((observations, 512, 3), dtype=np.float32)
    point_mask = np.zeros((observations, 512), dtype=np.bool_)
    point_mask[:, :128] = True
    gate_points = np.zeros(
        (observations, 8192, 3), dtype=np.float32
    )
    gate_mask = np.zeros((observations, 8192), dtype=np.bool_)
    gate_mask[:, :256] = True
    local_boxes = np.zeros((observations, 6), dtype=np.float32)
    local_boxes[:, 3:6] = 1.0
    frame_centers = np.zeros((observations, 3), dtype=np.float64)
    frame_centers[:, 2] = 3.0
    frame_basis = np.tile(
        np.eye(3, dtype=np.float64), (observations, 1, 1)
    )
    view_valid = np.zeros((observations, 5), dtype=np.bool_)
    view_valid[:, :2] = True
    view_frame_ids = np.full((observations, 5), -1, dtype=np.int64)
    view_frame_ids[:, :2] = np.asarray([0, 1], dtype=np.int64)
    view_scores = np.full(
        (observations, 5), np.nan, dtype=np.float32
    )
    view_scores[:, :2] = np.asarray([0.9, 0.8], dtype=np.float32)
    view_bboxes = np.full(
        (observations, 5, 4), np.nan, dtype=np.float32
    )
    view_bboxes[:, :2] = np.asarray(
        [42.0, 38.0, 66.0, 62.0], dtype=np.float32
    )
    intrinsics = np.asarray(
        [[60.0, 0.0, 50.0], [0.0, 60.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    view_intrinsics = np.full(
        (observations, 5, 3, 3), np.nan, dtype=np.float32
    )
    view_intrinsics[:, :2] = intrinsics
    view_poses = np.full(
        (observations, 5, 4, 4), np.nan, dtype=np.float32
    )
    view_poses[:, :2] = np.eye(4, dtype=np.float32)
    image_shapes = np.full(
        (observations, 5, 2), -1, dtype=np.int64
    )
    image_shapes[:, :2] = np.asarray([100, 100], dtype=np.int64)
    payload = {
        "scene_id": np.asarray(scene),
        "quality_features": np.full(
            (observations, 12), 0.5, dtype=np.float32
        ),
        "quality_feature_names": np.asarray(QUALITY_FEATURE_NAMES),
        "result_indices": np.arange(observations, dtype=np.int64),
        "track_ids": np.arange(7, 7 + observations, dtype=np.int64),
        "box_refiner_points_local": points,
        "box_refiner_point_mask": point_mask,
        "box_refiner_local_boxes": local_boxes,
        "box_refiner_frame_valid": np.ones(
            observations, dtype=np.bool_
        ),
        "box_refiner_gate_points_local": gate_points,
        "box_refiner_gate_point_mask": gate_mask,
        "box_refiner_frame_centers": frame_centers,
        "box_refiner_frame_basis": frame_basis,
        "box_refiner_view_valid": view_valid,
        "box_refiner_view_frame_ids": view_frame_ids,
        "box_refiner_view_scores": view_scores,
        "box_refiner_view_bboxes": view_bboxes,
        "box_refiner_view_intrinsics": view_intrinsics,
        "box_refiner_view_camera_to_world": view_poses,
        "box_refiner_view_image_shapes": image_shapes,
        "selected_view_counts": np.full(
            observations, 2, dtype=np.int64
        ),
        "selected_view_frame_ids": view_frame_ids.copy(),
        "top_k_view_valid": view_valid.copy(),
    }
    payload.update(_provenance_payload())
    return payload


def test_strict_k5_rejects_legacy_fallback_and_wrong_k(tmp_path):
    legacy = tmp_path / "legacy.npz"
    payload = {
        "scene_id": np.asarray("scene0000_00"),
        "quality_features": np.full((1, 12), 0.5, dtype=np.float32),
        "result_indices": np.asarray([0], dtype=np.int64),
        "points": np.zeros((1, 128, 3), dtype=np.float32),
        "point_mask": np.ones((1, 128), dtype=np.bool_),
    }
    np.savez(legacy, **payload)
    with pytest.raises(ValueError, match="missing fields"):
        load_scene_diagnostics(
            legacy,
            objective="improvement",
            strict_k5_diagnostics=True,
        )

    wrong_k = tmp_path / "wrong_k.npz"
    wrong_payload = _diagnostic_payload()
    wrong_payload["top_k_views"] = np.asarray(4, dtype=np.int64)
    np.savez(wrong_k, **wrong_payload)
    with pytest.raises(ValueError, match="top_k_views"):
        load_scene_diagnostics(
            wrong_k,
            objective="improvement",
            strict_k5_diagnostics=True,
        )


def test_strict_k5_uses_exact_local_inputs_and_checks_sentinels(tmp_path):
    path = tmp_path / "tracks.npz"
    payload = _diagnostic_payload()
    payload["box_refiner_points_local"][:, :128] = 0.125
    np.savez(path, **payload)
    loaded = load_scene_diagnostics(
        path,
        strict_k5_diagnostics=True,
    )
    np.testing.assert_array_equal(
        loaded.points[:, :128], 0.125
    )
    assert loaded.points.shape == (1, 512, 3)
    assert loaded.top_k_views == 5
    assert loaded.gate_points_local.shape == (1, 8192, 3)

    payload["box_refiner_view_scores"][0, 4] = 0.0
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="sentinel"):
        load_scene_diagnostics(path, strict_k5_diagnostics=True)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("online_ablation_profile", np.asarray("b5v2_refiner_only")),
        ("candidate_track_ttl", np.asarray(4, dtype=np.int64)),
        ("archive_confirmed_tracks", np.asarray(True, dtype=np.bool_)),
        ("mutation_quality_enabled", np.asarray(True, dtype=np.bool_)),
        ("output_minimum_extent", np.asarray(0.30, dtype=np.float64)),
    ],
)
def test_strict_k5_rejects_wrong_runtime_provenance(
    tmp_path, field, bad_value
):
    path = tmp_path / f"{field}.npz"
    payload = _diagnostic_payload()
    payload[field] = bad_value
    np.savez(path, **payload)
    with pytest.raises(ValueError, match=field):
        load_scene_diagnostics(path, strict_k5_diagnostics=True)


def test_strict_k5_rejects_summary_or_camera_semantic_drift(tmp_path):
    path = tmp_path / "tracks.npz"
    payload = _diagnostic_payload()
    summary = dict(STRICT_PROVENANCE_EXPECTED)
    summary["candidate_ttl_clock"] = "keyframe"
    payload["summary_json"] = np.asarray(json.dumps(summary))
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="summary_json provenance"):
        load_scene_diagnostics(path, strict_k5_diagnostics=True)

    payload = _diagnostic_payload()
    payload["box_refiner_view_camera_to_world"][0, 0, 3, 3] = 0.0
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="homogeneous"):
        load_scene_diagnostics(path, strict_k5_diagnostics=True)


def test_ap50_builder_rejects_forbidden_scene_by_content(tmp_path):
    roots = [tmp_path / name for name in ("diag", "pred", "scan", "gt")]
    for root in roots:
        root.mkdir()
    scene_list = tmp_path / "looks_like_train.txt"
    forbidden = tmp_path / "official_val.txt"
    scene_list.write_text("scene0000_00\n")
    forbidden.write_text("scene0000_00\n")
    with pytest.raises(ValueError, match="overlaps forbidden"):
        build_oriented_refiner_dataset(
            BuildConfig(
                diagnostics_root=roots[0],
                prediction_root=roots[1],
                scan_root=roots[2],
                gt_root=roots[3],
                scene_list=scene_list,
                output=tmp_path / "out.npz",
                objective="ap50",
                forbidden_scene_list=forbidden,
            )
        )


def test_ap50_targets_encode_gain_cross_near_and_soft_quality(tmp_path):
    path = tmp_path / "tracks.npz"
    np.savez(path, **_diagnostic_payload())
    diagnostics = load_scene_diagnostics(
        path, objective="ap50", strict_k5_diagnostics=True
    )
    prediction = _corners(
        [0.0, 0.0, 3.0], [1.0, 1.0, 1.0]
    )[None]
    gt = np.asarray([[0.4, 0.0, 3.0, 1.0, 1.0, 1.0]])
    config = BuildConfig(
        diagnostics_root=".",
        prediction_root=".",
        scan_root=".",
        gt_root=".",
        scene_list="unused",
        output="unused",
        objective="ap50",
        runtime_minimum_extent=0.40,
    )
    arrays, invalid = _scene_training_arrays(
        diagnostics,
        prediction,
        np.eye(4),
        gt,
        config,
        prediction_scores=np.asarray([0.9]),
    )
    assert invalid == 0
    assert arrays["runtime_eligible"].tolist() == [True]
    assert arrays["geometry_mask"].tolist() == [True]
    assert arrays["cross_iou50"].tolist() == [True]
    assert arrays["original_iou"][0] < 0.50
    assert arrays["refined_iou"][0] >= 0.50
    assert arrays["iou_gain"][0] == pytest.approx(
        arrays["refined_iou"][0] - arrays["original_iou"][0]
    )
    assert 0.0 < arrays["near_iou50"][0] <= 1.0
    assert arrays["ap50_weight"][0] > 5.0
    assert arrays["quality_target"][0] == pytest.approx(0.95)
    np.testing.assert_allclose(arrays["aligned_basis"][0], np.eye(3))
    np.testing.assert_allclose(
        arrays["original_aligned_center"][0], [0.0, 0.0, 3.0]
    )
    np.testing.assert_allclose(arrays["matched_gt_box"][0], gt[0])
    assert tuple(TARGET_LINE_SEARCH_ALPHAS) == (0.25, 0.5, 0.75, 1.0)


def _gate_kwargs(diagnostics, row=0):
    return {
        "gate_points_local": diagnostics.gate_points_local[row],
        "gate_point_mask": diagnostics.gate_point_mask[row],
        "frame_center": diagnostics.frame_centers[row],
        "frame_basis": diagnostics.frame_basis[row],
        "selected_view_frame_ids": (
            diagnostics.selected_view_frame_ids[row]
        ),
        "top_k_view_valid": diagnostics.top_k_view_valid[row],
        "view_valid": diagnostics.view_valid[row],
        "view_frame_ids": diagnostics.view_frame_ids[row],
        "view_scores": diagnostics.view_scores[row],
        "view_bboxes": diagnostics.view_bboxes[row],
        "view_intrinsics": diagnostics.view_intrinsics[row],
        "view_camera_to_world": (
            diagnostics.view_camera_to_world[row]
        ),
        "view_image_shapes": diagnostics.view_image_shapes[row],
    }


def test_runtime_gate_uses_memory_k_and_filters_evidence_records(tmp_path):
    path = tmp_path / "tracks.npz"
    payload = _diagnostic_payload()
    # EvidenceStats has two valid records, but neither belongs to the memory
    # Top-K selected frame ids.  View eligibility must still use memory K=2,
    # while reprojection must see zero selected records and reject.
    payload["box_refiner_view_frame_ids"][0, :2] = [8, 9]
    np.savez(path, **payload)
    diagnostics = load_scene_diagnostics(
        path, strict_k5_diagnostics=True
    )
    original = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    accepted, reason, _ = runtime_refit_gate(
        original,
        original.copy(),
        config=BuildConfig(
            diagnostics_root=".",
            prediction_root=".",
            scan_root=".",
            gt_root=".",
            scene_list="unused",
            output="unused",
            strict_k5_diagnostics=True,
        ),
        **_gate_kwargs(diagnostics),
    )
    assert accepted is False
    assert reason == "reprojection"

    payload = _diagnostic_payload()
    payload["selected_view_counts"][0] = 1
    payload["top_k_view_valid"][0, 1] = False
    payload["selected_view_frame_ids"][0, 1] = -1
    np.savez(path, **payload)
    diagnostics = load_scene_diagnostics(
        path, strict_k5_diagnostics=True
    )
    accepted, reason, _ = runtime_refit_gate(
        original,
        original.copy(),
        config=BuildConfig(
            diagnostics_root=".",
            prediction_root=".",
            scan_root=".",
            gt_root=".",
            scene_list="unused",
            output="unused",
            strict_k5_diagnostics=True,
        ),
        **_gate_kwargs(diagnostics),
    )
    assert accepted is False
    assert reason == "views"


def test_runtime_gate_preserves_duplicate_evidence_records(tmp_path):
    path = tmp_path / "tracks.npz"
    payload = _diagnostic_payload()
    # EvidenceStats is a bounded list of observations, not a frame-keyed
    # mapping. Runtime merging can therefore retain two records from the same
    # frame, and the offline gate must weight both exactly as runtime does.
    payload["box_refiner_view_frame_ids"][0, :2] = [0, 0]
    np.savez(path, **payload)
    diagnostics = load_scene_diagnostics(
        path, strict_k5_diagnostics=True
    )
    assert diagnostics.view_frame_ids[0, :2].tolist() == [0, 0]

    original = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    accepted, reason, _ = runtime_refit_gate(
        original,
        original.copy(),
        config=BuildConfig(
            diagnostics_root=".",
            prediction_root=".",
            scan_root=".",
            gt_root=".",
            scene_list="unused",
            output="unused",
            strict_k5_diagnostics=True,
        ),
        **_gate_kwargs(diagnostics),
    )
    assert accepted is True
    assert reason == "accepted"


def test_offline_corner_projection_matches_runtime_float_path():
    local_box = np.asarray(
        [0.013, -0.027, 0.019, 1.137, 0.923, 0.811],
        dtype=np.float32,
    )
    angle = np.deg2rad(17.0)
    basis = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    center = np.asarray([0.07, -0.11, 3.23], dtype=np.float64)
    corners = _local_box_to_world_corners(local_box, center, basis)
    assert corners.dtype == np.float32

    intrinsics = np.asarray(
        [[63.7, 0.0, 49.3], [0.0, 61.9, 51.1], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = np.asarray([0.03, -0.02, 0.01], dtype=np.float32)
    bbox = np.asarray([37.2, 39.4, 63.1, 61.7], dtype=np.float32)
    view = ViewEvidence(
        frame_index=4,
        score=0.83,
        bbox=bbox,
        intrinsics=intrinsics,
        camera_to_world=pose,
        image_shape=(100, 100),
        area_ratio=0.1,
    )
    online = OnlineRefinementController._projection_iou_for_corners(
        corners, view
    )
    offline = _projection_iou_for_corners(
        corners, bbox, intrinsics, pose, np.asarray([100, 100])
    )
    assert offline == online


def test_runtime_gate_enforces_original_and_candidate_point_support(tmp_path):
    path = tmp_path / "tracks.npz"
    payload = _diagnostic_payload()
    payload["box_refiner_gate_points_local"][0, :256, 0] = 2.0
    np.savez(path, **payload)
    diagnostics = load_scene_diagnostics(
        path, strict_k5_diagnostics=True
    )
    original = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    config = BuildConfig(
        diagnostics_root=".",
        prediction_root=".",
        scan_root=".",
        gt_root=".",
        scene_list="unused",
        output="unused",
        strict_k5_diagnostics=True,
    )
    accepted, reason, _ = runtime_refit_gate(
        original,
        original.copy(),
        config=config,
        **_gate_kwargs(diagnostics),
    )
    assert accepted is False
    assert reason == "support"

    payload = _diagnostic_payload()
    payload["box_refiner_gate_points_local"][0, :256, 0] = -0.45
    np.savez(path, **payload)
    diagnostics = load_scene_diagnostics(
        path, strict_k5_diagnostics=True
    )
    candidate = np.asarray([0.15, 0.0, 0.0, 1.0, 1.0, 1.0])
    accepted, reason, _ = runtime_refit_gate(
        original,
        candidate,
        config=config,
        **_gate_kwargs(diagnostics),
    )
    assert accepted is False
    assert reason == "candidate_support"


def test_line_search_keeps_smaller_candidate_when_full_target_fails_gate(
    tmp_path,
):
    path = tmp_path / "tracks.npz"
    payload = _diagnostic_payload()
    payload["box_refiner_local_boxes"][0, 3:6] = 0.45
    # The mask bbox favours shrinkage. Alpha >= .75 drops the candidate world
    # extent below 0.40, while alpha=.50 remains output-surviving.
    payload["box_refiner_view_bboxes"][0, :2] = np.asarray(
        [46.4, 46.4, 53.6, 53.6], dtype=np.float32
    )
    np.savez(path, **payload)
    diagnostics = load_scene_diagnostics(
        path, objective="ap50", strict_k5_diagnostics=True
    )
    prediction = _corners(
        [0.0, 0.0, 3.0], [0.45, 0.45, 0.45]
    )[None]
    gt = np.asarray([[0.0, 0.0, 3.0, 0.30, 0.30, 0.30]])
    config = BuildConfig(
        diagnostics_root=".",
        prediction_root=".",
        scan_root=".",
        gt_root=".",
        scene_list="unused",
        output="unused",
        objective="ap50",
    )
    full_candidate = np.asarray(
        [0.0, 0.0, 0.0, 0.36, 0.36, 0.36]
    )
    accepted, reason, _ = runtime_refit_gate(
        diagnostics.local_boxes[0],
        full_candidate,
        config=config,
        **_gate_kwargs(diagnostics),
    )
    assert accepted is False
    assert reason == "extent_filter"

    arrays, _ = _scene_training_arrays(
        diagnostics,
        prediction,
        np.eye(4),
        gt,
        config,
        prediction_scores=np.asarray([0.9]),
    )
    assert arrays["runtime_eligible"].tolist() == [True]
    # raw ratio clips to .8; alpha .5 is the largest passing step.
    np.testing.assert_allclose(
        arrays["target_residual"][0, 3:6],
        0.5 * np.log(0.8),
        atol=2e-6,
    )
    assert arrays["refined_iou"][0] > arrays["original_iou"][0]


def test_scene_aware_tp50_does_not_retry_second_best_or_mark_duplicate_cross(
    tmp_path,
):
    boxes = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ]
    )
    targets = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.3, 0.0, 0.0, 1.0, 1.0, 1.0],
        ]
    )
    assert greedy_scene_tp50_flags(
        boxes, np.asarray([0.9, 0.8]), targets
    ).tolist() == [True, False]

    path = tmp_path / "tracks.npz"
    payload = _diagnostic_payload(observations=2)
    payload["box_refiner_frame_centers"][1, 0] = -0.4
    np.savez(path, **payload)
    diagnostics = load_scene_diagnostics(
        path, objective="ap50", strict_k5_diagnostics=True
    )
    predictions = np.stack(
        (
            _corners([0.0, 0.0, 3.0], [1.0, 1.0, 1.0]),
            _corners([-0.4, 0.0, 3.0], [1.0, 1.0, 1.0]),
        )
    )
    gt = np.asarray([[0.0, 0.0, 3.0, 1.0, 1.0, 1.0]])
    arrays, _ = _scene_training_arrays(
        diagnostics,
        predictions,
        np.eye(4),
        gt,
        BuildConfig(
            diagnostics_root=".",
            prediction_root=".",
            scan_root=".",
            gt_root=".",
            scene_list="unused",
            output="unused",
            objective="ap50",
        ),
        prediction_scores=np.asarray([0.9, 0.8]),
    )
    assert arrays["identity_tp50"].tolist() == [True, False]
    assert arrays["candidate_oracle_tp50"].tolist() == [True, False]
    assert arrays["cross_iou50"].tolist() == [False, False]


def test_invalid_runtime_frames_and_empty_model_masks_are_not_training_rows(
    tmp_path,
):
    path = tmp_path / "tracks.npz"
    payload = _diagnostic_payload(observations=2)
    payload["box_refiner_frame_valid"][1] = False
    payload["box_refiner_point_mask"][1] = False
    payload["box_refiner_gate_point_mask"][1] = False
    payload["box_refiner_local_boxes"][1] = np.nan
    payload["box_refiner_frame_centers"][1] = np.nan
    payload["box_refiner_frame_basis"][1] = np.nan
    np.savez(path, **payload)
    diagnostics = load_scene_diagnostics(
        path, objective="ap50", strict_k5_diagnostics=True
    )
    predictions = np.stack(
        (
            _corners([0.0, 0.0, 3.0], [1.0, 1.0, 1.0]),
            _corners([2.0, 0.0, 3.0], [1.0, 1.0, 1.0]),
        )
    )
    arrays, invalid = _scene_training_arrays(
        diagnostics,
        predictions,
        np.eye(4),
        np.asarray([[0.4, 0.0, 3.0, 1.0, 1.0, 1.0]]),
        BuildConfig(
            diagnostics_root=".",
            prediction_root=".",
            scan_root=".",
            gt_root=".",
            scene_list="unused",
            output="unused",
            objective="ap50",
        ),
        prediction_scores=np.asarray([0.9, 0.8]),
    )
    assert invalid == 1
    assert arrays["result_indices"].tolist() == [0]
    assert arrays["point_mask"].any(axis=1).tolist() == [True]


def test_differentiable_aligned_iou_matches_identity_and_crosses_half():
    torch = pytest.importorskip("torch")
    center = torch.zeros((1, 3), requires_grad=True)
    dimensions = torch.zeros((1, 3), requires_grad=True)
    output = {
        "center_residual_fraction": center,
        "log_dimension_residual": dimensions,
    }
    boxes = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    basis = torch.eye(3).unsqueeze(0)
    original_center = torch.zeros((1, 3))
    gt = torch.tensor([[0.4, 0.0, 0.0, 1.0, 1.0, 1.0]])
    identity_iou = differentiable_aligned_aabb_iou(
        output, boxes, basis, original_center, gt
    )
    assert identity_iou.item() == pytest.approx(0.6 / 1.4, rel=1e-6)
    identity_iou.sum().backward()
    assert torch.isfinite(center.grad).all()
    assert center.grad[0, 0] != 0.0

    crossed = differentiable_aligned_aabb_iou(
        {
            "center_residual_fraction": torch.tensor([[0.15, 0.0, 0.0]]),
            "log_dimension_residual": torch.zeros((1, 3)),
        },
        boxes,
        basis,
        original_center,
        gt,
    )
    assert crossed.item() == pytest.approx(0.75 / 1.25, rel=1e-6)
    assert crossed.item() >= 0.50


def test_ap50_loss_prioritizes_cross_and_keeps_negative_geometry_masked():
    torch = pytest.importorskip("torch")
    boxes = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ]
    )
    basis = torch.eye(3).repeat(2, 1, 1)
    original_center = torch.zeros((2, 3))
    gt = torch.tensor(
        [
            [0.4, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.4, 0.0, 0.0, 1.0, 1.0, 1.0],
        ]
    )
    center = torch.zeros((2, 3), requires_grad=True)
    log_dimensions = torch.zeros((2, 3), requires_grad=True)
    quality_logits = torch.tensor([2.0, -2.0], requires_grad=True)
    common = dict(
        target_residual=torch.tensor(
            [[0.15, 0.0, 0.0, 0.0, 0.0, 0.0], [9.0] * 6]
        ),
        quality_target=torch.tensor([0.95, 0.0]),
        geometry_mask=torch.tensor([True, False]),
        objective="ap50",
        local_boxes=boxes,
        original_iou=torch.tensor([0.6 / 1.4, 0.6 / 1.4]),
        aligned_basis=basis,
        original_aligned_center=original_center,
        matched_gt_box=gt,
        iou_gain_target=torch.tensor([0.6 - 0.6 / 1.4, 0.0]),
        cross_iou50=torch.tensor([True, False]),
        ap50_weight=torch.tensor([8.0, 1.0]),
    )
    loss, metrics = oriented_refiner_loss(
        {
            "center_residual_fraction": center,
            "log_dimension_residual": log_dimensions,
            "quality": torch.sigmoid(quality_logits),
        },
        **common,
    )
    assert metrics["cross_iou50_loss"] > 0.0
    loss.backward()
    np.testing.assert_array_equal(center.grad[1].detach().numpy(), 0.0)
    np.testing.assert_array_equal(
        log_dimensions.grad[1].detach().numpy(), 0.0
    )
    assert quality_logits.grad[1] != 0.0

    _, crossed_metrics = oriented_refiner_loss(
        {
            "center_residual_fraction": torch.tensor(
                [[0.15, 0.0, 0.0], [0.0, 0.0, 0.0]]
            ),
            "log_dimension_residual": torch.zeros((2, 3)),
            "quality": torch.tensor([0.95, 0.1]),
        },
        **common,
    )
    assert crossed_metrics["cross_iou50_loss"] == pytest.approx(0.0)
    assert crossed_metrics["cross50_success_count"] == 1


def test_ap50_schema_v2_build_load_and_cpu_train_are_scene_safe(tmp_path):
    pytest.importorskip("torch")
    diagnostics_root = tmp_path / "diagnostics"
    prediction_root = tmp_path / "predictions"
    scan_root = tmp_path / "scans"
    gt_root = tmp_path / "gt"
    for root in (
        diagnostics_root,
        prediction_root,
        scan_root,
        gt_root,
    ):
        root.mkdir()
    scenes = ["scene0000_00", "scene0001_00"]
    scene_list = tmp_path / "train.txt"
    scene_list.write_text("\n".join(scenes) + "\n")
    forbidden = tmp_path / "official_val.txt"
    forbidden.write_text("scene0700_00\n")

    for scene in scenes:
        scene_scan_root = scan_root / scene
        scene_scan_root.mkdir()
        identity_values = " ".join(
            str(value) for value in np.eye(4).reshape(-1)
        )
        (scene_scan_root / f"{scene}.txt").write_text(
            f"axisAlignment = {identity_values}\n"
        )
        np.save(
            gt_root / f"{scene}_bbox.npy",
            np.asarray([[0.4, 0.0, 3.0, 1.0, 1.0, 1.0, 3.0]]),
        )
        positive = _corners([0.0, 0.0, 3.0], [1.0, 1.0, 1.0])
        negative = _corners([10.0, 0.0, 3.0], [1.0, 1.0, 1.0])
        with (prediction_root / f"{scene}_boxes.pkl").open("wb") as handle:
            pickle.dump(
                [[(0, positive, 0.9), (0, negative, 0.2)]],
                handle,
            )

        duplicated = _diagnostic_payload(scene, observations=2)
        duplicated["box_refiner_frame_centers"][1, 0] = 10.0
        np.savez(
            diagnostics_root / f"{scene}_tracks.npz", **duplicated
        )

    dataset = tmp_path / "ap50.npz"
    summary = build_oriented_refiner_dataset(
        BuildConfig(
            diagnostics_root=diagnostics_root,
            prediction_root=prediction_root,
            scan_root=scan_root,
            gt_root=gt_root,
            scene_list=scene_list,
            output=dataset,
            objective="ap50",
            forbidden_scene_list=forbidden,
        )
    )
    assert summary.samples == 4
    assert summary.geometry_positives == 2
    assert summary.cross_iou50_positives == 2
    with np.load(dataset, allow_pickle=False) as archive:
        assert set(archive.files) == set(
            V2_SAMPLE_KEYS | V2_METADATA_KEYS
        )
        assert str(np.asarray(archive["objective"]).item()) == "ap50"
        assert bool(np.asarray(archive["strict_k5_diagnostics"]))
        np.testing.assert_array_equal(
            archive["target_line_search_alphas"],
            np.asarray(TARGET_LINE_SEARCH_ALPHAS, dtype=np.float32),
        )
        assert len(str(archive["training_scene_sha256"].item())) == 64
        assert len(str(archive["forbidden_scene_sha256"].item())) == 64
    data = load_oriented_refiner_dataset(dataset)
    assert data.objective == "ap50"
    assert data.cross_iou50.tolist() == [True, False, True, False]

    strict_improvement_dataset = tmp_path / "strict_improvement.npz"
    strict_summary = build_oriented_refiner_dataset(
        BuildConfig(
            diagnostics_root=diagnostics_root,
            prediction_root=prediction_root,
            scan_root=scan_root,
            gt_root=gt_root,
            scene_list=scene_list,
            output=strict_improvement_dataset,
            objective="improvement",
            strict_k5_diagnostics=True,
            forbidden_scene_list=forbidden,
        )
    )
    assert strict_summary.samples == 4
    strict_data = load_oriented_refiner_dataset(
        strict_improvement_dataset
    )
    assert strict_data.objective == "improvement"
    assert np.isin(strict_data.quality_target, (0.0, 1.0)).all()

    checkpoint = tmp_path / "ap50.pt"
    result = train_oriented_box_refiner(
        dataset,
        checkpoint,
        objective="ap50",
        config=OrientedBoxRefinerConfig(
            point_hidden_dim=8,
            point_embedding_dim=8,
            head_hidden_dim=8,
        ),
        epochs=2,
        batch_size=2,
        validation_fraction=0.5,
        seed=11,
    )
    assert checkpoint.is_file()
    assert result["objective"] == "ap50"
    assert result["scene_leakage"] is False
    assert not (
        set(result["train_scenes"]) & set(result["validation_scenes"])
    )
    assert np.isfinite(result["best_validation_ap50_proxy"])
