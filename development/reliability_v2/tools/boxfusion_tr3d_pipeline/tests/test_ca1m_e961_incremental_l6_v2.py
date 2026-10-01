from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from boxfusion.ca1m_e961_incremental_l6_v2 import (
    FEATURE_NAMES,
    INCREMENTAL_OBSERVER_CONFIG,
    LIGHTWEIGHT_FUSION_CONFIG,
    TERMINAL_PASS_STATUS,
    TERMINAL_SCIENTIFIC_STOP_STATUS,
    assign_low_scores,
    post_terminal_gt_coverage,
    select_stage6_candidates,
    terminal_selection_for_state,
    threshold_grid,
    track_targets,
    validate_fold234_oof_plan,
    validate_r4_gate_oof_scoring,
    _aabb_iou_one_to_many,
)
from boxfusion.ca1m_e961_incremental_provider_v2 import (
    CA1ME961IncrementalProviderV2,
)
from boxfusion.tr3d_incremental_online import IncrementalTR3DConfig
from boxfusion.tr3d_incremental_online import _aabb_iou as pinned_incremental_aabb_iou
from boxfusion.tr3d_lightweight_fusion import (
    LightweightAsyncTR3DObserver,
    LightweightFusionConfig,
)
from tools.preflight_ca1m_e961_incremental_l6_v2 import (
    DEFAULT_CONFIG,
    validate_static_config,
)


ROOT = Path(__file__).resolve().parents[1]


class _UnusedProvider:
    def infer(self, **kwargs):  # pragma: no cover - association is tested directly
        raise AssertionError("provider must not run in this synthetic test")


def _corners(center: tuple[float, float, float]) -> np.ndarray:
    value = np.asarray(center, np.float32)
    offsets = np.asarray([
        [-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5],
        [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5],
        [0.5, -0.5, -0.5], [0.5, -0.5, 0.5],
        [0.5, 0.5, -0.5], [0.5, 0.5, 0.5],
    ], np.float32)
    return value + offsets


def test_static_preflight_is_pending_and_opens_no_dynamic_gt_gpu():
    report = validate_static_config(DEFAULT_CONFIG)
    assert report["static_contract_ready"] is True
    assert report["dynamic_prerequisites_complete"] is False
    assert report["run_authorized"] is False
    assert report["candidate_universe"] == "causal_lightweight_async_stage6_confirmed_tracks"
    assert report["one_shot_P_used_as_complete_l6_universe"] is False
    assert report["r4_near_collection_used_as_complete_l6_universe"] is False
    assert report["terminal_scientific_pass_allowed"] is True
    assert report["terminal_scientific_stop_allowed"] is True
    assert report["terminal_provenance_failure_allowed"] is False
    assert report["dynamic_artifacts_opened"] is False
    assert report["ground_truth_files_opened"] is False
    assert report["fold1_path_or_loader_opened"] is False
    assert report["official_validation_path_or_loader_opened"] is False
    assert report["gpu_started"] is False
    assert report["model_started"] is False


def test_original_incremental_observer_parameters_are_frozen_exactly():
    actual = IncrementalTR3DConfig()
    assert {
        name: getattr(actual, name) for name in INCREMENTAL_OBSERVER_CONFIG
    } == INCREMENTAL_OBSERVER_CONFIG
    lightweight = LightweightFusionConfig()
    assert {
        name: getattr(lightweight, name) for name in LIGHTWEIGHT_FUSION_CONFIG
    } == LIGHTWEIGHT_FUSION_CONFIG


def test_dual_head_feature_order_and_exact_threshold_grid_are_frozen():
    assert len(FEATURE_NAMES) == 23
    assert FEATURE_NAMES[:3] == ("best_score", "score_mean", "score_std")
    assert FEATURE_NAMES[11:14] == (
        "post_terminal_anchor_iou_max",
        "post_terminal_anchor_center_distance_m",
        "matched_post_terminal_anchor_score",
    )
    assert FEATURE_NAMES[-5:] == (
        "visibility_quality_mean", "support_ratio_mean", "free_space_ratio_mean",
        "invalid_ratio_mean", "selected_geometry_fused",
    )
    grid = threshold_grid()
    assert len(grid) == 181
    assert grid[0] == pytest.approx(0.05)
    assert grid[-1] == pytest.approx(0.95)
    assert np.diff(grid) == pytest.approx(np.full(180, 0.005))


def test_preflight_rejects_feature_order_or_scannet_gate_adjustment_drift(tmp_path: Path):
    value = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    value["training_protocol"]["feature_names"][0:2] = list(reversed(
        value["training_protocol"]["feature_names"][0:2]
    ))
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="feature order"):
        validate_static_config(changed)

    value = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    value["training_protocol"]["scannet_route_gate_audit"][
        "ca_e961_preregistered_adjustment"
    ] = "silent_change"
    changed.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="gate audit"):
        validate_static_config(changed)

    value = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    value["training_protocol"]["runtime_selection"][
        "free_space_ratio_mean_strictly_greater_than_rejected"
    ] = 0.46
    changed.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime hard"):
        validate_static_config(changed)


def test_stage6_confirmed_track_emits_visibility_and_selected_geometry():
    observer = LightweightAsyncTR3DObserver(
        IncrementalTR3DConfig(), _UnusedProvider(), LightweightFusionConfig(),
    )
    observer.reset_scene("00000001", np.eye(4))
    box = _corners((0.0, 0.0, 2.0))[None]
    depth = np.full((80, 100), 2.0, np.float32)
    intrinsics = np.asarray(
        [[80.0, 0.0, 50.0], [0.0, 80.0, 40.0], [0.0, 0.0, 1.0]],
        np.float64,
    )
    observer._latest_frame = (depth, intrinsics, np.eye(4), 3)
    observer.provider_calls = 1
    observer._associate_lightweight(
        box, np.asarray([0.8], np.float32), np.asarray([20]), "k0003",
    )
    assert observer.finalize()["confirmed_tracks"] == 0

    # Recreate because finalize closes the observer executor by contract.
    observer = LightweightAsyncTR3DObserver(
        IncrementalTR3DConfig(), _UnusedProvider(), LightweightFusionConfig(),
    )
    observer.reset_scene("00000001", np.eye(4))
    observer._latest_frame = (depth, intrinsics, np.eye(4), 3)
    observer.provider_calls = 1
    observer._associate_lightweight(
        box, np.asarray([0.8], np.float32), np.asarray([20]), "k0003",
    )
    observer._latest_frame = (depth, intrinsics, np.eye(4), 8)
    observer.provider_calls = 2
    observer._associate_lightweight(
        box, np.asarray([0.7], np.float32), np.asarray([18]), "k0008",
    )
    payload = observer.finalize(
        anchor_corners_world=np.empty((0, 8, 3)),
        anchor_scores=np.empty((0,)),
    )
    assert payload["schema"] == "boxfusion.tr3d_lightweight_online_observer.v1"
    assert payload["lightweight_stage"] == 6
    assert payload["async_latest_only"] is True
    assert payload["tracks"] == 1
    assert payload["confirmed_tracks"] == 1
    row = payload["confirmed"][0]
    assert row["hit_count"] == 2
    assert row["prefix_ids"] == ["k0003", "k0008"]
    assert row["lightweight_schema"] == "boxfusion.tr3d_lightweight_track.v1"
    assert row["visibility_view_count"] == 2
    assert row["diverse_topk_count"] == 2
    assert row["selected_geometry"] in {"raw", "fused"}
    assert np.asarray(row["selected_corners_world"]).shape == (8, 3)
    for name in (
        "visibility_quality_mean", "support_ratio_mean", "free_space_ratio_mean",
        "invalid_ratio_mean", "selected_anchor_iou_max",
        "selected_anchor_center_distance_m",
    ):
        assert np.isfinite(float(row[name]))


def test_fold234_scene_grouped_oof_plan_accepts_only_other_two_folds():
    validate_fold234_oof_plan([2, 3, 4], [(3, 4), (2, 4), (2, 3)])
    with pytest.raises(ValueError, match="scene-grouped"):
        validate_fold234_oof_plan([2, 3, 4], [(2, 3), (2, 4), (2, 3)])
    with pytest.raises(ValueError, match="2,3,4 only"):
        validate_fold234_oof_plan([0, 2, 3], [(2, 3, 4), (3, 4), (2, 4)])
    validate_r4_gate_oof_scoring([2, 3, 4], ["[3,4]", "[2,4]", "[2,3]"])
    with pytest.raises(ValueError, match="scene-grouped"):
        validate_r4_gate_oof_scoring([2, 3, 4], ["[2,3]", "[2,4]", "[2,3]"])


def test_r4_near_rows_are_explicitly_scattered_to_raw_proposal_identity():
    selected = terminal_selection_for_state(
        TERMINAL_PASS_STATUS, [3, 1], np.asarray([False, True]), 5,
    )
    assert selected.tolist() == [False, True, False, False, False]
    with pytest.raises(ValueError, match="near-to-raw"):
        terminal_selection_for_state(
            TERMINAL_PASS_STATUS, [1, 1], np.asarray([True, False]), 5,
        )
    with pytest.raises(ValueError, match="near-to-raw"):
        terminal_selection_for_state(
            TERMINAL_PASS_STATUS, [1, 5], np.asarray([True, False]), 5,
        )
    with pytest.raises(ValueError, match="near-to-raw"):
        terminal_selection_for_state(
            TERMINAL_PASS_STATUS, [1, 2], np.asarray([True]), 5,
        )


def test_scientific_stop_is_identity_anchors_but_failure_blocks():
    selected = terminal_selection_for_state(
        TERMINAL_SCIENTIFIC_STOP_STATUS, [], [], 4,
    )
    assert selected.tolist() == [False] * 4
    with pytest.raises(ValueError, match="must not claim"):
        terminal_selection_for_state(
            TERMINAL_SCIENTIFIC_STOP_STATUS, [1], [False], 4,
        )
    with pytest.raises(PermissionError, match="neither scientific"):
        terminal_selection_for_state("CORRUPT_OR_PARTIAL", [], [], 4)


def test_r4_raw_rows_rebuild_anchors_then_independent_tracks_are_labeled():
    selected = terminal_selection_for_state(
        TERMINAL_PASS_STATUS, [1, 3], np.asarray([True, False]), 4,
    )
    coverage = post_terminal_gt_coverage(
        anchor_best_gt=[0, 1], anchor_best_iou=[0.6, 0.2],
        candidate_anchor_positions=[0, 0, 1, 1],
        candidate_best_gt=[2, 0, 2, 1],
        candidate_max_gt_iou=[0.8, 0.9, 0.7, 0.5],
        terminal_selected=selected,
    )
    assert coverage == {0: 0.9, 1: 0.2}
    novel25, quality50, novel50 = track_targets(
        track_best_gt=[2, 0, 1],
        track_max_gt_iou=[0.8, 0.9, 0.5],
        post_terminal_coverage=coverage,
    )
    assert novel25.tolist() == [True, False, True]
    assert quality50.tolist() == [True, True, True]
    assert novel50.tolist() == [True, False, True]
    with pytest.raises(ValueError, match="coverage mapping"):
        track_targets(
            track_best_gt=[0], track_max_gt_iou=[0.5],
            post_terminal_coverage={0: float("nan")},
        )
    with pytest.raises(ValueError, match="integer array"):
        track_targets(
            track_best_gt=[1.5], track_max_gt_iou=[0.5],
            post_terminal_coverage={1: 0.0},
        )
    with pytest.raises(ValueError, match="integer array"):
        post_terminal_gt_coverage(
            anchor_best_gt=[0.5], anchor_best_iou=[0.6],
            candidate_anchor_positions=[0], candidate_best_gt=[0],
            candidate_max_gt_iou=[0.7], terminal_selected=[False],
        )


def test_stage6_dual_head_gate_free_space_order_and_strict_nms():
    rows = [
        {"track_id": 4, "best_score": 0.9, "post_terminal_anchor_iou_max": 0.1,
         "free_space_ratio_mean": 0.45, "visibility_quality_mean": 0.0,
         "support_ratio_mean": 0.0, "selected_geometry": "raw",
         "selected_corners_world": _corners((0, 0, 0))},
        {"track_id": 2, "best_score": 0.8, "post_terminal_anchor_iou_max": 0.0,
         "free_space_ratio_mean": 0.0, "visibility_quality_mean": 1.0,
         "support_ratio_mean": 1.0, "selected_geometry": "fused",
         "selected_corners_world": _corners((3, 0, 0))},
        {"track_id": 3, "best_score": 0.7, "post_terminal_anchor_iou_max": 0.0,
         "free_space_ratio_mean": 0.450001, "visibility_quality_mean": 1.0,
         "support_ratio_mean": 1.0, "selected_geometry": "fused",
         "selected_corners_world": _corners((6, 0, 0))},
    ]
    # Row 2 is rejected by strict free-space > .45. Quality is admission only;
    # row 1 ranks first because its novelty-derived source rank is largest.
    assert select_stage6_candidates(
        rows, [0.8, 0.75, 0.99], [0.9, 0.6, 0.99],
        novelty_threshold=0.7, quality_threshold=0.5,
    ) == (1, 0)
    assert select_stage6_candidates(
        rows, [0.8, 0.75, 0.99], [0.9, 0.49, 0.99],
        novelty_threshold=0.7, quality_threshold=0.5,
    ) == (0,)

    overlapping = [dict(rows[0]), dict(rows[0])]
    overlapping[1]["track_id"] = 1
    # Identical geometry overlaps by 1.0 (> .25), so only higher source rank remains.
    assert select_stage6_candidates(
        overlapping, [0.8, 0.9], [0.9, 0.9],
        novelty_threshold=0.5, quality_threshold=0.5,
    ) == (1,)
    with pytest.raises(ValueError, match="probability"):
        select_stage6_candidates(
            rows, [0.8, 0.75, 0.99], [0.9, 0.6, 0.99],
            novelty_threshold=float("nan"), quality_threshold=0.5,
        )
    with pytest.raises(ValueError, match="identity/numeric"):
        duplicate = [dict(rows[0]), dict(rows[0])]
        select_stage6_candidates(
            duplicate, [0.8, 0.7], [0.9, 0.9],
            novelty_threshold=0.5, quality_threshold=0.5,
        )
    with pytest.raises(ValueError, match="identity/numeric"):
        float_id = [dict(rows[0])]; float_id[0]["track_id"] = 1.0
        select_stage6_candidates(
            float_id, [0.8], [0.9], novelty_threshold=0.5,
            quality_threshold=0.5,
        )


def test_stage6_float32_aabb_matches_pinned_scannet_boundary_math():
    left = _corners((0.0, 0.0, 0.0)).astype(np.float64)
    right = np.stack([
        _corners((0.6, 0.0, 0.0)),
        _corners((0.600001, 0.0, 0.0)),
        _corners((0.599999, 0.0, 0.0)),
    ]).astype(np.float64)
    ours = _aabb_iou_one_to_many(left, right).reshape(-1)
    pinned = pinned_incremental_aabb_iou(
        np.ascontiguousarray(left[None], np.float32),
        np.ascontiguousarray(right, np.float32),
    ).reshape(-1)
    assert np.array_equal(ours, pinned)
    assert np.array_equal(ours > 0.25, pinned > 0.25)


class _FakeCAWorker:
    def infer(self, **kwargs):
        points = np.asarray(kwargs["points_world_xyzrgb"], np.float32)
        import hashlib
        return SimpleNamespace(
            corners_world=_corners((0, 0, 2))[None],
            scores=np.asarray([0.8], np.float32), labels=np.asarray([0]),
            point_counts=np.asarray([4]), model_runtime_s=0.01,
            source_points_sha256=hashlib.sha256(points.tobytes(order="C")).hexdigest(),
            adapter_mode="genuine",
        )


def test_ca_provider_adapter_binds_world_to_local_and_result_contract():
    transform = np.eye(4)
    adapter = CA1ME961IncrementalProviderV2(
        _FakeCAWorker(), world_to_local=transform,
    )
    points = np.ones((2, 6), np.float32)
    result = adapter.infer(
        scene_id="00000001", prefix_id="k0003", points_world_xyzrgb=points,
        axis_align_matrix=transform,
    )
    assert result.corners_world.shape == (1, 8, 3)
    wrong = transform.copy(); wrong[0, 3] = 1.0
    with pytest.raises(ValueError, match="transform differs"):
        adapter.infer(
            scene_id="00000001", prefix_id="k0003", points_world_xyzrgb=points,
            axis_align_matrix=wrong,
        )


def test_low_scores_preserve_rank_below_all_post_terminal_anchors():
    scores = assign_low_scores([(0, 2, 0.3), (0, 1, 0.8)], 0.4)
    assert 0.0 < scores[(0, 2)] < scores[(0, 1)] < 0.4


def test_new_namespace_does_not_import_or_authorize_old_l6():
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert config["forbidden_reuse"]["old_artifact_access"] is False
    for relative in (
        "boxfusion/ca1m_e961_incremental_l6_v2.py",
        "tools/preflight_ca1m_e961_incremental_l6_v2.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "from boxfusion.ca1m_incremental_l6 import" not in source
        assert "import boxfusion.ca1m_incremental_l6" not in source
