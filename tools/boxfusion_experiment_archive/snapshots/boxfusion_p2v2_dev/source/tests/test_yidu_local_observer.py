import numpy as np
import pytest

from boxfusion.ap50_safety_gate import AP50SafetyGate
from boxfusion.quality_score import QUALITY_FEATURE_DIM
from boxfusion.yidu_local_observer import (
    YIDU_COMPONENT_FEATURE_DIM,
    YIDU_GATE_FEATURE_DIM,
    YIDU_GATE_FEATURE_NAMES,
    observe_yidu_local_geometry,
)


def _corners(center=(0.0, 0.0, 0.0), dims=(1.0, 1.0, 1.0)):
    center = np.asarray(center, dtype=np.float32)
    dims = np.asarray(dims, dtype=np.float32)
    lower = center - dims * 0.5
    upper = center + dims * 0.5
    return np.asarray(
        [
            [lower[0], lower[1], lower[2]],
            [upper[0], lower[1], lower[2]],
            [upper[0], upper[1], lower[2]],
            [lower[0], upper[1], lower[2]],
            [lower[0], lower[1], upper[2]],
            [upper[0], lower[1], upper[2]],
            [upper[0], upper[1], upper[2]],
            [lower[0], upper[1], upper[2]],
        ],
        dtype=np.float32,
    )


def _records():
    axis = np.linspace(-0.45, 0.45, 7, dtype=np.float32)
    points = np.asarray(
        np.meshgrid(axis, axis, axis, indexing="ij"),
        dtype=np.float32,
    ).reshape(3, -1).T
    return (
        {
            "frame_id": 1,
            "points_world": points,
            "quality": 0.90,
            "valid_depth_ratio": 0.95,
            "projection_mask_iou": 0.80,
            "camera_position": np.asarray([2.0, 0.0, 0.0]),
        },
        {
            "frame_id": 2,
            "points_world": points + np.asarray([0.002, 0.0, 0.0]),
            "quality": 0.85,
            "valid_depth_ratio": 0.92,
            "projection_mask_iou": 0.78,
            "camera_position": np.asarray([-2.0, 0.0, 0.0]),
        },
    )


def _config():
    return {
        "minimum_component_points": 16,
        "minimum_component_voxels": 4,
        "minimum_component_views": 2,
        "minimum_inside_points": 8,
        "minimum_inside_fraction": 0.1,
        "voxel_size": 0.20,
        "occupancy_msr": {
            "min_views": 2,
            "min_points_per_view": 8,
            "min_total_points": 32,
            "fine_min_view_consensus": 1,
            "min_component_views": 2,
            "min_component_points": 16,
            "face_min_views": 1,
            "face_min_points_per_view": 4,
            "face_min_empty_evidence": 0.0,
        },
    }


def _quality():
    return np.linspace(
        0.1, 0.9, QUALITY_FEATURE_DIM, dtype=np.float32
    )


def test_a1_consumes_clean_secondary_evidence_without_mutation():
    original = _corners()
    result = observe_yidu_local_geometry(
        stage="A1",
        original_corners=original,
        view_records=_records(),
        detector_score=0.7,
        b6_quality_features=_quality(),
        config=_config(),
    )
    assert result.stage == "A1"
    assert result.input_point_count > 0
    assert not result.mutation_enabled
    assert not result.applied
    np.testing.assert_array_equal(result.original_corners, original)
    assert not result.selected_candidate_corners.flags.writeable


def test_a3_builds_multiview_component_and_fixed_features():
    result = observe_yidu_local_geometry(
        stage="A3",
        original_corners=_corners(),
        view_records=_records(),
        detector_score=0.7,
        b6_quality_features=_quality(),
        config=_config(),
    )
    assert result.component_set is not None
    assert result.component_set.component_count >= 1
    assert result.selected_component_id >= 0
    assert result.component_features.shape == (
        YIDU_COMPONENT_FEATURE_DIM,
    )
    assert np.isfinite(result.component_features).all()


def test_a4_runs_existing_occupancy_msr_and_freezes_48d_features():
    result = observe_yidu_local_geometry(
        stage="A4",
        original_corners=_corners(),
        view_records=_records(),
        detector_score=0.7,
        b6_quality_features=_quality(),
        config=_config(),
    )
    assert result.occupancy_proposal is not None
    assert result.occupancy_features.shape == (48,)
    assert np.isfinite(result.occupancy_features).all()


def test_a5_builds_raw_fused_query_but_never_applies_selection():
    result = observe_yidu_local_geometry(
        stage="A5",
        original_corners=_corners(),
        raw_candidate_corners=_corners(dims=(0.95, 0.95, 0.95)),
        raw_candidate_verified=True,
        view_records=_records(),
        detector_score=0.7,
        b6_quality_features=_quality(),
        config=_config(),
    )
    assert result.raw_fused_observation is not None
    assert result.raw_fused_observation.observer_only
    assert not result.raw_fused_observation.mutation_enabled
    assert not result.applied
    assert result.selected_source in {
        "original",
        "raw_mask",
        "superpoint",
        "occupancy",
    }


def _dummy_gate():
    weights = np.zeros((YIDU_GATE_FEATURE_DIM, 8), dtype=np.float32)
    biases = np.asarray(
        [0.2, -10.0, 4.0, -4.0, 0.0, 1.0, 1.0, 1.0],
        dtype=np.float32,
    )
    return AP50SafetyGate(
        feature_names=YIDU_GATE_FEATURE_NAMES,
        weights=(weights,),
        biases=(biases,),
        feature_mean=np.zeros(YIDU_GATE_FEATURE_DIM, dtype=np.float32),
        feature_scale=np.ones(YIDU_GATE_FEATURE_DIM, dtype=np.float32),
    )


def test_a6_evaluates_exact_schema_gate_and_remains_observer_only():
    result = observe_yidu_local_geometry(
        stage="A6",
        original_corners=_corners(),
        raw_candidate_corners=_corners(dims=(0.95, 0.95, 0.95)),
        raw_candidate_verified=True,
        view_records=_records(),
        detector_score=0.7,
        b6_quality_features=_quality(),
        config=_config(),
        quality_gate=_dummy_gate(),
    )
    assert result.gate_features.shape == (YIDU_GATE_FEATURE_DIM,)
    assert result.gate_decision is not None
    assert not result.applied


def test_a6_requires_gate_and_b0_does_not_execute_observer():
    kwargs = dict(
        original_corners=_corners(),
        view_records=_records(),
        detector_score=0.7,
        b6_quality_features=_quality(),
        config=_config(),
    )
    with pytest.raises(ValueError, match="B0"):
        observe_yidu_local_geometry(stage="B0", **kwargs)
    with pytest.raises(ValueError, match="requires"):
        observe_yidu_local_geometry(stage="A6", **kwargs)
