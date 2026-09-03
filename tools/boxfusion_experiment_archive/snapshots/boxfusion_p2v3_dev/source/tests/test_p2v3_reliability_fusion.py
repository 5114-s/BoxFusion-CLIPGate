from __future__ import annotations

import inspect

import numpy as np
import pytest

from boxfusion.p2_local_mask_geometry import (
    P2MaskGeometryCandidate,
    P2MaskGeometryStep,
    P2V2_DIAGNOSTIC_SCHEMA,
    P2V2_SOURCE,
)
from boxfusion.p2_reliability_fusion import (
    P2ReliabilityFusionObserver,
    P2V3_DIAGNOSTIC_SCHEMA,
    P2V3_PROFILE,
    P2V3_RELIABILITY_CONTRACT,
    P2V3_SOURCE,
    resolve_p2_reliability_fusion_config,
)
from boxfusion.residual_proposal import center_size_to_corners


def _config(**overrides):
    config = {
        "enabled": True,
        "observer_only": True,
        "mutate": False,
        "collect_diagnostics": True,
        "minimum_component_weight": 0.35,
        "maximum_component_weight": 0.85,
        "step_nms_iou": 0.25,
        "scene_nms_iou": 0.25,
    }
    config.update(overrides)
    return config


def _candidate(
    *,
    candidate_id: str = "p2v2:test",
    component_box=(0.4, 0.0, 0.0, 1.0, 2.0, 4.0),
    parent_box=(0.0, 0.0, 0.0, 2.0, 2.0, 2.0),
    mask_score: float = 0.95,
    valid_depth_ratio: float = 0.9,
    component_point_count: int = 128,
    component_voxel_count: int = 32,
    selected_voxels_inside: int = 4,
    parent_objectness: float = 0.55,
    occupancy_score: float = 0.65,
) -> P2MaskGeometryCandidate:
    component = np.asarray(component_box, dtype=np.float32)
    parent = np.asarray(parent_box, dtype=np.float32)
    return P2MaskGeometryCandidate(
        candidate_id=candidate_id,
        parent_p2_candidate_id="scene0000_00:0:1:2:3",
        mask_source_id="scene0000_00:000000:lifted:0000",
        frame_index=5,
        provider_step=1,
        box=component,
        corners=center_size_to_corners(component)[0],
        parent_box=parent,
        score=0.73,
        parent_objectness=parent_objectness,
        occupancy_score=occupancy_score,
        mask_score=mask_score,
        valid_depth_ratio=valid_depth_ratio,
        component_point_count=component_point_count,
        component_voxel_count=component_voxel_count,
        selected_voxels_inside=selected_voxels_inside,
        anchor_inside=True,
        parent_iou=0.30,
        normalized_center_distance=0.2,
        extent_ratios=component[3:] / parent[3:],
        center_shift_ratios=(
            np.abs(component[:3] - parent[:3]) / parent[3:]
        ),
    )


def _step(candidate: P2MaskGeometryCandidate) -> P2MaskGeometryStep:
    return P2MaskGeometryStep(
        frame_index=candidate.frame_index,
        provider_step=candidate.provider_step,
        selected_voxel_count=4,
        occupancy_component_count=1,
        mask_observation_count=1,
        mask_component_count=1,
        eligible_pair_count=1,
        candidates=(candidate,),
        seconds=0.01,
    )


def _observe(candidate, **config_overrides):
    observer = P2ReliabilityFusionObserver(
        _config(**config_overrides),
        parent_p2_checkpoint_sha256="a" * 64,
    )
    step = observer.observe(
        scene_id="scene0000_00",
        p2v2_step=_step(candidate),
    )
    assert len(step.candidates) == 1
    return observer, step.candidates[0]


def test_p2v3_config_and_api_are_strictly_observer_only():
    config = resolve_p2_reliability_fusion_config(_config())
    assert config.enabled is True
    assert config.observer_only is True
    assert config.mutate is False
    with pytest.raises(ValueError, match="observer_only"):
        resolve_p2_reliability_fusion_config({"observer_only": False})
    with pytest.raises(ValueError, match="cannot mutate"):
        resolve_p2_reliability_fusion_config({"mutate": True})
    with pytest.raises(ValueError, match="interval"):
        resolve_p2_reliability_fusion_config(
            {
                "minimum_component_weight": 0.8,
                "maximum_component_weight": 0.2,
            }
        )
    with pytest.raises(ValueError, match="Unknown"):
        resolve_p2_reliability_fusion_config({"semantic_label": True})
    signature = inspect.signature(P2ReliabilityFusionObserver.observe)
    lowered = " ".join(signature.parameters).lower()
    assert "gt" not in lowered
    assert "label" not in lowered
    assert "feature" not in lowered


def test_p2v3_reliability_is_monotonic_and_fusion_is_convex_axis_aware():
    _, high = _observe(_candidate())
    _, low = _observe(
        _candidate(
            candidate_id="p2v2:low",
            mask_score=0.1,
            valid_depth_ratio=0.1,
            component_point_count=1,
            component_voxel_count=1,
            selected_voxels_inside=0,
        )
    )
    assert high.component_reliability > low.component_reliability
    assert high.component_weight > low.component_weight
    assert 0.35 <= low.component_weight <= high.component_weight <= 0.85

    # The shifted x center and mismatched x/z extents receive less component
    # trust than their agreeing axes.
    assert high.center_component_weights[0] < high.center_component_weights[1]
    assert high.extent_component_weights[0] < high.extent_component_weights[1]
    assert high.extent_component_weights[2] < high.extent_component_weights[1]
    expected_center = (
        high.center_component_weights * high.component_box[:3]
        + (1.0 - high.center_component_weights) * high.parent_box[:3]
    )
    expected_extent = (
        high.extent_component_weights * high.component_box[3:]
        + (1.0 - high.extent_component_weights) * high.parent_box[3:]
    )
    np.testing.assert_allclose(high.fused_box[:3], expected_center)
    np.testing.assert_allclose(high.fused_box[3:], expected_extent)
    assert high.score == pytest.approx(0.73)
    assert high.fused_box.flags.writeable is False
    assert high.center_component_weights.flags.writeable is False


def test_p2v3_is_deterministic_and_emits_complete_lineage_diagnostics():
    first_observer, first = _observe(_candidate())
    second_observer, second = _observe(_candidate())
    assert first.candidate_id == second.candidate_id
    np.testing.assert_array_equal(first.fused_box, second.fused_box)
    np.testing.assert_array_equal(
        first.center_component_weights,
        second.center_component_weights,
    )

    payload = first_observer.diagnostic_payload()
    assert payload["p2v3_schema"].item() == P2V3_DIAGNOSTIC_SCHEMA
    assert payload["p2v3_stage"].item() == "P2V3"
    assert payload["p2v3_profile"].item() == P2V3_PROFILE
    assert payload["p2v3_source"].item() == P2V3_SOURCE
    assert payload["p2v3_parent_p2v2_schema"].item() == P2V2_DIAGNOSTIC_SCHEMA
    assert payload["p2v3_parent_p2v2_source"].item() == P2V2_SOURCE
    assert (
        payload["p2v3_reliability_contract"].item()
        == P2V3_RELIABILITY_CONTRACT
    )
    assert bool(payload["p2v3_complete"]) is True
    assert bool(payload["p2v3_observer_only"]) is True
    assert bool(payload["p2v3_uses_ground_truth"]) is False
    assert bool(payload["p2v3_reads_semantic_labels"]) is False
    assert bool(payload["p2v3_mutation_enabled"]) is False
    assert int(payload["p2v3_applied_count"]) == 0
    assert not np.any(payload["p2v3_candidate_applied"])
    assert payload["p2v3_candidate_fused_boxes"].shape == (1, 6)
    assert payload["p2v3_candidate_center_component_weights"].shape == (1, 3)
    assert payload["p2v3_candidate_extent_component_weights"].shape == (1, 3)
    assert payload["p2v3_parent_p2v2_candidate_ids"].item() == (
        first.parent_p2v2_candidate_id
    )

    second_observer.record_failure(
        scene_id="scene0000_00",
        frame_index=6,
        provider_step=2,
        input_candidate_count=1,
        elapsed_seconds=0.01,
        error=RuntimeError("synthetic"),
    )
    failed = second_observer.diagnostic_payload()
    assert bool(failed["p2v3_complete"]) is False
    assert np.count_nonzero(failed["p2v3_step_failed"]) == 1
    assert "RuntimeError: synthetic" in failed["p2v3_step_errors"][-1]
