from __future__ import annotations

import inspect

import numpy as np
import pytest

from boxfusion.occupancy_topk import (
    OccupancySelectedProposal,
    OccupancyTopKObservation,
)
from boxfusion.p2_local_mask_geometry import (
    P2LocalMaskGeometryObserver,
    P2MaskRGBDInput,
    P2V2_DIAGNOSTIC_SCHEMA,
    resolve_p2_local_mask_geometry_config,
)
from boxfusion.residual_proposal import (
    P1_FEATURE_DIM,
    ResidualObservation,
    ResidualProposal,
    ResidualVoxelBatch,
    center_size_to_corners,
)


def _p2_observation() -> OccupancyTopKObservation:
    coordinates = np.asarray(
        [[0, 0, 0], [1, 0, 0], [8, 8, 8]], dtype=np.int32
    )
    centers = np.asarray(
        [[0.0, 0.0, 0.0], [0.08, 0.0, 0.0], [0.64, 0.64, 0.64]],
        dtype=np.float32,
    )
    batch = ResidualVoxelBatch(
        coordinates=coordinates,
        centers=centers,
        features=np.zeros((3, P1_FEATURE_DIM), dtype=np.float32),
        point_counts=np.asarray([8, 7, 3], dtype=np.int32),
        input_point_count=100,
        explained_point_count=70,
        residual_point_count=30,
    )
    box = np.asarray([0.03, 0.0, 0.0, 0.4, 0.4, 0.4], dtype=np.float32)
    proposal = ResidualProposal(
        candidate_id="scene0000_00:0:0:0:0:0",
        frame_index=0,
        provider_step=0,
        box=box,
        corners=center_size_to_corners(box)[0],
        objectness=0.8,
        residual_point_count=15,
    )
    base = ResidualObservation(
        frame_index=0,
        provider_step=0,
        voxel_batch=batch,
        proposals=(proposal,),
    )
    return OccupancyTopKObservation(
        base=base,
        selected=(
            OccupancySelectedProposal(
                base=proposal,
                occupancy_score=0.9,
                occupancy_rank=0,
            ),
        ),
        eligible_voxels=2,
        selected_voxels=2,
        selected_voxel_indices=np.asarray([0, 1], dtype=np.int64),
        selected_voxel_scores=np.asarray([0.9, 0.8], dtype=np.float32),
        occupancy_seconds=0.001,
    )


def _mask_input(points: np.ndarray | None = None) -> P2MaskRGBDInput:
    if points is None:
        axis = np.linspace(-0.12, 0.12, 7, dtype=np.float32)
        points = np.stack(
            np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
        ).reshape(-1, 3)
    return P2MaskRGBDInput(
        source_id="scene0000_00:000000:lifted:0000",
        frame_index=0,
        provider_step=0,
        score=0.85,
        valid_depth_ratio=0.75,
        points_world=points,
    )


def _config() -> dict[str, object]:
    return {
        "enabled": True,
        "observer_only": True,
        "mutate": False,
        "collect_diagnostics": True,
        "occupancy_voxel_size": 0.08,
        "component_voxel_size": 0.04,
        "minimum_component_points": 12,
        "minimum_component_voxels": 4,
        "minimum_box_extent": 0.04,
        "maximum_box_extent": 1.0,
        "minimum_mask_score": 0.25,
        "minimum_valid_depth_ratio": 0.05,
    }


def test_p2v2_config_and_api_are_strictly_observer_only():
    cfg = resolve_p2_local_mask_geometry_config(_config())
    assert cfg.enabled is True
    assert cfg.observer_only is True
    assert cfg.mutate is False
    with pytest.raises(ValueError, match="observer_only"):
        resolve_p2_local_mask_geometry_config({"observer_only": False})
    with pytest.raises(ValueError, match="cannot mutate"):
        resolve_p2_local_mask_geometry_config({"mutate": True})
    with pytest.raises(ValueError, match="Unknown"):
        resolve_p2_local_mask_geometry_config({"semantic_label": True})
    signature = inspect.signature(P2LocalMaskGeometryObserver.observe)
    lowered = " ".join(signature.parameters).lower()
    assert "gt" not in lowered
    assert "label" not in lowered
    assert "feature" not in lowered


def test_mask_input_and_selected_voxel_arrays_are_detached_read_only():
    raw = np.zeros((16, 3), dtype=np.float32)
    mask = _mask_input(raw)
    raw[:] = 10.0
    assert np.all(mask.points_world == 0.0)
    assert mask.points_world.flags.writeable is False
    observation = _p2_observation()
    assert observation.selected_voxel_indices.flags.writeable is False
    assert observation.selected_voxel_scores.flags.writeable is False
    with pytest.raises(ValueError):
        observation.selected_voxel_indices[0] = 2


def test_p2v2_fits_deterministic_mask_component_without_mutation_path():
    observer = P2LocalMaskGeometryObserver(
        _config(),
        parent_p2_checkpoint_sha256="a" * 64,
        provider_name="cache_only",
    )
    p2 = _p2_observation()
    mask = _mask_input()
    first = observer.observe(
        scene_id="scene0000_00",
        p2_observation=p2,
        masks=[mask],
    )
    assert first.failed is False
    assert first.selected_voxel_count == 2
    assert first.mask_component_count == 1
    assert first.eligible_pair_count == 1
    assert len(first.candidates) == 1
    candidate = first.candidates[0]
    assert candidate.parent_p2_candidate_id == p2.selected[0].candidate_id
    assert candidate.mask_source_id == mask.source_id
    assert candidate.anchor_inside is True
    assert candidate.score == pytest.approx(0.9)
    assert np.all(candidate.box[3:] < candidate.parent_box[3:])

    second_observer = P2LocalMaskGeometryObserver(
        _config(),
        parent_p2_checkpoint_sha256="a" * 64,
        provider_name="cache_only",
    )
    second = second_observer.observe(
        scene_id="scene0000_00",
        p2_observation=_p2_observation(),
        masks=[_mask_input()],
    )
    np.testing.assert_array_equal(
        first.candidates[0].box, second.candidates[0].box
    )
    assert (
        first.candidates[0].candidate_id
        == second.candidates[0].candidate_id
    )

    payload = observer.diagnostic_payload()
    assert payload["p2v2_schema"].item() == P2V2_DIAGNOSTIC_SCHEMA
    assert payload["p2v2_stage"].item() == "P2V2"
    assert bool(payload["p2v2_observer_only"]) is True
    assert bool(payload["p2v2_uses_ground_truth"]) is False
    assert bool(payload["p2v2_reads_semantic_labels"]) is False
    assert bool(payload["p2v2_mutation_enabled"]) is False
    assert int(payload["p2v2_applied_count"]) == 0
    assert payload["p2v2_candidate_boxes"].shape == (1, 6)
    assert not np.any(payload["p2v2_candidate_applied"])


def test_p2v2_rejects_unaligned_mask_without_corrupting_history():
    observer = P2LocalMaskGeometryObserver(
        _config(),
        parent_p2_checkpoint_sha256="injected",
        provider_name="unit_test",
    )
    mask = _mask_input()
    bad = P2MaskRGBDInput(
        source_id=mask.source_id,
        frame_index=1,
        provider_step=0,
        score=mask.score,
        valid_depth_ratio=mask.valid_depth_ratio,
        points_world=mask.points_world,
    )
    with pytest.raises(ValueError, match="not aligned"):
        observer.observe(
            scene_id="scene0000_00",
            p2_observation=_p2_observation(),
            masks=[bad],
        )
    assert observer.steps == []
