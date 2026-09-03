from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from boxfusion.object_memory import aabb_corners
from boxfusion.online_refinement import OnlineRefinementController
from boxfusion.p_ablation import apply_p_ablation
from boxfusion.supplemental_proposals import SupplementalProposal


class _OneMaskProvider:
    def __init__(self) -> None:
        mask = np.zeros((24, 24), dtype=bool)
        mask[5:19, 5:19] = True
        self.proposal = SupplementalProposal(
            bbox=np.asarray([5.0, 5.0, 19.0, 19.0], dtype=np.float32),
            score=0.9,
            mask=mask,
            label="semantic-label-must-not-be-read",
        )

    def predict(self, images, *, frame_ids=None):
        assert len(images) == 1
        assert frame_ids is not None and len(frame_ids) == 1
        return [[self.proposal]]


class _FrozenP1:
    def __call__(self, features):
        count = len(features)
        logits = np.full((count, 1), 5.0, dtype=np.float32)
        regression = np.zeros((count, 6), dtype=np.float32)
        regression[:, 3:] = math.log(0.35)
        return logits, regression

    def eval(self):
        return self


class _Occupancy:
    def __call__(self, features):
        return np.full((len(features), 1), 5.0, dtype=np.float32)

    def eval(self):
        return self


def _config(root: Path, stage: str):
    base = {
        "dataset": "scannet",
        "online_refinement": {
            "enabled": True,
            "inference_every_keyframes": 1,
            "supplemental_proposals": {
                "enabled": False,
                "provider": "unit_test",
            },
            "appearance_memory": {"enabled": False, "masked_crop": True},
            "object_memory": {
                "min_depth": 0.1,
                "max_depth": 5.0,
                "depth_scale": 1.0,
                "mask_threshold": 0.5,
                "mask_edge_margin": 0,
                "depth_edge_threshold": None,
                "voxel_size": 0.0,
                "max_points_per_observation": 512,
                "max_points_per_object": 1024,
                "aabb_lower_quantile": 0.0,
                "aabb_upper_quantile": 1.0,
                "min_points_for_aabb": 8,
                "minimum_aabb_dimension": 0.04,
            },
            "quality": {
                "enabled": True,
                "mode": "heuristic",
                "checkpoint": None,
                "feature_geometry": "original",
                "blend_with_detector": 0.40,
                "preserve_original_floor": False,
                "apply_to_unobserved": False,
                "apply_to_supplemental": False,
                "support_reference_points": 8192,
                "target_views": 3,
                "max_view_records": 5,
                "soft_nms": {"enabled": False},
            },
            "refit": {"enabled": False},
            "box_refiner": {"enabled": False},
            "supplemental_output": {"enabled": False},
            "output_filter": {"minimum_extent": 0.0},
            "diagnostics": {
                "enabled": True,
                "dump_track_memory": True,
                "root": str(root),
                "point_count": 16,
            },
            "residual_proposal": {
                "enabled": False,
                "observer_only": True,
                "mutate": False,
                "collect_diagnostics": False,
                "mode": "infer",
                "checkpoint": None,
                "device": "cpu",
                "depth_stride": 1,
                "voxel_size": 0.08,
                "score_threshold": 0.01,
                "pre_nms_topk": 64,
                "max_candidates_per_step": 32,
                "max_scene_candidates": 64,
            },
            "occupancy_topk": {
                "enabled": False,
                "observer_only": True,
                "mutate": False,
                "collect_diagnostics": False,
                "checkpoint": None,
                "device": "cpu",
                "min_occupancy_score": 0.05,
                "topk_voxels_per_step": 32,
                "max_candidates_per_step": 16,
                "max_scene_candidates": 32,
            },
            "p2_local_mask_geometry": {
                "enabled": False,
                "observer_only": True,
                "mutate": False,
                "collect_diagnostics": False,
                "occupancy_voxel_size": 0.08,
                "component_voxel_size": 0.04,
                "minimum_component_points": 8,
                "minimum_component_voxels": 2,
                "minimum_box_extent": 0.04,
                "maximum_box_extent": 1.0,
                # The synthetic depth plane has the enforced 0.04-m minimum
                # z extent versus a 0.35-m parent; permit that valid fixture
                # so the downstream P2V3 fusion is actually exercised.
                "minimum_extent_ratio": 0.05,
            },
            "p2_reliability_fusion": {
                "enabled": False,
                "observer_only": True,
                "mutate": False,
                "collect_diagnostics": False,
                "minimum_component_weight": 0.35,
                "maximum_component_weight": 0.85,
                "max_candidates_per_step": 16,
                "max_scene_candidates": 64,
            },
        },
    }
    return apply_p_ablation(base, stage)


def _run(root: Path, stage: str):
    kwargs = {}
    if stage in {"P2V2", "P2V3"}:
        kwargs.update(
            residual_proposal_head=_FrozenP1(),
            occupancy_topk_head=_Occupancy(),
        )
    controller = OnlineRefinementController(
        _config(root, stage),
        provider=_OneMaskProvider(),
        device="cpu",
        **kwargs,
    )
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    depth = np.full((24, 24), 2.0, dtype=np.float32)
    intrinsics = np.asarray(
        [[120.0, 0.0, 11.5], [0.0, 120.0, 11.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    pose = np.eye(4, dtype=np.float32)
    corners = aabb_corners(
        np.asarray([3.0, 3.0, 3.0]),
        np.asarray([0.4, 0.4, 0.4]),
    )[None]
    scores = np.asarray([0.73123455], dtype=np.float32)
    stable_ids = np.asarray([17], dtype=np.int64)
    controller.process_keyframe(
        image=image,
        depth=depth,
        intrinsics=intrinsics,
        camera_to_world=pose,
        frame_id=0,
        scene_id="scene0000_00",
        cache_frame_id="000000",
        global_corners=corners,
        global_scores=scores,
        stable_ids=stable_ids,
    )
    result = controller.finalize(
        global_corners=corners,
        global_scores=scores,
        stable_ids=stable_ids,
        scene_id="scene0000_00",
    )
    return result, controller.summary()


def test_p2v2_runtime_is_formally_identical_and_dumps_detached_stream(
    tmp_path,
):
    p0, _ = _run(tmp_path / "p0", "P0")
    p2v2, summary = _run(tmp_path / "p2v2", "P2V2")
    for name in (
        "corners",
        "boxes",
        "scores",
        "source_indices",
        "stable_ids",
        "quality_features",
        "refit_applied",
    ):
        left = np.asarray(getattr(p0, name))
        right = np.asarray(getattr(p2v2, name))
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert left.tobytes(order="C") == right.tobytes(order="C")
    assert p0.labels == p2v2.labels
    assert summary["p2v2_calls"] == 1
    assert summary["p2v2_failures"] == 0
    assert summary["p2v2_mask_observations"] == 1
    assert summary["p2v2_applied_count"] == 0

    path = tmp_path / "p2v2" / "scene0000_00_tracks.npz"
    with np.load(path, allow_pickle=False) as payload:
        assert payload["p2v2_stage"].item() == "P2V2"
        assert bool(payload["p2v2_observer_only"]) is True
        assert bool(payload["p2v2_reads_semantic_labels"]) is False
        assert bool(payload["p2v2_mutation_enabled"]) is False
        assert int(payload["p2v2_applied_count"]) == 0
        assert len(payload["p2v2_step_frame_ids"]) == 1
        assert payload["p2v2_candidate_boxes"].shape[1:] == (6,)


def test_p2v3_runtime_is_formally_identical_and_dumps_fused_diagnostics(
    tmp_path,
):
    p0, _ = _run(tmp_path / "p0", "P0")
    p2v3, summary = _run(tmp_path / "p2v3", "P2V3")
    for name in (
        "corners",
        "boxes",
        "scores",
        "source_indices",
        "stable_ids",
        "quality_features",
        "refit_applied",
    ):
        left = np.asarray(getattr(p0, name))
        right = np.asarray(getattr(p2v3, name))
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert left.tobytes(order="C") == right.tobytes(order="C")
    assert p0.labels == p2v3.labels
    assert summary["p2v2_calls"] == 1
    assert summary["p2v2_failures"] == 0
    assert summary["p2v3_calls"] == 1
    assert summary["p2v3_failures"] == 0
    assert summary["p2v3_input_candidates"] > 0
    assert summary["p2v3_input_candidates"] == (
        summary["p2v3_eligible_candidates"]
    )
    assert summary["p2v3_step_candidates"] == (
        summary["p2v3_scene_candidates"]
    )
    assert summary["p2v3_mutation_enabled"] is False
    assert summary["p2v3_applied_count"] == 0

    path = tmp_path / "p2v3" / "scene0000_00_tracks.npz"
    with np.load(path, allow_pickle=False) as payload:
        assert payload["p2v3_stage"].item() == "P2V3"
        assert (
            payload["p2v3_profile"].item()
            == "p2v3_reliability_geometry_fusion_observer"
        )
        assert (
            payload["p2v3_source"].item()
            == "p2v3_reliability_geometry_fusion"
        )
        assert bool(payload["p2v3_complete"]) is True
        assert bool(payload["p2v3_observer_only"]) is True
        assert bool(payload["p2v3_reads_semantic_labels"]) is False
        assert bool(payload["p2v3_mutation_enabled"]) is False
        assert int(payload["p2v3_applied_count"]) == 0
        assert len(payload["p2v3_step_frame_ids"]) == 1
        assert np.array_equal(
            payload["p2v3_step_frame_ids"],
            payload["p2v2_step_frame_ids"],
        )
        assert np.array_equal(
            payload["p2v3_step_provider_steps"],
            payload["p2v2_step_provider_steps"],
        )
        candidate_count = len(payload["p2v3_candidate_ids"])
        assert payload["p2v3_candidate_fused_boxes"].shape == (
            candidate_count,
            6,
        )
        assert payload["p2v3_candidate_center_component_weights"].shape == (
            candidate_count,
            3,
        )
        assert payload["p2v3_candidate_extent_component_weights"].shape == (
            candidate_count,
            3,
        )
        assert not np.any(payload["p2v3_candidate_applied"])
