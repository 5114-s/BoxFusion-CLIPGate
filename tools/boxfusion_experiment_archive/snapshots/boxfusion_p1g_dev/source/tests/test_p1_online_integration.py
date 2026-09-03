"""End-to-end identity contract for the detached P1 controller hook."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from boxfusion.object_memory import aabb_corners
from boxfusion.online_refinement import (
    OnlineRefinementController,
    resolve_online_refinement_config,
)
from boxfusion.p_ablation import apply_p_ablation
from boxfusion.p1_multiview_geometry import (
    P1G_DIAGNOSTIC_SCHEMA,
    P1G_PROFILE,
)


class _EmptyProvider:
    def predict(self, images, *, frame_ids=None):
        assert len(images) == 1
        assert frame_ids is not None and len(frame_ids) == 1
        return [[]]


class _SingleNativeSparseHead:
    """Deterministic injected P1S head for controller-only tests."""

    def __call__(self, features, coordinates):
        del coordinates
        count = len(features)
        logits = np.full((count,), -20.0, dtype=np.float32)
        regression = np.zeros((count, 6), dtype=np.float32)
        if count:
            logits[0] = 20.0
            regression[0, 3:] = np.log(0.50)
        return logits, regression


def _config(root: Path, stage: str):
    base = {
        "dataset": "scannet",
        "online_refinement": {
            "enabled": True,
            "inference_every_keyframes": 1,
            "supplemental_proposals": {"enabled": False},
            "appearance_memory": {"enabled": False, "masked_crop": True},
            "object_memory": {},
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
                "depth_stride": 2,
                "voxel_size": 0.08,
                "max_history_steps": 4,
            },
        },
    }
    configured = apply_p_ablation(base, stage)
    residual = configured["online_refinement"]["residual_proposal"]
    if stage == "P1":
        residual["mode"] = "collect"
        residual["collect_voxel_inputs"] = True
    if stage == "P1G":
        geometry = configured["online_refinement"][
            "p1_multiview_geometry"
        ]
        geometry.update(
            {
                "association_iou": 0.05,
                "crop_scale": 2.0,
                "top_k_views": 1,
                "max_points_per_view": 128,
                "proposal": {
                    "max_views": 1,
                    "max_points_per_view": 128,
                    "crop_scale": 2.0,
                    "min_views": 1,
                    "min_points_per_view": 1,
                    "min_total_points": 1,
                    "fine_min_view_consensus": 1,
                    "min_component_views": 1,
                    "min_component_points": 1,
                    "face_min_views": 1,
                    "face_min_points_per_view": 1,
                },
            }
        )
    return configured


def _run(root: Path, stage: str, *, residual_head=None):
    controller = OnlineRefinementController(
        _config(root, stage),
        provider=_EmptyProvider(),
        device="cpu",
        residual_proposal_head=residual_head,
    )
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    depth = np.full((12, 12), 2.0, dtype=np.float32)
    intrinsics = np.asarray(
        [[100.0, 0.0, 5.5], [0.0, 100.0, 5.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    pose = np.eye(4, dtype=np.float32)
    corners = aabb_corners(
        np.asarray([0.0, 0.0, 2.0]),
        np.asarray([0.08, 0.08, 0.08]),
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
    return result


def test_p1_hook_preserves_every_formal_output_row_and_dumps_safe_schema(
    tmp_path,
):
    p0 = _run(tmp_path / "p0", "P0")
    p1 = _run(tmp_path / "p1", "P1")
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
        right = np.asarray(getattr(p1, name))
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert left.tobytes(order="C") == right.tobytes(order="C")
    assert p0.labels == p1.labels

    diagnostic = tmp_path / "p1" / "scene0000_00_tracks.npz"
    with np.load(diagnostic, allow_pickle=False) as payload:
        assert bool(payload["p1_enabled"]) is True
        assert bool(payload["p1_observer_only"]) is True
        assert bool(payload["p1_uses_ground_truth"]) is False
        assert bool(payload["p1_mutation_enabled"]) is False
        assert int(payload["p1_applied_count"]) == 0
        assert payload["p1_candidate_boxes"].shape[1:] == (6,)
        assert payload["p1_voxel_features"].shape[1] == 14


def test_p1g_consumes_frozen_p1s_stream_and_preserves_formal_output(
    tmp_path,
):
    p1s = _run(
        tmp_path / "p1s",
        "P1S",
        residual_head=_SingleNativeSparseHead(),
    )
    p1g = _run(
        tmp_path / "p1g",
        "P1G",
        residual_head=_SingleNativeSparseHead(),
    )
    for name in (
        "corners",
        "boxes",
        "scores",
        "source_indices",
        "stable_ids",
        "quality_features",
        "refit_applied",
    ):
        parent = np.asarray(getattr(p1s, name))
        observed = np.asarray(getattr(p1g, name))
        assert observed.dtype == parent.dtype
        assert observed.shape == parent.shape
        assert observed.tobytes(order="C") == parent.tobytes(order="C")
    assert p1g.labels == p1s.labels
    assert p1g.summary["p1g_enabled"] is True
    assert p1g.summary["p1g_mutation_enabled"] is False
    assert p1g.summary["p1g_applied_count"] == 0
    assert p1g.summary["p1g_scene_candidates"] > 0

    diagnostic = tmp_path / "p1g" / "scene0000_00_tracks.npz"
    with np.load(diagnostic, allow_pickle=False) as payload:
        assert payload["p1g_schema"].item() == P1G_DIAGNOSTIC_SCHEMA
        assert payload["p1g_stage"].item() == "P1G"
        assert payload["p1g_profile"].item() == P1G_PROFILE
        assert payload["p1g_parent_stage"].item() == "P1S"
        assert payload["p1g_parent_checkpoint_sha256"].item() == (
            "injected"
        )
        assert bool(payload["p1g_observer_only"]) is True
        assert bool(payload["p1g_uses_ground_truth"]) is False
        assert bool(payload["p1g_reads_semantic_labels"]) is False
        assert bool(payload["p1g_mutation_enabled"]) is False
        assert int(payload["p1g_applied_count"]) == 0
        assert payload["p1g_parent_boxes"].shape[1:] == (6,)
        assert payload["p1g_refined_boxes"].shape == (
            payload["p1g_parent_boxes"].shape
        )
        assert len(payload["p1g_parent_candidate_ids"]) == int(
            p1g.summary["p1g_scene_candidates"]
        )


def test_p1g_online_stage_profile_and_added_module_are_strict(tmp_path):
    online = _config(tmp_path / "valid", "P1G")["online_refinement"]
    resolved = resolve_online_refinement_config(online)
    assert resolved["p_ablation_stage"] == "P1G"
    assert resolved["p_ablation_profile"] == P1G_PROFILE
    assert resolved["p_added_module"] == (
        "multiview_occupancy_msr_refiner"
    )
    assert resolved["p1_multiview_geometry"]["enabled"] is True
    assert resolved["occupancy_topk"]["enabled"] is False

    wrong_profile = _config(
        tmp_path / "wrong-profile", "P1G"
    )["online_refinement"]
    wrong_profile["p_ablation_profile"] = (
        "p1s_native_sparse_context_observer"
    )
    with pytest.raises(ValueError, match="p_ablation_profile"):
        resolve_online_refinement_config(wrong_profile)

    missing_geometry = _config(
        tmp_path / "missing-geometry", "P1G"
    )["online_refinement"]
    missing_geometry["p1_multiview_geometry"]["enabled"] = False
    with pytest.raises(ValueError, match="p1_multiview_geometry"):
        resolve_online_refinement_config(missing_geometry)
