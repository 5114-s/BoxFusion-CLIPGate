"""End-to-end identity contract for the detached P1 controller hook."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from boxfusion.object_memory import aabb_corners
from boxfusion.online_refinement import OnlineRefinementController
from boxfusion.p_ablation import apply_p_ablation


class _EmptyProvider:
    def predict(self, images, *, frame_ids=None):
        assert len(images) == 1
        assert frame_ids is not None and len(frame_ids) == 1
        return [[]]


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
    return configured


def _run(root: Path, stage: str):
    controller = OnlineRefinementController(
        _config(root, stage), provider=_EmptyProvider(), device="cpu"
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
