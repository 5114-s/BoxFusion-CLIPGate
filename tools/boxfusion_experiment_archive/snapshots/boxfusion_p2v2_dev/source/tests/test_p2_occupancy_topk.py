from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from boxfusion.occupancy_topk import (
    P2OccupancyTopKObserver,
    assign_foreground_occupancy_targets,
    resolve_occupancy_topk_config,
    stable_occupancy_topk,
)


def test_p2_config_is_strict_and_output_safe():
    cfg = resolve_occupancy_topk_config(
        {
            "enabled": True,
            "observer_only": True,
            "mutate": False,
            "collect_diagnostics": True,
            "checkpoint": "p2.pt",
            "device": "cpu",
            "topk_voxels_per_step": 16,
            "max_candidates_per_step": 8,
        }
    )
    assert cfg.enabled is True
    assert cfg.observer_only is True
    assert cfg.mutate is False
    with pytest.raises(ValueError, match="Unknown"):
        resolve_occupancy_topk_config({"top_k": 1})
    with pytest.raises(ValueError, match="observer_only"):
        resolve_occupancy_topk_config({"observer_only": False})
    with pytest.raises(ValueError, match="cannot mutate"):
        resolve_occupancy_topk_config({"mutate": True})
    with pytest.raises(ValueError, match="cannot exceed"):
        resolve_occupancy_topk_config(
            {
                "topk_voxels_per_step": 4,
                "max_candidates_per_step": 5,
            }
        )


def test_foreground_occupancy_is_inside_any_residual_box():
    centers = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.49, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [4.0, 4.0, 4.0],
        ],
        dtype=np.float32,
    )
    boxes = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.2, 0.2, 0.2],
        ],
        dtype=np.float32,
    )
    target = assign_foreground_occupancy_targets(centers, boxes)
    np.testing.assert_array_equal(target, [1.0, 1.0, 1.0, 0.0])
    empty = assign_foreground_occupancy_targets(
        centers, np.empty((0, 6), dtype=np.float32)
    )
    np.testing.assert_array_equal(empty, np.zeros(len(centers)))


def test_stable_topk_is_score_then_coordinate_and_bounded():
    scores = np.asarray([0.7, 0.9, 0.9, 0.1], dtype=np.float64)
    coordinates = np.asarray(
        [[2, 0, 0], [1, 0, 0], [0, 0, 0], [3, 0, 0]],
        dtype=np.int32,
    )
    selected = stable_occupancy_topk(
        scores,
        coordinates,
        minimum_score=0.5,
        topk=2,
    )
    np.testing.assert_array_equal(selected, [2, 1])
    all_selected = stable_occupancy_topk(
        scores,
        coordinates,
        minimum_score=0.0,
        topk=99,
    )
    np.testing.assert_array_equal(all_selected, [2, 1, 0, 3])


class _FrozenP1:
    def __call__(self, features):
        count = len(features)
        logits = np.full((count, 1), 4.0, dtype=np.float32)
        regression = np.zeros((count, 6), dtype=np.float32)
        regression[:, 3:] = math.log(0.4)
        return logits, regression

    def eval(self):
        return self


class _Occupancy:
    def __call__(self, features):
        # Deterministic input-dependent logits with no hidden global state.
        return np.asarray(features[:, :1] * 4.0, dtype=np.float32)

    def eval(self):
        return self


def test_p2_online_api_has_no_gt_and_diagnostics_are_observer_only():
    signature = inspect.signature(P2OccupancyTopKObserver.observe)
    assert "gt" not in " ".join(signature.parameters).lower()
    observer = P2OccupancyTopKObserver(
        {
            "enabled": True,
            "observer_only": True,
            "mutate": False,
            "collect_diagnostics": True,
            "mode": "infer",
            "checkpoint": None,
            "device": "cpu",
            "depth_stride": 1,
            "voxel_size": 0.25,
            "score_threshold": 0.01,
            "pre_nms_topk": 32,
            "max_candidates_per_step": 16,
            "max_scene_candidates": 32,
            "nms_iou": 0.1,
            "scene_nms_iou": 0.1,
        },
        {
            "enabled": True,
            "observer_only": True,
            "mutate": False,
            "collect_diagnostics": True,
            "checkpoint": None,
            "device": "cpu",
            "topk_voxels_per_step": 4,
            "max_candidates_per_step": 4,
            "max_scene_candidates": 4,
        },
        p1_head=_FrozenP1(),
        occupancy_head=_Occupancy(),
    )
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    depth = np.ones((4, 4), dtype=np.float32)
    intrinsics = np.asarray(
        [[2.0, 0.0, 1.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    observation = observer.observe(
        image=image,
        depth=depth,
        intrinsics=intrinsics,
        camera_to_world=np.eye(4, dtype=np.float32),
        global_corners=np.empty((0, 8, 3), dtype=np.float32),
        global_stable_ids=np.empty((0,), dtype=np.int64),
        frame_index=0,
        provider_step=0,
        scene_id="scene0000_00",
    )
    assert observation.observer_only is True
    payload = observer.diagnostic_payload()
    assert payload["p2_stage"].item() == "P2"
    assert bool(payload["p2_observer_only"]) is True
    assert bool(payload["p2_uses_ground_truth"]) is False
    assert bool(payload["p2_mutation_enabled"]) is False
    assert int(payload["p2_applied_count"]) == 0
    assert len(payload["p2_candidate_boxes"]) <= 4
    assert payload["p2_candidate_boxes"].shape[1:] == (6,)
