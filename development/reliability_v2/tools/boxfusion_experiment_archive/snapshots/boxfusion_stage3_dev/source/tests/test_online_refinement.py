from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from boxfusion.object_memory import (
    aabb_corners,
    aabb_iou,
    project_aabb_to_image,
)
from boxfusion.online_refinement import (
    OnlineRefinementController,
    resolve_online_refinement_config,
)
from boxfusion.supplemental_proposals import SupplementalProposal


class SequenceProvider:
    def __init__(self, batches):
        self.batches = list(batches)
        self.calls = 0

    def predict(self, images, *, frame_ids=None):
        assert len(images) == 1
        assert frame_ids is not None and len(frame_ids) == 1
        batch = self.batches[min(self.calls, len(self.batches) - 1)]
        self.calls += 1
        return [batch]


class ExplodingProvider:
    def predict(self, images, *, frame_ids=None):
        raise AssertionError("disabled controller must not invoke provider")


def make_config(tmp_path: Path, *, diagnostics=False, quality=False):
    return {
        "online_refinement": {
            "enabled": True,
            "inference_every_keyframes": 1,
            "supplemental_proposals": {
                "enabled": False,
            },
            "object_memory": {
                "enabled": True,
                "min_depth": 0.1,
                "max_depth": 5.0,
                "depth_scale": 1.0,
                "mask_threshold": 0.5,
                "mask_edge_margin": 0,
                "depth_edge_threshold": None,
                "voxel_size": 0.0,
                "max_points_per_observation": 256,
                "max_points_per_object": 512,
                "aabb_lower_quantile": 0.0,
                "aabb_upper_quantile": 1.0,
                "min_points_for_aabb": 4,
                "minimum_aabb_dimension": 0.01,
                "min_confirmations": 2,
                "track_ttl": 5,
                "association_iou_threshold": 0.01,
                "association_center_distance": 0.5,
                "association_inside_fraction": 0.1,
            },
            "matching": {
                "global_match_iou": 0.01,
                "global_match_2d_iou": 0.01,
                "max_center_distance": 0.5,
                "crop_to_global_expansion": 1.5,
                "rekey_iou": 0.5,
                "absorb_supplemental_iou": 0.35,
            },
            "refit": {
                "enabled": True,
                "min_views": 2,
                "min_points": 8,
                "blend": 1.0,
                "extent_padding": 0.0,
                "max_center_shift_ratio": 1.0,
                "min_extent_ratio": 0.02,
                "max_extent_ratio": 3.0,
                "min_original_point_support": 0.0,
                "min_reprojection_iou": 0.0,
                "min_reprojection_improvement": -1.0,
            },
            "box_refiner": {
                "enabled": False,
                "checkpoint": None,
                "device": None,
                "point_count": 32,
                "min_quality": 0.2,
                "architecture": {},
            },
            "quality": {
                "enabled": quality,
                "mode": "heuristic",
                "checkpoint": None,
                "blend_with_detector": 0.5,
                "preserve_original_floor": False,
                "apply_to_unobserved": False,
                "support_reference_points": 32,
                "target_views": 2,
                "max_view_records": 3,
                "soft_nms": {
                    "enabled": quality,
                    "method": "gaussian",
                    "iou_threshold": 0.3,
                    "sigma": 0.5,
                    "score_threshold": 0.01,
                    "max_detections": None,
                },
            },
            "supplemental_output": {
                "enabled": True,
                "min_confirmations": 2,
                "min_score": 0.05,
                "min_projection_iou": 0.0,
                "drop_if_global_iou": 0.7,
            },
            "diagnostics": {
                "enabled": diagnostics,
                "dump_track_memory": diagnostics,
                "root": str(tmp_path),
                "point_count": 32,
            },
        }
    }


def synthetic_inputs():
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    depth = np.full((12, 12), 2.0, dtype=np.float32)
    intrinsics = np.asarray(
        [[100.0, 0.0, 5.5], [0.0, 100.0, 5.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    pose = np.eye(4, dtype=np.float32)
    mask = np.zeros((12, 12), dtype=bool)
    mask[3:9, 3:9] = True
    proposal = SupplementalProposal(
        bbox=np.asarray([3.0, 3.0, 9.0, 9.0], dtype=np.float32),
        score=0.8,
        mask=mask,
        label="chair",
        feature=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    )
    return image, depth, intrinsics, pose, proposal


def process(
    controller,
    *,
    frame_id,
    image,
    depth,
    intrinsics,
    pose,
    corners,
    scores,
    stable_ids,
):
    controller.process_keyframe(
        image=image,
        depth=depth,
        intrinsics=intrinsics,
        camera_to_world=pose,
        frame_id=frame_id,
        scene_id="scene0000_00",
        global_corners=corners,
        global_scores=scores,
        stable_ids=stable_ids,
    )


def test_disabled_path_does_not_validate_optional_blocks_or_call_provider(tmp_path):
    cfg = {
        "online_refinement": {
            "enabled": False,
            "supplemental_proposals": "intentionally invalid but disabled",
        }
    }
    controller = OnlineRefinementController(
        cfg, provider=ExplodingProvider()
    )
    corners = aabb_corners(
        np.asarray([0.0, 0.0, 2.0]),
        np.asarray([0.4, 0.4, 0.4]),
    )[None]
    scores = np.asarray([0.73123455], dtype=np.float32)
    result = controller.finalize(
        global_corners=corners,
        global_scores=scores,
        stable_ids=np.asarray([17]),
    )
    assert np.array_equal(result.corners, corners)
    assert np.array_equal(result.scores, scores)
    assert result.stable_ids.tolist() == [17]
    assert result.summary == {"enabled": False}


def test_two_view_mask_depth_memory_refits_existing_global(tmp_path):
    image, depth, intrinsics, pose, proposal = synthetic_inputs()
    provider = SequenceProvider([[proposal], [proposal]])
    controller = OnlineRefinementController(
        make_config(tmp_path), provider=provider
    )
    original_box = np.asarray([0.0, 0.0, 2.0, 0.4, 0.4, 0.4])
    corners = aabb_corners(original_box[:3], original_box[3:])[None]
    for frame_id in (0, 25):
        process(
            controller,
            frame_id=frame_id,
            image=image,
            depth=depth,
            intrinsics=intrinsics,
            pose=pose,
            corners=corners,
            scores=np.asarray([0.7]),
            stable_ids=np.asarray([3]),
        )

    result = controller.finalize(
        global_corners=corners,
        global_scores=np.asarray([0.7]),
        stable_ids=np.asarray([3]),
    )
    assert provider.calls == 2
    assert result.boxes.shape == (1, 6)
    assert np.all(result.boxes[0, 3:] < original_box[3:])
    assert not np.array_equal(result.corners, corners)
    assert controller.global_tracks[3].memory.unique_view_count == 2
    assert result.labels == ("chair",)
    assert result.summary["refits_accepted"] == 1


def test_two_view_unmatched_proposal_becomes_supplemental_detection(tmp_path):
    image, depth, intrinsics, pose, proposal = synthetic_inputs()
    controller = OnlineRefinementController(
        make_config(tmp_path),
        provider=SequenceProvider([[proposal], [proposal]]),
    )
    empty_corners = np.empty((0, 8, 3), dtype=np.float32)
    for frame_id in (0, 25):
        process(
            controller,
            frame_id=frame_id,
            image=image,
            depth=depth,
            intrinsics=intrinsics,
            pose=pose,
            corners=empty_corners,
            scores=np.empty(0),
            stable_ids=np.empty(0, dtype=np.int64),
        )
    result = controller.finalize(
        global_corners=empty_corners,
        global_scores=np.empty(0),
        stable_ids=np.empty(0, dtype=np.int64),
    )
    assert result.boxes.shape == (1, 6)
    assert result.source_indices.tolist() == [-1]
    assert result.stable_ids.tolist() == [-1]
    assert result.labels == ("chair",)
    assert result.scores[0] > 0.05


def test_supplemental_projection_gate_rejects_inconsistent_track(tmp_path):
    image, depth, intrinsics, pose, proposal = synthetic_inputs()
    cfg = make_config(tmp_path)
    cfg["online_refinement"]["supplemental_output"][
        "min_projection_iou"
    ] = 0.30
    controller = OnlineRefinementController(
        cfg,
        provider=SequenceProvider([[proposal], [proposal]]),
    )
    empty_corners = np.empty((0, 8, 3), dtype=np.float32)
    for frame_id in (0, 25):
        process(
            controller,
            frame_id=frame_id,
            image=image,
            depth=depth,
            intrinsics=intrinsics,
            pose=pose,
            corners=empty_corners,
            scores=np.empty(0),
            stable_ids=np.empty(0, dtype=np.int64),
        )

    metadata = controller.supplemental_metadata[0]
    metadata.stats.view_records = [
        replace(
            view,
            bbox=np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
        )
        for view in metadata.stats.view_records
    ]
    result = controller.finalize(
        global_corners=empty_corners,
        global_scores=np.empty(0),
        stable_ids=np.empty(0, dtype=np.int64),
    )

    assert result.boxes.shape == (0, 6)
    assert result.summary["supplemental_considered"] == 1
    assert result.summary["supplemental_rejected_projection"] == 1
    assert result.summary["supplemental_output"] == 0


def test_supplemental_projection_gate_accepts_threshold_boundary(tmp_path):
    image, depth, intrinsics, pose, proposal = synthetic_inputs()
    cfg = make_config(tmp_path)
    cfg["online_refinement"]["supplemental_output"][
        "min_projection_iou"
    ] = 1.0
    controller = OnlineRefinementController(
        cfg,
        provider=SequenceProvider([[proposal], [proposal]]),
    )
    empty_corners = np.empty((0, 8, 3), dtype=np.float32)
    for frame_id in (0, 25):
        process(
            controller,
            frame_id=frame_id,
            image=image,
            depth=depth,
            intrinsics=intrinsics,
            pose=pose,
            corners=empty_corners,
            scores=np.empty(0),
            stable_ids=np.empty(0, dtype=np.int64),
        )

    center, dims = controller.track_manager.tracks[0].memory.aabb
    box = np.concatenate((center, dims))
    metadata = controller.supplemental_metadata[0]
    exact_views = []
    for view in metadata.stats.view_records:
        projected = project_aabb_to_image(
            box[:3],
            box[3:6],
            view.intrinsics,
            view.camera_to_world,
            view.image_shape,
            require_all_in_front=False,
        )
        assert projected is not None
        exact_views.append(replace(view, bbox=projected))
    metadata.stats.view_records = exact_views

    result = controller.finalize(
        global_corners=empty_corners,
        global_scores=np.empty(0),
        stable_ids=np.empty(0, dtype=np.int64),
    )

    assert result.boxes.shape == (1, 6)
    assert result.quality_features[0, 4] == pytest.approx(1.0)
    assert result.summary["supplemental_rejected_projection"] == 0
    assert result.summary["supplemental_output"] == 1


def test_global_iou_gate_drops_candidate_above_configured_threshold(tmp_path):
    image, depth, intrinsics, pose, proposal = synthetic_inputs()
    controller = OnlineRefinementController(
        make_config(tmp_path),
        provider=SequenceProvider([[proposal], [proposal]]),
    )
    empty_corners = np.empty((0, 8, 3), dtype=np.float32)
    for frame_id in (0, 25):
        process(
            controller,
            frame_id=frame_id,
            image=image,
            depth=depth,
            intrinsics=intrinsics,
            pose=pose,
            corners=empty_corners,
            scores=np.empty(0),
            stable_ids=np.empty(0, dtype=np.int64),
        )

    center, dims = controller.track_manager.tracks[0].memory.aabb
    supplemental_box = np.concatenate((center, dims)).astype(np.float32)
    global_box = supplemental_box.copy()
    global_box[0] += float(dims[0]) * 0.53
    overlap = aabb_iou(
        supplemental_box[:3],
        supplemental_box[3:6],
        global_box[:3],
        global_box[3:6],
    )
    assert overlap > 0.30
    controller.config["supplemental_output"]["drop_if_global_iou"] = 0.30

    result = controller.finalize(
        global_corners=aabb_corners(
            global_box[:3], global_box[3:6]
        )[None],
        global_scores=np.asarray([0.8]),
        stable_ids=np.asarray([7]),
    )

    assert result.source_indices.tolist() == [0]
    assert result.summary["supplemental_rejected_global"] == 1
    assert result.summary["supplemental_output"] == 0


def test_invalid_small_supplemental_cannot_suppress_valid_track(tmp_path):
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    depth = np.full((12, 12), 2.0, dtype=np.float32)
    depth[:, 7:] = 4.0
    intrinsics = np.asarray(
        [[100.0, 0.0, 5.5], [0.0, 100.0, 5.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    pose = np.eye(4, dtype=np.float32)
    left_mask = np.zeros((12, 12), dtype=bool)
    left_mask[3:9, 0:5] = True
    right_mask = np.zeros((12, 12), dtype=bool)
    right_mask[3:9, 7:12] = True
    high_score_small = SupplementalProposal(
        bbox=np.asarray([0.0, 3.0, 5.0, 9.0], dtype=np.float32),
        score=0.9,
        mask=left_mask,
        label="small",
    )
    lower_score_valid = SupplementalProposal(
        bbox=np.asarray([7.0, 3.0, 12.0, 9.0], dtype=np.float32),
        score=0.8,
        mask=right_mask,
        label="valid",
    )
    cfg = make_config(tmp_path)
    online = cfg["online_refinement"]
    online["output_filter"] = {"minimum_extent": 0.30}
    online["supplemental_output"]["drop_if_supplemental_iou"] = 0.70
    controller = OnlineRefinementController(
        cfg,
        provider=SequenceProvider(
            [
                [high_score_small, lower_score_valid],
                [high_score_small, lower_score_valid],
            ]
        ),
    )
    empty_corners = np.empty((0, 8, 3), dtype=np.float32)
    for frame_id in (0, 25):
        process(
            controller,
            frame_id=frame_id,
            image=image,
            depth=depth,
            intrinsics=intrinsics,
            pose=pose,
            corners=empty_corners,
            scores=np.empty(0),
            stable_ids=np.empty(0, dtype=np.int64),
        )

    assert sorted(controller.track_manager.tracks) == [0, 1]
    center = np.asarray([0.0, 0.0, 2.0], dtype=np.float32)
    controller.track_manager.tracks[0].memory._points = aabb_corners(
        center, np.full(3, 0.29, dtype=np.float32)
    )
    controller.track_manager.tracks[1].memory._points = aabb_corners(
        center, np.full(3, 0.31, dtype=np.float32)
    )

    result = controller.finalize(
        global_corners=empty_corners,
        global_scores=np.empty(0),
        stable_ids=np.empty(0, dtype=np.int64),
    )

    assert result.boxes.shape == (1, 6)
    assert result.stable_ids.tolist() == [-2]
    assert result.labels == ("valid",)
    assert result.summary["supplemental_rejected_extent"] == 1
    assert result.summary["supplemental_deduplicated"] == 0
    assert result.summary["supplemental_output"] == 1


def test_provider_call_ttl_does_not_age_on_unscheduled_keyframes(tmp_path):
    image, depth, intrinsics, pose, proposal = synthetic_inputs()
    cfg = make_config(tmp_path)
    online = cfg["online_refinement"]
    online["inference_every_keyframes"] = 5
    online["candidate_lifecycle"] = {
        "ttl_clock": "provider_call",
        "archive_confirmed": False,
    }
    online["object_memory"]["track_ttl"] = 1
    controller = OnlineRefinementController(
        cfg,
        provider=SequenceProvider([[proposal], [proposal]]),
    )
    empty_corners = np.empty((0, 8, 3), dtype=np.float32)

    for frame_id in range(6):
        process(
            controller,
            frame_id=frame_id,
            image=image,
            depth=depth,
            intrinsics=intrinsics,
            pose=pose,
            corners=empty_corners,
            scores=np.empty(0),
            stable_ids=np.empty(0, dtype=np.int64),
        )

    assert controller.stats["provider_calls"] == 2
    assert sorted(controller.track_manager.tracks) == [0]
    assert controller.track_manager.tracks[0].confirmed is True
    assert controller.track_manager.tracks[0].view_count == 2
    assert controller.track_manager.tracks[0].last_frame == 5
    assert (
        controller.track_manager.tracks[0].last_lifecycle_step
        == 1
    )
    assert (
        controller.track_manager.tracks[0].memory.last_frame_id
        == 5
    )
    assert controller.stats["candidate_discarded"] == 0


def test_keyframe_ttl_remains_available_as_legacy_ablation(tmp_path):
    image, depth, intrinsics, pose, proposal = synthetic_inputs()
    cfg = make_config(tmp_path)
    online = cfg["online_refinement"]
    online["inference_every_keyframes"] = 5
    online["candidate_lifecycle"] = {
        "ttl_clock": "keyframe",
        "archive_confirmed": False,
    }
    online["object_memory"]["track_ttl"] = 1
    controller = OnlineRefinementController(
        cfg,
        provider=SequenceProvider([[proposal], [proposal]]),
    )
    empty_corners = np.empty((0, 8, 3), dtype=np.float32)

    for frame_id in range(6):
        process(
            controller,
            frame_id=frame_id,
            image=image,
            depth=depth,
            intrinsics=intrinsics,
            pose=pose,
            corners=empty_corners,
            scores=np.empty(0),
            stable_ids=np.empty(0, dtype=np.int64),
        )

    assert controller.stats["provider_calls"] == 2
    assert sorted(controller.track_manager.tracks) == [1]
    assert controller.track_manager.tracks[1].confirmed is False
    assert controller.stats["candidate_discarded"] == 1


def test_confirmed_expired_track_is_archived_and_finalized(tmp_path):
    image, depth, intrinsics, pose, proposal = synthetic_inputs()
    cfg = make_config(tmp_path)
    online = cfg["online_refinement"]
    online["candidate_lifecycle"] = {
        "ttl_clock": "provider_call",
        "archive_confirmed": True,
    }
    online["object_memory"]["track_ttl"] = 1
    controller = OnlineRefinementController(
        cfg,
        provider=SequenceProvider(
            [[proposal], [proposal], [], []]
        ),
    )
    empty_corners = np.empty((0, 8, 3), dtype=np.float32)

    for frame_id in range(4):
        process(
            controller,
            frame_id=frame_id,
            image=image,
            depth=depth,
            intrinsics=intrinsics,
            pose=pose,
            corners=empty_corners,
            scores=np.empty(0),
            stable_ids=np.empty(0, dtype=np.int64),
        )

    assert controller.track_manager.tracks == {}
    assert sorted(controller.track_manager.archived_tracks) == [0]
    assert sorted(controller.supplemental_metadata) == [0]

    result = controller.finalize(
        global_corners=empty_corners,
        global_scores=np.empty(0),
        stable_ids=np.empty(0, dtype=np.int64),
    )

    assert result.boxes.shape == (1, 6)
    assert result.source_indices.tolist() == [-1]
    assert result.stable_ids.tolist() == [-1]
    assert result.summary["candidate_ttl_clock"] == "provider_call"
    assert result.summary["active_supplemental_tracks"] == 0
    assert result.summary["archived_supplemental_tracks"] == 1
    assert result.summary["confirmed_supplemental_tracks"] == 1
    assert result.summary["candidate_archived_total"] == 1

    controller.reset_scene("scene0001_00")
    assert controller.keyframe_count == 0
    assert controller.track_manager.tracks == {}
    assert controller.track_manager.archived_tracks == {}
    assert controller.supplemental_metadata == {}
    assert controller.stats["provider_calls"] == 0
    assert controller.stats["candidate_archived"] == 0


def test_unconfirmed_expired_track_metadata_is_discarded(tmp_path):
    image, depth, intrinsics, pose, proposal = synthetic_inputs()
    cfg = make_config(tmp_path)
    online = cfg["online_refinement"]
    online["candidate_lifecycle"] = {
        "ttl_clock": "provider_call",
        "archive_confirmed": True,
    }
    online["object_memory"]["track_ttl"] = 0
    controller = OnlineRefinementController(
        cfg,
        provider=SequenceProvider([[proposal], []]),
    )
    empty_corners = np.empty((0, 8, 3), dtype=np.float32)

    for frame_id in range(2):
        process(
            controller,
            frame_id=frame_id,
            image=image,
            depth=depth,
            intrinsics=intrinsics,
            pose=pose,
            corners=empty_corners,
            scores=np.empty(0),
            stable_ids=np.empty(0, dtype=np.int64),
        )

    assert controller.track_manager.tracks == {}
    assert controller.track_manager.archived_tracks == {}
    assert controller.supplemental_metadata == {}
    assert controller.stats["candidate_discarded"] == 1


def test_archived_track_can_be_absorbed_by_later_global_match(tmp_path):
    image, depth, intrinsics, pose, proposal = synthetic_inputs()
    cfg = make_config(tmp_path)
    online = cfg["online_refinement"]
    online["candidate_lifecycle"] = {
        "ttl_clock": "provider_call",
        "archive_confirmed": True,
    }
    online["object_memory"]["track_ttl"] = 1
    online["refit"]["enabled"] = False
    controller = OnlineRefinementController(
        cfg,
        provider=SequenceProvider(
            [[proposal], [proposal], [], [], [proposal]]
        ),
    )
    empty_corners = np.empty((0, 8, 3), dtype=np.float32)
    for frame_id in range(4):
        process(
            controller,
            frame_id=frame_id,
            image=image,
            depth=depth,
            intrinsics=intrinsics,
            pose=pose,
            corners=empty_corners,
            scores=np.empty(0),
            stable_ids=np.empty(0, dtype=np.int64),
        )
    assert sorted(controller.track_manager.archived_tracks) == [0]

    global_box = np.asarray([0.0, 0.0, 2.0, 0.4, 0.4, 0.4])
    global_corners = aabb_corners(
        global_box[:3], global_box[3:]
    )[None]
    process(
        controller,
        frame_id=4,
        image=image,
        depth=depth,
        intrinsics=intrinsics,
        pose=pose,
        corners=global_corners,
        scores=np.asarray([0.8]),
        stable_ids=np.asarray([7]),
    )

    assert controller.track_manager.archived_tracks == {}
    assert controller.supplemental_metadata == {}
    assert controller.global_tracks[7].stats.feature_count == 3
    assert (
        controller.global_tracks[7].stats.appearance_consistency
        == pytest.approx(1.0)
    )
    assert controller.global_tracks[7].stats.label == "chair"
    result = controller.finalize(
        global_corners=global_corners,
        global_scores=np.asarray([0.8]),
        stable_ids=np.asarray([7]),
    )
    assert result.source_indices.tolist() == [0]
    assert result.stable_ids.tolist() == [7]


def test_filtered_global_cannot_suppress_valid_supplemental_track(tmp_path):
    image, depth, intrinsics, pose, proposal = synthetic_inputs()
    cfg = make_config(tmp_path)
    online = cfg["online_refinement"]
    online["candidate_lifecycle"] = {
        "ttl_clock": "provider_call",
        "archive_confirmed": True,
    }
    online["refit"]["enabled"] = False
    online["output_filter"] = {"minimum_extent": 0.0095}
    controller = OnlineRefinementController(
        cfg,
        provider=SequenceProvider([[proposal], [proposal]]),
    )
    empty_corners = np.empty((0, 8, 3), dtype=np.float32)
    for frame_id in range(2):
        process(
            controller,
            frame_id=frame_id,
            image=image,
            depth=depth,
            intrinsics=intrinsics,
            pose=pose,
            corners=empty_corners,
            scores=np.empty(0),
            stable_ids=np.empty(0, dtype=np.int64),
        )

    center, dims = controller.track_manager.tracks[0].memory.aabb
    invalid_dims = dims.copy()
    invalid_dims[int(np.argmin(invalid_dims))] = 0.009
    invalid_global = aabb_corners(center, invalid_dims)[None]
    result = controller.finalize(
        global_corners=invalid_global,
        global_scores=np.asarray([0.9]),
        stable_ids=np.asarray([5]),
    )

    assert result.boxes.shape == (1, 6)
    assert result.source_indices.tolist() == [-1]
    assert result.stable_ids.tolist() == [-1]


def test_global_match_prevents_duplicate_supplemental_output(tmp_path):
    image, depth, intrinsics, pose, proposal = synthetic_inputs()
    controller = OnlineRefinementController(
        make_config(tmp_path),
        provider=SequenceProvider([[proposal], [proposal]]),
    )
    box = np.asarray([0.0, 0.0, 2.0, 0.4, 0.4, 0.4])
    corners = aabb_corners(box[:3], box[3:])[None]
    for frame_id in (0, 25):
        process(
            controller,
            frame_id=frame_id,
            image=image,
            depth=depth,
            intrinsics=intrinsics,
            pose=pose,
            corners=corners,
            scores=np.asarray([0.8]),
            stable_ids=np.asarray([0]),
        )
    result = controller.finalize(
        global_corners=corners,
        global_scores=np.asarray([0.8]),
        stable_ids=np.asarray([0]),
    )
    assert len(result.boxes) == 1
    assert result.source_indices.tolist() == [0]


def test_controller_records_multiview_instance_appearance(tmp_path):
    image, depth, intrinsics, pose, proposal = synthetic_inputs()

    def encode(_image, proposals):
        return [
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            for _ in proposals
        ]

    proposal_without_feature = SupplementalProposal(
        bbox=proposal.bbox,
        score=proposal.score,
        mask=proposal.mask,
        label=proposal.label,
    )
    controller = OnlineRefinementController(
        make_config(tmp_path),
        provider=SequenceProvider(
            [[proposal_without_feature], [proposal_without_feature]]
        ),
        appearance_encoder=encode,
    )
    box = np.asarray([0.0, 0.0, 2.0, 0.4, 0.4, 0.4])
    corners = aabb_corners(box[:3], box[3:])[None]
    for frame_id in (0, 25):
        process(
            controller,
            frame_id=frame_id,
            image=image,
            depth=depth,
            intrinsics=intrinsics,
            pose=pose,
            corners=corners,
            scores=np.asarray([0.8]),
            stable_ids=np.asarray([0]),
        )
    stats = controller.global_tracks[0].stats
    assert stats.feature_count == 2
    assert stats.appearance_consistency == pytest.approx(1.0)


def test_controller_resets_all_scene_state_on_scene_change(tmp_path):
    image, depth, intrinsics, pose, proposal = synthetic_inputs()
    provider = SequenceProvider([[proposal], [proposal]])
    controller = OnlineRefinementController(
        make_config(tmp_path),
        provider=provider,
    )
    box = np.asarray([0.0, 0.0, 2.0, 0.4, 0.4, 0.4])
    corners = aabb_corners(box[:3], box[3:])[None]
    process(
        controller,
        frame_id=0,
        image=image,
        depth=depth,
        intrinsics=intrinsics,
        pose=pose,
        corners=corners,
        scores=np.asarray([0.8]),
        stable_ids=np.asarray([3]),
    )
    assert set(controller.global_tracks) == {3}

    controller.process_keyframe(
        image=image,
        depth=depth,
        intrinsics=intrinsics,
        camera_to_world=pose,
        frame_id=0,
        scene_id="scene0001_00",
        global_corners=corners,
        global_scores=np.asarray([0.7]),
        stable_ids=np.asarray([7]),
    )

    assert provider.calls == 2
    assert controller.scene_id == "scene0001_00"
    assert controller.keyframe_count == 1
    assert controller.stats["keyframes"] == 1
    assert set(controller.global_tracks) == {7}
    assert controller.global_tracks[7].memory.observation_count == 1


def test_minimum_extent_filter_runs_before_soft_nms(tmp_path):
    cfg = make_config(tmp_path, quality=True)
    cfg["online_refinement"]["output_filter"] = {
        "minimum_extent": 0.3,
    }
    controller = OnlineRefinementController(
        cfg,
        provider=SequenceProvider([[]]),
    )
    small = np.asarray([0.0, 0.0, 2.0, 0.2, 0.2, 0.2])
    valid = np.asarray([0.0, 0.0, 2.0, 0.4, 0.4, 0.4])
    corners = np.stack(
        (
            aabb_corners(small[:3], small[3:]),
            aabb_corners(valid[:3], valid[3:]),
        )
    )

    result = controller.finalize(
        global_corners=corners,
        global_scores=np.asarray([0.9, 0.8]),
        stable_ids=np.asarray([0, 1]),
    )

    assert result.boxes.shape == (1, 6)
    assert result.stable_ids.tolist() == [1]
    assert result.scores.tolist() == pytest.approx([0.8])


def test_diagnostics_are_pickle_free_and_training_compatible(tmp_path):
    image, depth, intrinsics, pose, proposal = synthetic_inputs()
    controller = OnlineRefinementController(
        make_config(tmp_path, diagnostics=True),
        provider=SequenceProvider([[proposal], [proposal]]),
    )
    box = np.asarray([0.0, 0.0, 2.0, 0.4, 0.4, 0.4])
    corners = aabb_corners(box[:3], box[3:])[None]
    for frame_id in (0, 25):
        process(
            controller,
            frame_id=frame_id,
            image=image,
            depth=depth,
            intrinsics=intrinsics,
            pose=pose,
            corners=corners,
            scores=np.asarray([0.8]),
            stable_ids=np.asarray([0]),
        )
    controller.finalize(
        global_corners=corners,
        global_scores=np.asarray([0.8]),
        stable_ids=np.asarray([0]),
        scene_id="scene0000_00",
    )
    path = tmp_path / "scene0000_00_tracks.npz"
    assert path.is_file()
    with np.load(path, allow_pickle=False) as payload:
        assert payload["boxes"].shape == (1, 6)
        assert payload["points"].shape == (1, 32, 3)
        assert payload["point_mask"].dtype == np.bool_
        assert payload["quality_features"].shape == (1, 12)
        assert payload["scene_id"].item() == "scene0000_00"


def test_invalid_config_and_misaligned_global_inputs_fail(tmp_path):
    with pytest.raises(ValueError, match="Unknown online_refinement key"):
        resolve_online_refinement_config(
            {"enabled": True, "not_a_real_option": 1}
        )
    for invalid_projection_iou in (-0.01, 1.01, True):
        cfg = make_config(tmp_path)
        cfg["online_refinement"]["supplemental_output"][
            "min_projection_iou"
        ] = invalid_projection_iou
        with pytest.raises(ValueError, match="min_projection_iou"):
            resolve_online_refinement_config(cfg)
    controller = OnlineRefinementController(
        make_config(tmp_path),
        provider=SequenceProvider([[]]),
    )
    image, depth, intrinsics, pose, _ = synthetic_inputs()
    with pytest.raises(ValueError, match="must align"):
        process(
            controller,
            frame_id=0,
            image=image,
            depth=depth,
            intrinsics=intrinsics,
            pose=pose,
            corners=np.empty((0, 8, 3)),
            scores=np.asarray([0.5]),
            stable_ids=np.empty(0, dtype=np.int64),
        )
