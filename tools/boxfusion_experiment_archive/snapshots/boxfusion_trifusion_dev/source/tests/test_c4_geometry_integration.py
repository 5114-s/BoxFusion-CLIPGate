"""CPU-only integration contracts for the B6 + C4 geometry observer.

C4 is deliberately a second, read-only Mask-RGBD evidence stream.  These
tests install its multi-view memory directly so the regression contract does
not depend on a SAM3 checkpoint, proposal-cache rasterization, or a GPU.
"""

from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest

from boxfusion.ap50_safety_gate import AP50SafetyGate
from boxfusion.object_memory import (
    MemoryViewRecord,
    ObjectGeometryMemory,
    ObjectObservation,
    aabb_corners,
    project_aabb_to_image,
)
from boxfusion.online_ablation import apply_online_ablation_profile
from boxfusion.online_refinement import (
    DEFAULT_ONLINE_REFINEMENT_CONFIG,
    EvidenceStats,
    GlobalEvidence,
    OnlineRefinementController,
    TRIFUSION_GATE_FEATURE_NAMES,
    ViewEvidence,
)
from boxfusion.supplemental_proposals import SupplementalProposal
from tools.export_trifusion_supplemental_candidates import (
    export_directory,
)
from tools.build_trifusion_geometry_candidates import (
    COMBINED_FEATURE_NAMES,
)
from tools.report_trifusion_oracles import (
    load_supplemental_candidates,
)


class _NoopProvider:
    def predict(self, images, *, frame_ids=None):
        assert frame_ids is not None
        assert len(images) == len(frame_ids)
        return [[] for _ in images]


class _ExplodingSecondaryProvider:
    def predict(self, images, *, frame_ids=None):
        raise RuntimeError("synthetic C4 cache failure")


class _SequenceSecondaryProvider:
    def __init__(self, batches):
        self.batches = list(batches)
        self.calls = 0

    def predict(self, images, *, frame_ids=None):
        assert len(images) == 1
        assert frame_ids is not None and len(frame_ids) == 1
        result = self.batches[min(self.calls, len(self.batches) - 1)]
        self.calls += 1
        return [result]


def _config(tmp_path, profile, *, diagnostics=False):
    online = deepcopy(DEFAULT_ONLINE_REFINEMENT_CONFIG)
    online["enabled"] = False
    online["supplemental_proposals"] = {"enabled": False}
    online["object_memory"] = {"enabled": True}
    # Both profiles receive the exact historical B6 feature contract before
    # profile application.  The injected scorer makes a checkpoint unnecessary.
    online["quality"].update(
        {
            "mode": "iou_mlp",
            "feature_geometry": "original",
            "refiner_quality_override": 0.5,
            "soft_nms": {"enabled": False},
        }
    )
    online["output_filter"].update(
        {
            "minimum_extent": 0.0,
            "final_minimum_extent": 0.0,
        }
    )
    online["diagnostics"].update(
        {
            "enabled": diagnostics,
            "dump_track_memory": diagnostics,
            "root": str(tmp_path),
            "point_count": 32,
        }
    )
    config = apply_online_ablation_profile(
        {
            "dataset": "scannet",
            "detection": {"score_thresh": 0.40},
            "online_refinement": online,
        },
        profile,
    )
    runtime = config["online_refinement"]
    runtime["output_filter"].update(
        {
            "minimum_extent": 0.0,
            "final_minimum_extent": 0.0,
        }
    )
    if profile in {
        "b6_c4_mask_rgbd_observer",
        "trifusion_plus10_observer",
    }:
        c4 = runtime["generic_local_geometry_refiner"]
        c4["secondary_proposals"] = {
            "enabled": True,
            "provider": "cache_only",
            "cache": {
                "enabled": True,
                "directory": str(tmp_path / "unused-cache"),
                "write": False,
                "namespace": "cpu-c4-regression-v1",
                "missing_policy": "error",
            },
        }
        # Keep the controller gates permissive in this integration fixture:
        # the standalone refiner tests own the production-threshold behavior.
        c4.update(
            {
                "minimum_mean_valid_depth_ratio": 0.0,
                "minimum_projection_views": 1,
                "minimum_projection_view_iou": 0.0,
                "minimum_weighted_projection_iou": 0.0,
                "maximum_projection_drop": 1.0,
                "minimum_raw_candidate_support": 0.0,
                "maximum_raw_support_drop": 1.0,
                "maximum_center_shift_ratio": 1.0,
                "minimum_extent_ratio": 0.01,
                "maximum_extent_ratio": 10.0,
                "minimum_original_candidate_iou": 0.0,
                "maximum_overlap_increase": 1.0,
                "maximum_new_overlap": 1.0,
            }
        )
        c4["proposal"].update(
            {
                "min_points_per_view": 24,
                "min_total_points": 96,
                "boundary_min_points_per_view": 8,
            }
        )
    return config


def _object_points(
    center=(0.0, 0.0, 2.0),
    lower=(-0.30, -0.20, -0.20),
    upper=(0.30, 0.20, 0.20),
):
    axes = [
        np.linspace(lower[index], upper[index], (9, 7, 7)[index])
        + center[index]
        for index in range(3)
    ]
    return np.asarray(
        np.meshgrid(*axes, indexing="ij"), dtype=np.float32
    ).reshape(3, -1).T


def _view_context(box):
    intrinsics = np.asarray(
        [
            [100.0, 0.0, 64.0],
            [0.0, 100.0, 64.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    camera_to_world = np.eye(4, dtype=np.float32)
    bbox = project_aabb_to_image(
        box[:3],
        box[3:],
        intrinsics,
        camera_to_world,
        (128, 128),
    )
    assert bbox is not None
    return intrinsics, camera_to_world, bbox


def _install_primary_evidence(controller, stable_id, box, label):
    points = _object_points(center=tuple(np.asarray(box)[:3]))
    memory = ObjectGeometryMemory(stable_id, controller.object_config)
    for frame_id in range(2):
        memory.add_observation(
            ObjectObservation(
                points_world=points,
                confidence=0.85,
                mask_pixels=len(points),
                valid_depth_pixels=len(points),
                projection_mask_iou=0.90,
                camera_position=np.zeros(3, dtype=np.float32),
            ),
            frame_id,
        )
    intrinsics, pose, bbox = _view_context(box)
    stats = EvidenceStats()
    stats.scores = [0.85, 0.82]
    stats.label_votes[label] = 1.67
    stats.view_records = [
        ViewEvidence(
            frame_index=frame_id,
            score=score,
            bbox=bbox,
            intrinsics=intrinsics,
            camera_to_world=pose,
            image_shape=(128, 128),
            area_ratio=0.08,
        )
        for frame_id, score in enumerate((0.85, 0.82))
    ]
    stats.box_records = [
        (score, frame_id, np.asarray(box, dtype=np.float32).copy())
        for frame_id, score in enumerate((0.85, 0.82))
    ]
    controller.global_tracks[stable_id] = GlobalEvidence(
        stable_id=stable_id,
        memory=memory,
        stats=stats,
        detector_score=0.8,
        last_box=np.asarray(box, dtype=np.float32).copy(),
    )


def _install_c4_multiview_memory(controller, stable_id, box):
    base_points = _object_points(center=tuple(np.asarray(box)[:3]))
    cameras = (
        (-2.0, -2.0, 0.0),
        (2.0, 2.0, 4.0),
        (-2.0, 2.0, 0.0),
        (2.0, -2.0, 4.0),
    )
    records = []
    for frame_id, camera in enumerate(cameras):
        points = base_points.copy()
        points[:, (frame_id + 1) % 3] += (
            frame_id - 1.5
        ) * 0.0005
        records.append(
            MemoryViewRecord(
                frame_id=frame_id,
                points_world=points,
                quality=0.95 - 0.03 * frame_id,
                confidence=0.90,
                valid_depth_ratio=0.95,
                projection_mask_iou=1.0,
                camera_position=np.asarray(camera, dtype=np.float32),
            )
        )

    memory = ObjectGeometryMemory(
        stable_id, controller.generic_geometry_object_config
    )
    # Install explicit independent MemoryViewRecord rows; this exercises the
    # exact Top-K interface consumed by the generic local refiner.
    memory._points = base_points.copy()
    memory._set_view_candidates(records)
    memory.observation_count = len(records)
    memory.unique_view_count = len(records)
    memory.first_frame_id = 0
    memory.last_frame_id = len(records) - 1

    intrinsics, pose, bbox = _view_context(box)
    stats = EvidenceStats()
    stats.scores = [record.confidence for record in records]
    stats.label_votes["chair"] = sum(stats.scores)
    stats.view_records = [
        ViewEvidence(
            frame_index=record.frame_id,
            score=record.confidence,
            bbox=bbox,
            intrinsics=intrinsics,
            camera_to_world=pose,
            image_shape=(128, 128),
            area_ratio=0.08,
        )
        for record in records
    ]
    controller.generic_geometry_global_tracks[stable_id] = GlobalEvidence(
        stable_id=stable_id,
        memory=memory,
        stats=stats,
        detector_score=0.8,
        last_box=np.asarray(box, dtype=np.float32).copy(),
    )


def _inputs():
    boxes = np.asarray(
        [
            [0.0, 0.0, 2.0, 0.80, 0.60, 0.50],
            [1.0, 0.0, 2.0, 0.90, 0.70, 0.60],
        ],
        dtype=np.float32,
    )
    return {
        "global_corners": np.stack(
            [aabb_corners(box[:3], box[3:]) for box in boxes]
        ).astype(np.float32),
        "global_scores": np.asarray([0.81, 0.73], dtype=np.float32),
        "stable_ids": np.asarray([7, 8], dtype=np.int64),
    }, boxes


def _assert_same_bytes(left, right):
    array_fields = (
        "corners",
        "boxes",
        "scores",
        "source_indices",
        "stable_ids",
        "quality_features",
        "refit_original_boxes",
        "refit_original_corners",
        "refit_applied",
        "refit_changed_axes",
        "refit_boundary_delta",
        "refit_local_original_boxes",
        "refit_local_candidate_boxes",
        "refit_local_basis",
        "refit_local_frame_valid",
    )
    for field in array_fields:
        left_value = getattr(left, field)
        right_value = getattr(right, field)
        assert left_value.shape == right_value.shape, field
        assert left_value.dtype == right_value.dtype, field
        assert left_value.tobytes() == right_value.tobytes(), field
    assert left.labels == right.labels
    assert left.refit_reasons == right.refit_reasons


def _paired_results(tmp_path, *, diagnostics=False):
    captured_b6 = []
    captured_c4 = []

    def scorer(target):
        def score(mapping):
            target.append(dict(mapping))
            return 0.15 + 0.70 * float(mapping["detector_score"])

        return score

    b6 = OnlineRefinementController(
        _config(tmp_path / "b6", "quality_only"),
        provider=_NoopProvider(),
        quality_scorer=scorer(captured_b6),
    )
    c4 = OnlineRefinementController(
        _config(
            tmp_path / "c4",
            "b6_c4_mask_rgbd_observer",
            diagnostics=diagnostics,
        ),
        provider=_NoopProvider(),
        generic_geometry_provider=_NoopProvider(),
        quality_scorer=scorer(captured_c4),
    )
    inputs, boxes = _inputs()
    for controller in (b6, c4):
        _install_primary_evidence(controller, 7, boxes[0], "chair")
        _install_primary_evidence(controller, 8, boxes[1], "table")
    _install_c4_multiview_memory(c4, 7, boxes[0])
    b6_result = b6.finalize(**inputs)
    c4_result = c4.finalize(
        **inputs,
        scene_id="scene0000_00" if diagnostics else None,
    )
    return b6_result, c4_result, c4, captured_b6, captured_c4


def test_c4_observer_is_bit_exact_to_quality_only_b6(tmp_path):
    b6, c4, controller, captured_b6, captured_c4 = _paired_results(
        tmp_path
    )

    _assert_same_bytes(c4, b6)
    assert captured_c4 == captured_b6
    assert c4.summary["c4_attempted"] == 1
    assert c4.summary["c4_proposed"] == 1
    assert c4.summary["c4_verified"] == 1
    assert c4.summary["c4_applied"] == 0
    assert (
        c4.summary["mutation_generic_local_geometry_enabled"] is False
    )
    runtime = controller._last_c4_runtime[7]
    assert runtime["attempted"] is True
    assert runtime["proposed"] is True
    assert runtime["verified"] is True
    assert runtime["applied"] is False


def test_c4_diagnostics_are_aligned_and_explicitly_non_mutating(tmp_path):
    b6, c4, _, _, _ = _paired_results(tmp_path, diagnostics=True)
    _assert_same_bytes(c4, b6)

    path = tmp_path / "c4" / "scene0000_00_tracks.npz"
    assert path.is_file()
    with np.load(path, allow_pickle=False) as payload:
        assert (
            str(payload["c4_diagnostics_schema"].item())
            == "generic_mask_rgbd_local_geometry_v2"
        )
        assert bool(payload["c4_enabled"].item()) is True
        assert bool(payload["c4_mutation_enabled"].item()) is False
        assert payload["c4_result_indices"].shape == (2,)
        assert payload["c4_stable_ids"].tolist() == [7, 8]
        assert payload["c4_attempted"].shape == (2,)
        assert payload["c4_proposed"].shape == (2,)
        assert payload["c4_verified"].shape == (2,)
        assert payload["c4_applied"].shape == (2,)
        assert payload["c4_attempted"].tolist() == [True, False]
        assert payload["c4_proposed"].tolist() == [True, False]
        assert payload["c4_verified"].tolist() == [True, False]
        assert not np.any(payload["c4_applied"])
        assert payload["c4_original_boxes"].shape == (2, 6)
        assert payload["c4_candidate_boxes"].shape == (2, 6)
        assert payload["c4_original_corners"].shape == (2, 8, 3)
        assert payload["c4_candidate_corners"].shape == (2, 8, 3)
        np.testing.assert_array_equal(
            payload["c4_original_corners"], c4.corners
        )
        assert payload["c4_memory_points"].shape == (2, 32, 3)
        assert payload["c4_view_points"].shape == (2, 5, 32, 3)
        summary = json.loads(str(payload["summary_json"].item()))
        assert (
            summary["mutation_generic_local_geometry_enabled"] is False
        )
        assert summary["c4_applied"] == 0


def test_secondary_provider_failure_is_fail_open_and_b6_exact(tmp_path):
    captured_b6 = []
    captured_c4 = []

    def scorer(target):
        def score(mapping):
            target.append(dict(mapping))
            return 0.25 + 0.50 * float(mapping["detector_score"])

        return score

    b6 = OnlineRefinementController(
        _config(tmp_path / "b6", "quality_only"),
        provider=_NoopProvider(),
        quality_scorer=scorer(captured_b6),
    )
    c4 = OnlineRefinementController(
        _config(tmp_path / "c4", "b6_c4_mask_rgbd_observer"),
        provider=_NoopProvider(),
        generic_geometry_provider=_ExplodingSecondaryProvider(),
        quality_scorer=scorer(captured_c4),
    )
    inputs, boxes = _inputs()
    for controller in (b6, c4):
        _install_primary_evidence(controller, 7, boxes[0], "chair")
        _install_primary_evidence(controller, 8, boxes[1], "table")

    frame = {
        "image": np.zeros((16, 16, 3), dtype=np.uint8),
        "depth": np.full((16, 16), 2.0, dtype=np.float32),
        "intrinsics": np.asarray(
            [
                [40.0, 0.0, 7.5],
                [0.0, 40.0, 7.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "camera_to_world": np.eye(4, dtype=np.float32),
        "frame_id": 0,
        "scene_id": "scene0000_00",
        **inputs,
    }
    b6.process_keyframe(**frame)
    c4.process_keyframe(**frame)
    b6_result = b6.finalize(**inputs)
    c4_result = c4.finalize(**inputs)

    _assert_same_bytes(c4_result, b6_result)
    assert captured_c4 == captured_b6
    assert c4_result.summary["c4_fail_open"] is True
    assert c4_result.summary["c4_applied"] == 0
    assert "synthetic C4 cache failure" in c4_result.summary["c4_error"]
    assert c4.generic_geometry_global_tracks == {}


def _trifusion_results(tmp_path, *, diagnostics, gate=None):
    captured_b6 = []
    captured_c4 = []
    captured_trifusion = []

    def scorer(target):
        def score(mapping):
            target.append(dict(mapping))
            return 0.15 + 0.70 * float(mapping["detector_score"])

        return score

    b6 = OnlineRefinementController(
        _config(tmp_path / "b6", "quality_only"),
        provider=_NoopProvider(),
        quality_scorer=scorer(captured_b6),
    )
    c4 = OnlineRefinementController(
        _config(
            tmp_path / "c4",
            "b6_c4_mask_rgbd_observer",
            diagnostics=diagnostics,
        ),
        provider=_NoopProvider(),
        generic_geometry_provider=_NoopProvider(),
        quality_scorer=scorer(captured_c4),
    )
    trifusion_config = _config(
        tmp_path / "trifusion",
        "trifusion_plus10_observer",
        diagnostics=diagnostics,
    )
    if gate is not None:
        trifusion_config["online_refinement"]["trifusion_observer"][
            "safety_gate"
        ].update(
            {
                "enabled": True,
                "checkpoint": str(tmp_path / "gate.npz"),
                "collect_diagnostics": True,
                "mutate": False,
            }
        )
    trifusion = OnlineRefinementController(
        trifusion_config,
        provider=_NoopProvider(),
        generic_geometry_provider=_NoopProvider(),
        quality_scorer=scorer(captured_trifusion),
        trifusion_ap50_gate=gate,
    )
    inputs, boxes = _inputs()
    for controller in (b6, c4, trifusion):
        _install_primary_evidence(controller, 7, boxes[0], "chair")
        _install_primary_evidence(controller, 8, boxes[1], "table")
    for controller in (c4, trifusion):
        _install_c4_multiview_memory(controller, 7, boxes[0])

    b6_result = b6.finalize(**inputs)
    scene = "scene0000_00" if diagnostics else None
    c4_result = c4.finalize(**inputs, scene_id=scene)
    trifusion_result = trifusion.finalize(**inputs, scene_id=scene)
    return (
        b6_result,
        c4_result,
        trifusion_result,
        trifusion,
        captured_b6,
        captured_c4,
        captured_trifusion,
    )


def test_trifusion_observer_is_bit_exact_to_c4_and_quality_only_b6(
    tmp_path,
):
    (
        b6,
        c4,
        trifusion,
        controller,
        captured_b6,
        captured_c4,
        captured_trifusion,
    ) = _trifusion_results(tmp_path, diagnostics=False)

    _assert_same_bytes(c4, b6)
    _assert_same_bytes(trifusion, b6)
    assert captured_trifusion == captured_c4 == captured_b6
    assert trifusion.summary["mutation_trifusion_enabled"] is False
    assert trifusion.summary["trifusion_applied"] == 0
    assert trifusion.summary["trifusion_attempted"] == 1
    runtime = controller._last_trifusion_runtime[7]
    assert runtime["stable_id"] == 7
    assert runtime["attempted"] is True
    assert runtime["candidate_valid"] is True
    assert runtime["applied"] is False


def test_trifusion_diagnostics_are_strict_aligned_and_leave_c4_unchanged(
    tmp_path,
):
    b6, c4, trifusion, _, _, _, _ = _trifusion_results(
        tmp_path, diagnostics=True
    )
    _assert_same_bytes(c4, b6)
    _assert_same_bytes(trifusion, b6)

    c4_path = tmp_path / "c4" / "scene0000_00_tracks.npz"
    trifusion_path = (
        tmp_path / "trifusion" / "scene0000_00_tracks.npz"
    )
    with (
        np.load(c4_path, allow_pickle=False) as c4_payload,
        np.load(trifusion_path, allow_pickle=False) as payload,
    ):
        for key in c4_payload.files:
            if key.startswith("c4_") and key != "c4_config_json":
                np.testing.assert_array_equal(
                    payload[key], c4_payload[key], err_msg=key
                )
        c4_config = json.loads(str(c4_payload["c4_config_json"].item()))
        trifusion_c4_config = json.loads(
            str(payload["c4_config_json"].item())
        )
        for config in (c4_config, trifusion_c4_config):
            config["secondary_proposals"]["cache"]["directory"] = (
                "<fixture-cache>"
            )
        assert trifusion_c4_config == c4_config
        assert (
            str(payload["trifusion_diagnostics_schema"].item())
            == "boxfusion.trifusion.occupancy_msr_observer.v1"
        )
        assert bool(payload["trifusion_enabled"].item()) is True
        assert (
            bool(payload["trifusion_mutation_enabled"].item()) is False
        )
        assert payload["trifusion_result_indices"].tolist() == [0]
        assert payload["trifusion_stable_ids"].tolist() == [7]
        assert payload["trifusion_feature_names"].shape == (48,)
        assert payload["trifusion_features"].shape == (1, 48)
        assert np.isfinite(payload["trifusion_features"]).all()
        assert bool(payload["trifusion_gate_enabled"].item()) is False
        assert payload["trifusion_gate_feature_names"].shape == (61,)
        assert payload["trifusion_gate_features"].shape == (1, 61)
        assert payload["trifusion_gate_evaluated"].tolist() == [False]
        assert payload["trifusion_gate_accepted"].tolist() == [False]
        assert payload["trifusion_original_corners"].shape == (1, 8, 3)
        assert payload["trifusion_candidate_corners"].shape == (1, 8, 3)
        np.testing.assert_array_equal(
            payload["trifusion_original_corners"][0],
            trifusion.corners[0],
        )
        assert payload["trifusion_candidate_valid"].tolist() == [True]
        assert payload["trifusion_is_candidate"].shape == (1,)
        assert payload["trifusion_candidate_verified"].shape == (1,)
        assert payload["trifusion_applied"].tolist() == [False]
        assert payload["trifusion_reason"].shape == (1,)
        assert (
            str(
                payload[
                    "trifusion_missing_diagnostics_schema"
                ].item()
            )
            == "boxfusion.trifusion.missing_graph_observer.v1"
        )
        assert bool(payload["trifusion_missing_enabled"].item()) is True
        assert (
            bool(
                payload[
                    "trifusion_missing_mutation_enabled"
                ].item()
            )
            is False
        )
        assert payload["trifusion_missing_feature_names"].shape == (20,)
        assert payload["trifusion_missing_corners"].shape == (0, 8, 3)
        assert payload["trifusion_missing_features"].shape == (0, 20)
        assert payload["trifusion_missing_applied"].shape == (0,)
        summary = json.loads(str(payload["summary_json"].item()))
        assert summary["mutation_trifusion_enabled"] is False
        assert summary["trifusion_applied"] == 0


def test_trifusion_ap50_gate_observes_verified_m3_without_mutation(
    tmp_path,
):
    assert tuple(COMBINED_FEATURE_NAMES) == (
        TRIFUSION_GATE_FEATURE_NAMES
    )
    feature_dim = len(TRIFUSION_GATE_FEATURE_NAMES)
    raw_bias = np.asarray(
        [
            np.arctanh(0.20),
            np.log(0.01**2),
            5.0,
            -5.0,
            0.0,
            1.0,
            5.0,
            5.0,
        ],
        dtype=np.float32,
    )
    gate = AP50SafetyGate(
        feature_names=TRIFUSION_GATE_FEATURE_NAMES,
        weights=(
            np.zeros((feature_dim, len(raw_bias)), dtype=np.float32),
        ),
        biases=(raw_bias,),
        feature_mean=np.zeros(feature_dim, dtype=np.float32),
        feature_scale=np.ones(feature_dim, dtype=np.float32),
    )
    b6, _, trifusion, controller, _, _, _ = _trifusion_results(
        tmp_path, diagnostics=True, gate=gate
    )

    _assert_same_bytes(trifusion, b6)
    assert trifusion.summary["trifusion_applied"] == 0
    assert trifusion.summary["trifusion_gate_enabled"] is True
    assert trifusion.summary["trifusion_gate_evaluated"] == 1
    assert trifusion.summary["trifusion_gate_accepted"] == 1
    runtime = controller._last_trifusion_runtime[7]
    assert runtime["candidate_verified"] is True
    assert runtime["gate_evaluated"] is True
    assert runtime["gate_accepted"] is True
    assert runtime["applied"] is False
    assert runtime["gate_features"].shape == (61,)

    path = tmp_path / "trifusion" / "scene0000_00_tracks.npz"
    with np.load(path, allow_pickle=False) as payload:
        assert (
            str(
                payload[
                    "trifusion_gate_diagnostics_schema"
                ].item()
            )
            == "boxfusion.trifusion.ap50_safety_observer.v1"
        )
        assert payload["trifusion_gate_feature_names"].shape == (61,)
        assert payload["trifusion_gate_features"].shape == (1, 61)
        assert payload["trifusion_gate_evaluated"].tolist() == [True]
        assert payload["trifusion_gate_accepted"].tolist() == [True]
        assert payload["trifusion_applied"].tolist() == [False]


def test_trifusion_secondary_unmatched_masks_confirm_missing_graph_only(
    tmp_path,
):
    mask = np.zeros((16, 16), dtype=bool)
    mask[3:13, 3:13] = True
    proposal = SupplementalProposal(
        bbox=np.asarray([3.0, 3.0, 13.0, 13.0], dtype=np.float32),
        score=0.90,
        mask=mask,
        label="chair",
        feature=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    captured_b6 = []
    captured_trifusion = []

    def scorer(target):
        def score(mapping):
            target.append(dict(mapping))
            return 0.15 + 0.70 * float(mapping["detector_score"])

        return score

    b6 = OnlineRefinementController(
        _config(tmp_path / "b6_missing", "quality_only"),
        provider=_NoopProvider(),
        quality_scorer=scorer(captured_b6),
    )
    trifusion = OnlineRefinementController(
        _config(
            tmp_path / "trifusion_missing",
            "trifusion_plus10_observer",
            diagnostics=True,
        ),
        provider=_NoopProvider(),
        generic_geometry_provider=_SequenceSecondaryProvider(
            [[proposal], [proposal]]
        ),
        quality_scorer=scorer(captured_trifusion),
    )
    inputs, boxes = _inputs()
    for controller in (b6, trifusion):
        _install_primary_evidence(controller, 7, boxes[0], "chair")
        _install_primary_evidence(controller, 8, boxes[1], "table")

    # A shifted principal point places the mask at x < -1 m, safely away
    # from both frozen B6 globals while keeping neighbouring depth pixels
    # close enough for the M1 connected-component gate.
    frame = {
        "image": np.zeros((16, 16, 3), dtype=np.uint8),
        "depth": np.full((16, 16), 2.0, dtype=np.float32),
        "intrinsics": np.asarray(
            [
                [40.0, 0.0, 40.0],
                [0.0, 40.0, 7.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "camera_to_world": np.eye(4, dtype=np.float32),
        "scene_id": "scene0000_00",
        **inputs,
    }
    for frame_id in (0, 1):
        b6.process_keyframe(**frame, frame_id=frame_id)
        trifusion.process_keyframe(**frame, frame_id=frame_id)

    b6_result = b6.finalize(**inputs)
    trifusion_result = trifusion.finalize(
        **inputs, scene_id="scene0000_00"
    )
    _assert_same_bytes(trifusion_result, b6_result)
    assert captured_trifusion == captured_b6
    assert trifusion_result.summary[
        "trifusion_missing_provider_calls"
    ] == 2
    assert trifusion_result.summary[
        "trifusion_missing_unmatched"
    ] == 2
    assert trifusion_result.summary[
        "trifusion_missing_candidates"
    ] == 1
    assert len(trifusion._last_trifusion_missing_candidates) == 1
    candidate = trifusion._last_trifusion_missing_candidates[0]
    assert candidate.confirmed is True
    assert len(candidate.frame_ids) == 2
    assert candidate.feature_vector.shape == (20,)

    path = (
        tmp_path
        / "trifusion_missing"
        / "scene0000_00_tracks.npz"
    )
    with np.load(path, allow_pickle=False) as payload:
        assert payload["trifusion_missing_candidate_ids"].shape == (1,)
        assert payload["trifusion_missing_corners"].shape == (1, 8, 3)
        assert payload["trifusion_missing_features"].shape == (1, 20)
        assert payload["trifusion_missing_valid"].tolist() == [True]
        assert payload["trifusion_missing_verified"].tolist() == [True]
        assert payload["trifusion_missing_confirmed"].tolist() == [True]
        assert payload["trifusion_missing_applied"].tolist() == [False]

    scene_list = tmp_path / "one_scene.txt"
    scene_list.write_text("scene0000_00\n", encoding="utf-8")
    export_root = tmp_path / "supplemental_export"
    export_summary = export_directory(
        diagnostics_root=tmp_path / "trifusion_missing",
        scene_list=scene_list,
        output_root=export_root,
    )
    assert export_summary["candidates"] == 1
    exported = load_supplemental_candidates(
        export_root
        / "scene0000_00_supplemental_candidates.npz",
        expected_scene_id="scene0000_00",
    )
    assert exported.candidate_ids == (
        "scene0000_00:missing_graph:track:0",
    )
    assert exported.candidate_verified.tolist() == [True]


@pytest.mark.parametrize(
    "mutation,message",
    [
        (
            lambda online: online["trifusion_observer"].__setitem__(
                "mutate", True
            ),
            "mutate must be false",
        ),
        (
            lambda online: online["trifusion_observer"][
                "proposal"
            ].__setitem__("unknown_key", 1),
            "Unknown online_refinement.trifusion_observer",
        ),
        (
            lambda online: online["trifusion_observer"][
                "safety_gate"
            ].__setitem__("mutate", True),
            "safety gate is observer-only",
        ),
        (
            lambda online: online["trifusion_observer"][
                "safety_gate"
            ].update(
                {
                    "enabled": True,
                    "collect_diagnostics": True,
                    "checkpoint": None,
                }
            ),
            "requires checkpoint",
        ),
        (
            lambda online: online["quality"]["soft_nms"].__setitem__(
                "enabled", True
            ),
            "strict quality-only B6 child",
        ),
    ],
)
def test_trifusion_resolver_fails_closed(tmp_path, mutation, message):
    config = _config(
        tmp_path, "trifusion_plus10_observer", diagnostics=False
    )
    mutation(config["online_refinement"])
    with pytest.raises(ValueError, match=message):
        OnlineRefinementController(
            config,
            provider=_NoopProvider(),
            generic_geometry_provider=_NoopProvider(),
            quality_scorer=lambda _: 0.5,
        )
