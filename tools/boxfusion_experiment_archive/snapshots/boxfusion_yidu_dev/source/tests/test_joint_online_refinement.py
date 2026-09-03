"""Runtime contract tests for the batched B3 -> B5 + B6-v2 head."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from boxfusion.joint_local_head import JointLocalHeadConfig
from boxfusion.object_memory import (
    MemoryViewRecord,
    ObjectGeometryMemory,
    aabb_corners,
)
from boxfusion.online_refinement import (
    DEFAULT_ONLINE_REFINEMENT_CONFIG,
    EvidenceStats,
    GlobalEvidence,
    OnlineRefinementController,
    resolve_online_refinement_config,
)


class _NoopProvider:
    def predict(self, images, *, frame_ids=None):
        return [[] for _ in images]


def _joint_config(tmp_path, *, diagnostics: bool = False) -> dict:
    online = deepcopy(DEFAULT_ONLINE_REFINEMENT_CONFIG)
    online["enabled"] = True
    online["supplemental_proposals"] = {"enabled": False}
    online["appearance_memory"]["enabled"] = False
    online["candidate_lifecycle"] = {
        "ttl_clock": "provider_call",
        "archive_confirmed": False,
    }
    online["object_memory"].update(
        {
            "enabled": True,
            "top_k_views": 5,
            "max_view_candidates": 12,
            "view_diversity_weight": 0.4,
            "voxel_size": 0.0,
            "max_points_per_object": 256,
            "min_points_for_aabb": 4,
        }
    )
    online["refit"].update(
        {
            "enabled": False,
            "min_views": 2,
            "min_points": 4,
            "max_center_shift_ratio": 0.16,
            "min_extent_ratio": 0.80,
            "max_extent_ratio": 1.25,
            "min_original_point_support": 0.0,
            "min_candidate_point_support": 0.0,
            "max_candidate_support_drop": 1.0,
            "min_reprojection_iou": 0.0,
            "min_reprojection_improvement": -1.0,
        }
    )
    online["box_refiner"]["enabled"] = False
    online["quality"]["enabled"] = False
    online["quality"]["soft_nms"]["enabled"] = False
    online["supplemental_output"]["enabled"] = False
    online["output_filter"]["minimum_extent"] = 0.0
    online["joint_local_head"].update(
        {
            "enabled": True,
            "checkpoint": None,
            "device": "cpu",
            "max_views": 5,
            "points_per_view": 16,
            "improvement_threshold": 0.50,
            "max_candidate_uncertainty": 1.0,
            "detector_blend": 0.40,
            "preserve_original_floor": False,
            "mutate_geometry": True,
            "mutate_scores": True,
            "collect_diagnostics": diagnostics,
            "architecture": {},
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
    return {"dataset": "scannet", "online_refinement": online}


def _memory(controller: OnlineRefinementController, track_id: int):
    rng = np.random.default_rng(track_id)
    memory = ObjectGeometryMemory(track_id, controller.object_config)
    records = []
    for frame_id, camera_x in ((10, -2.0), (20, 2.0)):
        points = rng.uniform(-0.42, 0.42, size=(45, 3)).astype(np.float32)
        records.append(
            MemoryViewRecord(
                frame_id=frame_id,
                points_world=points,
                quality=0.9,
                confidence=0.85,
                valid_depth_ratio=0.8,
                projection_mask_iou=0.75,
                camera_position=np.asarray(
                    [camera_x, 0.0, -2.0], dtype=np.float32
                ),
            )
        )
    memory._view_candidates = records
    memory._points = np.concatenate(
        [record.points_world for record in records], axis=0
    )
    memory.observation_count = 2
    memory.unique_view_count = 2
    memory.first_frame_id = 10
    memory.last_frame_id = 20
    memory._rebuild_geometry_points()
    return memory


def _attach(
    controller: OnlineRefinementController,
    *,
    stable_id: int,
    score: float,
    box: np.ndarray,
) -> None:
    controller.global_tracks[stable_id] = GlobalEvidence(
        stable_id=stable_id,
        memory=_memory(controller, stable_id),
        stats=EvidenceStats(scores=[score, score]),
        detector_score=score,
        last_box=box.copy(),
    )


def _mock_head():
    torch = pytest.importorskip("torch")

    class MockJointHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = JointLocalHeadConfig()
            self.calls = 0
            self.batch_sizes = []
            self.captured = None

        def forward(
            self,
            points_local,
            point_mask,
            view_features,
            view_mask,
            local_boxes,
            quality_features,
        ):
            self.calls += 1
            self.batch_sizes.append(int(points_local.shape[0]))
            self.captured = tuple(
                value.detach().cpu().numpy().copy()
                for value in (
                    points_local,
                    point_mask,
                    view_features,
                    view_mask,
                    local_boxes,
                    quality_features,
                )
            )
            batch = points_local.shape[0]
            zeros = torch.zeros(
                (batch, 3),
                dtype=points_local.dtype,
                device=points_local.device,
            )
            log_dimensions = torch.full_like(zeros, np.log(0.95))
            improvement = torch.tensor(
                [0.9, 0.1],
                dtype=points_local.dtype,
                device=points_local.device,
            )[:batch]
            original = torch.tensor(
                [0.30, 0.50, 0.30, 0.10],
                dtype=points_local.dtype,
                device=points_local.device,
            )
            candidate = torch.tensor(
                [0.80, 0.90, 0.75, 0.60],
                dtype=points_local.dtype,
                device=points_local.device,
            )
            components = torch.stack((original, candidate))[None].repeat(
                batch, 1, 1
            )
            rankings = torch.tensor(
                [0.25, 0.90],
                dtype=points_local.dtype,
                device=points_local.device,
            )[None].repeat(batch, 1)
            uncertainty = torch.full(
                (batch, 2),
                0.2,
                dtype=points_local.dtype,
                device=points_local.device,
            )
            attention = view_mask.to(points_local.dtype)
            attention = attention / attention.sum(dim=1, keepdim=True)
            return {
                "center_residual": zeros,
                "center_residual_fraction": zeros,
                "log_dimension_residual": log_dimensions,
                "improvement_probability": improvement,
                "quality_components": components,
                "ranking_scores": rankings,
                "quality_log_variance": torch.log(
                    uncertainty.square()
                ),
                "quality_uncertainty": uncertainty,
                "view_attention": attention,
            }

    return MockJointHead()


def test_joint_runtime_batches_once_and_uses_exported_geometry_branch(tmp_path):
    model = _mock_head()
    controller = OnlineRefinementController(
        _joint_config(tmp_path),
        provider=_NoopProvider(),
        joint_local_head=model,
    )
    boxes = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [2.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [4.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    corners = np.stack(
        [aabb_corners(box[:3], box[3:6]) for box in boxes]
    )
    scores = np.asarray([0.60, 0.70, 0.65], dtype=np.float32)
    stable_ids = np.asarray([10, 11, 12], dtype=np.int64)
    _attach(controller, stable_id=10, score=0.60, box=boxes[0])
    _attach(controller, stable_id=11, score=0.70, box=boxes[1])

    result = controller.finalize(
        global_corners=corners,
        global_scores=scores,
        stable_ids=stable_ids,
    )

    assert model.calls == 1
    assert model.batch_sizes == [2]
    assert result.source_indices.tolist() == [0, 1, 2]
    assert result.stable_ids.tolist() == [10, 11, 12]
    assert not np.array_equal(result.corners[0], corners[0])
    np.testing.assert_array_equal(result.corners[1:], corners[1:])
    assert result.scores[0] == pytest.approx(0.4 * 0.60 + 0.6 * 0.90)
    # Learned candidate was rejected, so the original quality branch is used.
    assert result.scores[1] == pytest.approx(0.4 * 0.70 + 0.6 * 0.25)
    # An unobserved box is an exact detector-score identity fallback.
    assert result.scores[2] == pytest.approx(scores[2])
    assert result.refit_reasons == (
        "joint_accepted",
        "joint_improvement",
        "joint_unobserved",
    )
    assert result.summary["joint_batches"] == 1
    assert result.summary["joint_forward_boxes"] == 2
    assert result.summary["joint_candidate_quality_branch"] == 1
    assert result.summary["joint_original_quality_branch"] == 1


def test_joint_diagnostics_are_the_exact_batched_p128_inputs(tmp_path):
    model = _mock_head()
    controller = OnlineRefinementController(
        _joint_config(tmp_path, diagnostics=True),
        provider=_NoopProvider(),
        joint_local_head=model,
    )
    boxes = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [2.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    corners = np.stack(
        [aabb_corners(box[:3], box[3:6]) for box in boxes]
    )
    scores = np.asarray([0.60, 0.70], dtype=np.float32)
    stable_ids = np.asarray([10, 11], dtype=np.int64)
    for index in range(2):
        _attach(
            controller,
            stable_id=int(stable_ids[index]),
            score=float(scores[index]),
            box=boxes[index],
        )

    controller.finalize(
        global_corners=corners,
        global_scores=scores,
        stable_ids=stable_ids,
        scene_id="scene0000_00",
    )
    captured = model.captured
    assert captured is not None
    with np.load(
        tmp_path / "scene0000_00_tracks.npz", allow_pickle=False
    ) as payload:
        assert payload["joint_points_local"].shape == (2, 5, 16, 3)
        assert payload["joint_input_valid"].tolist() == [True, True]
        assert payload["joint_output_valid"].tolist() == [True, True]
        assert payload["joint_quality_branch"].tolist() == [1, 0]
        for name, expected in zip(
            (
                "joint_points_local",
                "joint_point_mask",
                "joint_view_features",
                "joint_view_mask",
                "joint_local_boxes",
                "joint_quality_features",
            ),
            captured,
        ):
            np.testing.assert_array_equal(payload[name], expected)


def test_joint_config_rejects_legacy_mutations_and_missing_top_k(tmp_path):
    config = _joint_config(tmp_path)
    config["online_refinement"]["quality"]["enabled"] = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_online_refinement_config(config)

    config = _joint_config(tmp_path)
    config["online_refinement"]["object_memory"]["top_k_views"] = 4
    with pytest.raises(ValueError, match="top_k_views"):
        resolve_online_refinement_config(config)

