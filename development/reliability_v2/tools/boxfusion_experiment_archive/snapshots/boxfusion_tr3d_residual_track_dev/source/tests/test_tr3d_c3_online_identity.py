from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from boxfusion.supplemental_proposals import SupplementalProposal
from boxfusion.tr3d_c2_maskrgbd_cache import (
    TR3DC2MaskRGBDCache,
    canonical_json,
    sha256_file,
    sha256_bytes,
    write_sidecar,
)
from boxfusion.tr3d_c2_maskrgbd_observer import (
    C2Frame,
    C2MaskRGBDConfig,
    observe_scene,
)
from boxfusion.tr3d_c3_online_identity import (
    C3OnlineIdentityConfig,
    C3OnlineIdentityObserver,
    prediction_state_sha256,
)
from boxfusion.tr3d_residual_cache import (
    make_tr3d_residual_cache_from_aligned,
    write_tr3d_residual_cache,
)


def _frame(frame_id: int, *, proposals: bool = True) -> C2Frame:
    shape = (100, 100)
    depth = np.full(shape, 2.0, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.bool_)
    mask[30:70, 30:70] = True
    rows = (
        SupplementalProposal(
            bbox=np.asarray([30, 30, 70, 70], dtype=np.float32),
            score=0.9,
            mask=mask,
            label="diagnostic-only",
        ),
    ) if proposals else ()
    intrinsic = np.asarray(
        [
            [100.0, 0.0, 50.0, 0.0],
            [0.0, 100.0, 50.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return C2Frame(
        frame_id=frame_id,
        depth_meters=depth,
        intrinsics=intrinsic,
        depth_camera_to_world=np.eye(4, dtype=np.float64),
        proposals=rows,
        cache_sha256="f" * 64,
    )


def _observer(tmp_path: Path) -> tuple[C3OnlineIdentityObserver, np.ndarray]:
    scene_id = "scene0000_00"
    prefix_id = "p100"
    parent_root = tmp_path / "parent"
    c2_root = tmp_path / "c2"
    diagnostics_root = tmp_path / "diagnostics"
    parent = make_tr3d_residual_cache_from_aligned(
        scene_id=scene_id,
        boxes_aligned=np.asarray(
            [[0.0, 0.0, 2.0, 0.8, 0.8, 0.8, 0.0]],
            dtype=np.float32,
        ),
        scores_3d=np.asarray([0.8], dtype=np.float32),
        unaligned_to_aligned=np.eye(4, dtype=np.float64),
        checkpoint_sha256="a" * 64,
        config_sha256="b" * 64,
        source_scene_sha256="c" * 64,
        prefix_id=prefix_id,
        proposal_ids=np.asarray([7], dtype=np.int64),
    )
    parent_path = parent_root / scene_id / f"{prefix_id}.npz"
    write_tr3d_residual_cache(parent_path, parent)
    parent_sha = sha256_file(parent_path)
    config = C2MaskRGBDConfig()
    config_json = canonical_json(config.as_dict())
    frozen_observation = observe_scene(
        parent.boxes_world,
        (_frame(0), _frame(125)),
        config,
    )
    c2 = TR3DC2MaskRGBDCache(
        scene_id=scene_id,
        prefix_id=prefix_id,
        c1_sidecar_sha256="d" * 64,
        parent_cache_sha256=parent_sha,
        anchor_prediction_sha256="e" * 64,
        teacher_manifest_set_sha256="1" * 64,
        runtime_manifest_set_sha256="2" * 64,
        scene_frame_input_sha256="3" * 64,
        config_sha256=sha256_bytes(config_json.encode("utf-8")),
        code_sha256="4" * 64,
        config_json=config_json,
        source_c1_rows=np.asarray([0], dtype=np.int64),
        source_ranks=np.asarray([1], dtype=np.int32),
        proposal_ids=np.asarray([7], dtype=np.int64),
        parent_rows=np.asarray([0], dtype=np.int64),
        c1_track_scores=np.asarray([0.75], dtype=np.float32),
        frame_cache_sha256=np.asarray(["5" * 64, "6" * 64]),
        observation=frozen_observation,
        runtime_s=0.1,
    )
    write_sidecar(c2_root / scene_id / f"{prefix_id}.c2-maskrgbd.npz", c2)
    observer = C3OnlineIdentityObserver(
        C3OnlineIdentityConfig(
            enabled=True,
            c2_cache_root=str(c2_root),
            parent_cache_root=str(parent_root),
            diagnostics_root=str(diagnostics_root),
        )
    )
    return observer, parent.corners_world


def test_online_gate_matches_frozen_and_never_mutates_prediction(tmp_path):
    observer, corners = _observer(tmp_path)
    scores = np.asarray([0.6], dtype=np.float32)
    corners_before = corners.copy()
    scores_before = scores.copy()
    for frame_id in (0, 125):
        frame = _frame(frame_id)
        observer.observe_keyframe(
            scene_id="scene0000_00",
            frame_id=frame_id,
            proposals=frame.proposals,
            depth=frame.depth_meters,
            intrinsics=frame.intrinsics,
            camera_to_world=frame.depth_camera_to_world,
        )
    summary = observer.finalize(
        scene_id="scene0000_00",
        prediction_corners=corners,
        prediction_scores=scores,
    )
    np.testing.assert_array_equal(corners, corners_before)
    np.testing.assert_array_equal(scores, scores_before)
    assert summary["applied_count"] == 0
    assert summary["prediction_identity"] is True
    assert summary["selection_exact_match"] is True
    assert summary["frozen_selected_count"] == 1
    assert summary["online_selected_count"] == 1
    assert summary["intersection_count"] == 1
    assert summary["frame_ids"] == [0, 125]
    assert summary["rejection_telemetry_schema"].endswith(".v1")
    candidate = summary["candidates"][0]
    assert candidate["strong_predicate_fail_counts"] == {
        "mask_score": 0,
        "mask_containment": 0,
        "box_coverage": 0,
        "valid_depth_pixels": 0,
        "inside_expanded": 0,
        "component_points": 0,
        "component_inside": 0,
    }
    assert candidate["strong_predicate_pass_counts"]["mask_score"] == 2
    assert candidate["matched_metric_max"]["best_mask_score"] == pytest.approx(0.9)
    diagnostic = (
        tmp_path
        / "diagnostics"
        / "scene0000_00_c3_online_identity.json"
    )
    assert diagnostic.is_file()
    assert json.loads(diagnostic.read_text())["prediction_identity"] is True


def test_online_gate_fails_closed_without_runtime_masks(tmp_path):
    observer, corners = _observer(tmp_path)
    for frame_id in (0, 125):
        frame = _frame(frame_id, proposals=False)
        observer.observe_keyframe(
            scene_id="scene0000_00",
            frame_id=frame_id,
            proposals=(),
            depth=frame.depth_meters,
            intrinsics=frame.intrinsics,
            camera_to_world=frame.depth_camera_to_world,
        )
    summary = observer.finalize(
        scene_id="scene0000_00",
        prediction_corners=corners,
        prediction_scores=np.asarray([0.6], dtype=np.float32),
    )
    assert summary["frozen_selected_count"] == 1
    assert summary["online_selected_count"] == 0
    assert summary["false_negative_count"] == 1
    assert summary["selection_recall_vs_frozen"] == 0.0


def test_prediction_hash_is_dtype_canonical_and_shape_sensitive():
    corners = np.zeros((1, 8, 3), dtype=np.float64)
    scores = np.asarray([0.5], dtype=np.float64)
    assert prediction_state_sha256(corners, scores) == prediction_state_sha256(
        corners.astype(np.float32), scores.astype(np.float32)
    )
