import math

import numpy as np
import pytest

from boxfusion.stream3dv3_track_fusion import (
    AcceptanceConfig,
    TrackEvidenceView,
    accept_frozen_geometry,
    attach_boxer_observation,
    build_and_select_geometry,
    pack_mask,
)


K = np.asarray(
    [[574.0, 0.0, 320.0], [0.0, 577.0, 240.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)


def _view(index: int, camera_x: float, *, hb: bool) -> TrackEvidenceView:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = camera_x
    xs = np.linspace(-0.45, 0.45, 24)
    ys = np.linspace(-0.25, 0.25, 16)
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack((xx.reshape(-1), yy.reshape(-1), np.full(xx.size, 1.60)))
    mask = np.zeros((480, 640), dtype=np.bool_)
    mask[145:335, 150:490] = True
    view = TrackEvidenceView(
        source_id=f"scene0000_00/frame_{index:06d}/raw_000",
        frame_id=index,
        frame_ordinal=index,
        mask_confidence=0.95,
        residual_ratio=0.95,
        valid_ratio=0.98,
        tight_box_xyxy=np.asarray([150, 145, 489, 334], dtype=np.float64),
        mask_packbits=pack_mask(mask),
        points_world=points,
        world_q02=np.asarray([-0.45, -0.25, 1.58]),
        world_q98=np.asarray([0.45, 0.25, 1.62]),
        intrinsics=K,
        camera_to_world=pose,
    )
    if hb:
        view = attach_boxer_observation(
            view,
            center=np.asarray([0.005 * index, 0.0, 2.0]),
            extent=np.asarray([1.0, 0.60, 0.80]),
            rotation=np.eye(3),
            confidence=0.95,
        )
    return view


def _permissive_acceptance() -> AcceptanceConfig:
    return AcceptanceConfig(
        min_total_views=5,
        min_f4_views=2,
        min_view_ray_angle_deg=0.0,
        min_camera_baseline_m=0.0,
        max_center_rms_m=1.0,
        max_log_size_mad=1.0,
        max_yaw_mad_deg=180.0,
        max_normalized_center_std=2.0,
        max_center_std_m=1.0,
        max_log_size_std=1.0,
        min_mask_box_iou=0.01,
        min_mask_containment=0.01,
        min_point_inside=0.20,
        min_depth_support=0.20,
        max_free_space=0.90,
        min_quality=0.05,
        min_hypothesis_margin=0.0,
    )


def test_two_f4_views_then_independent_selection_and_acceptance():
    fitting = [_view(0, -0.30, hb=False), _view(1, 0.00, hb=True), _view(2, 0.30, hb=True)]
    selection = _view(3, -0.20, hb=False)
    acceptance = _view(4, 0.20, hb=False)

    frozen = build_and_select_geometry(fitting, selection)
    assert frozen.f4_view_count == 2
    assert frozen.geometry.decision_frame_ordinal == 3
    assert selection.source_id not in frozen.fit_source_ids
    assert len(frozen.geometry.hypotheses) == 3

    result = accept_frozen_geometry(
        frozen,
        acceptance,
        total_distinct_views=5,
        config=_permissive_acceptance(),
    )
    assert result.absolute_pass, result.reasons
    assert result.geometry.decision_frame_ordinal == 4
    assert acceptance.source_id not in result.fit_source_ids
    assert result.selection_receipt.source_id == selection.source_id
    assert result.acceptance_receipt.source_id == acceptance.source_id
    assert np.isfinite(result.covariance_7d).all()
    assert np.all(np.linalg.eigvalsh(result.covariance_7d) > 0.0)


def test_acceptance_view_must_be_strictly_later():
    fitting = [_view(0, -0.30, hb=False), _view(1, 0.00, hb=True), _view(2, 0.30, hb=True)]
    selection = _view(3, -0.20, hb=False)
    frozen = build_and_select_geometry(fitting, selection)
    with pytest.raises(ValueError, match="later"):
        accept_frozen_geometry(
            frozen,
            selection,
            total_distinct_views=5,
            config=_permissive_acceptance(),
        )


def test_default_gate_rejects_a_geometrically_inconsistent_track():
    fitting = [_view(0, -0.30, hb=False), _view(1, 0.00, hb=True), _view(2, 0.30, hb=True)]
    selection = _view(3, -0.20, hb=False)
    frozen = build_and_select_geometry(fitting, selection)
    bad = _view(4, 0.20, hb=False)
    shifted_points = np.asarray(bad.points_world) + np.asarray([4.0, 0.0, 0.0])
    bad = TrackEvidenceView(
        source_id=bad.source_id,
        frame_id=bad.frame_id,
        frame_ordinal=bad.frame_ordinal,
        mask_confidence=bad.mask_confidence,
        residual_ratio=bad.residual_ratio,
        valid_ratio=bad.valid_ratio,
        tight_box_xyxy=bad.tight_box_xyxy,
        mask_packbits=bad.mask_packbits,
        points_world=shifted_points,
        world_q02=bad.world_q02 + np.asarray([4.0, 0.0, 0.0]),
        world_q98=bad.world_q98 + np.asarray([4.0, 0.0, 0.0]),
        intrinsics=bad.intrinsics,
        camera_to_world=bad.camera_to_world,
    )
    result = accept_frozen_geometry(frozen, bad, total_distinct_views=5)
    assert not result.absolute_pass
    assert "acceptance_inside" in result.reasons

