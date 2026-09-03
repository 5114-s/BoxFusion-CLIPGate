from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import boxfusion.p1_multiview_geometry as p1g_module
from boxfusion.p1_multiview_geometry import (
    P1G_DIAGNOSTIC_SCHEMA,
    P1G_PROFILE,
    P1MultiViewGeometryObserver,
    resolve_p1_multiview_geometry_config,
)
from boxfusion.residual_proposal import (
    ResidualProposal,
    center_size_to_corners,
)


def _anchor(
    candidate_id: str = "scene0000_00:000000:0:0:0",
    *,
    box: np.ndarray | None = None,
    score: float = 0.8,
) -> ResidualProposal:
    if box is None:
        box = np.asarray(
            [0.0, 0.0, 0.0, 1.0, 0.8, 0.7], dtype=np.float32
        )
    return ResidualProposal(
        candidate_id=candidate_id,
        frame_index=0,
        provider_step=0,
        box=box,
        corners=center_size_to_corners(box)[0],
        objectness=score,
        residual_point_count=32,
    )


def _grid() -> np.ndarray:
    return np.stack(
        np.meshgrid(
            np.linspace(-0.30, 0.30, 13),
            np.linspace(-0.20, 0.20, 11),
            np.linspace(-0.17, 0.17, 9),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3).astype(np.float32)


def _observation(
    index: int,
    *,
    anchor: ResidualProposal | None = None,
    points: np.ndarray | None = None,
    camera: np.ndarray | None = None,
    score: float = 0.9,
) -> SimpleNamespace:
    if anchor is None:
        anchor = _anchor()
    if points is None:
        points = _grid()
        points = np.array(points, copy=True)
        points[:, (index + 1) % 3] += (index - 1.5) * 0.00025
    if camera is None:
        cameras = (
            (-2.0, -2.0, -2.0),
            (2.0, 2.0, 2.0),
            (-2.0, 2.0, 2.0),
            (2.0, -2.0, -2.0),
            (0.0, 2.0, -2.0),
            (0.0, -2.0, 2.0),
        )
        camera = np.asarray(cameras[index], dtype=np.float32)
    proposal = _anchor(
        candidate_id=f"scene0000_00:{index:06d}:0:0:0",
        box=np.asarray(anchor.box),
        score=score,
    )
    return SimpleNamespace(
        frame_index=10 + index,
        provider_step=index,
        proposals=(proposal,),
        geometry_points_world=np.asarray(points, dtype=np.float32),
        camera_position=np.asarray(camera, dtype=np.float32),
    )


def _config(**updates: object) -> dict[str, object]:
    values: dict[str, object] = {
        "enabled": True,
        "observer_only": True,
        "mutate": False,
        "collect_diagnostics": True,
        "association_iou": 0.10,
        "crop_scale": 1.35,
        "top_k_views": 5,
        "max_points_per_view": 512,
        "max_candidates": 256,
        "proposal": {
            "max_views": 5,
            "max_points_per_view": 512,
            "crop_scale": 1.35,
            "min_points_per_view": 20,
            "min_total_points": 80,
            "min_component_points": 48,
            "face_min_points_per_view": 8,
        },
    }
    values.update(updates)
    return values


def _observer(**updates: object) -> P1MultiViewGeometryObserver:
    return P1MultiViewGeometryObserver(
        _config(**updates),
        parent_checkpoint_sha256="a" * 64,
    )


def test_config_and_public_api_are_strictly_observer_only():
    config = resolve_p1_multiview_geometry_config(_config())
    assert config.enabled is True
    assert config.observer_only is True
    assert config.mutate is False
    assert config.association_iou == pytest.approx(0.10)
    assert config.proposal["max_views"] == config.top_k_views
    assert config.proposal["crop_scale"] == config.crop_scale

    with pytest.raises(ValueError, match="observer_only"):
        resolve_p1_multiview_geometry_config({"observer_only": False})
    with pytest.raises(ValueError, match="cannot mutate"):
        resolve_p1_multiview_geometry_config({"mutate": True})
    with pytest.raises(ValueError, match="Unknown"):
        resolve_p1_multiview_geometry_config({"semantic_label": True})
    with pytest.raises(ValueError, match="must match"):
        resolve_p1_multiview_geometry_config(
            {
                "top_k_views": 5,
                "proposal": {"max_views": 4},
            }
        )

    signature = inspect.signature(
        P1MultiViewGeometryObserver.observe_scene
    )
    lowered = " ".join(signature.parameters).lower()
    assert "gt" not in lowered
    assert "label" not in lowered
    assert "semantic" not in lowered


def test_actual_multiview_msr_returns_one_to_one_detached_candidate():
    anchor = _anchor()
    observer = _observer()
    rows = observer.observe_scene(
        scene_id="scene0000_00",
        anchors=(anchor,),
        observations=tuple(_observation(index) for index in range(4)),
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.parent_candidate_id == anchor.candidate_id
    assert row.refined_candidate_id == f"{anchor.candidate_id}:p1g"
    assert row.reason == "candidate"
    assert row.is_candidate is True
    assert row.applied is False
    assert row.matched_view_count == 4
    assert row.selected_view_count == 4
    assert len(row.selected_frame_ids) == 4
    assert row.cropped_point_count > 0
    assert row.feature_vector.shape == (48,)
    assert np.any(row.face_supported)
    assert not np.array_equal(row.parent_corners, row.refined_corners)
    assert row.parent_box.flags.writeable is False
    assert row.refined_corners.flags.writeable is False

    # Detached output: changing an unrelated source array cannot alter rows.
    original = np.array(row.parent_box, copy=True)
    np.asarray(anchor.box)[:] = 7.0
    np.testing.assert_array_equal(row.parent_box, original)


def test_association_chooses_one_best_proposal_and_topk_is_view_diverse(
    monkeypatch: pytest.MonkeyPatch,
):
    anchor = _anchor()
    observations = []
    cameras = (
        (-2.0, 0.0, 0.0),
        (-1.9, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
        (0.0, -2.0, 0.0),
        (0.0, 0.0, 2.0),
    )
    for index, camera in enumerate(cameras):
        observation = _observation(
            index,
            camera=np.asarray(camera, dtype=np.float32),
            score=0.95 - index * 0.03,
        )
        # A below-threshold distractor has a higher score.  The geometric
        # match, not score alone, must pick the anchor proposal.
        distractor_box = np.asarray(
            [2.5, 0.0, 0.0, 1.0, 0.8, 0.7], dtype=np.float32
        )
        distractor = _anchor(
            candidate_id=f"distractor:{index}",
            box=distractor_box,
            score=0.999,
        )
        observation.proposals = (
            distractor,
            *observation.proposals,
            # A duplicate exact proposal tests deterministic tie breaking.
            _anchor(
                candidate_id=f"z-duplicate:{index}",
                box=np.asarray(anchor.box),
                score=0.1,
            ),
        )
        observations.append(observation)

    captured: dict[str, object] = {}
    real = p1g_module.propose_local_occupancy_msr

    def capture(corners, views, config):
        captured["views"] = tuple(views)
        return real(corners, views, config)

    monkeypatch.setattr(
        p1g_module, "propose_local_occupancy_msr", capture
    )
    observer = _observer(top_k_views=3, proposal={
        **_config()["proposal"],
        "max_views": 3,
    })
    row = observer.observe_scene(
        scene_id="scene0000_00",
        anchors=(anchor,),
        observations=tuple(reversed(observations)),
    )[0]

    views = captured["views"]
    assert isinstance(views, tuple)
    assert row.matched_view_count == 6
    assert row.selected_view_count == 3
    assert len(views) == 3
    # All selected records came from the exact anchor match.
    assert all(view.projection_mask_iou == pytest.approx(1.0) for view in views)
    assert 10 in row.selected_frame_ids
    # Diversity must avoid retaining only the two nearly identical -X views.
    selected_cameras = [tuple(view.camera_position.tolist()) for view in views]
    assert not (
        (-2.0, 0.0, 0.0) in selected_cameras
        and (-1.9, 0.0, 0.0) in selected_cameras
        and len(selected_cameras) == 3
    )


def test_runtime_errors_and_malformed_observations_fail_open_to_identity(
    monkeypatch: pytest.MonkeyPatch,
):
    anchor = _anchor()
    valid = _observation(0)
    malformed = SimpleNamespace(
        frame_index=1,
        provider_step=1,
        proposals=(),
        geometry_points_world=np.asarray([[np.nan, 0.0, 0.0]]),
        camera_position=np.zeros(3, dtype=np.float32),
    )

    def explode(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("synthetic refiner failure")

    monkeypatch.setattr(
        p1g_module, "propose_local_occupancy_msr", explode
    )
    observer = _observer()
    row = observer.observe_scene(
        scene_id="scene0000_00",
        anchors=(anchor,),
        observations=(malformed, valid),
    )[0]

    assert row.reason == "identity_exception:RuntimeError"
    assert row.is_candidate is False
    assert row.applied is False
    np.testing.assert_array_equal(row.refined_box, row.parent_box)
    np.testing.assert_array_equal(row.refined_corners, row.parent_corners)
    assert observer.failure_count == 2


def test_diagnostic_payload_is_row_aligned_safe_and_pickle_free():
    anchors = (
        _anchor("scene0000_00:000000:0:0:0"),
        _anchor(
            "scene0000_00:000000:1:0:0",
            box=np.asarray(
                [3.0, 0.0, 0.0, 1.0, 0.8, 0.7], dtype=np.float32
            ),
        ),
    )
    observer = _observer()
    observer.observe_scene(
        scene_id="scene0000_00",
        anchors=anchors,
        observations=tuple(_observation(index) for index in range(4)),
    )
    payload = observer.diagnostic_payload()

    assert payload["p1g_schema"].item() == P1G_DIAGNOSTIC_SCHEMA
    assert payload["p1g_stage"].item() == "P1G"
    assert payload["p1g_profile"].item() == P1G_PROFILE
    assert payload["p1g_parent_stage"].item() == "P1S"
    assert bool(payload["p1g_observer_only"]) is True
    assert bool(payload["p1g_uses_ground_truth"]) is False
    assert bool(payload["p1g_reads_semantic_labels"]) is False
    assert bool(payload["p1g_mutation_enabled"]) is False
    assert int(payload["p1g_applied_count"]) == 0
    assert int(payload["p1g_regression_dim"]) == 6
    assert payload["p1g_parent_candidate_ids"].shape == (2,)
    assert payload["p1g_refined_candidate_ids"].shape == (2,)
    assert payload["p1g_parent_boxes"].shape == (2, 6)
    assert payload["p1g_refined_corners"].shape == (2, 8, 3)
    assert payload["p1g_selected_frame_ids"].shape == (2, 5)
    assert payload["p1g_face_residuals"].shape == (2, 3, 2)
    assert payload["p1g_feature_vectors"].shape == (2, 48)
    assert payload["p1g_step_total_seconds"].shape == (2,)
    assert not np.any(payload["p1g_candidate_applied"])
    assert not bool(payload["p1g_is_candidate"][1])
    np.testing.assert_array_equal(
        payload["p1g_refined_boxes"][1],
        payload["p1g_parent_boxes"][1],
    )
    np.testing.assert_array_equal(
        payload["p1g_refined_corners"][1],
        payload["p1g_parent_corners"][1],
    )
    assert all(not value.dtype.hasobject for value in payload.values())


def test_observation_and_point_permutations_are_deterministic():
    anchor = _anchor()
    observations = tuple(_observation(index) for index in range(4))
    first = _observer().observe_scene(
        scene_id="scene0000_00",
        anchors=(anchor,),
        observations=observations,
    )[0]
    rng = np.random.default_rng(91)
    shuffled = []
    for row in reversed(observations):
        shuffled.append(
            SimpleNamespace(
                frame_index=row.frame_index,
                provider_step=row.provider_step,
                proposals=tuple(reversed(row.proposals)),
                geometry_points_world=row.geometry_points_world[
                    rng.permutation(len(row.geometry_points_world))
                ],
                camera_position=np.array(row.camera_position, copy=True),
            )
        )
    second = _observer().observe_scene(
        scene_id="scene0000_00",
        anchors=(anchor,),
        observations=tuple(shuffled),
    )[0]

    assert first.reason == second.reason
    assert first.selected_frame_ids == second.selected_frame_ids
    np.testing.assert_array_equal(
        first.refined_corners, second.refined_corners
    )
    np.testing.assert_array_equal(
        first.face_residuals, second.face_residuals
    )
    np.testing.assert_array_equal(
        first.feature_vector, second.feature_vector
    )


def test_candidate_budget_is_deterministic_and_disabled_stage_is_empty():
    anchors = tuple(
        _anchor(f"scene0000_00:000000:{index}:0:0")
        for index in range(4)
    )
    observer = _observer(max_candidates=2)
    rows = observer.observe_scene(
        scene_id="scene0000_00",
        anchors=anchors,
        observations=(),
    )
    assert [row.parent_candidate_id for row in rows] == [
        anchors[0].candidate_id,
        anchors[1].candidate_id,
    ]
    assert all(row.reason == "identity_insufficient_views" for row in rows)

    disabled = P1MultiViewGeometryObserver(
        {},
        parent_checkpoint_sha256="injected",
    )
    assert (
        disabled.observe_scene(
            scene_id="scene0000_00",
            anchors=anchors,
            observations=(),
        )
        == ()
    )
    assert disabled.diagnostic_payload()["p1g_parent_boxes"].shape == (0, 6)
