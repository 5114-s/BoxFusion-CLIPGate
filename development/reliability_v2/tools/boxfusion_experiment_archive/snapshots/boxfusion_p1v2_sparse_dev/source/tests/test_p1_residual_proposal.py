"""CPU-only contracts for the P1 class-agnostic residual observer.

P1 is intentionally diagnostics-only.  These tests freeze the properties
which distinguish it from a second semantic detector: residual RGB-D points
are voxelised deterministically, the head predicts one objectness value and
six axis-aligned box residuals, and candidates receive diagnostic-only IDs.
No ScanNet class label or ground truth is accepted by the online API.
"""

from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from boxfusion.residual_proposal import (  # noqa: E402
    P1_FEATURE_DIM,
    P1ResidualProposalObserver,
    ResidualProposalConfig,
    ResidualVoxelProposalHead,
    pairwise_aabb_iou,
    resolve_residual_proposal_config,
    score_ordered_match,
)


def _aabb_corners(lower, upper):
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    return np.asarray(
        [
            [x, y, z]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ],
        dtype=np.float64,
    )


def _observer_config(**updates):
    values = {
        "enabled": True,
        "observer_only": True,
        "mutate": False,
        "mode": "collect",
        "checkpoint": None,
        "collect_diagnostics": True,
        "collect_voxel_inputs": True,
        "voxel_size": 0.25,
        "min_voxel_points": 1,
        "pre_nms_topk": 64,
        "max_candidates_per_step": 16,
        "score_threshold": 0.0,
        "nms_iou": 0.25,
    }
    values.update(updates)
    return resolve_residual_proposal_config(values)


def _make_observer(config=None, head=None):
    return P1ResidualProposalObserver(
        config or _observer_config(),
        head=head,
        device="cpu",
    )


def _batch_signature(batch):
    return (
        np.asarray(batch.coordinates).copy(),
        np.asarray(batch.centers_world).copy(),
        np.asarray(batch.features).copy(),
        np.asarray(batch.point_counts).copy(),
    )


def test_configuration_is_strict_and_forces_observer_only():
    defaults = resolve_residual_proposal_config()
    assert isinstance(defaults, ResidualProposalConfig)
    assert defaults.enabled is False
    assert defaults.observer_only is True
    assert defaults.mutate is False

    with pytest.raises((TypeError, ValueError), match="Unknown|unknown"):
        resolve_residual_proposal_config({"voxel_szie": 0.05})
    with pytest.raises((TypeError, ValueError)):
        resolve_residual_proposal_config({"enabled": 1})
    with pytest.raises((TypeError, ValueError)):
        resolve_residual_proposal_config({"voxel_size": 0.0})
    with pytest.raises((TypeError, ValueError), match="mutat|observer"):
        replace(defaults, mutate=True).validated()


def test_resolver_allows_collection_without_checkpoint_but_inference_fails_closed(
    tmp_path,
):
    collection = _observer_config(mode="collect", checkpoint=None)
    assert collection.checkpoint is None
    _make_observer(collection)

    inference = _observer_config(mode="infer", checkpoint=None)
    with pytest.raises((FileNotFoundError, ValueError), match="checkpoint"):
        _make_observer(inference)

    missing = _observer_config(
        mode="infer", checkpoint=str(tmp_path / "absent.pt")
    )
    with pytest.raises((FileNotFoundError, ValueError), match="checkpoint"):
        _make_observer(missing)


def test_voxelisation_is_permutation_invariant_and_translation_equivariant():
    # The first two points are explained by the frozen B6 box.  The remaining
    # points occupy three deterministic residual voxels.
    explained = np.asarray(
        [[0.25, 0.25, 1.25], [0.75, 0.75, 1.75]], dtype=np.float64
    )
    residual = np.asarray(
        [
            [1.10, 0.10, 2.10],
            [1.18, 0.12, 2.12],
            [1.62, 0.10, 2.10],
            [1.68, 0.14, 2.14],
            [2.12, 0.12, 2.12],
        ],
        dtype=np.float64,
    )
    points = np.concatenate((explained, residual), axis=0)
    colors = np.asarray(
        [
            [10, 20, 30],
            [20, 30, 40],
            [50, 60, 70],
            [60, 70, 80],
            [80, 90, 100],
            [90, 100, 110],
            [120, 130, 140],
        ],
        dtype=np.float32,
    )
    global_corners = _aabb_corners((0, 0, 1), (1, 1, 2))[None]
    camera = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    observer = _make_observer()

    first = observer.build_voxel_batch(
        points,
        colors=colors,
        camera_position=camera,
        global_corners=global_corners,
    )
    permutation = np.asarray([6, 2, 0, 5, 4, 1, 3], dtype=np.int64)
    permuted = observer.build_voxel_batch(
        points[permutation],
        colors=colors[permutation],
        camera_position=camera,
        global_corners=global_corners,
    )
    first_signature = _batch_signature(first)
    permuted_signature = _batch_signature(permuted)
    for left, right in zip(first_signature, permuted_signature):
        np.testing.assert_allclose(left, right, atol=0.0, rtol=0.0)

    assert first.input_point_count == len(points)
    assert first.explained_point_count == len(explained)
    assert first.residual_point_count == len(residual)
    assert first.coordinates.ndim == 2
    assert first.coordinates.shape[1] == 3
    assert first.sparse_coordinates.shape[1] == 4
    assert np.issubdtype(first.coordinates.dtype, np.integer)
    assert np.isfinite(first.features).all()

    shift = np.asarray([4.0, -3.0, 2.0], dtype=np.float64)
    shifted = observer.build_voxel_batch(
        points + shift,
        colors=colors,
        camera_position=camera + shift,
        global_corners=global_corners + shift,
    )
    np.testing.assert_array_equal(
        shifted.coordinates,
        first.coordinates
        + np.rint(shift / 0.25).astype(np.int32),
    )
    np.testing.assert_allclose(
        shifted.centers_world,
        first.centers_world + shift,
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        shifted.features, first.features, atol=1e-6, rtol=0.0
    )
    np.testing.assert_array_equal(first.point_counts, shifted.point_counts)

    for array in (
        first.coordinates,
        first.centers_world,
        first.features,
        first.point_counts,
    ):
        assert array.flags.writeable is False


def test_head_has_exactly_one_objectness_and_six_regression_outputs():
    head = ResidualVoxelProposalHead(
        input_dim=P1_FEATURE_DIM, hidden_dim=16
    )
    features = torch.randn(7, P1_FEATURE_DIM)
    objectness, regression = head(features)

    assert tuple(objectness.shape) == (7, 1)
    assert tuple(regression.shape) == (7, 6)
    assert torch.isfinite(objectness).all()
    assert torch.isfinite(regression).all()
    # No class-count-dependent branch may silently turn P1 into an 18-class
    # ScanNet detector.
    assert not hasattr(head, "classification_head")
    assert not hasattr(head, "num_classes")

    empty_objectness, empty_regression = head(
        torch.empty(0, P1_FEATURE_DIM)
    )
    assert tuple(empty_objectness.shape) == (0, 1)
    assert tuple(empty_regression.shape) == (0, 6)


def test_decode_and_nms_use_stable_score_then_anchor_order():
    observer = _make_observer()
    points = np.asarray(
        [
            [0.10, 0.10, 2.00],
            [0.15, 0.10, 2.00],
            [0.60, 0.10, 2.00],
        ],
        dtype=np.float64,
    )
    batch = observer.build_voxel_batch(
        points,
        camera_position=np.zeros(3),
        global_corners=np.empty((0, 8, 3), dtype=np.float64),
    )
    count = len(batch.coordinates)
    logits = torch.zeros((count, 1), dtype=torch.float32)
    regression = torch.zeros((count, 6), dtype=torch.float32)

    first = observer.decode(
        batch,
        logits,
        regression,
        scene_id="scene0000_00",
        frame_index=5,
        provider_step=1,
    )
    second = observer.decode(
        batch,
        logits,
        regression,
        scene_id="scene0000_00",
        frame_index=5,
        provider_step=1,
    )
    assert [proposal.candidate_id for proposal in first] == [
        proposal.candidate_id for proposal in second
    ]
    np.testing.assert_array_equal(
        np.asarray([item.box for item in first]),
        np.asarray([item.box for item in second]),
    )
    assert [item.score for item in first] == [
        item.score for item in second
    ]
    assert len({item.candidate_id for item in first}) == len(first)
    assert all(np.all(np.asarray(item.box)[3:] > 0.0) for item in first)


def test_pairwise_iou_and_score_ordered_match_penalise_duplicates():
    gt = np.asarray(
        [
            [-0.5, -0.5, -0.5, 0.5, 0.5, 0.5],
            [2.5, -0.5, -0.5, 3.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    candidates = np.asarray(
        [
            [-0.5, -0.5, -0.5, 0.5, 0.5, 0.5],
            [-0.5, -0.5, -0.5, 0.5, 0.5, 0.5],
            [2.5, -0.5, -0.5, 3.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    iou = pairwise_aabb_iou(candidates, gt)
    assert iou.shape == (3, 2)
    np.testing.assert_allclose(iou[[0, 2], [0, 1]], 1.0)
    match = score_ordered_match(
        iou,
        np.asarray([0.9, 0.8, 0.7], dtype=np.float64),
        0.5,
    )
    np.testing.assert_array_equal(match.prediction_to_gt, [0, -1, 1])
    np.testing.assert_allclose(match.matched_iou, [1.0, 0.0, 1.0])
    assert match.true_positive_count == 2


class _DeterministicStubHead(torch.nn.Module):
    def forward(self, features):
        count = int(features.shape[0])
        logits = torch.arange(
            count, dtype=features.dtype, device=features.device
        ).reshape(count, 1)
        regression = torch.zeros(
            (count, 6), dtype=features.dtype, device=features.device
        )
        return logits, regression


def _observe(observer, *, scene_id, frame_index, provider_step):
    height, width = 8, 8
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = np.arange(width, dtype=np.uint8)[None]
    depth = np.full((height, width), 2.0, dtype=np.float32)
    intrinsics = np.asarray(
        [[80.0, 0.0, 3.5], [0.0, 80.0, 3.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return observer.observe(
        image=image,
        depth=depth,
        intrinsics=intrinsics,
        camera_to_world=np.eye(4, dtype=np.float64),
        global_corners=np.empty((0, 8, 3), dtype=np.float64),
        global_stable_ids=np.empty((0,), dtype=np.int64),
        frame_index=frame_index,
        provider_step=provider_step,
        scene_id=scene_id,
    )


def test_stub_observer_has_diagnostic_ids_and_scene_reset():
    config = _observer_config(
        mode="infer",
        checkpoint=None,
        voxel_size=0.02,
        collect_voxel_inputs=False,
    )
    observer = _make_observer(config, _DeterministicStubHead())

    first = _observe(
        observer, scene_id="scene0000_00", frame_index=5, provider_step=1
    )
    repeated = _observe(
        observer, scene_id="scene0000_00", frame_index=10, provider_step=2
    )
    first_ids = [proposal.candidate_id for proposal in first.proposals]
    repeated_ids = [
        proposal.candidate_id for proposal in repeated.proposals
    ]
    assert first.observer_only is True
    assert first.mutation_enabled is False
    assert first.applied_count == 0
    assert len(first_ids) == len(set(first_ids))
    assert not set(first_ids).intersection(repeated_ids)
    assert all(
        isinstance(identifier, str)
        and identifier.startswith("scene0000_00:")
        for identifier in first_ids
    )

    observer.reset("scene0001_00")
    after_reset = _observe(
        observer, scene_id="scene0001_00", frame_index=5, provider_step=1
    )
    after_ids = [
        proposal.candidate_id for proposal in after_reset.proposals
    ]
    assert all(identifier.startswith("scene0001_00:") for identifier in after_ids)
    assert after_ids != first_ids


def test_online_observe_signature_has_no_ground_truth_or_class_target():
    import inspect

    parameters = inspect.signature(
        P1ResidualProposalObserver.observe
    ).parameters
    forbidden = {
        "ground_truth",
        "gt_boxes",
        "gt_labels",
        "class_labels",
        "semantic_labels",
    }
    assert forbidden.isdisjoint(parameters)
    assert {
        "image",
        "depth",
        "intrinsics",
        "camera_to_world",
        "global_corners",
        "global_stable_ids",
        "frame_index",
        "provider_step",
        "scene_id",
    }.issubset(parameters)
