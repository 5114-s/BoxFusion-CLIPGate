"""Contracts for the dependency-free native sparse P1-v2 head."""

from __future__ import annotations

import inspect

import pytest
import torch

from boxfusion.p1_spatial_residual import (
    NativeSparseResidualProposalHead,
    P1_SPATIAL_ARCHITECTURE,
    P1_SPATIAL_FEATURE_DIM,
    build_axis6_neighbor_indices,
)


def _model() -> NativeSparseResidualProposalHead:
    torch.manual_seed(17)
    return NativeSparseResidualProposalHead(
        hidden_dim=16, dilations=(1, 2)
    ).eval()


def test_axis6_topology_has_exact_direction_and_batch_contract():
    # Rows 0..3 are one batch; row 4 has the same xyz as row 1 but belongs to
    # another sparse sample and must never become its neighbour.
    coordinates = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, -1],
            [1, 1, 0, 0],
        ],
        dtype=torch.int32,
    )
    original = coordinates.clone()
    topology = build_axis6_neighbor_indices(
        coordinates, dilations=(1, 2)
    )
    assert topology.shape == (5, 2, 6)
    assert topology.dtype == torch.int64
    assert topology.requires_grad is False
    torch.testing.assert_close(coordinates, original, rtol=0.0, atol=0.0)

    # Direction order is -x,+x,-y,+y,-z,+z.
    assert topology[0, 0].tolist() == [-1, 1, -1, -1, 3, -1]
    assert topology[0, 1].tolist() == [-1, -1, -1, 2, -1, -1]
    assert not bool(torch.any(topology[0] == 4))
    assert not bool(torch.any(topology[1] == 4))


def test_topology_is_permutation_and_translation_equivariant():
    coordinates = torch.tensor(
        [
            [-3, 4, 2],
            [-2, 4, 2],
            [-1, 4, 2],
            [-3, 6, 2],
            [-3, 4, 6],
        ],
        dtype=torch.int64,
    )
    topology = build_axis6_neighbor_indices(coordinates, (1, 2, 4))
    translated = build_axis6_neighbor_indices(
        coordinates + torch.tensor([100, -71, 33]), (1, 2, 4)
    )
    torch.testing.assert_close(topology, translated, rtol=0.0, atol=0.0)

    permutation = torch.tensor([3, 0, 4, 2, 1])
    permuted = build_axis6_neighbor_indices(
        coordinates[permutation], (1, 2, 4)
    )
    old_to_new = torch.empty_like(permutation)
    old_to_new[permutation] = torch.arange(len(permutation))
    expected = topology[permutation]
    valid = expected >= 0
    expected = torch.where(valid, old_to_new[expected.clamp(min=0)], -1)
    torch.testing.assert_close(permuted, expected, rtol=0.0, atol=0.0)


def test_head_is_permutation_and_translation_equivariant():
    model = _model()
    coordinates = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2],
            [4, 0, 0],
        ],
        dtype=torch.int32,
    )
    features = torch.randn(len(coordinates), P1_SPATIAL_FEATURE_DIM)
    with torch.inference_mode():
        logits, regression = model(features, coordinates)
        shifted_logits, shifted_regression = model(
            features, coordinates + torch.tensor([27, -13, 5])
        )
    torch.testing.assert_close(
        logits, shifted_logits, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        regression, shifted_regression, rtol=0.0, atol=0.0
    )

    permutation = torch.tensor([5, 2, 0, 4, 1, 3])
    with torch.inference_mode():
        permuted_logits, permuted_regression = model(
            features[permutation], coordinates[permutation]
        )
    torch.testing.assert_close(
        permuted_logits, logits[permutation], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        permuted_regression,
        regression[permutation],
        rtol=0.0,
        atol=0.0,
    )


def test_head_output_empty_input_and_class_agnostic_contract():
    model = _model()
    features = torch.empty((0, P1_SPATIAL_FEATURE_DIM))
    coordinates = torch.empty((0, 3), dtype=torch.int64)
    encoded = model.encode(features, coordinates)
    logits, regression = model(features, coordinates)
    assert encoded.shape == (0, model.hidden_dim)
    assert logits.shape == (0, 1)
    assert regression.shape == (0, 6)
    assert not hasattr(model, "classification_head")
    assert not hasattr(model, "num_classes")
    assert set(inspect.signature(model.forward).parameters) == {
        "features",
        "coordinates",
    }


def test_public_encoding_is_exact_input_to_frozen_prediction_heads():
    model = _model()
    coordinates = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [0, 2, 0]], dtype=torch.int64
    )
    features = torch.randn(3, P1_SPATIAL_FEATURE_DIM)
    with torch.inference_mode():
        encoded = model.encode(features, coordinates)
        logits, regression = model(features, coordinates)
        expected_logits = model.objectness(encoded)
        expected_regression = model.regression(encoded)
    assert encoded.shape == (3, model.hidden_dim)
    torch.testing.assert_close(
        logits, expected_logits, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        regression, expected_regression, rtol=0.0, atol=0.0
    )


def test_spatial_context_changes_a_voxel_when_only_its_neighbour_changes():
    model = _model()
    coordinates = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [5, 5, 5]], dtype=torch.int64
    )
    first = torch.zeros((3, P1_SPATIAL_FEATURE_DIM))
    second = first.clone()
    second[1, 0] = 3.0
    with torch.inference_mode():
        logits_first, regression_first = model(first, coordinates)
        logits_second, regression_second = model(second, coordinates)
    assert not torch.equal(logits_first[0], logits_second[0])
    assert not torch.equal(regression_first[0], regression_second[0])
    # The disconnected voxel has exactly the same local sparse component.
    torch.testing.assert_close(
        logits_first[2], logits_second[2], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        regression_first[2], regression_second[2], rtol=0.0, atol=0.0
    )


def test_gradients_flow_through_spatial_blocks_but_not_topology():
    model = _model()
    features = torch.randn(
        4, P1_SPATIAL_FEATURE_DIM, requires_grad=True
    )
    coordinates = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 2, 0]],
        dtype=torch.int64,
    )
    logits, regression = model(features, coordinates)
    (logits.square().mean() + regression.square().mean()).backward()
    assert features.grad is not None
    assert bool(torch.isfinite(features.grad).all())
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    topology = build_axis6_neighbor_indices(coordinates, (1, 2))
    assert topology.grad_fn is None
    assert topology.requires_grad is False


def test_model_config_and_validation_fail_closed():
    model = NativeSparseResidualProposalHead()
    config = {
        "architecture": P1_SPATIAL_ARCHITECTURE,
        "input_dim": 14,
        "hidden_dim": 48,
        "regression_dim": 6,
        "dilations": [1, 2, 4],
        "neighborhood": "axis6_submanifold",
        "coordinate_layout": "xyz_or_batch_xyz",
        "regression_encoding": "center_delta_m_log_size_m",
    }
    assert model.model_config() == config
    reconstructed = NativeSparseResidualProposalHead.from_model_config(config)
    assert reconstructed.model_config() == config
    with pytest.raises(ValueError, match="architecture"):
        NativeSparseResidualProposalHead.from_model_config(
            {**config, "architecture": "not-the-frozen-architecture"}
        )
    with pytest.raises(ValueError, match="unique"):
        build_axis6_neighbor_indices(
            torch.tensor([[0, 0, 0], [0, 0, 0]]), (1,)
        )
    with pytest.raises(TypeError, match="integer"):
        build_axis6_neighbor_indices(torch.zeros((2, 3)), (1,))
    with pytest.raises(ValueError, match="equal V"):
        model(
            torch.zeros((2, P1_SPATIAL_FEATURE_DIM)),
            torch.zeros((1, 3), dtype=torch.int64),
        )
    with pytest.raises(ValueError, match="finite"):
        invalid = torch.zeros((1, P1_SPATIAL_FEATURE_DIM))
        invalid[0, 0] = float("nan")
        model(invalid, torch.zeros((1, 3), dtype=torch.int64))
    with pytest.raises(ValueError, match="dilations"):
        NativeSparseResidualProposalHead(dilations=(1, 1))
