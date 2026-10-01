"""CPU tests for the standalone raw/fused query observer."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from boxfusion.raw_fused_query import (
    RAW_FUSED_INPUT_QUALITY_DIM,
    RAW_FUSED_INPUT_QUALITY_NAMES,
    RAW_FUSED_QUERY_FEATURE_DIM,
    RAW_FUSED_QUERY_FEATURE_NAMES,
    RAW_FUSED_QUERY_SCHEMA,
    RAW_FUSED_QUERY_SCORER_FORMAT_VERSION,
    RAW_FUSED_QUERY_SCORER_SCHEMA,
    LinearRawFusedQueryScorer,
    MLPRawFusedQueryScorer,
    load_raw_fused_query_scorer,
    observe_raw_fused_query,
    raw_fused_input_quality_vector,
)


_SIGNS = np.asarray(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=np.float32,
)


def _quality(value: float) -> dict[str, float]:
    return {
        name: float(value) for name in RAW_FUSED_INPUT_QUALITY_NAMES
    }


def _corners(center, size) -> np.ndarray:
    return (
        np.asarray(center, dtype=np.float32)[None, :]
        + 0.5
        * np.asarray(size, dtype=np.float32)[None, :]
        * _SIGNS
    )


def _mixed_observation(*, scorer_checkpoint=None):
    original = np.asarray([0.0, 0.0, 0.0, 1.0, 0.8, 0.6])
    raw_mask = np.asarray(
        [
            [0.02, 0.00, 0.00, 1.0, 0.8, 0.6],
            [4.00, 0.00, 0.00, 1.0, 0.8, 0.6],
        ]
    )
    superpoint = _corners((-0.01, 0.01, 0.0), (1.0, 0.8, 0.6))
    occupancy = np.asarray([0.01, -0.01, 0.0, 1.0, 0.8, 0.6])
    observation = observe_raw_fused_query(
        original=original,
        raw_mask=raw_mask,
        superpoint=superpoint,
        occupancy=occupancy,
        quality_features={
            "original": _quality(0.75),
            "raw_mask": [_quality(0.98), _quality(0.99)],
            "superpoint": _quality(0.84),
            "occupancy": _quality(0.86),
        },
        scorer_checkpoint=scorer_checkpoint,
    )
    return observation, original, raw_mask, superpoint, occupancy


def _checkpoint_common(model_type: str) -> dict[str, np.ndarray]:
    return {
        "schema": np.asarray(RAW_FUSED_QUERY_SCORER_SCHEMA),
        "format_version": np.asarray(
            RAW_FUSED_QUERY_SCORER_FORMAT_VERSION
        ),
        "model_type": np.asarray(model_type),
        "feature_names": np.asarray(RAW_FUSED_QUERY_FEATURE_NAMES),
    }


def test_public_schemas_are_fixed_unique_and_quality_vectors_are_read_only():
    assert RAW_FUSED_QUERY_FEATURE_DIM == 30
    assert len(RAW_FUSED_QUERY_FEATURE_NAMES) == 30
    assert len(set(RAW_FUSED_QUERY_FEATURE_NAMES)) == 30
    assert RAW_FUSED_INPUT_QUALITY_DIM == 5

    values = {
        name: index / RAW_FUSED_INPUT_QUALITY_DIM
        for index, name in enumerate(
            reversed(RAW_FUSED_INPUT_QUALITY_NAMES), start=1
        )
    }
    vector = raw_fused_input_quality_vector(values)
    np.testing.assert_allclose(
        vector,
        [values[name] for name in RAW_FUSED_INPUT_QUALITY_NAMES],
    )
    assert vector.flags.writeable is False
    with pytest.raises(ValueError):
        vector[0] = 0.0


def test_mixed_6d_and_corner_candidates_form_immutable_observer_tables():
    observation, original, raw_mask, superpoint, occupancy = (
        _mixed_observation()
    )

    assert observation.schema == RAW_FUSED_QUERY_SCHEMA
    assert observation.observer_only
    assert not observation.mutation_enabled
    assert not observation.learned_scorer_used
    assert observation.selection_mode == "deterministic_heuristic"
    assert observation.scorer_model_type == "none"
    assert observation.scorer_checkpoint is None
    assert np.isnan(observation.learned_scores).all()

    table = observation.candidate_table
    assert len(table) == 5
    assert table.sources == (
        "original",
        "raw_mask",
        "raw_mask",
        "superpoint",
        "occupancy",
    )
    assert table.corners.shape == (5, 8, 3)
    assert table.center_sizes.shape == (5, 6)
    assert table.quality_features.shape == (5, 5)
    assert observation.features.shape == (5, 30)
    assert observation.feature_names == RAW_FUSED_QUERY_FEATURE_NAMES

    pairwise = observation.pairwise_consensus
    for matrix in (
        pairwise.iou_3d,
        pairwise.center_similarity,
        pairwise.extent_similarity,
        pairwise.consensus,
    ):
        assert matrix.shape == (5, 5)
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-7)
        np.testing.assert_allclose(np.diag(matrix), 1.0)
        assert matrix.flags.writeable is False

    # The distant high-quality raw proposal cannot beat the close
    # cross-source-consistent raw proposal under the documented heuristic.
    assert observation.selected.candidate_id == "raw_mask:0"
    assert observation.selected.observer_only
    assert not observation.selected.applied
    assert observation.selected.learned_score is None
    assert observation.selected.corners.flags.writeable is False
    assert observation.selected.feature_vector.flags.writeable is False

    for array in (
        table.source_indices,
        table.corners,
        table.center_sizes,
        table.aabbs,
        table.quality_features,
        observation.features,
        observation.heuristic_scores,
        observation.learned_scores,
        observation.selection_scores,
    ):
        assert array.flags.writeable is False
        with pytest.raises(ValueError):
            array.flat[0] = 0

    # Inputs stay untouched and writable; no final-box object is accepted or
    # returned by this observer API.
    np.testing.assert_array_equal(
        original, [0.0, 0.0, 0.0, 1.0, 0.8, 0.6]
    )
    np.testing.assert_array_equal(
        raw_mask[1], [4.0, 0.0, 0.0, 1.0, 0.8, 0.6]
    )
    assert original.flags.writeable
    assert raw_mask.flags.writeable
    assert superpoint.flags.writeable
    assert occupancy.flags.writeable
    assert not hasattr(observation, "final_boxes")
    with pytest.raises(FrozenInstanceError):
        observation.selection_mode = "learned_mlp_npz"


def test_heuristic_selection_is_repeatable_and_ties_use_source_index():
    boxes = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ]
    )
    kwargs = {
        "original": boxes,
        "quality_features": {
            "original": [_quality(0.8), _quality(0.8)]
        },
    }
    first = observe_raw_fused_query(**kwargs)
    second = observe_raw_fused_query(**kwargs)
    assert first.selected.candidate_id == "original:0"
    assert second.selected.candidate_id == first.selected.candidate_id
    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_array_equal(
        first.selection_scores, second.selection_scores
    )


def test_min_max_6d_format_is_explicit_and_matches_corner_geometry():
    observation = observe_raw_fused_query(
        original=np.asarray([-0.5, -0.4, -0.3, 0.5, 0.4, 0.3]),
        quality_features={"original": _quality(0.5)},
        six_d_format="min_max",
    )
    np.testing.assert_allclose(
        observation.candidate_table.center_sizes[0],
        [0.0, 0.0, 0.0, 1.0, 0.8, 0.6],
    )


def test_linear_npz_scorer_is_strict_and_marks_learning_truthfully(tmp_path):
    checkpoint = tmp_path / "linear_raw_fused.npz"
    weight = np.zeros(RAW_FUSED_QUERY_FEATURE_DIM, dtype=np.float32)
    weight[
        RAW_FUSED_QUERY_FEATURE_NAMES.index("source_occupancy")
    ] = 10.0
    np.savez(
        checkpoint,
        **_checkpoint_common("linear"),
        weight=weight,
        bias=np.asarray(-5.0),
    )
    loaded = load_raw_fused_query_scorer(checkpoint)
    assert isinstance(loaded, LinearRawFusedQueryScorer)

    observation, *_ = _mixed_observation(
        scorer_checkpoint=checkpoint
    )
    assert observation.learned_scorer_used
    assert observation.selection_mode == "learned_linear_npz"
    assert observation.scorer_model_type == "linear"
    assert observation.scorer_checkpoint == str(checkpoint.resolve())
    assert np.isfinite(observation.learned_scores).all()
    assert observation.selected.source == "occupancy"
    assert observation.selected.learned_score is not None
    assert observation.selected.selection_mode == "learned_linear_npz"


def test_mlp_npz_scorer_runs_relu_hidden_layer_and_selects_superpoint(
    tmp_path,
):
    checkpoint = tmp_path / "mlp_raw_fused.npz"
    first = np.zeros(
        (RAW_FUSED_QUERY_FEATURE_DIM, 2), dtype=np.float32
    )
    first[
        RAW_FUSED_QUERY_FEATURE_NAMES.index("source_superpoint"), 0
    ] = 1.0
    second = np.asarray([[10.0], [0.0]], dtype=np.float32)
    np.savez(
        checkpoint,
        **_checkpoint_common("mlp"),
        num_layers=np.asarray(2),
        weight_0=first,
        bias_0=np.zeros(2, dtype=np.float32),
        weight_1=second,
        bias_1=np.asarray([-5.0], dtype=np.float32),
    )
    loaded = load_raw_fused_query_scorer(checkpoint)
    assert isinstance(loaded, MLPRawFusedQueryScorer)

    observation, *_ = _mixed_observation(
        scorer_checkpoint=checkpoint
    )
    assert observation.selection_mode == "learned_mlp_npz"
    assert observation.selected.source == "superpoint"


@pytest.mark.parametrize(
    "mutator, message",
    [
        (
            lambda payload: payload.update(extra=np.asarray(1)),
            "keys",
        ),
        (
            lambda payload: payload.update(
                schema=np.asarray("wrong.schema")
            ),
            "schema",
        ),
        (
            lambda payload: payload.update(
                feature_names=np.asarray(
                    tuple(reversed(RAW_FUSED_QUERY_FEATURE_NAMES))
                )
            ),
            "feature schema",
        ),
    ],
)
def test_linear_checkpoint_rejects_non_exact_contract(
    tmp_path, mutator, message
):
    payload = {
        **_checkpoint_common("linear"),
        "weight": np.zeros(RAW_FUSED_QUERY_FEATURE_DIM),
        "bias": np.asarray(0.0),
    }
    mutator(payload)
    checkpoint = tmp_path / "invalid.npz"
    np.savez(checkpoint, **payload)
    with pytest.raises(ValueError, match=message):
        load_raw_fused_query_scorer(checkpoint)


def test_invalid_boxes_and_quality_fail_before_observer_selection():
    with pytest.raises(ValueError, match="positive dimensions"):
        observe_raw_fused_query(
            original=np.asarray([0.0, 0.0, 0.0, -1.0, 1.0, 1.0]),
            quality_features={"original": _quality(0.5)},
        )
    with pytest.raises(ValueError, match="missing quality"):
        observe_raw_fused_query(
            original=np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            raw_mask=np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            quality_features={"original": _quality(0.5)},
        )
    invalid_quality = _quality(0.5)
    invalid_quality["depth_quality"] = 1.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        observe_raw_fused_query(
            original=np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            quality_features={"original": invalid_quality},
        )
    with pytest.raises(ValueError, match="at least one original"):
        observe_raw_fused_query(
            original=None,
            occupancy=np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            quality_features={"occupancy": _quality(0.5)},
        )
