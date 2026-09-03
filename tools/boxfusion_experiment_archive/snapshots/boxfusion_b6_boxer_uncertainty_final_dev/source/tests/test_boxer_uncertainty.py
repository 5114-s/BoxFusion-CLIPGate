import numpy as np
import pytest
import torch

from boxfusion.boxer_uncertainty import (
    resolve_boxer_uncertainty_config,
    uncertainty_adjusted_selection,
)
from boxfusion.instances import Instances3D


def _cfg(**overrides):
    values = {
        "mode": "active",
        "confidence_power": 1.0,
        "minimum_confidence": 0.05,
    }
    values.update(overrides)
    return resolve_boxer_uncertainty_config(
        {"boxer_uncertainty": values}
    )


def _selection():
    weights = np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    confidence = np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    selected = np.asarray([0, 1, 2], dtype=np.int64)
    return {
        "weights": weights,
        "confidence": confidence,
        "selected_indices": selected,
        "selected_weights": weights[selected],
        "ranked_indices": np.arange(4, dtype=np.int64),
        "selected_mask": np.asarray([True, True, True, False]),
    }


def test_boxer_confidence_changes_top_k_without_mutating_base_selection():
    base = _selection()
    original_weights = base["weights"].copy()
    adjusted = uncertainty_adjusted_selection(
        base,
        boxer_confidence=np.asarray([0.1, 1.0, 1.0, 1.0]),
        boxer_geometry_applied=np.ones(4, dtype=bool),
        cfg=_cfg(),
    )

    np.testing.assert_array_equal(base["weights"], original_weights)
    np.testing.assert_allclose(
        adjusted["uncertainty_factors"], [0.1, 1.0, 1.0, 1.0]
    )
    np.testing.assert_array_equal(adjusted["selected_indices"], [1, 2, 3])
    assert bool(adjusted["selection_changed"])
    assert bool(adjusted["ranking_changed"])
    assert bool(adjusted["candidate_weights_changed"])
    assert bool(adjusted["effective_weights_changed"])


def test_cutr_fallback_and_invalid_confidence_are_neutral():
    adjusted = uncertainty_adjusted_selection(
        _selection(),
        boxer_confidence=np.asarray([0.01, np.nan, 0.7, 0.6]),
        boxer_geometry_applied=np.asarray([False, True, False, False]),
        cfg=_cfg(),
    )

    np.testing.assert_allclose(adjusted["uncertainty_factors"], 1.0)
    np.testing.assert_array_equal(adjusted["selected_indices"], [0, 1, 2])
    assert not bool(adjusted["selection_changed"])
    assert not bool(adjusted["ranking_changed"])
    assert not bool(adjusted["effective_weights_changed"])
    assert adjusted["boxer_confidence_valid"].tolist() == [True, False, True, True]


def test_unselected_uncertainty_change_is_not_an_effective_fusion_change():
    adjusted = uncertainty_adjusted_selection(
        _selection(),
        boxer_confidence=np.asarray([1.0, 1.0, 1.0, 0.5]),
        boxer_geometry_applied=np.ones(4, dtype=bool),
        cfg=_cfg(),
    )

    assert bool(adjusted["candidate_weights_changed"])
    assert not bool(adjusted["selection_changed"])
    assert not bool(adjusted["effective_weights_changed"])


def test_common_selected_factor_cancelled_by_normalization_is_not_effective():
    adjusted = uncertainty_adjusted_selection(
        _selection(),
        boxer_confidence=np.asarray([0.5, 0.5, 0.5, 0.5]),
        boxer_geometry_applied=np.ones(4, dtype=bool),
        cfg=_cfg(),
    )

    assert bool(adjusted["candidate_weights_changed"])
    assert not bool(adjusted["selection_changed"])
    assert not bool(adjusted["effective_weights_changed"])


def test_minimum_confidence_and_power_are_applied_only_to_boxer_rows():
    adjusted = uncertainty_adjusted_selection(
        _selection(),
        boxer_confidence=np.asarray([0.01, 0.5, 0.5, 0.5]),
        boxer_geometry_applied=np.asarray([True, True, False, False]),
        cfg=_cfg(confidence_power=2.0, minimum_confidence=0.05),
    )
    np.testing.assert_allclose(
        adjusted["uncertainty_factors"],
        [0.05**2, 0.5**2, 1.0, 1.0],
    )


@pytest.mark.parametrize(
    "values",
    [
        {"mode": "bad"},
        {"confidence_power": -1.0},
        {"confidence_power": float("nan")},
        {"minimum_confidence": 0.0},
        {"minimum_confidence": 1.1},
    ],
)
def test_invalid_uncertainty_config_fails_fast(values):
    with pytest.raises(ValueError):
        _cfg(**values)


def test_uncertainty_fields_survive_instance_cat_and_indexing():
    first = Instances3D()
    first.scores = torch.tensor([0.8, 0.7])
    first.boxer_aleatoric_confidence = torch.tensor([0.9, 0.6])
    first.boxer_aleatoric_logvar = torch.tensor([-2.0, -0.4])
    first.boxer_geometry_applied = torch.tensor([True, False])
    second = Instances3D()
    second.scores = torch.tensor([0.5])
    second.boxer_aleatoric_confidence = torch.tensor([0.75])
    second.boxer_aleatoric_logvar = torch.tensor([-1.1])
    second.boxer_geometry_applied = torch.tensor([True])

    merged = Instances3D.cat([first, second])
    selected = merged[torch.tensor([2, 0], dtype=torch.int64)]

    torch.testing.assert_close(
        selected.boxer_aleatoric_confidence,
        torch.tensor([0.75, 0.9]),
    )
    assert selected.boxer_geometry_applied.tolist() == [True, True]
