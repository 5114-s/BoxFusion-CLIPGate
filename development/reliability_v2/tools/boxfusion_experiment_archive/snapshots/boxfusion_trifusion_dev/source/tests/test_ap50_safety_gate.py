import json

import numpy as np
import pytest

from boxfusion.ap50_safety_gate import (
    AP50_SAFETY_GATE_FORMAT_VERSION,
    AP50_SAFETY_GATE_OUTPUT_NAMES,
    AP50_SAFETY_GATE_SCHEMA,
    AP50SafetyGate,
    AP50SafetyGateConfig,
    load_ap50_safety_gate,
)


FEATURE_NAMES = ("support", "projection", "view_count")


def _logit(probability):
    return np.log(probability / (1.0 - probability))


def _gate(final_bias=None):
    bias = np.asarray(
        [
            np.arctanh(0.10),
            -9.0,
            _logit(0.92),
            _logit(0.03),
            _logit(0.40),
            _logit(0.58),
            _logit(0.70),
            _logit(0.80),
        ]
        if final_bias is None
        else final_bias,
        dtype=np.float64,
    )
    return AP50SafetyGate(
        feature_names=FEATURE_NAMES,
        weights=(
            np.zeros((3, 4), dtype=np.float64),
            np.zeros((4, 8), dtype=np.float64),
        ),
        biases=(
            np.ones(4, dtype=np.float64),
            bias,
        ),
        feature_mean=np.zeros(3, dtype=np.float64),
        feature_scale=np.ones(3, dtype=np.float64),
        metadata={"split": "train-only"},
    )


def _checkpoint(path, *, extra=None):
    gate = _gate()
    arrays = {
        "schema": np.asarray(AP50_SAFETY_GATE_SCHEMA),
        "format_version": np.asarray(
            AP50_SAFETY_GATE_FORMAT_VERSION, dtype=np.int64
        ),
        "feature_names": np.asarray(FEATURE_NAMES),
        "output_names": np.asarray(AP50_SAFETY_GATE_OUTPUT_NAMES),
        "feature_mean": gate.feature_mean.astype(np.float32),
        "feature_scale": gate.feature_scale.astype(np.float32),
        "maximum_absolute_delta": np.asarray(1.0, dtype=np.float32),
        "num_layers": np.asarray(2, dtype=np.int64),
        "weight_0": gate.weights[0].astype(np.float32),
        "bias_0": gate.biases[0].astype(np.float32),
        "weight_1": gate.weights[1].astype(np.float32),
        "bias_1": gate.biases[1].astype(np.float32),
        "metadata_json": np.asarray(json.dumps(gate.metadata, sort_keys=True)),
    }
    arrays.update(extra or {})
    np.savez(path, **arrays)


def test_prediction_and_conservative_decision_accept():
    gate = _gate()
    prediction = gate.predict(
        {"support": 0.9, "projection": 0.8, "view_count": 0.6}
    )
    assert prediction.delta_mean == pytest.approx(0.10, abs=1e-6)
    assert prediction.delta_std == pytest.approx(np.exp(-4.5), rel=1e-5)
    assert prediction.improvement_probability == pytest.approx(0.92)
    assert prediction.harm_probability == pytest.approx(0.03)
    assert prediction.original_iou == pytest.approx(0.40)
    assert prediction.candidate_iou == pytest.approx(0.58)

    decision = gate.decide(
        np.asarray([0.9, 0.8, 0.6]),
        geometry_verified=True,
        config=AP50SafetyGateConfig(require_iou50_crossing=True),
    )
    assert decision.accepted
    assert decision.reason == "accepted"
    assert decision.lower_confidence_delta > 0.08


@pytest.mark.parametrize(
    ("geometry_verified", "bias_index", "bias_value", "reason"),
    [
        (False, None, None, "geometry_not_verified"),
        (True, 2, _logit(0.20), "improvement_probability"),
        (True, 3, _logit(0.80), "harm_probability"),
        (True, 5, _logit(0.401), "predicted_iou_margin"),
        (True, 7, _logit(0.10), "iou50_crossing_probability"),
    ],
)
def test_rejection_reasons_are_deterministic(
    geometry_verified, bias_index, bias_value, reason
):
    bias = _gate().biases[-1].copy()
    if bias_index is not None:
        bias[bias_index] = bias_value
    gate = _gate(final_bias=bias)
    config = AP50SafetyGateConfig(require_iou50_crossing=True)
    decision = gate.decide(
        np.asarray([0.2, 0.3, 0.4]),
        geometry_verified=geometry_verified,
        config=config,
    )
    assert not decision.accepted
    assert decision.reason == reason


def test_feature_schema_is_exact_and_batch_order_is_stable():
    gate = _gate()
    with pytest.raises(ValueError, match="schema mismatch"):
        gate.predict({"support": 1.0, "projection": 1.0})
    with pytest.raises(ValueError, match="schema mismatch"):
        gate.predict(
            {
                "support": 1.0,
                "projection": 1.0,
                "view_count": 1.0,
                "extra": 0.0,
            }
        )
    predictions = gate.predict(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    )
    assert len(predictions) == 2
    assert predictions[0] == predictions[1]


def test_checkpoint_round_trip_and_strict_keys(tmp_path):
    path = tmp_path / "gate.npz"
    _checkpoint(path)
    gate = load_ap50_safety_gate(path)
    assert gate.feature_names == FEATURE_NAMES
    assert gate.metadata == {"split": "train-only"}
    assert gate.decide(
        np.ones(3), geometry_verified=True
    ).accepted

    bad_path = tmp_path / "bad.npz"
    _checkpoint(bad_path, extra={"unexpected": np.asarray(1)})
    with pytest.raises(ValueError, match="strict schema"):
        load_ap50_safety_gate(bad_path)


def test_invalid_normalization_and_nonfinite_features_fail_closed():
    with pytest.raises(ValueError, match="strictly positive"):
        AP50SafetyGate(
            feature_names=FEATURE_NAMES,
            weights=(np.zeros((3, 8)),),
            biases=(np.zeros(8),),
            feature_mean=np.zeros(3),
            feature_scale=np.asarray([1.0, 0.0, 1.0]),
        )
    with pytest.raises(ValueError, match="finite"):
        _gate().predict(np.asarray([0.0, np.nan, 0.0]))
