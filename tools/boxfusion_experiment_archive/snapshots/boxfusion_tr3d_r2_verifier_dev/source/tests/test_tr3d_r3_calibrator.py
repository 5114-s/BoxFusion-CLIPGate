from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from boxfusion.tr3d_r3_calibrator import (
    R3VetoCalibrator,
    candidate_features,
    load_calibrator,
    materialize_calibrated_prediction,
)


def _corners(offset: float) -> np.ndarray:
    signs = np.asarray(
        [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)],
        dtype=np.float32,
    )
    return np.ascontiguousarray(signs + np.float32(offset))


def _source():
    return [[(0, _corners(0), 0.2), (1, _corners(3), 0.2)]]


def _cache():
    return SimpleNamespace(
        anchor_count=2,
        proposal_ids=np.asarray([10, 20], dtype=np.int64),
        proposal_corners_world=np.stack([_corners(8), _corners(9)]),
        anchor_index=np.asarray([0, 1], dtype=np.int64),
        tr3d_score=np.asarray([0.8, 0.8], dtype=np.float32),
        anchor_score=np.asarray([0.2, 0.2], dtype=np.float32),
        anchor_iou=np.asarray([0.8, 0.2], dtype=np.float32),
        center_distance_over_anchor_diagonal=np.asarray([0.1, 0.5], dtype=np.float32),
        volume_ratio=np.asarray([1.0, 2.0], dtype=np.float32),
        point_density_m3=np.asarray([9.0, 3.0], dtype=np.float32),
    )


def _model(authorized: bool = True) -> R3VetoCalibrator:
    coefficients = np.zeros((3, 6), dtype=np.float64)
    coefficients[0, 2] = 2.0
    coefficients[2, 2] = -2.0
    return R3VetoCalibrator(
        feature_mean=np.zeros(6, dtype=np.float64),
        feature_scale=np.ones(6, dtype=np.float64),
        coefficients=coefficients,
        intercept=np.asarray([-1.0, -10.0, 1.0], dtype=np.float64),
        activation_authorized=authorized,
        dataset_sha256="1" * 64,
        scene_list_sha256="2" * 64,
        metadata={"test": True},
    )


def test_features_use_actual_anchor_score_and_frozen_order() -> None:
    values = candidate_features(_source(), _cache(), [0, 1])
    assert values.shape == (2, 6)
    assert values[0, 0] == pytest.approx(np.log(4.0))
    assert values[0, 1] == pytest.approx(-np.log(4.0))
    assert values[:, 2].tolist() == pytest.approx([0.8, 0.2])
    assert values[:, 4].tolist() == pytest.approx([0.0, np.log(2.0)])
    assert values[:, 5].tolist() == pytest.approx([np.log(10.0), np.log(4.0)])


def test_calibrator_can_only_veto_primary_geometry() -> None:
    source = _source()
    output, summary = materialize_calibrated_prediction(source, _cache(), _model())
    assert summary["primary_count"] == 2
    assert summary["accepted_count"] == 1
    assert summary["vetoed_count"] == 1
    np.testing.assert_array_equal(output[0][0][1], _cache().proposal_corners_world[0])
    assert output[0][1] is source[0][1]
    assert [row[0] for row in output[0]] == [0, 1]
    assert [row[2] for row in output[0]] == [0.2, 0.2]


def test_unauthorized_calibrator_fails_closed() -> None:
    with pytest.raises(PermissionError, match="train-only gate"):
        materialize_calibrated_prediction(_source(), _cache(), _model(False))


def test_exact_harm_probability_tie_fails_closed() -> None:
    model = R3VetoCalibrator(
        feature_mean=np.zeros(6), feature_scale=np.ones(6),
        coefficients=np.zeros((3, 6)), intercept=np.zeros(3),
        activation_authorized=True, dataset_sha256="1" * 64,
        scene_list_sha256="2" * 64, metadata={},
    )
    assert model.accept(np.zeros((1, 6))).tolist() == [False]


def test_model_json_round_trip_and_contract(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    path.write_text(json.dumps(_model().as_dict()), encoding="utf-8")
    loaded = load_calibrator(path)
    np.testing.assert_array_equal(loaded.coefficients, _model().coefficients)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["may_add_primary_replacements"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="veto-only"):
        load_calibrator(path)


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_model_json_rejects_non_boolean_authorization(
    tmp_path: Path, value: object
) -> None:
    path = tmp_path / "model.json"
    payload = _model().as_dict()
    payload["activation_authorized"] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="strict JSON boolean"):
        load_calibrator(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [("veto_only", "yes"), ("veto_only", 1),
     ("may_add_primary_replacements", 0),
     ("may_add_primary_replacements", "false")],
)
def test_model_json_rejects_truthy_or_falsy_non_boolean_contract_fields(
    tmp_path: Path, field: str, value: object
) -> None:
    path = tmp_path / "model.json"
    payload = _model().as_dict()
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="veto-only"):
        load_calibrator(path)
