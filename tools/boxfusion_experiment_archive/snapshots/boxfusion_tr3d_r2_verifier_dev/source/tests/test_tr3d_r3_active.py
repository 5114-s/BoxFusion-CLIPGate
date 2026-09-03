from __future__ import annotations

from copy import copy
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from boxfusion.tr3d_r3_active import (
    R3_ACTIVE_CONFIG_SCHEMA,
    R3_ACTIVE_SUMMARY_SCHEMA,
    active_code_sha256,
    active_config,
    active_config_sha256,
    load_prediction_payload,
    materialize_shadow_active_prediction,
    validate_shadow_active_prediction,
)


def _corners(offset: float) -> np.ndarray:
    unit = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    return np.ascontiguousarray(unit + np.float32(offset))


def _source() -> list[list[tuple[int, np.ndarray, float]]]:
    return [
        [
            (0, _corners(0.0), 0.4),
            (1, _corners(2.0), 0.5),
            (2, _corners(4.0), 0.2),
        ]
    ]


def _cache() -> SimpleNamespace:
    return SimpleNamespace(
        anchor_count=3,
        proposal_ids=np.asarray([20, 10, 30, 40, 50], dtype=np.int64),
        proposal_corners_world=np.stack(
            [
                _corners(8.0),
                _corners(10.0),
                _corners(12.0),
                _corners(14.0),
                _corners(4.0),
            ]
        ),
        anchor_index=np.asarray([0, 0, 1, 2, 2], dtype=np.int64),
        tr3d_score=np.asarray([0.8, 0.8, 0.5, 0.7, 0.9], dtype=np.float32),
        anchor_score=np.asarray([0.4, 0.4, 0.5, 0.2, 0.2], dtype=np.float32),
    )


def _replace_cache(cache: SimpleNamespace, **updates: object) -> SimpleNamespace:
    result = copy(cache)
    for name, value in updates.items():
        setattr(result, name, value)
    return result


def test_materialize_selects_deterministically_and_only_replaces_geometry() -> None:
    source = _source()
    cache = _cache()
    source_geometry = [row[1].copy() for row in source[0]]

    output, summary = materialize_shadow_active_prediction(source, cache)

    assert type(output) is list
    assert type(output[0]) is list
    assert len(output[0]) == len(source[0]) == 3
    assert [row[0] for row in output[0]] == [0, 1, 2]
    assert [row[2] for row in output[0]] == [0.4, 0.5, 0.2]
    assert all(type(row) is tuple for row in output[0])

    # Anchor 0 has an exact score tie.  Proposal id 10 (row 1) must win.
    np.testing.assert_array_equal(output[0][0][1], cache.proposal_corners_world[1])
    assert output[0][0][1] is not cache.proposal_corners_world[1]
    # Equality does not pass the strict gate, so anchor 1 is reused verbatim.
    assert output[0][1] is source[0][1]
    assert output[0][1][1] is source[0][1][1]
    # Anchor 2 passes the score gate but its selected geometry is an exact noop.
    np.testing.assert_array_equal(output[0][2][1], source[0][2][1])
    assert output[0][2][1] is not source[0][2][1]

    assert summary.prediction_count == 3
    assert summary.candidate_count == 5
    assert summary.represented_anchor_count == 3
    assert summary.selected_count == 2
    assert summary.changed_count == 1
    assert summary.noop is False
    assert [record.anchor_index for record in summary.selections] == [0, 2]
    assert [record.proposal_id for record in summary.selections] == [10, 50]
    assert [record.geometry_changed for record in summary.selections] == [True, False]
    summary_dict = summary.as_dict()
    assert summary_dict["schema"] == R3_ACTIVE_SUMMARY_SCHEMA
    assert summary_dict["applied_count"] == summary.selected_count == 2
    assert summary_dict["byte_changed_count"] == summary.changed_count == 1
    assert summary_dict["selections"][0]["proposal_id"] == 10

    # Materialization must never mutate the frozen source payload.
    for row, before in zip(source[0], source_geometry):
        np.testing.assert_array_equal(row[1], before)


def test_materialize_preserves_tuple_and_detection_list_types() -> None:
    source = (([7, _corners(1.0), 0.25],),)
    cache = SimpleNamespace(
        anchor_count=1,
        proposal_ids=np.asarray([9], dtype=np.int64),
        proposal_corners_world=np.stack([_corners(7.0)]),
        anchor_index=np.asarray([0], dtype=np.int64),
        tr3d_score=np.asarray([0.75], dtype=np.float32),
        anchor_score=np.asarray([0.25], dtype=np.float32),
    )

    output, summary = materialize_shadow_active_prediction(source, cache)

    assert type(output) is tuple
    assert type(output[0]) is tuple
    assert type(output[0][0]) is list
    assert type(output[0][0][0]) is int
    assert type(output[0][0][2]) is float
    np.testing.assert_array_equal(output[0][0][1], cache.proposal_corners_world[0])
    assert summary.changed_count == 1


def test_empty_cache_is_exact_noop_and_validator_accepts_it() -> None:
    source = _source()
    cache = SimpleNamespace(
        anchor_count=3,
        proposal_ids=np.empty((0,), dtype=np.int64),
        proposal_corners_world=np.empty((0, 8, 3), dtype=np.float32),
        anchor_index=np.empty((0,), dtype=np.int64),
        tr3d_score=np.empty((0,), dtype=np.float32),
        anchor_score=np.empty((0,), dtype=np.float32),
    )

    output, summary = materialize_shadow_active_prediction(source, cache)

    assert summary.selected_count == 0
    assert summary.changed_count == 0
    assert summary.noop is True
    assert all(wanted is observed for wanted, observed in zip(source[0], output[0]))
    assert validate_shadow_active_prediction(source, output, cache) == summary


def test_validator_rejects_every_output_contract_violation() -> None:
    source = _source()
    cache = _cache()
    output, _ = materialize_shadow_active_prediction(source, cache)

    assert validate_shadow_active_prediction(source, output, cache).changed_count == 1

    wrong_outer = (output[0],)
    with pytest.raises(ValueError, match="container types"):
        validate_shadow_active_prediction(source, wrong_outer, cache)

    wrong_count = [list(output[0][:-1])]
    with pytest.raises(ValueError, match="prediction count"):
        validate_shadow_active_prediction(source, wrong_count, cache)

    wrong_type = [list(output[0])]
    wrong_type[0][0] = list(wrong_type[0][0])
    with pytest.raises(ValueError, match="detection type"):
        validate_shadow_active_prediction(source, wrong_type, cache)

    wrong_label = [list(output[0])]
    row = wrong_label[0][0]
    wrong_label[0][0] = (99, row[1], row[2])
    with pytest.raises(ValueError, match="changed label"):
        validate_shadow_active_prediction(source, wrong_label, cache)

    wrong_score = [list(output[0])]
    row = wrong_score[0][0]
    wrong_score[0][0] = (row[0], row[1], 0.41)
    with pytest.raises(ValueError, match="changed score"):
        validate_shadow_active_prediction(source, wrong_score, cache)

    wrong_geometry = [list(output[0])]
    row = wrong_geometry[0][0]
    wrong_geometry[0][0] = (row[0], _corners(99.0), row[2])
    with pytest.raises(ValueError, match="geometry mismatch"):
        validate_shadow_active_prediction(source, wrong_geometry, cache)

    wrong_order = [list(reversed(output[0]))]
    with pytest.raises(ValueError, match="changed label|changed score|geometry mismatch"):
        validate_shadow_active_prediction(source, wrong_order, cache)


@pytest.mark.parametrize(
    ("geometry", "message"),
    [
        (_corners(0.0).astype(np.float64), "float32"),
        (np.zeros((4, 6), dtype=np.float32), "float32"),
        (np.zeros((8, 6), dtype=np.float32)[:, ::2], "C-contiguous"),
        (np.full((8, 3), np.nan, dtype=np.float32), "finite"),
    ],
)
def test_source_geometry_validation_is_strict(
    geometry: np.ndarray, message: str
) -> None:
    source = [[(0, geometry, 0.4)]]
    cache = SimpleNamespace(
        anchor_count=1,
        proposal_ids=np.empty((0,), dtype=np.int64),
        proposal_corners_world=np.empty((0, 8, 3), dtype=np.float32),
        anchor_index=np.empty((0,), dtype=np.int64),
        tr3d_score=np.empty((0,), dtype=np.float32),
        anchor_score=np.empty((0,), dtype=np.float32),
    )
    with pytest.raises(ValueError, match=message):
        materialize_shadow_active_prediction(source, cache)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"proposal_ids": np.asarray([1, 1, 3, 4, 5], dtype=np.int64)}, "unique"),
        ({"proposal_ids": np.asarray([1, 2, 3, 4, 5], dtype=np.int32)}, "int64"),
        ({"anchor_index": np.asarray([0, 0, 1, 2, 3], dtype=np.int64)}, "range"),
        (
            {"tr3d_score": np.asarray([0.8, 0.8, np.nan, 0.7, 0.9], dtype=np.float32)},
            "finite",
        ),
        (
            {"proposal_corners_world": np.zeros((5, 8, 3), dtype=np.float64)},
            "float32",
        ),
    ],
)
def test_cache_shape_dtype_range_and_finite_validation_is_strict(
    update: dict[str, np.ndarray], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        materialize_shadow_active_prediction(
            _source(), _replace_cache(_cache(), **update)
        )


def test_cached_anchor_scores_must_match_fresh_payload_scores_exactly() -> None:
    cache = _cache()
    wrong = cache.anchor_score.copy()
    wrong[0] = np.float32(0.41)
    with pytest.raises(ValueError, match="disagree"):
        materialize_shadow_active_prediction(
            _source(), _replace_cache(cache, anchor_score=wrong)
        )


def test_config_and_hashes_freeze_shadow_active_contract() -> None:
    config = active_config()
    assert config["schema"] == R3_ACTIVE_CONFIG_SCHEMA
    assert config["ground_truth_access"] is False
    assert config["clip_access"] is False
    assert config["clip_semantics_unchanged"] is True
    assert config["axis_alignment_applied_by_materializer"] is False
    assert config["output_mutation"] == "geometry_only"
    assert set(config["preserved_fields"]) == {
        "label",
        "score",
        "order",
        "count",
        "container_types",
    }
    assert len(active_code_sha256()) == 64
    assert len(active_config_sha256()) == 64
    assert active_config_sha256() == active_config_sha256()


def test_load_prediction_payload_accepts_protocol_five_and_validates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scene_boxes.pkl"
    with path.open("wb") as handle:
        pickle.dump(_source(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    loaded = load_prediction_payload(path)

    assert type(loaded) is list
    assert len(loaded[0]) == 3
    assert path.read_bytes()[:2] == b"\x80\x05"
