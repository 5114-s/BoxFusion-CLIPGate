from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from boxfusion.tr3d_r3_calibration_dataset import (
    GAIN_CLASS,
    HARM_CLASS,
    R3CalibrationDataset,
    SAFE_NEUTRAL_CLASS,
    label_joint_replacements,
    label_dataset_global_leave_one_out,
    label_single_replacement,
    load_dataset,
    prediction_rows,
    write_dataset,
)


def _box(center: float) -> np.ndarray:
    return np.asarray([center - 0.5, 0, 0, center + 0.5, 1, 1], dtype=np.float64)


def test_single_replacement_labels_gain_harm_and_neutral() -> None:
    gain, gain_delta = label_single_replacement(
        np.stack([_box(10)]), np.asarray([0.8]), np.stack([_box(0)]),
        anchor_index=0, candidate_box=_box(0),
    )
    assert gain == GAIN_CLASS
    assert gain_delta.tolist() == [1, 1, 1]
    harm, harm_delta = label_single_replacement(
        np.stack([_box(0)]), np.asarray([0.8]), np.stack([_box(0)]),
        anchor_index=0, candidate_box=_box(10),
    )
    assert harm == HARM_CLASS
    assert harm_delta.tolist() == [-1, -1, -1]
    neutral, neutral_delta = label_single_replacement(
        np.stack([_box(10)]), np.asarray([0.8]), np.stack([_box(0)]),
        anchor_index=0, candidate_box=_box(11),
    )
    assert neutral == SAFE_NEUTRAL_CLASS
    assert neutral_delta.tolist() == [0, 0, 0]


def test_joint_leave_one_out_exposes_non_composable_neutral_candidates() -> None:
    # With GT A/B/C, replacing A->C or B->C alone keeps two matches.  Applying
    # both creates duplicate C predictions and leaves only one match.  The
    # joint leave-one-out label must mark both as harmful.
    labels, deltas = label_joint_replacements(
        np.stack([_box(0), _box(4)]),
        np.asarray([0.9, 0.8]),
        np.stack([_box(0), _box(4), _box(8)]),
        anchor_indices=np.asarray([0, 1]),
        candidate_boxes=np.stack([_box(8), _box(8)]),
    )
    assert labels.tolist() == [HARM_CLASS, HARM_CLASS]
    assert deltas.tolist() == [[-1, -1, -1], [-1, -1, -1]]


def _dataset() -> R3CalibrationDataset:
    provisional = R3CalibrationDataset(
        scene_ids=np.asarray(["scene0000_00", "scene0001_00"]),
        anchor_offsets=np.asarray([0, 1, 2]),
        anchor_boxes=np.stack([_box(10), _box(0)]),
        anchor_scores=np.asarray([0.8, 0.7]),
        gt_offsets=np.asarray([0, 1, 2]),
        gt_boxes=np.stack([_box(0), _box(0)]),
        sample_scene_index=np.asarray([0, 1]),
        sample_anchor_index=np.asarray([0, 0]),
        proposal_ids=np.asarray([10, 20]),
        candidate_boxes=np.stack([_box(0), _box(10)]),
        features=np.arange(12, dtype=np.float64).reshape(2, 6),
        labels=np.asarray([SAFE_NEUTRAL_CLASS, SAFE_NEUTRAL_CLASS], dtype=np.int8),
        tp_deltas=np.zeros((2, 3), dtype=np.int8),
        ap_deltas=np.zeros((2, 3), dtype=np.float64),
        provenance={"validation_scene_overlap": 0, "global_anchor_scores_unique": True},
    )
    labels, tp_deltas, ap_deltas = label_dataset_global_leave_one_out(provisional)
    return R3CalibrationDataset(
        **{
            **provisional.__dict__,
            "labels": labels,
            "tp_deltas": tp_deltas,
            "ap_deltas": ap_deltas,
        }
    )


def test_dataset_round_trip_and_reconstructs_only_accepted_rows(tmp_path: Path) -> None:
    path = tmp_path / "dataset.npz"
    write_dataset(path, _dataset())
    assert path.stat().st_mode & 0o222 == 0
    loaded = load_dataset(path)
    assert loaded.sample_count == 2
    rows = prediction_rows(loaded, np.asarray([True, False]))
    np.testing.assert_array_equal(rows[0][1][0], _box(0))
    np.testing.assert_array_equal(rows[1][1][0], _box(0))
    with pytest.raises(FileExistsError, match="immutable"):
        write_dataset(path, _dataset())


def test_dataset_rejects_duplicate_primary_anchor() -> None:
    dataset = _dataset()
    broken = R3CalibrationDataset(
        **{
            **dataset.__dict__,
            "sample_scene_index": np.asarray([0, 0]),
            "sample_anchor_index": np.asarray([0, 0]),
        }
    )
    with pytest.raises(ValueError, match="more than one"):
        broken.validate()


def test_global_rank_aware_label_detects_ap_loss_with_equal_tp_count() -> None:
    # Baseline order A,B,A has both TPs in the first two ranks.  Raw R3 changes
    # the top row A->B, yielding B,B,A: TP count is unchanged, but AP falls.
    provisional = R3CalibrationDataset(
        scene_ids=np.asarray(["scene0000_00"]),
        anchor_offsets=np.asarray([0, 3], dtype=np.int64),
        anchor_boxes=np.stack([_box(0), _box(4), _box(0)]),
        anchor_scores=np.asarray([0.9, 0.8, 0.7], dtype=np.float64),
        gt_offsets=np.asarray([0, 2], dtype=np.int64),
        gt_boxes=np.stack([_box(0), _box(4)]),
        sample_scene_index=np.asarray([0], dtype=np.int64),
        sample_anchor_index=np.asarray([0], dtype=np.int64),
        proposal_ids=np.asarray([10], dtype=np.int64),
        candidate_boxes=np.stack([_box(4)]),
        features=np.zeros((1, 6), dtype=np.float64),
        labels=np.asarray([SAFE_NEUTRAL_CLASS], dtype=np.int8),
        tp_deltas=np.zeros((1, 3), dtype=np.int8),
        ap_deltas=np.zeros((1, 3), dtype=np.float64),
        provenance={"global_anchor_scores_unique": True},
    )
    labels, tp_deltas, ap_deltas = label_dataset_global_leave_one_out(provisional)
    assert labels.tolist() == [HARM_CLASS]
    assert tp_deltas.tolist() == [[0, 0, 0]]
    assert np.all(ap_deltas < 0)


def test_dataset_rejects_silent_float_to_integer_coercion() -> None:
    dataset = _dataset()
    broken = R3CalibrationDataset(
        **{**dataset.__dict__, "labels": np.asarray([0.9, 2.0], dtype=np.float64)}
    )
    with pytest.raises(ValueError, match="exact dtype int8"):
        broken.validate()
