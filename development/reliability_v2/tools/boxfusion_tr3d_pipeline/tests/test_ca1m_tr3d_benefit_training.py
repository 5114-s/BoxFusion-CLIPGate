from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from boxfusion.ca1m_tr3d_terminal_gate import QUALITY_TARGET
from tools.build_ca1m_tr3d_benefit_dataset import (
    _validation_ids,
    _write_json_create_only as write_dataset_json_create_only,
    _write_npz_create_only as write_dataset_npz_create_only,
)
from tools.train_ca1m_tr3d_benefit_gate import (
    _auc,
    _fit_logistic,
    _gate_conditions,
    _metrics,
    _selection,
    _sigmoid,
    _write_json_create_only as write_training_json_create_only,
    _write_npz_create_only as write_training_npz_create_only,
)


def test_quality_target_names_strict_evaluator_inequality():
    assert QUALITY_TARGET == "candidate_max_gt_iou_strict_gt_0.25"


def test_validation_list_is_exact_sorted_full107(tmp_path: Path):
    target = tmp_path / "full107.txt"
    rows = [f"{40_000_000 + index:08d}" for index in range(107)]
    target.write_text("\n".join(rows) + "\n")
    assert _validation_ids(target) == tuple(rows)

    too_short = tmp_path / "short.txt"
    too_short.write_text("\n".join(rows[:-1]) + "\n")
    with pytest.raises(ValueError, match="exact sorted full107"):
        _validation_ids(too_short)

    reversed_path = tmp_path / "reversed.txt"
    reversed_path.write_text("\n".join(reversed(rows)) + "\n")
    with pytest.raises(ValueError, match="exact sorted full107"):
        _validation_ids(reversed_path)


def test_fast_metric_has_global_score_sort_and_duplicate_suppression():
    scene = np.asarray(["40000000", "40000000", "40000000"])
    score = np.asarray([0.9, 0.8, 0.7], np.float32)
    best_iou = np.asarray([0.6, 0.7, 0.3], np.float64)
    best_gt = np.asarray([0, 0, 1], np.int64)
    result = _metrics(
        scene_ids=scene,
        scores=score,
        best_iou=best_iou,
        best_gt=best_gt,
        scene_table=np.asarray(["40000000"]),
        gt_counts=np.asarray([2], np.int64),
    )
    assert result["iou_0.50"]["tp"] == 1
    assert result["iou_0.50"]["fp"] == 2
    assert result["iou_0.50"]["ap"] == pytest.approx(0.499999750000125)
    assert result["iou_0.25"]["tp"] == 2
    assert result["iou_0.25"]["fp"] == 1
    assert result["iou_0.25"]["ap"] == pytest.approx(0.833332916666875)


def test_dual_gate_selection_is_deterministic_and_capped_per_scene():
    selected = _selection(
        candidate_scene=np.asarray(
            ["40000000", "40000000", "40000000", "40000001"]
        ),
        candidate_rows=np.asarray([10, 11, 12, 2], np.int64),
        anchor_indices=np.asarray([0, 0, 1, 0], np.int64),
        candidate_scores=np.asarray([0.8, 0.9, 0.7, 0.6], np.float32),
        quality_probability=np.asarray([0.9, 0.8, 0.7, 0.9]),
        benefit_probability=np.asarray([0.8, 0.9, 0.95, 0.9]),
        quality_threshold=0.5,
        benefit_threshold=0.5,
        maximum=1,
    )
    # Scene 40000000 first keeps candidate row 12 because its benefit is
    # highest; scene 40000001 independently keeps its only candidate.
    assert selected.tolist() == [2, 3]


def test_class_balanced_logistic_is_deterministic_and_informative():
    x = np.asarray([[-2.0], [-1.0], [1.0], [2.0]], np.float64)
    y = np.asarray([0.0, 0.0, 1.0, 1.0], np.float64)
    # Duplicate rows so the hard minimum-five-per-class contract is met.
    x = np.tile(x, (3, 1))
    y = np.tile(y, 3)
    first = _fit_logistic(
        x, y, iterations=200, learning_rate=0.05, decay_steps=200.0, l2=0.002
    )
    second = _fit_logistic(
        x, y, iterations=200, learning_rate=0.05, decay_steps=200.0, l2=0.002
    )
    assert np.array_equal(first[0], second[0])
    assert first[1] == second[1]
    probability = _sigmoid(x @ first[0] + first[1])
    assert _auc(y, probability) == 1.0


def test_gate_conditions_are_all_fail_closed():
    contract = {
        "min_delta_ap15": 0.0,
        "min_delta_ap25": 0.0,
        "min_delta_ap50": 0.0025,
        "min_replacements": 10,
        "min_scenes": 5,
        "min_positive_gain_fraction": 0.6,
        "max_severe_harm_fraction": 0.1,
        "max_target_switch_fraction": 0.1,
    }
    failed = _gate_conditions(
        delta={"iou_0.15": 0.0, "iou_0.25": 0.0, "iou_0.50": 0.0024},
        replacements=10,
        scenes=5,
        positive_fraction=0.6,
        severe_harm_fraction=0.1,
        target_switch_fraction=0.1,
        contract=contract,
    )
    assert failed["delta_ap50"] is False
    assert failed["pass"] is False

    passed = _gate_conditions(
        delta={"iou_0.15": 0.0, "iou_0.25": 0.0, "iou_0.50": 0.0025},
        replacements=10,
        scenes=5,
        positive_fraction=0.6,
        severe_harm_fraction=0.1,
        target_switch_fraction=0.1,
        contract=contract,
    )
    assert passed["pass"] is True


@pytest.mark.parametrize(
    "writer,payload",
    (
        (write_dataset_npz_create_only, {"value": np.asarray([1], np.int64)}),
        (write_training_npz_create_only, {"value": np.asarray([1], np.int64)}),
        (write_dataset_json_create_only, {"value": 1}),
        (write_training_json_create_only, {"value": 1}),
    ),
)
def test_create_only_writers_never_replace_existing_bytes(
    tmp_path: Path, writer, payload
):
    target = tmp_path / "existing.artifact"
    original = b"preexisting-user-data\n"
    target.write_bytes(original)
    with pytest.raises(FileExistsError):
        writer(target, payload)
    assert target.read_bytes() == original
