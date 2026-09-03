import importlib.util
import sys
from pathlib import Path

import numpy as np


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "evaluate_sgcdet_candidate_oracle.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_sgcdet_candidate_oracle", SOURCE
)
oracle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = oracle
SPEC.loader.exec_module(oracle)


def test_reconstruct_candidates_matches_local_residual_equations():
    local = np.asarray([[0.0, 0.0, 0.0, 2.0, 4.0, 6.0]])
    center = np.asarray([[0.2, -0.4, 0.6]])
    log_dims = np.log(np.asarray([[1.1, 0.8, 1.25]]))
    result = oracle.reconstruct_candidates(
        local,
        center,
        log_dims,
        np.zeros((1, 3)),
        np.eye(3)[None, ...],
        np.asarray([True]),
        max_center_fraction=0.15,
        max_log_dimension_residual=np.log(1.25),
        minimum_dimension=0.001,
    )
    expected_center = center[0]
    expected_dims = local[0, 3:6] * np.asarray([1.1, 0.8, 1.25])
    expected = (
        expected_center[None, :]
        + oracle.SIGNS * expected_dims[None, :] * 0.5
    )
    np.testing.assert_allclose(result[0], expected, atol=1e-6)


def test_reconstruct_candidates_keeps_invalid_rows_as_nan():
    result = oracle.reconstruct_candidates(
        np.full((1, 6), np.nan),
        np.full((1, 3), np.nan),
        np.full((1, 3), np.nan),
        np.full((1, 3), np.nan),
        np.full((1, 3, 3), np.nan),
        np.asarray([False]),
        max_center_fraction=0.15,
        max_log_dimension_residual=np.log(1.25),
        minimum_dimension=0.001,
    )
    assert np.isnan(result).all()


def test_official_metrics_use_strict_threshold_and_duplicate_semantics():
    # Row zero is exactly on the threshold and is therefore an FP. Rows one
    # and two both overlap GT zero, so only the higher-scored row one is a TP.
    iou = {"scene": np.asarray([[0.5], [0.9], [0.8]])}
    scores = {"scene": np.asarray([0.99, 0.8, 0.7])}
    result = oracle._official_threshold_metrics(
        iou, scores, ("scene",), 1, 0.5
    )
    assert result["true_positives"] == 1
    assert result["false_positives"] == 2
    assert np.isclose(result["ap"], 0.5, atol=1e-6)


def test_official_metrics_never_fall_back_to_second_best_gt():
    # The first prediction consumes GT0. The second prediction's best overlap
    # is still GT0, so the official evaluator marks it FP even though GT1 is
    # also above threshold.
    iou = {
        "scene": np.asarray(
            [
                [0.9, 0.0],
                [0.8, 0.7],
            ]
        )
    }
    scores = {"scene": np.asarray([0.9, 0.8])}
    result = oracle._official_threshold_metrics(
        iou, scores, ("scene",), 2, 0.5
    )
    assert result["true_positives"] == 1
    assert result["false_positives"] == 1

