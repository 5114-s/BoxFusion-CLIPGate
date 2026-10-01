from __future__ import annotations

import numpy as np

from tools.evaluate_ca1m_tr3d_train_probe import _metrics


SIGNS = np.asarray(
    [
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1],
    ],
    dtype=np.float32,
)


def box(center):
    return np.asarray(center, dtype=np.float32) + 0.5 * SIGNS


def test_train_probe_metrics_reproduce_global_score_order_and_duplicate_rule():
    gt = {"42445670": np.stack((box([0, 0, 0]),))}
    corners = np.stack((box([5, 0, 0]), box([0, 0, 0])))
    scores = np.asarray([0.9, 0.8], dtype=np.float32)
    baseline = _metrics({"42445670": (corners, scores)}, gt)
    assert baseline["iou_0.50"]["tp"] == 1
    assert baseline["iou_0.50"]["fp"] == 1
    assert np.isclose(baseline["iou_0.50"]["ap"], 0.5, atol=1e-6)

    oracle_corners = np.stack((box([0, 0, 0]), box([0, 0, 0])))
    oracle = _metrics({"42445670": (oracle_corners, scores)}, gt)
    assert oracle["iou_0.50"]["tp"] == 1
    assert oracle["iou_0.50"]["fp"] == 1
    assert np.isclose(oracle["iou_0.50"]["ap"], 1.0, atol=1e-6)


def test_train_probe_metrics_include_zero_prediction_gt_scenes_in_recall():
    gt = {
        "42445670": np.stack((box([0, 0, 0]),)),
        "42446607": np.stack((box([0, 0, 0]),)),
    }
    predictions = {
        "42445670": (
            np.stack((box([0, 0, 0]),)),
            np.asarray([0.9], dtype=np.float32),
        ),
        "42446607": (
            np.empty((0, 8, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        ),
    }
    result = _metrics(predictions, gt)
    assert result["iou_0.50"]["tp"] == 1
    assert result["iou_0.50"]["fn"] == 1
    assert np.isclose(result["iou_0.50"]["recall"], 0.5, atol=1e-6)
