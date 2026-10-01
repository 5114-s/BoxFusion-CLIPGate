import importlib.util
import pickle
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "tools" / "analyze_fused_oracle.py"
SPEC = importlib.util.spec_from_file_location("analyze_fused_oracle", SOURCE)
oracle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = oracle
SPEC.loader.exec_module(oracle)


def corners(center, dims):
    center = np.asarray(center, dtype=np.float64)
    half = np.asarray(dims, dtype=np.float64) * 0.5
    offsets = np.asarray(
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
        dtype=np.float64,
    )
    return center + offsets * half


def test_pairwise_aabb_iou_exact():
    pred = np.asarray([[0, 0, 0, 2, 2, 2]], dtype=np.float64)
    gt = np.asarray(
        [[1, 0, 0, 3, 2, 2], [3, 3, 3, 4, 4, 4]],
        dtype=np.float64,
    )
    result = oracle.pairwise_aabb_iou(pred, gt)
    np.testing.assert_allclose(result, [[1.0 / 3.0, 0.0]])


def test_maximum_matching_does_not_count_duplicate_predictions():
    iou = np.asarray([[0.9], [0.8], [0.1]])
    pred, gt = oracle.maximum_matches(iou, 0.5)
    assert pred.shape == (1,)
    assert gt.tolist() == [0]


def test_oracle_ranking_places_true_positives_before_false_positives():
    iou = np.asarray([[0.0], [0.9]], dtype=np.float64)
    scores = np.asarray([0.99, 0.50])
    real, ranked, matched = oracle.score_scene(iou, scores, 0.5)
    real_ap, _, _ = oracle.ranked_metrics(real, ground_truth_count=1)
    oracle_ap, _, _ = oracle.ranked_metrics(ranked, ground_truth_count=1)
    assert matched == 1
    assert real_ap == 0.5
    assert oracle_ap == 1.0


def test_end_to_end_single_scene_report(tmp_path):
    pred_root = tmp_path / "pred"
    gt_root = tmp_path / "gt"
    scan_root = tmp_path / "scans"
    pred_root.mkdir()
    gt_root.mkdir()
    scene = "scene0000_00"
    (scan_root / scene).mkdir(parents=True)
    (scan_root / scene / f"{scene}.txt").write_text(
        "axisAlignment = "
        + " ".join(str(float(value)) for value in np.eye(4).reshape(-1))
        + "\n"
    )
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n")

    gt = np.asarray(
        [
            [0, 0, 0, 2, 2, 2, 3],
            [5, 0, 0, 2, 2, 2, 4],
        ],
        dtype=np.float32,
    )
    np.save(gt_root / f"{scene}_bbox.npy", gt)
    detections = [
        (0, corners([20, 0, 0], [2, 2, 2]), 0.99),
        (0, corners([0, 0, 0], [2, 2, 2]), 0.80),
        (0, corners([5, 0, 0], [2, 2, 2]), 0.70),
    ]
    with (pred_root / f"{scene}_boxes.pkl").open("wb") as handle:
        pickle.dump([detections], handle)

    report = oracle.analyze(
        Namespace(
            pred_root=pred_root,
            gt_root=gt_root,
            scan_root=scan_root,
            scene_list=scene_list,
            thresholds=(0.5,),
            constant_score=False,
        )
    )
    values = report["reports"][0]
    assert report["scene_count"] == 1
    assert values["matched"] == 2
    assert values["recall"] == 1.0
    assert values["oracle_rank_ap"] == 1.0
    assert values["average_precision"] < values["oracle_rank_ap"]
