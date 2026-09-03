from __future__ import annotations

import numpy as np

from tools.audit_tr3d_c3_shadow_counterfactual import _partition_report


def test_shadow_routes_keep_anchors_first_and_expose_oracle_headroom():
    rows = [
        {
            "scene_id": "scene0000_00",
            "anchor_boxes": np.asarray(
                [[0, 0, 0, 1, 1, 1], [20, 0, 0, 21, 1, 1]],
                dtype=np.float64,
            ),
            "anchor_scores": np.asarray([0.9, 0.8], dtype=np.float64),
            # Put the false candidate first so fixed-score stable ordering is
            # intentionally worse than the C1-ranked counterfactual.
            "candidate_boxes": np.asarray(
                [[30, 0, 0, 31, 1, 1], [5, 0, 0, 6, 1, 1]],
                dtype=np.float64,
            ),
            "c1_track_scores": np.asarray([0.1, 0.9], dtype=np.float64),
            "gt": np.asarray(
                [[0, 0, 0, 1, 1, 1], [5, 0, 0, 6, 1, 1]],
                dtype=np.float64,
            ),
        }
    ]

    report = _partition_report(rows)
    routes = report["routes"]
    anchor = routes["anchor"]["metrics"]["0.50"]
    fixed = routes["append_fixed_low"]["metrics"]["0.50"]
    ranked = routes["append_c1_track_rank"]["metrics"]["0.50"]
    oracle = routes["gt_oracle_upper_bound"]["metrics"]["0.50"]

    assert anchor["predictions"] == 2
    assert fixed["predictions"] == 4
    assert ranked["average_precision"] > fixed["average_precision"]
    assert oracle["average_precision"] >= ranked["average_precision"]
    assert oracle["matched_tp"] == 2
    assert oracle["oracle_novel_candidate_tp"] == 1
    assert routes["append_fixed_low"]["delta_vs_anchor"]["0.50"][
        "delta_matched_tp"
    ] == 1


def test_empty_candidate_route_is_exact_anchor_identity():
    rows = [
        {
            "scene_id": "scene0000_00",
            "anchor_boxes": np.asarray(
                [[0, 0, 0, 1, 1, 1]], dtype=np.float64
            ),
            "anchor_scores": np.asarray([0.9], dtype=np.float64),
            "candidate_boxes": np.empty((0, 6), dtype=np.float64),
            "c1_track_scores": np.empty(0, dtype=np.float64),
            "gt": np.asarray([[0, 0, 0, 1, 1, 1]], dtype=np.float64),
        }
    ]

    routes = _partition_report(rows)["routes"]
    anchor = routes["anchor"]["metrics"]
    assert routes["append_fixed_low"]["metrics"] == anchor
    assert routes["append_c1_track_rank"]["metrics"] == anchor
    assert routes["gt_oracle_upper_bound"]["metrics"] != anchor
    for row in routes["gt_oracle_upper_bound"]["metrics"].values():
        assert row["oracle_novel_candidate_tp"] == 0

