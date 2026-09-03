from __future__ import annotations

import numpy as np

from tools.materialize_tr3d_c3_active import (
    _append_payload,
    _assign_candidate_scores,
)


def test_candidate_scores_preserve_global_c1_order_below_anchors():
    entries = [(1, 0, 0.8), (0, 0, 0.2), (0, 1, 0.8), (1, 1, 0.1)]
    result = _assign_candidate_scores(entries, 0.4)
    ordered = sorted(entries, key=lambda item: (-item[2], item[0], item[1]))
    scores = [result[(scene, local)] for scene, local, _ in ordered]
    assert all(0.0 < value < 0.4 for value in scores)
    assert all(left > right for left, right in zip(scores, scores[1:]))
    assert scores[0] == result[(0, 1)]


def test_append_keeps_original_rows_and_creates_canonical_candidates():
    anchor_geometry = np.arange(24, dtype=np.float32).reshape(8, 3)
    source = [[(0, anchor_geometry, 0.7)]]
    candidates = np.stack((anchor_geometry + 1, anchor_geometry + 2))
    output = _append_payload(source, candidates, [0.2, 0.1])
    assert output[0][0] is source[0][0]
    assert len(output[0]) == 3
    for index, row in enumerate(output[0][1:]):
        assert type(row) is tuple
        assert row[0] == 0
        assert row[1].dtype == np.float32
        assert row[1].flags.c_contiguous
        assert np.array_equal(row[1], candidates[index])
        assert type(row[2]) is float
