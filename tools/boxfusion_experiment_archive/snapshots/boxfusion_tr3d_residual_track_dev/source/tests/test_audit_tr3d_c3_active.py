from __future__ import annotations

import copy

import numpy as np

from tools.audit_tr3d_c3_active import _audit_payload, _expected_rows


_SIGNS = np.asarray(
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


def _corners(center: float) -> np.ndarray:
    return np.ascontiguousarray(_SIGNS * 0.5 + [center, 0, 0])


def test_append_identity_and_global_c1_score_order() -> None:
    source0 = [[(0, _corners(0), 0.9), (0, _corners(3), 0.7)]]
    source1 = [[(0, _corners(6), 0.8)]]
    scenes = [
        {
            "source": source0,
            # Physical order remains the C2 source order.  The second row has
            # the larger C1 score and therefore the larger output score.
            "candidates": [
                {"corners": _corners(10), "c1_track_score": 0.2, "score": 0.1},
                {"corners": _corners(11), "c1_track_score": 0.9, "score": 0.3},
            ],
        },
        {
            "source": source1,
            "candidates": [
                {"corners": _corners(12), "c1_track_score": 0.5, "score": 0.2}
            ],
        },
    ]
    expected, anchor_floor, order_exact = _expected_rows(scenes)
    assert anchor_floor == 0.7
    assert order_exact is True
    output0 = [[
        *source0[0],
        (0, _corners(10), 0.1),
        (0, _corners(11), 0.3),
    ]]
    output1 = [[*source1[0], (0, _corners(12), 0.2)]]
    assert _audit_payload(source0, output0, expected[0])["ok"] is True
    assert _audit_payload(source1, output1, expected[1])["ok"] is True

    changed_anchor = copy.deepcopy(output0)
    changed_anchor[0][0] = (0, changed_anchor[0][0][1], np.float32(0.9))
    report = _audit_payload(source0, changed_anchor, expected[0])
    assert report["ok"] is False
    assert "anchor row 0 differs in type/dtype/bytes" in report["issues"]

    changed_candidate = copy.deepcopy(output0)
    changed_candidate[0][2] = (1, changed_candidate[0][2][1], 0.1)
    report = _audit_payload(source0, changed_candidate, expected[0])
    assert report["ok"] is False
    assert "candidate row 0 label is not Python int zero" in report["issues"]

