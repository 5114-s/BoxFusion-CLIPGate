import json

import numpy as np

from tools.audit_boxer_lifting_contract import (
    compare_with_array_atol,
    load_diagnostics,
)
from tools.summarize_boxer_lifting_ablation import load_runtime_rows


def _write_rows(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_attempt_id_makes_first_frame_retry_unambiguous(tmp_path):
    path = tmp_path / "scene0000_00_boxer_lifting.jsonl"
    _write_rows(
        path,
        [
            {
                "schema": "boxfusion.boxer_lifting.frame.v1",
                "frame_id": 0,
                "attempt_id": "primary",
            },
            {
                "schema": "boxfusion.boxer_lifting.frame.v1",
                "frame_id": 0,
                "attempt_id": "retry",
            },
        ],
    )
    rows = load_diagnostics(path)
    assert set(rows) == {(0, "primary"), (0, "retry")}


def test_runtime_summary_reads_only_requested_scenes(tmp_path):
    for scene, runtime in (("scene0000_00", 10.0), ("scene0001_00", 99.0)):
        _write_rows(
            tmp_path / f"{scene}_boxer_lifting.jsonl",
            [
                {
                    "count": 2,
                    "runtime_ms": runtime,
                },
                {
                    "count": 0,
                },
            ],
        )
    runtimes, proposals, observed_calls, forward_calls = load_runtime_rows(
        tmp_path,
        ["scene0000_00"],
    )
    assert runtimes == [10.0]
    assert proposals == 2
    assert observed_calls == 2
    assert forward_calls == 1


def test_identity_tolerance_applies_only_to_numeric_arrays():
    baseline = [[(0, np.zeros((8, 3), dtype=np.float32), 0.75)]]
    observer = [[
        (0, np.full((8, 3), 4e-5, dtype=np.float32), 0.75)
    ]]
    issue, maximum = compare_with_array_atol(
        baseline, observer, array_atol=1e-4
    )
    assert issue is None
    assert 3.9e-5 < maximum < 4.1e-5

    changed_score = [[
        (0, np.full((8, 3), 4e-5, dtype=np.float32), 0.74)
    ]]
    issue, _ = compare_with_array_atol(
        baseline, changed_score, array_atol=1e-4
    )
    assert issue is not None
    assert "value mismatch" in issue["message"]
