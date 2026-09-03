from __future__ import annotations

import pytest

from tools.verify_tr3d_r3_standard_eval import parse_eval_stdout, verify


def _paired() -> dict:
    metrics = {}
    for threshold, offset in (("0.15", 0.0), ("0.25", 0.1), ("0.50", 0.2)):
        metrics[threshold] = {
            "exact": True,
            "active": {
                "average_precision": 0.4 + offset + 4e-8,
                "final_precision": 0.5 + offset + 4e-8,
                "final_recall": 0.6 + offset + 4e-8,
            },
        }
    return {
        "schema": "boxfusion.tr3d_r3_shadow_active_paired_audit.v1",
        "ok": True,
        "shadow_only": True,
        "formal_active_authorized": False,
        "metrics": metrics,
    }


def _stdout(*, bad_last: bool = False) -> str:
    rows = []
    for index in range(3):
        rows.extend(
            [
                f"eval mAP: {0.4 + 0.1 * index:.6f}",
                f"eval APrec: {0.5 + 0.1 * index:.6f}",
                f"eval ARecall: {0.6 + 0.1 * index + (0.01 if bad_last and index == 2 else 0):.6f}",
            ]
        )
    return "\n".join(rows) + "\n"


def test_parse_requires_exact_three_rows_per_metric() -> None:
    parsed = parse_eval_stdout(_stdout())
    assert parsed["mAP"] == [0.4, 0.5, 0.6]
    with pytest.raises(ValueError, match="exactly three"):
        parse_eval_stdout("eval mAP: 0.1\n")


def test_verify_accepts_six_decimal_equivalence() -> None:
    result = verify(_stdout(), _paired())
    assert result["ok"]
    assert not result["formal_active_authorized"]


def test_verify_reports_mismatch() -> None:
    result = verify(_stdout(bad_last=True), _paired())
    assert not result["ok"]
    assert "ARecall" in result["issues"][0]


def test_verify_rejects_unpassed_paired_report() -> None:
    paired = _paired()
    paired["ok"] = False
    with pytest.raises(ValueError, match="did not pass"):
        verify(_stdout(), paired)
