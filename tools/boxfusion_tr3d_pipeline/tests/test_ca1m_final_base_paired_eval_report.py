from __future__ import annotations

import json
from pathlib import Path
import stat
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from report_ca1m_final_base_paired_eval import (  # noqa: E402
    parse_eval_log,
    write_json_create_only,
)


def _prediction_tree(tmp_path: Path) -> tuple[Path, list[str]]:
    root = tmp_path / "predictions"
    root.mkdir()
    scenes = [f"{42_000_000 + index:08d}" for index in range(10)]
    for scene in scenes:
        (root / f"{scene}_boxes.pkl").write_bytes(b"prediction")
    return root, scenes


def _log(root: Path, scenes: list[str]) -> str:
    rows = ["Num scenes in test dataset: 10"]
    for index, scene in enumerate(scenes):
        rows.extend(
            (
                f"Eval batch: {index} scan_idx {scene}",
                f"pred_path {root / f'{scene}_boxes.pkl'}",
                f"pred_labels Counter({{0: {index + 1}}})",
            )
        )
    for iou, values in (
        ("0.150000", ("0.35", "0.75", "0.39")),
        ("0.250000", ("0.30", "0.67", "0.35")),
        ("0.500000", ("0.13", "0.39", "0.20")),
    ):
        rows.append(f"---------- iou_thresh: {iou} ----------")
        rows.extend(f"eval {name}: {value}" for name, value in zip(("mAP", "APrec", "ARecall"), values, strict=True))
    return "\n".join(rows) + "\n"


def test_parse_eval_log_requires_exact10_and_three_metric_triplets(tmp_path: Path) -> None:
    root, scenes = _prediction_tree(tmp_path)
    path = tmp_path / "eval.log"
    path.write_text(_log(root, scenes), encoding="utf-8")
    parsed = parse_eval_log(path)
    assert parsed["scenes"] == scenes
    assert parsed["prediction_rows"] == 55
    assert parsed["metrics"]["AP25"] == {"mAP": 0.30, "APrec": 0.67, "ARecall": 0.35}


def test_parse_eval_log_rejects_missing_batch_and_metric(tmp_path: Path) -> None:
    root, scenes = _prediction_tree(tmp_path)
    text = _log(root, scenes)
    path = tmp_path / "bad.log"
    path.write_text(text.replace("Eval batch: 9 scan_idx", "Not a batch: 9 scan_idx"), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 10 Eval batch"):
        parse_eval_log(path)
    path.write_text(text.replace("eval ARecall: 0.20", "missing ARecall: 0.20"), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 3 metric triplets"):
        parse_eval_log(path)


def test_report_writer_is_create_only_and_read_only(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    write_json_create_only(path, {"complete": True})
    assert json.loads(path.read_text()) == {"complete": True}
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_json_create_only(path, {"complete": False})
