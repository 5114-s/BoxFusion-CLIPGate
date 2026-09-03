#!/usr/bin/env python3
"""Verify that the official ScanNet evaluator reproduces the paired R3 audit.

The paired auditor computes metrics directly from the materialized prediction
pickles.  This final check parses the ordinary ``eval_scannet.py`` stdout and
requires every printed mAP/APrec/ARecall value to equal the same metric rounded
to the evaluator's six decimal places.  It is an engineering-equivalence check,
not authorization for a formal active method.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence


SCHEMA = "boxfusion.tr3d_r3_standard_eval_equivalence.v1"
PAIRED_SCHEMA = "boxfusion.tr3d_r3_shadow_active_paired_audit.v1"
THRESHOLDS = ("0.15", "0.25", "0.50")
_METRIC_RE = re.compile(r"^eval (mAP|APrec|ARecall): ([0-9]+(?:\.[0-9]+)?)$")
_PAIRED_KEYS = {
    "mAP": "average_precision",
    "APrec": "final_precision",
    "ARecall": "final_recall",
}


def parse_eval_stdout(text: str) -> dict[str, list[float]]:
    result = {name: [] for name in _PAIRED_KEYS}
    for raw_line in text.splitlines():
        match = _METRIC_RE.fullmatch(raw_line.strip())
        if match is not None:
            result[match.group(1)].append(float(match.group(2)))
    for name, values in result.items():
        if len(values) != len(THRESHOLDS):
            raise ValueError(
                f"official evaluator must print exactly three {name} rows; "
                f"found {len(values)}"
            )
    return result


def verify(eval_stdout: str, paired: Mapping[str, Any]) -> dict[str, Any]:
    if paired.get("schema") != PAIRED_SCHEMA or not paired.get("ok"):
        raise ValueError("paired shadow-active audit is absent or did not pass")
    if not paired.get("shadow_only") or paired.get("formal_active_authorized"):
        raise ValueError("paired report violates the shadow-only contract")
    observed = parse_eval_stdout(eval_stdout)
    expected: dict[str, list[float]] = {name: [] for name in _PAIRED_KEYS}
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    for index, threshold in enumerate(THRESHOLDS):
        metric_row = paired.get("metrics", {}).get(threshold)
        if not isinstance(metric_row, Mapping) or not metric_row.get("exact"):
            raise ValueError(f"paired audit lacks exact metrics for IoU {threshold}")
        active = metric_row.get("active")
        if not isinstance(active, Mapping):
            raise ValueError(f"paired audit active metrics are malformed for IoU {threshold}")
        row: dict[str, Any] = {"iou_threshold": float(threshold)}
        for printed_name, paired_name in _PAIRED_KEYS.items():
            # eval_scannet.py uses Python's six-place fixed formatting.
            wanted = float(f"{float(active[paired_name]):.6f}")
            actual = observed[printed_name][index]
            expected[printed_name].append(wanted)
            exact = actual == wanted
            row[printed_name] = {
                "observed": actual,
                "paired_expected_rounded6": wanted,
                "exact": exact,
            }
            if not exact:
                issues.append(
                    f"IoU {threshold} {printed_name}: observed {actual:.6f}, "
                    f"expected {wanted:.6f}"
                )
        rows.append(row)
    return {
        "schema": SCHEMA,
        "ok": not issues,
        "shadow_only": True,
        "formal_active_authorized": False,
        "issues": issues,
        "rows": rows,
        "observed": observed,
        "expected_rounded6": expected,
    }


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"immutable standard-eval report exists: {path}") from error
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-log", type=Path, required=True)
    parser.add_argument("--paired-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paired = json.loads(args.paired_report.read_text(encoding="utf-8"))
    result = verify(args.eval_log.read_text(encoding="utf-8"), paired)
    _write_create_only(args.report.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
