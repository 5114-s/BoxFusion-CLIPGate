#!/usr/bin/env python3
"""Verify official ScanNet C3 AP against the frozen in-memory shadow result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.tr3d_c2_maskrgbd_cache import sha256_file  # noqa: E402
from tools.run_tr3d_c2_maskrgbd_observer import _write_json_create_only  # noqa: E402


SCHEMA = "boxfusion.tr3d_c3_standard_eval_verification.v1"
MATERIALIZE_SCHEMA = "boxfusion.tr3d_c3_shadow_active_manifest.v1"
SHADOW_SCHEMA = "boxfusion.tr3d_c3_shadow_counterfactual.v1"
AUDIT_SCHEMA = "boxfusion.tr3d_c3_active_identity_audit.v1"
THRESHOLDS = ("0.15", "0.25", "0.50")


def _parse_eval_log(text: str) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for name in ("mAP", "APrec", "ARecall"):
        values = [
            float(match.group(1))
            for match in re.finditer(
                rf"^eval {name}: ([0-9]+(?:\.[0-9]+)?)$", text, re.MULTILINE
            )
        ]
        if len(values) != 3:
            raise ValueError(f"official evaluator log must contain three {name} rows")
        result[name] = values
    return result


def verify(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.materialize_manifest.read_text(encoding="utf-8"))
    audit = json.loads(args.identity_audit.read_text(encoding="utf-8"))
    shadow = json.loads(args.shadow_report.read_text(encoding="utf-8"))
    if manifest.get("schema") != MATERIALIZE_SCHEMA or not manifest.get("complete"):
        raise ValueError("unsupported or incomplete C3 materialization manifest")
    if audit.get("schema") != AUDIT_SCHEMA or not audit.get("ok"):
        raise ValueError("C3 identity audit did not pass")
    if shadow.get("schema") != SHADOW_SCHEMA:
        raise ValueError("unsupported C3 shadow report")
    if (
        manifest.get("formal_active_authorized")
        or not manifest.get("shadow_only")
        or manifest.get("ground_truth_access")
    ):
        raise ValueError("C3 materialization safety contract changed")
    observed = _parse_eval_log(args.eval_log.read_text(encoding="utf-8"))
    route = shadow["partitions"]["all"]["routes"]["append_c1_track_rank"]
    expected = [
        float(route["metrics"][key]["average_precision"])
        for key in THRESHOLDS
    ]
    expected_rounded = [round(value, 6) for value in expected]
    exact = observed["mAP"] == expected_rounded
    if not exact:
        raise ValueError(
            f"official mAP {observed['mAP']} differs from shadow {expected_rounded}"
        )
    return {
        "schema": SCHEMA,
        "ok": True,
        "official_evaluator_equivalent_to_shadow": True,
        "thresholds": [float(value) for value in THRESHOLDS],
        "official": observed,
        "shadow_expected_mAP_full_precision": expected,
        "shadow_expected_mAP_rounded6": expected_rounded,
        "materialize_manifest": str(args.materialize_manifest.resolve()),
        "materialize_manifest_sha256": sha256_file(args.materialize_manifest),
        "identity_audit": str(args.identity_audit.resolve()),
        "identity_audit_sha256": sha256_file(args.identity_audit),
        "shadow_report": str(args.shadow_report.resolve()),
        "shadow_report_sha256": sha256_file(args.shadow_report),
        "eval_log": str(args.eval_log.resolve()),
        "eval_log_sha256": sha256_file(args.eval_log),
        "eval_script": str(args.eval_script.resolve()),
        "eval_script_sha256": sha256_file(args.eval_script),
        "formal_active_authorized": False,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--materialize-manifest", type=Path, required=True)
    value.add_argument("--identity-audit", type=Path, required=True)
    value.add_argument("--shadow-report", type=Path, required=True)
    value.add_argument("--eval-log", type=Path, required=True)
    value.add_argument("--eval-script", type=Path, required=True)
    value.add_argument("--report", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = verify(args)
    _write_json_create_only(args.report.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
