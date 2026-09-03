#!/usr/bin/env python3
"""Bind paired official ScanNet logs to an online C3 shadow manifest."""

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
from tools.audit_tr3d_c3_online_shadow import SCHEMA as AUDIT_SCHEMA  # noqa: E402
from tools.materialize_tr3d_c3_active import _write_json_create_only  # noqa: E402
from tools.materialize_tr3d_c3_online_shadow import SCHEMA as MANIFEST_SCHEMA  # noqa: E402


SCHEMA = "boxfusion.tr3d_c3_online_shadow_eval.v1"
_METRIC_RE = re.compile(r"^eval (mAP|APrec|ARecall): ([0-9]+(?:\.[0-9]+)?)$")


def _parse_log(path: Path) -> dict[str, dict[str, float]]:
    rows: list[tuple[str, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _METRIC_RE.fullmatch(line.strip())
        if match:
            rows.append((match.group(1), float(match.group(2))))
    expected = [name for _ in range(3) for name in ("mAP", "APrec", "ARecall")]
    if [name for name, _ in rows] != expected:
        raise ValueError(f"{path}: expected exactly three official metric triplets")
    result: dict[str, dict[str, float]] = {}
    for index, threshold in enumerate(("AP15", "AP25", "AP50")):
        triplet = rows[index * 3 : index * 3 + 3]
        result[threshold] = {name: value for name, value in triplet}
    return result


def compare(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path, audit_path = args.manifest.resolve(), args.audit.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA or not manifest.get("complete"):
        raise ValueError("invalid online shadow manifest")
    if (
        audit.get("schema") != AUDIT_SCHEMA
        or not audit.get("complete")
        or not audit.get("ok")
        or audit.get("manifest_sha256") != sha256_file(manifest_path)
    ):
        raise ValueError("invalid online shadow identity audit")
    baseline_log, shadow_log = args.baseline_log.resolve(), args.shadow_log.resolve()
    baseline, shadow = _parse_log(baseline_log), _parse_log(shadow_log)
    delta = {
        threshold: {
            name: shadow[threshold][name] - baseline[threshold][name]
            for name in ("mAP", "APrec", "ARecall")
        }
        for threshold in ("AP15", "AP25", "AP50")
    }
    report = {
        "schema": SCHEMA,
        "complete": True,
        "paired_official_evaluation": True,
        "fixed10_is_diagnostic_only": int(manifest["scene_count"]) == 10,
        "formal_active_authorized": False,
        "live_mutation_authorized": False,
        "ground_truth_access": True,
        "ground_truth_access_scope": "official_evaluator_and_this_report_only",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "identity_audit": str(audit_path),
        "identity_audit_sha256": sha256_file(audit_path),
        "scene_count": int(manifest["scene_count"]),
        "scene_list": manifest["scene_list"],
        "scene_list_sha256": manifest["scene_list_sha256"],
        "candidate_count": int(manifest["candidate_count"]),
        "anchor_prediction_tree_sha256": manifest["anchor_tree_before"]["tree_sha256"],
        "shadow_prediction_tree_sha256": manifest["output_tree"]["tree_sha256"],
        "evaluator": str(args.evaluator.resolve()),
        "evaluator_sha256": sha256_file(args.evaluator.resolve()),
        "baseline_log": str(baseline_log),
        "baseline_log_sha256": sha256_file(baseline_log),
        "shadow_log": str(shadow_log),
        "shadow_log_sha256": sha256_file(shadow_log),
        "baseline": baseline,
        "shadow": shadow,
        "delta": delta,
    }
    _write_json_create_only(args.report.resolve(), report)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--audit", type=Path, required=True)
    value.add_argument("--baseline-log", type=Path, required=True)
    value.add_argument("--shadow-log", type=Path, required=True)
    value.add_argument("--evaluator", type=Path, required=True)
    value.add_argument("--report", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    report = compare(parser().parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
