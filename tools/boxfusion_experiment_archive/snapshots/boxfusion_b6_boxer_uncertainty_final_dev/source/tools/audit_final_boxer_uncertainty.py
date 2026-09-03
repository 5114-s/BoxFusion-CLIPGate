#!/usr/bin/env python3
"""Audit the protected-field contract of final-only Boxer uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


SCHEMA = "boxfusion.final_boxer_uncertainty.scene.v1"


def validate_payload(
    payload: Mapping[str, Any], *, expected_mode: Optional[str] = None
) -> List[str]:
    issues: List[str] = []
    if payload.get("schema") != SCHEMA:
        issues.append("schema mismatch")
    config = payload.get("config")
    summary = payload.get("summary")
    contract = payload.get("contract")
    records = payload.get("records")
    if not isinstance(config, Mapping):
        return issues + ["config is missing or not a mapping"]
    if not isinstance(summary, Mapping):
        return issues + ["summary is missing or not a mapping"]
    if not isinstance(contract, Mapping):
        return issues + ["contract is missing or not a mapping"]
    if not isinstance(records, list):
        return issues + ["records is missing or not a list"]

    mode = config.get("mode")
    if mode not in {"observer", "active"}:
        issues.append(f"invalid diagnostic mode: {mode!r}")
    if expected_mode is not None and mode != expected_mode:
        issues.append(
            f"mode mismatch: expected {expected_mode!r}, got {mode!r}"
        )
    if not contract.get("protected_fields_equal", False):
        issues.append("protected_fields_equal is false")
    if contract.get("scene_fallback", False):
        issues.append("scene-level fallback was triggered")
    if contract.get("count_before") != contract.get("count_after"):
        issues.append("prediction count changed")
    for field in (
        "scores_sha256",
        "source_indices_sha256",
        "stable_ids_sha256",
    ):
        if contract.get(f"{field}_before") != contract.get(
            f"{field}_after"
        ):
            issues.append(f"{field} changed")
    if int(summary.get("selection_changed_rows", -1)) != 0:
        issues.append("Top-K selection changed")
    if int(summary.get("ranking_changed_rows", -1)) != 0:
        issues.append("Top-K ranking changed")
    if mode == "observer":
        if int(summary.get("applied_rows", -1)) != 0:
            issues.append("observer applied geometry")
        if contract.get("baseline_corners_sha256") != contract.get(
            "output_corners_sha256"
        ):
            issues.append("observer changed corners")

    applied_records = 0
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            issues.append(f"record {index} is not a mapping")
            continue
        if record.get("selection_changed", False):
            issues.append(f"record {index} changed Top-K selection")
        if record.get("ranking_changed", False):
            issues.append(f"record {index} changed Top-K ranking")
        if record.get("applied", False):
            applied_records += 1
            if mode != "active":
                issues.append(f"record {index} applied outside active mode")
            if "baseline_corners" not in record:
                issues.append(f"record {index} lacks baseline corners")
            if "candidate_corners" not in record:
                issues.append(f"record {index} lacks candidate corners")
    if applied_records != int(summary.get("applied_rows", -1)):
        issues.append("applied record count disagrees with summary")
    return issues


def audit_directory(
    diagnostics_root: Path,
    *,
    expected_mode: Optional[str] = None,
    expected_scenes: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    paths = sorted(
        diagnostics_root.glob("*_final_boxer_uncertainty.json")
    )
    scene_reports: Dict[str, Any] = {}
    aggregate = {
        "files": len(paths),
        "rows": 0,
        "matched": 0,
        "weight_changed": 0,
        "optimized": 0,
        "applied": 0,
    }
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        scene = str(payload.get("scene_id", path.stem))
        issues = validate_payload(payload, expected_mode=expected_mode)
        summary = payload.get("summary", {})
        scene_reports[scene] = {
            "path": str(path),
            "ok": not issues,
            "issues": issues,
            "summary": summary,
        }
        aggregate["rows"] += int(summary.get("output_rows", 0))
        aggregate["matched"] += int(summary.get("matched_rows", 0))
        aggregate["weight_changed"] += int(
            summary.get("weight_changed_rows", 0)
        )
        aggregate["optimized"] += int(summary.get("optimized_rows", 0))
        aggregate["applied"] += int(summary.get("applied_rows", 0))

    missing: List[str] = []
    unexpected: List[str] = []
    if expected_scenes is not None:
        expected = set(expected_scenes)
        observed = set(scene_reports)
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
    ok = (
        bool(paths)
        and not missing
        and not unexpected
        and all(report["ok"] for report in scene_reports.values())
    )
    return {
        "schema": "boxfusion.final_boxer_uncertainty.audit.v1",
        "diagnostics_root": str(diagnostics_root),
        "expected_mode": expected_mode,
        "ok": ok,
        "missing_scenes": missing,
        "unexpected_scenes": unexpected,
        "aggregate": aggregate,
        "scenes": scene_reports,
    }


def _scene_list(path: Optional[Path]) -> Optional[List[str]]:
    if path is None:
        return None
    scenes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            scenes.append(value.split()[0])
    return scenes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--expected-mode", choices=("observer", "active"))
    parser.add_argument("--scene-list", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_directory(
        args.diagnostics_root,
        expected_mode=args.expected_mode,
        expected_scenes=_scene_list(args.scene_list),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
