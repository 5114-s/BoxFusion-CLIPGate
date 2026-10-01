#!/usr/bin/env python3
"""Audit the frozen Selective-Boxer G0 contract in combined runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_b6_selective_boxer import _load_diagnostics, _validate_row


SCHEMA = "boxfusion.g0_boxer_active.audit.v1"


def _scenes(path: Path) -> Tuple[str, ...]:
    values = tuple(
        line.split()[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.split() and not line.lstrip().startswith("#")
    )
    if not values or len(values) != len(set(values)):
        raise ValueError(f"scene list is empty or contains duplicates: {path}")
    return values


def _stage(value: str) -> Tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("stage must have the form NAME=PATH")
    return name.strip(), Path(raw_path).expanduser().resolve()


def audit(scene_list: Path, stages: List[Tuple[str, Path]]) -> Dict[str, object]:
    scenes = _scenes(scene_list)
    issues: List[Dict[str, object]] = []
    summaries: Dict[str, Dict[str, int]] = {}
    reference_keys: Dict[str, Tuple[Tuple[int, str], ...]] = {}

    for stage_name, root in stages:
        frame_count = proposal_count = eligible = applied = fallback = 0
        for scene in scenes:
            path = root / f"{scene}_boxer_lifting.jsonl"
            if not path.is_file():
                issues.append(
                    {
                        "stage": stage_name,
                        "scene": scene,
                        "kind": "missing_diagnostic",
                        "path": str(path),
                    }
                )
                continue
            try:
                rows = _load_diagnostics(path)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                issues.append(
                    {
                        "stage": stage_name,
                        "scene": scene,
                        "kind": "invalid_diagnostic",
                        "error": str(error),
                    }
                )
                continue

            keys = tuple(sorted(rows))
            if scene not in reference_keys:
                reference_keys[scene] = keys
            elif keys != reference_keys[scene]:
                issues.append(
                    {
                        "stage": stage_name,
                        "scene": scene,
                        "kind": "schedule_mismatch",
                        "expected": reference_keys[scene],
                        "actual": keys,
                    }
                )

            frame_count += len(rows)
            for (frame_id, attempt_id), row in rows.items():
                context = {
                    "stage": stage_name,
                    "scene": scene,
                    "frame_id": frame_id,
                    "attempt_id": attempt_id,
                }
                _validate_row(
                    row,
                    role="active",
                    context=context,
                    issues=issues,
                )
                proposal_count += int(row.get("count", 0))
                eligible += int(row.get("eligible_count", 0))
                applied += int(row.get("applied_count", 0))
                fallback += int(row.get("fallback_count", 0))

        summaries[stage_name] = {
            "scenes": len(scenes),
            "frames": frame_count,
            "proposals": proposal_count,
            "eligible": eligible,
            "applied": applied,
            "fallback": fallback,
        }

    return {
        "schema": SCHEMA,
        "ok": not issues,
        "scene_list": str(scene_list.resolve()),
        "stages": summaries,
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument(
        "--stage",
        type=_stage,
        action="append",
        required=True,
        help="Combined stage and Boxer diagnostics root as NAME=PATH",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    report = audit(args.scene_list, args.stage)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
