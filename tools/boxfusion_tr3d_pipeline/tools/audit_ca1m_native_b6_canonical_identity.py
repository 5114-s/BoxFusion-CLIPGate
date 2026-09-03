#!/usr/bin/env python3
"""Aggregate same-run identity and feature audit for canonical103 observer data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from audit_ca1m_c3_native_b6_observer import (  # noqa: E402
    DIAGNOSTIC_SUFFIX,
    audit_diagnostic,
    audit_runtime_log,
    compare_same_run,
    exact_scene_files,
    read_jsonl,
    read_scenes,
    sha256,
    write_json_atomic,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--anchor-root", type=Path, required=True)
    parser.add_argument("--observer-root", type=Path, required=True)
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--boxer-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scenes = read_scenes(args.scene_list)
    if len(scenes) != 103:
        raise ValueError("canonical identity audit requires 103 scenes")
    anchors = exact_scene_files(args.anchor_root, scenes, "_boxes.pkl")
    observers = exact_scene_files(args.observer_root, scenes, "_boxes.pkl")
    diagnostics = exact_scene_files(args.diagnostics_root, scenes, DIAGNOSTIC_SUFFIX)
    boxer = exact_scene_files(args.boxer_root, scenes, "_boxer_lifting.jsonl")
    logs = exact_scene_files(args.log_root, scenes, ".log")

    rows = valid = projectable = 0
    observer_seconds = cost_seconds = frame_equivalent = 0.0
    per_scene = {}
    for scene in scenes:
        prediction, identity = compare_same_run(scene, anchors[scene], observers[scene])
        diagnostic = audit_diagnostic(scene, diagnostics[scene], prediction)
        runtime = audit_runtime_log(logs[scene])
        log_text = logs[scene].read_text(errors="replace")
        if "eval mAP:" in log_text:
            raise ValueError(f"{scene}: evaluation marker found in collection log")
        boxer_rows = read_jsonl(boxer[scene])
        if any(
            str(row.get("scene_id")) != scene
            or row.get("mode") != "active"
            or row.get("selective_gate_enabled") is not True
            for row in boxer_rows
        ):
            raise ValueError(f"{scene}: Selective Boxer G0 contract disagrees")
        rows += identity["rows"]
        valid += diagnostic["valid_evidence_rows"]
        projectable += diagnostic["projectable_rows"]
        observer_seconds += diagnostic["observer_seconds"]
        cost_seconds += runtime["cost_seconds"]
        frame_equivalent += runtime["frame_equivalent"]
        per_scene[scene] = {
            "identity": identity,
            "diagnostic": diagnostic,
            "runtime": runtime,
            "boxer_calls": len(boxer_rows),
            "boxer_sha256": sha256(boxer[scene]),
        }
    report = {
        "schema": "boxfusion.ca1m_native_b6_canonical103_identity_audit.v1",
        "ok": True,
        "dataset_split": "official_validation_canonical103",
        "scenes": 103,
        "scene_list_sha256": sha256(args.scene_list),
        "observer_only": True,
        "mutation_enabled": False,
        "same_run_byte_identity_scenes": 103,
        "prediction_rows": rows,
        "mapping_rows": rows,
        "mapping_coverage": 1.0,
        "valid_evidence_rows": valid,
        "valid_evidence_coverage": valid / rows if rows else 1.0,
        "projectable_rows": projectable,
        "observer_seconds": observer_seconds,
        "cache_assisted_frame_weighted_fps": (
            frame_equivalent / cost_seconds if cost_seconds else 0.0
        ),
        "ground_truth_access": False,
        "evaluation_invoked": False,
        "training_authorized": False,
        "per_scene": per_scene,
    }
    write_json_atomic(args.output, report)
    print(json.dumps({key: report[key] for key in (
        "ok", "scenes", "prediction_rows", "mapping_coverage",
        "valid_evidence_coverage", "ground_truth_access", "evaluation_invoked",
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
