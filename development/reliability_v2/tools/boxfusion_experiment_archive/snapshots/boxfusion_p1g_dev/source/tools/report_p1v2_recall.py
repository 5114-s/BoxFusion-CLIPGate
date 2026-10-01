#!/usr/bin/env python3
"""Report frozen-B6 proposal recall for the P1R/P1S observer stages."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.report_p1_residual_recall import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    evaluate,
    read_scene_ids,
)


SCHEMA = "boxfusion.p1v2.recall_report.v1"
_CONTRACT = {
    "P1R": {
        "head_architecture": "per_voxel_mlp",
        "target_assignment_scope": "snapshot_inside_only",
        "reference_stage": "P1",
        "minimum_extra_tp50": 2,
    },
    "P1S": {
        "head_architecture": "native_sparse_context_v1",
        "target_assignment_scope": "snapshot_inside_only",
        "reference_stage": "P1R",
        "minimum_extra_tp50": 1,
    },
}


def _threshold(report: Mapping[str, Any], value: str) -> Mapping[str, Any]:
    thresholds = report.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("reference report lacks thresholds")
    row = thresholds.get(value)
    if not isinstance(row, Mapping):
        raise ValueError(f"reference report lacks threshold {value}")
    return row


def _load_reference(path: Path, *, expected_stage: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("reference report must contain a mapping")
    observed_stage = payload.get("stage")
    # Historical P1 reports did not carry an explicit stage.
    if expected_stage == "P1":
        if observed_stage not in (None, "P1"):
            raise ValueError("P1R reference report must be P1")
    elif observed_stage != expected_stage:
        raise ValueError(
            f"reference stage {observed_stage!r}, expected {expected_stage}"
        )
    return dict(payload)


def build_report(
    *,
    stage: str,
    scene_list: Path,
    prediction_root: Path,
    diagnostics_root: Path,
    gt_root: Path,
    scans_root: Path,
    reference_report: Path,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    maximum_runtime_seconds_per_scene: float = 0.80,
    maximum_candidates_per_scene: float = 256.0,
) -> dict[str, Any]:
    stage = str(stage).strip().upper()
    if stage not in _CONTRACT:
        raise ValueError("stage must be P1R or P1S")
    if maximum_runtime_seconds_per_scene <= 0.0:
        raise ValueError("maximum runtime must be positive")
    if maximum_candidates_per_scene <= 0.0:
        raise ValueError("maximum candidates must be positive")
    report = evaluate(
        scenes=read_scene_ids(scene_list),
        prediction_root=prediction_root,
        diagnostics_root=diagnostics_root,
        gt_root=gt_root,
        scans_root=scans_root,
        thresholds=thresholds,
    )
    contract = _CONTRACT[stage]
    reference = _load_reference(
        reference_report, expected_stage=contract["reference_stage"]
    )
    if (
        int(reference.get("scene_count", -1)) != report["scene_count"]
        or int(reference.get("ground_truth_count", -1))
        != report["ground_truth_count"]
    ):
        raise ValueError("candidate/reference reports use different scene sets")

    row25 = _threshold(report, "0.25")
    row50 = _threshold(report, "0.50")
    reference25 = _threshold(reference, "0.25")
    reference50 = _threshold(reference, "0.50")
    recall25 = bool(float(row25["novel_recall_gain"]) >= 0.03)
    recall50 = bool(float(row50["novel_recall_gain"]) >= 0.01)
    absolute_recall = bool(recall25 and recall50)
    noninferior_tp25 = bool(
        int(row25["novel_true_positives"])
        >= int(reference25["novel_true_positives"])
    )
    improved_tp50 = bool(
        int(row50["novel_true_positives"])
        >= int(reference50["novel_true_positives"])
        + int(contract["minimum_extra_tp50"])
    )
    speed = bool(
        float(report["p1_runtime_seconds_per_scene"])
        <= float(maximum_runtime_seconds_per_scene)
    )
    bounded = bool(
        float(report["candidates_per_scene"])
        <= float(maximum_candidates_per_scene)
    )
    safety = bool(report["observer_only"] and not report["unsafe_scenes"])
    passes = bool(
        safety
        and absolute_recall
        and noninferior_tp25
        and improved_tp50
        and speed
        and bounded
    )
    report.update(
        {
            "schema": SCHEMA,
            "stage": stage,
            "head_architecture": contract["head_architecture"],
            "target_assignment_scope": contract[
                "target_assignment_scope"
            ],
            "reference": {
                "path": str(reference_report.resolve()),
                "stage": contract["reference_stage"],
                "novel_tp_at_0p25": int(
                    reference25["novel_true_positives"]
                ),
                "novel_tp_at_0p50": int(
                    reference50["novel_true_positives"]
                ),
            },
            "fixed10_go_no_go": {
                "safety_identity": safety,
                "absolute_delta_recall_at_0p25_ge_0p03": recall25,
                "absolute_delta_recall_at_0p50_ge_0p01": recall50,
                "novel_tp25_noninferior_to_reference": noninferior_tp25,
                "novel_tp50_improvement_over_reference": improved_tp50,
                "required_extra_tp50": int(
                    contract["minimum_extra_tp50"]
                ),
                "runtime_seconds_per_scene_le": float(
                    maximum_runtime_seconds_per_scene
                ),
                "runtime_passes": speed,
                "candidates_per_scene_le": float(
                    maximum_candidates_per_scene
                ),
                "candidate_bound_passes": bounded,
                "passes": passes,
                "decision": (
                    "GO_FULL100" if passes else f"STOP_{stage}"
                ),
                "frozen_protocol": (
                    "Do not tune thresholds or retrain from this fixed "
                    "validation-10 report."
                ),
            },
        }
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("P1R", "P1S"))
    parser.add_argument("--scene-list", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--diagnostics-root", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--scans-root", required=True, type=Path)
    parser.add_argument("--reference-report", required=True, type=Path)
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS
    )
    parser.add_argument(
        "--maximum-runtime-seconds-per-scene", type=float, default=0.80
    )
    parser.add_argument(
        "--maximum-candidates-per-scene", type=float, default=256.0
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_report(
        stage=args.stage,
        scene_list=args.scene_list,
        prediction_root=args.prediction_root,
        diagnostics_root=args.diagnostics_root,
        gt_root=args.gt_root,
        scans_root=args.scans_root,
        reference_report=args.reference_report,
        thresholds=args.thresholds,
        maximum_runtime_seconds_per_scene=(
            args.maximum_runtime_seconds_per_scene
        ),
        maximum_candidates_per_scene=args.maximum_candidates_per_scene,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(rendered + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(rendered)
    return 0 if report["fixed10_go_no_go"]["passes"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
