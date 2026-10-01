#!/usr/bin/env python3
"""Diagnose a paired train-only conservative/permissive P1G replay.

This tool never selects a production configuration and never reads validation
ground truth.  It compares two summaries over the exact same train-only P1S
candidate cohort.  A bounded-face GT oracle then separates insufficient parent
proposals from the default face clamp, internal MSR evidence gates, and the
association/boundary-estimation method itself.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


INPUT_SCHEMA = "boxfusion.p1g.train_msr_replay_summary.v1"
REPORT_SCHEMA = "boxfusion.p1g.train_msr_diagnosis.v1"
CLASSIFICATIONS = (
    "effective",
    "clamp_parameter_problem",
    "internal_gate_parameter_problem",
    "association_or_boundary_estimation_method_problem",
    "parent_proposal_method_limit",
)
_SCENE = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ACTUAL_FIELDS = (
    "matched_residual_count",
    "improved_count",
    "degraded_count",
    "cross_iou50",
    "fall_iou50",
)
_IDENTITY_FIELDS = (
    "scene_list_sha256",
    "forbidden_scene_list_sha256",
    "p1s_checkpoint_sha256",
    "b6_checkpoint_sha256",
)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _integer(
    mapping: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label}.{key} must be a non-negative integer")
    return int(value)


def _finite_float(
    mapping: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}.{key} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(
            f"{label}.{key} must be finite and non-negative"
        )
    return result


def _actual(value: Any, *, label: str) -> dict[str, int]:
    mapping = _mapping(value, label=label)
    result = {
        key: _integer(mapping, key, label=label)
        for key in _ACTUAL_FIELDS
    }
    matched = result["matched_residual_count"]
    if (
        result["improved_count"] + result["degraded_count"] > matched
        or result["cross_iou50"] > result["improved_count"]
        or result["fall_iou50"] > result["degraded_count"]
    ):
        raise ValueError(f"{label} contains inconsistent transition counts")
    return result


def _feasibility(
    value: Any,
    *,
    expected_keys: tuple[str, ...],
    label: str,
) -> dict[str, dict[str, int]]:
    mapping = _mapping(value, label=label)
    if set(mapping) != set(expected_keys):
        raise ValueError(
            f"{label} face-limit keys disagree with provenance"
        )
    result: dict[str, dict[str, int]] = {}
    sample_counts: set[int] = set()
    for key in expected_keys:
        row = _mapping(mapping[key], label=f"{label}.{key}")
        sample_count = _integer(
            row, "sample_count", label=f"{label}.{key}"
        )
        cross = _integer(row, "cross_iou50", label=f"{label}.{key}")
        if cross > sample_count:
            raise ValueError(
                f"{label}.{key}.cross_iou50 exceeds sample_count"
            )
        result[key] = {
            "sample_count": sample_count,
            "cross_iou50": cross,
        }
        sample_counts.add(sample_count)
    if len(sample_counts) > 1:
        raise ValueError(f"{label} feasibility sample cohorts disagree")
    if result["0.50"]["cross_iou50"] < result["0.18"]["cross_iou50"]:
        raise ValueError(f"{label} bounded-face feasibility is non-monotonic")
    return result


def _provenance(
    value: Any,
    *,
    expected_profile: str,
    label: str,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    mapping = _mapping(value, label=label)
    if mapping.get("msr_evidence_profile") != expected_profile:
        raise ValueError(
            f"{label}.msr_evidence_profile must be {expected_profile!r}"
        )
    normalized: dict[str, Any] = {
        "msr_evidence_profile": expected_profile
    }
    for key in _IDENTITY_FIELDS:
        value = mapping.get(key)
        if (
            not isinstance(value, str)
            or _SHA256.fullmatch(value.lower()) is None
        ):
            raise ValueError(f"{label}.{key} must be a SHA256 string")
        normalized[key] = value.lower()
    overlap = mapping.get("forbidden_overlap")
    if overlap != []:
        raise ValueError(f"{label}.forbidden_overlap must be empty")
    covered = mapping.get("covered_iou")
    if (
        isinstance(covered, bool)
        or not isinstance(covered, (int, float))
        or not math.isfinite(float(covered))
        or not 0.0 <= float(covered) <= 1.0
    ):
        raise ValueError(f"{label}.covered_iou must be in [0,1]")
    normalized["covered_iou"] = float(covered)
    limits = mapping.get("face_limits")
    if not isinstance(limits, list) or not limits:
        raise ValueError(f"{label}.face_limits must be a non-empty list")
    parsed_limits = []
    for value in limits:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{label}.face_limits contains an invalid value")
        parsed_limits.append(float(value))
    keys = tuple(f"{value:.2f}" for value in parsed_limits)
    if len(keys) != len(set(keys)) or not {"0.18", "0.50"}.issubset(keys):
        raise ValueError(
            f"{label}.face_limits must uniquely include 0.18 and 0.50"
        )
    normalized["face_limits"] = parsed_limits
    config = _mapping(mapping.get("p1g_config"), label=f"{label}.p1g_config")
    for key, expected in (
        ("enabled", True),
        ("observer_only", True),
        ("mutate", False),
        ("collect_diagnostics", True),
    ):
        if config.get(key) is not expected:
            raise ValueError(f"{label}.p1g_config.{key} is unsafe")
    return normalized, keys


def _scene_row(
    value: Any,
    *,
    expected_scene: str,
    feasibility_keys: tuple[str, ...],
    label: str,
) -> dict[str, Any]:
    mapping = _mapping(value, label=label)
    if mapping.get("scene_id") != expected_scene:
        raise ValueError(f"{label}.scene_id disagrees with scene_ids")
    candidates = _integer(mapping, "candidate_count", label=label)
    rows = _integer(mapping, "p1g_row_count", label=label)
    changed = _integer(mapping, "p1g_changed_count", label=label)
    failures = _integer(mapping, "p1g_failure_count", label=label)
    if rows != candidates:
        raise ValueError(f"{label} violates one-to-one P1G row count")
    if changed > candidates:
        raise ValueError(f"{label}.p1g_changed_count exceeds candidates")
    runtime = _finite_float(
        mapping, "p1g_runtime_seconds", label=label
    )
    return {
        "scene_id": expected_scene,
        "candidate_count": candidates,
        "p1g_row_count": rows,
        "p1g_changed_count": changed,
        "p1g_failure_count": failures,
        "p1g_runtime_seconds": runtime,
        "actual": _actual(mapping.get("actual"), label=f"{label}.actual"),
        "feasibility": _feasibility(
            mapping.get("feasibility"),
            expected_keys=feasibility_keys,
            label=f"{label}.feasibility",
        ),
    }


def _load_summary(path: Path, *, expected_profile: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON") from error
    root = _mapping(payload, label=str(path))
    if root.get("schema") != INPUT_SCHEMA:
        raise ValueError(f"{path}: replay summary schema mismatch")
    scene_count = _integer(root, "scene_count", label=str(path))
    if scene_count < 1:
        raise ValueError(f"{path}: scene_count must be positive")
    scene_ids = root.get("scene_ids")
    if (
        not isinstance(scene_ids, list)
        or len(scene_ids) != scene_count
        or len(set(scene_ids)) != scene_count
        or any(
            not isinstance(scene, str)
            or _SCENE.fullmatch(scene) is None
            for scene in scene_ids
        )
    ):
        raise ValueError(f"{path}: invalid or duplicate scene_ids")
    provenance, feasibility_keys = _provenance(
        root.get("provenance"),
        expected_profile=expected_profile,
        label=f"{path}.provenance",
    )
    candidate_count = _integer(root, "candidate_count", label=str(path))
    changed_count = _integer(
        root, "p1g_changed_count", label=str(path)
    )
    failure_count = _integer(
        root, "p1g_failure_count", label=str(path)
    )
    if changed_count > candidate_count:
        raise ValueError(f"{path}: changed count exceeds candidates")
    runtime = _finite_float(
        root, "p1g_runtime_seconds", label=str(path)
    )
    actual = _actual(root.get("actual"), label=f"{path}.actual")
    feasibility = _feasibility(
        root.get("feasibility"),
        expected_keys=feasibility_keys,
        label=f"{path}.feasibility",
    )
    raw_scenes = root.get("scenes")
    if not isinstance(raw_scenes, list) or len(raw_scenes) != scene_count:
        raise ValueError(f"{path}: scenes must align with scene_count")
    scenes = [
        _scene_row(
            row,
            expected_scene=scene,
            feasibility_keys=feasibility_keys,
            label=f"{path}.scenes[{index}]",
        )
        for index, (scene, row) in enumerate(zip(scene_ids, raw_scenes))
    ]
    if sum(row["candidate_count"] for row in scenes) != candidate_count:
        raise ValueError(f"{path}: aggregate candidate_count disagrees")
    if sum(row["p1g_changed_count"] for row in scenes) != changed_count:
        raise ValueError(f"{path}: aggregate changed count disagrees")
    if sum(row["p1g_failure_count"] for row in scenes) != failure_count:
        raise ValueError(f"{path}: aggregate failure count disagrees")
    scene_runtime = sum(row["p1g_runtime_seconds"] for row in scenes)
    if not math.isclose(scene_runtime, runtime, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{path}: aggregate runtime disagrees")
    for key in _ACTUAL_FIELDS:
        if sum(row["actual"][key] for row in scenes) != actual[key]:
            raise ValueError(f"{path}: aggregate actual.{key} disagrees")
    for limit in feasibility_keys:
        for key in ("sample_count", "cross_iou50"):
            if (
                sum(
                    row["feasibility"][limit][key] for row in scenes
                )
                != feasibility[limit][key]
            ):
                raise ValueError(
                    f"{path}: aggregate feasibility.{limit}.{key} "
                    "disagrees"
                )
    return {
        "path": str(path.resolve()),
        "scene_count": scene_count,
        "scene_ids": tuple(scene_ids),
        "candidate_count": candidate_count,
        "p1g_changed_count": changed_count,
        "p1g_failure_count": failure_count,
        "p1g_runtime_seconds": runtime,
        "actual": actual,
        "feasibility": feasibility,
        "provenance": provenance,
        "per_scene_candidate_counts": tuple(
            row["candidate_count"] for row in scenes
        ),
    }


def _paired_value(
    conservative: int | float,
    permissive: int | float,
) -> dict[str, int | float | None]:
    delta = permissive - conservative
    if isinstance(conservative, int) and isinstance(permissive, int):
        delta = int(delta)
    ratio = (
        float(permissive / conservative)
        if float(conservative) > 0.0
        else None
    )
    return {
        "conservative": conservative,
        "permissive": permissive,
        "delta": delta,
        "permissive_over_conservative": ratio,
    }


def diagnose(
    *,
    conservative_summary: Path,
    permissive_summary: Path,
    minimum_cross_iou50: int = 5,
) -> dict[str, Any]:
    """Validate a paired replay and return one frozen causal diagnosis."""

    if (
        isinstance(minimum_cross_iou50, bool)
        or not isinstance(minimum_cross_iou50, int)
        or minimum_cross_iou50 < 1
    ):
        raise ValueError("minimum_cross_iou50 must be a positive integer")
    conservative = _load_summary(
        Path(conservative_summary), expected_profile="conservative"
    )
    permissive = _load_summary(
        Path(permissive_summary), expected_profile="permissive"
    )
    for key in ("scene_ids", "candidate_count", "per_scene_candidate_counts"):
        if conservative[key] != permissive[key]:
            raise ValueError(f"paired replay {key} mismatch")
    for key in _IDENTITY_FIELDS:
        if (
            conservative["provenance"][key]
            != permissive["provenance"][key]
        ):
            raise ValueError(f"paired replay {key} mismatch")
    for key in ("covered_iou", "face_limits"):
        if (
            conservative["provenance"][key]
            != permissive["provenance"][key]
        ):
            raise ValueError(f"paired replay {key} mismatch")
    if conservative["feasibility"] != permissive["feasibility"]:
        raise ValueError("paired replay feasibility mismatch")

    threshold = int(minimum_cross_iou50)
    default_cross = conservative["actual"]["cross_iou50"]
    permissive_cross = permissive["actual"]["cross_iou50"]
    feasible_018 = conservative["feasibility"]["0.18"][
        "cross_iou50"
    ]
    feasible_050 = conservative["feasibility"]["0.50"][
        "cross_iou50"
    ]
    if default_cross >= threshold:
        classification = "effective"
    elif feasible_050 < threshold:
        classification = "parent_proposal_method_limit"
    elif feasible_018 < threshold:
        classification = "clamp_parameter_problem"
    elif permissive_cross >= threshold:
        classification = "internal_gate_parameter_problem"
    else:
        classification = (
            "association_or_boundary_estimation_method_problem"
        )
    assert classification in CLASSIFICATIONS

    actual_comparison = {
        key: _paired_value(
            conservative["actual"][key],
            permissive["actual"][key],
        )
        for key in _ACTUAL_FIELDS
    }
    runtime_comparison = _paired_value(
        conservative["p1g_runtime_seconds"],
        permissive["p1g_runtime_seconds"],
    )
    runtime_comparison.update(
        {
            "conservative_seconds_per_scene": float(
                conservative["p1g_runtime_seconds"]
                / conservative["scene_count"]
            ),
            "permissive_seconds_per_scene": float(
                permissive["p1g_runtime_seconds"]
                / permissive["scene_count"]
            ),
        }
    )
    report = {
        "schema": REPORT_SCHEMA,
        "train_only": True,
        "uses_ground_truth_for_diagnosis": True,
        "deployable_selector": False,
        "input_contract": {
            "same_scene_ids": True,
            "same_scene_list_sha256": True,
            "same_forbidden_list_sha256": True,
            "same_p1s_checkpoint_sha256": True,
            "same_b6_checkpoint_sha256": True,
            "same_candidate_count": True,
            "same_per_scene_candidate_counts": True,
            "same_feasibility_cohort": True,
        },
        "inputs": {
            "conservative": conservative["path"],
            "permissive": permissive["path"],
            "scene_count": conservative["scene_count"],
            "candidate_count": conservative["candidate_count"],
            "scene_list_sha256": conservative["provenance"][
                "scene_list_sha256"
            ],
            "p1s_checkpoint_sha256": conservative["provenance"][
                "p1s_checkpoint_sha256"
            ],
            "b6_checkpoint_sha256": conservative["provenance"][
                "b6_checkpoint_sha256"
            ],
        },
        "comparison": {
            "changed_count": _paired_value(
                conservative["p1g_changed_count"],
                permissive["p1g_changed_count"],
            ),
            "failure_count": _paired_value(
                conservative["p1g_failure_count"],
                permissive["p1g_failure_count"],
            ),
            "actual": actual_comparison,
            "runtime": runtime_comparison,
            "feasibility_cross_iou50": {
                "0.18": feasible_018,
                "0.50": feasible_050,
                "sample_count": conservative["feasibility"]["0.18"][
                    "sample_count"
                ],
            },
        },
        "decision": {
            "classification": classification,
            "minimum_cross_iou50": threshold,
            "default_actual_cross_reaches_threshold": bool(
                default_cross >= threshold
            ),
            "default_face_0p18_feasibility_reaches_threshold": bool(
                feasible_018 >= threshold
            ),
            "face_0p50_feasibility_reaches_threshold": bool(
                feasible_050 >= threshold
            ),
            "permissive_actual_cross_reaches_threshold": bool(
                permissive_cross >= threshold
            ),
            "interpretation": {
                "effective": (
                    "The conservative default already realizes the required "
                    "train-only IoU@0.50 crossings."
                ),
                "clamp_parameter_problem": (
                    "The parent cohort has enough 0.50-face oracle capacity, "
                    "but the default 0.18 face clamp does not."
                ),
                "internal_gate_parameter_problem": (
                    "The default clamp has enough oracle capacity and the "
                    "permissive evidence gates realize the target."
                ),
                "association_or_boundary_estimation_method_problem": (
                    "The default clamp has enough oracle capacity, but "
                    "neither evidence profile realizes it."
                ),
                "parent_proposal_method_limit": (
                    "Even the 0.50 bounded-face oracle lacks enough parent "
                    "proposal crossings."
                ),
            }[classification],
        },
    }
    json.dumps(report, allow_nan=False)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conservative-summary", required=True, type=Path
    )
    parser.add_argument("--permissive-summary", required=True, type=Path)
    parser.add_argument(
        "--minimum-cross-iou50", type=int, default=5
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = diagnose(
        conservative_summary=args.conservative_summary,
        permissive_summary=args.permissive_summary,
        minimum_cross_iou50=args.minimum_cross_iou50,
    )
    rendered = json.dumps(
        report, indent=2, sort_keys=True, allow_nan=False
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(rendered + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
