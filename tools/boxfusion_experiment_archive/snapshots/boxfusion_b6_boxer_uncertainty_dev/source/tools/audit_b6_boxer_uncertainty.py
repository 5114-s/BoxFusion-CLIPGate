#!/usr/bin/env python3
"""Audit the isolated Boxer uncertainty-aware fusion ablation."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


SCHEMA = "boxfusion.boxer_uncertainty_fusion.scene.v1"
LIFTING_SCHEMA = "boxfusion.boxer_lifting.frame.v1"
REPORT_SCHEMA = "boxfusion.b6_boxer_uncertainty.audit.v1"

PAIRED_LIFTING_KEYS = (
    "count",
    "attempt_id",
    "apply_stage",
    "mode",
    "mutation_enabled",
    "selective_gate_enabled",
    "selective_gate",
    "eligible_count",
    "applied_count",
    "fallback_count",
    "gate_accepted",
    "gate_reasons",
    "gate_rejection_counts",
    "scores_sha256",
    "boxes_2d_sha256",
    "input_pred_proj_xy_sha256",
    "cutr_geometry_sha256",
    "boxer_geometry_sha256",
    "actual_geometry_sha256",
    "boxer_checkpoint_sha256",
    "boxer_commit",
)


def read_scenes(path: Path) -> List[str]:
    scenes = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError(f"Scene list is empty or contains duplicates: {path}")
    return scenes


def as_numpy(value: Any):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return None


def compare_nested(
    left: Any,
    right: Any,
    *,
    array_atol: float,
    scalar_atol: float,
    path: str = "$",
) -> Tuple[Dict[str, Any] | None, float]:
    left_array = as_numpy(left)
    right_array = as_numpy(right)
    if left_array is not None or right_array is not None:
        if left_array is None or right_array is None:
            return {"path": path, "message": "array type mismatch"}, float("inf")
        if left_array.dtype != right_array.dtype or left_array.shape != right_array.shape:
            return {"path": path, "message": "array metadata mismatch"}, float("inf")
        if left_array.tobytes(order="C") == right_array.tobytes(order="C"):
            return None, 0.0
        if not np.issubdtype(left_array.dtype, np.number):
            return {"path": path, "message": "non-numeric array differs"}, float("inf")
        if not np.isfinite(left_array).all() or not np.isfinite(right_array).all():
            return {"path": path, "message": "non-finite array"}, float("inf")
        drift = float(
            np.max(
                np.abs(
                    left_array.astype(np.float64)
                    - right_array.astype(np.float64)
                )
            )
        )
        if drift > array_atol:
            return {
                "path": path,
                "message": "array drift exceeds tolerance",
                "max_abs": drift,
                "atol": array_atol,
            }, drift
        return None, drift
    if type(left) is not type(right):
        return {"path": path, "message": "type mismatch"}, float("inf")
    if isinstance(left, dict):
        if list(left) != list(right):
            return {"path": path, "message": "dictionary keys/order differ"}, float("inf")
        maximum = 0.0
        for key in left:
            issue, drift = compare_nested(
                left[key],
                right[key],
                array_atol=array_atol,
                scalar_atol=scalar_atol,
                path=f"{path}[{key!r}]",
            )
            maximum = max(maximum, drift)
            if issue:
                return issue, maximum
        return None, maximum
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return {"path": path, "message": "sequence length differs"}, float("inf")
        maximum = 0.0
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            issue, drift = compare_nested(
                left_item,
                right_item,
                array_atol=array_atol,
                scalar_atol=scalar_atol,
                path=f"{path}[{index}]",
            )
            maximum = max(maximum, drift)
            if issue:
                return issue, maximum
        return None, maximum
    if isinstance(left, (float, np.floating)):
        drift = abs(float(left) - float(right))
        if np.isfinite(drift) and drift <= scalar_atol:
            return None, drift
        return {"path": path, "message": "scalar drift exceeds tolerance"}, drift
    if left != right:
        return {"path": path, "message": "value mismatch"}, float("inf")
    return None, 0.0


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_lifting_diagnostics(path: Path):
    rows = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema") != LIFTING_SCHEMA:
            raise ValueError(
                f"{path}:{line_number}: unexpected lifting schema"
            )
        key = (int(row["frame_id"]), str(row.get("attempt_id", "primary")))
        if key in rows:
            raise ValueError(f"{path}: duplicate lifting row {key}")
        rows[key] = row
    if not rows:
        raise ValueError(f"No lifting diagnostics: {path}")
    return rows


def validate_paired_lifting(paths, scene, issues):
    """Prove U0/U1/U2 have identical pre-fusion Boxer proposals."""

    roles = ("control", "observer", "active")
    try:
        rows = {
            role: load_lifting_diagnostics(paths[f"{role}_boxer_diag"])
            for role in roles
        }
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        append_issue(
            issues,
            scene,
            "invalid_lifting_diagnostic",
            error=str(error),
        )
        return 0
    schedules = {role: set(rows[role]) for role in roles}
    if not (
        schedules["control"]
        == schedules["observer"]
        == schedules["active"]
    ):
        append_issue(
            issues,
            scene,
            "lifting_schedule_mismatch",
            counts={role: len(schedules[role]) for role in roles},
        )
        return 0

    matched = 0
    for key in sorted(schedules["control"]):
        reference = rows["control"][key]
        for role in roles:
            row = rows[role][key]
            if row.get("mode") != "active" or row.get("mutation_enabled") is not True:
                append_issue(
                    issues,
                    scene,
                    "lifting_not_frozen_active_g0",
                    role=role,
                    frame_id=key[0],
                    attempt_id=key[1],
                )
            for field in PAIRED_LIFTING_KEYS:
                if row.get(field) != reference.get(field):
                    append_issue(
                        issues,
                        scene,
                        "paired_lifting_mismatch",
                        role=role,
                        frame_id=key[0],
                        attempt_id=key[1],
                        field=field,
                    )
                    break
        matched += 1
    return matched


def append_issue(issues, scene, kind, **details):
    issues.append({"scene": scene, "kind": kind, **details})


def validate_diagnostic(path: Path, scene: str, expected_mode: str):
    payload = json.loads(path.read_text(encoding="utf-8"))
    issues = []
    if payload.get("schema") != SCHEMA:
        append_issue(issues, scene, "schema_mismatch")
        return payload, issues
    if payload.get("scene_id") != scene:
        append_issue(issues, scene, "scene_id_mismatch")
    cfg = payload.get("config", {})
    if cfg.get("mode") != expected_mode:
        append_issue(
            issues,
            scene,
            "mode_mismatch",
            expected=expected_mode,
            actual=cfg.get("mode"),
        )
    if cfg.get("confidence_power") != 1.0:
        append_issue(issues, scene, "confidence_power_changed")
    if cfg.get("minimum_confidence") != 0.05:
        append_issue(issues, scene, "minimum_confidence_changed")

    record_counts = {
        "fusion_groups": 0,
        "candidate_views": 0,
        "boxer_views": 0,
        "cutr_fallback_views": 0,
        "invalid_boxer_confidence": 0,
        "candidate_weight_changed_groups": 0,
        "weight_changed_groups": 0,
        "selection_changed_groups": 0,
        "ranking_changed_groups": 0,
        "active_groups": 0,
        "optimization_updated_groups": 0,
        "active_updated_groups": 0,
    }
    for record_index, record in enumerate(payload.get("records", [])):
        context = {"record": record_index}
        try:
            candidates = np.asarray(record["candidate_indices"], dtype=np.int64)
            confidence = np.asarray(record["boxer_confidence"], dtype=np.float64)
            applied = np.asarray(record["boxer_geometry_applied"], dtype=bool)
            valid = np.asarray(record["boxer_confidence_valid"], dtype=bool)
            base = np.asarray(record["base_weights"], dtype=np.float64)
            factors = np.asarray(record["uncertainty_factors"], dtype=np.float64)
            adjusted = np.asarray(record["uncertainty_weights"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            append_issue(issues, scene, "malformed_record", error=str(error), **context)
            continue
        shapes = {array.shape for array in (candidates, confidence, applied, valid, base, factors, adjusted)}
        if len(shapes) != 1:
            append_issue(issues, scene, "row_alignment_mismatch", **context)
            continue
        expected_valid = np.isfinite(confidence) & (confidence >= 0.0) & (confidence <= 1.0)
        if not np.array_equal(valid, expected_valid):
            append_issue(issues, scene, "confidence_validity_mismatch", **context)
        expected_factors = np.ones_like(confidence)
        weighted = applied & expected_valid
        expected_factors[weighted] = np.clip(confidence[weighted], 0.05, 1.0)
        if not np.allclose(factors, expected_factors, rtol=0.0, atol=1e-6):
            append_issue(issues, scene, "uncertainty_factor_mismatch", **context)
        if not np.allclose(adjusted, base * expected_factors, rtol=0.0, atol=1e-6):
            append_issue(issues, scene, "adjusted_weight_mismatch", **context)
        base_selected = list(record.get("base_selected_indices", []))
        adjusted_selected = list(
            record.get("uncertainty_selected_indices", [])
        )
        if (
            not base_selected
            or len(base_selected) != len(set(base_selected))
            or len(adjusted_selected) != len(set(adjusted_selected))
            or not set(base_selected).issubset(set(candidates.tolist()))
            or not set(adjusted_selected).issubset(set(candidates.tolist()))
        ):
            append_issue(issues, scene, "invalid_selected_indices", **context)
            continue
        changed = set(base_selected) != set(adjusted_selected)
        ranking_changed = base_selected != adjusted_selected
        if bool(record.get("selection_changed")) != changed:
            append_issue(issues, scene, "selection_change_flag_mismatch", **context)
        if bool(record.get("ranking_changed")) != ranking_changed:
            append_issue(issues, scene, "ranking_change_flag_mismatch", **context)

        candidate_changed = bool(
            np.any(np.abs(expected_factors - 1.0) > 1e-7)
        )
        candidate_to_local = {
            int(candidate): index
            for index, candidate in enumerate(candidates.tolist())
        }
        base_local = np.asarray(
            [candidate_to_local[int(value)] for value in base_selected],
            dtype=np.int64,
        )
        adjusted_local = np.asarray(
            [candidate_to_local[int(value)] for value in adjusted_selected],
            dtype=np.int64,
        )
        base_effective = np.zeros_like(base)
        base_selected_weights = base[base_local]
        base_effective[base_local] = base_selected_weights / max(
            float(base_selected_weights.mean()), 1e-12
        )
        adjusted_effective = np.zeros_like(adjusted)
        adjusted_selected_weights = adjusted[adjusted_local]
        adjusted_effective[adjusted_local] = adjusted_selected_weights / max(
            float(adjusted_selected_weights.mean()), 1e-12
        )
        effective_changed = not np.allclose(
            base_effective,
            adjusted_effective,
            rtol=1e-7,
            atol=1e-7,
        )
        if not np.allclose(
            np.asarray(record.get("base_effective_weights"), dtype=np.float64),
            base_effective,
            rtol=0.0,
            atol=1e-6,
        ):
            append_issue(issues, scene, "base_effective_weight_mismatch", **context)
        if not np.allclose(
            np.asarray(
                record.get("uncertainty_effective_weights"),
                dtype=np.float64,
            ),
            adjusted_effective,
            rtol=0.0,
            atol=1e-6,
        ):
            append_issue(
                issues,
                scene,
                "uncertainty_effective_weight_mismatch",
                **context,
            )
        if bool(record.get("candidate_weights_changed")) != candidate_changed:
            append_issue(
                issues, scene, "candidate_weight_change_flag_mismatch", **context
            )
        if bool(record.get("weights_changed")) != effective_changed:
            append_issue(issues, scene, "effective_weight_change_flag_mismatch", **context)
        expected_applied = expected_mode == "active"
        if bool(record.get("applied_to_fusion")) != expected_applied:
            append_issue(issues, scene, "application_flag_mismatch", **context)

        optimization_updated = bool(record.get("optimization_updated"))
        record_counts["fusion_groups"] += 1
        record_counts["candidate_views"] += int(candidates.size)
        record_counts["boxer_views"] += int(np.count_nonzero(applied))
        record_counts["cutr_fallback_views"] += int(np.count_nonzero(~applied))
        record_counts["invalid_boxer_confidence"] += int(np.count_nonzero(applied & ~valid))
        record_counts["candidate_weight_changed_groups"] += int(
            candidate_changed
        )
        record_counts["weight_changed_groups"] += int(effective_changed)
        record_counts["selection_changed_groups"] += int(changed)
        record_counts["ranking_changed_groups"] += int(ranking_changed)
        record_counts["active_groups"] += int(
            expected_applied and effective_changed
        )
        record_counts["optimization_updated_groups"] += int(
            optimization_updated
        )
        record_counts["active_updated_groups"] += int(
            expected_applied and effective_changed and optimization_updated
        )

    summary = payload.get("summary", {})
    for key, expected in record_counts.items():
        if int(summary.get(key, -1)) != expected:
            append_issue(
                issues,
                scene,
                "summary_count_mismatch",
                field=key,
                expected=expected,
                actual=summary.get(key),
            )
    return payload, issues


def audit(args):
    scenes = read_scenes(args.scene_list)
    issues = []
    source_identity = 0
    observer_identity = 0
    source_max_abs = 0.0
    observer_max_abs = 0.0
    lifting_identity_frames = 0
    aggregate = {
        role: {
            "fusion_groups": 0,
            "candidate_views": 0,
            "boxer_views": 0,
            "cutr_fallback_views": 0,
            "invalid_boxer_confidence": 0,
            "weight_changed_groups": 0,
            "selection_changed_groups": 0,
            "active_groups": 0,
        }
        for role in ("observer", "active")
    }

    for scene in scenes:
        paths = {
            "source": args.source_g0_root / f"{scene}_boxes.pkl",
            "control": args.control_root / f"{scene}_boxes.pkl",
            "observer": args.observer_root / f"{scene}_boxes.pkl",
            "active": args.active_root / f"{scene}_boxes.pkl",
            "observer_diag": args.observer_diagnostics / f"{scene}_boxer_uncertainty.json",
            "active_diag": args.active_diagnostics / f"{scene}_boxer_uncertainty.json",
            "control_boxer_diag": args.control_boxer_diagnostics / f"{scene}_boxer_lifting.jsonl",
            "observer_boxer_diag": args.observer_boxer_diagnostics / f"{scene}_boxer_lifting.jsonl",
            "active_boxer_diag": args.active_boxer_diagnostics / f"{scene}_boxer_lifting.jsonl",
        }
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            append_issue(issues, scene, "missing_artifact", names=missing)
            continue

        lifting_identity_frames += validate_paired_lifting(
            paths, scene, issues
        )

        source_issue, source_drift = compare_nested(
            load_pickle(paths["source"]),
            load_pickle(paths["control"]),
            array_atol=args.box_atol,
            scalar_atol=args.score_atol,
        )
        source_max_abs = max(source_max_abs, source_drift)
        if source_issue:
            append_issue(issues, scene, "control_not_frozen_g0", **source_issue)
        else:
            source_identity += 1

        observer_issue, observer_drift = compare_nested(
            load_pickle(paths["control"]),
            load_pickle(paths["observer"]),
            array_atol=args.box_atol,
            scalar_atol=args.score_atol,
        )
        observer_max_abs = max(observer_max_abs, observer_drift)
        if observer_issue:
            append_issue(issues, scene, "observer_prediction_mismatch", **observer_issue)
        else:
            observer_identity += 1

        for role in ("observer", "active"):
            payload, diagnostic_issues = validate_diagnostic(
                paths[f"{role}_diag"], scene, role
            )
            issues.extend(diagnostic_issues)
            summary = payload.get("summary", {})
            for key in aggregate[role]:
                aggregate[role][key] += int(summary.get(key, 0))

    groups = aggregate["observer"]["fusion_groups"]
    changed = aggregate["observer"]["weight_changed_groups"]
    return {
        "schema": REPORT_SCHEMA,
        "ok": not issues,
        "scene_count": len(scenes),
        "frozen_g0_identity_scenes": source_identity,
        "observer_identity_scenes": observer_identity,
        "box_atol": args.box_atol,
        "score_atol": args.score_atol,
        "frozen_g0_max_abs_drift": source_max_abs,
        "observer_max_abs_drift": observer_max_abs,
        "lifting_identity_frames": lifting_identity_frames,
        "observer": aggregate["observer"],
        "active": aggregate["active"],
        "observer_weight_coverage": changed / groups if groups else 0.0,
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--source-g0-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--observer-root", type=Path, required=True)
    parser.add_argument("--active-root", type=Path, required=True)
    parser.add_argument("--observer-diagnostics", type=Path, required=True)
    parser.add_argument("--active-diagnostics", type=Path, required=True)
    parser.add_argument(
        "--control-boxer-diagnostics", type=Path, required=True
    )
    parser.add_argument(
        "--observer-boxer-diagnostics", type=Path, required=True
    )
    parser.add_argument(
        "--active-boxer-diagnostics", type=Path, required=True
    )
    parser.add_argument("--box-atol", type=float, default=1e-4)
    parser.add_argument("--score-atol", type=float, default=1e-6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0.0 <= args.box_atol <= 1e-3:
        parser.error("--box-atol must lie in [0, 1e-3]")
    if not 0.0 <= args.score_atol <= 1e-5:
        parser.error("--score-atol must lie in [0, 1e-5]")
    report = audit(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
