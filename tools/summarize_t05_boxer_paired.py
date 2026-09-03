#!/usr/bin/env python3
"""Read-only summary for paired T05 + Boxer observer/active experiments.

The tool deliberately treats incomplete artifacts and not-yet-written evaluation
logs as ``pending``.  It never creates, edits, or deletes experiment artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_LIST = (
    REPO_ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt"
)
DIAGNOSTIC_SCHEMA = "boxfusion.boxer_lifting.frame.v1"

# These are the immutable inputs/provenance fields used by the original Boxer
# contract audit.  Geometry outputs are included to prove that paired arms ran
# the same deterministic Boxer counterfactual before active mutation.
PAIRED_PROTECTED_KEYS: Tuple[str, ...] = (
    "count",
    "boxes_2d_sha256",
    "scores_sha256",
    "image_sha256",
    "depth_sha256",
    "image_intrinsics_sha256",
    "depth_intrinsics_sha256",
    "camera_to_world_sha256",
    "protected_hashes",
    "input_pred_proj_xy_sha256",
    "use_sdp",
    "sdp_seed",
    "cutr_geometry_sha256",
    "boxer_geometry_sha256",
    "boxer_checkpoint_sha256",
    "boxer_commit",
)
PROTECTED_INSTANCE_KEYS: Tuple[str, ...] = (
    "object_desc",
    "pred_boxes",
    "pred_classes",
    "pred_logits",
    "scores",
)
COUNTED_ARRAY_FIELDS: Tuple[str, ...] = (
    "input_boxes_xyxy",
    "detector_scores",
    "confidence",
    "cutr_xyz_dims_camera",
    "cutr_rotation_camera_object",
    "output_xyz_dims_camera",
    "output_rotation_camera_object",
    "raw_params_voxel",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EVAL_MAP_RE = re.compile(
    r"eval\s+mAP\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
KEPT_SCANS_RE = re.compile(
    r"kept\s+(\d+)\s+scans\s+out\s+of\s+(\d+)", re.IGNORECASE
)


DiagnosticKey = Tuple[int, str]


@dataclass
class IssueCollector:
    """Keep exact counts while bounding verbose examples."""

    max_samples: int
    total: int = 0
    by_kind: Counter[str] = field(default_factory=Counter)
    samples: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, kind: str, **details: Any) -> None:
        self.total += 1
        self.by_kind[kind] += 1
        if len(self.samples) < self.max_samples:
            self.samples.append({"kind": kind, **details})

    def as_dict(self) -> Dict[str, Any]:
        return {
            "count": self.total,
            "by_kind": dict(sorted(self.by_kind.items())),
            "samples": self.samples,
            "samples_truncated": self.total > len(self.samples),
        }


def read_scene_list(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"scene list does not exist: {path}")
    scenes = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not scenes:
        raise ValueError(f"scene list is empty: {path}")
    duplicates = sorted(scene for scene, n in Counter(scenes).items() if n > 1)
    if duplicates:
        raise ValueError(f"duplicate scene IDs in {path}: {duplicates}")
    return scenes


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """NumPy-compatible linear percentile without adding a dependency."""

    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def is_finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, (list, tuple)):
        return all(is_finite_tree(item) for item in value)
    return False


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def prediction_box_count(path: Path) -> int:
    """Count evaluator-format predictions: batches of (class, corners, score)."""

    with path.open("rb") as handle:
        prediction = pickle.load(handle)
    if not isinstance(prediction, (list, tuple)):
        raise ValueError(
            f"expected list/tuple of prediction batches, got {type(prediction).__name__}"
        )
    total = 0
    for batch_index, batch in enumerate(prediction):
        if not isinstance(batch, (list, tuple)):
            raise ValueError(
                f"prediction batch {batch_index} is {type(batch).__name__}, not list/tuple"
            )
        for item_index, item in enumerate(batch):
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                raise ValueError(
                    f"prediction item {batch_index}/{item_index} is not a 3-field tuple"
                )
        total += len(batch)
    return total


def row_contract_issues(
    row: Mapping[str, Any],
    *,
    role: str,
    scene: str,
    line_number: int,
    issues: IssueCollector,
) -> None:
    context = {"arm": role, "scene": scene, "line": line_number}
    if row.get("schema") != DIAGNOSTIC_SCHEMA:
        issues.add(
            "unexpected_diagnostic_schema",
            **context,
            expected=DIAGNOSTIC_SCHEMA,
            actual=row.get("schema"),
        )
    if row.get("scene_id") != scene:
        issues.add(
            "diagnostic_scene_mismatch",
            **context,
            actual=row.get("scene_id"),
        )
    try:
        int(row["frame_id"])
    except (KeyError, TypeError, ValueError):
        issues.add("invalid_frame_id", **context, value=row.get("frame_id"))

    try:
        count = int(row.get("count", -1))
    except (TypeError, ValueError):
        count = -1
    if count < 0:
        issues.add("invalid_diagnostic_count", **context, value=row.get("count"))
        return

    expected_mutation = role == "active"
    if row.get("mode") != role:
        issues.add(
            "unexpected_diagnostic_mode",
            **context,
            expected=role,
            actual=row.get("mode"),
        )
    if row.get("mutation_enabled") is not expected_mutation:
        issues.add(
            "mutation_contract_mismatch",
            **context,
            expected=expected_mutation,
            actual=row.get("mutation_enabled"),
        )

    try:
        applied = int(row.get("applied_count", -1))
    except (TypeError, ValueError):
        applied = -1
    expected_applied = count if role == "active" else 0
    if applied != expected_applied:
        issues.add(
            "applied_count_contract_mismatch",
            **context,
            expected=expected_applied,
            actual=row.get("applied_count"),
        )

    projected_center_replaced = bool(row.get("projected_center_replaced", False))
    expected_center_replacement = (
        role == "active" and count > 0 and row.get("apply_stage") == "pre_filter"
    )
    if projected_center_replaced != expected_center_replacement:
        issues.add(
            "projected_center_contract_mismatch",
            **context,
            expected=expected_center_replacement,
            actual=projected_center_replaced,
        )

    protected = row.get("protected_hashes")
    if not isinstance(protected, dict):
        issues.add("missing_protected_hashes", **context)
    else:
        for key in PROTECTED_INSTANCE_KEYS:
            if not is_sha256(protected.get(key)):
                issues.add(
                    "invalid_protected_hash",
                    **context,
                    field=key,
                    value=protected.get(key),
                )

    # Empty proposal frames intentionally omit inference-only fields.
    if count == 0:
        return

    for field_name in COUNTED_ARRAY_FIELDS:
        value = row.get(field_name)
        if not isinstance(value, list) or len(value) != count:
            issues.add(
                "diagnostic_length_mismatch",
                **context,
                field=field_name,
                expected=count,
                actual=len(value) if isinstance(value, list) else None,
            )
        elif not is_finite_tree(value):
            issues.add(
                "diagnostic_nonfinite",
                **context,
                field=field_name,
            )

    runtime = row.get("runtime_ms")
    if (
        isinstance(runtime, bool)
        or not isinstance(runtime, (int, float))
        or not math.isfinite(float(runtime))
        or float(runtime) < 0.0
    ):
        issues.add("invalid_runtime_ms", **context, value=runtime)

    for field_name in (
        "boxes_2d_sha256",
        "scores_sha256",
        "image_sha256",
        "depth_sha256",
        "image_intrinsics_sha256",
        "depth_intrinsics_sha256",
        "camera_to_world_sha256",
        "input_pred_proj_xy_sha256",
        "cutr_geometry_sha256",
        "boxer_geometry_sha256",
        "boxer_checkpoint_sha256",
    ):
        if not is_sha256(row.get(field_name)):
            issues.add(
                "invalid_diagnostic_hash",
                **context,
                field=field_name,
                value=row.get(field_name),
            )
    if isinstance(protected, dict):
        if row.get("scores_sha256") != protected.get("scores"):
            issues.add("score_hash_contract_mismatch", **context)
        if row.get("boxes_2d_sha256") != protected.get("pred_boxes"):
            issues.add("box_hash_contract_mismatch", **context)


def load_diagnostics(
    path: Path,
    *,
    role: str,
    scene: str,
    issues: IssueCollector,
) -> Dict[DiagnosticKey, Dict[str, Any]]:
    rows: Dict[DiagnosticKey, Dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        issues.add(
            "diagnostic_read_error",
            arm=role,
            scene=scene,
            path=str(path),
            error=str(exc),
        )
        return rows
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.add(
                "diagnostic_json_error",
                arm=role,
                scene=scene,
                line=line_number,
                error=str(exc),
            )
            continue
        if not isinstance(row, dict):
            issues.add(
                "diagnostic_row_not_object",
                arm=role,
                scene=scene,
                line=line_number,
            )
            continue
        row_contract_issues(
            row,
            role=role,
            scene=scene,
            line_number=line_number,
            issues=issues,
        )
        try:
            key = (int(row["frame_id"]), str(row.get("attempt_id", "primary")))
        except (KeyError, TypeError, ValueError):
            continue
        if key in rows:
            issues.add(
                "duplicate_frame_attempt",
                arm=role,
                scene=scene,
                frame_id=key[0],
                attempt_id=key[1],
            )
            continue
        rows[key] = row
    if not rows:
        issues.add(
            "empty_diagnostic",
            arm=role,
            scene=scene,
            path=str(path),
        )
    return rows


def summarize_arm(
    *,
    role: str,
    prediction_root: Path,
    diagnostic_root: Path,
    scenes: Sequence[str],
    issues: IssueCollector,
) -> Tuple[Dict[str, Any], Dict[str, Dict[DiagnosticKey, Dict[str, Any]]]]:
    prediction_files = {
        path.name[: -len("_boxes.pkl")]: path
        for path in prediction_root.glob("*_boxes.pkl")
        if path.is_file()
    } if prediction_root.is_dir() else {}
    diagnostic_files = {
        path.name[: -len("_boxer_lifting.jsonl")]: path
        for path in diagnostic_root.glob("*_boxer_lifting.jsonl")
        if path.is_file()
    } if diagnostic_root.is_dir() else {}

    target = set(scenes)
    missing_predictions = [scene for scene in scenes if scene not in prediction_files]
    missing_diagnostics = [scene for scene in scenes if scene not in diagnostic_files]
    extra_predictions = sorted(set(prediction_files) - target)
    extra_diagnostics = sorted(set(diagnostic_files) - target)

    total_boxes = 0
    readable_predictions = 0
    for scene in scenes:
        path = prediction_files.get(scene)
        if path is None:
            continue
        try:
            total_boxes += prediction_box_count(path)
            readable_predictions += 1
        except Exception as exc:  # pickle/numpy errors must be reported, not hidden.
            issues.add(
                "prediction_read_error",
                arm=role,
                scene=scene,
                path=str(path),
                error=f"{type(exc).__name__}: {exc}",
            )

    scene_rows: Dict[str, Dict[DiagnosticKey, Dict[str, Any]]] = {}
    runtimes: List[float] = []
    calls = proposals = applied = observer_calls = 0
    for scene in scenes:
        path = diagnostic_files.get(scene)
        if path is None:
            continue
        rows = load_diagnostics(path, role=role, scene=scene, issues=issues)
        scene_rows[scene] = rows
        calls += len(rows)
        for row in rows.values():
            count = int(row.get("count", 0))
            proposals += count
            applied += int(row.get("applied_count", 0))
            if role == "observer" and count > 0:
                observer_calls += 1
            runtime = row.get("runtime_ms")
            if (
                not isinstance(runtime, bool)
                and isinstance(runtime, (int, float))
                and math.isfinite(float(runtime))
            ):
                runtimes.append(float(runtime))

    summary = {
        "prediction_root": str(prediction_root),
        "diagnostic_root": str(diagnostic_root),
        "expected_scenes": len(scenes),
        "prediction_files_present": len(scenes) - len(missing_predictions),
        "readable_prediction_files": readable_predictions,
        "diagnostic_files_present": len(scenes) - len(missing_diagnostics),
        "artifact_complete": (
            not missing_predictions
            and not missing_diagnostics
            and readable_predictions == len(scenes)
        ),
        "missing_predictions": missing_predictions,
        "missing_diagnostics": missing_diagnostics,
        "extra_predictions_ignored": extra_predictions,
        "extra_diagnostics_ignored": extra_diagnostics,
        "box_count": total_boxes,
        "boxer_calls": calls,
        "boxer_proposals": proposals,
        "boxer_applied": applied,
        "boxer_observer_calls": observer_calls,
        "runtime_samples": len(runtimes),
        "runtime_p50_ms": percentile(runtimes, 0.50),
        "runtime_p95_ms": percentile(runtimes, 0.95),
    }
    return summary, scene_rows


def pair_diagnostics(
    *,
    scenes: Sequence[str],
    observer_rows: Mapping[str, Mapping[DiagnosticKey, Mapping[str, Any]]],
    active_rows: Mapping[str, Mapping[DiagnosticKey, Mapping[str, Any]]],
    issues: IssueCollector,
) -> Dict[str, Any]:
    observer_frame_attempts = 0
    active_frame_attempts = 0
    shared_frame_attempts = 0
    observer_only = 0
    active_only = 0
    proposal_count_mismatch_frames = 0
    protected_mismatches = 0
    paired_scenes = 0

    for scene in scenes:
        if scene not in observer_rows or scene not in active_rows:
            continue
        paired_scenes += 1
        observer_scene = observer_rows[scene]
        active_scene = active_rows[scene]
        observer_keys = set(observer_scene)
        active_keys = set(active_scene)
        observer_frame_attempts += len(observer_keys)
        active_frame_attempts += len(active_keys)
        shared = observer_keys & active_keys
        shared_frame_attempts += len(shared)
        only_observer = observer_keys - active_keys
        only_active = active_keys - observer_keys
        observer_only += len(only_observer)
        active_only += len(only_active)
        for frame_id, attempt_id in sorted(only_observer):
            issues.add(
                "frame_coverage_observer_only",
                scene=scene,
                frame_id=frame_id,
                attempt_id=attempt_id,
            )
        for frame_id, attempt_id in sorted(only_active):
            issues.add(
                "frame_coverage_active_only",
                scene=scene,
                frame_id=frame_id,
                attempt_id=attempt_id,
            )
        for frame_id, attempt_id in sorted(shared):
            observer_row = observer_scene[(frame_id, attempt_id)]
            active_row = active_scene[(frame_id, attempt_id)]
            if observer_row.get("apply_stage") != active_row.get("apply_stage"):
                protected_mismatches += 1
                issues.add(
                    "paired_apply_stage_mismatch",
                    scene=scene,
                    frame_id=frame_id,
                    attempt_id=attempt_id,
                    observer=observer_row.get("apply_stage"),
                    active=active_row.get("apply_stage"),
                )
            if observer_row.get("count") != active_row.get("count"):
                proposal_count_mismatch_frames += 1
            for key in PAIRED_PROTECTED_KEYS:
                if observer_row.get(key) != active_row.get(key):
                    protected_mismatches += 1
                    issues.add(
                        "paired_protected_field_mismatch",
                        scene=scene,
                        frame_id=frame_id,
                        attempt_id=attempt_id,
                        field=key,
                        observer=observer_row.get(key),
                        active=active_row.get(key),
                    )

    observer_proposals = sum(
        int(row.get("count", 0))
        for scene_rows in observer_rows.values()
        for row in scene_rows.values()
    )
    active_proposals = sum(
        int(row.get("count", 0))
        for scene_rows in active_rows.values()
        for row in scene_rows.values()
    )
    all_scenes_paired = paired_scenes == len(scenes)
    frame_coverage_match = (
        all_scenes_paired and observer_only == 0 and active_only == 0
    )
    proposal_coverage_match = (
        frame_coverage_match
        and proposal_count_mismatch_frames == 0
        and observer_proposals == active_proposals
    )
    return {
        "expected_scenes": len(scenes),
        "paired_diagnostic_scenes": paired_scenes,
        "observer_frame_attempts": observer_frame_attempts,
        "active_frame_attempts": active_frame_attempts,
        "shared_frame_attempts": shared_frame_attempts,
        "observer_only_frame_attempts": observer_only,
        "active_only_frame_attempts": active_only,
        "frame_coverage_match": frame_coverage_match,
        "observer_proposals": observer_proposals,
        "active_proposals": active_proposals,
        "proposal_count_mismatch_frames": proposal_count_mismatch_frames,
        "proposal_coverage_match": proposal_coverage_match,
        "paired_protected_mismatches": protected_mismatches,
    }


def parse_eval_log(
    path: Path,
    prediction_root: Path,
    *,
    expected_scene_count: int,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": str(path),
        "status": "pending",
        "reason": None,
        "ap_percent": None,
        "reported_kept_scans": None,
        "mentions_prediction_root": None,
    }
    if not path.is_file():
        result["reason"] = "evaluation log does not exist yet"
        return result
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        result["reason"] = f"evaluation log is not readable yet: {exc}"
        return result
    root_spellings = {str(prediction_root), str(prediction_root.resolve())}
    result["mentions_prediction_root"] = any(
        spelling and spelling in text for spelling in root_spellings
    )
    kept = KEPT_SCANS_RE.findall(text)
    if kept:
        result["reported_kept_scans"] = {
            "kept": int(kept[-1][0]),
            "total": int(kept[-1][1]),
        }
    matches = EVAL_MAP_RE.findall(text)
    if len(matches) < 3:
        result["reason"] = (
            "evaluation has not emitted all AP15/AP25/AP50 values "
            f"({len(matches)}/3 found)"
        )
        return result
    try:
        values = [float(value) for value in matches[-3:]]
    except ValueError as exc:
        result["reason"] = f"could not parse evaluation values: {exc}"
        return result
    if not all(math.isfinite(value) for value in values):
        result["reason"] = "evaluation contains non-finite mAP"
        return result
    if result["mentions_prediction_root"] is not True:
        result["status"] = "invalid"
        result["reason"] = (
            "evaluation log is not bound to the requested prediction root"
        )
        return result
    reported_scans = result["reported_kept_scans"]
    if reported_scans is None:
        result["status"] = "invalid"
        result["reason"] = "evaluation log has no kept-scene receipt"
        return result
    if (
        reported_scans["kept"] != expected_scene_count
        or reported_scans["total"] != expected_scene_count
    ):
        result["status"] = "invalid"
        result["reason"] = (
            "evaluation scene receipt does not match the requested scene list: "
            f"{reported_scans['kept']}/{reported_scans['total']} vs "
            f"{expected_scene_count}/{expected_scene_count}"
        )
        return result
    # The official evaluator prints fractions.  Also accept logs which already
    # print percentage points, making the parser useful for compact receipts.
    if all(abs(value) <= 1.0 for value in values):
        values = [100.0 * value for value in values]
    result.update(
        {
            "status": "ready",
            "reason": None,
            "ap_percent": {
                "AP15": values[0],
                "AP25": values[1],
                "AP50": values[2],
            },
        }
    )
    return result


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    scenes = read_scene_list(args.scene_list)
    official_scenes = read_scene_list(DEFAULT_SCENE_LIST)
    issues = IssueCollector(max_samples=args.max_issue_samples)

    observer, observer_rows = summarize_arm(
        role="observer",
        prediction_root=args.observer_pred_root,
        diagnostic_root=args.observer_diagnostic_root,
        scenes=scenes,
        issues=issues,
    )
    active, active_rows = summarize_arm(
        role="active",
        prediction_root=args.active_pred_root,
        diagnostic_root=args.active_diagnostic_root,
        scenes=scenes,
        issues=issues,
    )
    pairing = pair_diagnostics(
        scenes=scenes,
        observer_rows=observer_rows,
        active_rows=active_rows,
        issues=issues,
    )
    observer_eval = parse_eval_log(
        args.observer_eval_log,
        args.observer_pred_root,
        expected_scene_count=len(scenes),
    )
    active_eval = parse_eval_log(
        args.active_eval_log,
        args.active_pred_root,
        expected_scene_count=len(scenes),
    )
    evaluation: Dict[str, Any] = {
        "observer": observer_eval,
        "active": active_eval,
        "delta_active_minus_observer_percent": None,
        "improves_any_threshold": None,
        "improves_all_thresholds": None,
        "non_regressive_all_thresholds": None,
    }
    if observer_eval["status"] == "ready" and active_eval["status"] == "ready":
        observer_ap = observer_eval["ap_percent"]
        active_ap = active_eval["ap_percent"]
        delta = {
            key: float(active_ap[key]) - float(observer_ap[key])
            for key in ("AP15", "AP25", "AP50")
        }
        evaluation.update(
            {
                "delta_active_minus_observer_percent": delta,
                "improves_any_threshold": any(value > 0.0 for value in delta.values()),
                "improves_all_thresholds": all(value > 0.0 for value in delta.values()),
                "non_regressive_all_thresholds": all(
                    value >= 0.0 for value in delta.values()
                ),
            }
        )

    artifact_complete = observer["artifact_complete"] and active["artifact_complete"]
    coverage_complete = (
        pairing["frame_coverage_match"] and pairing["proposal_coverage_match"]
    )
    eval_complete = (
        observer_eval["status"] == "ready" and active_eval["status"] == "ready"
    )
    if issues.total:
        status = "invalid"
        reason = "protected/pairing contract anomalies were found"
    elif not artifact_complete or not coverage_complete:
        status = "pending"
        reason = "paired prediction/diagnostic artifacts are not complete yet"
    elif (
        observer_eval["status"] == "invalid"
        or active_eval["status"] == "invalid"
    ):
        status = "invalid"
        reason = "one or both evaluation logs fail root/scene binding"
    elif not eval_complete:
        status = "pending"
        reason = "paired constant-score AP evaluation is not complete yet"
    else:
        status = "ready"
        reason = "paired artifacts, contracts, coverage, and AP logs are complete"

    return {
        "schema": "boxfusion.t05_boxer.paired_summary.v1",
        "status": status,
        "status_reason": reason,
        "protocol": {
            "scene_list": str(args.scene_list),
            "scene_count": len(scenes),
            "official100_reference": str(DEFAULT_SCENE_LIST),
            "official100_reference_count": len(official_scenes),
            "matches_official100_exactly": scenes == official_scenes,
            "official100_complete": (
                scenes == official_scenes
                and artifact_complete
                and coverage_complete
                and issues.total == 0
            ),
        },
        "arms": {"observer": observer, "active": active},
        "pairing": pairing,
        "protected_contract": {
            "ok_so_far": issues.total == 0,
            **issues.as_dict(),
        },
        "evaluation": evaluation,
    }


def format_number(value: Optional[float], digits: int = 3) -> str:
    return "pending" if value is None else f"{value:.{digits}f}"


def print_text_report(report: Mapping[str, Any]) -> None:
    protocol = report["protocol"]
    print(
        "T05 + Boxer paired summary | "
        f"status={report['status']} | scenes={protocol['scene_count']} | "
        f"official100={protocol['matches_official100_exactly']}"
    )
    print(f"Reason: {report['status_reason']}")
    print(
        "arm       pred/diag   boxes   calls   proposals   applied   "
        "observer_calls   runtime_p50/p95_ms"
    )
    for role in ("observer", "active"):
        arm = report["arms"][role]
        print(
            f"{role:<10} "
            f"{arm['prediction_files_present']:>3}/{arm['diagnostic_files_present']:<3} "
            f"{arm['box_count']:>7} "
            f"{arm['boxer_calls']:>7} "
            f"{arm['boxer_proposals']:>11} "
            f"{arm['boxer_applied']:>9} "
            f"{arm['boxer_observer_calls']:>16}   "
            f"{format_number(arm['runtime_p50_ms'])}/"
            f"{format_number(arm['runtime_p95_ms'])}"
        )
        if arm["missing_predictions"] or arm["missing_diagnostics"]:
            print(
                f"  {role} pending: missing predictions="
                f"{len(arm['missing_predictions'])}, diagnostics="
                f"{len(arm['missing_diagnostics'])}"
            )
    pairing = report["pairing"]
    print(
        "Pairing: "
        f"scenes={pairing['paired_diagnostic_scenes']}/{pairing['expected_scenes']}, "
        f"frames={pairing['observer_frame_attempts']}/"
        f"{pairing['active_frame_attempts']} "
        f"(match={pairing['frame_coverage_match']}), "
        f"proposals={pairing['observer_proposals']}/"
        f"{pairing['active_proposals']} "
        f"(match={pairing['proposal_coverage_match']})"
    )
    contract = report["protected_contract"]
    print(
        f"Protected contract: ok_so_far={contract['ok_so_far']}, "
        f"anomalies={contract['count']}"
    )
    if contract["count"]:
        print(f"  anomaly counts: {json.dumps(contract['by_kind'], sort_keys=True)}")
        for issue in contract["samples"]:
            print(f"  sample: {json.dumps(issue, sort_keys=True)}")

    evaluation = report["evaluation"]
    observer_eval = evaluation["observer"]
    active_eval = evaluation["active"]
    if observer_eval["status"] != "ready" or active_eval["status"] != "ready":
        print(
            "AP: pending | "
            f"observer={observer_eval['reason'] or observer_eval['status']}; "
            f"active={active_eval['reason'] or active_eval['status']}"
        )
        return
    observer_ap = observer_eval["ap_percent"]
    active_ap = active_eval["ap_percent"]
    delta = evaluation["delta_active_minus_observer_percent"]
    print("AP (percentage points):")
    print("arm          AP15      AP25      AP50")
    print(
        f"observer  {observer_ap['AP15']:8.4f} {observer_ap['AP25']:9.4f} "
        f"{observer_ap['AP50']:9.4f}"
    )
    print(
        f"active    {active_ap['AP15']:8.4f} {active_ap['AP25']:9.4f} "
        f"{active_ap['AP50']:9.4f}"
    )
    print(
        f"delta     {delta['AP15']:+8.4f} {delta['AP25']:+9.4f} "
        f"{delta['AP50']:+9.4f}"
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize paired T05+Boxer observer/active predictions, diagnostics, "
            "contracts, coverage, runtimes, and constant-score AP."
        )
    )
    parser.add_argument(
        "--observer-pred-root", "--observer-root", type=Path, required=True
    )
    parser.add_argument("--active-pred-root", "--active-root", type=Path, required=True)
    parser.add_argument(
        "--observer-diagnostic-root",
        "--observer-diagnostics",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--active-diagnostic-root",
        "--active-diagnostics",
        type=Path,
        required=True,
    )
    parser.add_argument("--observer-eval-log", type=Path, required=True)
    parser.add_argument("--active-eval-log", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, default=DEFAULT_SCENE_LIST)
    parser.add_argument(
        "--max-issue-samples",
        type=int,
        default=20,
        help="maximum anomaly examples retained in the report (default: 20)",
    )
    parser.add_argument(
        "--format", choices=("text", "json"), default="text"
    )
    args = parser.parse_args(argv)
    if args.max_issue_samples < 0:
        parser.error("--max-issue-samples must be >= 0")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = build_report(args)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text_report(report)
    # Pending and contract-invalid experiments are report states, not failures
    # of this read-only summarizer.  A nonzero exit is reserved for bad inputs.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
