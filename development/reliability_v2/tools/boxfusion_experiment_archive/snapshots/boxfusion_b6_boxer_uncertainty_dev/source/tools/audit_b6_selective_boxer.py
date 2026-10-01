#!/usr/bin/env python3
"""Audit the frozen-B6 / observer / Selective-Boxer paired experiment.

The audit is intentionally independent of ScanNet ground truth.  It proves
that the observer cannot change B6 predictions and that the active run only
replaces rows accepted by the pre-registered camera-frame geometry gate.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


FRAME_SCHEMA = "boxfusion.boxer_lifting.frame.v1"
REPORT_SCHEMA = "boxfusion.b6_selective_boxer.audit.v1"
EXPECTED_GATE = {
    "max_center_shift_m": 0.10,
    "min_volume_ratio": 0.50,
    "max_volume_ratio": 2.00,
}
PAIRED_KEYS = (
    "count",
    "apply_stage",
    "selective_gate_enabled",
    "selective_gate",
    "protected_hashes",
    "input_pred_proj_xy_sha256",
    "scores_sha256",
    "boxes_2d_sha256",
    "image_sha256",
    "depth_sha256",
    "image_intrinsics_sha256",
    "depth_intrinsics_sha256",
    "camera_to_world_sha256",
    "use_sdp",
    "sdp_seed",
    "cutr_geometry_sha256",
    "boxer_geometry_sha256",
    "boxer_checkpoint_sha256",
    "boxer_commit",
    "gate_accepted",
    "gate_reasons",
    "gate_rejection_counts",
    "center_shift_m",
    "volume_ratio",
    "cutr_xyz_dims_camera",
    "cutr_rotation_camera_object",
    "boxer_xyz_dims_camera",
    "boxer_rotation_camera_object",
    "selective_xyz_dims_camera",
    "selective_rotation_camera_object",
)


def _read_scenes(path: Path) -> List[str]:
    scenes = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not scenes:
        raise ValueError(f"Empty scene list: {path}")
    if len(scenes) != len(set(scenes)):
        raise ValueError(f"Duplicate scene IDs: {path}")
    return scenes


def _as_numpy(value: Any) -> Optional[np.ndarray]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return None


def _compare_nested(
    left: Any,
    right: Any,
    *,
    array_atol: float,
    scalar_atol: float,
    path: str = "$",
) -> Tuple[Optional[Dict[str, Any]], float]:
    """Compare structure/scalars exactly and numeric arrays with a bound."""

    left_array = _as_numpy(left)
    right_array = _as_numpy(right)
    if left_array is not None or right_array is not None:
        if left_array is None or right_array is None:
            return {"path": path, "message": "array/non-array type mismatch"}, float("inf")
        if left_array.dtype != right_array.dtype or left_array.shape != right_array.shape:
            return {
                "path": path,
                "message": "array metadata mismatch",
                "left": f"{left_array.dtype}/{left_array.shape}",
                "right": f"{right_array.dtype}/{right_array.shape}",
            }, float("inf")
        if left_array.tobytes(order="C") == right_array.tobytes(order="C"):
            return None, 0.0
        if not np.issubdtype(left_array.dtype, np.number):
            return {"path": path, "message": "non-numeric array bytes differ"}, float("inf")
        if not np.all(np.isfinite(left_array)) or not np.all(np.isfinite(right_array)):
            return {"path": path, "message": "non-finite array value"}, float("inf")
        maximum = float(
            np.max(
                np.abs(
                    left_array.astype(np.float64)
                    - right_array.astype(np.float64)
                )
            )
        )
        if maximum > array_atol:
            return {
                "path": path,
                "message": "array drift exceeds tolerance",
                "max_abs": maximum,
                "atol": array_atol,
            }, maximum
        return None, maximum
    if type(left) is not type(right):
        return {
            "path": path,
            "message": f"type mismatch: {type(left).__name__} != {type(right).__name__}",
        }, float("inf")
    if isinstance(left, dict):
        if list(left) != list(right):
            return {"path": path, "message": "dictionary keys/order differ"}, float("inf")
        maximum = 0.0
        for key in left:
            issue, child_maximum = _compare_nested(
                left[key],
                right[key],
                array_atol=array_atol,
                scalar_atol=scalar_atol,
                path=f"{path}[{key!r}]",
            )
            maximum = max(maximum, child_maximum)
            if issue:
                return issue, maximum
        return None, maximum
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return {"path": path, "message": "sequence length differs"}, float("inf")
        maximum = 0.0
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            issue, child_maximum = _compare_nested(
                left_item,
                right_item,
                array_atol=array_atol,
                scalar_atol=scalar_atol,
                path=f"{path}[{index}]",
            )
            maximum = max(maximum, child_maximum)
            if issue:
                return issue, maximum
        return None, maximum
    if isinstance(left, (float, np.floating)) and isinstance(
        right, (float, np.floating)
    ):
        if not np.isfinite(left) or not np.isfinite(right):
            if left == right:
                return None, 0.0
            return {"path": path, "message": "non-finite scalar differs"}, float("inf")
        difference = abs(float(left) - float(right))
        if difference <= scalar_atol:
            return None, difference
        return {
            "path": path,
            "message": "floating scalar drift exceeds tolerance",
            "max_abs": difference,
            "atol": scalar_atol,
        }, difference
    if left != right:
        return {"path": path, "message": f"value mismatch: {left!r} != {right!r}"}, float("inf")
    return None, 0.0


def _load_prediction(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


DiagnosticKey = Tuple[int, str]


def _load_diagnostics(path: Path) -> Dict[DiagnosticKey, Dict[str, Any]]:
    rows: Dict[DiagnosticKey, Dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema") != FRAME_SCHEMA:
            raise ValueError(f"{path}:{line_number}: unexpected schema")
        key = (int(row["frame_id"]), str(row.get("attempt_id", "primary")))
        if key in rows:
            raise ValueError(f"{path}: duplicate diagnostic key {key}")
        rows[key] = row
    if not rows:
        raise ValueError(f"No diagnostic rows: {path}")
    return rows


def _numeric(value: Any, *, shape0: int, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 1 or array.shape[0] != shape0:
        raise ValueError(f"{field}: expected first dimension {shape0}, got {array.shape}")
    return array


def _append(issues: List[Dict[str, Any]], context: Dict[str, Any], kind: str, **values: Any) -> None:
    issues.append({**context, "kind": kind, **values})


def _validate_row(
    row: Dict[str, Any],
    *,
    role: str,
    context: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    count = int(row.get("count", -1))
    expected_mode = "observer" if role == "observer" else "active"
    if count < 0:
        _append(issues, context, "invalid_count", actual=count)
        return
    if row.get("mode") != expected_mode:
        _append(issues, context, "mode_mismatch", expected=expected_mode, actual=row.get("mode"))
    if row.get("apply_stage") != "post_filter":
        _append(issues, context, "apply_stage_mismatch", actual=row.get("apply_stage"))
    if row.get("selective_gate_enabled") is not True:
        _append(issues, context, "selective_gate_disabled")
    if row.get("selective_gate") != EXPECTED_GATE:
        _append(issues, context, "gate_threshold_mismatch", actual=row.get("selective_gate"))
    expected_mutation = role == "active"
    if row.get("mutation_enabled") is not expected_mutation:
        _append(issues, context, "mutation_contract_mismatch", expected=expected_mutation)

    eligible = int(row.get("eligible_count", -1))
    applied = int(row.get("applied_count", -1))
    fallback = int(row.get("fallback_count", -1))
    if not (0 <= eligible <= count):
        _append(issues, context, "invalid_eligible_count", count=count, eligible=eligible)
    if fallback != count - eligible:
        _append(issues, context, "fallback_count_mismatch", expected=count - eligible, actual=fallback)
    expected_applied = 0 if role == "observer" else eligible
    if applied != expected_applied:
        _append(issues, context, "applied_count_mismatch", expected=expected_applied, actual=applied)
    if row.get("projected_center_replaced", False) is not False:
        _append(issues, context, "post_filter_projection_changed")
    if count == 0:
        return

    try:
        accepted = np.asarray(row["gate_accepted"], dtype=bool)
        if accepted.shape != (count,):
            raise ValueError(f"gate_accepted shape={accepted.shape}")
        if int(accepted.sum()) != eligible:
            _append(issues, context, "accepted_mask_count_mismatch")
        reasons = row["gate_reasons"]
        if not isinstance(reasons, list) or len(reasons) != count:
            raise ValueError("gate_reasons length mismatch")
        cutr_xyz = _numeric(row["cutr_xyz_dims_camera"], shape0=count, field="cutr_xyz")
        cutr_rot = _numeric(row["cutr_rotation_camera_object"], shape0=count, field="cutr_rot")
        boxer_xyz = _numeric(row["boxer_xyz_dims_camera"], shape0=count, field="boxer_xyz")
        boxer_rot = _numeric(row["boxer_rotation_camera_object"], shape0=count, field="boxer_rot")
        selective_xyz = _numeric(row["selective_xyz_dims_camera"], shape0=count, field="selective_xyz")
        selective_rot = _numeric(row["selective_rotation_camera_object"], shape0=count, field="selective_rot")
        actual_xyz = _numeric(row["actual_xyz_dims_camera"], shape0=count, field="actual_xyz")
        actual_rot = _numeric(row["actual_rotation_camera_object"], shape0=count, field="actual_rot")
    except (KeyError, TypeError, ValueError) as error:
        _append(issues, context, "malformed_geometry_diagnostic", error=str(error))
        return

    expected_selective_xyz = np.where(accepted[:, None], boxer_xyz, cutr_xyz)
    expected_selective_rot = np.where(accepted[:, None, None], boxer_rot, cutr_rot)
    if not np.array_equal(selective_xyz, expected_selective_xyz, equal_nan=True):
        _append(issues, context, "selective_xyz_row_contract_broken")
    if not np.array_equal(selective_rot, expected_selective_rot, equal_nan=True):
        _append(issues, context, "selective_rotation_row_contract_broken")
    expected_actual_xyz = cutr_xyz if role == "observer" else selective_xyz
    expected_actual_rot = cutr_rot if role == "observer" else selective_rot
    if not np.array_equal(actual_xyz, expected_actual_xyz, equal_nan=True):
        _append(issues, context, "actual_xyz_contract_broken")
    if not np.array_equal(actual_rot, expected_actual_rot, equal_nan=True):
        _append(issues, context, "actual_rotation_contract_broken")
    expected_hash = (
        row.get("cutr_geometry_sha256")
        if role == "observer"
        else row.get("actual_geometry_sha256")
    )
    if role == "observer" and row.get("actual_geometry_sha256") != expected_hash:
        _append(issues, context, "observer_geometry_hash_changed")


def _sum(rows: Iterable[Dict[str, Any]], key: str) -> int:
    return sum(int(row.get(key, 0)) for row in rows)


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    scenes = _read_scenes(args.scene_list)
    issues: List[Dict[str, Any]] = []
    observer_frames = active_frames = 0
    proposals = eligible = applied = fallback = 0
    identity_scenes = 0
    identity_max_abs = 0.0
    rejection_totals: Dict[str, int] = {}

    for scene in scenes:
        paths = {
            "control_prediction": args.control_root / f"{scene}_boxes.pkl",
            "observer_prediction": args.observer_root / f"{scene}_boxes.pkl",
            "observer_diagnostic": args.observer_diagnostics / f"{scene}_boxer_lifting.jsonl",
            "active_prediction": args.active_root / f"{scene}_boxes.pkl",
            "active_diagnostic": args.active_diagnostics / f"{scene}_boxer_lifting.jsonl",
        }
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            _append(issues, {"scene": scene}, "missing_artifact", names=missing)
            continue

        issue, maximum = _compare_nested(
            _load_prediction(paths["control_prediction"]),
            _load_prediction(paths["observer_prediction"]),
            array_atol=args.observer_box_atol,
            scalar_atol=args.observer_score_atol,
        )
        identity_max_abs = max(identity_max_abs, maximum)
        if issue:
            _append(issues, {"scene": scene}, "observer_prediction_mismatch", **issue)
        else:
            identity_scenes += 1

        observer_rows = _load_diagnostics(paths["observer_diagnostic"])
        active_rows = _load_diagnostics(paths["active_diagnostic"])
        observer_frames += len(observer_rows)
        active_frames += len(active_rows)
        if set(observer_rows) != set(active_rows):
            _append(
                issues,
                {"scene": scene},
                "diagnostic_schedule_mismatch",
                observer_only=sorted(set(observer_rows) - set(active_rows)),
                active_only=sorted(set(active_rows) - set(observer_rows)),
            )

        for key, observer_row in observer_rows.items():
            context = {"scene": scene, "frame_id": key[0], "attempt_id": key[1]}
            _validate_row(observer_row, role="observer", context=context, issues=issues)
        for key, active_row in active_rows.items():
            context = {"scene": scene, "frame_id": key[0], "attempt_id": key[1]}
            _validate_row(active_row, role="active", context=context, issues=issues)
            proposals += int(active_row.get("count", 0))
            eligible += int(active_row.get("eligible_count", 0))
            applied += int(active_row.get("applied_count", 0))
            fallback += int(active_row.get("fallback_count", 0))
            for reason, count in active_row.get("gate_rejection_counts", {}).items():
                rejection_totals[reason] = rejection_totals.get(reason, 0) + int(count)

        for key in sorted(set(observer_rows) & set(active_rows)):
            observer_row = observer_rows[key]
            active_row = active_rows[key]
            context = {"scene": scene, "frame_id": key[0], "attempt_id": key[1]}
            for field in PAIRED_KEYS:
                if observer_row.get(field) != active_row.get(field):
                    _append(issues, context, "paired_diagnostic_mismatch", field=field)

    return {
        "schema": REPORT_SCHEMA,
        "ok": not issues,
        "scene_count": len(scenes),
        "observer_identity_scenes": identity_scenes,
        "observer_box_atol": float(args.observer_box_atol),
        "observer_score_atol": float(args.observer_score_atol),
        "observer_box_max_abs_drift": identity_max_abs,
        "observer_frames": observer_frames,
        "active_frames": active_frames,
        "active_proposals": proposals,
        "selective_eligible": eligible,
        "selective_applied": applied,
        "cutr_fallback": fallback,
        "selective_acceptance_rate": (eligible / proposals if proposals else 0.0),
        "gate_rejection_totals": rejection_totals,
        "frozen_gate": EXPECTED_GATE,
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--observer-root", type=Path, required=True)
    parser.add_argument("--observer-diagnostics", type=Path, required=True)
    parser.add_argument("--active-root", type=Path, required=True)
    parser.add_argument("--active-diagnostics", type=Path, required=True)
    parser.add_argument("--observer-box-atol", type=float, default=1e-4)
    parser.add_argument("--observer-score-atol", type=float, default=1e-6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0.0 <= args.observer_box_atol <= 1e-3:
        parser.error("--observer-box-atol must be in [0, 1e-3]")
    if not 0.0 <= args.observer_score_atol <= 1e-5:
        parser.error("--observer-score-atol must be in [0, 1e-5]")
    report = audit(args)
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
