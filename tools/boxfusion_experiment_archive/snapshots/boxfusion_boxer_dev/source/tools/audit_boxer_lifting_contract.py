#!/usr/bin/env python3
"""Audit X0/X1/X2 Boxer lifting ablation contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


PROTECTED_DIAGNOSTIC_KEYS = (
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


def read_scene_list(path: Path) -> List[str]:
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


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def compare_exact(left: Any, right: Any, path: str = "$") -> Optional[Dict[str, str]]:
    if isinstance(left, torch.Tensor):
        left = left.detach().cpu().numpy()
    if isinstance(right, torch.Tensor):
        right = right.detach().cpu().numpy()
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            return {
                "path": path,
                "message": f"type mismatch: {type(left)} != {type(right)}",
            }
        if left.dtype != right.dtype:
            return {
                "path": path,
                "message": f"dtype mismatch: {left.dtype} != {right.dtype}",
            }
        if left.shape != right.shape:
            return {
                "path": path,
                "message": f"shape mismatch: {left.shape} != {right.shape}",
            }
        if left.tobytes(order="C") != right.tobytes(order="C"):
            return {"path": path, "message": "array value bytes differ"}
        return None
    if type(left) is not type(right):
        return {
            "path": path,
            "message": f"type mismatch: {type(left)} != {type(right)}",
        }
    if isinstance(left, dict):
        if list(left.keys()) != list(right.keys()):
            return {"path": path, "message": "dictionary keys/order differ"}
        for key in left:
            issue = compare_exact(left[key], right[key], f"{path}[{key!r}]")
            if issue:
                return issue
        return None
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return {
                "path": path,
                "message": f"length mismatch: {len(left)} != {len(right)}",
            }
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            issue = compare_exact(
                left_item,
                right_item,
                f"{path}[{index}]",
            )
            if issue:
                return issue
        return None
    if left != right:
        return {
            "path": path,
            "message": f"value mismatch: {left!r} != {right!r}",
        }
    return None


def compare_with_array_atol(
    left: Any,
    right: Any,
    *,
    array_atol: float,
    path: str = "$",
) -> Tuple[Optional[Dict[str, Any]], float]:
    """Keep structure/scalars exact; allow only bounded numeric array drift."""

    if isinstance(left, torch.Tensor):
        left = left.detach().cpu().numpy()
    if isinstance(right, torch.Tensor):
        right = right.detach().cpu().numpy()
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            return (
                {
                    "path": path,
                    "message": f"type mismatch: {type(left)} != {type(right)}",
                },
                float("inf"),
            )
        if left.dtype != right.dtype or left.shape != right.shape:
            return (
                {
                    "path": path,
                    "message": (
                        f"array metadata mismatch: "
                        f"{left.dtype}/{left.shape} != {right.dtype}/{right.shape}"
                    ),
                },
                float("inf"),
            )
        if left.tobytes(order="C") == right.tobytes(order="C"):
            return None, 0.0
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            return (
                {"path": path, "message": "non-finite array value"},
                float("inf"),
            )
        maximum = float(
            np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))
        )
        if maximum > float(array_atol):
            return (
                {
                    "path": path,
                    "message": "array drift exceeds frozen absolute tolerance",
                    "max_abs": maximum,
                    "atol": float(array_atol),
                },
                maximum,
            )
        return None, maximum
    if type(left) is not type(right):
        return (
            {
                "path": path,
                "message": f"type mismatch: {type(left)} != {type(right)}",
            },
            float("inf"),
        )
    if isinstance(left, dict):
        if list(left.keys()) != list(right.keys()):
            return (
                {"path": path, "message": "dictionary keys/order differ"},
                float("inf"),
            )
        maximum = 0.0
        for key in left:
            issue, child_maximum = compare_with_array_atol(
                left[key],
                right[key],
                array_atol=array_atol,
                path=f"{path}[{key!r}]",
            )
            maximum = max(maximum, child_maximum)
            if issue:
                return issue, maximum
        return None, maximum
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return (
                {
                    "path": path,
                    "message": f"length mismatch: {len(left)} != {len(right)}",
                },
                float("inf"),
            )
        maximum = 0.0
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            issue, child_maximum = compare_with_array_atol(
                left_item,
                right_item,
                array_atol=array_atol,
                path=f"{path}[{index}]",
            )
            maximum = max(maximum, child_maximum)
            if issue:
                return issue, maximum
        return None, maximum
    if left != right:
        return (
            {
                "path": path,
                "message": f"value mismatch: {left!r} != {right!r}",
            },
            float("inf"),
        )
    return None, 0.0


DiagnosticKey = Tuple[int, str]


def load_diagnostics(path: Path) -> Dict[DiagnosticKey, Dict[str, Any]]:
    rows: Dict[DiagnosticKey, Dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema") != "boxfusion.boxer_lifting.frame.v1":
            raise ValueError(f"{path}:{line_number}: unexpected schema")
        frame_id = int(row["frame_id"])
        attempt_id = str(row.get("attempt_id", "primary"))
        key = (frame_id, attempt_id)
        if key in rows:
            raise ValueError(
                f"{path}: duplicate frame/attempt={frame_id}/{attempt_id}"
            )
        rows[key] = row
    if not rows:
        raise ValueError(f"No diagnostic rows: {path}")
    return rows


def diagnostic_row_issues(
    row: Dict[str, Any],
    *,
    role: str,
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    count = int(row.get("count", -1))
    if count < 0:
        return [{"kind": "invalid_diagnostic_count", "value": count}]
    expected_mode = "observer" if role == "observer" else "active"
    if row.get("mode") != expected_mode:
        issues.append(
            {
                "kind": "unexpected_diagnostic_mode",
                "expected": expected_mode,
                "actual": row.get("mode"),
            }
        )
    if count == 0:
        return issues
    length_fields = {
        "input_boxes_xyxy": count,
        "detector_scores": count,
        "confidence": count,
        "cutr_xyz_dims_camera": count,
        "cutr_rotation_camera_object": count,
        "output_xyz_dims_camera": count,
        "output_rotation_camera_object": count,
        "raw_params_voxel": count,
    }
    for field, expected_length in length_fields.items():
        value = row.get(field)
        if not isinstance(value, list) or len(value) != expected_length:
            issues.append(
                {
                    "kind": "diagnostic_length_mismatch",
                    "field": field,
                    "expected": expected_length,
                    "actual": len(value) if isinstance(value, list) else None,
                }
            )
            continue
        try:
            finite = bool(np.all(np.isfinite(np.asarray(value))))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            issues.append(
                {
                    "kind": "diagnostic_nonfinite",
                    "field": field,
                }
            )
    return issues


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    scenes = read_scene_list(args.scene_list)
    issues: List[Dict[str, Any]] = []
    identity_scenes = 0
    observer_frames = 0
    active_frames = 0
    observer_proposals = 0
    active_proposals = 0
    schedule_differences: List[Dict[str, Any]] = []
    exact_identity_scenes = 0
    observer_box_max_abs_drift = 0.0

    for scene in scenes:
        baseline_path = args.baseline_root / f"{scene}_boxes.pkl"
        observer_path = args.observer_root / f"{scene}_boxes.pkl"
        if not baseline_path.is_file() or not observer_path.is_file():
            issues.append(
                {
                    "scene": scene,
                    "kind": "missing_prediction",
                    "baseline": baseline_path.is_file(),
                    "observer": observer_path.is_file(),
                }
            )
            continue
        baseline_prediction = load_pickle(baseline_path)
        observer_prediction = load_pickle(observer_path)
        exact_mismatch = compare_exact(
            baseline_prediction,
            observer_prediction,
        )
        if exact_mismatch is None:
            exact_identity_scenes += 1
        mismatch, scene_maximum = compare_with_array_atol(
            baseline_prediction,
            observer_prediction,
            array_atol=args.observer_box_atol,
        )
        observer_box_max_abs_drift = max(
            observer_box_max_abs_drift,
            scene_maximum,
        )
        if mismatch:
            issues.append(
                {
                    "scene": scene,
                    "kind": "observer_prediction_mismatch",
                    **mismatch,
                }
            )
        else:
            identity_scenes += 1

        observer_diag_path = (
            args.observer_diagnostics / f"{scene}_boxer_lifting.jsonl"
        )
        if not observer_diag_path.is_file():
            issues.append(
                {
                    "scene": scene,
                    "kind": "missing_observer_diagnostic",
                    "path": str(observer_diag_path),
                }
            )
            continue
        observer_rows = load_diagnostics(observer_diag_path)
        observer_frames += len(observer_rows)
        observer_proposals += sum(int(row["count"]) for row in observer_rows.values())
        for (frame_id, attempt_id), row in observer_rows.items():
            for issue in diagnostic_row_issues(row, role="observer"):
                issues.append(
                    {
                        "scene": scene,
                        "frame_id": frame_id,
                        "attempt_id": attempt_id,
                        **issue,
                    }
                )
            if row.get("mutation_enabled") is not False:
                issues.append(
                    {
                        "scene": scene,
                        "frame_id": frame_id,
                        "attempt_id": attempt_id,
                        "kind": "observer_mutation_enabled",
                    }
                )
            if row.get("projected_center_replaced", False) is not False:
                issues.append(
                    {
                        "scene": scene,
                        "frame_id": frame_id,
                        "attempt_id": attempt_id,
                        "kind": "observer_projected_center_replaced",
                    }
                )
            if int(row.get("applied_count", -1)) != 0:
                issues.append(
                    {
                        "scene": scene,
                        "frame_id": frame_id,
                        "attempt_id": attempt_id,
                        "kind": "observer_applied_nonzero",
                        "value": row.get("applied_count"),
                    }
                )

        if args.proposal_cache_root is not None:
            cache_scene_root = args.proposal_cache_root / scene
            cache_source_root = (
                args.proposal_cache_source_root
                if args.proposal_cache_source_root is not None
                else args.baseline_root
            )
            cache_source_path = cache_source_root / f"{scene}_boxes.pkl"
            manifest_path = cache_scene_root / "manifest.json"
            if not manifest_path.is_file():
                issues.append(
                    {
                        "scene": scene,
                        "kind": "missing_proposal_cache_manifest",
                        "path": str(manifest_path),
                    }
                )
            else:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                records = manifest.get("records", [])
                cache_keys = [
                    (int(row["frame_id"]), str(row["attempt_id"]))
                    for row in records
                ]
                if manifest.get("schema") != "boxfusion.cutr_postfilter_cache.v2":
                    issues.append(
                        {
                            "scene": scene,
                            "kind": "proposal_cache_schema_mismatch",
                            "actual": manifest.get("schema"),
                        }
                    )
                if cache_keys != sorted(observer_rows):
                    issues.append(
                        {
                            "scene": scene,
                            "kind": "proposal_cache_schedule_mismatch",
                            "cache": cache_keys,
                            "observer": sorted(observer_rows),
                        }
                    )
                cache_proposals = sum(int(row["count"]) for row in records)
                observer_scene_proposals = sum(
                    int(row["count"]) for row in observer_rows.values()
                )
                if cache_proposals != observer_scene_proposals:
                    issues.append(
                        {
                            "scene": scene,
                            "kind": "proposal_cache_count_mismatch",
                            "cache": cache_proposals,
                            "observer": observer_scene_proposals,
                        }
                    )
                if (
                    not cache_source_path.is_file()
                    or manifest.get("prediction_file") != cache_source_path.name
                    or manifest.get("prediction_sha256")
                    != sha256_file(cache_source_path)
                ):
                    issues.append(
                        {
                            "scene": scene,
                            "kind": "proposal_cache_baseline_mismatch",
                        }
                    )
                for record in records:
                    payload_path = cache_scene_root / (
                        f"frame_{int(record['frame_id']):06d}.pt"
                    )
                    if (
                        not payload_path.is_file()
                        or sha256_file(payload_path) != record.get("sha256")
                    ):
                        issues.append(
                            {
                                "scene": scene,
                                "frame_id": int(record["frame_id"]),
                                "kind": "proposal_cache_payload_mismatch",
                            }
                        )

        if args.active_root is None:
            continue
        active_path = args.active_root / f"{scene}_boxes.pkl"
        active_diag_path = (
            args.active_diagnostics / f"{scene}_boxer_lifting.jsonl"
        )
        if not active_path.is_file() or not active_diag_path.is_file():
            issues.append(
                {
                    "scene": scene,
                    "kind": "missing_active_artifact",
                    "prediction": active_path.is_file(),
                    "diagnostic": active_diag_path.is_file(),
                }
            )
            continue
        active_rows = load_diagnostics(active_diag_path)
        active_frames += len(active_rows)
        active_proposals += sum(int(row["count"]) for row in active_rows.values())
        observer_keys = set(observer_rows)
        active_keys = set(active_rows)
        if observer_keys != active_keys:
            difference = {
                "scene": scene,
                "kind": "frame_schedule_mismatch",
                "observer_only": sorted(observer_keys - active_keys),
                "active_only": sorted(active_keys - observer_keys),
            }
            schedule_differences.append(difference)
            schedule_change_is_retry_only = all(
                attempt_id == "retry"
                for _, attempt_id in observer_keys.symmetric_difference(
                    active_keys
                )
            )
            if (
                not args.allow_active_schedule_change
                or not schedule_change_is_retry_only
            ):
                issues.append(difference)

        for (frame_id, attempt_id), active_row in active_rows.items():
            for issue in diagnostic_row_issues(active_row, role="active"):
                issues.append(
                    {
                        "scene": scene,
                        "frame_id": frame_id,
                        "attempt_id": attempt_id,
                        **issue,
                    }
                )
            if active_row.get("mutation_enabled") is not True:
                issues.append(
                    {
                        "scene": scene,
                        "frame_id": frame_id,
                        "attempt_id": attempt_id,
                        "kind": "active_mutation_disabled",
                    }
                )
            if int(active_row.get("count", 0)) > 0:
                expected_projection_replacement = (
                    active_row.get("apply_stage") == "pre_filter"
                )
                if bool(
                    active_row.get("projected_center_replaced", False)
                ) != expected_projection_replacement:
                    issues.append(
                        {
                            "scene": scene,
                            "frame_id": frame_id,
                            "attempt_id": attempt_id,
                            "kind": "projected_center_contract_mismatch",
                            "expected": expected_projection_replacement,
                            "actual": active_row.get(
                                "projected_center_replaced"
                            ),
                        }
                    )
            if int(active_row.get("applied_count", -1)) != int(
                active_row.get("count", -2)
            ):
                issues.append(
                    {
                        "scene": scene,
                        "frame_id": frame_id,
                        "attempt_id": attempt_id,
                        "kind": "active_coverage_incomplete",
                        "count": active_row.get("count"),
                        "applied_count": active_row.get("applied_count"),
                    }
                )

        # The primary score-filtered CuTR rows must pair exactly.  A full
        # pre-filter replacement may naturally change whether the unchanged
        # first-frame low-threshold retry is reached; retry-only differences
        # are reported above and are allowed only with the explicit flag.
        for key in sorted(observer_keys & active_keys):
            frame_id, attempt_id = key
            observer_row = observer_rows[key]
            active_row = active_rows[key]
            if observer_row.get("apply_stage") != active_row.get("apply_stage"):
                issues.append(
                    {
                        "scene": scene,
                        "frame_id": frame_id,
                        "attempt_id": attempt_id,
                        "kind": "apply_stage_mismatch",
                        "observer": observer_row.get("apply_stage"),
                        "active": active_row.get("apply_stage"),
                    }
                )
            for key in PROTECTED_DIAGNOSTIC_KEYS:
                if observer_row.get(key) != active_row.get(key):
                    issues.append(
                        {
                            "scene": scene,
                            "frame_id": frame_id,
                            "attempt_id": attempt_id,
                            "kind": "paired_contract_mismatch",
                            "key": key,
                            "observer": observer_row.get(key),
                            "active": active_row.get(key),
                        }
                    )

    report = {
        "schema": "boxfusion.boxer_lifting.contract_audit.v1",
        "ok": not issues,
        "scene_count": len(scenes),
        "identity_scenes": identity_scenes,
        "exact_identity_scenes": exact_identity_scenes,
        "observer_box_atol": float(args.observer_box_atol),
        "observer_box_max_abs_drift": observer_box_max_abs_drift,
        "observer_identity_rule": (
            "structure/classes/scores/order exact; numeric box arrays "
            "absolute drift <= observer_box_atol; AP checked exactly "
            "by the paired summary"
        ),
        "observer_frames": observer_frames,
        "observer_proposals": observer_proposals,
        "active_frames": active_frames,
        "active_proposals": active_proposals,
        "schedule_differences": schedule_differences,
        "active_schedule_change_allowed": bool(
            args.allow_active_schedule_change
        ),
        "baseline_root": str(args.baseline_root),
        "observer_root": str(args.observer_root),
        "active_root": str(args.active_root) if args.active_root else None,
        "proposal_cache_root": (
            str(args.proposal_cache_root)
            if args.proposal_cache_root is not None
            else None
        ),
        "proposal_cache_source_root": (
            str(args.proposal_cache_source_root)
            if args.proposal_cache_source_root is not None
            else None
        ),
        "issues": issues,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--observer-root", type=Path, required=True)
    parser.add_argument("--observer-diagnostics", type=Path, required=True)
    parser.add_argument("--active-root", type=Path)
    parser.add_argument("--active-diagnostics", type=Path)
    parser.add_argument("--proposal-cache-root", type=Path)
    parser.add_argument("--proposal-cache-source-root", type=Path)
    parser.add_argument(
        "--observer-box-atol",
        type=float,
        default=1e-4,
        help=(
            "Frozen meter-scale tolerance for upstream PyCUDA atomicAdd "
            "fusion noise; all non-array fields remain exact."
        ),
    )
    parser.add_argument(
        "--allow-active-schedule-change",
        action="store_true",
        help=(
            "Allow only retry-attempt schedule differences caused by a "
            "pre-filter active replacement."
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.observer_box_atol < 0.0 or args.observer_box_atol > 1e-3:
        parser.error("--observer-box-atol must be in [0, 1e-3]")
    if (args.active_root is None) != (args.active_diagnostics is None):
        parser.error(
            "--active-root and --active-diagnostics must be supplied together"
        )

    report = audit(args)
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
