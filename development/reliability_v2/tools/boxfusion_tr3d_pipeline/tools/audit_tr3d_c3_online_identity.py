#!/usr/bin/env python3
"""Aggregate GT-free C3 online identity diagnostics.

The audit fails closed on missing scenes, prediction mutation, non-zero apply
counts, lineage/route drift, or optional paired prediction byte mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
from typing import Any, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.tr3d_c3_online_identity import PARENT_SCORE_ROUTE, ROUTE, SCHEMA
from tools.tr3d_data import read_scene_list


REPORT_SCHEMA = "boxfusion.tr3d_c3_online_identity_audit.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_create_only(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"refusing existing audit report: {path}") from error
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _load_prediction(path: Path) -> list[tuple[int, np.ndarray, float]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - local experiment artifact
    if type(payload) is not list or len(payload) != 1 or type(payload[0]) is not list:
        raise ValueError(f"{path}: non-canonical prediction container")
    rows: list[tuple[int, np.ndarray, float]] = []
    for index, raw in enumerate(payload[0]):
        if type(raw) is not tuple or len(raw) != 3 or type(raw[0]) is not int:
            raise ValueError(f"{path}: malformed row {index}")
        geometry = np.asarray(raw[1])
        score = float(raw[2])
        if (
            raw[0] != 0
            or geometry.shape != (8, 3)
            or geometry.dtype != np.float32
            or not np.isfinite(geometry).all()
            or not np.isfinite(score)
        ):
            raise ValueError(f"{path}: invalid canonical row {index}")
        rows.append((raw[0], geometry, score))
    return rows


def _paired_prediction_drift(
    baseline: Path,
    observer: Path,
) -> dict[str, Any]:
    exact_bytes = _sha256_file(baseline) == _sha256_file(observer)
    left = _load_prediction(baseline)
    right = _load_prediction(observer)
    if len(left) != len(right):
        raise ValueError("paired prediction count differs")
    geometry_max = 0.0
    score_max = 0.0
    for index, (before, after) in enumerate(zip(left, right)):
        if before[0] != after[0]:
            raise ValueError(f"paired prediction label differs at row {index}")
        geometry_max = max(
            geometry_max,
            float(np.max(np.abs(before[1] - after[1]), initial=0.0)),
        )
        score_max = max(score_max, abs(float(before[2] - after[2])))
    return {
        "bytes_equal": exact_bytes,
        "row_count": len(left),
        "geometry_max_abs_delta": geometry_max,
        "score_max_abs_delta": score_max,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    scene_list = args.scene_list.resolve()
    scenes = read_scene_list(scene_list)
    diagnostics_root = args.diagnostics_root.resolve()
    paired = (args.baseline_prediction_root, args.observer_prediction_root)
    if (
        not np.isfinite(args.paired_geometry_atol)
        or args.paired_geometry_atol < 0.0
        or not np.isfinite(args.paired_score_atol)
        or args.paired_score_atol < 0.0
    ):
        raise ValueError("paired prediction tolerances must be finite and nonnegative")
    if any(value is not None for value in paired) and not all(
        value is not None for value in paired
    ):
        raise ValueError("paired prediction roots must be supplied together")

    totals = {
        "candidates": 0,
        "joined": 0,
        "missing": 0,
        "frozen_selected": 0,
        "online_selected": 0,
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "out_of_universe": 0,
        "frames": 0,
        "proposals": 0,
        "runtime_s": 0.0,
        "prediction_bytes_equal": 0,
        "prediction_numeric_equal": 0,
    }
    scene_rows: list[dict[str, Any]] = []
    runtime_ms: list[float] = []
    parent_hashes: dict[str, str] = {}
    paired_drift_rows: list[dict[str, Any]] = []
    observed_route: str | None = None
    for scene_id in scenes:
        path = diagnostics_root / f"{scene_id}_c3_online_identity.json"
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            row.get("schema") != SCHEMA
            or not row.get("complete")
            or not row.get("observer_only")
            or row.get("mutation_enabled")
            or int(row.get("applied_count", -1)) != 0
            or row.get("scene_id") != scene_id
            or row.get("route") not in {ROUTE, PARENT_SCORE_ROUTE}
            or row.get("ground_truth_access")
            or row.get("clip_access")
            or not row.get("clip_semantics_unchanged")
            or row.get("teacher_labels_used_for_gate")
            or not row.get("prediction_identity")
            or row.get("prediction_state_before_sha256")
            != row.get("prediction_state_after_sha256")
            or float(row.get("identity_coverage", -1.0)) != 1.0
            or int(row.get("missing_identity_count", -1)) != 0
        ):
            raise ValueError(f"{path}: C3 online safety/identity contract failed")
        if observed_route is None:
            observed_route = str(row["route"])
        elif row.get("route") != observed_route:
            raise ValueError("C3 online audit refuses mixed candidate routes")
        candidate_rows = row.get("candidates")
        if not isinstance(candidate_rows, list) or len(candidate_rows) != int(
            row["candidate_count"]
        ):
            raise ValueError(f"{path}: candidate rows/count mismatch")
        keys = [item.get("identity_key") for item in candidate_rows]
        if any(not isinstance(value, str) or not value for value in keys):
            raise ValueError(f"{path}: malformed candidate identity key")
        if len(keys) != len(set(keys)):
            raise ValueError(f"{path}: duplicate candidate identity")
        parent_hashes[scene_id] = str(row["parent_cache_sha256"])

        for total_key, row_key in (
            ("candidates", "candidate_count"),
            ("joined", "exact_identity_joined_count"),
            ("missing", "missing_identity_count"),
            ("frozen_selected", "frozen_selected_count"),
            ("online_selected", "online_selected_count"),
            ("tp", "true_positive_count"),
            ("tn", "true_negative_count"),
            ("fp", "false_positive_count"),
            ("fn", "false_negative_count"),
            ("out_of_universe", "out_of_universe_selected_count"),
            ("frames", "provider_calls_observed"),
            ("proposals", "provider_proposals_observed"),
        ):
            totals[total_key] += int(row[row_key])
        totals["runtime_s"] += float(row["observer_runtime_s"])
        runtime_ms.append(float(row["observer_mean_runtime_ms_per_provider_call"]))

        prediction_equal = None
        prediction_numeric_equal = None
        if args.baseline_prediction_root is not None:
            baseline = args.baseline_prediction_root.resolve() / f"{scene_id}_boxes.pkl"
            observer = args.observer_prediction_root.resolve() / f"{scene_id}_boxes.pkl"
            if not baseline.is_file() or not observer.is_file():
                raise FileNotFoundError(f"paired prediction missing for {scene_id}")
            drift = _paired_prediction_drift(baseline, observer)
            prediction_equal = bool(drift["bytes_equal"])
            prediction_numeric_equal = bool(
                drift["geometry_max_abs_delta"] <= args.paired_geometry_atol
                and drift["score_max_abs_delta"] <= args.paired_score_atol
            )
            if not prediction_numeric_equal:
                raise ValueError(
                    f"{scene_id}: observer/control drift exceeds explicit "
                    f"tolerances: {drift}"
                )
            paired_drift_rows.append({"scene_id": scene_id, **drift})
            totals["prediction_bytes_equal"] += int(prediction_equal)
            totals["prediction_numeric_equal"] += 1
        scene_rows.append(
            {
                "scene_id": scene_id,
                "diagnostic": str(path.resolve()),
                "diagnostic_sha256": _sha256_file(path),
                "candidate_count": int(row["candidate_count"]),
                "frozen_selected_count": int(row["frozen_selected_count"]),
                "online_selected_count": int(row["online_selected_count"]),
                "tp": int(row["true_positive_count"]),
                "tn": int(row["true_negative_count"]),
                "fp": int(row["false_positive_count"]),
                "fn": int(row["false_negative_count"]),
                "exact_set": bool(row["scene_exact_set"]),
                "prediction_bytes_equal": prediction_equal,
                "prediction_numeric_equal_within_explicit_tolerance": (
                    prediction_numeric_equal
                ),
            }
        )

    candidate_count = totals["candidates"]
    frozen_count = totals["frozen_selected"]
    online_count = totals["online_selected"]
    agreement = totals["tp"] + totals["tn"]
    union = totals["tp"] + totals["fp"] + totals["fn"]
    runtime = np.asarray(runtime_ms, dtype=np.float64)
    report = {
        "schema": REPORT_SCHEMA,
        "complete": True,
        "pass": True,
        "observer_only": True,
        "mutation_enabled": False,
        "applied_count": 0,
        "ground_truth_access": False,
        "route": observed_route,
        "comparison_scope": (
            "frozen_c1_top5_online_yoloe_c2_gate"
            if observed_route == ROUTE
            else "parent_score_top5_online_yoloe_depth_gate"
        ),
        "scene_list": str(scene_list),
        "scene_list_sha256": _sha256_file(scene_list),
        "scene_count": len(scenes),
        "candidate_count": candidate_count,
        "exact_identity_joined_count": totals["joined"],
        "missing_identity_count": totals["missing"],
        "identity_coverage": (
            float(totals["joined"] / candidate_count) if candidate_count else 1.0
        ),
        "frozen_selected_count": frozen_count,
        "online_selected_count": online_count,
        "true_positive_count": totals["tp"],
        "true_negative_count": totals["tn"],
        "false_positive_count": totals["fp"],
        "false_negative_count": totals["fn"],
        "out_of_universe_selected_count": totals["out_of_universe"],
        "route_precision": (
            float(totals["tp"] / online_count) if online_count else 0.0
        ),
        "route_coverage_recall": (
            float(totals["tp"] / frozen_count) if frozen_count else 0.0
        ),
        "decision_agreement_conditional": (
            float(agreement / totals["joined"]) if totals["joined"] else 1.0
        ),
        "decision_agreement_e2e": (
            float(agreement / candidate_count) if candidate_count else 1.0
        ),
        "selection_jaccard": float(totals["tp"] / union) if union else 1.0,
        "scene_exact_set_count": sum(int(row["exact_set"]) for row in scene_rows),
        "scene_exact_set_rate": (
            float(sum(int(row["exact_set"]) for row in scene_rows) / len(scenes))
            if scenes
            else 1.0
        ),
        "provider_calls_observed": totals["frames"],
        "provider_proposals_observed": totals["proposals"],
        "observer_runtime_s": totals["runtime_s"],
        "observer_scene_mean_ms_per_call": {
            "mean": float(runtime.mean()) if runtime.size else 0.0,
            "p50": float(np.quantile(runtime, 0.50)) if runtime.size else 0.0,
            "p95": float(np.quantile(runtime, 0.95)) if runtime.size else 0.0,
            "max": float(runtime.max()) if runtime.size else 0.0,
        },
        "paired_prediction_bytes_checked": args.baseline_prediction_root is not None,
        "paired_prediction_bytes_equal_count": totals["prediction_bytes_equal"],
        "paired_prediction_numeric_equal_count": totals["prediction_numeric_equal"],
        "paired_prediction_geometry_atol": args.paired_geometry_atol,
        "paired_prediction_score_atol": args.paired_score_atol,
        "paired_prediction_drift": paired_drift_rows,
        "parent_cache_sha256_by_scene": parent_hashes,
        "scenes": scene_rows,
    }
    _write_create_only(args.output.resolve(), report)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--scene-list", type=Path, required=True)
    value.add_argument("--diagnostics-root", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--baseline-prediction-root", type=Path)
    value.add_argument("--observer-prediction-root", type=Path)
    value.add_argument("--paired-geometry-atol", type=float, default=0.0)
    value.add_argument("--paired-score-atol", type=float, default=0.0)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    report = audit(parser().parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
