#!/usr/bin/env python3
"""Fail-closed same-run identity audit for the CA-1M P2 B6 observer.

P2 may collect Mask-RGBD/object-memory diagnostics only.  Its authoritative
identity comparison is the pre-finalize snapshot and post-finalize output
from the *same* process.  The historical P1 run is retained only as a replay
drift and metric reference because independent GPU fusion is not bitwise
deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.quality_score import QUALITY_FEATURE_NAMES
from boxfusion.online_refinement import corners_to_center_size


DIAGNOSTIC_KEYS = {
    "scene_id",
    "boxes",
    "scores",
    "quality_features",
    "points",
    "point_mask",
    "source_indices",
    "track_ids",
    "result_indices",
    "labels",
    "quality_feature_names",
    "summary_json",
}

BOXER_STABLE_KEYS = (
    "schema",
    "scene_id",
    "frame_id",
    "attempt_id",
    "apply_stage",
    "mode",
    "mutation_enabled",
    "selective_gate_enabled",
    "selective_gate",
    "count",
    "eligible_count",
    "applied_count",
    "fallback_count",
    "gate_accepted",
    "gate_reasons",
    "gate_rejection_counts",
    "cutr_geometry_sha256",
    "actual_geometry_sha256",
    "scores_sha256",
    "input_pred_proj_xy_sha256",
    "boxer_commit",
    "boxer_checkpoint_sha256",
)
RUNTIME_PATTERN = re.compile(
    r"^Cost:\s*([0-9]+(?:\.[0-9]+)?)\s*s\s+Average FPS:\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*$",
    re.MULTILINE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_scenes(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list must be non-empty and unique")
    if any(not scene.isdigit() for scene in scenes):
        raise ValueError("CA-1M scene identifiers must be numeric")
    return scenes


def exact_scene_files(root: Path, scenes: Iterable[str], suffix: str) -> dict[str, Path]:
    expected = {f"{scene}{suffix}" for scene in scenes}
    actual = {path.name for path in root.glob(f"*{suffix}") if path.is_file()}
    if actual != expected:
        raise ValueError(
            f"scene files disagree in {root}: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return {scene: root / f"{scene}{suffix}" for scene in scenes}


def load_prediction(path: Path) -> list[tuple[int, np.ndarray, float]]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
        if handle.read(1):
            raise ValueError(f"trailing bytes in prediction: {path}")
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], list):
        raise ValueError(f"prediction must contain one list batch: {path}")
    rows: list[tuple[int, np.ndarray, float]] = []
    for index, row in enumerate(value[0]):
        if not isinstance(row, tuple) or len(row) != 3:
            raise ValueError(f"invalid prediction row {index}: {path}")
        label, corners, score = row
        corners = np.asarray(corners)
        score = float(score)
        if int(label) != 0:
            raise ValueError(f"non-class-agnostic row {index}: {path}")
        if corners.shape != (8, 3) or not np.isfinite(corners).all():
            raise ValueError(f"invalid OBB corners in row {index}: {path}")
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"invalid score in row {index}: {path}")
        rows.append((int(label), corners.copy(), score))
    return rows


def compare_predictions(
    scene: str,
    anchor: list[tuple[int, np.ndarray, float]],
    observer: list[tuple[int, np.ndarray, float]],
) -> dict[str, Any]:
    if len(anchor) != len(observer):
        raise ValueError(f"{scene}: prediction count differs")
    for index, (left, right) in enumerate(zip(anchor, observer)):
        if left[0] != right[0]:
            raise ValueError(f"{scene}: label differs at row {index}")
        if not np.array_equal(left[1], right[1]):
            delta = float(np.max(np.abs(left[1].astype(float) - right[1].astype(float))))
            raise ValueError(
                f"{scene}: OBB corners differ at row {index}, max_abs={delta}"
            )
        if left[2] != right[2]:
            raise ValueError(
                f"{scene}: score differs at row {index}: {left[2]} != {right[2]}"
            )
    return {"rows": len(anchor), "semantic_identity": True}


def _scalar(array: np.ndarray, name: str) -> Any:
    value = np.asarray(array)
    if value.shape != ():
        raise ValueError(f"diagnostic {name} must be scalar")
    return value.item()


def audit_diagnostic(
    scene: str,
    path: Path,
    prediction: list[tuple[int, np.ndarray, float]],
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != DIAGNOSTIC_KEYS:
            raise ValueError(
                f"{scene}: diagnostic keys disagree: "
                f"missing={sorted(DIAGNOSTIC_KEYS - set(payload.files))}, "
                f"extra={sorted(set(payload.files) - DIAGNOSTIC_KEYS)}"
            )
        if str(_scalar(payload["scene_id"], "scene_id")) != scene:
            raise ValueError(f"{scene}: diagnostic scene_id disagrees")
        names = tuple(str(value) for value in payload["quality_feature_names"].tolist())
        if names != QUALITY_FEATURE_NAMES:
            raise ValueError(f"{scene}: B6 feature schema/order disagrees")
        result_indices = np.asarray(payload["result_indices"])
        source_indices = np.asarray(payload["source_indices"])
        if result_indices.dtype.kind not in "iu" or result_indices.ndim != 1:
            raise ValueError(f"{scene}: result_indices must be an integer vector")
        result_indices = result_indices.astype(np.int64, copy=False)
        if len(result_indices) and (
            np.any(np.diff(result_indices) <= 0)
            or result_indices[0] < 0
            or result_indices[-1] >= len(prediction)
        ):
            raise ValueError(f"{scene}: result_indices are not strict in-range indices")
        if source_indices.dtype.kind not in "iu" or source_indices.shape != result_indices.shape:
            raise ValueError(f"{scene}: source_indices schema disagrees")
        if not np.array_equal(source_indices.astype(np.int64), result_indices):
            raise ValueError(f"{scene}: no-op P2 source_indices != result_indices")

        rows = len(result_indices)
        boxes = np.asarray(payload["boxes"])
        scores = np.asarray(payload["scores"])
        features = np.asarray(payload["quality_features"])
        points = np.asarray(payload["points"])
        point_mask = np.asarray(payload["point_mask"])
        track_ids = np.asarray(payload["track_ids"])
        labels = np.asarray(payload["labels"])
        if boxes.shape != (rows, 6) or not np.isfinite(boxes).all():
            raise ValueError(f"{scene}: invalid diagnostic boxes")
        if boxes.dtype != np.float32:
            raise ValueError(f"{scene}: diagnostic boxes must be float32")
        if scores.shape != (rows,) or not np.isfinite(scores).all():
            raise ValueError(f"{scene}: invalid diagnostic scores")
        if features.shape != (rows, len(QUALITY_FEATURE_NAMES)):
            raise ValueError(f"{scene}: invalid B6 feature shape")
        if not np.isfinite(features).all() or np.any(features < 0.0) or np.any(features > 1.0):
            raise ValueError(f"{scene}: B6 features must be finite in [0,1]")
        if points.ndim != 3 or points.shape[0] != rows or points.shape[2] != 3:
            raise ValueError(f"{scene}: invalid diagnostic point tensor")
        if point_mask.shape != points.shape[:2] or point_mask.dtype != np.bool_:
            raise ValueError(f"{scene}: invalid point mask")
        if not np.isfinite(points).all():
            raise ValueError(f"{scene}: non-finite diagnostic points")
        if track_ids.shape != (rows,) or track_ids.dtype.kind not in "iu":
            raise ValueError(f"{scene}: invalid track_ids")
        if rows and (np.any(track_ids < 0) or len(np.unique(track_ids)) != rows):
            raise ValueError(f"{scene}: track_ids must be non-negative and unique")
        if labels.shape != (rows,) or labels.dtype.kind not in "US":
            raise ValueError(f"{scene}: invalid labels")
        if rows and np.any(point_mask.sum(axis=1) == 0):
            raise ValueError(f"{scene}: every observed row must contain memory points")

        if rows:
            observed_corners = np.stack(
                [prediction[int(index)][1] for index in result_indices], axis=0
            ).astype(np.float32, copy=False)
        else:
            observed_corners = np.empty((0, 8, 3), dtype=np.float32)
        expected_boxes = corners_to_center_size(observed_corners)
        if not np.array_equal(boxes, expected_boxes):
            maximum = (
                float(np.max(np.abs(boxes.astype(np.float64) - expected_boxes)))
                if rows
                else 0.0
            )
            raise ValueError(
                f"{scene}: diagnostic boxes do not map to observer OBB rows, "
                f"max_abs={maximum}"
            )

        expected_scores = np.asarray(
            [prediction[int(index)][2] for index in result_indices], dtype=scores.dtype
        )
        if not np.array_equal(scores, expected_scores):
            raise ValueError(f"{scene}: diagnostic scores do not map to prediction rows")
        if rows and not np.array_equal(features[:, 0], scores.astype(features.dtype)):
            raise ValueError(f"{scene}: detector_score feature does not equal source score")

        summary = json.loads(str(_scalar(payload["summary_json"], "summary_json")))
        if not isinstance(summary, dict) or not summary.get("enabled", False):
            raise ValueError(f"{scene}: invalid enabled summary")
        for key in ("supplemental_output", "refits_accepted", "neural_refits_accepted"):
            if int(summary.get(key, -1)) != 0:
                raise ValueError(f"{scene}: observer mutation counter {key} is nonzero")
        if summary.get("candidate_ttl_clock") != "provider_call":
            raise ValueError(f"{scene}: candidate TTL clock drifted")
        if int(summary.get("candidate_archived_total", -1)) != 0:
            raise ValueError(f"{scene}: archive mutation is not allowed")
        return {
            "observed_rows": rows,
            "prediction_rows": len(prediction),
            "coverage": (float(rows) / len(prediction)) if prediction else 0.0,
            "provider_calls": int(summary.get("provider_calls", 0)),
            "provider_seconds": float(summary.get("provider_seconds", 0.0)),
            "appearance_seconds": float(summary.get("appearance_seconds", 0.0)),
            "geometry_seconds": float(summary.get("geometry_seconds", 0.0)),
        }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row {line_number}: {path}")
            rows.append(value)
    if not rows:
        raise ValueError(f"empty Boxer diagnostic: {path}")
    return rows


def audit_boxer(scene: str, anchor_path: Path, observer_path: Path) -> dict[str, Any]:
    anchor = read_jsonl(anchor_path)
    observer = read_jsonl(observer_path)
    if len(anchor) != len(observer):
        raise ValueError(f"{scene}: Boxer call count differs")
    for row_index, (left, right) in enumerate(zip(anchor, observer)):
        for key in BOXER_STABLE_KEYS:
            if left.get(key) != right.get(key):
                raise ValueError(
                    f"{scene}: Boxer deterministic field {key!r} differs at call {row_index}"
                )
    return {
        "calls": len(anchor),
        "proposals": sum(int(row.get("count", 0)) for row in anchor),
        "applied": sum(int(row.get("applied_count", 0)) for row in anchor),
        "deterministic_fields_identity": True,
    }


def audit_runtime_log(path: Path) -> dict[str, float | str]:
    matches = RUNTIME_PATTERN.findall(path.read_text(errors="strict"))
    if len(matches) != 1:
        raise ValueError(f"expected one Cost/Average FPS summary: {path}")
    cost, fps = (float(value) for value in matches[0])
    if not np.isfinite(cost) or not np.isfinite(fps) or cost <= 0.0 or fps <= 0.0:
        raise ValueError(f"invalid runtime summary: {path}")
    return {
        "cost_seconds": cost,
        "average_fps": fps,
        "frame_equivalent": cost * fps,
        "sha256": sha256(path),
    }


def write_json_atomic(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing existing audit report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing existing audit report: {path}"
            ) from error
    finally:
        if temporary.exists():
            temporary.unlink()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--scene-list", type=Path, required=True)
    result.add_argument("--anchor-root", type=Path, required=True)
    result.add_argument("--historical-anchor-root", type=Path, required=True)
    result.add_argument("--observer-root", type=Path, required=True)
    result.add_argument("--diagnostics-root", type=Path, required=True)
    result.add_argument("--anchor-boxer-root", type=Path, required=True)
    result.add_argument("--observer-boxer-root", type=Path, required=True)
    result.add_argument("--historical-log-root", type=Path, required=True)
    result.add_argument("--observer-log-root", type=Path, required=True)
    result.add_argument("--quality-checkpoint", type=Path, required=True)
    result.add_argument("--expected-quality-sha256", required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    scenes = read_scenes(args.scene_list)
    for path in (
        args.anchor_root,
        args.observer_root,
        args.diagnostics_root,
        args.anchor_boxer_root,
        args.observer_boxer_root,
        args.historical_log_root,
        args.observer_log_root,
    ):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if not args.historical_anchor_root.is_dir():
        raise FileNotFoundError(args.historical_anchor_root)
    historical_files = exact_scene_files(
        args.historical_anchor_root, scenes, "_boxes.pkl"
    )
    quality_sha = sha256(args.quality_checkpoint)
    if quality_sha != args.expected_quality_sha256:
        raise ValueError(
            f"quality checkpoint SHA disagrees: {quality_sha} != "
            f"{args.expected_quality_sha256}"
        )

    anchor_files = exact_scene_files(args.anchor_root, scenes, "_boxes.pkl")
    observer_files = exact_scene_files(args.observer_root, scenes, "_boxes.pkl")
    diagnostic_files = exact_scene_files(args.diagnostics_root, scenes, "_tracks.npz")
    anchor_boxer = exact_scene_files(
        args.anchor_boxer_root, scenes, "_boxer_lifting.jsonl"
    )
    observer_boxer = exact_scene_files(
        args.observer_boxer_root, scenes, "_boxer_lifting.jsonl"
    )
    historical_logs = exact_scene_files(args.historical_log_root, scenes, ".log")
    observer_logs = exact_scene_files(args.observer_log_root, scenes, ".log")

    report: dict[str, Any] = {
        "schema": "boxfusion.ca1m_c2_b6_zero_shot_identity.v1",
        "ok": True,
        "output_mutation_authorized": False,
        "dataset": "CA1M",
        "scenes": len(scenes),
        "scene_list": str(args.scene_list.resolve()),
        "scene_list_sha256": sha256(args.scene_list),
        "quality_checkpoint": str(args.quality_checkpoint.resolve()),
        "quality_checkpoint_sha256": quality_sha,
        "identity_anchor_contract": "same_run_pre_online_finalize",
        "historical_p1_role": "replay_drift_and_metric_reference_only",
        "per_scene": {},
    }
    total_rows = 0
    total_observed = 0
    total_provider_seconds = 0.0
    total_appearance_seconds = 0.0
    total_geometry_seconds = 0.0
    historical_runtime_seconds = 0.0
    historical_frame_equivalent = 0.0
    observer_runtime_seconds = 0.0
    observer_frame_equivalent = 0.0
    for scene in scenes:
        anchor_prediction = load_prediction(anchor_files[scene])
        observer_prediction = load_prediction(observer_files[scene])
        identity = compare_predictions(scene, anchor_prediction, observer_prediction)
        diagnostic = audit_diagnostic(
            scene, diagnostic_files[scene], observer_prediction
        )
        boxer = audit_boxer(scene, anchor_boxer[scene], observer_boxer[scene])
        historical_runtime = audit_runtime_log(historical_logs[scene])
        observer_runtime = audit_runtime_log(observer_logs[scene])
        report["per_scene"][scene] = {
            **identity,
            **diagnostic,
            "boxer": boxer,
            "anchor_prediction_sha256": sha256(anchor_files[scene]),
            "observer_prediction_sha256": sha256(observer_files[scene]),
            "diagnostic_sha256": sha256(diagnostic_files[scene]),
            "anchor_boxer_sha256": sha256(anchor_boxer[scene]),
            "observer_boxer_sha256": sha256(observer_boxer[scene]),
            "runtime": {
                "historical_p1": historical_runtime,
                "observer": observer_runtime,
                "observer_over_historical_p1_cost_ratio": float(
                    observer_runtime["cost_seconds"] / historical_runtime["cost_seconds"]
                ),
            },
        }
        historical_path = historical_files[scene]
        historical = load_prediction(historical_path)
        if len(historical) != len(observer_prediction):
            raise ValueError(f"{scene}: historical anchor prediction count differs")
        if any(
            left[0] != right[0] or left[2] != right[2]
            for left, right in zip(historical, observer_prediction)
        ):
            raise ValueError(f"{scene}: historical anchor label/score differs")
        corner_deltas = np.asarray(
            [
                np.max(
                    np.abs(
                        left[1].astype(np.float64)
                        - right[1].astype(np.float64)
                    )
                )
                for left, right in zip(historical, observer_prediction)
            ],
            dtype=np.float64,
        )
        score_deltas = np.asarray(
            [abs(left[2] - right[2]) for left, right in zip(historical, observer_prediction)],
            dtype=np.float64,
        )
        report["per_scene"][scene]["historical_anchor_drift"] = {
            "historical_prediction_sha256": sha256(historical_path),
            "rows": len(historical),
            "changed_corner_rows": int(np.count_nonzero(corner_deltas)),
            "corner_max_abs": float(corner_deltas.max()) if len(corner_deltas) else 0.0,
            "changed_score_rows": int(np.count_nonzero(score_deltas)),
            "score_max_abs": float(score_deltas.max()) if len(score_deltas) else 0.0,
            "informational_only": True,
        }
        total_rows += len(observer_prediction)
        total_observed += int(diagnostic["observed_rows"])
        total_provider_seconds += float(diagnostic["provider_seconds"])
        total_appearance_seconds += float(diagnostic["appearance_seconds"])
        total_geometry_seconds += float(diagnostic["geometry_seconds"])
        historical_runtime_seconds += float(historical_runtime["cost_seconds"])
        historical_frame_equivalent += float(historical_runtime["frame_equivalent"])
        observer_runtime_seconds += float(observer_runtime["cost_seconds"])
        observer_frame_equivalent += float(observer_runtime["frame_equivalent"])
    report["summary"] = {
        "prediction_rows": total_rows,
        "observed_rows": total_observed,
        "coverage": float(total_observed / total_rows) if total_rows else 0.0,
        "provider_seconds": total_provider_seconds,
        "appearance_seconds": total_appearance_seconds,
        "geometry_seconds": total_geometry_seconds,
        "identity_scenes": len(scenes),
        "runtime": {
            "historical_p1_total_seconds": historical_runtime_seconds,
            "observer_total_seconds": observer_runtime_seconds,
            "historical_p1_frame_weighted_fps": (
                historical_frame_equivalent / historical_runtime_seconds
                if historical_runtime_seconds
                else 0.0
            ),
            "observer_frame_weighted_fps": (
                observer_frame_equivalent / observer_runtime_seconds
                if observer_runtime_seconds
                else 0.0
            ),
            "observer_over_historical_p1_cost_ratio": (
                observer_runtime_seconds / historical_runtime_seconds
                if historical_runtime_seconds
                else 0.0
            ),
        },
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"CA-1M P2 observer identity audit passed: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
