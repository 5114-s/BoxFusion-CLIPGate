#!/usr/bin/env python3
"""Fail-closed validation for a complete P2 observer run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boxfusion.occupancy_topk import P2_DIAGNOSTIC_SCHEMA  # noqa: E402
from boxfusion.residual_proposal import P1_DIAGNOSTIC_SCHEMA  # noqa: E402
from tools.train_p1_residual_head import read_scene_ids  # noqa: E402
from tools.audit_p1_nondeterminism import (  # noqa: E402
    compare_prediction_roots,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(
    archive: np.lib.npyio.NpzFile,
    key: str,
    path: Path,
) -> np.ndarray:
    if key not in archive.files:
        raise ValueError(f"{path}: missing {key}")
    value = np.asarray(archive[key])
    if value.shape != () or value.dtype.hasobject:
        raise ValueError(f"{path}: {key} must be a non-object scalar")
    return value


def _text(
    archive: np.lib.npyio.NpzFile,
    key: str,
    path: Path,
) -> str:
    value = _scalar(archive, key, path).item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise ValueError(f"{path}: {key} must be text")
    return value


def _boolean(
    archive: np.lib.npyio.NpzFile,
    key: str,
    path: Path,
) -> bool:
    value = _scalar(archive, key, path)
    if value.dtype != np.dtype(bool):
        raise ValueError(f"{path}: {key} must be Boolean")
    return bool(value.item())


def _integer(
    archive: np.lib.npyio.NpzFile,
    key: str,
    path: Path,
) -> int:
    value = _scalar(archive, key, path)
    if not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{path}: {key} must be integer")
    return int(value.item())


def _aligned_rows(
    archive: np.lib.npyio.NpzFile, path: Path
) -> int:
    ids = np.asarray(archive["p2_candidate_ids"])
    boxes = np.asarray(archive["p2_candidate_boxes"])
    corners = np.asarray(archive["p2_candidate_corners"])
    objectness = np.asarray(archive["p2_candidate_objectness"])
    occupancy = np.asarray(archive["p2_candidate_occupancy_scores"])
    ranks = np.asarray(archive["p2_candidate_occupancy_ranks"])
    count = len(ids)
    if (
        ids.ndim != 1
        or ids.dtype.hasobject
        or len(set(str(value) for value in ids.tolist())) != count
        or boxes.shape != (count, 6)
        or corners.shape != (count, 8, 3)
        or objectness.shape != (count,)
        or occupancy.shape != (count,)
        or ranks.shape != (count,)
        or not np.isfinite(boxes).all()
        or not np.isfinite(corners).all()
        or not np.isfinite(objectness).all()
        or not np.isfinite(occupancy).all()
        or np.any(boxes[:, 3:] <= 0.0)
        or np.any(occupancy < 0.0)
        or np.any(occupancy > 1.0)
        or np.any(ranks < 0)
    ):
        raise ValueError(f"{path}: invalid or misaligned P2 candidates")
    return count


def _validate_step_alignment(
    archive: np.lib.npyio.NpzFile, path: Path
) -> int:
    required = (
        "p1_step_frame_ids",
        "p1_step_provider_steps",
        "p2_step_frame_ids",
        "p2_step_provider_steps",
        "p2_step_input_voxel_counts",
        "p2_step_eligible_voxel_counts",
        "p2_step_selected_voxel_counts",
        "p2_step_candidate_counts",
        "p2_step_seconds",
    )
    missing = [key for key in required if key not in archive.files]
    if missing:
        raise ValueError(f"{path}: missing {missing[0]}")
    rows = {key: np.asarray(archive[key]) for key in required}
    count = len(rows["p2_step_frame_ids"])
    if count < 1 or any(
        value.ndim != 1 or len(value) != count
        for value in rows.values()
    ):
        raise ValueError(f"{path}: P2 did not execute on every P1 step")
    for key in required[:-1]:
        if not np.issubdtype(rows[key].dtype, np.integer):
            raise ValueError(f"{path}: {key} must be integer")
    seconds = rows["p2_step_seconds"]
    if (
        not np.issubdtype(seconds.dtype, np.floating)
        or not np.isfinite(seconds).all()
        or np.any(seconds < 0.0)
    ):
        raise ValueError(f"{path}: invalid P2 timing array")
    if not np.array_equal(
        rows["p1_step_frame_ids"], rows["p2_step_frame_ids"]
    ) or not np.array_equal(
        rows["p1_step_provider_steps"],
        rows["p2_step_provider_steps"],
    ):
        raise ValueError(f"{path}: P1/P2 scheduling is not aligned")
    inputs = rows["p2_step_input_voxel_counts"]
    eligible = rows["p2_step_eligible_voxel_counts"]
    selected = rows["p2_step_selected_voxel_counts"]
    candidates = rows["p2_step_candidate_counts"]
    if (
        np.any(inputs < 0)
        or np.any(eligible < 0)
        or np.any(selected < 0)
        or np.any(candidates < 0)
        or np.any(eligible > inputs)
        or np.any(selected > eligible)
        or np.any(candidates > selected)
    ):
        raise ValueError(f"{path}: impossible P2 step counts")
    return count


def validate(
    *,
    scene_list: Path,
    prediction_root: Path,
    diagnostics_root: Path,
    expected_p1_checkpoint: Path,
    expected_p2_checkpoint: Path,
    baseline_prediction_root: Path | None = None,
    identity_corner_tolerance: float = 0.02,
    identity_score_tolerance: float = 0.02,
    identity_iou_loss_tolerance: float = 0.05,
) -> dict:
    for name, value in (
        ("identity_corner_tolerance", identity_corner_tolerance),
        ("identity_score_tolerance", identity_score_tolerance),
        ("identity_iou_loss_tolerance", identity_iou_loss_tolerance),
    ):
        if not np.isfinite(value) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    scenes = read_scene_ids(scene_list, role="P2 evaluation")
    expected_predictions = {f"{scene}_boxes.pkl" for scene in scenes}
    expected_diagnostics = {f"{scene}_tracks.npz" for scene in scenes}
    actual_predictions = {
        path.name
        for path in prediction_root.glob("scene*_boxes.pkl")
        if path.is_file()
    }
    actual_diagnostics = {
        path.name
        for path in diagnostics_root.glob("scene*_tracks.npz")
        if path.is_file()
    }
    if actual_predictions != expected_predictions:
        raise ValueError("P2 prediction scene set mismatch")
    if actual_diagnostics != expected_diagnostics:
        raise ValueError("P2 diagnostic scene set mismatch")
    expected_p1_sha = _sha256(expected_p1_checkpoint)
    expected_p2_sha = _sha256(expected_p2_checkpoint)
    observed_p1: set[str] = set()
    observed_p2: set[str] = set()
    candidate_count = 0
    step_count = 0
    for scene in scenes:
        prediction = prediction_root / f"{scene}_boxes.pkl"
        diagnostic = diagnostics_root / f"{scene}_tracks.npz"
        if prediction.stat().st_size <= 0 or diagnostic.stat().st_size <= 0:
            raise ValueError(f"empty P2 artifact for {scene}")
        with np.load(diagnostic, allow_pickle=False) as archive:
            for key in archive.files:
                if np.asarray(archive[key]).dtype.hasobject:
                    raise ValueError(f"{diagnostic}: object dtype in {key}")
            expected_text = {
                "scene_id": scene,
                "p1_schema": P1_DIAGNOSTIC_SCHEMA,
                "p2_schema": P2_DIAGNOSTIC_SCHEMA,
                "p2_stage": "P2",
                "p2_profile": "p2_occupancy_topk_observer",
            }
            for key, expected in expected_text.items():
                if _text(archive, key, diagnostic) != expected:
                    raise ValueError(f"{diagnostic}: invalid {key}")
            for prefix in ("p1", "p2"):
                expected_bool = {
                    f"{prefix}_enabled": True,
                    f"{prefix}_observer_only": True,
                    f"{prefix}_uses_ground_truth": False,
                    f"{prefix}_mutation_enabled": False,
                    f"{prefix}_complete": True,
                    f"{prefix}_class_agnostic": True,
                }
                for key, expected in expected_bool.items():
                    if _boolean(archive, key, diagnostic) is not expected:
                        raise ValueError(f"{diagnostic}: unsafe {key}")
                if _integer(
                    archive, f"{prefix}_applied_count", diagnostic
                ) != 0:
                    raise ValueError(
                        f"{diagnostic}: {prefix} mutated formal output"
                    )
            p1_sha = _text(
                archive, "p1_checkpoint_sha256", diagnostic
            )
            p2_sha = _text(
                archive, "p2_checkpoint_sha256", diagnostic
            )
            if _SHA256.fullmatch(p1_sha) is None:
                raise ValueError(f"{diagnostic}: invalid P1 SHA")
            if _SHA256.fullmatch(p2_sha) is None:
                raise ValueError(f"{diagnostic}: invalid P2 SHA")
            observed_p1.add(p1_sha)
            observed_p2.add(p2_sha)
            step_count += _validate_step_alignment(archive, diagnostic)
            candidate_count += _aligned_rows(archive, diagnostic)
    if observed_p1 != {expected_p1_sha}:
        raise ValueError("P2 diagnostics do not match requested P1")
    if observed_p2 != {expected_p2_sha}:
        raise ValueError("P2 diagnostics do not match requested P2")
    identity = None
    if baseline_prediction_root is not None:
        identity = compare_prediction_roots(
            scenes=scenes,
            baseline_root=baseline_prediction_root,
            candidate_root=prediction_root,
            match_iou=0.25,
        )
        aggregate = identity["aggregate"]
        structure_ok = all(
            int(aggregate[key]) == 0
            for key in (
                "baseline_missing_count",
                "candidate_extra_count",
                "label_mismatch_count",
                "order_inversions",
            )
        )
        numeric_ok = (
            float(aggregate["corner_abs"]["max"])
            <= float(identity_corner_tolerance)
            and float(aggregate["score_abs"]["max"])
            <= float(identity_score_tolerance)
            and float(aggregate["matched_iou"]["loss_max"])
            <= float(identity_iou_loss_tolerance)
        )
        identity["contract"] = {
            "structure_ok": structure_ok,
            "numeric_ok": numeric_ok,
            "corner_abs_max_tolerance": float(
                identity_corner_tolerance
            ),
            "score_abs_max_tolerance": float(
                identity_score_tolerance
            ),
            "matched_iou_loss_max_tolerance": float(
                identity_iou_loss_tolerance
            ),
            "basis": (
                "conservative envelope above the measured fixed-10 "
                "P0-repeat drift; rerun the nondeterminism audit when "
                "changing hardware or software"
            ),
        }
        if not structure_ok:
            raise ValueError(
                "P2 formal prediction structure differs from frozen P1"
            )
        if not numeric_ok:
            raise ValueError(
                "P2 formal prediction drift exceeds the accepted "
                "P0-repeat envelope"
            )
    return {
        "schema": "boxfusion.p2.run_artifact_validation.v1",
        "scene_count": len(scenes),
        "p2_candidate_count": candidate_count,
        "p2_step_count": step_count,
        "p1_checkpoint_sha256": expected_p1_sha,
        "p2_checkpoint_sha256": expected_p2_sha,
        "prediction_root": str(prediction_root.resolve()),
        "diagnostics_root": str(diagnostics_root.resolve()),
        "formal_identity": identity,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--diagnostics-root", required=True, type=Path)
    parser.add_argument(
        "--expected-p1-checkpoint", required=True, type=Path
    )
    parser.add_argument(
        "--expected-p2-checkpoint", required=True, type=Path
    )
    parser.add_argument("--baseline-prediction-root", type=Path)
    parser.add_argument(
        "--identity-corner-tolerance", type=float, default=0.02
    )
    parser.add_argument(
        "--identity-score-tolerance", type=float, default=0.02
    )
    parser.add_argument(
        "--identity-iou-loss-tolerance", type=float, default=0.05
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = validate(
        scene_list=args.scene_list,
        prediction_root=args.prediction_root,
        diagnostics_root=args.diagnostics_root,
        expected_p1_checkpoint=args.expected_p1_checkpoint,
        expected_p2_checkpoint=args.expected_p2_checkpoint,
        baseline_prediction_root=args.baseline_prediction_root,
        identity_corner_tolerance=args.identity_corner_tolerance,
        identity_score_tolerance=args.identity_score_tolerance,
        identity_iou_loss_tolerance=(
            args.identity_iou_loss_tolerance
        ),
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
