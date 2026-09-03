#!/usr/bin/env python3
"""Validate an exact, observer-only P1R/P1S artifact set.

The validator intentionally knows nothing about training GT.  It verifies the
frozen stage contract, exact checkpoint identity, aligned diagnostic arrays,
and the absence of any formal-output mutation request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "boxfusion.p1v2.run_artifact_validation.v1"
_SCENE = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTRACT = {
    "P1R": {
        "profile": "p1r_snapshot_target_residual_observer",
        "head_architecture": "per_voxel_mlp",
        "target_assignment_scope": "snapshot_inside_only",
    },
    "P1S": {
        "profile": "p1s_native_sparse_context_observer",
        "head_architecture": "native_sparse_context_v1",
        "target_assignment_scope": "snapshot_inside_only",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_scene_ids(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = tuple(
        row.strip()
        for row in path.read_text(encoding="utf-8").splitlines()
        if row.strip()
    )
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("scene list must be non-empty and unique")
    invalid = [row for row in rows if _SCENE.fullmatch(row) is None]
    if invalid:
        raise ValueError(f"invalid ScanNet scene id: {invalid[0]!r}")
    return rows


def _scalar(archive: Mapping[str, np.ndarray], key: str, path: Path) -> Any:
    if key not in archive:
        raise ValueError(f"{path}: missing {key}")
    value = np.asarray(archive[key])
    if value.shape != () or value.dtype.hasobject:
        raise ValueError(f"{path}: {key} must be a non-object scalar")
    result = value.item()
    return result.decode("utf-8") if isinstance(result, bytes) else result


def _bool_scalar(
    archive: Mapping[str, np.ndarray], key: str, expected: bool, path: Path
) -> None:
    value = np.asarray(archive.get(key))
    if value.shape != () or value.dtype != np.dtype(bool):
        raise ValueError(f"{path}: {key} must be a Boolean scalar")
    if bool(value.item()) is not expected:
        raise ValueError(f"{path}: unsafe {key}={bool(value.item())}")


def _checkpoint_contract(path: Path) -> tuple[str, str]:
    try:
        import torch
    except Exception as error:  # pragma: no cover - runtime preflight
        raise RuntimeError("checkpoint validation requires PyTorch") from error
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - old PyTorch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("P1-v2 checkpoint must contain a mapping")
    model = payload.get("model_config")
    training = payload.get("training_config")
    if not isinstance(model, Mapping) or not isinstance(training, Mapping):
        raise ValueError("P1-v2 checkpoint lacks model/training contract")
    explicit_head = model.get("head_architecture")
    native_head = model.get("architecture")
    if (
        explicit_head is not None
        and native_head is not None
        and explicit_head != native_head
    ):
        raise ValueError(
            "P1-v2 checkpoint has conflicting head architecture fields"
        )
    head = explicit_head if explicit_head is not None else native_head
    target = training.get("target_assignment_scope")
    if not isinstance(head, str) or not isinstance(target, str):
        raise ValueError(
            "P1-v2 checkpoint lacks head_architecture or architecture/"
            "target_assignment_scope"
        )
    return head, target


def _validate_diagnostic(
    path: Path,
    *,
    scene: str,
    stage: str,
    checkpoint_sha256: str,
) -> dict[str, int | float]:
    contract = _CONTRACT[stage]
    with np.load(path, allow_pickle=False) as loaded:
        archive = {
            key: np.array(loaded[key], copy=True) for key in loaded.files
        }
    expected_text = {
        "scene_id": scene,
        "p1_stage": stage,
        "p1_profile": contract["profile"],
        "p1_checkpoint_sha256": checkpoint_sha256,
        "p1_head_architecture": contract["head_architecture"],
        "p1_target_assignment_scope": contract["target_assignment_scope"],
    }
    for key, expected in expected_text.items():
        observed = _scalar(archive, key, path)
        if observed != expected:
            raise ValueError(
                f"{path}: {key}={observed!r}, expected {expected!r}"
            )
    for key, expected in {
        "p1_enabled": True,
        "p1_observer_only": True,
        "p1_uses_ground_truth": False,
        "p1_mutation_enabled": False,
        "p1_complete": True,
        "p1_class_agnostic": True,
    }.items():
        _bool_scalar(archive, key, expected, path)
    if "p1_reads_semantic_labels" in archive:
        _bool_scalar(archive, "p1_reads_semantic_labels", False, path)
    if int(_scalar(archive, "p1_applied_count", path)) != 0:
        raise ValueError(f"{path}: observer applied formal output rows")
    if int(_scalar(archive, "p1_regression_dim", path)) != 6:
        raise ValueError(f"{path}: residual regression must remain 6-D")

    step_fields = (
        "p1_step_frame_ids",
        "p1_step_provider_steps",
        "p1_step_voxel_counts",
        "p1_step_candidate_counts",
        "p1_step_voxelize_seconds",
        "p1_step_head_seconds",
        "p1_step_nms_seconds",
    )
    step_lengths: set[int] = set()
    runtime = 0.0
    for key in step_fields:
        if key not in archive:
            raise ValueError(f"{path}: missing {key}")
        values = np.asarray(archive[key])
        if values.ndim != 1 or values.dtype.hasobject:
            raise ValueError(f"{path}: {key} must be a non-object vector")
        step_lengths.add(len(values))
        if key.endswith("_seconds"):
            numeric = np.asarray(values, dtype=np.float64)
            if not np.isfinite(numeric).all() or np.any(numeric < 0.0):
                raise ValueError(f"{path}: invalid {key}")
            runtime += float(numeric.sum())
    if len(step_lengths) != 1:
        raise ValueError(f"{path}: P1-v2 step arrays disagree in length")
    step_count = next(iter(step_lengths))
    if "p1_step_failed" in archive:
        failed = np.asarray(archive["p1_step_failed"])
        if failed.shape != (step_count,) or failed.dtype != np.dtype(bool):
            raise ValueError(f"{path}: invalid p1_step_failed")
        if bool(np.any(failed)):
            raise ValueError(f"{path}: P1-v2 contains failed steps")

    boxes = np.asarray(archive.get("p1_candidate_boxes"))
    corners = np.asarray(archive.get("p1_candidate_corners"))
    scores = np.asarray(archive.get("p1_candidate_scores"))
    ids = np.asarray(archive.get("p1_candidate_ids"))
    count = len(boxes) if boxes.ndim else -1
    if (
        boxes.shape != (count, 6)
        or corners.shape != (count, 8, 3)
        or scores.shape != (count,)
        or ids.shape != (count,)
        or any(array.dtype.hasobject for array in (boxes, corners, scores, ids))
        or not np.isfinite(np.asarray(boxes, dtype=np.float64)).all()
        or not np.isfinite(np.asarray(corners, dtype=np.float64)).all()
        or not np.isfinite(np.asarray(scores, dtype=np.float64)).all()
    ):
        raise ValueError(f"{path}: invalid P1-v2 candidate arrays")
    if len(np.unique(ids)) != count:
        raise ValueError(f"{path}: candidate ids are not unique")
    return {
        "steps": int(step_count),
        "candidates": int(count),
        "runtime_seconds": float(runtime),
    }


def validate(
    *,
    stage: str,
    scene_list: Path,
    prediction_root: Path,
    diagnostics_root: Path,
    expected_checkpoint: Path,
) -> dict[str, Any]:
    stage = str(stage).strip().upper()
    if stage not in _CONTRACT:
        raise ValueError("stage must be P1R or P1S")
    scenes = read_scene_ids(scene_list)
    if not expected_checkpoint.is_file():
        raise FileNotFoundError(expected_checkpoint)
    head, target = _checkpoint_contract(expected_checkpoint)
    contract = _CONTRACT[stage]
    if (
        head != contract["head_architecture"]
        or target != contract["target_assignment_scope"]
    ):
        raise ValueError("checkpoint contract disagrees with requested stage")
    checkpoint_sha256 = _sha256(expected_checkpoint)
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
        raise ValueError(
            "P1-v2 prediction set mismatch: "
            f"missing={sorted(expected_predictions-actual_predictions)[:8]}, "
            f"extra={sorted(actual_predictions-expected_predictions)[:8]}"
        )
    if actual_diagnostics != expected_diagnostics:
        raise ValueError(
            "P1-v2 diagnostic set mismatch: "
            f"missing={sorted(expected_diagnostics-actual_diagnostics)[:8]}, "
            f"extra={sorted(actual_diagnostics-expected_diagnostics)[:8]}"
        )
    totals = {"steps": 0, "candidates": 0, "runtime_seconds": 0.0}
    for scene in scenes:
        prediction = prediction_root / f"{scene}_boxes.pkl"
        diagnostic = diagnostics_root / f"{scene}_tracks.npz"
        if prediction.stat().st_size <= 0 or diagnostic.stat().st_size <= 0:
            raise ValueError(f"empty P1-v2 artifact for {scene}")
        row = _validate_diagnostic(
            diagnostic,
            scene=scene,
            stage=stage,
            checkpoint_sha256=checkpoint_sha256,
        )
        for key in totals:
            totals[key] += row[key]
    return {
        "schema": SCHEMA,
        "ok": True,
        "stage": stage,
        "profile": contract["profile"],
        "head_architecture": head,
        "target_assignment_scope": target,
        "scene_count": len(scenes),
        "checkpoint_sha256": checkpoint_sha256,
        "prediction_root": str(prediction_root.resolve()),
        "diagnostics_root": str(diagnostics_root.resolve()),
        **totals,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("P1R", "P1S"))
    parser.add_argument("--scene-list", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--diagnostics-root", required=True, type=Path)
    parser.add_argument("--expected-checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = validate(
        stage=args.stage,
        scene_list=args.scene_list,
        prediction_root=args.prediction_root,
        diagnostics_root=args.diagnostics_root,
        expected_checkpoint=args.expected_checkpoint,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(args.output)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
