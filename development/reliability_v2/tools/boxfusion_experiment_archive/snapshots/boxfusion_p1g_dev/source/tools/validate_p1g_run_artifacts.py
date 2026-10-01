#!/usr/bin/env python3
"""Validate a complete, non-mutating P1S -> P1G artifact set.

P1G is a detached geometry observer.  This validator therefore checks both
halves of the contract: the embedded P1 stream must still be the frozen P1S
stream, and every P1G row must be a one-to-one child which can never enter the
formal prediction output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "boxfusion.p1g.run_artifact_validation.v1"
P1_SCHEMA = "boxfusion.p1.residual_proposal_observer.v1"
P1G_SCHEMA = "boxfusion.p1g.multiview_geometry_observer.v1"
P1S_PROFILE = "p1s_native_sparse_context_observer"
P1G_PROFILE = "p1g_multiview_occupancy_msr_observer"
_SCENE = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scene_ids(path: Path) -> tuple[str, ...]:
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


def _expect_scalar(
    archive: Mapping[str, np.ndarray],
    key: str,
    expected: Any,
    path: Path,
) -> None:
    observed = _scalar(archive, key, path)
    if observed != expected:
        raise ValueError(
            f"{path}: {key}={observed!r}, expected {expected!r}"
        )


def _expect_bool(
    archive: Mapping[str, np.ndarray],
    key: str,
    expected: bool,
    path: Path,
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
        raise ValueError("P1S checkpoint must contain a mapping")
    model = payload.get("model_config")
    training = payload.get("training_config")
    if not isinstance(model, Mapping) or not isinstance(training, Mapping):
        raise ValueError("P1S checkpoint lacks model/training contract")
    explicit = model.get("head_architecture")
    native = model.get("architecture")
    if explicit is not None and native is not None and explicit != native:
        raise ValueError("P1S checkpoint architecture fields disagree")
    architecture = explicit if explicit is not None else native
    assignment = training.get("target_assignment_scope")
    if (
        architecture != "native_sparse_context_v1"
        or assignment != "snapshot_inside_only"
    ):
        raise ValueError(
            "P1G requires native_sparse_context_v1 + "
            "snapshot_inside_only P1S checkpoint"
        )
    return str(architecture), str(assignment)


def _array(
    archive: Mapping[str, np.ndarray],
    key: str,
    path: Path,
) -> np.ndarray:
    if key not in archive:
        raise ValueError(f"{path}: missing {key}")
    value = np.asarray(archive[key])
    if value.dtype.hasobject:
        raise ValueError(f"{path}: {key} cannot have object dtype")
    return value


def _validate_box_corner_aliases(
    boxes: np.ndarray,
    corners: np.ndarray,
    *,
    label: str,
    path: Path,
) -> None:
    lower = boxes[:, :3] - 0.5 * boxes[:, 3:]
    upper = boxes[:, :3] + 0.5 * boxes[:, 3:]
    observed_lower = corners.min(axis=1)
    observed_upper = corners.max(axis=1)
    if not (
        np.allclose(lower, observed_lower, rtol=1e-5, atol=1e-5)
        and np.allclose(upper, observed_upper, rtol=1e-5, atol=1e-5)
    ):
        raise ValueError(f"{path}: {label} box/corner aliases disagree")


def _validate_scene(
    path: Path,
    *,
    scene: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as loaded:
        archive = {
            key: np.array(loaded[key], copy=True) for key in loaded.files
        }

    for key, expected in {
        "scene_id": scene,
        "p1_schema": P1_SCHEMA,
        "p1_stage": "P1S",
        "p1_profile": P1S_PROFILE,
        "p1_checkpoint_sha256": checkpoint_sha256,
        "p1_head_architecture": "native_sparse_context_v1",
        "p1_target_assignment_scope": "snapshot_inside_only",
        "p1g_schema": P1G_SCHEMA,
        "p1g_stage": "P1G",
        "p1g_profile": P1G_PROFILE,
        "p1g_parent_stage": "P1S",
        "p1g_parent_checkpoint_sha256": checkpoint_sha256,
    }.items():
        _expect_scalar(archive, key, expected, path)
    for key, expected in {
        "p1_enabled": True,
        "p1_observer_only": True,
        "p1_uses_ground_truth": False,
        "p1_reads_semantic_labels": False,
        "p1_mutation_enabled": False,
        "p1_complete": True,
        "p1_class_agnostic": True,
        "p1g_enabled": True,
        "p1g_observer_only": True,
        "p1g_uses_ground_truth": False,
        "p1g_reads_semantic_labels": False,
        "p1g_mutation_enabled": False,
        "p1g_complete": True,
        "p1g_class_agnostic": True,
    }.items():
        _expect_bool(archive, key, expected, path)
    if int(_scalar(archive, "p1_applied_count", path)) != 0:
        raise ValueError(f"{path}: frozen P1S mutated formal output")
    if int(_scalar(archive, "p1g_applied_count", path)) != 0:
        raise ValueError(f"{path}: P1G mutated formal output")
    if int(_scalar(archive, "p1g_regression_dim", path)) != 6:
        raise ValueError(f"{path}: P1G must remain a six-face refiner")

    try:
        config = json.loads(str(_scalar(archive, "p1g_config_json", path)))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid p1g_config_json") from error
    if (
        not isinstance(config, dict)
        or config.get("enabled") is not True
        or config.get("observer_only") is not True
        or config.get("mutate") is not False
        or config.get("collect_diagnostics") is not True
    ):
        raise ValueError(f"{path}: unsafe P1G resolved configuration")
    max_candidates = int(config.get("max_candidates", 0))
    top_k = int(config.get("top_k_views", 0))
    if max_candidates < 1 or top_k < 1:
        raise ValueError(f"{path}: invalid P1G bounds")

    p1_ids = _array(archive, "p1_candidate_ids", path)
    p1_boxes = _array(archive, "p1_candidate_boxes", path)
    p1_corners = _array(archive, "p1_candidate_corners", path)
    p1_scores = _array(archive, "p1_candidate_scores", path)
    count = min(len(p1_ids), max_candidates)
    parent_ids = _array(archive, "p1g_parent_candidate_ids", path)
    child_ids = _array(archive, "p1g_refined_candidate_ids", path)
    parent_boxes = _array(archive, "p1g_parent_boxes", path)
    parent_corners = _array(archive, "p1g_parent_corners", path)
    refined_boxes = _array(archive, "p1g_refined_boxes", path)
    refined_corners = _array(archive, "p1g_refined_corners", path)
    scores = _array(archive, "p1g_candidate_scores", path)
    applied = _array(archive, "p1g_candidate_applied", path)
    is_candidate = _array(archive, "p1g_is_candidate", path)
    reasons = _array(archive, "p1g_reasons", path)
    matched_views = _array(archive, "p1g_matched_view_counts", path)
    selected_views = _array(archive, "p1g_selected_view_counts", path)
    selected_frames = _array(archive, "p1g_selected_frame_ids", path)
    point_counts = _array(archive, "p1g_cropped_point_counts", path)
    residuals = _array(archive, "p1g_face_residuals", path)
    support = _array(archive, "p1g_face_support", path)
    uncertainty = _array(archive, "p1g_face_uncertainty", path)
    supported = _array(archive, "p1g_face_supported", path)
    features = _array(archive, "p1g_feature_vectors", path)
    step_seconds = _array(archive, "p1g_step_total_seconds", path)

    shapes = {
        "p1g_parent_candidate_ids": (count,),
        "p1g_refined_candidate_ids": (count,),
        "p1g_parent_boxes": (count, 6),
        "p1g_parent_corners": (count, 8, 3),
        "p1g_refined_boxes": (count, 6),
        "p1g_refined_corners": (count, 8, 3),
        "p1g_candidate_scores": (count,),
        "p1g_candidate_applied": (count,),
        "p1g_is_candidate": (count,),
        "p1g_reasons": (count,),
        "p1g_matched_view_counts": (count,),
        "p1g_selected_view_counts": (count,),
        "p1g_selected_frame_ids": (count, top_k),
        "p1g_cropped_point_counts": (count,),
        "p1g_face_residuals": (count, 3, 2),
        "p1g_face_support": (count, 3, 2),
        "p1g_face_uncertainty": (count, 3, 2),
        "p1g_face_supported": (count, 3, 2),
        "p1g_step_total_seconds": (count,),
    }
    values = {
        "p1g_parent_candidate_ids": parent_ids,
        "p1g_refined_candidate_ids": child_ids,
        "p1g_parent_boxes": parent_boxes,
        "p1g_parent_corners": parent_corners,
        "p1g_refined_boxes": refined_boxes,
        "p1g_refined_corners": refined_corners,
        "p1g_candidate_scores": scores,
        "p1g_candidate_applied": applied,
        "p1g_is_candidate": is_candidate,
        "p1g_reasons": reasons,
        "p1g_matched_view_counts": matched_views,
        "p1g_selected_view_counts": selected_views,
        "p1g_selected_frame_ids": selected_frames,
        "p1g_cropped_point_counts": point_counts,
        "p1g_face_residuals": residuals,
        "p1g_face_support": support,
        "p1g_face_uncertainty": uncertainty,
        "p1g_face_supported": supported,
        "p1g_step_total_seconds": step_seconds,
    }
    for key, shape in shapes.items():
        if values[key].shape != shape:
            raise ValueError(
                f"{path}: {key} shape {values[key].shape}, expected {shape}"
            )
    if features.ndim != 2 or features.shape[0] != count:
        raise ValueError(f"{path}: invalid p1g_feature_vectors")
    numeric = (
        parent_boxes,
        parent_corners,
        refined_boxes,
        refined_corners,
        scores,
        matched_views,
        selected_views,
        point_counts,
        residuals,
        support,
        uncertainty,
        features,
        step_seconds,
    )
    if any(not np.isfinite(np.asarray(row, dtype=np.float64)).all() for row in numeric):
        raise ValueError(f"{path}: non-finite P1G diagnostics")
    if applied.dtype != np.dtype(bool) or np.any(applied):
        raise ValueError(f"{path}: P1G rows cannot be applied")
    if is_candidate.dtype != np.dtype(bool):
        raise ValueError(f"{path}: p1g_is_candidate must be Boolean")
    if supported.dtype != np.dtype(bool):
        raise ValueError(f"{path}: p1g_face_supported must be Boolean")
    if (
        not np.array_equal(parent_ids, p1_ids[:count])
        or not np.array_equal(parent_boxes, p1_boxes[:count])
        or not np.array_equal(parent_corners, p1_corners[:count])
        or not np.array_equal(scores, p1_scores[:count])
    ):
        raise ValueError(f"{path}: P1G parents disagree with frozen P1S")
    expected_children = np.asarray(
        [f"{value}:p1g" for value in parent_ids.astype(str)]
    )
    if not np.array_equal(child_ids.astype(str), expected_children):
        raise ValueError(f"{path}: invalid one-to-one P1G child ids")
    if len(np.unique(parent_ids)) != count or len(np.unique(child_ids)) != count:
        raise ValueError(f"{path}: duplicate P1G parent/child rows")
    if np.any(refined_boxes[:, 3:] <= 0.0):
        raise ValueError(f"{path}: non-positive refined extent")
    _validate_box_corner_aliases(
        parent_boxes, parent_corners, label="parent", path=path
    )
    _validate_box_corner_aliases(
        refined_boxes, refined_corners, label="refined", path=path
    )
    geometry_changed = np.any(
        refined_corners != parent_corners, axis=(1, 2)
    )
    expected_candidate = (
        reasons.astype(str) == "candidate"
    ) & geometry_changed
    if not np.array_equal(is_candidate, expected_candidate):
        raise ValueError(
            f"{path}: P1G candidate flags disagree with reason/geometry"
        )
    unchanged = ~is_candidate
    if (
        not np.array_equal(refined_boxes[unchanged], parent_boxes[unchanged])
        or not np.array_equal(
            refined_corners[unchanged], parent_corners[unchanged]
        )
    ):
        raise ValueError(f"{path}: rejected P1G row did not fail open")
    if np.any(selected_views > matched_views) or np.any(selected_views > top_k):
        raise ValueError(f"{path}: invalid P1G view counts")
    for index in range(count):
        selected = int(selected_views[index])
        row = selected_frames[index]
        if np.any(row[:selected] < 0) or np.any(row[selected:] != -1):
            raise ValueError(f"{path}: invalid selected frame padding")
        if len(np.unique(row[:selected])) != selected:
            raise ValueError(f"{path}: repeated selected P1G frame")
    runtime = float(_scalar(archive, "p1g_runtime_seconds", path))
    failures = int(_scalar(archive, "p1g_failure_count", path))
    if not np.isfinite(runtime) or runtime < 0.0 or failures < 0:
        raise ValueError(f"{path}: invalid P1G runtime/failure count")
    return {
        "parents": count,
        "candidates": int(np.count_nonzero(is_candidate)),
        "matched_multiview": int(np.count_nonzero(matched_views >= 2)),
        "selected_multiview": int(np.count_nonzero(selected_views >= 2)),
        "runtime_seconds": runtime,
        "failures": failures,
        "reasons": Counter(reasons.astype(str).tolist()),
    }


def validate(
    *,
    scene_list: Path,
    prediction_root: Path,
    diagnostics_root: Path,
    expected_p1s_checkpoint: Path,
) -> dict[str, Any]:
    scenes = _scene_ids(scene_list)
    if not expected_p1s_checkpoint.is_file():
        raise FileNotFoundError(expected_p1s_checkpoint)
    architecture, assignment = _checkpoint_contract(
        expected_p1s_checkpoint
    )
    checkpoint_sha256 = _sha256(expected_p1s_checkpoint)
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
            "P1G prediction set mismatch: "
            f"missing={sorted(expected_predictions-actual_predictions)[:8]}, "
            f"extra={sorted(actual_predictions-expected_predictions)[:8]}"
        )
    if actual_diagnostics != expected_diagnostics:
        raise ValueError(
            "P1G diagnostic set mismatch: "
            f"missing={sorted(expected_diagnostics-actual_diagnostics)[:8]}, "
            f"extra={sorted(actual_diagnostics-expected_diagnostics)[:8]}"
        )

    totals: dict[str, Any] = {
        "parents": 0,
        "candidates": 0,
        "matched_multiview": 0,
        "selected_multiview": 0,
        "runtime_seconds": 0.0,
        "failures": 0,
    }
    reasons: Counter[str] = Counter()
    for scene in scenes:
        prediction = prediction_root / f"{scene}_boxes.pkl"
        diagnostic = diagnostics_root / f"{scene}_tracks.npz"
        if prediction.stat().st_size <= 0 or diagnostic.stat().st_size <= 0:
            raise ValueError(f"empty P1G artifact for {scene}")
        row = _validate_scene(
            diagnostic,
            scene=scene,
            checkpoint_sha256=checkpoint_sha256,
        )
        reasons.update(row.pop("reasons"))
        for key, value in row.items():
            totals[key] += value
    return {
        "schema": SCHEMA,
        "ok": True,
        "stage": "P1G",
        "profile": P1G_PROFILE,
        "parent_stage": "P1S",
        "head_architecture": architecture,
        "target_assignment_scope": assignment,
        "scene_count": len(scenes),
        "checkpoint_sha256": checkpoint_sha256,
        "prediction_root": str(prediction_root.resolve()),
        "diagnostics_root": str(diagnostics_root.resolve()),
        "reason_counts": dict(sorted(reasons.items())),
        **totals,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--diagnostics-root", required=True, type=Path)
    parser.add_argument(
        "--expected-p1s-checkpoint", required=True, type=Path
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = validate(
        scene_list=args.scene_list,
        prediction_root=args.prediction_root,
        diagnostics_root=args.diagnostics_root,
        expected_p1s_checkpoint=args.expected_p1s_checkpoint,
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
