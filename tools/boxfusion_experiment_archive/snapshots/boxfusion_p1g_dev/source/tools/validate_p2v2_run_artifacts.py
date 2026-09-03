#!/usr/bin/env python3
"""Fail-closed validation for a complete observer-only P2-v2 run.

This validator deliberately does not compare prediction pickle bytes across
independent runs.  The formal-output safety contract is established inside
the same run by the immutable observer flags and zero applied-candidate
counts.  Cross-run numerical drift belongs to the separately measured
nondeterminism audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boxfusion.p2_local_mask_geometry import (  # noqa: E402
    P2V2_DIAGNOSTIC_SCHEMA,
    P2V2_SOURCE,
)
from tools.train_p1_residual_head import read_scene_ids  # noqa: E402
from tools.validate_p2_run_artifacts import (  # noqa: E402
    validate as validate_p2,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_P2V2_PROFILE = "p2v2_local_component_mask_rgbd_observer"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array(
    archive: np.lib.npyio.NpzFile,
    key: str,
    path: Path,
) -> np.ndarray:
    if key not in archive.files:
        raise ValueError(f"{path}: missing {key}")
    value = np.asarray(archive[key])
    if value.dtype.hasobject:
        raise ValueError(f"{path}: object dtype in {key}")
    return value


def _scalar(
    archive: np.lib.npyio.NpzFile,
    key: str,
    path: Path,
) -> np.ndarray:
    value = _array(archive, key, path)
    if value.shape != ():
        raise ValueError(f"{path}: {key} must be a scalar")
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


def _config(
    archive: np.lib.npyio.NpzFile,
    path: Path,
) -> Mapping[str, Any]:
    try:
        value = json.loads(_text(archive, "p2v2_config_json", path))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid p2v2_config_json") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: P2-v2 config must be an object")
    expected = {
        "enabled": True,
        "observer_only": True,
        "mutate": False,
        "collect_diagnostics": True,
    }
    for key, required in expected.items():
        if value.get(key) is not required:
            raise ValueError(f"{path}: unsafe P2-v2 config {key}")
    return value


def _one_dimensional_integer(
    archive: np.lib.npyio.NpzFile,
    key: str,
    path: Path,
    count: int,
) -> np.ndarray:
    value = _array(archive, key, path)
    if (
        value.shape != (count,)
        or not np.issubdtype(value.dtype, np.integer)
    ):
        raise ValueError(f"{path}: {key} must be integer[{count}]")
    return value


def _validate_steps(
    archive: np.lib.npyio.NpzFile,
    path: Path,
    config: Mapping[str, Any],
) -> tuple[int, int]:
    p2_frame_ids = _array(archive, "p2_step_frame_ids", path)
    p2_provider_steps = _array(
        archive, "p2_step_provider_steps", path
    )
    frame_ids = _array(archive, "p2v2_step_frame_ids", path)
    provider_steps = _array(
        archive, "p2v2_step_provider_steps", path
    )
    if (
        p2_frame_ids.ndim != 1
        or p2_provider_steps.shape != p2_frame_ids.shape
        or frame_ids.shape != p2_frame_ids.shape
        or provider_steps.shape != p2_provider_steps.shape
        or len(frame_ids) < 1
    ):
        raise ValueError(f"{path}: invalid P2/P2-v2 step arrays")
    for key, value in (
        ("p2_step_frame_ids", p2_frame_ids),
        ("p2_step_provider_steps", p2_provider_steps),
        ("p2v2_step_frame_ids", frame_ids),
        ("p2v2_step_provider_steps", provider_steps),
    ):
        if not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f"{path}: {key} must be integer")
    if not np.array_equal(p2_frame_ids, frame_ids) or not np.array_equal(
        p2_provider_steps, provider_steps
    ):
        raise ValueError(f"{path}: P2/P2-v2 scheduling is not aligned")

    count = len(frame_ids)
    selected = _one_dimensional_integer(
        archive, "p2v2_step_selected_voxel_counts", path, count
    )
    p2_selected = _one_dimensional_integer(
        archive, "p2_step_selected_voxel_counts", path, count
    )
    occupancy_components = _one_dimensional_integer(
        archive,
        "p2v2_step_occupancy_component_counts",
        path,
        count,
    )
    masks = _one_dimensional_integer(
        archive, "p2v2_step_mask_observation_counts", path, count
    )
    mask_components = _one_dimensional_integer(
        archive, "p2v2_step_mask_component_counts", path, count
    )
    eligible_pairs = _one_dimensional_integer(
        archive, "p2v2_step_eligible_pair_counts", path, count
    )
    candidates = _one_dimensional_integer(
        archive, "p2v2_step_candidate_counts", path, count
    )
    seconds = _array(archive, "p2v2_step_seconds", path)
    failed = _array(archive, "p2v2_step_failed", path)
    errors = _array(archive, "p2v2_step_errors", path)
    if (
        seconds.shape != (count,)
        or not np.issubdtype(seconds.dtype, np.floating)
        or not np.isfinite(seconds).all()
        or np.any(seconds < 0.0)
        or failed.shape != (count,)
        or failed.dtype != np.dtype(bool)
        or errors.shape != (count,)
        or errors.dtype.kind not in {"U", "S"}
    ):
        raise ValueError(f"{path}: invalid P2-v2 status/timing arrays")
    if np.any(failed) or any(str(value) for value in errors.tolist()):
        raise ValueError(f"{path}: P2-v2 contains failed steps")
    if not np.array_equal(selected, p2_selected):
        raise ValueError(
            f"{path}: P2-v2 selected-voxel counts disagree with P2"
        )
    if (
        np.any(selected < 0)
        or np.any(occupancy_components < 0)
        or np.any(masks < 0)
        or np.any(mask_components < 0)
        or np.any(eligible_pairs < 0)
        or np.any(candidates < 0)
        or np.any(occupancy_components > selected)
        or np.any(candidates > eligible_pairs)
        or np.any(candidates > selected)
        or np.any(candidates > mask_components)
    ):
        raise ValueError(f"{path}: impossible P2-v2 step counts")

    max_masks = int(config.get("maximum_masks_per_step", -1))
    max_components = int(config.get("maximum_components_per_mask", -1))
    max_candidates = int(config.get("max_candidates_per_step", -1))
    if min(max_masks, max_components, max_candidates) < 1:
        raise ValueError(f"{path}: invalid bounded P2-v2 config")
    if (
        np.any(masks > max_masks)
        or np.any(mask_components > masks * max_components)
        or np.any(candidates > max_candidates)
    ):
        raise ValueError(f"{path}: P2-v2 step exceeds configured bounds")
    return count, int(np.sum(candidates))


def _bounded_float_vector(
    archive: np.lib.npyio.NpzFile,
    key: str,
    path: Path,
    count: int,
    *,
    lower: float = 0.0,
    upper: float | None = None,
    strictly_positive: bool = False,
) -> np.ndarray:
    value = _array(archive, key, path)
    invalid_lower = value <= lower if strictly_positive else value < lower
    if (
        value.shape != (count,)
        or not np.issubdtype(value.dtype, np.floating)
        or not np.isfinite(value).all()
        or np.any(invalid_lower)
        or (upper is not None and np.any(value > upper))
    ):
        raise ValueError(f"{path}: invalid {key}")
    return value


def _validate_candidates(
    archive: np.lib.npyio.NpzFile,
    path: Path,
    *,
    step_candidate_count: int,
    config: Mapping[str, Any],
) -> int:
    ids = _array(archive, "p2v2_candidate_ids", path)
    parent_ids = _array(
        archive, "p2v2_parent_p2_candidate_ids", path
    )
    mask_ids = _array(archive, "p2v2_mask_source_ids", path)
    if ids.ndim != 1 or ids.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{path}: invalid P2-v2 candidate IDs")
    count = len(ids)
    for key, value in (
        ("p2v2_parent_p2_candidate_ids", parent_ids),
        ("p2v2_mask_source_ids", mask_ids),
    ):
        if value.shape != (count,) or value.dtype.kind not in {"U", "S"}:
            raise ValueError(f"{path}: invalid {key}")
    candidate_ids = [str(value) for value in ids.tolist()]
    if (
        len(set(candidate_ids)) != count
        or any(not value.startswith("p2v2:") for value in candidate_ids)
        or any(not str(value) for value in parent_ids.tolist())
        or any(not str(value) for value in mask_ids.tolist())
    ):
        raise ValueError(f"{path}: invalid or duplicate candidate source IDs")

    boxes = _array(archive, "p2v2_candidate_boxes", path)
    corners = _array(archive, "p2v2_candidate_corners", path)
    parent_boxes = _array(
        archive, "p2v2_candidate_parent_boxes", path
    )
    if (
        boxes.shape != (count, 6)
        or corners.shape != (count, 8, 3)
        or parent_boxes.shape != (count, 6)
        or not np.issubdtype(boxes.dtype, np.floating)
        or not np.issubdtype(corners.dtype, np.floating)
        or not np.issubdtype(parent_boxes.dtype, np.floating)
        or not np.isfinite(boxes).all()
        or not np.isfinite(corners).all()
        or not np.isfinite(parent_boxes).all()
        or np.any(boxes[:, 3:] <= 0.0)
        or np.any(parent_boxes[:, 3:] <= 0.0)
    ):
        raise ValueError(f"{path}: invalid P2-v2 candidate geometry")

    for key in (
        "p2v2_candidate_scores",
        "p2v2_candidate_parent_objectness",
        "p2v2_candidate_occupancy_scores",
        "p2v2_candidate_mask_scores",
        "p2v2_candidate_valid_depth_ratios",
        "p2v2_candidate_parent_iou",
    ):
        _bounded_float_vector(
            archive, key, path, count, lower=0.0, upper=1.0
        )
    _bounded_float_vector(
        archive,
        "p2v2_candidate_normalized_center_distance",
        path,
        count,
        lower=0.0,
    )

    point_counts = _one_dimensional_integer(
        archive,
        "p2v2_candidate_component_point_counts",
        path,
        count,
    )
    voxel_counts = _one_dimensional_integer(
        archive,
        "p2v2_candidate_component_voxel_counts",
        path,
        count,
    )
    selected_inside = _one_dimensional_integer(
        archive,
        "p2v2_candidate_selected_voxels_inside",
        path,
        count,
    )
    if (
        np.any(point_counts < 1)
        or np.any(voxel_counts < 1)
        or np.any(selected_inside < 1)
    ):
        raise ValueError(f"{path}: invalid component support counts")

    anchors = _array(
        archive, "p2v2_candidate_anchor_inside", path
    )
    applied = _array(archive, "p2v2_candidate_applied", path)
    if (
        anchors.shape != (count,)
        or anchors.dtype != np.dtype(bool)
        or not np.all(anchors)
        or applied.shape != (count,)
        or applied.dtype != np.dtype(bool)
        or np.any(applied)
    ):
        raise ValueError(f"{path}: unsafe P2-v2 candidate flags")

    extent_ratios = _array(
        archive, "p2v2_candidate_extent_ratios", path
    )
    center_shift = _array(
        archive, "p2v2_candidate_center_shift_ratios", path
    )
    if (
        extent_ratios.shape != (count, 3)
        or center_shift.shape != (count, 3)
        or not np.issubdtype(extent_ratios.dtype, np.floating)
        or not np.issubdtype(center_shift.dtype, np.floating)
        or not np.isfinite(extent_ratios).all()
        or not np.isfinite(center_shift).all()
        or np.any(extent_ratios <= 0.0)
        or np.any(center_shift < 0.0)
    ):
        raise ValueError(f"{path}: invalid P2-v2 relative geometry")

    max_scene = int(config.get("max_scene_candidates", -1))
    if max_scene < 1:
        raise ValueError(f"{path}: invalid max_scene_candidates")
    if count > max_scene or count > step_candidate_count:
        raise ValueError(
            f"{path}: scene NMS candidate count is impossible"
        )
    return count


def _validate_scene(
    diagnostic: Path,
    *,
    scene: str,
    expected_p2_sha: str,
) -> tuple[int, int, int]:
    with np.load(diagnostic, allow_pickle=False) as archive:
        for key in archive.files:
            if np.asarray(archive[key]).dtype.hasobject:
                raise ValueError(f"{diagnostic}: object dtype in {key}")
        expected_text = {
            "scene_id": scene,
            "p2v2_schema": P2V2_DIAGNOSTIC_SCHEMA,
            "p2v2_stage": "P2V2",
            "p2v2_profile": _P2V2_PROFILE,
            "p2v2_source": P2V2_SOURCE,
        }
        for key, expected in expected_text.items():
            if _text(archive, key, diagnostic) != expected:
                raise ValueError(f"{diagnostic}: invalid {key}")
        if not _text(
            archive, "p2v2_mask_provider", diagnostic
        ).strip():
            raise ValueError(f"{diagnostic}: empty mask provider")
        expected_bools = {
            "p2v2_enabled": True,
            "p2v2_observer_only": True,
            "p2v2_uses_ground_truth": False,
            "p2v2_reads_semantic_labels": False,
            "p2v2_mutation_enabled": False,
            "p2v2_complete": True,
        }
        for key, expected in expected_bools.items():
            if _boolean(archive, key, diagnostic) is not expected:
                raise ValueError(f"{diagnostic}: unsafe {key}")
        if _integer(archive, "p2v2_applied_count", diagnostic) != 0:
            raise ValueError(
                f"{diagnostic}: P2-v2 mutated formal output"
            )
        parent_sha = _text(
            archive,
            "p2v2_parent_p2_checkpoint_sha256",
            diagnostic,
        )
        embedded_p2_sha = _text(
            archive, "p2_checkpoint_sha256", diagnostic
        )
        if (
            _SHA256.fullmatch(parent_sha) is None
            or parent_sha != expected_p2_sha
            or embedded_p2_sha != expected_p2_sha
        ):
            raise ValueError(
                f"{diagnostic}: P2-v2 parent P2 checkpoint mismatch"
            )

        # A class-agnostic geometry observer must not export a semantic
        # candidate stream.  The explicit read-contract scalar above is the
        # only allowed P2-v2 key containing the word "semantic".
        forbidden = [
            key
            for key in archive.files
            if key.startswith("p2v2_candidate_")
            and any(
                token in key
                for token in ("label", "class", "semantic", "clip", "text")
            )
        ]
        if forbidden:
            raise ValueError(
                f"{diagnostic}: semantic P2-v2 candidate field "
                f"{forbidden[0]}"
            )

        config = _config(archive, diagnostic)
        step_count, raw_candidate_count = _validate_steps(
            archive, diagnostic, config
        )
        scene_candidate_count = _validate_candidates(
            archive,
            diagnostic,
            step_candidate_count=raw_candidate_count,
            config=config,
        )
    return step_count, raw_candidate_count, scene_candidate_count


def validate(
    *,
    scene_list: Path,
    prediction_root: Path,
    diagnostics_root: Path,
    expected_p1_checkpoint: Path,
    expected_p2_checkpoint: Path,
) -> dict[str, Any]:
    """Validate P1/P2 ancestry plus the detached P2-v2 stream."""

    # Reuse the complete P2 validator without its optional cross-run
    # prediction comparison.  This checks the immutable P1/P2 parent chain.
    p2_report = validate_p2(
        scene_list=scene_list,
        prediction_root=prediction_root,
        diagnostics_root=diagnostics_root,
        expected_p1_checkpoint=expected_p1_checkpoint,
        expected_p2_checkpoint=expected_p2_checkpoint,
        baseline_prediction_root=None,
    )
    scenes = read_scene_ids(scene_list, role="P2-v2 evaluation")
    expected_p2_sha = _sha256(expected_p2_checkpoint)
    step_count = 0
    raw_candidate_count = 0
    scene_candidate_count = 0
    for scene in scenes:
        diagnostic = diagnostics_root / f"{scene}_tracks.npz"
        steps, raw, selected = _validate_scene(
            diagnostic,
            scene=scene,
            expected_p2_sha=expected_p2_sha,
        )
        step_count += steps
        raw_candidate_count += raw
        scene_candidate_count += selected
    return {
        "schema": "boxfusion.p2v2.run_artifact_validation.v1",
        "scene_count": len(scenes),
        "p2v2_step_count": step_count,
        "p2v2_pre_scene_nms_candidate_count": raw_candidate_count,
        "p2v2_scene_candidate_count": scene_candidate_count,
        "p1_checkpoint_sha256": p2_report["p1_checkpoint_sha256"],
        "p2_checkpoint_sha256": expected_p2_sha,
        "prediction_root": str(prediction_root.resolve()),
        "diagnostics_root": str(diagnostics_root.resolve()),
        "formal_output_safety": {
            "observer_only": True,
            "mutation_enabled": False,
            "applied_count": 0,
            "cross_run_pickle_byte_equality_required": False,
            "basis": (
                "same-run immutable observer contract; independent-run "
                "drift is handled by the nondeterminism audit"
            ),
        },
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = validate(
        scene_list=args.scene_list,
        prediction_root=args.prediction_root,
        diagnostics_root=args.diagnostics_root,
        expected_p1_checkpoint=args.expected_p1_checkpoint,
        expected_p2_checkpoint=args.expected_p2_checkpoint,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
