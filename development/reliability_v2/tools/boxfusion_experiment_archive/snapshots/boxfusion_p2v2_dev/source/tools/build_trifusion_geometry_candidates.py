#!/usr/bin/env python3
"""Build deterministic occupancy/MSR candidates from frozen C4 diagnostics.

This is an offline, CPU-only adapter.  It never edits prediction pickles and
never reads ground truth.  For each exported B6 global detection it replays
the already stored Top-K Mask-RGBD view records through the local
occupancy/MSR refiner and writes:

* the strict ragged candidate contract consumed by
  :mod:`tools.report_trifusion_oracles`; and
* a fixed 61-D feature row (12 frozen B6 features, one availability flag,
  and 48 occupancy/MSR features) for the train-only AP50 safety gate.

``candidate_valid`` means that occupancy/MSR produced a non-identity proposal.
``candidate_verified`` additionally enforces only inference-available hard
geometry checks (extent survival, locality, and neighbour-overlap safety).
Neither flag uses ScanNet ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.local_occupancy_msr_refiner import (  # noqa: E402
    LOCAL_OCCUPANCY_MSR_SOURCE,
    OCCUPANCY_MSR_FEATURE_NAMES,
    propose_local_occupancy_msr,
)
from boxfusion.quality_score import QUALITY_FEATURE_NAMES  # noqa: E402
from tools.analyze_fused_oracle import (  # noqa: E402
    load_scene_predictions,
    read_scene_ids,
)
from tools.report_c4_geometry_ablation import (  # noqa: E402
    C4_DIAGNOSTIC_SCHEMA,
    load_c4_diagnostics,
)
from tools.report_trifusion_oracles import (  # noqa: E402
    CORNER_FRAME,
    GEOMETRY_CANDIDATE_SCHEMA,
)


OUTPUT_FORMAT_VERSION = 1
OUTPUT_SUFFIX = "_geometry_candidates.npz"
COMBINED_FEATURE_NAMES = tuple(
    f"b6_original_{name}" for name in QUALITY_FEATURE_NAMES
) + ("b6_original_features_available",) + tuple(
    f"occupancy_msr_{name}" for name in OCCUPANCY_MSR_FEATURE_NAMES
)
assert len(COMBINED_FEATURE_NAMES) == 61

_REQUIRED_VIEW_KEYS = {
    "scene_id",
    "quality_features",
    "result_indices",
    "c4_diagnostics_schema",
    "c4_mutation_enabled",
    "c4_result_indices",
    "c4_source",
    "c4_original_corners",
    "c4_scores",
    "c4_view_points",
    "c4_view_point_mask",
    "c4_view_valid",
    "c4_view_frame_ids",
    "c4_view_quality",
    "c4_view_valid_depth_ratio",
    "c4_view_projection_iou",
    "c4_view_camera_position",
}


def _scalar_text(value: object, *, name: str, path: Path) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.hasobject:
        raise ValueError(f"{path}: {name} must be a scalar string")
    result = array.item()
    if isinstance(result, bytes):
        result = result.decode("utf-8")
    if not isinstance(result, str):
        raise ValueError(f"{path}: {name} must be a scalar string")
    return result


def _diagnostic_path(root: Path, scene_id: str) -> Path:
    path = root / f"{scene_id}_tracks.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _obb_dimensions(corners: np.ndarray) -> np.ndarray:
    values = np.asarray(corners, dtype=np.float64)
    if values.shape != (8, 3) or not np.isfinite(values).all():
        raise ValueError("corners must have finite shape [8,3]")
    dimensions = np.linalg.norm(
        np.stack(
            (
                values[1] - values[0],
                values[3] - values[0],
                values[4] - values[0],
            )
        ),
        axis=1,
    )
    if np.any(dimensions <= 0.0):
        raise ValueError("corners define a degenerate OBB")
    return dimensions


def _aabb_iou(corners_a: np.ndarray, corners_b: np.ndarray) -> float:
    a = np.asarray(corners_a, dtype=np.float64)
    b = np.asarray(corners_b, dtype=np.float64)
    lower = np.maximum(a.min(axis=0), b.min(axis=0))
    upper = np.minimum(a.max(axis=0), b.max(axis=0))
    intersection = float(np.prod(np.maximum(upper - lower, 0.0)))
    volume_a = float(np.prod(a.max(axis=0) - a.min(axis=0)))
    volume_b = float(np.prod(b.max(axis=0) - b.min(axis=0)))
    union = volume_a + volume_b - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def _max_other_iou(
    corners: np.ndarray,
    exported: np.ndarray,
    prediction_index: int,
) -> float:
    indices = [
        index for index in range(len(exported)) if index != prediction_index
    ]
    if not indices:
        return 0.0
    return max(_aabb_iou(corners, exported[index]) for index in indices)


def _view_records(
    raw: Mapping[str, np.ndarray], row: int
) -> tuple[dict[str, Any], ...]:
    view_valid = np.asarray(raw["c4_view_valid"])
    point_mask = np.asarray(raw["c4_view_point_mask"])
    points = np.asarray(raw["c4_view_points"], dtype=np.float64)
    frame_ids = np.asarray(raw["c4_view_frame_ids"])
    quality = np.asarray(raw["c4_view_quality"], dtype=np.float64)
    depth_ratio = np.asarray(
        raw["c4_view_valid_depth_ratio"], dtype=np.float64
    )
    projection = np.asarray(
        raw["c4_view_projection_iou"], dtype=np.float64
    )
    cameras = np.asarray(
        raw["c4_view_camera_position"], dtype=np.float64
    )
    records = []
    for view_index in range(view_valid.shape[1]):
        if not bool(view_valid[row, view_index]):
            continue
        selected = points[row, view_index][point_mask[row, view_index]]
        records.append(
            {
                "frame_id": str(int(frame_ids[row, view_index])),
                "points_world": selected,
                "camera_position": cameras[row, view_index],
                "quality": float(quality[row, view_index]),
                "valid_depth_ratio": float(depth_ratio[row, view_index]),
                "projection_mask_iou": float(projection[row, view_index]),
            }
        )
    return tuple(records)


def _load_raw_diagnostics(
    path: Path, *, expected_scene_id: str
) -> dict[str, np.ndarray]:
    # First run the released strict C4 core loader.
    load_c4_diagnostics(path, expected_scene_id=expected_scene_id)
    with np.load(path, allow_pickle=False) as archive:
        missing = _REQUIRED_VIEW_KEYS - set(archive.files)
        if missing:
            raise ValueError(
                f"{path}: missing occupancy/MSR fields {sorted(missing)}"
            )
        raw = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
        }
    if _scalar_text(
        raw["c4_diagnostics_schema"],
        name="c4_diagnostics_schema",
        path=path,
    ) != C4_DIAGNOSTIC_SCHEMA:
        raise ValueError(f"{path}: unsupported C4 diagnostic schema")
    if bool(np.asarray(raw["c4_mutation_enabled"]).item()):
        raise ValueError(f"{path}: source C4 run is not observer-only")

    rows = len(np.asarray(raw["c4_result_indices"]))
    expected_shapes = {
        "c4_view_points": (rows, 5, 512, 3),
        "c4_view_point_mask": (rows, 5, 512),
        "c4_view_valid": (rows, 5),
        "c4_view_frame_ids": (rows, 5),
        "c4_view_quality": (rows, 5),
        "c4_view_valid_depth_ratio": (rows, 5),
        "c4_view_projection_iou": (rows, 5),
        "c4_view_camera_position": (rows, 5, 3),
    }
    for name, shape in expected_shapes.items():
        if np.asarray(raw[name]).shape != shape:
            raise ValueError(
                f"{path}: {name} must have shape {shape}, "
                f"got {np.asarray(raw[name]).shape}"
            )
    quality = np.asarray(raw["quality_features"])
    if (
        quality.ndim != 2
        or quality.shape[1] != len(QUALITY_FEATURE_NAMES)
        or not np.isfinite(quality).all()
    ):
        raise ValueError(f"{path}: invalid frozen B6 quality_features")
    quality_result_indices = np.asarray(raw["result_indices"])
    if (
        quality_result_indices.shape != (len(quality),)
        or quality_result_indices.dtype.kind not in "iu"
        or np.any(quality_result_indices < 0)
        or len(np.unique(quality_result_indices))
        != len(quality_result_indices)
    ):
        raise ValueError(
            f"{path}: result_indices must uniquely map B6 feature rows"
        )
    return raw


def build_scene_candidates(
    *,
    scene_id: str,
    diagnostic_path: Path,
    prediction_path: Path,
    minimum_extent: float,
    minimum_original_candidate_iou: float,
    maximum_new_neighbour_iou: float,
    maximum_neighbour_iou_increase: float,
    proposal_config: Mapping[str, object] | None = None,
) -> dict[str, np.ndarray]:
    """Return one strict, non-pickled candidate payload."""

    raw = _load_raw_diagnostics(
        Path(diagnostic_path), expected_scene_id=scene_id
    )
    exported_corners, _ = load_scene_predictions(Path(prediction_path))
    result_indices = np.asarray(
        raw["c4_result_indices"], dtype=np.int64
    )
    source = np.asarray(raw["c4_source"]).astype(str)
    original_rows = np.asarray(
        raw["c4_original_corners"], dtype=np.float64
    )
    quality_features = np.asarray(
        raw["quality_features"], dtype=np.float32
    )
    quality_result_indices = np.asarray(
        raw["result_indices"], dtype=np.int64
    )
    quality_lookup = {
        int(result_index): quality_features[row]
        for row, result_index in enumerate(
            quality_result_indices.tolist()
        )
    }
    c4_scores = np.asarray(raw["c4_scores"], dtype=np.float32)
    stable_ids = np.asarray(raw["c4_stable_ids"], dtype=np.int64)

    selected_rows = [
        row
        for row, (prediction_index, row_source) in enumerate(
            zip(result_indices.tolist(), source.tolist())
        )
        if row_source == "global"
        and 0 <= int(prediction_index) < len(exported_corners)
    ]
    selected_prediction_indices = result_indices[selected_rows]
    if len(np.unique(selected_prediction_indices)) != len(
        selected_prediction_indices
    ):
        raise ValueError(
            f"{scene_id}: duplicate C4 result index in global rows"
        )
    paired_original = original_rows[selected_rows]
    if len(paired_original) and not np.allclose(
        paired_original,
        exported_corners[selected_prediction_indices],
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError(
            f"{scene_id}: C4 original corners disagree with exported B6"
        )

    candidate_corners = []
    candidate_ids = []
    candidate_valid = []
    candidate_verified = []
    candidate_features = []
    candidate_reasons = []
    candidate_detail_reasons = []
    candidate_stable_ids = []
    hard_gate_reasons = []

    for row, prediction_index in zip(
        selected_rows, selected_prediction_indices.tolist()
    ):
        original = original_rows[row]
        proposal = propose_local_occupancy_msr(
            original,
            _view_records(raw, row),
            config=proposal_config,
        )
        candidate = np.asarray(
            proposal.candidate_corners, dtype=np.float64
        )
        is_candidate = bool(
            proposal.is_candidate
            and not np.array_equal(candidate, original)
        )
        gate_reason = "verified"
        verified = is_candidate
        if not is_candidate:
            gate_reason = "not_candidate"
        else:
            dimensions = _obb_dimensions(candidate)
            if np.any(dimensions < float(minimum_extent)):
                verified = False
                gate_reason = "extent_survival"
            original_candidate_iou = _aabb_iou(original, candidate)
            if (
                verified
                and original_candidate_iou
                < float(minimum_original_candidate_iou)
            ):
                verified = False
                gate_reason = "original_iou"
            original_other = _max_other_iou(
                original, exported_corners, int(prediction_index)
            )
            candidate_other = _max_other_iou(
                candidate, exported_corners, int(prediction_index)
            )
            if (
                verified
                and candidate_other > float(maximum_new_neighbour_iou)
                and candidate_other - original_other
                > float(maximum_neighbour_iou_increase)
            ):
                verified = False
                gate_reason = "neighbour_overlap"

        candidate_corners.append(candidate)
        candidate_ids.append(
            f"{scene_id}:prediction:{int(prediction_index)}:"
            f"{LOCAL_OCCUPANCY_MSR_SOURCE}:v1"
        )
        candidate_valid.append(is_candidate)
        candidate_verified.append(verified)
        b6_features = quality_lookup.get(int(prediction_index))
        b6_available = b6_features is not None
        if b6_features is None:
            # Rows excluded from the historical per-track B6 diagnostic
            # table still exist in the immutable exported prediction list.
            # Keep them available to the geometry oracle and explicitly mark
            # the missing feature row for the learned gate.  The first slot
            # receives the frozen exported score; all other unavailable
            # evidence is zero rather than guessed.
            b6_features = np.zeros(
                len(QUALITY_FEATURE_NAMES), dtype=np.float32
            )
            b6_features[0] = float(
                np.clip(c4_scores[row], 0.0, 1.0)
            )
        candidate_features.append(
            np.concatenate(
                (
                    np.asarray(b6_features, dtype=np.float32),
                    np.asarray(
                        [float(b6_available)], dtype=np.float32
                    ),
                    np.asarray(proposal.feature_vector, dtype=np.float32),
                )
            )
        )
        candidate_reasons.append(str(proposal.reason))
        candidate_detail_reasons.append(
            json.dumps(
                list(proposal.detail_reasons),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        candidate_stable_ids.append(int(stable_ids[row]))
        hard_gate_reasons.append(gate_reason)

    count = len(selected_rows)
    candidates = (
        np.stack(candidate_corners).astype(np.float32)
        if candidate_corners
        else np.empty((0, 8, 3), dtype=np.float32)
    )
    features = (
        np.stack(candidate_features).astype(np.float32)
        if candidate_features
        else np.empty((0, len(COMBINED_FEATURE_NAMES)), dtype=np.float32)
    )
    maximum_text = max(
        (len(value) for value in candidate_detail_reasons), default=1
    )
    return {
        "schema": np.asarray(GEOMETRY_CANDIDATE_SCHEMA),
        "format_version": np.asarray(
            OUTPUT_FORMAT_VERSION, dtype=np.int64
        ),
        "scene_id": np.asarray(scene_id),
        "corner_frame": np.asarray(CORNER_FRAME),
        "prediction_indices": np.asarray(
            selected_prediction_indices, dtype=np.int64
        ),
        "original_corners": np.asarray(
            paired_original, dtype=np.float32
        ).reshape(count, 8, 3),
        "candidate_offsets": np.arange(
            count + 1, dtype=np.int64
        ),
        "candidate_corners": candidates,
        "candidate_ids": np.asarray(candidate_ids, dtype="<U128"),
        "candidate_sources": np.asarray(
            [LOCAL_OCCUPANCY_MSR_SOURCE] * count, dtype="<U32"
        ),
        "candidate_valid": np.asarray(candidate_valid, dtype=np.bool_),
        "candidate_verified": np.asarray(
            candidate_verified, dtype=np.bool_
        ),
        "candidate_stable_ids": np.asarray(
            candidate_stable_ids, dtype=np.int64
        ),
        "candidate_feature_names": np.asarray(
            COMBINED_FEATURE_NAMES, dtype="<U96"
        ),
        "candidate_features": features,
        "candidate_reason": np.asarray(
            candidate_reasons, dtype="<U64"
        ),
        "candidate_detail_reasons_json": np.asarray(
            candidate_detail_reasons,
            dtype=f"<U{maximum_text}",
        ),
        "candidate_hard_gate_reason": np.asarray(
            hard_gate_reasons, dtype="<U64"
        ),
        "observer_only": np.asarray(True, dtype=np.bool_),
        "uses_ground_truth": np.asarray(False, dtype=np.bool_),
    }


def build_candidate_directory(
    *,
    diagnostics_root: Path,
    prediction_root: Path,
    scene_list: Path,
    output_root: Path,
    minimum_extent: float = 0.40,
    minimum_original_candidate_iou: float = 0.45,
    maximum_new_neighbour_iou: float = 0.35,
    maximum_neighbour_iou_increase: float = 0.15,
    proposal_config: Mapping[str, object] | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    scenes = read_scene_ids(Path(scene_list))
    if not scenes:
        raise ValueError(f"scene list is empty: {scene_list}")
    if output_root.resolve() in {
        diagnostics_root.resolve(),
        prediction_root.resolve(),
    }:
        raise ValueError("output root must be separate from source artifacts")
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_count = valid_count = verified_count = 0
    per_scene = {}
    for scene_id in scenes:
        output_path = output_root / f"{scene_id}{OUTPUT_SUFFIX}"
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"refusing to overwrite candidate artifact: {output_path}"
            )
        payload = build_scene_candidates(
            scene_id=scene_id,
            diagnostic_path=_diagnostic_path(
                diagnostics_root, scene_id
            ),
            prediction_path=prediction_root
            / f"{scene_id}_boxes.pkl",
            minimum_extent=minimum_extent,
            minimum_original_candidate_iou=(
                minimum_original_candidate_iou
            ),
            maximum_new_neighbour_iou=maximum_new_neighbour_iou,
            maximum_neighbour_iou_increase=(
                maximum_neighbour_iou_increase
            ),
            proposal_config=proposal_config,
        )
        np.savez_compressed(output_path, **payload)
        count = len(payload["candidate_corners"])
        valid = int(np.sum(payload["candidate_valid"]))
        verified = int(np.sum(payload["candidate_verified"]))
        candidate_count += count
        valid_count += valid
        verified_count += verified
        per_scene[scene_id] = {
            "candidates": count,
            "valid": valid,
            "verified": verified,
        }
    return {
        "schema": "boxfusion.trifusion.candidate_build_summary",
        "format_version": 1,
        "scenes": len(scenes),
        "candidates": candidate_count,
        "valid": valid_count,
        "verified": verified_count,
        "output_root": str(output_root),
        "per_scene": per_scene,
        "uses_ground_truth": False,
    }


def _parse_config_json(value: str | None) -> Mapping[str, object] | None:
    if value is None:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, Mapping):
        raise ValueError("--proposal-config-json must decode to an object")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--minimum-extent", type=float, default=0.40)
    parser.add_argument(
        "--minimum-original-candidate-iou", type=float, default=0.45
    )
    parser.add_argument(
        "--maximum-new-neighbour-iou", type=float, default=0.35
    )
    parser.add_argument(
        "--maximum-neighbour-iou-increase", type=float, default=0.15
    )
    parser.add_argument("--proposal-config-json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args(argv)
    for name in (
        "minimum_extent",
        "minimum_original_candidate_iou",
        "maximum_new_neighbour_iou",
        "maximum_neighbour_iou_increase",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    summary = build_candidate_directory(
        diagnostics_root=args.diagnostics_root,
        prediction_root=args.prediction_root,
        scene_list=args.scene_list,
        output_root=args.output_root,
        minimum_extent=args.minimum_extent,
        minimum_original_candidate_iou=(
            args.minimum_original_candidate_iou
        ),
        maximum_new_neighbour_iou=args.maximum_new_neighbour_iou,
        maximum_neighbour_iou_increase=(
            args.maximum_neighbour_iou_increase
        ),
        proposal_config=_parse_config_json(args.proposal_config_json),
        overwrite=args.overwrite,
    )
    rendered = json.dumps(
        summary, indent=2, sort_keys=True, allow_nan=False
    )
    print(rendered)
    if args.summary_json is not None:
        if args.summary_json.exists() and not args.overwrite:
            raise FileExistsError(
                f"refusing to overwrite summary: {args.summary_json}"
            )
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
