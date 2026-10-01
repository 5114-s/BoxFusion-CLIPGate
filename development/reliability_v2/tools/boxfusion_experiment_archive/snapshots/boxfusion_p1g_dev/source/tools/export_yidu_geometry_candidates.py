#!/usr/bin/env python3
"""Export YiDu observer rows to the strict geometry-candidate contract.

The exporter is deliberately offline and ground-truth free.  It validates
that the source run is one exact, non-mutating YiDu observer stage and then
exports at most one selected candidate for each mapped B6 prediction.

Prediction pickle files are trusted local BoxFusion artifacts.  Do not use
this tool with untrusted pickle input.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.yidu_ablation import (  # noqa: E402
    YIDU_MODULES,
    YIDU_SCHEMA,
    YIDU_STAGE_MODULE_MATRIX,
    YIDU_STAGE_TO_PROFILE,
    resolve_yidu_stage,
)
from boxfusion.yidu_local_observer import (  # noqa: E402
    YIDU_GATE_FEATURE_DIM,
    YIDU_GATE_FEATURE_NAMES,
    YIDU_LOCAL_OBSERVER_SCHEMA,
)
from tools.analyze_fused_oracle import (  # noqa: E402
    load_scene_predictions,
    read_scene_ids,
)
from tools.report_trifusion_oracles import (  # noqa: E402
    CORNER_FRAME,
    GEOMETRY_CANDIDATE_SCHEMA,
    load_geometry_candidates,
)


OUTPUT_FORMAT_VERSION = 1
OUTPUT_SUFFIX = "_geometry_candidates.npz"
IDENTITY_ATOL = 1.0e-6
MINIMUM_AABB_EXTENT = 1.0e-8

_REQUIRED_KEYS = {
    "scene_id",
    "yidu_diagnostics_schema",
    "yidu_ablation_schema",
    "yidu_stage",
    "yidu_profile",
    "yidu_enabled",
    "yidu_mutation_enabled",
    "yidu_applied_count",
    "yidu_modules_json",
    "yidu_result_indices",
    "yidu_stable_ids",
    "yidu_attempted",
    "yidu_valid",
    "yidu_applied",
    "yidu_selected_source",
    "yidu_original_corners",
    "yidu_selected_candidate_corners",
    "yidu_gate_feature_names",
    "yidu_gate_features",
    "yidu_gate_evaluated",
    "yidu_gate_accepted",
}


def _scalar_text(value: object, *, name: str, path: Path) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.hasobject:
        raise ValueError(f"{path}: {name} must be a scalar string")
    result = array.item()
    if isinstance(result, bytes):
        result = result.decode("utf-8")
    if not isinstance(result, str) or not result:
        raise ValueError(f"{path}: {name} must be a non-empty scalar string")
    return result


def _scalar_bool(value: object, *, name: str, path: Path) -> bool:
    array = np.asarray(value)
    if array.shape != () or array.dtype != np.bool_:
        raise ValueError(f"{path}: {name} must be a Boolean scalar")
    return bool(array.item())


def _scalar_integer(value: object, *, name: str, path: Path) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iu":
        raise ValueError(f"{path}: {name} must be an integer scalar")
    return int(array.item())


def _vector(
    raw: Mapping[str, np.ndarray],
    name: str,
    *,
    rows: int,
    path: Path,
) -> np.ndarray:
    array = np.asarray(raw[name])
    if array.shape != (rows,) or array.dtype.hasobject:
        raise ValueError(f"{path}: {name} must have shape [{rows}]")
    return array


def _boolean_vector(
    raw: Mapping[str, np.ndarray],
    name: str,
    *,
    rows: int,
    path: Path,
) -> np.ndarray:
    array = _vector(raw, name, rows=rows, path=path)
    if array.dtype != np.bool_:
        raise ValueError(
            f"{path}: {name} must have Boolean shape [{rows}]"
        )
    return np.asarray(array, dtype=np.bool_)


def _integer_vector(
    raw: Mapping[str, np.ndarray],
    name: str,
    *,
    rows: int,
    path: Path,
) -> np.ndarray:
    array = _vector(raw, name, rows=rows, path=path)
    if array.dtype.kind not in "iu":
        raise ValueError(
            f"{path}: {name} must have integer shape [{rows}]"
        )
    return np.asarray(array, dtype=np.int64)


def _string_vector(
    raw: Mapping[str, np.ndarray],
    name: str,
    *,
    rows: int,
    path: Path,
) -> tuple[str, ...]:
    array = _vector(raw, name, rows=rows, path=path)
    output: list[str] = []
    for value in array.tolist():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"{path}: {name} entries must be non-empty strings"
            )
        output.append(value)
    return tuple(output)


def _corner_rows(
    raw: Mapping[str, np.ndarray],
    name: str,
    *,
    rows: int,
    path: Path,
    require_finite: bool,
) -> np.ndarray:
    try:
        array = np.asarray(raw[name], dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: {name} must be numeric") from error
    if array.shape != (rows, 8, 3):
        raise ValueError(
            f"{path}: {name} must have shape [{rows},8,3]"
        )
    if require_finite and not np.isfinite(array).all():
        raise ValueError(f"{path}: {name} contains non-finite corners")
    return array


def _is_non_degenerate(corners: np.ndarray) -> bool:
    if corners.shape != (8, 3) or not np.isfinite(corners).all():
        return False
    extents = np.max(corners, axis=0) - np.min(corners, axis=0)
    return bool(np.all(extents > MINIMUM_AABB_EXTENT))


def _load_diagnostics(
    path: Path,
    *,
    scene_id: str,
    expected_stage: str | None,
) -> tuple[dict[str, np.ndarray], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = _REQUIRED_KEYS - set(archive.files)
        if missing:
            raise ValueError(
                f"{path}: missing YiDu fields {sorted(missing)}"
            )
        raw = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
        }

    stored_scene = _scalar_text(
        raw["scene_id"], name="scene_id", path=path
    )
    if stored_scene != scene_id:
        raise ValueError(
            f"{path}: scene {stored_scene!r} does not match {scene_id!r}"
        )
    diagnostics_schema = _scalar_text(
        raw["yidu_diagnostics_schema"],
        name="yidu_diagnostics_schema",
        path=path,
    )
    if diagnostics_schema != YIDU_LOCAL_OBSERVER_SCHEMA:
        raise ValueError(
            f"{path}: unsupported YiDu diagnostics schema "
            f"{diagnostics_schema!r}"
        )
    ablation_schema = _scalar_text(
        raw["yidu_ablation_schema"],
        name="yidu_ablation_schema",
        path=path,
    )
    if ablation_schema != YIDU_SCHEMA:
        raise ValueError(
            f"{path}: unsupported YiDu ablation schema "
            f"{ablation_schema!r}"
        )

    stage = resolve_yidu_stage(
        _scalar_text(raw["yidu_stage"], name="yidu_stage", path=path)
    )
    if stage == "B0":
        raise ValueError(f"{path}: B0 has no YiDu geometry candidates")
    if expected_stage is not None:
        canonical_expected = resolve_yidu_stage(expected_stage)
        if canonical_expected == "B0":
            raise ValueError(
                "expected_stage must be one of A1,A2,A3,A4,A5,A6"
            )
        if stage != canonical_expected:
            raise ValueError(
                f"{path}: YiDu stage {stage} does not match "
                f"expected {canonical_expected}"
            )
    profile = _scalar_text(
        raw["yidu_profile"], name="yidu_profile", path=path
    )
    if profile != YIDU_STAGE_TO_PROFILE[stage]:
        raise ValueError(
            f"{path}: YiDu profile {profile!r} is not canonical for {stage}"
        )
    if not _scalar_bool(
        raw["yidu_enabled"], name="yidu_enabled", path=path
    ):
        raise ValueError(f"{path}: YiDu observer was not enabled")
    if _scalar_bool(
        raw["yidu_mutation_enabled"],
        name="yidu_mutation_enabled",
        path=path,
    ):
        raise ValueError(f"{path}: YiDu source is not observer-only")
    if _scalar_integer(
        raw["yidu_applied_count"],
        name="yidu_applied_count",
        path=path,
    ) != 0:
        raise ValueError(f"{path}: YiDu applied count must be zero")

    modules_text = _scalar_text(
        raw["yidu_modules_json"], name="yidu_modules_json", path=path
    )
    try:
        modules = json.loads(modules_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid yidu_modules_json") from error
    expected_modules = dict(YIDU_STAGE_MODULE_MATRIX[stage])
    if (
        not isinstance(modules, dict)
        or set(modules) != set(YIDU_MODULES)
        or any(type(value) is not bool for value in modules.values())
        or modules != expected_modules
    ):
        raise ValueError(
            f"{path}: YiDu module matrix disagrees with stage {stage}"
        )
    return raw, stage


def export_scene(
    *,
    scene_id: str,
    diagnostic_path: Path,
    prediction_path: Path,
    expected_stage: str | None = None,
) -> dict[str, np.ndarray]:
    """Build one in-memory geometry artifact without writing to disk."""

    diagnostic_path = Path(diagnostic_path)
    prediction_path = Path(prediction_path)
    raw, stage = _load_diagnostics(
        diagnostic_path,
        scene_id=scene_id,
        expected_stage=expected_stage,
    )
    prediction_corners, _ = load_scene_predictions(prediction_path)

    result_indices_array = np.asarray(raw["yidu_result_indices"])
    if (
        result_indices_array.ndim != 1
        or result_indices_array.dtype.kind not in "iu"
    ):
        raise ValueError(
            f"{diagnostic_path}: yidu_result_indices must be integer [N]"
        )
    result_indices = np.asarray(result_indices_array, dtype=np.int64)
    rows = len(result_indices)
    if (
        np.any(result_indices < 0)
        or np.any(result_indices >= len(prediction_corners))
        or len(np.unique(result_indices)) != rows
    ):
        raise ValueError(
            f"{diagnostic_path}: yidu_result_indices must uniquely map "
            "exported predictions"
        )

    stable_ids = _integer_vector(
        raw,
        "yidu_stable_ids",
        rows=rows,
        path=diagnostic_path,
    )
    if np.any(stable_ids < 0) or len(np.unique(stable_ids)) != rows:
        raise ValueError(
            f"{diagnostic_path}: yidu_stable_ids must be unique "
            "non-negative integers"
        )
    attempted = _boolean_vector(
        raw, "yidu_attempted", rows=rows, path=diagnostic_path
    )
    valid = _boolean_vector(
        raw, "yidu_valid", rows=rows, path=diagnostic_path
    )
    applied = _boolean_vector(
        raw, "yidu_applied", rows=rows, path=diagnostic_path
    )
    if np.any(applied):
        raise ValueError(f"{diagnostic_path}: YiDu applied rows must be empty")
    sources = _string_vector(
        raw,
        "yidu_selected_source",
        rows=rows,
        path=diagnostic_path,
    )
    gate_evaluated = _boolean_vector(
        raw,
        "yidu_gate_evaluated",
        rows=rows,
        path=diagnostic_path,
    )
    gate_accepted = _boolean_vector(
        raw,
        "yidu_gate_accepted",
        rows=rows,
        path=diagnostic_path,
    )
    if np.any(gate_accepted & ~gate_evaluated):
        raise ValueError(
            f"{diagnostic_path}: accepted YiDu gates must be evaluated"
        )
    if stage != "A6" and (
        np.any(gate_evaluated) or np.any(gate_accepted)
    ):
        raise ValueError(
            f"{diagnostic_path}: AP50 gate evidence is invalid before A6"
        )

    original = _corner_rows(
        raw,
        "yidu_original_corners",
        rows=rows,
        path=diagnostic_path,
        require_finite=True,
    )
    selected = _corner_rows(
        raw,
        "yidu_selected_candidate_corners",
        rows=rows,
        path=diagnostic_path,
        require_finite=False,
    )
    expected_original = np.asarray(
        prediction_corners[result_indices], dtype=np.float32
    ).reshape(rows, 8, 3)
    if not np.array_equal(original, expected_original):
        raise ValueError(
            f"{scene_id}: YiDu original corners disagree with "
            "exported B6 predictions"
        )
    for row in range(rows):
        if not _is_non_degenerate(original[row]):
            raise ValueError(
                f"{diagnostic_path}: degenerate original corners at row {row}"
            )

    names_array = np.asarray(raw["yidu_gate_feature_names"])
    if (
        names_array.ndim != 1
        or names_array.dtype.hasobject
        or tuple(str(value) for value in names_array.tolist())
        != YIDU_GATE_FEATURE_NAMES
    ):
        raise ValueError(
            f"{diagnostic_path}: YiDu gate feature names do not match "
            f"the fixed {YIDU_GATE_FEATURE_DIM}-D schema"
        )
    gate_features = np.asarray(
        raw["yidu_gate_features"], dtype=np.float32
    )
    if (
        gate_features.shape != (rows, YIDU_GATE_FEATURE_DIM)
        or not np.isfinite(gate_features).all()
    ):
        raise ValueError(
            f"{diagnostic_path}: yidu_gate_features must have finite "
            f"shape [{rows},{YIDU_GATE_FEATURE_DIM}]"
        )

    offsets = [0]
    candidate_corners: list[np.ndarray] = []
    candidate_ids: list[str] = []
    candidate_sources: list[str] = []
    candidate_stable_ids: list[int] = []
    candidate_features: list[np.ndarray] = []
    candidate_verified: list[bool] = []
    candidate_gate_evaluated: list[bool] = []
    candidate_gate_accepted: list[bool] = []
    for row, prediction_index in enumerate(result_indices.tolist()):
        candidate = selected[row]
        qualifies = bool(attempted[row] and valid[row])
        qualifies = qualifies and sources[row] != "original"
        qualifies = qualifies and _is_non_degenerate(candidate)
        qualifies = qualifies and not np.allclose(
            candidate,
            original[row],
            rtol=0.0,
            atol=IDENTITY_ATOL,
        )
        if qualifies:
            candidate_corners.append(
                np.asarray(candidate, dtype=np.float32).copy()
            )
            candidate_ids.append(
                f"{scene_id}:yidu:{stage}:prediction:"
                f"{int(prediction_index)}:{sources[row]}:v1"
            )
            candidate_sources.append(sources[row])
            candidate_stable_ids.append(int(stable_ids[row]))
            candidate_features.append(gate_features[row].copy())
            candidate_verified.append(
                bool(
                    stage != "A6"
                    or (
                        gate_evaluated[row]
                        and gate_accepted[row]
                    )
                )
            )
            candidate_gate_evaluated.append(bool(gate_evaluated[row]))
            candidate_gate_accepted.append(bool(gate_accepted[row]))
        offsets.append(len(candidate_corners))

    count = len(candidate_corners)
    flattened_corners = (
        np.stack(candidate_corners).astype(np.float32)
        if candidate_corners
        else np.empty((0, 8, 3), dtype=np.float32)
    )
    flattened_features = (
        np.stack(candidate_features).astype(np.float32)
        if candidate_features
        else np.empty(
            (0, YIDU_GATE_FEATURE_DIM), dtype=np.float32
        )
    )
    return {
        "schema": np.asarray(GEOMETRY_CANDIDATE_SCHEMA),
        "format_version": np.asarray(
            OUTPUT_FORMAT_VERSION, dtype=np.int64
        ),
        "scene_id": np.asarray(scene_id),
        "corner_frame": np.asarray(CORNER_FRAME),
        "prediction_indices": result_indices,
        "original_corners": original,
        "candidate_offsets": np.asarray(offsets, dtype=np.int64),
        "candidate_corners": flattened_corners,
        "candidate_ids": np.asarray(candidate_ids, dtype="<U192"),
        "candidate_sources": np.asarray(
            candidate_sources, dtype="<U96"
        ),
        "candidate_valid": np.ones(count, dtype=np.bool_),
        "candidate_verified": np.asarray(
            candidate_verified, dtype=np.bool_
        ),
        "candidate_stable_ids": np.asarray(
            candidate_stable_ids, dtype=np.int64
        ),
        "candidate_feature_names": np.asarray(
            YIDU_GATE_FEATURE_NAMES, dtype="<U96"
        ),
        "candidate_features": flattened_features,
        "candidate_gate_evaluated": np.asarray(
            candidate_gate_evaluated, dtype=np.bool_
        ),
        "candidate_gate_accepted": np.asarray(
            candidate_gate_accepted, dtype=np.bool_
        ),
        "yidu_stage": np.asarray(stage),
        "observer_only": np.asarray(True, dtype=np.bool_),
        "uses_ground_truth": np.asarray(False, dtype=np.bool_),
    }


def _write_npz_atomic_no_replace(
    path: Path,
    payload: Mapping[str, np.ndarray],
    *,
    scene_id: str,
) -> None:
    """Write one validated NPZ atomically without replacing an existing file."""

    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite geometry artifact: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".npz",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        load_geometry_candidates(
            temporary_path, expected_scene_id=scene_id
        )
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite geometry artifact: {path}"
            ) from None
    finally:
        temporary_path.unlink(missing_ok=True)


def export_directory(
    *,
    diagnostics_root: Path,
    prediction_root: Path,
    scene_list: Path,
    output_root: Path,
    expected_stage: str,
) -> dict[str, object]:
    """Export a scene list using one frozen stage and isolated output root."""

    diagnostics_root = Path(diagnostics_root)
    prediction_root = Path(prediction_root)
    output_root = Path(output_root)
    stage = resolve_yidu_stage(expected_stage)
    if stage == "B0":
        raise ValueError("expected_stage must be one of A1,A2,A3,A4,A5,A6")
    scenes = read_scene_ids(Path(scene_list))
    if not scenes:
        raise ValueError(f"scene list is empty: {scene_list}")
    if output_root.resolve() in {
        diagnostics_root.resolve(),
        prediction_root.resolve(),
    }:
        raise ValueError(
            "output root must differ from diagnostics and prediction roots"
        )

    destinations = [
        output_root / f"{scene_id}{OUTPUT_SUFFIX}"
        for scene_id in scenes
    ]
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(
            f"refusing to overwrite geometry artifact: {existing[0]}"
        )

    candidate_count = 0
    verified_count = 0
    prediction_rows = 0
    per_scene: dict[str, dict[str, int]] = {}
    for scene_id, destination in zip(scenes, destinations):
        payload = export_scene(
            scene_id=scene_id,
            diagnostic_path=(
                diagnostics_root / f"{scene_id}_tracks.npz"
            ),
            prediction_path=(
                prediction_root / f"{scene_id}_boxes.pkl"
            ),
            expected_stage=stage,
        )
        _write_npz_atomic_no_replace(
            destination, payload, scene_id=scene_id
        )
        candidates = len(payload["candidate_corners"])
        verified = int(
            np.count_nonzero(payload["candidate_verified"])
        )
        rows = len(payload["prediction_indices"])
        candidate_count += candidates
        verified_count += verified
        prediction_rows += rows
        per_scene[scene_id] = {
            "prediction_rows": rows,
            "candidates": candidates,
            "verified": verified,
        }
    return {
        "schema": "boxfusion.yidu.geometry_candidate_export_summary",
        "format_version": 1,
        "stage": stage,
        "scenes": len(scenes),
        "prediction_rows": prediction_rows,
        "candidates": candidate_count,
        "valid": candidate_count,
        "verified": verified_count,
        "output_root": str(output_root),
        "uses_ground_truth": False,
        "per_scene": per_scene,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export one frozen YiDu observer stage to offline geometry "
            "candidates without modifying predictions."
        )
    )
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("A1", "A2", "A3", "A4", "A5", "A6"),
        required=True,
    )
    args = parser.parse_args(argv)
    summary = export_directory(
        diagnostics_root=args.diagnostics_root,
        prediction_root=args.prediction_root,
        scene_list=args.scene_list,
        output_root=args.output_root,
        expected_stage=args.stage,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
