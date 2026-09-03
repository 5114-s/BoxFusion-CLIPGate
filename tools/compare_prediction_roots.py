#!/usr/bin/env python3
"""Strictly compare corresponding BoxFusion prediction pickles.

The tool is intended for record/replay non-interference checks.  It treats row,
class, score, box, and corner order as part of the prediction contract; it does
not try to rematch geometrically equivalent boxes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import pickle
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PREDICTION_SUFFIX = "_boxes.pkl"
SCENE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class PredictionComparisonError(ValueError):
    """Raised when inputs violate the strict comparison contract."""


@dataclass(frozen=True)
class PredictionScene:
    classes: tuple[int, ...]
    scores: tuple[float, ...]
    corners: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_scene_list(path: str | Path) -> tuple[str, ...]:
    scene_path = Path(path).resolve()
    if not scene_path.is_file():
        raise PredictionComparisonError(f"scene list is not a regular file: {scene_path}")
    rows = scene_path.read_text(encoding="utf-8").splitlines()
    if not rows:
        raise PredictionComparisonError("scene list must contain at least one scene")
    scenes: list[str] = []
    for line_number, row in enumerate(rows, start=1):
        scene = row.strip()
        if not scene:
            raise PredictionComparisonError(
                f"scene list contains an empty row at line {line_number}"
            )
        if scene != row:
            raise PredictionComparisonError(
                f"scene list has surrounding whitespace at line {line_number}"
            )
        if not SCENE_ID_RE.fullmatch(scene) or scene in {".", ".."}:
            raise PredictionComparisonError(
                f"unsafe scene ID at line {line_number}: {scene!r}"
            )
        scenes.append(scene)
    if len(set(scenes)) != len(scenes):
        duplicates = sorted({scene for scene in scenes if scenes.count(scene) > 1})
        raise PredictionComparisonError(
            "scene list contains duplicate IDs: " + ", ".join(duplicates)
        )
    return tuple(scenes)


def _validate_root(root: str | Path, expected_scenes: Sequence[str]) -> Path:
    resolved = Path(root).resolve()
    if not resolved.is_dir():
        raise PredictionComparisonError(f"prediction root is not a directory: {resolved}")
    discovered = {
        path.name[: -len(PREDICTION_SUFFIX)]
        for path in resolved.glob(f"*{PREDICTION_SUFFIX}")
        if path.is_file()
    }
    expected = set(expected_scenes)
    missing = sorted(expected - discovered)
    extra = sorted(discovered - expected)
    if missing or extra:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise PredictionComparisonError(
            f"prediction scene set mismatch for {resolved}: " + "; ".join(details)
        )
    return resolved


def _load_prediction(path: Path) -> PredictionScene:
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise PredictionComparisonError(f"could not load prediction {path}: {exc}") from exc

    if not isinstance(payload, (list, tuple)) or len(payload) != 1:
        raise PredictionComparisonError(
            f"prediction {path} must be a one-element outer sequence"
        )
    rows = payload[0]
    if not isinstance(rows, (list, tuple)):
        raise PredictionComparisonError(f"prediction rows are not a sequence: {path}")

    classes: list[int] = []
    scores: list[float] = []
    boxes: list[np.ndarray] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise PredictionComparisonError(
                f"{path}: row {row_index} must be (class, corners, score)"
            )
        class_id, corners, score = row
        if isinstance(class_id, (bool, np.bool_)) or not isinstance(
            class_id, numbers.Integral
        ):
            raise PredictionComparisonError(
                f"{path}: row {row_index} has a non-integral class ID"
            )
        if isinstance(score, (bool, np.bool_)) or not isinstance(score, numbers.Real):
            raise PredictionComparisonError(
                f"{path}: row {row_index} has a non-numeric score"
            )
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise PredictionComparisonError(
                f"{path}: row {row_index} has a non-finite score"
            )
        try:
            numeric_corners = np.asarray(corners, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise PredictionComparisonError(
                f"{path}: row {row_index} corners are not numeric"
            ) from exc
        if numeric_corners.shape != (8, 3):
            raise PredictionComparisonError(
                f"{path}: row {row_index} corners have shape "
                f"{numeric_corners.shape}, expected (8, 3)"
            )
        if not np.isfinite(numeric_corners).all():
            raise PredictionComparisonError(
                f"{path}: row {row_index} corners contain non-finite values"
            )
        classes.append(int(class_id))
        scores.append(numeric_score)
        boxes.append(numeric_corners)

    corner_array = (
        np.stack(boxes, axis=0)
        if boxes
        else np.empty((0, 8, 3), dtype=np.float64)
    )
    return PredictionScene(tuple(classes), tuple(scores), corner_array)


def _distribution(values_m: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(values_m, dtype=np.float64)
    if values.size == 0:
        meters = {"p50": None, "p95": None, "max": None}
    else:
        p50, p95 = np.percentile(values, (50.0, 95.0))
        meters = {
            "p50": float(p50),
            "p95": float(p95),
            "max": float(values.max()),
        }
    millimeters = {
        key: (None if value is None else value * 1000.0)
        for key, value in meters.items()
    }
    return {"meters": meters, "millimeters": millimeters}


def compare_prediction_roots(
    roots: Sequence[str | Path], scene_list: str | Path
) -> dict[str, Any]:
    if len(roots) not in (2, 3):
        raise PredictionComparisonError("exactly two or three prediction roots are required")

    scene_path = Path(scene_list).resolve()
    scenes = read_scene_list(scene_path)
    resolved_roots = tuple(_validate_root(root, scenes) for root in roots)
    if len(set(resolved_roots)) != len(resolved_roots):
        raise PredictionComparisonError("prediction roots must resolve to distinct directories")

    loaded: list[dict[str, PredictionScene]] = []
    root_file_hashes: list[dict[str, str]] = []
    for root in resolved_roots:
        by_scene: dict[str, PredictionScene] = {}
        hashes: dict[str, str] = {}
        for scene in scenes:
            path = root / f"{scene}{PREDICTION_SUFFIX}"
            by_scene[scene] = _load_prediction(path)
            hashes[scene] = _sha256(path)
        loaded.append(by_scene)
        root_file_hashes.append(hashes)

    row_counts: dict[str, int] = {}
    for scene in scenes:
        reference = loaded[0][scene]
        row_counts[scene] = len(reference.classes)
        for root_index in range(1, len(loaded)):
            candidate = loaded[root_index][scene]
            if len(candidate.classes) != len(reference.classes):
                raise PredictionComparisonError(
                    f"row count mismatch in {scene}: root 0 has "
                    f"{len(reference.classes)}, root {root_index} has "
                    f"{len(candidate.classes)}"
                )
            if candidate.classes != reference.classes:
                mismatch = next(
                    index
                    for index, pair in enumerate(zip(reference.classes, candidate.classes))
                    if pair[0] != pair[1]
                )
                raise PredictionComparisonError(
                    f"class order mismatch in {scene}, row {mismatch}: root 0 has "
                    f"{reference.classes[mismatch]}, root {root_index} has "
                    f"{candidate.classes[mismatch]}"
                )
            if candidate.scores != reference.scores:
                mismatch = next(
                    index
                    for index, pair in enumerate(zip(reference.scores, candidate.scores))
                    if pair[0] != pair[1]
                )
                raise PredictionComparisonError(
                    f"score order/value mismatch in {scene}, row {mismatch}: root 0 has "
                    f"{reference.scores[mismatch]!r}, root {root_index} has "
                    f"{candidate.scores[mismatch]!r}"
                )

    pairwise: list[dict[str, Any]] = []
    for root_a, root_b in combinations(range(len(loaded)), 2):
        all_box_errors: list[float] = []
        scene_statistics: dict[str, Any] = {}
        worst: dict[str, Any] | None = None
        for scene in scenes:
            a = loaded[root_a][scene].corners
            b = loaded[root_b][scene].corners
            corner_errors = np.linalg.norm(a - b, axis=2)
            box_errors = (
                corner_errors.max(axis=1)
                if corner_errors.shape[0]
                else np.empty((0,), dtype=np.float64)
            )
            values = box_errors.tolist()
            all_box_errors.extend(values)
            scene_statistics[scene] = {
                "box_count": int(box_errors.size),
                "per_box_max_corner_euclidean_error": _distribution(values),
            }
            if corner_errors.size:
                flat_index = int(np.argmax(corner_errors))
                box_index, corner_index = np.unravel_index(
                    flat_index, corner_errors.shape
                )
                error_m = float(corner_errors[box_index, corner_index])
                if worst is None or error_m > worst["error_m"]:
                    worst = {
                        "scene": scene,
                        "box_row": int(box_index),
                        "corner_index": int(corner_index),
                        "error_m": error_m,
                        "error_mm": error_m * 1000.0,
                    }

        pairwise.append(
            {
                "root_a": root_a,
                "root_b": root_b,
                "box_count": len(all_box_errors),
                "nonzero_box_count": sum(value != 0.0 for value in all_box_errors),
                "per_box_max_corner_euclidean_error": _distribution(all_box_errors),
                "worst_corner": worst,
                "scenes": scene_statistics,
            }
        )

    return {
        "schema_version": 1,
        "comparison_contract": {
            "scene_set": "exact",
            "row_order": "exact",
            "class_order": "exact",
            "score_order_and_value": "exact_no_tolerance",
            "geometry": "corresponding rows and corresponding corners",
        },
        "scene_list": {
            "path": str(scene_path),
            "sha256": _sha256(scene_path),
            "count": len(scenes),
            "scenes": list(scenes),
        },
        "roots": [
            {
                "index": index,
                "path": str(root),
                "prediction_sha256_by_scene": root_file_hashes[index],
            }
            for index, root in enumerate(resolved_roots)
        ],
        "validation": {
            "passed": True,
            "scene_sets_equal": True,
            "row_counts_equal": True,
            "class_order_equal": True,
            "score_order_and_value_equal": True,
            "scene_count": len(scenes),
            "total_box_count": sum(row_counts.values()),
            "row_count_by_scene": row_counts,
        },
        "pairwise": pairwise,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly compare two or three BoxFusion prediction roots and report "
            "corresponding-corner geometry drift as JSON."
        )
    )
    parser.add_argument(
        "--root",
        action="append",
        required=True,
        help="Prediction root; pass exactly two or three times.",
    )
    parser.add_argument("--scene-list", required=True)
    parser.add_argument(
        "--output",
        help="Optional new JSON file. Existing files are never overwritten.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        result = compare_prediction_roots(arguments.root, arguments.scene_list)
        rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if arguments.output:
            output_path = Path(arguments.output)
            if not output_path.parent.is_dir():
                raise PredictionComparisonError(
                    f"output parent is not a directory: {output_path.parent}"
                )
            try:
                with output_path.open("x", encoding="utf-8") as handle:
                    handle.write(rendered)
            except FileExistsError as exc:
                raise PredictionComparisonError(
                    f"refusing to overwrite existing output: {output_path}"
                ) from exc
    except PredictionComparisonError as exc:
        parser.error(str(exc))
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
