#!/usr/bin/env python3
"""Report offline YiDu geometry-candidate IoU deltas without mutation.

Each candidate is compared with the ground-truth box selected by its frozen
*original* prediction.  The target is never re-matched for the candidate, so a
candidate cannot receive credit for jumping to a neighbouring object.  This is
an offline diagnostic only: geometry artifacts, prediction pickles, ScanNet
ground truth, and scan metadata are opened read-only and never rewritten.
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
    resolve_yidu_stage,
)
from tools.analyze_fused_oracle import (  # noqa: E402
    center_size_to_minmax,
    corners_to_minmax,
    load_axis_alignment,
    load_scene_predictions,
    pairwise_aabb_iou,
    read_scene_ids,
    transform_corners,
)
from tools.export_yidu_geometry_candidates import (  # noqa: E402
    OUTPUT_FORMAT_VERSION,
    OUTPUT_SUFFIX,
)
from tools.report_trifusion_oracles import (  # noqa: E402
    CORNER_FRAME,
    load_geometry_candidates,
)


REPORT_SCHEMA = "boxfusion.yidu.candidate_delta_report"
REPORT_FORMAT_VERSION = 1
DELTA_EPSILON = 1.0e-6


def _scalar_text(value: object, *, name: str, path: Path) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.hasobject:
        raise ValueError(f"{path}: {name} must be a scalar string")
    item = array.item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    if not isinstance(item, str) or not item:
        raise ValueError(f"{path}: {name} must be a non-empty string")
    return item


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


def _boolean_rows(
    value: object,
    *,
    name: str,
    rows: int,
    path: Path,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (rows,) or array.dtype != np.bool_:
        raise ValueError(
            f"{path}: {name} must have Boolean shape [{rows}]"
        )
    return np.asarray(array, dtype=np.bool_)


def _load_export_provenance(
    path: Path,
    *,
    candidate_count: int,
    candidate_verified: np.ndarray,
) -> str:
    if path.suffix.lower() != ".npz":
        raise ValueError(f"{path}: YiDu geometry artifact must be NPZ")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "format_version",
            "corner_frame",
            "observer_only",
            "uses_ground_truth",
            "yidu_stage",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(
                f"{path}: missing YiDu export provenance "
                f"{sorted(missing)}"
            )
        version = _scalar_integer(
            archive["format_version"],
            name="format_version",
            path=path,
        )
        corner_frame = _scalar_text(
            archive["corner_frame"], name="corner_frame", path=path
        )
        observer_only = _scalar_bool(
            archive["observer_only"], name="observer_only", path=path
        )
        uses_ground_truth = _scalar_bool(
            archive["uses_ground_truth"],
            name="uses_ground_truth",
            path=path,
        )
        stage = resolve_yidu_stage(
            _scalar_text(
                archive["yidu_stage"], name="yidu_stage", path=path
            )
        )
        if stage == "B0":
            raise ValueError(f"{path}: B0 has no YiDu candidates")
        if stage == "A6":
            gate_required = {
                "candidate_gate_evaluated",
                "candidate_gate_accepted",
            }
            gate_missing = gate_required - set(archive.files)
            if gate_missing:
                raise ValueError(
                    f"{path}: missing A6 gate provenance "
                    f"{sorted(gate_missing)}"
                )
            gate_evaluated = _boolean_rows(
                archive["candidate_gate_evaluated"],
                name="candidate_gate_evaluated",
                rows=candidate_count,
                path=path,
            )
            gate_accepted = _boolean_rows(
                archive["candidate_gate_accepted"],
                name="candidate_gate_accepted",
                rows=candidate_count,
                path=path,
            )
            if np.any(gate_accepted & ~gate_evaluated):
                raise ValueError(
                    f"{path}: A6 accepted candidates must be evaluated"
                )
            if not np.array_equal(
                gate_accepted,
                np.asarray(candidate_verified, dtype=np.bool_),
            ):
                raise ValueError(
                    f"{path}: A6 candidate_verified must exactly equal "
                    "candidate_gate_accepted"
                )
    if version != OUTPUT_FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported YiDu export version")
    if corner_frame != CORNER_FRAME:
        raise ValueError(f"{path}: unsupported corner frame")
    if not observer_only:
        raise ValueError(f"{path}: source geometry is not observer-only")
    if uses_ground_truth:
        raise ValueError(
            f"{path}: runtime geometry export must not use ground truth"
        )
    return stage


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "mean": None,
            "q10": None,
            "q50": None,
            "q90": None,
        }
    quantiles = np.quantile(array, (0.10, 0.50, 0.90))
    return {
        "mean": float(np.mean(array)),
        "q10": float(quantiles[0]),
        "q50": float(quantiles[1]),
        "q90": float(quantiles[2]),
    }


def _ratio(numerator: int, denominator: int) -> float:
    return (
        0.0
        if denominator <= 0
        else float(numerator / float(denominator))
    )


def _summarize(
    records: Sequence[Mapping[str, Any]],
    *,
    scene_count: int,
    prediction_count: int,
    geometry_prediction_rows: int,
) -> dict[str, object]:
    count = len(records)
    deltas = np.asarray(
        [float(row["delta"]) for row in records], dtype=np.float64
    )
    improved = int(np.count_nonzero(deltas > DELTA_EPSILON))
    harmed = int(np.count_nonzero(deltas < -DELTA_EPSILON))
    identity = count - improved - harmed
    covered_rows = {
        (str(row["scene_id"]), int(row["prediction_index"]))
        for row in records
    }
    covered_scenes = {str(row["scene_id"]) for row in records}
    cross25 = sum(
        float(row["original_iou"]) < 0.25
        and float(row["candidate_iou"]) >= 0.25
        for row in records
    )
    cross50 = sum(
        float(row["original_iou"]) < 0.50
        and float(row["candidate_iou"]) >= 0.50
        for row in records
    )
    corner_identity = sum(
        bool(row["corner_identity"]) for row in records
    )
    return {
        "candidates": count,
        "original_iou": _distribution(
            [float(row["original_iou"]) for row in records]
        ),
        "candidate_iou": _distribution(
            [float(row["candidate_iou"]) for row in records]
        ),
        "delta": _distribution(deltas.tolist()),
        "improved": improved,
        "harmed": harmed,
        "identity": identity,
        "corner_identity": int(corner_identity),
        "improved_rate": _ratio(improved, count),
        "harmed_rate": _ratio(harmed, count),
        "identity_rate": _ratio(identity, count),
        "cross25_up": int(cross25),
        "cross50_up": int(cross50),
        "covered_prediction_rows": len(covered_rows),
        "geometry_prediction_row_coverage": _ratio(
            len(covered_rows), geometry_prediction_rows
        ),
        "all_prediction_coverage": _ratio(
            len(covered_rows), prediction_count
        ),
        "covered_scenes": len(covered_scenes),
        "scene_coverage": _ratio(len(covered_scenes), scene_count),
    }


def _path_is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_output_path(
    output: Path,
    *,
    geometry_root: Path,
    prediction_root: Path,
    scene_list: Path,
    gt_root: Path,
    scan_root: Path,
) -> Path:
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite report: {destination}")
    protected_files = {Path(scene_list).resolve()}
    protected_roots = tuple(
        Path(root).resolve()
        for root in (
            geometry_root,
            prediction_root,
            gt_root,
            scan_root,
        )
    )
    if destination in protected_files or any(
        _path_is_inside(destination, root) for root in protected_roots
    ):
        raise ValueError(
            "report output must remain outside every input file and root"
        )
    return destination


def _write_report_no_replace(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                report,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(
                f"refusing to overwrite report: {path}"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)


def report_yidu_candidate_deltas(
    *,
    geometry_root: Path,
    prediction_root: Path,
    scene_list: Path,
    gt_root: Path,
    scan_root: Path,
    output: Path | None = None,
) -> dict[str, object]:
    """Compute fixed-target candidate deltas and optionally write JSON."""

    geometry_root = Path(geometry_root)
    prediction_root = Path(prediction_root)
    scene_list = Path(scene_list)
    gt_root = Path(gt_root)
    scan_root = Path(scan_root)
    for role, root in (
        ("geometry", geometry_root),
        ("prediction", prediction_root),
        ("ground-truth", gt_root),
        ("scan", scan_root),
    ):
        if not root.is_dir():
            raise FileNotFoundError(f"{role} root not found: {root}")
    scenes = read_scene_ids(scene_list)

    all_records: list[dict[str, Any]] = []
    per_scene_context: dict[str, dict[str, Any]] = {}
    stages: set[str] = set()
    total_predictions = 0
    total_geometry_rows = 0
    total_ground_truth = 0

    for scene_id in scenes:
        geometry_path = geometry_root / f"{scene_id}{OUTPUT_SUFFIX}"
        geometry = load_geometry_candidates(
            geometry_path, expected_scene_id=scene_id
        )
        stage = _load_export_provenance(
            geometry_path,
            candidate_count=len(geometry.candidate_corners),
            candidate_verified=geometry.candidate_verified,
        )
        stages.add(stage)

        prediction_path = prediction_root / f"{scene_id}_boxes.pkl"
        prediction_corners, _ = load_scene_predictions(prediction_path)
        if (
            len(geometry.prediction_indices)
            and int(np.max(geometry.prediction_indices))
            >= len(prediction_corners)
        ):
            raise ValueError(
                f"{scene_id}: geometry prediction index is out of range"
            )
        paired = prediction_corners[geometry.prediction_indices]
        if not np.array_equal(paired, geometry.original_corners):
            raise ValueError(
                f"{scene_id}: geometry original corners disagree exactly "
                "with frozen predictions"
            )

        transform = load_axis_alignment(scan_root, scene_id)
        prediction_minmax = corners_to_minmax(
            transform_corners(prediction_corners, transform)
        )
        candidate_minmax = corners_to_minmax(
            transform_corners(geometry.candidate_corners, transform)
        )
        gt_path = gt_root / f"{scene_id}_bbox.npy"
        if not gt_path.is_file():
            raise FileNotFoundError(gt_path)
        gt_payload = np.load(gt_path, allow_pickle=False)
        gt_minmax = center_size_to_minmax(gt_payload)
        original_matrix = pairwise_aabb_iou(
            prediction_minmax, gt_minmax
        )
        candidate_matrix = pairwise_aabb_iou(
            candidate_minmax, gt_minmax
        )

        scene_records: list[dict[str, Any]] = []
        for geometry_row, prediction_index in enumerate(
            geometry.prediction_indices.tolist()
        ):
            start = int(geometry.candidate_offsets[geometry_row])
            stop = int(geometry.candidate_offsets[geometry_row + 1])
            target_index = (
                None
                if len(gt_minmax) == 0
                else int(np.argmax(original_matrix[prediction_index]))
            )
            original_iou = (
                0.0
                if target_index is None
                else float(
                    original_matrix[prediction_index, target_index]
                )
            )
            for candidate_index in range(start, stop):
                if not bool(geometry.candidate_valid[candidate_index]):
                    continue
                candidate_iou = (
                    0.0
                    if target_index is None
                    else float(
                        candidate_matrix[candidate_index, target_index]
                    )
                )
                record = {
                    "scene_id": scene_id,
                    "prediction_index": int(prediction_index),
                    "candidate_index": int(candidate_index),
                    "source": str(
                        geometry.candidate_sources[candidate_index]
                    ),
                    "verified": bool(
                        geometry.candidate_verified[candidate_index]
                    ),
                    "original_iou": original_iou,
                    "candidate_iou": candidate_iou,
                    "delta": candidate_iou - original_iou,
                    "corner_identity": bool(
                        np.array_equal(
                            geometry.candidate_corners[candidate_index],
                            geometry.original_corners[geometry_row],
                        )
                    ),
                }
                all_records.append(record)
                scene_records.append(record)

        total_predictions += len(prediction_corners)
        total_geometry_rows += len(geometry.prediction_indices)
        total_ground_truth += len(gt_minmax)
        per_scene_context[scene_id] = {
            "stage": stage,
            "records": scene_records,
            "predictions": len(prediction_corners),
            "geometry_rows": len(geometry.prediction_indices),
            "ground_truth": len(gt_minmax),
        }

    if len(stages) != 1:
        raise ValueError(
            "geometry root mixes YiDu stages: " + ", ".join(sorted(stages))
        )
    stage = next(iter(stages))
    verified_records = [
        row for row in all_records if bool(row["verified"])
    ]
    all_summary = _summarize(
        all_records,
        scene_count=len(scenes),
        prediction_count=total_predictions,
        geometry_prediction_rows=total_geometry_rows,
    )
    verified_summary = _summarize(
        verified_records,
        scene_count=len(scenes),
        prediction_count=total_predictions,
        geometry_prediction_rows=total_geometry_rows,
    )

    sources = sorted({str(row["source"]) for row in all_records})
    by_source: dict[str, object] = {}
    for source in sources:
        source_records = [
            row for row in all_records if row["source"] == source
        ]
        by_source[source] = {
            "all_valid": _summarize(
                source_records,
                scene_count=len(scenes),
                prediction_count=total_predictions,
                geometry_prediction_rows=total_geometry_rows,
            ),
            "verified_only": _summarize(
                [
                    row
                    for row in source_records
                    if bool(row["verified"])
                ],
                scene_count=len(scenes),
                prediction_count=total_predictions,
                geometry_prediction_rows=total_geometry_rows,
            ),
        }

    by_scene: dict[str, object] = {}
    for scene_id in scenes:
        context = per_scene_context[scene_id]
        records = context["records"]
        scene_predictions = int(context["predictions"])
        scene_geometry_rows = int(context["geometry_rows"])
        by_scene[scene_id] = {
            "stage": str(context["stage"]),
            "predictions": scene_predictions,
            "geometry_prediction_rows": scene_geometry_rows,
            "ground_truth": int(context["ground_truth"]),
            "all_valid": _summarize(
                records,
                scene_count=1,
                prediction_count=scene_predictions,
                geometry_prediction_rows=scene_geometry_rows,
            ),
            "verified_only": _summarize(
                [row for row in records if bool(row["verified"])],
                scene_count=1,
                prediction_count=scene_predictions,
                geometry_prediction_rows=scene_geometry_rows,
            ),
        }

    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "format_version": REPORT_FORMAT_VERSION,
        "stage": stage,
        "verified_only_semantics": (
            "a6_gate_accepted"
            if stage == "A6"
            else "export_candidate_verified"
        ),
        "target_policy": (
            "candidate IoU uses the ground-truth box selected by the "
            "original frozen prediction; candidates are never re-matched"
        ),
        "offline_ground_truth_diagnostic_only": True,
        "runtime_artifacts_mutated": False,
        "delta_epsilon": DELTA_EPSILON,
        "scene_count": len(scenes),
        "prediction_count": total_predictions,
        "geometry_prediction_rows": total_geometry_rows,
        "ground_truth_count": total_ground_truth,
        "all_valid": all_summary,
        "verified_only": verified_summary,
        "by_source": by_source,
        "by_scene": by_scene,
    }
    if output is not None:
        destination = _validate_output_path(
            Path(output),
            geometry_root=geometry_root,
            prediction_root=prediction_root,
            scene_list=scene_list,
            gt_root=gt_root,
            scan_root=scan_root,
        )
        _write_report_no_replace(destination, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-root", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--scene-list", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--scan-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = report_yidu_candidate_deltas(
        geometry_root=args.geometry_root,
        prediction_root=args.prediction_root,
        scene_list=args.scene_list,
        gt_root=args.gt_root,
        scan_root=args.scan_root,
        output=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
