#!/usr/bin/env python3
"""Post-hoc dev3 recall ceiling for the sealed S3 confirmed-track universe.

The candidate pool is immutable before this tool is allowed to read ScanNet
ground truth.  The report measures maximum-cardinality candidate/native-union
matching for several fixed point-quantile geometries, the bounded per-view box
records, and their post-hoc geometry oracle.  These are diagnostic ceilings,
not deployable selection or geometry policies, and they never authorize H10 or
full100 access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

from tools.audit_scannet_boxer_unexplained_oracle import (  # noqa: E402
    aligned_iou_matrix,
    load_axis_alignment,
    load_baseline_boxes,
    load_gt_minmax,
    official_constant_evaluate,
    strict_maximum_matching,
)
from tools.export_s3_yoloe_confirmed_universe import (  # noqa: E402
    DEV3_SCENES,
    EXPECTED_CONFIRMED_COUNTS,
    SCENE_SCHEMA,
    SEAL_SCHEMA,
    _array_content_sha256,
)


SCHEMA = "boxfusion.scannet_s3_yoloe_confirmed_universe_ceiling.v1"
THRESHOLDS = (0.15, 0.25, 0.50)
POINT_QUANTILES: Mapping[str, float | None] = {
    "producer_q02": None,
    "points_q00": 0.00,
    "points_q01": 0.01,
    "points_q02": 0.02,
    "points_q05": 0.05,
    "points_q10": 0.10,
}
PLUS_TEN_AP_POINTS = 10.0

_EXPECTED_ARRAYS = {
    "archived",
    "box_center_extent",
    "box_record_center_extent",
    "box_record_keyframe_index",
    "box_record_offsets",
    "box_record_score",
    "box_record_source_frame_id",
    "created_keyframe_index",
    "created_lifecycle_step",
    "created_source_frame_id",
    "hit_count",
    "last_keyframe_index",
    "last_lifecycle_step",
    "last_source_frame_id",
    "mean_score",
    "memory_observation_count",
    "memory_unique_view_count",
    "point_offsets",
    "points_world",
    "processed_source_frame_ids",
    "scene_id",
    "score_offsets",
    "source_scores",
    "terminal_output",
    "terminal_result_index",
    "track_id",
    "view_count",
}


class S3CeilingError(ValueError):
    """Raised when a sealed candidate or dev3 audit input is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise S3CeilingError(f"{label} must be a regular non-symlink file: {path}")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    path = _regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise S3CeilingError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise S3CeilingError(f"{label} must contain an object")
    return value


def _threshold_key(value: float) -> str:
    return f"{value:.2f}"


def _load_seal(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Path]]]:
    manifest = _read_json(path, "sealed S3 candidate universe")
    required = {
        "schema": SEAL_SCHEMA,
        "mode": "sealed_no_gt_candidate_universe",
        "gt_access": False,
        "oracle_access": False,
        "H10_accessed": False,
        "full100_accessed": False,
        "active_birth_authorized": False,
        "scene_order": list(DEV3_SCENES),
        "scene_count": len(DEV3_SCENES),
        "confirmed_track_count_by_scene": dict(EXPECTED_CONFIRMED_COUNTS),
        "confirmed_track_count": sum(EXPECTED_CONFIRMED_COUNTS.values()),
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise S3CeilingError(f"sealed S3 contract mismatch for {key}")
    scenes = manifest.get("scenes")
    if not isinstance(scenes, dict) or set(scenes) != set(DEV3_SCENES):
        raise S3CeilingError("sealed S3 scene ledger changed")
    records: dict[str, dict[str, Path]] = {}
    for scene in DEV3_SCENES:
        row = scenes[scene]
        if not isinstance(row, dict):
            raise S3CeilingError(f"invalid S3 scene ledger for {scene}")
        json_path = _regular_file(Path(str(row.get("manifest"))), f"S3 manifest for {scene}")
        npz_path = _regular_file(Path(str(row.get("npz"))), f"S3 NPZ for {scene}")
        if _sha256(json_path) != row.get("manifest_sha256") or _sha256(npz_path) != row.get(
            "npz_sha256"
        ):
            raise S3CeilingError(f"S3 scene artifact hash mismatch for {scene}")
        records[scene] = {"json": json_path, "npz": npz_path}
    return manifest, records


def _validate_offsets(offsets: np.ndarray, rows: int, values: int, label: str) -> None:
    if (
        offsets.dtype.kind not in "iu"
        or offsets.shape != (rows + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != values
        or np.any(np.diff(offsets) < 0)
    ):
        raise S3CeilingError(f"invalid {label} offsets")


def _load_scene(
    scene: str, paths: Mapping[str, Path], seal_row: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = _read_json(paths["json"], f"S3 scene manifest for {scene}")
    required = {
        "schema": SCENE_SCHEMA,
        "mode": "shadow",
        "output_inert": True,
        "birth": False,
        "active_authorized": False,
        "gt_access": False,
        "oracle_access": False,
        "labels_read": False,
        "labels_exported": False,
        "scene_id": scene,
        "confirmed_track_count": EXPECTED_CONFIRMED_COUNTS[scene],
        "expected_confirmed_track_count": EXPECTED_CONFIRMED_COUNTS[scene],
        "processed_source_frames_exactly_match_stream_seal": True,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise S3CeilingError(f"S3 scene contract mismatch for {scene}.{key}")
    if manifest.get("npz_file") != paths["npz"].name or manifest.get(
        "npz_sha256"
    ) != _sha256(paths["npz"]):
        raise S3CeilingError(f"S3 scene NPZ seal mismatch for {scene}")
    try:
        with np.load(paths["npz"], allow_pickle=False) as source:
            if set(source.files) != _EXPECTED_ARRAYS:
                raise S3CeilingError(f"unexpected S3 NPZ schema for {scene}")
            arrays = {name: np.array(source[name], copy=True) for name in source.files}
    except (OSError, ValueError) as error:
        if isinstance(error, S3CeilingError):
            raise
        raise S3CeilingError(f"invalid S3 NPZ for {scene}") from error
    if manifest.get("array_content_sha256") != _array_content_sha256(arrays):
        raise S3CeilingError(f"S3 array content hash mismatch for {scene}")
    if seal_row.get("array_content_sha256") != manifest.get("array_content_sha256"):
        raise S3CeilingError(f"aggregate/scene array hash mismatch for {scene}")
    count = EXPECTED_CONFIRMED_COUNTS[scene]
    # The deterministic NPZ writer canonicalizes every payload with
    # np.ascontiguousarray.  NumPy represents a contiguous scalar as a
    # one-element vector on disk, so require that exact sealed form here.
    if arrays["scene_id"].shape != (1,) or str(arrays["scene_id"][0]) != scene:
        raise S3CeilingError(f"S3 scene scalar mismatch for {scene}")
    row_arrays = {
        "track_id",
        "archived",
        "created_keyframe_index",
        "last_keyframe_index",
        "created_source_frame_id",
        "last_source_frame_id",
        "created_lifecycle_step",
        "last_lifecycle_step",
        "hit_count",
        "view_count",
        "memory_observation_count",
        "memory_unique_view_count",
        "mean_score",
        "terminal_output",
        "terminal_result_index",
    }
    for name in row_arrays:
        if arrays[name].shape != (count,):
            raise S3CeilingError(f"S3 {name} shape mismatch for {scene}")
    if arrays["box_center_extent"].shape != (count, 6):
        raise S3CeilingError(f"S3 box geometry shape mismatch for {scene}")
    if arrays["points_world"].ndim != 2 or arrays["points_world"].shape[1:] != (3,):
        raise S3CeilingError(f"S3 points shape mismatch for {scene}")
    if arrays["box_record_center_extent"].ndim != 2 or arrays[
        "box_record_center_extent"
    ].shape[1:] != (6,):
        raise S3CeilingError(f"S3 box-record shape mismatch for {scene}")
    _validate_offsets(arrays["point_offsets"], count, len(arrays["points_world"]), "point")
    _validate_offsets(arrays["score_offsets"], count, len(arrays["source_scores"]), "score")
    _validate_offsets(
        arrays["box_record_offsets"], count, len(arrays["box_record_center_extent"]), "box-record"
    )
    if not np.array_equal(arrays["track_id"], np.sort(arrays["track_id"])) or len(
        np.unique(arrays["track_id"])
    ) != count:
        raise S3CeilingError(f"S3 track IDs are not unique/sorted for {scene}")
    if np.any(arrays["view_count"] < 3) or np.any(arrays["hit_count"] < arrays["view_count"]):
        raise S3CeilingError(f"unconfirmed track leaked into S3 for {scene}")
    numeric = np.concatenate(
        (
            arrays["box_center_extent"].reshape(-1),
            arrays["points_world"].reshape(-1),
            arrays["source_scores"].reshape(-1),
            arrays["box_record_center_extent"].reshape(-1),
        )
    )
    if not np.isfinite(numeric).all() or np.any(arrays["box_center_extent"][:, 3:] <= 0.0):
        raise S3CeilingError(f"S3 geometry/provenance is non-finite for {scene}")
    if int(arrays["terminal_output"].sum()) != manifest.get("terminal_output_count"):
        raise S3CeilingError(f"S3 terminal membership count mismatch for {scene}")
    return manifest, arrays


def _center_extent_to_minmax(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64)
    return np.concatenate((boxes[:, :3] - boxes[:, 3:] / 2, boxes[:, :3] + boxes[:, 3:] / 2), axis=1)


def _align_minmax(boxes: np.ndarray, alignment: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64)
    signs = np.asarray([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)])
    corners = np.where(signs[None] == 0, boxes[:, None, :3], boxes[:, None, 3:])
    aligned = corners @ alignment[:3, :3].T + alignment[:3, 3]
    return np.concatenate((aligned.min(axis=1), aligned.max(axis=1)), axis=1)


def _point_quantile_minmax(
    points: np.ndarray, offsets: np.ndarray, quantile: float
) -> np.ndarray:
    rows = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        values = np.asarray(points[int(start) : int(end)], dtype=np.float64)
        if len(values) == 0:
            raise S3CeilingError("S3 track has no points")
        rows.append(
            np.concatenate(
                (np.quantile(values, quantile, axis=0), np.quantile(values, 1 - quantile, axis=0))
            )
        )
    return np.asarray(rows, dtype=np.float64).reshape((-1, 6))


def _box_record_oracle_iou(
    arrays: Mapping[str, np.ndarray], alignment: np.ndarray, gt: np.ndarray
) -> np.ndarray:
    records = _center_extent_to_minmax(arrays["box_record_center_extent"])
    aligned = _align_minmax(records, alignment) if len(records) else records
    record_iou = aligned_iou_matrix(aligned, gt)
    offsets = arrays["box_record_offsets"]
    rows = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        start, end = int(start), int(end)
        rows.append(
            record_iou[start:end].max(axis=0)
            if end > start
            else np.zeros(len(gt), dtype=np.float64)
        )
    return np.asarray(rows, dtype=np.float64).reshape((len(offsets) - 1, len(gt)))


def _geometry_report(
    *,
    scenes: Sequence[str],
    candidate_iou: Sequence[np.ndarray],
    native_iou: Sequence[np.ndarray],
    gt_counts: Sequence[int],
    baseline_official: Mapping[float, Mapping[str, Any]],
) -> dict[str, Any]:
    total_gt = int(sum(gt_counts))
    thresholds: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        native_matches = 0
        candidate_matches = 0
        union_matches = 0
        official_unmatched_recoveries = 0
        per_scene: dict[str, Any] = {}
        official_masks = baseline_official[threshold]["matched_gt_masks"]
        for scene, candidates, native, matched_mask in zip(
            scenes, candidate_iou, native_iou, official_masks
        ):
            native_pairs = strict_maximum_matching(native, threshold)
            candidate_pairs = strict_maximum_matching(candidates, threshold)
            union_pairs = strict_maximum_matching(
                np.concatenate((native, candidates), axis=0), threshold
            )
            recovery_pairs = strict_maximum_matching(
                candidates, threshold, ~np.asarray(matched_mask, dtype=bool)
            )
            native_matches += len(native_pairs)
            candidate_matches += len(candidate_pairs)
            union_matches += len(union_pairs)
            official_unmatched_recoveries += len(recovery_pairs)
            per_scene[scene] = {
                "native_maximum_matching_count": len(native_pairs),
                "candidate_maximum_matching_count": len(candidate_pairs),
                "native_union_maximum_matching_count": len(union_pairs),
                "additional_union_matching_over_native": len(union_pairs) - len(native_pairs),
                "official_baseline_unmatched_candidate_recovery_count": len(recovery_pairs),
            }
        baseline_ap_points = float(baseline_official[threshold]["ap_points"])
        native_recall_points = 100.0 * native_matches / total_gt
        union_recall_points = 100.0 * union_matches / total_gt
        additional_recall_points = union_recall_points - native_recall_points
        ap_ceiling_headroom = union_recall_points - baseline_ap_points
        thresholds[_threshold_key(threshold)] = {
            "iou_threshold": threshold,
            "gt_count": total_gt,
            "native_maximum_matching_count": native_matches,
            "candidate_maximum_matching_count": candidate_matches,
            "native_union_maximum_matching_count": union_matches,
            "additional_union_matching_over_native": union_matches - native_matches,
            "official_baseline_unmatched_candidate_recovery_count": official_unmatched_recoveries,
            "native_maximum_matching_recall_points": native_recall_points,
            "native_union_maximum_matching_recall_points": union_recall_points,
            "additional_union_recall_points": additional_recall_points,
            "frozen_t05_official_ap_points": baseline_ap_points,
            "optimistic_ap_ceiling_headroom_points_over_frozen_t05": ap_ceiling_headroom,
            "additional_union_recall_reaches_10_points": additional_recall_points
            >= PLUS_TEN_AP_POINTS,
            "necessary_recall_ceiling_can_support_plus_10_ap": ap_ceiling_headroom
            >= PLUS_TEN_AP_POINTS,
            "per_scene": per_scene,
        }
    return {"per_threshold": thresholds}


def audit_s3_confirmed_universe(
    *,
    sealed_universe: Path,
    baseline_root: Path,
    gt_root: Path,
    scan_root: Path,
    preregistration: Path,
) -> dict[str, Any]:
    """Read the already-sealed dev3 universe and compute post-hoc ceilings."""

    sealed_universe = sealed_universe.resolve()
    baseline_root = baseline_root.resolve()
    gt_root = gt_root.resolve()
    scan_root = scan_root.resolve()
    preregistration = _regular_file(preregistration, "S3 preregistration")
    seal, paths = _load_seal(sealed_universe)
    preregistration_seal = seal.get("preregistration")
    if (
        not isinstance(preregistration_seal, dict)
        or Path(str(preregistration_seal.get("path"))).resolve() != preregistration
        or preregistration_seal.get("sha256") != _sha256(preregistration)
    ):
        raise S3CeilingError("S3 preregistration differs from the no-GT candidate seal")
    seal_rows = seal["scenes"]
    before: dict[str, Any] = {
        "sealed_universe": _sha256(sealed_universe),
        "preregistration": _sha256(preregistration),
        "scenes": {},
    }
    arrays_by_scene: dict[str, dict[str, np.ndarray]] = {}
    native_iou: list[np.ndarray] = []
    gt_counts: list[int] = []
    geometry_iou: dict[str, list[np.ndarray]] = {
        name: [] for name in POINT_QUANTILES
    }
    geometry_iou["per_view_box_oracle"] = []
    for scene in DEV3_SCENES:
        manifest, arrays = _load_scene(scene, paths[scene], seal_rows[scene])
        arrays_by_scene[scene] = arrays
        baseline_path = _regular_file(
            baseline_root / f"{scene}_boxes.pkl", f"frozen T05 prediction for {scene}"
        )
        gt_path = _regular_file(gt_root / f"{scene}_bbox.npy", f"dev3 GT for {scene}")
        alignment_path = _regular_file(
            scan_root / scene / f"{scene}.txt", f"dev3 axis alignment for {scene}"
        )
        alignment = load_axis_alignment(alignment_path)
        gt = load_gt_minmax(gt_path)
        _, native_aligned = load_baseline_boxes(baseline_path, alignment)
        native_iou.append(aligned_iou_matrix(native_aligned, gt))
        gt_counts.append(len(gt))
        for name, quantile in POINT_QUANTILES.items():
            world = (
                _center_extent_to_minmax(arrays["box_center_extent"])
                if quantile is None
                else _point_quantile_minmax(
                    arrays["points_world"], arrays["point_offsets"], quantile
                )
            )
            geometry_iou[name].append(
                aligned_iou_matrix(_align_minmax(world, alignment), gt)
            )
        geometry_iou["per_view_box_oracle"].append(
            _box_record_oracle_iou(arrays, alignment, gt)
        )
        before["scenes"][scene] = {
            "s3_manifest": _sha256(paths[scene]["json"]),
            "s3_npz": _sha256(paths[scene]["npz"]),
            "baseline": _sha256(baseline_path),
            "gt": _sha256(gt_path),
            "axis_alignment": _sha256(alignment_path),
        }
    geometry_iou["geometry_oracle"] = [
        np.maximum.reduce([geometry_iou[name][scene_index] for name in geometry_iou])
        for scene_index in range(len(DEV3_SCENES))
    ]
    baseline_official = {
        threshold: official_constant_evaluate(native_iou, gt_counts, threshold)
        for threshold in THRESHOLDS
    }
    geometries = {
        name: {
            "posthoc_geometry_oracle": name in {"per_view_box_oracle", "geometry_oracle"},
            "candidate_membership_unchanged": True,
            **_geometry_report(
                scenes=DEV3_SCENES,
                candidate_iou=matrices,
                native_iou=native_iou,
                gt_counts=gt_counts,
                baseline_official=baseline_official,
            ),
        }
        for name, matrices in geometry_iou.items()
    }
    after = {
        "sealed_universe": _sha256(sealed_universe),
        "preregistration": _sha256(preregistration),
        "scenes": {
            scene: {
                "s3_manifest": _sha256(paths[scene]["json"]),
                "s3_npz": _sha256(paths[scene]["npz"]),
                "baseline": _sha256(baseline_root / f"{scene}_boxes.pkl"),
                "gt": _sha256(gt_root / f"{scene}_bbox.npy"),
                "axis_alignment": _sha256(scan_root / scene / f"{scene}.txt"),
            }
            for scene in DEV3_SCENES
        },
    }
    if after != before:
        raise S3CeilingError("an S3 ceiling input changed during audit")
    fixed = geometries["producer_q02"]["per_threshold"]
    oracle = geometries["geometry_oracle"]["per_threshold"]
    return {
        "schema": SCHEMA,
        "posthoc_dev3_diagnostic": True,
        "not_deployable": True,
        "gt_access_after_candidate_seal_only": True,
        "H10_accessed": False,
        "full100_accessed": False,
        "H10_authorized": False,
        "active_birth_authorized": False,
        "candidate_membership_selected_with_gt": False,
        "candidate_ranking_selected_with_gt": False,
        "labels_read": False,
        "labels_used": False,
        "class_mode": "class_agnostic",
        "strict_iou_comparison": ">",
        "scene_order": list(DEV3_SCENES),
        "confirmed_track_count_by_scene": dict(EXPECTED_CONFIRMED_COUNTS),
        "confirmed_track_count": sum(EXPECTED_CONFIRMED_COUNTS.values()),
        "gt_count": int(sum(gt_counts)),
        "thresholds": list(THRESHOLDS),
        "geometries": geometries,
        "plus_10_summary": {
            "target_ap_points": PLUS_TEN_AP_POINTS,
            "producer_q02_additional_union_recall_reaches_10_all_thresholds": all(
                row["additional_union_recall_reaches_10_points"] for row in fixed.values()
            ),
            "producer_q02_necessary_recall_ceiling_supports_plus_10_ap_all_thresholds": all(
                row["necessary_recall_ceiling_can_support_plus_10_ap"] for row in fixed.values()
            ),
            "geometry_oracle_additional_union_recall_reaches_10_all_thresholds": all(
                row["additional_union_recall_reaches_10_points"] for row in oracle.values()
            ),
            "geometry_oracle_necessary_recall_ceiling_supports_plus_10_ap_all_thresholds": all(
                row["necessary_recall_ceiling_can_support_plus_10_ap"] for row in oracle.values()
            ),
            "interpretation": (
                "Union maximum-matching recall is an optimistic necessary AP ceiling, "
                "not an achievable training-free gate or measured AP gain."
            ),
        },
        "sealed_universe": {
            "path": os.fspath(sealed_universe),
            "sha256": _sha256(sealed_universe),
            "schema": seal["schema"],
        },
        "preregistration": {
            "path": os.fspath(preregistration),
            "sha256": _sha256(preregistration),
        },
        "input_sha256_before": before,
        "input_sha256_after": after,
        "input_hash_identity": before == after,
    }


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise S3CeilingError(f"refusing to overwrite S3 ceiling report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Post-hoc dev3 S3 confirmed-universe ceiling")
    parser.add_argument("--sealed-universe", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--scan-root", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = audit_s3_confirmed_universe(
        sealed_universe=args.sealed_universe,
        baseline_root=args.baseline_root,
        gt_root=args.gt_root,
        scan_root=args.scan_root,
        preregistration=args.preregistration,
    )
    _write_json_create_only(args.out, report)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "confirmed_track_count": report["confirmed_track_count"],
                "gt_count": report["gt_count"],
                "H10_accessed": False,
                "out": os.fspath(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
