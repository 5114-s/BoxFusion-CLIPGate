#!/usr/bin/env python3
"""Fail-closed audit for the strict val100 p100 trajectory export.

The val100 export is required to be a content-identical superset of the
previously frozen fixed10 export.  Absolute path namespaces are deliberately
not compared.  The files referenced by those paths are hashed instead, so a
symlinked or relocated dataset is accepted only when its actual point, pose,
and calibration content is identical.

Pickle is executable serialization.  Only pass trusted local BoxFusion info
files to this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.frozen_b6_manifest import read_scene_list  # noqa: E402
from boxfusion.tr3d_residual_cache import (  # noqa: E402
    load_tr3d_residual_cache,
    tr3d_residual_cache_path,
)


REPORT_SCHEMA = "boxfusion.tr3d_prefix_superset_audit.v1"
_CALIBRATION_FILES = (
    "intrinsic_depth.txt",
    "intrinsic_color.txt",
    "extrinsic_depth.txt",
    "extrinsic_color.txt",
)
_CLOCK_AND_CONTENT_FIELDS = (
    "schema",
    "scene_id",
    "tag",
    "fraction",
    "frame_stride",
    "tail_guard_frames",
    "clock_policy",
    "pose_policy",
    "source_timestamp_semantics",
    "source_frame_count",
    "processed_frame_count",
    "pixel_stride",
    "voxel_size",
    "depth_scale",
    "coordinate_frame",
    "network_frame_after_pipeline",
    "sampled_frame_count",
    "first_frame_id",
    "last_frame_id",
    "frame_ids",
    "source_timestamps",
    "last_source_timestamp",
    "used_frame_ids",
    "used_source_timestamps",
    "axis_align_matrix",
    "min_observed_points",
    "min_visibility_fraction",
    "status",
    "point_count",
    "source_instance_count",
    "kept_instance_count",
    "instance_support",
)
_POSE_CONTENT_FIELDS = (
    "source_timestamp",
    "frame_id",
    "input_pose_frame_id",
    "input_pose_sha256",
    "pose_resolution",
    "resolved_pose_source_timestamp",
    "resolved_pose_frame_id",
    "resolved_pose_sha256",
)
_CACHE_ARRAY_FIELDS = (
    "boxes_world",
    "corners_world",
    "aligned_to_unaligned",
    "scores_3d",
    "labels_3d",
    "proposal_ids",
    "point_count",
)
_CACHE_SCALAR_FIELDS = (
    "scene_id",
    "sample_idx",
    "prefix_id",
    "prefix_fraction",
    "axis_alignment_sha256",
    "voxel_size",
    "num_input_points",
    "checkpoint_sha256",
    "config_sha256",
    "source_scene_sha256",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: malformed JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: row is not an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path}: manifest is empty")
    return rows


def _index_manifest(
    path: Path,
    *,
    expected_scenes: Sequence[str],
    expected_count: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = _read_jsonl(path)
    scene_ids = [str(row.get("scene_id", "")) for row in rows]
    if len(rows) != expected_count or len(set(scene_ids)) != expected_count:
        raise ValueError(
            f"{path}: expected {expected_count} unique p100 scenes, got "
            f"rows={len(rows)}, unique={len(set(scene_ids))}"
        )
    if scene_ids != list(expected_scenes):
        raise ValueError(f"{path}: manifest scene order/set disagrees with list")
    for row in rows:
        if row.get("tag") != "p100" or float(row.get("fraction", -1)) != 1.0:
            raise ValueError(
                f"{path}: {row.get('scene_id')}: expected exactly p100"
            )
        if row.get("status") != "exported":
            raise ValueError(
                f"{path}: {row.get('scene_id')}: prefix is not exported"
            )
    return rows, dict(zip(scene_ids, rows))


def _pose_content(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    provenance = row.get("pose_provenance")
    if not isinstance(provenance, list):
        raise ValueError(f"{row.get('scene_id')}: missing pose_provenance")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(provenance):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"{row.get('scene_id')}: pose row {index} is malformed"
            )
        normalized = {name: item.get(name) for name in _POSE_CONTENT_FIELDS}
        for path_key, sha_key in (
            ("input_pose_path", "input_pose_sha256"),
            ("resolved_pose_path", "resolved_pose_sha256"),
        ):
            path_value = item.get(path_key)
            if not isinstance(path_value, str) or not path_value:
                raise ValueError(
                    f"{row.get('scene_id')}: pose row {index} lacks {path_key}"
                )
            path = Path(path_value)
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = _sha256_file(path)
            if actual != item.get(sha_key):
                raise ValueError(
                    f"{row.get('scene_id')}: {path_key} content hash mismatch"
                )
        result.append(normalized)
    return result


def _calibration_content(row: Mapping[str, Any]) -> dict[str, str]:
    root_value = row.get("source_scene_frame_root")
    if not isinstance(root_value, str) or not root_value:
        raise ValueError(f"{row.get('scene_id')}: missing source frame root")
    root = Path(root_value)
    intrinsic = root / "intrinsic"
    result: dict[str, str] = {}
    for name in _CALIBRATION_FILES:
        path = intrinsic / name
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = _sha256_file(path)
    return result


def _point_content(row: Mapping[str, Any]) -> tuple[Path, str]:
    value = row.get("point_path")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{row.get('scene_id')}: missing point_path")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size % (6 * np.dtype(np.float32).itemsize) != 0:
        raise ValueError(f"{path}: malformed xyzrgb point file")
    count = path.stat().st_size // (6 * np.dtype(np.float32).itemsize)
    if count != int(row.get("point_count", -1)):
        raise ValueError(f"{path}: point_count disagrees with byte length")
    return path.resolve(), _sha256_file(path)


def _manifest_semantics(row: Mapping[str, Any]) -> dict[str, Any]:
    missing = [name for name in _CLOCK_AND_CONTENT_FIELDS if name not in row]
    if missing:
        raise ValueError(f"{row.get('scene_id')}: missing fields {missing}")
    return {name: _jsonable(row[name]) for name in _CLOCK_AND_CONTENT_FIELDS}


def _load_info(path: Path) -> tuple[Any, list[Mapping[str, Any]]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - trusted local artifact
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("data_list"), list
    ):
        raise ValueError(f"{path}: malformed MMDetection3D info")
    return payload.get("metainfo"), payload["data_list"]


def _index_info(
    path: Path,
    *,
    expected_scenes: Sequence[str],
) -> tuple[Any, dict[str, Mapping[str, Any]]]:
    metainfo, rows = _load_info(path)
    scene_ids: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not isinstance(
            row.get("trajectory_prefix"), Mapping
        ):
            raise ValueError(f"{path}: malformed info row {index}")
        scene_ids.append(str(row["trajectory_prefix"].get("scene_id", "")))
    if scene_ids != list(expected_scenes) or len(set(scene_ids)) != len(rows):
        raise ValueError(f"{path}: info scene order/set disagrees with list")
    return metainfo, dict(zip(scene_ids, rows))


def _normalized_info_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result = _jsonable(row)
    trajectory = dict(result["trajectory_prefix"])
    for key in (
        "source_frames_root",
        "source_scene_frame_root",
        "point_path",
    ):
        trajectory.pop(key, None)
    for item in trajectory.get("pose_provenance", []):
        item.pop("input_pose_path", None)
        item.pop("resolved_pose_path", None)
    result["trajectory_prefix"] = trajectory
    return result


def _validate_point_artifact_set(
    manifest_path: Path,
    expected: set[Path],
) -> None:
    prepared_root = manifest_path.parent.parent
    actual = {
        path.resolve() for path in (prepared_root / "points").rglob("*.bin")
    }
    if actual != expected:
        raise ValueError(
            f"{prepared_root}: point artifact set mismatch; "
            f"missing={sorted(map(str, expected-actual))[:4]}, "
            f"extra={sorted(map(str, actual-expected))[:4]}"
        )


def audit_prefix_superset(
    *,
    full_manifest: Path,
    fixed_manifest: Path,
    full_info: Path,
    fixed_info: Path,
    full_scene_list: Path,
    fixed_scene_list: Path,
    expected_full_scene_count: int = 100,
) -> dict[str, Any]:
    full_scenes = read_scene_list(full_scene_list)
    fixed_scenes = read_scene_list(fixed_scene_list)
    if len(full_scenes) != expected_full_scene_count:
        raise ValueError(
            f"full scene list must contain {expected_full_scene_count} scenes"
        )
    if len(fixed_scenes) != len(set(fixed_scenes)):
        raise ValueError("fixed scene list contains duplicates")
    if not set(fixed_scenes).issubset(full_scenes):
        raise ValueError("fixed scene list is not a full-scene subset")
    full_rows, full_index = _index_manifest(
        full_manifest,
        expected_scenes=full_scenes,
        expected_count=expected_full_scene_count,
    )
    fixed_rows, fixed_index = _index_manifest(
        fixed_manifest,
        expected_scenes=fixed_scenes,
        expected_count=len(fixed_scenes),
    )
    full_meta, full_info_index = _index_info(
        full_info, expected_scenes=full_scenes
    )
    fixed_meta, fixed_info_index = _index_info(
        fixed_info, expected_scenes=fixed_scenes
    )
    if _canonical_sha256(full_meta) != _canonical_sha256(fixed_meta):
        raise ValueError("full/fixed info metainfo differs")

    full_points: set[Path] = set()
    for row in full_rows:
        point, _ = _point_content(row)
        full_points.add(point)
        _pose_content(row)
        _calibration_content(row)
    fixed_points: set[Path] = set()
    for row in fixed_rows:
        point, _ = _point_content(row)
        fixed_points.add(point)
        _pose_content(row)
        _calibration_content(row)
    _validate_point_artifact_set(full_manifest, full_points)
    _validate_point_artifact_set(fixed_manifest, fixed_points)

    scene_reports: dict[str, Any] = {}
    for scene_id in fixed_scenes:
        full_row = full_index[scene_id]
        fixed_row = fixed_index[scene_id]
        if _manifest_semantics(full_row) != _manifest_semantics(fixed_row):
            raise ValueError(f"{scene_id}: clock/content manifest fields differ")
        full_pose = _pose_content(full_row)
        fixed_pose = _pose_content(fixed_row)
        if full_pose != fixed_pose:
            raise ValueError(f"{scene_id}: pose provenance/content differs")
        full_calibration = _calibration_content(full_row)
        fixed_calibration = _calibration_content(fixed_row)
        if full_calibration != fixed_calibration:
            raise ValueError(f"{scene_id}: calibration content differs")
        _, full_point_sha = _point_content(full_row)
        _, fixed_point_sha = _point_content(fixed_row)
        if full_point_sha != fixed_point_sha:
            raise ValueError(f"{scene_id}: point content differs")
        full_info_row = _normalized_info_row(full_info_index[scene_id])
        fixed_info_row = _normalized_info_row(fixed_info_index[scene_id])
        if full_info_row != fixed_info_row:
            raise ValueError(f"{scene_id}: annotation/info content differs")
        scene_reports[scene_id] = {
            "point_sha256": full_point_sha,
            "pose_content_sha256": _canonical_sha256(full_pose),
            "calibration_sha256": _canonical_sha256(full_calibration),
            "manifest_semantics_sha256": _canonical_sha256(
                _manifest_semantics(full_row)
            ),
            "info_semantics_sha256": _canonical_sha256(full_info_row),
        }

    return {
        "schema": REPORT_SCHEMA,
        "ok": True,
        "full_manifest": str(full_manifest.resolve()),
        "fixed_manifest": str(fixed_manifest.resolve()),
        "full_info": str(full_info.resolve()),
        "fixed_info": str(fixed_info.resolve()),
        "full_scene_count": len(full_rows),
        "fixed_scene_count": len(fixed_rows),
        "fixed_scene_ids": list(fixed_scenes),
        "content_identical_fixed_subset": True,
        "scenes": scene_reports,
    }


def audit_parent_cache_subset(
    *,
    full_cache_root: Path,
    fixed_cache_root: Path,
    fixed_scene_list: Path,
    prefix_id: str,
    checkpoint_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    scenes = read_scene_list(fixed_scene_list)
    for scene_id in scenes:
        expected = {
            "expected_scene_id": scene_id,
            "expected_prefix_id": prefix_id,
            "expected_checkpoint_sha256": checkpoint_sha256,
            "expected_config_sha256": config_sha256,
        }
        full = load_tr3d_residual_cache(
            tr3d_residual_cache_path(full_cache_root, scene_id, prefix_id),
            **expected,
        )
        fixed = load_tr3d_residual_cache(
            tr3d_residual_cache_path(fixed_cache_root, scene_id, prefix_id),
            **expected,
        )
        for name in _CACHE_SCALAR_FIELDS:
            if getattr(full, name) != getattr(fixed, name):
                raise ValueError(f"{scene_id}: cache scalar {name} differs")
        for name in _CACHE_ARRAY_FIELDS:
            if not np.array_equal(getattr(full, name), getattr(fixed, name)):
                raise ValueError(f"{scene_id}: cache array {name} differs")
    return {
        "ok": True,
        "full_cache_root": str(full_cache_root.resolve()),
        "fixed_cache_root": str(fixed_cache_root.resolve()),
        "fixed_scene_count": len(scenes),
        "prefix_id": prefix_id,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "arrays_exact": True,
        "runtime_s_ignored": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--fixed-manifest", type=Path, required=True)
    parser.add_argument("--full-info", type=Path, required=True)
    parser.add_argument("--fixed-info", type=Path, required=True)
    parser.add_argument("--full-scene-list", type=Path, required=True)
    parser.add_argument("--fixed-scene-list", type=Path, required=True)
    parser.add_argument("--expected-full-scene-count", type=int, default=100)
    parser.add_argument("--full-cache-root", type=Path)
    parser.add_argument("--fixed-cache-root", type=Path)
    parser.add_argument("--prefix-id", default="p100")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--config-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in (
        args.full_manifest,
        args.fixed_manifest,
        args.full_info,
        args.fixed_info,
        args.full_scene_list,
        args.fixed_scene_list,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = audit_prefix_superset(
        full_manifest=args.full_manifest,
        fixed_manifest=args.fixed_manifest,
        full_info=args.full_info,
        fixed_info=args.fixed_info,
        full_scene_list=args.full_scene_list,
        fixed_scene_list=args.fixed_scene_list,
        expected_full_scene_count=args.expected_full_scene_count,
    )
    cache_values = (
        args.full_cache_root,
        args.fixed_cache_root,
        args.checkpoint_sha256,
        args.config_sha256,
    )
    if any(value is not None for value in cache_values):
        if not all(value is not None for value in cache_values):
            raise ValueError(
                "cache audit requires both roots and checkpoint/config hashes"
            )
        report["parent_cache_subset"] = audit_parent_cache_subset(
            full_cache_root=args.full_cache_root,
            fixed_cache_root=args.fixed_cache_root,
            fixed_scene_list=args.fixed_scene_list,
            prefix_id=args.prefix_id,
            checkpoint_sha256=args.checkpoint_sha256,
            config_sha256=args.config_sha256,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
