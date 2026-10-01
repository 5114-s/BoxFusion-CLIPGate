#!/usr/bin/env python3
"""Recompute and audit GT-free CA-1M terminal TR3D observer artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_native_b6_score import (  # noqa: E402
    load_ca1m_native_b6_scorer,
    load_native_observer_diagnostic,
)
from boxfusion.ca1m_tr3d_terminal import (  # noqa: E402
    BOX_MODE,
    COORDINATE_FRAME,
    CORNER_SEMANTICS,
    SCHEMA,
    aligned_boxes_to_world_corners,
    associate_terminal_candidates,
    sha256_array,
    sha256_file,
    terminal_world_to_local,
    world_aabb,
)
from tools.run_ca1m_tr3d_terminal_observer import (  # noqa: E402
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CONFIG_SHA256,
    _prediction,
    _read_scenes,
    _scene_inputs,
    _terminal_points,
    _used_frame_ids,
)


EXPECTED_KEYS = {
    "active_anchor_scores_sha256",
    "adapter_mode",
    "anchor_corners",
    "anchor_scores",
    "applied_count",
    "best_anchor_center_distance_m",
    "best_anchor_indices",
    "best_anchor_iou",
    "box_mode",
    "candidate_boxes_local",
    "candidate_corners",
    "candidate_labels",
    "candidate_point_count",
    "candidate_scores",
    "checkpoint_sha256",
    "code_manifest_json",
    "code_manifest_sha256",
    "complete",
    "config_sha256",
    "coordinate_frame",
    "corner_semantics",
    "device",
    "ground_truth_access",
    "legacy_rule_selected_anchor_indices",
    "legacy_rule_selected_candidate_rows",
    "materialized_active_verified",
    "max_depth_m",
    "max_proposals",
    "min_depth_m",
    "model_runtime_s",
    "mutation_enabled",
    "native_b6_checkpoint_sha256",
    "native_b6_diagnostic_sha256",
    "native_b6_manifest_sha256",
    "near_iou",
    "near_mask",
    "observer_only",
    "pixel_stride",
    "point_count",
    "prefix_id",
    "represented_anchor_indices",
    "scene_id",
    "schema",
    "score_threshold",
    "source_anchor_prediction_sha256",
    "source_points_sha256",
    "summary_json",
    "used_frame_ids",
    "voxel_size_m",
    "world_to_local",
}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--scene-list", type=Path, required=True)
    value.add_argument("--scene", action="append", default=[])
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--anchor-root", type=Path, required=True)
    value.add_argument("--materialized-active-root", type=Path)
    value.add_argument("--native-b6-diagnostics-root", type=Path, required=True)
    value.add_argument("--native-b6-checkpoint", type=Path, required=True)
    value.add_argument("--native-b6-manifest", type=Path, required=True)
    value.add_argument("--observer-root", type=Path, required=True)
    value.add_argument("--worker-script", type=Path, required=True)
    value.add_argument("--runtime-root", type=Path, required=True)
    value.add_argument("--tr3d-config", type=Path, required=True)
    value.add_argument("--tr3d-checkpoint", type=Path, required=True)
    value.add_argument("--require-genuine", action="store_true")
    value.add_argument("--output", type=Path)
    return value


def _scalar(archive: Any, key: str, dtype: np.dtype | None = None) -> Any:
    value = np.asarray(archive[key])
    if value.shape != ():
        raise ValueError(f"{key} must be a scalar")
    if dtype is not None and value.dtype != np.dtype(dtype):
        raise ValueError(f"{key} has wrong dtype: {value.dtype}")
    return value.item()


def _equal(actual: np.ndarray, expected: np.ndarray, name: str) -> None:
    if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
        raise ValueError(f"{name} differs from independent recomputation")


def _code_sources(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "runner": ROOT / "tools/run_ca1m_tr3d_terminal_observer.py",
        "worker": args.worker_script,
        "terminal_core": ROOT / "boxfusion/ca1m_tr3d_terminal.py",
        "worker_client": ROOT / "boxfusion/ca1m_tr3d_worker_client.py",
        "native_b6_score": ROOT / "boxfusion/ca1m_native_b6_score.py",
        "rgbd_backprojection": ROOT / "boxfusion/tr3d_incremental_online.py",
        "official_adapter": args.runtime_root / "boxfusion/tr3d_inference.py",
    }


def _write_json_create_only(path: Path, payload: dict[str, Any]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{target.name}.", dir=target.parent, delete=False
        ) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        target.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite audit report: {target}") from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def run(args: argparse.Namespace) -> dict[str, Any]:
    scenes = _read_scenes(args.scene_list, list(args.scene))
    scorer = load_ca1m_native_b6_scorer(
        args.native_b6_checkpoint,
        args.native_b6_manifest,
        require_activation_authorized=True,
    )
    checkpoint_sha = sha256_file(args.tr3d_checkpoint)
    config_sha = sha256_file(args.tr3d_config)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("TR3D checkpoint SHA256 mismatch")
    if config_sha != EXPECTED_CONFIG_SHA256:
        raise ValueError("TR3D config SHA256 mismatch")
    b6_checkpoint_sha = sha256_file(args.native_b6_checkpoint)
    b6_manifest_sha = sha256_file(args.native_b6_manifest)
    source_hashes = {
        name: sha256_file(path) for name, path in sorted(_code_sources(args).items())
    }
    expected_code_manifest = json.dumps(
        {
            "schema": "boxfusion.ca1m_tr3d_terminal_code_manifest.v1",
            "files": source_hashes,
        },
        sort_keys=True,
    )
    expected_code_sha = hashlib.sha256(expected_code_manifest.encode()).hexdigest()
    reports: dict[str, Any] = {}
    totals = {"anchors": 0, "candidates": 0, "near": 0, "represented": 0}

    for scene in scenes:
        artifact = args.observer_root / f"{scene}_ca1m_tr3d_terminal.npz"
        if artifact.is_symlink() or not artifact.is_file():
            raise FileNotFoundError(f"missing regular observer artifact: {artifact}")
        if artifact.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError(f"observer artifact is writable: {artifact}")
        anchor_path = args.anchor_root / f"{scene}_boxes.pkl"
        diagnostic_path = (
            args.native_b6_diagnostics_root / f"{scene}_ca1m_native_b6.npz"
        )
        corners, detector_scores = _prediction(anchor_path)
        evidence = load_native_observer_diagnostic(
            diagnostic_path,
            scene_id=scene,
            corners=corners,
            scores=detector_scores,
        )
        active_scores = np.asarray(
            scorer.predict(evidence["features"], detector_scores).scores,
            dtype=np.float32,
        )
        expected_frames = _used_frame_ids(diagnostic_path, scene)
        rgb, depth, poses, intrinsics = _scene_inputs(
            args.data_root / scene, expected_frames
        )
        points = _terminal_points(
            rgb=rgb,
            depth=depth,
            poses=poses,
            intrinsics=intrinsics,
            used_frames=expected_frames,
            pixel_stride=4,
            min_depth=0.10,
            max_depth=6.0,
            voxel_size=0.01,
        )
        expected_point_sha = hashlib.sha256(points.tobytes(order="C")).hexdigest()
        expected_transform = terminal_world_to_local(poses[int(expected_frames[0])])

        with np.load(artifact, allow_pickle=False) as archive:
            if set(archive.files) != EXPECTED_KEYS:
                raise ValueError(
                    f"observer keys differ: missing={sorted(EXPECTED_KEYS-set(archive.files))}, "
                    f"extra={sorted(set(archive.files)-EXPECTED_KEYS)}"
                )
            if _scalar(archive, "schema") != SCHEMA or _scalar(archive, "scene_id") != scene:
                raise ValueError("observer schema/scene mismatch")
            for key, expected in (
                ("complete", True),
                ("observer_only", True),
                ("mutation_enabled", False),
                ("ground_truth_access", False),
                ("applied_count", 0),
                ("coordinate_frame", COORDINATE_FRAME),
                ("box_mode", BOX_MODE),
                ("corner_semantics", CORNER_SEMANTICS),
                ("prefix_id", "p100_gap20"),
                ("pixel_stride", 4),
                ("voxel_size_m", 0.01),
                ("min_depth_m", 0.10),
                ("max_depth_m", 6.0),
                ("near_iou", 0.15),
                ("score_threshold", 0.01),
                ("max_proposals", 256),
            ):
                if _scalar(archive, key) != expected:
                    raise ValueError(f"observer scalar contract differs: {key}")
            adapter_mode = str(_scalar(archive, "adapter_mode"))
            if adapter_mode not in {"genuine", "synthetic"}:
                raise ValueError("invalid adapter mode")
            if args.require_genuine and adapter_mode != "genuine":
                raise ValueError("formal observer audit requires genuine TR3D")
            runtime_s = float(_scalar(archive, "model_runtime_s"))
            if not np.isfinite(runtime_s) or runtime_s < 0.0:
                raise ValueError("invalid model runtime")
            if args.require_genuine and runtime_s <= 0.0:
                raise ValueError("genuine TR3D runtime must be positive")
            for key, expected in (
                ("source_anchor_prediction_sha256", sha256_file(anchor_path)),
                ("active_anchor_scores_sha256", sha256_array(active_scores)),
                ("native_b6_diagnostic_sha256", sha256_file(diagnostic_path)),
                ("native_b6_checkpoint_sha256", b6_checkpoint_sha),
                ("native_b6_manifest_sha256", b6_manifest_sha),
                ("source_points_sha256", expected_point_sha),
                ("checkpoint_sha256", checkpoint_sha),
                ("config_sha256", config_sha),
                ("code_manifest_sha256", expected_code_sha),
                ("code_manifest_json", expected_code_manifest),
            ):
                if _scalar(archive, key) != expected:
                    raise ValueError(f"observer provenance differs: {key}")
            _equal(np.array(archive["used_frame_ids"]), expected_frames, "used frames")
            _equal(np.array(archive["world_to_local"]), expected_transform, "world_to_local")
            _equal(np.array(archive["anchor_corners"]), corners, "anchor corners")
            _equal(np.array(archive["anchor_scores"]), active_scores, "active scores")
            if int(_scalar(archive, "point_count")) != len(points):
                raise ValueError("point count differs from RGB-D reconstruction")
            candidates = np.array(archive["candidate_corners"], copy=True)
            candidate_scores = np.array(archive["candidate_scores"], copy=True)
            local_boxes = np.array(archive["candidate_boxes_local"], copy=True)
            labels = np.array(archive["candidate_labels"], copy=True)
            support = np.array(archive["candidate_point_count"], copy=True)
            count = len(candidates)
            if (
                candidates.dtype != np.float32
                or candidates.shape != (count, 8, 3)
                or candidate_scores.dtype != np.float32
                or candidate_scores.shape != (count,)
                or local_boxes.dtype != np.float32
                or local_boxes.shape != (count, 7)
                or labels.dtype != np.int64
                or labels.shape != (count,)
                or support.dtype != np.int64
                or support.shape != (count,)
                or count > 256
                or not np.isfinite(candidates).all()
                or not np.isfinite(candidate_scores).all()
                or np.any(candidate_scores < 0.01)
                or np.any(candidate_scores > 1.0)
                or (count and np.any(np.diff(candidate_scores) > 0.0))
                or np.any(labels != 0)
                or np.any(support < 0)
                or np.any(support > len(points))
            ):
                raise ValueError("candidate array contract is invalid")
            world_aabb(candidates)
            _equal(
                aligned_boxes_to_world_corners(local_boxes, expected_transform),
                candidates,
                "local-to-world candidate geometry",
            )
            association = associate_terminal_candidates(
                anchor_corners=corners,
                anchor_scores=active_scores,
                candidate_corners=candidates,
                candidate_scores=candidate_scores,
                near_iou=0.15,
            )
            for key, expected in (
                ("best_anchor_indices", association.best_anchor_indices),
                ("best_anchor_iou", association.best_anchor_iou),
                (
                    "best_anchor_center_distance_m",
                    association.best_anchor_center_distance_m,
                ),
                ("near_mask", association.near_mask),
                ("represented_anchor_indices", association.represented_anchor_indices),
                (
                    "legacy_rule_selected_candidate_rows",
                    association.legacy_rule_selected_candidate_rows,
                ),
                (
                    "legacy_rule_selected_anchor_indices",
                    association.legacy_rule_selected_anchor_indices,
                ),
            ):
                _equal(np.array(archive[key]), expected, key)
            summary = json.loads(str(_scalar(archive, "summary_json")))
            if (
                summary.get("adapter_mode") != adapter_mode
                or summary.get("candidate_count") != count
                or summary.get("anchor_count") != len(corners)
                or summary.get("near_candidate_count")
                != int(association.near_mask.sum())
                or summary.get("represented_anchor_count")
                != len(association.represented_anchor_indices)
                or summary.get("legacy_rule_selected_count")
                != len(association.legacy_rule_selected_candidate_rows)
                or summary.get("applied_count") != 0
                or summary.get("validation_policy_selection_authorized") is not False
            ):
                raise ValueError("summary JSON disagrees with audited arrays")
            if args.materialized_active_root is not None:
                materialized_corners, materialized_scores = _prediction(
                    args.materialized_active_root / f"{scene}_boxes.pkl"
                )
                _equal(materialized_corners, corners, "materialized active corners")
                _equal(materialized_scores, active_scores, "materialized active scores")
                if _scalar(archive, "materialized_active_verified") is not True:
                    raise ValueError("materialized active verification flag is false")

        near = int(association.near_mask.sum())
        represented = len(association.represented_anchor_indices)
        reports[scene] = {
            "artifact_sha256": sha256_file(artifact),
            "adapter_mode": adapter_mode,
            "anchors": len(corners),
            "candidates": count,
            "near_candidates": near,
            "represented_anchors": represented,
            "legacy_diagnostic_selections": len(
                association.legacy_rule_selected_candidate_rows
            ),
            "model_runtime_s": runtime_s,
            "point_count": len(points),
            "positive_support_candidates": int(np.count_nonzero(support)),
        }
        totals["anchors"] += len(corners)
        totals["candidates"] += count
        totals["near"] += near
        totals["represented"] += represented

    result = {
        "schema": "boxfusion.ca1m_tr3d_terminal_observer_audit.v1",
        "ok": True,
        "observer_only": True,
        "ground_truth_access": False,
        "scene_count": len(scenes),
        "totals": totals,
        "scenes": reports,
    }
    if args.output is not None:
        _write_json_create_only(args.output, result)
    return result


def main() -> int:
    result = run(parser().parse_args())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
