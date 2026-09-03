#!/usr/bin/env python3
"""Export paired depth/free-space evidence for selected terminal-R3 changes.

The command is offline and observer-only.  It never opens prediction files in
write mode and verifies that the active R3 prediction tree is byte-identical
before and after export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.tr3d_r2_observer import (  # noqa: E402
    TR3DR2FrameBundle,
    TR3DR2ObserverConfig,
)
from boxfusion.tr3d_r2_provenance import (  # noqa: E402
    canonical_json_sha256,
    code_artifact_tree_sha256,
    frame_artifact_tree,
    load_prefix_manifest,
    load_resolved_poses,
    sha256_file,
)
from boxfusion.tr3d_residual_cache import (  # noqa: E402
    load_tr3d_residual_cache,
    tr3d_residual_cache_path,
)
from boxfusion.tr3d_r4_smov_cache import (  # noqa: E402
    load_r4_depth_sidecar,
    make_r4_depth_sidecar,
    write_r4_depth_sidecar,
)
from boxfusion.tr3d_r4_smov_observer import (  # noqa: E402
    corners_to_yaw_boxes,
    observe_r3_replacement_pairs,
)
from tools.tr3d_data import discover_frame_bundle, read_scene_list  # noqa: E402


REPORT_SCHEMA = "boxfusion.tr3d_r4_smov_depth_export.v1"
DIAGNOSTIC_SCHEMA = "boxfusion.tr3d_r3_terminal_active_scene.v1"
POSE_SOURCE = "scannet_g0_resolved_pose_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-cache-root", type=Path, required=True)
    parser.add_argument("--prefix-manifest", type=Path, required=True)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--same-run-baseline-root", type=Path, required=True)
    parser.add_argument("--active-prediction-root", type=Path, required=True)
    parser.add_argument("--r3-diagnostics-root", type=Path, required=True)
    parser.add_argument("--r4-cache-root", type=Path, required=True)
    parser.add_argument("--prefix-id", default="p100")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--min-depth", type=float, default=0.10)
    parser.add_argument("--max-depth", type=float, default=8.0)
    parser.add_argument("--near-clip", type=float, default=1e-3)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    return parser


def _array_sha(value: object) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - local immutable artifact
    if (
        not isinstance(payload, (list, tuple))
        or len(payload) != 1
        or not isinstance(payload[0], (list, tuple))
    ):
        raise ValueError(f"{path}: malformed prediction payload")
    rows = payload[0]
    corners: list[np.ndarray] = []
    scores: list[float] = []
    for index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError(f"{path}: malformed prediction row {index}")
        value = np.asarray(row[1])
        if value.shape != (8, 3) or not np.isfinite(value).all():
            raise ValueError(f"{path}: malformed corners {index}")
        corners.append(value)
        scores.append(float(row[2]))
    if corners:
        dtype = corners[0].dtype
        if any(value.dtype != dtype for value in corners):
            raise ValueError(f"{path}: mixed corner dtypes")
        geometry = np.ascontiguousarray(np.stack(corners))
    else:
        geometry = np.empty((0, 8, 3), dtype=np.float32)
    return geometry, np.asarray(scores, dtype=np.float64)


def _tree_hash(root: Path, scenes: Sequence[str]) -> str:
    records: list[dict[str, str]] = []
    for scene in scenes:
        path = root / f"{scene}_boxes.pkl"
        records.append({"name": path.name, "sha256": sha256_file(path)})
    return canonical_json_sha256(records)


def _load_diagnostic(path: Path, scene: str) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    encoded = path.read_bytes()
    try:
        value = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: malformed JSON") from error
    if (
        not isinstance(value, dict)
        or value.get("schema") != DIAGNOSTIC_SCHEMA
        or value.get("scene_id") != scene
        or value.get("ground_truth_access") is not False
        or value.get("clip_access") is not False
    ):
        raise ValueError(f"{path}: invalid terminal-R3 diagnostic contract")
    return value, hashlib.sha256(encoded).hexdigest()


def _config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": "boxfusion.tr3d_r4_smov_depth_config.v1",
        "observer_only": True,
        "mutation_enabled": False,
        "applied_count": 0,
        "ground_truth_access": False,
        "clip_access": False,
        "feature_consistency_enabled": False,
        "comparison": "candidate_minus_same_run_anchor",
        "view_policy": "common_visibility_min_projected_area_topk",
        "pair_roles": ["anchor", "candidate"],
        "depth_classes": ["support", "occluded", "free_space", "invalid"],
        "top_k": args.top_k,
        "pixel_stride": args.pixel_stride,
        "depth_scale": args.depth_scale,
        "margin_m": args.margin,
        "min_depth_m": args.min_depth,
        "max_depth_m": args.max_depth,
        "near_clip_m": args.near_clip,
        "image_shape": [args.image_height, args.image_width],
    }


def _write_create_only(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"immutable R4 report exists: {path}") from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def export(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index must lie in [0,num-shards)")
    all_scenes = read_scene_list(args.scene_list.resolve())
    scenes = [
        scene for index, scene in enumerate(all_scenes)
        if index % args.num_shards == args.shard_index
    ]
    prefix_rows = load_prefix_manifest(
        args.prefix_manifest.resolve(), prefix_id=args.prefix_id
    )
    config_payload = _config(args)
    config_sha = canonical_json_sha256(config_payload)
    code_sha = code_artifact_tree_sha256(
        (
            _ROOT / "boxfusion" / "tr3d_r2_geometry.py",
            _ROOT / "boxfusion" / "tr3d_r2_observer.py",
            _ROOT / "boxfusion" / "tr3d_r4_smov_observer.py",
            _ROOT / "boxfusion" / "tr3d_r4_smov_cache.py",
            Path(__file__),
        )
    )
    observer_config = TR3DR2ObserverConfig(
        image_shape=(args.image_height, args.image_width),
        pose_source=POSE_SOURCE,
        top_k=args.top_k,
        pixel_stride=args.pixel_stride,
        depth_scale=args.depth_scale,
        margin=args.margin,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        near_clip=args.near_clip,
    )
    tree_before = _tree_hash(args.active_prediction_root.resolve(), scenes)
    reports: list[dict[str, Any]] = []
    total_pairs = 0
    resumed = 0

    for position, scene in enumerate(scenes, start=1):
        if scene not in prefix_rows:
            raise ValueError(f"prefix manifest is missing {scene}")
        row = prefix_rows[scene]
        recorded_root = Path(str(row.get("source_frames_root", ""))).resolve()
        if recorded_root != args.frames_root.resolve():
            raise ValueError(f"{scene}: frames root provenance mismatch")
        baseline_path = args.same_run_baseline_root.resolve() / f"{scene}_boxes.pkl"
        baseline_geometry, baseline_scores = _prediction(baseline_path)
        diagnostic_path = (
            args.r3_diagnostics_root.resolve()
            / f"{scene}_tr3d_terminal.json"
        )
        diagnostic, diagnostic_sha = _load_diagnostic(diagnostic_path, scene)
        if (
            diagnostic.get("input_geometry_sha256")
            != _array_sha(baseline_geometry)
            or diagnostic.get("input_scores_sha256")
            != _array_sha(baseline_scores)
        ):
            raise ValueError(f"{scene}: diagnostic is not bound to baseline")
        manifest_sha = canonical_json_sha256(row)
        if diagnostic.get("manifest_row_sha256") != manifest_sha:
            raise ValueError(f"{scene}: diagnostic/manifest hash mismatch")

        parent_path = tr3d_residual_cache_path(
            args.parent_cache_root.resolve(), scene, args.prefix_id
        )
        parent_sha = sha256_file(parent_path)
        if diagnostic.get("cache_sha256") != parent_sha:
            raise ValueError(f"{scene}: diagnostic/parent hash mismatch")
        point_path = Path(str(row["point_path"])).resolve()
        parent = load_tr3d_residual_cache(
            parent_path,
            expected_scene_id=scene,
            expected_prefix_id=args.prefix_id,
            expected_source_scene_sha256=sha256_file(point_path),
        )
        selection_rows = diagnostic.get("selections")
        if not isinstance(selection_rows, list):
            raise ValueError(f"{scene}: diagnostic selections are malformed")
        if len(selection_rows) != int(diagnostic.get("selected_count", -1)):
            raise ValueError(f"{scene}: selected count mismatch")
        anchor_indices: list[int] = []
        parent_rows: list[int] = []
        tr3d_scores: list[float] = []
        anchor_scores: list[float] = []
        anchor_iou: list[float] = []
        for item in selection_rows:
            if not isinstance(item, dict):
                raise ValueError(f"{scene}: malformed selection")
            anchor = int(item["anchor_index"])
            parent_row = int(item["proposal_row"])
            if not 0 <= anchor < len(baseline_geometry):
                raise ValueError(f"{scene}: anchor index out of bounds")
            if not 0 <= parent_row < parent.proposal_count:
                raise ValueError(f"{scene}: parent row out of bounds")
            if (
                int(parent.proposal_ids[parent_row]) != int(item["proposal_id"])
                or float(parent.scores_3d[parent_row]) != float(item["tr3d_score"])
                or float(baseline_scores[anchor]) != float(item["anchor_score"])
            ):
                raise ValueError(f"{scene}: selection lineage mismatch")
            anchor_indices.append(anchor)
            parent_rows.append(parent_row)
            tr3d_scores.append(float(item["tr3d_score"]))
            anchor_scores.append(float(item["anchor_score"]))
            anchor_iou.append(float(item["anchor_iou"]))

        anchor_indices_array = np.asarray(anchor_indices, dtype=np.int64)
        parent_rows_array = np.asarray(parent_rows, dtype=np.int64)
        anchor_boxes = corners_to_yaw_boxes(
            baseline_geometry[anchor_indices_array]
        )
        candidate_boxes = np.asarray(
            parent.boxes_world[parent_rows_array], dtype=np.float64
        )
        proposal_ids = np.asarray(
            parent.proposal_ids[parent_rows_array], dtype=np.int64
        )
        bundle = discover_frame_bundle(args.frames_root.resolve(), scene)
        frame_tree_sha, _ = frame_artifact_tree(row, bundle)
        resolved_poses = load_resolved_poses(row, bundle)
        used_ids = [int(value) for value in row["used_frame_ids"]]
        frame_bundle = TR3DR2FrameBundle(
            scene_id=scene,
            pose_source=POSE_SOURCE,
            depth={frame_id: bundle.depth[frame_id] for frame_id in used_ids},
            pose=resolved_poses,
            intrinsic_depth=bundle.intrinsic_depth,
            extrinsic_depth=bundle.extrinsic_depth,
        )
        observer_manifest = {
            "scene_id": scene,
            "used_frame_ids": used_ids,
            "pose_source": POSE_SOURCE,
        }
        target = args.r4_cache_root.resolve() / scene / f"{args.prefix_id}.r4d.npz"
        if target.exists():
            if not args.resume:
                raise FileExistsError(f"immutable R4 sidecar exists: {target}")
            sidecar = load_r4_depth_sidecar(target)
            expected = {
                "parent_cache_sha256": parent_sha,
                "prefix_manifest_row_sha256": manifest_sha,
                "frame_artifact_tree_sha256": frame_tree_sha,
                "r3_diagnostic_sha256": diagnostic_sha,
                "input_geometry_sha256": diagnostic["input_geometry_sha256"],
                "input_scores_sha256": diagnostic["input_scores_sha256"],
                "r4_config_sha256": config_sha,
                "r4_code_sha256": code_sha,
            }
            if any(getattr(sidecar, name) != value for name, value in expected.items()):
                raise ValueError(f"{scene}: resumed sidecar provenance mismatch")
            resumed += 1
        else:
            observation = observe_r3_replacement_pairs(
                anchor_boxes_world=anchor_boxes,
                candidate_boxes_world=candidate_boxes,
                proposal_ids=proposal_ids,
                anchor_indices=anchor_indices_array,
                prefix_manifest=observer_manifest,
                frame_bundle=frame_bundle,
                config=observer_config,
            )
            sidecar = make_r4_depth_sidecar(
                observation=observation,
                scene_id=scene,
                prefix_id=args.prefix_id,
                final_source_timestamp=int(row["last_source_timestamp"]),
                parent_cache_sha256=parent_sha,
                prefix_manifest_row_sha256=manifest_sha,
                frame_artifact_tree_sha256=frame_tree_sha,
                r3_diagnostic_sha256=diagnostic_sha,
                input_geometry_sha256=diagnostic["input_geometry_sha256"],
                input_scores_sha256=diagnostic["input_scores_sha256"],
                r4_config_sha256=config_sha,
                r4_code_sha256=code_sha,
                tr3d_scores=tr3d_scores,
                anchor_scores=anchor_scores,
                anchor_iou=anchor_iou,
                anchor_boxes_world=anchor_boxes,
                candidate_boxes_world=candidate_boxes,
            )
            write_r4_depth_sidecar(target, sidecar)
        total_pairs += sidecar.pair_count
        reports.append(
            {
                "scene_id": scene,
                "selected_pairs": sidecar.pair_count,
                "common_views": int(sidecar.aggregate_view_count.sum()),
                "sampled_anchor_rays": int(sidecar.aggregate_point_count[:, 0].sum()),
                "sampled_candidate_rays": int(sidecar.aggregate_point_count[:, 1].sum()),
                "runtime_s": float(sidecar.runtime_s),
                "resumed": target.exists() and position <= resumed,
                "sidecar": str(target),
            }
        )
        print(
            f"[{position}/{len(scenes)}] {scene}: pairs={sidecar.pair_count}, "
            f"common_views={reports[-1]['common_views']}, "
            f"runtime={sidecar.runtime_s:.3f}s",
            flush=True,
        )

    tree_after = _tree_hash(args.active_prediction_root.resolve(), scenes)
    if tree_before != tree_after:
        raise RuntimeError("active R3 prediction tree changed during R4 observer")
    return {
        "schema": REPORT_SCHEMA,
        "observer_only": True,
        "mutation_enabled": False,
        "applied_count": 0,
        "ground_truth_access": False,
        "clip_access": False,
        "feature_consistency_enabled": False,
        "active_prediction_tree_sha256_before": tree_before,
        "active_prediction_tree_sha256_after": tree_after,
        "active_prediction_identity_ok": True,
        "scene_count": len(scenes),
        "selected_pair_count": total_pairs,
        "resumed_scene_count": resumed,
        "r4_config": config_payload,
        "r4_config_sha256": config_sha,
        "r4_code_sha256": code_sha,
        "parent_cache_root": str(args.parent_cache_root.resolve()),
        "prefix_manifest": str(args.prefix_manifest.resolve()),
        "same_run_baseline_root": str(args.same_run_baseline_root.resolve()),
        "active_prediction_root": str(args.active_prediction_root.resolve()),
        "r3_diagnostics_root": str(args.r3_diagnostics_root.resolve()),
        "r4_cache_root": str(args.r4_cache_root.resolve()),
        "scenes": reports,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = export(args)
    _write_create_only(args.report.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
