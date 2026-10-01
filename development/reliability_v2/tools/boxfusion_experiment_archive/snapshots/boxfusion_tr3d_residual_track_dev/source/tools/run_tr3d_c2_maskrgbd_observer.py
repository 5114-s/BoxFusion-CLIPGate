#!/usr/bin/env python3
"""Export C2 Mask-RGBD confirmations without mutating predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.supplemental_proposals import NpzProposalCache  # noqa: E402
from boxfusion.tr3d_c1_track_cache import load_sidecar as load_c1_sidecar  # noqa: E402
from boxfusion.tr3d_c1_track_cache import sidecar_path as c1_sidecar_path  # noqa: E402
from boxfusion.tr3d_c2_maskrgbd_cache import (  # noqa: E402
    TR3DC2MaskRGBDCache,
    canonical_json,
    sha256_bytes,
    sha256_file,
    sidecar_path,
    write_sidecar,
)
from boxfusion.tr3d_c2_maskrgbd_observer import (  # noqa: E402
    C2Frame,
    C2MaskRGBDConfig,
    GATE_NAMES,
    observe_scene,
)
from boxfusion.tr3d_r2_geometry import compose_depth_camera_to_world  # noqa: E402
from boxfusion.tr3d_residual_cache import (  # noqa: E402
    load_tr3d_residual_cache,
    tr3d_residual_cache_path,
)
from tools.run_tr3d_c1_track_observer import _tree_snapshot  # noqa: E402
from tools.tr3d_data import read_scene_list  # noqa: E402


REPORT_SCHEMA = "boxfusion.tr3d_c2_maskrgbd_observer_export.v1"
RUNTIME_SCHEMA = "boxfusion_scannet_runtime_rgb_v1"
TEACHER_SCHEMA = "boxfusion_scannet_sam3_teacher_provenance_v1"


def _code_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        _ROOT / "boxfusion" / "tr3d_c2_maskrgbd_observer.py",
        _ROOT / "boxfusion" / "tr3d_c2_maskrgbd_cache.py",
        Path(__file__).resolve(),
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_json_create_only(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"immutable C2 report exists: {path}") from error
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _manifest_set(paths: Sequence[Path], schema: str) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    identities: list[dict[str, str]] = []
    for path in sorted(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != schema:
            raise ValueError(f"{path}: unsupported manifest schema")
        rows.append(payload)
        identities.append({"name": path.name, "sha256": sha256_file(path)})
    if not rows:
        raise ValueError(f"no {schema} manifests found")
    return rows, sha256_bytes(canonical_json(identities).encode("utf-8"))


def _load_manifests(
    teacher_cache_root: Path,
    expected_scene_sha: str,
) -> tuple[dict[str, list[dict[str, Any]]], str, str, str]:
    teacher, teacher_set_sha = _manifest_set(
        list((teacher_cache_root / "manifests").glob("provenance_*.json")),
        TEACHER_SCHEMA,
    )
    runtime, runtime_set_sha = _manifest_set(
        list((teacher_cache_root / "runtime_rgb" / "manifests").glob("runtime_rgb_*.json")),
        RUNTIME_SCHEMA,
    )
    namespaces = {str(row["namespace"]) for row in teacher + runtime}
    if len(namespaces) != 1:
        raise ValueError("teacher/runtime namespace mismatch")
    for row in teacher + runtime:
        if str(row["scene_list"]["sha256"]) != expected_scene_sha:
            raise ValueError("teacher cache belongs to a different scene list")
    frames: dict[str, list[dict[str, Any]]] = {}
    for manifest in runtime:
        if not bool(manifest.get("complete")):
            raise ValueError("runtime RGB manifest is incomplete")
        for frame in manifest["frames"]:
            scene = str(frame["scene_id"])
            if str(frame.get("orientation")) != "upright":
                raise ValueError(
                    f"{scene}:{frame.get('frame_index')}: C2 currently requires upright cache frames"
                )
            frames.setdefault(scene, []).append(frame)
    for scene, values in frames.items():
        values.sort(key=lambda row: int(row["frame_index"]))
        ids = [int(row["frame_index"]) for row in values]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{scene}: duplicate runtime cache frame ids")
    return frames, teacher_set_sha, runtime_set_sha, namespaces.pop()


def _load_matrix(path: Path) -> np.ndarray:
    value = np.loadtxt(path, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f"{path}: expected a finite 4x4 matrix")
    return value


def _load_scene_frames(
    scene_id: str,
    frame_rows: Sequence[dict[str, Any]],
    teacher_cache: NpzProposalCache,
    config: C2MaskRGBDConfig,
) -> tuple[tuple[C2Frame, ...], str]:
    if not frame_rows:
        raise ValueError(f"{scene_id}: no SAM3 teacher frames")
    first_depth = Path(frame_rows[0]["sources"]["depth"]["path"])
    scan_root = first_depth.parents[1]
    if scan_root.name != scene_id:
        raise ValueError(f"{scene_id}: malformed ScanNet source path")
    intrinsic_path = scan_root / "intrinsic" / "intrinsic_depth.txt"
    extrinsic_path = scan_root / "intrinsic" / "extrinsic_depth.txt"
    intrinsic = _load_matrix(intrinsic_path)
    extrinsic = _load_matrix(extrinsic_path)
    input_rows: list[dict[str, Any]] = [
        {"intrinsic": sha256_file(intrinsic_path), "extrinsic": sha256_file(extrinsic_path)}
    ]
    frames: list[C2Frame] = []
    for row in frame_rows:
        frame_id = int(row["frame_index"])
        depth_path = Path(row["sources"]["depth"]["path"])
        pose_path = Path(row["sources"]["pose"]["path"])
        depth_sha = sha256_file(depth_path)
        pose_sha = sha256_file(pose_path)
        if depth_sha != str(row["sources"]["depth"]["sha256"]):
            raise ValueError(f"{scene_id}:{frame_id}: depth source hash mismatch")
        if pose_sha != str(row["sources"]["pose"]["sha256"]):
            raise ValueError(f"{scene_id}:{frame_id}: pose source hash mismatch")
        pose_raw = np.loadtxt(pose_path, dtype=np.float64)
        if pose_raw.shape != (4, 4) or not np.isfinite(pose_raw).all():
            input_rows.append(
                {
                    "frame_id": frame_id,
                    "depth_sha256": depth_sha,
                    "pose_sha256": pose_sha,
                    "skipped": "invalid_nonfinite_pose",
                }
            )
            continue
        depth_raw = np.asarray(Image.open(depth_path))
        if depth_raw.ndim != 2:
            raise ValueError(f"{depth_path}: depth is not single-channel")
        depth = depth_raw.astype(np.float32) / float(config.depth_scale)
        camera_to_world = compose_depth_camera_to_world(pose_raw, extrinsic)
        key = str(row["proposal_cache_key"])
        cache_path = teacher_cache.path_for_key(key)
        proposals = teacher_cache.load(key, expected_image_shape=depth.shape)
        if proposals is None:
            raise FileNotFoundError(f"missing strict SAM3 teacher cache: {cache_path}")
        cache_sha = sha256_file(cache_path)
        frames.append(
            C2Frame(
                frame_id=frame_id,
                depth_meters=depth,
                intrinsics=intrinsic,
                depth_camera_to_world=camera_to_world,
                proposals=tuple(proposals),
                cache_sha256=cache_sha,
            )
        )
        input_rows.append(
            {
                "frame_id": frame_id,
                "depth_sha256": depth_sha,
                "pose_sha256": pose_sha,
                "proposal_cache_key": key,
                "proposal_cache_sha256": cache_sha,
            }
        )
    if not frames:
        raise ValueError(f"{scene_id}: no valid-pose SAM3 teacher frames")
    return tuple(frames), sha256_bytes(canonical_json(input_rows).encode("utf-8"))


def export(args: argparse.Namespace) -> dict[str, Any]:
    scene_list = args.scene_list.resolve()
    scenes = read_scene_list(scene_list)
    scene_sha = sha256_file(scene_list)
    active_root = args.active_prediction_root.resolve()
    before = _tree_snapshot(active_root, scenes)
    frame_map, teacher_set_sha, runtime_set_sha, namespace = _load_manifests(
        args.teacher_cache_root.resolve(), scene_sha
    )
    if set(frame_map) != set(scenes):
        raise ValueError("teacher cache scene coverage differs from requested scene list")
    teacher_cache = NpzProposalCache(args.teacher_cache_root.resolve(), write_enabled=False)
    config = C2MaskRGBDConfig(source_budget=args.source_budget)
    config_json = canonical_json(config.as_dict())
    config_sha = sha256_bytes(config_json.encode("utf-8"))
    code_sha = _code_hash()
    rows: list[dict[str, Any]] = []
    totals = {
        "source_candidates": 0, "teacher_frames": 0, "projected_views": 0,
        "matched_views": 0, "strong_views": 0,
        "gates": {name: 0 for name in GATE_NAMES},
    }
    for position, scene_id in enumerate(scenes, 1):
        started = time.perf_counter()
        c1_path = c1_sidecar_path(args.c1_cache_root.resolve(), scene_id, args.prefix_id)
        c1 = load_c1_sidecar(c1_path)
        parent_path = tr3d_residual_cache_path(
            args.parent_cache_root.resolve(), scene_id, args.prefix_id
        )
        if sha256_file(parent_path) != c1.parent_cache_sha256:
            raise ValueError(f"{scene_id}: C1 parent cache hash mismatch")
        with np.load(parent_path, allow_pickle=False) as raw:
            checkpoint_sha = str(np.asarray(raw["checkpoint_sha256"]).item())
            parent_config_sha = str(np.asarray(raw["config_sha256"]).item())
        parent = load_tr3d_residual_cache(
            parent_path, expected_scene_id=scene_id, expected_prefix_id=args.prefix_id,
            expected_checkpoint_sha256=checkpoint_sha,
            expected_config_sha256=parent_config_sha,
        )
        anchor_path = active_root / f"{scene_id}_boxes.pkl"
        if sha256_file(anchor_path) != c1.anchor_prediction_sha256:
            raise ValueError(f"{scene_id}: frozen active anchor hash mismatch")
        order = np.argsort(-c1.depth_feature_track_score, kind="stable")[: config.source_budget]
        if not np.array_equal(parent.proposal_ids[c1.parent_rows], c1.proposal_ids):
            raise ValueError(f"{scene_id}: C1/parent row identity mismatch")
        frames, scene_input_sha = _load_scene_frames(
            scene_id, frame_map[scene_id], teacher_cache, config
        )
        boxes = parent.boxes_world[c1.parent_rows[order]]
        observation = observe_scene(boxes, frames, config)
        elapsed = time.perf_counter() - started
        sidecar = TR3DC2MaskRGBDCache(
            scene_id=scene_id,
            prefix_id=args.prefix_id,
            c1_sidecar_sha256=sha256_file(c1_path),
            parent_cache_sha256=sha256_file(parent_path),
            anchor_prediction_sha256=sha256_file(anchor_path),
            teacher_manifest_set_sha256=teacher_set_sha,
            runtime_manifest_set_sha256=runtime_set_sha,
            scene_frame_input_sha256=scene_input_sha,
            config_sha256=config_sha,
            code_sha256=code_sha,
            config_json=config_json,
            source_c1_rows=order.astype(np.int64),
            source_ranks=np.arange(1, len(order) + 1, dtype=np.int32),
            proposal_ids=c1.proposal_ids[order],
            parent_rows=c1.parent_rows[order],
            c1_track_scores=c1.depth_feature_track_score[order],
            frame_cache_sha256=np.asarray([frame.cache_sha256 for frame in frames]),
            observation=observation,
            runtime_s=elapsed,
        )
        target = sidecar_path(args.output_root.resolve(), scene_id, args.prefix_id)
        sidecar_sha = write_sidecar(target, sidecar)
        gate_counts = observation.gate_mask.sum(axis=0, dtype=np.int64)
        row = {
            "scene_id": scene_id,
            "source_candidates": len(order),
            "teacher_frames": len(frames),
            "projected_views": int(observation.projected_view_count.sum()),
            "matched_views": int(observation.matched_view_count.sum()),
            "strong_views": int(observation.strong_view_count.sum()),
            "gate_counts": {
                name: int(gate_counts[index]) for index, name in enumerate(GATE_NAMES)
            },
            "sidecar": str(target),
            "sidecar_sha256": sidecar_sha,
            "runtime_s": elapsed,
        }
        rows.append(row)
        for name in ("source_candidates", "teacher_frames", "projected_views", "matched_views", "strong_views"):
            totals[name] += int(row[name])
        for name in GATE_NAMES:
            totals["gates"][name] += row["gate_counts"][name]
        print(
            f"[{position}/{len(scenes)}] {scene_id}: source={len(order)}, "
            f"views(projected/matched/strong)={row['projected_views']}/"
            f"{row['matched_views']}/{row['strong_views']}, gates={row['gate_counts']}",
            flush=True,
        )
    after = _tree_snapshot(active_root, scenes)
    if before != after:
        raise RuntimeError("frozen R3-active prediction tree changed during C2 export")
    return {
        "schema": REPORT_SCHEMA,
        "observer_only": True,
        "mutation_enabled": False,
        "applied_count": 0,
        "ground_truth_access": False,
        "clip_access": False,
        "clip_semantics_unchanged": True,
        "teacher_labels_used_for_gate": False,
        "scene_list": str(scene_list),
        "scene_list_sha256": scene_sha,
        "scene_count": len(scenes),
        "prefix_id": args.prefix_id,
        "source": "C1_depth_feature_track_topk_per_scene",
        "source_budget": config.source_budget,
        "teacher_namespace": namespace,
        "teacher_cache_root": str(args.teacher_cache_root.resolve()),
        "teacher_manifest_set_sha256": teacher_set_sha,
        "runtime_manifest_set_sha256": runtime_set_sha,
        "c1_cache_root": str(args.c1_cache_root.resolve()),
        "parent_cache_root": str(args.parent_cache_root.resolve()),
        "active_prediction_root": str(active_root),
        "output_root": str(args.output_root.resolve()),
        "config": config.as_dict(),
        "config_sha256": config_sha,
        "code_sha256": code_sha,
        "counts": totals,
        "frozen_active_before": before,
        "frozen_active_after": after,
        "scenes": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--active-prediction-root", type=Path, required=True)
    parser.add_argument("--parent-cache-root", type=Path, required=True)
    parser.add_argument("--c1-cache-root", type=Path, required=True)
    parser.add_argument("--teacher-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--prefix-id", default="p100")
    parser.add_argument("--source-budget", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = export(args)
    _write_json_create_only(args.report.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
