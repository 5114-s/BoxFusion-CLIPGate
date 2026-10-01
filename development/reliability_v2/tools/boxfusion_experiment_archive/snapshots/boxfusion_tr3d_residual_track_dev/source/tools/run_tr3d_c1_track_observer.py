#!/usr/bin/env python3
"""Build immutable unmatched-TR3D cross-view evidence tracks.

This exporter has no GT input and no prediction-output argument.  The frozen
R3-active prediction files are read only to define which TR3D proposals are
unmatched.  Every output is an observer sidecar with ``applied_count=0``.
"""

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

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.tr3d_c1_track_cache import (  # noqa: E402
    FEATURE_COSINE_MIN,
    GATE_NAMES,
    MIN_VIEW_SAMPLES,
    RESIDUAL_ANCHOR_IOU_MAX,
    SUPPORTIVE_VIEW_FREE_MAX,
    SUPPORTIVE_VIEW_SUPPORT_MIN,
    TRACK_SCOPE,
    TR3DC1TrackCache,
    derive_track_features,
    sha256_file,
    sidecar_path,
    write_sidecar,
)
from boxfusion.tr3d_r2_cache import load_tr3d_r2_cache, tr3d_r2_cache_path  # noqa: E402
from boxfusion.tr3d_r2b_cache import load_tr3d_r2b_cache, tr3d_r2b_cache_path  # noqa: E402
from boxfusion.tr3d_residual_cache import (  # noqa: E402
    load_tr3d_residual_cache,
    tr3d_residual_cache_path,
)
from tools.audit_tr3d_residual_observer import (  # noqa: E402
    _alignment,
    _load_b6,
    _minmax,
    _transform,
    _validate_alignment_provenance,
    pairwise_iou,
)
from tools.tr3d_data import read_scene_list  # noqa: E402


REPORT_SCHEMA = "boxfusion.tr3d_c1_track_observer_export.v1"
CONFIG_SCHEMA = "boxfusion.tr3d_c1_track_config.v1"


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _code_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        _ROOT / "boxfusion" / "tr3d_c1_track_cache.py",
        Path(__file__).resolve(),
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _config() -> dict[str, Any]:
    return {
        "schema": CONFIG_SCHEMA,
        "observer_only": True,
        "mutation_enabled": False,
        "applied_count": 0,
        "track_scope": TRACK_SCOPE,
        "cross_prefix_tracking": False,
        "residual_anchor_iou_max": RESIDUAL_ANCHOR_IOU_MAX,
        "min_view_samples": MIN_VIEW_SAMPLES,
        "supportive_view_support_min": SUPPORTIVE_VIEW_SUPPORT_MIN,
        "supportive_view_free_max": SUPPORTIVE_VIEW_FREE_MAX,
        "feature_cosine_min": FEATURE_COSINE_MIN,
        "gate_names": list(GATE_NAMES),
        "candidate_geometry_source": "immutable_class_agnostic_tr3d_p100",
        "anchor_source": "frozen_r3_active_predictions",
        "semantics": "unchanged_not_accessed",
    }


def _load_json(path: Path, schema: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != schema:
        raise ValueError(f"{path}: unsupported schema")
    if not payload.get("observer_only") or payload.get("mutation_enabled"):
        raise ValueError(f"{path}: observer contract violation")
    if int(payload.get("applied_count", -1)) != 0:
        raise ValueError(f"{path}: applied_count must be zero")
    return payload


def _tree_snapshot(root: Path, scenes: Sequence[str]) -> dict[str, Any]:
    files = {}
    for scene in scenes:
        path = root / f"{scene}_boxes.pkl"
        if not path.is_file():
            raise FileNotFoundError(path)
        files[path.name] = sha256_file(path)
    return {
        "root": str(root.resolve()),
        "files": files,
        "tree_sha256": canonical_json_sha256(files),
    }


def _scalar_from_npz(path: Path, name: str) -> str:
    with np.load(path, allow_pickle=False) as archive:
        value = np.asarray(archive[name])
        if value.shape != () or value.dtype.hasobject:
            raise ValueError(f"{path}:{name} must be a scalar")
        return str(value.item())


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
        raise FileExistsError(f"immutable C1 report exists: {path}") from error
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def export(args: argparse.Namespace) -> dict[str, Any]:
    scenes = read_scene_list(args.scene_list.resolve())
    active_root = args.active_prediction_root.resolve()
    before = _tree_snapshot(active_root, scenes)
    r2a_report = _load_json(args.r2a_export_report.resolve(), "boxfusion.tr3d_r2a_observer_export.v1")
    r2b_report = _load_json(args.r2b_export_report.resolve(), "boxfusion.tr3d_r2b_feature_observer_export.v1")
    if str(Path(r2a_report["parent_cache_root"]).resolve()) != str(args.parent_cache_root.resolve()):
        raise ValueError("R2a report belongs to a different parent cache")
    if str(Path(r2b_report["r2a_cache_root"]).resolve()) != str(args.r2a_cache_root.resolve()):
        raise ValueError("R2b report belongs to a different R2a cache")
    r2a_config_sha = str(r2a_report["r2_config_sha256"])
    r2a_code_sha = str(r2a_report["r2_code_sha256"])
    feature_checkpoint_sha = str(r2b_report["feature_checkpoint_sha256"])
    feature_config_sha = str(r2b_report["feature_config_sha256"])
    feature_code_sha = str(r2b_report["feature_code_sha256"])
    config = _config()
    config_sha = canonical_json_sha256(config)
    code_sha = _code_hash()
    rows: list[dict[str, Any]] = []
    total_parent = total_residual = total_valid_views = 0
    total_gates = np.zeros(len(GATE_NAMES), dtype=np.int64)
    total_runtime = 0.0

    for scene_id in scenes:
        started = time.perf_counter()
        parent_path = tr3d_residual_cache_path(args.parent_cache_root.resolve(), scene_id, args.prefix_id)
        parent = load_tr3d_residual_cache(
            parent_path,
            expected_scene_id=scene_id,
            expected_prefix_id=args.prefix_id,
            expected_checkpoint_sha256=args.expected_parent_checkpoint_sha256,
            expected_config_sha256=args.expected_parent_config_sha256,
        )
        r2a_path = tr3d_r2_cache_path(args.r2a_cache_root.resolve(), scene_id, args.prefix_id)
        prefix_row_sha = _scalar_from_npz(r2a_path, "prefix_manifest_row_sha256")
        frame_tree_sha = _scalar_from_npz(r2a_path, "frame_artifact_tree_sha256")
        r2a = load_tr3d_r2_cache(
            r2a_path, parent_cache_path=parent_path,
            expected_prefix_manifest_row_sha256=prefix_row_sha,
            expected_frame_artifact_tree_sha256=frame_tree_sha,
            expected_r2_config_sha256=r2a_config_sha,
            expected_r2_code_sha256=r2a_code_sha,
            expected_scene_id=scene_id, expected_prefix_id=args.prefix_id,
            expected_prefix_fraction=parent.prefix_fraction,
        )
        r2b_path = tr3d_r2b_cache_path(args.r2b_cache_root.resolve(), scene_id, args.prefix_id)
        r2b = load_tr3d_r2b_cache(
            r2b_path, parent_r2a_cache_path=r2a_path,
            parent_tr3d_cache_path=parent_path,
            expected_parent_prefix_manifest_row_sha256=prefix_row_sha,
            expected_parent_frame_artifact_tree_sha256=frame_tree_sha,
            expected_parent_r2_config_sha256=r2a_config_sha,
            expected_parent_r2_code_sha256=r2a_code_sha,
            expected_feature_checkpoint_sha256=feature_checkpoint_sha,
            expected_feature_config_sha256=feature_config_sha,
            expected_feature_code_sha256=feature_code_sha,
            expected_scene_id=scene_id, expected_prefix_id=args.prefix_id,
            expected_prefix_fraction=parent.prefix_fraction,
        )
        if not (
            np.array_equal(parent.proposal_ids, r2a.proposal_ids)
            and np.array_equal(r2a.proposal_ids, r2b.proposal_ids)
            and np.array_equal(r2a.topk_frame_ids, r2b.topk_frame_ids)
        ):
            raise ValueError(f"{scene_id}: parent/R2 row identity mismatch")

        alignment = _alignment(args.scans_root.resolve(), scene_id)
        _validate_alignment_provenance(scene_id, alignment, parent.aligned_to_unaligned)
        candidate_boxes = _minmax(_transform(parent.corners_world, alignment))
        anchor_path = active_root / f"{scene_id}_boxes.pkl"
        anchor_corners, _ = _load_b6(anchor_path)
        anchor_boxes = _minmax(_transform(anchor_corners, alignment))
        candidate_anchor_iou = pairwise_iou(candidate_boxes, anchor_boxes)
        max_anchor = (
            candidate_anchor_iou.max(axis=1)
            if candidate_anchor_iou.shape[1]
            else np.zeros(parent.proposal_count, dtype=np.float64)
        )
        residual_mask = max_anchor <= RESIDUAL_ANCHOR_IOU_MAX
        selected = np.flatnonzero(residual_mask).astype(np.int64)
        derived = derive_track_features(
            tr3d_score=parent.scores_3d[selected],
            topk_frame_ids=r2a.topk_frame_ids[selected],
            topk_view_valid=r2a.topk_view_valid[selected],
            per_view_depth_counts=r2a.per_view_depth_counts[selected],
            aggregate_depth_evidence=r2a.aggregate_depth_evidence[selected],
            per_view_feature_valid=r2b.per_view_feature_valid[selected],
            pairwise_cosine_count=r2b.pairwise_cosine_count[selected],
            pairwise_cosine_mean=r2b.pairwise_cosine_mean[selected],
        )
        elapsed = time.perf_counter() - started
        cache = TR3DC1TrackCache(
            scene_id=scene_id, prefix_id=args.prefix_id,
            parent_cache_sha256=sha256_file(parent_path),
            r2a_cache_sha256=sha256_file(r2a_path),
            r2b_cache_sha256=sha256_file(r2b_path),
            anchor_prediction_sha256=sha256_file(anchor_path),
            config_sha256=config_sha, code_sha256=code_sha,
            proposal_ids=parent.proposal_ids[selected], parent_rows=selected,
            max_anchor_iou=max_anchor[selected].astype(np.float32),
            tr3d_score=parent.scores_3d[selected],
            topk_frame_ids=r2a.topk_frame_ids[selected],
            topk_view_valid=r2a.topk_view_valid[selected],
            per_view_sample_count=r2a.per_view_point_count[selected],
            aggregate_depth_evidence=r2a.aggregate_depth_evidence[selected],
            aggregate_point_count=r2a.aggregate_point_count[selected],
            feature_pair_count=r2b.pairwise_cosine_count[selected],
            feature_pair_cosine_mean=r2b.pairwise_cosine_mean[selected],
            runtime_s=elapsed,
            **derived,
        )
        target = sidecar_path(args.output_root.resolve(), scene_id, args.prefix_id)
        sidecar_sha = write_sidecar(target, cache)
        gate_counts = cache.gate_mask.sum(axis=0, dtype=np.int64)
        rows.append({
            "scene_id": scene_id,
            "parent_proposals": parent.proposal_count,
            "unmatched_tracks": cache.track_count,
            "selected_views": int(cache.valid_view_count.sum()),
            "gate_counts": {name: int(gate_counts[i]) for i, name in enumerate(GATE_NAMES)},
            "sidecar": str(target), "sidecar_sha256": sidecar_sha,
            "runtime_s": elapsed,
        })
        total_parent += parent.proposal_count
        total_residual += cache.track_count
        total_valid_views += int(cache.valid_view_count.sum())
        total_gates += gate_counts
        total_runtime += elapsed

    after = _tree_snapshot(active_root, scenes)
    if before != after:
        raise RuntimeError("frozen R3-active prediction tree changed during C1 export")
    return {
        "schema": REPORT_SCHEMA,
        "observer_only": True, "mutation_enabled": False, "applied_count": 0,
        "ground_truth_access": False, "clip_access": False,
        "clip_semantics_unchanged": True,
        "track_scope": TRACK_SCOPE, "cross_prefix_tracking": False,
        "scene_list": str(args.scene_list.resolve()), "scene_count": len(scenes),
        "active_prediction_root": str(active_root),
        "parent_cache_root": str(args.parent_cache_root.resolve()),
        "r2a_cache_root": str(args.r2a_cache_root.resolve()),
        "r2b_cache_root": str(args.r2b_cache_root.resolve()),
        "output_root": str(args.output_root.resolve()), "prefix_id": args.prefix_id,
        "config": config, "config_sha256": config_sha, "code_sha256": code_sha,
        "input_hashes": {
            "scene_list_sha256": sha256_file(args.scene_list.resolve()),
            "r2a_export_report_sha256": sha256_file(args.r2a_export_report.resolve()),
            "r2b_export_report_sha256": sha256_file(args.r2b_export_report.resolve()),
            "frozen_active_tree_sha256": before["tree_sha256"],
        },
        "frozen_active_before": before, "frozen_active_after": after,
        "counts": {
            "parent_proposals": total_parent,
            "unmatched_tracks": total_residual,
            "selected_views": total_valid_views,
            "gates": {name: int(total_gates[i]) for i, name in enumerate(GATE_NAMES)},
        },
        "runtime_s": total_runtime, "scenes": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-cache-root", type=Path, required=True)
    parser.add_argument("--r2a-cache-root", type=Path, required=True)
    parser.add_argument("--r2b-cache-root", type=Path, required=True)
    parser.add_argument("--r2a-export-report", type=Path, required=True)
    parser.add_argument("--r2b-export-report", type=Path, required=True)
    parser.add_argument("--active-prediction-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--scans-root", type=Path, default=Path("/extra/ZhaoX/scannet_data/scans"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prefix-id", default="p100")
    parser.add_argument("--expected-parent-checkpoint-sha256", required=True)
    parser.add_argument("--expected-parent-config-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = export(args)
    _write_json_create_only(args.report.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
