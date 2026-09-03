#!/usr/bin/env python3
"""Export immutable, observer-only TR3D anchor-near R3 sidecars.

The process reads no ground truth and never writes or reorders a frozen G0
prediction.  ScanNet ``axisAlignment`` is read only as detector/evaluator
input metadata.  R2a/R2b parents are optional; their evidence uses explicit
zero/false sentinels when absent.
"""

from __future__ import annotations

import argparse
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

from boxfusion.frozen_anchor_manifest import verify_frozen_anchor_manifest  # noqa: E402
from boxfusion.tr3d_r2_provenance import (  # noqa: E402
    canonical_json_sha256,
    code_artifact_tree_sha256,
    frame_artifact_tree,
    load_prefix_manifest,
    sha256_file,
)
from boxfusion.tr3d_r2_cache import tr3d_r2_cache_path  # noqa: E402
from boxfusion.tr3d_r2b_cache import tr3d_r2b_cache_path  # noqa: E402
from boxfusion.tr3d_r3_cache import (  # noqa: E402
    load_tr3d_r3_cache,
    make_tr3d_r3_cache,
    tr3d_r3_cache_path,
    write_tr3d_r3_cache,
)
from boxfusion.tr3d_r3_observer import (  # noqa: E402
    TR3D_R3_NEAR_ANCHOR_IOU,
    load_axis_alignment_input_metadata,
    load_frozen_anchor_prediction,
)
from boxfusion.tr3d_residual_cache import tr3d_residual_cache_path  # noqa: E402
from tools.run_tr3d_r2_observer import _load_bound_parent  # noqa: E402
from tools.tr3d_data import discover_frame_bundle, read_scene_list  # noqa: E402


REPORT_SCHEMA = "boxfusion.tr3d_r3_anchor_near_observer_export.v1"
CONFIG_SCHEMA = "boxfusion.tr3d_r3_anchor_near_config.v1"
R2A_REPORT_SCHEMA = "boxfusion.tr3d_r2a_observer_export.v1"
R2B_REPORT_SCHEMA = "boxfusion.tr3d_r2b_feature_observer_export.v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--parent-cache-root", type=Path, required=True)
    parser.add_argument("--prefix-manifest", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--scans-root", type=Path, required=True)
    parser.add_argument("--r3-cache-root", type=Path, required=True)
    parser.add_argument("--prefix-id", default="p100")
    parser.add_argument("--expected-parent-checkpoint-sha256", required=True)
    parser.add_argument("--expected-parent-config-sha256", required=True)
    parser.add_argument("--r2a-cache-root", type=Path)
    parser.add_argument("--r2a-export-report", type=Path)
    parser.add_argument("--frames-root", type=Path)
    parser.add_argument("--r2b-cache-root", type=Path)
    parser.add_argument("--r2b-export-report", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser


def _write_create_only(path: Path, encoded: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"immutable R3 report exists: {path}") from error
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _load_json(path: Path, expected_schema: str, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != expected_schema:
        raise ValueError(f"unsupported {label} schema")
    if not payload.get("observer_only") or payload.get("mutation_enabled"):
        raise ValueError(f"{label} violates observer-only contract")
    if int(payload.get("applied_count", -1)) != 0:
        raise ValueError(f"{label} applied_count must be zero")
    if payload.get("ground_truth_access"):
        raise ValueError(f"{label} accessed forbidden ground truth")
    return payload


def _optional_parent_contract(
    args: argparse.Namespace, scenes: list[str]
) -> tuple[dict[str, str], dict[str, Any]]:
    r2a_enabled = args.r2a_cache_root is not None
    r2b_enabled = args.r2b_cache_root is not None
    if r2b_enabled and not r2a_enabled:
        raise ValueError("R2b evidence requires the exact R2a parent")
    if r2a_enabled != (args.r2a_export_report is not None):
        raise ValueError("R2a cache root and export report must be supplied together")
    if r2a_enabled != (args.frames_root is not None):
        raise ValueError("R2a evidence requires frames-root for artifact verification")
    if r2b_enabled != (args.r2b_export_report is not None):
        raise ValueError("R2b cache root and export report must be supplied together")
    hashes = {
        "r2_config_sha256": "",
        "r2_code_sha256": "",
        "feature_checkpoint_sha256": "",
        "feature_config_sha256": "",
        "feature_code_sha256": "",
    }
    input_reports: dict[str, Any] = {
        "r2a_enabled": r2a_enabled,
        "r2b_enabled": r2b_enabled,
    }
    if r2a_enabled:
        r2a = _load_json(args.r2a_export_report.resolve(), R2A_REPORT_SCHEMA, "R2a report")
        ordered = [str(row.get("scene_id")) for row in r2a.get("scenes", [])]
        if ordered != scenes or int(r2a.get("scene_count", -1)) != len(scenes):
            raise ValueError("R2a report ordered scene set mismatch")
        paths = {
            "parent_cache_root": args.parent_cache_root,
            "r2_cache_root": args.r2a_cache_root,
            "prefix_manifest": args.prefix_manifest,
            "frames_root": args.frames_root,
            "scene_list": args.scene_list,
        }
        for name, expected in paths.items():
            if Path(str(r2a.get(name, ""))).resolve() != expected.resolve():
                raise ValueError(f"R2a report {name} mismatch")
        if r2a.get("prefix_id") != args.prefix_id:
            raise ValueError("R2a report prefix mismatch")
        if canonical_json_sha256(r2a.get("r2_config")) != r2a.get("r2_config_sha256"):
            raise ValueError("R2a report config hash mismatch")
        hashes["r2_config_sha256"] = str(r2a["r2_config_sha256"])
        hashes["r2_code_sha256"] = str(r2a["r2_code_sha256"])
        input_reports["r2a_export_report"] = str(args.r2a_export_report.resolve())
        input_reports["r2a_export_report_sha256"] = sha256_file(args.r2a_export_report.resolve())
    if r2b_enabled:
        r2b = _load_json(args.r2b_export_report.resolve(), R2B_REPORT_SCHEMA, "R2b report")
        ordered = [str(row.get("scene_id")) for row in r2b.get("scenes", [])]
        if ordered != scenes or int(r2b.get("scene_count", -1)) != len(scenes):
            raise ValueError("R2b report ordered scene set mismatch")
        paths = {
            "parent_cache_root": args.parent_cache_root,
            "r2a_cache_root": args.r2a_cache_root,
            "r2b_cache_root": args.r2b_cache_root,
            "prefix_manifest": args.prefix_manifest,
            "frames_root": args.frames_root,
            "scene_list": args.scene_list,
            "r2a_export_report": args.r2a_export_report,
        }
        for name, expected in paths.items():
            if Path(str(r2b.get(name, ""))).resolve() != expected.resolve():
                raise ValueError(f"R2b report {name} mismatch")
        if r2b.get("prefix_id") != args.prefix_id:
            raise ValueError("R2b report prefix mismatch")
        if str(r2b.get("parent_r2_config_sha256")) != hashes["r2_config_sha256"] or str(
            r2b.get("parent_r2_code_sha256")
        ) != hashes["r2_code_sha256"]:
            raise ValueError("R2b report R2a provenance mismatch")
        if canonical_json_sha256(r2b.get("feature_config")) != r2b.get("feature_config_sha256"):
            raise ValueError("R2b feature config hash mismatch")
        hashes["feature_checkpoint_sha256"] = str(r2b["feature_checkpoint_sha256"])
        hashes["feature_config_sha256"] = str(r2b["feature_config_sha256"])
        hashes["feature_code_sha256"] = str(r2b["feature_code_sha256"])
        input_reports["r2b_export_report"] = str(args.r2b_export_report.resolve())
        input_reports["r2b_export_report_sha256"] = sha256_file(args.r2b_export_report.resolve())
    return hashes, input_reports


def _config(r2a_enabled: bool, r2b_enabled: bool) -> dict[str, Any]:
    return {
        "schema": CONFIG_SCHEMA,
        "observer_only": True,
        "mutation_enabled": False,
        "applied_count": 0,
        "ground_truth_access": False,
        "clip_access": False,
        "clip_semantics_unchanged": True,
        "association": "maximum_axis_aligned_aabb_iou_then_nearest_center_then_stable_index",
        "association_frame": "scannet_axis_aligned_input_metadata",
        "axis_alignment_is_ground_truth": False,
        "near_anchor_iou_operator": ">",
        "near_anchor_iou": TR3D_R3_NEAR_ANCHOR_IOU,
        "proposal_geometry": "exact_tr3d_parent_corners_unaligned_world",
        "anchor_geometry": "exact_manifest_pinned_g0_prediction",
        "r2a_optional": True,
        "r2b_optional": True,
        "r2a_enabled": r2a_enabled,
        "r2b_enabled": r2b_enabled,
        "missing_evidence_sentinel": "zero_values_and_available_false",
        "output_policy": "diagnostic_sidecar_only_no_prediction_mutation",
    }


def _code_hash() -> str:
    return code_artifact_tree_sha256(
        (
            _ROOT / "boxfusion" / "tr3d_r3_observer.py",
            _ROOT / "boxfusion" / "tr3d_r3_cache.py",
            Path(__file__),
        )
    )


def export(args: argparse.Namespace) -> dict[str, Any]:
    scenes = read_scene_list(args.scene_list.resolve())
    prefix_rows = load_prefix_manifest(args.prefix_manifest.resolve(), prefix_id=args.prefix_id)
    if any(scene not in prefix_rows for scene in scenes):
        raise ValueError("prefix manifest is missing a requested R3 scene")
    anchor_before = verify_frozen_anchor_manifest(args.frozen_manifest.resolve())
    if not set(scenes).issubset(set(anchor_before["scene_ids"])):
        raise ValueError("R3 scene list is not a frozen-anchor subset")
    hashes, input_reports = _optional_parent_contract(args, scenes)
    config = _config(input_reports["r2a_enabled"], input_reports["r2b_enabled"])
    config_sha = canonical_json_sha256(config)
    code_sha = _code_hash()
    frozen_root = Path(anchor_before["reference_result_root"]).resolve()
    scene_reports: list[dict[str, Any]] = []
    total_near = total_parent = resumed = 0
    total_wall = 0.0

    for position, scene_id in enumerate(scenes, start=1):
        started = time.perf_counter()
        row = prefix_rows[scene_id]
        parent_path = tr3d_residual_cache_path(args.parent_cache_root.resolve(), scene_id, args.prefix_id)
        parent = _load_bound_parent(
            parent_path,
            row,
            args.prefix_manifest.resolve(),
            expected_scene_id=scene_id,
            expected_prefix_id=args.prefix_id,
            expected_checkpoint_sha256=args.expected_parent_checkpoint_sha256,
            expected_config_sha256=args.expected_parent_config_sha256,
        )
        anchor_path = frozen_root / f"{scene_id}_boxes.pkl"
        anchor_corners, anchor_scores = load_frozen_anchor_prediction(anchor_path)
        metadata_path = args.scans_root.resolve() / scene_id / f"{scene_id}.txt"
        alignment = load_axis_alignment_input_metadata(metadata_path)
        r2a_path = None
        r2b_path = None
        manifest_sha = frame_sha = ""
        if input_reports["r2a_enabled"]:
            manifest_sha = canonical_json_sha256(row)
            bundle = discover_frame_bundle(args.frames_root.resolve(), scene_id)
            frame_sha, _ = frame_artifact_tree(row, bundle)
            r2a_path = tr3d_r2_cache_path(args.r2a_cache_root.resolve(), scene_id, args.prefix_id)
        if input_reports["r2b_enabled"]:
            r2b_path = tr3d_r2b_cache_path(args.r2b_cache_root.resolve(), scene_id, args.prefix_id)
        target = tr3d_r3_cache_path(args.r3_cache_root.resolve(), scene_id, args.prefix_id)
        contract = {
            "parent_tr3d_cache_path": parent_path,
            "frozen_anchor_manifest_path": args.frozen_manifest.resolve(),
            "anchor_prediction_path": anchor_path,
            "anchor_corners_world": anchor_corners,
            "anchor_scores": anchor_scores,
            "axis_alignment_metadata_path": metadata_path,
            "expected_checkpoint_sha256": args.expected_parent_checkpoint_sha256,
            "expected_config_sha256": args.expected_parent_config_sha256,
            "expected_r3_config_sha256": config_sha,
            "expected_r3_code_sha256": code_sha,
            "parent_r2a_cache_path": r2a_path,
            "parent_r2b_cache_path": r2b_path,
            "expected_prefix_manifest_row_sha256": manifest_sha,
            "expected_frame_artifact_tree_sha256": frame_sha,
            "expected_r2_config_sha256": hashes["r2_config_sha256"],
            "expected_r2_code_sha256": hashes["r2_code_sha256"],
            "expected_feature_checkpoint_sha256": hashes["feature_checkpoint_sha256"],
            "expected_feature_config_sha256": hashes["feature_config_sha256"],
            "expected_feature_code_sha256": hashes["feature_code_sha256"],
            "expected_scene_id": scene_id,
            "expected_prefix_id": args.prefix_id,
        }
        was_resumed = target.exists()
        if was_resumed:
            if not args.resume:
                raise FileExistsError(f"immutable R3 cache exists: {target}")
            cache = load_tr3d_r3_cache(target, **contract)
            resumed += 1
        else:
            cache = make_tr3d_r3_cache(
                parent_tr3d_cache_path=parent_path,
                frozen_anchor_manifest_path=args.frozen_manifest.resolve(),
                anchor_prediction_path=anchor_path,
                anchor_corners_world=anchor_corners,
                anchor_scores=anchor_scores,
                axis_alignment_metadata_path=metadata_path,
                axis_alignment=alignment,
                expected_checkpoint_sha256=args.expected_parent_checkpoint_sha256,
                expected_config_sha256=args.expected_parent_config_sha256,
                r3_config_sha256=config_sha,
                r3_code_sha256=code_sha,
                parent_r2a_cache_path=r2a_path,
                parent_r2b_cache_path=r2b_path,
                prefix_manifest_row_sha256=manifest_sha,
                frame_artifact_tree_sha256=frame_sha,
                r2_config_sha256=hashes["r2_config_sha256"],
                r2_code_sha256=hashes["r2_code_sha256"],
                feature_checkpoint_sha256=hashes["feature_checkpoint_sha256"],
                feature_config_sha256=hashes["feature_config_sha256"],
                feature_code_sha256=hashes["feature_code_sha256"],
            )
            write_tr3d_r3_cache(target, cache, **contract)
        elapsed = time.perf_counter() - started
        total_wall += elapsed
        total_parent += parent.proposal_count
        total_near += cache.proposal_count
        scene_reports.append(
            {
                "scene_id": scene_id,
                "parent_proposal_count": parent.proposal_count,
                "anchor_count": cache.anchor_count,
                "near_candidate_count": cache.proposal_count,
                "parent_r2a_available": cache.parent_r2a_available,
                "parent_r2b_available": cache.parent_r2b_available,
                "r2a_evidence_count": int(np.count_nonzero(cache.r2a_evidence_available)),
                "r2b_feature_count": int(np.count_nonzero(cache.r2b_feature_available)),
                "r2b_multiview_count": int(np.count_nonzero(cache.r2b_multiview_available)),
                "resumed": was_resumed,
                "wall_s": float(elapsed),
                "r3_sidecar": str(target),
                "r3_sidecar_sha256": sha256_file(target),
            }
        )
        print(
            f"[{position}/{len(scenes)}] {scene_id}: parent={parent.proposal_count}, "
            f"near={cache.proposal_count}, anchors={cache.anchor_count}, wall={elapsed:.3f}s",
            flush=True,
        )

    anchor_after = verify_frozen_anchor_manifest(args.frozen_manifest.resolve())
    frozen_keys = ("prediction_tree_sha256", "artifact_tree_sha256", "scene_list_sha256")
    if any(anchor_before[key] != anchor_after[key] for key in frozen_keys):
        raise RuntimeError("frozen G0 anchor changed during R3 observation")
    return {
        "schema": REPORT_SCHEMA,
        "observer_only": True,
        "mutation_enabled": False,
        "applied_count": 0,
        "ground_truth_access": False,
        "axis_alignment_input_metadata_access": True,
        "clip_access": False,
        "clip_semantics_unchanged": True,
        "frozen_anchor_verified_before_and_after": True,
        "frozen_manifest": str(args.frozen_manifest.resolve()),
        "frozen_manifest_sha256": sha256_file(args.frozen_manifest.resolve()),
        "frozen_prediction_tree_sha256": anchor_before["prediction_tree_sha256"],
        "parent_cache_root": str(args.parent_cache_root.resolve()),
        "prefix_manifest": str(args.prefix_manifest.resolve()),
        "scene_list": str(args.scene_list.resolve()),
        "scans_root": str(args.scans_root.resolve()),
        "r3_cache_root": str(args.r3_cache_root.resolve()),
        "r2a_cache_root": str(args.r2a_cache_root.resolve()) if args.r2a_cache_root else None,
        "r2b_cache_root": str(args.r2b_cache_root.resolve()) if args.r2b_cache_root else None,
        "frames_root": str(args.frames_root.resolve()) if args.frames_root else None,
        "prefix_id": args.prefix_id,
        "expected_parent_checkpoint_sha256": args.expected_parent_checkpoint_sha256,
        "expected_parent_config_sha256": args.expected_parent_config_sha256,
        "input_reports": input_reports,
        "parent_evidence_hashes": hashes,
        "r3_config": config,
        "r3_config_sha256": config_sha,
        "r3_code_sha256": code_sha,
        "scene_count": len(scenes),
        "parent_proposal_count": total_parent,
        "near_candidate_count": total_near,
        "resumed_scene_count": resumed,
        "summed_wall_s": float(total_wall),
        "scenes": scene_reports,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = export(args)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        _write_create_only(args.report.resolve(), encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
