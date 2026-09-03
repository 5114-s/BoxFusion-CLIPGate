#!/usr/bin/env python3
"""Audit R2a evidence without activating or tuning a validation-time gate.

The fixed gate rows are diagnostics only.  They measure whether real-depth
support/free-space evidence separates useful residual TR3D proposals.  No
row is applied to BoxFusion output and no deployment threshold may be selected
from this validation audit; calibration belongs on ScanNet-train only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.frozen_anchor_manifest import (  # noqa: E402
    verify_frozen_anchor_manifest,
)
from boxfusion.tr3d_r2_cache import (  # noqa: E402
    load_tr3d_r2_cache,
    tr3d_r2_cache_path,
)
from boxfusion.tr3d_r2_provenance import (  # noqa: E402
    canonical_json_sha256,
    frame_artifact_tree,
    load_prefix_manifest,
)
from boxfusion.tr3d_residual_cache import (  # noqa: E402
    load_tr3d_residual_cache,
    tr3d_residual_cache_path,
)
from tools.audit_tr3d_residual_observer import (  # noqa: E402
    _alignment,
    _gt_boxes,
    _load_b6,
    _minmax,
    _transform,
    _validate_alignment_provenance,
    maximum_cardinality,
    pairwise_iou,
)
from tools.run_tr3d_r2_observer import (  # noqa: E402
    REPORT_SCHEMA as EXPORT_REPORT_SCHEMA,
    _code_hash,
    _load_bound_parent,
)
from tools.tr3d_data import discover_frame_bundle, read_scene_list  # noqa: E402


REPORT_SCHEMA = "boxfusion.tr3d_r2a_depth_audit.v1"
IOU_THRESHOLDS = (0.15, 0.25, 0.50)
RESIDUAL_ANCHOR_IOU = 0.15
FIXED_DEPTH_GATES = (
    ("visible", 0.00, 1.00, 1),
    ("depth_loose", 0.10, 0.75, 1),
    ("depth_medium", 0.20, 0.50, 2),
    ("depth_strict", 0.30, 0.25, 3),
    ("depth_very_strict", 0.40, 0.10, 3),
)


def _snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "anchor_name": payload["anchor_name"],
        "prediction_tree_sha256": payload["prediction_tree_sha256"],
        "artifact_tree_sha256": payload["artifact_tree_sha256"],
        "scene_list_sha256": payload["scene_list_sha256"],
    }


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0, "q10": None, "q50": None, "q90": None}
    q10, q50, q90 = np.quantile(finite, (0.10, 0.50, 0.90))
    return {
        "count": int(len(finite)),
        "q10": float(q10),
        "q50": float(q50),
        "q90": float(q90),
    }


def _maximum_recall(
    baseline_iou: np.ndarray,
    candidate_iou: np.ndarray,
    selected: np.ndarray,
    threshold: float,
) -> tuple[int, int, int]:
    anchor = maximum_cardinality(baseline_iou, threshold)
    union_iou = np.concatenate(
        (baseline_iou, candidate_iou[selected]), axis=0
    )
    union = maximum_cardinality(union_iou, threshold)
    return anchor, union, union - anchor


def _load_export_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != EXPORT_REPORT_SCHEMA:
        raise ValueError("unsupported R2a export report schema")
    if not report.get("observer_only") or report.get("mutation_enabled"):
        raise ValueError("R2a export report violates observer contract")
    if int(report.get("applied_count", -1)) != 0:
        raise ValueError("R2a export report applied_count must be zero")
    if report.get("ground_truth_access") or report.get("clip_enabled"):
        raise ValueError("R2a export improperly accessed GT or CLIP")
    config = report.get("r2_config")
    if not isinstance(config, dict) or canonical_json_sha256(config) != report.get(
        "r2_config_sha256"
    ):
        raise ValueError("R2a export config hash mismatch")
    if _code_hash() != report.get("r2_code_sha256"):
        raise ValueError("current R2a observer code differs from export code")
    return report


def audit(args: argparse.Namespace) -> dict[str, Any]:
    before = verify_frozen_anchor_manifest(args.frozen_manifest.resolve())
    before_snapshot = _snapshot(before)
    scenes = read_scene_list(args.scene_list.resolve())
    export_report = _load_export_report(args.r2_export_report.resolve())
    if export_report.get("scene_count") != len(scenes):
        raise ValueError("R2a export report scene count mismatch")
    if Path(export_report["parent_cache_root"]).resolve() != args.parent_cache_root.resolve():
        raise ValueError("R2a export parent cache root mismatch")
    if Path(export_report["r2_cache_root"]).resolve() != args.r2_cache_root.resolve():
        raise ValueError("R2a export sidecar root mismatch")
    exact_paths = {
        "prefix_manifest": args.prefix_manifest,
        "frames_root": args.frames_root,
        "scene_list": args.scene_list,
    }
    for name, expected in exact_paths.items():
        if Path(str(export_report.get(name, ""))).resolve() != expected.resolve():
            raise ValueError(f"R2a export {name} mismatch")
    if export_report.get("prefix_id") != args.prefix_id:
        raise ValueError("R2a export prefix id mismatch")
    exported_scenes = [
        str(row.get("scene_id")) for row in export_report.get("scenes", [])
    ]
    if exported_scenes != scenes:
        raise ValueError("R2a export ordered scene set mismatch")
    manifest = load_prefix_manifest(
        args.prefix_manifest.resolve(), prefix_id=args.prefix_id
    )
    frozen_root = Path(before["reference_result_root"]).resolve()
    config_sha = str(export_report["r2_config_sha256"])
    code_sha = str(export_report["r2_code_sha256"])

    totals = {
        "gt": 0,
        "anchor": 0,
        "candidate": 0,
        "residual": 0,
    }
    feature_names = ("support", "occluded", "free_space", "invalid")
    positive_features = {name: [] for name in feature_names}
    negative_features = {name: [] for name in positive_features}
    per_scene: list[dict[str, Any]] = []

    # Maximum-cardinality matching is scene-local.  Store per-scene matrices
    # and sum later instead of concatenating unrelated scene GT columns.
    scene_matrices: list[dict[str, Any]] = []
    for scene_id in scenes:
        if scene_id not in manifest:
            raise ValueError(f"prefix manifest lacks {scene_id}")
        row = manifest[scene_id]
        bundle = discover_frame_bundle(args.frames_root.resolve(), scene_id)
        frame_tree_sha, _ = frame_artifact_tree(row, bundle)
        manifest_sha = canonical_json_sha256(row)
        parent_path = tr3d_residual_cache_path(
            args.parent_cache_root.resolve(), scene_id, args.prefix_id
        )
        parent = _load_bound_parent(
            parent_path,
            row,
            args.prefix_manifest.resolve(),
            expected_scene_id=scene_id,
            expected_prefix_id=args.prefix_id,
            expected_checkpoint_sha256=(
                args.expected_parent_checkpoint_sha256
            ),
            expected_config_sha256=args.expected_parent_config_sha256,
        )
        sidecar = load_tr3d_r2_cache(
            tr3d_r2_cache_path(
                args.r2_cache_root.resolve(), scene_id, args.prefix_id
            ),
            parent_cache_path=parent_path,
            expected_prefix_manifest_row_sha256=manifest_sha,
            expected_frame_artifact_tree_sha256=frame_tree_sha,
            expected_r2_config_sha256=config_sha,
            expected_r2_code_sha256=code_sha,
            expected_scene_id=scene_id,
            expected_prefix_id=args.prefix_id,
            expected_prefix_fraction=float(row["fraction"]),
            expected_allowed_frame_ids=row["used_frame_ids"],
        )
        parent_index = {
            int(value): index
            for index, value in enumerate(parent.proposal_ids)
        }
        rows = np.asarray(
            [parent_index[int(value)] for value in sidecar.proposal_ids],
            dtype=np.int64,
        )
        transform = _alignment(args.scans_root.resolve(), scene_id)
        _validate_alignment_provenance(
            scene_id, transform, parent.aligned_to_unaligned
        )
        candidate_boxes = _minmax(
            _transform(parent.corners_world[rows], transform)
        )
        anchor_corners, _ = _load_b6(
            frozen_root / f"{scene_id}_boxes.pkl"
        )
        anchor_boxes = _minmax(_transform(anchor_corners, transform))
        gt_boxes = _gt_boxes(
            args.gt_root.resolve() / f"{scene_id}_bbox.npy"
        )
        anchor_vs_gt = pairwise_iou(anchor_boxes, gt_boxes)
        candidate_vs_gt = pairwise_iou(candidate_boxes, gt_boxes)
        candidate_vs_anchor = pairwise_iou(candidate_boxes, anchor_boxes)
        max_gt = (
            candidate_vs_gt.max(axis=1)
            if candidate_vs_gt.shape[1]
            else np.zeros(len(candidate_boxes), dtype=np.float64)
        )
        max_anchor = (
            candidate_vs_anchor.max(axis=1)
            if candidate_vs_anchor.shape[1]
            else np.zeros(len(candidate_boxes), dtype=np.float64)
        )
        residual = max_anchor <= RESIDUAL_ANCHOR_IOU
        evidence = np.asarray(sidecar.aggregate_depth_evidence)
        views = np.asarray(sidecar.aggregate_view_count)
        gates: dict[str, np.ndarray] = {}
        for name, support_min, free_max, min_views in FIXED_DEPTH_GATES:
            gates[name] = (
                residual
                & (views >= min_views)
                & (evidence[:, 0] >= support_min)
                & (evidence[:, 2] <= free_max)
            )

        positive = residual & (max_gt > 0.50)
        negative = residual & (max_gt <= 0.15)
        for index, name in enumerate(("support", "occluded", "free_space", "invalid")):
            positive_features[name].append(evidence[positive, index])
            negative_features[name].append(evidence[negative, index])

        scene_matrices.append({
            "anchor": anchor_vs_gt,
            "candidate": candidate_vs_gt,
            "gates": gates,
        })
        totals["gt"] += len(gt_boxes)
        totals["anchor"] += len(anchor_boxes)
        totals["candidate"] += len(candidate_boxes)
        totals["residual"] += int(np.count_nonzero(residual))
        per_scene.append({
            "scene_id": scene_id,
            "gt_count": len(gt_boxes),
            "anchor_count": len(anchor_boxes),
            "candidate_count": len(candidate_boxes),
            "residual_count": int(np.count_nonzero(residual)),
            "residual_tp50_independent": int(np.count_nonzero(positive)),
            "gate_counts": {
                name: int(np.count_nonzero(mask))
                for name, mask in gates.items()
            },
        })

    gate_report: dict[str, Any] = {}
    for name, support_min, free_max, min_views in FIXED_DEPTH_GATES:
        candidate_count = 0
        independent_tp50 = 0
        anchor_matches = {threshold: 0 for threshold in IOU_THRESHOLDS}
        union_matches = {threshold: 0 for threshold in IOU_THRESHOLDS}
        for item in scene_matrices:
            selected = item["gates"][name]
            candidate_count += int(np.count_nonzero(selected))
            selected_iou = item["candidate"][selected]
            independent_tp50 += int(
                np.count_nonzero(
                    selected_iou.max(axis=1) > 0.50
                ) if selected_iou.shape[1] else 0
            )
            for threshold in IOU_THRESHOLDS:
                anchor, union, _ = _maximum_recall(
                    item["anchor"], item["candidate"], selected, threshold
                )
                anchor_matches[threshold] += anchor
                union_matches[threshold] += union
        thresholds = {}
        for threshold in IOU_THRESHOLDS:
            novel = union_matches[threshold] - anchor_matches[threshold]
            thresholds[f"{threshold:.2f}"] = {
                "anchor_recall": (
                    anchor_matches[threshold] / totals["gt"]
                    if totals["gt"] else 0.0
                ),
                "union_oracle_recall": (
                    union_matches[threshold] / totals["gt"]
                    if totals["gt"] else 0.0
                ),
                "delta_recall": (
                    novel / totals["gt"] if totals["gt"] else 0.0
                ),
                "novel_oracle_tp": novel,
            }
        gate_report[name] = {
            "support_min": support_min,
            "free_space_max": free_max,
            "min_views": min_views,
            "candidate_count": candidate_count,
            "independent_tp50": independent_tp50,
            "independent_precision50_upper_bound": (
                independent_tp50 / candidate_count if candidate_count else 0.0
            ),
            "thresholds": thresholds,
        }

    separation = {}
    for name in positive_features:
        positives = np.concatenate(positive_features[name]) if positive_features[name] else np.empty(0)
        negatives = np.concatenate(negative_features[name]) if negative_features[name] else np.empty(0)
        separation[name] = {
            "positive_residual_iou50": _quantiles(positives),
            "negative_residual_iou15": _quantiles(negatives),
        }

    after = verify_frozen_anchor_manifest(args.frozen_manifest.resolve())
    after_snapshot = _snapshot(after)
    if before_snapshot != after_snapshot:
        raise RuntimeError("frozen anchor changed during R2a audit")
    return {
        "schema": REPORT_SCHEMA,
        "observer_contract": {
            "observer_only": True,
            "mutation_enabled": False,
            "applied_count": 0,
            "frozen_anchor_verified_before_and_after": True,
            "before": before_snapshot,
            "after": after_snapshot,
        },
        "purpose": (
            "fixed diagnostic gates only; calibrate on ScanNet-train and do "
            "not select an active threshold from validation"
        ),
        "anchor": {
            "name": before["anchor_name"],
            "metrics_percent": before["anchor_metrics_percent"],
        },
        "scene_count": len(scenes),
        "input_provenance": {
            "r2_export_report_canonical_sha256": canonical_json_sha256(export_report),
            "selected_prefix_rows_canonical_sha256": canonical_json_sha256(
                [manifest[scene] for scene in scenes]
            ),
            "ordered_scene_set_sha256": canonical_json_sha256(scenes),
            "parent_checkpoint_sha256": (
                args.expected_parent_checkpoint_sha256
            ),
            "parent_config_sha256": args.expected_parent_config_sha256,
            "r2_config_sha256": config_sha,
            "r2_code_sha256": code_sha,
        },
        "counts": totals,
        "residual_anchor_iou_max": RESIDUAL_ANCHOR_IOU,
        "fixed_depth_gates": gate_report,
        "evidence_separation": separation,
        "per_scene": per_scene,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--parent-cache-root", type=Path, required=True)
    parser.add_argument("--r2-cache-root", type=Path, required=True)
    parser.add_argument("--r2-export-report", type=Path, required=True)
    parser.add_argument("--prefix-manifest", type=Path, required=True)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--prefix-id", default="p100")
    parser.add_argument("--expected-parent-checkpoint-sha256", required=True)
    parser.add_argument("--expected-parent-config-sha256", required=True)
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=Path(
            "/data/ZhaoX/BoxFusion/evaluation/data_util/"
            "scannet_train_detection_data"
        ),
    )
    parser.add_argument(
        "--scans-root",
        type=Path,
        default=Path("/extra/ZhaoX/scannet_data/scans"),
    )
    parser.add_argument("--report", type=Path, required=True)
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
        raise FileExistsError(
            f"immutable R2a audit report exists: {path}"
        ) from error
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_path = args.report.resolve()
    frozen = verify_frozen_anchor_manifest(args.frozen_manifest.resolve())
    frozen_root = Path(frozen["reference_result_root"]).resolve()
    if report_path == frozen_root or frozen_root in report_path.parents:
        raise ValueError("R2a report must not be written inside frozen anchor")
    report = audit(args)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    _write_create_only(report_path, encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
