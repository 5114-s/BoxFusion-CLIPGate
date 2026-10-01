#!/usr/bin/env python3
"""Post-hoc ScanNet oracle for one sealed Boxer-Past3 shadow sidecar.

Unlike the shadow materializer, this tool intentionally reads ground truth.
It never selects or changes candidates: it evaluates the already fixed sidecar
as a suffix after the byte-identical native T05 prefix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(_REPOSITORY_ROOT))

from tools.audit_scannet_boxer_unexplained_oracle import (
    aligned_iou_matrix,
    load_axis_alignment,
    load_baseline_boxes,
    load_gt_minmax,
    official_constant_evaluate,
    strict_maximum_matching,
)


SCHEMA = "boxfusion.scannet_boxer_past3_oracle.v1"
SHADOW_SCHEMA = "boxfusion.boxer_past3_shadow.v1"
DEPTH_SHADOW_SCHEMA = "boxfusion.boxer_past3_depth_shadow.v1"
THRESHOLDS = (0.15, 0.25, 0.50)
SCENE_PATTERN = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")

_EXPECTED_ARRAYS = {
    "candidate_appended_score_diagnostic_only",
    "candidate_center_rms_m",
    "candidate_confirmation_frame_id",
    "candidate_corners_world",
    "candidate_evidence_offsets",
    "candidate_max_camera_baseline_m",
    "candidate_max_terminal_native_aabb_iou",
    "candidate_max_view_ray_span_deg",
    "candidate_median_pairwise_aabb_iou",
    "candidate_raw_mean_score",
    "candidate_scene_index",
    "candidate_track_id",
    "evidence_frame_id",
    "evidence_source_row",
    "scene_ids",
}

_EXPECTED_DEPTH_ARRAYS = {
    "candidate_appended_score_diagnostic_only",
    "candidate_center_rms_m",
    "candidate_confirmation_frame_id",
    "candidate_corners_world",
    "candidate_depth_node_offsets",
    "candidate_max_terminal_native_aabb_iou",
    "candidate_median_pairwise_aabb_iou",
    "candidate_qualification_frame_id",
    "candidate_raw_mean_score",
    "candidate_receipt_evidence_offsets",
    "candidate_scene_index",
    "candidate_support_edge_offsets",
    "candidate_track_id",
    "depth_node_frame_id",
    "depth_node_guide_count",
    "depth_node_source_row",
    "receipt_evidence_frame_id",
    "receipt_evidence_source_row",
    "scene_ids",
    "support_edge_camera_baseline_m",
    "support_edge_source_frame_id",
    "support_edge_target_frame_id",
    "support_edge_vb",
    "support_edge_vf",
    "support_edge_view_ray_span_deg",
}


class Past3OracleError(ValueError):
    """Raised when a sealed input or evaluation contract is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise Past3OracleError(f"{label} must be a regular non-symlink file: {path}")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Past3OracleError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise Past3OracleError(f"{label} must contain an object: {path}")
    return value


def _load_shadow(
    json_path: Path, npz_path: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray], tuple[str, ...]]:
    manifest = _read_json(json_path, "Boxer-Past3 shadow manifest")
    required = {
        "mode": "shadow",
        "output_inert": True,
        "birth": False,
        "active_authorized": False,
        "native_mutation_applied": False,
        "gt_access": False,
        "training_free": True,
        "online_learning": False,
        "future_frames_used": False,
        "native_clip_access": False,
        "native_clip_unchanged": True,
        "score_mode_for_formal_evaluation": "constant_1.0",
        "coordinate_frame": "scannet_world",
        "native_before_after_identity": True,
    }
    schema = manifest.get("schema")
    if schema == SHADOW_SCHEMA:
        required.update(
            {
                "past_only_association": True,
                "tracked_boxer_pool_used": False,
            }
        )
        expected_arrays = _EXPECTED_ARRAYS
    elif schema == DEPTH_SHADOW_SCHEMA:
        required.update(
            {
                "past_only": True,
                "receipt_geometry_frozen": True,
                "receipt_provenance_frozen": True,
                "later_evidence_changes_receipt": False,
                "detector_semantics_used": False,
            }
        )
        expected_arrays = _EXPECTED_DEPTH_ARRAYS
    else:
        raise Past3OracleError(f"unsupported Boxer-Past3 shadow schema: {schema!r}")
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise Past3OracleError(
                f"shadow contract mismatch for {key}: "
                f"expected={expected!r}, actual={manifest.get(key)!r}"
            )
    _regular_file(npz_path, "Boxer-Past3 shadow NPZ")
    if manifest.get("npz_file") != npz_path.name:
        raise Past3OracleError("shadow NPZ filename mismatch")
    if manifest.get("npz_sha256") != _sha256(npz_path):
        raise Past3OracleError("shadow NPZ SHA-256 mismatch")
    try:
        with np.load(npz_path, allow_pickle=False) as source:
            if set(source.files) != expected_arrays:
                raise Past3OracleError("unexpected Boxer-Past3 NPZ schema")
            arrays = {name: np.array(source[name], copy=True) for name in source.files}
    except (OSError, ValueError) as error:
        if isinstance(error, Past3OracleError):
            raise
        raise Past3OracleError(f"invalid shadow NPZ: {npz_path}") from error
    scenes = tuple(str(value) for value in arrays["scene_ids"].tolist())
    if (
        not scenes
        or len(set(scenes)) != len(scenes)
        or any(SCENE_PATTERN.fullmatch(scene) is None for scene in scenes)
        or manifest.get("scene_count") != len(scenes)
    ):
        raise Past3OracleError("invalid Boxer-Past3 scene order")
    count = int(manifest.get("candidate_count", -1))
    shapes = {
        "candidate_scene_index": (count,),
        "candidate_track_id": (count,),
        "candidate_confirmation_frame_id": (count,),
        "candidate_corners_world": (count, 8, 3),
        "candidate_raw_mean_score": (count,),
        "candidate_appended_score_diagnostic_only": (count,),
        "candidate_median_pairwise_aabb_iou": (count,),
        "candidate_center_rms_m": (count,),
        "candidate_max_terminal_native_aabb_iou": (count,),
    }
    if schema == SHADOW_SCHEMA:
        shapes.update(
            {
                "candidate_max_camera_baseline_m": (count,),
                "candidate_max_view_ray_span_deg": (count,),
                "candidate_evidence_offsets": (count + 1,),
            }
        )
    else:
        shapes.update(
            {
                "candidate_qualification_frame_id": (count,),
                "candidate_receipt_evidence_offsets": (count + 1,),
                "candidate_depth_node_offsets": (count + 1,),
                "candidate_support_edge_offsets": (count + 1,),
            }
        )
    for name, expected_shape in shapes.items():
        if arrays[name].shape != expected_shape:
            raise Past3OracleError(
                f"shadow array {name} shape mismatch: {arrays[name].shape}"
            )
    if count < 0 or not np.isfinite(arrays["candidate_corners_world"]).all():
        raise Past3OracleError("invalid Boxer-Past3 candidate count/geometry")
    if np.any(
        (arrays["candidate_scene_index"] < 0)
        | (arrays["candidate_scene_index"] >= len(scenes))
    ):
        raise Past3OracleError("candidate scene index out of range")
    if schema == SHADOW_SCHEMA:
        offsets = arrays["candidate_evidence_offsets"]
        evidence_frames = arrays["evidence_frame_id"]
        evidence_rows = arrays["evidence_source_row"]
    else:
        offsets = arrays["candidate_receipt_evidence_offsets"]
        evidence_frames = arrays["receipt_evidence_frame_id"]
        evidence_rows = arrays["receipt_evidence_source_row"]
    if (
        offsets.dtype.kind not in "iu"
        or len(offsets) != count + 1
        or int(offsets[0]) != 0
        or np.any(np.diff(offsets) < 3)
        or int(offsets[-1]) != len(evidence_frames)
        or len(evidence_rows) != len(evidence_frames)
    ):
        raise Past3OracleError("invalid Boxer-Past3 evidence offsets")
    for candidate_index in range(count):
        lower = int(offsets[candidate_index])
        upper = int(offsets[candidate_index + 1])
        frames = evidence_frames[lower:upper]
        if len(set(int(value) for value in frames)) < 3 or np.any(np.diff(frames) <= 0):
            raise Past3OracleError("candidate evidence is not three distinct ordered frames")
        if int(frames[-1]) != int(arrays["candidate_confirmation_frame_id"][candidate_index]):
            raise Past3OracleError("receipt evidence does not end at confirmation")

    if schema == DEPTH_SHADOW_SCHEMA:
        if np.any(
            arrays["candidate_qualification_frame_id"]
            < arrays["candidate_confirmation_frame_id"]
        ):
            raise Past3OracleError("depth qualification precedes the geometry receipt")
        node_offsets = arrays["candidate_depth_node_offsets"]
        edge_offsets = arrays["candidate_support_edge_offsets"]
        node_frames = arrays["depth_node_frame_id"]
        node_rows = arrays["depth_node_source_row"]
        guide_counts = arrays["depth_node_guide_count"]
        edge_source = arrays["support_edge_source_frame_id"]
        edge_target = arrays["support_edge_target_frame_id"]
        edge_vf = arrays["support_edge_vf"]
        edge_vb = arrays["support_edge_vb"]
        edge_baseline = arrays["support_edge_camera_baseline_m"]
        edge_ray = arrays["support_edge_view_ray_span_deg"]
        if (
            np.any(np.diff(node_offsets) < 3)
            or np.any(np.diff(edge_offsets) < 2)
            or int(node_offsets[0]) != 0
            or int(edge_offsets[0]) != 0
            or int(node_offsets[-1]) != len(node_frames)
            or len(node_rows) != len(node_frames)
            or len(guide_counts) != len(node_frames)
            or int(edge_offsets[-1]) != len(edge_source)
            or any(
                len(value) != len(edge_source)
                for value in (
                    edge_target,
                    edge_vf,
                    edge_vb,
                    edge_baseline,
                    edge_ray,
                )
            )
        ):
            raise Past3OracleError("invalid depth-qualified evidence offsets")
        if not all(
            np.isfinite(value).all()
            for value in (edge_vf, edge_vb, edge_baseline, edge_ray)
        ):
            raise Past3OracleError("non-finite depth support metrics")
        if np.any((guide_counts < 16) | (guide_counts > 64)):
            raise Past3OracleError("depth guide count violates the frozen gate")
        for candidate_index in range(count):
            node_lower = int(node_offsets[candidate_index])
            node_upper = int(node_offsets[candidate_index + 1])
            candidate_nodes = [int(value) for value in node_frames[node_lower:node_upper]]
            if len(candidate_nodes) != len(set(candidate_nodes)):
                raise Past3OracleError("depth component contains duplicate frames")
            allowed = set(candidate_nodes)
            edge_lower = int(edge_offsets[candidate_index])
            edge_upper = int(edge_offsets[candidate_index + 1])
            adjacency = {frame_id: set() for frame_id in candidate_nodes}
            for edge_index in range(edge_lower, edge_upper):
                source = int(edge_source[edge_index])
                target = int(edge_target[edge_index])
                if source not in allowed or target not in allowed or source >= target:
                    raise Past3OracleError("support edge is outside its causal component")
                if not (
                    float(edge_vf[edge_index]) > 0.30
                    and float(edge_vb[edge_index]) > 0.90
                    and float(edge_baseline[edge_index]) >= 0.15
                    and float(edge_ray[edge_index]) >= 10.0
                ):
                    raise Past3OracleError("support edge violates the frozen S1 gate")
                adjacency[source].add(target)
                adjacency[target].add(source)
            reached: set[int] = set()
            pending = [candidate_nodes[0]]
            while pending:
                current = pending.pop()
                if current in reached:
                    continue
                reached.add(current)
                pending.extend(sorted(adjacency[current] - reached, reverse=True))
            if reached != allowed:
                raise Past3OracleError("exported depth nodes are not one weak component")
    return manifest, arrays, scenes


def _threshold_key(threshold: float) -> str:
    return f"{threshold:.2f}"


def _json_evaluation(
    evaluation: Mapping[str, object], scenes: Sequence[str]
) -> dict[str, Any]:
    masks = evaluation["matched_gt_masks"]
    assert isinstance(masks, list)
    return {
        key: value
        for key, value in evaluation.items()
        if key not in {"matched_gt_masks", "evaluation_order"}
    } | {
        "per_scene": {
            scene: {
                "greedy_tp": int(np.count_nonzero(mask)),
                "matched_gt_indices": np.flatnonzero(mask).tolist(),
                "unmatched_gt_indices": np.flatnonzero(~mask).tolist(),
            }
            for scene, mask in zip(scenes, masks)
        }
    }


def audit_scannet_boxer_past3_oracle(
    *,
    shadow_json: Path,
    shadow_npz: Path,
    baseline_root: Path,
    gt_root: Path,
    scan_root: Path,
) -> dict[str, Any]:
    """Evaluate the fixed shadow candidates; never select using GT."""

    shadow_json = shadow_json.resolve()
    shadow_npz = shadow_npz.resolve()
    baseline_root = baseline_root.resolve()
    gt_root = gt_root.resolve()
    scan_root = scan_root.resolve()
    manifest, arrays, scenes = _load_shadow(shadow_json, shadow_npz)

    baseline_before: dict[str, str] = {}
    input_hashes: dict[str, Any] = {
        "shadow_json": _sha256(shadow_json),
        "shadow_npz": _sha256(shadow_npz),
        "scenes": {},
    }
    gt_counts: list[int] = []
    baseline_iou: list[np.ndarray] = []
    candidate_iou: list[np.ndarray] = []
    scene_reports: dict[str, Any] = {}

    for scene_index, scene in enumerate(scenes):
        prediction_path = _regular_file(
            baseline_root / f"{scene}_boxes.pkl", "native T05 prediction"
        )
        gt_path = _regular_file(gt_root / f"{scene}_bbox.npy", "ScanNet GT")
        metadata_path = _regular_file(
            scan_root / scene / f"{scene}.txt", "ScanNet axis alignment"
        )
        baseline_before[scene] = _sha256(prediction_path)
        sealed_hashes = manifest.get("native_prediction_sha256_before")
        if not isinstance(sealed_hashes, dict) or sealed_hashes.get(scene) != baseline_before[scene]:
            raise Past3OracleError(f"native T05 prediction differs from shadow seal: {scene}")
        alignment = load_axis_alignment(metadata_path)
        gt = load_gt_minmax(gt_path)
        _, baseline_aligned = load_baseline_boxes(prediction_path, alignment)
        positions = np.flatnonzero(arrays["candidate_scene_index"] == scene_index)
        corners = arrays["candidate_corners_world"][positions].astype(np.float64)
        if len(corners):
            aligned_corners = corners @ alignment[:3, :3].T + alignment[:3, 3]
            candidate_aligned = np.concatenate(
                (aligned_corners.min(axis=1), aligned_corners.max(axis=1)), axis=1
            )
        else:
            candidate_aligned = np.empty((0, 6), dtype=np.float64)
        baseline_matrix = aligned_iou_matrix(baseline_aligned, gt)
        candidate_matrix = aligned_iou_matrix(candidate_aligned, gt)
        gt_counts.append(len(gt))
        baseline_iou.append(baseline_matrix)
        candidate_iou.append(candidate_matrix)
        scene_reports[scene] = {
            "gt_count": int(len(gt)),
            "baseline_prediction_count": int(len(baseline_aligned)),
            "fixed_candidate_count": int(len(candidate_aligned)),
            "candidate_global_indices": [int(value) for value in positions],
            "candidate_track_ids": [
                int(value) for value in arrays["candidate_track_id"][positions]
            ],
        }
        input_hashes["scenes"][scene] = {
            "native_prediction": baseline_before[scene],
            "gt": _sha256(gt_path),
            "axis_alignment": _sha256(metadata_path),
        }

    baseline_eval = {
        threshold: official_constant_evaluate(baseline_iou, gt_counts, threshold)
        for threshold in THRESHOLDS
    }
    combined_iou = [
        np.concatenate((baseline, candidates), axis=0)
        for baseline, candidates in zip(baseline_iou, candidate_iou)
    ]
    combined_eval = {
        threshold: official_constant_evaluate(combined_iou, gt_counts, threshold)
        for threshold in THRESHOLDS
    }

    threshold_reports: dict[str, Any] = {}
    promotion_ap_nonnegative: dict[str, bool] = {}
    promotion_recovers: dict[str, bool] = {}
    total_candidates = int(sum(len(matrix) for matrix in candidate_iou))
    total_gt = int(sum(gt_counts))
    for threshold in THRESHOLDS:
        key = _threshold_key(threshold)
        candidate_matching = 0
        native_matching = 0
        union_matching = 0
        recovered_official_unmatched = 0
        per_scene: dict[str, Any] = {}
        baseline_masks = baseline_eval[threshold]["matched_gt_masks"]
        assert isinstance(baseline_masks, list)
        for scene, baseline, candidates, matched in zip(
            scenes, baseline_iou, candidate_iou, baseline_masks
        ):
            candidate_pairs = strict_maximum_matching(candidates, threshold)
            native_pairs = strict_maximum_matching(baseline, threshold)
            union_pairs = strict_maximum_matching(
                np.concatenate((baseline, candidates), axis=0), threshold
            )
            recovered_pairs = strict_maximum_matching(
                candidates, threshold, ~np.asarray(matched, dtype=bool)
            )
            candidate_matching += len(candidate_pairs)
            native_matching += len(native_pairs)
            union_matching += len(union_pairs)
            recovered_official_unmatched += len(recovered_pairs)
            per_scene[scene] = {
                "candidate_maximum_matching_count": len(candidate_pairs),
                "candidate_maximum_matching_pairs": [
                    list(pair) for pair in candidate_pairs
                ],
                "native_maximum_matching_count": len(native_pairs),
                "native_union_maximum_matching_count": len(union_pairs),
                "additional_union_matching_over_native": len(union_pairs)
                - len(native_pairs),
                "official_baseline_unmatched_recovered_count": len(recovered_pairs),
                "official_baseline_unmatched_recovery_pairs": [
                    list(pair) for pair in recovered_pairs
                ],
            }
        delta_ap = 100.0 * (
            float(combined_eval[threshold]["ap"])
            - float(baseline_eval[threshold]["ap"])
        )
        additional = union_matching - native_matching
        promotion_ap_nonnegative[key] = delta_ap >= -1e-12
        promotion_recovers[key] = additional >= 1
        threshold_reports[key] = {
            "iou_threshold": threshold,
            "candidate_maximum_matching_count": candidate_matching,
            "candidate_maximum_matching_precision": (
                candidate_matching / total_candidates if total_candidates else 0.0
            ),
            "candidate_maximum_matching_recall": (
                candidate_matching / total_gt if total_gt else 0.0
            ),
            "native_maximum_matching_count": native_matching,
            "native_union_maximum_matching_count": union_matching,
            "additional_union_matching_over_native": additional,
            "incremental_recall_headroom_points": (
                100.0 * additional / total_gt if total_gt else 0.0
            ),
            "official_baseline_unmatched_recovered_count": recovered_official_unmatched,
            "baseline_constant_score": _json_evaluation(
                baseline_eval[threshold], scenes
            ),
            "fixed_candidate_suffix_constant_score": _json_evaluation(
                combined_eval[threshold], scenes
            ),
            "fixed_suffix_delta_ap_points": delta_ap,
            "fixed_suffix_delta_greedy_tp": int(combined_eval[threshold]["greedy_tp"])
            - int(baseline_eval[threshold]["greedy_tp"]),
            "fixed_suffix_delta_false_positive": int(
                combined_eval[threshold]["false_positive"]
            )
            - int(baseline_eval[threshold]["false_positive"]),
            "per_scene": per_scene,
        }

    passes_ap = all(promotion_ap_nonnegative.values())
    passes_recovery = all(promotion_recovers.values())
    promotion = {
        "preregistered": True,
        "active_birth_enabled": False,
        "requires_nonnegative_constant_score_delta_ap_all_thresholds": True,
        "nonnegative_delta_ap": promotion_ap_nonnegative,
        "passes_nonnegative_delta_ap_all_thresholds": passes_ap,
        "requires_at_least_one_additional_union_match_all_thresholds": True,
        "additional_union_match": promotion_recovers,
        "passes_additional_union_match_all_thresholds": passes_recovery,
        "passes_three_scene_active_counterfactual_gate": passes_ap
        and passes_recovery,
        "decision": (
            "promote_to_bounded_three_scene_active_counterfactual"
            if passes_ap and passes_recovery
            else "reject_s0_active_birth"
        ),
        "full100_authorized": False,
    }

    baseline_after = {
        scene: _sha256(baseline_root / f"{scene}_boxes.pkl") for scene in scenes
    }
    if baseline_after != baseline_before:
        raise Past3OracleError("native T05 predictions changed during oracle")
    return {
        "schema": SCHEMA,
        "oracle_only": True,
        "deployable_candidate_selection": True,
        "candidate_selection_used_gt": False,
        "evaluation_used_gt": True,
        "birth_enabled": False,
        "native_predictions_modified": False,
        "score_mode": "constant_1.0",
        "class_mode": "class_agnostic",
        "strict_iou_comparison": ">",
        "candidate_order": "sealed_scene_order_then_shadow_rank",
        "native_rows_are_on_disk_prefix": True,
        "official_tie_order": "numpy.argsort_default_all_scores_1.0",
        "scene_order": list(scenes),
        "thresholds": list(THRESHOLDS),
        "totals": {
            "scene_count": len(scenes),
            "gt_count": total_gt,
            "baseline_prediction_count": int(
                sum(len(matrix) for matrix in baseline_iou)
            ),
            "fixed_candidate_count": total_candidates,
        },
        "per_threshold": threshold_reports,
        "promotion": promotion,
        "scenes": scene_reports,
        "shadow": {
            "json_path": os.fspath(shadow_json),
            "json_sha256": _sha256(shadow_json),
            "npz_path": os.fspath(shadow_npz),
            "npz_sha256": _sha256(shadow_npz),
            "schema": manifest["schema"],
            "preregistration": manifest.get("input", {}).get("preregistration"),
            "preregistration_sha256": manifest.get("input", {}).get(
                "preregistration_sha256"
            ),
        },
        "input_sha256": input_hashes,
        "native_prediction_sha256_before": baseline_before,
        "native_prediction_sha256_after": baseline_after,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one sealed Boxer-Past3 shadow candidate suffix"
    )
    parser.add_argument("--shadow-json", required=True, type=Path)
    parser.add_argument("--shadow-npz", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--scan-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out = args.out.resolve()
    if out.suffix.lower() != ".json":
        raise Past3OracleError("oracle output must have a .json suffix")
    if out.exists() or out.is_symlink():
        raise Past3OracleError(f"refusing to overwrite oracle output: {out}")
    report = audit_scannet_boxer_past3_oracle(
        shadow_json=args.shadow_json,
        shadow_npz=args.shadow_npz,
        baseline_root=args.baseline_root,
        gt_root=args.gt_root,
        scan_root=args.scan_root,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "out": os.fspath(out),
                "totals": report["totals"],
                "promotion": report["promotion"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
