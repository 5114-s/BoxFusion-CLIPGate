#!/usr/bin/env python3
"""Identity-constrained dual-geometry oracle for the sealed N0 paper100 shadow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if os.fspath(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, os.fspath(ROOT))

from tools.audit_scannet_boxer_unexplained_oracle import (  # noqa: E402
    aligned_iou_matrix,
    load_axis_alignment,
    load_baseline_boxes,
    load_gt_minmax,
    load_scene_list,
    official_constant_evaluate,
    strict_maximum_matching,
)
from tools.audit_scannet_fastsam_f1_paper100_oracle import (  # noqa: E402
    EXPECTED_BASELINE_AP_POINTS,
    REQUIRED_ADDITIONAL_MATCHES,
    THRESHOLDS,
)


SCHEMA = "boxfusion.scannet_sam2_n0_fullroute_paper100_oracle.v1"
MERGE_SCHEMA = "boxfusion.scannet_sam2_n0_fullroute_paper100.merge.v1"
SCENE_SCHEMA = "boxfusion.scannet_sam2_n0_fullroute_paper100.scene.v1"
PROTOCOL_ID = "F0-F3-N0-SAM2-TSDF-MV3DIS-DUAL-OBB-SHADOW-PAPER100-V1"
HYPOTHESES = ("NROBUST", "HBEST")
TARGET_AP50_DELTA_POINTS = 10.0
DEFAULT_ROOT = ROOT / "logs/scannet_sam2_n0_fullroute_paper100_score05"


class OracleError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise OracleError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OracleError(f"could not read {label}: {path}") from error
    if not isinstance(value, dict):
        raise OracleError(f"{label} must contain one JSON object")
    return value


def _aligned_box(corners_value: object, alignment: np.ndarray, label: str) -> np.ndarray:
    try:
        corners = np.asarray(corners_value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise OracleError(f"{label} corners are invalid") from error
    if corners.shape != (8, 3) or not np.isfinite(corners).all():
        raise OracleError(f"{label} corners must be finite [8,3]")
    transformed = corners @ alignment[:3, :3].T + alignment[:3, 3]
    box = np.concatenate((transformed.min(axis=0), transformed.max(axis=0)))
    if np.any(box[3:] <= box[:3]):
        raise OracleError(f"{label} aligned box is degenerate")
    return box


def _evaluation_json(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"matched_gt_masks", "evaluation_order"}
    }


def _load_candidates(
    sidecar: Path,
    expected_sha: str,
    scene: str,
    alignment: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if _sha(sidecar) != expected_sha:
        raise OracleError(f"sealed scene hash differs: {scene}")
    payload = _json(sidecar, f"N0 scene {scene}")
    contracts = payload.get("contracts")
    tracks = payload.get("tracks")
    if (
        payload.get("schema") != SCENE_SCHEMA
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("scene_id") != scene
        or payload.get("complete") is not True
        or not isinstance(contracts, dict)
        or contracts.get("ground_truth_access") is not False
        or contracts.get("annotation_access") is not False
        or contracts.get("evaluator_access") is not False
        or contracts.get("native_output_mutation") is not False
        or contracts.get("query_before_commit") is not True
        or not isinstance(tracks, list)
    ):
        raise OracleError(f"sealed N0 scene contract differs: {scene}")
    result = []
    invalid = 0
    for track in tracks:
        if not isinstance(track, dict) or type(track.get("track_id")) is not int:
            raise OracleError(f"invalid track row: {scene}")
        geometry = track.get("geometry")
        if not isinstance(geometry, dict):
            raise OracleError(f"missing track geometry: {scene}:{track.get('track_id')}")
        if geometry.get("valid") is not True:
            invalid += 1
            continue
        hypotheses = geometry.get("hypotheses")
        selector = geometry.get("selector")
        if not isinstance(hypotheses, dict) or not isinstance(selector, dict):
            raise OracleError(f"dual geometry ledger differs: {scene}:{track['track_id']}")
        boxes: dict[str, np.ndarray] = {}
        for name in HYPOTHESES:
            row = hypotheses.get(name)
            if isinstance(row, dict) and row.get("valid") is True:
                boxes[name] = _aligned_box(
                    row.get("world_corners"), alignment, f"{scene}:{track['track_id']}:{name}"
                )
        if not boxes or selector.get("chosen") not in boxes or selector.get("ground_truth") is not False:
            raise OracleError(f"valid dual geometry is inconsistent: {scene}:{track['track_id']}")
        result.append(
            {
                "track_id": int(track["track_id"]),
                "source_ids": list(track.get("source_ids", [])),
                "boxes": boxes,
                "selector_chosen": str(selector["chosen"]),
            }
        )
    return result, {"valid": len(result), "invalid": invalid, "total": len(tracks)}


def audit(
    *,
    scene_list: Path,
    receipt_path: Path,
    baseline_root: Path,
    gt_root: Path,
    scan_root: Path,
) -> dict[str, Any]:
    scenes = load_scene_list(scene_list)
    if len(scenes) != 100:
        raise OracleError("oracle requires the exact paper100 scene list")
    receipt_sha = _sha(receipt_path)
    receipt = _json(receipt_path, "sealed N0 merge receipt")
    rows = receipt.get("scenes")
    if (
        receipt.get("schema") != MERGE_SCHEMA
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("complete") is not True
        or receipt.get("overall_pass") is not True
        or receipt.get("scene_order") != scenes
        or not isinstance(rows, list)
        or len(rows) != 100
    ):
        raise OracleError("sealed N0 merge receipt differs")

    baseline_matrices = []
    candidate_hypothesis_matrices: list[list[dict[str, np.ndarray]]] = []
    candidate_max_matrices = []
    gt_counts = []
    scene_candidates: list[list[dict[str, Any]]] = []
    scene_reports: dict[str, Any] = {}
    total_valid = total_invalid = 0
    for scene_index, (scene, merge_row) in enumerate(zip(scenes, rows)):
        if not isinstance(merge_row, dict) or merge_row.get("scene_id") != scene or merge_row.get("scene_index") != scene_index:
            raise OracleError(f"merge scene order differs: {scene}")
        sidecar_row = merge_row.get("sidecar")
        if not isinstance(sidecar_row, dict):
            raise OracleError(f"merge sidecar receipt is missing: {scene}")
        alignment = load_axis_alignment(scan_root / scene / f"{scene}.txt")
        gt = load_gt_minmax(gt_root / f"{scene}_bbox.npy")
        _, native = load_baseline_boxes(baseline_root / f"{scene}_boxes.pkl", alignment)
        candidates, counts = _load_candidates(
            Path(str(sidecar_row.get("path", ""))),
            str(sidecar_row.get("sha256", "")),
            scene,
            alignment,
        )
        hypothesis_rows: list[dict[str, np.ndarray]] = []
        max_rows = []
        for candidate in candidates:
            matrices = {
                name: aligned_iou_matrix(box[None, :], gt)[0]
                for name, box in candidate["boxes"].items()
            }
            hypothesis_rows.append(matrices)
            max_rows.append(np.maximum.reduce(list(matrices.values())))
        candidate_max = (
            np.stack(max_rows, axis=0)
            if max_rows
            else np.empty((0, len(gt)), dtype=np.float64)
        )
        baseline_matrices.append(aligned_iou_matrix(native, gt))
        candidate_hypothesis_matrices.append(hypothesis_rows)
        candidate_max_matrices.append(candidate_max)
        gt_counts.append(len(gt))
        scene_candidates.append(candidates)
        total_valid += counts["valid"]
        total_invalid += counts["invalid"]
        scene_reports[scene] = {
            "native_prediction_count": len(native),
            "gt_count": len(gt),
            "valid_identity_constrained_track_count": counts["valid"],
            "invalid_geometry_track_count": counts["invalid"],
        }

    baseline_evaluations = {
        threshold: official_constant_evaluate(baseline_matrices, gt_counts, threshold)
        for threshold in THRESHOLDS
    }
    for threshold, evaluation in baseline_evaluations.items():
        expected = EXPECTED_BASELINE_AP_POINTS[f"{threshold:.2f}"]
        if abs(float(evaluation["ap_points"]) - expected) > 1.0e-9:
            raise OracleError(
                f"official baseline AP reproduction differs at {threshold}: "
                f"{evaluation['ap_points']} != {expected}"
            )

    per_threshold: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        baseline_eval = baseline_evaluations[threshold]
        baseline_masks = baseline_eval["matched_gt_masks"]
        suffix_matrices = []
        selections: dict[str, Any] = {}
        native_matching = candidate_matching = union_matching = selected_total = 0
        for scene, native, candidate_max, hypothesis_rows, candidates, matched_mask in zip(
            scenes,
            baseline_matrices,
            candidate_max_matrices,
            candidate_hypothesis_matrices,
            scene_candidates,
            baseline_masks,
        ):
            native_pairs = strict_maximum_matching(native, threshold)
            candidate_pairs = strict_maximum_matching(candidate_max, threshold)
            union_pairs = strict_maximum_matching(
                np.concatenate((native, candidate_max), axis=0), threshold
            )
            suffix_pairs = strict_maximum_matching(
                candidate_max, threshold, ~np.asarray(matched_mask, dtype=np.bool_)
            )
            chosen_rows = []
            chosen_receipts = []
            for candidate_index, gt_index in suffix_pairs:
                matrices = hypothesis_rows[candidate_index]
                name = sorted(
                    matrices,
                    key=lambda item: (
                        -float(matrices[item][gt_index]),
                        HYPOTHESES.index(item),
                    ),
                )[0]
                chosen_rows.append(matrices[name])
                chosen_receipts.append(
                    {
                        "candidate_index": candidate_index,
                        "track_id": candidates[candidate_index]["track_id"],
                        "hypothesis": name,
                        "target_gt_index": gt_index,
                        "target_iou": float(matrices[name][gt_index]),
                    }
                )
            suffix = (
                np.stack(chosen_rows, axis=0)
                if chosen_rows
                else np.empty((0, candidate_max.shape[1]), dtype=np.float64)
            )
            suffix_matrices.append(suffix)
            selections[scene] = chosen_receipts
            native_matching += len(native_pairs)
            candidate_matching += len(candidate_pairs)
            union_matching += len(union_pairs)
            selected_total += len(suffix_pairs)
        combined = [
            np.concatenate((native, suffix), axis=0)
            for native, suffix in zip(baseline_matrices, suffix_matrices)
        ]
        suffix_eval = official_constant_evaluate(combined, gt_counts, threshold)
        delta = float(suffix_eval["ap_points"]) - float(baseline_eval["ap_points"])
        additional = union_matching - native_matching
        per_threshold[f"{threshold:.2f}"] = {
            "iou_threshold": threshold,
            "identity_constraint": "one F3 track contributes at most one dual-geometry edge",
            "hypothesis_choice": "GT-assisted NROBUST-versus-HBEST; oracle only",
            "native_maximum_matching_count": native_matching,
            "candidate_maximum_matching_count": candidate_matching,
            "union_maximum_matching_count": union_matching,
            "additional_union_matching_over_native": additional,
            "required_additional_union_matches": REQUIRED_ADDITIONAL_MATCHES,
            "passes_144_match_capacity_gate": additional >= REQUIRED_ADDITIONAL_MATCHES,
            "baseline_official_constant_score": _evaluation_json(baseline_eval),
            "gt_selected_candidate_suffix": {
                "oracle_only": True,
                "deployable": False,
                "native_prefix_unchanged": True,
                "formal_score": 1.0,
                "selected_candidate_count": selected_total,
                "official_evaluation": _evaluation_json(suffix_eval),
                "delta_ap_points": delta,
                "passes_plus10_ap": delta > TARGET_AP50_DELTA_POINTS,
                "per_scene_selection": selections,
            },
        }

    ap50 = per_threshold["0.50"]["gt_selected_candidate_suffix"]
    ap50_capacity = per_threshold["0.50"]
    pass_gate = bool(
        float(ap50["delta_ap_points"]) > TARGET_AP50_DELTA_POINTS
        and int(ap50_capacity["additional_union_matching_over_native"])
        >= REQUIRED_ADDITIONAL_MATCHES
    )
    return {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "oracle_only": True,
        "deployable": False,
        "receipt": {"path": os.fspath(receipt_path.resolve()), "sha256": receipt_sha},
        "counts": {
            "scene_count": len(scenes),
            "gt_count": sum(gt_counts),
            "valid_identity_constrained_track_count": total_valid,
            "invalid_geometry_track_count": total_invalid,
        },
        "per_threshold": per_threshold,
        "decision": {
            "ap50_target_delta_points_strictly_greater_than": TARGET_AP50_DELTA_POINTS,
            "ap50_required_additional_union_matches": REQUIRED_ADDITIONAL_MATCHES,
            "passes_ap50_shadow_oracle_gate": pass_gate,
            "authorize_gt_free_active_experiment": pass_gate,
            "authorize_accuracy_claim": False,
            "next_step": (
                "freeze_and_run_gt_free_active_selector"
                if pass_gate
                else "stop_birth_route_candidate_capacity_below_target"
            ),
        },
        "scenes": scene_reports,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, default=ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt")
    parser.add_argument("--receipt", type=Path, default=DEFAULT_ROOT / "final/N0_FULLROUTE_PAPER100.json")
    parser.add_argument("--baseline-root", type=Path, default=ROOT / "results/scannet_t05_boxer_replay_active_score05")
    parser.add_argument("--gt-root", type=Path, default=ROOT / "evaluation/data_util/scannet_train_detection_data")
    parser.add_argument("--scan-root", type=Path, default=Path("/extra/ZhaoX/scannet_data/scans"))
    parser.add_argument("--out", type=Path, default=ROOT / "reports/sam2_n0_fullroute_paper100_oracle/N0_FULLROUTE_PAPER100_ORACLE.json")
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        raise OracleError(f"refusing to overwrite oracle report: {args.out}")
    report = audit(
        scene_list=args.scene_list,
        receipt_path=args.receipt,
        baseline_root=args.baseline_root,
        gt_root=args.gt_root,
        scan_root=args.scan_root,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"out": os.fspath(args.out), "counts": report["counts"], "decision": report["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
