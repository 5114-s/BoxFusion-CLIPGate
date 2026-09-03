#!/usr/bin/env python3
"""Evaluate the fixed L3-HBC geometry bank on paper100.

The source and geometry for every T1 track were selected without GT by the
L3-HBC shadow.  This audit may use GT only to compute capacity and a
constructive suffix; it cannot authorize a deployable AP claim by itself.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))

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
from tools.audit_scannet_l0_f3_f4_perview_paper100_oracle import (  # noqa: E402
    HYPOTHESES,
    L0OracleError,
    _aligned_geometry,
    _evaluation_json,
    _json,
    _sha,
    _source_map,
)
from tools.run_scannet_l3_hbc_t1_selector_paper100 import (  # noqa: E402
    PROTOCOL_ID,
    SCHEMA as SHADOW_SCHEMA,
)


SCHEMA = "boxfusion.scannet_l3_hbc_t1_selector_paper100_oracle.v1"
TARGET_DELTA_AP_POINTS = 10.0
DEFAULT_SHADOW = ROOT / "logs/scannet_l3_hbc_t1_selector_paper100_score05/final/L3_HBC_T1_SELECTOR_PAPER100.json"
DEFAULT_OUT = ROOT / "reports/l3_hbc_t1_selector_paper100_oracle/L3_HBC_T1_SELECTOR_PAPER100_ORACLE.json"


class L3OracleError(L0OracleError):
    pass


def _validate_shadow(
    value: Mapping[str, Any],
    scenes: list[str],
    *,
    expected_shadow_schema: str = SHADOW_SCHEMA,
    expected_protocol_id: str = PROTOCOL_ID,
) -> list[Mapping[str, Any]]:
    rows = value.get("scenes")
    contracts = value.get("contracts")
    if (
        value.get("schema") != expected_shadow_schema
        or value.get("protocol_id") != expected_protocol_id
        or value.get("complete") is not True
        or value.get("overall_pass") is not True
        or not isinstance(rows, list)
        or len(rows) != 100
        or not isinstance(contracts, Mapping)
        or contracts.get("shadow_only") is not True
        or contracts.get("past_only") is not True
        or any(
            contracts.get(name) is not False
            for name in (
                "birth_enabled",
                "native_output_mutation",
                "ground_truth_access",
                "annotation_access",
                "evaluator_access",
                "native_prediction_access",
                "future_frame_access",
                "training",
                "online_learning",
            )
        )
    ):
        raise L3OracleError("L3-HBC shadow contract differs")
    for index, (scene, row) in enumerate(zip(scenes, rows)):
        if not isinstance(row, Mapping) or row.get("scene_id") != scene or row.get("scene_index") != index:
            raise L3OracleError("L3-HBC scene order differs")
    return rows


def audit(
    *,
    scene_list: Path,
    shadow_path: Path,
    baseline_root: Path,
    gt_root: Path,
    scan_root: Path,
    expected_shadow_schema: str = SHADOW_SCHEMA,
    expected_protocol_id: str = PROTOCOL_ID,
    report_schema: str = SCHEMA,
) -> dict[str, Any]:
    scenes = load_scene_list(scene_list)
    if len(scenes) != 100 or len(set(scenes)) != 100:
        raise L3OracleError("L3-HBC requires exact paper100 scene list")
    shadow = _json(shadow_path, "L3-HBC shadow")
    rows = _validate_shadow(
        shadow,
        scenes,
        expected_shadow_schema=expected_shadow_schema,
        expected_protocol_id=expected_protocol_id,
    )

    native_iou: list[np.ndarray] = []
    candidate_iou: list[np.ndarray] = []
    candidate_receipts: list[list[Mapping[str, Any]]] = []
    gt_counts: list[int] = []
    total_tracks = 0
    scene_reports: dict[str, Any] = {}
    for scene, row in zip(scenes, rows):
        f4_receipt = row.get("f4")
        selections = row.get("selections")
        if not isinstance(f4_receipt, Mapping) or not isinstance(selections, list):
            raise L3OracleError(f"L3-HBC scene ledger differs: {scene}")
        f4_path = Path(str(f4_receipt.get("path", "")))
        if _sha(f4_path) != f4_receipt.get("sha256"):
            raise L3OracleError(f"sealed F4 hash differs: {scene}")
        source_map = _source_map(_json(f4_path, f"F4 scene {scene}"), scene)
        alignment = load_axis_alignment(scan_root / scene / f"{scene}.txt")
        gt = load_gt_minmax(gt_root / f"{scene}_bbox.npy")
        _, native = load_baseline_boxes(baseline_root / f"{scene}_boxes.pkl", alignment)
        candidate_rows: list[np.ndarray] = []
        normalized_receipts: list[Mapping[str, Any]] = []
        for expected_track_id, selection in enumerate(selections):
            if (
                not isinstance(selection, Mapping)
                or selection.get("track_id") != expected_track_id
                or selection.get("past_only_at_decision") is not True
                or selection.get("emit_event") != "track_retirement_or_end_of_stream"
            ):
                raise L3OracleError(f"L3-HBC selection order/causality differs: {scene}")
            source_id = str(selection.get("source_id", ""))
            name = str(selection.get("hypothesis", ""))
            source = source_map.get(source_id)
            hypotheses = source.get("hypotheses") if isinstance(source, Mapping) else None
            if name not in HYPOTHESES or not isinstance(hypotheses, Mapping):
                raise L3OracleError(f"L3-HBC chosen geometry is absent: {source_id}.{name}")
            box = _aligned_geometry(hypotheses[name], name, alignment, f"{source_id}.{name}")
            candidate_rows.append(aligned_iou_matrix(box[None, :], gt)[0])
            normalized_receipts.append(selection)
        if len(candidate_rows) != row.get("track_count"):
            raise L3OracleError(f"L3-HBC track census differs: {scene}")
        candidate = np.stack(candidate_rows, axis=0) if candidate_rows else np.empty((0, len(gt)), dtype=np.float64)
        native_iou.append(aligned_iou_matrix(native, gt))
        candidate_iou.append(candidate)
        candidate_receipts.append(normalized_receipts)
        gt_counts.append(len(gt))
        total_tracks += len(candidate_rows)
        scene_reports[scene] = {
            "native_prediction_count": len(native),
            "gt_count": len(gt),
            "fixed_track_geometry_count": len(candidate_rows),
        }
    if total_tracks != shadow.get("counts", {}).get("track_count"):
        raise L3OracleError("L3-HBC global track census differs")

    baseline_evaluations = {
        threshold: official_constant_evaluate(native_iou, gt_counts, threshold)
        for threshold in THRESHOLDS
    }
    for threshold, evaluation in baseline_evaluations.items():
        expected = EXPECTED_BASELINE_AP_POINTS[f"{threshold:.2f}"]
        if not math.isclose(float(evaluation["ap_points"]), expected, rel_tol=0.0, abs_tol=1e-9):
            raise L3OracleError(f"paper100 baseline differs at {threshold}")

    per_threshold: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        baseline = baseline_evaluations[threshold]
        suffix_rows: list[np.ndarray] = []
        per_scene_selection: dict[str, Any] = {}
        chosen: Counter[str] = Counter()
        native_mm = candidate_mm = union_mm = selected_total = 0
        for scene, native, candidate, receipts, official_mask in zip(
            scenes,
            native_iou,
            candidate_iou,
            candidate_receipts,
            baseline["matched_gt_masks"],
        ):
            native_pairs = strict_maximum_matching(native, threshold)
            candidate_pairs = strict_maximum_matching(candidate, threshold)
            union_pairs = strict_maximum_matching(np.concatenate((native, candidate), axis=0), threshold)
            suffix_pairs = strict_maximum_matching(candidate, threshold, ~np.asarray(official_mask, dtype=np.bool_))
            rows_for_scene: list[np.ndarray] = []
            selections_for_scene: list[dict[str, Any]] = []
            for candidate_index, gt_index in suffix_pairs:
                receipt = receipts[candidate_index]
                rows_for_scene.append(candidate[candidate_index])
                name = str(receipt["hypothesis"])
                chosen[name] += 1
                selections_for_scene.append(
                    {
                        "candidate_index": candidate_index,
                        "track_id": int(receipt["track_id"]),
                        "source_id": str(receipt["source_id"]),
                        "fixed_hypothesis": name,
                        "target_gt_index": gt_index,
                        "target_iou": float(candidate[candidate_index, gt_index]),
                        "retained_observation_count": int(receipt["retained_observation_count"]),
                    }
                )
            suffix_rows.append(np.stack(rows_for_scene, axis=0) if rows_for_scene else np.empty((0, candidate.shape[1]), dtype=np.float64))
            per_scene_selection[scene] = selections_for_scene
            native_mm += len(native_pairs)
            candidate_mm += len(candidate_pairs)
            union_mm += len(union_pairs)
            selected_total += len(suffix_pairs)
        combined = [np.concatenate((native, suffix), axis=0) for native, suffix in zip(native_iou, suffix_rows)]
        evaluation = official_constant_evaluate(combined, gt_counts, threshold)
        delta = float(evaluation["ap_points"]) - float(baseline["ap_points"])
        additional = union_mm - native_mm
        per_threshold[f"{threshold:.2f}"] = {
            "iou_threshold": threshold,
            "strict_iou_comparison": ">",
            "geometry_choice": "fixed_by_L3_HBC_without_GT",
            "candidate_maximum_matching_count": candidate_mm,
            "union_maximum_matching_count": union_mm,
            "native_maximum_matching_count": native_mm,
            "additional_union_matching_over_native": additional,
            "required_additional_union_matches": REQUIRED_ADDITIONAL_MATCHES,
            "passes_144_match_capacity_gate": additional >= REQUIRED_ADDITIONAL_MATCHES,
            "baseline_official_constant_score": _evaluation_json(baseline),
            "gt_selected_track_suffix": {
                "oracle_only": True,
                "deployable": False,
                "geometry_selection_uses_gt": False,
                "track_admission_uses_gt": True,
                "native_prefix_unchanged": True,
                "formal_score": 1.0,
                "selected_track_count": selected_total,
                "selected_fixed_hypothesis_counts": {name: int(chosen.get(name, 0)) for name in HYPOTHESES},
                "official_evaluation": _evaluation_json(evaluation),
                "delta_ap_points": delta,
                "passes_plus10_ap": delta >= TARGET_DELTA_AP_POINTS,
                "per_scene_selection": per_scene_selection,
            },
        }

    ap50 = per_threshold["0.50"]
    gate = bool(
        ap50["passes_144_match_capacity_gate"]
        and ap50["gt_selected_track_suffix"]["passes_plus10_ap"]
    )
    return {
        "schema": report_schema,
        "protocol_id": expected_protocol_id,
        "complete": True,
        "oracle_only": True,
        "deployable": False,
        "shadow": {"path": os.fspath(shadow_path.resolve()), "sha256": _sha(shadow_path)},
        "counts": {"scene_count": 100, "gt_count": sum(gt_counts), "fixed_track_geometry_count": total_tracks},
        "per_threshold": per_threshold,
        "decision": {
            "passes_ap50_plus10_and_144_match_gate": gate,
            "authorize_active_birth": False,
            "authorize_accuracy_claim": False,
            "next_step": "freeze_L4_gtfree_track_admission_policy" if gate else "discard_L3_HBC_geometry_selector_for_plus10_route",
        },
        "scenes": scene_reports,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, default=ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt")
    parser.add_argument("--shadow", type=Path, default=DEFAULT_SHADOW)
    parser.add_argument("--baseline-root", type=Path, default=ROOT / "results/scannet_t05_boxer_replay_active_score05")
    parser.add_argument("--gt-root", type=Path, default=ROOT / "evaluation/data_util/scannet_train_detection_data")
    parser.add_argument("--scan-root", type=Path, default=Path("/extra/ZhaoX/scannet_data/scans"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        raise L3OracleError(f"refusing to overwrite L3-HBC oracle: {args.out}")
    report = audit(scene_list=args.scene_list, shadow_path=args.shadow, baseline_root=args.baseline_root, gt_root=args.gt_root, scan_root=args.scan_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    summary = {
        key: {
            "ap_points": row["gt_selected_track_suffix"]["official_evaluation"]["ap_points"],
            "delta_ap_points": row["gt_selected_track_suffix"]["delta_ap_points"],
            "additional_matches": row["additional_union_matching_over_native"],
        }
        for key, row in report["per_threshold"].items()
    }
    print(json.dumps({"out": os.fspath(args.out), "summary": summary, "decision": report["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
