#!/usr/bin/env python3
"""Paper100 oracle separating F3 track compression from F4 source capacity.

L2 reads the no-GT L2 seal and evaluates five nested identity universes.  The
two endpoint modes are required to reproduce L1 (T2) and raw F4 G4 (SRAW), so
the three intermediate modes form an auditable attribution rather than a new
proposal generator.  This program is an oracle only: it never writes boxes or
authorizes a birth branch.
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
from tools.seal_scannet_l2_source_preserving_paper100 import (  # noqa: E402
    MODES,
    PROTOCOL_ID,
    SCHEMA as SEAL_SCHEMA,
)


SCHEMA = "boxfusion.scannet_l2_source_preserving_paper100_oracle.v1"
TARGET_DELTA_AP_POINTS = 10.0
DEFAULT_SEAL = ROOT / "logs/scannet_l2_source_preserving_paper100_score05/final/L2_SOURCE_PRESERVING_PAPER100.json"
DEFAULT_OUT = ROOT / "reports/l2_source_preserving_paper100_oracle/L2_SOURCE_PRESERVING_PAPER100_ORACLE.json"
DEFAULT_L1 = ROOT / "reports/l1_f3_2view_f4_perview_paper100_oracle/L1_F3_2VIEW_F4_PERVIEW_PAPER100_ORACLE.json"
DEFAULT_F4 = ROOT / "reports/fastsam_f4_boxer_paper100_oracle/F4_FASTSAM_BOXER_PAPER100_ORACLE.json"


class L2OracleError(L0OracleError):
    """Raised if an L2 input or endpoint reproduction differs."""


def _assert_regular(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise L2OracleError(f"{label} must be a regular file: {path}")


def _load_report(path: Path, label: str, schema: str) -> dict[str, Any]:
    value = _json(path, label)
    # F4 predates the common ``complete`` field; it exposes ``oracle_only``
    # instead.  The schema lock below is intentionally the primary contract.
    if value.get("schema") != schema or value.get("oracle_only") is not True:
        raise L2OracleError(f"{label} schema/oracle contract differs")
    if "complete" in value and value["complete"] is not True:
        raise L2OracleError(f"{label} is explicitly incomplete")
    return value


def _empty_rows(columns: int) -> np.ndarray:
    return np.empty((0, columns), dtype=np.float64)


def _normalized_hypotheses(value: Mapping[str, Any]) -> dict[str, int]:
    return {name: int(value.get(name, 0)) for name in HYPOTHESES}


def _core_evaluation(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "ap",
        "ap_points",
        "recall",
        "precision",
        "greedy_tp",
        "false_positive",
        "unmatched_gt_count",
    )
    if any(name not in value for name in keys):
        raise L2OracleError("reference official evaluation lacks a scalar")
    return {name: value[name] for name in keys}


def _same_scalars(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    if set(actual) != set(expected):
        return False
    for name in actual:
        left, right = actual[name], expected[name]
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12):
                return False
        elif left != right:
            return False
    return True


def _endpoint_projection(mode: Mapping[str, Any]) -> dict[str, Any]:
    suffix = mode["gt_selected_candidate_suffix"]
    return {
        "candidate_maximum_matching_count": int(mode["candidate_maximum_matching_count"]),
        "union_maximum_matching_count": int(mode["union_maximum_matching_count"]),
        "native_maximum_matching_count": int(mode["native_maximum_matching_count"]),
        "additional_union_matching_over_native": int(mode["additional_union_matching_over_native"]),
        "selected_identity_count": int(suffix["selected_identity_count"]),
        "selected_hypothesis_counts": _normalized_hypotheses(suffix["selected_hypothesis_counts"]),
        "official_evaluation": _core_evaluation(suffix["official_evaluation"]),
        "delta_ap_points": float(suffix["delta_ap_points"]),
    }


def _expected_l1_projection(mode: Mapping[str, Any]) -> dict[str, Any]:
    suffix = mode["gt_selected_candidate_suffix"]
    return {
        "candidate_maximum_matching_count": int(mode["candidate_maximum_matching_count"]),
        "union_maximum_matching_count": int(mode["union_maximum_matching_count"]),
        "native_maximum_matching_count": int(mode["native_maximum_matching_count"]),
        "additional_union_matching_over_native": int(mode["additional_union_matching_over_native"]),
        "selected_identity_count": int(suffix["selected_candidate_count"]),
        "selected_hypothesis_counts": _normalized_hypotheses(suffix["selected_hypothesis_counts"]),
        "official_evaluation": _core_evaluation(suffix["official_evaluation"]),
        "delta_ap_points": float(suffix["delta_ap_points"]),
    }


def _expected_f4_projection(mode: Mapping[str, Any]) -> dict[str, Any]:
    suffix = mode["gt_selected_candidate_suffix"]
    return {
        "candidate_maximum_matching_count": int(mode["candidate_maximum_matching_count"]),
        "union_maximum_matching_count": int(mode["union_maximum_matching_count"]),
        "native_maximum_matching_count": int(mode["native_maximum_matching_count"]),
        "additional_union_matching_over_native": int(mode["additional_union_matching_over_native"]),
        "selected_identity_count": int(suffix["selected_source_count"]),
        "selected_hypothesis_counts": _normalized_hypotheses(suffix["chosen_hypothesis_counts"]),
        "official_evaluation": _core_evaluation(suffix["official_evaluation"]),
        "delta_ap_points": float(suffix["delta_ap_points"]),
    }


def _mode_specs(
    row: Mapping[str, Any], source_order: list[str]
) -> dict[str, list[tuple[int | str, list[str]]]]:
    tracks = row.get("tracks")
    if not isinstance(tracks, list):
        raise L2OracleError("L2 scene lacks track ledger")
    parsed: list[tuple[int, list[str]]] = []
    seen: set[str] = set()
    for expected, track in enumerate(tracks):
        if not isinstance(track, Mapping) or track.get("track_id") != expected:
            raise L2OracleError("L2 track order differs")
        source_ids = track.get("source_ids")
        retained = track.get("retained_source_ids")
        if not isinstance(source_ids, list) or not isinstance(retained, list) or not retained:
            raise L2OracleError("L2 track source ledger differs")
        normalized = [str(item) for item in source_ids]
        selected = [str(item) for item in retained]
        if len(normalized) != len(set(normalized)) or not set(selected).issubset(normalized):
            raise L2OracleError("L2 source partition differs")
        if seen.intersection(normalized):
            raise L2OracleError("L2 source belongs to multiple tracks")
        seen.update(normalized)
        parsed.append((expected, selected))
    if seen != set(source_order):
        raise L2OracleError("L2 tracks do not partition raw source universe")
    t2 = [(track_id, sources) for track_id, sources in parsed if len(sources) >= 2]
    s2_ids = {source_id for _, sources in t2 for source_id in sources}
    s1_ids = {source_id for _, sources in parsed for source_id in sources}
    specs = {
        "T2": [(track_id, sources) for track_id, sources in t2],
        "S2": [(source_id, [source_id]) for source_id in source_order if source_id in s2_ids],
        "T1": [(track_id, sources) for track_id, sources in parsed],
        "S1R": [(source_id, [source_id]) for source_id in source_order if source_id in s1_ids],
        "SRAW": [(source_id, [source_id]) for source_id in source_order],
    }
    expected_counts = row.get("mode_identity_counts")
    if not isinstance(expected_counts, Mapping) or any(
        len(specs[name]) != expected_counts.get(name) for name in MODES
    ):
        raise L2OracleError("L2 mode census differs from seal")
    return specs


def _candidate_matrix(
    specs: Sequence[tuple[int | str, list[str]]], source_index: Mapping[str, int], source_max: np.ndarray
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for _, source_ids in specs:
        try:
            indices = [source_index[source_id] for source_id in source_ids]
        except KeyError as error:
            raise L2OracleError("L2 retained source is absent from F4") from error
        rows.append(np.max(source_max[np.asarray(indices, dtype=np.int64)], axis=0))
    return np.stack(rows, axis=0) if rows else _empty_rows(source_max.shape[1])


def _winner(
    source_ids: Sequence[str], source_index: Mapping[str, int], matrices: Mapping[str, np.ndarray], gt_index: int
) -> tuple[str, str, np.ndarray]:
    chosen: tuple[str, str, np.ndarray] | None = None
    best = -np.inf
    for source_id in source_ids:
        source_row = source_index[source_id]
        for name in HYPOTHESES:
            iou_row = matrices[name][source_row]
            score = float(iou_row[gt_index])
            if score > best:
                best = score
                chosen = (source_id, name, iou_row)
    if chosen is None:
        raise L2OracleError("empty L2 geometry bank")
    return chosen


def _new_accumulator() -> dict[str, Any]:
    return {
        "candidate_mm": 0,
        "union_mm": 0,
        "native_mm": 0,
        "selected": 0,
        "suffix": [],
        "selection": {},
        "per_scene": {},
        "chosen": Counter(),
    }


def _validate_seal(seal: Mapping[str, Any], scenes: list[str]) -> list[Mapping[str, Any]]:
    rows = seal.get("scenes")
    contracts = seal.get("contracts")
    if (
        seal.get("schema") != SEAL_SCHEMA
        or seal.get("protocol_id") != PROTOCOL_ID
        or seal.get("complete") is not True
        or seal.get("overall_pass") is not True
        or seal.get("scene_order") != scenes
        or not isinstance(rows, list)
        or len(rows) != 100
        or not isinstance(contracts, Mapping)
        or any(contracts.get(name) is not False for name in ("ground_truth_access", "annotation_access", "evaluator_access", "training", "online_learning"))
    ):
        raise L2OracleError("L2 no-GT seal contract differs")
    for index, (scene, row) in enumerate(zip(scenes, rows)):
        if not isinstance(row, Mapping) or row.get("scene_id") != scene or row.get("scene_index") != index:
            raise L2OracleError("L2 scene order differs")
    return rows


def audit(
    *, scene_list: Path, seal_path: Path, baseline_root: Path, gt_root: Path, scan_root: Path,
    l1_report_path: Path, f4_report_path: Path,
) -> dict[str, Any]:
    scenes = load_scene_list(scene_list)
    if len(scenes) != 100 or len(set(scenes)) != 100:
        raise L2OracleError("L2 requires exact paper100 scene list")
    _assert_regular(seal_path, "L2 seal")
    seal = _json(seal_path, "L2 seal")
    seal_rows = _validate_seal(seal, scenes)
    l1 = _load_report(l1_report_path, "L1 endpoint report", "boxfusion.scannet_l1_f3_2view_f4_perview_paper100_oracle.v1")
    f4 = _load_report(f4_report_path, "F4 endpoint report", "boxfusion.scannet_fastsam_f4_boxer_paper100_oracle.v1")

    baseline_matrices: list[np.ndarray] = []
    gt_counts: list[int] = []
    alignments: list[np.ndarray] = []
    gt_rows: list[np.ndarray] = []
    for scene in scenes:
        alignment = load_axis_alignment(scan_root / scene / f"{scene}.txt")
        gt = load_gt_minmax(gt_root / f"{scene}_bbox.npy")
        _, native = load_baseline_boxes(baseline_root / f"{scene}_boxes.pkl", alignment)
        alignments.append(alignment)
        gt_rows.append(gt)
        gt_counts.append(len(gt))
        baseline_matrices.append(aligned_iou_matrix(native, gt))
    baseline_evaluations = {
        threshold: official_constant_evaluate(baseline_matrices, gt_counts, threshold)
        for threshold in THRESHOLDS
    }
    for threshold, evaluation in baseline_evaluations.items():
        expected = EXPECTED_BASELINE_AP_POINTS[f"{threshold:.2f}"]
        if not math.isclose(float(evaluation["ap_points"]), expected, rel_tol=0.0, abs_tol=1e-9):
            raise L2OracleError(f"paper100 baseline differs at {threshold}")

    accumulators = {
        f"{threshold:.2f}": {mode: _new_accumulator() for mode in MODES}
        for threshold in THRESHOLDS
    }
    identities = {mode: 0 for mode in MODES}
    geometry_total = 0

    for scene_index, (scene, row, alignment, gt, native) in enumerate(
        zip(scenes, seal_rows, alignments, gt_rows, baseline_matrices)
    ):
        f4_receipt = row.get("f4")
        source_order = row.get("f4_source_order")
        if not isinstance(f4_receipt, Mapping) or not isinstance(source_order, list):
            raise L2OracleError(f"L2 F4 ledger differs: {scene}")
        f4_path = Path(str(f4_receipt.get("path", "")))
        if _sha(f4_path) != f4_receipt.get("sha256"):
            raise L2OracleError(f"sealed F4 hash differs: {scene}")
        source_map = _source_map(_json(f4_path, f"F4 scene {scene}"), scene)
        ordered_sources = [str(item) for item in source_order]
        if ordered_sources != list(source_map) or len(ordered_sources) != len(set(ordered_sources)):
            raise L2OracleError(f"F4 source order differs: {scene}")
        source_index = {source_id: index for index, source_id in enumerate(ordered_sources)}
        matrices_rows = {name: [] for name in HYPOTHESES}
        for source_id in ordered_sources:
            source = source_map[source_id]
            hypotheses = source.get("hypotheses") if isinstance(source, Mapping) else None
            if not isinstance(hypotheses, Mapping) or set(hypotheses) != set(HYPOTHESES):
                raise L2OracleError(f"F4 geometry bank differs: {source_id}")
            for name in HYPOTHESES:
                matrices_rows[name].append(
                    aligned_iou_matrix(_aligned_geometry(hypotheses[name], name, alignment, f"{source_id}.{name}")[None, :], gt)[0]
                )
                geometry_total += 1
        matrices = {name: np.stack(matrices_rows[name], axis=0) for name in HYPOTHESES}
        source_max = np.maximum.reduce([matrices[name] for name in HYPOTHESES])
        specs = _mode_specs(row, ordered_sources)
        candidate_matrices = {mode: _candidate_matrix(specs[mode], source_index, source_max) for mode in MODES}
        for mode in MODES:
            identities[mode] += len(specs[mode])

        for threshold in THRESHOLDS:
            key = f"{threshold:.2f}"
            official_mask = np.asarray(baseline_evaluations[threshold]["matched_gt_masks"][scene_index], dtype=np.bool_)
            native_pairs = strict_maximum_matching(native, threshold)
            for mode in MODES:
                candidate = candidate_matrices[mode]
                candidate_pairs = strict_maximum_matching(candidate, threshold)
                union_pairs = strict_maximum_matching(np.concatenate((native, candidate), axis=0), threshold)
                suffix_pairs = strict_maximum_matching(candidate, threshold, ~official_mask)
                selected_rows: list[np.ndarray] = []
                selected_receipts: list[dict[str, Any]] = []
                accumulator = accumulators[key][mode]
                for identity_index, gt_index in suffix_pairs:
                    identity_id, source_ids = specs[mode][identity_index]
                    source_id, hypothesis, iou_row = _winner(source_ids, source_index, matrices, gt_index)
                    selected_rows.append(iou_row)
                    accumulator["chosen"][hypothesis] += 1
                    selected_receipts.append({
                        "identity_index": identity_index,
                        "identity_id": identity_id,
                        "source_id": source_id,
                        "hypothesis": hypothesis,
                        "target_gt_index": gt_index,
                        "target_iou": float(iou_row[gt_index]),
                    })
                accumulator["candidate_mm"] += len(candidate_pairs)
                accumulator["union_mm"] += len(union_pairs)
                accumulator["native_mm"] += len(native_pairs)
                accumulator["selected"] += len(suffix_pairs)
                accumulator["suffix"].append(np.stack(selected_rows, axis=0) if selected_rows else _empty_rows(len(gt)))
                accumulator["selection"][scene] = selected_receipts
                accumulator["per_scene"][scene] = {
                    "native_maximum_matching_count": len(native_pairs),
                    "candidate_maximum_matching_count": len(candidate_pairs),
                    "union_maximum_matching_count": len(union_pairs),
                    "additional_union_matching_over_native": len(union_pairs) - len(native_pairs),
                    "gt_selected_suffix_count": len(suffix_pairs),
                }

    seal_counts = seal.get("counts", {}).get("mode_identity_counts", {})
    if any(identities[name] != seal_counts.get(name) for name in MODES):
        raise L2OracleError("L2 oracle mode census differs from no-GT seal")
    if geometry_total != seal.get("counts", {}).get("valid_geometry_count"):
        raise L2OracleError("L2 geometry census differs from no-GT seal")

    per_threshold: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        key = f"{threshold:.2f}"
        baseline = baseline_evaluations[threshold]
        modes: dict[str, Any] = {}
        for mode in MODES:
            accumulator = accumulators[key][mode]
            combined = [np.concatenate((native, suffix), axis=0) for native, suffix in zip(baseline_matrices, accumulator["suffix"])]
            evaluation = official_constant_evaluate(combined, gt_counts, threshold)
            delta = float(evaluation["ap_points"]) - float(baseline["ap_points"])
            additional = int(accumulator["union_mm"] - accumulator["native_mm"])
            modes[mode] = {
                "identity_unit": "F3_track" if mode.startswith("T") else "F4_source",
                "identity_definition": seal["modes"][mode],
                "source_can_match_at_most_one_gt": True,
                "candidate_maximum_matching_count": int(accumulator["candidate_mm"]),
                "union_maximum_matching_count": int(accumulator["union_mm"]),
                "native_maximum_matching_count": int(accumulator["native_mm"]),
                "additional_union_matching_over_native": additional,
                "required_additional_union_matches": REQUIRED_ADDITIONAL_MATCHES,
                "passes_144_match_capacity_gate": additional >= REQUIRED_ADDITIONAL_MATCHES,
                "gt_selected_candidate_suffix": {
                    "oracle_only": True,
                    "deployable": False,
                    "native_prefix_unchanged": True,
                    "formal_score": 1.0,
                    "selected_identity_count": int(accumulator["selected"]),
                    "selected_hypothesis_counts": _normalized_hypotheses(accumulator["chosen"]),
                    "official_evaluation": _evaluation_json(evaluation),
                    "delta_ap_points": delta,
                    "passes_plus10_ap": delta >= TARGET_DELTA_AP_POINTS,
                    "per_scene_selection": accumulator["selection"],
                },
                "per_scene": accumulator["per_scene"],
            }
        per_threshold[key] = {
            "iou_threshold": threshold,
            "strict_iou_comparison": ">",
            "baseline_official_constant_score": _evaluation_json(baseline),
            "modes": modes,
        }

    endpoint_reproduction: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        key = f"{threshold:.2f}"
        actual_t2 = _endpoint_projection(per_threshold[key]["modes"]["T2"])
        expected_t2 = _expected_l1_projection(l1["per_threshold"][key])
        actual_sraw = _endpoint_projection(per_threshold[key]["modes"]["SRAW"])
        expected_sraw = _expected_f4_projection(f4["per_threshold"][key]["identity_constrained_g4"])
        t2_pass = _same_scalars(actual_t2, expected_t2)
        sraw_pass = _same_scalars(actual_sraw, expected_sraw)
        endpoint_reproduction[key] = {"T2_reproduces_L1": t2_pass, "SRAW_reproduces_F4_G4": sraw_pass, "T2_actual": actual_t2, "T2_expected": expected_t2, "SRAW_actual": actual_sraw, "SRAW_expected": expected_sraw}
        if not t2_pass or not sraw_pass:
            raise L2OracleError(f"L2 endpoint reproduction failed at IoU {key}")

    attribution: dict[str, Any] = {}
    for key, threshold_report in per_threshold.items():
        modes = threshold_report["modes"]
        delta = {name: float(modes[name]["gt_selected_candidate_suffix"]["delta_ap_points"]) for name in MODES}
        capacity = {name: int(modes[name]["additional_union_matching_over_native"]) for name in MODES}
        attribution[key] = {
            "delta_ap_points_by_mode": delta,
            "additional_union_matches_by_mode": capacity,
            "source_identity_split_min2_minus_track_min2": {"delta_ap_points": delta["S2"] - delta["T2"], "additional_union_matches": capacity["S2"] - capacity["T2"]},
            "single_view_tracks_minus_min2_tracks": {"delta_ap_points": delta["T1"] - delta["T2"], "additional_union_matches": capacity["T1"] - capacity["T2"]},
            "source_identity_split_all_tracks_minus_track_all": {"delta_ap_points": delta["S1R"] - delta["T1"], "additional_union_matches": capacity["S1R"] - capacity["T1"]},
            "one_view_retained_sources_minus_min2_sources": {"delta_ap_points": delta["S1R"] - delta["S2"], "additional_union_matches": capacity["S1R"] - capacity["S2"]},
            "raw_unretained_sources_minus_retained_sources": {"delta_ap_points": delta["SRAW"] - delta["S1R"], "additional_union_matches": capacity["SRAW"] - capacity["S1R"]},
        }

    ap50 = per_threshold["0.50"]["modes"]
    first_passing = next((name for name in MODES if ap50[name]["gt_selected_candidate_suffix"]["passes_plus10_ap"] and ap50[name]["passes_144_match_capacity_gate"]), None)
    return {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "oracle_only": True,
        "deployable": False,
        "seal": {"path": os.fspath(seal_path.resolve()), "sha256": _sha(seal_path)},
        "endpoint_reports": {"l1_path": os.fspath(l1_report_path.resolve()), "l1_sha256": _sha(l1_report_path), "f4_path": os.fspath(f4_report_path.resolve()), "f4_sha256": _sha(f4_report_path)},
        "counts": {"scene_count": 100, "gt_count": sum(gt_counts), "mode_identity_counts": identities, "per_view_geometry_count": geometry_total},
        "per_threshold": per_threshold,
        "endpoint_reproduction": endpoint_reproduction,
        "attribution": attribution,
        "decision": {
            "endpoint_reproduction_passed": True,
            "ap50_first_mode_passing_plus10_and_144_match_gate": first_passing,
            "authorize_active_birth": False,
            "authorize_accuracy_claim": False,
            "next_step": "freeze_gt_free_past_only_source_preserving_selector" if first_passing else "F4_source_universe_below_AP50_target",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, default=ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt")
    parser.add_argument("--seal", type=Path, default=DEFAULT_SEAL)
    parser.add_argument("--baseline-root", type=Path, default=ROOT / "results/scannet_t05_boxer_replay_active_score05")
    parser.add_argument("--gt-root", type=Path, default=ROOT / "evaluation/data_util/scannet_train_detection_data")
    parser.add_argument("--scan-root", type=Path, default=Path("/extra/ZhaoX/scannet_data/scans"))
    parser.add_argument("--l1-report", type=Path, default=DEFAULT_L1)
    parser.add_argument("--f4-report", type=Path, default=DEFAULT_F4)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        raise L2OracleError(f"refusing to overwrite L2 oracle report: {args.out}")
    report = audit(scene_list=args.scene_list, seal_path=args.seal, baseline_root=args.baseline_root, gt_root=args.gt_root, scan_root=args.scan_root, l1_report_path=args.l1_report, f4_report_path=args.f4_report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    summary = {key: {mode: threshold["modes"][mode]["gt_selected_candidate_suffix"]["official_evaluation"]["ap_points"] for mode in MODES} for key, threshold in report["per_threshold"].items()}
    print(json.dumps({"out": os.fspath(args.out), "decision": report["decision"], "ap_points": summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
