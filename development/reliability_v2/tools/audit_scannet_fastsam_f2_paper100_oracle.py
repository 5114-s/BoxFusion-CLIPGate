#!/usr/bin/env python3
"""Identity-constrained paper100 oracle for the FastSAM F2 hypotheses.

F2 is a shadow geometry experiment.  Every sealed F0 source owns exactly three
effective geometries (H0, HL, HLG).  This audit treats the *source*, rather than
the geometry hypothesis, as the proposal identity.  Consequently a source can
match at most one GT even when several of its hypotheses cross an IoU
threshold.  Ground truth is used only by this offline audit; no prediction or
birth artifact is written.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(_REPOSITORY_ROOT))

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
    EXPECTED,
    EXPECTED_BASELINE_AP_POINTS,
    F0_SCENE_SCHEMA,
    GEOMETRY_ATOL,
    GEOMETRY_RTOL,
    REQUIRED_ADDITIONAL_MATCHES,
    TARGET_DELTA_AP_POINTS,
    THRESHOLDS,
    _input_snapshot as _f1_input_snapshot,
    _json_evaluation,
    _load_f0_candidates,
    _read_json,
    _regular_file,
    _sha256,
    _threshold_key,
    _validate_frozen_hashes,
    _validate_receipt as _validate_f0_receipt,
    canonical_ordered_hash_ledger,
)


SCHEMA = "boxfusion.scannet_fastsam_f2_paper100_oracle.v1"
F2_SCENE_SCHEMA = "boxfusion.scannet_fastsam_f2_paper100.scene.v1"
F2_RECEIPT_SCHEMA = "boxfusion.scannet_fastsam_f2_paper100.merge.v1"
F2_PROTOCOL_ID = "F2-DFU-LGF-lite-shadow-paper100"
F1_REPORT_SHA256 = "05fc3b740126fcc8ac83ac335cf62df85b8ebd99b9033d0fc452e52229105304"
HYPOTHESES = ("H0", "HL", "HLG")
F1_AP50_ADDITIONAL_UNION_MATCHES = 63
F2_MIN_AP50_ADDITIONAL_MATCH_GAIN = 15
F2_RETAIN_AP50_ADDITIONAL_UNION_MATCHES = (
    F1_AP50_ADDITIONAL_UNION_MATCHES + F2_MIN_AP50_ADDITIONAL_MATCH_GAIN
)
REQUIRED_RUNTIME_GATES = (
    "provider_runtime_p95_ms",
    "complete_runtime_p95_ms",
    "complete_runtime_max_ms",
    "amortized_complete_ms_per_source_frame",
    "amortized_f2_core_ms_per_source_frame",
    "gpu_peak_memory_bytes",
)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SIGNS = np.asarray(
    [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=bool
)


class F2OracleError(ValueError):
    """Raised when an input violates the frozen F2 oracle contract."""


@dataclass(frozen=True)
class F2Source:
    scene_id: str
    frame_id: int
    frame_ordinal: int
    candidate_index: int
    rank: int
    raw_index: int
    source_id: str
    world_minmax: Mapping[str, np.ndarray]
    aligned_minmax: Mapping[str, np.ndarray]
    applied: Mapping[str, bool]


def _vector3(value: Any, label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise F2OracleError(f"{label} must be a numeric length-3 vector") from error
    if array.shape != (3,) or not np.isfinite(array).all():
        raise F2OracleError(f"{label} must be a finite length-3 vector")
    return array


def _world_and_aligned_minmax(
    hypothesis: Mapping[str, Any], alignment: np.ndarray, label: str
) -> tuple[np.ndarray, np.ndarray]:
    if hypothesis.get("valid") is not True:
        raise F2OracleError(f"{label} must contain fail-open effective geometry")
    q02 = _vector3(hypothesis.get("q02"), f"{label}.q02")
    q98 = _vector3(hypothesis.get("q98"), f"{label}.q98")
    center = _vector3(hypothesis.get("center"), f"{label}.center")
    extent = _vector3(hypothesis.get("extent"), f"{label}.extent")
    if np.any(q98 <= q02):
        raise F2OracleError(f"{label} requires q98 > q02")
    if not np.allclose(center, (q02 + q98) / 2.0, rtol=GEOMETRY_RTOL, atol=GEOMETRY_ATOL):
        raise F2OracleError(f"{label}.center is inconsistent with q02/q98")
    if not np.allclose(extent, q98 - q02, rtol=GEOMETRY_RTOL, atol=GEOMETRY_ATOL):
        raise F2OracleError(f"{label}.extent is inconsistent with q02/q98")
    matrix = np.asarray(alignment, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise F2OracleError("axis alignment must be a finite 4x4 matrix")
    corners = np.where(_SIGNS, q98[None, :], q02[None, :])
    transformed = corners @ matrix[:3, :3].T + matrix[:3, 3]
    return np.concatenate((q02, q98)), np.concatenate(
        (transformed.min(axis=0), transformed.max(axis=0))
    )


def grouped_iou_matrix(
    matrices: Mapping[str, np.ndarray], hypotheses: Sequence[str] = HYPOTHESES
) -> np.ndarray:
    """Collapse hypothesis edges to one row per source via maximum IoU."""

    if tuple(matrices) != tuple(hypotheses):
        raise F2OracleError(
            f"hypothesis order must be exactly {tuple(hypotheses)}, got {tuple(matrices)}"
        )
    arrays = [np.asarray(matrices[name], dtype=np.float64) for name in hypotheses]
    if not arrays or any(array.ndim != 2 or not np.isfinite(array).all() for array in arrays):
        raise F2OracleError("hypothesis IoU matrices must be finite and two-dimensional")
    if len({array.shape for array in arrays}) != 1:
        raise F2OracleError("hypothesis IoU matrices must have identical shapes")
    return np.maximum.reduce(arrays)


def choose_hypothesis_for_edge(
    matrices: Mapping[str, np.ndarray], source_index: int, gt_index: int
) -> tuple[str, float]:
    """Choose the best geometry for one matched edge, preferring H0 on ties."""

    values = [float(matrices[name][source_index, gt_index]) for name in HYPOTHESES]
    best = int(np.argmax(np.asarray(values, dtype=np.float64)))
    return HYPOTHESES[best], values[best]


def _edge_delta(h0: np.ndarray, candidate: np.ndarray, threshold: float) -> dict[str, int]:
    old = np.asarray(h0) > threshold
    new = np.asarray(candidate) > threshold
    if old.shape != new.shape:
        raise F2OracleError("edge comparison shape mismatch")
    gain = new & ~old
    loss = old & ~new
    return {
        "gained_gt_edges": int(np.count_nonzero(gain)),
        "lost_gt_edges": int(np.count_nonzero(loss)),
        "retained_gt_edges": int(np.count_nonzero(old & new)),
        "sources_with_any_gained_edge": int(np.count_nonzero(np.any(gain, axis=1))) if gain.ndim == 2 else 0,
        "sources_with_any_lost_edge": int(np.count_nonzero(np.any(loss, axis=1))) if loss.ndim == 2 else 0,
    }


def _empty_rows(columns: int) -> np.ndarray:
    return np.empty((0, columns), dtype=np.float64)


def evaluate_f2_threshold(
    *,
    scenes: Sequence[str],
    native_iou: Sequence[np.ndarray],
    hypothesis_iou: Sequence[Mapping[str, np.ndarray]],
    sources: Sequence[Sequence[F2Source]],
    gt_counts: Sequence[int],
    baseline_evaluation: Mapping[str, Any],
    threshold: float,
) -> dict[str, Any]:
    """Evaluate H0/HL/HLG and the identity-constrained grouped oracle."""

    if len({len(scenes), len(native_iou), len(hypothesis_iou), len(sources), len(gt_counts)}) != 1:
        raise F2OracleError("per-scene F2 inputs have inconsistent lengths")
    baseline_masks = baseline_evaluation.get("matched_gt_masks")
    if not isinstance(baseline_masks, list) or len(baseline_masks) != len(scenes):
        raise F2OracleError("official baseline masks have inconsistent scene count")

    names = (*HYPOTHESES, "GROUPED")
    accumulators: dict[str, dict[str, Any]] = {
        name: {
            "candidate_mm": 0,
            "union_mm": 0,
            "native_mm": 0,
            "selected": 0,
            "suffix": [],
            "selection": {},
            "per_scene": {},
            "chosen": {hypothesis: 0 for hypothesis in HYPOTHESES},
        }
        for name in names
    }
    edge_delta = {
        name: {
            "gained_gt_edges": 0,
            "lost_gt_edges": 0,
            "retained_gt_edges": 0,
            "sources_with_any_gained_edge": 0,
            "sources_with_any_lost_edge": 0,
        }
        for name in ("HL", "HLG", "GROUPED")
    }

    for scene, native, matrices, scene_sources, gt_count, official_mask in zip(
        scenes, native_iou, hypothesis_iou, sources, gt_counts, baseline_masks
    ):
        if tuple(matrices) != HYPOTHESES:
            raise F2OracleError(f"hypothesis order mismatch: {scene}")
        if native.ndim != 2 or native.shape[1] != gt_count:
            raise F2OracleError(f"native IoU/GT shape mismatch: {scene}")
        for hypothesis in HYPOTHESES:
            if matrices[hypothesis].shape != (len(scene_sources), gt_count):
                raise F2OracleError(f"{hypothesis} IoU/source/GT shape mismatch: {scene}")
        grouped = grouped_iou_matrix(matrices)
        evaluated = {**matrices, "GROUPED": grouped}
        h0 = matrices["H0"]
        for name in ("HL", "HLG", "GROUPED"):
            delta = _edge_delta(h0, evaluated[name], threshold)
            for key, value in delta.items():
                edge_delta[name][key] += value

        native_pairs = strict_maximum_matching(native, threshold)
        official_unmatched = ~np.asarray(official_mask, dtype=bool)
        for name in names:
            matrix = evaluated[name]
            candidate_pairs = strict_maximum_matching(matrix, threshold)
            union_pairs = strict_maximum_matching(
                np.concatenate((native, matrix), axis=0), threshold
            )
            suffix_pairs = strict_maximum_matching(matrix, threshold, official_unmatched)
            target_by_source = {source_index: gt_index for source_index, gt_index in suffix_pairs}
            selected_indices = sorted(target_by_source)
            selected_matrices: list[np.ndarray] = []
            selected_rows: list[dict[str, Any]] = []
            for source_index in selected_indices:
                gt_index = target_by_source[source_index]
                if name == "GROUPED":
                    chosen, chosen_iou = choose_hypothesis_for_edge(
                        matrices, source_index, gt_index
                    )
                else:
                    chosen = name
                    chosen_iou = float(matrix[source_index, gt_index])
                selected_matrices.append(matrices[chosen][source_index])
                accumulators[name]["chosen"][chosen] += 1
                selected_rows.append(
                    {
                        "source_id": scene_sources[source_index].source_id,
                        "source_index": source_index,
                        "chosen_hypothesis": chosen,
                        "target_gt_index": gt_index,
                        "target_iou": chosen_iou,
                    }
                )
            suffix = (
                np.stack(selected_matrices)
                if selected_matrices
                else _empty_rows(gt_count)
            )
            accumulators[name]["candidate_mm"] += len(candidate_pairs)
            accumulators[name]["union_mm"] += len(union_pairs)
            accumulators[name]["native_mm"] += len(native_pairs)
            accumulators[name]["selected"] += len(selected_indices)
            accumulators[name]["suffix"].append(suffix)
            accumulators[name]["selection"][scene] = selected_rows
            accumulators[name]["per_scene"][scene] = {
                "native_maximum_matching_count": len(native_pairs),
                "candidate_maximum_matching_count": len(candidate_pairs),
                "union_maximum_matching_count": len(union_pairs),
                "additional_union_matching_over_native": len(union_pairs) - len(native_pairs),
                "gt_selected_suffix_count": len(selected_indices),
            }

    baseline_ap = float(baseline_evaluation["ap_points"])
    reports: dict[str, Any] = {}
    for name in names:
        acc = accumulators[name]
        combined = [
            np.concatenate((native, suffix), axis=0)
            for native, suffix in zip(native_iou, acc["suffix"])
        ]
        evaluation = official_constant_evaluate(combined, gt_counts, threshold)
        delta_ap = float(evaluation["ap_points"]) - baseline_ap
        additional = int(acc["union_mm"] - acc["native_mm"])
        report: dict[str, Any] = {
            "identity_unit": "sealed_F0_source",
            "source_can_match_at_most_one_gt": True,
            "candidate_maximum_matching_count": int(acc["candidate_mm"]),
            "union_maximum_matching_count": int(acc["union_mm"]),
            "native_maximum_matching_count": int(acc["native_mm"]),
            "additional_union_matching_over_native": additional,
            "required_additional_union_matches": REQUIRED_ADDITIONAL_MATCHES,
            "passes_geometry_capacity": additional >= REQUIRED_ADDITIONAL_MATCHES,
            "gt_selected_candidate_suffix": {
                "oracle_only": True,
                "deployable": False,
                "threshold_specific": True,
                "selection": "source_maximum_matching_to_official_native_greedy_unmatched_gt",
                "native_rows_are_unchanged_scene_prefix": True,
                "formal_score": 1.0,
                "selected_source_count": int(acc["selected"]),
                "chosen_hypothesis_counts": acc["chosen"],
                "official_evaluation": _json_evaluation(evaluation, scenes),
                "delta_ap_points": delta_ap,
                "passes_plus10_ap": delta_ap >= TARGET_DELTA_AP_POINTS,
                "per_scene_selection": acc["selection"],
            },
            "per_scene": acc["per_scene"],
        }
        if name in edge_delta:
            report["edges_vs_H0"] = edge_delta[name]
        reports[name] = report

    return {
        "iou_threshold": threshold,
        "strict_iou_comparison": ">",
        "baseline_official_constant_score": _json_evaluation(baseline_evaluation, scenes),
        "hypothesis_only": {name: reports[name] for name in HYPOTHESES},
        "identity_constrained_grouped": reports["GROUPED"],
    }


def validate_h0_reproduces_f1(
    per_threshold: Mapping[str, Any], f1_report: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed unless replayed H0 reproduces every published F1 scalar."""

    checks: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        key = _threshold_key(threshold)
        actual = per_threshold[key]["hypothesis_only"]["H0"]
        expected = f1_report["per_threshold"][key]
        scalar_pairs = {
            "candidate_maximum_matching_count": (
                actual["candidate_maximum_matching_count"],
                expected["candidate_maximum_matching_count"],
            ),
            "union_maximum_matching_count": (
                actual["union_maximum_matching_count"],
                expected["union_maximum_matching_count"],
            ),
            "additional_union_matching_over_native": (
                actual["additional_union_matching_over_native"],
                expected["additional_union_matching_over_native"],
            ),
            "selected_source_count": (
                actual["gt_selected_candidate_suffix"]["selected_source_count"],
                expected["gt_selected_candidate_suffix"]["selected_candidate_count"],
            ),
        }
        numeric_pairs = {
            "suffix_ap_points": (
                actual["gt_selected_candidate_suffix"]["official_evaluation"]["ap_points"],
                expected["gt_selected_candidate_suffix"]["official_evaluation"]["ap_points"],
            ),
            "delta_ap_points": (
                actual["gt_selected_candidate_suffix"]["delta_ap_points"],
                expected["gt_selected_candidate_suffix"]["delta_ap_points"],
            ),
        }
        scalar_pass = all(left == right for left, right in scalar_pairs.values())
        numeric_pass = all(
            math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
            for left, right in numeric_pairs.values()
        )
        checks[key] = {
            "passed": scalar_pass and numeric_pass,
            "integer_checks": {
                name: {"actual": left, "expected": right, "passed": left == right}
                for name, (left, right) in scalar_pairs.items()
            },
            "numeric_checks": {
                name: {
                    "actual": float(left),
                    "expected": float(right),
                    "absolute_error": abs(float(left) - float(right)),
                    "passed": math.isclose(
                        float(left), float(right), rel_tol=0.0, abs_tol=1e-12
                    ),
                }
                for name, (left, right) in numeric_pairs.items()
            },
        }
        if not checks[key]["passed"]:
            raise F2OracleError(f"H0 failed to reproduce sealed F1 at IoU {key}")
    return checks


def f2_retention_decision(
    *,
    grouped_ap50_additional_union_matches: int,
    h0_identity_passed: bool,
    merged_runtime_overall_pass: bool,
    later_joint_active_capacity_passed: bool,
) -> dict[str, Any]:
    """Apply the frozen F2 retention gate without authorizing active birth."""

    if type(grouped_ap50_additional_union_matches) is not int:
        raise F2OracleError("grouped AP50 additional union matches must be an integer")
    if type(h0_identity_passed) is not bool or type(merged_runtime_overall_pass) is not bool:
        raise F2OracleError("F2 identity/runtime gates must be booleans")
    gain = (
        grouped_ap50_additional_union_matches - F1_AP50_ADDITIONAL_UNION_MATCHES
    )
    capacity_pass = (
        grouped_ap50_additional_union_matches
        >= F2_RETAIN_AP50_ADDITIONAL_UNION_MATCHES
    )
    retain = capacity_pass and h0_identity_passed and merged_runtime_overall_pass
    result = {
        "f1_ap50_additional_union_matches": F1_AP50_ADDITIONAL_UNION_MATCHES,
        "required_ap50_additional_match_gain_over_f1": F2_MIN_AP50_ADDITIONAL_MATCH_GAIN,
        "required_grouped_ap50_additional_union_matches": F2_RETAIN_AP50_ADDITIONAL_UNION_MATCHES,
        "actual_grouped_ap50_additional_union_matches": grouped_ap50_additional_union_matches,
        "actual_ap50_additional_match_gain_over_f1": gain,
        "ap50_retention_capacity_passed": capacity_pass,
        "h0_identity_passed": h0_identity_passed,
        "merged_runtime_overall_pass": merged_runtime_overall_pass,
        "retain_f2_geometry_for_f3": retain,
        "authorize_f3_projection_self_validation_shadow": True,
        "f3_shadow_geometry_input": (
            "F2_H0_HL_HLG_hypothesis_set_for_GT_free_F3_selection"
            if retain
            else "F1_H0_only"
        ),
        "grouped_oracle_geometry_exported": False,
        "authorize_grouped_oracle_geometry": False,
        "later_f2_f3_joint_active_birth_capacity_gate_passed_diagnostic": later_joint_active_capacity_passed,
        "authorize_active_birth": False,
        "result": (
            "f2_retain_for_f3_shadow"
            if retain
            else "f2_discard_geometry_continue_f3_shadow_from_h0"
        ),
    }
    if result["grouped_oracle_geometry_exported"] or result["authorize_grouped_oracle_geometry"]:
        raise AssertionError("identity-constrained GROUPED oracle must never be exported")
    return result


def _receipt_entry(row: Mapping[str, Any], key: str, scene: str) -> Mapping[str, Any]:
    value = row.get(key)
    if not isinstance(value, dict) or _HASH_RE.fullmatch(str(value.get("sha256"))) is None:
        raise F2OracleError(f"F2 receipt {key} seal missing: {scene}")
    return value


def _validate_runtime_gates(
    receipt: Mapping[str, Any], overall_pass: bool
) -> dict[str, Any]:
    gates = receipt.get("gates")
    if not isinstance(gates, dict) or any(name not in gates for name in REQUIRED_RUNTIME_GATES):
        raise F2OracleError("F2 merged receipt is missing required runtime gates")
    runtime_summary = gates.get("runtime")
    recorded_gate_names = (
        runtime_summary.get("gate_names") if isinstance(runtime_summary, dict) else None
    )
    if (
        not isinstance(runtime_summary, dict)
        or type(runtime_summary.get("overall_pass")) is not bool
        or recorded_gate_names != list(REQUIRED_RUNTIME_GATES)
        or type(gates.get("overall_pass")) is not bool
        or gates.get("overall_pass") is not overall_pass
    ):
        raise F2OracleError("F2 merged runtime/overall gate summary is inconsistent")
    comparisons = {
        "<": lambda actual, threshold: actual < threshold,
        "<=": lambda actual, threshold: actual <= threshold,
        ">": lambda actual, threshold: actual > threshold,
        ">=": lambda actual, threshold: actual >= threshold,
        "==": lambda actual, threshold: actual == threshold,
    }
    result: dict[str, Mapping[str, Any]] = {}
    for name, value in gates.items():
        if name in {"runtime", "overall_pass"}:
            continue
        if not isinstance(value, dict):
            raise F2OracleError(f"F2 gate must be an object: {name}")
        actual = value.get("actual")
        threshold = value.get("threshold")
        comparator = value.get("comparator")
        passed = value.get("passed")
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isfinite(float(actual))
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or comparator not in comparisons
            or type(passed) is not bool
        ):
            raise F2OracleError(f"invalid F2 runtime gate: {name}")
        recomputed = comparisons[str(comparator)](float(actual), float(threshold))
        if recomputed is not passed:
            raise F2OracleError(f"inconsistent F2 runtime gate result: {name}")
        result[name] = value
    runtime_recomputed = all(bool(result[name]["passed"]) for name in REQUIRED_RUNTIME_GATES)
    if runtime_summary["overall_pass"] is not runtime_recomputed:
        raise F2OracleError("F2 runtime overall_pass is inconsistent with runtime gates")
    if overall_pass != all(bool(row["passed"]) for row in result.values()):
        raise F2OracleError("F2 overall_pass is inconsistent with merged runtime gates")
    return {
        "items": result,
        "runtime": runtime_summary,
        "overall_pass": overall_pass,
    }


def _validate_f2_receipt(
    receipt: Mapping[str, Any],
    scenes: Sequence[str],
    *,
    expected_scene_count: int = EXPECTED["scene_count"],
    expected_keyframe_count: int = EXPECTED["keyframe_count"],
    expected_successful_frame_count: int = EXPECTED["successful_frame_count"],
    expected_source_count: int = EXPECTED["candidate_count"],
) -> tuple[dict[str, Mapping[str, Any]], str, bool, dict[str, Any]]:
    for key, expected in {
        "schema": F2_RECEIPT_SCHEMA,
        "protocol_id": F2_PROTOCOL_ID,
        "complete": True,
    }.items():
        if receipt.get(key) != expected:
            raise F2OracleError(f"F2 receipt contract mismatch for {key}")
    overall_pass = receipt.get("overall_pass")
    if type(overall_pass) is not bool:
        raise F2OracleError("F2 receipt overall_pass must be boolean")
    runtime_gates = _validate_runtime_gates(receipt, overall_pass)
    contracts = receipt.get("contracts")
    required_contracts = {
        "shadow_only": True,
        "birth_enabled": False,
        "ground_truth_access": False,
        "prediction_access": False,
        "evaluator_access": False,
        "native_output_mutation": False,
        "training": False,
        "f0_exact_replay_required": True,
    }
    if not isinstance(contracts, dict) or any(
        contracts.get(key) != expected for key, expected in required_contracts.items()
    ):
        raise F2OracleError("F2 receipt shadow/training-free contracts mismatch")
    signature = receipt.get("run_signature_sha256")
    if not isinstance(signature, str) or _HASH_RE.fullmatch(signature) is None:
        raise F2OracleError("F2 receipt run signature is missing")
    coverage = receipt.get("coverage")
    if (
        len(scenes) != expected_scene_count
        or not isinstance(coverage, dict)
        or coverage.get("scene_count") != expected_scene_count
        or coverage.get("scene_order") != list(scenes)
    ):
        raise F2OracleError("F2 receipt paper100 coverage/order mismatch")
    totals = receipt.get("totals")
    required_totals = {
        "keyframe_count": expected_keyframe_count,
        "successful_frame_count": expected_successful_frame_count,
        "source_count": expected_source_count,
        "identity_verified_source_count": expected_source_count,
    }
    if not isinstance(totals, dict) or any(totals.get(k) != v for k, v in required_totals.items()):
        raise F2OracleError("F2 receipt census/identity totals mismatch")
    rows = receipt.get("scenes")
    if not isinstance(rows, list) or len(rows) != len(scenes):
        raise F2OracleError("F2 receipt scene ledger mismatch")
    result: dict[str, Mapping[str, Any]] = {}
    for index, (scene, row) in enumerate(zip(scenes, rows)):
        if not isinstance(row, dict) or row.get("scene_id") != scene or row.get("scene_index") != index:
            raise F2OracleError("F2 receipt scene order mismatch")
        _receipt_entry(row, "sidecar", scene)
        _receipt_entry(row, "evidence_npz", scene)
        result[scene] = row
    return result, signature, overall_pass, runtime_gates


def _load_f2_sources(
    *,
    path: Path,
    f0_path: Path,
    scene: str,
    scene_index: int,
    alignment: np.ndarray,
    receipt_sidecar_sha256: str,
    run_signature_sha256: str,
) -> tuple[list[F2Source], int, int]:
    if _sha256(_regular_file(path, f"F2 sidecar for {scene}")) != receipt_sidecar_sha256:
        raise F2OracleError(f"F2 receipt sidecar hash mismatch: {scene}")
    payload = _read_json(path, f"F2 sidecar for {scene}")
    for key, expected in {
        "schema": F2_SCENE_SCHEMA,
        "protocol_id": F2_PROTOCOL_ID,
        "complete": True,
        "scene_id": scene,
        "scene_index": scene_index,
        "run_signature_sha256": run_signature_sha256,
    }.items():
        if payload.get(key) != expected:
            raise F2OracleError(f"F2 sidecar contract mismatch: {scene}.{key}")
    f0 = _read_json(f0_path, f"F0 sidecar for {scene}")
    if f0.get("schema") != F0_SCENE_SCHEMA or f0.get("scene_id") != scene:
        raise F2OracleError(f"F0 sidecar contract mismatch while verifying F2: {scene}")
    frames = payload.get("frames")
    f0_frames = f0.get("frames")
    if not isinstance(frames, list) or not isinstance(f0_frames, list) or len(frames) != len(f0_frames):
        raise F2OracleError(f"F2/F0 frame ledger mismatch: {scene}")

    result: list[F2Source] = []
    successful = 0
    for ordinal, (frame, f0_frame) in enumerate(zip(frames, f0_frames)):
        if not isinstance(frame, dict) or not isinstance(f0_frame, dict):
            raise F2OracleError(f"invalid F2/F0 frame row: {scene}:{ordinal}")
        for key in ("frame_id", "frame_ordinal", "successful"):
            if frame.get(key) != f0_frame.get(key):
                raise F2OracleError(f"F2/F0 frame identity mismatch: {scene}:{ordinal}:{key}")
        frame_id = frame.get("frame_id")
        if type(frame_id) is not int or frame.get("frame_ordinal") != ordinal:
            raise F2OracleError(f"invalid F2 frame identity: {scene}:{ordinal}")
        if frame.get("successful") is True:
            successful += 1
            funnel = f0_frame.get("funnel")
            f0_rows = funnel.get("candidates") if isinstance(funnel, dict) else None
            if not isinstance(f0_rows, list):
                raise F2OracleError(f"successful F0 frame lacks candidates: {scene}:{frame_id}")
        else:
            f0_rows = []
        rows = frame.get("sources")
        if not isinstance(rows, list) or len(rows) != len(f0_rows):
            raise F2OracleError(f"F2/F0 source count mismatch: {scene}:{frame_id}")
        for candidate_index, (row, f0_row) in enumerate(zip(rows, f0_rows)):
            if not isinstance(row, dict) or not isinstance(f0_row, dict):
                raise F2OracleError(f"invalid F2 source row: {scene}:{frame_id}:{candidate_index}")
            raw_index = f0_row.get("raw_index")
            expected_source_id = f"{scene}/frame_{frame_id:06d}/raw_{raw_index:03d}"
            expected_identity = {
                "source_id": expected_source_id,
                "candidate_index": candidate_index,
                "rank": f0_row.get("rank"),
                "raw_index": raw_index,
                "mask_sha256": f0_row.get("mask_sha256"),
                "points_and_voxel_keys_sha256": f0_row.get("points_and_voxel_keys_sha256"),
                "f0_world_q02": f0_row.get("world_q02"),
                "f0_world_q98": f0_row.get("world_q98"),
            }
            for key, expected in expected_identity.items():
                if row.get(key) != expected:
                    raise F2OracleError(
                        f"F2/F0 source identity mismatch: {scene}:{frame_id}:{candidate_index}:{key}"
                    )
            hypotheses = row.get("hypotheses")
            if not isinstance(hypotheses, dict) or tuple(hypotheses) != HYPOTHESES:
                raise F2OracleError(f"F2 hypothesis order mismatch: {expected_source_id}")
            world: dict[str, np.ndarray] = {}
            aligned: dict[str, np.ndarray] = {}
            applied: dict[str, bool] = {}
            for name in HYPOTHESES:
                hypothesis = hypotheses[name]
                if not isinstance(hypothesis, dict):
                    raise F2OracleError(f"invalid {name} hypothesis: {expected_source_id}")
                world[name], aligned[name] = _world_and_aligned_minmax(
                    hypothesis, alignment, f"{expected_source_id}.{name}"
                )
                diagnostics = hypothesis.get("diagnostics")
                if not isinstance(diagnostics, dict) or type(diagnostics.get("applied")) is not bool:
                    raise F2OracleError(f"missing {name} applied diagnostic: {expected_source_id}")
                applied[name] = diagnostics["applied"]
            expected_h0 = np.asarray(
                [*f0_row["world_q02"], *f0_row["world_q98"]], dtype=np.float64
            )
            if not np.array_equal(world["H0"], expected_h0):
                raise F2OracleError(f"H0 does not bitwise reproduce F0: {expected_source_id}")
            result.append(
                F2Source(
                    scene_id=scene,
                    frame_id=frame_id,
                    frame_ordinal=ordinal,
                    candidate_index=candidate_index,
                    rank=int(row["rank"]),
                    raw_index=int(row["raw_index"]),
                    source_id=expected_source_id,
                    world_minmax=world,
                    aligned_minmax=aligned,
                    applied=applied,
                )
            )
    return result, len(frames), successful


def _f2_snapshot(
    *,
    scenes: Sequence[str],
    scene_list: Path,
    full_scene_list: Path,
    f0_receipt: Path,
    f0_sidecar_root: Path,
    f1_report: Path,
    f2_receipt: Path,
    f2_sidecar_root: Path,
    f2_array_root: Path,
    baseline_root: Path,
    gt_root: Path,
    scan_root: Path,
    official_evaluator: Path,
) -> dict[str, Any]:
    base = _f1_input_snapshot(
        scenes=scenes,
        scene_list=scene_list,
        full_scene_list=full_scene_list,
        f0_receipt=f0_receipt,
        official_evaluator=official_evaluator,
        sidecar_root=f0_sidecar_root,
        baseline_root=baseline_root,
        gt_root=gt_root,
        scan_root=scan_root,
    )
    base["fixed_files"].update(
        {
            "f1_report": {"path": os.fspath(f1_report), "sha256": _sha256(_regular_file(f1_report, "F1 report"))},
            "f2_receipt": {"path": os.fspath(f2_receipt), "sha256": _sha256(_regular_file(f2_receipt, "F2 receipt"))},
        }
    )
    base["ordered_scene_ledgers"].update(
        {
            "f2_sidecars": canonical_ordered_hash_ledger(
                scenes, [f2_sidecar_root / f"{scene}.json" for scene in scenes], "F2 sidecar"
            ),
            "f2_evidence_npz": canonical_ordered_hash_ledger(
                scenes, [f2_array_root / f"{scene}.npz" for scene in scenes], "F2 evidence NPZ"
            ),
        }
    )
    return base


def audit_scannet_fastsam_f2_paper100_oracle(
    *,
    scene_list: Path,
    full_scene_list: Path,
    f0_receipt: Path,
    f0_sidecar_root: Path,
    f1_report: Path,
    f2_receipt: Path,
    f2_sidecar_root: Path,
    f2_array_root: Path,
    baseline_root: Path,
    gt_root: Path,
    scan_root: Path,
    official_evaluator: Path,
) -> dict[str, Any]:
    scenes = load_scene_list(_regular_file(scene_list, "paper100 scene list"))
    full_scenes = load_scene_list(_regular_file(full_scene_list, "F0 full200 scene list"))
    if len(scenes) != EXPECTED["scene_count"] or scenes != full_scenes[: EXPECTED["scene_count"]]:
        raise F2OracleError("frozen paper100 scene order/count mismatch")
    if _sha256(_regular_file(f1_report, "F1 report")) != F1_REPORT_SHA256:
        raise F2OracleError("sealed F1 report SHA-256 mismatch")
    f1 = _read_json(f1_report, "F1 report")
    if f1.get("schema") != "boxfusion.scannet_fastsam_f1_paper100_oracle.v1":
        raise F2OracleError("unexpected F1 report schema")
    f2_receipt_payload = _read_json(f2_receipt, "F2 receipt")
    (
        f2_receipt_rows,
        f2_signature,
        f2_runtime_overall_pass,
        f2_runtime_gates,
    ) = _validate_f2_receipt(f2_receipt_payload, scenes)

    before = _f2_snapshot(
        scenes=scenes,
        scene_list=scene_list,
        full_scene_list=full_scene_list,
        f0_receipt=f0_receipt,
        f0_sidecar_root=f0_sidecar_root,
        f1_report=f1_report,
        f2_receipt=f2_receipt,
        f2_sidecar_root=f2_sidecar_root,
        f2_array_root=f2_array_root,
        baseline_root=baseline_root,
        gt_root=gt_root,
        scan_root=scan_root,
        official_evaluator=official_evaluator,
    )
    _validate_frozen_hashes(before)
    f0_receipt_rows = _validate_f0_receipt(
        _read_json(f0_receipt, "F0 receipt"), full_scenes
    )

    gt_counts: list[int] = []
    native_iou: list[np.ndarray] = []
    hypothesis_iou: list[dict[str, np.ndarray]] = []
    sources_by_scene: list[list[F2Source]] = []
    scene_reports: dict[str, Any] = {}
    totals = {
        "scene_count": len(scenes),
        "keyframe_count": 0,
        "successful_frame_count": 0,
        "source_count": 0,
        "native_prediction_count": 0,
        "gt_count": 0,
    }
    changed = {name: 0 for name in HYPOTHESES}
    applied = {name: 0 for name in HYPOTHESES}
    for scene_index, scene in enumerate(scenes):
        alignment = load_axis_alignment(scan_root / scene / f"{scene}.txt")
        gt = load_gt_minmax(gt_root / f"{scene}_bbox.npy")
        _, native = load_baseline_boxes(baseline_root / f"{scene}_boxes.pkl", alignment)
        f0_rows, _, _ = _load_f0_candidates(
            path=f0_sidecar_root / f"{scene}.json",
            scene=scene,
            scene_index=scene_index,
            alignment=alignment,
            receipt_sidecar_sha256=str(f0_receipt_rows[scene]["sidecar"]["sha256"]),
        )
        receipt_row = f2_receipt_rows[scene]
        evidence_entry = receipt_row["evidence_npz"]
        evidence_path = f2_array_root / f"{scene}.npz"
        if _sha256(_regular_file(evidence_path, f"F2 evidence NPZ for {scene}")) != str(
            evidence_entry["sha256"]
        ):
            raise F2OracleError(f"F2 receipt evidence NPZ hash mismatch: {scene}")
        sources, keyframes, successful = _load_f2_sources(
            path=f2_sidecar_root / f"{scene}.json",
            f0_path=f0_sidecar_root / f"{scene}.json",
            scene=scene,
            scene_index=scene_index,
            alignment=alignment,
            receipt_sidecar_sha256=str(receipt_row["sidecar"]["sha256"]),
            run_signature_sha256=f2_signature,
        )
        if len(sources) != len(f0_rows):
            raise F2OracleError(f"F2 source census does not reproduce F0: {scene}")
        for source, f0_source in zip(sources, f0_rows):
            if (
                source.frame_id != f0_source.frame_id
                or source.candidate_index != f0_source.candidate_index
                or source.rank != f0_source.rank
                or source.raw_index != f0_source.raw_index
                or not np.array_equal(source.aligned_minmax["H0"], f0_source.aligned_minmax)
            ):
                raise F2OracleError(f"F2 H0/source order does not reproduce F1: {source.source_id}")
        matrices: dict[str, np.ndarray] = {}
        for name in HYPOTHESES:
            boxes = (
                np.stack([source.aligned_minmax[name] for source in sources])
                if sources
                else np.empty((0, 6), dtype=np.float64)
            )
            matrices[name] = aligned_iou_matrix(boxes, gt)
            changed[name] += sum(
                not np.array_equal(source.world_minmax[name], source.world_minmax["H0"])
                for source in sources
            )
            applied[name] += sum(source.applied[name] for source in sources)
        gt_counts.append(len(gt))
        native_iou.append(aligned_iou_matrix(native, gt))
        hypothesis_iou.append(matrices)
        sources_by_scene.append(sources)
        totals["keyframe_count"] += keyframes
        totals["successful_frame_count"] += successful
        totals["source_count"] += len(sources)
        totals["native_prediction_count"] += len(native)
        totals["gt_count"] += len(gt)
        scene_reports[scene] = {
            "scene_index": scene_index,
            "keyframe_count": keyframes,
            "successful_frame_count": successful,
            "source_count": len(sources),
            "native_prediction_count": len(native),
            "gt_count": len(gt),
        }
    expected_totals = {
        "scene_count": EXPECTED["scene_count"],
        "keyframe_count": EXPECTED["keyframe_count"],
        "successful_frame_count": EXPECTED["successful_frame_count"],
        "source_count": EXPECTED["candidate_count"],
        "native_prediction_count": EXPECTED["native_count"],
        "gt_count": EXPECTED["gt_count"],
    }
    if totals != expected_totals:
        raise F2OracleError(f"F2 paper100 census mismatch: expected={expected_totals}, actual={totals}")

    baseline = {
        threshold: official_constant_evaluate(native_iou, gt_counts, threshold)
        for threshold in THRESHOLDS
    }
    baseline_checks: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        key = _threshold_key(threshold)
        actual = float(baseline[threshold]["ap_points"])
        expected = EXPECTED_BASELINE_AP_POINTS[key]
        passed = math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9)
        baseline_checks[key] = {
            "expected_ap_points": expected,
            "actual_ap_points": actual,
            "absolute_error": abs(actual - expected),
            "passed": passed,
        }
        if not passed:
            raise F2OracleError(f"official native AP reproduction failed at IoU {key}")

    per_threshold = {
        _threshold_key(threshold): evaluate_f2_threshold(
            scenes=scenes,
            native_iou=native_iou,
            hypothesis_iou=hypothesis_iou,
            sources=sources_by_scene,
            gt_counts=gt_counts,
            baseline_evaluation=baseline[threshold],
            threshold=threshold,
        )
        for threshold in THRESHOLDS
    }
    h0_checks = validate_h0_reproduces_f1(per_threshold, f1)
    sealed_f1_ap50_additional = int(
        f1["per_threshold"]["0.50"]["additional_union_matching_over_native"]
    )
    if sealed_f1_ap50_additional != F1_AP50_ADDITIONAL_UNION_MATCHES:
        raise F2OracleError("sealed F1 AP50 additional-union count differs")
    grouped = [row["identity_constrained_grouped"] for row in per_threshold.values()]
    geometry_pass = all(row["passes_geometry_capacity"] for row in grouped)
    suffix_pass = all(row["gt_selected_candidate_suffix"]["passes_plus10_ap"] for row in grouped)
    later_joint_active_capacity_pass = geometry_pass and suffix_pass
    h0_identity_pass = all(row["passed"] for row in h0_checks.values())
    decision = f2_retention_decision(
        grouped_ap50_additional_union_matches=int(
            per_threshold["0.50"]["identity_constrained_grouped"][
                "additional_union_matching_over_native"
            ]
        ),
        h0_identity_passed=h0_identity_pass,
        merged_runtime_overall_pass=f2_runtime_overall_pass,
        later_joint_active_capacity_passed=later_joint_active_capacity_pass,
    )

    after = _f2_snapshot(
        scenes=scenes,
        scene_list=scene_list,
        full_scene_list=full_scene_list,
        f0_receipt=f0_receipt,
        f0_sidecar_root=f0_sidecar_root,
        f1_report=f1_report,
        f2_receipt=f2_receipt,
        f2_sidecar_root=f2_sidecar_root,
        f2_array_root=f2_array_root,
        baseline_root=baseline_root,
        gt_root=gt_root,
        scan_root=scan_root,
        official_evaluator=official_evaluator,
    )
    if after != before:
        raise F2OracleError("one or more sealed inputs changed during F2 oracle")
    return {
        "schema": SCHEMA,
        "protocol": "F2-DFU-LGF-lite-paper100-identity-constrained-grouped-oracle",
        "oracle_only": True,
        "deployable": False,
        "gt_used": True,
        "birth_enabled": False,
        "native_predictions_modified": False,
        "source_count_not_hypothesis_count": True,
        "hypotheses": list(HYPOTHESES),
        "score_mode": "constant_1.0",
        "class_mode": "class_agnostic",
        "strict_iou_comparison": ">",
        "scene_order": scenes,
        "totals": totals,
        "geometry_change": {
            name: {
                "applied_source_count": applied[name],
                "changed_vs_H0_source_count": changed[name],
                "retained_H0_source_count": totals["source_count"] - changed[name],
                "changed_ratio": changed[name] / totals["source_count"],
            }
            for name in HYPOTHESES
        },
        "runtime": {
            "summary": f2_receipt_payload.get("runtime"),
            "gates": f2_runtime_gates,
            "overall_pass": f2_runtime_overall_pass,
        },
        "integrity": {
            "all_frozen_f1_hashes_passed": True,
            "f1_report_sha256": F1_REPORT_SHA256,
            "f2_receipt_sha256": before["fixed_files"]["f2_receipt"]["sha256"],
            "f2_run_signature_sha256": f2_signature,
            "f2_receipt_overall_pass": f2_runtime_overall_pass,
            "h0_bitwise_reproduces_f0": True,
            "h0_reproduces_f1_oracle": h0_checks,
            "official_baseline_reproduction": baseline_checks,
            "all_inputs_before_after_identity": True,
        },
        "per_threshold": per_threshold,
        "decision": decision | {
            "target_delta_ap_points_each_threshold": TARGET_DELTA_AP_POINTS,
            "required_additional_union_matches_each_threshold": REQUIRED_ADDITIONAL_MATCHES,
            "later_joint_grouped_geometry_capacity_passes_all_thresholds_diagnostic": geometry_pass,
            "later_joint_constructive_suffix_plus10_ap_passes_all_thresholds_diagnostic": suffix_pass,
            "overall_pass": decision["retain_f2_geometry_for_f3"],
        },
        "scenes": scene_reports,
        "input_sha256_before": before,
        "input_sha256_after": after,
    }


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _validate_output_path(output: Path, protected_roots: Sequence[Path]) -> None:
    if output.suffix.lower() != ".json":
        raise F2OracleError("F2 oracle output must have a .json suffix")
    if output.exists() or output.is_symlink():
        raise F2OracleError(f"refusing to overwrite F2 oracle output: {output}")
    if any(_is_within(output, root) for root in protected_roots):
        raise F2OracleError("F2 oracle output must not be inside a protected input root")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sealed FastSAM paper100 F2 oracle")
    parser.add_argument("--scene-list", type=Path, default=Path("evaluation/data_util/meta_data/scannetv2_val.txt"))
    parser.add_argument("--full-scene-list", type=Path, default=Path("evaluation/data_util/meta_data/scannetv2_val_f0_full200.txt"))
    parser.add_argument("--f0-receipt", type=Path, default=Path("logs/scannet_fastsam_f0_full200_score05/final/F0_FASTSAM_FULL200.json"))
    parser.add_argument("--f0-sidecar-root", type=Path, default=Path("logs/scannet_fastsam_f0_full200_score05/scenes"))
    parser.add_argument("--f1-report", type=Path, default=Path("reports/fastsam_f1_paper100_oracle/F1_FASTSAM_PAPER100_ORACLE.json"))
    parser.add_argument("--f2-receipt", type=Path, default=Path("logs/scannet_fastsam_f2_paper100_score05/final/F2_FASTSAM_PAPER100.json"))
    parser.add_argument("--f2-sidecar-root", type=Path, default=Path("logs/scannet_fastsam_f2_paper100_score05/scenes"))
    parser.add_argument("--f2-array-root", type=Path, default=Path("logs/scannet_fastsam_f2_paper100_score05/arrays"))
    parser.add_argument("--baseline-root", type=Path, default=Path("results/scannet_t05_boxer_replay_active_score05"))
    parser.add_argument("--gt-root", type=Path, default=Path("evaluation/data_util/scannet_train_detection_data"))
    parser.add_argument("--scan-root", type=Path, default=Path("/extra/ZhaoX/scannet_data/scans"))
    parser.add_argument("--official-evaluator", type=Path, default=Path("upstream_clean/BoxFusion_shallow/evaluation/eval_scannet.py"))
    parser.add_argument("--out", type=Path, default=Path("reports/fastsam_f2_paper100_oracle/F2_FASTSAM_PAPER100_ORACLE.json"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_output_path(
        args.out,
        (
            args.scene_list.parent,
            args.full_scene_list.parent,
            args.f0_receipt.parent,
            args.f0_sidecar_root,
            args.f1_report.parent,
            args.f2_receipt.parent,
            args.f2_sidecar_root,
            args.f2_array_root,
            args.baseline_root,
            args.gt_root,
            args.scan_root,
            args.official_evaluator.parent,
        ),
    )
    report = audit_scannet_fastsam_f2_paper100_oracle(
        scene_list=args.scene_list,
        full_scene_list=args.full_scene_list,
        f0_receipt=args.f0_receipt,
        f0_sidecar_root=args.f0_sidecar_root,
        f1_report=args.f1_report,
        f2_receipt=args.f2_receipt,
        f2_sidecar_root=args.f2_sidecar_root,
        f2_array_root=args.f2_array_root,
        baseline_root=args.baseline_root,
        gt_root=args.gt_root,
        scan_root=args.scan_root,
        official_evaluator=args.official_evaluator,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"schema": SCHEMA, "out": os.fspath(args.out), "totals": report["totals"], "decision": report["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
