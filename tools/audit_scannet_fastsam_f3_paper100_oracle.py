#!/usr/bin/env python3
"""Read-only paper100 capacity oracle for the sealed FastSAM F3 shadow.

F3 creates causal *shadow tracks*, not predictions.  Each track may own a
best-view AABB (B), a multi-view consensus AABB (C), and at most one geometry
chosen by the frozen no-GT selector.  This audit opens ScanNet GT only after
the merge receipt and every scene sidecar have been sealed.  It never writes a
prediction, changes a native row, or authorizes birth.

The grouped oracle treats a track as the identity unit.  Taking ``max(B, C)``
therefore creates alternative edges for one row; it never stacks B and C as
two objects.  Strict IoU comparisons are inherited from the frozen F1 audit.
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
    evaluate_f1_threshold,
)


SCHEMA = "boxfusion.scannet_fastsam_f3_paper100_oracle.v1"
F3_SCENE_SCHEMA = "boxfusion.scannet_fastsam_f3_openbox.scene.v1"
F3_RECEIPT_SCHEMA = "boxfusion.scannet_fastsam_f3_openbox.merge.v1"
F3_PROTOCOL_ID = "F3-FASTSAM-OPENBOX-PROJECTION-SHADOW-PAPER100"
F1_REPORT_SCHEMA = "boxfusion.scannet_fastsam_f1_paper100_oracle.v1"
F1_REPORT_SHA256 = "05fc3b740126fcc8ac83ac335cf62df85b8ebd99b9033d0fc452e52229105304"
HYPOTHESES = ("B", "C")
MODES = ("B", "C", "SELECTOR", "GROUPED")

F1_AP50_ADDITIONAL_UNION_MATCHES = 63
F3_MIN_AP50_ADDITIONAL_MATCH_GAIN = 15
F3_RETAIN_AP50_ADDITIONAL_UNION_MATCHES = (
    F1_AP50_ADDITIONAL_UNION_MATCHES + F3_MIN_AP50_ADDITIONAL_MATCH_GAIN
)

RUNTIME_GATE_SPECS: Mapping[str, tuple[str, float]] = {
    "f3_incremental_mean_ms": ("<=", 25.0),
    "f3_incremental_p95_ms": ("<=", 40.0),
    "amortized_f3_ms_per_source_frame": ("<=", 1.0),
    "composed_complete_p95_ms": ("<=", 250.0),
    "composed_complete_max_ms": ("<", 833.33),
    "amortized_composed_complete_ms_per_source_frame": ("<=", 10.0),
    "new_gpu_allocation_bytes": ("==", 0.0),
    "total_gpu_peak_memory_bytes": ("<=", float(4 * 1024**3)),
}
CAUSALITY_GATE_NAMES = (
    "prefix_invariance",
    "query_before_commit",
    "one_source_one_track",
    "maximum_logical_accessed_ordinal",
)

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SIGNS = np.asarray(
    [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=bool
)


class F3OracleError(ValueError):
    """Raised when a sealed F3 input violates the preregistered contract."""


@dataclass(frozen=True)
class F3Track:
    scene_id: str
    track_id: int
    source_ids: tuple[str, ...]
    frame_ids: tuple[int, ...]
    observation_count: int
    confirmed: bool
    world_minmax: Mapping[str, np.ndarray | None]
    aligned_minmax: Mapping[str, np.ndarray | None]
    valid: Mapping[str, bool]
    scores: Mapping[str, float | None]
    selector_chosen: str | None


def _finite_number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
        raise F3OracleError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise F3OracleError(f"{label} is outside its finite domain")
    return result


def _vector3(value: Any, label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise F3OracleError(f"{label} must be a numeric length-3 vector") from error
    if array.shape != (3,) or not np.isfinite(array).all():
        raise F3OracleError(f"{label} must be a finite length-3 vector")
    return np.ascontiguousarray(array)


def _world_and_aligned_minmax(
    row: Mapping[str, Any], alignment: np.ndarray, label: str
) -> tuple[np.ndarray, np.ndarray]:
    q02 = _vector3(row.get("q02"), f"{label}.q02")
    q98 = _vector3(row.get("q98"), f"{label}.q98")
    center = _vector3(row.get("center"), f"{label}.center")
    extent = _vector3(row.get("extent"), f"{label}.extent")
    if np.any(q98 <= q02):
        raise F3OracleError(f"{label} requires q98 > q02")
    if not np.allclose(
        center, (q02 + q98) / 2.0, rtol=GEOMETRY_RTOL, atol=GEOMETRY_ATOL
    ):
        raise F3OracleError(f"{label}.center is inconsistent with q02/q98")
    if not np.allclose(
        extent, q98 - q02, rtol=GEOMETRY_RTOL, atol=GEOMETRY_ATOL
    ):
        raise F3OracleError(f"{label}.extent is inconsistent with q02/q98")
    matrix = np.asarray(alignment, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise F3OracleError("axis alignment must be a finite 4x4 matrix")
    corners = np.where(_SIGNS, q98[None, :], q02[None, :])
    transformed = corners @ matrix[:3, :3].T + matrix[:3, 3]
    return np.concatenate((q02, q98)), np.concatenate(
        (transformed.min(axis=0), transformed.max(axis=0))
    )


def _hypothesis_geometry(
    row: Any, alignment: np.ndarray, label: str
) -> tuple[bool, np.ndarray | None, np.ndarray | None, float | None]:
    if not isinstance(row, dict) or type(row.get("valid")) is not bool:
        raise F3OracleError(f"{label} must contain a boolean valid flag")
    if not isinstance(row.get("reason"), str) or not row["reason"]:
        raise F3OracleError(f"{label}.reason must be a non-empty string")
    folds = row.get("fold_ious")
    fold_count = row.get("valid_fold_count")
    if (
        not isinstance(folds, list)
        or type(fold_count) is not int
        or fold_count < 0
        or fold_count != len(folds)
    ):
        raise F3OracleError(f"{label} fold ledger is inconsistent")
    fold_values = [_finite_number(value, f"{label}.fold_ious", minimum=0.0) for value in folds]
    if any(value > 1.0 for value in fold_values):
        raise F3OracleError(f"{label}.fold_ious must lie in [0,1]")
    if row["valid"]:
        if fold_count < 2:
            raise F3OracleError(f"{label} valid hypothesis needs at least two LOO folds")
        score = _finite_number(row.get("score"), f"{label}.score", minimum=0.0)
        if score > 1.0 or not math.isclose(
            score, float(np.median(np.asarray(fold_values, dtype=np.float64))),
            rel_tol=0.0, abs_tol=1e-12,
        ):
            raise F3OracleError(f"{label}.score must equal median fold IoU")
        world, aligned = _world_and_aligned_minmax(row, alignment, label)
        return True, world, aligned, score
    if row.get("score") is not None:
        _finite_number(row["score"], f"{label}.score", minimum=0.0)
    geometry_values = [row.get(name) for name in ("q02", "q98", "center", "extent")]
    if any(value is not None for value in geometry_values):
        raise F3OracleError(f"{label} invalid hypothesis must not expose geometry")
    return False, None, None, None


def _selector_choice(
    b_valid: bool,
    b_score: float | None,
    c_valid: bool,
    c_score: float | None,
) -> str | None:
    b_eligible = b_valid and b_score is not None and b_score >= 0.10
    c_eligible = c_valid and c_score is not None and c_score >= 0.10
    if c_eligible and (not b_eligible or c_score >= b_score + 0.03):
        return "C"
    if b_eligible:
        return "B"
    if c_eligible:
        return "C"
    return None


def _validate_selector(
    row: Any,
    *,
    hypotheses: Mapping[str, Mapping[str, Any]],
    parsed: Mapping[str, tuple[bool, np.ndarray | None, np.ndarray | None, float | None]],
    alignment: np.ndarray,
    label: str,
) -> tuple[str | None, np.ndarray | None, np.ndarray | None, float | None]:
    if not isinstance(row, dict):
        raise F3OracleError(f"{label} must be an object")
    chosen = row.get("chosen")
    if chosen not in (None, *HYPOTHESES):
        raise F3OracleError(f"{label}.chosen must be B, C, or null")
    if not isinstance(row.get("reason"), str) or not row["reason"]:
        raise F3OracleError(f"{label}.reason must be a non-empty string")
    if parsed["B"][0] and (parsed["B"][3] is None or parsed["B"][3] < 0.10):
        raise F3OracleError(f"{label} parent B.valid violates the 0.10 score gate")
    c_row = hypotheses["C"]
    if parsed["C"][0]:
        if parsed["C"][3] is None or parsed["C"][3] < 0.10:
            raise F3OracleError(f"{label} parent C.valid violates the 0.10 score gate")
        stability = c_row.get("loo_full_aabb_ious")
        if not isinstance(stability, list) or not stability:
            raise F3OracleError(f"{label} valid C lacks LOO/full stability IoUs")
        stability_values = [
            _finite_number(value, f"{label}.C.loo_full_aabb_iou", minimum=0.0)
            for value in stability
        ]
        if any(value > 1.0 for value in stability_values) or float(
            np.median(np.asarray(stability_values, dtype=np.float64))
        ) < 0.25:
            raise F3OracleError(f"{label} valid C violates the 0.25 stability gate")
        shift = c_row.get("center_shift_from_b_m")
        ratios = c_row.get("extent_ratios")
        volume = c_row.get("volume_ratio")
        has_b_relative = any(value is not None for value in (shift, ratios, volume))
        if parsed["B"][0] and not has_b_relative:
            raise F3OracleError(f"{label} valid C lacks B-relative safety metrics")
        if has_b_relative:
            shift_value = _finite_number(
                shift, f"{label}.C.center_shift_from_b_m", minimum=0.0
            )
            ratio_values = np.asarray(ratios, dtype=np.float64)
            volume_value = _finite_number(
                volume, f"{label}.C.volume_ratio", minimum=0.0
            )
            if (
                ratio_values.shape != (3,)
                or not np.isfinite(ratio_values).all()
                or shift_value > 0.50
                or np.any(ratio_values < 0.5)
                or np.any(ratio_values > 2.0)
                or not 0.25 <= volume_value <= 4.0
                or not math.isclose(
                    float(np.prod(ratio_values)), volume_value,
                    rel_tol=1e-10, abs_tol=1e-12,
                )
            ):
                raise F3OracleError(f"{label} valid C violates B-relative safety")
            if parsed["B"][0]:
                b_world = np.asarray(parsed["B"][1], dtype=np.float64)
                c_world = np.asarray(parsed["C"][1], dtype=np.float64)
                expected_shift = float(
                    np.linalg.norm(
                        (c_world[:3] + c_world[3:]) * 0.5
                        - (b_world[:3] + b_world[3:]) * 0.5
                    )
                )
                expected_ratios = (c_world[3:] - c_world[:3]) / (
                    b_world[3:] - b_world[:3]
                )
                if (
                    not math.isclose(shift_value, expected_shift, rel_tol=1e-10, abs_tol=1e-12)
                    or not np.allclose(
                        ratio_values, expected_ratios, rtol=1e-10, atol=1e-12
                    )
                ):
                    raise F3OracleError(f"{label} C B-relative metrics are inconsistent")
    expected = _selector_choice(
        parsed["B"][0], parsed["B"][3], parsed["C"][0], parsed["C"][3]
    )
    if chosen != expected:
        raise F3OracleError(
            f"{label} violates frozen selector: expected={expected!r}, actual={chosen!r}"
        )
    if chosen is None:
        if any(row.get(name) is not None for name in ("q02", "q98", "center", "extent", "score")):
            raise F3OracleError(f"{label} abstention must not expose geometry/score")
        return None, None, None, None
    valid, expected_world, expected_aligned, expected_score = parsed[chosen]
    assert valid and expected_world is not None and expected_aligned is not None
    world, aligned = _world_and_aligned_minmax(row, alignment, label)
    score = _finite_number(row.get("score"), f"{label}.score", minimum=0.0)
    if (
        not np.array_equal(world, expected_world)
        or not np.array_equal(aligned, expected_aligned)
        or score != expected_score
    ):
        raise F3OracleError(f"{label} geometry/score does not exactly copy chosen {chosen}")
    for field in ("q02", "q98", "center", "extent"):
        if row.get(field) != hypotheses[chosen].get(field):
            raise F3OracleError(f"{label}.{field} is not a bitwise JSON copy of {chosen}")
    return chosen, world, aligned, score


def grouped_iou_matrix(matrices: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return one max-IoU row per track, never one row per hypothesis."""

    if tuple(matrices) != HYPOTHESES:
        raise F3OracleError(
            f"hypothesis order must be exactly {HYPOTHESES}, got {tuple(matrices)}"
        )
    arrays = [np.asarray(matrices[name], dtype=np.float64) for name in HYPOTHESES]
    if any(array.ndim != 2 or not np.isfinite(array).all() for array in arrays):
        raise F3OracleError("hypothesis IoU matrices must be finite and two-dimensional")
    if len({array.shape for array in arrays}) != 1:
        raise F3OracleError("hypothesis IoU matrices must have identical shapes")
    return np.maximum(arrays[0], arrays[1])


def choose_hypothesis_for_edge(
    matrices: Mapping[str, np.ndarray], track_index: int, gt_index: int
) -> tuple[str, float]:
    """Choose the better B/C edge; np.argmax gives B exact-tie priority."""

    if tuple(matrices) != HYPOTHESES:
        raise F3OracleError("B/C hypothesis order mismatch")
    values = [float(matrices[name][track_index, gt_index]) for name in HYPOTHESES]
    best = int(np.argmax(np.asarray(values, dtype=np.float64)))
    return HYPOTHESES[best], values[best]


def _empty_rows(columns: int) -> np.ndarray:
    return np.empty((0, columns), dtype=np.float64)


def _track_iou_matrix(
    tracks: Sequence[F3Track], name: str, gt: np.ndarray
) -> np.ndarray:
    result = np.zeros((len(tracks), len(gt)), dtype=np.float64)
    indices = [index for index, track in enumerate(tracks) if track.valid[name]]
    if indices:
        boxes = np.stack(
            [np.asarray(tracks[index].aligned_minmax[name], dtype=np.float64) for index in indices]
        )
        result[np.asarray(indices, dtype=np.int64)] = aligned_iou_matrix(boxes, gt)
    return result


def evaluate_f3_threshold(
    *,
    scenes: Sequence[str],
    native_iou: Sequence[np.ndarray],
    track_iou: Sequence[Mapping[str, np.ndarray]],
    tracks: Sequence[Sequence[F3Track]],
    gt_counts: Sequence[int],
    baseline_evaluation: Mapping[str, Any],
    threshold: float,
) -> dict[str, Any]:
    """Evaluate B, C, fixed selector, and one-row-per-track grouped capacity."""

    if len({len(scenes), len(native_iou), len(track_iou), len(tracks), len(gt_counts)}) != 1:
        raise F3OracleError("per-scene F3 inputs have inconsistent lengths")
    baseline_masks = baseline_evaluation.get("matched_gt_masks")
    if not isinstance(baseline_masks, list) or len(baseline_masks) != len(scenes):
        raise F3OracleError("official baseline masks have inconsistent scene count")

    accumulators: dict[str, dict[str, Any]] = {
        name: {
            "native_mm": 0,
            "candidate_mm": 0,
            "union_mm": 0,
            "selected": 0,
            "valid": 0,
            "suffix": [],
            "selection": {},
            "per_scene": {},
            "chosen": {"B": 0, "C": 0},
        }
        for name in MODES
    }
    all_selector_suffix: list[np.ndarray] = []
    all_selector_count = 0

    for scene, native, matrices, scene_tracks, gt_count, official_mask in zip(
        scenes, native_iou, track_iou, tracks, gt_counts, baseline_masks
    ):
        if tuple(matrices) != ("B", "C", "SELECTOR"):
            raise F3OracleError(f"F3 matrix order mismatch: {scene}")
        if native.ndim != 2 or native.shape[1] != gt_count:
            raise F3OracleError(f"native IoU/GT shape mismatch: {scene}")
        for name in ("B", "C", "SELECTOR"):
            if matrices[name].shape != (len(scene_tracks), gt_count):
                raise F3OracleError(f"{name} IoU/track/GT shape mismatch: {scene}")
        grouped = grouped_iou_matrix({name: matrices[name] for name in HYPOTHESES})
        evaluated = {**matrices, "GROUPED": grouped}
        native_pairs = strict_maximum_matching(native, threshold)
        official_unmatched = ~np.asarray(official_mask, dtype=bool)

        selector_indices = [
            index for index, track in enumerate(scene_tracks)
            if track.selector_chosen is not None
        ]
        all_selector_suffix.append(
            matrices["SELECTOR"][selector_indices]
            if selector_indices else _empty_rows(gt_count)
        )
        all_selector_count += len(selector_indices)

        for name in MODES:
            matrix = evaluated[name]
            candidate_pairs = strict_maximum_matching(matrix, threshold)
            union_pairs = strict_maximum_matching(
                np.concatenate((native, matrix), axis=0), threshold
            )
            suffix_pairs = strict_maximum_matching(matrix, threshold, official_unmatched)
            target_by_track = {track_index: gt_index for track_index, gt_index in suffix_pairs}
            selected_indices = sorted(target_by_track)
            selected_matrices: list[np.ndarray] = []
            selected_rows: list[dict[str, Any]] = []
            for track_index in selected_indices:
                gt_index = target_by_track[track_index]
                if name == "GROUPED":
                    chosen, chosen_iou = choose_hypothesis_for_edge(
                        {hypothesis: matrices[hypothesis] for hypothesis in HYPOTHESES},
                        track_index,
                        gt_index,
                    )
                elif name == "SELECTOR":
                    chosen = scene_tracks[track_index].selector_chosen
                    if chosen not in HYPOTHESES:
                        raise F3OracleError("abstained selector unexpectedly matched GT")
                    chosen_iou = float(matrix[track_index, gt_index])
                else:
                    chosen = name
                    chosen_iou = float(matrix[track_index, gt_index])
                selected_matrices.append(matrices[chosen][track_index])
                accumulators[name]["chosen"][chosen] += 1
                selected_rows.append(
                    {
                        "track_id": scene_tracks[track_index].track_id,
                        "track_index": track_index,
                        "source_ids": list(scene_tracks[track_index].source_ids),
                        "chosen_hypothesis": chosen,
                        "target_gt_index": gt_index,
                        "target_iou": chosen_iou,
                    }
                )
            suffix = np.stack(selected_matrices) if selected_matrices else _empty_rows(gt_count)
            valid_count = sum(
                (
                    track.valid[name]
                    if name in HYPOTHESES
                    else track.selector_chosen is not None
                    if name == "SELECTOR"
                    else track.valid["B"] or track.valid["C"]
                )
                for track in scene_tracks
            )
            acc = accumulators[name]
            acc["native_mm"] += len(native_pairs)
            acc["candidate_mm"] += len(candidate_pairs)
            acc["union_mm"] += len(union_pairs)
            acc["selected"] += len(selected_indices)
            acc["valid"] += valid_count
            acc["suffix"].append(suffix)
            acc["selection"][scene] = selected_rows
            acc["per_scene"][scene] = {
                "track_count": len(scene_tracks),
                "valid_candidate_count": valid_count,
                "native_maximum_matching_count": len(native_pairs),
                "candidate_maximum_matching_count": len(candidate_pairs),
                "union_maximum_matching_count": len(union_pairs),
                "additional_union_matching_over_native": len(union_pairs) - len(native_pairs),
                "gt_selected_suffix_count": len(selected_indices),
            }

    baseline_ap = float(baseline_evaluation["ap_points"])
    reports: dict[str, Any] = {}
    for name in MODES:
        acc = accumulators[name]
        combined = [
            np.concatenate((native, suffix), axis=0)
            for native, suffix in zip(native_iou, acc["suffix"])
        ]
        evaluation = official_constant_evaluate(combined, gt_counts, threshold)
        delta_ap = float(evaluation["ap_points"]) - baseline_ap
        additional = int(acc["union_mm"] - acc["native_mm"])
        reports[name] = {
            "identity_unit": "sealed_F3_shadow_track",
            "track_can_match_at_most_one_gt": True,
            "valid_candidate_count": int(acc["valid"]),
            "native_maximum_matching_count": int(acc["native_mm"]),
            "candidate_maximum_matching_count": int(acc["candidate_mm"]),
            "union_maximum_matching_count": int(acc["union_mm"]),
            "additional_union_matching_over_native": additional,
            "required_additional_union_matches": REQUIRED_ADDITIONAL_MATCHES,
            "passes_final_geometry_capacity": additional >= REQUIRED_ADDITIONAL_MATCHES,
            "gt_selected_track_suffix": {
                "oracle_only": True,
                "deployable": False,
                "threshold_specific": True,
                "selection": "track_maximum_matching_to_official_native_greedy_unmatched_gt",
                "native_rows_are_unchanged_scene_prefix": True,
                "formal_score": 1.0,
                "selected_track_count": int(acc["selected"]),
                "chosen_hypothesis_counts": acc["chosen"],
                "official_evaluation": _json_evaluation(evaluation, scenes),
                "delta_ap_points": delta_ap,
                "passes_plus10_ap": delta_ap >= TARGET_DELTA_AP_POINTS,
                "per_scene_selection": acc["selection"],
            },
            "per_scene": acc["per_scene"],
        }

    fixed_combined = [
        np.concatenate((native, suffix), axis=0)
        for native, suffix in zip(native_iou, all_selector_suffix)
    ]
    fixed_evaluation = official_constant_evaluate(fixed_combined, gt_counts, threshold)
    reports["SELECTOR"]["fixed_selector_all_tracks"] = {
        "geometry_selection_uses_gt": False,
        "evaluation_uses_gt": True,
        "prediction_materialized": False,
        "native_rows_are_unchanged_scene_prefix": True,
        "formal_score": 1.0,
        "appended_track_count": all_selector_count,
        "official_evaluation": _json_evaluation(fixed_evaluation, scenes),
        "greedy_tp": int(fixed_evaluation["greedy_tp"]),
        "false_positive": int(fixed_evaluation["false_positive"]),
        "baseline_greedy_tp": int(baseline_evaluation["greedy_tp"]),
        "baseline_false_positive": int(baseline_evaluation["false_positive"]),
        "delta_greedy_tp": int(fixed_evaluation["greedy_tp"])
        - int(baseline_evaluation["greedy_tp"]),
        "additional_false_positive": int(fixed_evaluation["false_positive"])
        - int(baseline_evaluation["false_positive"]),
        "delta_ap_points": float(fixed_evaluation["ap_points"]) - baseline_ap,
    }

    return {
        "iou_threshold": threshold,
        "strict_iou_comparison": ">",
        "baseline_official_constant_score": _json_evaluation(baseline_evaluation, scenes),
        "hypothesis_only": {name: reports[name] for name in HYPOTHESES},
        "fixed_no_gt_selector": reports["SELECTOR"],
        "identity_constrained_grouped": reports["GROUPED"],
    }


def validate_h0_reproduces_f1(
    h0_per_threshold: Mapping[str, Any], f1_report: Mapping[str, Any]
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        key = _threshold_key(threshold)
        actual = h0_per_threshold[key]
        expected = f1_report["per_threshold"][key]
        integer_pairs = {
            name: (actual[name], expected[name])
            for name in (
                "candidate_maximum_matching_count",
                "union_maximum_matching_count",
                "additional_union_matching_over_native",
            )
        }
        integer_pairs["selected_candidate_count"] = (
            actual["gt_selected_candidate_suffix"]["selected_candidate_count"],
            expected["gt_selected_candidate_suffix"]["selected_candidate_count"],
        )
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
        passed = all(left == right for left, right in integer_pairs.values()) and all(
            math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
            for left, right in numeric_pairs.values()
        )
        checks[key] = {
            "passed": passed,
            "integer_checks": {
                name: {"actual": left, "expected": right, "passed": left == right}
                for name, (left, right) in integer_pairs.items()
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
        if not passed:
            raise F3OracleError(f"H0 failed to reproduce sealed F1 at IoU {key}")
    return checks


def f3_retention_decision(
    *,
    grouped_ap50_additional_union_matches: int,
    h0_identity_passed: bool,
    runtime_passed: bool,
    causality_passed: bool,
    final_geometry_capacity_passed: bool,
    final_plus10_ap_passed: bool,
) -> dict[str, Any]:
    if type(grouped_ap50_additional_union_matches) is not int:
        raise F3OracleError("grouped AP50 additional-union matches must be an integer")
    booleans = (
        h0_identity_passed,
        runtime_passed,
        causality_passed,
        final_geometry_capacity_passed,
        final_plus10_ap_passed,
    )
    if any(type(value) is not bool for value in booleans):
        raise F3OracleError("F3 decision gates must be booleans")
    capacity_pass = (
        grouped_ap50_additional_union_matches
        >= F3_RETAIN_AP50_ADDITIONAL_UNION_MATCHES
    )
    retain = capacity_pass and h0_identity_passed and runtime_passed and causality_passed
    return {
        "f1_h0_ap50_additional_union_matches": F1_AP50_ADDITIONAL_UNION_MATCHES,
        "required_ap50_additional_match_gain_over_f1": F3_MIN_AP50_ADDITIONAL_MATCH_GAIN,
        "required_grouped_ap50_additional_union_matches": F3_RETAIN_AP50_ADDITIONAL_UNION_MATCHES,
        "actual_grouped_ap50_additional_union_matches": grouped_ap50_additional_union_matches,
        "actual_ap50_additional_match_gain_over_f1": (
            grouped_ap50_additional_union_matches - F1_AP50_ADDITIONAL_UNION_MATCHES
        ),
        "ap50_retention_capacity_passed": capacity_pass,
        "h0_identity_passed": h0_identity_passed,
        "runtime_passed": runtime_passed,
        "causality_passed": causality_passed,
        "retain_f3_for_next_selector_filter_experiment": retain,
        "final_144_matches_each_threshold_passed_diagnostic": final_geometry_capacity_passed,
        "final_plus10_ap_each_threshold_passed_diagnostic": final_plus10_ap_passed,
        "authorize_next_preregistered_selector_filter_experiment": retain,
        "authorize_active_birth": False,
        "native_predictions_modified": False,
        "result": "retain_f3_shadow" if retain else "discard_f3_shadow",
        "overall_pass": retain,
    }


def _gate_pass(row: Any, label: str) -> bool:
    if type(row) is bool:
        return row
    if isinstance(row, dict) and type(row.get("passed")) is bool:
        return bool(row["passed"])
    raise F3OracleError(f"{label} must be a boolean or a gate object")


def _validate_causality(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise F3OracleError(f"{label} must be an object")
    result: dict[str, bool] = {}
    for name in CAUSALITY_GATE_NAMES:
        raw = value.get(name, value.get(f"{name}_passed"))
        result[name] = _gate_pass(raw, f"{label}.{name}")
    if not all(result.values()):
        raise F3OracleError(f"{label} contains a failed causal-safety gate")
    overall = value.get("overall_pass", value.get("passed", True))
    if type(overall) is not bool or overall is not all(result.values()):
        raise F3OracleError(f"{label}.overall_pass is inconsistent")
    return {"items": result, "overall_pass": True}


def _compare(actual: float, comparator: str, threshold: float) -> bool:
    return {
        "<": actual < threshold,
        "<=": actual <= threshold,
        ">": actual > threshold,
        ">=": actual >= threshold,
        "==": actual == threshold,
    }[comparator]


def _validate_runtime(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise F3OracleError("F3 merge runtime must be an object")
    gates = value.get("gates", value)
    if not isinstance(gates, dict):
        raise F3OracleError("F3 runtime gates must be an object")
    result: dict[str, Any] = {}
    for name, (expected_comparator, expected_threshold) in RUNTIME_GATE_SPECS.items():
        row = gates.get(name)
        if not isinstance(row, dict):
            raise F3OracleError(f"F3 runtime gate missing: {name}")
        actual = _finite_number(row.get("actual"), f"runtime.{name}.actual", minimum=0.0)
        comparator = row.get("comparator")
        threshold = _finite_number(row.get("threshold"), f"runtime.{name}.threshold", minimum=0.0)
        passed = row.get("passed")
        if (
            comparator != expected_comparator
            or threshold != expected_threshold
            or type(passed) is not bool
            or passed is not _compare(actual, comparator, threshold)
        ):
            raise F3OracleError(f"F3 runtime gate is inconsistent: {name}")
        result[name] = dict(row)
    computed = all(row["passed"] for row in result.values())
    overall = value.get("overall_pass")
    if type(overall) is not bool or overall is not computed:
        raise F3OracleError("F3 runtime overall_pass is inconsistent")
    return {"items": result, "overall_pass": computed}


def _required_contracts(value: Any, label: str) -> None:
    required = {
        "shadow_only": True,
        "birth_enabled": False,
        "ground_truth_access": False,
        "prediction_access": False,
        "evaluator_access": False,
        "native_output_mutation": False,
        "training": False,
        "online_learning": False,
        "future_frame_logical_access": False,
        "hl_hlg_access": False,
    }
    if not isinstance(value, dict) or any(value.get(key) != expected for key, expected in required.items()):
        raise F3OracleError(f"{label} shadow/training-free contracts mismatch")


def _receipt_entry(row: Mapping[str, Any], key: str, scene: str) -> Mapping[str, Any]:
    value = row.get(key)
    if not isinstance(value, dict) or _HASH_RE.fullmatch(str(value.get("sha256"))) is None:
        raise F3OracleError(f"F3 receipt {key} seal missing: {scene}")
    return value


def _validate_f3_receipt(
    receipt: Mapping[str, Any],
    scenes: Sequence[str],
    *,
    expected_scene_count: int = EXPECTED["scene_count"],
    expected_keyframe_count: int = EXPECTED["keyframe_count"],
    expected_successful_frame_count: int = EXPECTED["successful_frame_count"],
    expected_source_count: int = EXPECTED["candidate_count"],
) -> tuple[dict[str, Mapping[str, Any]], str, bool, dict[str, Any]]:
    for key, expected in {
        "schema": F3_RECEIPT_SCHEMA,
        "protocol_id": F3_PROTOCOL_ID,
        "complete": True,
    }.items():
        if receipt.get(key) != expected:
            raise F3OracleError(f"F3 receipt contract mismatch for {key}")
    if type(receipt.get("overall_pass")) is not bool:
        raise F3OracleError("F3 receipt overall_pass must be boolean")
    _required_contracts(receipt.get("contracts"), "F3 merge receipt")
    signature = receipt.get("run_signature_sha256")
    if not isinstance(signature, str) or _HASH_RE.fullmatch(signature) is None:
        raise F3OracleError("F3 receipt run signature is missing")
    coverage = receipt.get("coverage")
    expected_counts = {
        "scene_count": expected_scene_count,
        "keyframe_count": expected_keyframe_count,
        "successful_frame_count": expected_successful_frame_count,
        "source_count": expected_source_count,
    }
    if (
        len(scenes) != expected_scene_count
        or not isinstance(coverage, dict)
        or coverage.get("scene_order") != list(scenes)
        or any(coverage.get(key) != expected for key, expected in expected_counts.items())
    ):
        raise F3OracleError("F3 receipt paper100 coverage/census mismatch")
    integrity = receipt.get("integrity")
    if not isinstance(integrity, dict) or integrity.get("overall_pass") is not True:
        raise F3OracleError("F3 receipt integrity did not pass")
    merge_causality = _validate_causality(
        receipt.get("causality"), "F3 merge causality"
    )
    runtime = _validate_runtime(receipt.get("runtime"))
    if receipt["overall_pass"] is not (runtime["overall_pass"] and True):
        raise F3OracleError("F3 receipt overall_pass is inconsistent with integrity/runtime")
    rows = receipt.get("scenes")
    if not isinstance(rows, list) or len(rows) != len(scenes):
        raise F3OracleError("F3 receipt scene ledger mismatch")
    result: dict[str, Mapping[str, Any]] = {}
    for index, (scene, row) in enumerate(zip(scenes, rows)):
        actual_index = row.get("index") if isinstance(row, dict) else None
        if actual_index is None and isinstance(row, dict):
            actual_index = row.get("scene_index")
        if not isinstance(row, dict) or row.get("scene_id") != scene or actual_index != index:
            raise F3OracleError("F3 receipt scene order mismatch")
        _receipt_entry(row, "sidecar", scene)
        if not isinstance(row.get("counts"), dict):
            raise F3OracleError(f"F3 receipt scene counts missing: {scene}")
        result[scene] = row
    return result, signature, runtime["overall_pass"], {
        **runtime,
        "merge_causality": merge_causality,
    }


def _f0_expected_frames(payload: Mapping[str, Any], scene: str) -> list[dict[str, Any]]:
    if payload.get("schema") != F0_SCENE_SCHEMA or payload.get("scene_id") != scene:
        raise F3OracleError(f"F0 sidecar contract mismatch while verifying F3: {scene}")
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise F3OracleError(f"F0 frames missing while verifying F3: {scene}")
    result: list[dict[str, Any]] = []
    for ordinal, frame in enumerate(frames):
        if not isinstance(frame, dict) or frame.get("frame_ordinal") != ordinal:
            raise F3OracleError(f"invalid F0 frame while verifying F3: {scene}:{ordinal}")
        frame_id = frame.get("frame_id")
        if type(frame_id) is not int or type(frame.get("successful")) is not bool:
            raise F3OracleError(f"invalid F0 frame identity while verifying F3: {scene}:{ordinal}")
        if frame["successful"]:
            funnel = frame.get("funnel")
            candidates = funnel.get("candidates") if isinstance(funnel, dict) else None
            if not isinstance(candidates, list):
                raise F3OracleError(f"successful F0 frame lacks candidates: {scene}:{frame_id}")
        else:
            candidates = []
        source_ids = []
        for candidate in candidates:
            raw_index = candidate.get("raw_index") if isinstance(candidate, dict) else None
            if type(raw_index) is not int:
                raise F3OracleError(f"invalid F0 raw index: {scene}:{frame_id}")
            source_ids.append(f"{scene}/frame_{frame_id:06d}/raw_{raw_index:03d}")
        result.append(
            {
                "frame_id": frame_id,
                "ordinal": ordinal,
                "successful": frame["successful"],
                "source_ids": source_ids,
            }
        )
    return result


def _count_value(counts: Mapping[str, Any], *names: str) -> int:
    found = [counts[name] for name in names if name in counts]
    if len(found) != 1 or type(found[0]) is not int or found[0] < 0:
        raise F3OracleError(f"exactly one non-negative count field required: {names}")
    return int(found[0])


def _load_f3_tracks(
    *,
    path: Path,
    f0_path: Path,
    scene: str,
    scene_index: int,
    alignment: np.ndarray,
    receipt_sidecar_sha256: str,
    run_signature_sha256: str,
) -> tuple[list[F3Track], dict[str, int], dict[str, Any]]:
    if _sha256(_regular_file(path, f"F3 sidecar for {scene}")) != receipt_sidecar_sha256:
        raise F3OracleError(f"F3 receipt sidecar hash mismatch: {scene}")
    payload = _read_json(path, f"F3 sidecar for {scene}")
    for key, expected in {
        "schema": F3_SCENE_SCHEMA,
        "protocol_id": F3_PROTOCOL_ID,
        "complete": True,
        "scene_id": scene,
        "scene_index": scene_index,
        "run_signature_sha256": run_signature_sha256,
    }.items():
        if payload.get(key) != expected:
            raise F3OracleError(f"F3 sidecar contract mismatch: {scene}.{key}")
    _required_contracts(payload.get("contracts"), f"F3 scene {scene}")
    scene_causality = _validate_causality(payload.get("causality"), f"F3 scene {scene}.causality")
    expected_frames = _f0_expected_frames(_read_json(f0_path, f"F0 sidecar for {scene}"), scene)
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) != len(expected_frames):
        raise F3OracleError(f"F3/F0 frame ledger mismatch: {scene}")

    assigned: dict[str, tuple[int, int]] = {}
    assignments_by_track: dict[int, list[tuple[str, int]]] = {}
    created_track_ids: list[int] = []
    historical_max_created_track_id = -1
    successful = 0
    max_accessed = -1
    for expected, frame in zip(expected_frames, frames):
        if not isinstance(frame, dict):
            raise F3OracleError(f"invalid F3 frame row: {scene}:{expected['ordinal']}")
        for field in ("frame_id", "ordinal", "successful"):
            if frame.get(field) != expected[field]:
                raise F3OracleError(f"F3/F0 frame identity mismatch: {scene}:{expected['ordinal']}:{field}")
        if frame.get("source_ids") != expected["source_ids"]:
            raise F3OracleError(f"F3/F0 source identity/order mismatch: {scene}:{expected['frame_id']}")
        if frame["successful"]:
            successful += 1
        accessed = frame.get("max_logical_accessed_ordinal")
        if type(accessed) is not int or accessed > expected["ordinal"] or accessed < -1:
            raise F3OracleError(f"F3 future-frame logical access: {scene}:{expected['frame_id']}")
        max_accessed = max(max_accessed, accessed)
        _finite_number(frame.get("f3_core_ms"), f"{scene}:{expected['frame_id']}.f3_core_ms", minimum=0.0)
        retired = frame.get("retired_ids")
        if not isinstance(retired, list) or any(type(track_id) is not int or track_id < 0 for track_id in retired):
            raise F3OracleError(f"invalid F3 retired track ledger: {scene}:{expected['frame_id']}")
        assignments = frame.get("assignments")
        if not isinstance(assignments, list) or len(assignments) != len(expected["source_ids"]):
            raise F3OracleError(f"F3 assignment census mismatch: {scene}:{expected['frame_id']}")
        frame_created_track_ids: list[int] = []
        for source_id, assignment in zip(expected["source_ids"], assignments):
            if not isinstance(assignment, dict) or assignment.get("source_id") != source_id:
                raise F3OracleError(f"F3 assignment source order mismatch: {source_id}")
            track_id = assignment.get("track_id")
            action = assignment.get("action")
            if type(track_id) is not int or track_id < 0 or action not in {
                "create", "created", "match", "matched"
            }:
                raise F3OracleError(f"invalid F3 assignment: {source_id}")
            if source_id in assigned:
                raise F3OracleError(f"F3 source assigned more than once: {source_id}")
            prior = assignments_by_track.setdefault(track_id, [])
            expected_create = not prior
            if expected_create != (action in {"create", "created"}):
                raise F3OracleError(f"F3 assignment action/track history mismatch: {source_id}")
            if expected_create:
                frame_created_track_ids.append(track_id)
            if prior and prior[-1][1] == expected["frame_id"]:
                raise F3OracleError(f"F3 track accepted two sources in one frame: {track_id}")
            prior.append((source_id, expected["frame_id"]))
            assigned[source_id] = (track_id, expected["frame_id"])
        if frame_created_track_ids:
            ordered_created = sorted(frame_created_track_ids)
            if (
                len(ordered_created) != len(set(ordered_created))
                or ordered_created
                != list(range(ordered_created[0], ordered_created[-1] + 1))
                or ordered_created[0] <= historical_max_created_track_id
            ):
                raise F3OracleError(
                    f"F3 per-frame created track IDs are not a new consecutive interval: "
                    f"{scene}:{expected['frame_id']}"
                )
            historical_max_created_track_id = ordered_created[-1]
            created_track_ids.extend(ordered_created)

    tracks_payload = payload.get("tracks")
    if not isinstance(tracks_payload, list):
        raise F3OracleError(f"F3 tracks must be a list: {scene}")
    tracks: list[F3Track] = []
    seen_track_ids: set[int] = set()
    for track_index, row in enumerate(tracks_payload):
        if not isinstance(row, dict):
            raise F3OracleError(f"invalid F3 track row: {scene}:{track_index}")
        track_id = row.get("track_id")
        if type(track_id) is not int or track_id < 0 or track_id in seen_track_ids:
            raise F3OracleError(f"invalid/duplicate F3 track ID: {scene}:{track_index}")
        if tracks and track_id <= tracks[-1].track_id:
            raise F3OracleError(f"F3 terminal tracks are not in stable track order: {scene}")
        seen_track_ids.add(track_id)
        expected_observations = assignments_by_track.get(track_id)
        if not expected_observations:
            raise F3OracleError(f"F3 terminal track lacks assignment lineage: {scene}:{track_id}")
        source_ids = row.get("source_ids")
        frame_ids = row.get("frame_ids")
        observation_count = row.get("observation_count")
        retained_source_ids = row.get("retained_source_ids")
        retained_frame_ids = row.get("retained_frame_ids")
        retained_observation_count = row.get(
            "retained_observation_count", row.get("retained_view_count")
        )
        confirmed = row.get("confirmed")
        if (
            not isinstance(source_ids, list)
            or not all(isinstance(value, str) and value for value in source_ids)
            or not isinstance(frame_ids, list)
            or any(type(value) is not int for value in frame_ids)
            or type(observation_count) is not int
            or not isinstance(retained_source_ids, list)
            or not all(isinstance(value, str) and value for value in retained_source_ids)
            or not isinstance(retained_frame_ids, list)
            or any(type(value) is not int for value in retained_frame_ids)
            or type(retained_observation_count) is not int
            or type(confirmed) is not bool
        ):
            raise F3OracleError(f"invalid F3 track identity fields: {scene}:{track_id}")
        expected_source_ids = [value[0] for value in expected_observations]
        expected_frame_ids = [value[1] for value in expected_observations]
        if (
            source_ids != expected_source_ids
            or frame_ids != expected_frame_ids
            or observation_count != len(source_ids)
            or len(set(frame_ids)) != len(frame_ids)
            or retained_source_ids != source_ids[-5:]
            or retained_frame_ids != frame_ids[-5:]
            or retained_observation_count != len(retained_source_ids)
            or retained_observation_count > 5
            or confirmed is not (retained_observation_count >= 3)
        ):
            raise F3OracleError(f"F3 track lineage/count/confirmation mismatch: {scene}:{track_id}")
        hypotheses = row.get("hypotheses")
        if not isinstance(hypotheses, dict) or tuple(hypotheses) != HYPOTHESES:
            raise F3OracleError(f"F3 B/C hypothesis order mismatch: {scene}:{track_id}")
        parsed = {
            name: _hypothesis_geometry(
                hypotheses[name], alignment, f"{scene}.track{track_id}.{name}"
            )
            for name in HYPOTHESES
        }
        if not confirmed and any(parsed[name][0] for name in HYPOTHESES):
            raise F3OracleError(f"unconfirmed F3 track exposed geometry: {scene}:{track_id}")
        chosen, selector_world, selector_aligned, selector_score = _validate_selector(
            row.get("selector"),
            hypotheses=hypotheses,
            parsed=parsed,
            alignment=alignment,
            label=f"{scene}.track{track_id}.selector",
        )
        tracks.append(
            F3Track(
                scene_id=scene,
                track_id=track_id,
                source_ids=tuple(source_ids),
                frame_ids=tuple(frame_ids),
                observation_count=observation_count,
                confirmed=confirmed,
                world_minmax={
                    "B": parsed["B"][1],
                    "C": parsed["C"][1],
                    "SELECTOR": selector_world,
                },
                aligned_minmax={
                    "B": parsed["B"][2],
                    "C": parsed["C"][2],
                    "SELECTOR": selector_aligned,
                },
                valid={
                    "B": parsed["B"][0],
                    "C": parsed["C"][0],
                    "SELECTOR": chosen is not None,
                },
                scores={
                    "B": parsed["B"][3],
                    "C": parsed["C"][3],
                    "SELECTOR": selector_score,
                },
                selector_chosen=chosen,
            )
        )
    if seen_track_ids != set(assignments_by_track):
        raise F3OracleError(f"F3 terminal track ledger does not cover assignments: {scene}")
    if (
        created_track_ids != list(range(len(created_track_ids)))
        or seen_track_ids != set(range(len(tracks)))
    ):
        raise F3OracleError(f"F3 track IDs are not globally contiguous/non-reused: {scene}")
    expected_source_ids = [source_id for frame in expected_frames for source_id in frame["source_ids"]]
    if list(assigned) != expected_source_ids:
        raise F3OracleError(f"F3 assignments do not reproduce all F1/H0 source identities: {scene}")

    counts = payload.get("counts")
    if not isinstance(counts, dict):
        raise F3OracleError(f"F3 scene counts missing: {scene}")
    actual_counts = {
        "keyframe_count": len(expected_frames),
        "successful_frame_count": successful,
        "source_count": len(expected_source_ids),
        "track_count": len(tracks),
        "confirmed_track_count": sum(track.confirmed for track in tracks),
        "selected_track_count": sum(track.selector_chosen is not None for track in tracks),
    }
    recorded_counts = {
        "keyframe_count": _count_value(counts, "keyframe_count"),
        "successful_frame_count": _count_value(counts, "successful_frame_count"),
        "source_count": _count_value(counts, "source_count"),
        "track_count": _count_value(counts, "track_count"),
        "confirmed_track_count": _count_value(counts, "confirmed_track_count", "eligible_track_count"),
        "selected_track_count": _count_value(counts, "selected_track_count"),
    }
    if recorded_counts != actual_counts:
        raise F3OracleError(
            f"F3 scene census mismatch: {scene}: expected={actual_counts}, recorded={recorded_counts}"
        )
    receipt_runtime = payload.get("runtime")
    if not isinstance(receipt_runtime, dict):
        raise F3OracleError(f"F3 scene runtime missing: {scene}")
    return tracks, actual_counts, {
        "causality": scene_causality,
        "max_logical_accessed_ordinal": max_accessed,
        "runtime": receipt_runtime,
    }


def _f3_snapshot(
    *,
    scenes: Sequence[str],
    scene_list: Path,
    full_scene_list: Path,
    f0_receipt: Path,
    f0_sidecar_root: Path,
    f1_report: Path,
    f3_receipt: Path,
    f3_sidecar_root: Path,
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
            "f1_report": {
                "path": os.fspath(f1_report),
                "sha256": _sha256(_regular_file(f1_report, "F1 report")),
            },
            "f3_receipt": {
                "path": os.fspath(f3_receipt),
                "sha256": _sha256(_regular_file(f3_receipt, "F3 receipt")),
            },
        }
    )
    base["ordered_scene_ledgers"]["f3_sidecars"] = canonical_ordered_hash_ledger(
        scenes, [f3_sidecar_root / f"{scene}.json" for scene in scenes], "F3 sidecar"
    )
    return base


def audit_scannet_fastsam_f3_paper100_oracle(
    *,
    scene_list: Path,
    full_scene_list: Path,
    f0_receipt: Path,
    f0_sidecar_root: Path,
    f1_report: Path,
    f3_receipt: Path,
    f3_sidecar_root: Path,
    baseline_root: Path,
    gt_root: Path,
    scan_root: Path,
    official_evaluator: Path,
) -> dict[str, Any]:
    scenes = load_scene_list(_regular_file(scene_list, "paper100 scene list"))
    full_scenes = load_scene_list(_regular_file(full_scene_list, "F0 full200 scene list"))
    if len(scenes) != EXPECTED["scene_count"] or scenes != full_scenes[: EXPECTED["scene_count"]]:
        raise F3OracleError("frozen paper100 scene order/count mismatch")
    if _sha256(_regular_file(f1_report, "F1 report")) != F1_REPORT_SHA256:
        raise F3OracleError("sealed F1 report SHA-256 mismatch")
    f1 = _read_json(f1_report, "F1 report")
    if f1.get("schema") != F1_REPORT_SCHEMA:
        raise F3OracleError("unexpected F1 report schema")
    f3_receipt_payload = _read_json(f3_receipt, "F3 receipt")
    f3_rows, f3_signature, runtime_passed, runtime = _validate_f3_receipt(
        f3_receipt_payload, scenes
    )

    before = _f3_snapshot(
        scenes=scenes,
        scene_list=scene_list,
        full_scene_list=full_scene_list,
        f0_receipt=f0_receipt,
        f0_sidecar_root=f0_sidecar_root,
        f1_report=f1_report,
        f3_receipt=f3_receipt,
        f3_sidecar_root=f3_sidecar_root,
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
    h0_iou: list[np.ndarray] = []
    f0_candidates_by_scene: list[list[Any]] = []
    track_iou: list[dict[str, np.ndarray]] = []
    tracks_by_scene: list[list[F3Track]] = []
    scene_reports: dict[str, Any] = {}
    totals = {
        "scene_count": len(scenes),
        "keyframe_count": 0,
        "successful_frame_count": 0,
        "source_count": 0,
        "track_count": 0,
        "confirmed_track_count": 0,
        "selected_track_count": 0,
        "native_prediction_count": 0,
        "gt_count": 0,
    }
    for scene_index, scene in enumerate(scenes):
        alignment = load_axis_alignment(scan_root / scene / f"{scene}.txt")
        gt = load_gt_minmax(gt_root / f"{scene}_bbox.npy")
        _, native = load_baseline_boxes(baseline_root / f"{scene}_boxes.pkl", alignment)
        f0_candidates, keyframes, successful = _load_f0_candidates(
            path=f0_sidecar_root / f"{scene}.json",
            scene=scene,
            scene_index=scene_index,
            alignment=alignment,
            receipt_sidecar_sha256=str(f0_receipt_rows[scene]["sidecar"]["sha256"]),
        )
        receipt_row = f3_rows[scene]
        tracks, counts, diagnostics = _load_f3_tracks(
            path=f3_sidecar_root / f"{scene}.json",
            f0_path=f0_sidecar_root / f"{scene}.json",
            scene=scene,
            scene_index=scene_index,
            alignment=alignment,
            receipt_sidecar_sha256=str(receipt_row["sidecar"]["sha256"]),
            run_signature_sha256=f3_signature,
        )
        if counts["keyframe_count"] != keyframes or counts["successful_frame_count"] != successful:
            raise F3OracleError(f"F3 scene counts do not reproduce F0: {scene}")
        for key, value in counts.items():
            if receipt_row["counts"].get(key) != value:
                raise F3OracleError(f"F3 merge/scene count mismatch: {scene}:{key}")

        h0_boxes = (
            np.stack([candidate.aligned_minmax for candidate in f0_candidates])
            if f0_candidates else np.empty((0, 6), dtype=np.float64)
        )
        matrices = {
            name: _track_iou_matrix(tracks, name, gt)
            for name in ("B", "C", "SELECTOR")
        }
        gt_counts.append(len(gt))
        native_iou.append(aligned_iou_matrix(native, gt))
        h0_iou.append(aligned_iou_matrix(h0_boxes, gt))
        f0_candidates_by_scene.append(f0_candidates)
        track_iou.append(matrices)
        tracks_by_scene.append(tracks)
        for key in (
            "keyframe_count",
            "successful_frame_count",
            "source_count",
            "track_count",
            "confirmed_track_count",
            "selected_track_count",
        ):
            totals[key] += counts[key]
        totals["native_prediction_count"] += len(native)
        totals["gt_count"] += len(gt)
        scene_reports[scene] = {
            "scene_index": scene_index,
            **counts,
            "native_prediction_count": len(native),
            "gt_count": len(gt),
            "maximum_logical_accessed_ordinal": diagnostics["max_logical_accessed_ordinal"],
            "causality": diagnostics["causality"],
        }
    expected_base_totals = {
        "scene_count": EXPECTED["scene_count"],
        "keyframe_count": EXPECTED["keyframe_count"],
        "successful_frame_count": EXPECTED["successful_frame_count"],
        "source_count": EXPECTED["candidate_count"],
        "native_prediction_count": EXPECTED["native_count"],
        "gt_count": EXPECTED["gt_count"],
    }
    for key, expected in expected_base_totals.items():
        if totals[key] != expected:
            raise F3OracleError(
                f"F3 paper100 census mismatch for {key}: expected={expected}, actual={totals[key]}"
            )
    receipt_totals = f3_receipt_payload.get("totals")
    if not isinstance(receipt_totals, dict) or any(
        receipt_totals.get(key) != value
        for key, value in totals.items()
        if key != "native_prediction_count" and key != "gt_count"
    ):
        raise F3OracleError("F3 merge totals do not reproduce sealed scene receipts")
    if receipt_totals.get("identity_verified_source_count") != totals["source_count"]:
        raise F3OracleError("F3 merge identity-verified source total differs")

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
            raise F3OracleError(f"official native AP reproduction failed at IoU {key}")

    h0_per_threshold = {
        _threshold_key(threshold): evaluate_f1_threshold(
            scenes=scenes,
            native_iou=native_iou,
            candidate_iou=h0_iou,
            candidates=f0_candidates_by_scene,
            gt_counts=gt_counts,
            baseline_evaluation=baseline[threshold],
            threshold=threshold,
        )
        for threshold in THRESHOLDS
    }
    h0_checks = validate_h0_reproduces_f1(h0_per_threshold, f1)
    if int(f1["per_threshold"]["0.50"]["additional_union_matching_over_native"]) != F1_AP50_ADDITIONAL_UNION_MATCHES:
        raise F3OracleError("sealed F1 AP50 additional-union count differs")

    per_threshold = {
        _threshold_key(threshold): evaluate_f3_threshold(
            scenes=scenes,
            native_iou=native_iou,
            track_iou=track_iou,
            tracks=tracks_by_scene,
            gt_counts=gt_counts,
            baseline_evaluation=baseline[threshold],
            threshold=threshold,
        )
        for threshold in THRESHOLDS
    }
    grouped_reports = [
        per_threshold[_threshold_key(threshold)]["identity_constrained_grouped"]
        for threshold in THRESHOLDS
    ]
    final_geometry_pass = all(
        row["additional_union_matching_over_native"] >= REQUIRED_ADDITIONAL_MATCHES
        for row in grouped_reports
    )
    final_plus10_pass = all(
        row["gt_selected_track_suffix"]["delta_ap_points"] >= TARGET_DELTA_AP_POINTS
        for row in grouped_reports
    )
    causality_passed = all(
        report["causality"]["overall_pass"] for report in scene_reports.values()
    )
    decision = f3_retention_decision(
        grouped_ap50_additional_union_matches=int(
            per_threshold["0.50"]["identity_constrained_grouped"][
                "additional_union_matching_over_native"
            ]
        ),
        h0_identity_passed=all(row["passed"] for row in h0_checks.values()),
        runtime_passed=runtime_passed,
        causality_passed=causality_passed,
        final_geometry_capacity_passed=final_geometry_pass,
        final_plus10_ap_passed=final_plus10_pass,
    )

    after = _f3_snapshot(
        scenes=scenes,
        scene_list=scene_list,
        full_scene_list=full_scene_list,
        f0_receipt=f0_receipt,
        f0_sidecar_root=f0_sidecar_root,
        f1_report=f1_report,
        f3_receipt=f3_receipt,
        f3_sidecar_root=f3_sidecar_root,
        baseline_root=baseline_root,
        gt_root=gt_root,
        scan_root=scan_root,
        official_evaluator=official_evaluator,
    )
    if after != before:
        raise F3OracleError("one or more sealed inputs changed during F3 oracle")
    return {
        "schema": SCHEMA,
        "protocol": "F3-FastSAM-OpenBox-projection-paper100-track-grouped-oracle",
        "oracle_only": True,
        "deployable": False,
        "gt_used": True,
        "birth_enabled": False,
        "native_predictions_modified": False,
        "track_count_not_hypothesis_count": True,
        "hypotheses": list(HYPOTHESES),
        "score_mode": "constant_1.0",
        "class_mode": "class_agnostic",
        "strict_iou_comparison": ">",
        "scene_order": scenes,
        "totals": totals,
        "runtime": runtime,
        "integrity": {
            "all_frozen_f1_hashes_passed": True,
            "f1_report_sha256": F1_REPORT_SHA256,
            "f3_receipt_sha256": before["fixed_files"]["f3_receipt"]["sha256"],
            "f3_run_signature_sha256": f3_signature,
            "all_52299_h0_source_identities_reproduced": True,
            "h0_reproduces_f1_oracle": h0_checks,
            "official_baseline_reproduction": baseline_checks,
            "causality_passed": causality_passed,
            "all_inputs_before_after_identity": True,
        },
        "h0_reference": h0_per_threshold,
        "per_threshold": per_threshold,
        "decision": decision
        | {
            "target_delta_ap_points_each_threshold": TARGET_DELTA_AP_POINTS,
            "required_additional_union_matches_each_threshold": REQUIRED_ADDITIONAL_MATCHES,
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
        raise F3OracleError("F3 oracle output must have a .json suffix")
    if output.exists() or output.is_symlink():
        raise F3OracleError(f"refusing to overwrite F3 oracle output: {output}")
    if any(_is_within(output, root) for root in protected_roots):
        raise F3OracleError("F3 oracle output must not be inside a protected input root")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sealed FastSAM paper100 F3 oracle")
    parser.add_argument("--scene-list", type=Path, default=Path("evaluation/data_util/meta_data/scannetv2_val.txt"))
    parser.add_argument("--full-scene-list", type=Path, default=Path("evaluation/data_util/meta_data/scannetv2_val_f0_full200.txt"))
    parser.add_argument("--f0-receipt", type=Path, default=Path("logs/scannet_fastsam_f0_full200_score05/final/F0_FASTSAM_FULL200.json"))
    parser.add_argument("--f0-sidecar-root", type=Path, default=Path("logs/scannet_fastsam_f0_full200_score05/scenes"))
    parser.add_argument("--f1-report", type=Path, default=Path("reports/fastsam_f1_paper100_oracle/F1_FASTSAM_PAPER100_ORACLE.json"))
    parser.add_argument("--f3-receipt", type=Path, default=Path("logs/scannet_fastsam_f3_openbox_paper100_score05/final/F3_FASTSAM_OPENBOX_PAPER100.json"))
    parser.add_argument("--f3-sidecar-root", type=Path, default=Path("logs/scannet_fastsam_f3_openbox_paper100_score05/scenes"))
    parser.add_argument("--baseline-root", type=Path, default=Path("results/scannet_t05_boxer_replay_active_score05"))
    parser.add_argument("--gt-root", type=Path, default=Path("evaluation/data_util/scannet_train_detection_data"))
    parser.add_argument("--scan-root", type=Path, default=Path("/extra/ZhaoX/scannet_data/scans"))
    parser.add_argument("--official-evaluator", type=Path, default=Path("upstream_clean/BoxFusion_shallow/evaluation/eval_scannet.py"))
    parser.add_argument("--out", type=Path, default=Path("reports/fastsam_f3_paper100_oracle/F3_FASTSAM_PAPER100_ORACLE.json"))
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
            args.f3_receipt.parent,
            args.f3_sidecar_root,
            args.baseline_root,
            args.gt_root,
            args.scan_root,
            args.official_evaluator.parent,
        ),
    )
    report = audit_scannet_fastsam_f3_paper100_oracle(
        scene_list=args.scene_list,
        full_scene_list=args.full_scene_list,
        f0_receipt=args.f0_receipt,
        f0_sidecar_root=args.f0_sidecar_root,
        f1_report=args.f1_report,
        f3_receipt=args.f3_receipt,
        f3_sidecar_root=args.f3_sidecar_root,
        baseline_root=args.baseline_root,
        gt_root=args.gt_root,
        scan_root=args.scan_root,
        official_evaluator=args.official_evaluator,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "out": os.fspath(args.out),
                "totals": report["totals"],
                "decision": report["decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
