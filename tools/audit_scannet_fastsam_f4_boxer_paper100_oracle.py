#!/usr/bin/env python3
"""Sealed paper100 geometry-capacity oracle for FastSAM/F2 + BoxerNet F4.

F4 is an output-inert shadow.  Each sealed F0 source owns four alternative
geometries (H0, HL, HLG, HB), but remains exactly one proposal identity.  This
module opens ground truth only after the merged F4 receipt and every F4 source
sidecar have passed their no-GT integrity, runtime, and causality contracts.
It never writes predictions and never authorizes birth.
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
from tools.audit_scannet_fastsam_f2_paper100_oracle import (  # noqa: E402
    F2_RECEIPT_SCHEMA,
    F2_SCENE_SCHEMA,
    F2OracleError,
    F2Source,
    _load_f2_sources,
    _validate_f2_receipt,
    validate_h0_reproduces_f1,
)


SCHEMA = "boxfusion.scannet_fastsam_f4_boxer_paper100_oracle.v1"
F4_SCENE_SCHEMA = "boxfusion.scannet_fastsam_f4_boxer_paper100.scene.v1"
F4_RECEIPT_SCHEMA = "boxfusion.scannet_fastsam_f4_boxer_paper100.merge.v1"
F4_PROTOCOL_ID = "F4-FASTSAM-BOXER-GEOMETRY-SHADOW-PAPER100"
F2_PROTOCOL_ID = "F2-DFU-LGF-lite-shadow-paper100"
F1_REPORT_SCHEMA = "boxfusion.scannet_fastsam_f1_paper100_oracle.v1"
F2_REPORT_SCHEMA = "boxfusion.scannet_fastsam_f2_paper100_oracle.v1"
F1_REPORT_SHA256 = "05fc3b740126fcc8ac83ac335cf62df85b8ebd99b9033d0fc452e52229105304"
F2_REPORT_SHA256 = "2c3d73f777331617c798aca5e6fdcf819a0267b7d698bdab88f70f7b72dbaff5"

BASE_HYPOTHESES = ("H0", "HL", "HLG")
HYPOTHESES = (*BASE_HYPOTHESES, "HB")
MODES = (*HYPOTHESES, "GBASE", "G4")

RUNTIME_GATE_SPECS: Mapping[str, tuple[str, float]] = {
    "f4_incremental_warm_p95_ms": ("<=", 100.0),
    "replay_composed_warm_p95_ms": ("<=", 350.0),
    "replay_composed_warm_max_ms": ("<", 833.33),
    "replay_composed_mean_per_source_frame_ms": ("<=", 14.0),
    "gap25_warm_deadline_miss_count": ("==", 0.0),
    "cuda_peak_memory_bytes": ("<=", float(4 * 1024**3)),
}
MERGE_GATE_NAMES = (
    "integrity_complete",
    "exact_keyframes",
    "exact_successful_frames",
    "exact_sources",
    *RUNTIME_GATE_SPECS,
    "native_output_mutation_count",
)
CAUSALITY_EXPECTED: Mapping[str, Any] = {
    "overall_pass": True,
    "current_frame_only": True,
    "maximum_lookahead_frames": 0,
    "maximum_logical_accessed_ordinal": True,
    "future_frame_access": False,
    "source_order_identity": True,
    "provider_called_only_for_nonempty_successful_frames": True,
    "first_three_nonempty_forwards_per_shard_excluded_only_from_warm_distributions": True,
}

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SIGNS = np.asarray(
    [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, +1.0],
        [-1.0, +1.0, -1.0],
        [-1.0, +1.0, +1.0],
        [+1.0, -1.0, -1.0],
        [+1.0, -1.0, +1.0],
        [+1.0, +1.0, -1.0],
        [+1.0, +1.0, +1.0],
    ],
    dtype=np.float64,
)


class F4OracleError(ValueError):
    """Raised when an input violates the frozen F4 oracle contract."""


@dataclass(frozen=True)
class F4Source:
    """One identity row with four alternative, never-stacked geometries."""

    scene_id: str
    scene_index: int
    frame_id: int
    frame_ordinal: int
    candidate_index: int
    rank: int
    raw_index: int
    source_id: str
    mask_sha256: str
    points_and_voxel_keys_sha256: str
    tight_box_xyxy: tuple[float, float, float, float]
    world_minmax: Mapping[str, np.ndarray | None]
    aligned_minmax: Mapping[str, np.ndarray | None]
    valid: Mapping[str, bool]


@dataclass(frozen=True)
class _PrevalidatedF4Source:
    """No-GT source record sealed before axis alignment is opened."""

    scene_id: str
    scene_index: int
    frame_id: int
    frame_ordinal: int
    candidate_index: int
    rank: int
    raw_index: int
    source_id: str
    mask_sha256: str
    points_and_voxel_keys_sha256: str
    tight_box_xyxy: tuple[float, float, float, float]
    hb: Mapping[str, Any]
    source_lineage_sha256: str


def _finite_number(
    value: Any, label: str, *, minimum: float | None = None
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
        raise F4OracleError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise F4OracleError(f"{label} is outside its finite domain")
    return result


def _vector(
    value: Any, shape: tuple[int, ...], label: str
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise F4OracleError(f"{label} must be a numeric array of shape {shape}") from error
    if array.shape != shape or not np.isfinite(array).all():
        raise F4OracleError(f"{label} must be a finite array of shape {shape}")
    return np.ascontiguousarray(array)


def _tight_box(value: Any, label: str) -> tuple[float, float, float, float]:
    box = _vector(value, (4,), label)
    if box[2] <= box[0] or box[3] <= box[1]:
        raise F4OracleError(f"{label} must have positive XYXY extent")
    if box[0] < 0.0 or box[1] < 0.0 or box[2] > 640.0 or box[3] > 480.0:
        raise F4OracleError(f"{label} must lie inside the sealed 640x480 frame")
    return tuple(float(item) for item in box)


def _canonical_json_sha256(value: Any) -> str:
    import hashlib

    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise F4OracleError("value is not canonical finite ASCII JSON") from error
    return hashlib.sha256(payload).hexdigest()


def _hash_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise F4OracleError(f"{label} must be a lowercase SHA-256")
    return value


def _align_world_aabb(
    world_minmax: np.ndarray, alignment: np.ndarray, label: str
) -> np.ndarray:
    bounds = _vector(world_minmax, (6,), label)
    q02, q98 = bounds[:3], bounds[3:]
    if np.any(q98 <= q02):
        raise F4OracleError(f"{label} requires q98 > q02")
    matrix = _vector(alignment, (4, 4), "axis alignment")
    corners = np.where(_SIGNS > 0.0, q98[None, :], q02[None, :])
    transformed = corners @ matrix[:3, :3].T + matrix[:3, 3]
    return np.concatenate((transformed.min(axis=0), transformed.max(axis=0)))


def align_boxer_world_corners(
    world_corners: Any, alignment: Any, label: str = "HB.world_corners"
) -> np.ndarray:
    """Axis-align the eight OBB corners before taking an enclosing AABB."""

    corners = _vector(world_corners, (8, 3), label)
    matrix = _vector(alignment, (4, 4), "axis alignment")
    transformed = corners @ matrix[:3, :3].T + matrix[:3, 3]
    result = np.concatenate((transformed.min(axis=0), transformed.max(axis=0)))
    if np.any(result[3:] <= result[:3]):
        raise F4OracleError(f"{label} produces a degenerate aligned AABB")
    return result


def validate_boxer_hypothesis(
    row: Any, alignment: Any, label: str
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    """Validate a sealed HB row without applying any probability threshold."""

    if not isinstance(row, dict) or type(row.get("valid")) is not bool:
        raise F4OracleError(f"{label} must contain a boolean valid flag")
    validity = row.get("validity")
    if not isinstance(validity, dict):
        raise F4OracleError(f"{label}.validity must be an object")
    required_flags = (
        "finite_center",
        "finite_extent",
        "finite_rotation",
        "finite_corners",
        "positive_extent",
        "right_handed_orthonormal",
        "in_front",
    )
    if any(type(validity.get(name)) is not bool for name in required_flags):
        raise F4OracleError(f"{label}.validity flags are incomplete")
    expected_reasons: list[str] = []
    if not validity["finite_center"]:
        expected_reasons.append("nonfinite_center")
    if not validity["finite_extent"]:
        expected_reasons.append("nonfinite_extent")
    if not validity["finite_rotation"]:
        expected_reasons.append("nonfinite_rotation")
    if validity["finite_extent"] and not validity["positive_extent"]:
        expected_reasons.append("nonpositive_extent")
    if validity["finite_rotation"] and not validity["right_handed_orthonormal"]:
        expected_reasons.append("invalid_rotation")
    if not validity["in_front"]:
        expected_reasons.append("not_in_front")
    if not validity["finite_corners"]:
        expected_reasons.append("nonfinite_corners")
    if validity.get("reasons") != expected_reasons:
        raise F4OracleError(f"{label}.validity reasons differ from the frozen rule")
    for name in (
        "orthogonality_error",
        "determinant",
        "rotation_correction_max_abs",
    ):
        value = validity.get(name)
        if validity["finite_rotation"] or value is not None:
            _finite_number(value, f"{label}.validity.{name}")
    recomputed_valid = all(bool(validity[name]) for name in required_flags)
    if row["valid"] is not recomputed_valid:
        raise F4OracleError(f"{label}.valid disagrees with fixed geometry validity")
    if row.get("abstention_reason") is not None and not isinstance(
        row.get("abstention_reason"), str
    ):
        raise F4OracleError(f"{label}.abstention_reason must be a string or null")
    if "confidence" in row and row["confidence"] is not None:
        _finite_number(row["confidence"], f"{label}.confidence")
    if not row["valid"]:
        # Invalid rows keep their identity/diagnostics but expose no usable HB.
        if not isinstance(row.get("abstention_reason"), str) or not row[
            "abstention_reason"
        ]:
            raise F4OracleError(f"{label} invalid row lacks an abstention reason")
        if any(
            row.get(name) is not None
            for name in (
                "world_corners",
                "world_center",
                "local_extent",
                "world_rotation",
                "camera_depth",
            )
        ):
            raise F4OracleError(f"{label} invalid row retains usable geometry")
        return False, None, None
    if row.get("abstention_reason") is not None:
        raise F4OracleError(f"{label} valid row must not have an abstention reason")

    center = _vector(row.get("world_center"), (3,), f"{label}.world_center")
    extent = _vector(row.get("local_extent"), (3,), f"{label}.local_extent")
    rotation = _vector(row.get("world_rotation"), (3, 3), f"{label}.world_rotation")
    corners = _vector(row.get("world_corners"), (8, 3), f"{label}.world_corners")
    camera_depth = _finite_number(
        row.get("camera_depth"), f"{label}.camera_depth"
    )
    if np.any(extent <= 0.0) or camera_depth <= 1e-4:
        raise F4OracleError(f"{label} violates the frozen extent/depth validity")
    gram = rotation.T @ rotation
    determinant = float(np.linalg.det(rotation))
    if not np.allclose(gram, np.eye(3), rtol=0.0, atol=1e-3) or not math.isclose(
        determinant, 1.0, rel_tol=0.0, abs_tol=1e-3
    ):
        raise F4OracleError(f"{label} rotation is not right-handed orthonormal")
    recomputed_flags = {
        "finite_center": bool(np.isfinite(center).all()),
        "finite_extent": bool(np.isfinite(extent).all()),
        "finite_rotation": bool(np.isfinite(rotation).all()),
        "finite_corners": bool(np.isfinite(corners).all()),
        "positive_extent": bool(np.all(extent > 0.0)),
        "right_handed_orthonormal": True,
        "in_front": camera_depth > 1e-4,
    }
    if any(validity[name] is not value for name, value in recomputed_flags.items()):
        raise F4OracleError(f"{label}.validity flags disagree with sealed geometry")
    expected_corners = _SIGNS * (extent[None, :] / 2.0)
    expected_corners = expected_corners @ rotation.T + center[None, :]
    # Row order is frozen by the core, so this is deliberately not a set test.
    if not np.allclose(
        corners, expected_corners, rtol=GEOMETRY_RTOL, atol=GEOMETRY_ATOL
    ):
        raise F4OracleError(f"{label}.world_corners disagree with center/extent/rotation")
    world = np.concatenate((corners.min(axis=0), corners.max(axis=0)))
    if np.any(world[3:] <= world[:3]):
        raise F4OracleError(f"{label} produces a degenerate world AABB")
    aligned = align_boxer_world_corners(corners, alignment, f"{label}.world_corners")
    return True, world, aligned


def grouped_iou_matrix(
    matrices: Mapping[str, np.ndarray], hypotheses: Sequence[str]
) -> np.ndarray:
    """Collapse alternative geometry edges to one row per source."""

    expected = tuple(hypotheses)
    if not expected or any(name not in HYPOTHESES for name in expected):
        raise F4OracleError("grouped hypothesis set is invalid")
    if tuple(matrices) != HYPOTHESES:
        raise F4OracleError(
            f"hypothesis order must be exactly {HYPOTHESES}, got {tuple(matrices)}"
        )
    arrays = [np.asarray(matrices[name], dtype=np.float64) for name in expected]
    if any(array.ndim != 2 or not np.isfinite(array).all() for array in arrays):
        raise F4OracleError("hypothesis IoU matrices must be finite and two-dimensional")
    if len({array.shape for array in arrays}) != 1:
        raise F4OracleError("hypothesis IoU matrices must have identical shapes")
    return np.maximum.reduce(arrays)


def choose_hypothesis_for_edge(
    matrices: Mapping[str, np.ndarray],
    source_index: int,
    gt_index: int,
    hypotheses: Sequence[str],
) -> tuple[str, float]:
    """Choose greatest IoU; iteration order implements the frozen exact tie."""

    names = tuple(hypotheses)
    if not names or any(name not in HYPOTHESES for name in names):
        raise F4OracleError("hypothesis tie order is invalid")
    values = [float(matrices[name][source_index, gt_index]) for name in names]
    if not np.isfinite(values).all():
        raise F4OracleError("matched hypothesis IoU is non-finite")
    best = int(np.argmax(np.asarray(values, dtype=np.float64)))
    return names[best], values[best]


def _empty_rows(columns: int) -> np.ndarray:
    return np.empty((0, columns), dtype=np.float64)


def _edge_delta(
    old_matrix: np.ndarray, new_matrix: np.ndarray, threshold: float
) -> dict[str, int]:
    old = np.asarray(old_matrix, dtype=np.float64) > threshold
    new = np.asarray(new_matrix, dtype=np.float64) > threshold
    if old.shape != new.shape:
        raise F4OracleError("edge comparison shape mismatch")
    gained = new & ~old
    lost = old & ~new
    return {
        "gained_gt_edges": int(np.count_nonzero(gained)),
        "lost_gt_edges": int(np.count_nonzero(lost)),
        "retained_gt_edges": int(np.count_nonzero(old & new)),
        "sources_with_any_gained_edge": int(np.count_nonzero(np.any(gained, axis=1))),
        "sources_with_any_lost_edge": int(np.count_nonzero(np.any(lost, axis=1))),
    }


def evaluate_f4_threshold(
    *,
    scenes: Sequence[str],
    native_iou: Sequence[np.ndarray],
    hypothesis_iou: Sequence[Mapping[str, np.ndarray]],
    sources: Sequence[Sequence[F4Source]],
    gt_counts: Sequence[int],
    baseline_evaluation: Mapping[str, Any],
    threshold: float,
) -> dict[str, Any]:
    """Evaluate four hypotheses, Gbase, and source-identity-constrained G4."""

    lengths = {
        len(scenes), len(native_iou), len(hypothesis_iou), len(sources), len(gt_counts)
    }
    if len(lengths) != 1:
        raise F4OracleError("per-scene F4 inputs have inconsistent lengths")
    baseline_masks = baseline_evaluation.get("matched_gt_masks")
    if not isinstance(baseline_masks, list) or len(baseline_masks) != len(scenes):
        raise F4OracleError("official baseline masks have inconsistent scene count")

    accumulators: dict[str, dict[str, Any]] = {
        mode: {
            "candidate_mm": 0,
            "union_mm": 0,
            "native_mm": 0,
            "selected": 0,
            "suffix": [],
            "selection": {},
            "per_scene": {},
            "chosen": {name: 0 for name in HYPOTHESES},
        }
        for mode in MODES
    }
    g4_edge_delta = {
        "gained_gt_edges": 0,
        "lost_gt_edges": 0,
        "retained_gt_edges": 0,
        "sources_with_any_gained_edge": 0,
        "sources_with_any_lost_edge": 0,
    }

    for scene, native, matrices, scene_sources, gt_count, official_mask in zip(
        scenes, native_iou, hypothesis_iou, sources, gt_counts, baseline_masks
    ):
        if tuple(matrices) != HYPOTHESES:
            raise F4OracleError(f"hypothesis order mismatch: {scene}")
        native_array = np.asarray(native, dtype=np.float64)
        if native_array.ndim != 2 or native_array.shape[1] != gt_count:
            raise F4OracleError(f"native IoU/GT shape mismatch: {scene}")
        for name in HYPOTHESES:
            matrix = np.asarray(matrices[name], dtype=np.float64)
            if matrix.shape != (len(scene_sources), gt_count) or not np.isfinite(matrix).all():
                raise F4OracleError(f"{name} IoU/source/GT shape mismatch: {scene}")
        gbase = grouped_iou_matrix(matrices, BASE_HYPOTHESES)
        g4 = grouped_iou_matrix(matrices, HYPOTHESES)
        delta = _edge_delta(gbase, g4, threshold)
        for key, value in delta.items():
            g4_edge_delta[key] += value
        evaluated = {**matrices, "GBASE": gbase, "G4": g4}
        native_pairs = strict_maximum_matching(native_array, threshold)
        official_unmatched = ~np.asarray(official_mask, dtype=bool)

        for mode in MODES:
            matrix = evaluated[mode]
            if mode == "GBASE":
                tie_order = BASE_HYPOTHESES
            elif mode == "G4":
                tie_order = HYPOTHESES
            else:
                tie_order = (mode,)
            candidate_pairs = strict_maximum_matching(matrix, threshold)
            union_pairs = strict_maximum_matching(
                np.concatenate((native_array, matrix), axis=0), threshold
            )
            suffix_pairs = strict_maximum_matching(
                matrix, threshold, official_unmatched
            )
            target_by_source = {
                source_index: gt_index for source_index, gt_index in suffix_pairs
            }
            selected_indices = sorted(target_by_source)
            selected_matrices: list[np.ndarray] = []
            selected_rows: list[dict[str, Any]] = []
            for source_index in selected_indices:
                gt_index = target_by_source[source_index]
                chosen, chosen_iou = choose_hypothesis_for_edge(
                    matrices, source_index, gt_index, tie_order
                )
                selected_matrices.append(matrices[chosen][source_index])
                accumulators[mode]["chosen"][chosen] += 1
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
            acc = accumulators[mode]
            acc["candidate_mm"] += len(candidate_pairs)
            acc["union_mm"] += len(union_pairs)
            acc["native_mm"] += len(native_pairs)
            acc["selected"] += len(selected_indices)
            acc["suffix"].append(suffix)
            acc["selection"][scene] = selected_rows
            acc["per_scene"][scene] = {
                "native_maximum_matching_count": len(native_pairs),
                "candidate_maximum_matching_count": len(candidate_pairs),
                "union_maximum_matching_count": len(union_pairs),
                "additional_union_matching_over_native": len(union_pairs)
                - len(native_pairs),
                "gt_selected_suffix_count": len(selected_indices),
            }

    baseline_ap = float(baseline_evaluation["ap_points"])
    reports: dict[str, Any] = {}
    for mode in MODES:
        acc = accumulators[mode]
        combined = [
            np.concatenate((native, suffix), axis=0)
            for native, suffix in zip(native_iou, acc["suffix"])
        ]
        evaluation = official_constant_evaluate(combined, gt_counts, threshold)
        delta_ap = float(evaluation["ap_points"]) - baseline_ap
        additional = int(acc["union_mm"] - acc["native_mm"])
        reports[mode] = {
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
                "hypothesis_tie_order": list(
                    BASE_HYPOTHESES if mode == "GBASE" else HYPOTHESES
                ) if mode in ("GBASE", "G4") else [mode],
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

    gbase_report = reports["GBASE"]
    g4_report = reports["G4"]
    g4_minus_gbase = {
        "additional_union_matching_gain": g4_report[
            "additional_union_matching_over_native"
        ]
        - gbase_report["additional_union_matching_over_native"],
        "candidate_maximum_matching_gain": g4_report[
            "candidate_maximum_matching_count"
        ]
        - gbase_report["candidate_maximum_matching_count"],
        "constructive_suffix_selected_source_gain": g4_report[
            "gt_selected_candidate_suffix"
        ]["selected_source_count"]
        - gbase_report["gt_selected_candidate_suffix"]["selected_source_count"],
        "constructive_suffix_ap_point_gain": g4_report[
            "gt_selected_candidate_suffix"
        ]["official_evaluation"]["ap_points"]
        - gbase_report["gt_selected_candidate_suffix"]["official_evaluation"][
            "ap_points"
        ],
        "edges": g4_edge_delta,
    }
    if g4_minus_gbase["additional_union_matching_gain"] < 0:
        raise F4OracleError("adding HB unexpectedly reduced grouped capacity")
    return {
        "iou_threshold": threshold,
        "strict_iou_comparison": ">",
        "baseline_official_constant_score": _json_evaluation(
            baseline_evaluation, scenes
        ),
        "hypothesis_only": {name: reports[name] for name in HYPOTHESES},
        "identity_constrained_gbase": reports["GBASE"],
        "identity_constrained_g4": reports["G4"],
        "g4_minus_gbase": g4_minus_gbase,
    }


def _reproduction_scalars(report: Mapping[str, Any]) -> dict[str, Any]:
    suffix = report["gt_selected_candidate_suffix"]
    chosen = suffix["chosen_hypothesis_counts"]
    return {
        "candidate_maximum_matching_count": report["candidate_maximum_matching_count"],
        "union_maximum_matching_count": report["union_maximum_matching_count"],
        "native_maximum_matching_count": report["native_maximum_matching_count"],
        "additional_union_matching_over_native": report[
            "additional_union_matching_over_native"
        ],
        "selected_source_count": suffix["selected_source_count"],
        # The historical F2 schema predates HB.  Ignore only the necessarily
        # zero HB diagnostic while requiring every F2 hypothesis count.
        "chosen_hypothesis_counts": {
            name: int(chosen.get(name, 0)) for name in BASE_HYPOTHESES
        },
        "suffix_ap_points": suffix["official_evaluation"]["ap_points"],
        "delta_ap_points": suffix["delta_ap_points"],
    }


def validate_f2_reproduction(
    per_threshold: Mapping[str, Any], f2_report: Mapping[str, Any]
) -> dict[str, Any]:
    """Require exact H0/HL/HLG and Gbase reproduction of the sealed F2 audit."""

    checks: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        key = _threshold_key(threshold)
        actual_threshold = per_threshold[key]
        expected_threshold = f2_report["per_threshold"][key]
        mode_checks: dict[str, Any] = {}
        for mode in (*BASE_HYPOTHESES, "GBASE"):
            actual = (
                actual_threshold["identity_constrained_gbase"]
                if mode == "GBASE"
                else actual_threshold["hypothesis_only"][mode]
            )
            expected = (
                expected_threshold["identity_constrained_grouped"]
                if mode == "GBASE"
                else expected_threshold["hypothesis_only"][mode]
            )
            left = _reproduction_scalars(actual)
            right = _reproduction_scalars(expected)
            scalar_pass = all(
                left[name] == right[name]
                for name in left
                if name not in ("suffix_ap_points", "delta_ap_points")
            )
            numeric_pass = all(
                math.isclose(
                    float(left[name]), float(right[name]), rel_tol=0.0, abs_tol=1e-12
                )
                for name in ("suffix_ap_points", "delta_ap_points")
            )
            mode_checks[mode] = {
                "passed": scalar_pass and numeric_pass,
                "actual": left,
                "expected": right,
            }
            if not mode_checks[mode]["passed"]:
                raise F4OracleError(
                    f"F4 {mode} failed to reproduce sealed F2 at IoU {key}"
                )
        checks[key] = {
            "passed": all(row["passed"] for row in mode_checks.values()),
            "modes": mode_checks,
        }
    return checks


def f4_stopping_decision(
    *,
    per_threshold: Mapping[str, Any],
    integrity_passed: bool,
    causality_passed: bool,
    runtime_passed: bool,
) -> dict[str, Any]:
    """Apply the preregistered 144-match/+10-point gate without enabling birth."""

    if any(type(value) is not bool for value in (
        integrity_passed, causality_passed, runtime_passed
    )):
        raise F4OracleError("F4 integrity/causality/runtime gates must be booleans")
    rows = []
    for threshold in THRESHOLDS:
        key = _threshold_key(threshold)
        try:
            rows.append(per_threshold[key]["identity_constrained_g4"])
        except (KeyError, TypeError) as error:
            raise F4OracleError(f"missing F4 G4 result at IoU {key}") from error
    geometry_pass = all(
        int(row["additional_union_matching_over_native"])
        >= REQUIRED_ADDITIONAL_MATCHES
        for row in rows
    )
    suffix_pass = all(
        float(row["gt_selected_candidate_suffix"]["delta_ap_points"])
        >= TARGET_DELTA_AP_POINTS
        for row in rows
    )
    overall = (
        integrity_passed
        and causality_passed
        and runtime_passed
        and geometry_pass
        and suffix_pass
    )
    return {
        "integrity_passed": integrity_passed,
        "causality_passed": causality_passed,
        "runtime_passed": runtime_passed,
        "required_additional_union_matches_each_threshold": REQUIRED_ADDITIONAL_MATCHES,
        "target_delta_ap_points_each_threshold": TARGET_DELTA_AP_POINTS,
        "g4_geometry_capacity_passes_all_thresholds": geometry_pass,
        "constructive_suffix_plus10_ap_passes_all_thresholds": suffix_pass,
        "overall_pass": overall,
        "retain_f4_for_preregistered_gt_free_selector": overall,
        "authorize_active_birth": False,
        "result": (
            "f4_pass_authorize_new_preregistered_gt_free_selector_only"
            if overall
            else "discard_f4_shadow"
        ),
    }


def _gate_passes(actual: float, comparator: str, threshold: float) -> bool:
    comparisons = {
        "<": lambda: actual < threshold,
        "<=": lambda: actual <= threshold,
        "==": lambda: actual == threshold,
        ">=": lambda: actual >= threshold,
        ">": lambda: actual > threshold,
    }
    if comparator not in comparisons:
        raise F4OracleError(f"unsupported runtime comparator: {comparator}")
    return bool(comparisons[comparator]())


def _validate_distribution(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != {"count", "mean", "p50", "p95", "max"}:
        raise F4OracleError(f"{label} is not a frozen runtime distribution")
    if type(value["count"]) is not int or value["count"] < 0:
        raise F4OracleError(f"{label}.count must be a nonnegative integer")
    for name in ("mean", "p50", "p95", "max"):
        _finite_number(value[name], f"{label}.{name}", minimum=0.0)
    if value["p50"] > value["max"] or value["p95"] > value["max"]:
        raise F4OracleError(f"{label} quantiles exceed its maximum")
    return value


def _validate_runtime(
    receipt: Mapping[str, Any],
    *,
    expected_scene_count: int,
    expected_keyframe_count: int,
    expected_successful_frame_count: int,
    expected_source_count: int,
) -> dict[str, Any]:
    runtime = receipt.get("runtime")
    gates = receipt.get("gates")
    if not isinstance(runtime, dict) or not isinstance(gates, dict) or set(gates) != set(MERGE_GATE_NAMES):
        raise F4OracleError("F4 merge gate set is not the frozen set")
    expected_gates: dict[str, tuple[str, float]] = {
        "integrity_complete": ("==", float(expected_scene_count)),
        "exact_keyframes": ("==", float(expected_keyframe_count)),
        "exact_successful_frames": ("==", float(expected_successful_frame_count)),
        "exact_sources": ("==", float(expected_source_count)),
        **RUNTIME_GATE_SPECS,
        "native_output_mutation_count": ("==", 0.0),
    }
    checked: dict[str, Any] = {}
    for name, (expected_comparator, expected_threshold) in expected_gates.items():
        row = gates.get(name)
        if not isinstance(row, dict):
            raise F4OracleError(f"F4 merge gate is missing: {name}")
        actual = row.get("actual")
        threshold = row.get("threshold")
        comparator = row.get("comparator")
        passed = row.get("pass")
        passed_alias = row.get("passed")
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isfinite(float(actual))
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or comparator != expected_comparator
            or not math.isclose(
                float(threshold), expected_threshold, rel_tol=0.0, abs_tol=0.0
            )
            or type(passed) is not bool
            or type(passed_alias) is not bool
            or passed_alias is not passed
        ):
            raise F4OracleError(f"invalid frozen F4 merge gate: {name}")
        recomputed = _gate_passes(
            float(actual), str(comparator), float(threshold)
        )
        if passed is not recomputed:
            raise F4OracleError(f"inconsistent F4 merge gate result: {name}")
        if not passed:
            raise F4OracleError(f"F4 merge gate failed before GT access: {name}")
        checked[name] = row

    if runtime.get("gates") != {
        name: gates[name] for name in RUNTIME_GATE_SPECS
    }:
        raise F4OracleError("F4 runtime gate mirror differs from merged gates")

    incremental = _validate_distribution(
        runtime.get("f4_incremental_warm_ms"), "F4 incremental warm runtime"
    )
    composed = _validate_distribution(
        runtime.get("replay_composed_warm_ms"), "F4 composed warm runtime"
    )
    _validate_distribution(
        runtime.get("replay_composed_all_ms"), "F4 composed all runtime"
    )
    runtime_actuals = {
        "f4_incremental_warm_p95_ms": incremental["p95"],
        "replay_composed_warm_p95_ms": composed["p95"],
        "replay_composed_warm_max_ms": composed["max"],
        "replay_composed_mean_per_source_frame_ms": runtime.get(
            "replay_composed_mean_per_source_frame_ms"
        ),
        "gap25_warm_deadline_miss_count": runtime.get(
            "gap25_warm_deadline_miss_count"
        ),
        "cuda_peak_memory_bytes": runtime.get("cuda_peak_memory_bytes"),
    }
    for name, actual in runtime_actuals.items():
        _finite_number(actual, f"F4 runtime.{name}", minimum=0.0)
        if float(actual) != float(gates[name]["actual"]):
            raise F4OracleError(f"F4 runtime summary/gate mismatch: {name}")
    all_deadline_misses = runtime.get("gap25_all_deadline_miss_count")
    warm_deadline_misses = runtime.get("gap25_warm_deadline_miss_count")
    if (
        type(all_deadline_misses) is not int
        or type(warm_deadline_misses) is not int
        or all_deadline_misses < 0
        or warm_deadline_misses < 0
        or warm_deadline_misses > all_deadline_misses
    ):
        raise F4OracleError("F4 all/warm deadline-miss diagnostics are inconsistent")
    warmup = runtime.get("warmup_forward_count_per_shard")
    if (
        runtime.get("cold_model_load_excluded") is not True
        or not isinstance(warmup, list)
        or len(warmup) != 2
        or any(type(value) is not int or value < 0 or value > 3 for value in warmup)
    ):
        raise F4OracleError("F4 runtime warm-up/cold-load contract mismatch")
    if runtime.get("overall_pass") is not True:
        raise F4OracleError("F4 merged runtime overall_pass must be true")
    return {"gates": checked, "overall_pass": True}


def _validate_causality(receipt: Mapping[str, Any]) -> dict[str, Any]:
    causality = receipt.get("causality")
    if not isinstance(causality, dict) or causality != CAUSALITY_EXPECTED:
        raise F4OracleError("F4 merged causality contract differs or failed")
    return {"items": dict(causality), "overall_pass": True}


def _receipt_seal(row: Mapping[str, Any], scene: str) -> tuple[Path, str]:
    seal = row.get("sidecar")
    if not isinstance(seal, dict):
        raise F4OracleError(f"F4 sidecar seal missing: {scene}")
    path = seal.get("path")
    sha256 = _hash_string(seal.get("sha256"), f"F4 sidecar SHA-256 for {scene}")
    if not isinstance(path, str) or not path:
        raise F4OracleError(f"F4 sidecar path missing: {scene}")
    return Path(path), sha256


def _validate_f4_receipt(
    receipt: Mapping[str, Any],
    scenes: Sequence[str],
    *,
    expected_scene_count: int = EXPECTED["scene_count"],
    expected_keyframe_count: int = EXPECTED["keyframe_count"],
    expected_successful_frame_count: int = EXPECTED["successful_frame_count"],
    expected_source_count: int = EXPECTED["candidate_count"],
) -> tuple[dict[str, Mapping[str, Any]], str, dict[str, Any], dict[str, Any]]:
    for key, expected in {
        "schema": F4_RECEIPT_SCHEMA,
        "protocol_id": F4_PROTOCOL_ID,
        "complete": True,
        "overall_pass": True,
    }.items():
        if receipt.get(key) != expected:
            raise F4OracleError(f"F4 merge contract mismatch for {key}")
    contracts = receipt.get("contracts")
    required_contracts = {
        "shadow_only": True,
        "birth_enabled": False,
        "native_output_mutation": False,
        "gt_access": False,
        "prediction_access": False,
        "evaluator_access": False,
        "future_frame_access": False,
        "training": False,
        "online_learning": False,
    }
    if not isinstance(contracts, dict) or any(
        contracts.get(key) != value for key, value in required_contracts.items()
    ):
        raise F4OracleError("F4 merge shadow/training-free contracts mismatch")
    content_sha256 = _hash_string(
        receipt.get("content_sha256"), "F4 merge content SHA-256"
    )
    content_payload = dict(receipt)
    content_payload.pop("content_sha256", None)
    if _canonical_json_sha256(content_payload) != content_sha256:
        raise F4OracleError("F4 merge content hash mismatch")
    signature = _hash_string(
        receipt.get("run_signature_sha256"), "F4 merge run signature"
    )
    coverage = receipt.get("coverage")
    if (
        len(scenes) != expected_scene_count
        or not isinstance(coverage, dict)
        or coverage.get("scene_count") != expected_scene_count
        or coverage.get("scene_order") != list(scenes)
        or coverage.get("keyframe_count") != expected_keyframe_count
        or coverage.get("successful_frame_count")
        != expected_successful_frame_count
        or coverage.get("source_count") != expected_source_count
        or coverage.get("exact_source_partition") is not True
        or coverage.get("exact_source_order") is not True
    ):
        raise F4OracleError("F4 merge paper100 coverage/order mismatch")
    _hash_string(coverage.get("source_ids_sha256"), "F4 source-order aggregate")
    _hash_string(
        coverage.get("source_lineage_sha256"), "F4 source-lineage aggregate"
    )
    totals = receipt.get("totals")
    required_totals = {
        "keyframe_count": expected_keyframe_count,
        "successful_frame_count": expected_successful_frame_count,
        "source_count": expected_source_count,
        "identity_verified_source_count": expected_source_count,
    }
    if not isinstance(totals, dict) or any(
        totals.get(key) != value for key, value in required_totals.items()
    ):
        raise F4OracleError("F4 merge census/source-identity totals mismatch")
    for name in ("provider_forward_count", "valid_hb_count", "invalid_hb_count"):
        if type(totals.get(name)) is not int or totals[name] < 0:
            raise F4OracleError(f"F4 merge total is invalid: {name}")
    if totals["valid_hb_count"] + totals["invalid_hb_count"] != expected_source_count:
        raise F4OracleError("F4 merge HB validity totals do not partition sources")
    if receipt.get("native_output_mutation_count") != 0:
        raise F4OracleError("F4 merge reports native output mutation")
    authorization = receipt.get("oracle_authorization")
    if not isinstance(authorization, dict) or authorization != {
        "allowed": True,
        "scope": "separate_post_seal_f4_geometry_capacity_oracle_only",
        "active_birth_authorized": False,
    }:
        raise F4OracleError("F4 merge oracle authorization contract mismatch")
    rows = receipt.get("scenes")
    if not isinstance(rows, list) or len(rows) != len(scenes):
        raise F4OracleError("F4 merge scene seal ledger mismatch")
    result: dict[str, Mapping[str, Any]] = {}
    for scene_index, (scene, row) in enumerate(zip(scenes, rows)):
        if (
            not isinstance(row, dict)
            or row.get("scene_id") != scene
            or row.get("scene_index") != scene_index
        ):
            raise F4OracleError("F4 merge scene order mismatch")
        _receipt_seal(row, scene)
        result[scene] = row
    runtime = _validate_runtime(
        receipt,
        expected_scene_count=expected_scene_count,
        expected_keyframe_count=expected_keyframe_count,
        expected_successful_frame_count=expected_successful_frame_count,
        expected_source_count=expected_source_count,
    )
    causality = _validate_causality(receipt)
    return result, signature, runtime, causality


def _source_identity_from_f2(
    *,
    scene: str,
    scene_index: int,
    frame_id: int,
    frame_ordinal: int,
    candidate_index: int,
    f2_row: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "source_id": f2_row.get("source_id"),
        "scene_index": scene_index,
        "frame_ordinal": frame_ordinal,
        "frame_id": frame_id,
        "rank": f2_row.get("rank"),
        "raw_index": f2_row.get("raw_index"),
        "candidate_index": candidate_index,
        "mask_sha256": f2_row.get("mask_sha256"),
        "points_and_voxel_keys_sha256": f2_row.get(
            "points_and_voxel_keys_sha256"
        ),
    }


def _load_f4_sources_pre_gt(
    *,
    path: Path,
    f2_path: Path,
    f0_path: Path,
    scene: str,
    scene_index: int,
    receipt_sidecar_sha256: str,
    run_signature_sha256: str,
) -> tuple[list[_PrevalidatedF4Source], int, int]:
    """Validate a complete scene/source ledger without opening GT or predictions."""

    actual_sha = _sha256(_regular_file(path, f"F4 sidecar for {scene}"))
    if actual_sha != receipt_sidecar_sha256:
        raise F4OracleError(f"F4 merge sidecar hash mismatch: {scene}")
    payload = _read_json(path, f"F4 sidecar for {scene}")
    for key, expected in {
        "schema": F4_SCENE_SCHEMA,
        "protocol_id": F4_PROTOCOL_ID,
        "complete": True,
        "scene_id": scene,
        "scene_index": scene_index,
        "run_signature_sha256": run_signature_sha256,
    }.items():
        if payload.get(key) != expected:
            raise F4OracleError(f"F4 sidecar contract mismatch: {scene}.{key}")
    content_sha256 = _hash_string(
        payload.get("content_sha256"), f"F4 sidecar content SHA-256 for {scene}"
    )
    content_payload = dict(payload)
    content_payload.pop("content_sha256", None)
    if _canonical_json_sha256(content_payload) != content_sha256:
        raise F4OracleError(f"F4 sidecar content hash mismatch: {scene}")
    contracts = payload.get("contracts")
    required_contracts = {
        "shadow_only": True,
        "birth_enabled": False,
        "native_output_mutation": False,
        "gt_access": False,
        "prediction_access": False,
        "evaluator_access": False,
        "future_frame_access": False,
        "training": False,
        "online_learning": False,
    }
    if not isinstance(contracts, dict) or any(
        contracts.get(key) != value for key, value in required_contracts.items()
    ):
        raise F4OracleError(f"F4 sidecar contracts mismatch: {scene}")
    if payload.get("native_output_mutation_count") != 0:
        raise F4OracleError(f"F4 sidecar reports native output mutation: {scene}")

    f2 = _read_json(f2_path, f"F2 sidecar for {scene}")
    f0 = _read_json(f0_path, f"F0 sidecar for {scene}")
    if (
        f2.get("schema") != F2_SCENE_SCHEMA
        or f2.get("protocol_id") != F2_PROTOCOL_ID
        or f2.get("scene_id") != scene
        or f0.get("schema") != F0_SCENE_SCHEMA
        or f0.get("scene_id") != scene
    ):
        raise F4OracleError(f"F0/F2 source anchor mismatch: {scene}")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise F4OracleError(f"F4 frozen-input receipt missing: {scene}")
    for name, expected_path in (("f2_sidecar", f2_path), ("f0_sidecar", f0_path)):
        seal = inputs.get(name)
        if (
            not isinstance(seal, dict)
            or seal.get("path") != os.fspath(expected_path.resolve())
            or seal.get("sha256") != _sha256(_regular_file(expected_path, name))
        ):
            raise F4OracleError(f"F4 frozen-input seal mismatch: {scene}:{name}")
    if inputs.get("frozen_inputs_before_sha256") != inputs.get(
        "frozen_inputs_after_sha256"
    ):
        raise F4OracleError(f"F4 frozen inputs changed during replay: {scene}")
    _hash_string(inputs.get("model_receipts_sha256"), f"F4 model receipts for {scene}")
    frames = payload.get("frames")
    f2_frames = f2.get("frames")
    f0_frames = f0.get("frames")
    if (
        not isinstance(frames, list)
        or not isinstance(f2_frames, list)
        or not isinstance(f0_frames, list)
        or len(frames) != len(f2_frames)
        or len(frames) != len(f0_frames)
    ):
        raise F4OracleError(f"F4/F2/F0 frame ledger mismatch: {scene}")

    result: list[_PrevalidatedF4Source] = []
    successful = 0
    for ordinal, (frame, f2_frame, f0_frame) in enumerate(
        zip(frames, f2_frames, f0_frames)
    ):
        if not all(isinstance(row, dict) for row in (frame, f2_frame, f0_frame)):
            raise F4OracleError(f"invalid F4/F2/F0 frame row: {scene}:{ordinal}")
        for key in ("frame_id", "frame_ordinal", "successful"):
            if frame.get(key) != f2_frame.get(key) or frame.get(key) != f0_frame.get(key):
                raise F4OracleError(
                    f"F4/F2/F0 frame identity mismatch: {scene}:{ordinal}:{key}"
                )
        frame_id = frame.get("frame_id")
        if type(frame_id) is not int or frame.get("frame_ordinal") != ordinal:
            raise F4OracleError(f"invalid F4 frame identity: {scene}:{ordinal}")
        if frame.get("current_only") is not True:
            raise F4OracleError(f"F4 frame is not current-only: {scene}:{ordinal}")
        f4_rows = frame.get("sources")
        f2_rows = f2_frame.get("sources")
        funnel = f0_frame.get("funnel")
        f0_rows = funnel.get("candidates") if isinstance(funnel, dict) else []
        f0_masks = funnel.get("masks") if isinstance(funnel, dict) else []
        if not isinstance(f4_rows, list) or not isinstance(f2_rows, list):
            raise F4OracleError(f"F4/F2 source rows missing: {scene}:{frame_id}")
        if frame.get("successful") is True:
            successful += 1
            if (
                not isinstance(f0_rows, list)
                or not isinstance(f0_masks, list)
                or frame.get("max_accessed_frame_ordinal") != ordinal
                or frame.get("provider_invoked") is not bool(f4_rows)
            ):
                raise F4OracleError(f"successful F4 frame contract differs: {scene}:{frame_id}")
        else:
            f0_rows = []
            f0_masks = []
            if (
                frame.get("provider_invoked") is not False
                or frame.get("runtime") is not None
                or f4_rows
            ):
                raise F4OracleError(f"abstained F4 frame contract differs: {scene}:{frame_id}")
        if len(f4_rows) != len(f2_rows) or len(f4_rows) != len(f0_rows):
            raise F4OracleError(f"F4/F2/F0 source count mismatch: {scene}:{frame_id}")
        selected_masks: dict[tuple[Any, Any, Any], Mapping[str, Any]] = {}
        for mask in f0_masks:
            if isinstance(mask, dict) and mask.get("selected") is True:
                key = (mask.get("rank"), mask.get("raw_index"), mask.get("mask_sha256"))
                if key in selected_masks:
                    raise F4OracleError(f"ambiguous F0 selected-mask join: {scene}:{frame_id}")
                selected_masks[key] = mask
        if len(selected_masks) != len(f4_rows):
            raise F4OracleError(f"F0 selected-mask census mismatch: {scene}:{frame_id}")
        for candidate_index, (row, f2_row, f0_row) in enumerate(
            zip(f4_rows, f2_rows, f0_rows)
        ):
            if not all(isinstance(item, dict) for item in (row, f2_row, f0_row)):
                raise F4OracleError(
                    f"invalid F4 source row: {scene}:{frame_id}:{candidate_index}"
                )
            expected_identity = _source_identity_from_f2(
                scene=scene,
                scene_index=scene_index,
                frame_id=frame_id,
                frame_ordinal=ordinal,
                candidate_index=candidate_index,
                f2_row=f2_row,
            )
            for key, expected in expected_identity.items():
                if row.get(key) != expected:
                    raise F4OracleError(
                        f"F4/F2 source identity mismatch: {scene}:{frame_id}:"
                        f"{candidate_index}:{key}"
                    )
            expected_source_id = (
                f"{scene}/frame_{frame_id:06d}/raw_{int(row['raw_index']):03d}"
            )
            if row["source_id"] != expected_source_id:
                raise F4OracleError(f"non-canonical F4 source ID: {row['source_id']}")
            if row.get("rank") != candidate_index or row.get("candidate_index") != candidate_index:
                raise F4OracleError(f"F4 source rank/order mismatch: {row['source_id']}")
            for key in ("rank", "raw_index", "mask_sha256", "points_and_voxel_keys_sha256"):
                if f0_row.get(key) != row.get(key):
                    raise F4OracleError(
                        f"F4/F0 source identity mismatch: {row['source_id']}:{key}"
                    )
            mask_key = (row.get("rank"), row.get("raw_index"), row.get("mask_sha256"))
            f0_mask = selected_masks.get(mask_key)
            if (
                not isinstance(f0_mask, dict)
                or f0_mask.get("decision") != "selected"
                or f0_mask.get("tight_box_xyxy") != f0_row.get("tight_box_xyxy")
            ):
                raise F4OracleError(f"F4/F0 selected-mask join mismatch: {row['source_id']}")
            for f2_key, f0_key in (
                ("confidence", "confidence"),
                ("stored_point_count", "stored_point_count"),
                ("f0_world_q02", "world_q02"),
                ("f0_world_q98", "world_q98"),
            ):
                if f2_row.get(f2_key) != f0_row.get(f0_key):
                    raise F4OracleError(f"F4/F2/F0 source field mismatch: {row['source_id']}:{f2_key}")
            tight = _tight_box(
                row.get("tight_box_xyxy"), f"{row['source_id']}.tight_box_xyxy"
            )
            if list(tight) != [float(value) for value in f0_row.get("tight_box_xyxy", [])]:
                raise F4OracleError(f"F4/F0 tight-box mismatch: {row['source_id']}")
            hypotheses = row.get("hypotheses")
            f2_hypotheses = f2_row.get("hypotheses")
            if not isinstance(hypotheses, dict) or set(hypotheses) != set(HYPOTHESES):
                raise F4OracleError(f"F4 hypothesis set mismatch: {row['source_id']}")
            if not isinstance(f2_hypotheses, dict) or set(f2_hypotheses) != set(BASE_HYPOTHESES):
                raise F4OracleError(f"F2 hypothesis anchor mismatch: {row['source_id']}")
            copied = {name: hypotheses[name] for name in BASE_HYPOTHESES}
            if copied != f2_hypotheses:
                raise F4OracleError(
                    f"F4 base hypotheses are not exact F2 copies: {row['source_id']}"
                )
            sealed_hash = _hash_string(
                row.get("sealed_f2_hypotheses_sha256"),
                f"{row['source_id']}.sealed_f2_hypotheses_sha256",
            )
            if sealed_hash != _canonical_json_sha256(f2_hypotheses):
                raise F4OracleError(
                    f"F4 sealed F2 hypothesis hash mismatch: {row['source_id']}"
                )
            hb = hypotheses["HB"]
            if not isinstance(hb, dict):
                raise F4OracleError(f"invalid HB row: {row['source_id']}")
            if hb.get("row_index") != candidate_index:
                raise F4OracleError(f"HB row binding mismatch: {row['source_id']}")
            if hb.get("source_id") != row["source_id"]:
                raise F4OracleError(f"HB source binding mismatch: {row['source_id']}")
            hb_box = _tight_box(
                hb.get("input_tight_box_xyxy"),
                f"{row['source_id']}.HB.input_tight_box_xyxy",
            )
            if hb_box != tight:
                raise F4OracleError(f"HB input box mismatch: {row['source_id']}")
            hb_result_sha256 = _hash_string(
                hb.get("result_sha256"), f"{row['source_id']}.HB.result_sha256"
            )
            hb_payload = dict(hb)
            hb_payload.pop("result_sha256", None)
            if _canonical_json_sha256(hb_payload) != hb_result_sha256:
                raise F4OracleError(f"HB result hash mismatch: {row['source_id']}")
            provider_result = hb.get("provider_result_sha256")
            if provider_result is not None:
                _hash_string(provider_result, f"{row['source_id']}.HB.provider_result_sha256")
            identity = {
                key: row[key]
                for key in (
                    "scene_index",
                    "frame_ordinal",
                    "frame_id",
                    "rank",
                    "raw_index",
                    "mask_sha256",
                    "points_and_voxel_keys_sha256",
                    "source_id",
                )
            }
            expected_f0_lineage = {
                "candidate_sha256": _canonical_json_sha256(f0_row),
                "mask_diagnostic_sha256": _canonical_json_sha256(f0_mask),
                "provider_box_ignored": True,
            }
            expected_f2_lineage = {
                "source_sha256": _canonical_json_sha256(f2_row),
                "f2_receipt_result_sha256": (
                    f2_row.get("f2_receipt", {}).get("result_sha256")
                    if isinstance(f2_row.get("f2_receipt"), dict)
                    else None
                ),
            }
            if row.get("f0_source_lineage") != expected_f0_lineage or row.get(
                "f2_source_lineage"
            ) != expected_f2_lineage:
                raise F4OracleError(f"F4 source upstream lineage mismatch: {row['source_id']}")
            join_sha256 = _hash_string(
                row.get("join_sha256"), f"{row['source_id']}.join_sha256"
            )
            expected_join = _canonical_json_sha256(
                {
                    "identity": identity,
                    "f0": expected_f0_lineage,
                    "f2": expected_f2_lineage,
                    "tight_box_xyxy": list(tight),
                }
            )
            if join_sha256 != expected_join:
                raise F4OracleError(f"F4 source join hash mismatch: {row['source_id']}")
            lineage_sha256 = _hash_string(
                row.get("source_lineage_sha256"),
                f"{row['source_id']}.source_lineage_sha256",
            )
            expected_lineage = _canonical_json_sha256(
                {
                    "identity": identity,
                    "join_sha256": join_sha256,
                    "sealed_f2_hypotheses_sha256": sealed_hash,
                    "hb_result_sha256": hb_result_sha256,
                }
            )
            if lineage_sha256 != expected_lineage:
                raise F4OracleError(f"F4 source lineage hash mismatch: {row['source_id']}")
            # Identity alignment is validated pre-GT with an identity transform;
            # the real axis-aligned bounds are computed only after all scenes seal.
            validate_boxer_hypothesis(hb, np.eye(4), f"{row['source_id']}.HB")
            result.append(
                _PrevalidatedF4Source(
                    scene_id=scene,
                    scene_index=scene_index,
                    frame_id=frame_id,
                    frame_ordinal=ordinal,
                    candidate_index=candidate_index,
                    rank=int(row["rank"]),
                    raw_index=int(row["raw_index"]),
                    source_id=str(row["source_id"]),
                    mask_sha256=str(row["mask_sha256"]),
                    points_and_voxel_keys_sha256=str(
                        row["points_and_voxel_keys_sha256"]
                    ),
                    tight_box_xyxy=tight,
                    hb=hb,
                    source_lineage_sha256=lineage_sha256,
                )
            )
    summary = payload.get("counts")
    if not isinstance(summary, dict):
        raise F4OracleError(f"F4 scene counts missing: {scene}")
    expected_summary = {
        "keyframe_count": len(frames),
        "successful_frame_count": successful,
        "source_count": len(result),
    }
    if any(summary.get(key) != value for key, value in expected_summary.items()):
        raise F4OracleError(f"F4 scene census mismatch: {scene}")
    for name in ("provider_forward_count", "valid_hb_count", "invalid_hb_count"):
        if type(summary.get(name)) is not int or summary[name] < 0:
            raise F4OracleError(f"F4 scene census field is invalid: {scene}:{name}")
    if summary.get("valid_hb_count") + summary.get("invalid_hb_count") != len(result):
        raise F4OracleError(f"F4 scene HB census mismatch: {scene}")
    source_ids = [source.source_id for source in result]
    lineages = [source.source_lineage_sha256 for source in result]
    if (
        len(source_ids) != len(set(source_ids))
        or payload.get("source_ids_sha256") != _canonical_json_sha256(source_ids)
        or payload.get("source_lineage_sha256") != _canonical_json_sha256(lineages)
    ):
        raise F4OracleError(f"F4 scene source aggregate mismatch: {scene}")
    return result, len(frames), successful


def _f4_snapshot(
    *,
    scenes: Sequence[str],
    scene_list: Path,
    full_scene_list: Path,
    protocol: Path,
    f0_receipt: Path,
    f0_sidecar_root: Path,
    f1_report: Path,
    f2_receipt: Path,
    f2_report: Path,
    f2_sidecar_root: Path,
    f2_array_root: Path,
    f4_receipt: Path,
    f4_sidecar_root: Path,
    baseline_root: Path,
    gt_root: Path,
    scan_root: Path,
    official_evaluator: Path,
) -> dict[str, Any]:
    snapshot = _f1_input_snapshot(
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
    snapshot["fixed_files"].update(
        {
            "f4_protocol": {
                "path": os.fspath(protocol),
                "sha256": _sha256(_regular_file(protocol, "F4 protocol")),
            },
            "f1_report": {
                "path": os.fspath(f1_report),
                "sha256": _sha256(_regular_file(f1_report, "F1 report")),
            },
            "f2_receipt": {
                "path": os.fspath(f2_receipt),
                "sha256": _sha256(_regular_file(f2_receipt, "F2 receipt")),
            },
            "f2_report": {
                "path": os.fspath(f2_report),
                "sha256": _sha256(_regular_file(f2_report, "F2 report")),
            },
            "f4_receipt": {
                "path": os.fspath(f4_receipt),
                "sha256": _sha256(_regular_file(f4_receipt, "F4 receipt")),
            },
        }
    )
    snapshot["ordered_scene_ledgers"].update(
        {
            "f2_sidecars": canonical_ordered_hash_ledger(
                scenes,
                [f2_sidecar_root / f"{scene}.json" for scene in scenes],
                "F2 sidecar",
            ),
            "f2_evidence_npz": canonical_ordered_hash_ledger(
                scenes,
                [f2_array_root / f"{scene}.npz" for scene in scenes],
                "F2 evidence NPZ",
            ),
            "f4_sidecars": canonical_ordered_hash_ledger(
                scenes,
                [f4_sidecar_root / f"{scene}.json" for scene in scenes],
                "F4 sidecar",
            ),
        }
    )
    return snapshot


def _hypothesis_iou(
    sources: Sequence[F4Source], name: str, gt: np.ndarray
) -> np.ndarray:
    result = np.zeros((len(sources), len(gt)), dtype=np.float64)
    valid_indices = [
        index
        for index, source in enumerate(sources)
        if source.valid[name] and source.aligned_minmax[name] is not None
    ]
    if not valid_indices:
        return result
    boxes = np.stack(
        [sources[index].aligned_minmax[name] for index in valid_indices]
    )
    result[np.asarray(valid_indices, dtype=np.int64)] = aligned_iou_matrix(boxes, gt)
    return result


def audit_scannet_fastsam_f4_boxer_paper100_oracle(
    *,
    scene_list: Path,
    full_scene_list: Path,
    protocol: Path,
    f0_receipt: Path,
    f0_sidecar_root: Path,
    f1_report: Path,
    f2_receipt: Path,
    f2_report: Path,
    f2_sidecar_root: Path,
    f2_array_root: Path,
    f4_receipt: Path,
    f4_sidecar_root: Path,
    baseline_root: Path,
    gt_root: Path,
    scan_root: Path,
    official_evaluator: Path,
) -> dict[str, Any]:
    """Run the one-shot oracle after all shadow-only guards pass."""

    # Phase 1 deliberately has no GT, prediction, evaluator, or axis-alignment
    # deserialization.  It validates the complete F4/F2/F0 identity chain first.
    scenes = load_scene_list(_regular_file(scene_list, "paper100 scene list"))
    full_scenes = load_scene_list(_regular_file(full_scene_list, "F0 full200 scene list"))
    if len(scenes) != EXPECTED["scene_count"] or scenes != full_scenes[: EXPECTED["scene_count"]]:
        raise F4OracleError("frozen paper100 scene order/count mismatch")
    if _sha256(_regular_file(f1_report, "F1 report")) != F1_REPORT_SHA256:
        raise F4OracleError("sealed F1 report SHA-256 mismatch")
    if _sha256(_regular_file(f2_report, "F2 report")) != F2_REPORT_SHA256:
        raise F4OracleError("sealed F2 report SHA-256 mismatch")
    f1 = _read_json(f1_report, "F1 report")
    f2_report_payload = _read_json(f2_report, "F2 report")
    if f1.get("schema") != F1_REPORT_SCHEMA:
        raise F4OracleError("unexpected F1 report schema")
    if f2_report_payload.get("schema") != F2_REPORT_SCHEMA:
        raise F4OracleError("unexpected F2 report schema")
    try:
        f2_rows, f2_signature, f2_pass, _ = _validate_f2_receipt(
            _read_json(f2_receipt, "F2 receipt"), scenes
        )
    except F2OracleError as error:
        raise F4OracleError(f"sealed F2 receipt failed validation: {error}") from error
    if not f2_pass:
        raise F4OracleError("sealed F2 receipt did not pass its runtime/integrity gates")
    f0_rows = _validate_f0_receipt(
        _read_json(f0_receipt, "F0 receipt"), full_scenes
    )
    f4_receipt_payload = _read_json(f4_receipt, "F4 receipt")
    f4_rows, f4_signature, runtime, causality = _validate_f4_receipt(
        f4_receipt_payload, scenes
    )
    prevalidated: dict[str, list[_PrevalidatedF4Source]] = {}
    pre_totals = {"keyframe_count": 0, "successful_frame_count": 0, "source_count": 0}
    for scene_index, scene in enumerate(scenes):
        receipt_path, receipt_sha = _receipt_seal(f4_rows[scene], scene)
        expected_path = f4_sidecar_root / f"{scene}.json"
        if receipt_path.resolve() != expected_path.resolve():
            raise F4OracleError(f"F4 merge sidecar path mismatch: {scene}")
        f2_sidecar = f2_sidecar_root / f"{scene}.json"
        if _sha256(_regular_file(f2_sidecar, f"F2 sidecar for {scene}")) != str(
            f2_rows[scene]["sidecar"]["sha256"]
        ):
            raise F4OracleError(f"F2 receipt sidecar hash mismatch: {scene}")
        evidence = f2_array_root / f"{scene}.npz"
        if _sha256(_regular_file(evidence, f"F2 evidence for {scene}")) != str(
            f2_rows[scene]["evidence_npz"]["sha256"]
        ):
            raise F4OracleError(f"F2 receipt evidence hash mismatch: {scene}")
        sources, keyframes, successful = _load_f4_sources_pre_gt(
            path=expected_path,
            f2_path=f2_sidecar,
            f0_path=f0_sidecar_root / f"{scene}.json",
            scene=scene,
            scene_index=scene_index,
            receipt_sidecar_sha256=receipt_sha,
            run_signature_sha256=f4_signature,
        )
        prevalidated[scene] = sources
        pre_totals["keyframe_count"] += keyframes
        pre_totals["successful_frame_count"] += successful
        pre_totals["source_count"] += len(sources)
        receipt_counts = f4_rows[scene].get("counts")
        if (
            not isinstance(receipt_counts, dict)
            or receipt_counts.get("keyframe_count") != keyframes
            or receipt_counts.get("successful_frame_count") != successful
            or receipt_counts.get("source_count") != len(sources)
            or f4_rows[scene].get("source_ids_sha256")
            != _canonical_json_sha256([source.source_id for source in sources])
            or f4_rows[scene].get("source_lineage_sha256")
            != _canonical_json_sha256(
                [source.source_lineage_sha256 for source in sources]
            )
        ):
            raise F4OracleError(f"F4 merge/scene aggregate mismatch: {scene}")
    expected_pre_totals = {
        "keyframe_count": EXPECTED["keyframe_count"],
        "successful_frame_count": EXPECTED["successful_frame_count"],
        "source_count": EXPECTED["candidate_count"],
    }
    if pre_totals != expected_pre_totals:
        raise F4OracleError(
            f"F4 pre-GT census mismatch: expected={expected_pre_totals}, actual={pre_totals}"
        )
    all_pre_sources = [source for scene in scenes for source in prevalidated[scene]]
    coverage = f4_receipt_payload["coverage"]
    if (
        coverage.get("source_ids_sha256")
        != _canonical_json_sha256([source.source_id for source in all_pre_sources])
        or coverage.get("source_lineage_sha256")
        != _canonical_json_sha256(
            [source.source_lineage_sha256 for source in all_pre_sources]
        )
    ):
        raise F4OracleError("F4 merged global source aggregate mismatch")

    # Phase 2 begins only after the preceding complete-shadow validation.
    before = _f4_snapshot(
        scenes=scenes,
        scene_list=scene_list,
        full_scene_list=full_scene_list,
        protocol=protocol,
        f0_receipt=f0_receipt,
        f0_sidecar_root=f0_sidecar_root,
        f1_report=f1_report,
        f2_receipt=f2_receipt,
        f2_report=f2_report,
        f2_sidecar_root=f2_sidecar_root,
        f2_array_root=f2_array_root,
        f4_receipt=f4_receipt,
        f4_sidecar_root=f4_sidecar_root,
        baseline_root=baseline_root,
        gt_root=gt_root,
        scan_root=scan_root,
        official_evaluator=official_evaluator,
    )
    _validate_frozen_hashes(before)

    gt_counts: list[int] = []
    native_iou: list[np.ndarray] = []
    hypothesis_iou: list[dict[str, np.ndarray]] = []
    sources_by_scene: list[list[F4Source]] = []
    scene_reports: dict[str, Any] = {}
    totals = {
        "scene_count": len(scenes),
        "keyframe_count": 0,
        "successful_frame_count": 0,
        "source_count": 0,
        "hb_valid_source_count": 0,
        "hb_abstained_source_count": 0,
        "native_prediction_count": 0,
        "gt_count": 0,
    }
    for scene_index, scene in enumerate(scenes):
        alignment = load_axis_alignment(scan_root / scene / f"{scene}.txt")
        gt = load_gt_minmax(gt_root / f"{scene}_bbox.npy")
        _, native = load_baseline_boxes(
            baseline_root / f"{scene}_boxes.pkl", alignment
        )
        f0_sources, _, _ = _load_f0_candidates(
            path=f0_sidecar_root / f"{scene}.json",
            scene=scene,
            scene_index=scene_index,
            alignment=alignment,
            receipt_sidecar_sha256=str(f0_rows[scene]["sidecar"]["sha256"]),
        )
        try:
            f2_sources, keyframes, successful = _load_f2_sources(
                path=f2_sidecar_root / f"{scene}.json",
                f0_path=f0_sidecar_root / f"{scene}.json",
                scene=scene,
                scene_index=scene_index,
                alignment=alignment,
                receipt_sidecar_sha256=str(f2_rows[scene]["sidecar"]["sha256"]),
                run_signature_sha256=f2_signature,
            )
        except F2OracleError as error:
            raise F4OracleError(f"F2 source replay failed: {scene}: {error}") from error
        f4_pre = prevalidated[scene]
        if len(f0_sources) != len(f2_sources) or len(f2_sources) != len(f4_pre):
            raise F4OracleError(f"F4 source census does not reproduce F0/F2: {scene}")
        scene_sources: list[F4Source] = []
        for f0_source, f2_source, pre in zip(f0_sources, f2_sources, f4_pre):
            if (
                pre.source_id != f2_source.source_id
                or pre.frame_id != f2_source.frame_id
                or pre.frame_ordinal != f2_source.frame_ordinal
                or pre.candidate_index != f2_source.candidate_index
                or pre.rank != f2_source.rank
                or pre.raw_index != f2_source.raw_index
                or pre.source_id
                != f"{scene}/frame_{pre.frame_id:06d}/raw_{pre.raw_index:03d}"
                or not np.array_equal(
                    f2_source.aligned_minmax["H0"], f0_source.aligned_minmax
                )
            ):
                raise F4OracleError(f"F4/F2/F1 source order mismatch: {pre.source_id}")
            hb_valid, hb_world, hb_aligned = validate_boxer_hypothesis(
                pre.hb, alignment, f"{pre.source_id}.HB"
            )
            world: dict[str, np.ndarray | None] = {
                name: f2_source.world_minmax[name] for name in BASE_HYPOTHESES
            }
            aligned: dict[str, np.ndarray | None] = {
                name: f2_source.aligned_minmax[name] for name in BASE_HYPOTHESES
            }
            valid = {name: True for name in BASE_HYPOTHESES}
            world["HB"] = hb_world
            aligned["HB"] = hb_aligned
            valid["HB"] = hb_valid
            scene_sources.append(
                F4Source(
                    scene_id=scene,
                    scene_index=scene_index,
                    frame_id=pre.frame_id,
                    frame_ordinal=pre.frame_ordinal,
                    candidate_index=pre.candidate_index,
                    rank=pre.rank,
                    raw_index=pre.raw_index,
                    source_id=pre.source_id,
                    mask_sha256=pre.mask_sha256,
                    points_and_voxel_keys_sha256=pre.points_and_voxel_keys_sha256,
                    tight_box_xyxy=pre.tight_box_xyxy,
                    world_minmax=world,
                    aligned_minmax=aligned,
                    valid=valid,
                )
            )
        matrices = {
            name: _hypothesis_iou(scene_sources, name, gt) for name in HYPOTHESES
        }
        hb_valid_count = sum(source.valid["HB"] for source in scene_sources)
        gt_counts.append(len(gt))
        native_iou.append(aligned_iou_matrix(native, gt))
        hypothesis_iou.append(matrices)
        sources_by_scene.append(scene_sources)
        totals["keyframe_count"] += keyframes
        totals["successful_frame_count"] += successful
        totals["source_count"] += len(scene_sources)
        totals["hb_valid_source_count"] += hb_valid_count
        totals["hb_abstained_source_count"] += len(scene_sources) - hb_valid_count
        totals["native_prediction_count"] += len(native)
        totals["gt_count"] += len(gt)
        scene_reports[scene] = {
            "scene_index": scene_index,
            "keyframe_count": keyframes,
            "successful_frame_count": successful,
            "source_count": len(scene_sources),
            "hb_valid_source_count": hb_valid_count,
            "hb_abstained_source_count": len(scene_sources) - hb_valid_count,
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
    if any(totals[key] != value for key, value in expected_totals.items()):
        raise F4OracleError(
            f"F4 paper100 census mismatch: expected={expected_totals}, actual={totals}"
        )
    if totals["hb_valid_source_count"] + totals["hb_abstained_source_count"] != totals[
        "source_count"
    ]:
        raise F4OracleError("F4 HB validity census is not source-complete")

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
            raise F4OracleError(f"official native AP reproduction failed at IoU {key}")
    per_threshold = {
        _threshold_key(threshold): evaluate_f4_threshold(
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
    try:
        h0_checks = validate_h0_reproduces_f1(per_threshold, f1)
    except F2OracleError as error:
        raise F4OracleError(f"F4 H0 failed sealed F1 reproduction: {error}") from error
    f2_checks = validate_f2_reproduction(per_threshold, f2_report_payload)
    integrity_passed = (
        all(row["passed"] for row in h0_checks.values())
        and all(row["passed"] for row in f2_checks.values())
    )
    decision = f4_stopping_decision(
        per_threshold=per_threshold,
        integrity_passed=integrity_passed,
        causality_passed=causality["overall_pass"],
        runtime_passed=runtime["overall_pass"],
    )
    after = _f4_snapshot(
        scenes=scenes,
        scene_list=scene_list,
        full_scene_list=full_scene_list,
        protocol=protocol,
        f0_receipt=f0_receipt,
        f0_sidecar_root=f0_sidecar_root,
        f1_report=f1_report,
        f2_receipt=f2_receipt,
        f2_report=f2_report,
        f2_sidecar_root=f2_sidecar_root,
        f2_array_root=f2_array_root,
        f4_receipt=f4_receipt,
        f4_sidecar_root=f4_sidecar_root,
        baseline_root=baseline_root,
        gt_root=gt_root,
        scan_root=scan_root,
        official_evaluator=official_evaluator,
    )
    if after != before:
        raise F4OracleError("one or more sealed inputs changed during F4 oracle")
    return {
        "schema": SCHEMA,
        "protocol": "F4-FastSAM-F2-frozen-BoxerNet-paper100-source-identity-oracle",
        "oracle_only": True,
        "deployable": False,
        "gt_used": True,
        "birth_enabled": False,
        "native_predictions_modified": False,
        "source_count_not_hypothesis_count": True,
        "hypotheses": list(HYPOTHESES),
        "groups": {
            "Gbase": list(BASE_HYPOTHESES),
            "G4": list(HYPOTHESES),
        },
        "score_mode": "constant_1.0",
        "class_mode": "class_agnostic",
        "strict_iou_comparison": ">",
        "scene_order": scenes,
        "totals": totals,
        "runtime": {
            "summary": f4_receipt_payload.get("runtime"),
            "validated_gates": runtime,
            "overall_pass": runtime["overall_pass"],
        },
        "causality": causality,
        "integrity": {
            "pre_gt_f4_merge_and_all_source_hashes_validated": True,
            "f1_report_sha256": F1_REPORT_SHA256,
            "f2_report_sha256": F2_REPORT_SHA256,
            "f4_receipt_sha256": before["fixed_files"]["f4_receipt"]["sha256"],
            "f4_run_signature_sha256": f4_signature,
            "h0_bitwise_reproduces_f0": True,
            "h0_reproduces_f1_oracle": h0_checks,
            "h0_hl_hlg_gbase_reproduce_f2_oracle": f2_checks,
            "official_baseline_reproduction": baseline_checks,
            "all_inputs_before_after_identity": True,
            "overall_pass": integrity_passed,
        },
        "per_threshold": per_threshold,
        "decision": decision,
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
        raise F4OracleError("F4 oracle output must have a .json suffix")
    if output.exists() or output.is_symlink():
        raise F4OracleError(f"refusing to overwrite F4 oracle output: {output}")
    if any(_is_within(output, root) for root in protected_roots):
        raise F4OracleError("F4 oracle output must not be inside a protected input root")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the sealed FastSAM/F2 + BoxerNet F4 paper100 oracle"
    )
    parser.add_argument("--scene-list", type=Path, default=Path("evaluation/data_util/meta_data/scannetv2_val.txt"))
    parser.add_argument("--full-scene-list", type=Path, default=Path("evaluation/data_util/meta_data/scannetv2_val_f0_full200.txt"))
    parser.add_argument("--protocol", type=Path, default=Path("docs/F4_FASTSAM_BOXER_GEOMETRY_PROTOCOL_FREEZE.md"))
    parser.add_argument("--f0-receipt", type=Path, default=Path("logs/scannet_fastsam_f0_full200_score05/final/F0_FASTSAM_FULL200.json"))
    parser.add_argument("--f0-sidecar-root", type=Path, default=Path("logs/scannet_fastsam_f0_full200_score05/scenes"))
    parser.add_argument("--f1-report", type=Path, default=Path("reports/fastsam_f1_paper100_oracle/F1_FASTSAM_PAPER100_ORACLE.json"))
    parser.add_argument("--f2-receipt", type=Path, default=Path("logs/scannet_fastsam_f2_paper100_score05/final/F2_FASTSAM_PAPER100.json"))
    parser.add_argument("--f2-report", type=Path, default=Path("reports/fastsam_f2_paper100_oracle/F2_FASTSAM_PAPER100_ORACLE.json"))
    parser.add_argument("--f2-sidecar-root", type=Path, default=Path("logs/scannet_fastsam_f2_paper100_score05/scenes"))
    parser.add_argument("--f2-array-root", type=Path, default=Path("logs/scannet_fastsam_f2_paper100_score05/arrays"))
    parser.add_argument("--f4-receipt", type=Path, default=Path("logs/scannet_fastsam_f4_boxer_paper100_score05/final/F4_FASTSAM_BOXER_PAPER100.json"))
    parser.add_argument("--f4-sidecar-root", type=Path, default=Path("logs/scannet_fastsam_f4_boxer_paper100_score05/scenes"))
    parser.add_argument("--baseline-root", type=Path, default=Path("results/scannet_t05_boxer_replay_active_score05"))
    parser.add_argument("--gt-root", type=Path, default=Path("evaluation/data_util/scannet_train_detection_data"))
    parser.add_argument("--scan-root", type=Path, default=Path("/extra/ZhaoX/scannet_data/scans"))
    parser.add_argument("--official-evaluator", type=Path, default=Path("upstream_clean/BoxFusion_shallow/evaluation/eval_scannet.py"))
    parser.add_argument("--out", type=Path, default=Path("reports/fastsam_f4_boxer_paper100_oracle/F4_FASTSAM_BOXER_PAPER100_ORACLE.json"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_output_path(
        args.out,
        (
            args.scene_list.parent,
            args.full_scene_list.parent,
            args.protocol.parent,
            args.f0_receipt.parent,
            args.f0_sidecar_root,
            args.f1_report.parent,
            args.f2_receipt.parent,
            args.f2_report.parent,
            args.f2_sidecar_root,
            args.f2_array_root,
            args.f4_receipt.parent,
            args.f4_sidecar_root,
            args.baseline_root,
            args.gt_root,
            args.scan_root,
            args.official_evaluator.parent,
        ),
    )
    report = audit_scannet_fastsam_f4_boxer_paper100_oracle(
        scene_list=args.scene_list,
        full_scene_list=args.full_scene_list,
        protocol=args.protocol,
        f0_receipt=args.f0_receipt,
        f0_sidecar_root=args.f0_sidecar_root,
        f1_report=args.f1_report,
        f2_receipt=args.f2_receipt,
        f2_report=args.f2_report,
        f2_sidecar_root=args.f2_sidecar_root,
        f2_array_root=args.f2_array_root,
        f4_receipt=args.f4_receipt,
        f4_sidecar_root=args.f4_sidecar_root,
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
