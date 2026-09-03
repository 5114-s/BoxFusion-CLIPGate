#!/usr/bin/env python3
"""Create-only counterfactual materialization of CuTR residual S0 candidates.

The native prediction list is preserved as an exact Python-object prefix.  The
tool reads no ground truth, CLIP feature, or detector model and never edits its
input tree.  It exists only to measure the AP effect of the observer-only S0
candidate set in a separate prediction root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import pickle
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


SUMMARY_PREFIX = "CuTR-residual-birth-lite shadow JSON | "
SUMMARY_SCHEMA = "boxfusion.cutr_residual_birth_lite_shadow.v1"
MANIFEST_SCHEMA = "boxfusion.cutr_residual_shadow_materialization.v1"
R1_SUMMARY_PREFIX = "CuTR-residual-cross-view-R1 shadow JSON | "
R1_SUMMARY_SCHEMA = "boxfusion.cutr_residual_cross_view_r1_shadow.v1"
R1_MANIFEST_SCHEMA = "boxfusion.cutr_residual_cross_view_r1_materialization.v1"
MAX_CANDIDATES = 6
NOVELTY_IOU = 0.10
SELF_NMS_IOU = 0.25


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _unique_object(pairs: Sequence[Tuple[str, object]]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(text: str) -> Mapping[str, object]:
    value = json.loads(
        text,
        parse_constant=_reject_constant,
        object_pairs_hook=_unique_object,
    )
    if not isinstance(value, Mapping):
        raise ValueError("summary JSON must be an object")

    def walk(row: object, where: str) -> None:
        if isinstance(row, Mapping):
            for key, child in row.items():
                walk(child, f"{where}.{key}")
        elif isinstance(row, list):
            for index, child in enumerate(row):
                walk(child, f"{where}[{index}]")
        elif isinstance(row, float) and not math.isfinite(row):
            raise ValueError(f"non-finite value at {where}")

    walk(value, "summary")
    return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_scene_list(path: Path) -> Tuple[str, ...]:
    scenes = tuple(
        row.strip()
        for row in path.read_text(encoding="utf-8").splitlines()
        if row.strip()
    )
    if not scenes or len(set(scenes)) != len(scenes):
        raise ValueError("scene list must be non-empty and unique")
    if any("/" in scene or "\\" in scene or scene in {".", ".."} for scene in scenes):
        raise ValueError("scene ids must be plain file-name components")
    return scenes


def _load_native(path: Path) -> Tuple[List[Tuple[object, object, object]], bytes]:
    raw = path.read_bytes()
    with path.open("rb") as stream:
        value = pickle.load(stream)
        if stream.read() != b"":
            raise ValueError(f"native prediction has trailing bytes: {path}")
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], list):
        raise ValueError(f"native prediction must be [list[box]]: {path}")
    rows = value[0]
    for index, row in enumerate(rows):
        if not isinstance(row, tuple) or len(row) != 3:
            raise ValueError(f"native row {index} must be a 3-tuple")
        label, corners_value, score_value = row
        if isinstance(label, (bool, np.bool_)) or not isinstance(label, Integral) or int(label) != 0:
            raise ValueError(f"native row {index} must have integer label 0")
        corners = np.asarray(corners_value)
        if corners.shape != (8, 3) or not np.isfinite(corners).all():
            raise ValueError(f"native row {index} has invalid corners")
        if isinstance(score_value, (bool, np.bool_)) or not isinstance(score_value, Real):
            raise ValueError(f"native row {index} has invalid score")
        score = float(score_value)
        if not math.isfinite(score) or not 0.0 < score <= 1.0:
            raise ValueError(f"native row {index} score must be in (0,1]")
    return rows, raw


def _variant_contract(variant: str) -> Tuple[str, str, str]:
    if variant == "s0":
        return SUMMARY_PREFIX, SUMMARY_SCHEMA, MANIFEST_SCHEMA
    if variant == "r1":
        return R1_SUMMARY_PREFIX, R1_SUMMARY_SCHEMA, R1_MANIFEST_SCHEMA
    raise ValueError("variant must be 's0' or 'r1'")


def _strict_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _strict_id_list(value: object, name: str) -> Tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    result = tuple(
        _strict_nonnegative_int(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique ids")
    return result


def _require_frozen_r1(summary: Mapping[str, object]) -> None:
    frozen = {
        "closed": True,
        "descriptor_dim": 256,
        "descriptor_cosine": 0.80,
        "translation_gap_m": 0.80,
        "rotation_gap_deg": 30.0,
        "depth_alpha": 0.05,
        "frame_visibility": 0.30,
        "box_visibility": 0.90,
        "min_component_nodes": 3,
        "min_component_edges": 2,
        "max_nodes_per_track": 5,
        "projection_budget_points": 8192,
        "max_receipts": 1024,
        "history_depth_frames_retained": 0,
    }
    for key, expected in frozen.items():
        actual = summary.get(key)
        if isinstance(expected, bool):
            valid = type(actual) is bool and actual is expected
        elif isinstance(expected, int):
            valid = (
                not isinstance(actual, (bool, np.bool_))
                and isinstance(actual, Integral)
                and int(actual) == expected
            )
        else:
            valid = (
                not isinstance(actual, (bool, np.bool_))
                and isinstance(actual, Real)
                and math.isfinite(float(actual))
                and float(actual) == expected
            )
        if not valid:
            raise ValueError(f"unsafe or drifted R1 field {key}")


def _bounded_metric(
    value: object, name: str, *, minimum: float = 0.0, maximum: float = 1.0
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} is outside its valid range")
    return result


def _validate_r1_receipts(value: object) -> Tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError("receipts must be a list")
    receipt_ids = []
    for receipt_index, receipt in enumerate(value):
        where = f"receipts[{receipt_index}]"
        if not isinstance(receipt, Mapping):
            raise ValueError(f"{where} must be an object")
        track_id = _strict_nonnegative_int(receipt.get("track_id"), f"{where}.track_id")
        confirmation = _strict_nonnegative_int(
            receipt.get("confirmation_frame_id"),
            f"{where}.confirmation_frame_id",
        )
        frames = _strict_id_list(
            receipt.get("component_frame_ids"), f"{where}.component_frame_ids"
        )
        if len(frames) < 3 or frames != tuple(sorted(frames)):
            raise ValueError(f"{where} must contain at least three sorted frames")
        if confirmation != max(frames):
            raise ValueError(f"{where} confirmation must equal its latest frame")
        edges = receipt.get("supporting_edges")
        if not isinstance(edges, list):
            raise ValueError(f"{where}.supporting_edges must be a list")
        edge_count = _strict_nonnegative_int(
            receipt.get("supporting_edge_count"),
            f"{where}.supporting_edge_count",
        )
        if edge_count != len(edges) or edge_count < 2:
            raise ValueError(f"{where} has an invalid supporting-edge count")
        adjacency = {frame: set() for frame in frames}
        for edge_index, edge in enumerate(edges):
            edge_where = f"{where}.supporting_edges[{edge_index}]"
            if not isinstance(edge, Mapping) or edge.get("supporting") is not True:
                raise ValueError(f"{edge_where} must be a supporting edge")
            earlier = _strict_nonnegative_int(
                edge.get("earlier_frame_id"), f"{edge_where}.earlier_frame_id"
            )
            later = _strict_nonnegative_int(
                edge.get("later_frame_id"), f"{edge_where}.later_frame_id"
            )
            if earlier >= later or earlier not in adjacency or later not in adjacency:
                raise ValueError(f"{edge_where} has invalid causal endpoints")
            cosine = _bounded_metric(
                edge.get("descriptor_cosine"), f"{edge_where}.descriptor_cosine",
                minimum=-1.0,
            )
            translation = _bounded_metric(
                edge.get("translation_m"), f"{edge_where}.translation_m",
                maximum=float("inf"),
            )
            rotation = _bounded_metric(
                edge.get("rotation_deg"), f"{edge_where}.rotation_deg",
                maximum=180.0,
            )
            visibility = _bounded_metric(
                edge.get("frame_visibility"), f"{edge_where}.frame_visibility"
            )
            box_visibility = _bounded_metric(
                edge.get("box_visibility"), f"{edge_where}.box_visibility"
            )
            _bounded_metric(
                edge.get("depth_consistency"), f"{edge_where}.depth_consistency"
            )
            _bounded_metric(
                edge.get("box_depth_consistency"),
                f"{edge_where}.box_depth_consistency",
            )
            _bounded_metric(edge.get("affinity"), f"{edge_where}.affinity")
            _bounded_metric(
                edge.get("ray_angle_deg"), f"{edge_where}.ray_angle_deg",
                maximum=180.0,
            )
            if not (
                cosine >= 0.80
                and (translation > 0.80 or rotation > 30.0)
                and visibility > 0.30
                and box_visibility > 0.90
            ):
                raise ValueError(f"{edge_where} violates the frozen R1 gate")
            adjacency[earlier].add(later)
            adjacency[later].add(earlier)
        visited = set()
        stack = [frames[0]]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(adjacency[current] - visited)
        if visited != set(frames):
            raise ValueError(f"{where} supporting graph is disconnected")
        receipt_ids.append(track_id)
    if len(set(receipt_ids)) != len(receipt_ids):
        raise ValueError("receipt track ids must be unique")
    return tuple(receipt_ids)


def _read_summary(
    log_path: Path, *, variant: str = "s0"
) -> Mapping[str, object]:
    summary_prefix, summary_schema, _ = _variant_contract(variant)
    lines = [
        row[len(summary_prefix) :]
        for row in log_path.read_text(encoding="utf-8", errors="strict").splitlines()
        if row.startswith(summary_prefix)
    ]
    if len(lines) != 1:
        raise ValueError(f"expected exactly one residual summary in {log_path}")
    summary = _strict_json(lines[0])
    required = {
        "schema": summary_schema,
        "enabled": True,
        "observer_only": True,
        "active_authorized": False,
        "native_mutation_applied": False,
        "native_export_appended": False,
        "audit_complete": True,
        "training_free": True,
        "online_learning": False,
        "gt_access": False,
        "clip_access": False,
    }
    if variant == "r1":
        required.update(
            {
                "cutr_descriptor_access": True,
                "descriptor_is_clip": False,
            }
        )
        _require_frozen_r1(summary)
    for key, expected in required.items():
        if summary.get(key) != expected or type(summary.get(key)) is not type(expected):
            raise ValueError(f"unsafe or missing summary field {key}")
    close = summary.get("close_result")
    if not isinstance(close, Mapping):
        raise ValueError("summary.close_result must be an object")
    for key, expected in {
        "observer_only": True,
        "active_authorized": False,
        "native_mutation_applied": False,
        "audit_complete": True,
    }.items():
        if close.get(key) != expected or type(close.get(key)) is not type(expected):
            raise ValueError(f"unsafe close_result field {key}")
    candidates = close.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("close_result.candidates must be a list")
    count = summary.get("counterfactual_candidate_count")
    if isinstance(count, bool) or not isinstance(count, Integral) or int(count) != len(candidates):
        raise ValueError("counterfactual candidate count mismatch")
    if len(candidates) > MAX_CANDIDATES:
        raise ValueError("candidate count exceeds frozen cap")
    if variant == "r1":
        candidate_ids = tuple(
            _strict_nonnegative_int(
                candidate.get("track_id") if isinstance(candidate, Mapping) else None,
                f"close_result.candidates[{index}].track_id",
            )
            for index, candidate in enumerate(candidates)
        )
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("R1 candidate track ids must be unique")
        top_ids = _strict_id_list(
            summary.get("counterfactual_candidate_track_ids"),
            "counterfactual_candidate_track_ids",
        )
        admitted_ids = _strict_id_list(
            close.get("admitted_track_ids"), "close_result.admitted_track_ids"
        )
        rejected_ids = _strict_id_list(
            close.get("rejected_track_ids"), "close_result.rejected_track_ids"
        )
        base_count = _strict_nonnegative_int(
            summary.get("base_counterfactual_candidate_count"),
            "base_counterfactual_candidate_count",
        )
        if candidate_ids != top_ids or candidate_ids != admitted_ids:
            raise ValueError("R1 candidate/admitted/top-level ids disagree")
        if set(admitted_ids) & set(rejected_ids):
            raise ValueError("R1 admitted and rejected ids must be disjoint")
        if base_count != len(admitted_ids) + len(rejected_ids):
            raise ValueError("R1 base candidate count is inconsistent")

        receipts = summary.get("receipts")
        receipt_count = _strict_nonnegative_int(
            summary.get("receipt_count"), "receipt_count"
        )
        if not isinstance(receipts, list) or receipt_count != len(receipts):
            raise ValueError("R1 receipt count is inconsistent")
        receipt_ids = _validate_r1_receipts(receipts)
        if not set(admitted_ids).issubset(receipt_ids):
            raise ValueError("R1 admitted candidate lacks a receipt")
        if set(rejected_ids) & set(receipt_ids):
            raise ValueError("R1 rejected candidate unexpectedly has a receipt")

        s0 = _read_summary(log_path, variant="s0")
        s0_close = s0.get("close_result")
        assert isinstance(s0_close, Mapping)
        s0_candidates = s0_close.get("candidates")
        assert isinstance(s0_candidates, list)
        s0_by_id = {
            _strict_nonnegative_int(
                row.get("track_id") if isinstance(row, Mapping) else None,
                f"S0 candidate[{index}].track_id",
            ): row
            for index, row in enumerate(s0_candidates)
        }
        if len(s0_by_id) != len(s0_candidates):
            raise ValueError("S0 candidate track ids must be unique")
        if set(admitted_ids) | set(rejected_ids) != set(s0_by_id):
            raise ValueError("R1 admitted/rejected ids do not partition S0 candidates")
        for candidate_id, candidate in zip(candidate_ids, candidates):
            if candidate != s0_by_id[candidate_id]:
                raise ValueError("R1 candidate is not an exact S0 candidate subset")
    return summary


def _bounds(boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = boxes.min(axis=1)
    upper = boxes.max(axis=1)
    volume = np.prod(upper - lower, axis=1)
    return lower, upper, volume


def _iou_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if len(left) == 0 or len(right) == 0:
        return np.zeros((len(left), len(right)), dtype=np.float64)
    left_min, left_max, left_volume = _bounds(left)
    right_min, right_max, right_volume = _bounds(right)
    extent = np.maximum(
        np.minimum(left_max[:, None], right_max[None])
        - np.maximum(left_min[:, None], right_min[None]),
        0.0,
    )
    intersection = np.prod(extent, axis=2)
    union = left_volume[:, None] + right_volume[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


def _candidate_rows(
    summary: Mapping[str, object], native_rows: Sequence[Tuple[object, object, object]]
) -> List[Tuple[int, np.ndarray, float]]:
    close = summary["close_result"]
    assert isinstance(close, Mapping)
    raw_candidates = close["candidates"]
    assert isinstance(raw_candidates, list)
    native_boxes = np.asarray(
        [np.asarray(row[1], dtype=np.float64) for row in native_rows],
        dtype=np.float64,
    ).reshape((-1, 8, 3))
    native_scores = [float(row[2]) for row in native_rows]
    candidate_boxes: List[np.ndarray] = []
    candidate_scores: List[float] = []
    track_ids = set()
    declared_native_ious: List[float] = []
    for index, row in enumerate(raw_candidates):
        if not isinstance(row, Mapping):
            raise ValueError(f"candidate {index} must be an object")
        track_id = row.get("track_id")
        if isinstance(track_id, bool) or not isinstance(track_id, Integral) or int(track_id) < 0:
            raise ValueError(f"candidate {index} has invalid track_id")
        if int(track_id) in track_ids:
            raise ValueError("candidate track ids must be unique")
        track_ids.add(int(track_id))
        corners = np.asarray(row.get("corners"), dtype=np.float64)
        if corners.shape != (8, 3) or not np.isfinite(corners).all():
            raise ValueError(f"candidate {index} has invalid corners")
        score_value = row.get("appended_score")
        if isinstance(score_value, bool) or not isinstance(score_value, Real):
            raise ValueError(f"candidate {index} has invalid appended_score")
        score = float(score_value)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"candidate {index} has invalid appended_score")
        if native_scores and not score < min(native_scores):
            raise ValueError("candidate score must be strictly below every native score")
        declared = row.get("max_native_iou")
        if isinstance(declared, bool) or not isinstance(declared, Real):
            raise ValueError(f"candidate {index} has invalid max_native_iou")
        declared_float = float(declared)
        if not math.isfinite(declared_float) or not 0.0 <= declared_float < NOVELTY_IOU:
            raise ValueError(f"candidate {index} violates declared native novelty")
        candidate_boxes.append(np.array(corners, dtype=np.float64, order="C", copy=True))
        candidate_scores.append(score)
        declared_native_ious.append(declared_float)

    boxes = np.asarray(candidate_boxes, dtype=np.float64).reshape((-1, 8, 3))
    against_native = _iou_matrix(boxes, native_boxes)
    for index, declared in enumerate(declared_native_ious):
        recomputed = float(against_native[index].max()) if len(native_boxes) else 0.0
        if recomputed >= NOVELTY_IOU or not math.isclose(
            recomputed, declared, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(f"candidate {index} native novelty mismatch")
    self_iou = _iou_matrix(boxes, boxes)
    for left in range(len(boxes)):
        for right in range(left):
            if self_iou[left, right] >= SELF_NMS_IOU:
                raise ValueError("candidate pair violates frozen self NMS")
    return [
        (0, box, score)
        for box, score in zip(candidate_boxes, candidate_scores)
    ]


def materialize(
    *,
    scene_list: Path,
    native_root: Path,
    log_root: Path,
    output_root: Path,
    manifest_path: Path,
    variant: str = "s0",
) -> Mapping[str, object]:
    _, _, manifest_schema = _variant_contract(variant)
    scenes = _read_scene_list(scene_list)
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    if manifest_path.exists():
        raise FileExistsError(f"manifest already exists: {manifest_path}")

    prepared = []
    total_native = 0
    total_appended = 0
    for scene in scenes:
        native_path = native_root / f"{scene}_boxes.pkl"
        log_path = log_root / f"{scene}.log"
        native_rows, native_bytes = _load_native(native_path)
        summary = _read_summary(log_path, variant=variant)
        appended = _candidate_rows(summary, native_rows)
        output_rows = list(native_rows) + appended
        payload = pickle.dumps([output_rows], protocol=pickle.HIGHEST_PROTOCOL)
        # Validate the exact native prefix before any filesystem mutation.
        roundtrip = pickle.loads(payload)[0]
        if len(roundtrip) != len(output_rows):
            raise RuntimeError("materialized prediction row-count mismatch")
        for before, after in zip(native_rows, roundtrip[: len(native_rows)]):
            if type(before[0]) is not type(after[0]) or before[0] != after[0]:
                raise RuntimeError("native label prefix changed")
            if type(before[2]) is not type(after[2]) or before[2] != after[2]:
                raise RuntimeError("native score prefix changed")
            if type(before[1]) is not type(after[1]) or not np.array_equal(before[1], after[1]):
                raise RuntimeError("native geometry prefix changed")
        prepared.append((scene, payload, native_bytes, len(native_rows), len(appended)))
        total_native += len(native_rows)
        total_appended += len(appended)

    output_root.mkdir(parents=True, exist_ok=False)
    scene_entries = []
    for scene, payload, native_bytes, native_count, appended_count in prepared:
        output_path = output_root / f"{scene}_boxes.pkl"
        with output_path.open("xb") as stream:
            stream.write(payload)
        scene_entries.append(
            {
                "scene_id": scene,
                "native_prediction_sha256": _sha256(native_bytes),
                "materialized_prediction_sha256": _sha256(payload),
                "native_rows": native_count,
                "appended_rows": appended_count,
                "output_rows": native_count + appended_count,
                "native_prefix_exact": True,
            }
        )
    manifest = {
        "schema": manifest_schema,
        "variant": variant,
        "scenes": len(scenes),
        "native_rows": total_native,
        "appended_rows": total_appended,
        "output_rows": total_native + total_appended,
        "native_prefix_exact": True,
        "create_only": True,
        "gt_access": False,
        "clip_access": False,
        "training": False,
        "active_runtime_authorized": False,
        "scene_entries": scene_entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--variant", choices=("s0", "r1"), default="s0")
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = materialize(
        scene_list=args.scene_list,
        native_root=args.native_root,
        log_root=args.log_root,
        output_root=args.output_root,
        manifest_path=args.manifest,
        variant=args.variant,
    )
    print(json.dumps(manifest, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
