#!/usr/bin/env python3
"""One-shot F6 multi-view selected-geometry paper100 capacity evaluation.

The protected evaluation inputs are opened only after the complete, pinned
F6 no-GT receipt and every F4/F6 scene/source ledger have been authenticated.
F6 is shadow-only; the GT-selected suffix reported here is an oracle capacity
test and is never an actual deployable F6 result.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

from tools import audit_scannet_fastsam_f5_selector_paper100 as shared  # noqa: E402
from tools.audit_scannet_boxer_unexplained_oracle import (  # noqa: E402
    aligned_iou_matrix,
    load_axis_alignment,
    load_baseline_boxes,
    load_gt_minmax,
    load_scene_list,
)


SCHEMA = "boxfusion.scannet_fastsam_f6_mvdc_paper100_evaluation.v1"
EVALUATION_PROTOCOL_ID = "F6-GT-FREE-SELECTOR-ONE-SHOT-EVALUATION-PAPER100"
EVALUATION_PROTOCOL_SHA256 = "390b7b704b200b22ccc7e604d6b6992b0a9a99ba7c35e106c2465928b1d03e2a"
F6_PROTOCOL_ID = "F6-GT-FREE-PAST-ONLY-MULTIVIEW-DEPTH-PROJECTION-SELECTOR-PAPER100"
F6_PROTOCOL_SHA256 = "d0592d8ea69c2d8bcddd942f6ab57b077cdb899aafaadcd3d1c83462cd79768f"
F4_PROTOCOL_ID = shared.F4_PROTOCOL_ID
F6_MERGE_SCHEMA = "boxfusion.scannet_fastsam_f6_mvdc_paper100.merge.v1"
F6_SCENE_SCHEMA = "boxfusion.scannet_fastsam_f6_mvdc_paper100.scene.v1"
F6_SOURCE_SCHEMA = "boxfusion.fastsam_f6_mvdc_selector.v1"
F4_MERGE_SCHEMA = shared.F4_MERGE_SCHEMA
F4_SCENE_SCHEMA = shared.F4_SCENE_SCHEMA
F4_REPORT_SCHEMA = shared.F4_REPORT_SCHEMA

EXPECTED = dict(shared.EXPECTED)
THRESHOLDS = shared.THRESHOLDS
EXPECTED_BASELINE_AP_POINTS = dict(shared.EXPECTED_BASELINE_AP_POINTS)
EXPECTED_SCENE_LIST_SHA256 = shared.EXPECTED_SCENE_LIST_SHA256
EXPECTED_F4_RECEIPT_SHA256 = shared.EXPECTED_F4_RECEIPT_SHA256
EXPECTED_F6_RECEIPT_SHA256 = "1a9f701214fe2ee9de3ea3b3a106064dc5670de5110c5c5056d622949f863727"
EXPECTED_F4_REPORT_SHA256 = shared.EXPECTED_F4_REPORT_SHA256
EXPECTED_OFFICIAL_EVALUATOR_SHA256 = shared.EXPECTED_OFFICIAL_EVALUATOR_SHA256
EXPECTED_OFFICIAL_EVAL_DET_SHA256 = shared.EXPECTED_OFFICIAL_EVAL_DET_SHA256
EXPECTED_OFFICIAL_METRIC_UTIL_SHA256 = shared.EXPECTED_OFFICIAL_METRIC_UTIL_SHA256
EXPECTED_OFFICIAL_BOX_UTIL_SHA256 = shared.EXPECTED_OFFICIAL_BOX_UTIL_SHA256
EXPECTED_ORACLE_HELPER_SHA256 = shared.EXPECTED_ORACLE_HELPER_SHA256
EXPECTED_SHARED_EVALUATOR_SHA256 = "e7c7284c9d1751d33c807b84f23711c1853ae208f7010d6f2abba9dce908bb0e"
EXPECTED_SOURCE_IDS_SHA256 = shared.EXPECTED_SOURCE_IDS_SHA256
EXPECTED_SOURCE_LINEAGE_SHA256 = shared.EXPECTED_SOURCE_LINEAGE_SHA256
EXPECTED_RESULT_LEDGER_SHA256 = "380f906ea0a688c2f3590ffef82676ceb3d29b8ed65211f1ad9bdcf6913b61c3"
REQUIRED_ADDITIONAL_MATCHES = 144
TARGET_DELTA_AP_POINTS = 10.0
HYPOTHESES = shared.HYPOTHESES
F4_GATE_NAMES = shared.F4_GATE_NAMES
F6_GATE_NAMES = frozenset(
    {
        "integrity_complete",
        "exact_keyframes",
        "exact_successful_frames",
        "exact_unique_sources",
        "switch_min_sources",
        "switch_min_scenes",
        "switch_max_fraction",
        "maximum_buffered_frames",
        "maximum_sources_per_buffered_frame",
        "maximum_state_raw_array_payload_bytes",
        "forbidden_or_mutation_count",
        "f6_incremental_warm_p95_ms",
        "replay_composed_warm_p95_ms",
        "replay_composed_warm_max_ms",
        "replay_composed_warm_mean_per_source_frame_ms",
        "gap25_warm_deadline_miss_count",
        "cuda_peak_memory_bytes",
        "f6_cuda_allocated_bytes",
    }
)
_SIGNS = shared._SIGNS

DEFAULT_SCENE_LIST = shared.DEFAULT_SCENE_LIST
DEFAULT_EVALUATION_PROTOCOL = REPOSITORY_ROOT / "docs/F6_GT_FREE_SELECTOR_EVALUATION_PROTOCOL_FREEZE.md"
DEFAULT_F4_RECEIPT = shared.DEFAULT_F4_RECEIPT
DEFAULT_F4_SIDECARS = shared.DEFAULT_F4_SIDECARS
DEFAULT_F6_RECEIPT = REPOSITORY_ROOT / "logs/scannet_fastsam_f6_mvdc_paper100_score05/final/F6_GT_FREE_MVDC_PAPER100.json"
DEFAULT_F6_SIDECARS = REPOSITORY_ROOT / "logs/scannet_fastsam_f6_mvdc_paper100_score05/scenes"
DEFAULT_F4_REPORT = shared.DEFAULT_F4_REPORT
DEFAULT_BASELINE_ROOT = shared.DEFAULT_BASELINE_ROOT
DEFAULT_GT_ROOT = shared.DEFAULT_GT_ROOT
DEFAULT_SCAN_ROOT = shared.DEFAULT_SCAN_ROOT
DEFAULT_OFFICIAL_EVALUATOR = shared.DEFAULT_OFFICIAL_EVALUATOR
DEFAULT_OUTPUT = REPOSITORY_ROOT / "reports/fastsam_f6_mvdc_paper100/F6_FASTSAM_MVDC_PAPER100_EVALUATION.json"
ORACLE_HELPER_SOURCE = shared.ORACLE_HELPER_SOURCE
SHARED_EVALUATOR_SOURCE = Path(shared.__file__).resolve()

# Reuse the already independently tested file/hash/official-evaluator helpers.
F6EvaluationError = shared.F5EvaluationError
_sha256 = shared._sha256
_canonical_json_sha256 = shared._canonical_json_sha256
_regular_file = shared._regular_file
_read_json = shared._read_json
_read_pinned_json = shared._read_pinned_json
_require_frozen_file = shared._require_frozen_file
_content_hash = shared._content_hash
_hash_string = shared._hash_string
_array = shared._array
_threshold_key = shared._threshold_key
_official_dependency_paths = shared._official_dependency_paths
_load_official_eval_det = shared._load_official_eval_det
_authenticated_official_constant_evaluate = shared._authenticated_official_constant_evaluate
_validate_historical_f4_input_lineage = shared._validate_historical_f4_input_lineage
_align_corners = shared._align_corners
_seal_from_row = shared._seal_from_row


@dataclass(frozen=True)
class SelectedSource:
    scene_id: str
    scene_index: int
    frame_id: int
    frame_ordinal: int
    rank: int
    source_id: str
    source_lineage_sha256: str
    result_sha256: str
    selected_hypothesis: str
    base_hypothesis: str
    switched_from_base: bool
    world_corners: np.ndarray
    base_world_corners: np.ndarray
    aligned_minmax: np.ndarray | None = None
    aligned_base_minmax: np.ndarray | None = None


def _expected_selected_geometry(
    name: str, hypothesis: Mapping[str, Any]
) -> tuple[dict[str, Any], np.ndarray]:
    """Reproduce the frozen F6 normalization of exactly one F4 hypothesis."""
    if name in ("H0", "HL", "HLG"):
        if hypothesis.get("valid") is not True:
            raise F6EvaluationError(f"selected {name} is not a valid sealed F4 hypothesis")
        lower = _array(hypothesis.get("q02"), (3,), f"{name}.q02")
        upper = _array(hypothesis.get("q98"), (3,), f"{name}.q98")
        center = _array(hypothesis.get("center"), (3,), f"{name}.center")
        extent = _array(hypothesis.get("extent"), (3,), f"{name}.extent")
        if np.any(upper <= lower) or not np.allclose(
            center, (lower + upper) * 0.5, rtol=0.0, atol=1.0e-9
        ) or not np.allclose(extent, upper - lower, rtol=0.0, atol=1.0e-9):
            raise F6EvaluationError(f"{name} center/extent differ from q02/q98")
        corners = center[None, :] + _SIGNS * (extent[None, :] * 0.5)
        geometry = {
            "kind": "world_aabb",
            "hypothesis": name,
            "q02": lower.tolist(),
            "q98": upper.tolist(),
            "center": center.tolist(),
            "extent": extent.tolist(),
            "world_rotation": np.eye(3, dtype=np.float64).tolist(),
            "world_corners": corners.tolist(),
            "envelope_q02": lower.tolist(),
            "envelope_q98": upper.tolist(),
        }
        return geometry, corners
    if name != "HB" or hypothesis.get("valid") is not True:
        raise F6EvaluationError("selected HB is not a valid sealed F4 hypothesis")
    center = _array(hypothesis.get("world_center"), (3,), "HB.world_center")
    extent = _array(hypothesis.get("local_extent"), (3,), "HB.local_extent")
    rotation = _array(hypothesis.get("world_rotation"), (3, 3), "HB.world_rotation")
    corners = _array(hypothesis.get("world_corners"), (8, 3), "HB.world_corners")
    camera_depth = hypothesis.get("camera_depth")
    if (
        type(camera_depth) not in (int, float)
        or isinstance(camera_depth, bool)
        or not math.isfinite(float(camera_depth))
        or float(camera_depth) <= 1.0e-4
        or np.any(extent <= 0.0)
        or float(np.linalg.det(rotation)) <= 0.0
        or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1.0e-3)
    ):
        raise F6EvaluationError("selected HB violates the frozen OBB geometry")
    expected_corners = center[None, :] + (_SIGNS * (extent[None, :] * 0.5)) @ rotation.T
    if not np.allclose(corners, expected_corners, rtol=0.0, atol=2.0e-6):
        raise F6EvaluationError("selected HB corners differ from center/extent/rotation")
    lower = corners.min(axis=0)
    upper = corners.max(axis=0)
    geometry = {
        "kind": "world_obb",
        "hypothesis": "HB",
        "world_center": center.tolist(),
        "local_extent": extent.tolist(),
        "world_rotation": rotation.tolist(),
        "world_corners": corners.tolist(),
        "envelope_q02": lower.tolist(),
        "envelope_q98": upper.tolist(),
    }
    return geometry, corners


def validate_selected_source(
    *,
    scene: str,
    scene_index: int,
    frame_id: int,
    frame_ordinal: int,
    rank: int,
    f4_source: Mapping[str, Any],
    f6_source: Mapping[str, Any],
) -> SelectedSource:
    source_id = f4_source.get("source_id")
    lineage = _hash_string(
        f4_source.get("source_lineage_sha256"), f"{scene}/{frame_id}/{rank} source lineage"
    )
    if (
        not isinstance(source_id, str)
        or f6_source.get("schema") != F6_SOURCE_SCHEMA
        or f6_source.get("protocol_id") != F6_PROTOCOL_ID
        or f6_source.get("mode") != "shadow"
        or f6_source.get("source_id") != source_id
        or f4_source.get("scene_index") != scene_index
        or f4_source.get("frame_id") != frame_id
        or f4_source.get("frame_ordinal") != frame_ordinal
        or f4_source.get("rank") != rank
        or f4_source.get("candidate_index") != rank
        or f6_source.get("frame_id") != frame_id
        or f6_source.get("frame_ordinal") != frame_ordinal
        or f6_source.get("rank") != rank
        or f6_source.get("source_lineage_sha256") != lineage
        or f6_source.get("observer_only") is not True
        or f6_source.get("birth_applied") is not False
        or f6_source.get("native_output_mutation_applied") is not False
        or f6_source.get("maximum_lookahead_frames") != 0
    ):
        raise F6EvaluationError(f"F4/F6 source identity/shadow contract differs: {scene}/{frame_id}/{rank}")
    _hash_string(f6_source.get("input_evidence_sha256"), f"{source_id} input evidence hash")
    hypotheses = f4_source.get("hypotheses")
    if not isinstance(hypotheses, Mapping) or set(hypotheses) != set(HYPOTHESES):
        raise F6EvaluationError(f"F4 hypothesis ledger differs: {source_id}")
    expected_hashes = {name: _canonical_json_sha256(hypotheses[name]) for name in HYPOTHESES}
    if f6_source.get("input_hypothesis_sha256") != expected_hashes:
        raise F6EvaluationError(f"F6 hypothesis hash ledger differs: {source_id}")
    selected = f6_source.get("selected_hypothesis")
    base = f6_source.get("base_hypothesis")
    if selected not in HYPOTHESES or base not in ("H0", "HL", "HLG"):
        raise F6EvaluationError(f"F6 selected/base hypothesis differs: {source_id}")
    selected_geometry, corners = _expected_selected_geometry(str(selected), hypotheses[selected])
    base_geometry, base_corners = _expected_selected_geometry(str(base), hypotheses[base])
    if f6_source.get("selected_geometry") != selected_geometry:
        raise F6EvaluationError(f"F6 selected geometry is not an exact F4 copy: {source_id}")
    if f6_source.get("base_geometry") != base_geometry:
        raise F6EvaluationError(f"F6 base geometry is not an exact F4 copy: {source_id}")
    selected_sha = _canonical_json_sha256(selected_geometry)
    base_sha = _canonical_json_sha256(base_geometry)
    if (
        f6_source.get("selected_geometry_sha256") != selected_sha
        or f6_source.get("base_geometry_sha256") != base_sha
    ):
        raise F6EvaluationError(f"F6 selected/base geometry hash differs: {source_id}")
    switched = selected != base
    if type(f6_source.get("switched_from_base")) is not bool or f6_source.get("switched_from_base") is not switched:
        raise F6EvaluationError(f"F6 base/switch ledger differs: {source_id}")
    matched_count = f6_source.get("matched_past_frame_count")
    if type(matched_count) is not int or matched_count < 0 or matched_count > 2:
        raise F6EvaluationError(f"F6 matched-past count differs: {source_id}")
    if switched and (matched_count != 2 or f6_source.get("selection_reason") != "non_base_candidate_won"):
        raise F6EvaluationError(f"F6 switched source lacks complete three-view proof: {source_id}")
    if type(f6_source.get("formal_score")) is not float or f6_source.get("formal_score") != 1.0:
        raise F6EvaluationError(f"F6 formal score differs from 1.0: {source_id}")
    payload = dict(f6_source)
    result_sha = _hash_string(payload.pop("result_sha256", None), f"{source_id} result hash")
    if result_sha != _canonical_json_sha256(payload):
        raise F6EvaluationError(f"F6 result row hash differs: {source_id}")
    return SelectedSource(
        scene_id=scene,
        scene_index=scene_index,
        frame_id=frame_id,
        frame_ordinal=frame_ordinal,
        rank=rank,
        source_id=source_id,
        source_lineage_sha256=lineage,
        result_sha256=result_sha,
        selected_hypothesis=str(selected),
        base_hypothesis=str(base),
        switched_from_base=switched,
        world_corners=corners,
        base_world_corners=base_corners,
    )


def _validate_f6_merge(
    receipt: Mapping[str, Any], *, scenes: Sequence[str]
) -> dict[str, Mapping[str, Any]]:
    zero_fields = (
        "birth_count",
        "native_output_mutation_count",
        "source_addition_or_removal_count",
        "score_rank_semantic_mutation_count",
        "forbidden_access_count",
        "training_or_online_learning_count",
    )
    if (
        receipt.get("schema") != F6_MERGE_SCHEMA
        or receipt.get("protocol_id") != F6_PROTOCOL_ID
        or receipt.get("protocol_sha256") != F6_PROTOCOL_SHA256
        or receipt.get("complete") is not True
        or receipt.get("overall_pass") is not True
        or receipt.get("decision") != "retain_f6_for_one_separately_sealed_evaluation_only"
        or receipt.get("content_sha256") != _content_hash(receipt)
        or any(receipt.get(name) != 0 for name in zero_fields)
    ):
        raise F6EvaluationError("F6 merge integrity/pass/no-mutation contract differs")
    coverage = receipt.get("coverage")
    totals = receipt.get("totals")
    if (
        not isinstance(coverage, Mapping)
        or coverage.get("scene_count") != EXPECTED["scene_count"]
        or coverage.get("scene_order") != list(scenes)
        or coverage.get("exact_source_partition") is not True
        or coverage.get("exact_source_order") is not True
        or coverage.get("source_ids_sha256") != EXPECTED_SOURCE_IDS_SHA256
        or coverage.get("source_lineage_sha256") != EXPECTED_SOURCE_LINEAGE_SHA256
        or coverage.get("result_ledger_sha256") != EXPECTED_RESULT_LEDGER_SHA256
        or not isinstance(totals, Mapping)
        or totals.get("keyframe_count") != EXPECTED["keyframe_count"]
        or totals.get("successful_frame_count") != EXPECTED["successful_frame_count"]
        or totals.get("source_count") != EXPECTED["source_count"]
        or totals.get("identity_verified_source_count") != EXPECTED["source_count"]
    ):
        raise F6EvaluationError("F6 paper100 census/order/lineage differs")
    gates = receipt.get("gates")
    if (
        not isinstance(gates, Mapping)
        or set(gates) != F6_GATE_NAMES
        or any(
            not isinstance(row, Mapping)
            or row.get("pass") is not True
            or row.get("passed") is not True
            for row in gates.values()
        )
    ):
        raise F6EvaluationError("F6 frozen gate ledger differs")
    runtime = receipt.get("runtime")
    bounded = receipt.get("bounded_state")
    authorization = receipt.get("evaluation_authorization")
    contracts = receipt.get("contracts")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("overall_pass") is not True
        or not isinstance(bounded, Mapping)
        or bounded.get("overall_pass") is not True
        or not isinstance(authorization, Mapping)
        or authorization.get("allowed") is not True
        or authorization.get("birth_authorized") is not False
        or authorization.get("deployment_authorized") is not False
        or not isinstance(contracts, Mapping)
        or contracts.get("shadow_only") is not True
        or contracts.get("selector_only") is not True
        or any(contracts.get(name) is not False for name in (
            "ground_truth_access", "annotation_access", "evaluator_access",
            "prediction_access", "future_frame_access", "native_output_mutation",
            "source_addition_or_removal", "score_or_rank_mutation",
            "semantic_or_clip_access", "birth_enabled", "training", "online_learning",
        ))
    ):
        raise F6EvaluationError("F6 no-GT runtime/authorization/input boundary differs")
    selection = receipt.get("selection")
    selected_counts = [
        selection.get(f"selected_{name.lower()}_count") if isinstance(selection, Mapping) else None
        for name in HYPOTHESES
    ]
    if (
        not isinstance(selection, Mapping)
        or type(selection.get("formal_score")) is not float
        or selection.get("formal_score") != 1.0
        or any(type(value) is not int or value < 0 for value in selected_counts)
        or sum(selected_counts) != EXPECTED["source_count"]
        or selection.get("switch_count") != totals.get("switch_count")
        or selection.get("fallback_count") != totals.get("fallback_count")
        or selection.get("complete_three_view_switch_proof_count") != selection.get("switch_count")
    ):
        raise F6EvaluationError("F6 selection/switch census differs")
    f4_input = receipt.get("inputs", {}).get("f4_receipt")
    if not isinstance(f4_input, Mapping) or f4_input.get("sha256") != EXPECTED_F4_RECEIPT_SHA256:
        raise F6EvaluationError("F6/F4 receipt lineage differs")
    rows = receipt.get("scenes")
    if not isinstance(rows, list) or len(rows) != len(scenes):
        raise F6EvaluationError("F6 scene ledger differs")
    result: dict[str, Mapping[str, Any]] = {}
    for scene_index, (scene, row) in enumerate(zip(scenes, rows, strict=True)):
        if (
            not isinstance(row, Mapping)
            or row.get("scene_id") != scene
            or row.get("scene_index") != scene_index
            or row.get("causality", {}).get("overall_pass") is not True
            or row.get("causality", {}).get("maximum_lookahead_frames") != 0
            or row.get("determinism", {}).get("passed") is not True
            or row.get("prefix_replay", {}).get("passed") is not True
            or row.get("bounded_state", {}).get("overall_pass") is not True
        ):
            raise F6EvaluationError(f"F6 scene no-GT proof/order differs: {scene}")
        _seal_from_row(row, scene)
        result[scene] = row
    return result


def _load_scene_sources_pre_gt(
    *,
    scene: str,
    scene_index: int,
    f4_path: Path,
    f4_sha: str,
    f6_path: Path,
    f6_sha: str,
) -> tuple[list[SelectedSource], int, int, dict[str, Any]]:
    if _sha256(_regular_file(f4_path, f"{scene} F4 sidecar", ".json")) != f4_sha:
        raise F6EvaluationError(f"F4 sidecar rehash differs: {scene}")
    if _sha256(_regular_file(f6_path, f"{scene} F6 sidecar", ".json")) != f6_sha:
        raise F6EvaluationError(f"F6 sidecar rehash differs: {scene}")
    f4 = _read_json(f4_path, f"{scene} F4 sidecar")
    f6 = _read_json(f6_path, f"{scene} F6 sidecar")
    if _sha256(f4_path) != f4_sha or _sha256(f6_path) != f6_sha:
        raise F6EvaluationError(f"F4/F6 sidecar changed while read: {scene}")
    for value, schema, protocol, label in (
        (f4, F4_SCENE_SCHEMA, F4_PROTOCOL_ID, "F4"),
        (f6, F6_SCENE_SCHEMA, F6_PROTOCOL_ID, "F6"),
    ):
        if (
            value.get("schema") != schema
            or value.get("protocol_id") != protocol
            or value.get("complete") is not True
            or value.get("scene_id") != scene
            or value.get("scene_index") != scene_index
            or value.get("content_sha256") != _content_hash(value)
        ):
            raise F6EvaluationError(f"{label} scene contract differs: {scene}")
    if (
        f6.get("native_output_mutation_count") != 0
        or f6.get("birth_count") != 0
        or f6.get("causality", {}).get("overall_pass") is not True
        or f6.get("causality", {}).get("maximum_lookahead_frames") != 0
        or f6.get("prefix_replay", {}).get("passed") is not True
        or f6.get("determinism", {}).get("passed") is not True
        or f6.get("bounded_state", {}).get("overall_pass") is not True
    ):
        raise F6EvaluationError(f"F6 scene no-GT proof differs: {scene}")
    f6_f4_seal = f6.get("inputs", {}).get("f4_sidecar")
    if (
        not isinstance(f6_f4_seal, Mapping)
        or f6_f4_seal.get("path") != os.fspath(f4_path.resolve())
        or f6_f4_seal.get("sha256") != f4_sha
    ):
        raise F6EvaluationError(f"F6/F4 scene input seal differs: {scene}")
    f4_frames = f4.get("frames")
    f6_frames = f6.get("frames")
    if not isinstance(f4_frames, list) or not isinstance(f6_frames, list) or len(f4_frames) != len(f6_frames):
        raise F6EvaluationError(f"F4/F6 frame ledger differs: {scene}")
    selected: list[SelectedSource] = []
    successful = 0
    result_hashes: list[str] = []
    lineage_hashes: list[str] = []
    selected_counts = {name: 0 for name in HYPOTHESES}
    switch_count = 0
    evaluated_count = 0
    for ordinal, (f4_frame, f6_frame) in enumerate(zip(f4_frames, f6_frames, strict=True)):
        if not isinstance(f4_frame, Mapping) or not isinstance(f6_frame, Mapping):
            raise F6EvaluationError(f"invalid F4/F6 frame row: {scene}/{ordinal}")
        frame_id = f4_frame.get("frame_id")
        if (
            f4_frame.get("frame_ordinal") != ordinal
            or f6_frame.get("frame_ordinal") != ordinal
            or f6_frame.get("frame_id") != frame_id
            or f6_frame.get("successful") is not f4_frame.get("successful")
        ):
            raise F6EvaluationError(f"F4/F6 frame identity differs: {scene}/{ordinal}")
        f4_sources = f4_frame.get("sources")
        f6_sources = f6_frame.get("sources")
        if not isinstance(f4_sources, list) or not isinstance(f6_sources, list) or len(f4_sources) != len(f6_sources):
            raise F6EvaluationError(f"F4/F6 source count differs: {scene}/{frame_id}")
        if f6_frame.get("successful") is True:
            successful += 1
        elif f6_sources:
            raise F6EvaluationError(f"failed F6 frame retains sources: {scene}/{frame_id}")
        for rank, (f4_source, f6_source) in enumerate(zip(f4_sources, f6_sources, strict=True)):
            if not isinstance(f4_source, Mapping) or not isinstance(f6_source, Mapping):
                raise F6EvaluationError(f"invalid F4/F6 source row: {scene}/{frame_id}/{rank}")
            source = validate_selected_source(
                scene=scene,
                scene_index=scene_index,
                frame_id=int(frame_id),
                frame_ordinal=ordinal,
                rank=rank,
                f4_source=f4_source,
                f6_source=f6_source,
            )
            selected.append(source)
            result_hashes.append(source.result_sha256)
            lineage_hashes.append(source.source_lineage_sha256)
            selected_counts[source.selected_hypothesis] += 1
            switch_count += int(source.switched_from_base)
            evaluated_count += int(int(f6_source["matched_past_frame_count"]) >= 2)
    counts = f6.get("counts")
    f4_counts = f4.get("counts")
    source_ids = [row.source_id for row in selected]
    source_ids_sha256 = _canonical_json_sha256(source_ids)
    lineage_sha256 = _canonical_json_sha256(lineage_hashes)
    result_sha256 = _canonical_json_sha256(result_hashes)
    expected_counts = {
        "keyframe_count": len(f6_frames),
        "successful_frame_count": successful,
        "source_count": len(selected),
        "identity_verified_source_count": len(selected),
        "multiview_evaluated_source_count": evaluated_count,
        "switch_count": switch_count,
        "fallback_count": len(selected) - switch_count,
        **{f"selected_{name.lower()}_count": selected_counts[name] for name in HYPOTHESES},
    }
    if (
        counts != expected_counts
        or len(set(source_ids)) != len(source_ids)
        or f6.get("source_ids_sha256") != source_ids_sha256
        or f6.get("source_lineage_sha256") != lineage_sha256
        or f6.get("result_ledger_sha256") != result_sha256
    ):
        raise F6EvaluationError(f"F6 scene census/hash ledger differs: {scene}")
    if (
        not isinstance(f4_counts, Mapping)
        or f4_counts.get("keyframe_count") != len(f4_frames)
        or f4_counts.get("successful_frame_count") != successful
        or f4_counts.get("source_count") != len(selected)
        or f4_counts.get("valid_hb_count") != len(selected)
        or f4_counts.get("invalid_hb_count") != 0
        or f4.get("source_ids_sha256") != source_ids_sha256
        or f4.get("source_lineage_sha256") != lineage_sha256
    ):
        raise F6EvaluationError(f"F4 scene census/hash ledger differs: {scene}")
    return selected, len(f6_frames), successful, {
        "f4": {
            "counts": dict(f4_counts),
            "source_ids_sha256": source_ids_sha256,
            "source_lineage_sha256": lineage_sha256,
        },
        "f6": {
            "counts": dict(counts),
            "source_ids_sha256": source_ids_sha256,
            "source_lineage_sha256": lineage_sha256,
            "result_ledger_sha256": result_sha256,
        },
    }


def _subset_split(
    *,
    native_iou: Sequence[np.ndarray],
    selected_iou: Sequence[np.ndarray],
    selected_sources: Sequence[Sequence[SelectedSource]],
    threshold: float,
) -> dict[str, dict[str, int]]:
    result = {
        name: {"selected_source_count": 0, "maximum_matching_count": 0, "union_matching_count": 0}
        for name in ("base", "switch")
    }
    native_total = 0
    for native, selected, sources in zip(native_iou, selected_iou, selected_sources, strict=True):
        native_array = np.asarray(native, dtype=np.float64)
        selected_array = np.asarray(selected, dtype=np.float64)
        native_total += len(shared.strict_maximum_matching(native_array, threshold))
        for name, switched in (("base", False), ("switch", True)):
            indices = [index for index, source in enumerate(sources) if source.switched_from_base is switched]
            matrix = selected_array[indices] if indices else np.empty((0, selected_array.shape[1]), dtype=np.float64)
            result[name]["selected_source_count"] += len(indices)
            result[name]["maximum_matching_count"] += len(shared.strict_maximum_matching(matrix, threshold))
            result[name]["union_matching_count"] += len(
                shared.strict_maximum_matching(np.concatenate((native_array, matrix), axis=0), threshold)
            )
    for row in result.values():
        row["additional_union_matching_over_native"] = row["union_matching_count"] - native_total
    return result


def evaluate_selected_threshold(
    *,
    scenes: Sequence[str],
    native_iou: Sequence[np.ndarray],
    selected_iou: Sequence[np.ndarray],
    all_base_iou: Sequence[np.ndarray],
    selected_sources: Sequence[Sequence[SelectedSource]],
    gt_counts: Sequence[int],
    baseline_evaluation: Mapping[str, Any],
    threshold: float,
    f4_g4_additional_union_matches: int,
    official_eval_det: ModuleType,
) -> dict[str, Any]:
    report = shared.evaluate_selected_threshold(
        scenes=scenes,
        native_iou=native_iou,
        selected_iou=selected_iou,
        selected_sources=selected_sources,
        gt_counts=gt_counts,
        baseline_evaluation=baseline_evaluation,
        threshold=threshold,
        f4_g4_additional_union_matches=f4_g4_additional_union_matches,
        official_eval_det=official_eval_det,
    )
    capacity = report["f4_g4_capacity"]
    capacity["f6_retained_additional_union_matches"] = capacity.pop(
        "f5_retained_additional_union_matches"
    )
    report["selected_base_switch_split"] = _subset_split(
        native_iou=native_iou,
        selected_iou=selected_iou,
        selected_sources=selected_sources,
        threshold=threshold,
    )
    if not (
        len(scenes) == len(native_iou) == len(selected_iou) == len(all_base_iou)
    ):
        raise F6EvaluationError("F6 selected/all-base counterfactual inputs differ")
    all_base_matching = 0
    all_base_union_matching = 0
    selected_union_matching = 0
    for scene, native, selected, all_base in zip(
        scenes, native_iou, selected_iou, all_base_iou, strict=True
    ):
        native_array = np.asarray(native, dtype=np.float64)
        selected_array = np.asarray(selected, dtype=np.float64)
        base_array = np.asarray(all_base, dtype=np.float64)
        if selected_array.shape != base_array.shape or native_array.shape[1] != base_array.shape[1]:
            raise F6EvaluationError(f"F6 selected/all-base IoU shape differs: {scene}")
        all_base_matching += len(shared.strict_maximum_matching(base_array, threshold))
        all_base_union_matching += len(
            shared.strict_maximum_matching(np.concatenate((native_array, base_array), axis=0), threshold)
        )
        selected_union_matching += len(
            shared.strict_maximum_matching(np.concatenate((native_array, selected_array), axis=0), threshold)
        )
    if selected_union_matching != report["union_maximum_matching_count"]:
        raise F6EvaluationError("F6 selected-union counterfactual reproduction differs")
    report["all_base_counterfactual"] = {
        "oracle_only": True,
        "deployable": False,
        "all_base_maximum_matching_count": all_base_matching,
        "native_plus_all_base_union_matching_count": all_base_union_matching,
        "native_plus_selected_union_matching_count": selected_union_matching,
        "switch_replacement_delta_union_matching": selected_union_matching - all_base_union_matching,
        "attribution": "only_the_165_selected_vs_base_replacements_can_change_this_delta",
    }
    report["oracle_only"] = True
    report["deployable"] = False
    suffix = report["gt_selected_constructive_suffix"]
    suffix["threshold_specific_gt_selection"] = True
    suffix["shared_detection_list_across_iou_thresholds"] = False
    return report


def f6_decision(
    *, per_threshold: Mapping[str, Any], no_gt_merge_passed: bool, baseline_passed: bool
) -> dict[str, Any]:
    if type(no_gt_merge_passed) is not bool or type(baseline_passed) is not bool:
        raise F6EvaluationError("decision prerequisites must be booleans")
    rows = [per_threshold[_threshold_key(threshold)] for threshold in THRESHOLDS]
    match_pass = all(
        row["additional_union_matching_over_native"] >= REQUIRED_ADDITIONAL_MATCHES
        for row in rows
    )
    ap_pass = all(
        row["gt_selected_constructive_suffix"]["delta_ap_points"] >= TARGET_DELTA_AP_POINTS
        for row in rows
    )
    passed = no_gt_merge_passed and baseline_passed and match_pass and ap_pass
    return {
        "no_gt_f6_merge_passed": no_gt_merge_passed,
        "native_baseline_reproduction_passed": baseline_passed,
        "required_additional_union_matches_each_threshold": REQUIRED_ADDITIONAL_MATCHES,
        "target_delta_ap_points_each_threshold": TARGET_DELTA_AP_POINTS,
        "selected_geometry_capacity_passes_all_thresholds": match_pass,
        "constructive_suffix_plus10_passes_all_thresholds": ap_pass,
        "overall_pass": passed,
        "active_birth_authorized": False,
        "result": (
            "retain_f6_authorize_f7_high_precision_birth_shadow_only"
            if passed
            else "discard_f6_multiview_selector_for_plus10_route"
        ),
    }


def _selector_snapshot(
    *,
    scenes: Sequence[str],
    fixed: Mapping[str, Path],
    f4_sidecar_root: Path,
    f6_sidecar_root: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {"fixed": {}, "scenes": {}}
    for name, path in fixed.items():
        source = _regular_file(path, name)
        result["fixed"][name] = {"path": os.fspath(source), "sha256": _sha256(source)}
    for scene in scenes:
        paths = {
            "f4_sidecar": f4_sidecar_root / f"{scene}.json",
            "f6_sidecar": f6_sidecar_root / f"{scene}.json",
        }
        result["scenes"][scene] = {
            name: {
                "path": os.fspath(_regular_file(path, f"{scene} {name}")),
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        }
    return result


def _evaluation_snapshot(
    *,
    scenes: Sequence[str],
    fixed: Mapping[str, Path],
    selector_snapshot: Mapping[str, Any],
    baseline_root: Path,
    gt_root: Path,
    scan_root: Path,
) -> dict[str, Any]:
    selector_fixed = selector_snapshot.get("fixed")
    selector_scenes = selector_snapshot.get("scenes")
    if not isinstance(selector_fixed, Mapping) or not isinstance(selector_scenes, Mapping):
        raise F6EvaluationError("selector snapshot is malformed")
    result: dict[str, Any] = {
        "fixed": {name: dict(seal) for name, seal in selector_fixed.items()},
        "scenes": {},
    }
    for name, path in fixed.items():
        if name in result["fixed"]:
            raise F6EvaluationError(f"evaluation fixed input collides with selector seal: {name}")
        source = _regular_file(path, name)
        result["fixed"][name] = {"path": os.fspath(source), "sha256": _sha256(source)}
    for scene in scenes:
        selector_scene = selector_scenes.get(scene)
        if not isinstance(selector_scene, Mapping):
            raise F6EvaluationError(f"selector snapshot lacks scene: {scene}")
        paths = {
            "native": baseline_root / f"{scene}_boxes.pkl",
            "gt": gt_root / f"{scene}_bbox.npy",
            "alignment": scan_root / scene / f"{scene}.txt",
        }
        result["scenes"][scene] = {name: dict(seal) for name, seal in selector_scene.items()} | {
            name: {
                "path": os.fspath(_regular_file(path, f"{scene} {name}")),
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        }
    return result


def audit_scannet_fastsam_f6_mvdc_paper100(
    *,
    scene_list: Path,
    evaluation_protocol: Path,
    f4_receipt: Path,
    f4_sidecar_root: Path,
    f6_receipt: Path,
    f6_sidecar_root: Path,
    f4_report: Path,
    baseline_root: Path,
    gt_root: Path,
    scan_root: Path,
    official_evaluator: Path,
) -> dict[str, Any]:
    # Phase 1: no protected GT/native/alignment input is opened here.
    if _sha256(_regular_file(SHARED_EVALUATOR_SOURCE, "shared F5 evaluator source", ".py")) != EXPECTED_SHARED_EVALUATOR_SHA256:
        raise F6EvaluationError("shared F5 evaluator source hash differs")
    protocol_path = _regular_file(evaluation_protocol, "frozen F6 evaluation protocol", ".md")
    if _sha256(protocol_path) != EVALUATION_PROTOCOL_SHA256:
        raise F6EvaluationError("frozen F6 evaluation protocol hash differs")
    scene_list_path = _require_frozen_file(scene_list, "paper100 scene list", EXPECTED_SCENE_LIST_SHA256)
    scenes = load_scene_list(scene_list_path)
    if _sha256(scene_list_path) != EXPECTED_SCENE_LIST_SHA256:
        raise F6EvaluationError("paper100 scene list changed while read")
    if len(scenes) != EXPECTED["scene_count"] or len(set(scenes)) != len(scenes):
        raise F6EvaluationError("paper100 scene count/order differs")
    selector_fixed = {
        "scene_list": scene_list_path,
        "evaluation_protocol": protocol_path,
        "f4_receipt": f4_receipt,
        "f6_receipt": f6_receipt,
        "evaluator_source": Path(__file__).resolve(),
    }
    selector_before = _selector_snapshot(
        scenes=scenes,
        fixed=selector_fixed,
        f4_sidecar_root=f4_sidecar_root,
        f6_sidecar_root=f6_sidecar_root,
    )
    expected_selector_hashes = {
        "scene_list": EXPECTED_SCENE_LIST_SHA256,
        "evaluation_protocol": EVALUATION_PROTOCOL_SHA256,
        "f4_receipt": EXPECTED_F4_RECEIPT_SHA256,
        "f6_receipt": EXPECTED_F6_RECEIPT_SHA256,
    }
    for name, expected_sha in expected_selector_hashes.items():
        if selector_before["fixed"].get(name, {}).get("sha256") != expected_sha:
            raise F6EvaluationError(f"frozen selector input differs: {name}")
    f4_payload = _read_pinned_json(f4_receipt, "F4 merge", EXPECTED_F4_RECEIPT_SHA256)
    f6_payload = _read_pinned_json(f6_receipt, "F6 merge", EXPECTED_F6_RECEIPT_SHA256)
    f4_rows = shared._validate_merge(
        f4_payload,
        schema=F4_MERGE_SCHEMA,
        protocol_id=F4_PROTOCOL_ID,
        scenes=scenes,
        label="F4",
    )
    f6_rows = _validate_f6_merge(f6_payload, scenes=scenes)
    prevalidated: dict[str, list[SelectedSource]] = {}
    counts = {"keyframe_count": 0, "successful_frame_count": 0, "source_count": 0}
    global_ids: list[str] = []
    global_lineages: list[str] = []
    global_results: list[str] = []
    hypothesis_counts = {name: 0 for name in HYPOTHESES}
    switch_count = 0
    switch_scenes = 0
    for scene_index, scene in enumerate(scenes):
        f4_path, f4_sha = _seal_from_row(f4_rows[scene], scene)
        f6_path, f6_sha = _seal_from_row(f6_rows[scene], scene)
        if (
            f4_path.resolve() != (f4_sidecar_root / f"{scene}.json").resolve()
            or f6_path.resolve() != (f6_sidecar_root / f"{scene}.json").resolve()
            or selector_before["scenes"][scene]["f4_sidecar"]["sha256"] != f4_sha
            or selector_before["scenes"][scene]["f6_sidecar"]["sha256"] != f6_sha
        ):
            raise F6EvaluationError(f"F4/F6 selector sidecar seal differs: {scene}")
        sources, keyframes, successful, summary = _load_scene_sources_pre_gt(
            scene=scene,
            scene_index=scene_index,
            f4_path=f4_path,
            f4_sha=f4_sha,
            f6_path=f6_path,
            f6_sha=f6_sha,
        )
        for label, merge_row, keys in (
            ("F4", f4_rows[scene], ("counts", "source_ids_sha256", "source_lineage_sha256")),
            ("F6", f6_rows[scene], ("counts", "source_ids_sha256", "source_lineage_sha256", "result_ledger_sha256")),
        ):
            for key in keys:
                if merge_row.get(key) != summary[label.lower()][key]:
                    raise F6EvaluationError(f"{label} merge/sidecar ledger differs: {scene}/{key}")
        prevalidated[scene] = sources
        counts["keyframe_count"] += keyframes
        counts["successful_frame_count"] += successful
        counts["source_count"] += len(sources)
        global_ids.extend(source.source_id for source in sources)
        global_lineages.extend(source.source_lineage_sha256 for source in sources)
        global_results.extend(source.result_sha256 for source in sources)
        scene_switches = sum(source.switched_from_base for source in sources)
        switch_count += scene_switches
        switch_scenes += int(scene_switches > 0)
        for source in sources:
            hypothesis_counts[source.selected_hypothesis] += 1
    if counts != {key: EXPECTED[key] for key in counts} or len(set(global_ids)) != EXPECTED["source_count"]:
        raise F6EvaluationError(f"pre-GT F6 census/identity differs: {counts}")
    global_ledgers = {
        "source_ids_sha256": _canonical_json_sha256(global_ids),
        "source_lineage_sha256": _canonical_json_sha256(global_lineages),
        "result_ledger_sha256": _canonical_json_sha256(global_results),
    }
    expected_ledgers = {
        "source_ids_sha256": EXPECTED_SOURCE_IDS_SHA256,
        "source_lineage_sha256": EXPECTED_SOURCE_LINEAGE_SHA256,
        "result_ledger_sha256": EXPECTED_RESULT_LEDGER_SHA256,
    }
    f4_coverage = f4_payload["coverage"]
    f6_coverage = f6_payload["coverage"]
    for name, actual in global_ledgers.items():
        if actual != expected_ledgers[name] or f6_coverage.get(name) != actual:
            raise F6EvaluationError(f"F6 global {name} differs")
        if name != "result_ledger_sha256" and f4_coverage.get(name) != actual:
            raise F6EvaluationError(f"F4/F6 global {name} differs")
    selection = f6_payload["selection"]
    if (
        any(selection.get(f"selected_{name.lower()}_count") != count for name, count in hypothesis_counts.items())
        or selection.get("switch_count") != switch_count
        or selection.get("switch_scene_count") != switch_scenes
    ):
        raise F6EvaluationError("F6 selected-hypothesis/base-switch census differs")
    selector_after = _selector_snapshot(
        scenes=scenes,
        fixed=selector_fixed,
        f4_sidecar_root=f4_sidecar_root,
        f6_sidecar_root=f6_sidecar_root,
    )
    if selector_before != selector_after:
        raise F6EvaluationError("selector inputs changed during no-GT prevalidation")

    # Phase 2: only the fully authenticated selector is evaluated against GT.
    official_paths = _official_dependency_paths(official_evaluator)
    evaluation_fixed = {
        "f4_report": f4_report,
        "official_evaluator": official_paths["official_evaluator"],
        "official_eval_det": official_paths["official_eval_det"],
        "official_metric_util": official_paths["official_metric_util"],
        "official_box_util": official_paths["official_box_util"],
        "oracle_helper_source": ORACLE_HELPER_SOURCE,
        "shared_evaluator_source": SHARED_EVALUATOR_SOURCE,
    }
    before = _evaluation_snapshot(
        scenes=scenes,
        fixed=evaluation_fixed,
        selector_snapshot=selector_before,
        baseline_root=baseline_root,
        gt_root=gt_root,
        scan_root=scan_root,
    )
    expected_fixed_hashes = {
        **expected_selector_hashes,
        "f4_report": EXPECTED_F4_REPORT_SHA256,
        "official_evaluator": EXPECTED_OFFICIAL_EVALUATOR_SHA256,
        "official_eval_det": EXPECTED_OFFICIAL_EVAL_DET_SHA256,
        "official_metric_util": EXPECTED_OFFICIAL_METRIC_UTIL_SHA256,
        "official_box_util": EXPECTED_OFFICIAL_BOX_UTIL_SHA256,
        "oracle_helper_source": EXPECTED_ORACLE_HELPER_SHA256,
        "shared_evaluator_source": EXPECTED_SHARED_EVALUATOR_SHA256,
    }
    for name, expected_sha in expected_fixed_hashes.items():
        if before["fixed"].get(name, {}).get("sha256") != expected_sha:
            raise F6EvaluationError(f"frozen evaluation input differs: {name}")
    f4_report_payload = _read_pinned_json(f4_report, "historical F4 report", EXPECTED_F4_REPORT_SHA256)
    if (
        f4_report_payload.get("schema") != F4_REPORT_SCHEMA
        or f4_report_payload.get("scene_order") != list(scenes)
        or f4_report_payload.get("integrity", {}).get("all_inputs_before_after_identity") is not True
    ):
        raise F6EvaluationError("historical F4 report schema differs")
    _validate_historical_f4_input_lineage(f4_report_payload, snapshot=before, scenes=scenes)
    official_eval_det = _load_official_eval_det(official_paths)
    gt_counts: list[int] = []
    native_iou: list[np.ndarray] = []
    selected_iou: list[np.ndarray] = []
    all_base_iou: list[np.ndarray] = []
    selected_by_scene: list[list[SelectedSource]] = []
    totals = {**counts, "scene_count": len(scenes), "native_count": 0, "gt_count": 0}
    for scene in scenes:
        snapshot = before["scenes"][scene]
        alignment_path = scan_root / scene / f"{scene}.txt"
        gt_path = gt_root / f"{scene}_bbox.npy"
        native_path = baseline_root / f"{scene}_boxes.pkl"
        for name, path in (("alignment", alignment_path), ("gt", gt_path), ("native", native_path)):
            _require_frozen_file(path, f"{scene} {name}", snapshot[name]["sha256"])
        alignment = load_axis_alignment(alignment_path)
        gt = load_gt_minmax(gt_path)
        _, native = load_baseline_boxes(native_path, alignment)
        for name, path in (("alignment", alignment_path), ("gt", gt_path), ("native", native_path)):
            _require_frozen_file(path, f"{scene} {name}", snapshot[name]["sha256"])
        aligned_sources: list[SelectedSource] = []
        boxes: list[np.ndarray] = []
        base_boxes: list[np.ndarray] = []
        for source in prevalidated[scene]:
            aligned = _align_corners(source.world_corners, alignment, f"{source.source_id}.selected_corners")
            aligned_base = _align_corners(
                source.base_world_corners, alignment, f"{source.source_id}.base_corners"
            )
            boxes.append(aligned)
            base_boxes.append(aligned_base)
            aligned_sources.append(
                SelectedSource(
                    **{
                        **source.__dict__,
                        "aligned_minmax": aligned,
                        "aligned_base_minmax": aligned_base,
                    }
                )
            )
        selected_boxes = np.stack(boxes) if boxes else np.empty((0, 6), dtype=np.float64)
        all_base_boxes = np.stack(base_boxes) if base_boxes else np.empty((0, 6), dtype=np.float64)
        gt_counts.append(len(gt))
        native_iou.append(aligned_iou_matrix(native, gt))
        selected_iou.append(aligned_iou_matrix(selected_boxes, gt))
        all_base_iou.append(aligned_iou_matrix(all_base_boxes, gt))
        selected_by_scene.append(aligned_sources)
        totals["native_count"] += len(native)
        totals["gt_count"] += len(gt)
    if totals["native_count"] != EXPECTED["native_count"] or totals["gt_count"] != EXPECTED["gt_count"]:
        raise F6EvaluationError(f"native/GT paper100 census differs: {totals}")
    baseline = {
        threshold: _authenticated_official_constant_evaluate(native_iou, gt_counts, threshold, official_eval_det)
        for threshold in THRESHOLDS
    }
    baseline_checks: dict[str, Any] = {}
    per_threshold: dict[str, Any] = {}
    for threshold in THRESHOLDS:
        key = _threshold_key(threshold)
        actual_ap = float(baseline[threshold]["ap_points"])
        expected_ap = EXPECTED_BASELINE_AP_POINTS[key]
        baseline_checks[key] = {
            "expected_ap_points": expected_ap,
            "actual_ap_points": actual_ap,
            "passed": math.isclose(actual_ap, expected_ap, rel_tol=0.0, abs_tol=1.0e-9),
        }
        try:
            f4_threshold = f4_report_payload["per_threshold"][key]
            f4_g4 = int(f4_threshold["identity_constrained_g4"]["additional_union_matching_over_native"])
            f4_baseline_ap = float(f4_threshold["baseline_official_constant_score"]["ap_points"])
        except (KeyError, TypeError, ValueError) as error:
            raise F6EvaluationError(f"historical F4 aggregate is absent at {key}") from error
        if not math.isclose(f4_baseline_ap, actual_ap, rel_tol=0.0, abs_tol=1.0e-12):
            raise F6EvaluationError(f"historical F4 baseline reproduction differs at {key}")
        per_threshold[key] = evaluate_selected_threshold(
            scenes=scenes,
            native_iou=native_iou,
            selected_iou=selected_iou,
            all_base_iou=all_base_iou,
            selected_sources=selected_by_scene,
            gt_counts=gt_counts,
            baseline_evaluation=baseline[threshold],
            threshold=threshold,
            f4_g4_additional_union_matches=f4_g4,
            official_eval_det=official_eval_det,
        )
    baseline_passed = all(row["passed"] for row in baseline_checks.values())
    decision = f6_decision(
        per_threshold=per_threshold,
        no_gt_merge_passed=True,
        baseline_passed=baseline_passed,
    )
    after = _evaluation_snapshot(
        scenes=scenes,
        fixed=evaluation_fixed,
        selector_snapshot=selector_before,
        baseline_root=baseline_root,
        gt_root=gt_root,
        scan_root=scan_root,
    )
    selector_final = _selector_snapshot(
        scenes=scenes,
        fixed=selector_fixed,
        f4_sidecar_root=f4_sidecar_root,
        f6_sidecar_root=f6_sidecar_root,
    )
    if before != after or selector_before != selector_final:
        raise F6EvaluationError("one or more sealed inputs changed during F6 evaluation")
    return {
        "schema": SCHEMA,
        "evaluation_protocol_id": EVALUATION_PROTOCOL_ID,
        "oracle_only": True,
        "deployable": False,
        "gt_used": True,
        "actual_f6_ap": False,
        "active_birth_authorized": False,
        "one_selected_geometry_per_source": True,
        "formal_score": 1.0,
        "strict_iou_comparison": ">",
        "scene_order": scenes,
        "totals": totals,
        "selected_hypothesis_counts": hypothesis_counts,
        "selected_base_switch_counts": {
            "base": EXPECTED["source_count"] - switch_count,
            "switch": switch_count,
            "switch_scene_count": switch_scenes,
        },
        "global_source_ledgers": global_ledgers,
        "authenticated_official_evaluator": {
            "entrypoint_sha256": EXPECTED_OFFICIAL_EVALUATOR_SHA256,
            "eval_det_sha256": EXPECTED_OFFICIAL_EVAL_DET_SHA256,
            "matrix_lookup_supplied_explicitly": True,
            "baseline_and_suffix_crosschecked": True,
        },
        "no_gt_f6_merge": {
            "integrity_passed": True,
            "causality_passed": True,
            "determinism_passed": True,
            "runtime_passed": True,
            "all_frozen_gates_passed": True,
        },
        "native_baseline_reproduction": baseline_checks,
        "per_threshold": per_threshold,
        "decision": decision,
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
        raise F6EvaluationError("F6 evaluation output must use .json")
    if output.exists() or output.is_symlink():
        raise F6EvaluationError(f"refusing to overwrite F6 evaluation output: {output}")
    if any(_is_within(output, root) for root in protected_roots):
        raise F6EvaluationError("F6 evaluation output lies inside a protected input root")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, default=DEFAULT_SCENE_LIST)
    parser.add_argument("--evaluation-protocol", type=Path, default=DEFAULT_EVALUATION_PROTOCOL)
    parser.add_argument("--f4-receipt", type=Path, default=DEFAULT_F4_RECEIPT)
    parser.add_argument("--f4-sidecar-root", type=Path, default=DEFAULT_F4_SIDECARS)
    parser.add_argument("--f6-receipt", type=Path, default=DEFAULT_F6_RECEIPT)
    parser.add_argument("--f6-sidecar-root", type=Path, default=DEFAULT_F6_SIDECARS)
    parser.add_argument("--f4-report", type=Path, default=DEFAULT_F4_REPORT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--scan-root", type=Path, default=DEFAULT_SCAN_ROOT)
    parser.add_argument("--official-evaluator", type=Path, default=DEFAULT_OFFICIAL_EVALUATOR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_output_path(
        args.out,
        (
            args.scene_list.parent,
            args.evaluation_protocol.parent,
            args.f4_receipt.parent,
            args.f4_sidecar_root,
            args.f6_receipt.parent,
            args.f6_sidecar_root,
            args.f4_report.parent,
            args.baseline_root,
            args.gt_root,
            args.scan_root,
            args.official_evaluator.parent,
        ),
    )
    report = audit_scannet_fastsam_f6_mvdc_paper100(
        scene_list=args.scene_list,
        evaluation_protocol=args.evaluation_protocol,
        f4_receipt=args.f4_receipt,
        f4_sidecar_root=args.f4_sidecar_root,
        f6_receipt=args.f6_receipt,
        f6_sidecar_root=args.f6_sidecar_root,
        f4_report=args.f4_report,
        baseline_root=args.baseline_root,
        gt_root=args.gt_root,
        scan_root=args.scan_root,
        official_evaluator=args.official_evaluator,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="ascii") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"schema": SCHEMA, "out": os.fspath(args.out), "decision": report["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
