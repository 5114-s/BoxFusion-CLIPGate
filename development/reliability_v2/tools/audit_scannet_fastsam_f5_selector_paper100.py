#!/usr/bin/env python3
"""One-shot F5 selected-geometry paper100 capacity evaluation.

This program is intentionally separate from the GT-free F5 selector.  It
opens protected evaluation inputs only after the complete F4/F5 identity,
integrity, causality, determinism, and runtime receipts have passed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

from tools.audit_scannet_boxer_unexplained_oracle import (  # noqa: E402
    aligned_iou_matrix,
    load_axis_alignment,
    load_baseline_boxes,
    load_gt_minmax,
    load_scene_list,
    official_constant_evaluate,
    strict_maximum_matching,
)


SCHEMA = "boxfusion.scannet_fastsam_f5_selector_paper100_evaluation.v1"
EVALUATION_PROTOCOL_ID = "F5-GT-FREE-SELECTOR-ONE-SHOT-EVALUATION-PAPER100"
EVALUATION_PROTOCOL_SHA256 = "5eb3120808ff61fcc2ffabb3b2912f1057a82c3af4222a8d7018f767901b07f7"
F5_PROTOCOL_ID = "F5-GT-FREE-PAST-ONLY-GEOMETRY-SELECTOR-PAPER100"
F5_PROTOCOL_SHA256 = "2a6d62fa9d5912dc3871bbc485f44987565bda61b818722b3a4e6577d34a6afc"
F4_PROTOCOL_ID = "F4-FASTSAM-BOXER-GEOMETRY-SHADOW-PAPER100"
F5_MERGE_SCHEMA = "boxfusion.scannet_fastsam_f5_gtfree_selector_paper100.merge.v1"
F5_SCENE_SCHEMA = "boxfusion.scannet_fastsam_f5_gtfree_selector_paper100.scene.v1"
F4_MERGE_SCHEMA = "boxfusion.scannet_fastsam_f4_boxer_paper100.merge.v1"
F4_SCENE_SCHEMA = "boxfusion.scannet_fastsam_f4_boxer_paper100.scene.v1"
F4_REPORT_SCHEMA = "boxfusion.scannet_fastsam_f4_boxer_paper100_oracle.v1"

EXPECTED = {
    "scene_count": 100,
    "keyframe_count": 6_817,
    "successful_frame_count": 6_726,
    "source_count": 52_299,
    "native_count": 1_788,
    "gt_count": 1_433,
}
THRESHOLDS = (0.15, 0.25, 0.50)
EXPECTED_BASELINE_AP_POINTS = {
    "0.15": 31.0130259031,
    "0.25": 26.7911284298,
    "0.50": 12.0668518301,
}
EXPECTED_SCENE_LIST_SHA256 = "4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5"
EXPECTED_F4_RECEIPT_SHA256 = "0e00ab68e2525b8e1262dfb12bc08ee3a98f02d70b158960f49379e957f826a6"
EXPECTED_F5_RECEIPT_SHA256 = "9e8bd97a2e29a6f5b68fc1532ef1ff1f4c5698115df0631db38486269bea300e"
EXPECTED_F4_REPORT_SHA256 = "1cb6da29b31ad35471578ef56a355a578c0b66c77e9f36d38d607191dfba4669"
EXPECTED_OFFICIAL_EVALUATOR_SHA256 = "aea2a72940b7cc53ee273f9f235e2efc848e1994e22da5f439af9751e1e27c27"
EXPECTED_OFFICIAL_EVAL_DET_SHA256 = "6ef54c395e46716e364547115090bae96643bf346b3e8eb1b859719781a557dd"
EXPECTED_OFFICIAL_METRIC_UTIL_SHA256 = "08ae1dbdd0a9f8cae07b48e749ea6b300b6ddd73c1c9ed320b5c2d4678cc6661"
EXPECTED_OFFICIAL_BOX_UTIL_SHA256 = "44aadf0088c0ccd5e9f51a1cded22fb1080d59aa50d0fb914fe6e83896aaa107"
EXPECTED_ORACLE_HELPER_SHA256 = "fcd29c9cd33199544e7f5221f78d916b8c1ae7f8e83007a28e9c6fa834f65a50"
EXPECTED_SOURCE_IDS_SHA256 = "14f53caeb3e538756fb9edcebcee8fc33f2c446b902e607b6cb9371c495082e0"
EXPECTED_SOURCE_LINEAGE_SHA256 = "49944f8abb28b7bae6b28ad0e7031d9473f1d566ddcd2d479ed21b58e832a653"
EXPECTED_RESULT_LEDGER_SHA256 = "c8d8497a82db2ec9dcdc81c0bc28937c0c4ae966ff662fee624269a90569eda4"
REQUIRED_ADDITIONAL_MATCHES = 144
TARGET_DELTA_AP_POINTS = 10.0
HYPOTHESES = ("H0", "HL", "HLG", "HB")
F5_GATE_NAMES = frozenset(
    {
        "integrity_complete", "exact_keyframes", "exact_successful_frames",
        "exact_unique_sources", "identity_verified_sources", "one_selection_per_source",
        "selected_hb_proof_count", "selected_hb_min_sources", "selected_hb_min_scenes",
        "selected_hb_max_fraction", "prefix_replay", "independent_cpu_replay",
        "maximum_lookahead_frames", "maximum_buffered_frames",
        "maximum_sources_per_buffered_frame", "native_output_mutation_count",
        "source_addition_or_removal_count", "score_rank_semantic_mutation_count",
        "forbidden_access_count", "training_or_online_learning_count", "birth_count",
        "f5_incremental_warm_p95_ms", "replay_composed_warm_p95_ms",
        "replay_composed_warm_max_ms", "replay_composed_warm_mean_per_source_frame_ms",
        "gap25_warm_deadline_miss_count", "cuda_peak_memory_bytes",
        "f5_cuda_allocated_bytes",
    }
)
F4_GATE_NAMES = frozenset(
    {
        "integrity_complete", "exact_keyframes", "exact_successful_frames", "exact_sources",
        "native_output_mutation_count", "f4_incremental_warm_p95_ms",
        "replay_composed_warm_p95_ms", "replay_composed_warm_max_ms",
        "replay_composed_mean_per_source_frame_ms", "gap25_warm_deadline_miss_count",
        "cuda_peak_memory_bytes",
    }
)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SIGNS = np.asarray(
    [
        [-1.0, -1.0, -1.0], [-1.0, -1.0, +1.0],
        [-1.0, +1.0, -1.0], [-1.0, +1.0, +1.0],
        [+1.0, -1.0, -1.0], [+1.0, -1.0, +1.0],
        [+1.0, +1.0, -1.0], [+1.0, +1.0, +1.0],
    ],
    dtype=np.float64,
)

DEFAULT_SCENE_LIST = REPOSITORY_ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt"
DEFAULT_EVALUATION_PROTOCOL = REPOSITORY_ROOT / "docs/F5_GT_FREE_SELECTOR_EVALUATION_PROTOCOL_FREEZE.md"
DEFAULT_F4_RECEIPT = REPOSITORY_ROOT / "logs/scannet_fastsam_f4_boxer_paper100_score05/final/F4_FASTSAM_BOXER_PAPER100.json"
DEFAULT_F4_SIDECARS = REPOSITORY_ROOT / "logs/scannet_fastsam_f4_boxer_paper100_score05/scenes"
DEFAULT_F5_RECEIPT = REPOSITORY_ROOT / "logs/scannet_fastsam_f5_selector_paper100_score05/final/F5_GT_FREE_SELECTOR_PAPER100.json"
DEFAULT_F5_SIDECARS = REPOSITORY_ROOT / "logs/scannet_fastsam_f5_selector_paper100_score05/scenes"
DEFAULT_F4_REPORT = REPOSITORY_ROOT / "reports/fastsam_f4_boxer_paper100_oracle/F4_FASTSAM_BOXER_PAPER100_ORACLE.json"
DEFAULT_BASELINE_ROOT = REPOSITORY_ROOT / "results/scannet_t05_boxer_replay_active_score05"
DEFAULT_GT_ROOT = REPOSITORY_ROOT / "evaluation/data_util/scannet_train_detection_data"
DEFAULT_SCAN_ROOT = Path("/extra/ZhaoX/scannet_data/scans")
DEFAULT_OFFICIAL_EVALUATOR = REPOSITORY_ROOT / "upstream_clean/BoxFusion_shallow/evaluation/eval_scannet.py"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "reports/fastsam_f5_selector_paper100/F5_FASTSAM_SELECTOR_PAPER100_EVALUATION.json"
ORACLE_HELPER_SOURCE = REPOSITORY_ROOT / "tools/audit_scannet_boxer_unexplained_oracle.py"


class F5EvaluationError(ValueError):
    """Raised when an input or frozen evaluation contract differs."""


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
    world_corners: np.ndarray
    aligned_minmax: np.ndarray | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise F5EvaluationError("value is not canonical finite ASCII JSON") from error
    return hashlib.sha256(payload).hexdigest()


def _regular_file(path: Path, label: str, suffix: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file():
        raise F5EvaluationError(f"{label} must be a regular non-symlink file: {path}")
    source = path.resolve()
    if suffix is not None and source.suffix.lower() != suffix:
        raise F5EvaluationError(f"{label} must have suffix {suffix}: {source}")
    return source


def _read_json(path: Path, label: str) -> dict[str, Any]:
    source = _regular_file(path, label, ".json")
    try:
        result = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise F5EvaluationError(f"invalid {label}: {source}") from error
    if not isinstance(result, dict):
        raise F5EvaluationError(f"{label} must contain one JSON object")
    return result


def _read_pinned_json(path: Path, label: str, expected_sha256: str) -> dict[str, Any]:
    source = _regular_file(path, label, ".json")
    if _sha256(source) != expected_sha256:
        raise F5EvaluationError(f"{label} does not match its frozen SHA-256")
    result = _read_json(source, label)
    if _sha256(source) != expected_sha256:
        raise F5EvaluationError(f"{label} changed while it was read")
    return result


def _require_frozen_file(path: Path, label: str, expected_sha256: str) -> Path:
    source = _regular_file(path, label)
    if _sha256(source) != expected_sha256:
        raise F5EvaluationError(f"{label} does not match its frozen SHA-256")
    return source


def _content_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return _canonical_json_sha256(payload)


def _validate_historical_f4_input_lineage(
    report: Mapping[str, Any],
    *,
    snapshot: Mapping[str, Any],
    scenes: Sequence[str],
) -> None:
    historical_before = report.get("input_sha256_before")
    historical_after = report.get("input_sha256_after")
    if (
        not isinstance(historical_before, Mapping)
        or historical_before != historical_after
    ):
        raise F5EvaluationError("historical F4 before/after input seals differ")
    fixed = historical_before.get("fixed_files")
    ledgers = historical_before.get("ordered_scene_ledgers")
    current_fixed = snapshot.get("fixed")
    current_scenes = snapshot.get("scenes")
    if not all(isinstance(value, Mapping) for value in (fixed, ledgers, current_fixed, current_scenes)):
        raise F5EvaluationError("historical F4 input seal structure differs")
    fixed_mapping = {
        "scene_list": "scene_list",
        "official_evaluator": "official_evaluator",
        "f4_receipt": "f4_receipt",
    }
    for historical_name, current_name in fixed_mapping.items():
        historical = fixed.get(historical_name)
        current = current_fixed.get(current_name)
        if (
            not isinstance(historical, Mapping)
            or not isinstance(current, Mapping)
            or _hash_string(historical.get("sha256"), f"historical {historical_name} hash")
            != current.get("sha256")
        ):
            raise F5EvaluationError(f"current {current_name} differs from the historical F4 seal")
    ledger_mapping = {
        "f4_sidecars": "f4_sidecar",
        "native_predictions": "native",
        "ground_truth": "gt",
        "axis_alignment": "alignment",
    }
    for historical_name, current_name in ledger_mapping.items():
        ledger = ledgers.get(historical_name)
        if not isinstance(ledger, Mapping) or set(ledger) != {"entries", "sha256"}:
            raise F5EvaluationError(f"historical {historical_name} ledger is malformed")
        entries = ledger.get("entries")
        if not isinstance(entries, list) or _canonical_json_sha256(entries) != ledger.get("sha256"):
            raise F5EvaluationError(f"historical {historical_name} ledger hash differs")
        current_entries: list[list[str]] = []
        for scene in scenes:
            scene_snapshot = current_scenes.get(scene)
            if not isinstance(scene_snapshot, Mapping):
                raise F5EvaluationError(f"current snapshot lacks scene {scene}")
            seal = scene_snapshot.get(current_name)
            if not isinstance(seal, Mapping) or not isinstance(seal.get("sha256"), str):
                raise F5EvaluationError(f"current {current_name} seal is absent: {scene}")
            current_entries.append([scene, seal["sha256"]])
        if entries != current_entries:
            raise F5EvaluationError(f"current {current_name} differs from the historical F4 ledger")


def _official_dependency_paths(official_evaluator: Path) -> dict[str, Path]:
    entrypoint = _regular_file(official_evaluator, "official ScanNet evaluator", ".py")
    utility_root = entrypoint.parent / "utils"
    return {
        "official_evaluator": entrypoint,
        "official_eval_det": utility_root / "eval_det.py",
        "official_metric_util": utility_root / "metric_util.py",
        "official_box_util": utility_root / "box_util.py",
    }


def _load_official_eval_det(paths: Mapping[str, Path]) -> ModuleType:
    expected = {
        "official_evaluator": EXPECTED_OFFICIAL_EVALUATOR_SHA256,
        "official_eval_det": EXPECTED_OFFICIAL_EVAL_DET_SHA256,
        "official_metric_util": EXPECTED_OFFICIAL_METRIC_UTIL_SHA256,
        "official_box_util": EXPECTED_OFFICIAL_BOX_UTIL_SHA256,
    }
    resolved = {
        name: _require_frozen_file(path, name.replace("_", " "), expected[name])
        for name, path in paths.items()
    }
    def forbidden_default_iou(*_: object, **__: object) -> float:
        raise F5EvaluationError("official eval_det attempted an unauthenticated default IoU path")

    # The frozen eval_det module imports its default geometric IoU helpers at
    # module load time.  F5 always supplies the explicit sealed-matrix lookup
    # below, so inject fail-closed placeholders for these otherwise-unused
    # imports.  Their source files are still hash-pinned above.
    metric_stub = ModuleType("metric_util")
    metric_stub.calc_iou = forbidden_default_iou  # type: ignore[attr-defined]
    box_stub = ModuleType("box_util")
    box_stub.box3d_iou = forbidden_default_iou  # type: ignore[attr-defined]
    box_stub.box3d_iou_v2 = forbidden_default_iou  # type: ignore[attr-defined]
    tqdm_stub = ModuleType("tqdm")
    tqdm_stub.tqdm = lambda iterable: iterable  # type: ignore[attr-defined]
    previous_modules = {
        name: sys.modules.get(name) for name in ("metric_util", "box_util", "tqdm")
    }
    spec = importlib.util.spec_from_file_location(
        "_boxfusion_f5_frozen_official_eval_det", resolved["official_eval_det"]
    )
    if spec is None or spec.loader is None:
        raise F5EvaluationError("could not load the frozen official eval_det kernel")
    module = importlib.util.module_from_spec(spec)
    sys.modules["metric_util"] = metric_stub
    sys.modules["box_util"] = box_stub
    sys.modules["tqdm"] = tqdm_stub
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise F5EvaluationError("could not execute the frozen official eval_det kernel") from error
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    eval_det_cls = getattr(module, "eval_det_cls", None)
    code = getattr(eval_det_cls, "__code__", None)
    if code is None or Path(code.co_filename).resolve() != resolved["official_eval_det"]:
        raise F5EvaluationError("official eval_det kernel provenance differs")
    for name, path in resolved.items():
        if _sha256(path) != expected[name]:
            raise F5EvaluationError(f"{name} changed while the official evaluator was loaded")
    return module


def _authenticated_official_constant_evaluate(
    iou_by_scene: Sequence[np.ndarray],
    gt_counts: Sequence[int],
    threshold: float,
    official_eval_det: ModuleType,
) -> dict[str, object]:
    reference = official_constant_evaluate(iou_by_scene, gt_counts, threshold)
    pred: dict[int, list[tuple[np.ndarray, float]]] = {}
    gt: dict[int, list[np.ndarray]] = {}
    for scene_index, (matrix, gt_count) in enumerate(zip(iou_by_scene, gt_counts, strict=True)):
        array = np.asarray(matrix, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != int(gt_count) or not np.isfinite(array).all():
            raise F5EvaluationError("official evaluator IoU inputs are malformed")
        pred[scene_index] = [
            (np.asarray([scene_index, prediction_index], dtype=np.float64), 1.0)
            for prediction_index in range(array.shape[0])
        ]
        gt[scene_index] = [
            np.asarray([scene_index, gt_index], dtype=np.float64)
            for gt_index in range(array.shape[1])
        ]

    def lookup(prediction_box: np.ndarray, gt_box: np.ndarray) -> float:
        scene_index = int(prediction_box[0])
        if scene_index != int(gt_box[0]):
            raise F5EvaluationError("official evaluator crossed scene identities")
        return float(iou_by_scene[scene_index][int(prediction_box[1]), int(gt_box[1])])

    try:
        recall, precision, ap = official_eval_det.eval_det_cls(
            pred,
            gt,
            ovthresh=threshold,
            use_07_metric=False,
            get_iou_func=lookup,
        )
    except Exception as error:
        raise F5EvaluationError("frozen official eval_det execution failed") from error
    official_values = {
        "ap_points": 100.0 * float(ap),
        "recall": float(recall[-1]) if len(recall) else 0.0,
        "precision": float(precision[-1]) if len(precision) else 0.0,
    }
    for key, value in official_values.items():
        if not math.isclose(value, float(reference[key]), rel_tol=0.0, abs_tol=1.0e-12):
            raise F5EvaluationError(f"authenticated official eval_det differs at {key}")
    result = dict(reference)
    result["authenticated_official_eval_det"] = True
    result["official_eval_det_ap_points"] = official_values["ap_points"]
    return result


def _hash_string(value: object, label: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise F5EvaluationError(f"{label} must be a lowercase SHA-256")
    return value


def _array(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise F5EvaluationError(f"{label} must be a finite array of shape {shape}") from error
    if result.shape != shape or not np.isfinite(result).all():
        raise F5EvaluationError(f"{label} must be a finite array of shape {shape}")
    return np.ascontiguousarray(result)


def _threshold_key(threshold: float) -> str:
    return f"{threshold:.2f}"


def _json_evaluation(value: Mapping[str, Any], scenes: Sequence[str]) -> dict[str, Any]:
    masks = value.get("matched_gt_masks")
    if not isinstance(masks, list) or len(masks) != len(scenes):
        raise F5EvaluationError("official evaluation scene masks differ")
    return {
        key: item
        for key, item in value.items()
        if key not in {"matched_gt_masks", "evaluation_order"}
    } | {
        "per_scene": {
            scene: {
                "greedy_tp": int(np.count_nonzero(mask)),
                "unmatched_gt_count": int(len(mask) - np.count_nonzero(mask)),
                "matched_gt_indices": np.flatnonzero(mask).astype(int).tolist(),
            }
            for scene, mask in zip(scenes, masks, strict=True)
        }
    }


def _aabb_corners(q02: object, q98: object, label: str) -> np.ndarray:
    lower = _array(q02, (3,), f"{label}.q02")
    upper = _array(q98, (3,), f"{label}.q98")
    if np.any(upper <= lower):
        raise F5EvaluationError(f"{label} has non-positive extent")
    return np.where(_SIGNS > 0.0, upper[None, :], lower[None, :])


def _align_corners(corners: object, alignment: object, label: str) -> np.ndarray:
    points = _array(corners, (8, 3), label)
    matrix = _array(alignment, (4, 4), "axis alignment")
    transformed = points @ matrix[:3, :3].T + matrix[:3, 3]
    bounds = np.concatenate((transformed.min(axis=0), transformed.max(axis=0)))
    if np.any(bounds[3:] <= bounds[:3]):
        raise F5EvaluationError(f"{label} produces a degenerate aligned box")
    return bounds


def _expected_selected_geometry(
    name: str, hypothesis: Mapping[str, Any]
) -> tuple[dict[str, Any], np.ndarray]:
    if name in ("H0", "HL", "HLG"):
        corners = _aabb_corners(hypothesis.get("q02"), hypothesis.get("q98"), name)
        lower = corners.min(axis=0)
        upper = corners.max(axis=0)
        center = _array(hypothesis.get("center"), (3,), f"{name}.center")
        extent = _array(hypothesis.get("extent"), (3,), f"{name}.extent")
        if not np.allclose(center, (lower + upper) * 0.5, rtol=0.0, atol=1.0e-9) or not np.allclose(
            extent, upper - lower, rtol=0.0, atol=1.0e-9
        ):
            raise F5EvaluationError(f"{name} center/extent differ from q02/q98")
        geometry = {
            "kind": "world_aabb",
            "hypothesis": name,
            "q02": lower.tolist(),
            "q98": upper.tolist(),
            "center": center.tolist(),
            "extent": extent.tolist(),
        }
        return geometry, corners
    if name != "HB" or hypothesis.get("valid") is not True:
        raise F5EvaluationError("selected HB is not a valid sealed F4 hypothesis")
    center = _array(hypothesis.get("world_center"), (3,), "HB.world_center")
    extent = _array(hypothesis.get("local_extent"), (3,), "HB.local_extent")
    rotation = _array(hypothesis.get("world_rotation"), (3, 3), "HB.world_rotation")
    corners = _array(hypothesis.get("world_corners"), (8, 3), "HB.world_corners")
    if np.any(extent <= 0.0) or float(np.linalg.det(rotation)) <= 0.0 or not np.allclose(
        rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1.0e-3
    ):
        raise F5EvaluationError("selected HB violates the frozen OBB geometry")
    expected_corners = center[None, :] + (_SIGNS * (extent[None, :] * 0.5)) @ rotation.T
    if not np.allclose(corners, expected_corners, rtol=0.0, atol=2.0e-6):
        raise F5EvaluationError("selected HB corners differ from center/extent/rotation")
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
        "envelope_center": ((lower + upper) * 0.5).tolist(),
        "envelope_extent": (upper - lower).tolist(),
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
    f5_source: Mapping[str, Any],
) -> SelectedSource:
    source_id = f4_source.get("source_id")
    source_lineage_sha256 = _hash_string(
        f4_source.get("source_lineage_sha256"),
        f"{scene}/{frame_id}/{rank} source lineage",
    )
    if (
        not isinstance(source_id, str)
        or f5_source.get("source_id") != source_id
        or f4_source.get("scene_index") != scene_index
        or f4_source.get("frame_id") != frame_id
        or f4_source.get("frame_ordinal") != frame_ordinal
        or f4_source.get("rank") != rank
        or f4_source.get("candidate_index") != rank
        or f5_source.get("frame_id") != frame_id
        or f5_source.get("frame_ordinal") != frame_ordinal
        or f5_source.get("rank") != rank
        or f5_source.get("source_lineage_sha256") != source_lineage_sha256
    ):
        raise F5EvaluationError(f"F4/F5 source identity differs: {scene}/{frame_id}/{rank}")
    hypotheses = f4_source.get("hypotheses")
    if not isinstance(hypotheses, Mapping) or set(hypotheses) != set(HYPOTHESES):
        raise F5EvaluationError(f"F4 hypothesis ledger differs: {source_id}")
    input_hashes = f5_source.get("input_hypothesis_sha256")
    expected_hashes = {name: _canonical_json_sha256(hypotheses[name]) for name in HYPOTHESES}
    if input_hashes != expected_hashes:
        raise F5EvaluationError(f"F5 hypothesis hash ledger differs: {source_id}")
    selected = f5_source.get("selected_hypothesis")
    if selected not in HYPOTHESES:
        raise F5EvaluationError(f"F5 selected hypothesis differs: {source_id}")
    expected_geometry, corners = _expected_selected_geometry(selected, hypotheses[selected])
    actual_geometry = f5_source.get("selected_geometry")
    if actual_geometry != expected_geometry:
        raise F5EvaluationError(f"F5 selected geometry is not an exact F4 copy: {source_id}")
    expected_geometry_sha256 = _canonical_json_sha256(expected_geometry)
    if (
        not isinstance(actual_geometry, Mapping)
        or _canonical_json_sha256(actual_geometry) != expected_geometry_sha256
        or f5_source.get("selected_geometry_sha256") != expected_geometry_sha256
    ):
        raise F5EvaluationError(f"F5 selected geometry hash differs: {source_id}")
    if type(f5_source.get("formal_score")) is not float or f5_source.get("formal_score") != 1.0:
        raise F5EvaluationError(f"F5 formal score differs from 1.0: {source_id}")
    result_payload = dict(f5_source)
    result_sha = _hash_string(
        result_payload.pop("result_sha256", None), f"{source_id} result hash"
    )
    if result_sha != _canonical_json_sha256(result_payload):
        raise F5EvaluationError(f"F5 result row hash differs: {source_id}")
    return SelectedSource(
        scene_id=scene,
        scene_index=scene_index,
        frame_id=frame_id,
        frame_ordinal=frame_ordinal,
        rank=rank,
        source_id=source_id,
        source_lineage_sha256=source_lineage_sha256,
        result_sha256=result_sha,
        selected_hypothesis=str(selected),
        world_corners=corners,
    )


def _seal_from_row(row: Mapping[str, Any], scene: str) -> tuple[Path, str]:
    seal = row.get("sidecar")
    if not isinstance(seal, Mapping) or not isinstance(seal.get("path"), str):
        raise F5EvaluationError(f"scene sidecar seal is absent: {scene}")
    sha = _hash_string(seal.get("sha256"), f"{scene} sidecar hash")
    return Path(seal["path"]), sha


def _validate_merge(
    receipt: Mapping[str, Any],
    *,
    schema: str,
    protocol_id: str,
    scenes: Sequence[str],
    label: str,
) -> dict[str, Mapping[str, Any]]:
    if (
        receipt.get("schema") != schema
        or receipt.get("protocol_id") != protocol_id
        or receipt.get("complete") is not True
        or receipt.get("overall_pass") is not True
        or receipt.get("native_output_mutation_count") != 0
        or receipt.get("content_sha256") != _content_hash(receipt)
    ):
        raise F5EvaluationError(f"{label} merge integrity/pass contract differs")
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
        or not isinstance(totals, Mapping)
        or totals.get("keyframe_count") != EXPECTED["keyframe_count"]
        or totals.get("successful_frame_count") != EXPECTED["successful_frame_count"]
        or totals.get("source_count") != EXPECTED["source_count"]
        or totals.get("identity_verified_source_count") != EXPECTED["source_count"]
    ):
        raise F5EvaluationError(f"{label} paper100 census/order differs")
    gates = receipt.get("gates")
    expected_gate_names = F5_GATE_NAMES if label == "F5" else F4_GATE_NAMES
    if (
        not isinstance(gates, Mapping)
        or set(gates) != expected_gate_names
        or any(
            not isinstance(row, Mapping)
            or row.get("pass") is not True
            or row.get("passed") is not True
            for row in gates.values()
        )
    ):
        raise F5EvaluationError(f"{label} frozen gate ledger differs")
    if label == "F5":
        causality = receipt.get("causality")
        determinism = receipt.get("determinism")
        runtime = receipt.get("runtime")
        if (
            receipt.get("protocol_sha256") != F5_PROTOCOL_SHA256
            or receipt.get("decision") != "retain_f5_for_one_separately_sealed_evaluation_only"
            or coverage.get("result_ledger_sha256") != EXPECTED_RESULT_LEDGER_SHA256
            or not isinstance(causality, Mapping) or causality.get("overall_pass") is not True
            or not isinstance(determinism, Mapping) or determinism.get("overall_pass") is not True
            or not isinstance(runtime, Mapping) or runtime.get("overall_pass") is not True
        ):
            raise F5EvaluationError("F5 no-GT causality/determinism/runtime did not pass")
        authorization = receipt.get("evaluation_authorization")
        selection = receipt.get("selection")
        selected_counts = [
            selection.get(name) if isinstance(selection, Mapping) else None
            for name in ("selected_h0_count", "selected_hl_count", "selected_hlg_count", "selected_hb_count")
        ]
        if (
            not isinstance(authorization, Mapping)
            or authorization.get("allowed") is not True
            or authorization.get("birth_authorized") is not False
            or authorization.get("deployment_authorized") is not False
            or not isinstance(selection, Mapping)
            or type(selection.get("formal_score")) is not float
            or selection.get("formal_score") != 1.0
            or any(type(value) is not int or value < 0 for value in selected_counts)
            or sum(selected_counts) != EXPECTED["source_count"]
            or any(receipt.get(name) != 0 for name in (
                "birth_count", "native_output_mutation_count", "source_addition_or_removal_count",
                "score_rank_semantic_mutation_count", "forbidden_access_count",
                "training_or_online_learning_count",
            ))
        ):
            raise F5EvaluationError("F5 one-shot evaluation authorization differs")
    elif receipt.get("native_output_mutation_count") != 0:
        raise F5EvaluationError("F4 merge reports native output mutation")
    rows = receipt.get("scenes")
    if not isinstance(rows, list) or len(rows) != len(scenes):
        raise F5EvaluationError(f"{label} scene ledger differs")
    result: dict[str, Mapping[str, Any]] = {}
    for scene_index, (scene, row) in enumerate(zip(scenes, rows, strict=True)):
        if not isinstance(row, Mapping) or row.get("scene_id") != scene or row.get("scene_index") != scene_index:
            raise F5EvaluationError(f"{label} scene order differs")
        _seal_from_row(row, scene)
        result[scene] = row
    return result


def _load_scene_sources_pre_gt(
    *,
    scene: str,
    scene_index: int,
    f4_path: Path,
    f4_sha: str,
    f5_path: Path,
    f5_sha: str,
) -> tuple[list[SelectedSource], int, int, dict[str, Any]]:
    if _sha256(_regular_file(f4_path, f"{scene} F4 sidecar", ".json")) != f4_sha:
        raise F5EvaluationError(f"F4 sidecar rehash differs: {scene}")
    if _sha256(_regular_file(f5_path, f"{scene} F5 sidecar", ".json")) != f5_sha:
        raise F5EvaluationError(f"F5 sidecar rehash differs: {scene}")
    f4 = _read_json(f4_path, f"{scene} F4 sidecar")
    f5 = _read_json(f5_path, f"{scene} F5 sidecar")
    if _sha256(f4_path) != f4_sha or _sha256(f5_path) != f5_sha:
        raise F5EvaluationError(f"F4/F5 sidecar changed while it was read: {scene}")
    for value, schema, protocol, label in (
        (f4, F4_SCENE_SCHEMA, F4_PROTOCOL_ID, "F4"),
        (f5, F5_SCENE_SCHEMA, F5_PROTOCOL_ID, "F5"),
    ):
        if (
            value.get("schema") != schema
            or value.get("protocol_id") != protocol
            or value.get("complete") is not True
            or value.get("scene_id") != scene
            or value.get("scene_index") != scene_index
            or value.get("content_sha256") != _content_hash(value)
        ):
            raise F5EvaluationError(f"{label} scene contract differs: {scene}")
    if (
        f5.get("native_output_mutation_count") != 0
        or f5.get("birth_count") != 0
        or f5.get("causality", {}).get("overall_pass") is not True
        or f5.get("prefix_replay", {}).get("passed") is not True
        or f5.get("determinism", {}).get("passed") is not True
    ):
        raise F5EvaluationError(f"F5 scene no-GT proof differs: {scene}")
    f5_f4_seal = f5.get("inputs", {}).get("f4_sidecar")
    if not isinstance(f5_f4_seal, Mapping) or f5_f4_seal.get("path") != os.fspath(f4_path.resolve()) or f5_f4_seal.get("sha256") != f4_sha:
        raise F5EvaluationError(f"F5/F4 scene input seal differs: {scene}")
    f4_frames = f4.get("frames")
    f5_frames = f5.get("frames")
    if not isinstance(f4_frames, list) or not isinstance(f5_frames, list) or len(f4_frames) != len(f5_frames):
        raise F5EvaluationError(f"F4/F5 frame ledger differs: {scene}")
    selected: list[SelectedSource] = []
    successful = 0
    result_hashes: list[str] = []
    lineage_hashes: list[str] = []
    for ordinal, (f4_frame, f5_frame) in enumerate(zip(f4_frames, f5_frames, strict=True)):
        if not isinstance(f4_frame, Mapping) or not isinstance(f5_frame, Mapping):
            raise F5EvaluationError(f"invalid F4/F5 frame row: {scene}/{ordinal}")
        frame_id = f4_frame.get("frame_id")
        if (
            f4_frame.get("frame_ordinal") != ordinal
            or f5_frame.get("frame_ordinal") != ordinal
            or f5_frame.get("frame_id") != frame_id
            or f5_frame.get("successful") is not f4_frame.get("successful")
        ):
            raise F5EvaluationError(f"F4/F5 frame identity differs: {scene}/{ordinal}")
        f4_sources = f4_frame.get("sources")
        f5_sources = f5_frame.get("sources")
        if not isinstance(f4_sources, list) or not isinstance(f5_sources, list) or len(f4_sources) != len(f5_sources):
            raise F5EvaluationError(f"F4/F5 source count differs: {scene}/{frame_id}")
        if f5_frame.get("successful") is True:
            successful += 1
        elif f5_sources:
            raise F5EvaluationError(f"failed F5 frame retains sources: {scene}/{frame_id}")
        for rank, (f4_source, f5_source) in enumerate(zip(f4_sources, f5_sources, strict=True)):
            if not isinstance(f4_source, Mapping) or not isinstance(f5_source, Mapping):
                raise F5EvaluationError(f"invalid F4/F5 source row: {scene}/{frame_id}/{rank}")
            source = validate_selected_source(
                scene=scene,
                scene_index=scene_index,
                frame_id=int(frame_id),
                frame_ordinal=ordinal,
                rank=rank,
                f4_source=f4_source,
                f5_source=f5_source,
            )
            selected.append(source)
            result_hashes.append(str(f5_source["result_sha256"]))
            lineage_hashes.append(source.source_lineage_sha256)
    counts = f5.get("counts")
    f4_counts = f4.get("counts")
    source_ids = [row.source_id for row in selected]
    source_ids_sha256 = _canonical_json_sha256(source_ids)
    source_lineage_sha256 = _canonical_json_sha256(lineage_hashes)
    result_ledger_sha256 = _canonical_json_sha256(result_hashes)
    if (
        not isinstance(counts, Mapping)
        or counts.get("keyframe_count") != len(f5_frames)
        or counts.get("successful_frame_count") != successful
        or counts.get("source_count") != len(selected)
        or counts.get("identity_verified_source_count") != len(selected)
        or len(set(source_ids)) != len(source_ids)
        or f5.get("source_ids_sha256") != source_ids_sha256
        or f5.get("source_lineage_sha256") != source_lineage_sha256
        or f5.get("result_ledger_sha256") != result_ledger_sha256
    ):
        raise F5EvaluationError(f"F5 scene census/hash ledger differs: {scene}")
    if (
        not isinstance(f4_counts, Mapping)
        or f4_counts.get("keyframe_count") != len(f4_frames)
        or f4_counts.get("successful_frame_count") != successful
        or f4_counts.get("source_count") != len(selected)
        or f4_counts.get("valid_hb_count") != len(selected)
        or f4_counts.get("invalid_hb_count") != 0
        or f4.get("source_ids_sha256") != source_ids_sha256
        or f4.get("source_lineage_sha256") != source_lineage_sha256
    ):
        raise F5EvaluationError(f"F4 scene census/hash ledger differs: {scene}")
    summary = {
        "f4": {
            "counts": dict(f4_counts),
            "source_ids_sha256": source_ids_sha256,
            "source_lineage_sha256": source_lineage_sha256,
        },
        "f5": {
            "counts": dict(counts),
            "source_ids_sha256": source_ids_sha256,
            "source_lineage_sha256": source_lineage_sha256,
            "result_ledger_sha256": result_ledger_sha256,
        },
    }
    return selected, len(f5_frames), successful, summary


def evaluate_selected_threshold(
    *,
    scenes: Sequence[str],
    native_iou: Sequence[np.ndarray],
    selected_iou: Sequence[np.ndarray],
    selected_sources: Sequence[Sequence[SelectedSource]],
    gt_counts: Sequence[int],
    baseline_evaluation: Mapping[str, Any],
    threshold: float,
    f4_g4_additional_union_matches: int,
    official_eval_det: ModuleType,
) -> dict[str, Any]:
    if not (
        len(scenes) == len(native_iou) == len(selected_iou) == len(selected_sources) == len(gt_counts)
    ):
        raise F5EvaluationError("per-scene selected evaluation inputs differ")
    baseline_masks = baseline_evaluation.get("matched_gt_masks")
    if not isinstance(baseline_masks, list) or len(baseline_masks) != len(scenes):
        raise F5EvaluationError("official baseline unmatched-GT masks differ")
    totals = {"native": 0, "selected": 0, "union": 0, "suffix": 0}
    split = {
        name: {"selected_source_count": 0, "maximum_matching_count": 0, "union_matching_count": 0}
        for name in HYPOTHESES
    }
    hb = {"selected_source_count": 0, "matched_all_gt_count": 0, "matched_native_unmatched_gt_count": 0}
    suffix_matrices: list[np.ndarray] = []
    selections: dict[str, list[dict[str, Any]]] = {}
    for scene, native, selected, sources, gt_count, native_mask in zip(
        scenes, native_iou, selected_iou, selected_sources, gt_counts, baseline_masks, strict=True
    ):
        native_array = np.asarray(native, dtype=np.float64)
        selected_array = np.asarray(selected, dtype=np.float64)
        if native_array.ndim != 2 or native_array.shape[1] != gt_count or selected_array.shape != (len(sources), gt_count):
            raise F5EvaluationError(f"selected/native IoU shape differs: {scene}")
        native_pairs = strict_maximum_matching(native_array, threshold)
        selected_pairs = strict_maximum_matching(selected_array, threshold)
        union_pairs = strict_maximum_matching(np.concatenate((native_array, selected_array), axis=0), threshold)
        unmatched = ~np.asarray(native_mask, dtype=bool)
        suffix_pairs = strict_maximum_matching(selected_array, threshold, unmatched)
        suffix_indices = sorted(source_index for source_index, _ in suffix_pairs)
        suffix_matrices.append(
            selected_array[suffix_indices]
            if suffix_indices else np.empty((0, gt_count), dtype=np.float64)
        )
        selections[scene] = [
            {
                "source_id": sources[source_index].source_id,
                "source_index": source_index,
                "selected_hypothesis": sources[source_index].selected_hypothesis,
                "target_gt_index": gt_index,
                "target_iou": float(selected_array[source_index, gt_index]),
            }
            for source_index, gt_index in sorted(suffix_pairs)
        ]
        totals["native"] += len(native_pairs)
        totals["selected"] += len(selected_pairs)
        totals["union"] += len(union_pairs)
        totals["suffix"] += len(suffix_pairs)
        for name in HYPOTHESES:
            indices = [index for index, source in enumerate(sources) if source.selected_hypothesis == name]
            matrix = selected_array[indices] if indices else np.empty((0, gt_count), dtype=np.float64)
            split[name]["selected_source_count"] += len(indices)
            split[name]["maximum_matching_count"] += len(strict_maximum_matching(matrix, threshold))
            split[name]["union_matching_count"] += len(
                strict_maximum_matching(np.concatenate((native_array, matrix), axis=0), threshold)
            )
            if name == "HB":
                hb["selected_source_count"] += len(indices)
                hb["matched_all_gt_count"] += len(strict_maximum_matching(matrix, threshold))
                hb["matched_native_unmatched_gt_count"] += len(
                    strict_maximum_matching(matrix, threshold, unmatched)
                )
    combined = [
        np.concatenate((native, suffix), axis=0)
        for native, suffix in zip(native_iou, suffix_matrices, strict=True)
    ]
    suffix_evaluation = _authenticated_official_constant_evaluate(
        combined, gt_counts, threshold, official_eval_det
    )
    baseline_ap = float(baseline_evaluation["ap_points"])
    additional = totals["union"] - totals["native"]
    if f4_g4_additional_union_matches < 0:
        raise F5EvaluationError("historical F4 G4 capacity is negative")
    if additional > f4_g4_additional_union_matches:
        raise F5EvaluationError("F5 selected capacity exceeds its sealed F4 G4 universe")
    retention = (
        1.0 if f4_g4_additional_union_matches == 0 and additional == 0
        else 0.0 if f4_g4_additional_union_matches == 0
        else additional / f4_g4_additional_union_matches
    )
    for name in HYPOTHESES:
        split[name]["additional_union_matching_over_native"] = (
            split[name]["union_matching_count"] - totals["native"]
        )
    hb_count = hb["selected_source_count"]
    hb["match_fraction_all_gt"] = hb["matched_all_gt_count"] / hb_count if hb_count else 0.0
    hb["match_fraction_native_unmatched_gt"] = hb["matched_native_unmatched_gt_count"] / hb_count if hb_count else 0.0
    delta_ap = float(suffix_evaluation["ap_points"]) - baseline_ap
    return {
        "iou_threshold": threshold,
        "strict_iou_comparison": ">",
        "baseline_official_constant_score": _json_evaluation(baseline_evaluation, scenes),
        "native_maximum_matching_count": totals["native"],
        "selected_maximum_matching_count": totals["selected"],
        "union_maximum_matching_count": totals["union"],
        "additional_union_matching_over_native": additional,
        "selected_hypothesis_split": split,
        "hb_selected_matching": hb,
        "f4_g4_capacity": {
            "additional_union_matching_over_native": f4_g4_additional_union_matches,
            "f5_retained_additional_union_matches": additional,
            "retained_fraction": retention,
        },
        "gt_selected_constructive_suffix": {
            "oracle_only": True,
            "deployable": False,
            "formal_score": 1.0,
            "native_prefix_unchanged": True,
            "one_selected_geometry_per_source": True,
            "selected_source_count": totals["suffix"],
            "official_evaluation": _json_evaluation(suffix_evaluation, scenes),
            "delta_ap_points": delta_ap,
            "per_scene_selection": selections,
        },
    }


def f5_decision(
    *, per_threshold: Mapping[str, Any], no_gt_merge_passed: bool, baseline_passed: bool
) -> dict[str, Any]:
    if type(no_gt_merge_passed) is not bool or type(baseline_passed) is not bool:
        raise F5EvaluationError("decision prerequisites must be booleans")
    rows = [per_threshold[_threshold_key(threshold)] for threshold in THRESHOLDS]
    match_pass = all(row["additional_union_matching_over_native"] >= REQUIRED_ADDITIONAL_MATCHES for row in rows)
    ap_pass = all(row["gt_selected_constructive_suffix"]["delta_ap_points"] >= TARGET_DELTA_AP_POINTS for row in rows)
    passed = no_gt_merge_passed and baseline_passed and match_pass and ap_pass
    return {
        "no_gt_f5_merge_passed": no_gt_merge_passed,
        "native_baseline_reproduction_passed": baseline_passed,
        "required_additional_union_matches_each_threshold": REQUIRED_ADDITIONAL_MATCHES,
        "target_delta_ap_points_each_threshold": TARGET_DELTA_AP_POINTS,
        "selected_geometry_capacity_passes_all_thresholds": match_pass,
        "constructive_suffix_plus10_passes_all_thresholds": ap_pass,
        "overall_pass": passed,
        "active_birth_authorized": False,
        "result": (
            "retain_f5_authorize_new_preregistered_birth_confirmation_shadow_only"
            if passed else "discard_f5_geometry_selector_for_plus10_route"
        ),
    }


def _selector_snapshot(
    *,
    scenes: Sequence[str],
    fixed: Mapping[str, Path],
    f4_sidecar_root: Path,
    f5_sidecar_root: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {"fixed": {}, "scenes": {}}
    for name, path in fixed.items():
        source = _regular_file(path, name)
        result["fixed"][name] = {"path": os.fspath(source), "sha256": _sha256(source)}
    for scene in scenes:
        paths = {
            "f4_sidecar": f4_sidecar_root / f"{scene}.json",
            "f5_sidecar": f5_sidecar_root / f"{scene}.json",
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
        raise F5EvaluationError("selector snapshot is malformed")
    result: dict[str, Any] = {
        "fixed": {name: dict(seal) for name, seal in selector_fixed.items()},
        "scenes": {},
    }
    for name, path in fixed.items():
        if name in result["fixed"]:
            raise F5EvaluationError(f"evaluation fixed input collides with selector seal: {name}")
        source = _regular_file(path, name)
        result["fixed"][name] = {"path": os.fspath(source), "sha256": _sha256(source)}
    for scene in scenes:
        selector_scene = selector_scenes.get(scene)
        if not isinstance(selector_scene, Mapping):
            raise F5EvaluationError(f"selector snapshot lacks scene: {scene}")
        paths = {
            "native": baseline_root / f"{scene}_boxes.pkl",
            "gt": gt_root / f"{scene}_bbox.npy",
            "alignment": scan_root / scene / f"{scene}.txt",
        }
        result["scenes"][scene] = {name: dict(seal) for name, seal in selector_scene.items()} | {
            name: {"path": os.fspath(_regular_file(path, f"{scene} {name}")), "sha256": _sha256(path)}
            for name, path in paths.items()
        }
    return result


def audit_scannet_fastsam_f5_selector_paper100(
    *,
    scene_list: Path,
    evaluation_protocol: Path,
    f4_receipt: Path,
    f4_sidecar_root: Path,
    f5_receipt: Path,
    f5_sidecar_root: Path,
    f4_report: Path,
    baseline_root: Path,
    gt_root: Path,
    scan_root: Path,
    official_evaluator: Path,
) -> dict[str, Any]:
    # Phase 1: authenticate the complete no-GT selector before opening any
    # protected evaluation input or historical matching report.
    protocol_path = _regular_file(evaluation_protocol, "frozen F5 evaluation protocol", ".md")
    if _sha256(protocol_path) != EVALUATION_PROTOCOL_SHA256:
        raise F5EvaluationError("frozen F5 evaluation protocol hash differs")
    scene_list_path = _require_frozen_file(
        scene_list, "paper100 scene list", EXPECTED_SCENE_LIST_SHA256
    )
    scenes = load_scene_list(scene_list_path)
    if _sha256(scene_list_path) != EXPECTED_SCENE_LIST_SHA256:
        raise F5EvaluationError("paper100 scene list changed while it was read")
    if len(scenes) != EXPECTED["scene_count"] or len(set(scenes)) != len(scenes):
        raise F5EvaluationError("paper100 scene count/order differs")
    selector_fixed = {
        "scene_list": scene_list_path,
        "evaluation_protocol": protocol_path,
        "f4_receipt": f4_receipt,
        "f5_receipt": f5_receipt,
        "evaluator_source": Path(__file__).resolve(),
    }
    selector_before = _selector_snapshot(
        scenes=scenes,
        fixed=selector_fixed,
        f4_sidecar_root=f4_sidecar_root,
        f5_sidecar_root=f5_sidecar_root,
    )
    expected_selector_hashes = {
        "scene_list": EXPECTED_SCENE_LIST_SHA256,
        "evaluation_protocol": EVALUATION_PROTOCOL_SHA256,
        "f4_receipt": EXPECTED_F4_RECEIPT_SHA256,
        "f5_receipt": EXPECTED_F5_RECEIPT_SHA256,
    }
    for name, expected_sha256 in expected_selector_hashes.items():
        if selector_before["fixed"].get(name, {}).get("sha256") != expected_sha256:
            raise F5EvaluationError(f"frozen selector input differs: {name}")

    f4_receipt_payload = _read_pinned_json(
        f4_receipt, "F4 merge", EXPECTED_F4_RECEIPT_SHA256
    )
    f5_receipt_payload = _read_pinned_json(
        f5_receipt, "F5 merge", EXPECTED_F5_RECEIPT_SHA256
    )
    f4_rows = _validate_merge(
        f4_receipt_payload, schema=F4_MERGE_SCHEMA, protocol_id=F4_PROTOCOL_ID,
        scenes=scenes, label="F4",
    )
    f5_rows = _validate_merge(
        f5_receipt_payload, schema=F5_MERGE_SCHEMA, protocol_id=F5_PROTOCOL_ID,
        scenes=scenes, label="F5",
    )
    prevalidated: dict[str, list[SelectedSource]] = {}
    counts = {"keyframe_count": 0, "successful_frame_count": 0, "source_count": 0}
    global_source_ids: list[str] = []
    global_lineages: list[str] = []
    global_results: list[str] = []
    pre_gt_hypothesis_counts = {name: 0 for name in HYPOTHESES}
    for scene_index, scene in enumerate(scenes):
        f4_path, f4_sha = _seal_from_row(f4_rows[scene], scene)
        f5_path, f5_sha = _seal_from_row(f5_rows[scene], scene)
        if f4_path.resolve() != (f4_sidecar_root / f"{scene}.json").resolve() or f5_path.resolve() != (f5_sidecar_root / f"{scene}.json").resolve():
            raise F5EvaluationError(f"F4/F5 sidecar root differs: {scene}")
        if (
            selector_before["scenes"][scene]["f4_sidecar"]["sha256"] != f4_sha
            or selector_before["scenes"][scene]["f5_sidecar"]["sha256"] != f5_sha
        ):
            raise F5EvaluationError(f"F4/F5 selector snapshot differs from merge seal: {scene}")
        sources, keyframes, successful, scene_summary = _load_scene_sources_pre_gt(
            scene=scene, scene_index=scene_index,
            f4_path=f4_path, f4_sha=f4_sha, f5_path=f5_path, f5_sha=f5_sha,
        )
        for label, merge_row, keys in (
            ("F4", f4_rows[scene], ("counts", "source_ids_sha256", "source_lineage_sha256")),
            ("F5", f5_rows[scene], ("counts", "source_ids_sha256", "source_lineage_sha256", "result_ledger_sha256")),
        ):
            for key in keys:
                if merge_row.get(key) != scene_summary[label.lower()][key]:
                    raise F5EvaluationError(
                        f"{label} merge/sidecar scene ledger differs: {scene}/{key}"
                    )
        prevalidated[scene] = sources
        counts["keyframe_count"] += keyframes
        counts["successful_frame_count"] += successful
        counts["source_count"] += len(sources)
        global_source_ids.extend(source.source_id for source in sources)
        global_lineages.extend(source.source_lineage_sha256 for source in sources)
        global_results.extend(source.result_sha256 for source in sources)
        for source in sources:
            pre_gt_hypothesis_counts[source.selected_hypothesis] += 1
    if counts != {key: EXPECTED[key] for key in counts}:
        raise F5EvaluationError(f"pre-GT F5 census differs: {counts}")
    if len(set(global_source_ids)) != EXPECTED["source_count"]:
        raise F5EvaluationError("F5 global source identities are not unique")
    global_ledgers = {
        "source_ids_sha256": _canonical_json_sha256(global_source_ids),
        "source_lineage_sha256": _canonical_json_sha256(global_lineages),
        "result_ledger_sha256": _canonical_json_sha256(global_results),
    }
    expected_global_ledgers = {
        "source_ids_sha256": EXPECTED_SOURCE_IDS_SHA256,
        "source_lineage_sha256": EXPECTED_SOURCE_LINEAGE_SHA256,
        "result_ledger_sha256": EXPECTED_RESULT_LEDGER_SHA256,
    }
    f4_coverage = f4_receipt_payload["coverage"]
    f5_coverage = f5_receipt_payload["coverage"]
    for name, actual_sha256 in global_ledgers.items():
        if actual_sha256 != expected_global_ledgers[name] or f5_coverage.get(name) != actual_sha256:
            raise F5EvaluationError(f"F5 global {name} differs")
        if name != "result_ledger_sha256" and f4_coverage.get(name) != actual_sha256:
            raise F5EvaluationError(f"F4/F5 global {name} differs")
    selection = f5_receipt_payload["selection"]
    if any(
        selection.get(f"selected_{name.lower()}_count") != count
        for name, count in pre_gt_hypothesis_counts.items()
    ):
        raise F5EvaluationError("F5 selected-hypothesis census differs")

    # Phase 2: the complete selector is sealed and may now be evaluated once.
    selector_after_prevalidation = _selector_snapshot(
        scenes=scenes,
        fixed=selector_fixed,
        f4_sidecar_root=f4_sidecar_root,
        f5_sidecar_root=f5_sidecar_root,
    )
    if selector_before != selector_after_prevalidation:
        raise F5EvaluationError("selector inputs changed during no-GT prevalidation")
    official_paths = _official_dependency_paths(official_evaluator)
    evaluation_fixed = {
        "f4_report": f4_report,
        "official_evaluator": official_paths["official_evaluator"],
        "official_eval_det": official_paths["official_eval_det"],
        "official_metric_util": official_paths["official_metric_util"],
        "official_box_util": official_paths["official_box_util"],
        "oracle_helper_source": ORACLE_HELPER_SOURCE,
    }
    before = _evaluation_snapshot(
        scenes=scenes, fixed=evaluation_fixed, selector_snapshot=selector_before,
        baseline_root=baseline_root,
        gt_root=gt_root, scan_root=scan_root,
    )
    expected_fixed_hashes = {
        **expected_selector_hashes,
        "f4_report": EXPECTED_F4_REPORT_SHA256,
        "official_evaluator": EXPECTED_OFFICIAL_EVALUATOR_SHA256,
        "official_eval_det": EXPECTED_OFFICIAL_EVAL_DET_SHA256,
        "official_metric_util": EXPECTED_OFFICIAL_METRIC_UTIL_SHA256,
        "official_box_util": EXPECTED_OFFICIAL_BOX_UTIL_SHA256,
        "oracle_helper_source": EXPECTED_ORACLE_HELPER_SHA256,
    }
    for name, expected_sha256 in expected_fixed_hashes.items():
        if before["fixed"].get(name, {}).get("sha256") != expected_sha256:
            raise F5EvaluationError(f"frozen evaluation input differs: {name}")
    f4_report_payload = _read_pinned_json(
        f4_report, "historical F4 report", EXPECTED_F4_REPORT_SHA256
    )
    if (
        f4_report_payload.get("schema") != F4_REPORT_SCHEMA
        or f4_report_payload.get("scene_order") != list(scenes)
        or f4_report_payload.get("integrity", {}).get("all_inputs_before_after_identity") is not True
    ):
        raise F5EvaluationError("historical F4 report schema differs")
    _validate_historical_f4_input_lineage(
        f4_report_payload, snapshot=before, scenes=scenes
    )
    official_eval_det = _load_official_eval_det(official_paths)
    gt_counts: list[int] = []
    native_iou: list[np.ndarray] = []
    selected_iou: list[np.ndarray] = []
    selected_by_scene: list[list[SelectedSource]] = []
    totals = {**counts, "scene_count": len(scenes), "native_count": 0, "gt_count": 0}
    hypothesis_counts = {name: 0 for name in HYPOTHESES}
    for scene in scenes:
        scene_snapshot = before["scenes"][scene]
        alignment_path = scan_root / scene / f"{scene}.txt"
        gt_path = gt_root / f"{scene}_bbox.npy"
        native_path = baseline_root / f"{scene}_boxes.pkl"
        for name, path in (("alignment", alignment_path), ("gt", gt_path), ("native", native_path)):
            _require_frozen_file(path, f"{scene} {name}", scene_snapshot[name]["sha256"])
        alignment = load_axis_alignment(alignment_path)
        gt = load_gt_minmax(gt_path)
        _, native = load_baseline_boxes(native_path, alignment)
        for name, path in (("alignment", alignment_path), ("gt", gt_path), ("native", native_path)):
            _require_frozen_file(path, f"{scene} {name}", scene_snapshot[name]["sha256"])
        aligned_sources: list[SelectedSource] = []
        boxes: list[np.ndarray] = []
        for source in prevalidated[scene]:
            aligned = _align_corners(source.world_corners, alignment, f"{source.source_id}.selected_corners")
            boxes.append(aligned)
            hypothesis_counts[source.selected_hypothesis] += 1
            aligned_sources.append(
                SelectedSource(**{**source.__dict__, "aligned_minmax": aligned})
            )
        selected_boxes = np.stack(boxes) if boxes else np.empty((0, 6), dtype=np.float64)
        gt_counts.append(len(gt))
        native_iou.append(aligned_iou_matrix(native, gt))
        selected_iou.append(aligned_iou_matrix(selected_boxes, gt))
        selected_by_scene.append(aligned_sources)
        totals["native_count"] += len(native)
        totals["gt_count"] += len(gt)
    if totals["native_count"] != EXPECTED["native_count"] or totals["gt_count"] != EXPECTED["gt_count"]:
        raise F5EvaluationError(f"native/GT paper100 census differs: {totals}")
    baseline = {
        threshold: _authenticated_official_constant_evaluate(
            native_iou, gt_counts, threshold, official_eval_det
        )
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
            "passed": math.isclose(actual_ap, expected_ap, rel_tol=0.0, abs_tol=1e-9),
        }
        try:
            f4_threshold = f4_report_payload["per_threshold"][key]
            f4_g4 = int(f4_threshold["identity_constrained_g4"]["additional_union_matching_over_native"])
            f4_baseline_ap = float(f4_threshold["baseline_official_constant_score"]["ap_points"])
        except (KeyError, TypeError, ValueError) as error:
            raise F5EvaluationError(f"historical F4 aggregate is absent at {key}") from error
        if not math.isclose(f4_baseline_ap, actual_ap, rel_tol=0.0, abs_tol=1e-12):
            raise F5EvaluationError(f"historical F4 baseline reproduction differs at {key}")
        per_threshold[key] = evaluate_selected_threshold(
            scenes=scenes, native_iou=native_iou, selected_iou=selected_iou,
            selected_sources=selected_by_scene, gt_counts=gt_counts,
            baseline_evaluation=baseline[threshold], threshold=threshold,
            f4_g4_additional_union_matches=f4_g4,
            official_eval_det=official_eval_det,
        )
    baseline_passed = all(row["passed"] for row in baseline_checks.values())
    decision = f5_decision(
        per_threshold=per_threshold,
        no_gt_merge_passed=True,
        baseline_passed=baseline_passed,
    )
    after = _evaluation_snapshot(
        scenes=scenes, fixed=evaluation_fixed, selector_snapshot=selector_before,
        baseline_root=baseline_root,
        gt_root=gt_root, scan_root=scan_root,
    )
    selector_after_evaluation = _selector_snapshot(
        scenes=scenes,
        fixed=selector_fixed,
        f4_sidecar_root=f4_sidecar_root,
        f5_sidecar_root=f5_sidecar_root,
    )
    if before != after or selector_before != selector_after_evaluation:
        raise F5EvaluationError("one or more sealed inputs changed during F5 evaluation")
    return {
        "schema": SCHEMA,
        "evaluation_protocol_id": EVALUATION_PROTOCOL_ID,
        "oracle_only": True,
        "deployable": False,
        "gt_used": True,
        "active_birth_authorized": False,
        "one_selected_geometry_per_source": True,
        "formal_score": 1.0,
        "strict_iou_comparison": ">",
        "scene_order": scenes,
        "totals": totals,
        "selected_hypothesis_counts": hypothesis_counts,
        "global_source_ledgers": global_ledgers,
        "authenticated_official_evaluator": {
            "entrypoint_sha256": EXPECTED_OFFICIAL_EVALUATOR_SHA256,
            "eval_det_sha256": EXPECTED_OFFICIAL_EVAL_DET_SHA256,
            "matrix_lookup_supplied_explicitly": True,
            "baseline_and_suffix_crosschecked": True,
        },
        "no_gt_f5_merge": {
            "integrity_passed": True,
            "causality_passed": True,
            "determinism_passed": True,
            "runtime_passed": True,
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
        raise F5EvaluationError("F5 evaluation output must use .json")
    if output.exists() or output.is_symlink():
        raise F5EvaluationError(f"refusing to overwrite F5 evaluation output: {output}")
    if any(_is_within(output, root) for root in protected_roots):
        raise F5EvaluationError("F5 evaluation output lies inside a protected input root")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, default=DEFAULT_SCENE_LIST)
    parser.add_argument("--evaluation-protocol", type=Path, default=DEFAULT_EVALUATION_PROTOCOL)
    parser.add_argument("--f4-receipt", type=Path, default=DEFAULT_F4_RECEIPT)
    parser.add_argument("--f4-sidecar-root", type=Path, default=DEFAULT_F4_SIDECARS)
    parser.add_argument("--f5-receipt", type=Path, default=DEFAULT_F5_RECEIPT)
    parser.add_argument("--f5-sidecar-root", type=Path, default=DEFAULT_F5_SIDECARS)
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
            args.scene_list.parent, args.evaluation_protocol.parent,
            args.f4_receipt.parent, args.f4_sidecar_root,
            args.f5_receipt.parent, args.f5_sidecar_root,
            args.f4_report.parent, args.baseline_root, args.gt_root,
            args.scan_root, args.official_evaluator.parent,
        ),
    )
    report = audit_scannet_fastsam_f5_selector_paper100(
        scene_list=args.scene_list,
        evaluation_protocol=args.evaluation_protocol,
        f4_receipt=args.f4_receipt,
        f4_sidecar_root=args.f4_sidecar_root,
        f5_receipt=args.f5_receipt,
        f5_sidecar_root=args.f5_sidecar_root,
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
