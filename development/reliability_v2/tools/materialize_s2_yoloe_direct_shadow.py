#!/usr/bin/env python3
"""Seal S2 YOLOE-direct candidates as a no-GT terminal counterfactual.

The producer diagnostics are deliberately treated as an already-ranked,
observer-only proposal stream.  This materializer only:

* selects diagnostic rows whose ``source_indices`` value is ``-1``;
* rejects candidates with native T05 AABB IoU >= 0.10;
* applies stable, input-order candidate NMS at AABB IoU >= 0.25; and
* appends at most six candidates after the unchanged native row prefix.

No ground-truth, oracle, label, CLIP, or training input exists in the API.
Stored scores are diagnostic provenance only.  The preregistered evaluation
contract assigns every native and appended row the constant score ``1.0``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(_REPOSITORY_ROOT))

from tools.materialize_boxer_past3_shadow import (  # noqa: E402
    _array_content_sha256,
    _write_deterministic_npz,
    _write_json_exclusive,
)


SCHEMA = "boxfusion.s2_yoloe_direct_shadow.v1"
AUDIT_FILENAME = "s2_yoloe_direct_shadow.json"
ARRAY_FILENAME = "s2_yoloe_direct_shadow.npz"
PREDICTION_SUFFIX = "_boxes.pkl"
DIAGNOSTIC_SUFFIX = "_tracks.npz"

DEV3_SCENES = ("scene0568_00", "scene0606_01", "scene0377_02")
SCENE_PATTERN = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")

NATIVE_NOVELTY_IOU = 0.10
SELF_NMS_IOU = 0.25
MAX_OUTPUTS_PER_SCENE = 6
FORMAL_EVALUATION_SCORE = 1.0

FROZEN_CONFIG_PATH = (
    _REPOSITORY_ROOT / "config" / "scannet_s2_yoloe_direct_shadow_score05.yaml"
)
FROZEN_CHECKPOINT_PATH = Path(
    "/data/ZhaoX/OVM3D-Dett/boxfusion_b6_selective_boxer_dev/models/"
    "yoloe-11s-seg-pf.pt"
)
FROZEN_PREREGISTRATION_SHA256 = (
    "fc737deec401de54845a0b7d8cb1152443203d55eef335b0ca99270396f663f3"
)
FROZEN_HASHES: Mapping[str, tuple[Path, str]] = {
    "config": (
        FROZEN_CONFIG_PATH,
        "4f3e9739b296197d41c0d322c0a1e30230385ccb8c1384a36615ffa413e83441",
    ),
    "yoloe_checkpoint": (
        FROZEN_CHECKPOINT_PATH,
        "292bdf157a9ec7315f34b567cb93467c5043cd1889a1cc18abbfdeb88d7a948d",
    ),
    "demo_source": (
        _REPOSITORY_ROOT / "tools" / "boxfusion_tr3d_pipeline" / "demo.py",
        "57fb58596401324785ee9696d16ebc15eed082df00dd6afede9e6d440b217423",
    ),
    "online_refinement_source": (
        _REPOSITORY_ROOT
        / "tools"
        / "boxfusion_tr3d_pipeline"
        / "boxfusion"
        / "online_refinement.py",
        "0faf3d7d6242facdd9300a942fe1e2bf2364f5f9ebc17e8f8f278382a0102f61",
    ),
    "object_memory_source": (
        _REPOSITORY_ROOT
        / "tools"
        / "boxfusion_tr3d_pipeline"
        / "boxfusion"
        / "object_memory.py",
        "c2f3f0e0753a34430f0d9d03c65039aa6eee80114a1337676ec4b5f1eaa60938",
    ),
    "supplemental_proposals_source": (
        _REPOSITORY_ROOT
        / "tools"
        / "boxfusion_tr3d_pipeline"
        / "boxfusion"
        / "supplemental_proposals.py",
        "dcab601eb7bd70328be882e8944619e4dffd6d366214dd74eb6c2d5a3cfc001d",
    ),
    "tr3d_c2_observer_source": (
        _REPOSITORY_ROOT
        / "tools"
        / "boxfusion_tr3d_pipeline"
        / "boxfusion"
        / "tr3d_c2_maskrgbd_observer.py",
        "108e4c1684a6f5e3b352b31a9d6e026e393bc1872540653312e7bdfb0d1e4778",
    ),
}

_EXPECTED_DIAGNOSTIC_ARRAYS = {
    "boxes",
    "labels",
    "point_mask",
    "points",
    "quality_feature_names",
    "quality_features",
    "result_indices",
    "scene_id",
    "scores",
    "source_indices",
    "summary_json",
    "track_ids",
}

_QUALITY_FEATURE_NAMES = (
    "detector_score",
    "mask_confidence",
    "valid_depth_ratio",
    "depth_support",
    "projection_iou",
    "geometry_consistency",
    "appearance_consistency",
    "view_count_quality",
    "box_stability",
    "source_agreement",
    "area_quality",
    "refiner_quality",
)

_EXPECTED_SUMMARY_KEYS = {
    "active_supplemental_tracks",
    "appearance_seconds",
    "archived_supplemental_tracks",
    "candidate_archived_total",
    "candidate_discarded_total",
    "candidate_ttl_clock",
    "candidate_updates",
    "confirmed_supplemental_tracks",
    "enabled",
    "geometry_seconds",
    "global_memories",
    "keyframes",
    "lifted",
    "matched_global",
    "neural_refits_accepted",
    "proposals",
    "provider_calls",
    "provider_seconds",
    "refit_rejections",
    "refits_accepted",
    "refits_attempted",
    "supplemental_considered",
    "supplemental_deduplicated",
    "supplemental_output",
    "supplemental_rejected_extent",
    "supplemental_rejected_global",
    "supplemental_rejected_projection",
    "supplemental_rejected_score",
}


class S2ShadowError(ValueError):
    """Raised when an S2 frozen-input or no-GT contract is violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise S2ShadowError(f"{label} must be a regular non-symlink file: {path}")
    return path


def _nested(mapping: Mapping[str, Any], dotted: str) -> Any:
    value: Any = mapping
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise S2ShadowError(f"frozen S2 config is missing {dotted}")
        value = value[part]
    return value


def _validate_config(config_path: Path) -> dict[str, Any]:
    config_path = _regular_file(config_path, "frozen S2 config")
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise S2ShadowError(f"invalid frozen S2 config: {config_path}") from error
    if not isinstance(config, dict):
        raise S2ShadowError("frozen S2 config must decode to a mapping")
    expected = {
        "dataset": "scannet",
        "data.gap": 25,
        "data.post_process_min_extent": 0.30,
        "detection.score_thresh": 0.5,
        "lifting.backend": "cutr",
        "lifting.proposal_cache.mode": "disabled",
        "association.appearance_gate.enabled": False,
        "box_fusion.reliable_views.enabled": True,
        "box_fusion.reliable_views.top_k": 3,
        "box_fusion.reliable_views.min_views": 3,
        "online_refinement.enabled": True,
        "online_refinement.inference_every_keyframes": 1,
        "online_refinement.candidate_lifecycle.ttl_clock": "provider_call",
        "online_refinement.appearance_memory.enabled": False,
        "online_refinement.supplemental_proposals.enabled": True,
        "online_refinement.supplemental_proposals.provider": "yoloe",
        "online_refinement.supplemental_proposals.mode": "prompt_free",
        "online_refinement.supplemental_proposals.prompts": [],
        "online_refinement.supplemental_proposals.confidence": 0.25,
        "online_refinement.supplemental_proposals.iou": 0.70,
        "online_refinement.supplemental_proposals.image_size": 640,
        "online_refinement.supplemental_proposals.max_detections": 64,
        "online_refinement.supplemental_proposals.mask_threshold": 0.50,
        "online_refinement.supplemental_proposals.agnostic_nms": True,
        "online_refinement.supplemental_proposals.cache.enabled": False,
        "online_refinement.supplemental_proposals.cache.write": False,
        "online_refinement.object_memory.enabled": True,
        "online_refinement.object_memory.voxel_size": 0.02,
        "online_refinement.object_memory.aabb_lower_quantile": 0.02,
        "online_refinement.object_memory.aabb_upper_quantile": 0.98,
        "online_refinement.object_memory.min_confirmations": 3,
        "online_refinement.object_memory.track_ttl": 10,
        "online_refinement.matching.global_match_iou": 1.0,
        "online_refinement.matching.global_match_2d_iou": 1.0,
        "online_refinement.matching.max_center_distance": 0.000001,
        "online_refinement.matching.rekey_iou": 1.0,
        "online_refinement.matching.absorb_supplemental_iou": 1.0,
        "online_refinement.refit.enabled": False,
        "online_refinement.box_refiner.enabled": False,
        "online_refinement.quality.enabled": False,
        "online_refinement.quality.soft_nms.enabled": False,
        "online_refinement.supplemental_output.enabled": True,
        "online_refinement.supplemental_output.min_confirmations": 3,
        "online_refinement.supplemental_output.min_score": 0.25,
        "online_refinement.supplemental_output.min_projection_iou": 0.30,
        "online_refinement.supplemental_output.drop_if_global_iou": 1.0,
        "online_refinement.supplemental_output.drop_if_supplemental_iou": 0.70,
        "online_refinement.output_filter.minimum_extent": 0.30,
        "online_refinement.diagnostics.enabled": True,
        "online_refinement.diagnostics.dump_track_memory": True,
        "online_refinement.diagnostics.point_count": 512,
        "eval": True,
    }
    for dotted, wanted in expected.items():
        actual = _nested(config, dotted)
        if actual != wanted or type(actual) is not type(wanted):
            # YAML parses 0.30 as float and True as bool, making the strict
            # type check useful against accidental integer/Boolean aliases.
            raise S2ShadowError(
                f"frozen S2 config mismatch for {dotted}: "
                f"expected={wanted!r}, actual={actual!r}"
            )
    checkpoint = Path(
        str(_nested(config, "online_refinement.supplemental_proposals.checkpoint"))
    ).resolve()
    if checkpoint != FROZEN_CHECKPOINT_PATH.resolve():
        raise S2ShadowError(f"unexpected YOLOE checkpoint path: {checkpoint}")
    return config


def _validate_frozen_hashes(config_path: Path) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for name, (registered_path, expected_hash) in FROZEN_HASHES.items():
        path = config_path if name == "config" else registered_path
        path = _regular_file(path, f"frozen {name}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise S2ShadowError(
                f"frozen {name} SHA-256 mismatch: "
                f"expected={expected_hash}, actual={actual_hash}"
            )
        records[name] = {
            "path": os.fspath(path),
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
        }
    return records


def _as_nonnegative_int(summary: Mapping[str, Any], key: str) -> int:
    value = summary.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise S2ShadowError(f"diagnostic summary {key} must be a nonnegative integer")
    return value


def _as_nonnegative_float(summary: Mapping[str, Any], key: str) -> float:
    value = summary.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise S2ShadowError(f"diagnostic summary {key} must be finite and nonnegative")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise S2ShadowError(f"diagnostic summary {key} must be finite and nonnegative")
    return result


def _validate_summary(summary: Any, supplemental_count: int) -> dict[str, Any]:
    if not isinstance(summary, dict) or set(summary) != _EXPECTED_SUMMARY_KEYS:
        raise S2ShadowError("unexpected YOLOE-direct diagnostic summary schema")
    if summary.get("enabled") is not True:
        raise S2ShadowError("YOLOE-direct diagnostic observer was not enabled")
    if summary.get("candidate_ttl_clock") != "provider_call":
        raise S2ShadowError("YOLOE-direct diagnostic TTL clock is not provider_call")
    if summary.get("refit_rejections") != {}:
        raise S2ShadowError("S2 diagnostic unexpectedly contains refit rejections")

    integer_keys = _EXPECTED_SUMMARY_KEYS - {
        "appearance_seconds",
        "candidate_ttl_clock",
        "enabled",
        "geometry_seconds",
        "provider_seconds",
        "refit_rejections",
    }
    integers = {key: _as_nonnegative_int(summary, key) for key in integer_keys}
    for key in ("appearance_seconds", "geometry_seconds", "provider_seconds"):
        _as_nonnegative_float(summary, key)

    if summary["appearance_seconds"] != 0.0:
        raise S2ShadowError("appearance computation must remain disabled")
    if integers["provider_calls"] != integers["keyframes"]:
        raise S2ShadowError("one frozen provider call per keyframe was not preserved")
    # A valid mask proposal can still have no liftable depth pixels.  Such a
    # row is counted by ``proposals`` but correctly absent from both ``lifted``
    # and ``candidate_updates``.
    if not (
        integers["candidate_updates"]
        == integers["lifted"]
        <= integers["proposals"]
    ):
        raise S2ShadowError("proposal/lift/update diagnostic counts disagree")
    if integers["matched_global"] != 0 or integers["global_memories"] != 0:
        raise S2ShadowError("S2 universal candidates were absorbed into native memory")
    if any(
        integers[key] != 0
        for key in ("refits_attempted", "refits_accepted", "neural_refits_accepted")
    ):
        raise S2ShadowError("S2 geometry/refiner path was not observer-only")
    if integers["candidate_archived_total"] != integers["archived_supplemental_tracks"]:
        raise S2ShadowError("candidate archive counts disagree")
    if integers["supplemental_considered"] > integers["confirmed_supplemental_tracks"]:
        raise S2ShadowError("more supplemental tracks were considered than confirmed")
    accounted = sum(
        integers[key]
        for key in (
            "supplemental_rejected_extent",
            "supplemental_rejected_score",
            "supplemental_rejected_projection",
            "supplemental_rejected_global",
            "supplemental_deduplicated",
            "supplemental_output",
        )
    )
    if accounted != integers["supplemental_considered"]:
        raise S2ShadowError("supplemental terminal accounting is not exact")
    if integers["supplemental_rejected_global"] != 0:
        raise S2ShadowError("native overlap was unexpectedly applied inside the observer")
    if integers["supplemental_output"] != supplemental_count:
        raise S2ShadowError("diagnostic rows disagree with supplemental_output")
    return dict(summary)


def _load_diagnostic(path: Path, scene: str) -> dict[str, Any]:
    path = _regular_file(path, f"S2 diagnostic for {scene}")
    try:
        with np.load(path, allow_pickle=False) as source:
            if set(source.files) != _EXPECTED_DIAGNOSTIC_ARRAYS:
                raise S2ShadowError(f"unexpected diagnostic NPZ schema for {scene}")
            arrays = {name: np.array(source[name], copy=True) for name in source.files}
    except (OSError, ValueError) as error:
        if isinstance(error, S2ShadowError):
            raise
        raise S2ShadowError(f"invalid S2 diagnostic NPZ for {scene}: {path}") from error

    if arrays["scene_id"].shape != () or str(arrays["scene_id"].item()) != scene:
        raise S2ShadowError(f"diagnostic scene ID mismatch for {scene}")
    count = len(arrays["scores"])
    shapes = {
        "boxes": (count, 6),
        "scores": (count,),
        "quality_features": (count, len(_QUALITY_FEATURE_NAMES)),
        "points": (count, 512, 3),
        "point_mask": (count, 512),
        "source_indices": (count,),
        "track_ids": (count,),
        "result_indices": (count,),
        "labels": (count,),
        "quality_feature_names": (len(_QUALITY_FEATURE_NAMES),),
    }
    for name, expected_shape in shapes.items():
        if arrays[name].shape != expected_shape:
            raise S2ShadowError(
                f"diagnostic {name} shape mismatch for {scene}: {arrays[name].shape}"
            )
    if arrays["summary_json"].shape != ():
        raise S2ShadowError(f"diagnostic summary_json must be scalar for {scene}")
    if arrays["quality_feature_names"].tolist() != list(_QUALITY_FEATURE_NAMES):
        raise S2ShadowError(f"diagnostic quality feature schema changed for {scene}")
    for name in ("source_indices", "track_ids", "result_indices"):
        if arrays[name].dtype.kind not in "iu":
            raise S2ShadowError(f"diagnostic {name} must be integer for {scene}")
    if arrays["point_mask"].dtype.kind != "b":
        raise S2ShadowError(f"diagnostic point_mask must be Boolean for {scene}")
    if arrays["labels"].dtype.kind not in "US":
        raise S2ShadowError(f"diagnostic labels must be inert strings for {scene}")
    numeric = np.concatenate(
        (
            arrays["boxes"].reshape(-1),
            arrays["scores"].reshape(-1),
            arrays["quality_features"].reshape(-1),
            arrays["points"][arrays["point_mask"]].reshape(-1),
        )
    )
    if not np.isfinite(numeric).all():
        raise S2ShadowError(f"diagnostic arrays contain non-finite values for {scene}")
    if np.any(arrays["boxes"][:, 3:6] < 0.30):
        raise S2ShadowError(f"diagnostic output bypassed the 0.30 m extent gate for {scene}")
    if np.any((arrays["scores"] < 0.25) | (arrays["scores"] > 1.0)):
        raise S2ShadowError(f"diagnostic score bypassed the frozen score gate for {scene}")
    if len(np.unique(arrays["result_indices"])) != count:
        raise S2ShadowError(f"diagnostic result indices are not unique for {scene}")
    if count and np.any(np.diff(arrays["result_indices"]) <= 0):
        raise S2ShadowError(f"diagnostic rows are not in result order for {scene}")

    supplemental_positions = np.flatnonzero(arrays["source_indices"] == -1)
    if np.any(arrays["source_indices"] < -1):
        raise S2ShadowError(f"diagnostic source index below -1 for {scene}")
    supplemental_track_ids = arrays["track_ids"][supplemental_positions]
    if np.any(supplemental_track_ids >= 0) or len(np.unique(supplemental_track_ids)) != len(
        supplemental_track_ids
    ):
        raise S2ShadowError(f"supplemental track IDs are not unique negative IDs for {scene}")
    supplemental_scores = arrays["scores"][supplemental_positions]
    if len(supplemental_scores) > 1 and np.any(np.diff(supplemental_scores) > 1e-7):
        raise S2ShadowError(f"supplemental diagnostic ranking changed for {scene}")
    if len(supplemental_positions) and not np.allclose(
        arrays["quality_features"][supplemental_positions, 0],
        supplemental_scores,
        rtol=0.0,
        atol=1e-6,
    ):
        raise S2ShadowError(f"quality-disabled detector score identity failed for {scene}")
    try:
        summary = json.loads(str(arrays["summary_json"].item()))
    except json.JSONDecodeError as error:
        raise S2ShadowError(f"diagnostic summary is not JSON for {scene}") from error
    validated_summary = _validate_summary(summary, len(supplemental_positions))
    for value in arrays.values():
        value.setflags(write=False)
    return {
        "arrays": arrays,
        "summary": validated_summary,
        "supplemental_positions": supplemental_positions,
    }


def _load_native_payload(path: Path) -> tuple[Any, list[Any], np.ndarray, np.ndarray]:
    path = _regular_file(path, "native T05 prediction")
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as error:
        raise S2ShadowError(f"could not load native T05 prediction: {path}") from error
    if not isinstance(payload, (list, tuple)) or len(payload) != 1:
        raise S2ShadowError(f"invalid native prediction outer schema: {path}")
    rows = payload[0]
    if not isinstance(rows, (list, tuple)):
        raise S2ShadowError(f"invalid native prediction row container: {path}")
    corners: list[np.ndarray] = []
    scores: list[float] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise S2ShadowError(f"invalid native row {row_index}: {path}")
        box = np.asarray(row[1])
        try:
            score = float(row[2])
        except (TypeError, ValueError) as error:
            raise S2ShadowError(f"invalid native score at row {row_index}: {path}") from error
        if box.shape != (8, 3) or not np.issubdtype(box.dtype, np.number):
            raise S2ShadowError(f"invalid native corners at row {row_index}: {path}")
        if not np.isfinite(box).all() or not math.isfinite(score) or not 0.0 < score <= 1.0:
            raise S2ShadowError(f"non-finite native row {row_index}: {path}")
        corners.append(np.asarray(box, dtype=np.float64))
        scores.append(score)
    native_corners = (
        np.stack(corners)
        if corners
        else np.empty((0, 8, 3), dtype=np.float64)
    )
    return payload, list(rows), native_corners, np.asarray(scores, dtype=np.float64)


def _aabb_corners(box: np.ndarray) -> np.ndarray:
    box = np.asarray(box, dtype=np.float64)
    if box.shape != (6,) or not np.isfinite(box).all() or np.any(box[3:] <= 0.0):
        raise S2ShadowError("candidate center/extent box is invalid")
    lower = box[:3] - box[3:] / 2.0
    upper = box[:3] + box[3:] / 2.0
    return np.asarray(
        [
            [x, y, z]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ],
        dtype=np.float64,
    )


def _aabb_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_min, left_max = left.min(axis=0), left.max(axis=0)
    right_min, right_max = right.min(axis=0), right.max(axis=0)
    intersection = float(
        np.prod(np.maximum(np.minimum(left_max, right_max) - np.maximum(left_min, right_min), 0.0))
    )
    left_volume = float(np.prod(np.maximum(left_max - left_min, 0.0)))
    right_volume = float(np.prod(np.maximum(right_max - right_min, 0.0)))
    union = left_volume + right_volume - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _terminal_filter(
    rows: Sequence[dict[str, Any]], native_corners: np.ndarray
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    accepted: list[dict[str, Any]] = []
    native_rejected: list[int] = []
    nms_rejected: list[int] = []
    cap_rejected: list[int] = []
    for row in rows:
        item = dict(row)
        maximum_native_iou = max(
            (_aabb_iou(item["corners"], native) for native in native_corners),
            default=0.0,
        )
        item["max_native_aabb_iou"] = maximum_native_iou
        input_row = int(item["diagnostic_row"])
        if maximum_native_iou >= NATIVE_NOVELTY_IOU:
            native_rejected.append(input_row)
            continue
        if any(
            _aabb_iou(item["corners"], previous["corners"]) >= SELF_NMS_IOU
            for previous in accepted
        ):
            nms_rejected.append(input_row)
            continue
        if len(accepted) >= MAX_OUTPUTS_PER_SCENE:
            cap_rejected.append(input_row)
            continue
        accepted.append(item)
    return accepted, {
        "native_overlap_rejected_diagnostic_rows": native_rejected,
        "self_nms_rejected_diagnostic_rows": nms_rejected,
        "output_cap_rejected_diagnostic_rows": cap_rejected,
    }


def _row_payload_sha256(rows: Sequence[Any]) -> str:
    return hashlib.sha256(
        pickle.dumps(list(rows), protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()


def _write_pickle(path: Path, payload: Any) -> None:
    with path.open("xb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())


def materialize_s2_yoloe_direct_shadow(
    *,
    candidate_root: Path,
    baseline_root: Path,
    preregistration: Path,
    output_prediction_root: Path,
    config_path: Path = FROZEN_CONFIG_PATH,
) -> dict[str, Any]:
    """Validate and atomically publish the fixed dev3 S2 counterfactual."""

    candidate_root = candidate_root.resolve()
    baseline_root = baseline_root.resolve()
    preregistration = preregistration.resolve()
    output_root = output_prediction_root.resolve()
    config_path = config_path.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise S2ShadowError(f"refusing to overwrite output root: {output_root}")
    if not candidate_root.is_dir() or not baseline_root.is_dir():
        raise S2ShadowError("candidate and frozen T05 roots must be directories")
    _regular_file(preregistration, "S2 preregistration")

    _validate_config(config_path)
    frozen_before = _validate_frozen_hashes(config_path)
    preregistration_before = _sha256(preregistration)
    if preregistration_before != FROZEN_PREREGISTRATION_SHA256:
        raise S2ShadowError(
            "frozen S2 preregistration SHA-256 mismatch: "
            f"expected={FROZEN_PREREGISTRATION_SHA256}, "
            f"actual={preregistration_before}"
        )
    self_path = _regular_file(Path(__file__), "S2 materializer source")
    self_before = _sha256(self_path)

    candidate_before: dict[str, str] = {}
    native_before: dict[str, str] = {}
    analyzed: list[dict[str, Any]] = []
    accepted_global: list[dict[str, Any]] = []
    counterfactual_corners: list[np.ndarray] = []
    counterfactual_stored_scores: list[float] = []
    counterfactual_formal_scores: list[float] = []
    counterfactual_is_native: list[bool] = []
    counterfactual_offsets = [0]

    for scene_index, scene in enumerate(DEV3_SCENES):
        if SCENE_PATTERN.fullmatch(scene) is None:
            raise S2ShadowError(f"invalid frozen dev3 scene ID: {scene}")
        diagnostic_path = candidate_root / f"{scene}{DIAGNOSTIC_SUFFIX}"
        prediction_path = baseline_root / f"{scene}{PREDICTION_SUFFIX}"
        candidate_before[scene] = _sha256(
            _regular_file(diagnostic_path, "S2 candidate diagnostic")
        )
        native_before[scene] = _sha256(
            _regular_file(prediction_path, "frozen T05 prediction")
        )
        diagnostic = _load_diagnostic(diagnostic_path, scene)
        payload, native_rows, native_corners, native_scores = _load_native_payload(
            prediction_path
        )
        arrays = diagnostic["arrays"]
        source_rows: list[dict[str, Any]] = []
        for diagnostic_row in diagnostic["supplemental_positions"]:
            diagnostic_row = int(diagnostic_row)
            source_rows.append(
                {
                    "scene_id": scene,
                    "scene_index": scene_index,
                    "diagnostic_row": diagnostic_row,
                    "result_index": int(arrays["result_indices"][diagnostic_row]),
                    "track_id": int(arrays["track_ids"][diagnostic_row]),
                    "box": np.asarray(arrays["boxes"][diagnostic_row], dtype=np.float64),
                    "corners": _aabb_corners(arrays["boxes"][diagnostic_row]),
                    "raw_score": float(arrays["scores"][diagnostic_row]),
                    "valid_point_count": int(arrays["point_mask"][diagnostic_row].sum()),
                }
            )
        accepted, rejections = _terminal_filter(source_rows, native_corners)
        minimum_native_score = float(native_scores.min()) if len(native_scores) else None
        public_accepted: list[dict[str, Any]] = []
        appended_rows = []
        for output_rank, row in enumerate(accepted):
            raw_score = float(row["raw_score"])
            diagnostic_score = raw_score
            if minimum_native_score is not None:
                diagnostic_score = min(
                    diagnostic_score, float(np.nextafter(minimum_native_score, 0.0))
                )
            corners = np.asarray(row["corners"], dtype=np.float32)
            appended_rows.append((0, corners, diagnostic_score))
            public = {
                "scene_id": scene,
                "scene_index": scene_index,
                "terminal_rank": output_rank,
                "diagnostic_row": int(row["diagnostic_row"]),
                "result_index": int(row["result_index"]),
                "track_id": int(row["track_id"]),
                "box_center_extent": np.asarray(row["box"]).tolist(),
                "corners_world": corners.tolist(),
                "raw_score_provenance": raw_score,
                "stored_appended_score_diagnostic_only": diagnostic_score,
                "formal_evaluation_score": FORMAL_EVALUATION_SCORE,
                "max_native_aabb_iou": float(row["max_native_aabb_iou"]),
                "valid_point_count": int(row["valid_point_count"]),
            }
            public_accepted.append(public)
            accepted_global.append(public)

        output_payload = [list(native_rows) + appended_rows]
        analyzed.append(
            {
                "scene": scene,
                "scene_index": scene_index,
                "native_payload": payload,
                "native_rows": native_rows,
                "native_prefix_sha256": _row_payload_sha256(native_rows),
                "native_corners": native_corners,
                "native_scores": native_scores,
                "output_payload": output_payload,
                "accepted": public_accepted,
                "rejections": rejections,
                "diagnostic_summary": diagnostic["summary"],
                "diagnostic_row_count": len(arrays["scores"]),
                "supplemental_row_count": len(diagnostic["supplemental_positions"]),
            }
        )
        for native_corner, native_score in zip(native_corners, native_scores):
            counterfactual_corners.append(np.asarray(native_corner, dtype=np.float32))
            counterfactual_stored_scores.append(float(native_score))
            counterfactual_formal_scores.append(FORMAL_EVALUATION_SCORE)
            counterfactual_is_native.append(True)
        for row in public_accepted:
            counterfactual_corners.append(np.asarray(row["corners_world"], dtype=np.float32))
            counterfactual_stored_scores.append(
                float(row["stored_appended_score_diagnostic_only"])
            )
            counterfactual_formal_scores.append(FORMAL_EVALUATION_SCORE)
            counterfactual_is_native.append(False)
        counterfactual_offsets.append(len(counterfactual_corners))

    candidate_after = {
        scene: _sha256(candidate_root / f"{scene}{DIAGNOSTIC_SUFFIX}")
        for scene in DEV3_SCENES
    }
    native_after = {
        scene: _sha256(baseline_root / f"{scene}{PREDICTION_SUFFIX}")
        for scene in DEV3_SCENES
    }
    frozen_after = _validate_frozen_hashes(config_path)
    if candidate_after != candidate_before:
        raise S2ShadowError("S2 diagnostic inputs changed during materialization")
    if native_after != native_before:
        raise S2ShadowError("frozen T05 predictions changed during materialization")
    if frozen_after != frozen_before:
        raise S2ShadowError("frozen config/model/source inputs changed during materialization")
    if _sha256(preregistration) != preregistration_before:
        raise S2ShadowError("S2 preregistration changed during materialization")
    if _sha256(self_path) != self_before:
        raise S2ShadowError("S2 materializer source changed during execution")

    candidate_arrays: dict[str, np.ndarray] = {
        "scene_ids": np.asarray(DEV3_SCENES, dtype="<U12"),
        "candidate_scene_index": np.asarray(
            [row["scene_index"] for row in accepted_global], dtype=np.int16
        ),
        "candidate_terminal_rank": np.asarray(
            [row["terminal_rank"] for row in accepted_global], dtype=np.int16
        ),
        "candidate_diagnostic_row": np.asarray(
            [row["diagnostic_row"] for row in accepted_global], dtype=np.int32
        ),
        "candidate_result_index": np.asarray(
            [row["result_index"] for row in accepted_global], dtype=np.int32
        ),
        "candidate_track_id": np.asarray(
            [row["track_id"] for row in accepted_global], dtype=np.int64
        ),
        "candidate_box_center_extent": np.asarray(
            [row["box_center_extent"] for row in accepted_global], dtype=np.float32
        ).reshape((-1, 6)),
        "candidate_corners_world": np.asarray(
            [row["corners_world"] for row in accepted_global], dtype=np.float32
        ).reshape((-1, 8, 3)),
        "candidate_raw_score_provenance": np.asarray(
            [row["raw_score_provenance"] for row in accepted_global], dtype=np.float32
        ),
        "candidate_stored_appended_score_diagnostic_only": np.asarray(
            [row["stored_appended_score_diagnostic_only"] for row in accepted_global],
            dtype=np.float32,
        ),
        "candidate_formal_evaluation_score": np.ones(
            len(accepted_global), dtype=np.float32
        ),
        "candidate_max_native_aabb_iou": np.asarray(
            [row["max_native_aabb_iou"] for row in accepted_global], dtype=np.float32
        ),
        "candidate_valid_point_count": np.asarray(
            [row["valid_point_count"] for row in accepted_global], dtype=np.int16
        ),
        "counterfactual_scene_offsets": np.asarray(counterfactual_offsets, dtype=np.int32),
        "counterfactual_corners_world": np.asarray(
            counterfactual_corners, dtype=np.float32
        ).reshape((-1, 8, 3)),
        "counterfactual_stored_score_provenance": np.asarray(
            counterfactual_stored_scores, dtype=np.float32
        ),
        "counterfactual_formal_evaluation_score": np.asarray(
            counterfactual_formal_scores, dtype=np.float32
        ),
        "counterfactual_is_native_prefix": np.asarray(
            counterfactual_is_native, dtype=bool
        ),
    }
    for value in candidate_arrays.values():
        value.setflags(write=False)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    published = False
    try:
        scene_reports: dict[str, Any] = {}
        output_hashes: dict[str, str] = {}
        for row in analyzed:
            scene = row["scene"]
            output_path = staging / f"{scene}{PREDICTION_SUFFIX}"
            _write_pickle(output_path, row["output_payload"])
            _, reloaded_rows, _, _ = _load_native_payload(output_path)
            native_count = len(row["native_rows"])
            prefix_hash = _row_payload_sha256(reloaded_rows[:native_count])
            if prefix_hash != row["native_prefix_sha256"]:
                raise S2ShadowError(f"native T05 prefix changed in output for {scene}")
            if len(reloaded_rows) != native_count + len(row["accepted"]):
                raise S2ShadowError(f"counterfactual output row count changed for {scene}")
            output_hashes[scene] = _sha256(output_path)
            scene_reports[scene] = {
                "scene_index": int(row["scene_index"]),
                "diagnostic_sha256_before": candidate_before[scene],
                "diagnostic_sha256_after": candidate_after[scene],
                "diagnostic_input_unchanged": candidate_before[scene] == candidate_after[scene],
                "diagnostic_row_count": int(row["diagnostic_row_count"]),
                "supplemental_rows_read_source_index_minus_one": int(
                    row["supplemental_row_count"]
                ),
                "native_prediction_sha256_before": native_before[scene],
                "native_prediction_sha256_after": native_after[scene],
                "native_input_unchanged": native_before[scene] == native_after[scene],
                "native_prefix_row_count": native_count,
                "native_prefix_payload_sha256_input": row["native_prefix_sha256"],
                "native_prefix_payload_sha256_output": prefix_hash,
                "native_prefix_exact": prefix_hash == row["native_prefix_sha256"],
                "accepted_candidate_count": len(row["accepted"]),
                "counterfactual_row_count": native_count + len(row["accepted"]),
                "counterfactual_prediction_sha256": output_hashes[scene],
                "terminal_rejections": row["rejections"],
                "accepted_candidates": row["accepted"],
                "producer_summary": row["diagnostic_summary"],
            }

        npz_path = staging / ARRAY_FILENAME
        _write_deterministic_npz(npz_path, candidate_arrays)
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "mode": "shadow",
            "output_inert": True,
            "birth": False,
            "active_authorized": False,
            "native_mutation_applied": False,
            "gt_access": False,
            "oracle_access": False,
            "training_free": True,
            "online_learning": False,
            "past_current_only": True,
            "future_frames_used": False,
            "detector_semantics_used_for_gate": False,
            "native_clip_access": False,
            "native_clip_unchanged": True,
            "coordinate_frame": "scannet_world",
            "score_mode_for_formal_evaluation": "constant_1.0",
            "stored_scores_are_diagnostic_only": True,
            "scene_count": len(DEV3_SCENES),
            "scene_order": list(DEV3_SCENES),
            "candidate_count": len(accepted_global),
            "npz_file": ARRAY_FILENAME,
            "npz_sha256": _sha256(npz_path),
            "candidate_content_sha256": _array_content_sha256(candidate_arrays),
            "counterfactual_prediction_root": os.fspath(output_root),
            "counterfactual_prediction_sha256": output_hashes,
            "input": {
                "candidate_root": os.fspath(candidate_root),
                "baseline_root": os.fspath(baseline_root),
                "preregistration": os.fspath(preregistration),
                "preregistration_expected_sha256": FROZEN_PREREGISTRATION_SHA256,
                "preregistration_sha256": preregistration_before,
                "materializer_source": os.fspath(self_path),
                "materializer_source_sha256": self_before,
                "frozen_inputs": frozen_before,
            },
            "frozen_policy": {
                "candidate_source_index": -1,
                "diagnostic_order_preserved": True,
                "native_novelty_aabb_iou_strict_less_than": NATIVE_NOVELTY_IOU,
                "candidate_self_nms_aabb_iou_strict_less_than": SELF_NMS_IOU,
                "maximum_appended_candidates_per_scene": MAX_OUTPUTS_PER_SCENE,
                "terminal_candidate_labels_ignored": True,
                "terminal_clip_access": False,
                "native_prefix_rows_exact": True,
                "formal_evaluation_score": FORMAL_EVALUATION_SCORE,
            },
            "input_hash_identity": {
                "candidate_diagnostics_before_after": candidate_before == candidate_after,
                "native_predictions_before_after": native_before == native_after,
                "frozen_sources_before_after": frozen_before == frozen_after,
                "preregistration_before_after": _sha256(preregistration)
                == preregistration_before,
                "materializer_before_after": _sha256(self_path) == self_before,
            },
            "scenes": scene_reports,
        }
        _write_json_exclusive(staging / AUDIT_FILENAME, manifest)
        staging.rename(output_root)
        published = True
        return manifest
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal no-GT S2 YOLOE-direct dev3 terminal counterfactual"
    )
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--output-prediction-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=FROZEN_CONFIG_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = materialize_s2_yoloe_direct_shadow(
        candidate_root=args.candidate_root,
        baseline_root=args.baseline_root,
        preregistration=args.preregistration,
        output_prediction_root=args.output_prediction_root,
        config_path=args.config,
    )
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "scene_count": manifest["scene_count"],
                "candidate_count": manifest["candidate_count"],
                "output_prediction_root": os.fspath(args.output_prediction_root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
