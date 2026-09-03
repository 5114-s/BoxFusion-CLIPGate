#!/usr/bin/env python3
"""Materialize sealed SRAW-P3HB-CLIP births onto the unchanged B05 prefix.

The input is a completed, no-GT, past-only shadow decision ledger.  This
program does not rerun a proposal model or an admission policy.  It applies
only terminal native novelty and suffix self-NMS, then appends the surviving
world-space boxes with the formal ScanNet class/score ``0/1.0``.

Terminal novelty reads the final B05 predictions, so this is an active
paper100 materialization contract rather than proof of live per-keyframe
native-state novelty.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

from tools.materialize_scannet_raw_boxer_past3_birth_full100 import (  # noqa: E402
    BirthMaterializationError,
    NativePrediction,
    _aabb_overlap_matrices,
    _assert_native_prefix,
    _load_native_prediction,
    _regular_file,
    _scene_list,
    _sha256,
    _write_json,
    _write_pickle,
)
from tools.run_scannet_raw_boxer_clip_vocab_shadow_full100 import (  # noqa: E402
    TARGET_GROUP_ALIASES,
)


SHADOW_SCHEMA = "boxfusion.scannet_sraw_p3hb_clip_shadow_paper100.v1"
PROTOCOL_ID = "SRAW-P3HB-CLIP-V1"
EXPECTED_PROTOCOL_SHA256 = (
    "a7f10f6b73cbaa8fb6948f0de0fee9765f5dec535222401c27ec0cef80f66c0c"
)
SCHEMA = "boxfusion.scannet_sraw_p3hb_clip_birth_paper100.v1"
MANIFEST_NAME = "SRAW_P3HB_CLIP_BIRTH_PAPER100.json"
PREDICTION_SUFFIX = "_boxes.pkl"

NATIVE_NOVELTY_AABB_IOU = 0.10
NATIVE_MAX_BIDIRECTIONAL_CONTAINMENT = 0.50
SELF_NMS_AABB_IOU = 0.15
SELF_NMS_BIDIRECTIONAL_CONTAINMENT = 0.25
APPENDED_CLASS_ID = 0
APPENDED_SCORE = 1.0

SOURCE_RE = re.compile(
    r"^(?P<scene>scene[0-9]{4}_[0-9]{2})/"
    r"frame_(?P<frame>[0-9]{6})/raw_(?P<raw>[0-9]{3})$"
)

REQUIRED_SHADOW_CONTRACTS: Mapping[str, bool] = {
    "shadow_only": True,
    "birth_enabled": False,
    "native_output_mutation": False,
    "ground_truth_access": False,
    "annotation_access": False,
    "evaluator_access": False,
    "future_frame_access": False,
    "training": False,
    "online_learning": False,
    "past_only": True,
}


class SRAWBirthMaterializationError(BirthMaterializationError):
    """Raised when the SRAW shadow or active-output contract differs."""


@dataclass(frozen=True)
class AcceptedBirth:
    scene_id: str
    track_id: int
    confirmation_frame_id: int
    selected_source_id: str
    target_group: str
    evidence_source_ids: tuple[str, str, str]
    evidence_frame_ids: tuple[int, int, int]
    corners: np.ndarray


@dataclass(frozen=True)
class ShadowLedger:
    path: Path
    sha256: str
    births: Mapping[str, tuple[AcceptedBirth, ...]]
    accepted_birth_count: int


def _strict_int(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise SRAWBirthMaterializationError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise SRAWBirthMaterializationError(f"{label} must be >= {minimum}")
    return result


def _load_json(path: Path, label: str) -> dict[str, Any]:
    source = _regular_file(path, label)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SRAWBirthMaterializationError(f"invalid {label}: {source}") from error
    if not isinstance(value, dict):
        raise SRAWBirthMaterializationError(f"{label} must contain a JSON object")
    return value


def _parse_corners(value: object, label: str) -> np.ndarray:
    try:
        corners = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise SRAWBirthMaterializationError(
            f"{label} must be finite [8,3]"
        ) from error
    if (
        corners.shape != (8, 3)
        or not np.isfinite(corners).all()
        or np.any(np.ptp(corners, axis=0) <= 0.0)
    ):
        raise SRAWBirthMaterializationError(
            f"{label} must be finite nondegenerate [8,3]"
        )
    return np.ascontiguousarray(corners)


def _parse_birth(scene_id: str, index: int, value: object) -> AcceptedBirth:
    label = f"{scene_id}.accepted_births[{index}]"
    if not isinstance(value, Mapping):
        raise SRAWBirthMaterializationError(f"{label} must be an object")
    track_id = _strict_int(value.get("track_id"), f"{label}.track_id")
    confirmation = _strict_int(
        value.get("confirmation_frame_id"), f"{label}.confirmation_frame_id"
    )
    source_id = value.get("selected_source_id")
    match = SOURCE_RE.fullmatch(source_id) if isinstance(source_id, str) else None
    if match is None or match.group("scene") != scene_id:
        raise SRAWBirthMaterializationError(
            f"{label}.selected_source_id is not canonical for its scene"
        )
    evidence_source_values = value.get("evidence_source_ids")
    evidence_frame_values = value.get("evidence_frame_ids")
    if not isinstance(evidence_source_values, list) or len(evidence_source_values) != 3:
        raise SRAWBirthMaterializationError(
            f"{label}.evidence_source_ids must contain exactly three rows"
        )
    if not isinstance(evidence_frame_values, list) or len(evidence_frame_values) != 3:
        raise SRAWBirthMaterializationError(
            f"{label}.evidence_frame_ids must contain exactly three rows"
        )
    evidence_frames = tuple(
        _strict_int(frame_id, f"{label}.evidence_frame_ids[{ordinal}]")
        for ordinal, frame_id in enumerate(evidence_frame_values)
    )
    if not evidence_frames[0] < evidence_frames[1] < evidence_frames[2]:
        raise SRAWBirthMaterializationError(
            f"{label}.evidence_frame_ids must be strictly increasing"
        )
    if evidence_frames[2] != confirmation:
        raise SRAWBirthMaterializationError(
            f"{label} third evidence frame must equal confirmation frame"
        )
    evidence_sources: list[str] = []
    for ordinal, (raw_source, frame_id) in enumerate(
        zip(evidence_source_values, evidence_frames, strict=True)
    ):
        evidence_match = (
            SOURCE_RE.fullmatch(raw_source) if isinstance(raw_source, str) else None
        )
        if (
            evidence_match is None
            or evidence_match.group("scene") != scene_id
            or int(evidence_match.group("frame")) != frame_id
        ):
            raise SRAWBirthMaterializationError(
                f"{label}.evidence_source_ids[{ordinal}] disagrees with scene/frame"
            )
        evidence_sources.append(raw_source)
    if len(set(evidence_sources)) != 3:
        raise SRAWBirthMaterializationError(
            f"{label}.evidence_source_ids must be distinct"
        )
    if source_id not in evidence_sources:
        raise SRAWBirthMaterializationError(
            f"{label}.selected_source_id must belong to first-three evidence"
        )
    target_group = value.get("target_group")
    if not isinstance(target_group, str) or target_group not in TARGET_GROUP_ALIASES:
        raise SRAWBirthMaterializationError(
            f"{label}.target_group is outside frozen TARGET_GROUP_ALIASES"
        )
    corners = _parse_corners(value.get("corners_world"), f"{label}.corners_world")
    geometry = value.get("geometry")
    semantic = value.get("semantic")
    if not isinstance(geometry, Mapping) or geometry.get("gate_pass") is not True:
        raise SRAWBirthMaterializationError(
            f"{label}.geometry.gate_pass must be true"
        )
    if geometry.get("selected_source_id") != source_id:
        raise SRAWBirthMaterializationError(
            f"{label}.geometry.selected_source_id disagrees with top level"
        )
    geometry_corners = _parse_corners(
        geometry.get("corners_world"), f"{label}.geometry.corners_world"
    )
    if not np.array_equal(geometry_corners, corners):
        raise SRAWBirthMaterializationError(
            f"{label}.geometry.corners_world disagrees with top level"
        )
    if not isinstance(semantic, Mapping) or semantic.get("gate_pass") is not True:
        raise SRAWBirthMaterializationError(
            f"{label}.semantic.gate_pass must be true"
        )
    if semantic.get("target_group") != target_group:
        raise SRAWBirthMaterializationError(
            f"{label}.semantic.target_group disagrees with top level"
        )
    return AcceptedBirth(
        scene_id=scene_id,
        track_id=track_id,
        confirmation_frame_id=confirmation,
        selected_source_id=source_id,
        target_group=target_group,
        evidence_source_ids=tuple(evidence_sources),  # type: ignore[arg-type]
        evidence_frame_ids=evidence_frames,
        corners=corners,
    )


def load_shadow_ledger(path: Path, scenes: Sequence[str]) -> ShadowLedger:
    """Load and authenticate one exact no-GT, past-only shadow ledger."""

    source = _regular_file(path, "SRAW P3HB CLIP shadow")
    digest = _sha256(source)
    payload = _load_json(source, "SRAW P3HB CLIP shadow")
    if payload.get("schema") != SHADOW_SCHEMA:
        raise SRAWBirthMaterializationError("unsupported SRAW shadow schema")
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise SRAWBirthMaterializationError("SRAW shadow protocol_id differs")
    if payload.get("complete") is not True:
        raise SRAWBirthMaterializationError("SRAW shadow must declare complete=true")
    if payload.get("scene_count") != len(scenes):
        raise SRAWBirthMaterializationError("SRAW shadow scene_count differs")
    if payload.get("scene_order") != list(scenes):
        raise SRAWBirthMaterializationError("SRAW shadow scene_order differs")
    contracts = payload.get("contracts")
    if not isinstance(contracts, Mapping):
        raise SRAWBirthMaterializationError("SRAW shadow contracts are absent")
    for name, expected in REQUIRED_SHADOW_CONTRACTS.items():
        if contracts.get(name) is not expected:
            raise SRAWBirthMaterializationError(
                f"SRAW shadow contract {name} must be {expected!r}"
            )
    inputs = payload.get("inputs")
    if (
        not isinstance(inputs, Mapping)
        or inputs.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256
    ):
        raise SRAWBirthMaterializationError(
            "SRAW shadow inputs.protocol_sha256 differs"
        )

    scene_rows = payload.get("scenes")
    if not isinstance(scene_rows, list) or len(scene_rows) != len(scenes):
        raise SRAWBirthMaterializationError("SRAW shadow scenes must be an exact list")
    births_by_scene: dict[str, tuple[AcceptedBirth, ...]] = {}
    total = 0
    for scene_index, (scene_id, scene_row) in enumerate(
        zip(scenes, scene_rows, strict=True)
    ):
        if (
            not isinstance(scene_row, Mapping)
            or scene_row.get("scene_id") != scene_id
            or scene_row.get("scene_index") != scene_index
        ):
            raise SRAWBirthMaterializationError(
                f"SRAW shadow scene identity/order differs: {scene_id}"
            )
        raw_births = scene_row.get("accepted_births")
        if not isinstance(raw_births, list):
            raise SRAWBirthMaterializationError(
                f"{scene_id}.accepted_births must be a list"
            )
        if len(raw_births) > 2:
            raise SRAWBirthMaterializationError(
                f"{scene_id}.accepted_births exceeds frozen per-scene cap=2"
            )
        parsed = tuple(
            _parse_birth(scene_id, index, value)
            for index, value in enumerate(raw_births)
        )
        track_ids = [row.track_id for row in parsed]
        source_ids = [row.selected_source_id for row in parsed]
        confirmations = [row.confirmation_frame_id for row in parsed]
        if confirmations != sorted(confirmations):
            raise SRAWBirthMaterializationError(
                f"{scene_id}.accepted_births confirmation order must be nondecreasing"
            )
        if len(track_ids) != len(set(track_ids)):
            raise SRAWBirthMaterializationError(
                f"{scene_id} contains duplicate accepted track_id values"
            )
        if len(source_ids) != len(set(source_ids)):
            raise SRAWBirthMaterializationError(
                f"{scene_id} contains duplicate selected_source_id values"
            )
        births_by_scene[scene_id] = parsed
        total += len(parsed)
    declared = payload.get("accepted_birth_count")
    if declared is not None and declared != total:
        raise SRAWBirthMaterializationError(
            "SRAW shadow accepted_birth_count differs from scene census"
        )
    if _sha256(source) != digest:
        raise SRAWBirthMaterializationError("SRAW shadow changed while it was read")
    return ShadowLedger(
        path=source.resolve(),
        sha256=digest,
        births=births_by_scene,
        accepted_birth_count=total,
    )


def select_terminal_births(
    births: Sequence[AcceptedBirth], native_corners: np.ndarray
) -> tuple[tuple[AcceptedBirth, ...], list[dict[str, Any]]]:
    """Apply terminal native novelty and deterministic suffix self-NMS."""

    corners = (
        np.stack([row.corners for row in births])
        if births
        else np.empty((0, 8, 3), dtype=np.float64)
    )
    native_iou, candidate_in_native, native_in_candidate = _aabb_overlap_matrices(
        corners, native_corners
    )
    self_iou, self_left, self_right = _aabb_overlap_matrices(corners, corners)
    selected: list[AcceptedBirth] = []
    kept_indices: list[int] = []
    decisions: list[dict[str, Any]] = []
    for index, birth in enumerate(births):
        max_native_iou = (
            float(native_iou[index].max()) if native_iou.shape[1] else 0.0
        )
        max_candidate_in_native = (
            float(candidate_in_native[index].max())
            if candidate_in_native.shape[1]
            else 0.0
        )
        max_native_in_candidate = (
            float(native_in_candidate[index].max())
            if native_in_candidate.shape[1]
            else 0.0
        )
        decision = "accepted"
        if max_native_iou >= NATIVE_NOVELTY_AABB_IOU:
            decision = "native_overlap"
        elif (
            max_candidate_in_native >= NATIVE_MAX_BIDIRECTIONAL_CONTAINMENT
            or max_native_in_candidate >= NATIVE_MAX_BIDIRECTIONAL_CONTAINMENT
        ):
            decision = "native_containment"
        else:
            for kept_index in kept_indices:
                if (
                    self_iou[index, kept_index] >= SELF_NMS_AABB_IOU
                    or self_left[index, kept_index]
                    >= SELF_NMS_BIDIRECTIONAL_CONTAINMENT
                    or self_right[index, kept_index]
                    >= SELF_NMS_BIDIRECTIONAL_CONTAINMENT
                ):
                    decision = "self_nms"
                    break
        if decision == "accepted":
            kept_indices.append(index)
            selected.append(birth)
        decisions.append(
            {
                "input_index": index,
                "track_id": birth.track_id,
                "confirmation_frame_id": birth.confirmation_frame_id,
                "selected_source_id": birth.selected_source_id,
                "target_group": birth.target_group,
                "decision": decision,
                "max_native_aabb_iou": max_native_iou,
                "max_candidate_in_native_containment": max_candidate_in_native,
                "max_native_in_candidate_containment": max_native_in_candidate,
            }
        )
    return tuple(selected), decisions


def _augmented_payload(
    native: NativePrediction, selected: Sequence[AcceptedBirth]
) -> list[Any] | tuple[Any, ...]:
    suffix = [
        (
            APPENDED_CLASS_ID,
            np.ascontiguousarray(row.corners, dtype=np.float32),
            APPENDED_SCORE,
        )
        for row in selected
    ]
    rows: list[Any] | tuple[Any, ...]
    if isinstance(native.rows, tuple):
        rows = tuple(native.rows) + tuple(suffix)
    else:
        rows = list(native.rows) + suffix
    output: list[Any] | tuple[Any, ...]
    output = (rows,) if isinstance(native.payload, tuple) else [rows]
    _assert_native_prefix(native.rows, output[0], "in-memory SRAW birth output")
    return output


def _policy_manifest() -> dict[str, Any]:
    return {
        "shadow_order_is_active_rank": True,
        "native_novelty_aabb_iou_gte_reject": NATIVE_NOVELTY_AABB_IOU,
        "native_bidirectional_containment_gte_reject": (
            NATIVE_MAX_BIDIRECTIONAL_CONTAINMENT
        ),
        "self_nms_aabb_iou_gte_reject": SELF_NMS_AABB_IOU,
        "self_nms_bidirectional_containment_gte_reject": (
            SELF_NMS_BIDIRECTIONAL_CONTAINMENT
        ),
        "appended_class_id": APPENDED_CLASS_ID,
        "appended_score": APPENDED_SCORE,
    }


def materialize_scannet_sraw_p3hb_clip_birth_paper100(
    *,
    scene_list: Path,
    baseline_root: Path,
    shadow_path: Path,
    output_root: Path,
    expected_scene_count: int = 100,
    plan_only: bool = False,
) -> dict[str, Any]:
    """Validate the full ledger and optionally atomically publish predictions."""

    if baseline_root.is_symlink() or not baseline_root.is_dir():
        raise SRAWBirthMaterializationError(
            f"baseline root must be a non-symlink directory: {baseline_root}"
        )
    if expected_scene_count <= 0:
        raise SRAWBirthMaterializationError("expected scene count must be positive")
    output_root = output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise SRAWBirthMaterializationError(
            f"refusing to overwrite output root: {output_root}"
        )
    scenes = _scene_list(scene_list, expected_scene_count)
    shadow = load_shadow_ledger(shadow_path, scenes)

    stage: Path | None = None
    if not plan_only:
        output_root.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(prefix=f".{output_root.name}.stage-", dir=output_root.parent)
        )
    native_hashes: dict[str, str] = {}
    output_hashes: dict[str, str] = {}
    scene_reports: dict[str, Any] = {}
    total_native = 0
    total_births = 0
    try:
        for scene_id in scenes:
            native_path = _regular_file(
                baseline_root / f"{scene_id}{PREDICTION_SUFFIX}",
                "native B05 prediction",
            )
            native_digest = _sha256(native_path)
            native_hashes[scene_id] = native_digest
            native = _load_native_prediction(native_path)
            selected, decisions = select_terminal_births(
                shadow.births[scene_id], native.corners
            )
            if _sha256(native_path) != native_digest:
                raise SRAWBirthMaterializationError(
                    f"native prediction changed during materialization: {scene_id}"
                )

            if not plan_only:
                assert stage is not None
                output_path = stage / f"{scene_id}{PREDICTION_SUFFIX}"
                _write_pickle(output_path, _augmented_payload(native, selected))
                reloaded = _load_native_prediction(output_path)
                _assert_native_prefix(native.rows, reloaded.rows, scene_id)
                if len(reloaded.rows) != len(native.rows) + len(selected):
                    raise SRAWBirthMaterializationError(
                        f"suffix count changed after reload: {scene_id}"
                    )
                output_hashes[scene_id] = _sha256(output_path)

            reasons = (
                "accepted",
                "native_overlap",
                "native_containment",
                "self_nms",
            )
            scene_reports[scene_id] = {
                "native_count": len(native.rows),
                "shadow_accepted_birth_count": len(shadow.births[scene_id]),
                "birth_count": len(selected),
                "decision_counts": {
                    reason: sum(row["decision"] == reason for row in decisions)
                    for reason in reasons
                },
                "native_prefix_row_identity_verified": not plan_only,
                "suffix": [
                    {
                        "suffix_index": index,
                        "track_id": row.track_id,
                        "confirmation_frame_id": row.confirmation_frame_id,
                        "selected_source_id": row.selected_source_id,
                        "target_group": row.target_group,
                        "evidence_source_ids": list(row.evidence_source_ids),
                        "evidence_frame_ids": list(row.evidence_frame_ids),
                        "class_id": APPENDED_CLASS_ID,
                        "score": APPENDED_SCORE,
                        "corners_world": row.corners.tolist(),
                    }
                    for index, row in enumerate(selected)
                ],
                "terminal_decisions": decisions,
            }
            total_native += len(native.rows)
            total_births += len(selected)

        if _sha256(shadow.path) != shadow.sha256:
            raise SRAWBirthMaterializationError(
                "SRAW shadow changed during materialization"
            )
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "mode": "plan_only" if plan_only else "active_terminal_birth",
            "plan_only": plan_only,
            "complete": True,
            "scene_count": len(scenes),
            "scene_order": list(scenes),
            "training_free": True,
            "target_dataset_training": False,
            "online_learning": False,
            "past_only_shadow_admission": True,
            "causal_shadow_generation": True,
            "terminal_replay_materialization": True,
            "strict_online_native_novelty": False,
            "gt_access": False,
            "annotation_access": False,
            "evaluator_access": False,
            "native_rows_are_unchanged_prefix": True,
            "native_clip_unchanged": True,
            "coordinate_frame": "scannet_world",
            "score_mode": "constant_1.0",
            "class_mode": "inert_0_scannet_class_agnostic_evaluator",
            "native_count": total_native,
            "shadow_accepted_birth_count": shadow.accepted_birth_count,
            "birth_count": total_births,
            "frozen_policy": _policy_manifest(),
            "inputs": {
                "scene_list": os.fspath(scene_list.resolve()),
                "scene_list_sha256": _sha256(scene_list),
                "baseline_root": os.fspath(baseline_root.resolve()),
                "shadow": os.fspath(shadow.path),
                "shadow_sha256": shadow.sha256,
                "shadow_schema": SHADOW_SCHEMA,
                "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
                "materializer_source": os.fspath(Path(__file__).resolve()),
                "materializer_source_sha256": _sha256(Path(__file__).resolve()),
            },
            "native_prediction_sha256": native_hashes,
            "output_prediction_sha256": output_hashes,
            "scenes": scene_reports,
        }
        if plan_only:
            return manifest
        assert stage is not None
        _write_json(stage / MANIFEST_NAME, manifest)
        directory_fd = os.open(stage, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if output_root.exists() or output_root.is_symlink():
            raise SRAWBirthMaterializationError(
                f"refusing to overwrite output root: {output_root}"
            )
        os.rename(stage, output_root)
        stage = None
        return manifest
    except Exception:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-list",
        type=Path,
        default=REPOSITORY_ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=REPOSITORY_ROOT / "results/scannet_t05_boxer_replay_active_score05",
    )
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "results/scannet_sraw_p3hb_clip_birth_score05",
    )
    parser.add_argument("--expected-scene-count", type=int, default=100)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="validate and select every scene without creating an output root",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = materialize_scannet_sraw_p3hb_clip_birth_paper100(
        scene_list=args.scene_list,
        baseline_root=args.baseline_root,
        shadow_path=args.shadow,
        output_root=args.output_root,
        expected_scene_count=args.expected_scene_count,
        plan_only=args.plan_only,
    )
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "mode": manifest["mode"],
                "scene_count": manifest["scene_count"],
                "native_count": manifest["native_count"],
                "shadow_accepted_birth_count": manifest[
                    "shadow_accepted_birth_count"
                ],
                "birth_count": manifest["birth_count"],
                "output_root": (
                    None if args.plan_only else os.fspath(args.output_root.resolve())
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
