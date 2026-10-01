#!/usr/bin/env python3
"""Materialize frozen UDC-MobileSAM receipts as a Cbest birth suffix.

The input is a sealed, causal shadow sidecar.  This terminal program reads no
RGB-D data, annotation, or evaluator state: it validates the sidecar, compares
its fused boxes with the immutable native Cbest terminal boxes, and appends at
most four boxes per scene.  Native rows remain the exact prefix of every output
prediction.  ``--plan-only`` executes validation and selection without writing
an output directory.

The terminal native gate deliberately has *no* IoU cutoff.  It rejects only
when either directed AABB containment is at least 0.80.  Accepted receipts are
then class-agnostically suppressed at AABB IoU 0.15 or either directed
containment 0.50.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

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


SIDECAR_SCHEMA = "boxfusion.scannet_udc_mobilesam_full100.v1"
SCHEMA = "boxfusion.scannet_udc_mobilesam_birth_full100.v1"
SIDECAR_NAME = "UDC_MOBILESAM_FULL100.json"
MANIFEST_NAME = "UDC_MOBILESAM_BIRTH_FULL100.json"
PREDICTION_SUFFIX = "_boxes.pkl"

# Frozen terminal policy.  There is intentionally no native IoU hard gate.
NATIVE_BIDIRECTIONAL_CONTAINMENT = 0.80
SELF_NMS_AABB_IOU = 0.15
SELF_NMS_BIDIRECTIONAL_CONTAINMENT = 0.50
MAX_BIRTHS_PER_SCENE = 4
APPENDED_CLASS_ID = 0
APPENDED_SCORE = 1.0


@dataclass(frozen=True)
class UDCReceipt:
    scene_id: str
    track_id: int
    confirmation_frame_id: int
    evidence_frame_ids: tuple[int, int, int]
    mean_predicted_iou: float
    supported_voxel_count: int
    fused_obb_extent_xyz: tuple[float, float, float]
    corners: np.ndarray
    pre_novelty_pass: bool


@dataclass(frozen=True)
class UDCSidecar:
    path: Path
    sha256: str
    receipts: dict[str, tuple[UDCReceipt, ...]]
    receipt_count: int


def _strict_int(value: object, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise BirthMaterializationError(f"{label} must be an integer")
    result = int(value)
    if result < 0:
        raise BirthMaterializationError(f"{label} must be nonnegative")
    return result


def _strict_float(value: object, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise BirthMaterializationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise BirthMaterializationError(f"{label} must be finite")
    return result


def _triple_frame_ids(value: object, label: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise BirthMaterializationError(f"{label} must be a three-element list")
    result = tuple(
        _strict_int(item, f"{label}[{index}]") for index, item in enumerate(value)
    )
    if not result[0] < result[1] < result[2]:
        raise BirthMaterializationError(f"{label} must be strictly increasing")
    return result  # type: ignore[return-value]


def _parse_receipt(scene_id: str, value: object, index: int) -> UDCReceipt:
    if not isinstance(value, dict):
        raise BirthMaterializationError(
            f"{scene_id} receipt {index} must be an object"
        )
    label = f"{scene_id} receipt {index}"
    carried_scene = value.get("scene_id", value.get("scene"))
    if carried_scene is not None and carried_scene != scene_id:
        raise BirthMaterializationError(f"{label} carries a different scene")

    frames = _triple_frame_ids(
        value.get("evidence_frame_ids"), f"{label} evidence_frame_ids"
    )
    confirmation = _strict_int(
        value.get("confirmation_frame_id"), f"{label} confirmation_frame_id"
    )
    if confirmation != frames[-1]:
        raise BirthMaterializationError(
            f"{label} confirmation must equal the third causal evidence frame"
        )

    mean_iou = _strict_float(
        value.get("mean_predicted_iou"), f"{label} mean_predicted_iou"
    )
    if not 0.0 <= mean_iou <= 1.0:
        raise BirthMaterializationError(
            f"{label} mean_predicted_iou must be in [0,1]"
        )
    pre_novelty_pass = value.get("pre_novelty_pass")
    if not isinstance(pre_novelty_pass, bool):
        raise BirthMaterializationError(
            f"{label} pre_novelty_pass must be boolean"
        )

    geometry = value.get("fused_obb")
    if not isinstance(geometry, dict):
        raise BirthMaterializationError(f"{label} fused_obb must be an object")
    extent_value = geometry.get("extent_xyz")
    if not isinstance(extent_value, list) or len(extent_value) != 3:
        raise BirthMaterializationError(
            f"{label} fused_obb.extent_xyz must have length three"
        )
    extents = tuple(
        _strict_float(item, f"{label} fused_obb.extent_xyz[{axis}]")
        for axis, item in enumerate(extent_value)
    )
    if min(extents) <= 0.0:
        raise BirthMaterializationError(
            f"{label} fused_obb.extent_xyz must be strictly positive"
        )
    try:
        corners = np.asarray(geometry.get("corners_world"), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise BirthMaterializationError(
            f"{label} fused_obb.corners_world is invalid"
        ) from error
    if (
        corners.shape != (8, 3)
        or not np.isfinite(corners).all()
        or np.any(np.ptp(corners, axis=0) <= 0.0)
    ):
        raise BirthMaterializationError(
            f"{label} fused_obb.corners_world must be finite nondegenerate [8,3]"
        )

    return UDCReceipt(
        scene_id=scene_id,
        track_id=_strict_int(value.get("track_id"), f"{label} track_id"),
        confirmation_frame_id=confirmation,
        evidence_frame_ids=frames,
        mean_predicted_iou=mean_iou,
        supported_voxel_count=_strict_int(
            value.get("supported_voxel_count"),
            f"{label} supported_voxel_count",
        ),
        fused_obb_extent_xyz=extents,  # type: ignore[arg-type]
        corners=np.ascontiguousarray(corners),
        pre_novelty_pass=pre_novelty_pass,
    )


def _resolve_sidecar_json(path: Path) -> Path:
    if path.is_symlink():
        raise BirthMaterializationError(f"sidecar path must not be a symlink: {path}")
    if path.is_dir():
        path = path / SIDECAR_NAME
    return _regular_file(path, "UDC-MobileSAM sidecar")


def _enforce_contract(
    payload: dict[str, Any],
    contracts: dict[str, Any],
    key: str,
    expected: bool,
    *,
    required: bool,
) -> None:
    declarations = [owner[key] for owner in (payload, contracts) if key in owner]
    if required and not declarations:
        raise BirthMaterializationError(
            f"sidecar must declare {key}={str(expected).lower()}"
        )
    if any(value is not expected for value in declarations):
        raise BirthMaterializationError(
            f"sidecar must declare {key}={str(expected).lower()}"
        )


def load_udc_sidecar(path: Path) -> UDCSidecar:
    """Load the exact v1 UDC sidecar without consulting terminal predictions."""

    path = _resolve_sidecar_json(path)
    digest = _sha256(path)

    def _reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise BirthMaterializationError(f"invalid UDC-MobileSAM sidecar: {path}") from error
    if not isinstance(payload, dict):
        raise BirthMaterializationError("UDC-MobileSAM sidecar must contain an object")
    if payload.get("schema") != SIDECAR_SCHEMA:
        raise BirthMaterializationError(
            f"unsupported UDC-MobileSAM sidecar schema: {payload.get('schema')!r}"
        )

    contracts = payload.get("contracts", {})
    if not isinstance(contracts, dict):
        raise BirthMaterializationError("sidecar contracts must be an object")
    for key, expected in (
        ("gt_access", False),
        ("evaluator_access", False),
        ("causal_shadow_generation", True),
        ("native_prediction_access", False),
        ("output_inert", True),
    ):
        _enforce_contract(
            payload, contracts, key, expected, required=True
        )
    # These names were added to the v1 runner's detailed contract.  If a
    # compatible producer carries them, a contradictory declaration must fail
    # closed; absence alone does not invalidate older v1 sidecars.
    for key in (
        "current_frame_cutr_boxes_only",
        "past_only_tracking_and_confirmation",
    ):
        _enforce_contract(payload, contracts, key, True, required=False)
    for key in (
        "annotation_path_argument",
        "target_dataset_training",
        "online_learning",
    ):
        declarations = [owner[key] for owner in (payload, contracts) if key in owner]
        if any(item is not False for item in declarations):
            raise BirthMaterializationError(f"sidecar declares forbidden {key}")

    scenes_node = payload.get("scenes")
    if not isinstance(scenes_node, list):
        raise BirthMaterializationError("sidecar scenes must be a list")
    receipts_by_scene: dict[str, tuple[UDCReceipt, ...]] = {}
    for scene_index, scene_node in enumerate(scenes_node):
        if not isinstance(scene_node, dict):
            raise BirthMaterializationError(
                f"sidecar scenes[{scene_index}] must be an object"
            )
        scene_id = scene_node.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id:
            raise BirthMaterializationError(
                f"sidecar scenes[{scene_index}].scene_id must be a nonempty string"
            )
        if scene_id in receipts_by_scene:
            raise BirthMaterializationError(f"duplicate sidecar scene: {scene_id}")
        receipt_node = scene_node.get("receipts")
        if not isinstance(receipt_node, list):
            raise BirthMaterializationError(f"{scene_id} receipts must be a list")
        receipts = tuple(
            _parse_receipt(scene_id, record, receipt_index)
            for receipt_index, record in enumerate(receipt_node)
        )
        track_ids = [receipt.track_id for receipt in receipts]
        if len(set(track_ids)) != len(track_ids):
            raise BirthMaterializationError(
                f"{scene_id} contains duplicate receipt track_id values"
            )
        receipts_by_scene[scene_id] = receipts

    receipt_count = sum(len(receipts) for receipts in receipts_by_scene.values())
    if "receipt_count" in payload and _strict_int(
        payload["receipt_count"], "receipt_count"
    ) != receipt_count:
        raise BirthMaterializationError("sidecar receipt_count does not match scenes")
    if _sha256(path) != digest:
        raise BirthMaterializationError("UDC-MobileSAM sidecar changed while read")
    return UDCSidecar(
        path=path.resolve(),
        sha256=digest,
        receipts=receipts_by_scene,
        receipt_count=receipt_count,
    )


def _rank_key(receipt: UDCReceipt) -> tuple[float, int, int, int]:
    return (
        -receipt.mean_predicted_iou,
        -receipt.supported_voxel_count,
        receipt.confirmation_frame_id,
        receipt.track_id,
    )


def select_births(
    receipts: Sequence[UDCReceipt], native_corners: np.ndarray
) -> tuple[tuple[UDCReceipt, ...], list[dict[str, Any]]]:
    """Apply pre-novelty, containment-only native gate, self-NMS, and cap."""

    ranked = tuple(sorted(receipts, key=_rank_key))
    corners = (
        np.stack([receipt.corners for receipt in ranked])
        if ranked
        else np.empty((0, 8, 3), dtype=np.float64)
    )
    native_iou, candidate_in_native, native_in_candidate = _aabb_overlap_matrices(
        corners, native_corners
    )
    self_iou, self_left, self_right = _aabb_overlap_matrices(corners, corners)
    selected: list[UDCReceipt] = []
    kept_indices: list[int] = []
    decisions: list[dict[str, Any]] = []

    for index, receipt in enumerate(ranked):
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
        if not receipt.pre_novelty_pass:
            decision = "pre_novelty"
        elif (
            max_candidate_in_native >= NATIVE_BIDIRECTIONAL_CONTAINMENT
            or max_native_in_candidate >= NATIVE_BIDIRECTIONAL_CONTAINMENT
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
        if decision == "accepted" and len(selected) >= MAX_BIRTHS_PER_SCENE:
            decision = "scene_cap"
        if decision == "accepted":
            kept_indices.append(index)
            selected.append(receipt)

        decisions.append(
            {
                "track_id": receipt.track_id,
                "decision": decision,
                "confirmation_frame_id": receipt.confirmation_frame_id,
                "evidence_frame_ids": list(receipt.evidence_frame_ids),
                "mean_predicted_iou": receipt.mean_predicted_iou,
                "supported_voxel_count": receipt.supported_voxel_count,
                "pre_novelty_pass": receipt.pre_novelty_pass,
                "fused_obb_extent_xyz": list(receipt.fused_obb_extent_xyz),
                # IoU is diagnostic only.  It is never read by a native gate.
                "max_native_aabb_iou_diagnostic_only": max_native_iou,
                "max_candidate_in_native_containment": max_candidate_in_native,
                "max_native_in_candidate_containment": max_native_in_candidate,
            }
        )
    return tuple(selected), decisions


def _augmented_payload(
    native: NativePrediction, selected: Sequence[UDCReceipt]
) -> list[Any] | tuple[Any, ...]:
    suffix = [
        (
            APPENDED_CLASS_ID,
            np.ascontiguousarray(receipt.corners, dtype=np.float32),
            APPENDED_SCORE,
        )
        for receipt in selected
    ]
    rows: list[Any] | tuple[Any, ...]
    if isinstance(native.rows, tuple):
        rows = tuple(native.rows) + tuple(suffix)
    else:
        rows = list(native.rows) + suffix
    output: list[Any] | tuple[Any, ...]
    output = (rows,) if isinstance(native.payload, tuple) else [rows]
    _assert_native_prefix(native.rows, output[0], "in-memory UDC-MobileSAM output")
    return output


def _policy_manifest() -> dict[str, Any]:
    return {
        "pre_novelty_pass_required": True,
        "native_gate": "bidirectional_aabb_containment_only",
        "native_iou_hard_gate": False,
        "native_bidirectional_containment_gte_reject": (
            NATIVE_BIDIRECTIONAL_CONTAINMENT
        ),
        "self_nms_aabb_iou_gte_reject": SELF_NMS_AABB_IOU,
        "self_nms_bidirectional_containment_gte_reject": (
            SELF_NMS_BIDIRECTIONAL_CONTAINMENT
        ),
        "max_births_per_scene": MAX_BIRTHS_PER_SCENE,
        "ranking": [
            "mean_predicted_iou_desc",
            "supported_voxel_count_desc",
            "confirmation_frame_id_asc",
            "track_id_asc",
        ],
        "appended_class_id": APPENDED_CLASS_ID,
        "appended_score": APPENDED_SCORE,
    }


def materialize_scannet_udc_mobilesam_birth_full100(
    *,
    scene_list: Path,
    baseline_root: Path,
    udc_sidecar: Path,
    output_root: Path,
    expected_scene_count: int = 100,
    plan_only: bool = False,
) -> dict[str, Any]:
    """Validate and select all scenes, then atomically publish active output."""

    if baseline_root.is_symlink() or not baseline_root.is_dir():
        raise BirthMaterializationError(
            f"baseline root must be a non-symlink directory: {baseline_root}"
        )
    if expected_scene_count <= 0:
        raise BirthMaterializationError("expected scene count must be positive")
    if output_root.exists() or output_root.is_symlink():
        raise BirthMaterializationError(
            f"refusing to overwrite output root: {output_root}"
        )
    output_root = output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise BirthMaterializationError(
            f"refusing to overwrite output root: {output_root}"
        )

    scenes = _scene_list(scene_list, expected_scene_count)
    sidecar = load_udc_sidecar(udc_sidecar)
    if set(sidecar.receipts) != set(scenes):
        missing = sorted(set(scenes) - set(sidecar.receipts))
        extra = sorted(set(sidecar.receipts) - set(scenes))
        raise BirthMaterializationError(
            f"sidecar/protocol scene mismatch; missing={missing[:3]}, extra={extra[:3]}"
        )

    stage: Path | None = None
    if not plan_only:
        output_root.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(
                prefix=f".{output_root.name}.stage-", dir=output_root.parent
            )
        )

    native_hashes: dict[str, str] = {}
    output_hashes: dict[str, str] = {}
    reports: dict[str, Any] = {}
    total_native = 0
    total_births = 0
    try:
        for scene_id in scenes:
            native_path = _regular_file(
                baseline_root / f"{scene_id}{PREDICTION_SUFFIX}",
                "native Cbest prediction",
            )
            native_digest = _sha256(native_path)
            native_hashes[scene_id] = native_digest
            native = _load_native_prediction(native_path)
            selected, decisions = select_births(
                sidecar.receipts[scene_id], native.corners
            )
            if _sha256(native_path) != native_digest:
                raise BirthMaterializationError(
                    f"native prediction changed during materialization: {scene_id}"
                )

            if not plan_only:
                assert stage is not None
                output_path = stage / f"{scene_id}{PREDICTION_SUFFIX}"
                _write_pickle(output_path, _augmented_payload(native, selected))
                reloaded = _load_native_prediction(output_path)
                _assert_native_prefix(native.rows, reloaded.rows, scene_id)
                if len(reloaded.rows) != len(native.rows) + len(selected):
                    raise BirthMaterializationError(
                        f"suffix count changed after reload: {scene_id}"
                    )
                output_hashes[scene_id] = _sha256(output_path)

            reasons = (
                "accepted",
                "pre_novelty",
                "native_containment",
                "self_nms",
                "scene_cap",
            )
            reports[scene_id] = {
                "native_count": len(native.rows),
                "udc_receipt_count": len(sidecar.receipts[scene_id]),
                "birth_count": len(selected),
                "decision_counts": {
                    reason: sum(row["decision"] == reason for row in decisions)
                    for reason in reasons
                },
                "native_prefix_row_identity_verified": not plan_only,
                "suffix": [
                    {
                        "suffix_index": suffix_index,
                        "track_id": receipt.track_id,
                        "class_id": APPENDED_CLASS_ID,
                        "score": APPENDED_SCORE,
                        "corners_world": receipt.corners.tolist(),
                        "confirmation_frame_id": receipt.confirmation_frame_id,
                        "evidence_frame_ids": list(receipt.evidence_frame_ids),
                        "mean_predicted_iou": receipt.mean_predicted_iou,
                        "supported_voxel_count": receipt.supported_voxel_count,
                    }
                    for suffix_index, receipt in enumerate(selected)
                ],
                "receipt_decisions": decisions,
            }
            total_native += len(native.rows)
            total_births += len(selected)

        if _sha256(sidecar.path) != sidecar.sha256:
            raise BirthMaterializationError(
                "UDC-MobileSAM sidecar changed during materialization"
            )
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "mode": "plan_only" if plan_only else "active_birth",
            "plan_only": plan_only,
            "training_free": True,
            "target_dataset_training": False,
            "external_pretraining_frozen": True,
            "online_learning": False,
            "past_only_confirmation": True,
            "minimum_distinct_views": 3,
            "gt_access": False,
            "evaluator_access": False,
            "annotation_path_argument": False,
            "native_rows_are_unchanged_prefix": True,
            "native_clip_unchanged": True,
            "coordinate_frame": "scannet_world",
            "score_mode": "constant_1.0",
            "class_mode": "inert_0_scannet_class_agnostic_evaluator",
            "scene_count": len(scenes),
            "native_count": total_native,
            "udc_receipt_count": sidecar.receipt_count,
            "birth_count": total_births,
            "frozen_policy": _policy_manifest(),
            "inputs": {
                "scene_list": os.fspath(scene_list.resolve()),
                "scene_list_sha256": _sha256(scene_list),
                "baseline_root": os.fspath(baseline_root.resolve()),
                "udc_sidecar": os.fspath(sidecar.path),
                "udc_sidecar_sha256": sidecar.sha256,
                "udc_sidecar_schema": SIDECAR_SCHEMA,
                "materializer_source": os.fspath(Path(__file__).resolve()),
                "materializer_source_sha256": _sha256(Path(__file__).resolve()),
            },
            "native_prediction_sha256": native_hashes,
            "output_prediction_sha256": output_hashes,
            "scenes": reports,
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
            raise BirthMaterializationError(
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
    parser = argparse.ArgumentParser(
        description="Materialize frozen UDC-MobileSAM births onto native Cbest"
    )
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
    parser.add_argument("--udc-sidecar", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "results/scannet_udc_mobilesam_birth_score05",
    )
    parser.add_argument("--expected-scene-count", type=int, default=100)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="validate and select all scenes without creating output files",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = materialize_scannet_udc_mobilesam_birth_full100(
        scene_list=args.scene_list,
        baseline_root=args.baseline_root,
        udc_sidecar=args.udc_sidecar,
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
                "udc_receipt_count": manifest["udc_receipt_count"],
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
