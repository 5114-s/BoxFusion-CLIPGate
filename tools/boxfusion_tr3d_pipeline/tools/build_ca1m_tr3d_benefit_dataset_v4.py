#!/usr/bin/env python3
"""Join sealed terminal-v4 evidence with CA train GT for folds 2/3/4 + 0.

This program has one fixed partition: folds 2/3/4 are fit rows and fold 0 is
threshold-dev.  Fold 1 is absent and no validation input is accepted.  Anchor
scores are taken exclusively from the B6-v2 all-fold OOF sidecar; deployment
B6 scores are never used for stacked training.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_native_b6_observer import (  # noqa: E402
    FEATURE_NAMES as NATIVE_FEATURE_NAMES,
    SCHEMA as NATIVE_EVIDENCE_SCHEMA,
)
from boxfusion.ca1m_tr3d_terminal import (  # noqa: E402
    pairwise_world_aabb_iou,
    world_aabb,
)
from boxfusion.ca1m_tr3d_terminal_gate_v4 import (  # noqa: E402
    B6_CHECKPOINT_MANIFEST_SCHEMA,
    B6_CHECKPOINT_SCHEMA,
    BENEFIT_TARGET,
    BINDING_SCHEMA,
    CANDIDATE_EVIDENCE_MANIFEST_SCHEMA,
    DATASET_SCHEMA,
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    GATE_TRAIN_FOLDS,
    GT_SHADOW_INVENTORY_SCHEMA,
    LOCKED_INTERNAL_FOLDS,
    PREREGISTRATION_SCHEMA,
    QUALITY_TARGET,
    THRESHOLD_DEV_FOLDS,
    build_terminal_gate_features_v4,
    load_oof_row_scores,
    validate_candidate_evidence_artifact,
    validate_preregistration_record,
    validate_static_config,
    write_binding_create_only,
)
from boxfusion.ca1m_tr3d_terminal_v4 import (  # noqa: E402
    load_overlay_cache,
    load_proposal_cache,
    sha256_file,
)


MANIFEST_SCHEMA = "boxfusion.ca1m_tr3d_benefit_gate_dataset_manifest.v4"
SCORE_SOURCE = "ca1m_native_b6_final_base_all_fold_oof_row_scores_v2"
SOURCE_DATASET_SCHEMA = "boxfusion.ca1m_native_b6_quality_dataset.v1"
SOURCE_DATASET_MANIFEST_SCHEMA = "boxfusion.ca1m_native_b6_quality_dataset_manifest.v1"
GT_SCHEMA = "boxfusion.ca1m_native_b6_train_scene.v1"
EVIDENCE_SUFFIX = "_ca1m_tr3d_candidate_evidence_v4.npz"


def _regular(path: Path, name: str, *, sealed: bool = True) -> Path:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink: {path}")
    result = path.resolve()
    if not result.is_file() or result.is_symlink() or result.stat().st_size <= 0:
        raise FileNotFoundError(f"missing regular {name}: {result}")
    if sealed and result.stat().st_mode & 0o222:
        raise ValueError(f"{name} must be read-only: {result}")
    return result


def _json(path: Path, name: str) -> tuple[Path, dict[str, Any]]:
    source = _regular(path, name)
    try:
        payload = json.loads(source.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain an object")
    return source, payload


def _scalar(archive: Any, name: str) -> Any:
    if name not in archive.files:
        raise ValueError(f"archive lacks scalar {name}")
    value = np.asarray(archive[name])
    if value.shape != ():
        raise ValueError(f"archive field {name} must be scalar")
    return value.item()


def _target(corners: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Official CA world-enclosing-AABB best match with strict later gates."""

    world_aabb(corners)
    world_aabb(gt)
    if not len(corners):
        return (
            np.empty((0,), np.float64),
            np.empty((0,), np.int64),
            np.empty((0, len(gt)), np.float64),
        )
    if not len(gt):
        return (
            np.zeros(len(corners), np.float64),
            np.full(len(corners), -1, np.int64),
            np.empty((len(corners), 0), np.float64),
        )
    matrix = pairwise_world_aabb_iou(corners, gt)
    matched = np.argmax(matrix, axis=1).astype(np.int64)
    return matrix[np.arange(len(corners)), matched], matched, matrix


def labeled_scene(
    *,
    anchor_corners: np.ndarray,
    anchor_scores: np.ndarray,
    candidate_corners: np.ndarray,
    candidate_rows: np.ndarray,
    anchor_indices: np.ndarray,
    gt_corners: np.ndarray,
) -> dict[str, np.ndarray]:
    """Pure target construction used by the actual builder and synthetic tests."""

    anchor_best_iou, anchor_best_gt, _ = _target(anchor_corners, gt_corners)
    selected_candidates = candidate_corners[candidate_rows]
    candidate_best_iou, candidate_best_gt, candidate_matrix = _target(
        selected_candidates, gt_corners
    )
    anchor_target = anchor_best_gt[anchor_indices]
    same_target = candidate_best_gt == anchor_target
    if len(gt_corners):
        candidate_on_anchor = candidate_matrix[
            np.arange(len(candidate_rows)), anchor_target
        ]
    else:
        candidate_on_anchor = np.zeros(len(candidate_rows), np.float64)
    anchor_for_candidate = anchor_best_iou[anchor_indices]
    gain = candidate_on_anchor - anchor_for_candidate
    return {
        "quality25_target": (candidate_best_iou > 0.25).astype(np.bool_),
        "benefit05_target": (same_target & (gain >= 0.05)).astype(np.bool_),
        "target_switch": (~same_target).astype(np.bool_),
        "anchor_best_gt_indices": anchor_target.astype(np.int64),
        "candidate_best_gt_indices": candidate_best_gt.astype(np.int64),
        "anchor_best_iou_for_candidate": anchor_for_candidate.astype(np.float64),
        "candidate_best_iou": candidate_best_iou.astype(np.float64),
        "candidate_iou_on_anchor_gt": candidate_on_anchor.astype(np.float64),
        "same_gt_iou_gain": gain.astype(np.float64),
        "baseline_best_gt_indices": anchor_best_gt.astype(np.int64),
        "baseline_best_iou": anchor_best_iou.astype(np.float64),
        "baseline_scores": np.asarray(anchor_scores, dtype=np.float32),
    }


def _candidate_evidence(
    path: Path,
    *,
    scene: str,
    expected_corners: np.ndarray,
    expected_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source = _regular(path, f"candidate evidence {scene}")
    with np.load(source, allow_pickle=False) as archive:
        fixed = {
            "schema": NATIVE_EVIDENCE_SCHEMA,
            "complete": True,
            "observer_only": True,
            "mutation_enabled": False,
            "applied_count": 0,
            "ground_truth_access": False,
            "clip_access": False,
            "scene_id": scene,
        }
        for name, expected in fixed.items():
            if _scalar(archive, name) != expected:
                raise ValueError(f"candidate evidence field {name} differs: {scene}")
        corners = np.asarray(archive["corners"])
        scores = np.asarray(archive["scores"])
        features = np.array(archive["features"], copy=True)
        valid = np.array(archive["valid_evidence"], copy=True)
        names = tuple(str(value) for value in np.asarray(archive["feature_names"]).tolist())
    count = len(expected_corners)
    if (
        not np.array_equal(corners, expected_corners)
        or not np.array_equal(scores, expected_scores)
        or names != NATIVE_FEATURE_NAMES
        or features.dtype != np.dtype(np.float32)
        or features.shape != (count, len(NATIVE_FEATURE_NAMES))
        or valid.dtype != np.dtype(np.bool_)
        or valid.shape != (count,)
        or not np.isfinite(features).all()
    ):
        raise ValueError(f"candidate evidence identity differs: {scene}")
    return features, valid


def _write_npz_create_only(path: Path, arrays: Mapping[str, Any]) -> Path:
    target = path.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing existing terminal gate v4 dataset: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, target)
        published = True
    except BaseException:
        if published:
            target.unlink(missing_ok=True)
        raise
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def _binding(config: Path, binding: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path, cfg = validate_static_config(config)
    binding_path, sealed = _json(binding, "terminal gate v4 training binding")
    if (
        sealed.get("schema") != BINDING_SCHEMA
        or sealed.get("complete") is not True
        or sealed.get("train_only") is not True
        or sealed.get("validation_ground_truth_access") is not False
        or sealed.get("validation_prediction_access") is not False
        or sealed.get("official_validation_comparable") is not False
        or (sealed.get("config") or {}).get("path") != str(config_path)
        or (sealed.get("config") or {}).get("sha256") != sha256_file(config_path)
        or sealed.get("legacy_artifact_reuse") is not False
    ):
        raise ValueError("terminal gate v4 training binding differs")
    if binding_path.stat().st_mode & 0o222:
        raise ValueError("terminal gate v4 training binding must be read-only")
    return cfg, sealed


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dataset.resolve()
    manifest_output = args.output_manifest.resolve()
    if output == manifest_output or output.exists() or manifest_output.exists():
        raise FileExistsError("refusing existing/aliased terminal gate v4 dataset outputs")
    cfg, binding = _binding(args.config, args.training_binding)
    binding_path = _regular(args.training_binding, "training binding")
    prerequisites = cfg["prerequisites"]
    preregistration_path, _ = validate_preregistration_record(
        prerequisites["preregistration_manifest"]
    )
    binding_preregistration = binding.get("preregistration_manifest") or {}
    if (
        binding_preregistration.get("path") != str(preregistration_path)
        or binding_preregistration.get("sha256") != sha256_file(preregistration_path)
        or binding_preregistration.get("schema") != PREREGISTRATION_SCHEMA
        or binding_preregistration.get("sealed_before_first_gt_join") is not True
    ):
        raise ValueError("training binding does not bind the sealed preregistration")
    preregistration_sha256 = sha256_file(preregistration_path)

    def revalidate_preregistration() -> None:
        current_path, _ = validate_preregistration_record(
            prerequisites["preregistration_manifest"]
        )
        if (
            current_path != preregistration_path
            or sha256_file(current_path) != preregistration_sha256
        ):
            raise ValueError("terminal gate v4 preregistration changed during dataset build")
    checkpoint_record = prerequisites["native_b6_v2_checkpoint"]
    checkpoint = _regular(Path(checkpoint_record["path"]), "native-B6 v2 checkpoint")
    checkpoint_manifest_record = prerequisites["native_b6_v2_checkpoint_manifest"]
    checkpoint_manifest_path, checkpoint_manifest = _json(
        Path(checkpoint_manifest_record["path"]), "native-B6 v2 checkpoint manifest"
    )
    if (
        checkpoint_record.get("schema") != B6_CHECKPOINT_SCHEMA
        or checkpoint_record.get("sha256") != sha256_file(checkpoint)
        or checkpoint_manifest_record.get("schema") != B6_CHECKPOINT_MANIFEST_SCHEMA
        or checkpoint_manifest_record.get("sha256") != sha256_file(checkpoint_manifest_path)
    ):
        raise ValueError("native-B6 v2 checkpoint records differ")
    oof, oof_manifest = load_oof_row_scores(
        prerequisites["native_b6_v2_oof_row_scores"],
        prerequisites["native_b6_v2_oof_row_scores_manifest"],
        checkpoint=checkpoint,
        checkpoint_manifest=checkpoint_manifest_path,
    )
    if binding["native_b6_v2_oof"]["manifest_sha256"] != sha256_file(
        Path(prerequisites["native_b6_v2_oof_row_scores_manifest"]["path"])
    ):
        raise ValueError("training binding does not bind the loaded OOF sidecar")

    dataset_record = checkpoint_manifest.get("dataset") or {}
    source_dataset = _regular(Path(str(dataset_record.get("path", ""))), "B6-v2 dataset")
    source_manifest_path, source_manifest = _json(
        Path(str(dataset_record.get("manifest_path", ""))), "B6-v2 dataset manifest"
    )
    if (
        dataset_record.get("sha256") != sha256_file(source_dataset)
        or dataset_record.get("manifest_sha256") != sha256_file(source_manifest_path)
        or source_manifest.get("schema") != SOURCE_DATASET_MANIFEST_SCHEMA
        or source_manifest.get("complete") is not True
        or source_manifest.get("train_only") is not True
        or source_manifest.get("validation_ground_truth_access") is not False
        or source_manifest.get("validation_prediction_access") is not False
        or (source_manifest.get("train_collection") or {}).get("schema")
        != "boxfusion.ca1m_native_b6_final_base_train_collection.v2"
        or (source_manifest.get("train_collection") or {}).get("old_native_b6_diagnostics_reused") is not False
        or (source_manifest.get("train_collection") or {}).get("old_native_b6_checkpoint_reused") is not False
    ):
        raise ValueError("B6-v2 dataset/final-base provenance differs")
    with np.load(source_dataset, allow_pickle=False) as archive:
        if _scalar(archive, "schema") != SOURCE_DATASET_SCHEMA:
            raise ValueError("unsupported B6-v2 source dataset")
        # Deliberately do not open target_iou/matched_gt_indices: fold-1 GT
        # remains sealed even though its GT-free features are in this archive.
        source = {
            name: np.array(archive[name], copy=True)
            for name in (
                "quality_features", "scene_ids", "row_indices", "prediction_scores",
                "prediction_corners", "fold_ids", "valid_evidence",
            )
        }
    if (
        not np.array_equal(source["scene_ids"].astype(str), oof["scene_ids"].astype(str))
        or not np.array_equal(source["row_indices"], oof["source_row_indices"])
        or not np.array_equal(source["prediction_scores"], oof["detector_scores"])
        or not np.array_equal(source["fold_ids"], oof["fold_ids"])
        or not np.array_equal(source["quality_features"][:, 0], oof["detector_scores"])
    ):
        raise ValueError("B6-v2 source rows and OOF row identity differ")
    if (
        str(np.asarray(oof["dataset_sha256"]).item()) != sha256_file(source_dataset)
        or str(np.asarray(oof["dataset_manifest_sha256"]).item())
        != sha256_file(source_manifest_path)
    ):
        raise ValueError("OOF sidecar does not bind the B6-v2 source dataset")

    evidence_record = prerequisites["candidate_evidence_manifest"]
    evidence_manifest_path, evidence_manifest = _json(
        Path(evidence_record["path"]), "candidate evidence v4 manifest"
    )
    if (
        evidence_record.get("schema") != CANDIDATE_EVIDENCE_MANIFEST_SCHEMA
        or evidence_record.get("sha256") != sha256_file(evidence_manifest_path)
        or evidence_manifest.get("schema") != CANDIDATE_EVIDENCE_MANIFEST_SCHEMA
        or evidence_manifest.get("complete") is not True
        or evidence_manifest.get("ground_truth_access") is not False
        or evidence_manifest.get("scene_count") != 100
    ):
        raise ValueError("candidate evidence v4 manifest differs")
    evidence_rows = evidence_manifest.get("scenes") or {}
    gt_inventory_record = prerequisites["derived_train_gt_inventory_receipt"]
    gt_inventory_path, gt_inventory = _json(
        Path(gt_inventory_record["path"]), "derived train GT shadow inventory"
    )
    binding_gt_inventory = binding.get("derived_train_gt_inventory_receipt") or {}
    if (
        gt_inventory_record.get("schema") != GT_SHADOW_INVENTORY_SCHEMA
        or gt_inventory_record.get("sha256") != sha256_file(gt_inventory_path)
        or gt_inventory.get("schema") != GT_SHADOW_INVENTORY_SCHEMA
        or gt_inventory.get("complete") is not True
        or gt_inventory.get("scene_count") != 80
        or gt_inventory.get("file_count") != 160
        or gt_inventory.get("gt_array_content_loaded") is not False
        or gt_inventory.get("shadow_files_read_only") is not True
        or binding_gt_inventory.get("path") != str(gt_inventory_path)
        or binding_gt_inventory.get("sha256") != sha256_file(gt_inventory_path)
        or binding_gt_inventory.get("schema") != GT_SHADOW_INVENTORY_SCHEMA
    ):
        raise ValueError("derived train GT shadow inventory binding differs")
    gt_inventory_rows = gt_inventory.get("scenes") or {}
    source_scene_records = {
        str(row["scene_id"]): row for row in source_manifest.get("scenes", ())
    }
    oof_scenes = np.asarray(oof["scene_ids"]).astype(str)
    oof_folds = np.asarray(oof["fold_ids"], dtype=np.int64)
    scene_fold: dict[str, int] = {}
    for scene in sorted(set(oof_scenes.tolist())):
        folds = np.unique(oof_folds[oof_scenes == scene])
        if len(folds) != 1:
            raise ValueError(f"B6-v2 OOF scene crosses folds: {scene}")
        scene_fold[scene] = int(folds[0])
    if (
        len(scene_fold) != 100
        or set(scene_fold) != set(evidence_rows)
        or set(scene_fold) != set(source_scene_records)
    ):
        raise ValueError("v4 evidence/B6-v2 OOF scene sets differ")
    selected_scenes = tuple(
        scene for scene in sorted(scene_fold)
        if scene_fold[scene] in (*GATE_TRAIN_FOLDS, *THRESHOLD_DEV_FOLDS)
    )
    if (
        len(selected_scenes) != 80
        or sum(scene_fold[s] in GATE_TRAIN_FOLDS for s in selected_scenes) != 60
        or sum(scene_fold[s] in THRESHOLD_DEV_FOLDS for s in selected_scenes) != 20
        or any(scene_fold[s] in LOCKED_INTERNAL_FOLDS for s in selected_scenes)
    ):
        raise ValueError("terminal gate v4 fit/dev split is not fixed 60/20 with fold1 sealed")

    proposal_root = Path(binding["proposal_stage_p"]["root"])
    overlay_root = Path(binding["overlay_stage_o"]["root"])
    gt_root = Path(binding["derived_train_gt_root"]).resolve()
    if (
        Path(str(gt_inventory.get("output_root", ""))).resolve() != gt_root
        or set(gt_inventory_rows) != set(selected_scenes)
    ):
        raise ValueError("derived train GT shadow split/root differs")
    candidate_parts: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "features", "scene_ids", "fold_ids", "candidate_rows", "anchor_indices",
            "candidate_corners", "candidate_scores", "candidate_valid_evidence",
            "anchor_valid_evidence", "quality25_target", "benefit05_target",
            "target_switch", "anchor_best_gt_indices", "candidate_best_gt_indices",
            "anchor_best_iou", "candidate_best_iou", "candidate_iou_on_anchor_gt",
            "same_gt_iou_gain",
        )
    }
    baseline_parts: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "scene_ids", "fold_ids", "row_indices", "corners", "scores",
            "best_gt_indices", "best_iou",
        )
    }
    per_scene: dict[str, Any] = {}
    gt_counts: list[int] = []
    for scene in selected_scenes:
        source_rows = np.flatnonzero(oof_scenes == scene)
        canonical_rows = np.asarray(source["row_indices"])[source_rows]
        if not np.array_equal(canonical_rows, np.arange(len(source_rows), dtype=np.int64)):
            raise ValueError(f"B6-v2 source rows are not canonical: {scene}")
        proposal_path = proposal_root / f"{scene}_ca1m_tr3d_proposals_v4.npz"
        overlay_path = overlay_root / f"{scene}_ca1m_tr3d_overlay_v4.npz"
        proposal = load_proposal_cache(proposal_path, expected_scene=scene)
        overlay = load_overlay_cache(
            overlay_path,
            expected_scene=scene,
            expected_proposal_sha256=sha256_file(proposal_path),
        )
        evidence_row = evidence_rows[scene]
        evidence_path = validate_candidate_evidence_artifact(
            Path(str(evidence_row.get("path", ""))),
            scene=scene,
            expected_sha256=str(evidence_row.get("sha256", "")),
            expected_root=Path(str(
                (evidence_manifest.get("source_roots") or {}).get("evidence", "")
            )).resolve(),
        )
        if (
            evidence_row.get("proposal_sha256") != sha256_file(proposal_path)
            or evidence_row.get("overlay_sha256") != sha256_file(overlay_path)
        ):
            raise ValueError(f"candidate evidence P/O binding differs: {scene}")
        candidate_native, candidate_valid = _candidate_evidence(
            evidence_path,
            scene=scene,
            expected_corners=proposal["candidate_corners_world"],
            expected_scores=proposal["candidate_scores"],
        )
        anchor_corners = np.asarray(source["prediction_corners"])[source_rows]
        anchor_native = np.asarray(source["quality_features"])[source_rows]
        anchor_detector_scores = np.asarray(source["prediction_scores"])[source_rows]
        anchor_valid = np.asarray(source["valid_evidence"])[source_rows]
        oof_scores = np.asarray(oof["deployment_blend_oof_scores"], dtype=np.float32)[source_rows]
        if not np.array_equal(anchor_corners, overlay["anchor_corners"]):
            raise ValueError(f"final-base/B6-v2 geometry differs from overlay: {scene}")
        batch = build_terminal_gate_features_v4(
            proposal=proposal,
            overlay=overlay,
            anchor_native_evidence=anchor_native,
            anchor_native_detector_scores=anchor_detector_scores,
            candidate_native_evidence=candidate_native,
            anchor_scores=oof_scores,
            score_source=SCORE_SOURCE,
        )
        # This is the first GT access in the stage, after every upstream
        # artifact and OOF identity for this exact scene has been verified.
        # Re-hash preregistration plus all three executable sources immediately
        # before every GT open to fail closed on code drift during the join.
        revalidate_preregistration()
        gt_inventory_row = gt_inventory_rows[scene]
        gt_box_record = gt_inventory_row.get("box") or {}
        gt_manifest_record = gt_inventory_row.get("manifest") or {}
        gt_path = _regular(Path(str(gt_box_record.get("path", ""))), f"train GT {scene}")
        gt_manifest_path, gt_manifest = _json(
            Path(str(gt_manifest_record.get("path", ""))), f"train GT manifest {scene}"
        )
        gt = np.asarray(np.load(gt_path, allow_pickle=False))
        source_record = source_scene_records[scene]
        if (
            gt_inventory_row.get("fold_id") != scene_fold[scene]
            or gt_path != gt_root / scene / "derived_train_gt_boxes.npy"
            or gt_manifest_path != gt_root / scene / "derived_train_gt_manifest.json"
            or gt_box_record.get("sha256") != sha256_file(gt_path)
            or gt_manifest_record.get("sha256") != sha256_file(gt_manifest_path)
            or gt.dtype != np.dtype(np.float64)
            or gt.ndim != 3
            or gt.shape[1:] != (8, 3)
            or not np.isfinite(gt).all()
            or gt_manifest.get("schema") != GT_SCHEMA
            or gt_manifest.get("scene_id") != scene
            or gt_manifest.get("train_only") is not True
            or gt_manifest.get("official_validation_comparable") is not False
            or source_record.get("derived_gt_sha256") != sha256_file(gt_path)
            or (source_record.get("derived_gt_manifest") or {}).get("sha256")
            != sha256_file(gt_manifest_path)
            or source_record.get("fold_id") != scene_fold[scene]
        ):
            raise ValueError(f"derived CA train GT provenance differs: {scene}")
        targets = labeled_scene(
            anchor_corners=anchor_corners,
            anchor_scores=oof_scores,
            candidate_corners=proposal["candidate_corners_world"],
            candidate_rows=batch.candidate_rows,
            anchor_indices=batch.anchor_indices,
            gt_corners=gt,
        )
        count = len(batch.candidate_rows)
        candidate_parts["features"].append(batch.features)
        candidate_parts["scene_ids"].append(np.full(count, scene, dtype="U8"))
        candidate_parts["fold_ids"].append(np.full(count, scene_fold[scene], np.int8))
        candidate_parts["candidate_rows"].append(batch.candidate_rows)
        candidate_parts["anchor_indices"].append(batch.anchor_indices)
        candidate_parts["candidate_corners"].append(
            proposal["candidate_corners_world"][batch.candidate_rows]
        )
        candidate_parts["candidate_scores"].append(batch.candidate_scores)
        candidate_parts["candidate_valid_evidence"].append(candidate_valid[batch.candidate_rows])
        candidate_parts["anchor_valid_evidence"].append(anchor_valid[batch.anchor_indices])
        for key in (
            "quality25_target", "benefit05_target", "target_switch",
            "anchor_best_gt_indices", "candidate_best_gt_indices",
            "candidate_best_iou", "candidate_iou_on_anchor_gt", "same_gt_iou_gain",
        ):
            candidate_parts[key].append(targets[key])
        candidate_parts["anchor_best_iou"].append(targets["anchor_best_iou_for_candidate"])
        anchor_count = len(anchor_corners)
        baseline_parts["scene_ids"].append(np.full(anchor_count, scene, dtype="U8"))
        baseline_parts["fold_ids"].append(np.full(anchor_count, scene_fold[scene], np.int8))
        baseline_parts["row_indices"].append(np.arange(anchor_count, dtype=np.int64))
        baseline_parts["corners"].append(anchor_corners)
        baseline_parts["scores"].append(oof_scores)
        baseline_parts["best_gt_indices"].append(targets["baseline_best_gt_indices"])
        baseline_parts["best_iou"].append(targets["baseline_best_iou"])
        gt_counts.append(len(gt))
        per_scene[scene] = {
            "fold_id": scene_fold[scene],
            "candidate_rows": count,
            "anchor_rows": anchor_count,
            "gt_boxes": len(gt),
            "quality25_positive": int(np.count_nonzero(targets["quality25_target"])),
            "benefit05_positive": int(np.count_nonzero(targets["benefit05_target"])),
            "proposal_sha256": sha256_file(proposal_path),
            "overlay_sha256": sha256_file(overlay_path),
            "candidate_evidence_sha256": evidence_row["sha256"],
            "derived_gt_sha256": sha256_file(gt_path),
            "oof_row_count": anchor_count,
        }

    revalidate_preregistration()

    def concat(values: list[np.ndarray], shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        if not values:
            return np.empty(shape, dtype=dtype)
        return np.ascontiguousarray(np.concatenate(values, axis=0), dtype=dtype)

    arrays: dict[str, Any] = {
        "schema": np.asarray(DATASET_SCHEMA),
        "complete": np.asarray(True, np.bool_),
        "train_only": np.asarray(True, np.bool_),
        "validation_ground_truth_access": np.asarray(False, np.bool_),
        "validation_prediction_access": np.asarray(False, np.bool_),
        "official_validation_comparable": np.asarray(False, np.bool_),
        "locked_internal_fold1_gt_access": np.asarray(False, np.bool_),
        "feature_schema": np.asarray(FEATURE_SCHEMA),
        "feature_names": np.asarray(FEATURE_NAMES),
        "quality_target_schema": np.asarray(QUALITY_TARGET),
        "benefit_target_schema": np.asarray(BENEFIT_TARGET),
        "anchor_score_source": np.asarray(SCORE_SOURCE),
        "deploy_b6_scores_used_for_stacked_training": np.asarray(False, np.bool_),
        "training_binding_sha256": np.asarray(sha256_file(binding_path)),
        "preregistration_manifest_sha256": np.asarray(preregistration_sha256),
        "derived_train_gt_inventory_receipt_sha256": np.asarray(
            sha256_file(gt_inventory_path)
        ),
        "features": concat(candidate_parts["features"], (0, len(FEATURE_NAMES)), np.float32),
        "scene_ids": concat(candidate_parts["scene_ids"], (0,), "U8"),
        "fold_ids": concat(candidate_parts["fold_ids"], (0,), np.int8),
        "candidate_rows": concat(candidate_parts["candidate_rows"], (0,), np.int64),
        "anchor_indices": concat(candidate_parts["anchor_indices"], (0,), np.int64),
        "candidate_corners": concat(candidate_parts["candidate_corners"], (0, 8, 3), np.float32),
        "candidate_scores": concat(candidate_parts["candidate_scores"], (0,), np.float32),
        "candidate_valid_evidence": concat(candidate_parts["candidate_valid_evidence"], (0,), np.bool_),
        "anchor_valid_evidence": concat(candidate_parts["anchor_valid_evidence"], (0,), np.bool_),
        "quality25_target": concat(candidate_parts["quality25_target"], (0,), np.bool_),
        "benefit05_target": concat(candidate_parts["benefit05_target"], (0,), np.bool_),
        "target_switch": concat(candidate_parts["target_switch"], (0,), np.bool_),
        "anchor_best_gt_indices": concat(candidate_parts["anchor_best_gt_indices"], (0,), np.int64),
        "candidate_best_gt_indices": concat(candidate_parts["candidate_best_gt_indices"], (0,), np.int64),
        "anchor_best_iou": concat(candidate_parts["anchor_best_iou"], (0,), np.float64),
        "candidate_best_iou": concat(candidate_parts["candidate_best_iou"], (0,), np.float64),
        "candidate_iou_on_anchor_gt": concat(candidate_parts["candidate_iou_on_anchor_gt"], (0,), np.float64),
        "same_gt_iou_gain": concat(candidate_parts["same_gt_iou_gain"], (0,), np.float64),
        "baseline_scene_ids": concat(baseline_parts["scene_ids"], (0,), "U8"),
        "baseline_fold_ids": concat(baseline_parts["fold_ids"], (0,), np.int8),
        "baseline_row_indices": concat(baseline_parts["row_indices"], (0,), np.int64),
        "baseline_corners": concat(baseline_parts["corners"], (0, 8, 3), np.float32),
        "baseline_scores": concat(baseline_parts["scores"], (0,), np.float32),
        "baseline_best_gt_indices": concat(baseline_parts["best_gt_indices"], (0,), np.int64),
        "baseline_best_iou": concat(baseline_parts["best_iou"], (0,), np.float64),
        "scene_table": np.asarray(selected_scenes, dtype="U8"),
        "scene_fold_ids": np.asarray([scene_fold[s] for s in selected_scenes], np.int8),
        "scene_gt_counts": np.asarray(gt_counts, np.int64),
    }
    if (
        not len(arrays["features"])
        or not np.isfinite(arrays["features"]).all()
        or set(np.unique(arrays["fold_ids"]).tolist()) != {0, 2, 3, 4}
        or 1 in set(arrays["scene_fold_ids"].tolist())
    ):
        raise ValueError("constructed terminal gate v4 fit/dev dataset is invalid")
    dataset_path = _write_npz_create_only(output, arrays)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "complete": True,
        "train_only": True,
        "partition": "fit_folds234_threshold_dev_fold0",
        "scene_count": 80,
        "fit_scene_count": 60,
        "threshold_dev_scene_count": 20,
        "locked_internal_scene_count_accessed": 0,
        "fold_ids": [0, 2, 3, 4],
        "locked_internal_fold_ids": [1],
        "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "official_validation_comparable": False,
        "ground_truth_join_after_candidate_seal": True,
        "locked_internal_fold1_gt_access": False,
        "source_native_b6_target_arrays_opened": False,
        "anchor_score_source": SCORE_SOURCE,
        "deploy_b6_scores_used_for_stacked_training": False,
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "quality_target": QUALITY_TARGET,
        "benefit_target": BENEFIT_TARGET,
        "candidate_rows": len(arrays["features"]),
        "baseline_rows": len(arrays["baseline_scores"]),
        "quality25_positive": int(np.count_nonzero(arrays["quality25_target"])),
        "benefit05_positive": int(np.count_nonzero(arrays["benefit05_target"])),
        "dataset": {"path": str(dataset_path), "sha256": sha256_file(dataset_path)},
        "training_binding": {"path": str(binding_path), "sha256": sha256_file(binding_path)},
        "preregistration_manifest": {
            "path": str(preregistration_path),
            "sha256": preregistration_sha256,
            "schema": PREREGISTRATION_SCHEMA,
            "sealed_before_first_gt_join": True,
        },
        "derived_train_gt_inventory_receipt": {
            "path": str(gt_inventory_path),
            "sha256": sha256_file(gt_inventory_path),
            "schema": GT_SHADOW_INVENTORY_SCHEMA,
            "scene_count": 80,
            "file_count": 160,
        },
        "b6_v2_source_dataset": {"path": str(source_dataset), "sha256": sha256_file(source_dataset)},
        "b6_v2_oof_manifest_sha256": sha256_file(
            Path(prerequisites["native_b6_v2_oof_row_scores_manifest"]["path"])
        ),
        "candidate_evidence_manifest_sha256": sha256_file(evidence_manifest_path),
        "official_ca_iou": "world_enclosing_aabb_strict_gt",
        "legacy_artifact_reuse": False,
        "source_code_sha256": sha256_file(Path(__file__).resolve()),
        "per_scene": per_scene,
    }
    try:
        write_binding_create_only(manifest_output, manifest)
    except BaseException:
        dataset_path.unlink(missing_ok=True)
        raise
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", type=Path, required=True)
    value.add_argument("--training-binding", type=Path, required=True)
    value.add_argument("--output-dataset", type=Path, required=True)
    value.add_argument("--output-manifest", type=Path, required=True)
    return value


def main() -> int:
    report = run(parser().parse_args())
    print(json.dumps({
        "complete": report["complete"],
        "scene_count": report["scene_count"],
        "candidate_rows": report["candidate_rows"],
        "quality25_positive": report["quality25_positive"],
        "benefit05_positive": report["benefit05_positive"],
        "locked_internal_fold1_gt_access": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
