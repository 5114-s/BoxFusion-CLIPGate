#!/usr/bin/env python3
"""Train and gate the CA-1M-native quality head using train-only scene folds.

This wrapper reuses the deterministic IoU-aware MLP fitter from
``train_quality_calibrator.py`` with the native 14-column feature dimension.
It never creates a random sample split: the dataset's five scene folds are
independently recomputed, fold zero remains an untouched 20% development set,
and the deployable checkpoint is the fold-zero model trained only on folds
1--4.  All-fold OOF predictions are diagnostic; activation is authorized only
by frozen train-only development/OOF rules, never CA-1M validation data.

Default ``--preflight`` performs all provenance and split checks but does not
import torch, fit a model, or create output files.
"""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import platform
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from build_ca1m_native_b6_dataset import (
    DATASET_SCHEMA,
    DEFAULT_SPLIT_NAMESPACE,
    FINAL_BASE_COLLECTION_SCHEMA,
    FINAL_BASE_IDENTITY_SCHEMA,
    MANIFEST_SCHEMA as DATASET_MANIFEST_SCHEMA,
    SUPPORTED_COLLECTION_SCHEMAS,
    TARGET_SCHEMA,
    assign_scene_folds,
    sha256_file,
    validate_fixed10_paired_report_record,
)
from train_quality_calibrator import fit_iou_aware_mlp


CHECKPOINT_SCHEMA = "boxfusion.ca1m_native_b6_iou_mlp.v1"
MANIFEST_SCHEMA = "boxfusion.ca1m_native_b6_checkpoint_manifest.v1"
OOF_ROW_SCORE_SCHEMA = "boxfusion.ca1m_native_b6_oof_row_scores.v2"
OOF_ROW_SCORE_MANIFEST_SCHEMA = (
    "boxfusion.ca1m_native_b6_oof_row_scores_manifest.v2"
)
OUTPUT_NAMES = ("predicted_iou", "prob_iou_015", "prob_iou_025", "prob_iou_050")
IOU_THRESHOLDS = (0.15, 0.25, 0.50)
DEFAULT_RANKING_WEIGHTS = np.asarray((0.10, 0.20, 0.30, 0.40), dtype=np.float64)


def _sha256_arrays(items: Sequence[tuple[str, np.ndarray]]) -> str:
    """Hash named arrays without depending on NPZ ZIP timestamps."""

    digest = hashlib.sha256()
    for name, raw in items:
        value = np.ascontiguousarray(np.asarray(raw))
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def _fold_model_sha256(
    model: tuple[
        tuple[np.ndarray, ...], tuple[np.ndarray, ...], np.ndarray, np.ndarray
    ]
) -> str:
    weights, biases, feature_mean, feature_scale = model
    arrays: list[tuple[str, np.ndarray]] = []
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        arrays.extend(((f"weight_{index}", weight), (f"bias_{index}", bias)))
    arrays.extend((("feature_mean", feature_mean), ("feature_scale", feature_scale)))
    return _sha256_arrays(arrays)


def _oof_recipe(args: argparse.Namespace, ranking_weights: np.ndarray) -> dict[str, Any]:
    return {
        "schema": "boxfusion.ca1m_native_b6_oof_recipe.v2",
        "fit": "fit_iou_aware_mlp",
        "fold_count": 5,
        "heldout_rule": "model_fold_k_trained_on_all_scene_folds_except_k",
        "hidden_dims": [int(value) for value in args.hidden_dims],
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "l2_weight": float(args.l2_weight),
        "iou_loss_weight": float(args.iou_loss_weight),
        "threshold_loss_weight": float(args.threshold_loss_weight),
        "monotonic_loss_weight": float(args.monotonic_loss_weight),
        "base_seed": int(args.seed),
        "fold_seed_formula": "base_seed_plus_heldout_fold",
        "ranking_weights": [float(value) for value in ranking_weights],
        "detector_blend": float(args.detector_blend),
        "quality_blend": float(1.0 - args.detector_blend),
        "monotonic_probability_projection": True,
        "preserve_original_floor": False,
    }


def _oof_sidecar(
    *,
    values: Mapping[str, np.ndarray],
    dataset_path: Path,
    dataset_manifest_path: Path,
    dataset_manifest: Mapping[str, Any],
    models: Mapping[int, tuple[
        tuple[np.ndarray, ...], tuple[np.ndarray, ...], np.ndarray, np.ndarray
    ]],
    oof_outputs: np.ndarray,
    oof_components: np.ndarray,
    quality_oof_scores: np.ndarray,
    deployment_oof_scores: np.ndarray,
    ranking_weights: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build immutable, row-identical all-fold OOF score provenance."""

    scenes = np.asarray(values["scene_ids"], dtype=np.str_)
    folds = np.asarray(values["fold_ids"], dtype=np.int8)
    source_rows = np.asarray(values["row_indices"], dtype=np.int64)
    detector = np.asarray(values["prediction_scores"], dtype=np.float32)
    count = len(scenes)
    if (
        folds.shape != (count,)
        or source_rows.shape != (count,)
        or detector.shape != (count,)
        or oof_outputs.shape != (count, 4)
        or oof_components.shape != (count, 4)
        or quality_oof_scores.shape != (count,)
        or deployment_oof_scores.shape != (count,)
        or set(models) != set(range(5))
        or np.any((folds < 0) | (folds > 4))
    ):
        raise ValueError("OOF row-score inputs violate the five-fold row contract")
    if any(
        not np.isfinite(value).all()
        for value in (
            detector, oof_outputs, oof_components,
            quality_oof_scores, deployment_oof_scores,
        )
    ):
        raise ValueError("OOF row scores must be finite")

    manifest_scenes = tuple(
        str(row["scene_id"]) for row in dataset_manifest.get("scenes", ())
    )
    if len(manifest_scenes) != len(set(manifest_scenes)):
        raise ValueError("OOF source manifest has duplicate scenes")
    split_rows = {
        str(row["scene_id"]): int(row["fold_id"])
        for row in dataset_manifest.get("scenes", ())
    }
    if set(split_rows) != set(manifest_scenes) or set(split_rows.values()) != set(range(5)):
        raise ValueError("OOF source manifest lacks the exact five scene folds")
    if any(split_rows[str(scene)] != int(fold) for scene, fold in zip(scenes, folds)):
        raise ValueError("OOF row folds differ from the scene-level split")
    fold_model_hashes = tuple(_fold_model_sha256(models[fold]) for fold in range(5))
    fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        heldout_scene_ids = tuple(
            scene for scene in manifest_scenes if split_rows[scene] == fold
        )
        training_scene_ids = tuple(
            scene for scene in manifest_scenes if scene not in set(heldout_scene_ids)
        )
        if not heldout_scene_ids or set(heldout_scene_ids) & set(training_scene_ids):
            raise ValueError("OOF fold scene exclusion proof is incomplete")
        heldout_mask = folds == fold
        if np.any(scenes[heldout_mask] == ""):
            raise ValueError("OOF held-out row has an empty scene id")
        fold_rows.append(
            {
                "heldout_fold": fold,
                "model_sha256": fold_model_hashes[fold],
                "seed": int(args.seed + fold),
                "heldout_scene_ids": list(heldout_scene_ids),
                "training_scene_ids": list(training_scene_ids),
                "heldout_scene_count": len(heldout_scene_ids),
                "training_scene_count": len(training_scene_ids),
                "heldout_row_count": int(np.count_nonzero(heldout_mask)),
                "training_excludes_every_heldout_scene": True,
            }
        )

    recipe = _oof_recipe(args, ranking_weights)
    recipe_json = json.dumps(recipe, separators=(",", ":"), sort_keys=True)
    recipe_sha = hashlib.sha256(recipe_json.encode("utf-8")).hexdigest()
    dataset_sha = sha256_file(dataset_path)
    dataset_manifest_sha = sha256_file(dataset_manifest_path)
    arrays = {
        "schema": np.asarray(OOF_ROW_SCORE_SCHEMA),
        "complete": np.asarray(True, dtype=np.bool_),
        "train_only": np.asarray(True, dtype=np.bool_),
        "scene_group_oof": np.asarray(True, dtype=np.bool_),
        "validation_ground_truth_access": np.asarray(False, dtype=np.bool_),
        "validation_prediction_access": np.asarray(False, dtype=np.bool_),
        "official_validation_comparable": np.asarray(False, dtype=np.bool_),
        "each_row_model_excludes_scene": np.asarray(True, dtype=np.bool_),
        "fold_count": np.asarray(5, dtype=np.int8),
        "dataset_sha256": np.asarray(dataset_sha),
        "dataset_manifest_sha256": np.asarray(dataset_manifest_sha),
        "split_namespace": np.asarray(values["split_namespace"]),
        "feature_names": np.asarray(values["feature_names"], dtype=np.str_),
        "scene_ids": scenes,
        "fold_ids": folds,
        "heldout_model_fold_ids": folds.copy(),
        "source_row_indices": source_rows,
        "dataset_row_positions": np.arange(count, dtype=np.int64),
        "detector_scores": detector,
        "raw_oof_outputs": np.asarray(oof_outputs, dtype=np.float32),
        "monotonic_oof_components": np.asarray(oof_components, dtype=np.float32),
        "quality_oof_scores": np.asarray(quality_oof_scores, dtype=np.float32),
        "deployment_blend_oof_scores": np.asarray(
            deployment_oof_scores, dtype=np.float32
        ),
        "fold_model_sha256": np.asarray(fold_model_hashes, dtype=np.str_),
        "recipe_json": np.asarray(recipe_json),
        "recipe_sha256": np.asarray(recipe_sha),
    }
    manifest = {
        "schema": OOF_ROW_SCORE_MANIFEST_SCHEMA,
        "complete": True,
        "train_only": True,
        "scene_group_oof": True,
        "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "official_validation_comparable": False,
        "each_row_model_excludes_scene": True,
        "row_identity": ["scene_ids", "fold_ids", "source_row_indices"],
        "row_count": count,
        "scene_count": len(manifest_scenes),
        "dataset": {
            "path": str(dataset_path),
            "sha256": dataset_sha,
            "manifest_path": str(dataset_manifest_path),
            "manifest_sha256": dataset_manifest_sha,
            "schema": DATASET_SCHEMA,
            "manifest_schema": DATASET_MANIFEST_SCHEMA,
        },
        "split": {
            "namespace": str(np.asarray(values["split_namespace"]).item()),
            "fold_count": 5,
            "all_fold_oof": True,
            "gate_train_folds": [2, 3, 4],
            "threshold_dev_folds": [0],
            "locked_internal_check_folds": [1],
            "folds": fold_rows,
        },
        "scores": {
            "quality": "monotonic_projected_quality_rank_from_heldout_fold_model",
            "deployment_blend": (
                "detector_blend*detector_score + "
                "quality_blend*heldout_fold_quality_score"
            ),
            "detector_blend": float(args.detector_blend),
            "quality_blend": float(1.0 - args.detector_blend),
        },
        "fold_model_sha256": list(fold_model_hashes),
        "recipe": recipe,
        "recipe_sha256": recipe_sha,
    }
    return arrays, manifest


def _create_only(path: Path, data: bytes) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    published_identity: tuple[int, int] | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        published = path.stat(follow_symlinks=False)
        published_identity = (published.st_dev, published.st_ino)
        path.chmod(0o444)
        return published_identity
    except FileExistsError as error:
        raise FileExistsError(f"refusing to replace immutable artifact: {path}") from error
    except BaseException:
        if published_identity is not None:
            try:
                current = path.stat(follow_symlinks=False)
                if (
                    not path.is_symlink()
                    and (current.st_dev, current.st_ino) == published_identity
                    and hashlib.sha256(path.read_bytes()).hexdigest()
                    == hashlib.sha256(data).hexdigest()
                ):
                    path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _publish_transaction(artifacts: Sequence[tuple[Path, bytes]]) -> None:
    """Publish an all-owned bundle or roll back only inode-proven members."""

    paths = tuple(path for path, _ in artifacts)
    if len(paths) != len(set(paths)):
        raise ValueError("transaction output paths must be distinct")
    if any(path.exists() or path.is_symlink() for path in paths):
        raise FileExistsError("transaction output already exists")
    owned: list[tuple[Path, tuple[int, int], str]] = []
    try:
        for path, data in artifacts:
            identity = _create_only(path, data)
            owned.append((path, identity, hashlib.sha256(data).hexdigest()))
    except BaseException:
        for path, identity, expected_sha in reversed(owned):
            try:
                current = path.stat(follow_symlinks=False)
                if (
                    not path.is_symlink()
                    and (current.st_dev, current.st_ino) == identity
                    and hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha
                ):
                    path.unlink()
            except FileNotFoundError:
                pass
        raise


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _scalar(archive: Mapping[str, np.ndarray], name: str, expected: Any) -> None:
    value = np.asarray(archive[name])
    if value.shape != () or value.item() != expected:
        raise ValueError(f"dataset scalar {name} disagrees with {expected!r}")


def load_dataset(dataset_path: Path, manifest_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    dataset_path = Path(dataset_path)
    manifest_path = Path(manifest_path)
    if dataset_path.is_symlink() or manifest_path.is_symlink():
        raise ValueError("native-B6 dataset/manifest must not be symlinks")
    dataset_path = dataset_path.resolve()
    manifest_path = manifest_path.resolve()
    if not dataset_path.is_file() or dataset_path.is_symlink():
        raise ValueError(f"invalid native-B6 dataset: {dataset_path}")
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"invalid native-B6 dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != DATASET_MANIFEST_SCHEMA:
        raise ValueError("unsupported native-B6 dataset manifest schema")
    for key, expected in (
        ("complete", True), ("train_only", True),
        ("validation_ground_truth_access", False),
        ("validation_prediction_access", False),
        ("validation_scene_overlap_count", 0),
        ("activation_authorized", False), ("training_started", False),
    ):
        if manifest.get(key) != expected:
            raise ValueError(f"dataset manifest safety field {key} disagrees")
    record = manifest.get("dataset") or {}
    if Path(str(record.get("path", ""))).resolve() != dataset_path:
        raise ValueError("dataset path differs from its manifest")
    if str(record.get("sha256")) != sha256_file(dataset_path):
        raise ValueError("dataset SHA256 differs from its manifest")
    target_contract = manifest.get("target") or {}
    if (
        target_contract.get("schema") != TARGET_SCHEMA
        or target_contract.get("matches_evaluation")
        != "evaluation/eval_ca1m.py -> box3d_iou_v2"
        or target_contract.get("yaw_obb_iou_is_primary") is not False
        or target_contract.get("official_validation_comparable") is not False
    ):
        raise ValueError("dataset evaluator target contract disagrees")
    collection = manifest.get("train_collection") or {}
    collection_path = Path(str(collection.get("path", "")))
    if collection_path.is_symlink() or not collection_path.is_file():
        raise ValueError("dataset train collection manifest is missing/unsafe")
    collection_path = collection_path.resolve()
    collection_schema = str(collection.get("schema", ""))
    if (
        collection_schema not in SUPPORTED_COLLECTION_SCHEMAS
        or collection.get("evaluation_invoked") is not False
        or str(collection.get("sha256")) != sha256_file(collection_path)
    ):
        raise ValueError("dataset train collection provenance disagrees")
    collection_payload = json.loads(collection_path.read_text())
    if collection_payload.get("schema") != collection_schema:
        raise ValueError("dataset train collection schema differs from its source")
    if collection_schema == FINAL_BASE_COLLECTION_SCHEMA:
        for key, expected in (
            ("geometry_authority", "sealed_final_base_prediction"),
            ("offline_direct_observer", True),
            ("cross_run_boxfusion_replay_invoked", False),
            ("cross_run_exact_identity_required", False),
            ("old_native_b6_diagnostics_reused", False),
            ("old_native_b6_checkpoint_reused", False),
        ):
            if collection.get(key) != expected or collection_payload.get(key) != expected:
                raise ValueError(f"final-base native-B6 collection field {key} disagrees")
        modules = collection.get("source_modules") or {}
        if modules != {
            "selective_boxer_g0": True,
            "clip_appearance_gate": True,
            "reliable_view_top_k": 3,
            "b6_evidence_top_k": 5,
        }:
            raise ValueError("final-base native-B6 source modules disagree")
        source_record = collection.get("source_final_base_collection") or {}
        source_path = Path(str(source_record.get("path", "")))
        if source_path.is_symlink() or not source_path.is_file():
            raise ValueError("final-base source collection is missing/unsafe")
        source_path = source_path.resolve()
        if (
            source_record.get("schema") != FINAL_BASE_IDENTITY_SCHEMA
            or source_record.get("sha256") != sha256_file(source_path)
        ):
            raise ValueError("final-base source collection provenance disagrees")
        source_payload = json.loads(source_path.read_text())
        for key, expected in {
            "schema": FINAL_BASE_IDENTITY_SCHEMA,
            "ok": True,
            "dataset": "CA1M",
            "split": "train100",
            "ground_truth_access": False,
            "evaluation_invoked": False,
            "training_invoked": False,
            "scannet_learned_b6_or_gate_reused": False,
            "clip_appearance_gate_active": True,
            "reliable_view_top_k": 3,
        }.items():
            if source_payload.get(key) != expected:
                raise ValueError(f"final-base source collection field {key} disagrees")
        paired_record = collection.get("fixed10_paired_report") or {}
        paired_payload, _ = validate_fixed10_paired_report_record(paired_record)
        if paired_record != (collection_payload.get("fixed10_paired_report") or {}):
            raise ValueError("dataset fixed10 paired report provenance disagrees")
        if paired_payload.get("positive_map_at_all_thresholds") is not True:
            raise ValueError("fixed10 paired report did not authorize native-B6 retraining")

    with np.load(dataset_path, allow_pickle=False) as archive:
        required = {
            "schema", "complete", "train_only", "validation_ground_truth_access",
            "target_schema", "quality_features", "feature_names", "target_iou",
            "scene_ids", "fold_ids", "dev_mask", "prediction_scores",
            "matched_gt_indices", "row_indices", "fold_count", "dev_fold",
            "split_namespace",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"native-B6 dataset fields missing: {sorted(missing)}")
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    _scalar(values, "schema", DATASET_SCHEMA)
    _scalar(values, "complete", True)
    _scalar(values, "train_only", True)
    _scalar(values, "validation_ground_truth_access", False)
    _scalar(values, "target_schema", TARGET_SCHEMA)
    _scalar(values, "fold_count", 5)
    _scalar(values, "dev_fold", 0)
    features = values["quality_features"]
    targets = values["target_iou"]
    names = tuple(str(item) for item in values["feature_names"].tolist())
    count = len(features)
    if features.ndim != 2 or features.shape[1] != len(names) or count < 2:
        raise ValueError("native-B6 quality_features have invalid shape")
    if not names or len(names) != len(set(names)):
        raise ValueError("native-B6 feature names are empty/duplicate")
    if targets.shape != (count,) or values["scene_ids"].shape != (count,):
        raise ValueError("native-B6 target/scene rows are misaligned")
    if values["fold_ids"].shape != (count,) or values["dev_mask"].shape != (count,):
        raise ValueError("native-B6 split rows are misaligned")
    if (
        values["prediction_scores"].shape != (count,)
        or values["matched_gt_indices"].shape != (count,)
        or values["row_indices"].shape != (count,)
    ):
        raise ValueError("native-B6 detector score rows are misaligned")
    numeric = (features, targets, values["prediction_scores"])
    if any(not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all() for value in numeric):
        raise ValueError("native-B6 dataset contains non-finite/non-numeric arrays")
    if any(np.any((value < 0.0) | (value > 1.0)) for value in numeric):
        raise ValueError("native-B6 features/targets/scores must lie in [0,1]")
    scenes = np.asarray(values["scene_ids"], dtype=np.str_)
    scene_records = manifest.get("scenes", [])
    if not isinstance(scene_records, list):
        raise ValueError("dataset manifest scenes are invalid")
    unique_scenes = tuple(str(row.get("scene_id")) for row in scene_records)
    if not unique_scenes or len(unique_scenes) != len(set(unique_scenes)):
        raise ValueError("dataset manifest scenes are empty/duplicate")
    if not set(np.unique(scenes).tolist()).issubset(set(unique_scenes)):
        raise ValueError("dataset rows contain a scene absent from the manifest")
    gt_counts: dict[str, int] = {}
    for row in scene_records:
        scene = str(row.get("scene_id"))
        count_value = row.get("gt_boxes")
        if isinstance(count_value, bool) or not isinstance(count_value, int) or count_value < 1:
            raise ValueError(f"dataset manifest GT count is invalid for {scene}")
        gt_counts[scene] = count_value
    raw_matched = np.asarray(values["matched_gt_indices"])
    if not np.issubdtype(raw_matched.dtype, np.integer):
        raise ValueError("dataset matched GT indices must be integers")
    matched = raw_matched.astype(np.int64, copy=False)
    for row, scene in enumerate(scenes):
        if matched[row] < 0 or matched[row] >= gt_counts[str(scene)]:
            raise ValueError("dataset matched GT index is out of range")
    for scene in np.unique(scenes):
        scene_rows = np.flatnonzero(scenes == scene)
        if not np.array_equal(values["row_indices"][scene_rows], np.arange(len(scene_rows))):
            raise ValueError(f"dataset row indices are not contiguous for {scene}")
    namespace = str(np.asarray(values["split_namespace"]).item())
    if collection_schema == FINAL_BASE_COLLECTION_SCHEMA and namespace != DEFAULT_SPLIT_NAMESPACE:
        raise ValueError("final-base native-B6 dataset changed the established CA folds")
    expected_folds = assign_scene_folds(unique_scenes, namespace, fold_count=5)
    observed_folds = np.asarray(values["fold_ids"], dtype=np.int64)
    recomputed = np.asarray([expected_folds[scene] for scene in scenes], dtype=np.int64)
    if not np.array_equal(observed_folds, recomputed):
        raise ValueError("dataset scene-fold assignment is not deterministic")
    if not np.array_equal(values["dev_mask"], observed_folds == 0):
        raise ValueError("dataset dev_mask does not identify fold zero")
    split = manifest.get("split") or {}
    if (
        split.get("kind") != "deterministic_scene_grouped_5fold"
        or split.get("namespace") != namespace
        or split.get("dev_fold") != 0
        or split.get("train_folds") != [1, 2, 3, 4]
    ):
        raise ValueError("dataset manifest split contract disagrees")
    return values, manifest


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    output = np.empty_like(value)
    positive = value >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def predict(
    features: np.ndarray,
    weights: Sequence[np.ndarray],
    biases: Sequence[np.ndarray],
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    value = (np.asarray(features, dtype=np.float64) - mean) / scale
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        value = value @ np.asarray(weight, dtype=np.float64) + np.asarray(bias, dtype=np.float64)
        if index + 1 < len(weights):
            value = np.maximum(value, 0.0)
    return _sigmoid(value)


def voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Exact non-2007 VOC precision-envelope integration in eval_det.py."""

    recall = np.asarray(recall, dtype=np.float64)
    precision = np.asarray(precision, dtype=np.float64)
    if recall.ndim != 1 or precision.shape != recall.shape:
        raise ValueError("VOC AP inputs must be matching vectors")
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    changed = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]))


def detection_average_precision(
    scores: np.ndarray,
    target_iou: np.ndarray,
    scene_ids: np.ndarray,
    matched_gt_indices: np.ndarray,
    gt_counts: Mapping[str, int],
    indices: np.ndarray,
    selected_scenes: Sequence[str],
    threshold: float,
) -> float:
    """Class-agnostic CA-1M evaluator AP without rereading any GT artifact.

    ``target_iou`` and ``matched_gt_indices`` are the max-IoU result computed
    by the dataset join.  That is sufficient to reproduce eval_det's strict
    ``ovmax > threshold`` matching, including duplicate detections of one GT.
    """

    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(target_iou, dtype=np.float64)
    scenes = np.asarray(scene_ids, dtype=np.str_)
    matched = np.asarray(matched_gt_indices, dtype=np.int64)
    rows = np.asarray(indices, dtype=np.int64)
    if any(value.shape != scores.shape for value in (targets, scenes, matched)):
        raise ValueError("detection AP row arrays must have matching shapes")
    if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= len(scores)):
        raise ValueError("detection AP indices are invalid")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("detection AP indices contain duplicates")
    selected = tuple(str(scene) for scene in selected_scenes)
    if len(selected) != len(set(selected)) or any(scene not in gt_counts for scene in selected):
        raise ValueError("detection AP selected scenes are invalid")
    selected_set = set(selected)
    if any(str(scenes[index]) not in selected_set for index in rows):
        raise ValueError("detection AP rows escape the selected scene partition")
    positives = sum(int(gt_counts[scene]) for scene in selected)
    if positives <= 0:
        return float("nan")
    # CA1MDetectionDataset iterates ``sorted(os.listdir(data_root))`` and then
    # preserves prediction row order inside each scene.  Recreate that input
    # order before eval_det's intentionally unstable np.argsort call so even
    # tied scores have repository-evaluator parity.
    evaluator_input = rows[np.argsort(scenes[rows], kind="stable")]
    order = np.argsort(-scores[evaluator_input])
    true_positive = np.zeros(len(rows), dtype=np.float64)
    false_positive = np.zeros(len(rows), dtype=np.float64)
    detected: set[tuple[str, int]] = set()
    for rank, local_index in enumerate(order):
        row = int(evaluator_input[int(local_index)])
        scene = str(scenes[row])
        gt_index = int(matched[row])
        # eval_det is deliberately strict here: equality is a false positive.
        if targets[row] > threshold and 0 <= gt_index < int(gt_counts[scene]):
            key = (scene, gt_index)
            if key not in detected:
                true_positive[rank] = 1.0
                detected.add(key)
            else:
                false_positive[rank] = 1.0
        else:
            false_positive[rank] = 1.0
    true_positive = np.cumsum(true_positive)
    false_positive = np.cumsum(false_positive)
    recall = true_positive / float(positives + 1e-6)
    precision = true_positive / np.maximum(
        true_positive + false_positive, np.finfo(np.float64).eps
    )
    return voc_ap(recall, precision)


def ranking_metrics(
    scores: np.ndarray,
    values: Mapping[str, np.ndarray],
    gt_counts: Mapping[str, int],
    indices: np.ndarray,
    selected_scenes: Sequence[str],
) -> dict[str, float]:
    return {
        f"AP{int(threshold * 100):02d}": detection_average_precision(
            scores,
            values["target_iou"],
            values["scene_ids"],
            values["matched_gt_indices"],
            gt_counts,
            indices,
            selected_scenes,
            threshold,
        )
        for threshold in IOU_THRESHOLDS
    }


def deployment_scores(
    detector_scores: np.ndarray,
    raw_outputs: np.ndarray,
    ranking_weights: np.ndarray,
    detector_blend: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply runtime monotonic projection and detector/quality score blend."""

    outputs = np.asarray(raw_outputs, dtype=np.float64)
    detector = np.asarray(detector_scores, dtype=np.float64)
    if outputs.shape != (len(detector), 4) or not np.isfinite(outputs).all():
        raise ValueError("quality outputs are invalid")
    components = outputs.copy()
    components[:, 1:] = np.minimum.accumulate(components[:, 1:], axis=1)
    quality = components @ np.asarray(ranking_weights, dtype=np.float64)
    deployed = detector_blend * detector + (1.0 - detector_blend) * quality
    return components, quality, deployed


def _fit(
    features: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    if len(indices) < 2:
        raise ValueError("each scene-fold training partition requires at least two rows")
    return fit_iou_aware_mlp(
        features[indices], targets[indices], feature_dim=features.shape[1],
        strict_threshold_targets=True,
        hidden_dims=args.hidden_dims, epochs=args.epochs,
        learning_rate=args.learning_rate, weight_decay=args.l2_weight,
        iou_loss_weight=args.iou_loss_weight,
        threshold_loss_weight=args.threshold_loss_weight,
        monotonic_loss_weight=args.monotonic_loss_weight, seed=seed,
    )


def train(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    raw_output = Path(args.output)
    raw_manifest_output = Path(args.manifest_output)
    raw_oof_argument = getattr(args, "oof_output", None)
    raw_oof_manifest_argument = getattr(args, "oof_manifest_output", None)
    oof_requested = raw_oof_argument is not None or raw_oof_manifest_argument is not None
    if (raw_oof_argument is None) != (raw_oof_manifest_argument is None):
        raise ValueError("--oof-output and --oof-manifest-output are required together")
    if raw_output.is_symlink() or raw_manifest_output.is_symlink():
        raise ValueError("checkpoint/manifest outputs must not be symlinks")
    output = raw_output.resolve()
    manifest_output = raw_manifest_output.resolve()
    optional_outputs: tuple[Path, ...] = ()
    if oof_requested:
        raw_oof_output = Path(raw_oof_argument)
        raw_oof_manifest_output = Path(raw_oof_manifest_argument)
        if raw_oof_output.is_symlink() or raw_oof_manifest_output.is_symlink():
            raise ValueError("OOF sidecar outputs must not be symlinks")
        oof_output = raw_oof_output.resolve()
        oof_manifest_output = raw_oof_manifest_output.resolve()
        optional_outputs = (oof_output, oof_manifest_output)
    all_outputs = (output, manifest_output, *optional_outputs)
    if (
        len(set(all_outputs)) != len(all_outputs)
        or output.suffix.lower() != ".npz"
        or manifest_output.suffix.lower() != ".json"
        or (oof_requested and oof_output.suffix.lower() != ".npz")
        or (oof_requested and oof_manifest_output.suffix.lower() != ".json")
    ):
        raise ValueError("checkpoint/manifest outputs must be distinct .npz/.json paths")
    if any(path.exists() or path.is_symlink() for path in all_outputs):
        raise FileExistsError("refusing to start with an existing checkpoint/manifest output")
    values, dataset_manifest = load_dataset(args.dataset, args.dataset_manifest)
    features = np.asarray(values["quality_features"], dtype=np.float32)
    targets = np.asarray(values["target_iou"], dtype=np.float32)
    baseline = np.asarray(values["prediction_scores"], dtype=np.float64)
    folds = np.asarray(values["fold_ids"], dtype=np.int64)
    names = tuple(str(item) for item in values["feature_names"].tolist())
    rank_weights = np.asarray(args.ranking_weights, dtype=np.float64)
    if rank_weights.shape != (4,) or np.any(rank_weights < 0) or not np.isfinite(rank_weights).all() or rank_weights.sum() <= 0:
        raise ValueError("ranking weights must be four finite non-negative values")
    rank_weights /= rank_weights.sum()
    detector_blend = float(args.detector_blend)
    if not np.isfinite(detector_blend) or not 0.0 <= detector_blend <= 1.0:
        raise ValueError("detector blend must be finite in [0,1]")
    scene_records = dataset_manifest["scenes"]
    all_scenes = tuple(str(row["scene_id"]) for row in scene_records)
    gt_counts = {str(row["scene_id"]): int(row["gt_boxes"]) for row in scene_records}
    namespace = str(np.asarray(values["split_namespace"]).item())
    scene_folds = assign_scene_folds(all_scenes, namespace, fold_count=5)
    oof_outputs = np.full((len(features), 4), np.nan, dtype=np.float64)
    models: dict[int, tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], np.ndarray, np.ndarray]] = {}
    fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        training = np.flatnonzero(folds != fold)
        heldout = np.flatnonzero(folds == fold)
        if not len(heldout):
            raise ValueError(f"scene fold {fold} is empty")
        model = _fit(features, targets, training, args, args.seed + fold)
        models[fold] = model
        oof_outputs[heldout] = predict(features[heldout], *model)
    if not np.isfinite(oof_outputs).all():
        raise RuntimeError("OOF prediction coverage is incomplete")
    oof_components, quality_oof_score, deployed_oof_score = deployment_scores(
        baseline, oof_outputs, rank_weights, detector_blend
    )
    oof_arrays: dict[str, np.ndarray] | None = None
    oof_manifest: dict[str, Any] | None = None
    if oof_requested:
        oof_arrays, oof_manifest = _oof_sidecar(
            values=values,
            dataset_path=args.dataset.resolve(),
            dataset_manifest_path=args.dataset_manifest.resolve(),
            dataset_manifest=dataset_manifest,
            models=models,
            oof_outputs=oof_outputs,
            oof_components=oof_components,
            quality_oof_scores=quality_oof_score,
            deployment_oof_scores=deployed_oof_score,
            ranking_weights=rank_weights,
            args=args,
        )
    all_indices = np.arange(len(features), dtype=np.int64)
    dev = np.flatnonzero(folds == 0)
    dev_scenes = tuple(scene for scene in all_scenes if scene_folds[scene] == 0)
    baseline_oof = ranking_metrics(baseline, values, gt_counts, all_indices, all_scenes)
    learned_oof = ranking_metrics(
        deployed_oof_score, values, gt_counts, all_indices, all_scenes
    )
    quality_only_oof = ranking_metrics(
        quality_oof_score, values, gt_counts, all_indices, all_scenes
    )
    baseline_dev = ranking_metrics(baseline, values, gt_counts, dev, dev_scenes)
    learned_dev = ranking_metrics(
        deployed_oof_score, values, gt_counts, dev, dev_scenes
    )
    oof_delta = {name: learned_oof[name] - baseline_oof[name] for name in baseline_oof}
    dev_delta = {name: learned_dev[name] - baseline_dev[name] for name in baseline_dev}
    positive_ap50_folds = 0
    for fold in range(5):
        heldout = np.flatnonzero(folds == fold)
        fold_scenes = tuple(scene for scene in all_scenes if scene_folds[scene] == fold)
        base = ranking_metrics(baseline, values, gt_counts, heldout, fold_scenes)
        learned = ranking_metrics(
            deployed_oof_score, values, gt_counts, heldout, fold_scenes
        )
        delta = {
            key: learned[key] - base[key]
            if math.isfinite(learned[key]) and math.isfinite(base[key]) else float("nan")
            for key in base
        }
        if math.isfinite(delta["AP50"]) and delta["AP50"] > 0:
            positive_ap50_folds += 1
        fold_rows.append({
            "fold": fold,
            "scenes": len(fold_scenes),
            "rows": len(heldout), "baseline": base, "learned": learned, "delta": delta,
        })
    finite_gate = all(math.isfinite(value) for value in (*oof_delta.values(), *dev_delta.values()))
    authorized = bool(
        finite_gate
        and oof_delta["AP15"] >= -args.max_oof_ap15_loss
        and oof_delta["AP25"] >= -args.max_oof_ap25_loss
        and dev_delta["AP15"] >= -args.max_dev_ap15_loss
        and dev_delta["AP25"] >= -args.max_dev_ap25_loss
        and dev_delta["AP50"] >= args.min_dev_ap50_gain
        and oof_delta["AP50"] >= args.min_oof_ap50_gain
        and positive_ap50_folds >= args.min_positive_ap50_folds
    )
    dev_model = models[0]  # trained only on folds 1--4; dev fold remains untouched
    weights, biases, feature_mean, feature_scale = dev_model
    checkpoint_arrays: dict[str, np.ndarray] = {
        "schema": np.asarray(CHECKPOINT_SCHEMA),
        "complete": np.asarray(True, dtype=np.bool_),
        "train_only": np.asarray(True, dtype=np.bool_),
        "validation_ground_truth_access": np.asarray(False, dtype=np.bool_),
        "activation_authorized": np.asarray(authorized, dtype=np.bool_),
        "feature_names": np.asarray(names, dtype=np.str_),
        "output_names": np.asarray(OUTPUT_NAMES, dtype=np.str_),
        "iou_thresholds": np.asarray(IOU_THRESHOLDS, dtype=np.float32),
        "ranking_weights": rank_weights.astype(np.float32),
        "detector_blend": np.asarray(detector_blend, dtype=np.float32),
        "preserve_original_floor": np.asarray(False, dtype=np.bool_),
        "monotonic_probability_projection": np.asarray(True, dtype=np.bool_),
        "strict_iou_thresholds": np.asarray(True, dtype=np.bool_),
        "feature_mean": feature_mean.astype(np.float32),
        "feature_scale": feature_scale.astype(np.float32),
        "num_layers": np.asarray(len(weights), dtype=np.int64),
        "training_folds": np.asarray((1, 2, 3, 4), dtype=np.int8),
        "heldout_dev_fold": np.asarray(0, dtype=np.int8),
    }
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        checkpoint_arrays[f"weight_{index}"] = np.asarray(weight, dtype=np.float32)
        checkpoint_arrays[f"bias_{index}"] = np.asarray(bias, dtype=np.float32)
    checkpoint_bytes = _npz_bytes(checkpoint_arrays)
    checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    oof_record: dict[str, Any] | None = None
    oof_bytes: bytes | None = None
    oof_manifest_bytes: bytes | None = None
    if oof_requested:
        assert oof_arrays is not None and oof_manifest is not None
        oof_bytes = _npz_bytes(oof_arrays)
        oof_sha = hashlib.sha256(oof_bytes).hexdigest()
        oof_manifest["deployment_checkpoint"] = {
            "path": str(output),
            "sha256": checkpoint_sha,
            "schema": CHECKPOINT_SCHEMA,
            "heldout_dev_fold_model": 0,
            "fold_model_sha256": oof_manifest["fold_model_sha256"][0],
        }
        oof_manifest["checkpoint_manifest"] = {
            "path": str(manifest_output),
            "schema": MANIFEST_SCHEMA,
            "binds_this_sidecar": True,
            "sha256_omitted_to_avoid_manifest_hash_cycle": True,
        }
        oof_manifest["artifact"] = {
            "path": str(oof_output), "sha256": oof_sha,
            "schema": OOF_ROW_SCORE_SCHEMA,
        }
        oof_manifest_bytes = _canonical_json(oof_manifest)
        oof_record = {
            "path": str(oof_output),
            "sha256": oof_sha,
            "manifest_path": str(oof_manifest_output),
            "manifest_sha256": hashlib.sha256(oof_manifest_bytes).hexdigest(),
            "schema": OOF_ROW_SCORE_SCHEMA,
            "manifest_schema": OOF_ROW_SCORE_MANIFEST_SCHEMA,
            "checkpoint_manifest_binds_sidecar": True,
            "sidecar_manifest_binds_checkpoint": True,
        }
    report = {
        "schema": MANIFEST_SCHEMA,
        "complete": True,
        "train_only": True,
        "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "target_schema": TARGET_SCHEMA,
        "official_validation_comparable": False,
        "activation_authorized": authorized,
        "checkpoint": {"path": str(output), "sha256": checkpoint_sha},
        "dataset": {
            "path": str(args.dataset.resolve()), "sha256": sha256_file(args.dataset.resolve()),
            "manifest_path": str(args.dataset_manifest.resolve()),
            "manifest_sha256": sha256_file(args.dataset_manifest.resolve()),
            "source_subset_sha256": dataset_manifest["frozen_subset_manifest"]["sha256"],
        },
        "model": {
            "kind": "native_iou_mlp", "feature_names": list(names),
            "hidden_dims": list(args.hidden_dims), "epochs": args.epochs,
            "learning_rate": args.learning_rate, "l2_weight": args.l2_weight,
            "seed": args.seed, "ranking_weights": rank_weights.tolist(),
            "detector_blend": detector_blend,
            "quality_blend": 1.0 - detector_blend,
            "preserve_original_floor": False,
            "strict_iou_threshold_targets": True,
            "deployment_score_formula": (
                "detector_blend*detector_score + "
                "(1-detector_blend)*monotonic_projected_quality_rank"
            ),
            "deployable_training_folds": [1, 2, 3, 4], "untouched_dev_fold": 0,
        },
        "metrics": {
            "baseline_oof": baseline_oof, "learned_oof": learned_oof, "oof_delta": oof_delta,
            "quality_only_oof_diagnostic": quality_only_oof,
            "baseline_dev": baseline_dev, "learned_dev": learned_dev, "dev_delta": dev_delta,
            "folds": fold_rows, "positive_ap50_folds": positive_ap50_folds,
            "target_iou_mae_oof": float(np.mean(np.abs(oof_outputs[:, 0] - targets))),
        },
        "gate": {
            "finite_metrics": finite_gate,
            "max_dev_ap15_loss": args.max_dev_ap15_loss,
            "max_dev_ap25_loss": args.max_dev_ap25_loss,
            "max_oof_ap15_loss": args.max_oof_ap15_loss,
            "max_oof_ap25_loss": args.max_oof_ap25_loss,
            "min_dev_ap50_gain": args.min_dev_ap50_gain,
            "min_oof_ap50_gain": args.min_oof_ap50_gain,
            "min_positive_ap50_folds": args.min_positive_ap50_folds,
            "decision": "PASS" if authorized else "FAIL",
            "scope": "train-only scene-grouped OOF and frozen dev fold; no CA-1M val",
            "metric": (
                "CA-1M class-agnostic eval_det parity: box3d_iou_v2 max match, "
                "strict IoU threshold, duplicate-GT suppression, VOC precision envelope"
            ),
            "score_under_test": {
                "detector_blend": detector_blend,
                "quality_blend": 1.0 - detector_blend,
                "preserve_original_floor": False,
            },
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__},
    }
    if oof_record is not None:
        report["all_fold_oof_row_scores"] = oof_record
    source_collection = dataset_manifest["train_collection"]
    if source_collection["schema"] == FINAL_BASE_COLLECTION_SCHEMA:
        report["dataset"].update(
            {
                "source_collection_schema": FINAL_BASE_COLLECTION_SCHEMA,
                "source_final_base_collection": source_collection[
                    "source_final_base_collection"
                ],
                "fixed10_paired_report": source_collection[
                    "fixed10_paired_report"
                ],
                "source_modules": source_collection["source_modules"],
                "geometry_authority": "sealed_final_base_prediction",
                "offline_direct_observer": True,
                "cross_run_boxfusion_replay_invoked": False,
                "cross_run_exact_identity_required": False,
                "old_native_b6_diagnostics_reused": False,
                "old_native_b6_checkpoint_reused": False,
            }
        )
    manifest_bytes = _canonical_json(report)
    artifacts: list[tuple[Path, bytes]] = [(output, checkpoint_bytes)]
    if oof_requested:
        assert oof_bytes is not None and oof_manifest_bytes is not None
        artifacts.extend(
            (
                (oof_output, oof_bytes),
                (oof_manifest_output, oof_manifest_bytes),
            )
        )
    artifacts.append((manifest_output, manifest_bytes))
    _publish_transaction(artifacts)
    return report, authorized


def _csv(value: str, cast: type) -> tuple[Any, ...]:
    try:
        result = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not result:
        raise argparse.ArgumentTypeError("comma-separated value is empty")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--dataset-manifest", type=Path, required=True)
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--train", action="store_true")
    result.add_argument("--output", type=Path)
    result.add_argument("--manifest-output", type=Path)
    result.add_argument(
        "--oof-output", type=Path,
        help="optional create-only all-fold OOF row-score NPZ sidecar",
    )
    result.add_argument(
        "--oof-manifest-output", type=Path,
        help="required immutable manifest when --oof-output is set",
    )
    result.add_argument("--epochs", type=int, default=400)
    result.add_argument("--learning-rate", type=float, default=1e-3)
    result.add_argument("--l2-weight", type=float, default=1e-4)
    result.add_argument("--hidden-dims", type=lambda x: _csv(x, int), default=(64, 32))
    result.add_argument("--ranking-weights", type=lambda x: _csv(x, float), default=tuple(DEFAULT_RANKING_WEIGHTS))
    result.add_argument("--detector-blend", type=float, default=0.40)
    result.add_argument("--iou-loss-weight", type=float, default=1.0)
    result.add_argument("--threshold-loss-weight", type=float, default=1.0)
    result.add_argument("--monotonic-loss-weight", type=float, default=0.10)
    result.add_argument("--seed", type=int, default=1337)
    result.add_argument("--max-dev-ap15-loss", type=float, default=0.005)
    result.add_argument("--max-dev-ap25-loss", type=float, default=0.005)
    result.add_argument("--max-oof-ap15-loss", type=float, default=0.005)
    result.add_argument("--max-oof-ap25-loss", type=float, default=0.005)
    result.add_argument("--min-dev-ap50-gain", type=float, default=0.005)
    result.add_argument("--min-oof-ap50-gain", type=float, default=0.0)
    result.add_argument("--min-positive-ap50-folds", type=int, default=3)
    return result


def main() -> int:
    args = parser().parse_args()
    if not args.train:
        values, manifest = load_dataset(args.dataset, args.dataset_manifest)
        print(json.dumps({
            "ok": True, "mode": "preflight", "training_started": False,
            "train_only": True, "validation_ground_truth_access": False,
            "scenes": len(manifest["scenes"]), "rows": len(values["target_iou"]),
            "feature_dim": values["quality_features"].shape[1],
            "folds": {str(i): int(np.count_nonzero(values["fold_ids"] == i)) for i in range(5)},
        }, indent=2, sort_keys=True))
        return 0
    if args.output is None or args.manifest_output is None:
        raise ValueError("--train requires --output and --manifest-output")
    if args.epochs < 1 or args.min_positive_ap50_folds < 0 or args.min_positive_ap50_folds > 5:
        raise ValueError("invalid training/gate integer")
    loss_limits = (
        args.max_dev_ap15_loss,
        args.max_dev_ap25_loss,
        args.max_oof_ap15_loss,
        args.max_oof_ap25_loss,
    )
    if any(not np.isfinite(value) or value < 0.0 for value in loss_limits):
        raise ValueError("AP loss gates must be finite and non-negative")
    if any(
        not np.isfinite(value)
        for value in (args.min_dev_ap50_gain, args.min_oof_ap50_gain)
    ):
        raise ValueError("AP gain gates must be finite")
    report, authorized = train(args)
    print(json.dumps({
        "checkpoint": report["checkpoint"], "activation_authorized": authorized,
        "gate": report["gate"], "manifest": str(args.manifest_output.resolve()),
        "all_fold_oof_row_scores": report.get("all_fold_oof_row_scores"),
    }, indent=2, sort_keys=True))
    return 0 if authorized else 3


if __name__ == "__main__":
    raise SystemExit(main())
