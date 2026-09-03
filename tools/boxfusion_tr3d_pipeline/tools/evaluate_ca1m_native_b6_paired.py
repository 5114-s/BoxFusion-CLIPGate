#!/usr/bin/env python3
"""Fail-closed paired evaluation for the CA-1M-native B6 canonical103 run.

The ``preflight`` phase opens no validation ground truth and creates nothing.
The ``evaluate`` phase accepts only a sealed canonical103 observer collection
and an activation-authorized, score-only output tree.  It then evaluates the
same-run G0 anchor and native-B6 scores with the same clean upstream evaluator
command and records the paired metric delta.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


COLLECTION_SCHEMA = "boxfusion.ca1m_native_b6_canonical103_collection.v1"
SCENE_COMPLETION_SCHEMA = (
    "boxfusion.ca1m_native_b6_canonical103_scene_completion.v1"
)
IDENTITY_SCHEMA = "boxfusion.ca1m_native_b6_canonical103_identity_audit.v1"
ACTIVE_SCHEMA = "boxfusion.ca1m_native_b6_score_counterfactual.v1"
REPORT_SCHEMA = "boxfusion.ca1m_native_b6_canonical103_paired_evaluation.v1"
SCENE_LIST_SHA256 = (
    "c3efbe544c7403acc4183d7e4a799dad2bb40f60cbdba38830863f8712f4648f"
)
CHECKPOINT_SHA256 = (
    "d19b3471c84144634c4f50cc339d772a25ada33f873875235087636e8188ca77"
)
CHECKPOINT_MANIFEST_SHA256 = (
    "b941c1008dd6a8703010e731c3b0d3675b981c146ceb4cf8a065698c96c560ea"
)
TRAIN_DATASET_SHA256 = (
    "6dbcb8f996dee76d77261b7bf9a42ee9bbb2562c60384c920b7e1fff12a4ff04"
)
TRAIN_DATASET_MANIFEST_SHA256 = (
    "2a47d0ac606f7d892efc982a300262afe6a51e6030f2bcd151ddc2ece73c06d4"
)
TRAIN_SUBSET_SHA256 = (
    "29a32e92cfece667e9fef4389227eacba2b96c55737569fa6219ca7ab527fd23"
)
TRAIN_COLLECTION_SHA256 = (
    "53097374fe9f5f276f74049017cd3cedc6b89aa07c466c6e4af30d973d7a8a0f"
)
FULL_VAL_SCENE_LIST_SHA256 = (
    "bd5f3fc66168114048a1b12addc45949c8f54f9c016b921bacfb6fe9e3e7dc2f"
)
UPSTREAM_COMMIT = "b2e0219a7284249bad4a4a8925066839fe2fa33b"
UPSTREAM_FILES = {
    "evaluation/eval_ca1m.py": (
        "3c9260cd57da342fd25b664a0091c4345a44bba499c2cfbf3c8ecff4eaa4c788"
    ),
    "evaluation/utils/ap_helper.py": (
        "c2b08890cf6b6497165d7d7af0bf16f9205a65698c197639db70adf702f27d6f"
    ),
    "evaluation/utils/eval_det.py": (
        "6ef54c395e46716e364547115090bae96643bf346b3e8eb1b859719781a557dd"
    ),
    "evaluation/utils/box_util.py": (
        "44aadf0088c0ccd5e9f51a1cded22fb1080d59aa50d0fb914fe6e83896aaa107"
    ),
    "evaluation/data_util/dataset.py": (
        "50d4e03db6f1fa9e540fd7f9c6ceab85d180ed61a14251d0e1971c717e741f8d"
    ),
}
THRESHOLDS = (0.15, 0.25, 0.50)
METRIC_PATTERN = re.compile(
    r"^eval (mAP|APrec|ARecall): ([0-9]+(?:\.[0-9]+)?)$", re.MULTILINE
)
SCENE_PATTERN = re.compile(r"^[0-9]{8}$")
SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {raw}")
    resolved = raw.resolve()
    if not resolved.is_file() or resolved.is_symlink() or resolved.stat().st_size <= 0:
        raise ValueError(f"{label} must be a non-empty regular file: {resolved}")
    return resolved


def directory(path: Path, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {raw}")
    resolved = raw.resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise ValueError(f"{label} must be a directory: {resolved}")
    return resolved


def executable(path: Path, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError(f"{label} must resolve to an executable regular file: {path}")
    return resolved


def read_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(regular(path, label).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def read_scenes(path: Path) -> tuple[str, ...]:
    path = regular(path, "canonical103 scene list")
    if sha256(path) != SCENE_LIST_SHA256:
        raise ValueError("canonical103 scene-list SHA256 drifted")
    scenes = tuple(line.strip() for line in path.read_text().splitlines() if line.strip())
    if (
        len(scenes) != 103
        or len(set(scenes)) != 103
        or any(SCENE_PATTERN.fullmatch(scene) is None for scene in scenes)
    ):
        raise ValueError("canonical103 scene list must contain 103 unique numeric IDs")
    return scenes


def exact_files(
    root: Path, scenes: Sequence[str], suffix: str, label: str
) -> dict[str, Path]:
    root = directory(root, label)
    expected = {f"{scene}{suffix}" for scene in scenes}
    actual = {entry.name for entry in root.iterdir()}
    if actual != expected:
        raise ValueError(
            f"{label} artifact set differs: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return {
        scene: regular(root / f"{scene}{suffix}", f"{label} scene artifact")
        for scene in scenes
    }


def _artifact(record: Mapping[str, Any], name: str, path: Path) -> None:
    value = record.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"completion artifact {name} is missing")
    if Path(str(value.get("path", ""))).resolve() != path.resolve():
        raise ValueError(f"completion artifact {name} path disagrees")
    if value.get("sha256") != sha256(path):
        raise ValueError(f"completion artifact {name} SHA256 disagrees")


def validate_collection_chain(args: argparse.Namespace, scenes: Sequence[str]) -> dict[str, Any]:
    anchors = exact_files(args.anchor_root, scenes, "_boxes.pkl", "same-run anchor root")
    observers = exact_files(args.observer_root, scenes, "_boxes.pkl", "observer root")
    diagnostics = exact_files(
        args.diagnostics_root, scenes, "_ca1m_native_b6.npz", "native diagnostic root"
    )
    records = exact_files(
        args.record_completion_root, scenes, ".json", "record completion root"
    )
    completions = exact_files(
        args.observer_completion_root, scenes, ".json", "observer completion root"
    )
    collection = read_json(args.collection_manifest, "collection manifest")
    required_collection = {
        "schema": COLLECTION_SCHEMA,
        "complete": True,
        "dataset_split": "official_validation_canonical103",
        "scene_count": 103,
        "scene_list_sha256": SCENE_LIST_SHA256,
        "ground_truth_access": False,
        "evaluation_invoked": False,
        "training_authorized": False,
        "same_run_anchor_byte_identity_required": True,
    }
    for key, expected in required_collection.items():
        if collection.get(key) != expected:
            raise ValueError(f"collection manifest field {key} disagrees")
    collection_rows = collection.get("scenes")
    if not isinstance(collection_rows, list) or [row.get("scene_id") for row in collection_rows] != list(scenes):
        raise ValueError("collection manifest scene order/set disagrees")

    identity = read_json(args.identity_audit, "identity audit")
    required_identity = {
        "schema": IDENTITY_SCHEMA,
        "ok": True,
        "dataset_split": "official_validation_canonical103",
        "scenes": 103,
        "scene_list_sha256": SCENE_LIST_SHA256,
        "observer_only": True,
        "mutation_enabled": False,
        "same_run_byte_identity_scenes": 103,
        "mapping_coverage": 1.0,
        "ground_truth_access": False,
        "evaluation_invoked": False,
        "training_authorized": False,
    }
    for key, expected in required_identity.items():
        if identity.get(key) != expected:
            raise ValueError(f"identity audit field {key} disagrees")
    identity_scenes = identity.get("per_scene")
    if not isinstance(identity_scenes, dict) or set(identity_scenes) != set(scenes):
        raise ValueError("identity audit scene set disagrees")

    total_rows = 0
    all_scores: list[float] = []
    for scene, collection_row in zip(scenes, collection_rows):
        if not isinstance(collection_row, dict):
            raise ValueError(f"{scene}: collection scene row is invalid")
        if collection_row.get("record_completion_sha256") != sha256(records[scene]):
            raise ValueError(f"{scene}: record completion hash is stale")
        if collection_row.get("observer_completion_sha256") != sha256(completions[scene]):
            raise ValueError(f"{scene}: observer completion hash is stale")
        record = read_json(records[scene], f"{scene} record completion")
        observer = read_json(completions[scene], f"{scene} observer completion")
        for value, phase in ((record, "cutr_record"), (observer, "g0_native_b6_observer")):
            required = {
                "schema": SCENE_COMPLETION_SCHEMA,
                "phase": phase,
                "scene_id": scene,
                "complete": True,
                "dataset_split": "official_validation_canonical103",
                "ground_truth_access": False,
                "evaluation_invoked": False,
                "training_authorized": False,
            }
            for key, expected in required.items():
                if value.get(key) != expected:
                    raise ValueError(f"{scene}/{phase}: completion field {key} disagrees")
        if observer.get("output_mutation_authorized") is not False:
            raise ValueError(f"{scene}: observer completion authorizes mutation")
        artifacts = observer.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError(f"{scene}: observer completion artifacts are missing")
        _artifact(artifacts, "prediction", observers[scene])
        _artifact(artifacts, "same_run_anchor", anchors[scene])
        _artifact(artifacts, "native_b6_diagnostic", diagnostics[scene])

        scene_identity = identity_scenes[scene]
        if not isinstance(scene_identity, dict):
            raise ValueError(f"{scene}: identity scene record is invalid")
        prediction_identity = scene_identity.get("identity")
        diagnostic_identity = scene_identity.get("diagnostic")
        if not isinstance(prediction_identity, dict) or not isinstance(diagnostic_identity, dict):
            raise ValueError(f"{scene}: identity/diagnostic record is missing")
        anchor_hash = sha256(anchors[scene])
        if (
            anchor_hash != sha256(observers[scene])
            or prediction_identity.get("prediction_sha256") != anchor_hash
            or prediction_identity.get("byte_identity") is not True
            or prediction_identity.get("semantic_identity") is not True
            or diagnostic_identity.get("diagnostic_sha256") != sha256(diagnostics[scene])
            or diagnostic_identity.get("mapping_coverage") != 1.0
        ):
            raise ValueError(f"{scene}: sealed collection identity binding disagrees")
        rows = int(prediction_identity.get("rows", -1))
        if rows < 0 or observer.get("prediction_rows") != rows:
            raise ValueError(f"{scene}: sealed prediction row count disagrees")
        prediction_rows = load_prediction(anchors[scene])
        if len(prediction_rows) != rows:
            raise ValueError(f"{scene}: raw prediction row count disagrees")
        all_scores.extend(float(row[2]) for row in prediction_rows)
        total_rows += rows
    if identity.get("prediction_rows") != total_rows or identity.get("mapping_rows") != total_rows:
        raise ValueError("identity aggregate row count disagrees")
    score_array = np.asarray(all_scores, dtype=np.float64)
    if (
        len(score_array) < 2
        or not np.isfinite(score_array).all()
        or np.unique(score_array).size < 2
    ):
        raise ValueError("same-run G0 anchor scores are not finite/nonconstant real scores")
    return {
        "anchors": anchors,
        "observers": observers,
        "diagnostics": diagnostics,
        "collection": collection,
        "identity": identity,
        "rows": total_rows,
        "real_score_min": float(score_array.min()),
        "real_score_max": float(score_array.max()),
    }


def load_prediction(path: Path) -> list[tuple[int, np.ndarray, float]]:
    with regular(path, "prediction").open("rb") as handle:
        payload = pickle.load(handle)
        if handle.read(1):
            raise ValueError(f"prediction has trailing bytes: {path}")
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], list):
        raise ValueError(f"prediction must contain exactly one list batch: {path}")
    rows: list[tuple[int, np.ndarray, float]] = []
    for index, row in enumerate(payload[0]):
        if not isinstance(row, tuple) or len(row) != 3 or type(row[0]) is not int:
            raise ValueError(f"prediction row {index} is invalid: {path}")
        corners = np.asarray(row[1])
        score = float(row[2])
        if (
            row[0] != 0
            or corners.shape != (8, 3)
            or corners.dtype != np.float32
            or not corners.flags.c_contiguous
            or not np.isfinite(corners).all()
            or not np.isfinite(score)
            or not 0.0 <= score <= 1.0
        ):
            raise ValueError(f"prediction row {index} is invalid: {path}")
        rows.append((0, np.array(corners, dtype=np.float32, order="C", copy=True), score))
    return rows


def compare_score_only(
    anchor: Sequence[tuple[int, np.ndarray, float]],
    active: Sequence[tuple[int, np.ndarray, float]],
) -> int:
    if len(anchor) != len(active):
        raise ValueError("active prediction row count changed")
    changed = 0
    for index, (left, right) in enumerate(zip(anchor, active)):
        if (
            left[0] != right[0]
            or left[1].dtype != right[1].dtype
            or left[1].shape != right[1].shape
            or left[1].tobytes(order="C") != right[1].tobytes(order="C")
        ):
            raise ValueError(f"active prediction geometry/order changed at row {index}")
        changed += int(left[2] != right[2])
    return changed


def _stable_sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    result = np.empty_like(value)
    positive = value >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def load_checkpoint_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(regular(path, "native-B6 checkpoint"), allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    layer_count = int(np.asarray(arrays["num_layers"]).item())
    if not np.isclose(float(np.asarray(arrays["detector_blend"]).item()), 0.4, rtol=0, atol=1e-8):
        raise ValueError("native-B6 checkpoint detector blend is not frozen at 0.4")
    if layer_count < 1:
        raise ValueError("native-B6 checkpoint has no MLP layers")
    return arrays


def independently_recompute_scores(
    diagnostic_path: Path,
    scene: str,
    anchor: Sequence[tuple[int, np.ndarray, float]],
    checkpoint: Mapping[str, np.ndarray],
) -> np.ndarray:
    with np.load(regular(diagnostic_path, "native-B6 diagnostic"), allow_pickle=False) as archive:
        result_indices = np.asarray(archive["result_indices"])
        corners = np.asarray(archive["corners"])
        scores = np.asarray(archive["scores"])
        features = np.asarray(archive["features"])
    count = len(anchor)
    if not np.array_equal(result_indices, np.arange(count, dtype=result_indices.dtype)):
        raise ValueError(f"{scene}: diagnostic result indices do not map every row")
    anchor_corners = (
        np.stack([row[1] for row in anchor])
        if count else np.empty((0, 8, 3), dtype=np.float32)
    )
    detector = np.asarray([row[2] for row in anchor], dtype=np.float32)
    if (
        corners.dtype != np.float32
        or corners.shape != anchor_corners.shape
        or corners.tobytes(order="C") != anchor_corners.tobytes(order="C")
        or scores.dtype != np.float32
        or not np.array_equal(scores, detector)
        or features.shape[0] != count
        or not np.isfinite(features).all()
    ):
        raise ValueError(f"{scene}: diagnostic raw corner/score/feature mapping disagrees")
    value = (
        features.astype(np.float64)
        - np.asarray(checkpoint["feature_mean"], dtype=np.float64)
    ) / np.asarray(checkpoint["feature_scale"], dtype=np.float64)
    layer_count = int(np.asarray(checkpoint["num_layers"]).item())
    for layer in range(layer_count):
        value = (
            value @ np.asarray(checkpoint[f"weight_{layer}"], dtype=np.float64)
            + np.asarray(checkpoint[f"bias_{layer}"], dtype=np.float64)
        )
        if layer + 1 < layer_count:
            value = np.maximum(value, 0.0)
    components = _stable_sigmoid(value)
    components[:, 1:] = np.minimum.accumulate(components[:, 1:], axis=1)
    quality = components @ np.asarray(checkpoint["ranking_weights"], dtype=np.float64)
    # This formula is deliberately written independently of the deployment
    # scorer.  The stored float32 blend is separately required to represent
    # 0.4, so its complement represents the frozen 0.6 quality weight.
    blend = float(np.asarray(checkpoint["detector_blend"]).item())
    combined = blend * detector.astype(np.float64) + (1.0 - blend) * quality
    return np.asarray(np.clip(combined, 0.0, 1.0), dtype=np.float32)


def validate_active_chain(
    args: argparse.Namespace,
    scenes: Sequence[str],
    anchors: Mapping[str, Path],
    diagnostics: Mapping[str, Path],
) -> dict[str, Any]:
    active_files = exact_files(args.active_root, scenes, "_boxes.pkl", "active root")
    report = read_json(args.active_report, "active score report")
    required = {
        "schema": ACTIVE_SCHEMA,
        "complete": True,
        "dataset": "CA1M",
        "mode": "active",
        "score_only": True,
        "active_materialization": True,
        "activation_authorized": True,
        "ground_truth_access": False,
        "evaluation_invoked": False,
        "feature_dimension": 14,
        "obb_unchanged": True,
        "row_count_unchanged": True,
        "row_order_unchanged": True,
        "scene_list_sha256": SCENE_LIST_SHA256,
        "scenes": 103,
    }
    for key, expected in required.items():
        if report.get(key) != expected:
            raise ValueError(f"active score report field {key} disagrees")
    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, dict) or (
        checkpoint.get("sha256") != CHECKPOINT_SHA256
        or checkpoint.get("manifest_sha256") != CHECKPOINT_MANIFEST_SHA256
    ):
        raise ValueError("active score report checkpoint binding disagrees")
    if Path(str(report.get("prediction_output_root", ""))).resolve() != Path(args.active_root).resolve():
        raise ValueError("active score report output root disagrees")
    per_scene = report.get("per_scene")
    if not isinstance(per_scene, dict) or set(per_scene) != set(scenes):
        raise ValueError("active score report scene set disagrees")
    checkpoint_arrays = load_checkpoint_arrays(args.checkpoint)
    rows = changed = formula_rows = 0
    for scene in scenes:
        anchor_rows = load_prediction(anchors[scene])
        active_rows = load_prediction(active_files[scene])
        scene_changed = compare_score_only(anchor_rows, active_rows)
        expected_scores = independently_recompute_scores(
            diagnostics[scene], scene, anchor_rows, checkpoint_arrays
        )
        actual_scores = np.asarray([row[2] for row in active_rows], dtype=np.float32)
        if expected_scores.tobytes(order="C") != actual_scores.tobytes(order="C"):
            raise ValueError(f"{scene}: active score does not equal frozen 0.4/0.6 formula")
        record = per_scene[scene]
        if not isinstance(record, dict) or (
            record.get("rows") != len(anchor_rows)
            or record.get("changed_scores") != scene_changed
            or record.get("anchor_prediction_sha256") != sha256(anchors[scene])
            or record.get("active_prediction_sha256") != sha256(active_files[scene])
            or record.get("obb_unchanged") is not True
            or record.get("row_count_unchanged") is not True
            or record.get("row_order_unchanged") is not True
        ):
            raise ValueError(f"{scene}: active score report binding disagrees")
        rows += len(anchor_rows)
        changed += scene_changed
        formula_rows += len(anchor_rows)
    if report.get("rows") != rows or report.get("changed_scores") != changed or changed <= 0:
        raise ValueError("active score aggregate rows/changes disagree or no score changed")
    return {
        "files": active_files, "report": report, "rows": rows,
        "changed": changed, "formula_verified_rows": formula_rows,
    }


def validate_training_provenance(
    args: argparse.Namespace, checkpoint_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    frozen = {
        "dataset": (regular(args.training_dataset, "training dataset"), TRAIN_DATASET_SHA256),
        "dataset_manifest": (
            regular(args.training_dataset_manifest, "training dataset manifest"),
            TRAIN_DATASET_MANIFEST_SHA256,
        ),
        "subset_manifest": (
            regular(args.training_subset_manifest, "training subset manifest"),
            TRAIN_SUBSET_SHA256,
        ),
        "collection_manifest": (
            regular(args.training_collection_manifest, "training collection manifest"),
            TRAIN_COLLECTION_SHA256,
        ),
        "full_val_scene_list": (
            regular(args.full_val_scene_list, "full107 validation scene list"),
            FULL_VAL_SCENE_LIST_SHA256,
        ),
    }
    for label, (path, expected) in frozen.items():
        if sha256(path) != expected:
            raise ValueError(f"frozen training provenance {label} SHA256 drifted")
    dataset_manifest = read_json(frozen["dataset_manifest"][0], "training dataset manifest")
    subset_manifest = read_json(frozen["subset_manifest"][0], "training subset manifest")
    collection_manifest = read_json(
        frozen["collection_manifest"][0], "training collection manifest"
    )
    if (
        dataset_manifest.get("schema")
        != "boxfusion.ca1m_native_b6_quality_dataset_manifest.v1"
        or dataset_manifest.get("complete") is not True
        or dataset_manifest.get("train_only") is not True
        or dataset_manifest.get("validation_ground_truth_access") is not False
        or dataset_manifest.get("validation_prediction_access") is not False
        or dataset_manifest.get("counts", {}).get("scenes") != 100
    ):
        raise ValueError("training dataset manifest safety contract disagrees")
    dataset_record = dataset_manifest.get("dataset") or {}
    subset_record = dataset_manifest.get("frozen_subset_manifest") or {}
    if (
        Path(str(dataset_record.get("path", ""))).resolve() != frozen["dataset"][0]
        or dataset_record.get("sha256") != TRAIN_DATASET_SHA256
        or Path(str(subset_record.get("path", ""))).resolve()
        != frozen["subset_manifest"][0]
        or subset_record.get("sha256") != TRAIN_SUBSET_SHA256
    ):
        raise ValueError("training dataset manifest provenance binding disagrees")
    checkpoint_dataset = checkpoint_manifest.get("dataset") or {}
    if (
        Path(str(checkpoint_dataset.get("path", ""))).resolve() != frozen["dataset"][0]
        or checkpoint_dataset.get("sha256") != TRAIN_DATASET_SHA256
        or Path(str(checkpoint_dataset.get("manifest_path", ""))).resolve()
        != frozen["dataset_manifest"][0]
        or checkpoint_dataset.get("manifest_sha256") != TRAIN_DATASET_MANIFEST_SHA256
        or checkpoint_dataset.get("source_subset_sha256") != TRAIN_SUBSET_SHA256
    ):
        raise ValueError("checkpoint training-dataset provenance binding disagrees")
    if (
        subset_manifest.get("schema") != "boxfusion.ca1m_native_b6_train_subset.v1"
        or subset_manifest.get("selection", {}).get("subset_size") != 100
    ):
        raise ValueError("training subset manifest contract disagrees")
    train_scenes = tuple(str(row.get("scene_id")) for row in subset_manifest.get("entries", []))
    if len(train_scenes) != 100 or len(set(train_scenes)) != 100:
        raise ValueError("training subset does not contain 100 unique scenes")
    if (
        collection_manifest.get("schema") != "boxfusion.ca1m_native_b6_train_collection.v1"
        or collection_manifest.get("complete") is not True
        or collection_manifest.get("train_only") is not True
        or collection_manifest.get("validation_ground_truth_access") is not False
        or collection_manifest.get("evaluation_invoked") is not False
        or collection_manifest.get("scene_count") != 100
        or tuple(str(row.get("scene_id")) for row in collection_manifest.get("scenes", []))
        != train_scenes
    ):
        raise ValueError("training collection manifest contract disagrees")
    val_scenes = tuple(
        line.strip() for line in frozen["full_val_scene_list"][0].read_text().splitlines()
        if line.strip()
    )
    overlap = sorted(set(train_scenes) & set(val_scenes))
    if len(val_scenes) != 107 or len(set(val_scenes)) != 107 or overlap:
        raise ValueError(f"train100/full-val107 partition overlap: {overlap}")
    return {
        label: {"path": str(path), "sha256": expected}
        for label, (path, expected) in frozen.items()
    } | {"train_scenes": 100, "validation_scenes": 107, "intersection": []}


def validate_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = regular(args.checkpoint, "native-B6 checkpoint")
    manifest_path = regular(args.checkpoint_manifest, "native-B6 checkpoint manifest")
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("native-B6 checkpoint SHA256 drifted")
    if sha256(manifest_path) != CHECKPOINT_MANIFEST_SHA256:
        raise ValueError("native-B6 checkpoint manifest SHA256 drifted")
    manifest = read_json(manifest_path, "native-B6 checkpoint manifest")
    if (
        manifest.get("activation_authorized") is not True
        or manifest.get("train_only") is not True
        or manifest.get("validation_ground_truth_access") is not False
        or manifest.get("validation_prediction_access") is not False
    ):
        raise ValueError("native-B6 checkpoint authorization contract disagrees")
    record = manifest.get("checkpoint")
    if not isinstance(record, dict) or (
        Path(str(record.get("path", ""))).resolve() != checkpoint
        or record.get("sha256") != CHECKPOINT_SHA256
    ):
        raise ValueError("native-B6 checkpoint manifest binding disagrees")
    provenance = validate_training_provenance(args, manifest)
    return {
        "checkpoint": str(checkpoint), "manifest": str(manifest_path),
        "training_provenance": provenance,
    }


def validate_upstream(args: argparse.Namespace) -> dict[str, Any]:
    root = directory(args.official_root, "clean upstream root")
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    if head != UPSTREAM_COMMIT:
        raise ValueError("clean upstream commit drifted")
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--short", "--untracked-files=no"],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("clean upstream has tracked working-tree changes")
    hashes: dict[str, str] = {}
    for relative, expected in UPSTREAM_FILES.items():
        path = regular(root / relative, f"upstream {relative}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"upstream {relative} SHA256 drifted")
        hashes[relative] = actual
    return {"root": str(root), "commit": head, "files": hashes}


def validate_eval_view(
    eval_root: Path, data_root: Path, scenes: Sequence[str]
) -> Path:
    evaluation = directory(eval_root, "exact canonical103 evaluation view")
    data = directory(data_root, "canonical CA-1M data root")
    actual = {entry.name for entry in evaluation.iterdir()}
    if actual != set(scenes):
        raise ValueError("evaluation view is not the exact canonical103 scene set")
    for scene in scenes:
        source = data / scene
        view = evaluation / scene
        if not source.is_dir() or source.is_symlink():
            raise ValueError(f"canonical data scene is missing/non-regular: {source}")
        if not view.is_symlink() or view.resolve() != source.resolve():
            raise ValueError(f"evaluation view scene is not the exact data symlink: {view}")
    return evaluation


def gt_inventory(eval_root: Path, scenes: Sequence[str]) -> dict[str, Any]:
    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        path = eval_root / scene / "after_filter_boxes.npy"
        path = regular(path, f"{scene} canonical public ground truth")
        file_hash = sha256(path)
        size = path.stat().st_size
        digest.update(f"{scene}\t{size}\t{file_hash}\n".encode())
        rows.append({"scene_id": scene, "bytes": size, "sha256": file_hash})
    return {"scenes": len(rows), "inventory_sha256": digest.hexdigest(), "files": rows}


def parse_metrics(text: str) -> dict[str, dict[str, float]]:
    rows = METRIC_PATTERN.findall(text)
    if len(rows) != len(THRESHOLDS) * 3:
        raise ValueError(f"expected 9 official evaluator metrics, found {len(rows)}")
    result: dict[str, dict[str, float]] = {}
    for offset, threshold in enumerate(THRESHOLDS):
        chunk = rows[offset * 3 : offset * 3 + 3]
        values = {name: float(value) for name, value in chunk}
        if tuple(name for name, _ in chunk) != ("mAP", "APrec", "ARecall"):
            raise ValueError("official evaluator metric order/schema disagrees")
        result[f"{threshold:.2f}"] = values
    return result


def validate_eval_batches(text: str, scenes: Sequence[str]) -> None:
    batches = re.findall(r"^Eval batch: ([0-9]+) scan_idx ([0-9]{8})$", text, re.MULTILINE)
    expected = [(str(index), scene) for index, scene in enumerate(sorted(scenes))]
    if batches != expected:
        raise ValueError(
            f"official evaluator did not process exact103 batches: "
            f"observed={len(batches)}, expected=103"
        )


def metric_delta(
    baseline: Mapping[str, Mapping[str, float]],
    active: Mapping[str, Mapping[str, float]],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    delta: dict[str, dict[str, float]] = {}
    points: dict[str, dict[str, float]] = {}
    for threshold in (f"{value:.2f}" for value in THRESHOLDS):
        delta[threshold] = {
            name: float(active[threshold][name] - baseline[threshold][name])
            for name in ("mAP", "APrec", "ARecall")
        }
        points[threshold] = {name: 100.0 * value for name, value in delta[threshold].items()}
    return delta, points


def _write_create_only(path: Path, data: bytes, mode: int = 0o444) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = handle.name
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        path.chmod(mode)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def evaluator_argv(
    python: Path, evaluator: Path, eval_view: Path, prediction_root: Path, gpu: int
) -> list[str]:
    return [
        str(python), str(evaluator), "--dataset", "ca1m",
        "--data_path", str(eval_view), "--pred_root", str(prediction_root),
        "--ap_iou_thresholds", "0.15,0.25,0.5", "--batch_size", "1",
        "--cluster_sampling", "seed_fps", "--use_3d_nms", "--use_cls_nms",
        "--per_class_proposal", "--gpu", str(gpu),
    ]


def validate_short_tmp_root(path: Path) -> Path:
    resolved = Path(path).resolve()
    # PyTorch's worker creates ``TMPDIR/pymp-*/listener-*``.  Keeping the
    # caller-controlled prefix below 48 bytes leaves margin under Linux's
    # 108-byte AF_UNIX sockaddr limit even for both nested evaluation phases.
    length = len(os.fsencode(str(resolved)))
    if length > 48:
        raise ValueError(
            f"evaluation tmp root is too long for official DataLoader AF_UNIX "
            f"paths ({length}>48 bytes): {resolved}"
        )
    return resolved


def run_evaluator(
    *, argv: Sequence[str], work_root: Path, log_path: Path, runtime_tmp: Path,
    scenes: Sequence[str],
) -> tuple[dict[str, dict[str, float]], str]:
    work_root.mkdir(parents=True, exist_ok=False)
    runtime_tmp.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONUNBUFFERED": "1",
        "TMPDIR": str(runtime_tmp), "TMP": str(runtime_tmp), "TEMP": str(runtime_tmp),
        "MPLCONFIGDIR": str(runtime_tmp / "mplconfig"),
    })
    process = subprocess.run(
        list(argv), cwd=work_root, env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False,
    )
    _write_create_only(log_path, process.stdout.encode())
    if process.returncode != 0:
        raise RuntimeError(
            f"official CA-1M evaluator failed ({process.returncode}); inspect {log_path}"
        )
    validate_eval_batches(process.stdout, scenes)
    return parse_metrics(process.stdout), sha256(log_path)


def common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--collection-manifest", type=Path, required=True)
    parser.add_argument("--identity-audit", type=Path, required=True)
    parser.add_argument("--record-completion-root", type=Path, required=True)
    parser.add_argument("--observer-completion-root", type=Path, required=True)
    parser.add_argument("--anchor-root", type=Path, required=True)
    parser.add_argument("--observer-root", type=Path, required=True)
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--training-dataset", type=Path, required=True)
    parser.add_argument("--training-dataset-manifest", type=Path, required=True)
    parser.add_argument("--training-subset-manifest", type=Path, required=True)
    parser.add_argument("--training-collection-manifest", type=Path, required=True)
    parser.add_argument("--full-val-scene-list", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--eval-view", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="mode", required=True)
    preflight = sub.add_parser("preflight")
    common_parser(preflight)
    evaluate = sub.add_parser("evaluate")
    common_parser(evaluate)
    evaluate.add_argument("--active-root", type=Path, required=True)
    evaluate.add_argument("--active-report", type=Path, required=True)
    evaluate.add_argument("--log-root", type=Path, required=True)
    evaluate.add_argument("--tmp-root", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--gpu", type=int, default=0)
    return result


def validate_base(
    args: argparse.Namespace,
) -> tuple[tuple[str, ...], dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    scenes = read_scenes(args.scene_list)
    collection = validate_collection_chain(args, scenes)
    checkpoint = validate_checkpoint(args)
    upstream = validate_upstream(args)
    eval_view = validate_eval_view(args.eval_view, args.data_root, scenes)
    executable(args.python, "evaluation Python")
    return scenes, collection, checkpoint, upstream, eval_view


def main() -> int:
    args = parser().parse_args()
    scenes, collection, checkpoint, upstream, eval_view = validate_base(args)
    if args.mode == "preflight":
        print(json.dumps({
            "ok": True, "mode": "preflight", "scenes": len(scenes),
            "rows": collection["rows"], "outputs_created": False,
            "official_commit": upstream["commit"],
        }, indent=2, sort_keys=True))
        return 0

    for path, label in (
        (args.output, "paired report"), (args.log_root, "evaluation log root"),
        (args.tmp_root, "evaluation tmp root"),
    ):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing {label}: {path}")
    args.tmp_root = validate_short_tmp_root(args.tmp_root)
    active = validate_active_chain(
        args, scenes, collection["anchors"], collection["diagnostics"]
    )
    evaluator = Path(upstream["root"]) / "evaluation/eval_ca1m.py"
    python = executable(args.python, "evaluation Python")
    anchor_root = directory(args.anchor_root, "same-run anchor root")
    active_root = directory(args.active_root, "active root")
    baseline_argv = evaluator_argv(python, evaluator, eval_view, anchor_root, args.gpu)
    active_argv = evaluator_argv(python, evaluator, eval_view, active_root, args.gpu)
    inventory_before = gt_inventory(eval_view, scenes)
    baseline_metrics, baseline_log_sha = run_evaluator(
        argv=baseline_argv,
        work_root=args.tmp_root / "baseline_work",
        log_path=args.log_root / "baseline.log",
        runtime_tmp=args.tmp_root / "baseline_tmp",
        scenes=scenes,
    )
    inventory_after_baseline = gt_inventory(eval_view, scenes)
    if inventory_after_baseline != inventory_before:
        raise RuntimeError("canonical103 ground-truth inventory changed after baseline evaluation")
    active_metrics, active_log_sha = run_evaluator(
        argv=active_argv,
        work_root=args.tmp_root / "active_work",
        log_path=args.log_root / "active.log",
        runtime_tmp=args.tmp_root / "active_tmp",
        scenes=scenes,
    )
    inventory_after_active = gt_inventory(eval_view, scenes)
    if inventory_after_active != inventory_before:
        raise RuntimeError("canonical103 ground-truth inventory changed after active evaluation")
    delta, delta_points = metric_delta(baseline_metrics, active_metrics)
    gate_checks = {
        "delta_AP15_at_least_minus_0.005": delta["0.15"]["mAP"] >= -0.005,
        "delta_AP25_at_least_minus_0.005": delta["0.25"]["mAP"] >= -0.005,
        "delta_AP50_at_least_plus_0.005": delta["0.50"]["mAP"] >= 0.005,
    }
    gate_pass = all(gate_checks.values())
    report = {
        "schema": REPORT_SCHEMA,
        "complete": True,
        "dataset": "CA1M",
        "dataset_split": "official_validation_canonical103",
        "official_public_gt_subset": "103/107",
        "paper_full107_comparable": False,
        "paired_same_run": True,
        "score_only": True,
        "obb_geometry_row_count_order_unchanged": True,
        "metric_geometry": (
            "official 8-corner box3d_iou_v2 implementation; actual metric is "
            "world enclosing-AABB IoU, not true OBB IoU"
        ),
        "evaluation_ground_truth_access": True,
        "scene_list": str(Path(args.scene_list).resolve()),
        "scene_list_sha256": SCENE_LIST_SHA256,
        "scenes": 103,
        "rows": active["rows"],
        "changed_scores": active["changed"],
        "score_formula": {
            "formula": "0.4*detector_score + 0.6*monotonic_projected_quality_rank",
            "independently_verified_rows": active["formula_verified_rows"],
            "real_anchor_scores_finite_nonconstant": True,
            "real_anchor_score_min": collection["real_score_min"],
            "real_anchor_score_max": collection["real_score_max"],
        },
        "collection_manifest": {
            "path": str(Path(args.collection_manifest).resolve()),
            "sha256": sha256(Path(args.collection_manifest).resolve()),
        },
        "identity_audit": {
            "path": str(Path(args.identity_audit).resolve()),
            "sha256": sha256(Path(args.identity_audit).resolve()),
        },
        "active_report": {
            "path": str(Path(args.active_report).resolve()),
            "sha256": sha256(Path(args.active_report).resolve()),
        },
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()), "sha256": CHECKPOINT_SHA256,
            "manifest_path": str(Path(args.checkpoint_manifest).resolve()),
            "manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
        },
        "training_provenance": checkpoint["training_provenance"],
        "official_evaluator": upstream,
        "evaluation_contract": {
            "eval_view": str(eval_view),
            "isolated_short_tmp_root": str(args.tmp_root),
            "interruption_recovery": "create-only; select a new paired TAG",
            "baseline_argv": baseline_argv,
            "active_argv": active_argv,
            "only_argv_difference": "--pred_root value",
            "thresholds": list(THRESHOLDS),
            "eval_batch_count": {"baseline": 103, "active": 103},
        },
        "ground_truth_inventory": {
            "before": inventory_before,
            "after_baseline_sha256": inventory_after_baseline["inventory_sha256"],
            "after_active_sha256": inventory_after_active["inventory_sha256"],
            "unchanged": True,
        },
        "logs": {
            "baseline": {"path": str((args.log_root / "baseline.log").resolve()), "sha256": baseline_log_sha},
            "active": {"path": str((args.log_root / "active.log").resolve()), "sha256": active_log_sha},
        },
        "baseline": baseline_metrics,
        "active": active_metrics,
        "delta": delta,
        "delta_percentage_points": delta_points,
        "validation_gate": {
            "decision": "PASS" if gate_pass else "FAIL",
            "activation_authorized": gate_pass,
            "checks": gate_checks,
            "thresholds": {
                "min_delta_AP15": -0.005,
                "min_delta_AP25": -0.005,
                "min_delta_AP50": 0.005,
            },
        },
    }
    _write_create_only(
        args.output, (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    )
    print(json.dumps({
        "complete": True,
        "AP15_baseline_active_delta": [
            baseline_metrics["0.15"]["mAP"], active_metrics["0.15"]["mAP"], delta["0.15"]["mAP"]
        ],
        "AP25_baseline_active_delta": [
            baseline_metrics["0.25"]["mAP"], active_metrics["0.25"]["mAP"], delta["0.25"]["mAP"]
        ],
        "AP50_baseline_active_delta": [
            baseline_metrics["0.50"]["mAP"], active_metrics["0.50"]["mAP"], delta["0.50"]["mAP"]
        ],
        "report": str(args.output.resolve()),
        "validation_gate": "PASS" if gate_pass else "FAIL",
    }, indent=2, sort_keys=True))
    return 0 if gate_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
