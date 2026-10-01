#!/usr/bin/env python3
"""Join train-only CA-1M native-B6 evidence with derived 3D-box targets.

The tool is deliberately offline and fail-closed.  It accepts only scenes in a
frozen official-train subset, refuses any scene listed by the official
validation URL inventory, and never discovers data by directory listing.  The
continuous target is the maximum IoU used by ``evaluation/eval_ca1m.py``:
intersection-over-union of the world-axis enclosing AABBs of two corner sets.

Inputs per scene are:

* one immutable native-B6 observer NPZ;
* one sealed final-base ``*_boxes.pkl`` prediction queried directly offline;
* one create-only native train-scene directory with derived train GT.

The output NPZ and JSON manifest are create-only and contain a deterministic,
scene-grouped five-fold assignment.  Fold zero is the frozen 20% development
partition; no CA-1M validation GT is read or accepted.
"""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import pickle
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from boxfusion.ca1m_native_b6_observer import FEATURE_NAMES


DATASET_SCHEMA = "boxfusion.ca1m_native_b6_quality_dataset.v1"
MANIFEST_SCHEMA = "boxfusion.ca1m_native_b6_quality_dataset_manifest.v1"
SUBSET_SCHEMA = "boxfusion.ca1m_native_b6_train_subset.v1"
OBSERVER_SCHEMA = "boxfusion.ca1m_native_b6_observer.v1"
GT_SCHEMA = "boxfusion.ca1m_native_b6_train_scene.v1"
COLLECTION_SCHEMA = "boxfusion.ca1m_native_b6_train_collection.v1"
COMPLETION_SCHEMA = "boxfusion.ca1m_native_b6_train_scene_completion.v1"
FINAL_BASE_COLLECTION_SCHEMA = (
    "boxfusion.ca1m_native_b6_final_base_train_collection.v2"
)
FINAL_BASE_COMPLETION_SCHEMA = (
    "boxfusion.ca1m_native_b6_final_base_scene_completion.v2"
)
FINAL_BASE_IDENTITY_SCHEMA = "boxfusion.ca1m_final_base_identity_audit.v1"
FINAL_BASE_PAIRED_REPORT_SCHEMA = "boxfusion.ca1m_final_base_paired_eval.v1"
SUPPORTED_COLLECTION_SCHEMAS = (
    COLLECTION_SCHEMA,
    FINAL_BASE_COLLECTION_SCHEMA,
)
TARGET_SCHEMA = "ca1m_evaluator_box3d_iou_v2_world_enclosing_aabb.v1"
DEFAULT_SPLIT_NAMESPACE = "boxfusion.ca1m-native-b6.scene-folds.v1"
SCENE = re.compile(r"^[0-9]{8}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _create_only(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
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
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to replace immutable artifact: {path}") from error
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _regular(path: Path, label: str) -> Path:
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    path = path.resolve()
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        raise ValueError(f"{label} must be a non-empty regular file: {path}")
    return path


def _scene_scalar(value: Any, label: str) -> str:
    array = np.asarray(value)
    if array.dtype.hasobject or array.ndim != 0:
        raise ValueError(f"{label} must be a non-object scalar string")
    result = str(array.item())
    if SCENE.fullmatch(result) is None:
        raise ValueError(f"invalid CA-1M scene id in {label}: {result!r}")
    return result


def read_scene_list(path: Path) -> tuple[str, ...]:
    path = _regular(path, "train-only scene list")
    scenes = tuple(row.strip() for row in path.read_text().splitlines() if row.strip())
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list must be non-empty and duplicate-free")
    if any(SCENE.fullmatch(scene) is None for scene in scenes):
        raise ValueError("scene list contains an invalid CA-1M scene id")
    return scenes


def read_validation_ids(path: Path) -> tuple[str, ...]:
    path = _regular(path, "official validation URL list")
    ids: list[str] = []
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        url = raw.strip()
        if not url:
            continue
        parsed = urlsplit(url)
        matched = re.fullmatch(r"/datasets/ca1m/val/ca1m-val-([0-9]{8})\.tar", parsed.path)
        if parsed.scheme != "https" or parsed.netloc != "ml-site.cdn-apple.com" or matched is None:
            raise ValueError(f"{path}:{line_number}: invalid official CA-1M val URL")
        ids.append(matched.group(1))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("validation URL list must be non-empty and duplicate-free")
    return tuple(ids)


def load_subset_manifest(path: Path) -> tuple[dict[str, Any], tuple[str, ...], str]:
    path = _regular(path, "frozen train subset manifest")
    payload = json.loads(path.read_text())
    if payload.get("schema") != SUBSET_SCHEMA:
        raise ValueError("unsupported train subset manifest schema")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("train subset manifest entries are missing")
    scenes = tuple(str(row.get("scene_id")) for row in entries)
    if len(scenes) != len(set(scenes)) or any(SCENE.fullmatch(x) is None for x in scenes):
        raise ValueError("train subset manifest has invalid/duplicate scenes")
    if [row.get("rank") for row in entries] != list(range(len(entries))):
        raise ValueError("train subset manifest ranks are not contiguous")
    selection = payload.get("selection") or {}
    selection_bytes = ("\n".join(scenes) + "\n").encode("ascii")
    if (
        selection.get("subset_size") != len(scenes)
        or selection.get("scene_ids_sha256")
        != hashlib.sha256(selection_bytes).hexdigest()
    ):
        raise ValueError("train subset selection count/hash disagrees")
    for row, scene in zip(entries, scenes):
        parsed = urlsplit(str(row.get("url", "")))
        expected_path = f"/datasets/ca1m/train/ca1m-train-{scene}.tar"
        if (
            parsed.scheme != "https"
            or parsed.netloc != "ml-site.cdn-apple.com"
            or parsed.path != expected_path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(f"train subset contains a non-official URL for {scene}")
    safety = payload.get("safety_contract") or {}
    if (
        safety.get("train_only") is not True
        or safety.get("validation_ground_truth_access") is not False
        or int(safety.get("validation_scene_overlap_count", -1)) != 0
        or safety.get("training_started") is not False
    ):
        raise ValueError("train subset safety contract is not suitable for training")
    return payload, scenes, sha256_file(path)


def validate_fixed10_paired_report_record(
    record: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    path = _regular(
        Path(str(record.get("path", ""))), "authoritative fixed10 paired report"
    )
    if (
        record.get("schema") != FINAL_BASE_PAIRED_REPORT_SCHEMA
        or record.get("sha256") != sha256_file(path)
        or record.get("role")
        != "authoritative_fixed10_train100_and_retraining_gate"
    ):
        raise ValueError("fixed10 paired report binding disagrees")
    payload = json.loads(path.read_text())
    for key, expected in {
        "schema": FINAL_BASE_PAIRED_REPORT_SCHEMA,
        "complete": True,
        "dataset": "CA1M",
        "split": "validation_fixed10",
        "scene_count": 10,
        "paired_official_evaluation": True,
        "positive_map_at_all_thresholds": True,
        "training_invoked": False,
    }.items():
        if payload.get(key) != expected:
            raise ValueError(f"fixed10 paired report field {key} disagrees")
    decision = payload.get("decision") or {}
    for key, expected in {
        "train100_final_base_collection_authorized": True,
        "ca1m_native_b6_retraining_required": True,
        "canonical_active_authorized": False,
    }.items():
        if decision.get(key) != expected:
            raise ValueError(f"fixed10 paired decision field {key} disagrees")
    for threshold in ("AP15", "AP25", "AP50"):
        try:
            gain = float(payload["delta"][threshold]["mAP"])
            active = float(payload["active"][threshold]["mAP"])
            control = float(payload["control"][threshold]["mAP"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"fixed10 paired report lacks {threshold} mAP values"
            ) from error
        if not all(np.isfinite(value) for value in (gain, active, control)):
            raise ValueError(f"fixed10 paired report {threshold} mAP is non-finite")
        if gain <= 0.0:
            raise ValueError(f"fixed10 paired report {threshold} mAP delta is not positive")
        if not np.isclose(active - control, gain, rtol=0, atol=5e-7):
            raise ValueError(f"fixed10 paired report {threshold} mAP delta is inconsistent")
    return payload, path


def load_collection_manifest(
    path: Path,
    observer_completion_root: Path,
    *,
    scenes: Sequence[str],
    subset_sha: str,
    scene_ids_sha: str,
) -> tuple[dict[str, Any], Path, str, dict[str, dict[str, Any]]]:
    """Bind the join to the sealed, evaluation-free collection artifacts."""

    path = _regular(path, "train-only collection manifest")
    root = Path(observer_completion_root)
    if root.is_symlink():
        raise ValueError(f"observer completion root must not be a symlink: {root}")
    root = root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"observer completion root must be a directory: {root}")
    payload = json.loads(path.read_text())
    schema = str(payload.get("schema", ""))
    if schema not in SUPPORTED_COLLECTION_SCHEMAS:
        raise ValueError("unsupported train collection schema")
    expected_scalars = {
        "schema": schema,
        "complete": True,
        "train_only": True,
        "evaluation_invoked": False,
        "validation_ground_truth_access": False,
        "scene_count": len(scenes),
        "scene_ids_sha256": scene_ids_sha,
        "subset_manifest_sha256": subset_sha,
    }
    for key, expected in expected_scalars.items():
        if payload.get(key) != expected:
            raise ValueError(f"train collection field {key} disagrees")
    if schema == FINAL_BASE_COLLECTION_SCHEMA:
        for key, expected in (
            ("validation_prediction_access", False),
            ("official_validation_comparable", False),
            ("geometry_authority", "sealed_final_base_prediction"),
            ("offline_direct_observer", True),
            ("cross_run_boxfusion_replay_invoked", False),
            ("cross_run_exact_identity_required", False),
            ("rgb_pixels_accessed", False),
            ("old_native_b6_diagnostics_reused", False),
            ("old_native_b6_checkpoint_reused", False),
        ):
            if payload.get(key) != expected:
                raise ValueError(f"final-base train collection field {key} disagrees")
        modules = payload.get("source_modules") or {}
        if modules != {
            "selective_boxer_g0": True,
            "clip_appearance_gate": True,
            "reliable_view_top_k": 3,
            "b6_evidence_top_k": 5,
        }:
            raise ValueError("final-base train collection module binding disagrees")
        split = payload.get("split_protocol") or {}
        if split != {
            "kind": "deterministic_scene_grouped_5fold",
            "namespace": DEFAULT_SPLIT_NAMESPACE,
            "deployable_training_folds": [1, 2, 3, 4],
            "untouched_dev_fold": 0,
        }:
            raise ValueError("final-base train collection split protocol disagrees")
        source_record = payload.get("source_final_base_collection") or {}
        source_path = _regular(
            Path(str(source_record.get("path", ""))),
            "source final-base collection manifest",
        )
        if (
            source_record.get("schema") != FINAL_BASE_IDENTITY_SCHEMA
            or source_record.get("sha256") != sha256_file(source_path)
        ):
            raise ValueError("source final-base collection binding disagrees")
        source = json.loads(source_path.read_text())
        source_required = {
            "schema": FINAL_BASE_IDENTITY_SCHEMA,
            "ok": True,
            "dataset": "CA1M",
            "split": "train100",
            "scene_count": len(scenes),
            "ground_truth_access": False,
            "evaluation_invoked": False,
            "training_invoked": False,
            "scannet_learned_b6_or_gate_reused": False,
            "clip_appearance_gate_active": True,
            "reliable_view_top_k": 3,
        }
        for key, expected in source_required.items():
            if source.get(key) != expected:
                raise ValueError(f"source final-base field {key} disagrees")
        if set((source.get("per_scene") or {}).keys()) != set(scenes):
            raise ValueError("source final-base scenes differ from frozen train subset")
        validate_fixed10_paired_report_record(
            payload.get("fixed10_paired_report") or {}
        )
        source_root = Path(str(payload.get("source_final_base_root", "")))
        if source_root.is_symlink():
            raise ValueError("source final-base root must not be a symlink")
        source_root = source_root.resolve()
        if not source_root.is_dir() or source_root.is_symlink():
            raise ValueError("source final-base root is missing/unsafe")
    else:
        source = None
        source_root = None
    rows = payload.get("scenes")
    if not isinstance(rows, list) or [row.get("scene_id") for row in rows] != list(scenes):
        raise ValueError("train collection scenes do not exactly match the frozen subset")
    expected_files = {f"{scene}.json" for scene in scenes}
    actual_files = {item.name for item in root.iterdir() if item.is_file()}
    if actual_files != expected_files:
        raise ValueError("observer completion root is not the exact frozen scene set")
    completions: dict[str, dict[str, Any]] = {}
    for scene, row in zip(scenes, rows):
        completion_path = _regular(root / f"{scene}.json", "observer completion")
        completion_sha = sha256_file(completion_path)
        if completion_sha != str(row.get("observer_completion_sha256")):
            raise ValueError(f"{scene}: observer completion SHA256 disagrees")
        completion = json.loads(completion_path.read_text())
        required = {
            "schema": (
                FINAL_BASE_COMPLETION_SCHEMA
                if schema == FINAL_BASE_COLLECTION_SCHEMA
                else COMPLETION_SCHEMA
            ),
            "phase": (
                "sealed_final_base_offline_native_b6_observer"
                if schema == FINAL_BASE_COLLECTION_SCHEMA
                else "g0_native_b6_observer"
            ),
            "scene_id": scene,
            "complete": True,
            "train_only": True,
            "evaluation_invoked": False,
            "validation_ground_truth_access": False,
            "output_mutation_authorized": False,
        }
        if schema == FINAL_BASE_COLLECTION_SCHEMA:
            required.update(
                {
                    "validation_prediction_access": False,
                    "geometry_authority": "sealed_final_base_prediction",
                    "offline_direct_observer": True,
                    "cross_run_boxfusion_replay_invoked": False,
                    "cross_run_exact_identity_required": False,
                    "rgb_pixels_accessed": False,
                    "old_native_b6_diagnostics_reused": False,
                    "old_native_b6_checkpoint_reused": False,
                }
            )
        for key, expected in required.items():
            if completion.get(key) != expected:
                raise ValueError(f"{scene}: observer completion field {key} disagrees")
        if schema == FINAL_BASE_COLLECTION_SCHEMA:
            assert source is not None and source_root is not None
            final_anchor = _regular(
                source_root / f"{scene}_boxes.pkl", f"{scene} source final-base anchor"
            )
            final_record = (completion.get("artifacts") or {}).get(
                "final_base_anchor"
            ) or {}
            source_row = source["per_scene"][scene]
            final_sha = sha256_file(final_anchor)
            if (
                Path(str(final_record.get("path", ""))).resolve() != final_anchor
                or final_record.get("sha256") != final_sha
                or source_row.get("active_prediction_sha256") != final_sha
                or row.get("final_base_prediction_sha256") != final_sha
            ):
                raise ValueError(f"{scene}: final-base anchor binding disagrees")
        completions[scene] = completion
    return payload, path, sha256_file(path), completions


def validate_collection_artifact(
    completion: Mapping[str, Any],
    name: str,
    expected_path: Path,
    *,
    scene: str,
) -> None:
    record = (completion.get("artifacts") or {}).get(name) or {}
    actual_path = _regular(expected_path, f"{scene} collection {name}")
    if Path(str(record.get("path", ""))).resolve() != actual_path:
        raise ValueError(f"{scene}: collection {name} path disagrees")
    if str(record.get("sha256")) != sha256_file(actual_path):
        raise ValueError(f"{scene}: collection {name} SHA256 disagrees")


def ca1m_pairwise_iou_v2(predicted_corners: np.ndarray, gt_corners: np.ndarray) -> np.ndarray:
    """Vectorized exact equivalent of CA-1M ``box3d_iou_v2`` 3D IoU."""

    predicted = np.asarray(predicted_corners, dtype=np.float64)
    targets = np.asarray(gt_corners, dtype=np.float64)
    for value, name in ((predicted, "predicted_corners"), (targets, "gt_corners")):
        if value.ndim != 3 or value.shape[1:] != (8, 3) or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite [N,8,3]")
    pred_min, pred_max = predicted.min(axis=1), predicted.max(axis=1)
    gt_min, gt_max = targets.min(axis=1), targets.max(axis=1)
    pred_size, gt_size = pred_max - pred_min, gt_max - gt_min
    if np.any(pred_size <= 0.0) or np.any(gt_size <= 0.0):
        raise ValueError("CA-1M evaluator target boxes must have positive AABB extent")
    if not len(predicted) or not len(targets):
        return np.zeros((len(predicted), len(targets)), dtype=np.float64)
    inter_min = np.maximum(pred_min[:, None, :], gt_min[None, :, :])
    inter_max = np.minimum(pred_max[:, None, :], gt_max[None, :, :])
    inter = np.prod(np.maximum(inter_max - inter_min, 0.0), axis=2)
    pred_volume = np.prod(pred_size, axis=1)
    gt_volume = np.prod(gt_size, axis=1)
    union = pred_volume[:, None] + gt_volume[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0.0)


def _prediction_rows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the trusted local BoxFusion pickle and enforce its complete schema."""

    path = _regular(path, "same-run G0 prediction")
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # trusted pipeline artifact; schema checked below
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], list):
        raise ValueError(f"invalid BoxFusion prediction container: {path}")
    corners: list[np.ndarray] = []
    scores: list[float] = []
    for index, row in enumerate(payload[0]):
        if not isinstance(row, tuple) or len(row) != 3 or type(row[0]) is not int or row[0] != 0:
            raise ValueError(f"{path}: invalid class-agnostic prediction row {index}")
        box = np.asarray(row[1])
        score = row[2]
        if box.shape != (8, 3) or not np.issubdtype(box.dtype, np.floating):
            raise ValueError(f"{path}: prediction corner row {index} is invalid")
        if not np.isfinite(box).all() or isinstance(score, bool) or not np.isscalar(score):
            raise ValueError(f"{path}: prediction row {index} is non-finite")
        numeric_score = float(score)
        if not np.isfinite(numeric_score) or not 0.0 <= numeric_score <= 1.0:
            raise ValueError(f"{path}: prediction score row {index} is invalid")
        corners.append(np.asarray(box, dtype=np.float32))
        scores.append(numeric_score)
    return (
        np.stack(corners) if corners else np.empty((0, 8, 3), dtype=np.float32),
        np.asarray(scores, dtype=np.float32),
    )


def _observer(path: Path, scene: str) -> dict[str, np.ndarray]:
    path = _regular(path, "native-B6 observer diagnostic")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "schema", "complete", "observer_only", "mutation_enabled",
            "ground_truth_access", "scene_id", "result_indices", "stable_ids",
            "corners", "scores", "feature_names", "features", "valid_evidence",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path}: observer fields missing: {sorted(missing)}")
        result = {name: np.array(archive[name], copy=True) for name in required}
    if str(result["schema"].item()) != OBSERVER_SCHEMA or _scene_scalar(result["scene_id"], str(path)) != scene:
        raise ValueError(f"{scene}: observer schema/scene mismatch")
    for field, expected in (
        ("complete", True), ("observer_only", True),
        ("mutation_enabled", False), ("ground_truth_access", False),
    ):
        if result[field].shape != () or bool(result[field].item()) is not expected:
            raise ValueError(f"{scene}: observer safety scalar {field} disagrees")
    corners = result["corners"]
    count = len(corners)
    if corners.shape != (count, 8, 3) or result["scores"].shape != (count,):
        raise ValueError(f"{scene}: observer prediction tensors are invalid")
    if result["features"].shape[0] != count or result["valid_evidence"].shape != (count,):
        raise ValueError(f"{scene}: observer row tensors are misaligned")
    if not np.array_equal(result["result_indices"], np.arange(count, dtype=np.int64)):
        raise ValueError(f"{scene}: observer result_indices are not identity")
    if result["stable_ids"].shape != (count,) or len(np.unique(result["stable_ids"])) != count:
        raise ValueError(f"{scene}: observer stable_ids are invalid")
    if not np.isfinite(corners).all() or not np.isfinite(result["scores"]).all() or not np.isfinite(result["features"]).all():
        raise ValueError(f"{scene}: observer arrays are non-finite")
    return result


def _nested_sha(payload: Any, label: str) -> str:
    if isinstance(payload, Mapping):
        value = payload.get("sha256")
    else:
        value = payload
    value = str(value)
    if SHA256.fullmatch(value) is None:
        raise ValueError(f"invalid SHA256 in {label}")
    return value


def _derived_gt(root: Path, scene: str, subset_sha: str) -> tuple[np.ndarray, Path, dict[str, Any]]:
    scene_root = root / scene
    primary = _regular(scene_root / "derived_train_gt_boxes.npy", "derived train GT")
    compatibility = _regular(scene_root / "after_filter_boxes.npy", "compatibility train GT")
    manifest_path = _regular(scene_root / "derived_train_gt_manifest.json", "derived train GT manifest")
    manifest = json.loads(manifest_path.read_text())
    required = {
        "schema": GT_SCHEMA, "scene_id": scene, "source_split": "train",
        "train_only": True, "validation_scene_overlap": False,
        "validation_ground_truth_access": False, "derived_train_gt": True,
        "official_validation_comparable": False,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"{scene}: derived GT contract field {key} disagrees")
    if _nested_sha(manifest.get("frozen_subset_manifest"), f"{scene}.subset") != subset_sha:
        raise ValueError(f"{scene}: derived GT was built from another frozen subset")
    _nested_sha(manifest.get("source_tar"), f"{scene}.source_tar")
    primary_sha, compatibility_sha = sha256_file(primary), sha256_file(compatibility)
    if primary_sha != str(manifest.get("derived_train_gt_sha256")):
        raise ValueError(f"{scene}: derived GT SHA256 mismatch")
    if compatibility_sha != str(manifest.get("compat_after_filter_sha256")):
        raise ValueError(f"{scene}: compatibility GT SHA256 mismatch")
    primary_value = np.load(primary, allow_pickle=False)
    compatibility_value = np.load(compatibility, allow_pickle=False)
    if not np.array_equal(primary_value, compatibility_value):
        raise ValueError(f"{scene}: native and compatibility GT arrays differ")
    value = np.asarray(primary_value, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (8, 3) or not len(value) or not np.isfinite(value).all():
        raise ValueError(f"{scene}: derived GT must be non-empty finite [G,8,3]")
    return value, manifest_path, manifest


def assign_scene_folds(scenes: Sequence[str], namespace: str, fold_count: int = 5) -> dict[str, int]:
    if fold_count < 2 or len(scenes) < fold_count:
        raise ValueError("scene-grouped splitting requires at least one scene per fold")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("split namespace must be non-empty")
    ranked = sorted(
        scenes,
        key=lambda scene: (
            hashlib.sha256(f"{namespace}\0{scene}".encode()).hexdigest(), scene
        ),
    )
    return {scene: rank % fold_count for rank, scene in enumerate(ranked)}


def build(args: argparse.Namespace) -> dict[str, Any]:
    roots = (
        (Path(args.observer_root), "observer root"),
        (Path(args.prediction_root), "prediction root"),
        (Path(args.gt_root), "GT root"),
    )
    for path, label in roots:
        if path.is_symlink():
            raise ValueError(f"{label} must not be a symlink: {path}")
    observer_root, prediction_root, gt_root = (path.resolve() for path, _ in roots)
    for path, label in ((observer_root, "observer root"), (prediction_root, "prediction root"), (gt_root, "GT root")):
        if not path.is_dir() or path.is_symlink():
            raise ValueError(f"{label} must be a regular directory: {path}")
    raw_output = Path(args.output)
    raw_manifest_output = Path(args.manifest_output)
    if raw_output.is_symlink() or raw_manifest_output.is_symlink():
        raise ValueError("dataset/manifest outputs must not be symlinks")
    output = raw_output.resolve()
    manifest_output = raw_manifest_output.resolve()
    if output == manifest_output or output.suffix.lower() != ".npz" or manifest_output.suffix.lower() != ".json":
        raise ValueError("dataset/manifest outputs must be distinct .npz/.json paths")
    if output.exists() or output.is_symlink() or manifest_output.exists() or manifest_output.is_symlink():
        raise FileExistsError("refusing to start with an existing dataset/manifest output")
    subset, subset_scenes, subset_sha = load_subset_manifest(args.subset_manifest)
    scenes = read_scene_list(args.scene_list)
    if scenes != subset_scenes:
        raise ValueError("scene list must exactly equal the frozen subset entry order")
    collection, collection_path, collection_sha, completions = load_collection_manifest(
        args.collection_manifest,
        args.observer_completion_root,
        scenes=scenes,
        subset_sha=subset_sha,
        scene_ids_sha=str(subset["selection"]["scene_ids_sha256"]),
    )
    if (
        collection["schema"] == FINAL_BASE_COLLECTION_SCHEMA
        and args.split_namespace != DEFAULT_SPLIT_NAMESPACE
    ):
        raise ValueError(
            "final-base native-B6 v2 must preserve the established CA scene folds"
        )
    validation_ids = set(read_validation_ids(args.val_url_list))
    overlap = sorted(set(scenes) & validation_ids)
    if overlap:
        raise ValueError("train dataset scene overlaps official validation IDs: " + ",".join(overlap))
    folds = assign_scene_folds(scenes, args.split_namespace, fold_count=5)

    collected: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "quality_features", "target_iou", "scene_ids", "row_indices",
            "stable_ids", "prediction_scores", "prediction_corners",
            "matched_gt_indices", "fold_ids", "dev_mask", "valid_evidence",
        )
    }
    feature_names: tuple[str, ...] | None = None
    scene_reports: list[dict[str, Any]] = []
    for scene in scenes:
        observer_path = observer_root / f"{scene}_ca1m_native_b6.npz"
        prediction_path = prediction_root / f"{scene}_boxes.pkl"
        validate_collection_artifact(
            completions[scene], "native_b6_diagnostic", observer_path, scene=scene
        )
        validate_collection_artifact(
            completions[scene], "prediction", prediction_path, scene=scene
        )
        evidence = _observer(observer_path, scene)
        prediction_corners, prediction_scores = _prediction_rows(prediction_path)
        if not np.array_equal(evidence["corners"], prediction_corners) or not np.array_equal(evidence["scores"], prediction_scores):
            raise ValueError(f"{scene}: observer and sealed final-base predictions differ")
        names = tuple(str(item) for item in np.asarray(evidence["feature_names"]).tolist())
        if not names or len(names) != evidence["features"].shape[1] or len(names) != len(set(names)):
            raise ValueError(f"{scene}: invalid native-B6 feature schema")
        if names != FEATURE_NAMES:
            raise ValueError(f"{scene}: feature schema is not the frozen native-B6 schema")
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError(f"{scene}: native-B6 feature order differs")
        gt, gt_manifest_path, gt_manifest = _derived_gt(gt_root, scene, subset_sha)
        ious = ca1m_pairwise_iou_v2(prediction_corners, gt)
        if len(gt):
            matched = np.argmax(ious, axis=1).astype(np.int64)
            targets = ious[np.arange(len(ious)), matched]
        else:  # _derived_gt currently forbids empty GT, kept for totality.
            matched = np.full(len(prediction_corners), -1, dtype=np.int64)
            targets = np.zeros(len(prediction_corners), dtype=np.float64)
        count = len(prediction_corners)
        fold = folds[scene]
        collected["quality_features"].append(evidence["features"].astype(np.float32))
        # Keep evaluator-grade float64 targets so strict threshold crossings
        # cannot change through float32 quantization at 0.15/0.25/0.50.
        collected["target_iou"].append(targets.astype(np.float64))
        collected["scene_ids"].append(np.full(count, scene, dtype="<U8"))
        collected["row_indices"].append(np.arange(count, dtype=np.int64))
        collected["stable_ids"].append(evidence["stable_ids"].astype(np.int64))
        collected["prediction_scores"].append(prediction_scores)
        collected["prediction_corners"].append(prediction_corners)
        collected["matched_gt_indices"].append(matched)
        collected["fold_ids"].append(np.full(count, fold, dtype=np.int8))
        collected["dev_mask"].append(np.full(count, fold == 0, dtype=np.bool_))
        collected["valid_evidence"].append(evidence["valid_evidence"].astype(np.bool_))
        scene_reports.append({
            "scene_id": scene,
            "fold_id": fold,
            "rows": count,
            "gt_boxes": len(gt),
            "positive_iou15": int(np.count_nonzero(targets > 0.15)),
            "positive_iou25": int(np.count_nonzero(targets > 0.25)),
            "positive_iou50": int(np.count_nonzero(targets > 0.50)),
            "observer": {"path": str(observer_path), "sha256": sha256_file(observer_path)},
            "prediction": {"path": str(prediction_path), "sha256": sha256_file(prediction_path)},
            "derived_gt_manifest": {"path": str(gt_manifest_path), "sha256": sha256_file(gt_manifest_path)},
            "derived_gt_sha256": str(gt_manifest["derived_train_gt_sha256"]),
            "source_tar_sha256": _nested_sha(gt_manifest["source_tar"], f"{scene}.source_tar"),
        })
    assert feature_names is not None
    arrays = {key: np.concatenate(value, axis=0) for key, value in collected.items()}
    sample_count = len(arrays["target_iou"])
    if sample_count < 2 or any(len(value) != sample_count for value in arrays.values()):
        raise ValueError("native-B6 joined dataset has too few/misaligned rows")
    arrays.update({
        "schema": np.asarray(DATASET_SCHEMA),
        "complete": np.asarray(True, dtype=np.bool_),
        "train_only": np.asarray(True, dtype=np.bool_),
        "validation_ground_truth_access": np.asarray(False, dtype=np.bool_),
        "target_schema": np.asarray(TARGET_SCHEMA),
        "feature_names": np.asarray(feature_names, dtype=np.str_),
        "fold_count": np.asarray(5, dtype=np.int64),
        "dev_fold": np.asarray(0, dtype=np.int64),
        "split_namespace": np.asarray(args.split_namespace),
    })
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays)
    _create_only(output, buffer.getvalue())
    dataset_sha = sha256_file(output)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "complete": True,
        "train_only": True,
        "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "validation_scene_overlap_count": 0,
        "activation_authorized": False,
        "training_started": False,
        "target": {
            "schema": TARGET_SCHEMA,
            "definition": "maximum pairwise world-axis enclosing-AABB IoU",
            "matches_evaluation": "evaluation/eval_ca1m.py -> box3d_iou_v2",
            "yaw_obb_iou_is_primary": False,
            "official_validation_comparable": False,
            "ground_truth_provenance": "derived from official CA-1M train tar instances",
        },
        "dataset": {"path": str(output), "sha256": dataset_sha, "rows": sample_count},
        "feature_names": list(feature_names),
        "frozen_subset_manifest": {
            "path": str(args.subset_manifest.resolve()), "sha256": subset_sha,
            "scene_count": len(scenes),
            "scene_ids_sha256": subset["selection"]["scene_ids_sha256"],
        },
        "train_collection": {
            "path": str(collection_path),
            "sha256": collection_sha,
            "schema": collection["schema"],
            "observer_completion_root": str(Path(args.observer_completion_root).resolve()),
            "evaluation_invoked": False,
        },
        "scene_list": {"path": str(args.scene_list.resolve()), "sha256": sha256_file(args.scene_list.resolve())},
        "validation_url_list": {"path": str(args.val_url_list.resolve()), "sha256": sha256_file(args.val_url_list.resolve()), "ids_only": True},
        "split": {
            "kind": "deterministic_scene_grouped_5fold",
            "namespace": args.split_namespace,
            "fold_count": 5,
            "dev_fold": 0,
            "train_folds": [1, 2, 3, 4],
            "scene_counts": {str(fold): int(sum(value == fold for value in folds.values())) for fold in range(5)},
        },
        "counts": {
            "scenes": len(scenes), "rows": sample_count,
            "valid_evidence_rows": int(np.count_nonzero(arrays["valid_evidence"])),
            "positive_iou15": int(np.count_nonzero(arrays["target_iou"] > 0.15)),
            "positive_iou25": int(np.count_nonzero(arrays["target_iou"] > 0.25)),
            "positive_iou50": int(np.count_nonzero(arrays["target_iou"] > 0.50)),
        },
        "scenes": scene_reports,
    }
    if collection["schema"] == FINAL_BASE_COLLECTION_SCHEMA:
        manifest["train_collection"].update(
            {
                "source_final_base_collection": collection[
                    "source_final_base_collection"
                ],
                "source_final_base_root": collection["source_final_base_root"],
                "fixed10_paired_report": collection["fixed10_paired_report"],
                "geometry_authority": "sealed_final_base_prediction",
                "offline_direct_observer": True,
                "cross_run_boxfusion_replay_invoked": False,
                "cross_run_exact_identity_required": False,
                "old_native_b6_diagnostics_reused": False,
                "old_native_b6_checkpoint_reused": False,
                "source_modules": collection["source_modules"],
            }
        )
    _create_only(manifest_output, _canonical_json(manifest))
    print(json.dumps({
        "dataset": str(output), "manifest": str(manifest_output),
        "scenes": len(scenes), "rows": sample_count,
        "dev_scenes": manifest["split"]["scene_counts"]["0"],
    }, indent=2, sort_keys=True))
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--observer-root", type=Path, required=True)
    result.add_argument("--prediction-root", type=Path, required=True)
    result.add_argument("--gt-root", type=Path, required=True)
    result.add_argument("--scene-list", type=Path, required=True)
    result.add_argument("--subset-manifest", type=Path, required=True)
    result.add_argument("--collection-manifest", type=Path, required=True)
    result.add_argument("--observer-completion-root", type=Path, required=True)
    result.add_argument("--val-url-list", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--manifest-output", type=Path, required=True)
    result.add_argument("--split-namespace", default=DEFAULT_SPLIT_NAMESPACE)
    return result


def main() -> int:
    build(parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
