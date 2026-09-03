#!/usr/bin/env python3
"""Seal offline CA-train native-B6 evidence for sealed final-base boxes.

The sealed G0+CLIP+reliable-TopK3 prediction is the only geometry/score
authority.  V2 never replays BoxFusion: it queries those exact rows directly
with train-scene depth/K/pose and binds the create-only observer diagnostic to
an offline receipt.  No evaluator or validation artifact is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import stat
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


SCENE_SCHEMA = "boxfusion.ca1m_native_b6_final_base_scene_completion.v2"
COLLECTION_SCHEMA = "boxfusion.ca1m_native_b6_final_base_train_collection.v2"
FINAL_BASE_SCHEMA = "boxfusion.ca1m_final_base_identity_audit.v1"
DIAGNOSTIC_SCHEMA = "boxfusion.ca1m_native_b6_observer.v1"
SUBSET_SCHEMA = "boxfusion.ca1m_native_b6_train_subset.v1"
OFFLINE_CONFIG_SCHEMA = "boxfusion.ca1m_native_b6_final_base_offline_config.v2"
OFFLINE_RECEIPT_SCHEMA = "boxfusion.ca1m_native_b6_final_base_offline_receipt.v2"
PAIRED_REPORT_SCHEMA = "boxfusion.ca1m_final_base_paired_eval.v1"
FEATURE_NAMES = (
    "detector_score",
    "support_given_depth",
    "occluded_given_depth",
    "free_given_depth",
    "invalid_ratio",
    "view_coverage",
    "sample_support",
    "area_quality",
    "area_stability",
    "support_view_mean",
    "support_view_min",
    "free_view_max",
    "aspect_balance",
    "height_balance",
)
SCENE = re.compile(r"^[0-9]{8}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
CONTRACT_SCHEMA = "boxfusion.ca1m_native_b6_final_base_offline_contract.v2"


def regular(path: Path, label: str) -> Path:
    raw = Path(path)
    try:
        mode = raw.lstat().st_mode
    except FileNotFoundError as error:
        raise ValueError(f"missing {label}: {raw}") from error
    if raw.is_symlink() or not stat.S_ISREG(mode) or raw.stat().st_size <= 0:
        raise ValueError(f"{label} must be a non-empty regular file: {raw}")
    return raw.resolve()


def directory(path: Path, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {raw}")
    resolved = raw.resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise ValueError(f"{label} must be a regular directory: {resolved}")
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with regular(path, "hashed artifact").open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_or_verify(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if target.exists() or target.is_symlink():
        existing = regular(target, "sealed v2 artifact")
        if existing.read_bytes() != encoded:
            raise ValueError(f"existing sealed v2 artifact drifted: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        target.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to replace sealed v2 artifact: {target}") from error
    finally:
        temporary.unlink(missing_ok=True)


def prediction(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    source = regular(path, "prediction")
    with source.open("rb") as handle:
        payload = pickle.load(handle)  # trusted local pipeline artifact
        if handle.read(1):
            raise ValueError(f"trailing bytes in prediction: {source}")
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], list):
        raise ValueError(f"invalid prediction container: {source}")
    corners: list[np.ndarray] = []
    scores: list[float] = []
    for index, row in enumerate(payload[0]):
        if not isinstance(row, tuple) or len(row) != 3 or type(row[0]) is not int or row[0] != 0:
            raise ValueError(f"invalid prediction row {index}: {source}")
        box = np.asarray(row[1], dtype=np.float32)
        score = float(row[2])
        if box.shape != (8, 3) or not np.isfinite(box).all() or not np.isfinite(score):
            raise ValueError(f"non-finite prediction row {index}: {source}")
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"prediction score outside [0,1] at row {index}: {source}")
        corners.append(box)
        scores.append(score)
    return (
        np.stack(corners) if corners else np.empty((0, 8, 3), dtype=np.float32),
        np.asarray(scores, dtype=np.float32),
        len(corners),
    )


def load_final_base_manifest(path: Path) -> tuple[dict[str, Any], Path]:
    source = regular(path, "final-base collection manifest")
    value = json.loads(source.read_text())
    required = {
        "schema": FINAL_BASE_SCHEMA,
        "ok": True,
        "dataset": "CA1M",
        "split": "train100",
        "ground_truth_access": False,
        "evaluation_invoked": False,
        "training_invoked": False,
        "scannet_learned_b6_or_gate_reused": False,
        "clip_appearance_gate_active": True,
        "reliable_view_top_k": 3,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(f"final-base collection field {key} disagrees")
    count = value.get("scene_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("final-base collection scene_count is invalid")
    same_run = value.get("same_run") or {}
    if any(
        same_run.get(name) != count
        for name in (
            "byte_identity_scenes",
            "semantic_identity_scenes",
            "hard_link_identity_scenes",
        )
    ):
        raise ValueError("final-base same-run identity coverage is incomplete")
    rows = value.get("per_scene")
    if not isinstance(rows, dict) or len(rows) != count:
        raise ValueError("final-base per_scene mapping is incomplete")
    for scene, row in rows.items():
        if SCENE.fullmatch(str(scene)) is None or not isinstance(row, dict):
            raise ValueError("final-base per_scene entry is invalid")
        digest = str(row.get("active_prediction_sha256", ""))
        if SHA256.fullmatch(digest) is None or row.get("byte_identity") is not True:
            raise ValueError(f"{scene}: final-base identity record is invalid")
    return value, source


def load_paired_report(path: Path) -> tuple[dict[str, Any], Path]:
    source = regular(path, "authoritative fixed10 paired report")
    value = json.loads(source.read_text())
    required = {
        "schema": PAIRED_REPORT_SCHEMA,
        "complete": True,
        "dataset": "CA1M",
        "split": "validation_fixed10",
        "scene_count": 10,
        "paired_official_evaluation": True,
        "positive_map_at_all_thresholds": True,
        "training_invoked": False,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(f"fixed10 paired report field {key} disagrees")
    decision = value.get("decision") or {}
    expected_decision = {
        "train100_final_base_collection_authorized": True,
        "ca1m_native_b6_retraining_required": True,
        "canonical_active_authorized": False,
    }
    for key, expected in expected_decision.items():
        if decision.get(key) != expected:
            raise ValueError(f"fixed10 paired decision field {key} disagrees")
    delta = value.get("delta") or {}
    active = value.get("active") or {}
    control = value.get("control") or {}
    for threshold in ("AP15", "AP25", "AP50"):
        try:
            gain = float(delta[threshold]["mAP"])
            active_map = float(active[threshold]["mAP"])
            control_map = float(control[threshold]["mAP"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"fixed10 paired report lacks {threshold} mAP values"
            ) from error
        if not all(np.isfinite(value) for value in (gain, active_map, control_map)):
            raise ValueError(f"fixed10 paired report {threshold} mAP is non-finite")
        if gain <= 0.0:
            raise ValueError(f"fixed10 paired report {threshold} mAP delta is not positive")
        if not np.isclose(active_map - control_map, gain, rtol=0, atol=5e-7):
            raise ValueError(f"fixed10 paired report {threshold} mAP delta is inconsistent")
    return value, source


def paired_report_record(path: Path) -> dict[str, Any]:
    _, source = load_paired_report(path)
    return {
        "path": str(source),
        "sha256": sha256(source),
        "schema": PAIRED_REPORT_SCHEMA,
        "role": "authoritative_fixed10_train100_and_retraining_gate",
    }


def audit_contract(
    final_base_config: Path, offline_config: Path, paired_report: Path
) -> dict[str, Any]:
    final_path = regular(final_base_config, "final-base train100 config")
    offline_path = regular(offline_config, "native-B6 v2 offline config")
    final_cfg = yaml.safe_load(final_path.read_text())
    offline_cfg = yaml.safe_load(offline_path.read_text())
    if not isinstance(final_cfg, dict) or not isinstance(offline_cfg, dict):
        raise ValueError("v2 contract configurations must contain mappings")
    if final_cfg.get("ca1m_native_b6_observer") != {"enabled": False}:
        raise ValueError("source final-base config must not run native B6")
    if final_cfg.get("online_refinement") != {"enabled": False}:
        raise ValueError("source final-base config must not run online refinement")
    reliable = (final_cfg.get("box_fusion") or {}).get("reliable_views") or {}
    appearance = (final_cfg.get("association") or {}).get("appearance_gate") or {}
    selective = ((final_cfg.get("lifting") or {}).get("boxer") or {}).get(
        "selective_gate"
    ) or {}
    if reliable.get("enabled") is not True or reliable.get("top_k") != 3:
        raise ValueError("source final-base config requires reliable-view Top-K=3")
    if appearance.get("enabled") is not True or selective.get("enabled") is not True:
        raise ValueError("source final-base config lacks G0+CLIP modules")
    expected_offline = {
        "schema": OFFLINE_CONFIG_SCHEMA,
        "dataset": "CA1M",
        "data": {
            "gap": 20,
            "start": 0,
            "depth_scale": 1000.0,
            "image_height": 384,
            "image_width": 512,
        },
        "source_anchor": {
            "split": "train100",
            "geometry_authority": "sealed_final_base_prediction",
            "required_modules": {
                "selective_boxer_g0": True,
                "clip_appearance_gate": True,
                "reliable_view_top_k": 3,
            },
            "cross_run_replay_required": False,
            "cross_run_exact_identity_required": False,
        },
        "observer": {
            "top_k_views": 5,
            "pixel_stride": 4,
            "depth_margin_m": 0.05,
            "min_depth_m": 0.10,
            "max_depth_m": 8.0,
            "near_clip_m": 0.001,
            "max_cached_keyframes": 256,
            "stable_id_policy": "sealed_prediction_row_index",
        },
        "safety": {
            "train_only": True,
            "prediction_mutation_authorized": False,
            "ground_truth_access": False,
            "validation_ground_truth_access": False,
            "validation_prediction_access": False,
            "evaluator_invoked": False,
            "rgb_pixels_accessed": False,
            "old_native_b6_diagnostics_reused": False,
            "old_native_b6_checkpoint_reused": False,
        },
    }
    if offline_cfg != expected_offline:
        raise ValueError("v2 offline native-B6 protocol disagrees")
    checkpoint_paths: list[str] = []

    def walk(value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_path = path + (str(key),)
                if "checkpoint" in str(key).lower() and isinstance(child, str):
                    checkpoint_paths.append(".".join(child_path))
                walk(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, path + (str(index),))

    walk(offline_cfg)
    if checkpoint_paths:
        raise ValueError("v2 offline config must not contain checkpoint fields")
    return {
        "schema": CONTRACT_SCHEMA,
        "ok": True,
        "dataset": "CA1M",
        "train_only": True,
        "geometry_authority": "sealed_final_base_prediction",
        "offline_direct_observer": True,
        "cross_run_boxfusion_replay_invoked": False,
        "cross_run_exact_identity_required": False,
        "ground_truth_access": False,
        "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "evaluation_invoked": False,
        "training_invoked": False,
        "rgb_pixels_accessed": False,
        "prediction_mutation_authorized": False,
        "old_native_b6_diagnostics_reused": False,
        "old_native_b6_checkpoint_reused": False,
        "source_modules": {
            "selective_boxer_g0": True,
            "clip_appearance_gate": True,
            "reliable_view_top_k": 3,
            "b6_evidence_top_k": 5,
        },
        "configs": {
            "final_base_train100": {
                "path": str(final_path),
                "sha256": sha256(final_path),
            },
            "offline_v2": {
                "path": str(offline_path),
                "sha256": sha256(offline_path),
            },
        },
        "fixed10_paired_report": paired_report_record(paired_report),
    }


def _validate_final_anchor(
    manifest: Mapping[str, Any], root: Path, scene: str
) -> tuple[Path, str]:
    source_root = directory(root, "final-base prediction root")
    anchor = regular(source_root / f"{scene}_boxes.pkl", "final-base prediction")
    digest = sha256(anchor)
    row = (manifest.get("per_scene") or {}).get(scene)
    if not isinstance(row, Mapping) or row.get("active_prediction_sha256") != digest:
        raise ValueError(f"{scene}: final-base prediction differs from its sealed manifest")
    return anchor, digest


def audit_source(
    scene_list: Path,
    expected_scenes: int,
    final_base_root: Path,
    final_base_manifest: Path,
    paired_report: Path,
) -> dict[str, Any]:
    source_list = regular(scene_list, "frozen train scene list")
    scenes = [line.strip() for line in source_list.read_text().splitlines() if line.strip()]
    if (
        len(scenes) != expected_scenes
        or len(set(scenes)) != expected_scenes
        or any(SCENE.fullmatch(scene) is None for scene in scenes)
    ):
        raise ValueError(f"frozen train scene list must contain exactly {expected_scenes} scenes")
    manifest, manifest_path = load_final_base_manifest(final_base_manifest)
    if manifest.get("scene_count") != expected_scenes or set(manifest["per_scene"]) != set(scenes):
        raise ValueError("final-base manifest is not the exact frozen scene list")
    root = directory(final_base_root, "final-base prediction root")
    actual = {
        path.name for path in root.iterdir()
        if path.is_file() and path.name.endswith("_boxes.pkl")
    }
    expected = {f"{scene}_boxes.pkl" for scene in scenes}
    if actual != expected:
        raise ValueError("final-base prediction root is not the exact frozen scene list")
    row_count = 0
    digest = hashlib.sha256()
    for scene in scenes:
        anchor, anchor_sha = _validate_final_anchor(manifest, root, scene)
        _, _, rows = prediction(anchor)
        row_count += rows
        digest.update(f"{scene}\t{anchor_sha}\n".encode())
    return {
        "schema": "boxfusion.ca1m_native_b6_final_base_source_audit.v2",
        "ok": True,
        "dataset": "CA1M",
        "split": "train100",
        "scene_count": expected_scenes,
        "prediction_rows": row_count,
        "scene_list_sha256": sha256(source_list),
        "final_base_manifest": {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
            "schema": FINAL_BASE_SCHEMA,
        },
        "final_base_root": str(root),
        "prediction_collection_sha256": digest.hexdigest(),
        "fixed10_paired_report": paired_report_record(paired_report),
        "clip_appearance_gate_active": True,
        "reliable_view_top_k": 3,
        "ground_truth_access": False,
        "evaluation_invoked": False,
        "training_invoked": False,
    }


def _diagnostic(
    path: Path, scene: str, corners: np.ndarray, scores: np.ndarray
) -> tuple[int, list[int]]:
    source = regular(path, "native-B6 v2 diagnostic")
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "schema", "complete", "observer_only", "mutation_enabled",
            "applied_count", "ground_truth_access", "clip_access", "scene_id",
            "result_indices", "corners", "scores", "feature_names", "features",
            "stable_ids", "used_frame_ids", "topk_frame_ids", "valid_evidence",
        }
        if not required.issubset(archive.files):
            raise ValueError(f"{scene}: native-B6 diagnostic fields are incomplete")
        values = {name: np.array(archive[name], copy=True) for name in required}
    scalars = {
        "schema": DIAGNOSTIC_SCHEMA,
        "complete": True,
        "observer_only": True,
        "mutation_enabled": False,
        "applied_count": 0,
        "ground_truth_access": False,
        "clip_access": False,
        "scene_id": scene,
    }
    for name, expected in scalars.items():
        if values[name].shape != () or values[name].item() != expected:
            raise ValueError(f"{scene}: native-B6 diagnostic scalar {name} disagrees")
    count = len(scores)
    if not np.array_equal(values["result_indices"], np.arange(count, dtype=np.int64)):
        raise ValueError(f"{scene}: native-B6 result mapping is not identity")
    if not np.array_equal(values["stable_ids"], np.arange(count, dtype=np.int64)):
        raise ValueError(f"{scene}: native-B6 stable IDs are not sealed row indices")
    if not np.array_equal(values["corners"], corners) or not np.array_equal(values["scores"], scores):
        raise ValueError(f"{scene}: native-B6 evidence differs from the final-base rows")
    names = tuple(str(item) for item in values["feature_names"].tolist())
    features = np.asarray(values["features"], dtype=np.float32)
    if names != FEATURE_NAMES or features.shape != (count, len(FEATURE_NAMES)):
        raise ValueError(f"{scene}: native-B6 feature schema is not the frozen 14-D contract")
    if not np.isfinite(features).all() or np.any(features < 0.0) or np.any(features > 1.0):
        raise ValueError(f"{scene}: native-B6 features must be finite in [0,1]")
    if not np.array_equal(features[:, 0], scores):
        raise ValueError(f"{scene}: detector-score feature differs from final-base score")
    topk = np.asarray(values["topk_frame_ids"])
    used = np.asarray(values["used_frame_ids"])
    valid_evidence = np.asarray(values["valid_evidence"])
    if (
        topk.shape != (count, 5)
        or used.ndim != 1
        or valid_evidence.shape != (count,)
        or valid_evidence.dtype.kind != "b"
    ):
        raise ValueError(f"{scene}: B6 observer Top-K=5 evidence contract disagrees")
    if used.dtype.kind not in "iu":
        raise ValueError(f"{scene}: used frame IDs must be integers")
    used_ids = [int(value) for value in used.tolist()]
    if (
        any(value < 0 for value in used_ids)
        or used_ids != sorted(set(used_ids))
    ):
        raise ValueError(f"{scene}: used frame IDs must be unique and increasing")
    valid_topk = topk[topk >= 0]
    if len(valid_topk) and not set(valid_topk.tolist()).issubset(set(used_ids)):
        raise ValueError(f"{scene}: Top-K evidence refers to an unrecorded frame")
    return int(np.count_nonzero(valid_evidence)), used_ids


def _receipt(
    path: Path,
    scene: str,
    *,
    final_anchor: Path,
    final_sha: str,
    final_manifest_path: Path,
    diagnostic: Path,
    used_frame_ids: Sequence[int],
    prediction_rows: int,
) -> tuple[dict[str, Any], Path]:
    source = regular(path, "offline native-B6 receipt")
    value = json.loads(source.read_text())
    required = {
        "schema": OFFLINE_RECEIPT_SCHEMA,
        "complete": True,
        "mode": "run",
        "scene_id": scene,
        "train_only": True,
        "prediction_mutation_authorized": False,
        "ground_truth_access": False,
        "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "evaluator_invoked": False,
        "rgb_pixels_accessed": False,
        "cross_run_boxfusion_replay_invoked": False,
        "cross_run_exact_identity_required": False,
        "old_native_b6_diagnostics_reused": False,
        "old_native_b6_checkpoint_reused": False,
        "geometry_authority": "sealed_final_base_prediction",
        "stable_id_policy": "sealed_prediction_row_index",
        "prediction_rows": prediction_rows,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(f"{scene}: offline receipt field {key} disagrees")
    if value.get("source_modules") != {
        "selective_boxer_g0": True,
        "clip_appearance_gate": True,
        "reliable_view_top_k": 3,
        "b6_evidence_top_k": 5,
    }:
        raise ValueError(f"{scene}: offline receipt module binding disagrees")
    anchor = value.get("source_final_base") or {}
    if (
        Path(str(anchor.get("path", ""))).resolve() != final_anchor
        or anchor.get("sha256") != final_sha
        or Path(str(anchor.get("manifest_path", ""))).resolve()
        != final_manifest_path
        or anchor.get("manifest_sha256") != sha256(final_manifest_path)
        or anchor.get("manifest_schema") != FINAL_BASE_SCHEMA
    ):
        raise ValueError(f"{scene}: offline receipt lost final-base binding")
    diagnostic_record = value.get("diagnostic") or {}
    if (
        Path(str(diagnostic_record.get("path", ""))).resolve() != diagnostic
        or diagnostic_record.get("sha256") != sha256(diagnostic)
        or diagnostic_record.get("schema") != DIAGNOSTIC_SCHEMA
        or diagnostic_record.get("feature_names") != list(FEATURE_NAMES)
    ):
        raise ValueError(f"{scene}: offline receipt diagnostic binding disagrees")
    recovery = value.get("diagnostic_recovery") or {}
    recovered = recovery.get("preexisting_orphan")
    if recovered not in (True, False) or recovery.get(
        "semantic_recomputation_exact"
    ) is not recovered:
        raise ValueError(f"{scene}: offline diagnostic recovery record disagrees")
    expected_ignored = "summary_json.observer_seconds" if recovered else None
    if recovery.get("runtime_only_field_ignored") != expected_ignored:
        raise ValueError(f"{scene}: offline diagnostic recovery exception disagrees")
    protocol = value.get("frame_protocol") or {}
    if (
        protocol.get("gap") != 20
        or protocol.get("lineage")
        != "demo.py record-before-increment then early-finalize v1"
        or protocol.get("physical_terminal_frame_policy") != "not_forced"
        or protocol.get("used_frame_ids") != list(used_frame_ids)
    ):
        raise ValueError(f"{scene}: offline receipt frame lineage disagrees")
    selected = ((value.get("input_files") or {}).get("selected_depth"))
    if not isinstance(selected, list) or [row.get("frame_id") for row in selected] != list(
        used_frame_ids
    ):
        raise ValueError(f"{scene}: selected depth lineage disagrees")
    for row in selected:
        depth = regular(Path(str(row.get("depth_path", ""))), "selected train depth")
        if row.get("depth_sha256") != sha256(depth):
            raise ValueError(f"{scene}: selected train depth binding disagrees")
        for key in (
            "oriented_depth_array_sha256",
            "oriented_intrinsics_array_sha256",
            "camera_to_world_array_sha256",
        ):
            if SHA256.fullmatch(str(row.get(key, ""))) is None:
                raise ValueError(f"{scene}: offline array binding {key} is invalid")
    inputs = value.get("input_files") or {}
    if inputs.get("rgb_pixels_accessed") is not False:
        raise ValueError(f"{scene}: offline receipt claims RGB pixel access")
    for key in ("all_poses", "scene_intrinsics", "per_frame_intrinsics"):
        record = inputs.get(key) or {}
        input_path = regular(Path(str(record.get("path", ""))), f"offline {key}")
        if record.get("sha256") != sha256(input_path):
            raise ValueError(f"{scene}: offline input binding {key} disagrees")
    receipt_recorded = Path(str(value.get("receipt_path", ""))).resolve()
    if receipt_recorded != source:
        raise ValueError(f"{scene}: offline receipt self path disagrees")
    return value, source


def scene_completion(args: argparse.Namespace) -> dict[str, Any]:
    scene = str(args.scene)
    if SCENE.fullmatch(scene) is None:
        raise ValueError(f"invalid CA-1M scene id: {scene!r}")
    final_manifest, final_manifest_path = load_final_base_manifest(args.final_base_manifest)
    final_anchor, final_sha = _validate_final_anchor(
        final_manifest, args.final_base_root, scene
    )
    final_corners, final_scores, rows = prediction(final_anchor)
    diagnostic = regular(args.diagnostic, "native-B6 v2 diagnostic")
    valid_rows, used_ids = _diagnostic(
        diagnostic, scene, final_corners, final_scores
    )
    _, receipt = _receipt(
        args.offline_receipt,
        scene,
        final_anchor=final_anchor,
        final_sha=final_sha,
        final_manifest_path=final_manifest_path,
        diagnostic=diagnostic,
        used_frame_ids=used_ids,
        prediction_rows=rows,
    )
    return {
        "schema": SCENE_SCHEMA,
        "phase": "sealed_final_base_offline_native_b6_observer",
        "scene_id": scene,
        "complete": True,
        "train_only": True,
        "evaluation_invoked": False,
        "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "output_mutation_authorized": False,
        "geometry_authority": "sealed_final_base_prediction",
        "offline_direct_observer": True,
        "cross_run_boxfusion_replay_invoked": False,
        "cross_run_exact_identity_required": False,
        "rgb_pixels_accessed": False,
        "old_native_b6_diagnostics_reused": False,
        "old_native_b6_checkpoint_reused": False,
        "source_modules": {
            "selective_boxer_g0": True,
            "clip_appearance_gate": True,
            "reliable_view_top_k": 3,
            "b6_evidence_top_k": 5,
        },
        "prediction_rows": rows,
        "valid_evidence_rows": valid_rows,
        "used_frame_ids": used_ids,
        "stable_id_policy": "sealed_prediction_row_index",
        "source_final_base_manifest": {
            "path": str(final_manifest_path),
            "sha256": sha256(final_manifest_path),
            "schema": FINAL_BASE_SCHEMA,
        },
        "artifacts": {
            name: {"path": str(regular(path, name)), "sha256": sha256(path)}
            for name, path in {
                # Dataset construction consumes the sealed source directly.
                "prediction": final_anchor,
                "final_base_anchor": final_anchor,
                "native_b6_diagnostic": diagnostic,
                "offline_receipt": receipt,
            }.items()
        },
    }


def _subset(path: Path, expected_scenes: int) -> tuple[dict[str, Any], Path, list[str]]:
    source = regular(path, "frozen CA train subset manifest")
    value = json.loads(source.read_text())
    if value.get("schema") != SUBSET_SCHEMA:
        raise ValueError("unsupported frozen CA train subset schema")
    rows = value.get("entries")
    scenes = [str(row.get("scene_id")) for row in rows] if isinstance(rows, list) else []
    if (
        len(scenes) != expected_scenes
        or len(set(scenes)) != expected_scenes
        or any(SCENE.fullmatch(scene) is None for scene in scenes)
    ):
        raise ValueError(f"frozen CA train subset must contain exactly {expected_scenes} scenes")
    safety = value.get("safety_contract") or {}
    if (
        safety.get("train_only") is not True
        or safety.get("validation_ground_truth_access") is not False
        or int(safety.get("validation_scene_overlap_count", -1)) != 0
    ):
        raise ValueError("frozen CA train subset safety contract disagrees")
    return value, source, scenes


def collection(args: argparse.Namespace) -> dict[str, Any]:
    subset, subset_path, scenes = _subset(args.subset_manifest, args.expected_scenes)
    final_manifest, final_manifest_path = load_final_base_manifest(args.final_base_manifest)
    if final_manifest.get("scene_count") != len(scenes) or set(final_manifest["per_scene"]) != set(scenes):
        raise ValueError("final-base manifest is not the exact frozen train subset")
    final_root = directory(args.final_base_root, "final-base prediction root")
    actual_final = {
        path.name for path in final_root.iterdir()
        if path.is_file() and path.name.endswith("_boxes.pkl")
    }
    expected_final = {f"{scene}_boxes.pkl" for scene in scenes}
    if actual_final != expected_final:
        raise ValueError("final-base prediction root is not the exact frozen train subset")
    completion_root = directory(args.completion_root, "v2 completion root")
    actual_completions = {path.name for path in completion_root.iterdir() if path.is_file()}
    expected_completions = {f"{scene}.json" for scene in scenes}
    if actual_completions != expected_completions:
        raise ValueError("v2 completion root is not the exact frozen train subset")
    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        completion_path = regular(completion_root / f"{scene}.json", "v2 scene completion")
        completion_value = json.loads(completion_path.read_text())
        required = {
            "schema": SCENE_SCHEMA,
            "phase": "sealed_final_base_offline_native_b6_observer",
            "scene_id": scene,
            "complete": True,
            "train_only": True,
            "evaluation_invoked": False,
            "validation_ground_truth_access": False,
            "validation_prediction_access": False,
            "output_mutation_authorized": False,
            "geometry_authority": "sealed_final_base_prediction",
            "offline_direct_observer": True,
            "cross_run_boxfusion_replay_invoked": False,
            "cross_run_exact_identity_required": False,
            "rgb_pixels_accessed": False,
            "old_native_b6_diagnostics_reused": False,
            "old_native_b6_checkpoint_reused": False,
        }
        for key, expected in required.items():
            if completion_value.get(key) != expected:
                raise ValueError(f"{scene}: v2 scene completion field {key} disagrees")
        if completion_value.get("source_modules") != {
            "selective_boxer_g0": True,
            "clip_appearance_gate": True,
            "reliable_view_top_k": 3,
            "b6_evidence_top_k": 5,
        }:
            raise ValueError(f"{scene}: v2 scene completion module binding disagrees")
        final_anchor, final_sha = _validate_final_anchor(final_manifest, final_root, scene)
        artifact = (completion_value.get("artifacts") or {}).get("final_base_anchor") or {}
        if (
            Path(str(artifact.get("path", ""))).resolve() != final_anchor
            or artifact.get("sha256") != final_sha
        ):
            raise ValueError(f"{scene}: v2 completion lost its final-base binding")
        prediction_artifact = (completion_value.get("artifacts") or {}).get(
            "prediction"
        ) or {}
        if prediction_artifact != artifact:
            raise ValueError(f"{scene}: dataset prediction is not the sealed final base")
        for name in ("native_b6_diagnostic", "offline_receipt"):
            record = (completion_value.get("artifacts") or {}).get(name) or {}
            artifact_path = regular(
                Path(str(record.get("path", ""))), f"{scene} completion {name}"
            )
            if record.get("sha256") != sha256(artifact_path):
                raise ValueError(f"{scene}: v2 completion artifact {name} drifted")
        completion_sha = sha256(completion_path)
        digest.update(f"{scene}\t{completion_sha}\t{final_sha}\n".encode())
        rows.append(
            {
                "scene_id": scene,
                "observer_completion_sha256": completion_sha,
                "final_base_prediction_sha256": final_sha,
            }
        )
    scene_ids_sha = str((subset.get("selection") or {}).get("scene_ids_sha256", ""))
    expected_scene_ids_sha = hashlib.sha256(
        ("\n".join(scenes) + "\n").encode("ascii")
    ).hexdigest()
    if scene_ids_sha != expected_scene_ids_sha:
        raise ValueError("frozen CA train subset scene-list SHA256 disagrees")
    return {
        "schema": COLLECTION_SCHEMA,
        "complete": True,
        "train_only": True,
        "evaluation_invoked": False,
        "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "official_validation_comparable": False,
        "geometry_authority": "sealed_final_base_prediction",
        "offline_direct_observer": True,
        "cross_run_boxfusion_replay_invoked": False,
        "cross_run_exact_identity_required": False,
        "rgb_pixels_accessed": False,
        "old_native_b6_diagnostics_reused": False,
        "old_native_b6_checkpoint_reused": False,
        "source_modules": {
            "selective_boxer_g0": True,
            "clip_appearance_gate": True,
            "reliable_view_top_k": 3,
            "b6_evidence_top_k": 5,
        },
        "scene_count": len(scenes),
        "scene_ids_sha256": scene_ids_sha,
        "subset_manifest_sha256": sha256(subset_path),
        "source_final_base_collection": {
            "path": str(final_manifest_path),
            "sha256": sha256(final_manifest_path),
            "schema": FINAL_BASE_SCHEMA,
        },
        "source_final_base_root": str(final_root),
        "fixed10_paired_report": paired_report_record(args.paired_report),
        "completion_collection_sha256": digest.hexdigest(),
        "split_protocol": {
            "kind": "deterministic_scene_grouped_5fold",
            "namespace": "boxfusion.ca1m-native-b6.scene-folds.v1",
            "deployable_training_folds": [1, 2, 3, 4],
            "untouched_dev_fold": 0,
        },
        "scenes": rows,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    zero = sub.add_parser("contract")
    zero.add_argument("--final-base-config", type=Path, required=True)
    zero.add_argument("--offline-config", type=Path, required=True)
    zero.add_argument("--paired-report", type=Path, required=True)
    zero.add_argument("--output", type=Path)
    source = sub.add_parser("source")
    source.add_argument("--scene-list", type=Path, required=True)
    source.add_argument("--expected-scenes", type=int, default=100)
    source.add_argument("--final-base-root", type=Path, required=True)
    source.add_argument("--final-base-manifest", type=Path, required=True)
    source.add_argument("--paired-report", type=Path, required=True)
    source.add_argument("--output", type=Path)
    one = sub.add_parser("scene")
    one.add_argument("--scene", required=True)
    one.add_argument("--final-base-root", type=Path, required=True)
    one.add_argument("--final-base-manifest", type=Path, required=True)
    one.add_argument("--diagnostic", type=Path, required=True)
    one.add_argument("--offline-receipt", type=Path, required=True)
    one.add_argument("--output", type=Path, required=True)
    two = sub.add_parser("collection")
    two.add_argument("--subset-manifest", type=Path, required=True)
    two.add_argument("--expected-scenes", type=int, default=100)
    two.add_argument("--completion-root", type=Path, required=True)
    two.add_argument("--final-base-root", type=Path, required=True)
    two.add_argument("--final-base-manifest", type=Path, required=True)
    two.add_argument("--paired-report", type=Path, required=True)
    two.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.command == "contract":
        value = audit_contract(
            args.final_base_config, args.offline_config, args.paired_report
        )
    elif args.command == "source":
        if args.expected_scenes < 1:
            raise ValueError("--expected-scenes must be positive")
        value = audit_source(
            args.scene_list,
            args.expected_scenes,
            args.final_base_root,
            args.final_base_manifest,
            args.paired_report,
        )
    elif args.command == "scene":
        value = scene_completion(args)
    else:
        if args.expected_scenes < 1:
            raise ValueError("--expected-scenes must be positive")
        value = collection(args)
    if args.output is not None:
        create_or_verify(args.output, value)
    print(
        json.dumps(
            {
                "schema": value["schema"],
                "ok": value.get("ok", value.get("complete")),
                "complete": value.get("complete"),
                "scene_id": value.get("scene_id"),
                "scene_count": value.get("scene_count"),
                "output": (
                    str(Path(args.output).resolve())
                    if args.output is not None
                    else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
