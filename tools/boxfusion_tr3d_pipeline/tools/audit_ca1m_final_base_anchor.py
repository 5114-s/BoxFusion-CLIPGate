#!/usr/bin/env python3
"""Fail-closed contract and identity audit for the CA-1M final base anchor."""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_CONTRACT = "boxfusion.ca1m_final_base_contract.v1"
SCHEMA_IDENTITY = "boxfusion.ca1m_final_base_identity_audit.v1"
BOXER_SHA = "d5a30b348a8f5b0e5990ff3aa0e8f473ce77d860da22586322e7f47abc83ca6f"
DINO_SHA = "4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea"
APPEARANCE = {
    "enabled": True,
    "geometry_min_iou": 0.08,
    "hard_geometry_iou": 0.45,
    "low_similarity": 0.45,
    "high_similarity": 0.75,
    "max_iou_penalty": 0.10,
    "max_iou_bonus": 0.0,
    "confidence_floor": 0.35,
    "confidence_full": 0.75,
    "spatial": {"low_similarity": 0.65, "high_similarity": 0.85},
    "correspondence": {"low_similarity": 0.55, "high_similarity": 0.75},
}
RELIABLE = {
    "enabled": True,
    "top_k": 3,
    "min_views": 3,
    "confidence_power": 1.0,
    "area_power": 0.25,
    "area_reference_ratio": 0.02,
    "projection_iou_power": 0.50,
    "geometry_consistency_power": 0.50,
    "center_sigma": 0.75,
    "size_sigma": 0.50,
    "minimum_box_diagonal": 0.10,
    "minimum_weight": 0.05,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def regular(path: Path, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {raw}")
    resolved = raw.resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ValueError(f"{label} must be a non-empty regular file: {resolved}")
    return resolved


def directory(path: Path, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {raw}")
    resolved = raw.resolve()
    if not resolved.is_dir():
        raise ValueError(f"{label} must be a regular directory: {resolved}")
    return resolved


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(regular(path, "configuration").read_text())
    if not isinstance(value, dict):
        raise ValueError(f"configuration must contain one mapping: {path}")
    return value


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} disagrees: {actual!r} != {expected!r}")


def _algorithm_view(cfg: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(cfg))
    value["data"].pop("datadir", None)
    value["data"].pop("output_dir", None)
    value["lifting"]["proposal_cache"] = {"source": "immutable_cutr_replay"}
    value["lifting"]["boxer"].pop("diagnostics_dir", None)
    return value


def _base_contract(cfg: Mapping[str, Any], label: str) -> None:
    _assert_equal(str(cfg.get("dataset", "")).lower(), "ca1m", f"{label}.dataset")
    _assert_equal(cfg.get("experiment", {}).get("seed"), 0, f"{label}.seed")
    _assert_equal(
        cfg.get("cam"),
        {"H": 384, "W": 512, "png_depth_scale": 1000.0},
        f"{label}.cam",
    )
    detection = cfg.get("detection", {})
    _assert_equal(detection.get("score_thresh"), 0.4, f"{label}.score_thresh")
    _assert_equal(cfg.get("data", {}).get("gap"), 20, f"{label}.gap")
    lifting = cfg.get("lifting", {})
    _assert_equal(lifting.get("backend"), "boxer", f"{label}.lifting.backend")
    cache = lifting.get("proposal_cache", {})
    _assert_equal(cache.get("mode"), "replay", f"{label}.cache.mode")
    boxer = lifting.get("boxer", {})
    _assert_equal(boxer.get("mode"), "active", f"{label}.boxer.mode")
    _assert_equal(boxer.get("apply_stage"), "post_filter", f"{label}.boxer.stage")
    _assert_equal(boxer.get("checkpoint_sha256"), BOXER_SHA, f"{label}.boxer.sha")
    _assert_equal(boxer.get("dinov3_sha256"), DINO_SHA, f"{label}.dino.sha")
    _assert_equal(
        boxer.get("selective_gate"),
        {
            "enabled": True,
            "max_center_shift_m": 0.10,
            "min_volume_ratio": 0.50,
            "max_volume_ratio": 2.00,
        },
        f"{label}.selective_boxer_g0",
    )
    association = cfg.get("association", {})
    for key, expected in (
        ("small_threshold", 0.2),
        ("rotation_gap", 30),
        ("translation_gap", 0.8),
    ):
        _assert_equal(association.get(key), expected, f"{label}.association.{key}")
    fusion = cfg.get("box_fusion", {})
    for key, expected in (
        ("use", True),
        ("iters", 20),
        ("nms_threshold", 0.1),
        ("small_size", 0.5),
    ):
        _assert_equal(fusion.get(key), expected, f"{label}.box_fusion.{key}")
    _assert_equal(cfg.get("online_refinement"), {"enabled": False}, f"{label}.online")
    _assert_equal(
        cfg.get("ca1m_native_b6_observer"),
        {"enabled": False},
        f"{label}.native_b6",
    )
    if not cfg.get("eval", False):
        raise ValueError(f"{label} must enable serialization")

    checkpoint_paths: list[str] = []
    def walk(value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_path = path + (str(key),)
                if str(key).lower() == "checkpoint" and isinstance(child, str):
                    checkpoint_paths.append(".".join(child_path))
                walk(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, path + (str(index),))
    walk(cfg)
    _assert_equal(checkpoint_paths, ["lifting.boxer.checkpoint"], f"{label}.checkpoints")


def _function_names(source: Path, functions: Iterable[str]) -> set[str]:
    tree = ast.parse(regular(source, "module source").read_text())
    wanted = set(functions)
    found: set[str] = set()
    identifiers: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
            found.add(node.name)
            identifiers.update(
                child.id.lower()
                for child in ast.walk(node)
                if isinstance(child, ast.Name)
            )
            identifiers.update(
                child.attr.lower()
                for child in ast.walk(node)
                if isinstance(child, ast.Attribute)
            )
    if found != wanted:
        raise ValueError(f"module functions missing: {sorted(wanted - found)}")
    return identifiers


def audit_contract(control_path: Path, active_path: Path, train_path: Path) -> dict[str, Any]:
    control = load_yaml(control_path)
    active = load_yaml(active_path)
    train = load_yaml(train_path)
    for cfg, label in ((control, "control"), (active, "active"), (train, "train100")):
        _base_contract(cfg, label)

    if "appearance_gate" in control["association"]:
        raise ValueError("G0 control must not enable the appearance gate")
    if "reliable_views" in control["box_fusion"]:
        raise ValueError("G0 control must not enable reliable-view fusion")
    _assert_equal(active["association"].get("appearance_gate"), APPEARANCE, "active.appearance")
    _assert_equal(active["box_fusion"].get("reliable_views"), RELIABLE, "active.reliable")
    _assert_equal(train["association"].get("appearance_gate"), APPEARANCE, "train.appearance")
    _assert_equal(train["box_fusion"].get("reliable_views"), RELIABLE, "train.reliable")

    normalized_active = deepcopy(active)
    normalized_active["association"].pop("appearance_gate")
    normalized_active["box_fusion"].pop("reliable_views")
    if _algorithm_view(normalized_active) != _algorithm_view(control):
        raise ValueError("fixed10 active changes fields beyond CLIP/Top-K and artifact roots")
    if _algorithm_view(active) != _algorithm_view(train):
        raise ValueError("train100 algorithm differs from the fixed10 final base")

    _assert_equal(
        active["lifting"]["proposal_cache"]["namespace"],
        "ca1m-score04-gap20-c0-v2",
        "fixed10 cache namespace",
    )
    _assert_equal(
        train["lifting"]["proposal_cache"]["namespace"],
        "ca1m-native-b6-train100-score04-gap20-cutr-v1",
        "train100 source cache namespace",
    )
    if "ca1m_c4_final_base_g0_clip_topk3_fixed10_v1" not in active["data"]["output_dir"]:
        raise ValueError("fixed10 final-base artifact namespace is not isolated")
    if "ca1m_native_final_base_train100_v1" not in train["data"]["output_dir"]:
        raise ValueError("train100 final-base artifact namespace is not isolated")

    appearance_names = _function_names(
        ROOT / "boxfusion" / "instances.py",
        ("cosine_similarity_to_many", "resolve_appearance_gate_config", "appearance_gate_decisions"),
    )
    reliable_names = _function_names(
        ROOT / "boxfusion" / "reliable_views.py",
        (
            "valid_reliable_view_mask",
            "resolve_reliable_view_config",
            "select_top_k_reliable_views",
            "weighted_box_initialization",
        ),
    )
    forbidden = {"gt", "ground_truth", "optimizer", "backward", "loss", "target_labels"}
    if appearance_names & forbidden or reliable_names & forbidden:
        raise ValueError("frozen runtime modules reference forbidden GT/training identifiers")

    return {
        "schema": SCHEMA_CONTRACT,
        "ok": True,
        "dataset": "CA1M",
        "ca_geometry_contract": {
            "box": "world_obb_center_size_plus_rotation_3x3",
            "association": "world_obb_corners_8x3",
            "projection": "per_frame_projected_corners_8x2",
            "pose": "camera_to_world_4x4",
            "axis_alignment_required": False,
            "image_size": [384, 512],
            "depth_scale": 1000.0,
        },
        "modules": {
            "clip_appearance_gate": {
                "enabled": True,
                "training_required": False,
                "trainable_parameters": False,
                "ground_truth_access": False,
                "encoder": "frozen OpenCLIP runtime asset",
            },
            "reliable_view_topk3": {
                "enabled": True,
                "top_k": 3,
                "training_required": False,
                "trainable_parameters": False,
                "ground_truth_access": False,
                "numpy_deterministic_policy": True,
            },
            "selective_boxer_g0": {
                "enabled": True,
                "generic_frozen_checkpoint": True,
                "ca_geometry_gate": {"center_m": 0.10, "volume_ratio": [0.50, 2.00]},
            },
        },
        "learned_dataset_specific_assets": [],
        "scannet_learned_b6_or_gate_reused": False,
        "ground_truth_access": False,
        "training_invoked": False,
        "downstream_contract": {
            "native_b6_recollection_required": True,
            "native_b6_retraining_required": True,
            "old_ca_b6_activation_authorized": False,
        },
        "configs": {
            "control": {"path": str(control_path.resolve()), "sha256": sha256(control_path)},
            "active": {"path": str(active_path.resolve()), "sha256": sha256(active_path)},
            "train100": {"path": str(train_path.resolve()), "sha256": sha256(train_path)},
        },
    }


def read_scenes(path: Path, expected: int) -> list[str]:
    source = regular(path, "scene list")
    scenes = [line.strip() for line in source.read_text().splitlines() if line.strip()]
    if len(scenes) != expected or len(set(scenes)) != expected:
        raise ValueError(f"scene list must contain exactly {expected} unique IDs")
    if any(not scene.isdigit() for scene in scenes):
        raise ValueError("CA-1M scene IDs must be numeric")
    return scenes


def exact_files(root: Path, scenes: Iterable[str], suffix: str, label: str) -> dict[str, Path]:
    folder = directory(root, label)
    expected = {f"{scene}{suffix}" for scene in scenes}
    actual = {path.name for path in folder.iterdir() if path.is_file() and path.name.endswith(suffix)}
    if actual != expected:
        raise ValueError(
            f"{label} artifact set differs: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return {scene: regular(folder / f"{scene}{suffix}", label) for scene in scenes}


def prediction(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with regular(path, "prediction").open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], list):
        raise ValueError(f"invalid prediction payload: {path}")
    labels: list[int] = []
    corners: list[np.ndarray] = []
    scores: list[float] = []
    for row in payload[0]:
        if not isinstance(row, tuple) or len(row) != 3:
            raise ValueError(f"invalid prediction row: {path}")
        label, box, score = row
        box_array = np.asarray(box, dtype=np.float32)
        if box_array.shape != (8, 3) or not np.isfinite(box_array).all():
            raise ValueError(f"invalid OBB corners: {path}")
        if not np.isfinite(float(score)):
            raise ValueError(f"invalid score: {path}")
        labels.append(int(label))
        corners.append(box_array)
        scores.append(float(score))
    return (
        np.asarray(labels, dtype=np.int64),
        np.stack(corners).astype(np.float32) if corners else np.empty((0, 8, 3), np.float32),
        np.asarray(scores, dtype=np.float32),
    )


def audit_boxer(path: Path, scene: str) -> dict[str, int]:
    rows = []
    for line in regular(path, "Boxer diagnostic").read_text().splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{scene}: invalid Boxer diagnostic row")
            rows.append(value)
    if not rows:
        raise ValueError(f"{scene}: Boxer diagnostic is empty")
    for row in rows:
        count = int(row.get("count", -1))
        common_invalid = (
            str(row.get("scene_id")) != scene
            or row.get("mode") != "active"
            or row.get("selective_gate_enabled") is not True
            or count < 0
        )
        # A zero-proposal call never invokes Boxer, so the producer
        # deliberately leaves checkpoint/commit metadata unset.  Such a row
        # is valid only when every selective-mutation counter is also zero.
        if count == 0:
            checkpoint_invalid = (
                row.get("boxer_checkpoint_sha256") is not None
                or row.get("boxer_commit") is not None
                or any(
                    int(row.get(key, -1)) != 0
                    for key in ("eligible_count", "applied_count", "fallback_count")
                )
            )
        else:
            checkpoint_invalid = row.get("boxer_checkpoint_sha256") != BOXER_SHA
        if common_invalid or checkpoint_invalid:
            raise ValueError(f"{scene}: Selective Boxer G0 diagnostic disagrees")
    return {
        "calls": len(rows),
        "eligible": sum(int(row.get("eligible_count", 0)) for row in rows),
        "applied": sum(int(row.get("applied_count", 0)) for row in rows),
        "fallback": sum(int(row.get("fallback_count", 0)) for row in rows),
    }


def _row_hashes(labels: np.ndarray, corners: np.ndarray, scores: np.ndarray) -> set[str]:
    result = set()
    for label, box, score in zip(labels, corners, scores):
        digest = hashlib.sha256()
        digest.update(np.int64(label).tobytes())
        digest.update(np.ascontiguousarray(box, dtype=np.float32).tobytes())
        digest.update(np.float32(score).tobytes())
        result.add(digest.hexdigest())
    return result


def audit_identity(args: argparse.Namespace) -> dict[str, Any]:
    scenes = read_scenes(args.scene_list, args.expected_scenes)
    active = exact_files(args.active_root, scenes, "_boxes.pkl", "active prediction root")
    identity = exact_files(args.identity_root, scenes, "_boxes.pkl", "identity root")
    boxer = exact_files(args.boxer_root, scenes, "_boxer_lifting.jsonl", "Boxer root")
    logs = exact_files(args.log_root, scenes, ".log", "inference log root")
    controls = (
        exact_files(args.control_root, scenes, "_boxes.pkl", "control prediction root")
        if args.control_root is not None else None
    )
    control_boxer = None
    control_logs = None
    if controls is not None:
        if args.control_boxer_root is None or args.control_log_root is None:
            raise ValueError(
                "paired control audit requires control Boxer and log roots"
            )
        control_boxer = exact_files(
            args.control_boxer_root,
            scenes,
            "_boxer_lifting.jsonl",
            "control Boxer root",
        )
        control_logs = exact_files(
            args.control_log_root, scenes, ".log", "control inference log root"
        )

    total_active = total_control = exact_common = 0
    paired_changed = identity_scenes = 0
    per_scene: dict[str, Any] = {}
    boxer_totals = {"calls": 0, "eligible": 0, "applied": 0, "fallback": 0}
    control_boxer_totals = {
        "calls": 0,
        "eligible": 0,
        "applied": 0,
        "fallback": 0,
    }
    for scene in scenes:
        if sha256(active[scene]) != sha256(identity[scene]):
            raise ValueError(f"{scene}: same-run byte identity failed")
        if not os.path.samefile(active[scene], identity[scene]):
            raise ValueError(f"{scene}: identity paths are not the finalizer hard-link pair")
        active_labels, active_corners, active_scores = prediction(active[scene])
        identity_labels, identity_corners, identity_scores = prediction(identity[scene])
        if not (
            np.array_equal(active_labels, identity_labels)
            and np.array_equal(active_corners, identity_corners)
            and np.array_equal(active_scores, identity_scores)
        ):
            raise ValueError(f"{scene}: same-run semantic identity failed")
        log_text = logs[scene].read_text(errors="replace")
        for marker in (
            "Appearance gate summary |",
            "Reliable-view fusion summary |",
            "Prediction same-run byte-identity anchor saved to",
        ):
            if marker not in log_text:
                raise ValueError(f"{scene}: active module marker missing: {marker}")
        if "eval mAP:" in log_text or "eval APrec:" in log_text:
            raise ValueError(f"{scene}: evaluator marker found before identity audit")
        boxer_summary = audit_boxer(boxer[scene], scene)
        for key in boxer_totals:
            boxer_totals[key] += boxer_summary[key]
        row: dict[str, Any] = {
            "active_rows": len(active_scores),
            "active_prediction_sha256": sha256(active[scene]),
            "identity_prediction_sha256": sha256(identity[scene]),
            "byte_identity": True,
            "semantic_identity": True,
            "hard_link_identity": True,
            "active_geometry_sha256": array_sha(active_corners),
            "active_scores_sha256": array_sha(active_scores),
            "boxer": boxer_summary,
        }
        total_active += len(active_scores)
        identity_scenes += 1
        if controls is not None:
            assert control_boxer is not None and control_logs is not None
            control_boxer_summary = audit_boxer(control_boxer[scene], scene)
            for key in control_boxer_totals:
                control_boxer_totals[key] += control_boxer_summary[key]
            control_log_text = control_logs[scene].read_text(errors="replace")
            if (
                "Appearance gate summary |" in control_log_text
                or "Reliable-view fusion summary |" in control_log_text
                or "eval mAP:" in control_log_text
                or "eval APrec:" in control_log_text
            ):
                raise ValueError(
                    f"{scene}: G0 control log enabled an active module/evaluator"
                )
            control_labels, control_corners, control_scores = prediction(controls[scene])
            changed = not (
                np.array_equal(control_labels, active_labels)
                and np.array_equal(control_corners, active_corners)
                and np.array_equal(control_scores, active_scores)
            )
            common = len(
                _row_hashes(control_labels, control_corners, control_scores)
                & _row_hashes(active_labels, active_corners, active_scores)
            )
            paired_changed += int(changed)
            exact_common += common
            total_control += len(control_scores)
            row["paired_g0_control"] = {
                "control_rows": len(control_scores),
                "row_count_delta": len(active_scores) - len(control_scores),
                "exact_common_rows": common,
                "prediction_identical": not changed,
                "identity_expected": False,
                "control_geometry_sha256": array_sha(control_corners),
                "control_scores_sha256": array_sha(control_scores),
                "sorted_score_multiset_identical": np.array_equal(
                    np.sort(control_scores), np.sort(active_scores)
                ),
                "boxer": control_boxer_summary,
            }
        per_scene[scene] = row

    return {
        "schema": SCHEMA_IDENTITY,
        "ok": True,
        "dataset": "CA1M",
        "split": args.split,
        "scene_count": len(scenes),
        "scene_list_sha256": sha256(args.scene_list),
        "same_run": {
            "byte_identity_scenes": identity_scenes,
            "semantic_identity_scenes": identity_scenes,
            "hard_link_identity_scenes": identity_scenes,
            "active_rows": total_active,
        },
        "paired_g0_control": (
            {
                "identity_expected": False,
                "control_rows": total_control,
                "active_rows": total_active,
                "row_count_delta": total_active - total_control,
                "scenes_with_any_difference": paired_changed,
                "exact_common_rows": exact_common,
            }
            if controls is not None else None
        ),
        "boxer": boxer_totals,
        "control_boxer": control_boxer_totals if controls is not None else None,
        "clip_appearance_gate_active": True,
        "reliable_view_top_k": 3,
        "ground_truth_access": False,
        "evaluation_invoked": False,
        "training_invoked": False,
        "scannet_learned_b6_or_gate_reused": False,
        "downstream_native_b6_recollection_required": True,
        "per_scene": per_scene,
    }


def write_json_create_only(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite report: {target}") from error
        target.chmod(0o444)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    contract = subparsers.add_parser("contract")
    contract.add_argument("--control-config", type=Path, required=True)
    contract.add_argument("--active-config", type=Path, required=True)
    contract.add_argument("--train-config", type=Path, required=True)
    contract.add_argument("--output", type=Path)
    identity = subparsers.add_parser("identity")
    identity.add_argument("--scene-list", type=Path, required=True)
    identity.add_argument("--expected-scenes", type=int, required=True)
    identity.add_argument("--split", required=True)
    identity.add_argument("--control-root", type=Path)
    identity.add_argument("--control-boxer-root", type=Path)
    identity.add_argument("--control-log-root", type=Path)
    identity.add_argument("--active-root", type=Path, required=True)
    identity.add_argument("--identity-root", type=Path, required=True)
    identity.add_argument("--boxer-root", type=Path, required=True)
    identity.add_argument("--log-root", type=Path, required=True)
    identity.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "contract":
        report = audit_contract(args.control_config, args.active_config, args.train_config)
    else:
        if args.expected_scenes < 1:
            parser.error("--expected-scenes must be positive")
        report = audit_identity(args)
    if args.output is not None:
        write_json_create_only(args.output, report)
    print(json.dumps({
        "schema": report["schema"],
        "ok": report["ok"],
        "ground_truth_access": report.get("ground_truth_access", False),
        "training_invoked": report.get("training_invoked", False),
        "scene_count": report.get("scene_count"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
