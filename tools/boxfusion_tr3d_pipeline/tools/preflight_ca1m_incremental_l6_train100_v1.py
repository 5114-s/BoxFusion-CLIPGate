#!/usr/bin/env python3
"""Static, GT-free preflight for the isolated CA-1M incremental/L6 route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_incremental_l6 import (  # noqa: E402
    ACTIVE_SCHEMA,
    CA_TR3D_BINDING_SCHEMA,
    CA_TR3D_BINDING_SHA256,
    CA_TR3D_CHECKPOINT_SHA256,
    DATASET_SCHEMA,
    OBSERVER_SCHEMA,
    POLICY_SCHEMA,
    SCORE_POLICY,
    SOURCE_RANK_FORMULA,
    SPLIT_COUNTS,
    SPLIT_FOLDS,
    SPLIT_SHA256,
    UPSTREAM_ROUTE,
    sha256_file,
)
from boxfusion.ca1m_tr3d_checkpoint_binding import (  # noqa: E402
    FORBIDDEN_SCANNET_SHA256,
    load_checkpoint_binding,
)


CONFIG_SCHEMA = "boxfusion.ca1m_incremental_l6_protocol.v1"
REPORT_SCHEMA = "boxfusion.ca1m_incremental_l6_preflight.v1"
NAMESPACE = "ca1m_incremental_l6_ca_native_train100_v1"
FINAL_BASE_SCHEMA = "boxfusion.ca1m_final_base_identity_audit.v1"
B6_COLLECTION_SCHEMA = "boxfusion.ca1m_native_b6_final_base_train_collection.v2"
B6_CHECKPOINT_MANIFEST_SCHEMA = "boxfusion.ca1m_native_b6_checkpoint_manifest.v1"
TERMINAL_PROPOSAL_SCHEMA = "boxfusion.ca1m_tr3d_anchor_free_proposal_cache.v4"
TERMINAL_OVERLAY_SCHEMA = "boxfusion.ca1m_tr3d_terminal_cpu_overlay.v4"
TERMINAL_SEAL_SCHEMA = "boxfusion.ca1m_tr3d_terminal_train100_seal.v4"
TERMINAL_POLICY_SCHEMA = "boxfusion.ca1m_tr3d_terminal_gate_policy.v2"
TERMINAL_ACTIVE_SCHEMA = "boxfusion.ca1m_tr3d_terminal_active_anchor.v2"
TERMINAL_CROSSFIT_SCHEMA = "boxfusion.ca1m_tr3d_terminal_crossfit_train100.v2"
FRAME_LINEAGE_SCHEMA = "boxfusion.ca1m_demo_gap20_early_finalize_lineage.v1"

_SCENE = re.compile(r"^[0-9]{8}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_DYNAMIC_UPSTREAM_FIELDS = {
    "final_base": ("anchor_root", "manifest", "manifest_sha256"),
    "native_b6_v2": (
        "diagnostics_root",
        "collection_manifest",
        "collection_manifest_sha256",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_manifest",
        "checkpoint_manifest_sha256",
    ),
    "terminal_v4": (
        "proposal_root",
        "overlay_root",
        "sealed_manifest",
        "sealed_manifest_sha256",
    ),
    "terminal_benefit_v2": (
        "policy",
        "policy_sha256",
        "post_terminal_anchor_root",
        "post_terminal_anchor_manifest",
        "post_terminal_anchor_manifest_sha256",
        "cross_fitted_train100_receipt",
        "cross_fitted_train100_receipt_sha256",
    ),
}


def _keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected-set(value))}, "
            f"extra={sorted(set(value)-expected)}"
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _regular_file(path: Path, label: str, *, immutable: bool = False) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {raw}")
    source = raw.resolve()
    if not source.is_file() or source.is_symlink() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"missing regular {label}: {source}")
    if immutable and source.stat().st_mode & 0o222:
        raise ValueError(f"{label} must be read-only: {source}")
    return source


def _regular_directory(path: Path, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {raw}")
    source = raw.resolve()
    if not source.is_dir() or source.is_symlink():
        raise FileNotFoundError(f"missing regular {label}: {source}")
    return source


def _json(path: Path, label: str, *, immutable: bool = False) -> tuple[Path, dict[str, Any]]:
    source = _regular_file(path, label, immutable=immutable)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return source, value


def _scene_file(record: Mapping[str, Any], role: str) -> tuple[Path, tuple[str, ...]]:
    common = {"path", "sha256", "scene_count"}
    extras: set[str]
    if role in SPLIT_FOLDS:
        extras = {"folds"}
        if role == "locked_internal_check":
            extras.add("one_time_access")
    elif role == "official_validation_forbidden":
        extras = {"identity_only_no_ground_truth"}
    else:
        extras = set()
    _keys(record, common | extras, f"scene_splits.{role}")
    source = _regular_file(Path(str(record["path"])), f"{role} scene list")
    if (
        record["sha256"] != SPLIT_SHA256[role]
        or sha256_file(source) != SPLIT_SHA256[role]
        or record["scene_count"] != SPLIT_COUNTS[role]
    ):
        raise ValueError(f"{role} scene-list identity differs")
    if role in SPLIT_FOLDS and tuple(record["folds"]) != SPLIT_FOLDS[role]:
        raise ValueError(f"{role} fold role differs")
    if role == "locked_internal_check" and record["one_time_access"] is not True:
        raise ValueError("locked fold must remain one-time access")
    if role == "official_validation_forbidden" and (
        record["identity_only_no_ground_truth"] is not True
    ):
        raise ValueError("official validation list may be used only for identity exclusion")
    scenes = tuple(
        row.strip() for row in source.read_text(encoding="utf-8").splitlines()
        if row.strip()
    )
    if (
        len(scenes) != SPLIT_COUNTS[role]
        or len(scenes) != len(set(scenes))
        or any(_SCENE.fullmatch(scene) is None for scene in scenes)
    ):
        raise ValueError(f"{role} must contain exact unique numeric CA scene IDs")
    return source, scenes


def _validate_splits(value: Mapping[str, Any]) -> dict[str, Any]:
    roles = {
        "train100",
        "weights_train",
        "threshold_dev",
        "locked_internal_check",
        "official_validation_forbidden",
    }
    _keys(value, roles, "scene_splits")
    paths: dict[str, Path] = {}
    scenes: dict[str, tuple[str, ...]] = {}
    for role in sorted(roles):
        paths[role], scenes[role] = _scene_file(
            _mapping(value[role], f"scene_splits.{role}"), role
        )
    train_roles = ("weights_train", "threshold_dev", "locked_internal_check")
    union = set().union(*(set(scenes[role]) for role in train_roles))
    if (
        any(
            set(scenes[left]) & set(scenes[right])
            for index, left in enumerate(train_roles)
            for right in train_roles[index + 1 :]
        )
        or union != set(scenes["train100"])
    ):
        raise ValueError("CA 60/20/20 roles are not an exact disjoint train100 partition")
    overlap = sorted(set(scenes["train100"]) & set(scenes["official_validation_forbidden"]))
    if overlap:
        raise ValueError(f"CA train100 overlaps forbidden validation: {overlap[:5]}")
    return {
        "train100_scene_count": len(scenes["train100"]),
        "weights_train_scene_count": len(scenes["weights_train"]),
        "threshold_dev_scene_count": len(scenes["threshold_dev"]),
        "locked_internal_scene_count": len(scenes["locked_internal_check"]),
        "forbidden_validation_scene_count": len(scenes["official_validation_forbidden"]),
        "validation_overlap_count": 0,
        "ground_truth_files_opened": False,
        "scene_list_paths": {role: str(path) for role, path in paths.items()},
    }


def _validate_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    _keys(
        value,
        {
            "schema",
            "path",
            "sha256",
            "checkpoint_sha256",
            "initialization",
            "raw_checkpoint_argument_allowed",
            "raw_config_argument_allowed",
            "scannet_checkpoint_or_config_allowed",
        },
        "ca_native_tr3d_binding",
    )
    if value != {
        "schema": CA_TR3D_BINDING_SCHEMA,
        "path": str(
            ROOT
            / "manifests/ca1m_tr3d_terminal_ca_native_train100_v3/checkpoint_binding.json"
        ),
        "sha256": CA_TR3D_BINDING_SHA256,
        "checkpoint_sha256": CA_TR3D_CHECKPOINT_SHA256,
        "initialization": "ca1m_random_scratch",
        "raw_checkpoint_argument_allowed": False,
        "raw_config_argument_allowed": False,
        "scannet_checkpoint_or_config_allowed": False,
    }:
        raise ValueError("CA-scratch TR3D binding config differs")
    binding = load_checkpoint_binding(Path(str(value["path"])))
    if (
        binding.manifest_sha256 != CA_TR3D_BINDING_SHA256
        or binding.checkpoint_sha256 != CA_TR3D_CHECKPOINT_SHA256
        or binding.checkpoint_sha256 in FORBIDDEN_SCANNET_SHA256
        or binding.effective_config_sha256 in FORBIDDEN_SCANNET_SHA256
    ):
        raise ValueError("bound TR3D model is not the sealed CA-scratch checkpoint")
    return {
        "schema": CA_TR3D_BINDING_SCHEMA,
        "manifest": str(binding.manifest_path),
        "manifest_sha256": binding.manifest_sha256,
        "checkpoint_sha256": binding.checkpoint_sha256,
        "initialization": "ca1m_random_scratch",
        "scannet_artifact_access": False,
    }


def _sha_bound(path: Path, expected: str, label: str) -> Path:
    if _SHA.fullmatch(str(expected)) is None:
        raise ValueError(f"{label} lacks a lowercase SHA256")
    source = _regular_file(path, label, immutable=True)
    if sha256_file(source) != expected:
        raise ValueError(f"{label} SHA256 mismatch")
    if expected in FORBIDDEN_SCANNET_SHA256:
        raise ValueError(f"{label} matches a forbidden ScanNet artifact")
    return source


def _manifest(
    path: Path,
    expected_sha: str,
    expected_schema: str,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    source = _sha_bound(path, expected_sha, label)
    _, value = _json(source, label, immutable=True)
    if value.get("schema") != expected_schema or value.get("complete") is not True:
        raise ValueError(f"{label} schema/completion differs")
    if value.get("validation_ground_truth_access") is True:
        raise ValueError(f"{label} accessed validation GT")
    return source, value


def _flatten_dynamic(upstream: Mapping[str, Any]) -> list[Any]:
    return [
        _mapping(upstream[section], f"required_upstream_chain.{section}")[name]
        for section, names in _DYNAMIC_UPSTREAM_FIELDS.items()
        for name in names
    ]


def _validate_bound_upstream(upstream: Mapping[str, Any]) -> dict[str, Any]:
    final = _mapping(upstream["final_base"], "required_upstream_chain.final_base")
    b6 = _mapping(upstream["native_b6_v2"], "required_upstream_chain.native_b6_v2")
    terminal = _mapping(upstream["terminal_v4"], "required_upstream_chain.terminal_v4")
    benefit = _mapping(
        upstream["terminal_benefit_v2"],
        "required_upstream_chain.terminal_benefit_v2",
    )
    directories = {
        "final-base anchor root": final["anchor_root"],
        "native-B6 v2 diagnostics root": b6["diagnostics_root"],
        "terminal-v4 proposal root": terminal["proposal_root"],
        "terminal-v4 overlay root": terminal["overlay_root"],
        "post-terminal active anchor root": benefit["post_terminal_anchor_root"],
    }
    forbidden_names = (
        "ca1m_tr3d_terminal_train100_v1",
        "ca1m_tr3d_terminal_train100_v2",
        "ca1m_tr3d_terminal_ca_native_train100_v3",
        "ca1m_native_b6_canonical103_v1",
    )
    resolved_directories: dict[str, str] = {}
    for label, raw in directories.items():
        if any(name in str(raw) for name in forbidden_names):
            raise ValueError(f"{label} points at a forbidden old namespace")
        resolved_directories[label] = str(_regular_directory(Path(str(raw)), label))
    final_manifest, _ = _manifest(
        Path(str(final["manifest"])),
        str(final["manifest_sha256"]),
        FINAL_BASE_SCHEMA,
        "sealed final-base manifest",
    )
    b6_collection, b6_collection_value = _manifest(
        Path(str(b6["collection_manifest"])),
        str(b6["collection_manifest_sha256"]),
        B6_COLLECTION_SCHEMA,
        "sealed native-B6 v2 collection manifest",
    )
    if (
        b6_collection_value.get("train_only") is not True
        or b6_collection_value.get("old_native_b6_diagnostics_reused") is True
        or b6_collection_value.get("old_native_b6_checkpoint_reused") is True
    ):
        raise ValueError("native-B6 v2 collection provenance is not isolated")
    b6_checkpoint = _sha_bound(
        Path(str(b6["checkpoint"])),
        str(b6["checkpoint_sha256"]),
        "sealed native-B6 v2 checkpoint",
    )
    b6_checkpoint_manifest, b6_checkpoint_value = _manifest(
        Path(str(b6["checkpoint_manifest"])),
        str(b6["checkpoint_manifest_sha256"]),
        B6_CHECKPOINT_MANIFEST_SCHEMA,
        "sealed native-B6 v2 checkpoint manifest",
    )
    if (
        b6_checkpoint_value.get("train_only") is not True
        or b6_checkpoint_value.get("activation_authorized") is not True
        or (b6_checkpoint_value.get("checkpoint") or {}).get("sha256")
        != sha256_file(b6_checkpoint)
        or "final_base" not in json.dumps(b6_checkpoint_value, sort_keys=True)
    ):
        raise ValueError("native-B6 checkpoint is not the new final-base v2 model")
    terminal_manifest, terminal_value = _manifest(
        Path(str(terminal["sealed_manifest"])),
        str(terminal["sealed_manifest_sha256"]),
        TERMINAL_SEAL_SCHEMA,
        "sealed terminal-v4 train100 manifest",
    )
    if (
        terminal_value.get("train_only") is not True
        or terminal_value.get("proposal_schema") != TERMINAL_PROPOSAL_SCHEMA
        or terminal_value.get("overlay_schema") != TERMINAL_OVERLAY_SCHEMA
        or terminal_value.get("ca_native_tr3d_binding_sha256")
        != CA_TR3D_BINDING_SHA256
    ):
        raise ValueError("terminal-v4 seal does not bind the new CA-only route")
    policy, policy_value = _manifest(
        Path(str(benefit["policy"])),
        str(benefit["policy_sha256"]),
        TERMINAL_POLICY_SCHEMA,
        "sealed terminal-benefit v2 policy",
    )
    if (
        policy_value.get("activation_authorized") is not True
        or policy_value.get("train_only") is not True
        or policy_value.get("ca_native_tr3d_binding_sha256")
        != CA_TR3D_BINDING_SHA256
        or policy_value.get("terminal_v4_manifest_sha256")
        != sha256_file(terminal_manifest)
    ):
        raise ValueError("terminal-benefit policy is not the CA v4-derived v2 gate")
    post_manifest, post_value = _manifest(
        Path(str(benefit["post_terminal_anchor_manifest"])),
        str(benefit["post_terminal_anchor_manifest_sha256"]),
        TERMINAL_ACTIVE_SCHEMA,
        "sealed post-terminal anchor manifest",
    )
    if (
        post_value.get("activation_authorized") is not True
        or post_value.get("policy_sha256") != sha256_file(policy)
        or post_value.get("source_final_base_manifest_sha256")
        != sha256_file(final_manifest)
        or post_value.get("source_native_b6_collection_manifest_sha256")
        != sha256_file(b6_collection)
    ):
        raise ValueError("post-terminal anchor does not bind the complete new chain")
    crossfit, crossfit_value = _manifest(
        Path(str(benefit["cross_fitted_train100_receipt"])),
        str(benefit["cross_fitted_train100_receipt_sha256"]),
        TERMINAL_CROSSFIT_SCHEMA,
        "terminal-benefit train100 cross-fit receipt",
    )
    if (
        crossfit_value.get("train_only") is not True
        or crossfit_value.get("scene_count") != 100
        or crossfit_value.get("all_scene_predictions_out_of_fold") is not True
        or crossfit_value.get("official_validation_access") is not False
        or crossfit_value.get("terminal_policy_sha256") != sha256_file(policy)
    ):
        raise ValueError("terminal anchor lacks exact100 cross-fitted provenance")
    return {
        "bound": True,
        "final_base_manifest_sha256": sha256_file(final_manifest),
        "native_b6_v2_collection_manifest_sha256": sha256_file(b6_collection),
        "native_b6_v2_checkpoint_manifest_sha256": sha256_file(b6_checkpoint_manifest),
        "terminal_v4_manifest_sha256": sha256_file(terminal_manifest),
        "terminal_benefit_v2_policy_sha256": sha256_file(policy),
        "post_terminal_anchor_manifest_sha256": sha256_file(post_manifest),
        "crossfit_receipt_sha256": sha256_file(crossfit),
        "directories": resolved_directories,
    }


def _validate_upstream(value: Mapping[str, Any]) -> dict[str, Any]:
    _keys(
        value,
        {
            "route",
            "all_or_none_binding",
            "final_base",
            "native_b6_v2",
            "terminal_v4",
            "terminal_benefit_v2",
        },
        "required_upstream_chain",
    )
    if value["route"] != UPSTREAM_ROUTE or value["all_or_none_binding"] is not True:
        raise ValueError("upstream route/all-or-none contract differs")
    final = _mapping(value["final_base"], "required_upstream_chain.final_base")
    _keys(
        final,
        {"manifest_schema", *_DYNAMIC_UPSTREAM_FIELDS["final_base"]},
        "required_upstream_chain.final_base",
    )
    if final["manifest_schema"] != FINAL_BASE_SCHEMA:
        raise ValueError("final-base schema differs")
    b6 = _mapping(value["native_b6_v2"], "required_upstream_chain.native_b6_v2")
    _keys(
        b6,
        {"collection_manifest_schema", *_DYNAMIC_UPSTREAM_FIELDS["native_b6_v2"]},
        "required_upstream_chain.native_b6_v2",
    )
    if b6["collection_manifest_schema"] != B6_COLLECTION_SCHEMA:
        raise ValueError("native-B6 v2 schema differs")
    terminal = _mapping(value["terminal_v4"], "required_upstream_chain.terminal_v4")
    _keys(
        terminal,
        {
            "proposal_schema",
            "overlay_schema",
            "sealed_manifest_schema",
            *_DYNAMIC_UPSTREAM_FIELDS["terminal_v4"],
        },
        "required_upstream_chain.terminal_v4",
    )
    if (
        terminal["proposal_schema"] != TERMINAL_PROPOSAL_SCHEMA
        or terminal["overlay_schema"] != TERMINAL_OVERLAY_SCHEMA
        or terminal["sealed_manifest_schema"] != TERMINAL_SEAL_SCHEMA
    ):
        raise ValueError("terminal-v4 schemas differ")
    benefit = _mapping(
        value["terminal_benefit_v2"],
        "required_upstream_chain.terminal_benefit_v2",
    )
    _keys(
        benefit,
        {
            "policy_schema",
            "active_anchor_manifest_schema",
            *_DYNAMIC_UPSTREAM_FIELDS["terminal_benefit_v2"],
        },
        "required_upstream_chain.terminal_benefit_v2",
    )
    if (
        benefit["policy_schema"] != TERMINAL_POLICY_SCHEMA
        or benefit["active_anchor_manifest_schema"] != TERMINAL_ACTIVE_SCHEMA
    ):
        raise ValueError("terminal-benefit v2 schemas differ")
    dynamic = _flatten_dynamic(value)
    if all(item is None for item in dynamic):
        return {
            "bound": False,
            "pending_sections": list(_DYNAMIC_UPSTREAM_FIELDS),
            "artifacts_opened": False,
        }
    if any(item is None for item in dynamic):
        raise ValueError("new upstream chain is partially bound; all-or-none required")
    return _validate_bound_upstream(value)


def _validate_candidate(value: Mapping[str, Any]) -> None:
    _keys(
        value,
        {
            "schema",
            "status",
            "run_authorized",
            "processed_rgbd_root",
            "frame_lineage_schema",
            "frame_gap",
            "causal_incremental",
            "latest_only_async",
            "visibility_top_k",
            "source_features",
            "output_root",
            "ground_truth_access",
            "validation_access",
            "create_only",
            "exact_scene_count",
        },
        "candidate_collection",
    )
    expected = dict(value)
    scalar = {
        "schema": OBSERVER_SCHEMA,
        "status": "pending_upstream_chain",
        "run_authorized": False,
        "processed_rgbd_root": "/extra/ZhaoX/boxfusion_ca1m_native_b6_train100_v1",
        "frame_lineage_schema": FRAME_LINEAGE_SCHEMA,
        "frame_gap": 20,
        "causal_incremental": True,
        "latest_only_async": True,
        "visibility_top_k": 5,
        "source_features": [
            "visibility_quality_mean",
            "support_ratio_mean",
            "free_space_ratio_mean",
            "invalid_ratio_mean",
            "selected_geometry",
        ],
        "output_root": str(
            ROOT
            / "diagnostics/ca1m_incremental_l6_ca_native_train100_v1/observer"
        ),
        "ground_truth_access": False,
        "validation_access": False,
        "create_only": True,
        "exact_scene_count": 100,
    }
    if expected != scalar:
        raise ValueError("candidate-collection static contract differs")
    _regular_directory(
        Path(str(value["processed_rgbd_root"])), "processed CA train100 RGB-D root"
    )


def _validate_training(value: Mapping[str, Any]) -> None:
    _keys(
        value,
        {
            "dataset_schema",
            "policy_schema",
            "terminal_anchor_cross_fitted",
            "fit_folds",
            "threshold_calibration_folds",
            "one_time_locked_audit_folds",
            "targets",
            "model",
            "normalization",
            "threshold_selection",
            "activation_decision",
            "sample_gate",
            "output_dataset",
            "output_dataset_manifest",
            "output_policy",
        },
        "training_protocol",
    )
    if (
        value["dataset_schema"] != DATASET_SCHEMA
        or value["policy_schema"] != POLICY_SCHEMA
        or value["terminal_anchor_cross_fitted"] is not True
        or tuple(value["fit_folds"]) != (2, 3, 4)
        or tuple(value["threshold_calibration_folds"]) != (0,)
        or tuple(value["one_time_locked_audit_folds"]) != (1,)
        or value["model"] != "dual_scene_grouped_logistic_novelty25_quality50"
        or value["normalization"] != "weights_train_only"
        or value["threshold_selection"] != "fold0_only"
        or value["activation_decision"] != "fold1_one_time_only"
    ):
        raise ValueError("CA L6 scientific split/model contract differs")
    if value["targets"] != {
        "novel25": "candidate_gt_iou_ge_0.25_and_post_terminal_anchor_gt_iou_lt_0.25",
        "quality50": "candidate_gt_iou_ge_0.50",
        "target_switch": "hard_negative",
    }:
        raise ValueError("CA L6 training targets differ")
    if value["sample_gate"] != {
        "weights_train_candidates_min": 120,
        "weights_train_novel25_positive_min": 20,
        "weights_train_novel25_negative_min": 20,
        "weights_train_novel50_positive_min": 10,
        "threshold_dev_candidates_min": 20,
        "threshold_dev_positive_scenes_min": 4,
        "locked_internal_candidates_min": 20,
        "locked_internal_positive_scenes_min": 4,
        "failure_action": "block_without_lowering_thresholds",
    }:
        raise ValueError("CA L6 sample sufficiency gate differs")
    expected_outputs = {
        "output_dataset": ROOT / "datasets/ca1m_incremental_l6_train100_v1.npz",
        "output_dataset_manifest": ROOT / "reports/ca1m_incremental_l6_train100_v1/dataset_manifest.json",
        "output_policy": ROOT / "models/ca1m_incremental_l6_policy_train100_v1.json",
    }
    for name, expected in expected_outputs.items():
        if Path(str(value[name])).resolve() != expected.resolve():
            raise ValueError(f"CA L6 {name} namespace differs")


def _validate_runtime(value: Mapping[str, Any]) -> None:
    if value != {
        "schema": ACTIVE_SCHEMA,
        "append_only": True,
        "anchor_rows_first_and_byte_identical": True,
        "replacement_allowed": False,
        "deletion_allowed": False,
        "class_agnostic": True,
        "ground_truth_access": False,
        "source_rank_formula": SOURCE_RANK_FORMULA,
        "score_policy": SCORE_POLICY,
        "candidate_nms_iou": 0.25,
        "max_candidates_per_scene_upper_bound": 32,
        "create_only": True,
    }:
        raise ValueError("CA L6 append-only runtime contract differs")


def _validate_forbidden(value: Mapping[str, Any]) -> None:
    expected = {
        "scannet_policy_schemas": [
            "boxfusion.tr3d_incremental_novelty_gate.v1",
            "boxfusion.tr3d_c3_source_gate_policy.v1",
        ],
        "scannet_checkpoint_sha256": sorted(FORBIDDEN_SCANNET_SHA256),
        "old_ca_terminal_namespaces": [
            "ca1m_tr3d_terminal_train100_v1",
            "ca1m_tr3d_terminal_train100_v2",
            "ca1m_tr3d_terminal_ca_native_train100_v3",
        ],
        "old_ca_native_b6_namespaces": [
            "ca1m_native_b6_train100_v1",
            "ca1m_native_b6_canonical103_v1",
        ],
        "exception": "v3 checkpoint_binding.json_is_model_identity_only_not_a_terminal_cache",
        "old_policy_access": False,
        "old_terminal_cache_access": False,
        "old_native_b6_artifact_access": False,
        "raw_model_override_access": False,
    }
    actual = dict(value)
    actual["scannet_checkpoint_sha256"] = sorted(
        actual.get("scannet_checkpoint_sha256", [])
    )
    if actual != expected:
        raise ValueError("forbidden-reuse denylist differs")


def validate_config(config_path: Path) -> dict[str, Any]:
    source, cfg = _json(config_path, "CA L6 config")
    _keys(
        cfg,
        {
            "schema",
            "namespace",
            "dataset",
            "method_source",
            "status",
            "execution_driver_implemented",
            "run_authorized",
            "train_only",
            "observer_only_until_policy_authorized",
            "validation_ground_truth_access",
            "validation_prediction_access",
            "evaluator_access",
            "scene_splits",
            "ca_native_tr3d_binding",
            "required_upstream_chain",
            "candidate_collection",
            "training_protocol",
            "runtime_protocol",
            "forbidden_reuse",
        },
        "CA L6 config",
    )
    if (
        cfg["schema"] != CONFIG_SCHEMA
        or cfg["namespace"] != NAMESPACE
        or cfg["dataset"] != "ca1m"
        or cfg["method_source"]
        != "scannet_l6_method_only_no_scannet_trained_artifact"
        or cfg["status"] != "static_contract_only_pending_new_upstream_chain"
        or cfg["execution_driver_implemented"] is not False
        or cfg["run_authorized"] is not False
        or cfg["train_only"] is not True
        or cfg["observer_only_until_policy_authorized"] is not True
        or cfg["validation_ground_truth_access"] is not False
        or cfg["validation_prediction_access"] is not False
        or cfg["evaluator_access"] is not False
    ):
        raise ValueError("CA L6 top-level static isolation contract differs")
    split_report = _validate_splits(
        _mapping(cfg["scene_splits"], "scene_splits")
    )
    binding_report = _validate_binding(
        _mapping(cfg["ca_native_tr3d_binding"], "ca_native_tr3d_binding")
    )
    upstream_report = _validate_upstream(
        _mapping(cfg["required_upstream_chain"], "required_upstream_chain")
    )
    _validate_candidate(_mapping(cfg["candidate_collection"], "candidate_collection"))
    _validate_training(_mapping(cfg["training_protocol"], "training_protocol"))
    _validate_runtime(_mapping(cfg["runtime_protocol"], "runtime_protocol"))
    _validate_forbidden(_mapping(cfg["forbidden_reuse"], "forbidden_reuse"))
    blocked = []
    if not upstream_report["bound"]:
        blocked.append("new_final_base_b6_v2_terminal_benefit_chain_pending")
    if cfg["candidate_collection"]["run_authorized"] is not True:
        blocked.append("candidate_collection_authorization_pending")
    if cfg["execution_driver_implemented"] is not True:
        blocked.append("execution_driver_not_implemented")
    if cfg["run_authorized"] is not True:
        blocked.append("full_run_authorization_pending")
    return {
        "schema": REPORT_SCHEMA,
        "complete": True,
        "config": str(source),
        "config_sha256": sha256_file(source),
        "namespace": NAMESPACE,
        "static_contract_ready": True,
        "prerequisites_complete": not blocked,
        "run_authorized": not blocked,
        "blocked_reasons": blocked,
        "scene_split": split_report,
        "ca_native_tr3d_binding": binding_report,
        "required_upstream_chain": upstream_report,
        "training_split": {
            "fit_folds": [2, 3, 4],
            "threshold_dev_folds": [0],
            "locked_internal_folds": [1],
            "terminal_anchor_cross_fitted": True,
        },
        "append_contract": {
            "append_only": True,
            "anchor_rows_first_and_byte_identical": True,
            "candidate_scores_below_every_anchor": True,
            "source_rank_formula": SOURCE_RANK_FORMULA,
        },
        "old_scannet_policy_or_checkpoint_access": False,
        "old_ca_terminal_or_b6_cache_access": False,
        "validation_ground_truth_files_opened": False,
        "validation_prediction_files_opened": False,
        "evaluator_started": False,
        "gpu_started": False,
        "model_started": False,
        "prediction_written": False,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/ca1m_incremental_l6_train100_v1.json",
    )
    value.add_argument("--require-run", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        report = validate_config(args.config)
    except (ValueError, FileNotFoundError, PermissionError) as error:
        print(json.dumps({
            "schema": REPORT_SCHEMA,
            "complete": False,
            "run_authorized": False,
            "gpu_started": False,
            "model_started": False,
            "validation_ground_truth_files_opened": False,
            "error": str(error),
        }, indent=2, sort_keys=True))
        print(f"CA L6 preflight failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_run and not report["run_authorized"]:
        print(
            "--run is fail-closed: " + ", ".join(report["blocked_reasons"]),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
