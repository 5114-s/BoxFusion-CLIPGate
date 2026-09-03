"""Fail-closed provenance contract for the CA-1M terminal benefit gate v4.

The numerical 40-D feature construction and dual-logistic optimizer from the
older experiment remain reusable algorithms.  No old artifact is reusable.
This module binds the algorithms exclusively to the final-base train100,
CA-native B6-v2, CA-scratch terminal-v4 P/O chain.  In particular, stacked
training consumes all-fold OOF B6 anchor scores, never deploy-model scores.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

import numpy as np

from .ca1m_native_b6_observer import FEATURE_NAMES as NATIVE_FEATURE_NAMES
from .ca1m_tr3d_terminal import world_aabb
from .ca1m_tr3d_terminal_v4 import (
    OVERLAY_SCHEMA,
    PROPOSAL_SCHEMA,
    load_overlay_cache,
    load_proposal_cache,
    sha256_file,
)


CONFIG_SCHEMA = "boxfusion.ca1m_tr3d_benefit_gate_config.v4"
BINDING_SCHEMA = "boxfusion.ca1m_tr3d_benefit_gate_training_binding.v4"
PREREGISTRATION_SCHEMA = "boxfusion.ca1m_tr3d_benefit_gate_preregistration.v4"
GT_SHADOW_INVENTORY_SCHEMA = (
    "boxfusion.ca1m_tr3d_benefit_gate_gt_shadow_inventory.v1"
)
DATASET_SCHEMA = "boxfusion.ca1m_tr3d_benefit_gate_dataset.v4"
POLICY_SCHEMA = "boxfusion.ca1m_tr3d_terminal_gate_policy.v4"
MATERIALIZATION_SCHEMA = "boxfusion.ca1m_tr3d_terminal_materialization.v4"
FEATURE_SCHEMA = "boxfusion.ca1m_tr3d_terminal_gate_features.v4"
SELECTION_SCHEMA = "boxfusion.ca1m_tr3d_terminal_gate_selection.v4"
QUALITY_TARGET = "candidate_max_gt_iou_strict_gt_0.25"
BENEFIT_TARGET = "same_best_gt_and_same_gt_iou_gain_ge_0.05"
SELECTION_RULE = "benefit_quality_candidate_score_row_v1"
CANDIDATE_EVIDENCE_MANIFEST_SCHEMA = (
    "boxfusion.ca1m_tr3d_candidate_evidence_collection.v4"
)
PROPOSAL_COLLECTION_SCHEMA = "boxfusion.ca1m_tr3d_proposal_collection.v4"
OVERLAY_COLLECTION_SCHEMA = "boxfusion.ca1m_tr3d_terminal_overlay_collection.v2"
FINAL_BASE_SCHEMA = "boxfusion.ca1m_final_base_identity_audit.v1"
B6_COLLECTION_SCHEMA = "boxfusion.ca1m_native_b6_final_base_train_collection.v2"
B6_CHECKPOINT_SCHEMA = "boxfusion.ca1m_native_b6_iou_mlp.v1"
B6_CHECKPOINT_MANIFEST_SCHEMA = "boxfusion.ca1m_native_b6_checkpoint_manifest.v1"
B6_OOF_SCHEMA = "boxfusion.ca1m_native_b6_oof_row_scores.v2"
B6_OOF_MANIFEST_SCHEMA = "boxfusion.ca1m_native_b6_oof_row_scores_manifest.v2"
NAMESPACE = "ca1m_tr3d_benefit_gate_final_base_v4"
SPLIT_NAMESPACE = "boxfusion.ca1m-native-b6.scene-folds.v1"
GATE_TRAIN_FOLDS = (2, 3, 4)
THRESHOLD_DEV_FOLDS = (0,)
LOCKED_INTERNAL_FOLDS = (1,)
TRAIN_SCENE_COUNT = 60
DEV_SCENE_COUNT = 20
LOCKED_SCENE_COUNT = 20
IOU_THRESHOLDS = (0.15, 0.25, 0.50)
MAX_REPLACEMENTS = 16
THRESHOLD_GRID = tuple(round(0.05 * value, 2) for value in range(1, 20)) + (0.99,)
DEV_GATE = {
    "min_delta_ap15": 0.0,
    "min_delta_ap25": 0.0,
    "min_delta_ap50": 0.0025,
    "min_replacements": 10,
    "min_scenes": 5,
    "min_positive_gain_fraction": 0.60,
    "max_severe_harm_fraction": 0.10,
    "max_target_switch_fraction": 0.10,
}
OBJECTIVE_FIELDS = (
    "delta_ap50_desc",
    "delta_ap25_desc",
    "delta_ap15_desc",
    "positive_gain_fraction_desc",
    "replacement_count_asc",
    "quality_threshold_desc",
    "benefit_threshold_desc",
)
AP_PROTOCOL = {
    "box_geometry": "world_enclosing_axis_aligned_bounding_box",
    "ranking": "global_score_numpy_default_argsort",
    "matching": "per_scene_per_gt_duplicate_aware_first_detection",
    "iou_comparison": "strict_greater_than",
    "iou_thresholds": list(IOU_THRESHOLDS),
    "recall_denominator_epsilon": 1.0e-6,
    "voc_ap_2009_continuous": True,
    "single_class": True,
}
TIE_PROTOCOL = {
    "anchor_scores_preserved_by_geometry_only_materialization": True,
    "fold0_oof_scores_must_be_unique": True,
    "numpy_default_argsort_required": True,
    "default_and_stable_order_must_match": True,
}
FIT_ITERATIONS = 2000
FIT_LEARNING_RATE = 0.05
FIT_DECAY_STEPS = 200.0
FIT_L2 = 0.002
FIT_PROTOCOL = {
    "algorithm": "dual_class_balanced_logistic_v2_pure_math",
    "iterations": FIT_ITERATIONS,
    "learning_rate": FIT_LEARNING_RATE,
    "decay_steps": FIT_DECAY_STEPS,
    "l2": FIT_L2,
    "standardization_fit_rows_only": True,
    "class_balanced_fit_weights": True,
}
FAILURE_ACTION = "stop_without_opening_fold1_or_canonical103"

RELATION_FEATURE_NAMES = (
    "candidate_minus_anchor_score",
    "candidate_anchor_iou",
    "center_distance_over_anchor_diagonal",
    "log_candidate_over_anchor_volume",
    "extent_log_ratio_l2",
    "log1p_candidate_point_support",
    "log1p_candidate_point_density",
    "candidate_point_support_fraction",
    "candidate_global_rank_fraction",
    "candidate_anchor_group_rank_fraction",
    "log1p_anchor_group_size",
    "candidate_score_minus_best_sibling",
)
FEATURE_NAMES = (
    tuple(f"anchor_{name}" for name in NATIVE_FEATURE_NAMES)
    + tuple(f"candidate_{name}" for name in NATIVE_FEATURE_NAMES)
    + RELATION_FEATURE_NAMES
)
if len(FEATURE_NAMES) != 40:  # pragma: no cover - import-time invariant
    raise RuntimeError("terminal gate v4 feature schema is not 40-D")

_SCENE = re.compile(r"^[0-9]{8}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_OLD_TOKENS = (
    "scannet",
    "ca1m_tr3d_terminal_train100_v1",
    "ca1m_tr3d_terminal_train100_v2",
    "ca1m_tr3d_terminal_ca_native_train100_v3",
    "ca1m_native_b6_train100_v1",
    "ca1m_native_b6_canonical103_v1",
    "ca1m_native_b6_iou_mlp_v1",
    "ca1m_tr3d_benefit_gate_v1",
    "ca1m_tr3d_benefit_fit_dev_v1",
    "ca1m_tr3d_benefit_fit_dev_v2",
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _sha(value: Any, name: str) -> str:
    result = str(value)
    if _SHA.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return result


def _regular_file(path: Path, name: str, *, immutable: bool = True) -> Path:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink: {path}")
    result = path.resolve()
    if not result.is_file() or result.is_symlink() or result.stat().st_size <= 0:
        raise FileNotFoundError(f"missing regular {name}: {result}")
    if immutable and result.stat().st_mode & 0o222:
        raise ValueError(f"{name} must be read-only: {result}")
    return result


def _regular_directory(path: Path, name: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink: {path}")
    result = path.resolve()
    if not result.is_dir() or result.is_symlink():
        raise FileNotFoundError(f"missing directory {name}: {result}")
    return result


def validate_candidate_evidence_artifact(
    path: Path,
    *,
    scene: str,
    expected_sha256: str,
    expected_root: Path | None = None,
) -> Path:
    """Validate one sealed v4 evidence artifact before any feature/GT join."""

    if _SCENE.fullmatch(scene) is None:
        raise ValueError("candidate-evidence scene id is invalid")
    source = _regular_file(path, f"candidate evidence {scene}")
    if source.name != f"{scene}_ca1m_tr3d_candidate_evidence_v4.npz":
        raise ValueError(f"{scene}: candidate-evidence filename differs")
    if expected_root is not None and source.parent != expected_root.resolve():
        raise ValueError(f"{scene}: candidate-evidence root differs")
    if sha256_file(source) != _sha(expected_sha256, f"candidate evidence {scene} SHA256"):
        raise ValueError(f"{scene}: candidate-evidence SHA256 differs")
    return source


def _json(path: Path, name: str, *, immutable: bool = True) -> tuple[Path, dict[str, Any]]:
    source = _regular_file(path, name, immutable=immutable)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return source, payload


def _record(value: Any, name: str, expected_schema: str) -> tuple[Path, dict[str, Any]]:
    record = _mapping(value, name)
    source, payload = _json(Path(str(record.get("path", ""))), name)
    if (
        record.get("schema") != expected_schema
        or payload.get("schema") != expected_schema
        or _sha(record.get("sha256"), f"{name} SHA256") != sha256_file(source)
    ):
        raise ValueError(f"{name} path/SHA/schema binding differs")
    return source, payload


def _assert_no_old_path(value: Any, name: str) -> None:
    if value is None:
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_no_old_path(child, f"{name}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_old_path(child, f"{name}[{index}]")
        return
    # Hashes and prose may legitimately mention a forbidden digest/token.  The
    # isolation rule applies to artifact path/root fields only.
    if not any(part in name.lower() for part in ("path", "root", "output")):
        return
    if name == "prerequisites.derived_train_gt_root":
        # This is immutable CA train annotation/raw-data provenance, not a B6
        # diagnostic, checkpoint, score, dataset, or learned gate artifact.
        return
    lowered = str(value).lower()
    for token in _OLD_TOKENS:
        if token in lowered:
            raise ValueError(f"{name} references forbidden legacy artifact token {token}")


def _scene_list(record: Mapping[str, Any]) -> tuple[Path, tuple[str, ...]]:
    source = _regular_file(Path(str(record.get("path", ""))), "train100 scene list")
    if (
        int(record.get("count", -1)) != 100
        or record.get("exact") is not True
        or _sha(record.get("sha256"), "scene-list SHA256") != sha256_file(source)
    ):
        raise ValueError("train100 scene-list binding differs")
    scenes = tuple(row.strip() for row in source.read_text().splitlines() if row.strip())
    if (
        len(scenes) != 100
        or len(set(scenes)) != 100
        or any(_SCENE.fullmatch(scene) is None for scene in scenes)
    ):
        raise ValueError("gate v4 requires exact 100 unique numeric CA train scenes")
    return source, scenes


def validate_static_config(path: Path) -> tuple[Path, dict[str, Any]]:
    source, cfg = _json(path, "gate-v4 config", immutable=False)
    required = {
        "schema", "namespace", "state", "run_authorized", "train_only",
        "ground_truth_used_only_after_candidate_seal",
        "validation_ground_truth_access", "validation_prediction_access",
        "official_validation_comparable", "scene_contract", "split",
        "prerequisites", "algorithms", "outputs", "materialization",
        "forbidden_reuse",
    }
    if set(cfg) != required:
        raise ValueError("gate-v4 config keys differ")
    if (
        cfg["schema"] != CONFIG_SCHEMA
        or cfg["namespace"] != NAMESPACE
        or cfg["train_only"] is not True
        or cfg["ground_truth_used_only_after_candidate_seal"] is not True
        or cfg["validation_ground_truth_access"] is not False
        or cfg["validation_prediction_access"] is not False
        or cfg["official_validation_comparable"] is not False
        or not isinstance(cfg["run_authorized"], bool)
    ):
        raise ValueError("gate-v4 top-level isolation contract differs")
    split = _mapping(cfg["split"], "split")
    if (
        split.get("namespace") != SPLIT_NAMESPACE
        or split.get("fold_count") != 5
        or tuple(split.get("gate_train_folds", ())) != GATE_TRAIN_FOLDS
        or tuple(split.get("threshold_dev_folds", ())) != THRESHOLD_DEV_FOLDS
        or tuple(split.get("locked_internal_check_folds", ())) != LOCKED_INTERNAL_FOLDS
        or split.get("gate_train_scene_count") != TRAIN_SCENE_COUNT
        or split.get("threshold_dev_scene_count") != DEV_SCENE_COUNT
        or split.get("locked_internal_scene_count") != LOCKED_SCENE_COUNT
        or split.get("anchor_score_source")
        != "ca1m_native_b6_final_base_all_fold_oof_row_scores_v2"
        or split.get("deploy_b6_scores_allowed_for_stacked_training") is not False
        or split.get("scene_grouped") is not True
    ):
        raise ValueError("gate-v4 split/OOF contract differs")
    algorithms = _mapping(cfg["algorithms"], "algorithms")
    if (
        algorithms.get("feature_algorithm") != "terminal_gate_40d_v1_pure_math"
        or algorithms.get("trainer_algorithm") != "dual_class_balanced_logistic_v2_pure_math"
        or algorithms.get("quality_target") != "candidate_max_gt_iou_strict_gt_0.25"
        or algorithms.get("benefit_target")
        != "same_best_gt_and_same_gt_iou_gain_ge_0.05"
        or algorithms.get("legacy_artifact_reuse") is not False
    ):
        raise ValueError("gate-v4 algorithm/target contract differs")
    material = _mapping(cfg["materialization"], "materialization")
    for key, expected in {
        "geometry_only": True,
        "preserve_anchor_scores": True,
        "preserve_row_order": True,
        "preserve_row_count": True,
        "clip_semantics_unchanged": True,
        "canonical103_authorized": False,
        "training_materialization_uses_oof_anchor_scores": True,
    }.items():
        if material.get(key) is not expected:
            raise ValueError(f"gate-v4 materialization field {key} differs")
    forbidden = _mapping(cfg["forbidden_reuse"], "forbidden_reuse")
    if (
        forbidden.get("scannet_artifact_access") is not False
        or forbidden.get("old_ca_terminal_v1_v2_v3_access") is not False
        or forbidden.get("old_ca_b6_access") is not False
        or forbidden.get("old_ca_benefit_dataset_or_policy_access") is not False
        or tuple(forbidden.get("path_tokens", ())) != _OLD_TOKENS
    ):
        raise ValueError("gate-v4 forbidden-reuse contract differs")
    _assert_no_old_path(cfg["prerequisites"], "prerequisites")
    _assert_no_old_path(cfg["outputs"], "outputs")
    return source, cfg


def preregistration_science_contract() -> dict[str, Any]:
    """Return the single-source, JSON-safe frozen fit/dev science contract."""

    return {
        "threshold_grid": list(THRESHOLD_GRID),
        "dev_gate": dict(DEV_GATE),
        "max_replacements_per_scene": MAX_REPLACEMENTS,
        "objective_fields": list(OBJECTIVE_FIELDS),
        "fit_protocol": dict(FIT_PROTOCOL),
        "ap_protocol": dict(AP_PROTOCOL),
        "score_tie_protocol": dict(TIE_PROTOCOL),
        "failure_action": FAILURE_ACTION,
    }


def preregistration_code_records() -> dict[str, dict[str, str]]:
    """Return hashes for every executable that can affect the GT join/gate."""

    root = Path(__file__).resolve().parents[1]
    sources = {
        "gate_module": Path(__file__).resolve(),
        "dataset_builder": root / "tools/build_ca1m_tr3d_benefit_dataset_v4.py",
        "trainer": root / "tools/train_ca1m_tr3d_benefit_gate_v4.py",
    }
    records: dict[str, dict[str, str]] = {}
    for name, path in sources.items():
        source = _regular_file(path, f"preregistered {name}", immutable=False)
        records[name] = {"path": str(source), "sha256": sha256_file(source)}
    return records


def preregistration_upstream_records(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the GT-free upstream identities recorded by preregistration."""

    prerequisites = _mapping(cfg.get("prerequisites"), "prerequisites")

    def artifact(value: Any, name: str) -> dict[str, str]:
        record = _mapping(value, name)
        source = _regular_file(Path(str(record.get("path", ""))), name)
        digest = _sha(record.get("sha256"), f"{name} SHA256")
        schema = str(record.get("schema", ""))
        if digest != sha256_file(source) or not schema:
            raise ValueError(f"{name} path/SHA/schema binding differs")
        return {"path": str(source), "sha256": digest, "schema": schema}

    scene_path, scenes = _scene_list(_mapping(cfg.get("scene_contract"), "scene contract"))
    proposal = _mapping(prerequisites.get("proposal_stage_p"), "proposal stage P")
    overlay = _mapping(prerequisites.get("overlay_stage_o"), "overlay stage O")
    outputs = _mapping(cfg.get("outputs"), "outputs")
    return {
        "scene_contract": {
            "path": str(scene_path), "sha256": sha256_file(scene_path),
            "count": len(scenes),
        },
        "final_base_manifest": artifact(
            prerequisites.get("final_base_manifest"), "final-base manifest"
        ),
        "native_b6_v2_collection_manifest": artifact(
            prerequisites.get("native_b6_v2_collection_manifest"),
            "native-B6 v2 collection manifest",
        ),
        "native_b6_v2_checkpoint": artifact(
            prerequisites.get("native_b6_v2_checkpoint"), "native-B6 v2 checkpoint"
        ),
        "native_b6_v2_checkpoint_manifest": artifact(
            prerequisites.get("native_b6_v2_checkpoint_manifest"),
            "native-B6 v2 checkpoint manifest",
        ),
        "native_b6_v2_oof_row_scores": artifact(
            prerequisites.get("native_b6_v2_oof_row_scores"),
            "native-B6 v2 OOF row scores",
        ),
        "native_b6_v2_oof_row_scores_manifest": artifact(
            prerequisites.get("native_b6_v2_oof_row_scores_manifest"),
            "native-B6 v2 OOF row scores manifest",
        ),
        "proposal_collection_manifest": artifact(
            proposal.get("collection_manifest"), "proposal collection manifest"
        ),
        "overlay_collection_manifest": artifact(
            overlay.get("collection_manifest"), "overlay collection manifest"
        ),
        "candidate_evidence_manifest": artifact(
            prerequisites.get("candidate_evidence_manifest"),
            "candidate evidence manifest",
        ),
        "derived_train_gt_inventory_receipt": artifact(
            prerequisites.get("derived_train_gt_inventory_receipt"),
            "derived train GT shadow inventory receipt",
        ),
        "proposal_root": str(Path(str(proposal.get("root", ""))).resolve()),
        "overlay_root": str(Path(str(overlay.get("root", ""))).resolve()),
        "candidate_evidence_root": str(Path(str(
            outputs.get("candidate_evidence_root", "")
        )).resolve()),
        # This records only the authorized CA-train GT root name.  The sealer
        # neither lists it nor opens any scene/target file.
        "derived_train_gt_root": str(Path(str(
            prerequisites.get("derived_train_gt_root", "")
        )).resolve()),
    }


def validate_preregistration_record(value: Any) -> tuple[Path, dict[str, Any]]:
    """Validate the sealed preregistration before any train GT is opened."""

    path, payload = _record(
        value, "terminal gate v4 preregistration", PREREGISTRATION_SCHEMA
    )
    required = {
        "schema", "complete", "train_only", "sealed_before_first_gt_join",
        "ground_truth_access", "validation_ground_truth_access",
        "validation_prediction_access", "official_validation_comparable",
        "locked_internal_fold1_gt_access", "fit_fold_ids",
        "threshold_dev_fold_ids", "locked_internal_fold_ids",
        "anchor_score_source", "deploy_b6_scores_used_for_stacked_training",
        "feature_schema", "feature_names", "quality_target", "benefit_target",
        "selection_rule", "science", "code", "upstream",
    }
    if set(payload) != required:
        raise ValueError("terminal gate v4 preregistration keys differ")
    if (
        payload.get("complete") is not True
        or payload.get("train_only") is not True
        or payload.get("sealed_before_first_gt_join") is not True
        or payload.get("ground_truth_access") is not False
        or payload.get("validation_ground_truth_access") is not False
        or payload.get("validation_prediction_access") is not False
        or payload.get("official_validation_comparable") is not False
        or payload.get("locked_internal_fold1_gt_access") is not False
        or payload.get("fit_fold_ids") != list(GATE_TRAIN_FOLDS)
        or payload.get("threshold_dev_fold_ids") != list(THRESHOLD_DEV_FOLDS)
        or payload.get("locked_internal_fold_ids") != list(LOCKED_INTERNAL_FOLDS)
        or payload.get("anchor_score_source")
        != "ca1m_native_b6_final_base_all_fold_oof_row_scores_v2"
        or payload.get("deploy_b6_scores_used_for_stacked_training") is not False
        or payload.get("feature_schema") != FEATURE_SCHEMA
        or payload.get("feature_names") != list(FEATURE_NAMES)
        or payload.get("quality_target") != QUALITY_TARGET
        or payload.get("benefit_target") != BENEFIT_TARGET
        or payload.get("selection_rule") != SELECTION_RULE
        or payload.get("science") != preregistration_science_contract()
        or payload.get("code") != preregistration_code_records()
        or not isinstance(payload.get("upstream"), Mapping)
    ):
        raise ValueError("terminal gate v4 preregistration contract/code drifted")
    return path, payload


def load_oof_row_scores(
    sidecar_record: Any,
    manifest_record: Any,
    *,
    checkpoint: Path,
    checkpoint_manifest: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sidecar = _mapping(sidecar_record, "B6 OOF sidecar")
    source = _regular_file(Path(str(sidecar.get("path", ""))), "B6 OOF sidecar")
    if (
        sidecar.get("schema") != B6_OOF_SCHEMA
        or _sha(sidecar.get("sha256"), "B6 OOF sidecar SHA256") != sha256_file(source)
    ):
        raise ValueError("B6 OOF sidecar binding differs")
    manifest_path, manifest = _record(
        manifest_record, "B6 OOF manifest", B6_OOF_MANIFEST_SCHEMA
    )
    for key, expected in {
        "complete": True, "train_only": True, "scene_group_oof": True,
        "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "official_validation_comparable": False,
        "each_row_model_excludes_scene": True,
        "scene_count": 100,
    }.items():
        if manifest.get(key) != expected:
            raise ValueError(f"B6 OOF manifest field {key} differs")
    artifact = manifest.get("artifact") or {}
    deployment = manifest.get("deployment_checkpoint") or {}
    reverse = manifest.get("checkpoint_manifest") or {}
    if (
        Path(str(artifact.get("path", ""))).resolve() != source
        or artifact.get("sha256") != sha256_file(source)
        or artifact.get("schema") != B6_OOF_SCHEMA
        or Path(str(deployment.get("path", ""))).resolve() != checkpoint
        or deployment.get("sha256") != sha256_file(checkpoint)
        or deployment.get("schema") != B6_CHECKPOINT_SCHEMA
        or Path(str(reverse.get("path", ""))).resolve() != checkpoint_manifest
        or reverse.get("schema") != B6_CHECKPOINT_MANIFEST_SCHEMA
        or reverse.get("binds_this_sidecar") is not True
    ):
        raise ValueError("B6 OOF sidecar/checkpoint reverse binding differs")
    split = manifest.get("split") or {}
    folds = split.get("folds") or []
    if (
        split.get("namespace") != SPLIT_NAMESPACE
        or split.get("fold_count") != 5
        or split.get("all_fold_oof") is not True
        or tuple(split.get("gate_train_folds", ())) != GATE_TRAIN_FOLDS
        or tuple(split.get("threshold_dev_folds", ())) != THRESHOLD_DEV_FOLDS
        or tuple(split.get("locked_internal_check_folds", ())) != LOCKED_INTERNAL_FOLDS
        or len(folds) != 5
    ):
        raise ValueError("B6 OOF manifest split differs")
    scene_roles: dict[int, set[str]] = {}
    for fold, row in enumerate(folds):
        if (
            row.get("heldout_fold") != fold
            or row.get("training_excludes_every_heldout_scene") is not True
            or set(row.get("heldout_scene_ids", ())) & set(row.get("training_scene_ids", ()))
        ):
            raise ValueError("B6 OOF fold leakage proof differs")
        scene_roles[fold] = set(str(x) for x in row.get("heldout_scene_ids", ()))
    if (
        sum(len(scene_roles[fold]) for fold in GATE_TRAIN_FOLDS) != TRAIN_SCENE_COUNT
        or sum(len(scene_roles[fold]) for fold in THRESHOLD_DEV_FOLDS) != DEV_SCENE_COUNT
        or sum(len(scene_roles[fold]) for fold in LOCKED_INTERNAL_FOLDS) != LOCKED_SCENE_COUNT
        or len(set().union(*scene_roles.values())) != 100
    ):
        raise ValueError("B6 OOF scene-role counts differ from 60/20/20")
    with np.load(source, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    required = {
        "schema", "complete", "train_only", "scene_group_oof",
        "validation_ground_truth_access", "validation_prediction_access",
        "official_validation_comparable", "each_row_model_excludes_scene",
        "fold_count", "dataset_sha256", "dataset_manifest_sha256",
        "split_namespace", "feature_names", "scene_ids", "fold_ids",
        "heldout_model_fold_ids", "source_row_indices", "dataset_row_positions",
        "detector_scores", "raw_oof_outputs", "monotonic_oof_components",
        "quality_oof_scores", "deployment_blend_oof_scores",
        "fold_model_sha256", "recipe_json", "recipe_sha256",
    }
    if set(values) != required:
        raise ValueError("B6 OOF sidecar key set differs")
    scalars = {
        "schema": B6_OOF_SCHEMA, "complete": True, "train_only": True,
        "scene_group_oof": True, "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "official_validation_comparable": False,
        "each_row_model_excludes_scene": True, "fold_count": 5,
        "split_namespace": SPLIT_NAMESPACE,
    }
    for name, expected in scalars.items():
        value = np.asarray(values[name])
        if value.shape != () or value.item() != expected:
            raise ValueError(f"B6 OOF sidecar scalar {name} differs")
    scenes = np.asarray(values["scene_ids"])
    observed_folds = np.asarray(values["fold_ids"])
    heldout_folds = np.asarray(values["heldout_model_fold_ids"])
    rows = np.asarray(values["source_row_indices"])
    positions = np.asarray(values["dataset_row_positions"])
    scores = np.asarray(values["deployment_blend_oof_scores"])
    count = len(scenes)
    if (
        count != int(manifest.get("row_count", -1))
        or observed_folds.shape != (count,)
        or heldout_folds.shape != (count,)
        or rows.shape != (count,)
        or scores.shape != (count,)
        or not np.array_equal(observed_folds, heldout_folds)
        or not np.array_equal(positions, np.arange(count, dtype=np.int64))
        or not np.isfinite(scores).all()
        or np.any((scores < 0.0) | (scores > 1.0))
    ):
        raise ValueError("B6 OOF row identity/score arrays differ")
    for name, shape in (
        ("detector_scores", (count,)),
        ("quality_oof_scores", (count,)),
        ("raw_oof_outputs", (count, 4)),
        ("monotonic_oof_components", (count, 4)),
    ):
        value = np.asarray(values[name])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"B6 OOF sidecar array {name} differs")
    for scene in np.unique(scenes):
        scene_rows = np.flatnonzero(scenes == scene)
        if not np.array_equal(rows[scene_rows], np.arange(len(scene_rows), dtype=np.int64)):
            raise ValueError("B6 OOF source rows are not contiguous per scene")
    fold_hashes = tuple(str(value) for value in values["fold_model_sha256"].tolist())
    if (
        len(fold_hashes) != 5
        or any(_SHA.fullmatch(value) is None for value in fold_hashes)
        or fold_hashes != tuple(str(value) for value in manifest.get("fold_model_sha256", ()))
    ):
        raise ValueError("B6 OOF fold-model hashes differ")
    recipe_json = str(np.asarray(values["recipe_json"]).item())
    recipe_sha = str(np.asarray(values["recipe_sha256"]).item())
    try:
        recipe = json.loads(recipe_json)
    except json.JSONDecodeError as error:
        raise ValueError("B6 OOF recipe is not JSON") from error
    if (
        json.dumps(recipe, separators=(",", ":"), sort_keys=True) != recipe_json
        or hashlib.sha256(recipe_json.encode()).hexdigest() != recipe_sha
        or manifest.get("recipe_sha256") != recipe_sha
        or manifest.get("recipe") != recipe
    ):
        raise ValueError("B6 OOF recipe binding differs")
    dataset_record = manifest.get("dataset") or {}
    if (
        dataset_record.get("sha256") != str(np.asarray(values["dataset_sha256"]).item())
        or dataset_record.get("manifest_sha256")
        != str(np.asarray(values["dataset_manifest_sha256"]).item())
    ):
        raise ValueError("B6 OOF dataset binding differs")
    for scene, fold in zip(scenes.tolist(), observed_folds.tolist()):
        if str(scene) not in scene_roles[int(fold)]:
            raise ValueError("B6 OOF row scene is not held out by its scoring model")
    return values, manifest


def _inventory(
    root: Path,
    scenes: tuple[str, ...],
    *,
    kind: str,
    binding_sha256: str | None = None,
) -> dict[str, Any]:
    directory = _regular_directory(root, f"terminal-v4 {kind} root")
    if kind == "proposal":
        names = {f"{scene}_ca1m_tr3d_proposals_v4.npz": scene for scene in scenes}
    elif kind == "overlay":
        names = {f"{scene}_ca1m_tr3d_overlay_v4.npz": scene for scene in scenes}
    else:
        raise ValueError("unknown v4 inventory kind")
    actual = {
        item.name for item in directory.iterdir()
        if item.is_file() and not item.is_symlink() and item.suffix == ".npz"
    }
    if actual != set(names):
        raise ValueError(f"terminal-v4 {kind} root is not exact100")
    hashes: dict[str, str] = {}
    for name, scene in names.items():
        path = directory / name
        if kind == "proposal":
            load_proposal_cache(
                path, expected_scene=scene, expected_binding_sha256=binding_sha256
            )
        else:
            load_overlay_cache(path, expected_scene=scene)
        hashes[scene] = sha256_file(path)
    return {"root": str(directory), "scene_count": 100, "sha256": hashes}


def validate_ready(path: Path) -> dict[str, Any]:
    config_path, cfg = validate_static_config(path)
    if cfg["run_authorized"] is not True or cfg["state"] != "ready_after_all_train100_seals":
        raise PermissionError("gate-v4 prerequisites/run authorization are still pending")
    if config_path.stat().st_mode & 0o222:
        raise ValueError("ready gate-v4 config must be sealed read-only")
    _, scenes = _scene_list(_mapping(cfg["scene_contract"], "scene_contract"))
    prerequisites = _mapping(cfg["prerequisites"], "prerequisites")
    preregistration_path, preregistration = validate_preregistration_record(
        prerequisites.get("preregistration_manifest")
    )
    if preregistration.get("upstream") != preregistration_upstream_records(cfg):
        raise ValueError("terminal gate v4 preregistration upstream identity drifted")
    final_path, final = _record(
        prerequisites.get("final_base_manifest"), "final-base manifest", FINAL_BASE_SCHEMA
    )
    for key, expected in {
        "ok": True, "dataset": "CA1M", "split": "train100", "scene_count": 100,
        "ground_truth_access": False, "evaluation_invoked": False,
        "training_invoked": False, "scannet_learned_b6_or_gate_reused": False,
        "clip_appearance_gate_active": True, "reliable_view_top_k": 3,
    }.items():
        if final.get(key) != expected:
            raise ValueError(f"final-base manifest field {key} differs")
    if set((final.get("per_scene") or {})) != set(scenes):
        raise ValueError("final-base manifest does not cover exact train100")
    b6_collection_path, b6_collection = _record(
        prerequisites.get("native_b6_v2_collection_manifest"),
        "native-B6 v2 collection", B6_COLLECTION_SCHEMA,
    )
    for key, expected in {
        "complete": True, "train_only": True, "scene_count": 100,
        "evaluation_invoked": False, "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "old_native_b6_diagnostics_reused": False,
        "old_native_b6_checkpoint_reused": False,
    }.items():
        if b6_collection.get(key) != expected:
            raise ValueError(f"native-B6 v2 collection field {key} differs")
    b6_checkpoint_record = _mapping(
        prerequisites.get("native_b6_v2_checkpoint"), "native-B6 v2 checkpoint"
    )
    b6_checkpoint = _regular_file(
        Path(str(b6_checkpoint_record.get("path", ""))), "native-B6 v2 checkpoint"
    )
    if (
        b6_checkpoint_record.get("schema") != B6_CHECKPOINT_SCHEMA
        or _sha(b6_checkpoint_record.get("sha256"), "B6 checkpoint SHA256")
        != sha256_file(b6_checkpoint)
    ):
        raise ValueError("native-B6 v2 checkpoint binding differs")
    b6_manifest_path, b6_manifest = _record(
        prerequisites.get("native_b6_v2_checkpoint_manifest"),
        "native-B6 v2 checkpoint manifest", B6_CHECKPOINT_MANIFEST_SCHEMA,
    )
    if (
        b6_manifest.get("complete") is not True
        or b6_manifest.get("train_only") is not True
        or b6_manifest.get("activation_authorized") is not True
        or b6_manifest.get("validation_ground_truth_access") is not False
        or b6_manifest.get("validation_prediction_access") is not False
        or (b6_manifest.get("checkpoint") or {}).get("sha256") != sha256_file(b6_checkpoint)
    ):
        raise ValueError("native-B6 v2 checkpoint manifest is not authorized")
    oof_values, oof_manifest = load_oof_row_scores(
        prerequisites.get("native_b6_v2_oof_row_scores"),
        prerequisites.get("native_b6_v2_oof_row_scores_manifest"),
        checkpoint=b6_checkpoint,
        checkpoint_manifest=b6_manifest_path,
    )
    manifest_oof = b6_manifest.get("all_fold_oof_row_scores") or {}
    configured_oof = _mapping(
        prerequisites.get("native_b6_v2_oof_row_scores"), "configured B6 OOF"
    )
    configured_oof_manifest = _mapping(
        prerequisites.get("native_b6_v2_oof_row_scores_manifest"),
        "configured B6 OOF manifest",
    )
    if (
        Path(str(manifest_oof.get("path", ""))).resolve()
        != Path(str(configured_oof.get("path", ""))).resolve()
        or manifest_oof.get("sha256") != configured_oof.get("sha256")
        or Path(str(manifest_oof.get("manifest_path", ""))).resolve()
        != Path(str(configured_oof_manifest.get("path", ""))).resolve()
        or manifest_oof.get("manifest_sha256") != configured_oof_manifest.get("sha256")
        or manifest_oof.get("checkpoint_manifest_binds_sidecar") is not True
        or manifest_oof.get("sidecar_manifest_binds_checkpoint") is not True
    ):
        raise ValueError("B6 checkpoint manifest does not bind the OOF sidecar both ways")
    proposal = _mapping(prerequisites.get("proposal_stage_p"), "proposal stage P")
    if (
        proposal.get("schema") != PROPOSAL_SCHEMA
        or proposal.get("exact_output_count") != 100
    ):
        raise ValueError("proposal stage P static contract differs")
    proposal_manifest_path, proposal_manifest = _record(
        proposal.get("collection_manifest"),
        "proposal stage-P collection manifest",
        PROPOSAL_COLLECTION_SCHEMA,
    )
    if (
        proposal_manifest.get("complete") is not True
        or proposal_manifest.get("stage") != "P"
        or proposal_manifest.get("scene_count") != 100
        or proposal_manifest.get("ground_truth_access") is not False
        or proposal_manifest.get("validation_ground_truth_access") is not False
        or proposal_manifest.get("anchor_access") is not False
        or proposal_manifest.get("b6_access") is not False
        or (proposal_manifest.get("scene_list") or {}).get("sha256")
        != sha256_file(_scene_list(_mapping(cfg["scene_contract"], "scene_contract"))[0])
    ):
        raise ValueError("proposal stage-P collection manifest contract differs")
    proposal_inventory = _inventory(
        Path(str(proposal.get("root", ""))), scenes, kind="proposal",
        binding_sha256=_sha(proposal.get("ca_scratch_binding_sha256"), "CA TR3D binding SHA256"),
    )
    proposal_rows = {
        str(row.get("scene_id")): row
        for row in proposal_manifest.get("scenes", ())
        if isinstance(row, Mapping)
    }
    if set(proposal_rows) != set(scenes):
        raise ValueError("proposal stage-P collection manifest is not exact100")
    for scene in scenes:
        row = proposal_rows[scene]
        if (
            row.get("sha256") != proposal_inventory["sha256"][scene]
            or Path(str(row.get("path", ""))).resolve()
            != (Path(proposal_inventory["root"]) / f"{scene}_ca1m_tr3d_proposals_v4.npz")
        ):
            raise ValueError(f"{scene}: proposal collection/inventory identity differs")
    overlay = _mapping(prerequisites.get("overlay_stage_o"), "overlay stage O")
    if (
        overlay.get("schema") != OVERLAY_SCHEMA
        or overlay.get("exact_output_count") != 100
    ):
        raise ValueError("overlay stage O static contract differs")
    overlay_manifest_path, overlay_manifest = _record(
        overlay.get("collection_manifest"),
        "overlay stage-O collection manifest",
        OVERLAY_COLLECTION_SCHEMA,
    )
    overlay_upstream = _mapping(overlay_manifest.get("upstream"), "overlay upstream")
    expected_overlay_upstream = {
        "final_base_manifest": final_path,
        "native_b6_v2_collection_manifest": b6_collection_path,
        "native_b6_v2_deployment_checkpoint": b6_checkpoint,
        "native_b6_v2_deployment_checkpoint_manifest": b6_manifest_path,
        "native_b6_v2_oof_row_scores": Path(str(configured_oof["path"])).resolve(),
        "native_b6_v2_oof_row_scores_manifest": Path(
            str(configured_oof_manifest["path"])
        ).resolve(),
        "proposal_collection": proposal_manifest_path,
    }
    if (
        overlay_manifest.get("complete") is not True
        or overlay_manifest.get("stage") != "O"
        or overlay_manifest.get("scene_count") != 100
        or overlay_manifest.get("cpu_only") is not True
        or overlay_manifest.get("ground_truth_access") is not False
        or overlay_manifest.get("validation_ground_truth_access") is not False
        or (overlay_manifest.get("scene_list") or {}).get("sha256")
        != sha256_file(_scene_list(_mapping(cfg["scene_contract"], "scene_contract"))[0])
        or (overlay_manifest.get("score_roles") or {}).get(
            "deployment_scores_allowed_for_stacked_gate_training"
        ) is not False
        or (overlay_manifest.get("score_roles") or {}).get(
            "stacked_gate_training_score_source"
        ) != "all_fold_oof_row_scores_v2"
    ):
        raise ValueError("overlay stage-O collection manifest contract differs")
    for name, source in expected_overlay_upstream.items():
        record = _mapping(overlay_upstream.get(name), f"overlay upstream {name}")
        if (
            Path(str(record.get("path", ""))).resolve() != source.resolve()
            or record.get("sha256") != sha256_file(source)
        ):
            raise ValueError(f"overlay upstream {name} identity differs")
    overlay_inventory = _inventory(
        Path(str(overlay.get("root", ""))), scenes, kind="overlay"
    )
    overlay_rows = {
        str(row.get("scene_id")): row
        for row in overlay_manifest.get("scenes", ())
        if isinstance(row, Mapping)
    }
    if set(overlay_rows) != set(scenes):
        raise ValueError("overlay stage-O collection manifest is not exact100")
    for scene in scenes:
        row = overlay_rows[scene]
        if (
            row.get("sha256") != overlay_inventory["sha256"][scene]
            or row.get("proposal_sha256") != proposal_inventory["sha256"][scene]
            or Path(str(row.get("path", ""))).resolve()
            != (Path(overlay_inventory["root"]) / f"{scene}_ca1m_tr3d_overlay_v4.npz")
        ):
            raise ValueError(f"{scene}: overlay collection/inventory identity differs")
    for scene in scenes:
        loaded = load_overlay_cache(
            Path(overlay_inventory["root"]) / f"{scene}_ca1m_tr3d_overlay_v4.npz",
            expected_scene=scene,
            expected_proposal_sha256=proposal_inventory["sha256"][scene],
        )
        summary = loaded["summary"]
        if (
            summary.final_anchor_manifest_sha256 != sha256_file(final_path)
            or summary.native_b6_collection_manifest_sha256
            != sha256_file(b6_collection_path)
            or summary.native_b6_checkpoint_sha256 != sha256_file(b6_checkpoint)
            or summary.native_b6_checkpoint_manifest_sha256
            != sha256_file(b6_manifest_path)
        ):
            raise ValueError(f"{scene}: overlay is not bound to final-base/B6-v2 chain")
    evidence_path, evidence = _record(
        prerequisites.get("candidate_evidence_manifest"),
        "candidate-evidence v4 manifest", CANDIDATE_EVIDENCE_MANIFEST_SCHEMA,
    )
    for key, expected in {
        "complete": True, "train_only": True, "scene_count": 100,
        "ground_truth_access": False, "validation_ground_truth_access": False,
        "source_proposal_schema": PROPOSAL_SCHEMA,
        "source_overlay_schema": OVERLAY_SCHEMA,
        "native_b6_evidence_top_k": 5,
        "old_candidate_evidence_reused": False,
    }.items():
        if evidence.get(key) != expected:
            raise ValueError(f"candidate-evidence v4 manifest field {key} differs")
    evidence_rows = evidence.get("scenes") or {}
    if set(evidence_rows) != set(scenes):
        raise ValueError("candidate-evidence v4 manifest is not exact100")
    evidence_root = _regular_directory(
        Path(str(_mapping(evidence.get("source_roots"), "candidate evidence roots").get(
            "evidence", ""
        ))),
        "candidate-evidence v4 root",
    )
    configured_evidence_root = Path(str(
        _mapping(cfg["outputs"], "outputs").get("candidate_evidence_root", "")
    )).resolve()
    if (
        evidence_root != configured_evidence_root
        or Path(str(_mapping(evidence["source_roots"], "candidate evidence roots").get(
            "proposal", ""
        ))).resolve() != Path(proposal_inventory["root"])
        or Path(str(_mapping(evidence["source_roots"], "candidate evidence roots").get(
            "overlay", ""
        ))).resolve() != Path(overlay_inventory["root"])
        or evidence.get("shared_final_base_b6_v2_lineage") != [
            sha256_file(final_path), sha256_file(b6_collection_path),
            sha256_file(b6_checkpoint), sha256_file(b6_manifest_path),
        ]
    ):
        raise ValueError("candidate-evidence v4 source lineage differs")
    candidate_total = 0
    valid_total = 0
    for scene in scenes:
        row = evidence_rows[scene]
        evidence_file = validate_candidate_evidence_artifact(
            Path(str(row.get("path", ""))),
            scene=scene,
            expected_sha256=str(row.get("sha256", "")),
            expected_root=evidence_root,
        )
        if (
            row.get("proposal_sha256") != proposal_inventory["sha256"][scene]
            or row.get("overlay_sha256") != overlay_inventory["sha256"][scene]
            or row.get("final_anchor_manifest_sha256") != sha256_file(final_path)
            or row.get("native_b6_collection_manifest_sha256")
            != sha256_file(b6_collection_path)
            or row.get("native_b6_checkpoint_sha256") != sha256_file(b6_checkpoint)
            or row.get("native_b6_checkpoint_manifest_sha256")
            != sha256_file(b6_manifest_path)
        ):
            raise ValueError(f"{scene}: candidate evidence lacks P/O binding")
        candidate_total += int(row.get("candidate_rows", -1))
        valid_total += int(row.get("valid_evidence_rows", -1))
    if (
        candidate_total != int(evidence.get("candidate_rows", -1))
        or valid_total != int(evidence.get("valid_evidence_rows", -1))
        or candidate_total != int((proposal_manifest.get("totals") or {}).get(
            "candidates", -1
        ))
        or candidate_total != int((overlay_manifest.get("totals") or {}).get(
            "candidates", -1
        ))
    ):
        raise ValueError("candidate-evidence v4 collection totals differ")
    # The GT root is CA train-derived only and is opened only by the later
    # dataset stage, after the GT-free P/O/evidence seals above are verified.
    train_gt_root = _regular_directory(
        Path(str(prerequisites.get("derived_train_gt_root", ""))),
        "CA train-derived GT root",
    )
    gt_inventory_path, gt_inventory = _record(
        prerequisites.get("derived_train_gt_inventory_receipt"),
        "derived train GT shadow inventory",
        GT_SHADOW_INVENTORY_SCHEMA,
    )
    source_dataset_record = _mapping(
        b6_manifest.get("dataset"), "B6-v2 checkpoint source dataset"
    )
    inventory_source_dataset = _mapping(
        gt_inventory.get("source_dataset_manifest"),
        "GT shadow source dataset manifest",
    )
    inventory_oof = _mapping(gt_inventory.get("oof_sidecar"), "GT shadow OOF sidecar")
    configured_oof_path = Path(str(configured_oof.get("path", ""))).resolve()
    if (
        gt_inventory.get("complete") is not True
        or gt_inventory.get("create_only") is not True
        or gt_inventory.get("train_only") is not True
        or gt_inventory.get("scene_count") != 80
        or gt_inventory.get("file_count") != 160
        or gt_inventory.get("fit_fold_ids") != list(GATE_TRAIN_FOLDS)
        or gt_inventory.get("threshold_dev_fold_ids") != list(THRESHOLD_DEV_FOLDS)
        or gt_inventory.get("locked_internal_fold_ids") != list(LOCKED_INTERNAL_FOLDS)
        or gt_inventory.get("fit_scene_count") != TRAIN_SCENE_COUNT
        or gt_inventory.get("threshold_dev_scene_count") != DEV_SCENE_COUNT
        or gt_inventory.get("locked_internal_scene_count_accessed") != 0
        or gt_inventory.get("validation_ground_truth_access") is not False
        or gt_inventory.get("validation_prediction_access") is not False
        or gt_inventory.get("gt_array_content_loaded") is not False
        or gt_inventory.get("opaque_source_bytes_hashed_and_copied") is not True
        or gt_inventory.get("source_bytes_mutated") is not False
        or gt_inventory.get("shadow_files_read_only") is not True
        or Path(str(gt_inventory.get("output_root", ""))).resolve() != train_gt_root
        or Path(str(inventory_source_dataset.get("path", ""))).resolve()
        != Path(str(source_dataset_record.get("manifest_path", ""))).resolve()
        or inventory_source_dataset.get("sha256")
        != source_dataset_record.get("manifest_sha256")
        or Path(str(inventory_oof.get("path", ""))).resolve() != configured_oof_path
        or inventory_oof.get("sha256") != sha256_file(configured_oof_path)
    ):
        raise ValueError("derived train GT shadow inventory contract differs")
    oof_scene_ids = np.asarray(oof_values["scene_ids"]).astype(str)
    oof_fold_ids = np.asarray(oof_values["fold_ids"], dtype=np.int64)
    scene_fold: dict[str, int] = {}
    for scene in sorted(set(oof_scene_ids.tolist())):
        values = np.unique(oof_fold_ids[oof_scene_ids == scene])
        if len(values) != 1:
            raise ValueError(f"B6-v2 OOF scene crosses folds: {scene}")
        scene_fold[scene] = int(values[0])
    selected_gt_scenes = {
        scene for scene, fold in scene_fold.items()
        if fold in (*GATE_TRAIN_FOLDS, *THRESHOLD_DEV_FOLDS)
    }
    gt_inventory_rows = _mapping(gt_inventory.get("scenes"), "GT shadow scenes")
    if (
        len(selected_gt_scenes) != 80
        or set(gt_inventory_rows) != selected_gt_scenes
        or any(scene_fold[scene] in LOCKED_INTERNAL_FOLDS for scene in gt_inventory_rows)
    ):
        raise ValueError("derived train GT shadow inventory split differs")
    expected_gt_files: set[str] = set()
    for scene in sorted(selected_gt_scenes):
        row = _mapping(gt_inventory_rows[scene], f"GT shadow scene {scene}")
        box_record = _mapping(row.get("box"), f"GT shadow box {scene}")
        manifest_record = _mapping(row.get("manifest"), f"GT shadow manifest {scene}")
        box_path = _regular_file(
            Path(str(box_record.get("path", ""))), f"GT shadow box {scene}"
        )
        scene_manifest_path = _regular_file(
            Path(str(manifest_record.get("path", ""))),
            f"GT shadow manifest {scene}",
        )
        if (
            row.get("fold_id") != scene_fold[scene]
            or box_path != train_gt_root / scene / "derived_train_gt_boxes.npy"
            or scene_manifest_path
            != train_gt_root / scene / "derived_train_gt_manifest.json"
            or box_record.get("mode") != "0o444"
            or manifest_record.get("mode") != "0o444"
            or box_record.get("sha256") != sha256_file(box_path)
            or manifest_record.get("sha256") != sha256_file(scene_manifest_path)
        ):
            raise ValueError(f"derived train GT shadow identity differs: {scene}")
        expected_gt_files.update({
            f"{scene}/derived_train_gt_boxes.npy",
            f"{scene}/derived_train_gt_manifest.json",
        })
    actual_gt_files = {
        str(item.relative_to(train_gt_root))
        for item in train_gt_root.rglob("*")
        if item.is_file() and not item.is_symlink()
    }
    if actual_gt_files != expected_gt_files:
        raise ValueError("derived train GT shadow root is not exact fit/dev80")
    outputs = _mapping(cfg["outputs"], "outputs")
    binding_output = Path(str(outputs.get("binding_manifest", ""))).resolve()
    if binding_output.exists() or binding_output.is_symlink():
        raise FileExistsError("gate-v4 binding output already exists")
    return {
        "schema": BINDING_SCHEMA,
        "complete": True,
        "train_only": True,
        "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "official_validation_comparable": False,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "scene_count": 100,
        "split": dict(cfg["split"]),
        "preregistration_manifest": {
            "path": str(preregistration_path),
            "sha256": sha256_file(preregistration_path),
            "schema": PREREGISTRATION_SCHEMA,
            "sealed_before_first_gt_join": True,
        },
        "final_base_manifest": {"path": str(final_path), "sha256": sha256_file(final_path)},
        "native_b6_v2_collection_manifest": {
            "path": str(b6_collection_path), "sha256": sha256_file(b6_collection_path)
        },
        "native_b6_v2_checkpoint": {
            "path": str(b6_checkpoint), "sha256": sha256_file(b6_checkpoint)
        },
        "native_b6_v2_checkpoint_manifest": {
            "path": str(b6_manifest_path), "sha256": sha256_file(b6_manifest_path)
        },
        "native_b6_v2_oof": {
            "row_count": len(oof_values["scene_ids"]),
            "manifest_sha256": sha256_file(
                Path(str(_mapping(prerequisites["native_b6_v2_oof_row_scores_manifest"], "oof")["path"]))
            ),
            "each_row_model_excludes_scene": oof_manifest["each_row_model_excludes_scene"],
            "deploy_scores_used_for_stacked_training": False,
        },
        "proposal_stage_p": {
            **proposal_inventory,
            "collection_manifest": {
                "path": str(proposal_manifest_path),
                "sha256": sha256_file(proposal_manifest_path),
            },
        },
        "overlay_stage_o": {
            **overlay_inventory,
            "collection_manifest": {
                "path": str(overlay_manifest_path),
                "sha256": sha256_file(overlay_manifest_path),
            },
        },
        "candidate_evidence_manifest": {
            "path": str(evidence_path),
            "sha256": sha256_file(evidence_path),
            "candidate_rows": candidate_total,
            "valid_evidence_rows": valid_total,
        },
        "candidate_evidence_manifest_sha256": sha256_file(evidence_path),
        "derived_train_gt_root": str(train_gt_root),
        "derived_train_gt_inventory_receipt": {
            "path": str(gt_inventory_path),
            "sha256": sha256_file(gt_inventory_path),
            "schema": GT_SHADOW_INVENTORY_SCHEMA,
            "scene_count": 80,
            "file_count": 160,
        },
        "algorithm_only_reuse": dict(cfg["algorithms"]),
        "legacy_artifact_reuse": False,
        "binding_output": str(binding_output),
    }


def write_binding_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing existing gate-v4 binding: {target}")
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    published: tuple[int, int] | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.",
            suffix=".tmp", delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        stat = target.stat(follow_symlinks=False)
        published = (stat.st_dev, stat.st_ino)
        target.chmod(0o444)
    except BaseException:
        if published is not None:
            try:
                current = target.stat(follow_symlinks=False)
                if (
                    not target.is_symlink()
                    and (current.st_dev, current.st_ino) == published
                    and hashlib.sha256(target.read_bytes()).digest()
                    == hashlib.sha256(data).digest()
                ):
                    target.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _readonly(value: Any, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


def _evidence_matrix(
    value: Any,
    *,
    rows: int,
    expected_scores: np.ndarray,
    label: str,
) -> np.ndarray:
    result = np.asarray(value)
    if (
        result.dtype != np.dtype(np.float32)
        or result.shape != (rows, len(NATIVE_FEATURE_NAMES))
        or not np.isfinite(result).all()
        or np.any((result < 0.0) | (result > 1.0))
    ):
        raise ValueError(f"{label} native evidence violates the frozen 14-D schema")
    detector_column = NATIVE_FEATURE_NAMES.index("detector_score")
    if not np.array_equal(
        result[:, detector_column].astype(np.float32), expected_scores
    ):
        raise ValueError(f"{label} native detector score differs from its score source")
    return result


@dataclass(frozen=True)
class TerminalGateFeatureBatchV4:
    """GT-free, row-aligned 40-D features for the isolated v4 route."""

    schema: str
    scene_id: str
    score_source: str
    candidate_rows: np.ndarray
    anchor_indices: np.ndarray
    candidate_scores: np.ndarray
    features: np.ndarray


def build_terminal_gate_features_v4(
    *,
    proposal: Mapping[str, Any],
    overlay: Mapping[str, Any],
    anchor_native_evidence: Any,
    anchor_native_detector_scores: Any,
    candidate_native_evidence: Any,
    anchor_scores: Any,
    score_source: str,
) -> TerminalGateFeatureBatchV4:
    """Build the frozen 40-D features from v4 P/O plus B6-v2 evidence.

    ``anchor_scores`` is deliberately explicit.  Stacked fit/dev callers must
    pass all-fold OOF B6-v2 scores; deployment callers pass scores produced by
    the sealed deployment B6-v2 model.  The native evidence detector column is
    replaced by this explicit source before feature assembly, so deploy scores
    cannot silently leak into stacked training.
    """

    if score_source not in {
        "ca1m_native_b6_final_base_all_fold_oof_row_scores_v2",
        "ca1m_native_b6_final_base_deployment_scores_v2",
    }:
        raise ValueError("terminal gate v4 anchor score source is not authorized")
    proposal_summary = proposal.get("summary")
    overlay_summary = overlay.get("summary")
    if proposal_summary is None or overlay_summary is None:
        raise ValueError("terminal gate v4 requires validated P/O mappings")
    scene = str(getattr(proposal_summary, "scene_id", ""))
    if scene != str(getattr(overlay_summary, "scene_id", "")):
        raise ValueError("terminal gate v4 P/O scene identity differs")

    anchors = np.asarray(overlay.get("anchor_corners"))
    candidates = np.asarray(proposal.get("candidate_corners_world"))
    overlay_candidates = np.asarray(overlay.get("candidate_corners_world"))
    candidate_scores = np.asarray(proposal.get("candidate_scores"))
    overlay_candidate_scores = np.asarray(overlay.get("candidate_scores"))
    support = np.asarray(proposal.get("candidate_point_count"))
    best_anchor = np.asarray(overlay.get("best_anchor_indices"))
    best_iou = np.asarray(overlay.get("best_anchor_iou"))
    best_distance = np.asarray(overlay.get("best_anchor_center_distance_m"))
    near = np.asarray(overlay.get("near_mask"))
    explicit_anchor_scores = np.asarray(anchor_scores)
    anchor_count = len(anchors)
    candidate_count = len(candidates)
    if (
        anchors.dtype != np.dtype(np.float32)
        or anchors.shape != (anchor_count, 8, 3)
        or candidates.dtype != np.dtype(np.float32)
        or candidates.shape != (candidate_count, 8, 3)
        or candidate_scores.dtype != np.dtype(np.float32)
        or candidate_scores.shape != (candidate_count,)
        or support.dtype != np.dtype(np.int64)
        or support.shape != (candidate_count,)
        or best_anchor.dtype != np.dtype(np.int64)
        or best_anchor.shape != (candidate_count,)
        or best_iou.dtype != np.dtype(np.float32)
        or best_iou.shape != (candidate_count,)
        or best_distance.dtype != np.dtype(np.float32)
        or best_distance.shape != (candidate_count,)
        or near.dtype != np.dtype(np.bool_)
        or near.shape != (candidate_count,)
        or explicit_anchor_scores.dtype != np.dtype(np.float32)
        or explicit_anchor_scores.shape != (anchor_count,)
    ):
        raise ValueError("terminal gate v4 P/O feature arrays violate dtype/shape")
    if (
        not np.array_equal(candidates, overlay_candidates)
        or not np.array_equal(candidate_scores, overlay_candidate_scores)
        or not np.isfinite(anchors).all()
        or not np.isfinite(candidates).all()
        or not np.isfinite(candidate_scores).all()
        or not np.isfinite(explicit_anchor_scores).all()
        or np.any((candidate_scores < 0.0) | (candidate_scores > 1.0))
        or np.any((explicit_anchor_scores < 0.0) | (explicit_anchor_scores > 1.0))
        or np.any(support < 0)
    ):
        raise ValueError("terminal gate v4 P/O feature arrays differ or are non-finite")

    # The sealed B6-v2 observer evidence itself is score-agnostic except for
    # column zero.  Validate it against its recorded detector scores, then bind
    # the anchor column to the explicit OOF/deploy source used by this stage.
    recorded_anchor_scores = np.asarray(anchor_native_detector_scores)
    if (
        recorded_anchor_scores.dtype != np.dtype(np.float32)
        or recorded_anchor_scores.shape != (anchor_count,)
        or not np.isfinite(recorded_anchor_scores).all()
        or np.any((recorded_anchor_scores < 0.0) | (recorded_anchor_scores > 1.0))
    ):
        raise ValueError("anchor native detector-score provenance is invalid")
    anchor_native = _evidence_matrix(
        anchor_native_evidence,
        rows=anchor_count,
        expected_scores=recorded_anchor_scores,
        label="anchor",
    ).copy()
    candidate_native = _evidence_matrix(
        candidate_native_evidence,
        rows=candidate_count,
        expected_scores=candidate_scores,
        label="candidate",
    )
    detector_column = NATIVE_FEATURE_NAMES.index("detector_score")
    anchor_native[:, detector_column] = explicit_anchor_scores

    candidate_rows = np.flatnonzero(near).astype(np.int64)
    anchor_rows = best_anchor[candidate_rows]
    if np.any((anchor_rows < 0) | (anchor_rows >= anchor_count)):
        raise ValueError("near terminal-v4 candidate has invalid anchor association")
    if not len(candidate_rows):
        return TerminalGateFeatureBatchV4(
            schema=FEATURE_SCHEMA,
            scene_id=scene,
            score_source=score_source,
            candidate_rows=_readonly(candidate_rows, np.int64),
            anchor_indices=_readonly(anchor_rows, np.int64),
            candidate_scores=_readonly(np.empty((0,), np.float32), np.float32),
            features=_readonly(np.empty((0, len(FEATURE_NAMES))), np.float32),
        )

    anchor_boxes = world_aabb(anchors)
    candidate_boxes = world_aabb(candidates)
    anchor_extent = anchor_boxes[:, 3:] - anchor_boxes[:, :3]
    candidate_extent = candidate_boxes[:, 3:] - candidate_boxes[:, :3]
    anchor_volume = np.prod(anchor_extent, axis=1)
    candidate_volume = np.prod(candidate_extent, axis=1)
    anchor_diagonal = np.linalg.norm(anchor_extent, axis=1)

    order = np.lexsort(
        (np.arange(candidate_count, dtype=np.int64), -candidate_scores.astype(np.float64))
    )
    global_rank = np.empty(candidate_count, dtype=np.int64)
    global_rank[order] = np.arange(candidate_count, dtype=np.int64)
    global_denominator = max(candidate_count - 1, 1)
    group_rank = np.zeros(candidate_count, dtype=np.float64)
    group_size = np.zeros(candidate_count, dtype=np.int64)
    sibling_margin = np.zeros(candidate_count, dtype=np.float64)
    for anchor in np.unique(anchor_rows).tolist():
        rows = candidate_rows[anchor_rows == anchor]
        local_order = np.lexsort((rows, -candidate_scores[rows].astype(np.float64)))
        ordered = rows[local_order]
        denominator = max(len(ordered) - 1, 1)
        for rank, row in enumerate(ordered.tolist()):
            siblings = rows[rows != row]
            group_rank[row] = rank / denominator
            group_size[row] = len(ordered)
            sibling_margin[row] = (
                0.0
                if not len(siblings)
                else float(candidate_scores[row])
                - float(np.max(candidate_scores[siblings]))
            )

    point_count = int(getattr(proposal_summary, "point_count", 0))
    if point_count < 1:
        raise ValueError("terminal gate v4 proposal point count is invalid")
    relation = np.empty((len(candidate_rows), len(RELATION_FEATURE_NAMES)), np.float64)
    for output_row, (candidate, anchor) in enumerate(
        zip(candidate_rows.tolist(), anchor_rows.tolist())
    ):
        extent_ratio = np.maximum(candidate_extent[candidate], 1.0e-12) / np.maximum(
            anchor_extent[anchor], 1.0e-12
        )
        candidate_support = float(support[candidate])
        volume = max(float(candidate_volume[candidate]), 1.0e-12)
        relation[output_row] = (
            float(candidate_scores[candidate] - explicit_anchor_scores[anchor]),
            float(best_iou[candidate]),
            float(best_distance[candidate] / max(float(anchor_diagonal[anchor]), 1.0e-12)),
            math.log(volume / max(float(anchor_volume[anchor]), 1.0e-12)),
            float(np.linalg.norm(np.log(extent_ratio))),
            math.log1p(candidate_support),
            math.log1p(candidate_support / volume),
            candidate_support / point_count,
            float(global_rank[candidate] / global_denominator),
            float(group_rank[candidate]),
            math.log1p(int(group_size[candidate])),
            float(sibling_margin[candidate]),
        )
    features = np.concatenate(
        (anchor_native[anchor_rows], candidate_native[candidate_rows], relation), axis=1
    )
    if features.shape != (len(candidate_rows), len(FEATURE_NAMES)):
        raise RuntimeError("terminal gate v4 feature assembly changed shape")
    if not np.isfinite(features).all():
        raise ValueError("terminal gate v4 features contain non-finite values")
    return TerminalGateFeatureBatchV4(
        schema=FEATURE_SCHEMA,
        scene_id=scene,
        score_source=score_source,
        candidate_rows=_readonly(candidate_rows, np.int64),
        anchor_indices=_readonly(anchor_rows, np.int64),
        candidate_scores=_readonly(candidate_scores[candidate_rows], np.float32),
        features=_readonly(features, np.float32),
    )


def _stable_sigmoid(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value, dtype=np.float64)
    result = np.empty_like(source)
    positive = source >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-np.minimum(source[positive], 700.0)))
    exponential = np.exp(np.maximum(source[~positive], -700.0))
    result[~positive] = exponential / (1.0 + exponential)
    return result


@dataclass(frozen=True)
class LogisticGateHeadV4:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    threshold: float

    def predict(self, features: Any) -> np.ndarray:
        values = np.asarray(features)
        if values.shape != (len(values), len(FEATURE_NAMES)) or not np.isfinite(values).all():
            raise ValueError("terminal gate v4 policy received invalid 40-D features")
        probability = _stable_sigmoid(
            ((values.astype(np.float64) - self.mean) / self.scale) @ self.weights
            + self.bias
        )
        return _readonly(probability, np.float64)


@dataclass(frozen=True)
class CA1MTerminalGatePolicyV4:
    schema: str
    quality25: LogisticGateHeadV4
    benefit05: LogisticGateHeadV4
    max_replacements_per_scene: int
    training_binding_sha256: str
    preregistration_manifest_sha256: str
    dataset_sha256: str
    threshold_dev_gate_passed: bool


def _gate_head(value: Any, name: str) -> LogisticGateHeadV4:
    row = _mapping(value, name)
    mean = np.asarray(row.get("mean"), dtype=np.float64)
    scale = np.asarray(row.get("scale"), dtype=np.float64)
    weights = np.asarray(row.get("weights"), dtype=np.float64)
    bias = float(row.get("bias", float("nan")))
    threshold = float(row.get("threshold", float("nan")))
    if (
        mean.shape != (len(FEATURE_NAMES),)
        or scale.shape != mean.shape
        or weights.shape != mean.shape
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or not np.isfinite(weights).all()
        or np.any(scale <= 0.0)
        or not math.isfinite(bias)
        or not math.isfinite(threshold)
        or not 0.0 < threshold < 1.0
    ):
        raise ValueError(f"terminal gate v4 {name} parameters are invalid")
    return LogisticGateHeadV4(
        mean=_readonly(mean, np.float64),
        scale=_readonly(scale, np.float64),
        weights=_readonly(weights, np.float64),
        bias=bias,
        threshold=threshold,
    )


def load_gate_policy_v4(
    path: Path,
    *,
    expected_training_binding_sha256: str | None = None,
    require_dev_pass: bool = True,
) -> CA1MTerminalGatePolicyV4:
    source, payload = _json(path, "terminal gate v4 policy")
    if set(payload) != {
        "schema", "complete", "train_only", "validation_ground_truth_access",
        "validation_prediction_access", "official_validation_comparable",
        "formal_canonical103_authorized", "threshold_dev_gate_passed",
        "feature_schema", "feature_names", "quality_target", "benefit_target",
        "selection_rule", "fit_fold_ids", "threshold_dev_fold_ids",
        "locked_internal_fold_ids", "anchor_score_source",
        "deploy_b6_scores_used_for_stacked_training", "training_binding_sha256",
        "preregistration_manifest_sha256",
        "dataset_sha256", "dataset_manifest_sha256", "quality25", "benefit05",
        "max_replacements_per_scene", "source_code_sha256",
    }:
        raise ValueError("terminal gate v4 policy keys differ")
    fixed = {
        "schema": POLICY_SCHEMA,
        "complete": True,
        "train_only": True,
        "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "official_validation_comparable": False,
        "formal_canonical103_authorized": False,
        "feature_schema": FEATURE_SCHEMA,
        "quality_target": QUALITY_TARGET,
        "benefit_target": BENEFIT_TARGET,
        "selection_rule": SELECTION_RULE,
        "anchor_score_source": "ca1m_native_b6_final_base_all_fold_oof_row_scores_v2",
        "deploy_b6_scores_used_for_stacked_training": False,
    }
    for key, expected in fixed.items():
        if payload.get(key) != expected:
            raise ValueError(f"terminal gate v4 policy field {key} differs")
    if (
        tuple(payload.get("feature_names", ())) != FEATURE_NAMES
        or tuple(payload.get("fit_fold_ids", ())) != GATE_TRAIN_FOLDS
        or tuple(payload.get("threshold_dev_fold_ids", ())) != THRESHOLD_DEV_FOLDS
        or tuple(payload.get("locked_internal_fold_ids", ())) != LOCKED_INTERNAL_FOLDS
    ):
        raise ValueError("terminal gate v4 policy split/feature contract differs")
    binding_sha = _sha(payload.get("training_binding_sha256"), "policy binding SHA256")
    if (
        expected_training_binding_sha256 is not None
        and binding_sha != _sha(expected_training_binding_sha256, "expected binding SHA256")
    ):
        raise ValueError("terminal gate v4 policy training binding differs")
    passed = payload.get("threshold_dev_gate_passed") is True
    if require_dev_pass and not passed:
        raise PermissionError("terminal gate v4 threshold-dev gate did not pass")
    maximum = int(payload.get("max_replacements_per_scene", 0))
    if maximum != MAX_REPLACEMENTS:
        raise ValueError("terminal gate v4 max replacements is invalid")
    return CA1MTerminalGatePolicyV4(
        schema=POLICY_SCHEMA,
        quality25=_gate_head(payload.get("quality25"), "quality25"),
        benefit05=_gate_head(payload.get("benefit05"), "benefit05"),
        max_replacements_per_scene=maximum,
        training_binding_sha256=binding_sha,
        preregistration_manifest_sha256=_sha(
            payload.get("preregistration_manifest_sha256"),
            "policy preregistration SHA256",
        ),
        dataset_sha256=_sha(payload.get("dataset_sha256"), "policy dataset SHA256"),
        threshold_dev_gate_passed=passed,
    )


@dataclass(frozen=True)
class TerminalGateSelectionV4:
    schema: str
    scene_id: str
    candidate_rows: np.ndarray
    anchor_indices: np.ndarray
    quality_probability: np.ndarray
    benefit_probability: np.ndarray
    evaluated_count: int


def select_terminal_replacements_v4(
    batch: TerminalGateFeatureBatchV4,
    policy: CA1MTerminalGatePolicyV4,
) -> TerminalGateSelectionV4:
    quality = policy.quality25.predict(batch.features)
    benefit = policy.benefit05.predict(batch.features)
    eligible = np.flatnonzero(
        (quality >= policy.quality25.threshold)
        & (benefit >= policy.benefit05.threshold)
    )
    selected: list[int] = []
    for anchor in np.unique(batch.anchor_indices[eligible]).tolist():
        rows = eligible[batch.anchor_indices[eligible] == anchor]
        best = max(
            rows.tolist(),
            key=lambda row: (
                float(benefit[row]), float(quality[row]),
                float(batch.candidate_scores[row]), -int(batch.candidate_rows[row]),
            ),
        )
        selected.append(best)
    selected.sort(
        key=lambda row: (
            -float(benefit[row]), -float(quality[row]),
            -float(batch.candidate_scores[row]), int(batch.anchor_indices[row]),
            int(batch.candidate_rows[row]),
        )
    )
    selected = selected[: policy.max_replacements_per_scene]
    selected.sort(key=lambda row: (int(batch.anchor_indices[row]), int(batch.candidate_rows[row])))
    selection = np.asarray(selected, dtype=np.int64)
    return TerminalGateSelectionV4(
        schema=SELECTION_SCHEMA,
        scene_id=batch.scene_id,
        candidate_rows=_readonly(batch.candidate_rows[selection], np.int64),
        anchor_indices=_readonly(batch.anchor_indices[selection], np.int64),
        quality_probability=_readonly(quality[selection], np.float64),
        benefit_probability=_readonly(benefit[selection], np.float64),
        evaluated_count=len(batch.candidate_rows),
    )


@dataclass(frozen=True)
class GeometryOnlyMaterialization:
    """In-memory v4 materialization; scores/order/count cannot be changed."""

    schema: str
    corners: np.ndarray
    scores: np.ndarray
    replaced_anchor_indices: np.ndarray
    source_candidate_rows: np.ndarray


def materialize_geometry_only(
    *,
    anchor_corners: Any,
    anchor_scores: Any,
    candidate_corners: Any,
    anchor_indices: Any,
    candidate_rows: Any,
) -> GeometryOnlyMaterialization:
    """Apply selected geometry with an invariant-preserving return type.

    File publication/auditing is intentionally a later stage.  This pure
    operation makes the v4 materialization semantics testable while the
    upstream train100 bindings are still pending.
    """

    anchors = np.asarray(anchor_corners)
    scores = np.asarray(anchor_scores)
    candidates = np.asarray(candidate_corners)
    selected_anchors = np.array(anchor_indices, copy=True, order="C")
    selected_candidates = np.array(candidate_rows, copy=True, order="C")
    if (
        anchors.dtype != np.dtype(np.float32)
        or anchors.ndim != 3
        or anchors.shape[1:] != (8, 3)
        or scores.dtype != np.dtype(np.float32)
        or scores.shape != (len(anchors),)
        or candidates.dtype != np.dtype(np.float32)
        or candidates.ndim != 3
        or candidates.shape[1:] != (8, 3)
        or selected_anchors.dtype != np.dtype(np.int64)
        or selected_candidates.dtype != np.dtype(np.int64)
        or selected_anchors.ndim != 1
        or selected_candidates.shape != selected_anchors.shape
    ):
        raise ValueError("gate-v4 materialization arrays violate dtype/shape contract")
    if (
        not np.isfinite(anchors).all()
        or not np.isfinite(scores).all()
        or not np.isfinite(candidates).all()
        or np.any((scores < 0.0) | (scores > 1.0))
        or np.any((selected_anchors < 0) | (selected_anchors >= len(anchors)))
        or np.any((selected_candidates < 0) | (selected_candidates >= len(candidates)))
        or len(np.unique(selected_anchors)) != len(selected_anchors)
    ):
        raise ValueError("gate-v4 materialization selection is invalid")
    output_corners = np.array(anchors, copy=True, order="C")
    output_scores = np.array(scores, copy=True, order="C")
    output_corners[selected_anchors] = candidates[selected_candidates]
    if (
        output_corners.shape != anchors.shape
        or output_scores.shape != scores.shape
        or not np.array_equal(output_scores, scores)
    ):
        raise RuntimeError("gate-v4 geometry-only invariants changed")
    for value in (
        output_corners, output_scores, selected_anchors, selected_candidates
    ):
        value.setflags(write=False)
    return GeometryOnlyMaterialization(
        schema=MATERIALIZATION_SCHEMA,
        corners=output_corners,
        scores=output_scores,
        replaced_anchor_indices=selected_anchors,
        source_candidate_rows=selected_candidates,
    )


__all__ = [
    "AP_PROTOCOL", "B6_OOF_MANIFEST_SCHEMA", "B6_OOF_SCHEMA", "BENEFIT_TARGET",
    "BINDING_SCHEMA", "CA1MTerminalGatePolicyV4",
    "CANDIDATE_EVIDENCE_MANIFEST_SCHEMA", "CONFIG_SCHEMA", "DATASET_SCHEMA",
    "DEV_GATE", "FAILURE_ACTION", "FEATURE_NAMES", "FEATURE_SCHEMA",
    "FIT_DECAY_STEPS", "FIT_ITERATIONS", "FIT_L2", "FIT_LEARNING_RATE",
    "FIT_PROTOCOL", "GATE_TRAIN_FOLDS", "GT_SHADOW_INVENTORY_SCHEMA",
    "IOU_THRESHOLDS",
    "LOCKED_INTERNAL_FOLDS", "MATERIALIZATION_SCHEMA", "MAX_REPLACEMENTS",
    "NATIVE_FEATURE_NAMES", "OBJECTIVE_FIELDS", "PREREGISTRATION_SCHEMA",
    "GeometryOnlyMaterialization", "LogisticGateHeadV4", "NAMESPACE",
    "POLICY_SCHEMA", "QUALITY_TARGET", "RELATION_FEATURE_NAMES",
    "SELECTION_RULE", "SELECTION_SCHEMA", "SPLIT_NAMESPACE",
    "THRESHOLD_DEV_FOLDS", "THRESHOLD_GRID", "TIE_PROTOCOL",
    "TerminalGateFeatureBatchV4",
    "TerminalGateSelectionV4", "build_terminal_gate_features_v4",
    "load_gate_policy_v4", "load_oof_row_scores", "materialize_geometry_only",
    "preregistration_code_records", "preregistration_science_contract",
    "preregistration_upstream_records", "select_terminal_replacements_v4",
    "validate_candidate_evidence_artifact", "validate_preregistration_record",
    "validate_ready", "validate_static_config", "write_binding_create_only",
]
