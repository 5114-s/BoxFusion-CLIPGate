"""Pending, fail-closed contract for the CA-1M exploratory gate v5.

This revision contains no trainer and no materializer.  It defines the only
candidate/gate cross-fit topology that a later R2 implementation may bind.
Static validation reads only this JSON contract and its schema document.
Operational preflight always stops until a separately sealed ready revision
binds all asymmetric-xfit candidate artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "boxfusion.ca1m_tr3d_exploratory_gate_pending_config.v5"
SCHEMA_DOCUMENT_ID = CONFIG_SCHEMA
NAMESPACE = "ca1m_tr3d_exploratory_gate_xfit_r2_v5"
PENDING_STATE = "pending_asymmetric_xfit_r2_oof_candidates"
EXPECTED_SCHEMA_SHA256 = (
    "9381047c50e33f76e193c3afde34b20ebcb1a4720a85c1be70b75932176c68b5"
)

FIT_FOLDS = (2, 3, 4)
REUSED_DEV_FOLDS = (0,)
LOCKED_FOLDS = (1,)
DETECTOR_ROLES = (
    ("inner_holdout2", (3, 4), (2,), "gate_fit_oof"),
    ("inner_holdout3", (2, 4), (3,), "gate_fit_oof"),
    ("inner_holdout4", (2, 3), (4,), "gate_fit_oof"),
    ("outer_dev", (2, 3, 4), (0,), "reused_dev_diagnostic_only"),
)
GATE_ROLES = (
    ("gate_holdout2", (3, 4), (2,)),
    ("gate_holdout3", (2, 4), (3,)),
    ("gate_holdout4", (2, 3), (4,)),
)

AUTHORIZATIONS = {
    "candidate_collection": False,
    "ground_truth_join": False,
    "gate_fit": False,
    "threshold_selection": False,
    "fold0_reused_dev_diagnostic": False,
    "fold1_internal_check": False,
    "official_validation": False,
    "policy_activation": False,
    "geometry_materialization": False,
    "canonical103": False,
}
ACCESS = {
    "static_preflight_only": True,
    "ground_truth_access": False,
    "fold0_ground_truth_access": False,
    "fold1_metadata_or_ground_truth_access": False,
    "validation_ground_truth_access": False,
    "validation_prediction_access": False,
    "scannet_artifact_access": False,
}
FORBIDDEN_PATH_TOKENS = (
    "scannet",
    "ca1m_tr3d_benefit_gate_final_base_v4",
    "ca1m_tr3d_terminal_ca_native_train100_v4",
    "ca1m_tr3d_benefit_final_base_v4",
    "ca1m_fg_scratch_seed0_fp32_gb16_v1",
)
PENDING_PREREQUISITES = {
    "xfit_r2_training_contract": (
        "boxfusion.tr3d.ca1m_asymmetric_xfit_training_authorization.r2"
    ),
    "xfit_r2_candidate_collection": (
        "boxfusion.ca1m_tr3d_xfit_r2_candidate_collection.v1"
    ),
    "xfit_r2_outer_continuation_receipt": (
        "boxfusion.tr3d.ca1m_xfit_r2_outer_continuation_authorization.v1"
    ),
    "gate_v5_preregistration": (
        "boxfusion.ca1m_tr3d_exploratory_gate_preregistration.v5"
    ),
    "fold234_scene_grouped_gate_oof": (
        "boxfusion.ca1m_tr3d_exploratory_gate_oof_predictions.v5"
    ),
    "fold234_threshold_receipt": (
        "boxfusion.ca1m_tr3d_exploratory_gate_threshold_receipt.v5"
    ),
}

_SHA = re.compile(r"^[0-9a-f]{64}$")


class PendingProtocolError(RuntimeError):
    """Raised before any candidate, GT, checkpoint, or output is opened."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Sequence[str], name: str) -> None:
    if set(value) != set(expected):
        raise ValueError(f"{name} keys differ")


def _json(path: Path, name: str) -> tuple[Path, dict[str, Any]]:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink: {path}")
    source = path.resolve()
    if not source.is_file() or source.is_symlink() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"missing regular {name}: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain an object")
    return source, payload


def _validate_schema_document(record: Any) -> None:
    value = _mapping(record, "schema_document")
    _exact_keys(value, ("path", "sha256"), "schema_document")
    path, document = _json(Path(str(value.get("path", ""))), "v5 schema document")
    digest = str(value.get("sha256", ""))
    if (
        _SHA.fullmatch(digest) is None
        or digest != EXPECTED_SCHEMA_SHA256
        or sha256_file(path) != digest
        or document.get("$id") != SCHEMA_DOCUMENT_ID
        or document.get("additionalProperties") is not False
        or document.get("properties", {}).get("schema", {}).get("const")
        != CONFIG_SCHEMA
        or document.get("properties", {}).get("state", {}).get("const")
        != PENDING_STATE
    ):
        raise ValueError("v5 schema document binding differs")


def _validate_design_basis(value: Any) -> None:
    basis = _mapping(value, "design_basis")
    _exact_keys(basis, (
        "source", "source_is_formal_input", "rejected_policy_is_input",
        "old_candidate_pool_is_input", "fold0_baseline_ap",
        "rejected_gate_ap_delta", "candidate_pool_oracle_ap_delta",
        "candidate_diagnostic", "conclusion",
    ), "design_basis")
    if (
        basis.get("source")
        != "sealed_terminal_gate_v4_fold0_and_oracle_summary_only"
        or basis.get("source_is_formal_input") is not False
        or basis.get("rejected_policy_is_input") is not False
        or basis.get("old_candidate_pool_is_input") is not False
        or basis.get("conclusion")
        != "new_candidate_iou_and_groupwise_benefit_evidence_required_not_loss_only"
    ):
        raise ValueError("v5 design-basis isolation differs")
    baseline = _mapping(basis.get("fold0_baseline_ap"), "fold0 baseline")
    rejected = _mapping(basis.get("rejected_gate_ap_delta"), "rejected gate delta")
    if baseline != {
        "iou_0.15": 0.3760791624675025,
        "iou_0.25": 0.34014544404414054,
        "iou_0.50": 0.18617860029638017,
    } or rejected != {
        "iou_0.15": 0.0, "iou_0.25": 0.0,
        "iou_0.50": 0.0005023975640064959,
    }:
        raise ValueError("v5 frozen diagnostic summary differs")
    oracle = _mapping(
        basis.get("candidate_pool_oracle_ap_delta"), "candidate oracle summary"
    )
    if oracle != {
        "max_any_gt_164_replacements": [0.003209, 0.007226, 0.016358],
        "same_gt_positive_135_replacements": [0.005286, 0.006132, 0.014769],
        "same_gt_gain_ge_0.05_96_replacements": [0.004975, 0.006132, 0.014569],
    }:
        raise ValueError("v5 oracle headroom summary differs")
    diagnostic = _mapping(basis.get("candidate_diagnostic"), "candidate diagnostic")
    _exact_keys(diagnostic, (
        "anchors", "represented_anchors", "same_gt_gain_ge_0.05",
        "same_gt_gain_ge_0.05_scene_count", "candidate_rows",
        "rejected_gate_replacements", "rejected_gate_beneficial_replacements",
        "rejected_gate_severe_harm_replacements", "searched_operating_points",
        "passing_operating_points", "quality_gt_0.25_fraction",
        "benefit_fraction", "target_switch_fraction", "mean_same_gt_gain",
        "raw_score_auc_quality", "raw_score_auc_benefit",
    ), "candidate diagnostic")
    if (
        diagnostic.get("anchors") != 1505
        or diagnostic.get("represented_anchors") != 714
        or diagnostic.get("same_gt_gain_ge_0.05") != 96
        or diagnostic.get("same_gt_gain_ge_0.05_scene_count") != 19
        or diagnostic.get("candidate_rows") != 1863
        or diagnostic.get("rejected_gate_replacements") != 13
        or diagnostic.get("rejected_gate_beneficial_replacements") != 3
        or diagnostic.get("rejected_gate_severe_harm_replacements") != 6
        or diagnostic.get("searched_operating_points") != 400
        or diagnostic.get("passing_operating_points") != 0
        or diagnostic.get("quality_gt_0.25_fraction") != 0.5835
        or diagnostic.get("benefit_fraction") != 0.0633
        or diagnostic.get("target_switch_fraction") != 0.1621
        or diagnostic.get("mean_same_gt_gain") != -0.298
        or diagnostic.get("raw_score_auc_quality") != 0.569
        or diagnostic.get("raw_score_auc_benefit") != 0.465
    ):
        raise ValueError("v5 candidate diagnostic summary differs")


def _validate_protocol(value: Any) -> None:
    protocol = _mapping(value, "protocol")
    _exact_keys(protocol, (
        "split_namespace", "scene_grouped", "fit_folds",
        "gate_oof_threshold_folds", "fold0_reused_dev_diagnostic_folds",
        "locked_internal_folds", "fit_scene_count", "fold0_scene_count",
        "locked_scene_count", "threshold_selection_source",
        "anchor_score_source",
        "deploy_anchor_scores_allowed_for_gate_fit_or_threshold_selection",
        "row_random_split_allowed", "fold0_used_for_fit",
        "fold0_used_for_threshold_or_hyperparameter_selection",
        "fold1_or_validation_used_for_any_selection", "detector_candidate_roles",
        "gate_crossfit_roles", "outer_to_inner_continuation_gate",
        "final_exploratory_gate_fit_folds",
        "final_exploratory_gate_application_folds",
    ), "protocol")
    if (
        protocol.get("split_namespace")
        != "boxfusion.ca1m-native-b6.scene-folds.v1"
        or protocol.get("scene_grouped") is not True
        or tuple(protocol.get("fit_folds", ())) != FIT_FOLDS
        or tuple(protocol.get("gate_oof_threshold_folds", ())) != FIT_FOLDS
        or tuple(protocol.get("fold0_reused_dev_diagnostic_folds", ()))
        != REUSED_DEV_FOLDS
        or tuple(protocol.get("locked_internal_folds", ())) != LOCKED_FOLDS
        or protocol.get("fit_scene_count") != 60
        or protocol.get("fold0_scene_count") != 20
        or protocol.get("locked_scene_count") != 20
        or protocol.get("threshold_selection_source")
        != "fold234_scene_grouped_gate_oof_only"
        or protocol.get("anchor_score_source")
        != "ca1m_native_b6_final_base_all_fold_oof_row_scores_v2"
        or protocol.get(
            "deploy_anchor_scores_allowed_for_gate_fit_or_threshold_selection"
        ) is not False
        or protocol.get("row_random_split_allowed") is not False
        or protocol.get("fold0_used_for_fit") is not False
        or protocol.get("fold0_used_for_threshold_or_hyperparameter_selection")
        is not False
        or protocol.get("fold1_or_validation_used_for_any_selection") is not False
        or tuple(protocol.get("final_exploratory_gate_fit_folds", ())) != FIT_FOLDS
        or tuple(protocol.get("final_exploratory_gate_application_folds", ()))
        != REUSED_DEV_FOLDS
    ):
        raise ValueError("v5 scene-grouped fit/OOF partition differs")

    detector_roles = protocol.get("detector_candidate_roles")
    if not isinstance(detector_roles, list) or len(detector_roles) != len(DETECTOR_ROLES):
        raise ValueError("v5 detector role count differs")
    for row, expected in zip(detector_roles, DETECTOR_ROLES):
        record = _mapping(row, "detector role")
        _exact_keys(record, (
            "role", "detector_train_folds", "candidate_output_folds",
            "candidate_use",
        ), "detector role")
        observed = (
            record.get("role"), tuple(record.get("detector_train_folds", ())),
            tuple(record.get("candidate_output_folds", ())),
            record.get("candidate_use"),
        )
        if observed != expected or set(observed[1]) & set(observed[2]):
            raise ValueError("v5 asymmetric detector role differs")

    gate_roles = protocol.get("gate_crossfit_roles")
    if not isinstance(gate_roles, list) or len(gate_roles) != len(GATE_ROLES):
        raise ValueError("v5 gate cross-fit role count differs")
    for row, expected in zip(gate_roles, GATE_ROLES):
        record = _mapping(row, "gate role")
        _exact_keys(record, (
            "role", "gate_train_folds", "gate_oof_output_folds",
        ), "gate role")
        observed = (
            record.get("role"), tuple(record.get("gate_train_folds", ())),
            tuple(record.get("gate_oof_output_folds", ())),
        )
        if observed != expected or set(observed[1]) & set(observed[2]):
            raise ValueError("v5 scene-grouped gate OOF role differs")

    continuation = _mapping(
        protocol.get("outer_to_inner_continuation_gate"),
        "outer-to-inner continuation gate",
    )
    if continuation != {
        "checkpoint_policy": "fixed_final_iter_only_no_checkpoint_selection",
        "fold0_role": "reused_dev_continuation_diagnostic_only",
        "raw_detector_ap_role": "diagnostic_only_not_a_continuation_criterion",
        "oracle": "same_gt_iou_gain_ge_0.05",
        "min_replacements": 10,
        "min_scenes": 5,
        "min_delta_ap15": 0.0,
        "min_delta_ap25": 0.0,
        "min_delta_ap50": 0.005,
        "all_checks_required": True,
        "inner_models_require_sealed_continuation_receipt": True,
        "failure_action": "stop_without_training_inner_xfit_models",
    }:
        raise ValueError("v5 outer-to-inner continuation authorization differs")


def _pending_record(value: Any, name: str, schema: str) -> None:
    record = _mapping(value, name)
    _exact_keys(record, ("state", "path", "sha256", "schema"), name)
    if record != {
        "state": "pending", "path": None, "sha256": None, "schema": schema,
    }:
        raise ValueError(f"{name} must remain an unbound pending artifact")


def _validate_candidate_source(value: Any) -> None:
    source = _mapping(value, "candidate_source")
    _exact_keys(source, (
        "family", "initialization", "checkpoint_policy",
        "fold0_checkpoint_selection_allowed",
        "candidate_geometry_is_detector_oof_for_every_fit_scene",
        "fold0_geometry_uses_outer234_detector", "collection_schema",
        "scene_schema", "evidence_schema", "expected_scene_count",
        "expected_fit_scene_count", "expected_fold0_scene_count",
        "raw_tr3d_score_role", "raw_tr3d_score_direct_gate_allowed",
        "raw_score_only_model_allowed", "old_candidate_pool_allowed",
        "old_overlay_or_proposal_cache_allowed", "required_scene_provenance",
        "manifest", "role_artifacts",
    ), "candidate_source")
    if (
        source.get("family") != "ca_only_asymmetric_xfit_r2"
        or source.get("initialization") != "random_scratch_ca_only"
        or source.get("checkpoint_policy")
        != "fixed_final_iter_only_no_checkpoint_selection"
        or source.get("fold0_checkpoint_selection_allowed") is not False
        or source.get("candidate_geometry_is_detector_oof_for_every_fit_scene")
        is not True
        or source.get("fold0_geometry_uses_outer234_detector") is not True
        or source.get("collection_schema")
        != "boxfusion.ca1m_tr3d_xfit_r2_candidate_collection.v1"
        or source.get("scene_schema")
        != "boxfusion.ca1m_tr3d_xfit_r2_candidate_scene.v1"
        or source.get("evidence_schema")
        != "boxfusion.ca1m_tr3d_xfit_r2_candidate_evidence.v1"
        or source.get("expected_scene_count") != 80
        or source.get("expected_fit_scene_count") != 60
        or source.get("expected_fold0_scene_count") != 20
        or source.get("raw_tr3d_score_role")
        != "weak_auxiliary_feature_and_final_tie_break_only"
        or source.get("raw_tr3d_score_direct_gate_allowed") is not False
        or source.get("raw_score_only_model_allowed") is not False
        or source.get("old_candidate_pool_allowed") is not False
        or source.get("old_overlay_or_proposal_cache_allowed") is not False
    ):
        raise ValueError("v5 xfit-R2 candidate-source contract differs")
    expected_provenance = (
        "scene_id", "fold_id", "producer_role", "producer_checkpoint_sha256",
        "producer_train_folds", "candidate_corners_sha256",
        "candidate_feature_sha256", "anchor_identity_sha256",
    )
    if tuple(source.get("required_scene_provenance", ())) != expected_provenance:
        raise ValueError("v5 candidate scene provenance fields differ")
    _pending_record(
        source.get("manifest"), "candidate manifest",
        "boxfusion.ca1m_tr3d_xfit_r2_candidate_collection.v1",
    )
    artifacts = _mapping(source.get("role_artifacts"), "candidate role artifacts")
    expected_roles = tuple(role[0] for role in DETECTOR_ROLES)
    if tuple(artifacts) != expected_roles:
        raise ValueError("v5 pending candidate role-artifact order differs")
    expected = {
        "state": "pending", "checkpoint_path": None,
        "checkpoint_sha256": None, "candidate_manifest_path": None,
        "candidate_manifest_sha256": None,
    }
    for role in expected_roles:
        if _mapping(artifacts[role], f"{role} artifact") != expected:
            raise ValueError(f"{role} must remain an unbound pending artifact")


def _validate_learning(value: Any) -> None:
    learning = _mapping(value, "learning")
    _exact_keys(learning, (
        "family", "feature_construction_without_ground_truth",
        "targets_joined_only_after_candidate_collection_seal",
        "standardization_fit_scenes_only", "scene_weights_equalized",
        "raw_score_feature_penalty_multiplier", "required_non_score_feature_groups",
        "heads", "gate_oof_requirement",
    ), "learning")
    if (
        learning.get("family")
        != "ca_native_candidate_iou_and_groupwise_benefit_v1"
        or learning.get("feature_construction_without_ground_truth") is not True
        or learning.get("targets_joined_only_after_candidate_collection_seal")
        is not True
        or learning.get("standardization_fit_scenes_only") is not True
        or learning.get("scene_weights_equalized") is not True
        or float(learning.get("raw_score_feature_penalty_multiplier", 0.0)) < 4.0
    ):
        raise ValueError("v5 CA-native learning contract differs")
    groups = tuple(learning.get("required_non_score_feature_groups", ()))
    if groups != (
        "ca_native_visibility_and_depth_support", "candidate_anchor_geometry",
        "candidate_point_support_and_density", "anchor_group_sibling_context",
    ):
        raise ValueError("v5 non-score feature groups differ")
    heads = _mapping(learning.get("heads"), "learning heads")
    if set(heads) != {
        "candidate_iou_regression", "candidate_iou50_calibration",
        "pairwise_groupwise_benefit",
    } or any(_mapping(heads[name], name).get("required") is not True for name in heads):
        raise ValueError("v5 requires IoU regression, IoU50, and groupwise benefit heads")
    _exact_keys(heads["candidate_iou_regression"], (
        "target", "loss", "primary_operating_region", "required",
    ), "candidate_iou_regression")
    _exact_keys(heads["candidate_iou50_calibration"], (
        "target", "loss", "required",
    ), "candidate_iou50_calibration")
    _exact_keys(heads["pairwise_groupwise_benefit"], (
        "target", "group_keys", "preference_margin", "target_switch_is_harm",
        "loss", "required",
    ), "pairwise_groupwise_benefit")
    if (
        heads["candidate_iou_regression"].get("target")
        != "candidate_max_gt_iou_continuous_0_1"
        or heads["candidate_iou_regression"].get("loss") != "huber_delta_0.10"
        or heads["candidate_iou_regression"].get("primary_operating_region")
        != "iou_0.50"
        or heads["candidate_iou50_calibration"].get("target")
        != "candidate_max_gt_iou_strict_gt_0.50"
        or heads["candidate_iou50_calibration"].get("loss")
        != "scene_balanced_binary_cross_entropy"
        or heads["pairwise_groupwise_benefit"].get("target")
        != "candidate_iou_on_anchor_gt_minus_anchor_iou"
        or tuple(heads["pairwise_groupwise_benefit"].get("group_keys", ()))
        != ("scene_id", "anchor_index")
        or heads["pairwise_groupwise_benefit"].get("preference_margin") != 0.05
        or heads["pairwise_groupwise_benefit"].get("target_switch_is_harm") is not True
        or heads["pairwise_groupwise_benefit"].get("loss")
        != "scene_and_anchor_group_balanced_pairwise_logistic"
    ):
        raise ValueError("v5 target/head definition differs")
    oof = _mapping(learning.get("gate_oof_requirement"), "gate OOF requirement")
    if oof != {
        "each_row_scored_by_gate_excluding_its_scene": True,
        "each_heldout_fold_absent_from_gate_fit": True,
        "expected_oof_folds": [2, 3, 4],
        "thresholds_selected_from_oof_predictions_only": True,
    }:
        raise ValueError("v5 gate OOF requirement differs")


def _validate_selection(value: Any) -> None:
    selection = _mapping(value, "selection")
    _exact_keys(selection, (
        "group_keys", "eligibility", "candidate_iou_threshold_grid",
        "same_gt_gain_threshold_grid", "iou50_probability_threshold_grid",
        "within_group_order", "max_replacements_per_scene", "oof_safety_gate",
        "objective_order", "ap_protocol",
    ), "selection")
    if (
        tuple(selection.get("group_keys", ())) != ("scene_id", "anchor_index")
        or selection.get("eligibility")
        != "predicted_candidate_iou_and_predicted_same_gt_gain"
        or tuple(selection.get("candidate_iou_threshold_grid", ()))
        != (0.45, 0.5, 0.55, 0.6, 0.65, 0.7)
        or tuple(selection.get("same_gt_gain_threshold_grid", ()))
        != (0.0, 0.02, 0.05, 0.08, 0.1, 0.15)
        or tuple(selection.get("iou50_probability_threshold_grid", ()))
        != (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
        or selection.get("max_replacements_per_scene") != 16
        or selection.get("ap_protocol")
        != "world_enclosing_aabb_global_score_duplicate_aware_strict_gt_numpy_default_argsort"
    ):
        raise ValueError("v5 OOF threshold-selection contract differs")
    order = tuple(selection.get("within_group_order", ()))
    if order != (
        "predicted_same_gt_gain_desc", "predicted_candidate_iou_desc",
        "predicted_iou50_probability_desc",
        "raw_tr3d_score_desc_final_tie_break", "candidate_row_asc",
    ):
        raise ValueError("raw TR3D score must be only the final learned-signal tie break")
    gate = _mapping(selection.get("oof_safety_gate"), "OOF safety gate")
    if gate != {
        "min_delta_ap15": 0.0, "min_delta_ap25": 0.0,
        "min_delta_ap50": 0.0025, "min_replacements": 30,
        "min_scenes": 12, "min_positive_gain_fraction": 0.6,
        "max_severe_harm_fraction": 0.1, "max_target_switch_fraction": 0.1,
    }:
        raise ValueError("v5 fold234 OOF safety gate differs")
    if tuple(selection.get("objective_order", ())) != (
        "fold234_oof_delta_ap50_desc", "fold234_oof_delta_ap25_desc",
        "fold234_oof_delta_ap15_desc", "positive_gain_fraction_desc",
        "replacement_count_asc", "threshold_tuple_lexicographic_desc",
    ):
        raise ValueError("v5 fold234 OOF objective differs")


def _validate_diagnostics(value: Any) -> None:
    diagnostic = _mapping(value, "diagnostics")
    expected = {
        "fold0_role": "reused_dev_diagnostic_only",
        "frozen_gate_fit_folds": [2, 3, 4],
        "frozen_candidate_role": "outer_dev",
        "frozen_threshold_source": "fold234_scene_grouped_gate_oof",
        "fold0_retuning_allowed": False,
        "fold0_model_selection_allowed": False,
        "fold0_result_can_authorize_policy": False,
        "fold0_result_can_authorize_fold1_or_validation": False,
        "report_label": "noncanonical_reused_dev_exploratory_diagnostic",
    }
    if diagnostic != expected:
        raise ValueError("v5 fold0 reused-dev diagnostic contract differs")


def _validate_prerequisites(value: Any) -> None:
    records = _mapping(value, "prerequisites")
    if tuple(records) != tuple(PENDING_PREREQUISITES):
        raise ValueError("v5 pending prerequisite set/order differs")
    for name, schema in PENDING_PREREQUISITES.items():
        _pending_record(records.get(name), name, schema)


def _assert_no_forbidden_paths(value: Any, location: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_no_forbidden_paths(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_forbidden_paths(child, f"{location}[{index}]")
    elif isinstance(value, str) and any(
        token in location.lower() for token in ("path", "root", "output")
    ):
        lowered = value.lower()
        if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
            raise ValueError(f"{location} references a forbidden legacy path")


def validate_static_config(path: Path) -> tuple[Path, dict[str, Any]]:
    """Validate the pending contract without opening any future input."""

    source, cfg = _json(path, "exploratory gate v5 pending config")
    _exact_keys(cfg, (
        "schema", "schema_document", "namespace", "state", "exploratory_only",
        "authorizations", "access", "design_basis", "protocol",
        "candidate_source", "learning", "selection", "diagnostics",
        "prerequisites", "outputs", "forbidden_reuse",
    ), "v5 config")
    if (
        cfg.get("schema") != CONFIG_SCHEMA
        or cfg.get("namespace") != NAMESPACE
        or cfg.get("state") != PENDING_STATE
        or cfg.get("exploratory_only") is not True
        or _mapping(cfg.get("authorizations"), "authorizations") != AUTHORIZATIONS
        or _mapping(cfg.get("access"), "access") != ACCESS
    ):
        raise ValueError("v5 pending top-level fail-close state differs")
    _validate_schema_document(cfg.get("schema_document"))
    _validate_design_basis(cfg.get("design_basis"))
    _validate_protocol(cfg.get("protocol"))
    _validate_candidate_source(cfg.get("candidate_source"))
    _validate_learning(cfg.get("learning"))
    _validate_selection(cfg.get("selection"))
    _validate_diagnostics(cfg.get("diagnostics"))
    _validate_prerequisites(cfg.get("prerequisites"))

    outputs = _mapping(cfg.get("outputs"), "outputs")
    expected_output_keys = (
        "root", "dataset", "dataset_manifest", "oof_predictions",
        "threshold_receipt", "exploratory_policy", "fold0_diagnostic_report",
    )
    if tuple(outputs) != expected_output_keys:
        raise ValueError("v5 output set/order differs")
    paths = tuple(Path(str(outputs[key])) for key in expected_output_keys)
    if (
        any(not path.is_absolute() or NAMESPACE not in str(path) for path in paths)
        or len({str(path) for path in paths}) != len(paths)
    ):
        raise ValueError("v5 outputs must be unique absolute paths in the v5 namespace")

    forbidden = _mapping(cfg.get("forbidden_reuse"), "forbidden_reuse")
    expected_flags = {
        "scannet_weights_or_artifacts": False,
        "terminal_gate_v4_rejected_policy": False,
        "terminal_gate_v4_dataset": False,
        "terminal_gate_v4_candidate_evidence": False,
        "terminal_gate_v4_proposal_or_overlay_pool": False,
        "old_ca_tr3d_checkpoint": False,
        "deploy_b6_scores_for_gate_stacking": False,
    }
    if (
        {key: forbidden.get(key) for key in expected_flags} != expected_flags
        or tuple(forbidden.get("forbidden_path_tokens", ()))
        != FORBIDDEN_PATH_TOKENS
        or set(forbidden) != {*expected_flags, "forbidden_path_tokens"}
    ):
        raise ValueError("v5 forbidden-reuse contract differs")
    _assert_no_forbidden_paths(cfg.get("candidate_source"), "candidate_source")
    _assert_no_forbidden_paths(cfg.get("prerequisites"), "prerequisites")
    _assert_no_forbidden_paths(cfg.get("outputs"), "outputs")
    return source, cfg


def static_report(path: Path) -> dict[str, Any]:
    source, cfg = validate_static_config(path)
    return {
        "schema": "boxfusion.ca1m_tr3d_exploratory_gate_static_preflight.v5",
        "ok": True,
        "mode": "static_contract",
        "config": str(source),
        "config_sha256": sha256_file(source),
        "state": cfg["state"],
        "exploratory_only": True,
        "output_created": False,
        "candidate_or_gt_artifact_opened": False,
        "fit_folds": list(FIT_FOLDS),
        "threshold_source": "fold234_scene_grouped_gate_oof_only",
        "fold0_role": "reused_dev_diagnostic_only",
        "fold1_access": False,
        "official_validation_access": False,
        "run_authorized": False,
    }


def validate_ready(path: Path) -> None:
    """Fail before touching future artifacts; v5 currently has no runner."""

    validate_static_config(path)
    raise PendingProtocolError(
        "exploratory gate v5 is pending sealed asymmetric-xfit R2 candidates, "
        "a GT-free evidence manifest, and a new preregistered ready binding; "
        "training, threshold selection, diagnostics, activation, and "
        "materialization remain unauthorized"
    )


__all__ = [
    "ACCESS", "AUTHORIZATIONS", "CONFIG_SCHEMA", "DETECTOR_ROLES",
    "FIT_FOLDS", "FORBIDDEN_PATH_TOKENS", "GATE_ROLES", "LOCKED_FOLDS",
    "NAMESPACE", "PENDING_STATE", "PendingProtocolError", "REUSED_DEV_FOLDS",
    "sha256_file", "static_report", "validate_ready", "validate_static_config",
]
