#!/usr/bin/env python3
"""GT/GPU-free static preflight for CA-only E961 incremental/L6 v2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_e961_incremental_l6_v2 import (
    FEATURE_FORMULAS,
    FEATURE_NAMES,
    INCREMENTAL_OBSERVER_CONFIG,
    INCREMENTAL_PROVIDER_CONFIG,
    LIGHTWEIGHT_FUSION_CONFIG,
    NAMESPACE,
    PENDING_SCHEMA,
    R4_PROTOCOL_SCHEMA,
    R4_PROTOCOL_SHA256,
    R6_PREREGISTRATION_SCHEMA,
    R6_PREREGISTRATION_SHA256,
    SCORE_POLICY,
    SELECTION_REQUIREMENTS,
    SOURCE_RANK_FORMULA,
    TRAINING_HYPERPARAMETERS,
    sha256_file,
)


DEFAULT_CONFIG = ROOT / "config/ca1m_e961_incremental_l6_v2_pending.json"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _static_json(record: Any, name: str, schema: str, sha256: str) -> Path:
    value = _mapping(record, name)
    if set(value) != {"path", "schema", "sha256"}:
        raise ValueError(f"{name} record fields differ")
    path = Path(str(value["path"]))
    if value["schema"] != schema or value["sha256"] != sha256:
        raise ValueError(f"{name} frozen binding differs")
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
        raise ValueError(f"{name} must be an immutable regular file")
    if sha256_file(path) != sha256:
        raise ValueError(f"{name} SHA256 differs")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != schema:
        raise ValueError(f"{name} schema differs")
    return path


def validate_static_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("L6 pending config must be a regular file")
    cfg = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(cfg, Mapping):
        raise ValueError("L6 pending config must be an object")
    if cfg.get("schema") != PENDING_SCHEMA or cfg.get("namespace") != NAMESPACE:
        raise ValueError("L6 pending schema/namespace differs")
    required_false = (
        "run_authorized", "ready_sealed", "gpu_started",
        "ground_truth_access_at_static_stage",
        "formal_fold1_path_or_loader_present",
        "formal_official_validation_path_or_loader_present",
    )
    if any(cfg.get(name) is not False for name in required_false):
        raise ValueError("L6 pending access boundary differs")
    if cfg.get("status") != "static_protocol_only_pending_r6_and_r4_result":
        raise ValueError("L6 pending status differs")
    if cfg.get("method_source") != (
        "scannet_l6_method_only_no_scannet_learned_artifact"
    ):
        raise ValueError("L6 method-source boundary differs")

    bindings = _mapping(cfg.get("static_bindings"), "static_bindings")
    r4_path = _static_json(
        bindings.get("r4_terminal_protocol"), "R4 static protocol",
        R4_PROTOCOL_SCHEMA, R4_PROTOCOL_SHA256,
    )
    r6_path = _static_json(
        bindings.get("r6_terminal_inputs_preregistration"),
        "R6 static preregistration", R6_PREREGISTRATION_SCHEMA,
        R6_PREREGISTRATION_SHA256,
    )

    blockers = _mapping(cfg.get("dynamic_blockers"), "dynamic_blockers")
    expected_blockers = {
        "r6_exact80_receipt": "/extra/ZhaoX/ca1m_tr3d_e961_terminal_inputs_xfit_r2_v5_r6/manifests/M_EXACT80_R6_RECEIPT.json",
        "r4_run_receipt": "/extra/ZhaoX/ca1m_tr3d_terminal_gate_v5_final_r4/reports/RUN_RECEIPT.json",
        "r4_stop_receipt": "/extra/ZhaoX/ca1m_tr3d_terminal_gate_v5_final_r4/reports/STOP.json",
        "r4_fit_dataset": "/extra/ZhaoX/ca1m_tr3d_terminal_gate_v5_final_r4/datasets/fold234_fit60.npz",
        "r4_oof_predictions": "/extra/ZhaoX/ca1m_tr3d_terminal_gate_v5_final_r4/results/fold234_scene_grouped_oof.npz",
        "r4_threshold_receipt": "/extra/ZhaoX/ca1m_tr3d_terminal_gate_v5_final_r4/results/fold234_threshold_receipt.json",
        "state": "pending_not_opened_by_static_preflight",
    }
    if dict(blockers) != expected_blockers:
        raise ValueError("L6 dynamic blocker paths differ")

    states = _mapping(cfg.get("terminal_upstream_states"), "terminal states")
    if (
        (states.get("pass") or {}).get("l6_allowed") is not True
        or (states.get("scientific_stop") or {}).get("l6_allowed") is not True
        or (states.get("provenance_or_implementation_failure") or {}).get("l6_allowed") is not False
    ):
        raise ValueError("L6 terminal PASS/STOP/failure semantics differ")
    pass_state = _mapping(states.get("pass"), "terminal PASS state")
    if (
        pass_state.get("fold234_anchor_state")
        != "reconstruct_by_exact_select_replacements_v5_from_heldout_gate_oof_predictions_and_frozen_threshold_tuple"
        or pass_state.get("full_fit60_final_model_for_fold234_forbidden") is not True
        or pass_state.get("scoring_train_fold_must_exclude_each_row_heldout_fold") is not True
    ):
        raise ValueError("L6 R4 heldout gate OOF anchor-state semantics differ")

    dependencies = _mapping(
        cfg.get("incremental_implementation_dependencies"),
        "incremental implementation dependencies",
    )
    expected_dependencies = {
        "observer": (ROOT / "boxfusion/tr3d_incremental_online.py", "f2017f86187ab671df2bba9c3de0db82ad85092a7bf4cfba8690fa0dcef376f7"),
        "lightweight_stage6_observer": (ROOT / "boxfusion/tr3d_lightweight_fusion.py", "eb07218f2b6851704099c66cfff7382f88dca70153db6e974185168f972c162b"),
        "lightweight_depth_geometry": (ROOT / "boxfusion/tr3d_r2_geometry.py", "f617757e68480697a8485529efd241c4faf4b1a012230c8564924351f79728c7"),
        "lightweight_yaw_geometry": (ROOT / "boxfusion/tr3d_r4_smov_observer.py", "72d0fecdc3355327ff8c6cf47b26483b365dbe7b0efe36062f5d3731430d8464"),
        "ca_worker_client": (ROOT / "boxfusion/ca1m_tr3d_worker_client.py", "aad34038e2df45b8ac154196ed4bcd154b9eb225b2ff5a466068f84b835bdb6b"),
        "ca_worker": (ROOT / "tools/ca1m_tr3d_terminal_worker.py", "e01c8bcea1a00bcb30e2553787e50f8086fa2b73787e86b7d3cd94a039f770d0"),
        "ca_incremental_provider_adapter": (ROOT / "boxfusion/ca1m_e961_incremental_provider_v2.py", "9b84111ab20ffa66f35632674b7949f549465034735c04a66a8839b4545e94d6"),
        "ca_inference_contract": (ROOT / "boxfusion/ca1m_tr3d_inference_contract.py", "06068eee518a37bf091ecfed79f202e4f9a9dc9660ed06d59cb7a4231b167ced"),
        "point_inference_config": (Path("/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/config/tr3d/tr3d_ca1m_foreground_point_inference_xfit_r2.py"), "479f7e61eff9fd23fc086ebc2603e161caa876defe73c556a0e671a8fd35c052"),
        "native_visibility_observer": (ROOT / "boxfusion/ca1m_native_b6_observer.py", "e22965b5527d28369faa3848cbc2d92c4c927905ac29c4e605d338de73464280"),
        "native_feature_contract": (ROOT / "boxfusion/ca1m_native_b6_score.py", "6daea10fe05ad531245a3007839fafe40b380bf1ab201f9ff9d612ee2abb8750"),
        "r4_generic_gate_selection": (ROOT / "boxfusion/ca1m_tr3d_terminal_gate_v5.py", "818b3aa60e1706f8dc03fde6bb872d20e41f31b18e6df8c6dd4ee45ddc1e812d"),
        "scannet_l6_feature_reference": (ROOT / "boxfusion/tr3d_incremental_gate.py", "96b6bcd7ac89b7e388c5336e97f4fa562aa4109498631065332201bcb48f390c"),
        "scannet_l6_label_reference": (ROOT / "tools/build_tr3d_incremental_novelty_dataset.py", "598d260f471749ad31e1022dcbddd2cf136bde5f942eb7b84d08ab84da180c39"),
        "scannet_l6_materializer_reference": (ROOT / "tools/materialize_tr3d_lightweight_active.py", "5772fbd961753310ea6a4c47b95466d3242aefb62d38bc2ac570660a5f3e3cc3"),
    }
    if set(dependencies) != set(expected_dependencies):
        raise ValueError("L6 implementation dependency inventory differs")
    for name, (expected_path, expected_sha) in expected_dependencies.items():
        record = _mapping(dependencies[name], f"{name} dependency")
        if record != {"path": os.fspath(expected_path), "sha256": expected_sha}:
            raise ValueError(f"{name} dependency binding differs")
        if expected_path.is_symlink() or not expected_path.is_file():
            raise ValueError(f"{name} dependency is unavailable")
        if sha256_file(expected_path) != expected_sha:
            raise ValueError(f"{name} dependency SHA256 differs")

    universe = _mapping(cfg.get("candidate_universe"), "candidate_universe")
    if (
        universe.get("source")
        != "causal_lightweight_async_stage6_confirmed_tracks_rerun_per_e961_heldout_detector"
        or universe.get("one_shot_raw_P_is_complete_l6_universe") is not False
        or universe.get("r4_candidate_collection_is_complete_l6_universe") is not False
        or universe.get("r4_candidate_filter")
        != "best_anchor_iou_strictly_greater_than_0.15"
    ):
        raise ValueError("L6 full-proposal universe differs")
    roles = _mapping(universe.get("roles"), "candidate roles")
    expected_roles = {
        "inner_holdout2": {"train_folds": [3, 4], "output_fold": 2},
        "inner_holdout3": {"train_folds": [2, 4], "output_fold": 3},
        "inner_holdout4": {"train_folds": [2, 3], "output_fold": 4},
        "outer_dev": {"train_folds": [2, 3, 4], "output_fold": 0},
    }
    if dict(roles) != expected_roles:
        raise ValueError("L6 E961 role cross-fit differs")
    if universe.get("r6_lineage_manifest_templates") != {
        "P": "/extra/ZhaoX/ca1m_tr3d_e961_terminal_inputs_xfit_r2_v5_r6/manifests/P_{role}_exact20.json",
        "O": "/extra/ZhaoX/ca1m_tr3d_e961_terminal_inputs_xfit_r2_v5_r6/manifests/O_{role}_exact20.json",
        "E": "/extra/ZhaoX/ca1m_tr3d_e961_terminal_inputs_xfit_r2_v5_r6/manifests/E_{role}_exact20.json",
    } or universe.get("schemas") != {
        "P": "boxfusion.ca1m_tr3d_e961_anchor_free_proposal.v5.r2.collection",
        "O": "boxfusion.ca1m_tr3d_e961_oof_overlay.v5.r2.collection",
        "E": "boxfusion.ca1m_tr3d_e961_candidate_native_collection.v5.r2",
    }:
        raise ValueError("L6 R6 P/O/E template or schema differs")
    observer = _mapping(universe.get("incremental_observer"), "incremental observer")
    expected_observer = {
        "implementation_sha256": expected_dependencies["observer"][1],
        "lightweight_implementation_sha256": expected_dependencies["lightweight_stage6_observer"][1],
        "schema": "boxfusion.tr3d_lightweight_online_observer.v1",
        "causal": True,
        "frame_order": "each_scene_P_used_frame_ids_in_recorded_order",
        **INCREMENTAL_OBSERVER_CONFIG,
        "provider_score_threshold": INCREMENTAL_PROVIDER_CONFIG["score_threshold"],
        "provider_max_proposals": INCREMENTAL_PROVIDER_CONFIG["max_proposals"],
        "candidate_rows": "confirmed_tracks_only",
        "anchor_relative_features_computed_only_at_finalize": True,
        "lightweight_fusion": LIGHTWEIGHT_FUSION_CONFIG,
        "async_semantics": {
            "worker_threads": 1,
            "inflight_requests_max": 1,
            "pending_snapshots_max": 1,
            "pending_policy": "replace_stale_with_latest",
            "finalize_policy": "consume_only_if_already_done_then_drop_wait_for_inflight_and_drop_pending",
            "dropped_finalize_results_do_not_update_tracks": True,
        },
        "ca_provider_adapter": {
            "implementation_sha256": expected_dependencies["ca_incremental_provider_adapter"][1],
            "observer_axis_align_argument_semantics": "must_byte_equal_scene_R6_P_world_to_local",
            "transform_source": "each_scene_R6_P_proposal_world_to_local_record_bound_to_used_frame_ids",
            "points_coordinate_frame": "world_unaligned_xyzrgb",
            "worker_transform_argument": "world_to_local",
            "generic_ScanNet_PersistentTR3DWorker_allowed": False,
            "required_result_checks": [
                "source_points_sha256_matches_float32_snapshot",
                "adapter_mode_genuine",
                "all_labels_equal_CA_foreground_class_0",
                "finite_world_corners_scores_and_runtime",
                "scores_in_0_1",
                "nonnegative_point_counts",
                "scene_and_prefix_echo_verified_by_CA_worker_client",
            ],
        },
    }
    if dict(observer) != expected_observer:
        raise ValueError("L6 causal incremental observer parameters differ")
    lineage = _mapping(universe.get("lineage_use"), "lineage_use")
    if (
        lineage.get("P") != "scene_order_checkpoint_receipt_and_used_frame_ids_only"
        or lineage.get("O")
        != "B6_OOF_anchor_geometry_scores_and_near_to_raw_R4_row_mapping_only"
        or lineage.get("E") != "processed_RGBD_native_provenance_only"
        or lineage.get("one_shot_P_or_near_E_rows_used_as_L6_tracks") is not False
    ):
        raise ValueError("L6 R6 lineage-only boundary differs")
    inventory = _mapping(universe.get("future_required_inventory"), "future inventory")
    if (
        inventory.get("exact_scene_count") != 80
        or inventory.get("per_role_scene_count") != 20
        or any(inventory.get(name) is not True for name in (
            "P_O_E_scene_identity_equal", "P_O_E_proposal_sha256_equal",
            "P_O_E_candidate_count_equal", "report_used_keyframe_count",
            "report_provider_call_count", "report_track_count",
            "report_confirmed_track_count",
            "report_one_shot_raw_and_near_counts_as_noncandidate_diagnostics",
        ))
        or inventory.get("failure_action") != "block_before_gt_or_training"
    ):
        raise ValueError("L6 future P/O/E inventory gate differs")

    training = _mapping(cfg.get("training_protocol"), "training_protocol")
    if (
        training.get("fit_folds") != [2, 3, 4]
        or training.get("fit_scene_count") != 60
        or training.get("threshold_source")
        != "pooled_fold234_scene_grouped_l6_oof_only"
        or training.get("candidate_source")
        != "confirmed_causal_lightweight_stage6_tracks_not_one_shot_P_or_r4_near_rows"
        or training.get("fold0_used_for_fit_or_threshold_selection") is not False
        or training.get("fold1_access") is not False
        or training.get("official_validation_access") is not False
        or training.get("scannet_weight_or_policy_access") is not False
        or training.get("policy_activation_authorized") is not False
    ):
        raise ValueError("L6 fold/isolation training contract differs")
    if tuple(training.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("L6 dual-head feature order differs")
    if tuple(training.get("feature_formulas", ())) != FEATURE_FORMULAS:
        raise ValueError("L6 dual-head feature formulas differ")
    if training.get("feature_construction") != {
        "temporal_first18": "byte_compatible_ScanNet_candidate_features_on_raw_best_corners_except_anchor_iou_and_distance_overwritten_from_selected_geometry",
        "ca_native_last5": "stage6_diverse_topK_up_to5_visibility_evidence_on_confirmed_track_with_selected_geometry_raw_or_fused",
        "provider_call_fractions_denominator": "max(provider_calls,1)",
        "anchor_distance_clip_m": 5.0,
        "log_epsilon": 1e-6,
        "matched_anchor_score": "base_finalize_raw_best_corners_match_against_post_terminal_anchors_not_recomputed_for_selected_geometry",
        "volume_and_aspect": "raw_best_corners_world_not_selected_corners_world",
        "anchor_iou_and_distance": "selected_geometry_against_post_terminal_anchors",
        "ground_truth_feature": False,
    }:
        raise ValueError("L6 feature-construction compatibility differs")
    if training.get("targets") != {
        "novel25": "selected_geometry_best_gt_iou_ge_0.25_and_best_gt_post_terminal_coverage_lt_0.25",
        "quality50": "selected_geometry_best_gt_iou_greater_than_or_equal_to_0.50",
        "novel50_gate_metric": "selected_geometry_best_gt_iou_ge_0.50_and_best_gt_post_terminal_coverage_lt_0.50",
    } or training.get("model") != "dual_low_capacity_linear_novel25_quality50" or training.get("heads") != [
        "novel25_logistic", "quality50_logistic",
    ]:
        raise ValueError("L6 dual-head target/model semantics differ")
    if (
        training.get("normalization_source") != "each_oof_training_partition_only"
        or training.get("post_terminal_coverage")
        != "R4_PASS_selected_replacements_or_STOP_identity_B6_OOF"
        or training.get("terminal_raw_candidate_rows_and_l6_track_rows_are_distinct_universes") is not True
    ):
        raise ValueError("L6 OOF normalization/row-universe semantics differ")
    algorithm = _mapping(training.get("training_algorithm"), "training algorithm")
    if algorithm != {
        "optimizer": "deterministic_full_batch_gradient_descent",
        "iterations": TRAINING_HYPERPARAMETERS["iterations"],
        "learning_rate": TRAINING_HYPERPARAMETERS["learning_rate"],
        "learning_rate_schedule": TRAINING_HYPERPARAMETERS["learning_rate_schedule"],
        "l2": TRAINING_HYPERPARAMETERS["l2"],
        "class_weight": TRAINING_HYPERPARAMETERS["class_weight"],
        "iou50_row_weight_multiplier": TRAINING_HYPERPARAMETERS["iou50_row_weight_multiplier"],
        "sigmoid_training_clip": [-40.0, 40.0],
        "initial_coefficients": "all_zero",
        "initial_bias": 0.0,
    }:
        raise ValueError("L6 dual-head optimizer contract differs")
    if training.get("normalization") != "per_oof_" + TRAINING_HYPERPARAMETERS["normalization"]:
        raise ValueError("L6 per-fold normalization contract differs")
    sample_gate = _mapping(training.get("sample_gate"), "sample gate")
    if sample_gate != {
        "fold234_novel25_positive_min": 20,
        "fold234_novel25_negative_min": 20,
        "fold234_quality50_positive_min": 10,
        "each_head_each_training_partition_must_have_both_classes": True,
        "heldout_fold_zero_positive_allowed_and_reported": True,
        "empty_heldout_fold_allowed": False,
        "failure_action": "scientific_STOP_without_lowering_gate",
    }:
        raise ValueError("L6 sample/zero-positive semantics differ")
    selection = _mapping(training.get("threshold_selection"), "threshold selection")
    if (
        selection.get("grid_per_head") != {
            "start": SELECTION_REQUIREMENTS["threshold_grid_start"],
            "stop": SELECTION_REQUIREMENTS["threshold_grid_stop"],
            "step": SELECTION_REQUIREMENTS["threshold_grid_step"],
            "count": SELECTION_REQUIREMENTS["threshold_grid_count_per_head"],
        }
        or selection.get("joint_cartesian_count") != SELECTION_REQUIREMENTS["joint_grid_count"]
        or selection.get("min_selected") != SELECTION_REQUIREMENTS["min_selected"]
        or selection.get("min_precision_novel25") != SELECTION_REQUIREMENTS["min_precision_novel25"]
        or selection.get("min_precision_quality50") != SELECTION_REQUIREMENTS["min_precision_quality50"]
        or selection.get("min_recall_novel50") != SELECTION_REQUIREMENTS["min_recall_novel50"]
        or selection.get("min_positive_folds") != SELECTION_REQUIREMENTS["min_positive_folds"]
        or selection.get("choice_objective") != [
            "max_recall_novel50", "max_precision_quality50", "max_selected",
            "min_novelty_threshold", "min_quality_threshold",
        ]
        or selection.get("selected_mask")
        != "novel25_probability_ge_tn_and_quality50_probability_ge_tq"
        or selection.get("metrics_before_runtime_hard_iou_nms_and_scene_cap") is not True
        or selection.get("positive_fold_semantics")
        != "at_least_one_jointly_selected_row_in_each_of_folds_2_3_4"
        or selection.get("no_passing_choice")
        != "scientific_STOP_with_fallback_thresholds_0.95_0.95_non_deployable"
        or selection.get("fold_tie_is_resolved_by_explicit_choice_objective") is not True
    ):
        raise ValueError("L6 threshold grid/gate/tie contract differs")
    runtime_selection = _mapping(training.get("runtime_selection"), "runtime selection")
    if (
        runtime_selection.get("hard_max_post_terminal_anchor_iou")
        != SELECTION_REQUIREMENTS["hard_max_post_terminal_anchor_iou"]
        or runtime_selection.get("candidate_nms_iou_strictly_greater_than_suppressed")
        != SELECTION_REQUIREMENTS["candidate_nms_iou"]
        or runtime_selection.get("free_space_ratio_mean_strictly_greater_than_rejected") != 0.45
        or runtime_selection.get("quality50_head_position")
        != "admission_gate_after_novelty_gate_before_geometry_veto_not_a_ranking_key"
        or runtime_selection.get("ranking") != [
            "source_rank_desc", "novel25_probability_desc", "best_score_desc",
            "track_id_asc",
        ]
        or runtime_selection.get("source_rank_probability") != "novel25_probability"
        or runtime_selection.get("max_candidates_per_scene")
        != SELECTION_REQUIREMENTS["max_candidates_per_scene"]
        or runtime_selection.get("append_scores_below_every_post_terminal_anchor") is not True
    ):
        raise ValueError("L6 runtime hard filter/cap differs")
    route_audit = _mapping(training.get("scannet_route_gate_audit"), "ScanNet route gate audit")
    if route_audit != {
        "scene_coverage_gate_present": False,
        "ap_nonharm_gate_present": False,
        "original_min_positive_folds": 4,
        "original_fold_count": 5,
        "ca_e961_preregistered_adjustment": "min_positive_folds_3_of_3_because_only_folds234_are_fit_selection_surface",
        "ca_adaptation_changes": [
            "single_novel25_head_to_dual_novel25_and_quality50_heads",
            "original_novel50_precision_gate_to_quality50_precision_gate_at_same_0.45_value",
            "single_181_threshold_grid_to_two_head_181x181_joint_grid",
            "quality50_is_admission_only_and_not_a_runtime_rank_key",
            "explicit_tie_keys_append_min_novelty_threshold_then_min_quality_threshold",
        ],
        "adaptation_reason": "CA_E961_requires_independent_candidate_quality_control_while_preserving_novelty_source_rank",
        "unchanged_numeric_defaults": [
            "iterations_1800", "learning_rate_0.06", "l2_0.003",
            "min_selected_20", "min_precision_novel25_0.70",
            "precision50_gate_value_0.45", "min_recall_novel50_0.15",
            "max_candidates_per_scene_6", "hard_max_anchor_iou_0.10",
            "threshold_grid_0.05_to_0.95_step_0.005",
        ],
    }:
        raise ValueError("L6 ScanNet gate audit/CA adjustment differs")
    if _mapping(cfg.get("fold0_protocol"), "fold0_protocol") != {
        "role": "reused_continuation_diagnostic_only",
        "allowed_only_after_fold234_threshold_frozen": True,
        "retuning": False,
        "model_selection": False,
        "activation_authority": False,
        "execution_not_authorized_by_this_static_protocol": True,
    }:
        raise ValueError("L6 fold0 diagnostic boundary differs")
    deployment = _mapping(cfg.get("future_deployment_boundary"), "deployment boundary")
    if (
        deployment.get("fold1_activation_requires_new_preregistration") is not True
        or deployment.get("fold1_detector_must_exclude_fold1_from_training") is not True
        or deployment.get("this_protocol_contains_fold1_path_or_loader") is not False
        or deployment.get("official_validation_remains_unread") is not True
        or deployment.get("exploratory_policy_deployable") is not False
    ):
        raise ValueError("L6 future deployment boundary differs")
    runtime = _mapping(cfg.get("runtime_contract"), "runtime_contract")
    if runtime != {
        "append_only": True,
        "post_terminal_anchor_rows_first_and_byte_identical": True,
        "anchor_replacement_or_deletion_by_l6": False,
        "candidate_score_policy": SCORE_POLICY,
        "source_rank_formula": SOURCE_RANK_FORMULA,
        "create_only": True,
    }:
        raise ValueError("L6 append-only runtime contract differs")
    forbidden = _mapping(cfg.get("forbidden_reuse"), "forbidden_reuse")
    if forbidden != {
        "old_incremental_namespace": "ca1m_incremental_l6_ca_native_train100_v1",
        "old_incremental_core": "boxfusion/ca1m_incremental_l6.py",
        "old_incremental_config": "config/ca1m_incremental_l6_train100_v1.json",
        "old_incremental_runner": "scripts/run_ca1m_incremental_l6_train100_v1.sh",
        "old_artifact_access": False,
        "scannet_learned_artifact_access": False,
        "raw_checkpoint_or_policy_override": False,
    }:
        raise ValueError("L6 forbidden-reuse boundary differs")
    outputs = _mapping(cfg.get("outputs"), "outputs")
    if outputs != {
        "runtime_root": "/extra/ZhaoX/ca1m_e961_incremental_l6_v2",
        "dataset": "/extra/ZhaoX/ca1m_e961_incremental_l6_v2/datasets/fold234_l6_fit60.npz",
        "oof_predictions": "/extra/ZhaoX/ca1m_e961_incremental_l6_v2/results/fold234_l6_oof.npz",
        "threshold_receipt": "/extra/ZhaoX/ca1m_e961_incremental_l6_v2/results/fold234_l6_threshold.json",
        "exploratory_policy": "/extra/ZhaoX/ca1m_e961_incremental_l6_v2/models/l6.non_deployable.json",
        "fold0_report": "/extra/ZhaoX/ca1m_e961_incremental_l6_v2/reports/fold0_reused_diagnostic.json",
        "ready_config": os.fspath(ROOT / "manifests/ca1m_e961_incremental_l6_v2/READY_CONFIG.json"),
        "run_authorization": os.fspath(ROOT / "manifests/ca1m_e961_incremental_l6_v2/RUN_AUTHORIZATION.json"),
    }:
        raise ValueError("L6 canonical output namespace differs")

    return {
        "schema": "boxfusion.ca1m_e961_incremental_l6_static_preflight.v2",
        "complete": True,
        "static_contract_ready": True,
        "dynamic_prerequisites_complete": False,
        "run_authorized": False,
        "ready_sealed": False,
        "r4_static_protocol": {"path": os.fspath(r4_path), "sha256": sha256_file(r4_path)},
        "r6_static_preregistration": {"path": os.fspath(r6_path), "sha256": sha256_file(r6_path)},
        "candidate_universe": "causal_lightweight_async_stage6_confirmed_tracks",
        "one_shot_P_used_as_complete_l6_universe": False,
        "r4_near_collection_used_as_complete_l6_universe": False,
        "terminal_scientific_pass_allowed": True,
        "terminal_scientific_stop_allowed": True,
        "terminal_provenance_failure_allowed": False,
        "fold234_oof_threshold_selection": True,
        "fold0_reused_diagnostic_only": True,
        "fold1_path_or_loader_opened": False,
        "official_validation_path_or_loader_opened": False,
        "dynamic_artifacts_opened": False,
        "ground_truth_files_opened": False,
        "gpu_started": False,
        "model_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(validate_static_config(args.config), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
