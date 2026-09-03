"""Fail-closed static audit for the CA-1M locked-F1/deploy v2 design.

This module is intentionally read-only.  It has no detector, F1, ground-truth,
official-validation, CUDA, authorization, seal, or output-writing entry point.
The pending L6 protocol and locked-gate receipt fields must remain null, so a
successful audit is a STATIC DESIGN PASS but never operational authority.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/ca1m_e961_locked_fold1_deploy_v2_pending.json"
CONFIG_SCHEMA = "boxfusion.ca1m_e961_locked_fold1_deploy_pending_config.v2"
REPORT_SCHEMA = "boxfusion.ca1m_e961_locked_fold1_deploy_static_audit.v2"
NAMESPACE = "ca1m_e961_locked_fold1_deploy_v2"

E961_ROOT = Path(
    "/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/data/tr3d_ca1m_e961_v1"
)
SPLIT_ROOT = E961_ROOT / "splits"
CONFIG_ROOT = Path(
    "/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/config/tr3d"
)
EXPECTED_RECORD_PATHS = {
    "e961_selection_contract": E961_ROOT / "SELECTION_CONTRACT.json",
    "legacy_split_protocol_metadata": (
        ROOT / "manifests/ca1m_tr3d_benefit_gate_v1/split_manifest.json"
    ),
    "r4_terminal_static_protocol": (
        ROOT
        / "manifests/ca1m_tr3d_terminal_gate_v5_final_r4/PREREGISTRATION_PROTOCOL.json"
    ),
    "official_ca_ap_implementation": ROOT / "boxfusion/ca1m_tr3d_xfit_r2_eval.py",
}
EXPECTED_SPLITS = {
    "e961": SPLIT_ROOT / "e961_rank100_1060.txt",
    "e941": SPLIT_ROOT / "e941_outer_rank100_1040.txt",
    "fold0": SPLIT_ROOT / "fold0_heldout.txt",
    "fold2": SPLIT_ROOT / "fold2.txt",
    "fold3": SPLIT_ROOT / "fold3.txt",
    "fold4": SPLIT_ROOT / "fold4.txt",
}
EXPECTED_ROLE_LISTS = {
    "outer_dev": SPLIT_ROOT / "outer_dev_train1001.txt",
    "inner_holdout2": SPLIT_ROOT / "inner_holdout2_train1001.txt",
    "inner_holdout3": SPLIT_ROOT / "inner_holdout3_train1001.txt",
    "inner_holdout4": SPLIT_ROOT / "inner_holdout4_train1001.txt",
}
EXPECTED_TRAINING_CONFIGS = {
    "base": CONFIG_ROOT / "tr3d_ca1m_foreground_e961_xfit_v1.py",
    "outer_dev": CONFIG_ROOT / "ca1m_e961_xfit_v1/outer_dev.py",
    "inner_holdout2": CONFIG_ROOT / "ca1m_e961_xfit_v1/inner_holdout2.py",
    "inner_holdout3": CONFIG_ROOT / "ca1m_e961_xfit_v1/inner_holdout3.py",
    "inner_holdout4": CONFIG_ROOT / "ca1m_e961_xfit_v1/inner_holdout4.py",
}

EXPECTED_E921_SHA256 = "32bae2e6791c05b00f037df20dcb4ecc232e10e501b13d13d0a3de9cd48302b2"
EXPECTED_F1_TRAIN_SHA256 = "9510760f3a018354ed8cbf175332ee6102d664e46b909ca25b5c0fc8d3f0ffa0"
EXPECTED_E901_SHA256 = "010e8839c8c91f481010939d15e68c47e60e85178c3963736481b0048920b44c"
EXPECTED_DEPLOY_KNOWN_SHA256 = "f33fd498291e909248c7fde974a7d50b247ecac9814794897e82444f7f15b279"
EXPECTED_F1_COMMITMENT = "d6238bae873c98737858ac3a84c0706091fa9a91113321ac9736a8d64de6d6b6"
EXPECTED_VAL_COMMITMENT = "bd5f3fc66168114048a1b12addc45949c8f54f9c016b921bacfb6fe9e3e7dc2f"
EXPECTED_B6_F1_MODEL_SHA256 = "f97c39e9e99d21fd8e765e66777242b4e02f05968ed78b076ad965958358284d"
EXPECTED_OFFICIAL_CA_AP_SHA256 = "e0786e1d62a3131ae0aa06db23b41d994e39bcd4946e23eacba3c3d763b7a025"

_SCENE = re.compile(r"^[0-9]{8}$")

# Every science/access/authorization subtree is independently pinned.  The
# canonical JSON hash protects values and nesting; the key tuple separately
# rejects additions (including seemingly harmless false/null escape hatches).
_EXACT_SUBTREES = {
    ("authorizations",): (
        "static_protocol_seal,fold1_preregistration_seal,derive_or_publish_training_lists,locked_detector_training,fold1_source_path_resolution,fold1_candidate_generation,fold1_ground_truth_join,fold1_decision,canonical_detector_training,terminal_refit,incremental_refit,official_validation,policy_activation",
        "b16e0aa62f657da61916f6e19586409d3af19db183128f699904c2d3a1b99971",
    ),
    ("access_at_static_stage",): (
        "known_train_split_metadata_only,fold1_identity_commitment_metadata_only,fold1_source_path_or_loader_present,fold1_scene_list_opened,fold1_rgbd_opened,fold1_prediction_opened,fold1_ground_truth_opened,official_validation_path_or_loader_present,official_validation_identity_list_opened,official_validation_prediction_opened,official_validation_ground_truth_opened,native_b6_oof_sidecar_opened,native_b6_manifest_opened,gpu_started,runtime_output_created",
        "34722845e89a43dcf35fabadaaf01e2a989c7574cca29312a9a2eee3fa6afe8a",
    ),
    ("static_bindings",): (
        "e961_selection_contract,legacy_split_protocol_metadata,r4_terminal_static_protocol,official_ca_ap_implementation,incremental_l6_static_protocol,native_b6_fold1_oof_metadata_commitment",
        "11266115ae6e6a754d96f15ded21f643b0a306e3afc52f4c2eae08bf9e46c19f",
    ),
    ("audited_training_configs",): (
        "base,outer_dev,inner_holdout2,inner_holdout3,inner_holdout4",
        "20d069b4a4b9b5d7629d3112d2926add9fa418d1c68489bdade1ee673000e5c5",
    ),
    ("known_split_inputs",): (
        "e961,e941,fold0,fold2,fold3,fold4",
        "84088aae55b74ac55459a3bd2034e43e171992ae425ec9607df227f1f8b24064",
    ),
    ("existing_role_train_lists",): (
        "outer_dev,inner_holdout2,inner_holdout3,inner_holdout4",
        "1996b7806c5a2f276db568f78d8472187f080bad6cee23957ab813de24e8ca8c",
    ),
    ("opaque_heldout_commitments",): (
        "fold1,official_validation",
        "41bdad48b75518e4bd3c27fdbe9c02cd5329b8ae2c5865b63695622f186926e4",
    ),
    ("locked_fold1_detector",): (
        "role,heldout_fold,ordered_train_formula,e_prefix_count,e_prefix_rank_interval_inclusive,e_prefix_sha256,train_scene_count,train_scene_list_sha256,fold1_training_overlap_count,initialization,scannet_weight_or_module_access,global_batch,fp32,seed,optimizer_updates,lr_milestones_updates,checkpoint_name,checkpoint_selection,requires_separate_fold1_preregistration,training_may_precede_one_time_claim,training_opens_fold1_source_or_gt,future_train_list,future_success_receipt",
        "62fad4b8974a0ce9ccd12536a1e5c86a85d6a896ae01f70d289b9838b88f6709",
    ),
    ("pre_fold1_result_freeze_barrier", "deterministic_route_table"): (
        "r4_pass_l6_pass,r4_pass_l6_scientific_stop,r4_scientific_stop_l6_pass,r4_scientific_stop_l6_scientific_stop",
        "89856e72014ab43d7360078e0e11081b01fef391b3dec75e46f2588735e59d10",
    ),
    ("pre_fold1_result_freeze_barrier",): (
        "state,r4_result_exactly_one_of_pass_or_scientific_stop,r4_pass_terminal_state,r4_scientific_stop_terminal_state,r4_provenance_or_implementation_failure_action,l6_final_static_protocol_sha256_required_before_this_protocol_can_seal,l6_pending_config_cannot_be_authority,l6_fold234_result_must_be_sealed,l6_locked_gate_subtree_and_sha256_must_be_bound_before_fold1_preregistration,l6_fold0_reused_diagnostic_required_for_any_l6_pass_route,fold0_threshold_search_selection_retuning,fold1_threshold_search_selection_retuning,fold0_or_fold1_cannot_select_route,deterministic_route_table",
        "3943a3dd700cf8974644dab94f05a0b74e836320f7337c283fc2366f97171136",
    ),
    ("pre_fold1_route_lock", "route_receipt_requirements", "terminal_only"): (
        "r4_result,l6_result,l6_fold0_reused_diagnostic,terminal_policy,terminal_threshold_receipt,l6_policy,l6_threshold_receipt,anchor_state",
        "0503cffbd3cac400f765fecddc417e9bc4cf4dedd5a9132265b0bb0f8cd11749",
    ),
    ("pre_fold1_route_lock", "route_receipt_requirements", "terminal_plus_l6"): (
        "r4_result,l6_result,l6_fold0_reused_diagnostic,terminal_policy,terminal_threshold_receipt,l6_policy,l6_threshold_receipt,anchor_state",
        "873b543119ee0e07a2b3d6fc96b82affa211765133b468ae2bd136a64b474d5b",
    ),
    ("pre_fold1_route_lock", "route_receipt_requirements", "baseline_plus_l6"): (
        "r4_result,l6_result,l6_fold0_reused_diagnostic,terminal_policy,terminal_threshold_receipt_application,terminal_materialization,l6_policy,l6_threshold_receipt,anchor_state",
        "56ad37e40750d7d194b6644e9f3906f2ba4601085bbd93294498904919583b46",
    ),
    ("pre_fold1_route_lock", "route_receipt_requirements"): (
        "terminal_only,terminal_plus_l6,baseline_plus_l6",
        "4387fae83971727aacf58162336f6610d269742834fbdbfe8c7d4235bd5fe507",
    ),
    ("pre_fold1_route_lock",): (
        "state,create_only,sealed_before_locked_detector_training,sealed_before_one_time_claim,selection_source,fold0_or_fold1_used_for_route_selection,route_receipt_requirements",
        "5e3918e433ff61c7a9725193480291a6d876e780e7eb53a2f5cf93ff7a553ff2",
    ),
    ("locked_fold1_one_time_check", "claim"): (
        "create_only,created_before_fold1_source_path_resolution,binds_route_lock_sha256,binds_fold1_preregistration_sha256,binds_locked_detector_receipt_sha256,binds_all_route_policy_and_threshold_receipt_sha256,claim_failure_burns_attempt,retry_or_resume",
        "c139f935d1c1404209768324d006ca9ec9604fb676a2e88ef8f7d39a49668a10",
    ),
    ("locked_fold1_one_time_check", "l6_locked_gate_binding"): (
        "state,source_protocol_path,source_protocol_sha256,exact_json_subtree,exact_json_subtree_sha256,runtime_override_allowed",
        "d036b99c0bf6203c85db56b3fdc1795c63b3901c8fd62c37e58554dfbc70074d",
    ),
    ("locked_fold1_one_time_check", "terminal_locked_gate"): (
        "min_delta_ap15,min_delta_ap25,min_delta_ap50,min_replacements,min_scenes,min_positive_gain_fraction,max_severe_harm_fraction,max_target_switch_fraction,source",
        "cb5d2714670d0872f058ad50e97393dbeb7700617cbd9c82d596739cbd559461",
    ),
    ("locked_fold1_one_time_check",): (
        "state,unlock_requires,claim,fold1_source_path_resolution_after_claim_only,candidate_generation_exactly_once,candidate_detector_source,terminal_candidate_source,incremental_observer_source,incremental_candidate_source,canonical_all_train_detector_for_fold1_forbidden,native_b6_anchor_score_source,native_b6_fold1_model_sha256,candidate_and_feature_manifests_sealed_before_gt,ground_truth_join_exactly_once_after_candidate_seal,terminal_thresholds_applied_byte_identically_from_r4,l6_thresholds_applied_byte_identically_from_bound_l6_protocol_if_route_has_l6,fold0_threshold_search_selection_retuning,fold1_threshold_search_selection_retuning,model_fit_on_fold1,checkpoint_selection_on_fold1,post_fold1_route_switch_or_fallback,l6_locked_gate_binding,terminal_locked_gate,failure_action",
        "661a877c3d5b0f15c1c2174b6be19ec4943c9f2931235ccd4eac56fe60741dfe",
    ),
    ("fold1_metric_and_route_gates", "official_evaluator"): (
        "symbol,implementation_sha256",
        "2c89c40580aeddc85d1272bfe2435adde6aeb6cb90827d9e007f671584e7cb01",
    ),
    ("fold1_metric_and_route_gates", "route_gates", "terminal_only"): (
        "baseline,active,terminal_policy,l6_policy,requirements,frozen_terminal_threshold_application",
        "a3c29a72dd286929c851bb036711566b5c039c6ecb31f5a701748c8c8bc7b81c",
    ),
    ("fold1_metric_and_route_gates", "route_gates", "terminal_plus_l6"): (
        "baseline,active,terminal_policy,l6_policy,terminal_component_requirements,l6_component_requirements,end_to_end_min_delta_ap15,end_to_end_min_delta_ap25,end_to_end_min_delta_ap50,frozen_terminal_and_l6_threshold_application",
        "d62cb98936616876b61ac10cb0303243a4646a90920070f2bf327aad3bab2f3b",
    ),
    ("fold1_metric_and_route_gates", "route_gates", "baseline_plus_l6"): (
        "baseline,active,terminal_policy,terminal_materialization,l6_policy,requirements,frozen_l6_threshold_application",
        "ad45707b32e08542afd28a505d5a69295d1f6518fc791b2cfe0bb248cd790a38",
    ),
    ("fold1_metric_and_route_gates", "route_gates"): (
        "terminal_only,terminal_plus_l6,baseline_plus_l6",
        "24d099d768eb7903ea717cd6c109c68e487ef30bff1502daf1997c51496d79f7",
    ),
    ("fold1_metric_and_route_gates",): (
        "official_evaluator,box_geometry,candidate_best_gt,ranking,matching,iou_comparison,iou_thresholds,recall_denominator,ap_formula,delta_formula,terminal_threshold_comparison,terminal_candidate_tie_order,terminal_per_scene_cap,terminal_scores_row_order_and_row_count_preserved,l6_rank_score_and_tie_contract,l6_rows_append_after_all_route_anchor_rows,route_gates,fold1_result_can_change_threshold_model_family_formula_or_route,official_validation_evaluator_formula_and_ties_must_be_identical",
        "d3c662e2751c68baf73f4f3cc8d759469ae69ed483e5d24f5ef4a257e5da053b",
    ),
    ("canonical_deploy_after_fold1_pass", "detector"): (
        "role,ordered_train_formula,e_prefix_count,e_prefix_rank_interval_inclusive,e_prefix_sha256,known_components_without_fold1_scene_count,known_components_without_fold1_sha256,full_train_scene_count,full_train_scene_list_sha256,initialization,scannet_weight_or_module_access,global_batch,fp32,seed,optimizer_updates,lr_milestones_updates,checkpoint_name,checkpoint_selection,training_allowed_only_after_fold1_pass",
        "7b9b0999becfd42e5134c44655df902f3b3ba3067d9b80d8651cfed040b7c95f",
    ),
    ("canonical_deploy_after_fold1_pass", "five_fold_detector_oof_sources"): (
        "fold0,fold1,fold2,fold3,fold4,each_scene_detector_excludes_its_fold,canonical_all_train_detector_for_refit_rows",
        "9ced4c8f73d1d4e8ea6420b282a8ede9d06a6a7b7620ff5e86160953ab81e72e",
    ),
    ("canonical_deploy_after_fold1_pass", "terminal_refit"): (
        "required_routes,forbidden_route,scene_count,fit_folds,training_rows,family_features_heads_and_hyperparameters,normalization_refit_on_all100_including_f0_f1,head_weights_refit_on_all100_including_f0_f1,thresholds,fold0_or_fold1_threshold_search_selection_retuning,frozen_threshold_application_allowed,official_validation_use",
        "9829d0258b20ab0abf3d6625b9f1e9ffd362f7a14542e164ddeb6b202a9823f4",
    ),
    ("canonical_deploy_after_fold1_pass", "incremental_refit"): (
        "required_routes,forbidden_route,scene_count,fit_folds,training_rows,family_features_heads_and_hyperparameters,normalization_refit_on_all100_including_f0_f1,head_weights_refit_on_all100_including_f0_f1,thresholds,fold0_or_fold1_threshold_search_selection_retuning,frozen_threshold_application_allowed,official_validation_use,append_only_below_all_route_anchor_scores",
        "59d28c7d2939a797b07caf6d53326fb2f99fa8447c44e69f5d1af6e35d8b44b4",
    ),
    ("canonical_deploy_after_fold1_pass", "deployment_order_by_route"): (
        "terminal_only,terminal_plus_l6,baseline_plus_l6",
        "90b0e0c21e1f035bcbf11c00bb6c60699bd4a9a2eb8de6c52b2ee0ffa10ac7fb",
    ),
    ("canonical_deploy_after_fold1_pass",): (
        "official_validation_remains_unopened,detector,five_fold_detector_oof_sources,terminal_refit,incremental_refit,deployment_order_by_route",
        "6b962a0070145868ad037429259553e1e69c6f00d167754d5938bd1f1bb3855c",
    ),
    ("future_artifacts",): (
        "static_protocol,fold1_preregistration,ready_config,run_authorization,route_lock_receipt,one_time_claim,fold1_decision_receipt,stop_receipt,canonical_deploy_bundle",
        "4e85ebe2d3c3697a4b1a991cf221336ec444d84218e089a3f9c6cb2c1344f0d6",
    ),
}

_TOP_KEYS = {
    "schema", "namespace", "status", "operational_authority",
    "authorizations", "access_at_static_stage", "static_bindings",
    "audited_training_configs", "known_split_inputs",
    "existing_role_train_lists", "opaque_heldout_commitments",
    "locked_fold1_detector", "pre_fold1_result_freeze_barrier",
    "pre_fold1_route_lock", "locked_fold1_one_time_check",
    "fold1_metric_and_route_gates", "canonical_deploy_after_fold1_pass",
    "future_artifacts",
}


def _stable_bytes(path: Path, name: str) -> bytes:
    path = Path(path)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(f"{name} must be an absolute non-symlink path")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise ValueError(f"{name} must be a non-empty regular file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.stat(path, follow_symlinks=False)
    identity = lambda value: (  # noqa: E731
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
        value.st_nlink,
    )
    if identity(before) != identity(after) or identity(after) != identity(current):
        raise ValueError(f"{name} changed while being read")
    payload = b"".join(chunks)
    if len(payload) != before.st_size:
        raise ValueError(f"{name} byte count differs")
    return payload


def sha256_file(path: Path) -> str:
    return hashlib.sha256(_stable_bytes(Path(path), "SHA256 input")).hexdigest()


def _json(path: Path, name: str) -> dict[str, Any]:
    value = json.loads(_stable_bytes(path, name).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _semantic_sha(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _subtree(config: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"exact subtree {'.'.join(path)} is missing")
        value = value[key]
    return value


def _validate_exact_subtrees(config: Mapping[str, Any]) -> None:
    if set(config) != _TOP_KEYS:
        raise ValueError("exact top-level key set differs")
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("namespace") != NAMESPACE
        or config.get("status")
        != "static_design_only_not_sealable_pending_final_l6_protocol"
        or config.get("operational_authority") is not False
    ):
        raise PermissionError("pending identity or authority differs")
    for path, (csv_keys, expected_sha) in _EXACT_SUBTREES.items():
        value = _subtree(config, path)
        expected_keys = set(csv_keys.split(","))
        if not isinstance(value, Mapping) or set(value) != expected_keys:
            raise ValueError(f"exact key set differs at {'.'.join(path)}")
        if _semantic_sha(value) != expected_sha:
            raise ValueError(f"exact subtree differs at {'.'.join(path)}")


def _verify_record(record: Mapping[str, Any], key: str) -> dict[str, Any] | None:
    expected_path = EXPECTED_RECORD_PATHS[key]
    if Path(str(record.get("path", ""))) != expected_path:
        raise ValueError(f"{key} path differs")
    payload = _stable_bytes(expected_path, key)
    if hashlib.sha256(payload).hexdigest() != record.get("sha256"):
        raise ValueError(f"{key} SHA256 differs")
    if expected_path.suffix != ".json":
        return None
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict) or value.get("schema") != record.get("schema"):
        raise ValueError(f"{key} schema differs")
    return value


def _scene_list(record: Mapping[str, Any], path: Path, name: str) -> tuple[str, ...]:
    if Path(str(record.get("path", ""))) != path:
        raise ValueError(f"{name} path differs")
    payload = _stable_bytes(path, name)
    if hashlib.sha256(payload).hexdigest() != record.get("sha256"):
        raise ValueError(f"{name} SHA256 differs")
    try:
        rows = tuple(payload.decode("ascii").splitlines())
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name} must be ASCII") from exc
    if (
        len(rows) != record.get("scene_count")
        or len(rows) != len(set(rows))
        or any(_SCENE.fullmatch(row) is None for row in rows)
    ):
        raise ValueError(f"{name} scene identities differ")
    return rows


def _ordered_sha(rows: Sequence[str]) -> str:
    payload = "".join(f"{row}\n" for row in rows).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def compose_locked_fold1_train(
    e961: Sequence[str], fold0: Sequence[str], fold2: Sequence[str],
    fold3: Sequence[str], fold4: Sequence[str],
) -> tuple[str, ...]:
    """Compose E[:921]+F0+F2+F3+F4 without resolving or reading F1."""

    parts = tuple(map(tuple, (e961, fold0, fold2, fold3, fold4)))
    if tuple(map(len, parts)) != (961, 20, 20, 20, 20):
        raise ValueError("locked-F1 known component counts differ")
    if any(_SCENE.fullmatch(x) is None for part in parts for x in part):
        raise ValueError("locked-F1 known scene identity differs")
    result = parts[0][:921] + sum(parts[1:], ())
    if len(result) != 1001 or len(set(result)) != 1001:
        raise ValueError("locked-F1 exact1001 composition overlaps")
    return result


def compose_canonical_deploy_train(
    e961: Sequence[str], fold0: Sequence[str], fold1: Sequence[str],
    fold2: Sequence[str], fold3: Sequence[str], fold4: Sequence[str],
) -> tuple[str, ...]:
    """Pure future E[:901]+F0..F4 composition from caller-supplied sequences."""

    parts = tuple(map(tuple, (e961, fold0, fold1, fold2, fold3, fold4)))
    if tuple(map(len, parts)) != (961, 20, 20, 20, 20, 20):
        raise ValueError("canonical deploy component counts differ")
    if any(_SCENE.fullmatch(x) is None for part in parts for x in part):
        raise ValueError("canonical deploy scene identity differs")
    result = parts[0][:901] + sum(parts[1:], ())
    if len(result) != 1001 or len(set(result)) != 1001:
        raise ValueError("canonical deploy exact1001 composition overlaps")
    return result


def _validate_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    bindings = config["static_bindings"]
    selection = _verify_record(bindings["e961_selection_contract"], "e961_selection_contract")
    assert selection is not None
    disabled = selection.get("disabled_partitions") or {}
    if (
        disabled.get("fold1_scene_count") != 20
        or disabled.get("fold1_scene_list_sha256") != EXPECTED_F1_COMMITMENT
        or disabled.get("fold1_scene_list_opened") is not False
        or disabled.get("fold1_gt_opened") is not False
        or disabled.get("official_validation_scene_count") != 107
        or disabled.get("official_validation_scene_ids_sha256") != EXPECTED_VAL_COMMITMENT
        or selection.get("official_validation_gt_opened") is not False
    ):
        raise ValueError("opaque heldout selection metadata differs")

    legacy = _verify_record(
        bindings["legacy_split_protocol_metadata"], "legacy_split_protocol_metadata",
    )
    assert legacy is not None
    locked = (legacy.get("roles") or {}).get("locked_internal_check") or {}
    if (
        locked.get("folds") != [1]
        or locked.get("scene_count") != 20
        or locked.get("scene_list_sha256") != EXPECTED_F1_COMMITMENT
        or legacy.get("official_validation_access") is not False
    ):
        raise ValueError("legacy locked-fold1 metadata differs")

    r4 = _verify_record(bindings["r4_terminal_static_protocol"], "r4_terminal_static_protocol")
    assert r4 is not None
    isolation = ((r4.get("science_contract") or {}).get("isolation") or {})
    if (
        r4.get("operational_authority") is not False
        or (r4.get("access_at_seal") or {}).get("fold1") is not False
        or (r4.get("access_at_seal") or {}).get("official_validation") is not False
        or isolation.get("formal_fold1_path_or_loader_present") is not False
        or isolation.get("formal_official_validation_path_or_loader_present") is not False
    ):
        raise ValueError("R4 isolation boundary differs")

    official = bindings["official_ca_ap_implementation"]
    _verify_record(official, "official_ca_ap_implementation")
    if (
        official.get("symbol") != "official_ca_ap"
        or official.get("sha256") != EXPECTED_OFFICIAL_CA_AP_SHA256
    ):
        raise ValueError("official_ca_ap implementation binding differs")

    l6 = bindings["incremental_l6_static_protocol"]
    if l6 != {
        "state": "pending_final_static_protocol",
        "path": None,
        "sha256": None,
        "schema": "boxfusion.ca1m_e961_incremental_l6_preregistration_protocol.v2",
        "operational_authority": False,
        "pending_config_is_not_authority": True,
    }:
        raise ValueError("L6 final protocol must remain a null blocker")
    gate = config["locked_fold1_one_time_check"]["l6_locked_gate_binding"]
    if any(gate[key] is not None for key in (
        "source_protocol_path", "source_protocol_sha256",
        "exact_json_subtree", "exact_json_subtree_sha256",
    )):
        raise ValueError("L6 locked-gate subtree must remain a null blocker")
    return {
        "r4_static_protocol_bound": True,
        "official_ca_ap_implementation_sha256": official["sha256"],
        "l6_final_static_protocol_bound": False,
        "l6_locked_gate_subtree_bound": False,
        "b6_fold1_model_sha256": EXPECTED_B6_F1_MODEL_SHA256,
        "b6_manifest_opened": False,
        "b6_sidecar_opened": False,
    }


def _validate_training_configs(config: Mapping[str, Any]) -> None:
    records = config["audited_training_configs"]
    source = {}
    for role, path in EXPECTED_TRAINING_CONFIGS.items():
        record = records[role]
        if Path(str(record.get("path", ""))) != path:
            raise ValueError(f"{role} training config path differs")
        payload = _stable_bytes(path, f"{role} training config")
        if hashlib.sha256(payload).hexdigest() != record.get("sha256"):
            raise ValueError(f"{role} training config SHA256 differs")
        source[role] = payload.decode("utf-8")
    base = source["base"]
    required = (
        "batch_size=16", "load_from = None", "resume = False",
        'type="OptimWrapper"', "max_iters=11268",
        "milestones=[7512, 10329]", "val_dataloader = None",
        "test_dataloader = None", "randomness = dict(seed=0, deterministic=True)",
    )
    if any(fragment not in base for fragment in required) or "AmpOptimWrapper" in base:
        raise ValueError("shared scratch/fixed-update training config differs")
    roles = {
        "outer_dev": (0, "ca1m_infos_outer_dev_train1001_visible_foreground_e961_v1.pkl"),
        "inner_holdout2": (2, "ca1m_infos_inner_holdout2_train1001_visible_foreground_e961_v1.pkl"),
        "inner_holdout3": (3, "ca1m_infos_inner_holdout3_train1001_visible_foreground_e961_v1.pkl"),
        "inner_holdout4": (4, "ca1m_infos_inner_holdout4_train1001_visible_foreground_e961_v1.pkl"),
    }
    for role, (fold, annotation) in roles.items():
        if (
            f'xfit_role = "{role}"' not in source[role]
            or f"xfit_heldout_fold = {fold}" not in source[role]
            or annotation not in source[role]
            or records[role].get("heldout_fold") != fold
        ):
            raise ValueError(f"{role} effective role differs")


def _validate_splits(config: Mapping[str, Any]) -> dict[str, Any]:
    records = config["known_split_inputs"]
    splits = {
        key: _scene_list(records[key], path, key)
        for key, path in EXPECTED_SPLITS.items()
    }
    if splits["e941"] != splits["e961"][:941]:
        raise ValueError("E941 must be the ordered E961 prefix")
    known = [splits[x] for x in ("e961", "fold0", "fold2", "fold3", "fold4")]
    flat = [scene for rows in known for scene in rows]
    if len(flat) != len(set(flat)):
        raise ValueError("known E961/F0/F2/F3/F4 components overlap")

    role_expected = {
        "outer_dev": splits["e961"][:941] + splits["fold2"] + splits["fold3"] + splits["fold4"],
        "inner_holdout2": splits["e961"] + splits["fold3"] + splits["fold4"],
        "inner_holdout3": splits["e961"] + splits["fold2"] + splits["fold4"],
        "inner_holdout4": splits["e961"] + splits["fold2"] + splits["fold3"],
    }
    for role, expected in role_expected.items():
        actual = _scene_list(
            config["existing_role_train_lists"][role], EXPECTED_ROLE_LISTS[role],
            f"{role} train list",
        )
        if actual != expected:
            raise ValueError(f"{role} exact1001 ordered composition differs")

    f1_train = compose_locked_fold1_train(
        splits["e961"], splits["fold0"], splits["fold2"],
        splits["fold3"], splits["fold4"],
    )
    deploy_known = (
        splits["e961"][:901] + splits["fold0"] + splits["fold2"]
        + splits["fold3"] + splits["fold4"]
    )
    if (
        _ordered_sha(splits["e961"][:921]) != EXPECTED_E921_SHA256
        or _ordered_sha(f1_train) != EXPECTED_F1_TRAIN_SHA256
        or len(deploy_known) != 981
        or len(set(deploy_known)) != 981
        or _ordered_sha(splits["e961"][:901]) != EXPECTED_E901_SHA256
        or _ordered_sha(deploy_known) != EXPECTED_DEPLOY_KNOWN_SHA256
    ):
        raise ValueError("locked-F1/canonical known ordered composition differs")
    return {
        "current_role_exact1001": True,
        "locked_fold1_train_scene_count": 1001,
        "locked_fold1_train_sha256": _ordered_sha(f1_train),
        "deploy_known_without_fold1_scene_count": 981,
        "deploy_known_without_fold1_sha256": _ordered_sha(deploy_known),
    }


def validate_pending_config(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Validate the pending static design without opening F1 or official val."""

    config_path = Path(config_path)
    config = _json(config_path, "locked-F1/deploy v2 pending config")
    _validate_exact_subtrees(config)
    metadata = _validate_metadata(config)
    _validate_training_configs(config)
    splits = _validate_splits(config)
    return {
        "schema": REPORT_SCHEMA,
        "complete": True,
        "static_design_pass": True,
        "static_protocol_sealable": False,
        "static_protocol_seal_blockers": [
            "final_incremental_l6_static_protocol_path_and_sha256",
            "exact_l6_locked_gate_subtree_and_sha256",
            "final_l6_pass_stop_receipt_schemas_and_statuses",
        ],
        "namespace": NAMESPACE,
        "operational_authority": False,
        "config": {"path": os.fspath(config_path), "sha256": sha256_file(config_path)},
        "known_splits_opened": ["E961", "E941", "F0", "F2", "F3", "F4"],
        "fold1_source_path_resolved": False,
        "fold1_canonical_scene_list_opened": False,
        "fold1_ground_truth_or_prediction_opened": False,
        "official_validation_opened": False,
        "native_b6_manifest_or_sidecar_opened": False,
        "gpu_started": False,
        "runtime_output_created": False,
        "split_audit": splits,
        "metadata_audit": metadata,
        "next_state": "await_final_l6_static_protocol_then_create_separate_revision",
    }


__all__ = [
    "CONFIG_SCHEMA", "DEFAULT_CONFIG", "EXPECTED_B6_F1_MODEL_SHA256",
    "EXPECTED_F1_TRAIN_SHA256", "EXPECTED_OFFICIAL_CA_AP_SHA256",
    "NAMESPACE", "REPORT_SCHEMA", "compose_canonical_deploy_train",
    "compose_locked_fold1_train", "sha256_file", "validate_pending_config",
]
