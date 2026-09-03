"""Pure static audit for the future CA-1M locked-F1/deploy boundary.

The module intentionally has no runtime entry point, detector loader, GT
loader, CUDA import, or output writer.  It may read only the explicitly
allow-listed E961/F0/F2/F3/F4 split artifacts and sealed metadata records.
The held-out F1 and official-validation commitments are opaque hashes with no
path or loader in the pending config.
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
DEFAULT_CONFIG = ROOT / "config/ca1m_e961_locked_fold1_deploy_v1_pending.json"
CONFIG_SCHEMA = "boxfusion.ca1m_e961_locked_fold1_deploy_pending_config.v1"
REPORT_SCHEMA = "boxfusion.ca1m_e961_locked_fold1_deploy_static_audit.v1"
NAMESPACE = "ca1m_e961_locked_fold1_deploy_v1"

E961_ROOT = Path(
    "/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/data/tr3d_ca1m_e961_v1"
)
SPLIT_ROOT = E961_ROOT / "splits"
EXPECTED_PATHS = {
    "e961_selection_contract": E961_ROOT / "SELECTION_CONTRACT.json",
    "legacy_split_protocol_metadata": (
        ROOT / "manifests/ca1m_tr3d_benefit_gate_v1/split_manifest.json"
    ),
    "r4_terminal_static_protocol": (
        ROOT / "manifests/ca1m_tr3d_terminal_gate_v5_final_r4/PREREGISTRATION_PROTOCOL.json"
    ),
    "native_b6_all_fold_oof_manifest": (
        ROOT / "models/ca1m_native_b6_final_base_oof_row_scores_v2.manifest.json"
    ),
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
CONFIG_ROOT = Path(
    "/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/config/tr3d"
)
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

_SCENE = re.compile(r"^[0-9]{8}$")


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


def _record(
    record: Any, expected_path: Path, name: str, expected_schema: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(record, Mapping):
        raise ValueError(f"{name} record differs")
    path = Path(str(record.get("path", "")))
    if path != expected_path:
        raise ValueError(f"{name} path differs")
    digest = sha256_file(path)
    if record.get("sha256") != digest:
        raise ValueError(f"{name} SHA256 differs")
    value = _json(path, name)
    if expected_schema is not None and value.get("schema") != expected_schema:
        raise ValueError(f"{name} schema differs")
    if expected_schema is not None and record.get("schema") != expected_schema:
        raise ValueError(f"{name} record schema differs")
    return path, value


def _scene_list(record: Any, expected_path: Path, name: str) -> tuple[str, ...]:
    if not isinstance(record, Mapping) or Path(str(record.get("path", ""))) != expected_path:
        raise ValueError(f"{name} path differs")
    payload = _stable_bytes(expected_path, name)
    if record.get("sha256") != hashlib.sha256(payload).hexdigest():
        raise ValueError(f"{name} SHA256 differs")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name} is not ASCII") from exc
    rows = tuple(text.splitlines())
    if len(rows) != int(record.get("scene_count", 1001)):
        raise ValueError(f"{name} scene count differs")
    if len(rows) != len(set(rows)) or any(_SCENE.fullmatch(row) is None for row in rows):
        raise ValueError(f"{name} scene identities differ")
    return rows


def _ordered_sha(rows: Sequence[str]) -> str:
    return hashlib.sha256("".join(f"{row}\n" for row in rows).encode("ascii")).hexdigest()


def compose_locked_fold1_train(
    e961: Sequence[str], fold0: Sequence[str], fold2: Sequence[str],
    fold3: Sequence[str], fold4: Sequence[str],
) -> tuple[str, ...]:
    """Pure ordered E[:921]+F0+F2+F3+F4 composition; it cannot read F1."""

    components = (tuple(e961), tuple(fold0), tuple(fold2), tuple(fold3), tuple(fold4))
    if tuple(len(value) for value in components) != (961, 20, 20, 20, 20):
        raise ValueError("locked-F1 known component counts differ")
    if any(_SCENE.fullmatch(scene) is None for value in components for scene in value):
        raise ValueError("locked-F1 known scene identity differs")
    result = components[0][:921] + sum(components[1:], ())
    if len(result) != 1001 or len(set(result)) != 1001:
        raise ValueError("locked-F1 exact1001 composition overlaps")
    return result


def compose_canonical_deploy_train(
    e961: Sequence[str], fold0: Sequence[str], fold1: Sequence[str],
    fold2: Sequence[str], fold3: Sequence[str], fold4: Sequence[str],
) -> tuple[str, ...]:
    """Pure future E[:901]+F0+F1+F2+F3+F4 ordered composition.

    The caller must supply an already-authorized F1 sequence.  This helper has
    no path resolution or loader and is not an authorization boundary.
    """

    components = (
        tuple(e961), tuple(fold0), tuple(fold1), tuple(fold2),
        tuple(fold3), tuple(fold4),
    )
    if tuple(len(value) for value in components) != (961, 20, 20, 20, 20, 20):
        raise ValueError("canonical deploy component counts differ")
    if any(_SCENE.fullmatch(scene) is None for value in components for scene in value):
        raise ValueError("canonical deploy scene identity differs")
    result = components[0][:901] + sum(components[1:], ())
    if len(result) != 1001 or len(set(result)) != 1001:
        raise ValueError("canonical deploy exact1001 composition overlaps")
    return result


def _validate_training_configs(config: Mapping[str, Any]) -> None:
    records = config.get("audited_training_configs")
    if not isinstance(records, Mapping) or set(records) != set(EXPECTED_TRAINING_CONFIGS):
        raise ValueError("training-config inventory differs")
    source: dict[str, str] = {}
    for role, expected_path in EXPECTED_TRAINING_CONFIGS.items():
        record = records[role]
        if not isinstance(record, Mapping) or Path(str(record.get("path", ""))) != expected_path:
            raise ValueError(f"{role} training config path differs")
        payload = _stable_bytes(expected_path, f"{role} training config")
        if record.get("sha256") != hashlib.sha256(payload).hexdigest():
            raise ValueError(f"{role} training config SHA256 differs")
        source[role] = payload.decode("utf-8")
    base = source["base"]
    required_base = (
        "batch_size=16", "load_from = None", "resume = False",
        'type="OptimWrapper"', "max_iters=11268", "milestones=[7512, 10329]",
        "val_dataloader = None", "test_dataloader = None",
        "randomness = dict(seed=0, deterministic=True)",
    )
    if any(fragment not in base for fragment in required_base) or "AmpOptimWrapper" in base:
        raise ValueError("shared E961 scratch/fixed-update config differs")
    role_specs = {
        "outer_dev": (0, "ca1m_infos_outer_dev_train1001_visible_foreground_e961_v1.pkl"),
        "inner_holdout2": (2, "ca1m_infos_inner_holdout2_train1001_visible_foreground_e961_v1.pkl"),
        "inner_holdout3": (3, "ca1m_infos_inner_holdout3_train1001_visible_foreground_e961_v1.pkl"),
        "inner_holdout4": (4, "ca1m_infos_inner_holdout4_train1001_visible_foreground_e961_v1.pkl"),
    }
    for role, (fold, annotation) in role_specs.items():
        if (
            f'xfit_role = "{role}"' not in source[role]
            or f"xfit_heldout_fold = {fold}" not in source[role]
            or annotation not in source[role]
            or int(records[role].get("heldout_fold", -1)) != fold
        ):
            raise ValueError(f"{role} effective training role differs")


def _validate_known_splits(config: Mapping[str, Any]) -> dict[str, Any]:
    records = config.get("known_split_inputs")
    if not isinstance(records, Mapping) or set(records) != set(EXPECTED_SPLITS):
        raise ValueError("known split inventory differs")
    splits = {
        name: _scene_list(records[name], path, name)
        for name, path in EXPECTED_SPLITS.items()
    }
    if splits["e941"] != splits["e961"][:941]:
        raise ValueError("E941 is not the ordered E961 prefix")
    components = [splits[name] for name in ("e961", "fold0", "fold2", "fold3", "fold4")]
    flat = [scene for rows in components for scene in rows]
    if len(flat) != len(set(flat)):
        raise ValueError("E961 and known F0/F2/F3/F4 components overlap")

    role_records = config.get("existing_role_train_lists")
    if not isinstance(role_records, Mapping) or set(role_records) != set(EXPECTED_ROLE_LISTS):
        raise ValueError("role train-list inventory differs")
    role_expected = {
        "outer_dev": splits["e961"][:941] + splits["fold2"] + splits["fold3"] + splits["fold4"],
        "inner_holdout2": splits["e961"] + splits["fold3"] + splits["fold4"],
        "inner_holdout3": splits["e961"] + splits["fold2"] + splits["fold4"],
        "inner_holdout4": splits["e961"] + splits["fold2"] + splits["fold3"],
    }
    for role, expected in role_expected.items():
        record = dict(role_records[role])
        record["scene_count"] = 1001
        actual = _scene_list(record, EXPECTED_ROLE_LISTS[role], f"{role} train list")
        if actual != expected:
            raise ValueError(f"{role} ordered exact1001 composition differs")

    e921 = splits["e961"][:921]
    f1_train = compose_locked_fold1_train(
        splits["e961"], splits["fold0"], splits["fold2"],
        splits["fold3"], splits["fold4"],
    )
    e901 = splits["e961"][:901]
    deploy_known = e901 + splits["fold0"] + splits["fold2"] + splits["fold3"] + splits["fold4"]
    if (
        len(f1_train) != 1001 or len(set(f1_train)) != 1001
        or _ordered_sha(e921) != EXPECTED_E921_SHA256
        or _ordered_sha(f1_train) != EXPECTED_F1_TRAIN_SHA256
        or len(deploy_known) != 981 or len(set(deploy_known)) != 981
        or _ordered_sha(e901) != EXPECTED_E901_SHA256
        or _ordered_sha(deploy_known) != EXPECTED_DEPLOY_KNOWN_SHA256
    ):
        raise ValueError("future locked-F1/deploy known composition differs")
    return {
        "current_role_exact1001": True,
        "locked_fold1_train_scene_count": len(f1_train),
        "locked_fold1_train_sha256": _ordered_sha(f1_train),
        "deploy_known_without_fold1_scene_count": len(deploy_known),
        "deploy_known_without_fold1_sha256": _ordered_sha(deploy_known),
    }


def _validate_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    bindings = config.get("static_bindings")
    if (
        not isinstance(bindings, Mapping)
        or set(bindings) != set(EXPECTED_PATHS) | {"incremental_l6_static_protocol"}
    ):
        raise ValueError("static binding inventory differs")
    _, selection = _record(
        bindings["e961_selection_contract"], EXPECTED_PATHS["e961_selection_contract"],
        "E961 selection contract", "boxfusion.tr3d.ca1m_e961_selection.v1",
    )
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
        raise ValueError("opaque held-out identity commitments differ")

    _, legacy = _record(
        bindings["legacy_split_protocol_metadata"],
        EXPECTED_PATHS["legacy_split_protocol_metadata"],
        "legacy split protocol metadata", "boxfusion.ca1m_tr3d_benefit_split.v1",
    )
    expected_gate = {
        "min_delta_ap15": 0.0, "min_delta_ap25": 0.0,
        "min_delta_ap50": 0.005, "min_replacements": 10,
        "min_scenes": 5, "min_positive_gain_fraction": 0.6,
        "max_severe_harm_fraction": 0.1, "max_target_switch_fraction": 0.1,
    }
    locked = (legacy.get("roles") or {}).get("locked_internal_check") or {}
    if (
        legacy.get("locked_internal_gate") != expected_gate
        or locked.get("folds") != [1] or locked.get("scene_count") != 20
        or locked.get("scene_list_sha256") != EXPECTED_F1_COMMITMENT
        or legacy.get("official_validation_access") is not False
    ):
        raise ValueError("pre-existing locked-F1 gate metadata differs")

    _, r4 = _record(
        bindings["r4_terminal_static_protocol"],
        EXPECTED_PATHS["r4_terminal_static_protocol"],
        "R4 terminal static protocol",
        "boxfusion.ca1m_tr3d_terminal_gate_preregistration_protocol.v5.final.r4",
    )
    isolation = ((r4.get("science_contract") or {}).get("isolation") or {})
    if (
        r4.get("operational_authority") is not False
        or (r4.get("access_at_seal") or {}).get("fold1") is not False
        or (r4.get("access_at_seal") or {}).get("official_validation") is not False
        or isolation.get("formal_fold1_path_or_loader_present") is not False
        or isolation.get("formal_official_validation_path_or_loader_present") is not False
    ):
        raise ValueError("R4 isolation boundary differs")

    l6 = bindings["incremental_l6_static_protocol"]
    if l6 != {
        "state": "pending_final_static_protocol", "path": None,
        "sha256": None,
        "schema": "boxfusion.ca1m_e961_incremental_l6_preregistration_protocol.v2",
        "operational_authority": False,
        "pending_config_sha256_is_not_authority": True,
    }:
        raise ValueError("final incremental L6 protocol must remain an unbound blocker")

    _, b6 = _record(
        bindings["native_b6_all_fold_oof_manifest"],
        EXPECTED_PATHS["native_b6_all_fold_oof_manifest"],
        "native B6 all-fold OOF manifest",
        "boxfusion.ca1m_native_b6_oof_row_scores_manifest.v2",
    )
    folds = (b6.get("split") or {}).get("folds") or []
    fold1 = [row for row in folds if row.get("heldout_fold") == 1]
    if len(fold1) != 1:
        raise ValueError("native B6 fold1 OOF record differs")
    row = fold1[0]
    heldout = tuple(row.get("heldout_scene_ids") or ())
    train = tuple(row.get("training_scene_ids") or ())
    if (
        b6.get("each_row_model_excludes_scene") is not True
        or (b6.get("split") or {}).get("all_fold_oof") is not True
        or (b6.get("recipe") or {}).get("heldout_rule")
        != "model_fold_k_trained_on_all_scene_folds_except_k"
        or row.get("model_sha256") != EXPECTED_B6_F1_MODEL_SHA256
        or row.get("heldout_scene_count") != 20 or len(heldout) != 20
        or row.get("training_scene_count") != 80 or len(train) != 80
        or len(set(heldout)) != 20 or len(set(train)) != 80
        or set(heldout) & set(train)
        or row.get("training_excludes_every_heldout_scene") is not True
    ):
        raise ValueError("native B6 fold1 anchor model does not prove OOF exclusion")
    return {
        "r4_static_protocol_bound": True,
        "l6_final_static_protocol_bound": False,
        "b6_fold1_model_sha256": row["model_sha256"],
        "b6_fold1_train_heldout_overlap": 0,
        "b6_sidecar_opened": False,
    }


def _validate_pending_science(config: Mapping[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA or config.get("namespace") != NAMESPACE:
        raise ValueError("pending schema/namespace differs")
    if config.get("operational_authority") is not False:
        raise PermissionError("pending config cannot carry operational authority")
    authorizations = config.get("authorizations")
    if not isinstance(authorizations, Mapping) or not authorizations or any(
        value is not False for value in authorizations.values()
    ):
        raise PermissionError("every pending authorization must be false")
    access = config.get("access_at_static_stage") or {}
    required_false = (
        "fold1_scene_list_path_or_loader_present", "fold1_scene_list_opened",
        "fold1_rgbd_or_prediction_opened", "fold1_ground_truth_opened",
        "official_validation_path_or_loader_present", "official_validation_opened",
        "gpu_started", "output_created",
    )
    if any(access.get(key) is not False for key in required_false):
        raise PermissionError("static access boundary differs")
    commitments = config.get("opaque_heldout_commitments") or {}
    expected = {
        "fold1": (20, EXPECTED_F1_COMMITMENT),
        "official_validation": (107, EXPECTED_VAL_COMMITMENT),
    }
    for name, (count, digest) in expected.items():
        row = commitments.get(name) or {}
        if row != {
            "scene_count": count, "scene_ids_sha256": digest,
            "path": None, "loader": None, "opened": False,
        }:
            raise ValueError(f"{name} opaque commitment differs")

    detector = config.get("locked_fold1_detector") or {}
    if (
        detector.get("heldout_fold") != 1
        or detector.get("ordered_train_formula") != "E961[:921]+F0+F2+F3+F4"
        or detector.get("e_prefix_count") != 921
        or detector.get("e_prefix_sha256") != EXPECTED_E921_SHA256
        or detector.get("train_scene_count") != 1001
        or detector.get("train_scene_list_sha256") != EXPECTED_F1_TRAIN_SHA256
        or detector.get("fold1_training_overlap_count") != 0
        or detector.get("initialization") != "random_scratch_ca_only"
        or detector.get("scannet_weight_or_module_access") is not False
        or detector.get("global_batch") != 16 or detector.get("fp32") is not True
        or detector.get("optimizer_updates") != 11268
        or detector.get("lr_milestones_updates") != [7512, 10329]
        or detector.get("checkpoint_name") != "iter_11268.pth"
        or detector.get("checkpoint_selection") is not False
    ):
        raise ValueError("locked-F1 detector science contract differs")

    barrier = config.get("pre_fold1_result_freeze_barrier") or {}
    expected_routes = {
        "r4_pass_l6_pass": "terminal_plus_l6",
        "r4_pass_l6_scientific_stop": "terminal_only",
        "r4_scientific_stop_l6_pass": "baseline_plus_l6",
        "r4_scientific_stop_l6_scientific_stop": "no_fold1",
    }
    if (
        barrier.get("deterministic_route_table") != expected_routes
        or barrier.get("r4_scientific_stop_terminal_state")
        != "inactive_identity_b6_oof_anchors"
        or barrier.get("r4_scientific_stop_may_continue_only_via_passing_l6_identity_anchor_route") is not True
        or barrier.get("r4_provenance_or_implementation_failure_action") != "permanent_block"
        or barrier.get("l6_fold234_oof_result_must_be_sealed_before_fold1_preregistration") is not True
        or barrier.get("l6_fold0_reused_diagnostic_must_complete_without_retuning") is not True
        or barrier.get("l6_final_static_protocol_sha256_must_be_bound_before_this_protocol_can_seal") is not True
        or barrier.get("l6_pending_config_sha256_cannot_authorize_or_be_preregistered") is not True
        or barrier.get("l6_locked_gate_exact_json_subtree_and_sha256_must_be_copied_before_fold1_preregistration") is not True
        or barrier.get("fold0_or_fold1_cannot_select_l6_branch") is not True
    ):
        raise ValueError("pre-F1 result-freeze state machine differs")

    check = config.get("locked_fold1_one_time_check") or {}
    if (
        check.get("state") != "blocked"
        or check.get("single_attempt_claim_created_before_any_fold1_path_resolution") is not True
        or check.get("candidate_generation_exactly_once") is not True
        or check.get("candidate_detector_source") != "locked_holdout1_iter_11268_only"
        or check.get("candidate_observer_detector_source")
        != "causal_lightweight_stage6_observer_from_locked_holdout1_iter_11268_only"
        or check.get("incremental_candidate_source")
        != "causal_lightweight_stage6_confirmed_tracks_from_locked_holdout1_iter_11268_only"
        or check.get("terminal_candidate_source") != "locked_holdout1_iter_11268_only"
        or check.get("canonical_all_train_detector_for_fold1_forbidden") is not True
        or check.get("native_b6_fold1_model_sha256") != EXPECTED_B6_F1_MODEL_SHA256
        or check.get("terminal_thresholds_copied_byte_identically_from_r4") is not True
        or check.get("fold0_retuning") is not False
        or check.get("fold1_retuning") is not False
        or check.get("post_fold1_branch_switch_or_fallback") is not False
    ):
        raise ValueError("one-time locked-F1 boundary differs")
    l6_gate = check.get("l6_locked_gate_binding") or {}
    if l6_gate != {
        "state": "pending_final_l6_static_protocol",
        "source_protocol_path": None, "source_protocol_sha256": None,
        "exact_json_subtree": None, "exact_json_subtree_sha256": None,
        "runtime_override_allowed": False,
    }:
        raise ValueError("L6 locked gate must remain blocked on the final static protocol")

    metric = config.get("fold1_metric_and_tie_contract") or {}
    expected_ties = [
        "same_gt_gain_desc", "candidate_iou_desc",
        "strict_iou50_probability_desc", "candidate_raw_score_desc",
        "candidate_row_asc",
    ]
    route_gates = metric.get("route_end_to_end_gates") or {}
    if (
        metric.get("implementation")
        != "boxfusion.ca1m_tr3d_xfit_r2_eval.official_ca_ap"
        or metric.get("box_geometry") != "world_enclosing_aabb_from_8_corners"
        or metric.get("candidate_best_gt")
        != "argmax_pairwise_world_aabb_iou_first_index_on_tie"
        or metric.get("ranking")
        != "numpy_argsort_of_negative_float64_score_default_kind_no_override"
        or metric.get("matching")
        != "scene_local_duplicate_aware_key_scene_id_and_gt_index"
        or metric.get("iou_comparison") != "strict_greater_than"
        or metric.get("iou_thresholds") != [0.15, 0.25, 0.5]
        or metric.get("recall_denominator") != "ground_truth_count_plus_1e-6"
        or metric.get("ap_formula") != "continuous_voc_precision_envelope_integral"
        or metric.get("delta_formula") != "active_ap_minus_route_baseline_ap"
        or metric.get("terminal_candidate_tie_order") != expected_ties
        or metric.get("terminal_per_scene_cap") != 16
        or metric.get("terminal_scores_row_order_and_row_count_preserved") is not True
        or metric.get("l6_rows_append_after_all_route_anchor_rows") is not True
        or metric.get("fold1_result_can_change_threshold_model_family_formula_or_route") is not False
        or metric.get("official_validation_evaluator_and_ties_must_be_byte_identical") is not True
        or set(route_gates) != {"terminal_only", "terminal_plus_l6", "baseline_plus_l6"}
        or route_gates["terminal_plus_l6"].get("end_to_end_min_delta_ap15") != 0.0
        or route_gates["terminal_plus_l6"].get("end_to_end_min_delta_ap25") != 0.0
        or route_gates["terminal_plus_l6"].get("end_to_end_min_delta_ap50") != 0.005
    ):
        raise ValueError("fold1 official metric/tie/end-to-end gate contract differs")

    deploy = config.get("canonical_deploy_after_fold1_pass") or {}
    detector = deploy.get("detector") or {}
    if (
        deploy.get("official_validation_remains_unopened") is not True
        or detector.get("ordered_train_formula") != "E961[:901]+F0+F1+F2+F3+F4"
        or detector.get("e_prefix_count") != 901
        or detector.get("e_prefix_sha256") != EXPECTED_E901_SHA256
        or detector.get("known_components_without_fold1_scene_count") != 981
        or detector.get("known_components_without_fold1_sha256") != EXPECTED_DEPLOY_KNOWN_SHA256
        or detector.get("full_train_scene_count") != 1001
        or detector.get("full_train_scene_list_sha256") is not None
        or detector.get("optimizer_updates") != 11268
        or detector.get("checkpoint_selection") is not False
    ):
        raise ValueError("canonical detector pending contract differs")
    oof = deploy.get("five_fold_detector_oof_sources") or {}
    if (
        [oof.get(f"fold{fold}") for fold in range(5)]
        != ["outer_dev", "locked_holdout1", "inner_holdout2", "inner_holdout3", "inner_holdout4"]
        or oof.get("each_scene_detector_excludes_its_fold") is not True
        or oof.get("canonical_all_train_detector_for_refit_rows") is not False
    ):
        raise ValueError("five-fold detector OOF refit boundary differs")
    terminal = deploy.get("terminal_refit") or {}
    incremental = deploy.get("incremental_refit") or {}
    if (
        terminal.get("fit_folds") != [0, 1, 2, 3, 4]
        or terminal.get("threshold_search_or_selection") is not False
        or terminal.get("fold0_or_fold1_threshold_use") is not False
        or terminal.get("official_validation_use") is not False
        or incremental.get("fit_folds") != [0, 1, 2, 3, 4]
        or incremental.get("training_rows")
        != "five_fold_causal_lightweight_stage6_confirmed_track_oof_collection_after_route_specific_five_fold_terminal_or_identity_oof_state"
        or incremental.get("threshold_search_or_selection") is not False
        or incremental.get("fold0_or_fold1_threshold_use") is not False
        or incremental.get("official_validation_use") is not False
    ):
        raise ValueError("terminal/incremental all100 refit boundary differs")


def validate_pending_config(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Validate only the static pending protocol and return an audit report."""

    config_path = Path(config_path)
    config = _json(config_path, "locked-F1 pending config")
    _validate_pending_science(config)
    _validate_training_configs(config)
    split_audit = _validate_known_splits(config)
    metadata_audit = _validate_metadata(config)
    return {
        "schema": REPORT_SCHEMA,
        "complete": True,
        "static_design_pass": True,
        "static_protocol_sealable": False,
        "static_protocol_seal_blocker": "final_incremental_l6_static_protocol_sha256",
        "namespace": NAMESPACE,
        "operational_authority": False,
        "config": {"path": os.fspath(config_path), "sha256": sha256_file(config_path)},
        "four_existing_role_configs_audited": True,
        "known_splits_opened": ["E961", "E941", "F0", "F2", "F3", "F4"],
        "fold1_canonical_scene_list_opened": False,
        "fold1_ground_truth_or_prediction_opened": False,
        "official_validation_opened": False,
        "gpu_started": False,
        "output_created": False,
        "split_audit": split_audit,
        "metadata_audit": metadata_audit,
        "next_state": "await_frozen_r4_and_l6_results_then_separate_fold1_preregistration",
    }


__all__ = [
    "CONFIG_SCHEMA", "DEFAULT_CONFIG", "NAMESPACE", "REPORT_SCHEMA",
    "compose_canonical_deploy_train", "compose_locked_fold1_train",
    "sha256_file", "validate_pending_config",
]
