"""Pending production contract for E961 xfit-R2 terminal-gate-v5 inputs.

The module deliberately exposes no inference, GT join, or output writer.  It
validates the immutable CA-only scene topology and the already sealed
final-base/B6-OOF lineage.  Operational entry points stop before resolving a
checkpoint receipt or creating an output until a separately sealed ready
revision binds four authoritative successful R2-style training receipts.

The future execution topology is P/O/E/M:

* P: four independent exact-20, anchor-free detector proposal collections;
* O: CPU-only association with sealed final-base rows and B6-v2 OOF scores;
* E: freshly recomputed candidate-native evidence from CA train RGB-D; and
* M: exact fit60(folds 2/3/4) plus reused-dev20(fold 0) collection seal.

No fold-1 or official-validation path exists in this contract.
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
DEFAULT_CONFIG = ROOT / "config/ca1m_tr3d_e961_terminal_inputs_v5_pending.json"

CONFIG_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_pending_config.v1"
NAMESPACE = "ca1m_tr3d_e961_terminal_inputs_xfit_r2_v5"
PENDING_STATE = "pending_four_authoritative_e961_r2_success_receipts"
SELECTION_SCHEMA = "boxfusion.tr3d.ca1m_e961_selection.v1"
FINAL_BASE_SCHEMA = "boxfusion.ca1m_final_base_identity_audit.v1"
B6_COLLECTION_SCHEMA = "boxfusion.ca1m_native_b6_final_base_train_collection.v2"
B6_OOF_MANIFEST_SCHEMA = "boxfusion.ca1m_native_b6_oof_row_scores_manifest.v2"
B6_OOF_SCHEMA = "boxfusion.ca1m_native_b6_oof_row_scores.v2"
COMBINED_COLLECTION_SCHEMA = "boxfusion.ca1m_tr3d_xfit_r2_candidate_collection.v1"

ROLE_ORDER = (
    "outer_dev",
    "inner_holdout2",
    "inner_holdout3",
    "inner_holdout4",
)
ROLE_SPECS: dict[str, dict[str, Any]] = {
    "outer_dev": {
        "train_folds": (2, 3, 4),
        "output_fold": 0,
        "use": "reused_dev_diagnostic_only",
        "train_relative": "splits/outer_dev_train1001.txt",
        "predict_relative": "splits/predict_fold0.txt",
        "receipt_schema": "boxfusion.tr3d.ca1m_e961_outer_train_run.r2",
        "expansion": "e941_plus_folds234",
    },
    "inner_holdout2": {
        "train_folds": (3, 4),
        "output_fold": 2,
        "use": "gate_fit_detector_oof",
        "train_relative": "splits/inner_holdout2_train1001.txt",
        "predict_relative": "splits/predict_fold2.txt",
        "receipt_schema": "boxfusion.tr3d.ca1m_xfit_r2_inner_run_receipt.v1",
        "expansion": "e961_plus_folds34",
    },
    "inner_holdout3": {
        "train_folds": (2, 4),
        "output_fold": 3,
        "use": "gate_fit_detector_oof",
        "train_relative": "splits/inner_holdout3_train1001.txt",
        "predict_relative": "splits/predict_fold3.txt",
        "receipt_schema": "boxfusion.tr3d.ca1m_xfit_r2_inner_run_receipt.v1",
        "expansion": "e961_plus_folds24",
    },
    "inner_holdout4": {
        "train_folds": (2, 3),
        "output_fold": 4,
        "use": "gate_fit_detector_oof",
        "train_relative": "splits/inner_holdout4_train1001.txt",
        "predict_relative": "splits/predict_fold4.txt",
        "receipt_schema": "boxfusion.tr3d.ca1m_xfit_r2_inner_run_receipt.v1",
        "expansion": "e961_plus_folds23",
    },
}

AUTHORIZATIONS = {
    "gpu_proposal_collection": False,
    "cpu_overlay": False,
    "candidate_native_evidence": False,
    "candidate_collection_seal": False,
    "ground_truth_join": False,
    "gate_fit": False,
    "fold0_diagnostic": False,
    "fold1_internal_check": False,
    "official_validation": False,
    "policy_activation": False,
}
ACCESS = {
    "official_train_only": True,
    "ground_truth_access": False,
    "fold1_path_present": False,
    "official_validation_path_present": False,
    "scannet_weight_or_artifact_access": False,
    "old_ca_terminal_artifact_access": False,
    "checkpoint_or_candidate_opened_in_static_mode": False,
}
STAGE_ORDER = (
    "P_role_anchor_free",
    "O_cpu_overlay",
    "E_candidate_native",
    "M_exact80_manifest",
)

_SHA = re.compile(r"^[0-9a-f]{64}$")
_SCENE = re.compile(r"^[0-9]{8}$")


class PendingE961InputsError(RuntimeError):
    """Raised before any receipt, checkpoint, candidate, or output is opened."""


def _no_symlink_chain(path: Path, name: str, *, allow_missing_tail: bool = False) -> None:
    if not path.is_absolute():
        raise ValueError(f"{name} path must be absolute: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{name} path contains a symlink: {current}")
        if not current.exists():
            if allow_missing_tail:
                return
            raise FileNotFoundError(f"missing {name} path component: {current}")


def _stable_bytes(path: Path, name: str) -> bytes:
    """Read one regular file through a stable, no-follow file descriptor."""

    _no_symlink_chain(path, name)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise ValueError(f"{name} must be a non-empty regular file: {path}")
        parts: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            parts.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (  # noqa: E731 - compact immutable identity tuple
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns
    )
    current = os.stat(path, follow_symlinks=False)
    if identity(before) != identity(after) or identity(after) != identity(current):
        raise ValueError(f"{name} changed while being read: {path}")
    payload = b"".join(parts)
    if len(payload) != before.st_size:
        raise ValueError(f"{name} byte count changed while being read: {path}")
    return payload


def sha256_file(path: Path) -> str:
    return hashlib.sha256(_stable_bytes(Path(path), "SHA256 input")).hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: Sequence[str], name: str) -> None:
    if set(value) != set(expected):
        missing = sorted(set(expected) - set(value))
        extra = sorted(set(value) - set(expected))
        raise ValueError(f"{name} keys differ; missing={missing}, extra={extra}")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _regular(path: Path, name: str) -> Path:
    if not path.is_absolute():
        path = path.absolute()
    _no_symlink_chain(path, name)
    result = path.resolve(strict=True)
    if not result.is_file() or result.is_symlink() or result.stat().st_size <= 0:
        raise FileNotFoundError(f"missing regular {name}: {result}")
    return result


def _json(path: Path, name: str) -> tuple[Path, dict[str, Any]]:
    source = _regular(path, name)
    try:
        value = json.loads(_stable_bytes(source, name).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain an object")
    return source, value


def _sha(value: Any, name: str) -> str:
    result = str(value)
    if _SHA.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return result


def _bound_json(
    value: Any, name: str, *, schema: str
) -> tuple[Path, dict[str, Any]]:
    record = _mapping(value, f"{name} record")
    if not {"path", "sha256", "schema"}.issubset(record):
        raise ValueError(f"{name} binding fields differ")
    path, payload = _json(Path(str(record.get("path", ""))), name)
    digest = _sha(record.get("sha256"), f"{name} SHA256")
    if sha256_file(path) != digest:
        raise ValueError(f"{name} SHA256 differs")
    if record.get("schema") != schema or payload.get("schema") != schema:
        raise ValueError(f"{name} schema differs")
    return path, payload


def _bound_file(value: Any, name: str) -> Path:
    record = _mapping(value, f"{name} record")
    path = _regular(Path(str(record.get("path", ""))), name)
    digest = _sha(record.get("sha256"), f"{name} SHA256")
    if sha256_file(path) != digest:
        raise ValueError(f"{name} SHA256 differs")
    return path


def _scene_list(value: Any, name: str, expected_count: int) -> tuple[Path, tuple[str, ...]]:
    path = _bound_file(value, name)
    text = _stable_bytes(path, name).decode("utf-8")
    rows = tuple(row.strip() for row in text.splitlines() if row.strip())
    if (
        len(rows) != expected_count
        or len(set(rows)) != expected_count
        or any(_SCENE.fullmatch(row) is None for row in rows)
    ):
        raise ValueError(f"{name} is not exact{expected_count} unique CA scenes")
    return path, rows


def _artifact_list(
    selection_root: Path,
    selection: Mapping[str, Any],
    relative: str,
    expected_count: int,
) -> tuple[str, ...]:
    artifacts = _mapping(selection.get("artifacts"), "selection artifacts")
    record = _mapping(artifacts.get(relative), f"selection artifact {relative}")
    path = _regular(selection_root / relative, f"selection artifact {relative}")
    if (
        Path(str(record.get("relative_path", ""))) != Path(relative)
        or sha256_file(path) != _sha(record.get("sha256"), f"{relative} SHA256")
        or path.stat().st_size != int(record.get("bytes", -1))
    ):
        raise ValueError(f"selection artifact binding differs: {relative}")
    text = _stable_bytes(path, f"selection artifact {relative}").decode("utf-8")
    rows = tuple(row.strip() for row in text.splitlines() if row.strip())
    if len(rows) != expected_count or len(set(rows)) != expected_count:
        raise ValueError(f"selection artifact count differs: {relative}")
    return rows


def _pending_receipt(value: Any, role: str, schema: str) -> None:
    record = _mapping(value, f"{role} pending receipt")
    _exact_keys(record, ("state", "path", "sha256", "schema"), f"{role} pending receipt")
    if record != {
        "state": "pending",
        "path": None,
        "sha256": None,
        "schema": schema,
    }:
        raise ValueError(f"{role} receipt must remain unbound in pending config")


def _formal_input_paths(config: Mapping[str, Any]) -> tuple[str, ...]:
    result: list[str] = []
    scene = _mapping(config.get("scene_contract"), "scene_contract")
    result.append(str(_mapping(scene.get("selection_contract"), "selection").get("path")))
    for role in ROLE_ORDER:
        row = _mapping(_mapping(scene.get("roles"), "roles").get(role), role)
        result.extend((
            str(_mapping(row.get("train_scene_list"), "train list").get("path")),
            str(_mapping(row.get("candidate_scene_list"), "candidate list").get("path")),
        ))
    anchors = _mapping(config.get("anchor_inputs"), "anchor_inputs")
    for key in (
        "final_base_collection", "native_b6_collection",
        "native_b6_oof_sidecar_manifest", "native_b6_oof_sidecar",
    ):
        result.append(str(_mapping(anchors.get(key), key).get("path")))
    result.append(str(anchors.get("final_base_prediction_root")))
    candidate = _mapping(config.get("candidate_inputs"), "candidate_inputs")
    result.append(str(candidate.get("processed_rgbd_root")))
    result.append(str(_mapping(candidate.get("point_inference_config"), "point config").get("path")))
    return tuple(result)


def _guard_output(path: Path, root: Path, name: str) -> Path:
    if not path.is_absolute() or not root.is_absolute():
        raise ValueError(f"{name} must be absolute")
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes the new namespace") from error
    _no_symlink_chain(root, "output root", allow_missing_tail=True)
    _no_symlink_chain(path, name, allow_missing_tail=True)
    return path


def validate_static_config(path: Path = DEFAULT_CONFIG) -> tuple[Path, dict[str, Any]]:
    """Validate only safe, GT-free static inputs and the pending topology."""

    source, config = _json(path, "E961 terminal-input pending config")
    _exact_keys(config, (
        "schema", "namespace", "state", "static_contract_only",
        "authorizations", "access", "scene_contract", "continuation_receipt",
        "receipt_contract", "anchor_inputs", "candidate_inputs", "pipeline", "integrity",
        "outputs", "forbidden_reuse",
    ), "pending config")
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("namespace") != NAMESPACE
        or config.get("state") != PENDING_STATE
        or config.get("static_contract_only") is not True
        or config.get("authorizations") != AUTHORIZATIONS
        or config.get("access") != ACCESS
    ):
        raise ValueError("pending config identity/access/authorization differs")

    scene = _mapping(config.get("scene_contract"), "scene_contract")
    _exact_keys(scene, (
        "selection_contract", "split_namespace", "fit_folds",
        "reused_dev_folds", "fit_scene_count", "reused_dev_scene_count",
        "total_scene_count", "roles",
    ), "scene_contract")
    if (
        scene.get("split_namespace") != "boxfusion.ca1m-native-b6.scene-folds.v1"
        or tuple(scene.get("fit_folds", ())) != (2, 3, 4)
        or tuple(scene.get("reused_dev_folds", ())) != (0,)
        or scene.get("fit_scene_count") != 60
        or scene.get("reused_dev_scene_count") != 20
        or scene.get("total_scene_count") != 80
    ):
        raise ValueError("scene fold/count topology differs")
    selection_path, selection = _bound_json(
        scene.get("selection_contract"), "E961 selection contract",
        schema=SELECTION_SCHEMA,
    )
    if (
        selection.get("complete") is not True
        or selection.get("create_only") is not True
        or selection.get("static_contract_only") is not True
        or selection.get("fold0_gt_opened") is not False
        or selection.get("fold1_gt_opened") is not False
        or selection.get("official_validation_gt_opened") is not False
        or selection.get("optimization", {}).get("initialization")
        != "random_scratch_ca_only"
        or selection.get("optimization", {}).get("optimizer_updates") != 11268
        or selection.get("optimization", {}).get("global_batch") != 16
        or selection.get("optimization", {}).get("fp32") is not True
        or selection.get("optimization", {}).get("scannet_weight_or_module_access") is not False
    ):
        raise ValueError("E961 selection/isolation/training contract differs")

    roles = _mapping(scene.get("roles"), "roles")
    if tuple(roles) != ROLE_ORDER:
        raise ValueError("role order/set differs")
    role_scenes: dict[str, tuple[str, ...]] = {}
    role_trains: dict[str, tuple[str, ...]] = {}
    for role in ROLE_ORDER:
        spec = ROLE_SPECS[role]
        row = _mapping(roles.get(role), role)
        _exact_keys(row, (
            "detector_train_folds", "candidate_output_fold", "candidate_use",
            "train_scene_count", "train_scene_list", "candidate_scene_count",
            "candidate_scene_list", "source_success_receipt",
        ), role)
        if (
            tuple(row.get("detector_train_folds", ())) != spec["train_folds"]
            or row.get("candidate_output_fold") != spec["output_fold"]
            or row.get("candidate_use") != spec["use"]
            or row.get("train_scene_count") != 1001
            or row.get("candidate_scene_count") != 20
            or spec["output_fold"] in spec["train_folds"]
        ):
            raise ValueError(f"{role}: asymmetric detector topology differs")
        train_path, train_rows = _scene_list(row.get("train_scene_list"), f"{role} train list", 1001)
        predict_path, predict_rows = _scene_list(
            row.get("candidate_scene_list"), f"{role} candidate list", 20
        )
        if (
            train_path != selection_path.parent / spec["train_relative"]
            or predict_path != selection_path.parent / spec["predict_relative"]
            or set(train_rows) & set(predict_rows)
        ):
            raise ValueError(f"{role}: train/heldout scene identity differs")
        selection_role = _mapping(_mapping(selection.get("roles"), "selection roles").get(role), role)
        if (
            selection_role.get("heldout_fold") != spec["output_fold"]
            or selection_role.get("heldout_scene_count") != 20
            or selection_role.get("train_scene_count") != 1001
            or selection_role.get("heldout_scene_list") != spec["predict_relative"]
            or selection_role.get("train_scene_list") != spec["train_relative"]
            or selection_role.get("train_scene_list_sha256")
            != sha256_file(train_path)
        ):
            raise ValueError(f"{role}: selection role binding differs")
        _pending_receipt(row.get("source_success_receipt"), role, spec["receipt_schema"])
        role_scenes[role] = predict_rows
        role_trains[role] = train_rows

    flat = tuple(scene_id for role in ROLE_ORDER for scene_id in role_scenes[role])
    if len(flat) != 80 or len(set(flat)) != 80:
        raise ValueError("four candidate roles are not a disjoint exact80 cover")
    if set(role_scenes["inner_holdout2"] + role_scenes["inner_holdout3"] + role_scenes["inner_holdout4"]) & set(role_scenes["outer_dev"]):
        raise ValueError("fit60 overlaps reused-dev20")

    e961 = _artifact_list(selection_path.parent, selection, "splits/e961_rank100_1060.txt", 961)
    e941 = _artifact_list(selection_path.parent, selection, "splits/e941_outer_rank100_1040.txt", 941)
    expected_train_sets = {
        "outer_dev": set(e941).union(*(set(role_scenes[f"inner_holdout{fold}"]) for fold in (2, 3, 4))),
        "inner_holdout2": set(e961) | set(role_scenes["inner_holdout3"]) | set(role_scenes["inner_holdout4"]),
        "inner_holdout3": set(e961) | set(role_scenes["inner_holdout2"]) | set(role_scenes["inner_holdout4"]),
        "inner_holdout4": set(e961) | set(role_scenes["inner_holdout2"]) | set(role_scenes["inner_holdout3"]),
    }
    for role in ROLE_ORDER:
        if set(role_trains[role]) != expected_train_sets[role]:
            raise ValueError(f"{role}: exact1001 E961 composition differs")

    continuation = _mapping(config.get("continuation_receipt"), "continuation receipt")
    _exact_keys(continuation, (
        "state", "path", "sha256", "schema",
        "required_before_inner_receipt_acceptance",
    ), "continuation receipt")
    if continuation != {
        "state": "pending", "path": None, "sha256": None,
        "schema": "boxfusion.ca1m_tr3d_e961_outer_dev_continuation_receipt.v1",
        "required_before_inner_receipt_acceptance": True,
    }:
        raise ValueError("outer continuation receipt must remain pending and mandatory")

    receipt_contract = _mapping(config.get("receipt_contract"), "receipt_contract")
    _exact_keys(receipt_contract, (
        "producer_verifier_must_pass_before_normalization",
        "normalization_schema", "required_success_fields", "cross_checks",
    ), "receipt_contract")
    if (
        receipt_contract.get("producer_verifier_must_pass_before_normalization") is not True
        or receipt_contract.get("normalization_schema")
        != "boxfusion.ca1m_tr3d_xfit_r2_detector_role_receipt.v1"
        or receipt_contract.get("required_success_fields") != {
            "complete": True, "create_only": True, "status": "success",
            "exit_code": 0, "checkpoint_name": "iter_11268.pth",
            "optimizer_updates": 11268, "checkpoint_selection": False,
            "initialization": "random_scratch_ca_only", "global_batch": 16,
            "fp32": True, "train_scene_count": 1001,
            "fold1_access": False, "official_validation_access": False,
            "scannet_checkpoint_or_module_access": False,
        }
        or receipt_contract.get("cross_checks") != [
            "receipt_role_equals_config_role",
            "receipt_train_folds_equal_config_train_folds",
            "receipt_heldout_fold_equals_candidate_output_fold",
            "effective_config_train_list_sha256_equals_config",
            "checkpoint_sha256_rehash_equals_receipt",
            "checkpoint_and_receipt_inode_stable_during_read",
            "inner_receipts_bind_passing_outer_continuation_receipt",
        ]
    ):
        raise ValueError("authoritative R2 success-receipt contract differs")

    anchors = _mapping(config.get("anchor_inputs"), "anchor_inputs")
    _exact_keys(anchors, (
        "final_base_collection", "final_base_prediction_root",
        "native_b6_collection", "native_b6_oof_sidecar_manifest",
        "native_b6_oof_sidecar",
    ), "anchor_inputs")
    final_path, final = _bound_json(
        anchors.get("final_base_collection"), "sealed final-base collection",
        schema=FINAL_BASE_SCHEMA,
    )
    final_record = _mapping(anchors.get("final_base_collection"), "final base record")
    if final_record.get("geometry_and_row_authority") is not True:
        raise ValueError("final-base must be the only geometry/row authority")
    final_scenes = _mapping(final.get("per_scene"), "final-base per_scene")
    if (
        final.get("dataset") != "CA1M"
        or final.get("ground_truth_access") is not False
        or final.get("evaluation_invoked") is not False
        or len(final_scenes) != 100
        or not set(flat).issubset(final_scenes)
    ):
        raise ValueError("sealed final-base collection differs")
    prediction_root = Path(str(anchors.get("final_base_prediction_root", ""))).resolve()
    expected_prediction_root = ROOT / "results/ca1m_native_final_base_train100_v1/final_base"
    if prediction_root != expected_prediction_root.resolve():
        raise ValueError("final-base prediction root differs")

    b6_collection_path, b6_collection = _bound_json(
        anchors.get("native_b6_collection"), "native-B6-v2 collection",
        schema=B6_COLLECTION_SCHEMA,
    )
    b6_record = _mapping(anchors.get("native_b6_collection"), "B6 collection record")
    b6_rows = b6_collection.get("scenes")
    if (
        b6_record.get("anchor_native_evidence_only") is not True
        or b6_collection.get("complete") is not True
        or b6_collection.get("geometry_authority") != "sealed_final_base_prediction"
        or b6_collection.get("old_native_b6_checkpoint_reused") is not False
        or b6_collection.get("old_native_b6_diagnostics_reused") is not False
        or b6_collection.get("validation_ground_truth_access") is not False
        or b6_collection.get("validation_prediction_access") is not False
        or not isinstance(b6_rows, list)
        or len(b6_rows) != 100
        or {str(row.get("scene_id")) for row in b6_rows if isinstance(row, Mapping)}
        != set(final_scenes)
        or (b6_collection.get("source_final_base_collection") or {}).get("sha256")
        != sha256_file(final_path)
    ):
        raise ValueError("native-B6-v2 anchor evidence collection differs")

    oof_manifest_path, oof_manifest = _bound_json(
        anchors.get("native_b6_oof_sidecar_manifest"), "native-B6-v2 OOF manifest",
        schema=B6_OOF_MANIFEST_SCHEMA,
    )
    sidecar = _bound_file(anchors.get("native_b6_oof_sidecar"), "native-B6-v2 OOF sidecar")
    sidecar_record = _mapping(anchors.get("native_b6_oof_sidecar"), "OOF sidecar record")
    if (
        sidecar_record.get("schema") != B6_OOF_SCHEMA
        or sidecar_record.get("score_member") != "deployment_blend_oof_scores"
        or sidecar_record.get("each_row_model_excludes_scene") is not True
        or sidecar_record.get("deploy_or_in_sample_scores_allowed") is not False
        or oof_manifest.get("complete") is not True
        or oof_manifest.get("scene_group_oof") is not True
        or oof_manifest.get("each_row_model_excludes_scene") is not True
        or oof_manifest.get("scene_count") != 100
        or oof_manifest.get("row_count") != 6682
        or (oof_manifest.get("artifact") or {}).get("path") != str(sidecar)
        or (oof_manifest.get("artifact") or {}).get("sha256") != sha256_file(sidecar)
        or (oof_manifest.get("artifact") or {}).get("schema") != B6_OOF_SCHEMA
        or oof_manifest.get("row_identity") != [
            "scene_ids", "fold_ids", "source_row_indices"
        ]
    ):
        raise ValueError("native-B6-v2 all-fold OOF binding differs")
    # Static mode hashes the safe sidecar bytes but deliberately does not load
    # any NPZ member.  Exact row/fold/finite checks belong to the ready CPU
    # overlay preflight after all four detector receipts are sealed.
    _ = (b6_collection_path, oof_manifest_path)

    candidates = _mapping(config.get("candidate_inputs"), "candidate_inputs")
    _exact_keys(candidates, (
        "processed_rgbd_root", "point_inference_config", "protocol",
        "proposal_input_surface", "proposal_forbidden_surface",
    ), "candidate_inputs")
    point_config = _bound_file(candidates.get("point_inference_config"), "CA-only point inference config")
    protocol = _mapping(candidates.get("protocol"), "candidate protocol")
    if (
        Path(str(candidates.get("processed_rgbd_root", ""))).resolve()
        != Path("/extra/ZhaoX/boxfusion_ca1m_native_b6_train100_v1")
        or candidates.get("proposal_input_surface") != [
            "authoritative_role_checkpoint", "processed_train_rgbd_pose_intrinsics"
        ]
        or candidates.get("proposal_forbidden_surface") != [
            "final_base_anchor", "native_b6", "ground_truth"
        ]
        or protocol != {
            "pixel_stride": 4, "voxel_size_m": 0.01,
            "min_depth_m": 0.1, "max_depth_m": 6.0,
            "depth_scale": 1000.0, "score_threshold": 0.01,
            "max_proposals": 256, "near_iou": 0.15,
            "frame_schedule": "demo_gap20_early_finalize_reachable_frames",
        }
    ):
        raise ValueError("candidate point-only protocol differs")
    if "scannet" in _stable_bytes(
        point_config, "CA-only point inference config"
    ).decode("utf-8").lower():
        raise ValueError("point inference config names ScanNet")

    pipeline = _mapping(config.get("pipeline"), "pipeline")
    _exact_keys(pipeline, (*STAGE_ORDER, "stage_order"), "pipeline")
    if tuple(pipeline.get("stage_order", ())) != STAGE_ORDER:
        raise ValueError("P/O/E/M stage order differs")
    p_stage = _mapping(pipeline.get("P_role_anchor_free"), "P stage")
    o_stage = _mapping(pipeline.get("O_cpu_overlay"), "O stage")
    e_stage = _mapping(pipeline.get("E_candidate_native"), "E stage")
    m_stage = _mapping(pipeline.get("M_exact80_manifest"), "M stage")
    if (
        p_stage != {
            "device": "gpu", "exact_scenes_per_role": 20,
            "create_only": True, "anchor_free": True,
            "ground_truth_access": False,
            "schema": "boxfusion.ca1m_tr3d_e961_xfit_r2_anchor_free_proposal.v5",
        }
        or o_stage != {
            "device": "cpu", "geometry_source": "sealed_final_base_prediction",
            "anchor_score_source": "ca1m_native_b6_final_base_all_fold_oof_row_scores_v2",
            "preserve_anchor_row_order": True, "deploy_scores_used": False,
            "ground_truth_access": False,
            "schema": "boxfusion.ca1m_tr3d_e961_xfit_r2_cpu_overlay.v5",
        }
        or e_stage != {
            "device": "cpu",
            "candidate_native_evidence": "recomputed_from_processed_train_rgbd",
            "anchor_native_evidence": "sealed_ca_native_b6_v2_observer_rows",
            "feature_schema": "boxfusion.ca1m_tr3d_xfit_r2_candidate_evidence.v1",
            "ground_truth_access": False,
        }
        or m_stage != {
            "device": "cpu", "fit_scene_count": 60, "fit_folds": [2, 3, 4],
            "reused_dev_scene_count": 20, "reused_dev_folds": [0],
            "each_scene_detector_excludes_scene": True,
            "each_anchor_score_model_excludes_scene": True,
            "schema": COMBINED_COLLECTION_SCHEMA,
        }
    ):
        raise ValueError("P/O/E/M science or isolation contract differs")

    integrity = _mapping(config.get("integrity"), "integrity")
    if integrity != {
        "create_only": True, "no_symlinks": True,
        "hash_every_input_and_output": True,
        "inode_stable_during_read": True, "finite_arrays": True,
        "exact_directory_inventory": True,
        "canonical_scene_row_identity": True,
        "cross_validate_scene_fold_producer": True,
        "failure_action": "stop_before_creating_any_output",
    }:
        raise ValueError("create-only/hash/inode/finite identity contract differs")

    outputs = _mapping(config.get("outputs"), "outputs")
    _exact_keys(outputs, (
        "root", "proposal_roots", "overlay_root", "candidate_evidence_root",
        "normalized_receipt_root", "role_collection_root", "combined_manifest",
    ), "outputs")
    output_root = Path(str(outputs.get("root", "")))
    expected_root = Path("/extra/ZhaoX") / NAMESPACE
    if output_root != expected_root:
        raise ValueError("output root is not the new v5 namespace")
    proposal_roots = _mapping(outputs.get("proposal_roots"), "proposal roots")
    if tuple(proposal_roots) != ROLE_ORDER:
        raise ValueError("proposal output roles differ")
    for role in ROLE_ORDER:
        expected = output_root / "P" / role
        if _guard_output(Path(str(proposal_roots[role])), output_root, role) != expected:
            raise ValueError(f"{role}: proposal output path differs")
    expected_outputs = {
        "overlay_root": output_root / "O",
        "candidate_evidence_root": output_root / "E",
        "normalized_receipt_root": output_root / "receipts",
        "role_collection_root": output_root / "role_collections",
        "combined_manifest": output_root / "M/CANDIDATE_COLLECTION_EXACT80.json",
    }
    for name, expected in expected_outputs.items():
        if _guard_output(Path(str(outputs[name])), output_root, name) != expected:
            raise ValueError(f"{name} path differs")

    forbidden = _mapping(config.get("forbidden_reuse"), "forbidden_reuse")
    _exact_keys(forbidden, (
        "scannet_weights_or_artifacts", "terminal_gate_v1_v4_artifacts",
        "old_ca_terminal_proposals", "old_ca_terminal_overlays",
        "old_ca_terminal_candidate_evidence", "deploy_b6_scores",
        "forbidden_formal_input_path_tokens",
    ), "forbidden_reuse")
    if any(forbidden.get(key) is not False for key in (
        "scannet_weights_or_artifacts", "terminal_gate_v1_v4_artifacts",
        "old_ca_terminal_proposals", "old_ca_terminal_overlays",
        "old_ca_terminal_candidate_evidence", "deploy_b6_scores",
    )):
        raise ValueError("forbidden reuse must remain false")
    tokens = tuple(str(value).lower() for value in forbidden.get("forbidden_formal_input_path_tokens", ()))
    if not tokens:
        raise ValueError("forbidden formal input path tokens are missing")
    for formal_path in _formal_input_paths(config):
        lowered = formal_path.lower()
        for token in tokens:
            if token in lowered:
                raise ValueError(f"formal input path reuses forbidden artifact token {token}")

    return source, config


def static_report(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    source, config = validate_static_config(path)
    roles = config["scene_contract"]["roles"]
    return {
        "schema": "boxfusion.ca1m_tr3d_e961_terminal_inputs_static_preflight.v1",
        "ok": True,
        "mode": "static_contract",
        "namespace": NAMESPACE,
        "state": PENDING_STATE,
        "config": {"path": str(source), "sha256": sha256_file(source)},
        "role_topology": {
            role: {
                "train_folds": roles[role]["detector_train_folds"],
                "output_fold": roles[role]["candidate_output_fold"],
                "train_scenes": 1001,
                "candidate_scenes": 20,
                "receipt_state": "pending",
            }
            for role in ROLE_ORDER
        },
        "fit_scene_count": 60,
        "reused_dev_scene_count": 20,
        "anchor_geometry_source": "sealed_final_base_prediction_rows",
        "anchor_score_source": "ca_native_b6_v2_all_fold_oof_only",
        "candidate_native_evidence": "fresh_cpu_recompute_required",
        "runtime_ready": False,
        "four_success_receipts_opened": False,
        "checkpoint_opened": False,
        "candidate_or_ground_truth_artifact_opened": False,
        "fold1_or_official_validation_path_present": False,
        "gpu_started": False,
        "output_created": False,
    }


def validate_operational_ready(path: Path = DEFAULT_CONFIG) -> None:
    """Fail before receipt/checkpoint/output resolution for the pending revision."""

    _, config = validate_static_config(path)
    if config.get("static_contract_only") is True or config.get("authorizations") != {
        **AUTHORIZATIONS,
        "gpu_proposal_collection": True,
        "cpu_overlay": True,
        "candidate_native_evidence": True,
        "candidate_collection_seal": True,
    }:
        raise PendingE961InputsError(
            "pending four authoritative E961 R2 success receipts and a separately "
            "sealed ready/run authorization"
        )
    raise AssertionError("pending config unexpectedly became operational")


__all__ = [
    "ACCESS", "AUTHORIZATIONS", "CONFIG_SCHEMA", "DEFAULT_CONFIG",
    "NAMESPACE", "PENDING_STATE", "PendingE961InputsError", "ROLE_ORDER",
    "ROLE_SPECS", "STAGE_ORDER", "sha256_file", "static_report",
    "validate_operational_ready", "validate_static_config",
]
