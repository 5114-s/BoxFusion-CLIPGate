"""Final, R4-only execution boundary for the CA-1M terminal gate v5.

The generic :mod:`ca1m_tr3d_terminal_gate_v5` module contains the numerical
dataset, three-head learner, OOF selection and geometry materializer.  This
revision adds the missing production boundary without changing that frozen
generic dependency:

* the sole candidate input is the canonical E961 terminal-inputs R4 exact80
  wrapper and collection;
* the R4 preregistration/ready/authorization/bundle commit chain is reopened
  and hash checked before an annotation loader or output path is reachable;
* one scientific preregistration freezes the exact double-OOF topology,
  learning heads, threshold grid, AP protocol and implementation bytes;
* a last-published run authorization binds that preregistration and an opaque
  CA-train annotation inventory; and
* fold 0 remains a reused-dev continuation diagnostic.  Fold 1 and official
  validation are deliberately absent from the runnable API.

Importing this module is side-effect free.  In particular it does not import
the evolving R4 producer implementation, inspect R4 outputs, open annotations,
create directories, or initialize CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import ca1m_tr3d_terminal_gate_v5 as gate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PENDING_CONFIG = ROOT / "config/ca1m_tr3d_terminal_gate_v5_final_pending.json"
MANIFEST_ROOT = ROOT / "manifests/ca1m_tr3d_terminal_gate_v5_final"
PREREGISTRATION_PATH = MANIFEST_ROOT / "PREREGISTRATION.json"
READY_CONFIG_PATH = MANIFEST_ROOT / "READY_CONFIG.json"
RUN_AUTHORIZATION_PATH = MANIFEST_ROOT / "RUN_AUTHORIZATION.json"

R4_NAMESPACE = "ca1m_tr3d_e961_terminal_inputs_xfit_r2_v5_r4"
R4_ROOT = Path("/extra/ZhaoX") / R4_NAMESPACE
R4_COLLECTION_PATH = R4_ROOT / "manifests/CANDIDATE_COLLECTION_EXACT80.json"
R4_RECEIPT_PATH = R4_ROOT / "manifests/M_EXACT80_R4_RECEIPT.json"
R4_R2_EXECUTION_RECEIPT_PATH = R4_ROOT / "manifests/M_EXACT80_R2_RECEIPT.json"
R4_MANIFEST_ROOT = ROOT / "manifests/ca1m_tr3d_e961_terminal_inputs_v5_r4"
R4_PREREGISTRATION_PATH = R4_MANIFEST_ROOT / "PREREGISTRATION.json"
R4_READY_CONFIG_PATH = R4_MANIFEST_ROOT / "READY_CONFIG.json"
R4_RUN_AUTHORIZATION_PATH = R4_MANIFEST_ROOT / "RUN_AUTHORIZATION.json"
R4_AUTHORIZATION_BUNDLE_PATH = R4_MANIFEST_ROOT / "AUTHORIZATION_BUNDLE.json"

PENDING_SCHEMA = "boxfusion.ca1m_tr3d_terminal_gate_pending_config.v5.final"
PREREGISTRATION_SCHEMA = "boxfusion.ca1m_tr3d_terminal_gate_preregistration.v5.final"
READY_SCHEMA = "boxfusion.ca1m_tr3d_terminal_gate_ready_config.v5.final"
AUTHORIZATION_SCHEMA = "boxfusion.ca1m_tr3d_terminal_gate_run_authorization.v5.final"
RUN_SCHEMA = "boxfusion.ca1m_tr3d_terminal_gate_run.v5.final"
STOP_SCHEMA = "boxfusion.ca1m_tr3d_terminal_gate_stop.v5.final"
MATERIALIZATION_COLLECTION_SCHEMA = (
    "boxfusion.ca1m_tr3d_geometry_materialization_collection.v5.final"
)
GT_INVENTORY_SCHEMA = "boxfusion.ca1m_tr3d_benefit_gate_gt_shadow_inventory.v1"

R4_RECEIPT_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_exact80_receipt.v5.r4"
R4_R2_EXECUTION_RECEIPT_SCHEMA = (
    "boxfusion.ca1m_tr3d_e961_terminal_inputs_exact80_receipt.v5.r2"
)
R4_CONFIG_SCHEMA = "boxfusion.ca1m_tr3d_e961_terminal_inputs_config.v5.r4"
R4_PREREGISTRATION_SCHEMA = (
    "boxfusion.ca1m_tr3d_e961_terminal_inputs_preregistration.v5.r4"
)
R4_AUTHORIZATION_SCHEMA = (
    "boxfusion.ca1m_tr3d_e961_terminal_inputs_run_authorization.v5.r4"
)
R4_BUNDLE_SCHEMA = (
    "boxfusion.ca1m_tr3d_e961_terminal_inputs_authorization_bundle.v5.r4"
)
OUTER_RUN_SCHEMA = "boxfusion.tr3d.ca1m_e961_outer_train_run.r2"
INNER_RUN_SCHEMA = "boxfusion.tr3d.ca1m_e961_inner_train_run.r2"

NAMESPACE = "ca1m_tr3d_terminal_gate_v5_final"
RUNTIME_ROOT = Path("/extra/ZhaoX") / NAMESPACE

IMPLEMENTATION_PATHS = {
    "generic_gate_core": ROOT / "boxfusion/ca1m_tr3d_terminal_gate_v5.py",
    "final_gate_boundary": ROOT / "boxfusion/ca1m_tr3d_terminal_gate_v5_final.py",
    "static_preflight": ROOT / "tools/preflight_ca1m_tr3d_terminal_gate_v5_final.py",
    "scientific_sealer": ROOT / "tools/seal_ca1m_tr3d_terminal_gate_v5_final.py",
    "runner": ROOT / "tools/run_ca1m_tr3d_terminal_gate_v5_final.py",
    "regression_tests": ROOT / "tests/test_ca1m_tr3d_terminal_gate_v5_final.py",
}

OUTPUT_PATHS = {
    "fit_dataset": RUNTIME_ROOT / "datasets/fold234_fit60.npz",
    "fit_dataset_manifest": RUNTIME_ROOT / "datasets/fold234_fit60.manifest.json",
    "oof_predictions": RUNTIME_ROOT / "results/fold234_scene_grouped_oof.npz",
    "threshold_receipt": RUNTIME_ROOT / "results/fold234_threshold_receipt.json",
    "exploratory_policy": RUNTIME_ROOT / "models/gate_v5.non_deployable.json",
    "fold0_dataset": RUNTIME_ROOT / "datasets/fold0_reused_dev20.npz",
    "fold0_dataset_manifest": RUNTIME_ROOT / "datasets/fold0_reused_dev20.manifest.json",
    "fold0_report": RUNTIME_ROOT / "reports/fold0_reused_dev_diagnostic.json",
    "materialization_root": RUNTIME_ROOT / "materialized_fold0",
    "materialization_manifest": RUNTIME_ROOT / "reports/fold0_materialization_manifest.json",
    "stop_receipt": RUNTIME_ROOT / "reports/STOP.json",
    "run_receipt": RUNTIME_ROOT / "reports/RUN_RECEIPT.json",
}

_SHA = re.compile(r"^[0-9a-f]{64}$")
_SCENE = re.compile(r"^[0-9]{8}$")
_FORBIDDEN_CANDIDATE_PATH_TOKENS = (
    "scannet",
    "m_exact80_r2_receipt",
    "ca1m_tr3d_benefit_gate_final_base_v4",
    "ca1m_tr3d_terminal_ca_native_train100_v4",
    "ca1m_tr3d_benefit_final_base_v4",
    "ca1m_fg_scratch_seed0_fp32_gb16_v1",
)


class PendingR4Inputs(PermissionError):
    """Raised before GT/output when the canonical R4 commit is unavailable."""


class FinalGateProtocolError(RuntimeError):
    """A final-gate provenance or execution invariant was violated."""


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, name: str) -> str:
    result = str(value)
    if _SHA.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return result


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _assert_no_symlink_chain(path: Path, name: str) -> None:
    absolute = _absolute_lexical(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            raise FileNotFoundError(f"missing {name}: {absolute}") from None
        if stat.S_ISLNK(mode):
            raise ValueError(f"{name} path contains a symlink: {current}")


def _identity(info: os.stat_result) -> dict[str, int]:
    return {
        "st_dev": int(info.st_dev), "st_ino": int(info.st_ino),
        "st_size": int(info.st_size), "st_mtime_ns": int(info.st_mtime_ns),
        "st_nlink": int(info.st_nlink),
    }


def stable_bytes(
    path: Path, name: str, *, immutable: bool = True, allow_empty: bool = False,
) -> tuple[Path, bytes, dict[str, int]]:
    """Read one non-symlink file while pinning and rechecking its inode."""

    source = _absolute_lexical(path)
    _assert_no_symlink_chain(source, name)
    before_path = os.stat(source, follow_symlinks=False)
    if not stat.S_ISREG(before_path.st_mode):
        raise ValueError(f"{name} must be a regular file: {source}")
    if immutable and before_path.st_mode & 0o222:
        raise ValueError(f"{name} must be read-only: {source}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        before_fd = os.fstat(descriptor)
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            blocks.append(block)
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = os.stat(source, follow_symlinks=False)
    identities = (
        _identity(before_path), _identity(before_fd),
        _identity(after_fd), _identity(after_path),
    )
    if len({tuple(value.items()) for value in identities}) != 1:
        raise RuntimeError(f"{name} inode changed while reading: {source}")
    payload = b"".join(blocks)
    if len(payload) != before_fd.st_size or (not payload and not allow_empty):
        raise ValueError(f"{name} is empty or size-inconsistent: {source}")
    return source, payload, identities[0]


def stable_json(
    path: Path, name: str, *, schema: str | None = None,
    immutable: bool = True,
) -> tuple[Path, dict[str, Any], bytes, dict[str, int]]:
    source, data, identity = stable_bytes(path, name, immutable=immutable)
    try:
        value = json.loads(data)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    if schema is not None and value.get("schema") != schema:
        raise ValueError(f"{name} schema differs")
    return source, value, data, identity


def artifact_record(
    path: Path, name: str, *, schema: str | None = None,
    immutable: bool = True,
) -> dict[str, Any]:
    source, data, identity = stable_bytes(path, name, immutable=immutable)
    result: dict[str, Any] = {
        "path": os.fspath(source), "sha256": sha256_bytes(data),
        "identity": identity,
    }
    if schema is not None:
        result["schema"] = schema
    return result


def validate_artifact_record(
    value: Any, name: str, *, schema: str | None = None,
    canonical_path: Path | None = None, require_identity: bool = False,
    immutable: bool = True,
) -> tuple[Path, bytes, dict[str, int]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} record must be an object")
    path = _absolute_lexical(Path(str(value.get("path", ""))))
    if canonical_path is not None and path != _absolute_lexical(canonical_path):
        raise ValueError(f"{name} canonical path differs")
    source, data, identity = stable_bytes(path, name, immutable=immutable)
    if sha256_bytes(data) != _sha(value.get("sha256"), f"{name} SHA256"):
        raise ValueError(f"{name} SHA256 differs")
    if schema is not None and value.get("schema") != schema:
        raise ValueError(f"{name} record schema differs")
    if require_identity and value.get("identity") != identity:
        raise ValueError(f"{name} inode identity differs")
    return source, data, identity


def _record_json(
    value: Any, name: str, *, schema: str, canonical_path: Path | None = None,
    require_identity: bool = False,
) -> tuple[Path, dict[str, Any], bytes, dict[str, int]]:
    path, data, identity = validate_artifact_record(
        value, name, schema=schema, canonical_path=canonical_path,
        require_identity=require_identity,
    )
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not JSON") from error
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise ValueError(f"{name} payload schema differs")
    return path, payload, data, identity


def _path_fields(value: Any, location: str = "root") -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if isinstance(child, str) and any(
                token in key.lower() for token in ("path", "root", "file")
            ):
                result.append((child_location, child))
            result.extend(_path_fields(child, child_location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(_path_fields(child, f"{location}[{index}]"))
    return result


def _contains_scalar(value: Any, key: str, expected: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get(key) == expected:
            return True
        return any(_contains_scalar(child, key, expected) for child in value.values())
    if isinstance(value, list):
        return any(_contains_scalar(child, key, expected) for child in value)
    return False


def science_contract() -> dict[str, Any]:
    """Return the complete frozen scientific contract used by the sealer."""

    return {
        "candidate_input": {
            "only_source": "canonical_e961_terminal_inputs_r4_exact80_wrapper",
            "scene_count": 80, "fit_scene_count": 60,
            "fold0_scene_count": 20, "legacy_candidate_or_policy_reuse": False,
            "each_fit_scene_detector_excludes_scene": True,
            "old60_detector_receipts_allowed": False,
        },
        "detector_double_oof": {
            "inner_holdout2": {"train_folds": [3, 4], "output_fold": 2},
            "inner_holdout3": {"train_folds": [2, 4], "output_fold": 3},
            "inner_holdout4": {"train_folds": [2, 3], "output_fold": 4},
            "outer_dev": {"train_folds": [2, 3, 4], "output_fold": 0},
        },
        "anchor_scores": {
            "source": gate.ANCHOR_SCORE_SOURCE,
            "all_fold_oof_each_row_model_excludes_scene": True,
            "deploy_scores_allowed": False,
        },
        "gate_double_oof": {
            "fit_folds": [2, 3, 4],
            "gate_holdout2_train_folds": [3, 4],
            "gate_holdout3_train_folds": [2, 4],
            "gate_holdout4_train_folds": [2, 3],
            "threshold_source": "fold234_scene_grouped_gate_oof_only",
            "fold0_used_for_fit_or_selection": False,
        },
        "model": {
            "family": "three_head_low_capacity_linear_v1",
            "heads": [
                "continuous_candidate_iou_huber",
                "strict_candidate_iou_gt_0.50_logistic_calibration",
                "within_anchor_pairwise_same_gt_benefit",
            ],
            "pairwise_group_keys": ["scene_id", "anchor_index"],
            "target_switch_is_harm": True,
            "raw_score_penalty_multiplier": 4.0,
            "feature_names": list(gate.FEATURE_NAMES),
        },
        "selection": {
            "candidate_iou_grid": list(gate.IOU_GRID),
            "same_gt_gain_grid": list(gate.GAIN_GRID),
            "strict_iou50_probability_grid": list(gate.PROB_GRID),
            "max_replacements_per_scene": gate.MAX_REPLACEMENTS_PER_SCENE,
            "oof_safety_gate": dict(gate.OOF_SAFETY_GATE),
            "fold0_retuning": False,
        },
        "metric": {
            "box_geometry": "world_enclosing_aabb_from_8_corners",
            "ranking": "global_score_descending_numpy_default_argsort",
            "matching": "scene_local_duplicate_aware",
            "iou_comparison": "strict_greater_than",
            "thresholds": list(gate.IOU_THRESHOLDS),
            "scores_preserved": True,
        },
        "materialization": {
            "geometry_only": True, "score_values_preserved": True,
            "row_order_preserved": True, "row_count_preserved": True,
            "exploratory_noncanonical": True,
        },
        "isolation": {
            "fold0_role": "reused_dev_continuation_diagnostic_only",
            "fold1_path_or_loader_present": False,
            "official_validation_path_or_loader_present": False,
            "scannet_weight_or_artifact_access": False,
            "policy_activation_authorized": False,
        },
    }


def _independent_voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    padded_recall = np.concatenate(([0.0], recall, [1.0]))
    padded_precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(padded_precision) - 1, 0, -1):
        padded_precision[index - 1] = max(
            padded_precision[index - 1], padded_precision[index]
        )
    changes = np.flatnonzero(padded_recall[1:] != padded_recall[:-1])
    return float(np.sum(
        (padded_recall[changes + 1] - padded_recall[changes])
        * padded_precision[changes + 1]
    ))


def ap_parity_fixture() -> dict[str, Any]:
    """Cross-check the imported metric with an independent frozen reference."""

    scenes = np.asarray(("00000001", "00000002", "00000001", "00000002", "00000001"))
    scores = np.asarray((0.99, 0.95, 0.90, 0.80, 0.70), np.float64)
    best_iou = np.asarray((0.15, 0.90, 0.80, 0.50, 0.95), np.float64)
    best_gt = np.asarray((0, 0, 0, 1, 1), np.int64)
    positives = 4
    imported = gate.official_ca_ap(
        scene_ids=scenes, scores=scores, best_iou=best_iou,
        best_gt=best_gt, ground_truth_count=positives,
    )
    order = np.argsort(-scores)
    reference: dict[str, dict[str, float | int]] = {}
    for threshold in gate.IOU_THRESHOLDS:
        tp = np.zeros(len(order), np.float64)
        fp = np.zeros(len(order), np.float64)
        detected: set[tuple[str, int]] = set()
        for rank, row in enumerate(order.tolist()):
            key = (str(scenes[row]), int(best_gt[row]))
            if best_iou[row] > threshold and best_gt[row] >= 0 and key not in detected:
                tp[rank] = 1.0
                detected.add(key)
            else:
                fp[rank] = 1.0
        cumulative_tp = np.cumsum(tp)
        cumulative_fp = np.cumsum(fp)
        recall = cumulative_tp / (positives + 1.0e-6)
        precision = cumulative_tp / np.maximum(
            cumulative_tp + cumulative_fp, np.finfo(np.float64).eps
        )
        key = f"iou_{threshold:.2f}"
        final_tp = int(cumulative_tp[-1])
        reference[key] = {
            "ap": _independent_voc_ap(recall, precision),
            "precision": float(precision[-1]), "recall": float(recall[-1]),
            "tp": final_tp, "fp": int(cumulative_fp[-1]),
            "fn": positives - final_tp,
        }
    for key in reference:
        for field in ("ap", "precision", "recall"):
            if not math.isclose(
                float(imported[key][field]), float(reference[key][field]),
                rel_tol=0.0, abs_tol=1.0e-15,
            ):
                raise RuntimeError(f"official AP parity differs: {key}.{field}")
        for field in ("tp", "fp", "fn"):
            if imported[key][field] != reference[key][field]:
                raise RuntimeError(f"official AP parity differs: {key}.{field}")
    fixture = {
        "scene_ids": scenes.tolist(), "scores": scores.tolist(),
        "best_iou": best_iou.tolist(), "best_gt": best_gt.tolist(),
        "ground_truth_count": positives,
    }
    return {
        "schema": "boxfusion.ca1m_tr3d_terminal_gate_ap_parity.v5.final",
        "pass": True, "fixture_sha256": sha256_bytes(canonical_json(fixture)),
        "reference": reference, "imported": imported,
        "protocol": science_contract()["metric"],
    }


@dataclass(frozen=True)
class R4Exact80Binding:
    wrapper_path: Path
    wrapper_sha256: str
    wrapper_identity: Mapping[str, int]
    collection_path: Path
    collection_sha256: str
    collection_identity: Mapping[str, int]
    authorization_commit_id: str
    scene_folds: Mapping[str, int]
    upstream_records: Mapping[str, Mapping[str, Any]]


def _require_under(path: Path, root: Path, name: str) -> None:
    try:
        _absolute_lexical(path).relative_to(_absolute_lexical(root))
    except ValueError:
        raise ValueError(f"{name} is outside canonical R4 root") from None


def _validate_e961_producer_lineage(
    collection: gate.CandidateCollectionV5,
    r4_authorization: Mapping[str, Any],
) -> None:
    auth_roles = {
        str(row.get("role")): row for row in r4_authorization.get("roles", ())
        if isinstance(row, Mapping)
    }
    if tuple(auth_roles) != tuple(gate.ROLE_SPECS):
        # R4 authorizations use outer-first order, but exact membership is the
        # security property.  Do not use order as a hidden compatibility path.
        if set(auth_roles) != set(gate.ROLE_SPECS):
            raise ValueError("R4 authorization producer role set differs")
    for role in gate.ROLE_SPECS:
        role_path = collection.role_manifests[role]
        _require_under(role_path, R4_ROOT / "manifests", f"{role} role manifest")
        _, role_payload, _, _ = stable_json(
            role_path, f"{role} R4 role manifest", schema=gate.ROLE_COLLECTION_SCHEMA
        )
        normalized_record = role_payload.get("role_receipt") or {}
        normalized_path, normalized, _, _ = _record_json(
            normalized_record, f"{role} normalized receipt",
            schema=gate.ROLE_RECEIPT_SCHEMA,
        )
        _require_under(normalized_path, R4_ROOT / "normalized_receipts", f"{role} normalized receipt")
        adapter_path, adapter, _, _ = _record_json(
            normalized.get("source_training_receipt"), f"{role} producer adapter",
            schema="boxfusion.ca1m_tr3d_e961_verified_receipt_adapter.v2",
        )
        _require_under(adapter_path, R4_ROOT / "normalized_receipts", f"{role} producer adapter")
        source_record = adapter.get("source_producer_receipt") or {}
        expected_schema = OUTER_RUN_SCHEMA if role == "outer_dev" else INNER_RUN_SCHEMA
        source_path, source_payload, _, _ = _record_json(
            source_record, f"{role} raw E961 receipt", schema=expected_schema
        )
        if "e961" not in os.fspath(source_path).lower():
            raise ValueError(f"{role} producer receipt is not from E961")
        auth_row = auth_roles.get(role) or {}
        if (
            auth_row.get("receipt_path") != os.fspath(source_path)
            or auth_row.get("receipt_sha256") != sha256_file(source_path)
            or source_payload.get("status") != "success"
            or source_payload.get("exit_code") != 0
            or not _contains_scalar(source_payload, "train_scenes", 1001)
            or not _contains_scalar(source_payload, "optimizer_updates", 11268)
            or not _contains_scalar(source_payload, "initialization", "random_scratch_ca_only")
            or (adapter.get("checkpoint") or {}).get("sha256")
            != auth_row.get("checkpoint_sha256")
        ):
            raise ValueError(f"{role} is not the authorized CA-only E961 exact1001 producer")
        for location, raw_path in _path_fields({
            "normalized": normalized, "adapter": adapter,
        }, f"{role}"):
            lowered = raw_path.lower()
            if any(token in lowered for token in _FORBIDDEN_CANDIDATE_PATH_TOKENS):
                raise ValueError(f"{location} references forbidden legacy candidate path")


def load_r4_exact80_binding(
    receipt_path: Path = R4_RECEIPT_PATH,
) -> R4Exact80Binding:
    """Open the canonical R4 commit chain and exact80 collection, GT-free."""

    requested = _absolute_lexical(receipt_path)
    if requested != _absolute_lexical(R4_RECEIPT_PATH):
        raise ValueError("terminal gate accepts only the canonical R4 exact80 receipt")
    wrapper_path, wrapper, wrapper_bytes, wrapper_identity = stable_json(
        requested, "R4 exact80 wrapper", schema=R4_RECEIPT_SCHEMA
    )
    expected_keys = {
        "schema", "complete", "create_only", "namespace",
        "fit_scene_count", "fit_folds", "reused_dev_scene_count",
        "reused_dev_folds", "scene_count", "each_scene_detector_excludes_scene",
        "b6_score_source", "ground_truth_access", "fold1_access",
        "official_validation_access", "legacy_v1_v4_candidate_or_policy_reused",
        "r4_preregistration", "r4_ready_config", "r4_run_authorization",
        "r4_authorization_bundle", "authorization_commit_id",
        "candidate_collection", "r2_execution_receipt",
    }
    if set(wrapper) != expected_keys:
        raise ValueError("R4 exact80 wrapper key set differs")
    if (
        wrapper.get("complete") is not True
        or wrapper.get("create_only") is not True
        or wrapper.get("namespace") != R4_NAMESPACE
        or wrapper.get("fit_scene_count") != 60
        or wrapper.get("fit_folds") != [2, 3, 4]
        or wrapper.get("reused_dev_scene_count") != 20
        or wrapper.get("reused_dev_folds") != [0]
        or wrapper.get("scene_count") != 80
        or wrapper.get("each_scene_detector_excludes_scene") is not True
        or wrapper.get("b6_score_source")
        != "all_fold_oof_each_row_model_excludes_scene"
        or wrapper.get("ground_truth_access") is not False
        or wrapper.get("fold1_access") is not False
        or wrapper.get("official_validation_access") is not False
        or wrapper.get("legacy_v1_v4_candidate_or_policy_reused") is not False
    ):
        raise ValueError("R4 exact80 wrapper science/isolation contract differs")
    commit_id = _sha(wrapper.get("authorization_commit_id"), "R4 commit id")
    r2_record = wrapper.get("r2_execution_receipt") or {}
    if r2_record.get("operational_authority") is not False:
        raise ValueError("R2 execution receipt must be explicitly non-authoritative")
    r2_path, _, r2_data, r2_identity = _record_json(
        {key: r2_record.get(key) for key in ("path", "sha256", "schema")},
        "R2 internal execution receipt",
        schema=R4_R2_EXECUTION_RECEIPT_SCHEMA,
        canonical_path=R4_R2_EXECUTION_RECEIPT_PATH,
    )
    _require_under(r2_path, R4_ROOT / "manifests", "R2 internal execution receipt")
    upstream_specs = {
        "r4_preregistration": (R4_PREREGISTRATION_SCHEMA, R4_PREREGISTRATION_PATH),
        "r4_ready_config": (R4_CONFIG_SCHEMA, R4_READY_CONFIG_PATH),
        "r4_run_authorization": (R4_AUTHORIZATION_SCHEMA, R4_RUN_AUTHORIZATION_PATH),
        "r4_authorization_bundle": (R4_BUNDLE_SCHEMA, R4_AUTHORIZATION_BUNDLE_PATH),
    }
    upstream: dict[str, dict[str, Any]] = {}
    for key, (schema, canonical_path) in upstream_specs.items():
        path, payload, data, identity = _record_json(
            wrapper[key], key, schema=schema, canonical_path=canonical_path
        )
        upstream[key] = {
            "path": os.fspath(path), "sha256": sha256_bytes(data),
            "schema": schema, "identity": identity, "payload": payload,
        }
    prereg = upstream["r4_preregistration"]["payload"]
    ready = upstream["r4_ready_config"]["payload"]
    authorization = upstream["r4_run_authorization"]["payload"]
    bundle = upstream["r4_authorization_bundle"]["payload"]
    if (
        prereg.get("complete") is not True
        or prereg.get("namespace") != R4_NAMESPACE
        or authorization.get("complete") is not True
        or authorization.get("create_only") is not True
        or authorization.get("namespace") != R4_NAMESPACE
        or authorization.get("commit_id") != commit_id
        or authorization.get("ground_truth_access") is not False
        or authorization.get("fold1_access") is not False
        or authorization.get("official_validation_access") is not False
        or bundle.get("complete") is not True
        or bundle.get("create_only") is not True
        or bundle.get("namespace") != R4_NAMESPACE
        or bundle.get("commit_id") != commit_id
        or bundle.get("commit_role") != "last_published_unique_operational_gate"
        or (bundle.get("ready_config") or {}).get("sha256")
        != upstream["r4_ready_config"]["sha256"]
        or (bundle.get("run_authorization") or {}).get("sha256")
        != upstream["r4_run_authorization"]["sha256"]
        or (ready.get("run_authorization") or {}).get("state")
        != "committed_by_bundle"
        or (ready.get("run_authorization") or {}).get("commit_id") != commit_id
        or (ready.get("run_authorization") or {}).get("path")
        != os.fspath(R4_AUTHORIZATION_BUNDLE_PATH)
    ):
        raise ValueError("R4 ready/auth/bundle commit chain differs")
    collection_path, collection_data, collection_identity = validate_artifact_record(
        wrapper.get("candidate_collection"), "R4 candidate collection",
        schema=gate.COLLECTION_SCHEMA, canonical_path=R4_COLLECTION_PATH,
    )
    collection = gate.load_candidate_collection_v5(collection_path)
    if collection.payload.get("namespace") != gate.NAMESPACE:
        raise ValueError("R4 collection generic-v5 namespace differs")
    for row in collection.scenes.values():
        evidence_path = Path(str(row.get("path", "")))
        _require_under(evidence_path, R4_ROOT / "evidence", "R4 candidate evidence")
        if int(row["fold_id"]) in tuple(row["producer_train_folds"]):
            raise ValueError("R4 detector candidate is in-sample")
    _validate_e961_producer_lineage(collection, authorization)
    return R4Exact80Binding(
        wrapper_path=wrapper_path, wrapper_sha256=sha256_bytes(wrapper_bytes),
        wrapper_identity=wrapper_identity, collection_path=collection_path,
        collection_sha256=sha256_bytes(collection_data),
        collection_identity=collection_identity,
        authorization_commit_id=commit_id,
        scene_folds={scene: int(row["fold_id"]) for scene, row in collection.scenes.items()},
        upstream_records={
            **{
                key: {field: value for field, value in record.items() if field != "payload"}
                for key, record in upstream.items()
            },
            "r2_internal_execution_receipt": {
                "path": os.fspath(r2_path), "sha256": sha256_bytes(r2_data),
                "schema": R4_R2_EXECUTION_RECEIPT_SCHEMA,
                "identity": r2_identity, "operational_authority": False,
            },
        },
    )


def validate_pending_config(
    path: Path = DEFAULT_PENDING_CONFIG,
) -> tuple[Path, dict[str, Any]]:
    source, cfg, _, _ = stable_json(path, "terminal gate final pending config", immutable=False)
    if set(cfg) != {
        "schema", "namespace", "state", "authorizations", "access",
        "future_r4_input", "annotation_inventory", "science_contract",
        "implementation", "outputs", "formal_artifacts",
    }:
        raise ValueError("terminal gate final pending config keys differ")
    if (
        cfg.get("schema") != PENDING_SCHEMA
        or cfg.get("namespace") != NAMESPACE
        or cfg.get("state") != "pending_r4_exact80"
        or cfg.get("authorizations") != {
            "scientific_preregistration": False, "ground_truth_join": False,
            "gate_fit": False, "threshold_selection": False,
            "fold0_reused_dev_diagnostic": False,
            "geometry_materialization": False, "fold1": False,
            "official_validation": False, "policy_activation": False,
        }
        or cfg.get("access") != {
            "static_only": True, "candidate_opened": False,
            "ground_truth_access": False, "output_created": False,
            "gpu_started": False, "fold1_path_present": False,
            "official_validation_path_present": False,
        }
        or cfg.get("science_contract") != science_contract()
    ):
        raise ValueError("terminal gate final pending fail-close contract differs")
    expected_r4 = {
        "state": "pending", "path": os.fspath(R4_RECEIPT_PATH),
        "sha256": None, "schema": R4_RECEIPT_SCHEMA,
        "only_candidate_input": True,
    }
    if cfg.get("future_r4_input") != expected_r4:
        raise ValueError("pending R4 exact80 input differs")
    if cfg.get("annotation_inventory") != {
        "state": "pending", "path": None, "sha256": None,
        "schema": GT_INVENTORY_SCHEMA, "ca_train_annotations_only": True,
        "old_gate_dataset_or_policy_reuse": False,
    }:
        raise ValueError("pending annotation inventory differs")
    if cfg.get("implementation") != {
        key: os.fspath(value) for key, value in IMPLEMENTATION_PATHS.items()
    }:
        raise ValueError("pending final-gate implementation inventory differs")
    if cfg.get("outputs") != {key: os.fspath(value) for key, value in OUTPUT_PATHS.items()}:
        raise ValueError("pending final-gate output paths differ")
    if cfg.get("formal_artifacts") != {
        "preregistration": os.fspath(PREREGISTRATION_PATH),
        "ready_config": os.fspath(READY_CONFIG_PATH),
        "run_authorization": os.fspath(RUN_AUTHORIZATION_PATH),
    }:
        raise ValueError("pending formal artifact paths differ")
    return source, cfg


def static_preflight(
    path: Path = DEFAULT_PENDING_CONFIG,
) -> dict[str, Any]:
    source, _ = validate_pending_config(path)
    return {
        "schema": "boxfusion.ca1m_tr3d_terminal_gate_static_preflight.v5.final",
        "status": "PASS_STATIC_PENDING_R4", "config": os.fspath(source),
        "config_sha256": sha256_file(source), "runtime_ready": False,
        "r4_candidate_opened": False, "ground_truth_access": False,
        "output_created": False, "directory_created": False,
        "gpu_started": False, "fold1_access": False,
        "official_validation_access": False,
        "failure_action": "stop_before_gt_mkdir_or_output_until_r4_exact80_is_committed",
    }


def operational_preflight_pending(
    path: Path = DEFAULT_PENDING_CONFIG,
) -> None:
    """Fail before resolving R4 children, GT, output parents, or CUDA."""

    validate_pending_config(path)
    if not R4_RECEIPT_PATH.exists() and not R4_RECEIPT_PATH.is_symlink():
        raise PendingR4Inputs(
            "canonical R4 exact80 receipt is absent; no GT, output directory, "
            "formal preregistration, trainer, materializer, fold1, validation, or GPU was reached"
        )
    # Once the canonical commit exists, a caller may explicitly proceed to the
    # scientific sealer.  This function itself still creates no artifact.
    load_r4_exact80_binding(R4_RECEIPT_PATH)


def _publish_json_replay_safe(
    path: Path, payload: Mapping[str, Any], name: str,
) -> Path:
    """Create one immutable JSON or verify an identical crash replay."""

    data = canonical_json(payload)
    target = _absolute_lexical(path)
    if target.exists() or target.is_symlink():
        source, current, _ = stable_bytes(target, name)
        if current != data:
            raise FileExistsError(f"refusing differing existing {name}: {source}")
        return source
    return gate.write_bytes_create_only(target, data, name)


@dataclass(frozen=True)
class GTInventoryBinding:
    path: Path
    sha256: str
    identity: Mapping[str, int]
    scene_rows: Mapping[str, Mapping[str, Any]]


def validate_gt_inventory_metadata(
    path: Path, *, r4_binding: R4Exact80Binding,
) -> GTInventoryBinding:
    """Validate opaque CA-train annotation metadata without opening GT arrays."""

    source, value, data, identity = stable_json(
        path, "CA-train annotation inventory", schema=GT_INVENTORY_SCHEMA
    )
    rows = value.get("scenes")
    if not isinstance(rows, Mapping):
        raise ValueError("annotation inventory scenes must be an object")
    normalized = {str(scene): row for scene, row in rows.items() if isinstance(row, Mapping)}
    fold_counts = {
        fold: sum(int(row.get("fold_id", -1)) == fold for row in normalized.values())
        for fold in (0, 2, 3, 4)
    }
    if (
        value.get("complete") is not True
        or value.get("create_only") is not True
        or value.get("scene_count") != 80
        or value.get("fit_scene_count") != 60
        or value.get("threshold_dev_scene_count") != 20
        or value.get("fit_fold_ids") != [2, 3, 4]
        or value.get("threshold_dev_fold_ids") != [0]
        or value.get("locked_internal_scene_count_accessed") != 0
        or value.get("official_validation_comparable") is not False
        or value.get("validation_ground_truth_access") is not False
        or value.get("validation_prediction_access") is not False
        or value.get("train_only") is not True
        or value.get("gt_array_content_loaded") is not False
        or set(normalized) != set(r4_binding.scene_folds)
        or fold_counts != {0: 20, 2: 20, 3: 20, 4: 20}
    ):
        raise ValueError("CA-train annotation inventory partition/isolation differs")
    for scene, expected_fold in r4_binding.scene_folds.items():
        row = normalized[scene]
        box = row.get("box") or {}
        if (
            _SCENE.fullmatch(scene) is None
            or row.get("fold_id") != expected_fold
            or not isinstance(box, Mapping)
            or not Path(str(box.get("path", ""))).is_absolute()
            or _SHA.fullmatch(str(box.get("sha256", ""))) is None
        ):
            raise ValueError(f"{scene}: annotation inventory row differs")
    return GTInventoryBinding(
        source, sha256_bytes(data), identity, normalized,
    )


def _implementation_records() -> dict[str, Any]:
    return {
        key: artifact_record(path, f"final-gate implementation {key}", immutable=False)
        for key, path in IMPLEMENTATION_PATHS.items()
    }


def _r4_binding_record(binding: R4Exact80Binding) -> dict[str, Any]:
    return {
        "wrapper": {
            "path": os.fspath(binding.wrapper_path),
            "sha256": binding.wrapper_sha256,
            "schema": R4_RECEIPT_SCHEMA,
            "identity": dict(binding.wrapper_identity),
        },
        "candidate_collection": {
            "path": os.fspath(binding.collection_path),
            "sha256": binding.collection_sha256,
            "schema": gate.COLLECTION_SCHEMA,
            "identity": dict(binding.collection_identity),
        },
        "authorization_commit_id": binding.authorization_commit_id,
        "upstream_commit_chain": dict(binding.upstream_records),
    }


def build_preregistration_payload(
    *, gt_inventory_path: Path,
    pending_config_path: Path = DEFAULT_PENDING_CONFIG,
    r4_receipt_path: Path = R4_RECEIPT_PATH,
) -> dict[str, Any]:
    """Build the formal payload after R4, still before the first GT open."""

    pending_path, _ = validate_pending_config(pending_config_path)
    r4 = load_r4_exact80_binding(r4_receipt_path)
    inventory = validate_gt_inventory_metadata(gt_inventory_path, r4_binding=r4)
    parity = ap_parity_fixture()
    return {
        "schema": PREREGISTRATION_SCHEMA,
        "complete": True, "create_only": True, "static_science_only": True,
        "namespace": NAMESPACE,
        "sealed_after_r4_exact80_commit": True,
        "sealed_before_first_gt_array_open": True,
        "gt_array_content_access_at_seal": False,
        "fold0_gt_access_at_seal": False,
        "fold1_or_validation_access_at_seal": False,
        "gpu_started_at_seal": False,
        "pending_config": artifact_record(
            pending_path, "final-gate pending config", immutable=False
        ),
        "r4_exact80": _r4_binding_record(r4),
        "annotation_inventory": {
            "path": os.fspath(inventory.path), "sha256": inventory.sha256,
            "schema": GT_INVENTORY_SCHEMA, "identity": dict(inventory.identity),
            "metadata_only_at_seal": True,
            "only_scene_box_records_consumable": True,
            "old_gate_dataset_features_or_policy_consumable": False,
        },
        "science_contract": science_contract(),
        "official_ap_parity": parity,
        "implementation": _implementation_records(),
        "outputs": {key: os.fspath(value) for key, value in OUTPUT_PATHS.items()},
        "failure_actions": {
            "r4_missing_or_changed": "stop_before_gt_mkdir_output_or_gpu",
            "fold234_oof_gate_fail": "publish_stop_without_fold0_gt_or_materialization",
            "partial_output": "fail_closed_no_overwrite_or_resume",
        },
    }


def seal_scientific_preregistration(
    *, gt_inventory_path: Path,
    output_path: Path = PREREGISTRATION_PATH,
    pending_config_path: Path = DEFAULT_PENDING_CONFIG,
    r4_receipt_path: Path = R4_RECEIPT_PATH,
) -> Path:
    if _absolute_lexical(output_path) != _absolute_lexical(PREREGISTRATION_PATH):
        raise ValueError("final gate preregistration target is noncanonical")
    # All future inputs and science are validated before the writer is allowed
    # to create the manifest directory.
    payload = build_preregistration_payload(
        gt_inventory_path=gt_inventory_path,
        pending_config_path=pending_config_path,
        r4_receipt_path=r4_receipt_path,
    )
    if READY_CONFIG_PATH.exists() or RUN_AUTHORIZATION_PATH.exists():
        raise FileExistsError("cannot seal preregistration after ready/authorization output")
    output = _publish_json_replay_safe(output_path, payload, "final-gate scientific preregistration")
    # Recompute after publication so code or R4 drift during the write cannot
    # silently enter a formally sealed chain.
    expected = build_preregistration_payload(
        gt_inventory_path=gt_inventory_path,
        pending_config_path=pending_config_path,
        r4_receipt_path=r4_receipt_path,
    )
    if stable_bytes(output, "published final-gate preregistration")[1] != canonical_json(expected):
        raise RuntimeError("final-gate preregistration inputs drifted during publication")
    return output


def validate_scientific_preregistration(
    path: Path = PREREGISTRATION_PATH,
) -> tuple[Path, dict[str, Any], R4Exact80Binding, GTInventoryBinding]:
    if _absolute_lexical(path) != _absolute_lexical(PREREGISTRATION_PATH):
        raise ValueError("final gate preregistration path is noncanonical")
    source, value, data, _ = stable_json(
        path, "final-gate scientific preregistration", schema=PREREGISTRATION_SCHEMA
    )
    inventory_record = value.get("annotation_inventory") or {}
    expected = build_preregistration_payload(
        gt_inventory_path=Path(str(inventory_record.get("path", ""))),
        pending_config_path=Path(str((value.get("pending_config") or {}).get("path", ""))),
        r4_receipt_path=Path(str(((value.get("r4_exact80") or {}).get("wrapper") or {}).get("path", ""))),
    )
    if data != canonical_json(expected):
        raise ValueError("final-gate preregistration/current inputs or science differ")
    r4 = load_r4_exact80_binding(Path(expected["r4_exact80"]["wrapper"]["path"]))
    inventory = validate_gt_inventory_metadata(
        Path(expected["annotation_inventory"]["path"]), r4_binding=r4
    )
    return source, value, r4, inventory


def _ready_payload(prereg_path: Path, prereg: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": READY_SCHEMA, "complete": True, "create_only": True,
        "namespace": NAMESPACE, "state": "ready_r4_exact80_bound",
        "preregistration": artifact_record(
            prereg_path, "final-gate preregistration", schema=PREREGISTRATION_SCHEMA
        ),
        "r4_exact80": prereg["r4_exact80"],
        "annotation_inventory": prereg["annotation_inventory"],
        "science_contract_sha256": sha256_bytes(canonical_json(prereg["science_contract"])),
        "implementation": prereg["implementation"],
        "outputs": prereg["outputs"],
        "authorizations": {
            "ground_truth_join_fold234": True, "gate_fit_fold234": True,
            "threshold_selection_fold234_oof": True,
            "fold0_reused_dev_after_oof_pass": True,
            "geometry_materialization_after_oof_pass": True,
            "fold1": False, "official_validation": False,
            "policy_activation": False,
        },
        "access_before_run": {
            "gt_array_content": False, "output_directory_created": False,
            "gpu_started": False, "fold1": False, "official_validation": False,
        },
    }


def seal_ready_authorization(
    *, preregistration_path: Path = PREREGISTRATION_PATH,
    ready_path: Path = READY_CONFIG_PATH,
    authorization_path: Path = RUN_AUTHORIZATION_PATH,
) -> tuple[Path, Path]:
    if (
        _absolute_lexical(ready_path) != _absolute_lexical(READY_CONFIG_PATH)
        or _absolute_lexical(authorization_path) != _absolute_lexical(RUN_AUTHORIZATION_PATH)
    ):
        raise ValueError("final-gate ready/authorization target is noncanonical")
    prereg_path, prereg, _, _ = validate_scientific_preregistration(preregistration_path)
    ready = _ready_payload(prereg_path, prereg)
    ready_bytes = canonical_json(ready)
    commit_material = {
        "schema": "boxfusion.ca1m_tr3d_terminal_gate_commit_material.v5.final",
        "preregistration_sha256": sha256_file(prereg_path),
        "ready_config_sha256": sha256_bytes(ready_bytes),
        "r4_wrapper_sha256": prereg["r4_exact80"]["wrapper"]["sha256"],
        "r4_authorization_commit_id": prereg["r4_exact80"]["authorization_commit_id"],
        "annotation_inventory_sha256": prereg["annotation_inventory"]["sha256"],
    }
    commit_id = sha256_bytes(canonical_json(commit_material))
    authorization = {
        "schema": AUTHORIZATION_SCHEMA, "complete": True, "create_only": True,
        "namespace": NAMESPACE, "commit_id": commit_id,
        "commit_role": "last_published_unique_run_gate",
        "preregistration": artifact_record(
            prereg_path, "final-gate preregistration", schema=PREREGISTRATION_SCHEMA
        ),
        "ready_config": {
            "path": os.fspath(_absolute_lexical(ready_path)),
            "sha256": sha256_bytes(ready_bytes), "schema": READY_SCHEMA,
        },
        "r4_wrapper_sha256": prereg["r4_exact80"]["wrapper"]["sha256"],
        "r4_authorization_commit_id": prereg["r4_exact80"]["authorization_commit_id"],
        "annotation_inventory_sha256": prereg["annotation_inventory"]["sha256"],
        "authorizations": ready["authorizations"],
        "ground_truth_access_at_seal": False,
        "output_runtime_root_created_at_seal": False,
        "gpu_started_at_seal": False, "fold1_access": False,
        "official_validation_access": False,
    }
    # Ready is a non-authorizing leaf.  The authorization is published last;
    # a crash between them remains fail-closed and exact replay is byte-safe.
    ready_output = _publish_json_replay_safe(ready_path, ready, "final-gate ready config")
    authorization_output = _publish_json_replay_safe(
        authorization_path, authorization, "final-gate run authorization"
    )
    return ready_output, authorization_output


@dataclass(frozen=True)
class ExecutionContext:
    authorization_path: Path
    authorization_sha256: str
    authorization_identity: Mapping[str, int]
    preregistration_path: Path
    preregistration: Mapping[str, Any]
    ready_path: Path
    ready: Mapping[str, Any]
    r4: R4Exact80Binding
    inventory: GTInventoryBinding
    outputs: Mapping[str, Path]


def _validate_implementation_records(records: Any) -> None:
    if not isinstance(records, Mapping) or set(records) != set(IMPLEMENTATION_PATHS):
        raise ValueError("final-gate implementation records differ")
    for key, path in IMPLEMENTATION_PATHS.items():
        validate_artifact_record(
            records[key], f"final-gate implementation {key}",
            canonical_path=path, require_identity=True, immutable=False,
        )


def load_execution_context(
    authorization_path: Path = RUN_AUTHORIZATION_PATH,
    *, require_outputs_absent: bool = True,
) -> ExecutionContext:
    if _absolute_lexical(authorization_path) != _absolute_lexical(RUN_AUTHORIZATION_PATH):
        raise ValueError("final-gate run authorization path is noncanonical")
    auth_path, authorization, auth_data, auth_identity = stable_json(
        authorization_path, "final-gate run authorization", schema=AUTHORIZATION_SCHEMA
    )
    if (
        authorization.get("complete") is not True
        or authorization.get("create_only") is not True
        or authorization.get("namespace") != NAMESPACE
        or authorization.get("commit_role") != "last_published_unique_run_gate"
        or authorization.get("ground_truth_access_at_seal") is not False
        or authorization.get("output_runtime_root_created_at_seal") is not False
        or authorization.get("gpu_started_at_seal") is not False
        or authorization.get("fold1_access") is not False
        or authorization.get("official_validation_access") is not False
    ):
        raise PermissionError("final-gate run authorization semantics differ")
    prereg_path, prereg, r4, inventory = validate_scientific_preregistration(
        Path(str((authorization.get("preregistration") or {}).get("path", "")))
    )
    validate_artifact_record(
        authorization.get("preregistration"), "authorization preregistration",
        schema=PREREGISTRATION_SCHEMA, canonical_path=PREREGISTRATION_PATH,
        require_identity=True,
    )
    ready_path, ready, ready_data, _ = _record_json(
        authorization.get("ready_config"), "authorization ready config",
        schema=READY_SCHEMA, canonical_path=READY_CONFIG_PATH,
    )
    expected_ready = _ready_payload(prereg_path, prereg)
    if ready_data != canonical_json(expected_ready):
        raise ValueError("final-gate ready/current preregistration differ")
    commit_material = {
        "schema": "boxfusion.ca1m_tr3d_terminal_gate_commit_material.v5.final",
        "preregistration_sha256": sha256_file(prereg_path),
        "ready_config_sha256": sha256_bytes(ready_data),
        "r4_wrapper_sha256": r4.wrapper_sha256,
        "r4_authorization_commit_id": r4.authorization_commit_id,
        "annotation_inventory_sha256": inventory.sha256,
    }
    if (
        authorization.get("commit_id") != sha256_bytes(canonical_json(commit_material))
        or authorization.get("r4_wrapper_sha256") != r4.wrapper_sha256
        or authorization.get("r4_authorization_commit_id") != r4.authorization_commit_id
        or authorization.get("annotation_inventory_sha256") != inventory.sha256
        or authorization.get("authorizations") != expected_ready["authorizations"]
        or sha256_bytes(auth_data) != sha256_file(auth_path)
    ):
        raise PermissionError("final-gate authorization commit binding differs")
    _validate_implementation_records(prereg.get("implementation"))
    outputs = {key: Path(value) for key, value in prereg["outputs"].items()}
    if outputs != OUTPUT_PATHS:
        raise ValueError("final-gate runtime output paths differ")
    if require_outputs_absent:
        for key, path in outputs.items():
            if key == "materialization_root":
                exists = path.exists() or path.is_symlink()
            else:
                exists = path.exists() or path.is_symlink()
            if exists:
                raise FileExistsError(f"refusing pre-existing final-gate output {key}: {path}")
        if RUNTIME_ROOT.exists() or RUNTIME_ROOT.is_symlink():
            raise FileExistsError(f"refusing pre-existing final-gate runtime root: {RUNTIME_ROOT}")
    return ExecutionContext(
        auth_path, sha256_bytes(auth_data), auth_identity,
        prereg_path, prereg, ready_path, ready, r4, inventory, outputs,
    )


def revalidate_execution_inputs(context: ExecutionContext) -> None:
    source, _, data, identity = stable_json(
        context.authorization_path, "final-gate run authorization",
        schema=AUTHORIZATION_SCHEMA,
    )
    if (
        source != context.authorization_path
        or sha256_bytes(data) != context.authorization_sha256
        or identity != context.authorization_identity
    ):
        raise RuntimeError("final-gate authorization changed during execution")
    _, prereg, r4, inventory = validate_scientific_preregistration(
        context.preregistration_path
    )
    if (
        prereg != context.preregistration
        or r4.wrapper_sha256 != context.r4.wrapper_sha256
        or r4.collection_sha256 != context.r4.collection_sha256
        or r4.authorization_commit_id != context.r4.authorization_commit_id
        or inventory.sha256 != context.inventory.sha256
        or inventory.identity != context.inventory.identity
    ):
        raise RuntimeError("final-gate frozen input changed during execution")


def inventory_ground_truth_loader(
    context: ExecutionContext,
) -> Callable[[str], np.ndarray]:
    """Return the sole production GT loader; fold1/validation are unaddressable."""

    allowed = set(context.r4.scene_folds)

    def load(scene_id: str) -> np.ndarray:
        scene = str(scene_id)
        if scene not in allowed or context.r4.scene_folds[scene] not in (0, 2, 3, 4):
            raise PermissionError("GT request is outside exact fit60+fold0-dev20")
        revalidate_execution_inputs(context)
        row = context.inventory.scene_rows[scene]
        box_record = row.get("box") or {}
        path, data, _ = validate_artifact_record(
            box_record, f"CA-train GT boxes {scene}"
        )
        # np.load receives bytes from the already inode-bound read, avoiding a
        # second path open after the provenance check.
        from io import BytesIO
        value = np.asarray(np.load(BytesIO(data), allow_pickle=False))
        if (
            value.dtype != np.dtype(np.float64)
            or value.ndim != 3 or value.shape[1:] != (8, 3)
            or not np.isfinite(value).all()
            or sha256_bytes(data) != box_record.get("sha256")
            or path != _absolute_lexical(Path(str(box_record.get("path"))))
        ):
            raise ValueError(f"{scene}: CA-train GT array contract differs")
        return value

    return load


def _guarded_loader(
    context: ExecutionContext, loader: Callable[[str], Any], allowed_folds: Sequence[int],
) -> Callable[[str], Any]:
    allowed = {
        scene for scene, fold in context.r4.scene_folds.items()
        if fold in tuple(int(value) for value in allowed_folds)
    }

    def load(scene: str) -> Any:
        if str(scene) not in allowed:
            raise PermissionError("ground-truth request is outside preregistered partition")
        revalidate_execution_inputs(context)
        return loader(str(scene))

    return load


def _materialize_fold0(
    dataset: gate.GateDatasetV5, *, policy_path: Path,
    output_root: Path, manifest_path: Path,
) -> Path:
    policy_source, policy, model = gate.load_gate_policy_v5(policy_path, require_oof_pass=True)
    iou, probability, gain = gate.predict_gate_model_v5(model, dataset.features)
    predictions = gate.GatePredictionsV5(
        iou, probability, gain,
        np.asarray(["[2,3,4]"] * len(iou)),
    )
    thresholds = policy.get("thresholds") or {}
    selection = gate.select_replacements_v5(
        dataset, predictions,
        iou_threshold=float(thresholds["candidate_iou"]),
        gain_threshold=float(thresholds["same_gt_gain"]),
        probability_threshold=float(thresholds["iou50_probability"]),
    )
    selected_by_scene: dict[str, list[int]] = {scene: [] for scene in dataset.scene_order.astype(str)}
    for candidate_position in selection.candidate_positions.tolist():
        selected_by_scene[str(dataset.candidate_scene_ids[candidate_position])].append(
            int(candidate_position)
        )
    rows: list[dict[str, Any]] = []
    for scene in dataset.scene_order.astype(str).tolist():
        anchor_global = np.flatnonzero(dataset.anchor_scene_ids.astype(str) == scene)
        candidate_global = np.flatnonzero(dataset.candidate_scene_ids.astype(str) == scene)
        anchor_lookup = {int(global_row): local for local, global_row in enumerate(anchor_global.tolist())}
        candidate_lookup = {int(global_row): local for local, global_row in enumerate(candidate_global.tolist())}
        chosen = np.asarray(selected_by_scene[scene], np.int64)
        replace = np.asarray([
            anchor_lookup[int(dataset.candidate_anchor_positions[row])] for row in chosen
        ], np.int64)
        candidate_rows = np.asarray([candidate_lookup[int(row)] for row in chosen], np.int64)
        result = gate.materialize_geometry_only_v5(
            anchor_corners=np.asarray(dataset.anchor_corners[anchor_global], np.float32),
            anchor_scores=np.asarray(dataset.anchor_scores_oof[anchor_global], np.float32),
            candidate_corners=np.asarray(dataset.candidate_corners[candidate_global], np.float32),
            anchor_indices=replace, candidate_rows=candidate_rows,
        )
        target = output_root / f"{scene}_geometry_only_v5_final.npz"
        output = gate.write_materialized_geometry_v5(
            target, scene_id=scene,
            source_anchor_sha256=gate.sha256_array(dataset.anchor_corners[anchor_global]),
            source_candidate_sha256=gate.sha256_array(dataset.candidate_corners[candidate_global]),
            policy_sha256=sha256_file(policy_source), result=result,
        )
        rows.append({
            "scene_id": scene, "fold_id": 0,
            "path": os.fspath(output), "sha256": sha256_file(output),
            "identity": artifact_record(output, f"materialized scene {scene}")["identity"],
            "anchor_count": len(anchor_global), "replacement_count": len(chosen),
            "scores_preserved": True, "row_order_preserved": True,
            "row_count_preserved": True,
        })
    payload = {
        "schema": MATERIALIZATION_COLLECTION_SCHEMA,
        "complete": True, "create_only": True,
        "report_label": "noncanonical_reused_dev_exploratory_materialization",
        "scene_count": 20, "fold_id": 0,
        "geometry_only": True, "scores_preserved": True,
        "row_order_preserved": True, "row_count_preserved": True,
        "policy_activation_authorized": False,
        "fold1_or_validation_authorized": False,
        "policy": artifact_record(policy_source, "final-gate policy", schema=gate.POLICY_SCHEMA),
        "scenes": rows,
    }
    return gate.write_json_create_only(
        manifest_path, payload, "final-gate fold0 materialization manifest"
    )


def run_final_gate(
    *, context: ExecutionContext,
    ground_truth_loader: Callable[[str], Any],
) -> Path:
    """Execute fit60 OOF, then (only on PASS) the frozen fold0 diagnostic."""

    # This call is CPU-only.  There is no device argument and no import of a
    # detector, CUDA, fold1, validation or legacy v4 gate implementation.
    revalidate_execution_inputs(context)
    fit_loader = _guarded_loader(context, ground_truth_loader, (2, 3, 4))
    fit = gate.build_labeled_dataset_v5(
        context.r4.collection_path, purpose="fold234_oof_fit",
        ground_truth_loader=fit_loader,
    )
    fit_artifact, fit_manifest = gate.seal_gate_dataset_v5(
        fit, artifact_path=context.outputs["fit_dataset"],
        manifest_path=context.outputs["fit_dataset_manifest"],
    )
    result = gate.train_gate_oof_v5(fit)
    oof, threshold, policy = gate.seal_gate_oof_result_v5(
        fit, result, oof_path=context.outputs["oof_predictions"],
        threshold_path=context.outputs["threshold_receipt"],
        policy_path=context.outputs["exploratory_policy"],
    )
    common = {
        "authorization": artifact_record(
            context.authorization_path, "final-gate authorization",
            schema=AUTHORIZATION_SCHEMA,
        ),
        "r4_wrapper_sha256": context.r4.wrapper_sha256,
        "r4_candidate_collection_sha256": context.r4.collection_sha256,
        "fit_dataset": artifact_record(fit_artifact, "fit60 dataset", schema=gate.DATASET_SCHEMA),
        "fit_dataset_manifest": artifact_record(fit_manifest, "fit60 dataset manifest"),
        "oof_predictions": artifact_record(oof, "fold234 OOF", schema=gate.OOF_SCHEMA),
        "threshold_receipt": artifact_record(
            threshold, "fold234 threshold receipt", schema=gate.THRESHOLD_SCHEMA
        ),
        "exploratory_policy": artifact_record(policy, "exploratory policy", schema=gate.POLICY_SCHEMA),
        "fold1_access": False, "official_validation_access": False,
        "gpu_started": False, "policy_activation_authorized": False,
    }
    if result.threshold_receipt.get("safety_gate_passed") is not True:
        payload = {
            "schema": STOP_SCHEMA, "complete": True, "create_only": True,
            "status": "STOP_FOLD234_OOF_GATE_FAIL",
            "fold0_ground_truth_access": False,
            "fold0_materialization_created": False,
            **common,
        }
        return gate.write_json_create_only(
            context.outputs["stop_receipt"], payload, "final-gate STOP receipt"
        )
    fold0_loader = _guarded_loader(context, ground_truth_loader, (0,))
    fold0 = gate.build_labeled_dataset_v5(
        context.r4.collection_path, purpose="fold0_reused_dev",
        ground_truth_loader=fold0_loader,
    )
    fold0_artifact, fold0_manifest = gate.seal_gate_dataset_v5(
        fold0, artifact_path=context.outputs["fold0_dataset"],
        manifest_path=context.outputs["fold0_dataset_manifest"],
    )
    fold0_report = gate.evaluate_fold0_reused_dev_v5(
        fold0, policy_path=policy, output_path=context.outputs["fold0_report"]
    )
    materialization = _materialize_fold0(
        fold0, policy_path=policy,
        output_root=context.outputs["materialization_root"],
        manifest_path=context.outputs["materialization_manifest"],
    )
    payload = {
        "schema": RUN_SCHEMA, "complete": True, "create_only": True,
        "status": "PASS_EXPLORATORY_FOLD0_DIAGNOSTIC_COMPLETE",
        "thresholds_selected_only_from_fold234_oof": True,
        "fold0_retuning": False, "fold0_model_selection": False,
        "fold0_result_can_authorize_policy": False,
        **common,
        "fold0_dataset": artifact_record(
            fold0_artifact, "fold0 dataset", schema=gate.DATASET_SCHEMA
        ),
        "fold0_dataset_manifest": artifact_record(
            fold0_manifest, "fold0 dataset manifest"
        ),
        "fold0_report": artifact_record(
            fold0_report, "fold0 report", schema=gate.FOLD0_REPORT_SCHEMA
        ),
        "materialization_manifest": artifact_record(
            materialization, "fold0 materialization manifest",
            schema=MATERIALIZATION_COLLECTION_SCHEMA,
        ),
    }
    return gate.write_json_create_only(
        context.outputs["run_receipt"], payload, "final-gate run receipt"
    )


__all__ = [
    "AUTHORIZATION_SCHEMA", "DEFAULT_PENDING_CONFIG", "FinalGateProtocolError",
    "GT_INVENTORY_SCHEMA", "IMPLEMENTATION_PATHS", "MANIFEST_ROOT",
    "MATERIALIZATION_COLLECTION_SCHEMA", "NAMESPACE", "OUTPUT_PATHS",
    "PENDING_SCHEMA", "PREREGISTRATION_PATH", "PREREGISTRATION_SCHEMA",
    "PendingR4Inputs", "R4Exact80Binding", "R4_RECEIPT_PATH",
    "R4_RECEIPT_SCHEMA", "READY_CONFIG_PATH", "READY_SCHEMA",
    "RUN_AUTHORIZATION_PATH", "RUN_SCHEMA", "STOP_SCHEMA", "ap_parity_fixture",
    "ExecutionContext", "GTInventoryBinding", "artifact_record",
    "build_preregistration_payload", "canonical_json", "inventory_ground_truth_loader",
    "load_execution_context", "load_r4_exact80_binding",
    "operational_preflight_pending", "science_contract", "sha256_bytes",
    "sha256_file", "seal_ready_authorization", "seal_scientific_preregistration",
    "stable_bytes", "stable_json", "static_preflight", "run_final_gate",
    "validate_artifact_record", "validate_gt_inventory_metadata",
    "validate_pending_config", "validate_scientific_preregistration",
]
