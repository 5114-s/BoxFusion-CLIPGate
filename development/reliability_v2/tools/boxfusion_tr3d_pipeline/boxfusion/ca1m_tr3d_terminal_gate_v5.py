"""CA-only asymmetric-xfit terminal gate v5 execution primitives.

This module is deliberately data-agnostic.  Importing it never opens a
candidate, annotation, checkpoint, fold-1, or validation artifact.  Runtime
entry points validate immutable, SHA-bound inputs before invoking a caller
supplied ground-truth loader.  The production pending configuration therefore
continues to fail closed until those inputs are separately sealed.

The executable science contract is:

* detector candidates are exact scene-grouped OOF for folds 2/3/4
  (34->2, 24->3, 23->4), plus the 234->0 outer diagnostic;
* stacked anchor scores are CA-native B6 all-fold OOF scores, never deploy
  scores;
* three low-capacity linear heads learn continuous candidate IoU, strict
  IoU>0.50 probability, and within-anchor pairwise same-GT benefit;
* thresholds are selected only from gate OOF predictions on folds 2/3/4;
* fold 0 is a frozen reused-dev diagnostic; and
* materialization changes geometry only.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .ca1m_tr3d_terminal import pairwise_world_aabb_iou, sha256_array
from .ca1m_tr3d_xfit_r2_eval import metric_delta, official_ca_ap


NAMESPACE = "ca1m_tr3d_exploratory_gate_xfit_r2_v5"
ROLE_RECEIPT_SCHEMA = "boxfusion.ca1m_tr3d_xfit_r2_detector_role_receipt.v1"
EVIDENCE_SCHEMA = "boxfusion.ca1m_tr3d_xfit_r2_candidate_evidence.v1"
SCENE_MANIFEST_SCHEMA = "boxfusion.ca1m_tr3d_xfit_r2_candidate_scene.v1"
ROLE_COLLECTION_SCHEMA = "boxfusion.ca1m_tr3d_xfit_r2_candidate_role_collection.v1"
COLLECTION_SCHEMA = "boxfusion.ca1m_tr3d_xfit_r2_candidate_collection.v1"
DATASET_SCHEMA = "boxfusion.ca1m_tr3d_exploratory_gate_dataset.v5"
DATASET_MANIFEST_SCHEMA = f"{DATASET_SCHEMA}.manifest"
MODEL_SCHEMA = "boxfusion.ca1m_tr3d_exploratory_gate_linear_model.v5"
OOF_SCHEMA = "boxfusion.ca1m_tr3d_exploratory_gate_oof_predictions.v5"
THRESHOLD_SCHEMA = "boxfusion.ca1m_tr3d_exploratory_gate_threshold_receipt.v5"
POLICY_SCHEMA = "boxfusion.ca1m_tr3d_exploratory_gate_policy.v5"
FOLD0_REPORT_SCHEMA = "boxfusion.ca1m_tr3d_exploratory_gate_fold0_report.v5"
MATERIALIZATION_SCHEMA = "boxfusion.ca1m_tr3d_geometry_materialization.v5"
LOCKED_AUTH_SCHEMA = "boxfusion.ca1m_tr3d_exploratory_gate_fold1_authorization.v5"
LOCKED_RECEIPT_SCHEMA = "boxfusion.ca1m_tr3d_exploratory_gate_fold1_consumption.v5"

ANCHOR_SCORE_SOURCE = "ca1m_native_b6_final_base_all_fold_oof_row_scores_v2"
_NATIVE_FEATURE_NAMES = (
    "detector_score", "support_given_depth", "occluded_given_depth",
    "free_given_depth", "invalid_ratio", "view_coverage", "sample_support",
    "area_quality", "area_stability", "support_view_mean", "support_view_min",
    "free_view_max", "aspect_balance", "height_balance",
)
_RELATION_FEATURE_NAMES = (
    "candidate_minus_anchor_score", "candidate_anchor_iou",
    "center_distance_over_anchor_diagonal", "log_candidate_over_anchor_volume",
    "extent_log_ratio_l2", "log1p_candidate_point_support",
    "log1p_candidate_point_density", "candidate_point_support_fraction",
    "candidate_global_rank_fraction", "candidate_anchor_group_rank_fraction",
    "log1p_anchor_group_size", "candidate_score_minus_best_sibling",
)
FEATURE_NAMES = (
    tuple(f"anchor_{name}" for name in _NATIVE_FEATURE_NAMES)
    + tuple(f"candidate_{name}" for name in _NATIVE_FEATURE_NAMES)
    + _RELATION_FEATURE_NAMES
)
RAW_SCORE_FEATURE_NAMES = (
    "candidate_detector_score",
    "candidate_minus_anchor_score",
    "candidate_score_minus_best_sibling",
)
RAW_SCORE_FEATURE_INDICES = tuple(FEATURE_NAMES.index(name) for name in RAW_SCORE_FEATURE_NAMES)
REQUIRED_FEATURE_GROUPS = {
    "ca_native_visibility_and_depth_support": tuple(range(0, 28)),
    "candidate_anchor_geometry": tuple(range(28, 33)),
    "candidate_point_support_and_density": tuple(range(33, 36)),
    "anchor_group_sibling_context": tuple(range(36, 40)),
}

ROLE_SPECS: dict[str, tuple[tuple[int, ...], int]] = {
    "inner_holdout2": ((3, 4), 2),
    "inner_holdout3": ((2, 4), 3),
    "inner_holdout4": ((2, 3), 4),
    "outer_dev": ((2, 3, 4), 0),
}
FIT_FOLDS = (2, 3, 4)
OOF_GATE_ROLES = {
    2: (3, 4),
    3: (2, 4),
    4: (2, 3),
}
MAX_REPLACEMENTS_PER_SCENE = 16
IOU_THRESHOLDS = (0.15, 0.25, 0.50)
IOU_GRID = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
GAIN_GRID = (0.00, 0.02, 0.05, 0.08, 0.10, 0.15)
PROB_GRID = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
OOF_SAFETY_GATE = {
    "min_delta_ap15": 0.0,
    "min_delta_ap25": 0.0,
    "min_delta_ap50": 0.0025,
    "min_replacements": 30,
    "min_scenes": 12,
    "min_positive_gain_fraction": 0.60,
    "max_severe_harm_fraction": 0.10,
    "max_target_switch_fraction": 0.10,
}

_SCENE = re.compile(r"^[0-9]{8}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_RUNTIME_PATH_TOKENS = (
    "scannet",
    "ca1m_tr3d_benefit_gate_final_base_v4",
    "ca1m_tr3d_terminal_ca_native_train100_v4",
    "ca1m_tr3d_benefit_final_base_v4",
    "ca1m_fg_scratch_seed0_fp32_gb16_v1",
)


class V5ProtocolError(RuntimeError):
    """A v5 isolation, provenance, or OOF invariant was violated."""


class LockedFoldDisabledError(PermissionError):
    """Raised before a locked-fold input or loader is opened."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, name: str) -> str:
    result = str(value)
    if _SHA.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return result


def _scene(value: Any) -> str:
    result = str(value)
    if _SCENE.fullmatch(result) is None:
        raise ValueError(f"invalid CA-1M scene id: {result!r}")
    return result


def _regular(path: Path, name: str, *, immutable: bool = True) -> Path:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink: {path}")
    result = path.resolve()
    if not result.is_file() or result.is_symlink() or result.stat().st_size <= 0:
        raise FileNotFoundError(f"missing regular {name}: {result}")
    if immutable and result.stat().st_mode & 0o222:
        raise ValueError(f"{name} must be read-only: {result}")
    return result


def _json(path: Path, name: str, *, immutable: bool = True) -> tuple[Path, dict[str, Any]]:
    source = _regular(path, name, immutable=immutable)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return source, value


def _record(value: Any, name: str, *, schema: str | None = None) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} record must be an object")
    source, payload = _json(Path(str(value.get("path", ""))), name)
    digest = _sha(value.get("sha256"), f"{name} SHA256")
    if sha256_file(source) != digest:
        raise ValueError(f"{name} SHA256 differs")
    expected_schema = schema if schema is not None else value.get("schema")
    if not expected_schema or value.get("schema") != expected_schema or payload.get("schema") != expected_schema:
        raise ValueError(f"{name} schema differs")
    return source, payload


def _assert_safe_runtime_path(path: Path, name: str) -> None:
    lowered = str(path).lower()
    for token in _FORBIDDEN_RUNTIME_PATH_TOKENS:
        if token in lowered:
            raise ValueError(f"{name} references forbidden legacy token {token}")


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def write_bytes_create_only(path: Path, payload: bytes, name: str) -> Path:
    """Publish one immutable artifact with create-only inode/hash binding.

    The temporary inode is fully written, fsynced and made read-only before it
    is hard-linked into the destination directory.  Both links are addressed
    through one pinned directory descriptor.  Before returning, the published
    name is reopened with ``O_NOFOLLOW`` and must still identify the same inode
    and exact payload.  A failed publication removes only the inode created by
    this call; an attacker-created replacement is never unlinked.
    """

    if not isinstance(payload, bytes):
        raise TypeError(f"{name} payload must be bytes")
    requested = Path(path).absolute()
    if requested.name in {"", ".", ".."}:
        raise ValueError(f"invalid {name} target: {requested}")
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(f"refusing existing {name}: {requested}")
    requested.parent.mkdir(parents=True, exist_ok=True)
    if requested.parent.is_symlink():
        raise ValueError(f"{name} parent must not be a symlink: {requested.parent}")
    parent = requested.parent.resolve(strict=True)
    if parent != requested.parent:
        raise ValueError(f"{name} parent contains a symlink: {requested.parent}")
    target = parent / requested.name
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(parent, directory_flags)
    temporary_path: Path | None = None
    temporary_fd: int | None = None
    published: tuple[int, int] | None = None
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    try:
        temporary_fd, temporary_name = tempfile.mkstemp(
            dir=parent, prefix=f".{target.name}."
        )
        temporary_path = Path(temporary_name)
        offset = 0
        while offset < len(payload):
            offset += os.write(temporary_fd, payload[offset:])
        os.fsync(temporary_fd)
        os.fchmod(temporary_fd, 0o444)
        temp_stat = os.fstat(temporary_fd)
        published = (temp_stat.st_dev, temp_stat.st_ino)
        os.link(
            temporary_path.name, target.name,
            src_dir_fd=directory_fd, dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        verify_fd = os.open(target.name, read_flags, dir_fd=directory_fd)
        try:
            before = os.fstat(verify_fd)
            digest = hashlib.sha256()
            observed_size = 0
            while True:
                block = os.read(verify_fd, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
                observed_size += len(block)
            after = os.fstat(verify_fd)
        finally:
            os.close(verify_fd)
        if (
            (before.st_dev, before.st_ino) != published
            or (after.st_dev, after.st_ino) != published
            or before.st_size != len(payload)
            or after.st_size != len(payload)
            or observed_size != len(payload)
            or digest.hexdigest() != expected_sha256
            or before.st_mode & 0o222
            or after.st_mode & 0o222
        ):
            raise RuntimeError(f"published {name} inode/hash binding differs")
        os.close(temporary_fd)
        temporary_fd = None
        os.unlink(temporary_path.name, dir_fd=directory_fd)
        temporary_path = None
        final_stat = os.stat(target.name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            (final_stat.st_dev, final_stat.st_ino) != published
            or final_stat.st_nlink != 1
            or final_stat.st_size != len(payload)
            or final_stat.st_mode & 0o222
        ):
            raise RuntimeError(f"final {name} inode identity differs")
        os.fsync(directory_fd)
    except BaseException:
        if published is not None:
            try:
                current = os.stat(
                    target.name, dir_fd=directory_fd, follow_symlinks=False
                )
                if (current.st_dev, current.st_ino) == published:
                    os.unlink(target.name, dir_fd=directory_fd)
                    os.fsync(directory_fd)
            except FileNotFoundError:
                pass
        raise
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        if temporary_path is not None:
            try:
                os.unlink(temporary_path.name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)
    source = _regular(target, name)
    final = source.stat(follow_symlinks=False)
    if (final.st_dev, final.st_ino) != published or sha256_file(source) != expected_sha256:
        raise RuntimeError(f"returned {name} inode/hash binding differs")
    return source


def write_json_create_only(path: Path, payload: Mapping[str, Any], name: str) -> Path:
    return write_bytes_create_only(path, _canonical_json(payload), name)


def write_npz_create_only(path: Path, payload: Mapping[str, Any], name: str) -> Path:
    stream = BytesIO()
    np.savez_compressed(stream, **payload)
    return write_bytes_create_only(path, stream.getvalue(), name)


def _readonly(value: Any, dtype: Any | None = None) -> np.ndarray:
    result = np.ascontiguousarray(value if dtype is None else np.asarray(value, dtype=dtype))
    result.setflags(write=False)
    return result


def _scalar(archive: Mapping[str, Any], name: str) -> Any:
    value = np.asarray(archive[name])
    if value.shape != ():
        raise ValueError(f"{name} must be scalar")
    return value.item()


def _validate_continuation_payload(payload: Mapping[str, Any]) -> None:
    gate = payload.get("continuation_gate") or {}
    passed = payload.get("pass") is True or gate.get("pass") is True
    authorized = (
        payload.get("continue_inner_training_authorized") is True
        or gate.get("continue_inner_training_authorized") is True
    )
    roles = payload.get("authorized_inner_roles") or gate.get("authorized_inner_roles") or []
    if (
        payload.get("complete") is not True
        or not passed
        or not authorized
        or list(roles) != ["inner_holdout2", "inner_holdout3", "inner_holdout4"]
        or payload.get("fold1_access") is not False
        or payload.get("official_validation_access") is not False
        or payload.get("checkpoint_selection") is not False
    ):
        raise ValueError("outer continuation receipt is not a passing fixed-checkpoint receipt")


def _source_role_semantics(payload: Mapping[str, Any], role: str) -> tuple[str, str]:
    train_folds, heldout = ROLE_SPECS[role]
    protocol = payload.get("training_protocol") or {}
    checkpoint = payload.get("checkpoint") or {}
    if (
        payload.get("complete") is not True
        or ("status" in payload and payload.get("status") != "success")
        or payload.get("role") != role
        or list(protocol.get("train_folds", ())) != list(train_folds)
        or int(protocol.get("heldout_fold", -1)) != heldout
        or protocol.get("initialization") != "random_scratch_ca_only"
        or protocol.get("scannet_checkpoint_or_module_access", False) is not False
        or payload.get("checkpoint_selection", checkpoint.get("checkpoint_selection", False)) is not False
        or int(checkpoint.get("optimizer_updates", -1)) != 11268
    ):
        raise ValueError(f"{role}: source training receipt semantics differ")
    access = payload.get("access") or payload.get("evaluation_isolation") or {}
    if (
        access.get("fold1_metadata_or_ground_truth_access", access.get("fold1_access", False))
        is not False
        or access.get("official_validation_access", False) is not False
        or access.get("scannet_checkpoint_or_module_access", False) is not False
    ):
        raise ValueError(f"{role}: source training receipt access differs")
    checkpoint_path = _regular(Path(str(checkpoint.get("path", ""))), f"{role} checkpoint")
    checkpoint_sha = _sha(checkpoint.get("sha256"), f"{role} checkpoint SHA256")
    if sha256_file(checkpoint_path) != checkpoint_sha:
        raise ValueError(f"{role}: checkpoint SHA256 differs")
    return str(checkpoint_path), checkpoint_sha


def seal_detector_role_receipt_v5(
    output: Path,
    *,
    role: str,
    source_training_receipt: Mapping[str, Any],
    outer_continuation_receipt: Mapping[str, Any],
) -> Path:
    """Normalize an immutable outer/inner run receipt for candidate collection."""

    if role not in ROLE_SPECS:
        raise ValueError(f"unknown xfit role {role}")
    continuation_path, continuation = _record(
        outer_continuation_receipt, "outer continuation receipt"
    )
    _validate_continuation_payload(continuation)
    source_path, source = _record(source_training_receipt, f"{role} source receipt")
    checkpoint_path, checkpoint_sha = _source_role_semantics(source, role)
    train_folds, output_fold = ROLE_SPECS[role]
    payload = {
        "schema": ROLE_RECEIPT_SCHEMA,
        "complete": True,
        "create_only": True,
        "role": role,
        "detector_train_folds": list(train_folds),
        "candidate_output_fold": output_fold,
        "checkpoint_policy": "fixed_iter_11268_only_no_selection",
        "initialization": "random_scratch_ca_only",
        "checkpoint": {"path": checkpoint_path, "sha256": checkpoint_sha},
        "source_training_receipt": {
            "path": str(source_path), "sha256": sha256_file(source_path),
            "schema": source.get("schema"),
        },
        "outer_continuation_receipt": {
            "path": str(continuation_path), "sha256": sha256_file(continuation_path),
            "schema": continuation.get("schema"),
        },
        "access": {
            "fold1_metadata_or_ground_truth_access": False,
            "official_validation_access": False,
            "scannet_checkpoint_or_module_access": False,
        },
    }
    return write_json_create_only(output, payload, f"{role} normalized receipt")


def load_detector_role_receipt_v5(record: Mapping[str, Any], role: str) -> tuple[Path, dict[str, Any]]:
    path, payload = _record(record, f"{role} normalized receipt", schema=ROLE_RECEIPT_SCHEMA)
    train_folds, output_fold = ROLE_SPECS[role]
    if (
        payload.get("complete") is not True
        or payload.get("create_only") is not True
        or payload.get("role") != role
        or tuple(payload.get("detector_train_folds", ())) != train_folds
        or payload.get("candidate_output_fold") != output_fold
        or payload.get("checkpoint_policy") != "fixed_iter_11268_only_no_selection"
        or payload.get("initialization") != "random_scratch_ca_only"
        or payload.get("access") != {
            "fold1_metadata_or_ground_truth_access": False,
            "official_validation_access": False,
            "scannet_checkpoint_or_module_access": False,
        }
    ):
        raise ValueError(f"{role}: normalized receipt differs")
    _, continuation = _record(payload.get("outer_continuation_receipt"), "outer continuation receipt")
    _validate_continuation_payload(continuation)
    _, source = _record(payload.get("source_training_receipt"), f"{role} source receipt")
    checkpoint_path, checkpoint_sha = _source_role_semantics(source, role)
    checkpoint = payload.get("checkpoint") or {}
    if checkpoint != {"path": checkpoint_path, "sha256": checkpoint_sha}:
        raise ValueError(f"{role}: normalized checkpoint binding differs")
    return path, payload


def _evidence_payload(
    *, scene_id: str, fold_id: int, producer_role: str,
    producer_checkpoint_sha256: str, training_receipt_sha256: str,
    outer_continuation_receipt_sha256: str, b6_oof_sidecar_sha256: str,
    candidate_corners: Any, candidate_rows: Any, candidate_scores: Any,
    anchor_indices: Any, features: Any, anchor_corners: Any, anchor_scores: Any,
) -> dict[str, Any]:
    scene = _scene(scene_id)
    if producer_role not in ROLE_SPECS:
        raise ValueError("unknown producer role")
    train_folds, expected_fold = ROLE_SPECS[producer_role]
    if int(fold_id) != expected_fold or int(fold_id) in train_folds:
        raise ValueError("candidate producer is in-sample for its output fold")
    candidates = np.asarray(candidate_corners)
    rows = np.asarray(candidate_rows)
    scores = np.asarray(candidate_scores)
    anchors_for_candidate = np.asarray(anchor_indices)
    matrix = np.asarray(features)
    anchors = np.asarray(anchor_corners)
    oof_scores = np.asarray(anchor_scores)
    n, a = len(candidates), len(anchors)
    if (
        candidates.dtype != np.dtype(np.float32) or candidates.shape != (n, 8, 3)
        or rows.dtype != np.dtype(np.int64) or rows.shape != (n,)
        or scores.dtype != np.dtype(np.float32) or scores.shape != (n,)
        or anchors_for_candidate.dtype != np.dtype(np.int64)
        or anchors_for_candidate.shape != (n,)
        or matrix.dtype != np.dtype(np.float32)
        or matrix.shape != (n, len(FEATURE_NAMES))
        or anchors.dtype != np.dtype(np.float32) or anchors.shape != (a, 8, 3)
        or oof_scores.dtype != np.dtype(np.float32) or oof_scores.shape != (a,)
        or not np.array_equal(rows, np.sort(rows)) or len(np.unique(rows)) != n
        or np.any((anchors_for_candidate < 0) | (anchors_for_candidate >= a))
        or not all(np.isfinite(value).all() for value in (candidates, scores, matrix, anchors, oof_scores))
        or np.any((scores < 0.0) | (scores > 1.0))
        or np.any((oof_scores < 0.0) | (oof_scores > 1.0))
    ):
        raise ValueError(f"{scene}: candidate evidence dtype/shape/value contract differs")
    checkpoint_sha = _sha(producer_checkpoint_sha256, "producer checkpoint SHA256")
    receipt_sha = _sha(training_receipt_sha256, "training receipt SHA256")
    continuation_sha = _sha(outer_continuation_receipt_sha256, "continuation SHA256")
    b6_sha = _sha(b6_oof_sidecar_sha256, "B6 OOF sidecar SHA256")
    anchor_rows = np.arange(a, dtype=np.int64)
    anchor_identity = hashlib.sha256()
    anchor_identity.update(scene.encode())
    anchor_identity.update(bytes.fromhex(sha256_array(anchor_rows)))
    anchor_identity.update(bytes.fromhex(sha256_array(anchors)))
    anchor_identity.update(bytes.fromhex(sha256_array(oof_scores)))
    return {
        "schema": np.asarray(EVIDENCE_SCHEMA),
        "complete": np.asarray(True),
        "create_only": np.asarray(True),
        "ground_truth_access": np.asarray(False),
        "fold1_access": np.asarray(False),
        "official_validation_access": np.asarray(False),
        "scene_id": np.asarray(scene),
        "fold_id": np.asarray(int(fold_id), np.int64),
        "producer_role": np.asarray(producer_role),
        "producer_checkpoint_sha256": np.asarray(checkpoint_sha),
        "producer_train_folds": np.asarray(train_folds, np.int64),
        "training_receipt_sha256": np.asarray(receipt_sha),
        "outer_continuation_receipt_sha256": np.asarray(continuation_sha),
        "anchor_score_source": np.asarray(ANCHOR_SCORE_SOURCE),
        "b6_oof_sidecar_sha256": np.asarray(b6_sha),
        "feature_names": np.asarray(FEATURE_NAMES),
        "feature_groups_json": np.asarray(json.dumps(REQUIRED_FEATURE_GROUPS, sort_keys=True)),
        "candidate_corners": candidates,
        "candidate_rows": rows,
        "candidate_scores": scores,
        "anchor_indices": anchors_for_candidate,
        "features": matrix,
        "anchor_corners": anchors,
        "anchor_rows": anchor_rows,
        "anchor_scores_oof": oof_scores,
        "candidate_corners_sha256": np.asarray(sha256_array(candidates)),
        "candidate_feature_sha256": np.asarray(sha256_array(matrix)),
        "anchor_identity_sha256": np.asarray(anchor_identity.hexdigest()),
    }


def write_candidate_evidence_v5(path: Path, **kwargs: Any) -> Path:
    _assert_safe_runtime_path(path, "candidate evidence output")
    if not path.name.endswith("_ca1m_tr3d_candidate_evidence_xfit_r2_v5.npz"):
        raise ValueError("v5 evidence filename differs")
    return write_npz_create_only(path, _evidence_payload(**kwargs), "v5 candidate evidence")


@dataclass(frozen=True)
class CandidateEvidenceV5:
    path: Path
    scene_id: str
    fold_id: int
    producer_role: str
    producer_checkpoint_sha256: str
    training_receipt_sha256: str
    outer_continuation_receipt_sha256: str
    b6_oof_sidecar_sha256: str
    candidate_corners: np.ndarray
    candidate_rows: np.ndarray
    candidate_scores: np.ndarray
    anchor_indices: np.ndarray
    features: np.ndarray
    anchor_corners: np.ndarray
    anchor_rows: np.ndarray
    anchor_scores: np.ndarray
    candidate_corners_sha256: str
    candidate_feature_sha256: str
    anchor_identity_sha256: str


def load_candidate_evidence_v5(path: Path, *, expected_scene: str | None = None) -> CandidateEvidenceV5:
    source = _regular(path, "v5 candidate evidence")
    _assert_safe_runtime_path(source, "candidate evidence")
    if not source.name.endswith("_ca1m_tr3d_candidate_evidence_xfit_r2_v5.npz"):
        raise ValueError("v5 evidence filename differs")
    with np.load(source, allow_pickle=False) as archive:
        expected = set(_evidence_payload.__annotations__)  # not used as a schema shortcut
        del expected
        required = {
            "schema", "complete", "create_only", "ground_truth_access", "fold1_access",
            "official_validation_access", "scene_id", "fold_id", "producer_role",
            "producer_checkpoint_sha256", "producer_train_folds", "training_receipt_sha256",
            "outer_continuation_receipt_sha256", "anchor_score_source", "b6_oof_sidecar_sha256",
            "feature_names", "feature_groups_json", "candidate_corners", "candidate_rows",
            "candidate_scores", "anchor_indices", "features", "anchor_corners", "anchor_rows",
            "anchor_scores_oof", "candidate_corners_sha256", "candidate_feature_sha256",
            "anchor_identity_sha256",
        }
        if set(archive.files) != required:
            raise ValueError("v5 candidate evidence key set differs")
        if (
            _scalar(archive, "schema") != EVIDENCE_SCHEMA
            or bool(_scalar(archive, "complete")) is not True
            or bool(_scalar(archive, "create_only")) is not True
            or bool(_scalar(archive, "ground_truth_access")) is not False
            or bool(_scalar(archive, "fold1_access")) is not False
            or bool(_scalar(archive, "official_validation_access")) is not False
            or _scalar(archive, "anchor_score_source") != ANCHOR_SCORE_SOURCE
            or tuple(np.asarray(archive["feature_names"]).astype(str).tolist()) != FEATURE_NAMES
            or json.loads(str(_scalar(archive, "feature_groups_json")))
            != {key: list(value) for key, value in REQUIRED_FEATURE_GROUPS.items()}
        ):
            raise ValueError("v5 candidate evidence scalar contract differs")
        scene = _scene(_scalar(archive, "scene_id"))
        fold = int(_scalar(archive, "fold_id"))
        role = str(_scalar(archive, "producer_role"))
        checkpoint_sha = _sha(_scalar(archive, "producer_checkpoint_sha256"), "checkpoint SHA256")
        receipt_sha = _sha(_scalar(archive, "training_receipt_sha256"), "receipt SHA256")
        continuation_sha = _sha(_scalar(archive, "outer_continuation_receipt_sha256"), "continuation SHA256")
        b6_sha = _sha(_scalar(archive, "b6_oof_sidecar_sha256"), "B6 OOF SHA256")
        arrays = {name: np.array(archive[name], copy=True) for name in (
            "candidate_corners", "candidate_rows", "candidate_scores", "anchor_indices",
            "features", "anchor_corners", "anchor_rows", "anchor_scores_oof",
            "producer_train_folds",
        )}
        recorded = {
            name: _sha(_scalar(archive, name), name)
            for name in ("candidate_corners_sha256", "candidate_feature_sha256", "anchor_identity_sha256")
        }
    if expected_scene is not None and scene != _scene(expected_scene):
        raise ValueError("candidate evidence scene differs")
    train_folds, output_fold = ROLE_SPECS.get(role, ((), -1))
    if fold != output_fold or fold in train_folds or tuple(arrays.pop("producer_train_folds").tolist()) != train_folds:
        raise ValueError("v5 evidence detector OOF topology differs")
    # Re-run the full array contract and compare all derived identities.
    rebuilt = _evidence_payload(
        scene_id=scene, fold_id=fold, producer_role=role,
        producer_checkpoint_sha256=checkpoint_sha, training_receipt_sha256=receipt_sha,
        outer_continuation_receipt_sha256=continuation_sha, b6_oof_sidecar_sha256=b6_sha,
        candidate_corners=arrays["candidate_corners"], candidate_rows=arrays["candidate_rows"],
        candidate_scores=arrays["candidate_scores"], anchor_indices=arrays["anchor_indices"],
        features=arrays["features"], anchor_corners=arrays["anchor_corners"],
        anchor_scores=arrays["anchor_scores_oof"],
    )
    for name in recorded:
        if str(np.asarray(rebuilt[name]).item()) != recorded[name]:
            raise ValueError(f"v5 evidence {name} differs")
    if not np.array_equal(arrays["anchor_rows"], rebuilt["anchor_rows"]):
        raise ValueError("v5 anchor row identity differs")
    return CandidateEvidenceV5(
        path=source, scene_id=scene, fold_id=fold, producer_role=role,
        producer_checkpoint_sha256=checkpoint_sha, training_receipt_sha256=receipt_sha,
        outer_continuation_receipt_sha256=continuation_sha, b6_oof_sidecar_sha256=b6_sha,
        candidate_corners=_readonly(arrays["candidate_corners"]),
        candidate_rows=_readonly(arrays["candidate_rows"]),
        candidate_scores=_readonly(arrays["candidate_scores"]),
        anchor_indices=_readonly(arrays["anchor_indices"]), features=_readonly(arrays["features"]),
        anchor_corners=_readonly(arrays["anchor_corners"]), anchor_rows=_readonly(arrays["anchor_rows"]),
        anchor_scores=_readonly(arrays["anchor_scores_oof"]),
        candidate_corners_sha256=recorded["candidate_corners_sha256"],
        candidate_feature_sha256=recorded["candidate_feature_sha256"],
        anchor_identity_sha256=recorded["anchor_identity_sha256"],
    )


def seal_role_candidate_collection_v5(
    output: Path,
    *,
    role: str,
    expected_scenes: Sequence[str],
    role_receipt: Mapping[str, Any],
    evidence_paths: Mapping[str, Path],
    b6_oof_sidecar: Mapping[str, Any],
) -> Path:
    """Seal one exact-20 role collection without opening ground truth."""

    if role not in ROLE_SPECS:
        raise ValueError("unknown xfit detector role")
    scenes = tuple(_scene(value) for value in expected_scenes)
    if len(scenes) != 20 or len(set(scenes)) != 20:
        raise ValueError(f"{role}: role collection requires exact 20 unique scenes")
    if set(evidence_paths) != set(scenes):
        raise ValueError(f"{role}: evidence map is not exact20")
    receipt_path, receipt = load_detector_role_receipt_v5(role_receipt, role)
    b6_path = _regular(Path(str(b6_oof_sidecar.get("path", ""))), "B6 OOF sidecar")
    b6_sha = _sha(b6_oof_sidecar.get("sha256"), "B6 OOF sidecar SHA256")
    if (
        sha256_file(b6_path) != b6_sha
        or b6_oof_sidecar.get("schema") != "boxfusion.ca1m_native_b6_oof_row_scores.v2"
        or b6_oof_sidecar.get("score_source") != ANCHOR_SCORE_SOURCE
        or b6_oof_sidecar.get("each_row_model_excludes_scene") is not True
        or b6_oof_sidecar.get("deploy_scores") is not False
    ):
        raise ValueError("B6 all-fold OOF binding differs")
    continuation = receipt["outer_continuation_receipt"]
    checkpoint = receipt["checkpoint"]
    rows: list[dict[str, Any]] = []
    candidate_count = 0
    anchor_count = 0
    _, output_fold = ROLE_SPECS[role]
    for scene in scenes:
        evidence = load_candidate_evidence_v5(evidence_paths[scene], expected_scene=scene)
        if (
            evidence.producer_role != role
            or evidence.fold_id != output_fold
            or evidence.producer_checkpoint_sha256 != checkpoint["sha256"]
            or evidence.training_receipt_sha256 != sha256_file(receipt_path)
            or evidence.outer_continuation_receipt_sha256 != continuation["sha256"]
            or evidence.b6_oof_sidecar_sha256 != b6_sha
        ):
            raise ValueError(f"{scene}: evidence/role receipt provenance differs")
        candidate_count += len(evidence.candidate_rows)
        anchor_count += len(evidence.anchor_rows)
        rows.append({
            "schema": SCENE_MANIFEST_SCHEMA,
            "scene_id": scene,
            "fold_id": output_fold,
            "producer_role": role,
            "producer_checkpoint_sha256": evidence.producer_checkpoint_sha256,
            "producer_train_folds": list(ROLE_SPECS[role][0]),
            "path": str(evidence.path),
            "sha256": sha256_file(evidence.path),
            "candidate_count": len(evidence.candidate_rows),
            "anchor_count": len(evidence.anchor_rows),
            "candidate_corners_sha256": evidence.candidate_corners_sha256,
            "candidate_feature_sha256": evidence.candidate_feature_sha256,
            "anchor_identity_sha256": evidence.anchor_identity_sha256,
        })
    payload = {
        "schema": ROLE_COLLECTION_SCHEMA,
        "complete": True,
        "create_only": True,
        "ground_truth_access": False,
        "fold1_access": False,
        "official_validation_access": False,
        "role": role,
        "detector_train_folds": list(ROLE_SPECS[role][0]),
        "candidate_output_fold": output_fold,
        "scene_count": 20,
        "candidate_count": candidate_count,
        "anchor_count": anchor_count,
        "candidate_geometry_oof": True,
        "checkpoint_selection": False,
        "role_receipt": {
            "path": str(receipt_path), "sha256": sha256_file(receipt_path),
            "schema": ROLE_RECEIPT_SCHEMA,
        },
        "b6_oof_sidecar": {
            "path": str(b6_path), "sha256": b6_sha,
            "schema": "boxfusion.ca1m_native_b6_oof_row_scores.v2",
            "score_source": ANCHOR_SCORE_SOURCE,
            "each_row_model_excludes_scene": True,
            "deploy_scores": False,
        },
        "scenes": rows,
    }
    return write_json_create_only(output, payload, f"{role} exact20 candidate collection")


@dataclass(frozen=True)
class CandidateCollectionV5:
    path: Path
    payload: Mapping[str, Any]
    scenes: Mapping[str, Mapping[str, Any]]
    role_manifests: Mapping[str, Path]


def _load_role_collection_v5(
    record: Mapping[str, Any], role: str,
) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    path, value = _record(record, f"{role} candidate collection", schema=ROLE_COLLECTION_SCHEMA)
    train_folds, output_fold = ROLE_SPECS[role]
    if (
        value.get("complete") is not True
        or value.get("create_only") is not True
        or value.get("ground_truth_access") is not False
        or value.get("fold1_access") is not False
        or value.get("official_validation_access") is not False
        or value.get("role") != role
        or tuple(value.get("detector_train_folds", ())) != train_folds
        or value.get("candidate_output_fold") != output_fold
        or value.get("scene_count") != 20
        or value.get("candidate_geometry_oof") is not True
        or value.get("checkpoint_selection") is not False
    ):
        raise ValueError(f"{role}: role collection contract differs")
    receipt_path, receipt = load_detector_role_receipt_v5(value.get("role_receipt"), role)
    if sha256_file(receipt_path) != value["role_receipt"]["sha256"]:
        raise ValueError(f"{role}: role receipt changed")
    b6 = value.get("b6_oof_sidecar") or {}
    b6_path = _regular(Path(str(b6.get("path", ""))), "B6 OOF sidecar")
    if (
        sha256_file(b6_path) != _sha(b6.get("sha256"), "B6 OOF SHA256")
        or b6.get("schema") != "boxfusion.ca1m_native_b6_oof_row_scores.v2"
        or b6.get("score_source") != ANCHOR_SCORE_SOURCE
        or b6.get("each_row_model_excludes_scene") is not True
        or b6.get("deploy_scores") is not False
    ):
        raise ValueError("B6 OOF role binding differs")
    raw_rows = value.get("scenes")
    if not isinstance(raw_rows, list) or len(raw_rows) != 20:
        raise ValueError(f"{role}: role scene rows differ")
    rows = {str(row.get("scene_id")): row for row in raw_rows if isinstance(row, Mapping)}
    if len(rows) != 20:
        raise ValueError(f"{role}: role scenes are not unique exact20")
    candidates = 0
    anchors = 0
    for scene, row in rows.items():
        _scene(scene)
        evidence_path = _regular(Path(str(row.get("path", ""))), f"{scene} evidence")
        if sha256_file(evidence_path) != _sha(row.get("sha256"), f"{scene} evidence SHA256"):
            raise ValueError(f"{scene}: evidence SHA256 differs")
        evidence = load_candidate_evidence_v5(evidence_path, expected_scene=scene)
        if (
            row.get("schema") != SCENE_MANIFEST_SCHEMA
            or row.get("fold_id") != output_fold
            or row.get("producer_role") != role
            or tuple(row.get("producer_train_folds", ())) != train_folds
            or row.get("producer_checkpoint_sha256") != receipt["checkpoint"]["sha256"]
            or row.get("candidate_count") != len(evidence.candidate_rows)
            or row.get("anchor_count") != len(evidence.anchor_rows)
            or row.get("candidate_corners_sha256") != evidence.candidate_corners_sha256
            or row.get("candidate_feature_sha256") != evidence.candidate_feature_sha256
            or row.get("anchor_identity_sha256") != evidence.anchor_identity_sha256
            or evidence.training_receipt_sha256 != sha256_file(receipt_path)
            or evidence.b6_oof_sidecar_sha256 != b6["sha256"]
        ):
            raise ValueError(f"{scene}: role scene provenance differs")
        candidates += len(evidence.candidate_rows)
        anchors += len(evidence.anchor_rows)
    if value.get("candidate_count") != candidates or value.get("anchor_count") != anchors:
        raise ValueError(f"{role}: role aggregate counts differ")
    return path, value, rows


def seal_candidate_collection_v5(
    output: Path,
    *,
    role_collections: Mapping[str, Mapping[str, Any]],
) -> Path:
    """Merge exact inner-60 plus outer-20 role manifests into one sealed collection."""

    if tuple(role_collections) != tuple(ROLE_SPECS):
        raise ValueError("combined collection requires the four frozen roles in order")
    role_records: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    scene_ids: set[str] = set()
    b6_identity: tuple[str, str] | None = None
    continuation_sha: str | None = None
    for role in ROLE_SPECS:
        path, value, rows = _load_role_collection_v5(role_collections[role], role)
        overlap = scene_ids & set(rows)
        if overlap:
            raise ValueError(f"combined candidate scenes overlap: {sorted(overlap)}")
        scene_ids.update(rows)
        b6 = value["b6_oof_sidecar"]
        observed_b6 = (str(b6["path"]), str(b6["sha256"]))
        if b6_identity is None:
            b6_identity = observed_b6
        elif b6_identity != observed_b6:
            raise ValueError("role collections use different B6 OOF artifacts")
        _, receipt = load_detector_role_receipt_v5(value["role_receipt"], role)
        observed_continuation = str(receipt["outer_continuation_receipt"]["sha256"])
        if continuation_sha is None:
            continuation_sha = observed_continuation
        elif continuation_sha != observed_continuation:
            raise ValueError("role collections use different outer continuation receipts")
        role_records.append({
            "role": role, "path": str(path), "sha256": sha256_file(path),
            "schema": ROLE_COLLECTION_SCHEMA, "detector_train_folds": list(ROLE_SPECS[role][0]),
            "candidate_output_fold": ROLE_SPECS[role][1], "scene_count": 20,
        })
        scene_rows.extend(rows[scene] for scene in sorted(rows))
    fold_counts = {str(fold): sum(int(row["fold_id"]) == fold for row in scene_rows) for fold in (0, 2, 3, 4)}
    if len(scene_rows) != 80 or fold_counts != {"0": 20, "2": 20, "3": 20, "4": 20}:
        raise ValueError("combined candidate collection is not exact fit60+outer20")
    payload = {
        "schema": COLLECTION_SCHEMA,
        "complete": True,
        "create_only": True,
        "namespace": NAMESPACE,
        "scene_grouped": True,
        "candidate_geometry_oof_for_every_fit_scene": True,
        "fit_scene_count": 60,
        "outer_scene_count": 20,
        "scene_count": 80,
        "fold_counts": fold_counts,
        "fit_folds": [2, 3, 4],
        "outer_folds": [0],
        "ground_truth_access": False,
        "fold1_access": False,
        "official_validation_access": False,
        "deploy_candidate_or_anchor_scores": False,
        "legacy_v4_candidate_or_policy_reused": False,
        "roles": role_records,
        "b6_oof_sidecar": {
            "path": b6_identity[0], "sha256": b6_identity[1],
            "schema": "boxfusion.ca1m_native_b6_oof_row_scores.v2",
            "score_source": ANCHOR_SCORE_SOURCE,
            "each_row_model_excludes_scene": True,
            "deploy_scores": False,
        },
        "outer_continuation_receipt_sha256": continuation_sha,
        "scenes": sorted(scene_rows, key=lambda row: (int(row["fold_id"]), str(row["scene_id"]))),
    }
    return write_json_create_only(output, payload, "combined exact60+outer20 candidate collection")


def load_candidate_collection_v5(path: Path) -> CandidateCollectionV5:
    source, value = _json(path, "v5 candidate collection")
    if (
        value.get("schema") != COLLECTION_SCHEMA
        or value.get("complete") is not True
        or value.get("create_only") is not True
        or value.get("namespace") != NAMESPACE
        or value.get("scene_grouped") is not True
        or value.get("candidate_geometry_oof_for_every_fit_scene") is not True
        or value.get("fit_scene_count") != 60
        or value.get("outer_scene_count") != 20
        or value.get("scene_count") != 80
        or value.get("fold_counts") != {"0": 20, "2": 20, "3": 20, "4": 20}
        or value.get("fit_folds") != [2, 3, 4]
        or value.get("outer_folds") != [0]
        or value.get("ground_truth_access") is not False
        or value.get("fold1_access") is not False
        or value.get("official_validation_access") is not False
        or value.get("deploy_candidate_or_anchor_scores") is not False
        or value.get("legacy_v4_candidate_or_policy_reused") is not False
    ):
        raise ValueError("v5 combined candidate collection contract differs")
    role_rows = value.get("roles")
    if not isinstance(role_rows, list) or [row.get("role") for row in role_rows] != list(ROLE_SPECS):
        raise ValueError("v5 combined role order differs")
    manifests: dict[str, Path] = {}
    scenes: dict[str, Mapping[str, Any]] = {}
    for row in role_rows:
        role = str(row["role"])
        path, role_value, role_scenes = _load_role_collection_v5(row, role)
        if (
            row.get("detector_train_folds") != list(ROLE_SPECS[role][0])
            or row.get("candidate_output_fold") != ROLE_SPECS[role][1]
            or row.get("scene_count") != 20
        ):
            raise ValueError(f"{role}: combined role row differs")
        if set(scenes) & set(role_scenes):
            raise ValueError("combined scenes overlap")
        scenes.update(role_scenes)
        manifests[role] = path
        if role_value["b6_oof_sidecar"] != value["b6_oof_sidecar"]:
            raise ValueError("combined/role B6 OOF binding differs")
    expected_rows = sorted(scenes.values(), key=lambda row: (int(row["fold_id"]), str(row["scene_id"])))
    if len(scenes) != 80 or value.get("scenes") != expected_rows:
        raise ValueError("combined scene inventory differs from role manifests")
    return CandidateCollectionV5(source, value, scenes, manifests)


@dataclass(frozen=True)
class GateDatasetV5:
    purpose: str
    source_collection_path: Path
    source_collection_sha256: str
    scene_order: np.ndarray
    scene_folds: np.ndarray
    scene_gt_counts: np.ndarray
    anchor_scene_ids: np.ndarray
    anchor_fold_ids: np.ndarray
    anchor_local_indices: np.ndarray
    anchor_corners: np.ndarray
    anchor_scores_oof: np.ndarray
    anchor_best_iou: np.ndarray
    anchor_best_gt: np.ndarray
    candidate_scene_ids: np.ndarray
    candidate_fold_ids: np.ndarray
    candidate_rows: np.ndarray
    candidate_anchor_positions: np.ndarray
    candidate_corners: np.ndarray
    candidate_raw_scores: np.ndarray
    features: np.ndarray
    candidate_max_gt_iou: np.ndarray
    candidate_best_gt: np.ndarray
    candidate_iou_on_anchor_gt: np.ndarray
    same_gt_gain: np.ndarray
    target_switch: np.ndarray
    strict_iou50_target: np.ndarray


def _match_boxes(corners: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairwise = pairwise_world_aabb_iou(corners, gt)
    if not len(gt):
        return (
            np.zeros(len(corners), np.float64),
            np.full(len(corners), -1, np.int64),
            pairwise,
        )
    best_gt = np.argmax(pairwise, axis=1).astype(np.int64)
    return pairwise[np.arange(len(corners)), best_gt], best_gt, pairwise


def _dataset_partition(purpose: str) -> tuple[tuple[int, ...], int, tuple[str, ...]]:
    if purpose == "fold234_oof_fit":
        return FIT_FOLDS, 60, ("inner_holdout2", "inner_holdout3", "inner_holdout4")
    if purpose == "fold0_reused_dev":
        return (0,), 20, ("outer_dev",)
    if purpose == "fold1_locked_once":
        return (1,), 20, ("locked_fold1",)
    raise ValueError("unknown v5 dataset purpose")


def build_labeled_dataset_v5(
    collection_path: Path,
    *,
    purpose: str,
    ground_truth_loader: Callable[[str], Any],
) -> GateDatasetV5:
    """Join labels only after the exact GT-free collection is fully validated.

    ``ground_truth_loader`` is not called during collection validation and is
    called only for the exact partition selected by ``purpose``.  There is no
    fold-1 or validation purpose in this API.
    """

    collection = load_candidate_collection_v5(collection_path)
    folds, expected_count, roles = _dataset_partition(purpose)
    rows = [row for row in collection.scenes.values() if int(row["fold_id"]) in folds]
    rows.sort(key=lambda row: (int(row["fold_id"]), str(row["scene_id"])))
    if (
        len(rows) != expected_count
        or {str(row["producer_role"]) for row in rows} != set(roles)
        or any(int(row["fold_id"]) in tuple(row["producer_train_folds"]) for row in rows)
    ):
        raise ValueError("v5 selected partition is not detector OOF exact scene-grouped")

    scene_order: list[str] = []
    scene_folds: list[int] = []
    scene_gt_counts: list[int] = []
    anchor_scene_ids: list[np.ndarray] = []
    anchor_fold_ids: list[np.ndarray] = []
    anchor_local: list[np.ndarray] = []
    anchor_corners: list[np.ndarray] = []
    anchor_scores: list[np.ndarray] = []
    anchor_best_iou: list[np.ndarray] = []
    anchor_best_gt: list[np.ndarray] = []
    candidate_scene_ids: list[np.ndarray] = []
    candidate_fold_ids: list[np.ndarray] = []
    candidate_rows: list[np.ndarray] = []
    candidate_anchor_positions: list[np.ndarray] = []
    candidate_corners: list[np.ndarray] = []
    candidate_scores: list[np.ndarray] = []
    candidate_features: list[np.ndarray] = []
    candidate_best_iou: list[np.ndarray] = []
    candidate_best_gt: list[np.ndarray] = []
    candidate_iou_on_anchor_gt: list[np.ndarray] = []
    same_gt_gain: list[np.ndarray] = []
    target_switch: list[np.ndarray] = []
    anchor_offset = 0

    for row in rows:
        scene = str(row["scene_id"])
        fold = int(row["fold_id"])
        evidence = load_candidate_evidence_v5(Path(str(row["path"])), expected_scene=scene)
        # This is the first and only point at which caller-owned GT is reached.
        gt = np.asarray(ground_truth_loader(scene), dtype=np.float64)
        if gt.ndim != 3 or gt.shape[1:] != (8, 3) or not np.isfinite(gt).all():
            raise ValueError(f"{scene}: GT corners must be finite [G,8,3]")
        # Positive-volume validation is shared with the official AABB metric.
        pairwise_world_aabb_iou(np.empty((0, 8, 3), np.float64), gt)
        a_iou, a_gt, _ = _match_boxes(evidence.anchor_corners, gt)
        c_iou, c_gt, c_matrix = _match_boxes(evidence.candidate_corners, gt)
        local_anchor = evidence.anchor_indices.astype(np.int64, copy=False)
        on_anchor_gt = np.zeros(len(local_anchor), np.float64)
        valid_anchor_gt = a_gt[local_anchor] >= 0
        if np.any(valid_anchor_gt):
            candidate_positions = np.flatnonzero(valid_anchor_gt)
            on_anchor_gt[candidate_positions] = c_matrix[
                candidate_positions, a_gt[local_anchor[candidate_positions]]
            ]
        switch = valid_anchor_gt & (c_gt != a_gt[local_anchor])
        gain = on_anchor_gt - a_iou[local_anchor]

        scene_order.append(scene)
        scene_folds.append(fold)
        scene_gt_counts.append(len(gt))
        anchor_scene_ids.append(np.full(len(evidence.anchor_rows), scene))
        anchor_fold_ids.append(np.full(len(evidence.anchor_rows), fold, np.int64))
        anchor_local.append(evidence.anchor_rows.astype(np.int64, copy=False))
        anchor_corners.append(evidence.anchor_corners)
        anchor_scores.append(evidence.anchor_scores)
        anchor_best_iou.append(a_iou)
        anchor_best_gt.append(a_gt)
        candidate_scene_ids.append(np.full(len(evidence.candidate_rows), scene))
        candidate_fold_ids.append(np.full(len(evidence.candidate_rows), fold, np.int64))
        candidate_rows.append(evidence.candidate_rows)
        candidate_anchor_positions.append(local_anchor + anchor_offset)
        candidate_corners.append(evidence.candidate_corners)
        candidate_scores.append(evidence.candidate_scores)
        candidate_features.append(evidence.features)
        candidate_best_iou.append(c_iou)
        candidate_best_gt.append(c_gt)
        candidate_iou_on_anchor_gt.append(on_anchor_gt)
        same_gt_gain.append(gain)
        target_switch.append(switch)
        anchor_offset += len(evidence.anchor_rows)

    def concatenate(values: list[np.ndarray], shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        if values:
            return _readonly(np.concatenate(values, axis=0), dtype)
        return _readonly(np.empty(shape, dtype=dtype))

    dataset = GateDatasetV5(
        purpose=purpose,
        source_collection_path=collection.path,
        source_collection_sha256=sha256_file(collection.path),
        scene_order=_readonly(np.asarray(scene_order)),
        scene_folds=_readonly(scene_folds, np.int64),
        scene_gt_counts=_readonly(scene_gt_counts, np.int64),
        anchor_scene_ids=concatenate(anchor_scene_ids, (0,), np.str_),
        anchor_fold_ids=concatenate(anchor_fold_ids, (0,), np.int64),
        anchor_local_indices=concatenate(anchor_local, (0,), np.int64),
        anchor_corners=concatenate(anchor_corners, (0, 8, 3), np.float32),
        anchor_scores_oof=concatenate(anchor_scores, (0,), np.float32),
        anchor_best_iou=concatenate(anchor_best_iou, (0,), np.float64),
        anchor_best_gt=concatenate(anchor_best_gt, (0,), np.int64),
        candidate_scene_ids=concatenate(candidate_scene_ids, (0,), np.str_),
        candidate_fold_ids=concatenate(candidate_fold_ids, (0,), np.int64),
        candidate_rows=concatenate(candidate_rows, (0,), np.int64),
        candidate_anchor_positions=concatenate(candidate_anchor_positions, (0,), np.int64),
        candidate_corners=concatenate(candidate_corners, (0, 8, 3), np.float32),
        candidate_raw_scores=concatenate(candidate_scores, (0,), np.float32),
        features=concatenate(candidate_features, (0, len(FEATURE_NAMES)), np.float32),
        candidate_max_gt_iou=concatenate(candidate_best_iou, (0,), np.float64),
        candidate_best_gt=concatenate(candidate_best_gt, (0,), np.int64),
        candidate_iou_on_anchor_gt=concatenate(candidate_iou_on_anchor_gt, (0,), np.float64),
        same_gt_gain=concatenate(same_gt_gain, (0,), np.float64),
        target_switch=concatenate(target_switch, (0,), np.bool_),
        strict_iou50_target=concatenate(
            [value > 0.50 for value in candidate_best_iou], (0,), np.bool_
        ),
    )
    validate_gate_dataset_v5(dataset, expected_purpose=purpose)
    return dataset


def validate_gate_dataset_v5(dataset: GateDatasetV5, *, expected_purpose: str | None = None) -> None:
    folds, scene_count, _ = _dataset_partition(dataset.purpose)
    if expected_purpose is not None and dataset.purpose != expected_purpose:
        raise ValueError("v5 dataset purpose differs")
    scenes = np.asarray(dataset.scene_order).astype(str)
    scene_folds = np.asarray(dataset.scene_folds, np.int64)
    gt_counts = np.asarray(dataset.scene_gt_counts, np.int64)
    a = len(dataset.anchor_scene_ids)
    n = len(dataset.candidate_scene_ids)
    if (
        len(scenes) != scene_count or len(set(scenes.tolist())) != scene_count
        or scene_folds.shape != (scene_count,) or gt_counts.shape != (scene_count,)
        or set(scene_folds.tolist()) != set(folds)
        or any(int(np.sum(scene_folds == fold)) != 20 for fold in folds)
        or np.any(gt_counts < 0)
        or dataset.anchor_fold_ids.shape != (a,)
        or dataset.anchor_local_indices.shape != (a,)
        or dataset.anchor_corners.shape != (a, 8, 3)
        or dataset.anchor_scores_oof.shape != (a,)
        or dataset.anchor_best_iou.shape != (a,)
        or dataset.anchor_best_gt.shape != (a,)
        or dataset.candidate_fold_ids.shape != (n,)
        or dataset.candidate_rows.shape != (n,)
        or dataset.candidate_anchor_positions.shape != (n,)
        or dataset.candidate_corners.shape != (n, 8, 3)
        or dataset.candidate_raw_scores.shape != (n,)
        or dataset.features.shape != (n, len(FEATURE_NAMES))
        or dataset.candidate_max_gt_iou.shape != (n,)
        or dataset.candidate_best_gt.shape != (n,)
        or dataset.candidate_iou_on_anchor_gt.shape != (n,)
        or dataset.same_gt_gain.shape != (n,)
        or dataset.target_switch.shape != (n,)
        or dataset.strict_iou50_target.shape != (n,)
        or np.any((dataset.candidate_anchor_positions < 0) | (dataset.candidate_anchor_positions >= a))
        or not np.array_equal(dataset.strict_iou50_target, dataset.candidate_max_gt_iou > 0.50)
        or not all(np.isfinite(value).all() for value in (
            dataset.anchor_corners, dataset.anchor_scores_oof, dataset.anchor_best_iou,
            dataset.candidate_corners, dataset.candidate_raw_scores, dataset.features,
            dataset.candidate_max_gt_iou, dataset.candidate_iou_on_anchor_gt, dataset.same_gt_gain,
        ))
        or np.any((dataset.anchor_scores_oof < 0.0) | (dataset.anchor_scores_oof > 1.0))
        or np.any((dataset.candidate_max_gt_iou < 0.0) | (dataset.candidate_max_gt_iou > 1.0))
    ):
        raise ValueError("v5 labeled dataset array contract differs")
    for scene, fold in zip(scenes.tolist(), scene_folds.tolist()):
        anchor_rows = np.flatnonzero(dataset.anchor_scene_ids.astype(str) == scene)
        candidate_rows = np.flatnonzero(dataset.candidate_scene_ids.astype(str) == scene)
        if (
            not len(anchor_rows)
            or np.any(dataset.anchor_fold_ids[anchor_rows] != fold)
            or np.any(dataset.candidate_fold_ids[candidate_rows] != fold)
            or not np.array_equal(dataset.anchor_local_indices[anchor_rows], np.arange(len(anchor_rows)))
            or np.any(~np.isin(dataset.candidate_anchor_positions[candidate_rows], anchor_rows))
        ):
            raise ValueError(f"{scene}: v5 scene row identity differs")


def _dataset_npz_payload(dataset: GateDatasetV5) -> dict[str, Any]:
    validate_gate_dataset_v5(dataset)
    return {
        "schema": np.asarray(DATASET_SCHEMA),
        "complete": np.asarray(True),
        "create_only": np.asarray(True),
        "purpose": np.asarray(dataset.purpose),
        "scene_grouped": np.asarray(True),
        "candidate_geometry_detector_oof": np.asarray(True),
        "anchor_scores_all_fold_oof": np.asarray(True),
        "deploy_scores_used": np.asarray(False),
        "fold1_access": np.asarray(False),
        "official_validation_access": np.asarray(False),
        "feature_names": np.asarray(FEATURE_NAMES),
        "raw_score_feature_indices": np.asarray(RAW_SCORE_FEATURE_INDICES, np.int64),
        "source_collection_path": np.asarray(str(dataset.source_collection_path)),
        "source_collection_sha256": np.asarray(dataset.source_collection_sha256),
        **{
            name: np.asarray(getattr(dataset, name))
            for name in GateDatasetV5.__dataclass_fields__
            if name not in {"purpose", "source_collection_path", "source_collection_sha256"}
        },
    }


def seal_gate_dataset_v5(
    dataset: GateDatasetV5, *, artifact_path: Path, manifest_path: Path
) -> tuple[Path, Path]:
    artifact = write_npz_create_only(artifact_path, _dataset_npz_payload(dataset), "v5 gate dataset")
    payload = {
        "schema": DATASET_MANIFEST_SCHEMA,
        "complete": True,
        "create_only": True,
        "purpose": dataset.purpose,
        "scene_count": len(dataset.scene_order),
        "anchor_count": len(dataset.anchor_scene_ids),
        "candidate_count": len(dataset.candidate_scene_ids),
        "fold_ids": sorted(set(dataset.scene_folds.tolist())),
        "scene_grouped": True,
        "detector_oof": True,
        "anchor_scores_all_fold_oof": True,
        "deploy_or_in_sample_scores": False,
        "fold1_access": False,
        "official_validation_access": False,
        "artifact": {"path": str(artifact), "sha256": sha256_file(artifact), "schema": DATASET_SCHEMA},
        "source_collection": {
            "path": str(dataset.source_collection_path),
            "sha256": dataset.source_collection_sha256,
            "schema": COLLECTION_SCHEMA,
        },
    }
    manifest = write_json_create_only(manifest_path, payload, "v5 gate dataset manifest")
    return artifact, manifest


@dataclass(frozen=True)
class LinearHeadV5:
    weights: np.ndarray
    bias: float


@dataclass(frozen=True)
class GateModelV5:
    train_folds: tuple[int, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    iou_regression: LinearHeadV5
    iou50_calibration: LinearHeadV5
    pairwise_benefit: LinearHeadV5
    pairwise_gain_scale: float
    pairwise_gain_bias: float
    pair_count: int


@dataclass(frozen=True)
class GatePredictionsV5:
    candidate_iou: np.ndarray
    iou50_probability: np.ndarray
    same_gt_gain: np.ndarray
    scoring_train_fold_json: np.ndarray


@dataclass(frozen=True)
class GateOOFResultV5:
    predictions: GatePredictionsV5
    fold_models: Mapping[int, GateModelV5]
    final_model: GateModelV5
    threshold_receipt: Mapping[str, Any]


def _stable_sigmoid(value: Any) -> np.ndarray:
    source = np.asarray(value, dtype=np.float64)
    result = np.empty_like(source)
    positive = source >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-np.minimum(source[positive], 700.0)))
    exponential = np.exp(np.maximum(source[~positive], -700.0))
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _scene_equal_weights(scene_ids: Any) -> np.ndarray:
    scenes = np.asarray(scene_ids).astype(str)
    unique = np.unique(scenes)
    if not len(scenes) or not len(unique):
        raise ValueError("scene-equal weights require rows")
    weights = np.empty(len(scenes), np.float64)
    for scene in unique.tolist():
        rows = scenes == scene
        weights[rows] = 1.0 / (len(unique) * int(np.sum(rows)))
    return weights


def _weighted_standardize(x: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = float(np.sum(weights))
    mean = np.sum(x * weights[:, None], axis=0) / total
    variance = np.sum(np.square(x - mean) * weights[:, None], axis=0) / total
    scale = np.sqrt(np.maximum(variance, 1.0e-12))
    scale[scale < 1.0e-6] = 1.0
    return (x - mean) / scale, mean, scale


def _ridge_solve(x: np.ndarray, y: np.ndarray, weights: np.ndarray, penalty: np.ndarray) -> tuple[np.ndarray, float]:
    design = np.concatenate((x, np.ones((len(x), 1), np.float64)), axis=1)
    gram = design.T @ (weights[:, None] * design)
    gram += np.diag(np.concatenate((penalty, np.zeros(1, np.float64))))
    rhs = design.T @ (weights * y)
    try:
        solution = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    return solution[:-1], float(solution[-1])


def _fit_huber_head(
    x: np.ndarray, target: np.ndarray, scene_weights: np.ndarray,
    penalty: np.ndarray, *, delta: float = 0.10,
) -> LinearHeadV5:
    weights, bias = _ridge_solve(x, target, scene_weights, penalty)
    for _ in range(24):
        residual = target - (x @ weights + bias)
        robust = np.ones(len(residual), np.float64)
        large = np.abs(residual) > delta
        robust[large] = delta / np.abs(residual[large])
        updated, updated_bias = _ridge_solve(x, target, scene_weights * robust, penalty)
        if np.max(np.abs(updated - weights), initial=0.0) < 1.0e-9 and abs(updated_bias - bias) < 1.0e-9:
            weights, bias = updated, updated_bias
            break
        weights, bias = updated, updated_bias
    return LinearHeadV5(_readonly(weights, np.float64), bias)


def _fit_logistic_head(
    x: np.ndarray, target: np.ndarray, scene_weights: np.ndarray, penalty: np.ndarray,
) -> LinearHeadV5:
    y = np.asarray(target, np.float64)
    positives = y > 0.5
    if not np.any(positives) or np.all(positives):
        raise ValueError("strict IoU50 calibration requires both strict classes")
    balanced = scene_weights.copy()
    balanced[positives] *= 0.5 / float(np.sum(scene_weights[positives]))
    balanced[~positives] *= 0.5 / float(np.sum(scene_weights[~positives]))
    weights = np.zeros(x.shape[1], np.float64)
    prevalence = float(np.average(y, weights=balanced))
    bias = math.log(prevalence / max(1.0 - prevalence, 1.0e-12))
    for _ in range(48):
        logits = x @ weights + bias
        probability = np.clip(_stable_sigmoid(logits), 1.0e-6, 1.0 - 1.0e-6)
        curvature = balanced * probability * (1.0 - probability)
        working = logits + (y - probability) / (probability * (1.0 - probability))
        updated, updated_bias = _ridge_solve(x, working, curvature, penalty)
        if np.max(np.abs(updated - weights), initial=0.0) < 1.0e-8 and abs(updated_bias - bias) < 1.0e-8:
            weights, bias = updated, updated_bias
            break
        weights, bias = updated, updated_bias
    return LinearHeadV5(_readonly(weights, np.float64), bias)


def _pairwise_training_rows(
    dataset: GateDatasetV5, fit_rows: np.ndarray, standardized: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    allowed = np.zeros(len(dataset.candidate_scene_ids), np.bool_)
    allowed[fit_rows] = True
    utility = np.asarray(dataset.same_gt_gain, np.float64).copy()
    utility[np.asarray(dataset.target_switch, np.bool_)] = -1.0
    pair_x: list[np.ndarray] = []
    pair_scene: list[str] = []
    pair_group: list[int] = []
    for anchor in np.unique(dataset.candidate_anchor_positions[fit_rows]).tolist():
        rows = np.flatnonzero(allowed & (dataset.candidate_anchor_positions == anchor))
        if len(rows) < 2:
            continue
        pairs: list[tuple[float, int, int]] = []
        for left_index in range(len(rows)):
            for right_index in range(left_index + 1, len(rows)):
                left, right = int(rows[left_index]), int(rows[right_index])
                difference = float(utility[left] - utility[right])
                if abs(difference) < 0.05:
                    continue
                high, low = (left, right) if difference > 0.0 else (right, left)
                pairs.append((abs(difference), high, low))
        pairs.sort(key=lambda row: (-row[0], row[1], row[2]))
        for _, high, low in pairs[:64]:
            pair_x.append(standardized[high] - standardized[low])
            pair_scene.append(str(dataset.candidate_scene_ids[high]))
            pair_group.append(int(anchor))
    if not pair_x:
        raise ValueError("pairwise benefit head requires within-anchor margin pairs")
    matrix = np.asarray(pair_x, np.float64)
    scenes = np.asarray(pair_scene)
    groups = np.asarray(pair_group, np.int64)
    weights = np.empty(len(matrix), np.float64)
    unique_scenes = np.unique(scenes)
    for scene in unique_scenes.tolist():
        scene_rows = scenes == scene
        scene_groups = np.unique(groups[scene_rows])
        for group in scene_groups.tolist():
            rows = scene_rows & (groups == group)
            weights[rows] = 1.0 / (
                len(unique_scenes) * len(scene_groups) * int(np.sum(rows))
            )
    return matrix, weights, len(matrix)


def _fit_pairwise_head(
    dataset: GateDatasetV5, fit_rows: np.ndarray, standardized: np.ndarray,
    penalty: np.ndarray, scene_weights: np.ndarray,
) -> tuple[LinearHeadV5, float, float, int]:
    pair_x, pair_weight, pair_count = _pairwise_training_rows(dataset, fit_rows, standardized)
    weights = np.zeros(standardized.shape[1], np.float64)
    for iteration in range(1200):
        probability = _stable_sigmoid(pair_x @ weights)
        gradient = -(pair_x.T @ (pair_weight * (1.0 - probability))) + penalty * weights
        learning_rate = 0.20 / math.sqrt(1.0 + iteration / 50.0)
        updated = weights - learning_rate * gradient
        if np.max(np.abs(updated - weights), initial=0.0) < 1.0e-9:
            weights = updated
            break
        weights = updated
    raw = standardized[fit_rows] @ weights
    target = np.asarray(dataset.same_gt_gain[fit_rows], np.float64).copy()
    target[np.asarray(dataset.target_switch[fit_rows], np.bool_)] = -1.0
    calibration_x = raw[:, None]
    calibrated_weight, calibrated_bias = _ridge_solve(
        calibration_x, target, scene_weights, np.asarray((0.01,), np.float64)
    )
    scale = max(float(calibrated_weight[0]), 0.0)
    return LinearHeadV5(_readonly(weights, np.float64), 0.0), scale, calibrated_bias, pair_count


def fit_gate_model_v5(dataset: GateDatasetV5, *, train_folds: Sequence[int]) -> GateModelV5:
    validate_gate_dataset_v5(dataset, expected_purpose="fold234_oof_fit")
    folds = tuple(int(value) for value in train_folds)
    if folds not in ((3, 4), (2, 4), (2, 3), (2, 3, 4)):
        raise ValueError("v5 model train-fold topology differs")
    fit_rows = np.flatnonzero(np.isin(dataset.candidate_fold_ids, folds))
    if not len(fit_rows) or set(dataset.candidate_fold_ids[fit_rows].tolist()) != set(folds):
        raise ValueError("v5 fit rows do not cover the requested scene folds")
    x = np.asarray(dataset.features, np.float64)
    row_weights = _scene_equal_weights(dataset.candidate_scene_ids[fit_rows])
    fit_x, mean, scale = _weighted_standardize(x[fit_rows], row_weights)
    standardized = (x - mean) / scale
    penalty = np.full(x.shape[1], 0.002, np.float64)
    penalty[list(RAW_SCORE_FEATURE_INDICES)] *= 4.0
    iou = _fit_huber_head(
        fit_x, np.asarray(dataset.candidate_max_gt_iou[fit_rows], np.float64),
        row_weights, penalty,
    )
    iou50 = _fit_logistic_head(
        fit_x, np.asarray(dataset.strict_iou50_target[fit_rows], np.float64),
        row_weights, penalty,
    )
    pairwise, gain_scale, gain_bias, pair_count = _fit_pairwise_head(
        dataset, fit_rows, standardized, penalty, row_weights
    )
    return GateModelV5(
        train_folds=folds, feature_mean=_readonly(mean, np.float64),
        feature_scale=_readonly(scale, np.float64), iou_regression=iou,
        iou50_calibration=iou50, pairwise_benefit=pairwise,
        pairwise_gain_scale=gain_scale, pairwise_gain_bias=gain_bias,
        pair_count=pair_count,
    )


def predict_gate_model_v5(model: GateModelV5, features: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(features, np.float64)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES) or not np.isfinite(values).all():
        raise ValueError("v5 prediction features differ")
    standardized = (values - model.feature_mean) / model.feature_scale
    iou = np.clip(
        standardized @ model.iou_regression.weights + model.iou_regression.bias,
        0.0, 1.0,
    )
    probability = _stable_sigmoid(
        standardized @ model.iou50_calibration.weights + model.iou50_calibration.bias
    )
    raw_benefit = standardized @ model.pairwise_benefit.weights
    gain = np.clip(model.pairwise_gain_scale * raw_benefit + model.pairwise_gain_bias, -1.0, 1.0)
    return _readonly(iou, np.float64), _readonly(probability, np.float64), _readonly(gain, np.float64)


@dataclass(frozen=True)
class SelectionV5:
    candidate_positions: np.ndarray
    anchor_positions: np.ndarray


def select_replacements_v5(
    dataset: GateDatasetV5, predictions: GatePredictionsV5,
    *, iou_threshold: float, gain_threshold: float, probability_threshold: float,
) -> SelectionV5:
    n = len(dataset.candidate_scene_ids)
    iou = np.asarray(predictions.candidate_iou, np.float64)
    probability = np.asarray(predictions.iou50_probability, np.float64)
    gain = np.asarray(predictions.same_gt_gain, np.float64)
    if any(value.shape != (n,) or not np.isfinite(value).all() for value in (iou, probability, gain)):
        raise ValueError("v5 selection prediction arrays differ")
    eligible = np.flatnonzero(
        (iou >= float(iou_threshold))
        & (gain >= float(gain_threshold))
        & (probability >= float(probability_threshold))
    )
    winners: list[int] = []
    for anchor in np.unique(dataset.candidate_anchor_positions[eligible]).tolist():
        rows = eligible[dataset.candidate_anchor_positions[eligible] == anchor]
        order = np.lexsort((
            dataset.candidate_rows[rows],
            -dataset.candidate_raw_scores[rows].astype(np.float64),
            -probability[rows], -iou[rows], -gain[rows],
        ))
        winners.append(int(rows[order[0]]))
    selected: list[int] = []
    for scene in np.unique(dataset.candidate_scene_ids).tolist():
        rows = np.asarray([row for row in winners if str(dataset.candidate_scene_ids[row]) == str(scene)], np.int64)
        if not len(rows):
            continue
        order = np.lexsort((
            dataset.candidate_rows[rows],
            -dataset.candidate_raw_scores[rows].astype(np.float64),
            -probability[rows], -iou[rows], -gain[rows],
        ))
        selected.extend(rows[order[:MAX_REPLACEMENTS_PER_SCENE]].tolist())
    selected_array = np.asarray(selected, np.int64)
    anchor_array = dataset.candidate_anchor_positions[selected_array]
    if len(np.unique(anchor_array)) != len(anchor_array):
        raise RuntimeError("v5 selection emitted duplicate anchors")
    return SelectionV5(_readonly(selected_array), _readonly(anchor_array, np.int64))


def _evaluate_selection_v5(
    dataset: GateDatasetV5, selection: SelectionV5,
) -> dict[str, Any]:
    baseline_iou = np.asarray(dataset.anchor_best_iou, np.float64)
    baseline_gt = np.asarray(dataset.anchor_best_gt, np.int64)
    active_iou = baseline_iou.copy()
    active_gt = baseline_gt.copy()
    selected = np.asarray(selection.candidate_positions, np.int64)
    anchors = np.asarray(selection.anchor_positions, np.int64)
    active_iou[anchors] = dataset.candidate_max_gt_iou[selected]
    active_gt[anchors] = dataset.candidate_best_gt[selected]
    gt_count = int(np.sum(dataset.scene_gt_counts))
    baseline = official_ca_ap(
        scene_ids=dataset.anchor_scene_ids,
        scores=dataset.anchor_scores_oof,
        best_iou=baseline_iou,
        best_gt=baseline_gt,
        ground_truth_count=gt_count,
    )
    active = official_ca_ap(
        scene_ids=dataset.anchor_scene_ids,
        scores=dataset.anchor_scores_oof,
        best_iou=active_iou,
        best_gt=active_gt,
        ground_truth_count=gt_count,
    )
    delta = metric_delta(active, baseline)
    gains = np.asarray(dataset.same_gt_gain[selected], np.float64)
    switches = np.asarray(dataset.target_switch[selected], np.bool_)
    count = len(selected)
    positive_fraction = float(np.mean(gains > 0.0)) if count else 0.0
    severe = switches | (gains <= -0.05)
    severe_fraction = float(np.mean(severe)) if count else 0.0
    switch_fraction = float(np.mean(switches)) if count else 0.0
    scene_count = len(set(dataset.candidate_scene_ids[selected].astype(str).tolist()))
    return {
        "baseline": baseline,
        "active": active,
        "delta_ap": delta,
        "replacement_count": count,
        "replacement_scene_count": scene_count,
        "positive_gain_fraction": positive_fraction,
        "severe_harm_fraction": severe_fraction,
        "target_switch_fraction": switch_fraction,
        "selected_candidate_positions": selected.tolist(),
        "selected_anchor_positions": anchors.tolist(),
        "scores_preserved": True,
        "row_order_preserved": True,
        "row_count_preserved": True,
    }


def _safety_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    delta = report["delta_ap"]
    checks = {
        "delta_ap15": float(delta["iou_0.15"]) >= OOF_SAFETY_GATE["min_delta_ap15"],
        "delta_ap25": float(delta["iou_0.25"]) >= OOF_SAFETY_GATE["min_delta_ap25"],
        "delta_ap50": float(delta["iou_0.50"]) >= OOF_SAFETY_GATE["min_delta_ap50"],
        "replacement_count": int(report["replacement_count"]) >= OOF_SAFETY_GATE["min_replacements"],
        "replacement_scene_count": int(report["replacement_scene_count"]) >= OOF_SAFETY_GATE["min_scenes"],
        "positive_gain_fraction": float(report["positive_gain_fraction"])
        >= OOF_SAFETY_GATE["min_positive_gain_fraction"],
        "severe_harm_fraction": float(report["severe_harm_fraction"])
        <= OOF_SAFETY_GATE["max_severe_harm_fraction"],
        "target_switch_fraction": float(report["target_switch_fraction"])
        <= OOF_SAFETY_GATE["max_target_switch_fraction"],
    }
    return {"pass": all(checks.values()), "checks": checks, "requirements": dict(OOF_SAFETY_GATE)}


def _model_payload(model: GateModelV5) -> dict[str, Any]:
    def head(value: LinearHeadV5) -> dict[str, Any]:
        return {"weights": value.weights.tolist(), "bias": float(value.bias)}
    return {
        "schema": MODEL_SCHEMA,
        "family": "three_head_low_capacity_linear_v1",
        "train_folds": list(model.train_folds),
        "scene_grouped": True,
        "standardization_fit_scenes_only": True,
        "raw_score_feature_penalty_multiplier": 4.0,
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": model.feature_mean.tolist(),
        "feature_scale": model.feature_scale.tolist(),
        "candidate_iou_regression": {
            **head(model.iou_regression), "loss": "huber_delta_0.10",
        },
        "candidate_iou50_calibration": {
            **head(model.iou50_calibration), "target": "candidate_max_gt_iou_strict_gt_0.50",
            "loss": "scene_balanced_binary_cross_entropy",
        },
        "pairwise_groupwise_benefit": {
            **head(model.pairwise_benefit),
            "gain_scale": float(model.pairwise_gain_scale),
            "gain_bias": float(model.pairwise_gain_bias),
            "pair_count": int(model.pair_count),
            "preference_margin": 0.05,
            "target_switch_is_harm": True,
            "loss": "scene_and_anchor_group_balanced_pairwise_logistic",
        },
    }


def _model_from_payload(value: Mapping[str, Any]) -> GateModelV5:
    if (
        value.get("schema") != MODEL_SCHEMA
        or value.get("family") != "three_head_low_capacity_linear_v1"
        or value.get("scene_grouped") is not True
        or value.get("standardization_fit_scenes_only") is not True
        or value.get("raw_score_feature_penalty_multiplier") != 4.0
        or tuple(value.get("feature_names", ())) != FEATURE_NAMES
    ):
        raise ValueError("v5 low-capacity model contract differs")
    train_folds = tuple(value.get("train_folds", ()))
    if train_folds not in ((3, 4), (2, 4), (2, 3), (2, 3, 4)):
        raise ValueError("v5 model train folds differ")
    mean = np.asarray(value.get("feature_mean"), np.float64)
    scale = np.asarray(value.get("feature_scale"), np.float64)
    if (
        mean.shape != (len(FEATURE_NAMES),) or scale.shape != mean.shape
        or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0.0)
    ):
        raise ValueError("v5 model standardization differs")
    def head(name: str) -> LinearHeadV5:
        record = value.get(name) or {}
        weights = np.asarray(record.get("weights"), np.float64)
        bias = float(record.get("bias", math.nan))
        if weights.shape != mean.shape or not np.isfinite(weights).all() or not math.isfinite(bias):
            raise ValueError(f"v5 model head {name} differs")
        return LinearHeadV5(_readonly(weights, np.float64), bias)
    regression = value.get("candidate_iou_regression") or {}
    calibration = value.get("candidate_iou50_calibration") or {}
    pairwise = value.get("pairwise_groupwise_benefit") or {}
    if (
        regression.get("loss") != "huber_delta_0.10"
        or calibration.get("target") != "candidate_max_gt_iou_strict_gt_0.50"
        or calibration.get("loss") != "scene_balanced_binary_cross_entropy"
        or pairwise.get("loss") != "scene_and_anchor_group_balanced_pairwise_logistic"
        or pairwise.get("preference_margin") != 0.05
        or pairwise.get("target_switch_is_harm") is not True
        or int(pairwise.get("pair_count", 0)) < 1
        or not math.isfinite(float(pairwise.get("gain_scale", math.nan)))
        or float(pairwise.get("gain_scale", -1.0)) < 0.0
        or not math.isfinite(float(pairwise.get("gain_bias", math.nan)))
    ):
        raise ValueError("v5 three-head science contract differs")
    return GateModelV5(
        train_folds=train_folds, feature_mean=_readonly(mean), feature_scale=_readonly(scale),
        iou_regression=head("candidate_iou_regression"),
        iou50_calibration=head("candidate_iou50_calibration"),
        pairwise_benefit=head("pairwise_groupwise_benefit"),
        pairwise_gain_scale=float(pairwise["gain_scale"]),
        pairwise_gain_bias=float(pairwise["gain_bias"]), pair_count=int(pairwise["pair_count"]),
    )


def train_gate_oof_v5(dataset: GateDatasetV5) -> GateOOFResultV5:
    """Train double-OOF gate heads and select thresholds from fold234 only."""

    validate_gate_dataset_v5(dataset, expected_purpose="fold234_oof_fit")
    n = len(dataset.candidate_scene_ids)
    pred_iou = np.full(n, np.nan, np.float64)
    pred_probability = np.full(n, np.nan, np.float64)
    pred_gain = np.full(n, np.nan, np.float64)
    train_fold_json = np.full(n, "", dtype="<U8")
    models: dict[int, GateModelV5] = {}
    for heldout in FIT_FOLDS:
        train_folds = OOF_GATE_ROLES[heldout]
        model = fit_gate_model_v5(dataset, train_folds=train_folds)
        rows = np.flatnonzero(dataset.candidate_fold_ids == heldout)
        if heldout in model.train_folds or not len(rows):
            raise RuntimeError("v5 gate OOF topology leaked its heldout fold")
        iou, probability, gain = predict_gate_model_v5(model, dataset.features[rows])
        pred_iou[rows], pred_probability[rows], pred_gain[rows] = iou, probability, gain
        train_fold_json[rows] = json.dumps(list(train_folds), separators=(",", ":"))
        models[heldout] = model
    if (
        not np.isfinite(pred_iou).all() or not np.isfinite(pred_probability).all()
        or not np.isfinite(pred_gain).all() or np.any(train_fold_json == "")
    ):
        raise RuntimeError("v5 gate OOF predictions are incomplete")
    predictions = GatePredictionsV5(
        _readonly(pred_iou), _readonly(pred_probability), _readonly(pred_gain),
        _readonly(train_fold_json),
    )
    search: list[dict[str, Any]] = []
    for iou_threshold in IOU_GRID:
        for gain_threshold in GAIN_GRID:
            for probability_threshold in PROB_GRID:
                selection = select_replacements_v5(
                    dataset, predictions, iou_threshold=iou_threshold,
                    gain_threshold=gain_threshold, probability_threshold=probability_threshold,
                )
                evaluation = _evaluate_selection_v5(dataset, selection)
                gate = _safety_gate(evaluation)
                search.append({
                    "thresholds": {
                        "candidate_iou": iou_threshold,
                        "same_gt_gain": gain_threshold,
                        "iou50_probability": probability_threshold,
                    },
                    "evaluation": evaluation,
                    "safety_gate": gate,
                })
    passing = [row for row in search if row["safety_gate"]["pass"]]
    def objective(row: Mapping[str, Any]) -> tuple[Any, ...]:
        evaluation = row["evaluation"]
        delta = evaluation["delta_ap"]
        thresholds = row["thresholds"]
        return (
            float(delta["iou_0.50"]), float(delta["iou_0.25"]), float(delta["iou_0.15"]),
            float(evaluation["positive_gain_fraction"]), -int(evaluation["replacement_count"]),
            float(thresholds["candidate_iou"]), float(thresholds["same_gt_gain"]),
            float(thresholds["iou50_probability"]),
        )
    chosen = max(passing, key=objective) if passing else None
    receipt = {
        "schema": THRESHOLD_SCHEMA,
        "complete": True,
        "scene_grouped_oof": True,
        "threshold_source_folds": [2, 3, 4],
        "fold0_used_for_fit_or_selection": False,
        "fold1_or_validation_access": False,
        "searched_operating_point_count": len(search),
        "passing_operating_point_count": len(passing),
        "safety_gate_passed": chosen is not None,
        "chosen_operating_point": chosen,
        "failure_action": None if chosen is not None else "stop_without_fold0_materialization_or_locked_fold_access",
    }
    final_model = fit_gate_model_v5(dataset, train_folds=(2, 3, 4))
    return GateOOFResultV5(predictions, models, final_model, receipt)


def seal_gate_oof_result_v5(
    dataset: GateDatasetV5, result: GateOOFResultV5, *,
    oof_path: Path, threshold_path: Path, policy_path: Path,
) -> tuple[Path, Path, Path]:
    validate_gate_dataset_v5(dataset, expected_purpose="fold234_oof_fit")
    n = len(dataset.candidate_scene_ids)
    oof_payload = {
        "schema": np.asarray(OOF_SCHEMA), "complete": np.asarray(True),
        "create_only": np.asarray(True), "scene_grouped": np.asarray(True),
        "threshold_source_folds": np.asarray(FIT_FOLDS, np.int64),
        "fold0_used": np.asarray(False), "fold1_or_validation_access": np.asarray(False),
        "scene_ids": dataset.candidate_scene_ids,
        "fold_ids": dataset.candidate_fold_ids,
        "candidate_rows": dataset.candidate_rows,
        "candidate_anchor_positions": dataset.candidate_anchor_positions,
        "candidate_iou_predictions": result.predictions.candidate_iou,
        "iou50_probability_predictions": result.predictions.iou50_probability,
        "same_gt_gain_predictions": result.predictions.same_gt_gain,
        "scoring_train_fold_json": result.predictions.scoring_train_fold_json,
        "dataset_source_collection_sha256": np.asarray(dataset.source_collection_sha256),
        "fold_model_json": np.asarray(json.dumps(
            {str(fold): _model_payload(model) for fold, model in result.fold_models.items()},
            separators=(",", ":"), sort_keys=True,
        )),
    }
    if any(np.asarray(oof_payload[name]).shape != (n,) for name in (
        "scene_ids", "fold_ids", "candidate_rows", "candidate_anchor_positions",
        "candidate_iou_predictions", "iou50_probability_predictions",
        "same_gt_gain_predictions", "scoring_train_fold_json",
    )):
        raise ValueError("v5 OOF output row alignment differs")
    oof = write_npz_create_only(oof_path, oof_payload, "v5 fold234 gate OOF predictions")
    threshold_payload = dict(result.threshold_receipt)
    threshold_payload.update({
        "create_only": True,
        "oof_predictions": {"path": str(oof), "sha256": sha256_file(oof), "schema": OOF_SCHEMA},
        "source_collection": {
            "path": str(dataset.source_collection_path), "sha256": dataset.source_collection_sha256,
            "schema": COLLECTION_SCHEMA,
        },
    })
    threshold = write_json_create_only(threshold_path, threshold_payload, "v5 OOF threshold receipt")
    chosen = result.threshold_receipt.get("chosen_operating_point")
    policy_payload = {
        "schema": POLICY_SCHEMA,
        "complete": True,
        "create_only": True,
        "exploratory_only": True,
        "deployable": False,
        "activation_authorized": False,
        "oof_safety_gate_passed": chosen is not None,
        "fit_folds": [2, 3, 4],
        "threshold_source": "fold234_scene_grouped_gate_oof_only",
        "fold0_used_for_fit_or_selection": False,
        "fold1_or_validation_access": False,
        "feature_names": list(FEATURE_NAMES),
        "final_model": _model_payload(result.final_model),
        "thresholds": None if chosen is None else chosen["thresholds"],
        "oof_predictions": {"path": str(oof), "sha256": sha256_file(oof), "schema": OOF_SCHEMA},
        "threshold_receipt": {
            "path": str(threshold), "sha256": sha256_file(threshold), "schema": THRESHOLD_SCHEMA,
        },
        "failure_action": result.threshold_receipt.get("failure_action"),
    }
    policy = write_json_create_only(policy_path, policy_payload, "v5 non-deployable exploratory policy")
    return oof, threshold, policy


def load_gate_policy_v5(path: Path, *, require_oof_pass: bool = True) -> tuple[Path, dict[str, Any], GateModelV5]:
    source, value = _json(path, "v5 exploratory policy")
    if (
        value.get("schema") != POLICY_SCHEMA
        or value.get("complete") is not True
        or value.get("create_only") is not True
        or value.get("exploratory_only") is not True
        or value.get("deployable") is not False
        or value.get("activation_authorized") is not False
        or value.get("fit_folds") != [2, 3, 4]
        or value.get("threshold_source") != "fold234_scene_grouped_gate_oof_only"
        or value.get("fold0_used_for_fit_or_selection") is not False
        or value.get("fold1_or_validation_access") is not False
        or value.get("feature_names") != list(FEATURE_NAMES)
    ):
        raise ValueError("v5 policy contract differs")
    threshold_path, threshold = _record(value.get("threshold_receipt"), "v5 threshold receipt", schema=THRESHOLD_SCHEMA)
    oof_path = _regular(Path(str((value.get("oof_predictions") or {}).get("path", ""))), "v5 OOF predictions")
    if (
        sha256_file(oof_path) != _sha((value.get("oof_predictions") or {}).get("sha256"), "OOF SHA256")
        or (threshold.get("oof_predictions") or {}).get("sha256") != sha256_file(oof_path)
        or value.get("oof_safety_gate_passed") is not threshold.get("safety_gate_passed")
        or value.get("thresholds") != (
            None if threshold.get("chosen_operating_point") is None
            else threshold["chosen_operating_point"]["thresholds"]
        )
    ):
        raise ValueError("v5 policy/OOF/threshold chain differs")
    if require_oof_pass and value.get("oof_safety_gate_passed") is not True:
        raise PermissionError("v5 fold234 OOF safety gate did not pass")
    if sha256_file(threshold_path) != value["threshold_receipt"]["sha256"]:
        raise ValueError("v5 threshold receipt changed")
    return source, value, _model_from_payload(value["final_model"])


def evaluate_fold0_reused_dev_v5(
    dataset: GateDatasetV5, *, policy_path: Path, output_path: Path
) -> Path:
    """Run one frozen fold-0 diagnostic; no threshold/model selection occurs."""

    validate_gate_dataset_v5(dataset, expected_purpose="fold0_reused_dev")
    policy_source, policy, model = load_gate_policy_v5(policy_path, require_oof_pass=True)
    if model.train_folds != (2, 3, 4):
        raise ValueError("fold0 diagnostic model must be fit on folds234")
    iou, probability, gain = predict_gate_model_v5(model, dataset.features)
    predictions = GatePredictionsV5(
        iou, probability, gain,
        _readonly(np.full(len(iou), "[2,3,4]")),
    )
    thresholds = policy.get("thresholds") or {}
    if set(thresholds) != {"candidate_iou", "same_gt_gain", "iou50_probability"}:
        raise ValueError("frozen v5 threshold tuple differs")
    selection = select_replacements_v5(
        dataset, predictions,
        iou_threshold=float(thresholds["candidate_iou"]),
        gain_threshold=float(thresholds["same_gt_gain"]),
        probability_threshold=float(thresholds["iou50_probability"]),
    )
    evaluation = _evaluate_selection_v5(dataset, selection)
    payload = {
        "schema": FOLD0_REPORT_SCHEMA,
        "complete": True,
        "create_only": True,
        "report_label": "noncanonical_reused_dev_exploratory_diagnostic",
        "fold_id": 0,
        "scene_count": 20,
        "gate_fit_folds": [2, 3, 4],
        "candidate_producer_role": "outer_dev",
        "threshold_source": "fold234_scene_grouped_gate_oof",
        "thresholds_frozen_before_fold0": True,
        "fold0_retuning": False,
        "fold0_model_selection": False,
        "policy_activation_authorized": False,
        "fold1_or_validation_authorized": False,
        "policy": {"path": str(policy_source), "sha256": sha256_file(policy_source), "schema": POLICY_SCHEMA},
        "source_collection": {
            "path": str(dataset.source_collection_path), "sha256": dataset.source_collection_sha256,
            "schema": COLLECTION_SCHEMA,
        },
        "evaluation": evaluation,
    }
    return write_json_create_only(output_path, payload, "v5 fold0 reused-dev report")


@dataclass(frozen=True)
class MaterializedGeometryV5:
    corners: np.ndarray
    scores: np.ndarray
    row_indices: np.ndarray


def materialize_geometry_only_v5(
    *, anchor_corners: Any, anchor_scores: Any, candidate_corners: Any,
    anchor_indices: Any, candidate_rows: Any,
) -> MaterializedGeometryV5:
    anchors = np.asarray(anchor_corners)
    scores = np.asarray(anchor_scores)
    candidates = np.asarray(candidate_corners)
    replace = np.asarray(anchor_indices)
    rows = np.asarray(candidate_rows)
    count = len(anchors)
    if (
        anchors.dtype != np.dtype(np.float32) or anchors.shape != (count, 8, 3)
        or scores.dtype != np.dtype(np.float32) or scores.shape != (count,)
        or candidates.dtype != np.dtype(np.float32) or candidates.ndim != 3
        or candidates.shape[1:] != (8, 3)
        or replace.dtype != np.dtype(np.int64) or rows.dtype != np.dtype(np.int64)
        or replace.shape != rows.shape or replace.ndim != 1
        or len(np.unique(replace)) != len(replace)
        or np.any((replace < 0) | (replace >= count))
        or np.any((rows < 0) | (rows >= len(candidates)))
        or not np.isfinite(anchors).all() or not np.isfinite(scores).all()
        or not np.isfinite(candidates).all()
    ):
        raise ValueError("v5 geometry-only materializer input differs")
    output = anchors.copy()
    output[replace] = candidates[rows]
    preserved_scores = scores.copy()
    row_indices = np.arange(count, dtype=np.int64)
    if (
        output.shape != anchors.shape
        or not np.array_equal(preserved_scores, scores)
        or not np.array_equal(row_indices, np.arange(count, dtype=np.int64))
    ):
        raise RuntimeError("v5 materializer changed score/order/count")
    return MaterializedGeometryV5(
        _readonly(output, np.float32), _readonly(preserved_scores, np.float32),
        _readonly(row_indices, np.int64),
    )


def write_materialized_geometry_v5(
    path: Path, *, scene_id: str, source_anchor_sha256: str,
    source_candidate_sha256: str, policy_sha256: str,
    result: MaterializedGeometryV5,
) -> Path:
    scene = _scene(scene_id)
    payload = {
        "schema": np.asarray(MATERIALIZATION_SCHEMA),
        "complete": np.asarray(True), "create_only": np.asarray(True),
        "geometry_only": np.asarray(True), "scores_preserved": np.asarray(True),
        "row_order_preserved": np.asarray(True), "row_count_preserved": np.asarray(True),
        "scene_id": np.asarray(scene),
        "source_anchor_sha256": np.asarray(_sha(source_anchor_sha256, "anchor SHA256")),
        "source_candidate_sha256": np.asarray(_sha(source_candidate_sha256, "candidate SHA256")),
        "policy_sha256": np.asarray(_sha(policy_sha256, "policy SHA256")),
        "corners": result.corners, "scores": result.scores, "row_indices": result.row_indices,
        "corners_sha256": np.asarray(sha256_array(result.corners)),
        "scores_sha256": np.asarray(sha256_array(result.scores)),
    }
    return write_npz_create_only(path, payload, "v5 geometry-only materialization")


def run_locked_fold1_once_v5(
    *,
    enabled: bool = False,
    authorization_path: Path | None = None,
    policy_path: Path | None = None,
    dataset_loader: Callable[[], GateDatasetV5] | None = None,
    report_path: Path | None = None,
    consumption_receipt_path: Path | None = None,
) -> tuple[Path, Path]:
    """Consume a separately sealed fold-1 authorization exactly once.

    The default branch raises before resolving or opening any supplied path and
    before invoking ``dataset_loader``.  Fold-1 results cannot tune thresholds,
    activate the exploratory policy, or authorize official validation.
    """

    if enabled is not True:
        raise LockedFoldDisabledError(
            "locked fold1 is disabled; no authorization, candidate, metadata, or GT was opened"
        )
    if any(value is None for value in (
        authorization_path, policy_path, dataset_loader, report_path, consumption_receipt_path
    )):
        raise ValueError("enabled locked fold1 requires all explicit one-time inputs")
    assert authorization_path is not None and policy_path is not None
    assert dataset_loader is not None and report_path is not None and consumption_receipt_path is not None
    if report_path.exists() or report_path.is_symlink() or consumption_receipt_path.exists() or consumption_receipt_path.is_symlink():
        raise FileExistsError("locked fold1 authorization has already been consumed or partially consumed")
    authorization_source, authorization = _json(authorization_path, "locked fold1 authorization")
    policy_source, policy, model = load_gate_policy_v5(policy_path, require_oof_pass=True)
    if (
        authorization.get("schema") != LOCKED_AUTH_SCHEMA
        or authorization.get("complete") is not True
        or authorization.get("create_only") is not True
        or authorization.get("enabled") is not True
        or authorization.get("one_time") is not True
        or authorization.get("fold_id") != 1
        or authorization.get("scene_count") != 20
        or authorization.get("thresholds_frozen") is not True
        or authorization.get("fold0_result_used_for_selection") is not False
        or authorization.get("official_validation_access") is not False
        or authorization.get("policy_activation_authorized") is not False
        or (authorization.get("policy") or {}).get("sha256") != sha256_file(policy_source)
        or (authorization.get("policy") or {}).get("schema") != POLICY_SCHEMA
        or policy.get("oof_safety_gate_passed") is not True
        or model.train_folds != (2, 3, 4)
    ):
        raise PermissionError("locked fold1 authorization contract differs")
    # The first fold-1 data access occurs only after every authorization/output
    # guard above has passed.
    dataset = dataset_loader()
    validate_gate_dataset_v5(dataset, expected_purpose="fold1_locked_once")
    iou, probability, gain = predict_gate_model_v5(model, dataset.features)
    predictions = GatePredictionsV5(iou, probability, gain, _readonly(np.full(len(iou), "[2,3,4]")))
    thresholds = policy["thresholds"]
    selection = select_replacements_v5(
        dataset, predictions, iou_threshold=thresholds["candidate_iou"],
        gain_threshold=thresholds["same_gt_gain"],
        probability_threshold=thresholds["iou50_probability"],
    )
    report_payload = {
        "schema": "boxfusion.ca1m_tr3d_exploratory_gate_fold1_report.v5",
        "complete": True, "create_only": True, "fold_id": 1, "scene_count": 20,
        "one_time_locked_internal_check": True, "threshold_retuning": False,
        "model_selection": False, "policy_activation_authorized": False,
        "official_validation_authorized": False,
        "authorization": {"path": str(authorization_source), "sha256": sha256_file(authorization_source)},
        "policy": {"path": str(policy_source), "sha256": sha256_file(policy_source)},
        "evaluation": _evaluate_selection_v5(dataset, selection),
    }
    report = write_json_create_only(report_path, report_payload, "v5 locked fold1 report")
    receipt_payload = {
        "schema": LOCKED_RECEIPT_SCHEMA,
        "complete": True, "create_only": True, "consumed_once": True,
        "fold_id": 1, "scene_count": 20, "threshold_retuning": False,
        "policy_activation_authorized": False, "official_validation_authorized": False,
        "authorization": {"path": str(authorization_source), "sha256": sha256_file(authorization_source)},
        "policy": {"path": str(policy_source), "sha256": sha256_file(policy_source)},
        "report": {"path": str(report), "sha256": sha256_file(report)},
    }
    receipt = write_json_create_only(
        consumption_receipt_path, receipt_payload, "v5 locked fold1 consumption receipt"
    )
    return report, receipt


def pending_runtime_preflight_v5(config_path: Path) -> dict[str, Any]:
    """Validate only the pending JSON contract; never open future inputs."""

    from .ca1m_tr3d_exploratory_gate_v5 import static_report

    report = static_report(config_path)
    return {
        **report,
        "runtime_surface_implemented": True,
        "runtime_ready": False,
        "failure_action": "stop_before_opening_candidate_gt_fold1_validation_or_output",
    }


__all__ = [
    "ANCHOR_SCORE_SOURCE", "COLLECTION_SCHEMA", "CandidateCollectionV5",
    "CandidateEvidenceV5", "DATASET_SCHEMA", "EVIDENCE_SCHEMA", "FEATURE_NAMES",
    "FOLD0_REPORT_SCHEMA", "FIT_FOLDS", "GateDatasetV5", "GateModelV5",
    "GateOOFResultV5", "GatePredictionsV5", "LOCKED_AUTH_SCHEMA",
    "LockedFoldDisabledError", "MATERIALIZATION_SCHEMA", "NAMESPACE",
    "OOF_SCHEMA", "POLICY_SCHEMA", "RAW_SCORE_FEATURE_INDICES", "ROLE_COLLECTION_SCHEMA",
    "ROLE_RECEIPT_SCHEMA", "ROLE_SPECS", "SelectionV5", "THRESHOLD_SCHEMA",
    "V5ProtocolError", "build_labeled_dataset_v5", "evaluate_fold0_reused_dev_v5",
    "fit_gate_model_v5", "load_candidate_collection_v5", "load_candidate_evidence_v5",
    "load_detector_role_receipt_v5", "load_gate_policy_v5", "materialize_geometry_only_v5",
    "pending_runtime_preflight_v5", "predict_gate_model_v5", "run_locked_fold1_once_v5",
    "seal_candidate_collection_v5", "seal_detector_role_receipt_v5",
    "seal_gate_dataset_v5", "seal_gate_oof_result_v5", "seal_role_candidate_collection_v5",
    "select_replacements_v5", "sha256_file", "train_gate_oof_v5",
    "write_candidate_evidence_v5", "write_json_create_only",
    "write_materialized_geometry_v5", "write_npz_create_only",
]
