#!/usr/bin/env python3
"""Materialize and audit geometry-only CA terminal-v4 train100 OOF outputs.

This is an active geometry replay on the 80 fit/dev train scenes, but it is not
canonical103 activation authority.  Scores, row order, row count, and class
labels are copied from the B6-v2 OOF anchor rows byte-for-value.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import pickle
import sys
import tempfile
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_terminal_gate_v4 import (  # noqa: E402
    DATASET_SCHEMA,
    FEATURE_NAMES,
    MATERIALIZATION_SCHEMA,
    PREREGISTRATION_SCHEMA,
    TerminalGateFeatureBatchV4,
    load_gate_policy_v4,
    materialize_geometry_only,
    select_terminal_replacements_v4,
    validate_preregistration_record,
    write_binding_create_only,
)
from boxfusion.ca1m_tr3d_terminal_v4 import sha256_file  # noqa: E402
from tools.build_ca1m_tr3d_benefit_dataset_v4 import (  # noqa: E402
    MANIFEST_SCHEMA as DATASET_MANIFEST_SCHEMA,
    SCORE_SOURCE,
)
from tools.train_ca1m_tr3d_benefit_gate_v4 import (  # noqa: E402
    REPORT_SCHEMA as TRAINING_REPORT_SCHEMA,
)


AUDIT_SCHEMA = "boxfusion.ca1m_tr3d_terminal_materialization_audit.v4"


def _regular(path: Path, name: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink: {path}")
    result = path.resolve()
    if (
        not result.is_file()
        or result.is_symlink()
        or result.stat().st_size <= 0
        or result.stat().st_mode & 0o222
    ):
        raise ValueError(f"{name} must be a sealed regular file: {result}")
    return result


def _json(path: Path, name: str) -> tuple[Path, dict[str, Any]]:
    source = _regular(path, name)
    try:
        payload = json.loads(source.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain an object")
    return source, payload


def _scalar(archive: Any, name: str) -> Any:
    value = np.asarray(archive[name])
    if value.shape != ():
        raise ValueError(f"dataset {name} must be scalar")
    return value.item()


def _load_dataset(path: Path, manifest_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    dataset = _regular(path, "terminal gate v4 dataset")
    manifest_source, manifest = _json(manifest_path, "terminal gate v4 dataset manifest")
    if (
        manifest.get("schema") != DATASET_MANIFEST_SCHEMA
        or manifest.get("complete") is not True
        or manifest.get("train_only") is not True
        or manifest.get("scene_count") != 80
        or manifest.get("locked_internal_fold1_gt_access") is not False
        or manifest.get("validation_ground_truth_access") is not False
        or manifest.get("anchor_score_source") != SCORE_SOURCE
        or manifest.get("deploy_b6_scores_used_for_stacked_training") is not False
        or (manifest.get("dataset") or {}).get("path") != str(dataset)
        or (manifest.get("dataset") or {}).get("sha256") != sha256_file(dataset)
    ):
        raise ValueError("terminal gate v4 materialization dataset manifest differs")
    preregistration_record = manifest.get("preregistration_manifest") or {}
    preregistration_path, preregistration = validate_preregistration_record(
        preregistration_record
    )
    preregistration_sha256 = sha256_file(preregistration_path)
    if (
        preregistration_record.get("schema") != PREREGISTRATION_SCHEMA
        or preregistration_record.get("sha256") != preregistration_sha256
        or preregistration_record.get("sealed_before_first_gt_join") is not True
        or manifest.get("source_code_sha256")
        != (preregistration.get("code") or {}).get("dataset_builder", {}).get("sha256")
    ):
        raise ValueError("materialization dataset lacks preregistration reverse binding")
    with np.load(dataset, allow_pickle=False) as archive:
        if (
            _scalar(archive, "schema") != DATASET_SCHEMA
            or _scalar(archive, "complete") is not True
            or _scalar(archive, "train_only") is not True
            or _scalar(archive, "locked_internal_fold1_gt_access") is not False
            or _scalar(archive, "anchor_score_source") != SCORE_SOURCE
            or _scalar(archive, "deploy_b6_scores_used_for_stacked_training") is not False
            or _scalar(archive, "preregistration_manifest_sha256")
            != preregistration_sha256
            or tuple(str(value) for value in archive["feature_names"].tolist()) != FEATURE_NAMES
        ):
            raise ValueError("terminal gate v4 materialization dataset fields differ")
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    if 1 in set(np.asarray(arrays["scene_fold_ids"], np.int64).tolist()):
        raise ValueError("terminal gate v4 materialization dataset exposes fold1")
    if manifest_source.stat().st_mode & 0o222:
        raise ValueError("terminal gate v4 dataset manifest must be sealed")
    return arrays, manifest


def _write_prediction(path: Path, corners: np.ndarray, scores: np.ndarray) -> str:
    payload = [[
        (0, np.array(corner, dtype=np.float32, order="C", copy=True), float(score))
        for corner, score in zip(corners, scores)
    ]]
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        path.chmod(0o444)
        published = True
    except BaseException:
        if published:
            path.unlink(missing_ok=True)
        raise
    finally:
        Path(temporary).unlink(missing_ok=True)
    return sha256_file(path)


def _prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    source = _regular(path, "terminal gate v4 prediction")
    with source.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - local sealed artifact
    if type(payload) is not list or len(payload) != 1 or type(payload[0]) is not list:
        raise ValueError(f"non-canonical prediction payload: {source}")
    corners: list[np.ndarray] = []
    scores: list[float] = []
    for index, row in enumerate(payload[0]):
        if type(row) is not tuple or len(row) != 3 or type(row[0]) is not int or row[0] != 0:
            raise ValueError(f"non-canonical prediction row {index}: {source}")
        geometry = np.asarray(row[1])
        score = row[2]
        if (
            type(row[1]) is not np.ndarray
            or geometry.dtype != np.dtype(np.float32)
            or geometry.shape != (8, 3)
            or not geometry.flags.c_contiguous
            or not np.isfinite(geometry).all()
            or type(score) is not float
            or not math.isfinite(score)
        ):
            raise ValueError(f"non-canonical prediction value {index}: {source}")
        corners.append(geometry)
        scores.append(score)
    return (
        np.asarray(corners, dtype=np.float32).reshape((-1, 8, 3)),
        np.asarray(scores, dtype=np.float32),
    )


def _scene_arrays(arrays: dict[str, np.ndarray], scene: str) -> dict[str, np.ndarray]:
    candidate_mask = np.asarray(arrays["scene_ids"]).astype(str) == scene
    baseline_mask = np.asarray(arrays["baseline_scene_ids"]).astype(str) == scene
    return {
        "features": np.asarray(arrays["features"], np.float32)[candidate_mask],
        "candidate_rows": np.asarray(arrays["candidate_rows"], np.int64)[candidate_mask],
        "anchor_indices": np.asarray(arrays["anchor_indices"], np.int64)[candidate_mask],
        "candidate_scores": np.asarray(arrays["candidate_scores"], np.float32)[candidate_mask],
        "candidate_corners": np.asarray(arrays["candidate_corners"], np.float32)[candidate_mask],
        "anchor_corners": np.asarray(arrays["baseline_corners"], np.float32)[baseline_mask],
        "anchor_scores": np.asarray(arrays["baseline_scores"], np.float32)[baseline_mask],
        "anchor_rows": np.asarray(arrays["baseline_row_indices"], np.int64)[baseline_mask],
    }


def _selected_local_rows(
    selection_candidate_rows: np.ndarray,
    selection_anchor_indices: np.ndarray,
    scene: dict[str, np.ndarray],
) -> np.ndarray:
    lookup = {
        (int(anchor), int(candidate)): index
        for index, (anchor, candidate) in enumerate(zip(
            scene["anchor_indices"].tolist(), scene["candidate_rows"].tolist()
        ))
    }
    rows = [
        lookup[(int(anchor), int(candidate))]
        for anchor, candidate in zip(
            selection_anchor_indices.tolist(), selection_candidate_rows.tolist()
        )
    ]
    return np.asarray(rows, np.int64)


def _validate_training_report(
    path: Path,
    *,
    policy_path: Path,
    dataset_path: Path,
    dataset_manifest_path: Path,
    binding_path: Path,
    preregistration_record: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    report_path, report = _json(path, "terminal gate v4 training report")
    checked_policy_path, policy_payload = _json(policy_path, "terminal gate v4 policy")
    chosen = report.get("chosen_operating_point") or {}
    preregistration_path, preregistration = validate_preregistration_record(
        preregistration_record
    )
    if (
        report.get("schema") != TRAINING_REPORT_SCHEMA
        or report.get("complete") is not True
        or report.get("train_only") is not True
        or report.get("threshold_dev_gate_passed") is not True
        or report.get("failure_action") is not None
        or report.get("locked_internal_fold1_accessed") is not False
        or report.get("validation_ground_truth_access") is not False
        or report.get("validation_prediction_access") is not False
        or report.get("formal_canonical103_authorized") is not False
        or report.get("eligible_operating_point_count", 0) < 1
        or (chosen.get("gate") or {}).get("pass") is not True
        or (report.get("policy") or {}).get("path") != str(checked_policy_path)
        or (report.get("policy") or {}).get("sha256") != sha256_file(checked_policy_path)
        or (report.get("dataset") or {}).get("path") != str(dataset_path)
        or (report.get("dataset") or {}).get("sha256") != sha256_file(dataset_path)
        or report.get("dataset_manifest_sha256") != sha256_file(dataset_manifest_path)
        or report.get("training_binding_sha256") != sha256_file(binding_path)
        or (report.get("preregistration_manifest") or {}).get("path")
        != str(preregistration_path)
        or (report.get("preregistration_manifest") or {}).get("sha256")
        != sha256_file(preregistration_path)
        or report.get("source_code_sha256")
        != (preregistration.get("code") or {}).get("trainer", {}).get("sha256")
        or policy_payload.get("training_binding_sha256") != sha256_file(binding_path)
        or policy_payload.get("preregistration_manifest_sha256")
        != sha256_file(preregistration_path)
        or policy_payload.get("dataset_sha256") != sha256_file(dataset_path)
        or policy_payload.get("dataset_manifest_sha256")
        != sha256_file(dataset_manifest_path)
        or policy_payload.get("source_code_sha256")
        != (preregistration.get("code") or {}).get("trainer", {}).get("sha256")
        or float((policy_payload.get("quality25") or {}).get("threshold", -1.0))
        != float(chosen.get("quality_threshold", -2.0))
        or float((policy_payload.get("benefit05") or {}).get("threshold", -1.0))
        != float(chosen.get("benefit_threshold", -2.0))
    ):
        raise ValueError("terminal gate v4 training report/policy chain differs")
    return report_path, report


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    arrays, dataset_manifest = _load_dataset(args.dataset, args.dataset_manifest)
    dataset_path = _regular(args.dataset, "terminal gate v4 dataset")
    dataset_manifest_path = _regular(
        args.dataset_manifest, "terminal gate v4 dataset manifest"
    )
    binding_record = dataset_manifest["training_binding"]
    binding_path = _regular(Path(binding_record["path"]), "terminal gate v4 training binding")
    if binding_record["sha256"] != sha256_file(binding_path):
        raise ValueError("terminal gate v4 dataset training binding changed")
    policy_path = _regular(args.policy, "terminal gate v4 policy")
    policy = load_gate_policy_v4(
        policy_path,
        expected_training_binding_sha256=sha256_file(binding_path),
        require_dev_pass=True,
    )
    if policy.dataset_sha256 != sha256_file(_regular(args.dataset, "dataset")):
        raise ValueError("terminal gate v4 policy dataset differs")
    preregistration_record = dataset_manifest["preregistration_manifest"]
    if policy.preregistration_manifest_sha256 != preregistration_record["sha256"]:
        raise ValueError("terminal gate v4 policy preregistration differs")
    training_report_path, _ = _validate_training_report(
        args.training_report,
        policy_path=policy_path,
        dataset_path=dataset_path,
        dataset_manifest_path=dataset_manifest_path,
        binding_path=binding_path,
        preregistration_record=preregistration_record,
    )
    output_root = args.output_root.resolve()
    manifest_output = args.manifest.resolve()
    if output_root.exists() or output_root.is_symlink() or manifest_output.exists():
        raise FileExistsError("refusing existing terminal gate v4 materialization output")
    output_root.mkdir(parents=True)
    scenes = tuple(str(value) for value in np.asarray(arrays["scene_table"]).tolist())
    if len(scenes) != 80 or len(set(scenes)) != 80:
        raise ValueError("terminal gate v4 materialization scene table differs")
    reports: dict[str, Any] = {}
    total_replacements = 0
    for scene_id in scenes:
        scene = _scene_arrays(arrays, scene_id)
        if not np.array_equal(scene["anchor_rows"], np.arange(len(scene["anchor_rows"]), dtype=np.int64)):
            raise ValueError(f"terminal gate v4 anchor rows are not canonical: {scene_id}")
        batch = TerminalGateFeatureBatchV4(
            schema="boxfusion.ca1m_tr3d_terminal_gate_features.v4",
            scene_id=scene_id,
            score_source=SCORE_SOURCE,
            candidate_rows=scene["candidate_rows"],
            anchor_indices=scene["anchor_indices"],
            candidate_scores=scene["candidate_scores"],
            features=scene["features"],
        )
        selection = select_terminal_replacements_v4(batch, policy)
        local_rows = _selected_local_rows(
            selection.candidate_rows, selection.anchor_indices, scene
        )
        result = materialize_geometry_only(
            anchor_corners=scene["anchor_corners"],
            anchor_scores=scene["anchor_scores"],
            candidate_corners=scene["candidate_corners"],
            anchor_indices=selection.anchor_indices,
            candidate_rows=local_rows,
        )
        target = output_root / f"{scene_id}_boxes.pkl"
        output_sha = _write_prediction(target, result.corners, result.scores)
        loaded_corners, loaded_scores = _prediction(target)
        expected_corners = np.array(scene["anchor_corners"], copy=True)
        expected_corners[selection.anchor_indices] = scene["candidate_corners"][local_rows]
        changed = np.flatnonzero(
            np.any(loaded_corners != scene["anchor_corners"], axis=(1, 2))
        )
        if (
            not np.array_equal(loaded_scores, scene["anchor_scores"])
            or len(loaded_corners) != len(scene["anchor_corners"])
            or not np.array_equal(loaded_corners, expected_corners)
            or not set(changed.tolist()).issubset(set(selection.anchor_indices.tolist()))
        ):
            raise RuntimeError(f"terminal gate v4 geometry-only audit failed: {scene_id}")
        reports[scene_id] = {
            "output_path": str(target),
            "output_sha256": output_sha,
            "anchor_rows": len(scene["anchor_corners"]),
            "output_rows": len(loaded_corners),
            "replacement_count": len(selection.anchor_indices),
            "replaced_anchor_indices": selection.anchor_indices.tolist(),
            "source_candidate_rows": selection.candidate_rows.tolist(),
            "source_dataset_local_rows": local_rows.tolist(),
            "actual_changed_anchor_indices": changed.tolist(),
            "scores_sha256": hashlib_sha(scene["anchor_scores"]),
            "geometry_only_verified": True,
        }
        total_replacements += len(selection.anchor_indices)
    payload = {
        "schema": MATERIALIZATION_SCHEMA,
        "complete": True,
        "train_only": True,
        "training_oof_materialization": True,
        "active_geometry_replay": True,
        "formal_canonical103_authorized": False,
        "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "locked_internal_fold1_accessed": False,
        "official_validation_comparable": False,
        "geometry_only": True,
        "preserve_anchor_scores": True,
        "preserve_row_order": True,
        "preserve_row_count": True,
        "clip_semantics_unchanged": True,
        "anchor_score_source": SCORE_SOURCE,
        "deploy_b6_scores_used_for_stacked_training": False,
        "scene_count": 80,
        "replacement_count": total_replacements,
        "dataset": {"path": str(_regular(args.dataset, "dataset")), "sha256": sha256_file(_regular(args.dataset, "dataset"))},
        "dataset_manifest_sha256": sha256_file(_regular(args.dataset_manifest, "dataset manifest")),
        "training_binding_sha256": sha256_file(binding_path),
        "preregistration_manifest": dict(preregistration_record),
        "policy": {"path": str(policy_path), "sha256": sha256_file(policy_path)},
        "training_report": {
            "path": str(training_report_path),
            "sha256": sha256_file(training_report_path),
            "schema": TRAINING_REPORT_SCHEMA,
        },
        "output_root": str(output_root),
        "scenes": reports,
    }
    write_binding_create_only(manifest_output, payload)
    return payload


def hashlib_sha(value: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def audit(args: argparse.Namespace) -> dict[str, Any]:
    arrays, dataset_manifest = _load_dataset(args.dataset, args.dataset_manifest)
    manifest_path, manifest = _json(args.manifest, "terminal gate v4 materialization manifest")
    output_root = Path(str(manifest.get("output_root", ""))).resolve()
    scenes = tuple(str(value) for value in np.asarray(arrays["scene_table"]).tolist())
    if (
        manifest.get("schema") != MATERIALIZATION_SCHEMA
        or manifest.get("complete") is not True
        or manifest.get("train_only") is not True
        or manifest.get("training_oof_materialization") is not True
        or manifest.get("formal_canonical103_authorized") is not False
        or manifest.get("geometry_only") is not True
        or manifest.get("preserve_anchor_scores") is not True
        or manifest.get("preserve_row_order") is not True
        or manifest.get("preserve_row_count") is not True
        or manifest.get("locked_internal_fold1_accessed") is not False
        or manifest.get("scene_count") != 80
        or set((manifest.get("scenes") or {})) != set(scenes)
    ):
        raise ValueError("terminal gate v4 materialization manifest differs")
    dataset_path = _regular(args.dataset, "terminal gate v4 dataset")
    dataset_manifest_path = _regular(
        args.dataset_manifest, "terminal gate v4 dataset manifest"
    )
    binding_record = dataset_manifest["training_binding"]
    binding_path = _regular(
        Path(binding_record["path"]), "terminal gate v4 training binding"
    )
    if binding_record["sha256"] != sha256_file(binding_path):
        raise ValueError("terminal gate v4 audit binding differs")
    policy_record = manifest.get("policy") or {}
    policy_path = _regular(Path(str(policy_record.get("path", ""))), "terminal gate v4 policy")
    if policy_record.get("sha256") != sha256_file(policy_path):
        raise ValueError("terminal gate v4 audit policy differs")
    policy = load_gate_policy_v4(
        policy_path,
        expected_training_binding_sha256=sha256_file(binding_path),
        require_dev_pass=True,
    )
    preregistration_record = dataset_manifest["preregistration_manifest"]
    if policy.preregistration_manifest_sha256 != preregistration_record["sha256"]:
        raise ValueError("terminal gate v4 audit policy preregistration differs")
    training_report_record = manifest.get("training_report") or {}
    training_report_path = _regular(
        Path(str(training_report_record.get("path", ""))),
        "terminal gate v4 training report",
    )
    if (
        training_report_record.get("schema") != TRAINING_REPORT_SCHEMA
        or training_report_record.get("sha256") != sha256_file(training_report_path)
    ):
        raise ValueError("terminal gate v4 audit training report differs")
    _validate_training_report(
        training_report_path,
        policy_path=policy_path,
        dataset_path=dataset_path,
        dataset_manifest_path=dataset_manifest_path,
        binding_path=binding_path,
        preregistration_record=preregistration_record,
    )
    actual = {
        item.name for item in output_root.iterdir()
        if item.is_file() and not item.is_symlink()
    }
    if actual != {f"{scene}_boxes.pkl" for scene in scenes}:
        raise ValueError("terminal gate v4 materialization root is not exact80")
    total = 0
    for scene_id in scenes:
        scene = _scene_arrays(arrays, scene_id)
        path = output_root / f"{scene_id}_boxes.pkl"
        corners, scores = _prediction(path)
        row = manifest["scenes"][scene_id]
        anchors = np.asarray(row["replaced_anchor_indices"], np.int64)
        local = np.asarray(row["source_dataset_local_rows"], np.int64)
        expected_corners = np.array(scene["anchor_corners"], copy=True)
        expected_corners[anchors] = scene["candidate_corners"][local]
        changed = np.flatnonzero(
            np.any(corners != scene["anchor_corners"], axis=(1, 2))
        )
        if (
            row.get("output_sha256") != sha256_file(path)
            or row.get("geometry_only_verified") is not True
            or not np.array_equal(scores, scene["anchor_scores"])
            or len(corners) != len(scene["anchor_corners"])
            or not np.array_equal(corners, expected_corners)
            or changed.tolist() != row.get("actual_changed_anchor_indices")
            or not set(changed.tolist()).issubset(set(anchors.tolist()))
        ):
            raise ValueError(f"terminal gate v4 materialization audit differs: {scene_id}")
        total += len(anchors)
    if total != manifest.get("replacement_count"):
        raise ValueError("terminal gate v4 replacement total differs")
    report = {
        "schema": AUDIT_SCHEMA,
        "ok": True,
        "complete": True,
        "train_only": True,
        "scene_count": 80,
        "replacement_count": total,
        "geometry_only": True,
        "scores_preserved": True,
        "row_order_preserved": True,
        "row_count_preserved": True,
        "locked_internal_fold1_accessed": False,
        "validation_ground_truth_access": False,
        "formal_canonical103_authorized": False,
        "materialization_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
    }
    if args.audit_output is not None:
        write_binding_create_only(args.audit_output, report)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    mode = value.add_mutually_exclusive_group(required=True)
    mode.add_argument("--materialize", action="store_true")
    mode.add_argument("--audit", action="store_true")
    value.add_argument("--dataset", type=Path, required=True)
    value.add_argument("--dataset-manifest", type=Path, required=True)
    value.add_argument("--policy", type=Path)
    value.add_argument("--training-report", type=Path)
    value.add_argument("--output-root", type=Path)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--audit-output", type=Path)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.materialize:
        if (
            args.policy is None
            or args.training_report is None
            or args.output_root is None
            or args.audit_output is not None
        ):
            raise ValueError(
                "--materialize requires --policy/--training-report/--output-root "
                "and forbids --audit-output"
            )
        report = materialize(args)
    else:
        if (
            args.policy is not None
            or args.training_report is not None
            or args.output_root is not None
        ):
            raise ValueError(
                "--audit reads output_root/policy/training-report only through the sealed manifest"
            )
        report = audit(args)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
