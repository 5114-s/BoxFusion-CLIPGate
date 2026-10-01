#!/usr/bin/env python3
"""Build an immutable train-only CA-1M terminal-TR3D benefit dataset.

The candidate collection is GT-free and sealed before this program runs.  This
program is the only join between those caches and the derived CA-1M *training*
GT.  ``fit_dev`` exposes folds 2/3/4 for fitting and fold 0 for threshold
calibration.  ``locked_internal_check`` exposes only fold 1 and is intentionally
run only after the calibration report has been sealed.

No CA-1M validation prediction or validation GT path is accepted by the CLI.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_native_b6_observer import (  # noqa: E402
    FEATURE_NAMES as NATIVE_FEATURE_NAMES,
    SCHEMA as NATIVE_OBSERVER_SCHEMA,
)
from boxfusion.ca1m_tr3d_terminal import (  # noqa: E402
    pairwise_world_aabb_iou,
    sha256_file,
    world_aabb,
)
from boxfusion.ca1m_tr3d_terminal_gate import (  # noqa: E402
    BENEFIT_TARGET,
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    QUALITY_TARGET,
    build_terminal_gate_features,
)


SCHEMA = "boxfusion.ca1m_tr3d_benefit_dataset.v1"
MANIFEST_SCHEMA = "boxfusion.ca1m_tr3d_benefit_dataset_manifest.v1"
SPLIT_SCHEMA = "boxfusion.ca1m_tr3d_benefit_split.v1"
TERMINAL_AUDIT_SCHEMA = "boxfusion.ca1m_tr3d_terminal_observer_audit.v1"
CANDIDATE_AUDIT_SCHEMA = "boxfusion.ca1m_tr3d_candidate_evidence_audit.v1"
SOURCE_DATASET_SCHEMA = "boxfusion.ca1m_native_b6_quality_dataset.v1"
SOURCE_MANIFEST_SCHEMA = "boxfusion.ca1m_native_b6_quality_dataset_manifest.v1"
PARTITIONS = {
    "fit_dev": {
        "fold_ids": (0, 2, 3, 4),
        "roles": ("weights_train", "threshold_dev"),
    },
    "locked_internal_check": {
        "fold_ids": (1,),
        "roles": ("locked_internal_check",),
    },
}
ROLE_FOLDS = {
    "weights_train": (2, 3, 4),
    "threshold_dev": (0,),
    "locked_internal_check": (1,),
}
_SCENE_RE = re.compile(r"^[0-9]{8}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_ROLE_COUNTS = {
    "weights_train": 60,
    "threshold_dev": 20,
    "locked_internal_check": 20,
}
EXPECTED_VALIDATION_COUNT = 107
LOCKED_RECEIPT_PATH = (
    ROOT
    / "reports/ca1m_tr3d_benefit_gate_v1/locked_internal_access_receipt.json"
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--partition", choices=tuple(PARTITIONS), required=True)
    value.add_argument("--split-manifest", type=Path, required=True)
    value.add_argument("--source-dataset", type=Path, required=True)
    value.add_argument("--source-dataset-manifest", type=Path, required=True)
    value.add_argument("--official-val-list", type=Path, required=True)
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--terminal-root", type=Path, required=True)
    value.add_argument("--terminal-audit", type=Path, required=True)
    value.add_argument("--anchor-native-root", type=Path, required=True)
    value.add_argument("--candidate-native-root", type=Path, required=True)
    value.add_argument("--candidate-audit", type=Path, required=True)
    value.add_argument("--output-dataset", type=Path, required=True)
    value.add_argument("--output-manifest", type=Path, required=True)
    value.add_argument("--calibration-model", type=Path)
    value.add_argument("--calibration-report", type=Path)
    value.add_argument("--locked-access-receipt", type=Path)
    return value


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    result = path.resolve()
    if not result.is_file() or result.is_symlink() or result.stat().st_size <= 0:
        raise FileNotFoundError(f"missing regular {label}: {result}")
    return result


def _sealed_regular(path: Path, label: str) -> Path:
    result = _regular(path, label)
    if result.stat().st_mode & 0o222:
        raise ValueError(f"{label} must be sealed read-only: {result}")
    return result


def _directory(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    result = path.resolve()
    if not result.is_dir() or result.is_symlink():
        raise FileNotFoundError(f"missing directory {label}: {result}")
    return result


def _json(path: Path, label: str) -> tuple[Path, Mapping[str, Any]]:
    source = _regular(path, label)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON {label}: {source}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return source, payload


def _scene_rows(path: Path, label: str) -> tuple[str, ...]:
    source = _regular(path, label)
    rows = tuple(row.strip() for row in source.read_text().splitlines() if row.strip())
    if (
        not rows
        or len(rows) != len(set(rows))
        or tuple(sorted(rows)) != rows
        or any(_SCENE_RE.fullmatch(row) is None for row in rows)
    ):
        raise ValueError(f"invalid sorted unique CA-1M scene list: {source}")
    return rows


def _validation_ids(path: Path) -> tuple[str, ...]:
    source = _regular(path, "frozen official CA-1M full107 scene list")
    result = [row.strip() for row in source.read_text().splitlines() if row.strip()]
    if (
        len(result) != EXPECTED_VALIDATION_COUNT
        or len(result) != len(set(result))
        or result != sorted(result)
        or any(_SCENE_RE.fullmatch(row) is None for row in result)
    ):
        raise ValueError("official CA-1M validation list must be exact sorted full107 IDs")
    return tuple(result)


def _scalar(archive: Any, key: str) -> Any:
    if key not in archive.files:
        raise ValueError(f"archive is missing scalar {key}")
    value = np.asarray(archive[key])
    if value.shape != ():
        raise ValueError(f"archive field {key} must be scalar")
    return value.item()


def _native_evidence(
    path: Path,
    *,
    scene: str,
    expected_corners: np.ndarray,
    expected_scores: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    source = _regular(path, f"native evidence for {scene}")
    with np.load(source, allow_pickle=False) as archive:
        if (
            str(_scalar(archive, "schema")) != NATIVE_OBSERVER_SCHEMA
            or bool(_scalar(archive, "complete")) is not True
            or bool(_scalar(archive, "observer_only")) is not True
            or bool(_scalar(archive, "mutation_enabled")) is not False
            or bool(_scalar(archive, "ground_truth_access")) is not False
            or str(_scalar(archive, "scene_id")) != scene
        ):
            raise ValueError(f"native evidence contract mismatch: {scene}")
        corners = np.array(archive["corners"], copy=True)
        scores = np.array(archive["scores"], copy=True)
        indices = np.array(archive["result_indices"], copy=True)
        names = tuple(np.asarray(archive["feature_names"]).astype(str).tolist())
        features = np.array(archive["features"], copy=True)
        valid = np.array(archive["valid_evidence"], copy=True)
    rows = len(expected_corners)
    if (
        corners.dtype != np.float32
        or corners.shape != (rows, 8, 3)
        or scores.dtype != np.float32
        or scores.shape != (rows,)
        or indices.dtype != np.int64
        or not np.array_equal(indices, np.arange(rows, dtype=np.int64))
        or names != NATIVE_FEATURE_NAMES
        or features.dtype != np.float32
        or features.shape != (rows, len(NATIVE_FEATURE_NAMES))
        or valid.dtype != np.bool_
        or valid.shape != (rows,)
        or not np.isfinite(features).all()
        or np.any(features < 0.0)
        or np.any(features > 1.0)
        or not np.array_equal(features[:, 0].astype(np.float32), scores)
        or not np.array_equal(corners, expected_corners)
    ):
        raise ValueError(f"native evidence row mapping mismatch: {scene}")
    if expected_scores is not None and not np.array_equal(scores, expected_scores):
        raise ValueError(f"native evidence scores differ from candidate cache: {scene}")
    return features, valid, sha256_file(source)


def _target(corners: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not len(corners):
        return (
            np.empty((0,), np.float64),
            np.empty((0,), np.int64),
            np.empty((0, 0), np.float64),
        )
    if not len(gt):
        return (
            np.zeros(len(corners), np.float64),
            np.full(len(corners), -1, np.int64),
            np.empty((len(corners), 0), np.float64),
        )
    matrix = pairwise_world_aabb_iou(corners, gt)
    matched = np.argmax(matrix, axis=1).astype(np.int64)
    return matrix[np.arange(len(corners)), matched], matched, matrix


def _write_npz_create_only(path: Path, arrays: Mapping[str, Any]) -> Path:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, target)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite dataset: {target}") from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return target


def _write_json_create_only(path: Path, payload: Mapping[str, Any]) -> Path:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{target.name}.", suffix=".tmp",
            dir=target.parent, delete=False
        ) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, target)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite manifest: {target}") from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return target


def _code_manifest() -> dict[str, str]:
    sources = (
        Path(__file__).resolve(),
        ROOT / "boxfusion/ca1m_tr3d_terminal_gate.py",
        ROOT / "boxfusion/ca1m_tr3d_terminal.py",
        ROOT / "boxfusion/ca1m_native_b6_observer.py",
        ROOT / "tools/run_ca1m_tr3d_candidate_evidence.py",
        ROOT / "tools/audit_ca1m_tr3d_candidate_evidence.py",
    )
    return {
        str(source.relative_to(ROOT)): sha256_file(_regular(source, "dataset code source"))
        for source in sources
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output_dataset_target = args.output_dataset.resolve()
    output_manifest_target = args.output_manifest.resolve()
    if output_dataset_target == output_manifest_target:
        raise ValueError("dataset and manifest outputs must be distinct")
    if output_dataset_target.exists() or output_manifest_target.exists():
        raise FileExistsError("refusing a partial/overwriting dataset transaction")
    split_path, split = _json(args.split_manifest, "frozen benefit split manifest")
    if split_path.stat().st_mode & 0o222:
        raise ValueError("frozen benefit split manifest must be sealed read-only")
    if (
        split.get("schema") != SPLIT_SCHEMA
        or split.get("complete") is not True
        or split.get("train_only") is not True
        or split.get("official_validation_access") is not False
    ):
        raise ValueError("benefit split manifest contract mismatch")
    roles = split.get("roles")
    if not isinstance(roles, Mapping):
        raise ValueError("benefit split manifest lacks roles")
    partition = PARTITIONS[args.partition]
    role_scene_lists: dict[str, tuple[str, ...]] = {}
    all_split_scenes: list[str] = []
    for role_name in ROLE_FOLDS:
        row = roles.get(role_name)
        if not isinstance(row, Mapping):
            raise ValueError(f"split manifest lacks role {role_name}")
        scene_path = split_path.parent / str(row.get("scene_list", ""))
        scenes = _scene_rows(scene_path, f"{role_name} scene list")
        if (
            scene_path.resolve().stat().st_mode & 0o222
            or
            len(scenes) != EXPECTED_ROLE_COUNTS[role_name]
            or len(scenes) != int(row.get("scene_count", -1))
            or sha256_file(scene_path) != row.get("scene_list_sha256")
            or tuple(int(value) for value in row.get("folds", ()))
            != ROLE_FOLDS[role_name]
        ):
            raise ValueError(f"split role contract mismatch: {role_name}")
        role_scene_lists[role_name] = scenes
        all_split_scenes.extend(scenes)
    if (
        len(all_split_scenes) != 100
        or len(set(all_split_scenes)) != 100
        or len(roles) != len(ROLE_FOLDS)
    ):
        raise ValueError("frozen split roles must be an exact disjoint 60/20/20 train100")
    selected_scenes = [
        scene for role_name in partition["roles"] for scene in role_scene_lists[role_name]
    ]
    scenes = tuple(sorted(selected_scenes))
    if len(scenes) != len(set(scenes)):
        raise ValueError("partition roles overlap")

    val_path = _regular(args.official_val_list, "frozen official validation list")
    val_ids = _validation_ids(val_path)
    if (
        sha256_file(val_path) != split.get("official_validation_scene_list_sha256")
        or val_path.stat().st_mode & 0o222
    ):
        raise ValueError("official full107 validation identity is not frozen read-only")
    if set(all_split_scenes) & set(val_ids):
        raise ValueError("train100 split overlaps official CA-1M validation")

    if args.partition == "fit_dev":
        if (
            args.calibration_model is not None
            or args.calibration_report is not None
            or args.locked_access_receipt is not None
        ):
            raise ValueError("fit_dev must not consume a prior calibration artifact")
        locked_receipt_payload: Mapping[str, Any] | None = None
    else:
        if (
            args.calibration_model is None
            or args.calibration_report is None
            or args.locked_access_receipt is None
        ):
            raise ValueError(
                "locked_internal_check requires model, report, and one-time receipt"
            )
        if args.locked_access_receipt.resolve() != LOCKED_RECEIPT_PATH.resolve():
            raise ValueError("locked fold access receipt must use the frozen canonical path")
        if args.locked_access_receipt.exists():
            raise FileExistsError("locked fold has already been consumed or attempted")
        calibration_model = _regular(
            args.calibration_model, "sealed benefit calibration model"
        )
        calibration_report_path, calibration_report = _json(
            args.calibration_report, "sealed benefit calibration report"
        )
        if (
            calibration_model.stat().st_mode & 0o222
            or calibration_report_path.stat().st_mode & 0o222
            or calibration_report.get("schema")
            != "boxfusion.ca1m_tr3d_benefit_calibration_report.v1"
            or calibration_report.get("complete") is not True
            or calibration_report.get("train_only") is not True
            or calibration_report.get("threshold_dev_gate_passed") is not True
            or calibration_report.get("locked_internal_check_accessed") is not False
            or calibration_report.get("locked_internal_check_authorized") is not True
            or calibration_report.get("model_sha256") != sha256_file(calibration_model)
            or calibration_report.get("split_manifest_sha256") != sha256_file(split_path)
        ):
            raise ValueError("locked fold remains sealed because calibration did not pass")
        with np.load(calibration_model, allow_pickle=False) as calibration_archive:
            if (
                str(_scalar(calibration_archive, "schema"))
                != "boxfusion.ca1m_tr3d_benefit_calibration.v1"
                or bool(_scalar(calibration_archive, "complete")) is not True
                or bool(_scalar(calibration_archive, "train_only")) is not True
                or bool(_scalar(calibration_archive, "activation_authorized")) is not False
                or bool(_scalar(calibration_archive, "threshold_dev_gate_passed"))
                is not True
                or bool(_scalar(calibration_archive, "one_time_internal_check_pending"))
                is not True
                or str(_scalar(calibration_archive, "split_manifest_sha256"))
                != sha256_file(split_path)
                or str(_scalar(calibration_archive, "dataset_sha256"))
                != calibration_report.get("dataset_sha256")
                or str(_scalar(calibration_archive, "dataset_manifest_sha256"))
                != calibration_report.get("dataset_manifest_sha256")
            ):
                raise ValueError("calibration model/report cross-contract mismatch")
        calibration_code = calibration_report.get("code_manifest")
        if not isinstance(calibration_code, Mapping):
            raise ValueError("calibration report lacks frozen code identity")
        for relative in (
            "tools/build_ca1m_tr3d_benefit_dataset.py",
            "tools/train_ca1m_tr3d_benefit_gate.py",
            "boxfusion/ca1m_tr3d_terminal_gate.py",
        ):
            if calibration_code.get(relative) != sha256_file(
                _regular(ROOT / relative, relative)
            ):
                raise ValueError(f"code changed before locked fold access: {relative}")
        locked_receipt_payload = {
            "schema": "boxfusion.ca1m_tr3d_benefit_locked_access_receipt.v1",
            "complete": True,
            "access_started": True,
            "train_only": True,
            "official_validation_access": False,
            "fold_ids": [1],
            "calibration_model_sha256": sha256_file(calibration_model),
            "calibration_report_sha256": sha256_file(calibration_report_path),
            "split_manifest_sha256": sha256_file(split_path),
            "output_dataset": str(args.output_dataset.resolve()),
            "output_manifest": str(args.output_manifest.resolve()),
        }

    source_path = _sealed_regular(
        args.source_dataset, "frozen native-B6 train dataset"
    )
    source_manifest_path, source_manifest = _json(
        args.source_dataset_manifest, "native-B6 dataset manifest"
    )
    if source_manifest_path.stat().st_mode & 0o222:
        raise ValueError("native-B6 dataset manifest must be sealed read-only")
    if (
        source_manifest.get("schema") != SOURCE_MANIFEST_SCHEMA
        or source_manifest.get("complete") is not True
        or source_manifest.get("train_only") is not True
        or source_manifest.get("validation_ground_truth_access") is not False
        or source_manifest.get("validation_prediction_access") is not False
        or source_manifest.get("dataset", {}).get("sha256") != sha256_file(source_path)
        or split.get("source_dataset_sha256") != sha256_file(source_path)
        or split.get("source_dataset_manifest_sha256")
        != sha256_file(source_manifest_path)
    ):
        raise ValueError("frozen native-B6 source dataset provenance mismatch")
    source_scene_records_raw = source_manifest.get("scenes")
    if not isinstance(source_scene_records_raw, list):
        raise ValueError("native-B6 source manifest lacks per-scene provenance")
    source_scene_records: dict[str, Mapping[str, Any]] = {}
    for row in source_scene_records_raw:
        if not isinstance(row, Mapping):
            raise ValueError("native-B6 source manifest has invalid scene record")
        scene = str(row.get("scene_id", ""))
        if _SCENE_RE.fullmatch(scene) is None or scene in source_scene_records:
            raise ValueError("native-B6 source manifest scene IDs are invalid")
        source_scene_records[scene] = row
    with np.load(source_path, allow_pickle=False) as source:
        if str(_scalar(source, "schema")) != SOURCE_DATASET_SCHEMA:
            raise ValueError("unsupported native-B6 source dataset")
        source_arrays = {
            key: np.array(source[key], copy=True)
            for key in (
                "quality_features", "scene_ids", "row_indices",
                "prediction_scores", "prediction_corners", "fold_ids",
                "valid_evidence",
            )
        }
    source_scenes = np.asarray(source_arrays["scene_ids"]).astype(str)
    source_folds = np.asarray(source_arrays["fold_ids"], dtype=np.int64)
    source_rows_total = len(source_scenes)
    if (
        np.asarray(source_arrays["quality_features"]).dtype != np.float32
        or np.asarray(source_arrays["quality_features"]).shape
        != (source_rows_total, len(NATIVE_FEATURE_NAMES))
        or np.asarray(source_arrays["prediction_corners"]).dtype != np.float32
        or np.asarray(source_arrays["prediction_corners"]).shape
        != (source_rows_total, 8, 3)
        or np.asarray(source_arrays["prediction_scores"]).dtype != np.float32
        or np.asarray(source_arrays["prediction_scores"]).shape != (source_rows_total,)
        or np.asarray(source_arrays["row_indices"]).dtype != np.int64
        or np.asarray(source_arrays["row_indices"]).shape != (source_rows_total,)
        or np.asarray(source_arrays["valid_evidence"]).dtype != np.bool_
        or np.asarray(source_arrays["valid_evidence"]).shape != (source_rows_total,)
        or not np.isfinite(source_arrays["quality_features"]).all()
        or not np.isfinite(source_arrays["prediction_corners"]).all()
        or not np.isfinite(source_arrays["prediction_scores"]).all()
    ):
        raise ValueError("native-B6 source dataset row schema is invalid")
    scene_fold: dict[str, int] = {}
    for scene in sorted(set(source_scenes.tolist())):
        values = np.unique(source_folds[source_scenes == scene])
        if len(values) != 1:
            raise ValueError(f"native-B6 source scene crosses folds: {scene}")
        scene_fold[scene] = int(values[0])
        source_indices = np.flatnonzero(source_scenes == scene)
        source_rows = np.asarray(source_arrays["row_indices"])[source_indices]
        if not np.array_equal(source_rows, np.arange(len(source_indices), dtype=np.int64)):
            raise ValueError(f"native-B6 source row indices are not canonical: {scene}")
    if set(all_split_scenes) != set(scene_fold) or len(scene_fold) != 100:
        raise ValueError("frozen split is not the exact native-B6 train100 scene set")
    if set(scenes) - set(scene_fold):
        raise ValueError("benefit partition is missing from native-B6 source dataset")
    if any(scene_fold[scene] not in partition["fold_ids"] for scene in scenes):
        raise ValueError("benefit partition fold mapping differs from source dataset")
    for role_name, role_scenes in role_scene_lists.items():
        if any(scene_fold[scene] not in ROLE_FOLDS[role_name] for scene in role_scenes):
            raise ValueError(f"split role-to-fold mapping differs for {role_name}")

    terminal_root = _directory(args.terminal_root, "terminal observer root")
    anchor_native_root = _directory(args.anchor_native_root, "anchor native root")
    candidate_native_root = _directory(
        args.candidate_native_root, "candidate native root"
    )
    data_root = _directory(args.data_root, "derived CA-1M train root")
    terminal_audit_path, terminal_audit = _json(
        args.terminal_audit, "sealed terminal observer audit"
    )
    candidate_audit_path, candidate_audit = _json(
        args.candidate_audit, "sealed candidate evidence audit"
    )
    if (
        terminal_audit_path.stat().st_mode & 0o222
        or candidate_audit_path.stat().st_mode & 0o222
    ):
        raise ValueError("GT-free collection audits must be sealed read-only")
    if (
        terminal_audit.get("schema") != TERMINAL_AUDIT_SCHEMA
        or terminal_audit.get("ok") is not True
        or terminal_audit.get("observer_only") is not True
        or terminal_audit.get("ground_truth_access") is not False
        or int(terminal_audit.get("scene_count", -1)) != 100
        or not isinstance(terminal_audit.get("scenes"), Mapping)
        or candidate_audit.get("schema") != CANDIDATE_AUDIT_SCHEMA
        or candidate_audit.get("ok") is not True
        or candidate_audit.get("complete") is not True
        or candidate_audit.get("ground_truth_access") is not False
        or candidate_audit.get("mutation_enabled") is not False
        or int(candidate_audit.get("scene_count", -1)) != 100
        or candidate_audit.get("terminal_audit_sha256")
        != sha256_file(terminal_audit_path)
        or not isinstance(candidate_audit.get("scenes"), Mapping)
    ):
        raise ValueError("sealed GT-free candidate collection audit mismatch")
    terminal_scene_audit = terminal_audit["scenes"]
    candidate_scene_audit = candidate_audit["scenes"]
    if (
        set(terminal_scene_audit) != set(scene_fold)
        or set(candidate_scene_audit) != set(scene_fold)
        or set(source_scene_records) != set(scene_fold)
    ):
        raise ValueError("sealed candidate audits are not the exact train100 set")
    frozen_native = split.get("frozen_native_b6_condition")
    frozen_terminal = split.get("frozen_terminal_condition")
    if not isinstance(frozen_native, Mapping) or not isinstance(frozen_terminal, Mapping):
        raise ValueError("split does not freeze native-B6 and terminal model identities")
    if (
        frozen_terminal.get("terminal_observer_audit_sha256")
        != sha256_file(terminal_audit_path)
        or frozen_terminal.get("candidate_evidence_audit_sha256")
        != sha256_file(candidate_audit_path)
    ):
        raise ValueError("split terminal audit identity differs from sealed collection")

    # Finish the complete GT-free identity preflight for all 100 scenes before
    # opening even one derived GT file.  This prevents a half-verified cache set
    # from being joined with labels.
    preflight_hashes: dict[str, dict[str, str]] = {}
    expected_scenes = set(scene_fold)
    actual_roots = {
        "terminal": {
            path.name[: -len("_ca1m_tr3d_terminal.npz")]
            for path in terminal_root.glob("*_ca1m_tr3d_terminal.npz")
            if path.is_file() and not path.is_symlink()
        },
        "anchor_native": {
            path.name[: -len("_ca1m_native_b6.npz")]
            for path in anchor_native_root.glob("*_ca1m_native_b6.npz")
            if path.is_file() and not path.is_symlink()
        },
        "candidate_native": {
            path.name[: -len("_ca1m_native_b6.npz")]
            for path in candidate_native_root.glob("*_ca1m_native_b6.npz")
            if path.is_file() and not path.is_symlink()
        },
    }
    if any(actual != expected_scenes for actual in actual_roots.values()):
        raise ValueError("candidate collection roots are not exact train100 file sets")
    for scene in sorted(expected_scenes):
        terminal_path = _regular(
            terminal_root / f"{scene}_ca1m_tr3d_terminal.npz",
            f"terminal cache {scene}",
        )
        anchor_path = _regular(
            anchor_native_root / f"{scene}_ca1m_native_b6.npz",
            f"anchor native evidence {scene}",
        )
        candidate_path = _regular(
            candidate_native_root / f"{scene}_ca1m_native_b6.npz",
            f"candidate native evidence {scene}",
        )
        if any(path.stat().st_mode & 0o222 for path in (terminal_path, anchor_path, candidate_path)):
            raise ValueError(f"GT-free candidate artifact is not sealed read-only: {scene}")
        terminal_sha = sha256_file(terminal_path)
        anchor_sha = sha256_file(anchor_path)
        candidate_sha = sha256_file(candidate_path)
        source_record = source_scene_records[scene]
        if (
            terminal_sha != terminal_scene_audit[scene].get("artifact_sha256")
            or terminal_sha != candidate_scene_audit[scene].get("terminal_cache_sha256")
            or candidate_sha
            != candidate_scene_audit[scene].get("candidate_evidence_sha256")
            or anchor_sha != source_record.get("observer", {}).get("sha256")
        ):
            raise ValueError(f"GT-free artifact hash changed before label join: {scene}")
        with np.load(terminal_path, allow_pickle=False) as terminal_preflight:
            contract = {
                "native_b6_checkpoint_sha256": frozen_native.get("checkpoint_sha256"),
                "native_b6_manifest_sha256": frozen_native.get("manifest_sha256"),
                "checkpoint_sha256": frozen_terminal.get("checkpoint_sha256"),
                "config_sha256": frozen_terminal.get("config_sha256"),
                "code_manifest_sha256": frozen_terminal.get("code_manifest_sha256"),
                "native_b6_diagnostic_sha256": anchor_sha,
            }
            for key, expected in contract.items():
                if not isinstance(expected, str) or _SHA_RE.fullmatch(expected) is None:
                    raise ValueError(f"invalid frozen hash contract {key}")
                if str(_scalar(terminal_preflight, key)) != expected:
                    raise ValueError(f"terminal cache violates frozen {key}: {scene}")
        preflight_hashes[scene] = {
            "terminal_cache_sha256": terminal_sha,
            "anchor_native_evidence_sha256": anchor_sha,
            "candidate_native_evidence_sha256": candidate_sha,
        }

    if locked_receipt_payload is not None:
        # This is the deliberate one-time transition from GT-free metadata to
        # fold-1 labels.  A failure after this point leaves the immutable
        # receipt in place and therefore cannot be silently retried/tuned.
        _write_json_create_only(args.locked_access_receipt, locked_receipt_payload)

    candidate_features: list[np.ndarray] = []
    candidate_scene_ids: list[np.ndarray] = []
    candidate_fold_ids: list[np.ndarray] = []
    candidate_rows_all: list[np.ndarray] = []
    candidate_anchor_rows: list[np.ndarray] = []
    candidate_corners_all: list[np.ndarray] = []
    candidate_scores_all: list[np.ndarray] = []
    candidate_valid_all: list[np.ndarray] = []
    anchor_valid_for_candidate_all: list[np.ndarray] = []
    terminal_sha_for_candidate_all: list[np.ndarray] = []
    anchor_native_sha_for_candidate_all: list[np.ndarray] = []
    candidate_native_sha_for_candidate_all: list[np.ndarray] = []
    quality_targets: list[np.ndarray] = []
    benefit_targets: list[np.ndarray] = []
    target_switches: list[np.ndarray] = []
    anchor_best_for_candidate: list[np.ndarray] = []
    candidate_best_indices: list[np.ndarray] = []
    anchor_best_ious_for_candidate: list[np.ndarray] = []
    candidate_best_ious: list[np.ndarray] = []
    candidate_on_anchor_ious: list[np.ndarray] = []
    same_gt_gains: list[np.ndarray] = []
    cross_gain: dict[float, list[np.ndarray]] = {
        threshold: [] for threshold in (0.15, 0.25, 0.50)
    }
    cross_loss: dict[float, list[np.ndarray]] = {
        threshold: [] for threshold in (0.15, 0.25, 0.50)
    }
    identity_cross_gain: dict[float, list[np.ndarray]] = {
        threshold: [] for threshold in (0.15, 0.25, 0.50)
    }
    identity_cross_loss: dict[float, list[np.ndarray]] = {
        threshold: [] for threshold in (0.15, 0.25, 0.50)
    }

    baseline_scene_ids: list[np.ndarray] = []
    baseline_fold_ids: list[np.ndarray] = []
    baseline_rows: list[np.ndarray] = []
    baseline_corners: list[np.ndarray] = []
    baseline_scores: list[np.ndarray] = []
    baseline_best_gt_indices: list[np.ndarray] = []
    baseline_best_ious: list[np.ndarray] = []
    gt_counts: list[int] = []
    per_scene: dict[str, Any] = {}

    for scene in scenes:
        terminal_path = _regular(
            terminal_root / f"{scene}_ca1m_tr3d_terminal.npz",
            f"terminal cache {scene}",
        )
        candidate_native_path = _regular(
            candidate_native_root / f"{scene}_ca1m_native_b6.npz",
            f"candidate native evidence {scene}",
        )
        if (
            sha256_file(terminal_path)
            != preflight_hashes[scene]["terminal_cache_sha256"]
            or sha256_file(candidate_native_path)
            != preflight_hashes[scene]["candidate_native_evidence_sha256"]
        ):
            raise ValueError(f"sealed candidate artifact changed: {scene}")
        with np.load(terminal_path, allow_pickle=False) as terminal:
            anchor_corners = np.array(terminal["anchor_corners"], copy=True)
            anchor_scores = np.array(terminal["anchor_scores"], copy=True)
            all_candidate_corners = np.array(terminal["candidate_corners"], copy=True)
            all_candidate_scores = np.array(terminal["candidate_scores"], copy=True)
            anchor_features, anchor_valid, anchor_evidence_sha = _native_evidence(
                anchor_native_root / f"{scene}_ca1m_native_b6.npz",
                scene=scene,
                expected_corners=anchor_corners,
                expected_scores=None,
            )
            candidate_native, candidate_valid, _ = _native_evidence(
                candidate_native_path,
                scene=scene,
                expected_corners=all_candidate_corners,
                expected_scores=all_candidate_scores,
            )
            batch = build_terminal_gate_features(
                terminal,
                anchor_native_evidence=anchor_features,
                candidate_native_evidence=candidate_native,
            )

        gt_path = _regular(
            data_root / scene / "derived_train_gt_boxes.npy", f"derived train GT {scene}"
        )
        gt_manifest_path, gt_manifest = _json(
            data_root / scene / "derived_train_gt_manifest.json",
            f"derived train GT manifest {scene}",
        )
        gt = np.load(gt_path, allow_pickle=False)
        source_record = source_scene_records[scene]
        gt_sha = sha256_file(gt_path)
        gt_manifest_sha = sha256_file(gt_manifest_path)
        if (
            gt.dtype != np.float64
            or gt.ndim != 3
            or gt.shape[1:] != (8, 3)
            or not np.isfinite(gt).all()
            or gt_manifest.get("schema") != "boxfusion.ca1m_native_b6_train_scene.v1"
            or gt_manifest.get("scene_id") != scene
            or gt_manifest.get("train_only") is not True
            or gt_manifest.get("official_validation_comparable") is not False
            or gt_manifest.get("artifacts", {})
            .get("derived_train_gt_boxes.npy", {})
            .get("sha256")
            != gt_sha
            or source_record.get("derived_gt_sha256") != gt_sha
            or source_record.get("derived_gt_manifest", {}).get("sha256")
            != gt_manifest_sha
            or int(source_record.get("fold_id", -1)) != scene_fold[scene]
        ):
            raise ValueError(f"derived train GT provenance mismatch: {scene}")
        world_aabb(gt)
        anchor_best_iou, anchor_best_index, anchor_matrix = _target(anchor_corners, gt)
        source_indices = np.flatnonzero(source_scenes == scene)
        if (
            len(source_indices) != len(anchor_corners)
            or not np.array_equal(
                np.asarray(source_arrays["row_indices"])[source_indices],
                np.arange(len(anchor_corners), dtype=np.int64),
            )
            or not np.array_equal(
                np.asarray(source_arrays["prediction_corners"])[source_indices],
                anchor_corners,
            )
            or not np.array_equal(
                np.asarray(source_arrays["prediction_scores"])[source_indices],
                anchor_features[:, 0].astype(np.float32),
            )
            or not np.array_equal(
                np.asarray(source_arrays["quality_features"])[source_indices],
                anchor_features,
            )
            or not np.array_equal(
                np.asarray(source_arrays["valid_evidence"])[source_indices],
                anchor_valid,
            )
        ):
            raise ValueError(f"source native-B6 rows differ from recomputation: {scene}")
        selected_candidate_corners = all_candidate_corners[batch.candidate_rows]
        selected_candidate_scores = all_candidate_scores[batch.candidate_rows]
        candidate_best_iou, candidate_best_index, candidate_matrix = _target(
            selected_candidate_corners, gt
        )
        anchor_index = batch.anchor_indices
        anchor_target_index = anchor_best_index[anchor_index]
        same_target = candidate_best_index == anchor_target_index
        if len(gt):
            candidate_on_anchor = candidate_matrix[
                np.arange(len(batch.candidate_rows)), anchor_target_index
            ]
        else:
            candidate_on_anchor = np.zeros(len(batch.candidate_rows), np.float64)
        anchor_for_candidate_iou = anchor_best_iou[anchor_index]
        gain = candidate_on_anchor - anchor_for_candidate_iou
        quality = candidate_best_iou > 0.25
        benefit = same_target & (gain >= 0.05)
        fold = scene_fold[scene]
        count = len(batch.candidate_rows)

        candidate_features.append(batch.features)
        candidate_scene_ids.append(np.full(count, scene, dtype="U8"))
        candidate_fold_ids.append(np.full(count, fold, dtype=np.int8))
        candidate_rows_all.append(batch.candidate_rows)
        candidate_anchor_rows.append(anchor_index)
        candidate_corners_all.append(selected_candidate_corners)
        candidate_scores_all.append(selected_candidate_scores)
        candidate_valid_all.append(candidate_valid[batch.candidate_rows])
        anchor_valid_for_candidate_all.append(anchor_valid[anchor_index])
        terminal_sha_for_candidate_all.append(
            np.full(count, preflight_hashes[scene]["terminal_cache_sha256"], dtype="U64")
        )
        anchor_native_sha_for_candidate_all.append(
            np.full(
                count,
                preflight_hashes[scene]["anchor_native_evidence_sha256"],
                dtype="U64",
            )
        )
        candidate_native_sha_for_candidate_all.append(
            np.full(
                count,
                preflight_hashes[scene]["candidate_native_evidence_sha256"],
                dtype="U64",
            )
        )
        quality_targets.append(quality.astype(np.bool_))
        benefit_targets.append(benefit.astype(np.bool_))
        target_switches.append((~same_target).astype(np.bool_))
        anchor_best_for_candidate.append(anchor_target_index)
        candidate_best_indices.append(candidate_best_index)
        anchor_best_ious_for_candidate.append(anchor_for_candidate_iou)
        candidate_best_ious.append(candidate_best_iou)
        candidate_on_anchor_ious.append(candidate_on_anchor)
        same_gt_gains.append(gain)
        for threshold in (0.15, 0.25, 0.50):
            cross_gain[threshold].append(
                ((anchor_for_candidate_iou <= threshold) & (candidate_best_iou > threshold)).astype(np.bool_)
            )
            cross_loss[threshold].append(
                ((anchor_for_candidate_iou > threshold) & (candidate_best_iou <= threshold)).astype(np.bool_)
            )
            identity_cross_gain[threshold].append(
                (
                    same_target
                    & (anchor_for_candidate_iou <= threshold)
                    & (candidate_on_anchor > threshold)
                ).astype(np.bool_)
            )
            identity_cross_loss[threshold].append(
                (
                    same_target
                    & (anchor_for_candidate_iou > threshold)
                    & (candidate_on_anchor <= threshold)
                ).astype(np.bool_)
            )

        anchor_count = len(anchor_corners)
        baseline_scene_ids.append(np.full(anchor_count, scene, dtype="U8"))
        baseline_fold_ids.append(np.full(anchor_count, fold, dtype=np.int8))
        baseline_rows.append(np.arange(anchor_count, dtype=np.int64))
        baseline_corners.append(anchor_corners)
        baseline_scores.append(anchor_scores)
        baseline_best_gt_indices.append(anchor_best_index)
        baseline_best_ious.append(anchor_best_iou)
        gt_counts.append(len(gt))
        per_scene[scene] = {
            "fold_id": fold,
            "candidate_rows": count,
            "quality25_positive": int(np.count_nonzero(quality)),
            "benefit05_positive": int(np.count_nonzero(benefit)),
            "target_switch_rows": int(np.count_nonzero(~same_target)),
            "baseline_rows": anchor_count,
            "gt_boxes": len(gt),
            "terminal_cache_sha256": preflight_hashes[scene]["terminal_cache_sha256"],
            "anchor_native_evidence_sha256": anchor_evidence_sha,
            "candidate_native_evidence_sha256": preflight_hashes[scene]["candidate_native_evidence_sha256"],
            "derived_gt_sha256": gt_sha,
            "derived_gt_manifest_sha256": gt_manifest_sha,
        }

    def concatenate(values: list[np.ndarray], *, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        if not values:
            return np.empty(shape, dtype=dtype)
        return np.ascontiguousarray(np.concatenate(values, axis=0), dtype=dtype)

    arrays: dict[str, Any] = {
        "schema": np.asarray(SCHEMA),
        "complete": np.asarray(True, np.bool_),
        "train_only": np.asarray(True, np.bool_),
        "official_validation_access": np.asarray(False, np.bool_),
        "partition": np.asarray(args.partition),
        "feature_schema": np.asarray(FEATURE_SCHEMA),
        "feature_names": np.asarray(FEATURE_NAMES),
        "quality_target_schema": np.asarray(QUALITY_TARGET),
        "benefit_target_schema": np.asarray(BENEFIT_TARGET),
        "features": concatenate(
            candidate_features, shape=(0, len(FEATURE_NAMES)), dtype=np.float32
        ),
        "scene_ids": concatenate(candidate_scene_ids, shape=(0,), dtype="U8"),
        "fold_ids": concatenate(candidate_fold_ids, shape=(0,), dtype=np.int8),
        "candidate_rows": concatenate(candidate_rows_all, shape=(0,), dtype=np.int64),
        "anchor_indices": concatenate(candidate_anchor_rows, shape=(0,), dtype=np.int64),
        "candidate_corners": concatenate(
            candidate_corners_all, shape=(0, 8, 3), dtype=np.float32
        ),
        "candidate_scores": concatenate(
            candidate_scores_all, shape=(0,), dtype=np.float32
        ),
        "candidate_valid_evidence": concatenate(
            candidate_valid_all, shape=(0,), dtype=np.bool_
        ),
        "anchor_valid_evidence": concatenate(
            anchor_valid_for_candidate_all, shape=(0,), dtype=np.bool_
        ),
        "terminal_cache_sha256": concatenate(
            terminal_sha_for_candidate_all, shape=(0,), dtype="U64"
        ),
        "anchor_native_evidence_sha256": concatenate(
            anchor_native_sha_for_candidate_all, shape=(0,), dtype="U64"
        ),
        "candidate_native_evidence_sha256": concatenate(
            candidate_native_sha_for_candidate_all, shape=(0,), dtype="U64"
        ),
        "quality25_target": concatenate(quality_targets, shape=(0,), dtype=np.bool_),
        "benefit05_target": concatenate(benefit_targets, shape=(0,), dtype=np.bool_),
        "target_switch": concatenate(target_switches, shape=(0,), dtype=np.bool_),
        "anchor_best_gt_indices": concatenate(
            anchor_best_for_candidate, shape=(0,), dtype=np.int64
        ),
        "candidate_best_gt_indices": concatenate(
            candidate_best_indices, shape=(0,), dtype=np.int64
        ),
        "anchor_best_iou": concatenate(
            anchor_best_ious_for_candidate, shape=(0,), dtype=np.float64
        ),
        "candidate_best_iou": concatenate(
            candidate_best_ious, shape=(0,), dtype=np.float64
        ),
        "candidate_iou_on_anchor_gt": concatenate(
            candidate_on_anchor_ious, shape=(0,), dtype=np.float64
        ),
        "same_gt_iou_gain": concatenate(same_gt_gains, shape=(0,), dtype=np.float64),
        "baseline_scene_ids": concatenate(
            baseline_scene_ids, shape=(0,), dtype="U8"
        ),
        "baseline_fold_ids": concatenate(
            baseline_fold_ids, shape=(0,), dtype=np.int8
        ),
        "baseline_row_indices": concatenate(
            baseline_rows, shape=(0,), dtype=np.int64
        ),
        "baseline_corners": concatenate(
            baseline_corners, shape=(0, 8, 3), dtype=np.float32
        ),
        "baseline_scores": concatenate(
            baseline_scores, shape=(0,), dtype=np.float32
        ),
        "baseline_best_gt_indices": concatenate(
            baseline_best_gt_indices, shape=(0,), dtype=np.int64
        ),
        "baseline_best_iou": concatenate(
            baseline_best_ious, shape=(0,), dtype=np.float64
        ),
        "scene_table": np.asarray(scenes, dtype="U8"),
        "scene_fold_ids": np.asarray([scene_fold[scene] for scene in scenes], np.int8),
        "scene_gt_counts": np.asarray(gt_counts, np.int64),
    }
    for threshold in (0.15, 0.25, 0.50):
        suffix = f"{int(round(threshold * 100)):02d}"
        arrays[f"cross{suffix}_gain"] = concatenate(
            cross_gain[threshold], shape=(0,), dtype=np.bool_
        )
        arrays[f"cross{suffix}_loss"] = concatenate(
            cross_loss[threshold], shape=(0,), dtype=np.bool_
        )
        arrays[f"identity_cross{suffix}_gain"] = concatenate(
            identity_cross_gain[threshold], shape=(0,), dtype=np.bool_
        )
        arrays[f"identity_cross{suffix}_loss"] = concatenate(
            identity_cross_loss[threshold], shape=(0,), dtype=np.bool_
        )
    row_count = len(arrays["features"])
    baseline_count = len(arrays["baseline_corners"])
    if row_count < 1 or baseline_count < 1 or not np.isfinite(arrays["features"]).all():
        raise ValueError("constructed benefit dataset is empty or non-finite")
    output_dataset = _write_npz_create_only(output_dataset_target, arrays)
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "complete": True,
        "train_only": True,
        "official_validation_access": False,
        "official_validation_comparable": False,
        "partition": args.partition,
        "fold_ids": list(partition["fold_ids"]),
        "role_names": list(partition["roles"]),
        "scene_count": len(scenes),
        "candidate_rows": row_count,
        "baseline_rows": baseline_count,
        "quality25_positive": int(np.count_nonzero(arrays["quality25_target"])),
        "benefit05_positive": int(np.count_nonzero(arrays["benefit05_target"])),
        "target_switch_rows": int(np.count_nonzero(arrays["target_switch"])),
        "candidate_valid_evidence_rows": int(
            np.count_nonzero(arrays["candidate_valid_evidence"])
        ),
        "anchor_valid_evidence_rows": int(
            np.count_nonzero(arrays["anchor_valid_evidence"])
        ),
        "crossing_counts": {
            suffix: {
                "gain": int(np.count_nonzero(arrays[f"cross{suffix}_gain"])),
                "loss": int(np.count_nonzero(arrays[f"cross{suffix}_loss"])),
                "identity_gain": int(
                    np.count_nonzero(arrays[f"identity_cross{suffix}_gain"])
                ),
                "identity_loss": int(
                    np.count_nonzero(arrays[f"identity_cross{suffix}_loss"])
                ),
            }
            for suffix in ("15", "25", "50")
        },
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "quality_target": QUALITY_TARGET,
        "benefit_target": BENEFIT_TARGET,
        "dataset_path": str(output_dataset),
        "dataset_sha256": sha256_file(output_dataset),
        "split_manifest_path": str(split_path),
        "split_manifest_sha256": sha256_file(split_path),
        "source_dataset_sha256": sha256_file(source_path),
        "source_dataset_manifest_sha256": sha256_file(source_manifest_path),
        "terminal_observer_audit_sha256": sha256_file(terminal_audit_path),
        "candidate_evidence_audit_sha256": sha256_file(candidate_audit_path),
        "official_validation_scene_list_sha256": sha256_file(
            _regular(args.official_val_list, "official validation list")
        ),
        "validation_scene_ids": list(val_ids),
        "validation_overlap_count": 0,
        "ground_truth_join_after_candidate_seal": True,
        "all_train100_gt_free_artifacts_rehashed_before_gt_join": True,
        "source_native_b6_rows_recomputed_and_exact": True,
        "source_native_b6_target_arrays_opened": False,
        "code_manifest": _code_manifest(),
        "locked_calibration_model_sha256": (
            None
            if args.partition == "fit_dev"
            else sha256_file(_regular(args.calibration_model, "calibration model"))
        ),
        "locked_calibration_report_sha256": (
            None
            if args.partition == "fit_dev"
            else sha256_file(_regular(args.calibration_report, "calibration report"))
        ),
        "per_scene": per_scene,
    }
    try:
        _write_json_create_only(output_manifest_target, manifest)
    except BaseException:
        # The NPZ was created by this transaction and no manifest was sealed;
        # remove only that just-created orphan so the create-only run can retry.
        try:
            output_dataset.unlink()
        except FileNotFoundError:
            pass
        raise
    return manifest


def main() -> int:
    args = parser().parse_args()
    result = run(args)
    print(json.dumps({
        "complete": result["complete"],
        "partition": result["partition"],
        "scene_count": result["scene_count"],
        "candidate_rows": result["candidate_rows"],
        "quality25_positive": result["quality25_positive"],
        "benefit05_positive": result["benefit05_positive"],
        "dataset_sha256": result["dataset_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
