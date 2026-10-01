#!/usr/bin/env python3
"""Create an isolated OpenBox-SMOV R2 visibility-v2 prediction tree.

Only geometry rows selected by ``would_replace_mask`` are replaced.  Native
labels, scores, row order, and row count are copied exactly.  The command has
no evaluator, ground-truth, CLIP, model, or training input and never edits an
input artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import struct
import sys
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


SCHEMA = "boxfusion.openbox_smov_r2_counterfactual_materialization.v1"
SIDECAR_SCHEMA = "boxfusion.openbox_smov_r2_shadow.v2"
PREDICTION_SUFFIX = "_boxes.pkl"
SIDECAR_SUFFIX = "_openbox_smov_r2_shadow.npz"
SIDECAR_FIELDS = frozenset(
    {
        "schema",
        "native_corners",
        "native_scores",
        "stable_ids",
        "counterfactual_corners",
        "would_replace_mask",
        "receipts_json",
    }
)
RECEIPT_FIELDS = frozenset(
    {
        "native_index",
        "stable_id",
        "reason",
        "hypothesis",
        "view_frame_ids",
        "native_corners",
        "candidate_corners",
        "native_projection_iou",
        "candidate_projection_iou",
        "native_support",
        "candidate_support",
        "native_free_space",
        "candidate_free_space",
        "center_shift_m",
        "volume_ratio",
        "would_replace",
        "face_extension_signs",
        "face_extension_delta_m",
        "face_strong_mask",
        "face_weak_mask",
    }
)
ALLOWED_HYPOTHESES = frozenset(
    f"{yaw}+{recipe}"
    for yaw in ("native_yaw_quantile", "pca_yaw_quantile")
    for recipe in ("base", "face_x", "face_y", "face_xy")
)
MAX_PREDICTION_BYTES = 64 * 1024 * 1024
MAX_SIDECAR_COMPRESSED_BYTES = 64 * 1024 * 1024
MAX_SIDECAR_UNCOMPRESSED_BYTES = 32 * 1024 * 1024
FACE_EXTENSION_MIN_M = 0.05
FACE_EXTENSION_MAX_M = 0.30


class MaterializationError(ValueError):
    """One frozen artifact violated the visibility-v2 contract."""


def _require(condition: object, message: str) -> None:
    if not bool(condition):
        raise MaterializationError(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path, label: str, maximum_bytes: int | None = None) -> Path:
    if path.is_symlink():
        raise MaterializationError(f"{label} must not be a symlink: {path}")
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(f"{label} is not a regular file: {path}")
    size = resolved.stat().st_size
    if size < 1 or (maximum_bytes is not None and size > maximum_bytes):
        raise MaterializationError(f"{label} has invalid size {size}: {path}")
    return resolved


def _directory(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise MaterializationError(f"{label} must not be a symlink: {path}")
    resolved = path.resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise FileNotFoundError(f"{label} is not a directory: {path}")
    return resolved


def _plain_scene_id(value: str) -> bool:
    return bool(value) and value not in {".", ".."} and "/" not in value and "\\" not in value


def read_scene_list(path: Path) -> tuple[str, ...]:
    source = _regular(path, "scene list")
    scenes = tuple(
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    _require(bool(scenes), "scene list must not be empty")
    _require(len(set(scenes)) == len(scenes), "scene list must contain unique scenes")
    _require(
        all(_plain_scene_id(scene) for scene in scenes),
        "scene ids must be plain file-name components",
    )
    return scenes


def _exact_artifacts(
    root: Path, scenes: tuple[str, ...], suffix: str, label: str
) -> dict[str, Path]:
    resolved = _directory(root, label)
    expected = {f"{scene}{suffix}" for scene in scenes}
    actual = {
        child.name
        for child in resolved.iterdir()
        if child.name.endswith(suffix)
    }
    _require(
        actual == expected,
        f"{label} artifact set differs: "
        f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}",
    )
    return {
        scene: _regular(
            resolved / f"{scene}{suffix}",
            f"{scene} {label}",
            MAX_PREDICTION_BYTES if suffix == PREDICTION_SUFFIX else MAX_SIDECAR_COMPRESSED_BYTES,
        )
        for scene in scenes
    }


class _PredictionUnpickler(pickle.Unpickler):
    """Allow only NumPy ndarray reconstruction helpers."""

    _ALLOWED = {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy.core.numeric", "_frombuffer"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy._core.numeric", "_frombuffer"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self._ALLOWED:
            raise pickle.UnpicklingError(
                f"forbidden prediction pickle global {module}.{name}"
            )
        return super().find_class(module, name)


@dataclass(frozen=True)
class NativePrediction:
    raw_bytes: bytes
    labels: tuple[int, ...]
    corners: np.ndarray
    scores: tuple[float, ...]


def _load_prediction(path: Path) -> NativePrediction:
    source = _regular(path, "native prediction", MAX_PREDICTION_BYTES)
    raw = source.read_bytes()
    stream = io.BytesIO(raw)
    try:
        payload = _PredictionUnpickler(stream).load()
    except (pickle.PickleError, AttributeError, EOFError, ImportError) as error:
        raise MaterializationError(f"malformed native prediction: {path}") from error
    _require(stream.read() == b"", f"native prediction has trailing bytes: {path}")
    _require(
        type(payload) is list
        and len(payload) == 1
        and type(payload[0]) is list,
        f"native prediction must be one canonical list batch: {path}",
    )
    labels: list[int] = []
    corners: list[np.ndarray] = []
    scores: list[float] = []
    for index, row in enumerate(payload[0]):
        _require(
            type(row) is tuple and len(row) == 3 and type(row[0]) is int,
            f"malformed prediction row {index}: {path}",
        )
        geometry = np.asarray(row[1])
        _require(
            type(row[1]) is np.ndarray
            and geometry.shape == (8, 3)
            and geometry.dtype == np.float32
            and (
                geometry.flags.c_contiguous
                or geometry.flags.f_contiguous
            )
            and np.isfinite(geometry).all(),
            f"invalid prediction geometry row {index}: {path}",
        )
        _require(
            type(row[2]) is float and math.isfinite(row[2]),
            f"invalid prediction score row {index}: {path}",
        )
        labels.append(int(row[0]))
        corners.append(np.array(geometry, dtype=np.float32, order="C", copy=True))
        scores.append(float(row[2]))
    stacked = (
        np.stack(corners)
        if corners
        else np.empty((0, 8, 3), dtype=np.float32)
    )
    return NativePrediction(raw, tuple(labels), stacked, tuple(scores))


def _reject_json_constant(value: str) -> None:
    raise MaterializationError(f"non-finite JSON constant: {value}")


def _unique_json_object(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise MaterializationError(f"duplicate receipt JSON key: {key}")
        result[key] = value
    return result


def _strict_receipts(raw: np.ndarray, path: Path) -> list[Mapping[str, object]]:
    _require(
        raw.ndim == 1 and raw.dtype == np.uint8,
        f"invalid receipt byte array: {path}",
    )
    try:
        value = json.loads(
            raw.tobytes().decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MaterializationError(f"invalid receipt JSON: {path}") from error
    _require(type(value) is list, f"receipts must be a list: {path}")
    return value


def _finite_metric(value: object, label: str) -> float:
    _require(
        type(value) in (int, float) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    return result


@dataclass(frozen=True)
class R2Sidecar:
    path_sha256: str
    native_corners: np.ndarray
    native_scores: np.ndarray
    stable_ids: np.ndarray
    counterfactual_corners: np.ndarray
    would_replace_mask: np.ndarray
    hypotheses: tuple[str | None, ...]


def _load_sidecar(path: Path, native: NativePrediction) -> R2Sidecar:
    source = _regular(path, "visibility-v2 sidecar", MAX_SIDECAR_COMPRESSED_BYTES)
    try:
        with zipfile.ZipFile(source) as archive:
            infos = archive.infolist()
            expected_members = {f"{name}.npy" for name in SIDECAR_FIELDS}
            _require(
                {info.filename for info in infos} == expected_members
                and len(infos) == len(expected_members),
                f"unexpected sidecar members: {path}",
            )
            _require(
                sum(info.file_size for info in infos)
                <= MAX_SIDECAR_UNCOMPRESSED_BYTES,
                f"sidecar exceeds expanded-size budget: {path}",
            )
        with np.load(source, allow_pickle=False) as archive:
            _require(
                set(archive.files) == set(SIDECAR_FIELDS),
                f"unexpected sidecar fields: {path}",
            )
            schema = np.asarray(archive["schema"])
            native_corners = np.array(archive["native_corners"], copy=True)
            native_scores = np.array(archive["native_scores"], copy=True)
            stable_ids = np.array(archive["stable_ids"], copy=True)
            counterfactual = np.array(archive["counterfactual_corners"], copy=True)
            mask = np.array(archive["would_replace_mask"], copy=True)
            receipt_bytes = np.array(archive["receipts_json"], copy=True)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise MaterializationError(f"invalid sidecar: {path}") from error

    _require(
        schema.shape == ()
        and schema.dtype.kind in "US"
        and schema.item() == SIDECAR_SCHEMA,
        f"sidecar is not strict visibility-v2: {path}",
    )
    count = len(native.labels)
    _require(
        native_corners.shape == (count, 8, 3)
        and native_corners.dtype == np.float32
        and native_scores.shape == (count,)
        and native_scores.dtype == np.float32
        and stable_ids.shape == (count,)
        and stable_ids.dtype == np.int64
        and counterfactual.shape == (count, 8, 3)
        and counterfactual.dtype == np.float32
        and mask.shape == (count,)
        and mask.dtype == np.bool_
        and np.isfinite(native_corners).all()
        and np.isfinite(native_scores).all()
        and np.isfinite(counterfactual).all()
        and np.all(stable_ids >= 0)
        and len(np.unique(stable_ids)) == count,
        f"sidecar arrays are malformed or not row-aligned: {path}",
    )
    source_scores = np.asarray(native.scores, dtype=np.float64)
    _require(
        np.array_equal(native.corners, native_corners)
        and all(
            float(source_scores[index]) == float(native_scores[index])
            for index in range(count)
        ),
        f"sidecar native arrays do not map prediction rows exactly: {path}",
    )
    _require(
        np.array_equal(counterfactual[~mask], native_corners[~mask]),
        f"unselected counterfactual geometry differs from native: {path}",
    )

    receipts = _strict_receipts(receipt_bytes, source)
    _require(len(receipts) == count, f"receipt count differs from prediction: {path}")
    hypotheses: list[str | None] = []
    metric_names = (
        "native_projection_iou",
        "candidate_projection_iou",
        "native_support",
        "candidate_support",
        "native_free_space",
        "candidate_free_space",
        "center_shift_m",
        "volume_ratio",
    )
    for index, receipt in enumerate(receipts):
        where = f"{path}:{index}"
        _require(
            isinstance(receipt, Mapping) and set(receipt) == set(RECEIPT_FIELDS),
            f"malformed visibility-v2 receipt: {where}",
        )
        _require(
            type(receipt["native_index"]) is int
            and receipt["native_index"] == index
            and type(receipt["stable_id"]) is int
            and receipt["stable_id"] == int(stable_ids[index])
            and type(receipt["would_replace"]) is bool
            and receipt["would_replace"] == bool(mask[index]),
            f"receipt identity or mask mismatch: {where}",
        )
        receipt_native = np.asarray(receipt["native_corners"], dtype=np.float32)
        _require(
            receipt_native.shape == (8, 3)
            and np.isfinite(receipt_native).all()
            and np.array_equal(receipt_native, native_corners[index]),
            f"receipt native geometry mismatch: {where}",
        )
        frames = receipt["view_frame_ids"]
        _require(
            type(frames) is list
            and all(type(frame) is int and frame >= 0 for frame in frames)
            and frames == sorted(set(frames)),
            f"receipt frame ids are not causal and unique: {where}",
        )
        candidate_raw = receipt["candidate_corners"]
        candidate: np.ndarray | None
        if candidate_raw is None:
            candidate = None
            _require(
                receipt["hypothesis"] is None
                and receipt["face_extension_signs"] is None
                and receipt["face_extension_delta_m"] is None
                and receipt["face_strong_mask"] is None
                and receipt["face_weak_mask"] is None,
                f"empty candidate has hypothesis or face metadata: {where}",
            )
        else:
            candidate = np.asarray(candidate_raw, dtype=np.float32)
            _require(
                candidate.shape == (8, 3) and np.isfinite(candidate).all(),
                f"invalid receipt candidate: {where}",
            )
            _require(
                receipt["hypothesis"] in ALLOWED_HYPOTHESES,
                f"unknown visibility-v2 hypothesis: {where}",
            )
            signs = receipt["face_extension_signs"]
            deltas = receipt["face_extension_delta_m"]
            strong = receipt["face_strong_mask"]
            weak = receipt["face_weak_mask"]
            _require(
                type(signs) is list
                and len(signs) == 2
                and all(type(value) is int and value in (-1, 0, 1) for value in signs)
                and type(deltas) is list
                and len(deltas) == 2
                and type(strong) is list
                and type(weak) is list
                and len(strong) == len(weak) == 4
                and all(type(value) is bool for value in strong)
                and all(type(value) is bool for value in weak)
                and all(not left or right for left, right in zip(strong, weak)),
                f"invalid visibility-v2 face metadata: {where}",
            )
            delta_values = [
                _finite_metric(value, f"{where}:face_delta") for value in deltas
            ]
            recipe = str(receipt["hypothesis"]).rsplit("+", 1)[-1]
            expected_axes = {
                "base": (False, False),
                "face_x": (True, False),
                "face_y": (False, True),
                "face_xy": (True, True),
            }[recipe]
            for axis, active in enumerate(expected_axes):
                sign = signs[axis]
                delta = delta_values[axis]
                if not active:
                    _require(
                        sign == 0 and delta == 0.0,
                        f"inactive recipe axis has extension metadata: {where}",
                    )
                    continue
                _require(
                    sign != 0 and FACE_EXTENSION_MIN_M <= delta <= FACE_EXTENSION_MAX_M,
                    f"active recipe axis has an invalid extension: {where}",
                )
                negative, positive = ((0, 1), (2, 3))[axis]
                unseen = positive if sign > 0 else negative
                anchor = negative if sign > 0 else positive
                _require(
                    strong[anchor] and not weak[unseen],
                    f"face extension lacks visible-anchor/unseen-face evidence: {where}",
                )

        present = [receipt[name] is not None for name in metric_names]
        _require(all(present) or not any(present), f"partial receipt metrics: {where}")
        dominates = False
        if all(present):
            metrics = {
                name: _finite_metric(receipt[name], f"{where}:{name}")
                for name in metric_names
            }
            for name in metric_names[:6]:
                _require(0.0 <= metrics[name] <= 1.0, f"metric outside [0,1]: {where}")
            dominates = (
                metrics["candidate_projection_iou"] >= metrics["native_projection_iou"]
                and metrics["candidate_support"] >= metrics["native_support"]
                and metrics["candidate_free_space"] <= metrics["native_free_space"]
                and (
                    metrics["candidate_projection_iou"]
                    > metrics["native_projection_iou"] + 1e-9
                    or metrics["candidate_support"] > metrics["native_support"] + 1e-9
                    or metrics["candidate_free_space"]
                    < metrics["native_free_space"] - 1e-9
                )
            )
            _require(candidate is not None, f"receipt metrics lack candidate: {where}")
        else:
            _require(candidate is None, f"candidate lacks receipt metrics: {where}")

        if mask[index]:
            _require(
                receipt["reason"] == "loo_improved"
                and candidate is not None
                and dominates
                and np.array_equal(candidate, counterfactual[index]),
                f"selected mask row is not a complete LOO receipt: {where}",
            )
        else:
            _require(not dominates, f"unselected receipt claims LOO dominance: {where}")
        hypotheses.append(
            None if receipt["hypothesis"] is None else str(receipt["hypothesis"])
        )

    return R2Sidecar(
        path_sha256=_sha256_file(source),
        native_corners=native_corners,
        native_scores=native_scores,
        stable_ids=stable_ids,
        counterfactual_corners=counterfactual,
        would_replace_mask=mask,
        hypotheses=tuple(hypotheses),
    )


@dataclass(frozen=True)
class PreparedScene:
    scene_id: str
    prediction_path: Path
    sidecar_path: Path
    prediction_sha256: str
    sidecar_sha256: str
    native: NativePrediction
    output_corners: np.ndarray
    mask: np.ndarray
    stable_ids: np.ndarray
    hypotheses: tuple[str | None, ...]


def _score_bits(values: Sequence[float]) -> str:
    return _sha256_bytes(b"".join(struct.pack(">d", value) for value in values))


def _label_order_sha256(values: Sequence[int]) -> str:
    encoded = json.dumps(list(values), separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def _write_bytes_create_only(path: Path, payload: bytes, label: str) -> Path:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing existing {label}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Make the inode read-only before publishing its create-only hard link,
        # so a post-link chmod failure cannot leave a visible partial result.
        os.chmod(temporary_name, 0o444)
        try:
            os.link(temporary_name, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing existing {label}: {path}") from error
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    return path


def _prediction_bytes(
    labels: Sequence[int], corners: np.ndarray, scores: Sequence[float]
) -> bytes:
    geometry = np.asarray(corners)
    _require(
        geometry.shape == (len(labels), 8, 3)
        and geometry.dtype == np.float32
        and np.isfinite(geometry).all()
        and len(scores) == len(labels),
        "materialized prediction arrays are malformed",
    )
    payload = [[
        (
            int(labels[index]),
            np.array(geometry[index], dtype=np.float32, order="C", copy=True),
            float(scores[index]),
        )
        for index in range(len(labels))
    ]]
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def _same_scores(left: Sequence[float], right: Sequence[float]) -> bool:
    return len(left) == len(right) and all(
        struct.pack(">d", first) == struct.pack(">d", second)
        for first, second in zip(left, right)
    )


def _quarantine(root: Path) -> Path | None:
    if not root.exists() or root.is_symlink():
        return None
    candidate = root.with_name(f"{root.name}.failed.{os.getpid()}")
    suffix = 0
    while candidate.exists() or candidate.is_symlink():
        suffix += 1
        candidate = root.with_name(f"{root.name}.failed.{os.getpid()}.{suffix}")
    os.replace(root, candidate)
    return candidate


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _code_sha256() -> str:
    return _sha256_file(Path(__file__).resolve())


def materialize(args: argparse.Namespace) -> dict[str, object]:
    scene_list = _regular(Path(args.scene_list), "scene list")
    scenes = read_scene_list(scene_list)
    prediction_root = _directory(Path(args.prediction_root), "prediction root")
    diagnostics_root = _directory(Path(args.diagnostics_root), "diagnostics root")
    output_raw = Path(args.output_root)
    manifest_raw = Path(args.manifest)
    if output_raw.exists() or output_raw.is_symlink():
        raise FileExistsError(f"refusing existing output root: {output_raw}")
    if manifest_raw.exists() or manifest_raw.is_symlink():
        raise FileExistsError(f"refusing existing manifest: {manifest_raw}")
    output_root = output_raw.resolve()
    manifest_path = manifest_raw.resolve()
    for input_root in (prediction_root, diagnostics_root):
        _require(
            not _paths_overlap(output_root, input_root),
            "output root must be isolated from input roots",
        )
        _require(
            not _paths_overlap(manifest_path, input_root),
            "manifest must be isolated from input roots",
        )
    _require(
        manifest_path not in output_root.parents
        and manifest_path != output_root
        and output_root not in manifest_path.parents,
        "manifest must be outside the prediction output root",
    )

    prediction_paths = _exact_artifacts(
        prediction_root, scenes, PREDICTION_SUFFIX, "prediction root"
    )
    sidecar_paths = _exact_artifacts(
        diagnostics_root, scenes, SIDECAR_SUFFIX, "diagnostics root"
    )

    # Complete preflight occurs before creating any output path.
    prepared: list[PreparedScene] = []
    for scene in scenes:
        native = _load_prediction(prediction_paths[scene])
        sidecar = _load_sidecar(sidecar_paths[scene], native)
        output_corners = np.array(native.corners, dtype=np.float32, order="C", copy=True)
        output_corners[sidecar.would_replace_mask] = sidecar.counterfactual_corners[
            sidecar.would_replace_mask
        ]
        prepared.append(
            PreparedScene(
                scene_id=scene,
                prediction_path=prediction_paths[scene],
                sidecar_path=sidecar_paths[scene],
                prediction_sha256=_sha256_file(prediction_paths[scene]),
                sidecar_sha256=sidecar.path_sha256,
                native=native,
                output_corners=output_corners,
                mask=np.array(sidecar.would_replace_mask, copy=True),
                stable_ids=np.array(sidecar.stable_ids, copy=True),
                hypotheses=sidecar.hypotheses,
            )
        )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(exist_ok=False)
    manifest_written = False
    rows: list[dict[str, object]] = []
    try:
        for row in prepared:
            target = output_root / f"{row.scene_id}{PREDICTION_SUFFIX}"
            if np.any(row.mask):
                encoded = _prediction_bytes(
                    row.native.labels, row.output_corners, row.native.scores
                )
            else:
                # A no-op scene remains byte-identical, not merely value-equal.
                encoded = row.native.raw_bytes
            _write_bytes_create_only(target, encoded, "materialized prediction")
            reloaded = _load_prediction(target)
            _require(
                reloaded.labels == row.native.labels
                and _same_scores(reloaded.scores, row.native.scores)
                and len(reloaded.labels) == len(row.native.labels)
                and np.array_equal(reloaded.corners, row.output_corners),
                f"{row.scene_id}: output changed labels, scores, order, count, or expected geometry",
            )
            changed = np.flatnonzero(row.mask).astype(np.int64)
            rows.append(
                {
                    "scene_id": row.scene_id,
                    "native_rows": len(row.native.labels),
                    "replaced_rows": len(changed),
                    "replaced_native_indices": changed.tolist(),
                    "replaced_stable_ids": row.stable_ids[row.mask].tolist(),
                    "replaced_hypotheses": [
                        row.hypotheses[index] for index in changed.tolist()
                    ],
                    "native_prediction_sha256": row.prediction_sha256,
                    "sidecar_sha256": row.sidecar_sha256,
                    "output_prediction_sha256": _sha256_file(target),
                    "no_replacement_byte_identity": (
                        _sha256_file(target) == row.prediction_sha256
                        if not np.any(row.mask)
                        else None
                    ),
                    "label_order_sha256_before": _label_order_sha256(row.native.labels),
                    "label_order_sha256_after": _label_order_sha256(reloaded.labels),
                    "score_bits_sha256_before": _score_bits(row.native.scores),
                    "score_bits_sha256_after": _score_bits(reloaded.scores),
                    "labels_unchanged": True,
                    "scores_unchanged": True,
                    "row_order_unchanged": True,
                    "row_count_unchanged": True,
                }
            )

        for row in prepared:
            _require(
                _sha256_file(row.prediction_path) == row.prediction_sha256
                and _sha256_file(row.sidecar_path) == row.sidecar_sha256,
                f"{row.scene_id}: an input artifact changed during materialization",
            )
        manifest: dict[str, object] = {
            "schema": SCHEMA,
            "complete": True,
            "contract": "visibility-v2",
            "sidecar_schema": SIDECAR_SCHEMA,
            "geometry_only": True,
            "selection_policy": "would_replace_mask_only",
            "offline_counterfactual": True,
            "online_activation_authorized": False,
            "ground_truth_access": False,
            "evaluation_invoked": False,
            "training_invoked": False,
            "clip_access": False,
            "labels_unchanged": True,
            "scores_unchanged": True,
            "row_order_unchanged": True,
            "row_count_unchanged": True,
            "native_inputs_mutated": False,
            "scene_list": str(scene_list),
            "scene_list_sha256": _sha256_file(scene_list),
            "prediction_root": str(prediction_root),
            "diagnostics_root": str(diagnostics_root),
            "output_root": str(output_root),
            "scene_count": len(rows),
            "native_rows": sum(int(row["native_rows"]) for row in rows),
            "replaced_rows": sum(int(row["replaced_rows"]) for row in rows),
            "materializer_code_sha256": _code_sha256(),
            "scenes": rows,
        }
        encoded_manifest = (
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        _write_bytes_create_only(manifest_path, encoded_manifest, "manifest")
        manifest_written = True
        return manifest
    except BaseException:
        if not manifest_written:
            failed = _quarantine(output_root)
            if failed is not None:
                print(
                    f"Incomplete materialization quarantined at {failed}",
                    file=sys.stderr,
                )
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--scene-list", type=Path, required=True)
    value.add_argument("--prediction-root", type=Path, required=True)
    value.add_argument("--diagnostics-root", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--manifest", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    manifest = materialize(args)
    print(
        "OpenBox-SMOV R2 counterfactual materialized:",
        f"scenes={manifest['scene_count']}",
        f"rows={manifest['native_rows']}",
        f"replaced={manifest['replaced_rows']}",
    )
    print("Prediction root:", Path(args.output_root).resolve())
    print("Immutable manifest:", Path(args.manifest).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
