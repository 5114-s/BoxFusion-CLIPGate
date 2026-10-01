#!/usr/bin/env python3
"""Read-only identity audit for YiDu observer ablations.

The tool compares trusted local BoxFusion prediction artifacts from a frozen
``B0`` run and one observer-stage run.  The default remains the historical
strict mode: pickle containers are compared recursively and NumPy arrays and
scalars must have identical shapes, dtypes, and value bytes.

An explicit cross-run numerical-envelope mode is also available for canonical
``*_boxes.pkl`` prediction files.  It never re-matches or reorders prediction
rows: container structure, label sequence, row count, dtypes, and score rank
must remain exact.  Only same-index box corners, scores, and axis-aligned IoU
loss may vary within caller-supplied limits.  All three limits must be supplied
together; this utility deliberately has no permissive numerical defaults.
Non-prediction ``.pkl``, ``.npy``, and ``.npz`` artifacts stay bitwise-strict
in both modes.

Observer diagnostics are audited independently.  Every ``*_tracks.npz`` file
must explicitly contain all three observer-safety indicators:

* ``yidu_mutation_enabled``: Boolean scalar ``False``;
* ``yidu_applied_count``: integer scalar ``0``;
* ``yidu_applied``: Boolean array containing no ``True`` value.

Missing safety keys are failures, rather than being interpreted as false.

Prediction pickle files are executable serialization.  Only use this utility
with trusted, locally produced experiment artifacts.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import struct
import sys
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boxfusion.yidu_ablation import (  # noqa: E402
    YIDU_STAGE_MODULE_MATRIX,
    YIDU_STAGE_TO_PROFILE,
    resolve_yidu_stage,
)


SUPPORTED_SUFFIXES = frozenset({".pkl", ".npy", ".npz"})
DIAGNOSTIC_SUFFIX = "_tracks.npz"
_SCENE_PATTERN = re.compile(r"(scene\d{4}_\d{2})")
_PREDICTION_SUFFIX = "_boxes.pkl"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class NumericEnvelope:
    """Explicit limits for same-index cross-run prediction comparison."""

    max_corner_abs: float
    max_score_abs: float
    max_matched_iou_loss: float

    def validated(self) -> "NumericEnvelope":
        values = {
            "max_corner_abs": self.max_corner_abs,
            "max_score_abs": self.max_score_abs,
            "max_matched_iou_loss": self.max_matched_iou_loss,
        }
        normalized: dict[str, float] = {}
        for name, value in values.items():
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
                raise ValueError(f"{name} must be a finite scalar")
            number = float(value)
            if not np.isfinite(number) or number < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            normalized[name] = number
        if normalized["max_matched_iou_loss"] > 1.0:
            raise ValueError("max_matched_iou_loss must not exceed 1")
        return NumericEnvelope(**normalized)


@dataclass
class _NumericSummaryBuilder:
    """Bounded-size aggregate statistics for canonical prediction rows."""

    prediction_files: int = 0
    prediction_rows: int = 0
    changed_prediction_rows: int = 0
    corner_abs_values: list[float] = field(default_factory=list)
    score_abs_values: list[float] = field(default_factory=list)
    matched_iou_losses: list[float] = field(default_factory=list)

    def add_file(self) -> None:
        self.prediction_files += 1

    def add_row(
        self,
        corner_abs: np.ndarray,
        score_abs: float,
        matched_iou_loss: float,
    ) -> None:
        flattened = np.asarray(corner_abs, dtype=np.float64).reshape(-1)
        self.prediction_rows += 1
        if (
            bool(np.any(flattened != 0.0))
            or score_abs != 0.0
            or matched_iou_loss != 0.0
        ):
            self.changed_prediction_rows += 1
        self.corner_abs_values.extend(float(value) for value in flattened)
        self.score_abs_values.append(float(score_abs))
        self.matched_iou_losses.append(float(matched_iou_loss))

    @staticmethod
    def _distribution(values: Sequence[float]) -> dict[str, float | int]:
        array = np.asarray(values, dtype=np.float64)
        if not len(array):
            return {
                "count": 0,
                "changed_count": 0,
                "max": 0.0,
                "mean": 0.0,
                "p99": 0.0,
            }
        return {
            "count": int(len(array)),
            "changed_count": int(np.count_nonzero(array)),
            "max": float(np.max(array)),
            "mean": float(np.mean(array)),
            "p99": float(np.quantile(array, 0.99)),
        }

    def to_dict(self) -> dict[str, Any]:
        iou_loss = self._distribution(self.matched_iou_losses)
        return {
            "canonical_prediction_files": int(self.prediction_files),
            "prediction_rows": int(self.prediction_rows),
            "changed_prediction_rows": int(self.changed_prediction_rows),
            "corner_abs": self._distribution(self.corner_abs_values),
            "score_abs": self._distribution(self.score_abs_values),
            "same_index_iou_loss": iou_loss,
            "same_index_iou_min": (
                1.0 - float(iou_loss["max"])
                if int(iou_loss["count"])
                else 1.0
            ),
            "ordering": "same_index_no_rematching_and_exact_score_rank",
        }


@dataclass(frozen=True)
class AuditIssue:
    """One deterministic identity or observer-safety failure."""

    kind: str
    relative_path: str
    object_path: str
    message: str


@dataclass(frozen=True)
class AuditReport:
    """Machine-readable result of a complete identity audit."""

    comparison_mode: str
    numeric_envelope: NumericEnvelope | None
    numeric_summary: Mapping[str, Any]
    require_zero_write: bool
    baseline_root: str
    observer_root: str
    diagnostics_root: str
    prediction_files: int
    prediction_scenes: tuple[str, ...]
    diagnostic_files: int
    diagnostic_scenes: tuple[str, ...]
    issues: tuple[AuditIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "boxfusion.yidu.identity_audit.v1",
            "ok": self.ok,
            "comparison_mode": self.comparison_mode,
            "numeric_envelope": (
                None
                if self.numeric_envelope is None
                else asdict(self.numeric_envelope)
            ),
            "numeric_summary": dict(self.numeric_summary),
            "require_zero_write": self.require_zero_write,
            "baseline_root": self.baseline_root,
            "observer_root": self.observer_root,
            "diagnostics_root": self.diagnostics_root,
            "prediction_files": self.prediction_files,
            "prediction_scenes": list(self.prediction_scenes),
            "diagnostic_files": self.diagnostic_files,
            "diagnostic_scenes": list(self.diagnostic_scenes),
            "issues": [asdict(issue) for issue in self.issues],
        }


def _scene_id(path: Path) -> str:
    match = _SCENE_PATTERN.search(path.as_posix())
    return match.group(1) if match is not None else path.stem


def _supported_files(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"artifact root does not exist: {root}")
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    }


def _load_artifact(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        with path.open("rb") as handle:
            return pickle.load(handle)  # noqa: S301 - trusted local artifacts
    if suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as archive:
            return {
                name: np.array(archive[name], copy=True)
                for name in archive.files
            }
    raise ValueError(f"unsupported artifact suffix: {path}")


def _join_object_path(parent: str, child: str) -> str:
    if not parent:
        return child
    if child.startswith("["):
        return f"{parent}{child}"
    return f"{parent}.{child}"


def _array_bytes(array: np.ndarray) -> bytes:
    """Return logical C-order bytes without depending on memory strides."""

    return np.ascontiguousarray(array).tobytes(order="C")


def _compare_values(
    baseline: Any,
    observer: Any,
    *,
    object_path: str = "$",
) -> str | None:
    """Return the first strict structural mismatch, or ``None``.

    Array values use byte identity so ``-0.0``, NaN payloads, and integer
    representations cannot silently compare equal.
    """

    if isinstance(baseline, np.ndarray) or isinstance(observer, np.ndarray):
        if not isinstance(baseline, np.ndarray) or not isinstance(
            observer, np.ndarray
        ):
            return (
                f"{object_path}: type mismatch "
                f"{type(baseline).__name__} != {type(observer).__name__}"
            )
        if baseline.shape != observer.shape:
            return (
                f"{object_path}: shape mismatch "
                f"{baseline.shape!r} != {observer.shape!r}"
            )
        if baseline.dtype != observer.dtype:
            return (
                f"{object_path}: dtype mismatch "
                f"{baseline.dtype!s} != {observer.dtype!s}"
            )
        if baseline.dtype.hasobject:
            for index, (left, right) in enumerate(
                zip(
                    baseline.reshape(-1, order="C"),
                    observer.reshape(-1, order="C"),
                )
            ):
                mismatch = _compare_values(
                    left,
                    right,
                    object_path=_join_object_path(
                        object_path, f"[flat:{index}]"
                    ),
                )
                if mismatch is not None:
                    return mismatch
            return None
        if _array_bytes(baseline) != _array_bytes(observer):
            return f"{object_path}: array value bytes differ"
        return None

    if isinstance(baseline, np.generic) or isinstance(observer, np.generic):
        if not isinstance(baseline, np.generic) or not isinstance(
            observer, np.generic
        ):
            return (
                f"{object_path}: type mismatch "
                f"{type(baseline).__name__} != {type(observer).__name__}"
            )
        if baseline.dtype != observer.dtype:
            return (
                f"{object_path}: scalar dtype mismatch "
                f"{baseline.dtype!s} != {observer.dtype!s}"
            )
        if baseline.tobytes() != observer.tobytes():
            return f"{object_path}: scalar value bytes differ"
        return None

    if type(baseline) is not type(observer):
        return (
            f"{object_path}: type mismatch "
            f"{type(baseline).__name__} != {type(observer).__name__}"
        )

    if isinstance(baseline, Mapping):
        baseline_keys = tuple(baseline.keys())
        observer_keys = tuple(observer.keys())
        try:
            same_keys = set(baseline_keys) == set(observer_keys)
        except TypeError:
            same_keys = baseline_keys == observer_keys
        if not same_keys:
            return (
                f"{object_path}: mapping keys differ "
                f"{baseline_keys!r} != {observer_keys!r}"
            )
        for key in baseline_keys:
            mismatch = _compare_values(
                baseline[key],
                observer[key],
                object_path=_join_object_path(object_path, f"[{key!r}]"),
            )
            if mismatch is not None:
                return mismatch
        return None

    if isinstance(baseline, (list, tuple)):
        if len(baseline) != len(observer):
            return (
                f"{object_path}: sequence length mismatch "
                f"{len(baseline)} != {len(observer)}"
            )
        for index, (left, right) in enumerate(zip(baseline, observer)):
            mismatch = _compare_values(
                left,
                right,
                object_path=_join_object_path(object_path, f"[{index}]"),
            )
            if mismatch is not None:
                return mismatch
        return None

    if is_dataclass(baseline) and not isinstance(baseline, type):
        for field in fields(baseline):
            mismatch = _compare_values(
                getattr(baseline, field.name),
                getattr(observer, field.name),
                object_path=_join_object_path(object_path, field.name),
            )
            if mismatch is not None:
                return mismatch
        return None

    if isinstance(baseline, float):
        if struct.pack(">d", baseline) != struct.pack(">d", observer):
            return f"{object_path}: float value bytes differ"
        return None

    if isinstance(baseline, complex):
        packed_left = struct.pack(">dd", baseline.real, baseline.imag)
        packed_right = struct.pack(">dd", observer.real, observer.imag)
        if packed_left != packed_right:
            return f"{object_path}: complex value bytes differ"
        return None

    if isinstance(
        baseline,
        (str, bytes, bytearray, memoryview, int, bool, type(None), Path),
    ):
        if baseline != observer:
            return f"{object_path}: value differs"
        return None

    if hasattr(baseline, "__dict__"):
        return _compare_values(
            vars(baseline),
            vars(observer),
            object_path=_join_object_path(object_path, "__dict__"),
        )

    try:
        equal = baseline == observer
    except Exception as exc:  # pragma: no cover - defensive unsupported type
        return (
            f"{object_path}: unsupported comparison for "
            f"{type(baseline).__name__}: {exc}"
        )
    if not isinstance(equal, (bool, np.bool_)) or not bool(equal):
        return f"{object_path}: value differs"
    return None


def _resolve_numeric_envelope(
    *,
    max_corner_abs: float | None,
    max_score_abs: float | None,
    max_matched_iou_loss: float | None,
) -> NumericEnvelope | None:
    values = (max_corner_abs, max_score_abs, max_matched_iou_loss)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(
            "cross-run numerical mode requires max_corner_abs, "
            "max_score_abs, and max_matched_iou_loss together"
        )
    return NumericEnvelope(
        max_corner_abs=max_corner_abs,
        max_score_abs=max_score_abs,
        max_matched_iou_loss=max_matched_iou_loss,
    ).validated()


def _same_index_aabb_iou(
    baseline_corners: np.ndarray,
    observer_corners: np.ndarray,
) -> float:
    """Return axis-aligned IoU for two finite ``[8, 3]`` corner arrays."""

    baseline_lower = np.min(baseline_corners.astype(np.float64), axis=0)
    baseline_upper = np.max(baseline_corners.astype(np.float64), axis=0)
    observer_lower = np.min(observer_corners.astype(np.float64), axis=0)
    observer_upper = np.max(observer_corners.astype(np.float64), axis=0)
    baseline_extent = baseline_upper - baseline_lower
    observer_extent = observer_upper - observer_lower
    if np.any(baseline_extent <= 0.0) or np.any(observer_extent <= 0.0):
        raise ValueError("prediction corners must span positive AABB extents")
    intersection_extent = np.maximum(
        np.minimum(baseline_upper, observer_upper)
        - np.maximum(baseline_lower, observer_lower),
        0.0,
    )
    intersection = float(np.prod(intersection_extent))
    baseline_volume = float(np.prod(baseline_extent))
    observer_volume = float(np.prod(observer_extent))
    union = baseline_volume + observer_volume - intersection
    if not np.isfinite(union) or union <= 0.0:
        raise ValueError("prediction corner AABB union must be positive")
    return float(np.clip(intersection / union, 0.0, 1.0))


def _prediction_structure_issue(
    relative_path: str,
    object_path: str,
    message: str,
    *,
    kind: str = "prediction_structure_mismatch",
) -> list[AuditIssue]:
    return [
        AuditIssue(
            kind=kind,
            relative_path=relative_path,
            object_path=object_path,
            message=message,
        )
    ]


def _compare_box_predictions_with_envelope(
    baseline: Any,
    observer: Any,
    *,
    relative_path: str,
    summary: _NumericSummaryBuilder,
    envelope: NumericEnvelope | None,
) -> list[AuditIssue]:
    """Compare canonical BoxFusion rows without any cross-row re-matching.

    When ``envelope`` is ``None`` this function only gathers numerical
    statistics; the caller still applies the legacy recursive bitwise
    comparison.  Supplying an envelope makes this function the authoritative
    comparison for canonical prediction pickles.
    """

    summary.add_file()
    if type(baseline) is not type(observer):
        return _prediction_structure_issue(
            relative_path,
            "$",
            "top-level prediction container types differ",
        )
    if not isinstance(baseline, (list, tuple)):
        return _prediction_structure_issue(
            relative_path,
            "$",
            "canonical prediction payload must be a list or tuple of batches",
        )
    if len(baseline) != len(observer):
        return _prediction_structure_issue(
            relative_path,
            "$",
            f"batch count differs: {len(baseline)} != {len(observer)}",
        )

    issues: list[AuditIssue] = []
    corner_violation: tuple[float, str] | None = None
    score_violation: tuple[float, str] | None = None
    iou_violation: tuple[float, str] | None = None
    for batch_index, (baseline_batch, observer_batch) in enumerate(
        zip(baseline, observer)
    ):
        batch_path = f"$[{batch_index}]"
        if type(baseline_batch) is not type(observer_batch):
            return _prediction_structure_issue(
                relative_path,
                batch_path,
                "prediction batch container types differ",
            )
        if not isinstance(baseline_batch, (list, tuple)):
            return _prediction_structure_issue(
                relative_path,
                batch_path,
                "prediction batch must be a list or tuple",
            )
        if len(baseline_batch) != len(observer_batch):
            return _prediction_structure_issue(
                relative_path,
                batch_path,
                "prediction row count differs: "
                f"{len(baseline_batch)} != {len(observer_batch)}",
            )

        baseline_scores: list[float] = []
        observer_scores: list[float] = []
        for row_index, (baseline_row, observer_row) in enumerate(
            zip(baseline_batch, observer_batch)
        ):
            row_path = f"{batch_path}[{row_index}]"
            if type(baseline_row) is not type(observer_row):
                return _prediction_structure_issue(
                    relative_path,
                    row_path,
                    "prediction row container types differ",
                )
            if not isinstance(baseline_row, (list, tuple)):
                return _prediction_structure_issue(
                    relative_path,
                    row_path,
                    "prediction row must be a list or tuple",
                )
            if len(baseline_row) != 3 or len(observer_row) != 3:
                return _prediction_structure_issue(
                    relative_path,
                    row_path,
                    "prediction row must contain label, corners, and score",
                )

            label_mismatch = _compare_values(
                baseline_row[0],
                observer_row[0],
                object_path=f"{row_path}[0]",
            )
            if label_mismatch is not None:
                _, _, message = label_mismatch.partition(": ")
                return _prediction_structure_issue(
                    relative_path,
                    f"{row_path}[0]",
                    message or label_mismatch,
                    kind="prediction_label_or_order_mismatch",
                )

            baseline_corners = baseline_row[1]
            observer_corners = observer_row[1]
            if not isinstance(baseline_corners, np.ndarray) or not isinstance(
                observer_corners, np.ndarray
            ):
                return _prediction_structure_issue(
                    relative_path,
                    f"{row_path}[1]",
                    "prediction corners must be NumPy arrays",
                )
            if baseline_corners.shape != observer_corners.shape:
                return _prediction_structure_issue(
                    relative_path,
                    f"{row_path}[1]",
                    "corner shape differs: "
                    f"{baseline_corners.shape!r} != "
                    f"{observer_corners.shape!r}",
                )
            if baseline_corners.shape != (8, 3):
                return _prediction_structure_issue(
                    relative_path,
                    f"{row_path}[1]",
                    "canonical prediction corners must have shape [8, 3]",
                )
            if baseline_corners.dtype != observer_corners.dtype:
                return _prediction_structure_issue(
                    relative_path,
                    f"{row_path}[1]",
                    "corner dtype differs: "
                    f"{baseline_corners.dtype!s} != "
                    f"{observer_corners.dtype!s}",
                )
            if (
                not np.issubdtype(baseline_corners.dtype, np.number)
                or not np.isfinite(baseline_corners).all()
                or not np.isfinite(observer_corners).all()
            ):
                return _prediction_structure_issue(
                    relative_path,
                    f"{row_path}[1]",
                    "prediction corners must be finite numeric values",
                )

            baseline_score = baseline_row[2]
            observer_score = observer_row[2]
            if type(baseline_score) is not type(observer_score):
                return _prediction_structure_issue(
                    relative_path,
                    f"{row_path}[2]",
                    "score scalar types differ",
                )
            if isinstance(baseline_score, (bool, np.bool_)) or not isinstance(
                baseline_score, (float, np.floating)
            ):
                return _prediction_structure_issue(
                    relative_path,
                    f"{row_path}[2]",
                    "prediction score must be a floating-point scalar",
                )
            baseline_score_value = float(baseline_score)
            observer_score_value = float(observer_score)
            if not np.isfinite(
                [baseline_score_value, observer_score_value]
            ).all():
                return _prediction_structure_issue(
                    relative_path,
                    f"{row_path}[2]",
                    "prediction scores must be finite",
                )
            baseline_scores.append(baseline_score_value)
            observer_scores.append(observer_score_value)

            corner_abs = np.abs(
                baseline_corners.astype(np.float64)
                - observer_corners.astype(np.float64)
            )
            score_abs = abs(baseline_score_value - observer_score_value)
            try:
                matched_iou = _same_index_aabb_iou(
                    baseline_corners, observer_corners
                )
            except ValueError as exc:
                return _prediction_structure_issue(
                    relative_path,
                    f"{row_path}[1]",
                    str(exc),
                    kind="invalid_prediction_geometry",
                )
            matched_iou_loss = max(0.0, 1.0 - matched_iou)
            summary.add_row(corner_abs, score_abs, matched_iou_loss)

            if envelope is not None:
                corner_max = float(np.max(corner_abs, initial=0.0))
                if (
                    corner_max > envelope.max_corner_abs
                    and (
                        corner_violation is None
                        or corner_max > corner_violation[0]
                    )
                ):
                    corner_violation = (corner_max, f"{row_path}[1]")
                if (
                    score_abs > envelope.max_score_abs
                    and (
                        score_violation is None
                        or score_abs > score_violation[0]
                    )
                ):
                    score_violation = (score_abs, f"{row_path}[2]")
                if (
                    matched_iou_loss > envelope.max_matched_iou_loss
                    and (
                        iou_violation is None
                        or matched_iou_loss > iou_violation[0]
                    )
                ):
                    iou_violation = (matched_iou_loss, f"{row_path}[1]")

        baseline_rank = np.argsort(
            -np.asarray(baseline_scores, dtype=np.float64), kind="stable"
        )
        observer_rank = np.argsort(
            -np.asarray(observer_scores, dtype=np.float64), kind="stable"
        )
        if not np.array_equal(baseline_rank, observer_rank):
            return _prediction_structure_issue(
                relative_path,
                batch_path,
                "stable descending score rank differs; prediction ordering "
                "must remain exact",
                kind="prediction_label_or_order_mismatch",
            )

    if envelope is None:
        return issues
    for metric, violation, limit in (
        ("corner absolute difference", corner_violation, envelope.max_corner_abs),
        ("score absolute difference", score_violation, envelope.max_score_abs),
        (
            "same-index matched IoU loss",
            iou_violation,
            envelope.max_matched_iou_loss,
        ),
    ):
        if violation is None:
            continue
        actual, object_path = violation
        issues.append(
            AuditIssue(
                kind="prediction_numeric_envelope_exceeded",
                relative_path=relative_path,
                object_path=object_path,
                message=f"{metric} {actual:.12g} exceeds {limit:.12g}",
            )
        )
    return issues


def _scalar(
    payload: Mapping[str, np.ndarray],
    key: str,
) -> tuple[Any | None, str | None]:
    if key not in payload:
        return None, f"missing required diagnostic key {key!r}"
    value = np.asarray(payload[key])
    if value.size != 1:
        return None, (
            f"diagnostic key {key!r} must contain exactly one value; "
            f"shape={value.shape!r}"
        )
    return value.reshape(()).item(), None


def _audit_diagnostic(
    path: Path,
    relative_path: str,
    *,
    expected_stage: str | None,
    require_zero_write: bool,
) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    try:
        with np.load(path, allow_pickle=False) as archive:
            payload = {
                name: np.array(archive[name], copy=True)
                for name in archive.files
            }
    except Exception as exc:
        return [
            AuditIssue(
                kind="diagnostic_load_error",
                relative_path=relative_path,
                object_path="$",
                message=str(exc),
            )
        ]

    if expected_stage is not None:
        expected_profile = YIDU_STAGE_TO_PROFILE[expected_stage]
        expected_modules = dict(YIDU_STAGE_MODULE_MATRIX[expected_stage])
        for key, expected in (
            ("yidu_stage", expected_stage),
            ("yidu_profile", expected_profile),
        ):
            actual, error = _scalar(payload, key)
            if error is not None:
                issues.append(
                    AuditIssue(
                        "missing_or_invalid_stage_key",
                        relative_path,
                        f"$.{key}",
                        error,
                    )
                )
            elif str(actual) != expected:
                issues.append(
                    AuditIssue(
                        "stage_contract_mismatch",
                        relative_path,
                        f"$.{key}",
                        f"expected {expected!r}, got {actual!r}",
                    )
                )
        modules_text, error = _scalar(payload, "yidu_modules_json")
        if error is not None:
            issues.append(
                AuditIssue(
                    "missing_or_invalid_stage_key",
                    relative_path,
                    "$.yidu_modules_json",
                    error,
                )
            )
        else:
            try:
                modules = json.loads(str(modules_text))
            except (TypeError, json.JSONDecodeError) as exc:
                issues.append(
                    AuditIssue(
                        "stage_contract_mismatch",
                        relative_path,
                        "$.yidu_modules_json",
                        f"invalid JSON module matrix: {exc}",
                    )
                )
            else:
                if modules != expected_modules:
                    issues.append(
                        AuditIssue(
                            "stage_contract_mismatch",
                            relative_path,
                            "$.yidu_modules_json",
                            "module matrix disagrees with the canonical "
                            f"{expected_stage} stage",
                        )
                    )

    mutation, error = _scalar(payload, "yidu_mutation_enabled")
    if error is not None:
        issues.append(
            AuditIssue(
                "missing_or_invalid_safety_key",
                relative_path,
                "$.yidu_mutation_enabled",
                error,
            )
        )
    else:
        mutation_array = np.asarray(payload["yidu_mutation_enabled"])
        if mutation_array.dtype.kind != "b":
            issues.append(
                AuditIssue(
                    "invalid_safety_dtype",
                    relative_path,
                    "$.yidu_mutation_enabled",
                    "must have Boolean dtype",
                )
            )
        elif bool(mutation):
            issues.append(
                AuditIssue(
                    "observer_mutation_enabled",
                    relative_path,
                    "$.yidu_mutation_enabled",
                    "observer diagnostics report mutation_enabled=true",
                )
            )

    applied_count, error = _scalar(payload, "yidu_applied_count")
    if error is not None:
        issues.append(
            AuditIssue(
                "missing_or_invalid_safety_key",
                relative_path,
                "$.yidu_applied_count",
                error,
            )
        )
    else:
        count_array = np.asarray(payload["yidu_applied_count"])
        if count_array.dtype.kind not in {"i", "u"}:
            issues.append(
                AuditIssue(
                    "invalid_safety_dtype",
                    relative_path,
                    "$.yidu_applied_count",
                    "must have integer dtype",
                )
            )
        elif int(applied_count) != 0:
            issues.append(
                AuditIssue(
                    "observer_applied",
                    relative_path,
                    "$.yidu_applied_count",
                    f"observer diagnostics report applied_count={applied_count}",
                )
            )

    if "yidu_applied" not in payload:
        issues.append(
            AuditIssue(
                "missing_or_invalid_safety_key",
                relative_path,
                "$.yidu_applied",
                "missing required diagnostic key 'yidu_applied'",
            )
        )
    else:
        applied = np.asarray(payload["yidu_applied"])
        if applied.dtype.kind != "b":
            issues.append(
                AuditIssue(
                    "invalid_safety_dtype",
                    relative_path,
                    "$.yidu_applied",
                    "must have Boolean dtype",
                )
            )
        elif bool(np.any(applied)):
            true_count = int(np.count_nonzero(applied))
            issues.append(
                AuditIssue(
                    "observer_applied",
                    relative_path,
                    "$.yidu_applied",
                    f"observer diagnostics contain {true_count} applied row(s)",
                )
            )

    if require_zero_write:
        for key in (
            "yidu_zero_write_check_enabled",
            "yidu_zero_write_verified",
        ):
            value, error = _scalar(payload, key)
            if error is not None:
                issues.append(
                    AuditIssue(
                        "missing_or_invalid_zero_write_key",
                        relative_path,
                        f"$.{key}",
                        error,
                    )
                )
                continue
            array = np.asarray(payload[key])
            if array.dtype.kind != "b":
                issues.append(
                    AuditIssue(
                        "invalid_zero_write_dtype",
                        relative_path,
                        f"$.{key}",
                        "must have Boolean dtype",
                    )
                )
            elif not bool(value):
                issues.append(
                    AuditIssue(
                        "observer_zero_write_unverified",
                        relative_path,
                        f"$.{key}",
                        f"{key} must be true",
                    )
                )

        hashes: dict[str, str] = {}
        for key in (
            "yidu_zero_write_pre_sha256",
            "yidu_zero_write_post_sha256",
        ):
            value, error = _scalar(payload, key)
            if error is not None:
                issues.append(
                    AuditIssue(
                        "missing_or_invalid_zero_write_key",
                        relative_path,
                        f"$.{key}",
                        error,
                    )
                )
                continue
            text = str(value)
            hashes[key] = text
            if _SHA256_PATTERN.fullmatch(text) is None:
                issues.append(
                    AuditIssue(
                        "invalid_zero_write_hash",
                        relative_path,
                        f"$.{key}",
                        "must contain one lowercase SHA256 digest",
                    )
                )
        if (
            len(hashes) == 2
            and hashes["yidu_zero_write_pre_sha256"]
            != hashes["yidu_zero_write_post_sha256"]
        ):
            issues.append(
                AuditIssue(
                    "observer_zero_write_hash_mismatch",
                    relative_path,
                    "$.yidu_zero_write_post_sha256",
                    "observer pre/post output hashes differ",
                )
            )

        for key, require_nonempty in (
            ("yidu_zero_write_array_names", True),
            ("yidu_zero_write_changed_fields", False),
        ):
            if key not in payload:
                issues.append(
                    AuditIssue(
                        "missing_or_invalid_zero_write_key",
                        relative_path,
                        f"$.{key}",
                        f"missing required diagnostic key {key!r}",
                    )
                )
                continue
            values = np.asarray(payload[key])
            if values.dtype.kind not in {"U", "S"}:
                issues.append(
                    AuditIssue(
                        "invalid_zero_write_dtype",
                        relative_path,
                        f"$.{key}",
                        "must have string dtype",
                    )
                )
            elif require_nonempty and values.size == 0:
                issues.append(
                    AuditIssue(
                        "observer_zero_write_unverified",
                        relative_path,
                        f"$.{key}",
                        "audited array list must not be empty",
                    )
                )
            elif not require_nonempty and values.size != 0:
                issues.append(
                    AuditIssue(
                        "observer_zero_write_changed",
                        relative_path,
                        f"$.{key}",
                        "observer reports changed output field(s): "
                        + ", ".join(str(value) for value in values.reshape(-1)),
                    )
                )

    return issues


def audit_identity(
    baseline_root: Path,
    observer_root: Path,
    diagnostics_root: Path,
    *,
    expected_stage: str | None = None,
    max_corner_abs: float | None = None,
    max_score_abs: float | None = None,
    max_matched_iou_loss: float | None = None,
    require_zero_write: bool = False,
) -> AuditReport:
    """Audit predictions and observer diagnostics without modifying artifacts."""

    numeric_envelope = _resolve_numeric_envelope(
        max_corner_abs=max_corner_abs,
        max_score_abs=max_score_abs,
        max_matched_iou_loss=max_matched_iou_loss,
    )
    comparison_mode = (
        "strict_bitwise"
        if numeric_envelope is None
        else "explicit_numeric_envelope"
    )
    numeric_summary = _NumericSummaryBuilder()
    canonical_stage = (
        None
        if expected_stage is None
        else resolve_yidu_stage(expected_stage)
    )
    if canonical_stage == "B0":
        raise ValueError("identity audit expects an A1-A6 observer stage")

    baseline_root = Path(baseline_root).resolve()
    observer_root = Path(observer_root).resolve()
    diagnostics_root = Path(diagnostics_root).resolve()

    baseline_files = _supported_files(baseline_root)
    observer_files = _supported_files(observer_root)
    issues: list[AuditIssue] = []

    if not baseline_files:
        issues.append(
            AuditIssue(
                "missing_baseline_predictions",
                ".",
                "$",
                "B0 root contains no supported .pkl/.npy/.npz artifacts",
            )
        )

    baseline_names = set(baseline_files)
    observer_names = set(observer_files)
    for relative_path in sorted(baseline_names - observer_names):
        issues.append(
            AuditIssue(
                "missing_observer_file",
                relative_path,
                "$",
                "artifact exists in B0 but is missing from observer output",
            )
        )
    for relative_path in sorted(observer_names - baseline_names):
        issues.append(
            AuditIssue(
                "extra_observer_file",
                relative_path,
                "$",
                "artifact exists in observer output but not in B0",
            )
        )

    for relative_path in sorted(baseline_names & observer_names):
        try:
            baseline_value = _load_artifact(baseline_files[relative_path])
            observer_value = _load_artifact(observer_files[relative_path])
        except Exception as exc:
            issues.append(
                AuditIssue(
                    "artifact_load_error",
                    relative_path,
                    "$",
                    str(exc),
                )
            )
            continue
        is_canonical_prediction = relative_path.endswith(_PREDICTION_SUFFIX)
        envelope_issues: list[AuditIssue] = []
        if is_canonical_prediction:
            envelope_issues = _compare_box_predictions_with_envelope(
                baseline_value,
                observer_value,
                relative_path=relative_path,
                summary=numeric_summary,
                envelope=numeric_envelope,
            )
        if numeric_envelope is not None and is_canonical_prediction:
            issues.extend(envelope_issues)
        else:
            mismatch = _compare_values(baseline_value, observer_value)
            if mismatch is not None:
                object_path, _, message = mismatch.partition(": ")
                issues.append(
                    AuditIssue(
                        "prediction_mismatch",
                        relative_path,
                        object_path,
                        message or mismatch,
                    )
                )

    if not diagnostics_root.is_dir():
        issues.append(
            AuditIssue(
                "missing_diagnostics_root",
                ".",
                "$",
                f"diagnostics root does not exist: {diagnostics_root}",
            )
        )
        diagnostic_files: dict[str, Path] = {}
    else:
        diagnostic_files = {
            path.relative_to(diagnostics_root).as_posix(): path
            for path in sorted(diagnostics_root.rglob(f"*{DIAGNOSTIC_SUFFIX}"))
            if path.is_file()
        }
        if not diagnostic_files:
            issues.append(
                AuditIssue(
                    "missing_diagnostics",
                    ".",
                    "$",
                    f"no *{DIAGNOSTIC_SUFFIX} files found",
                )
            )
        for relative_path, path in diagnostic_files.items():
            issues.extend(
                _audit_diagnostic(
                    path,
                    relative_path,
                    expected_stage=canonical_stage,
                    require_zero_write=bool(require_zero_write),
                )
            )

    prediction_scenes = tuple(
        sorted({_scene_id(Path(name)) for name in baseline_files})
    )
    diagnostic_scenes = tuple(
        sorted({_scene_id(Path(name)) for name in diagnostic_files})
    )
    missing_diagnostic_scenes = sorted(
        set(prediction_scenes) - set(diagnostic_scenes)
    )
    extra_diagnostic_scenes = sorted(
        set(diagnostic_scenes) - set(prediction_scenes)
    )
    for scene_id in missing_diagnostic_scenes:
        issues.append(
            AuditIssue(
                "missing_scene_diagnostic",
                scene_id,
                "$",
                "prediction scene has no observer diagnostic",
            )
        )
    for scene_id in extra_diagnostic_scenes:
        issues.append(
            AuditIssue(
                "extra_scene_diagnostic",
                scene_id,
                "$",
                "observer diagnostic has no matching B0 prediction scene",
            )
        )

    return AuditReport(
        comparison_mode=comparison_mode,
        numeric_envelope=numeric_envelope,
        numeric_summary=numeric_summary.to_dict(),
        require_zero_write=bool(require_zero_write),
        baseline_root=str(baseline_root),
        observer_root=str(observer_root),
        diagnostics_root=str(diagnostics_root),
        prediction_files=len(baseline_files),
        prediction_scenes=prediction_scenes,
        diagnostic_files=len(diagnostic_files),
        diagnostic_scenes=diagnostic_scenes,
        issues=tuple(issues),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen B0 and YiDu observer predictions, then audit "
            "explicit observer-only safety flags.  The default is bitwise "
            "strict; numerical limits are opt-in and must be supplied together."
        )
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        required=True,
        help="Frozen B0 prediction directory.",
    )
    parser.add_argument(
        "--observer-root",
        type=Path,
        required=True,
        help="Observer-stage prediction directory.",
    )
    parser.add_argument(
        "--diagnostics-root",
        type=Path,
        required=True,
        help="Observer-stage diagnostics directory containing *_tracks.npz.",
    )
    parser.add_argument(
        "--expected-stage",
        choices=("A1", "A2", "A3", "A4", "A5", "A6"),
        required=True,
        help=(
            "Canonical observer stage. Diagnostics must contain its exact "
            "profile and cumulative one-module-at-a-time matrix."
        ),
    )
    parser.add_argument(
        "--max-corner-abs",
        type=float,
        default=None,
        help=(
            "Explicit maximum same-index corner absolute difference in metres. "
            "Requires --max-score-abs and --max-matched-iou-loss."
        ),
    )
    parser.add_argument(
        "--max-score-abs",
        type=float,
        default=None,
        help=(
            "Explicit maximum same-index score absolute difference. Requires "
            "--max-corner-abs and --max-matched-iou-loss."
        ),
    )
    parser.add_argument(
        "--max-matched-iou-loss",
        type=float,
        default=None,
        help=(
            "Explicit maximum 1-IoU for same-index axis-aligned prediction "
            "boxes. Requires both absolute-difference limits."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Write the report to stdout; the utility never edits artifacts.",
    )
    parser.add_argument(
        "--require-zero-write",
        action="store_true",
        help=(
            "Require the in-process observer pre/post byte audit, matching "
            "SHA256 digests, and an empty changed-field list in every "
            "diagnostic. Legacy diagnostics without these fields fail."
        ),
    )
    return parser


def _print_text(report: AuditReport) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"YiDu observer identity audit: {status}")
    print(f"comparison mode: {report.comparison_mode}")
    print(f"in-process zero-write required: {report.require_zero_write}")
    if report.numeric_envelope is not None:
        print(
            "numeric envelope: "
            f"corner<={report.numeric_envelope.max_corner_abs:g}, "
            f"score<={report.numeric_envelope.max_score_abs:g}, "
            "same-index IoU loss<="
            f"{report.numeric_envelope.max_matched_iou_loss:g}"
        )
    summary = report.numeric_summary
    print(
        "numeric summary: "
        f"rows={summary.get('prediction_rows', 0)}, "
        f"changed={summary.get('changed_prediction_rows', 0)}, "
        "corner_max="
        f"{summary.get('corner_abs', {}).get('max', 0.0):.12g}, "
        "score_max="
        f"{summary.get('score_abs', {}).get('max', 0.0):.12g}, "
        "same-index_iou_loss_max="
        f"{summary.get('same_index_iou_loss', {}).get('max', 0.0):.12g}"
    )
    print(
        "predictions: "
        f"{report.prediction_files} files / "
        f"{len(report.prediction_scenes)} scenes"
    )
    print(
        "diagnostics: "
        f"{report.diagnostic_files} files / "
        f"{len(report.diagnostic_scenes)} scenes"
    )
    for issue in report.issues:
        print(
            f"[{issue.kind}] {issue.relative_path} "
            f"{issue.object_path}: {issue.message}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = audit_identity(
            args.baseline_root,
            args.observer_root,
            args.diagnostics_root,
            expected_stage=args.expected_stage,
            max_corner_abs=args.max_corner_abs,
            max_score_abs=args.max_score_abs,
            max_matched_iou_loss=args.max_matched_iou_loss,
            require_zero_write=args.require_zero_write,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"identity audit configuration error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        _print_text(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
