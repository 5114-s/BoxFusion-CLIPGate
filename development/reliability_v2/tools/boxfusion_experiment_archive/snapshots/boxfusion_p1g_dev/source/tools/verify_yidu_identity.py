#!/usr/bin/env python3
"""Strict, read-only identity audit for YiDu observer ablations.

The tool compares trusted local BoxFusion prediction artifacts from a frozen
``B0`` run and one observer-stage run.  Pickle containers are compared
recursively; NumPy arrays and scalars must have identical shapes, dtypes, and
value bytes.  ``.npy`` and ``.npz`` artifacts follow the same rules.

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
from dataclasses import asdict, dataclass, fields, is_dataclass
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

    return issues


def audit_identity(
    baseline_root: Path,
    observer_root: Path,
    diagnostics_root: Path,
    *,
    expected_stage: str | None = None,
) -> AuditReport:
    """Audit predictions and observer diagnostics without modifying artifacts."""

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
            "Strictly compare frozen B0 and YiDu observer predictions, then "
            "audit explicit observer-only safety flags."
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
        "--format",
        choices=("text", "json"),
        default="text",
        help="Write the report to stdout; the utility never edits artifacts.",
    )
    return parser


def _print_text(report: AuditReport) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"YiDu observer identity audit: {status}")
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
