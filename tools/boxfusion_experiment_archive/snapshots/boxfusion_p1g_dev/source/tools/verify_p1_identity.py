#!/usr/bin/env python3
"""Strict identity and observer-safety audit for the P1 experiment.

Frozen B6 and P1-observer prediction artifacts are compared recursively.
NumPy arrays/scalars require identical shape, dtype, and logical C-order value
bytes; Python floats are compared by IEEE bytes.  P1 diagnostics additionally
must state:

``p1_mutation_enabled == False`` and ``p1_applied_count == 0``.

Missing safety fields fail closed.  If ``p1_applied`` or
``p1_candidate_applied`` is present, it must be a Boolean array with no true
entry.

Pickle files are executable serialization.  Use this utility only with trusted
local BoxFusion outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import struct
import sys
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPORT_SCHEMA = "boxfusion.p1.identity_audit.v1"
SUPPORTED_SUFFIXES = frozenset({".pkl", ".npy", ".npz"})
DIAGNOSTIC_SUFFIX = "_tracks.npz"
_SCENE_PATTERN = re.compile(r"(scene\d{4}_\d{2})")


@dataclass(frozen=True)
class AuditIssue:
    kind: str
    relative_path: str
    object_path: str
    message: str


@dataclass(frozen=True)
class AuditReport:
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
            "schema": REPORT_SCHEMA,
            "ok": self.ok,
            "baseline_root": self.baseline_root,
            "observer_root": self.observer_root,
            "diagnostics_root": self.diagnostics_root,
            "prediction_files": self.prediction_files,
            "prediction_scenes": list(self.prediction_scenes),
            "diagnostic_files": self.diagnostic_files,
            "diagnostic_scenes": list(self.diagnostic_scenes),
            "required_safety_contract": {
                "p1_mutation_enabled": False,
                "p1_applied_count": 0,
            },
            "issues": [asdict(issue) for issue in self.issues],
        }


def _scene_id(path: Path) -> str:
    match = _SCENE_PATTERN.search(path.as_posix())
    return match.group(1) if match is not None else path.stem


def _prediction_files(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"prediction root does not exist: {root}")
    files: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.suffix.lower() not in SUPPORTED_SUFFIXES
            or path.name.endswith(DIAGNOSTIC_SUFFIX)
        ):
            continue
        relative = path.relative_to(root).as_posix()
        # BoxFusion result roots principally contain *_boxes.pkl.  Supporting
        # NumPy prediction sidecars makes this audit useful for frozen exports
        # while avoiding logs and JSON metadata.
        files[relative] = path
    return files


def _diagnostic_files(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"diagnostics root does not exist: {root}")
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob(f"*{DIAGNOSTIC_SUFFIX}"))
        if path.is_file()
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


def _join(parent: str, child: str) -> str:
    if not parent:
        return child
    if child.startswith("["):
        return f"{parent}{child}"
    return f"{parent}.{child}"


def _array_bytes(value: np.ndarray) -> bytes:
    return np.ascontiguousarray(value).tobytes(order="C")


def compare_values(
    baseline: Any, observer: Any, *, object_path: str = "$"
) -> str | None:
    """Return the first exact structural/value mismatch, or ``None``."""

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
                mismatch = compare_values(
                    left,
                    right,
                    object_path=_join(object_path, f"[flat:{index}]"),
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
        left_keys = tuple(baseline.keys())
        right_keys = tuple(observer.keys())
        try:
            same_keys = set(left_keys) == set(right_keys)
        except TypeError:
            same_keys = left_keys == right_keys
        if not same_keys:
            return (
                f"{object_path}: mapping keys differ "
                f"{left_keys!r} != {right_keys!r}"
            )
        for key in left_keys:
            mismatch = compare_values(
                baseline[key],
                observer[key],
                object_path=_join(object_path, f"[{key!r}]"),
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
            mismatch = compare_values(
                left,
                right,
                object_path=_join(object_path, f"[{index}]"),
            )
            if mismatch is not None:
                return mismatch
        return None

    if is_dataclass(baseline) and not isinstance(baseline, type):
        for field in fields(baseline):
            mismatch = compare_values(
                getattr(baseline, field.name),
                getattr(observer, field.name),
                object_path=_join(object_path, field.name),
            )
            if mismatch is not None:
                return mismatch
        return None

    if isinstance(baseline, float):
        if struct.pack("!d", baseline) != struct.pack("!d", observer):
            return f"{object_path}: float value bytes differ"
        return None

    if isinstance(baseline, complex):
        if (
            struct.pack("!dd", baseline.real, baseline.imag)
            != struct.pack("!dd", observer.real, observer.imag)
        ):
            return f"{object_path}: complex value bytes differ"
        return None

    if isinstance(baseline, Path):
        if baseline.as_posix() != observer.as_posix():
            return f"{object_path}: path values differ"
        return None

    try:
        equal = baseline == observer
    except Exception as error:  # pragma: no cover - defensive for custom payloads
        return f"{object_path}: equality raised {type(error).__name__}: {error}"
    if isinstance(equal, np.ndarray):
        if not bool(np.all(equal)):
            return f"{object_path}: values differ"
    elif not bool(equal):
        return f"{object_path}: values differ"
    return None


_compare_values = compare_values


def _scalar_bool(
    archive: Mapping[str, np.ndarray], key: str, path: Path
) -> bool:
    if key not in archive:
        raise ValueError(f"missing required diagnostic key {key!r}")
    value = np.asarray(archive[key])
    if value.shape != () or value.dtype.kind != "b":
        raise TypeError(f"{path}: {key} must be a Boolean scalar")
    return bool(value.item())


def _scalar_int(
    archive: Mapping[str, np.ndarray], key: str, path: Path
) -> int:
    if key not in archive:
        raise ValueError(f"missing required diagnostic key {key!r}")
    value = np.asarray(archive[key])
    if value.shape != () or value.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{path}: {key} must be an integer scalar")
    return int(value.item())


def audit_diagnostic(path: str | os.PathLike[str]) -> tuple[str, ...]:
    """Return safety failures for one P1 diagnostic archive."""

    diagnostic_path = Path(path)
    if not diagnostic_path.is_file():
        return (f"diagnostic not found: {diagnostic_path}",)
    try:
        with np.load(diagnostic_path, allow_pickle=False) as archive_obj:
            archive = {
                name: np.array(archive_obj[name], copy=True)
                for name in archive_obj.files
            }
    except (OSError, ValueError, TypeError) as error:
        return (f"could not load diagnostic without pickle: {error}",)
    failures: list[str] = []
    try:
        mutation_enabled = _scalar_bool(
            archive, "p1_mutation_enabled", diagnostic_path
        )
        if mutation_enabled:
            failures.append(
                "p1_mutation_enabled: observer diagnostics report "
                "mutation_enabled=true"
            )
    except (ValueError, TypeError) as error:
        failures.append(str(error))
    try:
        applied_count = _scalar_int(
            archive, "p1_applied_count", diagnostic_path
        )
        if applied_count != 0:
            failures.append(
                "p1_applied_count: observer diagnostics report "
                f"applied_count={applied_count}"
            )
    except (ValueError, TypeError) as error:
        failures.append(str(error))

    for optional_key in ("p1_applied", "p1_candidate_applied"):
        if optional_key not in archive:
            continue
        value = np.asarray(archive[optional_key])
        if value.dtype.kind != "b":
            failures.append(f"{optional_key} must use Boolean dtype")
        elif bool(np.any(value)):
            failures.append(
                f"{optional_key}: observer diagnostics contain "
                f"{int(np.count_nonzero(value))} applied row(s)"
            )

    if "p1_candidate_boxes" in archive and "p1_candidate_scores" in archive:
        boxes = np.asarray(archive["p1_candidate_boxes"])
        scores = np.asarray(archive["p1_candidate_scores"])
        if boxes.ndim != 2 or boxes.shape[1] != 6:
            failures.append("p1_candidate_boxes must have shape [C,6]")
        elif scores.shape != (len(boxes),):
            failures.append(
                "p1_candidate_scores length disagrees with candidate boxes"
            )
    return tuple(failures)


def audit_identity(
    baseline_root: str | os.PathLike[str],
    observer_root: str | os.PathLike[str],
    diagnostics_root: str | os.PathLike[str],
) -> AuditReport:
    baseline_directory = Path(baseline_root)
    observer_directory = Path(observer_root)
    diagnostic_directory = Path(diagnostics_root)
    baseline_files = _prediction_files(baseline_directory)
    observer_files = _prediction_files(observer_directory)
    issues: list[AuditIssue] = []
    if not baseline_files:
        issues.append(
            AuditIssue(
                "missing_baseline_predictions",
                ".",
                "$",
                "B6 root contains no supported prediction artifacts",
            )
        )
    if diagnostic_directory.is_dir():
        diagnostic_files = _diagnostic_files(diagnostic_directory)
        if not diagnostic_files:
            issues.append(
                AuditIssue(
                    "missing_diagnostics",
                    ".",
                    "$",
                    f"no *{DIAGNOSTIC_SUFFIX} files found",
                )
            )
    else:
        diagnostic_files = {}
        issues.append(
            AuditIssue(
                "missing_diagnostics_root",
                ".",
                "$",
                f"diagnostics root does not exist: {diagnostic_directory}",
            )
        )

    baseline_names = set(baseline_files)
    observer_names = set(observer_files)
    for relative in sorted(baseline_names - observer_names):
        issues.append(
            AuditIssue(
                "missing_observer_file",
                relative,
                "$",
                "prediction exists in B6 root but not P1 observer root",
            )
        )
    for relative in sorted(observer_names - baseline_names):
        issues.append(
            AuditIssue(
                "extra_observer_file",
                relative,
                "$",
                "prediction exists in P1 observer root but not B6 root",
            )
        )
    for relative in sorted(baseline_names & observer_names):
        try:
            baseline = _load_artifact(baseline_files[relative])
            observer = _load_artifact(observer_files[relative])
            mismatch = compare_values(baseline, observer)
        except Exception as error:  # preserve full audit instead of early abort
            issues.append(
                AuditIssue(
                    "artifact_load_error",
                    relative,
                    "$",
                    f"{type(error).__name__}: {error}",
                )
            )
            continue
        if mismatch is not None:
            object_path, _, detail = mismatch.partition(": ")
            issues.append(
                AuditIssue(
                    "prediction_mismatch",
                    relative,
                    object_path,
                    detail or mismatch,
                )
            )

    prediction_scenes = tuple(
        sorted({_scene_id(Path(name)) for name in baseline_names & observer_names})
    )
    diagnostics_by_scene: dict[str, list[tuple[str, Path]]] = {}
    for relative, path in diagnostic_files.items():
        diagnostics_by_scene.setdefault(_scene_id(Path(relative)), []).append(
            (relative, path)
        )
    for scene_id in prediction_scenes:
        matches = diagnostics_by_scene.get(scene_id, [])
        if not matches:
            issues.append(
                AuditIssue(
                    "missing_scene_diagnostic",
                    f"{scene_id}{DIAGNOSTIC_SUFFIX}",
                    "$",
                    "no P1 diagnostic for prediction scene",
                )
            )
            continue
        if len(matches) > 1:
            issues.append(
                AuditIssue(
                    "duplicate_diagnostic",
                    scene_id,
                    "$",
                    f"found {len(matches)} diagnostics for scene",
                )
            )
        for relative, path in matches:
            for failure in audit_diagnostic(path):
                kind = "observer_safety_failure"
                object_path = "$"
                if "missing required diagnostic key" in failure:
                    kind = "missing_or_invalid_safety_key"
                elif "must be a Boolean" in failure or "must be an integer" in failure:
                    kind = "invalid_safety_dtype"
                elif "mutation_enabled=true" in failure:
                    kind = "observer_mutation_enabled"
                elif "applied_count=" in failure or "applied row" in failure:
                    kind = "observer_applied"
                for key in (
                    "p1_mutation_enabled",
                    "p1_applied_count",
                    "p1_applied",
                    "p1_candidate_applied",
                ):
                    if key in failure:
                        object_path = f"$.{key}"
                        break
                issues.append(
                    AuditIssue(
                        kind,
                        relative,
                        object_path,
                        failure,
                    )
                )
    extra_diagnostic_scenes = sorted(
        set(diagnostics_by_scene) - set(prediction_scenes)
    )
    for scene_id in extra_diagnostic_scenes:
        for relative, _ in diagnostics_by_scene[scene_id]:
            issues.append(
                AuditIssue(
                    "extra_scene_diagnostic",
                    relative,
                    "$",
                    "diagnostic scene has no paired prediction artifact",
                )
            )

    return AuditReport(
        baseline_root=str(baseline_directory.resolve()),
        observer_root=str(observer_directory.resolve()),
        diagnostics_root=str(diagnostic_directory.resolve()),
        prediction_files=int(len(baseline_files)),
        prediction_scenes=prediction_scenes,
        diagnostic_files=int(len(diagnostic_files)),
        diagnostic_scenes=tuple(sorted(diagnostics_by_scene)),
        issues=tuple(issues),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--observer-root", required=True, type=Path)
    parser.add_argument("--diagnostics-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--format", choices=("text", "json"), default="json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = audit_identity(
            baseline_root=args.baseline_root,
            observer_root=args.observer_root,
            diagnostics_root=args.diagnostics_root,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"P1 identity audit configuration error: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        os.replace(temporary, args.output)
    if args.format == "json":
        print(rendered)
    else:
        status = "PASS" if report.ok else "FAIL"
        print(f"P1 observer identity audit: {status}")
        print(
            f"predictions: {report.prediction_files} files / "
            f"{len(report.prediction_scenes)} scenes"
        )
        print(
            f"diagnostics: {report.diagnostic_files} files / "
            f"{len(report.diagnostic_scenes)} scenes"
        )
        for issue in report.issues:
            print(
                f"[{issue.kind}] {issue.relative_path} "
                f"{issue.object_path}: {issue.message}"
            )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
