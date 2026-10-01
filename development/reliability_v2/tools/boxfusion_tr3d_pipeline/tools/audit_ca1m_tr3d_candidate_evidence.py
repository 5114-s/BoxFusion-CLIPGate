#!/usr/bin/env python3
"""Audit exact GT-free depth evidence for CA-1M terminal candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_native_b6_observer import (  # noqa: E402
    FEATURE_NAMES,
    SCHEMA as EVIDENCE_SCHEMA,
)
from boxfusion.ca1m_tr3d_terminal import (  # noqa: E402
    SCHEMA as TERMINAL_SCHEMA,
    sha256_file,
)


SCENE_RE = re.compile(r"^[0-9]{8}$")
AUDIT_SCHEMA = "boxfusion.ca1m_tr3d_candidate_evidence_audit.v1"


def _regular(path: Path, name: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink: {path}")
    result = path.resolve()
    if not result.is_file() or result.is_symlink() or result.stat().st_size <= 0:
        raise FileNotFoundError(f"missing regular {name}: {result}")
    return result


def _scenes(path: Path) -> tuple[str, ...]:
    source = _regular(path, "scene list")
    result = tuple(row.strip() for row in source.read_text().splitlines() if row.strip())
    if (
        not result
        or len(result) != len(set(result))
        or any(SCENE_RE.fullmatch(scene) is None for scene in result)
    ):
        raise ValueError("scene list is empty, duplicate, or malformed")
    return result


def _exact_files(root: Path, suffix: str, scenes: tuple[str, ...], name: str) -> dict[str, Path]:
    if root.is_symlink():
        raise ValueError(f"{name} root must not be a symlink")
    directory = root.resolve()
    if not directory.is_dir() or directory.is_symlink():
        raise FileNotFoundError(f"missing {name} root: {directory}")
    expected = {f"{scene}{suffix}" for scene in scenes}
    actual = {
        path.name for path in directory.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    symlinks = [path.name for path in directory.iterdir() if path.is_symlink()]
    if actual != expected or symlinks:
        raise ValueError(
            f"{name} exact-set mismatch: missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}, symlinks={sorted(symlinks)}"
        )
    return {scene: directory / f"{scene}{suffix}" for scene in scenes}


def _scalar(archive: Any, name: str, expected: Any) -> None:
    value = np.asarray(archive[name])
    if value.shape != () or value.item() != expected:
        raise ValueError(f"candidate evidence scalar {name} disagrees")


def _write_create_only(path: Path, payload: dict[str, Any]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.",
            suffix=".tmp", delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        target.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"refusing existing candidate evidence audit: {target}") from error
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    scenes = _scenes(args.scene_list)
    terminal_paths = _exact_files(
        args.terminal_cache_root,
        "_ca1m_tr3d_terminal.npz",
        scenes,
        "terminal cache",
    )
    evidence_paths = _exact_files(
        args.candidate_evidence_root,
        "_ca1m_native_b6.npz",
        scenes,
        "candidate evidence",
    )
    terminal_audit_path = _regular(args.terminal_audit, "sealed terminal audit")
    terminal_audit = json.loads(terminal_audit_path.read_text())
    if (
        terminal_audit.get("schema")
        != "boxfusion.ca1m_tr3d_terminal_observer_audit.v1"
        or terminal_audit.get("ok") is not True
        or terminal_audit.get("ground_truth_access") is not False
        or terminal_audit.get("scene_count") != len(scenes)
        or set((terminal_audit.get("scenes") or {}).keys()) != set(scenes)
    ):
        raise ValueError("terminal observer audit contract mismatch")

    per_scene: dict[str, Any] = {}
    total_candidates = 0
    total_valid = 0
    for scene in scenes:
        terminal_path = terminal_paths[scene]
        evidence_path = evidence_paths[scene]
        terminal_sha = sha256_file(terminal_path)
        if terminal_sha != terminal_audit["scenes"][scene]["artifact_sha256"]:
            raise ValueError(f"terminal cache changed after seal: {scene}")
        with np.load(terminal_path, allow_pickle=False) as terminal:
            for name, expected in (
                ("schema", TERMINAL_SCHEMA), ("complete", True),
                ("observer_only", True), ("mutation_enabled", False),
                ("ground_truth_access", False), ("adapter_mode", "genuine"),
                ("scene_id", scene),
            ):
                value = np.asarray(terminal[name])
                if value.shape != () or value.item() != expected:
                    raise ValueError(f"terminal cache {name} mismatch: {scene}")
            corners = np.array(terminal["candidate_corners"], copy=True)
            scores = np.array(terminal["candidate_scores"], copy=True)
            frames = np.array(terminal["used_frame_ids"], copy=True)
        with np.load(evidence_path, allow_pickle=False) as evidence:
            required = {
                "schema", "complete", "observer_only", "mutation_enabled",
                "applied_count", "ground_truth_access", "clip_access", "scene_id",
                "result_indices", "stable_ids", "corners", "scores", "used_frame_ids",
                "feature_names", "features", "valid_evidence", "summary_json",
            }
            if not required.issubset(set(evidence.files)):
                raise ValueError(f"candidate evidence fields missing: {scene}")
            for name, expected in (
                ("schema", EVIDENCE_SCHEMA), ("complete", True),
                ("observer_only", True), ("mutation_enabled", False),
                ("applied_count", 0), ("ground_truth_access", False),
                ("clip_access", False), ("scene_id", scene),
            ):
                _scalar(evidence, name, expected)
            count = len(corners)
            if not np.array_equal(evidence["result_indices"], np.arange(count, dtype=np.int64)):
                raise ValueError(f"candidate evidence result_indices mismatch: {scene}")
            if not np.array_equal(evidence["stable_ids"], np.arange(count, dtype=np.int64)):
                raise ValueError(f"candidate evidence stable_ids mismatch: {scene}")
            if not np.array_equal(evidence["corners"], corners):
                raise ValueError(f"candidate evidence corners differ from cache: {scene}")
            if not np.array_equal(evidence["scores"], scores):
                raise ValueError(f"candidate evidence scores differ from cache: {scene}")
            if not np.array_equal(evidence["used_frame_ids"], frames):
                raise ValueError(f"candidate evidence frame lineage differs: {scene}")
            names = tuple(str(value) for value in np.asarray(evidence["feature_names"]).tolist())
            features = np.asarray(evidence["features"])
            valid = np.asarray(evidence["valid_evidence"])
            if names != FEATURE_NAMES or features.shape != (count, len(FEATURE_NAMES)):
                raise ValueError(f"candidate evidence feature schema mismatch: {scene}")
            if (
                not np.issubdtype(features.dtype, np.floating)
                or not np.isfinite(features).all()
                or np.any(features < 0.0)
                or np.any(features > 1.0)
                or valid.dtype != np.bool_
                or valid.shape != (count,)
            ):
                raise ValueError(f"candidate evidence numeric contract mismatch: {scene}")
            if count and not np.array_equal(features[:, 0].astype(np.float32), scores):
                raise ValueError(f"candidate evidence score feature mismatch: {scene}")
            summary = json.loads(str(np.asarray(evidence["summary_json"]).item()))
            if (
                summary.get("ground_truth_access") is not False
                or summary.get("mutation_enabled") is not False
                or summary.get("mapping_rows") != count
                or summary.get("prediction_rows") != count
            ):
                raise ValueError(f"candidate evidence summary mismatch: {scene}")
            valid_count = int(np.count_nonzero(valid))
        total_candidates += len(corners)
        total_valid += valid_count
        per_scene[scene] = {
            "terminal_cache_sha256": terminal_sha,
            "candidate_evidence_sha256": sha256_file(evidence_path),
            "candidate_rows": len(corners),
            "valid_evidence_rows": valid_count,
        }
    payload = {
        "schema": AUDIT_SCHEMA,
        "ok": True,
        "complete": True,
        "ground_truth_access": False,
        "mutation_enabled": False,
        "scene_count": len(scenes),
        "candidate_rows": total_candidates,
        "valid_evidence_rows": total_valid,
        "scene_list_sha256": sha256_file(_regular(args.scene_list, "scene list")),
        "terminal_audit_sha256": hashlib.sha256(terminal_audit_path.read_bytes()).hexdigest(),
        "scenes": per_scene,
    }
    if args.output is not None:
        _write_create_only(args.output, payload)
    return payload


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--scene-list", type=Path, required=True)
    value.add_argument("--terminal-cache-root", type=Path, required=True)
    value.add_argument("--candidate-evidence-root", type=Path, required=True)
    value.add_argument("--terminal-audit", type=Path, required=True)
    value.add_argument("--output", type=Path)
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), sort_keys=True))
