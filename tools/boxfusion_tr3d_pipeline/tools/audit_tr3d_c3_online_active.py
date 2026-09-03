#!/usr/bin/env python3
"""GT-free identity audit for an online C3 append-only prediction tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.tr3d_c2_maskrgbd_cache import sha256_file
from boxfusion.tr3d_c3_online_active import (
    C3SourceGatePolicy,
    RESULT_SCHEMA,
)
from tools.tr3d_data import read_scene_list


SCHEMA = "boxfusion.tr3d_c3_online_active_audit.v1"


def _load_prediction(path: Path) -> list[tuple[int, np.ndarray, float]]:
    if not path.is_file() or path.is_symlink() or path.stat().st_mode & 0o022:
        raise ValueError(f"prediction must be immutable and regular: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - local immutable artifact
    if type(payload) is not list or len(payload) != 1 or type(payload[0]) is not list:
        raise ValueError(f"malformed prediction container: {path}")
    rows: list[tuple[int, np.ndarray, float]] = []
    for index, row in enumerate(payload[0]):
        if type(row) is not tuple or len(row) != 3 or type(row[0]) is not int:
            raise ValueError(f"malformed prediction row {path}:{index}")
        geometry = np.asarray(row[1])
        score = float(row[2])
        if (
            row[0] != 0
            or type(row[1]) is not np.ndarray
            or geometry.dtype != np.float32
            or geometry.shape != (8, 3)
            or not geometry.flags.c_contiguous
            or not np.isfinite(geometry).all()
            or type(row[2]) is not float
            or not math.isfinite(score)
            or score <= 0.0
        ):
            raise ValueError(f"non-canonical prediction row {path}:{index}")
        rows.append((row[0], geometry, score))
    return rows


def _row_equal(left: tuple[int, np.ndarray, float], right: tuple[int, np.ndarray, float]) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[1].dtype == right[1].dtype
        and left[1].tobytes(order="C") == right[1].tobytes(order="C")
        and left[2] == right[2]
    )


def _write_create_only(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"refusing existing C3 active audit: {path}") from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def audit(args: argparse.Namespace) -> dict[str, Any]:
    scenes = read_scene_list(args.scene_list.resolve())
    policy = C3SourceGatePolicy.load(args.policy.resolve())
    anchor_root = args.anchor_root.resolve()
    active_root = args.active_root.resolve()
    diagnostics_root = args.diagnostics_root.resolve()
    issues: list[str] = []
    scene_reports: list[dict[str, Any]] = []
    anchor_digest = hashlib.sha256()
    active_digest = hashlib.sha256()
    total_anchor = 0
    total_applied = 0

    for scene_id in scenes:
        anchor_path = anchor_root / f"{scene_id}_boxes.pkl"
        active_path = active_root / f"{scene_id}_boxes.pkl"
        report_path = diagnostics_root / f"{scene_id}_c3_online_active.json"
        anchors = _load_prediction(anchor_path)
        active = _load_prediction(active_path)
        if not report_path.is_file() or report_path.is_symlink() or report_path.stat().st_mode & 0o022:
            raise ValueError(f"C3 active report must be immutable: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        applied = int(report.get("applied_count", -1))
        if (
            report.get("schema") != RESULT_SCHEMA
            or not report.get("complete")
            or not report.get("active")
            or not report.get("mutation_enabled")
            or not report.get("append_only")
            or report.get("ground_truth_access")
            or report.get("clip_access")
            or report.get("policy_checkpoint_sha256") != policy.sha256
            or report.get("scene_id") != scene_id
            or int(report.get("anchor_count", -1)) != len(anchors)
            or int(report.get("output_count", -1)) != len(active)
            or applied != len(active) - len(anchors)
            or applied < 0
        ):
            issues.append(f"{scene_id}: active report contract mismatch")
        exact_anchor_rows = sum(
            _row_equal(anchor, observed)
            for anchor, observed in zip(anchors, active[: len(anchors)])
        )
        if len(active) < len(anchors) or exact_anchor_rows != len(anchors):
            issues.append(f"{scene_id}: anchor prefix changed")
        anchor_floor = min((row[2] for row in anchors), default=float("inf"))
        candidate_scores = [row[2] for row in active[len(anchors) :]]
        if candidate_scores and not all(score < anchor_floor for score in candidate_scores):
            issues.append(f"{scene_id}: candidate score perturbs anchor order")
        if candidate_scores != sorted(candidate_scores, reverse=True):
            issues.append(f"{scene_id}: candidate scores are not monotonic")
        anchor_sha = sha256_file(anchor_path)
        active_sha = sha256_file(active_path)
        anchor_digest.update(f"{scene_id}\0{anchor_sha}\n".encode())
        active_digest.update(f"{scene_id}\0{active_sha}\n".encode())
        total_anchor += len(anchors)
        total_applied += max(applied, 0)
        scene_reports.append(
            {
                "scene_id": scene_id,
                "anchor_count": len(anchors),
                "applied_count": applied,
                "output_count": len(active),
                "exact_anchor_rows": exact_anchor_rows,
                "anchor_prediction_sha256": anchor_sha,
                "active_prediction_sha256": active_sha,
                "active_report_sha256": sha256_file(report_path),
            }
        )

    result = {
        "schema": SCHEMA,
        "complete": True,
        "ok": not issues,
        "ground_truth_access": False,
        "clip_access": False,
        "append_only": True,
        "policy_checkpoint": str(policy.path),
        "policy_checkpoint_sha256": policy.sha256,
        "scene_list": str(args.scene_list.resolve()),
        "scene_list_sha256": sha256_file(args.scene_list.resolve()),
        "scene_count": len(scenes),
        "anchor_count": total_anchor,
        "applied_count": total_applied,
        "output_count": total_anchor + total_applied,
        "anchor_tree_sha256": anchor_digest.hexdigest(),
        "active_tree_sha256": active_digest.hexdigest(),
        "issues": issues,
        "scenes": scene_reports,
    }
    _write_create_only(args.report.resolve(), result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--scene-list", type=Path, required=True)
    value.add_argument("--policy", type=Path, required=True)
    value.add_argument("--anchor-root", type=Path, required=True)
    value.add_argument("--active-root", type=Path, required=True)
    value.add_argument("--diagnostics-root", type=Path, required=True)
    value.add_argument("--report", type=Path, required=True)
    return value


def main() -> int:
    result = audit(parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
