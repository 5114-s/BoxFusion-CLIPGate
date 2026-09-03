#!/usr/bin/env python3
"""Audit or materialize the isolated CA-1M-native B6 score counterfactual.

No evaluator or ground-truth path exists in this tool.  ``preflight`` is the
default and writes nothing.  ``observer`` computes proposed score changes but
only creates a JSON report.  ``active`` additionally creates a prediction
tree, and is allowed only for an ``activation_authorized=true`` checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import tempfile
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_native_b6_score import (
    load_ca1m_native_b6_scorer,
    load_native_observer_diagnostic,
    sha256_file,
)
from boxfusion.tr3d_terminal_active import save_prediction_create_only


SCHEMA = "boxfusion.ca1m_native_b6_score_counterfactual.v1"
_SCENE = re.compile(r"^[0-9]{8}$")


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    path = path.resolve()
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        raise ValueError(f"{label} must be a non-empty regular file: {path}")
    return path


def _root(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    path = path.resolve()
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"{label} must be a directory: {path}")
    return path


def read_scenes(path: Path) -> tuple[str, ...]:
    path = _regular(path, "scene list")
    scenes = tuple(line.strip() for line in path.read_text().splitlines() if line.strip())
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list must be non-empty and duplicate-free")
    if any(_SCENE.fullmatch(scene) is None for scene in scenes):
        raise ValueError("scene list contains an invalid CA-1M scene id")
    return scenes


def exact_files(root: Path, scenes: tuple[str, ...], suffix: str) -> dict[str, Path]:
    root = _root(root, "artifact root")
    expected = {f"{scene}{suffix}" for scene in scenes}
    actual = {path.name for path in root.iterdir() if path.is_file()}
    if actual != expected:
        raise ValueError(
            f"artifact set differs in {root}: missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )
    return {scene: _regular(root / f"{scene}{suffix}", "scene artifact") for scene in scenes}


def load_prediction(path: Path) -> list[tuple[int, np.ndarray, float]]:
    with _regular(path, "prediction").open("rb") as handle:
        payload = pickle.load(handle)
        if handle.read(1):
            raise ValueError(f"prediction has trailing bytes: {path}")
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], list):
        raise ValueError(f"prediction must contain exactly one list batch: {path}")
    rows: list[tuple[int, np.ndarray, float]] = []
    for index, row in enumerate(payload[0]):
        if not isinstance(row, tuple) or len(row) != 3 or type(row[0]) is not int or row[0] != 0:
            raise ValueError(f"invalid class-agnostic prediction row {index}: {path}")
        corners = np.asarray(row[1])
        score = float(row[2])
        if corners.shape != (8, 3) or not np.isfinite(corners).all():
            raise ValueError(f"invalid prediction OBB row {index}: {path}")
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"invalid prediction score row {index}: {path}")
        rows.append((0, np.array(corners, dtype=np.float32, order="C", copy=True), score))
    return rows


def _write_json_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"refusing existing counterfactual report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"refusing existing counterfactual report: {path}") from error
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _identity_rows(
    scene: str,
    anchor_path: Path,
    observer_path: Path,
) -> tuple[list[tuple[int, np.ndarray, float]], str]:
    anchor_sha = sha256_file(anchor_path)
    if anchor_sha != sha256_file(observer_path):
        raise ValueError(f"{scene}: observer prediction is not byte-identical to anchor")
    anchor = load_prediction(anchor_path)
    observer = load_prediction(observer_path)
    if len(anchor) != len(observer):
        raise ValueError(f"{scene}: observer prediction count differs")
    for index, (left, right) in enumerate(zip(anchor, observer)):
        if left[0] != right[0] or left[2] != right[2] or not np.array_equal(left[1], right[1]):
            raise ValueError(f"{scene}: observer prediction row {index} differs")
    return anchor, anchor_sha


def run(args: argparse.Namespace) -> dict[str, Any]:
    mode = str(args.mode)
    if mode not in {"preflight", "observer", "active"}:
        raise ValueError("counterfactual mode is invalid")
    scenes = read_scenes(args.scene_list)
    anchors = exact_files(args.anchor_root, scenes, "_boxes.pkl")
    observers = exact_files(args.observer_root, scenes, "_boxes.pkl")
    diagnostics = exact_files(args.diagnostics_root, scenes, "_ca1m_native_b6.npz")
    scorer = load_ca1m_native_b6_scorer(
        args.checkpoint,
        args.checkpoint_manifest,
        require_activation_authorized=mode == "active",
    )
    if mode == "preflight" and (args.output is not None or args.prediction_output_root is not None):
        raise ValueError("preflight accepts no output paths")
    if mode == "observer" and args.prediction_output_root is not None:
        raise ValueError("observer mode must not materialize predictions")
    if mode != "preflight" and args.output is None:
        raise ValueError("observer/active modes require --output")
    if args.output is not None and (args.output.exists() or args.output.is_symlink()):
        raise FileExistsError(f"refusing existing counterfactual report: {args.output}")
    if mode == "active" and args.prediction_output_root is None:
        raise ValueError("active mode requires --prediction-output-root")
    active_root: Path | None = None
    if mode == "active":
        raw_root = Path(args.prediction_output_root)
        if raw_root.exists() or raw_root.is_symlink():
            raise FileExistsError(f"refusing existing active prediction root: {raw_root}")
        active_root = raw_root.resolve()

    per_scene: dict[str, Any] = {}
    total_rows = 0
    changed_rows = 0
    proposed_delta: list[np.ndarray] = []
    materialized_paths: list[Path] = []
    if active_root is not None:
        active_root.mkdir(parents=True, exist_ok=False)
    try:
        for scene in scenes:
            rows, identity_sha = _identity_rows(scene, anchors[scene], observers[scene])
            corners = (
                np.stack([row[1] for row in rows])
                if rows else np.empty((0, 8, 3), dtype=np.float32)
            )
            scores = np.asarray([row[2] for row in rows], dtype=np.float32)
            evidence = load_native_observer_diagnostic(
                diagnostics[scene], scene_id=scene, corners=corners, scores=scores
            )
            prediction = scorer.predict(evidence["features"], scores)
            proposed_scores = np.asarray(prediction.scores, dtype=np.float32)
            changed = int(np.count_nonzero(proposed_scores != scores))
            output_sha = None
            if active_root is not None:
                output_path = save_prediction_create_only(
                    corners, proposed_scores, active_root / f"{scene}_boxes.pkl"
                )
                materialized_paths.append(output_path)
                reloaded = load_prediction(output_path)
                if len(reloaded) != len(rows):
                    raise RuntimeError(f"{scene}: active output row count changed")
                for index, (source, target) in enumerate(zip(rows, reloaded)):
                    if source[0] != target[0] or not np.array_equal(source[1], target[1]):
                        raise RuntimeError(f"{scene}: active output OBB/order changed at {index}")
                output_sha = sha256_file(output_path)
            delta = proposed_scores.astype(np.float64) - scores.astype(np.float64)
            proposed_delta.append(delta)
            total_rows += len(rows)
            changed_rows += changed
            per_scene[scene] = {
                "rows": len(rows),
                "changed_scores": changed,
                "anchor_observer_byte_identity": True,
                "obb_unchanged": True,
                "row_count_unchanged": True,
                "row_order_unchanged": True,
                "anchor_prediction_sha256": identity_sha,
                "observer_diagnostic_sha256": sha256_file(diagnostics[scene]),
                "active_prediction_sha256": output_sha,
                "score_delta_min": float(delta.min()) if len(delta) else 0.0,
                "score_delta_max": float(delta.max()) if len(delta) else 0.0,
                "score_delta_mean": float(delta.mean()) if len(delta) else 0.0,
            }
    except Exception:
        if active_root is not None and active_root.exists():
            failed = active_root.with_name(active_root.name + f".failed.{os.getpid()}")
            os.replace(active_root, failed)
        raise

    all_delta = np.concatenate(proposed_delta) if proposed_delta else np.empty(0)
    report = {
        "schema": SCHEMA,
        "complete": True,
        "dataset": "CA1M",
        "mode": mode,
        "score_only": True,
        "active_materialization": mode == "active",
        "activation_authorized": scorer.activation_authorized,
        "ground_truth_access": False,
        "evaluation_invoked": False,
        "scan_net_12d_quality_imported": False,
        "feature_dimension": 14,
        "obb_unchanged": True,
        "row_count_unchanged": True,
        "row_order_unchanged": True,
        "scene_list": str(Path(args.scene_list).resolve()),
        "scene_list_sha256": sha256_file(Path(args.scene_list).resolve()),
        "scenes": len(scenes),
        "rows": total_rows,
        "changed_scores": changed_rows,
        "checkpoint": {
            "path": str(scorer.checkpoint_path),
            "sha256": scorer.checkpoint_sha256,
            "manifest_path": str(scorer.manifest_path),
            "manifest_sha256": scorer.manifest_sha256,
            "detector_blend": scorer.detector_blend,
        },
        "score_delta": {
            "min": float(all_delta.min()) if len(all_delta) else 0.0,
            "max": float(all_delta.max()) if len(all_delta) else 0.0,
            "mean": float(all_delta.mean()) if len(all_delta) else 0.0,
        },
        "prediction_output_root": str(active_root) if active_root is not None else None,
        "per_scene": per_scene,
    }
    if mode == "preflight":
        print(json.dumps({
            "ok": True, "mode": mode, "scenes": len(scenes), "rows": total_rows,
            "activation_authorized": scorer.activation_authorized,
            "outputs_created": False,
        }, indent=2, sort_keys=True))
        return report
    assert args.output is not None
    _write_json_create_only(args.output, report)
    print(json.dumps({
        "mode": mode, "scenes": len(scenes), "rows": total_rows,
        "changed_scores": changed_rows, "activation_authorized": scorer.activation_authorized,
        "report": str(args.output.resolve()),
    }, indent=2, sort_keys=True))
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("preflight", "observer", "active"), default="preflight")
    result.add_argument("--scene-list", type=Path, required=True)
    result.add_argument("--anchor-root", type=Path, required=True)
    result.add_argument("--observer-root", type=Path, required=True)
    result.add_argument("--diagnostics-root", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--checkpoint-manifest", type=Path, required=True)
    result.add_argument("--prediction-output-root", type=Path)
    result.add_argument("--output", type=Path)
    return result


def main() -> int:
    run(parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
