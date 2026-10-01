#!/usr/bin/env python3
"""Validate the exact P1 prediction/diagnostic set before evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.train_p1_residual_head import (  # noqa: E402
    load_scene_voxels,
    read_scene_ids,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _scalar_text(archive: np.lib.npyio.NpzFile, key: str, path: Path) -> str:
    if key not in archive.files:
        raise ValueError(f"{path}: missing {key}")
    value = np.asarray(archive[key])
    if value.shape != () or value.dtype.hasobject:
        raise ValueError(f"{path}: {key} must be a scalar string")
    result = value.item()
    if isinstance(result, bytes):
        result = result.decode("utf-8")
    if not isinstance(result, str):
        raise ValueError(f"{path}: {key} must be a scalar string")
    return result


def validate(
    *,
    scene_list: Path,
    prediction_root: Path,
    diagnostics_root: Path,
    require_checkpoint: bool,
    expected_checkpoint: Path | None = None,
) -> dict:
    scenes = read_scene_ids(scene_list, role="P1 evaluation")
    expected_predictions = {
        f"{scene}_boxes.pkl" for scene in scenes
    }
    expected_diagnostics = {
        f"{scene}_tracks.npz" for scene in scenes
    }
    actual_predictions = {
        path.name
        for path in prediction_root.glob("scene*_boxes.pkl")
        if path.is_file()
    }
    actual_diagnostics = {
        path.name
        for path in diagnostics_root.glob("scene*_tracks.npz")
        if path.is_file()
    }
    if actual_predictions != expected_predictions:
        missing = sorted(expected_predictions - actual_predictions)
        extra = sorted(actual_predictions - expected_predictions)
        raise ValueError(
            f"P1 prediction set mismatch: missing={missing[:8]}, "
            f"extra={extra[:8]}"
        )
    if actual_diagnostics != expected_diagnostics:
        missing = sorted(expected_diagnostics - actual_diagnostics)
        extra = sorted(actual_diagnostics - expected_diagnostics)
        raise ValueError(
            f"P1 diagnostic set mismatch: missing={missing[:8]}, "
            f"extra={extra[:8]}"
        )
    checkpoint_hashes: set[str] = set()
    for scene in scenes:
        prediction = prediction_root / f"{scene}_boxes.pkl"
        diagnostic = diagnostics_root / f"{scene}_tracks.npz"
        if prediction.stat().st_size <= 0 or diagnostic.stat().st_size <= 0:
            raise ValueError(f"empty P1 artifact for {scene}")
        load_scene_voxels(diagnostic, expected_scene_id=scene)
        with np.load(diagnostic, allow_pickle=False) as archive:
            checkpoint_sha = _scalar_text(
                archive, "p1_checkpoint_sha256", diagnostic
            )
        if require_checkpoint and _SHA256.fullmatch(checkpoint_sha) is None:
            raise ValueError(
                f"{diagnostic}: missing valid P1 checkpoint SHA"
            )
        checkpoint_hashes.add(checkpoint_sha)
    if len(checkpoint_hashes) != 1:
        raise ValueError("P1 diagnostics mix multiple checkpoints")
    checkpoint_sha256 = next(iter(checkpoint_hashes))
    if expected_checkpoint is not None:
        digest = hashlib.sha256()
        with expected_checkpoint.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if checkpoint_sha256 != digest.hexdigest():
            raise ValueError(
                "P1 diagnostics do not match the requested checkpoint"
            )
    return {
        "schema": "boxfusion.p1.run_artifact_validation.v1",
        "scene_count": len(scenes),
        "checkpoint_sha256": checkpoint_sha256,
        "prediction_root": str(prediction_root.resolve()),
        "diagnostics_root": str(diagnostics_root.resolve()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--diagnostics-root", required=True, type=Path)
    parser.add_argument("--require-checkpoint", action="store_true")
    parser.add_argument("--expected-checkpoint", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = validate(
        scene_list=args.scene_list,
        prediction_root=args.prediction_root,
        diagnostics_root=args.diagnostics_root,
        require_checkpoint=args.require_checkpoint,
        expected_checkpoint=args.expected_checkpoint,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
