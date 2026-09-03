#!/usr/bin/env python3
"""Build leakage-checked train-only labels for incremental TR3D tracks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.tr3d_incremental_gate import (  # noqa: E402
    DATASET_SCHEMA, FEATURE_NAMES, candidate_features,
)
from tools.audit_tr3d_residual_observer import (  # noqa: E402
    _alignment, _gt_boxes, _minmax, _transform, pairwise_iou,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scenes(path: Path) -> tuple[str, ...]:
    rows = tuple(line.split()[0] for line in path.read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith("#"))
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("scene list is empty or contains duplicates")
    return rows


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.link(temporary, path)
        path.chmod(0o444)
    finally:
        try: os.unlink(temporary)
        except FileNotFoundError: pass


def build(args: argparse.Namespace) -> dict:
    train = scenes(args.train_scene_list.resolve())
    forbidden = scenes(args.forbidden_validation_scene_list.resolve())
    overlap = sorted(set(train) & set(forbidden))
    if overlap or len(forbidden) < 100:
        raise ValueError(f"train/validation leakage: {overlap[:5]}")
    features, scene_ids, track_ids = [], [], []
    target_iou, novel15, novel25, novel50 = [], [], [], []
    per_scene = {}
    for scene in train:
        path = args.diagnostics_root / f"{scene}_tr3d_incremental.json"
        payload = json.loads(path.read_text())
        expected_schema = (
            "boxfusion.tr3d_lightweight_online_observer.v1"
            if args.lightweight_stage is not None
            else "boxfusion.tr3d_incremental_online_observer.v3"
        )
        if (payload.get("schema") != expected_schema
                or payload.get("scene_id") != scene
                or payload.get("observer_only") is not True
                or payload.get("mutation_enabled") is not False
                or payload.get("ground_truth_access") is not False
                or payload.get("applied_count") != 0):
            raise ValueError(f"{path}: invalid observer contract")
        if args.lightweight_stage is not None and (
            int(payload.get("lightweight_stage", -1))
            != args.lightweight_stage
        ):
            raise ValueError(f"{path}: lightweight stage mismatch")
        transform = _alignment(args.scans_root.resolve(), scene)
        gt = _gt_boxes(args.ground_truth_root / f"{scene}_bbox.npy")
        anchors = np.asarray(payload["anchor_corners_world"], dtype=np.float64)
        if not len(anchors): anchors = np.empty((0, 8, 3), dtype=np.float64)
        anchor_iou = pairwise_iou(_minmax(_transform(anchors, transform)), gt)
        accepted = 0
        for row in payload["confirmed"]:
            corners_key = (
                "selected_corners_world"
                if args.lightweight_stage is not None
                and args.lightweight_stage >= 5
                else "best_corners_world"
            )
            corners = np.asarray(row[corners_key], dtype=np.float64)[None]
            candidate_iou = pairwise_iou(_minmax(_transform(corners, transform)), gt)[0]
            best_gt = int(np.argmax(candidate_iou)) if len(gt) else -1
            maximum = float(candidate_iou[best_gt]) if best_gt >= 0 else 0.0
            anchor_for_gt = float(anchor_iou[:, best_gt].max(initial=0.0)) if best_gt >= 0 else 0.0
            feature_row = dict(row)
            if args.lightweight_stage is not None:
                feature_row["anchor_iou_max"] = float(
                    row["selected_anchor_iou_max"]
                )
                feature_row["anchor_center_distance_m"] = float(
                    row["selected_anchor_center_distance_m"]
                )
            features.append(candidate_features(
                feature_row, int(payload["provider_calls"])
            ))
            target_iou.append(maximum)
            novel15.append(maximum >= 0.15 and anchor_for_gt < 0.15)
            novel25.append(maximum >= 0.25 and anchor_for_gt < 0.25)
            novel50.append(maximum >= 0.50 and anchor_for_gt < 0.50)
            scene_ids.append(scene); track_ids.append(int(row["track_id"])); accepted += 1
        per_scene[scene] = accepted
    if len(features) < 100 or len(set(scene_ids)) < 10:
        raise ValueError("incremental novelty dataset is too small")
    arrays = {
        "features": np.asarray(features, dtype=np.float32),
        "target_iou": np.asarray(target_iou, dtype=np.float32),
        "novel_iou15": np.asarray(novel15, dtype=np.bool_),
        "novel_iou25": np.asarray(novel25, dtype=np.bool_),
        "novel_iou50": np.asarray(novel50, dtype=np.bool_),
        "scene_ids": np.asarray(scene_ids), "track_ids": np.asarray(track_ids, dtype=np.int64),
        "feature_names": np.asarray(FEATURE_NAMES), "schema": np.asarray(DATASET_SCHEMA),
        "train_scene_list_sha256": np.asarray(sha256(args.train_scene_list.resolve())),
        "forbidden_validation_scene_list_sha256": np.asarray(sha256(args.forbidden_validation_scene_list.resolve())),
    }
    atomic_npz(args.output.resolve(), arrays)
    report = {"schema": DATASET_SCHEMA, "train_only": True,
              "ground_truth_used_only_for_training": True,
              "validation_predictions_used_for_training": False,
              "validation_overlap_count": 0, "scenes": len(train),
              "samples": len(features), "positive_novel_iou15": int(np.sum(novel15)),
              "positive_novel_iou25": int(np.sum(novel25)),
              "positive_novel_iou50": int(np.sum(novel50)), "per_scene": per_scene,
              "dataset": str(args.output.resolve()), "dataset_sha256": sha256(args.output.resolve())}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.report.exists(): raise FileExistsError(args.report)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.report.chmod(0o444)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--train-scene-list", type=Path, required=True)
    value.add_argument("--forbidden-validation-scene-list", type=Path, required=True)
    value.add_argument("--diagnostics-root", type=Path, required=True)
    value.add_argument("--ground-truth-root", type=Path, required=True)
    value.add_argument("--scans-root", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--report", type=Path, required=True)
    value.add_argument("--lightweight-stage", type=int, choices=range(1, 7))
    return value


if __name__ == "__main__":
    print(json.dumps(build(parser().parse_args()), indent=2, sort_keys=True))
