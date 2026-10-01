#!/usr/bin/env python3
"""Evaluate BoxFusion prediction roots on prepared 3RScan scans."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np

from tools.true_fusion_audit_core import class_agnostic_ap


def load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = pickle.loads(path.read_bytes())
    rows = payload[0] if isinstance(payload, list) and len(payload) == 1 else payload
    corners, scores = [], []
    for row in rows:
        if len(row) != 3:
            raise ValueError(f"{path}: malformed prediction row")
        box = np.asarray(row[1], dtype=np.float64)
        score = float(row[2])
        if box.shape != (8, 3) or not np.isfinite(box).all() or not np.isfinite(score):
            raise ValueError(f"{path}: invalid prediction geometry/score")
        corners.append(box)
        scores.append(score)
    return (
        np.stack(corners) if corners else np.zeros((0, 8, 3), dtype=np.float64),
        np.asarray(scores, dtype=np.float64),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--prediction", action="append", nargs=2, metavar=("NAME", "ROOT"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prepared = args.prepared_root.resolve()
    manifest = json.loads((prepared / "manifest.json").read_text())
    scenes = [row["alias"] for row in manifest["scans"]]
    ground_truth = {}
    raw_ground_truth_boxes = 0
    ignored_ground_truth = []
    for scene in scenes:
        boxes = np.load(prepared / scene / "gt_aabb_scannet18.npy").astype(np.float64)
        raw_ground_truth_boxes += len(boxes)
        finite = np.isfinite(boxes).all(axis=(1, 2))
        nondegenerate = (np.ptp(boxes, axis=1) > 0).all(axis=1)
        valid = finite & nondegenerate
        for index in np.flatnonzero(~valid):
            ignored_ground_truth.append(
                {
                    "scene": scene,
                    "index": int(index),
                    "finite": bool(finite[index]),
                    "extent_xyz": np.ptp(boxes[index], axis=0).tolist(),
                }
            )
        ground_truth[scene] = boxes[valid]
    report = {
        "protocol": manifest["protocol"],
        "scenes": len(scenes),
        "ground_truth_boxes_raw": raw_ground_truth_boxes,
        "ground_truth_boxes": sum(len(value) for value in ground_truth.values()),
        "ignored_ground_truth_boxes": len(ignored_ground_truth),
        "ignored_ground_truth": ignored_ground_truth,
        "thresholds": [0.15, 0.25, 0.5],
        "arms": {},
    }
    for name, root_text in args.prediction:
        root = Path(root_text).resolve()
        predictions = {}
        for scene in scenes:
            path = root / f"{scene}_boxes.pkl"
            if not path.is_file():
                raise FileNotFoundError(path)
            predictions[scene] = load_prediction(path)
        metrics = {
            str(threshold): class_agnostic_ap(predictions, ground_truth, threshold)
            for threshold in report["thresholds"]
        }
        report["arms"][name] = {
            "prediction_root": str(root),
            "prediction_boxes": int(sum(len(value[0]) for value in predictions.values())),
            "metrics": metrics,
        }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# 3RScan BoxFusion AP\n",
        f"Protocol: `{report['protocol']}`; scenes: {report['scenes']}; "
        f"valid GT boxes: {report['ground_truth_boxes']} "
        f"(ignored degenerate/nonfinite: {report['ignored_ground_truth_boxes']}).\n",
        "| Arm | AP15 | AP25 | AP50 | Predictions |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, arm in report["arms"].items():
        values = [arm["metrics"][str(t)]["ap"] for t in report["thresholds"]]
        lines.append(f"| {name} | {values[0]:.2f} | {values[1]:.2f} | {values[2]:.2f} | {arm['prediction_boxes']} |")
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
