#!/usr/bin/env python3
"""Evaluate BoxFusion predictions on prepared MultiScan scans."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np

from tools.true_fusion_audit_core import class_agnostic_ap


PROTOCOLS = {
    "nonstructural": "gt_aabb_nonstructural.npy",
    "scannet18": "gt_aabb_scannet18.npy",
}


def load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = pickle.loads(path.read_bytes())
    rows = payload[0] if isinstance(payload, (list, tuple)) and len(payload) == 1 else payload
    corners, scores = [], []
    for row in rows:
        if len(row) != 3:
            raise ValueError(f"{path}: malformed prediction row")
        box, score = np.asarray(row[1], dtype=np.float64), float(row[2])
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
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    scenes = [row["alias"] for row in manifest["scans"]]
    predictions = {}
    for name, root_text in args.prediction:
        root = Path(root_text).resolve()
        predictions[name] = {
            scene: load_prediction(root / f"{scene}_boxes.pkl") for scene in scenes
        }

    report = {"manifest_protocol": manifest["protocol"], "scenes": len(scenes), "thresholds": [0.15, 0.25, 0.5], "protocols": {}}
    for protocol, filename in PROTOCOLS.items():
        ground_truth = {scene: np.load(prepared / scene / filename).astype(np.float64) for scene in scenes}
        protocol_report = {
            "description": manifest["protocols"][protocol],
            "ground_truth_boxes": int(sum(len(value) for value in ground_truth.values())),
            "arms": {},
        }
        for name, per_scene in predictions.items():
            protocol_report["arms"][name] = {
                "prediction_boxes": int(sum(len(value[0]) for value in per_scene.values())),
                "metrics": {
                    str(threshold): class_agnostic_ap(per_scene, ground_truth, threshold)
                    for threshold in report["thresholds"]
                },
            }
        report["protocols"][protocol] = protocol_report

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# MultiScan five-scene preflight AP", ""]
    for protocol, values in report["protocols"].items():
        lines += [
            f"## {protocol}",
            "",
            f"GT boxes: {values['ground_truth_boxes']}. {values['description']}.",
            "",
            "| Arm | AP15 | AP25 | AP50 | Predictions |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, arm in values["arms"].items():
            scores = [arm["metrics"][str(t)]["ap"] for t in report["thresholds"]]
            lines.append(f"| {name} | {scores[0]:.2f} | {scores[1]:.2f} | {scores[2]:.2f} | {arm['prediction_boxes']} |")
        lines.append("")
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
