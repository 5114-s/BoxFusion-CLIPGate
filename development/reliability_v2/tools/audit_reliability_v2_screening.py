#!/usr/bin/env python3
"""Fast AP and fixed-FP screening for reliability-v2 materialized arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path("/data/ZhaoX/BoxFusion")
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:] = [
    value
    for value in sys.path
    if Path(value or ".").resolve() != SCRIPT_DIR
]
sys.path.insert(0, str(MAIN))

from tools.audit_final_ledger import scannet_inputs
from tools.true_fusion_audit_core import aabb_iou, class_agnostic_ap


THRESHOLDS = (0.15, 0.25, 0.50)
FP_BUDGETS = (1, 5, 10, 20)


def read(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)[0]
    if not payload:
        return np.empty((0, 8, 3)), np.empty(0)
    return (
        np.asarray([row[1] for row in payload], dtype=np.float64).reshape(-1, 8, 3),
        np.asarray([row[2] for row in payload], dtype=np.float64),
    )


def fixed_fp(predictions, ground_truth, threshold: float) -> dict[str, float]:
    entries = []
    overlaps = {}
    for scene, (boxes, scores) in predictions.items():
        overlaps[scene] = aabb_iou(boxes, ground_truth[scene])
        entries.extend((float(score), scene, row) for row, score in enumerate(scores))
    entries.sort(key=lambda value: (-value[0], value[1], value[2]))
    used = {scene: np.zeros(len(ground_truth[scene]), dtype=bool) for scene in ground_truth}
    total_gt = sum(len(rows) for rows in ground_truth.values())
    limits = {budget: budget * total_gt / 100.0 for budget in FP_BUDGETS}
    result = {str(budget): 0.0 for budget in FP_BUDGETS}
    tp = fp = 0
    pending = set(FP_BUDGETS)
    for _, scene, row in entries:
        values = overlaps[scene][row]
        if len(values) and float(values.max()) > threshold:
            target = int(values.argmax())
            if not used[scene][target]:
                used[scene][target] = True
                tp += 1
            else:
                fp += 1
        else:
            fp += 1
        for budget in list(pending):
            if fp <= limits[budget]:
                result[str(budget)] = 100.0 * tp / max(total_gt, 1)
            else:
                pending.remove(budget)
        if not pending:
            break
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scenes = [line.strip() for line in args.scene_list.read_text().splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
    manifest = json.loads((args.arms / "manifest.json").read_text())
    if manifest["scene_count"] != 100 or scenes != manifest["scenes"]:
        raise RuntimeError("scene order/manifest mismatch")
    gts, aligns, transform = scannet_inputs()
    results = {}
    for directory in sorted(path for path in args.arms.iterdir() if path.is_dir()):
        predictions = {}
        for scene in scenes:
            boxes, scores = read(directory / f"{scene}_boxes.pkl")
            predictions[scene] = (transform(boxes, aligns[scene]), scores)
        results[directory.name] = {
            "rows": sum(len(value[1]) for value in predictions.values()),
            "ap": {
                str(threshold): class_agnostic_ap(predictions, gts, threshold)
                for threshold in THRESHOLDS
            },
            "fixed_fp_recall": {
                str(threshold): fixed_fp(predictions, gts, threshold)
                for threshold in THRESHOLDS
            },
        }
        values = results[directory.name]["ap"]
        print(directory.name, *(f"AP{int(t*100)}={values[str(t)]['ap']:.4f}" for t in THRESHOLDS), flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "boxfusion.reliability_v2.screening_metrics.v1",
        "selection_performed": False,
        "thresholds": THRESHOLDS,
        "fp_budgets_per_100_gt": FP_BUDGETS,
        "results": results,
    }
    (args.output / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = ["# Reliability-v2 screening", "", "No final variant is selected by this script.", "",
             "| Arm | Rows | AP15 | AP25 | AP50 |", "|---|---:|---:|---:|---:|"]
    for name, row in results.items():
        lines.append("| {} | {} | {:.4f} | {:.4f} | {:.4f} |".format(
            name, row["rows"], *(row["ap"][str(t)]["ap"] for t in THRESHOLDS)))
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(args.output / "REPORT.md")


if __name__ == "__main__":
    main()
