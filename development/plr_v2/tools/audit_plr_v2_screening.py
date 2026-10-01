#!/usr/bin/env python3
"""Fast AP and fixed-FP screening for matched cumulative PLR-v2 arms."""
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
PLR_STAGES = ("current", "assoc", "reliability", "score")


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
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scenes = [line.strip() for line in args.scene_list.read_text().splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
    manifest = json.loads((args.arms / "manifest.json").read_text())
    if manifest["scene_count"] != 100 or scenes != manifest["scenes"]:
        raise RuntimeError("scene order/manifest mismatch")
    gts, aligns, transform = scannet_inputs()
    results = {}
    prediction_cache = {}
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
        prediction_cache[directory.name] = predictions
        values = results[directory.name]["ap"]
        print(directory.name, *(f"AP{int(t*100)}={values[str(t)]['ap']:.4f}" for t in THRESHOLDS), flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    component_audit = {}
    prefix_predictions = prediction_cache["prefix"]
    for stage in PLR_STAGES:
        arm = f"prefix_plr_{stage}"
        newly_covered = {str(value): 0 for value in THRESHOLDS}
        plr_count = calr_overlap = prefix_overlap = internal_duplicates = 0
        for scene in scenes:
            prefix_boxes, _ = read(args.arms / "prefix" / f"{scene}_boxes.pkl")
            full_boxes, _ = read(args.arms / arm / f"{scene}_boxes.pkl")
            native_count = int(manifest["per_scene"][scene]["native"])
            calr_count = int(manifest["per_scene"][scene]["calr_v1"])
            if not np.array_equal(full_boxes[: len(prefix_boxes)], prefix_boxes):
                raise RuntimeError(f"prefix rows changed in {arm}/{scene}")
            plr = full_boxes[len(prefix_boxes):]
            calr = prefix_boxes[native_count:native_count + calr_count]
            plr_count += len(plr)
            if len(plr) and len(calr):
                calr_overlap += int(
                    np.any(aabb_iou(plr, calr) >= 0.25, axis=1).sum()
                )
            if len(plr) and len(prefix_boxes):
                prefix_overlap += int(
                    np.any(aabb_iou(plr, prefix_boxes) >= 0.25, axis=1).sum()
                )
            if len(plr) > 1:
                pair = aabb_iou(plr, plr)
                internal_duplicates += int(
                    np.any(np.triu(pair >= 0.50, k=1), axis=1).sum()
                )
            aligned_prefix = transform(prefix_boxes, aligns[scene])
            aligned_plr = transform(plr, aligns[scene])
            prefix_gt = aabb_iou(aligned_prefix, gts[scene])
            plr_gt = aabb_iou(aligned_plr, gts[scene])
            prefix_best = (
                prefix_gt.max(axis=0) if len(aligned_prefix)
                else np.zeros(len(gts[scene]))
            )
            plr_best = (
                plr_gt.max(axis=0) if len(aligned_plr)
                else np.zeros(len(gts[scene]))
            )
            for threshold in THRESHOLDS:
                newly_covered[str(threshold)] += int(
                    np.sum((prefix_best <= threshold) & (plr_best > threshold))
                )

        timing_rows = []
        for scene in scenes:
            diagnostic = json.loads(
                (args.diagnostics / f"{scene}.json").read_text(encoding="utf-8")
            )
            timing_rows.append(
                diagnostic["state"]["plr_timing_and_stability"][stage]
            )
        total_updates = sum(int(row["count"]) for row in timing_rows)
        weighted_mean_ms = sum(
            float(row["mean_ms"]) * int(row["count"]) for row in timing_rows
        ) / max(total_updates, 1)
        weighted_jaccard = sum(
            float(row["output_jaccard_mean"]) * int(row["count"])
            for row in timing_rows
        ) / max(total_updates, 1)
        component_audit[stage] = {
            "plr_outputs": plr_count,
            "new_gt_coverage": newly_covered,
            "calr_overlap_count_iou25": calr_overlap,
            "calr_overlap_rate_iou25": calr_overlap / max(plr_count, 1),
            "prefix_overlap_count_iou25": prefix_overlap,
            "prefix_overlap_rate_iou25": prefix_overlap / max(plr_count, 1),
            "internal_duplicate_rows_iou50": internal_duplicates,
            "mean_update_ms": weighted_mean_ms,
            "scene_p95_update_ms_median": float(
                np.median([float(row["p95_ms"]) for row in timing_rows])
            ),
            "output_jaccard_mean": weighted_jaccard,
            "output_jaccard_min": min(
                float(row["output_jaccard_min"]) for row in timing_rows
            ),
        }

    payload = {
        "schema": "boxfusion.plr_v2.screening_metrics.v1",
        "selection_performed": False,
        "thresholds": THRESHOLDS,
        "fp_budgets_per_100_gt": FP_BUDGETS,
        "results": results,
        "plr_component_audit": component_audit,
    }
    (args.output / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = ["# PLR-v2 matched screening", "", "Selection is applied only after official evaluation.", "",
             "| Arm | Rows | AP15 | AP25 | AP50 |", "|---|---:|---:|---:|---:|"]
    for name, row in results.items():
        lines.append("| {} | {} | {:.4f} | {:.4f} | {:.4f} |".format(
            name, row["rows"], *(row["ap"][str(t)]["ap"] for t in THRESHOLDS)))
    lines.extend([
        "", "## PLR behavior", "",
        "| Stage | Outputs | New GT@15 | New GT@25 | New GT@50 | CALR overlap@25 | Prefix overlap@25 | Mean ms/KF | Mean output Jaccard |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for stage in PLR_STAGES:
        row = component_audit[stage]
        lines.append(
            "| {} | {} | {} | {} | {} | {:.2%} | {:.2%} | {:.3f} | {:.4f} |".format(
                stage,
                row["plr_outputs"],
                row["new_gt_coverage"]["0.15"],
                row["new_gt_coverage"]["0.25"],
                row["new_gt_coverage"]["0.5"],
                row["calr_overlap_rate_iou25"],
                row["prefix_overlap_rate_iou25"],
                row["mean_update_ms"],
                row["output_jaccard_mean"],
            )
        )
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(args.output / "REPORT.md")


if __name__ == "__main__":
    main()
