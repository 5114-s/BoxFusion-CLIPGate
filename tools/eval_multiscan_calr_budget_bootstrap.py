#!/usr/bin/env python3
"""Evaluate MultiScan CALR per-scene budgets with paired scene bootstrap."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np

from tools.true_fusion_audit_core import aabb_iou, class_agnostic_ap


THRESHOLDS = (0.15, 0.25, 0.5)
PROTOCOLS = {
    "nonstructural": "gt_aabb_nonstructural.npy",
    "scannet18": "gt_aabb_scannet18.npy",
}


def read_rows(path: Path) -> list[tuple[int, np.ndarray, float]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    rows = payload[0] if isinstance(payload, (list, tuple)) and len(payload) == 1 else payload
    result = []
    for label, box, score in rows:
        corners = np.asarray(box, dtype=np.float64).reshape(8, 3)
        score = float(score)
        if not np.isfinite(corners).all() or not np.isfinite(score):
            raise ValueError(f"non-finite prediction in {path}")
        result.append((int(label), corners, score))
    return result


def as_prediction(rows: list[tuple[int, np.ndarray, float]]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.stack([row[1] for row in rows]) if rows else np.empty((0, 8, 3), dtype=np.float64),
        np.asarray([row[2] for row in rows], dtype=np.float64),
    )


def top_k(rows: list[tuple[int, np.ndarray, float]], budget: int) -> list[tuple[int, np.ndarray, float]]:
    order = sorted(range(len(rows)), key=lambda index: (-rows[index][2], index))
    return [rows[index] for index in order[:budget]]


def ranked_pairs(prediction, ground_truth: np.ndarray, threshold: float) -> np.ndarray:
    corners, scores = prediction
    order = np.argsort(-scores, kind="stable")
    overlap = aabb_iou(corners, ground_truth)
    matched = np.zeros(len(ground_truth), dtype=bool)
    rows = np.zeros((len(order), 3), dtype=np.float64)
    for rank, index in enumerate(order):
        rows[rank, 0] = scores[index]
        values = overlap[index]
        if len(values) and values.max() > threshold:
            target = int(values.argmax())
            if not matched[target]:
                matched[target] = True
                rows[rank, 1] = 1.0
            else:
                rows[rank, 2] = 1.0
        else:
            rows[rank, 2] = 1.0
    return rows


def ap_from_pairs(rows: np.ndarray, gt_count: int) -> float:
    if not len(rows):
        return 0.0
    order = np.argsort(-rows[:, 0], kind="stable")
    tp = rows[order, 1].cumsum()
    fp = rows[order, 2].cumsum()
    recall = tp / (gt_count + 1e-6)
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    mr = np.r_[0.0, recall, 1.0]
    mp = np.r_[0.0, precision, 0.0]
    mp = np.maximum.accumulate(mp[::-1])[::-1]
    changes = np.flatnonzero(mr[1:] != mr[:-1])
    return float(100.0 * np.sum((mr[changes + 1] - mr[changes]) * mp[changes + 1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mvsr-component",
        default="native_max",
        help="Component directory containing the native boxes after MVSR.",
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=(50, 100, 200, 500))
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()
    manifest = json.loads((args.prepared_root / "manifest.json").read_text())
    scenes = [row["alias"] for row in manifest["scans"]]
    if len(scenes) != 50 or len(set(scenes)) != 50:
        raise RuntimeError("expected the fixed 50-scene MultiScan split")
    budgets = sorted(set(int(value) for value in args.budgets))
    if not budgets or budgets[0] < 1:
        raise ValueError("budgets must be positive")

    predictions: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
        "base": {}, "plr_mvsr": {}, "full": {},
        **{f"calr_top{budget}": {} for budget in budgets},
    }
    component_counts = {"native": 0, "plr": 0, "calr": 0}
    for scene in scenes:
        native = read_rows(
            args.components / args.mvsr_component / f"{scene}_boxes.pkl"
        )
        plr = read_rows(args.components / "births_plr_v1" / f"{scene}_boxes.pkl")
        calr = read_rows(args.components / "births_calr_v1" / f"{scene}_boxes.pkl")
        base = read_rows(args.components / "native_native" / f"{scene}_boxes.pkl")
        predictions["base"][scene] = as_prediction(base)
        predictions["plr_mvsr"][scene] = as_prediction(native + plr)
        predictions["full"][scene] = as_prediction(native + plr + calr)
        for budget in budgets:
            predictions[f"calr_top{budget}"][scene] = as_prediction(
                native + plr + top_k(calr, budget)
            )
        component_counts["native"] += len(native)
        component_counts["plr"] += len(plr)
        component_counts["calr"] += len(calr)

    report = {
        "schema": "boxfusion.multiscan_calr_budget_bootstrap.v1",
        "scene_count": len(scenes),
        "budgets": budgets,
        "component_counts": component_counts,
        "mvsr_component": args.mvsr_component,
        "bootstrap": {"replicates": args.bootstraps, "seed": args.seed},
        "protocols": {},
    }
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(scenes), size=(args.bootstraps, len(scenes)))
    for protocol, gt_name in PROTOCOLS.items():
        ground_truth = {
            scene: np.load(args.prepared_root / scene / gt_name).astype(np.float64)
            for scene in scenes
        }
        pooled = {
            arm: {
                str(threshold): class_agnostic_ap(rows, ground_truth, threshold)
                for threshold in THRESHOLDS
            }
            for arm, rows in predictions.items()
        }
        pair_cache = {
            arm: {
                str(threshold): {
                    scene: ranked_pairs(rows[scene], ground_truth[scene], threshold)
                    for scene in scenes
                }
                for threshold in THRESHOLDS
            }
            for arm, rows in predictions.items()
        }
        comparisons = {}
        for left, right in (("full", "base"), ("full", "plr_mvsr")):
            name = f"{left}_minus_{right}"
            comparisons[name] = {}
            for threshold in THRESHOLDS:
                key = str(threshold)
                values = np.empty(args.bootstraps, dtype=np.float64)
                for replicate, draw in enumerate(draws):
                    picked = [scenes[index] for index in draw]
                    gt_count = sum(len(ground_truth[scene]) for scene in picked)
                    left_rows = np.concatenate([pair_cache[left][key][scene] for scene in picked])
                    right_rows = np.concatenate([pair_cache[right][key][scene] for scene in picked])
                    values[replicate] = ap_from_pairs(left_rows, gt_count) - ap_from_pairs(right_rows, gt_count)
                comparisons[name][key] = {
                    "mean_delta": float(values.mean()),
                    "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
                    "p_positive": float(np.mean(values > 0.0)),
                }
        report["protocols"][protocol] = {
            "ground_truth_boxes": int(sum(len(value) for value in ground_truth.values())),
            "pooled": pooled,
            "paired_scene_bootstrap": comparisons,
        }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = ["# MultiScan-50 CALR budget robustness", ""]
    for protocol, values in report["protocols"].items():
        lines += [f"## {protocol}", "", "| Arm | AP15 | AP25 | AP50 | Predictions |", "|---|---:|---:|---:|---:|"]
        for arm, metrics in values["pooled"].items():
            aps = [metrics[str(threshold)]["ap"] for threshold in THRESHOLDS]
            count = metrics[str(THRESHOLDS[0])]["predictions"]
            lines.append(f"| {arm} | {aps[0]:.2f} | {aps[1]:.2f} | {aps[2]:.2f} | {count} |")
        lines += ["", "Paired scene bootstrap (mean delta, 95% CI):", ""]
        for comparison, thresholds in values["paired_scene_bootstrap"].items():
            formatted = []
            for threshold in THRESHOLDS:
                row = thresholds[str(threshold)]
                formatted.append(f"AP{int(threshold*100)} {row['mean_delta']:+.2f} [{row['ci95'][0]:+.2f}, {row['ci95'][1]:+.2f}]")
            lines.append(f"- {comparison}: " + "; ".join(formatted))
        lines.append("")
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
