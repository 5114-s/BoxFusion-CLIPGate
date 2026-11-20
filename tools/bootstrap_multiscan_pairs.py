#!/usr/bin/env python3
"""Paired scene bootstrap for matched MultiScan prediction roots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.bootstrap_final_recar_pairs import ap_from_pairs, scene_pairs
from tools.eval_multiscan_boxfusion import load_prediction
from tools.true_fusion_audit_core import class_agnostic_ap


THRESHOLDS = (0.15, 0.25, 0.50)
PROTOCOLS = {
    "nonstructural": "gt_aabb_nonstructural.npy",
    "scannet18": "gt_aabb_scannet18.npy",
}


def parse(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, separator, raw = value.partition("=")
        if not separator or not name or name in result:
            raise ValueError("--treatment must be unique NAME=PATH")
        result[name] = Path(raw)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--treatment", action="append", required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    treatments = parse(args.treatment)
    roots = {"base": args.base_root, **treatments}
    manifest = json.loads((args.prepared_root / "manifest.json").read_text())
    scenes = [row["alias"] for row in manifest["scans"]]
    if len(scenes) != 50 or len(set(scenes)) != 50:
        raise RuntimeError("expected MultiScan-50")
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty {args.output}")
    predictions = {
        name: {scene: load_prediction(root / f"{scene}_boxes.pkl") for scene in scenes}
        for name, root in roots.items()
    }
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(scenes), size=(args.replicates, len(scenes)))
    result = {
        "schema": "boxfusion.multiscan50.paired_scene_bootstrap.v1",
        "replicates": args.replicates, "seed": args.seed, "protocols": {},
    }
    for protocol, filename in PROTOCOLS.items():
        gts = {scene: np.load(args.prepared_root / scene / filename).astype(np.float64) for scene in scenes}
        exact = {
            name: {str(t): class_agnostic_ap(rows, gts, t)["ap"] for t in THRESHOLDS}
            for name, rows in predictions.items()
        }
        pairs = {
            name: {t: {scene: scene_pairs(*rows[scene], gts[scene], t) for scene in scenes} for t in THRESHOLDS}
            for name, rows in predictions.items()
        }
        contrasts = {}
        for name in treatments:
            contrasts[name] = {}
            for threshold in THRESHOLDS:
                samples = np.empty(args.replicates, dtype=np.float64)
                for index, draw in enumerate(draws):
                    selected = [scenes[int(value)] for value in draw]
                    gt_count = sum(len(gts[scene]) for scene in selected)
                    base_rows = np.concatenate([pairs["base"][threshold][scene] for scene in selected])
                    treatment_rows = np.concatenate([pairs[name][threshold][scene] for scene in selected])
                    samples[index] = ap_from_pairs(treatment_rows, gt_count) - ap_from_pairs(base_rows, gt_count)
                point = exact[name][str(threshold)] - exact["base"][str(threshold)]
                contrasts[name][str(threshold)] = {
                    "point_ap": point,
                    "ci95_ap": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
                    "p_positive": float(np.mean(samples > 0.0)),
                }
        result["protocols"][protocol] = {"exact_ap": exact, "contrasts_over_base": contrasts}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = ["# MultiScan-50 paired scene bootstrap", "", f"B={args.replicates}, seed={args.seed}.", ""]
    for protocol, values in result["protocols"].items():
        lines += [f"## {protocol}", "", "| Contrast | IoU | ΔAP | 95% CI | P(Δ>0) |", "|---|---:|---:|---:|---:|"]
        for name, by_threshold in values["contrasts_over_base"].items():
            for threshold in THRESHOLDS:
                row = by_threshold[str(threshold)]
                lines.append(f"| {name} − base | {threshold:.2f} | {row['point_ap']:+.4f} | [{row['ci95_ap'][0]:+.4f}, {row['ci95_ap'][1]:+.4f}] | {row['p_positive']:.4f} |")
        lines.append("")
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
