#!/usr/bin/env python3
"""Paired scene bootstrap for final ReCaR-3D prediction roots."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.audit_ca1m_nms_child_headroom import valid_boxes
from tools.audit_m1m2_remaining_children import read_prediction
from tools.audit_seedless_ablation_matrix import build_scannet_eval
from tools.true_fusion_audit_core import aabb_iou, class_agnostic_ap


THRESHOLDS = (0.15, 0.25, 0.50)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scene_pairs(corners, scores, gt, threshold):
    scores = np.asarray(scores, dtype=np.float64)
    matrix = (
        aabb_iou(corners, gt)
        if len(corners) and len(gt)
        else np.zeros((len(corners), len(gt)), dtype=np.float64)
    )
    matched = np.zeros(len(gt), dtype=bool)
    is_tp = np.zeros(len(scores), dtype=np.float64)
    for row in np.argsort(-scores, kind="mergesort"):
        values = matrix[row]
        if len(values) and values.max() > threshold:
            target = int(values.argmax())
            if not matched[target]:
                matched[target] = True
                is_tp[row] = 1.0
    return np.column_stack((scores, is_tp))


def ap_from_pairs(pairs, gt_count):
    order = np.argsort(-pairs[:, 0], kind="mergesort")
    tp = pairs[order, 1]
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1.0 - tp)
    recall = cum_tp / (gt_count + 1e-6)
    precision = cum_tp / np.maximum(
        cum_tp + cum_fp, np.finfo(np.float64).eps
    )
    mr = np.r_[0.0, recall, 1.0]
    mp = np.r_[0.0, precision, 0.0]
    mp = np.maximum.accumulate(mp[::-1])[::-1]
    changes = np.flatnonzero(mr[1:] != mr[:-1])
    return 100.0 * float(
        np.sum((mr[changes + 1] - mr[changes]) * mp[changes + 1])
    )


def parse_treatments(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--treatment must be NAME=PATH")
        name, raw_path = value.split("=", 1)
        if not name or name in output:
            raise ValueError(f"invalid/duplicate treatment name: {name!r}")
        output[name] = Path(raw_path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("scannet", "ca1m"), required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--treatment", action="append", default=[], required=True)
    parser.add_argument("--ca1m-data-root", type=Path)
    parser.add_argument("--scannet-scan-root", type=Path,
                        default=Path("/extra/ZhaoX/scannet_data/scans"))
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("--replicates must be positive")
    treatments = parse_treatments(args.treatment)
    roots = {"base": args.base_root, **treatments}
    scenes = [
        row.strip()
        for row in args.scene_list.read_text(encoding="utf-8").splitlines()
        if row.strip() and not row.lstrip().startswith("#")
    ]
    expected = 100 if args.dataset == "scannet" else 107
    if len(scenes) != expected or len(set(scenes)) != expected:
        raise RuntimeError(f"expected {expected} unique scenes")
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty {args.output}")
    for name, root in roots.items():
        missing = [s for s in scenes if not (root / f"{s}_boxes.pkl").is_file()]
        if missing:
            raise RuntimeError(f"{name}: missing {len(missing)} predictions")

    if args.dataset == "scannet":
        protocol = {"scenes": scenes, "data": str(args.scannet_scan_root)}
        gts, aligns, transform = build_scannet_eval(protocol)
    else:
        if args.ca1m_data_root is None:
            parser.error("CA-1M requires --ca1m-data-root")
        gts = {
            scene: valid_boxes(
                np.load(
                    args.ca1m_data_root / scene / "after_filter_boxes.npy",
                    allow_pickle=False,
                ),
                scene,
            )
            for scene in scenes
        }
        aligns = {scene: None for scene in scenes}
        transform = lambda boxes, _alignment: boxes

    predictions = {name: {} for name in roots}
    input_hashes = {str(args.scene_list.resolve()): sha256(args.scene_list)}
    for name, root in roots.items():
        for scene in scenes:
            path = root / f"{scene}_boxes.pkl"
            boxes, scores = read_prediction(path)
            predictions[name][scene] = (transform(boxes, aligns[scene]), scores)
            input_hashes[str(path.resolve())] = sha256(path)

    exact = {
        name: {
            str(threshold): class_agnostic_ap(values, gts, threshold)["ap"]
            for threshold in THRESHOLDS
        }
        for name, values in predictions.items()
    }
    pairs = {
        name: {
            threshold: {
                scene: scene_pairs(*values[scene], gts[scene], threshold)
                for scene in scenes
            }
            for threshold in THRESHOLDS
        }
        for name, values in predictions.items()
    }
    deterministic = {
        name: {
            str(threshold): ap_from_pairs(
                np.concatenate([pairs[name][threshold][scene] for scene in scenes]),
                sum(len(gts[scene]) for scene in scenes),
            )
            for threshold in THRESHOLDS
        }
        for name in roots
    }
    max_tie_delta = max(
        abs(exact[name][str(threshold)] - deterministic[name][str(threshold)])
        for name in roots for threshold in THRESHOLDS
    )
    if max_tie_delta > 0.05:
        raise RuntimeError(
            f"deterministic tie convention differs by {max_tie_delta:.4f} AP"
        )

    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(scenes), size=(args.replicates, len(scenes)))
    gt_counts = {scene: len(gts[scene]) for scene in scenes}
    contrasts = {}
    for name in treatments:
        contrasts[name] = {}
        for threshold in THRESHOLDS:
            samples = np.empty(args.replicates, dtype=np.float64)
            for index, draw in enumerate(draws):
                selected = [scenes[int(value)] for value in draw]
                gt_count = sum(gt_counts[scene] for scene in selected)
                base_pairs = np.concatenate(
                    [pairs["base"][threshold][scene] for scene in selected]
                )
                treatment_pairs = np.concatenate(
                    [pairs[name][threshold][scene] for scene in selected]
                )
                samples[index] = (
                    ap_from_pairs(treatment_pairs, gt_count)
                    - ap_from_pairs(base_pairs, gt_count)
                )
            point = exact[name][str(threshold)] - exact["base"][str(threshold)]
            contrasts[name][str(threshold)] = {
                "point_ap": point,
                "bootstrap_mean_ap": float(samples.mean()),
                "ci95_ap": [
                    float(np.percentile(samples, 2.5)),
                    float(np.percentile(samples, 97.5)),
                ],
                "p_positive": float(np.mean(samples > 0.0)),
            }

    result = {
        "schema": "boxfusion.final_recar.paired_scene_bootstrap.v1",
        "dataset": args.dataset,
        "scene_count": len(scenes),
        "replicates": args.replicates,
        "seed": args.seed,
        "thresholds": list(THRESHOLDS),
        "tie_convention": "per-scene and pooled mergesort; exact point estimates use class_agnostic_ap",
        "maximum_exact_vs_deterministic_tie_delta_ap": max_tie_delta,
        "roots": {name: str(path.resolve()) for name, path in roots.items()},
        "exact_ap": exact,
        "contrasts_over_base": contrasts,
        "input_sha256": input_hashes,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        f"# {args.dataset} final paired scene bootstrap", "",
        f"B={args.replicates}, seed={args.seed}; values are AP points.", "",
        "| Contrast | IoU | Point | 95% CI | P(Δ>0) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, by_threshold in contrasts.items():
        for threshold in THRESHOLDS:
            row = by_threshold[str(threshold)]
            lines.append(
                f"| {name} − base | {threshold:.2f} | {row['point_ap']:+.4f} | "
                f"[{row['ci95_ap'][0]:+.4f}, {row['ci95_ap'][1]:+.4f}] | "
                f"{row['p_positive']:.4f} |"
            )
    (args.output / "REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(contrasts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
