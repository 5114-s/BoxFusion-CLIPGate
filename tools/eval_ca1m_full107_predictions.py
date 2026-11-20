#!/usr/bin/env python3
"""Evaluate saved CA-1M predictions with the paper's full107 AP protocol.

This entry point intentionally reuses ``class_agnostic_ap`` from the CaRR-3D
factorial evaluator. Ground truth is loaded directly from every scene's
``after_filter_boxes.npy``; ``instances.json`` is neither required nor used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_ca1m_nms_child_headroom import valid_boxes
from tools.true_fusion_audit_core import class_agnostic_ap


THRESHOLDS = (0.15, 0.25, 0.50)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_ids_from_components(path: Path) -> list[str]:
    scenes = sorted(item.stem for item in path.glob("*.json") if item.stem.isdecimal())
    if len(scenes) != 107 or len(set(scenes)) != 107:
        raise RuntimeError(f"expected 107 unique scene components, found {len(scenes)}")
    return scenes


def parse_prediction(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("prediction must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("prediction must be NAME=PATH")
    return name, Path(raw_path)


def read_prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)) or len(payload) != 1:
        raise RuntimeError(f"invalid prediction container: {path}")
    rows = payload[0]
    if any(not isinstance(row, (list, tuple)) or len(row) != 3 for row in rows):
        raise RuntimeError(f"invalid prediction row: {path}")
    corners = valid_boxes([row[1] for row in rows], str(path))
    scores = np.asarray([float(row[2]) for row in rows], dtype=np.float64)
    if len(corners) != len(scores) or not np.isfinite(scores).all():
        raise RuntimeError(f"invalid prediction scores: {path}")
    return corners, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path("/extra/ZhaoX/boxfusion_ca1m")
    )
    parser.add_argument(
        "--scene-components",
        type=Path,
        default=ROOT
        / "development/final_controls/experiments/ca1m_novis107_20261006/components",
        help="directory whose 107 JSON filenames define the frozen scene set",
    )
    parser.add_argument(
        "--prediction",
        action="append",
        type=parse_prediction,
        required=True,
        help="saved prediction set as NAME=PATH; repeat for multiple methods",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scenes = scene_ids_from_components(args.scene_components)
    ground_truth = {}
    gt_hashes = {}
    for scene in scenes:
        path = args.data_root / scene / "after_filter_boxes.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        ground_truth[scene] = valid_boxes(np.load(path, allow_pickle=False), str(path))
        gt_hashes[scene] = sha256(path)
    gt_count = sum(len(value) for value in ground_truth.values())
    if gt_count != 12911:
        raise RuntimeError(f"expected 12911 full107 GT boxes, found {gt_count}")

    results = {}
    for name, prediction_root in args.prediction:
        prediction_root = prediction_root.resolve()
        actual = {
            item.stem[:-6] if item.stem.endswith("_boxes") else item.stem
            for item in prediction_root.glob("*_boxes.pkl")
        }
        if actual != set(scenes):
            missing = sorted(set(scenes) - actual)
            extra = sorted(actual - set(scenes))
            raise RuntimeError(f"{name}: prediction set mismatch; missing={missing}, extra={extra}")
        predictions = {
            scene: read_prediction(prediction_root / f"{scene}_boxes.pkl")
            for scene in scenes
        }
        metrics = {
            threshold: class_agnostic_ap(predictions, ground_truth, threshold)
            for threshold in THRESHOLDS
        }
        results[name] = {
            "prediction_root": str(prediction_root),
            "boxes": sum(len(value[0]) for value in predictions.values()),
            "ap15": metrics[0.15],
            "ap25": metrics[0.25],
            "ap50": metrics[0.50],
        }
        print(
            f"{name}: AP15={metrics[0.15]['ap']:.4f} "
            f"AP25={metrics[0.25]['ap']:.4f} AP50={metrics[0.50]['ap']:.4f}",
            flush=True,
        )

    payload = {
        "schema": "boxfusion.ca1m.full107.saved_predictions.v1",
        "scene_count": len(scenes),
        "gt_count": gt_count,
        "scenes_skipped": 0,
        "scene_components": str(args.scene_components.resolve()),
        "scene_components_sha256": hashlib.sha256(
            "\n".join(scenes).encode("utf-8")
        ).hexdigest(),
        "metric": {
            "implementation": "tools.true_fusion_audit_core:class_agnostic_ap",
            "iou": "world-corner AABB",
            "threshold_comparison": "strict >",
            "ap": "pooled score sorting, greedy matching, VOC precision envelope",
            "class_labels": "ignored",
        },
        "ground_truth": "after_filter_boxes.npy",
        "ground_truth_hashes": gt_hashes,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
