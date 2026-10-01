#!/usr/bin/env python3
"""Evaluate and report the paired CA-1M online no-child factorial."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.audit_ca1m_nms_child_headroom import valid_boxes
from tools.true_fusion_audit_core import class_agnostic_ap


ARMS = ("base", "p", "a", "m2", "p_a", "p_m2", "a_m2", "p_a_m2")
THRESHOLDS = (0.15, 0.25, 0.50)


def read_prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)) or len(payload) != 1:
        raise RuntimeError(f"invalid prediction container: {path}")
    rows = payload[0]
    if any(len(row) != 3 or int(row[0]) != 0 for row in rows):
        raise RuntimeError(f"non-class-agnostic prediction: {path}")
    boxes = valid_boxes([row[1] for row in rows], str(path))
    scores = np.asarray([row[2] for row in rows], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise RuntimeError(f"non-finite prediction score: {path}")
    return boxes, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--factorial-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scenes = [
        line.strip()
        for line in args.scene_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    manifest = json.loads(
        (args.factorial_root / "manifest.json").read_text(encoding="utf-8")
    )
    if len(scenes) != 107 or manifest.get("scene_count") != 107:
        raise RuntimeError("CA-1M paired factorial must contain 107 scenes")
    if manifest.get("children_enabled") is not False:
        raise RuntimeError("child evidence must be disabled")

    ground_truth = {
        scene: valid_boxes(
            np.load(args.data_root / scene / "after_filter_boxes.npy"), scene
        )
        for scene in scenes
    }

    results = {}
    for arm in ARMS:
        prediction_root = args.factorial_root / arm
        missing = [
            scene for scene in scenes
            if not (prediction_root / f"{scene}_boxes.pkl").is_file()
        ]
        if missing:
            raise RuntimeError(f"{arm}: missing {len(missing)} predictions")
        predictions = {
            scene: read_prediction(prediction_root / f"{scene}_boxes.pkl")
            for scene in scenes
        }
        metrics = {
            threshold: class_agnostic_ap(
                predictions, ground_truth, threshold
            )
            for threshold in THRESHOLDS
        }
        results[arm] = {
            "boxes": int(manifest["arm_counts"][arm]),
            "ap15": float(metrics[0.15]["ap"]),
            "ap25": float(metrics[0.25]["ap"]),
            "ap50": float(metrics[0.50]["ap"]),
        }
        print(arm, results[arm], flush=True)

    def delta(left: str, right: str) -> dict[str, float]:
        return {
            key: results[right][key] - results[left][key]
            for key in ("ap15", "ap25", "ap50")
        }

    payload = {
        "schema": "boxfusion.ca1m.strict_causal_nochild.factorial.v1",
        "scene_count": 107,
        "paired_from_one_run": True,
        "children_enabled": False,
        "results": results,
        "conditional_effects": {
            "PLR|base": delta("base", "p"),
            "CALR|base": delta("base", "a"),
            "MVSR|base": delta("base", "m2"),
            "MVSR|PLR": delta("p", "p_m2"),
            "MVSR|CALR": delta("a", "a_m2"),
            "MVSR|PLR+CALR": delta("p_a", "p_a_m2"),
            "CALR|PLR+MVSR": delta("p_m2", "p_a_m2"),
            "PLR|CALR+MVSR": delta("a_m2", "p_a_m2"),
            "full|base": delta("base", "p_a_m2"),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    labels = {
        "base": "Base", "p": "PLR", "a": "CALR", "m2": "MVSR",
        "p_a": "PLR+CALR", "p_m2": "PLR+MVSR",
        "a_m2": "CALR+MVSR", "p_a_m2": "PLR+CALR+MVSR",
    }
    lines = [
        "# CA-1M-107 strict-causal online no-child factorial", "",
        "All eight arms are materialized from one complete online run. PLR uses",
        "post-NMS proposals only; NMS-suppressed children are disabled.", "",
        "| Configuration | Boxes | AP15 | AP25 | AP50 |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = results[arm]
        lines.append(
            f"| {labels[arm]} | {row['boxes']:,} | {row['ap15']:.4f} | "
            f"{row['ap25']:.4f} | {row['ap50']:.4f} |"
        )
    lines.extend(["", "## Conditional effects", "", "| Effect | AP15 | AP25 | AP50 |", "|---|---:|---:|---:|"])
    for name, values in payload["conditional_effects"].items():
        lines.append(
            f"| {name} | {values['ap15']:+.4f} | {values['ap25']:+.4f} | "
            f"{values['ap50']:+.4f} |"
        )
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
