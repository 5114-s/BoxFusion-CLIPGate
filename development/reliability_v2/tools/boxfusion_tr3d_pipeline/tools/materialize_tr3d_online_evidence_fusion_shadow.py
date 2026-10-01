#!/usr/bin/env python3
"""Materialize a clearly marked, non-authorized evidence-fusion shadow tree."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.tr3d_online_evidence_fusion import EvidenceFusionPolicy, OnlineEvidenceFusion
from boxfusion.tr3d_residual_cache import load_tr3d_residual_cache
from boxfusion.tr3d_terminal_active import save_prediction_create_only
from tools.build_tr3d_c3_source_gate_dataset import read_scenes


def prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        rows = pickle.load(handle)[0]
    corners = np.asarray([row[1] for row in rows], dtype=np.float32)
    scores = np.asarray([row[2] for row in rows], dtype=np.float32)
    if not rows:
        corners = np.empty((0, 8, 3), dtype=np.float32)
    return corners, scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--anchor-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.policy.read_text())
    operating = raw["replace_gate"]["oof"].get("best_unautorized_operating_point")
    if raw.get("activation_authorized") or not isinstance(operating, dict):
        raise ValueError("shadow expects a rejected policy with a frozen diagnostic point")
    policy = EvidenceFusionPolicy.load(args.policy, require_authorized=False)
    policy = replace(
        policy,
        replace=replace(policy.replace, threshold=float(operating["threshold"])),
    )
    fusion = OnlineEvidenceFusion(policy, active=True)
    totals = {"append": 0, "replace": 0, "keep": 0}
    for scene in read_scenes(args.scene_list):
        identity = json.loads((args.diagnostics_root / f"{scene}_c3_online_identity.json").read_text())
        anchors, scores = prediction(args.anchor_root / f"{scene}_boxes.pkl")
        parent_path = Path(identity["parent_cache"])
        with np.load(parent_path, allow_pickle=False) as archive:
            checkpoint = str(archive["checkpoint_sha256"].item())
            config = str(archive["config_sha256"].item())
        parent = load_tr3d_residual_cache(
            parent_path, expected_scene_id=scene, expected_prefix_id="p100",
            expected_checkpoint_sha256=checkpoint, expected_config_sha256=config,
        )
        result = fusion.apply(
            scene_id=scene, identity_summary=identity,
            parent_corners=parent.corners_world, parent_scores=parent.scores_3d,
            anchor_corners=anchors, anchor_scores=scores,
        )
        save_prediction_create_only(result.corners, result.scores, args.output_root / f"{scene}_boxes.pkl")
        totals["append"] += int(result.summary["append_count"])
        totals["replace"] += int(result.summary["replace_count"])
        totals["keep"] += sum(row["action"].startswith("keep") for row in result.summary["decisions"])
    report = {
        "schema": "boxfusion.tr3d_online_evidence_fusion_shadow.v1",
        "complete": True, "formal_activation_authorized": False,
        "validation_predictions_used_for_threshold_selection": False,
        "train_only_frozen_replace_threshold": float(operating["threshold"]),
        "scene_count": len(read_scenes(args.scene_list)), "totals": totals,
        "policy": str(args.policy.resolve()), "output_root": str(args.output_root.resolve()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
