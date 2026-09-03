#!/usr/bin/env python3
"""Run the online evidence-fusion head without mutating predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.tr3d_online_evidence_fusion import OnlineEvidenceFusion
from boxfusion.tr3d_residual_cache import load_tr3d_residual_cache
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
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    fusion = OnlineEvidenceFusion.from_checkpoint(args.policy, active=False)
    counts: dict[str, int] = {}
    runtimes = []
    rows = []
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
        started = time.perf_counter()
        result = fusion.apply(
            scene_id=scene, identity_summary=identity,
            parent_corners=parent.corners_world, parent_scores=parent.scores_3d,
            anchor_corners=anchors, anchor_scores=scores,
        )
        runtimes.append((time.perf_counter() - started) * 1000.0)
        if not np.array_equal(result.corners, anchors) or not np.array_equal(result.scores, scores):
            raise RuntimeError(f"{scene}: observer mutated predictions")
        scene_counts: dict[str, int] = {}
        for decision in result.summary["decisions"]:
            action = str(decision["action"])
            counts[action] = counts.get(action, 0) + 1
            scene_counts[action] = scene_counts.get(action, 0) + 1
        rows.append({"scene_id": scene, "decisions": scene_counts, "runtime_ms": runtimes[-1]})
    report = {
        "schema": "boxfusion.tr3d_online_evidence_fusion_observer_audit.v1",
        "complete": True, "observer_only": True, "prediction_identity": True,
        "activation_authorized": fusion.policy.activation_authorized,
        "scenes": len(rows), "decision_counts": counts,
        "runtime_ms": {
            "mean": float(np.mean(runtimes)), "median": float(np.median(runtimes)),
            "p95": float(np.quantile(runtimes, 0.95)), "max": float(np.max(runtimes)),
        },
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("activation_authorized", "scenes", "decision_counts", "runtime_ms")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
