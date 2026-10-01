#!/usr/bin/env python3
"""Run a fixed-prefix CA1M true-observation pilot in a new artifact directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=10)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--config", type=Path, default=ROOT / "config/ca1m_thr15.yaml")
    args = parser.parse_args()
    if not 1 <= args.scenes <= 30:
        parser.error("pilot scene count must be between 1 and 30")
    args.run_dir = args.run_dir.resolve()
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    scene_path = ROOT / "tools/boxfusion_tr3d_pipeline/evaluation/data_util/meta_data/ca1m_val_full107.txt"
    scenes = [line.strip() for line in scene_path.read_text().splitlines() if line.strip()][:args.scenes]
    sources = [args.config, scene_path, ROOT / "demo.py", ROOT / "boxfusion/box_fusion.py",
               ROOT / "boxfusion/instances.py", ROOT / "boxfusion/box_manager.py",
               ROOT / "boxfusion/reliable_views.py", ROOT / "boxfusion/boxer_lifter.py",
               ROOT / "tools/capture_true_fusion_observations.py",
               Path(__file__), ROOT / "tools/audit_true_fusion_observations.py",
               ROOT / "tools/true_fusion_audit_core.py"]
    hashes = {str(path.resolve()): digest(path) for path in sources}
    args.run_dir.mkdir()
    protocol = {
        "schema": "boxfusion.true_fusion_pilot.v1", "scenes": scenes,
        "selection": "first N scenes in fixed official CA1M full107 list; not GT-selected",
        "baseline": "CuTR+Boxer active+TopK3+real score; threshold=.15; gap20",
        "training": "none; inference captures do not read annotation boxes",
        "primary_identity_proxy": "exactly one GT at strict AABB IoU>.15; unmatched and ambiguous remain unresolved",
        "sensitivity": "IoU>.25 and >.50; best-GT mapping separately and explicitly permissive",
        "denominators": ["actual PFO events", "unique selected observation sets",
                         "pre-TopK retained sources", "final retained membership (not full lineage)"],
        "conditional_next_stage": "if actual mixed inputs exist, identity replay then GT-partitioned native PFO",
        "refusion_constraint": "retain native minimum of 3 valid distinct source frames; no threshold relaxation",
        "limitations": "pilot only, neither full107 active AP nor end-to-end FPS; no universal impossibility claim",
        "input_sha256": hashes,
    }
    with (args.run_dir / "protocol.json").open("x") as handle:
        json.dump(protocol, handle, indent=2)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONDONTWRITEBYTECODE="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               HF_HUB_OFFLINE="1", PYTHONNOUSERSITE="1",
               MPLCONFIGDIR="/tmp/mpl_true_fusion_audit", XDG_CACHE_HOME="/tmp/boxfusion_xdg")
    started = time.time()
    for index, scene in enumerate(scenes, 1):
        log = args.run_dir / f"{scene}.log"
        print(f"[{index}/{len(scenes)}] capture {scene}; log={log}", flush=True)
        with log.open("x") as handle:
            result = subprocess.run(
                [sys.executable, "-u", str(ROOT / "tools/capture_true_fusion_observations.py"),
                 "--scene", scene, "--run-dir", str(args.run_dir / scene),
                 "--config", str(args.config.resolve())], cwd=ROOT, env=env,
                stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            print(f"FAILED {scene} rc={result.returncode}; inspect {log}", flush=True)
            return result.returncode
        for source, expected in hashes.items():
            if digest(source) != expected:
                raise RuntimeError(f"Source changed during capture: {source}; do not combine artifacts")
        print(f"[{index}/{len(scenes)}] completed {scene}", flush=True)
    result = subprocess.run(
        [sys.executable, "-u", str(ROOT / "tools/audit_true_fusion_observations.py"),
         "--run-dir", str(args.run_dir), "--scenes", *scenes,
         "--output", str(args.run_dir / "audit.json")], cwd=ROOT, env=env)
    if result.returncode:
        return result.returncode
    with (args.run_dir / "complete.json").open("x") as handle:
        json.dump({"completed": True, "scenes": scenes, "wall_seconds": time.time() - started,
                   "capture_sources_unchanged": True}, handle, indent=2)
    print("TRUE_FUSION_AUDIT_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
