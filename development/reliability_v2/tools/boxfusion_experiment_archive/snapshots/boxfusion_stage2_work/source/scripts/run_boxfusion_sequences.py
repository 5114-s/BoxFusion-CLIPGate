#!/usr/bin/env python3
"""Run BoxFusion demo.py over a list of sequences with resume-friendly logs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def read_sequences(path: Path, dataset: str) -> list[str]:
    if dataset.lower() == "ca1m":
        seqs = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            stem = Path(line).stem
            seqs.append(stem.replace("ca1m-val-", ""))
        return seqs
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def resolve_sequence_datadir(config_datadir: str, dataset: str, seq: str) -> Path:
    if dataset.lower() == "ca1m":
        if "example" in config_datadir:
            return Path(config_datadir)
        return Path(config_datadir).parent.parent / seq
    return Path(config_datadir).parent.parent / seq / "frames"


def missing_reason(datadir: Path, dataset: str) -> str | None:
    if dataset.lower() == "ca1m":
        required_files = [
            datadir / "K_depth.txt",
            datadir / "K_rgb.txt",
            datadir / "all_poses.npy",
            datadir / "after_filter_boxes.npy",
        ]
        required_dirs = [datadir / "rgb", datadir / "depth"]
    else:
        required_files = [datadir / "intrinsic" / "intrinsic_depth.txt"]
        required_dirs = [datadir / "color", datadir / "depth", datadir / "pose"]

    missing = [str(p) for p in required_files if not p.exists()]
    missing.extend(str(p) for p in required_dirs if not p.is_dir())
    if missing:
        return "missing " + ", ".join(missing[:4])
    if dataset.lower() == "ca1m":
        rgb_count = len(list((datadir / "rgb").glob("*.png")))
        depth_count = len(list((datadir / "depth").glob("*.png")))
        try:
            import numpy as np

            expected = int(np.load(datadir / "all_poses.npy", mmap_mode="r").shape[0])
        except Exception:
            expected = None
        if rgb_count == 0 or depth_count == 0:
            return f"incomplete frames rgb={rgb_count} depth={depth_count}"
        if rgb_count != depth_count:
            return f"incomplete frames rgb={rgb_count} depth={depth_count}"
        if expected is not None and rgb_count < expected:
            return f"incomplete frames rgb={rgb_count} depth={depth_count} expected={expected}"
    else:
        color_count = len(list((datadir / "color").glob("*.jpg")))
        depth_count = len(list((datadir / "depth").glob("*.png")))
        pose_count = len(list((datadir / "pose").glob("*.txt")))
        if color_count == 0 or depth_count == 0 or pose_count == 0:
            return f"incomplete frames color={color_count} depth={depth_count} pose={pose_count}"
        if color_count != depth_count or color_count != pose_count:
            return f"incomplete frames color={color_count} depth={depth_count} pose={pose_count}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["CA1M", "scannet"], required=True)
    parser.add_argument("--seq-list", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", default="./models/cutr_rgbd.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    if args.num_shards < 1:
        parser.error("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        parser.error("--shard-index must satisfy 0 <= shard-index < num-shards")

    with open(args.config, "r") as f:
        cfg = yaml.full_load(f)
    out_dir = Path(cfg["data"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    all_seqs = read_sequences(Path(args.seq_list), args.dataset)
    seqs = [seq for i, seq in enumerate(all_seqs) if i % args.num_shards == args.shard_index]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("MPLCONFIGDIR", "/tmp/boxfusion_mpl")
    env.setdefault("XDG_CACHE_HOME", "/tmp/boxfusion_xdg")

    failures: list[str] = []
    skipped_missing: list[str] = []
    for idx, seq in enumerate(seqs, start=1):
        pred = out_dir / f"{seq}_boxes.pkl"
        if pred.exists():
            print(f"[{idx}/{len(seqs)}] skip {seq}: {pred} exists", flush=True)
            continue

        if args.skip_missing:
            datadir = resolve_sequence_datadir(cfg["data"]["datadir"], args.dataset, seq)
            reason = missing_reason(datadir, args.dataset)
            if reason is not None:
                skipped_missing.append(seq)
                print(f"[{idx}/{len(seqs)}] skip {seq}: {reason}", flush=True)
                continue

        log_path = log_dir / f"{seq}.log"
        cmd = [
            sys.executable,
            "demo.py",
            args.dataset,
            "--model-path",
            args.model_path,
            "--config",
            args.config,
            "--device",
            args.device,
            "--seq",
            seq,
        ]
        print(f"[{idx}/{len(seqs)}] run {seq}; log={log_path}", flush=True)
        with open(log_path, "w") as log:
            proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            failures.append(seq)
            print(f"[{idx}/{len(seqs)}] FAILED {seq} rc={proc.returncode}", flush=True)
        else:
            print(f"[{idx}/{len(seqs)}] done {seq}", flush=True)

    if failures:
        print("failed sequences:", " ".join(failures), flush=True)
        return 1
    if skipped_missing:
        print("skipped missing sequences:", " ".join(skipped_missing), flush=True)
    print("all requested sequences finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
