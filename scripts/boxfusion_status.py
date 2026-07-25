#!/usr/bin/env python3
"""Print local BoxFusion reproduction progress."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCANNET_ROOT = Path("/extra/ZhaoX/scannet_data/scans")
CA1M_ROOT = Path("/extra/ZhaoX/boxfusion_ca1m")


def count_files(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern))) if path.is_dir() else 0


def scannet_status() -> None:
    val_path = ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt"
    seqs = [line.strip() for line in val_path.read_text().splitlines() if line.strip()]
    done: list[str] = []
    partial: list[tuple[str, int, int, int, bool]] = []
    for seq in seqs:
        d = SCANNET_ROOT / seq
        color = count_files(d / "color", "*.jpg")
        depth = count_files(d / "depth", "*.png")
        pose = count_files(d / "pose", "*.txt")
        intr = (d / "intrinsic" / "intrinsic_depth.txt").exists()
        if color and color == depth == pose and intr:
            done.append(seq)
        elif color or depth or pose:
            partial.append((seq, color, depth, pose, intr))

    pred_dir = ROOT / "results/scannet"
    preds = sorted(p.name.replace("_boxes.pkl", "") for p in pred_dir.glob("*_boxes.pkl")) if pred_dir.exists() else []

    print(f"ScanNet frames: {len(done)}/{len(seqs)} ready")
    print(f"ScanNet predictions: {len(preds)}/{len(seqs)} ready")
    if done:
        print("  latest frame-ready:", ", ".join(done[-5:]))
    if preds:
        print("  latest predictions:", ", ".join(preds[-8:]))
    if partial:
        print("  partial:", partial[:3])


def ca1m_status() -> None:
    seqs = sorted(p for p in CA1M_ROOT.iterdir() if p.is_dir() and p.name.isdigit()) if CA1M_ROOT.exists() else []
    complete: list[str] = []
    partial: list[tuple[str, int, int, int | str]] = []
    try:
        import numpy as np
    except Exception:
        np = None

    for seq in seqs:
        rgb = count_files(seq / "rgb", "*.png")
        depth = count_files(seq / "depth", "*.png")
        expected: int | str = "?"
        if np is not None and (seq / "all_poses.npy").exists():
            try:
                expected = int(np.load(seq / "all_poses.npy", mmap_mode="r").shape[0])
            except Exception:
                expected = "?"
        required = [
            seq / "K_depth.txt",
            seq / "K_rgb.txt",
            seq / "all_poses.npy",
            seq / "after_filter_boxes.npy",
        ]
        if isinstance(expected, int) and rgb >= expected and depth >= expected and all(p.exists() for p in required):
            complete.append(seq.name)
        else:
            partial.append((seq.name, rgb, depth, expected))

    pred_dir = ROOT / "results/full"
    preds = sorted(p.name.replace("_boxes.pkl", "") for p in pred_dir.glob("*_boxes.pkl")) if pred_dir.exists() else []

    print(f"CA-1M downloaded: {len(complete)}/107 complete ({len(seqs)} dirs present)")
    print(f"CA-1M predictions: {len(preds)}/107 ready")
    if complete:
        print("  complete:", ", ".join(complete[:8]))
    if partial:
        print("  partial:", partial[:5])


def main() -> int:
    scannet_status()
    print()
    ca1m_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
