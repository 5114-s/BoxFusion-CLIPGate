#!/usr/bin/env python3
"""Convert official OnlineAnySeg instance masks to auditable 3D AABBs.

OnlineAnySeg exports a reconstructed point cloud (``final.ply``) and a
point-by-instance Boolean matrix (``ckpt_final.npz``).  It does not export
3D detection boxes.  For the paper's class-agnostic AABB visualization, this
script takes the coordinate-wise minimum and maximum of every exported
instance.  ScanNet boxes are additionally transformed by the sequence's
official ``axisAlignment`` matrix; CA-1M boxes remain in their world frame.

No confidence ranking is invented: the official export assigns score 1 to
every instance, and those scores are preserved verbatim.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData


CORNER_CODES = np.asarray(list(itertools.product((0, 1), repeat=3)), dtype=float)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read_points(path: Path) -> np.ndarray:
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    points = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(float)
    if not np.isfinite(points).all():
        raise ValueError(f"Non-finite point coordinate in {path}")
    return points


def axis_alignment(meta: Path | None) -> np.ndarray:
    if meta is None:
        return np.eye(4, dtype=float)
    line = next((item for item in meta.read_text().splitlines()
                 if item.startswith("axisAlignment")), None)
    if line is None:
        raise ValueError(f"axisAlignment is absent from {meta}")
    return np.asarray(line.split("=", 1)[1].split(), dtype=float).reshape(4, 4)


def bounds_to_corners(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return lower[:, None, :] + CORNER_CODES[None, :, :] * (upper-lower)[:, None, :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--dataset", required=True, choices=("ScanNet", "CA-1M"))
    parser.add_argument("--pointcloud", required=True, type=Path)
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scannet-meta", type=Path)
    args = parser.parse_args()

    if args.dataset == "ScanNet" and args.scannet_meta is None:
        parser.error("--scannet-meta is required for ScanNet")
    if args.dataset == "CA-1M" and args.scannet_meta is not None:
        parser.error("--scannet-meta must not be used for CA-1M")

    points = read_points(args.pointcloud)
    export = np.load(args.instances)
    masks = np.asarray(export["pred_masks"], dtype=bool)
    scores = np.asarray(export["pred_score"], dtype=float)
    if masks.ndim != 2 or masks.shape[0] != len(points):
        raise ValueError(
            f"Point/mask mismatch: {len(points)} points versus mask shape {masks.shape}"
        )
    if masks.shape[1] != len(scores):
        raise ValueError(f"Mask/score mismatch: {masks.shape[1]} versus {len(scores)}")
    counts = masks.sum(0).astype(np.int64)
    if np.any(counts == 0):
        raise ValueError("The official export contains one or more empty instances")

    lower = np.stack([points[masks[:, index]].min(0) for index in range(masks.shape[1])])
    upper = np.stack([points[masks[:, index]].max(0) for index in range(masks.shape[1])])
    raw_boxes = bounds_to_corners(lower, upper)
    transform = axis_alignment(args.scannet_meta)
    world_boxes = raw_boxes @ transform[:3, :3].T + transform[:3, 3]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        raw_boxes=raw_boxes,
        world_boxes=world_boxes,
        scores=scores,
        point_counts=counts,
        axis_alignment=transform,
    )
    audit = {
        "scene": args.scene,
        "dataset": args.dataset,
        "pointcloud": str(args.pointcloud),
        "pointcloud_sha256": digest(args.pointcloud),
        "instances": str(args.instances),
        "instances_sha256": digest(args.instances),
        "scannet_meta": str(args.scannet_meta) if args.scannet_meta else None,
        "scannet_meta_sha256": digest(args.scannet_meta) if args.scannet_meta else None,
        "points": int(len(points)),
        "instances_count": int(masks.shape[1]),
        "point_count_min_median_max": [
            int(counts.min()), float(np.median(counts)), int(counts.max())
        ],
        "score_source": "Official OnlineAnySeg ckpt_final.npz; no score invented or recalibrated",
        "box_rule": "Coordinate-wise min/max of all points in each exported final instance mask",
        "coordinate_rule": (
            "Official ScanNet axisAlignment applied after AABB construction"
            if args.dataset == "ScanNet" else
            "Identity transform; OnlineAnySeg reconstruction and CA-1M annotations share world coordinates"
        ),
        "output": str(args.output),
        "output_sha256": digest(args.output),
    }
    audit_path = args.output.with_suffix(".json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
