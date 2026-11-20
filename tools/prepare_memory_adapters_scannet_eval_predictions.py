#!/usr/bin/env python3
"""Convert aligned Memory-based Adapters boxes to evaluator input coordinates.

The Online3D model consumes and predicts boxes in ScanNet's axis-aligned world
frame.  BoxFusion's frozen ScanNet evaluator expects predictions in the raw
ScanNet world frame and applies ``axisAlignment`` itself.  This utility removes
the alignment once, avoiding an accidental double transformation while keeping
the original inference outputs untouched.
"""

from __future__ import annotations

import argparse
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_scene_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def load_axis_alignment(scene_root: Path) -> np.ndarray:
    metadata = scene_root / f"{scene_root.name}.txt"
    for line in metadata.read_text(encoding="utf-8").splitlines():
        if line.startswith("axisAlignment"):
            values = line.split("=", 1)[1].split()
            return np.asarray(values, dtype=np.float64).reshape(4, 4)
    return np.eye(4, dtype=np.float64)


def transform_corners(corners: object, transform: np.ndarray) -> np.ndarray:
    array = np.asarray(corners, dtype=np.float64).reshape(8, 3)
    homogeneous = np.concatenate(
        (array, np.ones((len(array), 1), dtype=array.dtype)), axis=1
    )
    return (homogeneous @ transform.T)[:, :3]


def atomic_pickle_dump(value: object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=output_path.parent, prefix=f".{output_path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", type=Path, default=ROOT / "results/memory_adapters_scannet100"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/memory_adapters_scannet100_evalcoords",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("/extra/ZhaoX/scannet_data/scans")
    )
    parser.add_argument(
        "--scene-list",
        type=Path,
        default=ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt",
    )
    args = parser.parse_args()

    scenes = read_scene_list(args.scene_list)
    if len(scenes) != 100:
        raise RuntimeError(f"expected 100 scenes, found {len(scenes)}")

    total_boxes = 0
    for scene in scenes:
        input_path = args.input_root / f"{scene}_boxes.pkl"
        with input_path.open("rb") as handle:
            predictions = pickle.load(handle)
        if not isinstance(predictions, list) or len(predictions) != 1:
            raise ValueError(f"invalid prediction structure: {input_path}")

        raw_from_aligned = np.linalg.inv(
            load_axis_alignment(args.data_root / scene)
        )
        converted = []
        for label, corners, score in predictions[0]:
            converted.append(
                (int(label), transform_corners(corners, raw_from_aligned), float(score))
            )
        atomic_pickle_dump([converted], args.output_root / input_path.name)
        total_boxes += len(converted)

    print(
        f"converted {len(scenes)} scenes and {total_boxes} boxes to {args.output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
