#!/usr/bin/env python3
"""Rewrite Map-Det3D corner arrays as portable Python lists, atomically."""

from __future__ import annotations

import argparse
import os
import pickle
import tempfile
from pathlib import Path


def rewrite(path: Path) -> None:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    rows = [
        (
            int(class_id),
            corners.tolist() if hasattr(corners, "tolist") else corners,
            float(score),
        )
        for class_id, corners, score in payload[0]
    ]
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        pickle.dump([rows], handle, protocol=4)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_root", type=Path)
    args = parser.parse_args()
    paths = sorted(args.prediction_root.glob("*_boxes.pkl"))
    for path in paths:
        rewrite(path)
    print(f"rewritten={len(paths)} root={args.prediction_root}")


if __name__ == "__main__":
    main()
