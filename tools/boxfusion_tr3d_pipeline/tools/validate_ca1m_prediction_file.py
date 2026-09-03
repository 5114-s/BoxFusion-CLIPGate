#!/usr/bin/env python3
"""Validate one CA-1M prediction pickle before treating it as resumable."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    args = parser.parse_args()

    path = args.prediction.resolve()
    with path.open("rb") as handle:
        prediction = pickle.load(handle)
        if handle.read(1):
            raise ValueError(f"trailing bytes after prediction payload: {path}")
    if not isinstance(prediction, list) or len(prediction) != 1:
        raise ValueError(f"prediction must contain exactly one batch: {path}")
    for row_index, item in enumerate(prediction[0]):
        if not isinstance(item, tuple) or len(item) != 3:
            raise ValueError(f"invalid row {row_index}: {path}")
        label, corners, score = item
        corners = np.asarray(corners)
        score = float(score)
        if int(label) != 0:
            raise ValueError(f"non-class-agnostic label in row {row_index}: {path}")
        if corners.shape != (8, 3) or not np.isfinite(corners).all():
            raise ValueError(f"invalid corners in row {row_index}: {path}")
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"invalid detector score in row {row_index}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
