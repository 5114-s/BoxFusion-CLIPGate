#!/usr/bin/env python3
"""Audit split-level AP50 potential before training a sparse refiner."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from tools.train_sgcdet_sparse_refiner import (
    deterministic_scene_holdout,
    load_sgcdet_sparse_refiner_dataset,
)


def split_metrics(data: Any, indices: np.ndarray) -> dict[str, Any]:
    eligible = data.runtime_eligible[indices] & (
        data.matched_gt_index[indices] >= 0
    )
    geometry = data.geometry_mask[indices]
    cross = data.cross_iou50[indices]
    preserve = eligible & data.identity_tp50[indices]
    oracle_drop = (
        eligible
        & data.identity_tp50[indices]
        & ~data.candidate_oracle_tp50[indices]
    )
    oracle_net = int(np.count_nonzero(cross)) - int(
        np.count_nonzero(oracle_drop)
    )
    scenes = data.scene_ids[indices]
    cross_scenes = sorted(set(scenes[cross].tolist()))
    eligible_count = int(np.count_nonzero(eligible))
    return {
        "samples": int(len(indices)),
        "scenes": int(len(set(scenes.tolist()))),
        "eligible": eligible_count,
        "geometry_positive": int(np.count_nonzero(geometry)),
        "geometry_negative": int(np.count_nonzero(~geometry)),
        "cross_iou50": int(np.count_nonzero(cross)),
        "cross_iou50_scenes": len(cross_scenes),
        "cross_iou50_scene_ids": cross_scenes,
        "preserve_iou50": int(np.count_nonzero(preserve)),
        "oracle_drop_iou50": int(np.count_nonzero(oracle_drop)),
        "oracle_net_iou50": oracle_net,
        "oracle_net_rate": (
            float(oracle_net) / float(eligible_count)
            if eligible_count
            else 0.0
        ),
    }


def check_split(
    name: str,
    metrics: dict[str, Any],
    *,
    min_eligible: int,
    min_geometry_positive: int,
    min_geometry_negative: int,
    min_cross: int,
    min_cross_scenes: int,
    min_preserve: int,
    min_oracle_net: int,
    min_oracle_net_rate: float,
) -> list[str]:
    requirements = {
        "eligible": min_eligible,
        "geometry_positive": min_geometry_positive,
        "geometry_negative": min_geometry_negative,
        "cross_iou50": min_cross,
        "cross_iou50_scenes": min_cross_scenes,
        "preserve_iou50": min_preserve,
        "oracle_net_iou50": min_oracle_net,
        "oracle_net_rate": min_oracle_net_rate,
    }
    return [
        f"{name}.{key}={metrics[key]} < {minimum}"
        for key, minimum in requirements.items()
        if metrics[key] < minimum
    ]


def audit(args: argparse.Namespace) -> dict[str, Any]:
    data = load_sgcdet_sparse_refiner_dataset(args.input)
    train_indices, validation_indices = deterministic_scene_holdout(
        data.scene_ids, args.validation_fraction, args.seed
    )
    train = split_metrics(data, train_indices)
    validation = split_metrics(data, validation_indices)
    issues = check_split(
        "train",
        train,
        min_eligible=args.train_min_eligible,
        min_geometry_positive=args.train_min_geometry_positive,
        min_geometry_negative=args.train_min_geometry_negative,
        min_cross=args.train_min_cross,
        min_cross_scenes=args.train_min_cross_scenes,
        min_preserve=args.train_min_preserve,
        min_oracle_net=args.train_min_oracle_net,
        min_oracle_net_rate=args.min_oracle_net_rate,
    )
    issues += check_split(
        "validation",
        validation,
        min_eligible=args.validation_min_eligible,
        min_geometry_positive=args.validation_min_geometry_positive,
        min_geometry_negative=args.validation_min_geometry_negative,
        min_cross=args.validation_min_cross,
        min_cross_scenes=args.validation_min_cross_scenes,
        min_preserve=args.validation_min_preserve,
        min_oracle_net=args.validation_min_oracle_net,
        min_oracle_net_rate=args.min_oracle_net_rate,
    )
    report = {
        "schema": "boxfusion.sgcdet_training_potential.v1",
        "dataset": str(args.input.resolve()),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "train": train,
        "validation": validation,
        "issues": issues,
        "ok": not issues,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, args.output)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train-min-eligible", type=int, default=200)
    parser.add_argument("--train-min-geometry-positive", type=int, default=100)
    parser.add_argument("--train-min-geometry-negative", type=int, default=100)
    parser.add_argument("--train-min-cross", type=int, default=20)
    parser.add_argument("--train-min-cross-scenes", type=int, default=10)
    parser.add_argument("--train-min-preserve", type=int, default=50)
    parser.add_argument("--train-min-oracle-net", type=int, default=1)
    parser.add_argument("--validation-min-eligible", type=int, default=50)
    parser.add_argument("--validation-min-geometry-positive", type=int, default=20)
    parser.add_argument("--validation-min-geometry-negative", type=int, default=20)
    parser.add_argument("--validation-min-cross", type=int, default=5)
    parser.add_argument("--validation-min-cross-scenes", type=int, default=3)
    parser.add_argument("--validation-min-preserve", type=int, default=10)
    parser.add_argument("--validation-min-oracle-net", type=int, default=5)
    parser.add_argument("--min-oracle-net-rate", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0,1)")
    if not 0.0 <= args.min_oracle_net_rate <= 1.0:
        raise ValueError("min_oracle_net_rate must lie in [0,1]")
    report = audit(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
