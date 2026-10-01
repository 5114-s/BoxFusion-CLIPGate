#!/usr/bin/env python3
"""Reject non-T1 or validation-selecting configs before TR3D training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


def _leaf_dataset(value: Mapping[str, Any]) -> Mapping[str, Any]:
    current = value
    for _ in range(8):
        nested = current.get("dataset")
        if not isinstance(nested, Mapping):
            return current
        current = nested
    raise ValueError("dataset wrapper nesting exceeds safety limit")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("full", "prefix"), required=True)
    parser.add_argument("--allow-disabled-validation", action="store_true")
    args = parser.parse_args(argv)

    from mmengine.config import Config

    cfg = Config.fromfile(os.fspath(args.config.resolve()))
    head_type = cfg.model.bbox_head.get("type")
    if head_type != "TR3DClassAgnosticHead":
        raise ValueError(
            "training requires TR3DClassAgnosticHead, "
            f"not {head_type!r}"
        )
    train = _leaf_dataset(cfg.train_dataloader.dataset)
    train_ann = str(train.get("ann_file", ""))
    if cfg.get("val_dataloader") is None:
        if not args.allow_disabled_validation:
            raise ValueError("validation is disabled without explicit permission")
        validation_ann = None
    else:
        validation = _leaf_dataset(cfg.val_dataloader.dataset)
        validation_ann = str(validation.get("ann_file", ""))
    expected = (
        "scannet_infos_prefix_train_foreground.pkl"
        if args.mode == "prefix"
        else "scannet_infos_train_foreground.pkl"
    )
    if Path(train_ann).name != expected:
        raise ValueError(
            f"{args.mode} training config uses unexpected annotation: "
            f"{train_ann!r}"
        )
    if validation_ann is not None and Path(validation_ann).name != (
        "scannet_infos_calibration_foreground.pkl"
    ):
        raise ValueError(
            "checkpoint selection must use frozen train-only calibration, "
            f"not {validation_ann!r}"
        )
    report = {
        "schema": "boxfusion.tr3d.training_config_validation.v1",
        "ok": True,
        "config": os.fspath(args.config.resolve()),
        "mode": args.mode,
        "head_type": head_type,
        "train_annotation": train_ann,
        "validation_annotation": validation_ann,
        "validation_disabled": validation_ann is None,
        "official_val_checkpoint_selection": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
