#!/usr/bin/env python3
"""Build genuine one-class TR3D and strictly load its initialization."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "config/tr3d/tr3d_scannet_foreground.py",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root
        / "models/tr3d_1xb16_scannet-3d-foreground-init.pth",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    vendor = root / "third_party/mmdetection3d"

    import sys

    for path in (root, vendor):
        value = os.fspath(path)
        if value not in sys.path:
            sys.path.insert(0, value)

    import torch
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner.checkpoint import load_checkpoint
    from mmengine.utils import import_modules_from_strings
    from mmdet3d.registry import MODELS

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    cfg = Config.fromfile(os.fspath(args.config.resolve()))
    init_default_scope(cfg.get("default_scope", "mmdet3d"))
    import_modules_from_strings(**cfg.custom_imports)
    model = MODELS.build(cfg.model)
    # strict=True is the contract under test.  MMDetection3D's versioned
    # loader deterministically migrates the official old ``head.`` prefix to
    # the current ``bbox_head.`` prefix before checking all keys.
    load_checkpoint(
        model,
        os.fspath(args.checkpoint.resolve()),
        map_location="cpu",
        strict=True,
        logger=None,
    )
    model.to(args.device)
    state = model.state_dict()
    kernel = state["bbox_head.conv_cls.kernel"]
    bias = state["bbox_head.conv_cls.bias"]
    if tuple(kernel.shape) != (128, 1) or tuple(bias.shape) != (1, 1):
        raise AssertionError(
            f"wrong loaded classifier shapes: {kernel.shape}, {bias.shape}"
        )
    report = {
        "schema": "boxfusion.tr3d_foreground_strict_load_smoke.v1",
        "ok": True,
        "device": args.device,
        "config": os.fspath(args.config.resolve()),
        "checkpoint": os.fspath(args.checkpoint.resolve()),
        "model_type": type(model).__name__,
        "head_type": type(model.bbox_head).__name__,
        "label2level": list(model.bbox_head.label2level),
        "classifier_kernel_shape": list(kernel.shape),
        "classifier_bias_shape": list(bias.shape),
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "note": "strict model loading only; no inference or accuracy claim",
    }
    if report["head_type"] != "TR3DClassAgnosticHead":
        raise AssertionError("config did not build TR3DClassAgnosticHead")
    if report["label2level"] != [0]:
        raise AssertionError("loaded head is not genuinely one-class")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
