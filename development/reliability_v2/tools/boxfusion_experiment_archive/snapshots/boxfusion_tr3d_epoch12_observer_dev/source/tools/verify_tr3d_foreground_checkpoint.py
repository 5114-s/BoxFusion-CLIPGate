#!/usr/bin/env python3
"""Re-audit a converted TR3D foreground initialization against its source."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from boxfusion.tr3d_foreground_checkpoint import (
    BIAS_KEY,
    KERNEL_KEY,
    SCHEMA,
    collapse_sigmoid_classes_to_foreground,
    sha256_file,
    tensor_sha256,
)


def _load(path: Path) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and list(left.keys()) == list(right.keys())
            and all(_nested_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_nested_equal(a, b) for a, b in zip(left, right))
        )
    return bool(left == right)


def verify(
    source_path: Path, output_path: Path, provenance_path: Path
) -> Mapping[str, Any]:
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("schema") != SCHEMA:
        raise ValueError("unexpected foreground checkpoint provenance schema")
    source_sha = sha256_file(source_path)
    output_sha = sha256_file(output_path)
    if source_sha != provenance["source"]["sha256"]:
        raise ValueError("source SHA256 disagrees with provenance")
    if output_sha != provenance["output"]["sha256"]:
        raise ValueError("output SHA256 disagrees with provenance")

    source = _load(source_path)
    output = _load(output_path)
    source_state = source["state_dict"]
    output_state = output["state_dict"]
    if list(source_state.keys()) != list(output_state.keys()):
        raise ValueError("state_dict keys or order changed")
    for key in source_state:
        if key in (KERNEL_KEY, BIAS_KEY):
            continue
        if not _nested_equal(source_state[key], output_state[key]):
            raise ValueError(f"untouched tensor changed: {key}")
    if getattr(source_state, "_metadata", None) != getattr(
        output_state, "_metadata", None
    ):
        raise ValueError("state_dict version metadata changed")

    expected_kernel, expected_bias, _ = (
        collapse_sigmoid_classes_to_foreground(
            source_state[KERNEL_KEY], source_state[BIAS_KEY]
        )
    )
    if not torch.equal(output_state[KERNEL_KEY], expected_kernel):
        raise ValueError("foreground classifier kernel is not reproducible")
    if not torch.equal(output_state[BIAS_KEY], expected_bias):
        raise ValueError("foreground classifier bias is not reproducible")
    for key in source:
        if key == "state_dict":
            continue
        if key not in output or not _nested_equal(source[key], output[key]):
            raise ValueError(f"top-level checkpoint object changed: {key}")
    if list(source.keys()) != list(output.keys()):
        raise ValueError("top-level checkpoint keys or order changed")

    return {
        "schema": "boxfusion.tr3d_foreground_checkpoint_verification.v1",
        "ok": True,
        "source_sha256": source_sha,
        "output_sha256": output_sha,
        "state_dict_tensor_count": len(output_state),
        "modified_keys": [KERNEL_KEY, BIAS_KEY],
        "untouched_tensor_count": len(output_state) - 2,
        "kernel_shape": list(output_state[KERNEL_KEY].shape),
        "bias_shape": list(output_state[BIAS_KEY].shape),
        "kernel_sha256": tensor_sha256(output_state[KERNEL_KEY]),
        "bias_sha256": tensor_sha256(output_state[BIAS_KEY]),
        "meta_preserved": True,
        "optimizer_preserved": "optimizer" in source,
        "warning": "initialization only; do not resume the source optimizer",
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "models/tr3d_1xb16_scannet-3d-18class.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models/tr3d_1xb16_scannet-3d-foreground-init.pth",
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=ROOT
        / "manifests/tr3d_scannet_foreground_init_checkpoint.json",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = verify(
        args.source.resolve(),
        args.output.resolve(),
        args.provenance.resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
