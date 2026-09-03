#!/usr/bin/env python3
"""Create or verify the immutable frozen-B6 content manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.frozen_b6_manifest import (
    build_frozen_b6_manifest,
    write_frozen_b6_manifest,
)


DEFAULT_B6_ROOT = Path(
    "/data/ZhaoX/OVM3D-Dett/boxfusion_b6_dev/results/"
    "b6_iou_mlp_blend040_extent040_full100"
)
DEFAULT_B6_CHECKPOINT = Path(
    "/data/ZhaoX/OVM3D-Dett/boxfusion_b6_dev/models/"
    "scannet_b6_iou_mlp.npz"
)
DEFAULT_VAL_LIST = Path(
    "/data/ZhaoX/OVM3D-Dett/boxfusion_b6_dev/evaluation/"
    "data_util/meta_data/scannetv2_val.txt"
)
DEFAULT_OUTPUT = (
    _ROOT
    / "manifests"
    / "frozen_b6_full100.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_B6_ROOT)
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_B6_CHECKPOINT
    )
    parser.add_argument("--scene-list", type=Path, default=DEFAULT_VAL_LIST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--required-scene-count", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_frozen_b6_manifest(
        reference_root=args.reference_root,
        checkpoint=args.checkpoint,
        scene_list=args.scene_list,
        required_scene_count=args.required_scene_count,
    )
    status = write_frozen_b6_manifest(args.output, payload)
    print(
        json.dumps(
            {
                "status": status,
                "manifest": str(args.output.resolve()),
                "scene_count": payload["scene_count"],
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "scene_list_sha256": payload["scene_list_sha256"],
                "prediction_tree_sha256": payload[
                    "prediction_tree_sha256"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
