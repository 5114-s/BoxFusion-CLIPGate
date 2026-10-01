#!/usr/bin/env python3
"""Build an immutable generic prediction anchor manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.frozen_anchor_manifest import (  # noqa: E402
    build_frozen_anchor_manifest,
    write_frozen_anchor_manifest,
)


def _named_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"artifact must be NAME=PATH: {value!r}")
        name, raw_path = value.split("=", 1)
        if name in result:
            raise ValueError(f"duplicate artifact name: {name}")
        result[name] = Path(raw_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-name", required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--ap15", type=float, required=True)
    parser.add_argument("--ap25", type=float, required=True)
    parser.add_argument("--ap50", type=float, required=True)
    parser.add_argument("--metadata-json", default="{}")
    parser.add_argument("--required-scene-count", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = json.loads(args.metadata_json)
    payload = build_frozen_anchor_manifest(
        anchor_name=args.anchor_name,
        reference_root=args.reference_root,
        scene_list=args.scene_list,
        artifacts=_named_paths(args.artifact),
        anchor_metrics_percent={
            "AP15": args.ap15,
            "AP25": args.ap25,
            "AP50": args.ap50,
        },
        metadata=metadata,
        required_scene_count=args.required_scene_count,
    )
    status = write_frozen_anchor_manifest(args.output, payload)
    print(json.dumps({
        "status": status,
        "manifest": str(args.output.resolve()),
        "anchor_name": payload["anchor_name"],
        "scene_count": payload["scene_count"],
        "prediction_tree_sha256": payload["prediction_tree_sha256"],
        "artifact_tree_sha256": payload["artifact_tree_sha256"],
        "anchor_metrics_percent": payload["anchor_metrics_percent"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
