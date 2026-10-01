#!/usr/bin/env python3
"""Validate an exact set of immutable TR3D residual cache files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.frozen_b6_manifest import read_scene_list
from boxfusion.tr3d_residual_cache import (
    tr3d_residual_cache_path,
    validate_tr3d_residual_cache_set,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--prefix-id", default="full")
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--config-sha256", required=True)
    return parser


def validate(
    *,
    cache_root: Path,
    scene_list: Path,
    prefix_id: str,
    checkpoint_sha256: str,
    config_sha256: str,
) -> dict[str, object]:
    scenes = read_scene_list(scene_list)
    caches = validate_tr3d_residual_cache_set(
        cache_root,
        scenes,
        prefix_id=prefix_id,
        expected_checkpoint_sha256=checkpoint_sha256,
        expected_config_sha256=config_sha256,
    )
    expected = {
        tr3d_residual_cache_path(cache_root, scene, prefix_id).resolve()
        for scene in scenes
    }
    actual = {path.resolve() for path in cache_root.rglob("*.npz")}
    if actual != expected:
        raise ValueError(
            "TR3D cache artifact set mismatch; "
            f"missing={sorted(map(str, expected-actual))[:8]}, "
            f"extra={sorted(map(str, actual-expected))[:8]}"
        )
    return {
        "schema": "boxfusion.tr3d_residual_cache_validation.v1",
        "cache_root": str(cache_root.resolve()),
        "scene_count": len(scenes),
        "proposal_count": sum(
            cache.proposal_count for cache in caches.values()
        ),
        "runtime_s": sum(cache.runtime_s for cache in caches.values()),
        "prefix_id": prefix_id,
        "checkpoint_sha256": checkpoint_sha256.lower(),
        "config_sha256": config_sha256.lower(),
        "ok": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate(
        cache_root=args.cache_root,
        scene_list=args.scene_list,
        prefix_id=args.prefix_id,
        checkpoint_sha256=args.checkpoint_sha256,
        config_sha256=args.config_sha256,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
