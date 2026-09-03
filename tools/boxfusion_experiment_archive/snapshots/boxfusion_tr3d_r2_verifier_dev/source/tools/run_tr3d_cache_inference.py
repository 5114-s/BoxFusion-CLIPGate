#!/usr/bin/env python3
"""Run genuine one-class TR3D and export immutable observer caches."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.tr3d_inference import (  # noqa: E402
    OfficialMMDet3DTR3DAdapter,
    SyntheticTR3DAdapter,
    artifact_sha256,
    direct_inference_input,
    export_inference_inputs,
    load_inference_manifest,
    select_scenes,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-manifest", type=Path)
    source.add_argument("--point-file", type=Path)
    parser.add_argument("--scene-list", type=Path)
    parser.add_argument("--scene-id")
    parser.add_argument("--prefix-id", action="append")
    parser.add_argument("--prefix-fraction", type=float, default=1.0)
    parser.add_argument("--axis-alignment", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-threshold", type=float, default=0.01)
    parser.add_argument("--max-proposals", type=int, default=1000)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run-synthetic",
        action="store_true",
        help=(
            "Exercise point/alignment/cache conversion without importing "
            "OpenMMLab. No cache file is written."
        ),
    )
    parser.add_argument("--report", type=Path)
    return parser


def _inputs(args: argparse.Namespace):
    if args.input_manifest is not None:
        if args.scene_id is not None or args.axis_alignment is not None:
            raise ValueError(
                "--scene-id/--axis-alignment apply only to --point-file"
            )
        return load_inference_manifest(
            args.input_manifest,
            scene_ids=select_scenes(args.scene_list),
            prefix_ids=args.prefix_id,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
    if args.scene_list is not None:
        raise ValueError("--scene-list requires --input-manifest")
    if args.scene_id is None or args.axis_alignment is None:
        raise ValueError(
            "--point-file requires --scene-id and --axis-alignment"
        )
    if args.num_shards != 1 or args.shard_index != 0:
        raise ValueError("sharding is only valid with --input-manifest")
    prefix = "full" if not args.prefix_id else args.prefix_id[0]
    if args.prefix_id is not None and len(args.prefix_id) != 1:
        raise ValueError("direct point inference accepts one --prefix-id")
    return (
        direct_inference_input(
            scene_id=args.scene_id,
            prefix_id=prefix,
            prefix_fraction=args.prefix_fraction,
            point_path=args.point_file,
            axis_alignment_path=args.axis_alignment,
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    inputs = _inputs(args)
    config_sha = artifact_sha256(config)
    checkpoint_sha = artifact_sha256(checkpoint)
    if args.dry_run_synthetic:
        adapter = SyntheticTR3DAdapter()
    else:
        adapter = OfficialMMDet3DTR3DAdapter(
            config_path=config,
            checkpoint_path=checkpoint,
            device=args.device,
            project_root=_ROOT,
            vendor_root=_ROOT / "third_party" / "mmdetection3d",
        )
        if not math.isclose(
            args.voxel_size,
            adapter.voxel_size,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "--voxel-size disagrees with the loaded TR3D head: "
                f"{args.voxel_size} != {adapter.voxel_size}"
            )
    report = export_inference_inputs(
        inputs=inputs,
        adapter=adapter,
        cache_root=args.cache_root,
        checkpoint_sha256=checkpoint_sha,
        config_sha256=config_sha,
        score_threshold=args.score_threshold,
        max_proposals=args.max_proposals,
        voxel_size=args.voxel_size,
        resume=args.resume,
        write_cache=not args.dry_run_synthetic,
    )
    report.update(
        {
            "config_path": str(config),
            "checkpoint_path": str(checkpoint),
            "device": args.device,
            "dry_run_synthetic": bool(args.dry_run_synthetic),
            "cache_root": str(args.cache_root.resolve()),
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
        }
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        report_path = args.report.resolve()
        cache_root = args.cache_root.resolve()
        if report_path == cache_root or cache_root in report_path.parents:
            raise ValueError("report must not be written inside immutable cache")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
