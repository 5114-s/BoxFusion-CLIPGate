#!/usr/bin/env python3
"""Materialize frozen target-first SAM2 births with the unchanged R15 policy.

This is a thin, fail-closed adapter over the MobileSAM R15 materializer.  The
two target-first providers intentionally share the complete receipt fields and
active policy; only the authenticated top-level schema and artifact names
differ.  The program has no GT, annotation, or evaluator input surface.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))

from tools.materialize_scannet_target_first_mobilesam_birth_full100 import (
    materialize_scannet_target_first_mobilesam_birth_full100,
)


SAM2_MASKLIFT_SCHEMA = (
    "boxfusion.scannet_target_first_sam2_masklift_paper100.v1"
)
SAM2_MASKLIFT_MANIFEST_NAME = "TARGET_FIRST_SAM2_MASKLIFT_PAPER100.json"
SCHEMA = "boxfusion.scannet_target_first_sam2_birth_paper100.v1"
MANIFEST_NAME = "TARGET_FIRST_SAM2_BIRTH_PAPER100.json"
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT / "results/scannet_target_first_sam2_birth_r15_score05"
)


def materialize_scannet_target_first_sam2_birth_paper100(
    *,
    scene_list: Path,
    baseline_root: Path,
    masklift_sidecar: Path,
    output_root: Path,
    expected_scene_count: int = 100,
    plan_only: bool = False,
) -> dict[str, object]:
    """Apply the shared fixed R15 policy to the exact SAM2 shadow schema."""

    return materialize_scannet_target_first_mobilesam_birth_full100(
        scene_list=scene_list,
        baseline_root=baseline_root,
        masklift_sidecar=masklift_sidecar,
        output_root=output_root,
        expected_scene_count=expected_scene_count,
        plan_only=plan_only,
        exact_sidecar_schema=SAM2_MASKLIFT_SCHEMA,
        sidecar_directory_manifest_name=SAM2_MASKLIFT_MANIFEST_NAME,
        output_schema=SCHEMA,
        output_manifest_name=MANIFEST_NAME,
        materializer_adapter_source=Path(__file__),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize frozen target-first SAM2 R15 births"
    )
    parser.add_argument(
        "--scene-list",
        type=Path,
        default=REPOSITORY_ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=REPOSITORY_ROOT / "results/scannet_t05_boxer_replay_active_score05",
    )
    parser.add_argument("--masklift-sidecar", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expected-scene-count", type=int, default=100)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="validate and select all scenes without creating an output root",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = materialize_scannet_target_first_sam2_birth_paper100(
        scene_list=args.scene_list,
        baseline_root=args.baseline_root,
        masklift_sidecar=args.masklift_sidecar,
        output_root=args.output_root,
        expected_scene_count=args.expected_scene_count,
        plan_only=args.plan_only,
    )
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "mode": manifest["mode"],
                "scene_count": manifest["scene_count"],
                "native_count": manifest["native_count"],
                "masklift_receipt_count": manifest["masklift_receipt_count"],
                "birth_count": manifest["birth_count"],
                "output_root": (
                    None if args.plan_only else os.fspath(args.output_root.resolve())
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
