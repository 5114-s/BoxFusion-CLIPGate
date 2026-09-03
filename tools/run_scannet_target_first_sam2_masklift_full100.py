#!/usr/bin/env python3
"""Frozen SAM2 target-first mask-lift shadow for ScanNet paper100.

This is a deliberately thin specialization of the sealed MobileSAM
target-first runner.  Candidate selection, RGB-D lifting, the past-only
tracker, routing gates, and sidecar layout remain in the tested base runner;
only the current-frame box-prompt mask engine and its frozen asset identity
are replaced by SAM2.1 Hiera-L.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPOSITORY_ROOT))
TOOLS_ROOT = REPOSITORY_ROOT / "tools"
if os.fspath(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(TOOLS_ROOT))

from boxfusion.sam2_boxprompt_provider import (  # noqa: E402
    PRODUCTION_CONFIG,
    FrozenSAM2BoxPromptProvider,
)
import run_scannet_target_first_mobilesam_masklift_full100 as base  # noqa: E402


SCHEMA = "boxfusion.scannet_target_first_sam2_masklift_paper100.v1"
OUTPUT_JSON = "TARGET_FIRST_SAM2_MASKLIFT_PAPER100.json"
OUTPUT_NPZ = "TARGET_FIRST_SAM2_MASKLIFT_PAPER100.npz"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "logs/scannet_target_first_sam2_masklift_paper100_score05"


class SAM2TargetFirstEngine:
    """Adapt the frozen SAM2 provider to the tested target-first engine API."""

    def __init__(
        self,
        device: str,
        *,
        provider_factory: Callable[..., FrozenSAM2BoxPromptProvider] = FrozenSAM2BoxPromptProvider,
    ) -> None:
        config = replace(PRODUCTION_CONFIG, device=device)
        self._provider = provider_factory(config=config)
        self._config = config

    def predict(
        self, image_rgb: np.ndarray, boxes_xyxy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
        # The legacy OWL mapper permits the half-open upper bounds 640/480,
        # whereas the frozen SAM2 provider consumes inclusive pixel
        # coordinates.  Convert only those upper bounds; source order and all
        # lower coordinates remain unchanged.
        prompts = np.asarray(boxes_xyxy, dtype=np.float32).copy()
        if prompts.ndim != 2 or prompts.shape[1:] != (4,):
            raise ValueError("boxes_xyxy must have shape [N,4]")
        prompts[:, 2] = np.minimum(prompts[:, 2], 639.0)
        prompts[:, 3] = np.minimum(prompts[:, 3], 479.0)
        if len(prompts) and np.any(
            (prompts[:, 2] <= prompts[:, 0]) | (prompts[:, 3] <= prompts[:, 1])
        ):
            raise ValueError("SAM2 inclusive prompt conversion produced an empty box")
        result = self._provider.predict(image_rgb, prompts)
        timing = result.timing
        return (
            result.masks,
            result.predicted_ious,
            result.selected_hypothesis_indices,
            {
                "encoder_ms": float(timing.encoder_ms),
                "decoder_and_host_mask_ms": float(timing.decoder_and_host_mask_ms),
                "provider_ms": float(timing.complete_ms),
            },
        )

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "mask_engine": "FrozenSAM2BoxPromptProvider",
            "mask_engine_schema": "boxfusion.sam2_boxprompt_provider.n0.v1",
            "sam2_source_root": os.fspath(self._config.source_root),
            "sam2_config_name": self._config.config_name,
            "sam2_checkpoint_sha256": self._config.checkpoint_sha256,
            "sam2_source_tree_sha256": self._config.source_tree_sha256,
            "sam2_device": self._config.device,
            "sam2_multimask_selection": "max_predicted_iou_lowest_index_tie",
            "sam2_state_scope": "current_frame_only_reset_after_every_forward",
        }


def run_shadow(**kwargs: Any) -> dict[str, Any]:
    """Run the unchanged target-first pipeline with the frozen SAM2 engine."""

    checkpoint = Path(kwargs.pop("checkpoint", PRODUCTION_CONFIG.checkpoint_path))
    scene_start = kwargs.pop("scene_start", 0)
    if isinstance(scene_start, bool) or not isinstance(scene_start, int) or scene_start < 0:
        raise base.TargetFirstMaskLiftError("scene-start must be a non-negative integer")
    if scene_start and kwargs.get("scene") is not None:
        raise base.TargetFirstMaskLiftError("scene-start cannot be combined with --scene")
    resolved_checkpoint = checkpoint.resolve()
    if resolved_checkpoint != PRODUCTION_CONFIG.checkpoint_path.resolve():
        raise base.TargetFirstMaskLiftError("SAM2 checkpoint path differs from frozen production asset")

    # The base runner validates the engine checkpoint through this imported
    # module constant and uses three output identity globals.  Override them
    # only for this synchronous call and restore them even on failure.  No
    # algorithm, threshold, candidate, tracker, or sidecar field is changed.
    previous = (
        base.SCHEMA,
        base.OUTPUT_JSON,
        base.OUTPUT_NPZ,
        base.s3a.MOBILESAM_CHECKPOINT,
        base._scene_order,
        base._process_scene,
        base._canonical_scene_receipts,
    )
    base.SCHEMA = SCHEMA
    base.OUTPUT_JSON = OUTPUT_JSON
    base.OUTPUT_NPZ = OUTPUT_NPZ
    # The SAM2 production asset is intentionally exposed through a stable
    # symlink, while the reused MobileSAM runner accepts only a regular file.
    # Pass the resolved identity after proving it is the exact frozen asset;
    # the SAM2 provider still performs its own byte-count and SHA-256 checks.
    base.s3a.MOBILESAM_CHECKPOINT = resolved_checkpoint
    if scene_start:
        original_scene_order = base._scene_order
        original_process_scene = base._process_scene
        original_canonical = base._canonical_scene_receipts

        def sliced_scene_order(scene_list_path, expected_scene_count, scene, max_scenes):
            full, selected = original_scene_order(
                scene_list_path, expected_scene_count, scene, None
            )
            if scene_start >= len(selected):
                raise base.TargetFirstMaskLiftError("scene-start is outside official scene list")
            selected = selected[scene_start:]
            if max_scenes is not None:
                if max_scenes < 1:
                    raise base.TargetFirstMaskLiftError("max-scenes must be positive")
                selected = selected[:max_scenes]
            return full, selected

        def globally_indexed_process_scene(**call_kwargs):
            call_kwargs["scene_index"] = int(call_kwargs["scene_index"]) + scene_start
            return original_process_scene(**call_kwargs)

        def globally_indexed_receipts(scene_order, tracks):
            local_tracks = []
            for row in tracks:
                copied = dict(row)
                copied["scene_index"] = int(copied["scene_index"]) - scene_start
                local_tracks.append(copied)
            return original_canonical(scene_order, local_tracks)

        base._scene_order = sliced_scene_order
        base._process_scene = globally_indexed_process_scene
        base._canonical_scene_receipts = globally_indexed_receipts
    try:
        return base.run_shadow(
            checkpoint=resolved_checkpoint,
            engine_factory=SAM2TargetFirstEngine,
            **kwargs,
        )
    finally:
        (
            base.SCHEMA,
            base.OUTPUT_JSON,
            base.OUTPUT_NPZ,
            base.s3a.MOBILESAM_CHECKPOINT,
            base._scene_order,
            base._process_scene,
            base._canonical_scene_receipts,
        ) = previous


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-manifest", type=Path, default=REPOSITORY_ROOT / "results/scannet_cbest_raw_boxer_past3_birth_v2_m50_score05/RAW_BOXER_PAST3_BIRTH_FULL100.json")
    parser.add_argument("--raw-log-root", type=Path, default=REPOSITORY_ROOT / "logs/scannet_raw_boxer_full100_score05_v1")
    parser.add_argument("--schedule-root", type=Path, default=Path("/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/cache/cutr_proposals/scannet-score05-gap25-postfilter-v2"))
    parser.add_argument("--scene-root", type=Path, default=REPOSITORY_ROOT / "upstream_clean/scannet_readme_frames")
    parser.add_argument("--scene-list", type=Path, default=REPOSITORY_ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt")
    parser.add_argument("--baseline-root", type=Path, default=REPOSITORY_ROOT / "results/scannet_t05_boxer_replay_active_score05")
    parser.add_argument("--checkpoint", type=Path, default=PRODUCTION_CONFIG.checkpoint_path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-scene-count", type=int, default=100)
    parser.add_argument("--scene")
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--scene-start", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    run_shadow(
        receipt_manifest_path=args.receipt_manifest,
        raw_log_root=args.raw_log_root,
        schedule_root=args.schedule_root,
        scene_root=args.scene_root,
        scene_list_path=args.scene_list,
        baseline_root=args.baseline_root,
        checkpoint=args.checkpoint,
        output_root=args.output_root,
        device=args.device,
        expected_scene_count=args.expected_scene_count,
        scene=args.scene,
        max_scenes=args.max_scenes,
        scene_start=args.scene_start,
        plan_only=args.plan_only,
    )


if __name__ == "__main__":
    main()
