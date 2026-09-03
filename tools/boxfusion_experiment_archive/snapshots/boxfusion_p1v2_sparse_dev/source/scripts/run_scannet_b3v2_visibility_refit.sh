#!/usr/bin/env bash
set -euo pipefail

# B3-v2 geometry-only ablation.  Mask-RGBD boundaries are fused in each
# upstream box's local frame, then the original box orientation is restored.
# Scores, detection count, supplemental output, BoxRefiner, and Soft-NMS stay
# unchanged.
#
# Usage:
#   bash scripts/run_scannet_b3v2_visibility_refit.sh 0,1

GPU_SPEC="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_TAG="${BOXFUSION_B3V2_RUN_TAG:-b3v2_oriented_pair_refit_extent040_ablation10}"
SCENE_LIST="${BOXFUSION_B3V2_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"

unset BOXFUSION_REFINER_CHECKPOINT
unset BOXFUSION_QUALITY_CHECKPOINT
unset BOXFUSION_QUALITY_MODE
unset BOXFUSION_QUALITY_DETECTOR_BLEND
unset BOXFUSION_DISABLE_ONLINE_REFINEMENT

export BOXFUSION_ONLINE_CONFIG="$ROOT/config/scannet_online_refinement.yaml"
export BOXFUSION_SCENE_LIST="$SCENE_LIST"
export BOXFUSION_ONLINE_ABLATION_PROFILE="b3v2_visibility_refit_only"
export BOXFUSION_SCANNET_MIN_EXTENT="${BOXFUSION_B3V2_MIN_EXTENT:-0.40}"
export BOXFUSION_PROPOSAL_INTERVAL="${BOXFUSION_B3V2_PROPOSAL_INTERVAL:-5}"
export BOXFUSION_CANDIDATE_TTL_CLOCK="provider_call"
export BOXFUSION_CANDIDATE_TRACK_TTL="3"
export BOXFUSION_ARCHIVE_CONFIRMED_TRACKS="0"
export BOXFUSION_INFERENCE_SEED="0"
export BOXFUSION_EVAL_SEED="0"

export BOXFUSION_ONLINE_PRED_ROOT="${BOXFUSION_B3V2_PRED_ROOT:-$ROOT/results/$RUN_TAG}"
export BOXFUSION_ONLINE_LOG_ROOT="${BOXFUSION_B3V2_LOG_ROOT:-$ROOT/logs/$RUN_TAG}"
export BOXFUSION_DIAGNOSTICS_ROOT="${BOXFUSION_B3V2_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/$RUN_TAG}"
export BOXFUSION_EVAL_ROOT="${BOXFUSION_B3V2_EVAL_ROOT:-$ROOT/evaluation/$RUN_TAG}"

echo "B3-v2 visibility refit: tag=$RUN_TAG, scenes=$SCENE_LIST, GPUs=$GPU_SPEC"
exec bash "$ROOT/scripts/run_scannet_online_refinement.sh" "$GPU_SPEC"
