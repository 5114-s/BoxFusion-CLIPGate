#!/usr/bin/env bash
set -euo pipefail

# Pure B5-v2 ablation: learned object-local BoxRefiner on K=5 Mask-RGBD
# memory. Detector scores/count/order stay unchanged; hand-written refit,
# B6 quality scoring, supplemental output, and Soft-NMS are disabled.
#
# Usage:
#   bash scripts/run_scannet_b5v2_refiner.sh 0,1

GPU_SPEC="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKPOINT="${BOXFUSION_B5V2_REFINER_CHECKPOINT:-${BOXFUSION_REFINER_CHECKPOINT:-$ROOT/models/scannet_b5v2_oriented_refiner.pt}}"
RUN_TAG="${BOXFUSION_B5V2_RUN_TAG:-b5v2_refiner_only_extent040_ablation10}"
SCENE_LIST="${BOXFUSION_B5V2_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"

if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
    echo "Missing trained B5-v2 BoxRefiner checkpoint: $CHECKPOINT" >&2
    echo "Run: bash scripts/train_scannet_b5v2_refiner.sh" >&2
    exit 1
fi

unset BOXFUSION_QUALITY_CHECKPOINT
unset BOXFUSION_QUALITY_MODE
unset BOXFUSION_QUALITY_DETECTOR_BLEND
unset BOXFUSION_DISABLE_ONLINE_REFINEMENT

export BOXFUSION_REFINER_CHECKPOINT="$CHECKPOINT"
export BOXFUSION_ONLINE_CONFIG="$ROOT/config/scannet_online_refinement.yaml"
export BOXFUSION_SCENE_LIST="$SCENE_LIST"
export BOXFUSION_ONLINE_ABLATION_PROFILE="b5v2_refiner_only"
export BOXFUSION_SCANNET_MIN_EXTENT="${BOXFUSION_B5V2_MIN_EXTENT:-0.40}"
export BOXFUSION_PROPOSAL_INTERVAL="${BOXFUSION_B5V2_PROPOSAL_INTERVAL:-5}"
export BOXFUSION_CANDIDATE_TTL_CLOCK="provider_call"
export BOXFUSION_CANDIDATE_TRACK_TTL="3"
export BOXFUSION_ARCHIVE_CONFIRMED_TRACKS="0"
export BOXFUSION_INFERENCE_SEED="0"
export BOXFUSION_EVAL_SEED="0"

export BOXFUSION_ONLINE_PRED_ROOT="${BOXFUSION_B5V2_PRED_ROOT:-$ROOT/results/$RUN_TAG}"
export BOXFUSION_ONLINE_LOG_ROOT="${BOXFUSION_B5V2_LOG_ROOT:-$ROOT/logs/$RUN_TAG}"
export BOXFUSION_DIAGNOSTICS_ROOT="${BOXFUSION_B5V2_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/$RUN_TAG}"
export BOXFUSION_EVAL_ROOT="${BOXFUSION_B5V2_EVAL_ROOT:-$ROOT/evaluation/$RUN_TAG}"

echo "B5-v2 refiner-only: tag=$RUN_TAG, scenes=$SCENE_LIST, GPUs=$GPU_SPEC"
exec bash "$ROOT/scripts/run_scannet_online_refinement.sh" "$GPU_SPEC"
